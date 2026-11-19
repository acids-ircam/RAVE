import math
import numpy as np
import torch.nn.functional as F
import torch, torch.nn as nn
from einops import rearrange
from sklearn.decomposition import PCA, FastICA

        
@torch.fx.wrap
def get_noise(z: torch.Tensor, full_latent_size: int):
    return torch.randn(
            z.shape[0],
            full_latent_size - z.shape[1],
            z.shape[-1],
    ).type_as(z)


class RAVEProjection(nn.Module):

    def __init__(self, name: str | None = None):
        super().__init__()
        self.name = torch.jit.Attribute(name or "", str)

    def __repr__(self): 
        return "%s(name=%s)"%(type(self).__name__, self.name.value)
    
    def forward(self, x): 
        return x

    @torch.jit.export
    def inverse(self, x):
        return x



class PCAProjection(RAVEProjection): 
    def __init__(self, name: str | None = None, seed: int = 0):
        super().__init__(name=name)
        self.seed = seed
        self.latent_mean = nn.Parameter(torch.tensor(0), requires_grad=False)
        self.latent_var = nn.Parameter(torch.tensor(0), requires_grad=False)
        self.components = nn.Parameter(torch.tensor(0), requires_grad=False)

    @staticmethod
    def from_params(latent_mean, latent_var, components, name=None):
        self = PCAProjection(name=name)
        self.latent_mean.data = latent_mean
        self.latent_var.data = latent_var
        self.components.data = components
        return self

    @staticmethod
    def from_buffer(buffer, name=None):
        self = PCAProjection(name=name)
        latent_mean, latent_var, components = torch.split(buffer, (1, 1, buffer.shape[0]-2), dim=0)
        self.latent_mean.data = latent_mean[0]
        self.latent_var.data = latent_var[0]
        self.components.data = components
        return self


    def fit(self, z):
        z = rearrange(z, "b c t -> (b t) c")
        z_mean = z.mean(0)
        z = z - z_mean
        
        pca = PCA(z.shape[-1], random_state = self.seed).fit(z)
        components = pca.components_
        components = torch.from_numpy(components)

        var = pca.explained_variance_ / np.sum(pca.explained_variance_)
        var = torch.from_numpy(np.cumsum(var))

        self.latent_mean = z_mean
        self.latent_var = var
        self.components = components
        return components, z_mean, var

    def forward(self, z: torch.Tensor, latent_size: int | None = None):
        z = z - self.latent_mean.unsqueeze(-1).to(z)
        z = F.conv1d(z, self.components.unsqueeze(-1).to(z))
        if latent_size is not None: 
            z = z[:, :latent_size]
        return z

    def inverse(self, z: torch.Tensor, latent_size: int | None = None, temperature: float | None = None):
        proj_dim = self.components.shape[0]
        if temperature is None :
            temperature = 1.
        if z.shape[1] < proj_dim:
            noise = get_noise(z, proj_dim)
            z = torch.cat([z, noise * temperature], 1)
        z = F.conv1d(z, self.components.T.unsqueeze(-1).to(z))
        z = z + self.latent_mean.unsqueeze(-1).to(z)
        return z



# ICA (not computed during training: see rave dimred)
#
# ICA does not reduce dimensionality by itself: latents are first whitened with a PCA
# truncated to a target dimension k, then ICA finds the rotation of this whitened
# subspace maximising the independence of the sources. The projection is completed
# with the remaining principal components, so that it stays a (c, c) invertible map:
#     projected dims [0, k)  : ICA sources (unit variance, sorted by contribution)
#     projected dims [k, c)  : residual principal components (sorted by variance)

def _pca_whitening(z, n_components: int | None = None, fidelity: float = .99):
    """returns (x (n, c), mean (c,), eigenvectors (c, c), eigenvalues (c,), k)."""
    x = rearrange(z, "b c t -> (b t) c").detach().cpu().double().numpy()
    mean = x.mean(0)
    pca = PCA(x.shape[-1]).fit(x)
    eigvals = np.maximum(pca.explained_variance_, 0)
    if n_components is None:
        var = np.cumsum(eigvals) / max(eigvals.sum(), 1e-12)
        n_components = int(np.searchsorted(var, fidelity) + 1)
    n_components = int(min(max(n_components, 1), x.shape[-1]))
    return x, mean, pca.components_, eigvals, n_components


def ica_relevance(z, n_components: int | None = None, fidelity: float = .99,
                  min_relative_var: float = 1e-4, skew_threshold: float = .2,
                  kurtosis_threshold: float = .3):
    """Preliminary test telling whether an ICA of latents (b, c, t), on the PCA-whitened
    subspace of dimension k (n_components, or reaching `fidelity`), is worth computing.

    ICA requires
      * enough samples;
      * a non-degenerate whitened subspace (k-th eigenvalue above min_relative_var times
        the first one);
      * non-gaussian whitened dimensions (gaussian sources cannot be separated): tested
        with |skewness| and |excess kurtosis|, whose thresholds are raised by 3 standard
        errors of their gaussian estimates.

    Returns (worth: bool, info: dict)."""
    x, mean, eigvecs, eigvals, k = _pca_whitening(z, n_components, fidelity)
    n = x.shape[0]
    info = {"n_dims": x.shape[-1], "n_components": k}
    if n < 10 * k:
        info["reason"] = f"not enough samples ({n} frames for {k} components)"
        return False, info
    if eigvals[k - 1] <= min_relative_var * eigvals[0]:
        info["reason"] = (f"degenerate whitening: component {k} has a relative variance of "
                          f"{eigvals[k - 1] / max(eigvals[0], 1e-12):.1e}")
        return False, info
    y = (x - mean) @ eigvecs[:k].T / np.sqrt(eigvals[:k])
    skew = np.abs((y ** 3).mean(0))
    kurt = np.abs((y ** 4).mean(0) - 3)
    skew_thr = skew_threshold + 3 * math.sqrt(6 / n)
    kurt_thr = kurtosis_threshold + 3 * math.sqrt(24 / n)
    n_nongaussian = int(((skew > skew_thr) | (kurt > kurt_thr)).sum())
    info.update(n_nongaussian=n_nongaussian, max_skew=float(skew.max()), max_kurtosis=float(kurt.max()))
    if n_nongaussian == 0:
        info["reason"] = "whitened dimensions are gaussian (no independent sources to separate)"
        return False, info
    info["reason"] = f"{n_nongaussian} non-gaussian dimensions out of {k}"
    return True, info


def compute_ica(z, n_components: int | None = None, fidelity: float = .99,
                seed: int = 0, max_iter: int = 1000):
    """ICA of latents (b, c, t) on their PCA-whitened subspace of dimension k
    (n_components, or the number of principal components reaching `fidelity`).

    Returns a (4, c, c) buffer [mean (repeated on rows), components, mixing, info]
    (info: number of sources k, contribution of each source) such that projected = (z - mean) @ components.T and z = projected @ mixing.T + mean
    (see ICAProjection); dims [0, k) are ICA sources, dims [k, c) residual principal
    components."""
    x, mean, eigvecs, eigvals, k = _pca_whitening(z, n_components, fidelity)
    n_dims = x.shape[-1]
    scale = np.sqrt(np.maximum(eigvals[:k], 1e-12))
    whitening = eigvecs[:k] / scale[:, None]                     # (k, c)
    y = (x - mean) @ whitening.T                                 # whitened, (n, k)
    if k > 1:
        rotation = FastICA(whiten=False, max_iter=max_iter, random_state=seed).fit(y).components_
    else:
        rotation = np.ones((1, 1))
    # rotation is orthogonal: sources = y @ rotation.T, y = sources @ rotation
    components = rotation @ whitening                            # (k, c)
    mixing = eigvecs[:k].T @ (scale[:, None] * rotation.T)       # (c, k)
    order = np.argsort(-np.linalg.norm(mixing, axis=0))          # largest contribution first
    components, mixing = components[order], mixing[:, order]
    # complete with residual principal components
    components = np.concatenate([components, eigvecs[k:]], 0)
    mixing = np.concatenate([mixing, eigvecs[k:].T], 1)
    info = np.zeros((n_dims, n_dims))
    info[0, 0] = k
    info[1, :k] = np.linalg.norm(mixing[:, :k], axis=0)   # contribution of each source
    mean = np.repeat(mean[None], n_dims, 0)
    return torch.from_numpy(np.stack([mean, components, mixing, info])).float()


# key prefixes of linear projections (LinearProjection buffers) and their type
LINEAR_PROJECTION_PREFIXES = {"ica": "ICA", "wpca": "whitened PCA", "sfa": "SFA", "pls": "PLS",
                              "cca": "CCA", "lda": "LDA"}


class LinearProjection(RAVEProjection):
    """invertible linear projection of the latent space, stored as a (4, c, c) buffer
    [mean (repeated on rows), components, mixing, info]:
        projected = (z - mean) @ components.T        z = projected @ mixing.T + mean
    info is auxiliary (see linear_projection_info): info[0, 0] number of fitted dimensions,
    info[1] score of each projected dimension. Used by ica, wpca, sfa, pls, cca, lda."""

    def __init__(self, name: str | None = None, seed: int = 0):
        super().__init__(name=name)
        self.seed = seed
        self.latent_mean = nn.Parameter(torch.tensor(0), requires_grad=False)
        self.components = nn.Parameter(torch.tensor(0), requires_grad=False)
        self.mixing = nn.Parameter(torch.tensor(0), requires_grad=False)
        # auxiliary slot (whitening matrix in older ICA buffers, info otherwise)
        self.whitening = nn.Parameter(torch.tensor(0), requires_grad=False)

    @classmethod
    def from_params(cls, latent_mean, components, mixing, whitening, name=None):
        self = cls(name=name)
        self.latent_mean.data = latent_mean
        self.components.data = components
        self.mixing.data = mixing
        self.whitening.data = whitening
        return self

    @classmethod
    def from_buffer(cls, buffer, name=None):
        """buffer: [mean (repeated on rows), components, mixing, info], or
        [components, mixing, info] (zero mean)."""
        self = cls(name=name)
        if buffer.shape[0] == 4:
            latent_mean, components, mixing, whitening = buffer.unbind(0)
            latent_mean = latent_mean[0]
        else:
            components, mixing, whitening = buffer.unbind(0)
            latent_mean = torch.zeros(components.shape[-1])
        self.latent_mean.data = latent_mean
        self.components.data = components
        self.mixing.data = mixing
        self.whitening.data = whitening
        return self

    def forward(self, z: torch.Tensor, latent_size: int | None = None):
        # projected = (z - mean) @ components.T
        z = z - self.latent_mean.unsqueeze(-1).to(z)
        z = F.conv1d(z, self.components.unsqueeze(-1).to(z))
        if latent_size is not None: 
            z = z[:, :latent_size]
        return z

    def inverse(self, z: torch.Tensor, latent_size: int | None = None, temperature: float | None = None):
        # z = projected @ mixing.T + mean ; missing dimensions are filled with noise * temperature
        proj_dim = self.components.shape[0]
        if temperature is None :
            temperature = 1.
        if z.shape[1] < proj_dim:
            noise = get_noise(z, proj_dim)
            z = torch.cat([z, noise * temperature], 1)
        z = F.conv1d(z, self.mixing.unsqueeze(-1).to(z))
        z = z + self.latent_mean.unsqueeze(-1).to(z)
        return z


class ICAProjection(LinearProjection):

    def fit(self, z, n_components: int | None = None, fidelity: float = .99):
        """fits the projection on latents (b, c, t); see compute_ica."""
        buffer = compute_ica(z, n_components=n_components, fidelity=fidelity, seed=self.seed)
        fitted = ICAProjection.from_buffer(buffer)
        for k in ("latent_mean", "components", "mixing", "whitening"):
            getattr(self, k).data = getattr(fitted, k).data
        return buffer


# --------------------------------------------------------------------------- #
#  other linear projections (wpca, sfa, pls, cca, lda)
# --------------------------------------------------------------------------- #
#
#  Each method computes k projection directions A (k, c); the projection is completed
#  with an orthonormal basis of the null space of A, ordered by decreasing variance of
#  the latents in it, such that the (c, c) map is exactly invertible and the first k
#  projected dimensions are the fitted ones. Methods needing a well-conditioned
#  covariance (sfa, cca, lda) work on the PCA-whitened subspace of dimension d
#  (n_components, or reaching fidelity), as ICA.

def _frames(z) -> np.ndarray:
    return rearrange(z, "b c t -> (b t) c").detach().cpu().double().numpy()


def _pca_basis(x):
    """mean, eigenvectors (rows, decreasing variance) and eigenvalues of frames x (n, c)."""
    mean = x.mean(0)
    xc = x - mean
    cov = xc.T @ xc / max(len(x) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    return mean, eigvecs[:, order].T, np.maximum(eigvals[order], 0)


def _n_dims(eigvals, n_components=None, fidelity=.99) -> int:
    if n_components is None:
        var = np.cumsum(eigvals) / max(eigvals.sum(), 1e-12)
        n_components = int(np.searchsorted(var, fidelity) + 1)
    return int(min(max(n_components, 1), len(eigvals)))


def _whitening(eigvecs, eigvals, d):
    """(d, c) whitening matrix of the first d principal components."""
    scale = np.sqrt(np.maximum(eigvals[:d], 1e-12 * max(eigvals[0], 1e-12)))
    return eigvecs[:d] / scale[:, None]


def _linear_buffer(x, mean, directions, scores=None, n_fitted=None, normalize=False):
    """(4, c, c) buffer of the invertible projection whose first rows are `directions`
    (k, c), completed with the null space of directions ordered by variance of x. If
    normalize, directions are scaled such that projected dimensions have unit variance."""
    n_dims = x.shape[-1]
    directions = np.atleast_2d(directions)
    if normalize:
        directions = directions / np.maximum(((x - mean) @ directions.T).std(0), 1e-12)[:, None]
    k = directions.shape[0]
    if k < n_dims:
        _, _, vt = np.linalg.svd(directions, full_matrices=True)
        null = vt[k:].T                                          # (c, c - k), orthonormal
        coords = (x - mean) @ null
        cov = coords.T @ coords / max(len(x) - 1, 1)
        ev, u = np.linalg.eigh(cov)
        complement = (null @ u[:, np.argsort(ev)[::-1]]).T      # (c - k, c)
        components = np.concatenate([directions, complement], 0)
    else:
        components = directions
    mixing = np.linalg.inv(components)
    info = np.zeros((n_dims, n_dims))
    info[0, 0] = n_fitted if n_fitted is not None else k
    if scores is not None:
        scores = np.asarray(scores, dtype=np.float64).ravel()[:n_dims]
        info[1, :len(scores)] = scores
    means = np.repeat(mean[None], n_dims, 0)
    return torch.from_numpy(np.stack([means, components, mixing, info])).float()


def linear_projection_info(buffer) -> dict | None:
    """number of fitted dimensions and scores stored in a linear projection buffer
    (None for older buffers holding a whitening matrix)."""
    if buffer.shape[0] != 4:
        return None
    info = buffer[3].double()
    k = float(info[0, 0])
    if not (info[0, 1:].abs().max() == 0 and k == int(k) and 1 <= k <= info.shape[0]):
        return None
    return {"n_fitted": int(k), "scores": info[1, :int(k)].tolist()}


def compute_whitened_pca(z, n_components: int | None = None, fidelity: float = .99):
    """PCA scaled to unit variance on every axis. n_components (or fidelity) only sets the
    number of dimensions reported as fitted; scores are cumulated explained variances."""
    x = _frames(z)
    mean, eigvecs, eigvals = _pca_basis(x)
    k = _n_dims(eigvals, n_components, fidelity)
    directions = _whitening(eigvecs, eigvals, len(eigvals))
    var = np.cumsum(eigvals) / max(eigvals.sum(), 1e-12)
    return _linear_buffer(x, mean, directions, scores=var, n_fitted=k)


def compute_sfa(z, n_components: int | None = None, fidelity: float = .99):
    """Slow Feature Analysis on the PCA-whitened subspace of dimension d (n_components, or
    reaching fidelity): projected dimensions are uncorrelated, unit-variance, and sorted
    from the slowest to the fastest (scores: variance of their frame-to-frame difference)."""
    x = _frames(z)
    mean, eigvecs, eigvals = _pca_basis(x)
    d = _n_dims(eigvals, n_components, fidelity)
    w = _whitening(eigvecs, eigvals, d)                                    # (d, c)
    zc = z.detach().cpu().double().numpy() - mean[None, :, None]
    y = np.einsum("dc,bct->bdt", w, zc)
    dy = rearrange(torch.from_numpy(np.diff(y, axis=-1)), "b d t -> (b t) d").numpy()
    slowness, r = np.linalg.eigh(dy.T @ dy / max(len(dy) - 1, 1))        # ascending: slowest first
    return _linear_buffer(x, mean, r.T @ w, scores=slowness)


def _standardize(y):
    y = y - y.mean(0)
    return y / np.maximum(y.std(0), 1e-12)


def compute_cca(z, descriptors, n_components: int | None = None, fidelity: float = .99):
    """Canonical Correlation Analysis between latents and descriptors (b, n_desc, t), on the
    PCA-whitened latent subspace reaching fidelity. Projected dimensions are the latent
    directions maximally correlated with combinations of descriptors (at most n_desc),
    sorted by canonical correlation (scores)."""
    x = _frames(z)
    yd = _standardize(_frames(descriptors))
    mean, eigvecs, eigvals = _pca_basis(x)
    d = _n_dims(eigvals, None, fidelity)
    w = _whitening(eigvecs, eigvals, d)
    xw = (x - mean) @ w.T
    _, ye, yv = _pca_basis(yd)
    yw = yd @ _whitening(ye, yv, yd.shape[1]).T
    u, corr, _ = np.linalg.svd(xw.T @ yw / max(len(x) - 1, 1), full_matrices=False)
    m = min(n_components or len(corr), len(corr))
    return _linear_buffer(x, mean, u[:, :m].T @ w, scores=corr[:m])


def compute_pls(z, descriptors, n_components: int | None = None):
    """Partial Least Squares regression of descriptors (b, n_desc, t) on latents: projected
    dimensions are latent directions of maximal covariance with the (standardized)
    descriptors (default: n_desc components), scaled to unit variance; scores: explained
    descriptor variance."""
    from sklearn.cross_decomposition import PLSRegression
    x = _frames(z)
    yd = _standardize(_frames(descriptors))
    mean = x.mean(0)
    m = int(min(n_components or yd.shape[1], x.shape[1]))
    pls = PLSRegression(n_components=m, scale=False).fit(x - mean, yd)
    t = (x - mean) @ pls.x_rotations_
    explained = [(np.outer(t[:, i], pls.y_loadings_[:, i]) ** 2).sum() / (yd ** 2).sum() for i in range(m)]
    return _linear_buffer(x, mean, pls.x_rotations_.T, scores=explained, normalize=True)


def compute_lda(z, labels, n_components: int | None = None, fidelity: float = .99):
    """Linear Discriminant Analysis of latents with one class label per chunk (b,), on the
    PCA-whitened subspace reaching fidelity: projected dimensions best separate the classes
    (at most n_classes - 1), scaled to unit variance; scores: explained between-class
    variance ratios."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    labels = np.asarray(labels)
    n_classes = len(np.unique(labels))
    if n_classes < 2:
        raise ValueError(f"LDA requires at least 2 classes, got {n_classes}")
    x = _frames(z)
    frame_labels = np.repeat(labels, z.shape[-1])
    mean, eigvecs, eigvals = _pca_basis(x)
    d = _n_dims(eigvals, None, fidelity)
    w = _whitening(eigvecs, eigvals, d)
    lda = LinearDiscriminantAnalysis(solver="svd").fit((x - mean) @ w.T, frame_labels)
    m = min(n_components or n_classes - 1, n_classes - 1, lda.scalings_.shape[1])
    return _linear_buffer(x, mean, lda.scalings_[:, :m].T @ w, scores=lda.explained_variance_ratio_[:m],
                          normalize=True)


# --------------------------------------------------------------------------- #
#  audio descriptors (per latent frame)
# --------------------------------------------------------------------------- #

DESCRIPTORS = ("loudness", "centroid", "bandwidth", "flatness", "f0")
PITCH_MODES_WITH_VOICING = ("pyin", "pesto", "praat")


def compute_f0(x: torch.Tensor, sr: int, n_frames: int, mode: str = "pyin",
               fmin: str = "C2", fmax: str = "C7") -> torch.Tensor:
    """continuous f0 of audio x (b, channels, samples) on n_frames frames, with the
    acids_dataset F0 tracker, on a log scale in semitones (12 * log2(f0 / 440) + 69: same
    unit as MIDI notes, not quantized). Unvoiced frames are NaN (voicing flag / confidence
    of pyin, pesto, praat; f0 out of [fmin, fmax[ otherwise)."""
    import librosa
    from acids_dataset.features.pitch import F0
    audio = x.float().mean(1).detach().cpu().numpy()
    hop = max(audio.shape[-1] // n_frames, 1)
    out = F0.predict(audio, mode, sr, frame_length=hop, hop_length=hop, fmin=fmin, fmax=fmax)
    if out.ndim == 3:                                     # (b, [f0, voicing], frames)
        f0, voiced = out[:, 0], out[:, 1] > .5
    else:
        f0 = out
        voiced = (f0 > librosa.note_to_hz(fmin)) & (f0 < .99 * librosa.note_to_hz(fmax))
    f0 = torch.from_numpy(np.asarray(f0, dtype=np.float64))
    voiced = torch.from_numpy(np.asarray(voiced)) & (f0 > 0)
    log_f0 = torch.where(voiced, 69 + 12 * torch.log2(f0.clamp_min(1e-6) / 440), torch.full_like(f0, float("nan")))
    if log_f0.shape[-1] >= n_frames:
        return log_f0[..., :n_frames]
    return torch.cat([log_f0, torch.full(log_f0.shape[:-1] + (n_frames - log_f0.shape[-1],), float("nan"),
                                         dtype=log_f0.dtype)], -1)


def compute_descriptors(x: torch.Tensor, sr: int, n_frames: int, pitch_mode: str = "pyin") -> torch.Tensor:
    """descriptors of audio x (b, channels, samples) on n_frames frames aligned with the
    latent frames: loudness (dB), log spectral centroid / bandwidth (log Hz), spectral
    flatness, f0 (continuous, log scale in semitones, NaN when unvoiced; see compute_f0;
    not computed, i.e. NaN, if pitch_mode is None).
    Returns (b, len(DESCRIPTORS), n_frames)."""
    x_mono = x.float().mean(1)
    hop = max(x_mono.shape[-1] // n_frames, 1)
    n_fft = 1 << max(int(math.ceil(math.log2(2 * hop))), 6)
    spec = torch.stft(x_mono, n_fft, hop_length=hop, window=torch.hann_window(n_fft, device=x.device),
                      center=True, return_complex=True).abs().pow(2)[..., :n_frames]   # (b, f, t)
    eps = 1e-10
    freqs = torch.linspace(0, sr / 2, spec.shape[1], device=x.device)[None, :, None]
    power = spec.sum(1) + eps
    loudness = 10 * torch.log10(spec.mean(1) + eps)
    centroid = (spec * freqs).sum(1) / power
    bandwidth = ((spec * (freqs - centroid[:, None]) ** 2).sum(1) / power).sqrt()
    flatness = torch.exp(torch.log(spec + eps).mean(1)) / (spec.mean(1) + eps)
    out = torch.stack([loudness, torch.log(centroid + 1), torch.log(bandwidth + 1), flatness], 1)
    if out.shape[-1] < n_frames:
        out = F.pad(out, (0, n_frames - out.shape[-1]), mode="replicate")
    if pitch_mode is None:  # f0 not computed (undefined)
        f0 = torch.full_like(out[:, 0], float("nan"))
    else:
        f0 = compute_f0(x, sr, n_frames, mode=pitch_mode).to(out)
    return torch.cat([out, f0[:, None]], 1)


# LATENT MAPPER

class Mapper(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=300, hidden_layers=3):
        super().__init__()
        layer_shapes = [input_size] + [hidden_size] * hidden_layers + [output_size]
        layers = []
        for i in range(len(layer_shapes) - 1):
            layers.append(nn.Linear(layer_shapes[i], layer_shapes[i+1]))
            if i != len(layer_shapes) - 2:
                layers.append(nn.BatchNorm1d(layer_shapes[i+1]))
                layers.append(nn.ReLU())
        self._module = nn.Sequential(*layers)

    def forward(self, x):
        return self._module(x)



class MapperProjection(RAVEProjection):
    """non-linear projection: MLP autoencoder latent (input_size) -> output_size -> latent."""

    def __init__(self, input_size: int = 128, output_size: int = 8, hidden_size: int = 300,
                 hidden_layers: int = 3, name: str | None = None):
        super().__init__(name=name)
        self.input_size = input_size
        self.output_size = output_size
        self.encoder = Mapper(input_size, output_size, hidden_size, hidden_layers)
        self.decoder = Mapper(output_size, input_size, hidden_size, hidden_layers)

    @staticmethod
    def from_file(mapper_file, name=None, input_size: int = 128, output_size: int = 8,
                  hidden_size: int = 300, hidden_layers: int = 3):
        """loads a mapper checkpoint ({'state_dict': {encoder.*, decoder.*}})."""
        self = MapperProjection(input_size, output_size, hidden_size, hidden_layers, name=name)
        state_dict = torch.load(mapper_file, map_location="cpu")['state_dict']
        encoder_state_dict, decoder_state_dict = {}, {}
        for k, v in state_dict.items():
            module, param_name = k.split('.')[0], ".".join(k.split('.')[1:])
            if module == "encoder": encoder_state_dict[param_name] = v
            if module == "decoder": decoder_state_dict[param_name] = v
        self.encoder.load_state_dict(encoder_state_dict)
        self.decoder.load_state_dict(decoder_state_dict)
        return self.eval()

    def _encode(self, z: torch.Tensor) -> torch.Tensor:
        n_batch, n_channels, n_steps = z.shape
        z = self.encoder(z.permute(0, 2, 1).reshape(-1, n_channels))
        return z.reshape(n_batch, n_steps, -1).permute(0, 2, 1)

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        n_batch, n_channels, n_steps = z.shape
        z = self.decoder(z.permute(0, 2, 1).reshape(-1, n_channels))
        return z.reshape(n_batch, n_steps, -1).permute(0, 2, 1)

    def forward(self, z: torch.Tensor, latent_size: int | None = None):
        z = self._encode(z)
        if latent_size is not None:
            if latent_size < z.shape[1]:
                z = z[:, :latent_size]
            elif latent_size > z.shape[1]:
                z = torch.cat([z, torch.zeros(z.shape[0], latent_size - z.shape[1], z.shape[2],
                                              dtype=z.dtype, device=z.device)], dim=1)
        return z

    def inverse(self, z: torch.Tensor, latent_size: int | None = None, temperature: float | None = None):
        # as other projections, missing bottleneck dims are filled with noise * temperature
        # (zero if temperature is None); extra ones are ignored
        if z.shape[1] < self.output_size:
            fill = torch.zeros(z.shape[0], self.output_size - z.shape[1], z.shape[2], dtype=z.dtype, device=z.device)
            if temperature is not None:
                fill = torch.randn_like(fill) * temperature
            z = torch.cat([z, fill], dim=1)
        elif z.shape[1] > self.output_size:
            z = z[:, :self.output_size]
        return self._decode(z)


def train_mapper(z, output_size: int, hidden_size: int = 300, hidden_layers: int = 3, steps: int = 5000,
                 batch_size: int = 1024, lr: float = 1e-3, device: str = "cpu", seed: int = 0,
                 name: str | None = None, log=None) -> "MapperProjection":
    """trains a MapperProjection (MLP autoencoder, MSE) on latents z (b, c, t).
    Returns the projection in eval mode; its relative reconstruction error is stored in
    `.relative_error`."""
    generator = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    x = rearrange(z, "b c t -> (b t) c").float()
    mapper = MapperProjection(x.shape[-1], output_size, hidden_size, hidden_layers, name=name).to(device)
    optimizer = torch.optim.Adam(mapper.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    mapper.train()
    batch_size = min(batch_size, x.shape[0])
    with torch.enable_grad():  # export runs under no_grad
        _train_mapper_loop(mapper, x, steps, batch_size, optimizer, scheduler, generator, device, log)
    mapper = mapper.eval().cpu()
    with torch.no_grad():
        idx = torch.randperm(x.shape[0], generator=generator)[:100000]
        xv = x[idx]
        err = (mapper.decoder(mapper.encoder(xv)) - xv).pow(2).mean() / xv.var(0).sum().div(x.shape[-1])
    mapper.relative_error = float(err)
    return mapper


def _train_mapper_loop(mapper, x, steps, batch_size, optimizer, scheduler, generator, device, log):
    for step in range(steps):
        idx = torch.randint(0, x.shape[0], (batch_size,), generator=generator)
        xb = x[idx].to(device)
        loss = F.mse_loss(mapper.decoder(mapper.encoder(xb)), xb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if log is not None and (step % max(steps // 10, 1) == 0 or step == steps - 1):
            log(f"mapper step {step + 1}/{steps}: mse = {loss.item():.4e}")
