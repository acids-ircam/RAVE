"""
Evaluate disentanglement of RAVE latent representations.

Factors : speaker (categorical), content (categorical), pitch (continuous → binned)

Metrics
-------
MIG          Mutual Information Gap  (Chen et al., 2018)
             (I(z_j*; v) - I(z_j2*; v)) / H(v)
             Ranges [0, 1]. Higher = better compactness.

JEMMIG       Joint-Entropy-normalised MIG
             (I(z_j*; v) - I(z_j2*; v)) / H(z_j*, v)
             Penalises when the best latent dim is itself uncertain w.r.t. the factor.

IRS          Interventional Robustness Score  (Suter et al., 2019)
             1 - Var_within(z_j* | v) / Var_total(z_j*)
             Ranges [0, 1]. Higher = latent dim is stable within a factor class.

Explicitness Linear-probe accuracy via 5-fold cross-validated logistic regression.
             Ranges [0, 1]. Higher = factor is linearly decodable from z.

Usage:
  python scripts/evaluate_disentanglement.py --latents latents_test.npz
"""

import argparse
import warnings

import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import KBinsDiscretizer


# ── information-theoretic helpers ────────────────────────────────────────────

def entropy(labels):
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log(p + 1e-12)))


def joint_entropy(a, b):
    n = len(a)
    pairs, counts = np.unique(np.stack([a, b], axis=1), axis=0, return_counts=True)
    p = counts / n
    return float(-np.sum(p * np.log(p + 1e-12)))


def mi_per_dim(z, labels):
    """MI between every latent dimension and a categorical label vector."""
    return mutual_info_classif(z, labels, discrete_features=False, random_state=42)


# ── metrics ──────────────────────────────────────────────────────────────────

def compute_mig(z, labels):
    mi = mi_per_dim(z, labels)
    h_v = entropy(labels)
    if h_v < 1e-8:
        return 0.0
    sorted_mi = np.sort(mi)[::-1]
    return float((sorted_mi[0] - sorted_mi[1]) / h_v)


def compute_jemmig(z, labels):
    mi = mi_per_dim(z, labels)
    j_star = int(np.argmax(mi))
    sorted_mi = np.sort(mi)[::-1]
    gap = sorted_mi[0] - sorted_mi[1] if len(sorted_mi) > 1 else sorted_mi[0]

    # discretise the best latent dim for joint entropy computation
    z_best = z[:, j_star].reshape(-1, 1)
    disc = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
    z_disc = disc.fit_transform(z_best).ravel().astype(int)

    h_jv = joint_entropy(z_disc, labels)
    if h_jv < 1e-8:
        return 0.0
    return float(gap / h_jv)


def compute_irs(z, labels):
    mi = mi_per_dim(z, labels)
    j_star = int(np.argmax(mi))
    z_best = z[:, j_star]

    total_var = np.var(z_best)
    if total_var < 1e-8:
        return 0.0

    classes = np.unique(labels)
    intra_vars = [np.var(z_best[labels == c])
                  for c in classes if (labels == c).sum() > 1]
    intra_var = float(np.mean(intra_vars)) if intra_vars else total_var
    return float(1.0 - intra_var / total_var)


def compute_explicitness(z, labels):
    n_classes = len(np.unique(labels))
    # multinomial logistic regression; increase max_iter for convergence
    clf = LogisticRegression(
        max_iter=2000,
        solver='lbfgs',
        multi_class='multinomial' if n_classes > 2 else 'auto',
        random_state=42,
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        scores = cross_val_score(clf, z, labels, cv=5, scoring='accuracy')
    return float(scores.mean())


# ── per-factor evaluation ─────────────────────────────────────────────────────

def evaluate_factor(name, z, labels):
    n_cls = len(np.unique(labels))
    n_samples = len(labels)
    print(f"  {name:<10}  {n_cls} classes, {n_samples} samples")
    return {
        'MIG':          compute_mig(z, labels),
        'JEMMIG':       compute_jemmig(z, labels),
        'IRS':          compute_irs(z, labels),
        'Explicitness': compute_explicitness(z, labels),
    }


def print_table(results):
    metrics = ['MIG', 'JEMMIG', 'IRS', 'Explicitness']
    col = 14
    header = f"{'Factor':<14}" + "".join(f"{m:>{col}}" for m in metrics)
    sep = "─" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for factor, vals in results.items():
        row = f"{factor:<14}" + "".join(f"{vals[m]:>{col}.4f}" for m in metrics)
        print(row)
    print(sep)


def print_per_dim_mi(z, labels, factor_name, top_k=5):
    """Show which latent dims carry the most info about a factor."""
    mi = mi_per_dim(z, labels)
    top_idx = np.argsort(mi)[::-1][:top_k]
    print(f"\n  Top-{top_k} dims for [{factor_name}]:")
    for rank, idx in enumerate(top_idx, 1):
        print(f"    #{rank}  dim {idx:03d}  MI = {mi[idx]:.4f}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--latents', required=True,
                        help='Path to .npz produced by extract_latents.py')
    parser.add_argument('--pitch_bins', type=int, default=20,
                        help='Number of quantile bins for pitch discretisation')
    parser.add_argument('--verbose', action='store_true',
                        help='Also print top-5 most informative dims per factor')
    args = parser.parse_args()

    data = np.load(args.latents, allow_pickle=True)
    z       = data['z']        # [N, latent_size]
    speaker = data['speaker']  # [N] int
    content = data['content']  # [N] int
    pitch   = data['pitch']    # [N] float

    print(f"Loaded  {z.shape[0]} samples  |  latent dim = {z.shape[1]}")
    print(f"Speakers: {len(np.unique(speaker))}  |  "
          f"Content IDs: {len(np.unique(content))}  |  "
          f"Pitch range: {pitch.min():.0f}–{pitch.max():.0f} Hz\n")

    pitch_disc = KBinsDiscretizer(
        n_bins=args.pitch_bins, encode='ordinal', strategy='quantile'
    ).fit_transform(pitch.reshape(-1, 1)).ravel().astype(int)

    print("Computing metrics …")
    results = {}
    results['speaker'] = evaluate_factor('speaker', z, speaker)
    results['content'] = evaluate_factor('content', z, content)
    results['pitch']   = evaluate_factor('pitch',   z, pitch_disc)

    print_table(results)

    if args.verbose:
        print("\nMost informative latent dimensions per factor:")
        print_per_dim_mi(z, speaker,    'speaker')
        print_per_dim_mi(z, content,    'content')
        print_per_dim_mi(z, pitch_disc, 'pitch')


if __name__ == '__main__':
    main()
