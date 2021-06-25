import torch
from einops import rearrange


def multiscale_stft(signal, scales, overlap):
    """
    Compute a stft on several scales, with a constant overlap value.
    Parameters
    ----------
    signal: torch.Tensor
        input signal to process ( B X C X T )
    
    scales: list
        scales to use
    overlap: float
        overlap between windows ( 0 - 1 )
    """
    signal = rearrange(signal, "b c t -> (b c) t")
    stfts = []
    for s in scales:
        S = torch.stft(
            signal,
            s,
            int(s * (1 - overlap)),
            s,
            torch.hann_window(s).to(signal),
            True,
            normalized=True,
            return_complex=True,
        ).abs()
        stfts.append(S)
    return stfts
