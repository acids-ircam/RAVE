import torch
import torch.nn.functional as F
import numpy as np
import gin

##

from torchaudio.transforms import Spectrogram

import torch


class ISTFT(torch.nn.Module):
    """
    Custom implementation of ISTFT since torch.istft doesn't allow custom padding (other than `center=True`) with
    windowing. This is because the NOLA (Nonzero Overlap Add) check fails at the edges.
    See issue: https://github.com/pytorch/pytorch/issues/62323
    Specifically, in the context of neural vocoding we are interested in "same" padding analogous to CNNs.
    The NOLA constraint is met as we trim padded samples anyway.

    Args:
        n_fft (int): Size of Fourier transform.
        hop_length (int): The distance between neighboring sliding window frames.
        win_length (int): The size of window frame and STFT filter.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
    """

    def __init__(self,
                 n_fft: int,
                 hop_length: int,
                 win_length: int,
                 padding: str = "same"):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        window = torch.hann_window(win_length)
        self.register_buffer("window", window)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Compute the Inverse Short Time Fourier Transform (ISTFT) of a complex spectrogram.

        Args:
            spec (Tensor): Input complex spectrogram of shape (B, N, T), where B is the batch size,
                            N is the number of frequency bins, and T is the number of time frames.

        Returns:
            Tensor: Reconstructed time-domain signal of shape (B, L), where L is the length of the output signal.
        """
        if self.padding == "center":
            # Fallback to pytorch native implementation
            return torch.istft(spec,
                               self.n_fft,
                               self.hop_length,
                               self.win_length,
                               self.window,
                               center=True)
        elif self.padding == "same":
            pad = (self.win_length - self.hop_length) // 2
        else:
            raise ValueError("Padding must be 'center' or 'same'.")

        assert spec.dim() == 3, "Expected a 3D tensor as input"
        B, N, T = spec.shape

        # Inverse FFT
        ifft = torch.fft.irfft(spec, self.n_fft, dim=1, norm="backward")
        ifft = ifft * self.window[None, :, None]

        # Overlap and Add
        output_size = (T - 1) * self.hop_length + self.win_length
        y = torch.nn.functional.fold(
            ifft,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        )[:, 0, 0, pad:-pad]

        # Window envelope
        window_sq = self.window.square().expand(1, T, -1).transpose(1, 2)
        window_envelope = torch.nn.functional.fold(
            window_sq,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        ).squeeze()[pad:-pad]

        # Normalize
        assert (window_envelope > 1e-11).all()
        y = y / window_envelope

        return y


@gin.configurable
class StreamableSTFT(torch.nn.Module):

    def __init__(self,
                 nfft=1024,
                 hop_size=256,
                 stream=False,
                 skip_features=-1,
                 normalize=True,
                 log1p=False,
                 alpha_rescale=0.3,
                 beta_rescale=0.66):
        super().__init__()
        self.nfft = nfft
        self.hop_size = hop_size
        self.stream = stream
        self.log1p = log1p
        self.skip_features = skip_features
        self.alpha_rescale = alpha_rescale
        self.beta_rescale = beta_rescale
        self.normalize = normalize
        self.nskip = 1
        self.n_fade = 4

        # if stream:
        self.register_buffer('audio_buffer',
                             torch.zeros((8, 1, nfft - hop_size)),
                             persistent=False)
        
        self.register_buffer("out_buffer",
                             torch.zeros(8, 1, self.hop_size * self.n_fade),
                             persistent=False)
        
        self.register_buffer("spec_buffer",
                             torch.zeros(8,
                                         self.nfft // 2 + 1,
                                         self.n_fade,
                                         dtype=torch.complex64),
                             persistent=False)

        self.transform = Spectrogram(n_fft=nfft,
                                     win_length=nfft,
                                     hop_length=hop_size,
                                     center=not stream,
                                     normalized=False,
                                     power=None)

        self.inverse_transform = ISTFT(n_fft=nfft,
                                       win_length=nfft,
                                       hop_length=hop_size,
                                       padding="same" if stream else "center")

    def normalize_complex(self, x):
        mag = self.beta_rescale * (torch.abs(x) + 1e-8)**self.alpha_rescale
        phase = torch.angle(x)
        unit_phase = torch.complex(torch.cos(phase), torch.sin(phase))
        return mag.to(unit_phase.dtype) * unit_phase

    def denormalize_complex(self, x):
        x = x / self.beta_rescale
        mag = x.abs()**(1.0 / self.alpha_rescale)
        phase = torch.angle(x)
        unit_phase = torch.complex(torch.cos(phase), torch.sin(phase))
        return mag.to(unit_phase.dtype) * unit_phase

    @torch.jit.export
    def forward(self, x):
        # x: B x C x T
        batch_size, n_channels = x.shape[:2]
        x = x.reshape(batch_size * n_channels, 1, x.shape[-1])

        if self.stream == True:
            if self.audio_buffer.shape[0] < x.shape[0]:
                extra = x.new_zeros((x.shape[0] - self.audio_buffer.shape[0], 1,
                                     self.nfft - self.hop_size))
                self.audio_buffer = torch.cat((self.audio_buffer, extra), dim=0)
            n = x.shape[0]
            x = torch.cat([self.audio_buffer[:n], x], dim=-1)
            self.audio_buffer[:n].copy_(x[..., -(self.nfft - self.hop_size):].detach())

        spec = self.transform(x)
        spec = spec.squeeze(1)

        spec = spec if self.stream else spec[..., :-1]

        if self.normalize:
            spec = self.normalize_complex(spec)

        if self.skip_features is not None:
            if self.skip_features > 0:
                spec = spec[:, self.skip_features:]  # Drop constant component.
            elif self.skip_features < 0:
                spec = spec[:, :self.skip_features]

        freq_bins = spec.shape[-2]
        n_frames = spec.shape[-1]
        spec = spec.reshape(batch_size, n_channels, freq_bins, n_frames)
        return torch.stack((torch.real(spec), torch.imag(spec)), -3)

    def _decode_spec(self, spec):
        real, imag = spec.unbind(-3)
        batch_size, n_channels = real.shape[:2]

        spec = torch.complex(real, imag)
        freq_bins = spec.shape[-2]
        n_frames = spec.shape[-1]
        spec = spec.reshape(batch_size * n_channels,freq_bins, n_frames)
        if self.normalize:
            spec = self.denormalize_complex(spec)

        if self.skip_features is not None:
            if self.skip_features > 0:
                spec = torch.cat(
                    (torch.zeros_like(spec)[:, :self.skip_features], spec),
                    -2)
            elif self.skip_features < 0:
                spec = torch.cat(
                    (spec,
                     torch.zeros_like(spec)[:, :abs(self.skip_features)]),
                    -2)

        return spec, batch_size, n_channels

    def _resize_stream_buffers(self, n:int, spec):
        if self.spec_buffer.shape[1] != spec.shape[-2]:
            self.spec_buffer = torch.zeros(
                n,
                spec.shape[-2],
                self.n_fade,
                dtype=spec.dtype,
                device=spec.device,
            )
        elif self.spec_buffer.shape[0] < n:
            self.spec_buffer = torch.cat((self.spec_buffer, spec.new_zeros(
                (n - self.spec_buffer.shape[0], spec.shape[-2], self.n_fade))), dim=0)
        if self.out_buffer.shape[0] < n:
            extra = torch.zeros(
                n - self.out_buffer.shape[0],
                1,
                self.hop_size * self.n_fade,
                dtype=spec.real.dtype,
                device=spec.device,
            )
            self.out_buffer = torch.cat((self.out_buffer, extra), dim=0)

    # @torch.no_grad
    def inverse(self, spec):
        spec, batch_size, n_channels = self._decode_spec(spec)

        if self.stream == False:
            spec = torch.cat((spec, torch.zeros_like(spec)[:, :, :1]), -1)
        else:
            n = spec.shape[0]
            self._resize_stream_buffers(n, spec)
            spec = torch.cat((self.spec_buffer[:n], spec), -1)
            self.spec_buffer[:n] = spec[:n, :, -self.n_fade:]

        x = self.inverse_transform(spec.squeeze(1)).unsqueeze(1)

        if self.stream:
            n = x.shape[0]  # batch size
            fade_len = self.hop_size * self.n_fade

            # Create fade window
            alpha = torch.linspace(0, 1, fade_len, device=x.device)[None,
                                                                    None, :]

            # Apply crossfade
            x[..., :fade_len] = (1 - alpha) * self.out_buffer[:n].to(
                x) + alpha * x[..., :fade_len]

            # Update output buffer
            self.out_buffer[:n] = x[..., -fade_len:].detach().clone()

            # Trim fade from output
            x = x[..., :-fade_len]
        return x.reshape(batch_size, n_channels, -1)

    def inverse_stream(self, spec):
        return self.inverse(spec)


class CausalSTFT(torch.nn.Module):
    """Exact causal STFT/ISTFT using stateful overlap-add, without crossfades.

    The transform accepts audio shaped B x C x T and returns real/imag features
    shaped B x C x 2 x F x frames. In streaming use, pass chunks whose length is
    a multiple of hop_size; inverse emits the same number of samples, delayed by
    nfft - hop_size samples.
    """

    def __init__(self,
                 nfft=1024,
                 hop_size=256,
                 stream=False,
                 skip_features=None,
                 normalize=True,
                 alpha_rescale=0.3,
                 beta_rescale=0.66):
        super().__init__()
        self.nfft = nfft
        self.hop_size = hop_size
        self.stream = stream
        self.skip_features = skip_features
        self.normalize = normalize
        self.alpha_rescale = alpha_rescale
        self.beta_rescale = beta_rescale

        self.register_buffer("window", torch.hann_window(nfft))
        self.register_buffer("audio_buffer",
                             torch.zeros(8, 1, nfft - hop_size),
                             persistent=False)
        self.register_buffer("ola_buffer",
                             torch.zeros(8, 1, nfft),
                             persistent=False)
        self.register_buffer("envelope_buffer",
                             torch.zeros(8, 1, nfft),
                             persistent=False)

    @property
    def delay(self):
        return self.nfft - self.hop_size

    def normalize_complex(self, x):
        mag = self.beta_rescale * (torch.abs(x) + 1e-8)**self.alpha_rescale
        phase = torch.angle(x)
        unit_phase = torch.complex(torch.cos(phase), torch.sin(phase))
        return mag.to(unit_phase.dtype) * unit_phase

    def denormalize_complex(self, x):
        x = x / self.beta_rescale
        mag = x.abs()**(1.0 / self.alpha_rescale)
        phase = torch.angle(x)
        unit_phase = torch.complex(torch.cos(phase), torch.sin(phase))
        return mag.to(unit_phase.dtype) * unit_phase

    def _resize_buffers(self, n:int, x):
        if self.audio_buffer.shape[0] < n:
            extra = n - self.audio_buffer.shape[0]
            self.audio_buffer = torch.cat((self.audio_buffer, torch.zeros(
                extra,
                1,
                self.nfft - self.hop_size,
                dtype=x.dtype,
                device=x.device,
            )), dim=0)
            self.ola_buffer = torch.cat((self.ola_buffer, torch.zeros(
                extra,
                1,
                self.nfft,
                dtype=x.dtype,
                device=x.device,
            )), dim=0)
            self.envelope_buffer = torch.cat((self.envelope_buffer, torch.zeros(
                extra,
                1,
                self.nfft,
                dtype=x.dtype,
                device=x.device,
            )), dim=0)

    def _trim_features(self, spec):
        if self.skip_features is not None:
            if self.skip_features > 0:
                spec = spec[:, self.skip_features:]
            elif self.skip_features < 0:
                spec = spec[:, :self.skip_features]
        return spec

    def _pad_features(self, spec):
        if self.skip_features is not None:
            if self.skip_features > 0:
                spec = torch.cat(
                    (torch.zeros_like(spec)[:, :self.skip_features], spec),
                    -2)
            elif self.skip_features < 0:
                spec = torch.cat(
                    (spec,
                     torch.zeros_like(spec)[:, :abs(self.skip_features)]),
                    -2)
        return spec

    def forward(self, x):
        batch_size, n_channels = x.shape[:2]
        if x.shape[-1] % self.hop_size:
            raise ValueError("CausalSTFT input length must be a multiple of hop_size")

        x = x.reshape(batch_size * n_channels, 1, x.shape[-1])
        if self.stream:
            self._resize_buffers(x.shape[0], x)
            self.audio_buffer.detach_()
            audio_buffer = self.audio_buffer[:x.shape[0]]
        else:
            audio_buffer = torch.zeros(
                x.shape[0],
                1,
                self.delay,
                dtype=x.dtype,
                device=x.device,
            )

        frames = []
        for chunk in x.split(self.hop_size, dim=-1):
            frame = torch.cat([audio_buffer, chunk], dim=-1)
            if self.stream:
                audio_buffer.copy_(frame[..., -self.delay:].detach())
            else:
                audio_buffer = frame[..., -self.delay:]
            frame = frame.squeeze(1) * self.window.to(frame)
            spec = torch.fft.rfft(frame, n=self.nfft, dim=-1)
            if self.normalize:
                spec = self.normalize_complex(spec)
            frames.append(self._trim_features(spec))

        spec = torch.stack(frames, -1)
        freq_bins = spec.shape[-2]
        n_frames = spec.shape[-1]
        spec = spec.reshape(batch_size, n_channels, freq_bins, n_frames)
        return torch.stack((torch.real(spec), torch.imag(spec)), -3)

    def inverse(self, spec):
        real, imag = spec.unbind(-3)
        batch_size, n_channels = real.shape[:2]

        spec = torch.complex(real, imag)
        
        freq_bins = spec.shape[-2]
        n_frames = spec.shape[-1]
        spec = spec.reshape(batch_size * n_channels, freq_bins, n_frames)
        spec = self._pad_features(spec)
        if self.normalize:
            spec = self.denormalize_complex(spec)

        n = spec.shape[0]
        if self.stream:
            self._resize_buffers(n, spec.real)
            self.ola_buffer.detach_()
            self.envelope_buffer.detach_()
            ola_buffer = self.ola_buffer[:n]
            envelope_buffer = self.envelope_buffer[:n]
        else:
            ola_buffer = torch.zeros(
                n,
                1,
                self.nfft,
                dtype=spec.real.dtype,
                device=spec.device,
            )
            envelope_buffer = torch.zeros_like(ola_buffer)
        window = self.window.to(spec.real)
        window_sq = window.square()[None, None]

        output = []
        for frame_spec in spec.unbind(-1):
            frame = torch.fft.irfft(frame_spec, n=self.nfft, dim=-1)
            frame = frame[:, None] * window[None, None]

            ola_buffer += frame
            envelope_buffer += window_sq

            denom = envelope_buffer[:, :, :self.hop_size]
            y = torch.where(
                denom > 1e-8,
                ola_buffer[:, :, :self.hop_size] / denom.clamp_min(1e-8),
                torch.zeros_like(denom),
            )
            output.append(y)

            ola_buffer[:, :, :-self.hop_size] = (
                ola_buffer[:, :, self.hop_size:].clone())
            ola_buffer[:, :, -self.hop_size:] = 0
            envelope_buffer[:, :, :-self.hop_size] = ( 
                envelope_buffer[:, :, self.hop_size:].clone())
            envelope_buffer[:, :, -self.hop_size:] = 0

        x = torch.cat(output, -1)
        return x.reshape(batch_size, n_channels, -1)
