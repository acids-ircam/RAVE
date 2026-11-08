from functools import partial
from typing import Callable, Optional, Sequence, Union, Any
import torch
import torch.nn as nn

try:
    import torchbend as tb
    mark_fn = tb.mark
except ModuleNotFoundError:
    @torch.jit.ignore
    def mark_fn(obj=None, *args, **kwargs):
        if obj is None: 
            return lambda x: x
        else:
            return obj

    class MarkPlaceholder:

        @staticmethod
        def mark(obj: Optional[Any] = None, name: Optional[str] = None, mode: Optional[str] = "post") -> torch.Tensor:
            if torch.jit.is_scripting() or torch.jit.is_tracing():
                if torch.jit.isinstance(obj, torch.Tensor):
                    return obj
                else: 
                    raise RuntimeError("when scripting, torchbend marking is only scriptable with tensors")
            else: 
                return mark_fn(obj, name, mode)
                


    tb = MarkPlaceholder

import cached_conv as cc
import gin
import numpy as np
from torch.nn.utils import weight_norm
from torchaudio.transforms import Spectrogram
import math
from einops.layers.torch import Rearrange


from .cachedConv2D import CachedConv2d, CachedConvTranspose2d


from .core import amp_to_impulse_response, fft_convolve, mod_sigmoid
amp_to_impulse_response = torch.fx.wrap(amp_to_impulse_response)


@gin.configurable
def normalization(module: nn.Module, mode: str = 'identity'):
    if mode == 'identity':
        return module
    elif mode == 'weight_norm':
        return weight_norm(module)
    else:
        raise Exception(f'Normalization mode {mode} not supported')


class SampleNorm(nn.Module):

    def forward(self, x):
        return x / torch.norm(x, 2, 1, keepdim=True)


class Residual(nn.Module):

    def __init__(self, module, cumulative_delay=0):
        super().__init__()
        additional_delay = module.cumulative_delay
        self.aligned = cc.AlignBranches(
            module,
            nn.Identity(),
            delays=[additional_delay, 0],
        )
        self.cumulative_delay = additional_delay + cumulative_delay

    def forward(self, x):
        x_net, x_res = self.aligned(x)
        return x_net + x_res


class ResidualLayer(nn.Module):

    def __init__(
        self,
        dim,
        kernel_size,
        dilations,
        cumulative_delay=0,
        activation: Callable[[int], nn.Module] = lambda dim: nn.LeakyReLU(.2)):
        super().__init__()
        net = []
        cd = 0
        for d in dilations:
            net.append(activation(dim))
            net.append(
                normalization(
                    cc.Conv1d(
                        dim,
                        dim,
                        kernel_size,
                        dilation=d,
                        padding=cc.get_padding(kernel_size, dilation=d),
                        cumulative_delay=cd,
                    )))
            cd = net[-1].cumulative_delay
        self.net = Residual(
            cc.CachedSequential(*net),
            cumulative_delay=cumulative_delay,
        )
        self.cumulative_delay = self.net.cumulative_delay

    def forward(self, x):
        return self.net(x)


class DilatedUnit(nn.Module):

    def __init__(
        self,
        dim: int,
        kernel_size: int,
        dilation: int,
        activation: Callable[[int], nn.Module] = lambda dim: nn.LeakyReLU(.2)
    ) -> None:
        super().__init__()
        net = [
            activation(dim),
            normalization(
                cc.Conv1d(dim,
                          dim,
                          kernel_size=kernel_size,
                          dilation=dilation,
                          padding=cc.get_padding(
                              kernel_size,
                              dilation=dilation,
                          ))),
            activation(dim),
            normalization(cc.Conv1d(dim, dim, kernel_size=1)),
        ]

        self.net = cc.CachedSequential(*net)
        self.cumulative_delay = net[1].cumulative_delay

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualBlock(nn.Module):

    def __init__(self,
                 dim,
                 kernel_size,
                 dilations_list,
                 cumulative_delay=0) -> None:
        super().__init__()
        layers = []
        cd = 0

        for dilations in dilations_list:
            layers.append(
                ResidualLayer(
                    dim,
                    kernel_size,
                    dilations,
                    cumulative_delay=cd,
                ))
            cd = layers[-1].cumulative_delay

        self.net = cc.CachedSequential(
            *layers,
            cumulative_delay=cumulative_delay,
        )
        self.cumulative_delay = self.net.cumulative_delay

    def forward(self, x):
        return self.net(x)


@gin.configurable
class ResidualStack(nn.Module):

    def __init__(self,
                 dim,
                 kernel_sizes,
                 dilations_list,
                 cumulative_delay=0) -> None:
        super().__init__()
        blocks = []
        for k in kernel_sizes:
            blocks.append(ResidualBlock(dim, k, dilations_list))
        self.net = cc.AlignBranches(*blocks, cumulative_delay=cumulative_delay)
        self.cumulative_delay = self.net.cumulative_delay

    def forward(self, x):
        x = self.net(x)
        x = torch.stack(x, 0).sum(0)
        return x


class UpsampleLayer(nn.Module):

    def __init__(
        self,
        in_dim,
        out_dim,
        ratio,
        cumulative_delay=0,
        activation: Callable[[int], nn.Module] = lambda dim: nn.LeakyReLU(.2)):
        super().__init__()
        net = [activation(in_dim)]
        if ratio > 1:
            net.append(
                normalization(
                    cc.ConvTranspose1d(in_dim,
                                       out_dim,
                                       2 * ratio,
                                       stride=ratio,
                                       padding=ratio // 2)))
        else:
            net.append(
                normalization(
                    cc.Conv1d(in_dim, out_dim, 3, padding=cc.get_padding(3))))

        self.net = cc.CachedSequential(*net)
        self.cumulative_delay = self.net.cumulative_delay + cumulative_delay * ratio

    def forward(self, x):
        return self.net(x)


@gin.configurable
class NoiseGenerator(nn.Module):

    def __init__(self, in_size, data_size, ratios, noise_bands):
        super().__init__()
        net = []
        channels = [in_size] * len(ratios) + [data_size * noise_bands]
        cum_delay = 0
        for i, r in enumerate(ratios):
            net.append(
                cc.Conv1d(
                    channels[i],
                    channels[i + 1],
                    3,
                    padding=cc.get_padding(3, r),
                    stride=r,
                    cumulative_delay=cum_delay,
                ))
            cum_delay = net[-1].cumulative_delay
            if i != len(ratios) - 1:
                net.append(nn.LeakyReLU(.2))

        self.net = cc.CachedSequential(*net)
        self.data_size = data_size
        self.cumulative_delay = self.net.cumulative_delay * int(
            np.prod(ratios))

        self.register_buffer(
            "target_size",
            torch.tensor(np.prod(ratios)).long(),
        )

    def forward(self, x):
        amp = mod_sigmoid(self.net(x) - 5)
        amp = amp.permute(0, 2, 1)
        amp = amp.reshape(amp.shape[0], amp.shape[1], self.data_size, -1)

        ir = amp_to_impulse_response(amp, self.target_size)
        noise = torch.rand_like(ir) * 2 - 1

        noise = fft_convolve(noise, ir).permute(0, 2, 1, 3)
        noise = noise.reshape(noise.shape[0], noise.shape[1], -1)
        return noise


class NoiseGeneratorV2(nn.Module):

    def __init__(
        self,
        in_size: int,
        hidden_size: int,
        data_size: int,
        ratios: int,
        noise_bands: int,
        n_channels: int = 1,
        activation: Callable[[int], nn.Module] = lambda dim: nn.LeakyReLU(.2),
    ):
        super().__init__()
        net = []
        self.n_channels = n_channels
        channels = [in_size]
        channels.extend((len(ratios) - 1) * [hidden_size])
        channels.append(data_size * noise_bands * n_channels)

        for i, r in enumerate(ratios):
            net.append(
                cc.Conv1d(
                    channels[i],
                    channels[i + 1],
                    2 * r,
                    padding=(r, 0),
                    stride=r,
                ))
            if i != len(ratios) - 1:
                net.append(activation(channels[i + 1]))

        self.net = nn.Sequential(*net)
        self.data_size = data_size

        self.register_buffer(
            "target_size",
            torch.tensor(np.prod(ratios)).long(),
        )

    def forward(self, x):
        amp = mod_sigmoid(self.net(x) - 5)
        amp = amp.permute(0, 2, 1)
        amp = amp.reshape(amp.shape[0], amp.shape[1], self.n_channels * self.data_size, -1)

        ir = amp_to_impulse_response(amp, self.target_size)
        noise = torch.rand_like(ir) * 2 - 1

        noise = fft_convolve(noise, ir).permute(0, 2, 1, 3)
        noise = noise.reshape(noise.shape[0], noise.shape[1], -1)
        return noise


class GRU(nn.Module):

    def __init__(self, latent_size: int, num_layers: int) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=latent_size,
            hidden_size=latent_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.register_buffer("gru_state", torch.tensor(0))
        self.enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled: return x
        x = x.permute(0, 2, 1)
        x = self.gru(x)[0]
        x = x.permute(0, 2, 1)
        return x

    def disable(self):
        self.enabled = False

    def enable(self):
        self.enabled = True


class Generator(nn.Module):

    def __init__(
        self,
        latent_size,
        capacity,
        data_size,
        ratios,
        loud_stride,
        use_noise,
        n_channels: int = 1,
        recurrent_layer: Optional[Callable[[], nn.Module]] = None,
    ):
        super().__init__()
        net = [
            normalization(
                cc.Conv1d(
                    latent_size,
                    2**len(ratios) * capacity,
                    7,
                    padding=cc.get_padding(7),
                ))
        ]

        if recurrent_layer is not None:
            net.append(
                recurrent_layer(
                    dim=2**len(ratios) * capacity,
                    cumulative_delay=net[0].cumulative_delay,
                ))

        for i, r in enumerate(ratios):
            in_dim = 2**(len(ratios) - i) * capacity
            out_dim = 2**(len(ratios) - i - 1) * capacity

            net.append(
                UpsampleLayer(
                    in_dim,
                    out_dim,
                    r,
                    cumulative_delay=net[-1].cumulative_delay,
                ))
            net.append(
                ResidualStack(out_dim,
                              cumulative_delay=net[-1].cumulative_delay))

        self.net = cc.CachedSequential(*net)

        wave_gen = normalization(
            cc.Conv1d(out_dim, data_size * n_channels, 7, padding=cc.get_padding(7)))

        loud_gen = normalization(
            cc.Conv1d(
                out_dim,
                1,
                2 * loud_stride + 1,
                stride=loud_stride,
                padding=cc.get_padding(2 * loud_stride + 1, loud_stride),
            ))

        branches = [wave_gen, loud_gen]

        if use_noise:
            noise_gen = NoiseGenerator(out_dim, data_size * n_channels)
            branches.append(noise_gen)

        self.synth = cc.AlignBranches(
            *branches,
            cumulative_delay=self.net.cumulative_delay,
        )

        self.use_noise = use_noise
        self.loud_stride = loud_stride
        self.cumulative_delay = self.synth.cumulative_delay

        self.register_buffer("warmed_up", torch.tensor(0))

    def set_warmed_up(self, state: bool):
        state = torch.tensor(int(state), device=self.warmed_up.device)
        self.warmed_up = state

    def forward(self, x):
        x = self.net(x)

        if self.use_noise:
            waveform, loudness, noise = self.synth(x)
        else:
            waveform, loudness = self.synth(x)
            noise = torch.zeros_like(waveform)

        if self.loud_stride != 1:
            loudness = loudness.repeat_interleave(self.loud_stride)
        loudness = loudness.reshape(x.shape[0], 1, -1)

        waveform = torch.tanh(waveform) * mod_sigmoid(loudness)

        if self.warmed_up and self.use_noise:
            waveform = waveform + noise

        return waveform


class Encoder(nn.Module):

    def __init__(
        self,
        data_size,
        capacity,
        latent_size,
        ratios,
        n_out,
        sample_norm,
        repeat_layers,
        n_channels: int = 1,
        recurrent_layer: Optional[Callable[[], nn.Module]] = None,
        # retro-compatiblity
        spectrogram = None
    ):
        super().__init__()
        data_size = data_size or n_channels
        net = [cc.Conv1d(data_size * n_channels, capacity, 7, padding=cc.get_padding(7))]

        for i, r in enumerate(ratios):
            in_dim = 2**i * capacity
            out_dim = 2**(i + 1) * capacity

            if sample_norm:
                net.append(SampleNorm())
            else:
                net.append(nn.BatchNorm1d(in_dim))
            net.append(nn.LeakyReLU(.2))
            net.append(
                cc.Conv1d(
                    in_dim,
                    out_dim,
                    2 * r + 1,
                    padding=cc.get_padding(2 * r + 1, r),
                    stride=r,
                    cumulative_delay=net[-3].cumulative_delay,
                ))

            for i in range(repeat_layers - 1):
                if sample_norm:
                    net.append(SampleNorm())
                else:
                    net.append(nn.BatchNorm1d(out_dim))
                net.append(nn.LeakyReLU(.2))
                net.append(
                    cc.Conv1d(
                        out_dim,
                        out_dim,
                        3,
                        padding=cc.get_padding(3),
                        cumulative_delay=net[-3].cumulative_delay,
                    ))

        net.append(nn.LeakyReLU(.2))

        if recurrent_layer is not None:
            net.append(
                recurrent_layer(
                    dim=out_dim,
                    cumulative_delay=net[-2].cumulative_delay,
                ))
            net.append(nn.LeakyReLU(.2))

        net.append(
            cc.Conv1d(
                out_dim,
                latent_size * n_out,
                5,
                padding=cc.get_padding(5),
                groups=n_out,
                cumulative_delay=net[-2].cumulative_delay,
            ))

        self.net = cc.CachedSequential(*net)
        self.cumulative_delay = self.net.cumulative_delay

    def forward(self, x):
        z = self.net(x)
        return z


def normalize_dilations(dilations: Union[Sequence[int],
                                         Sequence[Sequence[int]]],
                        ratios: Sequence[int]):
    if isinstance(dilations[0], int):
        dilations = [dilations for _ in ratios]
    return dilations


class EncoderV2(nn.Module):

    def __init__(
        self,
        data_size: Union[int, None],
        capacity: int,
        ratios: Sequence[int],
        latent_size: int,
        n_out: int,
        kernel_size: int,
        dilations: Sequence[int],
        keep_dim: bool = False,
        recurrent_layer: Optional[Callable[[], nn.Module]] = None,
        n_channels: int = 1,
        activation: Callable[[int], nn.Module] = lambda dim: nn.LeakyReLU(.2),
        adain: Optional[Callable[[int], nn.Module]] = None,
        spectrogram = None
    ) -> None:
        super().__init__()
        dilations_list = normalize_dilations(dilations, ratios)
        data_size = data_size or n_channels

        net = [
            normalization(
                cc.Conv1d(
                    data_size * n_channels,
                    capacity,
                    kernel_size=kernel_size * 2 + 1,
                    padding=cc.get_padding(kernel_size * 2 + 1),
                )),
        ]

        num_channels = capacity
        for r, dilations in zip(ratios, dilations_list):
            # ADD RESIDUAL DILATED UNITS
            for d in dilations:
                if adain is not None:
                    net.append(adain(dim=num_channels))
                net.append(
                    Residual(
                        DilatedUnit(
                            dim=num_channels,
                            kernel_size=kernel_size,
                            dilation=d,
                        )))

            # ADD DOWNSAMPLING UNIT
            net.append(
                mark_fn(activation(num_channels), "encoder_act", "pre")
            )

            if keep_dim:
                out_channels = num_channels * r
            else:
                out_channels = num_channels * 2
            net.append(
                normalization(
                    cc.Conv1d(
                        num_channels,
                        out_channels,
                        kernel_size=2 * r,
                        stride=r,
                        padding=cc.get_padding(2 * r, r),
                    )))

            num_channels = out_channels

        net.append(
            mark_fn(activation(num_channels), "encoder_act", "pre")
        )
        net.append(
            normalization(
                cc.Conv1d(
                    num_channels,
                    latent_size * n_out,
                    kernel_size=kernel_size,
                    padding=cc.get_padding(kernel_size),
                )))

        if recurrent_layer is not None:
            net.append(recurrent_layer(latent_size * n_out))

        self.net = cc.CachedSequential(*net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return x
    
    

def _wn(module: nn.Module):
    layer = torch.nn.utils.weight_norm(module)

    def _prepare_scriptable(self):
        for hook in self._forward_pre_hooks.values():
            if (hook.__module__ == "torch.nn.utils.weight_norm"
                    and hook.__class__.__name__ == "WeightNorm"):
                torch.nn.utils.remove_weight_norm(self)
        return self

    layer.__prepare_scriptable__ = _prepare_scriptable.__get__(layer)
    return layer

def Conv2d(*args, **kwargs) -> nn.Module:
    return CachedConv2d(*args, **kwargs, normalization=True)


def ConvTranspose2d(*args, **kwargs) -> nn.Module:
    return _wn(CachedConvTranspose2d(*args, **kwargs))


def Conv1d(*args, **kwargs) -> nn.Module:
    return _wn(cc.Conv1d(*args, **kwargs))


class EncoderBlock2D(nn.Module):

    def __init__(self,
                 in_c: int,
                 out_c: int,
                 kernel_size=3,
                 ratio: int =2,
                 freq_ratio: int =2,
                 kernel_multiplier: int =2,
                 activation: Callable[[int], nn.Module] = lambda dim: nn.SiLU()):
        super().__init__()
        self.act = activation(in_c)
        self.conv = Conv2d(in_c,
                           in_c,
                           kernel_size=kernel_size,
                           padding_vert="same",
                           padding_time=cc.get_padding(kernel_size=kernel_size,
                                                       stride=1,
                                                       mode="causal"))
        self.pool = Conv2d(in_channels=in_c,
                           out_channels=out_c,
                           kernel_size=(freq_ratio * kernel_multiplier + 1,
                                        ratio * kernel_multiplier + 1),
                           stride=(freq_ratio, ratio),
                           padding_vert="same",
                           padding_time=cc.get_padding(
                               kernel_size=ratio * kernel_multiplier + 1,
                               mode="causal",
                               stride=ratio))
        self.proj_pool = nn.AvgPool2d(kernel_size=(freq_ratio, ratio),
                                      stride=(freq_ratio, ratio))
        self.proj = Conv2d(in_c, out_c, kernel_size=1, padding=0)

    def forward(self, x):
        res = x.clone()
        x = self.act(x)
        x = self.conv(x)
        x = self.act(x)
        x = self.pool(x)
        res = self.proj_pool(res)
        res = self.proj(res)
        return x + res
    


class ConvBlock1d(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 *,
                 kernel_size: int = 3,
                 stride: int = 1,
                 dilation: int = 1,
                 activation: Callable[[int], nn.Module] = lambda dim: nn.SiLU(),
                 cumulative_delay=0):
        super().__init__()
        self.activation = activation(in_channels)
        self.project = Conv1d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=cc.get_padding(kernel_size,
                                                     dilation=dilation,
                                                     mode="causal"),
                              dilation=dilation,
                              cumulative_delay=cumulative_delay)

    def __prepare_scriptable__(self):
        for hook in self.project._forward_pre_hooks.values():
            if (hook.__module__ == "torch.nn.utils.weight_norm"
                    and hook.__class__.__name__ == "WeightNorm"):
                torch.nn.utils.remove_weight_norm(self.project)
        return self

    def forward(self, x):
        x = self.activation(x)
        x = self.project(x)
        return x

class Encoder2D(nn.Module):
    def __init__(
        self,
        data_size: int,
        n_freqs: int,
        n_out: int,
        capacity: int,
        n_channels: int,
        ratios_time: Sequence[int],
        ratios_freq: Sequence[int],
        ratios_channels: Sequence[int],
        latent_size: int,
        kernel_size: int,
        recurrent_layer: Optional[Callable[[], nn.Module]] = None,
        activation: Callable[[int], nn.Module] = lambda dim: nn.SiLU(),
        adain: Optional[Callable[[int], nn.Module]] = None,
    )-> None:
        super().__init__()
    

        if adain is not None:
            raise NotImplementedError("AdaIN not implemented for 2D generator yet")

        # Stereo packing/unpacking
        self.n_channels = n_channels
        self.recurrent_layer = recurrent_layer
        self.pack_input = Rearrange('b ch c f t -> (b ch) c f t',
                                     ch=n_channels)
        self.merge_stereo_features = Rearrange('(b ch) c f t -> b (ch c) f t',
                                               ch=n_channels)

        channels = [capacity * m for m in ratios_channels]
        n = len(channels)
        
        self.preconv = Conv2d(data_size,
                              channels[0],
                              kernel_size=7,
                              padding_vert=(7 - 1) // 2,
                              padding_time=cc.get_padding(kernel_size=7,
                                                          stride=1,
                                                          mode="causal"))
        
        self.down_layers = nn.ModuleList()
        for i in range(n - 1):
            self.down_layers.append(
                EncoderBlock2D(in_c=channels[i],
                               out_c=channels[i + 1],
                               kernel_size=kernel_size,
                               ratio=ratios_time[i],
                               freq_ratio=ratios_freq[i],
                               activation=activation))

        self.channels_merge = (EncoderBlock2D(in_c=n_channels * channels[-1],
                                            out_c=channels[-1],
                                            kernel_size=kernel_size,
                                            ratio=1,
                                            freq_ratio=1,
                                            activation=activation)
                             if n_channels > 1 else nn.Identity())
        
        freq_total_ratio = math.prod(ratios_freq)
        self.freq_final_dim = (n_freqs//2) // freq_total_ratio
        
        self.middle_block_encode = ConvBlock1d(
            in_channels=self.freq_final_dim * channels[-1],
            out_channels=latent_size * n_out,
            kernel_size=3,
            activation=activation)
        self.rearrange_encode = Rearrange('b c f t -> b (c f) t')
        
        self.reccurent_layer = recurrent_layer(latent_size * n_out) if recurrent_layer is not None else None
    
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pack_input(x)
        x = x.squeeze(1)
        x = self.preconv(x)
        for layer in self.down_layers:
            x = layer(x)
        if self.n_channels > 1:
            x = self.merge_stereo_features(x)
            x = self.channels_merge(x)
        x = self.rearrange_encode(x)
        x = self.middle_block_encode(x)
        if self.recurrent_layer is not None:
            x = self.reccurent_layer(x)
        return x
        
        




class GeneratorV2(nn.Module):

    def __init__(
        self,
        capacity: int,
        ratios: Sequence[int],
        latent_size: int,
        kernel_size: int,
        dilations: Sequence[int],
        keep_dim: bool = False,
        data_size: Union[int, None] = None,
        recurrent_layer: Optional[Callable[[], nn.Module]] = None,
        n_channels: int = 1,
        amplitude_modulation: bool = False,
        noise_module: Optional[NoiseGeneratorV2] = None,
        activation: Callable[[int], nn.Module] = lambda dim: nn.LeakyReLU(.2),
        adain: Optional[Callable[[int], nn.Module]] = None,
    ) -> None:
        super().__init__()
        if data_size is None:
            data_size = n_channels
        else:
            data_size = data_size * n_channels 
        dilations_list = normalize_dilations(dilations, ratios)[::-1]
        ratios = ratios[::-1]

        if keep_dim:
            num_channels = np.prod(ratios) * capacity
        else:
            num_channels = 2**len(ratios) * capacity

        net = []

        if recurrent_layer is not None:
            net.append(recurrent_layer(latent_size))

        net.append(
            normalization(
                cc.Conv1d(
                    latent_size,
                    num_channels,
                    kernel_size=kernel_size,
                    padding=cc.get_padding(kernel_size),
                )), )

        for r, dilations in zip(ratios, dilations_list):
            # ADD UPSAMPLING UNIT
            if keep_dim:
                out_channels = num_channels // r
            else:
                out_channels = num_channels // 2
            activation_layer = mark_fn(activation(num_channels), "decoder_act", "pre")
            net.append(activation_layer)
            net.append(
                normalization(
                    cc.ConvTranspose1d(num_channels,
                                       out_channels,
                                       2 * r,
                                       stride=r,
                                       padding=r // 2)))

            num_channels = out_channels

            # ADD RESIDUAL DILATED UNITS
            for d in dilations:
                if adain is not None:
                    net.append(adain(num_channels))
                net.append(
                    Residual(
                        DilatedUnit(
                            dim=num_channels,
                            kernel_size=kernel_size,
                            dilation=d,
                        )))

        net.append(
            mark_fn(activation(num_channels), "decoder_act")
        )

        waveform_module = normalization(
            cc.Conv1d(
                num_channels,
                data_size * 2 if amplitude_modulation else data_size,
                kernel_size=kernel_size * 2 + 1,
                padding=cc.get_padding(kernel_size * 2 + 1),
            ))

        self.noise_module = None
        self.waveform_module = None

        if noise_module is not None:
            self.waveform_module = waveform_module
            self.noise_module = noise_module(out_channels, n_channels = n_channels)
        else:
            net.append(waveform_module)

        self.net = cc.CachedSequential(*net)

        self.amplitude_modulation = amplitude_modulation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)

        noise = 0.
        if self.noise_module is not None:
            noise = mark_fn(self.noise_module(x), "noise")
            x = self.waveform_module(x)

        if self.amplitude_modulation:
            x, amplitude = x.split(x.shape[1] // 2, 1)
            x = mark_fn(x, "audio")
            amplitude = mark_fn(torch.sigmoid(amplitude), "amplitude")
            x = x * amplitude

        x = x + noise

        return torch.tanh(x)

    def set_warmed_up(self, state: bool):
        pass



class DecoderBlock2D(nn.Module):

    def __init__(self,
                 in_c:int ,
                 out_c:int ,
                 kernel_size:int ,
                 activation: Callable[[int], nn.Module] = lambda dim: nn.SiLU(),
                 ratio:int =2,
                 freq_ratio:int =2):
        super().__init__()
        self.act = activation(in_c)
        self.up = ConvTranspose2d(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=(2 * freq_ratio + (1 if freq_ratio == 1 else 0),
                         2 * ratio + (1 if ratio == 1 else 0)),
            stride=(freq_ratio, ratio),
            padding=(freq_ratio // 2 + (1 if freq_ratio == 1 else 0),
                     ratio // 2 + (1 if ratio == 1 else 0)))
        self.conv = Conv2d(in_c,
                           in_c,
                           kernel_size=kernel_size,
                           padding_vert="same",
                           padding_time=cc.get_padding(kernel_size=kernel_size,
                                                       stride=1,
                                                       mode="causal"))
        self.proj_pool = nn.Upsample(scale_factor=(freq_ratio, ratio),
                                     mode='nearest')
        self.proj = Conv2d(in_c, out_c, kernel_size=1, padding=0)
        self.cumulative_delay = self.conv.cumulative_delay

    def forward(self, x):
        res = x.clone()
        x = self.act(x)
        x = self.conv(x)
        x = self.act(x)
        x = self.up(x)
        res = self.proj_pool(res)
        res = self.proj(res)
        return x + res


class Generator2D(nn.Module):
    def __init__(
        self,
        data_size: int,
        n_freqs: int,
        capacity: int,
        n_channels: int,
        ratios_time: Sequence[int],
        ratios_freq: Sequence[int],
        ratios_channels: Sequence[int],
        latent_size: int,
        kernel_size: int,
        recurrent_layer: Optional[Callable[[], nn.Module]] = None,
        activation: Callable[[int], nn.Module] = lambda dim: nn.SiLU(),
        adain: Optional[Callable[[int], nn.Module]] = None,
    )-> None:
        super().__init__()

        if adain is not None:
            raise NotImplementedError("AdaIN not implemented for 2D generator yet")
        
        if recurrent_layer is not None:
            self.reccurent_layer = recurrent_layer(latent_size)
        self.recurrent_layer = recurrent_layer
        
        self.n_channels = n_channels


        channels = [2 * capacity * m for m in ratios_channels]
        n = len(channels)
        
        self.outconv = Conv2d(channels[0],
                              data_size,
                              kernel_size=7,
                              padding_vert=(7 - 1) // 2,
                              padding_time=cc.get_padding(kernel_size=7,
                                                          stride=1,
                                                          mode="causal"))
        
        freq_total_ratio = math.prod(ratios_freq)
        self.freq_final_dim = (n_freqs//2) // freq_total_ratio
        
        
        self.rearrange_decode = Rearrange('b (c f) t -> b c f t',
                                          f=self.freq_final_dim)
        
        self.middle_block_decode = ConvBlock1d(
            in_channels=latent_size,
            out_channels=self.freq_final_dim * channels[-1],
            kernel_size=3)
        
        
        self.up_layers = nn.ModuleList()

        for i in range(1, n):
            self.up_layers.append(
                DecoderBlock2D(in_c=channels[n - i],
                               out_c=channels[n - i - 1],
                               kernel_size=kernel_size,
                               ratio=ratios_time[n - i - 1],
                               freq_ratio=ratios_freq[n - i - 1],
                               activation=activation),)


        # Multichannel unpacking
        self.channel_split = (DecoderBlock2D(in_c=channels[-1],
                                            out_c=n_channels * channels[-1],
                                            kernel_size=kernel_size,
                                            ratio=1,
                                            freq_ratio=1,
                                            activation=activation)
                             if n_channels > 1 else nn.Identity())
        
        
        self.unpack_input = Rearrange('(b ch) c f t -> b ch c f t',
                                      ch=n_channels)
        self.split_features = Rearrange('b (ch c) f t -> (b ch) c f t',
                                               ch=n_channels)
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.recurrent_layer is not None:
            x = self.reccurent_layer(x)

        x = self.middle_block_decode(x)

        x = self.rearrange_decode(x)

        if self.n_channels > 1:
            x = self.channel_split(x)
            x = self.split_features(x)

        for layer in self.up_layers:
            x = layer(x)
        x = self.outconv(x)

        x = self.unpack_input(x)
        return x
        
    def set_warmed_up(self, state: bool):
        pass
        



class VariationalEncoder(nn.Module):

    def __init__(self, encoder, beta: float = 1.0, n_channels=1, no_freeze_when_warmed_up=False):
        super().__init__()
        self.encoder = encoder(n_channels=n_channels)
        self.beta = beta
        self.register_buffer("warmed_up", torch.tensor(0))
        self.no_freeze_when_warmed_up = no_freeze_when_warmed_up

    def reparametrize(self, z, temperature: float = 1.):
        mean, scale = z.chunk(2, 1)
        mean = mark_fn(mean, name="latent_mean")
        std = nn.functional.softplus(scale) + 1e-4
        std = mark_fn(std, name="latent_std")
        var = std * std
        logvar = torch.log(var)

        z = torch.randn_like(mean) * temperature * std + mean
        kl = (mean * mean + var - logvar - 1).sum(1).mean()
        z = mark_fn(z, name="latent")

        return z, self.beta * kl

    def set_warmed_up(self, state: bool):
        state = torch.tensor(int(state), device=self.warmed_up.device)
        self.warmed_up = state

    def forward(self, x: torch.Tensor):
        z = mark_fn(self.encoder(x), name="encoder_out")

        if self.warmed_up and not self.no_freeze_when_warmed_up:
            z = z.detach()
        return z


class WasserteinEncoder(nn.Module):

    def __init__(
        self,
        encoder_cls,
        noise_augmentation: int = 0,
        n_channels: int = 1,
        no_freeze_when_warmed_up = False
    ):
        super().__init__()
        self.encoder = encoder_cls(n_channels=n_channels)
        self.register_buffer("warmed_up", torch.tensor(0))
        self.noise_augmentation = noise_augmentation

        self.no_freeze_when_warmed_up = no_freeze_when_warmed_up

    def compute_mean_kernel(self, x, y):
        kernel_input = (x[:, None] - y[None]).pow(2).mean(2) / x.shape[-1]
        return torch.exp(-kernel_input).mean()

    def compute_mmd(self, x, y):
        x_kernel = self.compute_mean_kernel(x, x)
        y_kernel = self.compute_mean_kernel(y, y)
        xy_kernel = self.compute_mean_kernel(x, y)
        mmd = x_kernel + y_kernel - 2 * xy_kernel
        return mmd

    def reparametrize(self, z):
        z_reshaped = z.permute(0, 2, 1).reshape(-1, z.shape[1])
        reg = self.compute_mmd(z_reshaped, torch.randn_like(z_reshaped))
        if self.noise_augmentation:
            noise = torch.randn(z.shape[0], self.noise_augmentation,
                                z.shape[-1]).type_as(z)
            z = torch.cat([z, noise], 1)
        return z, reg.mean()

    def set_warmed_up(self, state: bool):
        state = torch.tensor(int(state), device=self.warmed_up.device)
        self.warmed_up = state

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        if self.warmed_up and not self.no_freeze_when_warmed_up:
            z = z.detach()
        return z


class DiscreteEncoder(nn.Module):

    def __init__(self,
                 encoder_cls,
                 vq_cls,
                 num_quantizers,
                 noise_augmentation: int = 0,
                 n_channels: int = 1,
                 no_freeze_when_warmed_up = False):
        super().__init__()
        self.encoder = encoder_cls(n_channels=n_channels)
        self.rvq = vq_cls()
        self.num_quantizers = num_quantizers
        self.register_buffer("warmed_up", torch.tensor(0))
        self.register_buffer("enabled", torch.tensor(0))
        self.noise_augmentation = noise_augmentation
        self.no_freeze_when_warmed_up = no_freeze_when_warmed_up

    @torch.jit.ignore
    def reparametrize(self, z):
        if self.enabled:
            z, diff, _ = self.rvq(z)
        else:
            diff = torch.zeros_like(z).mean()

        if self.noise_augmentation:
            noise = torch.randn(z.shape[0], self.noise_augmentation,
                                z.shape[-1]).type_as(z)
            z = torch.cat([z, noise], 1)

        return z, diff

    def set_warmed_up(self, state: bool):
        state = torch.tensor(int(state), device=self.warmed_up.device)
        self.warmed_up = state

    def forward(self, x):
        z = self.encoder(x)
        return z


class SphericalEncoder(nn.Module):

    def __init__(self, encoder_cls: Callable[[], nn.Module], n_channels: int = 1, no_freeze_when_warmed_up = False) -> None:
        super().__init__()
        self.encoder = encoder_cls(n_channels=n_channels)
        self.no_freeze_when_warmed_up = no_freeze_when_warmed_up

    def reparametrize(self, z):
        norm_z = z / torch.norm(z, p=2, dim=1, keepdim=True)
        reg = torch.zeros_like(z).mean()
        return norm_z, reg

    def set_warmed_up(self, state: bool):
        pass

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        return z

class Snake(nn.Module):

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(dim, 1))

    @mark_fn(name="snake")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + (self.alpha + 1e-9).reciprocal() * (self.alpha *
                                                       x).sin().pow(2)


class AdaptiveInstanceNormalization(nn.Module):

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.register_buffer("mean_x", torch.zeros(cc.MAX_BATCH_SIZE, dim, 1))
        self.register_buffer("std_x", torch.ones(cc.MAX_BATCH_SIZE, dim, 1))
        self.register_buffer("learn_x", torch.zeros(1))
        self.register_buffer("num_update_x", torch.zeros(1))

        self.register_buffer("mean_y", torch.zeros(cc.MAX_BATCH_SIZE, dim, 1))
        self.register_buffer("std_y", torch.ones(cc.MAX_BATCH_SIZE, dim, 1))
        self.register_buffer("learn_y", torch.zeros(1))
        self.register_buffer("num_update_y", torch.zeros(1))

    def update(self, target: torch.Tensor, source: torch.Tensor,
               num_updates: torch.Tensor) -> None:
        bs = source.shape[0]
        target[:bs] += (source - target[:bs]) / (num_updates + 1)

    def reset_x(self):
        self.mean_x.zero_()
        self.std_x.zero_().add_(1)
        self.num_update_x.zero_()

    def reset_y(self):
        self.mean_y.zero_()
        self.std_y.zero_().add_(1)
        self.num_update_y.zero_()

    def transfer(self, x: torch.Tensor) -> torch.Tensor:
        bs = x.shape[0]

        x = (x - self.mean_x[:bs]) / (self.std_x[:bs] + 1e-5)
        x = x * self.std_y[:bs] + self.mean_y[:bs]

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return x

        if self.learn_y:
            mean = x.mean(-1, keepdim=True)
            std = x.std(-1, keepdim=True)

            self.update(self.mean_y, mean, self.num_update_y)
            self.update(self.std_y, std, self.num_update_y)
            self.num_update_y += 1

            return x

        else:
            if self.learn_x:
                mean = x.mean(-1, keepdim=True)
                std = x.std(-1, keepdim=True)

                self.update(self.mean_x, mean, self.num_update_x)
                self.update(self.std_x, std, self.num_update_x)
                self.num_update_x += 1

            if self.num_update_x and self.num_update_y:
                x = self.transfer(x)

            return x


def leaky_relu(dim: int, alpha: float):
    return nn.LeakyReLU(alpha)


def unit_norm_vector_to_angles(x: torch.Tensor) -> torch.Tensor:
    norms = x.flip(1).pow(2)
    norms[:, 1] += norms[:, 0]
    norms = norms[:, 1:]
    norms = norms.cumsum(1).flip(1).sqrt()
    angles = torch.arccos(x[:, :-1] / norms)
    angles[:, -1] = torch.where(
        x[:, -1] >= 0,
        angles[:, -1],
        2 * np.pi - angles[:, -1],
    )
    angles[:, :-1] = angles[:, :-1] / np.pi
    angles[:, -1] = angles[:, -1] / (2 * np.pi)
    return 2 * (angles - .5)


def angles_to_unit_norm_vector(angles: torch.Tensor) -> torch.Tensor:
    angles = (angles / 2 + .5) % 1
    angles[:, :-1] = angles[:, :-1] * np.pi
    angles[:, -1] = angles[:, -1] * (2 * np.pi)
    cos = angles.cos()
    sin = angles.sin().cumprod(dim=1)
    cos = torch.cat([
        cos,
        torch.ones(cos.shape[0], 1, cos.shape[-1]).type_as(cos),
    ], 1)
    sin = torch.cat([
        torch.ones(sin.shape[0], 1, sin.shape[-1]).type_as(sin),
        sin,
    ], 1)
    return cos * sin


def wrap_around_value(x: torch.Tensor, value: float = 1) -> torch.Tensor:
    return (x + value) % (2 * value) - value
