"""
Unified discriminator file for audio autoencoders.
Contains SpectroDiscriminator (multi-scale STFT-based) and re-exports
DescriptDiscriminator for PQMF models.
"""
import typing as tp

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from torch.nn.utils import weight_norm
import gin

from rave.stft import StreamableSTFT

# ─── Loss functions ────────────────────────────────────────────────────────────


def hinge_gan(score_real, score_fake):
    loss_dis = torch.relu(1 - score_real) + torch.relu(1 + score_fake)
    loss_dis = loss_dis.mean()
    loss_gen = -score_fake.mean()
    return loss_dis, loss_gen


def feature_matching_function(x, y, normalize_losses):
    distance = abs(x - y).mean()
    if normalize_losses:
        scale = abs(x).mean()
        return distance / (scale + 1e-8)
    return distance


# ─── Building blocks ──────────────────────────────────────────────────────────


class DiscEncoderBlock2D(nn.Module):
    """2D encoder block for the spectral discriminator (non-causal, no caching)."""

    def __init__(self,
                 in_c,
                 out_c,
                 kernel_size=3,
                 ratio=2,
                 freq_ratio=2,
                 kernel_multiplier=2):
        super().__init__()
        self.act = nn.LeakyReLU(negative_slope=0.2)
        self.ln1 = nn.InstanceNorm2d(in_c)
        self.ln2 = nn.InstanceNorm2d(out_c)

        self.conv = nn.Conv2d(in_c,
                              in_c,
                              kernel_size=kernel_size,
                              padding="same")
        self.pool = nn.Conv2d(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=(freq_ratio * kernel_multiplier + 1,
                         ratio * kernel_multiplier + 1),
            stride=(freq_ratio, ratio),
            padding=((freq_ratio + 1) // 2 + (1 if freq_ratio > 1 else 0),
                     (ratio + 1) // 2 + (1 if ratio > 1 else 0)),
        )
        self.proj_pool = nn.AvgPool2d(kernel_size=(freq_ratio, ratio),
                                      stride=(freq_ratio, ratio))
        self.proj = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0)

    def forward(self, x):
        res = x.clone()
        x = self.act(x)
        x = self.conv(x)
        x = self.ln1(x)
        x = self.act(x)
        x = self.pool(x)
        x = self.ln2(x)
        res = self.proj_pool(res)
        res = self.proj(res)
        return x + res


class DiscriminatorSTFT(nn.Module):
    """Single-scale STFT discriminator."""

    def __init__(self,
                 in_size=3,
                 n_fft=1024,
                 hop_length=512,
                 channels=None,
                 time_ratios=None,
                 freq_ratios=None,
                 kernel_size=3,
                 audio_channels=1):
        super().__init__()
        if channels is None:
            channels = [32, 32, 64, 64, 128, 128]
        if time_ratios is None:
            time_ratios = [1, 2, 1, 2, 1, 2]
        if freq_ratios is None:
            freq_ratios = [2, 2, 2, 2, 2, 2]

        self.nfft = n_fft
        self.hop_length = hop_length
        self.audio_channels = audio_channels

        self.spec_transform = StreamableSTFT(nfft=n_fft,
                                             hop_size=hop_length,
                                             stream=False,
                                             skip_features=-1,
                                             normalize=False,
                                             log1p=False)

        self.preconv = nn.Conv2d(in_size,
                                 channels[0],
                                 kernel_size=7,
                                 padding="same")
        self.pre_ln = nn.InstanceNorm2d(channels[0])

        n = len(time_ratios)
        self.down_layers = nn.ModuleList()
        for i in range(n):
            self.down_layers.append(
                DiscEncoderBlock2D(
                    in_c=channels[i],
                    out_c=channels[min(i + 1, n - 1)],
                    kernel_size=kernel_size,
                    ratio=time_ratios[i],
                    freq_ratio=freq_ratios[i],
                ))

        if audio_channels == 2:
            self.stereo_merge = DiscEncoderBlock2D(in_c=2 * channels[-1],
                                                   out_c=channels[-1],
                                                   kernel_size=kernel_size,
                                                   ratio=1,
                                                   freq_ratio=1)

        freq_comp = int(np.prod(freq_ratios))
        freq_stride = max(1, n_fft // freq_comp)
        self.post_conv = nn.Conv2d(channels[-1],
                                   1,
                                   kernel_size=(freq_stride, 1),
                                   stride=(freq_stride, 1),
                                   padding=(((freq_stride + 1) // 2), 0))

    def forward(self, x):
        fmaps = []
        if self.audio_channels > 1:
            x = rearrange(x, 'b ch t -> (b ch) 1 t')

        x = self.spec_transform(x)
        x = x.squeeze(1)
        magnitude = torch.sqrt(x[:, :1]**2 + x[:, 1:]**2 + 1e-8)
        x = torch.cat((magnitude, x), dim=1)

        x = self.preconv(x)
        x = self.pre_ln(x)

        for layer in self.down_layers:
            x = layer(x)
            fmaps.append(x.clone())

        if self.audio_channels > 1:
            x = rearrange(x,
                          '(b ch) c f t -> b (ch c) f t',
                          ch=self.audio_channels)
            x = self.stereo_merge(x)

        x = self.post_conv(x)
        fmaps.append(x)
        return fmaps


class MultiScaleSTFTDiscriminator(nn.Module):

    def __init__(self,
                 channels=None,
                 n_ffts=None,
                 hop_lengths=None,
                 audio_channels=1,
                 **kwargs):
        super().__init__()
        if channels is None:
            channels = [32, 32, 64, 64, 128]
        if n_ffts is None:
            n_ffts = [1024, 2048, 512]
        if hop_lengths is None:
            hop_lengths = [256, 512, 128]
        assert len(n_ffts) == len(hop_lengths)

        self.discriminators = nn.ModuleList([
            DiscriminatorSTFT(n_fft=n_ffts[i],
                              hop_length=hop_lengths[i],
                              channels=channels,
                              audio_channels=audio_channels,
                              **kwargs) for i in range(len(n_ffts))
        ])

    def forward(self, x):
        fmaps = []
        for disc in self.discriminators:
            fmap = disc(x)
            fmaps.append(fmap)
        return fmaps


# ─── Top-level discriminator ──────────────────────────────────────────────────


@gin.configurable
class SpectroDiscriminator(nn.Module):
    """
    Multi-scale STFT discriminator with hinge GAN loss and feature matching.
    Compatible with the Trainer interface: forward(x, y) -> (loss_gen, loss_dis, loss_dict).
    """

    def __init__(self,
                 n_ffts=None,
                 hop_lengths=None,
                 capacity=32,
                 channels=None,
                 weights=None,
                 normalize_losses=True,
                 n_channels=1):
        super().__init__()
        if n_ffts is None:
            n_ffts = [2048, 1024, 512, 256, 128]
        if hop_lengths is None:
            hop_lengths = [512, 256, 128, 64, 32]
        if channels is None:
            channels = [1, 2, 4, 4, 4]
        if weights is None:
            weights = {"feature_matching": 100.0, "adversarial": 1.0}

        scaled_channels = [capacity * c for c in channels]
        self.discriminators = MultiScaleSTFTDiscriminator(
            n_ffts=n_ffts,
            hop_lengths=hop_lengths,
            channels=scaled_channels,
            audio_channels=n_channels)
        self.weights = weights
        self.normalize_losses = normalize_losses

    def get_losses_names(self):
        return [
            "feature_matching",
            "pred_real",
            "pred_fake",
            "discriminator",
            "adversarial",
        ]

    def forward(self, x):
        features = self.discriminators(x)
        return features 
    
        # logits_fake, feature_fake = self.discriminators(y)

        # dis_loss = torch.tensor(0.)
        # adv_loss = torch.tensor(0.)
        # fm_distance = 0.
        # pred_fake, pred_true = 0., 0.

        # for i, (scale_true,
        #         scale_fake) in enumerate(zip(feature_true, feature_fake)):
        #     fm_distance += sum(
        #         feature_matching_function(a, b, self.normalize_losses)
        #         for a, b in zip(scale_true, scale_fake)) / len(scale_true)

        #     _dis, _adv = hinge_gan(logits_true[i], logits_fake[i])
        #     dis_loss = dis_loss + _dis
        #     adv_loss = adv_loss + _adv
        #     pred_fake += logits_fake[i].mean()
        #     pred_true += logits_true[i].mean()

        # n = len(logits_true)
        # dis_loss = dis_loss / n
        # adv_loss = adv_loss / n
        # fm_distance = fm_distance / n

        # loss_dict = {
        #     "discriminator":
        #     dis_loss.item(),
        #     "adversarial":
        #     adv_loss.item(),
        #     "feature_matching":
        #     fm_distance.item()
        #     if torch.is_tensor(fm_distance) else fm_distance,
        #     "pred_real":
        #     pred_true.mean().item()
        #     if torch.is_tensor(pred_true) else pred_true,
        #     "pred_fake":
        #     pred_fake.mean().item()
        #     if torch.is_tensor(pred_fake) else pred_fake,
        # }

        # loss_gen = (adv_loss * self.weights["adversarial"] +
        #             fm_distance * self.weights["feature_matching"])

        # return loss_gen, dis_loss, loss_dict
