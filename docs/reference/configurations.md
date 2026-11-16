# Configurations

RAVE models are described with [gin](https://github.com/google/gin-config) configuration files,
stored in [`rave/configs/`](https://github.com/acids-ircam/RAVE/tree/master/rave/configs). You select them with `--config` at training time.
Several configurations can be stacked; they are applied in order, later ones overriding earlier
ones:

```bash
rave train --config v2 --config causal --config noise ...
```

> **The default (`v2`) is replaced as soon as you pass `--config`.** Add-on configurations such as
> `causal`, `wasserstein` or `noise` do not include an architecture: always start the list with an
> architecture (`v1`, `v2`, `v2_small`, `v3`, `discrete`...).

The configuration actually used is saved as `config.gin` in the run folder, and embedded in
exported `.ts` files.

## Architectures

Pick exactly one.

| Name              | Description | Latent | Min. GPU memory |
| ----------------- | ----------- | ------ | --------------- |
| `v1`              | Original RAVE: 16-band PQMF, `Encoder`/`Generator`, capacity 64, noise generator. | variational, 128 dims | ~8 GB |
| `v2` *(default)*  | Improved continuous model: `EncoderV2`/`GeneratorV2` with dilated residual stacks, multi-period + multi-scale discriminators, capacity 96. Faster and higher quality. | variational, 128 dims | ~16 GB |
| `v2_small`        | Smaller v2 (capacity 48, ratios `[4,2,2,2]`, i.e. one latent frame every 512 samples instead of 2048), with noise generator and discriminator updated every step. Good for timbre transfer on stationary sounds, and the cheapest v2-family model to train. | variational | ~8 GB |
| `v2_nopqmf`       | *(experimental)* v2 whose decoder outputs the raw waveform (PQMF only at the encoder input). Convenient for model bending. | variational | ~16 GB |
| `v2_nopqmf_small` | *(experimental)* Smaller variant of `v2_nopqmf` (capacity 32), with random compression augmentation. | variational | — |
| `v2_with_augs`    | *(experimental)* v2 with mel-spectrogram encoder input and random compression augmentation. | variational | — |
| `v3`              | v2 + Snake activations + Descript discriminator + Adaptive Instance Normalization (real-time style transfer through `nn~` attributes). | variational | ~32 GB |
| `v4`              | *(experimental)* Spectrogram-domain model: 2D encoder/decoder on a complex STFT, no PQMF. | variational | — |
| `discrete`        | Residual vector quantization (SoundStream / EnCodec-like): 16 quantizers × 1024 codes. Use for token-based priors (e.g. `msprior`). | discrete | ~18 GB |
| `discrete_v3`     | `discrete` + Snake + Descript discriminator. | discrete | — |
| `onnx`            | v1 with capacity 32 and no noise generator. | variational | ~6 GB |
| `raspberry`       | `onnx` with capacity 16: lightweight enough for real-time inference on a Raspberry Pi 4. | variational | ~5 GB |
| `hybrid`          | v2 with a mel-spectrogram encoder input and a GRU in the decoder. | variational | — |

## Latent regularization

Applied on top of a v2-family architecture.

| Name           | Description |
| -------------- | ----------- |
| *(none)*       | Variational auto-encoder (KL divergence). The latent space is large (128) and compressed afterwards with a PCA, selected by `--fidelity` at export. |
| `wasserstein`  | Wasserstein auto-encoder (MMD regularization), 16 latent dimensions plus 128 noise dimensions fed to the decoder. Smoother, more uniformly used latent space; phase 1 lasts 200k steps. No PCA is computed during training: export with `--db_path` to get one, or the raw space is exported. |
| `spherical`    | Spherical auto-encoder: latent vectors live on a unit hypersphere (16 dims), exposed as angles. Projections are not available. |

## Add-ons

| Name                     | Description |
| ------------------------ | ----------- |
| `causal`                 | Causal convolutions: lower latency in real time, slightly lower quality. Choose it at training time (cannot be changed after). |
| `noise`                  | Adds the v2 noise synthesizer (`NoiseGeneratorV2`) to the decoder; helps with noisy / percussive sounds. |
| `snake`                  | Snake activations everywhere (better periodic signals). |
| `adain`                  | Adaptive Instance Normalization in encoder and decoder (style transfer; included in `v3`). |
| `descript_discriminator` | Use the Descript Audio Codec discriminator. |
| `spectral_discriminator` | Add the EnCodec multi-scale spectral discriminator. |
| `no_encoder_freeze`      | Keep training the encoder during phase 2 (latent space keeps evolving). Can be added when resuming. |
| `adv_at_start`           | *(experimental)* Adversarial training from the first step. Can be added when resuming. |
| `normalize_ambient`      | Adds a compressor augmentation tuned for ambient material. |

Multichannel output transforms (`output_transforms/midside`, `output_transforms/ambisonics`) are
selected with `rave train --midside` / `--ambisonics`, not with `--config`.

## Overriding values

Any configurable value can be changed from the command line with `--override`. Upper-case names are
gin *macros* defined at the top of the configuration files; other bindings use the
`Class.parameter` form.

```bash
# bigger / smaller network
rave train --config v2 --override "CAPACITY=64" ...

# model sampling rate (must match `rave preprocess --sampling_rate`)
rave train --config v2 --override "SAMPLING_RATE=48000" ...

# larger latent space for a Wasserstein model
rave train --config v2 --config wasserstein --override "LATENT_SIZE=32" ...

# shorter first phase
rave train --config v2 --override "RAVE.phase_1_duration=300000" ...

# KL target (same as --beta)
rave train --config v2 --override "BetaWarmupCallback.target_value=0.1" ...
```

Useful macros / parameters:

| Binding                               | In                  | Meaning |
| ------------------------------------- | ------------------- | ------- |
| `SAMPLING_RATE`                       | all                 | Model sampling rate (44100). |
| `CAPACITY`                            | all                 | Base number of channels of the networks. |
| `LATENT_SIZE`                         | all                 | Size of the latent space (before PCA). |
| `RATIOS`                              | v1/v2               | Downsampling ratios. Their product × `N_BAND` = number of audio samples per latent frame (2048 for v2). |
| `N_BAND`                              | v1/v2               | Number of PQMF bands (16). |
| `NOISE_AUGMENTATION`                  | v2, wasserstein, discrete | Number of noise channels concatenated to the latent before decoding. |
| `PHASE_1_DURATION` / `RAVE.phase_1_duration` | all          | Steps before adversarial training starts. |
| `RAVE.update_discriminator_every`     | all                 | Discriminator update period. |
| `RAVE.weights`                        | all                 | Loss weights: `audio_distance`, `multiband_audio_distance`, `adversarial`, `feature_matching`. |
| `BetaWarmupCallback.initial_value / target_value / warmup_len` | all | Regularization weight schedule. |

You can also write your own `.gin` file and pass it by path (`--config path/to/my_config.gin`);
start it with `include "configs/v2.gin"` to build on an existing architecture.

## Data augmentations

`--augment` loads transforms from acids-dataset (`acids_dataset/configs/transforms/`), applied to
training chunks on the fly. Pass the file name without extension:

```bash
rave train --config v2 --augment mute --augment compress --augment gain ...
```

Available transforms include:

| Family        | Names |
| ------------- | ----- |
| Dynamics      | `gain`, `gaintransition`, `compress`, `compressrange`, `limiter`, `normalize`, `loudnessnormalization`, `mute` |
| Filtering / EQ| `lowpassfilter`, `highpassfilter`, `bandpassfilter`, `bandstopfilter`, `lowshelffilter`, `highshelffilter`, `peakingfilter`, `sevenbandparametriceq`, `randomeq`, `frequencymasking`, `airabsorption`, `preemphasis` |
| Distortion    | `clip`, `clippingdistortion`, `tanhdistortion`, `bitcrush`, `aliasing`, `mp3compression`, `randomdistort` |
| Noise         | `addnoise`, `addgaussiannoise`, `addgaussiansnr`, `addcolornoise`, `addbackgroundnoise`, `addshortnoises` |
| Time / pitch  | `pitchshift`, `timestretch`, `stretch`, `shift`, `reverse`, `repeatpart`, `timemask`, `randomdelay` |
| Space         | `roomsimulator`, `applyimpulseresponse` |
| Other         | `phasemangle`, `dequantize`, `polarityinversion` |

Augmentations are **mandatory for small datasets** (less than about an hour of audio). More
generally, they train the latent representation on transformed versions of the data: the model
learns to encode and decode everything reachable by the chosen transformations, not only the original
recordings (e.g. `pitchshift` for a wider pitch range, `gain` / `compress` for robustness to input
level, `roomsimulator` for reverberant inputs). Pick them according to the material the model will
receive at use time.

**Always use `mute`**: it replaces ~10 % of the chunks with silence. Recordings rarely contain
silence, so without it the model never learns to output silence and keeps producing noise or hum
when the input stops.

Random cropping, dequantization and phase mangling are always applied.
