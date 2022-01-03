![rave_logo](docs/rave.png)

# RAVE: Realtime Audio Variational autoEncoder

Official implementation of _RAVE: A variational autoencoder for fast and high-quality neural audio synthesis_ ([article link](https://arxiv.org/abs/2111.05011))

## Installation

RAVE needs `python 3.9`. Install the dependencies using

```bash
pip install -r requirements.txt
```

## Training

Both RAVE and the prior model are available in this repo. For most users we recommand to use the `cli_helper.py` script, since it will generate a set of instructions allowing the training and export of both RAVE and the prior model on a specific dataset.

```bash
python cli_helper.py
```

However, if you want to customize even more your training, you can use the provided `train_{rave, prior}.py` and `export_{rave, prior}.py` scripts manually.

## Reconstructing audio

Once trained, you can reconstruct an entire folder containing wav files using

```bash
python reconstruct.py --ckpt /path/to/checkpoint --wav-folder /path/to/wav/folder
```

You can also export RAVE to a `torchscript` file using `export_rave.py` and use the `encode` and `decode` methods on tensors.

## MAX / MSP - PureData usage

**[NOT AVAILABLE YET]**

RAVE and the prior model can be used in realtime inside [max/msp](https://cycling74.com/), allowing creative interactions with both models. Code and details about this part of the project are not available yet, we are currently working on the corresponding article !

![max_msp_screenshot](docs/maxmsp_screenshot.png)

An audio example of the prior sampling patch is available in the `docs/` folder.
