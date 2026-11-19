![logo](rave.png)

# Training setup

1. You should train on a _CUDA-enabled_ machine (i.e with an nvidia-card)
   - You can use either **Linux** or **Windows**
   - However we advise to use **Linux** if available
   - Training RAVE without a hardware accelerator (GPU, TPU) will take ages, and is not recommended
2. Make sure that you have CUDA enabled
   - Go to a terminal an enter `nvidia-smi`
   - If a message appears with the name of your graphic card and the available memory, it's all good !
   - Otherwise, you have to install **cuda** on your computer (we don't provide support for that, lots of guides are available online)
3. Let's install python !

# Python installation

Python is often pre-installed on most computers, but we won't use this version. Instead, we will install a **conda** distribution on the machine. This keeps different versions of python separate for different projects, and allows regular users to install new packages without sudo access.

You can follow the [instructions here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) to install a miniconda environment on your computer.

Once installed, you know that you are inside your miniconda environment if there's a "`(base)`" at the beginning of your terminal.

# RAVE installation

We will create a new virtual environment for RAVE.

```bash
conda create -n rave python=3.9
```

Each time we want to use RAVE, we can (and **should**) activate this environment using

```bash
conda activate rave
```

Let's clone RAVE and install the requirements !

```bash
git clone https://github.com/acids-ircam/RAVE
cd RAVE
pip install -r requirements.txt
```

You can now use `python cli_helper.py` to start a new training !

# About the dataset

A good rule of thumb is **more is better**. You might want to have _at least_ 3h of homogeneous recordings to train RAVE, more if your dataset is complex (e.g mixtures of instruments, lots of variations...)

If you have a folder filled with various audio files (any extension, any sampling rate), you can use the `resample` utility in this folder

```bash
conda activate rave
resample --sr TARGET_SAMPLING_RATE --augment
```

It will convert, resample, crop and augment all audio files present in the directory to an output directory called `out_TARGET_SAMPLING_RATE/` (which is the one you should give to `cli_helper.py` when asked for the path of the .wav files).
