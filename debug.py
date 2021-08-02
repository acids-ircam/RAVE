# %%
import numpy as np
from random import random, choice
import matplotlib.pyplot as plt
from glob import glob
import librosa as li
import sounddevice as sd
from scipy.signal import lfilter


def get_omega(min_f=20, max_f=8000, sr=24000):
    min_f = np.log(min_f)
    max_f = np.log(max_f)
    rand = np.exp(random() * (max_f - min_f) + min_f)
    rand = 2 * np.pi * rand / sr
    return rand


def get_coef(omega, amplitude=.9):
    z0 = amplitude * np.exp(1j * omega)
    a = [1, -2 * np.real(z0), abs(z0)**2]
    b = [abs(z0)**2, -2 * np.real(z0), 1]
    return b, a


# %%

audio = choice(glob("/Users/acaillon/Desktop/out_24k/*.wav"))
x, sr = li.load(audio, None)

# %%

omega = get_omega()
b, a = get_coef(omega, .9)
print(omega, b, a)

y = lfilter(b, a, x)

sd.play(x[:sr], sr)
sd.wait()
sd.play(y[:sr], sr)

#%%
plt.plot(x)
plt.plot(y)
plt.xlim([sr//4,sr//8])

print(len(x), len(y))

# %%
