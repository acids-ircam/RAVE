# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import firwin

x = np.random.randn(16).repeat(1024) / 3

fc = 1000
width = 100
sr = 48000

y = np.sin(2 * np.pi * np.cumsum(fc + width * x) / sr)

plt.plot(np.linspace(0, sr / 2, len(y) // 2 + 1), abs(np.fft.rfft(y)))
plt.xlim([500, 10000])
plt.show()


def demodulate(y, fc, sr):
    shift = int(sr / fc / 4)
    y = y[shift:] * y[:-shift]
    filt = firwin(300, 100, fs=sr)
    y = np.convolve(y, filt, mode="same")
    return y


y = -4 * np.pi * demodulate(y, fc, sr)
plt.plot(y)
plt.plot(x)
plt.show()


# %%
