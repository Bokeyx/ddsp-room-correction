import numpy as np


def unit_impulse(length, sr):
    signal = np.zeros(length, dtype=np.float64)
    signal[0] = 1.0
    return signal, sr


def decaying_noise_rir(sr, duration_s, rt60_s, seed):
    n = round(duration_s * sr)
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n)

    t = np.arange(n) / sr
    tau = rt60_s / (3 * np.log(10))
    envelope = np.exp(-t / tau)

    return noise * envelope, sr
