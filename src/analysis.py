import numpy as np


def frequency_response(signal, sr, n_fft=None):
    if n_fft is None:
        n_fft = len(signal)

    spectrum = np.fft.rfft(signal, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

    eps = 1e-12
    magnitude_db = 20 * np.log10(np.abs(spectrum) + eps)

    return freqs, magnitude_db
