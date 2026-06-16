import numpy as np


def band_mask(freqs_hz, fmin=20.0, fmax=20000.0):
    freqs_hz = np.asarray(freqs_hz, dtype=np.float64)
    return (freqs_hz >= fmin) & (freqs_hz <= fmax)


def flatness_std_db(response_db, freqs_hz, fmin=20.0, fmax=20000.0):
    response_db = np.asarray(response_db, dtype=np.float64)
    mask = band_mask(freqs_hz, fmin, fmax)
    return float(np.std(response_db[mask]))


def deviation_rmse_db(response_db, target_db, freqs_hz, fmin=20.0, fmax=20000.0, align=True):
    response_db = np.asarray(response_db, dtype=np.float64)
    target_db = np.asarray(target_db, dtype=np.float64)
    mask = band_mask(freqs_hz, fmin, fmax)

    dev = response_db[mask] - target_db[mask]
    if align:
        dev = dev - np.mean(dev)
    return float(np.sqrt(np.mean(dev ** 2)))
