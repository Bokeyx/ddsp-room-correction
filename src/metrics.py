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


def perceptual_sigma(response_db, freqs_hz, weights, fmin=20.0, fmax=20000.0):
    """Weighted std of the response over the band (perceptual flatness).

    With uniform weights this equals ``flatness_std_db``. Pass a weight vector
    from ``perceptual.perceptual_weights`` to score perceptual flatness.
    """
    response_db = np.asarray(response_db, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    mask = band_mask(freqs_hz, fmin, fmax)
    x = response_db[mask]
    w = weights[mask]
    wmean = np.sum(w * x) / np.sum(w)
    return float(np.sqrt(np.sum(w * (x - wmean) ** 2) / np.sum(w)))
