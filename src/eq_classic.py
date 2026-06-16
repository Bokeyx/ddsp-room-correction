from dataclasses import dataclass

import numpy as np
from scipy.signal import freqz

from src.metrics import band_mask

# NOTE: This baseline corrects MAGNITUDE only (dB-domain summation of peaking
# biquads). Phase / time-domain correction is out of scope here and is handled
# in the later FIR stage.


@dataclass
class PeakingFilter:
    freq_hz: float
    gain_db: float
    q: float


def peaking_response_db(filt, freqs_hz, sr):
    freqs_hz = np.asarray(freqs_hz, dtype=np.float64)

    A = 10.0 ** (filt.gain_db / 40.0)
    w0 = 2.0 * np.pi * filt.freq_hz / sr
    alpha = np.sin(w0) / (2.0 * filt.q)
    cos_w0 = np.cos(w0)

    b = np.array([1.0 + alpha * A, -2.0 * cos_w0, 1.0 - alpha * A], dtype=np.float64)
    a = np.array([1.0 + alpha / A, -2.0 * cos_w0, 1.0 - alpha / A], dtype=np.float64)

    _, h = freqz(b, a, worN=freqs_hz, fs=sr)
    return 20.0 * np.log10(np.abs(h))


def apply_eq_db(filters, freqs_hz, sr):
    freqs_hz = np.asarray(freqs_hz, dtype=np.float64)
    eq = np.zeros_like(freqs_hz)
    for filt in filters:
        eq += peaking_response_db(filt, freqs_hz, sr)
    return eq


def design_classic_eq(
    response_db,
    target_db,
    freqs_hz,
    sr,
    n_filters=8,
    q=4.0,
    fmin=20.0,
    fmax=20000.0,
):
    response_db = np.asarray(response_db, dtype=np.float64)
    target_db = np.asarray(target_db, dtype=np.float64)
    freqs_hz = np.asarray(freqs_hz, dtype=np.float64)

    mask = band_mask(freqs_hz, fmin, fmax)
    filters = []
    eq = np.zeros_like(freqs_hz)

    for _ in range(n_filters):
        residual = (response_db + eq) - target_db

        # Only consider in-band bins when locating the worst error.
        masked_abs = np.where(mask, np.abs(residual), -np.inf)
        idx = int(np.argmax(masked_abs))

        if not np.isfinite(masked_abs[idx]) or masked_abs[idx] < 0.1:
            break  # nothing left worth correcting

        filt = PeakingFilter(
            freq_hz=float(freqs_hz[idx]),
            gain_db=float(-residual[idx]),
            q=q,
        )
        filters.append(filt)
        eq = eq + peaking_response_db(filt, freqs_hz, sr)

    return filters
