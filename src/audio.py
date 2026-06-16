"""M6c: apply a designed correction to actual audio (time domain).

The frequency-domain stages (classic EQ, FIR) decide *what* correction to make;
this module renders that correction onto a real signal so it can be heard in an
A/B listening demo. The peaking biquads reuse `peaking_coeffs` (same source of
truth as the dB response), so the audio you hear matches the curves you design.
"""

import numpy as np
from scipy.signal import sosfilt, tf2sos

from src.eq_classic import peaking_coeffs


def apply_eq_to_signal(filters, signal, sr):
    """Apply a cascade of peaking biquads to `signal` (same length out).

    Empty filter list -> a copy of the signal (no correction = no change).
    """
    signal = np.asarray(signal, dtype=np.float64)
    if not filters:
        return signal.copy()

    sos = np.vstack([tf2sos(*peaking_coeffs(f, sr)) for f in filters])
    return sosfilt(sos, signal)


def apply_fir_to_signal(taps, signal):
    """Convolve `signal` with FIR `taps`, preserving length (mode='same')."""
    taps = np.asarray(taps, dtype=np.float64)
    signal = np.asarray(signal, dtype=np.float64)
    return np.convolve(signal, taps, mode="same")


def pink_noise(n, seed):
    """Reproducible 1/f pink noise, n samples, normalised to max|x| ~= 0.99.

    White noise is shaped in the frequency domain by 1/sqrt(f) (power ~ 1/f),
    with the DC bin guarded (f=0 set to 0) to avoid division by zero / a DC
    offset.
    """
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(n)

    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)

    scale = np.ones_like(freqs)
    scale[1:] = 1.0 / np.sqrt(freqs[1:])
    scale[0] = 0.0  # guard f=0: drop DC

    pink = np.fft.irfft(spectrum * scale, n=n)

    peak = np.max(np.abs(pink))
    if peak > 0:
        pink = pink / peak * 0.99
    return pink
