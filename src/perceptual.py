"""Perceptual weighting for the DDSP loss and the perceptual-sigma metric.

The DDSP loss and sigma are computed over linearly-spaced FFT bins, which are
dense in the treble and so over-weight high frequencies. These weights re-tilt
the objective toward how we hear: a critical-band density term (1/ERB) and an
ISO 226 40-phon equal-loudness term.
"""
import numpy as np

# ISO 226:2003 equal-loudness contour at 40 phon: frequency (Hz) -> SPL (dB).
_ISO226_40PHON = {
    20: 99.85, 25: 93.94, 31.5: 88.17, 40: 82.63, 50: 77.78, 63: 73.08,
    80: 68.48, 100: 64.37, 125: 60.59, 160: 56.70, 200: 53.41, 250: 50.40,
    315: 47.58, 400: 44.98, 500: 43.05, 630: 41.34, 800: 40.06, 1000: 40.01,
    1250: 41.82, 1600: 42.51, 2000: 39.23, 2500: 36.51, 3150: 35.61,
    4000: 36.65, 5000: 40.01, 6300: 45.83, 8000: 51.80, 10000: 54.28,
    12500: 51.49,
}


def _erb(f):
    """Glasberg-Moore ERB bandwidth (Hz) at frequency f (Hz)."""
    return 24.7 * (4.37 * np.asarray(f, dtype=np.float64) / 1000.0 + 1.0)


def _equal_loudness_weight(freqs_hz):
    """Hearing sensitivity from the ISO 226 40-phon contour.

    Higher where less SPL is needed to reach 40 phon (most sensitive ~3 kHz),
    lower in the bass and top treble. Interpolated in log-frequency, clamped to
    the tabulated range [20, 12500] Hz.
    """
    f = np.asarray(freqs_hz, dtype=np.float64)
    keys = sorted(_ISO226_40PHON)
    tab_f = np.array(keys, dtype=np.float64)
    tab_spl = np.array([_ISO226_40PHON[k] for k in keys], dtype=np.float64)
    fc = np.clip(f, tab_f[0], tab_f[-1])
    spl = np.interp(np.log(fc), np.log(tab_f), tab_spl)
    return 10.0 ** (-(spl - tab_spl.min()) / 20.0)


def perceptual_weights(freqs_hz, use_density=True, use_loudness=True):
    """Per-frequency perceptual weight, normalised to mean 1.

    Combines a critical-band density term (1/ERB, counters the treble-heavy
    linear FFT-bin spacing) and an ISO 226 40-phon equal-loudness term. With
    both flags False returns all-ones (the flat fallback).
    """
    f = np.asarray(freqs_hz, dtype=np.float64)
    w = np.ones_like(f)
    if use_density:
        w = w / _erb(f)
    if use_loudness:
        w = w * _equal_loudness_weight(f)
    return w / w.mean()
