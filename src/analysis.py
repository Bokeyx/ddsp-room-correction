import numpy as np


def frequency_response(signal, sr, n_fft=None):
    if n_fft is None:
        n_fft = len(signal)

    spectrum = np.fft.rfft(signal, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

    eps = 1e-12
    magnitude_db = 20 * np.log10(np.abs(spectrum) + eps)

    return freqs, magnitude_db


def fractional_octave_smooth(freqs_hz, magnitude_db, fraction=3):
    """Average the dB curve within a 1/`fraction`-octave window per bin.

    Each output bin at frequency f is the mean (in the dB domain) of all input
    bins inside [f * 2^(-1/(2*fraction)), f * 2^(+1/(2*fraction))]. This kills
    bin-to-bin statistical noise while preserving the broadband trend. The
    f = 0 (DC) bin has no octave neighbourhood and is passed through unchanged.
    """
    freqs_hz = np.asarray(freqs_hz, dtype=np.float64)
    magnitude_db = np.asarray(magnitude_db, dtype=np.float64)

    factor = 2.0 ** (1.0 / (2.0 * fraction))
    out = magnitude_db.copy()

    for i, f in enumerate(freqs_hz):
        if f <= 0.0:
            continue  # DC bin has no octave window; pass through
        lo = f / factor
        hi = f * factor
        sel = (freqs_hz >= lo) & (freqs_hz <= hi)
        if np.any(sel):
            out[i] = magnitude_db[sel].mean()

    return out
