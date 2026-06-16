import numpy as np


# Target-curve functions follow the signature `freqs_hz -> target_db`.
def flat_target(freqs_hz):
    return np.zeros_like(np.asarray(freqs_hz, dtype=np.float64))


def harman_target(freqs_hz, tilt_db_per_oct=-1.0, ref_hz=1000.0):
    """Simplified Harman-style in-room target: a gentle downward tilt.

    Listeners prefer a slightly falling in-room response over a perfectly flat
    one. The full Harman room curve adds a low-shelf bass lift, but the dominant
    feature is a ~-1 dB/octave downward tilt, which is what we model here:
    0 dB at ``ref_hz``, sloping by ``tilt_db_per_oct`` for every octave away.
    """
    freqs_hz = np.asarray(freqs_hz, dtype=np.float64)
    # Frequencies <= 0 (e.g. the DC bin) have no octave relation to ref; clamp
    # them to ref_hz so they read as 0 dB instead of -inf.
    safe = np.where(freqs_hz > 0.0, freqs_hz, ref_hz)
    return tilt_db_per_oct * np.log2(safe / ref_hz)
