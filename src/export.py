"""Export a designed correction to formats real audio tools import.

Pure functions only -- they take the objects ``pipeline.correct`` already returns
(``list[PeakingFilter]`` for classic/ddsp, an ndarray of taps for fir) and return
strings or bytes. No file IO and no Streamlit dependency, so the app and the
notebook can both reuse them and they are trivially unit tested.

Number formatting is fixed (Fc integer Hz, Gain 2dp, Q 3dp) so output is
deterministic.
"""
import io

import numpy as np
import soundfile as sf


def _peak_line(filt):
    return (f"Fc {round(filt.freq_hz)} Hz "
            f"Gain {filt.gain_db:.2f} dB "
            f"Q {filt.q:.3f}")


def to_eqapo_config(filters):
    """Equalizer APO ``config.txt`` body for a list of PeakingFilter."""
    lines = [
        "# DDSP Room Correction - Equalizer APO config",
        f"# {len(filters)} peaking filters",
    ]
    lines += [f"Filter: ON PK {_peak_line(f)}" for f in filters]
    return "\n".join(lines) + "\n"


def to_rew_filters(filters):
    """REW parametric-filter import text (numbered filters)."""
    lines = ["Filter Settings file"]
    lines += [
        f"Filter {i}: ON PK {_peak_line(f)}"
        for i, f in enumerate(filters, start=1)
    ]
    return "\n".join(lines) + "\n"


def to_fir_wav_bytes(taps, sr):
    """FIR impulse encoded as a mono float32 WAV (for convolution engines)."""
    taps = np.asarray(taps, dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, taps, int(sr), format="WAV", subtype="FLOAT")
    return buf.getvalue()
