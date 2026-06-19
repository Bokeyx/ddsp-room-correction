"""Friendly preset rooms for the app, backed by the synthetic RIR generator.

A general visitor has no room measurement, so the app offers named rooms instead
of seed/RT60 knobs. Each name maps to a deterministic synthetic RIR. Pure: no IO,
no Streamlit. The rooms are simulated, and the app labels them as such.
"""
from src.synthetic import decaying_noise_rir

# friendly name -> (rt60_s, seed)
PRESETS = {
    "Small bedroom": (0.3, 1),
    "Living room": (0.5, 2),
    "Large room": (0.7, 3),
    "Echoey hall": (0.9, 4),
}


def preset_names():
    """Display order of the preset room names."""
    return list(PRESETS)


def preset_rir(name, sr=48000, duration_s=0.5):
    """Return ``(rir, sr)`` for a named preset. Deterministic (fixed seed per
    name). Raises ValueError for an unknown name."""
    if name not in PRESETS:
        raise ValueError(f"unknown preset room: {name!r}; choose from {preset_names()}")
    rt60, seed = PRESETS[name]
    return decaying_noise_rir(sr, duration_s, rt60, seed=seed)
