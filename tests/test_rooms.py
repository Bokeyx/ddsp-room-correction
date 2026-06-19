import numpy as np
import pytest

from src.rooms import PRESETS, preset_names, preset_rir


def test_preset_names_are_keys_of_presets():
    names = preset_names()
    assert names == list(PRESETS)
    assert "Small bedroom" in names


def test_preset_rir_is_deterministic_48k():
    rir, sr = preset_rir("Small bedroom")
    assert sr == 48000
    assert rir.ndim == 1 and len(rir) > 0
    rir2, _ = preset_rir("Small bedroom")
    assert np.array_equal(rir, rir2)


def test_preset_rir_unknown_name_raises():
    with pytest.raises(ValueError):
        preset_rir("not a room")
