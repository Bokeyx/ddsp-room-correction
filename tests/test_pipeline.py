import numpy as np
import pytest

from src.analysis import frequency_response
from src.pipeline import correct, smoothed_sigma
from src.synthetic import decaying_noise_rir
from src.targets import flat_target


@pytest.mark.parametrize("method", ["classic", "ddsp", "fir"])
def test_correct_reduces_smoothed_sigma(method):
    rir, sr = decaying_noise_rir(48000, 0.5, 0.4, seed=42)
    freqs, resp = frequency_response(rir, sr)
    target = flat_target(freqs)

    before = smoothed_sigma(resp, freqs)
    corrected_db, _ = correct(resp, target, freqs, sr, method=method, n_filters=32)
    after = smoothed_sigma(corrected_db, freqs)

    assert after < before


def test_correct_rejects_unknown_method():
    freqs = np.fft.rfftfreq(1024, 1.0 / 48000)
    resp = np.zeros_like(freqs)

    with pytest.raises(ValueError):
        correct(resp, resp, freqs, 48000, method="bogus")
