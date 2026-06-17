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


def _ramp_to_nyquist(sr=16000, n=4096):
    """A response that grows toward Nyquist, so the worst error sits at the top
    bin -- this is what made a 16 kHz real RIR place a peaking filter exactly at
    Nyquist (0/0 -> NaN)."""
    freqs = np.fft.rfftfreq(n, 1.0 / sr)  # even n -> freqs[-1] == sr/2 exactly
    resp = 30.0 * (freqs / freqs[-1])
    return freqs, resp


@pytest.mark.parametrize("method", ["classic", "ddsp", "fir"])
def test_correct_stays_finite_for_low_sample_rate(method):
    sr = 16000
    freqs, resp = _ramp_to_nyquist(sr)
    target = flat_target(freqs)

    corrected, _ = correct(resp, target, freqs, sr, method=method,
                           n_filters=8, ddsp_iters=20)

    assert np.isfinite(corrected).all()


def test_correct_keeps_filters_below_nyquist_for_low_sample_rate():
    sr = 16000
    freqs, resp = _ramp_to_nyquist(sr)

    _, filters = correct(resp, flat_target(freqs), freqs, sr, method="classic",
                         n_filters=8)

    assert filters  # at least one placed
    assert all(f.freq_hz < sr / 2 for f in filters)


def test_smoothed_sigma_restricts_to_requested_band():
    sr = 16000
    freqs = np.fft.rfftfreq(8192, 1.0 / sr)
    resp = np.where(freqs >= 7900, 30.0, 0.0)  # energy only in the top band

    full = smoothed_sigma(resp, freqs)               # default [20, 20000]
    capped = smoothed_sigma(resp, freqs, fmax=6000)  # excludes the top spike

    assert capped < full
    assert capped < 0.5
