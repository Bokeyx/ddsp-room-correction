import pytest

from src.analysis import frequency_response
from src.evaluation import evaluate_rir
from src.pipeline import design_band, smoothed_sigma
from src.synthetic import decaying_noise_rir


def test_evaluate_rir_scores_on_the_same_band_it_corrects():
    # On a 16 kHz RIR the design band is capped below Nyquist; sigma must be
    # measured on that same band, not the full [20, 20000], or the score would
    # penalise a region the correction never touches.
    rir, sr = decaying_noise_rir(16000, 0.5, rt60_s=0.4, seed=0)
    freqs, resp = frequency_response(rir, sr)
    fmin, fmax = design_band(sr)

    result = evaluate_rir(rir, sr, methods=("classic",))

    assert result["before"] == pytest.approx(smoothed_sigma(resp, freqs, fmin, fmax))


def test_evaluate_rir_returns_before_and_each_method():
    rir, sr = decaying_noise_rir(16000, 0.5, rt60_s=0.4, seed=0)

    result = evaluate_rir(rir, sr, methods=("classic",))

    assert set(result) == {"before", "classic"}


def test_evaluate_rir_classic_reduces_sigma_on_colored_room():
    rir, sr = decaying_noise_rir(16000, 0.5, rt60_s=0.4, seed=0)

    result = evaluate_rir(rir, sr, methods=("classic",))

    assert result["classic"] < result["before"]
