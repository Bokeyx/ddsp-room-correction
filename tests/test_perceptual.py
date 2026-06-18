import numpy as np
import pytest

from src.perceptual import perceptual_weights


def test_weights_positive_and_mean_normalised():
    freqs = np.linspace(0.0, 24000.0, 2049)
    w = perceptual_weights(freqs)
    assert np.all(w > 0)
    assert w.mean() == pytest.approx(1.0)


def test_density_upweights_low_frequencies():
    f = np.array([50.0, 5000.0])
    w = perceptual_weights(f, use_loudness=False)
    assert w[0] > w[1]


def test_both_flags_off_is_flat():
    freqs = np.linspace(20.0, 20000.0, 512)
    w = perceptual_weights(freqs, use_density=False, use_loudness=False)
    assert np.allclose(w, 1.0)


def test_equal_loudness_peaks_in_mid():
    f = np.array([50.0, 3000.0, 12000.0])
    w = perceptual_weights(f, use_density=False)
    assert w[1] > w[0]
    assert w[1] > w[2]
