import numpy as np
import pytest

from src.metrics import band_mask, flatness_std_db, deviation_rmse_db, perceptual_sigma


def test_band_mask_includes_boundaries():
    freqs = np.array([0.0, 10.0, 20.0, 100.0, 20000.0, 21000.0])

    mask = band_mask(freqs)

    expected = np.array([False, False, True, True, True, False])
    assert mask.dtype == bool
    assert np.array_equal(mask, expected)


def test_flatness_constant_response_is_zero():
    freqs = np.array([20.0, 100.0, 1000.0, 10000.0, 20000.0])
    response = np.full_like(freqs, 5.0)

    assert flatness_std_db(response, freqs) == 0.0


def test_flatness_known_plus_minus_3_is_3():
    # All bins in band; half at +3, half at -3 -> std == 3.
    freqs = np.array([100.0, 200.0, 300.0, 400.0])
    response = np.array([3.0, 3.0, -3.0, -3.0])

    assert abs(flatness_std_db(response, freqs) - 3.0) < 1e-9


def test_flatness_is_gain_invariant():
    freqs = np.array([100.0, 200.0, 300.0, 400.0])
    response = np.array([3.0, 3.0, -3.0, -3.0])

    base = flatness_std_db(response, freqs)
    shifted = flatness_std_db(response + 10.0, freqs)

    assert abs(base - shifted) < 1e-9


def test_flatness_ignores_out_of_band_bins():
    # Out-of-band bins are wild; in-band is flat -> std must be 0.
    freqs = np.array([5.0, 20.0, 100.0, 20000.0, 30000.0])
    response = np.array([999.0, 1.0, 1.0, 1.0, -999.0])

    assert flatness_std_db(response, freqs) == 0.0


def test_rmse_equal_to_target_is_zero():
    freqs = np.array([20.0, 100.0, 1000.0, 20000.0])
    response = np.array([1.0, -2.0, 3.0, 0.5])
    target = response.copy()

    assert deviation_rmse_db(response, target, freqs) == 0.0


def test_rmse_align_true_removes_offset():
    freqs = np.array([20.0, 100.0, 1000.0, 20000.0])
    target = np.array([1.0, -2.0, 3.0, 0.5])
    response = target + 5.0

    assert abs(deviation_rmse_db(response, target, freqs, align=True)) < 1e-9


def test_rmse_align_false_keeps_offset():
    freqs = np.array([20.0, 100.0, 1000.0, 20000.0])
    target = np.array([1.0, -2.0, 3.0, 0.5])
    response = target + 5.0

    assert abs(deviation_rmse_db(response, target, freqs, align=False) - 5.0) < 1e-9


def test_perceptual_sigma_equals_flatness_std_with_uniform_weights():
    freqs = np.linspace(0.0, 24000.0, 2049)
    rng = np.random.default_rng(0)
    resp = rng.normal(size=freqs.shape)
    w = np.ones_like(freqs)
    assert perceptual_sigma(resp, freqs, w) == pytest.approx(flatness_std_db(resp, freqs))
