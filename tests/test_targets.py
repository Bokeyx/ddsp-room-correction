import numpy as np

from src.targets import flat_target, harman_target


def test_flat_target_is_all_zeros_same_shape():
    freqs = np.array([0.0, 10.0, 100.0, 1000.0, 20000.0])

    target = flat_target(freqs)

    assert target.shape == freqs.shape
    assert np.all(target == 0.0)


def test_harman_target_zero_at_reference():
    freqs = np.array([1000.0])

    target = harman_target(freqs, tilt_db_per_oct=-1.0, ref_hz=1000.0)

    assert np.allclose(target, 0.0)


def test_harman_target_tilt_one_octave():
    # One octave above the reference drops by exactly the tilt; one octave
    # below rises by the same amount (downward tilt = negative slope).
    freqs = np.array([500.0, 1000.0, 2000.0])

    target = harman_target(freqs, tilt_db_per_oct=-1.0, ref_hz=1000.0)

    assert np.allclose(target, [1.0, 0.0, -1.0])


def test_harman_target_monotonic_decreasing_and_shape():
    freqs = np.array([20.0, 200.0, 2000.0, 20000.0])

    target = harman_target(freqs)

    assert target.shape == freqs.shape
    assert np.all(np.diff(target) < 0.0)


def test_harman_target_handles_zero_frequency_without_error():
    freqs = np.array([0.0, 1000.0])

    target = harman_target(freqs)

    assert np.all(np.isfinite(target))
