import numpy as np

from src.targets import flat_target


def test_flat_target_is_all_zeros_same_shape():
    freqs = np.array([0.0, 10.0, 100.0, 1000.0, 20000.0])

    target = flat_target(freqs)

    assert target.shape == freqs.shape
    assert np.all(target == 0.0)
