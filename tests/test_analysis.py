import numpy as np

from src.analysis import frequency_response
from src.synthetic import unit_impulse


def test_unit_impulse_response_is_flat():
    signal, sr = unit_impulse(1024, 48000)
    freqs, mag_db = frequency_response(signal, sr)

    # A unit impulse has a flat (0 dB) spectrum across all frequencies.
    assert np.std(mag_db) < 1e-6
    assert np.allclose(mag_db, 0.0, atol=1e-6)


def test_frequency_response_axes():
    sr = 48000
    n_fft = 1024  # even
    signal = np.zeros(n_fft, dtype=np.float64)
    signal[0] = 1.0

    freqs, mag_db = frequency_response(signal, sr)

    assert len(freqs) == len(mag_db)
    assert freqs[0] == 0.0
    assert freqs[-1] == sr / 2
