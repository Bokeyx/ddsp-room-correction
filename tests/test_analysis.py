import numpy as np

from src.analysis import frequency_response, fractional_octave_smooth
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


# --- fractional_octave_smooth ----------------------------------------------

def test_smoothing_preserves_length():
    freqs = np.logspace(np.log10(20.0), np.log10(20000.0), 500)
    mag = np.random.default_rng(0).standard_normal(500)

    smoothed = fractional_octave_smooth(freqs, mag)

    assert smoothed.shape == mag.shape
    assert len(smoothed) == len(mag)


def test_smoothing_leaves_flat_curve_flat():
    # A constant curve has no detail to smooth away; it must stay constant.
    freqs = np.logspace(np.log10(20.0), np.log10(20000.0), 400)
    mag = np.full_like(freqs, 7.5)

    smoothed = fractional_octave_smooth(freqs, mag)

    assert np.allclose(smoothed, 7.5, atol=1e-9)
    assert np.std(smoothed) < 1e-9


def test_smoothing_reduces_noise_std():
    # Heavy bin-to-bin noise on top of a flat trend must be averaged down.
    freqs = np.linspace(0.0, 24000.0, 4000)
    rng = np.random.default_rng(123)
    mag = 10.0 * rng.standard_normal(freqs.shape[0])

    smoothed = fractional_octave_smooth(freqs, mag, fraction=3)

    # The smoothing window spans many bins at higher frequencies, so the
    # spread of the smoothed curve must be clearly smaller than the raw noise.
    assert np.std(smoothed) < 0.5 * np.std(mag)
