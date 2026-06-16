import numpy as np

from src.analysis import frequency_response, fractional_octave_smooth
from src.eq_classic import (
    PeakingFilter,
    peaking_coeffs,
    peaking_response_db,
    apply_eq_db,
    design_classic_eq,
)
from scipy.signal import freqz
from src.metrics import flatness_std_db
from src.synthetic import decaying_noise_rir
from src.targets import flat_target


SR = 48000


def _log_freqs(n=400, fmin=20.0, fmax=20000.0):
    return np.logspace(np.log10(fmin), np.log10(fmax), n)


# --- peaking_coeffs (DRY: shared biquad coeffs) ----------------------------

def test_peaking_coeffs_match_response_db():
    """peaking_response_db must be the freqz of the (b, a) returned by
    peaking_coeffs -- proving the response is derived from the same coeffs
    that the time-domain audio path will use (no duplicated formula)."""
    freqs = _log_freqs()
    filt = PeakingFilter(1000.0, 6.0, 4.0)

    b, a = peaking_coeffs(filt, SR)
    assert isinstance(b, np.ndarray) and isinstance(a, np.ndarray)
    assert b.shape == (3,) and a.shape == (3,)

    _, h = freqz(b, a, worN=freqs, fs=SR)
    expected = 20.0 * np.log10(np.abs(h))

    assert np.allclose(peaking_response_db(filt, freqs, SR), expected, atol=1e-12)


# --- peaking_response_db ---------------------------------------------------

def test_peaking_zero_gain_is_flat():
    freqs = _log_freqs()
    resp = peaking_response_db(PeakingFilter(1000.0, 0.0, 4.0), freqs, SR)

    assert resp.shape == freqs.shape
    assert np.allclose(resp, 0.0, atol=1e-9)


def test_peaking_peaks_at_center_and_flat_far_away():
    freqs = np.array([100.0, 1000.0, 10000.0])
    resp = peaking_response_db(PeakingFilter(1000.0, 6.0, 4.0), freqs, SR)

    # At f0 the gain is the design gain.
    assert abs(resp[1] - 6.0) < 0.1
    # Far below and far above, the peaking filter is essentially flat.
    assert abs(resp[0]) < 0.5
    assert abs(resp[2]) < 0.5


# --- apply_eq_db -----------------------------------------------------------

def test_apply_eq_empty_is_zero():
    freqs = _log_freqs()
    eq = apply_eq_db([], freqs, SR)

    assert eq.shape == freqs.shape
    assert np.array_equal(eq, np.zeros_like(freqs))


def test_apply_eq_is_sum_of_individual_responses():
    freqs = _log_freqs()
    f1 = PeakingFilter(200.0, 4.0, 2.0)
    f2 = PeakingFilter(5000.0, -3.0, 3.0)

    combined = apply_eq_db([f1, f2], freqs, SR)
    expected = (
        peaking_response_db(f1, freqs, SR) + peaking_response_db(f2, freqs, SR)
    )

    assert np.allclose(combined, expected, atol=1e-12)


# --- design_classic_eq (headline) -----------------------------------------

def test_design_eq_flattens_known_distortion():
    freqs = _log_freqs()
    # Inject a known +10 dB bump at 1 kHz as the "room" response.
    response_db = peaking_response_db(PeakingFilter(1000.0, 10.0, 3.0), freqs, SR)
    target_db = flat_target(freqs)

    sigma_before = flatness_std_db(response_db, freqs)

    eq = design_classic_eq(response_db, target_db, freqs, SR, n_filters=8, q=3.0)
    corrected = response_db + apply_eq_db(eq, freqs, SR)
    sigma_after = flatness_std_db(corrected, freqs)

    # The bump must be clearly tamed.
    assert sigma_after < 0.5 * sigma_before
    assert sigma_after < 1.0


def test_design_eq_does_not_break_flat_response():
    freqs = _log_freqs()
    response_db = np.zeros_like(freqs)  # already flat
    target_db = flat_target(freqs)

    eq = design_classic_eq(response_db, target_db, freqs, SR, n_filters=8, q=4.0)
    corrected = response_db + apply_eq_db(eq, freqs, SR)

    # Must not introduce ripple where there was none.
    assert flatness_std_db(corrected, freqs) < 1e-6


def test_design_eq_returns_peaking_filters_in_band():
    freqs = _log_freqs()
    response_db = peaking_response_db(PeakingFilter(2000.0, -8.0, 3.0), freqs, SR)
    target_db = flat_target(freqs)

    eq = design_classic_eq(response_db, target_db, freqs, SR, n_filters=5, q=3.0)

    assert all(isinstance(f, PeakingFilter) for f in eq)
    assert all(20.0 <= f.freq_hz <= 20000.0 for f in eq)


# --- M3.5 headline: real (synthetic) RIR ----------------------------------

def test_smoothing_eq_flattens_real_rir():
    """Smoothing + clamp must genuinely reduce sigma on a real RIR.

    Comparison is made on the SMOOTHED response: the raw FFT carries thousands
    of statistical-noise bins that no peaking EQ can (or should) chase, so
    comparing raw sigma is not a fair test of the correction.
    """
    rir, sr = decaying_noise_rir(48000, 0.5, 0.4, seed=42)
    freqs, resp = frequency_response(rir, sr)
    target = flat_target(freqs)

    eq = design_classic_eq(
        resp, target, freqs, sr,
        n_filters=48, q=4.0,
        smoothing_fraction=3, max_gain_db=12.0,
    )
    corrected = resp + apply_eq_db(eq, freqs, sr)

    before = flatness_std_db(fractional_octave_smooth(freqs, resp), freqs)
    after = flatness_std_db(fractional_octave_smooth(freqs, corrected), freqs)

    # Genuine, fair reduction of the broadband trend.
    assert after < before
    assert after < 0.7 * before  # at least 30% reduction

    # Gains must respect the clamp.
    assert all(abs(f.gain_db) <= 12.0 + 1e-9 for f in eq)


def test_naive_mode_does_not_flatten_real_rir():
    """Documents WHY smoothing is required.

    The naive path (smoothing_fraction=None) greedily chases raw-FFT noise
    spikes with wide filters, so on a real RIR it fails to reduce the smoothed
    sigma -- it does not improve it (and typically makes it worse).
    """
    rir, sr = decaying_noise_rir(48000, 0.5, 0.4, seed=42)
    freqs, resp = frequency_response(rir, sr)
    target = flat_target(freqs)

    eq = design_classic_eq(
        resp, target, freqs, sr,
        n_filters=8, q=4.0,
        smoothing_fraction=None,
    )
    corrected = resp + apply_eq_db(eq, freqs, sr)

    before = flatness_std_db(fractional_octave_smooth(freqs, resp), freqs)
    after = flatness_std_db(fractional_octave_smooth(freqs, corrected), freqs)

    # The naive baseline does NOT achieve a meaningful reduction; it is no
    # better than (and in practice worse than) doing nothing.
    assert after >= 0.9 * before
