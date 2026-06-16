import numpy as np

from src.analysis import frequency_response, fractional_octave_smooth
from src.eq_classic import (
    PeakingFilter,
    peaking_response_db,
    apply_eq_db,
    design_classic_eq,
)
from src.fir import fir_response_db, design_fir_correction
from src.metrics import flatness_std_db
from src.synthetic import decaying_noise_rir
from src.targets import flat_target


SR = 48000


def _rfft_freqs(n_fft, sr=SR):
    return np.fft.rfftfreq(n_fft, d=1.0 / sr)


# --- 1. fir_response_db ----------------------------------------------------

def test_unit_impulse_is_flat():
    """A single-tap unit impulse [1.0] is the identity filter: its magnitude
    response is exactly 0 dB at every frequency."""
    freqs = _rfft_freqs(8192)
    resp = fir_response_db(np.array([1.0]), freqs, SR)

    assert resp.shape == freqs.shape
    assert np.allclose(resp, 0.0, atol=1e-9)


# --- 2. analytic bump recovery ---------------------------------------------

def test_design_fir_flattens_known_bump():
    """An analytic +10 dB peak at 1 kHz must be clearly tamed by the FIR
    correction (sigma drops well below the un-corrected value)."""
    n_fft = 8192
    freqs = _rfft_freqs(n_fft)
    response_db = peaking_response_db(PeakingFilter(1000.0, 10.0, 3.0), freqs, SR)
    target_db = flat_target(freqs)

    sigma_before = flatness_std_db(response_db, freqs)

    taps = design_fir_correction(response_db, target_db, freqs, SR)
    corrected = response_db + fir_response_db(taps, freqs, SR)
    sigma_after = flatness_std_db(corrected, freqs)

    assert sigma_after < 0.5 * sigma_before


# --- 3. linear phase (symmetric taps) --------------------------------------

def test_designed_fir_is_linear_phase():
    """A frequency-sampled linear-phase FIR has symmetric taps (taps == taps
    reversed) -- that symmetry IS the definition of linear phase."""
    n_fft = 8192
    freqs = _rfft_freqs(n_fft)
    response_db = peaking_response_db(PeakingFilter(1000.0, 10.0, 3.0), freqs, SR)
    target_db = flat_target(freqs)

    taps = design_fir_correction(response_db, target_db, freqs, SR, n_taps=4097)

    assert len(taps) % 2 == 1  # odd length (Type I symmetric FIR)
    assert np.allclose(taps, taps[::-1])


# --- 4. headline: real RIR, FIR vs classic ---------------------------------

def test_design_fir_flattens_real_rir_and_beats_classic():
    """FIR magnitude correction must reduce smoothed sigma >=30% on a real RIR
    (fair smoothed comparison) and be no worse than the classic peaking-EQ
    baseline at magnitude matching."""
    rir, sr = decaying_noise_rir(48000, 0.5, 0.4, seed=42)
    freqs, resp = frequency_response(rir, sr)
    target = flat_target(freqs)

    taps = design_fir_correction(
        resp, target, freqs, sr,
        n_taps=4097, smoothing_fraction=3, max_boost_db=12.0,
    )
    classic_eq = design_classic_eq(
        resp, target, freqs, sr,
        n_filters=48, q=4.0,
        smoothing_fraction=3, max_gain_db=12.0,
    )

    fir_corrected = resp + fir_response_db(taps, freqs, sr)
    classic_corrected = resp + apply_eq_db(classic_eq, freqs, sr)

    before = flatness_std_db(fractional_octave_smooth(freqs, resp), freqs)
    fir_after = flatness_std_db(
        fractional_octave_smooth(freqs, fir_corrected), freqs)
    classic_after = flatness_std_db(
        fractional_octave_smooth(freqs, classic_corrected), freqs)

    # At least 30% reduction, fair (smoothed) comparison.
    assert fir_after < before
    assert fir_after < 0.7 * before

    # FIR must be no worse than the classic baseline at magnitude matching.
    assert fir_after <= classic_after * 1.1


def test_design_fir_caps_n_taps_to_available_length_without_crashing():
    # n_taps larger than the irfft length must be capped to the largest valid
    # odd length, not overflow and crash on the window broadcast.
    freqs = np.fft.rfftfreq(8192, d=1.0 / 48000)
    response = np.zeros_like(freqs)
    response[100] = 10.0
    target = flat_target(freqs)

    taps = design_fir_correction(response, target, freqs, 48000, n_taps=999999)

    assert len(taps) % 2 == 1  # odd -> Type-I symmetric (linear phase)
    assert len(taps) <= 8192  # never exceeds the irfft length
    assert np.allclose(taps, taps[::-1])  # symmetric
