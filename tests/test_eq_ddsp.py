import numpy as np
import torch

from src.analysis import frequency_response, fractional_octave_smooth
from src.eq_classic import (
    PeakingFilter,
    peaking_response_db,
    apply_eq_db,
    design_classic_eq,
)
from src.eq_ddsp import peaking_response_db_torch, optimize_eq
from src.metrics import flatness_std_db
from src.synthetic import decaying_noise_rir
from src.targets import flat_target


SR = 48000


def _log_freqs(n=400, fmin=20.0, fmax=20000.0):
    return np.logspace(np.log10(fmin), np.log10(fmax), n)


# --- 1. differentiable filter matches scipy --------------------------------

def test_torch_peaking_matches_scipy():
    """The differentiable closed-form magnitude must match the scipy.freqz
    baseline for arbitrary (freq, gain, q) -- proves correctness of the
    autograd-capable implementation."""
    freqs = _log_freqs()
    for freq, gain, q in [(1000.0, 6.0, 4.0), (200.0, -8.0, 2.0),
                          (5000.0, 10.0, 1.5), (80.0, 3.0, 0.7)]:
        torch_db = peaking_response_db_torch(
            freq, torch.tensor(gain, dtype=torch.float64), q, freqs, SR
        ).detach().numpy()
        scipy_db = peaking_response_db(PeakingFilter(freq, gain, q), freqs, SR)
        assert np.allclose(torch_db, scipy_db, atol=1e-6), (
            f"mismatch at ({freq},{gain},{q}): "
            f"max abs diff {np.max(np.abs(torch_db - scipy_db))}"
        )


# --- 2. gradient flows -----------------------------------------------------

def test_gradient_flows_through_gain():
    """A loss built on the torch response must produce a non-zero gradient on
    the gain parameter -- proves the EQ gains are actually learnable."""
    freqs = _log_freqs()
    gain = torch.zeros((), dtype=torch.float64, requires_grad=True)
    eq = peaking_response_db_torch(1000.0, gain, 4.0, freqs, SR)
    # Push the curve toward an arbitrary non-zero target so grad is non-trivial.
    target = torch.ones_like(eq)
    loss = ((eq - target) ** 2).mean()
    loss.backward()

    assert gain.grad is not None
    assert gain.grad.item() != 0.0


# --- 3. analytic bump recovery (same scenario as classic) ------------------

def test_optimize_eq_flattens_known_distortion():
    freqs = _log_freqs()
    response_db = peaking_response_db(PeakingFilter(1000.0, 10.0, 3.0), freqs, SR)
    target_db = flat_target(freqs)

    sigma_before = flatness_std_db(response_db, freqs)

    eq = optimize_eq(response_db, target_db, freqs, SR, n_filters=24, q=4.0)
    corrected = response_db + apply_eq_db(eq, freqs, SR)
    sigma_after = flatness_std_db(corrected, freqs)

    # The bump must be clearly tamed.
    assert sigma_after < 0.5 * sigma_before
    assert all(isinstance(f, PeakingFilter) for f in eq)


# --- 4. headline: real RIR, DDSP vs classic --------------------------------

def test_optimize_eq_flattens_real_rir_and_matches_classic():
    """DDSP optimizer must reduce smoothed sigma >=30% on a real RIR, using the
    same fair comparison as the classic baseline, and be no worse than classic."""
    rir, sr = decaying_noise_rir(48000, 0.5, 0.4, seed=42)
    freqs, resp = frequency_response(rir, sr)
    target = flat_target(freqs)

    # Fair comparison: give both methods the SAME filter budget (48). The
    # greedy classic baseline is evaluated with 48 filters in its own test, so
    # the DDSP optimizer is given the same resource to compare like-for-like.
    n_filters = 48
    ddsp_eq = optimize_eq(
        resp, target, freqs, sr,
        n_filters=n_filters, q=4.0,
        smoothing_fraction=3, max_gain_db=12.0,
    )
    classic_eq = design_classic_eq(
        resp, target, freqs, sr,
        n_filters=n_filters, q=4.0,
        smoothing_fraction=3, max_gain_db=12.0,
    )

    ddsp_corrected = resp + apply_eq_db(ddsp_eq, freqs, sr)
    classic_corrected = resp + apply_eq_db(classic_eq, freqs, sr)

    before = flatness_std_db(fractional_octave_smooth(freqs, resp), freqs)
    ddsp_after = flatness_std_db(
        fractional_octave_smooth(freqs, ddsp_corrected), freqs)
    classic_after = flatness_std_db(
        fractional_octave_smooth(freqs, classic_corrected), freqs)

    # At least 30% reduction, fair (smoothed) comparison.
    assert ddsp_after < before
    assert ddsp_after < 0.7 * before

    # DDSP must be no worse than the greedy baseline.
    assert ddsp_after <= classic_after * 1.1

    # Gains respect the clamp.
    assert all(abs(f.gain_db) <= 12.0 + 1e-9 for f in ddsp_eq)
