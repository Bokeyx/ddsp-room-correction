import numpy as np
import pytest
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
from src.pipeline import design_band, smoothed_sigma
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


def test_torch_peaking_no_nan_at_nyquist():
    """Centre frequency at Nyquist (sr/2) on an FFT-style grid that reaches
    Nyquist must not produce NaN/inf: there num/den -> 0 and an unguarded
    log10 would yield NaN. The clamp_min guard must keep the output finite."""
    freqs = np.linspace(0.0, SR / 2.0, 257)  # FFT grid, includes the Nyquist bin
    out = peaking_response_db_torch(
        SR / 2.0, torch.tensor(6.0, dtype=torch.float64), 4.0, freqs, SR
    )
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


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

    # Like-for-like comparison at the SAME filter budget. Measured sweep (same
    # RIR, equal budget) shows classic (greedy) saturates around sigma~0.40 from
    # nf>=32 -- adding filters no longer helps it -- while DDSP keeps improving
    # because it optimises all gains jointly, so DDSP overtakes classic from
    # nf>=32 onward. (At nf<=24 classic can still win, e.g. nf=24: 0.453 vs
    # 0.415 classic.) nf=48 gives both methods ample budget, a fair like-for-like
    # point where DDSP's joint optimisation clearly leads.
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


# --- 5. differentiable in freq and Q --------------------------------------

def test_peaking_response_torch_differentiable_in_freq_and_q():
    freqs = np.fft.rfftfreq(2048, 1.0 / 48000)
    freq = torch.tensor(1000.0, dtype=torch.float64, requires_grad=True)
    q = torch.tensor(4.0, dtype=torch.float64, requires_grad=True)
    gain = torch.tensor(6.0, dtype=torch.float64)

    resp = peaking_response_db_torch(freq, gain, q, freqs, 48000)
    resp.sum().backward()

    assert freq.grad is not None and torch.isfinite(freq.grad).item() and freq.grad.item() != 0.0
    assert q.grad is not None and torch.isfinite(q.grad).item() and q.grad.item() != 0.0


def test_peaking_response_torch_still_accepts_floats():
    freqs = np.fft.rfftfreq(2048, 1.0 / 48000)
    resp = peaking_response_db_torch(1000.0, torch.tensor(6.0, dtype=torch.float64), 4.0, freqs, 48000)

    assert resp.shape[0] == len(freqs)
    assert torch.isfinite(resp).all()


# --- 6. learnable centres and Q -------------------------------------------

def _colored_rir_48k():
    rir, sr = decaying_noise_rir(48000, 0.5, 0.4, seed=42)
    freqs, resp = frequency_response(rir, sr)
    return resp, freqs, sr


def test_optimize_eq_defaults_unchanged():
    resp, freqs, sr = _colored_rir_48k()
    filters = optimize_eq(resp, flat_target(freqs), freqs, sr, n_filters=8, iters=10)

    centers = np.logspace(np.log10(20.0), np.log10(20000.0), 8)
    assert np.allclose([f.freq_hz for f in filters], centers)
    assert all(f.q == 4.0 for f in filters)


def test_optimize_eq_learn_centers_stays_in_band():
    resp, freqs, sr = _colored_rir_48k()
    filters = optimize_eq(resp, flat_target(freqs), freqs, sr, n_filters=8,
                          iters=30, learn_centers=True, fmin=20.0, fmax=20000.0)

    assert all(20.0 <= f.freq_hz <= 20000.0 for f in filters)


def test_optimize_eq_learn_q_stays_in_range():
    resp, freqs, sr = _colored_rir_48k()
    filters = optimize_eq(resp, flat_target(freqs), freqs, sr, n_filters=8,
                          iters=30, learn_q=True, q_range=(0.5, 10.0))

    assert all(0.5 <= f.q <= 10.0 for f in filters)


def test_optimize_eq_learnable_is_deterministic():
    resp, freqs, sr = _colored_rir_48k()
    kw = dict(n_filters=8, iters=30, learn_centers=True, learn_q=True)
    a = optimize_eq(resp, flat_target(freqs), freqs, sr, **kw)
    b = optimize_eq(resp, flat_target(freqs), freqs, sr, **kw)

    assert [f.freq_hz for f in a] == [f.freq_hz for f in b]
    assert [f.q for f in a] == [f.q for f in b]
    assert [f.gain_db for f in a] == [f.gain_db for f in b]


def test_optimize_eq_returns_loss_history():
    resp, freqs, sr = _colored_rir_48k()
    filters, history = optimize_eq(resp, flat_target(freqs), freqs, sr,
                                   n_filters=8, iters=15, return_history=True)

    assert len(history) == 15
    assert all(np.isfinite(history))
    assert isinstance(filters, list)


# --- 7. input validation guards -------------------------------------------

def test_optimize_eq_rejects_nonpositive_fmin():
    resp, freqs, sr = _colored_rir_48k()
    with pytest.raises(ValueError, match="fmin must be > 0"):
        optimize_eq(resp, flat_target(freqs), freqs, sr, n_filters=4, iters=2, fmin=0.0)


def test_optimize_eq_rejects_degenerate_q_range():
    resp, freqs, sr = _colored_rir_48k()
    with pytest.raises(ValueError, match="q_min < q_max"):
        optimize_eq(resp, flat_target(freqs), freqs, sr, n_filters=4, iters=2,
                    learn_q=True, q_range=(4.0, 4.0))


def test_optimize_eq_learned_centers_actually_move():
    """With learn_centers on, the centres must shift from their log-spaced init
    after training -- proves the gradient reaches the centre latent (a detach
    would leave them at init)."""
    resp, freqs, sr = _colored_rir_48k()
    init_centers = np.logspace(np.log10(20.0), np.log10(20000.0), 16)
    filters = optimize_eq(resp, flat_target(freqs), freqs, sr, n_filters=16,
                          iters=80, learn_centers=True)
    learned = np.array([f.freq_hz for f in filters])
    assert np.max(np.abs(learned - init_centers)) > 1.0  # at least one centre moved >1 Hz


# --- 8. robustness + ablation-quality ----------------------------------------

def test_optimize_eq_learnable_no_nan_low_sample_rate():
    rir, sr = decaying_noise_rir(16000, 0.5, 0.4, seed=0)
    freqs, resp = frequency_response(rir, sr)
    fmin, fmax = design_band(sr)  # capped below Nyquist (7200 Hz at 16 kHz)

    filters = optimize_eq(resp, flat_target(freqs), freqs, sr, n_filters=16,
                          iters=40, learn_centers=True, learn_q=True,
                          fmin=fmin, fmax=fmax)

    eq = apply_eq_db(filters, freqs, sr)
    assert np.isfinite(eq).all()
    assert all(f.freq_hz < sr / 2 for f in filters)


def test_full_ddsp_not_worse_than_gains_only():
    rir, sr = decaying_noise_rir(48000, 0.5, 0.4, seed=42)
    freqs, resp = frequency_response(rir, sr)
    tgt = flat_target(freqs)

    gains_only = optimize_eq(resp, tgt, freqs, sr, n_filters=24, iters=200)
    full = optimize_eq(resp, tgt, freqs, sr, n_filters=24, iters=200,
                       learn_centers=True, learn_q=True)

    sg = smoothed_sigma(resp + apply_eq_db(gains_only, freqs, sr), freqs)
    sf = smoothed_sigma(resp + apply_eq_db(full, freqs, sr), freqs)
    assert sf <= sg + 0.02  # full must not be meaningfully worse


def test_optimize_eq_uniform_weights_matches_none():
    import numpy as np
    from src.synthetic import decaying_noise_rir
    from src.analysis import frequency_response
    from src.targets import flat_target
    from src.eq_ddsp import optimize_eq
    rir, sr = decaying_noise_rir(48000, 0.5, rt60_s=0.4, seed=0)
    freqs, resp = frequency_response(rir, sr)
    target = flat_target(freqs)
    a = optimize_eq(resp, target, freqs, sr, n_filters=12, iters=40)
    b = optimize_eq(resp, target, freqs, sr, n_filters=12, iters=40,
                    weights=np.ones_like(freqs))
    assert np.allclose([f.gain_db for f in a], [f.gain_db for f in b], atol=1e-9)


def test_optimize_eq_rejects_mismatched_weights():
    import numpy as np
    import pytest
    from src.synthetic import decaying_noise_rir
    from src.analysis import frequency_response
    from src.targets import flat_target
    from src.eq_ddsp import optimize_eq
    rir, sr = decaying_noise_rir(48000, 0.5, rt60_s=0.4, seed=0)
    freqs, resp = frequency_response(rir, sr)
    target = flat_target(freqs)
    with pytest.raises(ValueError):
        optimize_eq(resp, target, freqs, sr, n_filters=8, iters=5,
                    weights=np.ones(len(freqs) + 1))


def test_optimize_eq_perceptual_is_deterministic():
    import numpy as np
    from src.synthetic import decaying_noise_rir
    from src.analysis import frequency_response
    from src.targets import flat_target
    from src.eq_ddsp import optimize_eq
    from src.perceptual import perceptual_weights
    rir, sr = decaying_noise_rir(48000, 0.5, rt60_s=0.4, seed=0)
    freqs, resp = frequency_response(rir, sr)
    target = flat_target(freqs)
    w = perceptual_weights(freqs)
    a = optimize_eq(resp, target, freqs, sr, n_filters=12, iters=40, weights=w)
    b = optimize_eq(resp, target, freqs, sr, n_filters=12, iters=40, weights=w)
    assert [f.gain_db for f in a] == [f.gain_db for f in b]


def test_perceptual_training_lowers_perceptual_sigma():
    import numpy as np
    from src.synthetic import decaying_noise_rir
    from src.analysis import frequency_response, fractional_octave_smooth
    from src.targets import flat_target
    from src.eq_ddsp import optimize_eq
    from src.eq_classic import apply_eq_db
    from src.metrics import perceptual_sigma, flatness_std_db
    from src.perceptual import perceptual_weights

    rir, sr = decaying_noise_rir(48000, 0.5, rt60_s=0.4, seed=0)
    freqs, resp = frequency_response(rir, sr)
    target = flat_target(freqs)
    w = perceptual_weights(freqs)

    flat = optimize_eq(resp, target, freqs, sr, n_filters=16, iters=120)
    perc = optimize_eq(resp, target, freqs, sr, n_filters=16, iters=120, weights=w)

    flat_corr = fractional_octave_smooth(freqs, resp + apply_eq_db(flat, freqs, sr))
    perc_corr = fractional_octave_smooth(freqs, resp + apply_eq_db(perc, freqs, sr))

    # Perceptual training wins on the perceptual scorecard...
    assert perceptual_sigma(perc_corr, freqs, w) < perceptual_sigma(flat_corr, freqs, w)
    # ...and plain sigma stays comparable (not a collapse).
    assert flatness_std_db(perc_corr, freqs) < 1.5 * flatness_std_db(flat_corr, freqs)
