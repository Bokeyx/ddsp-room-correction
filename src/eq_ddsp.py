"""M4: Differentiable optimisation EQ (DDSP).

Same problem as the greedy classic baseline (`design_classic_eq`), but the
peaking-filter gains are learnable torch parameters optimised by Adam against a
frequency-response-deviation loss. The magnitude response is computed with a
differentiable closed form (no scipy.freqz) so autograd can flow into the gains.

All torch ops run in float64 with zero-initialised gains for deterministic,
reproducible results (no RNG is drawn, so no seeding is required).
"""

import numpy as np
import torch

from src.analysis import fractional_octave_smooth
from src.eq_classic import PeakingFilter
from src.metrics import band_mask


def peaking_response_db_torch(freq_hz, gain_db, q, freqs_hz, sr):
    """Differentiable dB magnitude response of a single RBJ peaking biquad.

    `gain_db`, `freq_hz`, and `q` may each be a torch tensor (a learnable
    parameter); autograd flows through all three. Uses the closed-form
    |H(e^jw)|^2 of a biquad so no scipy.freqz is involved. Plain floats are
    accepted too (the gains-only path passes float freq/q).
    """
    def _t(x):
        if torch.is_tensor(x):
            return x.to(torch.float64)
        return torch.tensor(x, dtype=torch.float64)

    gain_db = _t(gain_db)
    freq_hz = _t(freq_hz)
    q = _t(q)

    w = torch.as_tensor(
        2.0 * np.pi * np.asarray(freqs_hz, dtype=np.float64) / sr,
        dtype=torch.float64,
    )

    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * freq_hz / sr
    alpha = torch.sin(w0) / (2.0 * q)
    cos_w0 = torch.cos(w0)

    b0 = 1.0 + alpha * A
    b1 = -2.0 * cos_w0
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha / A

    cos_w = torch.cos(w)
    cos_2w = torch.cos(2.0 * w)

    num = (b0 ** 2 + b1 ** 2 + b2 ** 2
           + 2.0 * (b0 * b1 + b1 * b2) * cos_w
           + 2.0 * b0 * b2 * cos_2w)
    den = (a0 ** 2 + a1 ** 2 + a2 ** 2
           + 2.0 * (a0 * a1 + a1 * a2) * cos_w
           + 2.0 * a0 * a2 * cos_2w)

    # Guard log10 against a degenerate 0/0 at Nyquist (centre == Nyquist).
    # eps is ~17 orders below normal magnitudes, so the real result is unchanged.
    # Defense-in-depth: pipeline.correct caps the design band below Nyquist, so via
    # that path no centre reaches Nyquist; this guard protects direct callers.
    eps = 1e-20
    return 10.0 * torch.log10(num.clamp_min(eps) / den.clamp_min(eps))


def optimize_eq(
    response_db,
    target_db,
    freqs_hz,
    sr,
    n_filters=32,  # DDSP overtakes greedy classic from nf>=32 (see headline test)
    q=4.0,
    fmin=20.0,
    fmax=20000.0,
    smoothing_fraction=3,
    max_gain_db=12.0,
    iters=300,
    lr=0.5,
    learn_centers=False,
    learn_q=False,
    q_range=(0.5, 10.0),
    return_history=False,
):
    """Optimise peaking-filter gains (and optionally centres and Q) to flatten
    `response_db` toward `target_db` by gradient descent.

    By default only the gains are learned (centres log-spaced over [fmin, fmax],
    Q fixed at `q`) -- identical to the original behaviour. With `learn_centers`
    and/or `learn_q`, the centre frequency / Q become learnable too, optimised
    jointly. They are bounded by a sigmoid reparametrisation -- centres stay in
    [fmin, fmax], Q stays in `q_range` -- so training cannot push a filter past
    Nyquist or to a non-positive Q (which would produce NaNs). Centre/Q latents
    are initialised so the starting filter bank equals the gains-only one.

    Determinism: zero/derived initialisation, deterministic Adam, no RNG.

    Returns a `list[PeakingFilter]`; with `return_history=True` returns
    `(filters, loss_history)` where loss_history is one float per iteration.
    """
    if fmin <= 0.0:
        raise ValueError(f"fmin must be > 0 (got {fmin}); log-spaced filter banks need positive frequencies")

    response_db = np.asarray(response_db, dtype=np.float64)
    target_db = np.asarray(target_db, dtype=np.float64)
    freqs_hz = np.asarray(freqs_hz, dtype=np.float64)

    if smoothing_fraction is not None:
        design_resp = fractional_octave_smooth(
            freqs_hz, response_db, fraction=smoothing_fraction
        )
    else:
        design_resp = response_db

    log_fmin = np.log(fmin)
    log_fmax = np.log(fmax)
    init_centers = np.logspace(np.log10(fmin), np.log10(fmax), n_filters)

    resp_t = torch.as_tensor(design_resp, dtype=torch.float64)
    target_t = torch.as_tensor(target_db, dtype=torch.float64)
    mask = torch.as_tensor(band_mask(freqs_hz, fmin, fmax))

    gains = torch.nn.Parameter(torch.zeros(n_filters, dtype=torch.float64))
    param_groups = [{"params": [gains], "lr": lr}]

    # Centre latents: sigmoid(c) maps a real latent into the normalised
    # log-frequency position; init via logit so the start equals init_centers.
    c = None
    if learn_centers:
        norm = (np.log(init_centers) - log_fmin) / (log_fmax - log_fmin)
        norm = np.clip(norm, 1e-4, 1.0 - 1e-4)
        c_init = np.log(norm / (1.0 - norm))
        c = torch.nn.Parameter(torch.as_tensor(c_init, dtype=torch.float64))
        param_groups.append({"params": [c], "lr": lr * 0.1})

    # Q latents: sigmoid(qc) maps into [q_min, q_max]; init via logit so Q == q.
    q_min, q_max = q_range
    qc = None
    if learn_q:
        if q_min >= q_max:
            raise ValueError(f"q_range must satisfy q_min < q_max; got {q_range}")
        norm_q = (q - q_min) / (q_max - q_min)
        norm_q = float(np.clip(norm_q, 1e-4, 1.0 - 1e-4))
        qc_init = np.log(norm_q / (1.0 - norm_q))
        qc = torch.nn.Parameter(torch.full((n_filters,), qc_init, dtype=torch.float64))
        param_groups.append({"params": [qc], "lr": lr * 0.1})

    opt = torch.optim.Adam(param_groups)
    init_centers_t = torch.as_tensor(init_centers, dtype=torch.float64)

    def current_centers():
        # Re-evaluates sigmoid(c) from the live Parameter each call, rebuilding
        # the graph so gradients flow into c on loss.backward() (no detach).
        if learn_centers:
            return torch.exp(log_fmin + torch.sigmoid(c) * (log_fmax - log_fmin))
        return init_centers_t

    def current_qs():
        if learn_q:
            return q_min + torch.sigmoid(qc) * (q_max - q_min)
        return None

    history = []
    for _ in range(iters):
        opt.zero_grad()
        centers_t = current_centers()
        qs_t = current_qs()
        eq = torch.zeros_like(resp_t)
        for i in range(n_filters):
            fc = centers_t[i] if learn_centers else float(init_centers[i])
            qi = qs_t[i] if learn_q else q
            eq = eq + peaking_response_db_torch(fc, gains[i], qi, freqs_hz, sr)

        current = resp_t + eq
        residual = current - target_t
        residual = residual - residual[mask].mean()
        loss = (residual[mask] ** 2).mean()
        loss.backward()
        opt.step()
        history.append(float(loss.detach()))

    with torch.no_grad():
        final_centers = current_centers().numpy() if learn_centers else init_centers
        final_qs = current_qs().numpy() if learn_q else np.full(n_filters, q)
        final_gains = torch.clamp(gains.detach(), -max_gain_db, max_gain_db).numpy()

    filters = [
        PeakingFilter(freq_hz=float(fc), gain_db=float(g), q=float(qq))
        for fc, g, qq in zip(final_centers, final_gains, final_qs)
    ]
    if return_history:
        return filters, history
    return filters
