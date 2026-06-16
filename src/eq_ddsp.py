"""M4: Differentiable optimisation EQ (DDSP).

Same problem as the greedy classic baseline (`design_classic_eq`), but the
peaking-filter gains are learnable torch parameters optimised by Adam against a
frequency-response-deviation loss. The magnitude response is computed with a
differentiable closed form (no scipy.freqz) so autograd can flow into the gains.

All torch ops run in float64 with a fixed seed and zero-initialised gains for
deterministic, reproducible results.
"""

import numpy as np
import torch

from src.analysis import fractional_octave_smooth
from src.eq_classic import PeakingFilter
from src.metrics import band_mask


def peaking_response_db_torch(freq_hz, gain_db, q, freqs_hz, sr):
    """Differentiable dB magnitude response of a single RBJ peaking biquad.

    `gain_db` may be a torch tensor (the learnable parameter); autograd flows
    through it. Uses the closed-form |H(e^jw)|^2 of a biquad so no scipy.freqz
    is involved.
    """
    if not torch.is_tensor(gain_db):
        gain_db = torch.tensor(gain_db, dtype=torch.float64)
    gain_db = gain_db.to(torch.float64)

    w = torch.as_tensor(
        2.0 * np.pi * np.asarray(freqs_hz, dtype=np.float64) / sr,
        dtype=torch.float64,
    )

    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * float(freq_hz) / sr
    alpha = np.sin(w0) / (2.0 * float(q))
    cos_w0 = np.cos(w0)

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

    return 10.0 * torch.log10(num / den)


def optimize_eq(
    response_db,
    target_db,
    freqs_hz,
    sr,
    n_filters=24,
    q=4.0,
    fmin=20.0,
    fmax=20000.0,
    smoothing_fraction=3,
    max_gain_db=12.0,
    iters=300,
    lr=0.5,
):
    """Optimise peaking-filter gains by gradient descent to flatten `response_db`.

    Centre frequencies are fixed (log-spaced over [fmin, fmax]); only the gains
    are learned (torch.nn.Parameter, zero-initialised). The loss mirrors the
    classic baseline for a fair comparison: smooth the response, sum the
    differentiable peaking curves, mean-centre the residual in-band (sigma is
    gain-invariant), and minimise its MSE.

    Returns a `list[PeakingFilter]` (same representation as the classic baseline)
    so it can be evaluated with `apply_eq_db`.
    """
    torch.manual_seed(0)

    response_db = np.asarray(response_db, dtype=np.float64)
    target_db = np.asarray(target_db, dtype=np.float64)
    freqs_hz = np.asarray(freqs_hz, dtype=np.float64)

    if smoothing_fraction is not None:
        design_resp = fractional_octave_smooth(
            freqs_hz, response_db, fraction=smoothing_fraction
        )
    else:
        design_resp = response_db

    centers = np.logspace(np.log10(fmin), np.log10(fmax), n_filters)

    resp_t = torch.as_tensor(design_resp, dtype=torch.float64)
    target_t = torch.as_tensor(target_db, dtype=torch.float64)
    mask = torch.as_tensor(band_mask(freqs_hz, fmin, fmax))

    gains = torch.nn.Parameter(torch.zeros(n_filters, dtype=torch.float64))
    opt = torch.optim.Adam([gains], lr=lr)

    for _ in range(iters):
        opt.zero_grad()
        eq = torch.zeros_like(resp_t)
        for i, fc in enumerate(centers):
            eq = eq + peaking_response_db_torch(fc, gains[i], q, freqs_hz, sr)

        current = resp_t + eq
        residual = (current - target_t)
        # Mean-centre in-band: correct the shape, not the absolute level.
        residual = residual - residual[mask].mean()
        loss = (residual[mask] ** 2).mean()
        loss.backward()
        opt.step()

    final_gains = torch.clamp(gains.detach(), -max_gain_db, max_gain_db).numpy()

    return [
        PeakingFilter(freq_hz=float(fc), gain_db=float(g), q=q)
        for fc, g in zip(centers, final_gains)
    ]
