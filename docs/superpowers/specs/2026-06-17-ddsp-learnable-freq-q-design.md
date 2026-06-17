# Design: Learnable-frequency/Q DDSP EQ

- **Date:** 2026-06-17
- **Status:** Approved (ready for implementation plan)
- **Topic:** Let the DDSP optimizer learn each peaking filter's centre frequency and Q,
  not just its gain.

## Goal

Today the DDSP EQ (`src/eq_ddsp.optimize_eq`) optimizes only the **gains** of a bank of
peaking filters whose **centre frequencies are fixed** (log-spaced) and whose **Q is fixed**
(4.0). This extends it so the centre frequency and Q are also learnable parameters, optimized
jointly by gradient descent. That is the real strength of a *differentiable* approach, and it
gives a clean ablation: does learning freq/Q beat learning gains alone?

A peaking filter has three knobs — **where** it sits (centre frequency), **how much** it
boosts/cuts (gain), and **how wide** it is (Q). Only "how much" is learned today; this adds
"where" and "how wide."

## Background (current state)

- `optimize_eq` builds `n_filters` peaking biquads at log-spaced centres over `[fmin, fmax]`,
  Q fixed. Only the gains are `torch.nn.Parameter` (zero-init), optimized with Adam against the
  in-band, mean-centred MSE deviation from the target. Deterministic (no RNG). Returns
  `list[PeakingFilter]`.
- `peaking_response_db_torch(freq_hz, gain_db, q, freqs_hz, sr)` computes a differentiable
  closed-form `|H|^2`, but **freq_hz and q are plain Python floats** (w0/alpha/cos are numpy),
  so autograd flows only into `gain_db`.
- `pipeline.correct` caps the design band to `design_band(sr)` (≤ 0.45·sr) for Nyquist safety.

## Design

### 1. API surface (`src/eq_ddsp.py`)

Extend `optimize_eq` with opt-in flags; **defaults reproduce today's behaviour exactly** so the
existing headline numbers and tests are unchanged:

```
optimize_eq(response_db, target_db, freqs_hz, sr,
            n_filters=32, q=4.0, fmin=20.0, fmax=20000.0,
            smoothing_fraction=3, max_gain_db=12.0, iters=300, lr=0.5,
            learn_centers=False, learn_q=False, q_range=(0.5, 10.0),
            return_history=False)
```

- `learn_centers` / `learn_q`: when on, centres / Q become learnable.
- `q_range`: bounds for a learnable Q.
- `return_history`: when `True`, return `(filters, loss_history)` (a list of per-iteration loss
  floats) for the convergence plot; default `False` returns just `list[PeakingFilter]`
  (backward compatible).
- Return type stays `list[PeakingFilter]`; learned freq/Q values are written back into the
  filters so the rest of the pipeline (`apply_eq_db`, audio) is unchanged.

### 2. Bounded reparametrization (stability)

Optimizing raw Hz / Q diverges (filters leave the band, Q goes non-positive → NaN). Instead
optimize unconstrained latents mapped through `sigmoid` into safe ranges:

- **Centre frequency:** latent `c_i ∈ ℝ`, `log_f = log(fmin) + sigmoid(c_i)·(log(fmax) − log(fmin))`,
  `freq_i = exp(log_f)`. Initialize `c_i` (via logit of the normalized log-position) so the
  starting centres equal today's log-spaced centres — keep the good init.
- **Q:** latent `qc_i ∈ ℝ`, `Q_i = q_min + sigmoid(qc_i)·(q_max − q_min)`. Initialize so `Q_i = q`
  (default 4.0).
- **Gains:** unchanged (unconstrained latent, zero-init, post-hoc clamp to ±`max_gain_db`).

This guarantees `freq ∈ [fmin, fmax]` (already below Nyquist via `design_band`) and
`Q ∈ q_range` (positive) throughout training, so no divergence or NaN.

### 3. Differentiable response upgrade

`peaking_response_db_torch` must accept `freq_hz` and `q` as tensors (compute w0, alpha, cos
with `torch` ops) so gradients flow into them. Float inputs must still work (the gains-only path
passes floats). Keep the eps-guarded `log10`.

### 4. Optimizer

A single Adam with **separate parameter groups / learning rates**: gains at `lr` (e.g. 0.5),
centre and Q latents at a smaller lr (e.g. ~0.05) — a small centre nudge changes the response a
lot, so it needs a gentler step. Only add the freq/Q latents to the optimizer when their flags
are on. Deterministic (derived init, no RNG).

### 5. Enhancements (same theme, portfolio value)

- **Convergence curves:** plot the loss history (gains-only vs full) decreasing — visual proof
  the optimizer is actually learning.
- **Filter-movement plot:** show each centre's initial → learned position (plus final gains), so
  a reader can see *what* the model learned.
- **Ablation:** compare σ for "gains only" vs "gains + freq + Q" on the synthetic headline RIR
  (and, if cheap, across the real MIT rooms via `evaluate_rir`).
- **Regularization:** none by default. If learned centres collapse together, add a small
  spread/repulsion penalty then — not before (YAGNI).

### 6. Tests (TDD)

- **Backward compatibility:** with `learn_centers=False, learn_q=False`, returned filters have
  the log-spaced centres and `q == 4.0` (behaviour unchanged).
- **Bounded frequency:** with `learn_centers=True`, every returned `freq_hz ∈ [fmin, fmax]`.
- **Bounded Q:** with `learn_q=True`, every returned `q ∈ q_range`.
- **Determinism:** two runs with the flags on are identical.
- **No NaN at low sample rate:** learnable run on a 16 kHz input stays finite.
- **Improvement:** on a synthetic colored RIR, full-DDSP σ ≤ gains-only σ (small tolerance).
- **History:** `return_history=True` returns a loss list whose length == `iters` and is finite.

### 7. Notebook + README

- New ablation section: convergence curves, filter-movement plot, gains-only vs full σ.
- Update the conclusion table and README DDSP description with the ablation result.

## Success criteria

Full-DDSP reaches σ ≤ gains-only on the synthetic headline, runs are deterministic and
NaN-free, and the full test suite stays green.

## Non-goals (YAGNI)

- No new filter types (shelves), no regularization unless collapse is observed, no full-rate
  real RIRs, no app/product changes. This work is scoped to the DDSP optimizer's learnable
  parameters and the ablation that demonstrates them.

## Risks

- **Non-convexity:** learning freq/Q may occasionally do slightly worse than gains-only on some
  inputs; the bounded init mitigates this and the ablation reports results honestly.
- **Stability:** mitigated by bounded reparametrization + smaller lr for freq/Q.
