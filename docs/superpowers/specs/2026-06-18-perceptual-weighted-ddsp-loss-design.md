# Perceptual-weighted DDSP loss + perceptual-σ metric — design

- **Date:** 2026-06-18
- **Status:** approved (brainstorming)
- **Sub-project:** 1 of 3 in the "deepen + ship" push (this one; then a usable app with a pastel UI; then robustness/generalization).

## Motivation

The DDSP optimiser minimises `mean(residual² )` over the FFT bins inside the design band, and σ is the
std of the response over those same bins. Those bins are **linearly spaced** (an rfft over a constant
`Δf`), so half of them sit in the top octave. The loss therefore over-weights treble and under-weights
the bass/mid region where the ear actually lives. Hearing is roughly logarithmic in frequency and
non-uniform in sensitivity, so the current objective is perceptually tilted the wrong way.

This sub-project re-weights the loss to match hearing, and adds a matching perceptual scorecard so the
benefit is measurable (plain σ alone would hide it — a perceptually better correction can score the same
or slightly worse on unweighted σ).

## Goals

- A perceptual weight vector `w(f)` combining **critical-band density** (1/ERB) and **equal-loudness**
  sensitivity (ISO 226, 40-phon contour), each toggleable for ablation.
- `optimize_eq` accepts an optional `weights` array and minimises the **weighted** mean-squared residual
  (with weighted mean-centring so it stays gain-invariant). `weights=None` reproduces today's behaviour
  bit-for-bit.
- A `perceptual_sigma` metric (weighted std) as a second scorecard alongside the existing σ.
- A notebook ablation that compares flat-MSE vs perceptual training on **both** plain σ and perceptual-σ,
  reported honestly.

## Non-goals (parked for later sub-projects)

- The usable app and its **pastel UI** (sub-project 2). A pastel palette may be applied to the new
  ablation figure only; the notebook's global palette is not changed here.
- Multi-position robust correction, L1 sparsity, phase/group-delay (later core-deepening candidates).

## Components

### `src/perceptual.py` (new)

```
perceptual_weights(freqs_hz, use_density=True, use_loudness=True) -> np.ndarray
```
Returns a positive weight per frequency, **normalised so its mean over the band is 1.0** (keeps the
weighted loss on the same scale as the flat loss). With both flags False it returns all-ones (== flat).

- **Density term** (Glasberg–Moore ERB): `ERB(f) = 24.7 * (4.37 * f/1000 + 1)`, `w_density(f) = 1/ERB(f)`.
  Larger at low f (narrow critical bands), so it counteracts the treble-heavy linear-bin density.
- **Equal-loudness term** (ISO 226, 40-phon): interpolate the 40-phon SPL contour `L40(f)` in
  log-frequency from the table below; sensitivity is higher where less SPL is needed, so
  `w_loud(f) = 10**(-(L40(f) - min(L40)) / 20)` — peaks near ~3.15 kHz, rolls off in bass and treble.
  Outside the tabulated range [20, 12500] Hz, clamp to the nearest endpoint.
- **Combine:** multiply the enabled terms, then normalise to mean 1 over the design band.

ISO 226:2003 40-phon contour table embedded in the module (Hz : dB SPL):

```
20:99.85 25:93.94 31.5:88.17 40:82.63 50:77.78 63:73.08 80:68.48 100:64.37
125:60.59 160:56.70 200:53.41 250:50.40 315:47.58 400:44.98 500:43.05 630:41.34
800:40.06 1000:40.01 1250:41.82 1600:42.51 2000:39.23 2500:36.51 3150:35.61
4000:36.65 5000:40.01 6300:45.83 8000:51.80 10000:54.28 12500:51.49
```

### `src/eq_ddsp.py` — `optimize_eq(..., weights=None)`

- New keyword `weights=None`. When `None`, the loss is exactly today's `(residual[mask]**2).mean()` and
  the existing behaviour is unchanged (no new tensors, no scale change).
- When an array is given (length == `len(freqs_hz)`), build a float64 tensor `w`, restricted to the band
  mask, and compute:
  - weighted mean: `wm = (w * residual)[mask].sum() / w[mask].sum()`
  - centre: `residual = residual - wm`  (weighted, gain-invariant)
  - loss: `(w * residual**2)[mask].sum() / w[mask].sum()`
- Determinism is preserved (no RNG). Works with `learn_centers` / `learn_q` unchanged (orthogonal).
- Validation: if `weights` length mismatches `freqs_hz`, raise `ValueError`.

### `src/metrics.py` — `perceptual_sigma`

```
perceptual_sigma(response_db, freqs_hz, weights, fmin=20.0, fmax=20000.0) -> float
```
Weighted std over the band: `wmean = Σwx/Σw`, `sqrt(Σw(x-wmean)²/Σw)`. With uniform weights it equals
`flatness_std_db`. (The notebook smooths first, as the existing σ path does.)

## Data flow

```
freqs ── perceptual_weights(freqs) ──► w
resp,target,w ──► optimize_eq(resp, target, freqs, sr, weights=w) ──► filters
corrected ──► smoothed_sigma(...)           # plain, unweighted (unchanged)
          └─► perceptual_sigma(..., w)      # new scorecard
```

The pipeline/notebook constructs `w`; `optimize_eq` stays low-level (takes the array, not a flag).

## Evaluation / notebook §4c

On the headline synthetic room, train two filter banks at the same NF/iters:
1. flat-MSE (`weights=None`), 2. perceptual (`weights=perceptual_weights(freqs)`).
Score **both** on plain `smoothed_sigma` and `perceptual_sigma`. Expected, stated honestly:
perceptual training lowers **perceptual-σ** while plain σ stays comparable (it may tick up slightly — by
design, since effort moves off inaudible treble ripple). Add a small filter-allocation / weight-curve
figure (pastel palette allowed here) showing where the weighting pushes correction effort.

## Testing (TDD)

1. `perceptual_weights` is all positive and **normalised to mean 1** over the band.
2. Density-only weights **upweight low frequencies**: assert `w(50 Hz) > w(5 kHz)`.
3. `use_density=False, use_loudness=False` ⇒ all-ones (flat fallback).
4. Equal-loudness weight **peaks in the 2–4 kHz region** and is lower at 50 Hz and 12 kHz.
5. `optimize_eq(weights=ones)` matches `optimize_eq(weights=None)` (uniform == flat; refactor safety net).
6. `optimize_eq` raises `ValueError` on a length-mismatched `weights`.
7. `perceptual_sigma` with uniform weights equals `flatness_std_db`.
8. Determinism: perceptual run is bit-identical across two calls.
9. **Headline claim:** on a coloured room, perceptual training yields a strictly lower `perceptual_sigma`
   than flat-MSE training, while its plain `smoothed_sigma` stays within 1.5× of the flat-MSE plain σ
   (comparable, not a collapse).

## Success criteria

- All new tests green; existing 86 still green (default path unchanged).
- Notebook §4c shows perceptual training winning on perceptual-σ with plain σ comparable, reported honestly.
- README/explainer gain a short, human-readable "hearing-weighted objective" note with the measured numbers.
