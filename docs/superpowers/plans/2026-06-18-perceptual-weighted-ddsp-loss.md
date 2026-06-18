# Perceptual-weighted DDSP loss Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-weight the DDSP optimiser's loss to match hearing (critical-band density + equal-loudness) and add a perceptual-σ scorecard, so a perceptually better correction is measurable.

**Architecture:** A new `src/perceptual.py` builds a per-frequency weight vector. `optimize_eq` gains an optional `weights=` array and minimises a weighted MSE (with weighted mean-centring); `weights=None` is the existing behaviour unchanged. `metrics.perceptual_sigma` is the weighted-std scorecard. A notebook ablation compares flat vs perceptual on both σ and perceptual-σ.

**Tech Stack:** Python, numpy, PyTorch (float64, deterministic, no RNG), pytest, matplotlib.

## Global Constraints

- Python executable: `D:\Bokey\Sound_Quality\.venv\Scripts\python.exe`; run tests with `-m pytest -q`.
- Determinism: no RNG in `optimize_eq`; zero/derived init only.
- Backward compatibility: `optimize_eq(weights=None)` must reproduce current behaviour bit-for-bit (existing 86 tests stay green).
- Repo content in English. Commit style: single-author, no `Co-Authored-By` trailer.
- Notebook is regenerated via `notebooks/_build_notebook.py` (never hand-edit the `.ipynb`).
- All prose (docs/README/notebook) must read as human-written.

---

### Task 1: Perceptual weight vector (`src/perceptual.py`)

**Files:**
- Create: `src/perceptual.py`
- Test: `tests/test_perceptual.py`

**Interfaces:**
- Produces: `perceptual_weights(freqs_hz, use_density=True, use_loudness=True) -> np.ndarray` (same length as `freqs_hz`, all positive, mean ≈ 1 over the passed frequencies). Helpers `_erb(f)`, `_equal_loudness_weight(freqs_hz)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_perceptual.py
import numpy as np
import pytest

from src.perceptual import perceptual_weights


def test_weights_positive_and_mean_normalised():
    freqs = np.linspace(0.0, 24000.0, 2049)
    w = perceptual_weights(freqs)
    assert np.all(w > 0)
    assert w.mean() == pytest.approx(1.0)


def test_density_upweights_low_frequencies():
    f = np.array([50.0, 5000.0])
    w = perceptual_weights(f, use_loudness=False)
    assert w[0] > w[1]


def test_both_flags_off_is_flat():
    freqs = np.linspace(20.0, 20000.0, 512)
    w = perceptual_weights(freqs, use_density=False, use_loudness=False)
    assert np.allclose(w, 1.0)


def test_equal_loudness_peaks_in_mid():
    f = np.array([50.0, 3000.0, 12000.0])
    w = perceptual_weights(f, use_density=False)
    assert w[1] > w[0]
    assert w[1] > w[2]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `D:\Bokey\Sound_Quality\.venv\Scripts\python.exe -m pytest tests/test_perceptual.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.perceptual'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/perceptual.py
"""Perceptual weighting for the DDSP loss and the perceptual-sigma metric.

The DDSP loss and sigma are computed over linearly-spaced FFT bins, which are
dense in the treble and so over-weight high frequencies. These weights re-tilt
the objective toward how we hear: a critical-band density term (1/ERB) and an
ISO 226 40-phon equal-loudness term.
"""
import numpy as np

# ISO 226:2003 equal-loudness contour at 40 phon: frequency (Hz) -> SPL (dB).
_ISO226_40PHON = {
    20: 99.85, 25: 93.94, 31.5: 88.17, 40: 82.63, 50: 77.78, 63: 73.08,
    80: 68.48, 100: 64.37, 125: 60.59, 160: 56.70, 200: 53.41, 250: 50.40,
    315: 47.58, 400: 44.98, 500: 43.05, 630: 41.34, 800: 40.06, 1000: 40.01,
    1250: 41.82, 1600: 42.51, 2000: 39.23, 2500: 36.51, 3150: 35.61,
    4000: 36.65, 5000: 40.01, 6300: 45.83, 8000: 51.80, 10000: 54.28,
    12500: 51.49,
}


def _erb(f):
    """Glasberg-Moore ERB bandwidth (Hz) at frequency f (Hz)."""
    return 24.7 * (4.37 * np.asarray(f, dtype=np.float64) / 1000.0 + 1.0)


def _equal_loudness_weight(freqs_hz):
    """Hearing sensitivity from the ISO 226 40-phon contour.

    Higher where less SPL is needed to reach 40 phon (most sensitive ~3 kHz),
    lower in the bass and top treble. Interpolated in log-frequency, clamped to
    the tabulated range [20, 12500] Hz.
    """
    f = np.asarray(freqs_hz, dtype=np.float64)
    keys = sorted(_ISO226_40PHON)
    tab_f = np.array(keys, dtype=np.float64)
    tab_spl = np.array([_ISO226_40PHON[k] for k in keys], dtype=np.float64)
    fc = np.clip(f, tab_f[0], tab_f[-1])
    spl = np.interp(np.log(fc), np.log(tab_f), tab_spl)
    return 10.0 ** (-(spl - tab_spl.min()) / 20.0)


def perceptual_weights(freqs_hz, use_density=True, use_loudness=True):
    """Per-frequency perceptual weight, normalised to mean 1.

    Combines a critical-band density term (1/ERB, counters the treble-heavy
    linear FFT-bin spacing) and an ISO 226 40-phon equal-loudness term. With
    both flags False returns all-ones (the flat fallback).
    """
    f = np.asarray(freqs_hz, dtype=np.float64)
    w = np.ones_like(f)
    if use_density:
        w = w / _erb(f)
    if use_loudness:
        w = w * _equal_loudness_weight(f)
    return w / w.mean()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `D:\Bokey\Sound_Quality\.venv\Scripts\python.exe -m pytest tests/test_perceptual.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/perceptual.py tests/test_perceptual.py
git commit -m "feat: perceptual weight vector (ERB density + ISO 226 equal-loudness)"
```

---

### Task 2: `perceptual_sigma` metric (`src/metrics.py`)

**Files:**
- Modify: `src/metrics.py` (append a function; `band_mask` already defined there)
- Test: `tests/test_metrics.py` (append)

**Interfaces:**
- Consumes: `band_mask` (already in `metrics.py`).
- Produces: `perceptual_sigma(response_db, freqs_hz, weights, fmin=20.0, fmax=20000.0) -> float`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_metrics.py  (append)
def test_perceptual_sigma_equals_flatness_std_with_uniform_weights():
    import numpy as np
    from src.metrics import flatness_std_db, perceptual_sigma
    freqs = np.linspace(0.0, 24000.0, 2049)
    rng = np.random.default_rng(0)
    resp = rng.normal(size=freqs.shape)
    w = np.ones_like(freqs)
    assert perceptual_sigma(resp, freqs, w) == pytest.approx(flatness_std_db(resp, freqs))
```

(If `import pytest` is not already at the top of `tests/test_metrics.py`, add it.)

- [ ] **Step 2: Run test to verify it fails**

Run: `D:\Bokey\Sound_Quality\.venv\Scripts\python.exe -m pytest tests/test_metrics.py::test_perceptual_sigma_equals_flatness_std_with_uniform_weights -q`
Expected: FAIL with `ImportError: cannot import name 'perceptual_sigma'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/metrics.py  (append)
def perceptual_sigma(response_db, freqs_hz, weights, fmin=20.0, fmax=20000.0):
    """Weighted std of the response over the band (perceptual flatness).

    With uniform weights this equals ``flatness_std_db``. Pass a weight vector
    from ``perceptual.perceptual_weights`` to score perceptual flatness.
    """
    response_db = np.asarray(response_db, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    mask = band_mask(freqs_hz, fmin, fmax)
    x = response_db[mask]
    w = weights[mask]
    wmean = np.sum(w * x) / np.sum(w)
    return float(np.sqrt(np.sum(w * (x - wmean) ** 2) / np.sum(w)))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `D:\Bokey\Sound_Quality\.venv\Scripts\python.exe -m pytest tests/test_metrics.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/metrics.py tests/test_metrics.py
git commit -m "feat: perceptual_sigma (weighted-std flatness metric)"
```

---

### Task 3: Weighted loss in `optimize_eq` (`src/eq_ddsp.py`)

**Files:**
- Modify: `src/eq_ddsp.py` (signature + the loss block at lines ~180-182)
- Test: `tests/test_eq_ddsp.py` (append)

**Interfaces:**
- Consumes: `band_mask` (already imported in `eq_ddsp.py`).
- Produces: `optimize_eq(..., weights=None)`. When `weights` is an array of length `len(freqs_hz)`, the loss is the weighted MSE with weighted mean-centring. `None` ⇒ unchanged.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_eq_ddsp.py  (append)
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `D:\Bokey\Sound_Quality\.venv\Scripts\python.exe -m pytest tests/test_eq_ddsp.py -k "weights or perceptual" -q`
Expected: FAIL with `TypeError: optimize_eq() got an unexpected keyword argument 'weights'`

- [ ] **Step 3: Write minimal implementation**

In `src/eq_ddsp.py`, add `weights=None` to the signature (place it next to `return_history=False`):

```python
    return_history=False,
    weights=None,
):
```

After `freqs_hz = np.asarray(freqs_hz, dtype=np.float64)` (near the top of the body), add:

```python
    w_t = None
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != freqs_hz.shape:
            raise ValueError(
                f"weights length {weights.shape} must match freqs_hz {freqs_hz.shape}"
            )
        w_t = torch.as_tensor(weights, dtype=torch.float64)
```

Replace the loss block (currently lines ~180-182):

```python
        current = resp_t + eq
        residual = current - target_t
        residual = residual - residual[mask].mean()
        loss = (residual[mask] ** 2).mean()
```

with:

```python
        current = resp_t + eq
        residual = current - target_t
        if w_t is None:
            residual = residual - residual[mask].mean()
            loss = (residual[mask] ** 2).mean()
        else:
            wm = (w_t * residual)[mask].sum() / w_t[mask].sum()
            residual = residual - wm
            loss = (w_t * residual ** 2)[mask].sum() / w_t[mask].sum()
```

Also add a one-line mention to the `optimize_eq` docstring: `weights` (optional, per-frequency) tilts the loss; `None` keeps the flat MSE.

- [ ] **Step 4: Run tests to verify they pass**

Run: `D:\Bokey\Sound_Quality\.venv\Scripts\python.exe -m pytest tests/test_eq_ddsp.py -q`
Expected: PASS (existing DDSP tests + 3 new)

- [ ] **Step 5: Commit**

```bash
git add src/eq_ddsp.py tests/test_eq_ddsp.py
git commit -m "feat: optional perceptual weights in optimize_eq loss"
```

---

### Task 4: Headline claim — perceptual training lowers perceptual-σ

**Files:**
- Test: `tests/test_eq_ddsp.py` (append one integration test)

**Interfaces:**
- Consumes: `optimize_eq(weights=...)` (Task 3), `perceptual_weights` (Task 1), `perceptual_sigma` (Task 2), `apply_eq_db` (existing in `src.eq_classic`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eq_ddsp.py  (append)
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
```

- [ ] **Step 2: Run test to verify it fails (or errors before implementation of Tasks 1-3)**

Run: `D:\Bokey\Sound_Quality\.venv\Scripts\python.exe -m pytest tests/test_eq_ddsp.py::test_perceptual_training_lowers_perceptual_sigma -q`
Expected (after Tasks 1-3): PASS. If run earlier: import/`TypeError`.

- [ ] **Step 3: No new implementation** — this is an integration check over Tasks 1-3. If it fails, debug the weighting (do not weaken the assertion without cause).

- [ ] **Step 4: Run the full suite**

Run: `D:\Bokey\Sound_Quality\.venv\Scripts\python.exe -m pytest -q`
Expected: PASS (90 tests: 86 + 4 new across Tasks 1-4)

- [ ] **Step 5: Commit**

```bash
git add tests/test_eq_ddsp.py
git commit -m "test: perceptual training lowers perceptual-sigma on a colored room"
```

---

### Task 5: Notebook §4c ablation + README/explainer note

**Files:**
- Modify: `notebooks/_build_notebook.py` (add a §4c section after the §4b ablation)
- Modify: `README.md`, `docs/understanding.md` (and the local `docs/understanding_ko.md`)

**Interfaces:**
- Consumes: everything above, plus the notebook's existing `NF`, `ITERS`, `resp`, `freqs`, `sr`, `target`, `apply_eq_db`, `fractional_octave_smooth`.

- [ ] **Step 1: Add the §4c builder section**

Insert after the §4b ablation block (search for the string `## 4b` / the cell that ends with `assets/11_ddsp_ablation.png`). Add:

```python
md(
    "### 4c. Hearing-weighted objective (perceptual loss)\n"
    "\n"
    "The loss so far is a flat MSE over linearly-spaced FFT bins — which are dense in the treble, so it\n"
    "quietly over-weights high frequencies. Here we re-tilt it toward hearing: a critical-band density\n"
    "term (1/ERB) plus the ISO 226 40-phon equal-loudness curve. We score both the flat-MSE and the\n"
    "perceptual fit on plain σ *and* a perceptual-σ (weighted std), reported honestly."
)

code(
    "from src.perceptual import perceptual_weights\n"
    "from src.metrics import perceptual_sigma, flatness_std_db\n"
    "\n"
    "w = perceptual_weights(freqs)\n"
    "flat_eq = optimize_eq(resp, target, freqs, sr, n_filters=NF, iters=ITERS)\n"
    "perc_eq = optimize_eq(resp, target, freqs, sr, n_filters=NF, iters=ITERS, weights=w)\n"
    "flat_corr = fractional_octave_smooth(freqs, resp + apply_eq_db(flat_eq, freqs, sr))\n"
    "perc_corr = fractional_octave_smooth(freqs, resp + apply_eq_db(perc_eq, freqs, sr))\n"
    "\n"
    "rows = [('flat MSE', flat_corr), ('perceptual', perc_corr)]\n"
    "for name, c in rows:\n"
    "    print(f'{name:11s} plain sigma {flatness_std_db(c, freqs):.3f} | '\n"
    "          f'perceptual sigma {perceptual_sigma(c, freqs, w):.3f}')\n"
    "\n"
    "fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4))\n"
    "axL.semilogx(freqs, w, color='#7FB5B5', lw=1.8)\n"
    "axL.set(xlim=(20, 20000), title='Perceptual weight w(f): ERB density x equal-loudness',\n"
    "        xlabel='frequency [Hz]', ylabel='weight (mean 1)')\n"
    "axR.semilogx(freqs, flat_corr, color='#F6C28B', lw=1.6, label='flat MSE')\n"
    "axR.semilogx(freqs, perc_corr, color='#B5A7E6', lw=1.8, label='perceptual')\n"
    "axR.axhline(0, color='k', lw=0.8, ls='--', alpha=0.5)\n"
    "axR.legend(); axR.set(xlim=(20, 20000), title='Corrected response: flat vs hearing-weighted',\n"
    "                      xlabel='frequency [Hz]', ylabel='magnitude [dB]')\n"
    "plt.tight_layout(); plt.savefig('../assets/12_perceptual.png', dpi=110); plt.show()"
)
```

- [ ] **Step 2: Rebuild and execute the notebook**

Run:
```bash
D:\Bokey\Sound_Quality\.venv\Scripts\python.exe notebooks/_build_notebook.py
cd notebooks && ../.venv/Scripts/python.exe -m jupyter nbconvert --to notebook --execute --inplace room_correction.ipynb
```
Expected: writes the `.ipynb`; no error outputs; `assets/12_perceptual.png` created. Read the printed σ / perceptual-σ line and the image to confirm perceptual-σ drops for the perceptual fit.

- [ ] **Step 3: Update README + explainer with the measured numbers**

In `README.md`, after the DDSP ablation note, add a short human-readable paragraph: the flat MSE over linear bins over-weights treble; the hearing-weighted objective (ERB density + ISO 226 equal-loudness) lowers perceptual-σ from X to Y while plain σ stays comparable. Use the exact numbers printed by the notebook. Reference `assets/12_perceptual.png`. Mirror a 1-2 line note into `docs/understanding.md` §6b and the local `docs/understanding_ko.md`. Bump test counts to 90.

- [ ] **Step 4: Verify**

Run: `D:\Bokey\Sound_Quality\.venv\Scripts\python.exe -m pytest -q`
Expected: PASS (90).

- [ ] **Step 5: Commit**

```bash
git add notebooks/_build_notebook.py notebooks/room_correction.ipynb assets/12_perceptual.png README.md docs/understanding.md
git commit -m "feat: notebook 4c perceptual-loss ablation + README/explainer note"
```

---

## Self-Review

- **Spec coverage:** perceptual_weights (T1) ✓, perceptual_sigma (T2) ✓, optimize_eq weights + weighted centring + ValueError (T3) ✓, headline claim test (T4) ✓, notebook ablation + docs + pastel figure (T5) ✓, all 9 spec tests mapped (1-4→T1, 7→T2, 5/6/8→T3, 9→T4). Backward-compat (weights=None) ✓ T3 Step 3. Out-of-scope items not implemented ✓.
- **Placeholders:** none — every code/test step has complete code.
- **Type consistency:** `perceptual_weights`, `perceptual_sigma(response_db, freqs_hz, weights, ...)`, `optimize_eq(..., weights=None)`, `apply_eq_db`, `fractional_octave_smooth` used consistently across tasks.

## Notes for the implementer

- The DDSP test suite is slow (~minutes); the new tests use small `n_filters`/`iters` to stay quick. Task 5's full notebook execution is ~18 min — expect it.
- Pastel palette in §4c uses hex `#7FB5B5` / `#F6C28B` / `#B5A7E6` (teal / peach / lavender) — a preview of the app's pastel direction; the rest of the notebook keeps its current palette.
