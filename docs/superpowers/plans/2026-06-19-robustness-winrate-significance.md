# Robustness Win-rate + Significance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add per-room win-rate and a paired Wilcoxon significance test of DDSP vs classic, surfaced in a notebook section over the 20 real MIT rooms.

**Architecture:** A new pure module `src/robustness.py` with two functions over the per-RIR σ dicts that `evaluate_rir` already returns. A notebook section (§10) reuses the section-8 `real_rows` to plot win-rate and print the DDSP-vs-classic p-value. No new data, no Streamlit, no IO in the module.

**Tech Stack:** Python 3, numpy, scipy (`scipy.stats.wilcoxon`), matplotlib, pytest. Python executable: `D:\Bokey\Sound_Quality\.venv\Scripts\python.exe`. Tests: `python -m pytest -q`.

## Global Constraints

- Single-author commits. NEVER add a `Co-Authored-By` trailer.
- All repo content in English.
- `src/robustness.py` is PURE: no file IO, no Streamlit, no matplotlib import. Returns plain dicts.
- Input `rows` is a `list[dict]`, each dict mapping method name -> σ (float), e.g.
  `{"before": 4.2, "classic": 1.3, "ddsp": 0.9, "fir": 1.3}` (the `evaluate_rir` output).
- `win_counts` tie rule: a row credits a method only if its σ is the **strict** minimum among `methods`
  (ties credit nobody), so the counts sum to `<= len(rows)`.
- `paired_sigma_test`: `median_diff < 0` means `method_a` is flatter; empty `rows` raises `ValueError`;
  all-zero differences return `pvalue == 1.0` (do not let scipy raise).
- Notebook is regenerated via `python notebooks/_build_notebook.py`; never hand-edit the `.ipynb`.
- Pastel palette (shared): classic `#7FB5B5`, ddsp `#F6C28B`, fir `#B5A7E6`.

---

### Task 1: `win_counts` — per-room flattest-method tally

**Files:**
- Create: `src/robustness.py`
- Test: `tests/test_robustness.py`

**Interfaces:**
- Consumes: `rows: list[dict]`, `methods: list[str]`.
- Produces: `win_counts(rows, methods) -> dict[str, int]` (one entry per method in `methods`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_robustness.py
from src.robustness import win_counts


def test_win_counts_strict_minimum_per_row():
    rows = [
        {"before": 4.0, "classic": 1.2, "ddsp": 0.9, "fir": 1.3},  # ddsp
        {"before": 3.0, "classic": 1.1, "ddsp": 0.8, "fir": 1.0},  # ddsp
        {"before": 5.0, "classic": 0.7, "ddsp": 0.9, "fir": 1.1},  # classic
    ]
    assert win_counts(rows, ["classic", "ddsp", "fir"]) == {"classic": 1, "ddsp": 2, "fir": 0}


def test_win_counts_ignores_before_and_unlisted_methods():
    rows = [{"before": 0.1, "classic": 1.0, "ddsp": 2.0, "fir": 3.0}]
    # "before" is small but is not in `methods`, so classic still wins.
    assert win_counts(rows, ["classic", "ddsp", "fir"]) == {"classic": 1, "ddsp": 0, "fir": 0}


def test_win_counts_tie_credits_nobody():
    rows = [{"classic": 1.0, "ddsp": 1.0, "fir": 2.0}]
    assert win_counts(rows, ["classic", "ddsp", "fir"]) == {"classic": 0, "ddsp": 0, "fir": 0}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `D:\Bokey\Sound_Quality\.venv\Scripts\python.exe -m pytest tests/test_robustness.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.robustness'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/robustness.py
"""Robustness statistics over per-RIR flatness results.

Pure functions over the list of sigma dicts that ``evaluation.evaluate_rir``
returns (one dict per room, mapping method name -> sigma). No IO, no plotting --
the notebook does the plotting; these just compute the numbers, so they are
unit tested directly.
"""
import numpy as np
from scipy.stats import wilcoxon


def win_counts(rows, methods):
    """Count, per method in ``methods``, the rooms where it has the strict
    minimum sigma. A tie for the minimum credits nobody (counts sum to
    <= len(rows)). Methods missing from a given row are skipped for that row.
    Returns a dict with an entry for every method in ``methods``.
    """
    counts = {m: 0 for m in methods}
    for row in rows:
        present = [(m, row[m]) for m in methods if m in row]
        if not present:
            continue
        min_sigma = min(v for _, v in present)
        winners = [m for m, v in present if v == min_sigma]
        if len(winners) == 1:
            counts[winners[0]] += 1
    return counts
```

- [ ] **Step 4: Run test to verify it passes**

Run: `D:\Bokey\Sound_Quality\.venv\Scripts\python.exe -m pytest tests/test_robustness.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/robustness.py tests/test_robustness.py
git commit -m "feat: win_counts (per-room flattest-method tally)"
```

---

### Task 2: `paired_sigma_test` — paired Wilcoxon of two methods

**Files:**
- Modify: `src/robustness.py`
- Test: `tests/test_robustness.py`

**Interfaces:**
- Consumes: `rows: list[dict]`, `method_a: str`, `method_b: str`.
- Produces: `paired_sigma_test(rows, method_a, method_b) -> dict` with keys `n` (int),
  `median_diff` (float), `statistic` (float), `pvalue` (float).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_robustness.py
import pytest
from src.robustness import paired_sigma_test  # add to existing imports


def test_paired_test_a_consistently_lower_is_significant():
    rows = [{"ddsp": 0.9, "classic": 1.3} for _ in range(8)]
    res = paired_sigma_test(rows, "ddsp", "classic")
    assert res["n"] == 8
    assert res["median_diff"] < 0          # ddsp flatter
    assert res["pvalue"] < 0.05


def test_paired_test_all_equal_returns_p_one():
    rows = [{"ddsp": 1.0, "classic": 1.0} for _ in range(5)]
    res = paired_sigma_test(rows, "ddsp", "classic")
    assert res["pvalue"] == 1.0
    assert res["median_diff"] == 0.0


def test_paired_test_empty_raises():
    with pytest.raises(ValueError):
        paired_sigma_test([], "ddsp", "classic")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `D:\Bokey\Sound_Quality\.venv\Scripts\python.exe -m pytest tests/test_robustness.py -k paired -q`
Expected: FAIL — `ImportError: cannot import name 'paired_sigma_test'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/robustness.py
def paired_sigma_test(rows, method_a, method_b):
    """Paired Wilcoxon signed-rank test on the per-row differences
    (sigma_a - sigma_b). ``median_diff`` < 0 means ``method_a`` is flatter on
    average. Raises ValueError on empty ``rows``. If every paired difference is
    zero, returns pvalue 1.0 (scipy would otherwise raise on degenerate input).
    """
    if not rows:
        raise ValueError("rows is empty; need at least one paired observation")
    a = np.array([row[method_a] for row in rows], dtype=float)
    b = np.array([row[method_b] for row in rows], dtype=float)
    diff = a - b
    median_diff = float(np.median(diff))
    if np.allclose(diff, 0.0):
        return {"n": len(rows), "median_diff": median_diff, "statistic": 0.0, "pvalue": 1.0}
    stat, pvalue = wilcoxon(a, b)
    return {"n": len(rows), "median_diff": median_diff,
            "statistic": float(stat), "pvalue": float(pvalue)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `D:\Bokey\Sound_Quality\.venv\Scripts\python.exe -m pytest tests/test_robustness.py -q`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/robustness.py tests/test_robustness.py
git commit -m "feat: paired_sigma_test (paired Wilcoxon of two methods' sigma)"
```

---

### Task 3: Notebook §10 + README/docs

**Files:**
- Modify: `notebooks/_build_notebook.py` (insert before the `## Conclusion` md call, ~line 573)
- Modify: `README.md`
- Modify: `docs/understanding.md`

**Interfaces:**
- Consumes: `win_counts`, `paired_sigma_test` (Tasks 1-2); the notebook's existing `real_rows`
  (list of `(label, sigma_dict)` from section 8), `np`, `plt`, and the `md`/`code` cell helpers.
- Produces: notebook section §10 and the asset `assets/14_winrate.png`.

- [ ] **Step 1: Add the §10 markdown + code cells to the builder**

In `notebooks/_build_notebook.py`, immediately before the `md(\n    "## Conclusion\n"` call, insert:

```python
md(
    "## 10. Robustness: win-rate and statistical significance\n"
    "\n"
    "Mean sigma can hide rooms where a method loses, and a gap in means may be noise. Two stronger\n"
    "checks on the 20 real MIT rooms from section 8: a per-room **win-rate** (who is flattest, room by\n"
    "room) and a **paired Wilcoxon signed-rank test** of DDSP vs the classic baseline."
)

code(
    "from src.robustness import win_counts, paired_sigma_test\n"
    "\n"
    "rows = [d for _, d in real_rows]\n"
    "if not rows:\n"
    "    print('No real rooms loaded; run scripts/download_mit_rir.py to see this section.')\n"
    "else:\n"
    "    wr_methods = ['classic', 'ddsp', 'fir']\n"
    "    wins = win_counts(rows, wr_methods)\n"
    "    wr_colors = ['#7FB5B5', '#F6C28B', '#B5A7E6']\n"
    "    bars = plt.bar(wr_methods, [wins[m] for m in wr_methods], color=wr_colors)\n"
    "    for b, m in zip(bars, wr_methods):\n"
    "        plt.text(b.get_x()+b.get_width()/2, wins[m], str(wins[m]), ha='center', va='bottom')\n"
    "    plt.title(f'Flattest-method win-rate over {len(rows)} real rooms')\n"
    "    plt.ylabel('rooms won (lowest sigma)')\n"
    "    plt.tight_layout(); plt.savefig('../assets/14_winrate.png', dpi=110); plt.show()\n"
    "    print('win counts:', wins)\n"
    "    res = paired_sigma_test(rows, 'ddsp', 'classic')\n"
    "    print(f\"DDSP vs classic: n={res['n']}, median diff={res['median_diff']:.3f} dB, \"\n"
    "          f\"Wilcoxon p={res['pvalue']:.4g}\")"
)
```

- [ ] **Step 2: Regenerate the notebook**

Run: `cd D:\Bokey\Sound_Quality && .venv\Scripts\python.exe notebooks/_build_notebook.py`
Expected: prints `wrote .../room_correction.ipynb with <N> cells` (N is two larger than before).

- [ ] **Step 3: Execute the notebook to produce real numbers + the asset**

Run (background, ~18 min — the DDSP rooms are slow):
```bash
cd D:\Bokey\Sound_Quality && .venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 --ExecutePreprocessor.startup_timeout=180 notebooks/room_correction.ipynb
```
Expected: completes without error; `assets/14_winrate.png` is created; the §10 cell output prints
`win counts: {...}` and `DDSP vs classic: n=20, median diff=... dB, Wilcoxon p=...`.

Record the printed `win counts` and the `p` value — they are used verbatim in Step 4. (If MIT data is
not present locally the section prints the skip message; in that case run
`python scripts/download_mit_rir.py` first, then re-run this step.)

- [ ] **Step 4: Add the result to README and understanding.md**

In `README.md`, in the `## Evaluation` section, add one line using the numbers printed in Step 3 (fill
`<ddsp_wins>`, `<total>`, `<p>` from that output):

```markdown
- **Win-rate & significance (20 real rooms):** DDSP is the flattest in <ddsp_wins>/<total> rooms; its σ
  improvement over the classic baseline is statistically significant (paired Wilcoxon, p = <p>).

![win-rate](assets/14_winrate.png)
```

In `README.md` `## Roadmap`, add:

```markdown
- [x] **M5d** robustness: per-room win-rate + paired-Wilcoxon significance over the 20 real rooms
```

In `docs/understanding.md`, in the `## 6b.` list, add one bullet (same numbers):

```markdown
- **Not cherry-picked.** Beyond averages, DDSP is the flattest in <ddsp_wins> of <total> real rooms, and
  its edge over the classic EQ is statistically significant (paired Wilcoxon signed-rank, p = <p>) — the
  improvement is real, not a lucky room.
```

- [ ] **Step 5: Run the full suite**

Run: `cd D:\Bokey\Sound_Quality && .venv\Scripts\python.exe -m pytest -q`
Expected: PASS (~110 passed — 6 new robustness tests on top of 106; the notebook is not run by pytest).

- [ ] **Step 6: Commit**

```bash
git add notebooks/_build_notebook.py notebooks/room_correction.ipynb assets/14_winrate.png README.md docs/understanding.md
git commit -m "feat: notebook section 10 robustness win-rate + significance (+docs)"
```

---

## Self-Review notes

- **Spec coverage:** `win_counts` (T1), `paired_sigma_test` (T2), notebook §10 reusing `real_rows` with
  pastel win-rate chart + DDSP-vs-classic p-value (T3), README/understanding lines (T3). Non-goals
  (multi-position, new datasets, extra stats) are respected — none added.
- **Type consistency:** `rows: list[dict]` and the return dict keys (`n`, `median_diff`, `statistic`,
  `pvalue`) are identical across spec, tests, impl, and the notebook's `res['...']` access. `win_counts`
  returns `{method: int}` used the same way in the notebook bar chart.
- **Placeholder scan:** the only fill-ins are measured numbers from the deterministic Step-3 run, with
  the exact source and target sentence given — not deferred design.
- **Empty-data path:** §10 guards `if not rows`, so the notebook still builds/executes when MIT data is
  absent (prints a skip message instead of raising in `paired_sigma_test`).
