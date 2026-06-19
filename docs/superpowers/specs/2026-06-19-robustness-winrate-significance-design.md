# Robustness validation — win-rate + statistical significance — design

- **Date:** 2026-06-19
- **Status:** approved (brainstorming)
- **Sub-project:** 3 of 3 in the "deepen + ship" push (perceptual loss done; pastel app done; this one).

## Motivation

The project already reports mean ± std σ over real rooms (notebook §8, MIT 20 rooms), full-rate rooms
(§8b, Aachen AIR), and multi-seed synthetic RIRs (§9). A mean can hide that a method loses on individual
rooms, and a difference in means says nothing about whether it is statistically real or just noise. This
sub-project adds the two checks that turn "DDSP is lower on average" into a defensible claim:

- **Win-rate** — per-room head-to-head: in how many rooms is each method the flattest?
- **Statistical significance** — a paired Wilcoxon signed-rank test on the per-room σ of DDSP vs the
  classic baseline, reported with a p-value and the median paired difference.

## Goals

- A pure module `src/robustness.py` with two functions over the per-RIR σ dicts that `evaluate_rir`
  already returns (no new data, no Streamlit, no IO):
  - `win_counts(rows, methods)` — counts, per method, the rooms where it has the minimum σ.
  - `paired_sigma_test(rows, method_a, method_b)` — Wilcoxon signed-rank on the paired differences.
- A notebook section (§10) that reuses the §8 real-room results (`real_rows`, 20 rooms) to show a
  pastel win-rate bar chart and the DDSP-vs-classic p-value, reported honestly (rooms DDSP loses are
  shown, not hidden).
- README / understanding.md lines summarizing the win-rate and significance result.

## Module: `src/robustness.py`

Input `rows` is a `list[dict]`, each dict mapping method name -> σ (the `evaluate_rir` output), e.g.
`{"before": 4.2, "classic": 1.3, "ddsp": 0.9, "fir": 1.3}`.

```python
def win_counts(rows, methods):
    """Count, per method in `methods`, the rows where that method's sigma is the
    strict minimum among `methods`. A row with a tie for the minimum credits no
    method (avoids double-counting). "before" is never a method here. Returns a
    dict {method: int} with an entry for every method in `methods` (0 if none)."""

def paired_sigma_test(rows, method_a, method_b):
    """Paired Wilcoxon signed-rank test on the per-row differences
    (sigma_a - sigma_b). Returns {"n": int, "median_diff": float,
    "statistic": float, "pvalue": float}. `median_diff` < 0 means method_a is
    flatter on average. Requires at least one row; raises ValueError on empty
    input. If every paired difference is zero, returns pvalue 1.0 (no effect)
    rather than letting scipy raise."""
```

- `win_counts`: tie handling = strict minimum, ties credit nobody. Documented so the bars always sum to
  `<= len(rows)`.
- `paired_sigma_test`: uses `scipy.stats.wilcoxon` (scipy is already a dependency). The all-zero-diff
  guard avoids scipy's "zero_method" edge raising on degenerate input.

## Notebook section §10

- Reuses `real_rows` from §8 (no new heavy computation). Build `rows = [d for _, d in real_rows]`.
- `win_counts(rows, ["classic", "ddsp", "fir"])` -> pastel bar chart (classic `#7FB5B5`,
  ddsp `#F6C28B`, fir `#B5A7E6`), counts annotated, saved to `assets/14_winrate.png`.
- `paired_sigma_test(rows, "ddsp", "classic")` -> print n, median_diff, p-value; one honest sentence in
  the markdown (e.g. "DDSP is flattest in X/20 rooms; the σ improvement over classic is significant,
  Wilcoxon p = …").

## Testing (`tests/test_robustness.py`)

- `win_counts`: three known rows (ddsp min in 2, classic min in 1) -> `{classic:1, ddsp:2, fir:0}`;
  a `"before"` key present in the rows is ignored.
- `win_counts` tie: a row where two methods share the minimum credits neither.
- `paired_sigma_test`: rows where `a < b` in every row -> `n` correct, `median_diff < 0`, `pvalue < 0.05`.
- `paired_sigma_test` all-equal: every `a == b` -> `pvalue == 1.0`, `median_diff == 0`.
- Edge: empty `rows` -> `ValueError`.

Suite: 106 -> ~110.

## Non-goals (parked)

- No multi-position RIRs (different mic spots in one room) — needs separate dataset wrangling; parked.
- No new datasets; reuse the MIT real rooms and the existing synthetic/AIR loaders.
- No extra statistics (bootstrap CIs, effect-size families) — one paired test is enough here.

## Success criteria

- `src/robustness.py` is pure and passes the tests above.
- Notebook §10 shows the win-rate chart and the DDSP-vs-classic p-value on the 20 real rooms, with any
  losing rooms reported rather than hidden.
