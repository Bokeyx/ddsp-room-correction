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
