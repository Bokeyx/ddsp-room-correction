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
