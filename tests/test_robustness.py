import pytest

from src.robustness import win_counts, paired_sigma_test


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
