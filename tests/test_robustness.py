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
