"""Per-RIR flatness evaluation shared by the multi-room and multi-seed studies.

Wraps the standard chain (frequency response -> correct -> smoothed sigma) so
both notebook sections compute sigma the same way instead of duplicating it.
"""
from src.analysis import frequency_response
from src.pipeline import correct, design_band, smoothed_sigma
from src.targets import flat_target


def evaluate_rir(rir, sr, methods=("classic", "ddsp", "fir"), target_fn=flat_target,
                 n_filters=32, ddsp_iters=150):
    """Smoothed sigma before correction and after each method.

    Sigma is measured on the same band the methods design over (``design_band``),
    so low-sample-rate RIRs are not scored on a region the filters never touched.
    Returns a dict with key ``"before"`` plus one key per method in ``methods``.
    """
    freqs, resp = frequency_response(rir, sr)
    target = target_fn(freqs)
    fmin, fmax = design_band(sr)

    result = {"before": smoothed_sigma(resp, freqs, fmin, fmax)}
    for method in methods:
        corrected, _ = correct(resp, target, freqs, sr, method=method,
                               n_filters=n_filters, ddsp_iters=ddsp_iters)
        result[method] = smoothed_sigma(corrected, freqs, fmin, fmax)
    return result
