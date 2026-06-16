"""Unified correction pipeline so the three methods share one interface.

Used by the notebook and the Streamlit demo: pick a method, get back the
corrected magnitude response plus the correction object (filters or FIR taps).
"""
from src.analysis import fractional_octave_smooth
from src.eq_classic import apply_eq_db, design_classic_eq
from src.eq_ddsp import optimize_eq
from src.fir import design_fir_correction, fir_response_db
from src.metrics import flatness_std_db


def smoothed_sigma(response_db, freqs_hz):
    """Fair flatness metric: sigma of the 1/3-octave-smoothed response.

    Raw FFT bin noise is uncorrectable by any peaking EQ, so flatness is
    measured on the smoothed response (the convention used across the project).
    """
    return flatness_std_db(fractional_octave_smooth(freqs_hz, response_db), freqs_hz)


def correct(response_db, target_db, freqs_hz, sr, method="ddsp", n_filters=32, ddsp_iters=150):
    """Correct ``response_db`` toward ``target_db`` with the chosen method.

    Returns ``(corrected_db, correction)`` where ``correction`` is a list of
    PeakingFilter (classic/ddsp) or an ndarray of FIR taps (fir).
    """
    if method == "classic":
        eq = design_classic_eq(response_db, target_db, freqs_hz, sr, n_filters=n_filters)
        return response_db + apply_eq_db(eq, freqs_hz, sr), eq
    if method == "ddsp":
        eq = optimize_eq(response_db, target_db, freqs_hz, sr, n_filters=n_filters, iters=ddsp_iters)
        return response_db + apply_eq_db(eq, freqs_hz, sr), eq
    if method == "fir":
        taps = design_fir_correction(response_db, target_db, freqs_hz, sr)
        return response_db + fir_response_db(taps, freqs_hz, sr), taps
    raise ValueError(f"unknown method: {method!r}")
