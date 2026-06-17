"""Unified correction pipeline so the three methods share one interface.

Used by the notebook and the Streamlit demo: pick a method, get back the
corrected magnitude response plus the correction object (filters or FIR taps).
"""
from src.analysis import fractional_octave_smooth
from src.eq_classic import apply_eq_db, design_classic_eq
from src.eq_ddsp import optimize_eq
from src.fir import design_fir_correction, fir_response_db
from src.metrics import flatness_std_db


# Keep the EQ design band safely below Nyquist. A peaking biquad centred at
# Nyquist evaluates to 0/0 there (NaN), and a centre even a bin or two below it
# gives a near-degenerate (b/a -> 0/0) response, so we leave a ~10% margin rather
# than only dropping the exact top bin. Placing filters above Nyquist is
# meaningless anyway. This only bites low-sample-rate inputs (e.g. 16 kHz RIRs);
# for 48 kHz the 20 kHz default already sits below 0.45*sr.
_DESIGN_FMAX_FRACTION = 0.45


def design_band(sr, fmin=20.0, fmax=20000.0):
    """The frequency band the correctors design (and are scored) on.

    ``fmax`` is capped below Nyquist (see ``_DESIGN_FMAX_FRACTION``). Sharing one
    band between correction and the sigma metric keeps the score honest: we never
    grade a region the filters were not allowed to touch.
    """
    return fmin, min(fmax, _DESIGN_FMAX_FRACTION * sr)


def smoothed_sigma(response_db, freqs_hz, fmin=20.0, fmax=20000.0):
    """Fair flatness metric: sigma of the 1/3-octave-smoothed response.

    Raw FFT bin noise is uncorrectable by any peaking EQ, so flatness is
    measured on the smoothed response (the convention used across the project).
    ``fmin``/``fmax`` restrict the band; pass the ``design_band`` to match the
    correction band on low-sample-rate inputs.
    """
    smoothed = fractional_octave_smooth(freqs_hz, response_db)
    return flatness_std_db(smoothed, freqs_hz, fmin, fmax)


def correct(response_db, target_db, freqs_hz, sr, method="ddsp", n_filters=32,
            ddsp_iters=150, fmin=20.0, fmax=20000.0):
    """Correct ``response_db`` toward ``target_db`` with the chosen method.

    All three methods design over the same ``design_band(sr, fmin, fmax)`` so the
    comparison stays fair and numerically safe near Nyquist. Returns
    ``(corrected_db, correction)`` where ``correction`` is a list of PeakingFilter
    (classic/ddsp) or an ndarray of FIR taps (fir).
    """
    fmin, fmax = design_band(sr, fmin, fmax)
    if method == "classic":
        eq = design_classic_eq(response_db, target_db, freqs_hz, sr,
                               n_filters=n_filters, fmin=fmin, fmax=fmax)
        return response_db + apply_eq_db(eq, freqs_hz, sr), eq
    if method == "ddsp":
        eq = optimize_eq(response_db, target_db, freqs_hz, sr, n_filters=n_filters,
                        iters=ddsp_iters, fmin=fmin, fmax=fmax)
        return response_db + apply_eq_db(eq, freqs_hz, sr), eq
    if method == "fir":
        taps = design_fir_correction(response_db, target_db, freqs_hz, sr,
                                     fmin=fmin, fmax=fmax)
        return response_db + fir_response_db(taps, freqs_hz, sr), taps
    raise ValueError(f"unknown method: {method!r}")
