"""Tidy (long-form) DataFrame for the app's Altair response chart.

Pure: numpy + pandas only, no Streamlit. The app builds an Altair line chart
from the frame returned here. Keeping only in-band rows also drops the 0 Hz rfft
bin so the log x-axis is safe.
"""
import numpy as np
import pandas as pd


def response_dataframe(freqs, series, fmin=20.0, fmax=20000.0):
    """Long-form frame for an Altair line chart.

    ``series`` maps a label -> a magnitude-dB array aligned with ``freqs``.
    Returns columns ``freq_hz``, ``magnitude_db``, ``series`` with only the rows
    where ``fmin <= freq <= fmax`` (which also removes the 0 Hz bin).
    """
    freqs = np.asarray(freqs, dtype=float)
    mask = (freqs >= fmin) & (freqs <= fmax)
    f_in = freqs[mask]
    frames = []
    for label, values in series.items():
        values = np.asarray(values, dtype=float)
        frames.append(pd.DataFrame({
            "freq_hz": f_in,
            "magnitude_db": values[mask],
            "series": label,
        }))
    if not frames:
        return pd.DataFrame(columns=["freq_hz", "magnitude_db", "series"])
    return pd.concat(frames, ignore_index=True)
