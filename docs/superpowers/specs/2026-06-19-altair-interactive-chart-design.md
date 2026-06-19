# App chart: matplotlib → Altair interactive — design

- **Date:** 2026-06-19
- **Status:** approved (brainstorming)
- **Scope:** the Streamlit app's single response plot. Notebook charts are untouched.

## Motivation

The app's before/after frequency-response plot is a static matplotlib PNG embedded in an otherwise clean
pastel web UI — it looks out of place and is not interactive. Streamlit renders Altair charts natively:
they are interactive (hover read-out of Hz/dB, pan/zoom), match the pastel palette, and draw text with
browser fonts — so the Korean labels render correctly with no bundled font at all. Switching this one
chart fixes the "ugly graph" problem and makes the recently added matplotlib Korean-font workaround
unnecessary for the app.

## Goals

- A pure `src/charts.py` that turns the response arrays into a tidy DataFrame for Altair.
- The app draws the before/after response as an Altair line chart: log-frequency x-axis (20–20 kHz),
  magnitude-dB y-axis, one coloured line per series (pastel), the target as a dashed line, and a tooltip.
- Remove the now-dead matplotlib path from the app: the `matplotlib` import, the `register_korean_font`
  call, and the font machinery added only for the app's matplotlib plot (`src/fonts.py`,
  `tests/test_fonts.py`, `assets/fonts/NanumGothic-Regular.ttf`).
- Add axis-title i18n keys (EN/KO).

## Module: `src/charts.py`

```python
import pandas as pd

def response_dataframe(freqs, series, fmin=20.0, fmax=20000.0):
    """Tidy (long-form) DataFrame for an Altair line chart.

    ``series`` maps a label -> a magnitude-dB array aligned with ``freqs``.
    Keeps only in-band rows (fmin <= f <= fmax), which also drops the 0 Hz bin so
    the log x-axis is safe. Columns: ``freq_hz`` (float), ``magnitude_db``
    (float), ``series`` (str). Row order: each series' in-band points in turn.
    """
```

- Pure: numpy + pandas only, no Streamlit. Each series contributes one row per in-band frequency.
- The band filter removes the 0 Hz rfft bin (log scale can't show 0) and matches the design band.

## app.py changes

- Replace the `matplotlib` figure block. Smooth each curve with `fractional_octave_smooth` as today,
  assemble a `series` dict (`before`, each method with its σ in the label, and the target), build the
  DataFrame via `response_dataframe`, and render:
  - `alt.Chart(df).mark_line()` with
    `x = alt.X("freq_hz", scale=alt.Scale(type="log", domain=[20, 20000]), title=t(lang,"freq_axis"))`,
    `y = alt.Y("magnitude_db", title=t(lang,"mag_axis"))`,
    `color = alt.Color("series", scale=alt.Scale(domain=[...], range=[pastel...]))`,
    `tooltip = ["series", "freq_hz", "magnitude_db"]`.
  - The target is drawn as a **separate layered mark** (`mark_line(strokeDash=[6, 4], color="#9AA0A6")`)
    built from its own one-series DataFrame, layered under the coloured lines. This keeps the target
    visually distinct without entangling it in the colour scale. Title via
    `properties(title=t(lang,"plot_title"))`.
  - `st.altair_chart(chart, use_container_width=True)`.
- Pastel mapping: before `#9AA0A6`, classic `#7FB5B5`, ddsp `#F6C28B`, fir `#B5A7E6`, target a muted grey.
- Remove `import matplotlib` / `import matplotlib.pyplot as plt`, the `colors` dict, the
  `register_korean_font` import+call. Keep `fractional_octave_smooth` (it's from `analysis`, not mpl).
- **Delete** `src/fonts.py`, `tests/test_fonts.py`, `assets/fonts/NanumGothic-Regular.ttf` (only existed
  for the app's matplotlib Korean labels, now obsolete). The notebook keeps its own matplotlib (English
  labels), so nothing there breaks.
- i18n: add `freq_axis` ("Frequency [Hz]" / "주파수 [Hz]") and `mag_axis` ("Magnitude [dB]" / "크기 [dB]")
  to both `en` and `ko`. Reuse `plot_title` for the chart title.

## Dependencies

`altair` and `pandas` are already installed transitively via Streamlit, but the app now imports them
directly, so add them to `requirements.txt` pinned to the versions Streamlit Cloud resolved
(`altair==6.2.1`, `pandas==3.0.3`).

## Testing (`tests/test_charts.py`)

- `response_dataframe` columns are exactly `freq_hz, magnitude_db, series`.
- Two series over 5 freqs (two out of band) -> 2 series × 3 in-band rows = 6 rows; the 0 Hz / out-of-band
  rows are absent.
- Series labels are preserved and each label's `magnitude_db` values match the input arrays (in band).
- A single-series call yields one row per in-band frequency.

The existing `test_i18n` parity test guards the two new keys; removing `test_fonts.py` drops 2 tests.

Suite: 121 - 2 (fonts) + ~4 (charts) ≈ **123**.

## Non-goals

- One chart type (line); no animation, no theme toggle, no Plotly. Notebook charts stay matplotlib.
- No change to correction maths, metrics, export, or the A/B audio section.

## Success criteria

- The app's response plot is an interactive Altair chart matching the pastel UI, with working hover and
  correct Korean labels — and no bundled font.
- `charts.py` is pure and passes its tests; the full suite stays green.
