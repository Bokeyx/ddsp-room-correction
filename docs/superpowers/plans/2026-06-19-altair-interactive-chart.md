# Altair Interactive App Chart Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the app's static matplotlib response plot with an interactive Altair chart (hover, zoom, pastel, browser-rendered Korean), and remove the now-obsolete matplotlib font machinery from the app.

**Architecture:** A pure `src/charts.py` turns the response arrays into a tidy DataFrame; `app.py` renders it as a layered Altair line chart (coloured series + dashed target) via `st.altair_chart`. The notebook keeps its own matplotlib; only the app changes.

**Tech Stack:** Python 3, numpy, pandas, altair, Streamlit, pytest. Python executable: `D:\Bokey\Sound_Quality\.venv\Scripts\python.exe`. Tests: `python -m pytest -q`.

## Global Constraints

- Single-author commits. NEVER add a `Co-Authored-By` trailer.
- All repo content in English (UI strings get an EN default + KO translation).
- `src/charts.py` is PURE: numpy + pandas only, no Streamlit, no IO.
- Pastel palette: before `#9AA0A6`, classic `#7FB5B5`, ddsp `#F6C28B`, fir `#B5A7E6`, target muted grey `#9AA0A6` dashed.
- i18n: `STRINGS["en"]` and `STRINGS["ko"]` keep identical key sets (enforced by `test_i18n`).
- Notebook (`_build_notebook.py`, `room_correction.ipynb`) is NOT touched.

---

### Task 1: `src/charts.py` — tidy response DataFrame

**Files:**
- Create: `src/charts.py`
- Test: `tests/test_charts.py`

**Interfaces:**
- Consumes: numpy, pandas.
- Produces: `response_dataframe(freqs, series, fmin=20.0, fmax=20000.0) -> pandas.DataFrame`
  with columns `freq_hz`, `magnitude_db`, `series`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_charts.py
import numpy as np

from src.charts import response_dataframe


def test_columns_and_inband_filtering():
    freqs = np.array([0.0, 10.0, 100.0, 1000.0, 30000.0])  # 0,10 below band; 30000 above
    s = {"a": np.array([1., 2., 3., 4., 5.]), "b": np.array([10., 20., 30., 40., 50.])}
    df = response_dataframe(freqs, s)
    assert list(df.columns) == ["freq_hz", "magnitude_db", "series"]
    assert len(df) == 4  # in-band freqs {100,1000} x 2 series
    assert set(df["series"]) == {"a", "b"}
    assert sorted(df["freq_hz"].unique().tolist()) == [100.0, 1000.0]


def test_values_preserved_in_band():
    freqs = np.array([100.0, 1000.0, 10000.0])
    df = response_dataframe(freqs, {"x": np.array([1.0, 2.0, 3.0])})
    assert df["magnitude_db"].tolist() == [1.0, 2.0, 3.0]


def test_single_series_row_count():
    freqs = np.array([20.0, 200.0, 2000.0, 20000.0])
    df = response_dataframe(freqs, {"only": np.zeros(4)})
    assert len(df) == 4
    assert (df["series"] == "only").all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `D:\Bokey\Sound_Quality\.venv\Scripts\python.exe -m pytest tests/test_charts.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.charts'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/charts.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `D:\Bokey\Sound_Quality\.venv\Scripts\python.exe -m pytest tests/test_charts.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/charts.py tests/test_charts.py
git commit -m "feat: response_dataframe (tidy frame for the Altair chart)"
```

---

### Task 2: Render the Altair chart in app.py; remove matplotlib + font machinery

**Files:**
- Modify: `src/i18n.py`, `app.py`, `requirements.txt`
- Delete: `src/fonts.py`, `tests/test_fonts.py`, `assets/fonts/NanumGothic-Regular.ttf`

**Interfaces:**
- Consumes: `response_dataframe` (Task 1); existing `fractional_octave_smooth`, `correct`,
  `smoothed_sigma`, `t`, `LANGUAGES`.
- Produces: a reworked app (no importable symbols for later tasks).

- [ ] **Step 1: Add axis-title i18n keys (both languages)**

In `src/i18n.py`, add to the `"en"` dict (next to `plot_title`):

```python
        "freq_axis": "Frequency [Hz]",
        "mag_axis": "Magnitude [dB]",
```

and the same keys to the `"ko"` dict:

```python
        "freq_axis": "주파수 [Hz]",
        "mag_axis": "크기 [dB]",
```

- [ ] **Step 2: Run the i18n parity test**

Run: `D:\Bokey\Sound_Quality\.venv\Scripts\python.exe -m pytest tests/test_i18n.py -q`
Expected: PASS (5 passed).

- [ ] **Step 3: Swap imports in app.py**

In `app.py`, remove `import matplotlib.pyplot as plt` and replace with `import altair as alt`. Then remove
the `from src.fonts import register_korean_font` line and add the charts import. The import block top
becomes:

```python
import io

import altair as alt
import numpy as np
import scipy.signal as sps
import soundfile as sf
import streamlit as st

from src.analysis import fractional_octave_smooth, frequency_response
from src.audio import apply_eq_to_signal, apply_fir_to_signal, pink_noise, prepare_clip
from src.charts import response_dataframe
from src.export import to_eqapo_config, to_rew_filters, to_fir_wav_bytes, to_csv
from src.i18n import LANGUAGES, t
from src.pipeline import correct, smoothed_sigma
from src.rooms import preset_names, preset_rir
from src.targets import flat_target, harman_target
```

- [ ] **Step 4: Remove the font registration block**

In `app.py`, delete these lines (added earlier for matplotlib):

```python
# Korean plot labels need a CJK font; register the bundled one before any figure.
try:
    register_korean_font()
except FileNotFoundError:
    pass
```

- [ ] **Step 5: Replace the matplotlib figure with an Altair chart**

In `app.py`, replace this whole block:

```python
colors = {"classic": "#7FB5B5", "ddsp": "#F6C28B", "fir": "#B5A7E6"}

fig, ax = plt.subplots(figsize=(10, 4))
ax.semilogx(freqs, fractional_octave_smooth(freqs, resp), color="#9AA0A6", lw=1.5,
            label=f"{t(lang, 'before')} (σ={before:.2f})")
results = {}
with st.spinner(t(lang, "spinner")):
    for m in methods:
        corrected_db, corr = _correct(resp, target, freqs, sr, m, n_filters)
        sigma = smoothed_sigma(corrected_db, freqs)
        results[m] = (corrected_db, corr, sigma)
        ax.semilogx(freqs, fractional_octave_smooth(freqs, corrected_db), color=colors[m],
                    lw=1.8, label=f"{m} (σ={sigma:.2f})")
ax.semilogx(freqs, target, "k--", lw=1, alpha=0.6, label=f"{target_name} {t(lang, 'target_suffix')}")
ax.set(xlim=(20, 20000), xlabel="frequency [Hz]", ylabel="magnitude [dB]",
       title=t(lang, "plot_title"))
ax.legend()
ax.grid(alpha=0.3)
st.pyplot(fig)
```

with:

```python
PALETTE = {"before": "#9AA0A6", "classic": "#7FB5B5", "ddsp": "#F6C28B", "fir": "#B5A7E6"}

before_label = f"{t(lang, 'before')} (σ={before:.2f})"
series = {before_label: fractional_octave_smooth(freqs, resp)}
domain, scheme = [before_label], [PALETTE["before"]]
results = {}
with st.spinner(t(lang, "spinner")):
    for m in methods:
        corrected_db, corr = _correct(resp, target, freqs, sr, m, n_filters)
        sigma = smoothed_sigma(corrected_db, freqs)
        results[m] = (corrected_db, corr, sigma)
        label = f"{m} (σ={sigma:.2f})"
        series[label] = fractional_octave_smooth(freqs, corrected_db)
        domain.append(label)
        scheme.append(PALETTE[m])

resp_df = response_dataframe(freqs, series)
target_df = response_dataframe(freqs, {f"{target_name} {t(lang, 'target_suffix')}": target})

x_enc = alt.X("freq_hz:Q", scale=alt.Scale(type="log", domain=[20, 20000]),
              title=t(lang, "freq_axis"))
lines = alt.Chart(resp_df).mark_line().encode(
    x=x_enc,
    y=alt.Y("magnitude_db:Q", title=t(lang, "mag_axis")),
    color=alt.Color("series:N", scale=alt.Scale(domain=domain, range=scheme), title=None),
    tooltip=[alt.Tooltip("series:N", title="series"),
             alt.Tooltip("freq_hz:Q", title="Hz", format=".0f"),
             alt.Tooltip("magnitude_db:Q", title="dB", format=".1f")],
)
target_line = alt.Chart(target_df).mark_line(strokeDash=[6, 4], color="#9AA0A6").encode(
    x=x_enc, y="magnitude_db:Q",
)
chart = (target_line + lines).properties(title=t(lang, "plot_title"), height=360).interactive()
st.altair_chart(chart, use_container_width=True)
```

- [ ] **Step 6: Delete the obsolete font files**

Run:
```bash
cd D:\Bokey\Sound_Quality && git rm src/fonts.py tests/test_fonts.py assets/fonts/NanumGothic-Regular.ttf
```
Expected: git stages three deletions.

- [ ] **Step 7: Add altair + pandas to requirements.txt**

In `requirements.txt`, add these two lines under the existing pins:

```
altair==6.2.1
pandas==3.0.3
```

- [ ] **Step 8: Smoke-check (parse + headless chart build) and run the full suite**

Run:
```bash
D:\Bokey\Sound_Quality\.venv\Scripts\python.exe -c "import ast; ast.parse(open('app.py', encoding='utf-8').read()); print('app.py parses')"
D:\Bokey\Sound_Quality\.venv\Scripts\python.exe -c "import numpy as np, altair as alt; from src.charts import response_dataframe; df = response_dataframe(np.array([20.,200.,2000.,20000.]), {'a': np.zeros(4)}); alt.Chart(df).mark_line().encode(x=alt.X('freq_hz:Q', scale=alt.Scale(type='log')), y='magnitude_db:Q', color='series:N').to_dict(); print('altair chart builds')"
D:\Bokey\Sound_Quality\.venv\Scripts\python.exe -m pytest -q
```
Expected: `app.py parses`; `altair chart builds`; full suite PASS (~123 passed — Task-1 charts tests added,
`test_fonts.py` removed, everything else green). Then a manual check: `streamlit run app.py` shows an
interactive chart that matches the pastel UI, hovers show Hz/dB, and Korean labels render.

- [ ] **Step 9: Commit**

```bash
git add app.py src/i18n.py requirements.txt
git commit -m "feat: interactive Altair response chart in the app (drop matplotlib + bundled font)"
```

---

## Self-Review notes

- **Spec coverage:** `response_dataframe` (T1); Altair layered chart with log-x, pastel colours, dashed
  target, tooltip, Korean axis titles (T2 Steps 1,5); removal of matplotlib import + font block + font
  files (T2 Steps 3,4,6); altair/pandas pinned in requirements (T2 Step 7). Notebook untouched.
- **Type consistency:** `response_dataframe(freqs, series, ...)` signature matches T1 and the two call
  sites in T2 (`resp_df`, `target_df`); columns `freq_hz`/`magnitude_db`/`series` match the Altair field
  refs (`freq_hz:Q`, `magnitude_db:Q`, `series:N`). `domain`/`scheme` lists are built in lockstep so the
  colour scale maps each label to its pastel colour.
- **i18n parity:** `freq_axis`/`mag_axis` added to both en+ko (T2 Step 1), guarded by `test_i18n` (Step 2).
- **Dead-code check:** after T2, app.py no longer references `plt`, `colors`, `register_korean_font`; the
  font files are deleted; `fractional_octave_smooth`/`np` are still used and kept.
- **Placeholder scan:** every step shows exact code; the only verification "fill-in" is reading the smoke
  outputs, not deferred design.
