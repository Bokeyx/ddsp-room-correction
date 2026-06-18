# Pastel-UI Usable Tool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a user export a designed room correction (EQ APO / REW / FIR WAV / CSV) and give the Streamlit demo a restrained pastel coat.

**Architecture:** A new pure-function module `src/export.py` turns the existing `list[PeakingFilter]` (classic/DDSP) or FIR `taps` ndarray into strings/bytes that real audio tools import. `app.py` is a thin shell that calls these and exposes `st.download_button`s per method, plus a pastel matplotlib palette and a `.streamlit/config.toml` theme. No correction maths changes.

**Tech Stack:** Python 3, numpy, soundfile, Streamlit, pytest. Python executable: `D:\Bokey\Sound_Quality\.venv\Scripts\python.exe`. Tests: `python -m pytest -q`.

## Global Constraints

- Single-author commits. NEVER add a `Co-Authored-By` trailer.
- All repo content in English.
- Export functions are PURE: no file IO, no Streamlit import in `src/export.py`. Return `str` or `bytes`.
- Number formatting fixed for deterministic output: `Fc` = integer Hz (`round`), `Gain` = 2 decimals, `Q` = 3 decimals.
- `PeakingFilter` is a dataclass with fields `freq_hz`, `gain_db`, `q` (from `src/eq_classic.py`).
- FIR `taps` is a 1-D float ndarray.
- Notebook is regenerated via `notebooks/_build_notebook.py`; never hand-edit the `.ipynb` (not touched in this plan).
- Pastel palette (shared with notebook §4c): classic `#7FB5B5`, ddsp `#F6C28B`, fir `#B5A7E6`, before = grey.

---

### Task 1: `to_eqapo_config` — Equalizer APO export

**Files:**
- Create: `src/export.py`
- Test: `tests/test_export.py`

**Interfaces:**
- Consumes: `PeakingFilter(freq_hz, gain_db, q)` from `src.eq_classic`.
- Produces: `to_eqapo_config(filters: list[PeakingFilter]) -> str`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_export.py
from src.eq_classic import PeakingFilter
from src.export import to_eqapo_config


def test_eqapo_config_lines_and_header():
    filters = [
        PeakingFilter(freq_hz=120.4, gain_db=-3.0, q=4.0),
        PeakingFilter(freq_hz=1000.0, gain_db=2.5, q=2.0),
    ]
    text = to_eqapo_config(filters)
    lines = text.splitlines()
    assert lines[0].startswith("#")
    assert "2 peaking filters" in text
    assert "Filter: ON PK Fc 120 Hz Gain -3.00 dB Q 4.000" in lines
    assert "Filter: ON PK Fc 1000 Hz Gain 2.50 dB Q 2.000" in lines


def test_eqapo_config_empty_filters_has_header_no_filter_lines():
    text = to_eqapo_config([])
    assert "0 peaking filters" in text
    assert "Filter:" not in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `D:\Bokey\Sound_Quality\.venv\Scripts\python.exe -m pytest tests/test_export.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.export'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/export.py
"""Export a designed correction to formats real audio tools import.

Pure functions only -- they take the objects `pipeline.correct` already returns
(`list[PeakingFilter]` for classic/ddsp, an ndarray of taps for fir) and return
strings or bytes. No file IO and no Streamlit dependency, so the app and the
notebook can both reuse them and they are trivially unit tested.

Number formatting is fixed (Fc integer Hz, Gain 2dp, Q 3dp) so output is
deterministic.
"""


def _peak_line(filt):
    return (f"Fc {round(filt.freq_hz)} Hz "
            f"Gain {filt.gain_db:.2f} dB "
            f"Q {filt.q:.3f}")


def to_eqapo_config(filters):
    """Equalizer APO ``config.txt`` body for a list of PeakingFilter."""
    lines = [
        "# DDSP Room Correction - Equalizer APO config",
        f"# {len(filters)} peaking filters",
    ]
    lines += [f"Filter: ON PK {_peak_line(f)}" for f in filters]
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `D:\Bokey\Sound_Quality\.venv\Scripts\python.exe -m pytest tests/test_export.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/export.py tests/test_export.py
git commit -m "feat: to_eqapo_config export (Equalizer APO config.txt)"
```

---

### Task 2: `to_rew_filters` — REW parametric export

**Files:**
- Modify: `src/export.py`
- Test: `tests/test_export.py`

**Interfaces:**
- Consumes: `PeakingFilter`, `_peak_line` (from Task 1).
- Produces: `to_rew_filters(filters: list[PeakingFilter]) -> str`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_export.py
from src.export import to_rew_filters  # add to existing imports


def test_rew_filters_numbered_with_header():
    filters = [
        PeakingFilter(freq_hz=120.0, gain_db=-3.0, q=4.0),
        PeakingFilter(freq_hz=1000.0, gain_db=2.5, q=2.0),
    ]
    text = to_rew_filters(filters)
    lines = text.splitlines()
    assert lines[0] == "Filter Settings file"
    assert "Filter 1: ON PK Fc 120 Hz Gain -3.00 dB Q 4.000" in lines
    assert "Filter 2: ON PK Fc 1000 Hz Gain 2.50 dB Q 2.000" in lines
```

- [ ] **Step 2: Run test to verify it fails**

Run: `D:\Bokey\Sound_Quality\.venv\Scripts\python.exe -m pytest tests/test_export.py::test_rew_filters_numbered_with_header -q`
Expected: FAIL — `ImportError: cannot import name 'to_rew_filters'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/export.py
def to_rew_filters(filters):
    """REW parametric-filter import text (numbered filters)."""
    lines = ["Filter Settings file"]
    lines += [
        f"Filter {i}: ON PK {_peak_line(f)}"
        for i, f in enumerate(filters, start=1)
    ]
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `D:\Bokey\Sound_Quality\.venv\Scripts\python.exe -m pytest tests/test_export.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/export.py tests/test_export.py
git commit -m "feat: to_rew_filters export (REW parametric filter list)"
```

---

### Task 3: `to_fir_wav_bytes` — FIR impulse WAV export

**Files:**
- Modify: `src/export.py`
- Test: `tests/test_export.py`

**Interfaces:**
- Consumes: FIR `taps` (1-D ndarray), `sr` (int).
- Produces: `to_fir_wav_bytes(taps, sr) -> bytes`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_export.py
import io
import numpy as np
import soundfile as sf
from src.export import to_fir_wav_bytes  # add to existing imports


def test_fir_wav_bytes_roundtrip():
    sr = 48000
    taps = np.array([0.0, 1.0, -0.5, 0.25], dtype=np.float64)
    blob = to_fir_wav_bytes(taps, sr)
    assert isinstance(blob, bytes)

    data, read_sr = sf.read(io.BytesIO(blob), dtype="float32")
    assert read_sr == sr
    assert data.ndim == 1
    assert len(data) == len(taps)
    assert np.allclose(data, taps, atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `D:\Bokey\Sound_Quality\.venv\Scripts\python.exe -m pytest tests/test_export.py::test_fir_wav_bytes_roundtrip -q`
Expected: FAIL — `ImportError: cannot import name 'to_fir_wav_bytes'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/export.py (add `import io`, `import numpy as np`,
# `import soundfile as sf` at the top of the module)
def to_fir_wav_bytes(taps, sr):
    """FIR impulse encoded as a mono float32 WAV (for convolution engines)."""
    taps = np.asarray(taps, dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, taps, int(sr), format="WAV", subtype="FLOAT")
    return buf.getvalue()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `D:\Bokey\Sound_Quality\.venv\Scripts\python.exe -m pytest tests/test_export.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/export.py tests/test_export.py
git commit -m "feat: to_fir_wav_bytes export (FIR impulse WAV for convolution)"
```

---

### Task 4: `to_csv` — parameters + before/after response archive

**Files:**
- Modify: `src/export.py`
- Test: `tests/test_export.py`

**Interfaces:**
- Consumes: `filters` (`list[PeakingFilter]`, or `None` for the FIR method), `freqs_hz`, `before_db`, `after_db` (1-D ndarrays of equal length), and `n_taps` (int, only used when `filters is None`).
- Produces: `to_csv(filters, freqs_hz, before_db, after_db, n_taps=None) -> str`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_export.py
from src.export import to_csv  # add to existing imports


def test_csv_peaking_preamble_and_table():
    filters = [PeakingFilter(freq_hz=120.0, gain_db=-3.0, q=4.0)]
    freqs = np.array([20.0, 100.0, 1000.0])
    before = np.array([1.0, 2.0, 3.0])
    after = np.array([0.1, 0.2, 0.3])
    text = to_csv(filters, freqs, before, after)
    lines = text.splitlines()

    assert any(l.startswith("# filter,freq_hz,gain_db,q") for l in lines)
    assert "# 1,120,-3.00,4.000" in lines
    assert "freq_hz,before_db,after_db" in lines
    data_rows = [l for l in lines if not l.startswith("#") and "," in l
                 and not l.startswith("freq_hz")]
    assert len(data_rows) == len(freqs)
    assert data_rows[0] == "20.0,1.0,0.1"


def test_csv_fir_preamble_notes_tap_count():
    freqs = np.array([20.0, 100.0])
    before = np.array([1.0, 2.0])
    after = np.array([0.1, 0.2])
    text = to_csv(None, freqs, before, after, n_taps=2048)
    assert "# fir filter, 2048 taps" in text
    assert "freq_hz,before_db,after_db" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `D:\Bokey\Sound_Quality\.venv\Scripts\python.exe -m pytest tests/test_export.py -k csv -q`
Expected: FAIL — `ImportError: cannot import name 'to_csv'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/export.py
def to_csv(filters, freqs_hz, before_db, after_db, n_taps=None):
    """Filter parameters (commented preamble) + per-frequency before/after table.

    ``filters=None`` is the FIR case: the preamble notes the tap count
    (``n_taps``) instead of per-filter lines.
    """
    freqs_hz = np.asarray(freqs_hz, dtype=np.float64)
    before_db = np.asarray(before_db, dtype=np.float64)
    after_db = np.asarray(after_db, dtype=np.float64)

    lines = ["# DDSP Room Correction - export"]
    if filters is None:
        lines.append(f"# fir filter, {int(n_taps)} taps")
    else:
        lines.append("# filter,freq_hz,gain_db,q")
        for i, f in enumerate(filters, start=1):
            lines.append(f"# {i},{round(f.freq_hz)},{f.gain_db:.2f},{f.q:.3f}")

    lines.append("freq_hz,before_db,after_db")
    for fr, b, a in zip(freqs_hz, before_db, after_db):
        lines.append(f"{fr},{b},{a}")
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `D:\Bokey\Sound_Quality\.venv\Scripts\python.exe -m pytest tests/test_export.py -q`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/export.py tests/test_export.py
git commit -m "feat: to_csv export (filter params + before/after response archive)"
```

---

### Task 5: Wire exports + pastel theme into `app.py`

**Files:**
- Modify: `app.py`
- Create: `.streamlit/config.toml`

**Interfaces:**
- Consumes: `to_eqapo_config`, `to_rew_filters`, `to_fir_wav_bytes`, `to_csv` (Tasks 1-4); existing `correct`, `smoothed_sigma`, `fractional_octave_smooth`.
- Produces: a running app (no importable symbols for later tasks).

This task has no unit test (it is Streamlit glue); the deliverable is verified by importing `app.py`'s logic indirectly — the export functions it calls are already tested. Verification step is an import smoke check plus the full suite staying green.

- [ ] **Step 1: Create the pastel Streamlit theme**

```toml
# .streamlit/config.toml
[theme]
base = "light"
primaryColor = "#7FB5B5"
backgroundColor = "#FBF9F6"
secondaryBackgroundColor = "#F1ECE6"
textColor = "#3A3A3A"
```

- [ ] **Step 2: Swap the matplotlib palette to the shared pastel colors**

In `app.py`, replace the `colors` dict:

```python
# app.py  (was: {"classic": "tab:green", "ddsp": "tab:red", "fir": "tab:blue"})
colors = {"classic": "#7FB5B5", "ddsp": "#F6C28B", "fir": "#B5A7E6"}
```

And change the "before" curve color to grey if not already (`color="#9AA0A6"` in the `ax.semilogx(... before ...)` call).

- [ ] **Step 3: Add the export import**

At the top of `app.py`, with the other `from src...` imports:

```python
from src.export import to_eqapo_config, to_rew_filters, to_fir_wav_bytes, to_csv
```

- [ ] **Step 4: Add per-method download buttons**

After the metrics block (after the `cols[i + 1].metric(...)` loop, before the A/B section), insert:

```python
# --- export the correction ---
st.subheader("Export correction")
for m in methods:
    corrected_db, corr, _ = results[m]
    st.markdown(f"**{m}**")
    cdl = st.columns(3)
    if isinstance(corr, np.ndarray):
        # FIR: impulse WAV + CSV
        cdl[0].download_button(
            "FIR impulse WAV", data=to_fir_wav_bytes(corr, sr),
            file_name=f"correction_{m}.wav", mime="audio/wav", key=f"wav_{m}")
        cdl[1].download_button(
            "CSV", data=to_csv(None, freqs, resp, corrected_db, n_taps=len(corr)),
            file_name=f"correction_{m}.csv", mime="text/csv", key=f"csv_{m}")
    else:
        # peaking filters: EQ APO + REW + CSV
        cdl[0].download_button(
            "Equalizer APO", data=to_eqapo_config(corr),
            file_name=f"correction_{m}_eqapo.txt", mime="text/plain", key=f"apo_{m}")
        cdl[1].download_button(
            "REW filters", data=to_rew_filters(corr),
            file_name=f"correction_{m}_rew.txt", mime="text/plain", key=f"rew_{m}")
        cdl[2].download_button(
            "CSV", data=to_csv(corr, freqs, resp, corrected_db),
            file_name=f"correction_{m}.csv", mime="text/csv", key=f"csv_{m}")
```

Note: `results[m]` currently unpacks as `(corrected_db, corr, sigma)` — match that order.

- [ ] **Step 5: Smoke-check the app imports and the suite stays green**

Run:
```bash
D:\Bokey\Sound_Quality\.venv\Scripts\python.exe -c "import ast; ast.parse(open('app.py').read()); print('app.py parses')"
D:\Bokey\Sound_Quality\.venv\Scripts\python.exe -m pytest -q
```
Expected: `app.py parses`; full suite PASS (~103 passed — 6 new export tests on top of 95, plus the import wiring touches no tests).

Then a manual visual check (user-run): `streamlit run app.py` shows the pastel plot and per-method download buttons.

- [ ] **Step 6: Commit**

```bash
git add app.py .streamlit/config.toml
git commit -m "feat: pastel theme + per-method correction export in Streamlit app"
```

---

## Self-Review notes

- **Spec coverage:** EQ APO (T1), REW (T2), FIR WAV (T3), CSV with FIR tap-count branch (T4), app wiring + pastel theme + config.toml (T5). All four export formats and both visual goals covered.
- **Type consistency:** `_peak_line` defined in T1, reused T2. `to_csv` signature `(filters, freqs_hz, before_db, after_db, n_taps=None)` consistent between T4 test, T4 impl, and T5 call sites (peaking passes `corr`; FIR passes `None` + `n_taps=len(corr)`).
- **Pastel palette** identical across T2-config, T5 matplotlib, and spec.
- **Determinism:** all string formats use fixed precision; FIR WAV roundtrip tolerance 1e-6 (float32).
