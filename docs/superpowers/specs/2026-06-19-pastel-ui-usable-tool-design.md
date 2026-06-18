# Pastel-UI usable tool — export the correction + theme the demo — design

- **Date:** 2026-06-19
- **Status:** approved (brainstorming)
- **Sub-project:** 2 of 3 in the "deepen + ship" push (perceptual loss done; this one; then robustness/generalization).

## Motivation

The Streamlit demo (`app.py`) already does the interesting half of a usable tool: upload a RIR (or
pick a synthetic one), correct it with classic / DDSP / FIR, compare before/after on a smoothed
response plot with σ metrics, and A/B the result through pink noise. What it cannot do is let a person
*keep* the correction. The filters live and die inside the browser session — there is no way to take a
designed EQ out to the system that actually plays music (Equalizer APO, REW, a convolution engine).

This sub-project closes that gap: it turns the designed correction into files real audio tools import,
and gives the demo a calm pastel coat that matches the notebook/README figures. The maths is unchanged;
the value added is purely *getting the result out the door* plus visual polish.

## Goals

- A pure-function export module (`src/export.py`) that turns the existing `list[PeakingFilter]`
  (classic/DDSP) or FIR `taps` into the formats real tools read:
  - **Equalizer APO** `config.txt` text,
  - **REW** parametric-filter text,
  - **FIR impulse WAV** bytes (for convolution engines),
  - **CSV** of the filter parameters and the before/after response (for analysis / reproducibility).
- The functions return strings / bytes only — no file IO, no Streamlit dependency — so they are unit
  tested without the app and can be reused by the notebook.
- `app.py` wires each method's result to download buttons (only the formats that method can produce),
  and re-uses the project's pastel palette on the plot.
- A restrained pastel theme via `.streamlit/config.toml` (and the existing palette on the matplotlib
  figure) — no heavy custom CSS, to avoid the generic-AI-slop look.

## Non-goals (deliberately cut — YAGNI)

- No login, session persistence, or database. Single-page demo stays single-page.
- No cloud deployment, Docker, or hosting. Local `streamlit run app.py`.
- No custom CSS component library or web fonts. config.toml theme + the shared palette only.
- No multi-language UI. Repo content stays English (project rule).
- No new correction maths. This sub-project is export + theme only.

## Architecture

```
src/export.py        (NEW)  pure functions: correction -> str / bytes
tests/test_export.py (NEW)  TDD red->green, deterministic string compares
app.py               (EDIT) call export.* + st.download_button per method + pastel palette
.streamlit/config.toml (NEW) pastel Streamlit theme
```

The split keeps the core/UI boundary the rest of the project already follows (`pipeline.correct`
returns the correction object; the notebook and the app both consume it). Export is core, testable,
and notebook-reusable; the app is a thin shell over it.

## `src/export.py` — function specs

All inputs are objects the pipeline already returns. `PeakingFilter` is a dataclass with
`freq_hz`, `gain_db`, `q`. FIR `taps` is a 1-D ndarray.

Number formatting is fixed so output is deterministic and tests can compare strings exactly:
`Fc` as integer Hz, `Gain` to 2 decimals, `Q` to 3 decimals.

### `to_eqapo_config(filters) -> str`
Equalizer APO `config.txt` body. A leading comment header (tool name, filter count), then one line
per filter:
```
# DDSP Room Correction - Equalizer APO config
# 3 peaking filters
Filter: ON PK Fc 120 Hz Gain -3.00 dB Q 4.000
Filter: ON PK Fc 1000 Hz Gain 2.50 dB Q 4.000
...
```

### `to_rew_filters(filters) -> str`
REW parametric-filter import text. Numbered filters in REW's convention:
```
Filter Settings file
Filter 1: ON PK Fc 120 Hz Gain -3.00 dB Q 4.000
Filter 2: ON PK Fc 1000 Hz Gain 2.50 dB Q 4.000
...
```
(Separate from EQ APO because the header and per-line numbering differ; each targets the exact
format its tool imports.)

### `to_fir_wav_bytes(taps, sr) -> bytes`
The FIR impulse encoded as a WAV (float32, mono) via soundfile writing to an in-memory buffer. For
convolution engines (Equalizer APO Convolution, Roon, etc.). Only offered for the `fir` method.

### `to_csv(filters, freqs_hz, before_db, after_db) -> str`
Two logical parts in one CSV-friendly text return: a short commented preamble listing the filter
parameters (one `# filter,freq_hz,gain_db,q` line each), then the per-frequency before/after table with
header `freq_hz,before_db,after_db` and one row per frequency. For analysis and reproducibility — an
archive format, not a "tool" format. For the FIR method (no peaking filters) the preamble notes the tap
count instead of per-filter lines.

## app.py wiring

- Under each selected method's result, show download buttons for the formats that method can produce:
  - `classic` / `ddsp` (PeakingFilter list): EQ APO, REW, CSV.
  - `fir` (taps ndarray): FIR WAV, CSV.
- Replace the matplotlib `tab:` colors with the shared pastel palette: before = grey,
  classic = `#7FB5B5` (teal), ddsp = `#F6C28B` (peach), fir = `#B5A7E6` (lavender), target = dashed.
- No change to the correction call, metrics, or A/B section.

## Pastel theme

`.streamlit/config.toml`:
- `base = "light"`, `primaryColor = "#7FB5B5"`, soft `backgroundColor` / `secondaryBackgroundColor`.
- Matplotlib figure uses the same palette (above) so the app and the notebook figures match.
- Minimal-to-no custom CSS — restraint over decoration.

## Testing

- `to_eqapo_config`: known 1-2 filters -> exact line strings + header present.
- `to_rew_filters`: numbering and per-line format compared exactly.
- `to_fir_wav_bytes`: read the bytes back with soundfile; assert length, sample rate, mono, finite.
- `to_csv`: header columns present; one data row per frequency; before/after values round-trip.
- Edge cases: empty filter list (valid header, no filter lines); a filter with `gain_db == 0`.

Expected suite: 95 -> ~103.

## Success criteria

- The four export functions exist, are pure (no IO / no Streamlit), and pass the tests above.
- `streamlit run app.py` shows the pastel palette and offers, per method, download buttons that
  produce files in the documented formats.
- A correction designed in the app can be downloaded as an EQ APO `config.txt` whose lines match the
  documented format (verified by the unit test on the same function the app calls).
