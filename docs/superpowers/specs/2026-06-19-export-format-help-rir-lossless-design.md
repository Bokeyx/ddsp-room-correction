# Export format tooltips + lossless RIR upload — design

- **Date:** 2026-06-19
- **Status:** approved (brainstorming)
- **Scope:** the app's export buttons + the advanced RIR uploader. Two small, related UI changes.

## Motivation

Two follow-ups from using the app:

1. The export buttons (Equalizer APO / REW / CSV / FIR WAV) give no hint what each format is for. A hover
   tooltip per button explains it without cluttering the layout.
2. The advanced "upload my room" RIR input only accepts WAV. Now that `decode_audio` exists, the RIR can
   accept FLAC too. We deliberately keep it **lossless only** (WAV/FLAC): a RIR is an impulse response,
   and a lossy codec (mp3/m4a) would distort it and degrade the correction.

## Goals

- A per-format help tooltip on each export download button (Streamlit `help=`), in EN/KO.
- The RIR uploader accepts `wav`/`flac` and reads via `decode_audio` (consistent with the music path);
  lossy formats stay out.

## Part A — export tooltips

New i18n keys (both languages), one per format (CSV is shared by the peaking and FIR buttons):

- `help_eqapo` — EN: "Equalizer APO config.txt — paste into a free Windows system-wide EQ." /
  KO: "Equalizer APO config.txt — Windows 무료 시스템 EQ에 붙여넣기."
- `help_rew` — EN: "Room EQ Wizard parametric-filter list." / KO: "Room EQ Wizard 파라메트릭 필터 리스트."
- `help_csv` — EN: "Filter parameters + before/after response (for analysis)." /
  KO: "필터 파라미터 + 보정 전·후 응답 (분석용)."
- `help_firwav` — EN: "Impulse for convolution engines (Equalizer APO Convolution, Roon)." /
  KO: "convolution 엔진용 임펄스 (Equalizer APO Convolution, Roon)."

app.py: add `help=t(lang, "help_…")` to each `st.download_button` (eqapo→help_eqapo, rew→help_rew, both
CSV buttons→help_csv, FIR WAV→help_firwav). No layout change.

## Part B — lossless RIR upload

In `app.py`, the advanced RIR branch currently:

```python
up = st.sidebar.file_uploader(t(lang, "uploader"), type=["wav"])
if up is None:
    st.info(t(lang, "upload_info"))
    st.stop()
rir, sr = sf.read(io.BytesIO(up.read()), dtype="float64")
if rir.ndim > 1:
    rir = rir.mean(axis=1)
```

becomes:

```python
up = st.sidebar.file_uploader(t(lang, "uploader"), type=["wav", "flac"])
if up is None:
    st.info(t(lang, "upload_info"))
    st.stop()
rir, sr = decode_audio(up.read())
if rir.ndim > 1:
    rir = rir.mean(axis=1)
```

`decode_audio` already handles WAV/FLAC via its soundfile path (no ffmpeg needed). `decode_audio` is
already imported in app.py.

## Testing

- `tests/test_i18n.py`: add a test asserting `help_eqapo`, `help_rew`, `help_csv`, `help_firwav` exist in
  both `en` and `ko`. The existing key-parity test keeps the tables in sync.
- The RIR `decode_audio` path is already covered by `test_audio` (soundfile path). The app wiring is glue
  verified by the parse smoke check + full suite.

Suite: 126 -> 127.

## Non-goals

- No per-format captions or expander (tooltips only). No lossy RIR formats. No new modules. No change to
  correction maths, the chart, or the A/B section.

## Success criteria

- Hovering an export button shows a one-line format description (EN/KO). The advanced RIR uploader accepts
  WAV and FLAC and corrects them. The suite stays green.
