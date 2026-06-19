# Export Format Tooltips + Lossless RIR Upload Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a per-format help tooltip to each export button and let the advanced RIR uploader accept WAV/FLAC via `decode_audio`.

**Architecture:** New EN/KO i18n help strings passed as `help=` on the export download buttons; the RIR branch reads with `decode_audio` and accepts `wav`/`flac`. Pure UI/glue change on top of existing code.

**Tech Stack:** Python 3, Streamlit, soundfile, pytest. Python: `D:\Bokey\Sound_Quality\.venv\Scripts\python.exe`. Tests: `python -m pytest -q`.

## Global Constraints

- Single-author commits. NEVER add a `Co-Authored-By` trailer.
- All repo content in English (English default + Korean translation).
- `STRINGS["en"]` / `STRINGS["ko"]` keep identical key sets (enforced by `test_i18n`).
- RIR upload stays lossless only (`wav`, `flac`); lossy formats are deliberately excluded.
- `decode_audio` (already in `src/audio.py`, already imported in `app.py`) reads WAV/FLAC via soundfile.

---

### Task 1: Export tooltips + lossless RIR upload

**Files:**
- Modify: `src/i18n.py`, `app.py`
- Test: `tests/test_i18n.py`

**Interfaces:**
- Consumes: existing `t`, `decode_audio`, the export loop and RIR branch in `app.py`.
- Produces: i18n keys `help_eqapo`, `help_rew`, `help_csv`, `help_firwav` (both languages).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_i18n.py`:

```python
def test_export_format_help_keys_present_both_languages():
    for key in ("help_eqapo", "help_rew", "help_csv", "help_firwav"):
        assert key in STRINGS["en"], key
        assert key in STRINGS["ko"], key
```

- [ ] **Step 2: Run test to verify it fails**

Run: `D:\Bokey\Sound_Quality\.venv\Scripts\python.exe -m pytest tests/test_i18n.py::test_export_format_help_keys_present_both_languages -q`
Expected: FAIL — `assert 'help_eqapo' in STRINGS["en"]` (key not yet added).

- [ ] **Step 3: Add the help strings to both i18n tables**

In `src/i18n.py`, in the `"en"` dict, after the `"desc_fir": ...` line add:

```python
        "help_eqapo": "Equalizer APO config.txt — paste into a free Windows system-wide EQ.",
        "help_rew": "Room EQ Wizard parametric-filter list.",
        "help_csv": "Filter parameters + before/after response (for analysis).",
        "help_firwav": "Impulse for convolution engines (Equalizer APO Convolution, Roon).",
```

In the `"ko"` dict, after its `"desc_fir": ...` line add:

```python
        "help_eqapo": "Equalizer APO config.txt — Windows 무료 시스템 EQ에 붙여넣기.",
        "help_rew": "Room EQ Wizard 파라메트릭 필터 리스트.",
        "help_csv": "필터 파라미터 + 보정 전·후 응답 (분석용).",
        "help_firwav": "convolution 엔진용 임펄스 (Equalizer APO Convolution, Roon).",
```

- [ ] **Step 4: Run the i18n tests to verify they pass**

Run: `D:\Bokey\Sound_Quality\.venv\Scripts\python.exe -m pytest tests/test_i18n.py -q`
Expected: PASS (7 passed — new test + existing, incl. key-parity).

- [ ] **Step 5: Add `help=` to each export download button**

In `app.py`, the export buttons currently are:

```python
    if isinstance(corr, np.ndarray):
        # FIR: impulse WAV + CSV
        cdl[0].download_button(
            t(lang, "btn_firwav"), data=to_fir_wav_bytes(corr, sr),
            file_name=f"correction_{m}.wav", mime="audio/wav", key=f"wav_{m}")
        cdl[1].download_button(
            t(lang, "btn_csv"), data=to_csv(None, freqs, resp, corrected_db, n_taps=len(corr)),
            file_name=f"correction_{m}.csv", mime="text/csv", key=f"csv_{m}")
    else:
        # peaking filters: EQ APO + REW + CSV
        cdl[0].download_button(
            t(lang, "btn_eqapo"), data=to_eqapo_config(corr),
            file_name=f"correction_{m}_eqapo.txt", mime="text/plain", key=f"apo_{m}")
        cdl[1].download_button(
            t(lang, "btn_rew"), data=to_rew_filters(corr),
            file_name=f"correction_{m}_rew.txt", mime="text/plain", key=f"rew_{m}")
        cdl[2].download_button(
            t(lang, "btn_csv"), data=to_csv(corr, freqs, resp, corrected_db),
            file_name=f"correction_{m}.csv", mime="text/csv", key=f"csv_{m}")
```

Add a `help=` argument to each (matching format → help key):

```python
    if isinstance(corr, np.ndarray):
        # FIR: impulse WAV + CSV
        cdl[0].download_button(
            t(lang, "btn_firwav"), data=to_fir_wav_bytes(corr, sr),
            file_name=f"correction_{m}.wav", mime="audio/wav", key=f"wav_{m}",
            help=t(lang, "help_firwav"))
        cdl[1].download_button(
            t(lang, "btn_csv"), data=to_csv(None, freqs, resp, corrected_db, n_taps=len(corr)),
            file_name=f"correction_{m}.csv", mime="text/csv", key=f"csv_{m}",
            help=t(lang, "help_csv"))
    else:
        # peaking filters: EQ APO + REW + CSV
        cdl[0].download_button(
            t(lang, "btn_eqapo"), data=to_eqapo_config(corr),
            file_name=f"correction_{m}_eqapo.txt", mime="text/plain", key=f"apo_{m}",
            help=t(lang, "help_eqapo"))
        cdl[1].download_button(
            t(lang, "btn_rew"), data=to_rew_filters(corr),
            file_name=f"correction_{m}_rew.txt", mime="text/plain", key=f"rew_{m}",
            help=t(lang, "help_rew"))
        cdl[2].download_button(
            t(lang, "btn_csv"), data=to_csv(corr, freqs, resp, corrected_db),
            file_name=f"correction_{m}.csv", mime="text/csv", key=f"csv_{m}",
            help=t(lang, "help_csv"))
```

- [ ] **Step 6: Accept WAV/FLAC RIR via decode_audio**

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

Change to:

```python
    up = st.sidebar.file_uploader(t(lang, "uploader"), type=["wav", "flac"])
    if up is None:
        st.info(t(lang, "upload_info"))
        st.stop()
    rir, sr = decode_audio(up.read())
    if rir.ndim > 1:
        rir = rir.mean(axis=1)
```

- [ ] **Step 7: Smoke-check and run the full suite**

Run:
```bash
D:\Bokey\Sound_Quality\.venv\Scripts\python.exe -c "import ast; ast.parse(open('app.py', encoding='utf-8').read()); print('app.py parses')"
D:\Bokey\Sound_Quality\.venv\Scripts\python.exe -m pytest -q
```
Expected: `app.py parses`; full suite PASS (127 passed). Then a manual check: `streamlit run app.py` —
hovering an export button shows the format tooltip; the advanced RIR uploader accepts .wav and .flac.

- [ ] **Step 8: Commit**

```bash
git add src/i18n.py app.py tests/test_i18n.py
git commit -m "feat: export format tooltips + lossless (wav/flac) RIR upload"
```

---

## Self-Review notes

- **Spec coverage:** help strings (Step 3) on every export button via `help=` (Step 5, all four formats,
  CSV shared); RIR uploader accepts wav/flac and reads via `decode_audio` (Step 6); both-language keys
  tested (Step 1) with parity guarded by the existing test. Non-goals (captions/expander, lossy RIR)
  respected.
- **Type consistency:** the four `help_*` keys added in Step 3 match the `help=t(lang, "help_*")` args in
  Step 5; `decode_audio(up.read())` returns `(signal, sr)`, matching the existing `rir, sr = ...` unpack
  and the `rir.ndim > 1` mono step.
- **Placeholder scan:** every step shows exact code/strings; no deferred content.
