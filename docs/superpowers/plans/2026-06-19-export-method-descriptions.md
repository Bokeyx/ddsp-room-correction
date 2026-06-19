# Export Method Descriptions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a one-line intro and a per-method description to the app's Export correction section.

**Architecture:** Pure UI-copy change: new EN/KO i18n strings rendered as `st.caption` in the export section. No new modules or logic.

**Tech Stack:** Python 3, Streamlit, pytest. Python: `D:\Bokey\Sound_Quality\.venv\Scripts\python.exe`. Tests: `python -m pytest -q`.

## Global Constraints

- Single-author commits. NEVER add a `Co-Authored-By` trailer.
- All repo content in English (English default + Korean translation).
- `STRINGS["en"]` and `STRINGS["ko"]` keep identical key sets (enforced by `test_i18n`).
- Method ids are always one of `classic` / `ddsp` / `fir`; the description key is `desc_{m}`.

---

### Task 1: Export-section copy (i18n strings + captions)

**Files:**
- Modify: `src/i18n.py`, `app.py`
- Test: `tests/test_i18n.py`

**Interfaces:**
- Consumes: existing `t`, `LANGUAGES`, the export loop in `app.py`.
- Produces: i18n keys `export_intro`, `desc_classic`, `desc_ddsp`, `desc_fir` (both languages).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_i18n.py`:

```python
def test_export_description_keys_present_both_languages():
    for key in ("export_intro", "desc_classic", "desc_ddsp", "desc_fir"):
        assert key in STRINGS["en"], key
        assert key in STRINGS["ko"], key
```

- [ ] **Step 2: Run test to verify it fails**

Run: `D:\Bokey\Sound_Quality\.venv\Scripts\python.exe -m pytest tests/test_i18n.py::test_export_description_keys_present_both_languages -q`
Expected: FAIL — `KeyError`/`assert` on `export_intro` (key not yet added).

- [ ] **Step 3: Add the strings to both i18n tables**

In `src/i18n.py`, in the `"en"` dict, after the `"export_header": "Export correction",` line add:

```python
        "export_intro": "Download the correction in formats real audio tools read.",
        "desc_classic": "Classic rule-based EQ — simple and predictable.",
        "desc_ddsp": "ML-optimized EQ — usually the flattest, still interpretable.",
        "desc_fir": "FIR filter — very precise but a black box.",
```

In the `"ko"` dict, after the `"export_header": "보정 내보내기",` line add:

```python
        "export_intro": "보정 결과를 실제 오디오 도구가 읽는 형식으로 내려받으세요.",
        "desc_classic": "고전 규칙 기반 EQ — 단순하고 예측 가능.",
        "desc_ddsp": "ML 최적화 EQ — 보통 가장 평평하고, 그래도 해석 가능.",
        "desc_fir": "FIR 필터 — 매우 정밀하지만 블랙박스.",
```

- [ ] **Step 4: Run the i18n tests to verify they pass**

Run: `D:\Bokey\Sound_Quality\.venv\Scripts\python.exe -m pytest tests/test_i18n.py -q`
Expected: PASS (6 passed — new test + the 5 existing, including key-parity).

- [ ] **Step 5: Render the captions in the export section**

In `app.py`, the export section currently reads:

```python
# --- export the correction ---
st.subheader(t(lang, "export_header"))
for m in methods:
    corrected_db, corr, _ = results[m]
    st.markdown(f"**{m}**")
    cdl = st.columns(3)
```

Change it to:

```python
# --- export the correction ---
st.subheader(t(lang, "export_header"))
st.caption(t(lang, "export_intro"))
for m in methods:
    corrected_db, corr, _ = results[m]
    st.markdown(f"**{m}**")
    st.caption(t(lang, f"desc_{m}"))
    cdl = st.columns(3)
```

- [ ] **Step 6: Smoke-check and run the full suite**

Run:
```bash
D:\Bokey\Sound_Quality\.venv\Scripts\python.exe -c "import ast; ast.parse(open('app.py', encoding='utf-8').read()); print('app.py parses')"
D:\Bokey\Sound_Quality\.venv\Scripts\python.exe -m pytest -q
```
Expected: `app.py parses`; full suite PASS (123 passed — one new i18n test on top of 122). Then a manual
check: `streamlit run app.py` shows the intro line and a one-liner under each of classic/ddsp/fir.

- [ ] **Step 7: Commit**

```bash
git add src/i18n.py app.py tests/test_i18n.py
git commit -m "feat: per-method descriptions in the export section"
```

---

## Self-Review notes

- **Spec coverage:** export_intro + desc_classic/ddsp/fir strings (Step 3), rendered as captions
  (Step 5), tested for presence in both languages (Step 1) with parity guarded by the existing test.
- **Type consistency:** the loop uses `t(lang, f"desc_{m}")` and `m` ∈ {classic, ddsp, fir}, matching the
  three `desc_*` keys added in Step 3. `export_intro` is rendered once under the subheader.
- **Placeholder scan:** every step shows exact code/strings; no deferred content.
