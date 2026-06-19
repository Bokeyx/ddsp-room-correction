# Public-Friendly Rooms + Own-Music Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a visitor pick a named example room and audition their own uploaded music through it (before vs corrected), instead of facing RIR/seed jargon and only a pink-noise sample.

**Architecture:** Two pure helpers — `src/rooms.py` (friendly presets over the synthetic RIR generator) and `audio.prepare_clip` (mono + resample + length-cap an uploaded clip). `app.py` defaults to an example-room picker plus an optional music uploader, and the existing in-room A/B auditions that music (or pink noise). New UI strings added to the EN/KO i18n tables.

**Tech Stack:** Python 3, numpy, scipy (`scipy.signal.resample_poly`), soundfile, Streamlit, pytest. Python executable: `D:\Bokey\Sound_Quality\.venv\Scripts\python.exe`. Tests: `python -m pytest -q`.

## Global Constraints

- Single-author commits. NEVER add a `Co-Authored-By` trailer.
- All repo content in English (UI strings get an EN default + KO translation).
- `src/rooms.py` and `audio.prepare_clip` are PURE: no Streamlit, no file IO. Return ndarrays / tuples.
- `decaying_noise_rir(sr, duration_s, rt60_s, seed)` returns `(rir, sr)` (from `src/synthetic.py`).
- i18n: `STRINGS["en"]` and `STRINGS["ko"]` must keep identical key sets (enforced by `test_i18n`).
- Notebook is not touched by this plan.

---

### Task 1: `src/rooms.py` — friendly preset rooms

**Files:**
- Create: `src/rooms.py`
- Test: `tests/test_rooms.py`

**Interfaces:**
- Consumes: `decaying_noise_rir` from `src.synthetic`.
- Produces: `preset_names() -> list[str]`, `preset_rir(name, sr=48000, duration_s=0.5) -> (ndarray, int)`,
  module dict `PRESETS`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rooms.py
import numpy as np
import pytest

from src.rooms import PRESETS, preset_names, preset_rir


def test_preset_names_are_keys_of_presets():
    names = preset_names()
    assert names == list(PRESETS)
    assert "Small bedroom" in names


def test_preset_rir_is_deterministic_48k():
    rir, sr = preset_rir("Small bedroom")
    assert sr == 48000
    assert rir.ndim == 1 and len(rir) > 0
    rir2, _ = preset_rir("Small bedroom")
    assert np.array_equal(rir, rir2)


def test_preset_rir_unknown_name_raises():
    with pytest.raises(ValueError):
        preset_rir("not a room")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `D:\Bokey\Sound_Quality\.venv\Scripts\python.exe -m pytest tests/test_rooms.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.rooms'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/rooms.py
"""Friendly preset rooms for the app, backed by the synthetic RIR generator.

A general visitor has no room measurement, so the app offers named rooms instead
of seed/RT60 knobs. Each name maps to a deterministic synthetic RIR. Pure: no IO,
no Streamlit. The rooms are simulated, and the app labels them as such.
"""
from src.synthetic import decaying_noise_rir

# friendly name -> (rt60_s, seed)
PRESETS = {
    "Small bedroom": (0.3, 1),
    "Living room": (0.5, 2),
    "Large room": (0.7, 3),
    "Echoey hall": (0.9, 4),
}


def preset_names():
    """Display order of the preset room names."""
    return list(PRESETS)


def preset_rir(name, sr=48000, duration_s=0.5):
    """Return ``(rir, sr)`` for a named preset. Deterministic (fixed seed per
    name). Raises ValueError for an unknown name."""
    if name not in PRESETS:
        raise ValueError(f"unknown preset room: {name!r}; choose from {preset_names()}")
    rt60, seed = PRESETS[name]
    return decaying_noise_rir(sr, duration_s, rt60, seed=seed)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `D:\Bokey\Sound_Quality\.venv\Scripts\python.exe -m pytest tests/test_rooms.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/rooms.py tests/test_rooms.py
git commit -m "feat: preset rooms (friendly named synthetic RIRs)"
```

---

### Task 2: `audio.prepare_clip` — make an uploaded clip playable

**Files:**
- Modify: `src/audio.py`
- Test: `tests/test_audio.py`

**Interfaces:**
- Consumes: numpy, `scipy.signal.resample_poly`.
- Produces: `prepare_clip(signal, src_sr, target_sr, max_seconds=20.0) -> np.ndarray` (1-D float64).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_audio.py  (create if absent; otherwise append these + the import)
import numpy as np

from src.audio import prepare_clip


def test_prepare_clip_stereo_to_mono_same_sr():
    stereo = np.ones((100, 2), dtype=np.float64)
    out = prepare_clip(stereo, 48000, 48000)
    assert out.ndim == 1
    assert len(out) == 100


def test_prepare_clip_resamples_up():
    mono = np.zeros(1000, dtype=np.float64)
    out = prepare_clip(mono, 24000, 48000)
    assert len(out) == 2000  # 24k -> 48k doubles the sample count


def test_prepare_clip_trims_to_max_seconds():
    mono = np.zeros(30 * 48000, dtype=np.float64)
    out = prepare_clip(mono, 48000, 48000, max_seconds=20.0)
    assert len(out) == 20 * 48000


def test_prepare_clip_short_mono_passthrough():
    mono = np.linspace(-1.0, 1.0, 500)
    out = prepare_clip(mono, 48000, 48000)
    assert np.array_equal(out, mono)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `D:\Bokey\Sound_Quality\.venv\Scripts\python.exe -m pytest tests/test_audio.py -q`
Expected: FAIL — `ImportError: cannot import name 'prepare_clip'`

- [ ] **Step 3: Write minimal implementation**

At the top of `src/audio.py`, add to the scipy import (the file already imports from `scipy.signal`):

```python
from math import gcd

from scipy.signal import resample_poly
```

Then append the function:

```python
def prepare_clip(signal, src_sr, target_sr, max_seconds=20.0):
    """Make an uploaded clip ready for in-room playback: average to mono if 2-D,
    resample to ``target_sr`` (when it differs from ``src_sr``), then trim to the
    first ``max_seconds``. Returns a 1-D float64 ndarray. The length cap protects
    the deployed app's limited RAM/compute from a long upload.
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    src_sr, target_sr = int(src_sr), int(target_sr)
    if src_sr != target_sr:
        g = gcd(src_sr, target_sr)
        signal = resample_poly(signal, target_sr // g, src_sr // g)
    return signal[: int(max_seconds * target_sr)]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `D:\Bokey\Sound_Quality\.venv\Scripts\python.exe -m pytest tests/test_audio.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/audio.py tests/test_audio.py
git commit -m "feat: prepare_clip (mono + resample + length-cap an uploaded clip)"
```

---

### Task 3: app.py rework (preset rooms + own-music A/B) + i18n

**Files:**
- Modify: `src/i18n.py`
- Modify: `app.py`

**Interfaces:**
- Consumes: `preset_names`, `preset_rir` (Task 1); `prepare_clip` (Task 2); existing `pink_noise`,
  `apply_eq_to_signal`, `apply_fir_to_signal`, `correct`, `t`, `LANGUAGES`.
- Produces: a reworked app (no importable symbols for later tasks).

No unit test (Streamlit glue); verified by an import-parse smoke check, the i18n key-parity test, and the
full suite staying green.

- [ ] **Step 1: Update the i18n tables (both languages, same keys)**

In `src/i18n.py`, in the `"en"` dict: remove the `"synthetic"`, `"seed"`, `"rt60"` entries; change
`"upload"` and `"ab_header"`; and add the new keys. The `"en"` block's changed/added entries:

```python
        "data_label": "Data",
        "example": "Example room",
        "upload": "Upload my room (advanced)",
        "room_picker": "Pick a room",
        "room_caption": "Simulated room — pick one, then play your own music below.",
        "music_uploader": "Play your own music (optional)",
        "music_error": "Couldn't read that audio — try WAV or FLAC. Using pink noise instead.",
        "uploader": "RIR WAV",
        "upload_info": "Upload a RIR WAV file.",
        ...
        "ab_header": "A/B listening (through the room)",
```

Make the **same** edits in the `"ko"` dict (remove `synthetic`/`seed`/`rt60`; change `upload`/`ab_header`;
add the five new keys):

```python
        "data_label": "데이터",
        "example": "예시 방",
        "upload": "내 방 업로드 (고급)",
        "room_picker": "방 선택",
        "room_caption": "시뮬레이션 방 — 하나 고른 뒤 아래에서 내 음악을 재생하세요.",
        "music_uploader": "내 음악 재생 (선택)",
        "music_error": "이 오디오를 읽지 못했어요 — WAV/FLAC로 올려주세요. 대신 핑크노이즈를 씁니다.",
        "uploader": "RIR WAV 파일",
        "upload_info": "RIR WAV 파일을 업로드하세요.",
        ...
        "ab_header": "A/B 청취 (방 통과)",
```

- [ ] **Step 2: Run the i18n parity test**

Run: `D:\Bokey\Sound_Quality\.venv\Scripts\python.exe -m pytest tests/test_i18n.py -q`
Expected: PASS (5 passed) — confirms en/ko still have identical key sets after the edits.

- [ ] **Step 3: Swap the imports in app.py**

In `app.py`, replace the synthetic import with the rooms import and add `prepare_clip`:

```python
from src.audio import apply_eq_to_signal, apply_fir_to_signal, pink_noise, prepare_clip
```
```python
from src.rooms import preset_names, preset_rir
```
Remove the line `from src.synthetic import decaying_noise_rir` (no longer used).

- [ ] **Step 4: Replace the input-source block**

In `app.py`, replace the whole `# --- sidebar: input RIR ---` block:

```python
# --- sidebar: input RIR ---
st.sidebar.header(t(lang, "input_header"))
source = st.sidebar.radio(t(lang, "data_label"), ["synthetic", "upload"],
                          format_func=lambda k: t(lang, k))
if source == "synthetic":
    seed = st.sidebar.slider(t(lang, "seed"), 0, 100, 42)
    rt60 = st.sidebar.slider(t(lang, "rt60"), 0.1, 1.0, 0.4, 0.1)
    rir, sr = decaying_noise_rir(48000, 0.5, rt60, seed=seed)
else:
    up = st.sidebar.file_uploader(t(lang, "uploader"), type=["wav"])
    if up is None:
        st.info(t(lang, "upload_info"))
        st.stop()
    rir, sr = sf.read(io.BytesIO(up.read()), dtype="float64")
    if rir.ndim > 1:
        rir = rir.mean(axis=1)
```

with:

```python
# --- sidebar: room ---
st.sidebar.header(t(lang, "input_header"))
source = st.sidebar.radio(t(lang, "data_label"), ["example", "upload"],
                          format_func=lambda k: t(lang, k))
if source == "example":
    room = st.sidebar.selectbox(t(lang, "room_picker"), preset_names())
    rir, sr = preset_rir(room)
    st.sidebar.caption(t(lang, "room_caption"))
else:
    up = st.sidebar.file_uploader(t(lang, "uploader"), type=["wav"])
    if up is None:
        st.info(t(lang, "upload_info"))
        st.stop()
    rir, sr = sf.read(io.BytesIO(up.read()), dtype="float64")
    if rir.ndim > 1:
        rir = rir.mean(axis=1)
```

- [ ] **Step 5: Add the music uploader to the sidebar**

In `app.py`, immediately after the `methods = st.sidebar.multiselect(...)` block, add:

```python
music_up = st.sidebar.file_uploader(
    t(lang, "music_uploader"), type=["wav", "flac", "ogg", "mp3"]
)
```

- [ ] **Step 6: Use the uploaded music as the A/B source**

In `app.py`, in the A/B section, replace the single line:

```python
    dry = pink_noise(int(2.0 * sr), seed=0)
```

with:

```python
    if music_up is not None:
        try:
            clip, clip_sr = sf.read(io.BytesIO(music_up.read()), dtype="float64")
            dry = prepare_clip(clip, clip_sr, sr)
        except Exception:
            st.warning(t(lang, "music_error"))
            dry = pink_noise(int(2.0 * sr), seed=0)
    else:
        dry = pink_noise(int(2.0 * sr), seed=0)
```

- [ ] **Step 7: Smoke-check and run the full suite**

Run:
```bash
D:\Bokey\Sound_Quality\.venv\Scripts\python.exe -c "import ast; ast.parse(open('app.py', encoding='utf-8').read()); print('app.py parses')"
D:\Bokey\Sound_Quality\.venv\Scripts\python.exe -m pytest -q
```
Expected: `app.py parses`; full suite PASS (~119 passed — 3 rooms + 4 prepare_clip on top of 112; i18n
parity still green).

Then a manual check (user-run): `streamlit run app.py` — default shows an example-room picker and a
"play your own music" uploader; uploading a song makes the A/B players use it.

- [ ] **Step 8: Commit**

```bash
git add src/i18n.py app.py
git commit -m "feat: preset-room picker + own-music A/B in the app"
```

---

## Self-Review notes

- **Spec coverage:** preset rooms (T1), `prepare_clip` (T2), app rework — example-room default, RIR upload
  as advanced, optional music uploader, in-room A/B on the user's clip, graceful read-error fallback, and
  EN/KO strings (T3). Non-goals (stereo out, waveform, uncapped length, new datasets) are respected.
- **Type consistency:** `preset_rir` returns `(rir, sr)` used directly in app; `prepare_clip(clip, clip_sr,
  sr)` matches its signature `(signal, src_sr, target_sr, max_seconds)`; the source radio codes
  `"example"`/`"upload"` match the `t(lang, k)` keys added in T3 Step 1 and the `if source == "example"`
  branch in Step 4.
- **i18n parity:** the same keys are removed/changed/added in both `en` and `ko`, kept honest by the
  existing `test_i18n` parity test (T3 Step 2).
- **Placeholder scan:** no deferred decisions; every step shows the exact code.
