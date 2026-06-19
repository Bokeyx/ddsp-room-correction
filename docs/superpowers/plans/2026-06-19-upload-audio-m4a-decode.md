# Robust Upload Decoding (m4a/aac) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decode uploaded music that libsndfile can't read (m4a/aac/opus) via a bundled ffmpeg fallback, keeping the fast soundfile path for everything else.

**Architecture:** A new `audio.decode_audio(raw_bytes)` tries `soundfile` first, then transcodes with the `imageio-ffmpeg` bundled binary. The app calls `decode_audio` then the existing `prepare_clip`; the existing warn→pink-noise fallback stays.

**Tech Stack:** Python 3, numpy, soundfile, imageio-ffmpeg (bundled ffmpeg binary), Streamlit, pytest. Python: `D:\Bokey\Sound_Quality\.venv\Scripts\python.exe`. Tests: `python -m pytest -q`.

## Global Constraints

- Single-author commits. NEVER add a `Co-Authored-By` trailer.
- All repo content in English.
- `decode_audio` returns the raw decoded signal (possibly multi-channel) + sample rate; mono/resample/
  trim stay in `prepare_clip`.
- `imageio_ffmpeg` is imported lazily inside the fallback branch only (the soundfile path needs no extra
  dependency at import time).
- m4a needs a seekable input, so the ffmpeg fallback writes a temp file (not a stdin pipe).
- Notebook is not touched.

---

### Task 1: `audio.decode_audio` — soundfile-then-ffmpeg decoder

**Files:**
- Modify: `src/audio.py`
- Test: `tests/test_audio.py`

**Interfaces:**
- Consumes: numpy, soundfile, `imageio_ffmpeg` (lazy), stdlib subprocess/tempfile/os/io.
- Produces: `decode_audio(raw_bytes) -> (np.ndarray, int)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_audio.py` (add `import io`, `import pytest` at the top if not present, and
`decode_audio` to the existing `from src.audio import ...` line):

```python
def test_decode_audio_wav_via_soundfile():
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, np.linspace(-0.5, 0.5, 1000), 44100, format="WAV")
    sig, sr = decode_audio(buf.getvalue())
    assert sr == 44100
    assert len(sig) == 1000
    assert np.all(np.isfinite(sig))


def test_decode_audio_garbage_raises():
    with pytest.raises(ValueError):
        decode_audio(b"this is not audio at all")


def test_decode_audio_m4a_via_ffmpeg(tmp_path):
    imageio_ffmpeg = pytest.importorskip("imageio_ffmpeg")
    import subprocess
    import soundfile as sf
    try:
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pytest.skip("no ffmpeg binary available")

    wav = tmp_path / "tone.wav"
    sr = 44100
    t = np.linspace(0, 0.5, int(0.5 * sr), endpoint=False)
    sf.write(str(wav), 0.2 * np.sin(2 * np.pi * 440 * t), sr, format="WAV")
    m4a = tmp_path / "tone.m4a"
    proc = subprocess.run([exe, "-y", "-i", str(wav), str(m4a)], capture_output=True)
    if proc.returncode != 0 or not m4a.exists():
        pytest.skip("ffmpeg could not produce m4a in this environment")

    sig, out_sr = decode_audio(m4a.read_bytes())
    assert len(sig) > 0
    assert out_sr > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `D:\Bokey\Sound_Quality\.venv\Scripts\python.exe -m pytest tests/test_audio.py -k decode_audio -q`
Expected: FAIL — `ImportError: cannot import name 'decode_audio'`.

- [ ] **Step 3: Implement `decode_audio`**

At the top of `src/audio.py`, add the imports it needs (the module currently imports only `gcd`, numpy,
`scipy.signal`, and `src.eq_classic`):

```python
import io
import os
import subprocess
import tempfile

import soundfile as sf
```

Then append the function:

```python
def decode_audio(raw_bytes):
    """Decode uploaded audio bytes to ``(signal, sr)``.

    Tries soundfile first (WAV/FLAC/OGG/MP3 via libsndfile). On failure, writes
    the bytes to a temp file and transcodes to WAV with the bundled ffmpeg
    (imageio-ffmpeg) -- this covers m4a/aac/opus etc. that libsndfile can't read.
    Raises ValueError if neither path can decode. Mono/resample/trim are left to
    prepare_clip.
    """
    try:
        return sf.read(io.BytesIO(raw_bytes), dtype="float64")
    except Exception:
        pass  # fall through to the ffmpeg path

    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise ValueError("could not decode audio (no ffmpeg available)") from exc

    fin = tempfile.NamedTemporaryFile(delete=False)
    fout = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    fout.close()
    try:
        fin.write(raw_bytes)
        fin.close()
        proc = subprocess.run([exe, "-y", "-i", fin.name, fout.name], capture_output=True)
        if proc.returncode != 0:
            raise ValueError("could not decode audio (ffmpeg failed)")
        return sf.read(fout.name, dtype="float64")
    finally:
        for path in (fin.name, fout.name):
            try:
                os.unlink(path)
            except OSError:
                pass
```

- [ ] **Step 4: Dev-install imageio-ffmpeg and run the tests**

Run:
```bash
D:\Bokey\Sound_Quality\.venv\Scripts\python.exe -m pip install imageio-ffmpeg
D:\Bokey\Sound_Quality\.venv\Scripts\python.exe -m pytest tests/test_audio.py -k decode_audio -q
```
Expected: PASS (the wav and garbage tests pass; the m4a test passes if the binary downloads/exists,
otherwise it `skips`). If `pip install` has no network, the m4a test skips and the other two still pass.

- [ ] **Step 5: Commit**

```bash
git add src/audio.py tests/test_audio.py
git commit -m "feat: decode_audio (soundfile + ffmpeg fallback for m4a/aac)"
```

---

### Task 2: Wire decode_audio into the app + accept m4a/aac + pin the dep

**Files:**
- Modify: `app.py`, `requirements.txt`

**Interfaces:**
- Consumes: `decode_audio` (Task 1); existing `prepare_clip`, `pink_noise`, `t`.
- Produces: a reworked app (no importable symbols for later tasks).

- [ ] **Step 1: Import decode_audio in app.py**

Change the audio import line in `app.py` to include `decode_audio`:

```python
from src.audio import apply_eq_to_signal, apply_fir_to_signal, pink_noise, prepare_clip, decode_audio
```

- [ ] **Step 2: Use decode_audio for the uploaded clip**

In `app.py`, in the A/B section, replace:

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

with:

```python
    if music_up is not None:
        try:
            clip, clip_sr = decode_audio(music_up.read())
            dry = prepare_clip(clip, clip_sr, sr)
        except Exception:
            st.warning(t(lang, "music_error"))
            dry = pink_noise(int(2.0 * sr), seed=0)
    else:
        dry = pink_noise(int(2.0 * sr), seed=0)
```

- [ ] **Step 3: Accept m4a/aac/opus in the music uploader**

In `app.py`, change the music uploader's `type=` list:

```python
music_up = st.sidebar.file_uploader(
    t(lang, "music_uploader"), type=["wav", "flac", "ogg", "mp3", "m4a", "aac", "opus"]
)
```

- [ ] **Step 4: Pin imageio-ffmpeg in requirements.txt**

Find the installed version, then add the pinned line:
```bash
D:\Bokey\Sound_Quality\.venv\Scripts\python.exe -c "import imageio_ffmpeg as i; print(i.__version__)"
```
Add to `requirements.txt` (using the version printed above, e.g. if it prints `0.6.0`):
```
imageio-ffmpeg==<printed version>
```

- [ ] **Step 5: Smoke-check and run the full suite**

Run:
```bash
D:\Bokey\Sound_Quality\.venv\Scripts\python.exe -c "import ast; ast.parse(open('app.py', encoding='utf-8').read()); print('app.py parses')"
D:\Bokey\Sound_Quality\.venv\Scripts\python.exe -m pytest -q
```
Expected: `app.py parses`; full suite PASS (~125 passed, possibly 1 skipped for the m4a test). Then a
manual check: `streamlit run app.py`, upload an .m4a song, and confirm the A/B players use it.

- [ ] **Step 6: Commit**

```bash
git add app.py requirements.txt
git commit -m "feat: accept m4a/aac uploads via decode_audio; pin imageio-ffmpeg"
```

---

## Self-Review notes

- **Spec coverage:** `decode_audio` soundfile→ffmpeg with temp-file input and clear ValueError (T1);
  app wiring, m4a/aac/opus accepted, imageio-ffmpeg pinned (T2). Non-goals (stereo, video, apt ffmpeg)
  respected.
- **Type consistency:** `decode_audio(raw_bytes) -> (signal, sr)` matches the T2 call site
  `clip, clip_sr = decode_audio(music_up.read())`, then `prepare_clip(clip, clip_sr, sr)` (existing
  signature). Lazy `import imageio_ffmpeg` only in the fallback.
- **Test robustness:** the m4a test uses `importorskip` + guarded `get_ffmpeg_exe()`/return-code skips,
  so the suite stays green where no binary/network exists; the soundfile and garbage tests always run.
- **Placeholder scan:** the only fill-in is the imageio-ffmpeg version, resolved by the install command
  in T2 Step 4 — a concrete value, not deferred design.
