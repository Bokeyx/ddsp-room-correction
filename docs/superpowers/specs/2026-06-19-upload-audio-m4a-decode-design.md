# Robust upload decoding: m4a/aac via ffmpeg — design

- **Date:** 2026-06-19
- **Status:** approved (brainstorming)
- **Scope:** the app's uploaded-music decode path + one pure-ish helper. Notebook untouched.

## Motivation

The own-music feature reads uploads with `soundfile`. libsndfile 1.2.2 (local and on Streamlit Cloud)
already decodes WAV/FLAC/OGG/MP3, so mp3 is **not** broken. The real gap is formats libsndfile cannot
read — notably **m4a/aac**, which is what phones and Apple devices most commonly produce. A general
visitor dropping their own song will often hit exactly that. We add an ffmpeg fallback so those decode
too, while keeping the fast soundfile path for everything it already handles.

## Goals

- A `decode_audio(raw_bytes)` helper: try `soundfile` first; on failure, transcode with a bundled ffmpeg
  binary (`imageio-ffmpeg`) and read the result; raise a clear error if both fail.
- The app uses `decode_audio` then the existing `prepare_clip`, and accepts m4a/aac/opus uploads. The
  existing "couldn't read → warn + pink noise" fallback stays.

## Why imageio-ffmpeg

`imageio-ffmpeg` ships the ffmpeg binary inside the pip wheel (`get_ffmpeg_exe()`), so there is no apt
package and no runtime download — important for a reliable Streamlit Cloud deploy. It is imported lazily,
only inside the fallback branch, so the soundfile path has no extra dependency.

## Helper: `audio.decode_audio`

```python
def decode_audio(raw_bytes):
    """Decode uploaded audio bytes to (signal, sr).

    Tries soundfile first (WAV/FLAC/OGG/MP3 via libsndfile). On failure, writes
    the bytes to a temp file and transcodes to WAV with the bundled ffmpeg
    (imageio-ffmpeg) — this covers m4a/aac/opus etc. that libsndfile can't read.
    Raises ValueError if neither path can decode. Returns the raw decoded signal
    (possibly multi-channel) and its sample rate; mono/resample/trim are left to
    prepare_clip.
    """
```

- soundfile path: `sf.read(io.BytesIO(raw_bytes), dtype="float64")`.
- ffmpeg path: write bytes to a `NamedTemporaryFile` (m4a needs a seekable input, so a temp file, not a
  stdin pipe), run `<ffmpeg> -y -i <in> <out>.wav`, `sf.read` the wav, clean up both temp files in a
  `finally`. `imageio_ffmpeg` is imported inside this branch; if its import or the transcode fails, raise
  `ValueError` with a clear message.

## app.py changes

- Replace the inline `clip, clip_sr = sf.read(io.BytesIO(music_up.read()), dtype="float64")` with
  `clip, clip_sr = decode_audio(music_up.read())`; keep the surrounding try/except that warns and falls
  back to pink noise.
- Uploader `type=["wav", "flac", "ogg", "mp3", "m4a", "aac", "opus"]`.
- Import `decode_audio` from `src.audio`.

## Dependencies

Add `imageio-ffmpeg` (pinned) to `requirements.txt`.

## Testing (`tests/test_audio.py`)

- `decode_audio` on valid WAV bytes (written via `soundfile` to a buffer) returns a finite array and the
  correct sample rate — exercises the soundfile path, no ffmpeg needed.
- `decode_audio` on garbage bytes raises `ValueError` (both paths fail).
- ffmpeg integration: **skipped** unless an ffmpeg binary is importable
  (`pytest.importorskip("imageio_ffmpeg")` + a guarded `get_ffmpeg_exe()`); when present, synthesize a
  short sine, encode it to `.m4a` (AAC) with that binary, then assert `decode_audio` returns a non-empty
  signal. This keeps the suite green where no binary/network exists while still covering the real path.

Suite: 123 -> ~125 (one test may skip).

## Non-goals

- No stereo preservation (prepare_clip averages to mono), no video, no streaming, no ffmpeg filter
  tuning, no apt ffmpeg. Notebook unchanged.

## Success criteria

- An m4a/aac upload plays through the in-room A/B (decoded via the ffmpeg fallback); WAV/MP3/FLAC/OGG
  still take the fast soundfile path; an undecodable file degrades gracefully to the warning + pink
  noise. `decode_audio` is covered by tests; the suite stays green.
