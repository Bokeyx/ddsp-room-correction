# Public-friendly app: preset rooms + your own music — design

- **Date:** 2026-06-19
- **Status:** approved (brainstorming)
- **Scope:** the Streamlit app (`app.py`) + two small pure helpers. Sub-project 4.

## Motivation

The deployed app is faithful to the engineering but not to a general visitor. It asks for a **room
impulse response** (a measurement most people don't have) and only auditions a **pink-noise sample**. Two
changes make it approachable: let people pick a **named example room** instead of facing seed/RT60
sliders or a RIR upload, and let them hear **their own music** through that room, before vs after
correction.

### The honest constraint (why music plays *through a room*)

A correction EQ is the inverse of the room. Applying it to clean music in isolation pre-distorts the
sound — it does not sound "better." The audible improvement only appears when the music passes through
the room:

- before = music → room (coloured)
- after = music → correction → room (flat)

So the own-music feature reuses the existing in-room A/B path; it swaps the pink-noise source for the
user's clip. A real room measurement is still needed, which the preset rooms provide (simulated).

## Goals

- A pure `src/rooms.py` with friendly preset rooms backed by the existing synthetic RIR generator, so a
  visitor picks "Small bedroom" rather than tuning seed/RT60.
- A pure `audio.prepare_clip` helper that makes an arbitrary uploaded clip usable: mono, resampled to the
  room's sample rate, length-capped.
- `app.py` reworked so the default input is an example room + an optional "play your own music" upload;
  the A/B auditions that music (or pink noise if none) through the room, before vs corrected. RIR upload
  stays as an "advanced" option. New UI strings added to EN/KO `i18n`.

## Module: `src/rooms.py`

```python
PRESETS = {            # friendly name -> (rt60_s, seed)
    "Small bedroom": (0.3, 1),
    "Living room": (0.5, 2),
    "Large room": (0.7, 3),
    "Echoey hall": (0.9, 4),
}

def preset_names() -> list[str]:
    """Display order of the preset room names."""

def preset_rir(name, sr=48000, duration_s=0.5):
    """(rir, sr) for a named preset via decaying_noise_rir. Deterministic
    (fixed seed per name). Raises ValueError for an unknown name."""
```

- Pure, deterministic (reuses `synthetic.decaying_noise_rir(sr, duration_s, rt60, seed)`), no IO.
- Unknown name raises `ValueError` (caught/avoided by the app, which only offers known names).

## Helper: `audio.prepare_clip`

```python
def prepare_clip(signal, src_sr, target_sr, max_seconds=20.0):
    """Make an uploaded clip ready for in-room playback: average to mono if 2-D,
    resample to target_sr (scipy.signal.resample_poly) when src_sr != target_sr,
    and trim to the first max_seconds. Returns a 1-D float64 ndarray."""
```

- Mono via channel mean; resample via `resample_poly(sig, target_sr, src_sr)` reduced by gcd; trim last.
- The length cap protects Streamlit Cloud's ~1 GB RAM / compute from a long upload.

## app.py rework

- **Input source** radio: `["Example room", "Upload my room (advanced)"]` (codes `example` / `upload`).
  - `example` → a selectbox of `preset_names()`; `rir, sr = preset_rir(name)`. No seed/RT60 sliders.
  - `upload` → the existing RIR WAV uploader (unchanged behaviour).
- **Your own music** (optional) `st.file_uploader` accepting `wav/flac/ogg/mp3`. If provided, read with
  `soundfile` inside a try/except; on failure show a friendly error and fall back to pink noise. The
  decoded clip goes through `prepare_clip(..., target_sr=sr)`.
- **A/B section**: source = the prepared music if uploaded, else `pink_noise`. Unchanged downstream:
  convolve with the room (before) vs apply correction then convolve (after), normalize, two players.
- **i18n**: add keys for the new labels (example-room label, room-picker, music uploader, the "simulated
  room" caption, the read-error message) to both `en` and `ko` tables. Method ids / units stay untranslated.

## Testing

`tests/test_rooms.py`:
- `preset_names()` returns the expected names; each is a key of `PRESETS`.
- `preset_rir("Small bedroom")` returns a non-empty 1-D array and `sr == 48000`, deterministically (two
  calls equal).
- `preset_rir("nope")` raises `ValueError`.

`tests/test_audio.py` (extend or new):
- `prepare_clip` stereo (shape (n,2)) -> 1-D mono of the same length when src_sr == target_sr.
- `prepare_clip` resamples: src_sr 24000 -> target 48000 roughly doubles the length.
- `prepare_clip` trims a 30 s mono clip at 48000 to 20 s (== `20*48000` samples).
- `prepare_clip` mono passthrough when src_sr == target_sr and already short.

`tests/test_i18n.py` key-parity test already guards that en/ko stay in sync after new keys are added.

Suite: 112 -> ~118.

## Non-goals

- No stereo A/B output (mono is enough to hear the tonal change). No waveform display, no real-time
  streaming, no uncapped clip length. No new datasets / bundled measured RIRs (presets are synthetic).
- MP3 is accepted but not guaranteed — it depends on the server's libsndfile build; failure degrades
  gracefully to the error message + pink-noise fallback rather than crashing.

## Success criteria

- A visitor can pick an example room, upload a song, and hear an audible before/after without touching a
  slider or knowing what a RIR is — and still download the EQ.
- `rooms.py` and `prepare_clip` are pure and pass the tests above; the full suite stays green.
