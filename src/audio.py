"""M6c: apply a designed correction to actual audio (time domain).

The frequency-domain stages (classic EQ, FIR) decide *what* correction to make;
this module renders that correction onto a real signal so it can be heard in an
A/B listening demo. The peaking biquads reuse `peaking_coeffs` (same source of
truth as the dB response), so the audio you hear matches the curves you design.
"""

import io
import os
import subprocess
import tempfile
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly, sosfilt, tf2sos

from src.eq_classic import peaking_coeffs


def apply_eq_to_signal(filters, signal, sr):
    """Apply a cascade of peaking biquads to `signal` (same length out).

    Empty filter list -> a copy of the signal (no correction = no change).
    """
    signal = np.asarray(signal, dtype=np.float64)
    if not filters:
        return signal.copy()

    sos = np.vstack([tf2sos(*peaking_coeffs(f, sr)) for f in filters])
    return sosfilt(sos, signal)


def apply_fir_to_signal(taps, signal):
    """Convolve `signal` with FIR `taps`, preserving length (mode='same')."""
    taps = np.asarray(taps, dtype=np.float64)
    signal = np.asarray(signal, dtype=np.float64)
    return np.convolve(signal, taps, mode="same")


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


def _midi_to_freq(m):
    return 440.0 * 2.0 ** ((m - 69) / 12.0)


def _harmonic_stack(freq, n_samples, sr, n_harm):
    """A bright harmonic tone (1/h amplitudes) of length n_samples."""
    t = np.arange(n_samples) / sr
    return sum((1.0 / h) * np.sin(2.0 * np.pi * freq * h * t) for h in range(1, n_harm + 1))


def _plucked(freq, dur, sr, n_harm=12, decay=3.0):
    """Plucked note: bright harmonic stack with a fast attack and decay."""
    n = int(dur * sr)
    t = np.arange(n) / sr
    env = np.minimum(t / 0.008, 1.0) * np.exp(-decay * t / dur)
    return env * _harmonic_stack(freq, n, sr, n_harm)


def _sustained(freq, dur, sr, n_harm=4):
    """Sustained note (pad/bass): soft attack and release, level body."""
    n = int(dur * sr)
    t = np.arange(n) / sr
    env = np.minimum(t / 0.05, 1.0) * np.clip((dur - t) / 0.12, 0.0, 1.0)
    return env * _harmonic_stack(freq, n, sr, n_harm)


def demo_music(sr, duration_s=10.0):
    """A short, license-clean synthesized music clip for the A/B listening demo.

    An Am-F-C-G progression layered as bass + sustained pad + strummed plucks +
    a bright melody, so the signal carries energy from the low end to the highs
    -- that breadth is what makes the EQ correction audible. Fully generated
    (no samples, no RNG), so it is deterministic and safe to commit.
    """
    chords = [
        (45, [57, 60, 64], [69, 72, 76]),  # Am
        (41, [53, 57, 60], [65, 69, 72]),  # F
        (48, [60, 64, 67], [72, 76, 79]),  # C
        (43, [55, 59, 62], [67, 71, 74]),  # G
    ]
    seg = duration_s / len(chords)
    seg_len = int(seg * sr)
    strum = int(0.04 * sr)  # gap between strummed notes
    out = np.zeros(len(chords) * seg_len + sr)  # +1 s tail headroom

    def add(segment, start):
        end = min(start + len(segment), len(out))
        out[start:end] += segment[: end - start]

    for c, (root, triad, melody) in enumerate(chords):
        base = c * seg_len
        add(_sustained(_midi_to_freq(root), seg, sr, n_harm=3), base)          # bass
        for m in triad:
            add(0.4 * _sustained(_midi_to_freq(m), seg, sr, n_harm=4), base)   # pad
        for i, m in enumerate(triad):
            add(0.7 * _plucked(_midi_to_freq(m), seg, sr), base + i * strum)   # strum
        step = seg / len(melody)
        for j, m in enumerate(melody):
            add(0.5 * _plucked(_midi_to_freq(m), step * 1.2, sr, decay=2.5),
                base + int(j * step * sr))                                     # melody

    out = out[: len(chords) * seg_len]
    peak = np.max(np.abs(out))
    if peak > 0:
        out = out / peak * 0.9
    return out


def pink_noise(n, seed):
    """Reproducible 1/f pink noise, n samples, normalised to max|x| ~= 0.99.

    White noise is shaped in the frequency domain by 1/sqrt(f) (power ~ 1/f),
    with the DC bin guarded (f=0 set to 0) to avoid division by zero / a DC
    offset.
    """
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(n)

    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)

    scale = np.ones_like(freqs)
    scale[1:] = 1.0 / np.sqrt(freqs[1:])
    scale[0] = 0.0  # guard f=0: drop DC

    pink = np.fft.irfft(spectrum * scale, n=n)

    peak = np.max(np.abs(pink))
    if peak > 0:
        pink = pink / peak * 0.99
    return pink
