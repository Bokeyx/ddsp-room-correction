"""M6b: FIR room-correction filter (frequency-sampling, linear phase).

The third correction method after the classic peaking EQ (M3) and the DDSP
optimiser (M4). Instead of a handful of interpretable biquads, FIR approximates
the *inverse* of the room as a single many-tap filter. It can in principle
correct phase/time as well as magnitude, at the cost of interpretability
(black box). Here we design a MAGNITUDE-correction FIR via frequency sampling
so it slots into the same sigma comparison as the other two methods.

Same correction philosophy as the EQ baselines: smooth the response first (do
not chase raw-FFT noise), mean-centre in-band (sigma is gain-invariant), and
clamp the boost so deep nulls are not driven to infinite gain.
"""

import numpy as np
from scipy.signal import freqz

from src.analysis import fractional_octave_smooth
from src.metrics import band_mask


def fir_response_db(taps, freqs_hz, sr):
    """dB magnitude response of an FIR filter at the given frequencies."""
    taps = np.asarray(taps, dtype=np.float64)
    freqs_hz = np.asarray(freqs_hz, dtype=np.float64)

    _, h = freqz(taps, worN=freqs_hz, fs=sr)
    eps = 1e-12
    return 20.0 * np.log10(np.abs(h) + eps)


def design_fir_correction(
    response_db,
    target_db,
    freqs_hz,
    sr,
    n_taps=4097,
    smoothing_fraction=3,
    max_boost_db=12.0,
    fmin=20.0,
    fmax=20000.0,
):
    """Design a linear-phase magnitude-correction FIR by frequency sampling.

    Assumes `freqs_hz` is the rfft grid (np.fft.rfftfreq), so the desired
    half-spectrum can be inverted directly with irfft.

    Steps:
      1. Smooth the response (1/`smoothing_fraction` octave) so the FIR targets
         the broadband trend, not raw-FFT noise -- same reason as the EQ
         baselines.
      2. desired_db = target - smoothed response, mean-centred in-band (correct
         the shape, not the absolute level; sigma is gain-invariant).
      3. Clamp desired_db to [-max_boost_db, +max_boost_db] to stop deep nulls
         from demanding infinite boost, and force the correction smoothly to 0
         outside [fmin, fmax] so band edges do not ring.
      4. desired_lin = 10**(desired_db/20); irfft to a zero-phase impulse,
         fftshift to centre it (-> linear phase), then window+truncate to n_taps.
    """
    response_db = np.asarray(response_db, dtype=np.float64)
    target_db = np.asarray(target_db, dtype=np.float64)
    freqs_hz = np.asarray(freqs_hz, dtype=np.float64)

    if smoothing_fraction is not None:
        design_resp = fractional_octave_smooth(
            freqs_hz, response_db, fraction=smoothing_fraction
        )
    else:
        design_resp = response_db

    mask = band_mask(freqs_hz, fmin, fmax)

    # Desired correction (dB), mean-centred in-band so we match shape only.
    correction_db = target_db - design_resp
    correction_db = correction_db - correction_db[mask].mean()

    # Clamp the boost so deep nulls are not driven to infinite gain.
    correction_db = np.clip(correction_db, -max_boost_db, max_boost_db)

    # Outside the audible band the correction is undefined / not measured, so
    # set it to 0 dB there. The hard step at the band edge is acceptable
    # because the time-domain Hann window applied below smooths the resulting
    # impulse response (a sharp spectral edge maps to gentle time-domain
    # ringing, which the window suppresses).
    correction_db = np.where(mask, correction_db, 0.0)

    # rfft grid -> linear amplitude half-spectrum (zero phase).
    desired_lin = 10.0 ** (correction_db / 20.0)

    # Length of the full (real) signal that this rfft half-spectrum represents.
    n_fft = 2 * (len(freqs_hz) - 1)

    # Zero-phase impulse response, then centre it for linear phase.
    h_full = np.fft.irfft(desired_lin, n=n_fft)
    h_centered = np.fft.fftshift(h_full)

    if n_taps % 2 == 0:
        n_taps += 1  # force odd length for a Type-I symmetric FIR
    if n_taps > len(h_centered):
        # largest odd value <= available samples (h_centered length is even)
        n_taps = (len(h_centered) - 1) | 1

    centre = len(h_centered) // 2
    half = n_taps // 2
    segment = h_centered[centre - half: centre + half + 1]

    window = np.hanning(n_taps)
    taps = segment * window

    return taps
