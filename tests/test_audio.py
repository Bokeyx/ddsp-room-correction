import numpy as np

from src.audio import apply_eq_to_signal, apply_fir_to_signal, demo_music, pink_noise, prepare_clip
from src.analysis import frequency_response
from src.eq_classic import PeakingFilter, apply_eq_db
from src.fir import fir_response_db, design_fir_correction
from src.metrics import band_mask
from src.synthetic import unit_impulse


SR = 48000


# --- 1. empty correction is a no-op ----------------------------------------

def test_apply_eq_empty_is_passthrough():
    sig = np.random.default_rng(0).standard_normal(2048)
    out = apply_eq_to_signal([], sig, SR)

    assert out.shape == sig.shape
    assert np.allclose(out, sig)


# --- 2. impulse cross-check: time-domain EQ matches dB design --------------

def test_apply_eq_impulse_matches_db_design():
    """Filtering a unit impulse with the cascaded biquads must reproduce the
    dB-domain design response (apply_eq_db) across the audible band. This is
    the core proof that the time-domain audio path is consistent with the
    frequency-domain design used everywhere else."""
    n = 16384
    imp, sr = unit_impulse(n, SR)
    filt = PeakingFilter(1000.0, 6.0, 4.0)

    out = apply_eq_to_signal([filt], imp, sr)
    assert out.shape == imp.shape  # length preserved

    freqs, meas_db = frequency_response(out, sr)
    design_db = apply_eq_db([filt], freqs, sr)

    mask = band_mask(freqs, 20.0, 20000.0)
    mae = np.mean(np.abs(meas_db[mask] - design_db[mask]))
    assert mae < 0.5  # mean abs error across audible band < 0.5 dB

    # The +6 dB peak near 1 kHz must be reproduced.
    peak_bin = int(np.argmin(np.abs(freqs - 1000.0)))
    assert abs(meas_db[peak_bin] - 6.0) < 0.5


# --- 3. FIR application cross-check -----------------------------------------

def test_apply_fir_impulse_matches_response_and_preserves_length():
    """Convolving a unit impulse with the FIR taps (mode='same') must
    reproduce the FIR magnitude response and preserve the signal length.
    Linear-phase FIR -> magnitude matches regardless of phase delay."""
    n = 16384
    imp, sr = unit_impulse(n, SR)

    # A real designed correction filter (linear phase, symmetric).
    freqs_design = np.fft.rfftfreq(8192, d=1.0 / sr)
    response_db = np.zeros_like(freqs_design)
    response_db[100] = 10.0
    target_db = np.zeros_like(freqs_design)
    taps = design_fir_correction(response_db, target_db, freqs_design, sr, n_taps=1025)

    out = apply_fir_to_signal(taps, imp)
    assert out.shape == imp.shape  # length preserved by mode='same'

    freqs, meas_db = frequency_response(out, sr)
    design_db = fir_response_db(taps, freqs, sr)

    mask = band_mask(freqs, 20.0, 20000.0)
    mae = np.mean(np.abs(meas_db[mask] - design_db[mask]))
    assert mae < 0.5


# --- 4. pink noise ----------------------------------------------------------

def test_pink_noise_length_and_reproducible():
    a = pink_noise(8192, seed=7)
    b = pink_noise(8192, seed=7)

    assert a.shape == (8192,)
    assert np.array_equal(a, b)  # same seed -> identical


def test_pink_noise_low_freq_tilt_and_amplitude():
    n = 16384
    x = pink_noise(n, seed=1)

    assert np.max(np.abs(x)) <= 1.0

    freqs, mag_db = frequency_response(x, SR)
    mask = band_mask(freqs, 20.0, 20000.0)
    f = freqs[mask]
    m = mag_db[mask]

    # Low band (below 200 Hz) must carry more energy than high band
    # (above 2 kHz) -- the 1/f tilt.
    low = m[f < 200.0].mean()
    high = m[f > 2000.0].mean()
    assert low > high


# --- 5. demo music clip (A/B listening) -------------------------------------

def test_demo_music_length_mono_and_normalised():
    x = demo_music(SR, duration_s=2.0)

    assert x.ndim == 1
    assert abs(len(x) - 2.0 * SR) <= SR // 10   # ~2 s (allow chord-grid rounding)
    peak = np.max(np.abs(x))
    assert 0.0 < peak <= 1.0                     # audible but not clipping


def test_demo_music_deterministic():
    assert np.array_equal(demo_music(SR, duration_s=2.0), demo_music(SR, duration_s=2.0))


def test_demo_music_is_broadband():
    """A good EQ A/B clip must carry energy across the band (bass to highs),
    otherwise the correction is inaudible. Check low and high bands are not
    empty relative to the total."""
    x = demo_music(SR, duration_s=2.0)
    freqs = np.fft.rfftfreq(len(x), d=1.0 / SR)
    power = np.abs(np.fft.rfft(x)) ** 2
    total = power.sum()

    low = power[(freqs >= 40) & (freqs < 200)].sum() / total
    high = power[freqs >= 3000].sum() / total
    assert low > 0.02     # real low-end content
    assert high > 0.001   # real high-end content


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
