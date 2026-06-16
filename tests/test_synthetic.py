import numpy as np

from src.synthetic import unit_impulse, decaying_noise_rir


def test_unit_impulse_shape_and_values():
    length = 1024
    sr = 48000
    signal, out_sr = unit_impulse(length, sr)

    assert out_sr == sr
    assert signal.shape == (length,)
    assert signal[0] == 1.0
    assert np.sum(signal[1:]) == 0.0


def test_decaying_noise_rir_length():
    sr = 48000
    duration_s = 0.5
    signal, out_sr = decaying_noise_rir(sr, duration_s, rt60_s=0.3, seed=0)

    assert out_sr == sr
    assert signal.shape == (round(duration_s * sr),)


def test_decaying_noise_rir_reproducible_with_same_seed():
    a, _ = decaying_noise_rir(48000, 0.5, rt60_s=0.3, seed=42)
    b, _ = decaying_noise_rir(48000, 0.5, rt60_s=0.3, seed=42)

    assert np.array_equal(a, b)


def test_decaying_noise_rir_energy_decreases_over_time():
    signal, sr = decaying_noise_rir(48000, 1.0, rt60_s=0.4, seed=1)
    n = len(signal)
    head = signal[: n // 10]
    tail = signal[-(n // 10):]

    rms_head = np.sqrt(np.mean(head ** 2))
    rms_tail = np.sqrt(np.mean(tail ** 2))

    assert rms_head > rms_tail
