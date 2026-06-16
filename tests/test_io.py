import numpy as np

from src.io import load_wav, save_wav


def test_save_load_roundtrip_preserves_samplerate_and_signal(tmp_path):
    sr = 48000
    signal = np.sin(2 * np.pi * 440 * np.arange(sr) / sr)

    path = tmp_path / "roundtrip.wav"
    save_wav(str(path), signal, sr)
    loaded, loaded_sr = load_wav(str(path))

    assert loaded_sr == sr
    assert loaded.ndim == 1
    assert loaded.dtype == np.float64
    assert np.allclose(loaded, signal, atol=1e-4)


def test_load_stereo_is_averaged_to_mono(tmp_path):
    import soundfile as sf

    sr = 16000
    left = np.full(100, 0.5)
    right = np.full(100, -0.1)
    stereo = np.stack([left, right], axis=1)

    path = tmp_path / "stereo.wav"
    sf.write(str(path), stereo, sr)

    mono, loaded_sr = load_wav(str(path))

    assert loaded_sr == sr
    assert mono.ndim == 1
    assert mono.shape == (100,)
    assert np.allclose(mono, 0.2, atol=1e-4)
