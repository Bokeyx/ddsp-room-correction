import io

import numpy as np
import soundfile as sf

from src.eq_classic import PeakingFilter
from src.export import to_eqapo_config, to_rew_filters, to_fir_wav_bytes, to_csv


def test_eqapo_config_lines_and_header():
    filters = [
        PeakingFilter(freq_hz=120.4, gain_db=-3.0, q=4.0),
        PeakingFilter(freq_hz=1000.0, gain_db=2.5, q=2.0),
    ]
    text = to_eqapo_config(filters)
    lines = text.splitlines()
    assert lines[0].startswith("#")
    assert "2 peaking filters" in text
    assert "Filter: ON PK Fc 120 Hz Gain -3.00 dB Q 4.000" in lines
    assert "Filter: ON PK Fc 1000 Hz Gain 2.50 dB Q 2.000" in lines


def test_eqapo_config_empty_filters_has_header_no_filter_lines():
    text = to_eqapo_config([])
    assert "0 peaking filters" in text
    assert "Filter:" not in text


def test_rew_filters_numbered_with_header():
    filters = [
        PeakingFilter(freq_hz=120.0, gain_db=-3.0, q=4.0),
        PeakingFilter(freq_hz=1000.0, gain_db=2.5, q=2.0),
    ]
    text = to_rew_filters(filters)
    lines = text.splitlines()
    assert lines[0] == "Filter Settings file"
    assert "Filter 1: ON PK Fc 120 Hz Gain -3.00 dB Q 4.000" in lines
    assert "Filter 2: ON PK Fc 1000 Hz Gain 2.50 dB Q 2.000" in lines


def test_fir_wav_bytes_roundtrip():
    sr = 48000
    taps = np.array([0.0, 1.0, -0.5, 0.25], dtype=np.float64)
    blob = to_fir_wav_bytes(taps, sr)
    assert isinstance(blob, bytes)

    data, read_sr = sf.read(io.BytesIO(blob), dtype="float32")
    assert read_sr == sr
    assert data.ndim == 1
    assert len(data) == len(taps)
    assert np.allclose(data, taps, atol=1e-6)


def test_csv_peaking_preamble_and_table():
    filters = [PeakingFilter(freq_hz=120.0, gain_db=-3.0, q=4.0)]
    freqs = np.array([20.0, 100.0, 1000.0])
    before = np.array([1.0, 2.0, 3.0])
    after = np.array([0.1, 0.2, 0.3])
    text = to_csv(filters, freqs, before, after)
    lines = text.splitlines()

    assert any(l.startswith("# filter,freq_hz,gain_db,q") for l in lines)
    assert "# 1,120,-3.00,4.000" in lines
    assert "freq_hz,before_db,after_db" in lines
    data_rows = [l for l in lines if not l.startswith("#") and "," in l
                 and not l.startswith("freq_hz")]
    assert len(data_rows) == len(freqs)
    assert data_rows[0] == "20.0,1.0,0.1"


def test_csv_fir_preamble_notes_tap_count():
    freqs = np.array([20.0, 100.0])
    before = np.array([1.0, 2.0])
    after = np.array([0.1, 0.2])
    text = to_csv(None, freqs, before, after, n_taps=2048)
    assert "# fir filter, 2048 taps" in text
    assert "freq_hz,before_db,after_db" in text
