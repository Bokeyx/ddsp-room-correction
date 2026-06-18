from src.eq_classic import PeakingFilter
from src.export import to_eqapo_config, to_rew_filters


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
