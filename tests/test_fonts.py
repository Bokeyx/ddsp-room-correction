import os

from src.fonts import FONT_PATH, register_korean_font


def test_korean_font_is_bundled():
    assert os.path.exists(FONT_PATH)


def test_register_korean_font_returns_family_name():
    name = register_korean_font()
    assert isinstance(name, str) and name
    assert "Nanum" in name
