from src.i18n import LANGUAGES, STRINGS, t


def test_language_key_parity():
    assert set(STRINGS["en"]) == set(STRINGS["ko"])


def test_languages_map_to_known_codes():
    assert set(LANGUAGES.values()) <= set(STRINGS)
    assert LANGUAGES["English"] == "en"
    assert LANGUAGES["한국어"] == "ko"


def test_t_returns_language_specific_string():
    # pick any key and confirm en/ko differ and match their tables
    key = "input_header"
    assert t("en", key) == STRINGS["en"][key]
    assert t("ko", key) == STRINGS["ko"][key]
    assert t("en", key) != t("ko", key)


def test_t_unknown_key_returns_key():
    assert t("en", "no_such_key") == "no_such_key"


def test_t_unknown_language_falls_back_to_english():
    assert t("xx", "input_header") == STRINGS["en"]["input_header"]
