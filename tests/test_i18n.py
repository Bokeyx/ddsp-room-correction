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


def test_export_description_keys_present_both_languages():
    for key in ("export_intro", "desc_classic", "desc_ddsp", "desc_fir"):
        assert key in STRINGS["en"], key
        assert key in STRINGS["ko"], key


def test_export_format_help_keys_present_both_languages():
    for key in ("help_eqapo", "help_rew", "help_csv", "help_firwav"):
        assert key in STRINGS["en"], key
        assert key in STRINGS["ko"], key
