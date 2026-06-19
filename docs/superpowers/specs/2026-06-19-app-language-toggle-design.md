# App language toggle (EN/KO) — design

- **Date:** 2026-06-19
- **Status:** approved (brainstorming)
- **Scope:** the Streamlit app UI only — not the repo docs/README.

## Motivation

The deployed demo (https://room-correction.streamlit.app/) is English-only. The author and many likely
viewers are Korean, so a one-click EN/KO toggle makes the live app readable to both audiences without
forking the UI. The app's user-facing text is a fixed set of labels, so this is a simple lookup table —
no translation API, no cost, works offline and on Streamlit Cloud.

## Approach

A small pure module `src/i18n.py` holds a per-language string table and a lookup helper. `app.py` adds a
language selector at the top of the sidebar (default English, to match the repo's English-only rule) and
routes every user-facing label through the helper.

## Module: `src/i18n.py`

```python
LANGUAGES = {"English": "en", "한국어": "ko"}   # display name -> code
STRINGS = {"en": {...}, "ko": {...}}            # same key set in both
def t(lang, key) -> str                          # STRINGS[lang][key]; unknown key -> key itself
```

- `t` returns the string for `(lang, key)`. An unknown `key` returns the key verbatim (safe fallback,
  never raises). An unknown `lang` falls back to English.
- Both language dicts must contain the **same keys** (enforced by a test).

## What is translated vs left as-is

- **Translated:** section headers, input labels, buttons, captions, info text, spinner, plot title,
  the "before"/"after" word in curve/metric labels.
- **Left as-is (technical tokens):** method ids (`classic`/`ddsp`/`fir`), target ids (`flat`/`Harman`),
  units (`Hz`, `dB`), the `σ` symbol, and the plot axis labels. Translating these would tangle the
  control logic (they double as dict keys / comparison values) for no real benefit.

To keep the source radio's comparison stable while showing translated text, the radio uses stable option
codes (`"synthetic"`, `"upload"`) with a `format_func` that displays `t(lang, code)`.

## Testing (`tests/test_i18n.py`)

- Key parity: `STRINGS["en"]` and `STRINGS["ko"]` have identical key sets.
- `t("ko", key)` returns the Korean string; `t("en", key)` returns the English string (differ for a
  sample key).
- Unknown key returns the key string unchanged.
- Unknown language falls back to English.

Suite: 101 → ~105.

## Non-goals

- No external/machine translation. No languages beyond EN/KO. No persistence of the choice beyond the
  session. No translation of README/docs (repo stays English).

## Success criteria

- `src/i18n.py` is pure and passes the tests above.
- On the live app, switching the sidebar language flips all chrome text between English and Korean while
  the plots, method names, and metrics keep working unchanged.
