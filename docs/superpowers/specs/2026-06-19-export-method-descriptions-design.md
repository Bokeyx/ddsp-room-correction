# Export section: per-method descriptions — design

- **Date:** 2026-06-19
- **Status:** approved (brainstorming)
- **Scope:** the app's "Export correction" section copy. Small UI-text addition.

## Motivation

The Export correction section lists each method (classic / ddsp / fir) as a bare bold label above its
download buttons. A visitor has no way to know which method to download or how they differ. A one-line
description per method makes the choice obvious without cluttering the section.

## Goals

- A short section intro under the "Export correction" subheader.
- A one-line description under each method label, explaining what that method is.
- All copy added to the EN/KO i18n tables (English default + Korean), kept in parity.

## Strings (i18n, both languages)

- `export_intro` — EN: "Download the correction in formats real audio tools read." / KO: "보정 결과를 실제
  오디오 도구가 읽는 형식으로 내려받으세요."
- `desc_classic` — EN: "Classic rule-based EQ — simple and predictable." / KO: "고전 규칙 기반 EQ — 단순하고
  예측 가능."
- `desc_ddsp` — EN: "ML-optimized EQ — usually the flattest, still interpretable." / KO: "ML 최적화 EQ —
  보통 가장 평평하고, 그래도 해석 가능."
- `desc_fir` — EN: "FIR filter — very precise but a black box." / KO: "FIR 필터 — 매우 정밀하지만 블랙박스."

## app.py changes

- After the `st.subheader(t(lang, "export_header"))` line, add `st.caption(t(lang, "export_intro"))`.
- Inside the per-method loop, after `st.markdown(f"**{m}**")`, add `st.caption(t(lang, f"desc_{m}"))`.
  The key `desc_{m}` resolves to the matching description; `m` is always one of classic/ddsp/fir.

## Testing

- `tests/test_i18n.py`: add a test asserting `desc_classic`, `desc_ddsp`, `desc_fir`, and `export_intro`
  exist in both `en` and `ko`. The existing key-parity test guards that the two tables stay in sync.

Suite: 122 -> 123.

## Non-goals

- No per-format descriptions (method-level only). No new modules, no logic/chart/audio changes.

## Success criteria

- The Export section shows an intro line and a one-line description under each method; both languages
  render correctly; the suite stays green.
