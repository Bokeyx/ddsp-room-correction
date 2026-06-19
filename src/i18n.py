"""UI string tables for the Streamlit app's EN/KO language toggle.

Pure data + one lookup helper -- no Streamlit dependency, so it is unit tested
on its own. Only the app's chrome (headers, labels, buttons, captions) is
translated; technical tokens (method ids, target names, units, the sigma symbol,
axis labels) are left untranslated because they double as control-flow values.
"""

# display name shown in the selector -> language code
LANGUAGES = {"English": "en", "한국어": "ko"}

STRINGS = {
    "en": {
        "language": "Language",
        "title": "🎧 DDSP Room Correction",
        "caption": "Analyze a room impulse response and auto-design EQ correction — classic vs DDSP vs FIR",
        "input_header": "Input RIR",
        "data_label": "Data",
        "example": "Example room",
        "upload": "Upload my room (advanced)",
        "room_picker": "Pick a room",
        "room_caption": "Simulated room — pick one, then play your own music below.",
        "music_uploader": "Play your own music (optional)",
        "music_error": "Couldn't read that audio — try WAV or FLAC. Using pink noise instead.",
        "uploader": "RIR WAV",
        "upload_info": "Upload a RIR WAV file.",
        "options_header": "Options",
        "target_curve": "Target curve",
        "n_filters": "Number of filters (EQ)",
        "methods_label": "Methods to compare",
        "spinner": "Computing correction... (DDSP takes a few seconds)",
        "plot_title": "Before/after frequency response (1/3-octave smoothed)",
        "target_suffix": "target",
        "before": "before",
        "after": "after",
        "before_metric": "before σ",
        "export_header": "Export correction",
        "btn_eqapo": "Equalizer APO",
        "btn_rew": "REW filters",
        "btn_csv": "CSV",
        "btn_firwav": "FIR impulse WAV",
        "ab_header": "A/B listening (through the room)",
        "before_inroom": "before (in-room)",
    },
    "ko": {
        "language": "언어",
        "title": "🎧 DDSP 룸 보정",
        "caption": "방의 임펄스 응답을 분석해 EQ 보정을 자동 설계 — classic vs DDSP vs FIR",
        "input_header": "입력 RIR",
        "data_label": "데이터",
        "example": "예시 방",
        "upload": "내 방 업로드 (고급)",
        "room_picker": "방 선택",
        "room_caption": "시뮬레이션 방 — 하나 고른 뒤 아래에서 내 음악을 재생하세요.",
        "music_uploader": "내 음악 재생 (선택)",
        "music_error": "이 오디오를 읽지 못했어요 — WAV/FLAC로 올려주세요. 대신 핑크노이즈를 씁니다.",
        "uploader": "RIR WAV 파일",
        "upload_info": "RIR WAV 파일을 업로드하세요.",
        "options_header": "옵션",
        "target_curve": "목표 곡선",
        "n_filters": "필터 개수 (EQ)",
        "methods_label": "비교할 방법",
        "spinner": "보정 계산 중... (DDSP는 몇 초 걸립니다)",
        "plot_title": "보정 전후 주파수 응답 (1/3-옥타브 평활)",
        "target_suffix": "목표",
        "before": "보정 전",
        "after": "보정 후",
        "before_metric": "보정 전 σ",
        "export_header": "보정 내보내기",
        "btn_eqapo": "Equalizer APO",
        "btn_rew": "REW 필터",
        "btn_csv": "CSV",
        "btn_firwav": "FIR 임펄스 WAV",
        "ab_header": "A/B 청취 (방 통과)",
        "before_inroom": "보정 전 (방 안)",
    },
}


def t(lang, key):
    """Look up a UI string. Unknown language -> English; unknown key -> the key."""
    table = STRINGS.get(lang, STRINGS["en"])
    return table.get(key, key)
