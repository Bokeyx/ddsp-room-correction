"""Streamlit demo: analyse a room impulse response and compare correction methods.

Run with:  streamlit run app.py
"""
import io

import altair as alt
import numpy as np
import scipy.signal as sps
import soundfile as sf
import streamlit as st

from src.analysis import fractional_octave_smooth, frequency_response
from src.audio import apply_eq_to_signal, apply_fir_to_signal, pink_noise, prepare_clip, decode_audio
from src.charts import response_dataframe
from src.export import to_eqapo_config, to_rew_filters, to_fir_wav_bytes, to_csv
from src.i18n import LANGUAGES, t
from src.pipeline import correct, smoothed_sigma
from src.rooms import preset_names, preset_rir
from src.targets import flat_target, harman_target

st.set_page_config(page_title="DDSP Room Correction", layout="wide")

# --- sidebar: language ---
lang = LANGUAGES[st.sidebar.radio("🌐 Language / 언어", list(LANGUAGES), horizontal=True)]

st.title(t(lang, "title"))
st.caption(t(lang, "caption"))

# --- sidebar: room ---
st.sidebar.header(t(lang, "input_header"))
source = st.sidebar.radio(t(lang, "data_label"), ["example", "upload"],
                          format_func=lambda k: t(lang, k))
if source == "example":
    room = st.sidebar.selectbox(t(lang, "room_picker"), preset_names())
    rir, sr = preset_rir(room)
    st.sidebar.caption(t(lang, "room_caption"))
else:
    up = st.sidebar.file_uploader(t(lang, "uploader"), type=["wav"])
    if up is None:
        st.info(t(lang, "upload_info"))
        st.stop()
    rir, sr = sf.read(io.BytesIO(up.read()), dtype="float64")
    if rir.ndim > 1:
        rir = rir.mean(axis=1)

# --- sidebar: options ---
st.sidebar.header(t(lang, "options_header"))
target_name = st.sidebar.selectbox(t(lang, "target_curve"), ["flat", "Harman"])
n_filters = st.sidebar.slider(t(lang, "n_filters"), 8, 64, 32, 8)
methods = st.sidebar.multiselect(
    t(lang, "methods_label"), ["classic", "ddsp", "fir"], default=["classic", "ddsp"]
)
music_up = st.sidebar.file_uploader(
    t(lang, "music_uploader"), type=["wav", "flac", "ogg", "mp3", "m4a", "aac", "opus"]
)

freqs, resp = frequency_response(rir, sr)
target = flat_target(freqs) if target_name == "flat" else harman_target(freqs)
before = smoothed_sigma(resp, freqs)


@st.cache_data(show_spinner=False)
def _correct(resp, target, freqs, sr, method, n_filters):
    return correct(resp, target, freqs, sr, method=method, n_filters=n_filters)


PALETTE = {"before": "#9AA0A6", "classic": "#7FB5B5", "ddsp": "#F6C28B", "fir": "#B5A7E6"}

before_label = f"{t(lang, 'before')} (σ={before:.2f})"
series = {before_label: fractional_octave_smooth(freqs, resp)}
domain, scheme = [before_label], [PALETTE["before"]]
results = {}
with st.spinner(t(lang, "spinner")):
    for m in methods:
        corrected_db, corr = _correct(resp, target, freqs, sr, m, n_filters)
        sigma = smoothed_sigma(corrected_db, freqs)
        results[m] = (corrected_db, corr, sigma)
        label = f"{m} (σ={sigma:.2f})"
        series[label] = fractional_octave_smooth(freqs, corrected_db)
        domain.append(label)
        scheme.append(PALETTE[m])

resp_df = response_dataframe(freqs, series)
target_df = response_dataframe(freqs, {f"{target_name} {t(lang, 'target_suffix')}": target})

x_enc = alt.X("freq_hz:Q", scale=alt.Scale(type="log", domain=[20, 20000]),
              title=t(lang, "freq_axis"))
lines = alt.Chart(resp_df).mark_line().encode(
    x=x_enc,
    y=alt.Y("magnitude_db:Q", title=t(lang, "mag_axis")),
    color=alt.Color("series:N", scale=alt.Scale(domain=domain, range=scheme), title=None),
    tooltip=[alt.Tooltip("series:N", title="series"),
             alt.Tooltip("freq_hz:Q", title="Hz", format=".0f"),
             alt.Tooltip("magnitude_db:Q", title="dB", format=".1f")],
)
target_line = alt.Chart(target_df).mark_line(strokeDash=[6, 4], color="#9AA0A6").encode(
    x=x_enc, y="magnitude_db:Q",
)
chart = (target_line + lines).properties(title=t(lang, "plot_title"), height=360).interactive()
st.altair_chart(chart, use_container_width=True)

# --- metrics ---
cols = st.columns(len(methods) + 1)
cols[0].metric(t(lang, "before_metric"), f"{before:.2f} dB")
for i, m in enumerate(methods):
    sigma = results[m][2]
    cols[i + 1].metric(f"{m} σ", f"{sigma:.2f} dB", f"{100 * (sigma / before - 1):.0f}%",
                       delta_color="inverse")

# --- export the correction ---
st.subheader(t(lang, "export_header"))
st.caption(t(lang, "export_intro"))
for m in methods:
    corrected_db, corr, _ = results[m]
    st.markdown(f"**{m}**")
    st.caption(t(lang, f"desc_{m}"))
    cdl = st.columns(3)
    if isinstance(corr, np.ndarray):
        # FIR: impulse WAV + CSV
        cdl[0].download_button(
            t(lang, "btn_firwav"), data=to_fir_wav_bytes(corr, sr),
            file_name=f"correction_{m}.wav", mime="audio/wav", key=f"wav_{m}")
        cdl[1].download_button(
            t(lang, "btn_csv"), data=to_csv(None, freqs, resp, corrected_db, n_taps=len(corr)),
            file_name=f"correction_{m}.csv", mime="text/csv", key=f"csv_{m}")
    else:
        # peaking filters: EQ APO + REW + CSV
        cdl[0].download_button(
            t(lang, "btn_eqapo"), data=to_eqapo_config(corr),
            file_name=f"correction_{m}_eqapo.txt", mime="text/plain", key=f"apo_{m}")
        cdl[1].download_button(
            t(lang, "btn_rew"), data=to_rew_filters(corr),
            file_name=f"correction_{m}_rew.txt", mime="text/plain", key=f"rew_{m}")
        cdl[2].download_button(
            t(lang, "btn_csv"), data=to_csv(corr, freqs, resp, corrected_db),
            file_name=f"correction_{m}.csv", mime="text/csv", key=f"csv_{m}")

# --- A/B listening for the first selected method ---
if methods:
    st.subheader(t(lang, "ab_header"))
    m0 = methods[0]
    corr = results[m0][1]
    if music_up is not None:
        try:
            clip, clip_sr = decode_audio(music_up.read())
            dry = prepare_clip(clip, clip_sr, sr)
        except Exception:
            st.warning(t(lang, "music_error"))
            dry = pink_noise(int(2.0 * sr), seed=0)
    else:
        dry = pink_noise(int(2.0 * sr), seed=0)
    uncorrected = sps.fftconvolve(dry, rir)[: len(dry)]
    pre = apply_fir_to_signal(corr, dry) if isinstance(corr, np.ndarray) \
        else apply_eq_to_signal(corr, dry, sr)
    corrected = sps.fftconvolve(pre, rir)[: len(dry)]
    uncorrected = uncorrected / np.max(np.abs(uncorrected)) * 0.95
    corrected = corrected / np.max(np.abs(corrected)) * 0.95
    c1, c2 = st.columns(2)
    c1.caption(t(lang, "before_inroom"))
    c1.audio(uncorrected, sample_rate=sr)
    c2.caption(f"{t(lang, 'after')} ({m0})")
    c2.audio(corrected, sample_rate=sr)
