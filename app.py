"""Streamlit demo: analyse a room impulse response and compare correction methods.

Run with:  streamlit run app.py
"""
import io

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sps
import soundfile as sf
import streamlit as st

from src.analysis import fractional_octave_smooth, frequency_response
from src.audio import apply_eq_to_signal, apply_fir_to_signal, pink_noise
from src.export import to_eqapo_config, to_rew_filters, to_fir_wav_bytes, to_csv
from src.i18n import LANGUAGES, t
from src.pipeline import correct, smoothed_sigma
from src.synthetic import decaying_noise_rir
from src.targets import flat_target, harman_target

st.set_page_config(page_title="DDSP Room Correction", layout="wide")

# --- sidebar: language ---
lang = LANGUAGES[st.sidebar.radio("🌐 Language / 언어", list(LANGUAGES), horizontal=True)]

st.title(t(lang, "title"))
st.caption(t(lang, "caption"))

# --- sidebar: input RIR ---
st.sidebar.header(t(lang, "input_header"))
source = st.sidebar.radio(t(lang, "data_label"), ["synthetic", "upload"],
                          format_func=lambda k: t(lang, k))
if source == "synthetic":
    seed = st.sidebar.slider(t(lang, "seed"), 0, 100, 42)
    rt60 = st.sidebar.slider(t(lang, "rt60"), 0.1, 1.0, 0.4, 0.1)
    rir, sr = decaying_noise_rir(48000, 0.5, rt60, seed=seed)
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

freqs, resp = frequency_response(rir, sr)
target = flat_target(freqs) if target_name == "flat" else harman_target(freqs)
before = smoothed_sigma(resp, freqs)


@st.cache_data(show_spinner=False)
def _correct(resp, target, freqs, sr, method, n_filters):
    return correct(resp, target, freqs, sr, method=method, n_filters=n_filters)


colors = {"classic": "#7FB5B5", "ddsp": "#F6C28B", "fir": "#B5A7E6"}

fig, ax = plt.subplots(figsize=(10, 4))
ax.semilogx(freqs, fractional_octave_smooth(freqs, resp), color="#9AA0A6", lw=1.5,
            label=f"{t(lang, 'before')} (σ={before:.2f})")
results = {}
with st.spinner(t(lang, "spinner")):
    for m in methods:
        corrected_db, corr = _correct(resp, target, freqs, sr, m, n_filters)
        sigma = smoothed_sigma(corrected_db, freqs)
        results[m] = (corrected_db, corr, sigma)
        ax.semilogx(freqs, fractional_octave_smooth(freqs, corrected_db), color=colors[m],
                    lw=1.8, label=f"{m} (σ={sigma:.2f})")
ax.semilogx(freqs, target, "k--", lw=1, alpha=0.6, label=f"{target_name} {t(lang, 'target_suffix')}")
ax.set(xlim=(20, 20000), xlabel="frequency [Hz]", ylabel="magnitude [dB]",
       title=t(lang, "plot_title"))
ax.legend()
ax.grid(alpha=0.3)
st.pyplot(fig)

# --- metrics ---
cols = st.columns(len(methods) + 1)
cols[0].metric(t(lang, "before_metric"), f"{before:.2f} dB")
for i, m in enumerate(methods):
    sigma = results[m][2]
    cols[i + 1].metric(f"{m} σ", f"{sigma:.2f} dB", f"{100 * (sigma / before - 1):.0f}%",
                       delta_color="inverse")

# --- export the correction ---
st.subheader(t(lang, "export_header"))
for m in methods:
    corrected_db, corr, _ = results[m]
    st.markdown(f"**{m}**")
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
