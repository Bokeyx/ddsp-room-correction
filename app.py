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
from src.pipeline import correct, smoothed_sigma
from src.synthetic import decaying_noise_rir
from src.targets import flat_target, harman_target

st.set_page_config(page_title="DDSP Room Correction", layout="wide")
st.title("🎧 DDSP Room Correction")
st.caption("방의 임펄스 응답(RIR)을 분석해 EQ 보정 필터를 자동 설계 — 고전 vs DDSP vs FIR 비교")

# --- sidebar: input RIR ---
st.sidebar.header("입력 RIR")
source = st.sidebar.radio("데이터", ["합성 RIR", "WAV 업로드"])
if source == "합성 RIR":
    seed = st.sidebar.slider("seed", 0, 100, 42)
    rt60 = st.sidebar.slider("RT60 [s]", 0.1, 1.0, 0.4, 0.1)
    rir, sr = decaying_noise_rir(48000, 0.5, rt60, seed=seed)
else:
    up = st.sidebar.file_uploader("RIR WAV", type=["wav"])
    if up is None:
        st.info("RIR WAV 파일을 업로드하세요.")
        st.stop()
    rir, sr = sf.read(io.BytesIO(up.read()), dtype="float64")
    if rir.ndim > 1:
        rir = rir.mean(axis=1)

# --- sidebar: options ---
st.sidebar.header("옵션")
target_name = st.sidebar.selectbox("목표 곡선", ["flat", "Harman"])
n_filters = st.sidebar.slider("필터 개수 (EQ)", 8, 64, 32, 8)
methods = st.sidebar.multiselect(
    "비교할 방식", ["classic", "ddsp", "fir"], default=["classic", "ddsp"]
)

freqs, resp = frequency_response(rir, sr)
target = flat_target(freqs) if target_name == "flat" else harman_target(freqs)
before = smoothed_sigma(resp, freqs)


@st.cache_data(show_spinner=False)
def _correct(resp, target, freqs, sr, method, n_filters):
    return correct(resp, target, freqs, sr, method=method, n_filters=n_filters)


colors = {"classic": "tab:green", "ddsp": "tab:red", "fir": "tab:blue"}

fig, ax = plt.subplots(figsize=(10, 4))
ax.semilogx(freqs, fractional_octave_smooth(freqs, resp), color="tab:gray", lw=1.5,
            label=f"before (σ={before:.2f})")
results = {}
with st.spinner("보정 계산 중... (DDSP는 수 초 소요)"):
    for m in methods:
        corrected_db, corr = _correct(resp, target, freqs, sr, m, n_filters)
        sigma = smoothed_sigma(corrected_db, freqs)
        results[m] = (corrected_db, corr, sigma)
        ax.semilogx(freqs, fractional_octave_smooth(freqs, corrected_db), color=colors[m],
                    lw=1.8, label=f"{m} (σ={sigma:.2f})")
ax.semilogx(freqs, target, "k--", lw=1, alpha=0.6, label=f"{target_name} target")
ax.set(xlim=(20, 20000), xlabel="frequency [Hz]", ylabel="magnitude [dB]",
       title="보정 전/후 주파수 응답 (1/3-octave smoothed)")
ax.legend()
ax.grid(alpha=0.3)
st.pyplot(fig)

# --- metrics ---
cols = st.columns(len(methods) + 1)
cols[0].metric("보정 전 σ", f"{before:.2f} dB")
for i, m in enumerate(methods):
    sigma = results[m][2]
    cols[i + 1].metric(f"{m} σ", f"{sigma:.2f} dB", f"{100 * (sigma / before - 1):.0f}%",
                       delta_color="inverse")

# --- A/B listening for the first selected method ---
if methods:
    st.subheader("A/B 청취 (핑크노이즈를 방에 통과)")
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
    c1.caption("보정 전 (in-room)")
    c1.audio(uncorrected, sample_rate=sr)
    c2.caption(f"보정 후 ({m0})")
    c2.audio(corrected, sample_rate=sr)
