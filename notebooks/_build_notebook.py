"""Generate room_correction.ipynb programmatically with nbformat.

Building the notebook from a script guarantees valid .ipynb JSON. Run this,
then execute the notebook to fill in plot outputs:

    python notebooks/_build_notebook.py
    jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=900 notebooks/room_correction.ipynb
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


md(
    "# 룸 보정: 고전 EQ vs 미분가능 최적화(DDSP)\n"
    "\n"
    "방의 임펄스 응답(RIR)을 분석해 EQ 보정 필터를 자동 설계한다. 같은 문제를\n"
    "**(1) 고전 그리디 휴리스틱**과 **(2) 경사하강법 기반 미분가능 최적화(DDSP)**로\n"
    "풀고 정량 비교한다.\n"
    "\n"
    "핵심 지표는 **σ(가청대역 주파수응답 표준편차)** — 작을수록 평탄(보정 잘 됨).\n"
    "σ는 전체 게인에 불변이라 '소리의 모양'만 평가한다."
)

code(
    "import os, sys\n"
    "sys.path.insert(0, os.path.abspath('..'))\n"
    "\n"
    "import numpy as np\n"
    "import matplotlib.pyplot as plt\n"
    "\n"
    "from src.synthetic import decaying_noise_rir\n"
    "from src.analysis import frequency_response, fractional_octave_smooth\n"
    "from src.targets import flat_target, harman_target\n"
    "from src.metrics import flatness_std_db\n"
    "from src.eq_classic import design_classic_eq, apply_eq_db\n"
    "from src.eq_ddsp import optimize_eq\n"
    "from src.fir import design_fir_correction, fir_response_db\n"
    "\n"
    "os.makedirs('../assets', exist_ok=True)\n"
    "plt.rcParams['figure.figsize'] = (10, 4)\n"
    "plt.rcParams['axes.grid'] = True\n"
    "plt.rcParams['grid.alpha'] = 0.3\n"
    "\n"
    "ITERS = 150  # DDSP iterations (loss plateaus well before this)\n"
    "\n"
    "def smoothed_sigma(resp, freqs):\n"
    "    '''Fair flatness metric: sigma of the 1/3-octave-smoothed response.\n"
    "    Raw FFT bin noise is uncorrectable by any peaking EQ, so we compare on\n"
    "    the smoothed response (same convention used in the test suite).'''\n"
    "    return flatness_std_db(fractional_octave_smooth(freqs, resp), freqs)"
)

md(
    "## 1. 합성 RIR과 주파수 응답\n"
    "\n"
    "검증을 위해 잔향(RT60=0.4s)을 가진 합성 RIR을 만든다. 이 방의 주파수 응답은\n"
    "평탄하지 않고 들쭉날쭉하다 — 이게 보정 대상이다."
)

code(
    "rir, sr = decaying_noise_rir(48000, 0.5, 0.4, seed=42)\n"
    "freqs, resp = frequency_response(rir, sr)\n"
    "\n"
    "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))\n"
    "t = np.arange(len(rir)) / sr\n"
    "ax1.plot(t, rir, lw=0.5)\n"
    "ax1.set(title='Room Impulse Response (RIR)', xlabel='time [s]', ylabel='amplitude')\n"
    "ax2.semilogx(freqs, resp, lw=0.6, color='tab:gray')\n"
    "ax2.set(title='Raw frequency response', xlabel='frequency [Hz]', ylabel='magnitude [dB]', xlim=(20, 20000))\n"
    "plt.tight_layout(); plt.savefig('../assets/01_rir_response.png', dpi=110); plt.show()"
)

md(
    "## 2. 왜 스무딩이 필요한가\n"
    "\n"
    "raw FFT는 bin이 수천 개라 통계적 잡음으로 가득하다. 이걸 그대로 EQ로 쫓으면\n"
    "오히려 응답을 망친다. **1/3-옥타브 스무딩**으로 광대역 추세만 남긴다 — 이게\n"
    "보정이 실제로 다뤄야 할 신호다."
)

code(
    "resp_sm = fractional_octave_smooth(freqs, resp)\n"
    "\n"
    "plt.semilogx(freqs, resp, lw=0.5, color='lightgray', label='raw')\n"
    "plt.semilogx(freqs, resp_sm, lw=2, color='tab:blue', label='1/3-octave smoothed')\n"
    "plt.legend(); plt.xlim(20, 20000)\n"
    "plt.title(f'Raw vs smoothed  (raw sigma={flatness_std_db(resp, freqs):.2f}, '\n"
    "          f'smoothed sigma={flatness_std_db(resp_sm, freqs):.2f} dB)')\n"
    "plt.xlabel('frequency [Hz]'); plt.ylabel('magnitude [dB]')\n"
    "plt.tight_layout(); plt.savefig('../assets/02_smoothing.png', dpi=110); plt.show()"
)

md(
    "## 3. 고전 EQ baseline (그리디)\n"
    "\n"
    "스무딩한 편차에서 가장 큰 봉우리/골을 찾아 피킹 필터를 하나씩 배치(그리디).\n"
    "게인은 ±12 dB로 클램프. 목표는 flat(0 dB)."
)

code(
    "target = flat_target(freqs)\n"
    "NF = 48\n"
    "classic_eq = design_classic_eq(resp, target, freqs, sr, n_filters=NF)\n"
    "classic_corr = resp + apply_eq_db(classic_eq, freqs, sr)\n"
    "\n"
    "before = smoothed_sigma(resp, freqs)\n"
    "classic_after = smoothed_sigma(classic_corr, freqs)\n"
    "\n"
    "plt.semilogx(freqs, resp_sm, lw=1.5, color='tab:gray', label=f'before (sigma={before:.2f})')\n"
    "plt.semilogx(freqs, fractional_octave_smooth(freqs, classic_corr), lw=2, color='tab:green',\n"
    "             label=f'classic corrected (sigma={classic_after:.2f})')\n"
    "plt.axhline(0, color='k', lw=0.8, ls='--', alpha=0.6, label='flat target')\n"
    "plt.legend(); plt.xlim(20, 20000)\n"
    "plt.title(f'Classic greedy EQ ({NF} filters): sigma {before:.2f} -> {classic_after:.2f} dB '\n"
    "          f'({100*(1-classic_after/before):.0f}% flatter)')\n"
    "plt.xlabel('frequency [Hz]'); plt.ylabel('magnitude [dB]')\n"
    "plt.tight_layout(); plt.savefig('../assets/03_classic.png', dpi=110); plt.show()"
)

md(
    "## 4. DDSP 최적화 EQ (헤드라인)\n"
    "\n"
    "필터 게인을 **학습 파라미터**로 두고, '목표와의 편차 MSE'를 손실로 정의해\n"
    "**PyTorch autograd + Adam**으로 모든 게인을 **동시에** 최적화한다. 그리디가\n"
    "필터를 하나씩 순차로 놓는 것과 달리, 필터 간 상호작용까지 고려한다."
)

code(
    "ddsp_eq = optimize_eq(resp, target, freqs, sr, n_filters=NF, iters=ITERS)\n"
    "ddsp_corr = resp + apply_eq_db(ddsp_eq, freqs, sr)\n"
    "ddsp_after = smoothed_sigma(ddsp_corr, freqs)\n"
    "\n"
    "plt.semilogx(freqs, resp_sm, lw=1.5, color='tab:gray', label=f'before (sigma={before:.2f})')\n"
    "plt.semilogx(freqs, fractional_octave_smooth(freqs, ddsp_corr), lw=2, color='tab:red',\n"
    "             label=f'DDSP corrected (sigma={ddsp_after:.2f})')\n"
    "plt.axhline(0, color='k', lw=0.8, ls='--', alpha=0.6, label='flat target')\n"
    "plt.legend(); plt.xlim(20, 20000)\n"
    "plt.title(f'DDSP optimised EQ ({NF} filters): sigma {before:.2f} -> {ddsp_after:.2f} dB '\n"
    "          f'({100*(1-ddsp_after/before):.0f}% flatter)')\n"
    "plt.xlabel('frequency [Hz]'); plt.ylabel('magnitude [dB]')\n"
    "plt.tight_layout(); plt.savefig('../assets/04_ddsp.png', dpi=110); plt.show()"
)

md(
    "## 5. 비교: 고전 EQ vs DDSP vs FIR\n"
    "\n"
    "세 번째 방식으로 **FIR**(주파수 샘플링 선형위상, 4097탭)을 더한다. 수천 탭으로\n"
    "전 대역을 직접 매칭하므로 크기 평탄도는 좋지만, 8개 biquad처럼 '무엇을 왜'\n"
    "보정했는지 읽을 수 없는 블랙박스다. 흥미롭게도 **해석 가능한 DDSP가 4097탭 FIR과\n"
    "동등하거나 더 평탄**하다(아래)."
)

code(
    "fir_taps = design_fir_correction(resp, target, freqs, sr, n_taps=4097)\n"
    "fir_corr = resp + fir_response_db(fir_taps, freqs, sr)\n"
    "fir_after = smoothed_sigma(fir_corr, freqs)\n"
    "\n"
    "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))\n"
    "ax1.semilogx(freqs, resp_sm, lw=1.2, color='tab:gray', label=f'before ({before:.2f})')\n"
    "ax1.semilogx(freqs, fractional_octave_smooth(freqs, classic_corr), lw=1.6, color='tab:green',\n"
    "             label=f'classic ({classic_after:.2f})')\n"
    "ax1.semilogx(freqs, fractional_octave_smooth(freqs, ddsp_corr), lw=1.6, color='tab:red',\n"
    "             label=f'DDSP ({ddsp_after:.2f})')\n"
    "ax1.semilogx(freqs, fractional_octave_smooth(freqs, fir_corr), lw=1.6, color='tab:blue',\n"
    "             label=f'FIR ({fir_after:.2f})')\n"
    "ax1.axhline(0, color='k', lw=0.8, ls='--', alpha=0.5)\n"
    "ax1.legend(); ax1.set(xlim=(20, 20000), title='Corrected response (smoothed)',\n"
    "                      xlabel='frequency [Hz]', ylabel='magnitude [dB]')\n"
    "labels = ['before', 'classic', 'DDSP', 'FIR']\n"
    "vals = [before, classic_after, ddsp_after, fir_after]\n"
    "bars = ax2.bar(labels, vals, color=['tab:gray', 'tab:green', 'tab:red', 'tab:blue'])\n"
    "ax2.set(title='Flatness sigma (lower = better)', ylabel='sigma [dB]')\n"
    "for b, v in zip(bars, vals):\n"
    "    ax2.text(b.get_x()+b.get_width()/2, v, f'{v:.2f}', ha='center', va='bottom')\n"
    "plt.tight_layout(); plt.savefig('../assets/05_compare.png', dpi=110); plt.show()"
)

md(
    "### nf 스윕: 왜 DDSP가 이기는가\n"
    "\n"
    "필터 개수를 늘려가며 두 방법을 비교한다. **고전 그리디는 nf≥32에서 σ≈0.40으로\n"
    "포화**(필터를 더 줘도 못 채운다)하는 반면, **DDSP는 동시 최적화라 필터 예산이\n"
    "늘수록 계속 개선**된다. 그래서 nf≥32부터 DDSP가 앞선다. (nf≤24처럼 예산이\n"
    "빠듯하면 고전이 이길 수도 있다 — 정직하게 표기.)"
)

code(
    "nfs = [8, 16, 24, 32, 48]\n"
    "classic_sigmas, ddsp_sigmas = [], []\n"
    "for nf in nfs:\n"
    "    c = resp + apply_eq_db(design_classic_eq(resp, target, freqs, sr, n_filters=nf), freqs, sr)\n"
    "    d = resp + apply_eq_db(optimize_eq(resp, target, freqs, sr, n_filters=nf, iters=ITERS), freqs, sr)\n"
    "    classic_sigmas.append(smoothed_sigma(c, freqs))\n"
    "    ddsp_sigmas.append(smoothed_sigma(d, freqs))\n"
    "    print(f'nf={nf:3d}  classic={classic_sigmas[-1]:.3f}  DDSP={ddsp_sigmas[-1]:.3f}')\n"
    "\n"
    "plt.plot(nfs, classic_sigmas, 'o-', color='tab:green', label='classic (greedy)')\n"
    "plt.plot(nfs, ddsp_sigmas, 's-', color='tab:red', label='DDSP (optimised)')\n"
    "plt.xlabel('number of filters'); plt.ylabel('sigma [dB] (lower = better)')\n"
    "plt.title('Classic saturates; DDSP keeps improving with filter budget')\n"
    "plt.legend(); plt.tight_layout(); plt.savefig('../assets/06_nf_sweep.png', dpi=110); plt.show()"
)

md(
    "## 6. 목표 곡선: flat vs Harman\n"
    "\n"
    "완전 평탄(flat)이 청감상 정답은 아니다. 청취 실험에 따르면 사람은 살짝 우하향하는\n"
    "**Harman 스타일 인룸 곡선**(≈ -1 dB/oct 틸트)을 선호한다. 목표 곡선만 바꾸면\n"
    "같은 파이프라인이 그대로 동작한다 — DDSP를 Harman 목표로 최적화한 결과를 본다."
)

code(
    "harman = harman_target(freqs)\n"
    "ddsp_h_eq = optimize_eq(resp, harman, freqs, sr, n_filters=NF, iters=ITERS)\n"
    "ddsp_h_corr = resp + apply_eq_db(ddsp_h_eq, freqs, sr)\n"
    "\n"
    "# level-align the corrected curves to their targets for display\n"
    "def align(curve, tgt):\n"
    "    m = (freqs >= 20) & (freqs <= 20000)\n"
    "    return curve - np.mean((curve - tgt)[m])\n"
    "\n"
    "plt.semilogx(freqs, np.zeros_like(freqs), 'k--', lw=1, alpha=0.6, label='flat target')\n"
    "plt.semilogx(freqs, harman, color='tab:purple', lw=1.5, ls='--', label='Harman target (-1 dB/oct)')\n"
    "plt.semilogx(freqs, align(fractional_octave_smooth(freqs, ddsp_corr), np.zeros_like(freqs)),\n"
    "             color='tab:red', lw=1.8, label='DDSP -> flat')\n"
    "plt.semilogx(freqs, align(fractional_octave_smooth(freqs, ddsp_h_corr), harman),\n"
    "             color='tab:orange', lw=1.8, label='DDSP -> Harman')\n"
    "plt.legend(); plt.xlim(20, 20000)\n"
    "plt.title('Same pipeline, swap the target curve: flat vs Harman')\n"
    "plt.xlabel('frequency [Hz]'); plt.ylabel('magnitude [dB]')\n"
    "plt.tight_layout(); plt.savefig('../assets/07_targets.png', dpi=110); plt.show()"
)

md(
    "## 결론\n"
    "\n"
    "| 방법 | σ (보정 후) | 파라미터 | 비고 |\n"
    "|---|---|---|---|\n"
    "| 보정 전 | ~0.68 | — | 들쭉날쭉한 방 응답 |\n"
    "| 고전 그리디 EQ | ~0.41 | 48 biquad | nf≥32에서 포화 |\n"
    "| FIR (선형위상) | ~0.25 | 4097 tap | 정밀하나 블랙박스 |\n"
    "| **DDSP 최적화 EQ** | **~0.23** | 48 biquad | 해석 가능 + 최고 평탄도 |\n"
    "\n"
    "- 고전 신호처리 baseline·FIR·ML 최적화(DDSP)를 한 파이프라인에서 공정 비교했다.\n"
    "- **DDSP는 48개의 해석 가능한 파라미터만으로 4097탭 FIR과 동등 이상의 크기 평탄도**를\n"
    "  달성한다. 모든 게인을 동시에 최적화해 그리디의 포화 한계를 넘는다.\n"
    "- 목표 곡선(flat/Harman)을 주입식으로 바꿔 청감 선호까지 반영할 수 있다.\n"
    "- **σ는 크기 평탄도만 본다.** FIR의 진짜 강점인 위상·시간축 보정은 이 지표에\n"
    "  안 잡힌다 — 정직하게 짚어둘 점.\n"
    "- **한계/향후**: 실제 측정 RIR 검증, A/B 청취 음원, 다수 피험자 블라인드 테스트."
)

nb["cells"] = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}

with open("room_correction.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("wrote room_correction.ipynb with", len(cells), "cells")
