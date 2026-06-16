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
    "# Room Correction: Classic EQ vs Differentiable Optimization (DDSP)\n"
    "\n"
    "Analyze a room impulse response (RIR) and automatically design EQ correction filters.\n"
    "The same problem is solved with **(1) a classic greedy heuristic** and **(2) gradient-descent\n"
    "differentiable optimization (DDSP)**, then compared quantitatively.\n"
    "\n"
    "The headline metric is **σ (standard deviation of the audible-band frequency response)** —\n"
    "lower means flatter (better correction). σ is gain-invariant, so it judges only the *shape* of the sound."
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
    "    the smoothed response (the same convention used in the test suite).'''\n"
    "    return flatness_std_db(fractional_octave_smooth(freqs, resp), freqs)"
)

md(
    "## 1. Synthetic RIR and frequency response\n"
    "\n"
    "For validation we generate a synthetic RIR with reverberation (RT60=0.4s). The room's\n"
    "frequency response is not flat but jagged — that is what we will correct."
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
    "## 2. Why smoothing is needed\n"
    "\n"
    "A raw FFT has thousands of bins, full of statistical noise. Chasing that noise with EQ actually\n"
    "wrecks the response. **1/3-octave smoothing** keeps only the broadband trend — that is the signal\n"
    "the correction should actually target."
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
    "## 3. Classic EQ baseline (greedy)\n"
    "\n"
    "Find the largest peak/dip in the smoothed deviation and place one peaking filter at a time\n"
    "(greedy). Gains are clamped to ±12 dB. Target is flat (0 dB)."
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
    "## 4. DDSP optimization EQ (headline)\n"
    "\n"
    "Treat the filter gains as **learnable parameters** and optimize all of them **jointly** with\n"
    "**PyTorch autograd + Adam**, minimizing the MSE deviation from the target. Unlike the greedy\n"
    "method placing filters one at a time, this accounts for interactions between filters."
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
    "## 5. Comparison: classic EQ vs DDSP vs FIR\n"
    "\n"
    "As a third method we add **FIR** (linear-phase, frequency sampling, 4097 taps). With thousands\n"
    "of taps it matches the whole band directly, so its magnitude flatness is good, but unlike 8\n"
    "biquads you cannot read 'what it corrected and why' — it is a black box. Interestingly, the\n"
    "**interpretable DDSP is as flat as, or flatter than, the 4097-tap FIR** (below)."
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
    "### nf sweep: why DDSP wins\n"
    "\n"
    "Compare the two methods as the filter count grows. **The classic greedy method saturates around\n"
    "σ≈0.40 from nf≥32** (more filters no longer help), while **DDSP keeps improving because it\n"
    "optimizes all gains jointly**. So DDSP overtakes from nf≥32 onward. (When the budget is tight,\n"
    "e.g. nf≤24, classic can still win — stated honestly.)"
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
    "## 6. Target curve: flat vs Harman\n"
    "\n"
    "Perfectly flat is not the perceptual ideal. Listening studies show people prefer a slightly\n"
    "downward-sloping **Harman-style in-room curve** (≈ -1 dB/oct tilt). Swapping only the target\n"
    "curve lets the same pipeline run unchanged — here is DDSP optimized to the Harman target."
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
    "## 7. A/B listening demo\n"
    "\n"
    "Compare pink noise sent through the room **before** correction and **after** pre-correcting it\n"
    "with the DDSP EQ. The spectrogram energy spreads more evenly, and you can listen directly with\n"
    "the players below (WAVs in `assets/audio/`). The reverb is intrinsic to the room and remains;\n"
    "what changes is the frequency balance."
)

code(
    "import scipy.signal as sps\n"
    "from IPython.display import Audio, display\n"
    "from src.audio import pink_noise, apply_eq_to_signal\n"
    "from src.io import save_wav\n"
    "\n"
    "os.makedirs('../assets/audio', exist_ok=True)\n"
    "dry = pink_noise(int(2.0 * sr), seed=0)\n"
    "# 'in-room' = signal convolved with the room; correction pre-filters the dry signal\n"
    "uncorrected = sps.fftconvolve(dry, rir)[:len(dry)]\n"
    "corrected = sps.fftconvolve(apply_eq_to_signal(ddsp_eq, dry, sr), rir)[:len(dry)]\n"
    "uncorrected = uncorrected / np.max(np.abs(uncorrected)) * 0.95\n"
    "corrected = corrected / np.max(np.abs(corrected)) * 0.95\n"
    "save_wav('../assets/audio/uncorrected.wav', uncorrected, sr)\n"
    "save_wav('../assets/audio/corrected.wav', corrected, sr)\n"
    "\n"
    "fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4), sharey=True)\n"
    "for ax, sig, ttl in [(a1, uncorrected, 'uncorrected (in-room)'),\n"
    "                     (a2, corrected, 'corrected (DDSP)')]:\n"
    "    f, t, Sxx = sps.spectrogram(sig, sr, nperseg=2048)\n"
    "    ax.pcolormesh(t, f, 10*np.log10(Sxx + 1e-12), shading='auto', cmap='magma')\n"
    "    ax.set(title=ttl, xlabel='time [s]', yscale='log', ylim=(20, 20000))\n"
    "a1.set_ylabel('frequency [Hz]')\n"
    "plt.tight_layout(); plt.savefig('../assets/08_spectrogram.png', dpi=110); plt.show()\n"
    "\n"
    "print('Uncorrected (in-room):'); display(Audio(uncorrected, rate=sr))\n"
    "print('Corrected (DDSP):'); display(Audio(corrected, rate=sr))"
)

md(
    "## Conclusion\n"
    "\n"
    "| Method | σ (after) | Parameters | Notes |\n"
    "|---|---|---|---|\n"
    "| before | ~0.68 | — | jagged room response |\n"
    "| classic greedy EQ | ~0.41 | 48 biquads | saturates at nf≥32 |\n"
    "| FIR (linear phase) | ~0.25 | 4097 taps | precise but a black box |\n"
    "| **DDSP optimized EQ** | **~0.23** | 48 biquads | interpretable + flattest |\n"
    "\n"
    "- A classic DSP baseline, an FIR, and ML optimization (DDSP) were compared fairly in one pipeline.\n"
    "- **DDSP reaches magnitude flatness equal to or better than a 4097-tap FIR using just 48\n"
    "  interpretable parameters.** It optimizes all gains jointly, surpassing the greedy method's plateau.\n"
    "- The target curve (flat/Harman) is injectable, so listener preference can be reflected too.\n"
    "- **σ only measures magnitude flatness.** FIR's real strength — phase / time-domain correction —\n"
    "  is not captured by this metric; stated honestly.\n"
    "- **Limitations / future work**: validation on real measured RIRs, A/B music tracks, multi-subject blind tests."
)

nb["cells"] = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}

with open("room_correction.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("wrote room_correction.ipynb with", len(cells), "cells")
