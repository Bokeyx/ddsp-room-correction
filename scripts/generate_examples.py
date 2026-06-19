"""Generate committed example export artifacts so the export feature can be
inspected without running the app or any Python.

Runs the same pipeline the Streamlit app uses on a deterministic synthetic room
(seed 42), writes one example file per export format into ``assets/exports/``,
and renders a pastel before/after preview PNG that mirrors the app's plot.

Re-run with:  .venv\\Scripts\\python.exe scripts/generate_examples.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.analysis import fractional_octave_smooth, frequency_response
from src.export import to_eqapo_config, to_rew_filters, to_fir_wav_bytes, to_csv
from src.pipeline import correct, smoothed_sigma
from src.synthetic import decaying_noise_rir
from src.targets import flat_target

OUT = os.path.join("assets", "exports")
COLORS = {"classic": "#7FB5B5", "ddsp": "#F6C28B", "fir": "#B5A7E6"}


def main():
    os.makedirs(OUT, exist_ok=True)

    rir, sr = decaying_noise_rir(48000, 0.5, 0.4, seed=42)
    freqs, resp = frequency_response(rir, sr)
    target = flat_target(freqs)
    before = smoothed_sigma(resp, freqs)

    results = {}
    for m in ("classic", "ddsp", "fir"):
        corrected_db, corr = correct(resp, target, freqs, sr, method=m, n_filters=24)
        results[m] = (corrected_db, corr, smoothed_sigma(corrected_db, freqs))

    # peaking methods -> EQ APO + REW text (tiny); fir -> WAV.
    for m in ("classic", "ddsp"):
        _, corr, _ = results[m]
        _write(f"correction_{m}_eqapo.txt", to_eqapo_config(corr))
        _write(f"correction_{m}_rew.txt", to_rew_filters(corr))

    fir_db, taps, _ = results["fir"]
    _write_bytes("correction_fir.wav", to_fir_wav_bytes(taps, sr))

    # One CSV per preamble variant (peaking vs FIR); the per-frequency table is
    # large, so we don't triplicate it across methods.
    ddsp_db, ddsp_corr, _ = results["ddsp"]
    _write("correction_ddsp.csv", to_csv(ddsp_corr, freqs, resp, ddsp_db))
    _write("correction_fir.csv", to_csv(None, freqs, resp, fir_db, n_taps=len(taps)))

    _preview(freqs, resp, target, before, results)
    print("done:", sorted(os.listdir(OUT)))


def _write(name, text):
    with open(os.path.join(OUT, name), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def _write_bytes(name, blob):
    with open(os.path.join(OUT, name), "wb") as f:
        f.write(blob)


def _preview(freqs, resp, target, before, results):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.semilogx(freqs, fractional_octave_smooth(freqs, resp), color="#9AA0A6",
                lw=1.5, label=f"before (σ={before:.2f})")
    for m, (corrected_db, _, sigma) in results.items():
        ax.semilogx(freqs, fractional_octave_smooth(freqs, corrected_db),
                    color=COLORS[m], lw=1.8, label=f"{m} (σ={sigma:.2f})")
    ax.semilogx(freqs, target, "k--", lw=1, alpha=0.6, label="flat target")
    ax.set(xlim=(20, 20000), xlabel="frequency [Hz]", ylabel="magnitude [dB]",
           title="App preview — before/after (1/3-octave smoothed), pastel palette")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join("assets", "13_app_preview.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
