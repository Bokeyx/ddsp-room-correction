"""Render the music A/B demo: a synthesized clip played 'in-room' before vs
after DDSP correction. License-clean (the clip is fully generated).

Run from the repo root:
    .venv\\Scripts\\python.exe notebooks/make_music_demo.py
Outputs to assets/audio/music_{dry,uncorrected,corrected}.wav
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import scipy.signal as sps

from src.audio import apply_eq_to_signal, demo_music
from src.analysis import frequency_response
from src.targets import flat_target
from src.eq_ddsp import optimize_eq
from src.synthetic import decaying_noise_rir
from src.io import save_wav

SR = 48000


def main():
    os.makedirs("assets/audio", exist_ok=True)
    music = demo_music(SR, duration_s=10.0)

    # A full-bandwidth synthetic room (48 kHz) keeps the music wide-band; the
    # real-RIR validation lives in the notebook (section 8).
    rir, sr = decaying_noise_rir(SR, 0.5, 0.4, seed=42)
    freqs, resp = frequency_response(rir, sr)
    eq = optimize_eq(resp, flat_target(freqs), freqs, sr, n_filters=48, iters=150)

    uncorrected = sps.fftconvolve(music, rir)[: len(music)]
    corrected = sps.fftconvolve(apply_eq_to_signal(eq, music, sr), rir)[: len(music)]

    def norm(x):
        return x / np.max(np.abs(x)) * 0.95

    save_wav("assets/audio/music_dry.wav", norm(music), sr)
    save_wav("assets/audio/music_uncorrected.wav", norm(uncorrected), sr)
    save_wav("assets/audio/music_corrected.wav", norm(corrected), sr)
    print(f"wrote music_dry / music_uncorrected / music_corrected "
          f"({len(music) / sr:.1f}s each) to assets/audio/")


if __name__ == "__main__":
    main()
