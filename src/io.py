import numpy as np
import soundfile as sf


def load_wav(path):
    signal, samplerate = sf.read(path, dtype="float64")
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    return signal, samplerate


def save_wav(path, signal, samplerate):
    sf.write(path, signal, samplerate)
