import numpy as np


# Target-curve functions follow the signature `freqs_hz -> target_db`
# (so a harman_target can be added later with the same interface).
def flat_target(freqs_hz):
    return np.zeros_like(np.asarray(freqs_hz, dtype=np.float64))
