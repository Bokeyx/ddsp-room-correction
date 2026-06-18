"""Helpers for the MIT IR Survey on disk (Traer & McDermott, PNAS 2016).

The survey mixes enclosed rooms (bedrooms, offices, classrooms) with open-air
and transit spaces (streets, campus, parking lots). Room correction only makes
sense for enclosed rooms with real modal coloration, so ``list_rirs`` filters
out the outdoor/transit recordings by default.

File names follow ``h<NNN>_<RoomType>_<description>.wav``, e.g.
``h001_Bedroom_65txts.wav`` or ``h003_Office_LargeBrick_56txts.wav``.
"""
import os
from glob import glob

import numpy as np
import scipy.io as sio

# Substrings (case-insensitive) that mark an open-air or transit space.
_OUTDOOR_SUBSTRINGS = (
    "outside",
    "outdoor",
    "street",
    "campus",
    "campground",
    "parkinglot",
    "balcony",
    "porch",
    "backyard",
)
# Whole labels that are enclosed but not listening rooms.
_OUTDOOR_EXACT = {"car", "train", "trainstation", "subwaystation", "swimmingpool"}


def rir_label(path):
    """Return the room-type token from an MIT IR Survey file name."""
    stem = os.path.basename(path).rsplit(".", 1)[0]
    parts = stem.split("_")
    return parts[1] if len(parts) > 1 else stem


def is_indoor(label):
    """True for enclosed listening rooms, False for outdoor/transit spaces."""
    low = label.lower()
    if low in _OUTDOOR_EXACT:
        return False
    return not any(token in low for token in _OUTDOOR_SUBSTRINGS)


def list_rirs(data_dir, indoor_only=True):
    """Sorted list of RIR wav paths in ``data_dir``.

    With ``indoor_only`` (default) outdoor and transit recordings are dropped.
    Raises ``FileNotFoundError`` if the directory does not exist.
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"RIR directory not found: {data_dir!r}. "
            "Run scripts/download_mit_rir.py to fetch the dataset."
        )
    paths = sorted(glob(os.path.join(data_dir, "*.wav")))
    if indoor_only:
        paths = [p for p in paths if is_indoor(rir_label(p))]
    return paths


# --- Aachen AIR database (RWTH, 48 kHz measured RIRs in MATLAB .mat files) ----
#
# A full-bandwidth (48 kHz) second opinion that lets us score the top octave the
# 16 kHz MIT mirror cannot reach. Each .mat holds one mono response `h_air` plus
# an `air_info` struct (fs, room, channel, ...). We keep the small/medium
# listening-type rooms and drop the band-limited phone recordings and the
# extreme spaces (stairway, the aula_carolina church hall) that are not rooms
# anyone corrects a speaker for.
_AIR_LISTENING_ROOMS = ("booth", "office", "meeting", "lecture")


def load_air_mat(path):
    """Load one Aachen AIR ``.mat`` as ``(h, fs, room)``.

    ``h`` is the 1-D mono impulse response (float64), ``fs`` the sample rate in
    Hz, ``room`` the room label from the file's ``air_info`` struct.
    """
    m = sio.loadmat(path)
    h = np.ravel(m["h_air"]).astype(np.float64)
    info = m["air_info"][0, 0]
    fs = int(np.ravel(info["fs"])[0])
    room = str(np.ravel(info["room"])[0])
    return h, fs, room


def list_air_rirs(air_dir, rooms=_AIR_LISTENING_ROOMS):
    """Sorted list of Aachen AIR ``.mat`` paths for listening-type rooms.

    Keeps the binaural room measurements whose file name names one of ``rooms``
    and drops the band-limited phone recordings. Raises ``FileNotFoundError`` if
    the directory does not exist.
    """
    if not os.path.isdir(air_dir):
        raise FileNotFoundError(f"AIR directory not found: {air_dir!r}.")
    paths = sorted(glob(os.path.join(air_dir, "*.mat")))
    keep = []
    for p in paths:
        low = os.path.basename(p).lower()
        if "phone" in low or "_hhp" in low or "_hfrp" in low:
            continue
        if any(room in low for room in rooms):
            keep.append(p)
    return keep
