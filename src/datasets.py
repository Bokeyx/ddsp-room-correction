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
