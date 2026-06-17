"""Fetch the MIT IR Survey room impulse responses into data/public/MIT_Survey.

Data: MIT Acoustical Reverberation Scene Statistics Survey
      (Traer & McDermott, "Statistics of natural reverberation enable
       perceptual separation of sound and space", PNAS 2016).
Source used here: the davidscripka/MIT_environmental_impulse_responses mirror on
      Hugging Face, which stores the 271 survey IRs as individual 16 kHz mono
      wavs. The original site (mcdermottlab.mit.edu) serves a full-rate Audio.zip
      but is frequently offline; the mirror is what this project was built on, so
      we pin to it for reproducibility.

Free for research/education; cite the PNAS paper above. The wavs are NOT
committed to this repo (data/public is gitignored) -- run this once locally.

Requires git + git-lfs on PATH and network access to huggingface.co. Some
networks block Hugging Face; if so, clone the repo from an unblocked network and
copy its `16khz` folder into data/public/MIT_Survey yourself.

    python scripts/download_mit_rir.py
"""
import shutil
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://huggingface.co/datasets/davidscripka/MIT_environmental_impulse_responses"
ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "data" / "public" / "MIT_Survey"
TMP = ROOT / "data" / "public" / "_mit_clone_tmp"


def _run(cmd, **kw):
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, **kw)


def main():
    existing = sorted(DEST.glob("*.wav"))
    if existing:
        print(f"Already present: {len(existing)} wavs in {DEST} -- nothing to do.")
        return

    if shutil.which("git") is None:
        sys.exit("git not found on PATH. Install git first.")
    if shutil.which("git-lfs") is None:
        sys.exit("git-lfs not found on PATH. The wavs are LFS-tracked; without "
                 "git-lfs the clone yields tiny pointer files, not audio. "
                 "Install git-lfs (https://git-lfs.com) and run `git lfs install`.")

    if TMP.exists():
        shutil.rmtree(TMP)
    DEST.mkdir(parents=True, exist_ok=True)

    try:
        _run(["git", "clone", "--depth", "1", REPO_URL, str(TMP)])
        wavs = sorted((TMP / "16khz").glob("*.wav"))
        if not wavs:
            sys.exit("Clone succeeded but no wavs found under 16khz/.")
        # Guard against LFS pointer files (~130 bytes of text) sneaking through.
        if wavs[0].stat().st_size < 1000:
            sys.exit("Cloned wavs look like LFS pointers, not audio. Run "
                     "`git lfs install` and try again.")
        for w in wavs:
            shutil.copy2(w, DEST / w.name)
        print(f"Copied {len(wavs)} wavs to {DEST}")
    finally:
        if TMP.exists():
            shutil.rmtree(TMP)


if __name__ == "__main__":
    main()
