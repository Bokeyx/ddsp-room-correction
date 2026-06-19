# Deploy the demo as a public web app

The Streamlit app (`app.py`) can run as a public URL on **Streamlit Community Cloud** — free, and it
redeploys automatically on every push to `main`. The repo is already set up for it:

- `requirements.txt` — Python deps (includes `streamlit`, `torch`, `soundfile`)
- `packages.txt` — system package `libsndfile1` (needed by `soundfile` on Linux)
- `.streamlit/config.toml` — the pastel theme

## One-time setup (about 3 minutes, done in the browser)

1. Go to **https://share.streamlit.io** and sign in with the GitHub account that owns
   `Bokeyx/ddsp-room-correction`.
2. Click **Create app → Deploy a public app from GitHub**.
3. Fill in:
   - **Repository:** `Bokeyx/ddsp-room-correction`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **Deploy**. The first build takes a few minutes (it installs `torch`).
5. You get a public URL like `https://ddsp-room-correction.streamlit.app`. Anyone can open it.

That's it. After this, every `git push` to `main` redeploys the live app automatically — no further steps.

## After you have the URL

Paste it back here and the README "Live demo" badge will be pointed at it.

## Notes

- Free tier has ~1 GB RAM. The synthetic-room default and uploaded RIRs work; the DDSP method takes a
  few seconds per run on the shared CPU (cached after the first run for the same settings).
- If the app ever sleeps from inactivity, the first visitor's load wakes it (a few seconds).
- Hugging Face Spaces is an alternative host (also free); the same `requirements.txt` / `packages.txt`
  work there with a Streamlit SDK Space.
