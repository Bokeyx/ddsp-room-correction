# Understanding this project (plain language)

> Written so you can follow it even without an audio signal-processing background.

---

## 1. In one sentence

A project where **the computer measures how much a room distorts sound and automatically finds the
"correction (EQ)" that fixes it.**

---

## 2. Why it's needed (the problem)

The same speaker **sounds different in different rooms**. Walls, ceiling, and furniture boost some
notes (boomy resonance) and suppress others. The "originally recorded sound" becomes **uneven** after
passing through the room.

Goal: measure that unevenness and cut/boost the opposite amount to make it **flat (close to the original)**.

---

## 3. Five core concepts (by analogy)

| Term | In plain words |
|------|----------------|
| **RIR (Room Impulse Response)** | The room's "acoustic fingerprint." Think of recording a single hand-clap in the room — it captures all of the room's echo behavior. |
| **Frequency response** | A graph of "which notes (bass/mid/treble) are loud or quiet." Flat = good; uneven = the room is distorting the sound. |
| **EQ (equalizer)** | The "equalizer" in your music app. Correction knobs like *"cut 100 Hz by 5, boost 8000 Hz by 3."* |
| **σ (sigma)** | A **single number for "how uneven it is."** **Lower = flatter = better correction.** This project's main scorecard. |
| **Target curve** | What we treat as ideal. `flat` = perfectly flat; `Harman` = a slightly downward-sloping curve people actually prefer. |

---

## 4. "Why smooth first?"

A raw room measurement comes out **full of fine hairs** (measurement noise). Trying to chase those
hairs with correction actually wrecks the sound.

So we first **smooth it out** to keep only the "big trends," then correct that. In fact, skipping
smoothing in this project makes the score (σ) **6× worse** — which proves *why* smoothing is essential.

---

## 5. The three correction methods (the heart of the comparison)

| Method | Analogy | Character |
|--------|---------|-----------|
| **Classic EQ (greedy)** | A person turning knobs **one at a time** by hand | Simple, traditional. Even with more knobs, it **plateaus (saturates)** past a point. |
| **DDSP (the star)** | A robot fine-tuning **all knobs at once**, the way ML trains (gradient descent) | Accounts for interactions between knobs → keeps **getting better** as knobs increase. |
| **FIR** | One big filter that **inverts the room's distortion wholesale** (4097 numbers) | Precise but a **black box** — you can't read "what it did and why." |

**Key result**: DDSP, with **48 readable knobs**, matches or beats the **4097-number black-box FIR**.

---

## 6. Reading the results

Score (σ, lower is better):

```
before:   0.68   ← uneven room
classic:  0.41   ← 41% flatter
FIR:      0.25
DDSP:     0.23   ← flattest (66% better) ★
```

- **Top graph in the README**: gray (before, jagged) → colored lines (after, flattened).
- **nf-sweep graph**: x-axis = number of knobs. Green (classic) **flattens out and stops (saturates)**;
  red (DDSP) **keeps dropping (improving)**. That's the evidence for "why DDSP wins."

---

## 6b. Beyond the synthetic test (what was added later)

- **Real measured rooms (MIT IR Survey).** The numbers above use a synthetic room. We also validated on
  **20 real measured rooms** (bedrooms, offices, classrooms). Average σ: before 4.20 → classic 1.27 →
  FIR 1.31 → **DDSP 0.91**. DDSP is the flattest *and* the most consistent room-to-room. **Honest twist:**
  on these short, noisy real measurements **FIR drops behind both EQ methods** — the synthetic ranking
  doesn't transfer blindly, which is exactly why measuring real data matters.
- **DDSP can learn more than volume.** Originally DDSP only learned each knob's *amount* (gain). Since the
  math is differentiable, it can also learn **where** each knob sits (frequency) and **how wide** it is (Q).
  Ablation on the headline room: gains-only σ 0.232 → gains+freq+Q σ **0.230** (a little flatter). The
  knobs are kept inside a safe range so training can't push them somewhere that breaks the sound.
- **Music A/B.** Besides pink noise, there's a short license-clean **synthesized music clip** played
  through the room before/after correction — easier to *hear* the tonal balance even out.
- **The top octave, at full rate.** The MIT mirror is 16 kHz, so it stops just under 8 kHz. A second
  measured set — the 48 kHz **Aachen AIR** rooms — reaches the **8–20 kHz top octave**: on a representative
  room DDSP flattens it from σ 1.06 → 0.33. Honest nuance: over the *whole* band the three methods land
  close (classic 0.62 and DDSP 0.63 are a near-tie) because 48 filters now cover a 3× wider band — the
  synthetic ranking doesn't transfer unchanged, which is the point of checking real, full-rate data.
- **An ear-shaped objective.** The plain loss treats every FFT bin equally, but those bins crowd the
  treble, so it over-weights highs. Weighting the loss the way we hear (critical-band density + the
  ISO 226 equal-loudness curve) only matters when filters are scarce: under a tight 12-filter budget the
  hearing-weighted fit lowers a perceptual flatness score (1.04 → 0.95) by spending its filters where
  they're audible — accepting a slightly worse raw σ. A clear, honest trade-off (notebook §4c).
- **You can take the correction home.** A designed correction used to vanish when the browser tab closed.
  The app now exports it to formats real audio tools import — **Equalizer APO** config, **REW** filter
  list, a **FIR impulse WAV** for convolution engines, and a **CSV** archive — so the result can actually
  EQ the system that plays your music, not just a plot.
- **Not cherry-picked.** Beyond averages, DDSP is the flattest in **14 of 20** real rooms (FIR 6,
  classic 0), and its edge over the classic EQ is statistically significant (paired Wilcoxon signed-rank,
  **p ≈ 8.2e-05**) — the improvement is real, not a lucky room.
- **Anyone can try it.** The live app no longer needs a measurement file: pick a named example room
  (simulated) and play **your own music** to hear the before/after through the room — no jargon, no upload
  required. Power users can still upload a real measured RIR.

---

## 7. How to see it yourself

```bash
# 1) Setup
python -m venv .venv
.venv\Scripts\activate        # (Mac/Linux: source .venv/bin/activate)
pip install -r requirements-dev.txt

# 2) Open the notebook with all graphs and explanations (recommended)
jupyter notebook notebooks/room_correction.ipynb

# 3) Interactive web demo
streamlit run app.py

# 4) Check that the code is correct
pytest
```

The notebook also has **play buttons to hear the before/after** directly.

---

## 8. What's in the folders

```
src/         the actual feature code (one file = one job)
  io.py        read/write audio files
  synthetic.py make a fake room (test RIR)
  analysis.py  sound → frequency graph + smoothing
  targets.py   target curves (flat / Harman)
  metrics.py   compute the score (σ)
  eq_classic.py classic EQ
  eq_ddsp.py    DDSP (ML optimization) ★ the star
  fir.py        FIR
  audio.py      apply correction to real sound + the demo music clip
  datasets.py   list MIT IR Survey rooms + load Aachen AIR 48 kHz (.mat) rooms
  evaluation.py before/after σ per room (multi-room & multi-seed studies)
  export.py     correction → Equalizer APO / REW / FIR WAV / CSV files
  i18n.py       EN/KO UI strings for the app's language toggle
  rooms.py      friendly preset rooms for the app (simulated)
  charts.py     tidy data for the app's interactive chart
  pipeline.py   one entry point that calls the methods
scripts/     download_mit_rir.py (fetch the real RIRs, gitignored data)
tests/       automated checks that the code is correct (122)
notebooks/   analysis story + graphs
app.py       Streamlit web demo
assets/      generated graphs and audio
```

---

## 9. What this project demonstrates (portfolio view)

- Ability to **compare traditional signal processing (classic EQ, FIR) with machine learning (DDSP)**
  on one problem.
- **Honest** treatment of results: not just what works, but limitations too ("fails without smoothing,"
  "σ doesn't see phase").
- The full **measure → analyze → optimize → validate loop**, implemented end to end.
