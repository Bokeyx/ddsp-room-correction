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
  audio.py      apply correction to real sound
  pipeline.py   one entry point that calls the methods
tests/       automated checks that the code is correct (53)
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
