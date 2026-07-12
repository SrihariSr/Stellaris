# Project Stellaris

A deep learning pipeline for exoplanet transit detection in NASA Kepler and TESS photometric
data. A CNN is trained on Kepler, frozen, and deployed unchanged on three TESS populations to
measure **which stage of the pipeline fails, and where**.

---

## What this does

Stellaris takes raw light-curve data from NASA's TESS mission, searches it for periodic
transit-like signals, scores those signals with a Kepler-trained convolutional neural network,
and produces a ranked list of candidates with vetting diagnostics.

The pipeline was trained on 6,726 Kepler Objects of Interest, then deployed **unchanged** across
three TESS regimes of decreasing prior information:

- **1,102 confirmed TESS planets** — for measuring recall. Recovered 62.2% at CNN ≥ 0.5 with both
  vetting rules passed.
- **4,002 unresolved TESS Objects of Interest** — for ranking. Endorsed 1,564 at CNN ≥ 0.9.
  Screening the 30 highest-ranked, 10 survive all four checks. **This is a positive control on the
  ranking, not a discovery**: all 30 are catalogued TOIs, and all 10 survivors are known planet
  candidates whose published ephemerides the pipeline independently recovered.
- **474 faint stars (Tmag 13-15)** absent from the TOI catalogue — for exploration. One target
  cleared the CNN threshold and both vetting rules (TIC 155810890), then **failed** the four-check
  screening: the transit appears in only one of the three sectors observed. **Screened yield is
  zero.**

**The faint-star null is not a sensitivity limit.** The median faint-star detection is 10,525 ppm
deep, nearly *double* the confirmed-planet median of 5,319 ppm. These are not shallow transits the
model is missing. They are deep, low-significance detections the *search stage* is manufacturing:
41.6% of faint-star BLS periods fall within 10% of the 13.7-day TESS orbital period, and all four
signals the network scores above 0.9 are among them. Because the views are depth-normalised, the
network cannot see depth, and so cannot reject these systematics.

---

## Headline results

| Metric | Value |
|---|---|
| Held-out Kepler test PR-AUC (released checkpoint) | **0.945** |
| Held-out Kepler test ROC-AUC | **0.959** |
| XGBoost baseline (19 hand-engineered statistics of the same views) | 0.918 |
| **Local-view-only CNN** (1.32M params, 7.3× smaller) | **0.943 ± 0.003** |
| **Full two-view CNN** (9.71M params) | **0.944 ± 0.006** |
| Global-view-only CNN (8.91M params) | 0.433 ± 0.026 |
| Recovery of confirmed TESS planets (frozen transfer, no fine-tuning) | **62.2%** |
| Endorsements (CNN ≥ 0.9) across unresolved TESS catalogue | **1,564** |
| Survivors of four-check screening, from top 30 | **10 / 30** |
| Screened faint-star candidates | **0** |

Neural rows are mean ± sample s.d. over **ten training seeds (0–9)** on a fixed grouped split.

**The two-view architecture is not earning its parameters.** Over ten seeds, the local branch alone
is statistically indistinguishable from the full network (difference +0.0005 PR-AUC, 95% CI
[−0.004, +0.005], Welch p = 0.83), while consuming 86% fewer parameters. The global branch alone is
near-random.

---

## How it works

```
NASA Kepler / TESS FITS files
            ↓
Light curve extraction (PDCSAP flux via lightkurve)
            ↓
Quality masking + one-sided 5σ clip + wotan median detrend (1.0 day window)
            ↓
Box Least Squares search: 50,000 trial periods (0.5–20 d), 8 durations (0.5–8 h)
            ↓
Phase-folded views: 2001-bin global + 201-bin local
   NOTE: each view is divided by abs(min), so the deepest bin is always exactly -1.
   The network sees transit SHAPE and is architecturally blind to transit DEPTH.
            ↓
StellarisNet two-branch CNN (9.7M params, PyTorch)
            ↓
Rule-based vetting: depth cap (3% of stellar flux) + odd/even consistency (15% max diff)
            ↓
Four-check screening (top-ranked endorsements with BLS S/N ≥ 10 only):
  • Secondary eclipse check (phase ±0.5, flags EBs)
  • TESS systematic period check (1, 2, 6.85, 13.7, 27.4 d at half/unit/double/triple, 5% tol)
  • Period alias check (masks primary, re-runs BLS on residuals)
  • Sector consistency check (transit must reproduce across sectors)
            ↓
Ranked candidate list with full diagnostics
```

---

## Repository structure

```
├── README.md
├── pyproject.toml
│
├── data/
│   ├── catalogs/                      # KOI catalogue, TOI catalogue, SPOC target lists
│   │   ├── kepler_koi.csv
│   │   └── tess_toi.csv               # TFOPWG dispositions; ground truth for R1
│   ├── raw/                           # Cached FITS files (gitignored, ~150 GB)
│   └── processed/                     # (gitignored)
│       ├── kepler_trainset.h5         # 6,726 examples, built by build_kepler_trainset.py
│       └── split.json                 # Written by dump_split.py; the CNN's own split
│
├── src/stellaris/                     # Core modules
│   ├── fetch.py                       # Kepler light-curve fetching
│   ├── tess_fetch.py                  # TESS light-curve fetching
│   ├── preprocess.py                  # Kepler preprocessing
│   ├── tess_preprocess.py             # TESS preprocessing
│   ├── views.py                       # Global / local views. _normalise() divides each view by
│   │                                  #   abs(min), so the network cannot see absolute depth
│   ├── dataset.py                     # PyTorch Dataset; make_splits() groups by kepid (seed 42)
│   ├── model.py                       # StellarisNet. use_global / use_local flags drive the ablation
│   ├── bls_search.py                  # BLS wrapper for transit detection
│   └── vetting.py                     # Depth cap (3%) and odd-even (15%) rules
│
├── scripts/
│   │  # --- data construction ---
│   ├── fetch_catalogs.py              # Build catalogue files
│   ├── fetch_kepler_lightcurves.py    # Download Kepler light curves
│   ├── build_kepler_trainset.py       # Build the HDF5 training set
│   ├── build_tic_target_list.py       # TIC list from the TOI catalogue
│   ├── build_unresolved_list.py       # R2 target list (unresolved TOIs)
│   ├── build_novel_target_list.py     # R3 target list (Tmag 13-15, multi-sector, non-TOI)
│   │
│   │  # --- training and evaluation ---
│   ├── train.py                       # Train StellarisNet (see reproducibility note below)
│   ├── evaluate.py                    # Held-out Kepler test set
│   ├── dump_split.py                  # MUST run before the XGBoost baseline, or the baseline
│   │                                  #   scores on a different test set from the CNN
│   ├── train_xgboost_baseline.py      # 19 hand-engineered statistics of the same two views
│   ├── run_ablation.py                # 3 variants × 10 seeds → the ablation table above
│   │
│   │  # --- deployment ---
│   ├── hunting.py                     # BLS → CNN → vetting, across all three regimes
│   ├── validate_candidates.py         # Four-check screening
│   ├── inspect_candidates.py          # Phase-fold diagnostic plots
│   │
│   │  # --- reproducibility ---
│   └── verify_paper_numbers.py        # Recomputes EVERY quantitative claim from the artefacts
│                                      #   below and prints PASS/FAIL. Exits non-zero on failure.
│
├── checkpoints/
│   └── stellaris_best.pt              # Frozen model used for all TESS results (epoch 33)
│
├── figures/
│   └── r3_period_histogram.pdf        # The 13.7-day pile-up on faint stars
│
└── results/
    │  # --- Kepler ---
    ├── test_predictions.npz           # Released checkpoint: 0.945 PR-AUC on 1,020 examples
    ├── ablation_results.json          # 30 runs (3 variants × 10 seeds)
    ├── xgboost_baseline_metrics.json  # 0.918 PR-AUC, on the CNN's own split
    │
    │  # --- the three TESS regimes ---
    ├── tier1_results.csv              # R1: confirmed planets (1,102 rows)
    ├── tier3_results.csv              # R2: unresolved TOIs (4,002 rows)
    ├── novel_results.csv              # R3: faint, unflagged stars (474 rows)
    ├── novel_failures.log             # 1,524 of 1,526 R3 losses occur at view construction
    │
    │  # --- screening ---
    ├── validation/
    │   └── validation_summary.csv     # R2: 30 highest-ranked screened, 10 survive
    ├── validation_novel/
    │   └── validation_summary.csv     # R3: TIC 155810890 FAILS (transit in 1 of 3 sectors)
    │
    └── candidate_plots/               # Phase-folds for top-ranked targets
```

---

## Setup

### Requirements

- Python 3.14+
- ~50 GB free disk for the Kepler training data, plus 50–150 GB more for cached TESS FITS
- Apple Silicon Mac or a CUDA-capable GPU recommended; CPU works but is slow

### Installation

```bash
git clone https://github.com/SrihariSr/Stellaris.git
cd Stellaris
python3.14 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Key dependencies

- `torch` — the CNN
- `lightkurve` — NASA mission data access
- `astropy` — FITS handling and BLS
- `wotan` — light-curve detrending
- `numpy`, `pandas`, `h5py`, `scipy`, `scikit-learn`
- `xgboost` — baseline model
- `matplotlib` — figures

---

## Quickstart

### 1. Build the training set

```bash
python scripts/fetch_catalogs.py            # KOI and TOI catalogues
python scripts/fetch_kepler_lightcurves.py  # ~41 GB, takes hours
python scripts/build_kepler_trainset.py     # → data/processed/kepler_trainset.h5
```

### 2. Train

```bash
PYTHONPATH=src python scripts/train.py
```

Writes `checkpoints/stellaris_best.pt`. Best validation PR-AUC 0.9463 at epoch 33.

### 3. Evaluate on the held-out Kepler test set

```bash
PYTHONPATH=src python scripts/evaluate.py
```

### 4. Baseline and ablation

**Order matters.** `dump_split.py` writes the CNN's own train/val/test split to disk. Without it,
`train_xgboost_baseline.py` falls back to a different RNG and scores on a **different set of
stars**, making the comparison meaningless.

```bash
PYTHONPATH=src python scripts/dump_split.py             # → data/processed/split.json
PYTHONPATH=src python scripts/train_xgboost_baseline.py # 0.918 PR-AUC
PYTHONPATH=src python scripts/run_ablation.py           # 30 runs, ~25 min on an M4 Max
```

### 5. Run a hunt on TESS targets

```bash
PYTHONPATH=src python scripts/build_novel_target_list.py   # or build_unresolved_list.py

caffeinate -i python scripts/hunting.py \
  --tic-list data/catalogs/<target_list>.txt \
  --output results/<name>_results.csv \
  --failures results/<name>_failures.log
```

Fetches TESS light curves from MAST, preprocesses, runs BLS, scores with the CNN, applies vetting.
One row per target. Resumable — the CSV is the state, so Ctrl+C is safe.

### 6. Screen the top candidates

```bash
PYTHONPATH=src python scripts/validate_candidates.py \
  --results results/tier3_results.csv \
  --threshold 0.9 \
  --min-snr 10 \
  --max-candidates 30
```

### 7. Verify every number

```bash
PYTHONPATH=src python scripts/verify_paper_numbers.py
```

Recomputes every quantitative claim in this README and the accompanying paper from the committed
artefacts, and prints PASS or FAIL for each. Exits non-zero on any failure.

---

## Model architecture

**StellarisNet** is a two-branch 1D CNN with **9,710,305 parameters**, following Shallue &
Vanderburg (2018). **9,046,017 of them (93.2%) sit in the fully connected head.**

Each ConvBlock is: `Conv1d(kernel 5, same padding) → ReLU → Conv1d → ReLU → MaxPool(kernel 5, stride 2)`.

**There is no batch normalisation anywhere in the network.**

**Global branch** (input: 2001-bin phase-folded light curve)
- 5 ConvBlocks, channels 1 → 16 → 32 → 64 → 128 → 256

**Local branch** (input: 201-bin view spanning ±4 transit durations)
- 2 ConvBlocks, channels 1 → 16 → 32

**Head**
- Concatenate both branches
- 3 × [Linear(512) → ReLU → Dropout(0.5)]
- Single logit → sigmoid → probability

**Training** (exactly as set in `scripts/train.py`)
- `BCEWithLogitsLoss` with `pos_weight = n_neg / n_pos ≈ 1.46`
- Adam, **learning rate 1e-4**, **weight decay 1e-5**
- **Batch size 64**, 40 epochs
- Best validation PR-AUC at epoch 33
- Train / val / test split grouped by `kepid` (4,680 / 1,026 / 1,020)

### The ablation

`scripts/run_ablation.py` trains three variants × ten seeds on the fixed split:

| Variant | Params | PR-AUC | ROC-AUC | R@95%P |
|---|---|---|---|---|
| Global view only | 8,914,737 | 0.433 ± 0.026 | 0.551 ± 0.046 | 0.000 ± 0.000 |
| **Local view only** | **1,321,905** | **0.943 ± 0.003** | 0.958 ± 0.002 | 0.685 ± 0.026 |
| Both views | 9,710,305 | 0.944 ± 0.006 | 0.959 ± 0.004 | 0.674 ± 0.047 |

The 8.4M-parameter global branch, **86% of the network**, buys at most 0.005 PR-AUC at 95%
confidence. At the 95%-precision operating point the point estimate actually *favours* the smaller
model. This is a negative architectural result: at 4,680 training examples, the two-view design is
not paying for itself.

---

## Screening framework

`scripts/validate_candidates.py` runs four false-positive checks beyond the basic vetting rules, on
endorsed candidates with BLS S/N ≥ 10:

1. **Secondary eclipse search.** Examines phase ±0.5 for a dip indicating an eclipsing binary.
2. **TESS systematic period check.** Rejects candidates within 5% of `{1, 2, 6.85, 13.7, 27.4}` days
   at half, unit, double or triple period.
3. **Period alias check.** Masks the primary signal and re-runs BLS on the residuals.
4. **Per-sector consistency.** The transit must reproduce across observed sectors.

**This is screening, not validation.** It computes no false-positive probability. Tools like
TRICERATOPS or vespa model blended eclipsing binaries, hierarchical triples and background
contaminants explicitly. This is a pre-filter, not a replacement for community vetting.

---

## Limitations and honest caveats

- **No centroid analysis.** TESS pixels are 21 arcseconds wide and often contain several stars. A
  faint background eclipsing binary blended into a target's pixel produces a diluted signal that is
  indistinguishable from a planet by shape alone. This pipeline does not catch those.

- **No statistical validation, no follow-up.** The pipeline produces candidates, not confirmations.

- **The network is blind to transit depth.** `_normalise()` in `views.py` divides each view by
  `abs(min)`, so the deepest bin is always exactly −1. The network classifies on shape only. This is
  why TOI-2533, a confirmed eclipsing binary, scored 0.988, and why the 3% depth cap is doing work
  the network architecturally cannot.

- **Search-stage failure on faint stars.** The faint-star hunt produced one endorsed candidate from
  474 usable targets, and it **failed screening** on sector consistency. Across all 474 targets,
  41.6% have a BLS period within 10% of the 13.7-day TESS orbital period, and all four targets
  scoring above 0.9 are among them. The classifier does not discriminate against the systematic —
  its median score on near-13.7-day detections is 0.030, against 0.032 elsewhere. It is
  *indifferent*, not selective. Fine-tuning on TESS labels would likely help.

- **The faint sample is depleted by construction.** R3 targets are absent from the TOI catalogue,
  meaning SPOC has already searched them and flagged nothing. An endorsement rate on such a sample
  approximates the false-positive rate. **Without injected signal we cannot separate a low base rate
  from low recall.** This is the largest weakness of the design; injection-recovery is the fix.

- **Conservative screening.** The systematic-period check flags anything within 5% of an
  instrumental period, which can reject real planets. Single-sector candidates fail the
  sector-consistency check by default, where reproduction is untestable rather than falsified.

**What this means:** candidates surfaced here are worth investigating, not declaring discovered.
Treat the output as a prioritisation tool.

---

## Reproducibility

```bash
PYTHONPATH=src python scripts/verify_paper_numbers.py
```

recomputes every number quoted here from the committed artefacts and prints PASS/FAIL.

**Known gaps, stated plainly:**

- **`scripts/train.py` sets no random seed.** The released checkpoint
  (`checkpoints/stellaris_best.pt`) is therefore **not bit-reproducible**. Its test PR-AUC (0.945)
  sits within 0.2 s.d. of the ten-seed mean (0.944 ± 0.006), so it is a typical draw rather than a
  favourable one, but re-running `train.py` will not reproduce it exactly.
  `scripts/run_ablation.py` seeds explicitly (seeds 0–9) and is reproducible.
- **The train/val/test split is deterministic** (`GroupShuffleSplit`, `random_state=42`, grouped by
  `kepid`), so no star crosses partitions and the test set is stable across runs.
- **TESS hunt outputs depend on MAST availability at run time.** Results may shift slightly if MAST
  data is updated.

---

## Hardware notes

Developed on an Apple MacBook Pro (M4 Max, 36 GB RAM) using PyTorch's MPS backend. The code is
device-agnostic.

Approximate runtimes:

- Training, 40 epochs: **~1 minute**
- Full ablation (30 runs): **~25 minutes**
- Test evaluation: ~5 seconds
- Per-target inference: 15–30 s, mostly bottlenecked by MAST download speed
- Confirmed-planet hunt (1,190 targets): ~7.5 hours
- Unresolved-candidate hunt (5,107 targets): ~7–8 hours

---

## Scope and prioritisation

> *He who defends everything defends nothing.* — Frederick the Great

On a single-machine budget, ranking already-flagged TESS candidates and hunting new faint-star
signals could not both be done at depth, so the known-target work was prioritised. The pipeline
ranks 4,002 unresolved candidates and puts the strongest through four-check screening; 1,564 are
endorsed at CNN ≥ 0.9 and 10 of the top 30 survive all four checks.

The faint-star hunt was kept deliberately bounded (2,000 targets, 474 usable) and surfaced one
endorsed candidate, which then failed screening. **That null is the informative result, and its
cause is specific:** BLS locks onto the 13.7-day TESS orbital systematic on faint stars, and a
depth-blind CNN cannot reject it. The attrition is itself evidence of this — 1,524 of the 1,526
losses occur at view construction, on noise-dominated curves where the search returns short-period,
maximum-duration solutions.

---

## Prior work

- **Shallue & Vanderburg (2018)**, *Identifying Exoplanets with Deep Learning*. The two-branch CNN
  architecture this work reimplements.
- **Yu et al. (2019)**, **Osborn et al. (2020)** — CNNs trained directly on TESS labels.
- **Valizadegan et al. (2022)**, *ExoMiner* — validation-grade classification with a wider
  diagnostic set.

This project's contribution is not a new architecture. It is an empirical account of **which stage
of a frozen cross-mission pipeline fails**: an ablation showing 86% of the parameters are
redundant, a per-stage failure decomposition attributing the shortfall to the network rather than
the vetting rules, and a mechanistic explanation of the faint-star null.

---

## License

MIT (see `LICENSE`).

---

## Acknowledgements

This work uses public data from NASA's Kepler and TESS missions, accessed via the Mikulski Archive
for Space Telescopes (MAST) at the Space Telescope Science Institute. The TESS Objects of Interest
catalogue is maintained by the TESS Follow-up Observing Program Working Group (TFOPWG). Per-sector
SPOC target lists are made available by the MIT TESS team.

---

## Author

**Srihari Srinivasan, 2026**