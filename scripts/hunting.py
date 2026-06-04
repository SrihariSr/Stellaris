"""Batch TESS exoplanet hunt: fetch -> BLS -> CNN -> vetting -> ranked CSV.

Run on a list of TIC IDs. For each target:
  1. Fetch and preprocess the light curve from MAST.
  2. Run BLS to find the strongest periodic signal.
  3. Generate global+local views from BLS detection.
  4. Score with the trained CNN.
  5. Apply rule-based vetting.

Output: results/hunt_results.csv with one row per target.
Resumable: re-running skips targets already in the output CSV.
"""
from pathlib import Path
import time as timer
import csv
import argparse

import numpy as np
import torch
from tqdm import tqdm

from stellaris.tess_fetch import fetch_tess_lightcurve
from stellaris.tess_preprocess import preprocess_tess
from stellaris.bls_search import run_bls
from stellaris.views import make_global_view, make_local_view
from stellaris.model import StellarisNetwork
from stellaris.vetting import vet_candidate


CHECKPOINT_PATH = Path("checkpoints/stellaris_best.pt")
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)
DEFAULT_OUTPUT = RESULTS / "hunt_results.csv"
DEFAULT_FAILURES = RESULTS / "hunt_failures.log"

DEVICE = (
    'mps' if torch.backends.mps.is_available()
    else 'cuda' if torch.cuda.is_available()
    else 'cpu'
)

CSV_COLUMNS = [
    "tic_id", "period_days", "epoch_btjd", "duration_hours",
    "depth_ppm", "bls_snr", "cnn_probability",
    "vetting_passed", "vetting_reasons", "rank_score",
]


def load_model():
    ckpt = torch.load(CHECKPOINT_PATH, weights_only=False, map_location=DEVICE)
    model = StellarisNetwork().to(DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


def process_target(tic_id: int, model) -> dict | None:
    """Process one TIC. Returns a result dict, or None on failure."""
    lcc = fetch_tess_lightcurve(tic_id)
    clean = preprocess_tess(lcc)

    time_arr = clean.time.value
    flux_arr = clean.flux.value

    if len(time_arr) < 5000:
        raise ValueError(f"Too few cadences ({len(time_arr)}); skipping")

    bls = run_bls(time_arr, flux_arr, min_period=0.5, max_period=20.0)

    global_view = make_global_view(
        time_arr, flux_arr, bls.period, bls.epoch
    )
    local_view = make_local_view(
        time_arr, flux_arr, bls.period, bls.epoch, bls.duration
    )

    g_tensor = torch.from_numpy(global_view).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
    l_tensor = torch.from_numpy(local_view).float().unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logit = model(g_tensor, l_tensor)
        probability = float(torch.sigmoid(logit).item())

    vet = vet_candidate(
        time_arr, flux_arr,
        period=bls.period, epoch=bls.epoch,
        duration=bls.duration, depth=bls.depth,
    )

    # Rank score: CNN probability if vetting passes, else 0 (effectively buries failures)
    rank_score = probability if vet.passed else 0.0

    return {
        "tic_id": tic_id,
        "period_days": round(bls.period, 6),
        "epoch_btjd": round(bls.epoch, 4),
        "duration_hours": round(bls.duration * 24, 3),
        "depth_ppm": round(abs(bls.depth) * 1e6, 1),
        "bls_snr": round(bls.snr, 2),
        "cnn_probability": round(probability, 4),
        "vetting_passed": vet.passed,
        "vetting_reasons": "; ".join(vet.reasons) if vet.reasons else "",
        "rank_score": round(rank_score, 4),
    }


def load_existing_results(output_path: Path) -> set[int]:
    """Return set of TIC IDs already in the output CSV (for resumability)."""
    if not output_path.exists():
        return set()
    seen = set()
    with open(output_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            seen.add(int(row["tic_id"]))
    return seen


def append_result(output_path: Path, row: dict, write_header: bool):
    """Append one result row. Writes header if requested."""
    with open(output_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tic-list",
        type=Path,
        required=True,
        help="Path to a text file with one TIC ID per line.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--failures", type=Path, default=DEFAULT_FAILURES)
    args = parser.parse_args()

    # Load TIC list
    with open(args.tic_list) as f:
        tics = [int(line.strip()) for line in f if line.strip() and not line.strip().startswith("#")]
    print(f"Targets to process: {len(tics)}")

    # Skip already-processed
    seen = load_existing_results(args.output)
    todo = [t for t in tics if t not in seen]
    print(f"Already processed: {len(seen)}")
    print(f"To process now: {len(todo)}")

    if not todo:
        print("Nothing to do. Existing results in:", args.output)
        return

    # Load model once
    print(f"\nDevice: {DEVICE}")
    print(f"Loading model from {CHECKPOINT_PATH}")
    model = load_model()

    # Process
    write_header = not args.output.exists()
    failures = []
    t_start = timer.time()

    for tic in tqdm(todo, desc="Hunting"):
        try:
            result = process_target(tic, model)
            if result is None:
                failures.append((tic, "process returned None"))
                continue
            append_result(args.output, result, write_header)
            write_header = False
        except Exception as e:
            failures.append((tic, str(e)[:200]))
            continue

    elapsed = timer.time() - t_start
    print(f"\nProcessed {len(todo)} in {elapsed/60:.1f} min "
          f"({elapsed/max(1, len(todo)):.1f}s/target)")
    print(f"Successes: {len(todo) - len(failures)}")
    print(f"Failures:  {len(failures)}")

    if failures:
        with open(args.failures, 'a') as f:
            for tic, reason in failures:
                f.write(f"{tic}\t{reason}\n")
        print(f"Failures logged to {args.failures}")

    print(f"\nResults in {args.output}")


if __name__ == '__main__':
    main()