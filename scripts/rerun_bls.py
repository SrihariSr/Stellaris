"""Re-run the hunt's BLS on cached light curves and write full-precision ephemerides in the hunt CSV layout."""
from pathlib import Path
import argparse
import csv
import multiprocessing as mp
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from stellaris.bls_search import run_bls
from stellaris.targets import PROJECT_ROOT, REGIME_CSVS
from stellaris.tess_cache import DEFAULT_CACHE_DIR, cache_path, load_cache

MIN_CADENCES = 5000
COLUMNS = [
    "tic_id", "period_days", "epoch_btjd", "duration_hours", "depth_ppm", "bls_snr",
    "n_cadences", "matches_hunt_csv", "status",
]


def bls_one(task: tuple[int, str]) -> dict:
    tic_id, cache_dir = task
    row = {c: "" for c in COLUMNS}
    row["tic_id"] = tic_id
    path = cache_path(Path(cache_dir), tic_id)
    if not path.exists():
        return {**row, "status": "missing_cache"}
    data = load_cache(path)
    row["n_cadences"] = len(data["time"])
    if len(data["time"]) < MIN_CADENCES:
        return {**row, "status": "cadence_floor"}
    try:
        bls = run_bls(data["time"], data["flux"], min_period=0.5, max_period=20.0)
    except Exception as exc:
        return {**row, "status": "bls_error:" + str(exc).splitlines()[0][:120]}
    return {
        **row,
        "period_days": bls.period,
        "epoch_btjd": bls.epoch,
        "duration_hours": bls.duration * 24,
        "depth_ppm": abs(bls.depth) * 1e6,
        "bls_snr": bls.snr,
        "status": "ok",
    }


def matches_hunt(row: dict, hunt: pd.DataFrame) -> str:
    """Whether rounding as hunting.py does reproduces the committed hunt row."""
    if row["status"] != "ok" or row["tic_id"] not in hunt.index:
        return ""
    ref = hunt.loc[row["tic_id"]]
    return str(
        round(row["period_days"], 6) == ref.period_days
        and round(row["epoch_btjd"], 4) == ref.epoch_btjd
        and round(row["duration_hours"], 3) == ref.duration_hours
        and round(row["depth_ppm"], 1) == ref.depth_ppm
        and round(row["bls_snr"], 2) == ref.bls_snr
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime", choices=["r0", "r1", "r2", "r3"], required=True)
    parser.add_argument("--regime-csv", type=Path, default=None, help="Hunt CSV giving the targets.")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--out", type=Path, default=None,
                        help="Default results/camera_ready/bls_full/{regime}_bls_full.csv")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    regime_csv = args.regime_csv or REGIME_CSVS[args.regime]
    out = args.out or PROJECT_ROOT / "results" / "camera_ready" / "bls_full" / f"{args.regime}_bls_full.csv"
    hunt = pd.read_csv(regime_csv)
    if args.limit is not None:
        hunt = hunt.head(args.limit)
    hunt = hunt.drop_duplicates("tic_id").set_index("tic_id")

    done: set[int] = set()
    if out.exists():
        done = {int(t) for t in pd.read_csv(out).tic_id}
    todo = [int(t) for t in hunt.index if int(t) not in done]
    print(f"Targets {len(hunt)}  done {len(done & set(hunt.index))}  to run {len(todo)}")
    if not todo:
        return

    out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out.exists()
    tasks = [(t, str(args.cache_dir)) for t in todo]
    counts: dict[str, int] = {}
    with open(out, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        if write_header:
            writer.writeheader()
        pool = mp.get_context("spawn").Pool(args.workers) if args.workers > 1 else None
        results = pool.imap_unordered(bls_one, tasks) if pool else map(bls_one, tasks)
        try:
            for row in tqdm(results, total=len(tasks), desc="BLS"):
                row["matches_hunt_csv"] = matches_hunt(row, hunt)
                writer.writerow(row)
                fh.flush()
                key = row["status"].split(":")[0]
                counts[key] = counts.get(key, 0) + 1
        finally:
            if pool is not None:
                pool.close()
                pool.join()

    print("Status:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    frame = pd.read_csv(out)
    ok = frame[frame.status == "ok"]
    if len(ok):
        n_match = int(ok.matches_hunt_csv.astype(str).eq("True").sum())
        print(f"Rows reproducing the hunt CSV after rounding: {n_match}/{len(ok)}")
    print(f"Written: {out}")


if __name__ == "__main__":
    main()
