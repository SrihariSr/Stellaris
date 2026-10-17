"""Fetch Kepler long-cadence light curves for the v1 kepids from MAST into a directory outside OneDrive."""
from pathlib import Path
import argparse
import csv
import multiprocessing as mp
import time
import warnings

import h5py
import pandas as pd
from tqdm import tqdm

from stellaris.fetch import KEPLER_DATA_DIR
from stellaris.kepler_local import fully_local
from stellaris.kepler_v1 import V1_PATH

DEFAULT_OUT = Path.home() / "stellaris_data" / "kepler"
SYNC_MARKERS = ("OneDrive", "iCloud", "CloudStorage")
FAILURE_COLUMNS = ["kepid", "stage", "message"]


def verify_fits(path: Path) -> str:
    """Empty when the file opens and is as long as its headers require, else the problem."""
    from astropy.io import fits

    size = path.stat().st_size
    if size == 0 or size % 2880:
        return f"size {size} is not a positive multiple of 2880"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            with fits.open(path, memmap=False, lazy_load_hdus=False) as hdul:
                expected = max(h._data_offset + -(-h._data_size // 2880) * 2880 for h in hdul)
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"[:200]
    if expected > size or any("truncat" in str(w.message).lower() for w in caught):
        return f"truncated: {size} bytes, headers need {expected}"
    return ""


def fetch_one(task: tuple[int, str, int]) -> tuple[int, str, str, int]:
    """
    Download one kepid into a staging directory and rename it to <kepid:09d> only
    when every LLC product is present and opens cleanly, so a final directory is
    always complete. Returns (kepid, stage, message, n_files); stage is "ok" on success.
    """
    from astroquery.mast import Observations

    kepid, out_dir, retries = task
    final = Path(out_dir) / f"{kepid:09d}"
    staging = Path(out_dir) / f".{kepid:09d}.partial"
    stage, message = "query", ""
    for attempt in range(retries):
        if attempt:
            time.sleep(min(60, 5 * 2 ** attempt))
        try:
            stage = "query"
            obs = Observations.query_criteria(
                target_name=f"kplr{kepid:09d}",
                obs_collection="Kepler",
                dataproduct_type="timeseries",
            )
            if len(obs) == 0:
                return kepid, "query", "no observations", 0
            products = Observations.get_product_list(obs)
            lc_products = Observations.filter_products(
                products,
                productSubGroupDescription="LLC",
                extension="fits",
            )
            if len(lc_products) == 0:
                return kepid, "query", "no LLC products", 0

            stage = "download"
            staging.mkdir(parents=True, exist_ok=True)
            manifest = Observations.download_products(lc_products, download_dir=str(staging), verbose=False)
            errors = [str(m) for s, m in zip(manifest["Status"], manifest["Message"]) if str(s) == "ERROR"]

            stage = "verify"
            files = {f.name: f for f in staging.rglob("*_llc.fits")}
            missing = sorted({str(n) for n in lc_products["productFilename"]} - set(files))
            bad = {name: problem for name, f in files.items() if (problem := verify_fits(f))}
            if not errors and not missing and not bad:
                staging.rename(final)
                return kepid, "ok", "", len(files)
            for name in bad:
                files[name].unlink()
            message = "; ".join(filter(None, [
                f"{len(errors)} download errors ({errors[0][:80]})" if errors else "",
                f"{len(missing)} products missing" if missing else "",
                f"{len(bad)} failed to verify ({next(iter(bad.values()))})" if bad else "",
            ]))
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"[:200]
    return kepid, stage, message, 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--v1", type=Path, default=V1_PATH, help="Training set whose kepid dataset lists the stars.")
    parser.add_argument("--local-dir", type=Path, default=KEPLER_DATA_DIR,
                        help="Kepids whose LLC files here are all fully local are skipped.")
    parser.add_argument("--kepids", type=int, nargs="+", default=None, help="Fetch only these kepids.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="Fetch at most N kepids.")
    args = parser.parse_args()

    out_dir = args.out_dir.expanduser().resolve()
    if any(marker in str(out_dir) for marker in SYNC_MARKERS):
        print(f"[warn] {out_dir} is inside a cloud-sync folder; files may be evicted to placeholders again")
    with h5py.File(args.v1, "r") as f:
        kepids = [int(k) for k in pd.unique(f["kepid"][:])]
    if args.kepids is not None:
        wanted = set(args.kepids)
        kepids = [k for k in kepids if k in wanted]

    local = [k for k in kepids if fully_local(k, args.local_dir)]
    done = [k for k in kepids if (out_dir / f"{k:09d}").is_dir()]
    skip = set(local) | set(done)
    todo = [k for k in kepids if k not in skip]
    if args.limit is not None:
        todo = todo[: args.limit]
    print(f"v1 kepids: {len(kepids)}   fully local in {args.local_dir}: {len(local)}   "
          f"already complete in {out_dir}: {len(done)}   to fetch: {len(todo)}")
    if not todo:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    failures_path = out_dir / "failures.csv"
    write_header = not failures_path.exists()
    counts: dict[str, int] = {}
    n_files = 0
    tasks = [(k, str(out_dir), args.retries) for k in todo]
    with open(failures_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FAILURE_COLUMNS)
        if write_header:
            writer.writeheader()
        pool = mp.get_context("spawn").Pool(args.workers) if args.workers > 1 else None
        results = pool.imap_unordered(fetch_one, tasks) if pool else map(fetch_one, tasks)
        try:
            for kepid, stage, message, n in tqdm(results, total=len(tasks), desc="MAST"):
                counts[stage] = counts.get(stage, 0) + 1
                n_files += n
                if stage != "ok":
                    writer.writerow({"kepid": kepid, "stage": stage, "message": message})
                    fh.flush()
        finally:
            if pool is not None:
                pool.close()
                pool.join()
    print("Outcome:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())), f"  files: {n_files}")
    print(f"Output: {out_dir}   failures: {failures_path}")


if __name__ == "__main__":
    main()
