"""Build the local-only TESS cache: preprocessed time, flux and sector labels per TIC."""
from pathlib import Path
import argparse
import csv
import multiprocessing as mp
import os

from tqdm import tqdm

from stellaris.targets import resolve_targets
from stellaris.tess_cache import (
    DEFAULT_CACHE_DIR,
    RAW_TESS_DIR,
    cache_path,
    dataless_files,
    load_local_tess,
    local_fits_paths,
    preprocess_with_sectors,
    save_cache,
)

FAILURE_COLUMNS = ["tic_id", "stage", "message"]


def first_line(exc: BaseException) -> str:
    text = str(exc).strip()
    return (text.splitlines()[0] if text else type(exc).__name__)[:200]


def build_one(task: tuple[int, str, str, bool]) -> tuple[int, str, str]:
    """Returns (tic_id, stage, message); stage is "ok" on success."""
    tic_id, cache_dir, raw_dir, skip_dataless = task
    files = local_fits_paths(tic_id, Path(raw_dir))
    if not files:
        return tic_id, "no_fits", "no local *lc.fits"
    if skip_dataless:
        cloud_only = dataless_files(files)
        if cloud_only:
            return tic_id, "dataless", f"{len(cloud_only)} of {len(files)} FITS are cloud-only placeholders"
    try:
        lcc = load_local_tess(tic_id, Path(raw_dir))
    except Exception as exc:
        return tic_id, "read", first_line(exc)
    try:
        time, flux, sector = preprocess_with_sectors(lcc)
    except Exception as exc:
        return tic_id, "preprocess", first_line(exc)
    try:
        n_raw = int(sum(len(lc) for lc in lcc))
        save_cache(cache_path(Path(cache_dir), tic_id), time, flux, sector, n_raw, [f.name for f in files])
    except Exception as exc:
        return tic_id, "write", first_line(exc)
    return tic_id, "ok", ""


def read_failed(path: Path) -> set[int]:
    if not path.exists():
        return set()
    with open(path, newline="") as fh:
        return {int(row["tic_id"]) for row in csv.DictReader(fh)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", nargs="+", required=True,
                        help="Any of r0 r1 r2 r3, or paths to txt/csv files of TIC ids.")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--raw-dir", type=Path, default=RAW_TESS_DIR)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--limit", type=int, default=None, help="Use only the first N resolved targets.")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild targets that are already cached.")
    parser.add_argument("--retry-failures", action="store_true",
                        help="Retry targets already listed in failures.csv (skipped by default).")
    parser.add_argument("--skip-dataless", action="store_true",
                        help="Skip targets with any cloud-only (OneDrive placeholder) FITS instead of "
                             "letting the read download it.")
    args = parser.parse_args()

    tics = resolve_targets(args.targets)
    if args.limit is not None:
        tics = tics[: args.limit]
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    failures_path = args.cache_dir / "failures.csv"
    failed_before = set() if (args.overwrite or args.retry_failures) else read_failed(failures_path)

    todo = [
        t for t in tics
        if (args.overwrite or not cache_path(args.cache_dir, t).exists()) and t not in failed_before
    ]
    print(f"Targets: {len(tics)}  cached: {sum(cache_path(args.cache_dir, t).exists() for t in tics)}  "
          f"previous failures skipped: {len(failed_before & set(tics))}  to build: {len(todo)}")
    if not todo:
        return

    tasks = [(t, str(args.cache_dir), str(args.raw_dir), args.skip_dataless) for t in todo]
    write_header = not failures_path.exists()
    counts: dict[str, int] = {}
    with open(failures_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FAILURE_COLUMNS)
        if write_header:
            writer.writeheader()
        if args.workers <= 1:
            results = map(build_one, tasks)
            pool = None
        else:
            pool = mp.get_context("spawn").Pool(args.workers)
            results = pool.imap_unordered(build_one, tasks, chunksize=1)
        try:
            for tic_id, stage, message in tqdm(results, total=len(tasks), desc="Caching"):
                counts[stage] = counts.get(stage, 0) + 1
                if stage != "ok":
                    writer.writerow({"tic_id": tic_id, "stage": stage, "message": message})
                    fh.flush()
        finally:
            if pool is not None:
                pool.close()
                pool.join()

    print("Outcome:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"Cache: {args.cache_dir}   failures: {failures_path}")


if __name__ == "__main__":
    main()
