"""Fetch TIC stellar radius, Teff, log g and Tmag from MAST in chunks (network; resumable)."""
from pathlib import Path
import argparse
import csv
import time

import numpy as np
from tqdm import tqdm

from stellaris.targets import TIC_STELLAR, resolve_targets

MAX_CHUNK = 500
COLUMNS = ["tic_id", "srad", "teff", "logg", "tmag"]
TIC_FIELDS = {"srad": "rad", "teff": "Teff", "logg": "logg", "tmag": "Tmag"}


def already_fetched(path: Path) -> set[int]:
    if not path.exists():
        return set()
    with open(path, newline="") as fh:
        return {int(row["tic_id"]) for row in csv.DictReader(fh)}


def query_chunk(chunk: list[int], retries: int = 3) -> dict[int, dict]:
    from astroquery.mast import Catalogs

    for attempt in range(retries):
        try:
            table = Catalogs.query_criteria(catalog="TIC", ID=[str(t) for t in chunk])
            break
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(10 * (attempt + 1))
    found = {}
    for row in table:
        values = {}
        for out_name, tic_name in TIC_FIELDS.items():
            value = row[tic_name]
            values[out_name] = float(value) if value is not None and not np.ma.is_masked(value) else np.nan
        found[int(row["ID"])] = values
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", nargs="+", default=["r3"],
                        help="Any of r0 r1 r2 r3, or txt/csv paths of TIC ids (default r3).")
    parser.add_argument("--out", type=Path, default=TIC_STELLAR)
    parser.add_argument("--chunk-size", type=int, default=MAX_CHUNK)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    chunk_size = max(1, min(args.chunk_size, MAX_CHUNK))

    tics = resolve_targets(args.targets)
    if args.limit is not None:
        tics = tics[: args.limit]
    done = already_fetched(args.out)
    todo = [t for t in tics if t not in done]
    print(f"Targets {len(tics)}  already fetched {len(tics) - len(todo)}  to query {len(todo)}")
    if not todo:
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.out.exists()
    with open(args.out, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        if write_header:
            writer.writeheader()
        for start in tqdm(range(0, len(todo), chunk_size), desc="TIC chunks"):
            chunk = todo[start:start + chunk_size]
            found = query_chunk(chunk)
            for tic in chunk:
                # TICs MAST does not return are written with NaNs so a resume does not re-query them.
                writer.writerow({"tic_id": tic, **found.get(tic, {k: np.nan for k in TIC_FIELDS})})
            fh.flush()
    print(f"Written: {args.out}")


if __name__ == "__main__":
    main()
