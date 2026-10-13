"""Build a TESS training set in the v1 HDF5 layout from the local cache and TOI catalogue ephemerides."""
from pathlib import Path
import argparse
import csv
import datetime as dt
import os

import h5py
import numpy as np
from tqdm import tqdm

from stellaris.rescore import build_views, catalogue_ephemeris, load_toi_table
from stellaris.targets import FP_HOSTS, PROJECT_ROOT, TOI_TABLE, resolve_targets
from stellaris.tess_cache import DEFAULT_CACHE_DIR, cache_path, load_cache

V1_PATH = PROJECT_ROOT / "data" / "processed" / "kepler_trainset.h5"
DEFAULT_OUT = PROJECT_ROOT / "data" / "processed" / "tess_trainset.h5"
POSITIVE = {"CP", "KP"}
NEGATIVE = {"FP", "FA"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positives", nargs="+", default=["r1"], help="TIC sources for CP/KP TOIs (default r1).")
    parser.add_argument("--negatives", nargs="+", default=[str(FP_HOSTS)],
                        help="TIC sources for FP/FA TOIs (default the R0 host list).")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--toi-table", type=Path, default=TOI_TABLE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=None, help="At most N TOIs from each class.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out = args.out.resolve()
    if out == V1_PATH.resolve():
        parser.error(f"refusing to write to the v1 training set {V1_PATH}")
    if out.exists() and not args.overwrite:
        parser.error(f"{out} exists; pass --overwrite to replace it")

    toi = load_toi_table(args.toi_table)
    tasks = []
    for label, sources, dispositions in ((1, args.positives, POSITIVE), (0, args.negatives, NEGATIVE)):
        missing = [s for s in sources if s.lower() not in ("r0", "r1", "r2", "r3") and not Path(s).exists()]
        tics = resolve_targets([s for s in sources if s not in missing])
        if missing:
            print(f"[warn] skipping missing target lists: {missing}")
        rows = toi[toi["TIC ID"].isin(tics) & toi["TFOPWG Disposition"].isin(dispositions)]
        rows = rows.sort_values(["TIC ID", "TOI"])
        if args.limit is not None:
            rows = rows.head(args.limit)
        tasks += [(label, row) for _, row in rows.iterrows()]
    print(f"TOIs: {sum(t[0] for t in tasks)} positive, {sum(1 - t[0] for t in tasks)} negative")

    global_views, local_views, labels, tics, tois, failures = [], [], [], [], [], []
    loaded: dict[int, dict | None] = {}
    for label, row in tqdm(tasks, desc="Views"):
        tic = int(row["TIC ID"])
        try:
            eph = catalogue_ephemeris(row)
            if not (np.isfinite(eph["period_days"]) and eph["period_days"] > 0):
                raise ValueError(f"Bad period: {eph['period_days']}")
            if not (np.isfinite(eph["duration_days"]) and eph["duration_days"] > 0):
                raise ValueError(f"Bad duration: {eph['duration_days']}")
            if not np.isfinite(eph["epoch_btjd"]):
                raise ValueError(f"Bad epoch: {eph['epoch_btjd']}")
            if tic not in loaded:
                path = cache_path(args.cache_dir, tic)
                loaded = {tic: load_cache(path) if path.exists() else None}
            data = loaded[tic]
            if data is None:
                raise FileNotFoundError("missing_cache")
            gv, lv, _, _ = build_views(data["time"], data["flux"], eph["period_days"], eph["epoch_btjd"],
                                       eph["duration_days"])
        except Exception as exc:
            failures.append((tic, str(row["TOI"]), label, str(exc).splitlines()[0][:150]))
            continue
        global_views.append(gv.astype(np.float32))
        local_views.append(lv.astype(np.float32))
        labels.append(label)
        tics.append(tic)
        tois.append(str(row["TOI"]))

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.{os.getpid()}.tmp")
    labels_arr = np.asarray(labels, dtype=np.int8)
    with h5py.File(tmp, "w") as f:
        f.create_dataset("global_view", data=np.asarray(global_views, dtype=np.float32).reshape(-1, 2001),
                         compression="gzip", compression_opts=4)
        f.create_dataset("local_view", data=np.asarray(local_views, dtype=np.float32).reshape(-1, 201),
                         compression="gzip", compression_opts=4)
        f.create_dataset("label", data=labels_arr)
        # "kepid" keeps the v1 name so StellarisDataset and make_splits work unchanged; it holds TIC ids.
        f.create_dataset("kepid", data=np.asarray(tics, dtype=np.int64))
        f.create_dataset("toi", data=np.asarray(tois, dtype=object), dtype=h5py.string_dtype())
        f.attrs["n_examples"] = len(labels)
        f.attrs["n_positives"] = int(labels_arr.sum())
        f.attrs["n_negatives"] = int((labels_arr == 0).sum())
        f.attrs["id_column"] = "kepid holds TESS TIC ids"
        f.attrs["created"] = dt.datetime.now().isoformat(timespec="seconds")
    tmp.replace(out)

    failures_path = out.with_name(out.name + ".failures.csv")
    with open(failures_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["tic_id", "toi", "label", "message"])
        writer.writerows(failures)
    print(f"Examples {len(labels)} ({int(labels_arr.sum())} positive); failures {len(failures)}")
    print(f"Written: {out}\nFailures: {failures_path}")


if __name__ == "__main__":
    main()
