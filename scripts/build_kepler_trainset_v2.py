"""Build kepler_trainset_v2.h5: the v1 views plus normalisation scales, catalogue scalars and FP flags."""
from pathlib import Path
import argparse
import csv
import datetime as dt
import hashlib
import multiprocessing as mp
import os
import sys

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from stellaris.fetch import KEPLER_DATA_DIR, load_kepler_lightcurve
from stellaris.kepler_local import first_local_dir
from stellaris.kepler_v1 import KOI_CSV, V1_FAILURES, V1_PATH, v1_csv_rows
from stellaris.preprocess import preprocess
from stellaris.views import make_global_view, make_local_view

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "data" / "processed" / "kepler_trainset_v2.h5"

GLOBAL_VIEW_BINS = 2001
LOCAL_VIEW_BINS = 201
KEPIDS_PER_TASK = 16
SF_DATALESS = 0x40000000


def validate_row(period: float, epoch: float, duration_hours: float) -> None:
    """The same checks, in the same order, as build_kepler_trainset.process_one_koi."""
    if not np.isfinite(period) or period <= 0:
        raise ValueError(f"Bad period: {period}")
    if not np.isfinite(epoch):
        raise ValueError(f"Bad epoch: {epoch}")
    if not np.isfinite(duration_hours) or duration_hours <= 0:
        raise ValueError(f"Bad duration: {duration_hours}")


def cloud_only(kepid: int) -> bool:
    files = sorted((KEPLER_DATA_DIR / f"{kepid:09d}").rglob("*_llc.fits"))
    return any(getattr(os.stat(f), "st_flags", 0) & SF_DATALESS for f in files)


def process_task(task: dict) -> dict:
    """
    Process every row of a few kepids. The light curve is preprocessed once per
    kepid; preprocess is deterministic, so views match v1's per-row rebuild.
    """
    rows = task["rows"]
    n = len(rows)
    out = {
        "csv_row": np.array([r["csv_row"] for r in rows], dtype=np.int64),
        "ok": np.zeros(n, dtype=bool),
        "global_view": np.full((n, GLOBAL_VIEW_BINS), np.nan, dtype=np.float32),
        "local_view": np.full((n, LOCAL_VIEW_BINS), np.nan, dtype=np.float32),
        "global_scale": np.full(n, np.nan),
        "local_scale": np.full(n, np.nan),
        "message": np.array([""] * n, dtype=object),
        "source_dir": np.array([""] * n, dtype=object),
    }
    cleaned: dict[int, tuple] = {}
    for i, r in enumerate(rows):
        kepid = r["kepid"]
        try:
            validate_row(r["period"], r["epoch"], r["duration_hours"])
            if kepid not in cleaned:
                root = KEPLER_DATA_DIR
                try:
                    if task["kepler_dirs"] is not None:
                        # Search order: the first directory with a fully local copy; placeholders are never opened.
                        root = first_local_dir(kepid, [Path(d) for d in task["kepler_dirs"]])
                        if root is None:
                            raise RuntimeError("no fully local copy in --kepler-dirs")
                    elif task["skip_dataless"] and cloud_only(kepid):
                        raise RuntimeError("dataless: cloud-only FITS skipped (--skip-dataless)")
                    clean = preprocess(load_kepler_lightcurve(kepid, root=root))
                    cleaned[kepid] = (clean.time.value, clean.flux.value, None, str(root))
                except Exception as exc:
                    cleaned[kepid] = (None, None, exc, "" if root is None else str(root))
            time, flux, error, source_dir = cleaned[kepid]
            out["source_dir"][i] = source_dir
            if error is not None:
                raise error
            duration_days = r["duration_hours"] / 24.0
            gv, gs = make_global_view(time, flux, r["period"], r["epoch"],
                                      num_bins=GLOBAL_VIEW_BINS, return_scale=True)
            lv, ls = make_local_view(time, flux, r["period"], r["epoch"], duration_days,
                                     num_bins=LOCAL_VIEW_BINS, return_scale=True)
            out["global_view"][i] = gv
            out["local_view"][i] = lv
            out["global_scale"][i] = gs
            out["local_scale"][i] = ls
            out["ok"][i] = True
        except Exception as exc:
            out["message"][i] = str(exc)[:200]
    np.savez(task["part"], **{k: (v.astype(str) if k in ("message", "source_dir") else v) for k, v in out.items()})
    return {"part": task["part"], "n_ok": int(out["ok"].sum()), "n": n}


def make_tasks(koi: pd.DataFrame, rows: list[int], parts_dir: Path, skip_dataless: bool,
               kepler_dirs: list[str] | None = None) -> list[dict]:
    groups: dict[int, list[int]] = {}
    for i in rows:
        groups.setdefault(int(koi.at[i, "kepid"]), []).append(i)
    kepids = list(groups)
    tasks = []
    for start in range(0, len(kepids), KEPIDS_PER_TASK):
        chunk_rows = [i for k in kepids[start:start + KEPIDS_PER_TASK] for i in groups[k]]
        key = ",".join(map(str, chunk_rows)) + "|" + "|".join(kepler_dirs or [])
        digest = hashlib.sha1(key.encode()).hexdigest()[:10]
        tasks.append({
            "part": str(parts_dir / f"part_{start // KEPIDS_PER_TASK:05d}_{digest}.npz"),
            "skip_dataless": skip_dataless,
            "kepler_dirs": kepler_dirs,
            "rows": [{
                "csv_row": i,
                "kepid": int(koi.at[i, "kepid"]),
                "period": float(koi.at[i, "koi_period"]),
                "epoch": float(koi.at[i, "koi_time0bk"]),
                "duration_hours": float(koi.at[i, "koi_duration"]),
            } for i in chunk_rows],
        })
    return tasks


def load_parts(parts: list[Path]) -> dict[int, dict]:
    merged: dict[int, dict] = {}
    for part in parts:
        with np.load(part, allow_pickle=False) as data:
            for j, row in enumerate(data["csv_row"]):
                merged[int(row)] = {k: data[k][j] for k in data.files}
    return merged


def write_h5(out: Path, koi: pd.DataFrame, ordered: list[int], merged: dict[int, dict], attrs: dict) -> None:
    """Write the given CSV rows, in the given order, atomically."""
    sel = koi.loc[ordered]

    def col(name, dtype):
        return sel[name].to_numpy(dtype=dtype)

    tmp = out.with_name(f".{out.name}.{os.getpid()}.tmp")
    with h5py.File(tmp, "w") as f:
        f.create_dataset("global_view", data=np.stack([merged[i]["global_view"] for i in ordered]).astype(np.float32)
                         if ordered else np.zeros((0, GLOBAL_VIEW_BINS), np.float32),
                         compression="gzip", compression_opts=4)
        f.create_dataset("local_view", data=np.stack([merged[i]["local_view"] for i in ordered]).astype(np.float32)
                         if ordered else np.zeros((0, LOCAL_VIEW_BINS), np.float32),
                         compression="gzip", compression_opts=4)
        f.create_dataset("label", data=col("label", np.int8))
        f.create_dataset("kepid", data=col("kepid", np.int64))
        f.create_dataset("csv_row", data=np.asarray(ordered, dtype=np.int64))
        f.create_dataset("kepoi_name", data=sel["kepoi_name"].astype(str).to_numpy(dtype=object),
                         dtype=h5py.string_dtype())
        f.create_dataset("kepler_dir", data=np.array([str(merged[i]["source_dir"]) for i in ordered], dtype=object),
                         dtype=h5py.string_dtype())
        f.create_dataset("global_scale", data=np.array([merged[i]["global_scale"] for i in ordered], dtype=np.float64))
        f.create_dataset("local_scale", data=np.array([merged[i]["local_scale"] for i in ordered], dtype=np.float64))
        f.create_dataset("period_days", data=col("koi_period", np.float64))
        f.create_dataset("duration_hours", data=col("koi_duration", np.float64))
        f.create_dataset("koi_depth_ppm", data=col("koi_depth", np.float64))
        f.create_dataset("srad", data=col("koi_srad", np.float64))
        f.create_dataset("steff", data=col("koi_steff", np.float64))
        f.create_dataset("slogg", data=col("koi_slogg", np.float64))
        for flag in ("nt", "ss", "co", "ec"):
            f.create_dataset(f"fpflag_{flag}", data=col(f"koi_fpflag_{flag}", np.int8))
        labels = col("label", np.int8)
        f.attrs["n_examples"] = len(ordered)
        f.attrs["n_positives"] = int(labels.sum())
        f.attrs["n_negatives"] = int((labels == 0).sum())
        for key, value in attrs.items():
            f.attrs[key] = value
        f.attrs["created"] = dt.datetime.now().isoformat(timespec="seconds")
    tmp.replace(out)


def write_rows(path: Path, header: list[str], rows: list[list]) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--koi-csv", type=Path, default=KOI_CSV)
    parser.add_argument("--v1", type=Path, default=V1_PATH, help="v1 training set whose rows are matched.")
    parser.add_argument("--v1-failures", type=Path, default=V1_FAILURES)
    parser.add_argument("--match-v1", action=argparse.BooleanOptionalAction, default=True,
                        help="Build exactly v1's rows in v1's order (default). --no-match-v1 processes the "
                             "selected CSV rows instead and the result is not v1-compatible.")
    parser.add_argument("--allow-missing", action="store_true",
                        help="With --match-v1, write the subset even when some v1 rows fail.")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--limit", type=int, default=None,
                        help="First N v1 rows (with --match-v1) or first N CSV rows (without).")
    parser.add_argument("--rows", type=int, nargs="+", default=None,
                        help="Only these CSV row indices (0-based); overrides --limit. With --match-v1, "
                             "rows absent from v1 are dropped and listed as extra.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-dataless", action="store_true",
                        help="Fail kepids with cloud-only (OneDrive placeholder) FITS instead of downloading them.")
    parser.add_argument("--kepler-dirs", type=Path, nargs="+", default=None,
                        help="Search order of Kepler data directories. Each kepid is read from the first one "
                             "holding a fully local copy; cloud-only placeholders are never opened. "
                             "Default: data/raw/kepler only, as before.")
    parser.add_argument("--keep-parts", action="store_true", help="Keep the per-chunk resume files.")
    args = parser.parse_args()

    out = args.out.resolve()
    if out == args.v1.resolve() or out == V1_PATH.resolve():
        parser.error(f"refusing to write to the v1 training set {V1_PATH}")
    if out.exists() and not args.overwrite:
        parser.error(f"{out} exists; pass --overwrite to replace it")

    koi = pd.read_csv(args.koi_csv)
    requested = sorted(set(args.rows)) if args.rows is not None else None
    try:
        v1_rows, v1_failed = v1_csv_rows(koi, args.v1, args.v1_failures)
    except (ValueError, FileNotFoundError) as exc:
        if args.match_v1:
            parser.error(f"--match-v1 needs the v1 row mapping: {exc}")
        v1_rows, v1_failed = None, {}
        print(f"[warn] v1 row mapping unavailable ({exc}); extra rows cannot be identified")
    in_v1 = set() if v1_rows is None else {int(r) for r in v1_rows}
    v1_position = {} if v1_rows is None else {int(r): j for j, r in enumerate(v1_rows)}

    if args.match_v1:
        if requested is not None:
            wanted = set(requested)
            rows = [int(r) for r in v1_rows if int(r) in wanted]
            extras = [r for r in requested if r not in in_v1]
        elif args.limit is not None:
            rows = [int(r) for r in v1_rows[: args.limit]]
            extras = [r for r in range(rows[-1] + 1 if rows else 0) if r not in in_v1]
        else:
            rows = [int(r) for r in v1_rows]
            extras = [r for r in range(len(koi)) if r not in in_v1]
        print(f"Matching v1: {len(rows)} of {len(v1_rows)} v1 rows selected; "
              f"{len(extras)} CSV rows in scope are absent from v1 and will not be written")
    else:
        rows = requested if requested is not None else list(
            range(len(koi) if args.limit is None else min(args.limit, len(koi))))
        extras = []
        print("[warn] --no-match-v1: rows follow the CSV selection, not v1; the file is not v1-compatible")

    parts_dir = out.with_name(out.name + ".parts")
    parts_dir.mkdir(parents=True, exist_ok=True)
    kepler_dirs = None if args.kepler_dirs is None else [str(d.expanduser().resolve()) for d in args.kepler_dirs]
    if kepler_dirs is not None:
        print("Kepler directories, in search order: " + "  ".join(kepler_dirs))
    tasks = make_tasks(koi, rows, parts_dir, args.skip_dataless, kepler_dirs)
    todo = [t for t in tasks if not Path(t["part"]).exists()]
    print(f"Rows {len(rows)} in {len(tasks)} chunks; {len(tasks) - len(todo)} chunks already done")

    if todo:
        pool = mp.get_context("spawn").Pool(args.workers) if args.workers > 1 else None
        results = pool.imap_unordered(process_task, todo) if pool else map(process_task, todo)
        try:
            for _ in tqdm(results, total=len(todo), desc="Chunks"):
                pass
        finally:
            if pool is not None:
                pool.close()
                pool.join()

    parts = [Path(t["part"]) for t in tasks]
    merged = load_parts(parts)
    ok_rows = [i for i in rows if merged[i]["ok"]]
    failed = [i for i in rows if not merged[i]["ok"]]

    def koi_fields(i: int) -> list:
        return [i, int(koi.at[i, "kepid"]), koi.at[i, "kepoi_name"]]

    def v1_reason(i: int) -> str:
        text = v1_failed.get(i, "").strip()
        return text.splitlines()[0][:200] if text else ""

    if not args.match_v1:
        # Without matching, extras are rows that are written although v1 lacks them.
        extras = [i for i in ok_rows if v1_rows is not None and i not in in_v1]
    write_rows(out.with_name(out.name + ".failures.csv"), ["csv_row", "kepid", "kepoi_name", "message"],
               [koi_fields(i) + [merged[i]["message"]] for i in failed])
    extra_path = out.with_name(out.name + ".extra_rows.csv")
    write_rows(extra_path, ["csv_row", "kepid", "kepoi_name", "label", "v1_failure"],
               [koi_fields(i) + [int(koi.at[i, "label"]), v1_reason(i)] for i in extras])
    if args.match_v1:
        missing_path = out.with_name(out.name + ".missing_rows.csv")
        write_rows(missing_path, ["csv_row", "kepid", "kepoi_name", "v1_index", "message"],
                   [koi_fields(i) + [v1_position[i], merged[i]["message"]] for i in failed])
        if failed and not args.allow_missing:
            print(f"ERROR: {len(failed)} of {len(rows)} v1 rows failed in v2; {out.name} was not written"
                  + (" (the existing file was left unchanged)" if out.exists() else ""))
            print(f"Missing rows: {missing_path}\nRe-run once they are fixed (finished chunks are kept), "
                  "or pass --allow-missing to write the subset.")
            sys.exit(1)
        if failed:
            banner = "!" * 78
            print(f"{banner}\nWARNING: --allow-missing: {len(failed)} v1 rows failed and are NOT in {out.name}.\n"
                  f"This file is a subset of v1 and does not reproduce v1's splits. See {missing_path}\n{banner}")
    elif extras:
        print(f"[warn] {len(extras)} written rows are absent from v1 (listed in {extra_path.name})")

    matched_prefix = args.match_v1 and not failed and rows == [int(r) for r in v1_rows[: len(rows)]]
    write_h5(out, koi, ok_rows, merged, {
        "matches_v1": bool(args.match_v1 and not failed),
        "v1_prefix": bool(matched_prefix),
        "n_v1_rows": -1 if v1_rows is None else len(v1_rows),
        "n_missing_v1_rows": len(failed) if args.match_v1 else -1,
        "partial": bool(len(ok_rows) < (len(v1_rows) if args.match_v1 else len(koi))),
        "n_csv_rows_total": len(koi),
        "n_csv_rows_processed": len(rows),
        "skip_dataless": bool(args.skip_dataless),
        "kepler_dirs": "|".join(kepler_dirs) if kepler_dirs is not None else str(KEPLER_DATA_DIR),
        "source_csv": str(args.koi_csv),
    })

    if not args.keep_parts:
        for part in parts:
            part.unlink(missing_ok=True)
        try:
            parts_dir.rmdir()
        except OSError:
            print(f"Left {parts_dir} in place: it still holds parts from another row selection.")

    print(f"Written {len(ok_rows)}/{len(rows)} rows to {out}")
    sources = pd.Series([str(merged[i]["source_dir"]) for i in ok_rows]).value_counts()
    print("Rows by source directory: " + "; ".join(f"{d}: {n}" for d, n in sources.items()))
    print(f"Failures: {len(failed)}   extra rows listed: {len(extras)} ({extra_path.name})")


if __name__ == "__main__":
    main()
