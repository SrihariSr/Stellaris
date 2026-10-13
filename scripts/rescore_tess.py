"""Rescore a TESS regime from the local cache with one or more checkpoints (long-format CSV)."""
from pathlib import Path
import argparse
import csv

import numpy as np
import pandas as pd
from tqdm import tqdm

from stellaris.rescore import (
    CATALOGUE_DISPOSITIONS,
    build_views,
    catalogue_ephemeris,
    load_models,
    load_stellar_table,
    load_toi_table,
    score,
    select_toi,
    stellar_from_toi,
)
from stellaris.targets import PROJECT_ROOT, REGIME_CSVS, TIC_STELLAR
from stellaris.tess_cache import DEFAULT_CACHE_DIR, cache_path, load_cache
from stellaris.vetting import vet_candidate

DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "stellaris_best.pt"
BLS_FULL_DIR = PROJECT_ROOT / "results" / "camera_ready" / "bls_full"
OUT_COLUMNS = [
    "tic_id", "regime", "ephemeris", "toi", "selection",
    "period_days", "epoch_btjd", "duration_hours", "depth_ppm",
    "local_scale", "global_scale",
    "checkpoint", "variant", "seed", "prob",
    "vetting_passed", "vetting_reasons", "status",
]


def read_done(path: Path) -> set[tuple[int, str]]:
    if not path.exists():
        return set()
    with open(path, newline="") as fh:
        return {(int(r["tic_id"]), r["checkpoint"]) for r in csv.DictReader(fh)}


def bls_ephemerides(frame: pd.DataFrame, selection: str = "bls") -> dict[int, dict | None]:
    out: dict[int, dict | None] = {}
    for r in frame.itertuples():
        values = [float(r.period_days), float(r.epoch_btjd), float(r.duration_hours)]
        if not all(np.isfinite(values)) or values[0] <= 0 or values[2] <= 0:
            out[int(r.tic_id)] = None
            continue
        out[int(r.tic_id)] = {
            "toi": "",
            "selection": selection,
            "period_days": values[0],
            "epoch_btjd": values[1],
            "duration_days": values[2] / 24.0,
            "depth_ppm": float(r.depth_ppm),
        }
    return out


def catalogue_ephemerides(
    frame: pd.DataFrame, regime: str, toi: pd.DataFrame
) -> tuple[dict[int, dict | None], dict[int, str]]:
    """Ephemeris per TIC (None when unusable) and, where known, why it is missing."""
    dispositions = CATALOGUE_DISPOSITIONS[regime]
    pool = toi[toi["TFOPWG Disposition"].isin(dispositions)]
    by_tic = {tic: rows for tic, rows in pool.groupby("TIC ID")}
    out: dict[int, dict | None] = {}
    reasons: dict[int, str] = {}
    for r in frame.itertuples():
        rows = by_tic.get(int(r.tic_id))
        if rows is None:
            out[int(r.tic_id)] = None
            continue
        chosen, selection = select_toi(rows, float(r.period_days))
        out[int(r.tic_id)] = None if chosen is None else {**catalogue_ephemeris(chosen), "selection": selection}
        if chosen is None and selection == "single_transit":
            reasons[int(r.tic_id)] = selection
    return out, reasons


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime", choices=["r0", "r1", "r2", "r3"], required=True)
    parser.add_argument("--ephemeris", choices=["bls", "bls_full", "catalogue"], required=True,
                        help="bls: the hunt CSV's rounded BLS ephemeris; bls_full: full-precision BLS "
                             "from scripts/rerun_bls.py; catalogue: the TOI table.")
    parser.add_argument("--regime-csv", type=Path, default=None,
                        help="Hunt CSV for the regime (default: tier1, tier3, novel or r0_results; "
                             "for bls_full, results/camera_ready/bls_full/{regime}_bls_full.csv).")
    parser.add_argument("--checkpoints", nargs="+", default=[str(DEFAULT_CHECKPOINT)],
                        help="Checkpoint paths or globs.")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--stellar-table", type=Path, default=None,
                        help="CSV with tic_id, srad, teff, logg (default: TOI table; tic_stellar.csv for r3).")
    parser.add_argument("--toi-table", type=Path, default=PROJECT_ROOT / "data" / "catalogs" / "tess_toi.csv")
    parser.add_argument("--out", type=Path, default=None,
                        help="Default results/camera_ready/rescore/{regime}_{ephemeris}.csv")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.ephemeris == "catalogue" and args.regime == "r3":
        parser.error("r3 has no catalogue ephemeris; use --ephemeris bls")
    if args.regime_csv is not None:
        regime_csv = args.regime_csv
    elif args.ephemeris == "bls_full":
        regime_csv = BLS_FULL_DIR / f"{args.regime}_bls_full.csv"
    else:
        regime_csv = REGIME_CSVS[args.regime]
    out = args.out or PROJECT_ROOT / "results" / "camera_ready" / "rescore" / f"{args.regime}_{args.ephemeris}.csv"

    frame = pd.read_csv(regime_csv)
    if args.limit is not None:
        frame = frame.head(args.limit)

    toi = load_toi_table(args.toi_table)
    missing_reasons: dict[int, str] = {}
    if args.ephemeris in ("bls", "bls_full"):
        ephemerides = bls_ephemerides(frame, args.ephemeris)
    else:
        ephemerides, missing_reasons = catalogue_ephemerides(frame, args.regime, toi)

    stellar_path = args.stellar_table
    if stellar_path is None and args.regime == "r3" and TIC_STELLAR.exists():
        stellar_path = TIC_STELLAR
    stellar = load_stellar_table(stellar_path) if stellar_path else stellar_from_toi(toi)
    print(f"Stellar values from {stellar_path or args.toi_table}")

    models = load_models(args.checkpoints, args.device)
    print(f"Regime {args.regime} ({regime_csv}), ephemeris {args.ephemeris}, "
          f"{len(frame)} targets, {len(models)} checkpoint(s)")

    done = read_done(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out.exists()
    counts: dict[str, int] = {}

    with open(out, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUT_COLUMNS)
        if write_header:
            writer.writeheader()
        for tic_id in tqdm([int(t) for t in frame.tic_id], desc=f"Rescoring {args.regime}"):
            pending = [m for m in models if (tic_id, m[0]) not in done]
            if not pending:
                continue
            eph = ephemerides.get(tic_id)
            base = {"tic_id": tic_id, "regime": args.regime, "ephemeris": args.ephemeris,
                    "toi": "", "selection": "", "period_days": np.nan, "epoch_btjd": np.nan,
                    "duration_hours": np.nan, "depth_ppm": np.nan, "local_scale": np.nan,
                    "global_scale": np.nan, "prob": np.nan, "vetting_passed": "", "vetting_reasons": ""}
            status = "ok"
            views = None
            if eph is None:
                reason = missing_reasons.get(tic_id)
                status = "missing_ephemeris" + (f":{reason}" if reason else "")
            else:
                base.update({
                    "toi": eph["toi"], "selection": eph["selection"],
                    "period_days": eph["period_days"], "epoch_btjd": eph["epoch_btjd"],
                    "duration_hours": eph["duration_days"] * 24.0, "depth_ppm": eph["depth_ppm"],
                })
                path = cache_path(args.cache_dir, tic_id)
                if not path.exists():
                    status = "missing_cache"
                else:
                    data = load_cache(path)
                    time, flux = data["time"], data["flux"]
                    try:
                        views = build_views(time, flux, eph["period_days"], eph["epoch_btjd"], eph["duration_days"])
                    except ValueError as exc:
                        status = "view_error:" + str(exc).splitlines()[0][:120]
                    vet = vet_candidate(time, flux, eph["period_days"], eph["epoch_btjd"],
                                        eph["duration_days"], eph["depth_ppm"] / 1e6)
                    base["vetting_passed"] = vet.passed
                    base["vetting_reasons"] = "; ".join(vet.reasons)
            if views is not None:
                base["global_scale"], base["local_scale"] = views[2], views[3]
                values = {
                    "local_scale": views[3], "global_scale": views[2],
                    "period_days": eph["period_days"], "duration_days": eph["duration_days"],
                }
                if tic_id in stellar.index:
                    star = stellar.loc[tic_id]
                    values.update({k: float(star[k]) for k in ("srad", "teff", "logg") if k in star.index})
            for label, model, spec in pending:
                row = dict(base, checkpoint=label, variant=spec.get("variant", ""),
                           seed="" if spec.get("seed") is None else spec["seed"], status=status)
                if views is not None:
                    row["prob"] = score(model, spec, views[0], views[1], values, args.device)
                writer.writerow(row)
            fh.flush()
            key = "view_error" if status.startswith("view_error") else status
            counts[key] = counts.get(key, 0) + 1

    print("Status:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "nothing to do")
    print(f"Written: {out}")


if __name__ == "__main__":
    main()
