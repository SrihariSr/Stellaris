"""Injection-recovery: inject transits into raw local light curves and run the unchanged hunt pipeline."""
from pathlib import Path
import argparse
import csv
import multiprocessing as mp
import os

import lightkurve as lk
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from stellaris.bls_search import run_bls
from stellaris.injection import a_over_rstar, robust_sigma, t14_days, transit_model
from stellaris.rescore import load_models, load_stellar_table, score
from stellaris.targets import PROJECT_ROOT, resolve_targets
from stellaris.tess_cache import RAW_TESS_DIR, dataless_files, load_local_tess, local_fits_paths
from stellaris.tess_preprocess import preprocess_tess
from stellaris.views import make_global_view, make_local_view
from stellaris.vetting import vet_candidate

DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "stellaris_best.pt"
DEFAULT_OUT = PROJECT_ROOT / "results" / "camera_ready" / "injection" / "r3_injections.csv"
MIN_CADENCES = 5000
MIN_PERIOD, MAX_PERIOD = 0.5, 20.0
DEPTH_RANGE = (5e-4, 3e-2)
MAX_IMPACT = 0.9
MATCH_TOL = 0.02
ALIASES = (0.5, 2.0, 1.0 / 3.0, 3.0)
MAX_GRID_DURATION_HOURS = 8.0

COLUMNS = [
    "tic_id", "injection", "seed", "stage", "message",
    "inj_period_days", "inj_epoch_btjd", "inj_depth_ppm", "inj_rp", "inj_b", "inj_a_rs",
    "inj_t14_hours", "stellar_source", "realised_depth_ppm", "sigma_ppm", "n_in_transit", "expected_snr",
    "n_cadences", "bls_period_days", "bls_epoch_btjd", "bls_duration_hours", "bls_depth_ppm", "bls_snr",
    "period_match", "alias_factor", "epoch_match", "in_grid",
    "checkpoint", "variant", "ckpt_seed", "prob", "vetting_passed", "vetting_reasons",
    "recovered_05", "endorsed_09",
]

_MODELS: list = []


def init_worker(checkpoints: list[str]) -> None:
    torch.set_num_threads(1)
    _MODELS[:] = load_models(checkpoints, "cpu")


def draw_injection(seed: int, tic_id: int, k: int, t_first: float, star: dict | None) -> dict:
    """Draw order is fixed (P, t0, depth, b) so a (seed, tic, k) triple always gives the same signal."""
    rng = np.random.default_rng([seed, tic_id, k])
    period = float(np.exp(rng.uniform(np.log(MIN_PERIOD), np.log(MAX_PERIOD))))
    t0 = float(t_first + rng.uniform(0.0, period)) if np.isfinite(t_first) else np.nan
    depth = float(np.exp(rng.uniform(np.log(DEPTH_RANGE[0]), np.log(DEPTH_RANGE[1]))))
    b = float(rng.uniform(0.0, MAX_IMPACT))
    if star is not None and np.isfinite(star.get("srad", np.nan)) and np.isfinite(star.get("logg", np.nan)):
        a_rs, source = a_over_rstar(period, star["srad"], star["logg"]), "table"
    else:
        a_rs, source = a_over_rstar(period), "solar"
    rp = float(np.sqrt(depth))
    return {
        "inj_period_days": period, "inj_epoch_btjd": t0, "inj_depth_ppm": depth * 1e6, "inj_rp": rp,
        "inj_b": b, "inj_a_rs": a_rs, "inj_t14_hours": t14_days(period, a_rs, rp, b) * 24,
        "stellar_source": source,
    }


def inject(lcc: lk.LightCurveCollection, inj: dict) -> lk.LightCurveCollection:
    out = []
    for lc in lcc:
        lc = lc.copy()
        model = transit_model(lc.time.value, inj["inj_period_days"], inj["inj_epoch_btjd"],
                              inj["inj_rp"], inj["inj_a_rs"], inj["inj_b"])
        lc.flux = lc.flux * model
        if "pdcsap_flux" in lc.colnames:
            lc["pdcsap_flux"] = lc["pdcsap_flux"] * model
        out.append(lc)
    return lk.LightCurveCollection(out)


def match_flags(inj: dict, bls) -> dict:
    period = inj["inj_period_days"]
    ratio = bls.period / period
    period_match = abs(ratio - 1.0) <= MATCH_TOL
    alias = next((f for f in ALIASES if abs(ratio / f - 1.0) <= MATCH_TOL), np.nan)
    epoch_match = False
    if period_match:
        offset = ((bls.epoch - inj["inj_epoch_btjd"]) / period) % 1.0
        offset = offset - 1.0 if offset >= 0.5 else offset
        epoch_match = abs(offset) < 0.5 * (inj["inj_t14_hours"] / 24) / period
    return {"period_match": bool(period_match), "alias_factor": alias if not period_match else np.nan,
            "epoch_match": bool(epoch_match)}


def in_grid(inj: dict) -> bool:
    t14 = inj["inj_t14_hours"] / 24
    return bool(inj["inj_period_days"] <= MAX_PERIOD and inj["inj_t14_hours"] <= MAX_GRID_DURATION_HOURS
                and 4 * t14 / inj["inj_period_days"] < 0.5)


def expand(base: dict, scored: list[dict] | None = None) -> list[dict]:
    """One row per checkpoint; unscored stages carry an empty probability."""
    if scored is not None:
        return [dict(base, **s) for s in scored]
    return [dict(base, checkpoint=label, variant=spec.get("variant", ""),
                 ckpt_seed="" if spec.get("seed") is None else spec["seed"], prob=np.nan,
                 recovered_05=False, endorsed_09=False)
            for label, _, spec in _MODELS]


def process_target(task: dict) -> list[dict]:
    tic_id, seed, star = task["tic_id"], task["seed"], task["star"]
    rows: list[dict] = []
    files = local_fits_paths(tic_id, Path(task["raw_dir"]))
    lcc, problem = None, None
    if not files:
        problem = "no local *lc.fits"
    elif task["skip_dataless"] and dataless_files(files):
        problem = "dataless: cloud-only FITS skipped (--skip-dataless)"
    else:
        try:
            lcc = load_local_tess(tic_id, Path(task["raw_dir"]))
        except Exception as exc:
            problem = "read failed: " + (str(exc).strip().splitlines() or [type(exc).__name__])[0][:150]

    t_first = np.nan
    clean_time = clean_flux = None
    if lcc is not None:
        t_first = float(np.nanmin(np.concatenate([np.asarray(lc.time.value, dtype=float) for lc in lcc])))
        try:
            clean = preprocess_tess(lcc)
            clean_time, clean_flux = clean.time.value, clean.flux.value
        except Exception:
            pass
    sigma = robust_sigma(clean_flux) if clean_flux is not None and len(clean_flux) else np.nan

    for k in task["injections"]:
        inj = draw_injection(seed, tic_id, k, t_first, star)
        base = {c: "" for c in COLUMNS}
        base.update({"tic_id": tic_id, "injection": k, "seed": seed, **inj, "in_grid": in_grid(inj),
                     "sigma_ppm": sigma * 1e6, "period_match": False, "epoch_match": False})
        if lcc is None:
            rows += expand(dict(base, stage="no_fits", message=problem))
            continue
        ref_time = clean_time if clean_time is not None else np.concatenate([lc.time.value for lc in lcc])
        model = transit_model(ref_time, inj["inj_period_days"], inj["inj_epoch_btjd"],
                              inj["inj_rp"], inj["inj_a_rs"], inj["inj_b"])
        realised = float(1.0 - np.nanmin(model))
        n_in = int(np.sum(model < 1.0))
        base.update({"realised_depth_ppm": realised * 1e6, "n_in_transit": n_in,
                     "expected_snr": realised / sigma * np.sqrt(n_in) if sigma > 0 else np.nan})

        try:
            clean = preprocess_tess(inject(lcc, inj))
        except Exception as exc:
            rows += expand(dict(base, stage="preprocess", message=str(exc)[:150]))
            continue
        time_arr, flux_arr = clean.time.value, clean.flux.value
        base["n_cadences"] = len(time_arr)
        if len(time_arr) < MIN_CADENCES:
            rows += expand(dict(base, stage="cadence_floor", message=f"Too few cadences ({len(time_arr)})"))
            continue
        try:
            bls = run_bls(time_arr, flux_arr, min_period=MIN_PERIOD, max_period=MAX_PERIOD)
        except Exception as exc:
            rows += expand(dict(base, stage="bls", message=str(exc)[:150]))
            continue
        base.update({"bls_period_days": bls.period, "bls_epoch_btjd": bls.epoch,
                     "bls_duration_hours": bls.duration * 24, "bls_depth_ppm": abs(bls.depth) * 1e6,
                     "bls_snr": bls.snr, **match_flags(inj, bls)})
        try:
            # return_scale=True yields views bit-identical to the default call (tests/camera_ready).
            gv, global_scale = make_global_view(time_arr, flux_arr, bls.period, bls.epoch, return_scale=True)
            lv, local_scale = make_local_view(time_arr, flux_arr, bls.period, bls.epoch, bls.duration,
                                              return_scale=True)
        except ValueError as exc:
            rows += expand(dict(base, stage="view", message=str(exc).splitlines()[0][:150]))
            continue

        vet = vet_candidate(time_arr, flux_arr, period=bls.period, epoch=bls.epoch,
                            duration=bls.duration, depth=bls.depth)
        values = {"local_scale": local_scale, "global_scale": global_scale,
                  "period_days": bls.period, "duration_days": bls.duration, **(star or {})}
        scored = []
        for label, net, spec in _MODELS:
            prob = score(net, spec, gv, lv, values)
            scored.append({
                "checkpoint": label, "variant": spec.get("variant", ""),
                "ckpt_seed": "" if spec.get("seed") is None else spec["seed"], "prob": prob,
                "vetting_passed": vet.passed, "vetting_reasons": "; ".join(vet.reasons),
                "recovered_05": bool(base["period_match"] and prob >= 0.5 and vet.passed),
                "endorsed_09": bool(base["period_match"] and prob >= 0.9 and vet.passed),
            })
        rows += expand(dict(base, stage="scored"), scored)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", nargs="+", default=["r3"],
                        help="Default r3: all 2,000 TICs in novel_targets.txt. Also r0-r2 or txt/csv paths.")
    parser.add_argument("--injections-per-target", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoints", nargs="+", default=[str(DEFAULT_CHECKPOINT)])
    parser.add_argument("--stellar-table", type=Path, default=None, help="CSV with tic_id, srad, logg (teff optional).")
    parser.add_argument("--raw-dir", type=Path, default=RAW_TESS_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-dataless", action="store_true",
                        help="Treat targets with cloud-only (OneDrive placeholder) FITS as no_fits.")
    args = parser.parse_args()

    tics = resolve_targets(args.targets)
    if args.limit is not None:
        tics = tics[: args.limit]
    stellar = load_stellar_table(args.stellar_table) if args.stellar_table else None

    done: set[tuple[int, int]] = set()
    if args.out.exists():
        frame = pd.read_csv(args.out, usecols=["tic_id", "injection"])
        done = set(zip(frame.tic_id.astype(int), frame.injection.astype(int)))
    tasks = []
    for tic in tics:
        todo = [k for k in range(args.injections_per_target) if (tic, k) not in done]
        if not todo:
            continue
        star = None
        if stellar is not None and tic in stellar.index:
            star = {k: float(stellar.loc[tic, k]) for k in ("srad", "teff", "logg") if k in stellar.columns}
        tasks.append({"tic_id": tic, "injections": todo, "seed": args.seed, "star": star,
                      "raw_dir": str(args.raw_dir), "skip_dataless": args.skip_dataless})
    print(f"Targets {len(tics)}, injections/target {args.injections_per_target}; "
          f"{len(tasks)} targets with work to do")
    if not tasks:
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.out.exists()
    stages: dict[str, int] = {}
    with open(args.out, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        if write_header:
            writer.writeheader()
        if args.workers <= 1:
            init_worker(args.checkpoints)
            pool, results = None, map(process_target, tasks)
        else:
            pool = mp.get_context("spawn").Pool(args.workers, initializer=init_worker, initargs=(args.checkpoints,))
            results = pool.imap_unordered(process_target, tasks, chunksize=1)
        try:
            for rows in tqdm(results, total=len(tasks), desc="Injecting"):
                writer.writerows(rows)
                fh.flush()
                for r in {(r["injection"], r["stage"]) for r in rows}:
                    stages[r[1]] = stages.get(r[1], 0) + 1
        finally:
            if pool is not None:
                pool.close()
                pool.join()
    print("Injections by stage reached:", ", ".join(f"{k}={v}" for k, v in sorted(stages.items())))
    print(f"Written: {args.out}")


if __name__ == "__main__":
    main()
