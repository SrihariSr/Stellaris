"""Probe depth sensitivity of frozen checkpoints: flux-scale invariance and a fixed-noise depth sweep."""
from pathlib import Path
import argparse
import csv

import numpy as np
import pandas as pd
from tqdm import tqdm

from stellaris.injection import a_over_rstar, robust_sigma, t14_days, transit_model
from stellaris.rescore import (
    build_views,
    load_models,
    load_stellar_table,
    load_toi_table,
    score,
    stellar_from_toi,
)
from stellaris.targets import PROJECT_ROOT, REGIME_CSVS
from stellaris.tess_cache import DEFAULT_CACHE_DIR, cache_path, load_cache

DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "stellaris_best.pt"
FLUX_FACTORS = (0.01, 100.0)
SWEEP_DEPTHS_PPM = (300, 1000, 3000, 10000, 20000, 40000)
IMPACT = 0.3

INVARIANCE_COLUMNS = [
    "tic_id", "regime", "factor", "checkpoint", "variant", "seed",
    "period_days", "epoch_btjd", "duration_hours",
    "max_abs_dglobal", "max_abs_dlocal", "global_scale_ratio", "local_scale_ratio",
    "prob_ref", "prob_scaled", "dprob", "status",
]
SWEEP_COLUMNS = [
    "tic_id", "regime", "checkpoint", "variant", "seed",
    "period_days", "epoch_btjd", "impact", "target_depth_ppm", "rp", "a_rs", "t14_hours",
    "realised_depth_ppm", "sigma_ppm", "n_in_transit", "expected_snr",
    "local_scale", "global_scale", "prob", "status",
]


def pick_targets(regime: str, n: int, cache_dir: Path, seed: int) -> pd.DataFrame:
    frame = pd.read_csv(REGIME_CSVS[regime]).drop_duplicates("tic_id")
    order = np.random.default_rng(seed).permutation(len(frame))
    frame = frame.iloc[order]
    cached = frame[[cache_path(cache_dir, int(t)).exists() for t in frame.tic_id]]
    return cached.head(n)


def values_for(tic_id: int, stellar: pd.DataFrame, views, period: float, duration: float) -> dict:
    values = {"global_scale": views[2], "local_scale": views[3], "period_days": period, "duration_days": duration}
    if tic_id in stellar.index:
        star = stellar.loc[tic_id]
        values.update({k: float(star[k]) for k in ("srad", "teff", "logg") if k in star.index})
    return values


def appender(path: Path, columns: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    fh = open(path, "a", newline="")
    writer = csv.DictWriter(fh, fieldnames=columns)
    if new:
        writer.writeheader()
    return fh, writer


def done_tics(path: Path) -> set[int]:
    return set(pd.read_csv(path).tic_id.astype(int)) if path.exists() else set()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", default=[str(DEFAULT_CHECKPOINT)])
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--regimes", nargs="+", default=["r1", "r3"], choices=["r0", "r1", "r2", "r3"])
    parser.add_argument("--n-targets", type=int, default=100, help="Targets per regime.")
    parser.add_argument("--stellar-table", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "results" / "camera_ready" / "depth_probe")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N targets in total.")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    models = load_models(args.checkpoints, args.device)
    stellar = load_stellar_table(args.stellar_table) if args.stellar_table else stellar_from_toi(load_toi_table())

    inv_path = args.out_dir / "invariance.csv"
    sweep_path = args.out_dir / "depth_sweep.csv"
    done = done_tics(inv_path) & done_tics(sweep_path)
    targets = pd.concat(
        [pick_targets(r, args.n_targets, args.cache_dir, args.seed).assign(regime=r) for r in args.regimes]
    )
    targets = targets[~targets.tic_id.astype(int).isin(done)]
    if args.limit is not None:
        targets = targets.head(args.limit)
    print(f"{len(targets)} targets to probe ({', '.join(args.regimes)}), {len(models)} checkpoint(s)")

    inv_fh, inv_writer = appender(inv_path, INVARIANCE_COLUMNS)
    sweep_fh, sweep_writer = appender(sweep_path, SWEEP_COLUMNS)
    try:
        for row in tqdm(list(targets.itertuples()), desc="Probing"):
            tic_id = int(row.tic_id)
            data = load_cache(cache_path(args.cache_dir, tic_id))
            time, flux = data["time"], data["flux"]
            period, epoch, duration = float(row.period_days), float(row.epoch_btjd), float(row.duration_hours) / 24
            base = {"tic_id": tic_id, "regime": row.regime}

            try:
                ref = build_views(time, flux, period, epoch, duration)
            except ValueError as exc:
                ref, ref_error = None, "view_error:" + str(exc).splitlines()[0][:120]
            for factor in FLUX_FACTORS:
                scaled = None
                if ref is not None:
                    try:
                        scaled = build_views(time, factor * flux, period, epoch, duration)
                    except ValueError as exc:
                        scaled_error = "view_error:" + str(exc).splitlines()[0][:120]
                for label, model, spec in models:
                    out = dict(base, factor=factor, checkpoint=label, variant=spec.get("variant", ""),
                               seed="" if spec.get("seed") is None else spec["seed"],
                               period_days=period, epoch_btjd=epoch, duration_hours=duration * 24)
                    if ref is None or scaled is None:
                        out["status"] = ref_error if ref is None else scaled_error
                    else:
                        p_ref = score(model, spec, ref[0], ref[1], values_for(tic_id, stellar, ref, period, duration))
                        p_scaled = score(model, spec, scaled[0], scaled[1],
                                         values_for(tic_id, stellar, scaled, period, duration))
                        out.update({
                            "max_abs_dglobal": float(np.max(np.abs(scaled[0] - ref[0]))),
                            "max_abs_dlocal": float(np.max(np.abs(scaled[1] - ref[1]))),
                            "global_scale_ratio": scaled[2] / ref[2] if ref[2] > 0 else np.nan,
                            "local_scale_ratio": scaled[3] / ref[3] if ref[3] > 0 else np.nan,
                            "prob_ref": p_ref, "prob_scaled": p_scaled, "dprob": p_scaled - p_ref,
                            "status": "ok",
                        })
                    inv_writer.writerow(out)

            rng = np.random.default_rng([args.seed, tic_id])
            inj_period = float(rng.uniform(1.0, 12.0))
            inj_epoch = float(time.min() + rng.uniform(0.0, inj_period))
            a_rs = a_over_rstar(inj_period)
            sigma = robust_sigma(flux)
            for depth_ppm in SWEEP_DEPTHS_PPM:
                rp = float(np.sqrt(depth_ppm / 1e6))
                model_flux = transit_model(time, inj_period, inj_epoch, rp, a_rs, IMPACT)
                t14 = t14_days(inj_period, a_rs, rp, IMPACT)
                realised = float(1.0 - model_flux.min())
                n_in = int(np.sum(model_flux < 1.0))
                injected = flux + (model_flux - 1.0)
                try:
                    views = build_views(time, injected, inj_period, inj_epoch, t14)
                    status = "ok"
                except ValueError as exc:
                    views, status = None, "view_error:" + str(exc).splitlines()[0][:120]
                for label, model, spec in models:
                    out = dict(base, checkpoint=label, variant=spec.get("variant", ""),
                               seed="" if spec.get("seed") is None else spec["seed"],
                               period_days=inj_period, epoch_btjd=inj_epoch, impact=IMPACT,
                               target_depth_ppm=depth_ppm, rp=rp, a_rs=a_rs, t14_hours=t14 * 24,
                               realised_depth_ppm=realised * 1e6, sigma_ppm=sigma * 1e6, n_in_transit=n_in,
                               expected_snr=realised / sigma * np.sqrt(n_in) if sigma > 0 else np.nan,
                               status=status)
                    if views is not None:
                        out.update({"local_scale": views[3], "global_scale": views[2],
                                    "prob": score(model, spec, views[0], views[1],
                                                  values_for(tic_id, stellar, views, inj_period, t14))})
                    sweep_writer.writerow(out)
            inv_fh.flush()
            sweep_fh.flush()
    finally:
        inv_fh.close()
        sweep_fh.close()
    print(f"Written: {inv_path}\n         {sweep_path}")


if __name__ == "__main__":
    main()
