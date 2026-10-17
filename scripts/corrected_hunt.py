"""Corrected hunt tables: the hunt CSV schema rebuilt from full-precision BLS on chronological light curves."""
from pathlib import Path
import argparse

import pandas as pd

from stellaris.targets import PROJECT_ROOT, REGIME_CSVS

RELEASED = "checkpoints/stellaris_best.pt"
REGIMES = ("r1", "r2", "r3")
HUNT_COLUMNS = [
    "tic_id", "period_days", "epoch_btjd", "duration_hours", "depth_ppm", "bls_snr",
    "cnn_probability", "vetting_passed", "vetting_reasons", "rank_score",
]
RESCORE_DIR = PROJECT_ROOT / "results" / "camera_ready" / "rescore"
BLS_FULL_DIR = PROJECT_ROOT / "results" / "camera_ready" / "bls_full"
DEFAULT_OUT = PROJECT_ROOT / "results" / "camera_ready" / "corrected_hunt"


def corrected_table(rescore: pd.DataFrame, bls: pd.DataFrame, hunt_order: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (hunt-schema rows for status-ok targets, targets that are not ok).

    Ephemeris and S/N come from the bls_full CSV, rounded as hunting.py rounds;
    score and vetting come from the released checkpoint's rescore row.
    """
    released = rescore[rescore.checkpoint == RELEASED].drop_duplicates("tic_id", keep="last").set_index("tic_id")
    bls = bls.drop_duplicates("tic_id", keep="last").set_index("tic_id")
    not_ok = [(int(t), s if t not in bls.index or bls.at[t, "status"] == "ok" else f"{s} (bls_full: {bls.at[t, 'status']})")
              for t, s in released.status.items() if s != "ok"]
    not_ok += [(int(t), "not rescored: " + str(s)) for t, s in bls.status.items() if t not in released.index]
    ok = released[(released.status == "ok") & released.index.isin(bls.index)]
    not_ok += [(int(t), "ok in rescore but absent from bls_full") for t in released.index
               if released.at[t, "status"] == "ok" and t not in bls.index]

    rows = []
    for tic, r in ok.iterrows():
        b = bls.loc[tic]
        prob = float(r.prob)
        vetted = str(r.vetting_passed).strip().lower() == "true"
        reasons = r.vetting_reasons if isinstance(r.vetting_reasons, str) else ""
        rows.append({
            "tic_id": int(tic),
            "period_days": round(float(b.period_days), 6),
            "epoch_btjd": round(float(b.epoch_btjd), 4),
            "duration_hours": round(float(b.duration_hours), 3),
            "depth_ppm": round(abs(float(b.depth_ppm)), 1),
            "bls_snr": round(float(b.bls_snr), 2),
            "cnn_probability": round(prob, 4),
            "vetting_passed": vetted,
            "vetting_reasons": reasons,
            "rank_score": round(prob if vetted else 0.0, 4),
        })
    table = pd.DataFrame(rows, columns=HUNT_COLUMNS)
    position = {t: i for i, t in enumerate(hunt_order)}
    table = table.assign(_order=table.tic_id.map(lambda t: position.get(t, len(position) + t)))
    table = table.sort_values("_order").drop(columns="_order").reset_index(drop=True)

    mismatch = [int(t) for t in ok.index if float(ok.at[t, "period_days"]) != float(bls.at[t, "period_days"])]
    if mismatch:
        print(f"  [warn] {len(mismatch)} rescore rows carry a different period from bls_full "
              f"(was the rescore made from another bls_full?): {mismatch[:10]}")
    return table, pd.DataFrame(not_ok, columns=["tic_id", "status"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for regime in REGIMES:
        parser.add_argument(f"--{regime}", type=Path, default=RESCORE_DIR / f"{regime}_bls_full.csv",
                            help=f"rescore_tess.py output for {regime} made with --ephemeris bls_full.")
    parser.add_argument("--bls-dir", type=Path, default=BLS_FULL_DIR,
                        help="Directory with rerun_bls.py outputs {regime}_bls_full.csv (for bls_snr).")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=None, help="Use at most N targets per regime.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for regime in REGIMES:
        rescore = pd.read_csv(getattr(args, regime))
        if args.limit is not None:
            rescore = rescore[rescore.tic_id.isin(rescore.tic_id.drop_duplicates().head(args.limit))]
        if "ephemeris" in rescore and not (rescore.ephemeris == "bls_full").all():
            print(f"  [warn] {regime}: some rescore rows were not made with --ephemeris bls_full")
        bls = pd.read_csv(args.bls_dir / f"{regime}_bls_full.csv")
        if args.limit is not None:
            bls = bls[bls.tic_id.isin(rescore.tic_id)]
        hunt_order = [int(t) for t in pd.read_csv(REGIME_CSVS[regime]).tic_id]
        table, not_ok = corrected_table(rescore, bls, hunt_order)
        out = args.out_dir / f"{regime}_results.csv"
        table.to_csv(out, index=False)
        print(f"{regime}: {len(table)} ok rows written to {out}; {len(not_ok)} not ok")
        for status, group in not_ok.groupby("status"):
            print(f"  {status} ({len(group)}): {' '.join(map(str, group.tic_id))}")


if __name__ == "__main__":
    main()
