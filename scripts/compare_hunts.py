"""Compare the paper's TESS numbers, row values and R2 screening set between committed and corrected hunts."""
from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_paper_numbers import load_regime, tess_numbers  # noqa: E402

from stellaris.targets import PROJECT_ROOT, REGIME_CSVS, TOI_TABLE  # noqa: E402

CORRECTED_DIR = PROJECT_ROOT / "results" / "camera_ready" / "corrected_hunt"
ROW_COLUMNS = ["period_days", "epoch_btjd", "duration_hours", "depth_ppm", "bls_snr",
               "cnn_probability", "vetting_passed", "vetting_reasons", "rank_score"]


def screening_set(r2: pd.DataFrame, threshold: float, min_snr: float, top: int) -> list[int]:
    """The rows validate_candidates.py would screen, selected with its own call."""
    eligible = r2[
        (r2['cnn_probability'] >= threshold) &
        (r2['vetted']) &
        (r2['bls_snr'] >= min_snr)
    ].sort_values('cnn_probability', ascending=False).head(top)
    return [int(t) for t in eligible.tic_id]


def row_differences(committed: pd.DataFrame, corrected: pd.DataFrame) -> dict:
    a = committed.drop_duplicates("tic_id").set_index("tic_id")
    b = corrected.drop_duplicates("tic_id").set_index("tic_id")
    both = a.index.intersection(b.index)
    out = {"only_committed": len(a.index.difference(b.index)), "only_corrected": len(b.index.difference(a.index)),
           "in_both": len(both)}
    differs = pd.Series(False, index=both)
    for col in ROW_COLUMNS:
        x, y = a.loc[both, col], b.loc[both, col]
        if col == "vetting_reasons":
            changed = x.fillna("").astype(str) != y.fillna("").astype(str)
        elif col == "vetting_passed":
            changed = x.astype(str).str.lower() != y.astype(str).str.lower()
        else:
            changed = ~np.isclose(x.astype(float), y.astype(float), rtol=0, atol=0, equal_nan=True)
        out[f"rows_differing_{col}"] = int(changed.sum())
        differs |= changed
    out["rows_identical"] = int((~differs).sum())
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--committed", type=Path, nargs=3, default=[REGIME_CSVS[r] for r in ("r1", "r2", "r3")],
                        metavar=("R1", "R2", "R3"))
    parser.add_argument("--corrected", type=Path, nargs=3,
                        default=[CORRECTED_DIR / f"{r}_results.csv" for r in ("r1", "r2", "r3")],
                        metavar=("R1", "R2", "R3"))
    parser.add_argument("--out", type=Path, default=CORRECTED_DIR / "comparison.csv")
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--min-snr", type=float, default=10)
    parser.add_argument("--max-candidates", type=int, default=30)
    parser.add_argument("--limit", type=int, default=None, help="Print at most N rows of the comparison table.")
    parser.add_argument("--common-only", action="store_true",
                        help="Restrict both sides to the TICs present in both, isolating changes to shared targets.")
    args = parser.parse_args()

    toi = pd.read_csv(TOI_TABLE, comment="#")
    toi["TIC ID"] = toi["TIC ID"].astype(int)
    confirmed = toi[toi["TFOPWG Disposition"].isin(["CP", "KP"])]
    committed = [load_regime(p) for p in args.committed]
    corrected = [load_regime(p) for p in args.corrected]
    if args.common_only:
        shared = [set(a.tic_id) & set(b.tic_id) for a, b in zip(committed, corrected)]
        committed = [a[a.tic_id.isin(s)] for a, s in zip(committed, shared)]
        corrected = [b[b.tic_id.isin(s)] for b, s in zip(corrected, shared)]
        print("Restricted to shared TICs: " + ", ".join(f"R{i + 1} {len(s)}" for i, s in enumerate(shared)))
    before = tess_numbers(*committed, confirmed=confirmed)
    after = tess_numbers(*corrected, confirmed=confirmed)

    table = pd.DataFrame({"name": list(before), "committed": list(before.values()),
                          "corrected": [after[k] for k in before]})
    table["difference"] = table.corrected.astype(float) - table.committed.astype(float)
    same = np.isclose(table.committed.astype(float), table.corrected.astype(float), rtol=0, atol=0, equal_nan=True)
    table.loc[same, "difference"] = 0.0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)

    with pd.option_context("display.max_rows", None, "display.width", 200, "display.float_format", "{:.6g}".format):
        print(table.head(args.limit).to_string(index=False) if args.limit else table.to_string(index=False))
    print(f"\nNumbers that differ: {int((~same).sum())} of {len(table)}")

    print("\nRow-level differences (committed vs corrected):")
    for regime, a, b in zip(("R1", "R2", "R3"), committed, corrected):
        d = row_differences(a, b)
        cols = ", ".join(f"{c.replace('rows_differing_', '')} {v}" for c, v in d.items() if c.startswith("rows_differing_") and v)
        print(f"  {regime}: in both {d['in_both']}, identical {d['rows_identical']}, only committed {d['only_committed']}, "
              f"only corrected {d['only_corrected']}" + (f"; differing: {cols}" if cols else ""))

    old = screening_set(committed[1], args.threshold, args.min_snr, args.max_candidates)
    new = screening_set(corrected[1], args.threshold, args.min_snr, args.max_candidates)
    entering = [t for t in new if t not in old]
    leaving = [t for t in old if t not in new]
    print(f"\nR2 screening set (CNN >= {args.threshold}, vetted, BLS S/N >= {args.min_snr}, top {args.max_candidates}):")
    print(f"  committed {len(old)}, corrected {len(new)}, changed: {bool(entering or leaving)}")
    if entering or leaving:
        print(f"  entering: {entering}\n  leaving:  {leaving}")
        print("  To screen the corrected set without touching results/validation:\n"
              f"    PYTHONPATH=src python scripts/validate_candidates.py --results {args.corrected[1]} "
              f"--threshold {args.threshold} --min-snr {args.min_snr:g} --max-candidates {args.max_candidates} "
              "--out-dir results/camera_ready/validation_corrected")
    print(f"\nWritten: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
