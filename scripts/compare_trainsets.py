"""Compare kepler_trainset_v2.h5 against v1: row counts, kepid and label equality, exact view equality."""
from pathlib import Path
import argparse
import json
import sys

import h5py
import numpy as np
import pandas as pd

from stellaris.kepler_v1 import KOI_CSV, V1_FAILURES, V1_PATH, v1_csv_rows

PROJECT_ROOT = Path(__file__).resolve().parents[1]
V2_PATH = PROJECT_ROOT / "data" / "processed" / "kepler_trainset_v2.h5"
FIELDS = ("kepid", "label", "global_view", "local_view")


def rows_equal(v1: dict, v2: dict, i1: np.ndarray, i2: np.ndarray) -> dict:
    """Per-field equality over paired rows, plus the largest view differences."""
    out = {}
    for key in FIELDS:
        a, b = v1[key][i1], v2[key][i2]
        out[key] = np.all((a == b).reshape(len(i1), -1), axis=1) if len(i1) else np.zeros(0, dtype=bool)
    for key in ("global_view", "local_view"):
        diff = np.abs(v1[key][i1] - v2[key][i2])
        out[f"max_abs_{key}_diff"] = float(np.nanmax(diff)) if diff.size else float("nan")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1", type=Path, default=V1_PATH)
    parser.add_argument("--v2", type=Path, default=V2_PATH)
    parser.add_argument("--v1-failures", type=Path, default=V1_FAILURES)
    parser.add_argument("--koi-csv", type=Path, default=KOI_CSV)
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON summary path.")
    parser.add_argument("--limit", type=int, default=None, help="Compare at most N v2 rows (never IDENTICAL).")
    args = parser.parse_args()

    with h5py.File(args.v1, "r") as f:
        v1 = {k: f[k][:] for k in FIELDS}
    with h5py.File(args.v2, "r") as f:
        v2 = {k: f[k][:] for k in FIELDS + ("csv_row",) if k in f}
    if args.limit is not None:
        v2 = {k: v[: args.limit] for k, v in v2.items()}
    n1, n2 = len(v1["kepid"]), len(v2["kepid"])
    summary: dict = {"v1_rows": n1, "v2_rows": n2, "row_counts_equal": n1 == n2}
    print(f"v1 rows: {n1}   v2 rows: {n2}")

    # IDENTICAL is decided positionally: same row count, and row i of v2 equals row i of v1.
    n = min(n1, n2)
    positional = rows_equal(v1, v2, np.arange(n), np.arange(n))
    positional_ok = all(positional[k].all() for k in FIELDS)
    identical = n1 == n2 and positional_ok
    summary.update({
        "positional_rows_compared": n,
        **{f"positional_{k}_equal_rows": int(positional[k].sum()) for k in FIELDS},
        "positional_max_abs_global_view_diff": positional["max_abs_global_view_diff"],
        "positional_max_abs_local_view_diff": positional["max_abs_local_view_diff"],
    })
    print(f"Positional comparison over {n} rows: " + ", ".join(
        f"{k} {int(positional[k].sum())}/{n}" for k in FIELDS))

    verdict = "IDENTICAL" if identical else "DIFFERENT"
    if not identical and "csv_row" in v2:
        try:
            v1_rows, _ = v1_csv_rows(pd.read_csv(args.koi_csv), args.v1, args.v1_failures)
        except (ValueError, FileNotFoundError) as exc:
            v1_rows = None
            print(f"v1 row mapping unavailable ({exc}); no subset diagnosis")
        if v1_rows is not None:
            position = {int(r): j for j, r in enumerate(v1_rows)}
            mapped = np.array([position.get(int(r), -1) for r in v2["csv_row"]], dtype=np.int64)
            in_v1 = mapped >= 0
            aligned = rows_equal(v1, v2, mapped[in_v1], np.flatnonzero(in_v1))
            aligned_ok = all(aligned[k].all() for k in FIELDS)
            in_order = bool(np.all(np.diff(mapped[in_v1]) > 0))
            prefix = bool(in_v1.all() and np.array_equal(mapped, np.arange(n2)))
            summary.update({
                "v2_rows_absent_from_v1": int((~in_v1).sum()),
                "csv_rows_absent_from_v1": [int(r) for r in v2["csv_row"][~in_v1]][:50],
                "v2_rows_in_v1_order": in_order,
                **{f"aligned_{k}_equal_rows": int(aligned[k].sum()) for k in FIELDS},
            })
            print(f"Aligned on csv_row: {int(in_v1.sum())} v2 rows found in v1, {int((~in_v1).sum())} absent; "
                  f"in v1 order: {in_order}; " + ", ".join(
                      f"{k} {int(aligned[k].sum())}/{int(in_v1.sum())}" for k in FIELDS))
            if in_v1.all() and in_order and aligned_ok and n2 < n1:
                verdict = "PARTIAL MATCH: " + ("prefix" if prefix else "subset") + f" of {n2}/{n1} v1 rows, all identical"

    summary["verdict"] = verdict
    summary["identical"] = identical
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2, default=bool))
    print(verdict)
    return 0 if identical else 1


if __name__ == "__main__":
    sys.exit(main())
