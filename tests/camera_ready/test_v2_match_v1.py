"""Check that a --match-v1 build holds exactly v1's rows and trains identically to v1 (small, local rows only)."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stellaris.fetch import KEPLER_DATA_DIR  # noqa: E402
from stellaris.kepler_v1 import KOI_CSV, V1_PATH, v1_csv_rows  # noqa: E402

WORK = PROJECT_ROOT / ".cr_scratch" / "smoke" / "test_v2_match_v1"
N_LOCAL = 60
SF_DATALESS = 0x40000000
METRIC_KEYS = ("n_params", "best_epoch", "val_pr_auc", "best_val", "test", "test_logit_std", "history")

failures: list[str] = []


def expect(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'ok' if ok else 'FAIL'}] {label}{(': ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


def is_local(kepid: int) -> bool | None:
    """True if every FITS is on disk, False if any is a cloud-only placeholder, None if there are none."""
    files = sorted((KEPLER_DATA_DIR / f"{kepid:09d}").rglob("*_llc.fits"))
    if not files:
        return None
    return not any(getattr(os.stat(f), "st_flags", 0) & SF_DATALESS for f in files)


def run(*args: str) -> tuple[int, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("NUMBA_CACHE_DIR", str(PROJECT_ROOT / ".cr_scratch" / "numba"))
    env.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cr_scratch" / "mpl"))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run([sys.executable, *args], cwd=PROJECT_ROOT, env=env, capture_output=True, text=True)
    return proc.returncode, proc.stdout + proc.stderr


def train(group: str, variant: str, h5: Path, tag: str) -> dict:
    out_dir, ckpt_dir = WORK / f"train_{tag}", WORK / f"ckpt_{tag}"
    code, log = run("scripts/train_variants.py", "--group", group, "--variants", variant, "--seeds", "0",
                    "--epochs", "2", "--max-batches", "2", "--device", "cpu", "--h5", str(h5),
                    "--out-dir", str(out_dir), "--ckpt-dir", str(ckpt_dir))
    expect(f"train {group}/{variant} on {h5.name} exits 0", code == 0, log.strip().splitlines()[-1] if log else "")
    run_json = json.loads((out_dir / "runs" / group / f"{variant}__seed0.json").read_text())
    state = torch.load(ckpt_dir / f"{variant}__seed0.pt", weights_only=False)["model_state_dict"]
    return {"json": run_json, "state": state}


def same_metrics(a: dict, b: dict) -> bool:
    # json.dumps writes NaN as NaN, so single-class smoke metrics still compare equal.
    return all(json.dumps(a["json"][k], sort_keys=True) == json.dumps(b["json"][k], sort_keys=True) for k in METRIC_KEYS)


def same_weights(a: dict, b: dict) -> bool:
    return list(a["state"]) == list(b["state"]) and all(torch.equal(a["state"][k], b["state"][k]) for k in a["state"])


def main() -> int:
    if not V1_PATH.exists():
        print(f"SKIP: {V1_PATH} not present")
        return 0
    koi = pd.read_csv(KOI_CSV)
    v1_rows, _ = v1_csv_rows(koi)
    v1_index = {int(r): j for j, r in enumerate(v1_rows)}
    local_rows, cloud_row = [], None
    status: dict[int, bool | None] = {}
    for r in v1_rows:
        kepid = int(koi.at[int(r), "kepid"])
        if kepid not in status:
            status[kepid] = is_local(kepid)
        if status[kepid] and len(local_rows) < N_LOCAL:
            local_rows.append(int(r))
        elif status[kepid] is False and cloud_row is None:
            cloud_row = int(r)
        if len(local_rows) == N_LOCAL and cloud_row is not None:
            break
    extra_row = next(i for i in range(len(koi)) if i not in v1_index)
    if len(local_rows) < N_LOCAL or cloud_row is None:
        print(f"SKIP: need {N_LOCAL} fully local v1 rows and one cloud-only row (found {len(local_rows)})")
        return 0
    print(f"  using {N_LOCAL} local v1 rows (CSV {local_rows[0]}..{local_rows[-1]}), cloud-only v1 row {cloud_row}, "
          f"CSV row absent from v1 {extra_row}")

    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    matched = WORK / "v2_matched.h5"
    code, log = run("scripts/build_kepler_trainset_v2.py", "--rows", *map(str, local_rows + [extra_row]),
                    "--skip-dataless", "--workers", "4", "--out", str(matched))
    expect("matched build exits 0", code == 0, log.strip().splitlines()[-1] if log else "")
    with h5py.File(matched, "r") as f:
        v2 = {k: f[k][:] for k in ("csv_row", "kepid", "label", "global_view", "local_view")}
        attrs = dict(f.attrs)
    positions = np.array([v1_index[int(r)] for r in v2["csv_row"]])
    expect("exactly the requested v1 rows, in v1 order", list(v2["csv_row"]) == local_rows)
    expect("row absent from v1 not written", extra_row not in set(v2["csv_row"].tolist()))
    extra = pd.read_csv(WORK / "v2_matched.h5.extra_rows.csv")
    expect("row absent from v1 listed in extra_rows.csv", list(extra.csv_row) == [extra_row])
    expect("no missing rows", pd.read_csv(WORK / "v2_matched.h5.missing_rows.csv").empty)
    expect("attrs: matches_v1, partial", bool(attrs["matches_v1"]) and bool(attrs["partial"]))
    with h5py.File(V1_PATH, "r") as f:
        v1 = {k: f[k][:][positions] for k in ("kepid", "label", "global_view", "local_view")}
    expect("kepid, label and both views equal v1 at the mapped rows",
           all(np.array_equal(v1[k], v2[k]) for k in v1))

    code, log = run("scripts/compare_trainsets.py", "--v2", str(matched))
    verdict = log.strip().splitlines()[-1]
    expect("compare: partial file is never IDENTICAL", code != 0 and verdict != "IDENTICAL", verdict)
    expect("compare: reports an all-identical subset", verdict.startswith("PARTIAL MATCH") and "all identical" in verdict)

    missing = WORK / "v2_missing.h5"
    code, log = run("scripts/build_kepler_trainset_v2.py", "--rows", *map(str, local_rows[:5] + [cloud_row]),
                    "--skip-dataless", "--workers", "2", "--out", str(missing))
    rows_missing = pd.read_csv(WORK / "v2_missing.h5.missing_rows.csv")
    expect("a failing v1 row stops the build with a non-zero exit", code != 0 and not missing.exists())
    expect("missing_rows.csv lists the failing v1 row", list(rows_missing.csv_row) == [cloud_row])
    code, log = run("scripts/build_kepler_trainset_v2.py", "--rows", *map(str, local_rows[:5] + [cloud_row]),
                    "--skip-dataless", "--workers", "2", "--out", str(missing), "--allow-missing")
    with h5py.File(missing, "r") as f:
        n_subset, subset_matches = len(f["kepid"]), bool(f.attrs["matches_v1"])
    expect("--allow-missing writes the subset with a warning",
           code == 0 and n_subset == 5 and not subset_matches and "WARNING: --allow-missing" in log)

    v1_subset = WORK / "v1_subset.h5"
    with h5py.File(v1_subset, "w") as f:
        for key, value in v1.items():
            f.create_dataset(key, data=value)
    with h5py.File(matched, "r") as f:
        labels = f["label"][:]
    from stellaris.dataset import make_splits
    splits = make_splits(hdf5_path=matched)
    print("  split sizes (positives): " + ", ".join(
        f"{k} {len(v)} ({int(labels[v].sum())})" for k, v in splits.items()))

    scalars_a = train("scalars", "A", matched, "scalars_v2")
    both_v2 = train("ablation", "both", matched, "ablation_v2")
    both_v1 = train("ablation", "both", v1_subset, "ablation_v1")
    print(f"  scalars/A test {scalars_a['json']['test']}")
    print(f"  ablation/both test {both_v2['json']['test']}")
    expect("scalars A == ablation both on the matched v2 file (metrics)", same_metrics(scalars_a, both_v2))
    expect("scalars A == ablation both on the matched v2 file (weights)", same_weights(scalars_a, both_v2))
    expect("scalars A on v2 == ablation both on the same rows cut from v1 (metrics)", same_metrics(scalars_a, both_v1))
    expect("scalars A on v2 == ablation both on the same rows cut from v1 (weights)", same_weights(scalars_a, both_v1))

    print("PASS" if not failures else f"FAIL ({len(failures)} checks)")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
