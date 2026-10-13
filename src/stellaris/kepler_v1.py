"""Recover which kepler_koi_train.csv row produced each example of the v1 training set."""
from pathlib import Path
import re

import h5py
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
V1_PATH = PROJECT_ROOT / "data" / "processed" / "kepler_trainset.h5"
V1_FAILURES = PROJECT_ROOT / "data" / "processed" / "build_failures.log"
KOI_CSV = PROJECT_ROOT / "data" / "catalogs" / "kepler_koi_train.csv"

LOCAL_WINDOW = re.compile(r"^Local window \(([\d.]+)\)")


def parse_failure_log(path: Path) -> list[tuple[int, str]]:
    """Records start with '<kepid>\\t'; other lines continue the previous record."""
    records: list[list] = []
    for line in Path(path).read_text().splitlines():
        match = re.match(r"^(\d+)\t(.*)$", line)
        if match:
            records.append([int(match.group(1)), match.group(2)])
        elif records:
            records[-1][1] += "\n" + line
    return [(k, m) for k, m in records]


def reconstruct_v1_rows(koi: pd.DataFrame, failures: list[tuple[int, str]]) -> tuple[np.ndarray, dict[int, str]] | None:
    """
    Returns (CSV rows behind the v1 examples in v1 order, {CSV row: v1 failure message}).

    v1 logged failures in CSV order. 'Local window (x)' failures are pinned to
    the row whose 4*duration/period rounds to x; the rest to the next row of
    the same kepid. None if the log cannot be walked to its end.
    """
    rows, failed = [], {}
    f = 0
    for i, r in enumerate(koi.itertuples()):
        if f < len(failures) and failures[f][0] == r.kepid:
            match = LOCAL_WINDOW.match(failures[f][1])
            half_window = 4 * (r.koi_duration / 24.0) / r.koi_period
            if match is None or f"{half_window:.3f}" == match.group(1):
                failed[i] = failures[f][1]
                f += 1
                continue
        rows.append(i)
    if f != len(failures):
        return None
    return np.asarray(rows, dtype=np.int64), failed


def v1_csv_rows(
    koi: pd.DataFrame,
    v1_path: Path = V1_PATH,
    failures_path: Path = V1_FAILURES,
) -> tuple[np.ndarray, dict[int, str]]:
    """
    The validated v1 mapping: CSV row per v1 example (v1 order) and the v1
    failure message per excluded row. Raises ValueError unless the rebuilt
    mapping reproduces v1's kepid and label sequences exactly.
    """
    result = reconstruct_v1_rows(koi, parse_failure_log(failures_path))
    if result is None:
        raise ValueError(f"{failures_path} does not walk to its end against {len(koi)} CSV rows")
    rows, failed = result
    with h5py.File(v1_path, "r") as f:
        kepid, label = f["kepid"][:], f["label"][:]
    if not (len(rows) == len(kepid)
            and np.array_equal(koi.kepid.to_numpy()[rows], kepid)
            and np.array_equal(koi.label.to_numpy()[rows], label)):
        raise ValueError(f"rebuilt mapping does not reproduce the kepid and label sequences of {v1_path}")
    return rows, failed
