"""Resolve regime names (r0-r3) and TIC-list files into ordered TIC ids."""
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results"
CATALOGS = PROJECT_ROOT / "data" / "catalogs"

REGIME_CSVS = {
    "r1": RESULTS / "tier1_results.csv",
    "r2": RESULTS / "tier3_results.csv",
    "r3": RESULTS / "novel_results.csv",
    "r0": RESULTS / "r0_results.csv",
}
NOVEL_TARGETS = CATALOGS / "novel_targets.txt"
FP_HOSTS = CATALOGS / "tess_false_positive_hosts.txt"
TOI_TABLE = CATALOGS / "tess_toi.csv"
TIC_STELLAR = CATALOGS / "tic_stellar.csv"


def read_tic_file(path: Path) -> list[int]:
    """One TIC per line (txt, '#' comments allowed) or a CSV with a tic_id or 'TIC ID' column."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, comment="#")
        for column in ("tic_id", "TIC ID", "tic"):
            if column in frame.columns:
                return [int(t) for t in frame[column].dropna()]
        return [int(t) for t in frame.iloc[:, 0].dropna()]
    tics = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            tics.append(int(line.split(",")[0].split()[0]))
    return tics


def resolve_targets(specs: list[str]) -> list[int]:
    """
    r1/r2 take tic_id from the hunt CSVs, r3 is all of novel_targets.txt and r0
    is the false-positive host list when it exists. Anything else is a file path.
    Order is preserved and duplicates are dropped.
    """
    tics: list[int] = []
    for spec in specs:
        key = spec.lower()
        if key in ("r1", "r2"):
            tics += [int(t) for t in pd.read_csv(REGIME_CSVS[key]).tic_id]
        elif key == "r3":
            tics += read_tic_file(NOVEL_TARGETS)
        elif key == "r0":
            if FP_HOSTS.exists():
                tics += read_tic_file(FP_HOSTS)
            else:
                print(f"[warn] {FP_HOSTS} not found; r0 contributes no targets "
                      "(run scripts/build_false_positive_list.py first)", file=sys.stderr)
        else:
            tics += read_tic_file(Path(spec))
    seen: set[int] = set()
    ordered = []
    for tic in tics:
        if tic not in seen:
            seen.add(tic)
            ordered.append(tic)
    return ordered
