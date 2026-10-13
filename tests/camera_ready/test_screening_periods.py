"""Check the screening period set and that it reproduces the committed systematic flags."""
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# validate_candidates.py resolves results/validation relative to the working directory at import.
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from validate_candidates import TESS_SYSTEMATIC_PERIODS, check_systematic_period  # noqa: E402

EXPECTED = {
    "results/validation/validation_summary.csv": (12, 30),
    "results/validation_novel/validation_summary.csv": (0, 1),
}


def main() -> int:
    failures = []
    if TESS_SYSTEMATIC_PERIODS.get("TESS orbit half") != 6.85:
        failures.append("'TESS orbit half': 6.85 missing")
    if 2.85 in TESS_SYSTEMATIC_PERIODS.values():
        failures.append("2.85 d entry still present")
    for path, (n_flagged, n_rows) in EXPECTED.items():
        frame = pd.read_csv(PROJECT_ROOT / path)
        flags = frame.period_days.map(lambda p: check_systematic_period(p)["flags_systematic"]).astype(bool)
        stored = frame.flags_systematic.astype(bool)
        print(f"  {path}: recomputed {int(flags.sum())}/{len(frame)}, stored {int(stored.sum())}/{len(frame)}")
        if not (flags.values == stored.values).all():
            failures.append(f"{path}: recomputed flags differ from flags_systematic")
        if (int(flags.sum()), len(frame)) != (n_flagged, n_rows):
            failures.append(f"{path}: expected {n_flagged} of {n_rows}")
    for line in failures:
        print("  " + line)
    print("PASS" if not failures else "FAIL")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
