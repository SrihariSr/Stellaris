"""Build the R0 target list: TICs hosting only FP/FA TOIs and absent from the R1-R3 hunts."""
from pathlib import Path
import argparse
import datetime as dt

import pandas as pd

from stellaris.targets import FP_HOSTS, REGIME_CSVS, TOI_TABLE

FALSE_DISPOSITIONS = {"FP", "FA"}
PLANET_DISPOSITIONS = {"CP", "KP", "PC", "APC"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toi-table", type=Path, default=TOI_TABLE)
    parser.add_argument("--out", type=Path, default=FP_HOSTS)
    parser.add_argument("--limit", type=int, default=None, help="Write at most N TICs (in ascending TIC order).")
    args = parser.parse_args()

    toi = pd.read_csv(args.toi_table, comment="#")
    toi["TIC ID"] = toi["TIC ID"].astype(int)
    disposition = toi["TFOPWG Disposition"]
    fp_hosts = set(toi.loc[disposition.isin(FALSE_DISPOSITIONS), "TIC ID"])
    planet_hosts = set(toi.loc[disposition.isin(PLANET_DISPOSITIONS), "TIC ID"])
    hunted = {regime: set(pd.read_csv(REGIME_CSVS[regime]).tic_id.astype(int)) for regime in ("r1", "r2", "r3")}

    no_planet = fp_hosts - planet_hosts
    selected = no_planet - set().union(*hunted.values())
    tics = sorted(selected)
    if args.limit is not None:
        tics = tics[: args.limit]

    counts = [
        f"TICs with any FP or FA TOI: {len(fp_hosts)}",
        f"  also hosting a CP, KP, PC or APC TOI (excluded): {len(fp_hosts & planet_hosts)}",
        *(f"  remaining TICs already in {r.upper()} ({REGIME_CSVS[r].name}, excluded): {len(no_planet & hunted[r])}"
          for r in ("r1", "r2", "r3")),
        f"Selected: {len(selected)}" + (f" (written: first {len(tics)} by TIC, --limit)" if args.limit else ""),
    ]
    header = [
        "# R0: TESS false-positive hosts (TFOPWG FP/FA only), not in the R1-R3 hunts",
        f"# Source: {args.toi_table.name}; built {dt.date.today().isoformat()} by scripts/build_false_positive_list.py",
        *(f"# {line}" for line in counts),
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(header + [str(t) for t in tics]) + "\n")
    print("\n".join(counts))
    print(f"Written: {args.out}")


if __name__ == "__main__":
    main()
