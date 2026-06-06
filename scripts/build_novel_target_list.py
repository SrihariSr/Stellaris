"""
Build a TIC target list for the novel exoplanet hunt.

Strategy:
  1. Download MIT/STScI's per-sector SPOC 2-minute target lists for early sectors.
  2. Combine, deduplicate, keep TICs observed in multiple sectors.
  3. Filter to Tmag 13-15 (faint, under-mined regime).
  4. Exclude TOI catalogue and previous hunt TICs.
  5. Sample N targets.

Output: data/catalogs/novel_targets.txt
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import requests

CATALOGS = Path("data/catalogs")
RESULTS = Path("results")
SPOC_CACHE = CATALOGS / "spoc_target_lists"
SPOC_CACHE.mkdir(parents=True, exist_ok=True)
DEFAULT_OUTPUT = CATALOGS / "novel_targets.txt"

# MIT TESS sector target list URLs follow a predictable pattern.
# Files have TIC IDs of all 2-minute-cadence targets per sector.
SPOC_URL_TEMPLATE = (
    "https://tess.mit.edu/wp-content/uploads/"
    "all_targets_S{sector:03d}_v1.csv"
)

def download_sector_list(sector: int) -> Path | None:
    """
    Download a single sector's target list if not already cached.
    """
    cache_path = SPOC_CACHE / f"all_targets_S{sector:03d}.csv"
    if cache_path.exists():
        return cache_path

    url = SPOC_URL_TEMPLATE.format(sector=sector)
    try:
        r = requests.get(url, timeout=60)
        if r.status_code != 200:
            print(f"  Sector {sector}: HTTP {r.status_code}, skipping")
            return None
        with open(cache_path, 'wb') as f:
            f.write(r.content)
        return cache_path
    except Exception as e:
        print(f"  Sector {sector}: download failed ({e}), skipping")
        return None


def load_spoc_targets(sectors: list[int]) -> pd.DataFrame:
    """
    Download and combine SPOC target lists across multiple sectors.

    Returns a DataFrame with columns: TICID, Tmag, sectors_observed (list).
    """
    all_targets = {}  # tic_id -> dict(tmag, sectors)

    for sector in sectors:
        print(f"  Loading sector {sector}...")
        path = download_sector_list(sector)
        if path is None:
            continue

        # MIT files have a comment header line starting with #; skip
        df = pd.read_csv(path, comment='#')

        # Column names are commonly TICID, RA, Dec, Tmag etc.
        tic_col = next((c for c in df.columns if 'TIC' in c.upper()), None)
        tmag_col = next((c for c in df.columns if 'TMAG' in c.upper()), None)

        if tic_col is None or tmag_col is None:
            print(f"Couldn't find TIC/Tmag columns. Columns: {list(df.columns)[:6]}")
            continue

        for _, row in df.iterrows():
            try:
                tic = int(row[tic_col])
                tmag = float(row[tmag_col])
            except (ValueError, TypeError):
                continue
            if tic in all_targets:
                all_targets[tic]['sectors'].append(sector)
            else:
                all_targets[tic] = {'tmag': tmag, 'sectors': [sector]}

        print(f"Sector {sector}: {len(df)} entries, cumulative unique: {len(all_targets)}")

    if not all_targets:
        raise RuntimeError("No SPOC target lists could be loaded.")

    out = pd.DataFrame([
        {'tic_id': tic, 'tmag': v['tmag'],
         'n_sectors': len(v['sectors']), 'sectors': v['sectors']}
        for tic, v in all_targets.items()
    ])
    return out

def load_excluded_tics() -> set[int]:
    """TICs already in the TOI catalogue or previous hunts."""
    excluded = set()
    toi_path = CATALOGS / "tess_toi.csv"
    if toi_path.exists():
        toi = pd.read_csv(toi_path)
        excluded.update(int(t) for t in toi['TIC ID'].dropna())
        print(f"  Loaded {len(excluded)} from TOI catalogue")

    for tier_csv in [RESULTS / "tier1_results.csv", RESULTS / "tier3_results.csv"]:
        if tier_csv.exists():
            df = pd.read_csv(tier_csv)
            n_before = len(excluded)
            excluded.update(int(t) for t in df['tic_id'])
            print(f"  Loaded {len(excluded) - n_before} new from {tier_csv.name}")
    return excluded

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--tmag-min", type=float, default=13.0)
    parser.add_argument("--tmag-max", type=float, default=15.0)
    parser.add_argument("--sectors", type=int, nargs='+',
                        default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                        help="TESS sectors to include (default: 1-13)")
    parser.add_argument("--min-sectors-observed", type=int, default=2,
                        help="Require TIC was observed in at least N sectors")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("-" * 60)
    print("BUILDING NOVEL TARGET LIST (SPOC method)")
    print("-" * 60)

    print(f"\nDownloading SPOC target lists for sectors {args.sectors[0]}-{args.sectors[-1]}...")
    spoc = load_spoc_targets(args.sectors)
    print(f"\nTotal unique TICs across sectors: {len(spoc)}")
    print(f"Mean sectors per TIC: {spoc['n_sectors'].mean():.2f}")

    print(f"\nLoading exclusion set...")
    excluded = load_excluded_tics()
    print(f"Total excluded: {len(excluded)}")

    print(f"\nFiltering...")
    before = len(spoc)
    spoc = spoc[
        (spoc['tmag'] >= args.tmag_min) &
        (spoc['tmag'] <= args.tmag_max) &
        (spoc['n_sectors'] >= args.min_sectors_observed) &
        (~spoc['tic_id'].isin(excluded))
    ]
    print(f"Tmag [{args.tmag_min}, {args.tmag_max}] + "
          f">={args.min_sectors_observed} sectors + not excluded: "
          f"{len(spoc)} (from {before})")

    if len(spoc) < args.count:
        print(f"\nWARNING: only {len(spoc)} candidates. Using all of them.")
        sample = spoc
    else:
        sample = spoc.sample(n=args.count, random_state=args.seed)

    sample = sample.sort_values('tic_id')

    print(f"\nWriting {len(sample)} TICs to {args.output}")
    with open(args.output, 'w') as f:
        f.write(f"# Novel hunt: SPOC sectors {args.sectors[0]}-{args.sectors[-1]}, "
                f"Tmag [{args.tmag_min}, {args.tmag_max}], "
                f">={args.min_sectors_observed} sectors observed\n")
        f.write(f"# Excluded {len(excluded)} TICs from TOI / previous hunts\n")
        f.write(f"# {len(sample)} TICs sampled, seed={args.seed}\n")
        for tic in sample['tic_id']:
            f.write(f"{tic}\n")

    print(f"\nDistribution of sampled targets:")
    print(f"Tmag        median: {sample['tmag'].median():.2f}, "
          f"range: [{sample['tmag'].min():.2f}, {sample['tmag'].max():.2f}]")
    print(f"N sectors   median: {sample['n_sectors'].median():.0f}, "
          f"range: [{sample['n_sectors'].min()}, {sample['n_sectors'].max()}]")
    print(f"\nDone.")

if __name__ == '__main__':
    main()