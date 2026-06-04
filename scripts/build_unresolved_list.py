"""
Building TIC list for unresolved TESS candidates
"""
from pathlib import Path
import pandas as pd

CATALOGS = Path("data/catalogs")
OUTPUT = CATALOGS / "tess_discovery_unresolved.txt"

toi = pd.read_csv(CATALOGS / "tess_toi.csv")
unresolved = toi[toi['TFOPWG Disposition'].isin(['PC', 'APC'])].copy()

# Remove any TIC already processed in Tier 1 (in case of catalogue overlap)
tier1_done = set()
tier1_path = Path("results/tier1_results.csv")
if tier1_path.exists():
    tier1_done = set(pd.read_csv(tier1_path)['tic_id'].astype(int))

tics_all = set(int(t) for t in unresolved['TIC ID'].dropna())
tics = sorted(tics_all - tier1_done)

print(f"Unresolved TOIs (PC + APC): {len(unresolved)}")
print(f"Unique TIC IDs: {len(tics_all)}")
print(f"Already processed in Tier 1: {len(tics_all & tier1_done)}")
print(f"To process now: {len(tics)}")

with open(OUTPUT, 'w') as f:
    f.write("# Unresolved TESS candidates\n")
    f.write(f"# {len(tics)} unique TIC IDs\n")
    for tic in tics:
        f.write(f"{tic}\n")

print(f"\nWritten to {OUTPUT}")