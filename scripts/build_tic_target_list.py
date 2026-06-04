"""Build a sorted, deduplicated list of TIC IDs from the TOI catalogue."""
from pathlib import Path
import pandas as pd

CATALOGS = Path("data/catalogs")
OUTPUT = CATALOGS / "tic_target_list.txt"

toi = pd.read_csv(CATALOGS / "tess_toi.csv")
print(f"TOI catalogue rows: {len(toi)}")

# Some TOIs share a star (multi-planet systems). De-duplicate by TIC.
tic_ids = sorted(set(int(t) for t in toi['TIC ID'].dropna()))
print(f"Unique TIC IDs: {len(tic_ids)}")

# Quick disposition summary so we know what we're hunting through
print("\nDisposition distribution:")
print(toi['TFOPWG Disposition'].value_counts(dropna=False))

# Write the list
with open(OUTPUT, 'w') as f:
    f.write("# TIC target list built from tess_toi.csv\n")
    f.write(f"# {len(tic_ids)} unique targets\n")
    for tic in tic_ids:
        f.write(f"{tic}\n")

print(f"\nWritten to {OUTPUT}")