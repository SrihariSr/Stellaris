"""Build TIC list for Tier 1: confirmed and known TESS planets."""
from pathlib import Path
import pandas as pd

CATALOGS = Path("data/catalogs")
OUTPUT = CATALOGS / "tess_validation_confirmed.txt"

toi = pd.read_csv(CATALOGS / "tess_toi.csv")
confirmed = toi[toi['TFOPWG Disposition'].isin(['CP', 'KP'])].copy()
tics = sorted(set(int(t) for t in confirmed['TIC ID'].dropna()))

print(f"Confirmed (CP) + Known (KP) TOIs: {len(confirmed)}")
print(f"Unique TIC IDs (after deduplication): {len(tics)}")

with open(OUTPUT, 'w') as f:
    f.write("Confirmed and known TESS planets (CP + KP)\n")
    f.write(f"{len(tics)} unique TIC IDs\n")
    for tic in tics:
        f.write(f"{tic}\n")

print(f"Written to {OUTPUT}")