"""Fetch the reference catalogues from the NASA Exoplanet Archive."""
from pathlib import Path
import subprocess

BASE = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
OUT = Path("data/catalogs")
OUT.mkdir(parents=True, exist_ok=True)

QUERIES = {
    "kepler_koi.csv":        "select * from cumulative",
    "tess_toi.csv":          "select * from toi",
    "confirmed_planets.csv": "select * from pscomppars",
}

for filename, query in QUERIES.items():
    url = f"{BASE}?query={query.replace(' ', '+')}&format=csv"
    out = OUT / filename
    print(f"Fetching {filename}...")
    subprocess.run(["curl", "-o", str(out), url], check=True)
    print(f"  -> {out.stat().st_size / 1e6:.1f} MB")