"""Retry KOIs that failed during the bulk fetch."""
from pathlib import Path
from astroquery.mast import Observations
from tqdm import tqdm

FAILURE_LOG = Path("data/catalogs/fetch_failures.log")
RAW_DIR = Path("data/raw/kepler")

# Read failed kepids
with open(FAILURE_LOG) as f:
    failed_kepids = [int(line.split("\t")[0]) for line in f if line.strip()]

print(f"Retrying {len(failed_kepids)} failed targets...")

still_failed = []
for kepid in tqdm(failed_kepids):
    target_dir = RAW_DIR / f"{kepid:09d}"
    if target_dir.exists() and any(target_dir.rglob("*.fits")):
        continue

    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        obs = Observations.query_criteria(
            target_name=f"kplr{kepid:09d}",
            obs_collection="Kepler",
            dataproduct_type="timeseries",
        )
        if len(obs) == 0:
            still_failed.append((kepid, "no observations"))
            continue
        products = Observations.get_product_list(obs)
        lc_products = Observations.filter_products(
            products,
            productSubGroupDescription="LLC",
            extension="fits",
        )
        Observations.download_products(
            lc_products,
            download_dir=str(target_dir),
            verbose=False,
        )
    except Exception as e:
        still_failed.append((kepid, str(e)[:100]))
        continue

print(f"\nRecovered: {len(failed_kepids) - len(still_failed)}")
print(f"Still failed: {len(still_failed)}")
if still_failed:
    with open("data/catalogs/fetch_failures_round2.log", "w") as f:
        for kepid, reason in still_failed:
            f.write(f"{kepid}\t{reason}\n")