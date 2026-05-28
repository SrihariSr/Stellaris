"""Bulk-fetch Kepler light curves for all training-set KOIs."""
from pathlib import Path
import pandas as pd
from astroquery.mast import Observations
from tqdm import tqdm

CATALOGS = Path("data/catalogs")
RAW_DIR = Path("data/raw/kepler")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Safety check: don't run from a sync folder
if any(x in str(RAW_DIR.absolute()) for x in ["OneDrive", "iCloud", "CloudStorage"]):
    raise RuntimeError(
        f"RAW_DIR is inside a cloud sync folder: {RAW_DIR.absolute()}. Move the project or exclude data/ from sync before running."
    )

train_koi = pd.read_csv(CATALOGS / "kepler_koi_train.csv")
kepids = sorted(train_koi['kepid'].unique())
print(f"Unique stars to fetch: {len(kepids)}")

failures = []
for kepid in tqdm(kepids):
    target_dir = RAW_DIR / f"{kepid:09d}"

    # Skip if we already have files
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
            failures.append((kepid, "no observations"))
            continue

        products = Observations.get_product_list(obs)
        lc_products = Observations.filter_products(
            products,
            productSubGroupDescription="LLC",
            extension="fits",
        )

        if len(lc_products) == 0:
            failures.append((kepid, "no LLC products"))
            continue

        Observations.download_products(
            lc_products,
            download_dir=str(target_dir),
            verbose=False,
        )
    except Exception as e:
        failures.append((kepid, str(e)[:100]))
        continue

# Report
print(f"\nCompleted. Failures: {len(failures)}")
if failures:
    with open("data/catalogs/fetch_failures.log", "w") as f:
        for kepid, reason in failures:
            f.write(f"{kepid}\t{reason}\n")
    print("Failures logged to data/catalogs/fetch_failures.log")