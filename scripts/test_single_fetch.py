"""Verify the MAST download pipeline works for one known target."""
from astroquery.mast import Observations
from pathlib import Path

kepid = 10666592  # TrES-2b, a well-known hot Jupiter
target_dir = Path(f"data/raw/kepler/{kepid:09d}")
target_dir.mkdir(parents=True, exist_ok=True)

print(f"Querying MAST for kepid={kepid}...")
obs = Observations.query_criteria(
    target_name=f"kplr{kepid:09d}",
    obs_collection="Kepler",
    dataproduct_type="timeseries",
)
print(f"Observations found: {len(obs)}")

products = Observations.get_product_list(obs)
lc_products = Observations.filter_products(
    products,
    productSubGroupDescription="LLC",
    extension="fits",
)
print(f"Light curve files to download: {len(lc_products)}")

print("Downloading...")
manifest = Observations.download_products(
    lc_products,
    download_dir=str(target_dir),
    verbose=False,
)
print(f"Done. Files in {target_dir}:")
for f in sorted(target_dir.rglob("*.fits")):
    print(f"{f.relative_to(target_dir)} ({f.stat().st_size / 1e6:.1f} MB)")