from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

# pyrefly: ignore [missing-import]
from stellaris.fetch import load_kepler_lightcurve
# pyrefly: ignore [missing-import]
from stellaris.preprocess import preprocess
# pyrefly: ignore [missing-import]
from stellaris.views import make_global_view, make_local_view

CATALOGS = Path("data/catalogs")
PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = PROCESSED / "kepler_trainset.h5"
FAILURE_LOG = PROCESSED / "build_failures.log"

GLOBAL_VIEW_BINS = 2001
LOCAL_VIEW_BINS = 201

def process_one_koi(row) -> tuple[np.ndarray, np.ndarray] | None:
    """Process a single KOI row into (global_view, local_view).
    Returns None on any failure."""
    kepid = int(row['kepid'])
    period = float(row['koi_period'])
    epoch = float(row['koi_time0bk'])
    duration_hours = float(row['koi_duration'])

    # Catalogue can have NaN durations or zero periods for marginal cases
    if not np.isfinite(period) or period <= 0:
        raise ValueError(f"Bad period: {period}")
    if not np.isfinite(epoch):
        raise ValueError(f"Bad epoch: {epoch}")
    if not np.isfinite(duration_hours) or duration_hours <= 0:
        raise ValueError(f"Bad duration: {duration_hours}")

    duration_days = duration_hours / 24.0

    lcc = load_kepler_lightcurve(kepid)
    clean = preprocess(lcc)

    time = clean.time.value
    flux = clean.flux.value

    global_view = make_global_view(time, flux, period, epoch, num_bins=GLOBAL_VIEW_BINS)
    local_view = make_local_view(time, flux, period, epoch, duration_days, num_bins=LOCAL_VIEW_BINS)

    return global_view, local_view

def main():
    train_koi = pd.read_csv(CATALOGS / "kepler_koi_train.csv")
    print(f"Processing {len(train_koi)} labelled KOIs...")

    # Pre-allocate arrays for speed
    n = len(train_koi)
    global_views = np.full((n, GLOBAL_VIEW_BINS), np.nan, dtype=np.float32)
    local_views = np.full((n, LOCAL_VIEW_BINS), np.nan, dtype=np.float32)
    labels = np.zeros(n, dtype=np.int8)
    kepids = np.zeros(n, dtype=np.int64)
    success = np.zeros(n, dtype=bool)

    failures = []

    for i, (_, row) in enumerate(tqdm(train_koi.iterrows(), total=n)):
        try:
            result = process_one_koi(row)
            if result is None:
                failures.append((int(row['kepid']), "process returned None"))
                continue
            gv, lv = result
            global_views[i] = gv
            local_views[i] = lv
            labels[i] = int(row['label'])
            kepids[i] = int(row['kepid'])
            success[i] = True
        except Exception as e:
            failures.append((int(row['kepid']), str(e)[:100]))
            continue

    # Filter to successful examples
    success_count = success.sum()
    print(f"\nProcessed {success_count}/{n} KOIs successfully "
          f"({100 * success_count / n:.1f}%)")

    # Write HDF5
    print(f"Writing to {OUTPUT_PATH}...")
    with h5py.File(OUTPUT_PATH, 'w') as f:
        f.create_dataset('global_view', data=global_views[success],
                         compression='gzip', compression_opts=4)
        f.create_dataset('local_view', data=local_views[success],
                         compression='gzip', compression_opts=4)
        f.create_dataset('label', data=labels[success])
        f.create_dataset('kepid', data=kepids[success])
        f.attrs['n_examples'] = int(success_count)
        f.attrs['n_positives'] = int(labels[success].sum())
        f.attrs['n_negatives'] = int((labels[success] == 0).sum())

    # Log failures
    if failures:
        with open(FAILURE_LOG, 'w') as f:
            for kepid, reason in failures:
                f.write(f"{kepid}\t{reason}\n")
        print(f"Failures logged to {FAILURE_LOG}")

    file_size_mb = OUTPUT_PATH.stat().st_size / 1e6
    print(f"\nDone. Output file: {file_size_mb:.1f} MB")


if __name__ == '__main__':
    main()