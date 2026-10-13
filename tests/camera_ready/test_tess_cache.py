"""Check tess_cache.py: local-only loading, sector labels, and the npz round trip."""
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stellaris.tess_cache import (  # noqa: E402
    RAW_TESS_DIR, dataless_files, load_cache, load_local_tess, local_fits_paths, preprocess_with_sectors,
    save_cache,
)
from stellaris.tess_preprocess import preprocess_tess  # noqa: E402

# Fully local (not OneDrive placeholders) on the development machine; skipped otherwise.
CANDIDATES = [1003831, 926898, 3784444]

failures: list[str] = []


def expect(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'ok' if ok else 'FAIL'}] {label}{(': ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        try:
            load_local_tess(123456789, Path(tmp))
            expect("missing target raises FileNotFoundError", False)
        except FileNotFoundError:
            expect("missing target raises FileNotFoundError", True)

        time = np.linspace(0, 10, 500)
        flux = np.sin(time) * 1e-3
        sector = np.repeat([3, 4], 250).astype(np.int32)
        path = Path(tmp) / "000000042.npz"
        save_cache(path, time, flux, sector, 512, ["a_lc.fits", "b_lc.fits"])
        back = load_cache(path)
        expect("npz round trip", np.array_equal(back["time"], time) and np.array_equal(back["flux"], flux)
               and np.array_equal(back["sector"], sector) and back["n_raw_cadences"] == 512
               and back["source_files"] == ["a_lc.fits", "b_lc.fits"])
        expect("no temporary files left behind", sorted(p.name for p in Path(tmp).iterdir()) == ["000000042.npz"])

    tic = next((t for t in CANDIDATES if local_fits_paths(t) and not dataless_files(local_fits_paths(t))), None)
    if tic is None:
        print(f"  [skip] none of {CANDIDATES} is fully local under {RAW_TESS_DIR}")
    else:
        lcc = load_local_tess(tic)
        expect(f"TIC {tic}: one light curve per local file", len(lcc) == len(local_fits_paths(tic)))
        reference = preprocess_tess(lcc)
        time, flux, sector = preprocess_with_sectors(lcc)
        expect(f"TIC {tic}: time and flux identical to preprocess_tess",
               np.array_equal(time, reference.time.value) and np.array_equal(flux, reference.flux.value))
        sectors = {int(lc.meta["SECTOR"]) for lc in lcc}
        expect(f"TIC {tic}: every cadence labelled with an observed sector",
               set(np.unique(sector).tolist()) <= sectors and not (sector == -1).any(),
               f"{sorted(set(np.unique(sector).tolist()))}")

    print("PASS" if not failures else f"FAIL ({len(failures)} checks)")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
