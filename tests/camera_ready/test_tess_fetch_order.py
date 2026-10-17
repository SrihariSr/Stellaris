"""Check that a fresh download and a cache hit in fetch_tess_lightcurve return the same chronological collection."""
import os
import shutil
import sys
from pathlib import Path
from unittest import mock

import lightkurve as lk
import numpy as np
from astropy.io import fits
from astropy.time import Time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stellaris import tess_fetch  # noqa: E402
from stellaris.tess_cache import DEFAULT_CACHE_DIR, cache_path, dataless_files, load_cache  # noqa: E402
from stellaris.tess_preprocess import preprocess_tess  # noqa: E402

# Sectors 9, 35, 62, 89, 99 and 100; lightkurve's search order puts 100 before 99.
TIC = 22221375
SOURCE_DIR = PROJECT_ROOT / "data" / "raw" / "tess" / f"{TIC:09d}"
WORK = PROJECT_ROOT / ".cr_scratch" / "smoke" / "test_tess_fetch_order"

failures: list[str] = []


def expect(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'ok' if ok else 'FAIL'}] {label}{(': ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


def lightkurve_order(files: list[Path]) -> list[Path]:
    """SearchResult sorts by year, then by the 'TESS Sector NN' string (other keys are equal here)."""
    def key(f: Path):
        year = Time(fits.getheader(f, 1)["TSTART"] + 2457000.0, format="jd").datetime.year
        return year, f"TESS Sector {int(fits.getheader(f, 0)['SECTOR']):02d}"
    return sorted(files, key=key)


class FakeSearch:
    """Stands in for a lightkurve SearchResult; download_all copies local files instead of fetching."""

    def __init__(self, files: list[Path]):
        self.files = files
        self.returned: list[str] = []

    def __len__(self) -> int:
        return len(self.files)

    def download_all(self, download_dir: str) -> lk.LightCurveCollection:
        out = []
        for src in self.files:
            dst = Path(download_dir) / src.relative_to(SOURCE_DIR)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            out.append(lk.read(str(dst)))
            self.returned.append(dst.name)
        return lk.LightCurveCollection(out)


def sectors(lcc) -> list[int]:
    return [int(lc.meta["SECTOR"]) for lc in lcc]


def same_collection(a, b) -> bool:
    return len(a) == len(b) and all(
        x.meta["SECTOR"] == y.meta["SECTOR"]
        and np.array_equal(np.asarray(x.time.value), np.asarray(y.time.value))
        and np.array_equal(np.ma.filled(x.flux.value, np.nan), np.ma.filled(y.flux.value, np.nan), equal_nan=True)
        for x, y in zip(a, b)
    )


def main() -> int:
    source = sorted(SOURCE_DIR.rglob("*lc.fits"))
    npz = cache_path(DEFAULT_CACHE_DIR, TIC)
    if not source or dataless_files(source) or not npz.exists() or os.stat(npz).st_flags & 0x40000000:
        print(f"SKIP: TIC {TIC} raw FITS or cache file is not available locally")
        return 0
    before = [(f.stat().st_size, f.stat().st_mtime_ns) for f in source]
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)

    wrong = lightkurve_order(source)
    fake = FakeSearch(wrong)
    calls = []

    def fake_search(target, **kwargs):
        calls.append((target, kwargs))
        return fake

    with mock.patch.object(tess_fetch, "TESS_CACHE_DIR", WORK), \
            mock.patch.object(tess_fetch.lk, "search_lightcurve", fake_search):
        fresh = tess_fetch.fetch_tess_lightcurve(TIC)
        hit = tess_fetch.fetch_tess_lightcurve(TIC)

    expect("one search, for the right target and product", len(calls) == 1 and calls[0][0] == f"TIC {TIC}"
           and calls[0][1] == {"mission": "TESS", "author": "SPOC", "exptime": 120}, str(calls))
    returned_order = [int(n.split("-s")[1][:4]) for n in fake.returned]
    expect("download_all returned the wrong order (sector 100 before 99)",
           returned_order != sorted(returned_order) and returned_order.index(100) < returned_order.index(99),
           str(returned_order))
    copied = sorted((WORK / f"{TIC:09d}").rglob("*lc.fits"))
    expect("fresh download is in sorted-path order",
           [Path(lc.meta["FILENAME"]).name for lc in fresh] == [f.name for f in copied]
           and sectors(fresh) == sorted(sectors(fresh)), str(sectors(fresh)))
    expect("fresh download is identical to the cache-hit result", same_collection(fresh, hit))
    original = lk.LightCurveCollection([lk.read(str(f)) for f in source])
    expect("and identical to reading the original files in sorted-path order", same_collection(fresh, original))

    cached = load_cache(npz)
    clean = preprocess_tess(fresh)
    expect("preprocess_tess equals the cached preprocess_with_sectors output",
           np.array_equal(clean.time.value, cached["time"]) and np.array_equal(clean.flux.value, cached["flux"]),
           f"{len(clean)} cadences")
    old = preprocess_tess(lk.LightCurveCollection([lk.read(str(f)) for f in wrong]))
    expect("the old download order would not have matched the cache",
           not (np.array_equal(old.time.value, cached["time"]) and np.array_equal(old.flux.value, cached["flux"])))
    expect("source files untouched (size and mtime)",
           [(f.stat().st_size, f.stat().st_mtime_ns) for f in source] == before)

    print("PASS" if not failures else f"FAIL ({len(failures)} checks)")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
