"""Local-only TESS loading and an npz cache of preprocessed light curves with sector labels."""
import os
from pathlib import Path

import lightkurve as lk
import numpy as np

from stellaris.tess_preprocess import preprocess_tess

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_TESS_DIR = PROJECT_ROOT / "data" / "raw" / "tess"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "processed" / "tess_cache"
# macOS st_flags bit for cloud-only placeholders (OneDrive Files On-Demand): reading one downloads it.
SF_DATALESS = 0x40000000


def local_fits_paths(tic_id: int, raw_dir: Path = RAW_TESS_DIR) -> list[Path]:
    """The files the cache-hit branch of fetch_tess_lightcurve would read, in its order."""
    target_dir = Path(raw_dir) / f"{tic_id:09d}"
    if not target_dir.is_dir():
        return []
    return sorted(target_dir.rglob("*lc.fits"))


def dataless_files(paths: list[Path]) -> list[Path]:
    return [p for p in paths if getattr(os.stat(p), "st_flags", 0) & SF_DATALESS]


def load_local_tess(tic_id: int, raw_dir: Path = RAW_TESS_DIR) -> lk.LightCurveCollection:
    """
    Read every cached light curve for a target; there is no MAST fallback.
    Cloud-only placeholders are still downloaded by the OS on access (see dataless_files).
    """
    files = local_fits_paths(tic_id, raw_dir)
    if not files:
        raise FileNotFoundError(f"No local *lc.fits for TIC {tic_id} under {Path(raw_dir) / f'{tic_id:09d}'}")
    return lk.LightCurveCollection([lk.read(str(f)) for f in files])


def preprocess_with_sectors(lcc: lk.LightCurveCollection) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run preprocess_tess unchanged and label each surviving cadence with its sector.

    Returns (time, flux, sector). A cadence takes the SECTOR of the first light
    curve whose raw time span contains it, or -1 if none does.
    """
    clean = preprocess_tess(lcc)
    time = np.asarray(clean.time.value, dtype=np.float64)
    flux = np.asarray(clean.flux.value, dtype=np.float64)
    sector = np.full(time.size, -1, dtype=np.int32)
    for lc in lcc:
        raw_time = np.asarray(lc.time.value, dtype=np.float64)
        raw_time = raw_time[np.isfinite(raw_time)]
        if raw_time.size == 0:
            continue
        inside = (time >= raw_time.min()) & (time <= raw_time.max()) & (sector == -1)
        sector[inside] = int(lc.meta.get("SECTOR", -1))
    return time, flux, sector


def cache_path(cache_dir: Path, tic_id: int) -> Path:
    return Path(cache_dir) / f"{tic_id:09d}.npz"


def save_cache(
    path: Path,
    time: np.ndarray,
    flux: np.ndarray,
    sector: np.ndarray,
    n_raw_cadences: int,
    source_files: list[str],
) -> None:
    """Write atomically so an interrupted run never leaves a truncated file behind."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(
            fh,
            time=np.asarray(time, dtype=np.float64),
            flux=np.asarray(flux, dtype=np.float64),
            sector=np.asarray(sector, dtype=np.int32),
            n_raw_cadences=np.int64(n_raw_cadences),
            source_files=np.asarray(source_files, dtype=str),
        )
    tmp.replace(path)


def load_cache(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {
            "time": data["time"],
            "flux": data["flux"],
            "sector": data["sector"],
            "n_raw_cadences": int(data["n_raw_cadences"]),
            "source_files": [str(s) for s in data["source_files"]],
        }
