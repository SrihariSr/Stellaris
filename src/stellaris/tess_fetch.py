"""Fetch TESS light curves from MAST, with local caching."""
from pathlib import Path
import lightkurve as lk


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TESS_CACHE_DIR = PROJECT_ROOT / "data" / "raw" / "tess"
TESS_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def fetch_tess_lightcurve(
    tic_id: int,
    sectors: list[int] | None = None,
    author: str = "SPOC",
    exptime: int = 120,
) -> lk.LightCurveCollection:
    """
    Fetch all TESS light curves for a target, caching to disk.
    """
    target_dir = TESS_CACHE_DIR / f"{tic_id:09d}"
    target_dir.mkdir(parents=True, exist_ok=True)

    # First, check disk cache
    cached = sorted(target_dir.rglob("*lc.fits"))
    if cached and sectors is None:
        # Have at least one cached light curve for this target
        return lk.LightCurveCollection([lk.read(str(f)) for f in cached])

    # Cache miss or specific sectors requested
    search_kwargs = {
        "mission": "TESS",
        "author": author,
        "exptime": exptime,
    }
    if sectors is not None:
        search_kwargs["sector"] = sectors

    search = lk.search_lightcurve(f"TIC {tic_id}", **search_kwargs)

    if len(search) == 0:
        raise FileNotFoundError(f"No {author} {exptime}s light curves for TIC {tic_id} in sectors {sectors}" if sectors else "")
        
    # Download to cache directory
    lcc = search.download_all(download_dir=str(target_dir))
    return lcc

def list_cached_tics() -> list[int]:
    """Return TIC IDs with at least one cached light curve on disk."""
    tics = []
    for child in TESS_CACHE_DIR.iterdir():
        if not child.is_dir():
            continue
        try:
            tic = int(child.name)
        except ValueError:
            continue
        if any(child.rglob("*lc.fits")):
            tics.append(tic)
    return sorted(tics)
    