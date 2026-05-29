"""Load Kepler light curves from the local FITS cache."""
from pathlib import Path
import lightkurve as lk

# Resolve project root from this file's location. This works regardless of
# where Python is invoked from, which matters because the same module is
# imported by scripts, notebooks, and tests.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
KEPLER_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "kepler"


def get_kepler_data_dir() -> Path:
    """Return the absolute path to the local Kepler data directory."""
    if not KEPLER_DATA_DIR.exists():
        raise FileNotFoundError(
            f"Kepler data directory not found: {KEPLER_DATA_DIR}\n"
            "Run scripts/fetch_kepler_lightcurves.py first."
        )
    return KEPLER_DATA_DIR


def list_available_kepids() -> list[int]:
    """Return all kepids with at least one FITS file on disk."""
    root = get_kepler_data_dir()
    kepids = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        try:
            kepid = int(child.name)
        except ValueError:
            continue  # ignore non-numeric folder names
        if any(child.rglob("*_llc.fits")):
            kepids.append(kepid)
    return sorted(kepids)


def load_kepler_lightcurve(kepid: int) -> lk.LightCurveCollection:
    """
    Load all available quarters for a Kepler target from local cache.
    """
    target_dir = get_kepler_data_dir() / f"{kepid:09d}"

    if not target_dir.exists():
        raise FileNotFoundError(
            f"No directory for kepid={kepid} at {target_dir}"
        )

    fits_files = sorted(target_dir.rglob("*_llc.fits"))

    if not fits_files:
        raise FileNotFoundError(
            f"No LLC FITS files found for kepid={kepid} under {target_dir}"
        )

    light_curves = [lk.read(str(f)) for f in fits_files]
    return lk.LightCurveCollection(light_curves)