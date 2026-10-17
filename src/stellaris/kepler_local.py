"""Find Kepler long-cadence FITS across several directories without opening cloud-only placeholders."""
from pathlib import Path

from stellaris.tess_cache import dataless_files


def llc_files(kepid: int, root: Path) -> list[Path]:
    target = Path(root) / f"{kepid:09d}"
    return sorted(target.rglob("*_llc.fits")) if target.is_dir() else []


def fully_local(kepid: int, root: Path) -> bool:
    """True when root holds at least one LLC file for kepid and none is a cloud-only placeholder."""
    files = llc_files(kepid, root)
    return bool(files) and not dataless_files(files)


def first_local_dir(kepid: int, roots: list[Path]) -> Path | None:
    """The first directory, in search order, that holds a fully local copy of kepid."""
    return next((Path(r) for r in roots if fully_local(kepid, r)), None)
