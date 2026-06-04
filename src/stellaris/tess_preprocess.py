"""
Preprocess TESS light curves. Mostly identical to Kepler preprocessing,
with adjustments for TESS-specific systematics.
"""
import numpy as np
import lightkurve as lk

SIGMA_CLIP = 5.0
DETREND_WINDOW_DAYS = 1.0

def _clean_sector(lc: lk.LightCurve) -> lk.LightCurve:
    """
    Clean a single TESS sector with NaN removal, 
    quality mask, sigma clip and normalisation.
    """
    lc = lc.remove_nans(column='pdcsap_flux')
    lc = lc[lc.quality == 0]
    if len(lc) == 0:
        return lc

    lc = lc.normalize()

    # Asymmetric sigma clip: upward only, preserves transit dips
    flux = lc.flux.value
    median = np.nanmedian(flux)
    std = np.nanstd(flux)
    keep = flux < median + SIGMA_CLIP * std
    lc = lc[keep]
    return lc

def preprocess_tess(lcc: lk.LightCurveCollection) -> lk.LightCurve:
    """
    Full TESS preprocessing: clean each sector, stitch, detrend, normalise.
    """
    try:
        from wotan import flatten
    except ImportError:
        raise ImportError("wotan required :( \nInstall first")

    # Clean each sector
    cleaned = [_clean_sector(lc) for lc in lcc]
    cleaned = [c for c in cleaned if len(c) > 0]
    if not cleaned:
        raise ValueError("No usable cadences after cleaning.")

    # Stitch (normalises medians to 1, then concatenates)
    stitched = lk.LightCurveCollection(cleaned).stitch()

    # Detrend
    time = stitched.time.value
    flux = stitched.flux.value
    flat_flux, _ = flatten(
        time,
        flux,
        method='median',
        window_length=DETREND_WINDOW_DAYS,
        return_trend=True,
        edge_cutoff=0,
        break_tolerance=0.5,
    )

    out = stitched.copy()
    out.flux = flat_flux
    out = out.remove_nans()

    # Centre on 0
    out.flux = out.flux.value - 1.0
    return out