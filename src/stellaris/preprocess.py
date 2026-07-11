"""This file cleans the Kepler light curves data."""
import numpy as np
import lightkurve as lk

DETREND_WINDOW_DAYS = 0.5
SIGMA_CLIP = 5.0

def clean_quarter(lc: lk.LightCurve) -> lk.LightCurve:
    """
    Cleans a single Kepler quarter by removing NaNs, applying quality mask, sigma clipping 
    and normalising the data to a median of 1.0.
    """

    # PDCSAP is NASA's pre-search-data-conditioned flux: already corrected
    # for known instrumental systematics. We use it as the starting point.
    lc = lc.remove_nans(column='pdcsap_flux')

    lc = lc[lc.quality == 0] # Quality = 0 means there are no flagged issues

    if len(lc) == 0:
        return lc
    
    median_flux = np.nanmedian(lc.pdcsap_flux.value)
    lc = lc.normalize()

    flux = lc.flux.value
    median = np.nanmedian(flux)
    std = np.nanstd(flux)
    retain = flux < median + (SIGMA_CLIP * std)
    
    lc = lc[retain]
    
    return lc

def stitch(lcc: lk.LightCurveCollection) -> lk.LightCurve:
    """
    Cleans each quarter and then stitches it into a single light curve.
    """
    cleaned = []

    for lc in lcc:
        cq = clean_quarter(lc)
        if len(cq) > 0:
            cleaned.append(cq)
    
    if not cleaned:
        raise ValueError("No usable cadences available after cleaning all quarters :(")
    
    return lk.LightCurveCollection(cleaned).stitch()

def detrend(lc: lk.LightCurve, window_days: float = DETREND_WINDOW_DAYS) -> lk.LightCurve:
    """Remove long-term trends with a biweight running filter."""
    try:
        from wotan import flatten
    except ImportError:
        raise ImportError(
            "wotan is required for detrending. Install it first .__."
        )

    time = lc.time.value
    flux = lc.flux.value
    
    flat_flux, _trend = flatten(
        time,
        flux,
        method='median',
        window_length=window_days * 2,  # longer to avoid eating transits
        return_trend=True,
        edge_cutoff=0,
        break_tolerance=0.5,
    )

    out = lc.copy()
    out.flux = flat_flux
    out = out.remove_nans()
    return out

def normalise(lc: lk.LightCurve) -> lk.LightCurve:
    """Center the flux data on 0."""

    out = lc.copy()
    out.flux = lc.flux.value - 1.0 # After detrending, the median is usually ~1.0

    return out

def preprocess(lcc: lk.LightCurveCollection) -> lk.LightCurveCollection:
    """
    Complete preprocessing pipeline:
    clean -> stitch -> detrend -> normalise
    """
    stitched = stitch(lcc)
    detrended = detrend(stitched)

    return normalise(detrended)
