"""
Box Least Squares (BLS) transit search.

Wraps astropy's BLS implementation with sensible defaults for finding
transit candidates in cleaned light curves.
"""
from dataclasses import dataclass
import numpy as np
from astropy.timeseries import BoxLeastSquares
import astropy.units as u

@dataclass
class BLSResult:
    """
    One candidate signal from a BLS search.
    """
    period: float   # days
    epoch: float    # BTJD/BKJD
    duration: float # days
    depth: float    # fractional flux
    power: float    # BLS SDE-equivalent (higher = stronger)
    snr: float      # signal-to-noise estimate


def run_bls(
    time: np.ndarray,
    flux: np.ndarray,
    min_period: float = 0.5,
    max_period: float = 30.0,
    duration_grid: np.ndarray | None = None,
) -> BLSResult:
    """
    Run BLS on a cleaned light curve and return the top candidate.
    """
    # astropy BLS expects flux with baseline = 1
    # Add 1 back as data has been centered around 0 so far.
    flux_for_bls = flux + 1.0

    if duration_grid is None:
        duration_grid = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]) / 24.0  # Converting hours to days

    bls = BoxLeastSquares(
        time * u.day,
        flux_for_bls,
    )

    # Build period grid. Logarithmic spacing finds short and long periods alike.
    n_periods = 50000
    period_grid = np.exp(
        np.linspace(np.log(min_period), np.log(max_period), n_periods)
    ) * u.day

    periodogram = bls.power(period_grid, duration_grid * u.day)

    # Find the peak
    idx = np.argmax(periodogram.power)
    best_period = float(periodogram.period[idx].value)
    best_t0 = float(periodogram.transit_time[idx].value)
    best_duration = float(periodogram.duration[idx].value)
    best_depth = float(periodogram.depth[idx])
    best_power = float(periodogram.power[idx])

    # SNR: ratio of peak power to median power away from peak
    median_power = np.median(periodogram.power)
    snr = best_power / median_power if median_power > 0 else best_power

    return BLSResult(
        period=best_period,
        epoch=best_t0,
        duration=best_duration,
        depth=best_depth,
        power=best_power,
        snr=snr,
    )