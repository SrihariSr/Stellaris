import numpy as np

GLOBAL_VIEW_BINS = 2001
LOCAL_VIEW_BINS = 201
LOCAL_VIEW_NUM_DURATIONS = 4 # window = +- 4 transit durations

def _phase_fold(time: np.ndarray, period: float, epoch: float) -> np.ndarray:
    """
    Return phase in [-0.5, +0.5), with transit at phase 0.
    """
    time = np.asarray(time, dtype=np.float64)
    phase = np.mod(time - epoch, period) / period  # phase in [0, 1)
    phase = np.where(phase > 0.5, phase - 1.0, phase)  # shift to [-0.5, 0.5)
    return phase

def _bin_flux(
    phase: np.ndarray,
    flux: np.ndarray,
    num_bins: int,
    phase_min: float,
    phase_max: float
    ) -> np.ndarray:
    """
    Bin flux values by phase, taking the median of each bin.
    Empty bins are resolved by linear interpolation from it's neighbours.
    """
    bin_edges = np.linspace(phase_min, phase_max, num_bins + 1)
    bin_indices = np.digitize(phase, bin_edges) - 1

    binned = np.full(num_bins, np.nan)
    for b in range(num_bins):
        in_bin = bin_indices == b
        if in_bin.any():
            binned[b] = np.median(flux[in_bin])
    
    if np.isnan(binned).any():
        valid = ~np.isnan(binned)
        if valid.sum() < 2:
            raise ValueError("Too few valid bins to interpolate :(")
        x = np.arange(num_bins)
        binned = np.interp(x, x[valid], binned[valid])
    
    return binned


def _normalise(view: np.ndarray) -> np.ndarray:
    """
    Centre on 0, scale such that the deepest bin is at -1.
    """
    view = view - np.median(view)
    depth = np.abs(view.min())
    if depth < 1e-10:
        # Flat view, nothing to normalise
        return view
    return view / depth

def make_global_view(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    epoch: float,
    num_bins: int = GLOBAL_VIEW_BINS
    ) -> np.ndarray:
    """
    Generate the global view: full phase-folded light curve, binned.
    """
    phase = _phase_fold(time, period, epoch)
    binned = _bin_flux(phase, flux, num_bins, phase_min=-0.5, phase_max=0.5)
    return _normalise(binned)

def make_local_view(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    epoch: float,
    duration: float,
    num_bins: int=LOCAL_VIEW_BINS,
    num_durations: int=LOCAL_VIEW_NUM_DURATIONS
    ) -> np.ndarray:
    """
    Generate the local view: zoomed window around the transit, binned.
    """
    phase = _phase_fold(time, period, epoch)

    half_window = (num_durations * duration) / period
    if half_window >= 0.5:
        raise ValueError(f"Local window ({half_window:.3f}) exceeds phase range (0.5).\n This is caused by num_durations being too large or the transit being unusually long.")
    
    mask = np.abs(phase) < half_window
    if mask.sum() < num_bins // 4:
        raise ValueError(f"Only {mask.sum()} cadences present in the local window, require at least {num_bins // 4} cadences for reliable binning.")
    
    binned = _bin_flux(phase[mask], flux[mask], num_bins, phase_min=-half_window, phase_max=half_window)

    return _normalise(binned)
