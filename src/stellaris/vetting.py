"""Classical vetting checks to filter false positives that the CNN can't tell apart.

The CNN identifies transit-shaped signals. Many transit-shaped signals come from
eclipsing binaries, not planets. These rule-based diagnostics catch the most
obvious EB signatures:
  - depth cap (deep dips are usually EBs)
  - odd/even transit depth comparison (two stars often have different sizes)

A V-shape check exists in this module but is not used in the default pipeline
because a single heuristic doesn't reliably distinguish V from U at variable
signal-to-noise. Proper V-vs-U discrimination requires fitting a transit model
(Mandel-Agol with limb darkening); leaving that as future work.
"""
from dataclasses import dataclass, field
import numpy as np


# A signal with depth > 3% on any star is almost always an EB.
# Real Jupiter-sized planets are at ~1-2%; below 3% covers practically all planets.
MAX_DEPTH_FRAC = 0.03

# Odd and even transits of a real planet should have nearly identical depth.
# EBs frequently show 5-20% depth differences. We allow up to 15% relative diff.
MAX_ODD_EVEN_DEPTH_DIFF = 0.15


@dataclass
class VettingResult:
    """Outcome of all vetting checks for one candidate."""
    passed: bool
    reasons: list[str] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)


def check_depth(depth_fraction: float, max_depth: float = MAX_DEPTH_FRAC) -> tuple[bool, str | None]:
    """Reject signals deeper than max_depth (default 3%).

    Real planets rarely exceed 2% transit depth. Deeper signals are almost
    always eclipsing binaries or stellar variability.
    """
    abs_depth = abs(depth_fraction)
    if abs_depth > max_depth:
        return False, f"depth {abs_depth*100:.2f}% exceeds {max_depth*100:.1f}% cap"
    return True, None


def check_odd_even(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    epoch: float,
    duration: float,
    max_diff: float = MAX_ODD_EVEN_DEPTH_DIFF,
) -> tuple[bool, str | None, dict]:
    """Compare median flux at odd-numbered vs even-numbered transits.

    For each cadence, compute which transit number it belongs to. Take the
    median in-transit flux for odd-numbered transits and for even-numbered
    transits. If they differ by more than `max_diff` relative, flag as EB.
    """
    transit_number = np.floor((time - epoch + 0.5 * period) / period).astype(int)
    phase = ((time - epoch + 0.5 * period) % period) / period - 0.5

    half_window_phase = (duration / 2) / period
    in_transit = np.abs(phase) < half_window_phase

    if not in_transit.any():
        return True, None, {"odd_depth": 0, "even_depth": 0}

    is_odd = (transit_number % 2) == 1
    odd_in_transit = in_transit & is_odd
    even_in_transit = in_transit & ~is_odd

    if not odd_in_transit.any() or not even_in_transit.any():
        return True, None, {"odd_depth": 0, "even_depth": 0, "note": "missing parity"}

    odd_depth = abs(np.median(flux[odd_in_transit]))
    even_depth = abs(np.median(flux[even_in_transit]))

    mean_depth = (odd_depth + even_depth) / 2
    if mean_depth < 1e-6:
        return True, None, {"odd_depth": odd_depth, "even_depth": even_depth}

    rel_diff = abs(odd_depth - even_depth) / mean_depth
    diagnostics = {
        "odd_depth": float(odd_depth),
        "even_depth": float(even_depth),
        "rel_diff": float(rel_diff),
    }

    if rel_diff > max_diff:
        reason = (f"odd/even depth differ by {rel_diff*100:.1f}% "
                  f"({odd_depth*1e6:.0f} vs {even_depth*1e6:.0f} ppm)")
        return False, reason, diagnostics

    return True, None, diagnostics


def check_v_shape(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    epoch: float,
    duration: float,
    min_ratio: float = 0.6,
) -> tuple[bool, str | None, dict]:
    """[Not currently used] Reject signals with V-shaped (rather than U-shaped) transits.

    Disabled in the default pipeline because the simple inner/outer ratio
    measures transit-duration-vs-ingress-time rather than U-vs-V geometry,
    and doesn't generalise across signal-to-noise regimes.

    Kept here for reference and future re-implementation with proper model
    fitting (Mandel-Agol vs. V-model comparison).
    """
    phase = ((time - epoch + 0.5 * period) % period) / period - 0.5
    half_window_phase = (duration / 2) / period

    in_transit = np.abs(phase) < half_window_phase
    if not in_transit.any():
        return True, None, {"v_shape_ratio": 1.0}

    inner_mask = in_transit & (np.abs(phase) < half_window_phase / 3)
    outer_mask = in_transit & (np.abs(phase) >= half_window_phase / 3)

    if not inner_mask.any() or not outer_mask.any():
        return True, None, {"v_shape_ratio": 1.0}

    inner_depth = abs(np.median(flux[inner_mask]))
    outer_depth = abs(np.median(flux[outer_mask]))

    if inner_depth < 1e-7:
        return True, None, {"v_shape_ratio": 1.0}

    ratio = outer_depth / inner_depth
    diagnostics = {
        "inner_depth_ppm": float(inner_depth * 1e6),
        "outer_depth_ppm": float(outer_depth * 1e6),
        "v_shape_ratio": float(ratio),
    }

    if ratio < min_ratio:
        reason = (f"V-shaped transit (outer/inner depth ratio {ratio:.2f} "
                  f"< {min_ratio})")
        return False, reason, diagnostics

    return True, None, diagnostics


def vet_candidate(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    epoch: float,
    duration: float,
    depth: float,
) -> VettingResult:
    """Run all vetting checks and return combined verdict."""
    result = VettingResult(passed=True)

    # Check 1: depth cap
    passed, reason = check_depth(depth)
    if not passed:
        result.passed = False
        result.reasons.append(reason)
    result.diagnostics["depth_fraction"] = abs(depth)

    # Check 2: odd vs even depth
    passed, reason, diag = check_odd_even(time, flux, period, epoch, duration)
    result.diagnostics["odd_even"] = diag
    if not passed:
        result.passed = False
        result.reasons.append(reason)

    return result