"""Check that views.py at default arguments is bit-identical to the HEAD version."""
import subprocess
import sys
import types
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stellaris import views as new  # noqa: E402

N_CURVES = 90


def load_head_views() -> types.ModuleType:
    source = subprocess.run(
        ["git", "show", "HEAD:src/stellaris/views.py"],
        cwd=PROJECT_ROOT, check=True, capture_output=True, text=True,
    ).stdout
    module = types.ModuleType("views_head")
    exec(compile(source, "views_head.py", "exec"), module.__dict__)
    return module


def synthetic_curve(rng: np.random.Generator, flat: bool = False):
    period = float(np.exp(rng.uniform(np.log(0.5), np.log(20.0))))
    duration = float(rng.uniform(1.0, 8.0)) / 24.0
    epoch = float(rng.uniform(0.0, period))
    span = float(rng.uniform(25.0, 80.0))
    time = np.sort(rng.uniform(0.0, span, size=int(span * 720)))  # ~2-min cadence
    if flat:
        return time, np.zeros_like(time), period, epoch, duration
    depth = float(np.exp(rng.uniform(np.log(1e-4), np.log(5e-2))))
    phase = np.mod(time - epoch + 0.5 * period, period) - 0.5 * period
    flux = np.where(np.abs(phase) < duration / 2, -depth, 0.0)
    flux = flux + rng.normal(0.0, float(rng.uniform(1e-4, 5e-3)), size=time.size)
    return time, flux, period, epoch, duration


def expected_scale(binned: np.ndarray) -> float:
    centred = binned - np.median(binned)
    depth = np.abs(centred.min())
    return float(depth) if depth >= 1e-10 else 0.0


def call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs), None
    except ValueError as exc:
        return None, str(exc)


def main() -> int:
    head = load_head_views()
    rng = np.random.default_rng(20261010)
    failures: list[str] = []
    n_global = n_local = n_errors = n_flat = 0

    cases = [synthetic_curve(rng) for _ in range(N_CURVES)] + [synthetic_curve(rng, flat=True)]
    # A wide transit on a short period: the local window exceeds half a phase.
    t, f, _, e, _ = synthetic_curve(rng)
    cases.append((t, f, 0.6, e, 8.0 / 24.0))

    for i, (time, flux, period, epoch, duration) in enumerate(cases):
        g_head, g_err_head = call(head.make_global_view, time, flux, period, epoch)
        g_new, g_err_new = call(new.make_global_view, time, flux, period, epoch)
        gs_new, gs_err_new = call(new.make_global_view, time, flux, period, epoch, return_scale=True)
        if g_err_head != g_err_new or g_err_head != gs_err_new:
            failures.append(f"case {i}: global error mismatch {g_err_head!r} vs {g_err_new!r}")
        elif g_head is not None:
            n_global += 1
            view, scale = gs_new
            if not (np.array_equal(g_head, g_new) and np.array_equal(g_head, view)):
                failures.append(f"case {i}: global view differs")
            if scale == 0.0:
                n_flat += 1
            binned = head._bin_flux(head._phase_fold(time, period, epoch), flux, head.GLOBAL_VIEW_BINS, -0.5, 0.5)
            expected = expected_scale(binned)
            if scale != expected:
                failures.append(f"case {i}: global scale {scale} != {expected}")

        l_head, l_err_head = call(head.make_local_view, time, flux, period, epoch, duration)
        l_new, l_err_new = call(new.make_local_view, time, flux, period, epoch, duration)
        ls_new, ls_err_new = call(new.make_local_view, time, flux, period, epoch, duration, return_scale=True)
        if l_err_head != l_err_new or l_err_head != ls_err_new:
            failures.append(f"case {i}: local error mismatch {l_err_head!r} vs {l_err_new!r}")
        elif l_head is None:
            n_errors += 1
        else:
            n_local += 1
            view, scale = ls_new
            if not (np.array_equal(l_head, l_new) and np.array_equal(l_head, view)):
                failures.append(f"case {i}: local view differs")
            phase = head._phase_fold(time, period, epoch)
            half_window = head.LOCAL_VIEW_NUM_DURATIONS * duration / period
            mask = np.abs(phase) < half_window
            binned = head._bin_flux(phase[mask], flux[mask], head.LOCAL_VIEW_BINS, -half_window, half_window)
            expected = expected_scale(binned)
            if scale != expected:
                failures.append(f"case {i}: local scale {scale} != {expected}")

    flat_view, flat_scale = new.make_global_view(*cases[N_CURVES][:4], return_scale=True)
    if flat_scale != 0.0 or not np.array_equal(flat_view, head.make_global_view(*cases[N_CURVES][:4])):
        failures.append("flat case: expected scale 0.0 and an unchanged view")

    if n_global < 50 or n_local < 50:
        failures.append(f"too few successful comparisons (global {n_global}, local {n_local})")
    if n_errors < 1:
        failures.append("no ValueError case exercised")

    print(f"global views compared: {n_global} (flat: {n_flat}); local views compared: {n_local}; "
          f"matching ValueError cases: {n_errors}")
    for line in failures:
        print("  " + line)
    print("PASS" if not failures else "FAIL")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
