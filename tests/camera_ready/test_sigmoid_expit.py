"""Check that scipy.special.expit matches 1 / (1 + exp(-x)) wherever the old form does not overflow."""
import sys
import warnings

import numpy as np
from scipy.special import expit

FIXED = [-1e4, -1001.0, -1000.0, -500.0, -104.0, -103.9, -100.0, -88.8, -88.7, -88.0, -50.0, -20.0,
         -10.0, -1.0, -1e-3, 0.0, 1e-3, 1.0, 10.0, 20.0, 50.0, 88.0, 100.0, 500.0, 1000.0, 1001.0, 1e4]


def main() -> int:
    failures = []
    grid = np.concatenate([FIXED, np.linspace(-120, 120, 200_001),
                           np.random.default_rng(0).normal(0.0, 15.0, 100_000)])
    for dtype in (np.float32, np.float64):
        x = grid.astype(dtype)
        with warnings.catch_warnings(record=True) as old_warnings:
            warnings.simplefilter("always")
            e = np.exp(-x)
            old = 1.0 / (1.0 + e)
        overflow = ~np.isfinite(e)
        with warnings.catch_warnings(record=True) as new_warnings:
            warnings.simplefilter("always")
            new = expit(x)
        outside = ~overflow
        n_diff_outside = int((old[outside] != new[outside]).sum())
        n_diff_inside = int((old[overflow] != new[overflow]).sum())
        print(f"  {dtype.__name__}: {len(x)} logits, old form overflowed on {int(overflow.sum())} "
              f"(warnings: {len(old_warnings)}), expit warnings: {len(new_warnings)}; "
              f"differences outside overflow {n_diff_outside}, inside overflow {n_diff_inside}; "
              f"expit(+/-1e4) = {new[x == 1e4][0]:g}, {new[x == -1e4][0]:g}")
        if new.dtype != old.dtype:
            failures.append(f"{dtype.__name__}: expit returned {new.dtype}, old form {old.dtype}")
        if n_diff_outside:
            failures.append(f"{dtype.__name__}: {n_diff_outside} values differ where the old form did not overflow")
        if not overflow[np.abs(x) >= 1000].any() or not old_warnings:
            failures.append(f"{dtype.__name__}: the fixed logits did not exercise the overflow")
        if new_warnings:
            failures.append(f"{dtype.__name__}: expit raised {[str(w.message) for w in new_warnings]}")
        if not (np.all((new >= 0) & (new <= 1)) and np.all(np.diff(new[np.argsort(x, kind="stable")]) >= 0)):
            failures.append(f"{dtype.__name__}: expit is not a monotone map into [0, 1]")
    for line in failures:
        print("  " + line)
    print("PASS" if not failures else "FAIL")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
