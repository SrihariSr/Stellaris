"""Check the injection geometry (a/R*, T14) against batman and known solar values."""
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stellaris.injection import a_over_rstar, robust_sigma, t14_days, transit_model  # noqa: E402


def main() -> int:
    failures = []
    # Earth around the Sun: a/R* = 215.0; a 1-day orbit: about 4.2.
    earth = a_over_rstar(365.25)
    if abs(earth - 215.0) > 1.0:
        failures.append(f"a/R* for 365.25 d = {earth:.1f}, expected ~215")
    if abs(a_over_rstar(1.0) - 4.2) > 0.05:
        failures.append(f"a/R* for 1 d = {a_over_rstar(1.0):.3f}, expected ~4.2")

    rng = np.random.default_rng(1)
    for _ in range(20):
        period = float(np.exp(rng.uniform(np.log(0.5), np.log(20.0))))
        depth = float(np.exp(rng.uniform(np.log(5e-4), np.log(3e-2))))
        b = float(rng.uniform(0.0, 0.9))
        rp = np.sqrt(depth)
        a_rs = a_over_rstar(period)
        t14 = t14_days(period, a_rs, rp, b)
        step = t14 / 4000
        time = np.arange(-t14, t14, step)
        model = transit_model(time, period, 0.0, rp, a_rs, b)
        in_transit = time[model < 1.0]
        measured = in_transit.max() - in_transit.min() + step
        if abs(measured - t14) > 3 * step:
            failures.append(f"P={period:.2f} b={b:.2f}: batman width {measured:.5f} d vs T14 {t14:.5f} d")
        realised = 1.0 - model.min()
        if not 0.5 * depth < realised < 1.6 * depth:
            failures.append(f"realised depth {realised:.2e} far from rp^2 {depth:.2e}")

    nan_model = transit_model(np.array([np.nan, 0.0]), 3.0, 0.0, 0.1, a_over_rstar(3.0), 0.3)
    if nan_model[0] != 1.0 or not nan_model[1] < 1.0:
        failures.append("non-finite times should get model 1")
    if abs(robust_sigma(rng.normal(0, 2e-3, 200_000)) - 2e-3) > 2e-5:
        failures.append("robust_sigma does not recover a Gaussian sigma")

    for line in failures:
        print("  " + line)
    print("PASS" if not failures else "FAIL")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
