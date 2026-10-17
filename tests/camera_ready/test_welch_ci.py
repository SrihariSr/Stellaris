"""Check analyse_camera_ready.welch_ci: p equals scipy's Welch t-test and the CI contains the difference."""
import sys
from pathlib import Path

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from analyse_camera_ready import welch_ci  # noqa: E402

failures: list[str] = []


def expect(label: str, ok: bool, detail: str = "") -> None:
    if not ok:
        failures.append(f"{label}: {detail}")


def main() -> int:
    rng = np.random.default_rng(0)
    n_cases = 0
    worst = 0.0
    for _ in range(500):
        na, nb = rng.integers(2, 30, size=2)
        a = rng.normal(rng.normal(0, 1), rng.uniform(0.01, 3), na)
        b = rng.normal(rng.normal(0, 1), rng.uniform(0.01, 3), nb)
        r = welch_ci(a, b)
        ref = stats.ttest_ind(a, b, equal_var=False)
        worst = max(worst, abs(r["p"] - ref.pvalue) / max(ref.pvalue, 1e-300))
        expect("p equals scipy", np.isclose(r["p"], ref.pvalue, rtol=1e-10, atol=0), f"{r['p']} vs {ref.pvalue}")
        expect("CI contains the difference", r["ci_low"] <= r["difference"] <= r["ci_high"])
        expect("difference is mean(a) - mean(b)", np.isclose(r["difference"], a.mean() - b.mean(), rtol=0, atol=1e-12))
        lo, hi = ref.confidence_interval(0.95)
        expect("CI equals scipy's Welch interval", np.isclose(r["ci_low"], lo, rtol=1e-10) and np.isclose(r["ci_high"], hi, rtol=1e-10),
               f"[{r['ci_low']}, {r['ci_high']}] vs [{lo}, {hi}]")
        n_cases += 1

    tiny = welch_ci([1.0], [2.0, 3.0])
    expect("fewer than two values gives NaN", np.isnan(tiny["p"]) and np.isnan(tiny["ci_low"]))
    flat = welch_ci([1.0, 1.0, 1.0], [1.0, 1.0])
    expect("zero spread gives a degenerate CI and NaN p", flat["ci_low"] == flat["ci_high"] == 0.0 and np.isnan(flat["p"]))
    nan = welch_ci([1.0, np.nan, 2.0, 3.0], [2.0, 4.0, 6.0])
    ref = stats.ttest_ind([1.0, 2.0, 3.0], [2.0, 4.0, 6.0], equal_var=False)
    expect("NaN values are dropped", np.isclose(nan["p"], ref.pvalue, rtol=1e-10))

    print(f"  {n_cases} random cases; worst relative p difference from scipy {worst:.2e}")
    for line in failures[:10]:
        print("  " + line)
    print("PASS" if not failures else f"FAIL ({len(failures)} checks)")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
