"""Check rescore.py: both checkpoint formats, scalar transforms, and TOI selection."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stellaris.model import StellarisNetwork  # noqa: E402
from stellaris.rescore import (  # noqa: E402
    SCALAR_NAMES, fit_scalar_normalisation, load_model, make_scalars, scalar_features, score, select_toi,
)

failures: list[str] = []


def expect(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'ok' if ok else 'FAIL'}] {label}{(': ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


def main() -> int:
    released, spec = load_model(PROJECT_ROOT / "checkpoints" / "stellaris_best.pt", "cpu")
    expect("released format loads as the default architecture",
           spec["variant"] == "released" and not spec["scalars"] and not released.training)
    rng = np.random.default_rng(0)
    g, l = rng.normal(size=2001), rng.normal(size=201)
    with torch.no_grad():
        direct = torch.sigmoid(released(torch.tensor(g).float().view(1, 1, -1), torch.tensor(l).float().view(1, 1, -1)))
    expect("score() equals a direct forward pass", abs(score(released, spec, g, l) - float(direct)) < 1e-7)

    values = {
        "local_scale": np.array([2e-3, 0.0, np.nan]), "global_scale": np.array([1e-3, 1e-9, 5e-3]),
        "period_days": np.array([3.0, 10.0, -1.0]), "duration_days": np.array([0.1, 0.2, 0.15]),
        "srad": np.array([1.0, np.nan, 0.5]), "teff": np.array([5800.0, 4000.0, np.nan]),
        "logg": np.array([4.4, np.nan, 4.6]),
    }
    feats = scalar_features(values)
    expect("log10 of the local scale", np.isclose(feats["log_local_scale"][0], np.log10(2e-3)))
    expect("scales floored at 1e-7", feats["log_local_scale"][1] == -7.0 and feats["log_global_scale"][1] == -7.0)
    expect("missing and non-positive inputs become NaN",
           np.isnan(feats["log_local_scale"][2]) and np.isnan(feats["log_period"][2]) and np.isnan(feats["log_srad"][1]))
    expect("teff and logg pass through untransformed", feats["teff_k"][0] == 5800.0 and feats["logg"][2] == 4.6)

    names = list(SCALAR_NAMES)
    norm = fit_scalar_normalisation(feats, names)
    full_spec = {"scalars": names, **norm}
    matrix = make_scalars(full_spec, values)
    expect("matrix shape (rows, n_scalars)", matrix.shape == (3, len(names)), str(matrix.shape))
    expect("no NaN after imputation", np.isfinite(matrix).all())
    expect("standardised train columns have mean 0", np.allclose(matrix.mean(axis=0), 0, atol=1e-6))
    single = make_scalars(full_spec, {k: v[0] for k, v in values.items()})
    expect("single-target call matches the vectorised row", np.allclose(single, matrix[0]))
    expect("no scalars gives an empty vector", make_scalars({"scalars": []}, {"local_scale": 1e-3}).shape == (0,))

    net = StellarisNetwork(n_scalars=len(names))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "D__seed0.pt"
        torch.save({"model_state_dict": net.state_dict(), "spec": {**full_spec, "variant": "D", "seed": 0,
                    "use_global": True, "use_local": True, "batch_norm": False, "pool": "flatten"},
                    "best_epoch": 1, "val_pr_auc": 0.5}, path)
        loaded, loaded_spec = load_model(path, "cpu")
    expect("train_variants format loads with scalars", loaded.n_scalars == len(names) and loaded_spec["variant"] == "D")
    p = score(loaded, loaded_spec, g, l, {k: v[0] for k, v in values.items()})
    expect("scalar model scores to a probability", 0.0 < p < 1.0, f"{p:.4f}")

    tois = pd.DataFrame({
        "TOI": [100.01, 100.02, 100.03], "Period (days)": [5.0, 10.1, 0.0],
        "Epoch (BJD)": [2459000.5, 2459001.5, 2459002.5], "Duration (hours)": [2.0, 3.0, 1.0],
        "Planet SNR": [50.0, 20.0, 90.0], "Depth (ppm)": [1000.0, 2000.0, 3000.0],
    })
    row, how = select_toi(tois, 10.0)
    expect("BLS period within 2% picks that TOI", how == "bls_match" and row["TOI"] == 100.02)
    row, how = select_toi(tois, 7.0)
    expect("otherwise the highest-SNR usable TOI", how == "max_snr" and row["TOI"] == 100.01)
    row, how = select_toi(tois.iloc[[2]], 7.0)
    expect("only period-0 TOIs are reported as single_transit", row is None and how == "single_transit")
    unusable = tois.iloc[[2]].assign(**{"Period (days)": np.nan})
    expect("other unusable TOIs are reported as none", select_toi(unusable, 7.0) == (None, "none"))

    print("PASS" if not failures else f"FAIL ({len(failures)} checks)")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
