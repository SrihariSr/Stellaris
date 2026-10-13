"""Shared pieces for rescoring frozen or retrained checkpoints: views, model loading, scalars, TOI lookup."""
from pathlib import Path
import glob

import numpy as np
import pandas as pd
import torch

from stellaris.model import StellarisNetwork
from stellaris.views import make_global_view, make_local_view

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOI_TABLE = PROJECT_ROOT / "data" / "catalogs" / "tess_toi.csv"
BTJD_OFFSET = 2457000.0

SCALAR_NAMES = [
    "log_local_scale", "log_global_scale", "log_period", "log_duration",
    "log_srad", "teff_k", "logg",
]
SCALE_FLOOR = 1e-7

RELEASED_SPEC = {
    "group": "released",
    "variant": "released",
    "seed": None,
    "use_global": True,
    "use_local": True,
    "n_scalars": 0,
    "scalars": [],
    "batch_norm": False,
    "pool": "flatten",
}

CATALOGUE_DISPOSITIONS = {
    "r1": {"CP", "KP"},
    "r2": {"PC", "APC"},
    "r0": {"FP", "FA"},
}
PERIOD_MATCH_TOL = 0.02


def build_views(
    time: np.ndarray,
    flux: np.ndarray,
    period_days: float,
    epoch_btjd: float,
    duration_days: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Returns (global_view, local_view, global_scale, local_scale)."""
    global_view, global_scale = make_global_view(time, flux, period_days, epoch_btjd, return_scale=True)
    local_view, local_scale = make_local_view(
        time, flux, period_days, epoch_btjd, duration_days, return_scale=True
    )
    return global_view, local_view, global_scale, local_scale


def load_model(path: Path, device: str = "cpu") -> tuple[StellarisNetwork, dict]:
    """
    Load either the released format ({"model_state_dict", "epoch", ...}, default
    architecture) or a train_variants checkpoint ({"model_state_dict", "spec", ...}).
    """
    ckpt = torch.load(path, weights_only=False, map_location=device)
    spec = dict(ckpt["spec"]) if "spec" in ckpt else dict(RELEASED_SPEC)
    model = StellarisNetwork(
        use_global=spec.get("use_global", True),
        use_local=spec.get("use_local", True),
        n_scalars=len(spec.get("scalars", [])),
        batch_norm=spec.get("batch_norm", False),
        pool=spec.get("pool", "flatten"),
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, spec


def expand_checkpoints(patterns: list[str]) -> list[Path]:
    """Paths or globs, de-duplicated in order; a glob that matches nothing is an error."""
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern)) if any(c in pattern for c in "*?[") else [pattern]
        if not matches:
            raise FileNotFoundError(f"No checkpoints match {pattern}")
        paths += [Path(m) for m in matches]
    return list(dict.fromkeys(paths))


def checkpoint_label(path: Path) -> str:
    """Repo-relative path when inside the project, else absolute."""
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def load_models(patterns: list[str], device: str = "cpu") -> list[tuple[str, StellarisNetwork, dict]]:
    return [(checkpoint_label(path), *load_model(path, device)) for path in expand_checkpoints(patterns)]


def _positive_log10(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.isfinite(x) & (x > 0), np.log10(np.where(x > 0, x, 1.0)), np.nan)


def scalar_features(values: dict) -> dict[str, np.ndarray]:
    """
    Transform raw values into the named scalar features, NaN where missing.

    values holds local_scale, global_scale, period_days, duration_days, srad,
    teff and logg, as floats or equal-length arrays. Missing keys count as missing.
    """
    def get(key):
        return np.asarray(values.get(key, np.nan), dtype=np.float64)

    def floored_log(key):
        x = get(key)
        return np.where(np.isfinite(x), np.log10(np.maximum(x, SCALE_FLOOR)), np.nan)

    return {
        "log_local_scale": floored_log("local_scale"),
        "log_global_scale": floored_log("global_scale"),
        "log_period": _positive_log10(get("period_days")),
        "log_duration": _positive_log10(get("duration_days")),
        "log_srad": _positive_log10(get("srad")),
        "teff_k": np.where(np.isfinite(get("teff")), get("teff"), np.nan),
        "logg": np.where(np.isfinite(get("logg")), get("logg"), np.nan),
    }


def fit_scalar_normalisation(features: dict[str, np.ndarray], names: list[str]) -> dict:
    """Median imputation, then mean and std, all from the rows given (the train split)."""
    impute, mean, std = {}, {}, {}
    for name in names:
        x = np.asarray(features[name], dtype=np.float64)
        finite = np.isfinite(x)
        impute[name] = float(np.median(x[finite])) if finite.any() else 0.0
        filled = np.where(finite, x, impute[name])
        mean[name] = float(filled.mean())
        sd = float(filled.std())
        std[name] = sd if sd > 0 else 1.0
    return {"scalar_impute": impute, "scalar_mean": mean, "scalar_std": std}


def make_scalars(spec: dict, values: dict) -> np.ndarray:
    """Scalar vector (or matrix, for array values) for spec["scalars"], imputed and standardised."""
    names = list(spec.get("scalars", []))
    features = scalar_features(values)
    shape = np.broadcast(*features.values()).shape
    if not names:
        return np.zeros(shape + (0,), dtype=np.float32)
    columns = []
    for name in names:
        x = np.broadcast_to(features[name], shape)
        x = np.where(np.isfinite(x), x, spec["scalar_impute"][name])
        columns.append((x - spec["scalar_mean"][name]) / spec["scalar_std"][name])
    return np.stack(columns, axis=-1).astype(np.float32)


def score(
    model: StellarisNetwork,
    spec: dict,
    global_view: np.ndarray,
    local_view: np.ndarray,
    values: dict | None = None,
    device: str = "cpu",
) -> float:
    g = torch.from_numpy(np.asarray(global_view)).float().unsqueeze(0).unsqueeze(0).to(device)
    l = torch.from_numpy(np.asarray(local_view)).float().unsqueeze(0).unsqueeze(0).to(device)
    scalars = None
    if spec.get("scalars"):
        scalars = torch.from_numpy(make_scalars(spec, values or {})).reshape(1, -1).to(device)
    with torch.no_grad():
        logit = model(g, l, scalars) if scalars is not None else model(g, l)
        return float(torch.sigmoid(logit).item())


def load_toi_table(path: Path = TOI_TABLE) -> pd.DataFrame:
    toi = pd.read_csv(path, comment="#")
    toi["TIC ID"] = toi["TIC ID"].astype(int)
    return toi


def select_toi(candidates: pd.DataFrame, bls_period: float | None) -> tuple[pd.Series | None, str]:
    """
    Pick the TOI whose period is within 2% of the BLS period (the closest if
    several), else the one with the highest Planet SNR. Returns (row, selection).
    With no usable TOI the selection is "single_transit" when every candidate
    has period <= 0 (the catalogue's single-transit marker), else "none".
    """
    usable = candidates[
        np.isfinite(candidates["Period (days)"]) & (candidates["Period (days)"] > 0)
        & np.isfinite(candidates["Epoch (BJD)"])
        & np.isfinite(candidates["Duration (hours)"]) & (candidates["Duration (hours)"] > 0)
    ]
    if usable.empty:
        single = len(candidates) > 0 and bool((candidates["Period (days)"] <= 0).all())
        return None, "single_transit" if single else "none"
    if bls_period is not None and np.isfinite(bls_period):
        frac = (bls_period / usable["Period (days)"] - 1).abs()
        if (frac <= PERIOD_MATCH_TOL).any():
            return usable.loc[frac[frac <= PERIOD_MATCH_TOL].idxmin()], "bls_match"
    snr = usable["Planet SNR"].astype(float).fillna(-np.inf)
    return usable.loc[snr.idxmax()], "max_snr"


def catalogue_ephemeris(row: pd.Series) -> dict:
    """period_days, epoch_btjd, duration_days and depth_ppm from one TOI row."""
    return {
        "toi": str(row["TOI"]),
        "period_days": float(row["Period (days)"]),
        "epoch_btjd": float(row["Epoch (BJD)"]) - BTJD_OFFSET,
        "duration_days": float(row["Duration (hours)"]) / 24.0,
        "depth_ppm": float(row["Depth (ppm)"]),
    }


def stellar_from_toi(toi: pd.DataFrame) -> pd.DataFrame:
    """First non-null stellar radius, Teff and log g per TIC, indexed by tic_id."""
    cols = {
        "Stellar Radius (R_Sun)": "srad",
        "Stellar Eff Temp (K)": "teff",
        "Stellar log(g) (cm/s^2)": "logg",
    }
    table = toi.groupby("TIC ID")[list(cols)].first().rename(columns=cols)
    table.index.name = "tic_id"
    return table


def load_stellar_table(path: Path) -> pd.DataFrame:
    """A CSV with tic_id, srad, teff, logg (extra columns kept), indexed by tic_id."""
    table = pd.read_csv(path, comment="#")
    table["tic_id"] = table["tic_id"].astype(int)
    return table.drop_duplicates("tic_id", keep="last").set_index("tic_id")
