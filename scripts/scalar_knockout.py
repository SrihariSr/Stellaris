"""Scalar knockout: does a scalar-input checkpoint use its scalars? Kepler permutation test and a TESS depth swap."""
from pathlib import Path
import argparse
import glob
import os
import sys

import h5py
import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_variants import h5_scalar_values  # noqa: E402

from stellaris.dataset import StellarisDataset, make_splits  # noqa: E402
from stellaris.rescore import build_views, checkpoint_label, load_model, make_scalars, scalar_features  # noqa: E402
from stellaris.tess_cache import DEFAULT_CACHE_DIR, SF_DATALESS, cache_path, load_cache  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = PROJECT_ROOT / "checkpoints" / "camera_ready" / "scalars"
RESCORE = PROJECT_ROOT / "results" / "camera_ready" / "rescore" / "r2_bls_full.csv"
PER_CHECKPOINT = PROJECT_ROOT / "results" / "camera_ready" / "analysis" / "rescore_per_checkpoint_bls_full.csv"
DEFAULT_OUT = PROJECT_ROOT / "results" / "camera_ready" / "analysis"
DEPTH_SCALAR = "log_local_scale"


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def branch_features(model, g: torch.Tensor, l: torch.Tensor, batch: int) -> torch.Tensor:
    """The concatenated branch outputs that StellarisNetwork.forward feeds to the head, before the scalars."""
    parts = []
    with torch.no_grad():
        for s in range(0, len(g), batch):
            feats = []
            if model.use_global:
                feats.append(model.global_branch(g[s:s + batch]))
            if model.use_local:
                feats.append(model.local_branch(l[s:s + batch]))
            parts.append(torch.cat(feats, dim=1))
    return torch.cat(parts)


def head_logits(model, feats: torch.Tensor, scalars: np.ndarray, batch: int) -> np.ndarray:
    s = torch.from_numpy(np.ascontiguousarray(scalars, dtype=np.float32))
    with torch.no_grad():
        return np.concatenate([model.head(torch.cat([feats[i:i + batch], s[i:i + batch]], dim=1)).squeeze(-1).numpy()
                               for i in range(0, len(feats), batch)])


def kepler_part(paths: list[Path], n_perm: int, batch: int) -> list[dict]:
    rows, loaded = [], {}
    for path in paths:
        model, spec = load_model(path, "cpu")
        h5 = resolve(spec["h5"])
        if os.stat(h5).st_flags & SF_DATALESS:
            raise RuntimeError(f"{h5} is a cloud-only placeholder; not opening it")
        if h5 not in loaded:
            test = make_splits(hdf5_path=h5)["test"]
            ds = StellarisDataset(h5, test)
            loaded[h5] = (test, torch.from_numpy(ds.global_views[test]).unsqueeze(1),
                          torch.from_numpy(ds.local_views[test]).unsqueeze(1), ds.labels[test], h5_scalar_values(h5))
        test, g, l, labels, values = loaded[h5]
        scalars = make_scalars(spec, values)[test]

        feats = branch_features(model, g, l, batch)
        intact = head_logits(model, feats, scalars, batch)
        with torch.no_grad():
            full = np.concatenate([model(g[i:i + batch], l[i:i + batch],
                                         torch.from_numpy(scalars[i:i + batch])).numpy()
                                   for i in range(0, len(g), batch)])
        if not np.array_equal(intact, full):
            raise RuntimeError(f"{path}: cached-feature head differs from the full forward pass "
                               f"(max |d| {np.abs(intact - full).max():.2e})")
        pr_intact = float(average_precision_score(labels, expit(intact)))
        for j, name in enumerate(spec["scalars"]):
            permuted = []
            for p in range(n_perm):
                perm = np.random.default_rng(p).permutation(len(scalars))
                shuffled = scalars.copy()
                shuffled[:, j] = scalars[perm, j]
                permuted.append(float(average_precision_score(labels, expit(head_logits(model, feats, shuffled, batch)))))
            rows.append({"part": "kepler_permutation", "checkpoint": checkpoint_label(path), "variant": spec["variant"],
                         "seed": spec["seed"], "scalar": name, "n": len(labels), "pr_auc_intact": pr_intact,
                         "pr_auc_permuted_mean": float(np.mean(permuted)), "pr_auc_drop": pr_intact - float(np.mean(permuted))})
    return rows


def training_positive_median(h5: Path) -> float:
    """Median of the depth scalar (before standardisation) over the training split's positives."""
    train = make_splits(hdf5_path=h5)["train"]
    with h5py.File(h5, "r") as f:
        positive = f["label"][:][train] == 1
    return float(np.median(scalar_features(h5_scalar_values(h5))[DEPTH_SCALAR][train][positive]))


def tess_part(paths: list[Path], rescore: pd.DataFrame, per_ckpt: pd.DataFrame, cache_dir: Path) -> list[dict]:
    rows, views, medians = [], {}, {}
    for path in paths:
        model, spec = load_model(path, "cpu")
        label = checkpoint_label(path)
        h5 = resolve(spec["h5"])
        if h5 not in medians:
            medians[h5] = training_positive_median(h5)
        median_depth = medians[h5]
        column = spec["scalars"].index(DEPTH_SCALAR)
        replaced_value = (median_depth - spec["scalar_mean"][DEPTH_SCALAR]) / spec["scalar_std"][DEPTH_SCALAR]

        sub = rescore[rescore.checkpoint == label]
        cap = sub[sub.vetting_reasons.fillna("").str.contains("cap")]
        t_star = float(per_ckpt.loc[per_ckpt.checkpoint == label, "t_star"].iloc[0])
        real, swapped, depth_real = [], [], []
        for r in cap.itertuples():
            if r.tic_id not in views:
                npz = cache_path(cache_dir, int(r.tic_id))
                if os.stat(npz).st_flags & SF_DATALESS:
                    raise RuntimeError(f"{npz} is a cloud-only placeholder; not opening it")
                data = load_cache(npz)
                views[r.tic_id] = build_views(data["time"], data["flux"], float(r.period_days), float(r.epoch_btjd),
                                              float(r.duration_hours) / 24.0)
            gv, lv, gs, ls = views[r.tic_id]
            vals = {"local_scale": ls, "global_scale": gs, "period_days": float(r.period_days),
                    "duration_days": float(r.duration_hours) / 24.0}
            s_real = make_scalars(spec, vals).reshape(1, -1)
            s_swap = s_real.copy()
            s_swap[0, column] = replaced_value
            g = torch.from_numpy(np.asarray(gv)).float().view(1, 1, -1)
            l = torch.from_numpy(np.asarray(lv)).float().view(1, 1, -1)
            with torch.no_grad():
                real.append(float(torch.sigmoid(model(g, l, torch.from_numpy(s_real))).item()))
                swapped.append(float(torch.sigmoid(model(g, l, torch.from_numpy(s_swap.astype(np.float32)))).item()))
            depth_real.append(float(scalar_features(vals)[DEPTH_SCALAR]))
        real, swapped = np.array(real), np.array(swapped)
        stored = cap.prob.to_numpy(dtype=float)
        stored_high = per_ckpt.loc[per_ckpt.checkpoint == label, "r2_high_among_depth_cap"]
        rows.append({
            "part": "tess_depth_swap", "checkpoint": label, "variant": spec["variant"], "seed": spec["seed"],
            "scalar": DEPTH_SCALAR, "n": len(cap), "t_star": t_star,
            "depth_scalar_real_median": float(np.median(depth_real)), "depth_scalar_replacement": median_depth,
            "mean_score_real": float(real.mean()), "mean_score_replaced": float(swapped.mean()),
            "mean_score_change": float((swapped - real).mean()),
            "n_ge_tstar_real": int((real >= t_star).sum()), "n_ge_tstar_replaced": int((swapped >= t_star).sum()),
            "max_abs_diff_vs_rescore": float(np.abs(real - stored).max()),
            "n_ge_tstar_matches_rescore_table": bool(len(stored_high) and int(stored_high.iloc[0]) == int((real >= t_star).sum())),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=CKPT_DIR)
    parser.add_argument("--variants", nargs="+", default=["B", "C", "D"], help="Variants for the Kepler test.")
    parser.add_argument("--tess-variant", default="B")
    parser.add_argument("--n-perm", type=int, default=5)
    parser.add_argument("--rescore", type=Path, default=RESCORE, help="R2 rescore CSV (rescore_tess.py, bls_full).")
    parser.add_argument("--per-checkpoint", type=Path, default=PER_CHECKPOINT,
                        help="analyse_camera_ready.py rescore table giving each checkpoint's t*.")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None, help="At most N seeds per variant.")
    args = parser.parse_args()
    torch.set_grad_enabled(False)

    def checkpoints(variant: str) -> list[Path]:
        paths = sorted(Path(p) for p in glob.glob(str(args.checkpoint_dir / f"{variant}__seed*.pt")))
        paths.sort(key=lambda p: int(p.stem.split("seed")[1]))
        return paths[: args.limit] if args.limit else paths

    rows = kepler_part([p for v in args.variants for p in checkpoints(v)], args.n_perm, args.batch_size)
    rescore = pd.read_csv(args.rescore, low_memory=False)
    rescore = rescore[rescore.status == "ok"].drop_duplicates(["tic_id", "checkpoint"], keep="last")
    rows += tess_part(checkpoints(args.tess_variant), rescore, pd.read_csv(args.per_checkpoint), args.cache_dir)
    table = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "scalar_knockout.csv"
    table.to_csv(out, index=False)

    kep = table[table.part == "kepler_permutation"]
    print(f"(a) Kepler test split: PR-AUC drop when one scalar column is permuted ({args.n_perm} permutations), "
          "mean +/- s.d. over seeds")
    for (variant, scalar), sub in kep.groupby(["variant", "scalar"], sort=False):
        print(f"  {variant}  {scalar:<16} drop {sub.pr_auc_drop.mean():+.4f} +/- {sub.pr_auc_drop.std(ddof=1):.4f}"
              f"   (intact PR-AUC {sub.pr_auc_intact.mean():.4f}, n seeds {len(sub)})")
    tess = table[table.part == "tess_depth_swap"]
    if len(tess):
        print(f"(b) TESS R2 depth-cap rejections (n={int(tess.n.iloc[0])}), {args.tess_variant} checkpoints, "
              f"{DEPTH_SCALAR} set to the Kepler training-positive median ({tess.depth_scalar_replacement.iloc[0]:.3f}; "
              f"real median {tess.depth_scalar_real_median.mean():.3f}), mean +/- s.d. over {len(tess)} seeds")
        print(f"  mean score change          {tess.mean_score_change.mean():+.4f} +/- {tess.mean_score_change.std(ddof=1):.4f}")
        for col, name in (("n_ge_tstar_real", "n >= t*, real depth"), ("n_ge_tstar_replaced", "n >= t*, replaced depth")):
            print(f"  {name:<26} {tess[col].mean():.1f} +/- {tess[col].std(ddof=1):.1f}")
        print(f"  checks: real-depth scores vs rescore CSV max |d| {tess.max_abs_diff_vs_rescore.max():.2e}; "
              f"n >= t* (real) equals the rescore table for {int(tess.n_ge_tstar_matches_rescore_table.sum())}/{len(tess)}")
    print(f"Written: {out}")


if __name__ == "__main__":
    main()
