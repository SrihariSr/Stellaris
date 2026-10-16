"""Test-time branch knockout: score the Kepler test split with one view zeroed or swapped between examples."""
from pathlib import Path
import argparse
import csv

import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from stellaris.dataset import DEFAULT_DATASET_PATH, StellarisDataset, make_splits
from stellaris.rescore import checkpoint_label, expand_checkpoints, load_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINTS = [
    "checkpoints/stellaris_best.pt",
    "checkpoints/camera_ready/ablation/both__seed*.pt",
    "checkpoints/camera_ready/bn-ablation/both__seed*.pt",
]
DEFAULT_OUT = PROJECT_ROOT / "results" / "camera_ready" / "analysis" / "branch_knockout.csv"
RELEASED = "checkpoints/stellaris_best.pt"
PREDICTIONS = PROJECT_ROOT / "results" / "test_predictions.npz"
CONDITIONS = ["intact", "global_zero", "global_shuffle", "local_zero", "local_shuffle"]
COLUMNS = [
    "checkpoint", "group", "variant", "seed", "condition", "n_test", "pr_auc", "roc_auc",
    "pr_auc_drop", "roc_auc_drop", "mean_abs_dlogit", "max_abs_dlogit", "frac_cross_05", "permutation_seed",
]


def resolve_checkpoints(patterns: list[str]) -> list[Path]:
    """Relative patterns are read from the repository root; a default glob with no match is skipped."""
    paths: list[Path] = []
    for pattern in patterns:
        full = pattern if Path(pattern).is_absolute() else str(PROJECT_ROOT / pattern)
        try:
            paths += expand_checkpoints([full])
        except FileNotFoundError:
            if pattern not in DEFAULT_CHECKPOINTS:
                raise
            print(f"[warn] no checkpoints match the default {pattern}; skipped")
    return list(dict.fromkeys(paths))


def derangement(n: int, seed: int = 0) -> np.ndarray:
    """A permutation with no fixed points: redraw until none remain (about e draws on average)."""
    rng = np.random.default_rng(seed)
    while True:
        perm = rng.permutation(n)
        if n < 2 or not np.any(perm == np.arange(n)):
            return perm


def condition_views(global_views: np.ndarray, local_views: np.ndarray, condition: str,
                    perm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Views for one condition; arrays are (n, bins) and the untouched view is passed through as is."""
    if condition == "intact":
        return global_views, local_views
    branch, action = condition.split("_")
    target = global_views if branch == "global" else local_views
    replaced = np.zeros_like(target) if action == "zero" else target[perm]
    return (replaced, local_views) if branch == "global" else (global_views, replaced)


def score_logits(model: torch.nn.Module, global_views: np.ndarray, local_views: np.ndarray,
                 device: str, batch_size: int) -> np.ndarray:
    out = []
    with torch.no_grad():
        for start in range(0, len(global_views), batch_size):
            g = torch.from_numpy(np.ascontiguousarray(global_views[start:start + batch_size])).unsqueeze(1).to(device)
            l = torch.from_numpy(np.ascontiguousarray(local_views[start:start + batch_size])).unsqueeze(1).to(device)
            out.append(model(g, l).cpu().numpy())
    return np.concatenate(out)


def knockout_rows(model, spec: dict, label: str, global_views: np.ndarray, local_views: np.ndarray,
                  labels: np.ndarray, perm: np.ndarray, perm_seed: int, device: str, batch_size: int) -> tuple[list[dict], np.ndarray]:
    """One row per condition, plus the intact probabilities."""
    rows, intact_logits, intact_probs, intact = [], None, None, {}
    for condition in CONDITIONS:
        g, l = condition_views(global_views, local_views, condition, perm)
        logits = score_logits(model, g, l, device, batch_size)
        probs = expit(logits)
        if condition == "intact":
            intact_logits, intact_probs = logits, probs
        pr_auc = float(average_precision_score(labels, probs))
        roc_auc = float(roc_auc_score(labels, probs))
        if condition == "intact":
            intact = {"pr_auc": pr_auc, "roc_auc": roc_auc}
        dlogit = np.abs(logits.astype(np.float64) - intact_logits.astype(np.float64))
        rows.append({
            "checkpoint": label, "group": spec.get("group", ""), "variant": spec.get("variant", ""),
            "seed": "" if spec.get("seed") is None else spec["seed"], "condition": condition,
            "n_test": len(labels), "pr_auc": pr_auc, "roc_auc": roc_auc,
            "pr_auc_drop": intact["pr_auc"] - pr_auc, "roc_auc_drop": intact["roc_auc"] - roc_auc,
            "mean_abs_dlogit": float(dlogit.mean()), "max_abs_dlogit": float(dlogit.max()),
            "frac_cross_05": float(np.mean((probs >= 0.5) != (intact_probs >= 0.5))),
            "permutation_seed": perm_seed,
        })
    return rows, intact_probs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", default=DEFAULT_CHECKPOINTS, help="Paths or quoted globs.")
    parser.add_argument("--h5", type=Path, default=DEFAULT_DATASET_PATH, help="Default: the v1 training set.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--permutation-seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="Use at most N checkpoints.")
    args = parser.parse_args()

    paths = resolve_checkpoints(args.checkpoints)
    if args.limit is not None:
        paths = paths[: args.limit]
    labels_of = [checkpoint_label(p) for p in paths]

    splits = make_splits(hdf5_path=args.h5)
    ds = StellarisDataset(args.h5, splits["test"])
    global_views, local_views = ds.global_views[ds.indices], ds.local_views[ds.indices]
    labels = ds.labels[ds.indices]
    perm = derangement(len(labels), args.permutation_seed)
    print(f"Test split: {len(labels)} examples ({int(labels.sum())} positive) from {args.h5}")
    print(f"Swap permutation: seed {args.permutation_seed}, fixed points {int(np.sum(perm == np.arange(len(perm))))}; "
          f"swapped pairs with the same label: {np.mean(labels[perm] == labels):.3f}")

    done: set[str] = set()
    if args.out.exists():
        previous = pd.read_csv(args.out)
        done = {c for c, n in previous.groupby("checkpoint").condition.nunique().items() if n == len(CONDITIONS)}
    todo = [(p, lab) for p, lab in zip(paths, labels_of) if lab not in done]
    print(f"{len(paths)} checkpoint(s); {len(paths) - len(todo)} already in {args.out}")

    status = 0
    if todo:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_header = not args.out.exists()
        with open(args.out, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=COLUMNS)
            if write_header:
                writer.writeheader()
            for path, label in tqdm(todo, desc="Knockout"):
                model, spec = load_model(path, args.device)
                if spec.get("scalars"):
                    print(f"[skip] {label}: scalar-input checkpoints need the v2 scalars; not supported here")
                    continue
                rows, intact_probs = knockout_rows(model, spec, label, global_views, local_views, labels, perm,
                                                   args.permutation_seed, args.device, args.batch_size)
                writer.writerows(rows)
                fh.flush()
                if label == RELEASED and Path(args.h5).resolve() == DEFAULT_DATASET_PATH.resolve() and PREDICTIONS.exists():
                    ref = np.load(PREDICTIONS)
                    max_diff = float(np.max(np.abs(intact_probs - ref["probs"])))
                    same = np.array_equal(labels, ref["labels"]) and max_diff <= 1e-5
                    print(f"[check] released checkpoint, intact: PR-AUC {rows[0]['pr_auc']:.4f}, labels match "
                          f"test_predictions.npz: {np.array_equal(labels, ref['labels'])}, max |dprob| {max_diff:.2e}"
                          + ("" if same else "  <-- MISMATCH"))
                    status = 0 if same else 1

    frame = pd.read_csv(args.out)
    frame = frame[frame.checkpoint.isin(labels_of)].drop_duplicates(["checkpoint", "condition"], keep="last")
    print("\nPR-AUC drop (intact minus condition), mean +/- s.d. over seeds:")
    for group, sub in frame.groupby("group", sort=False):
        intact = sub[sub.condition == "intact"].pr_auc
        print(f"  [{group}] n={sub.checkpoint.nunique()}  intact PR-AUC {intact.mean():.4f}"
              + (f" +/- {intact.std(ddof=1):.4f}" if len(intact) > 1 else ""))
        for condition in CONDITIONS[1:]:
            drop = sub[sub.condition == condition]
            sd = f" +/- {drop.pr_auc_drop.std(ddof=1):.4f}" if len(drop) > 1 else ""
            print(f"    {condition:<15} drop {drop.pr_auc_drop.mean():+.4f}{sd}   "
                  f"mean |dlogit| {drop.mean_abs_dlogit.mean():.3f}   crossing 0.5 {drop.frac_cross_05.mean():.3f}")
    print(f"\nWritten: {args.out}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
