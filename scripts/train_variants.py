"""Train camera-ready model variants (ablation, longer ablation, global-branch tuning, scalar inputs)."""
from pathlib import Path
import argparse
import csv
import datetime as dt
import json
import os

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
)
from scipy.special import expit
from tqdm import tqdm

from stellaris.dataset import StellarisDataset, make_splits, DEFAULT_DATASET_PATH
from stellaris.model import StellarisNetwork
from stellaris.rescore import fit_scalar_normalisation, make_scalars, scalar_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]
V2_PATH = PROJECT_ROOT / "data" / "processed" / "kepler_trainset_v2.h5"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "camera_ready" / "training"
DEFAULT_CKPT_ROOT = PROJECT_ROOT / "checkpoints" / "camera_ready"

BATCH_SIZE = 64
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
COLLAPSE_LOGIT_STD = 1e-3

SCALAR_VARIANTS = {
    "A": [],
    "B": ["log_local_scale"],
    "C": ["log_local_scale", "log_period", "log_duration"],
    "D": ["log_local_scale", "log_period", "log_duration", "log_srad", "teff_k", "logg"],
}
LR_TAGS = {1e-5: "1e-5", 1e-4: "1e-4", 3e-4: "3e-4"}


def default_device() -> str:
    return (
        'mps' if torch.backends.mps.is_available()
        else 'cuda' if torch.cuda.is_available()
        else 'cpu'
    )


def group_variants(group: str) -> dict[str, dict]:
    """Variant name -> architecture and optimiser settings, without seed or epochs."""
    base = {"use_global": True, "use_local": True, "batch_norm": False, "pool": "flatten",
            "scalars": [], "lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE}
    if group in ("ablation", "ablation-long", "bn-ablation"):
        if group == "bn-ablation":
            base = dict(base, batch_norm=True)
        return {
            "both": dict(base),
            "global-only": dict(base, use_local=False),
            "local-only": dict(base, use_global=False),
        }
    if group == "global-tuning":
        return {
            f"lr{LR_TAGS[lr]}_bn{int(bn)}_{pool}": dict(base, use_local=False, lr=lr, batch_norm=bn, pool=pool)
            for lr in (1e-5, 1e-4, 3e-4) for bn in (False, True) for pool in ("flatten", "gap")
        }
    if group == "scalars":
        return {name: dict(base, scalars=list(names)) for name, names in SCALAR_VARIANTS.items()}
    raise ValueError(f"Unknown group {group}")


GROUP_DEFAULTS = {
    "ablation": {"epochs": 40, "seeds": list(range(10)), "h5": DEFAULT_DATASET_PATH},
    "ablation-long": {"epochs": 100, "seeds": list(range(10)), "h5": DEFAULT_DATASET_PATH},
    "bn-ablation": {"epochs": 100, "seeds": list(range(10)), "h5": DEFAULT_DATASET_PATH},
    "global-tuning": {"epochs": 100, "seeds": list(range(3)), "h5": DEFAULT_DATASET_PATH},
    "scalars": {"epochs": 40, "seeds": list(range(10)), "h5": V2_PATH},
}


class VariantDataset(StellarisDataset):
    """StellarisDataset plus an optional per-row scalar matrix indexed like the HDF5 file."""

    def __init__(self, hdf5_path: Path, indices: np.ndarray, scalars: np.ndarray | None = None):
        super().__init__(hdf5_path, indices)
        self.scalars = None if scalars is None else np.asarray(scalars, dtype=np.float32)

    def __getitem__(self, idx: int) -> dict:
        item = super().__getitem__(idx)
        if self.scalars is not None:
            item['scalars'] = torch.from_numpy(self.scalars[self.indices[idx]])
        return item


def h5_scalar_values(h5_path: Path) -> dict[str, np.ndarray]:
    with h5py.File(h5_path, 'r') as f:
        missing = [k for k in ("local_scale", "global_scale", "period_days", "duration_hours",
                               "srad", "steff", "slogg") if k not in f]
        if missing:
            raise KeyError(f"{h5_path} lacks {missing}; scalar variants need the v2 training set")
        return {
            "local_scale": f["local_scale"][:],
            "global_scale": f["global_scale"][:],
            "period_days": f["period_days"][:],
            "duration_days": f["duration_hours"][:] / 24.0,
            "srad": f["srad"][:],
            "teff": f["steff"][:],
            "logg": f["slogg"][:],
        }


def forward(model: nn.Module, batch: dict, device: str) -> torch.Tensor:
    gv = batch['global_view'].to(device)
    lv = batch['local_view'].to(device)
    if 'scalars' in batch:
        return model(gv, lv, batch['scalars'].to(device))
    return model(gv, lv)


def evaluate(model: nn.Module, loader: DataLoader, device: str, max_batches: int | None = None) -> dict:
    """run_ablation.evaluate plus the logit spread; NaN metrics when only one class is present."""
    model.eval()
    all_logits, all_labels = [], []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            logits = forward(model, batch, device)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(batch['label'].numpy())

    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    probs = expit(logits)

    out = {'logit_std': float(np.std(logits))}
    if len(np.unique(labels)) < 2:
        return {'pr_auc': float('nan'), 'roc_auc': float('nan'), 'recall_at_99': float('nan'),
                'recall_at_95': float('nan'), 'recall_at_90': float('nan'), **out}
    out = {
        'pr_auc': float(average_precision_score(labels, probs)),
        'roc_auc': float(roc_auc_score(labels, probs)),
        **out,
    }
    precision, recall, _ = precision_recall_curve(labels, probs)
    for p in (0.99, 0.95, 0.90):
        mask = precision >= p
        out[f'recall_at_{int(p * 100)}'] = float(recall[mask].max()) if mask.any() else 0.0
    return out


def train_one(
    spec: dict,
    splits: dict,
    h5_path: Path,
    device: str,
    max_batches: int | None = None,
    scalar_matrix: np.ndarray | None = None,
) -> dict:
    """
    Mirrors run_ablation.train_one: seed, then datasets and loaders, then model,
    then optimiser; best validation PR-AUC (strict >) picks the state that is
    scored once on the test set. Per-epoch history is recorded on the side.
    """
    torch.manual_seed(spec["seed"])
    np.random.seed(spec["seed"])

    train_ds = VariantDataset(h5_path, splits['train'], scalar_matrix)
    val_ds   = VariantDataset(h5_path, splits['val'], scalar_matrix)
    test_ds  = VariantDataset(h5_path, splits['test'], scalar_matrix)

    batch_size = spec["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device != 'mps'))
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=(device != 'mps'))
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=(device != 'mps'))

    n_pos = float(train_ds.labels[splits['train']].sum())
    n_neg = float(len(splits['train']) - n_pos)
    pos_weight = torch.tensor(n_neg / n_pos, device=device)

    model = StellarisNetwork(
        use_global=spec["use_global"], use_local=spec["use_local"],
        n_scalars=len(spec["scalars"]), batch_norm=spec["batch_norm"], pool=spec["pool"],
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimiser = torch.optim.Adam(model.parameters(), lr=spec["lr"], weight_decay=spec["weight_decay"])

    n_params = sum(p.numel() for p in model.parameters())

    best_val_pr = -1.0
    best_state = None
    best_epoch = -1
    best_val: dict = {}
    history = []
    epochs = spec["epochs"]

    for epoch in range(1, epochs + 1):
        model.train()
        loss_total, n_seen = 0.0, 0
        for i, batch in enumerate(tqdm(train_loader, desc=f"  epoch {epoch}/{epochs}", leave=False)):
            if max_batches is not None and i >= max_batches:
                break
            labels = batch['label'].to(device)

            optimiser.zero_grad()
            loss = criterion(forward(model, batch, device), labels)
            loss.backward()
            optimiser.step()
            loss_total += loss.item() * labels.size(0)
            n_seen += labels.size(0)

        val = evaluate(model, val_loader, device, max_batches)
        history.append({
            'epoch': epoch,
            'train_loss': loss_total / max(1, n_seen),
            'val_pr_auc': val['pr_auc'],
            'val_roc_auc': val['roc_auc'],
            'val_recall_at_95': val['recall_at_95'],
            'val_logit_std': val['logit_std'],
        })
        if val['pr_auc'] > best_val_pr:
            best_val_pr = val['pr_auc']
            best_epoch = epoch
            best_val = val
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        # Only reachable when every validation PR-AUC is NaN (a single-class smoke split).
        best_epoch, best_val = epochs, val
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test = evaluate(model, test_loader, device, max_batches)

    return {
        'n_params': n_params,
        'best_epoch': best_epoch,
        'val_pr_auc': best_val_pr if best_val_pr >= 0 else float('nan'),
        'best_val': best_val,
        'test': {k: test[k] for k in ('pr_auc', 'roc_auc', 'recall_at_99', 'recall_at_95', 'recall_at_90')},
        'test_logit_std': test['logit_std'],
        'pos_weight': n_neg / n_pos,
        'history': history,
        'state_dict': {k: v.cpu() for k, v in best_state.items()},
    }


def parse_seeds(tokens: list[str] | None) -> list[int] | None:
    if tokens is None:
        return None
    seeds: list[int] = []
    for token in tokens:
        for part in token.split(","):
            if "-" in part:
                lo, hi = part.split("-")
                seeds += list(range(int(lo), int(hi) + 1))
            elif part:
                seeds.append(int(part))
    return seeds


def write_json(path: Path, payload: dict) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def write_summary(out_dir: Path, group: str) -> Path:
    rows = []
    for path in sorted((out_dir / "runs" / group).glob("*.json")):
        run = json.loads(path.read_text())
        spec = run["spec"]
        best = run["best_val"]
        rows.append({
            "group": group, "variant": spec["variant"], "seed": spec["seed"],
            "lr": spec["lr"], "batch_norm": spec["batch_norm"], "pool": spec["pool"],
            "use_global": spec["use_global"], "use_local": spec["use_local"],
            "scalars": "+".join(spec["scalars"]), "epochs": spec["epochs"],
            "n_params": run["n_params"], "best_epoch": run["best_epoch"],
            "val_pr_auc": run["val_pr_auc"], "val_roc_auc": best.get("roc_auc"),
            "val_recall_at_95": best.get("recall_at_95"), "val_logit_std": best.get("logit_std"),
            "collapsed": bool(best.get("logit_std", np.inf) < COLLAPSE_LOGIT_STD),
            **{f"test_{k}": v for k, v in run["test"].items()},
        })
    path = out_dir / f"summary_{group}.csv"
    if rows:
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=list(GROUP_DEFAULTS), required=True)
    parser.add_argument("--variants", nargs="+", default=None, help="Subset of the group's variants.")
    parser.add_argument("--seeds", nargs="+", default=None, help="e.g. 0-9, or 0 1 2, or 0,3")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default=None, help="Default: mps, then cuda, then cpu.")
    parser.add_argument("--max-batches", type=int, default=None, help="Smoke tests only.")
    parser.add_argument("--h5", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--ckpt-dir", type=Path, default=None, help="Default checkpoints/camera_ready/<group>")
    parser.add_argument("--limit", type=int, default=None, help="Run at most N (variant, seed) pairs.")
    args = parser.parse_args()

    defaults = GROUP_DEFAULTS[args.group]
    device = args.device or default_device()
    h5_path = args.h5 or defaults["h5"]
    epochs = args.epochs or defaults["epochs"]
    seeds = parse_seeds(args.seeds) or defaults["seeds"]
    ckpt_dir = args.ckpt_dir or DEFAULT_CKPT_ROOT / args.group
    variants = group_variants(args.group)
    if args.variants:
        unknown = set(args.variants) - set(variants)
        if unknown:
            parser.error(f"unknown variants for {args.group}: {sorted(unknown)}; choose from {list(variants)}")
        variants = {k: v for k, v in variants.items() if k in args.variants}

    runs_dir = args.out_dir / "runs" / args.group
    runs_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    splits = make_splits(hdf5_path=h5_path)
    print(f"Device: {device}   h5: {h5_path}")
    print(f"Split: train={len(splits['train'])}  val={len(splits['val'])}  test={len(splits['test'])}")

    scalar_values = None
    if any(v["scalars"] for v in variants.values()):
        scalar_values = h5_scalar_values(h5_path)
        train_features = {k: v[splits['train']] for k, v in scalar_features(scalar_values).items()}

    todo = [(name, seed) for name in variants for seed in seeds
            if not (runs_dir / f"{name}__seed{seed}.json").exists()]
    print(f"Group {args.group}: {len(variants)} variants x {len(seeds)} seeds, {epochs} epochs; "
          f"{len(todo)} runs to do")
    if args.limit is not None:
        todo = todo[: args.limit]

    for name, seed in todo:
        spec = {
            "group": args.group, "variant": name, "seed": seed, "epochs": epochs,
            **variants[name], "n_scalars": len(variants[name]["scalars"]),
            "h5": str(h5_path), "split_seed": 42, "device": device,
            "max_batches": args.max_batches, "torch_version": torch.__version__,
        }
        scalar_matrix = None
        if spec["scalars"]:
            spec.update(fit_scalar_normalisation(train_features, spec["scalars"]))
            scalar_matrix = make_scalars(spec, scalar_values)
        print(f"[{args.group}/{name}] seed={seed}")
        result = train_one(spec, splits, h5_path, device, args.max_batches, scalar_matrix)
        spec["pos_weight"] = result["pos_weight"]
        spec["created"] = dt.datetime.now().isoformat(timespec="seconds")

        torch.save({
            "model_state_dict": result["state_dict"],
            "spec": spec,
            "best_epoch": result["best_epoch"],
            "val_pr_auc": result["val_pr_auc"],
        }, ckpt_dir / f"{name}__seed{seed}.pt")
        write_json(runs_dir / f"{name}__seed{seed}.json", {
            "spec": spec,
            "n_params": result["n_params"],
            "best_epoch": result["best_epoch"],
            "val_pr_auc": result["val_pr_auc"],
            "best_val": result["best_val"],
            "test": result["test"],
            "test_logit_std": result["test_logit_std"],
            "history": result["history"],
        })
        print(f"  params={result['n_params']:,}  best_epoch={result['best_epoch']}  "
              f"val_PR={result['val_pr_auc']:.4f}  TEST_PR={result['test']['pr_auc']:.4f}  "
              f"TEST_ROC={result['test']['roc_auc']:.4f}")

    print(f"Summary: {write_summary(args.out_dir, args.group)}")


if __name__ == "__main__":
    main()
