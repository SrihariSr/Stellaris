from pathlib import Path
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
)
from tqdm import tqdm

from stellaris.dataset import StellarisDataset, make_splits, DEFAULT_DATASET_PATH
from stellaris.model import StellarisNetwork

BATCH_SIZE = 64
NUM_EPOCHS = 40
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

SEEDS = list(range(10))
VARIANTS = [
    # (name, use_global, use_local)
    ("both",        True,  True),
    ("global-only", True,  False),
    ("local-only",  False, True),
]

DEVICE = (
    'mps' if torch.backends.mps.is_available()
    else 'cuda' if torch.cuda.is_available()
    else 'cpu'
)

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

def evaluate(model: nn.Module, loader: DataLoader) -> dict:
    """
    Score a model on a dataloader.

    Returns PR-AUC, ROC-AUC, and recall at three fixed precision operating
    points. PR-AUC is the headline: with 40% positives and a detection task
    where false positives cost telescope time, it is the metric that matters.
    ROC-AUC is reported because the literature does, not because it is the more
    informative number here.
    """
    model.eval()
    all_logits, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            gv = batch['global_view'].to(DEVICE)
            lv = batch['local_view'].to(DEVICE)
            logits = model(gv, lv)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(batch['label'].numpy())

    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    probs = 1.0 / (1.0 + np.exp(-logits))   # sigmoid: logit -> probability

    out = {
        'pr_auc': float(average_precision_score(labels, probs)),
        'roc_auc': float(roc_auc_score(labels, probs)),
    }

    # Recall at fixed precision. "If I insist on being right 95% of the time,
    # what fraction of the real planets do I still catch?" This is the number a
    # working astronomer actually cares about, because it sets how much
    # follow-up time gets wasted.
    precision, recall, _ = precision_recall_curve(labels, probs)
    for p in (0.99, 0.95, 0.90):
        mask = precision >= p
        out[f'recall_at_{int(p * 100)}'] = float(recall[mask].max()) if mask.any() else 0.0

    return out

def train_one(use_global: bool, use_local: bool, seed: int, splits: dict) -> dict:
    """
    Train a single variant with a single seed, then score it on the TEST set.

    We select the checkpoint by best VALIDATION PR-AUC (exactly as train.py
    does), and only then look at the test set. Selecting on test would be
    cheating: the test number would no longer be an honest estimate of how the
    model behaves on data it has never influenced.
    """
    # Seed everything that introduces randomness: weight init and batch shuffling.
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = StellarisDataset(DEFAULT_DATASET_PATH, splits['train'])
    val_ds   = StellarisDataset(DEFAULT_DATASET_PATH, splits['val'])
    test_ds  = StellarisDataset(DEFAULT_DATASET_PATH, splits['test'])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=(DEVICE != 'mps'))
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=(DEVICE != 'mps'))
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=(DEVICE != 'mps'))

    # Class weighting. There are ~1.46 negatives per positive, so a positive
    # example is up-weighted by that factor in the loss. Without this the model
    # could improve its loss simply by leaning towards "not a planet".
    n_pos = float(train_ds.labels[splits['train']].sum())
    n_neg = float(len(splits['train']) - n_pos)
    pos_weight = torch.tensor(n_neg / n_pos, device=DEVICE)

    model = StellarisNetwork(use_global=use_global, use_local=use_local).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY)

    n_params = sum(p.numel() for p in model.parameters())

    best_val_pr = -1.0
    best_state = None
    best_epoch = -1

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        for batch in tqdm(train_loader, desc=f"  epoch {epoch}/{NUM_EPOCHS}", leave=False):
            gv = batch['global_view'].to(DEVICE)
            lv = batch['local_view'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            optimiser.zero_grad()
            loss = criterion(model(gv, lv), labels)
            loss.backward()
            optimiser.step()

        val = evaluate(model, val_loader)
        if val['pr_auc'] > best_val_pr:
            best_val_pr = val['pr_auc']
            best_epoch = epoch
            # .clone() because state_dict() hands back live tensors that keep
            # training. Without the clone we would be saving a moving target.
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    # Restore the best-validation weights, then and only then touch the test set.
    model.load_state_dict(best_state)
    test = evaluate(model, test_loader)

    return {
        'n_params': n_params,
        'best_epoch': best_epoch,
        'val_pr_auc': best_val_pr,
        **{f'test_{k}': v for k, v in test.items()},
    }

def main() -> None:
    print(f"Device: {DEVICE}\n")

    # The CNN's own split, seed 42. Fixed across every run below, so the test
    # set never moves and all numbers stay comparable.
    splits = make_splits()
    print(f"Split (fixed for all runs): "
          f"train={len(splits['train'])}  val={len(splits['val'])}  test={len(splits['test'])}")
    print(f"Variants: {[v[0] for v in VARIANTS]}   Seeds: {SEEDS}")
    print(f"Total runs: {len(VARIANTS) * len(SEEDS)}\n")

    records = []
    t0 = time.time()

    for name, ug, ul in VARIANTS:
        for seed in SEEDS:
            print(f"[{name}] seed={seed}")
            r = train_one(ug, ul, seed, splits)
            r.update({'variant': name, 'seed': seed})
            records.append(r)
            print(f"  params={r['n_params']:,}  best_epoch={r['best_epoch']}  "
                  f"val_PR={r['val_pr_auc']:.4f}  TEST_PR={r['test_pr_auc']:.4f}  "
                  f"TEST_ROC={r['test_roc_auc']:.4f}\n")

    with open(RESULTS / "ablation_results.json", "w") as f:
        json.dump(records, f, indent=2)

    # summary table
    print("=" * 78)
    print("ABLATION: Kepler test set (412 positives, 608 negatives), mean +/- std over seeds")
    print("=" * 78)
    print(f"{'variant':<13} {'params':>11}  {'PR-AUC':>15}  {'ROC-AUC':>15}  {'R@95%P':>13}")
    print("-" * 78)

    for name, _, _ in VARIANTS:
        rows = [r for r in records if r['variant'] == name]
        prs = np.array([r['test_pr_auc'] for r in rows])
        rocs = np.array([r['test_roc_auc'] for r in rows])
        r95 = np.array([r['test_recall_at_95'] for r in rows])
        params = rows[0]['n_params']
        print(f"{name:<13} {params:>11,}  "
              f"{prs.mean():.4f} +/- {prs.std(ddof=1):.4f}  "
              f"{rocs.mean():.4f} +/- {rocs.std(ddof=1):.4f}  "
              f"{r95.mean():.3f} +/- {r95.std(ddof=1):.3f}")

    print("-" * 78)
    print("XGBoost (19 hand-engineered view features): run train_xgboost_baseline.py")
    print("  AFTER dump_split.py, or the comparison is on the wrong test set.")
    print(f"\nTotal time: {(time.time() - t0) / 60:.1f} min")
    print(f"Written: {RESULTS / 'ablation_results.json'}")

if __name__ == '__main__':
    main()
