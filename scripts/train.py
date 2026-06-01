"""Train StellarisNetwork on the Kepler training set."""
from pathlib import Path
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


# Hyperparameters
BATCH_SIZE = 64
NUM_EPOCHS = 40
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

DEVICE = (
    'mps' if torch.backends.mps.is_available()
    else 'cuda' if torch.cuda.is_available()
    else 'cpu'
)

CHECKPOINTS = Path("checkpoints")
CHECKPOINTS.mkdir(exist_ok=True)


def evaluate(model: nn.Module, loader: DataLoader) -> dict:
    """Compute validation metrics on a dataloader."""
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            global_view = batch['global_view'].to(DEVICE)
            local_view = batch['local_view'].to(DEVICE)
            logits = model(global_view, local_view)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(batch['label'].numpy())

    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    probs = 1 / (1 + np.exp(-logits))  # sigmoid function

    pr_auc = average_precision_score(labels, probs)
    roc_auc = roc_auc_score(labels, probs)

    # Recall at 99% precision
    precision, recall, _thresholds = precision_recall_curve(labels, probs)
    # find highest recall where precision >= 0.99
    valid = precision >= 0.99
    recall_at_99 = recall[valid].max() if valid.any() else 0.0

    return {
        'pr_auc': pr_auc,
        'roc_auc': roc_auc,
        'recall_at_99_precision': float(recall_at_99),
    }


def main():
    print(f"Device: {DEVICE}")

    # Build splits and dataloaders
    splits = make_splits()
    print(f"Train: {len(splits['train'])}  Val: {len(splits['val'])}  Test: {len(splits['test'])}")

    train_ds = StellarisDataset(DEFAULT_DATASET_PATH, splits['train'])
    val_ds = StellarisDataset(DEFAULT_DATASET_PATH, splits['val'])

    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE != 'mps')
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE != 'mps')
    )

    # Class weight for the modest imbalance (1 positive per ~1.45 negative)
    n_pos = float(train_ds.labels[splits['train']].sum())
    n_neg = float(len(splits['train']) - n_pos)
    pos_weight = torch.tensor(n_neg / n_pos, device=DEVICE)
    print(f"pos_weight: {pos_weight.item():.3f}")

    # Model, loss, optimiser
    model = StellarisNetwork().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Training loop
    best_pr_auc = 0.0
    history = []

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss_total = 0.0
        n_seen = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}"):
            global_view = batch['global_view'].to(DEVICE)
            local_view = batch['local_view'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            optimiser.zero_grad()
            logits = model(global_view, local_view)
            loss = criterion(logits, labels)
            loss.backward()
            optimiser.step()

            train_loss_total += loss.item() * labels.size(0)
            n_seen += labels.size(0)

        train_loss = train_loss_total / n_seen
        val_metrics = evaluate(model, val_loader)

        epoch_record = {
            'epoch': epoch,
            'train_loss': train_loss,
            **val_metrics,
        }
        history.append(epoch_record)

        print(
            f"Epoch {epoch}/{NUM_EPOCHS}: "
            f"train_loss={train_loss:.4f}  "
            f"val_pr_auc={val_metrics['pr_auc']:.4f}  "
            f"val_roc_auc={val_metrics['roc_auc']:.4f}  "
            f"recall@99%P={val_metrics['recall_at_99_precision']:.3f}"
        )

        # Save the best checkpoint
        if val_metrics['pr_auc'] > best_pr_auc:
            best_pr_auc = val_metrics['pr_auc']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'pr_auc': best_pr_auc,
                'history': history,
            }, CHECKPOINTS / "stellaris_best.pt")
            print(f"  -> new best, checkpoint saved")

    # Always save final
    torch.save({
        'epoch': NUM_EPOCHS,
        'model_state_dict': model.state_dict(),
        'history': history,
    }, CHECKPOINTS / "stellaris_final.pt")
    print(f"\nDone. Best val PR-AUC: {best_pr_auc:.4f}")


if __name__ == '__main__':
    main()