"""Evaluate the best Stellaris checkpoint on the held-out test set.

Reports headline metrics (PR-AUC, ROC-AUC, recall at 99% precision), saves
plots of the PR curve, ROC curve, and confusion matrix at the threshold that
maximises operational utility.
"""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
)

from stellaris.dataset import StellarisDataset, make_splits, DEFAULT_DATASET_PATH
from stellaris.model import StellarisNetwork


CHECKPOINTS = Path("checkpoints")
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

DEVICE = (
    'mps' if torch.backends.mps.is_available()
    else 'cuda' if torch.cuda.is_available()
    else 'cpu'
)


def predict(model, loader):
    """Run model in eval mode; return (probs, labels) as numpy arrays."""
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            g = batch['global_view'].to(DEVICE)
            l = batch['local_view'].to(DEVICE)
            logits = model(g, l)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(batch['label'].numpy())
    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    probs = 1 / (1 + np.exp(-logits))
    return probs, labels


def main():
    print(f"Device: {DEVICE}")

    # Load splits and test dataset
    splits = make_splits()
    test_ds = StellarisDataset(DEFAULT_DATASET_PATH, splits['test'])
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    print(f"Test set: {len(test_ds)} examples")

    # Load best checkpoint
    ckpt_path = CHECKPOINTS / "stellaris_best.pt"
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=DEVICE)
    model = StellarisNetwork().to(DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Trained for: {ckpt['epoch']} epochs")
    print(f"Best val PR-AUC during training: {ckpt['pr_auc']:.4f}")

    # Get predictions
    probs, labels = predict(model, test_loader)
    print(f"\nTest set positive rate: {labels.mean():.3f}")

    # Headline metrics
    pr_auc = average_precision_score(labels, probs)
    roc_auc = roc_auc_score(labels, probs)

    precision, recall, thresholds_pr = precision_recall_curve(labels, probs)
    fpr, tpr, thresholds_roc = roc_curve(labels, probs)

    # Recall at various precision targets
    print("\n" + "-" * 60)
    print("HEADLINE METRICS ON TEST SET")
    print("-" * 60)
    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

    print(f"\nRecall at fixed precision targets:")
    for target in [0.99, 0.95, 0.90, 0.80, 0.50]:
        valid = precision >= target
        r = recall[valid].max() if valid.any() else 0.0
        print(f"Recall @ {target * 100:.0f}% precision: {r:.3f}")

    # Operating point: maximise F1 (balanced precision/recall tradeoff)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    best_idx = np.argmax(f1)
    best_threshold = thresholds_pr[min(best_idx, len(thresholds_pr) - 1)]
    best_f1 = f1[best_idx]

    # Confusion matrix at best F1 threshold
    pred_labels = (probs >= best_threshold).astype(int)
    cm = confusion_matrix(labels, pred_labels)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n" + "-" * 60)
    print(f"OPERATING POINT (best F1)")
    print("-" * 60)
    print(f"Threshold: {best_threshold:.4f}")
    print(f"F1 score:  {best_f1:.4f}")
    print(f"Precision: {tp / (tp + fp):.4f}")
    print(f"Recall:    {tp / (tp + fn):.4f}")
    print(f"\n  Confusion matrix:")
    print(f"                  Predicted")
    print(f"                  FP    Planet")
    print(f"    True FP      {tn:4d}   {fp:4d}")
    print(f"    True Planet  {fn:4d}   {tp:4d}")

    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # PR curve
    axes[0].plot(recall, precision, linewidth=1.5)
    axes[0].fill_between(recall, precision, alpha=0.2)
    axes[0].set_xlabel('Recall')
    axes[0].set_ylabel('Precision')
    axes[0].set_title(f'Precision-Recall  (AUC = {pr_auc:.3f})')
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1.02)
    axes[0].grid(alpha=0.3)

    # ROC curve
    axes[1].plot(fpr, tpr, linewidth=1.5)
    axes[1].plot([0, 1], [0, 1], 'k--', linewidth=0.5, alpha=0.5)
    axes[1].fill_between(fpr, tpr, alpha=0.2)
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title(f'ROC curve  (AUC = {roc_auc:.3f})')
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1.02)
    axes[1].grid(alpha=0.3)

    # Confusion matrix
    ax = axes[2]
    im = ax.imshow(cm, cmap='Blues', aspect='auto')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['FP', 'Planet'])
    ax.set_yticklabels(['FP', 'Planet'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion matrix at F1-optimal threshold')
    # Annotate cells
    for i in range(2):
        for j in range(2):
            colour = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color=colour, fontsize=14, fontweight='bold')

    plt.suptitle('Stellaris test-set evaluation', y=1.02)
    plt.tight_layout()
    plot_path = RESULTS / "test_evaluation.png"
    plt.savefig(plot_path, dpi=120, bbox_inches='tight')
    print(f"\nPlots saved to {plot_path}")

    # Save raw predictions for later analysis
    np.savez(
        RESULTS / "test_predictions.npz",
        probs=probs,
        labels=labels,
        threshold=best_threshold,
    )
    print(f"Predictions saved to {RESULTS / 'test_predictions.npz'}")


if __name__ == '__main__':
    main()