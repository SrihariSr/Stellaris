"""Plot training and validation curves from a saved Stellaris checkpoint."""
from pathlib import Path
import torch
import matplotlib.pyplot as plt

CHECKPOINTS = Path("checkpoints")
CHECKPOINT_PATH = CHECKPOINTS / "stellaris_final.pt"
OUTPUT_PATH = CHECKPOINTS / "training_curves.png"


def main():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"No checkpoint at {CHECKPOINT_PATH}. Train the model first.")

    ckpt = torch.load(CHECKPOINT_PATH, weights_only=False, map_location="cpu")
    history = ckpt['history']

    epochs = [h['epoch'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    val_pr_auc = [h['pr_auc'] for h in history]
    val_roc_auc = [h['roc_auc'] for h in history]
    recall_at_99 = [h['recall_at_99_precision'] for h in history]

    # Find best epoch by val PR-AUC
    best_idx = max(range(len(val_pr_auc)), key=lambda i: val_pr_auc[i])
    best_epoch = epochs[best_idx]
    best_pr_auc = val_pr_auc[best_idx]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Train loss
    ax = axes[0, 0]
    ax.plot(epochs, train_loss, marker='o', markersize=3, linewidth=1)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training loss')
    ax.set_title('Training loss')
    ax.grid(alpha=0.3)

    # Val PR-AUC
    ax = axes[0, 1]
    ax.plot(epochs, val_pr_auc, marker='o', markersize=3, linewidth=1, color='C1')
    ax.axvline(best_epoch, linestyle='--', color='green', alpha=0.5, label=f'best (epoch {best_epoch}, {best_pr_auc:.4f})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Val PR-AUC')
    ax.set_title('Validation PR-AUC (operational metric)')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)

    # Val ROC-AUC
    ax = axes[1, 0]
    ax.plot(epochs, val_roc_auc, marker='o', markersize=3, linewidth=1, color='C2')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Val ROC-AUC')
    ax.set_title('Validation ROC-AUC')
    ax.grid(alpha=0.3)

    # Recall at 99% precision
    ax = axes[1, 1]
    ax.plot(epochs, recall_at_99, marker='o', markersize=3, linewidth=1, color='C3')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Recall @ 99% precision')
    ax.set_title('Operational metric: recall when demanding 99% precision')
    ax.grid(alpha=0.3)

    plt.suptitle('Stellaris training curves')
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=120, bbox_inches='tight')
    print(f"Saved plot to {OUTPUT_PATH}")

    # Diagnostic text
    print("\n" + "-" * 60)
    print("TRAINING DIAGNOSTICS")
    print("-" * 60)
    print(f"Best validation PR-AUC: {best_pr_auc:.4f} at epoch {best_epoch}/{epochs[-1]}")
    print(f"Final validation PR-AUC: {val_pr_auc[-1]:.4f}")
    print(f"Final training loss: {train_loss[-1]:.4f}")
    print(f"First-epoch training loss: {train_loss[0]:.4f}")
    print()

    # Heuristic: did it plateau?
    last_5_pr_auc = val_pr_auc[-5:]
    early_window_pr_auc = val_pr_auc[max(0, len(val_pr_auc) - 10):len(val_pr_auc) - 5]
    if len(early_window_pr_auc) >= 2:
        recent_gain = max(last_5_pr_auc) - max(early_window_pr_auc)
        print(f"PR-AUC gain in final 5 epochs vs preceding 5: {recent_gain:+.4f}")
        if recent_gain < 0.002:
            print("Model appears to have plateaued. 40 epochs was sufficient.")
        elif recent_gain > 0.01:
            print("Model is still improving meaningfully. Consider training longer.")
        else:
            print("Marginal late-stage gains. Borderline case; could go either way.")

    # Overfitting check
    if len(train_loss) >= 5 and len(val_pr_auc) >= 5:
        train_still_improving = train_loss[-1] < train_loss[-5] - 0.01
        val_declining = val_pr_auc[-1] < max(val_pr_auc) - 0.02
        if train_still_improving and val_declining:
            print("Possible overfitting: train loss still falling but val PR-AUC dropping.")
            print("The best checkpoint is from earlier and is the right one to use.")


if __name__ == '__main__':
    main()