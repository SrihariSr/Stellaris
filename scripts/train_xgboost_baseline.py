"""
XGBoost baseline on the same Kepler dataset the CNN was trained on.

Extracts hand-engineered features from global and local views, trains
XGBoost, evaluates on the same test set used for the CNN. Result is
directly comparable to the CNN's PR-AUC and ROC-AUC numbers.
"""
from pathlib import Path
import json

import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve, roc_auc_score, average_precision_score,
    roc_curve, confusion_matrix
)
import matplotlib.pyplot as plt
import xgboost as xgb


DATA_PATH = Path("data/processed/kepler_trainset.h5")
SPLIT_PATH = Path("data/processed/split.json")  # train/val/test kepid lists
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_features(global_view: np.ndarray, local_view: np.ndarray) -> np.ndarray:
    """
    Engineered features from a single (global, local) example.
    
    These are the standard transit-vetting features used in literature
    (Shallue & Vanderburg's baseline, Yu et al., others).
    """
    g = global_view  # length 2001
    l = local_view   # length 201
    
    features = [
        # Global view statistics
        np.min(g),                   # deepest dip
        np.max(g),                   # highest peak  
        np.mean(g),                  # mean flux
        np.std(g),                   # noise level
        np.median(g),
        np.percentile(g, 10),        # lower tail
        np.percentile(g, 90),        # upper tail
        
        # Local view (zoomed on transit)
        np.min(l),                   # transit depth
        np.max(l),                   # ingress/egress max
        np.mean(l),
        np.std(l),
        np.median(l),
        
        # Transit shape features
        np.argmin(l) / len(l),       # transit center position
        np.argmin(g) / len(g),
        
        # Duration proxy: how many local-view bins are below half-depth
        np.sum(l < np.min(l) * 0.5),
        
        # Asymmetry: difference between left and right halves of local view
        np.mean(l[:len(l)//2]) - np.mean(l[len(l)//2:]),
        
        # Sharpness: difference between min and surrounding region
        np.min(l) - np.mean(np.concatenate([l[:50], l[-50:]])),
        
        # Global vs local depth consistency  
        np.min(g) - np.min(l),
        
        # Noise outside transit (using global view ends)
        np.std(np.concatenate([g[:200], g[-200:]])),
    ]
    return np.array(features)


def load_data():
    """Load all examples and split into train/val/test by kepid."""
    print(f"Loading {DATA_PATH}...")
    
    with h5py.File(DATA_PATH, 'r') as f:
        globals_all = f['global_view'][:]
        locals_all = f['local_view'][:]
        labels_all = f['label'][:]
        kepids_all = f['kepid'][:]
    
    print(f"Loaded {len(globals_all)} examples")
    
    # Extract features for every example
    print("Extracting features...")
    X = np.array([
        extract_features(globals_all[i], locals_all[i])
        for i in range(len(globals_all))
    ])
    y = labels_all
    
    print(f"  Feature matrix: {X.shape}")
    
    # Load existing train/val/test split
    if SPLIT_PATH.exists():
        with open(SPLIT_PATH) as f:
            split = json.load(f)
        train_kepids = set(split['train'])
        val_kepids = set(split['val'])
        test_kepids = set(split['test'])
        print(f"  Loaded split: {len(train_kepids)} train, {len(val_kepids)} val, {len(test_kepids)} test kepids")
    else:
        # Fallback: same kepid-based split as your CNN training
        # (deterministic by kepid hash)
        print("  No split.json found, creating deterministic kepid split...")
        unique_kepids = np.unique(kepids_all)
        rng = np.random.default_rng(42)
        rng.shuffle(unique_kepids)
        n = len(unique_kepids)
        train_kepids = set(unique_kepids[:int(0.7 * n)])
        val_kepids = set(unique_kepids[int(0.7 * n):int(0.85 * n)])
        test_kepids = set(unique_kepids[int(0.85 * n):])
    
    train_mask = np.array([k in train_kepids for k in kepids_all])
    val_mask = np.array([k in val_kepids for k in kepids_all])
    test_mask = np.array([k in test_kepids for k in kepids_all])
    
    return {
        'X_train': X[train_mask], 'y_train': y[train_mask],
        'X_val': X[val_mask], 'y_val': y[val_mask],
        'X_test': X[test_mask], 'y_test': y[test_mask],
    }


def train_and_evaluate(data: dict):
    """Train XGBoost and evaluate on test set."""
    print("\nTraining XGBoost...")
    
    pos_weight = (data['y_train'] == 0).sum() / max(1, (data['y_train'] == 1).sum())
    print(f"  Positive class weight: {pos_weight:.3f}")
    
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=pos_weight,
        eval_metric='aucpr',
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
    )
    
    model.fit(
        data['X_train'], data['y_train'],
        eval_set=[(data['X_val'], data['y_val'])],
        verbose=False,
    )
    
    print(f"  Best iteration: {model.best_iteration}")
    
    # Test set evaluation
    print("\nEvaluating on test set...")
    y_pred_proba = model.predict_proba(data['X_test'])[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    pr_auc = average_precision_score(data['y_test'], y_pred_proba)
    roc_auc = roc_auc_score(data['y_test'], y_pred_proba)
    
    print(f"\n{'-'*60}")
    print(f"XGBOOST BASELINE TEST SET RESULTS")
    print(f"{'-'*60}")
    print(f"PR-AUC:  {pr_auc:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"\nComparison to CNN (Stellaris):")
    print(f"CNN PR-AUC:  0.9451")
    print(f"CNN ROC-AUC: 0.9588")
    
    # Recall at high precision (the metric that matters for triage)
    precision, recall, thresholds = precision_recall_curve(data['y_test'], y_pred_proba)
    for target_p in [0.99, 0.95, 0.90]:
        mask = precision >= target_p
        if mask.any():
            best_recall = recall[mask].max()
            print(f"Recall @ {target_p:.0%} precision: {best_recall:.3f}")
    
    cm = confusion_matrix(data['y_test'], y_pred)
    print(f"\nConfusion matrix (threshold 0.5):")
    print(f"  {cm}")
    
    return {
        'pr_auc': float(pr_auc),
        'roc_auc': float(roc_auc),
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'feature_importance': model.feature_importances_.tolist(),
    }


def plot_results(metrics: dict):
    """Side-by-side comparison plot."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    ax.plot(metrics['recall'], metrics['precision'],
            label=f"XGBoost (PR-AUC={metrics['pr_auc']:.4f})",
            color='#E74C3C', linewidth=2)
    
    # If you have the CNN PR data saved, you could overlay here
    # For now just plot a reference horizontal line at CNN PR-AUC
    ax.axhline(0.9451, color='#3498DB', linestyle='--',
               linewidth=1.5, alpha=0.7,
               label='Stellaris CNN PR-AUC = 0.9451')
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('XGBoost baseline vs Stellaris CNN (Kepler test set)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    
    out = OUTPUT_DIR / "xgboost_baseline_comparison.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot: {out}")


def main():
    data = load_data()
    metrics = train_and_evaluate(data)
    plot_results(metrics)
    
    # Save metrics
    with open(OUTPUT_DIR / "xgboost_baseline_metrics.json", 'w') as f:
        json.dump({k: v for k, v in metrics.items() if k != 'feature_importance'}, f, indent=2)
    
    print("\nDone.")


if __name__ == '__main__':
    main()