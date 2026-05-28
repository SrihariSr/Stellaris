import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
koi = pd.read_csv(ROOT / "data/catalogs/kepler_koi.csv")

# Drop CANDIDATE rows for training, these are unresolved
train_mask = koi['koi_disposition'].isin(['CONFIRMED', 'FALSE POSITIVE'])
train_koi = koi[train_mask].copy()

train_koi['label'] = (train_koi['koi_disposition'] == 'CONFIRMED').astype(int)

eval_koi = koi[koi['koi_disposition'] == 'CANDIDATE'].copy()
eval_koi.to_csv(ROOT / "data/catalogs/kepler_koi_candidates_eval.csv", index=False)
print(f"Saved {len(eval_koi)} CANDIDATE KOIs for evaluation.")

train_koi.to_csv(ROOT / "data/catalogs/kepler_koi_train.csv", index=False)
print(f"Saved {len(train_koi)} labelled KOIs for training.")

print(f"Training KOIs: {len(train_koi)}")
print(f"Positives (CONFIRMED): {train_koi['label'].sum()}")
print(f"Negatives (FALSE POSITIVE): {(train_koi['label'] == 0).sum()}")
print(f"Class imbalance: 1 : {(train_koi['label'] == 0).sum() / train_koi['label'].sum():.2f}")

