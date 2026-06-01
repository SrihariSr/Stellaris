"""Verify the dataset class and splits."""
import torch
from torch.utils.data import DataLoader

# pyrefly: ignore [missing-import]
from stellaris.dataset import StellarisDataset, make_splits, DEFAULT_DATASET_PATH


# Build splits
splits = make_splits()
for name, idx in splits.items():
    print(f"{name}: {len(idx)} examples")

# Build a Dataset for the training split
train_ds = StellarisDataset(DEFAULT_DATASET_PATH, splits['train'])

# Inspect a sample
sample = train_ds[0]
print(f"\nSample inspection:")
print(f"global_view shape: {sample['global_view'].shape}  (expect torch.Size([1, 2001]))")
print(f"local_view shape:  {sample['local_view'].shape}   (expect torch.Size([1, 201]))")
print(f"label: {sample['label']}  (expect 0.0 or 1.0)")
print(f"global_view dtype: {sample['global_view'].dtype}  (expect torch.float32)")

# Try a DataLoader iteration
loader = DataLoader(train_ds, batch_size=32, shuffle=True)
batch = next(iter(loader))
print(f"\nBatch inspection:")
print(f"global_view batch shape: {batch['global_view'].shape}  (expect [32, 1, 2001])")
print(f"local_view batch shape:  {batch['local_view'].shape}   (expect [32, 1, 201])")
print(f"label batch shape:       {batch['label'].shape}        (expect [32])")
print(f"Positives in batch: {batch['label'].sum().item():.0f}/32")