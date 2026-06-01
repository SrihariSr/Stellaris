"""Verify the StellarisNetwork model assembles correctly and a forward pass works."""
import torch
# pyrefly: ignore [missing-import]
from stellaris.model import StellarisNetwork


# Instantiate
model = StellarisNetwork()

# Parameter count
n_params = sum(p.numel() for p in model.parameters())
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {n_params:,}")
print(f"Trainable parameters: {n_trainable:,}")

# Dummy forward pass
batch_size = 8
global_view = torch.randn(batch_size, 1, 2001)
local_view = torch.randn(batch_size, 1, 201)

logits = model(global_view, local_view)
print(f"\nOutput shape: {logits.shape}  (expect torch.Size([8]))")
print(f"Output dtype: {logits.dtype}  (expect torch.float32)")
print(f"Output sample: {logits.tolist()}")

# Convert logits to probabilities (for sanity)
probs = torch.sigmoid(logits)
print(f"\nProbabilities (after sigmoid): {probs.tolist()}")
print(f"All in [0, 1]: {((probs >= 0) & (probs <= 1)).all().item()}")

# Backward pass (will training actually work?)
labels = torch.randint(0, 2, (batch_size,), dtype=torch.float32)
loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
loss.backward()
print(f"\nLoss: {loss.item():.4f}")
print(f"Gradient sample (first conv layer): "
      f"{model.global_branch.blocks[0].conv1.weight.grad.abs().mean().item():.6f}")