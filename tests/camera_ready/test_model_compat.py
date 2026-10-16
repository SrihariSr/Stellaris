"""Check that StellarisNetwork() is unchanged and the new options build and run."""
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stellaris.model import StellarisNetwork  # noqa: E402

CHECKPOINT = PROJECT_ROOT / "checkpoints" / "stellaris_best.pt"
TRAINSET = PROJECT_ROOT / "data" / "processed" / "kepler_trainset.h5"
PREDICTIONS = PROJECT_ROOT / "results" / "test_predictions.npz"

failures: list[str] = []


def expect(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'ok' if ok else 'FAIL'}] {label}{(': ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


def n_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def load_head_model() -> types.ModuleType:
    source = subprocess.run(
        ["git", "show", "HEAD:src/stellaris/model.py"],
        cwd=PROJECT_ROOT, check=True, capture_output=True, text=True,
    ).stdout
    module = types.ModuleType("model_head")
    exec(compile(source, "model_head.py", "exec"), module.__dict__)
    return module


def check_counts() -> None:
    default = StellarisNetwork()
    expect("default parameters 9,710,305", n_params(default) == 9_710_305, f"{n_params(default):,}")
    expect("head parameters 9,046,017", n_params(default.head) == 9_046_017, f"{n_params(default.head):,}")
    g = n_params(StellarisNetwork(use_local=False))
    expect("global-only parameters 8,914,737", g == 8_914_737, f"{g:,}")
    l = n_params(StellarisNetwork(use_global=False))
    expect("local-only parameters 1,321,905", l == 1_321_905, f"{l:,}")


def check_checkpoint() -> None:
    ckpt = torch.load(CHECKPOINT, weights_only=False, map_location="cpu")
    model = StellarisNetwork()
    expect("state_dict keys equal checkpoint keys",
           list(model.state_dict().keys()) == list(ckpt["model_state_dict"].keys()))
    result = model.load_state_dict(ckpt["model_state_dict"], strict=True)
    expect("strict load", not result.missing_keys and not result.unexpected_keys)


def check_against_head() -> None:
    head = load_head_model()
    for kwargs in ({}, {"use_local": False}, {"use_global": False}):
        torch.manual_seed(123)
        old = head.StellarisNetwork(**kwargs)
        after_old = torch.rand(4)
        torch.manual_seed(123)
        new = StellarisNetwork(**kwargs)
        after_new = torch.rand(4)
        same_init = all(
            torch.equal(a, b) for a, b in zip(old.state_dict().values(), new.state_dict().values())
        ) and list(old.state_dict()) == list(new.state_dict())
        # Equal draws afterwards mean construction consumed the generator identically.
        same_rng = torch.equal(after_old, after_new)
        expect(f"init identical to HEAD {kwargs or '(default)'}", same_init and same_rng)

        g = torch.randn(5, 1, 2001)
        l = torch.randn(5, 1, 201)
        old.eval()
        new.eval()
        with torch.no_grad():
            expect(f"eval forward identical to HEAD {kwargs or '(default)'}", torch.equal(old(g, l), new(g, l)))
        old.train()
        new.train()
        torch.manual_seed(7)
        out_old = old(g, l)
        torch.manual_seed(7)
        out_new = new(g, l)
        expect(f"train forward identical to HEAD {kwargs or '(default)'}", torch.equal(out_old, out_new))


def check_new_options() -> None:
    g = torch.randn(4, 1, 2001)
    l = torch.randn(4, 1, 201)
    s = torch.randn(4, 3)
    model = StellarisNetwork()
    expect("positional model(g, l)", model(g, l).shape == (4,))
    try:
        model(g, l, s)
        expect("default network rejects scalars", False)
    except ValueError:
        expect("default network rejects scalars", True)

    for kwargs in (
        {"n_scalars": 3},
        {"batch_norm": True},
        {"pool": "gap"},
        {"n_scalars": 3, "batch_norm": True, "pool": "gap"},
        {"use_local": False, "batch_norm": True, "pool": "gap"},
        {"use_global": False, "n_scalars": 3},
    ):
        net = StellarisNetwork(**kwargs)
        args = (g, l, s) if kwargs.get("n_scalars") else (g, l)
        out = net(*args)
        expect(f"forward shape {kwargs}", out.shape == (4,), str(tuple(out.shape)))
        if kwargs.get("n_scalars"):
            try:
                net(g, l)
                expect(f"missing scalars rejected {kwargs}", False)
            except ValueError:
                expect(f"missing scalars rejected {kwargs}", True)

    bn = StellarisNetwork(batch_norm=True)
    bns = [m for m in bn.modules() if isinstance(m, torch.nn.BatchNorm1d)]
    expect("batch_norm adds 14 BatchNorm1d layers", len(bns) == 14, str(len(bns)))
    expect("BatchNorm running stats untouched by the dummy pass",
           all(int(m.num_batches_tracked) == 0 and torch.all(m.running_mean == 0) for m in bns))
    keys = list(bn.state_dict())
    expect("conv1, conv2 and pool names kept",
           "global_branch.blocks.0.conv1.weight" in keys and "global_branch.blocks.0.conv2.weight" in keys
           and hasattr(bn.global_branch.blocks[0], "pool"))
    gap = StellarisNetwork(pool="gap")
    expect("gap head width 288 = 256 + 32", gap.head[0].in_features == 288, str(gap.head[0].in_features))
    expect("scalars widen the head by n_scalars",
           StellarisNetwork(n_scalars=3).head[0].in_features == StellarisNetwork().head[0].in_features + 3)


def check_test_split() -> None:
    if not TRAINSET.exists():
        print(f"  [skip] {TRAINSET} not present; test-split scoring skipped")
        return
    from torch.utils.data import DataLoader
    from stellaris.dataset import StellarisDataset, make_splits

    splits = make_splits()
    ds = StellarisDataset(TRAINSET, splits["test"])
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    model = StellarisNetwork()
    ckpt = torch.load(CHECKPOINT, weights_only=False, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    logits, labels = [], []
    with torch.no_grad():
        for batch in loader:
            logits.append(model(batch["global_view"], batch["local_view"]).numpy())
            labels.append(batch["label"].numpy())
    probs = expit(np.concatenate(logits))
    labels = np.concatenate(labels)
    ref = np.load(PREDICTIONS)
    max_diff = float(np.max(np.abs(probs - ref["probs"])))
    expect("test split size 1,020", len(labels) == 1020, str(len(labels)))
    expect("test positives 412", int(labels.sum()) == 412, str(int(labels.sum())))
    expect("labels identical to test_predictions.npz", np.array_equal(labels, ref["labels"]))
    expect("probs within 1e-5 of test_predictions.npz", max_diff <= 1e-5, f"max |diff| = {max_diff:.2e}")


def main() -> int:
    check_counts()
    check_checkpoint()
    check_against_head()
    check_new_options()
    check_test_split()
    print("PASS" if not failures else f"FAIL ({len(failures)} checks)")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
