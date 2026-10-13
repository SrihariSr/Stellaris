"""Check that train_variants.train_one reproduces HEAD run_ablation.train_one exactly (short CPU runs)."""
import argparse
import subprocess
import sys
import time
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from stellaris.dataset import DEFAULT_DATASET_PATH, make_splits  # noqa: E402
import train_variants  # noqa: E402

VARIANTS = {"both": (True, True), "global-only": (True, False), "local-only": (False, True)}


def load_head_ablation(epochs: int) -> types.ModuleType:
    source = subprocess.run(
        ["git", "show", "HEAD:scripts/run_ablation.py"],
        cwd=PROJECT_ROOT, check=True, capture_output=True, text=True,
    ).stdout
    # The only module-level side effect; dropped so nothing is written under results/.
    source = source.replace("RESULTS.mkdir(exist_ok=True)", "pass")
    module = types.ModuleType("run_ablation_head")
    exec(compile(source, "run_ablation_head.py", "exec"), module.__dict__)
    module.DEVICE = "cpu"
    module.NUM_EPOCHS = epochs
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--variants", nargs="+", default=["local-only"],
                        help="Also passes for both and global-only (about 10 min on CPU for 2 epochs).")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not DEFAULT_DATASET_PATH.exists():
        print(f"SKIP: {DEFAULT_DATASET_PATH} not present")
        return 0
    head = load_head_ablation(args.epochs)
    splits = make_splits()
    variants = train_variants.group_variants("ablation")
    failures = []
    for name in args.variants:
        use_global, use_local = VARIANTS[name]
        t0 = time.time()
        ref = head.train_one(use_global, use_local, args.seed, splits)
        t1 = time.time()
        spec = {**variants[name], "variant": name, "seed": args.seed, "epochs": args.epochs}
        new = train_variants.train_one(spec, splits, DEFAULT_DATASET_PATH, "cpu")
        t2 = time.time()
        got = {
            "n_params": new["n_params"],
            "best_epoch": new["best_epoch"],
            "val_pr_auc": new["val_pr_auc"],
            **{f"test_{k}": v for k, v in new["test"].items()},
        }
        same = all(got[k] == ref[k] for k in ref)
        print(f"  {name} seed {args.seed}, {args.epochs} epochs: HEAD {t1 - t0:.0f}s, new {t2 - t1:.0f}s, "
              f"identical={same}")
        for k in ref:
            print(f"    {k:<22} HEAD {ref[k]!r:<24} new {got[k]!r}")
        if not same:
            failures.append(name)
        if len(new["history"]) != args.epochs or new["history"][new["best_epoch"] - 1]["val_pr_auc"] != ref["val_pr_auc"]:
            failures.append(f"{name}: history inconsistent with the selected epoch")
    print("PASS" if not failures else f"FAIL {failures}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
