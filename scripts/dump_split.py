"""
Write out the exact train/val/test split that the CNN was trained on.

WHY THIS FILE EXISTS
--------------------
Two different pieces of code disagree about what "the test set" means, and that
disagreement quietly invalidates your headline comparison.

  * dataset.py `make_splits()` builds the split with GroupShuffleSplit(seed=42),
    grouped by `kepid` so no star can land in two partitions. It returns *row
    indices* into the HDF5 file. This is what train.py uses, so this is the
    split behind the CNN's PR-AUC of 0.9451.

  * train_xgboost_baseline.py looks for `data/processed/split.json`, expecting
    *kepid lists*. That file has never existed in the repo. So the script fell
    through to its own fallback:
        np.random.default_rng(42).shuffle(unique_kepids)  ->  70/15/15 cut
    That is a different RNG and a different partitioning scheme. It produces a
    different test set.

CONSEQUENCE: XGBoost's 0.898 was measured on a different set of stars from the
CNN's 0.945. Those two numbers were never comparable. A reviewer who checks this
finds the baseline is not a baseline.

WHAT THIS SCRIPT DOES
---------------------
Calls make_splits() (the CNN's split), converts row indices into kepids, and
writes split.json. Converting indices to kepids is lossless here precisely
because the split is *grouped* by kepid: every example belonging to a star is
guaranteed to sit in the same partition, so a kepid unambiguously names one
partition.

USAGE
-----
    PYTHONPATH=src python scripts/dump_split.py

Then re-run the baseline, which will now pick up split.json:

    PYTHONPATH=src python scripts/train_xgboost_baseline.py

Expect the 0.898 to move. Whatever it becomes is the number that belongs in the
paper.
"""
import json
from pathlib import Path

import h5py
import numpy as np

from stellaris.dataset import make_splits, DEFAULT_DATASET_PATH

OUT_PATH = Path("data/processed/split.json")


def main() -> None:
    # This is the CNN's split. Deterministic given seed=42, so calling it now
    # reproduces exactly the partition that trained stellaris_best.pt.
    splits = make_splits()

    # make_splits gives row indices; we need the kepid sitting at each row.
    with h5py.File(DEFAULT_DATASET_PATH, "r") as f:
        kepids = np.asarray(f["kepid"][:])

    out = {}
    for name, idx in splits.items():
        # set() collapses the many rows per star down to unique star IDs.
        # sorted() only makes the file readable and diff-friendly.
        out[name] = sorted({int(kepids[i]) for i in idx})

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote {OUT_PATH}")
    for name, idx in splits.items():
        print(f"  {name:5s}: {len(idx):5d} examples across {len(out[name]):5d} stars")

    # Safety check: the whole point is that no star is shared between partitions.
    # If this ever fires, the grouping has broken and nothing downstream is valid.
    tr, va, te = set(out["train"]), set(out["val"]), set(out["test"])
    assert not (tr & va) and not (tr & te) and not (va & te), "kepid leak between splits"
    print("\nNo kepid appears in more than one split. Good.")

    # Sanity: the test partition should be the 1,020 examples you already report.
    n_test = len(splits["test"])
    if n_test != 1020:
        print(f"\nWARNING: test set has {n_test} examples, expected 1020.")
        print("Your published test metrics were computed on 1020. Investigate before using.")


if __name__ == "__main__":
    main()
