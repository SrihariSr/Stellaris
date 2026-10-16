"""Check branch_knockout.py: the swap permutation and that each condition touches only its own view."""
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from branch_knockout import CONDITIONS, condition_views, derangement, knockout_rows  # noqa: E402
from stellaris.model import StellarisNetwork  # noqa: E402

failures: list[str] = []


def expect(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'ok' if ok else 'FAIL'}] {label}{(': ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


def main() -> int:
    perm = derangement(1020, 0)
    expect("derangement is a permutation", np.array_equal(np.sort(perm), np.arange(1020)))
    expect("derangement has no fixed points", not np.any(perm == np.arange(1020)))
    expect("derangement is deterministic for a seed", np.array_equal(perm, derangement(1020, 0)))
    expect("two examples swap", list(derangement(2, 0)) == [1, 0])

    rng = np.random.default_rng(1)
    n = 40
    g = rng.normal(size=(n, 2001)).astype(np.float32)
    l = rng.normal(size=(n, 201)).astype(np.float32)
    perm = derangement(n, 0)
    for condition in CONDITIONS:
        cg, cl = condition_views(g, l, condition, perm)
        want_g = np.zeros_like(g) if condition == "global_zero" else g[perm] if condition == "global_shuffle" else g
        want_l = np.zeros_like(l) if condition == "local_zero" else l[perm] if condition == "local_shuffle" else l
        expect(f"{condition} changes only its own view", np.array_equal(cg, want_g) and np.array_equal(cl, want_l))

    labels = np.tile([0.0, 1.0], n // 2).astype(np.float32)
    for kwargs, untouched in (({"use_global": False}, "global"), ({"use_local": False}, "local")):
        torch.manual_seed(0)
        model = StellarisNetwork(**kwargs).eval()
        rows, _ = knockout_rows(model, {"group": "t", "variant": "t"}, "t", g, l, labels, perm, 0, "cpu", 16)
        by = {r["condition"]: r for r in rows}
        absent_zero = all(by[f"{untouched}_{a}"]["max_abs_dlogit"] == 0.0 for a in ("zero", "shuffle"))
        other = "local" if untouched == "global" else "global"
        present_moves = all(by[f"{other}_{a}"]["max_abs_dlogit"] > 0.0 for a in ("zero", "shuffle"))
        expect(f"{kwargs}: knocking out the absent {untouched} branch changes nothing", absent_zero)
        expect(f"{kwargs}: knocking out the {other} branch changes the logits", present_moves)
        expect(f"{kwargs}: intact row has zero deltas and zero drop",
               by["intact"]["max_abs_dlogit"] == 0.0 and by["intact"]["pr_auc_drop"] == 0.0)

    print("PASS" if not failures else f"FAIL ({len(failures)} checks)")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
