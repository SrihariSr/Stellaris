"""Check --kepler-dirs: MAST copies give v1-identical views, the search order is honoured, placeholders stay shut."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stellaris.fetch import KEPLER_DATA_DIR  # noqa: E402
from stellaris.kepler_local import fully_local, llc_files  # noqa: E402
from stellaris.kepler_v1 import KOI_CSV, V1_PATH, v1_csv_rows  # noqa: E402

MAST_DIR = PROJECT_ROOT / ".cr_scratch" / "smoke" / "kepler_mast"
SMOKE_KEPIDS = [10797460, 10848459]
WORK = PROJECT_ROOT / ".cr_scratch" / "smoke" / "test_kepler_dirs"
SF_DATALESS = 0x40000000

failures: list[str] = []


def expect(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'ok' if ok else 'FAIL'}] {label}{(': ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


def run(*args: str) -> tuple[int, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("NUMBA_CACHE_DIR", str(PROJECT_ROOT / ".cr_scratch" / "numba"))
    env.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cr_scratch" / "mpl"))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run([sys.executable, *args], cwd=PROJECT_ROOT, env=env, capture_output=True, text=True)
    return proc.returncode, proc.stdout + proc.stderr


def flags(kepid: int) -> list[int]:
    return [os.stat(f).st_flags & SF_DATALESS for f in llc_files(kepid, KEPLER_DATA_DIR)]


def main() -> int:
    if not V1_PATH.exists() or not all(fully_local(k, MAST_DIR) for k in SMOKE_KEPIDS):
        print(f"SKIP: run scripts/fetch_kepler_mast.py --out-dir {MAST_DIR} --kepids "
              + " ".join(map(str, SMOKE_KEPIDS)) + " first")
        return 0
    koi = pd.read_csv(KOI_CSV)
    v1_rows, _ = v1_csv_rows(koi)
    v1_index = {int(r): j for j, r in enumerate(v1_rows)}
    kepid_of = koi.kepid.to_numpy()
    smoke_rows = [int(r) for r in v1_rows if kepid_of[r] in SMOKE_KEPIDS]
    local_row = next(int(r) for r in v1_rows if fully_local(int(kepid_of[r]), KEPLER_DATA_DIR)
                     and not (MAST_DIR / f"{int(kepid_of[r]):09d}").exists())
    shut_row = next(int(r) for r in v1_rows if not fully_local(int(kepid_of[r]), KEPLER_DATA_DIR)
                    and not (MAST_DIR / f"{int(kepid_of[r]):09d}").exists())
    rows = sorted(smoke_rows + [local_row])
    print(f"  smoke kepids {SMOKE_KEPIDS} -> v1 CSV rows {smoke_rows}; fully local row {local_row} "
          f"(kepid {kepid_of[local_row]}); cloud-only everywhere row {shut_row} (kepid {kepid_of[shut_row]})")
    expect("smoke kepids are cloud-only in data/raw/kepler", all(all(flags(k)) for k in SMOKE_KEPIDS))
    flags_before = {k: flags(k) for k in SMOKE_KEPIDS + [int(kepid_of[shut_row])]}

    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    with h5py.File(V1_PATH, "r") as f:
        v1 = {k: f[k][:] for k in ("kepid", "label", "global_view", "local_view")}

    builds = {}
    for name, dirs in (("mast_first", [MAST_DIR, KEPLER_DATA_DIR]), ("raw_first", [KEPLER_DATA_DIR, MAST_DIR])):
        out = WORK / f"v2_{name}.h5"
        code, log = run("scripts/build_kepler_trainset_v2.py", "--rows", *map(str, rows), "--workers", "2",
                        "--out", str(out), "--kepler-dirs", *map(str, dirs))
        expect(f"{name}: build exits 0", code == 0, log.strip().splitlines()[-1] if log.strip() else "")
        with h5py.File(out, "r") as f:
            v2 = {k: f[k][:] for k in ("csv_row", "kepid", "label", "global_view", "local_view")}
            v2["kepler_dir"] = [d.decode() if isinstance(d, bytes) else d for d in f["kepler_dir"][:]]
        builds[name] = v2
        expect(f"{name}: exactly the requested v1 rows in v1 order", list(v2["csv_row"]) == rows)
        want_dirs = [str(KEPLER_DATA_DIR.resolve()) if r == local_row else str(MAST_DIR.resolve()) for r in rows]
        expect(f"{name}: each row read from the first fully local copy", v2["kepler_dir"] == want_dirs,
               ", ".join(f"{r}:{Path(d).parent.name}/{Path(d).name}" for r, d in zip(rows, v2["kepler_dir"])))
        positions = np.array([v1_index[r] for r in rows])
        expect(f"{name}: kepid, label and both views bit-identical to v1",
               all(np.array_equal(v1[k][positions], v2[k]) for k in ("kepid", "label", "global_view", "local_view")))

    out = WORK / "v2_shut.h5"
    code, log = run("scripts/build_kepler_trainset_v2.py", "--rows", str(shut_row), "--workers", "1",
                    "--out", str(out), "--kepler-dirs", str(MAST_DIR), str(KEPLER_DATA_DIR))
    missing = pd.read_csv(WORK / "v2_shut.h5.missing_rows.csv")
    expect("a kepid with no fully local copy fails without a file", code != 0 and not out.exists()
           and list(missing.csv_row) == [shut_row] and missing.message.str.contains("no fully local copy").all(),
           missing.message.iloc[0] if len(missing) else "")
    expect("placeholders were never opened (flags unchanged)", {k: flags(k) for k in flags_before} == flags_before)

    print("PASS" if not failures else f"FAIL ({len(failures)} checks)")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
