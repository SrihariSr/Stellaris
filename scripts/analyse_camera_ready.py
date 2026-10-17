"""Camera-ready analyses: stage sensitivity, training variants, rescoring, injection-recovery, depth probe, negatives."""
from pathlib import Path
import argparse
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from stellaris.rescore import CATALOGUE_DISPOSITIONS, load_toi_table, select_toi
from stellaris.targets import PROJECT_ROOT, REGIME_CSVS

DEFAULT_OUT = PROJECT_ROOT / "results" / "camera_ready" / "analysis"
DEFAULT_FIG = PROJECT_ROOT / "figures" / "camera_ready"
TIER1_FAILURES = PROJECT_ROOT / "results" / "tier1_failures.log"

TESS_ORBIT_DAYS = 13.7
BAND_FRAC = 0.10
MATCH_TOL = 0.02
ODD_EVEN_TOLERANCES = [15, 20, 25, 30, 40, 50, 75, 100, np.inf]
ODD_EVEN = re.compile(r"odd/even depth differ by ([\d.]+)%")
COLLAPSE_LOGIT_STD = 1e-3
TEST_METRICS = ["pr_auc", "roc_auc", "recall_at_99", "recall_at_95", "recall_at_90"]

# Reference categorical order (blue, orange, aqua, ...); text stays in neutral ink.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#dcdbd6"


def style(ax, xlabel: str, ylabel: str, title: str | None = None) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelcolor=INK)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel(ylabel, color=INK)
    if title:
        ax.set_title(title, color=INK, fontsize=10, loc="left")


def save(fig, fig_dir: Path, name: str) -> Path:
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def write_table(frame: pd.DataFrame, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    frame.to_csv(path, index=False)
    return path


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().map({"true": True, "false": False})


def load_hunt(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["vetted"] = as_bool(frame.vetting_passed)
    return frame


def parse_failure_log(path: Path) -> list[tuple[int, str]]:
    """Records start with '<tic>\\t'; any other line continues the previous record."""
    records: list[list] = []
    for line in path.read_text().splitlines():
        match = re.match(r"^(\d+)\t(.*)$", line)
        if match:
            records.append([int(match.group(1)), match.group(2)])
        elif records:
            records[-1][1] += "\n" + line
    return [(t, m) for t, m in records]


def loss_category(message: str) -> str:
    if "exceeds phase range" in message or "Local window" in message:
        return "view_construction"
    if "Connection aborted" in message or "RemoteDisconnected" in message:
        return "connection"
    if "Download of" in message or "HTTPError" in message:
        return "download"
    if not message.strip():
        return "empty_message"
    return "other"


def crossover_threshold(probs: np.ndarray, vetting_fr: float) -> float:
    for t in np.round(np.linspace(0, 1, 1001), 3):
        if np.mean(probs < t) >= vetting_fr:
            return float(t)
    return float("nan")


def period_relation(bls_period: float, catalogue_periods: np.ndarray) -> str:
    """match, half, double, third, triple or other, each within 2%, in that priority."""
    if len(catalogue_periods) == 0:
        return "other"
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = bls_period / np.asarray(catalogue_periods, dtype=float)
    for name, factor in (("match", 1.0), ("half", 0.5), ("double", 2.0), ("third", 1 / 3), ("triple", 3.0)):
        if np.any(np.abs(ratio / factor - 1.0) <= MATCH_TOL):
            return name
    return "other"


UNRECOVERABLE_ORDER = ["single-transit only", "beyond 20 d", "beyond 20 d and 8 h", "beyond 8 h only", "mixed"]


def unrecoverable_category(periods: pd.Series, durations: pd.Series) -> str | None:
    """
    None if any catalogued planet sits inside the search box (0 < P <= 20 d and
    T14 <= 8 h). A period <= 0 is a single transit and counts as beyond 20 d
    when the system also has periodic planets.
    """
    single = periods <= 0
    beyond_period = single | (periods > 20)
    beyond_duration = durations > 8
    if len(periods) == 0 or not (beyond_period | beyond_duration).all():
        return None
    if single.all():
        return "single-transit only"
    if beyond_period.all() and beyond_duration.all():
        return "beyond 20 d and 8 h"
    if beyond_period.all():
        return "beyond 20 d"
    if beyond_duration.all():
        return "beyond 8 h only"
    return "mixed"


def mean_sd(values) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan")
    return float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else float("nan")


def welch_p(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    return float(stats.ttest_ind(a, b, equal_var=False).pvalue)


def welch_ci(a, b, level: float = 0.95) -> dict:
    """Difference of means a - b with its Welch-Satterthwaite confidence interval and two-sided p."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    out = {"n_a": len(a), "n_b": len(b), "mean_a": float(a.mean()) if len(a) else float("nan"),
           "mean_b": float(b.mean()) if len(b) else float("nan")}
    out["difference"] = out["mean_a"] - out["mean_b"]
    if len(a) < 2 or len(b) < 2:
        return {**out, "ci_low": float("nan"), "ci_high": float("nan"), "df": float("nan"), "p": float("nan")}
    var_a, var_b = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    err = np.sqrt(var_a + var_b)
    if err == 0:
        return {**out, "ci_low": out["difference"], "ci_high": out["difference"], "df": float("nan"), "p": float("nan")}
    df = (var_a + var_b) ** 2 / (var_a ** 2 / (len(a) - 1) + var_b ** 2 / (len(b) - 1))
    half = stats.t.ppf(0.5 + level / 2, df) * err
    p = 2 * stats.t.sf(abs(out["difference"]) / err, df)
    return {**out, "ci_low": out["difference"] - half, "ci_high": out["difference"] + half,
            "df": float(df), "p": float(p)}


def sequential_table(frame: pd.DataFrame, value_cols: list[str], pairs: list[tuple]) -> pd.DataFrame:
    """Welch comparisons for (label, later rows, earlier rows) pairs on each value column."""
    rows = []
    for label, later, earlier in pairs:
        for col in value_cols:
            r = welch_ci(frame.loc[later, col], frame.loc[earlier, col])
            rows.append({"comparison": label, "metric": col, "mean_later": r["mean_a"], "mean_earlier": r["mean_b"],
                         "difference": r["difference"], "ci95_low": r["ci_low"], "ci95_high": r["ci_high"],
                         "df": r["df"], "p": r["p"], "n_later": r["n_a"], "n_earlier": r["n_b"]})
    return pd.DataFrame(rows)


def print_sequential(table: pd.DataFrame) -> None:
    for r in table.itertuples():
        print(f"  {r.comparison:<22} {r.metric:<26} diff {r.difference:+.4f}  "
              f"95% CI [{r.ci95_low:+.4f}, {r.ci95_high:+.4f}]  p {r.p:.3g}")


def mcnemar_exact(b: int, c: int) -> float:
    return float(stats.binomtest(min(b, c), b + c, 0.5).pvalue) if b + c > 0 else 1.0


def cmd_stages(args) -> None:
    r1 = load_hunt(args.r1)
    toi = load_toi_table(args.toi_table)
    probs = r1.cnn_probability.to_numpy()
    n = len(r1)

    thresholds = np.round(np.arange(0.05, 0.951, 0.05), 2)
    fn = pd.DataFrame({"threshold": thresholds,
                       "n_fn": [int((probs < t).sum()) for t in thresholds],
                       "fn_rate": [float((probs < t).mean()) for t in thresholds]})
    write_table(fn, args.out_dir, "stages_classifier_fn.csv")

    reasons = r1.vetting_reasons.fillna("")
    depth_cap = reasons.str.contains("cap")
    odd_even_pct = reasons.map(lambda s: float(m.group(1)) if (m := ODD_EVEN.search(s)) else np.nan)
    fr_rows = []
    for tol in ODD_EVEN_TOLERANCES:
        if tol == 15:
            rejected = ~r1.vetted
        else:
            rejected = depth_cap | (odd_even_pct > tol)
        fr_rows.append({"odd_even_tolerance_pct": tol, "n_rejected": int(rejected.sum()),
                        "fr_rate": float(rejected.mean()),
                        "n_odd_even_rejected": int((odd_even_pct > tol).sum()) if tol != 15 else int(reasons.str.contains("odd/even").sum()),
                        "n_depth_cap": int(depth_cap.sum())})
    fr = pd.DataFrame(fr_rows)
    write_table(fr, args.out_dir, "stages_vetting_fr.csv")
    parsed_15 = depth_cap | (odd_even_pct > 15)
    parsed_agreement = int((parsed_15 == ~r1.vetted).sum())

    vetting_fr = float((~r1.vetted).mean())
    crossover = crossover_threshold(probs, vetting_fr)

    confirmed = toi[toi["TFOPWG Disposition"].isin(CATALOGUE_DISPOSITIONS["r1"])]
    periods = confirmed.groupby("TIC ID")["Period (days)"].apply(lambda s: s.to_numpy())
    r1["relation"] = [period_relation(p, periods.get(t, np.array([]))) for t, p in zip(r1.tic_id, r1.period_days)]
    r1["classifier_fail"] = r1.cnn_probability < 0.5
    r1["odd_even_fail"] = reasons.str.contains("odd/even")
    r1["vetting_fail"] = ~r1.vetted
    order = ["match", "half", "double", "third", "triple", "other"]
    relation = (r1.groupby("relation")
                .agg(n=("tic_id", "size"), classifier_fail=("classifier_fail", "sum"),
                     odd_even_fail=("odd_even_fail", "sum"), vetting_fail=("vetting_fail", "sum"))
                .reindex(order, fill_value=0).reset_index())
    for col in ("classifier_fail", "odd_even_fail", "vetting_fail"):
        relation[f"{col}_rate"] = relation[col] / relation.n.replace(0, np.nan)
    write_table(relation, args.out_dir, "stages_period_relation.csv")
    crosstab = pd.crosstab([r1.relation, r1.classifier_fail], r1.odd_even_fail).reindex(order, level=0)
    crosstab.columns = [f"odd_even_fail={c}" for c in crosstab.columns]
    write_table(crosstab.reset_index(), args.out_dir, "stages_period_crosstab.csv")

    unrecoverable_rows = []
    for tic in r1.loc[r1.relation != "match", "tic_id"]:
        planets = confirmed[confirmed["TIC ID"] == tic]
        category = unrecoverable_category(planets["Period (days)"], planets["Duration (hours)"])
        if category is not None:
            unrecoverable_rows.append({
                "tic_id": tic, "category": category, "n_planets": len(planets),
                "periods_days": ";".join(f"{v:g}" for v in planets["Period (days)"]),
                "durations_hours": ";".join(f"{v:g}" for v in planets["Duration (hours)"]),
            })
    unrecoverable = pd.DataFrame(unrecoverable_rows, columns=["tic_id", "category", "n_planets",
                                                                "periods_days", "durations_hours"])
    write_table(unrecoverable, args.out_dir, "stages_unrecoverable.csv")
    unrecoverable_counts = {c: int((unrecoverable.category == c).sum()) for c in UNRECOVERABLE_ORDER}

    records = parse_failure_log(args.r1_failures)
    categories = pd.Series([loss_category(m) for _, m in records], dtype=object)
    losses = categories.value_counts().rename_axis("category").reset_index(name="n")
    write_table(losses, args.out_dir, "stages_view_losses.csv")
    n_view = int((categories == "view_construction").sum())
    n_other = len(records) - n_view
    endorsed = int(((r1.cnn_probability >= 0.5) & r1.vetted).sum())

    summary = {
        "n_r1": n,
        "classifier_fn_rate_at_0.5": float((probs < 0.5).mean()),
        "classifier_fn_rate_at_0.3": float((probs < 0.3).mean()),
        "vetting_fr_rate": vetting_fr,
        "crossover_threshold": crossover,
        "parsed_odd_even_at_15_agrees_with_flag": f"{parsed_agreement}/{n}",
        "bls_period_matches": int((r1.relation == "match").sum()),
        "bls_misses": int((r1.relation != "match").sum()),
        "unrecoverable_by_construction": len(unrecoverable),
        "unrecoverable_categories": unrecoverable_counts,
        "view_construction_losses": n_view,
        "other_losses": n_other,
        "endorsed_at_0.5": endorsed,
        "endorsed_over_r1_plus_view_losses": endorsed / (n + n_view),
    }
    (args.out_dir / "stages_summary.json").write_text(json.dumps(summary, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].plot(fn.threshold, fn.fn_rate * 100, color=SERIES[0], linewidth=1.5, marker="o", markersize=3)
    axes[0].axhline(vetting_fr * 100, color=INK_MUTED, linewidth=0.8, linestyle="--")
    axes[0].axvline(crossover, color=INK_MUTED, linewidth=0.8, linestyle=":")
    axes[0].annotate(f"vetting FR {vetting_fr * 100:.1f}%", (0.06, vetting_fr * 100 + 1.5), color=INK_MUTED, fontsize=8)
    axes[0].annotate(f"crossover {crossover:.3f}", (crossover + 0.01, 2), color=INK_MUTED, fontsize=8)
    axes[0].set_ylim(0, None)
    style(axes[0], "classifier threshold", "R1 classifier false-negative rate (%)", "Classifier stage")
    labels = ["inf" if not np.isfinite(t) else f"{t:g}" for t in fr.odd_even_tolerance_pct]
    axes[1].bar(range(len(fr)), fr.fr_rate * 100, color=SERIES[1], width=0.7, edgecolor="white", linewidth=1)
    axes[1].set_xticks(range(len(fr)), labels)
    style(axes[1], "odd-even tolerance (%), depth cap fixed", "R1 vetting false-rejection rate (%)", "Vetting stage")
    print(f"Figure: {save(fig, args.fig_dir, 'stage_sensitivity.png')}")

    print(f"R1 classifier FN at 0.5: {summary['classifier_fn_rate_at_0.5'] * 100:.1f}%  at 0.3: "
          f"{summary['classifier_fn_rate_at_0.3'] * 100:.1f}%")
    print(f"R1 vetting FR: {vetting_fr * 100:.1f}%   crossover threshold: {crossover:.3f}")
    print(f"Parsed odd-even rule at 15% agrees with the stored flag on {parsed_agreement}/{n}")
    print(f"BLS period matches (2%): {summary['bls_period_matches']}   misses: {summary['bls_misses']}")
    print(f"Unrecoverable by construction: {len(unrecoverable)} ("
          + ", ".join(f"{c}: {n}" for c, n in unrecoverable_counts.items()) + ")")
    print(f"View-construction losses: {n_view}   other losses: {n_other}   "
          f"endorsed/(R1 + view losses): {endorsed}/{n + n_view} = {summary['endorsed_over_r1_plus_view_losses'] * 100:.1f}%")
    print(f"Tables: {args.out_dir}")


def load_runs(training_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted((training_dir / "runs").glob("*/*.json")):
        run = json.loads(path.read_text())
        spec, best = run["spec"], run.get("best_val", {})
        rows.append({
            "group": spec["group"], "variant": spec["variant"], "seed": spec["seed"],
            "lr": spec.get("lr"), "batch_norm": spec.get("batch_norm"), "pool": spec.get("pool"),
            "n_params": run["n_params"], "best_epoch": run["best_epoch"], "val_pr_auc": run["val_pr_auc"],
            "best_val_logit_std": best.get("logit_std", np.nan),
            "n_collapsed_epochs": sum(1 for h in run["history"] if h["val_logit_std"] < COLLAPSE_LOGIT_STD),
            "history": run["history"],
            **{f"test_{k}": run["test"][k] for k in TEST_METRICS},
        })
    return pd.DataFrame(rows)


def cmd_training(args) -> None:
    runs = load_runs(args.training_dir)
    if runs.empty:
        print(f"No runs under {args.training_dir / 'runs'}")
        return
    runs["collapsed"] = runs.best_val_logit_std < COLLAPSE_LOGIT_STD
    write_table(runs[runs.collapsed].drop(columns="history"), args.out_dir, "training_collapsed.csv")

    for group, frame in runs.groupby("group"):
        reference = {"scalars": "A", "ablation": "both", "ablation-long": "both", "bn-ablation": "both"}.get(group)
        ref = frame[frame.variant == reference] if reference else frame.iloc[:0]
        rows = []
        for variant, sub in frame.groupby("variant", sort=False):
            row = {"group": group, "variant": variant, "n_seeds": len(sub), "n_params": int(sub.n_params.iloc[0]),
                   "best_epoch_mean": float(sub.best_epoch.mean()), "n_collapsed": int(sub.collapsed.sum())}
            for metric in TEST_METRICS:
                m, s = mean_sd(sub[f"test_{metric}"])
                row[f"test_{metric}_mean"], row[f"test_{metric}_sd"] = m, s
                if not ref.empty and variant != reference:
                    row[f"welch_p_{metric}_vs_{reference}"] = welch_p(sub[f"test_{metric}"], ref[f"test_{metric}"])
            rows.append(row)
        table = pd.DataFrame(rows)
        write_table(table, args.out_dir, f"training_summary_{group}.csv")
        print(f"\n[{group}] Welch tests against: {reference or 'none (grid)'}")
        for r in rows:
            print(f"  {r['variant']:<22} n={r['n_seeds']:<3} PR-AUC {r['test_pr_auc_mean']:.4f} +/- {r['test_pr_auc_sd']:.4f}  "
                  f"ROC {r['test_roc_auc_mean']:.4f} +/- {r['test_roc_auc_sd']:.4f}  collapsed {r['n_collapsed']}")
            if reference and r["variant"] != reference:
                print(f"  {'':<22} Welch p, {reference} vs {r['variant']}: " + ", ".join(
                    f"{m} {r.get(f'welch_p_{m}_vs_{reference}', float('nan')):.3g}"
                    for m in ("pr_auc", "roc_auc", "recall_at_95")))

        if group == "global-tuning":
            grid = (table.assign(lr=[frame[frame.variant == v].lr.iloc[0] for v in table.variant],
                                 batch_norm=[frame[frame.variant == v].batch_norm.iloc[0] for v in table.variant],
                                 pool=[frame[frame.variant == v].pool.iloc[0] for v in table.variant])
                    [["lr", "batch_norm", "pool", "n_seeds", "n_params", "test_pr_auc_mean", "test_pr_auc_sd",
                      "test_roc_auc_mean", "test_roc_auc_sd", "n_collapsed"]]
                    .sort_values(["lr", "batch_norm", "pool"]))
            write_table(grid, args.out_dir, "training_global_tuning_grid.csv")

        if group in ("ablation", "ablation-long", "bn-ablation"):
            fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
            for colour, (variant, sub) in zip(SERIES, frame.groupby("variant", sort=False)):
                epochs = max(len(h) for h in sub.history)
                for ax, key in ((axes[0], "val_pr_auc"), (axes[1], "train_loss")):
                    curves = np.full((len(sub), epochs), np.nan)
                    for i, history in enumerate(sub.history):
                        curves[i, :len(history)] = [h[key] for h in history]
                    x = np.arange(1, epochs + 1)
                    mean = np.nanmean(curves, axis=0)
                    sd = np.nanstd(curves, axis=0, ddof=1) if len(sub) > 1 else np.zeros(epochs)
                    ax.plot(x, mean, color=colour, linewidth=1.5, label=f"{variant} (n={len(sub)})")
                    ax.fill_between(x, mean - sd, mean + sd, color=colour, alpha=0.18, linewidth=0)
            style(axes[0], "epoch", "validation PR-AUC", f"{group}: mean +/- s.d. over seeds")
            style(axes[1], "epoch", "training loss")
            axes[0].legend(frameon=False, fontsize=8, labelcolor=INK)
            print(f"  Figure: {save(fig, args.fig_dir, f'learning_curves_{group}.png')}")

    scalars = runs[runs.group == "scalars"]
    if not scalars.empty:
        print("\n[scalars] test recall at 95% precision, mean +/- s.d. over seeds:")
        for variant in ("A", "B", "C", "D"):
            m, s = mean_sd(scalars[scalars.variant == variant].test_recall_at_95)
            print(f"  {variant}  {m:.4f} +/- {s:.4f}")
        pairs = [(f"{b} vs {a}", scalars.index[scalars.variant == b], scalars.index[scalars.variant == a])
                 for b, a in (("B", "A"), ("C", "B"), ("D", "C"))]
        table = sequential_table(scalars, ["test_pr_auc", "test_roc_auc", "test_recall_at_95"], pairs)
        write_table(table, args.out_dir, "scalars_sequential_kepler.csv")
        print("[scalars] sequential Welch comparisons (later minus earlier):")
        print_sequential(table)
    print(f"\nTables: {args.out_dir}")


def checkpoint_group(label: str, variant: str) -> str:
    """train_variants writes <ckpt-dir>/<group>/<variant>__seed<k>.pt by default."""
    return "released" if variant == "released" else Path(str(label)).parent.name


def load_rescore(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    frame = frame[frame.status == "ok"].drop_duplicates(["tic_id", "checkpoint"], keep="last").copy()
    frame["vetted"] = as_bool(frame.vetting_passed)
    frame["group"] = [checkpoint_group(c, v) for c, v in zip(frame.checkpoint, frame.variant)]
    return frame


def matched_threshold(probs: np.ndarray, pass_rate: float) -> float:
    """The threshold whose pass rate (prob >= t) equals pass_rate."""
    probs = np.sort(np.asarray(probs, dtype=float))[::-1]
    k = int(round(pass_rate * len(probs)))
    return float(probs[k - 1]) if k >= 1 else float(probs[0] + 1e-9)


def cmd_rescore(args) -> None:
    tag = args.ephemeris_tag
    r1 = load_rescore(args.rescore_dir / f"r1_{tag}.csv")
    r2 = load_rescore(args.rescore_dir / f"r2_{tag}.csv")
    r3 = load_rescore(args.rescore_dir / f"r3_{tag}.csv")
    r1_cat = load_rescore(args.rescore_dir / "r1_catalogue.csv")
    if r1 is None:
        print(f"Missing {args.rescore_dir / f'r1_{tag}.csv'}; the threshold matching needs R1.")
        return

    released = r1[r1.variant == "released"]
    if len(released):
        pass_rate = float((released.prob >= 0.5).mean())
        rate_source = "released checkpoint rescored on R1"
    else:
        pass_rate = float((pd.read_csv(REGIME_CSVS["r1"]).cnn_probability >= 0.5).mean())
        rate_source = "tier1_results.csv cnn_probability"
    print(f"Released R1 pass rate at 0.5: {pass_rate * 100:.1f}% ({rate_source})")

    toi = load_toi_table(args.toi_table)
    confirmed = toi[toi["TFOPWG Disposition"].isin(CATALOGUE_DISPOSITIONS["r1"])]
    by_tic = {tic: rows for tic, rows in confirmed.groupby("TIC ID")}
    cat_depth = {}
    for tic, period in r1.drop_duplicates("tic_id")[["tic_id", "period_days"]].itertuples(index=False):
        chosen, _ = select_toi(by_tic[tic], period) if tic in by_tic else (None, "none")
        cat_depth[tic] = float(chosen["Depth (ppm)"]) if chosen is not None else np.nan
    depth_bins = [(-np.inf, 1000, "<1000"), (1000, 5000, "1000-5000"), (5000, np.inf, ">5000")]

    rows = []
    for checkpoint, sub1 in r1.groupby("checkpoint"):
        variant, seed, group = sub1.variant.iloc[0], sub1.seed.iloc[0], sub1.group.iloc[0]
        t_star = 0.5 if variant == "released" else matched_threshold(sub1.prob, pass_rate)
        row = {"checkpoint": checkpoint, "group": group, "variant": variant, "seed": seed,
               "t_star": t_star, "r1_pass_rate": float((sub1.prob >= t_star).mean()), "r1_n": len(sub1)}
        depth = sub1.tic_id.map(cat_depth)
        for lo, hi, name in depth_bins:
            in_bin = (depth >= lo) & (depth < hi) if np.isfinite(hi) else depth >= lo
            row[f"r1_pass_rate_depth_{name}"] = float((sub1.prob[in_bin] >= t_star).mean()) if in_bin.any() else np.nan
            row[f"r1_n_depth_{name}"] = int(in_bin.sum())
        if r3 is not None and checkpoint in set(r3.checkpoint):
            sub3 = r3[r3.checkpoint == checkpoint]
            band = (sub3.period_days - TESS_ORBIT_DAYS).abs() / TESS_ORBIT_DAYS <= BAND_FRAC
            high = sub3.prob >= t_star
            row.update({
                "r3_median_prob_band": float(sub3.prob[band].median()) if band.any() else np.nan,
                "r3_median_prob_elsewhere": float(sub3.prob[~band].median()) if (~band).any() else np.nan,
                "r3_mannwhitney_p": float(stats.mannwhitneyu(sub3.prob[band], sub3.prob[~band]).pvalue)
                if band.any() and (~band).any() else np.nan,
                "r3_n_band": int(band.sum()), "r3_n_elsewhere": int((~band).sum()),
                "r3_high_band": int((high & band).sum()), "r3_high_band_vetted": int((high & band & sub3.vetted).sum()),
                "r3_high_elsewhere": int((high & ~band).sum()),
                "r3_high_elsewhere_vetted": int((high & ~band & sub3.vetted).sum()),
            })
        if r2 is not None and checkpoint in set(r2.checkpoint):
            sub2 = r2[r2.checkpoint == checkpoint]
            cap = sub2.vetting_reasons.fillna("").str.contains("cap")
            row.update({"r2_depth_cap_rejections": int(cap.sum()),
                        "r2_high_among_depth_cap": int((cap & (sub2.prob >= t_star)).sum())})
        if r1_cat is not None and checkpoint in set(r1_cat.checkpoint):
            both = sub1.merge(r1_cat[r1_cat.checkpoint == checkpoint][["tic_id", "prob"]],
                              on="tic_id", suffixes=("_bls", "_cat"))
            fn_bls, fn_cat = both.prob_bls < t_star, both.prob_cat < t_star
            b, c = int((fn_bls & ~fn_cat).sum()), int((~fn_bls & fn_cat).sum())
            row.update({"cat_vs_bls_n": len(both), "fn_rate_bls": float(fn_bls.mean()) if len(both) else np.nan,
                        "fn_rate_catalogue": float(fn_cat.mean()) if len(both) else np.nan,
                        "fn_bls_only": b, "fn_catalogue_only": c, "mcnemar_exact_p": mcnemar_exact(b, c)})
        rows.append(row)
    per_ckpt = pd.DataFrame(rows)
    write_table(per_ckpt, args.out_dir, f"rescore_per_checkpoint_{tag}.csv")

    numeric = [c for c in per_ckpt.columns if c not in ("checkpoint", "group", "variant", "seed")
               and pd.api.types.is_numeric_dtype(per_ckpt[c])]
    agg = per_ckpt.groupby(["group", "variant"])[numeric].agg(["mean", lambda s: s.std(ddof=1)])
    agg.columns = [f"{c}_{'mean' if s == 'mean' else 'sd'}" for c, s in agg.columns]
    agg.insert(0, "n_seeds", per_ckpt.groupby(["group", "variant"]).size())
    agg = agg.reset_index()
    write_table(agg, args.out_dir, f"rescore_by_variant_{tag}.csv")
    focus = agg[(agg.group == "scalars") | agg.variant.isin(["released", "both", "local-only"])]
    means = ["t_star_mean", "r1_pass_rate_mean"] + [c for c in ("r3_high_band_mean", "r2_high_among_depth_cap_mean",
                                                                 "fn_rate_catalogue_mean") if c in focus]
    print(focus[["group", "variant", "n_seeds"] + [c for m in means for c in (m, m[:-len("_mean")] + "_sd")]]
          .to_string(index=False))

    scalars = per_ckpt[per_ckpt.group == "scalars"]
    values = [c for c in ("r3_high_band", "r2_high_among_depth_cap") if c in per_ckpt]
    if not scalars.empty and values:
        idx = lambda g, v: per_ckpt.index[(per_ckpt.group == g) & (per_ckpt.variant == v)]
        pairs = [(f"{b} vs {a}", idx("scalars", b), idx("scalars", a)) for b, a in (("B", "A"), ("C", "B"), ("D", "C"))]
        pairs.append(("A vs ablation/both", idx("scalars", "A"), idx("ablation", "both")))
        table = sequential_table(per_ckpt, values, pairs)
        write_table(table, args.out_dir, "scalars_sequential_tess.csv")
        print("[scalars] sequential Welch comparisons at matched thresholds (later minus earlier):")
        print_sequential(table)
    print(f"Tables: {args.out_dir}")


def completeness(frame: pd.DataFrame, column: str, edges: list[float], labels: list[str]) -> pd.DataFrame:
    binned = pd.cut(frame[column], edges, labels=labels, right=False)
    return (frame.groupby(binned, observed=False)
            .agg(n=("tic_id", "size"), recovered_05=("recovered_05", "mean"),
                 endorsed_09=("endorsed_09", "mean"), period_match=("period_match", "mean"))
            .rename_axis("bin").reset_index())


def cmd_injection(args) -> None:
    if not args.injection_csv.exists():
        print(f"Missing {args.injection_csv}")
        return
    frame = pd.read_csv(args.injection_csv).drop_duplicates(["tic_id", "injection", "checkpoint"], keep="last")
    for col in ("recovered_05", "endorsed_09", "period_match", "in_grid"):
        frame[col] = as_bool(frame[col]).fillna(False).astype(bool)

    snr_edges = [0, 3, 5, 7, 10, 15, 20, 30, 50, 100, np.inf]
    snr_labels = [f"{a:g}-{b:g}" for a, b in zip(snr_edges[:-1], snr_edges[1:])]
    depth_edges = [500, 1000, 2000, 5000, 10000, 20000, 30001]
    depth_labels = [f"{a:g}-{b:g}" for a, b in zip(depth_edges[:-1], depth_edges[1:])]
    tables = []
    for checkpoint, sub in frame.groupby("checkpoint"):
        for subset, part in (("all", sub), ("in_grid", sub[sub.in_grid])):
            for kind, column, edges, labels in (("expected_snr", "expected_snr", snr_edges, snr_labels),
                                                ("injected_depth_ppm", "inj_depth_ppm", depth_edges, depth_labels)):
                table = completeness(part, column, edges, labels)
                table.insert(0, "binning", kind)
                table.insert(0, "subset", subset)
                table.insert(0, "checkpoint", checkpoint)
                tables.append(table)
            overall = {"checkpoint": checkpoint, "subset": subset, "binning": "overall", "bin": "all",
                       "n": len(part), "recovered_05": part.recovered_05.mean(),
                       "endorsed_09": part.endorsed_09.mean(), "period_match": part.period_match.mean()}
            tables.append(pd.DataFrame([overall]))
    comp = pd.concat(tables, ignore_index=True)
    write_table(comp, args.out_dir, "injection_completeness.csv")

    unique = frame.drop_duplicates(["tic_id", "injection"])
    stage_order = ["no_fits", "preprocess", "cadence_floor", "bls", "view", "scored"]
    stages = pd.DataFrame({
        "stage": stage_order,
        "n_all": [int((unique.stage == s).sum()) for s in stage_order],
        "n_in_grid": [int(((unique.stage == s) & unique.in_grid).sum()) for s in stage_order],
    })
    stages["frac_all"] = stages.n_all / max(1, len(unique))
    stages["frac_in_grid"] = stages.n_in_grid / max(1, int(unique.in_grid.sum()))
    write_table(stages, args.out_dir, "injection_stage_losses.csv")
    print(stages.to_string(index=False))
    overall = comp[comp.binning == "overall"]
    print(overall[["checkpoint", "subset", "n", "recovered_05", "endorsed_09", "period_match"]].to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    axes[0].barh(stages.stage, stages.n_all, color=SERIES[0], height=0.6, edgecolor="white")
    for y, n in enumerate(stages.n_all):
        axes[0].annotate(str(n), (n, y), xytext=(3, 0), textcoords="offset points", va="center", fontsize=8, color=INK)
    axes[0].invert_yaxis()
    style(axes[0], "injections", "", "Stage reached (all injections)")
    released = comp[(comp.binning == "expected_snr") & (comp.subset == "all")]
    first = released.checkpoint.iloc[0] if len(released) else None
    part = released[released.checkpoint == first]
    x = np.arange(len(part))
    for colour, key, name in ((SERIES[0], "period_match", "period match"),
                              (SERIES[1], "recovered_05", "recovered (0.5, vetted)"),
                              (SERIES[2], "endorsed_09", "endorsed (0.9, vetted)")):
        axes[1].plot(x, part[key] * 100, color=colour, linewidth=1.5, marker="o", markersize=3, label=name)
    axes[1].set_xticks(x, part["bin"].astype(str), rotation=45, ha="right", fontsize=7)
    style(axes[1], "expected S/N", "completeness (%)", f"Completeness, {first}")
    axes[1].legend(frameon=False, fontsize=7, labelcolor=INK)
    print(f"Figure: {save(fig, args.fig_dir, 'injection_recovery.png')}")
    print(f"Tables: {args.out_dir}")


def cmd_probe(args) -> None:
    inv_path, sweep_path = args.probe_dir / "invariance.csv", args.probe_dir / "depth_sweep.csv"
    if not inv_path.exists() or not sweep_path.exists():
        print(f"Missing {inv_path} or {sweep_path}")
        return
    inv = pd.read_csv(inv_path)
    inv = inv[inv.status == "ok"].drop_duplicates(["tic_id", "factor", "checkpoint"], keep="last")
    inv["abs_dprob"] = inv.dprob.abs()
    inv_summary = (inv.groupby(["checkpoint", "variant", "factor"])
                   .agg(n=("tic_id", "size"), max_abs_dglobal=("max_abs_dglobal", "max"),
                        max_abs_dlocal=("max_abs_dlocal", "max"), max_abs_dprob=("abs_dprob", "max"),
                        median_abs_dprob=("abs_dprob", "median"))
                   .reset_index())
    write_table(inv_summary, args.out_dir, "probe_invariance.csv")
    print(inv_summary.to_string(index=False))

    sweep = pd.read_csv(sweep_path)
    sweep = sweep[sweep.status == "ok"].drop_duplicates(["tic_id", "target_depth_ppm", "checkpoint"], keep="last")
    q = lambda p: (lambda s: s.quantile(p))
    by_depth = (sweep.groupby(["checkpoint", "variant", "target_depth_ppm"])
                .agg(n=("tic_id", "size"), median_prob=("prob", "median"), q25=("prob", q(0.25)),
                     q75=("prob", q(0.75)), median_local_scale=("local_scale", "median"),
                     median_expected_snr=("expected_snr", "median"))
                .reset_index())
    write_table(by_depth, args.out_dir, "probe_prob_vs_depth.csv")
    snr_edges = [0, 5, 10, 20, 50, 100, 300, 1000, np.inf]
    sweep["snr_bin"] = pd.cut(sweep.expected_snr, snr_edges, right=False).astype(str)
    by_snr = (sweep.groupby(["checkpoint", "variant", "snr_bin"], observed=True)
              .agg(n=("tic_id", "size"), median_prob=("prob", "median")).reset_index())
    write_table(by_snr, args.out_dir, "probe_prob_vs_snr.csv")

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    # The released checkpoint always takes the first colour slot; the rest follow in name order.
    order = sorted(by_depth.checkpoint.unique(), key=lambda c: (not c.endswith("stellaris_best.pt"), c))
    for colour, checkpoint in zip(SERIES, order):
        sub = by_depth[by_depth.checkpoint == checkpoint]
        name = f"{sub.variant.iloc[0]} ({Path(checkpoint).stem})"
        axes[0].plot(sub.target_depth_ppm, sub.median_prob, color=colour, linewidth=1.5, marker="o", markersize=3, label=name)
        axes[0].fill_between(sub.target_depth_ppm, sub.q25, sub.q75, color=colour, alpha=0.15, linewidth=0)
        pts = sweep[sweep.checkpoint == checkpoint]
        axes[1].scatter(pts.expected_snr, pts.prob, s=9, color=colour, alpha=0.6, edgecolors="none", label=name)
    axes[0].set_xscale("log")
    axes[1].set_xscale("log")
    style(axes[0], "injected depth (ppm), fixed noise", "classifier probability", "Median and IQR over targets")
    style(axes[1], "expected S/N", "classifier probability", "All injections")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), frameon=False, fontsize=7,
               labelcolor=INK, bbox_to_anchor=(0.5, 1.08))
    if by_depth.checkpoint.nunique() > len(SERIES):
        print(f"[note] only the first {len(SERIES)} checkpoints are drawn; the tables hold all of them")
    print(f"Figure: {save(fig, args.fig_dir, 'depth_probe.png')}")
    print(f"Tables: {args.out_dir}")


def flag_breakdown(frame: pd.DataFrame) -> dict:
    flags = frame[["fpflag_nt", "fpflag_ss", "fpflag_co", "fpflag_ec"]].astype(int)
    n_flags = flags.sum(axis=1)
    out = {"n_negatives": len(frame)}
    for col in flags:
        out[f"n_{col}"] = int(flags[col].sum())
        out[f"n_only_{col}"] = int(((flags[col] == 1) & (n_flags == 1)).sum())
    out["n_no_flag"] = int((n_flags == 0).sum())
    out["n_multiple_flags"] = int((n_flags > 1).sum())
    return out


def cmd_negatives(args) -> None:
    rows = []
    if args.v2.exists():
        import h5py
        from stellaris.dataset import make_splits

        with h5py.File(args.v2, "r") as f:
            frame = pd.DataFrame({k: f[k][:] for k in ("label", "fpflag_nt", "fpflag_ss", "fpflag_co", "fpflag_ec")})
        splits = make_splits(hdf5_path=args.v2)
        for name in ("train", "val", "test"):
            part = frame.iloc[splits[name]]
            rows.append({"source": f"v2 training set ({args.v2.name})", "split": name,
                         **flag_breakdown(part[part.label == 0])})
        rows.append({"source": f"v2 training set ({args.v2.name})", "split": "all",
                     **flag_breakdown(frame[frame.label == 0])})
    else:
        koi = pd.read_csv(args.koi_csv).rename(columns=lambda c: c.replace("koi_", ""))
        rows.append({"source": f"catalogue-level ({args.koi_csv.name}, all labelled rows; v2 absent)",
                     "split": "catalogue", **flag_breakdown(koi[koi.label == 0])})
    table = pd.DataFrame(rows)
    for col in [c for c in table.columns if c.startswith("n_") and c != "n_negatives"]:
        table[col.replace("n_", "frac_", 1)] = table[col] / table.n_negatives.replace(0, np.nan)
    write_table(table, args.out_dir, "negatives_fp_flags.csv")
    print(table[["source", "split", "n_negatives", "n_fpflag_nt", "n_fpflag_ss", "n_fpflag_co", "n_fpflag_ec",
                 "n_no_flag", "n_multiple_flags"]].to_string(index=False))
    print(f"Tables: {args.out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, func, help_text: str):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--out-dir", type=Path, default=None,
                       help=f"Directory for every file the subcommand writes, tables and figures "
                            f"(default: tables to {DEFAULT_OUT}, figures to {DEFAULT_FIG}).")
        p.add_argument("--fig-dir", type=Path, default=None, help="Override the figure directory only.")
        p.add_argument("--toi-table", type=Path, default=PROJECT_ROOT / "data" / "catalogs" / "tess_toi.csv")
        p.set_defaults(func=func)
        return p

    p = add("stages", cmd_stages, "Stage sensitivity from the committed R1 CSV and failure log.")
    p.add_argument("--r1", type=Path, default=REGIME_CSVS["r1"])
    p.add_argument("--r1-failures", type=Path, default=TIER1_FAILURES)

    p = add("training", cmd_training, "Summaries of train_variants.py runs.")
    p.add_argument("--training-dir", type=Path, default=PROJECT_ROOT / "results" / "camera_ready" / "training")

    p = add("rescore", cmd_rescore, "Matched-threshold comparisons of rescored checkpoints.")
    p.add_argument("--rescore-dir", type=Path, default=PROJECT_ROOT / "results" / "camera_ready" / "rescore")
    p.add_argument("--ephemeris-tag", default="bls", help="Reads {regime}_{tag}.csv: bls or bls_full.")

    p = add("injection", cmd_injection, "Completeness and stage losses from injection_recovery.py.")
    p.add_argument("--injection-csv", type=Path,
                   default=PROJECT_ROOT / "results" / "camera_ready" / "injection" / "r3_injections.csv")

    p = add("probe", cmd_probe, "Invariance and depth-sweep summaries from depth_probe.py.")
    p.add_argument("--probe-dir", type=Path, default=PROJECT_ROOT / "results" / "camera_ready" / "depth_probe")

    p = add("negatives", cmd_negatives, "False-positive flag breakdown of Kepler negatives.")
    p.add_argument("--v2", type=Path, default=PROJECT_ROOT / "data" / "processed" / "kepler_trainset_v2.h5")
    p.add_argument("--koi-csv", type=Path, default=PROJECT_ROOT / "data" / "catalogs" / "kepler_koi_train.csv")

    args = parser.parse_args()
    if args.fig_dir is None:
        args.fig_dir = DEFAULT_FIG if args.out_dir is None else args.out_dir
    if args.out_dir is None:
        args.out_dir = DEFAULT_OUT
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.func(args)


if __name__ == "__main__":
    main()
