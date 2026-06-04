"""Analyse hunt results CSV: summary stats, top candidates, failure breakdown, plots."""
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def load(results_path: Path) -> pd.DataFrame:
    df = pd.read_csv(results_path)
    df["vetting_passed"] = df["vetting_passed"].astype(str).str.strip().str.lower() == "true"
    return df


def print_summary(df: pd.DataFrame):
    total = len(df)
    passed = df["vetting_passed"].sum()
    failed = total - passed

    print("-" * 60)
    print("HUNT SUMMARY")
    print("-" * 60)
    print(f"Total candidates processed : {total}")
    print(f"Passed vetting             : {passed}  ({100*passed/total:.1f}%)")
    print(f"Failed vetting             : {failed}  ({100*failed/total:.1f}%)")

    p = df[df["vetting_passed"]]
    print(f"\n    Passed-vetting stats    ")
    print(f"CNN probability  : mean={p['cnn_probability'].mean():.3f}  "
          f"median={p['cnn_probability'].median():.3f}")
    print(f"BLS SNR          : mean={p['bls_snr'].mean():.1f}  "
          f"median={p['bls_snr'].median():.1f}")
    print(f"Period [days]    : mean={p['period_days'].mean():.2f}  "
          f"median={p['period_days'].median():.2f}")
    print(f"Depth [ppm]      : mean={p['depth_ppm'].mean():.0f}  "
          f"median={p['depth_ppm'].median():.0f}")
    print(f"Duration [hours] : mean={p['duration_hours'].mean():.2f}  "
          f"median={p['duration_hours'].median():.2f}")


def print_top_candidates(df: pd.DataFrame, n: int = 20):
    top = (
        df[df["vetting_passed"]]
        .sort_values("rank_score", ascending=False)
        .head(n)
    )
    print(f"\n{'-' * 60}")
    print(f"TOP {n} CANDIDATES (by rank score)")
    print("-" * 60)
    cols = ["tic_id", "period_days", "duration_hours", "depth_ppm", "bls_snr", "cnn_probability", "rank_score"]
    print(top[cols].to_string(index=False))


def print_failure_breakdown(df: pd.DataFrame):
    failed = df[~df["vetting_passed"]]
    if failed.empty:
        print("\nNo vetting failures.")
        return

    all_reasons = []
    for reasons_str in failed["vetting_reasons"].dropna():
        for r in str(reasons_str).split(";"):
            r = r.strip()
            if r:
                # Normalise numeric values so similar reasons group together
                import re
                r_key = re.sub(r"\d+\.?\d*%?", "N", r)
                r_key = re.sub(r"\(.*?\)", "", r_key).strip()
                all_reasons.append(r_key)

    counts = Counter(all_reasons)
    print(f"\n{'-' * 60}")
    print("FAILURE REASON BREAKDOWN")
    print("-" * 60)
    for reason, count in counts.most_common(15):
        print(f"{count:4d}  {reason}")


def make_plots(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    passed = df[df["vetting_passed"]]
    failed = df[~df["vetting_passed"]]

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("Hunt Results Analysis", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # CNN probability histogram
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(passed["cnn_probability"], bins=30, color="steelblue", alpha=0.8, label="passed")
    ax.hist(failed["cnn_probability"], bins=30, color="tomato", alpha=0.6, label="failed")
    ax.set_xlabel("CNN Probability")
    ax.set_ylabel("Count")
    ax.set_title("CNN Score Distribution")
    ax.legend(fontsize=8)

    # BLS SNR histogram 
    ax = fig.add_subplot(gs[0, 1])
    snr_max = df["bls_snr"].quantile(0.98)
    ax.hist(passed["bls_snr"].clip(upper=snr_max), bins=30, color="steelblue", alpha=0.8, label="passed")
    ax.hist(failed["bls_snr"].clip(upper=snr_max), bins=30, color="tomato", alpha=0.6, label="failed")
    ax.set_xlabel("BLS SNR")
    ax.set_ylabel("Count")
    ax.set_title("BLS SNR Distribution")
    ax.legend(fontsize=8)

    # Period histogram
    ax = fig.add_subplot(gs[0, 2])
    ax.hist(passed["period_days"], bins=40, color="steelblue", alpha=0.8, label="passed")
    ax.hist(failed["period_days"], bins=40, color="tomato", alpha=0.6, label="failed")
    ax.set_xlabel("Period (days)")
    ax.set_ylabel("Count")
    ax.set_title("Period Distribution")
    ax.legend(fontsize=8)

    # Depth vs Period scatter
    depth_max = df["depth_ppm"].quantile(0.99)
    ax = fig.add_subplot(gs[1, 0])
    ax.scatter(failed["period_days"], failed["depth_ppm"].clip(upper=depth_max),
               s=10, color="tomato", alpha=0.4, label="failed")
    ax.scatter(passed["period_days"], passed["depth_ppm"].clip(upper=depth_max),
               s=10, color="steelblue", alpha=0.7, label="passed")
    ax.set_xlabel("Period (days)")
    ax.set_ylabel("Depth (ppm)")
    ax.set_title("Depth vs Period")
    ax.legend(fontsize=8)

    # CNN prob vs BLS SNR scatter
    ax = fig.add_subplot(gs[1, 1])
    ax.scatter(failed["bls_snr"].clip(upper=snr_max), failed["cnn_probability"],
               s=10, color="tomato", alpha=0.4, label="failed")
    ax.scatter(passed["bls_snr"].clip(upper=snr_max), passed["cnn_probability"],
               s=10, color="steelblue", alpha=0.7, label="passed")
    ax.set_xlabel("BLS SNR")
    ax.set_ylabel("CNN Probability")
    ax.set_title("CNN vs BLS SNR")
    ax.legend(fontsize=8)

    # Rank score distribution (top candidates)
    ax = fig.add_subplot(gs[1, 2])
    top_scores = passed["rank_score"].sort_values(ascending=False).head(50)
    ax.barh(range(len(top_scores)), top_scores.values, color="steelblue")
    ax.set_xlabel("Rank Score")
    ax.set_ylabel("Candidate rank")
    ax.set_title("Top 50 Rank Scores")
    ax.invert_yaxis()

    plot_path = out_dir / "hunt_analysis.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved to {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyse hunt results CSV.")
    parser.add_argument("--results", type=Path, required=True, help="Path to results CSV.")
    parser.add_argument("--top", type=int, default=20, help="Number of top candidates to show.")
    parser.add_argument("--plots", type=Path, default=Path("results"), help="Directory for output plots.")
    parser.add_argument("--no-plots", action="store_true", help="Skip generating plots.")
    args = parser.parse_args()

    df = load(args.results)

    print_summary(df)
    print_top_candidates(df, n=args.top)
    print_failure_breakdown(df)

    if not args.no_plots:
        make_plots(df, args.plots)


if __name__ == "__main__":
    main()
