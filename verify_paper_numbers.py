"""
Regenerate and check every number that appears in the paper.
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

from stellaris.model import StellarisNetwork

# Set to True once results/ablation_results.json contains the ten-seed run.
ABLATION_FROM_JSON = True

# The ten-seed ablation summary, mean and sample s.d. (ddof=1) over seeds 0-9.
TEN_SEED = {
    "both":        {"pr": (0.9439, 0.0060), "roc": (0.9589, 0.0043), "r95": (0.674, 0.047)},
    "global-only": {"pr": (0.4334, 0.0264), "roc": (0.5515, 0.0461), "r95": (0.000, 0.000)},
    "local-only":  {"pr": (0.9434, 0.0032), "roc": (0.9581, 0.0018), "r95": (0.685, 0.026)},
}
N_SEEDS = 10

_passed = 0
_failed: list[str] = []

def check(label: str, got, want, tol: float = 0.0) -> None:
    """Compare a recomputed value against what the manuscript claims."""
    global _passed
    if isinstance(want, (int, float)) and isinstance(got, (int, float, np.floating)):
        ok = abs(float(got) - float(want)) <= tol
    else:
        ok = got == want
    if ok:
        _passed += 1
        print(f"  [PASS] {label:<50} {got}")
    else:
        _failed.append(label)
        print(f"  [FAIL] {label:<50} got {got}, paper says {want}")

def load_regime(path: str) -> pd.DataFrame:
    """Load a per-target result table and coerce the vetting flag to a real bool."""
    frame = pd.read_csv(path)
    # The CSV stores "True"/"False" as text; comparing strings to booleans silently
    # fails, so map explicitly rather than relying on truthiness.
    frame["vetted"] = (
        frame["vetting_passed"].astype(str).str.strip().str.lower()
        .map({"true": True, "false": False})
    )
    return frame

def main() -> None:
    r1 = load_regime("results/tier1_results.csv") # confirmed planets
    r2 = load_regime("results/tier3_results.csv") # unresolved TOIs
    r3 = load_regime("results/novel_results.csv") # faint, unflagged

    toi = pd.read_csv("data/catalogs/tess_toi.csv", comment="#")
    toi["TIC ID"] = toi["TIC ID"].astype(int)
    # CP = confirmed planet, KP = known planet. These are the ground truth for R1.
    confirmed = toi[toi["TFOPWG Disposition"].isin(["CP", "KP"])]

    # Method
    print("\n---- METHOD ----")
    model = StellarisNetwork()
    total = sum(p.numel() for p in model.parameters())
    head = sum(p.numel() for p in model.head.parameters())
    check("total parameters", total, 9_710_305)
    check("head parameters", head, 9_046_017)
    check("head as % of network", round(head / total * 100, 1), 93.2)
    check("global-only parameters",
          sum(p.numel() for p in StellarisNetwork(use_local=False).parameters()), 8_914_737)
    check("local-only parameters",
          sum(p.numel() for p in StellarisNetwork(use_global=False).parameters()), 1_321_905)

    ckpt = torch.load("checkpoints/stellaris_best.pt", weights_only=False, map_location="cpu")
    check("released checkpoint epoch", ckpt["epoch"], 33)
    check("released checkpoint val PR-AUC", round(float(ckpt["pr_auc"]), 4), 0.9463)

    # Table 1 released model
    print("\n--- TABLE 1: RELEASED CHECKPOINT ---")
    preds = np.load("results/test_predictions.npz")
    probs = preds["probs"]
    labels = preds["labels"].astype(int)
    check("test set size", len(labels), 1020)
    check("test positives", int(labels.sum()), 412)
    check("test negatives", int((1 - labels).sum()), 608)
    check("positive rate", round(float(labels.mean()), 3), 0.404)
    check("PR-AUC", round(average_precision_score(labels, probs), 3), 0.945)
    check("ROC-AUC", round(roc_auc_score(labels, probs), 3), 0.959)
    precision, recall, _ = precision_recall_curve(labels, probs)
    # Recall at 95% precision: the most restrictive operating point we quote.
    at_95 = precision >= 0.95
    check("recall at 95% precision", round(float(recall[at_95].max()), 3), 0.716)

    # Table 1 baseline
    print("\n=== TABLE 1: XGBOOST BASELINE ===")
    xgb = json.load(open("results/xgboost_baseline_metrics.json"))
    check("PR-AUC", round(xgb["pr_auc"], 3), 0.918)
    check("ROC-AUC", round(xgb["roc_auc"], 3), 0.934)
    xp, xr = np.array(xgb["precision"]), np.array(xgb["recall"])
    check("recall at 95% precision", round(float(xr[xp >= 0.95].max()), 3), 0.575)

    # Table 1 ablation
    print("\n---- TABLE 1: TEN-SEED ABLATION ----")
    if ABLATION_FROM_JSON:
        runs = pd.DataFrame(json.load(open("results/ablation_results.json")))
        assert runs.seed.nunique() == N_SEEDS, "ablation JSON is not the ten-seed run"
        stats_of = lambda v, k: (runs[runs.variant == v][k].mean(),
                                 runs[runs.variant == v][k].std(ddof=1))
        both_pr = stats_of("both", "test_pr_auc")
        local_pr = stats_of("local-only", "test_pr_auc")
    else:
        print("[WARN] reading the ten-seed summary from this file, not from")
        print("results/ablation_results.json, which still holds the 3-seed run.")
        both_pr = TEN_SEED["both"]["pr"]
        local_pr = TEN_SEED["local-only"]["pr"]

    # Welch's t-test: unequal variances, which is the right default here because the
    # full model's seed spread is visibly larger than the local-only model's.
    mean_b, sd_b = both_pr
    mean_l, sd_l = local_pr
    difference = mean_b - mean_l
    std_err = np.sqrt(sd_b**2 / N_SEEDS + sd_l**2 / N_SEEDS)
    # Welch-Satterthwaite degrees of freedom.
    dof = (sd_b**2 / N_SEEDS + sd_l**2 / N_SEEDS) ** 2 / (
        (sd_b**2 / N_SEEDS) ** 2 / (N_SEEDS - 1)
        + (sd_l**2 / N_SEEDS) ** 2 / (N_SEEDS - 1)
    )
    t_stat = difference / std_err
    p_value = 2 * stats.t.sf(abs(t_stat), dof)
    half_width = stats.t.ppf(0.975, dof) * std_err

    check("both minus local-only (PR-AUC)", round(difference, 4), 0.0005)
    check("Welch p-value", round(p_value, 2), 0.83)
    check("95% CI lower bound", round(difference - half_width, 3), -0.004)
    check("95% CI upper bound", round(difference + half_width, 3), 0.005)
    check("parameters shed by dropping global", total - 1_321_905, 8_388_400)
    check("shed as % of network", round((total - 1_321_905) / total * 100), 86)
    # The deployed model should be a typical draw, not a lucky one.
    check("released checkpoint, s.d. from ten-seed mean",
          round(abs(0.9451 - mean_b) / sd_b, 1), 0.2, tol=0.05)

    # R1
    print("\n---- R1: CONFIRMED PLANETS ----")
    check("processed", len(r1), 1102)
    endorsed = (r1.cnn_probability >= 0.5) & r1.vetted
    check("recovered at 0.5 with both rules", int(endorsed.sum()), 685)
    check("recovery rate %", round(float(endorsed.mean()) * 100, 1), 62.2)
    check("missed", int((~endorsed).sum()), 417)

    # Attribute each miss to the stage that lost it. A target can fail both.
    cnn_failed = r1.cnn_probability < 0.5
    vet_failed = ~r1.vetted
    check("lost to vetting alone", int((vet_failed & ~cnn_failed).sum()), 131)
    check("lost to the classifier alone", int((cnn_failed & ~vet_failed).sum()), 179)
    check("lost to both", int((cnn_failed & vet_failed).sum()), 107)
    check("classifier false-negative rate %", round(float(cnn_failed.mean()) * 100, 1), 26.0)
    check("vetting false-rejection rate %", round(float(vet_failed.mean()) * 100, 1), 21.6)

    reasons = r1.vetting_reasons.fillna("")
    # The odd-even rejection string contains the word "depth", so match on the rule
    # name rather than grepping for "depth", which would double-count.
    odd_even = reasons.str.contains("odd/even")
    depth_cap = reasons.str.contains("cap")
    check("odd-even rejections", int(odd_even.sum()), 215)
    check("depth-cap rejections", int(depth_cap.sum()), 28)
    check("rejected by both rules", int((odd_even & depth_cap).sum()), 5)
    # With odd-even switched off, a target passes vetting unless the depth cap kills it.
    without_oe = (r1.cnn_probability >= 0.5) & ~depth_cap
    check("recovery with odd-even off %", round(float(without_oe.mean()) * 100, 1), 72.5)

    # Did BLS even find the right period? Match against every catalogued planet on
    # the target, and accept if any one of them lands within 2%.
    found = {}
    for _, row in r1.iterrows():
        planets = confirmed[confirmed["TIC ID"] == row.tic_id]
        found[row.tic_id] = (not planets.empty) and bool(
            ((row.period_days / planets["Period (days)"] - 1).abs() <= 0.02).any()
        )
    r1["bls_correct"] = r1.tic_id.map(found)
    check("BLS within 2% of catalogue", int(r1.bls_correct.sum()), 936)
    check("BLS recovery rate %", round(float(r1.bls_correct.mean()) * 100, 1), 84.9)
    check("BLS misses", int((~r1.bls_correct).sum()), 166)

    # A target is only "unrecoverable by construction" if EVERY catalogued planet on
    # it sits outside the search box. Multi-planet systems make this non-trivial.
    period_only = duration_only = both_out = 0
    for tic in r1[~r1.bls_correct].tic_id:
        planets = confirmed[confirmed["TIC ID"] == tic]
        all_long_period = bool((planets["Period (days)"] > 20).all())
        all_long_duration = bool((planets["Duration (hours)"] > 8).all())
        if all_long_period and all_long_duration:
            both_out += 1
        elif all_long_period:
            period_only += 1
        elif all_long_duration:
            duration_only += 1
    check("beyond the 20-day ceiling only", period_only, 53)
    check("beyond the 8-hour grid only", duration_only, 3)
    check("beyond both", both_out, 10)
    check("unrecoverable by construction", period_only + duration_only + both_out, 66)

    # R2
    print("\n---- R2: UNRESOLVED TOIs ----")
    check("processed", len(r2), 4002)
    check("endorsed at 0.9", int(((r2.cnn_probability >= 0.9) & r2.vetted).sum()), 1564)
    # These are the signals the network likes and the rules then kill. They are the
    # clearest evidence that the network cannot see depth.
    check("score >= 0.9 then rejected", int(((r2.cnn_probability >= 0.9) & ~r2.vetted).sum()), 361)

    screened = pd.read_csv("results/validation/validation_summary.csv")
    check("screened", len(screened), 30)
    check("survivors", int(screened.overall_pass.sum()), 10)
    check("flagged instrumental", int(screened.flags_systematic.sum()), 12)
    no_reproduce = ~screened.transit_reproduces.astype(bool)
    check("failed sector reproduction", int(no_reproduce.sum()), 11)
    # Single-sector targets cannot fail this test; the test simply cannot run.
    check("  of those, single-sector", int((no_reproduce & (screened.n_sectors_observed == 1)).sum()), 10)

    merged = (screened.merge(toi[["TIC ID", "TFOPWG Disposition"]],
                             left_on="tic_id", right_on="TIC ID")
              .drop_duplicates("tic_id"))
    dispositions = collections.Counter(merged["TFOPWG Disposition"])
    check("all 30 screened: planet candidates", dispositions["PC"], 28)
    check("all 30 screened: ambiguous", dispositions["APC"], 2)
    survivors = collections.Counter(merged[merged.overall_pass]["TFOPWG Disposition"])
    check("survivors: planet candidates", survivors["PC"], 9)
    check("survivors: ambiguous", survivors["APC"], 1)

    # Table 2
    print("\n---- TABLE 2 ----")
    for name, frame, vetted_pct, endorsed_pct in [
        ("R1", r1, 78.4, 44.9), ("R2", r2, 70.3, 39.1), ("R3", r3, 29.3, 0.2),
    ]:
        check(f"{name} vetting pass rate %", round(float(frame.vetted.mean()) * 100, 1), vetted_pct)
        check(f"{name} endorsed at 0.9 %",
              round(float(((frame.cnn_probability >= 0.9) & frame.vetted).mean()) * 100, 1),
              endorsed_pct)

    # R3
    print("\n---- R3: FAINT, UNFLAGGED ----")
    check("usable views", len(r3), 474)
    failures = Path("results/novel_failures.log").read_text().splitlines()
    # The local view spans +/-4 durations. If 4*duration/period >= 0.5 the window
    # wraps past half a phase and views.py refuses to build it. That is the failure.
    view_losses = sum(1 for line in failures
                      if "exceeds phase range" in line or "Local window" in line)
    check("lost at view construction", view_losses, 1524)
    check("total losses", 2000 - len(r3), 1526)

    check("median depth ppm", int(round(r3.depth_ppm.median())), 10525)
    check("R1 median depth ppm", int(round(r1.depth_ppm.median())), 5319)
    check("median BLS S/N", round(float(r3.bls_snr.median()), 1), 5.8)
    check("R1 median BLS S/N", round(float(r1.bls_snr.median()), 1), 49.8)
    check("median classifier score", round(float(r3.cnn_probability.median()), 3), 0.031)

    # TESS orbits Earth every 13.7 days. Momentum dumps and perigee passages recur
    # on that period, so BLS finds it on anything noise-dominated.
    tess_orbit = 13.7
    near_orbit = (r3.period_days - tess_orbit).abs() / tess_orbit <= 0.10
    check("within 10% of 13.7 d, %", round(float(near_orbit.mean()) * 100, 1), 41.6)
    high = r3[r3.cnn_probability >= 0.9]
    check("scoring above 0.9", len(high), 4)
    check("  of those, near 13.7 d",
          int((((high.period_days - tess_orbit).abs() / tess_orbit) <= 0.10).sum()), 4)
    top8 = r3.nlargest(8, "cnn_probability")
    check("top 8 by score, near 13.7 d",
          int((((top8.period_days - tess_orbit).abs() / tess_orbit) <= 0.10).sum()), 6)
    # The network scores the systematic no lower than anything else: it is indifferent
    # to it, not discriminating against it. This is the paper's central mechanism.
    check("median score, near 13.7 d",
          round(float(r3.loc[near_orbit, "cnn_probability"].median()), 3), 0.030)
    check("median score, everything else",
          round(float(r3.loc[~near_orbit, "cnn_probability"].median()), 3), 0.032)

    candidate = r3[r3.tic_id == 155810890].iloc[0]
    check("TIC 155810890 period", round(float(candidate.period_days), 3), 14.390)
    check("TIC 155810890 % from 13.7 d",
          round(abs(candidate.period_days - tess_orbit) / tess_orbit * 100, 2), 5.04)
    check("TIC 155810890 BLS S/N", round(float(candidate.bls_snr), 1), 9.3)
    novel = pd.read_csv("results/validation_novel/validation_summary.csv").iloc[0]
    check("TIC 155810890 sectors observed", int(novel.n_sectors_observed), 3)
    check("TIC 155810890 sectors with transit", int(novel.n_sectors_with_transit), 1)
    check("TIC 155810890 survives screening", bool(novel.overall_pass), False)

    # Discussion
    print("\n---- DISCUSSION ----")
    # Conditioning on a correct ephemeris: does the network still lose more planets
    # than the rules do? If the answer flipped, the paper's headline would collapse.
    clean = r1[r1.bls_correct]
    check("classifier FN rate | BLS correct %",
          round(float((clean.cnn_probability < 0.5).mean()) * 100, 1), 20.3)
    check("vetting FR rate | BLS correct %",
          round(float((~clean.vetted).mean()) * 100, 1), 13.0)
    check("R1 targets below S/N 6", int((r1.bls_snr < 6).sum()), 21)
    check("R3 targets below S/N 6", int((r3.bls_snr < 6).sum()), 244)

    # Reweight R1 to R3's signal-to-noise profile: how much of the vetting gap is
    # composition rather than population? The answer moves with the binning, which
    # is exactly why the paper quotes a range and calls the estimate thin.
    print("\n  Reweighting R1 to R3's S/N distribution (paper quotes 54-66%):")
    for width in (1, 2, 3):
        edges = np.arange(0, r1.bls_snr.max() + width, width)
        r1_bin = pd.cut(r1.bls_snr, edges)
        r3_bin = pd.cut(r3.bls_snr, edges)
        weights = r3_bin.value_counts(normalize=True)
        pass_rate = r1.groupby(r1_bin, observed=True).vetted.mean()
        table = pd.DataFrame({"rate": pass_rate, "weight": weights}).dropna()
        reweighted = (table.rate * table.weight).sum() / table.weight.sum()
        print(f"    width-{width} bins: {reweighted * 100:.0f}%")

    print("\n" + "-" * 60)
    print(f"{_passed} passed, {len(_failed)} failed")
    if _failed:
        print("\n  FAILURES:")
        for name in _failed:
            print(f"- {name}")
        sys.exit(1)
    print("  Every number in the paper reproduces from committed artefacts.")
    print("-" * 60)


if __name__ == "__main__":
    main()
