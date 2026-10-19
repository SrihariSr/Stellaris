"""
Regenerate and check every number that appears in the paper.
"""
from __future__ import annotations

import ast
import collections
import json
import re
import sys
import types
import warnings
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from unittest import mock

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

TESS_ORBIT_DAYS = 13.7
N_NOVEL_TARGETS = 2000
# A log-uniform period prior over the 0.5-20 d search box puts this much mass within 10% of 13.7 d.
BAND_SHARE = np.log(1.1 / 0.9) / np.log(20 / 0.5)


def _p_value(k: int, n: int) -> float:
    return float(stats.binomtest(k, n, 0.5).pvalue) if n > 0 else float("nan")


def tess_numbers(r1: pd.DataFrame, r2: pd.DataFrame, r3: pd.DataFrame,
                 confirmed: pd.DataFrame | None = None) -> dict:
    """
    Every number the paper derives from the three hunt tables, unrounded and keyed by name.

    The frames need the hunt CSV columns plus "vetted" (see load_regime). confirmed is
    the CP/KP subset of the TOI table, read from disk when omitted.
    """
    if confirmed is None:
        toi = pd.read_csv("data/catalogs/tess_toi.csv", comment="#")
        toi["TIC ID"] = toi["TIC ID"].astype(int)
        confirmed = toi[toi["TFOPWG Disposition"].isin(["CP", "KP"])]
    n: dict = {}

    n["r1_processed"] = len(r1)
    endorsed = (r1.cnn_probability >= 0.5) & r1.vetted
    n["r1_recovered"] = int(endorsed.sum())
    n["r1_recovery_rate"] = float(endorsed.mean())
    n["r1_missed"] = int((~endorsed).sum())

    # Attribute each miss to the stage that lost it. A target can fail both.
    cnn_failed = r1.cnn_probability < 0.5
    vet_failed = ~r1.vetted
    n["r1_lost_vetting_only"] = int((vet_failed & ~cnn_failed).sum())
    n["r1_lost_classifier_only"] = int((cnn_failed & ~vet_failed).sum())
    n["r1_lost_both"] = int((cnn_failed & vet_failed).sum())
    n["r1_classifier_fn_rate"] = float(cnn_failed.mean())
    n["r1_vetting_fr_rate"] = float(vet_failed.mean())

    reasons = r1.vetting_reasons.fillna("")
    # The odd-even rejection string contains the word "depth", so match on the rule
    # name rather than grepping for "depth", which would double-count.
    odd_even = reasons.str.contains("odd/even")
    depth_cap = reasons.str.contains("cap")
    n["r1_odd_even_rejections"] = int(odd_even.sum())
    n["r1_depth_cap_rejections"] = int(depth_cap.sum())
    n["r1_rejected_by_both_rules"] = int((odd_even & depth_cap).sum())
    # With odd-even switched off, a target passes vetting unless the depth cap kills it.
    n["r1_recovery_rate_odd_even_off"] = float(((r1.cnn_probability >= 0.5) & ~depth_cap).mean())

    # Did BLS even find the right period? Match against every catalogued planet on
    # the target, and accept if any one of them lands within 2%.
    found = {}
    for _, row in r1.iterrows():
        planets = confirmed[confirmed["TIC ID"] == row.tic_id]
        found[row.tic_id] = (not planets.empty) and bool(
            ((row.period_days / planets["Period (days)"] - 1).abs() <= 0.02).any()
        )
    bls_correct = r1.tic_id.map(found).astype(bool)
    n["r1_bls_matches"] = int(bls_correct.sum())
    n["r1_bls_match_rate"] = float(bls_correct.mean())
    n["r1_bls_misses"] = int((~bls_correct).sum())

    # A target is only "unrecoverable by construction" if EVERY catalogued planet on
    # it sits outside the search box. Multi-planet systems make this non-trivial.
    # A catalogue period of 0 marks a single transit, which no periodic search can recover.
    single_only = period_out = both_out = duration_out = unrecoverable = 0
    for tic in r1[~bls_correct].tic_id:
        planets = confirmed[confirmed["TIC ID"] == tic]
        single = planets["Period (days)"] <= 0
        # A single-transit companion counts as beyond the 20-day ceiling.
        beyond_period = single | (planets["Period (days)"] > 20)
        beyond_duration = planets["Duration (hours)"] > 8
        if planets.empty or not (beyond_period | beyond_duration).all():
            continue
        unrecoverable += 1
        if single.all():
            single_only += 1
        elif beyond_period.all() and beyond_duration.all():
            both_out += 1
        elif beyond_period.all():
            period_out += 1
        elif beyond_duration.all():
            duration_out += 1
    n["r1_single_transit_only"] = single_only
    n["r1_beyond_20d"] = period_out
    n["r1_beyond_20d_and_8h"] = both_out
    n["r1_beyond_8h_only"] = duration_out
    n["r1_unrecoverable"] = unrecoverable

    n["r2_processed"] = len(r2)
    n["r2_endorsed_09"] = int(((r2.cnn_probability >= 0.9) & r2.vetted).sum())
    # These are the signals the network likes and the rules then kill. They are the
    # clearest evidence that the network cannot see depth.
    n["r2_high_then_rejected"] = int(((r2.cnn_probability >= 0.9) & ~r2.vetted).sum())

    for name, frame in (("r1", r1), ("r2", r2), ("r3", r3)):
        n[f"{name}_vetting_pass_rate"] = float(frame.vetted.mean())
        n[f"{name}_endorsed_09_rate"] = float(((frame.cnn_probability >= 0.9) & frame.vetted).mean())

    n["r3_usable_views"] = len(r3)
    n["r3_total_losses"] = N_NOVEL_TARGETS - len(r3)
    n["r3_median_depth_ppm"] = float(r3.depth_ppm.median())
    n["r1_median_depth_ppm"] = float(r1.depth_ppm.median())
    n["r3_median_bls_snr"] = float(r3.bls_snr.median())
    n["r1_median_bls_snr"] = float(r1.bls_snr.median())
    n["r3_median_score"] = float(r3.cnn_probability.median())

    # TESS orbits Earth every 13.7 days. Momentum dumps and perigee passages recur
    # on that period, so BLS finds it on anything noise-dominated.
    near_orbit = (r3.period_days - TESS_ORBIT_DAYS).abs() / TESS_ORBIT_DAYS <= 0.10
    n["r3_near_orbit_fraction"] = float(near_orbit.mean())
    high = r3[r3.cnn_probability >= 0.9]
    n["r3_scoring_above_09"] = len(high)
    n["r3_above_09_near_orbit"] = int((((high.period_days - TESS_ORBIT_DAYS).abs() / TESS_ORBIT_DAYS) <= 0.10).sum())
    top8 = r3.nlargest(8, "cnn_probability")
    n["r3_top8_near_orbit"] = int((((top8.period_days - TESS_ORBIT_DAYS).abs() / TESS_ORBIT_DAYS) <= 0.10).sum())
    # The network scores the systematic no lower than anything else: it is indifferent
    # to it, not discriminating against it. This is the paper's central mechanism.
    n["r3_median_score_near_orbit"] = float(r3.loc[near_orbit, "cnn_probability"].median())
    n["r3_median_score_elsewhere"] = float(r3.loc[~near_orbit, "cnn_probability"].median())

    candidate = r3[r3.tic_id == 155810890]
    period = float(candidate.period_days.iloc[0]) if len(candidate) else float("nan")
    n["tic155810890_period_days"] = period
    n["tic155810890_pct_from_orbit"] = abs(period - TESS_ORBIT_DAYS) / TESS_ORBIT_DAYS * 100
    n["tic155810890_bls_snr"] = float(candidate.bls_snr.iloc[0]) if len(candidate) else float("nan")

    # Conditioning on a correct ephemeris: does the network still lose more planets
    # than the rules do? If the answer flipped, the paper's headline would collapse.
    clean = r1[bls_correct]
    n["r1_classifier_fn_rate_given_match"] = float((clean.cnn_probability < 0.5).mean())
    n["r1_vetting_fr_rate_given_match"] = float((~clean.vetted).mean())
    n["r1_below_snr_6"] = int((r1.bls_snr < 6).sum())
    n["r3_below_snr_6"] = int((r3.bls_snr < 6).sum())

    # Reweight R1 to R3's signal-to-noise profile: how much of the vetting gap is
    # composition rather than population? The answer moves with the binning, which
    # is exactly why the paper quotes a range and calls the estimate thin.
    for width in (1, 2, 3):
        edges = np.arange(0, r1.bls_snr.max() + width, width)
        r1_bin = pd.cut(r1.bls_snr, edges)
        r3_bin = pd.cut(r3.bls_snr, edges)
        weights = r3_bin.value_counts(normalize=True)
        pass_rate = r1.groupby(r1_bin, observed=True).vetted.mean()
        table = pd.DataFrame({"rate": pass_rate, "weight": weights}).dropna()
        n[f"r1_reweighted_pass_rate_w{width}"] = float((table.rate * table.weight).sum() / table.weight.sum())

    # Exact McNemar test on the discordant pairs: lost only to the classifier vs only to vetting.
    clf_only, vet_only = n["r1_lost_classifier_only"], n["r1_lost_vetting_only"]
    n["r1_mcnemar_p"] = _p_value(min(clf_only, vet_only), clf_only + vet_only)
    n["r1_match_classifier_only"] = int(((clean.cnn_probability < 0.5) & clean.vetted).sum())
    n["r1_match_vetting_only"] = int(((clean.cnn_probability >= 0.5) & ~clean.vetted).sum())
    a, b = n["r1_match_classifier_only"], n["r1_match_vetting_only"]
    n["r1_match_mcnemar_p"] = _p_value(min(a, b), a + b)

    in_band = r3[near_orbit]
    n["r3_band_excess"] = n["r3_near_orbit_fraction"] / BAND_SHARE
    n["r3_band_median_depth_ppm"] = float(in_band.depth_ppm.median())
    n["r3_band_below_cap"] = int((in_band.depth_ppm < 30_000).sum())
    n["r3_band_count"] = len(in_band)
    n["r3_band_below_cap_fraction"] = n["r3_band_below_cap"] / len(in_band) if len(in_band) else float("nan")
    n["r3_above_09_depth_cap_failures"] = int(high.vetting_reasons.fillna("").str.contains("cap").sum())

    rejected_high = r2[(r2.cnn_probability >= 0.9) & ~r2.vetted].vetting_reasons.fillna("")
    oe_high, cap_high = rejected_high.str.contains("odd/even"), rejected_high.str.contains("cap")
    n["r2_high_rejected_odd_even_only"] = int((oe_high & ~cap_high).sum())
    n["r2_high_rejected_depth_cap_only"] = int((cap_high & ~oe_high).sum())
    n["r2_high_rejected_both"] = int((oe_high & cap_high).sum())

    n["r1_classifier_fn_rate_03"] = float((r1.cnn_probability < 0.3).mean())
    n["r1_crossover_threshold"] = next((float(t) for t in np.round(np.linspace(0, 1, 1001), 3)
                                        if float((r1.cnn_probability < t).mean()) >= n["r1_vetting_fr_rate"]),
                                       float("nan"))
    return n


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

    n = tess_numbers(r1, r2, r3, confirmed)

    # R1
    print("\n---- R1: CONFIRMED PLANETS ----")
    check("processed", n["r1_processed"], 1102)
    check("recovered at 0.5 with both rules", n["r1_recovered"], 685)
    check("recovery rate %", round(n["r1_recovery_rate"] * 100, 1), 62.2)
    check("missed", n["r1_missed"], 417)
    check("lost to vetting alone", n["r1_lost_vetting_only"], 131)
    check("lost to the classifier alone", n["r1_lost_classifier_only"], 179)
    check("lost to both", n["r1_lost_both"], 107)
    check("classifier false-negative rate %", round(n["r1_classifier_fn_rate"] * 100, 1), 26.0)
    check("vetting false-rejection rate %", round(n["r1_vetting_fr_rate"] * 100, 1), 21.6)
    check("odd-even rejections", n["r1_odd_even_rejections"], 215)
    check("depth-cap rejections", n["r1_depth_cap_rejections"], 28)
    check("rejected by both rules", n["r1_rejected_by_both_rules"], 5)
    check("recovery with odd-even off %", round(n["r1_recovery_rate_odd_even_off"] * 100, 1), 72.5)
    check("BLS within 2% of catalogue", n["r1_bls_matches"], 936)
    check("BLS recovery rate %", round(n["r1_bls_match_rate"] * 100, 1), 84.9)
    check("BLS misses", n["r1_bls_misses"], 166)
    check("single-transit planets only", n["r1_single_transit_only"], 6)
    check("beyond the 20-day ceiling", n["r1_beyond_20d"], 54)
    check("beyond 20 d and the 8-hour grid", n["r1_beyond_20d_and_8h"], 10)
    check("beyond the 8-hour grid only", n["r1_beyond_8h_only"], 0)
    check("unrecoverable by construction", n["r1_unrecoverable"], 70)

    # R2
    print("\n---- R2: UNRESOLVED TOIs ----")
    check("processed", n["r2_processed"], 4002)
    check("endorsed at 0.9", n["r2_endorsed_09"], 1564)
    check("score >= 0.9 then rejected", n["r2_high_then_rejected"], 361)

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
    for name, vetted_pct, endorsed_pct in [("R1", 78.4, 44.9), ("R2", 70.3, 39.1), ("R3", 29.3, 0.2)]:
        check(f"{name} vetting pass rate %", round(n[f"{name.lower()}_vetting_pass_rate"] * 100, 1), vetted_pct)
        check(f"{name} endorsed at 0.9 %", round(n[f"{name.lower()}_endorsed_09_rate"] * 100, 1), endorsed_pct)

    # R3
    print("\n---- R3: FAINT, UNFLAGGED ----")
    check("usable views", n["r3_usable_views"], 474)
    failures = Path("results/novel_failures.log").read_text().splitlines()
    # The local view spans +/-4 durations. If 4*duration/period >= 0.5 the window
    # wraps past half a phase and views.py refuses to build it. That is the failure.
    view_losses = sum(1 for line in failures
                      if "exceeds phase range" in line or "Local window" in line)
    check("lost at view construction", view_losses, 1524)
    check("total losses", n["r3_total_losses"], 1526)

    check("median depth ppm", int(round(n["r3_median_depth_ppm"])), 10525)
    check("R1 median depth ppm", int(round(n["r1_median_depth_ppm"])), 5319)
    check("median BLS S/N", round(n["r3_median_bls_snr"], 1), 5.8)
    check("R1 median BLS S/N", round(n["r1_median_bls_snr"], 1), 49.8)
    check("median classifier score", round(n["r3_median_score"], 3), 0.031)

    check("within 10% of 13.7 d, %", round(n["r3_near_orbit_fraction"] * 100, 1), 41.6)
    check("scoring above 0.9", n["r3_scoring_above_09"], 4)
    check("  of those, near 13.7 d", n["r3_above_09_near_orbit"], 4)
    check("top 8 by score, near 13.7 d", n["r3_top8_near_orbit"], 6)
    check("median score, near 13.7 d", round(n["r3_median_score_near_orbit"], 3), 0.030)
    check("median score, everything else", round(n["r3_median_score_elsewhere"], 3), 0.032)

    check("TIC 155810890 period", round(n["tic155810890_period_days"], 3), 14.390)
    check("TIC 155810890 % from 13.7 d", round(n["tic155810890_pct_from_orbit"], 2), 5.04)
    check("TIC 155810890 BLS S/N", round(n["tic155810890_bls_snr"], 1), 9.3)
    novel = pd.read_csv("results/validation_novel/validation_summary.csv").iloc[0]
    check("TIC 155810890 sectors observed", int(novel.n_sectors_observed), 3)
    check("TIC 155810890 sectors with transit", int(novel.n_sectors_with_transit), 1)
    check("TIC 155810890 survives screening", bool(novel.overall_pass), False)

    # Discussion
    print("\n---- DISCUSSION ----")
    check("classifier FN rate | BLS correct %", round(n["r1_classifier_fn_rate_given_match"] * 100, 1), 20.3)
    check("vetting FR rate | BLS correct %", round(n["r1_vetting_fr_rate_given_match"] * 100, 1), 13.0)
    check("R1 targets below S/N 6", n["r1_below_snr_6"], 21)
    check("R3 targets below S/N 6", n["r3_below_snr_6"], 244)

    print("\n  Reweighting R1 to R3's S/N distribution (bin widths 1, 2, 3 give 60%, 54%, 53%):")
    for width in (1, 2, 3):
        print(f"    width-{width} bins: {n[f'r1_reweighted_pass_rate_w{width}'] * 100:.0f}%")

    # Camera-ready additions
    print("\n---- CAMERA-READY ----")
    ablation = pd.DataFrame(json.load(open("results/ablation_results.json")))
    for variant, want in (("global-only", 0.551), ("both", 0.959), ("local-only", 0.958)):
        check(f"{variant} mean test ROC-AUC",
              round(float(ablation[ablation.variant == variant].test_roc_auc.mean()), 3), want)
    for variant, (want_mean, want_sd) in (("both", (0.674, 0.047)), ("local-only", (0.685, 0.026))):
        r95 = ablation[ablation.variant == variant].test_recall_at_95
        check(f"{variant} R@95%P mean", round(float(r95.mean()), 3), want_mean)
        check(f"{variant} R@95%P s.d.", round(float(r95.std(ddof=1)), 3), want_sd)
    global_only = ablation[ablation.variant == "global-only"].sort_values("seed")
    check("global-only best epochs, seeds 0-9", [int(e) for e in global_only.best_epoch], [2, 1, 2, 8, 4, 3, 4, 5, 3, 1])
    # A collapsed seed emits one constant score, which puts its test ROC-AUC at exactly 0.5.
    check("global-only seeds with test ROC-AUC exactly 0.5",
          [int(s) for s in global_only[global_only.test_roc_auc == 0.5].seed], [3, 4])

    check("McNemar exact p, classifier-only vs vetting-only (179 vs 131)", round(n["r1_mcnemar_p"], 3), 0.007)
    check("  given a BLS period match: classifier-only losses", n["r1_match_classifier_only"], 143)
    check("  given a BLS period match: vetting-only losses", n["r1_match_vetting_only"], 75)
    check("  given a BLS period match: McNemar exact p < 1e-4", bool(n["r1_match_mcnemar_p"] < 1e-4), True)

    check("log-uniform share of 0.5-20 d within 10% of 13.7 d, %", round(BAND_SHARE * 100, 1), 5.4)
    check("excess of R3 periods in that band", round(n["r3_band_excess"], 1), 7.6)
    check("median depth of near-13.7 d R3 detections, ppm", int(round(n["r3_band_median_depth_ppm"])), 13791)
    check("  of them below the 3% depth cap", n["r3_band_below_cap"], 133)
    check("  near-13.7 d R3 detections", n["r3_band_count"], 197)
    check("  fraction below the 3% depth cap", round(n["r3_band_below_cap_fraction"], 3), 0.675)
    check("R3 scoring >= 0.9 that fail the depth cap", n["r3_above_09_depth_cap_failures"], 3)

    # tier1_failures.log records span several lines: a record starts with "<tic>\t".
    records: list[str] = []
    for line in Path("results/tier1_failures.log").read_text().splitlines():
        if re.match(r"^\d+\t", line):
            records.append(line)
        elif records:
            records[-1] += "\n" + line
    r1_view_losses = sum(1 for r in records if "exceeds phase range" in r or "Local window" in r)
    check("R1 lost at view construction", r1_view_losses, 57)
    check("R1 lost elsewhere", len(records) - r1_view_losses, 31)
    check("R1 processed plus view-construction losses", n["r1_processed"] + r1_view_losses, 1159)
    check("R1 recovered over processed plus view losses %",
          round(n["r1_recovered"] / (n["r1_processed"] + r1_view_losses) * 100, 1), 59.1)

    failed_screen = ~screened.overall_pass.astype(bool)
    systematic = screened.flags_systematic.astype(bool)
    alias = screened.alias_found.astype(bool)
    eclipse = screened.flags_eb.astype(bool)
    check("R2 screening failures", int(failed_screen.sum()), 20)
    check("  flagged systematic", int(systematic.sum()), 12)
    check("  failing sector reproduction", int(no_reproduce.sum()), 11)
    check("  failing both", int((systematic & no_reproduce).sum()), 4)
    check("  failing only the alias check", int((alias & ~systematic & ~no_reproduce & ~eclipse).sum()), 1)
    check("  secondary-eclipse flags", int(eclipse.sum()), 2)
    check("  secondary-eclipse flags inside the systematic set", bool((systematic | ~eclipse).all()), True)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from validate_candidates import TESS_SYSTEMATIC_PERIODS, check_systematic_period
    check("screening period set holds the 6.85 d half orbit", TESS_SYSTEMATIC_PERIODS.get("TESS orbit half"), 6.85)
    recomputed = screened.period_days.map(lambda p: check_systematic_period(p)["flags_systematic"]).astype(bool)
    check("systematic flags recomputed from that set", int((recomputed == systematic).sum()), len(screened))

    check("R2 >= 0.9 rejected: odd-even only", n["r2_high_rejected_odd_even_only"], 316)
    check("R2 >= 0.9 rejected: depth cap only", n["r2_high_rejected_depth_cap_only"], 41)
    check("R2 >= 0.9 rejected: both", n["r2_high_rejected_both"], 4)

    for width, want in ((1, 60), (2, 54), (3, 53)):
        check(f"R1 vetting pass reweighted to R3 S/N, width-{width} bins %",
              int(round(n[f"r1_reweighted_pass_rate_w{width}"] * 100)), want)

    check("classifier FN rate at threshold 0.3 %", round(n["r1_classifier_fn_rate_03"] * 100, 1), 20.4)
    check("threshold where classifier FN first reaches vetting FR", n["r1_crossover_threshold"], 0.345, tol=1e-9)

    print("\n---- CAMERA-READY: TRAINING AND KNOCKOUT ----")
    trained = pd.DataFrame([
        {"group": run["spec"]["group"], "variant": run["spec"]["variant"], "seed": int(run["spec"]["seed"]),
         "pr_auc": float(run["test"]["pr_auc"]), "best_epoch": int(run["best_epoch"]),
         "test_logit_std": float(run["test_logit_std"])}
        for run in (json.loads(p.read_text()) for p in sorted(Path("results/camera_ready/training/runs").glob("*/*.json")))
    ])
    knockout = pd.read_csv("results/camera_ready/analysis/branch_knockout.csv")

    def runs_of(group: str, variant: str) -> pd.DataFrame:
        return trained[(trained.group == group) & (trained.variant == variant)].sort_values("seed")

    def welch(a: pd.Series, b: pd.Series) -> tuple[float, float, float, float]:
        """Mean difference a - b with its Welch 95% CI and two-sided p-value."""
        var_a, var_b = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
        diff, err = a.mean() - b.mean(), np.sqrt(var_a + var_b)
        dof = (var_a + var_b) ** 2 / (var_a ** 2 / (len(a) - 1) + var_b ** 2 / (len(b) - 1))
        half = stats.t.ppf(0.975, dof) * err
        return float(diff), float(diff - half), float(diff + half), float(2 * stats.t.sf(abs(diff / err), dof))

    # A test-logit spread this small means one constant score for every test example.
    check("ablation global-only seeds with test logit s.d. < 1e-6",
          int((runs_of("ablation", "global-only").test_logit_std < 1e-6).sum()), 10)

    long_both = runs_of("ablation-long", "both").pr_auc
    long_local = runs_of("ablation-long", "local-only").pr_auc
    check("ablation-long both PR-AUC mean", round(float(long_both.mean()), 4), 0.9448)
    check("ablation-long both PR-AUC s.d.", round(float(long_both.std(ddof=1)), 4), 0.0061)
    check("ablation-long local-only PR-AUC mean", round(float(long_local.mean()), 4), 0.9442)
    check("ablation-long local-only PR-AUC s.d.", round(float(long_local.std(ddof=1)), 4), 0.0031)
    diff, lower, upper, p_long = welch(long_both, long_local)
    check("ablation-long both minus local-only", round(diff, 4), 0.0007)
    check("  95% CI lower bound", round(lower, 4), -0.0039)
    check("  95% CI upper bound", round(upper, 4), 0.0053)
    check("  Welch p-value", round(p_long, 2), 0.75)

    tuning = trained[trained.group == "global-tuning"].groupby("variant").pr_auc.mean()
    with_bn = tuning[tuning.index.str.contains("_bn1_")]
    without_bn = tuning[tuning.index.str.contains("_bn0_")]
    check("global-tuning bn1 configs with mean PR-AUC >= 0.90 (of 6)", int((with_bn >= 0.90).sum()), 6)
    check("global-tuning bn0 configs with mean PR-AUC <= 0.50 (of 6)", int((without_bn <= 0.50).sum()), 6)

    for variant, (want_mean, want_sd) in (("both", (0.9561, 0.0039)), ("local-only", (0.9381, 0.0040)),
                                          ("global-only", (0.9281, 0.0073))):
        pr = runs_of("bn-ablation", variant).pr_auc
        check(f"bn-ablation {variant} PR-AUC mean", round(float(pr.mean()), 4), want_mean)
        check(f"bn-ablation {variant} PR-AUC s.d.", round(float(pr.std(ddof=1)), 4), want_sd)
    diff, lower, upper, p_bn = welch(runs_of("bn-ablation", "both").pr_auc, runs_of("bn-ablation", "local-only").pr_auc)
    check("bn-ablation both minus local-only", round(diff, 4), 0.0180)
    check("  95% CI lower bound", round(lower, 4), 0.0143)
    check("  95% CI upper bound", round(upper, 4), 0.0217)
    check("  Welch p < 1e-7", bool(p_bn < 1e-7), True)
    # Same spec and seeds as global-tuning lr1e-4_bn1_flatten, so the runs must coincide exactly.
    bn_global = runs_of("bn-ablation", "global-only").head(3)
    tuned = runs_of("global-tuning", "lr1e-4_bn1_flatten").head(3)
    check("bn-ablation global-only seeds 0-2 equal global-tuning",
          bool(list(bn_global.seed) == list(tuned.seed) == [0, 1, 2]
               and np.array_equal(bn_global.pr_auc.to_numpy(), tuned.pr_auc.to_numpy())
               and np.array_equal(bn_global.best_epoch.to_numpy(), tuned.best_epoch.to_numpy())), True)

    global_conditions = knockout.condition.isin(["global_zero", "global_shuffle"])
    released_ko = knockout[(knockout.group == "released") & global_conditions]
    check("released: global knockout max |dlogit| < 1e-5",
          bool(len(released_ko) == 2 and (released_ko.max_abs_dlogit < 1e-5).all()), True)
    ablation_ko = knockout[(knockout.group == "ablation") & (knockout.variant == "both")]
    dead_global = ablation_ko[global_conditions.loc[ablation_ko.index]].groupby("seed").max_abs_dlogit.max() < 1e-5
    check("ablation both: seeds with global knockout max |dlogit| < 1e-5", int(dead_global.sum()), 9)
    check("  the exception", [int(s) for s in dead_global.index[~dead_global]], [2])
    seed2_shuffle = ablation_ko[(ablation_ko.seed == 2) & (ablation_ko.condition == "global_shuffle")]
    check("  seed 2 global_shuffle PR-AUC drop", round(float(seed2_shuffle.pr_auc_drop.iloc[0]), 4), 0.0685)
    # The knockout picks the seeds; the PR-AUCs come from the training runs, as for local-only.
    nine = runs_of("ablation", "both")
    nine = nine[nine.seed.isin([int(s) for s in dead_global.index[dead_global]])].pr_auc
    check("nine-seed ablation both PR-AUC mean", round(float(nine.mean()), 4), 0.9421)
    check("nine-seed ablation both PR-AUC s.d.", round(float(nine.std(ddof=1)), 4), 0.0026)
    diff, lower, upper, p_nine = welch(nine, runs_of("ablation", "local-only").pr_auc)
    check("nine-seed both minus local-only", round(diff, 4), -0.0013)
    check("  95% CI lower bound", round(lower, 4), -0.0041)
    check("  95% CI upper bound", round(upper, 4), 0.0015)
    check("  Welch p-value", round(p_nine, 2), 0.35)
    bn_ko = knockout[(knockout.group == "bn-ablation") & (knockout.variant == "both")]
    for condition, (want_mean, want_sd) in (("global_shuffle", (0.388, 0.033)), ("local_shuffle", (0.086, 0.024))):
        drop = bn_ko[bn_ko.condition == condition].pr_auc_drop
        check(f"bn-ablation both {condition} PR-AUC drop mean", round(float(drop.mean()), 3), want_mean)
        check(f"bn-ablation both {condition} PR-AUC drop s.d.", round(float(drop.std(ddof=1)), 3), want_sd)

    camera_ready_paper_draft1(n, r1, r2, r3, confirmed, welch)

    print("\n" + "-" * 60)
    print(f"{_passed} passed, {len(_failed)} failed")
    if _failed:
        print("\n  FAILURES:")
        for name in _failed:
            print(f"- {name}")
        sys.exit(1)
    print("  Every number in the paper reproduces from committed artefacts.")
    print("-" * 60)


# ---- CAMERA-READY: PAPER DRAFT 1 ----

RELEASED_CKPT = "checkpoints/stellaris_best.pt"


def half_up(value: float, digits: int = 0) -> float:
    """Round half away from zero, as the paper prints (10.25 -> 10.3); round() rounds half to even."""
    return float(Decimal(repr(float(value))).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP))


def pct(k: int, total: int, digits: int = 1) -> float:
    """k/total as a percentage, rounded half-up from the exact fraction."""
    exact = Decimal(int(k)) * 100 / Decimal(int(total))
    return float(exact.quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP))


def sci(p: float, digits: int = 0) -> str:
    """A p-value as the paper prints m x 10^e: '7e-09' is 7 x 10^-9."""
    return f"{p:.{digits}e}"


def source_numbers(path: str, pattern: str) -> list[float]:
    """Every number the regex captures in a committed source file, for literals that are not module constants."""
    found = re.findall(pattern, Path(path).read_text())
    return [float(x) for m in found for x in (m if isinstance(m, tuple) else (m,))]


def xgboost_feature_count() -> int:
    """Entries in train_xgboost_baseline.extract_features' feature list, read without importing the script."""
    tree = ast.parse(Path("scripts/train_xgboost_baseline.py").read_text())
    func = next(f for f in ast.walk(tree) if isinstance(f, ast.FunctionDef) and f.name == "extract_features")
    lists = [a.value for a in ast.walk(func) if isinstance(a, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == "features" for t in a.targets)]
    return len(lists[0].elts) if len(lists) == 1 and isinstance(lists[0], ast.List) else -1


def kepler_detrend_window_days() -> float:
    """The window preprocess.detrend hands to wotan, captured with a stand-in wotan module."""
    import lightkurve as lk
    from stellaris.preprocess import detrend

    seen: dict = {}

    def flatten(time, flux, **kwargs):
        seen.update(kwargs)
        return flux, flux

    stand_in = types.ModuleType("wotan")
    stand_in.flatten = flatten
    with mock.patch.dict(sys.modules, {"wotan": stand_in}):
        detrend(lk.LightCurve(time=np.arange(10.0), flux=np.ones(10)))
    return float(seen["window_length"])


def screening_set(frame: pd.DataFrame) -> set | None:
    """The 30 highest-scoring endorsements with BLS S/N >= 10; None if rank 30 is tied."""
    eligible = (frame[(frame.cnn_probability >= 0.9) & frame.vetted & (frame.bls_snr >= 10)]
                .sort_values("cnn_probability", ascending=False))
    if len(eligible) > 30 and eligible.cnn_probability.iloc[29] == eligible.cnn_probability.iloc[30]:
        return None
    return set(eligible.tic_id.head(30))


def released_rescore(path: str) -> pd.DataFrame:
    """Released-checkpoint rescore rows with status ok, one per target."""
    frame = pd.read_csv(path)
    frame = frame[(frame.checkpoint == RELEASED_CKPT) & (frame.status == "ok")]
    frame = frame.drop_duplicates("tic_id", keep="last").set_index("tic_id")
    frame["vetted"] = (frame.vetting_passed.astype(str).str.strip().str.lower()
                       .map({"true": True, "false": False}).astype(bool))
    return frame


def failure_records(path: str) -> int:
    """Records in a hunt failure log; a record starts with '<tic>\\t'."""
    return sum(1 for line in Path(path).read_text().splitlines() if re.match(r"^\d+\t", line))


def camera_ready_paper_draft1(n: dict, r1: pd.DataFrame, r2: pd.DataFrame, r3: pd.DataFrame,
                              confirmed: pd.DataFrame, welch) -> None:
    """Numbers in camera-ready draft 1 not checked above, grouped by paper section."""
    from stellaris import preprocess as kepler_preprocess
    from stellaris import tess_preprocess, views, vetting
    from stellaris.dataset import make_splits
    from stellaris.rescore import CATALOGUE_DISPOSITIONS, select_toi
    from stellaris.targets import read_tic_file
    from validate_candidates import SYSTEMATIC_TOLERANCE, TESS_SYSTEMATIC_PERIODS, check_systematic_period

    print("\n---- CAMERA-READY: PAPER DRAFT 1 ----")

    def section(title: str) -> None:
        print(f"\n  [{title}]")

    def check_ms(label: str, values, want: tuple, digits: int) -> None:
        values = pd.Series(values, dtype=float)
        check(f"{label}, mean", half_up(values.mean(), digits), want[0])
        check(f"{label}, s.d.", half_up(values.std(ddof=1), digits), want[1])

    def check_welch(label: str, a, b, want: tuple, digits: int, want_p, p_digits: int) -> None:
        """want is (difference, CI low, CI high); a str want_p is compared in m x 10^e form."""
        diff, low, high, p = welch(pd.Series(a, dtype=float).reset_index(drop=True),
                                   pd.Series(b, dtype=float).reset_index(drop=True))
        check(f"{label}: difference", half_up(diff, digits), want[0])
        check(f"{label}: 95% CI lower", half_up(low, digits), want[1])
        check(f"{label}: 95% CI upper", half_up(high, digits), want[2])
        check(f"{label}: p", sci(p, p_digits) if isinstance(want_p, str) else half_up(p, p_digits), want_p)

    runs = pd.DataFrame([{
        "group": run["spec"]["group"], "variant": run["spec"]["variant"], "seed": int(run["spec"]["seed"]),
        "epochs": run["spec"]["epochs"], "lr": run["spec"]["lr"], "weight_decay": run["spec"]["weight_decay"],
        "batch_size": run["spec"]["batch_size"], "batch_norm": bool(run["spec"]["batch_norm"]),
        "pool": run["spec"]["pool"], "n_scalars": run["spec"]["n_scalars"], "pos_weight": run["spec"]["pos_weight"],
        "split_seed": run["spec"]["split_seed"], "n_params": run["n_params"],
        "pr_auc": run["test"]["pr_auc"], "roc_auc": run["test"]["roc_auc"], "r95": run["test"]["recall_at_95"],
        "test_logit_std": run["test_logit_std"], "val_logit_std": run["best_val"]["logit_std"],
    } for run in (json.loads(p.read_text())
                  for p in sorted(Path("results/camera_ready/training/runs").glob("*/*.json")))])

    def seeds(group: str, variant: str) -> pd.DataFrame:
        return runs[(runs.group == group) & (runs.variant == variant)].sort_values("seed")

    per_ckpt = pd.read_csv("results/camera_ready/analysis/rescore_per_checkpoint_bls_full.csv")
    by_variant = pd.read_csv("results/camera_ready/analysis/rescore_by_variant_bls_full.csv")

    def rescored(group: str, variant: str) -> pd.DataFrame:
        return per_ckpt[(per_ckpt.group == group) & (per_ckpt.variant == variant)].sort_values("seed")

    knockout = pd.read_csv("results/camera_ready/analysis/branch_knockout.csv")
    released_ko = knockout[knockout.group == "released"].set_index("condition")
    scalar_ko = pd.read_csv("results/camera_ready/analysis/scalar_knockout.csv")
    swap = scalar_ko[scalar_ko.part == "tess_depth_swap"]
    perm = scalar_ko[scalar_ko.part == "kepler_permutation"]
    negatives = pd.read_csv("results/camera_ready/analysis/negatives_fp_flags.csv").set_index("split")
    trainset_v2 = json.loads(Path("results/camera_ready/trainset_v2_comparison.json").read_text())
    screened = pd.read_csv("results/validation/validation_summary.csv")

    inj = pd.read_csv("results/camera_ready/injection/r3_injections.csv")
    for col in ("period_match", "in_grid", "recovered_05", "endorsed_09"):
        inj[col] = inj[col].astype(str).str.strip().str.lower().eq("true")
    inj_vetted = inj.vetting_passed.astype(str).str.strip().str.lower().eq("true")
    snr = inj.expected_snr
    scored = inj.stage == "scored"

    bls_full = pd.concat([pd.read_csv(f"results/camera_ready/bls_full/{r}_bls_full.csv").assign(regime=r)
                          for r in ("r1", "r2", "r3")], ignore_index=True)
    rebuilt = bls_full[bls_full.status == "ok"]
    identical = rebuilt.matches_hunt_csv.astype(str).eq("True")

    count = lambda net: sum(p.numel() for p in net.parameters())
    net = StellarisNetwork()
    local_params = count(StellarisNetwork(use_global=False))
    global_only = count(net) - local_params
    test_size = len(np.load("results/test_predictions.npz")["labels"])
    r3_view_losses = sum(1 for line in Path("results/novel_failures.log").read_text().splitlines()
                         if "exceeds phase range" in line or "Local window" in line)

    # --- Abstract ---
    section("Abstract")
    check('"its 8.4M-parameter global branch" (also Sec. 2, Sec. 3)', half_up(global_only / 1e6, 1), 8.4)
    _, _, _, p_bn = welch(seeds("bn-ablation", "both").pr_auc.reset_index(drop=True),
                          seeds("bn-ablation", "local-only").pr_auc.reset_index(drop=True))
    check('"two views beat one by 0.018 PR-AUC (p = 7 x 10^-9)" (also Sec. 3): p', sci(p_bn), "7e-09")

    # --- Sec. 2 Method ---
    section("Sec. 2 Method: data and views")
    koi = pd.read_csv("data/catalogs/kepler_koi.csv", comment="#")
    check('"6,726 of the 7,586 such objects": CONFIRMED + FALSE POSITIVE KOIs',
          int(koi.koi_disposition.isin(["CONFIRMED", "FALSE POSITIVE"]).sum()), 7586)
    check('"6,726 of the 7,586 such objects": examples with views', trainset_v2["v1_rows"], 6726)
    # train_variants.py sets pos_weight = n_neg / n_pos on the training split, so it fixes the positives.
    pos_weight = runs.pos_weight.unique()
    train_neg = int(negatives.loc["train", "n_negatives"])
    train_pos = int(round(train_neg / pos_weight[0]))
    check("  one pos_weight across runs, equal to train negatives / positives",
          bool(len(pos_weight) == 1 and abs(train_neg / train_pos - pos_weight[0]) < 1e-12), True)
    check('"4,680 train" (negatives + positives)', train_neg + train_pos, 4680)
    check('"1,026 validation" (6,726 - train - test)', trainset_v2["v1_rows"] - train_neg - train_pos - test_size, 1026)
    check('"one-sided 5 sigma clip": Kepler', kepler_preprocess.SIGMA_CLIP, 5.0)
    check('"one-sided 5 sigma clip": TESS', tess_preprocess.SIGMA_CLIP, 5.0)
    check('"detrend with a 1.0-day running median": Kepler window', kepler_detrend_window_days(), 1.0)
    check('"detrend with a 1.0-day running median": TESS window', tess_preprocess.DETREND_WINDOW_DAYS, 1.0)
    check('"a global view of 2,001 bins"', views.GLOBAL_VIEW_BINS, 2001)
    check('"a local view of 201 bins"', views.LOCAL_VIEW_BINS, 201)
    check('"covering +/-4 transit durations" (also Sec. 5, App. H)', views.LOCAL_VIEW_NUM_DURATIONS, 4)
    check('  views refuse a window past half a phase (0.5)',
          source_numbers("src/stellaris/views.py", r"if half_window >= ([\d.]+):"), [0.5])

    section("Sec. 2 Method: architecture and training")
    convs = [m for m in net.modules() if isinstance(m, torch.nn.Conv1d)]
    pools = [m for m in net.modules() if isinstance(m, torch.nn.MaxPool1d)]
    check('"two Conv1d layers (kernel 5": kernel sizes', sorted({c.kernel_size[0] for c in convs}), [5])
    check('"same padding": padding k//2 at stride 1',
          all(c.padding[0] == c.kernel_size[0] // 2 and c.stride[0] == 1 for c in convs), True)
    check('"max-pooling (kernel 5, stride 2)"', sorted({(p.kernel_size, p.stride) for p in pools}), [(5, 2)])
    channels = lambda branch: [branch.blocks[0].conv1.in_channels] + [b.conv2.out_channels for b in branch.blocks]
    check('"the global branch stacks five (1->16->32->64->128->256)"', channels(net.global_branch),
          [1, 16, 32, 64, 128, 256])
    check('"the local branch two (1->16->32)"', channels(net.local_branch), [1, 16, 32])
    with torch.no_grad():
        global_width = net.global_branch(torch.zeros(1, 1, views.GLOBAL_VIEW_BINS)).shape[1]
        local_width = net.local_branch(torch.zeros(1, 1, views.LOCAL_VIEW_BINS)).shape[1]
    check('"The flattened outputs, 15,104 and 1,536 wide" (also App. B)', [global_width, local_width], [15104, 1536])
    dense = [m for m in net.head if isinstance(m, torch.nn.Linear)]
    check('"three dense layers of width 512 ... then a single logit"', [d.out_features for d in dense], [512, 512, 512, 1])
    check('"with dropout 0.5"', sorted({m.p for m in net.head if isinstance(m, torch.nn.Dropout)}), [0.5])
    check('"There is no batch normalisation"', any(isinstance(m, torch.nn.BatchNorm1d) for m in net.modules()), False)
    global_convs = count(net.global_branch)
    # The head concatenates global features first, so these columns see only the global view.
    global_dense = dense[0].weight[:, :global_width].numel()
    check('"0.66M in its convolutions" (M)', half_up(global_convs / 1e6, 2), 0.66)
    check('"7.73M in the first dense layer\'s weights on its flattened output" (M)', half_up(global_dense / 1e6, 2), 7.73)
    check('  the two make up the 8.4M global-only parameters', global_convs + global_dense, global_only)
    check('"positive weight ~1.46"', half_up(pos_weight[0], 2), 1.46)
    ten_seed = runs[runs.group.isin(["ablation", "ablation-long", "bn-ablation", "scalars"])]
    check('"learning rate 10^-4" (ten-seed runs)', sorted(ten_seed.lr.unique().tolist()), [1e-4])
    check('"weight decay 10^-5" (all runs)', sorted(runs.weight_decay.unique().tolist()), [1e-5])
    check('"batch size 64" (all runs)', sorted(runs.batch_size.unique().tolist()), [64])
    check('"for 40 epochs" (ablation and scalar runs)',
          sorted(runs[runs.group.isin(["ablation", "scalars"])].epochs.unique().tolist()), [40])
    check('"Seed-controlled runs use seeds 0 to 9" (also App. A)',
          all(sorted(g.seed) == list(range(10)) for _, g in ten_seed.groupby(["group", "variant"])), True)

    section("Sec. 2 Method: TESS deployment and screening")
    check('"two-minute SPOC light curves": fetch exptime (s)',
          source_numbers("src/stellaris/tess_fetch.py", r"exptime: int = (\d+)"), [120])
    check('"require 5,000 surviving cadences": hunting.py floor',
          source_numbers("scripts/hunting.py", r"len\(time_arr\) < (\d+)"), [5000])
    check("  fewest cadences among rebuilt targets is at least 5,000", bool(rebuilt.n_cadences.min() >= 5000), True)
    # A period from the search sits at an integer index of the 50,000-point log grid over 0.5-20 d.
    grid_index = np.log(rebuilt.period_days / 0.5) / np.log(20 / 0.5) * (50_000 - 1)
    check('"50,000 log-spaced periods (0.5 to 20 d)": all BLS periods on that grid',
          bool(np.abs(grid_index - grid_index.round()).max() < 1e-6 and grid_index.min() > -1e-6
               and grid_index.max() < 49_999 + 1e-6), True)
    check('"eight durations (0.5 to 8 h)": durations BLS returned',
          sorted(rebuilt.duration_hours.round(6).unique().tolist()), [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0])
    check('"a depth cap rejecting events deeper than 3%"', vetting.MAX_DEPTH_FRAC, 0.03)
    check('"alternating depths differ by more than 15%"', vetting.MAX_ODD_EVEN_DEPTH_DIFF, 0.15)
    check('"highest-ranked endorsements with BLS S/N >= 10 are then screened": top 30 = screened set',
          screening_set(r2) == set(screened.tic_id), True)
    check('"proximity within 5%"', SYSTEMATIC_TOLERANCE, 0.05)
    check('"the instrumental-period set {1, 2, 6.85, 13.7, 27.4} d"', sorted(TESS_SYSTEMATIC_PERIODS.values()),
          [1.0, 2.0, 6.85, 13.7, 27.4])
    multiples = re.findall(r"for n in \[([^\]]*)\]", Path("scripts/validate_candidates.py").read_text())
    check('"at half, unit, double and triple period"',
          [[float(x) for x in m.split(",")] for m in multiples], [[0.5, 1.0, 2.0, 3.0]])
    check('"data separated by gaps of more than 5 d"',
          source_numbers("scripts/validate_candidates.py", r"gaps > (\d+)\)"), [5])

    section("Sec. 2 Method: rescoring at matched thresholds")
    check('"identical to the original search for 5,242 of 5,504 targets": identical', int(identical.sum()), 5242)
    check('"... of 5,504 targets" (also App. A "5,504 targets we could rebuild")', len(rebuilt), 5504)
    released_r1 = released_rescore("results/camera_ready/rescore/r1_bls_full.csv")
    check('"as the released model does at 0.5 (74.1%)"',
          pct((released_r1.prob >= 0.5).sum(), len(released_r1)), 74.1)
    check("  every rescored checkpoint passes that same R1 fraction at its t*",
          bool(per_ckpt.r1_pass_rate.max() - per_ckpt.r1_pass_rate.min() == 0
               and abs(per_ckpt.r1_pass_rate.iloc[0] - (released_r1.prob >= 0.5).mean()) < 1e-12), True)

    # --- Sec. 3 Probe 1 ---
    section("Sec. 3 Probe 1 and Table 1")
    check('"19 hand-engineered statistics" (also Table 1)', xgboost_feature_count(), 19)
    check('"a local-view-only network of 1.32M parameters" (M)', half_up(local_params / 1e6, 2), 1.32)
    long_runs = runs[runs.group.isin(["ablation-long", "bn-ablation", "global-tuning"])]
    check('"training for 100 epochs instead of 40" (also Table 1, App. B)', sorted(long_runs.epochs.unique().tolist()), [100])
    working = runs[(runs.group == "ablation") & runs.variant.isin(["both", "local-only"])].test_logit_std
    check('"against 3.2 to 3.8 for working models": lowest', half_up(working.min(), 1), 3.2)
    check('"against 3.2 to 3.8 for working models": highest', half_up(working.max(), 1), 3.8)
    tuning = runs[runs.group == "global-tuning"]
    check('"Across learning rates from 10^-5 to 3 x 10^-4"', sorted(tuning.lr.unique().tolist()), [1e-5, 1e-4, 3e-4])
    check('"and two pooling heads"', sorted(tuning.pool.unique()), ["flatten", "gap"])
    tuned_bn = tuning[tuning.batch_norm].groupby("variant").pr_auc.mean()
    check('"every one with it reaches 0.91 to 0.93" (also App. B): lowest', half_up(tuned_bn.min(), 2), 0.91)
    check('"every one with it reaches 0.91 to 0.93" (also App. B): highest', half_up(tuned_bn.max(), 2), 0.93)
    table1 = (
        ("global view only", "ablation", "global-only", {}, 8_914_737,
         {"pr": (0.433, 0.026), "roc": (0.551, 0.046), "r95": (0.000, 0.000)}),
        ("local view only", "ablation", "local-only", {"use_global": False}, 1_321_905,
         {"pr": (0.943, 0.003), "roc": (0.958, 0.002), "r95": (0.685, 0.026)}),
        ("both views", "ablation", "both", {}, 9_710_305,
         {"pr": (0.944, 0.006), "roc": (0.959, 0.004), "r95": (0.674, 0.047)}),
        ("BN global view only", "bn-ablation", "global-only", {"batch_norm": True}, 8_916_721,
         {"pr": (0.928, 0.007), "roc": (0.949, 0.005), "r95": (0.606, 0.041)}),
        ("BN local view only", "bn-ablation", "local-only", {"batch_norm": True, "use_global": False}, 1_322_097,
         {"pr": (0.938, 0.004), "roc": (0.955, 0.002), "r95": (0.666, 0.038)}),
        ("BN both views", "bn-ablation", "both", {"batch_norm": True}, 9_712_481,
         {"pr": (0.956, 0.004), "roc": (0.967, 0.003), "r95": (0.764, 0.051)}),
    )
    for label, group, variant, kwargs, params, want in table1:
        sub = seeds(group, variant)
        if variant == "global-only":
            kwargs = {**kwargs, "use_local": False}
        check(f"Table 1 {label}: ten seeds", len(sub), 10)
        check(f"Table 1 {label}: Params (model, run JSON)",
              [count(StellarisNetwork(**kwargs)), *sub.n_params.unique().tolist()], [params, params])
        for metric, column, key in (("PR-AUC", "pr_auc", "pr"), ("ROC-AUC", "roc_auc", "roc"), ("R@95%P", "r95", "r95")):
            check_ms(f"Table 1 {label}: {metric}", sub[column], want[key], 3)
    check('"changes no logit by more than 5 x 10^-7" (released, global zero and shuffle)',
          bool(released_ko.loc[["global_zero", "global_shuffle"], "max_abs_dlogit"].max() <= 5e-7), True)
    check('"shuffling the local input drops PR-AUC by 0.549" (also App. B)',
          half_up(released_ko.loc["local_shuffle", "pr_auc_drop"], 3), 0.549)
    seed2 = float(seeds("ablation", "both").set_index("seed").pr_auc[2])
    check('"The one full model whose global branch trained (seed 2) ... (0.960)"', half_up(seed2, 3), 0.960)
    plain = runs[~runs.batch_norm & (runs.n_scalars == 0)]
    check('"is the best model without batch normalisation or scalar inputs"', bool(seed2 >= plain.pr_auc.max()), True)
    check('"in effect a 1.3M-parameter local-view model" (M)', half_up(local_params / 1e6, 1), 1.3)

    # --- Sec. 4 Probe 2 ---
    section("Sec. 4 Probe 2: Table 2 and R1")
    for regime, path, processed, log, want in (
        ("R1", "data/catalogs/tess_validation_confirmed.txt", len(r1), "results/tier1_failures.log", 1190),
        ("R2", "data/catalogs/tess_discovery_unresolved.txt", len(r2), "results/tier3_failures.log", 5107),
        ("R3", "data/catalogs/novel_targets.txt", len(r3), "results/novel_failures.log", 2000),
    ):
        check(f"Table 2 {regime} Attempted (target list)", len(set(read_tic_file(Path(path)))), want)
        check(f"  {regime} processed + failure-log records", processed + failure_records(log), want)
    check('"of the 88 losses"', failure_records("results/tier1_failures.log"), 88)
    check('"the classifier\'s false-negative rate, 286/1,102"', int((r1.cnn_probability < 0.5).sum()), 286)
    check('"the vetting\'s false-rejection rate, 238/1,102"', int((~r1.vetted).sum()), 238)
    check('"of the 310 discordant cases"', n["r1_lost_classifier_only"] + n["r1_lost_vetting_only"], 310)
    for threshold, want in ((0.1, 9.8), (0.345, 21.6), (0.9, 47.3)):
        check(f'"The classifier\'s rate is ... at threshold {threshold}" (Table 6)',
              pct((r1.cnn_probability < threshold).sum(), len(r1)), want)
    reasons = r1.vetting_reasons.fillna("")
    odd_even_pct = reasons.str.extract(r"odd/even depth differ by ([\d.]+)%")[0].astype(float)
    depth_cap = reasons.str.contains("cap")
    for tolerance, want in ((20, 17.4), (25, 15.3), (50, 10.6)):
        rejected = depth_cap | (odd_even_pct > tolerance)
        check(f'"relaxing the odd-even tolerance" to {tolerance}%: vetting rate (Table 6)',
              pct(rejected.sum(), len(r1)), want)
    # The 2% rule of tess_numbers, applied at a period ratio of `factor` (0.5: BLS found half the period).
    catalogue_periods = confirmed.groupby("TIC ID")["Period (days)"].apply(lambda s: s.to_numpy(dtype=float))

    def bls_within(factor: float) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.array([tic in catalogue_periods.index
                             and bool((np.abs(period / catalogue_periods[tic] / factor - 1) <= 0.02).any())
                             for tic, period in zip(r1.tic_id, r1.period_days)])

    bls_match = bls_within(1.0)
    half_alias = bls_within(0.5) & ~bls_match
    odd_even = reasons.str.contains("odd/even").to_numpy()
    check("  the 2% rule reproduces the 936 BLS matches", int(bls_match.sum()), n["r1_bls_matches"])
    check('"all 33 half-period BLS aliases": count', int(half_alias.sum()), 33)
    check('"all 33 half-period BLS aliases are odd-even rejections"', int((half_alias & odd_even).sum()), 33)
    check('"113 of the 215 odd-even rejections fall on targets where BLS missed"',
          int((odd_even & ~bls_match).sum()), 113)
    check('"that net also catches ordinary three- and four-day periods"',
          [check_systematic_period(p)["flags_systematic"] for p in (3.0, 4.0)], [True, True])

    # --- Sec. 5 Probe 3 ---
    section("Sec. 5 Probe 3: R3, Fig. 1 and restoring inputs")
    endorsed_r3 = r3[(r3.cnn_probability >= 0.9) & r3.vetted]
    check('"One target is endorsed" (TIC 155810890)', endorsed_r3.tic_id.tolist(), [155810890])
    check('"admitted with the S/N floor waived": its BLS S/N is below 10', bool(n["tic155810890_bls_snr"] < 10), True)
    ratio = n["r3_median_depth_ppm"] / n["r1_median_depth_ppm"]
    check('"nearly double R1\'s": R3/R1 median depth in [1.9, 2.0)', bool(1.9 <= ratio < 2.0), True)
    check('Fig. 1b caption "3,958 injections with a light curve"', int((inj.stage != "no_fits").sum()), 3958)
    released_r2 = released_rescore("results/camera_ready/rescore/r2_bls_full.csv")
    released_r3 = released_rescore("results/camera_ready/rescore/r3_bls_full.csv")
    r2_cap = released_r2.vetting_reasons.fillna("").str.contains("cap")
    r3_band = (released_r3.period_days - TESS_ORBIT_DAYS).abs() / TESS_ORBIT_DAYS <= 0.10
    check('"R3 detections within 10% of 13.7 d (of 195 rescored)"',
          [int(r3_band.sum()), *per_ckpt.r3_n_band.unique().tolist()], [195, 195])
    check('"the R2 targets rejected by the depth cap (of 120)"',
          [int(r2_cap.sum()), *per_ckpt.r2_depth_cap_rejections.unique().tolist()], [120, 120])
    check('"ten seeds" for each of A to D (rescored)',
          [len(rescored("scalars", v)) for v in "ABCD"], [10, 10, 10, 10])
    table3 = (
        ("Local view only", "ablation", "local-only", (0.943, 0.003), (18.1, 5.3), (62.2, 1.5)),
        ("Both views, batch norm.", "bn-ablation", "both", (0.956, 0.004), (46.0, 9.2), (63.5, 3.0)),
        ("A", "scalars", "A", (0.944, 0.006), (16.0, 2.9), (62.7, 3.3)),
        ("B", "scalars", "B", (0.947, 0.004), (15.5, 5.6), (64.0, 2.9)),
        ("C", "scalars", "C", (0.958, 0.003), (10.1, 3.1), (62.6, 4.0)),
        ("D", "scalars", "D", (0.977, 0.002), (12.1, 4.0), (65.7, 2.6)),
    )
    released_counts = rescored("released", "released").iloc[0]
    check('Table 3 "Released checkpoint": R3 near 13.7 d >= t*', int(released_counts.r3_high_band), 15)
    check('Table 3 "Released checkpoint": R2 depth-cap >= t*', int(released_counts.r2_high_among_depth_cap), 59)
    cap_means = [float(released_counts.r2_high_among_depth_cap)]
    for label, group, variant, kepler, band, cap in table3:
        sub = rescored(group, variant)
        check_ms(f'Table 3 "{label}": Kepler test PR-AUC', seeds(group, variant).pr_auc, kepler, 3)
        check_ms(f'Table 3 "{label}": R3 near 13.7 d >= t*', sub.r3_high_band, band, 1)
        check_ms(f'Table 3 "{label}": R2 depth-cap >= t*', sub.r2_high_among_depth_cap, cap, 1)
        cap_means.append(float(sub.r2_high_among_depth_cap.mean()))
    agg = per_ckpt.groupby(["group", "variant"])[["r3_high_band", "r2_high_among_depth_cap"]].agg(["mean", "std"])
    summary = by_variant.set_index(["group", "variant"])
    consistent = all(np.allclose(agg[(col, stat)].loc[summary.index].to_numpy(dtype=float),
                                 summary[f"{col}_{'mean' if stat == 'mean' else 'sd'}"].to_numpy(dtype=float),
                                 equal_nan=True)
                     for col in ("r3_high_band", "r2_high_among_depth_cap") for stat in ("mean", "std"))
    check("  rescore_by_variant_bls_full.csv agrees with the per-checkpoint table", consistent, True)
    a_, b_, c_, d_ = (seeds("scalars", v).pr_auc for v in "ABCD")
    tess = {v: rescored("scalars", v) for v in "ABCD"}
    for label, later, earlier, column, want in (
        ('"B minus A is -0.5 [-4.8, +3.8] in the 13.7-day band"', "B", "A", "r3_high_band", (-0.5, -4.8, 3.8)),
        ('"and +1.3 [-1.6, +4.2] among the deep R2 signals"', "B", "A", "r2_high_among_depth_cap", (1.3, -1.6, 4.2)),
        ('"move the 13.7-day tail, by -5.4 [-9.8, -1.0]"', "C", "B", "r3_high_band", (-5.4, -9.8, -1.0)),
    ):
        diff, low, high, _ = welch(tess[later][column].reset_index(drop=True), tess[earlier][column].reset_index(drop=True))
        check(label, [half_up(diff, 1), half_up(low, 1), half_up(high, 1)], list(want))
    diff_ba, _, _, p_ba = welch(b_.reset_index(drop=True), a_.reset_index(drop=True))
    check('"on Kepler B gains +0.003 PR-AUC (p = 0.18)"', [half_up(diff_ba, 3), half_up(p_ba, 2)], [0.003, 0.18])
    diff_r95, _, _, p_r95 = welch(seeds("scalars", "B").r95.reset_index(drop=True),
                                  seeds("scalars", "A").r95.reset_index(drop=True))
    check('"with recall at 95% precision +0.050 (p = 0.03"', [half_up(diff_r95, 3), half_up(p_r95, 2)], [0.050, 0.03])
    drop = lambda variant, scalar: perm[(perm.variant == variant) & (perm.scalar == scalar)].pr_auc_drop
    check_ms('"permuting it across the Kepler test set costs 0.0075 +/- 0.0032 PR-AUC"',
             drop("B", "log_local_scale"), (0.0075, 0.0032), 4)
    check('"Setting the depth input of the 120 deep R2 signals": rows per B seed, seeds',
          [sorted(swap.n.unique().tolist()), sorted(swap.seed.tolist())], [[120], list(range(10))])
    replacement = swap.depth_scalar_replacement.unique()
    check('"(about 530 ppm": 10^replacement to the nearest 10 ppm',
          half_up(10 ** replacement[0] * 1e6, -1) if len(replacement) == 1 else float("nan"), 530.0)
    real_median = float(np.log10(released_r2.local_scale[r2_cap]).median())
    check('"against their real median of 5.8%)" (from the released rescore)', half_up(10 ** real_median * 100, 1), 5.8)
    check_ms('"lowers B\'s scores (by 0.042 +/- 0.036 on average)"', -swap.mean_score_change, (0.042, 0.036), 3)
    check_ms('"the number at or above t* falls from 64.0 +/- 2.9"', swap.n_ge_tstar_real, (64.0, 2.9), 1)
    check_ms('"to 55.8 +/- 3.3"', swap.n_ge_tstar_replaced, (55.8, 3.3), 1)
    sequential_tess = pd.read_csv("results/camera_ready/analysis/scalars_sequential_tess.csv")
    check('"uncorrected for six TESS comparisons"',
          int(sequential_tess.comparison.isin(["B vs A", "C vs B", "D vs C"]).sum()), 6)
    tail_p = {pair: welch(tess[pair[0]].r3_high_band.reset_index(drop=True),
                          tess[pair[1]].r3_high_band.reset_index(drop=True))[3]
              for pair in (("B", "A"), ("C", "B"), ("D", "C"))}
    check('"Period and duration (C) are the only additions that move the 13.7-day tail" (p < 0.05 only for C)',
          [p < 0.05 for p in tail_p.values()], [False, True, False])
    check('"(p = 0.02, ..." for C minus B', half_up(tail_p[("C", "B")], 2), 0.02)
    check('"permuting either costs 0.06 to 0.07 PR-AUC": period, duration',
          [half_up(drop("C", "log_period").mean(), 2), half_up(drop("C", "log_duration").mean(), 2)], [0.06, 0.07])
    kepler_gain = {pair: welch(seeds("scalars", pair[0]).pr_auc.reset_index(drop=True),
                               seeds("scalars", pair[1]).pr_auc.reset_index(drop=True))
                   for pair in (("B", "A"), ("C", "B"), ("D", "C"))}
    check('"Stellar parameters (D) give the largest Kepler gain"',
          max(kepler_gain, key=lambda k: kepler_gain[k][0]), ("D", "C"))
    check('"+0.018 [+0.016, +0.020]"', [half_up(x, 3) for x in kepler_gain[("D", "C")][:3]], [0.018, 0.016, 0.020])
    d_tess_p = [welch(tess["D"][c].reset_index(drop=True), tess["C"][c].reset_index(drop=True))[3]
                for c in ("r3_high_band", "r2_high_among_depth_cap")]
    check('"and nothing further on TESS" (D minus C, p >= 0.05 on both counts)', [p >= 0.05 for p in d_tess_p], [True, True])
    check('"scores 59 to 66 of the 120 deep signals"', [half_up(min(cap_means)), half_up(max(cap_means))], [59.0, 66.0])
    bn = rescored("bn-ablation", "both")
    check('"(median score 0.43 in the band, 0.05 elsewhere)"',
          [half_up(bn.r3_median_prob_band.mean(), 2), half_up(bn.r3_median_prob_elsewhere.mean(), 2)], [0.43, 0.05])

    section("Sec. 5 Probe 3: injection-recovery")
    check('"two ... transits into each of the 2,000 R3 light curves"',
          inj.groupby("tic_id").size().value_counts().to_dict(), {2: 2000})
    injection_constants = source_numbers("scripts/injection_recovery.py",
                                         r"MIN_PERIOD, MAX_PERIOD = ([\d.]+), ([\d.]+)")
    check('"period log-uniform in 0.5 to 20 d": draw range', injection_constants, [0.5, 20.0])
    check('  every injected period in 0.5-20 d', bool(inj.inj_period_days.between(0.5, 20).all()), True)
    check('"depth log-uniform in 500 to 30,000 ppm": draw range (ppm)',
          [x * 1e6 for x in source_numbers("scripts/injection_recovery.py",
                                           r"DEPTH_RANGE = \(([\de.-]+), ([\de.-]+)\)")], [500.0, 30000.0])
    check('  every injected depth in 500-30,000 ppm', bool(inj.inj_depth_ppm.between(500, 30_000).all()), True)
    check('"impact parameter uniform below 0.9"',
          [source_numbers("scripts/injection_recovery.py", r"MAX_IMPACT = ([\d.]+)"),
           bool(((inj.inj_b >= 0) & (inj.inj_b < 0.9)).all())], [[0.9], True])
    check('"Half are lost before scoring"', pct((~scored).sum(), len(inj), 0), 50.0)
    check('"48.6% at view construction" (also App. D)', pct((inj.stage == "view").sum(), len(inj)), 48.6)
    check('"the failure that removes 76% of real R3 targets" (also App. H)', pct(r3_view_losses, N_NOVEL_TARGETS, 0), 76.0)
    check('"BLS finds the injected period for 34.9% of all 4,000"', [pct(inj.period_match.sum(), len(inj)), len(inj)],
          [34.9, 4000])
    check('"15.7% are recovered" (also Abstract)', pct(inj.recovered_05.sum(), len(inj)), 15.7)
    check('"and 10.3% endorsed"', pct(inj.endorsed_09.sum(), len(inj)), 10.3)
    check('"Completeness is zero below expected S/N 7" (also Abstract)', int(inj.recovered_05[snr < 7].sum()), 0)
    snr_edges = [0, 3, 5, 7, 10, 15, 20, 30, 50, 100, np.inf]
    snr_bin = pd.cut(snr, snr_edges, right=False)
    by_snr = inj.groupby(snr_bin, observed=False).agg(n=("tic_id", "size"), found=("period_match", "sum"),
                                                     recovered=("recovered_05", "sum"), endorsed=("endorsed_09", "sum"))
    near_half = [str(b) for b, r in by_snr.iterrows() if r.recovered / r.n >= 0.45]
    check('"reaches about 50% only at S/N 30 to 100" (bins with recovery >= 45%)', near_half,
          ["[30.0, 50.0)", "[50.0, 100.0)"])
    missed = scored & ~inj.period_match & inj.alias_factor.isna()
    near_orbit = (inj.bls_period_days - TESS_ORBIT_DAYS).abs() / TESS_ORBIT_DAYS <= 0.10
    check('"Where BLS missed an injected transit, 41.4%" (scored; neither period nor alias)',
          pct((missed & near_orbit).sum(), missed.sum()), 41.4)

    # --- Sec. 6 Discussion and limitations ---
    section("Sec. 6 Discussion and limitations")
    novel = read_tic_file(Path("data/catalogs/novel_targets.txt"))
    tmag: dict = {}
    sectors: dict = {}
    for path in sorted(Path("data/catalogs/spoc_target_lists").glob("all_targets_S*.csv")):
        sector = int(re.search(r"S(\d+)", path.name).group(1))
        lists = pd.read_csv(path, comment="#")
        for tic, mag in zip(lists.TICID.astype(int), lists.Tmag):
            tmag[tic] = mag
            sectors.setdefault(tic, set()).add(sector)
    mags = np.array([tmag.get(t, np.nan) for t in novel])
    check('"R3 is depleted by construction (T_mag 13 to 15": all R3 targets inside',
          bool(np.isfinite(mags).all() and mags.min() >= 13 and mags.max() <= 15), True)
    check('"multi-sector": every R3 target in at least two SPOC sector lists',
          min(len(sectors.get(t, ())) for t in novel) >= 2, True)
    toi_tics = set(pd.read_csv("data/catalogs/tess_toi.csv", comment="#")["TIC ID"].astype(int))
    check('"no TOI": R3 targets in the TOI table', len(set(novel) & toi_tics), 0)
    gap = n["r1_vetting_pass_rate"] - n["r3_vetting_pass_rate"]
    shares = [(n["r1_vetting_pass_rate"] - n[f"r1_reweighted_pass_rate_w{w}"]) / gap * 100 for w in (1, 2, 3)]
    check('"Differences in S/N explain 37-52%" (also App. G): lowest', half_up(min(shares)), 37.0)
    check('"Differences in S/N explain 37-52%" (also App. G): highest', half_up(max(shares)), 52.0)

    # --- Sec. 7 Responsible use ---
    section("Sec. 7 Responsible use")
    check('"the classifier assigns 0.99 to TIC 155810890"',
          half_up(float(r3.loc[r3.tic_id == 155810890, "cnn_probability"].iloc[0]), 2), 0.99)
    check('"ours recovers few transits below S/N 20" (under 5%)', bool(inj.recovered_05[snr < 20].mean() < 0.05), True)

    # --- Appendix A ---
    section("App. A: splits, seeds, truncated downloads")
    check('"random_state=42": run specs and make_splits default',
          [sorted(runs.split_seed.unique().tolist()), inspect_default(make_splits, "seed")], [[42], 42])
    check('"Rebuilding all 6,726 examples ... changes only those three rows": rows, global views changed',
          [trainset_v2["v2_rows"], trainset_v2["v2_rows"] - trainset_v2["aligned_global_view_equal_rows"]], [6726, 3])
    check("  no local view or kepid/label changes beyond those three",
          bool(trainset_v2["v2_rows"] - trainset_v2["aligned_local_view_equal_rows"] <= 3
               and trainset_v2["aligned_kepid_equal_rows"] == trainset_v2["aligned_label_equal_rows"] == 6726), True)
    check_ms('"variant A scores 0.9438 +/- 0.0060"', a_, (0.9438, 0.0060), 4)
    check_ms('"against 0.9439 +/- 0.0060 for the original full model"', seeds("ablation", "both").pr_auc, (0.9439, 0.0060), 4)

    section("App. A: sector order and Table 5")
    differs = rebuilt[~identical]
    check('"262 (26 in R1, 230 in R2, 6 in R3)" (also Sec. 6): rebuilt targets whose search differs',
          [len(differs)] + [int((differs.regime == r).sum()) for r in ("r1", "r2", "r3")], [262, 26, 230, 6])
    corrected = {r: load_regime(f"results/camera_ready/corrected_hunt/{r}_results.csv") for r in ("r1", "r2", "r3")}
    original = {"r1": r1, "r2": r2, "r3": r3}
    common = {r: original[r][original[r].tic_id.isin(corrected[r].tic_id)].reset_index(drop=True) for r in original}
    check('Table 5 "targets in both runs: 1,092 / 3,943 / 469"', [len(common[r]) for r in common], [1092, 3943, 469])
    paired = {r: common[r].merge(corrected[r], on="tic_id", suffixes=("", "_new")) for r in original}
    r1_pair = paired["r1"]
    r1_moved = ((r1_pair.depth_ppm != r1_pair.depth_ppm_new) | (r1_pair.bls_snr != r1_pair.bls_snr_new)
                | (r1_pair.period_days != r1_pair.period_days_new))
    check('"In R1 only depths and S/N move ..., with no period changes": period, epoch or duration changes',
          int(((r1_pair.period_days != r1_pair.period_days_new) | (r1_pair.epoch_btjd != r1_pair.epoch_btjd_new)
               | (r1_pair.duration_hours != r1_pair.duration_hours_new)).sum()), 0)
    check('"(median change 9 ppm)"',
          half_up((r1_pair.depth_ppm_new - r1_pair.depth_ppm).abs()[r1_moved].median()), 9.0)
    r23 = pd.concat([paired["r2"], paired["r3"]])
    check('"In R2 and R3, 21 periods change, 16 of them by more than 2%"',
          [int((r23.period_days != r23.period_days_new).sum()),
           int(((r23.period_days_new / r23.period_days - 1).abs() > 0.02).sum())], [21, 16])
    not_rebuilt = bls_full[bls_full.status != "ok"]
    check('"Seventy-four targets (10, 59 and 5 in R1 to R3) could not be rebuilt"',
          [len(not_rebuilt)] + [int((not_rebuilt.regime == r).sum()) for r in ("r1", "r2", "r3")], [74, 10, 59, 5])
    before = tess_numbers(common["r1"], common["r2"], common["r3"], confirmed)
    after = tess_numbers(corrected["r1"], corrected["r2"], corrected["r3"], confirmed)
    for key, label, want, digits in (
        ("r1_recovered", "R1 recovered", (680, 681), None),
        ("r1_lost_classifier_only", "R1 lost to classifier only", (178, 178), None),
        ("r1_lost_vetting_only", "R1 lost to vetting only", (129, 128), None),
        ("r1_mcnemar_p", "R1 McNemar exact p", (0.006, 0.005), 3),
        ("r1_crossover_threshold", "R1 threshold where classifier and vetting rates meet", (0.345, 0.342), 3),
        ("r2_endorsed_09", "R2 endorsed", (1540, 1545), None),
        ("r2_high_then_rejected", "R2 scoring >= 0.9 then rejected by the rules", (355, 351), None),
        ("r3_band_count", "R3 near-13.7 d detections", (195, 195), None),
        ("r3_scoring_above_09", "R3 scoring >= 0.9", (4, 4), None),
    ):
        got = [before[key], after[key]] if digits is None else [half_up(before[key], digits), half_up(after[key], digits)]
        check(f'Table 5 "{label}": original, time-ordered', got, list(want))
    original_set, corrected_set = screening_set(common["r2"]), screening_set(corrected["r2"])
    check('Table 5 "R2 screening set (top 30)": identical', original_set is not None and original_set == corrected_set, True)
    counts = [k for k, v in before.items() if isinstance(v, (int, np.integer)) and not isinstance(v, bool)]
    shift = max(abs(int(after[k]) - int(before[k])) for k in counts)
    check('"moves no reported count by more than five targets" (Sec. 6; largest shift in Table 5 runs)', bool(shift <= 5), True)

    # --- Appendix B ---
    section("App. B: Table 4 tuning grid and knockout")
    check('"three seeds each"', sorted(tuning.groupby("variant").size().unique().tolist()), [3])
    grid = (
        (1e-5, False, "flatten", 8_914_737, (0.485, 0.006), 3), (1e-5, False, "gap", 1_312_561, (0.465, 0.026), 3),
        (1e-4, False, "flatten", 8_914_737, (0.456, 0.021), 3), (1e-4, False, "gap", 1_312_561, (0.433, 0.031), 3),
        (3e-4, False, "flatten", 8_914_737, (0.413, 0.018), 3), (3e-4, False, "gap", 1_312_561, (0.404, 0.000), 3),
        (1e-5, True, "flatten", 8_916_721, (0.926, 0.003), 0), (1e-5, True, "gap", 1_314_545, (0.910, 0.001), 0),
        (1e-4, True, "flatten", 8_916_721, (0.930, 0.001), 0), (1e-4, True, "gap", 1_314_545, (0.920, 0.002), 0),
        (3e-4, True, "flatten", 8_916_721, (0.924, 0.018), 0), (3e-4, True, "gap", 1_314_545, (0.927, 0.009), 0),
    )
    for lr, batch_norm, pool, params, want_pr, want_collapsed in grid:
        sub = tuning[(tuning.lr == lr) & (tuning.batch_norm == batch_norm) & (tuning.pool == pool)]
        label = f"Table 4 lr {lr:g}, batch norm {'yes' if batch_norm else 'no'}, {pool}"
        built = count(StellarisNetwork(use_local=False, batch_norm=batch_norm, pool=pool))
        check(f"{label}: Params (model, run JSON)", [built, *sub.n_params.unique().tolist()], [params, params])
        check_ms(f"{label}: PR-AUC", sub.pr_auc, want_pr, 3)
        # Collapsed: validation-logit s.d. below 10^-3 at the selected epoch (Table 4 caption).
        check(f"{label}: Collapsed", f"{int((sub.val_logit_std < 1e-3).sum())}/{len(sub)}", f"{want_collapsed}/3")
    check('"both global interventions leave PR-AUC at 0.9451"',
          [half_up(released_ko.loc[c, "pr_auc"], 4) for c in ("global_zero", "global_shuffle")], [0.9451, 0.9451])
    check('"change no logit by more than 4.8 x 10^-7"',
          sci(released_ko.loc[["global_zero", "global_shuffle"], "max_abs_dlogit"].max(), 1), "4.8e-07")
    check('"zeroing ... the local view drops PR-AUC by 0.541"', half_up(released_ko.loc["local_zero", "pr_auc_drop"], 3), 0.541)
    ablation_ko = knockout[(knockout.group == "ablation") & (knockout.variant == "both")
                           & knockout.condition.isin(["global_zero", "global_shuffle"])]
    check('"global knockout changes no logit by more than 10^-6 in nine seeds"',
          int((ablation_ko.groupby("seed").max_abs_dlogit.max() <= 1e-6).sum()), 9)
    check_ms('"0.9434 +/- 0.0032 for local-only"', seeds("ablation", "local-only").pr_auc, (0.9434, 0.0032), 4)
    check_welch('"including the 100-epoch local-only model (+0.0119 [+0.0086, +0.0152], p = 7 x 10^-7)"',
                seeds("bn-ablation", "both").pr_auc, seeds("ablation-long", "local-only").pr_auc,
                (0.0119, 0.0086, 0.0152), 4, "7e-07", 0)
    single_view = runs[~runs.batch_norm & (runs.n_scalars == 0)
                       & (runs.variant.isin(["local-only", "global-only"]) | (runs.group == "global-tuning"))]
    check('"the both-view model also beats every single-view model without it"',
          bool(seeds("bn-ablation", "both").pr_auc.mean() > single_view.groupby(["group", "variant"]).pr_auc.mean().max()),
          True)

    # --- Appendix C ---
    section("App. C: Table 7, R1 rescored under each ephemeris (Abstract and Sec. 4 quote its first row)")
    bls_eph = released_r1
    cat_eph = released_rescore("results/camera_ready/rescore/r1_catalogue.csv")
    both_ok = bls_eph.index.intersection(cat_eph.index)
    period_right = cat_eph.loc[both_ok, "selection"].eq("bls_match").to_numpy()
    table7 = (
        ("All rescored", both_ok, 1085, (25.6, 21.2, 0.007, 3, (175, 127)), (21.6, 20.8, 0.66, 2, (129, 121))),
        ("BLS period correct", both_ok[period_right], 928,
         (20.4, 12.7, "1.5e-06", 1, (143, 72)), (21.4, 18.4, 0.05, 2, (111, 83))),
        ("BLS period wrong", both_ok[~period_right], 157,
         (56.7, 71.3, 0.02, 2, (32, 55)), (22.3, 35.0, 0.01, 2, (18, 38))),
    )
    right_period_rates = []
    for name, idx, want_n, *ephemerides in table7:
        check(f'Table 7 "{name}": n', len(idx), want_n)
        for ephemeris, frame, (clf_want, vet_want, p_want, p_digits, pairs) in zip(
                ("BLS", "catalogue"), (bls_eph, cat_eph), ephemerides):
            sub = frame.loc[idx]
            clf_fail, vet_fail = sub.prob < 0.5, ~sub.vetted
            b, c = int((clf_fail & ~vet_fail).sum()), int((~clf_fail & vet_fail).sum())
            p = _p_value(min(b, c), b + c)
            check(f'Table 7 "{name}", {ephemeris} ephemeris: classifier / vetting %',
                  [pct(clf_fail.sum(), len(sub)), pct(vet_fail.sum(), len(sub))], [clf_want, vet_want])
            check(f'Table 7 "{name}", {ephemeris} ephemeris: McNemar p',
                  sci(p, p_digits) if isinstance(p_want, str) else half_up(p, p_digits), p_want)
            check(f'App. C discordant pairs, "{name}", {ephemeris} ephemeris', [b, c], list(pairs))
            if ephemeris == "catalogue" or name == "BLS period correct":
                right_period_rates.append(float(clf_fail.mean() * 100))
    check('Sec. 4 "the classifier\'s own rate stays near 21% whenever the period is right" (each rounds to 20-22%)',
          {half_up(r) for r in right_period_rates} <= {20.0, 21.0, 22.0}, True)
    hosts = {tic: rows for tic, rows in
             load_toi_table_for_r1(CATALOGUE_DISPOSITIONS["r1"]).groupby("TIC ID")}
    reselected = 0
    for tic in both_ok:
        chosen, how = select_toi(hosts[tic], float(bls_eph.at[tic, "period_days"]))
        reselected += int(how == cat_eph.at[tic, "selection"] and float(chosen["TOI"]) == float(cat_eph.at[tic, "toi"]))
    check('"the catalogue ephemeris is the planet BLS matched, or otherwise ... highest catalogued S/N"',
          reselected, len(both_ok))

    # --- Appendix D ---
    section("App. D: injection-recovery details and Table 8")
    check('"with a fixed random seed per target and injection": one seed, (target, injection) unique',
          [inj.seed.nunique(), int(inj.duplicated(["tic_id", "injection"]).sum())], [1, 0])
    check('"from the TIC for 3,452 injections; the other 548 use solar values"',
          [int((inj.stellar_source == "table").sum()), int((inj.stellar_source == "solar").sum())], [3452, 548])
    check('"3,990 of the 4,000 lie inside the search grid"', int(inj.in_grid.sum()), 3990)
    no_fits = int((inj.stage == "no_fits").sum())
    check('"42 have no FITS file (1.1%)"', [no_fits, pct(no_fits, len(inj))], [42, 1.1])
    check('"1,942 are lost at view construction ... and 2,016 are scored"',
          [int((inj.stage == "view").sum()), int(scored.sum())], [1942, 2016])
    view = inj.stage == "view"
    check('"from 73% below expected S/N 7"', pct((view & (snr < 7)).sum(), (snr < 7).sum(), 0), 73.0)
    check('"to 17% above 50"', pct((view & (snr > 50)).sum(), (snr > 50).sum(), 0), 17.0)
    table8 = {
        "Injections": [959, 431, 291, 328, 380, 252, 330, 381, 349, 257],
        "Period found": [1.0, 2.1, 5.5, 17.1, 46.3, 60.3, 70.6, 68.2, 77.9, 81.7],
        "Recovered (0.5)": [0.0, 0.0, 0.0, 0.3, 3.7, 17.5, 32.7, 47.0, 52.1, 39.3],
        "Endorsed (0.9)": [0.0, 0.0, 0.0, 0.0, 0.3, 9.1, 18.5, 34.6, 35.2, 27.2],
    }
    bin_names = ["0-3", "3-5", "5-7", "7-10", "10-15", "15-20", "20-30", "30-50", "50-100", ">100"]
    for i, (name, row) in enumerate(zip(bin_names, by_snr.itertuples())):
        check(f'Table 8 S/N {name}: Injections', int(row.n), table8["Injections"][i])
        check(f'Table 8 S/N {name}: Period found %', pct(row.found, row.n), table8["Period found"][i])
        check(f'Table 8 S/N {name}: Recovered (0.5) %', pct(row.recovered, row.n), table8["Recovered (0.5)"][i])
        check(f'Table 8 S/N {name}: Endorsed (0.9) %', pct(row.endorsed, row.n), table8["Endorsed (0.9)"][i])
    depth_bin = pd.cut(inj.inj_depth_ppm, [500, 1000, 2000, 5000, 10_000, 20_000, 30_000], right=False)
    by_depth = inj.groupby(depth_bin, observed=False).recovered_05.agg(["sum", "size"])
    check('"By injected depth, recovery is 0%, 0.7%, 6.7%, 23.3%, 35.4% and 42.9%"',
          [pct(s, t) for s, t in zip(by_depth["sum"], by_depth["size"])], [0.0, 0.7, 6.7, 23.3, 35.4, 42.9])
    found = scored & inj.period_match
    check('"Among the 1,350 scored injections whose period BLS found"', int(found.sum()), 1350)
    clf_rates, rule_rates = [], []
    for low, high in ((0, 10), (10, 20), (20, 50), (50, 100), (100, np.inf)):
        in_bin = found & (snr >= low) & (snr < high)
        clf_rates.append(pct((inj.prob[in_bin] < 0.5).sum(), in_bin.sum(), 0))
        rule_rates.append(pct((~inj_vetted[in_bin]).sum(), in_bin.sum(), 0))
    check('"the classifier scores below 0.5 for 97%, 58%, 25%, 30% and 47%" (Sec. 5 quotes 58%)',
          clf_rates, [97.0, 58.0, 25.0, 30.0, 47.0])
    check('"and the rules reject 66%, 61%, 24%, 4% and 3%" (Sec. 5 quotes 61%)', rule_rates, [66.0, 61.0, 24.0, 4.0, 3.0])
    rule_rejected = found & (snr < 50) & ~inj_vetted
    check('"almost all through odd-even below S/N 50" (at least 95% of rule rejections)',
          bool(inj.vetting_reasons[rule_rejected].fillna("").str.contains("odd/even").mean() >= 0.95), True)
    bright = found & (snr >= 100)
    fails = inj.prob < 0.5
    t14, period, depth = inj.inj_t14_hours, inj.inj_period_days, inj.inj_depth_ppm
    share = lambda m: pct((fails & bright & m).sum(), (bright & m).sum(), 0)
    check('"54% of those with T14 < 2 h against 15% at 2 to 4 h"', [share(t14 < 2), share((t14 >= 2) & (t14 < 4))],
          [54.0, 15.0])
    check('"and 59% at P < 2 d against 24% at P >= 5 d"', [share(period < 2), share(period >= 5)], [59.0, 24.0])
    check('"(56% below 15,000 ppm, 44% above)"', [share(depth < 15_000), share(depth >= 15_000)], [56.0, 44.0])

    # --- Appendix E ---
    section("App. E: Table 9 sequential tests, Table 10 permutations")
    kepler_rows = (
        ("Kepler PR-AUC", "pr_auc", 4, (
            ("B", "A", (0.0031, -0.0016, 0.0078), 0.18, 2), ("C", "B", (0.0116, 0.0085, 0.0146), "3e-07", 0),
            ("D", "C", (0.0182, 0.0159, 0.0204), "2e-11", 0))),
        ("Kepler ROC-AUC", "roc_auc", 4, (
            ("B", "A", (0.0023, -0.0009, 0.0055), 0.14, 2), ("C", "B", (0.0100, 0.0082, 0.0117), "8e-10", 0),
            ("D", "C", (0.0116, 0.0102, 0.0129), "1e-12", 0))),
        ("Kepler R@95%P", "r95", 3, (
            ("B", "A", (0.050, 0.006, 0.093), 0.03, 2), ("C", "B", (0.038, 0.004, 0.073), 0.03, 2),
            ("D", "C", (0.120, 0.093, 0.147), "9e-07", 0))),
    )
    for metric, column, digits, comparisons in kepler_rows:
        for later, earlier, want, want_p, p_digits in comparisons:
            check_welch(f"Table 9 {metric}, {later} - {earlier}", seeds("scalars", later)[column],
                        seeds("scalars", earlier)[column], want, digits, want_p, p_digits)
    tess_rows = (
        ("R3 near 13.7 d", "r3_high_band", (
            ("B", "A", (-0.5, -4.8, 3.8), 0.81), ("C", "B", (-5.4, -9.8, -1.0), 0.02), ("D", "C", (2.0, -1.4, 5.4), 0.23))),
        ("R2 depth-cap", "r2_high_among_depth_cap", (
            ("B", "A", (1.3, -1.6, 4.2), 0.36), ("C", "B", (-1.4, -4.7, 1.9), 0.38), ("D", "C", (3.1, -0.1, 6.3), 0.06))),
    )
    for metric, column, comparisons in tess_rows:
        for later, earlier, want, want_p in comparisons:
            check_welch(f"Table 9 {metric} >= t*, {later} - {earlier}", tess[later][column], tess[earlier][column],
                        want, 1, want_p, 2)
    table10 = {
        "B": {"log_local_scale": (0.007, 0.003)},
        "C": {"log_local_scale": (0.010, 0.006), "log_period": (0.060, 0.012), "log_duration": (0.070, 0.007)},
        "D": {"log_local_scale": (0.023, 0.003), "log_period": (0.061, 0.007), "log_duration": (0.066, 0.007),
              "log_srad": (0.010, 0.002), "teff_k": (0.031, 0.002), "logg": (0.023, 0.004)},
    }
    for variant, cells in table10.items():
        check(f"Table 10 {variant}: permuted inputs", sorted(perm[perm.variant == variant].scalar.unique()), sorted(cells))
        for scalar, want in cells.items():
            check_ms(f"Table 10 {variant}, {scalar}", drop(variant, scalar), want, 3)
    for variant, want in zip("ABCD", ((0.667, 0.054), (0.716, 0.037), (0.755, 0.037), (0.875, 0.012))):
        check_ms(f'"Recall at 95% precision is ..." for {variant}', seeds("scalars", variant).r95, want, 3)
    original_full = rescored("ablation", "both")
    check_welch('"variant A differs on TESS by 0.0 [-3.3, +3.3] (p = 1.0)" (App. A "differences 0.0 ... p = 1.0")',
                tess["A"].r3_high_band, original_full.r3_high_band, (0.0, -3.3, 3.3), 1, 1.0, 1)
    check_welch('"and +0.6 [-2.4, +3.6] (p = 0.68)" (App. A "+0.6 ... p = 0.68")',
                tess["A"].r2_high_among_depth_cap, original_full.r2_high_among_depth_cap, (0.6, -2.4, 3.6), 1, 0.68, 2)
    check('"For the counterfactual in Probe 3 it is set to -3.27" (value used)',
          half_up(replacement[0], 2) if len(replacement) == 1 else float("nan"), -3.27)
    check('"whose own median is -1.24" (released rescore; matches the knockout table)',
          [half_up(real_median, 2), bool(np.allclose(swap.depth_scalar_real_median, real_median))], [-1.24, True])
    check('"the unedited scores reproduce the rescoring to within 10^-15"',
          bool(swap.max_abs_diff_vs_rescore.max() <= 1e-15), True)

    # --- Appendix F ---
    section("App. F: Kepler training negatives")
    everything = negatives.loc["all"]
    check('"The 3,983 negatives in the training data"', int(everything.n_negatives), 3983)
    for flag, name, want_n, want_pct in (("ss", "stellar eclipse", 1832, 46.0), ("co", "centroid offset", 1517, 38.1),
                                         ("nt", "not transit-like", 1204, 30.2), ("ec", "ephemeris match", 793, 19.9)):
        k = int(everything[f"n_fpflag_{flag}"])
        check(f'"{name} {want_n:,} ({want_pct}%)"', [k, pct(k, int(everything.n_negatives))], [want_n, want_pct])
    check('"1,120 carry more than one flag and 82 none"',
          [int(everything.n_multiple_flags), int(everything.n_no_flag)], [1120, 82])
    check('"train (2,779), validation (596) and test (608)"',
          [int(negatives.loc[s, "n_negatives"]) for s in ("train", "val", "test")], [2779, 596, 608])
    flag_cols = [f"frac_fpflag_{f}" for f in ("ss", "co", "nt", "ec")]
    spread = max(abs(float(negatives.loc[s, c]) - float(everything[c])) for s in ("train", "val", "test") for c in flag_cols)
    check('"The split shares are close to these" (each within 5 points)', bool(spread < 0.05), True)

    # --- Appendix G ---
    section("App. G: signal-to-noise reweighting")
    check('"The gap being explained is 49.1 points"', half_up(gap * 100, 1), 49.1)


def inspect_default(func, name: str):
    """A function's default argument value."""
    import inspect
    return inspect.signature(func).parameters[name].default


def load_toi_table_for_r1(dispositions) -> pd.DataFrame:
    """TOI rows with the R1 dispositions, as the catalogue rescoring selected from."""
    from stellaris.rescore import load_toi_table
    toi = load_toi_table()
    return toi[toi["TFOPWG Disposition"].isin(dispositions)]


if __name__ == "__main__":
    main()
