"""
Regenerate and check every number that appears in the paper.
"""
from __future__ import annotations

import collections
import json
import re
import sys
import warnings
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
