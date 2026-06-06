"""
Run targeted validation diagnostics on novel-hunt CNN >= 0.9 candidates.

For each candidate, produces:
  1. Phase-folded plot (full + zoomed + binned)
  2. Secondary eclipse check searches phase ±0.5 for a dip
  3. Period alias check re-runs BLS on residuals (after masking the
     primary signal) to look for stronger signals at related periods
  4. TESS systematic period check flags candidates near known instrumental
     periods (1, 2, 13.7 days, half-orbit, etc.)
  5. Per-sector consistency check verifies the transit reproduces across
     observed sectors

Output: results/validation/<TIC>_report.png and a summary CSV with verdicts.

This is NOT planet confirmation. It's targeted false-positive checks that
the basic vetting pipeline doesn't do. Surviving these checks means a
candidate is more likely worth deeper investigation, not that it's a
confirmed planet.
"""
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.timeseries import BoxLeastSquares
import astropy.units as u

from stellaris.tess_fetch import fetch_tess_lightcurve
from stellaris.tess_preprocess import preprocess_tess


VALIDATION_DIR = Path("results/validation")
VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

# Known TESS systematic periods to flag (days)
TESS_SYSTEMATIC_PERIODS = {
    "TESS orbit half (momentum dump)": 6.85,
    "TESS orbit": 13.7,
    "Daily aliasing (1d)": 1.0,
    "Two-day aliasing": 2.0,
    "Sector boundary harmonic": 27.4,
}
# Reject if BLS period is within this fractional tolerance of a systematic
SYSTEMATIC_TOLERANCE = 0.05  # 5%


def check_secondary_eclipse(time, flux, period, epoch, duration):
    """
    Search for a dip at phase ±0.5 (secondary eclipse signature).

    Returns dict with:
      secondary_depth_ppm: estimated depth at phase 0.5 (positive = real dip)
      secondary_significance: how strong the secondary is vs noise floor
      flags_eb: True if secondary is significant enough to suggest EB
    """
    phase = ((time - epoch + 0.5 * period) % period) / period - 0.5
    
    # Look near phase ±0.5 (which is at the edges of [-0.5, 0.5])
    # Phase 0.5 wraps to ±0.5, so include both sides
    secondary_window = duration / period  # window of one transit duration
    near_secondary = (np.abs(np.abs(phase) - 0.5) < secondary_window / 2)
    
    # Out-of-transit baseline for noise estimate
    baseline_mask = (np.abs(phase) > 0.1) & (np.abs(np.abs(phase) - 0.5) > 0.1)
    
    if not near_secondary.any() or not baseline_mask.any():
        return {
            "secondary_depth_ppm": 0.0,
            "secondary_significance": 0.0,
            "flags_eb": False,
        }
    
    secondary_flux = np.median(flux[near_secondary])
    baseline_std = np.std(flux[baseline_mask])
    n_in_secondary = near_secondary.sum()
    
    # Standard error of the median
    secondary_uncertainty = baseline_std / np.sqrt(n_in_secondary)
    
    secondary_depth_ppm = -secondary_flux * 1e6  # negative flux = dip
    significance = secondary_depth_ppm / (secondary_uncertainty * 1e6) if secondary_uncertainty > 0 else 0
    
    # Flag as EB if secondary is > 3 sigma deep
    flags_eb = significance > 3
    
    return {
        "secondary_depth_ppm": float(secondary_depth_ppm),
        "secondary_significance": float(significance),
        "flags_eb": bool(flags_eb),
    }


def check_systematic_period(period):
    """Flag if BLS period is suspiciously close to TESS instrumental periods."""
    for name, sys_period in TESS_SYSTEMATIC_PERIODS.items():
        # Check both the period itself and integer multiples
        for n in [0.5, 1, 2, 3]:
            target = sys_period * n
            if abs(period - target) / target < SYSTEMATIC_TOLERANCE:
                return {
                    "flags_systematic": True,
                    "systematic_match": f"{name} (n={n}, target={target:.3f}d)",
                }
    return {"flags_systematic": False, "systematic_match": None}


def check_period_alias(time, flux, period, epoch, duration_days):
    """Mask the primary signal and re-run BLS to look for stronger signals."""
    phase = ((time - epoch + 0.5 * period) % period) / period - 0.5
    in_transit = np.abs(phase) < (duration_days / period)
    
    # Mask transits, search residuals
    masked_flux = flux.copy()
    masked_flux[in_transit] = np.nan
    
    valid = ~np.isnan(masked_flux)
    if valid.sum() < 1000:  # too few cadences for meaningful BLS
        return {"alias_found": False, "alias_period": None, "alias_ratio": 1.0}
    
    bls = BoxLeastSquares(
        time[valid] * u.day, masked_flux[valid] + 1.0
    )
    period_grid = np.exp(
        np.linspace(np.log(0.5), np.log(20.0), 20000)
    ) * u.day
    
    try:
        periodogram = bls.power(period_grid, np.array([1.0, 2.0, 4.0]) / 24.0 * u.day)
        max_residual_power = float(periodogram.power.max())
        max_residual_period = float(periodogram.period[np.argmax(periodogram.power)].value)
    except Exception:
        return {"alias_found": False, "alias_period": None, "alias_ratio": 1.0}
    
    # Original signal power (re-compute on full data for comparison)
    bls_full = BoxLeastSquares(time * u.day, flux + 1.0)
    periodogram_full = bls_full.power(
        np.array([period]) * u.day,
        np.array([duration_days]) * u.day,
    )
    primary_power = float(periodogram_full.power[0])
    
    ratio = max_residual_power / primary_power if primary_power > 0 else 1.0
    
    # If a residual signal is comparable to the primary, the original detection
    # might be an alias or there might be a second signal
    alias_found = ratio > 0.5
    
    return {
        "alias_found": bool(alias_found),
        "alias_period": max_residual_period,
        "alias_ratio": float(ratio),
    }


def check_sector_consistency(time, flux, period, epoch, duration_days):
    """Check whether the transit appears in multiple separate time windows.
    
    A real planet's transit reproduces across all observed sectors.
    A one-off signal (instrument glitch, single eclipse) only appears once.
    """
    # Detect sector boundaries from time gaps
    sorted_idx = np.argsort(time)
    time_sorted = time[sorted_idx]
    flux_sorted = flux[sorted_idx]
    
    gaps = np.diff(time_sorted)
    # Boundaries: any gap > 5 days is a sector boundary
    sector_breaks = np.where(gaps > 5)[0] + 1
    sector_starts = np.concatenate([[0], sector_breaks])
    sector_ends = np.concatenate([sector_breaks, [len(time_sorted)]])
    n_sectors = len(sector_starts)
    
    # Count sectors where at least one transit window has below-baseline flux
    phase = ((time_sorted - epoch + 0.5 * period) % period) / period - 0.5
    in_transit = np.abs(phase) < (duration_days / period)
    
    sectors_with_transit = 0
    for start, end in zip(sector_starts, sector_ends):
        sector_in_transit = in_transit[start:end]
        if sector_in_transit.any():
            sector_flux = flux_sorted[start:end][sector_in_transit]
            if np.median(sector_flux) < -1e-4:  # noticeable dip
                sectors_with_transit += 1
    
    return {
        "n_sectors_observed": int(n_sectors),
        "n_sectors_with_transit": int(sectors_with_transit),
        "transit_reproduces": bool(sectors_with_transit >= max(2, n_sectors * 0.6)),
    }


def plot_validation(tic_id, time, flux, period, epoch, duration_days,
                    depth_ppm, cnn_prob, bls_snr, diagnostics):
    """Generate a 4-panel validation figure with annotations."""
    phase = ((time - epoch + 0.5 * period) % period) / period - 0.5
    phase_window = 4 * duration_days / period
    in_window = np.abs(phase) < phase_window
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel 1: full phase fold with secondary region highlighted
    ax = axes[0, 0]
    ax.scatter(phase, flux, s=0.4, alpha=0.3, color='C0')
    ax.axvline(0, color='green', linewidth=0.8, label='Primary')
    ax.axvline(0.5, color='red', linewidth=0.8, linestyle='--', label='Secondary')
    ax.axvline(-0.5, color='red', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Phase')
    ax.set_ylabel('Normalised flux')
    ax.set_title('Full phase fold')
    ax.legend(fontsize=8)
    
    # Panel 2: zoomed transit
    ax = axes[0, 1]
    n_bins = 80
    bin_edges = np.linspace(-phase_window, phase_window, n_bins + 1)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    binned = np.array([
        np.median(flux[in_window & (phase >= bin_edges[i]) & (phase < bin_edges[i+1])])
        if (in_window & (phase >= bin_edges[i]) & (phase < bin_edges[i+1])).any()
        else np.nan
        for i in range(n_bins)
    ])
    ax.scatter(phase[in_window], flux[in_window], s=0.5, alpha=0.15, color='gray')
    ax.plot(bin_centres, binned, 'o-', color='C0', markersize=4)
    ax.axhline(0, color='red', linewidth=0.5)
    ax.set_xlabel('Phase')
    ax.set_ylabel('Normalised flux')
    ax.set_title(f'Primary transit (depth ~{depth_ppm:.0f} ppm)')
    
    # Panel 3: zoom on secondary eclipse region
    ax = axes[1, 0]
    secondary_window = phase_window
    near_secondary_mask = (np.abs(np.abs(phase) - 0.5) < secondary_window)
    if near_secondary_mask.any():
        # Show phase offset from 0.5
        secondary_phase = phase[near_secondary_mask] - np.sign(phase[near_secondary_mask]) * 0.5
        ax.scatter(secondary_phase, flux[near_secondary_mask], s=0.5, alpha=0.3, color='gray')
        ax.axhline(0, color='red', linewidth=0.5)
        sec_depth = diagnostics['secondary']['secondary_depth_ppm']
        sec_sig = diagnostics['secondary']['secondary_significance']
        ax.axhline(-sec_depth / 1e6, color='orange', linewidth=1,
                   label=f'Secondary: {sec_depth:.0f} ppm (sig: {sec_sig:.1f}σ)')
        ax.set_xlabel('Phase offset from 0.5')
        ax.set_ylabel('Normalised flux')
        ax.set_title('Secondary eclipse region')
        ax.legend(fontsize=8)
    
    # Panel 4: text summary of all diagnostics
    ax = axes[1, 1]
    ax.axis('off')
    
    sec = diagnostics['secondary']
    sys = diagnostics['systematic']
    alias = diagnostics['alias']
    sect = diagnostics['sector']
    
    verdict_color = "green"
    flags = []
    if sec['flags_eb']:
        flags.append("EB (secondary eclipse)")
        verdict_color = "red"
    if sys['flags_systematic']:
        flags.append(f"Systematic period ({sys['systematic_match']})")
        verdict_color = "red"
    if alias['alias_found']:
        flags.append(f"Strong residual signal at {alias['alias_period']:.3f}d (ratio: {alias['alias_ratio']:.2f})")
        verdict_color = "orange" if verdict_color == "green" else verdict_color
    if not sect['transit_reproduces']:
        flags.append(f"Inconsistent across sectors ({sect['n_sectors_with_transit']}/{sect['n_sectors_observed']})")
        verdict_color = "orange" if verdict_color == "green" else verdict_color
    
    verdict = "PASS, survives validation" if not flags else "FAIL, see flags"
    
    text = (
        f"TIC {tic_id}\n"
        f"{'='*40}\n\n"
        f"Inputs:\n"
        f"Period:    {period:.4f} days\n"
        f"Duration:  {duration_days*24:.2f} hours\n"
        f"Depth:     {depth_ppm:.0f} ppm\n"
        f"BLS SNR:   {bls_snr:.1f}\n"
        f"CNN prob:  {cnn_prob:.4f}\n\n"
        f"Validation diagnostics:\n"
        f"  Secondary eclipse: {sec['secondary_depth_ppm']:.0f} ppm "
        f"({sec['secondary_significance']:.1f}σ)\n"
        f"Systematic period match: {'YES' if sys['flags_systematic'] else 'NO'}\n"
        f"Residual BLS power ratio: {alias['alias_ratio']:.2f}\n"
        f"Sectors with transit: {sect['n_sectors_with_transit']}/{sect['n_sectors_observed']}\n\n"
        f"Verdict: {verdict}\n"
    )
    if flags:
        text += "\nFlags:\n" + "\n".join(f"  • {f}" for f in flags)
    
    ax.text(0.05, 0.95, text, transform=ax.transAxes,
            verticalalignment='top', family='monospace', fontsize=9,
            color=verdict_color if 'PASS' in verdict else 'black')
    
    plt.suptitle(f"Validation report: TIC {tic_id}", fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    out_path = VALIDATION_DIR / f"validation_{tic_id}.png"
    plt.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close()
    return out_path


def validate_candidate(row):
    """Run all validation checks on a single candidate. Returns a dict of results."""
    tic_id = int(row['tic_id'])
    period = float(row['period_days'])
    epoch = float(row['epoch_btjd'])
    duration_days = float(row['duration_hours']) / 24.0
    depth_ppm = float(row['depth_ppm'])
    cnn_prob = float(row['cnn_probability'])
    bls_snr = float(row['bls_snr'])
    
    print(f"\nValidating TIC {tic_id}...")
    try:
        lcc = fetch_tess_lightcurve(tic_id)
        clean = preprocess_tess(lcc)
        time = clean.time.value
        flux = clean.flux.value
    except Exception as e:
        print(f"Skipping: {e}")
        return None
    
    secondary = check_secondary_eclipse(time, flux, period, epoch, duration_days)
    systematic = check_systematic_period(period)
    alias = check_period_alias(time, flux, period, epoch, duration_days)
    sector = check_sector_consistency(time, flux, period, epoch, duration_days)
    
    diagnostics = {
        'secondary': secondary,
        'systematic': systematic,
        'alias': alias,
        'sector': sector,
    }
    
    plot_path = plot_validation(
        tic_id, time, flux, period, epoch, duration_days,
        depth_ppm, cnn_prob, bls_snr, diagnostics
    )
    
    flags_eb = secondary['flags_eb']
    flags_systematic = systematic['flags_systematic']
    flags_alias = alias['alias_found']
    flags_inconsistent = not sector['transit_reproduces']
    
    overall_pass = not (flags_eb or flags_systematic or flags_alias or flags_inconsistent)
    
    result = {
        'tic_id': tic_id,
        'period_days': period,
        'depth_ppm': depth_ppm,
        'bls_snr': bls_snr,
        'cnn_probability': cnn_prob,
        'secondary_depth_ppm': secondary['secondary_depth_ppm'],
        'secondary_significance': secondary['secondary_significance'],
        'flags_eb': flags_eb,
        'flags_systematic': flags_systematic,
        'systematic_match': systematic.get('systematic_match', ''),
        'alias_found': flags_alias,
        'alias_period': alias.get('alias_period', None),
        'alias_ratio': alias['alias_ratio'],
        'n_sectors_observed': sector['n_sectors_observed'],
        'n_sectors_with_transit': sector['n_sectors_with_transit'],
        'transit_reproduces': sector['transit_reproduces'],
        'overall_pass': overall_pass,
        'plot_path': str(plot_path),
    }
    
    status = "PASS" if overall_pass else "FLAGGED"
    print(f"{status}: secondary={secondary['secondary_significance']:.1f}σ, "
          f"systematic={'Y' if flags_systematic else 'N'}, "
          f"alias_ratio={alias['alias_ratio']:.2f}, "
          f"sectors={sector['n_sectors_with_transit']}/{sector['n_sectors_observed']}")
    
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path,
                        default=Path("results/novel_results.csv"),
                        help="Hunt results CSV")
    parser.add_argument("--threshold", type=float, default=0.9,
                        help="Minimum CNN probability to validate")
    parser.add_argument("--min-snr", type=float, default=10,
                        help="Minimum BLS SNR to validate")
    parser.add_argument("--max-candidates", type=int, default=50,
                        help="Maximum candidates to validate (top by rank_score)")
    args = parser.parse_args()
    
    df = pd.read_csv(args.results)
    print(f"Loaded {len(df)} rows from {args.results}")
    
    eligible = df[
        (df['cnn_probability'] >= args.threshold) &
        (df['vetting_passed']) &
        (df['bls_snr'] >= args.min_snr)
    ].sort_values('cnn_probability', ascending=False).head(args.max_candidates)
    
    print(f"\nValidating {len(eligible)} candidates "
          f"(CNN >= {args.threshold}, vetted, BLS SNR >= {args.min_snr})")
    
    if len(eligible) == 0:
        print("\nNo candidates meet the threshold. Nothing to validate.")
        return
    
    results = []
    for _, row in eligible.iterrows():
        result = validate_candidate(row)
        if result is not None:
            results.append(result)
    
    if not results:
        print("\nNo candidates produced validation results.")
        return
    
    results_df = pd.DataFrame(results)
    out_csv = VALIDATION_DIR / "validation_summary.csv"
    results_df.to_csv(out_csv, index=False)
    
    print(f"\n{'='*60}")
    print(f"VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Validated:          {len(results_df)}")
    print(f"Passed all checks:  {results_df['overall_pass'].sum()}")
    print(f"Flagged as EB:      {results_df['flags_eb'].sum()}")
    print(f"Flagged systematic: {results_df['flags_systematic'].sum()}")
    print(f"Flagged alias:      {results_df['alias_found'].sum()}")
    print(f"Flagged sectors:    {(~results_df['transit_reproduces']).sum()}")
    print(f"\nSummary CSV: {out_csv}")
    print(f"Plots in:    {VALIDATION_DIR}/")
    
    if results_df['overall_pass'].any():
        print(f"\nCandidates that passed all checks:")
        passers = results_df[results_df['overall_pass']]
        print(passers[['tic_id', 'period_days', 'depth_ppm', 'bls_snr', 'cnn_probability']].to_string(index=False))


if __name__ == '__main__':
    main()