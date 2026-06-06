"""Generate diagnostic plots for top Tier 3 candidates.

For each candidate, produces a 3-panel figure:
  - Full phase fold (every cadence mapped to one orbit)
  - Zoomed transit with phase-binned overlay
  - Local view (what the CNN actually saw)

Usage:
    python scripts/inspect_candidates.py --results results/tier3_results.csv --top 5
    python scripts/inspect_candidates.py --results results/tier3_results.csv --tics 239361732 136122328
"""
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from stellaris.tess_fetch import fetch_tess_lightcurve
from stellaris.tess_preprocess import preprocess_tess
from stellaris.views import make_local_view


PLOTS_DIR = Path("results/candidate_plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def rank_candidates(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """
    Select top candidates by joint CNN + BLS SNR score.

    Filter: CNN >= 0.95 AND vetting passed AND BLS SNR >= 20.
    Then sort by CNN probability descending.
    """
    eligible = df[
        (df['cnn_probability'] >= 0.95) &
        (df['vetting_passed']) &
        (df['bls_snr'] >= 20)
    ].copy()
    eligible = eligible.sort_values('cnn_probability', ascending=False)
    return eligible.head(top_n)


def plot_candidate(tic_id: int, period: float, epoch: float, duration_days: float, depth_ppm: float, cnn_prob: float, bls_snr: float) -> Path | None:
    """
    Generate 3-panel diagnostic plot for one candidate.
    """
    print(f"\nFetching TIC {tic_id}...")
    try:
        lcc = fetch_tess_lightcurve(tic_id)
        clean = preprocess_tess(lcc)
    except Exception as e:
        print(f"Skipping: {e}")
        return None

    time = clean.time.value
    flux = clean.flux.value

    # Phase fold
    phase = ((time - epoch + 0.5 * period) % period) / period - 0.5
    phase_window = 4 * duration_days / period
    in_window = np.abs(phase) < phase_window

    # Phase-binned values for the zoomed view
    n_bins = 80
    bin_edges = np.linspace(-phase_window, phase_window, n_bins + 1)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    binned_flux = np.array([
        np.median(flux[in_window & (phase >= bin_edges[i]) & (phase < bin_edges[i+1])])
        if ((in_window & (phase >= bin_edges[i]) & (phase < bin_edges[i+1])).any())
        else np.nan
        for i in range(n_bins)
    ])

    # Local view (what the CNN saw)
    local_view = make_local_view(time, flux, period, epoch, duration_days)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))

    # Panel 1: full phase fold
    axes[0].scatter(phase, flux, s=0.4, alpha=0.3, color='C0')
    axes[0].axhline(0, color='red', linewidth=0.5)
    axes[0].axvline(0, color='green', linewidth=0.5)
    axes[0].set_xlabel('Phase')
    axes[0].set_ylabel('Normalised flux')
    axes[0].set_title('Full phase fold')

    # Panel 2: zoomed transit
    axes[1].scatter(phase[in_window], flux[in_window], s=0.5, alpha=0.15, color='gray', label='cadences')
    axes[1].plot(bin_centres, binned_flux, 'o-', color='C0', markersize=4, label='phase-binned')
    axes[1].axhline(0, color='red', linewidth=0.5)
    axes[1].axvline(0, color='green', linewidth=0.5)
    axes[1].set_xlabel('Phase')
    axes[1].set_ylabel('Normalised flux')
    axes[1].set_title(f'Zoomed (depth ~{depth_ppm:.0f} ppm, {duration_days*24:.1f}h)')
    axes[1].legend(loc='lower right', fontsize=8)

    # Panel 3: local view (what CNN saw)
    axes[2].plot(local_view, linewidth=0.9, marker='o', markersize=2)
    axes[2].axhline(0, color='red', linewidth=0.5)
    axes[2].axhline(-1, color='green', linewidth=0.5, linestyle='--', label='depth floor')
    axes[2].set_xlabel('Bin')
    axes[2].set_ylabel('Normalised flux')
    axes[2].set_title('Local view (CNN input)')
    axes[2].legend(loc='lower right', fontsize=8)

    plt.suptitle(
        f'TIC {tic_id}: P={period:.4f}d, CNN={cnn_prob:.3f}, BLS SNR={bls_snr:.1f}',
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()

    out_path = PLOTS_DIR / f"candidate_{tic_id}.png"
    plt.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True, help="Path to hunt results CSV")
    parser.add_argument("--top", type=int, default=5, help="Number of top candidates to inspect")
    parser.add_argument("--tics", type=int, nargs="+", default=None, help="Optional explicit list of TIC IDs (overrides --top)")
    args = parser.parse_args()

    df = pd.read_csv(args.results)
    print(f"Loaded {len(df)} rows from {args.results}")

    if args.tics:
        candidates = df[df['tic_id'].isin(args.tics)].copy()
        if len(candidates) == 0:
            raise SystemExit(f"None of the specified TICs found in results.")
    else:
        candidates = rank_candidates(df, args.top)

    print(f"\nSelected {len(candidates)} candidates for inspection:")
    print(candidates[['tic_id', 'period_days', 'depth_ppm', 'bls_snr', 'cnn_probability']].to_string(index=False))

    # Convert duration from hours -> days for plot_candidate
    paths_created = []
    for _, row in candidates.iterrows():
        path = plot_candidate(
            tic_id=int(row['tic_id']),
            period=row['period_days'],
            epoch=row['epoch_btjd'],
            duration_days=row['duration_hours'] / 24.0,
            depth_ppm=row['depth_ppm'],
            cnn_prob=row['cnn_probability'],
            bls_snr=row['bls_snr'],
        )
        if path is not None:
            paths_created.append(path)

    print(f"\n{'-' * 60}")
    print(f"Done. {len(paths_created)} plots saved to {PLOTS_DIR}/")
    print(f"{'-' * 60}")


if __name__ == '__main__':
    main()