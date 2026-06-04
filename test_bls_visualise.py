"""Phase-fold the BLS detection and visualize it."""
import numpy as np
import matplotlib.pyplot as plt

from stellaris.tess_fetch import fetch_tess_lightcurve
from stellaris.tess_preprocess import preprocess_tess
from stellaris.bls_search import run_bls


tic_id = 150428135
lcc = fetch_tess_lightcurve(tic_id)
clean = preprocess_tess(lcc)
time = clean.time.value
flux = clean.flux.value

# Use the period BLS found
result = run_bls(time, flux, min_period=0.5, max_period=20.0)
period = result.period
epoch = result.epoch
duration_days = result.duration

print(f"Folding on BLS-found period: {period:.4f} days, epoch: {epoch:.4f} BTJD")

# Phase fold
phase = ((time - epoch + 0.5 * period) % period) / period - 0.5

# Plot full phase range + zoom on transit
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# Full phase
axes[0].scatter(phase, flux, s=0.5, alpha=0.3)
axes[0].axhline(0, color='red', linewidth=0.5)
axes[0].axvline(0, color='green', linewidth=0.5)
axes[0].set_xlabel('Phase')
axes[0].set_ylabel('Normalised flux')
axes[0].set_title('Full phase range')

# Zoom on transit — bin the data so the signal is visible above noise
phase_window = 4 * duration_days / period   # ±4 transit durations
in_window = np.abs(phase) < phase_window
n_bins = 80
bin_edges = np.linspace(-phase_window, phase_window, n_bins + 1)
bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
binned_flux = np.array([
    np.median(flux[in_window & (phase >= bin_edges[i]) & (phase < bin_edges[i+1])])
    if ((in_window & (phase >= bin_edges[i]) & (phase < bin_edges[i+1])).any())
    else np.nan
    for i in range(n_bins)
])

axes[1].scatter(phase[in_window], flux[in_window], s=0.5, alpha=0.15, color='gray',
                label='Individual cadences')
axes[1].plot(bin_centres, binned_flux, 'o-', color='C0', markersize=4,
             label='Phase-binned')
axes[1].axhline(0, color='red', linewidth=0.5)
axes[1].axvline(0, color='green', linewidth=0.5)
axes[1].set_xlabel('Phase')
axes[1].set_ylabel('Normalised flux')
axes[1].set_title(f'Zoomed (depth ~{abs(result.depth)*1e6:.0f} ppm, '
                  f'duration ~{result.duration * 24:.1f}h)')
axes[1].legend(loc='lower right', fontsize=8)

plt.suptitle(f'TIC {tic_id}: recovered planet at P = {period:.4f} days '
             f'(TOI-700 c: 16.051 days)')
plt.tight_layout()
plt.savefig('toi700c_recovery.png', dpi=120, bbox_inches='tight')
plt.show()
print("Saved to toi700c_recovery.png")