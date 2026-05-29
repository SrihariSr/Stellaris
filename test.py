"""Verify stellaris.preprocess on HAT-P-7 / Kepler-2 (KIC 10666592)."""
import numpy as np
import matplotlib.pyplot as plt

from stellaris.fetch import load_kepler_lightcurve
from stellaris.preprocess import preprocess


# HAT-P-7b / Kepler-2 — a bright, well-studied hot Jupiter
kepid = 10666592
period = 2.2047358    # days
epoch_bkjd = 121.3585 # BKJD
depth_ppm = 7000      # published transit depth, for sanity-checking

lcc = load_kepler_lightcurve(kepid)
clean = preprocess(lcc)

time = clean.time.value
flux = clean.flux.value

# Basic pipeline check

print("=" * 60)
print("BASIC PIPELINE STATS")
print("=" * 60)
print(f"Length: {len(clean)} cadences")
print(f"Time span: {time[-1] - time[0]:.1f} days")
print(f"Flux mean: {flux.mean():.6f}")
print(f"Global flux std: {flux.std() * 1e6:.0f} ppm")
print(f"Point-to-point std: {np.nanstd(np.diff(flux)) / np.sqrt(2) * 1e6:.0f} ppm")

# 2. Does the fold cluster the deepest points at phase 0?

print("\n" + "=" * 60)
print("PHASE FOLDING CHECK")
print("=" * 60)

deepest_idx = np.argsort(flux)[:200]
deepest_times = time[deepest_idx]
phase = ((deepest_times - epoch_bkjd + 0.5 * period) % period) / period - 0.5

print(f"Deepest 200 cadences:")
print(f"Median phase: {np.median(phase):+.4f}  (should be near 0)")
print(f"Fraction within ±0.02 of phase 0: {np.mean(np.abs(phase) < 0.02):.1%}")
print(f"Fraction within ±0.05 of phase 0: {np.mean(np.abs(phase) < 0.05):.1%}")

# Visual: 20-day time-domain window

t_center = time[len(time) // 2]
mask = (time > t_center) & (time < t_center + 20)

plt.figure(figsize=(14, 4))
plt.scatter(time[mask], flux[mask], s=2)
plt.axhline(0, color='red', linewidth=0.5)
plt.xlabel('Time (BKJD)')
plt.ylabel('Normalised flux')
plt.title(f'HAT-P-7b — 20-day window centred at BKJD {t_center:.1f}')
plt.tight_layout()
plt.savefig('hatp7b_time_domain.png', dpi=100)

# Visual: phase-folded transit

folded = clean.fold(period=period, epoch_time=epoch_bkjd)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# Wide view (full phase)
axes[0].scatter(folded.phase.value, folded.flux.value, s=1, alpha=0.2)
axes[0].set_xlabel('Phase')
axes[0].set_ylabel('Normalised flux')
axes[0].set_title('Full phase range')
axes[0].axhline(0, color='red', linewidth=0.5)
axes[0].axvline(0, color='green', linewidth=0.5)

# Zoomed view (transit)
axes[1].scatter(folded.phase.value, folded.flux.value, s=1, alpha=0.3)
axes[1].set_xlim(-0.05, 0.05)
axes[1].set_ylim(-0.010, 0.002)
axes[1].set_xlabel('Phase')
axes[1].set_ylabel('Normalised flux')
axes[1].set_title('Zoomed on transit')
axes[1].axhline(0, color='red', linewidth=0.5)
axes[1].axvline(0, color='green', linewidth=0.5)

plt.suptitle('HAT-P-7b phase-folded transit')
plt.tight_layout()
plt.savefig('hatp7b_folded.png', dpi=100)
plt.show()