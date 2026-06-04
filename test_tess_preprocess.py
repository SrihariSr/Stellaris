from stellaris.tess_fetch import fetch_tess_lightcurve
from stellaris.tess_preprocess import preprocess_tess
import matplotlib.pyplot as plt
import numpy as np

tic_id = 281408474
lcc = fetch_tess_lightcurve(tic_id)
print(f"Loaded {len(lcc)} sectors")

clean = preprocess_tess(lcc)
print(f"After preprocess: {len(clean)} cadences over {clean.time.value[-1] - clean.time.value[0]:.1f} days")
print(f"Flux mean: {clean.flux.value.mean():.6f} (should be ~0)")
print(f"Flux std (full): {clean.flux.value.std() * 1e6:.0f} ppm")
print(f"Point-to-point std: {np.nanstd(np.diff(clean.flux.value)) / np.sqrt(2) * 1e6:.0f} ppm")

# Plot a 20-day window
time = clean.time.value
flux = clean.flux.value
t_center = time[len(time) // 4]
mask = (time > t_center) & (time < t_center + 20)
plt.figure(figsize=(14, 4))
plt.scatter(time[mask], flux[mask], s=1)
plt.axhline(0, color='red', linewidth=0.5)
plt.xlabel('Time (BTJD)')
plt.ylabel('Normalised flux')
plt.title(f'TIC {tic_id} preprocessed: 20-day window')
plt.tight_layout()
plt.savefig('toi700_preprocessed.png', dpi=100)
plt.show()