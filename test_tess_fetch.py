"""Verify TESS fetch on TOI-700, a known multi-planet system."""
from stellaris.tess_fetch import fetch_tess_lightcurve


tic_id = 150428135  # TOI-700, host of confirmed planets including TOI-700 d
print(f"Fetching TESS light curves for TIC {tic_id} (TOI-700)...")
lcc = fetch_tess_lightcurve(tic_id)

print(f"Sectors available: {len(lcc)}")
for i, lc in enumerate(lcc):
    sector = lc.meta.get('SECTOR', '?')
    n_cadences = len(lc)
    span = lc.time.value[-1] - lc.time.value[0]
    print(f"[{i}] Sector {sector}: {n_cadences} cadences, {span:.1f} days")

# Plot one sector for sanity
import matplotlib.pyplot as plt
lc = lcc[0].remove_nans()
plt.figure(figsize=(14, 4))
plt.scatter(lc.time.value, lc.flux.value, s=1)
plt.xlabel('Time (BTJD)')
plt.ylabel('Flux (electron/s)')
plt.title(f'TIC {tic_id} sector {lcc[0].meta.get("SECTOR", "?")}')
plt.tight_layout()
plt.savefig('toi700_raw.png', dpi=100)
plt.show()