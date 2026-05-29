"""Verify stellaris.views on HAT-P-7b."""
import numpy as np
import matplotlib.pyplot as plt

from stellaris.fetch import load_kepler_lightcurve
from stellaris.preprocess import preprocess
from stellaris.views import make_global_view, make_local_view


# HAT-P-7b parameters
kepid = 10666592
period = 2.2047358        # days
epoch_bkjd = 121.3585     # BKJD
duration_hours = 4.04     # published transit duration
duration_days = duration_hours / 24.0


lcc = load_kepler_lightcurve(kepid)
clean = preprocess(lcc)

time = clean.time.value
flux = clean.flux.value

# Generate views
global_view = make_global_view(time, flux, period, epoch_bkjd)
local_view = make_local_view(time, flux, period, epoch_bkjd, duration_days)

print(f"Global view shape: {global_view.shape}  (expect (2001,))")
print(f"Local view shape:  {local_view.shape}   (expect (201,))")
print(f"Global view: min={global_view.min():.3f}, max={global_view.max():.3f}")
print(f"Local view:  min={local_view.min():.3f},  max={local_view.max():.3f}")
print(f"Both minima should be ~-1.0")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].plot(global_view, linewidth=0.5)
axes[0].set_title("Global view (full orbit, 2001 bins)")
axes[0].set_xlabel("Bin")
axes[0].set_ylabel("Normalised flux")
axes[0].axhline(0, color='red', linewidth=0.5)
axes[0].axhline(-1, color='green', linewidth=0.5, linestyle='--')

axes[1].plot(local_view, linewidth=0.8)
axes[1].set_title("Local view (transit close-up, 201 bins)")
axes[1].set_xlabel("Bin")
axes[1].set_ylabel("Normalised flux")
axes[1].axhline(0, color='red', linewidth=0.5)
axes[1].axhline(-1, color='green', linewidth=0.5, linestyle='--')

plt.suptitle(f"HAT-P-7b views: KepID={kepid}")
plt.tight_layout()
plt.savefig('hatp7b_views.png', dpi=100)
plt.show()