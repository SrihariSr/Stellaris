"""Run BLS on TOI-700 and see what comes back."""
import time as timer
from stellaris.tess_fetch import fetch_tess_lightcurve
from stellaris.tess_preprocess import preprocess_tess
from stellaris.bls_search import run_bls


tic_id = 150428135
print(f"Loading TIC {tic_id}...")
lcc = fetch_tess_lightcurve(tic_id)
clean = preprocess_tess(lcc)
time_arr = clean.time.value
flux_arr = clean.flux.value
print(f"Loaded {len(time_arr)} cadences over {time_arr[-1] - time_arr[0]:.1f} days")

print("\nRunning BLS search (this may take a minute)...")
t0 = timer.time()
result = run_bls(time_arr, flux_arr, min_period=0.5, max_period=20.0)
elapsed = timer.time() - t0
print(f"Done in {elapsed:.1f}s")

print("\n" + "-" * 60)
print("BLS BEST CANDIDATE")
print("-" * 60)
print(f"Period:   {result.period:.4f} days")
print(f"Epoch:    {result.epoch:.4f} BTJD")
print(f"Duration: {result.duration * 24:.2f} hours")
print(f"Depth:    {result.depth * 1e6:.0f} ppm")
print(f"Power:    {result.power:.4f}")
print(f"SNR:      {result.snr:.1f}")

# Known TOI-700 planet periods for comparison
print("\n  Known TOI-700 planets:")
print("b: 9.977 days  (sub-Earth)")
print("c: 16.051 days (sub-Neptune)")
print("d: 37.426 days (Earth-sized, habitable zone)")
print("e: 27.809 days (Earth-sized)")