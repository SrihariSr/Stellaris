"""
Verify vetting catches TOI-2533 (EB) and lets TOI-700 c through.
"""
from stellaris.tess_fetch import fetch_tess_lightcurve
from stellaris.tess_preprocess import preprocess_tess
from stellaris.bls_search import run_bls
from stellaris.vetting import vet_candidate


CASES = [
    (150428135, "TOI-700 c", "Confirmed planet, should pass"),
    (117979897, "TOI-2533",  "Eclipsing binary, should fail"),
    (281408474, "Pi Mensae c", "Confirmed planet, should pass"),
]

for tic_id, name, expected in CASES:
    print("\n" + "-" * 60)
    print(f"TIC {tic_id}: {name}")
    print(f"Expected: {expected}")
    print("-" * 60)

    lcc = fetch_tess_lightcurve(tic_id)
    clean = preprocess_tess(lcc)
    time = clean.time.value
    flux = clean.flux.value

    bls = run_bls(time, flux, min_period=0.5, max_period=20.0)
    print(f"BLS: P={bls.period:.4f}d  D={bls.duration*24:.1f}h depth={abs(bls.depth)*1e6:.0f}ppm")

    result = vet_candidate(
        time, flux,
        period=bls.period, epoch=bls.epoch,
        duration=bls.duration, depth=bls.depth,
    )

    print(f"Vetting passed: {result.passed}")
    if result.reasons:
        for r in result.reasons:
            print(f"  REJECTED: {r}")
    print(f"Diagnostics:")
    for k, v in result.diagnostics.items():
        print(f"{k}: {v}")