"""
End-to-end pipeline: TESS data -> BLS -> CNN classification on TOI-700.
"""
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from stellaris.tess_fetch import fetch_tess_lightcurve
from stellaris.tess_preprocess import preprocess_tess
from stellaris.bls_search import run_bls
from stellaris.views import make_global_view, make_local_view
from stellaris.model import StellarisNetwork


DEVICE = (
    'mps' if torch.backends.mps.is_available()
    else 'cuda' if torch.cuda.is_available()
    else 'cpu'
)
CHECKPOINT_PATH = Path("checkpoints/stellaris_best.pt")


def main():
    print(f"Device: {DEVICE}")

    tic_id = 117979897
    print(f"\nFetch + preprocess TIC {tic_id}")
    lcc = fetch_tess_lightcurve(tic_id)
    clean = preprocess_tess(lcc)
    time = clean.time.value
    flux = clean.flux.value
    print(f"- {len(time)} cadences over {time[-1] - time[0]:.1f} days")

    print(f"\nRun BLS search")
    bls_result = run_bls(time, flux, min_period=0.5, max_period=20.0)
    print(f"- Period: {bls_result.period:.4f} days "
          f"({bls_result.duration*24:.1f}h transit, "
          f"{abs(bls_result.depth)*1e6:.0f} ppm depth)")

    print(f"\nGenerate global + local views from BLS detection")
    global_view = make_global_view(
        time, flux, bls_result.period, bls_result.epoch
    )
    local_view = make_local_view(
        time, flux, bls_result.period, bls_result.epoch,
        duration=bls_result.duration,
    )
    print(f"- Global view: {global_view.shape}, min={global_view.min():.3f}, max={global_view.max():.3f}")
    print(f"- Local view:  {local_view.shape}, min={local_view.min():.3f}, max={local_view.max():.3f}")

    print(f"\nStep 4: Load trained CNN")
    ckpt = torch.load(CHECKPOINT_PATH, weights_only=False, map_location=DEVICE)
    model = StellarisNetwork().to(DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"- Checkpoint from epoch {ckpt['epoch']}, val PR-AUC {ckpt['pr_auc']:.4f}")

    print(f"\nStep 5: CNN inference")
    global_tensor = torch.from_numpy(global_view).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
    local_tensor = torch.from_numpy(local_view).float().unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logit = model(global_tensor, local_tensor)
        probability = torch.sigmoid(logit).item()

    print(f"- Logit: {logit.item():.4f}")
    print(f"- Probability of being a planet: {probability:.4f} ({probability*100:.1f}%)")

    print("\n" + "-" * 60)
    print("END-TO-END PIPELINE RESULT")
    print("-" * 60)
    print(f"TIC {tic_id} (TOI-700):")
    print(f"BLS found a signal at period {bls_result.period:.4f} days")
    print(f"CNN classifies it as planet with probability {probability:.1%}")
    if probability > 0.9:
        print(f"- STRONG planet candidate")
    elif probability > 0.5:
        print(f"- Plausible candidate, would need further vetting")
    else:
        print(f"- CNN rejects this signal")

    # Visualise the views the model actually saw
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(global_view, linewidth=0.6)
    axes[0].axhline(0, color='red', linewidth=0.5)
    axes[0].axhline(-1, color='green', linewidth=0.5, linestyle='--')
    axes[0].set_xlabel('Bin')
    axes[0].set_ylabel('Normalised flux')
    axes[0].set_title('Global view (what the CNN saw)')

    axes[1].plot(local_view, linewidth=0.9, marker='o', markersize=2)
    axes[1].axhline(0, color='red', linewidth=0.5)
    axes[1].axhline(-1, color='green', linewidth=0.5, linestyle='--')
    axes[1].set_xlabel('Bin')
    axes[1].set_ylabel('Normalised flux')
    axes[1].set_title('Local view (what the CNN saw)')

    plt.suptitle(f'TIC {tic_id} (TOI-700 c) pipeline output, CNN p(planet) = {probability:.3f}')
    plt.tight_layout()
    plt.savefig('toi700c_pipeline_output.png', dpi=120, bbox_inches='tight')
    print("\nSaved plot to toi700c_pipeline_output.png")

if __name__ == '__main__':
    main()