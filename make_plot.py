from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# TESS orbits Earth every ~13.7 days. Momentum dumps and perigee passages recur
# on that period, which makes it the most notorious systematic in TESS
# photometry. validate_candidates.py already lists it as a known instrumental
# period, with a 5% rejection tolerance.
TESS_ORBIT = 13.7
BAND = 0.10      # shade +/- 10%
CHECK_TOL = 0.05 # the screening check's actual rejection tolerance

OUT = Path("figures")
OUT.mkdir(exist_ok=True)


def main() -> None:
    df = pd.read_csv("results/novel_results.csv")
    df["vp"] = (df["vetting_passed"].astype(str).str.strip().str.lower()
                == "true")

    periods = df["period_days"].values
    near = np.abs(periods - TESS_ORBIT) / TESS_ORBIT <= BAND
    frac = near.mean()

    fig, ax = plt.subplots(figsize=(7.0, 3.4))

    # Shaded band: +/- 10% of the TESS orbital period.
    ax.axvspan(TESS_ORBIT * (1 - BAND), TESS_ORBIT * (1 + BAND),
               color="#d62728", alpha=0.12, zorder=0,
               label=f"$\\pm${int(BAND*100)}% of 13.7 d ({frac*100:.1f}% of detections)")
    ax.axvline(TESS_ORBIT, color="#d62728", lw=1.2, ls="--", zorder=1)

    # The full distribution of recovered periods.
    ax.hist(periods, bins=np.linspace(0.5, 20, 60), color="#4c72b0",
            alpha=0.85, zorder=2, label=f"All {len(df)} faint-star detections")

    # Overlay the four the network actually likes. Every one is in the band.
    hi = df[df["cnn_probability"] >= 0.9]
    ymax = ax.get_ylim()[1]
    for _, r in hi.iterrows():
        endorsed = bool(r["vp"])
        ax.plot(r["period_days"], ymax * 0.72,
                marker="o", ms=9, zorder=4,
                mfc="none" if endorsed else "#e8a33d",
                mec="#8c2d04" if endorsed else "#8c5d04",
                mew=2.0)

    ax.plot([], [], marker="o", ms=9, ls="none", mfc="#e8a33d",
            mec="#8c5d04", mew=2.0,
            label="CNN score $\\geq 0.9$, vetting failed")
    ax.plot([], [], marker="o", ms=9, ls="none", mfc="none",
            mec="#8c2d04", mew=2.0,
            label="CNN score $\\geq 0.9$, endorsed (fails screening)")

    ax.set_xlabel("BLS period (days)")
    ax.set_ylabel("Faint-star targets")
    ax.set_xlim(0.5, 20)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT / "r3_period_histogram.pdf", bbox_inches="tight")
    fig.savefig(OUT / "r3_period_histogram.png", dpi=180, bbox_inches="tight")

    # Print the numbers that appear in the caption so they can be checked.
    print(f"n targets                       : {len(df)}")
    print(f"within {int(BAND*100)}% of 13.7 d            : {near.sum()} ({frac*100:.1f}%)")
    print(f"CNN >= 0.9                      : {len(hi)}")
    inband = (np.abs(hi['period_days'] - TESS_ORBIT) / TESS_ORBIT <= BAND).sum()
    print(f"  of which inside the band      : {inband}")
    for _, r in hi.iterrows():
        d = abs(r["period_days"] - TESS_ORBIT) / TESS_ORBIT * 100
        print(f"    TIC {int(r['tic_id']):>10}  P={r['period_days']:.3f} d  "
              f"{d:5.2f}% from 13.7 d  CNN={r['cnn_probability']:.4f}  "
              f"vetted={bool(r['vp'])}")
    print(f"\nScreening tolerance is {CHECK_TOL*100:.0f}%. Anything between "
          f"{CHECK_TOL*100:.0f}% and {BAND*100:.0f}% clears the check but is "
          f"still plausibly instrumental.")
    print(f"Wrote {OUT/'r3_period_histogram.pdf'}")


if __name__ == "__main__":
    main()