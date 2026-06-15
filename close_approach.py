"""Does instability correlate with closest approach?  (physical-insight test)

Hypothesis: in the three-body problem, linear instability is generated mainly
at close encounters (two bodies passing near each other amplify perturbations).
If so, the Floquet lambda_max of a periodic orbit should be controlled by the
MINIMUM inter-body separation r_min reached over one period -- and angular
momentum L would enter only indirectly, by setting a centrifugal floor on how
small r_min can get.

This computes r_min for all 110 catalogued orbits (75 BHH + 35 symmetric, with
lambda_max already in floquet_catalogue.json) and tests:
  - log10(lambda_max) vs r_min        (the core hypothesis)
  - r_min vs L                        (centrifugal floor)
  - lambda_max vs L                   (the documented trend -- is it via r_min?)

Reports Spearman (monotonic) and Pearson correlations, and overlays the
discovered stable b^3 orbit at L=0.8308.

Usage: python close_approach.py
Outputs: close_approach.json, close_approach.png
"""

import json

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr, pearsonr

from three_body import (
    initial_conditions_from_params,
    build_state_symmetric,
    integrate_orbit,
)

PAIRS = [(0, 1), (0, 2), (1, 2)]

# The discovered stable orbit (not a catalogue entry; lambda_max ~ 1).
STABLE = {"name": "discovered stable b^3", "parametrisation": "bhh",
          "L": 0.8308, "params": [0.246486, -2.035290], "T": 4.880107,
          "word": "b^3", "max_instability": 1.0000000000005, "is_stable": True}


def build_state(entry):
    p = entry["params"]
    if entry["parametrisation"] == "bhh":
        return initial_conditions_from_params(p[0], p[1], entry["L"])
    return build_state_symmetric(p[0], p[1])


def min_separation(state0, T, n=8000):
    """Closest approach (min inter-body distance) over one period."""
    sol = integrate_orbit(state0, T, rtol=1e-11, atol=1e-13, max_step=0.02)
    ts = np.linspace(0, T, n)

    def sep_at(t):
        s = sol.sol(t)
        r = s[:6].reshape(3, 2)
        return min(np.hypot(*(r[i] - r[j])) for i, j in PAIRS)

    d = np.array([sep_at(t) for t in ts])
    k = int(np.argmin(d))
    # refine around the discrete minimum
    lo, hi = ts[max(0, k - 1)], ts[min(n - 1, k + 1)]
    res = minimize_scalar(sep_at, bounds=(lo, hi), method="bounded")
    return float(min(d[k], res.fun))


def main():
    catalogue = json.load(open("floquet_catalogue.json"))
    rows = []
    for entry in catalogue + [STABLE]:
        try:
            rmin = min_separation(build_state(entry), entry["T"])
        except Exception as e:
            print(f"  skip {entry.get('name')}: {e}")
            continue
        rows.append({
            "name": entry["name"], "param": entry["parametrisation"],
            "L": entry["L"], "T": entry["T"],
            "lambda_max": entry["max_instability"],
            "r_min": rmin, "is_stable": entry.get("is_stable", False),
            "discovered": entry["name"].startswith("discovered"),
        })
    print(f"computed r_min for {len(rows)} orbits")

    lam = np.array([r["lambda_max"] for r in rows])
    rmin = np.array([r["r_min"] for r in rows])
    L = np.array([r["L"] for r in rows])
    loglam = np.log10(lam)
    is_bhh = np.array([r["param"] == "bhh" for r in rows])

    def corr(x, y, label):
        sp = spearmanr(x, y).correlation
        pe = pearsonr(x, y)[0]
        print(f"  {label:<42} Spearman={sp:+.3f}  Pearson={pe:+.3f}")
        return {"spearman": float(sp), "pearson": float(pe)}

    print("\n=== Correlations (all orbits) ===")
    c_all = {
        "loglam_vs_rmin": corr(loglam, rmin, "log10(lambda_max) vs r_min"),
        "loglam_vs_logrmin": corr(loglam, np.log10(rmin),
                                  "log10(lambda_max) vs log10(r_min)"),
        "rmin_vs_L": corr(rmin, L, "r_min vs L"),
        "loglam_vs_L": corr(loglam, L, "log10(lambda_max) vs L"),
    }
    print("\n=== BHH (L != 0) only ===")
    c_bhh = {
        "loglam_vs_rmin": corr(loglam[is_bhh], rmin[is_bhh],
                               "log10(lambda_max) vs r_min"),
        "rmin_vs_L": corr(rmin[is_bhh], L[is_bhh], "r_min vs L"),
        "loglam_vs_L": corr(loglam[is_bhh], L[is_bhh],
                            "log10(lambda_max) vs L"),
    }

    # Partial check: does r_min explain lambda_max BEYOND what L does?
    # Compare |corr(loglam, rmin)| vs |corr(loglam, L)| on BHH set.
    print("\n=== Interpretation hint ===")
    cl_rmin = abs(c_bhh["loglam_vs_rmin"]["spearman"])
    cl_L = abs(c_bhh["loglam_vs_L"]["spearman"])
    print(f"  BHH: |corr(logλ, r_min)|={cl_rmin:.3f} vs "
          f"|corr(logλ, L)|={cl_L:.3f} -> "
          f"{'r_min is the stronger predictor' if cl_rmin > cl_L else 'L is the stronger predictor'}")

    sb = next(r for r in rows if r["discovered"])
    print(f"\n  Discovered stable b^3: r_min={sb['r_min']:.4f}, "
          f"L={sb['L']}, lambda_max~1. "
          f"(r_min percentile among BHH: "
          f"{100*np.mean(rmin[is_bhh] < sb['r_min']):.0f}%)")

    with open("close_approach.json", "w") as f:
        json.dump({"rows": rows, "corr_all": c_all, "corr_bhh": c_bhh}, f,
                  indent=1)
    print("\nSaved: close_approach.json")
    _plot(rows)


def _plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lam = np.array([r["lambda_max"] for r in rows])
    rmin = np.array([r["r_min"] for r in rows])
    L = np.array([r["L"] for r in rows])
    disc = np.array([r["discovered"] for r in rows])

    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    sc = ax[0].scatter(rmin[~disc], lam[~disc], c=L[~disc], cmap="viridis",
                       s=25)
    ax[0].scatter(rmin[disc], lam[disc], marker="*", s=300, c="red",
                  edgecolor="k", label="discovered stable b^3", zorder=5)
    ax[0].set_yscale("log"); ax[0].set_xlabel("r_min (closest approach)")
    ax[0].set_ylabel("lambda_max"); ax[0].legend(fontsize=8)
    ax[0].set_title("Instability vs closest approach (color = L)")
    plt.colorbar(sc, ax=ax[0], label="L")

    ax[1].scatter(L[~disc], rmin[~disc], s=25, c="steelblue")
    ax[1].scatter(L[disc], rmin[disc], marker="*", s=300, c="red",
                  edgecolor="k", zorder=5)
    ax[1].set_xlabel("L"); ax[1].set_ylabel("r_min")
    ax[1].set_title("Closest approach vs angular momentum")

    ax[2].scatter(L[~disc], lam[~disc], s=25, c="darkorange")
    ax[2].scatter(L[disc], lam[disc], marker="*", s=300, c="red",
                  edgecolor="k", zorder=5)
    ax[2].set_yscale("log"); ax[2].set_xlabel("L")
    ax[2].set_ylabel("lambda_max")
    ax[2].set_title("Instability vs L (for comparison)")
    plt.tight_layout()
    plt.savefig("close_approach.png", dpi=130)
    print("Plot: close_approach.png")


if __name__ == "__main__":
    main()
