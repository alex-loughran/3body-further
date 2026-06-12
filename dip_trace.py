"""Fine-resolution stability analysis of the Jankovic #2 family dip.

The coarse continuation of the #2 (b^3) family found lambda_max dipping
to 1.021 at L = 0.830, bracketed by unit-circle crossings — a candidate
stability window. This script:

1. re-traces the family densely through the dip region with small fixed
   continuation steps,
2. runs ACCURATE Floquet analysis (auto-segmented monodromy, not the
   single-segment approximation used inside the continuation corrector)
   on every curve point, in parallel,
3. reports whether any sub-interval is linearly stable, and plots the
   non-trivial multiplier magnitudes vs L.

Usage:
    python dip_trace.py [--L-min 0.79] [--L-max 0.86] [--ds 0.004]

Outputs: dip_trace.json, dip_trace.png
"""

import argparse
import json
import multiprocessing as mp
import time

import numpy as np

from continuation import trace_family
from three_body import initial_conditions_from_params
from floquet import analyse_orbit

# Dip point from the coarse #2 family trace, re-converged with accurate
# monodromy (lambda_max = 1.0211 there).
DIP = {"a": 0.247401, "c": -2.030148, "T": 4.877870, "L": 0.830020}


def _accurate_floquet(point):
    """Worker: accurate Floquet analysis for one curve point."""
    state0 = initial_conditions_from_params(point["a"], point["c"], point["L"])
    try:
        res = analyse_orbit(state0, point["T"], verbose=False)
    except (RuntimeError, FloatingPointError, np.linalg.LinAlgError) as e:
        return {**point, "accurate": False, "error": str(e)}
    mags = sorted(float(abs(m)) for m in res["multipliers"])
    return {
        **point,
        "accurate": True,
        "multiplier_magnitudes_accurate": mags,
        "lambda_max_accurate": mags[-1],
        "is_stable": bool(res["stability"]["is_stable"]),
        "n_unstable_accurate": sum(1 for m in mags if m > 1.001),
        "monodromy_valid": bool(res["valid"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--L-min", type=float, dest="L_min", default=0.79)
    parser.add_argument("--L-max", type=float, dest="L_max", default=0.86)
    parser.add_argument("--ds", type=float, default=0.004)
    parser.add_argument("--workers", type=int, default=mp.cpu_count())
    args = parser.parse_args()

    print(f"=== Fine trace of #2 family dip, L in "
          f"[{args.L_min}, {args.L_max}], ds={args.ds} ===")
    family = trace_family(DIP["a"], DIP["c"], DIP["T"], DIP["L"],
                          L_min=args.L_min, L_max=args.L_max,
                          ds0=args.ds, ds_max=args.ds, max_steps=300)
    print(f"\n  {len(family)} curve points traced")

    print(f"  Accurate Floquet on all points ({args.workers} workers)...")
    t0 = time.time()
    with mp.Pool(args.workers) as pool:
        results = pool.map(_accurate_floquet, family)
    results = [r for r in results if r.get("accurate")]
    results.sort(key=lambda r: r["L"])
    print(f"  done in {time.time() - t0:.0f}s "
          f"({len(results)} points analysed)")

    stable = [r for r in results if r["is_stable"]]
    lam_min = min(results, key=lambda r: r["lambda_max_accurate"])

    print(f"\n=== Verdict ===")
    print(f"  lambda_max range: "
          f"{lam_min['lambda_max_accurate']:.6f} (at L={lam_min['L']:.5f}) "
          f"to {max(r['lambda_max_accurate'] for r in results):.4f}")
    if stable:
        Ls = [r["L"] for r in stable]
        print(f"  STABLE WINDOW FOUND: {len(stable)} points, "
              f"L in [{min(Ls):.5f}, {max(Ls):.5f}]")
        best = min(stable, key=lambda r: r["lambda_max_accurate"])
        print(f"  most stable point: a={best['a']:.6f} c={best['c']:.6f} "
              f"T={best['T']:.6f} L={best['L']:.6f} "
              f"λ_max={best['lambda_max_accurate']:.6f}")
    else:
        print(f"  No linearly stable point at this resolution.")
        print(f"  Closest: a={lam_min['a']:.6f} c={lam_min['c']:.6f} "
              f"T={lam_min['T']:.6f} L={lam_min['L']:.6f} "
              f"λ_max={lam_min['lambda_max_accurate']:.6f}, "
              f"n_unstable={lam_min['n_unstable_accurate']}")

    with open("dip_trace.json", "w") as f:
        json.dump(results, f, indent=1)
    print(f"  Saved: dip_trace.json")

    # Plot all non-trivial multiplier magnitudes vs L
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    L_vals = [r["L"] for r in results]
    for i in range(12):
        ax.plot(L_vals, [r["multiplier_magnitudes_accurate"][i]
                         for r in results], ".", ms=2.5)
    ax.axhline(1.0, color="k", lw=0.8)
    ax.axhline(1.001, color="k", lw=0.5, ls="--")
    ax.axhline(0.999, color="k", lw=0.5, ls="--")
    ax.set_xlabel("L")
    ax.set_ylabel("|multiplier| (accurate)")
    ax.set_title("Floquet multiplier magnitudes through the dip")

    ax = axes[1]
    ax.plot(L_vals, [r["lambda_max_accurate"] for r in results], "b.-",
            ms=3, lw=0.7)
    if stable:
        ax.plot([r["L"] for r in stable],
                [r["lambda_max_accurate"] for r in stable], "g.", ms=6,
                label="stable")
        ax.legend()
    ax.axhline(1.001, color="k", lw=0.5, ls="--")
    ax.set_xlabel("L")
    ax.set_ylabel("λ_max (accurate)")
    ax.set_title("Jankovic #2 family: stability through the dip")
    plt.tight_layout()
    plt.savefig("dip_trace.png", dpi=130)
    print(f"  Plot: dip_trace.png")


if __name__ == "__main__":
    main()
