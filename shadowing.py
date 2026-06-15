"""Long-time shadowing test of the stable b^3 orbit (validation #2).

Linear Floquet already bounds any linear instability rate below ~1e-13/period,
so this is a NONLINEAR test: perturb the orbit by a non-trivial amount (within
the KAM region the Poincare section found, ~1e-3) and integrate for many
periods, checking the trajectory stays trapped near the orbit's invariant loop
rather than slowly diffusing away (Arnold diffusion / resonance escape) -- a
failure mode the linearisation cannot see.

Observable: the distance, in rotation-invariant Z-space, from the perturbed
trajectory to the reference orbit's Z-loop (the closed shape-sphere curve it
traces). Sampled many times per period and tracked vs period number.

  - flat / bounded envelope            -> confined (nonlinearly stable)
  - steady upward trend or sudden jump -> diffusion / escape

Three amplitudes: two inside the KAM region (1e-4, 1e-3) and one large control
(3e-2) that the section study showed should escape -- so a "bounded" verdict is
falsifiable.

Usage: python shadowing.py [--periods 10000] [--chunk 100]
Outputs: shadowing.json, shadowing.png
"""

import argparse
import json
import time

import numpy as np

from three_body import (
    initial_conditions_from_params,
    integrate_orbit,
    to_Z_vector,
)
from floquet import newton_refine_bhh

STABLE = {"a": 0.246486, "c": -2.035290, "L": 0.830800, "T": 4.880107}

# Relaxed tolerances: over 1e4-1e5 periods we track gross bounded-vs-drift
# behaviour at the ~1e-3 scale, not machine precision.
RTOL, ATOL, MAXSTEP = 1e-11, 1e-13, 0.1


def reference_loop(y0, T, n=600):
    """Sample the reference orbit's Z-loop over one period."""
    sol = integrate_orbit(y0, T, rtol=1e-12, atol=1e-14, max_step=0.01)
    ts = np.linspace(0, T, n, endpoint=False)
    return np.array([to_Z_vector(sol.sol(t)) for t in ts])


def dist_to_loop(Z, loop):
    """Min Euclidean distance from a Z point to the sampled loop."""
    return float(np.min(np.linalg.norm(loop - Z, axis=1)))


def run_one(y0, T, loop, direction, eps, n_periods, chunk):
    """Perturb by eps along `direction`, integrate n_periods, track distance."""
    y = y0 + eps * direction
    samples_per_period = 8
    rec_p, rec_d = [], []
    state = y.copy()
    done = 0
    while done < n_periods:
        k = min(chunk, n_periods - done)
        sol = integrate_orbit(state, k * T, rtol=RTOL, atol=ATOL,
                              max_step=MAXSTEP)
        for j in range(k * samples_per_period):
            t = (j + 1) * T / samples_per_period
            d = dist_to_loop(to_Z_vector(sol.sol(t)), loop)
            rec_p.append(done + (j + 1) / samples_per_period)
            rec_d.append(d)
        state = sol.sol(k * T)
        done += k
    return np.array(rec_p), np.array(rec_d)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--periods", type=int, default=10000)
    parser.add_argument("--chunk", type=int, default=100)
    args = parser.parse_args()

    a, c, T_ref, ok, info = newton_refine_bhh(
        STABLE["a"], STABLE["c"], STABLE["L"], STABLE["T"], tol=1e-13)
    y0 = info["state"]
    T = T_ref
    print(f"refined: a={a:.10g} c={c:.10g} T={T:.10g} ok={ok}")

    loop = reference_loop(y0, T)
    print(f"reference loop: {len(loop)} samples, "
          f"Z-extent ~{np.ptp(loop, axis=0).max():.3f}")

    # Fixed, reproducible perturbation direction (unit vector).
    rng = np.random.default_rng(0)
    direction = rng.standard_normal(12)
    direction /= np.linalg.norm(direction)

    amps = [("1e-4 (KAM)", 1e-4), ("1e-3 (KAM)", 1e-3),
            ("3e-2 (control: should escape)", 3e-2)]
    results = {}
    print(f"\nIntegrating {args.periods} periods per amplitude "
          f"(chunk {args.chunk})...")
    for label, eps in amps:
        t0 = time.time()
        p, d = run_one(y0, T, loop, direction, eps, args.periods, args.chunk)
        d0 = float(d[: 8 * 10].max())    # envelope over first ~10 periods
        dmax = float(d.max())
        dfin = float(d[-8 * 50:].max())  # envelope over last ~50 periods
        growth = dfin / d0 if d0 > 0 else float("inf")
        bounded = bool(growth < 5.0 and dmax < 0.3)
        results[label] = {
            "eps": eps, "d_initial_env": d0, "d_max": dmax,
            "d_final_env": dfin, "growth_ratio": growth, "bounded": bounded,
            "periods": args.periods,
            # store a thinned time series for plotting/inspection
            "series_p": p[::max(1, len(p) // 2000)].tolist(),
            "series_d": d[::max(1, len(d) // 2000)].tolist(),
        }
        print(f"  {label:<32} eps={eps:.0e}: "
              f"d0~{d0:.2e} dmax={dmax:.2e} dfinal~{dfin:.2e} "
              f"growth={growth:.2f} -> {'BOUNDED' if bounded else 'ESCAPES'} "
              f"({time.time()-t0:.0f}s)")

    with open("shadowing.json", "w") as f:
        json.dump({"periods": args.periods, "refined": {"a": a, "c": c, "T": T},
                   "results": results}, f, indent=1)
    print("\nSaved: shadowing.json")

    # Verdict
    kam = [results[k] for k in results if "KAM" in k]
    ctrl = [results[k] for k in results if "control" in k]
    kam_ok = all(r["bounded"] for r in kam)
    ctrl_escapes = ctrl and not ctrl[0]["bounded"]
    print("\n=== Verdict ===")
    if kam_ok and ctrl_escapes:
        print("  KAM-region perturbations stay BOUNDED over "
              f"{args.periods} periods; the large control escapes (so the test "
              "can detect escape). -> nonlinear confinement confirmed.")
    elif kam_ok:
        print(f"  KAM-region perturbations bounded over {args.periods} periods. "
              "(Control did not clearly escape -- consider a larger control or "
              "more periods.)")
    else:
        print("  A KAM-region perturbation drifted -- inspect shadowing.png.")

    _plot(results)


def _plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    for label, r in results.items():
        plt.semilogy(r["series_p"], r["series_d"], lw=0.6, label=label)
    plt.xlabel("period number")
    plt.ylabel("distance to reference Z-loop")
    plt.title("Long-time shadowing of the stable b^3 orbit")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("shadowing.png", dpi=130)
    print("Plot: shadowing.png")


if __name__ == "__main__":
    main()
