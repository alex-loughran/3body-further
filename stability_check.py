"""Nonlinear (long-time) stability check of the stable b^3 orbit at L=0.8308.

Linear stability (all Floquet multipliers on the unit circle) says nothing
about survival over many periods — Arnold diffusion or resonance overlap
could still destroy the orbit slowly. This script applies the direct test:
perturb the initial state, integrate for N periods, and watch the
stroboscopic distance

    d_n = | Z(state at t = n*T) - Z(unperturbed start) |

in rotation-invariant shape space. For a nonlinearly stable orbit, d_n
stays bounded at the perturbation scale (libration on a nearby torus).
For an unstable one it grows like lambda^n until O(1) and the system
disintegrates.

Controls run alongside: an unperturbed copy (closure-error floor), an
unstable family neighbour at L=0.8285 (lambda ~ 2), and the Krein-bubble
point at L=0.8300 (lambda ~ 1.02, expect slow e-folding).

Integration: period-by-period DOP853 without dense output (memory-safe
for thousands of periods), rtol=1e-12.

Usage:
    python stability_check.py [--periods 1000] [--eps 1e-6]

Outputs: stability_check.json, stability_check.png
"""

import argparse
import json
import multiprocessing as mp
import time

import numpy as np
from scipy.integrate import solve_ivp

from three_body import (
    three_body_eom,
    to_Z_vector,
    compute_energy,
    initial_conditions_from_params,
)

# The confirmed stable orbit (window L in [0.83050, 0.83095])
STABLE = {"a": 0.246486, "c": -2.035290, "L": 0.830800, "T": 4.880107}
# Controls on the same family curve (from dip_trace.json)
UNSTABLE = {"a": 0.249122, "c": -2.020368, "L": 0.828850, "T": 4.873082}
KREIN = {"a": 0.247401, "c": -2.030148, "L": 0.830020, "T": 4.877870}

ESCAPE_D = 0.5     # stroboscopic distance treated as disintegration


def _orbit_loop(state0, T, n_loop=2000):
    """Dense sampling of the orbit's loop in Z-space (one period)."""
    sol = solve_ivp(three_body_eom, (0.0, T), state0, method="DOP853",
                    rtol=1e-12, atol=1e-14, dense_output=True)
    ts = np.linspace(0.0, T, n_loop, endpoint=False)
    return np.array([to_Z_vector(sol.sol(t)) for t in ts])


def stroboscopic_run(args):
    """Worker: one (orbit, perturbation) case.

    d_n = min distance from the stroboscopic point to the reference
    orbit's Z-loop — phase drift along the orbit doesn't register,
    only genuine departure from it.
    """
    label, orbit, eps, seed, n_periods = args
    # Re-refine to machine precision: stored parameters are truncated,
    # and a 1e-6 period error alone produces visible phase drift.
    from floquet import newton_refine_bhh
    a, c, T, conv, _ = newton_refine_bhh(orbit["a"], orbit["c"], orbit["L"],
                                         orbit["T"], tol=1e-12)
    state = initial_conditions_from_params(a, c, orbit["L"])
    loop = _orbit_loop(state, T)

    if eps > 0:
        rng = np.random.default_rng(seed)
        delta = rng.standard_normal(12)
        state = state + eps * delta / np.linalg.norm(delta)

    E0 = compute_energy(state)
    d_series = []
    t0 = time.time()
    escaped_at = None
    for n in range(1, n_periods + 1):
        sol = solve_ivp(three_body_eom, (0.0, T), state, method="DOP853",
                        rtol=1e-12, atol=1e-14)
        if not sol.success:
            escaped_at = n
            break
        state = sol.y[:, -1]
        d = float(np.min(np.linalg.norm(loop - to_Z_vector(state), axis=1)))
        d_series.append(d)
        if d > ESCAPE_D:
            escaped_at = n
            break

    return {
        "label": label,
        "eps": eps,
        "seed": seed,
        "d_series": d_series,
        "escaped_at": escaped_at,
        "d_max": max(d_series) if d_series else None,
        "d_final": d_series[-1] if d_series else None,
        "energy_drift": float(abs(compute_energy(state) - E0)),
        "wall_time_s": round(time.time() - t0, 1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--periods", type=int, default=5000)
    parser.add_argument("--eps", type=float, default=1e-6)
    args = parser.parse_args()

    cases = [
        ("stable_unperturbed", STABLE, 0.0, 0),
        ("stable_eps_seed1", STABLE, args.eps, 1),
        ("stable_eps_seed2", STABLE, args.eps, 2),
        ("stable_eps_seed3", STABLE, args.eps, 3),
        ("stable_100x_eps", STABLE, 100 * args.eps, 4),
        ("unstable_ctrl_eps", UNSTABLE, args.eps, 5),
        ("krein_ctrl_eps", KREIN, args.eps, 6),
    ]
    worker_args = [(lab, orb, eps, seed, args.periods)
                   for lab, orb, eps, seed in cases]

    print(f"=== Nonlinear stability check: {args.periods} periods, "
          f"eps={args.eps} ===")
    with mp.Pool(len(cases)) as pool:
        results = pool.map(stroboscopic_run, worker_args)

    print(f"\n{'case':<22} {'survived':>9} {'d_max':>10} {'d_final':>10} "
          f"{'ΔE':>9} {'wall':>7}")
    print("-" * 75)
    for r in results:
        n_done = len(r["d_series"])
        surv = f"{n_done}/{args.periods}" if r["escaped_at"] is None \
            else f"ESC@{r['escaped_at']}"
        print(f"{r['label']:<22} {surv:>9} {r['d_max']:>10.3e} "
              f"{r['d_final']:>10.3e} {r['energy_drift']:>9.1e} "
              f"{r['wall_time_s']:>6.0f}s")

    with open("stability_check.json", "w") as f:
        json.dump(results, f, indent=1)
    print("\nSaved: stability_check.json")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    for r in results:
        ax.plot(range(1, len(r["d_series"]) + 1), r["d_series"],
                lw=0.8, label=r["label"])
    ax.set_yscale("log")
    ax.set_xlabel("period number n")
    ax.set_ylabel("stroboscopic distance d_n in Z-space")
    ax.set_title(f"Long-time stability: stable b^3 orbit vs controls "
                 f"({args.periods} periods)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("stability_check.png", dpi=130)
    print("Plot: stability_check.png")


if __name__ == "__main__":
    main()
