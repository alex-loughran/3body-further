"""Maximum Lyapunov exponent along trajectories near the stable b^3 orbit.

The distance-based stability check saturated: kicked-stable libration,
confined-chaos wandering, and numerical secular drift all sit at similar
amplitudes (1e-3..1e-2), so the metric cannot separate them. The Lyapunov
exponent can: tangent vectors grow polynomially on a KAM torus
(lambda -> 0 as 1/t) but exponentially in a chaotic layer (lambda -> const
> 0).

Method: co-integrate the 12-component tangent vector with the orbit
(delta_r' = delta_v, delta_v' = J(r) @ delta_r with J the gravity
Jacobian), renormalise once per period, accumulate lambda = sum(ln g) / t.

Cases: the periodic orbit itself (exponent 0 — sanity), kicked copies
(the question), and the unstable + Krein family neighbours (positive
exponents — controls).

Usage:
    python lyapunov.py [--periods 1000] [--eps 1e-6]

Outputs: lyapunov.json, lyapunov.png
"""

import argparse
import json
import multiprocessing as mp
import time

import numpy as np
from scipy.integrate import solve_ivp

from three_body import three_body_eom, initial_conditions_from_params
from floquet import _gravity_jacobian
from stability_check import STABLE, UNSTABLE, KREIN


def _eom_tangent(t, y):
    """24-component RHS: orbit (12) + one tangent vector (12)."""
    state = y[:12]
    delta = y[12:]
    ds = three_body_eom(t, state)
    J = _gravity_jacobian(state[:6].reshape(3, 2))
    dd = np.empty(12)
    dd[:6] = delta[6:]
    dd[6:] = J @ delta[:6]
    return np.concatenate([ds, dd])


def lyapunov_run(args):
    """Worker: max Lyapunov exponent along one trajectory."""
    label, orbit, eps, seed, n_periods = args
    from floquet import newton_refine_bhh
    a, c, T, conv, _ = newton_refine_bhh(orbit["a"], orbit["c"], orbit["L"],
                                         orbit["T"], tol=1e-12)
    state = initial_conditions_from_params(a, c, orbit["L"])

    rng = np.random.default_rng(seed)
    if eps > 0:
        kick = rng.standard_normal(12)
        state = state + eps * kick / np.linalg.norm(kick)

    delta = rng.standard_normal(12)
    delta /= np.linalg.norm(delta)

    log_growth_sum = 0.0
    running = []          # running lambda estimate at each period
    t0 = time.time()
    for n in range(1, n_periods + 1):
        y0 = np.concatenate([state, delta])
        sol = solve_ivp(_eom_tangent, (0.0, T), y0, method="DOP853",
                        rtol=1e-11, atol=1e-13)
        if not sol.success:
            break
        state = sol.y[:12, -1]
        delta = sol.y[12:, -1]
        g = np.linalg.norm(delta)
        log_growth_sum += np.log(g)
        delta /= g
        running.append(log_growth_sum / (n * T))

    return {
        "label": label,
        "eps": eps,
        "n_periods": len(running),
        "lambda_running": running,
        "lambda_final": running[-1] if running else None,
        "wall_time_s": round(time.time() - t0, 1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--periods", type=int, default=1000)
    parser.add_argument("--eps", type=float, default=1e-6)
    args = parser.parse_args()

    cases = [
        ("stable_orbit_itself", STABLE, 0.0, 0, args.periods),
        ("stable_kick_s1", STABLE, args.eps, 1, args.periods),
        ("stable_kick_s3", STABLE, args.eps, 3, args.periods),
        ("stable_kick_100x", STABLE, 100 * args.eps, 4, args.periods),
        ("unstable_ctrl", UNSTABLE, args.eps, 5, args.periods),
        ("krein_ctrl", KREIN, args.eps, 6, args.periods),
    ]

    print(f"=== Lyapunov exponents: {args.periods} periods ===")
    with mp.Pool(len(cases)) as pool:
        results = pool.map(lyapunov_run, cases)

    # Reference: the unstable control's expected exponent from its Floquet
    # multiplier (lambda ~ 1.99 per period of T ~ 4.87 -> 0.141 / time unit)
    print(f"\n{'case':<22} {'periods':>8} {'λ_max (1/time)':>15} "
          f"{'e-fold every':>13} {'wall':>7}")
    print("-" * 72)
    for r in results:
        lam = r["lambda_final"]
        efold = f"{1/lam:.0f} t.u." if lam and lam > 1e-4 else "—"
        print(f"{r['label']:<22} {r['n_periods']:>8} {lam:>15.6f} "
              f"{efold:>13} {r['wall_time_s']:>6.0f}s")

    with open("lyapunov.json", "w") as f:
        json.dump(results, f, indent=1)
    print("\nSaved: lyapunov.json")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    for r in results:
        ax.plot(range(1, len(r["lambda_running"]) + 1), r["lambda_running"],
                lw=1.0, label=r["label"])
    ax.axhline(0.141, color="k", ls="--", lw=0.8,
               label="Floquet prediction, unstable ctrl")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("period number n")
    ax.set_ylabel("running λ_max estimate (1 / time unit)")
    ax.set_title("Running max Lyapunov exponent: torus → 0, chaos → const > 0")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("lyapunov.png", dpi=130)
    print("Plot: lyapunov.png")


if __name__ == "__main__":
    main()
