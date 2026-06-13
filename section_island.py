"""KAM island test for the stable b^3 orbit: single-cluster section analysis.

The b^3 orbit pierces the z=0 section at three fixed points (period-3), so
a whole-section view is dominated by that triple structure and the
loop-vs-scatter question lives *within* each cluster at scale <= 1e-2.
This script isolates ONE cluster and sweeps the perturbation amplitude
from tiny to large to find the island boundary:

  small kick  -> section points trace a clean closed loop (invariant
                 torus, correlation dimension D ~ 1) -> nonlinearly stable
  large kick  -> points fill a 2D blob (chaotic layer, D ~ 2)

The amplitude at which loops give way to filling is the island half-width.
The large-amplitude cases double as a positive control proving the
dimension estimator actually resolves 2D (otherwise "everything is a
torus" could just be the metric flooring).

Usage:
    python section_island.py [--periods 600]

Outputs: section_island.json, section_island.png
"""

import argparse
import json
import multiprocessing as mp

import numpy as np
from scipy.integrate import solve_ivp

from three_body import three_body_eom, to_Z_vector, initial_conditions_from_params
from surface_section import _z_event
from stability_check import STABLE

# Perturbation amplitudes to sweep (kick added to the 12-state).
AMPS = [1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
# A reference fixed point of the b^3 section (from earlier runs, x≈0.4647).
FP_REF = np.array([0.4647, 0.0])     # (x, x_dot)


def _section_points(state, T, n_periods):
    sol = solve_ivp(three_body_eom, (0.0, n_periods * T), state,
                    method="DOP853", rtol=1e-11, atol=1e-13,
                    events=_z_event, max_step=T / 4)
    if not sol.t_events or not len(sol.y_events[0]):
        return np.empty((0, 6)), sol.success
    Z = np.array([to_Z_vector(ys) for ys in sol.y_events[0]])
    return Z, sol.success


def _within_cluster_dim(P):
    """Correlation dimension of a single recentred cluster (full Z minus z).

    P is one cluster's points. Distances range over 5..50th percentile,
    safely below cluster diameter, so the slope reflects loop (1) vs
    fill (2), not inter-cluster gaps.
    """
    Q = P[:, [0, 1, 3, 4, 5]]
    Q = Q - Q.mean(0)
    s = Q.std(0)
    s[s < 1e-18] = 1.0
    Q = Q / s
    n = len(Q)
    if n < 40:
        return float("nan")
    d = np.linalg.norm(Q[:, None] - Q[None], axis=2)
    d = d[np.triu_indices(n, 1)]
    d = d[d > 0]
    lo, hi = np.percentile(d, [5, 50])
    rs = np.logspace(np.log10(lo), np.log10(hi), 12)
    C = np.array([(d < r).mean() for r in rs])
    ok = C > 0
    return float(np.polyfit(np.log(rs[ok]), np.log(C[ok]), 1)[0])


def run_amp(args):
    """Worker: one perturbation amplitude. Isolates the FP_REF cluster."""
    amp, seed, n_periods = args
    from floquet import newton_refine_bhh
    a, c, T, _, _ = newton_refine_bhh(STABLE["a"], STABLE["c"], STABLE["L"],
                                      STABLE["T"], tol=1e-12)
    state = initial_conditions_from_params(a, c, STABLE["L"])
    rng = np.random.default_rng(seed)
    kick = rng.standard_normal(12)
    state = state + amp * kick / np.linalg.norm(kick)

    Z, ok = _section_points(state, T, n_periods)
    if len(Z) == 0:
        return {"amp": amp, "n": 0, "completed": ok}

    # Cluster nearest FP_REF in (x, x_dot); radius < 1/3 inter-cluster gap
    xz = Z[:, [0, 3]]
    near = np.linalg.norm(xz - FP_REF, axis=1) < 0.15
    cluster = Z[near]
    if len(cluster) < 40:
        return {"amp": amp, "n": int(len(cluster)), "completed": ok,
                "too_few": True}

    Qc = cluster[:, [0, 1, 3, 4, 5]]
    Qc = Qc - Qc.mean(0)
    R = np.linalg.norm(Qc, axis=1)
    return {
        "amp": amp,
        "n": int(len(cluster)),
        "completed": bool(ok),
        "max_radius": float(R.max()),
        "corr_dim": _within_cluster_dim(cluster),
        "cluster_points": cluster.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--periods", type=int, default=600)
    args = parser.parse_args()

    print(f"=== KAM island sweep: {len(AMPS)} amplitudes, "
          f"{args.periods} periods, single cluster ===")
    worker_args = [(amp, 100 + i, args.periods) for i, amp in enumerate(AMPS)]
    with mp.Pool(min(len(AMPS), mp.cpu_count())) as pool:
        results = pool.map(run_amp, worker_args)

    print(f"\n{'kick amp':>10} {'cluster pts':>12} {'island R':>10} "
          f"{'corr dim D':>11}  verdict")
    print("-" * 60)
    for r in results:
        if r.get("n", 0) < 40:
            print(f"{r['amp']:>10.0e} {r.get('n', 0):>12}  (cluster lost — "
                  f"likely escaped/chaotic)")
            continue
        D = r["corr_dim"]
        v = ("torus" if D < 1.35 else "CHAOS" if D > 1.65 else "transition")
        print(f"{r['amp']:>10.0e} {r['n']:>12} {r['max_radius']:>10.2e} "
              f"{D:>11.2f}  {v}")

    # strip heavy point lists before saving summary, keep them in a sidecar
    pts = {f"{r['amp']:.0e}": r.pop("cluster_points")
           for r in results if "cluster_points" in r}
    with open("section_island.json", "w") as f:
        json.dump({"summary": results, "amps": AMPS}, f, indent=1)

    _plot(results, pts, args)


def _plot(results, pts, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    have = [r for r in results if r.get("n", 0) >= 40]
    ncol = 4
    nrow = int(np.ceil(len(have) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.0 * nrow))
    for ax, r in zip(axes.flat, have):
        P = np.array(pts[f"{r['amp']:.0e}"])
        Q = P[:, [0, 1, 3, 4, 5]]
        Q = Q - Q.mean(0)
        _, _, Vt = np.linalg.svd(Q, full_matrices=False)
        u, v = Q @ Vt[0], Q @ Vt[1]
        ax.plot(u, v, ".", ms=1.8)
        ax.set_title(f"kick {r['amp']:.0e}  D={r['corr_dim']:.2f}")
        ax.set_aspect("equal", "datalim")
        ax.ticklabel_format(style="sci", scilimits=(-2, 2), axis="both")
    for ax in axes.flat[len(have):]:
        ax.axis("off")
    fig.suptitle("Stable b^3 orbit: one section cluster vs perturbation "
                 "amplitude (loop=torus, blob=chaos)")
    plt.tight_layout()
    plt.savefig("section_island.png", dpi=110)
    print("Plot: section_island.png")


if __name__ == "__main__":
    main()
