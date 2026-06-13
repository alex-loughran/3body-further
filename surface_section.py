"""Poincare surface of section: nonlinear stability of the stable b^3 orbit.

The Lyapunov/distance probes were inconclusive because they measure
exponential growth rates, which decay like ln(t)/t for regular (torus)
motion and cannot be told from "small positive" in finite time. A surface
of section sidesteps this entirely — it is geometric, not rate-based:

    on a fixed section of phase space, a trajectory on an invariant torus
    pierces the section on a CLOSED CURVE; a chaotic trajectory SCATTERS
    over a 2D area.

So if small perturbations of the stable orbit pierce the section on tidy
closed loops surrounding the orbit's fixed points, the orbit sits inside
a KAM island and is nonlinearly stable; if they smear into an area, it is
only linearly stable inside a chaotic sea.

Section: the shape-sphere equator z = 0 crossed with dz/dt > 0 (a syzygy).
At each crossing we record the rotation/translation-invariant Z-vector and
plot (x, x_dot). The orbit's own crossings are fixed points; perturbed
trajectories surround them.

Controls: the stable orbit unperturbed (fixed points), perturbations at
several amplitudes (island extent), and the unstable + Krein family
neighbours (expected to scatter / leave the island).

Usage:
    python surface_section.py [--periods 400] [--eps 1e-5]

Outputs: surface_section.json, surface_section.png
"""

import argparse
import json
import multiprocessing as mp
import time

import numpy as np
from scipy.integrate import solve_ivp

from three_body import three_body_eom, to_Z_vector, initial_conditions_from_params
from stability_check import STABLE, UNSTABLE, KREIN

SQRT2 = np.sqrt(2.0)
SQRT6 = np.sqrt(6.0)


def _z_event(t, s):
    """Proportional to the shape-sphere z coordinate (sign-identical)."""
    r = s[:6].reshape(3, 2)
    rho0 = (r[0, 0] - r[1, 0]) / SQRT2
    rho1 = (r[0, 1] - r[1, 1]) / SQRT2
    lam0 = (r[0, 0] + r[1, 0] - 2 * r[2, 0]) / SQRT6
    lam1 = (r[0, 1] + r[1, 1] - 2 * r[2, 1]) / SQRT6
    return rho0 * lam1 - rho1 * lam0


_z_event.direction = 1.0   # only z: -> + crossings (dz/dt > 0)


def section_run(args):
    """Worker: collect section crossings for one (orbit, perturbation) case.

    Returns the list of (x, x_dot) section points, plus escape info.
    """
    label, orbit, eps, seed, n_periods = args
    from floquet import newton_refine_bhh
    a, c, T, conv, _ = newton_refine_bhh(orbit["a"], orbit["c"], orbit["L"],
                                         orbit["T"], tol=1e-12)
    state = initial_conditions_from_params(a, c, orbit["L"])

    if eps > 0:
        rng = np.random.default_rng(seed)
        kick = rng.standard_normal(12)
        state = state + eps * kick / np.linalg.norm(kick)

    t_end = n_periods * T
    t0 = time.time()
    # One long integration; events are stored regardless of dense_output.
    sol = solve_ivp(three_body_eom, (0.0, t_end), state, method="DOP853",
                    rtol=1e-11, atol=1e-13, events=_z_event,
                    dense_output=False, max_step=T / 4)

    pts = []          # full Z = (x, y, z≈0, x_dot, y_dot, z_dot) per crossing
    if sol.t_events and len(sol.y_events[0]):
        for ys in sol.y_events[0]:
            pts.append([float(v) for v in to_Z_vector(ys)])

    return {
        "label": label,
        "eps": eps,
        "n_section_points": len(pts),
        "points": pts,
        "completed": bool(sol.success),
        "t_reached": float(sol.t[-1]),
        "t_target": float(t_end),
        "wall_time_s": round(time.time() - t0, 1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--periods", type=int, default=400)
    parser.add_argument("--eps", type=float, default=1e-5)
    args = parser.parse_args()

    e = args.eps
    cases = [
        ("stable_unperturbed", STABLE, 0.0, 0),
        ("stable_eps", STABLE, e, 1),
        ("stable_3eps", STABLE, 3 * e, 2),
        ("stable_10eps", STABLE, 10 * e, 3),
        ("stable_30eps", STABLE, 30 * e, 4),
        ("unstable_ctrl", UNSTABLE, e, 5),
        ("krein_ctrl", KREIN, e, 6),
    ]
    worker_args = [(lab, orb, eps, seed, args.periods)
                   for lab, orb, eps, seed in cases]

    print(f"=== Surface of section (z=0, dz/dt>0): {args.periods} periods, "
          f"eps={args.eps} ===")
    with mp.Pool(len(cases)) as pool:
        results = pool.map(section_run, worker_args)

    print(f"\n{'case':<20} {'pts':>6} {'completed':>10} {'wall':>7}")
    print("-" * 48)
    for r in results:
        comp = "yes" if r["completed"] else f"NO@{r['t_reached']:.0f}"
        print(f"{r['label']:<20} {r['n_section_points']:>6} {comp:>10} "
              f"{r['wall_time_s']:>6.0f}s")

    with open("surface_section.json", "w") as f:
        json.dump(results, f, indent=1)
    print("\nSaved: surface_section.json")

    # Correlation dimension: D≈1 -> curve (torus), D≈2 -> area (chaos).
    print(f"\n{'case':<20} {'corr. dim D':>12}  interpretation")
    print("-" * 56)
    for r in results:
        D = correlation_dimension(np.array(r["points"]))
        r["corr_dim"] = D
        interp = ("torus (1D)" if D < 1.4 else
                  "chaotic (2D)" if D > 1.7 else "mixed/marginal")
        print(f"{r['label']:<20} {D:>12.2f}  {interp}")
    with open("surface_section.json", "w") as f:
        json.dump(results, f, indent=1)

    _plot(results, args)


def correlation_dimension(pts):
    """Grassberger-Procaccia correlation dimension of a section point set.

    Centres and whitens by the per-coordinate scale, then fits
    log C(r) vs log r over the mid-range of pair distances. C(r) is the
    fraction of point pairs closer than r; its slope is the dimension.
    Projection-independent — uses the full Z-vector (minus the ~0 z column).
    """
    P = np.asarray(pts, dtype=float)
    # drop the constrained z column (index 2, ~0 on the section)
    P = P[:, [0, 1, 3, 4, 5]]
    P = P - P.mean(0)
    scale = P.std(0)
    scale[scale < 1e-15] = 1.0
    P = P / scale                      # isotropic so D reflects shape, not units
    n = len(P)
    if n < 50:
        return float("nan")
    # pairwise distances (n up to ~1200 -> ~7e5 pairs, fine)
    d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    d = d[np.triu_indices(n, k=1)]
    d = d[d > 0]
    lo, hi = np.percentile(d, [10, 60])      # mid-range, avoids saturation
    rs = np.logspace(np.log10(lo), np.log10(hi), 12)
    C = np.array([(d < r).mean() for r in rs])
    ok = C > 0
    slope = np.polyfit(np.log(rs[ok]), np.log(C[ok]), 1)[0]
    return float(slope)


def _pca_project(P):
    """Project a section cloud onto its 2 dominant PCA axes (auto-selects
    the non-degenerate plane, avoiding the near-constant-x degeneracy)."""
    P = np.asarray(P)[:, [0, 1, 3, 4, 5]]
    P0 = P - P.mean(0)
    _, _, Vt = np.linalg.svd(P0, full_matrices=False)
    return P0 @ Vt[0], P0 @ Vt[1]


def _plot(results, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # One zoomed PCA-projected panel per case: the clean torus-vs-chaos view.
    cases = ["stable_eps", "stable_3eps", "stable_10eps", "stable_30eps",
             "unstable_ctrl", "krein_ctrl"]
    by_label = {r["label"]: r for r in results}
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, lab in zip(axes.flat, cases):
        r = by_label[lab]
        u, v = _pca_project(r["points"])
        ax.plot(u, v, ".", ms=1.8)
        D = r.get("corr_dim", float("nan"))
        ax.set_title(f"{lab}  (n={len(u)}, D={D:.2f})")
        ax.set_xlabel("PCA axis 1")
        ax.set_ylabel("PCA axis 2")
        ax.set_aspect("equal", "datalim")
    fig.suptitle(f"Poincare section near the stable b^3 orbit, PCA-projected "
                 f"(L=0.8308, {args.periods} periods) — "
                 f"closed loop = torus, filled = chaos")
    plt.tight_layout()
    plt.savefig("surface_section.png", dpi=110)
    print("Plot: surface_section.png")


if __name__ == "__main__":
    main()
