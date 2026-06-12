"""Pseudo-arclength continuation of BHH periodic orbits in angular momentum L.

Treats a periodic orbit family as a curve (a(s), c(s), T(s), L(s)) in the
joint parameter space and traces it numerically — no grid scanning, no
detection lottery. Each known orbit becomes a family tracer:

  - fills the gaps in the Jankovic tables (the family exists at every L
    along the curve, not just where a grid got lucky)
  - finds folds (turning points in L) where families are born/die
  - tracks Floquet multipliers along the curve; unit-circle crossings
    flag bifurcations where new families branch off

Method. The periodicity residual F(a, c, T; L) = Z(T) - Z(0) (6 components,
rotation-invariant Z-space, same as floquet.refine_newton) defines the
family as the solution curve of an underdetermined system: 6 equations,
4 unknowns x = (a, c, T, L), rank 3 at regular points. The tangent is the
right singular vector of the Jacobian with smallest singular value. Each
step: predict x + ds*t, then correct with Newton on the system augmented
by the arclength constraint t.(x - x_pred) = 0. The Jacobian reuses
floquet._monodromy_jacobian with a 3-parameter builder (a, c, L) so the
dF/dL column comes from the same single variational integration.

Components are scaled before arclength algebra (a, c, T, L have very
different magnitudes) so no single coordinate dominates step control.

Usage:
    python continuation.py trace --orbit 1 --L-max 1.1        Jankovic #1 up in L
    python continuation.py trace --orbit 1 --L-min 0.5        ...down in L
    python continuation.py trace --start A C T L0 --L-max 1.2 Custom start
    python continuation.py validate                           #1 -> #2 family test

Output: continuation_family_<name>.json (+ .png curve plot)
"""

import argparse
import json

import numpy as np

from three_body import initial_conditions_from_params, compute_energy
from floquet import _monodromy_jacobian, floquet_multipliers
from three_body import ALL_ORBITS

# Typical variation scales of (a, c, T, L) along a family — used to
# non-dimensionalise the tangent/arclength algebra.
SCALES = np.array([0.2, 1.0, 2.0, 0.2])

RESIDUAL_TOL = 1e-9
UNIT_TOL = 1e-3          # |multiplier| tolerance for unit-circle counting


def _builder3(b=1.0):
    """3-parameter state builder (a, c, L) -> state, for the Jacobian."""
    def build(p):
        return initial_conditions_from_params(p[0], p[1], p[2], b)
    return build


def _system(x, b=1.0):
    """Residual F (6,) and Jacobian G (6, 4) at x = (a, c, T, L).

    One variational integration. floquet._monodromy_jacobian returns
    columns ordered (params..., T) = (a, c, L, T); reorder to (a, c, T, L).
    """
    a, c, T, L = x
    J, F, M, final_state = _monodromy_jacobian(
        _builder3(b), np.array([a, c, L]), T, use_z_space=True)
    G = J[:, [0, 1, 3, 2]]
    return F, G, M


def _tangent(G, t_prev=None):
    """Unit tangent along the family: smallest right singular vector of the
    scaled Jacobian. Sign-aligned with the previous tangent."""
    _, _, Vt = np.linalg.svd(G * SCALES[None, :])
    t = Vt[-1]                       # scaled-space tangent, |t| = 1
    if t_prev is not None and t @ t_prev < 0:
        t = -t
    return t


def _correct(x_pred, t, ds, x_prev, b=1.0, max_iter=8):
    """Newton corrector for the augmented system:

        F(x) = 0
        t . (x_scaled - x_prev_scaled) - ds = 0

    Returns (x, F_norm, monodromy, n_iter, converged).
    """
    x = x_pred.copy()
    M = None
    for it in range(max_iter):
        F, G, M = _system(x, b)
        N = t @ ((x - x_prev) / SCALES) - ds
        res = np.linalg.norm(F)
        if res < RESIDUAL_TOL and abs(N) < 1e-10:
            return x, res, M, G, it, True
        # Stack periodicity rows (scaled columns) + arclength row
        A = np.vstack([G * SCALES[None, :], t])
        rhs = -np.concatenate([F, [N]])
        dy, _, _, _ = np.linalg.lstsq(A, rhs, rcond=None)
        x = x + dy * SCALES
        if x[0] <= 0.005 or x[2] <= 0.1:
            return x, np.inf, M, G, it, False   # left the physical domain
    F, G, M = _system(x, b)
    res = np.linalg.norm(F)
    return x, res, M, G, max_iter, res < 1e-7


def trace_family(a, c, T, L, L_min=None, L_max=None, ds0=0.02,
                 ds_max=0.08, ds_min=1e-4, max_steps=400, b=1.0,
                 verbose=True):
    """Trace a periodic orbit family from a converged starting orbit.

    Continues in both directions along the curve until L leaves
    [L_min, L_max], the orbit hits a domain boundary (a -> 0, T -> 0,
    E -> 0), or the corrector fails at the minimum step.

    Returns a list of point dicts ordered by arclength (the starting
    orbit is in the middle if both directions run).
    """
    x0 = np.array([a, c, T, L], dtype=float)

    # Converge the starting point first (fixed L: zero out the L column
    # by correcting along a tangent with no L component)
    F, G, M = _system(x0, b)
    if np.linalg.norm(F) > 1e-6:
        from floquet import newton_refine_bhh
        a_r, c_r, T_r, conv, _ = newton_refine_bhh(a, c, L, T, b=b)
        if not conv:
            raise ValueError("Starting orbit failed to converge")
        x0 = np.array([a_r, c_r, T_r, L])
        F, G, M = _system(x0, b)

    def make_point(x, M):
        mults = floquet_multipliers(M)
        mags = sorted(float(abs(m)) for m in mults)
        return {
            "a": x[0], "c": x[1], "T": x[2], "L": x[3],
            "E": float(compute_energy(
                initial_conditions_from_params(x[0], x[1], x[3], b))),
            "multiplier_magnitudes": mags,
            "lambda_max": mags[-1],
            "n_unstable": sum(1 for m in mags if m > 1 + UNIT_TOL),
        }

    start_point = make_point(x0, M)
    t0 = _tangent(G)

    def run_direction(sign):
        points = []
        x_prev = x0.copy()
        t_prev = sign * t0
        ds = ds0
        steps = fails = 0
        while steps < max_steps:
            x_pred = x_prev + ds * t_prev * SCALES
            x_new, res, M_new, G_new, n_it, ok = _correct(
                x_pred, t_prev, ds, x_prev, b)
            if not ok:
                fails += 1
                ds /= 2
                if ds < ds_min:
                    if verbose:
                        print(f"    stop: corrector failed at ds_min "
                              f"(L={x_prev[3]:.4f})")
                    break
                continue
            # Accept
            steps += 1
            a_n, c_n, T_n, L_n = x_new
            if (L_min is not None and L_n < L_min) or \
               (L_max is not None and L_n > L_max):
                if verbose:
                    print(f"    stop: L bound reached (L={L_n:.4f})")
                break
            pt = make_point(x_new, M_new)
            points.append(pt)
            if verbose and steps % 10 == 0:
                print(f"    step {steps}: L={L_n:.4f} a={a_n:.4f} "
                      f"c={c_n:.4f} T={T_n:.4f} λ_max={pt['lambda_max']:.3g} "
                      f"[ds={ds:.3g}]")
            # Tangent for next step from the corrector's final Jacobian
            t_prev = _tangent(G_new, t_prev)
            x_prev = x_new
            if n_it <= 3 and ds < ds_max:
                ds *= 1.3
        return points

    if verbose:
        print(f"  Tracing family from (a={x0[0]:.4f}, c={x0[1]:.4f}, "
              f"T={x0[2]:.4f}, L={x0[3]:.4f})")
        print("  -> direction 1")
    fwd = run_direction(+1)
    if verbose:
        print("  -> direction 2")
    bwd = run_direction(-1)

    family = list(reversed(bwd)) + [start_point] + fwd
    return family


def find_bifurcations(family):
    """Stability changes between consecutive family points.

    A change in n_unstable means a Floquet multiplier crossed the unit
    circle in that interval — a bifurcation candidate.
    """
    events = []
    for p1, p2 in zip(family, family[1:]):
        if p1["n_unstable"] != p2["n_unstable"]:
            events.append({
                "L_interval": [p1["L"], p2["L"]],
                "n_unstable": [p1["n_unstable"], p2["n_unstable"]],
            })
    return events


def find_folds(family):
    """Turning points of L along the curve (sign change of dL between steps)."""
    folds = []
    for i in range(1, len(family) - 1):
        dL1 = family[i]["L"] - family[i - 1]["L"]
        dL2 = family[i + 1]["L"] - family[i]["L"]
        if dL1 * dL2 < 0:
            folds.append({"L": family[i]["L"], "a": family[i]["a"],
                          "c": family[i]["c"], "T": family[i]["T"]})
    return folds


def plot_family(family, name, path=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    L = [p["L"] for p in family]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, key in zip(axes, ("a", "c", "T", "lambda_max")):
        ax.plot(L, [p[key] for p in family], "b.-", ms=3, lw=0.8)
        ax.set_xlabel("L")
        ax.set_ylabel(key)
        if key == "lambda_max":
            ax.set_yscale("log")
    known = [(o[1], o[2], o[3], o[4]) for o in ALL_ORBITS]
    for Lk, ak, ck, Tk in known:
        if min(L) - 0.02 <= Lk <= max(L) + 0.02:
            axes[0].plot(Lk, ak, "r*", ms=8)
            axes[1].plot(Lk, ck, "r*", ms=8)
            axes[2].plot(Lk, Tk, "r*", ms=8)
    fig.suptitle(f"Family: {name} (red stars = Jankovic catalogue)")
    plt.tight_layout()
    path = path or f"continuation_family_{name}.png"
    plt.savefig(path, dpi=130)
    print(f"  Plot: {path}")


def save_family(family, name):
    out = {
        "name": name,
        "points": family,
        "folds": find_folds(family),
        "bifurcations": find_bifurcations(family),
    }
    path = f"continuation_family_{name}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"  Saved: {path} ({len(family)} points, "
          f"{len(out['folds'])} folds, {len(out['bifurcations'])} "
          f"stability changes)")
    return path


def _jankovic(nr):
    for n, L, a, c, T, k in ALL_ORBITS:
        if n == nr:
            return a, c, T, L, k
    raise KeyError(nr)


def cmd_trace(args):
    if args.orbit is not None:
        a, c, T, L, k = _jankovic(args.orbit)
        name = f"jankovic{args.orbit}_b{k}"
    else:
        a, c, T, L = args.start
        name = args.name or f"custom_a{a:.3f}_c{c:.3f}_L{L:.3f}"
    family = trace_family(a, c, T, L, L_min=args.L_min, L_max=args.L_max,
                          ds0=args.ds, max_steps=args.max_steps)
    print(f"\n  Family traced: {len(family)} points, "
          f"L in [{min(p['L'] for p in family):.4f}, "
          f"{max(p['L'] for p in family):.4f}]")
    for fold in find_folds(family):
        print(f"  FOLD at L={fold['L']:.5f} (a={fold['a']:.4f}, "
              f"c={fold['c']:.4f})")
    for ev in find_bifurcations(family):
        print(f"  BIFURCATION in L=[{ev['L_interval'][0]:.5f}, "
              f"{ev['L_interval'][1]:.5f}]: n_unstable "
              f"{ev['n_unstable'][0]} -> {ev['n_unstable'][1]}")
    save_family(family, name)
    plot_family(family, name)


def cmd_validate(args):
    """Trace Jankovic #1 (b^3, L=0.7) and independently verify sampled
    curve points: RPF says periodic, word reader says b^3.

    Note: #1 and #2 turned out to be DISTINCT b^3 families (the #1 curve
    folds at L~0.926 and does not pass through #2; #2's own family folds
    at L~0.757 and continues above L=1). Validation therefore checks the
    method — every curve point must be a genuine b^3 orbit — rather than
    assuming catalogue entries share a family.
    """
    from three_body import (return_proximity, integrate_orbit,
                            read_free_group_word)

    a1, c1, T1, L1, k1 = _jankovic(1)
    print(f"Tracing #1 (b^{k1}, L={L1})...")
    family = trace_family(a1, c1, T1, L1, L_min=0.6, L_max=0.95, ds0=0.02)
    print(f"\n  {len(family)} points; verifying 5 samples independently:")
    idx = np.linspace(0, len(family) - 1, 5).astype(int)
    n_pass = 0
    for i in idx:
        p = family[i]
        st = initial_conditions_from_params(p["a"], p["c"], p["L"])
        d, _, _ = return_proximity(st, p["T"] * 1.2, n_samples=2000)
        sol = integrate_orbit(st, p["T"])
        word = read_free_group_word(sol, p["T"])
        ok = d < 1e-4 and word == "b" * k1
        n_pass += ok
        print(f"    [{'PASS' if ok else 'FAIL'}] L={p['L']:.4f} "
              f"a={p['a']:.4f} c={p['c']:.4f} T={p['T']:.4f} "
              f"d_min={d:.1e} word={word}")
    print(f"\n  {n_pass}/5 verified")
    save_family(family, "validate_j1")
    plot_family(family, "validate_j1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_tr = sub.add_parser("trace")
    p_tr.add_argument("--orbit", type=int, help="Jankovic orbit number")
    p_tr.add_argument("--start", type=float, nargs=4,
                      metavar=("A", "C", "T", "L"))
    p_tr.add_argument("--name")
    p_tr.add_argument("--L-min", type=float, dest="L_min", default=0.3)
    p_tr.add_argument("--L-max", type=float, dest="L_max", default=1.5)
    p_tr.add_argument("--ds", type=float, default=0.02)
    p_tr.add_argument("--max-steps", type=int, default=400)
    p_tr.set_defaults(func=cmd_trace)

    p_val = sub.add_parser("validate")
    p_val.set_defaults(func=cmd_validate)

    args = parser.parse_args()
    if args.command == "trace" and args.orbit is None and args.start is None:
        parser.error("trace needs --orbit or --start")
    args.func(args)


if __name__ == "__main__":
    main()
