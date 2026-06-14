"""High-precision Floquet check for the stable b^3 orbit (validation #1).

Double precision can only place |lambda|_max ~ 1 to ~6-7 digits, so it cannot
rule out a weak instability |lambda| = 1 + eps with eps < ~1e-6. This recomputes
the monodromy in arbitrary precision (mpmath, default 40 digits) and reports the
Floquet multipliers to many digits.

Method (avoids re-deriving variational equations in mpmath):
  - integrate the 12-dim EOM at high precision with mpmath.odefun (a validated
    arbitrary-precision Taylor integrator),
  - build the monodromy M = d state(T)/d state(0) by finite differences:
    perturb each of the 12 initial components by delta and difference. At high
    precision this does NOT suffer cancellation -- with ~40 guard digits and
    delta ~ 1e-25, the STM is good to ~1e-13.
  - eigenvalues of M = Floquet multipliers.

RESOLUTION FLOOR: the orbit's initial condition is Newton-refined to a
periodicity residual ~1e-12 (double precision), so the multipliers are
meaningful to ~1e-12 -- 6 orders below the double-precision Floquet floor.
Going deeper would need high-precision Newton on the IC as well.

Usage: python hp_floquet.py [--dps 40] [--delta 1e-25]
Outputs: hp_floquet.json
"""

import argparse
import json
import time

import mpmath as mp
import numpy as np

import numpy as np
from three_body import initial_conditions_from_params, integrate_orbit, to_Z_vector
from floquet import analyse_orbit, newton_refine_bhh

# Seed values (low precision); refined to full double precision before use.
STABLE = {"a": 0.246486, "c": -2.035290, "L": 0.830800, "T": 4.880107}


def eom(t, y):
    """Three-body EOM (G=1, m=1) in mpmath."""
    r = [y[0:2], y[2:4], y[4:6]]
    v = [y[6:8], y[8:10], y[10:12]]
    acc = [[mp.mpf(0), mp.mpf(0)] for _ in range(3)]
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            dx = r[j][0] - r[i][0]
            dy = r[j][1] - r[i][1]
            inv_d3 = (dx * dx + dy * dy) ** mp.mpf("-1.5")
            acc[i][0] += dx * inv_d3
            acc[i][1] += dy * inv_d3
    return [v[0][0], v[0][1], v[1][0], v[1][1], v[2][0], v[2][1],
            acc[0][0], acc[0][1], acc[1][0], acc[1][1], acc[2][0], acc[2][1]]


def integrate(y0, T, tol):
    f = mp.odefun(eom, mp.mpf(0), y0, tol=tol)
    return f(T)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dps", type=int, default=40)
    parser.add_argument("--delta", type=str, default="1e-25")
    args = parser.parse_args()

    mp.mp.dps = args.dps
    delta = mp.mpf(args.delta)
    tol = mp.mpf(10) ** -(args.dps - 6)

    # Refine the orbit to FULL double precision first -- the resolution floor of
    # this whole test is set by how exactly (a, c, T) hit the true periodic
    # orbit. Using the full-precision Newton output (~1e-12 residual) instead of
    # hand-typed digits is what sets the floor at ~1e-12 rather than ~1e-8.
    a, c, T_ref, ok, info = newton_refine_bhh(
        STABLE["a"], STABLE["c"], STABLE["L"], STABLE["T"], tol=1e-13)
    y0_np = info["state"]
    res_dp = float(np.linalg.norm(
        to_Z_vector(integrate_orbit(y0_np, T_ref).sol(T_ref)) - to_Z_vector(y0_np)))
    print(f"  double-refine: a={a:.15g} c={c:.15g} T={T_ref:.15g} "
          f"converged={ok}, Z-residual={res_dp:.2e} (this is the floor)")
    T = mp.mpf(repr(float(T_ref)))
    y0 = [mp.mpf(repr(float(x))) for x in y0_np]

    print(f"=== High-precision Floquet: dps={args.dps}, delta={args.delta} ===")
    t0 = time.time()
    base = integrate(y0, T, tol)
    print(f"  base integration done ({time.time()-t0:.0f}s); "
          f"building 12 columns of M...")

    # Monodromy by one-sided finite differences.
    M = mp.zeros(12, 12)
    for k in range(12):
        yk = list(y0)
        yk[k] = yk[k] + delta
        out = integrate(yk, T, tol)
        for i in range(12):
            M[i, k] = (out[i] - base[i]) / delta
        print(f"    column {k+1}/12 done ({time.time()-t0:.0f}s)")

    # Eigenvalues = Floquet multipliers.
    evals = mp.eig(M, left=False, right=False)
    mags = sorted((abs(z) for z in evals), reverse=True)
    det = mp.det(M)
    lam_max = mags[0]

    print(f"\n  det(M)        = {mp.nstr(det, 20)}  (should be 1)")
    print(f"  |lambda|_max  = {mp.nstr(lam_max, 20)}")
    print(f"  |lambda|_max - 1 = {mp.nstr(lam_max - 1, 6)}")
    print(f"  all |multipliers| (top 6): "
          f"{[mp.nstr(m, 14) for m in mags[:6]]}")

    # Double-precision cross-check.
    dp = analyse_orbit(initial_conditions_from_params(
        STABLE["a"], STABLE["c"], STABLE["L"]), STABLE["T"], verbose=False)
    dp_max = float(max(abs(m) for m in dp["multipliers"]))
    print(f"\n  double-precision |lambda|_max = {dp_max:.7f} (cross-check)")

    n_off = sum(1 for m in mags if abs(m - 1) > mp.mpf(10) ** -10)
    verdict = ("ALL multipliers on the unit circle to <1e-10 -> linearly "
               "stable, NOT a double-precision artifact"
               if n_off == 0 else
               f"{n_off} multiplier(s) off the unit circle by >1e-10 -> "
               "weak instability resolved by high precision")
    print(f"\n  VERDICT: {verdict}")

    with open("hp_floquet.json", "w") as f:
        json.dump({"dps": args.dps, "delta": args.delta,
                   "det": mp.nstr(det, 25),
                   "lambda_max": mp.nstr(lam_max, 25),
                   "lambda_max_minus_1": mp.nstr(lam_max - 1, 10),
                   "multiplier_mags": [mp.nstr(m, 20) for m in mags],
                   "double_precision_lambda_max": dp_max,
                   "n_off_circle_1e-10": n_off}, f, indent=1)
    print("\n  Saved: hp_floquet.json")


if __name__ == "__main__":
    main()
