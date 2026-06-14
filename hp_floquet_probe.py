"""Probe: is mpmath.odefun a viable high-precision integrator for our EOM?

Integrates the stable b^3 orbit's 12-dim equations of motion over one period at
40 digits and checks (a) it runs in reasonable time, (b) the final state agrees
with the double-precision DOP853 integrator to ~double precision.
"""

import time
import mpmath as mp
import numpy as np

from three_body import integrate_orbit, initial_conditions_from_params

STABLE = {"a": 0.24648555, "c": -2.03528985, "L": 0.830800, "T": 4.88010691}

mp.mp.dps = 40


def eom(t, y):
    """Three-body equations of motion (G=1, m=1), mpmath."""
    r = [y[0:2], y[2:4], y[4:6]]
    v = [y[6:8], y[8:10], y[10:12]]
    acc = [[mp.mpf(0), mp.mpf(0)] for _ in range(3)]
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            dx = r[j][0] - r[i][0]
            dy = r[j][1] - r[i][1]
            d2 = dx * dx + dy * dy
            inv_d3 = d2 ** mp.mpf("-1.5")
            acc[i][0] += dx * inv_d3
            acc[i][1] += dy * inv_d3
    return [v[0][0], v[0][1], v[1][0], v[1][1], v[2][0], v[2][1],
            acc[0][0], acc[0][1], acc[1][0], acc[1][1], acc[2][0], acc[2][1]]


def main():
    a, c, L, T = STABLE["a"], STABLE["c"], STABLE["L"], STABLE["T"]
    y0_np = initial_conditions_from_params(a, c, L)
    y0 = [mp.mpf(repr(float(x))) for x in y0_np]

    print(f"dps = {mp.mp.dps}, integrating 12-dim EOM over T={T} ...")
    t0 = time.time()
    f = mp.odefun(eom, mp.mpf(0), y0, tol=mp.mpf(10) ** -32)
    yT = f(mp.mpf(repr(T)))
    dt = time.time() - t0
    print(f"  odefun build+eval: {dt:.1f}s")

    # Compare to double-precision DOP853.
    sol = integrate_orbit(y0_np, T)
    yT_dp = sol.sol(T)
    diff = max(abs(float(yT[i]) - yT_dp[i]) for i in range(12))
    print(f"  max|odefun - DOP853| at T: {diff:.2e}  (expect ~1e-10..1e-12)")
    print(f"  sample yT[0] (40 digits): {mp.nstr(yT[0], 30)}")


if __name__ == "__main__":
    main()
