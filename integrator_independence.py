"""Integrator / tolerance independence of the L=0.83 stability verdict (test #4).

The linear-stability claim rests on Floquet multipliers computed with one
integrator (DOP853) at one tolerance. A skeptic's worry: the |lambda|=1 verdict
could be a precision/conditioning artifact. This script recomputes the
monodromy of the SAME orbit across:

  - integrators: DOP853, Radau (implicit), LSODA (auto stiff/non-stiff), RK45
  - tolerances : (rtol, atol) in {(1e-10,1e-12), (1e-12,1e-14), (1e-13,1e-15)}

and reports lambda_max, n_unstable, det(M), symplectic defect ||M^T J M - J||,
and the Z-space closure residual ||Z(T)-Z(0)||.

Run on TWO orbits as a control:
  - STABLE   (inside the window): verdict should be lambda_max ~ 1, n_unst = 0
  - UNSTABLE (window neighbour):  verdict should be lambda_max >> 1, n_unst > 0
If the stable verdict survives every (method, tol) AND the unstable one is
consistently flagged, the verdict is not an artifact of one integrator.

Usage: python integrator_independence.py
Outputs: integrator_independence.json
"""

import json
import numpy as np
from scipy.integrate import solve_ivp

from three_body import initial_conditions_from_params, to_Z_vector
from floquet import _eom_with_variational, newton_refine_bhh

# Representative orbits from the #2 b^3 family (continuation-refined).
STABLE = {"a": 0.246486, "c": -2.035290, "L": 0.830800, "T": 4.880107}
UNSTABLE = {"a": 0.249122, "c": -2.020368, "L": 0.828850, "T": 4.873082}

METHODS = ["DOP853", "Radau", "LSODA", "RK45"]
TOLS = [(1e-10, 1e-12), (1e-12, 1e-14), (1e-13, 1e-15)]

J = np.zeros((12, 12))
J[:6, 6:] = np.eye(6)
J[6:, :6] = -np.eye(6)


def monodromy(state0, T, method, rtol, atol):
    """Single-shot monodromy with a chosen integrator/tolerance."""
    ext0 = np.zeros(156)
    ext0[:12] = state0
    ext0[12:] = np.eye(12).ravel()
    kw = dict(method=method, rtol=rtol, atol=atol, dense_output=True)
    if method in ("DOP853", "RK45"):
        kw["max_step"] = 0.01           # match the production setting
    sol = solve_ivp(_eom_with_variational, [0.0, T], ext0, **kw)
    if not sol.success:
        raise RuntimeError(sol.message)
    final = sol.sol(T)
    return final[12:].reshape(12, 12), final[:12]


def assess(name, orbit):
    print(f"\n=== {name}: a={orbit['a']:.6f} c={orbit['c']:.6f} "
          f"L={orbit['L']:.6f} T={orbit['T']:.6f} ===")
    # Re-refine to make the period/IC as exact as possible before testing.
    a, c, T, ok, _ = newton_refine_bhh(orbit["a"], orbit["c"], orbit["L"],
                                       orbit["T"], tol=1e-12)
    if ok:
        print(f"  re-refined -> a={a:.8f} c={c:.8f} T={T:.8f}")
    else:
        a, c, T = orbit["a"], orbit["c"], orbit["T"]
        print("  re-refine did not fully converge; using catalogue values")
    state0 = initial_conditions_from_params(a, c, orbit["L"])
    Z0 = to_Z_vector(state0)

    print(f"  {'method':<8} {'rtol':>7} {'lam_max':>12} {'n_unst':>7} "
          f"{'|det-1|':>10} {'symp_def':>10} {'Zclose':>10}")
    rows = []
    for method in METHODS:
        for rtol, atol in TOLS:
            try:
                M, sT = monodromy(state0, T, method, rtol, atol)
            except Exception as e:
                print(f"  {method:<8} {rtol:>7.0e}  FAILED: {str(e)[:40]}")
                rows.append({"method": method, "rtol": rtol, "error": str(e)})
                continue
            ev = np.linalg.eigvals(M)
            lam_max = float(np.max(np.abs(ev)))
            n_unst = int(np.sum(np.abs(ev) > 1.001))
            det_err = float(abs(np.prod(ev).real - 1.0))
            symp = float(np.linalg.norm(M.T @ J @ M - J))
            zclose = float(np.linalg.norm(to_Z_vector(sT) - Z0))
            print(f"  {method:<8} {rtol:>7.0e} {lam_max:>12.7f} {n_unst:>7d} "
                  f"{det_err:>10.2e} {symp:>10.2e} {zclose:>10.2e}")
            rows.append({"method": method, "rtol": rtol, "atol": atol,
                         "lambda_max": lam_max, "n_unstable": n_unst,
                         "det_err": det_err, "symp_defect": symp,
                         "z_close": zclose})
    # Verdict spread across the good runs.
    good = [r for r in rows if "lambda_max" in r]
    if good:
        lams = [r["lambda_max"] for r in good]
        nunst = set(r["n_unstable"] for r in good)
        print(f"  --> lambda_max range [{min(lams):.7f}, {max(lams):.7f}], "
              f"n_unstable values seen: {sorted(nunst)}")
    return {"orbit": {"a": a, "c": c, "L": orbit["L"], "T": T}, "rows": rows}


def main():
    out = {"stable": assess("STABLE", STABLE),
           "unstable": assess("UNSTABLE (control)", UNSTABLE)}
    with open("integrator_independence.json", "w") as f:
        json.dump(out, f, indent=1)
    print("\nSaved: integrator_independence.json")


if __name__ == "__main__":
    main()
