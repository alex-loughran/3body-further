"""Trace the period-doubled branch off the L=0.83 stable b^3 window (task #10, Part B).

At L_PD = 0.831064 a real Floquet multiplier of the Jankovic #2 (b^3) family
crosses -1 -> a branch of period-~2T orbits is born. These doubled orbits leave
the 3-parameter BHH manifold, so the (a,c,L) continuation cannot represent them.
This implements a FULL-PHASE-SPACE shooting Newton and continues the branch.

The doubled branch bifurcates ABOVE the stable window (the window's upper edge
is a Krein collision at ~0.83097), so the base orbit is already unstable there
and the doubled orbits are born unstable too -- this completes the bifurcation
diagram, it does not produce a new stable orbit.

Full-state periodic-orbit conditions (unknowns: state x0 in R^12, period P):
  residual F (9 components):
    [0:6] Z(P; x0) - Z(0; x0)              rotation-invariant periodicity
    [6]   L(x0) - L_target                 fix angular momentum (continuation param)
    [7]   <x0 - x_ref, f(x_ref)>           phase: remove time-translation gauge
    [8]   <x0 - x_ref, g(x_ref)>           remove rotation gauge (Z is rot-invariant)
  COM and total momentum are fixed to zero by projection each iteration.
Jacobian uses the monodromy (1 variational integration/iter). Solved by
least squares; min-norm step naturally handles the remaining gauge directions.

Usage: python period_double_branch.py
Outputs: period_double_branch.json
"""

import json
import numpy as np

from three_body import (
    _eom_core, to_Z_vector, integrate_orbit,
    initial_conditions_from_params, compute_angular_momentum, compute_energy,
)
from floquet import (
    compute_monodromy, _z_jacobian, newton_refine_bhh, floquet_multipliers,
)

L_PD = 0.831064
T_BASE = 4.8779   # period near L_PD (refined below)
SEED = {"a": 0.246486, "c": -2.035290}


def rot_gen(s):
    """Infinitesimal-rotation generator g(s) (d state / d rotation angle)."""
    return np.array([-s[1], s[0], -s[3], s[2], -s[5], s[4],
                     -s[7], s[6], -s[9], s[8], -s[11], s[10]])


def dL_dx(s):
    """Gradient of total angular momentum wrt the 12-state."""
    g = np.zeros(12)
    for i in range(3):
        x, y = s[2 * i], s[2 * i + 1]
        vx, vy = s[6 + 2 * i], s[6 + 2 * i + 1]
        g[2 * i] = vy
        g[2 * i + 1] = -vx
        g[6 + 2 * i] = -y
        g[6 + 2 * i + 1] = x
    return g


def project(s):
    """Set centre of mass and total momentum to zero (unit masses)."""
    s = s.copy()
    r = s[:6].reshape(3, 2)
    v = s[6:].reshape(3, 2)
    r -= r.mean(axis=0)
    v -= v.mean(axis=0)
    s[:6] = r.ravel()
    s[6:] = v.ravel()
    return s


def residual(x0, P, L_target, x_ref, f_ref, g_ref):
    M_unused = None
    sol = integrate_orbit(x0, P)
    xP = sol.sol(P)
    Zc = to_Z_vector(xP) - to_Z_vector(x0)
    F = np.empty(9)
    F[:6] = Zc
    F[6] = compute_angular_momentum(x0) - L_target
    F[7] = (x0 - x_ref) @ f_ref
    F[8] = (x0 - x_ref) @ g_ref
    return F


def jacobian(x0, P, f_ref, g_ref):
    """9x13 Jacobian (cols 0:12 = d/dx0, col 12 = d/dP) via monodromy."""
    M, xP = compute_monodromy(x0, P, n_segments=1)
    dZ_P = _z_jacobian(xP)
    dZ_0 = _z_jacobian(x0)
    J = np.zeros((9, 13))
    J[:6, :12] = dZ_P @ M - dZ_0
    J[:6, 12] = dZ_P @ _eom_core(xP)
    J[6, :12] = dL_dx(x0)
    J[7, :12] = f_ref
    J[8, :12] = g_ref
    return J, M, xP


def solve(x0_guess, P_guess, L_target, max_iter=40, tol=1e-10, verbose=False):
    x0 = project(np.array(x0_guess, float))
    P = float(P_guess)
    x_ref = x0.copy()
    f_ref = _eom_core(x_ref)
    g_ref = rot_gen(x_ref)
    last = None
    for it in range(max_iter):
        try:                                  # a guess may fly into a collision
            F = residual(x0, P, L_target, x_ref, f_ref, g_ref)
        except (RuntimeError, FloatingPointError):
            return x0, P, False, np.inf
        nrm = np.linalg.norm(F)
        if verbose:
            print(f"    iter {it}: |F|={nrm:.3e}")
        if nrm < tol:
            return x0, P, True, nrm
        try:
            J, M, xP = jacobian(x0, P, f_ref, g_ref)
            delta, *_ = np.linalg.lstsq(J, -F, rcond=None)
        except (np.linalg.LinAlgError, RuntimeError, FloatingPointError):
            return x0, P, False, nrm
        # damping if step is huge
        step = delta
        if np.linalg.norm(step) > 1.0:
            step = step / np.linalg.norm(step)
        x0 = project(x0 + step[:12])
        P = P + step[12]
        last = nrm
        if not np.isfinite(P) or P <= 0:
            return x0, P, False, nrm
    return x0, P, False, last if last is not None else np.inf


def lam_max(x0, P):
    try:
        M, _ = compute_monodromy(x0, P)
        return float(max(abs(m) for m in floquet_multipliers(M)))
    except (RuntimeError, FloatingPointError, np.linalg.LinAlgError):
        return float("nan")


def main():
    out = {}

    # Base orbit near L_PD.
    a, c, T, ok, _ = newton_refine_bhh(SEED["a"], SEED["c"], L_PD, T_BASE,
                                       tol=1e-12)
    x0_base = initial_conditions_from_params(a, c, L_PD)
    L0 = compute_angular_momentum(x0_base)
    print(f"base @ L_PD: a={a:.8f} c={c:.8f} T={T:.8f} L={L0:.6f} ok={ok}")

    # --- VALIDATION: full-state Newton must re-find the base orbit ---
    print("\n[validate] re-find base orbit with full-state Newton...")
    xv, Pv, okv, rv = solve(x0_base, T, L0, verbose=True)
    drift = np.linalg.norm(to_Z_vector(xv) - to_Z_vector(x0_base))
    print(f"  converged={okv} |F|={rv:.2e}  P={Pv:.6f} (T={T:.6f})  "
          f"Z-drift from base={drift:.2e}")
    out["validate"] = {"converged": bool(okv), "residual": float(rv),
                       "P": float(Pv), "T": float(T), "z_drift": float(drift)}
    if not okv or rv > 1e-8:
        print("  VALIDATION FAILED -> machinery is wrong; not trusting the "
              "doubled-branch results. Stopping.")
        json.dump(out, open("period_double_branch.json", "w"), indent=1)
        return
    print("  validation OK.\n")

    # --- doubled orbit: at L PAST L_PD the multiplier has split onto the
    # negative real axis; use the REAL eigenvector of the |lambda|>1 branch
    # (the genuine period-doubling direction) as the guess.
    found = None
    for L_t in [L_PD + 0.0002, L_PD + 0.0004, L_PD + 0.0006, L_PD + 0.001,
                L_PD + 0.0015]:
        a2, c2, T2, ok2, _ = newton_refine_bhh(a, c, L_t, T, tol=1e-12)
        if not ok2:
            continue
        xb2 = initial_conditions_from_params(a2, c2, L_t)
        M2, _ = compute_monodromy(xb2, T2)
        ev, evec = np.linalg.eig(M2)
        cand = [(abs(ev[i]), i) for i in range(12)
                if ev[i].real < 0 and abs(ev[i].imag) < 1e-3
                and abs(ev[i]) > 1.001]
        if not cand:
            print(f"  L={L_t:.6f}: not split yet (no real -lambda>1); skip")
            continue
        _, ki = max(cand)
        v2 = np.real(evec[:, ki]); v2 /= np.linalg.norm(v2)
        print(f"  L={L_t:.6f}: split eigenvalue {ev[ki].real:.4f}, "
              f"trying doubled guesses...")
        for eps in [1e-4, 3e-4, 1e-3, 3e-3, -1e-4, -3e-4, -1e-3, -3e-3]:
            x0g = project(xb2 + eps * v2)
            x0d, Pd, okd, rd = solve(x0g, 2.0 * T2, L_t)
            if not okd or rd > 1e-8:
                continue
            half = np.linalg.norm(
                to_Z_vector(integrate_orbit(x0d, Pd / 2).sol(Pd / 2))
                - to_Z_vector(x0d))
            if half > 1e-3 and abs(Pd - 2 * T2) < 0.5 * T2:
                found = {"L": L_t, "eps": eps, "x0": x0d.tolist(), "P": Pd,
                         "half_close": float(half),
                         "lambda_max": lam_max(x0d, Pd),
                         "E": float(compute_energy(x0d))}
                print(f"\nDOUBLED ORBIT FOUND: L={L_t:.6f} eps={eps:+.0e} "
                      f"P={Pd:.5f} (2T={2*T2:.5f}) half-close={half:.2e} "
                      f"lambda_max={found['lambda_max']:.4f}")
                break
        if found:
            break

    if not found:
        print("\nNo period-2T orbit converged from the eigenvector guesses "
              "(branch may be on the other side / need finer guess).")
        out["doubled"] = None
        json.dump(out, open("period_double_branch.json", "w"), indent=1)
        return

    # --- continue the doubled branch in L ---
    print("\nContinuing the doubled branch in L...")
    branch = [found]
    x_prev = np.array(found["x0"]); P_prev = found["P"]; L_cur = found["L"]
    for _ in range(20):
        L_cur += 0.0015
        x0d, Pd, okd, rd = solve(x_prev, P_prev, L_cur)
        if not okd or rd > 1e-8:
            print(f"  stop at L={L_cur:.5f} (no convergence)")
            break
        lm = lam_max(x0d, Pd)
        branch.append({"L": L_cur, "P": Pd, "lambda_max": lm,
                       "E": float(compute_energy(x0d)), "x0": x0d.tolist()})
        print(f"  L={L_cur:.5f} P={Pd:.4f} lambda_max={lm:.4f}")
        x_prev, P_prev = x0d, Pd

    out["doubled"] = found
    out["branch"] = [{kk: b[kk] for kk in ("L", "P", "lambda_max", "E")}
                     for b in branch]
    json.dump(out, open("period_double_branch.json", "w"), indent=1)
    print(f"\nSaved: period_double_branch.json  ({len(branch)} branch points)")


if __name__ == "__main__":
    main()
