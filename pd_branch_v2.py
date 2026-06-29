"""Period-doubled branch off the L=0.831 stable b^4 orbit -- attempt #3.

Diagnosis of why attempts #1/#2 failed (period_double_branch.py / pd_branch_fix.py):
the SEED was never the problem -- every solve in pd_branch_fix.json *diverged*
(|F| ~ 1..12, monotonically growing), none fell back to base. The home-grown
Gauss-Newton in period_double_branch.solve() takes the full lstsq min-norm step
(merely clipped to norm 1.0). For a near-collision b^4 orbit with |lambda|~3 PER
PERIOD, integrated over 2T (~9x amplification), the shooting Jacobian is brutally
ill-conditioned and that step overshoots -> the residual explodes.

Fix here:
  (1) locate_L_PD(): bisect L for the lambda=-1 crossing (where a complex pair
      reaches angle pi and splits onto the negative real axis). Branch-switch AT
      L_PD, where the daughter amplitude -> 0, not far past it.
  (2) solve_lm(): Levenberg-Marquardt corrector with diagonal (Marquardt) scaling
      and adaptive damping. The mu*diag(JtJ) term both tames the ill-conditioning
      AND regularizes the gauge nullspace -- replacing the lstsq min-norm hack.
  (3) seed with the clean -1 eigenvector at L_PD+dL, amplitude ~ A*sqrt(dL)
      (pitchfork/PD amplitude law), scanned finely, both signs.

Reuses the VALIDATED residual/jacobian/projection from period_double_branch
(that module's full-state Newton re-finds the base orbit to 1e-12).

Usage: python pd_branch_v2.py
Outputs: pd_branch_v2.json, pd_branch_v2_report.txt
"""

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"

import json
import numpy as np

from three_body import (
    initial_conditions_from_params, integrate_orbit, to_Z_vector,
    compute_angular_momentum, compute_energy,
)
from floquet import compute_monodromy, newton_refine_bhh, floquet_multipliers
from period_double_branch import (
    residual, jacobian, project, rot_gen, dL_dx, lam_max, L_PD, T_BASE,
)
from pd_branch_fix import split_eigvec, clean_direction, SEED


def solve_lm(x0_guess, P_guess, L_target, max_iter=60, tol=1e-10,
             mu0=1e-3, verbose=False):
    """Levenberg-Marquardt full-state shooting corrector.

    Same residual/Jacobian as period_double_branch.solve(), but the update is
    (JtJ + mu*diag(JtJ)) d = -Jt F with mu adapted by the gain ratio. Diagonal
    scaling makes it scale-invariant; the damping regularizes both the
    ill-conditioning and the gauge nullspace, so it cannot take the wild
    overshoot that blew up the previous attempts.
    """
    x0 = project(np.array(x0_guess, float))
    P = float(P_guess)
    x_ref = x0.copy()
    f_ref = np.array(_flow(x_ref))
    g_ref = rot_gen(x_ref)
    mu = mu0
    try:
        F = residual(x0, P, L_target, x_ref, f_ref, g_ref)
    except (RuntimeError, FloatingPointError):
        return x0, P, False, np.inf
    nrm = float(np.linalg.norm(F))

    for it in range(max_iter):
        if verbose:
            print(f"    it {it:2d}: |F|={nrm:.3e}  mu={mu:.1e}", flush=True)
        if nrm < tol:
            return x0, P, True, nrm
        try:
            J, _, _ = jacobian(x0, P, f_ref, g_ref)          # 9 x 13
        except (np.linalg.LinAlgError, RuntimeError, FloatingPointError):
            return x0, P, False, nrm
        JtJ = J.T @ J
        JtF = J.T @ F
        diag = np.clip(np.diag(JtJ), 1e-12, None)
        accepted = False
        for _try in range(30):                                # adapt mu inline
            A = JtJ + mu * np.diag(diag)
            try:
                delta = np.linalg.solve(A, -JtF)
            except np.linalg.LinAlgError:
                mu *= 4.0
                continue
            x0n = project(x0 + delta[:12])
            Pn = P + delta[12]
            if not np.isfinite(Pn) or Pn <= 0:
                mu *= 4.0
                continue
            try:
                Fn = residual(x0n, Pn, L_target, x_ref, f_ref, g_ref)
                nn = float(np.linalg.norm(Fn))
            except (RuntimeError, FloatingPointError):
                mu *= 4.0
                continue
            if nn < nrm:                                       # step improved
                x0, P, F, nrm = x0n, Pn, Fn, nn
                mu = max(mu * 0.4, 1e-12)
                accepted = True
                break
            mu *= 3.0
            if mu > 1e14:
                break
        if not accepted:
            return x0, P, False, nrm
    return x0, P, False, nrm


def _flow(s):
    from three_body import _eom_core
    return _eom_core(s)


def half_close(x0, P):
    """Distance (rotation-reduced) between the half-period point and the start.
    Large for a genuine period-2T orbit, ~0 if it collapsed to the base T-orbit."""
    xh = integrate_orbit(x0, P / 2).sol(P / 2)
    return float(np.linalg.norm(to_Z_vector(xh) - to_Z_vector(x0)))


def locate_L_PD(a0, c0, T0, lo=0.8305, hi=0.8320, iters=22):
    """Bisect L for the lambda=-1 crossing: below it the destabilizing pair is
    still complex on the unit circle (split_eigvec -> None); above it a real
    negative multiplier |lambda|>1 has appeared. Returns (L_PD, a,c,T at L_PD)."""
    def is_split(L):
        a, c, T, ok, _ = newton_refine_bhh(a0, c0, L, T0, tol=1e-12)
        if not ok:
            return None, (a0, c0, T0)
        xb = initial_conditions_from_params(a, c, L)
        M, _ = compute_monodromy(xb, T)
        lam, _v = split_eigvec(M)
        return (lam is not None), (a, c, T)

    s_lo, base_lo = is_split(lo)
    s_hi, base_hi = is_split(hi)
    print(f"  bracket: split({lo})={s_lo}  split({hi})={s_hi}", flush=True)
    if s_lo is None or s_hi is None or s_lo == s_hi:
        print("  WARN: could not bracket the -1 crossing in [%.4f,%.4f]"
              % (lo, hi), flush=True)
        # fall back to the labelled value
        a, c, T, _, _ = newton_refine_bhh(a0, c0, L_PD, T0, tol=1e-12)
        return L_PD, (a, c, T)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        s_mid, base_mid = is_split(mid)
        if s_mid is None:
            lo = mid                       # treat refine-fail as "before"
            continue
        if s_mid == s_lo:
            lo, s_lo = mid, s_mid
        else:
            hi, base_hi = mid, base_mid
    L_pd = hi                              # first L that is split
    a, c, T, _, _ = newton_refine_bhh(a0, c0, L_pd, T0, tol=1e-12)
    return L_pd, (a, c, T)


def main():
    out = {}
    print("=== refine base orbit & locate L_PD precisely ===", flush=True)
    a, c, T, ok, _ = newton_refine_bhh(SEED["a"], SEED["c"], L_PD, T_BASE,
                                       tol=1e-12)
    print(f"  base seed @L={L_PD}: a={a:.8f} c={c:.8f} T={T:.8f} ok={ok}",
          flush=True)

    L_pd, (a_pd, c_pd, T_pd) = locate_L_PD(a, c, T)
    xb_pd = initial_conditions_from_params(a_pd, c_pd, L_pd)
    M_pd, _ = compute_monodromy(xb_pd, T_pd)
    mults = floquet_multipliers(M_pd)
    print(f"  L_PD ~= {L_pd:.7f}  T={T_pd:.6f}", flush=True)
    print(f"  |lambda|_max at L_PD = {max(abs(m) for m in mults):.5f}",
          flush=True)
    out["L_PD"] = L_pd
    out["base_at_L_PD"] = {"a": a_pd, "c": c_pd, "T": T_pd}

    # --- branch switch: seed at L_PD+dL with clean -1 eigenvector, amp~sqrt(dL)
    print("\n=== branch-switch search (LM corrector) ===", flush=True)
    print(f"  {'L':>10} {'dL':>8} {'eps':>9} {'|F|':>10} {'P/2T':>7} "
          f"{'half':>10} {'verdict':>9}", flush=True)
    found = None
    scan = []
    for dL in [3e-4, 6e-4, 1e-3, 2e-3, 4e-3]:
        L_t = L_pd + dL
        a2, c2, T2, ok2, _ = newton_refine_bhh(a_pd, c_pd, L_t, T_pd, tol=1e-12)
        if not ok2:
            continue
        xb2 = initial_conditions_from_params(a2, c2, L_t)
        M2, _ = compute_monodromy(xb2, T2)
        lam2, v2 = split_eigvec(M2)
        if lam2 is None:
            print(f"  L={L_t:.6f}: not split; skip", flush=True)
            continue
        v2c, _ = clean_direction(v2, xb2)
        amp = np.sqrt(dL)
        for fac in [0.5, 1.0, 1.5, 2.0, 3.0, -0.5, -1.0, -1.5, -2.0, -3.0]:
            eps = fac * amp
            x0g = project(xb2 + eps * v2c)
            x0d, Pd, okd, rd = solve_lm(x0g, 2.0 * T2, L_t)
            hc = np.nan
            verdict = "diverge"
            if okd and rd < 1e-8:
                hc = half_close(x0d, Pd)
                if hc > 1e-3 and abs(Pd - 2 * T2) < 0.5 * T2:
                    verdict = "DOUBLED"
                elif hc < 1e-4:
                    verdict = "->base"
                else:
                    verdict = "other"
            scan.append({"L": L_t, "dL": dL, "eps": float(eps),
                         "F": float(rd), "P": float(Pd),
                         "half": None if hc != hc else hc, "verdict": verdict})
            mark = "  <==" if verdict == "DOUBLED" else ""
            hs = "  nan" if hc != hc else f"{hc:.2e}"
            print(f"  {L_t:>10.6f} {dL:>8.4f} {eps:>9.4f} {rd:>10.2e} "
                  f"{Pd/(2*T2):>7.3f} {hs:>10} {verdict:>9}{mark}", flush=True)
            if verdict == "DOUBLED" and found is None:
                found = {"L": L_t, "eps": float(eps), "x0": x0d.tolist(),
                         "P": float(Pd), "half_close": hc,
                         "lambda_max": lam_max(x0d, Pd),
                         "E": float(compute_energy(x0d))}
        if found:
            break

    out["scan"] = scan
    out["doubled"] = found
    json.dump(out, open("pd_branch_v2.json", "w"), indent=1)

    lines = ["", "=" * 60, "PERIOD-DOUBLED BRANCH (attempt #3, LM corrector)",
             "=" * 60, f"  L_PD (lambda=-1 crossing): {L_pd:.7f}"]
    if not found:
        lines.append("  RESULT: no period-2T daughter converged.")
        lines.append("  (the -1 crossing itself still documents the PD "
                     "bifurcation; daughter construction unresolved.)")
        print("\n".join(lines), flush=True)
        open("pd_branch_v2_report.txt", "w").write("\n".join(lines))
        print("\nSaved: pd_branch_v2.json, pd_branch_v2_report.txt", flush=True)
        return

    print(f"\nDOUBLED ORBIT FOUND: L={found['L']:.6f} eps={found['eps']:+.4f} "
          f"P={found['P']:.5f} (2T={2*T2:.5f}) half={found['half_close']:.2e} "
          f"lambda_max={found['lambda_max']:.4f}", flush=True)

    # --- continue the daughter branch in L to draw the bifurcation diagram ---
    print("\n=== continue daughter branch in L ===", flush=True)
    branch = [found]
    x_prev = np.array(found["x0"]); P_prev = found["P"]; L_cur = found["L"]
    for _ in range(25):
        L_cur += 0.0015
        x0d, Pd, okd, rd = solve_lm(x_prev, P_prev, L_cur)
        if not okd or rd > 1e-8:
            print(f"  stop at L={L_cur:.5f} (no convergence)", flush=True)
            break
        lm = lam_max(x0d, Pd)
        branch.append({"L": L_cur, "P": Pd, "lambda_max": lm,
                       "E": float(compute_energy(x0d)), "x0": x0d.tolist()})
        print(f"  L={L_cur:.5f} P={Pd:.4f} lambda_max={lm:.4f}", flush=True)
        x_prev, P_prev = x0d, Pd

    out["branch"] = [{k: b[k] for k in ("L", "P", "lambda_max", "E")}
                     for b in branch]
    json.dump(out, open("pd_branch_v2.json", "w"), indent=1)

    lines += [f"  daughter found at L={found['L']:.6f}, P={found['P']:.5f}",
              f"  daughter lambda_max={found['lambda_max']:.3f} "
              f"({'UNSTABLE' if found['lambda_max'] > 1.001 else 'stable'})",
              f"  branch traced over {len(branch)} points in L"]
    open("pd_branch_v2_report.txt", "w").write("\n".join(lines))
    print("\n".join(lines), flush=True)
    print(f"\nSaved: pd_branch_v2.json ({len(branch)} branch pts), "
          f"pd_branch_v2_report.txt", flush=True)


if __name__ == "__main__":
    main()
