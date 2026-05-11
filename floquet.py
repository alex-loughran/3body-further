"""Floquet stability analysis and Newton-Raphson refinement for periodic orbits.

Computes the monodromy matrix by integrating the variational equations
alongside the orbit for one period. The monodromy eigenvalues (Floquet
multipliers) classify linear stability. The monodromy matrix also serves
as the Jacobian for Newton-Raphson refinement of the periodicity condition.
"""

import numpy as np
from scipy.integrate import solve_ivp

from three_body import (
    _eom_core,
    to_Z_vector,
    integrate_orbit,
    build_state_symmetric,
    initial_conditions_from_params,
)


# ---------------------------------------------------------------------------
# Gravitational Jacobian: ∂(accelerations)/∂(positions)
# ---------------------------------------------------------------------------

def _gravity_jacobian(positions):
    """Compute the 6×6 Jacobian of gravitational accelerations w.r.t. positions.

    For body i, acceleration due to body j is:
        a_ij = -(r_i - r_j) / |r_i - r_j|^3

    The 2×2 block ∂a_i/∂r_j (off-diagonal, i≠j) is:
        [I/|r_ij|^3 - 3·(r_ij⊗r_ij)/|r_ij|^5]

    The 2×2 block ∂a_i/∂r_i (diagonal) is:
        -sum_{j≠i} [I/|r_ij|^3 - 3·(r_ij⊗r_ij)/|r_ij|^5]

    Parameters
    ----------
    positions : (6,) array — [x1, y1, x2, y2, x3, y3]

    Returns
    -------
    G : (6, 6) array — ∂(accelerations)/∂(positions)
    """
    r = positions.reshape(3, 2)
    G = np.zeros((6, 6))
    I2 = np.eye(2)

    for i in range(3):
        diag_block = np.zeros((2, 2))
        for j in range(3):
            if i == j:
                continue
            rij = r[i] - r[j]
            dist2 = rij @ rij
            dist = np.sqrt(dist2)
            inv3 = 1.0 / (dist * dist2)        # 1/|r|^3
            inv5 = inv3 / dist2                  # 1/|r|^5

            # The 2×2 interaction block
            block = I2 * inv3 - 3.0 * np.outer(rij, rij) * inv5

            # Off-diagonal: ∂a_i/∂r_j = +block
            G[2*i:2*i+2, 2*j:2*j+2] = block

            # Accumulate diagonal: ∂a_i/∂r_i = -sum of blocks
            diag_block -= block

        G[2*i:2*i+2, 2*i:2*i+2] = diag_block

    return G


def _validate_gravity_jacobian(positions, eps=1e-7):
    """Validate _gravity_jacobian against central finite differences on _eom_core.

    Returns (max_error, analytic, finite_diff).
    """


    G_analytic = _gravity_jacobian(positions)
    G_fd = np.zeros((6, 6))

    for k in range(6):
        state_plus = np.zeros(12)
        state_plus[:6] = positions.copy()
        state_plus[:6][k] += eps
        state_minus = np.zeros(12)
        state_minus[:6] = positions.copy()
        state_minus[:6][k] -= eps
        G_fd[:, k] = (_eom_core(state_plus)[6:] - _eom_core(state_minus)[6:]) / (2 * eps)

    max_err = np.max(np.abs(G_analytic - G_fd))
    return max_err, G_analytic, G_fd


# ---------------------------------------------------------------------------
# Extended equations of motion (orbit + variational)
# ---------------------------------------------------------------------------

def _eom_with_variational(t, state_ext):
    """Equations of motion for the 156-component extended state.

    state_ext = [orbit (12), Phi flattened (144)]

    The orbit part evolves under the standard three-body EOM.
    The Phi part evolves under dPhi/dt = J(t) @ Phi, where J is the
    12×12 Jacobian of the EOM evaluated along the orbit.
    """
    orbit = state_ext[:12]
    Phi = state_ext[12:].reshape(12, 12)

    # Standard orbit derivatives

    orbit_dot = _eom_core(orbit)

    # Build the 12×12 Jacobian: J = [[0, I], [G, 0]]
    # where G = ∂(accel)/∂(pos) is 6×6
    G = _gravity_jacobian(orbit[:6])

    J = np.zeros((12, 12))
    J[:6, 6:] = np.eye(6)    # ∂(velocities)/∂(velocities) = I
    J[6:, :6] = G             # ∂(accels)/∂(positions) = G

    # Variational equation: dPhi/dt = J @ Phi
    Phi_dot = J @ Phi

    deriv = np.empty(156)
    deriv[:12] = orbit_dot
    deriv[12:] = Phi_dot.ravel()
    return deriv


# ---------------------------------------------------------------------------
# Monodromy matrix computation
# ---------------------------------------------------------------------------

def compute_monodromy(state0, T_period, rtol=1e-12, atol=1e-14, max_step=0.01,
                      n_segments="auto"):
    """Integrate orbit + variational equations for one period to get monodromy matrix.

    For highly unstable or long-period orbits, the integration is split
    into shorter segments. Each segment integrates the variational equations
    independently (resetting Φ to I), producing a well-conditioned segment
    STM M_i. The full monodromy is then M = M_n · M_{n-1} · ... · M_1.

    Parameters
    ----------
    state0 : (12,) array — initial conditions
    T_period : float — orbital period
    rtol, atol : float — integrator tolerances
    max_step : float — maximum step size
    n_segments : int or "auto" — number of integration segments.
        "auto" (default) uses 1 segment for T < 20, otherwise ceil(T / 8)
        to keep each segment's condition number within float64 range.

    Returns
    -------
    monodromy : (12, 12) array — state transition matrix after one period
    final_state : (12,) array — orbit state at t=T_period
    """
    if n_segments == "auto":
        # For most orbits, single-shot is fine. Only trigger segmentation
        # for very long orbits (T > 60) where STM conditioning is likely
        # to be an issue. The auto-selection tries several segment counts
        # which is expensive, so the threshold is kept high.
        if T_period < 60:
            n_segments = 1
        else:
            base = max(2, int(np.ceil(T_period / 8)))
            candidates_n = [base - 1, base, base + 1, base + 2]
            candidates_n = [n for n in candidates_n if n >= 2]
            best_n = candidates_n[0]
            best_det_err = np.inf
            for n in candidates_n:
                try:
                    M_try, _ = _compute_segmented(
                        state0, T_period, n, rtol, atol, max_step)
                    det = np.prod(np.linalg.eigvals(M_try)).real
                    err = abs(det - 1.0)
                    if err < best_det_err:
                        best_det_err = err
                        best_n = n
                except RuntimeError:
                    continue
            n_segments = best_n

    if n_segments <= 1:
        return _compute_monodromy_single(
            state0, T_period, 0, T_period, rtol, atol, max_step)

    return _compute_segmented(state0, T_period, n_segments, rtol, atol, max_step)


def _compute_segmented(state0, T_period, n_segments, rtol, atol, max_step):
    """Segmented monodromy: M = M_n · ... · M_1."""
    t_boundaries = np.linspace(0, T_period, n_segments + 1)
    orbit_state = state0.copy()
    monodromy = np.eye(12)

    for seg in range(n_segments):
        t_start = t_boundaries[seg]
        t_end = t_boundaries[seg + 1]

        M_seg, orbit_state = _compute_monodromy_single(
            orbit_state, t_end - t_start, t_start, t_end, rtol, atol, max_step)

        monodromy = M_seg @ monodromy

    return monodromy, orbit_state


def _compute_monodromy_single(state0, duration, t_start, t_end,
                              rtol=1e-12, atol=1e-14, max_step=0.01):
    """Single-segment monodromy computation."""
    state_ext0 = np.zeros(156)
    state_ext0[:12] = state0
    state_ext0[12:] = np.eye(12).ravel()

    sol = solve_ivp(
        _eom_with_variational, [t_start, t_end], state_ext0,
        method="DOP853", rtol=rtol, atol=atol, max_step=max_step,
        dense_output=True,
    )
    if not sol.success:
        raise RuntimeError(f"Variational integration failed: {sol.message}")

    final = sol.sol(t_end)
    monodromy = final[12:].reshape(12, 12)
    final_state = final[:12]
    return monodromy, final_state


# ---------------------------------------------------------------------------
# Floquet multipliers
# ---------------------------------------------------------------------------

def floquet_multipliers(monodromy):
    """Extract Floquet multipliers (eigenvalues) from the monodromy matrix.

    Returns eigenvalues sorted by |λ|, largest first.
    """
    eigenvalues = np.linalg.eigvals(monodromy)
    # Sort by magnitude, largest first
    idx = np.argsort(-np.abs(eigenvalues))
    return eigenvalues[idx]


def classify_stability(multipliers, tol=1e-3):
    """Classify orbit stability from Floquet multipliers.

    Returns a dict with:
        stable_count: number of |λ| < 1 directions
        unstable_count: number of |λ| > 1 directions
        marginal_count: number of |λ| ≈ 1 directions
        max_instability: largest |λ| (> 1 means unstable)
        is_stable: True if no |λ| > 1+tol (excluding marginal)
        determinant: product of all multipliers (should be 1)
        n_unit: count of |λ| ≈ 1 (should be ≥ 2)
    """
    mags = np.abs(multipliers)
    det = np.prod(multipliers).real

    n_unit = np.sum(np.abs(mags - 1.0) < tol)
    n_stable = np.sum(mags < 1.0 - tol)
    n_unstable = np.sum(mags > 1.0 + tol)

    return {
        "stable_count": int(n_stable),
        "unstable_count": int(n_unstable),
        "marginal_count": int(n_unit),
        "max_instability": float(np.max(mags)),
        "is_stable": int(n_unstable) == 0,
        "determinant": float(det),
        "n_unit": int(n_unit),
        "multipliers": multipliers,
    }


def validate_monodromy(multipliers, det_tol=1e-2, unit_tol=1e-2, verbose=True):
    """Check that the monodromy matrix passes basic sanity tests.

    Returns True if all checks pass.
    """
    mags = np.abs(multipliers)
    det = np.prod(multipliers).real

    # Check 1: determinant should be 1 (symplecticity)
    det_ok = abs(det - 1.0) < det_tol
    # Check 2: at least two unit eigenvalues
    n_unit = np.sum(np.abs(mags - 1.0) < unit_tol)
    unit_ok = n_unit >= 2

    if verbose:
        print(f"  Determinant: {det:.8f} (should be 1.0) — {'PASS' if det_ok else 'FAIL'}")
        print(f"  Unit eigenvalues: {n_unit} (should be ≥ 2) — {'PASS' if unit_ok else 'FAIL'}")
        print(f"  |multipliers|: {np.sort(mags)}")

    return det_ok and unit_ok


# ---------------------------------------------------------------------------
# Newton-Raphson refinement (reduced parametrisation)
# ---------------------------------------------------------------------------

def _monodromy_jacobian(param_to_state, params, T, use_z_space=False,
                        eps=1e-8):
    """Compute Jacobian of the periodicity residual using the monodromy matrix.

    The monodromy matrix M from the variational integration gives:
        ∂(state(T))/∂(state0) = M

    For the residual F = state(T) - state0 (Cartesian) or Z(T) - Z(0):
        ∂F/∂(params) = (M - I) @ ∂(state0)/∂(params)   [Cartesian]
        ∂F/∂T = f(state(T))                              [EOM at endpoint]

    The ∂(state0)/∂(params) Jacobian is computed via cheap finite differences
    on param_to_state (no integration needed).

    Returns (J, F, monodromy, final_state) where J is (m, n+1).
    """


    n = len(params)
    state0 = param_to_state(params)
    # Use n_segments=1 for the Newton Jacobian — it only needs an
    # approximate search direction, not a high-accuracy monodromy.
    # The auto-segment selection is too expensive to run every iteration.
    M, final_state = compute_monodromy(state0, T, n_segments=1)

    # Residual
    if use_z_space:
        F = to_Z_vector(final_state) - to_Z_vector(state0)
    else:
        F = final_state - state0
    m = len(F)

    # ∂(state0)/∂(params) via finite differences on param_to_state (cheap)
    ds0_dp = np.zeros((12, n))
    for k in range(n):
        p_plus = params.copy()
        p_minus = params.copy()
        p_plus[k] += eps
        p_minus[k] -= eps
        ds0_dp[:, k] = (param_to_state(p_plus) - param_to_state(p_minus)) / (2 * eps)

    J = np.zeros((m, n + 1))

    if use_z_space:
        # Need ∂Z/∂state at both endpoints for the chain rule
        # ∂F/∂(params) = ∂Z/∂state|_T @ M @ ds0_dp - ∂Z/∂state|_0 @ ds0_dp
        dZ_ds_T = _z_jacobian(final_state)     # (6, 12)
        dZ_ds_0 = _z_jacobian(state0)           # (6, 12)
        J[:, :n] = dZ_ds_T @ M @ ds0_dp - dZ_ds_0 @ ds0_dp
        J[:, n] = dZ_ds_T @ _eom_core(final_state)
    else:
        J[:, :n] = (M - np.eye(12)) @ ds0_dp
        J[:, n] = _eom_core(final_state)

    return J, F, M, final_state


def _z_jacobian(state, eps=1e-8):
    """Compute the 6×12 Jacobian ∂Z/∂state via finite differences."""

    Z0 = to_Z_vector(state)
    J = np.zeros((6, 12))
    for k in range(12):
        s_plus = state.copy()
        s_minus = state.copy()
        s_plus[k] += eps
        s_minus[k] -= eps
        J[:, k] = (to_Z_vector(s_plus) - to_Z_vector(s_minus)) / (2 * eps)
    return J


def _finite_diff_jacobian(param_to_state, params, T, use_z_space=False,
                          eps=1e-8):
    """Compute Jacobian of the periodicity residual w.r.t. parameters + T.

    Uses central finite differences. For n parameters, returns an
    (m, n+1) matrix where m=6 if use_z_space else 12, and the last
    column is ∂F/∂T.

    This is the fallback method — prefer _monodromy_jacobian which uses
    the variational equations and requires only 1 integration per step.
    """
    n = len(params)
    F0 = _periodicity_residual(param_to_state, params, T, use_z_space)
    m = len(F0)

    J = np.zeros((m, n + 1))

    # Partials w.r.t. parameters
    for k in range(n):
        p_plus = params.copy()
        p_minus = params.copy()
        p_plus[k] += eps
        p_minus[k] -= eps
        F_plus = _periodicity_residual(param_to_state, p_plus, T, use_z_space)
        F_minus = _periodicity_residual(param_to_state, p_minus, T, use_z_space)
        J[:, k] = (F_plus - F_minus) / (2 * eps)

    # Partial w.r.t. T
    F_plus = _periodicity_residual(param_to_state, params, T + eps, use_z_space)
    F_minus = _periodicity_residual(param_to_state, params, T - eps, use_z_space)
    J[:, n] = (F_plus - F_minus) / (2 * eps)

    return J


def _periodicity_residual(param_to_state, params, T, use_z_space=False):
    """Compute the periodicity residual for given parameters and period.

    If use_z_space is True, computes Z(T) - Z(0) in the rotation-invariant
    shape-sphere phase space (6 components). This is necessary for L≠0 orbits
    where the Cartesian positions don't return but the shape does.

    If False, computes x(T) - x(0) in Cartesian space (12 components).
    Only valid for L=0 orbits.
    """

    state0 = param_to_state(params)
    sol = integrate_orbit(state0, T)
    state_T = sol.sol(T)
    if use_z_space:
        return to_Z_vector(state_T) - to_Z_vector(state0)
    return state_T - state0


def refine_newton(params0, T_guess, param_to_state, max_iter=20,
                  tol=1e-10, use_z_space=False, use_monodromy=True,
                  verbose=False):
    """Refine a periodic orbit using Newton-Raphson in reduced parameter space.

    Instead of solving the full 13-variable system (which is rank-deficient
    due to the many symmetries of the three-body problem), works in the
    reduced space of the parametrisation. For symmetric orbits this is
    (vx, vy, T) — 3 unknowns. For BHH it's (a, c, T) — also 3 unknowns.

    Uses least-squares to solve the overdetermined system, which naturally
    picks the minimum-norm correction.

    Parameters
    ----------
    params0 : array — initial guess for parameters (e.g. [vx, vy])
    T_guess : float — initial guess for the period
    param_to_state : callable — maps params array to 12-component state vector
    max_iter : int — maximum Newton iterations
    tol : float — convergence tolerance on |F|
    use_z_space : bool — if True, use rotation-invariant Z-space residual
        (necessary for L≠0 orbits where Cartesian positions don't return)
    use_monodromy : bool — if True (default), use monodromy-based Jacobian.
        Falls back to finite differences if False.
    verbose : bool — print convergence info

    Returns
    -------
    params_refined : array — refined parameters
    T_refined : float — refined period
    converged : bool
    info : dict with convergence history and final monodromy (if use_monodromy)
    """
    params = np.array(params0, dtype=float)
    T = T_guess
    history = []
    last_monodromy = None

    for iteration in range(max_iter):
        if use_monodromy:
            J, F, M, final_state = _monodromy_jacobian(
                param_to_state, params, T, use_z_space)
            last_monodromy = M
        else:
            F = _periodicity_residual(param_to_state, params, T, use_z_space)
            J = _finite_diff_jacobian(param_to_state, params, T, use_z_space)

        residual = np.linalg.norm(F)
        history.append(residual)

        if verbose:
            print(f"  Newton iter {iteration}: |F| = {residual:.4e}")

        if residual < tol:
            if verbose:
                print(f"  Converged in {iteration} iterations")
            state = param_to_state(params)
            info = {"history": history, "iterations": iteration, "state": state}
            if last_monodromy is not None:
                info["monodromy"] = last_monodromy
            return params, T, True, info

        # Stagnation detection: if residual has plateaued and is
        # already small, declare convergence at the precision floor.
        # The 1e-5 threshold handles long-period orbits where the
        # monodromy Jacobian is noisy due to accumulated integration error.
        if len(history) >= 4:
            recent = history[-4:]
            if max(recent) / min(recent) < 3.0 and residual < 1e-5:
                if verbose:
                    print(f"  Stagnated at |F| = {residual:.4e} (precision floor)")
                state = param_to_state(params)
                info = {"history": history, "iterations": iteration, "state": state}
                if last_monodromy is not None:
                    info["monodromy"] = last_monodromy
                return params, T, True, info

        # Solve overdetermined system via least squares
        delta, _, _, _ = np.linalg.lstsq(J, -F, rcond=None)

        params = params + delta[:-1]
        T = T + delta[-1]

    if verbose:
        print(f"  Did not converge in {max_iter} iterations (|F| = {history[-1]:.4e})")
    state = param_to_state(params)
    info = {"history": history, "iterations": max_iter, "state": state}
    if last_monodromy is not None:
        info["monodromy"] = last_monodromy
    return params, T, False, info


# ---------------------------------------------------------------------------
# Convenience wrappers for common parametrisations
# ---------------------------------------------------------------------------

def newton_refine_symmetric(vx, vy, T_guess, **kwargs):
    """Refine a symmetric orbit (L=0) via Newton-Raphson.

    Returns (vx, vy, T, converged, info).
    """


    def builder(p):
        return build_state_symmetric(p[0], p[1])

    params, T, converged, info = refine_newton(
        [vx, vy], T_guess, builder, **kwargs)
    return params[0], params[1], T, converged, info


def newton_refine_bhh(a, c, L, T_guess, b=1.0, **kwargs):
    """Refine a BHH orbit (L≠0) via Newton-Raphson in Z-space.

    Returns (a, c, T, converged, info).
    """


    def builder(p):
        return initial_conditions_from_params(p[0], p[1], L, b)

    # BHH orbits have L≠0 so Cartesian state doesn't return — use Z-space
    kwargs.setdefault("use_z_space", True)
    params, T, converged, info = refine_newton(
        [a, c], T_guess, builder, **kwargs)
    return params[0], params[1], T, converged, info


def analyse_orbit(state0, T_period, verbose=True):
    """One-shot Floquet analysis: monodromy, multipliers, stability.

    Returns dict with monodromy, multipliers, stability info, and validation.
    """
    M, final_state = compute_monodromy(state0, T_period)
    mults = floquet_multipliers(M)
    stab = classify_stability(mults)
    valid = validate_monodromy(mults, verbose=verbose)

    return {
        "monodromy": M,
        "final_state": final_state,
        "multipliers": mults,
        "stability": stab,
        "valid": valid,
    }
