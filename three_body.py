"""Reusable functions for hunting periodic three-body orbits.

Based on: Janković, Dmitrašinović & Šuvakov,
Computer Physics Communications 250 (2020) 107052

Equal masses m=1, gravitational constant G=1, planar (2D) problem.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar, minimize
from scipy.ndimage import maximum_filter
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Equations of motion (vectorized NumPy)
# ---------------------------------------------------------------------------

try:
    from numba import njit

    @njit(cache=True)
    def _eom_core(state):
        """Core EOM computation (Numba JIT-compiled)."""
        out = np.empty(12)
        # Velocities -> position derivatives
        for i in range(6):
            out[i] = state[i + 6]

        # Accelerations from pairwise gravitational interactions
        for i in range(6):
            out[i + 6] = 0.0

        for i in range(3):
            for j in range(i + 1, 3):
                dx = state[2 * i] - state[2 * j]
                dy = state[2 * i + 1] - state[2 * j + 1]
                dist2 = dx * dx + dy * dy
                inv_dist3 = dist2 ** (-1.5)
                fx = dx * inv_dist3
                fy = dy * inv_dist3
                # Body i: a_i += -(r_i - r_j) / |r_ij|^3
                out[6 + 2 * i] -= fx
                out[6 + 2 * i + 1] -= fy
                # Body j: a_j += -(r_j - r_i) / |r_ij|^3 = +(r_i - r_j) / |r_ij|^3
                out[6 + 2 * j] += fx
                out[6 + 2 * j + 1] += fy
        return out

except ImportError:
    def _eom_core(state):
        """Core EOM computation using vectorized NumPy (fallback without Numba)."""
        out = np.empty(12)
        out[:6] = state[6:]

        r = state[:6].reshape(3, 2)
        diff = r[:, np.newaxis, :] - r[np.newaxis, :, :]
        dist2 = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist2, 1.0)
        inv_dist3 = dist2 ** (-1.5)
        np.fill_diagonal(inv_dist3, 0.0)

        accel = -np.einsum('ijk,ij->ik', diff, inv_dist3)
        out[6:] = accel.ravel()
        return out


def three_body_eom(t, state):
    """Equations of motion for the planar three-body problem (equal masses, G=1).

    state = [x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3]
    """
    return _eom_core(state)


# ---------------------------------------------------------------------------
# Coordinate transforms
# ---------------------------------------------------------------------------

def to_jacobi(r, v):
    """Convert Cartesian positions/velocities to Jacobi coordinates.

    Parameters
    ----------
    r : (3, 2) array -- positions
    v : (3, 2) array -- velocities

    Returns
    -------
    rho, lam, rho_dot, lam_dot : each a 2-vector
    """
    rho = (r[0] - r[1]) / np.sqrt(2)
    lam = (r[0] + r[1] - 2 * r[2]) / np.sqrt(6)
    rho_dot = (v[0] - v[1]) / np.sqrt(2)
    lam_dot = (v[0] + v[1] - 2 * v[2]) / np.sqrt(6)
    return rho, lam, rho_dot, lam_dot


def to_shape_sphere(rho, lam):
    """Compute shape-sphere coordinates (x, y, z) from Jacobi vectors."""
    R2 = rho @ rho + lam @ lam
    x = 2 * (rho @ lam) / R2
    y = (lam @ lam - rho @ rho) / R2
    z = 2 * (rho[0] * lam[1] - rho[1] * lam[0]) / R2
    return np.array([x, y, z])


def to_Z_vector(state):
    """Compute the 6-vector Z = (x, y, z, xdot, ydot, zdot) from state.

    This is the rotation-invariant phase-space vector used for the
    return proximity function.
    """
    r = state[:6].reshape(3, 2)
    v = state[6:].reshape(3, 2)
    rho, lam, rho_dot, lam_dot = to_jacobi(r, v)

    R2 = rho @ rho + lam @ lam
    R = np.sqrt(R2)

    x = 2 * (rho @ lam) / R2
    y = (lam @ lam - rho @ rho) / R2
    z = 2 * (rho[0] * lam[1] - rho[1] * lam[0]) / R2

    R_dot = (rho @ rho_dot + lam @ lam_dot) / R
    dR2dt = 2 * R * R_dot

    x_dot = (2 * (rho_dot @ lam + rho @ lam_dot) * R2
             - 2 * (rho @ lam) * dR2dt) / R2**2
    y_dot = (2 * (lam @ lam_dot - rho @ rho_dot) * R2
             - (lam @ lam - rho @ rho) * dR2dt) / R2**2
    z_num = 2 * (rho[0] * lam[1] - rho[1] * lam[0])
    z_num_dot = 2 * (rho_dot[0] * lam[1] + rho[0] * lam_dot[1]
                     - rho_dot[1] * lam[0] - rho[1] * lam_dot[0])
    z_dot = (z_num_dot * R2 - z_num * dR2dt) / R2**2

    return np.array([x, y, z, x_dot, y_dot, z_dot])


# ---------------------------------------------------------------------------
# Initial conditions & conserved quantities
# ---------------------------------------------------------------------------

def initial_conditions_from_params(a, c, L, b=1.0):
    """Build the 12-component state vector from BHH parameters (a, b, c, d).

    d is determined from angular momentum: L = a*c + b*d => d = (L - a*c) / b.
    """
    d = (L - a * c) / b

    rho0 = np.array([a, 0.0])
    lam0 = np.array([b, 0.0])
    rho_dot0 = np.array([0.0, c])
    lam_dot0 = np.array([0.0, d])

    r1 = rho0 / np.sqrt(2) + lam0 / np.sqrt(6)
    r2 = -rho0 / np.sqrt(2) + lam0 / np.sqrt(6)
    r3 = -2 * lam0 / np.sqrt(6)

    v1 = rho_dot0 / np.sqrt(2) + lam_dot0 / np.sqrt(6)
    v2 = -rho_dot0 / np.sqrt(2) + lam_dot0 / np.sqrt(6)
    v3 = -2 * lam_dot0 / np.sqrt(6)

    return np.concatenate([r1, r2, r3, v1, v2, v3])


def compute_energy(state):
    """Compute total energy (kinetic + potential)."""
    r = state[:6].reshape(3, 2)
    v = state[6:].reshape(3, 2)
    KE = 0.5 * np.sum(v**2)
    PE = 0.0
    for i in range(3):
        for j in range(i + 1, 3):
            dr = r[i] - r[j]
            PE -= 1.0 / np.sqrt(dr @ dr)
    return KE + PE


def compute_angular_momentum(state):
    """Compute total angular momentum (z-component)."""
    r = state[:6].reshape(3, 2)
    v = state[6:].reshape(3, 2)
    L = 0.0
    for i in range(3):
        L += r[i, 0] * v[i, 1] - r[i, 1] * v[i, 0]
    return L


def rescale_to_energy(state, E_target=-0.5):
    """Rescale a state to a target energy using the Newtonian scaling symmetry.

    Under r -> alpha*r, v -> v/sqrt(alpha), t -> alpha^(3/2)*t,
    energy scales as E -> E/alpha.

    Returns (new_state, time_factor) where time_factor = alpha^(3/2).
    """
    E_current = compute_energy(state)
    if E_current >= 0 or E_target >= 0:
        raise ValueError(f"Both energies must be negative (E_current={E_current}, E_target={E_target})")
    alpha = E_current / E_target
    new_state = np.empty_like(state)
    new_state[:6] = state[:6] * alpha
    new_state[6:] = state[6:] / np.sqrt(alpha)
    time_factor = alpha**1.5
    return new_state, time_factor


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

def integrate_orbit(state0, T_max, rtol=1e-12, atol=1e-14, max_step=0.01):
    """Integrate the three-body equations of motion from t=0 to T_max.

    DOP853 (8th-order Runge-Kutta-Fehlberg) is used rather than symplectic
    methods (e.g. velocity Verlet) for three reasons specific to orbit hunting:

    1. **Single-period precision over long-term stability**: the return
       proximity function (Section 2.4) requires d_min ~ 10^-7 after one
       period, demanding energy conservation at ~10^-12. Symplectic methods
       conserve energy *boundedly* over long times but achieve lower per-step
       accuracy at the same computational cost (Verlet is O(dt^2) vs DOP853's
       O(dt^8)). See Hairer, Norsett & Wanner, "Solving ODEs I" (1993), Ch. II.

    2. **Adaptive timestep**: three-body orbits include close encounters where
       the gravitational potential varies rapidly. Fixed-step methods either
       waste time in slow phases or lose accuracy during fast ones. DOP853
       adapts dt automatically based on local error estimates.

    3. **Dense output**: the r.p.f. refinement (minimize_scalar) requires
       evaluating the solution at arbitrary times between integration steps.
       DOP853 provides this via polynomial interpolation at no extra cost.
       Fixed-step methods would require separate interpolation or a very fine
       grid.

    Raises RuntimeError on integration failure (including near-collisions).
    """
    sol = solve_ivp(
        three_body_eom, [0, T_max], state0,
        method="DOP853", rtol=rtol, atol=atol, max_step=max_step,
        dense_output=True
    )
    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")
    return sol


# ---------------------------------------------------------------------------
# Return proximity function (r.p.f.)
# ---------------------------------------------------------------------------

def _z_distance(t, sol, Z0):
    """Distance in Z-space between state at time t and Z0."""
    st = sol.sol(t)
    Z = to_Z_vector(st)
    return np.linalg.norm(Z - Z0)


def return_proximity(state0, T_max, t_min_frac=0.1, n_samples=5000):
    """Compute the return proximity function with continuous refinement.

    Two-stage approach:
    1. Coarse sampling to find all local minima of d(t).
    2. scipy.optimize.minimize_scalar to refine each local minimum.

    Returns (d_min, t_min, sol).
    """
    sol = integrate_orbit(state0, T_max)
    Z0 = to_Z_vector(state0)

    t_eval = np.linspace(T_max * t_min_frac, T_max, n_samples)
    d_vals = np.array([_z_distance(t, sol, Z0) for t in t_eval])

    # Find all local minima (points lower than both neighbours)
    local_min_indices = []
    for i in range(1, len(d_vals) - 1):
        if d_vals[i] < d_vals[i - 1] and d_vals[i] < d_vals[i + 1]:
            local_min_indices.append(i)

    # Also check the last point in case minimum is at the boundary
    if len(local_min_indices) == 0:
        local_min_indices.append(np.argmin(d_vals))

    # Refine each local minimum with minimize_scalar
    best_d = np.inf
    best_t = 0.0

    for idx in local_min_indices:
        lo = t_eval[max(idx - 1, 0)]
        hi = t_eval[min(idx + 1, len(t_eval) - 1)]
        try:
            res = minimize_scalar(
                _z_distance, bounds=(lo, hi), method="bounded",
                args=(sol, Z0),
                options={"xatol": 1e-14}
            )
            if res.fun < best_d:
                best_d = res.fun
                best_t = res.x
        except (ValueError, RuntimeError):
            # ValueError: degenerate bounds (lo == hi)
            # RuntimeError: minimiser convergence failure
            if d_vals[idx] < best_d:
                best_d = d_vals[idx]
                best_t = t_eval[idx]

    return best_d, best_t, sol


def return_proximity_curve(state0, T_max, t_min_frac=0.1, n_samples=5000):
    """Compute the full d(t) curve for diagnostics.

    Returns (t_array, d_array, d_min, t_min, sol).
    """
    sol = integrate_orbit(state0, T_max)
    Z0 = to_Z_vector(state0)

    t_eval = np.linspace(T_max * t_min_frac, T_max, n_samples)
    d_vals = np.array([_z_distance(t, sol, Z0) for t in t_eval])

    # Refine the global minimum
    i_min = np.argmin(d_vals)
    lo = t_eval[max(i_min - 1, 0)]
    hi = t_eval[min(i_min + 1, len(t_eval) - 1)]
    res = minimize_scalar(
        _z_distance, bounds=(lo, hi), method="bounded",
        args=(sol, Z0), options={"xatol": 1e-14}
    )

    return t_eval, d_vals, res.fun, res.x, sol


# ---------------------------------------------------------------------------
# Energy constraint
# ---------------------------------------------------------------------------

def is_negative_energy(a, c, L, b=1.0):
    """Check if the initial conditions yield negative total energy."""
    d = (L - a * c) / b
    KE = 0.5 * (c**2 + d**2)
    sqrt2 = np.sqrt(2)
    sqrt3 = np.sqrt(3)
    r12 = sqrt2 * abs(a)
    r13 = abs(a + sqrt3 * b) / sqrt2
    r23 = abs(-a + sqrt3 * b) / sqrt2
    if r12 < 1e-12 or r13 < 1e-12 or r23 < 1e-12:
        return False
    PE = -(1.0 / r12 + 1.0 / r13 + 1.0 / r23)
    return (KE + PE) < 0


# ---------------------------------------------------------------------------
# Parameter space scanning
# ---------------------------------------------------------------------------

def scan_rpf(L, a_range, c_range, T_max=8.0, n_grid=80, n_time_samples=800,
             save_path=None, verbose=True):
    """Scan the (a, c) plane at fixed L, computing the return proximity function.

    Returns (a_vals, c_vals, rpf_map) where rpf_map contains -log10(d).

    If save_path is given, saves results to an .npz file after completion.
    """
    a_vals = np.linspace(*a_range, n_grid)
    c_vals = np.linspace(*c_range, n_grid)
    rpf_map = np.full((n_grid, n_grid), np.nan)

    total = n_grid * n_grid
    for i, a in enumerate(a_vals):
        if verbose:
            done = i * n_grid
            print(f"\r  Scanning: row {i+1}/{n_grid} ({100*done/total:.0f}%)", end="", flush=True)
        for j, c in enumerate(c_vals):
            if a <= 0.001:
                continue
            if not is_negative_energy(a, c, L):
                continue
            try:
                state0 = initial_conditions_from_params(a, c, L)
                d_min, _, _ = return_proximity(state0, T_max,
                                               t_min_frac=0.15,
                                               n_samples=n_time_samples)
                rpf_map[j, i] = -np.log10(max(d_min, 1e-15))
            except (RuntimeError, FloatingPointError):
                continue

    if verbose:
        print(f"\r  Scanning: done ({n_grid}x{n_grid} grid)        ")

    if save_path is not None:
        np.savez(save_path, a_vals=a_vals, c_vals=c_vals, rpf_map=rpf_map,
                 L=L, a_range=a_range, c_range=c_range, T_max=T_max)
        if verbose:
            print(f"  Saved to {save_path}")

    return a_vals, c_vals, rpf_map


def load_scan(path):
    """Load a saved scan result from an .npz file.

    Returns (a_vals, c_vals, rpf_map, L).
    """
    data = np.load(path)
    return data["a_vals"], data["c_vals"], data["rpf_map"], float(data["L"])


# ---------------------------------------------------------------------------
# Candidate extraction and refinement
# ---------------------------------------------------------------------------

def find_candidates(a_vals, c_vals, rpf_map, threshold=3.0, min_separation=0.02):
    """Extract candidate orbit locations from a r.p.f. scan map.

    Finds local maxima of -log10(d) above threshold.

    Returns list of (a, c, rpf_value) tuples.
    """
    # Replace NaN with 0 for filtering
    data = np.nan_to_num(rpf_map, nan=0.0)

    # Find local maxima using maximum_filter
    local_max = maximum_filter(data, size=5)
    peaks = (data == local_max) & (data > threshold)

    candidates = []
    peak_indices = np.argwhere(peaks)
    for jj, ii in peak_indices:
        a_val = a_vals[ii]
        c_val = c_vals[jj]
        rpf_val = rpf_map[jj, ii]
        candidates.append((a_val, c_val, rpf_val))

    # Sort by rpf_value (best first)
    candidates.sort(key=lambda x: -x[2])

    # Merge nearby candidates
    merged = []
    for a, c, val in candidates:
        too_close = False
        for a2, c2, _ in merged:
            if abs(a - a2) < min_separation and abs(c - c2) < min_separation:
                too_close = True
                break
        if not too_close:
            merged.append((a, c, val))

    return merged


def refine_orbit(a0, c0, L, T_max=8.0, maxiter=100):
    """Refine orbit parameters using Nelder-Mead minimization.

    maxiter caps the number of function evaluations to prevent runaway runtime.
    """
    def objective(params):
        a, c = params
        if a <= 0.001:
            return 100.0
        if not is_negative_energy(a, c, L):
            return 100.0
        try:
            state0 = initial_conditions_from_params(a, c, L)
            d_min, _, _ = return_proximity(state0, T_max, n_samples=1000)
            return np.log10(max(d_min, 1e-15))
        except (RuntimeError, FloatingPointError):
            return 100.0

    result = minimize(
        objective, [a0, c0],
        method="Nelder-Mead",
        options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": maxiter,
                 "maxfev": maxiter}
    )
    return result


def refine_all_candidates(candidates, L, T_max=8.0, verbose=True):
    """Refine a list of candidates and return sorted results.

    Returns list of dicts with keys: a, c, L, d_min, T, E.
    """
    import time as _time
    results = []
    total = len(candidates)
    for idx, (a0, c0, _) in enumerate(candidates, 1):
        if verbose:
            print(f"  Refining candidate {idx}/{total}: a={a0:.4f}, c={c0:.4f} ...",
                  end="", flush=True)
        t0 = _time.time()
        try:
            res = refine_orbit(a0, c0, L, T_max)
            a_opt, c_opt = res.x
            state0 = initial_conditions_from_params(a_opt, c_opt, L)
            d_min, t_rec, _ = return_proximity(state0, T_max, n_samples=2000)
            E = compute_energy(state0)
            elapsed = _time.time() - t0
            if verbose:
                print(f" d_min={d_min:.2e}, T={t_rec:.4f} ({elapsed:.0f}s)")
            results.append({
                "a": a_opt, "c": c_opt, "L": L,
                "d_min": d_min, "T": t_rec, "E": E,
            })
        except Exception as ex:
            elapsed = _time.time() - t0
            if verbose:
                print(f" FAILED: {ex} ({elapsed:.0f}s)")
            continue

    results.sort(key=lambda x: x["d_min"])
    return results


# ---------------------------------------------------------------------------
# Šuvakov gradient descent refinement (5x5 zoom method)
# ---------------------------------------------------------------------------

def refine_orbit_gradient(vx0, vy0, T_max=100.0, dp=1e-3, n_steps=30, verbose=False):
    """Refine orbit using the Šuvakov 5x5 grid descent method.

    Works in the (vx, vy) search plane with fixed positions:
    r1=(-1,0), r2=(1,0), r3=(0,0), zero angular momentum.

    At each step, evaluates a 5x5 grid centred on (vx0, vy0) with spacing dp.
    If the minimum is at the centre, halves dp (zoom in).
    Otherwise, moves the centre to the minimum.

    Returns dict with keys: vx, vy, d_min, T, E, n_steps_used.
    """
    vx, vy = vx0, vy0
    best_t = 0.0

    for step in range(n_steps):
        best_d = np.inf
        best_i, best_j = 0, 0

        for i in range(-2, 3):
            for j in range(-2, 3):
                vx_t = vx + i * dp
                vy_t = vy + j * dp
                try:
                    state0 = build_state_symmetric(vx_t, vy_t)
                    d_min, t_rec, _ = return_proximity(state0, T_max,
                                                       t_min_frac=0.05,
                                                       n_samples=1500)
                    if d_min < best_d:
                        best_d = d_min
                        best_i, best_j = i, j
                        best_t = t_rec
                except (RuntimeError, FloatingPointError):
                    continue

        if best_i == 0 and best_j == 0:
            dp /= 2.0
            if verbose:
                print(f"  Step {step+1}: zoom in, dp={dp:.2e}, d_min={best_d:.3e}")
        else:
            vx += best_i * dp
            vy += best_j * dp
            if verbose:
                print(f"  Step {step+1}: move to vx={vx:.10f}, vy={vy:.10f}, d_min={best_d:.3e}")

        if best_d < 1e-12:
            break

    state0 = build_state_symmetric(vx, vy)
    d_min, t_rec, _ = return_proximity(state0, T_max, n_samples=3000)
    E = compute_energy(state0)

    return {"vx": vx, "vy": vy, "d_min": d_min, "T": t_rec, "E": E,
            "n_steps_used": step + 1}


def build_state_symmetric(vx, vy):
    """Build state vector for the symmetric initial conditions:
    r1=(-1,0), r2=(1,0), r3=(0,0)
    v1=(vx,vy), v2=(vx,vy), v3=(-2vx,-2vy)
    """
    return np.array([
        -1.0, 0.0,   # r1
         1.0, 0.0,   # r2
         0.0, 0.0,   # r3
         vx, vy,     # v1
         vx, vy,     # v2
        -2*vx, -2*vy # v3
    ])


# ---------------------------------------------------------------------------
# Free group word reading algorithm (Šuvakov & Dmitrašinović 2014, Appendix)
# ---------------------------------------------------------------------------

def read_free_group_word(sol, T_period, n_points=50000):
    """Read the free group element (word) from a periodic orbit.

    Implements the algorithm from Šuvakov & Dmitrašinović (2014) Appendix:
    1. Track equator crossings on the shape sphere (z sign changes)
    2. Interpolate to exact crossing time to determine which body is in the middle
    3. Convert syzygy sequence -> directed semi-circles -> free group letters

    Returns the free group word as a string (e.g., 'abAB' for figure-eight).
    """
    t_eval = np.linspace(0, T_period, n_points)

    def _get_z(t):
        st = sol.sol(t)
        r = st[:6].reshape(3, 2)
        rho = (r[0] - r[1]) / np.sqrt(2)
        lam = (r[0] + r[1] - 2 * r[2]) / np.sqrt(6)
        R2 = rho @ rho + lam @ lam
        return 2 * (rho[0] * lam[1] - rho[1] * lam[0]) / R2

    def _get_middle_body(t):
        """Determine which body is in the middle at an equator crossing.

        At z=0 on the shape sphere, the three bodies are collinear.
        Projects body positions onto the line direction and returns the
        body with the median coordinate (1-indexed). This is robust
        regardless of the collinear line's orientation in the plane.
        """
        st = sol.sol(t)
        r = st[:6].reshape(3, 2)
        # Line direction: use the two most separated bodies
        d01 = np.linalg.norm(r[0] - r[1])
        d02 = np.linalg.norm(r[0] - r[2])
        d12 = np.linalg.norm(r[1] - r[2])
        # Pick the pair with largest separation as line direction
        if d01 >= d02 and d01 >= d12:
            direction = r[1] - r[0]
        elif d02 >= d12:
            direction = r[2] - r[0]
        else:
            direction = r[2] - r[1]
        direction = direction / np.linalg.norm(direction)
        # Project all three bodies onto the line
        projs = np.array([r[i] @ direction for i in range(3)])
        return int(np.argsort(projs)[1]) + 1

    def _find_crossing(t_lo, t_hi, z_lo, z_hi):
        """Bisect to find the exact equator crossing time."""
        for _ in range(50):
            t_mid = (t_lo + t_hi) / 2
            z_mid = _get_z(t_mid)
            if z_mid == 0 or (t_hi - t_lo) < 1e-14:
                return t_mid
            if z_lo * z_mid < 0:
                t_hi, z_hi = t_mid, z_mid
            else:
                t_lo, z_lo = t_mid, z_mid
        return (t_lo + t_hi) / 2

    # Step 1: Find all equator crossings with bisection refinement
    syzygies = []
    z_prev = _get_z(t_eval[0])

    for idx in range(1, len(t_eval)):
        t = t_eval[idx]
        z = _get_z(t)
        if z_prev * z < 0:
            # Bisect to find exact crossing
            t_cross = _find_crossing(t_eval[idx - 1], t, z_prev, z)
            middle = _get_middle_body(t_cross)
            syzygies.append(middle)
        z_prev = z

    if len(syzygies) < 2:
        return "?"

    # The algorithm assumes the sequence starts at "segment 2" (body 2 in middle).
    # For symmetric i.c.s, the initial config has body 3 in the middle (segment 3).
    # Per footnote 46 of the paper, we must renumber: swap labels 2 <-> 3
    # so that the starting segment becomes "2" in the algorithm's convention.
    # Detect the initial middle body and apply renumbering if needed.
    st0 = sol.sol(0)
    r0 = st0[:6].reshape(3, 2)
    init_middle = int(np.argsort(r0[:, 0])[1]) + 1

    if init_middle == 3:
        # Swap 2 and 3 in all syzygies
        remap = {1: 1, 2: 3, 3: 2}
        syzygies = [remap[s] for s in syzygies]
    elif init_middle == 1:
        # Swap 1 and 2
        remap = {1: 2, 2: 1, 3: 3}
        syzygies = [remap[s] for s in syzygies]

    # Remove consecutive duplicate syzygies — these arise from numerical
    # near-misses where the orbit grazes the equator, producing two rapid
    # crossings with the same body in the middle. Such pairs aren't valid
    # transitions in Table III.
    filtered = [syzygies[0]]
    for s in syzygies[1:]:
        if s != filtered[-1]:
            filtered.append(s)
    syzygies = filtered

    # Prepend the starting segment (now guaranteed to be "2")
    syzygies.insert(0, 2)
    # Append closing segment
    if syzygies[-1] != 2:
        syzygies.append(2)

    # Step 2: Convert syzygy pairs to directed semi-circles (Table III)
    # Parity alternates: first transition is "odd" (upper semi-circles),
    # second is "even" (lower), etc.
    table_iii_odd = {
        (1, 2): "F", (1, 3): "FA", (2, 1): "B", (2, 3): "A",
        (3, 1): "EB", (3, 2): "E",
    }
    table_iii_even = {
        (1, 2): "D", (1, 3): "DG", (2, 1): "H", (2, 3): "G",
        (3, 1): "CH", (3, 2): "C",
    }

    semi_circles = []
    for i in range(len(syzygies) - 1):
        pair = (syzygies[i], syzygies[i + 1])
        # i=0 is the first transition (odd), i=1 is second (even), etc.
        if i % 2 == 0:  # odd crossing (1st, 3rd, 5th...)
            sc = table_iii_odd.get(pair, "?")
        else:  # even crossing (2nd, 4th, 6th...)
            sc = table_iii_even.get(pair, "?")
        semi_circles.append(sc)

    sc_str = "".join(semi_circles)

    # Step 3: Convert pairs of semi-circles to free group letters (Table IV)
    table_iv = {
        "AC": "a", "GE": "A", "BD": "b", "HF": "B",
    }

    word = []
    i = 0
    while i < len(sc_str) - 1:
        pair = sc_str[i:i+2]
        letter = table_iv.get(pair)
        if letter is not None:
            word.append(letter)
            i += 2
        else:
            word.append("?")
            i += 1
    # Handle trailing single character
    if i < len(sc_str):
        word.append("?")

    return "".join(word)


# ---------------------------------------------------------------------------
# Symmetric initial condition orbits (Šuvakov / Li-Liao format)
# ---------------------------------------------------------------------------

# Table I from Šuvakov & Dmitrašinović (2014)
SUVAKOV_TABLE1 = [
    ("I.A.1 butterfly I",    0.306892758965492,  0.125506782829762,   6.23564136316479),
    ("I.A.2 butterfly II",   0.392955223941802,  0.0975792352080344,  7.00390738764014),
    ("I.A.3 bumblebee",      0.184278506469727,  0.587188195800781,  63.5345412733264),
    ("I.B.1 moth I",         0.464445237398184,  0.396059973403921,  14.8939113169584),
    ("I.B.2 moth II",        0.439165939331987,  0.452967645644678,  28.6702783225658),
    ("I.B.3 butterfly III",  0.405915588857606,  0.230163127422333,  13.8657626785699),
    ("I.B.4 moth III",       0.383443534851074,  0.377363693237305,  25.8406180475758),
    ("I.B.5 goggles",        0.0833000564575194, 0.127889282226563,  10.4668176954385),
    ("I.B.6 butterfly IV",   0.350112121391296,  0.0793394773483276, 79.4758748952101),
    ("I.B.7 dragonfly",      0.080584285736084,  0.588836087036132,  21.2709751966648),
    ("II.B.1 yarn",          0.559064247131347,  0.349191558837891,  55.5017624421301),
    ("II.C.2a yin-yang I",   0.513938054919243,  0.304736003875733,  17.328369755004),
    ("II.C.2b yin-yang I",   0.282698682308198,  0.327208786129952,  10.9625630756217),
    ("II.C.3a yin-yang II",  0.416822143554688,  0.330333312988282,  55.78982856891),
    ("II.C.3b yin-yang II",  0.417342877101898,  0.313100116109848,  54.2075992141846),
]

# Table II — figure-eight satellites
SUVAKOV_TABLE2 = [
    ("M8",  0.3471128135672417, 0.532726851767674,   6.3250, 1),
    ("S8",  0.3393928985595663, 0.536191205596924,   6.2917, 1),
    ("NC1", 0.2554309326049807, 0.516385834327506,  35.042,  7),
    ("NC2", 0.4103549868164067, 0.551985438720704,  57.544,  7),
    ("O1",  0.2034916865234370, 0.5181128588867190, 32.850,  7),
    ("O2",  0.4568108129224680, 0.5403305086130216, 64.834,  7),
    ("O3",  0.2022171409759519, 0.5311040339355467, 53.621, 11),
    ("O4",  0.2712627822083244, 0.5132559436920279, 55.915, 11),
    ("O5",  0.2300043496704103, 0.5323028446350102, 71.011, 14),
    ("O6",  0.2108318037109371, 0.5174100244140625, 80.323, 17),
    ("O7",  0.2132731670875545, 0.5165434524230961, 80.356, 17),
    ("O8",  0.2138543002929687, 0.5198665707397461, 81.217, 17),
    ("O9",  0.2193730914764402, 0.5177814195442197, 81.271, 17),
    ("O10", 0.2272123532714848, 0.5200484344272606, 82.671, 17),
    ("O11", 0.2199766127929685, 0.5234338500976567, 82.743, 17),
    ("O12", 0.2266987607727048, 0.5246235168190009, 83.786, 17),
    ("O13", 0.2686383642458915, 0.5227270888731481, 88.674, 17),
    ("O14", 0.2605047016601568, 0.5311685141601564, 89.941, 17),
    ("O15", 0.2899041109619139, 0.5226240653076171, 91.982, 17),
]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_orbit(sol, T_period, title="", n_points=5000):
    """Plot a three-body orbit in real space and on the shape sphere."""
    t_eval = np.linspace(0, T_period, n_points)
    states = np.array([sol.sol(t) for t in t_eval])

    r1 = states[:, 0:2]
    r2 = states[:, 2:4]
    r3 = states[:, 4:6]

    shape_pts = []
    for st in states:
        r = st[:6].reshape(3, 2)
        v = st[6:].reshape(3, 2)
        rho, lam, _, _ = to_jacobi(r, v)
        shape_pts.append(to_shape_sphere(rho, lam))
    shape_pts = np.array(shape_pts)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.plot(r1[:, 0], r1[:, 1], "r-", lw=0.5, label="Body 1")
    ax.plot(r2[:, 0], r2[:, 1], "g-", lw=0.5, label="Body 2")
    ax.plot(r3[:, 0], r3[:, 1], "b-", lw=0.5, label="Body 3")
    ax.plot(*r1[0], "ro", ms=5)
    ax.plot(*r2[0], "go", ms=5)
    ax.plot(*r3[0], "bo", ms=5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    ax.set_title("Real space")

    ax = axes[1]
    sp_x = shape_pts[:, 0] / (1 + shape_pts[:, 2] + 1e-15)
    sp_y = shape_pts[:, 1] / (1 + shape_pts[:, 2] + 1e-15)
    ax.plot(sp_x, sp_y, "k-", lw=0.4)
    for angle in [0, 2 * np.pi / 3, 4 * np.pi / 3]:
        cx, cy = np.cos(angle), np.sin(angle)
        ax.plot(cx / 2, cy / 2, "r*", ms=10)
    ax.set_xlabel("X (stereo)")
    ax.set_ylabel("Y (stereo)")
    ax.set_aspect("equal")
    ax.set_title("Shape sphere (stereographic)")

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.show()


def count_shape_sphere_loops(sol, T_period, n_points=10000):
    """Count loops on the shape sphere by counting equator crossings (z=0).

    Each complete loop around a pole crosses the equator twice.
    Returns (k_estimate, n_crossings).
    """
    t_eval = np.linspace(0, T_period, n_points)
    z_prev = None
    crossings = 0
    for t in t_eval:
        st = sol.sol(t)
        r = st[:6].reshape(3, 2)
        v = st[6:].reshape(3, 2)
        rho, lam, _, _ = to_jacobi(r, v)
        z = 2 * (rho[0] * lam[1] - rho[1] * lam[0]) / (rho @ rho + lam @ lam)
        if z_prev is not None and z_prev * z < 0:
            crossings += 1
        z_prev = z
    return crossings // 2, crossings


# ---------------------------------------------------------------------------
# Orbit data from the paper (Tables 3, 4, 5)
# ---------------------------------------------------------------------------

# CORRECTED from layout-preserved PDF extraction
# Columns: (Nr, L, a, c, T, k)
TABLE3_ORBITS = [
    ( 1, 0.7,          0.427052524289, -1.336907801590,  4.46125383,  3),
    ( 2, 0.85,         0.225423376709, -2.161172667330,  4.93326124,  3),
    ( 3, 0.65,         0.226748054608,  2.494853501883,  5.38634911,  3),
    ( 4, 0.85,         0.236597473885, -1.986209145030,  4.79387855,  4),
    ( 5, 0.7,          0.410445264670, -0.901755581763,  3.04224052,  4),
    ( 6, 0.65,         0.089473243424,  3.863464841380,  6.80175586,  4),
    ( 7, 0.65,         0.351891702719,  1.961519344674,  5.95736952,  4),
    ( 8, 0.8,          0.278773932080, -1.610811894950,  4.47639401,  5),
    ( 9, 0.65,         0.131937959644,  3.165458555994,  8.78754480,  5),
    (10, 0.8,          0.228399714670, -1.771422978980,  4.12780456,  6),
    (11, 0.9,          0.162112751455, -2.424555946410,  4.72091689,  6),
    (12, 0.65,         0.335276789538,  2.040219149776,  9.43988995,  6),
    (13, 0.8,          0.195389423297, -1.911145472440,  3.90877460,  7),
    (14, 0.85,         0.213186587101, -1.831539642360,  4.45339868,  7),
    (15, 0.9,          0.174854596011, -2.217240443720,  4.74829390,  7),
    (16, 0.9,          0.269439116363, -1.452760161970,  4.74215052,  7),
    (17, 0.8,          0.171971993864, -2.034163513840,  3.75598535,  8),
    (18, 0.935548917,  0.129471314426, -2.721144023250,  4.76584900,  8),
    (19, 0.7,          0.537026752182, -1.208756213130, 15.23729361,  8),
    (20, 0.85,         0.186299773074, -1.955937487530,  4.23559872,  8),
    (21, 0.9,          0.205445523859, -1.864664426590,  4.90547644,  8),
    (22, 0.935548917,  0.273518188668, -1.304218790410,  4.75908509,  8),
    (23, 0.8,          0.154250567982, -2.146266402400,  3.64212942,  9),
    (24, 0.935548917,  0.232402133831, -1.514749892810,  4.78046677,  9),
    (25, 0.85,         0.166323483739, -2.067914324720,  4.07742799,  9),
]

TABLE4_ORBITS = [
    (26, 0.9,          0.182069791972, -1.978004135720,  4.67040604,  9),
    (27, 0.935548917,  0.137285145946, -2.545654735050,  4.78370980,  9),
    (28, 1.0,          0.294303286736, -1.008322699800,  4.76801186,  9),
    (29, 1.0,          0.238218402625, -1.735467337160,  6.99274465,  9),
    (30, 0.85,         0.150808938415, -2.170190077810,  3.95629338, 10),
    (31, 1.0,          0.209243455808, -1.847936324290,  6.46969425, 10),
    (32, 1.0,          0.266925744815, -1.097311225480,  4.78597293, 10),
    (33, 0.7,          0.442402892100, -0.700265953090,  6.66684025, 10),
    (34, 0.8,          0.129160165070, -2.343250256240,  3.48168164, 11),
    (35, 1.0,          0.243617171755, -1.190238664360,  4.79985962, 11),
    (36, 0.8,          0.119882517495, -2.431498122390,  3.42239489, 12),
    (37, 0.935548917,  0.147716034231, -2.192473293620,  4.68169607, 12),
    (38, 1.0,          0.223340706817, -1.288029535680,  4.81093626, 12),
    (39, 0.8,          0.112051269923, -2.514553762900,  3.37232423, 13),
    (40, 0.935548917,  0.137088743692, -2.274985930540,  4.56392294, 13),
    (41, 1.0,          0.205347639306, -1.392000632830,  4.81998871, 13),
    (42, 0.8,          0.105352054293, -2.592836811550,  3.32933694, 14),
    (43, 1.0,          0.146284147722, -2.203021298400,  5.44909516, 14),
    (44, 0.85,         0.105854763654, -2.586702862280,  3.60999927, 15),
    (45, 1.0,          0.136829854037, -2.277089258430,  5.30678948, 15),
    (46, 1.0,          0.173982588477, -1.627461935580,  4.83391624, 15),
    (47, 0.85,         0.100315012262, -2.656853706660,  3.56736722, 16),
    (48, 0.9,          0.102148274757, -2.632996062570,  3.90923742, 17),
    (49, 0.935548917,  0.102773178918, -2.625007735230,  4.19114405, 18),
    (50, 1.0,          0.110009104810, -2.537603649510,  4.91429079, 19),
]

TABLE5_ORBITS = [
    (51, 1.03,  0.111843109779, -2.516815645433, 5.31721934, 20),
    (52, 1.03,  0.106999360702, -2.572864154118, 5.23675314, 21),
    (53, 1.03,  0.102629340641, -2.626824252565, 5.16461551, 22),
    (54, 1.07,  0.112532299117, -2.509130097108, 5.93058960, 22),
    (55, 1.03,  0.098663558847, -2.678884782752, 5.09949046, 23),
    (56, 1.07,  0.107950445783, -2.561552366744, 5.83679015, 23),
    (57, 1.03,  0.095045815614, -2.729207988143, 5.04033295, 24),
    (58, 1.07,  0.103792166702, -2.612125153377, 5.75238033, 24),
    (59, 1.07,  0.099998313989, -2.661009860717, 5.67592271, 25),
    (60, 1.07,  0.096520577947, -2.708345753387, 5.60626742, 26),
    (61, 1.07,  0.093319072567, -2.754253774713, 5.54248254, 27),
    (62, 1.07,  0.090360532835, -2.798839977984, 5.48380387, 28),
    (63, 1.07,  0.087616970029, -2.842197991476, 5.42959820, 29),
    (64, 1.03,  0.078546062730, -3.001449136918, 4.77275161, 30),
    (65, 1.07,  0.085064647578, -2.884410942736, 5.37933595, 30),
    (66, 1.03,  0.076424987284, -3.042734364817, 4.73851381, 31),
    (67, 1.07,  0.082683286816, -2.925553119986, 5.33257043, 31),
    (68, 1.03,  0.074436173120, -3.083042211579, 4.70642108, 32),
    (69, 1.07,  0.080455449983, -2.965691193228, 5.28892198, 32),
    (70, 1.03,  0.072567055606, -3.122428961620, 4.67626201, 33),
    (71, 1.07,  0.078366051119, -3.004885281315, 5.24806564, 33),
    (72, 1.03,  0.070806647363, -3.160945841855, 4.64785297, 34),
    (73, 1.07,  0.076401964229, -3.043189884203, 5.20972146, 34),
    (74, 1.03,  0.069145295338, -3.198639607087, 4.62103356, 35),
    (75, 1.07,  0.074551712461, -3.080654510153, 5.17364685, 35),
]

ALL_ORBITS = TABLE3_ORBITS + TABLE4_ORBITS + TABLE5_ORBITS


def plot_rpf_map(a_vals, c_vals, rpf_map, L, candidates=None):
    """Plot the return proximity function heatmap with optional candidate markers."""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(
        rpf_map, origin="lower", aspect="auto",
        extent=[a_vals[0], a_vals[-1], c_vals[0], c_vals[-1]],
        cmap="hot", vmin=0, vmax=12
    )
    if candidates:
        for a, c, val in candidates:
            ax.plot(a, c, "cs", ms=8, mew=1.5, mfc="none")
    ax.set_xlabel("a", fontsize=12)
    ax.set_ylabel("c", fontsize=12)
    ax.set_title(f"$-\\log_{{10}}(d)$ at $L = {L}$", fontsize=13)
    plt.colorbar(im, ax=ax, label="$-\\log_{10}(d)$")
    plt.tight_layout()
    plt.show()
