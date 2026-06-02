di # Method: Periodic Orbit Detection and Classification in the Planar Equal-Mass Three-Body Problem

## 1. Problem Statement

We seek periodic orbits of three equal-mass bodies (m₁ = m₂ = m₃ = 1) interacting gravitationally (G = 1) in a plane. The state vector has 12 components: positions (x₁, y₁, x₂, y₂, x₃, y₃) and velocities (vₓ₁, vᵧ₁, vₓ₂, vᵧ₂, vₓ₃, vᵧ₃). The acceleration on body i is:

    aᵢ = −Σⱼ≠ᵢ (rᵢ − rⱼ) / |rᵢ − rⱼ|³

This is a 6-body second-order ODE system (12 first-order). The full initial condition space is 12-dimensional, reduced to 6 after fixing centre of mass, total momentum, and energy.

---

## 2. Integration

We use **DOP853** (8th-order Runge-Kutta-Fehlberg) via SciPy's `solve_ivp`, not symplectic methods. Three reasons:

1. **Per-step accuracy**: We need the return proximity d_min ∼ 10⁻⁷ after a single period. DOP853 achieves O(dt⁸) local error vs O(dt²) for velocity Verlet, requiring far fewer steps.
2. **Adaptive stepping**: Three-body orbits include close encounters where forces change rapidly. Fixed-step methods either waste time in slow phases or lose accuracy in fast ones.
3. **Dense output**: The RPF refinement evaluates the solution at arbitrary times between steps. DOP853 provides this via polynomial interpolation at no extra cost.

**Default tolerances**: rtol = 10⁻¹², atol = 10⁻¹⁴, max_step = 0.01.

The RHS function is compiled with **Numba `@njit`** for approximately 3× speedup over vectorised NumPy. A pure-NumPy fallback is included if Numba is unavailable.

---

## 3. Parametrisations of Initial Conditions

The full 6D search space (after constraints) is impractical to scan exhaustively. We use two 2D slices:

### 3.1 Symmetric (Suvakov)
Fixed positions r₁ = (−1, 0), r₂ = (1, 0), r₃ = (0, 0). Velocities v₁ = v₂ = (vₓ, vᵧ), v₃ = (−2vₓ, −2vᵧ). Angular momentum L = 0 by construction. Search space: (vₓ, vᵧ).

This is the parametrisation used by Suvakov & Dmitrasinovic (2014) and Li & Liao (2017).

### 3.2 BHH (Jankovic)
Collinear Jacobi coordinates: ρ = (a, 0), λ = (b, 0), ρ̇ = (0, c), λ̇ = (0, d) where d = (L − ac)/b and b = 1. Jacobi coordinates are defined as:

    ρ = (r₁ − r₂) / √2
    λ = (r₁ + r₂ − 2r₃) / √6

Search space: (a, c) at fixed angular momentum L. This accesses orbits with L ≠ 0, which Li & Liao did not search.

Both parametrisations implement picklable builder classes for multiprocessing compatibility.

---

## 4. Return Proximity Function (RPF)

The RPF measures how close an orbit comes to repeating after integration over [0, T_max].

### 4.1 Rotation-invariant distance

To handle orbits that return to the same shape but at a different orientation (L ≠ 0), we work in **shape-sphere phase space**. The shape sphere coordinates are:

    R² = ρ·ρ + λ·λ
    x = 2(ρ·λ) / R²
    y = (λ·λ − ρ·ρ) / R²
    z = 2(ρ₀λ₁ − ρ₁λ₀) / R²

These encode the triangle shape independent of size, position, and orientation. The Z-vector Z = (x, y, z, ẋ, ẏ, ż) is 6-dimensional and rotation-invariant. The RPF is d(t) = ‖Z(t) − Z(0)‖₂.

### 4.2 Two-stage algorithm

**Stage 1 — Coarse sampling**: Evaluate d(t) at n_samples uniformly spaced points over [t_min_frac × T_max, T_max]. Default n_samples = 800 for scanning, 5000 for single-orbit analysis. The first 15% of the interval is skipped (t_min_frac = 0.15) to avoid trivial near-matches near t = 0.

Find all local minima: points where d(tᵢ) < d(tᵢ₋₁) and d(tᵢ) < d(tᵢ₊₁).

**Stage 2 — Brent refinement**: For each local minimum at index i, bracket with [t_{i−1}, t_{i+1}] and refine using `minimize_scalar` (bounded Brent method, xatol = 10⁻¹⁴).

Return (d_min, t_min, sol) — the global minimum over all refined local minima. An orbit is considered periodic if d_min < 10⁻⁴.

---

## 5. Parallel Grid Scanning

The scanner evaluates the RPF at every point in a 2D parameter grid using `multiprocessing.Pool` with `imap_unordered`.

### 5.1 Architecture
- Each grid point is an independent RPF evaluation (embarrassingly parallel)
- `imap_unordered` processes results as workers finish, achieving full saturation despite variable evaluation times (near-collisions take longer)
- Points that fail integration (near-collisions → RuntimeError) return NaN silently
- The RPF heatmap stores −log₁₀(d_min) at each grid point

### 5.2 Crash recovery
- Results are saved to `.npz` every 10 completed rows
- On resume, completed rows are loaded and skipped
- This makes the scanner robust to crashes during multi-hour scans

### 5.3 Default parameters
- Grid: 200×200 (adjustable via CLI)
- T_max: 8.0 (symmetric), 16.0 (BHH — known orbits have periods up to 15.2)
- n_samples: 800

### 5.4 Adaptive refinement
After a coarse scan, `adaptive_scan` identifies peaks above threshold, defines a zoomed region around each (±margin coarse cells), and re-scans at higher resolution (default 200×200 per region). This is a zoom tool for better Newton starting guesses, not a replacement for the full grid.

---

## 6. Candidate Extraction

Peaks in the RPF heatmap are extracted using `scipy.ndimage.maximum_filter` with a 5×5 kernel. Points where −log₁₀(d_min) exceeds the threshold (default 3.5, i.e. d_min < 3×10⁻⁴) and equal their local maximum are flagged as candidates.

Nearby candidates (within min_separation = 0.02 in parameter space) are merged, keeping the one with highest RPF value.

---

## 7. Newton-Raphson Refinement

Candidates from the scan have approximate parameters. Newton-Raphson refines them to machine precision.

### 7.1 Periodicity residual
- For L = 0 (symmetric): F = state(T) − state(0) (12 components)
- For L ≠ 0 (BHH): F = Z(T) − Z(0) (6 components, rotation-invariant)

The unknowns are (params, T) — for symmetric orbits, (vₓ, vᵧ, T); for BHH, (a, c, T).

### 7.2 Monodromy-based Jacobian

The monodromy matrix M (Section 8) gives ∂state(T)/∂state(0). By the chain rule:

    ∂F/∂params = (M − I) · ∂state₀/∂params
    ∂F/∂T = f(state(T))

where f is the RHS of the EOM. The ∂state₀/∂params term is computed via finite differences on the cheap `param_to_state` function (no integration needed).

This requires only **one variational integration per Newton step** (to get both M and the residual F), compared to 2(n+1) plain integrations for finite-difference Jacobians.

### 7.3 Convergence
- Solve the overdetermined system J·δ = −F via least squares (`numpy.linalg.lstsq`)
- Convergence: ‖F‖ < 10⁻¹⁰ (default)
- Stagnation detection: if the last 4 residuals plateau (max/min < 3.0) and ‖F‖ < 10⁻⁵, declare convergence at the precision floor. This handles long-period orbits where the monodromy Jacobian is noisy.
- Maximum 20 iterations (default)

Typical convergence: quadratic (2e-2 → 1e-4 → 7e-6 → 3e-8 → 2e-13 in 4 iterations for the figure-eight).

---

## 8. Floquet Stability Analysis

### 8.1 Variational equations

The state transition matrix (STM) Φ(t) satisfies dΦ/dt = J(t)·Φ where J is the 12×12 Jacobian of the EOM:

    J = [[0₆, I₆], [G, 0₆]]

where G is the 6×6 gravitational Jacobian (tidal tensor):

    ∂aᵢ/∂rⱼ = I/|rᵢⱼ|³ − 3(rᵢⱼ ⊗ rᵢⱼ)/|rᵢⱼ|⁵    (i ≠ j)
    ∂aᵢ/∂rᵢ = −Σⱼ≠ᵢ ∂aᵢ/∂rⱼ

The extended state vector has 12 + 144 = 156 components (orbit + flattened STM), integrated simultaneously with DOP853.

### 8.2 Monodromy matrix

The monodromy matrix M = Φ(T) evaluated after one period. Its eigenvalues λᵢ are the Floquet multipliers.

### 8.3 Segmented integration for long/unstable orbits

For orbits with T > 60 or extreme instability (λ_max ≫ 1), the STM columns collapse onto the dominant eigenvector, losing information about stable directions. The fix: split [0, T] into N segments, integrate each independently (resetting Φ to I), and multiply the segment STMs:

    M = M_N · M_{N−1} · ... · M₁

Each segment has condition number λ_max^(T/N), manageable in float64 if T/N is small enough. The code tries 4 candidate segment counts (⌈T/8⌉ ± 1, ⌈T/8⌉ ± 2) and picks the one with det(M) closest to 1.

### 8.4 Stability classification

- **Stable**: all |λᵢ| ≈ 1 (perturbations neither grow nor shrink)
- **Unstable**: at least one |λᵢ| > 1 + tol (perturbations grow exponentially)
- **Validation**: det(M) ≈ 1 (symplecticity), at least 2 unit eigenvalues (energy conservation + time-translation symmetry)

---

## 9. Free Group Word Reading

Periodic orbits are classified topologically by their free group word — a string of letters {a, A, b, B} encoding how the orbit winds around the collision points on the shape sphere.

### 9.1 Equator crossing detection

At z = 0 on the shape sphere, the three bodies are collinear (syzygy). Track z(t) along the orbit and detect sign changes. Each crossing is refined to 10⁻¹⁴ precision via 50 bisection iterations.

### 9.2 Middle body identification

At each equator crossing, the three collinear bodies have a well-defined ordering. The **middle body** is found by:
1. Computing the line direction (vector between the two most separated bodies)
2. Projecting all three bodies onto this line
3. Taking the body with the median projection

This is robust to arbitrary line orientations (unlike x-coordinate sorting, which fails for non-horizontal configurations).

### 9.3 Syzygy-to-word conversion

The syzygy sequence (which body is in the middle at each crossing) is converted to a free group word via lookup tables from Suvakov & Dmitrasinovic (2014):

- **Table III**: Pairs of consecutive syzygies → directed semi-circles (letters A-H), with odd/even parity alternation
- **Table IV**: Pairs of semi-circles → free group generators (a, A, b, B)

Consecutive duplicate syzygies (from numerical noise at near-grazing crossings) are filtered before conversion.

### 9.4 Canonical form

Cyclic permutations of a word represent the same orbit (different starting point). The **canonical cyclic form** is the lexicographically smallest rotation: e.g. `bABa` and `ABab` both canonicalise to `ABab`.

### 9.5 Period multiple detection

If word_candidate = word_known repeated k times (up to cyclic permutation), the candidate is a period-k multiple of the known orbit. Detected by checking all cyclic rotations of word_known^k against the canonical form of word_candidate.

---

## 10. Cross-Reference Against Known Catalogues

### 10.1 Known orbit databases

| Source | Count | Type | Parameters |
|--------|-------|------|------------|
| Figure-eight (Moore 1993) | 1 | Symmetric, L=0 | vₓ, vᵧ, T |
| Suvakov Table 1 (2014) | 15 | Symmetric, L=0 | Named families |
| Suvakov Table 2 (2014) | 19 | Symmetric, L=0 | Figure-eight satellites |
| Jankovic Tables 3-5 (2020) | 75 | BHH, L≠0 | a, c, L, T, k |
| Li & Liao (2017) | 695 | Symmetric, L=0 | vₓ, vᵧ, T, T*, Lf |

Total: 805 known orbits with pre-computed free group words cached to disk.

### 10.2 Matching algorithm

For each candidate:
1. Compute its canonical free group word
2. Compare against all known orbits' canonical words:
   - **Exact match**: same canonical word
   - **Period multiple**: candidate word = known word repeated k times
   - **Sub-period**: known word = candidate word repeated k times
3. Confirm match via period ratio (T_candidate / T_known ≈ k ± 0.5)

### 10.3 Energy-normalised deduplication

Two candidates with the same free group word at different energies are the same orbit family (related by the Newtonian scaling symmetry). Both are rescaled to E = −0.5 and compared by normalised period. If |T_norm_new − T_norm_prev| / max(T_norm) < 10⁻³, the duplicate is skipped.

---

## 11. Topological Kepler's Third Law

The scale-invariant period T* = T|E|^{3/2} / L_f (where L_f is the free group word length) is approximately constant across orbit families. The theoretical prediction is T* ≈ 2.433.

### Findings

- **Li & Liao 695 families** (mixed-letter words): T* = 2.432 ± 0.075. Converges from ~2.23 at short words to ~2.49 at long words. The law holds well.
- **Jankovic BHH families** (single-letter words a^k, b^k): T* decreases monotonically from ~3.3 at k=3 to ~1.9 at k=35. The law breaks down for pure single-letter words.

This suggests the topological Kepler's third law has a narrower domain of validity than previously reported — it applies to mixed-letter (topologically complex) words but not to single-letter (topologically simple) ones.

---

## 12. Stability-Topology Correlation

From the Floquet catalogue of 110 known orbits:

- **Pure-letter words** (a^k, b^k): 0/75 stable (always unstable)
- **Mixed-letter words**: 9/35 stable (26% stability rate)
- All 9 stable orbits are L = 0, figure-eight family members
- Instability decreases with word length: log₁₀(λ_max) drops from ~3 at L_f = 4 to ~0.9 at L_f = 35
- Most orbits have 2 unstable directions (63/110); only 2 have 4

---

## 13. Output Format

### Floquet catalogue (floquet_catalogue.json)
Per-orbit JSON with: name, parametrisation, L, refined parameters, T, E, free group word, is_stable, max_instability, determinant, n_unit_eigenvalues, monodromy_valid, multiplier magnitudes.

### Pipeline candidates (candidates.json)
Per-candidate JSON with: id, parametrisation, raw/refined parameters, L, T, d_min, E, converged, word, stability info, cross-reference match, is_new flag.

### Orbit plots (candidates_plots/)
Per-orbit PNG with real-space trajectories (3 bodies, colour-coded) and shape sphere stereographic projection (with collision point markers).

---

## References

1. Suvakov, M. & Dmitrasinovic, V. (2014). *Am. J. Phys.* 82(6):609-619. [Hunting method, free group classification]
2. Jankovic, M., Dmitrasinovic, V. & Suvakov, M. (2020). *Comp. Phys. Comm.* 250:107052. [BHH orbits, L≠0]
3. Li, X. & Liao, S. (2017). *Sci. China Phys. Mech. Astron.* 60:129511. [695 families, arXiv:1705.00527]
4. Montgomery, R. (1998). *Nonlinearity* 11(2):363-376. [Free group / braid group classification]
5. Hairer, E., Norsett, S.P. & Wanner, G. (1993). *Solving Ordinary Differential Equations I*. [DOP853 integrator]
