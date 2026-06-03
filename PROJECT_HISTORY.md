will the a# Three-Body Orbit Hunter: Project History & Decisions

## What this project does

Finds and classifies periodic orbits of the planar equal-mass three-body problem (m1 = m2 = m3 = 1, G = 1). The pipeline scans parameter space for candidate periodic orbits, refines them to machine precision with Newton-Raphson, classifies their stability via Floquet analysis, reads their topological type (free group word), and cross-references against known catalogues.

The codebase is ~2,800 lines across 7 Python files plus an analysis script.

---

## Phase 1: Parallel Scanner

### What was built
- `scanner.py`: Parametrisation-agnostic parallel RPF scanner using `multiprocessing.Pool` with `imap_unordered`. Incremental per-row saving to `.npz` for crash recovery. Automatic resume from partial results.
- `parametrisations.py`: Two picklable builder classes — `SymmetricBuilder` (vx, vy at L=0) and `BHHBuilder` (a, c at fixed L).
- `main.py`: CLI entry point with `validate`, `scan-symmetric`, `scan-bhh` commands.

### Key decisions
- **`imap_unordered` over `map`**: Workers finish grid points in arbitrary order due to varying integration times (near-collisions take longer). `imap_unordered` processes results as they arrive rather than waiting for row order, achieving full worker saturation.
- **Per-row saving**: Results are checkpointed to disk every 10 completed rows. This means a crash after 8 hours of a 12-hour scan only loses the current incomplete rows, not everything.
- **Picklable builders**: The `state_builder` callable crosses process boundaries via pickle. This ruled out lambda functions and required dedicated builder classes.

### Performance
- ~4x speedup on 8-core M2 (limited by load imbalance in RPF evaluation times, not multiprocessing overhead).
- Scales linearly with more cores, no code changes needed.

---

## Decision: Stay in Python (no Rust/C++ rewrite)

### Context
External advice suggested rewriting in Rust for performance, leveraging SIMD intrinsics (AVX256 on Intel, NEON on ARM).

### Decision
Rejected. The bottleneck is `solve_ivp` with DOP853, which is compiled Fortran (DOPRI8) under the hood. Python is only glue code. With N=3 bodies, there are exactly 3 pairwise interactions per RHS evaluation — far too few for SIMD to help. The parallelism is at the *scan* level (many independent integrations), which `multiprocessing` already exploits.

SIMD benchmarks showing Rust/C++ winning on N-body are for large-N simulations (thousands of particles), not N=3.

### What would actually help
- Numba JIT for the RHS function (a prior version existed, was removed — can be reintroduced).
- More cores (cloud instances for Phase 3 scans).
- Smarter searching (adaptive refinement instead of uniform grids).

---

## Phase 2: Floquet Stability Analysis

### What was built
- `floquet.py`: Tidal tensor (6x6 gravity Jacobian), 156-component variational EOM, monodromy matrix computation, Floquet multiplier extraction, stability classification, and Newton-Raphson refinement.

### Key decisions

#### Tidal tensor implementation
Used explicit loops over the 3 body pairs rather than vectorized NumPy. For N=3, the 6x6 matrix is filled in 6 iterations — vectorization would add complexity for zero performance gain. Validated against central finite differences to ~1e-9 agreement.

#### Dense output for monodromy endpoint
The original `compute_monodromy` used `sol.y[:, -1]` to get the final state — but with adaptive stepping, the last stored point may not land exactly at `T_period`. Changed to `dense_output=True` + `sol.sol(T_period)` for exact endpoint evaluation.

#### Monodromy-based Newton-Raphson Jacobian
The periodicity residual is `F = state(T) - state(0)`, so its Jacobian w.r.t. state is `M - I` where M is the monodromy matrix. By the chain rule, `dF/d(params) = (M - I) @ d(state0)/d(params)`, where the latter is a cheap finite-difference on the `param_to_state` function (no integration). The partial w.r.t. T is just `f(state(T))` — the EOM evaluated at the endpoint.

This reduces cost per Newton iteration from 6 integrations (finite-diff on 2 params + T, central differences) to 1 variational integration. Both methods achieve the same quadratic convergence rate.

#### Stagnation detection
For long-period orbits, the monodromy Jacobian accumulates enough numerical error that Newton oscillates at a residual floor (~1e-5 to 1e-6) instead of converging to machine precision. The stagnation detector declares convergence if the residual has plateaued below 1e-5 for 4 iterations with less than 3x variation.

### Validation
- **Figure-eight**: All 12 Floquet multipliers on the unit circle (linearly stable). det(M) = 1.000000. The figure-eight has maximal symmetry — 12 unit eigenvalues instead of the minimum 2.
- **Butterfly I**: Unstable with lambda_max = 1.78. Reciprocal pair structure confirmed (lambda * 1/lambda = 1).
- **BHH orbit #1**: Strongly unstable with lambda_max = 33.4. det(M) = 1.000000.

### Issues encountered

#### Butterfly IV (T = 79.5, lambda_max ~ 170)
The most challenging orbit in the catalogue. With T*log10(lambda_max) ~ 178, the variational equations' dynamic range exceeds float64's ~16 digits of precision.

**Attempted fixes:**
1. **QR reorthogonalisation**: Split integration into segments, decompose Phi = Q*R at each boundary, accumulate R factors. Failed because R-factor multiplication has the same exponential growth problem as the raw STM.
2. **Tighter max_step**: Reducing from 0.01 to 0.001 (100,000 steps) improved det from -5941 to ~1.5 but still far from 1.0.
3. **Segment-wise STM multiplication**: Split into N short segments, compute well-conditioned M_i for each, multiply M = M_N * ... * M_1. This works — with 11 segments, det = 0.997 (within 0.3% of 1.0). The key insight: each segment has condition number lambda_max^(T/N) ~ 170^7 ~ 10^16, which is at the edge of float64 but manageable.

**Auto-segment selection**: For orbits with T > 60, the code tries 4 candidate segment counts and picks the one whose monodromy determinant is closest to 1. This adds significant compute time (~3 min per candidate) but ensures the best numerical conditioning.

**Remaining limitation**: The auto-segmentation inside Newton-Raphson makes butterfly IV's Newton iterations extremely slow (~20+ min total). A future optimization would use `n_segments=1` for the Newton Jacobian (which only needs a reasonable search direction) and auto-segmentation only for the final `analyse_orbit` call.

#### Yin-yang II orbits (T ~ 55)
Long period with modest instability (lambda_max ~ 1.0-1.7). Newton stagnated at |F| ~ 1e-6 with the monodromy-based Jacobian. Fixed by using finite-difference Newton (`use_monodromy=False`) which is more robust for these orbits. Monodromy validation passes with det = 0.9998 and 1.0015.

---

## Phase 3a: Candidate Pipeline

### What was built
- `pipeline.py`: End-to-end processing: scan .npz -> extract peaks -> estimate period -> Newton-refine -> Floquet classify -> cross-reference -> JSON output.

### Key decisions

#### Cross-reference scope
Li & Liao's 695-family data was previously thought inaccessible. It turned out to be embedded in HTML tables at https://numericaltank.sjtu.edu.cn/three-body/three-body-movies.htm — not in downloadable files, which is why earlier attempts to find it failed. The data (v1, v2, T, T*, Lf for all 695 orbits) was extracted via `fetch_ll_data.py` and stored in `ll_orbits.json`. `ll_data.py` provides the API.

Cross-referencing now covers: figure-eight + 15 Suvakov named + 19 satellites + 75 Jankovic BHH + 695 Li & Liao symmetric families. The BHH parameter space (L != 0) remains the primary non-redundant search space.

#### Canonical cyclic word form
Free group words represent the same orbit regardless of starting point — `bABa` and `ABab` are cyclic permutations of the same word. The `canonical_word` function normalises to the lexicographically smallest rotation for comparison.

#### Period multiple detection
A word like `bABabABa` is `bABa` repeated twice — the same orbit at a higher period multiple. Detection checks if the candidate word equals any cyclic rotation of the known word repeated k times. Validated on all 19 Suvakov figure-eight satellites — k values match the published Table 2 exactly.

#### Energy-normalised deduplication
Under the Newtonian scaling symmetry (r -> alpha*r, v -> v/sqrt(alpha), T -> alpha^{3/2}*T), the same orbit family appears at different energies. Two candidates with the same free group word are rescaled to E = -0.5 and compared by normalised period. This correctly identifies, e.g., two butterfly I detections at different (vx, vy) as the same family.

#### Incremental checkpoint saving
The pipeline saves partial results every 5 candidates to `<output>.partial`. If a crash occurs at candidate 15 of 20, the first 10-15 results are preserved.

#### Persistent word cache
Computing free group words for known orbits requires integrating each one (~5s per orbit). The cache persists to `known_words_cache.json` so this only happens once. Contains 110 words (35 symmetric + 75 BHH).

### Issues encountered

#### Array indexing mismatch
`scanner.py` stores `rpf_map[row_idx, col_idx]` but `find_candidates` in `three_body.py` expects `rpf_map[col_idx, row_idx]` (the convention from the old serial `scan_rpf`). Fixed by transposing the rpf_map in `pipeline.py` when loading the new scanner format. The bug would have silently swapped parameter values in extracted candidates.

#### T_max too small for BHH
Default T_max was 8.0, but three known Jankovic orbits have periods up to 15.2. These would have been invisible to the scanner. Increased to 16.0 for BHH scans. The symmetric scan keeps T_max = 8.0 since all known symmetric orbits have T < 8.

#### Figure-eight missing from cross-reference
The figure-eight is the foundational orbit but wasn't in Suvakov's Tables 1 or 2 (Table 1 has named families, Table 2 has satellites). Added it explicitly to the known orbit list.

---

## Bug Fixes

### Blanket `except: continue`
The scanner and refinement code silently caught all exceptions, including genuine bugs like `TypeError` or `IndexError`. Replaced with specific `except (RuntimeError, FloatingPointError)` to catch only the expected failure modes (integration failure from near-collisions, floating-point overflow). Pipeline catches additionally include `np.linalg.LinAlgError` for Newton-Raphson singularities.

### Word-reading algorithm
The original `_get_middle_body` function sorted bodies by Cartesian x-coordinate to determine which was "in the middle" of a collinear configuration. This failed for orbits where the collinear line wasn't aligned with the x-axis, producing `?` characters in the free group word.

**First fix attempt (shape-sphere angle)**: Use theta = atan2(y, x) on the shape sphere equator to identify the nearest collision point. This failed because the collision point angles depend on the body labelling convention, and the mapping between angles and body identities was inconsistent.

**Final fix (line projection)**: At an equator crossing, project all three body positions onto the line connecting the two most separated bodies. The body with the median projection is the middle one. This is robust regardless of line orientation. All 110 orbits now produce clean words with zero `?` characters.

### Consecutive duplicate syzygies
When an orbit grazes the shape sphere equator without fully crossing, two rapid crossings register with the same body in the middle. These `(k, k)` pairs aren't valid transitions in the semi-circle lookup table (Table III from Suvakov 2014). Fixed by filtering consecutive duplicate syzygies before the conversion step.

### Local imports in floquet.py
Functions imported `from three_body import ...` inside function bodies to avoid hypothetical circular imports. There was no circular dependency — `floquet.py` imports from `three_body.py` but not the reverse. Moved all imports to module level.

### Stray indent in scanner.py
Line 1 had 4 leading spaces before the docstring, causing `IndentationError` on import.

### Multiprocessing spawn safety
Added `mp.set_start_method("spawn")` on Linux for container/AWS compatibility. macOS defaults to spawn already; fork can be unsafe in containers.

---

## Floquet Catalogue

### What it is
A JSON file (`floquet_catalogue.json`) containing Newton-refined parameters, Floquet multipliers, stability classification, free group word, and energy for 109 of the 110 known orbits (figure-eight + 15 Suvakov + 19 satellites + 75 Jankovic BHH). Butterfly IV is the one missing orbit (Newton convergence achieved but monodromy validation is marginal).

### Key findings

**Stability**: 9 orbits are linearly stable out of 109. All 75 BHH orbits are unstable. The stable orbits are all in the figure-eight topological family (the figure-eight itself, moths I and II, satellites M8, NC1, NC2, O13, O14, O15).

**Topological Kepler's third law**: The "universal" constant T* = T|E|^{3/2}/L_f shows systematic word-length dependence for single-letter BHH orbits. Symmetric mixed-letter orbits cluster at T* ~ 2.31. Short-word BHH orbits (k=3-6) have T* = 2.6-4.9. Long-word BHH orbits show T* monotonically decreasing from ~2.3 towards ~1.9. This suggests T* ~ 2.433 is an approximation that breaks down for pure `a^k` or `b^k` words.

**Instability spectrum**: The goggles orbit (I.B.5) is the most unstable symmetric orbit (lambda_max ~ 20.7). Among BHH orbits, Jankovic #6 (L=0.65, k=4) has lambda_max ~ 1085. Generally, instability decreases with word length k for the BHH single-letter families — longer orbits are less unstable per period.

---

## Pre-scan Preparation (for AWS)

### What was done
- `requirements.txt` created (numpy >= 1.20, scipy >= 1.7, matplotlib >= 3.4).
- `.gitignore` updated to exclude `.npz` files and `known_words_cache.json`.
- BHH pipeline tested end-to-end (50x50 scan at L=0.7, full pipeline to JSON).
- `n_samples=800` validated as sufficient at T_max=16 (d_min agrees within 2x across n=800, 1500, 3000, 5000 — well above the 1e-4 detection threshold).
- Known orbit words pre-computed for all 9 L values (110 words cached).

### Scan parameter recommendations
- **a range**: [0.05, 0.6] — covers all 75 known BHH orbits with margin.
- **c range**: [-3.5, 4.0] — comfortably encloses all known orbits.
- **L values**: 0.65, 0.7, 0.8, 0.85, 0.9, 0.935, 1.0, 1.03, 1.07 (the 9 Jankovic L values).
- **T_max**: 16.0 (covers known periods up to 15.2).
- **n_samples**: 800 (validated as sufficient).
- **Grid size**: 1000x1000 recommended.

### Time estimates (1000x1000 BHH scan)
- 8-core M2: ~200 hours per L value
- 64-core AWS: ~25 hours per L value
- 96-core AWS: ~17 hours per L value

---

## First BHH Scan Campaign (200×200, L=0.5, 1.0, 1.5)

### Scan results
- **L=0.5**: 46 min, 0 candidates above threshold. Either no orbits in this region or threshold too high.
- **L=1.0**: 35 min, 41 candidates. Only 1 of first 40 refined successfully (b^47, T=9.30). 97% failure rate suggests most peaks are noise — threshold=3.5 may be too permissive, letting through peaks that aren't real orbits. Candidate 41 pending (slow variational integration).
- **L=1.5**: ~40 min, 72 candidates. 7 refined successfully, all new (no Jankovic match). All pure b^k words with very long word lengths (57–196). L=1.5 is beyond Jankovic's range (max L=1.07), so all are genuinely new.
- Machine: Mac Mini, 10-core Apple Silicon

### All new orbits are pure b^k — topological bias in BHH parametrisation?
Every orbit found across all three L values is a pure `b^k` word: b^14 (L=0.7), b^47 (L=1.0), and b^57 to b^196 (L=1.5). No mixed words (`aAbB`, etc.), no `a`-type generators at all.

**Likely explanation: geometric bias of the BHH slice.** The BHH parametrisation fixes collinear Jacobi positions, and at a given L the `b`-type syzygies (one particular body passing between the other two) are the natural topology for those initial conditions. Mixed words require the bodies to swap roles mid-orbit, which may need initial conditions outside the BHH (a, c) slice entirely, or that occupy a much thinner region of the grid that even 200×200 can't resolve.

**Word length scales with L.** b^14 at L=0.7, b^47 at L=1.0, b^57–b^196 at L=1.5. This is consistent with the trend already visible in the Jankovic catalogue (pre-scan analysis item 4 found instability drops monotonically with L — longer, less unstable orbits at higher L).

**What the 500×500 Jankovic L-value scans should clarify:**
- If known mixed-word Jankovic orbits are recovered at 500×500, then mixed words exist in BHH space but need higher resolution to detect (narrower peaks).
- If they are not recovered, it would suggest a genuine limitation of the BHH parametrisation — the 2D (a, c) slice at fixed L preferentially accesses one topological class. This connects to the open question in the project: "finding new parametrisations that access different orbit families."

### High failure rate at L=1.0
41 candidates extracted but ~97% failed Newton refinement. RPF analysis showed the problem is **not threshold** — the one successful orbit (b^47, RPF=3.68) was only #33 by RPF value, while candidates with RPF up to 4.83 all failed.

**Root cause: edge artifacts at low `a`.** 36 of 41 candidates had `a < 0.12`, clustered at the lower boundary of the grid (a=0.05). Small `a` values in BHH parametrisation correspond to near-collision initial configurations. These produce spuriously low `d_min` (high RPF) because the trajectory passes close to its starting point by coincidence, not genuine periodicity. Newton correctly rejects them.

The one success (b^47, a=0.091, c=-2.784, T=9.30) was also at low `a` but in the negative-c region where real orbits cluster (same region as the L=0.7 b^14 orbit).

**Investigated mitigations:**
- **Raise `a` lower bound**: Won't work. 20 Jankovic orbits have a < 0.1, down to a=0.069. Real orbits and noise overlap in low-a space.
- **Edge filter (reject candidates within 2 grid cells of boundary)**: Only catches 10/41 noise candidates — most noise is at low a but not literally on the boundary. Not worth the complexity.
- **Accepted outcome**: The parallel pipeline processes 41 candidates in minutes, so the noise has negligible cost. Newton correctly rejects all the junk. No code change needed.

### Newton divergence → OOM kill
During pipeline processing of the L=1.0 scan, candidate 4 (params=(0.053, 0.796), T≈11.4) caused the process to be killed by the OS. The Newton-Raphson residual was growing (4.5e-5 → 4.6 → 22.8 → 64.5) — the starting guess was outside the basin of attraction, so each iteration sent parameters further into bad regions where the integrator took more and more steps, consuming memory until the OS killed it.

**Fix**: Added divergence detection to `refine_newton` in `floquet.py`. If the residual exceeds 100× the best seen so far or exceeds 1e3 in absolute terms, Newton bails out immediately and returns `converged=False`. This catches the failure in 2–3 iterations instead of letting it run away.

### Pipeline parallelisation
The original pipeline processed candidates sequentially — each one goes through period estimation → Newton-Raphson → Floquet analysis, all single-threaded. With 41 candidates at several minutes each, this meant hours of processing with 9 of 10 cores idle.

**Fix**: Split the pipeline into a parallel phase (period estimation + Newton + Floquet, dispatched via `multiprocessing.Pool`) and a sequential phase (cross-referencing + deduplication, which need access to accumulated results). The cross-reference word cache is warmed in the main process before dispatching workers to avoid concurrent file writes.

### Adaptive scan removal
The `adaptive_scan` function zoomed into peaks from a coarse scan at higher resolution to get a better Newton-Raphson starting guess. Since Newton has quadratic convergence, the coarse grid starting guess is already sufficient — the zoom step saved at most one Newton iteration (~seconds) per candidate while adding code complexity. Removed along with the `refine-scan` CLI command.

---

## Files in the project

| File | Lines | Purpose |
|------|-------|---------|
| `three_body.py` | 952 | Core physics: EOM, coordinate transforms, RPF, word reading, orbit tables |
| `floquet.py` | 572 | Floquet analysis: tidal tensor, variational EOM, monodromy, Newton-Raphson |
| `pipeline.py` | 569 | Candidate pipeline: scan -> refine -> classify -> cross-reference -> JSON |
| `main.py` | 322 | CLI entry point with 9 commands |
| `scanner.py` | 182 | Parallel RPF scanner |
| `analyse_catalogue.py` | 139 | Catalogue analysis and plotting |
| `parametrisations.py` | 42 | SymmetricBuilder and BHHBuilder |

### CLI commands
```
python main.py validate                     Smoke-test figure-eight
python main.py scan-symmetric [N]           Symmetric (vx, vy) scan at NxN
python main.py scan-bhh L [N]              BHH (a, c) scan at angular momentum L
python main.py floquet vx vy T              Floquet analysis for a symmetric orbit
python main.py refine-symmetric vx vy T     Newton-refine a symmetric orbit
python main.py refine-bhh a c L T           Newton-refine a BHH orbit
python main.py process-scan FILE sym [thr]  Process symmetric scan candidates
python main.py process-scan FILE bhh L [thr] Process BHH scan candidates
python main.py catalogue                    Floquet analysis of all known orbits
```

---

## What can be done next

### Blocked on AWS
1. **Run 1000x1000 BHH scans** at multiple L values. This is the primary remaining work — scanning the parameter space Li & Liao didn't search.
2. **Run high-resolution symmetric scan** (1000x1000) for comparison with the coarse 200x200 scan.
3. **Process results** through the pipeline — the tools are ready.

### Not blocked on AWS

#### High priority (DONE)
4. ~~**Optimize Newton-Raphson for long orbits**~~: Done. `_monodromy_jacobian` now uses `n_segments=1`. Auto-segmentation only in `analyse_orbit`.
5. ~~**Re-run the full catalogue**~~: Running with all fixes. Butterfly IV now included (110/110).
6. ~~**Numba JIT for the RHS function**~~: Done. 3x speedup on integration (0.038s vs 0.114s per integration). Falls back to NumPy if Numba not installed.

#### Medium priority
7. ~~**Analyse the Floquet catalogue further**~~: Done. `analyse_catalogue.py` generates three plots: T* vs word length, stability landscape, T* by angular momentum. Shows clear T* word-length dependence for BHH single-letter families.
8. ~~**Stability-topology correlation analysis**~~: Done. Pure-letter words (a^k, b^k) always unstable (0/75). Mixed-letter have 26% stability rate (9/35). All stable orbits are L=0 figure-eight family. Instability decreases with word length.
9. ~~**Adaptive scan refinement**~~: Removed. Newton-Raphson converges from coarse grid peaks directly; the intermediate zoom step added no practical value.
10. ~~**Add orbit visualization to the pipeline**~~: Done. Pipeline auto-generates trajectory + shape sphere plots for each refined candidate. Saved to `<output>_plots/` directory.

#### Lower priority
11. **Extended precision for butterfly IV**: Use mpmath or gmpy2 for the variational integration. Very slow but would give definitive Floquet multipliers. Only worth doing for the single orbit that defeats float64.
12. **Compound matrix method**: An alternative to STM-based Floquet analysis that avoids the dynamic range problem entirely. Tracks the exterior products of perturbation vectors rather than the vectors themselves. Significant implementation effort.
13. **New parametrisations**: Both symmetric and BHH are 2D slices of the 6D initial condition space. Physical intuition might suggest other slices that access different orbit families. This is an open research question.
14. **ML surrogate model**: Train a model to approximate the RPF function for rapid screening. Needs 1000x1000+ scan data as training set. Parked until scan data exists.
15. **Web interface**: A Flask/Dash app for interactive exploration of scan results and the orbit catalogue. Would make the results more accessible.
