# Three-Body Periodic Orbit Hunter

## What this project is

A computational pipeline for finding and analysing periodic orbits of the planar equal-mass three-body problem (m1 = m2 = m3 = 1, G = 1). Personal research project building on a completed university mini-project. The original coursework is irrelevant -- only the extension work matters.

## Current state of the codebase

### Core module: `three_body.py` (migrated from `~/PycharmProjects/orbits/`)
- **Equations of motion**: Numba JIT-compiled (`@njit`) with NumPy fallback, integrated via `scipy.integrate.solve_ivp` with DOP853 (8th-order adaptive Runge-Kutta). ~3x faster than pure NumPy.
- **Coordinate transforms**: Cartesian → Jacobi → shape sphere (rotation/translation-invariant)
- **Return proximity function (RPF)**: measures how close an orbit comes to repeating. `d_min < 1e-4` = periodic. Two-stage: coarse sampling then Brent refinement via `minimize_scalar`
- **Two parametrisations of initial conditions**:
  - **Symmetric** (Suvakov): r1=(-1,0), r2=(1,0), r3=(0,0), v1=v2=(vx,vy), v3=(-2vx,-2vy). Zero angular momentum. 2D search space (vx, vy).
  - **BHH** (Jankovic et al.): collinear Jacobi positions, parametrised by (a, c, L). Used for L≠0 orbits.
- **Free group word reading**: topological classification via shape sphere equator crossings
- **Published orbit data**: Tables 3-5 from Jankovic (75 BHH orbits) and Tables 1-2 from Suvakov (15 named + 19 figure-eight satellites)
- **Candidate extraction/refinement**: `find_candidates` (peak detection via `maximum_filter`), `refine_orbit` (Nelder-Mead), `refine_orbit_gradient` (Suvakov 5×5 grid descent)
- **Plotting**: orbit trajectories in real space + shape sphere stereographic projection

### Parallel scanner: `scanner.py` + `parametrisations.py`
- **`scanner.py`**: Parametrisation-agnostic parallel RPF scanner. Takes any picklable callable mapping params → state vector. Uses `multiprocessing.Pool` with `imap_unordered` for full worker saturation. Incremental per-row saving to `.npz`, automatic resume from partial results.
- **`parametrisations.py`**: Two builder classes — `SymmetricBuilder` (vx, vy) and `BHHBuilder` (a, c at fixed L). Both are picklable for multiprocessing.
- Verified: parallel results match serial computation exactly.
- Measured ~4× speedup on 8-core M2. Scales linearly with more cores, no code changes needed.

### Floquet analysis: `floquet.py`
- **Tidal tensor**: 6×6 `_gravity_jacobian` (∂accelerations/∂positions), validated against finite differences
- **Variational equations**: 156-component extended state (orbit + 12×12 state transition matrix), co-integrated with DOP853
- **Monodromy matrix**: `compute_monodromy(state0, T)` → (M, final_state), with dense output for exact endpoint
- **Floquet multipliers**: eigenvalue extraction, stability classification, validation checks (det=1, unit eigenvalue count)
- **Newton-Raphson refinement**: monodromy-based Jacobian (1 variational integration per step vs 6 finite-diff integrations). Convenience wrappers: `newton_refine_symmetric`, `newton_refine_bhh`
- **`analyse_orbit`**: one-shot monodromy + multipliers + stability classification

### Candidate pipeline: `pipeline.py`
- **End-to-end**: scan → extract peaks → estimate period → Newton-refine → Floquet classify → cross-reference → JSON output
- **Cross-reference**: matches against figure-eight, 15 Suvakov named orbits, 19 satellites, 75 Jankovic BHH orbits
- **Word matching**: canonical cyclic form, period multiple detection via word repetition + period ratio
- **`process_scan(scan_path, parametrisation, L, threshold)`**: the main pipeline function

### Entry point: `main.py`
CLI commands:
- `python main.py validate` — smoke-test figure-eight orbit
- `python main.py scan-symmetric [N]` — symmetric (vx, vy) scan at N×N
- `python main.py scan-bhh L [N]` — BHH (a, c) scan at angular momentum L, N×N
- `python main.py floquet vx vy T` — Floquet analysis for a symmetric orbit
- `python main.py refine-symmetric vx vy T` — Newton-refine a symmetric orbit
- `python main.py refine-bhh a c L T` — Newton-refine a BHH orbit
- `python main.py process-scan FILE sym/bhh [L] [threshold]` — full candidate pipeline

### Data
- `scan_vxvy_200x200.npz` — existing 200×200 symmetric scan from the original project (scan settings unknown, not directly reproducible)

## Strategic context

### Li & Liao data (now available)
LL found 695 families using Clean Numerical Simulation — arbitrary-precision Taylor series integration with 128+ significant digits, run on ~1000 CPU cores. Their numerical data (v1, v2, T, T*, Lf) was extracted from HTML tables at https://numericaltank.sjtu.edu.cn/three-body/three-body-movies.htm and is stored in `ll_orbits.json` (695 entries). `ll_data.py` provides the API. Cross-referencing against LL families is wired into the pipeline.

### Where we can contribute
- **BHH parameter space (L≠0)**: LL searched only the symmetric (vx, vy) plane (L=0). The 75 Jankovic orbits at L≠0 come from BHH space, which LL didn't scan. Scaling up BHH scans is non-redundant with LL.
- **Floquet stability analysis**: Computing stability classification (Floquet multipliers) for BHH orbits + new candidates. LL data is inaccessible (confirmed). This is a separate piece of work from finding orbits.
- **Topological/stability structure mapping**: Systematic analysis of how stability varies with free group word length, which topological classes are stable vs unstable, etc.

### Roadmap
See `ROADMAP.md` for the full plan. Current status: Phases 1-2 complete, Phase 3 pipeline complete, big scans not yet run.

## Key physics

- G = 1, equal masses m = 1 throughout. All orbits are planar (2D).
- DOP853 over symplectic methods: need high per-step accuracy over single periods, not long-term bounded energy error.
- The shape sphere encodes triangle shape independent of size/position/orientation. Periodic orbits trace closed curves on it.
- Collision points on the equator at 120° intervals act as topological punctures — orbits classified by winding (free group on generators a, b).
- Topological Kepler's third law: T* = T|E|^{3/2}/L_f ≈ 2.433 across all families, where L_f = free group word length.
- The full initial condition space is 6D after constraints. Symmetric and BHH are each 2D slices — finding new parametrisations that access different orbit families is an open question.

## Key references
- Suvakov & Dmitrasinovic, Am. J. Phys. 82(6):609-619, 2014 (hunting method, free group classification)
- Jankovic, Dmitrasinovic & Suvakov, Comp. Phys. Comm. 250:107052, 2020 (BHH orbits, L≠0)
- Li & Liao, Sci. China Phys. Mech. Astron. 60:129511, 2017 (695 families, arXiv: 1705.00527)
- Montgomery, Nonlinearity 11(2):363-376, 1998 (free group / braid group classification)

## Notes for Claude
- The user has limited dynamical systems / chaos theory background but strong computational skills (Python, NumPy, SciPy). Explain dynamics concepts concretely — tie to things in the codebase rather than abstract theory.
- The user prefers direct, critical feedback. Don't sugarcoat.
- Don't touch or reference the original coursework/submission — it's done.
