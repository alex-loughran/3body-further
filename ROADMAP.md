# Three-Body Orbit Hunter — Roadmap

## Phase 1: Parametrisation-agnostic parallel scanner ✅

Build a single scanner that takes:
- A picklable callable `params → state_vector | None`
- A 2D grid of parameter values
- Integration settings

Handles: parallel dispatch via `multiprocessing.Pool`, per-row incremental saving, progress reporting, resume from crashes. Knows nothing about physics or parametrisations.

**Status: Done.**
- `scanner.py` — core parallel scanner using `imap_unordered`
- `parametrisations.py` — `SymmetricBuilder` (vx, vy) and `BHHBuilder` (a, c at fixed L)
- `main.py` — CLI entry point: `validate`, `scan-symmetric`, `scan-bhh`
- Verified: results match serial computation exactly
- Measured speedup: ~4× on 8-core M2 (limited by load imbalance in RPF eval times, not overhead)
- Scales to more cores with no code changes

## Phase 2: Floquet analysis + Newton-Raphson refinement ✅

**Goal:** Classify orbit stability AND refine candidates to machine precision. The monodromy matrix serves both purposes.

**Status: Done.**
- `floquet.py` — tidal tensor (gravity Jacobian), variational EOM (156-component state), monodromy matrix computation with dense output
- Floquet multiplier extraction + stability classification + validation checks
- Monodromy-based Newton-Raphson Jacobian (1 variational integration per step vs 6 finite-diff integrations)
- Convenience wrappers: `newton_refine_symmetric`, `newton_refine_bhh`, `analyse_orbit`
- CLI: `floquet`, `refine-symmetric`, `refine-bhh` commands in `main.py`
- Validated: figure-eight (stable, all |λ|≈1), butterfly I (unstable, λ_max≈1.78), BHH orbit #1 (unstable, λ_max≈33)
- All validation gates pass: det(M)=1, ≥2 unit eigenvalues, figure-eight confirmed linearly stable

## Phase 3: Scan campaigns + catalogue cross-reference (in progress)

**Goal:** Produce a list of candidates verified as genuinely new or identified as rediscoveries.

**Status: Pipeline complete. Big scans not yet run.**
- `pipeline.py` — full end-to-end pipeline: scan → extract peaks → estimate period → Newton-refine → Floquet classify → cross-reference → JSON output
- CLI: `python main.py process-scan <file> sym/bhh [L] [threshold]`
- Cross-references against: figure-eight, 15 Suvakov named orbits, 19 figure-eight satellites, 75 Jankovic BHH orbits
- Canonical cyclic word form for rotation-invariant matching
- Period multiple detection via word repetition + period ratio check
- Li & Liao's 695-family data is inaccessible (confirmed) — cross-reference is against Jankovic/Suvakov only
- Tested on existing 200×200 symmetric scan: correctly identifies figure-eight + butterfly I

**Remaining work:**
- Run scanner at 1000×1000+ in BHH space across multiple L values (planned for AWS)
- Run scanner at higher resolution in symmetric space for comparison
- Analyse results

**This is the phase that determines whether results are publishable.**

## Known gaps to address

- **Period multiple detection** — implemented in `pipeline.py` (word repetition + period ratio check), needs testing on real multiples from a larger scan
- **Close approach robustness** — blanket `except: continue` replaced with specific `except (RuntimeError, FloatingPointError)` across scanner.py, three_body.py, and pipeline.py
- **New parametrisations** — both symmetric and BHH are 2D slices of a 6D space; physical intuition may suggest others

## ML integration — parked for later

Needs 1000×1000+ scan data. Surrogate model for rapid RPF approximation, orbit classification from trajectory features, autoencoder analysis of parameter space structure.
