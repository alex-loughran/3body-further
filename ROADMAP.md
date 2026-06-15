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

## Phase 3: Scan campaigns + catalogue cross-reference

**Goal:** Produce a list of candidates verified as genuinely new or identified as rediscoveries.

**Status: Pipeline complete. Big scans not yet run.**
- `pipeline.py` — full end-to-end pipeline: scan → extract peaks → estimate period → Newton-refine → Floquet classify → cross-reference → JSON output
- CLI: `python main.py process-scan <file> sym/bhh [L] [threshold]`
- Cross-references against: figure-eight, 15 Suvakov named orbits, 19 figure-eight satellites, 75 Jankovic BHH orbits, AND Li & Liao 695 families (symmetric scans)
- Canonical cyclic word form for rotation-invariant matching
- Period multiple detection via word repetition + period ratio check
- Li & Liao cross-referencing fully wired in: `ll_orbits.json` (695 entries with precomputed words) checked for symmetric candidates via `_get_known_ll_words()` in `pipeline.py`
- Tested on existing 200×200 symmetric scan: correctly identifies figure-eight + butterfly I

**Remaining work:**
- Run scanner at 1000×1000+ in BHH space across multiple L values (planned for AWS)
- Run scanner at higher resolution in symmetric space for comparison
- Analyse results

**This is the phase that determines how strong the results are.**

## Pre-scan work (can be done locally before AWS)

### 1. Pipeline hardening ✅
- Extracted `n_samples` as a parameter in `estimate_period` and `refine_candidate` (was hardcoded at 2000 in two places)
- Fixed deduplication to use `canonical_word()` so cyclic permutations are caught

### 2. Scan parameter sensitivity checks ✅
- **Threshold**: 200×200 symmetric scan at threshold=3.0 finds 7 peaks (vs 0 at 3.5). Only 2 refine successfully (butterfly I + figure-eight, both known). The other 5 are noise — threshold=3.0 is usable but doesn't find new orbits at 200×200.
- **T_max**: Only 1 Jankovic orbit has T>14 (#19 at T=15.24). T_max=16.0 has adequate margin.
- **Grid range**: All 75 Jankovic orbits fall within the default grid (a∈[0.05, 0.6], c∈[−3.5, 4.0]). No coverage gap.

### 3. L=0.7 candidate validated ✅ — confirmed new orbit
The 50×50 BHH scan at L=0.7 produced one unmatched candidate. Independently verified:
- **Word**: `b^14` (14 pure-b syzygies)
- **Refined params**: a=0.09506943645521, c=−2.72789842748187, T=2.85148374
- **d_min** = 1.2×10⁻⁶ (well below 10⁻⁴ periodicity threshold)
- **Monodromy**: det(M)=1.0000000000, 8 unit eigenvalues — clean pass
- **Stability**: unstable, λ_max=20.27 (2 unstable directions)
- **Cross-reference**: no k=14 orbit at L=0.7 in Jankovic Tables 3–5. Nearest are k=14 at L=0.8 (#42) and L=1.0 (#43). This fills a gap in the Jankovic catalogue.

### 4. Extended catalogue analysis ✅
Added to `analyse_catalogue.py`: T* residual analysis, multiplier structure breakdown, unstable directions plot. Key new findings:
- **T* divergence is systematic**: pure-letter T*=2.412±0.587, mixed-letter T*=2.429±0.282. Short pure-letter words at low L deviate worst (up to T*=4.92).
- **Instability scales with L**: mean log₁₀(λ_max) drops monotonically from 2.55 at L=0.65 to 0.97 at L=1.07.
- **Unstable directions**: pure-letter always 2 (rarely 4), mixed-letter either 0 (stable) or 2. No mixed-letter orbit has 4 unstable directions.

### 5. Local 200×200 BHH scans (not yet started)
- Run at L=0.5, L=1.0, L=1.5 locally (~2–4 hrs each on M2). Process through full pipeline.
- Purpose: shake out pipeline bugs before AWS runs

## Known gaps

- **Period multiple detection** — implemented but untested on real multiples from a larger scan
- **New parametrisations** — both symmetric and BHH are 2D slices of a 6D space; finding others is an open research question

## Investigated and parked

- **Compound matrix method** — 2nd exterior power of the STM as alternative Floquet computation. Implemented in `compound.py`, compared against standard method on 5 orbits. Standard + segmentation wins on speed and accuracy. See METHOD.md §14 for full comparison table. Code kept for cross-validation.

## ML integration — parked for later

Needs 1000×1000+ scan data. Surrogate model for rapid RPF approximation, orbit classification from trajectory features, autoencoder analysis of parameter space structure.
