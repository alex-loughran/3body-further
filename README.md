# Three-Body Periodic Orbit Hunter

A computational pipeline for finding and classifying periodic orbits of the planar equal-mass three-body problem (m1 = m2 = m3 = 1, G = 1).

Scans 2D parameter spaces, refines candidates to machine precision with Newton-Raphson, classifies stability via Floquet analysis, reads topological type (free group words), and cross-references against all known catalogues.

## Results so far

**New orbits found in BHH parameter space (L != 0)**, a region not searched by Li & Liao (2017). Scans at 500x500 resolution across 9 angular momentum values have produced new periodic orbit families, all with pure b^k free group topology. Known Jankovic orbits are correctly recovered and matched, validating the pipeline.

| L | New orbits | Known matched | Notes |
|---|-----------|---------------|-------|
| 0.65 | 2 | 0 | b^22, b^41 |
| 0.7 | 2 | 0 | b^14 (50x50), b^26 (500x500) |
| 0.8 | 11 | 2 | Matched Jankovic #34, #10. Word lengths b^6 to b^374. |
| 1.0 | 1 | 0 | b^47 (200x200) |
| 1.5 | 7 | 0 | b^57 to b^196. Beyond Jankovic's L range. |

Additional L values (0.85, 0.9, 0.935, 1.03, 1.07) are currently scanning.

## Current work: why scans miss orbits, and what to do instead

Three results from analysing the search methodology itself:

**Detection is a geometric lottery.** Measuring the RPF peak widths around all
75 Jankovic orbits (at the campaign's own scan settings) shows peaks are far
narrower than any affordable grid spacing. A simple model — a peak is found
only if a grid point lands inside it — correctly predicts the observed
recovery rates (e.g. predicted 1.17 recoveries at L=0.8, observed 2; predicted
~0 at L=0.65/0.7, observed 0). Reaching 80% recovery would need a ~8300x8300
grid, ~300x the current compute. Counterintuitively, peak width *grows* with
word length, so the shortest words are the hardest to find. (`peak_sharpness.py`)

**The pure-b^k bias is dynamical selection, not a topological restriction.**
A syzygy census of ~30,000 trajectories shows the BHH (a, c) plane splits into
ordered single-winding domains — where every known periodic orbit lives —
separated by a chaotic zone where trajectories mix windings but nothing
detectable closes up. The letter (a vs b) is set by the angular momentum split
the parametrisation hard-codes: L_rho = a*c to the binary, L_lam = L - a*c to
the outer body. The symmetric L=0 plane is the inverse case (91% mixed),
which is why it yields mixed words. (`bk_bias.py`)

**Continuation replaces scanning.** `continuation.py` traces orbit families as
curves in (a, c, T, L) by pseudo-arclength continuation — no grid, no lottery.
First results: Jankovic #1 and #2 (both b^3) are *distinct* families with
folds at L≈0.926 and L≈0.757; their overlap hosts four b^3 orbits per L where
the catalogue knew at most two; verified b^3 orbits exist at L=0.8/0.9 where
the tables have no k=3 entry. **A linearly stable orbit at L≠0.** Following the #2 family through its
stability dip and resolving the multiplier structure at fine resolution
(`dip_trace.py`) found a genuine stable window: **L ∈ [0.83050, 0.83095]**,
where all twelve Floquet multipliers sit on the unit circle
(|λ|_max = 1.000000 across ten independently refined points, accurate
segmented monodromy). The window is bounded below by a Krein quartet
re-landing on the circle and above by period-doubling exit through −1.
Representative orbit: a=0.246486, c=−2.035290, L=0.830800, T=4.880107,
E=−1.5766, word b³. Every previously known linearly stable orbit (checked
across all 110 published orbits via the Floquet catalogue) is an L=0
figure-eight relative — this appears to be the first at L≠0, found by
continuation at parameters no grid scan visited.

## How it works

### 1. Parameter space scan
The return proximity function (RPF) measures how close an orbit comes to repeating: d_min < 10^-4 indicates periodicity. The scanner evaluates RPF across a dense 2D grid using `multiprocessing` for full CPU saturation, with incremental saving for crash recovery.

Two parametrisations are supported:
- **Symmetric** (Suvakov): zero angular momentum, 2D search in (vx, vy)
- **BHH** (Jankovic): non-zero angular momentum L, 2D search in (a, c)

### 2. Newton-Raphson refinement
Scan peaks have ~3-4 digits of accuracy. The monodromy matrix (from co-integrating the 156-component variational equations) provides the Jacobian of the periodicity residual, enabling Newton-Raphson refinement to ~13 digits in a handful of iterations.

### 3. Floquet stability analysis
The monodromy matrix eigenvalues (Floquet multipliers) classify each orbit as linearly stable or unstable. Validation checks enforce det(M) = 1 and the expected unit eigenvalue count.

### 4. Topological classification
Orbits are classified by their free group word -- a string encoding how the trajectory winds around collision configurations on the shape sphere. This identifies orbit families independent of energy, enabling cross-referencing via the Newtonian scaling symmetry.

### 5. Cross-referencing
Each candidate is matched against 805 known orbits: the figure-eight, 15 Suvakov named families, 19 figure-eight satellites, 75 Jankovic BHH orbits, and 695 Li & Liao families. Matching uses canonical cyclic word form and period-multiple detection.

## Usage

```bash
# Smoke test
python main.py validate

# Scan the symmetric (vx, vy) plane at 200x200
python main.py scan-symmetric 200

# Scan BHH (a, c) plane at L=0.8, 500x500
python main.py scan-bhh 0.8 500

# Process scan results through the full pipeline
python main.py process-scan scan_bhh_L0.8_500x500.npz bhh 0.8

# Run a reproducible campaign from a config file (scan + process per L)
python main.py campaign configs/bhh_500x500_jankovic.toml 0.8

# Reproduction suite: 17 known orbits end-to-end (pre-scan sanity check)
python reproduce.py            # or --quick for a 5-orbit subset

# Orbit catalogue database (rebuilt from JSON result files)
python catalogue.py ingest
python catalogue.py summary
python catalogue.py query --L 0.8 --new

# Peak sharpness / detection-rate analysis across known Jankovic orbits
python peak_sharpness.py

# Floquet analysis for a specific orbit
python main.py floquet 0.3471 0.5327 6.325

# Newton-refine a BHH orbit
python main.py refine-bhh 0.0951 -2.7279 0.7 2.851
```

## Project structure

| File | Purpose |
|------|---------|
| `three_body.py` | Core physics: equations of motion (Numba JIT), coordinate transforms, RPF, free group word reading, published orbit tables |
| `scanner.py` | Parallel RPF scanner with incremental saving and crash recovery |
| `parametrisations.py` | Picklable state builders for symmetric and BHH parametrisations |
| `floquet.py` | Floquet analysis: variational equations, monodromy matrix, multiplier extraction, Newton-Raphson refinement |
| `pipeline.py` | End-to-end candidate pipeline: scan -> refine -> classify -> cross-reference -> JSON |
| `main.py` | CLI entry point |
| `config.py` + `configs/` | TOML campaign configs — reproducible scan settings per campaign |
| `catalogue.py` | SQLite orbit database: ingestion from result JSONs, query API |
| `reproduce.py` | Reproduction suite: 17 known orbits validated end-to-end |
| `peak_sharpness.py` | RPF peak width measurement + grid detection-rate model |
| `bk_bias.py` | Geometric analysis of the pure-b^k bias: IC geometry + syzygy census |
| `continuation.py` | Pseudo-arclength continuation of orbit families in L |
| `dip_trace.py` | Fine-resolution stability analysis of the L≈0.83 dip |
| `ll_data.py` | Li & Liao 695-family data loader |
| `analyse_catalogue.py` | Floquet catalogue analysis and plots |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy scipy matplotlib numba
python main.py validate
```

## Key findings from Floquet catalogue analysis

Analysis of all 110 known orbits (figure-eight + Suvakov + Jankovic) revealed:

- **Stability-topology correlation**: All 9 linearly stable orbits are mixed-letter, L=0, figure-eight family members. Pure-letter words (a^k, b^k) are always unstable.
- **T\* divergence**: The topological Kepler's third law (T\* ~ 2.433) holds for mixed-letter words but breaks down systematically for pure-letter words, with short words at low L deviating by up to 2x.
- **Instability-angular momentum scaling**: Mean instability (log10 lambda_max) decreases monotonically from 2.55 at L=0.65 to 0.97 at L=1.07.

## References

- Suvakov & Dmitrasinovic, *Am. J. Phys.* 82(6):609-619, 2014 -- hunting method, free group classification
- Jankovic, Dmitrasinovic & Suvakov, *Comp. Phys. Comm.* 250:107052, 2020 -- BHH parametrisation, L != 0 orbits
- Li & Liao, *Sci. China Phys. Mech. Astron.* 60:129511, 2017 -- 695 families ([arXiv:1705.00527](https://arxiv.org/abs/1705.00527))
- Montgomery, *Nonlinearity* 11(2):363-376, 1998 -- free group / braid group classification
