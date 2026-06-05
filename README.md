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
