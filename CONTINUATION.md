# Session Continuation Notes

## Update (2026-06-12, second session): Phase 1 stabilisation built

While Mac Mini results remain inaccessible, all four Phase 1 tasks were completed:

- **1A Catalogue** — `catalogue.py`: SQLite DB (`orbits.db`, gitignored, regenerable).
  `python catalogue.py ingest|summary|query`. Dedup keys scoped by source so scan
  re-detections of known orbits stay as separate rows (recovery-rate data).
  `pipeline.py` now also emits `multiplier_magnitudes` so future candidates carry
  full Floquet spectra (enables n_unstable in the DB).
- **1B Configs** — `config.py` + `configs/*.toml` (stdlib tomllib, no new deps).
  `python main.py campaign configs/bhh_500x500_jankovic.toml [L]` reproduces the
  running campaign exactly (same save paths, resume-compatible).
- **1C Reproduction suite** — `reproduce.py`: 17 known orbits end-to-end
  (refine → RPF → Floquet → word → cross-ref), 17/17 pass in ~18 s.
  Encoded caveats: butterfly I/II share a word; Jankovic #18/#22, #28/#29 share
  words at the same L; Jankovic #3 reads as a^3.
- **1D Peak sharpness** — `peak_sharpness.py`: peak widths along a/c slices at
  campaign scan settings + geometric-lottery detection model.
  **L=0.8 validation: predicted 1.17 recoveries at 500×500, observed 2.**
  **Surprise: w_c GROWS with k** (~10× per Δk≈10) — short-word orbits are the
  hardest to detect, explaining the high-k bias in discoveries. w_a ≈ 5e-4 is
  flat in k and is a second bottleneck (da ≈ 1.1e-3 at 500×500).
  Full 75-orbit results (`peak_sharpness.json` / `.png` / `_report.txt`):
  80% mean recovery needs ~8300×8300 (274× the 500×500 cost) — quantitative case
  for ML guidance. Predicted recoveries at 500×500 per L (vs observed where known):
  L=0.65: 0.01 (obs 0 ✓), L=0.7: 0.01 (obs 0 ✓), L=0.8: 1.17 (obs 2 ✓),
  L=0.85: 0.86, L=0.9: 0.36, L=0.935: 0.63, L=1.0: 1.17, L=1.03: 5.06, L=1.07: 3.30.
  **The L=0.85–1.07 numbers are pre-registered predictions — test them against the
  Mac Mini results when they arrive.** Caveat: the pooled width-vs-k fit is
  confounded (high-k orbits live at high L); per-L fits show the k-dependence
  cleanly (e.g. L=0.8: w_c slope +0.104/k).

## Where we left off (2026-06-12)

### Mac Mini status
- **500×500 BHH scans running** at 9 Jankovic L values (0.65, 0.7, 0.8, 0.85, 0.9, 0.935, 1.0, 1.03, 1.07)
- L=0.65, 0.7, 0.8 completed. L=0.85 through 1.07 were running or pending.
- `push_results.sh` is waiting in a second tab to auto-push when scans finish
- `caffeinate -s` is keeping the machine awake
- **Has not pushed yet as of 2026-06-12.** Need to check Mac Mini when accessible.
- GitHub token is set in the remote URL on the Mac Mini — clean up with `git remote set-url origin https://github.com/alex-loughran/3body-further.git` after results are pushed.

### Results so far

**200×200 campaign (first scans):**
| L | Candidates | Refined | New |
|---|-----------|---------|-----|
| 0.5 | 0 | 0 | 0 |
| 1.0 | 41 | 1 | 1 (b^47) |
| 1.5 | 72 | 7 | 7 (b^57 to b^196) |

**500×500 campaign (Jankovic L values):**
| L | Candidates | Refined | Matched | New |
|---|-----------|---------|---------|-----|
| 0.65 | 4 | 2 | 0 | 2 (b^22, b^41) |
| 0.7 | 1 | 1 | 0 | 1 (b^26) |
| 0.8 | 24 | 13 | 2 | 11 (b^6 to b^374) |
| 0.85–1.07 | pending Mac Mini results | | | |

Plus b^14 at L=0.7 from the original 50×50 scan.

### Key findings

1. **All new orbits are pure b^k** — no mixed words across any L value. Likely a geometric property of the BHH parametrisation, not a resolution issue.

2. **RPF peaks are extremely sharp** — drop from ~6 to below threshold within dc ~0.005. At 500×500 (dc=0.015), detection is a geometric lottery. This is the root cause of low Jankovic recovery (2/9 at L=0.8). Only orbits with a grid point within dc < 0.001 are found.

3. **Pipeline validated** — L=0.8 correctly matched Jankovic #34 and #10, confirming end-to-end correctness.

4. **Word length scales with L** — b^14 at L=0.7, b^47 at L=1.0, b^57–196 at L=1.5.

### Code changes made this session
- Removed `adaptive_scan` and `refine-scan` CLI command (dead code)
- Added Newton-Raphson divergence guard in `floquet.py` (prevents OOM kills)
- Parallelised pipeline candidate processing in `pipeline.py`
- Added `run_batch.sh`, `push_results.sh`, `check_rpf.py`
- Added `README.md`

---

## Immediate tasks (when Mac Mini is accessible)

1. **Check Mac Mini** — did scans complete? Did push succeed? If not, manually push results.
2. **Pull results** to this machine and inspect all L values.
3. **Update the results table** in README.md and PROJECT_HISTORY.md with final numbers.

## Tasks en route to ML integration

### Phase 1: Stabilise the research system (pre-ML, ~4–6 weeks)

#### 1A. Catalogue system (high priority)
- Design orbit database schema (SQLite or JSON):
  - Unique orbit ID
  - Initial conditions (parametrisation, raw params, refined params)
  - Period T, energy E, angular momentum L
  - Stability: Floquet multipliers, lambda_max, stable/unstable, n_unstable_directions
  - Topology: free group word, canonical form, word length
  - Cross-reference: matched catalogue name or "new"
  - Convergence info: d_min, Newton iterations, converged flag
  - Monodromy validation: det(M), n_unit_eigenvalues
- Write ingestion script that loads all `*_candidates.json` files into the database
- Write query API (filter by L, word, stability, novelty, etc.)

#### 1B. Config system
- Replace hardcoded scan parameters with YAML/dataclass configs
- Each scan campaign gets a config file specifying: L values, grid resolution, a/c ranges, T_max, n_samples, threshold
- Enables reproducible runs

#### 1C. Reproduction suite
- Expand `validate` to test 10–20 known orbits end-to-end (scan → refine → classify → cross-reference)
- Automated pass/fail for each
- Run as CI or pre-scan sanity check

#### 1D. Peak sharpness analysis (extends current investigation)
- Quantify peak width vs word length k and angular momentum L across all known orbits
- Estimate expected detection rate as function of grid resolution
- Determine optimal resolution for target recovery rate (e.g. 80% of known orbits)
- This analysis directly motivates the ML surrogate model

### Phase 2: ML integration (~6–8 weeks after Phase 1)

#### 2A. Training data preparation
- Extract (a, c, L) → RPF value pairs from all scan .npz files
- This is millions of labelled data points, already generated
- Split by L value for train/test
- Feature engineering: include conserved quantities, symmetry features

#### 2B. RPF surrogate model (first ML model)
- Predict RPF value from (a, c, L)
- Baseline: logistic regression (periodic vs non-periodic)
- Main model: MLP in PyTorch
- Evaluation: can it identify peaks the grid scan found? Can it find peaks between grid points?
- This directly addresses the sharp-peak detection problem

#### 2C. Newton convergence predictor (second ML model)
- Predict: will Newton-Raphson converge from this starting guess?
- Training data: all candidates from pipeline (converged=True/False)
- Saves compute by skipping candidates unlikely to refine

#### 2D. Active learning loop (key research contribution)
- Replace uniform grid scan with:
  1. Coarse initial sample
  2. Train surrogate on results
  3. Sample next batch where model uncertainty is highest
  4. Evaluate, retrain, repeat
- Compare orbits-found-per-evaluation vs uniform grid
- This is the publishable result: ML-guided discovery outperforms brute-force

### Phase 3: Representation learning (~6–8 weeks after Phase 2)

#### 3A. Trajectory embeddings
- Input: orbit trajectories (time series of positions, or shape-sphere paths)
- Train contrastive model or autoencoder
- Goal: similar orbit families cluster together

#### 3B. Clustering and family discovery
- HDBSCAN on embeddings
- Validate against known classification (free group words)
- Does ML recover known families? Discover sub-structure?

#### 3C. Novel orbit detection
- Outlier detection in embedding space
- Embedding distance from known catalogue
- Flag candidates that are topologically unusual

### Phase 4: Paper (~8–12 weeks after Phase 3)

**Strongest angle: "ML-guided discovery of periodic orbits in the three-body problem"**

Required experiments:
- Baseline: uniform grid search results (already have this)
- ML-guided search: active learning results
- Comparison: orbits per compute-hour, recovery rate of known orbits
- Stability statistics across all discovered families
- Family structure validation (ML clusters vs free group words)
- Ablation: with/without ML guidance

---

## Publication angles (non-ML, from current results alone)

These are publishable without any ML work:

1. **New orbit families in BHH space** — new b^k families at L values not in existing catalogues
2. **Pure b^k topological bias** — why does BHH only access one word type? Geometric analysis of the parametrisation.
3. **Word length scaling with L** — quantitative relationship
4. **RPF peak sharpness** — the sharp-peak / geometric-lottery finding, with implications for scan methodology
5. **Floquet catalogue analysis** — T* divergence for pure-letter words, instability-L scaling (already done)

---

## Files to know about

| File | What it does |
|------|-------------|
| `PROJECT_HISTORY.md` | Detailed record of all decisions, findings, bugs, and results |
| `ROADMAP.md` | Original phased plan (Phases 1-3 complete) |
| `METHOD.md` | Technical methods documentation |
| `SUMMARY.md` | High-level project summary |
| `CLAUDE.md` | Instructions for Claude (codebase context, physics background, references) |
| `run_batch.sh` | Batch scan script (currently running on Mac Mini) |
| `push_results.sh` | Auto-push script (waiting on Mac Mini) |
| `check_rpf.py` | Quick RPF distribution checker |
