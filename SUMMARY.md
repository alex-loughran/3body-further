# Work Summary: Three-Body Periodic Orbit Hunter

## 1. Core physics engine (`three_body.py`)

The fundamental task is finding periodic orbits — initial conditions where three bodies return to their starting configuration after some time T. This requires (a) integrating the equations of motion accurately over one period, and (b) measuring how close the final state is to the initial state. We chose DOP853 (8th-order adaptive Runge-Kutta) over symplectic methods because we need high single-period accuracy (d_min ~ 10⁻⁷), not long-term energy conservation. The RHS was compiled with Numba for a 3× speedup.

Two parametrisations were implemented because they access different parts of the orbit space: the **symmetric** parametrisation (Suvakov) searches at zero angular momentum (L = 0), while **BHH** (Jankovic) searches at L ≠ 0. This matters because Li & Liao already exhaustively searched the symmetric space — BHH is where undiscovered orbits are most likely to exist.

## 2. Parallel scanner (`scanner.py`, `parametrisations.py`)

Finding orbits means evaluating the return proximity function (RPF) across a dense 2D grid — at 1000×1000, that's a million independent integrations. Each takes ~0.04s, so serial execution would take ~11 hours. Since each grid point is independent, this is embarrassingly parallel. The scanner uses `multiprocessing` with `imap_unordered` (not `map`) because RPF evaluation times vary wildly — near-collision orbits take much longer, so ordered collection would leave workers idle. Incremental per-row saving was added because production scans take 17–200 hours; losing everything to a crash would be unacceptable.

## 3. Floquet stability analysis (`floquet.py`)

Finding an orbit is only half the problem — we also need to know if it's **stable** (nearby trajectories stay close) or **unstable** (nearby trajectories diverge exponentially). This is determined by the Floquet multipliers, which are eigenvalues of the monodromy matrix (the state transition matrix evaluated after one period). Computing the monodromy requires co-integrating the 12×12 variational equations alongside the orbit — a 156-component system.

This step also enabled **Newton-Raphson refinement**: the monodromy matrix gives the Jacobian of the periodicity residual essentially for free, reducing the cost per Newton step from 6 integrations (finite differences) to 1 variational integration. Scan candidates have only ~3–4 digits of accuracy; Newton-Raphson pushes them to machine precision (~13 digits), which is necessary both for reliable stability classification and for meaningful cross-referencing against published orbits.

## 4. Topological classification (free group words)

Two orbits at different energies can be the same family related by a scaling symmetry. The free group word — a string like `aAbB` encoding how the orbit winds around collision configurations on the shape sphere — is a topological invariant that identifies orbit families regardless of energy. This is essential for cross-referencing: we compare words, not raw parameter values. Implementing this required detecting when the three bodies become collinear (shape sphere equator crossings), identifying which body is in the middle, and converting the syzygy sequence to a word via lookup tables from Suvakov & Dmitrasinovic (2014).

## 5. Li & Liao data extraction (`ll_data.py`, `ll_orbits.json`)

Li & Liao (2017) found 695 orbit families in the symmetric (L = 0) space using supercomputer-scale computation. Any candidate we find in symmetric space must be checked against their catalogue to determine novelty. Their data wasn't in downloadable files but was embedded in HTML tables on their website. We scraped it and precomputed the free group words for all 695 families. This means the pipeline can immediately tell whether a symmetric candidate is a rediscovery.

## 6. End-to-end pipeline (`pipeline.py`)

The preceding steps were independent tools. The pipeline chains them into a single automated workflow: load scan → extract peaks from RPF heatmap → estimate periods → Newton-refine each candidate → compute Floquet multipliers → read the free group word → cross-reference against all 805 known orbits → output JSON + plots. Without this, processing a 1000×1000 scan (which may contain dozens of candidates) would require manual intervention at every stage. The pipeline also handles energy-normalised deduplication (same family detected at different grid points) and incremental checkpointing.

## 7. Floquet catalogue (`floquet_catalogue.json`, `analyse_catalogue.py`)

Before running large scans for new orbits, we applied the full Floquet analysis to all 110 known orbits as both a **validation exercise** and a source of **new results**. This produced two findings: (a) the topological Kepler's third law (T* ≈ 2.433) breaks down for single-letter words (BHH orbits), and (b) all 9 stable orbits are mixed-letter, L = 0, figure-eight family members — pure-letter words are always unstable. These stability-topology correlations have not been reported in the literature for this set of orbits.

## 8. Compound matrix method investigation (`compound.py`)

The 2nd compound matrix method was implemented as an alternative to the standard variational approach for computing Floquet multipliers. Instead of the 12×12 state transition matrix (144 components), it tracks the 66×66 2nd exterior power (4356 components), whose eigenvalues are products λᵢλⱼ of pairs of multipliers. In theory, this should be better conditioned for highly unstable orbits.

Empirical comparison across five orbits (stable through λ_max = 1085) showed the standard method with segmented integration is both faster (~2×) and more accurate (2–5 orders of magnitude better determinant errors). The compound method also fails entirely for long-period orbits (T > 60) where the large system becomes too stiff. The code is retained as a cross-validation tool but is not used in the production pipeline.

## 9. Extended catalogue analysis and new orbit

The Floquet catalogue analysis was extended with three additional findings, all potential results sections for a write-up:

1. **T* residual structure**: The topological Kepler's third law (T* ≈ 2.433) shows qualitatively different behaviour for pure-letter vs mixed-letter words. Mixed-letter words cluster tightly around the prediction (T* = 2.429 ± 0.282), but pure-letter words scatter widely (T* = 2.412 ± 0.587), with short words at low L deviating by up to a factor of 2. This refines the finding from §7: the law doesn't just "break down" for pure-letter words — the deviation is systematic and depends on both word length and angular momentum.

2. **Instability-angular momentum scaling**: For BHH orbits, mean log₁₀(λ_max) decreases monotonically from 2.55 at L=0.65 to 0.97 at L=1.07. Higher angular momentum systematically stabilises orbits. This has not been quantified for this set of orbits.

3. **New orbit at L=0.7**: A 50×50 BHH scan produced one orbit not in any existing catalogue — word b^14, T=2.85, d_min=1.2×10⁻⁶, independently verified with clean monodromy (det=1, 8 unit eigenvalues). This fills a gap in the Jankovic (k, L) coverage and demonstrates that even small exploratory scans in BHH space can yield new results.

## Current status

The infrastructure is complete and validated. One new orbit has been found at L=0.7 from a small exploratory scan. The remaining work is running large-scale BHH scans (1000×1000 at 9 angular momentum values) on AWS, which is where systematic discovery of new orbits — in parameter space Li & Liao did not search — would occur. The pipeline is ready to process whatever comes out.