---
name: Stay in Python — no Rust rewrite
description: Decision to keep the project in Python rather than rewriting in Rust or C++ for performance
type: project
---

Evaluated rewriting in Rust/C++ for performance (suggested externally). Decided against it.

**Why:** The bottleneck is `solve_ivp` DOP853, which is already compiled Fortran under the hood. Python is glue code. N=3 means SIMD intrinsics (AVX256, ARM NEON) don't help — parallelism is at the scan level (many independent integrations), not the particle-interaction level. Rust rewrite would mean reimplementing a mature integrator for marginal gain.

**How to apply:** Performance improvements should be surgical (Numba/Cython for RHS, more cores, smarter search strategies) rather than language rewrites. A Numba-JIT'd RHS existed in a prior version of the project and was removed — reintroducing it should be straightforward when needed. PySpark also ruled out — too much overhead for this workload; `multiprocessing` is sufficient. If cluster scale is ever needed, prefer lightweight distribution (SSH + parameter splitting) over big-data frameworks.
