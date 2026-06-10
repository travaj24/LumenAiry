# PMM roadmap — post-v5.14 audit (2026-06-10)

The 22-agent accuracy/speed audit (41 findings, 16/16 adversarially confirmed)
shipped its P1/P2 fixes in v5.14. The remaining CONFIRMED-FEASIBLE items, with
the audit's effort/value assessments, in recommended order:

1. **JAX cell twin** — DONE (v5.14): `pmm_efficiency_2d_cell` accepts a
   traced `eps_cell` + a concrete `region_layout` int grid; per-region
   weight vectors enter linearly. Forward parity 2e-14, AD==FD 1.2e-10.
2. **1-D homogeneous-region eig share.** 51-64% of the 1-D eig time (a quarter
   to a third of total 1-D wall time) diagonalizes the two UNIFORM half-spaces;
   `eig(L, S0)` is k0/eps-independent for a uniform region (verified
   `q^2 = eps - mu/k0^2` to 3e-13), so one geometric eig per grid serves both
   half-spaces at every wavelength. Biggest remaining 1-D speed lever (the
   layer eig is irreducible LAPACK).
3. **PMM2DStack out-of-plane layers** — DONE (v5.14): any OOP tensor layer
   promotes the whole cascade to the generalized S-matrix; a single-OOP-layer
   stack is byte-identical to `pmm_jones_2d`.
4. **1-D graded-profile helper** — DONE (v5.14): `pmm_graded_segments`
   (midpoint sampling, O(1/n^2)).
5. **Native 2-D slant** (Edée & Granet 2024, josaa-41-9-1803, in PMM_Papers).
   Research-grade: the crossed-slanted coordinate map + Gegenbauer basis.
   Until then: `PMM2DStack.add_tapered_pillar` z-staircase.
6. **Li-1997 per-direction mixed inverse rules** for the 2-D tensor diagonal
   slots (Eqs. 8/9 + 31): would lift patterned tensor cells off the Laurent
   ~1e-3 floor toward the scalar li-rule rate.
7. **Metal-TM deep-convergence study** — RESOLVED (v5.14 run): PMM degree
   18/24/30 ladder converges monotonically TOWARD the RCWA-li 1/n
   extrapolation; PMM(deg30) vs RCWA(extrapolated) = 2.4e-4 and shrinking.
   The audit's ~7e-4 gap was mutual unconvergence, not a real floor.

**Explicitly assessed and REJECTED:**
- **Numba**: confirmed no-win. The solvers are LAPACK-eig-bound (1-D) and,
  post-factorization, small-matmul-bound (2-D); the residual pure-Python loops
  (axis projection quadrature, mass sums) are <2% of wall time. END-TO-END
  speedup would be <1.1x — not worth the dependency surface.
- **GPU for the modal eig**: `jnp.linalg.eig` is CPU-only; no path.
- **Staggered near-Wood accuracy**: the ~1/sqrt(cutoff-distance) divergence is
  intrinsic to its H-partner construction; v5.14 warns inside the band and
  points to the (clean) hybrid. A regularized partner is possible but the
  hybrid already covers the regime.
