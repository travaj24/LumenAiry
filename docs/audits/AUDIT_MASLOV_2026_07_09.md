# Maslov Propagator Audit — 2026-07-09

Scope: `elements/lenses_maslov.py` (3,503 — the Maslov
canonical-operator / phase-space beam-summation lens propagator, the
caustic-uniform peer of GBD/HFPI).  **Deep-read** (line-level, physics
re-derived): the Chebyshev 4-var value/derivative evaluators and their
numba/cupy twins (124-740), the per-pixel Newton saddle solver, the
normal-equations fit and auto-order selection (741-880), and the two
diffraction-catastrophe special functions (882-982).  **Structural**
(read for control flow, conventions, and the specific defect classes
Maslov methods are prone to — branch amplitude sign, KMAH phase,
caustic seams): the 1,170-line driver `apply_real_lens_maslov`
(984-2156), the vector variant, and the five integration backends
(quadrature / stationary-phase / Levin / local-quadrature, numpy +
cupy).  Read-only; largest single file in the campaign.

---

## 1. Verdict

**The Maslov physics is correct.**  The verifications that mattered:

* **Chebyshev evaluators** — the value recurrence
  `T_m = 2u T_{m-1} − T_{m-2}`, the first derivative `T'_n = n U_{n-1}`
  (via the `U_m = 2u U_{m-1} − U_{m-2}` second-kind recurrence), and
  the differentiated second-derivative recurrence
  `T''_{m+1} = 2u T''_m + 4 T'_m − T''_{m-1}` (seeded `T''_2 = 4`) are
  all correct; the numba kernels are structurally ULP-identical to the
  `_opd*_numpy` references (same basis products, same coefficient
  reductions), and the value/derivative separation (derivatives only in
  the v2 = `(v2x, v2y)` pupil variables, `T1·T2` shared over the s2
  output variables) is consistent across the vd3/vd9/6-output kernels.
* **`pearcey`** (cusp catastrophe) — I re-derived the quartic-Gaussian
  moment `∫ t^{2k} e^{i t⁴} dt = ½ Γ((2k+1)/4) e^{iπ(2k+1)/8}` (via the
  `u = t⁴` substitution + the `∫₀^∞ u^{a-1}e^{iu}du = Γ(a)e^{iπa/2}`
  contour rotation) and the even-in-y double series; the `P(0,0) =
  ½Γ(¼)e^{iπ/8}` anchor (`|P|=1.8128`, arg 22.5°) matches.
* **`uniform_fold_airy`** (fold catastrophe, Chester-Friedman-Ursell)
  — the cubic-normal-form mapping `A = ½(f₁+f₂)`, `ζ = [¾(f₂−f₁)]^{2/3}`
  is correct (the cubic `u³/3 − ζu` has saddle-phase split `(4/3)ζ^{3/2}
  = f₂−f₁`), the exact Airy identity `∫ e^{ik(t³/3 − ct)}dt = 2π k^{-1/3}
  Ai(−k^{2/3}c)` checks out, and — the crux — the branch phase is pinned
  by the per-saddle Maslov factor `exp(i·sgn(f'')·π/4)`, sidestepping the
  ambiguous `√(f''/2u)` root sign.
* **KMAH / Maslov index** — handled correctly and locally: the
  per-saddle `sgn(f'')π/4` phases (stationary-phase path) and the direct
  oscillatory-integral evaluation (quadrature / Levin / local-quadrature)
  carry the catastrophe phase automatically, with **no** global mod-2/mod-4
  index accumulation to get wrong (contrast the delta-audit D3 concern in
  `caustic_diagnostic` — no analog defect exists here).
* **Newton saddle** — the 2×2 inverse-Hessian step `dv = −H⁻¹g`
  (`dv3 = −(H₄₄g₃−H₃₄g₄)/det`, `dv4 = −(−H₃₄g₃+H₃₃g₄)/det`) is correct,
  as is folding the linear-in-v2 OPD term into the gradient (zero Hessian
  contribution).  Step-length damping and the freeze-converged-pixel
  (GPU) / shrink-active-subset (CPU) forms are numerically equivalent.
* **`_solve_fit`** — the normal-equations Cholesky (`G = AᵀA`,
  `cho_solve`) with the LU→SVD fallback ladder is sound for the
  well-conditioned oversampled tensor-Chebyshev Vandermonde;
  `_select_poly_order_auto` correctly scores on a **held-out** ray subset
  (rejecting node-noise overfit) and degrades to in-sample only when
  undersampled.
* **Phase upsample** (3.5.6) — interpolating `cos φ` / `sin φ` then
  `angle()` is the right fix for the caustic-seam artifact of 2-D
  phase-unwrap; the Nyquist-aware fine-grid re-application of a freeform
  OPD slope (`c1 > N_coarse/4` aliases under the phase-zoom) is
  correctly reasoned and load-bearing (verified prism case).

One real finding follows.

## 2. Findings

### MSL-1 (P4) — the near-singular-Hessian guard produces `det_safe = 0` (NaN saddle step) for a tiny NEGATIVE determinant
All three saddle-based integrators regularise the Newton denominator as

```python
det_safe = where(abs(det_H) < 1e-30, sign(det_H) * 1e-30 + 1e-30, det_H)
```

at `lenses_maslov.py:764` (GPU `_maslov_newton_saddle_xp`),
`:2689` (CPU `_integrate_stationary_phase`), and `:3218` (CPU
`_integrate_local_quadrature`).  The `+ 1e-30` addend was meant to floor
the exactly-zero case (`sign(0)=0 → 1e-30`, fine), but it **cancels the
floor on the negative branch**: for `det_H ∈ (−1e-30, 0)`, `sign = −1`
gives `−1e-30 + 1e-30 = 0` → division by zero → `inf` step → (after the
`0.5/step` damping) a `NaN` step → the pixel's `u_v2` clips to `NaN` and
stays poisoned, contributing `NaN` to the output field.  A saddle's OPD
Hessian determinant passes through zero exactly at a **fold caustic**,
so near-caustic pixels are precisely where near-zero `det_H` occurs —
though hitting the specific `(−1e-30, 0)` window (relative measure
~1e-30) on a discrete Newton iterate in float64 is astronomically
unlikely, making this a latent rather than routine failure.  The direct
oscillatory-integral integrators (`quadrature`, `Levin`) do no saddle
division and are immune; near-caustic charts are also meant to route to
those.  **Fix** (all three sites): replace the small-branch value with a
sign-preserving floor that never returns zero, e.g.
`np.where(det_H < 0, -1e-30, 1e-30)`.

## 3. Coverage statement

Deep-read (physics re-derived / recurrences checked): lines 124-982 —
the `_cheb4d*` / `_opd*` evaluators (numpy + numba), `_maslov_newton_
saddle_xp`, `_solve_fit`, `_gram_cho_factor`, `_select_poly_order_auto`,
`uniform_fold_airy`, `pearcey`.  Structurally covered (control flow,
conventions, seam/KMAH/amplitude defect classes; not every line):
`apply_real_lens_maslov` (984-2156), `apply_real_lens_maslov_vector`,
and the five integrators (`_integrate_quadrature`,
`_integrate_stationary_phase`, `_integrate_levin`,
`_integrate_local_quadrature`, + cupy twins, 2229-3456) — their saddle
solve, Chebyshev-fit, and phase conventions were checked at the seams;
the MSL-1 guard was traced across all three saddle sites.  **Not
line-verified**: the interior quadrature-node bookkeeping of each
integrator (the oscillatory-integral weight assembly), which would need
a dedicated numerical-integration pass.  Not audited: `bsdf.py`,
`segment_geometry.py`, `eme/`, the `io/` siblings, `optimize/`.

---

*Audit performed single-context against lumenairy v5.21, 2026-07-09.
Companion docs: `AUDIT_COATINGS_ELEMENTS_2026_07_09.md`,
`AUDIT_DOE_GRATING_FREEFORM_2026_07_09.md`, and the raytrace/propagators
set (`AUDIT_RAYTRACE_CORE_2026_07_08.md` D3/KMAH cross-ref).*
