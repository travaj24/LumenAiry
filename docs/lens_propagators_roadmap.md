# LumenAiry Lens Propagators (traced / multibranch / GBD / Maslov-Levin) — Roadmap

Status as of 2026-07-07 (branch `feat/maslov-gpu-asymptotic`, unreleased v5.21).
The lens-propagator family models a real (aberrated, multi-surface) lens on a
sampled field. This roadmap tracks what ships, the measured performance state,
and — honestly — which accuracy REGIMES remain open and what each would cost.

Verdict from the propagator-landscape review (2026-07-05): there is
**no single universal lens propagator**. GBD is the engineering-universal
workhorse (polarization, coatings, folds, differentiability); Maslov is the
caustic specialist (finite through focus by construction); traced is the fast
single-congruence model, now extended THROUGH focus by multibranch; the Levin
engine makes Maslov's caustic-uniform quadrature production-fast.

**Update 2026-07-15 -- FGA (`apply_real_lens_fga`, `propagators/fga.py`):** the
Gaussian-beam-summation *family* is now caustic-accurate too, via the Frozen
Gaussian Approximation (Lu-Yang 2011, the wave-equation Herman-Kluk propagator).
A dedicated literature round (`docs/gbd_caustic_accuracy_literature.md`)
established that GBD's caustic error is a phase/interference problem, NOT an
amplitude singularity (each beamlet's complex Q keeps `det Q != 0`), and that
the fix is to FREEZE the beamlet width and weight each by the Herman-Kluk
prefactor `a = sqrt(det Z)`, `Z = (A+D) + i(k w0^2 C - B/(k w0^2))`, built from
the SAME ray-transfer/monodromy blocks GBD already computes -- a retrofit, not a
rewrite.  Validated: reproduces the angular-spectrum field to 0.9998 free-space,
matches GBD + the ASM oracle through a real singlet to 0.997-0.999, and **beats
GBD at a spherical-aberration caustic** (peak-intensity error 0.01-0.07 vs GBD
0.03-0.34).  Energy is a controllable knob (the frozen width `w0` is the FGA
convergence parameter).  Open follow-ons: through-lens STRONG-caustic demo at
high NA, higher-order FGA for energy at small `w0`, sqrt-Husimi launch sampling,
and vector/polarization FGA (the elastic-FGA Berry-phase term).  This narrows
the "no universal propagator" gap: GBD+FGA now covers the caustic regime that
previously required the Maslov hand-off.

---

## 1. Regime matrix

Legend: ✅ shipped+validated · 🔶 shipped with documented scope limit ·
⬜ planned / open · ❌ known-hard (research).

| Regime | traced | multibranch | GBD | Maslov |
|---|---|---|---|---|
| Single-congruence, pre-focus | ✅ | ✅ (reduces to plain sum) | ✅ | ✅ |
| Through focus / fold caustics | ❌ (single-valued by construction) | ✅ Ludwig band-swap (`caustic_band='ludwig'`) | ✅ (per-beamlet, no 1/√J anywhere) | ✅ `integration_method='levin'` (uniform, rigorous bound) |
| Cusp / axial (point-focus) caustics | ❌ | ⬜ Pearcey (off-axis) / Bessoid (axial) — see §5 | ✅ | ✅ (levin is cusp-accurate by quadrature) |
| Tilted / carrier input | ✅ `tilt_aware_rays` | ✅ `input_carrier=None\|'auto'\|(kx,ky)` (obliquity <0.1% below ~2.5°) | ✅ Husimi `direction_sampling` | ✅ (`lin_v3/lin_v4` carrier) |
| Multi-emitter / disjoint congruences | ✅ `traced_multi` / `traced_segmented` | ⬜ (per-emitter calls) | ✅ | ✅ |
| Polarization / Fresnel / coatings | ⬜ scalar (see §4) | ⬜ scalar | ✅ Jones + metallic/multilayer coatings | ✅ `apply_real_lens_maslov_vector` (via GBD Jones) |
| Folds / mirrors | ✅ | ⬜ untested with folds | 🔶 flat folds world-frame; ❌ powered fold mirror | ✅ `fold_split=True` |
| Differentiable (JAX) | ✅ `traced_jax` + geometry grads | ⬜ | ✅ free-space/thin-lens/analytic Jacobians; ❌ >~30° coord-break | 🔶 `maslov_jax` (Morse-index phase) |
| Edge diffraction | ❌ (GO, by construction) | ❌ (GO) | ✅ soft-edge aperture | ✅ (wave integral) |

---

## 2. Shipped this cycle (v5.21 branch, LOCAL — merge owned by the concurrent agent)

| Item | Result |
|---|---|
| `apply_real_lens_traced_multibranch` | Wavefront-construction multi-arrival traced field; analytic KMAH via quadratic `det Q(z)`; 6.4% vs exact RS decouple-pipeline oracle in the multipath annulus |
| Ludwig fold band-swap (`caustic_band='ludwig'`, default) | Coalescing pair inside `k\|S+-S-\|<=pi` swapped for the uniform-Airy field; tames the fold divergence; byte-identical elsewhere |
| Vectorized triangle rasterization | Per-cell Python loop → flat-array setup + pow-2 bbox-bucket batches; **20.3 s → 3.5 s (6×)** @192², identical contribution set |
| `input_carrier` (tilt/carrier input) | Launch follows the input phase plane; eikonal carries the carrier exactly; envelope sampled carrier-stripped (super-Nyquist OK when explicit); centroid == chief-ray oracle to 1%, energy parity 0.1% |
| Adaptive delaminating Levin engine (`lumenairy._math.levin`) | Caustic-uniform 2-D oscillatory quadrature, NO saddle finding, rigorous residual bound; fold validated 8.9e-10 with bound honored |
| Batched Levin Maslov integrator (`integration_method='levin'`) | Pair-batched lockstep waves + `_opd_vd9` shared-basis kernel + adaptive-budget acceptance: **9.7 s @ tol 1e-2 / 74.6 s @ 1e-3** on the 16×16 hard-chart benchmark (~0.04–0.3 s/px, 300–2000× the per-pixel engine), relerr reference-limited vs dense quadrature |

GBD had its own three optimization rounds earlier this cycle (windowed
reconstruct 147×, FFT-conv reconstruct 2000–3700×, GPU reconstruct 35×,
closed-form 2×2, analytic Jacobians 21.6×, memory budgets) — see
`project_lumenairy_gbd_feature_complete` history and the v5.21/v5.22 CHANGELOG
entries. GBD is considered **perf-tapped-out** on CPU.

---

## 3. Performance — deliberately left on the table

Each item was assessed and *not* taken, with the reason. Revisit only if the
listed trigger occurs.

> **Queued next dedicated pass (deferred from the v5.22.0 round, 2026-07-14,
> by owner decision — "not rushed" disposition):** (C1) the GPU CuPy
> `_opd_vd9` twin + device batched solves row below — environment confirmed
> ready this round (cupy 14.0.1 installed, RTX 4070 Ti present; the 6-output
> RawKernel template is `lenses_maslov.py` `_opd6_cupy` near the `_opd_vd9` /
> `_get_cheb4d_vd9_numba` kernels, wire into the `_run_waves` batched solves);
> (C2) Maslov phase partitioning (§4 Maslov item 1); (C3) the Maslov portable
> trace+fit cache (#4) + composite-map assessment (#11) rows below.  Run on a
> dedicated branch with benchmarks + the full Maslov/GBD regression.

| Item | Est. gain | Why deferred | Trigger to revisit |
|---|---|---|---|
| GPU (CuPy) `_opd_vd9` twin + device batched solves for the Levin integrator | ~5–20× | Real engineering; the 6-output CuPy RawKernel template exists (`lenses_maslov.py` `_opd6_cupy`) | Levin becomes a hot production path (full-grid sweeps, not ROI) |
| Levin k=9 collocation | maybe 1.3–2× | Untested box-count vs per-box-cost tradeoff | A profiling session shows wave-depth dominating |
| Levin residual fine grid 2k → k+3 | ~1.5× | **Don't**: weakens the paper-faithful rigorous bound — the bound is why the method is trustworthy at caustics | — |
| Cholesky for the Tikhonov normal-equation solves | ~1.05× | Solves are a minor slice after vd9 | — |
| Maslov portable trace+fit cache (perf item #4) | ~2× on same-optic sweeps | Invasive to the audited ~300-line core; stale-cache risk | A dedicated pass with full regression, not rushed |
| Maslov composite map per lens train (#11) | small | Orchestration-level only (single calls already trace the whole prescription) | Relay/train chaining workloads appear |
| GBD tensor-Q full GPU port | marginal | Superseded by GPU reconstruct | — |
| GBD FFT-ASM hybrid (#14) | — | **Don't revive**: known GBD/ASM convention handoff bug (7% phase residual); FFT-conv (#9) covers the same regime convention-safely | — |
| Multibranch `_kmah_free_leg` (1.5 s) / trace (0.5 s) | <2× residual | Already vectorized; not worth churn | — |

Memory is production-safe everywhere: Levin peak is chunk-bounded (~hundreds
of MB independent of grid size), GBD reconstruct is windowed/budgeted, traced
is FFT-plan-dominated (session-cached).

---

## 4. Accuracy regimes — open items

### Traced / multibranch
1. **Cusp + axial caustics** — see §5 (the only remaining on-caustic gap in
   the ray model; Maslov-levin already covers it rigorously).
2. **Per-surface Fresnel loss / Jones weighting** — traced/multibranch are
   scalar with no transmission apodization. `gbd._fresnel_jones_matrix_per_beamlet`
   already takes per-ray `(x, y, ux, uy, prescription)` — bolting a per-launch-ray
   amplitude (or 2×2 Jones) weight onto the traced families is a moderate,
   well-scoped extension. Matters for high-incidence systems.
3. **Lambare paraxial-misfit adaptive launch refinement** — accuracy-per-ray
   near caustics (split launch cells when the paraxial prediction misses by
   > a pixel); currently uniform launch + `ray_subsample`.
4. **Cell-diagonal choice by position+slowness mismatch** (Lambare §6) —
   currently fixed diagonal; matters only for very coarse launch grids.
5. **Multibranch + folds/mirrors** — untested; the trace supports mirrors but
   the exit-leg KMAH bookkeeping assumes a forward homogeneous leg.

### GBD (from the earlier dedicated rounds — each a substantial focused effort)
1. **Curved/powered fold mirror in world-frame** (flat folds exact; powered
   raises `NotImplementedError`) — needs full world per-surface differential
   transfer.
2. **Large-fold (>~30°) differentiability** — slope-space `u = L/N` degeneracy;
   needs reduced momentum or a world-frame JAX trace.
3. **Coherent ghost FIELD propagation** — currently a stray-light power budget
   (`gbd_ghost_analysis`); the coherent double-bounce field shares the
   curved-fold world-frame machinery.
4. **Tolerancing gradients dField/dTilt** — tilt/decenter as JAX-traced
   parameters through the GBD envelope + reconstruct.
5. **GRIN media** — eikonal ray-march (research).

### Maslov
1. **Phase partitioning** (subtract a fitted quadratic reference from the v2
   chart before quadrature) — cuts `n_v2` for the plain quadrature/oracle
   path; orthogonal quick win, unimplemented.
2. **Levin engine work-budget knob + platform-robust accept test**
   (v5.21.0 release forensics): `levin2d`'s quadtree accepts a box on
   parent-vs-4-child agreement to `tol`; on GitHub runners' libm that
   comparison sits at the FP-agreement floor AREA-WIDE for the fold-test
   phase, so the tree explodes toward `4^max_depth` boxes (>20–33 min,
   never completed at any tolerance tried) while the same test runs in
   seconds locally. The engine needs (a) a global `max_boxes` budget that
   degrades to a returned (honest, larger) bound instead of unbounded
   refinement, and (b) an accept test that is robust to libm rounding
   (e.g. residual-based like `levin1d_adaptive`, not value-agreement).
   Until then the two Levin CI tests are `integration`-marked (local /
   full-suite only).

---

## 5. Cusp-caustic plan (assessed 2026-07-07; NOT started)

**Effort: moderate-research, ~2–4 focused sessions.** Two distinct
deliverables, one catch:

- Already in place: per-pixel branch lists with eikonals + COMPLEX amplitudes
  (the Ludwig data structure — exactly the 3-branch uniform's input), the
  machine-precision `pearcey` evaluator, KMAH indices, and the RS
  decouple-pipeline oracle methodology for validation.
- **Pearcey cusp (off-axis / astigmatic / meridional)**: needs (a) 3-branch
  coalescence detection + region hierarchy (cusp ⊃ fold ⊃ plain, no
  double-swap with the Ludwig pair logic); (b) normal-form inversion — 3
  eikonals → Pearcey `(x, y, piston)`, a nonlinear solve that is
  ill-conditioned toward the fold edges of the cusp region; (c) CFU amplitude
  matching, which needs Pearcey DERIVATIVES `P_x, P_y` (easy series
  extension).
- **THE CATCH — axial focus is not a generic cusp**: for a rotationally
  symmetric system the on-axis focus is the axial/Bessoid catastrophe (a RING
  of rays coalesces). Its uniform field is the Bessoid J0-integral ("Bessoid
  matching", Kofler & Arnold, J. Opt. Soc. Am. A 23, 1404 (2006)) — a NEW
  special-function evaluator, the hardest single piece. Pearcey alone covers
  only tilted/cylindrical/astigmatic cusps.
- **Honest payoff check**: `apply_real_lens_maslov(integration_method='levin')`
  is ALREADY cusp-accurate (caustic-uniform by quadrature, no catastrophe
  classification) at ~0.04–0.3 s/px. So a ray-native cusp uniform is a
  speed/completeness play, not a capability gap.
- **Pragmatic interim (recommended first step, ~a day)**: route the O(few-px)
  cusp-band pixels of a multibranch map to the Maslov-levin evaluator and keep
  Ludwig for the fold lines — uniform accuracy everywhere today, at ray-model
  speed away from the cusp points.
