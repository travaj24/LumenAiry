# RCWA Convergence-Acceleration Techniques — Review & Roadmap

**Date:** 2026-06-01
**Author:** Claude Opus 4.8 (1M context)
**Type:** Forward-looking techniques review + gap analysis (NOT a version bug-audit)
**Scope:** Advanced numerical techniques for *faster harmonic convergence* of `lumenairy/elements/rcwa.py`, focused on the hard cases — **metals, high index-contrast, and TM polarization** — plus the eigen/linear-algebra speed levers that compound with them.
**Module state read:** `rcwa.py` at HEAD `3460d00` (v5.5.2). All "what's there today" claims below are grounded in the actual source (file:line cited); the technique landscape is from FMM literature (references in Part 6, with an honesty note on citation precision).

---

## Part 0 — TL;DR

RCWA's classic pain point is that **TM polarization and metallic/high-contrast structures converge slowly in the harmonic count N**, because the electric field normal to a permittivity jump is discontinuous and its Fourier series rings (Gibbs, ~1/N). Lumenairy already handles this *correctly in 1-D* (Li's inverse rule), but the **2-D path uses only the naive Laurent rule** — exactly the slow/wrong-converging case — and there is no coordinate-adaptivity, no alternative basis, and no series acceleration anywhere.

**The single highest-value addition is the Fast Fourier Factorization (normal-vector) rule for 2-D** — the module's docstring already *claims* it ("provided separately") and already *cites* the foundational paper (Li 1997), but it is **not implemented**. After that: **Adaptive Spatial Resolution** (Granet), then their combination ("matched coordinates"), then the cheap wins (circular truncation, normal-incidence symmetry halving, eig-reuse across sweeps).

Every technique here fits one of three physical strategies:
1. **Factorize the products correctly** so the *truncated* series still satisfies the interface boundary conditions → FFF/normal-vector (the 2-D generalization of the 1-D inverse rule already in the code).
2. **Put resolution where the discontinuity is** → Adaptive Spatial Resolution, C-method.
3. **Use a basis that natively carries the discontinuity** → Polynomial/B-spline Modal Method (exponential, not algebraic, convergence).

Dielectrics already converge fast (Laurent is fine); the wins below matter for **metals, plasmonics, high-contrast, and TM**. A subset (ASR, circular truncation, eig-reuse, extrapolation) helps everything.

---

## Part 1 — What the module implements today (verified baseline)

| Capability | Status | Location |
|---|---|---|
| **1-D Li inverse rule (TM/metals)** — wall-normal `E_x`/`E_z` use `[[1/ε]]⁻¹`, tangential `E_y` uses Laurent `[[ε]]` | ✅ correct & properly placed | `_layer_eigenmodes` (rcwa.py:671-756), docstring 682-695 |
| **`formulation` auto-select** — `'li'` chosen for metallic `n`, `'laurent'` for dielectric | ✅ 1-D only | `rcwa_efficiency_1d` (854-), `_metallic` (938) |
| **S-matrix / Redheffer recursion** — stable for thick/metallic layers | ✅ | `_redheffer_star`/`_interface_smatrix`/`_propagation_smatrix` (794-852) |
| **Enhanced-transmittance / Rumpf gap-medium bookkeeping** | ✅ | header refs 22-37 |
| **Analytic uniform-layer eigenmodes** (skips eig for half-spaces) | ✅ | `_homogeneous_eigenmodes` (758-792) |
| **Analytic shape form-factors** (disks/rectangles, not staircased FFT of the cell) | ✅ | `_validate_shapes` (506-), cumulative-area note 519 |
| **Wood/Rayleigh-anomaly + non-propagating-incidence guards**, evanescent regularization | ✅ | `_grazing_safe_wavelength` (382-), `_require_propagating_incidence` (365-), `_inv_lam`/`_sqrt_decay` (324-363) |
| **2-D factorization** | ⚠️ **Laurent only** | `rcwa_efficiency_2d` (1184-), `EPS_xx = EPS` (1300), guard rejecting non-`'laurent'` (1249-1252) |
| **FFF / normal-vector for 2-D metals** | ❌ **claimed but absent** | docstring says "provided separately" (1232-1233); no implementation in file; Li 1997 cited in header (33-34) but its 2-D rule unused |
| Adaptive Spatial Resolution (coordinate transform) | ❌ none | — |
| Circular / disk truncation for 2-D order set | ❌ rectangular box only | `_harmonic_orders_2d` (1147-) |
| Normal-incidence symmetry block-diagonalization | ❌ full 2N eig always | `_layer_eigenmodes` (671-) |
| Alternative basis (PMM / B-spline) | ❌ Fourier only | — |
| Eig reuse / warm-start across λ or θ sweep | ❌ re-solve per point | `rcwa_efficiency_vs_wavelength` (1060-) |
| Spectral filtering / series extrapolation | ❌ none | — |

**Bottom line:** 1-D metals/TM are already in good shape (Li rule present and correctly placed — the module even documents the subtle "which block gets the inverse rule" point at 682-695). **The gap is everything 2-D, plus all coordinate/basis/series-acceleration levers.**

---

## Part 2 — Why metals & TM converge slowly (the physics to exploit)

At a permittivity discontinuity, Maxwell requires **D⊥ continuous but E⊥ discontinuous** (it jumps by the permittivity ratio). The product `D = εE` therefore couples two functions with *concurrent* jumps. A truncated Fourier series of such a product, formed by the naive (Laurent/direct) convolution `[[ε]]·Ê`, does **not** converge to the correct truncated boundary condition — it rings as ~1/N (Gibbs). This is:
- **Worst for TM/p** (the field has a component normal to the walls) — TE/s is tangential everywhere and converges fast even with Laurent;
- **Worst for metals / high contrast** (the jump magnitude ∝ the permittivity ratio, which is huge and complex for metals);
- **Benign for low-contrast dielectrics** (small jump → small ringing → Laurent is fine).

**Li's insight (1996):** factor a product of two functions with concurrent jumps using the **inverse rule** `[[1/ε]]⁻¹` for the discontinuous (normal) component, and Laurent `[[ε]]` for the continuous (tangential) one. This restores fast convergence. Lumenairy does exactly this in 1-D. The advanced techniques below are essentially *the ways to extend that idea to 2-D arbitrary geometry, and to spend resolution more cleverly.*

---

## Part 3 — The technique landscape (grouped by lever, prioritized)

### A. Correct Fourier factorization — the #1 lever for 2-D metal/TM

**A1. Fast Fourier Factorization / Normal-Vector Field method (2-D).** ★ top priority
*Popov & Nevière 2001; Schuster/Weiss et al. 2007; Li 1997.*
Construct a continuous unit-normal field **N**(x,y) over the cell (pointing across each material boundary) and split **E** into normal and tangential projections via the operators **NN**ᵀ and (**I** − **NN**ᵀ); apply the inverse rule to the normal projection and Laurent to the tangential one. The permittivity convolution becomes a *tensor* built from these projections rather than the scalar `[[ε]]`. This is the genuine 2-D generalization of the 1-D rule already in `_layer_eigenmodes` — it changes 2-D metal/plasmonic convergence from ~1/N to nearly the 1-D-TM rate.
- For Lumenairy's **analytic shapes** (disks, rectangles) the normal field is closed-form on the boundary, so it composes directly with the existing form-factor path — no level-set machinery required for the common cases.
- For arbitrary masks, generate **N** from the (smoothed) gradient of a level-set/indicator, or the "complex normal vector field" construction (Schuster 2007).
- **This closes the stale-docstring gap** (rcwa.py:1232-1233 claims it exists) and uses the already-cited Li 1997 reference (header line 33-34).

**A2. Analytic (closed-form) form-factors — already present, keep leveraging.**
Using the analytic FT of a disk/rectangle (Bessel/sinc form factors) instead of FFT-sampling a staircased cell removes *in-plane* staircasing. Lumenairy has this. Note it is **orthogonal to** the factorization-rule problem (A1): you still need FFF for the *normal-direction discontinuity* even with perfect form-factors.

### B. Coordinate transformations — put resolution at the walls

**B1. Adaptive Spatial Resolution (ASR).** ★ biggest multiplier for deep/lamellar metal
*Granet 1999 (1-D); Vallius & Honkanen 2002 (2-D).*
A nonlinear coordinate map x→x′(x) whose Jacobian crowds collocation points near the interfaces, so a fixed N resolves the steep field there. Implemented as a modified metric in the Fourier operator (a smooth, monotonic map parameterized to concentrate near wall positions). Typically **cuts required N by 3–10× on metallic lamellar gratings**; since the eig is O(N³), halving N is ~8× wall-clock.

**B2. Matched coordinates = ASR + FFF together.** ★ state of the art for hard plasmonics
*Weiss, Granet et al. 2009.*
Combine the coordinate stretch (B1) with the normal-vector factorization (A1) in the transformed frame. This is what practitioners reach for on genuinely hard metallic/plasmonic gratings. Natural follow-on once A1 and B1 both exist.

**B3. C-method (Chandezon curvilinear coordinates).** situational
*Chandezon et al. 1982.*
For **smoothly profiled** gratings (sinusoidal, blazed/echelle) transform to a coordinate where the interface is flat → exponential convergence where staircased RCWA crawls. A different solver family; worth a hybrid path if blazed/echelle metasurfaces ever enter scope. Not relevant to binary/lamellar metasurfaces.

### C. Basis & truncation set

**C1. Circular ("disk") truncation for 2-D.** ★ cheap, immediate
*Lalanne 1997.*
Retain orders inside |**G**| < G_max rather than a rectangular (m,n) box → isotropic resolution, smoother/faster convergence, and far fewer wasted high-|G| corner orders — especially for non-separable (hexagonal/oblique) lattices. Nearly free: it only changes which orders `_harmonic_orders_2d` (rcwa.py:1147) emits. Since the eig is O(N³), trimming the corner orders is a direct speed win too.

**C2. Non-orthogonal / true reciprocal-lattice basis.** situational
For hexagonal/oblique lattices, expand in the true reciprocal-lattice vectors instead of forcing a rectangular supercell — avoids large artificial cells (and their inflated N).

**C3. Polynomial / B-spline Modal Method (PMM).** "beyond Fourier"
*Edee 2011 (Gegenbauer); Edee & Guizal (B-spline).*
Replace the plane-wave basis with piecewise polynomials/B-splines that place a knot *at* the interface → no Gibbs at all → **exponential** convergence for lamellar metals instead of algebraic. More surgery than A/B (it's a different modal expansion, not an accelerated Fourier one), but it's the answer if the goal is a step-change in 1-D deep-metal solver speed rather than an acceleration of the existing one.

### D. Eigenproblem / linear-algebra acceleration (wall-clock, not harmonic count)

**D1. Normal-incidence symmetry block-diagonalization.** ★ exact, ~8×
For mirror-symmetric cells at normal incidence the 2N eigenproblem decouples into even/odd N-blocks. Solving two (N)³ eigs instead of one (2N)³ is ~8× cheaper, exactly. Very common in metalens/metasurface unit cells. Pure speed, no accuracy cost.

**D2. Eigendecomposition reuse / warm-start across λ and θ sweeps.** ★ cross-connects with existing API
The layer modes vary slowly along a wavelength or angle sweep. Perturbative update / interpolation / warm-start of the eig (rather than a cold `zgeev` per point) speeds up exactly the sweeps `rcwa_efficiency_vs_wavelength` (rcwa.py:1060) and metasurface-dispersion studies run most. Ties directly into the speed/cross-connect roadmap already on the books.

**D3. Skip-eig / analytic modes for uniform layers — already present.** (`_homogeneous_eigenmodes`, 758). Extend the same idea to *any* layer that is laterally uniform within a stack.

**D4. Mixed-precision eig with iterative refinement.** opportunistic
Solve the dense eig in lower precision, refine the retained (propagating + low-order evanescent) modes in double. The module already documents that the eig is ill-conditioned in single precision (`_warn_if_jax_f32`, 267-281), so any mixed-precision path must refine carefully — but the propagating subset is well-conditioned.

### E. Series acceleration & field-reconstruction filtering

**E1. Richardson / Shanks(epsilon) extrapolation of efficiency-vs-N.** cheap accuracy
Extrapolate the slowly-converging efficiency to N→∞ from a few finite-N solves — 1–2 extra digits for the cost of 2–3 solves, solver-agnostic. Useful as a convergence *estimator* even if not used to report values.

**E2. Spectral filtering (Lanczos-σ / raised-cosine / Gegenbauer reconstruction) of the near-field.** ★ directly fixes the multi-order P1
Applied to the *reconstructed* real-space field (not the efficiencies), these suppress Gibbs ringing in `to_multiorder_field` / `to_jones_field(order=)`. This is the same near-field reconstruction the v5.5.2 audit flagged for energy non-conservation — filtering won't fix the missing flux weight, but it addresses the *ringing* half of reconstruction quality and should be considered alongside the flux-weight fix.

---

## Part 4 — Prioritized roadmap for Lumenairy (mapped to the gaps)

| Priority | Technique | Why / payoff | Effort | Refs |
|---|---|---|---|---|
| **1** | **FFF / normal-vector for 2-D** (A1) | Closes the stale-docstring gap; fixes the actual 2-D metal/TM convergence hole; uses already-cited Li 1997; composes with existing analytic shapes | med | Popov-Nevière 2001; Schuster 2007; Li 1997 |
| **2** | **Adaptive Spatial Resolution** (B1) | 3–10× fewer harmonics on metallic lamellar gratings → ~O(N³) wall-clock win | med | Granet 1999; Vallius-Honkanen 2002 |
| **3** | **Circular truncation** (C1) + **normal-incidence symmetry halving** (D1) | Both small code, immediate accuracy + ~8× eig speed at normal incidence | low | Lalanne 1997 |
| **4** | **Eig reuse across λ/θ sweeps** (D2) | Speeds the dispersion sweeps the library runs most; cross-connects with the existing speed roadmap | med | — |
| **5** | **Matched coordinates** (B2) = A1+B1 | SOTA for hard plasmonics; natural once 1 & 2 land | med | Weiss 2009 |
| **6** | **Spectral filtering of reconstructed field** (E2) + **extrapolation estimator** (E1) | Improves the multi-order field-reconstruction feature (pairs with the v5.5.2 flux-weight fix); cheap convergence estimator | low | Edee 2011 (Gegenbauer); std. |
| **7** | **PMM / B-spline modal method** (C3) | Exponential-convergence 1-D deep-metal solver — a separate solver, not an acceleration | high | Edee 2011; Edee-Guizal |
| **—** | C-method (B3), non-orthogonal basis (C2) | Situational (blazed/echelle; hexagonal lattices) — only if those geometries enter scope | — | Chandezon 1982; Lalanne 1997 |

**Fast follow regardless:** fix the `rcwa_efficiency_2d` docstring (rcwa.py:1232-1233) so it no longer claims an FFF rule that isn't implemented — either build A1 or correct the text. The header already cites Li 1997 (33-34) for a 2-D rule the code doesn't use.

---

## Part 5 — Notes & caveats

- **Dielectrics are already fine.** Laurent converges fast for low-contrast dielectric structures; do not over-invest in factorization rules for that regime. The metal/TM/high-contrast cases are where A/B/C earn their keep.
- **Convergence (N) vs wall-clock are distinct axes.** A1/B1/C3 reduce the N needed for a target accuracy (and *because* the eig is O(N³), reducing N is the dominant wall-clock lever). D1/D2/D4 reduce cost at fixed N. They compound.
- **FFF and analytic form-factors are orthogonal** — Lumenairy has the latter (in-plane staircase removal) but still lacks the former (normal-direction discontinuity factorization). Both are needed for 2-D metals.
- **Validation:** any new factorization/basis must be checked against the existing independent oracles (grcwa / inkstone / S4 / RETICOLO are used as validation-only per header 39-40) on a TM metallic grating convergence-vs-N curve — the canonical test is a deep gold lamellar grating in TM, where Laurent plateaus and Li/FFF/ASR converge.

---

## Part 6 — References

**Confirmed in the module header** (`rcwa.py` lines 22-37) — already part of the codebase:
- M. G. Moharam, E. B. Grann, D. A. Pommet, T. K. Gaylord, "Formulation for stable and efficient implementation of the RCWA of binary gratings," *JOSA A* **12**, 1068 (1995); enhanced-transmittance companion, *JOSA A* **12**, 1077 (1995).
- L. Li, "Use of Fourier series in the analysis of discontinuous periodic structures," *JOSA A* **13**, 1870 (1996) — the **inverse rule** (TM/metals).
- L. Li, "Formulation and comparison of two recursive matrix algorithms for modeling layered diffraction gratings," *JOSA A* **13**, 1024 (1996) — S-matrix recursion.
- L. Li, "New formulation of the Fourier modal method for crossed surface-relief gratings," *JOSA A* **14**, 2758 (1997) — **2-D factorization** (cited but not implemented).
- R. C. Rumpf, "Improved formulation of scattering matrices...," *PIER B* **35**, 241 (2011).

**Advanced-technique references** (from FMM literature / domain knowledge — author-year-journal-topic mappings are reliable; *exact volume/page should be verified before formal citation*, and I can run a verified deep-research pass to pin them down):
- E. Popov & M. Nevière, "Maxwell equations in Fourier space: fast-converging formulation for diffraction by arbitrary-shaped, periodic, anisotropic media," *JOSA A* **18**, 2886 (2001) — **FFF**.
- T. Schuster, J. Ruoff, N. Kerwien, S. Rafler, W. Osten, "Normal vector method for convergence improvement using the RCWA for crossed gratings," *JOSA A* **24**, 2880 (2007) — **normal-vector field**.
- G. Granet & B. Guizal, "Efficient implementation of the coupled-wave method for metallic lamellar gratings in TM polarization," *JOSA A* **13**, 1019 (1996); P. Lalanne & G. M. Morris, *JOSA A* **13**, 779 (1996) — independent TM fast-factorization.
- G. Granet, "Reformulation of the lamellar grating problem through the concept of adaptive spatial resolution," *JOSA A* **16**, 2510 (1999) — **ASR**.
- K. Vallius & M. Honkanen, "Reformulation of the Fourier modal method with adaptive spatial resolution: application to multilevel profiles," *Opt. Express* **10**, 24 (2002) — 2-D ASR.
- T. Weiss, G. Granet, N. A. Gippius, S. G. Tikhodeev, H. Giessen, "Matched coordinates and adaptive spatial resolution in the Fourier modal method," *Opt. Express* **17**, 8051 (2009) — **matched coordinates** (ASR+FFF).
- P. Lalanne, "Improved formulation of the coupled-wave method for two-dimensional gratings," *JOSA A* **14**, 1592 (1997) — 2-D formulation / **truncation**.
- K. Edee, "Modal method based on subsectional Gegenbauer polynomial expansion for lamellar gratings," *JOSA A* **28**, 2006 (2011); Edee & Guizal (B-spline modal method) — **PMM**.
- J. Chandezon, M. T. Dupuis, G. Cornet, D. Maystre, "Multicoated gratings: a differential formalism applicable in the entire optical region," *JOSA* **72**, 839 (1982) — **C-method**.

---

## Appendix — One-line state summary

```
1-D metals/TM:  Li inverse rule PRESENT + correctly placed (rcwa.py:671-756) -- good.
2-D metals/TM:  LAURENT ONLY (rcwa.py:1249-1300) -- the slow/wrong case. FFF claimed
                (docstring 1232-1233) + Li-1997 cited (header 33-34) but NOT built.
No ASR, no circular truncation, no symmetry halving, no PMM, no eig-reuse, no filtering.

Priority adds: (1) FFF/normal-vector 2-D  (2) ASR  (3) circular trunc + symmetry halving
               (4) eig-reuse across sweeps  (5) matched coords  (6) field filtering  (7) PMM
Dielectrics already converge fast -- these target metals / high-contrast / TM.
```
