# AUDIT — PMM in-plane oblique: the union-grid wall-collision pathology

**Date:** 2026-07-28 · **Library:** lumenairy 5.31.0 · **Reporter:** metasurface LC-QWP out-coupler campaign (exp18–exp22)
**Severity:** **P1 (silent-wrong, user-facing)** — energy-clean results that are wrong by up to 20×, with no warning on the affected construction path.

---

## 1. Executive summary

A 1-D `PMMStack` built from a **2° tapered** `SegmentStackGeometry` returns **in-plane (φ=0) oblique
extinction ratios that do not converge in polynomial `degree`** — deg 6/8/10 give mutually inconsistent
answers (spread up to **91 %**) while total power stays clean and no warning fires. The **out-of-plane
(φ=90) cut of the identical stack converges to 0.03 %**, so the defect is not the conical path, the
materials, or the geometry description.

**Root cause (confirmed by measurement):** the solution depends on **how the shared union grid was
built** (`_pmm_union_grid`, `elements/pmm/_core.py:3300`) — specifically on `min_feature`, the threshold
that merges near-coincident cross-layer walls. A tapered staircase offsets each slice's walls, so the
union of all layers' walls contains many closely-spaced cross-layer pairs. `min_feature` **defaults to
`period × 1e-5` = 0.007 nm on a 700 nm period — ~200× smaller than the ~1.5 nm the library's own docstring
recommends** — so on this geometry it merged *nothing*, and the resulting grid gives an answer that is not
degree-convergent.

> **Mechanism, stated honestly.** The initial hypothesis for this audit was a single degenerate
> (`J = ½(x_r − x_l) → 0`) element. **The measured grid statistics do not support that** and it is
> withdrawn: at the default the union grid has 61 elements with a **median width of 1.095 nm and a
> thinnest element of 0.214 nm** — only ~5× below the median, not a near-zero-width sliver (and the
> device's own 1 nm Ta liner legitimately produces ~1 nm elements). At `min_feature = 1.5 nm` the grid
> instead has **43 elements with a median of 5.206 nm**. What is established by measurement is the
> *causal* dependence on `min_feature` and the resulting loss of degree-convergence; the precise
> conditioning mechanism that maps one grid to a corrupted deep null is **not** pinned down here and
> remains open. The recommendations below deliberately do not depend on it.

**Fix (measured):** setting `min_feature = 1.5e-9` m collapses the degree spread from **91 % → 0.1 %**
and is **2.1× faster**. This is the remedy the library already documents; it is simply not reachable by
default, and the guard meant to detect the failure cannot fire on this construction path.

---

## 2. Symptom

Design: coated Cu pillar out-coupler, period 700 nm, λ = 1310 nm, `sidewall_deg = 2.0`,
`n_slice = n_slice_A = n_refl_slice = 2`, 38 layers, absorbing Cu substrate, anisotropic (planar) LC.
Metric: zeroth-order reflection Jones → `reflective_outcoupling`, ER = peak/null over the LC switch, with
the LC angles frozen at their normal-incidence values.

In-plane ER at θ = 8°, default `min_feature`:

| degree | ER (θ8) |
|---|---|
| 6 | 41.363 |
| 8 | 3.822 |
| 10 | 3.674 |

*(a second harness, differing only in the LC-search grid, gave 60.1 / 2.2 / 41.1 at the same three
degrees — i.e. the result is not even stable across equivalent call sites.)*

Out-of-plane ER at θ = 8°, same stack, same degrees: **14.418 / 14.380 / 14.376 (0.03 %).**

Total power and peak coupling stay physical throughout (peak 0.82–0.91), so no energy tripwire fires.
This is precisely the class the library names elsewhere as **"passive-but-wrong"**.

---

## 3. Root cause

`_pmm_union_grid` builds one shared nodal grid as the union of every layer's wall positions. Its own
docstring states the failure mode and even the geometry:

> "Distinct layers' staircase walls collide at offsets far above float noise (**a 2-deg taper** at
> n_slices=8 puts walls **~1.2 nm apart** → near-zero-width union elements → **a passive-but-wrong or
> blowing-up solve**)."

The mitigation is the `min_feature` snap: cross-layer wall pairs closer than `min_feature` are merged to
their midpoint. But the default is set in `PMMStack.__init__` (`elements/pmm/stack.py:194`):

```python
self.min_feature = (float(period) * 1e-5 if min_feature is None else float(min_feature))
```

`700 nm × 1e-5 = 0.007 nm`. The colliding pairs on this geometry sit between ~0.01 nm and ~0.5 nm apart —
**above** the default threshold, therefore never snapped, and each one seeds a `J → 0` element.

`PMMStack.__init__`'s own comment recommends the working value and quantifies the benefit:

> "measured 5.7× (321 s → 56 s) on an ns8 coated taper at **min_feature = 1.5e-9**, at a ~1.6 %
> geometry-perturbation cost (±0.75 nm wall moves)."

### Decisive experiment

`min_feature` × `degree` matrix, same geometry, ffo 21 (`min_feature_matrix.py`):

| `min_feature` | ER θ8 (deg 6 / 8 / 10) | degree spread θ8 | deg-6 runtime |
|---|---|---|---|
| **0.007 nm** (default) | 41.363 / 3.822 / 3.674 | **91.1 % SCATTERED** | 147 s |
| 0.5 nm | 57.694 / 36.139 / 36.023 | 37.6 % SCATTERED | 78 s |
| **1.5 nm** (documented) | **54.889 / 54.834 / 54.833** | **0.1 % CONVERGED** | **71 s** |
| 3.0 nm | 18.368 / 11.328 / 18.340 | 38.3 % SCATTERED | 57 s |

At `min_feature = 1.5 nm` every tested angle converges (θ0 0.2 %, θ8 0.1 %, θ10 0.0 %) **and** the solve
is 2.1× faster. At 3.0 nm the snap over-perturbs the geometry and conditioning degrades again — the
usable window is bounded on both sides.

### Staircase convergence: `min_feature` cures `degree`, and exposes the n_slice limit

Fixing `degree`-convergence is necessary but not sufficient: the answer must also be stationary in
`n_slice`, the knob that actually approaches the physical taper (`O(1/n_slices^2)`). Measured at
`min_feature = 1.5 nm` (in-plane ER, W_hiER2):

| n_slice | θ0 (deg6 / deg8) | θ8 (deg6 / deg8) | θ10 (deg6 / deg8) | degree-spread |
|---|---|---|---|---|
| 1 | 17.442 / 17.470 | 12.506 / 12.525 | 10.398 / 10.413 | ≤ 0.16 % |
| 2 | 49.640 / 49.555 | 54.889 / 54.834 | 54.702 / 54.679 | ≤ 0.17 % |
| 4 | 49.964 / 49.848 | 55.959 / 55.877 | 56.233 / **47.400** | θ10 **15.7 %** |

Two results:

1. **The staircase is converging.** `n_slice = 1` is degenerate (one slab, no taper resolution) and must be
   excluded; on the meaningful `2 → 4` step, five of six entries agree to **0.6–2.8 %**, consistent with the
   documented `O(1/n_slices^2)` law. At `n_slice = 2` the residual staircase error is therefore ~1–3 % —
   an engineering-usable number, versus the 20× corruption at the default `min_feature`.
2. **But refining the staircase re-breaks the conditioning.** At `n_slice = 4`, θ10, the degree spread jumps
   to **15.7 %** where every other cell is ≤ 0.23 %: `min_feature = 1.5 nm` is sufficient at `n_slice = 2`
   and **not** at `n_slice = 4`.

This is the structural vise, now measured: converging the taper needs MORE slices, but more slices grow the
shared union grid (cost `O(n^3.4)`) and multiply cross-layer wall collisions, so conditioning degrades as
geometry fidelity improves. `min_feature` mitigates but does not remove it, and it must be re-tuned per
`n_slice`. Accuracy for tapered devices is therefore capped by the SHARED-GRID ARCHITECTURE, not by the
staircase law. See R-6.

### Corroboration

Removing the thin conformal coats (`ta = al = sin = 0`), which eliminates most cross-layer wall
collisions, makes the **default** `min_feature` converge: deg 6/8/10 → 2.811 / 2.800 / 2.795 (0.5 %).
The solver core is therefore sound; the defect is the union-grid geometry conditioning.

### Why in-plane fails and out-of-plane does not

Both cuts share the union grid, so both are perturbed identically. The in-plane null is far deeper
(ER ≈ 50–60 ⇒ null ≈ 1.7 %) than the out-of-plane null (ER ≈ 8–26). ER is a ratio with the null in the
denominator, so the same absolute conditioning error is amplified ~4–7× more in-plane. The asymmetry is a
**conditioning-sensitivity** effect, not a difference in physics paths.

---

## 4. Secondary findings

**S-1 [P1] — the guard cannot fire on `SegmentStackGeometry`-built stacks.**
`solve(stabilize='slices')` is the documented tripwire for exactly this pathology. On a stack built via
`SegmentStackGeometry.add_ridges → to_pmm_stack` it returns:

> "no taper builder recorded on this stack (hand-added layers cannot be re-sliced); the consensus check
> was skipped."

The consensus check only works for `PMMStack.add_tapered_ridges` / `add_tapered_grating`, which record a
taper recipe. **Every geometry built through the `SegmentStackGeometry` route — the documented
device-geometry path — is silently unprotected.** Suggested fix: either record an equivalent re-slice
recipe in `to_pmm_stack`, or make `stabilize='slices'` fall back to a *union-grid* consensus (perturb
`min_feature` and compare the zeroth-order Jones), which needs no builder recipe.

**S-2 [P2] — the default `min_feature` is inappropriate for tapered stacks.**
`period × 1e-5` is described as "far above float noise, far below intentional features", which is true for
*vertical* stacks. For any tapered stack the relevant scale is the **wall-offset per slice**,
`≈ (thickness/n_slices) · tan(sidewall)`, which is nanometres — three orders larger. Suggested fix: derive
the default from the recorded taper geometry when a taper is present, or warn when the union grid contains
elements below some fraction of the mean element width.

**S-3 [P3] — `min_feature` is a silent accuracy knob.**
The converged ER depends on it: θ0 gives 58.8 / 52.6 / 49.6 / 15.5 at 0.007 / 0.5 / 1.5 / 3.0 nm. It is
documented as a *cost* knob ("ALSO THE COST KNOB"), but it is equally an accuracy knob, and nothing warns
when the snap has moved walls far enough to change the answer. Suggested fix: report the total snapped
displacement and warn above a threshold.

**S-4 [informational] — RCWA cannot cross-validate this structure.**
`to_rcwa_stack` on the identical geometry never converged: ER at θ0 = 1.1 / 0.2 / 0.7 / 3.7 / 6.4 / 22.3 at
n_orders 21 / 31 / 41 / 81 / 151 / 251, against a PMM value of ~50–59, with peak coupling 0.13–0.33 vs
0.824. Expected: the 1 nm Ta liner and 5 nm conformal coats are far below the Fourier floor `≈ P/2N`.
This is a legitimate, documented strength of PMM (laterally exact, no Gibbs floor) — but it means **no
independent in-repo oracle exists for this device class**, which is why the defect survived. A
`min_feature`-perturbation self-consistency check is the practical substitute.

---

## 5. Literature context

The modules cite Granet (45×), Edee (7×), Popov & Nevière (5×), Li 1997/1999, Lalanne — the correct
lineage for a subsectional polynomial modal method, and the formulation details that were checked
(div-conforming `1/ezz` placement "between the derivatives", Granet 2023 Eq. 16–18 / Popov–Nevière
App. B; Li's inverse rule) are consistent with them.

Two points from that literature bear on this defect:

1. **The staircase itself is not the problem.** The library's own measurement — z-staircase error falling
   `3.82e-3 → 5.70e-6` for `n_slices` 8 → 256, a factor 3.9 per doubling, i.e. `O(1/n_slices²)`, validated
   against an RCWA twin at `n_slices = 768` — matches the classical expectation. The observed failure is
   *not* staircase truncation error; it is conditioning of the union grid the staircase induces.
2. **Coordinate transformation is the literature's answer to profiled gratings.** Staircasing slanted or
   tapered profiles is long known to be the weak approach, particularly for metallic gratings; the
   Chandezon C-method and Granet-style coordinate transformations treat the profile exactly. lumenairy
   already exploits this for the pure-shear sub-case — `add_sheared_grating` (v5.31) emits **one exact
   slanted layer, no z-staircase**, because `u = x − z·tanφ` keeps the modal coefficients z-independent.
   The **symmetric trapezoid has no such transformation** (walls converge rather than translate), which is
   why it must be staircased today and why the union grid grows. The library's own cost note names the
   right destination: *"for a scalable no-floor taper prefer a single covariant taper-metric layer (a
   roadmap item)."* **This audit is direct evidence that that roadmap item is worth doing** — it removes
   the union-grid collision class entirely rather than mitigating it.

---

## 6. Recommendations

### For the library

| # | Priority | Recommendation |
|---|---|---|
| R-1 | **P1** | Make `stabilize='slices'` (or an equivalent) work for `SegmentStackGeometry`-built stacks — e.g. a `min_feature`-perturbation consensus that needs no taper recipe. Today the documented guard is unreachable on the documented geometry path. |
| R-2 | **P1** | Warn when the union grid contains elements far below the mean element width (the direct observable of this pathology), regardless of whether a snap occurred. Energy checks provably cannot catch it. |
| R-3 | **P2** | Make the `min_feature` default taper-aware: scale from `(thickness/n_slices)·tan(sidewall)` when a taper is present, instead of `period × 1e-5`. |
| R-4 | **P2** | Document `min_feature` as an **accuracy** knob, not only a cost knob, and report cumulative snapped displacement. |
| R-5 | **P3** | Advance the covariant taper-metric layer (general trapezoid, the roadmap item) — the structural fix that removes the collision class, as `add_sheared_grating` already does for pure shear. |
| R-6 | **P2** | **Break the shared-union-grid coupling: give each layer its own element grid, with a projection / mortar at the interfaces.** This is the tractable half of R-5 and addresses the measured cap directly. Today every slice's walls enter ONE global grid, so refining a taper degrades every layer's conditioning and costs `O(n_slices^3.4)` — the vise measured in §3: `n_slice` 2 → 4 improves the geometry (0.6–2.8 % shift) but breaks degree-convergence at one angle (15.7 %). With per-layer grids a slice's walls stay local, cross-layer collisions cannot form, `min_feature` stops being an accuracy knob, and `n_slice` scales. Standard non-conforming spectral-element practice — engineering, not research, unlike R-5. |

### For users of tapered `PMMStack` geometry (immediately actionable)

1. **Set `min_feature` explicitly** — do not accept the default on a tapered stack. Validate by sweeping
   it: the usable value is the one where the answer is stationary in **both** `degree` and `min_feature`.
2. **Always converge in `degree`, not just in `far_field_orders`.** In this campaign `ffo` was fully
   converged at 11 while `degree` was the live variable — checking only `ffo` gives false confidence.
3. **Treat deep-null ratios as ill-conditioned.** Extinction ≳ 30 at oblique incidence should be reported
   with a conditioning check or as a bound.

---

## 7. Impact on the reporting campaign

- **Unaffected:** all normal-incidence extinction (0.2 % across degrees), all reflectivity / peak-coupling
  values, and the entire out-of-plane angular dataset (0.03 %). The campaign's central physical
  conclusion — the pillar is out-of-plane-fragile, i.e. the axis flip — rests on the out-of-plane data and
  **stands**.
- **Affected:** in-plane oblique extinction, which must be recomputed at `min_feature = 1.5e-9`. Under the
  corrected setting the in-plane response is smooth and *more* tolerant than previously reported
  (ER ≈ 49.6 / 54.9 / 54.7 at θ = 0 / 8 / 10°), replacing a jagged curve that was a conditioning artifact.
- **Residual uncertainty, stated:** at the campaign's `n_slice = 2` the recomputed values carry a **~1–3 %
  staircase uncertainty** (§3, the `n_slice` 2 → 4 step). They should be quoted with that band, not as
  exact. Driving it lower is blocked by the vise above, i.e. by R-6 / R-5 — not by anything the user can
  set. For the campaign's conclusions (which turn on 10 dB-scale differences) a 3 % band is immaterial.

---

## 8. Reproduction

All scripts in `Metasurface_QWP/experiments/exp21/`:

| script | purpose |
|---|---|
| `min_feature_matrix.py` | the decisive `min_feature` × `degree` matrix (§3) |
| `oblique_convergence_v531.py` | degree/ffo ladder showing in-plane non-convergence vs OOP convergence |
| `rcwa_crosscheck.py` | RCWA oracle attempt (§ S-4) |
| `pmm_pathology_probe.py` | `stabilize='slices'` inertness + simplified-structure control |

Geometry: `exp11_common.coated_pillar_geometry(w_c=340, w_f=170, g=95, H=385, t=70, pillar_w=90,
sio2_gap=80, d_refl=120, d_below=120)`, `n_slice=n_slice_A=n_refl_slice=2`, λ = 1310 nm,
`n_superstrate = 1.50`, `n_substrate = sqrt(eps_Cu)`.
