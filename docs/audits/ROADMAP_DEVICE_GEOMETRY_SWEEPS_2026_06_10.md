# Roadmap — device-geometry builders, sweeps & diagnostics (2026-06-10)

**Provenance.** The `pbs_qwp_mirror_sim` application (two-tooth tapered Cu/SiCN LC
out-coupler, 1 nm Ta + 6 nm Al2O3 coatings, anisotropic LC fill) finished migrating to
`PMMStack` on v5.14.0 and was re-validated (coated PMM at `ta=0/al=0` reproduces the
nannos-checked bare device bit-for-bit; metal-TM adjudication confirms PMM-at-low-degree
as the reference). This doc inventories **what the application script still has to
hand-build**, proposes the library items that would eliminate it, and assesses each gap
across all four solver families: **PMM-1D** (`PMMStack` + segment solvers), **PMM-2D**
(`PMM2DStack` + cell/jones solvers), **RCWA-1D** (`RCWAStack` 1-D + segment solvers),
**RCWA-2D** (`RCWAStack` 2-D cells + direct 2-D solvers). Complements (does not
duplicate) `docs/pmm_roadmap_v5_14.md` and `docs/rcwa_roadmap_v5_14.md`.

**The driving observation: the solvers are no longer the bottleneck — geometry
construction is.** Of `src/pmm_taper.py`'s ~280 lines, ~140 are geometry arithmetic
(boundary lists per staircase slice, coating offsets, sliver avoidance, guards) and ~40
are material/sweep plumbing; the solver calls themselves are ~10. Three of the four
geometry bugs found during the port (left-anchored taper drift, sub-nm edge slivers,
occluded liner) were in hand-built geometry, none in solver calls.

---

## 0. Coverage matrix

| Capability | PMM-1D | PMM-2D | RCWA-1D | RCWA-2D |
|---|---|---|---|---|
| 1. Multi-ridge/pillar tapered builder | ✗ (`add_tapered_grating` = 1 centred ridge) | ✗ (`add_tapered_pillar` = 1 pillar) | ✗ (`add_tapered_grating` = 1 ridge, `shear=`) | ✗ (no 2-D tapered builder at all) |
| 2. Conformal-coat / liner / fill geometry ops | ✗ | ✗ | ✗ | ✗ |
| 3a. Physical wall-snap in the union grid | ✗ (`tol=1e-9` fractional = float-noise only) | n/a (per-layer grids, no union) | n/a (Fourier) | n/a |
| 3b. Passive-but-wrong staircase detection | ✗ (tripwire fires only on `R+T>1`) | ✗ | n/a | n/a |
| 4. Native exact symmetric-trapezoid layer | ✗ (slant/parallelogram only) | ✗ (native slant itself is roadmap #5) | ✗ (research-grade for FMM) | ✗ |
| 5a. Dispersive Jones sweep, single layer | partial (`pmm_jones_1d_vs_wavelength` = binary only; **no segments variant**) | ✗ (`pmm_jones_2d` has no sweep) | ✓ (`rcwa_jones_vs_wavelength` + `_segments`) | ✗ (`rcwa_jones_2d` has no sweep) |
| 5b. Dispersive **Jones** sweep, stack | ✗ (`solve_vs_wavelength` = non-dispersive, R/T only) | ✗ (same) | partial (5.14.1 sweep is dispersive but R/T only) | partial (same path) |
| 6. Internal field + per-layer absorption | ✗ | ✗ | ✓ (`retain_internal` → `internal_field`, `layer_absorption` incl. tensor quadratic form) | ✓ (same `RCWAResult`) |
| 7. Stack geometry viewer | ✗ | ✗ | ✗ | ✗ |
| 8. Prepared solve w/ swappable material slot | ✗ | partial (`prepare_pmm_2d` hoists for λ only) | ✗ | partial (`PreparedRCWA2D`, λ only) |

Already-exists (application should adopt, **not** gaps): `reflective_outcoupling`
(top-level, differentiable), `interdigitated_grating_segments` (vertical single-layer
multi-tooth — literally the device's lateral pattern), `grating_segments`,
RCWA `layer_absorption`/`internal_field`, `pmm_graded_segments` (lateral grading).

---

## 1. Multi-ridge / multi-pillar tapered builders

**Gap.** Every tapered builder takes exactly ONE centred feature
(`PMMStack.add_tapered_grating(eps_ridge, eps_groove, duty_*)`;
`PMM2DStack.add_tapered_pillar(x/y_bounds_*)`; `RCWAStack.add_tapered_grating(...,
shear=)`). A unit cell with several features per period — two different-width teeth
here; interdigitated electrodes generally — forces the application to hand-write
per-slice boundary lists. The center-alignment is the subtle part: a width-sequence
construction **left-anchors each ridge and drifts its center as it tapers** (a real bug
found in this port, worth ~3% in out-coupling); each ridge must narrow about its own
fixed center.

**Sketch (1-D).**
```python
stack.add_tapered_ridges(thickness,
    ridges=[(center, w_top, w_bottom, eps), ...],   # absolute positions, nm-like units
    eps_groove=...,                                  # scalar or (3,3); fills remainder
    n_slices=8, rule="midpoint")
```
Internally: per slice, emit the boundary list `sorted([c ± w(z)/2 for each ridge])` with
wrap handling; reuse the existing staircase/tripwire machinery. The existing
single-ridge builder becomes the 1-element case. `interdigitated_grating_segments`
already encodes the lateral pattern — this is its z-aware tapered generalization.
2-D mirror: `PMM2DStack.add_tapered_pillars([...])` (the per-layer-grid design means no
union-grid penalty); RCWA-2D currently has **no** tapered builder, so the same ridge-list
API applied to `eps_cell` pixelation would close two gaps at once.

**Validation.** The two-tooth bare device: `w_c=130, w_f=220, g=100, H=350, t=70`,
2°, P=550. Builder output must match the hand-built reference (`tapered_jones`):
vertical φ0 **82.3% / Rx 0.923**; and the center-of-mass of each ridge must be
z-independent (the drift detector).

**Effort.** Small. Pure assembly code on existing machinery.

---

## 2. Conformal-coating / segment-stack geometry algebra

**Gap (all four families).** The most delicate, error-prone application code is coating
logic: a 6 nm Al2O3 film following tooth tops + sidewalls + gap floor; a 1 nm Ta liner
on *specific* interfaces only (under teeth, on buried column walls — NOT gap-facing
walls); LC filling the remainder. Three review rounds were spent on exactly this. It is
pure geometry algebra on a z-stack of 1-D segment layers (or 2-D cells), independent of
the solver consuming it.

**Sketch.** A tiny geometry layer operating on `[(thickness, bnds, mats), ...]`
cross-sections (materials as small ints/keys, eps resolved at solve time):
```python
g = SegmentStackGeometry(period)
g.add_ridges(...)                                  # item 1's builder, geometry-side
g.coat(t=6, mat="Al2O3", where="tops|walls|floor") # conformal offset of named surfaces
g.line_interface("Cu", "SiCN", t=1, mat="Ta")      # liner wherever Cu meets SiCN
g.fill("LC")                                       # remainder up to a cap plane
g.to_pmm_stack(...) / g.to_rcwa_stack(...) / g.plot(...)
```
Each operation is interval arithmetic per z-band plus band splitting at new interface
depths — no solver knowledge. This also gives single-source-of-truth for free: the same
object feeds the PMM solver, the RCWA cross-check, and the viewer (item 7) — the
drift-proofing the application had to build by hand.

**Validation.** The coated device above: layer dump must reproduce the hand-built
9-layer stack (cap Al2O3=w+2t walls; floor; under-tooth Ta; column liners) and solve to
φ0 **77.9% / Rx 0.892** (deg 10, ns 4, n_sup 1.50) via `to_pmm_stack`.

**Effort.** Medium (the interval algebra + a wrap-aware surface tracker). Highest
hand-crafting payoff of any item.

---

## 3. Staircase robustness: physical wall-snap + passive-but-wrong detection

**Evidence (new, v5.14.0, this device).** The union-grid merge (`_pmm_union_grid`,
`tol=1e-9` fractional ≈ sub-pm) only removes float-noise duplicates. Genuinely distinct
staircase walls collide earlier: at `n_slices=8` the 2° taper's per-slice offset is
1.20 nm and:

| config | oc | Rx | energy tripwire |
|---|---|---|---|
| ns=4, deg 10 (converged) | 77.9% | 0.892 | quiet |
| ns=6, deg 10 | 78.0% | 0.892 | quiet |
| ns=8, deg 10 | 29.9% | **1.121** | **fires** ✓ |
| ns=8, deg 14 | **27.0%** | 0.892 | **silent** ✗ |
| ns=8, deg 18 | 88.6% | **1.121** | **fires** ✓ |

There is **no degree cure**: degrees 10/14/18 give three *different* wrong answers,
non-monotonically (loud gain → silent passive-but-wrong → loud gain). The ns8/deg14 row
is the dangerous one — an energy-clean, plausible-looking, completely wrong answer the
`R+T>1` tripwire cannot catch (the tripwire's own "raise degree" advice does not hold
in this regime). Two library-side fixes:

- **(a) Physical wall-snap.** A `min_feature=` (absolute length or fraction) on the
  union builder / staircase builders: snap union walls closer than it (midpoint eps
  reassignment already handles labeling), emit one warning naming the merged pairs.
  Default conservative (say period×1e-5); builders pass something physically motivated.
  Note the asymmetry: an *intentional* 1 nm liner is one thin element on an otherwise
  coarse grid (fine — the device solves with its 1 nm Ta walls at ns≤6); the pathology
  is specifically *near-coincident pairs from different layers*. Snap distinct layers'
  walls together; never thin a single layer's own feature.
- **(b) Consistency tripwire.** The degree-scan `stabilize` consensus pattern applied to
  `n_slices` (solve at n and n±1-ish, compare zeroth-order jones): catches
  passive-but-wrong staircases that energy cannot. Opt-in (`stabilize="slices"`), since
  it multiplies cost.

PMM-2D note: `PMM2DStack` keeps per-layer grids (no union), so (a) is moot there, but
each layer's own staircase can still produce thin-element pathologies inside
`add_tapered_pillar` at large `n_slices` — (b) applies. RCWA staircases are immune
(Fourier basis, no nodal elements) — n/a.

**Validation.** The table above is the regression test: (a) must keep ns≤6 byte-stable
and make ns=8 either correct or loudly merged; (b) must flag ns8/deg14.

**Effort.** (a) small; (b) small-medium (pattern exists).

---

## 4. Native exact symmetric-trapezoid layer (PMM-1D first)

**Gap.** The slanted machinery solves *parallelogram* shear exactly (both walls tilt the
same way); a fabrication taper is a *symmetric trapezoid* (walls tilt opposite ways,
feature narrows with depth). Today that means z-staircase + items 3's failure class +
union-grid growth (`n_slices` × walls × layers — the device's 9-layer coated stack is
~80 union elements/layer, and PMM lost its 3× speed edge over RCWA exactly here). The
`add_tapered_grating` cost note already names the fix: "a single covariant taper-metric
layer (a roadmap item)". The natural map is per-feature linear width scaling
`u = (x - c)/w(z)` (Granet-style matched coordinates with a z-dependent metric, the
trapezoid analogue of the inclined-coordinate slant); like the slant it makes the modal
problem quadratic-in-q via a linear convection term — companion linearization and the
generalized S-matrix already exist.

This eliminates `n_slices` (no staircase error, no wall collisions, items 3a/3b moot for
1-D tapers) and shrinks the union (1 wall per tooth side instead of `n_slices`), so the
coated device would solve *faster* than its vertical equivalent rather than ~9× slower.

**Families.** PMM-1D: natural next step after the slant solver. PMM-2D: blocked on
native 2-D slant (existing roadmap #5; same Gegenbauer/coordinate machinery). RCWA-1D/2D:
no nodal grid to map — would need C-method/matched-coordinate FMM, research-grade and
probably out of scope; the staircase (immune to item 3) remains RCWA's path.

**Validation.** (i) vertical limit ↔ vertical layer byte-exact; (ii) the bare 2° device
vs the converged staircase (ns 4/6 agree at 77.9–78.0% coated, and the bare-device
staircase lineage); (iii) slant limit: trapezoid with both walls at the same angle ↔
existing slanted solver.

**Effort.** Large (research-grade, but with the slant solver as a worked precedent).
Highest physics payoff: removes a failure class rather than guarding it.

---

## 5. Wavelength sweeps: dispersive + Jones, stack-level; PMM segments parity

**Gap matrix.** Dispersive **Jones** sweeps top out at single-layer:
`rcwa_jones_vs_wavelength(_segments)` (RCWA-1D, callables ✓ jones ✓) and
`pmm_jones_1d_vs_wavelength` (binary only — **the segments variant doesn't exist**, an
asymmetry with RCWA). At stack level nobody returns jones, and the PMM stacks are
additionally non-dispersive (`PMMStack.solve_vs_wavelength`, `PMM2DStack.…`: fixed eps,
R/T only). `RCWAStack.solve_vs_wavelength` (5.14.1) got dispersive callables + NaN-row
robustness — the right pattern — but also returns R/T only. 2-D direct jones solvers
have no sweeps at all.

Any polarization-chain device (this one: out-coupling needs the 2×2 jones through
QWP/PBS) with metal layers (Cu/Ta strongly dispersive over ±100 nm) therefore
hand-loops wavelengths rebuilding the stack — which is what the application does.

**Sketch.**
1. `pmm_jones_1d_segments_vs_wavelength(...)` — trivial parity item (loop + callables,
   mirror of the RCWA segments sweep).
2. Material callables in `PMMStack.add_layer` / segment eps slots (the 5.14.1 GAP5
   recipe: materialize per λ, NaN-row + summary warning on unstable points), and a
   `jones` array in BOTH stack sweeps' returns:
   `solve_vs_wavelength(wls) -> (orders, R(n,2,N), T(n,2,N), jones(n,2,2))`.
   The stack already computes jones per solve — this is plumbing, not physics.
   Same addition to `RCWAStack.solve_vs_wavelength` (it materializes dispersive layers
   already; just emit the jones rows).
3. Honest-perf note stays: these are convenience/correctness wrappers (eig-bound), not
   speedups — which is exactly why the missing piece is the *API*, not performance.

**Validation.** Coated device, λ ∈ 1210–1410 nm step 15: per-λ
`coated_tapered_jones(wl=…)` loop (the application's current path, with McPeak Cu /
Werner Ta / IMEC SiCN / Al2O3 CSV loaders as callables) must equal the stack sweep
row-for-row; the 1295 nm point is pinned in the application's validation set.

**Effort.** Small (1) / small-medium (2). Best effort-to-payoff ratio in this doc.

---

## 6. Internal fields + per-layer absorption for the PMM family

**Gap.** `RCWAResult` has `internal_field` (GAP1) and `layer_absorption` (GAP6, incl.
the tensor quadratic form `Im(E*·eps·E)`, energy-conserving normalization). The PMM
family — 1-D and 2-D, direct and stacks — exposes **nothing inside the structure**: no
fields, no loss attribution. Consequence in practice: the Ta-liner redesign (the most
consequential design decision in the device's history — 33–58% absorption traced to the
liner sitting in the gap-plasmon field) had to be inferred from material-removal
experiments, and the only direct loss map available is the RCWA cross-check — which on
this metal is provably under-resolved (its own 5.14.1 monotone-trend warning fires).
The accurate engine is the one that can't show where the power goes.

**Sketch.** PMM holds *nodal* fields per layer — better suited to this than RCWA's
Fourier reconstruction. Mirror the RCWA API: `PMMStack.solve(retain_internal=True)`
storing per-layer modal amplitudes; `layer_absorption()` integrating
`Im(E*·eps·E)` per element (GLL quadrature is exact per element — the integral is a
weighted sum over nodes already in hand); optionally per-MATERIAL attribution (the
segment eps labels make "how much in the Ta vs the Cu walls" a one-liner — finer than
RCWA's per-layer split, and the question the application actually asks).
`internal_field(z)` from the same stored amplitudes. 2-D mirror on `PMM2DStack`.

**Validation.** (i) Σ layer_absorption = 1 − ΣR − ΣT to 1e-12 (construction); (ii)
coated device φ0: total A ≈ 0.108 (= 1 − Rx 0.892), with the Ta-vs-Cu split
qualitatively matching the RCWA layer_absorption map and the historical
material-removal numbers (sidewall-Ta variant: A 0.33 φ0 / 0.58 φ90, mostly Ta).

**Effort.** Medium. Physics is already in hand (fields exist internally); this is
exposure + bookkeeping.

---

## 7. Stack geometry viewer

**Gap (all four).** No stack can draw itself. `lumenairy.analysis.plotting` is
field-plotting only. The application built a 30-line exact-rectangle renderer
(`view_coated_1d`) for the "the picture IS the model" guarantee — after a real incident
where the figure and the solver geometry silently diverged. Every stack knows its
geometry exactly; `PMMStack` especially (analytic walls → crisp 1 nm features, no
pixelation).

**Sketch.** `PMMStack.plot_geometry(ax=None, material_names=None)` — one rectangle per
(layer × segment), substrate/superstrate bands, legend from eps identity (or names).
`RCWAStack`: imshow of the pixel/tensor cells per layer. `PMM2DStack`: per-layer 2-D
cell maps + an optional z-slice strip. Name it `plot_geometry` (NOT `plot_cross_section`
— that name is taken by the field cut in `analysis.plotting`).

**Validation.** Visual regression on the coated device against the application's
`figures/coated_model_view.png`.

**Effort.** Trivial-small. Disproportionate trust payoff (pairs with the tripwires:
*see* what you solved).

---

## 8. Prepared stacks with swappable material slots

**Gap.** All `prepare_*` machinery hoists geometry for **wavelength** sweeps only. The
equally common loop is a *material* sweep at fixed λ: the device's LC director tuning
curve re-solves the stack 13× per (geometry, λ) when only the LC tensor changes —
re-assembling the union grid, re-running SEM assembly, and re-eiging the 4–5 LC-free
layers (Ta layer, Al2O3 floor, column slices) whose operators are bit-identical across
the sweep. Director/field tuning is the defining operation of an LC device.

**Sketch.** Material *keys* in segments (`(width, "LC")` + a `materials=` dict at
solve), then `prepared = stack.prepare(); prepared.solve(materials={"LC": tensor})`
re-eigs only layers whose key set intersects the override. Composes with item 5 (a
key's value may be a λ-callable) and the existing pmm-roadmap #2 (uniform half-space
eig share) — the same "what actually changed" bookkeeping. Honest-perf note: saving is
the LC-free layers' eigs + all assembly; for this device ~40% of a solve, not an order
of magnitude.

**Validation.** 13-point φ-curve on the coated device: prepared path equals the rebuild
path to 1e-12 per point; wall-clock reduction ≈ the LC-free eig fraction.

**Effort.** Medium. The factorized-assembly work (5.14 P1) did the hard part; this is
cache keying.

---

## 9. Not gaps — application adoption notes (for completeness)

- `reflective_outcoupling(jones, qwp_angle=…)` exists, differentiable; the application
  still hand-rolls its QWP chain (historical). Adopt after one convention cross-check.
- `interdigitated_grating_segments` exists and names this exact device pattern; the
  segments format feeds `PMMStack.add_layer` directly. Adopt for vertical single-layer
  studies; item 1 is its tapered generalization.
- RCWA `layer_absorption`/`internal_field` exist and work today — usable for
  *qualitative* loss maps on this device (with the under-resolution caveat) until
  item 6 lands.
- Tabulated-material helpers (`Material.from_csv(wl_unit=…)`,
  `Material.from_refractiveindex(shelf, book, page)`) are arguably application
  territory, but every project re-writes the same `np.interp` loaders (this one wrote
  four); a 30-line helper would end that. Listed last deliberately — smallest scope,
  least core.

---

## Reference numbers for validation (v5.14.0, this device)

Bare vertical (`w_c=130, w_f=220, g=100, H=350, t=70`, n_sup=1.0, φ0, deg 10):
**oc 82.3% / Rx 0.923** (= nannos-validated lineage; coated path at ta=0/al=0
reproduces bit-for-bit; φ90 0.6%). Coated (Ta 1 / Al2O3 6, n_sup 1.50, 2°, deg 10,
ns 4): **φ0 77.9% / Rx 0.892**, ns 6: 78.0% / 0.892; ns 8 pathology table in item 3.
RCWA cross-check on identical geometry reads 4–5% low at affordable orders and its
5.14.1 monotone-trend warning fires (under-resolved — expected on Cu ε≈−83; the
adjudicated limit agrees with PMM). Application-side sources:
`pbs_qwp_mirror_sim/src/pmm_taper.py` (`_coated_layers`, `coated_tapered_jones`,
`view_coated_1d`), `validation/check_bare_paths.py` (the gold-standard match),
`check_pmm_straddle.py`, `check_v5_14.py` (regression + tripwire + ns8/deg14 row).


---

## EXECUTION STATUS (2026-06-10, same day)

Critically assessed and implemented in the v5.14.1 working tree (see
CHANGELOG "Device-geometry roadmap" for the shipped surface):

| Item | Verdict | Status |
|---|---|---|
| 1 multi-ridge/pillar builders | sound, small | **SHIPPED** all four families (center-anchor pinned by a drift-detector test) |
| 2 geometry algebra | sound; `coat(where=...)` SUBSET filters deliberately not offered in v1 (a wrong subset classification is the failure class this layer removes) | **SHIPPED** (`SegmentStackGeometry`, conformal coat = exact L∞ dilation, liners both orientations, hand-exact tests) |
| 3a physical wall-snap | sound, incl. the cross-layer-only asymmetry | **SHIPPED** (default period×1e-5; own-layer liners never merged) |
| 3b slices consensus | sound | **SHIPPED** (`stabilize='slices'`, warns on Jones disagreement; cannot re-slice hand-added layers — warns) |
| 4 trapezoid-metric layer | **sketch refuted**: `u=(x−c)/w(z)` leaves `1/w(z)²` z-dependence in the lateral operator for a linear taper — NOT a constant-coefficient modal problem; the "quadratic-in-q via linear convection" claim only holds for the slant (z-independent shear) | DEFERRED (research; exponential-taper gauge or Magnus z-ODE are the honest starts) |
| 5 sweeps | sound; NB the matrix row 5b was stale — `RCWAStack.solve_vs_wavelength` (v5.14.1) already returns jones | **SHIPPED** (segments jones sweep; PMM stacks dispersive + `jones=True` opt-in to keep released 3-tuples) |
| 6 PMM absorption | sound; per-material = renormalized volume-density split | **SHIPPED** (flux-based per-layer, closure ~1e-14 as an honest invariant; cross-solver totals agree on single-lossy stacks; the multi-lossy TM split exposes exactly the documented RCWA under-resolution) |
| 7 viewers | sound | **SHIPPED** (all three stacks + the geometry object) |
| 8 prepared material slots | sound | **SHIPPED** (`prepare()` + `solve(materials=...)`, bit-equal, eig-cache verified) |
| 9 material loaders | `from_csv` sound; `from_refractiveindex` (network) rejected — a numerics library should not fetch URLs at solve time | **SHIPPED** (`Material.from_csv` + `.index()`) |

Validation caveat: the roadmap's device-specific reference numbers (82.3% /
77.9% etc.) live in the application repo (`pbs_qwp_mirror_sim`), which is not
part of this library; the shipped tests pin the same INVARIANTS on synthetic
geometry (single-feature == legacy builders bit-for-bit, center-drift
detector, hand-exact coat/liner bands, closure invariants, prepared ==
rebuild).  Re-running the application validation set against v5.14.1 is the
recommended acceptance step.

---

## DEVICE ACCEPTANCE (2026-06-10, v5.14.1 working tree @32c6e1d — application side)

The recommended acceptance step was run: the full coated device
(`pbs_qwp_mirror_sim`, Ta1/Al2O3-6, n_sup 1.50, 2°, deg 10) rebuilt through the
NEW APIs only and exercised end-to-end
(`validation/check_new_geometry_api*.py`, `check_min_feature_snap.py`).

| Test | Result |
|---|---|
| Geometry-algebra build (8 calls, no boundary arithmetic) | φ0 **76.5% / Rx 0.887**, φ90 **8.7% / 0.770** vs hand-built 77.9/0.892, 10.4/0.774 — two ns4 staircase *realizations* of one trapezoid (different z-banding); see convergence row |
| Staircase convergence | geom-obj ns4→6→8: 76.5 → 77.1 → **77.4%**; hand ns4→6: 77.9 → 78.0% — both → one limit ✓ |
| Under-grounded-tooth Ta (Cu-Cu plane) | expressible via distinct keys, same eps (`"Cu"`/`"CuCol"` + `line_interface("Cu","CuCol",…)`) ✓ |
| `plot_geometry` | correct (straddle, taper steps, liners) ✓ — polish: legend shows eps values; material KEY names would be better when built via `to_pmm_stack(materials=…)` |
| `prepare()` + LC swap (4-pt φ-curve) | 76.5 / 63.5 / 26.1 / 8.7% in 158 s; endpoints ≡ rebuild ✓ |
| Dispersive `solve_vs_wavelength(jones=True)` + `Material.from_csv` (real IMEC SiCN nm-CSV + Al2O3 µm-CSV + callable Cu/Ta) | 1295/1310/1325 nm → 68.8 / 76.5 / 82.3% (129 s), trend ≡ per-λ refs ✓ |
| `layer_absorption(by_material=True)` | closure 8e-7 ✓; φ0: Cu(all) 6.4% + Ta **2.1%** + ~2.4% into the Cu substrate (= the 10.8% lump 1−Rx, now attributed); φ90: Cu 14.1% + Ta 3.8% — the Ta-under-teeth redesign confirmed DIRECTLY (was 33–58% in the sidewall design, by inference). NB same-eps keys merge ("Cu"+"CuCol" → one entry): attribute by KEY when keys exist, or give twin keys an eps·(1+1e-12) to split them |
| `min_feature=1.5e-9` snap on the ns8 build (1.22 nm collisions) | snap **fires** (44 cross-layer pairs named), result physical (75.8%/0.885), and **5.7× faster** (56 s vs 321 s — smaller union); ~1.6% geometry perturbation vs unsnapped, as expected for ±0.75 nm wall moves |
| Existing-path regression on this tree | hand-built `coated_tapered_jones` defaults: **77.9% / Rx 0.892 EXACT** — the batch changes nothing it shouldn't ✓ |

Bonus observation: the geometry-object ns8 build happens to solve physically
even UNSNAPPED (77.4%, 321 s) — the near-coincidence pathology is
realization-sensitive (wall-pattern dependent), which is exactly why the
defensive posture (snap or ns ≤ 6, plus `stabilize='slices'` for the silent
case) is the right contract rather than trusting any particular realization.

Verdict: **ACCEPTED.** The application's ~140 lines of geometry arithmetic
reduce to an 8-call build; the φ-curve, λ-sweep, loss budget, and viewer are
one-call library operations; the two residual polish items are cosmetic
(viewer label names, by-key absorption attribution).
