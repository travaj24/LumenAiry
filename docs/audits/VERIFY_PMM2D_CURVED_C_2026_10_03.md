# VERIFY -- curved cells for the pure staggered 2-D PMM, Phase C (shape primitives, the stack-level merge, the incident decomposition, curved viewers): an independent adversarial re-measurement

Verifier: Claude Opus 5.5 (family Claude Opus, version 5.5, model ID
`claude-opus-5-5`).  Date: 2026-10-03.
Mount: worktree `C:/tmp/lum_vcurved_c`, branch `verify/pmm2d-curved-c`
on the build tip `607d43b0` (`feat/pmm2d-curved-c`, 15 commits `bcd8a1fd ..
607d43b0` on Phase B `91d00288`, Phase A verifier merged).  PRE tree for
every before/after: my own `git archive 91d00288` in `C:/tmp/vcc_pre`.
Builds: Windows 11 (tesla-ryzen), CPython 3.14.6, numpy 2.4.4, scipy 1.17.1;
WSL Ubuntu, CPython 3.12.3, numpy 2.4.6, scipy 1.17.1 (`~/lumvenv`).  BLAS
pinned to one thread on every command line; every probe asserts
`lumenairy.__file__` under the tree it measures (`verify_c/_vc.py`).  The box
was loaded throughout (up to 32 Python processes, 99 % CPU): every wall time
is an upper bound.  Probes and JSON (both builds):
`validation/probe_pmm2d_curved/verify_c/`.  Decision tests:
`tests/unit/test_verify_pmm2d_curved_c.py`.

Nothing in `lumenairy/` was edited.  Two candidate fixes were applied to a
SCRATCH copy (`C:/tmp/vcc_fix`, `git archive HEAD` + the edits of section 9)
only to measure them.

---

## 0. Words used here

* **Merge.**  `shapes2d._merge`: the union of every shape's walls becomes the
  wall grid; each grid vertex or edge lying on a shape's outline is CLAIMED
  by that shape (its exact physical image / curve); every other vertex keeps
  its `(u, v)` position and every other edge is straight.  The builder calls
  this "per-edge claims".
* **Macro-cell rule.**  The plan's alternative (section 4.3): a wall that
  crosses another shape's curved cell SUBDIVIDES it, the sub-cells evaluating
  that cell's own transfinite blend (what `RefinedMap` does for squaring).
* **Geometry oracle** (`verify_c/_geom.py`, reads nothing of the merge's
  bookkeeping): (i) PAINT -- interior Gauss points of every `(u, v)` cell are
  mapped to the physical plane and the shapes' analytic `contains` says which
  material is there; it must be the cell's `eps`; (ii) SIDES -- every
  material boundary of the painted grid is sampled through the map and the
  analytic materials 1e-7 p either side along its normal must be the two
  cells' two `eps`; (iii) COVER -- every visible analytic outline point lies
  on a drawn material boundary (segment projection; chord sag ~1e-8).  A map
  passing (i)-(iii) reproduces every outline exactly; a "WRONG MAP" verdict
  would be the silent defect this phase exists to prevent.
* **Mode-pick** (my comparison arm for the incident field): the incident wave
  taken as the TWO discrete superstrate eigenmodes with the largest order-0
  far field, combined so their order-0 far field is exactly the input -- an
  exact discrete half-space mode, which a vacuum spacer can only phase.

---

## 1. Verdict table

| # | claim / ask | verdict | evidence (this verifier) |
|---|---|---|---|
| C1 | no shape, no map = the bytes of `91d00288` | **CONFIRMED** | own fixture set (non-square 1.1 x 0.9, lambda 0.95, n 1.3 / 1.6; operators of scalar / non-uniform / tensor / OOP / magnetic / slant / oblique, OOP modes, far projectors, efficiency TE/TM incl. conical and lossy, Jones scalar / lossy / tensor / OOP sym+nosym+conical / magnetic / slant, stacks with tensor, magnetic, OOP, slant, lossy layers + absorption, mortar normal and oblique): **118 / 118** SHA-256 identical; the 1e-15 "see" key differs (`v1_compare.json`) |
| C1 | rectangles-only `shapes=` runs the UNMAPPED solver | **CONFIRMED ONLY WHEN THE WALLS ARE BIT-EXACT -- DEFECT V-D1** | with every solver-side mapped function AND every map `geom` trapped: the builder's lone `Rect` rides unmapped, but on HEAD my two three-rectangle stacks (walls coinciding only to round-off, 1 ulp) fire `_stag_map_nodes` -- the mapped quadrature path -- and a tensor rectangle is REFUSED as "CURVED" (`v3_struct_win.json`); 7.5 % (15 / 200, 18 / 200, 12 / 200 at three periods incl. SI units) of random two-rectangle layouts miss the identity.  With the V-D1 edit every case rides unmapped, no trap fires, tensors accepted (`v3_struct_fixtree_win.json`) |
| C1 | the F-B4 fix is reachable ONLY with a map | **CONFIRMED** | same traps: no unmapped solve reaches `_stag_incident_*_mapped`; a curved circle fires `SP._stag_incident_coeffs_mapped` |
| C1 | shapes route = `compile_shapes` + `cmap=` to the bit | CONFIRMED (C1 / C2 / C10 ids re-run green both builds) | -- |
| C2 | circle primitive = Phase B map; FEM | CONFIRMED (fingerprints; test re-run) | not re-laddered to M = 11 (cost); the spacer ladder below runs the same map to M = 9 |
| C3 | fillet fingerprints = Phase B | CONFIRMED | fillet primitives on my own parameters (off-centre, w != h) EXACT, area 2e-16 |
| C4 | F5 / D6 traps closed | CONFIRMED | every primitive vertex maps to its physical point (ids re-run); my mutation of the rotated ellipse (outline on the walls) caught by C12 / C17 |
| C5 | refusals name the shapes; rollback | CONFIRMED | every refusal in my 37 merge scenarios names both shapes (`v2_merge_win.json`); the fold message's explanation is misleading for far-apart shapes (V-D3) |
| C6 | two-layer merged map: closure, absorption, vacuum identity | **CONFIRMED** on my own stacks | annulus (5 x 5), circle beside a sinusoid (4 x 4), nested fillets of different radius (9 x 9), Phase B circle: lossless-layer absorption 4.7e-16 .. 1.4e-14, vacuum identity 7.4e-15 .. 1.7e-13, vs Phase B's explicit map + vacuum layer 1.5e-14 (section 5) |
| C7 / C8 | convenience = one-layer stack; efficiency refuses | CONFIRMED (ids re-run) | -- |
| C9 | F-B4 floor / window / spacer | **CONFIRMED, with a sharper statement** (section 3) | window-free to 0 (exactly) on the stretch and 3e-16 on the circle; the spacer residual falls spectrally (pairwise) 2.6e-5 -> 1.3e-8 at M = 5 -> 9, identical for every incident arm at M >= 6, so it is NOT an incident-projection artefact; but the shipped projection is not the best object (mode-pick is equal or better everywhere, up to 200x) |
| C10 | oblique / conical; reciprocity | CONFIRMED | off-centre disk at (20 deg, 30 deg): reciprocity 1.1e-4 / 3.0e-5 / 2.3e-6 at M = 5 / 6 / 7 (least squares 1.4e-4 / 2.7e-5 / 1.2e-6) |
| C11 | viewer draws curves | **CONFIRMED for every primitive** | drawn vertices on the analytic outline to <= 4.4e-16 for circle, 5 x 5 circle, fillet, ellipse, ellipse rotated +20 / -15 deg, sinusoidal ridge, two-wave half-plane; each merged two-layer panel draws ITS layer's outline only; `plot_section` boundaries 2e-16 (`v8_viewer_win.json`) |
| C12 | area / perimeter | CONFIRMED | my primitives: area <= 9e-16 rel.; ellipse perimeter 2e-16; sinusoid perimeter 1.4e-11 vs a 4e5-point polyline |
| C13 / C14 / C17 | 2 x 2 array, four-fold, rotated mirror | CONFIRMED (ids re-run) | C17's ellipse (aspect 1.43, 20 deg) sits inside the fold-free domain; the documented domain is much larger (V-D2) |
| C16 | tensor under a curve raises naming Phase D | CONFIRMED | and spuriously for rectangles-only on round-off walls (V-D1) |
| ask 2 | the merge rule | builder's 0.105 **CONFIRMED** (0.1054 = r - r / sqrt 2 in the limit); per-edge claims **never built a wrong map** in 37 scenarios; but it **over-refuses** common devices the macro-cell rule lays out exactly (V-D3) | section 2 |
| ask 3 | primitives at their limits | fillet limit exact at sqrt(2) 1e-3 p (doc wording off, V-D5); circles tangent / overlapping the cell edge REFUSE (documented); aspect-5 ellipse fine; **rotated ellipse folds over most of its documented range (V-D2)**; near-touching sinusoid lays out but converges slowly | section 4 |
| ask 4 | incident projection | right object class, renormalisation genuine (not a gate fit) | section 3 |
| ask 7 | 21 tests vs 16 mutants | 10 caught, 6 survived (+1 equivalent): all 6 closed by my file | section 7 |
| ask 8 | docs | 7 / 8 worked examples RUN as written (incl. the cookbook stack, 1075 s); the CHANGELOG stack is correct but too expensive at its printed `n_modes = 7` (> 7000 s; 187 s at 4); wording defects V-D4 .. V-D7 | section 8 |
| ask 9 | four departures | per-edge claims: sound but a gap (V-D3); no steep split: a trap for extreme sinusoids; eps_cell / shapes exclusion: sound; flat-side edge raises: sound, documented | section 6 |
| ask 10 | 368 mypy findings | all 146 `no-untyped-def` + 222 `no-untyped-call`; `--check-untyped-defs` on both modules: **no issues** -- none hides a type inconsistency | section 8 |
| pre-existing | `__all__` walker on `material_key` | **CONFIRMED pre-existing**: fails identically on `91d00288` | section 10 |

---

## 2. The merge rule

### 2.1 The builder's 0.105 (macro-cell blend bends a crossing wall)

Re-measured on the 3 x 3 circle (r = 0.36, p = 1.2) with `RefinedMap` (the
macro-cell rule): a straight `u`-wall at `v0` inside the circle's top cell
maps to a curve whose largest excursion is 0.1038 / 0.0977 / 0.0916 / 0.0733
/ 0.0610 / 0.0305 at `v0` = 0.86 / 0.88 / 0.90 / 0.96 / 1.0 / 1.1, and
0.10544 in the limit `v0 -> arc` = `r - r / sqrt 2` = 0.10544
(`v2_merge_win.json`, `macro_cell_blend_bulge`).  **The number is right.**
Its relevance is narrower than the BUILD doc implies: the bend only matters
when the crossing wall is a MATERIAL boundary inside the curved cell; where
the wall is a mere grid line of another shape far away, bending it is
harmless.

### 2.2 Does per-edge claiming build a wrong map?  Never, in 37 scenarios

Every scenario was run through `_merge` and, when it solved, through the
geometry oracle (paint / sides / cover) on every layer:

| scenario | outcome | grid | paint wrong / checked | sides wrong / checked | cover max | min det J / sigma spread |
|---|---|---|---|---|---|---|
| annulus, circle r 0.25 (layer 1) + r 0.45 (layer 2) | SOLVE EXACT | 5 x 5 | 0 / 1800 | 0 / 3200 | 1.9e-8 | 3.7e-2 / 1.38 |
| the same off-centre (0.55, 0.62) | SOLVE EXACT | 5 x 5 | 0 / 1800 | 0 / 3200 | 1.5e-8 | 3.8e-2 / 1.38 |
| annulus in ONE layer (disk minus disk) | SOLVE EXACT | 5 x 5 | 0 / 900 | 0 / 3200 | 1.9e-8 | 3.7e-2 / 1.38 |
| annulus, radius ratio 1.2 (inner 45-deg wall inside the outer bulge) | SOLVE EXACT | 5 x 5 | 0 / 1800 | 0 / 3200 | 2.3e-8 | 3.2e-2 / 1.38 |
| the same circle in two layers | SOLVE EXACT | 3 x 3 | 0 / 648 | 0 / 1600 | 2.8e-8 | 8.1e-2 / 1.37 |
| circle 3 x 3 vs the same circle `core=0.5` | RAISE (vertex claims 1.05e-2 apart), names both | -- | -- | -- | -- | -- |
| circle vs `Ellipse(a = b = r)` (same outline, EllipseArc) | SOLVE EXACT | 3 x 3 | 0 / 648 | 0 / 1600 | 2.8e-8 | -- |
| circle vs r (1 + 1e-13) | SOLVE EXACT (walls snapped) | 3 x 3 | 0 / 648 | 0 / 1600 | -- | -- |
| circle vs r (1 + 1e-10) / (1 + 1e-6) | RAISE SLIVER, names both | -- | -- | -- | -- | -- |
| fillet r 0.06 vs r 0.12, same box, two layers | RAISE FOLD, names both | -- | -- | -- | -- | -- |
| nested fillets 0.5 / r 0.05 in 0.9 / r 0.12 | SOLVE EXACT | 9 x 9 | 0 / 5832 | 0 / 8000 | 2.3e-9 | 4.7e-2 / 1.37 |
| circle beside a fillet (period 2.0) | SOLVE EXACT | 7 x 7 | 0 / 3528 | 0 / 4000 | 2.3e-8 | 1.5e-2 / 1.38 |
| circle whose 45-deg wall cuts the fillet's arc row | SOLVE EXACT | 7 x 7 | 0 / 3528 | 0 / 4000 | 6.6e-9 | 4.7e-2 / 1.38 |
| three equal circles in a row (3.6 x 1.2) | SOLVE EXACT | 7 x 7 | 0 / 1764 | 0 / 4800 | 2.8e-8 | 5.1e-2 / 1.46 |
| **three circles r 0.36 / 0.30 / 0.40 in a row** | **RAISE FOLD** | -- | -- | -- | -- | -- |
| **two circles r 0.30 / 0.40 side by side (one or two layers)** | **RAISE FOLD** | -- | -- | -- | -- | -- |
| two circles r 0.25 / 0.36 (ratio 1.44 > sqrt 2) | SOLVE EXACT | 5 x 5 | 0 / 900 | 0 / 2400 | 2.8e-8 | 7.4e-2 / 1.44 |
| two equal circles, centres 0.1 apart in y | SOLVE EXACT | 5 x 5 | 0 / 900 | 0 / 2400 | 2.3e-8 | 6.8e-2 / 1.38 |
| sinusoid x0 0.15 A 0.05 beside a circle's transition cell (other layer) | SOLVE EXACT | 4 x 4 | 0 / 1152 | 0 / 2800 | 2.3e-8 | 6.1e-2 / 1.51 |
| the same, A 0.14 (wall reaches 0.29, circle at 0.30) | SOLVE EXACT | 4 x 4 | 0 / 1152 | 0 / 2800 | 2.3e-8 | 6.1e-2 / 1.90 |
| sinusoid x0 0.25 A 0.1 (near the circle, not crossing) | SOLVE EXACT | 4 x 4 | 0 / 1152 | 0 / 2800 | 2.3e-8 | 6.1e-2 / 1.81 |
| sinusoid x0 0.32 A 0.08 (CROSSES the circle) | RAISE CROSS, names both | -- | -- | -- | -- | -- |
| two-wave `y`-sinusoid under a circle | SOLVE EXACT | 4 x 4 | 0 / 1152 | 0 / 2800 | 2.6e-8 | -- |
| sinusoidal ridge under a circle | SOLVE EXACT | 5 x 5 | 0 / 1800 | 0 / 3200 | 2.3e-8 | 6.1e-2 / 1.62 |
| **an electrode stripe under a pillar (outlines cross in plan view)** | RAISE CROSS, names both | -- | -- | -- | -- | -- |
| stripe beside the circle | SOLVE EXACT | 4 x 4 | 0 / 1152 | 0 / 2800 | 2.8e-8 | -- |
| rect far right, its bottom wall 1.5e-3 above the circle's top | SOLVE EXACT | 5 x 5 | 0 / 1800 | 0 / 1600 | 2.8e-8 | 3.0e-2 / 1.37 |
| small square / off-centre square over a disk | SOLVE EXACT | 5 x 5 | 0 / 1800 | 0 / 3200 | 3.9e-9 / 7.7e-9 | -- |
| rect corner exactly at the circle's 45-deg point (touching) | SOLVE EXACT | 4 x 4 | 0 / 1152 | 0 / 1600 | 2.8e-8 | 8.1e-2 / 1.37 |
| **rect corner exactly ON the circle at 30 deg (touching)** | **RAISE FOLD** | -- | -- | -- | -- | -- |
| rect + circle touching at the 45-deg point, one layer | SOLVE EXACT | 4 x 4 | 0 / 576 | 0 / 1600 | -- | -- |

Shallow intrusions are caught too: a rectangle or a sinusoid poking 1e-3 ..
1e-6 into a circle raises CROSS at every depth.  **No silent wrong map was
found.**  The per-edge rule's own failure is OVER-REFUSAL, measured
(`v11_departures_win.json`): two circles side by side in one row FOLD for
every radius ratio in (1, sqrt 2) tested (1.05 .. 1.40) and solve at 1.0
and >= 1.414; two EQUAL circles whose centres differ by 0.02 or 0.05 in `y`
FOLD (0.1 solves).  Mechanism: one circle's 45-degree wall falls inside the
other circle's arc bulge zone `(c + r / sqrt 2, c + r)`, where the per-edge
rule keeps that wall straight and the arc crosses it.  A supercell of
pillars of different radii -- the gradient metasurface -- is exactly this.

**The macro-cell rule lays these out.**  `v10_macro.py` builds the r 0.30 /
0.40 pair (period 2.4 x 1.2) as each circle's own Phase B 3 x 3 map on its
half-cell, the union walls subdividing each half's cells with the half's own
blend: the geometry oracle reads EXACT, and the solve converges -- lossless
closure 6.3e-2 / 3.9e-3 / 2.3e-3 / 2.3e-3 and rung change 9.8e-2 / 4.3e-2 /
2.4e-3 at M = 3 .. 6 (the closure plateau is the pillar rim cap, as for the
lone circle; `n_orders = 4` so that every propagating order of the doubled
period is counted).  Control: on two EQUAL circles the composite and the
shipped merge agree to 7.2e-4 / 1.2e-4 at M = 4 / 5 with the same closure
(`v10b_check_*.json`).  So the over-refusal is a property of the rule, not
of the geometry.  Verdict: **sound (never wrong), but a usability gap with a
measured remedy** -- V-D3; the hybrid "claims for material boundaries,
macro-cell blend for foreign non-material walls" belongs in Phase E.

---

## 3. The incident field under a map (F-B4 / D4)

**What the right object is.**  In a homogeneous half-space under a map the
discrete system's own modes are the eigenvectors of the mapped homogeneous
pencil; a plane wave is not one of them (its covariant field `J^T E` is not a
polynomial), so any incident representation is an approximation whose error
must vanish with M.  Three consistent choices: (a) the L2 projection of the
covariant field onto the basis (window-free, unique) -- shipped, with a 2 x 2
renormalisation so its OWN order-0 far field is exactly the input; (b) the
least-squares fit of the far field (window-dependent, under-determined);
(c) the discrete (0, 0) eigenmode pair itself (mode-pick).  (c) is the
cleanest physics -- an exact discrete half-space mode, so it carries no
spurious incident content into the structure -- and I measured it against
(a), (b) and the bare L2:

Film n = 2 under a separable STRETCH (sines 0.10 / -0.07 on a 3 x 3 grid) and
under the 3 x 3 CIRCLE map, max |R, T - Airy| (my own s / p Airy), every order,
both inputs (`v4_film_*_win.json`):

| map, incidence | M | renormalised L2 (shipped) | bare L2 | least squares | mode-pick |
|---|---|---|---|---|---|
| stretch, normal | 4 / 5 / 6 / 7 | 9.5e-8 / 2.1e-8 / 3.4e-13 / 2.2e-14 | 9.5e-8 / 2.1e-8 / 3.4e-13 / 2.4e-14 | 8.2e-8 / 3.1e-8 / 5.1e-13 / 2.6e-14 | 1.9e-7 / 6.8e-8 / 1.1e-12 / 2.2e-14 |
| stretch, 25 / 0 deg | 4 / 5 / 6 / 7 / 8 / 9 | 1.8e-4 / 1.9e-5 / 4.1e-8 / 7.0e-9 / 5.0e-9 / 1.2e-10 | 1.0e-3 / 7.7e-5 / 9.5e-7 / 1.2e-8 / 5.7e-9 / 5.6e-10 | 1.7e-4 / 1.8e-5 / 4.0e-8 / 6.8e-9 / 4.9e-9 / 1.2e-10 | 7.1e-4 / 1.8e-5 / 4.0e-8 / 6.8e-9 / 4.9e-9 / 1.2e-10 |
| stretch, 25 / 40 deg | 4 / 5 / 6 / 7 | 4.7e-5 / 6.0e-6 / 7.7e-9 / 1.1e-9 | 6.6e-4 / 3.8e-5 / 2.0e-7 / 4.0e-9 | 4.6e-5 / 5.6e-6 / 7.2e-9 / 1.1e-9 | 4.3e-5 / 5.6e-6 / 7.0e-9 / 1.1e-9 |
| circle, normal | 4 / 5 / 6 / 7 / 8 | 2.0e-6 / 1.1e-8 / 8.4e-12 / 3.2e-13 / 5.5e-14 | 6.8e-5 / 2.6e-6 / 8.8e-8 / 4.8e-9 / 1.6e-10 | 8.1e-7 / 2.8e-8 / 4.3e-11 / 3.3e-13 / 5.5e-14 | **6.8e-8 / 8.3e-10 / 4.1e-14 / 2.8e-14** / 5.5e-14 |
| circle, 25 / 0 deg | 5 / 6 / 7 / 8 | 7.1e-5 / 3.6e-6 / 2.1e-7 / 1.0e-8 | 1.7e-4 / 1.2e-5 / 1.6e-6 / 2.1e-7 | 6.8e-5 / 3.5e-6 / 2.1e-7 / 1.0e-8 | 4.1e-5 / 3.0e-6 / 1.9e-7 / 9.7e-9 |
| circle, 25 / 40 deg | 5 / 6 / 7 / 8 | 2.4e-5 / 1.4e-6 / 6.5e-8 / 3.5e-9 | 7.3e-5 / 8.0e-6 / 4.0e-7 / 9.0e-8 | 2.7e-5 / 1.4e-6 / 6.3e-8 / 3.4e-9 | 1.7e-5 / 1.2e-6 / 5.9e-8 / 3.3e-9 |

`n_orders` 2 vs 5 (same table): renormalised L2 and mode-pick 0 (bitwise) on
the stretch and <= 3.3e-16 on the circle at every M; least squares 1.6e-4 /
2.4e-3 / 1.4e-3 at M = 4 (stretch normal / 25-0 / 25-40) and 2.0e-5 on the
circle.

Vacuum spacer (0.3 and 0.77 thick) on the pillar, max |R, T change|
(`v4_spacer_*_win.json`):

| map | M | renormalised L2 | least squares | mode-pick |
|---|---|---|---|---|
| circle pillar | 4 / 5 / 6 / 7 / 8 / 9 | 4.4e-5 / 2.6e-5 / 1.9e-6 / 1.2e-6 / 4.3e-8 / 1.3e-8 | 1.2e-4 / 2.5e-5 / 2.0e-6 / 1.2e-6 / 4.3e-8 / 1.3e-8 | 2.9e-5 / 2.4e-5 / 2.0e-6 / 1.2e-6 / 4.3e-8 / 1.3e-8 |
| stretch pillar | 4 / 5 / 6 / 7 / 8 / 9 | 4.1e-4 / 1.2e-4 / 6.4e-7 / 2.4e-7 / 5.8e-9 / 3.7e-10 | 3.6e-4 / 2.1e-4 / 1.3e-6 / 1.8e-7 / 5.8e-9 / 3.9e-10 | **3.3e-6 / 1.0e-6 / 2.6e-7 / 1.6e-8** / 5.8e-9 / 3.6e-10 |

Reciprocity of an OFF-CENTRE disk (0.5, 0.7, r 0.3) at (20 deg, 30 deg),
reflection channels (-1, 0) / (0, -1) (`v4_recip*_win.json`): renormalised
L2 1.1e-4 / 9.0e-5, 3.0e-5 / 2.7e-5, 2.3e-6 / 2.1e-6 at M = 5 / 6 / 7;
least squares 6.1e-5 / 1.4e-4, 2.7e-5 / 2.5e-5, 1.2e-6 / 9.3e-7; mode-pick
4.4e-5 / 6.7e-5, 2.7e-6 / 6.1e-6 at M = 5 / 6.

Findings:

1. **The renormalisation is genuine, not a gate fit.**  It removes the bare
   L2's representation error on order 0 on BOTH maps at every incidence
   (5-25x at oblique, 10^2-10^4 on the normal circle film), and keeps the
   window independence (only order 0 enters).
2. **The vacuum-spacer dependence on the circle is discretisation, falling
   spectrally** in a pairwise staircase (2.6e-5, 1.9e-6, 1.2e-6, 4.3e-8,
   1.3e-8 at M = 5 .. 9, the same odd/even pattern as Phase B's FEM distance),
   and it is IDENTICAL for all three arms at M >= 6: it comes from the
   outgoing side (discrete reflected modes are not single plane waves), not
   from the incident projection.  It is 10-1000x below the distance to the FEM
   oracle at every rung (8.2e-4 at M = 7, 2.8e-5 at M = 9), so it is a lower
   bound of the error, not an estimate of it.
3. **The shipped projection is not the best object.**  Mode-pick is equal or
   better on every quantity measured: the circle film at normal incidence
   13-200x closer (6.8e-8 vs 2.0e-6 at M = 4, 8.3e-10 vs 1.1e-8 at M = 5,
   4.1e-14 vs 8.4e-12 at M = 6), the stretch spacer 2.5-120x smaller at
   M = 4 .. 7, reciprocity 1.3-2.5x tighter at M = 5 and 4-11x at M = 6.  It needs a robust identification of the (0, 0) pair (largest
   order-0 far field here); near a Wood anomaly or an accidental degeneracy
   that choice can mix -- a reason to evaluate it, not to ship it in C.
4. **Pre-existing, not this phase:** the SHIPPED unmapped solver has the same
   least-squares under-determination at OBLIQUE incidence: `n_orders`
   dependence 1.5e-5 / 1.8e-7 / 4.2e-9 at M = 4 / 5 / 6 (integer-grid pillar,
   0.3 rad), identical on `91d00288` (`v3b_unmapped_floor_{pre,post}_win.json`);
   a 1-ulp wall difference (array walls vs the integer grid) moves R / T by
   3.1e-10 at M = 5 oblique (`v3_struct_win.json`).

---

## 4. The primitives at their limits (`v5_primitives_geom_win.json`)

| case | outcome |
|---|---|
| `FilletRect` off-centre, w != h (0.7 x 0.4, r 0.09) | EXACT, area 2.0e-16 |
| `FilletRect` r = 1.414e-3 p | **RAISE** -- the true limit is sqrt(2) 1e-3 p = 1.41421e-3 p, so the documented "below 1.414e-3" refuses 1.414e-3 itself (V-D5) |
| r = sqrt(2) 1e-3 p exactly; r = 1.5e-3 p | EXACT (segment = 1e-3 p, the sliver contract's edge) |
| r = 1.3e-3 p | RAISE, names the remedy |
| non-square period (1.2 x 0.8), r = 1.42e-3 x 0.8 | RAISE (the limit uses max(p_x, p_y): correct, both axes carry r / sqrt 2) |
| `Circle` off-centre; `core=0.4` off-centre | EXACT, area 4.9e-16 / 1.6e-16 |
| `Circle` tangent to the cell edge (r = 0.5 p); r = 0.5 p - 1e-3 p | RAISE "at least the sliver width 1e-3 of the period from the cell edges" |
| `Circle` overlapping the edge (r = 0.6 p) | RAISE "shapes do not wrap ... shift the lattice origin" -- documented in the module's Known limits |
| `Ellipse` aspect 5 (0.5 x 0.1) | EXACT, area 5.3e-16; sigma spread 4.03; converges: closure 1.2e-2 .. 3.0e-6, rung 3.7e-4 at M = 8 |
| **`Ellipse` aspect 5 rotated 30 deg; aspect 1.75 at 44.9 deg** | **RAISE FOLD** (lone shape) -- V-D2 |
| `SinusoidalWall` 1.5e-3 p from the cell edge | EXACT; min det J 2.5e-2, spread 2.06; **not converged by M = 8** (rung 4.3e-2 / 1.2e-1 / 7.5e-2 / 7.1e-2 at M = 5 .. 8, against 5.4e-2 .. 2.4e-3 for the same wall mid-cell) |
| `SinusoidalWall` touching the edge | RAISE (sliver margin) |
| three-wave phased ridge (axis y, phase 0.7) | EXACT (the 2.7e-7 cover reading is chord sag: 1.7e-8 at 4x the samples) |

The rotated-ellipse fold domain of a LONE ellipse (`v5b_ellipse_layout_win.json`):

| aspect | shipped layout folds at | normal-45 layout (V-D2 edit) |
|---|---|---|
| 1.05 | 40, 44.9 deg | EXACT at 5 .. 44.9 deg and -30 deg |
| 1.5 | 30, 40, 44.9, -30 deg | EXACT everywhere |
| 2.0 | 20 .. 44.9, -30 deg | EXACT everywhere |
| 3.0 | 20 .. 44.9, -30 deg | EXACT everywhere |
| 5.0 | 10 .. 44.9, -30 deg | EXACT everywhere |

With the edit, the Phase B and C gates stay green (21 C + 18 B ids, section 9)
and on the case both layouts handle (0.40 x 0.28 at 20 deg) the film under
the map converges at a comparable rate (5.4e-6 / 1.2e-7 / 7.8e-11 / 1.8e-12
against the shipped 1.1e-6 / 1.7e-8 / 7.8e-11 / 2.6e-13 at M = 4 .. 7,
`v5c_ellipse_film_*.json`), the device's R00 / T00 agreeing at the
discretisation level.

---

## 5. C6 on my own two-layer stacks (`v6_*_win.json`, theta 0.15, phi 0.3)

| stack | grid | M | lossless closure | lossless layer's absorption (layer 2 lossy) | sum absorption vs 1 - R - T | vacuum-painted layer 2 vs uniform vacuum, same map |
|---|---|---|---|---|---|---|
| annulus r 0.25 / 0.45 | 5 x 5 | 3 / 4 | 1.1e-2 / 1.5e-3 | 4.1e-15 / 4.0e-15 | 7.1e-3 / 1.0e-3 | 1.7e-13 / 3.4e-14 |
| sinusoid beside a circle's transition cell / circle | 4 x 4 | 3 / 4 | 9.6e-3 / 1.9e-3 | 2.3e-15 / 4.7e-16 | 6.8e-3 / 1.3e-3 | 7.8e-15 / 7.4e-15 |
| fillet 0.5 r 0.05 / fillet 0.9 r 0.12 | 9 x 9 | 3 | 5.6e-3 | 1.4e-14 | 3.8e-3 | 3.6e-14 |
| Phase B circle + vacuum-painted circle | 3 x 3 | 4 / 5 | 5.4e-3 / 1.7e-3 | 9.2e-16 / 9.3e-15 | 4.5e-3 / 1.2e-3 | 1.5e-14 / 1.5e-14 |

Against Phase B's explicit `_circle_map_3x3` + `eps_cell` + a uniform vacuum
layer 2: fingerprints equal, R / T 1.5e-14 at M = 4 and 5 (round-off).
Against NO second layer the difference is the vacuum-spacer residual of
section 3 (discretisation level, falling spectrally) -- "agreement to
round-off with the single-layer circle" holds on the same map with the vacuum
layer present, not with the layer removed, and it cannot: under a curved map
a vacuum layer is a physical no-op but not a discrete one.

---

## 6. The builder's four departures

1. **Per-edge claims instead of the macro-cell blend.**  Sound for exactness
   (no wrong map in 37 scenarios), but it over-refuses supercells of
   unequal pillars and slightly offset equal ones (section 2.2) -- a GAP with
   a measured remedy (V-D3).  Not a ship blocker: the refusal is loud and
   names both shapes; the message should say why.
2. **No automatic steep-cell split.**  A TRAP for extreme geometry: a
   sinusoidal wall 1.5e-3 p from the cell edge lays out EXACT with a valid
   map but is not converged by M = 8 (rung change 7e-2), while the closure
   (1.1e-4) looks healthy.  The builder's 2-34x on a strong sinusoid is
   consistent.  Recommend a warning when a cell's sigma spread exceeds ~2
   or min det J relative falls below ~3e-2 (both measured here), until the
   steepness-aware split exists.
3. **Raw `eps_cell` layers cannot join shape layers.**  SOUND: the message is
   clear and `Rect` covers every rectangular layer.  Consequence a user
   meets: `add_tapered_pillar` needs `layer_grids='per-layer'` and shapes
   need the shared grid, so a tapered CURVED pillar must be concentric
   `Circle` layers -- which merge exactly (2 / 3 / 4 / 6 steps -> 5 x 5 /
   7 x 7 / 9 x 9 / 13 x 13 grids; the pencil grows as the square of that).
4. **A straight edge on a fillet's flat side raises.**  SOUND and documented:
   a rounded pillar on a pedestal of the SAME footprint raises FOLD; a
   pedestal 1.2e-3 wider solves.

---

## 7. The 21 tests against 16 mutants (`v7_mutation_matrix.json`)

| mutant | caught by (Phase C ids) |
|---|---|
| m01 rotated ellipse: outline drawn on the `(u, v)` walls, not on their images | C12, C17 |
| m02 the merge drops every curve of layers >= 2 | C16 only (incidentally: the map turns into the identity); C6 MISSES it |
| m03 `background_eps` ignored | **survived** |
| m04 the map fingerprint ignores the arc radius | survived -- **EQUIVALENT** (the key holds the centre and the end vertices, which fix the radius; fillets r vs r (1 + 1e-12) still fingerprint differently under the mutant) |
| m05 the order-0 renormalisation dropped | C4 (rate), C9 (film) |
| m06 the shape layer's sliver constant halved | C5 |
| m07 the viewer draws the `(u, v)` cell outline | C11 |
| m08 rectangles never recognised as the identity | C1, C3, C16 |
| m09 vertex / edge claims compared at 1e-2 p | **survived** |
| m10 the plan-view crossing test disabled | C5 |
| m11 the fold scan that names the shapes skipped | C5 |
| m12 an interior crossing of a curved edge placed on the chord | **survived** (no Phase C test cuts a curved edge with another shape's wall) |
| m13 painting order reversed | **survived** (route-equality tests compare the merge with itself) |
| m14 the incident load without `conj` on the test functions | C10 |
| m15 the map's sinusoid ignores `phase=` | **survived** (every Phase C sinusoid in C4 has phase 0; C12's phased one checks area and length, which a phase does not change) |
| m16 the fillet arcs 1e-3 off the declared radius | C3, C4, C5, C6, C12 |

Closed in `tests/unit/test_verify_pmm2d_curved_c.py` (each run in every
surviving mutant tree; `v7_survivors_closed.json`): m03 by `test_vc1` and
`test_vc2[phased_sine_circle]`; m13 by `test_vc1`; m02 and m12 by all three
`test_vc2` scenarios; m15 by `test_vc2[phased_sine_circle]`; m09 by
`test_vc3`.  The file runs in about 1 s (geometry only).

---

## 8. Docs as an optical physicist reads them

* **Every worked example runs as written** (`v9_docs_win.json`): `Rect`
  (closure 9.8e-11, 55 s), `FilletRect` (3.1e-6, 260 s), `Circle` (2.6e-5,
  43 s), `Ellipse`, `SinusoidalWall`, `compile_shapes` (its stated
  `eps_cell` comment is exact), the cookbook stack (closure 2.6e-4, 1075 s
  loaded; a 7 x 7 merged grid, pencil 2450 DOF at `n_modes = 6`) and the
  CHANGELOG stack (correct as written -- the same code at `n_modes = 4` runs
  in 187 s, closure 1.3e-3, `v9_docs_changelog_M4_win.json` -- but at its
  printed `n_modes = 7` it did NOT finish inside the probe's 7000 s limit).  Cost is a usability point: the
  CHANGELOG example is a 7 x 7 merged grid at `n_modes = 7` (pencil 3528 DOF)
  and ran past 7000 s single-threaded on the loaded box; `n_modes = 4` or 5
  makes it a few-minute example (V-D4).
* `shapes2d.py` defines its terms (wall grid, preimage, hard edge, tangency
  point, singular vertex) before using them; painting, merging and refusing
  are explained physically; the Known-limits list is accurate except for the
  rotated-ellipse range (V-D2) and the over-refusal (V-D3).
* Wording defects: "exact modal decomposition" (CHANGELOG, both docstrings,
  the stack comment) overstates an L2 projection (V-D4); the fillet limit
  "below 1.414e-3" (V-D5); a forward version token "v5.50" in a test comment
  (V-D6); `np.float64(...)` reprs in the sliver / vertex-claim messages
  (V-D7); the roadmap's own "Phase E" heading contains "curved-cell Phase E"
  as an open item -- two different Phase E's in one paragraph (V-D4).
* No default moved (C1 above; every new keyword defaults to `None`), so no
  `Migration-Guide.md` entry is needed -- confirmed.
* mypy: the 368 findings are 146 `no-untyped-def` and 222 `no-untyped-call`
  (21 sampled lines, all of these two codes); `mypy --check-untyped-defs` on
  both modules with an empty config: **Success, no issues** -- the bodies
  type-check, so no untyped def hides an inconsistency.  The configured
  strict list: `Success: no issues found in 33 source files`.

---

## 9. Defects, with exact edits

**V-D1 (P2) -- rectangles-only merges miss the identity at round-off.**
`_merge` stores each claim's own arithmetic; walls are snapped to one value
(`_WALL_SNAP`) but the vertex images are not, and a straight edge's interior
crossings are placed by linear interpolation; both land 1 ulp off.  Effect:
the stack runs the MAPPED quadrature path for pure rectangles (different
bytes from the unmapped route, and at oblique incidence a discretisation-level
difference -- the BUILD doc's F-C3, 6.0e-5 at M = 4), and a tensor rectangle
is refused as "CURVED".  Edit, `lumenairy/elements/pmm/shapes2d.py`: after
`_CROSS_TOL = 1e-9` add

```python
#: A vertex claim this close to its merged grid vertex (relative to the
#: period) is the grid vertex itself: a few ulps, far below any real move.
_VERTEX_SNAP = 1e-13
```

and in `_merge`, between `        V[key] = xy0` (end of the vertex-claim loop)
and `    curved = {}`, insert

```python
    # A claim within a few ulps of its grid vertex IS the grid vertex: two
    # shapes' walls are snapped into one (_WALL_SNAP) but their vertex
    # claims keep each shape's own arithmetic (cx - w / 2 vs cy + h / 2), and
    # a straight edge's interior crossing is placed by linear interpolation;
    # both land 1 ulp off the merged wall, which would make a rectangles-only
    # merge a non-identity map (the mapped solver, tensors refused).
    G = np.empty_like(V)
    G[..., 0] = U1[:, None]
    G[..., 1] = V1[None, :]
    near = np.max(np.abs(V - G), axis=-1) <= _VERTEX_SNAP * scale
    V[near] = G[near]
```

Measured on the scratch tree: 0 / 600 random two-rectangle layouts miss the
identity; all five structural cases ride the unmapped solver with every map
function trapped, tensors accepted; Phase B + C ids 39 / 39 green; lone
primitives' fingerprints unchanged (C2 / C3 green).  Pinned by
`test_vc4_rectangles_only_merge_is_the_identity_to_round_off`
(`xfail(strict=True)`; remove the marker with the fix).

**V-D2 (P2) -- the rotated `Ellipse` folds over most of its documented
range.**  Replace, in `Ellipse._layout`, the rotated branch from
`        c = (self.cx, self.cy)` / `        P = {k: np.array(self._pt(t * _DEG))
...` through the closing `]` of its `edges` list by

```python
        c = (self.cx, self.cy)
        al = self.angle
        # the corners where the OUTWARD NORMAL points at 225 / 315 / 45 /
        # 135 degrees in the lab (the analogue of the circle's 45-degree
        # points): parametric t = atan2(b sin(psi - angle), a cos(psi -
        # angle)).  The parametric 45-degree points fold the side cells for
        # aspect >= 1.5 at 30 deg, 2 at 20 deg, 5 at 10 deg (verifier V5b).
        def tn(psi):
            p = psi * _DEG - al
            return float(np.arctan2(b * np.sin(p), a * np.cos(p)))
        tBL = tn(225.0) % (2 * np.pi)
        tBR = tBL + (tn(315.0) - tBL) % (2 * np.pi)
        tTR = tBR + (tn(45.0) - tBR) % (2 * np.pi)
        tTL = tTR + (tn(135.0) - tTR) % (2 * np.pi)
        P = {k: np.array(self._pt(t)) for k, t in
             (("BL", tBL), ("BR", tBR), ("TR", tTR), ("TL", tTL))}
        u = [0.5 * (P["BL"][0] + P["TL"][0]), 0.5 * (P["BR"][0] + P["TR"][0])]
        v = [0.5 * (P["BL"][1] + P["BR"][1]), 0.5 * (P["TL"][1] + P["TR"][1])]
        vrot = {(u[0], v[0]): P["BL"], (u[1], v[0]): P["BR"],
                (u[1], v[1]): P["TR"], (u[0], v[1]): P["TL"]}
        ax = (a, b)
        edges = [
            _HardEdge("h", v[0], u[0], u[1],
                      EllipseArc(c, ax, tBL, tBR, angle=al)),
            _HardEdge("h", v[1], u[0], u[1],
                      EllipseArc(c, ax, tTL, tTR, angle=al)),
            _HardEdge("v", u[0], v[0], v[1],
                      EllipseArc(c, ax, tBL + 2 * np.pi, tTL, angle=al)),
            _HardEdge("v", u[1], v[0], v[1],
                      EllipseArc(c, ax, tBR, tTR, angle=al)),
        ]
```

and in the class docstring replace "its corners the ellipse's points at
PARAMETRIC angles 45, 135, 225, 315 degrees (`c + Rot(angle) (a cos t, b sin
t)`)" by "its corners the points where the outline's outward normal points
at 45, 135, 225, 315 degrees".  Measured: EXACT over aspect 1.05 .. 5 and
angles -30 .. 44.9 deg; C12 / C17 green; convergence comparable (section 4).
If the edit is deferred, the docstring must state the fold-free domain
measured in section 4 instead of "up to 45 degrees".  Pinned by
`test_vc5_rotated_ellipse_lays_out_over_its_documented_range`
(`xfail(strict=True)`).

**V-D3 (P3, gap) -- over-refusal of the per-edge merge, and a misleading
fold message.**  Not fixable inside Phase C.  Edits now: in the
`shapes2d.py` module docstring, "How shapes combine", after the FOLDS
bullet add "-- including shapes that are far apart: a wall of one shape that
crosses the arc 'bulge' of another (between its 45-degree points and its
extreme) is kept straight and folds the cell.  Measured: two circles side
by side in one row with radius ratio between 1 and 1.414, or equal circles
whose centres differ by 0.02-0.05 of the period in y, raise; a common map
for them (the plan's macro-cell rule) is Phase E."  In `_check_fold`'s
message replace "two outlines are too close in plan view, or a straight wall
of one shape passes through the curved transition cell of another" by "two
outlines are too close in plan view, or a straight wall of one shape crosses
the arc bulge of another shape (even far away along that wall -- e.g. pillars
of different radii in one row)".

**V-D4 (P3, doc) -- "exact" incident decomposition; an hour-long example.**  In the CHANGELOG example change `n_modes=7` to `n_modes=4`.  In CHANGELOG.md, the
`pmm_jones_2d_staggered` and `PMM2DStackPure` docstrings and the stack
comment, replace "its EXACT modal decomposition" / "exact modal
decomposition" by "its unique L2 modal projection (window-free)".  In
`docs/PMM_ROADMAP.md`'s Phase E paragraph write "curved-cell PLAN Phase D /
E" where the open items are listed, so the roadmap's own Phase E and the
plan's Phase E cannot be confused.

**V-D5 (P3, doc) -- the fillet limit.**  `FilletRect` docstring and the
`_layout` message: "1.414e-3 of the period" -> "sqrt(2) x 1e-3 of the period
(1.4142e-3)".

**V-D6 (P3) -- forward version token.**  `tests/unit/
test_v4_16_0_walker_all_symmetry.py`: "# v5.50 (curved-cell Phase C,
2026-10-02):" -> "# curved-cell Phase C (2026-10-02, unreleased):".

**V-D7 (P3, cosmetic) -- numpy reprs in refusals.**  `_Grid1D.check_slivers`:
`{self.b[k]!r} and {self.b[k + 1]!r}` -> `{float(self.b[k])!r} and
{float(self.b[k + 1])!r}`; the vertex-claim message: `{tuple(xy0)} vs
{tuple(xy)}` -> `{tuple(map(float, xy0))} vs {tuple(map(float, xy))}`.

---

## 10. Gates and test tails

* Windows, this tree: Phase C file `21 passed` in every non-mutated run;
  verifier file `5 passed, 2 xfailed in 1.01s`.
* Windows sweep (`logs/win_sweep.txt`; every `pmm2d` / `stack2d` /
  `stagger` / `curved` file incl. Phases A / B / C and both verifier files,
  census, public API, doc identifiers, doc consistency, except budget, every
  walker, history lint / relocation / fingerprint tool, re-exports, kernel
  consistency; `-n 6`): `1 failed, 1620 passed, 8 skipped, 2 xfailed, 96 warnings in
  548.10s` -- the one failure is the walker below.
* The one red, `test_v4_16_0_walker_all_symmetry.py::
  test_all_submodule_entries_reexported_or_exempt` on `material_key`:
  identical failure on `91d00288` (run in `C:/tmp/vcc_pre`) -- pre-existing,
  not touched.
* Scratch fix tree (`C:/tmp/vcc_fix`, V-D1 + V-D2): Phase B + C ids + the
  verifier file `2 failed, 44 passed` (18 B + 21 C + 5 verifier) -- the two
  failures are the strict XPASS of `test_vc4` / `test_vc5`, as designed.
* WSL (CPython 3.12.3, numpy 2.4.6, scipy 1.17.1, `lumenairy` from
  `/mnt/c/tmp/lum_vcurved_c`): Phases A / B / C + both verifier files + the
  walker, except-budget, doc-identifier and public-API gates, `-n 4`:
  `2 failed, 85 passed, 2 xfailed in 244.17s` -- the walker (pre-existing, as
  above) and `test_public_api.py::test_installed_metadata_version_matches_
  source_version`, an ENVIRONMENT failure of the WSL venv (its installed
  `lumenairy` metadata is stale): it fails identically on `91d00288`.
  The WSL probe batch (`wsl_batch.sh`): own C1 set `118 / 118` identical
  pre vs post (`v1_compare_wsl.json`); every merge / primitive / departure /
  rotated-ellipse verdict identical to Windows (37 / 21 / all / all), the
  same trap fired in the same structural cases (V-D1 reproduces), the viewer
  vertices on the outline to 3.9e-16, the incident arms' readings equal to
  3 digits (`v12_crossbuild.json`), the unmapped oblique floor identical
  pre vs post.
* `python scripts/record_history_fingerprints.py --check`: "OK: every history
  document matches its module."  `python -m mypy` (configured list):
  "Success: no issues found in 33 source files".  WSL ruff 0.15.16 on
  `lumenairy/ tests/ scripts/ validation/probe_pmm2d_curved/verify_c/`:
  "All checks passed!".

---

## 11. Ship recommendation (A + B + C as 5.50.0)

**Ship, after V-D1 and V-D2 (or V-D2's docstring alternative) and the
V-D4 .. V-D7 wording.**  The core claim of this phase holds under adversarial
geometry: the shape route never built a silently wrong map (37 scenarios,
every outline checked against its analytic form), nothing shipped moved (118
own keys + the builder's 290), the incident fix is a genuine improvement
(window-free, renormalisation not a fit), the viewers draw the analytic
curves.  V-D1 is a one-block, measured-safe edit that makes the "rectangles
ride the unmapped solver" promise true for real layouts; V-D2 makes a
documented primitive work over its documented range.  V-D3 is a limitation to
DOCUMENT now and lift in Phase E.

## 12. What Phase D / E must carry

1. The merge rule: claims for material boundaries, the macro-cell blend for a
   foreign wall that is not a material boundary inside a curved cell
   (lifts V-D3; `v10_macro.py` is the measured prototype).
2. The incident representation: evaluate the discrete-eigenmode (mode-pick)
   incident on the mapped path (equal or up to 200x better on every
   measure here) with a robust (0, 0) identification; and decide whether the
   UNMAPPED oblique path (least squares, `n_orders` dependence 1.5e-5 at
   M = 4, pre-existing) should take the window-free projection -- a shipped
   change that needs a Migration-Guide entry.
3. Tensors / `mu` under a map (D); crossing outlines, slant, per-layer maps,
   the JAX twin (E); curved shapes with `add_tapered_pillar` (today: concentric
   Circle layers, grid 2 n + 1 per axis for n steps).
4. A steepness warning (sigma spread > ~2, relative min det J < ~3e-2) until
   the steepness-aware split exists.
5. Supercell users must widen `n_orders` to every propagating order of the
   larger period (my first composite run at `n_orders = 2` read a 5e-2
   "closure" that was only missing orders) -- worth a sentence in the
   cookbook.

## 13. Not measured

* The circle FEM ladder beyond M = 9 on this tree (the builder's M = 10 / 11
  rungs were not re-run; the spacer ladder runs the same map to M = 9).
* Idle wall times (box saturated).
* A FEM oracle for any merged two-shape stack, the rotated ellipse or the
  macro-cell composite: their references are geometric (the oracle of
  section 0), physical identities and convergence.
* Mode-pick robustness near Wood anomalies or degenerate (0, 0) pairs.
* WSL copies of the M >= 7 ladders (the WSL batch repeats the M <= 6 rungs).
