# Feedback — device-geometry batch (v5.14.1 @32c6e1d), from the consuming application

**Context.** Application-side acceptance feedback on the device-geometry batch
(`ROADMAP_DEVICE_GEOMETRY_SWEEPS_2026_06_10.md`, EXECUTION STATUS → SHIPPED ×7).
The full coated device (`pbs_qwp_mirror_sim`: two-tooth 2°-tapered Cu/SiCN LC
out-coupler, 1 nm Ta under-tooth/column liners, 6 nm conformal Al2O3, anisotropic
LC, n_sup 1.50) was rebuilt through the new APIs only and exercised end-to-end:
build → solve → φ-tuning via `prepare()` → dispersive Jones λ-sweep → per-material
absorption → viewer → `min_feature` snap. Scripts:
`pbs_qwp_mirror_sim/validation/check_new_geometry_api.py`, `…_api2.py`,
`check_min_feature_snap.py`. Full numbers in the roadmap doc's DEVICE ACCEPTANCE
section; this doc is the qualitative feedback + the residual asks.

**Verdict: sufficient.** The application's ~140 lines of geometry arithmetic
(boundary lists, coating offsets, sliver avoidance) reduce to an 8-call
`SegmentStackGeometry` build; the φ-curve, λ-sweep, loss budget, and viewer are
one-call operations; the hand-built legacy path is byte-stable on this tree
(77.9% / Rx 0.892 exact). The application is migrating onto these APIs
(including `reflective_outcoupling` for the QWP/PBS chain) and retiring its
hand-built geometry.

---

## What worked notably well

1. **The key-based material model.** String keys + eps-at-export turned out to be
   the load-bearing design choice: the under-grounded-tooth Ta — a liner crossing
   a *Cu–Cu* plane, inexpressible by material-pair interface rules alone — falls
   out naturally by giving tooth-Cu and column-Cu distinct keys mapped to the same
   eps (`line_interface("Cu", "CuCol", …)`). Keys also make the `prepare()` LC
   swap and the dispersive-callable slots compose cleanly.
2. **`coat(where='all')`-only.** Initially looked restrictive; in use it is the
   right v1 call. The build-order idiom (ridges → `coat` → `fill("LC")` → column
   ridges → `fill("SiCN")` → liners) expresses "coat THESE surfaces, not those"
   through *when* things exist rather than a surface classifier — and that is
   exactly the failure class the layer was built to remove.
3. **The snap's side effect: speed.** `min_feature=1.5e-9` on the ns8 build did
   not just prevent the near-coincidence pathology — it cut the solve from 321 s
   to 56 s (5.7×) by shrinking the union grid, at a ~1.6% geometry-perturbation
   cost (±0.75 nm wall moves). Worth a docstring note: the snap is also the cost
   knob for dense staircases.
4. **`layer_absorption(by_material=True)` answers a design question directly.**
   φ0: Cu 6.4% + Ta **2.1%** (+ ~2.4% exiting into the lossy Cu substrate,
   bookkept as T) — the Ta-under-teeth redesign, previously justified by slow
   material-removal inference (33–58% Ta absorption in the old sidewall design),
   is now confirmed in one solve. Closure |ΣA − (1−R−T)| ≈ 8e-7 on this stack.
5. **Honest invariants.** `BACKGROUND` refusing to export, dispersive stacks
   refusing single-λ `solve()`, the snap warning *naming* merged pairs — every
   failure encountered during integration was loud and self-explaining. Zero
   silent surprises.

## Residual asks (cosmetic, in priority order)

> **RESOLUTION (same day, 47846d6 + 690087b, verified application-side
> 2026-06-10, `validation/check_v5_14_1_fixes.py`):** all four addressed and
> adopted. (1) viewer legends are key-named; (2) by-key absorption verified —
> the twin-key split now resolves teeth-Cu vs column-Cu: φ0 **Cu(teeth)=6.14%,
> CuCol(column)<0.02%, Ta=1.97%**, i.e. the copper loss is entirely in the
> gap-plasmon tooth walls and the buried column is shielded, a design
> confirmation the merged map could not show; (3) docstrings; (4)
> `to_rcwa_stack` tensor materials verified — the application deleted its
> hand pixelation and gets the identical value (59.9% @ nh=80 smoke).
> Regression pin unchanged (coated φ0 77.8% / Rx 0.892 on 5.14.1).

1. **Viewer label names.** `plot_geometry` legends show eps values
   (`eps=2.25`); when the stack was built via `to_pmm_stack(materials=…)` the
   key names are known at export time — pass them through so the legend reads
   `LC`, `Ta`. (Same for `SegmentStackGeometry.plot`, which has the names
   natively.)
2. **By-key absorption attribution.** `by_material=True` keys on distinct *eps*
   values, so same-eps keys merge — the deliberate `"Cu"`/`"CuCol"` twin-key
   trick splits the geometry but not the loss map (teeth-Cu vs column-Cu lump
   into one entry). When keys exist, attribute by key; eps-keyed remains the
   fallback for raw-eps stacks. Workaround until then: give twin keys
   `eps·(1+1e-12)`.
3. **`plot_geometry` returns the Axes** — correct and documented, but the first
   thing a consumer does is `fig.savefig`; a `Returns: matplotlib Axes (use
   `.figure` to save)` docstring line would save the one obvious AttributeError.
4. **`to_rcwa_stack` rejects tensor materials** (found during the application
   migration): its pixelation does `complex(materials[m])`, so an anisotropic
   key (the LC's (3,3) tensor) raises `TypeError: only 0-dimensional arrays…`.
   `RCWAStack` itself supports `eps_tensor_cell`, so the fix is local: build a
   tensor cell when any resolved material has `ndim == 2` (promote scalars by
   `eps·I3`). Until then the application pixelates `gm.layers()` itself
   (~12 lines, single-source preserved) — see
   `pbs_qwp_mirror_sim/src/pmm_taper.py::coated_jones_rcwa`.

## Operational notes for other consumers (no library change needed)

- **Staircase realizations differ legitimately.** The geometry-object build and
  a hand-banded build of the same trapezoid are different ns4 staircase
  *realizations*: φ0 76.5% vs 77.9%, converging toward one limit with n_slices
  (76.5 → 77.1 → 77.4 vs 77.9 → 78.0). Pin references per realization; compare
  realizations only at convergence. Banding can be steered where continuity with
  a legacy model matters (e.g. splitting the tooth band at the coat plane).
- **The ns8 pathology is realization-sensitive.** The geometry-object ns8 build
  happened to solve physically even unsnapped; the hand-banded ns8 was the
  documented garbage/passive-but-wrong case. Do not trust any particular
  realization at dense slicing — keep the contract: snap (`min_feature`) or
  ns ≤ 6, plus `stabilize='slices'` where it matters.
- `Material.from_csv` handled both real-world CSV conventions here (IMEC SiCN
  with wl in nm; Al2O3 with wl in µm) via `wl_unit` — and the out-of-range raise
  caught one sweep that strayed past the table on the first try. Working as
  intended.
