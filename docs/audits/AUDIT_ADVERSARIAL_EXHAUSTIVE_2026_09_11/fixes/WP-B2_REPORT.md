# WP-B2 — the 2-D displaced remap: structured inversion, and the launch lattice as a keyword

Finding **L9** of `AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11.md` §2.1 — the half
WP-A2 deferred (its §6 item 2).  Files owned: `lumenairy/elements/_lens_real.py`,
`lumenairy/elements/lens_config.py` (one field), their history document, the new
b2 test file, this report and `WP-B2_CHANGELOG.md`.

Environment: CPython 3.14, numpy 2.4.6, scipy 1.17.1, numexpr NOT installed,
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` on every
invocation, one process at a time.  Branch `audit-fixes-2026-09`, HEAD
81d5b586.  No git write command of any kind was run; no process was killed; no
file outside the list above was edited.

---

## 1. Summary

| item | status | files:lines | tests | oracle | measured before → after |
|---|---|---|---|---|---|
| **L9-a** the reflection instability that blocked the lattice raise | **fixed — and it was not what WP-A2 concluded** | `_lens_real.py:2596` (the symmetric input window) | `test_audit2609_b2_…::TestTheInputWindowIsSymmetric` (10), `::TestTheMirrorSymmetryDoesNotDependOnTheLattice` (4) | the mirror symmetry of the physics itself | p10 image-plane mirror residual at `n_side` 512 / 1025 / 2049: **6.3e-03 / 7.4e-03 / 5.3e-03 → 5.6e-14 / 8.5e-14 / 8.5e-12**, with BOTH interpolation backends |
| **L9-b** structured inversion replacing the Delaunay backend | fixed | `_lens_real.py:2388` `_remap2d_interp_structured`, `:2347` `_remap2d_affine_seed`, `:2305` `_remap2d_interp_delaunay` (the retained oracle) | `::TestTheStructuredInversionOnAnAffineMap` (3), `::TestAgainstTheRefinedLatticeLimit` (3), `::TestTheDefaultPathBuildsNoTriangulation` (2) | a ray-exact oracle (per-field-point Newton on the TRUE trace, no lattice); a tilted plate whose exit map is exactly affine | vs the oracle, amplitude rms at `n_side` 181/257/513/1025: delaunay 9.21/4.57/1.16/0.29e-04, **structured 8.72/4.25/1.08/0.27e-04**; on the affine map **4e-16 amplitude, 3e-11 rad phase, exactly lattice-independent**; hull holes inside the illuminated pupil **61–360 of 3782 → 0**; transmitted power **0.97720 → 0.98658** |
| **L9-c** the lattice raised and exposed | fixed | `_DISP_REMAP_2D_N_SIDE` `:2025` (181 → 257), `_normalise_displaced_n_side` `:2119`, `apply_real_lens(displaced_n_side=)`, `_check_displaced_support` `:4809`, `_routes_to_displaced_remap_2d` `:1558`, `lens_config.py:494` | `::TestDisplacedNSideValidation` (21), `::TestLensNumericsCarriesTheField` (4), `::TestTheLatticeSetsTheTransverseResolution` (3) | the ray-exact oracle for the accuracy the raise buys; contrast transfer for the resolution it buys | launch pitch on a 10 mm aperture **55.6 → 39.1 µm**; oracle amplitude rms **8.7e-04 → 4.3e-04**, phase **4.7e-02 → 2.4e-02 rad** |
| **L9-d** the smoothing warning restated | fixed | `_warn_if_remap_lattice_smooths` `:2076` | `::TestTheLatticeSetsTheTransverseResolution::test_the_warning_states_the_lattice_that_would_clear_the_bar` | — | the message now names `displaced_n_side=<n>`, computed from the call's own aperture and field pitch, and the test checks that `n` actually silences it |

Not-regressed gates, re-measured after every change: `pytest tests/unit -k
real_lens` (159 passed, 3 skipped, 1 failed — the config census, §5),
`test_niche_p10_transverse_walk_remap.py` (13), the three WP-A2 `a2` suites
(109), `test_audit2609_a15a_lens_covering_array.py` (45),
`test_niche_p2_displaced_extreme` + `test_g2_displaced_congruence` +
`test_niche_p9_decenter_tilt` + `test_niche_p3_pointwise_obliquity` +
`test_niche_r1_cosgrid_cache` (60), `validation/run_all.py test_lenses` (PASS),
`ruff check` on my three files (clean), `record_history_fingerprints.py
--check` (my module OK).

---

## 2. L9 in full

### 2.1 What WP-A2 left, and why

WP-A2 fixed L9's *silence* (`_warn_if_remap_lattice_smooths`) and the Newton
sweep counts, but left `_DISP_REMAP_2D_N_SIDE = 181` because raising it alone
was a measured regression.  Its reading reproduces exactly on this build
(relL2 of the p10 +d / −d image-plane mirror pair: 7.927e-14 / 5.498e-14 /
6.288e-03 / 4.124e-14 / 7.409e-03 at 181 / 257 / 512 / 513 / 1025; the
"4.1e-03 at 512" quoted in the WP-A2 report is the EE80 column of the same
sweep, 4.082e-03).  Its *diagnosis* was:

> The instability is the Delaunay backend's, not the resolution's: a denser
> scattered set hands QHull more near-degenerate cells to resolve arbitrarily,
> and which way it resolves them is not reflection-stable.

and the deferred design followed from it: replace `LinearNDInterpolator` with a
structured inversion, *then* raise the lattice.

### 2.2 The diagnosis was wrong, and the measurement that shows it

I implemented the structured inversion first, as instructed, and swept the same
lattices through it.  **It showed the same instability**: 6.807e-03 at 512 and
3.888e-03 at 1025, with no triangulation anywhere in the path.

Localising the residual put it at r ≈ 1.94–1.99 mm on a grid whose half-width
is 2.048 mm — the grid edge — at a magnitude of 0.63 of the peak, which is
exactly `exp(-(N dx / 2 / w0)²)`, the input Gaussian's value there.  The ray
map itself is mirror-antisymmetric to 2e-14 m at every lattice, so the trace
was never in question.

The cause is one line, upstream of both backends: the remap samples the input
envelope at the launch points with
`map_coordinates(..., mode='constant', cval=0.0)`.  The field axis
`(arange(N) - N/2)·dx` runs from `-(N/2)dx` to `+(N/2 - 1)dx` — one whole
sample further on the −x side — so a ray launched in the half-open band
`(x[-1], x[-1] + dx]` samples off the grid and carries nothing, while its
mirror between `x[0] - dx` and `x[0]` carries the full envelope.

Whether any ray lands in that one-pixel band is decided by the launch pitch,
and that is the entire "lattice instability":

| `n_side` | `dstep` | is `k·dstep − r_fan` in (2.040, 2.048] mm? | mirror-asymmetric launch samples | max sampled-envelope asymmetry |
|---|---|---|---|---|
| 181 | 57.22 µm | no (k = 125.7–125.8) | 0 | 4.4e-16 |
| 257 | 40.23 µm | no (k = 178.7–178.9) | 0 | 3.3e-16 |
| **512** | 20.157 µm | **yes (k = 357)** | 406 | **6.281e-01** |
| 513 | 20.117 µm | no (k = 357.4–357.8) | 0 | 3.3e-16 |
| **1025** | 10.059 µm | **yes (k = 715)** | 812 | **6.292e-01** |
| **2049** | 5.029 µm | **yes (k = 1430, 1431)** | — | — |

a perfect match to the stability pattern, including why 513 is fine and 512 and
1025 are not (it is not parity, and it is not pitch).

**Fixed** by carrying the envelope only over the largest CENTRED window the
caller's grid holds, `|x| ≤ x[-1]` and `|y| ≤ y[-1]`
(`_lens_real.py:2596`).  The bilinear stencil then touches only indices
`1 … N-1`, which form a mirror-symmetric set, and `fl(q + N/2)` against
`fl(N/2 − q)` is the only asymmetry left — one ULP of the interpolation
weights.  Measured sampled-envelope asymmetry 4.4e-16 of peak at every lattice;
image-plane mirror residual 5.6e-14 to 8.5e-12 across 181 / 257 / 512 / 513 /
1025 / 2049, **with either backend**.

Cost: the outermost input row and column no longer contribute to the carried
envelope.  That is the honest price of making a ray on one side of the axis
carry exactly what its mirror carries.

### 2.3 The structured inversion

`_apply_displaced_remap_2d` rebuilt the exit field by triangulating the
scattered exit points.  The launch fan is a REGULAR lattice, so
`(x_out, y_out)(u, v)` is a smooth curvilinear grid.  It is now inverted on that
grid: Newton on the bilinear interpolant of the map and of its lattice
gradients (`map_coordinates`, `order=1`), seeded by inverting the map's own
global affine part, with the transported amplitude and OPL read at the launch
coordinate that comes back.  The aperture is cut on that inverted LAUNCH
coordinate, i.e. on the entrance footprint at sub-launch-pitch resolution.

Three implementation decisions worth recording, each of them measured:

* **the loop cannot stop on a bitwise fixed point.**  The four
  surface-intersection Newtons in this module do (WP-A2's L9 change), because
  their residual genuinely reaches zero.  Here the Jacobian is a central
  difference of the lattice while the residual is of the bilinear interpolant,
  so the two disagree at O(h²) and the iteration limit-cycles in its last bits:
  measured, 95 % of a 512² grid was still moving at sweep 32.  The bar is the
  RESIDUAL against the local exit cell, derived in the source
  (`_DISP_REMAP_2D_INV_TOL_FRAC`) from the ratio of the inversion's own
  contribution to the bilinear error it feeds — 1e-06 against a right-hand
  side measured at 8.7e-03 / 6.2e-03 / 3.1e-03 at `n_side` 181 / 257 / 513.
* **an active set, and an affine seed.**  Together they took the p10 grid from
  32 sweeps (the cap) to two full sweeps plus a 3 % tail, at identical output:
  every error column in the oracle table is bit-for-bit what it was before the
  optimisation.  The affine seed's residual is 2.4e-06 m against 5.5e-05 m for
  the identity seed — one squaring of the Newton error, i.e. one whole sweep.
* **a search box.**  Only field points inside the bounding box of the
  amplitude-carrying exit points (plus one exit cell) are inverted.  Exact —
  the scattered path's hull is empty out there too — and it is what keeps a
  heavily padded grid cheap.

**Byte-identity with the shipped path was not achievable and is not claimed.**
Barycentric interpolation over the exit triangulation and bilinear
interpolation in launch space are different O(h²) approximations of the same
map; they agree only in the limit.  So the brief's alternative applies, and the
oracle is a ray-exact one: the same fan, but with each field point's launch
coordinate found by Newton on the TRUE trace (re-tracing individual rays, no
interpolation of the map anywhere), so the oracle has no lattice at all.  The
oracle's trace is checked against the production builder on the builder's own
lattice: bit-for-bit on `x_out` / `y_out` at `n_side = 37` and within 1 ULP
(8.7e-19 m on ~5e-3 m) at 181 and 257, and within 2 ULP on OPL for 2 of 1369
rays — those two being rays where the production intersection Newton stops at
its 24-sweep cap and the oracle, allowed 64, converges one ULP further.  The
oracle's own field-point inversion converges to a max residual of 1.25e-14 m,
eleven decades below the coarsest lattice's error.

Peak-relative over the illuminated core, N = 512, 0.7 mm beam, ±0.6 mm
decenter (2026-09-13):

| `n_side` | delaunay \|E\| max | delaunay \|E\| rms | structured \|E\| max | structured \|E\| rms | delaunay phase | structured phase |
|---|---|---|---|---|---|---|
| 65 | 2.491e-02 | 7.307e-03 | 2.432e-02 | **6.847e-03** | 3.864e-01 | 3.864e-01 |
| 91 | 1.238e-02 | 3.707e-03 | 1.250e-02 | **3.481e-03** | 1.856e-01 | 1.856e-01 |
| 181 | 2.952e-03 | 9.213e-04 | 2.789e-03 | **8.722e-04** | 4.715e-02 | 4.715e-02 |
| 257 | 1.511e-03 | 4.571e-04 | 1.505e-03 | **4.251e-04** | 2.410e-02 | 2.410e-02 |
| 513 | 3.586e-04 | 1.156e-04 | 3.657e-04 | **1.079e-04** | 5.882e-03 | 5.883e-03 |
| 1025 | 9.730e-05 | 2.918e-05 | 9.580e-05 | **2.743e-05** | 1.439e-03 | 1.439e-03 |

Second order in the launch pitch for both, the structured backend 5–6 % closer
in amplitude and identical in phase to four digits.  **The decisive difference
is not that column.**  It is the pupil rim: on a grid-truncated input the hull
stops at the outermost retained exit point, so the scattered backend returns a
ring of exactly-zero pixels inside the illuminated region (61 to 360 of 3782
sampled core points, lattice-dependent) and loses the power in it (transmitted
0.97720 against 0.98658 at `n_side = 181`).  The structured backend has none,
because the aperture is a disk in launch space rather than a polygon in exit
space.

An absolute correctness bar is available on one fixture and is used: a tilted
plane-parallel plate refracts every ray through the same angle (a flat face has
one normal) and shortens the in-glass path linearly with the tilted entry
height, so its exit map is exactly AFFINE with an exactly linear OPL.  Degree 1
in the launch coordinate, so the bilinear interpolant and its inverse are both
exact and the remap has no discretisation error at all.  Measured against the
closed form built from the traced map's own affine fit: amplitude 4e-16 of
peak, phase 3e-11 rad; and the answer does not move with the lattice (1.8e-11
rad across 91 / 181 / 512 / 513, which is ten float64 ULP of `k0·OPL` and is
the bar the test derives at runtime).

### 2.4 The lattice raise: 181 → 257, and what the measurement supports

**257 is derived, not chosen**: it is the largest lattice whose ray TRACE still
costs less than the field-grid interpolation it feeds, on every grid measured
(trace 0.152 s at 257 against structured interpolation 0.17 / 0.64 / 2.36 s at
N = 512 / 1024 / 2048; the next step, 361, costs 0.432 s of trace and is already
the expensive half).  It halves the remap's interpolation error against the
ray-exact oracle and takes the launch pitch on a 10 mm aperture from 55.6 µm to
39.1 µm.

**What the raise does not buy, said plainly.**  On a SMOOTH input the model's
own observables were already converged at 181.  On the p10 decentered singlet
at N = 1280 (d = 0.5 mm):

| `n_side` | pitch | centroid x [µm] | RMS radius [µm] | EE80 [µm] | P/P_in |
|---|---|---|---|---|---|
| 91 | 111.1 µm | 261.1792 | 14.2296 | 15.3991 | 0.99692 |
| 181 | 55.6 µm | 261.1760 | 14.2208 | 15.2905 | 0.99729 |
| 257 | 39.1 µm | 261.1740 | 14.2188 | 15.2831 | 0.99735 |
| 513 | 19.5 µm | 261.1759 | 14.2188 | 15.2822 | 0.99740 |
| 2049 | 4.9 µm | 261.1773 | 14.2193 | 15.2829 | 0.99741 |

i.e. 8e-06, 1.4e-04 and 5e-04 relative between 181 and 2049.  A larger default
would have bought nothing measurable there while multiplying the trace.

**What it does govern is input STRUCTURE**, which is what L9 actually names.
Contrast transfer of a sinusoidal amplitude ripple collapses onto one curve in
launch-samples-per-period — 0.94 / 0.92 at 7.2 samples, 0.76 / 0.70 at 3.6 and
3.1, 0.57 / 0.60 at 2.2, across `n_side` 181 / 257 / 361 / 513 — reproducing the
audit's own "2.2 launch samples/period returns at 0.51" to within the fixture
difference.  The resolved period therefore scales with the pitch, and **no
constant can resolve an arbitrary caller's field pitch**: on a 10 mm aperture
at dx = 8 µm the pitch bar needs `n_side = 626`.  That is what the keyword and
the restated warning are for, and it is the honest answer to "raise the
resolution ceiling".

### 2.5 `displaced_n_side`

Public on `apply_real_lens`, `None` by default, validated by
`_normalise_displaced_n_side` with the CONVENTIONS §2 prefix: a bool, a string,
a tuple, a non-finite, a float with a fractional part, or anything below the
3-ray structural floor raises rather than being truncated.  A call that would
DISCARD it raises too — any `surface_model` but `'displaced'`, a rotationally
symmetric element, or `displaced_obliquity='pointwise'` — naming which of those
it was; the routing rule is now one shared predicate
`_routes_to_displaced_remap_2d` used by both the guard and the dispatch, so
they cannot drift apart.  Carried as `LensNumerics.displaced_n_side`, floored
against the model's own constant through the existing `_vocab` accessor, and
wired into `_NUMERICS_FOR['apply_real_lens']` so `from_kwargs`/`to_kwargs`
round trips.

### 2.6 Residual risk

* **The 1-D symmetric remap has the same input-window asymmetry.**
  `_apply_displaced_remap` samples with the same `mode='constant'` convention
  on the same asymmetric axis.  It is rotationally symmetric, so no fixture in
  the suite exercises a mirror pair through it, and
  `test_niche_p10_…::test_symmetric_remap_is_the_p2_1d_remap_byte_identical`
  pins it byte-for-byte — which is why I did not change it inside an L9 pass.
  Recorded as deferred (§6.1) with the same two-line fix.
* **A field the remap resamples near its own Nyquist limit still aliases, and
  nothing says so.**  Found while measuring the covering-array fixture: its
  120 mm diverging source turns the residual phase by 3.72 rad per launch cell
  at `n_side = 181` — past Nyquist — while `_warn_if_remap_lattice_smooths` is
  silent, because it scores the launch pitch against the FIELD pitch (33.3 µm
  is already finer than 2·dx = 56.2 µm).  The answer moves 0.60 / 0.43 / 0.26 /
  0.13 of peak per lattice doubling from 181 to 2049 on that fixture.  I
  prototyped a warning on the measured phase step, **and removed it**: on that
  fixture the input is itself at Nyquist on its own field grid (3.14 rad per
  pixel), so the step saturates at π at every lattice and the warning would
  have named a keyword that cannot fix it — the exact defect class this
  campaign exists to close.  A correct two-armed diagnostic needs its own
  measurement; recorded as deferred (§6.2) with the design.
* **Knife-edge membership at the window and aperture boundaries.**  A launch
  point within 1 ULP of `|x| = x[-1]` or of `r_ap` can fall on either side.
  Measure-zero, and the same class of test `in_ap` has always used.
* **CuPy / JAX.**  `apply_real_lens` has no JAX twin, and the remap is
  scipy-bound (`map_coordinates`, and `distance_transform_edt` for the
  dead-ray fill) exactly as it was before; CuPy is not installed here and the
  displaced path already refuses `use_gpu=True`.

---

## 3. Files touched

Modified (owned):
* `lumenairy/elements/_lens_real.py`
* `lumenairy/elements/lens_config.py` — `LensNumerics.displaced_n_side`, its
  docstring, its `__post_init__` check, `_NUMERICS_FOR['apply_real_lens']`, and
  one word of `_vocab`'s docstring plus one name in its preload tuple
* `docs/history/lumenairy.elements._lens_real.md` — re-recorded
  (`record_history_fingerprints.py … --reason …`) in this change, and the WP-B2
  version narrative appended so the source carries none

New:
* `tests/unit/test_audit2609_b2_displaced_remap_inversion.py` (50 tests)
* `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-B2_REPORT.md`
* `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-B2_CHANGELOG.md`

`lumenairy/elements/lens_config.py` has no history document, so there is
nothing to re-record for it.  `docs/subsystems/real_lens.md` was read and
checked: it documents the displaced family's routing and the screen/remap
peers, and says nothing about the launch lattice or the interpolation backend,
so it remains true unchanged.

Not touched, and verified so: `lumenairy/elements/lenses_maslov.py`,
`lumenairy/propagators/asymptotic*.py` (WP-B1), `lumenairy/propagators/hf.py`,
`hfpi.py`, `rs.py`, `mft.py`, `carrier.py` (WP-B3).  Those files and their
history documents are dirty in the working tree from the concurrent work
packages; the two history-document drifts and the one ratchet entry that
`test_audit2609_a17_*` report are theirs, not mine (mine are OK).

---

## 4. Tests run

All with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` and
`python -m pytest <file> -q --no-header -p no:cacheprovider`.

| command | result | duration |
|---|---|---|
| `test_audit2609_b2_displaced_remap_inversion.py` | **50 passed** | 62 s |
| `test_niche_p10_transverse_walk_remap.py` (the acceptance trio) | **13 passed** | 130 s |
| `test_audit2609_a2_displaced_models.py` + `…_a2_analytic_lens.py` + `…_a2_verify_lens_analytic.py` | **109 passed** | 35 s |
| `test_audit2609_a15a_lens_covering_array.py` | **45 passed** | 3 s |
| `test_niche_p2_displaced_extreme` + `test_g2_displaced_congruence` + `test_niche_p9_decenter_tilt` + `test_niche_p3_pointwise_obliquity` + `test_niche_r1_cosgrid_cache` | **60 passed** | 615 s |
| the five files above plus `test_niche_p10_…` and the b2 file, one run | **217 passed** | 219 s |
| `pytest tests/unit -k real_lens` | **159 passed, 3 skipped, 1 failed** — the failure is the config census, §5 | 267 s |
| `test_audit2609_a16_lens_config_round_trip.py` + `…_a16_lens_arch` + `…_a16_lens_config_bit_identity` + `…_a16_verify_config_and_arch` | **164 passed, 2 failed** — both the config census, §5 | 16 s |
| the same file with the §5 two-line patch applied to a scratch COPY (the repo file untouched) | **84 passed** | 2 s |
| `test_audit2609_a17_history_lint.py` + `…_a17_history_relocation.py` | **765 passed** for everything but `lenses_maslov` / `carrier` (WP-B1 / WP-B3, dirty in the tree) | 83 s |
| `python validation/run_all.py test_lenses` | **PASS** (`ALL 1 files passed`) | 27 s |
| `ruff check` on my three files | **All checks passed** | — |
| `ruff check` (repo-wide) | clean when this pass began; 5 `F821` now, all in `lumenairy/elements/rcwa/twod.py` and `lumenairy/raytrace/trace.py`, both dirty in the tree from concurrent work packages | — |
| `python scripts/record_history_fingerprints.py --check` | **OK** for `lumenairy.elements._lens_real.md`; the two DRIFTs are `carrier.py` and `lenses_maslov.py` | — |
| the b2 + p10 + three a2 + a15a + four a16 files, run LAST as one command | **381 passed, 2 failed** — the two are the config census, §5 | 220 s |

Measurement scripts (in the session scratchpad, not committed): the ray-exact
oracle and its bit-for-bit check against the production builder, the mirror
sweep over six lattices × two backends, the backend/oracle convergence table,
the observable-convergence table, the contrast-transfer table, the interleaved
perf tables (cos-grid analogue, covering array, p10 singlet) and the
launch-window localisation.

**No wall-clock assertion exists anywhere in the test file** (TESTING_STANDARDS
S1).  The performance claims are pinned as operation counts — the default path
constructs no `LinearNDInterpolator` at all and the scattered arm constructs
exactly one; the inversion retires the grid in ≤ 30 `map_coordinates` calls
against a bar derived from the algorithm (seed + ≤ 4 sweeps of 6 + 2 + 3), with
16 measured — and as structure: on an affine map the answer is
lattice-independent to ten ULP.

---

## 5. Requested changes outside my ownership

**1. `tests/unit/test_audit2609_a16_lens_config_round_trip.py` — two lines, for
the new `LensNumerics` field.**  This is the WP-A16 census walker doing its
job: it asserts the exact field count and constructs "every field at a
non-default value" by hand, so a 39th field cannot be added without it.  The
brief mandates the field; the file is not mine.  The exact edit, verified green
on a scratch copy (84 passed):

```diff
@@ def test_to_kwargs_emits_only_requests_unless_asked_for_defaults():
-    assert len(everything) == n_fields == 38, (
+    assert len(everything) == n_fields == 39, (
@@ def test_to_kwargs_for_an_entry_point_is_exactly_what_it_accepts(ep):
         numerics=LensNumerics(bandlimit=False, wave_propagator='rs',
                               ray_subsample=4, output_subsample=2,
-                              remap_order=5,
+                              remap_order=5, displaced_n_side=513,
                               min_coarse_samples_per_aperture=8,
```

Until it lands, `test_to_kwargs_emits_only_requests_unless_asked_for_defaults`
and `test_to_kwargs_for_an_entry_point_is_exactly_what_it_accepts[apply_real_lens]`
are the only two red tests attributable to this work package.

**2. `docs/lens_configuration.md` — one row and one count.**  Its
`### LensNumerics (17 fields)` heading becomes `(18 fields)`, and the table
gains a row after `remap_order` (same column order: field, default, analytic,
traced, prepare, maslov, gbd, fga, multibranch):

> | `displaced_n_side` | None | Y | -- | -- | -- | -- | -- | -- |

It is the launch-lattice side, in rays, of the 2-D transverse-walk remap of
`surface_model='displaced'` — it sets that model's transverse resolution
(launch pitch `2·r_aperture/(n−1)`), costs its square in traced rays, and is
refused when the call does not route to that remap.

**3. `CHANGELOG.md` — the 5.47.0 block.**  `WP-B2_CHANGELOG.md` is written in
release-note form and can be pasted under the 5.47.0 heading as
`### Fixed` / `### Changed` / `### Added` / `### Performance`, with its
Migration section under the release's migration notes.

**4. `docs/subsystems/real_lens.md` — two additions.**  Nothing in it is
falsified by this change (it documents the displaced family's routing, the
smooth-residual transport and the no-in-glass-diffraction limit, none of which
moved), so these are enrichment rather than correction and I have left the file
alone.  §3.1, after "The 2-D remap carries **no in-glass diffraction**":

> The 2-D remap's transverse resolution is the LAUNCH pitch
> `2·r_aperture/(displaced_n_side − 1)`, not `dx`: it is a geometric transfer,
> so input structure finer than that pitch is smoothed to the lattice, and the
> call warns (naming the `displaced_n_side` that would clear it) whenever the
> launch pitch is coarser than twice the field pitch.  Default 257 rays a side.

and one row in §8's table:

> | 2-D remap: structured inversion, mirror stability vs the launch lattice, `displaced_n_side` | `tests/unit/test_audit2609_b2_displaced_remap_inversion.py` |

---

## 6. Deferred

1. **The 1-D symmetric remap carries the same input-window asymmetry.**
   `_apply_displaced_remap` samples `F` with
   `cx = (X·scale)/dx + Nx/2`, `mode='constant'`, on the same axis that reaches
   one sample further on −x.  Nothing in the suite exercises a mirror pair
   through it (it is rotationally symmetric, and its byte-identity is pinned),
   so nothing measures the exposure.  Design: the same two lines,
   `|x| ≤ x[-1]` and `|y| ≤ y[-1]` on the sampled envelope, plus a mirror
   fixture built on a DECENTRED input field rather than a decentred element.
   ~0.25 d, and it moves the byte-identity pin, so it wants its own change.

2. **A field resampled near its own Nyquist limit aliases silently.**  Measured
   above (§2.6).  Design, two arms because the two causes have different fixes:
   (a) measure the phase step per LAUNCH cell over the amplitude support and
   the step per FIELD pixel over the same support; (b) when the launch step is
   the binding one and exceeds ~π/2, warn and name the `displaced_n_side` that
   clears it; when the field step is already at Nyquist, warn and name
   `conjugate=` (the residual the remap transports is flat once the congruence
   matches, which is what `_residual_input_field` exists for).  Both arms need
   an amplitude-support threshold — `_TF_REMAP_SUPPORT_FRAC` is the module's
   existing convention — and a fixture ladder, because the naive form saturates
   at π on a field that is itself aliased.  ~1 d, and it closes the L3 residual
   risk WP-A2 documented but nothing measures.

3. **The trace, not the interpolation, is now what a raised lattice costs.**
   `_build_displaced_ray_map_2d` spends ~4 µs/ray (1.06 s for 513² rays on a
   two-surface singlet), because WP-A2's bitwise-fixed-point break needs ~13
   sweeps × 3 `_surface_sag_general` evaluations.  A residual-scaled break
   would land in 3–4, but it moves bits, and the byte-identity WP-A2 rests on
   is the gate.  Design: keep the bitwise break as the default and add a
   `displaced_trace_tol` that a caller explicitly opts into, with the
   byte-identity matrix as the acceptance.  ~0.5 d.

4. **The structured inversion is ~12 bilinear gathers per field point.**  That
   is what makes it lose to the triangulation on a coarse lattice over a large
   field grid (§Performance in the changelog).  The inverse map is as smooth as
   the forward one, so solving it on a grid of launch-lattice resolution and
   interpolating to the field points would make the full-grid work 5–6 gathers
   — at the price of a second O(h²) interpolation layer, which has to be
   measured against the oracle before it can be defended.  ~1 d, with the
   existing oracle table as the acceptance.

5. **`interp_method` is private.**  `_apply_displaced_remap_2d` takes it and
   `_build_displaced_cos_grid` takes an identically-named one; neither is
   public.  If a caller ever needs the scattered backend at the top level, the
   two want one shared public spelling rather than two private ones.  Not
   needed by anything measured here.

---

## 7. Changelog text

`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-B2_CHANGELOG.md`

---

## Addendum (orchestrator, 2026-09-14): the K3 remap pins were not restated

* `tests/unit/test_niche_k3_perf.py` pinned the 2-D displaced remap's K3 win -- byte-identity to the pre-K3 two-interpolator
  Delaunay algorithm at N = 384 / 512, and ONE QHull triangulation where the pre-K3 path built two.  This package replaced that
  construction with the structured inversion (section 2 / audit L9, a default move with a Migration note), so the three pins have
  failed since 47bc7a79; neither this package's selection nor VERIFY-B2's ran the file, and the 5.47.0 release gate found them.
  Restated at the close: the byte-identity pin retires (its oracle is the retired construction; `test_audit2609_b2_*` pin the
  structured inversion), and the count pin reads ZERO triangulations and ZERO `LinearNDInterpolator`s with the historical reference
  kept as the spy's control.  The same lesson as the WP-B7 / WP-B8 / WP-B9 follow-ups: a package that retires a construction must
  grep the tests for its name.

