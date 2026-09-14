# WP-B2 changelog text — the 2-D displaced remap's inversion and launch lattice

Release 5.47.0.  Finding **L9** of
`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11.md` §2.1 (the half WP-A2
deferred, §6 item 2).  Files: `lumenairy/elements/_lens_real.py`,
`lumenairy/elements/lens_config.py`.

---

### Fixed -- apply_real_lens: `surface_model='displaced'` on a decentered / tilted / freeform element no longer lets a half-pixel of grid convention decide a rim of pupil amplitude (L9)

The 2-D transverse-walk remap carries the input envelope along traced rays, and
it sampled that envelope at the launch points with
`map_coordinates(..., mode='constant', cval=0.0)`
(`_lens_real.py:2629`).  The field axis `(arange(N) - N/2) * dx` runs from
`-(N/2) dx` to `+(N/2 - 1) dx` — one whole sample further on the −x side — so a
ray launched in the band `(x[-1], x[-1] + dx]` sampled off the grid and carried
NOTHING, while its mirror between `x[0] - dx` and `x[0]` carried the full
envelope.  On the `test_niche_p10_transverse_walk_remap.py` fixture (f/5
singlet, 10 mm aperture, ±0.6 mm decenter, N = 512 at dx = 8 µm) that band
holds 0.63 of the peak.

Whether any ray lands in it is decided by the launch pitch, which is why the
effect presented as a lattice instability:

| `n_side` | launch pitch | ray in the band? | mirror-asymmetric launch samples | max sampled-envelope asymmetry |
|---|---|---|---|---|
| 181 | 57.22 µm | no | 0 | 4.4e-16 |
| 257 | 40.23 µm | no | 0 | 3.3e-16 |
| **512** | 20.16 µm | **yes** | 406 | **6.281e-01** |
| 513 | 20.12 µm | no | 0 | 3.3e-16 |
| **1025** | 10.06 µm | **yes** | 812 | **6.292e-01** |

The remap now carries the envelope over the largest CENTRED window the caller's
grid holds — `|x| <= x[-1]`, `|y| <= y[-1]` (`_lens_real.py:2652`) — so a ray on
one side of the axis can never carry amplitude its mirror cannot.  Image-plane
mirror residual of the +d / −d pair, an exact symmetry of the physics:

| `n_side` | 181 | 257 | 512 | 513 | 1025 | 2049 |
|---|---|---|---|---|---|---|
| before | 7.9e-14 | 5.5e-14 | **6.3e-03** | 4.1e-14 | **7.4e-03** | **5.3e-03** |
| after | 7.5e-14 | 8.4e-14 | 5.6e-14 | 7.4e-14 | 8.5e-14 | 8.5e-12 |

**This was NOT the triangulation.** WP-A2 attributed the jump to QHull
resolving near-degenerate cells arbitrarily and deferred the lattice raise
behind a backend replacement on that basis.  Running the same sweep through the
new structured inversion, with no triangulation anywhere, reproduced it
exactly (6.807e-03 at 512, 3.888e-03 at 1025) before this fix, and both
backends are stable after it.  The reproduction is pinned as the fail-before
arm of `test_audit2609_b2_displaced_remap_inversion.py`.

---

### Changed -- apply_real_lens: the 2-D displaced remap inverts its launch→exit map on its own structured grid instead of triangulating the exit points (L9)

`_apply_displaced_remap_2d` rebuilt the exit field by Delaunay-triangulating
the scattered exit points and interpolating onto the field grid.  The launch
fan is a REGULAR lattice, so the exit map is a smooth curvilinear grid: it is
now inverted on that grid — Newton on the bilinear interpolant of `(x_out,
y_out)` and of its lattice gradients, seeded from the map's own global affine
part — and the transported amplitude and OPL are read at the launch coordinate
that comes back (`_lens_real.py:2444` `_remap2d_interp_structured`,
`:2347` `_remap2d_affine_seed`).

What that buys, measured:

* **the aperture is cut on the inverted LAUNCH coordinate**, i.e. on the
  entrance footprint at sub-launch-pitch resolution, instead of at the convex
  hull of the retained exit points.  A grid-truncated pupil came back with a
  ring of exactly-zero pixels INSIDE the illuminated region — 61 to 360 of 3782
  sampled core points, lattice-dependent — and the power in it was lost
  (transmitted 0.97720 of the input against 0.98658 now, at `n_side = 181`);
* **no combinatorial backend.** Every step is a smooth function of the traced
  data, so a 1-ULP perturbation of the launch grid stays a 1-ULP perturbation
  of the output.  At `n_side = 2049` the triangulation's own degeneracy is
  visible in the mirror residual (1.7e-12 against 8.5e-12 for the structured
  arm — the same order, but it is the only lattice where the two differ at all
  above 1e-13);
* **cost that scales with the lattice, not against it** (below).

The scattered backend is retained, byte-identical, as
`_remap2d_interp_delaunay` and is reachable as
`_apply_displaced_remap_2d(..., interp_method='delaunay')`.  It is the oracle
the structured inversion is checked against: the two approximate the same map
by different O(h²) rules, so their converging to one answer is a check neither
gives alone.

**Byte-identity with the shipped path was not achievable and is not claimed.**
Barycentric interpolation over the exit triangulation and bilinear
interpolation in launch space are different second-order approximations.
Refereed against a ray-exact oracle — the same fan, but with each field point's
launch coordinate found by Newton on the TRUE trace, so the oracle has no
lattice at all — peak-relative over the illuminated core (N = 512, 0.7 mm
beam, ±0.6 mm decenter):

| `n_side` | delaunay \|E\| rms | structured \|E\| rms | delaunay phase (rad) | structured phase (rad) |
|---|---|---|---|---|
| 181 | 9.213e-04 | **8.722e-04** | 4.715e-02 | 4.715e-02 |
| 257 | 4.571e-04 | **4.251e-04** | 2.410e-02 | 2.410e-02 |
| 513 | 1.156e-04 | **1.079e-04** | 5.882e-03 | 5.883e-03 |
| 1025 | 2.918e-05 | **2.743e-05** | 1.439e-03 | 1.439e-03 |

Second order in the launch pitch for both — a clean 4× per doubling — with the
structured backend 5–6 % closer in amplitude and identical in phase to four
digits.  The convergence order and the backend agreement in the limit are
pinned at runtime (`TestAgainstTheRefinedLatticeLimit`), not quoted.

On a tilted plane-parallel plate — where the exit map is exactly affine, so
both the bilinear interpolant and its inverse are exact — the remap now has NO
discretisation error: amplitude 4e-16 and phase 3e-11 rad against the closed
form, and the answer does not move with the lattice (1.8e-11 rad across
`n_side` 91 / 181 / 512 / 513, which is ten float64 ULP of `k0 · OPL`).

---

### Added -- apply_real_lens: `displaced_n_side`, the 2-D remap's launch lattice, as a validated keyword and a `LensNumerics` field (L9)

```
displaced_n_side : int or None, default None
```

Side of the square launch lattice the 2-D transverse-walk remap traces, in
RAYS; `None` uses the module default.  The remap is a geometric transfer, so
this — not `dx` — sets the transverse resolution of its output: the launch
pitch is `2 · r_aperture / (displaced_n_side − 1)`, and input structure finer
than that (a hard stop edge, an obscuration, an upstream DOE, speckle) is
smoothed to the lattice.  Cost is `displaced_n_side²` rays traced through the
prescription; accuracy is second order in the pitch.

Validated with the CONVENTIONS.md §2 prefix (`_normalise_displaced_n_side`,
`_lens_real.py:2168`): a non-integer, a float with a fractional part, a bool, or
anything below the 3-ray structural floor raises rather than being truncated or
silently accepted.  A call that would DISCARD the setting — any
`surface_model` other than `'displaced'`, a rotationally symmetric element, or
`displaced_obliquity='pointwise'` — raises too, naming which of those it was
(`_check_displaced_support`, `_lens_real.py:4978`).  The routing rule is now a
single shared predicate `_routes_to_displaced_remap_2d` (`:1558`) so the guard
and the dispatch cannot drift apart.

Carried as `LensNumerics.displaced_n_side` (`lens_config.py:514`), floored
against `_lens_real._DISP_REMAP_2D_MIN_N_SIDE` through the existing `_vocab`
accessor so a config cannot accept a value the call would refuse, and wired
into `_NUMERICS_FOR['apply_real_lens']` so `from_kwargs` / `to_kwargs` round
trips.

---

* Restated at the release close: `tests/unit/test_niche_k3_perf.py` pinned the K3 win of the
  triangulating remap this entry retires (byte-identity to the pre-K3 two-interpolator algorithm
  at N = 384 / 512, and one QHull triangulation where the pre-K3 path built two); the
  byte-identity pin is retired with a note and the count pin now reads zero triangulations and
  zero `LinearNDInterpolator`s (`test_remap_2d_builds_no_triangulation_since_the_structured_inversion`),
  the historical reference kept as the spy's control.

### Changed -- apply_real_lens: the 2-D displaced remap's default launch lattice, 181 → 257 rays (L9)

Raising the lattice is what L9 asked for and what the reflection-stability fix
above unblocks.  257 is derived, not chosen: it is the largest lattice whose
ray TRACE still costs less than the field-grid interpolation it feeds, on every
grid measured (trace 0.152 s at 257 against an interpolation of 0.17 / 0.64 /
2.36 s at N = 512 / 1024 / 2048; the next step, 361, costs 0.432 s of trace and
is already the expensive half).  It halves the remap's own interpolation error
against the ray-exact oracle (amplitude rms 8.7e-04 → 4.3e-04, phase 4.7e-02 →
2.4e-02 rad) and takes the launch pitch on a 10 mm aperture from 55.6 µm to
39.1 µm.

**What the raise does not buy, stated because it is easy to assume.** On a
SMOOTH input the model's own observables were already converged at 181: on the
p10 decentered singlet at N = 1280 the image-plane centroid, RMS radius and
EE80 move by 8e-06, 1.4e-04 and 5e-04 relative between `n_side` 181 and 2049.
What the lattice governs is input STRUCTURE — contrast transfer collapses onto
one curve in launch-samples-per-period (0.94 / 0.92 at 7.2 samples, 0.76 / 0.70
at 3.6 and 3.1, 0.57 / 0.60 at 2.2, across four lattices) — and no constant can
resolve an arbitrary caller's field pitch, which is why the keyword exists.

---

### Changed -- apply_real_lens: the remap's smoothing warning names the lattice that would clear it (L9)

`_warn_if_remap_lattice_smooths` said the launch lattice was coarser than the
field grid and offered `displaced_obliquity='pointwise'` or
`apply_real_lens_traced`.  Both are different models.  It now also names
`displaced_n_side=<n>`, computed from the call's own `r_aperture` and field
pitch, that resolves the field pitch within the SAME model — and the test
suite checks that the `n` it names actually silences it.

---

### Performance -- the structured inversion against the triangulation it replaces

Interleaved medians of three, one process, `OPENBLAS_NUM_THREADS=1
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, CPython 3.14 / numpy 2.4.6 / scipy
1.17.1, 2026-09-13.  TESTING_STANDARDS S1: **no wall clock is asserted
anywhere**; the test file pins the claim as operation counts (the default path
constructs no `LinearNDInterpolator` at all, and the inversion retires the grid
in ≤ 30 `map_coordinates` calls).

**The cos-grid analogue** (`_build_displaced_cos_grid`, the structured-vs-QHull
pair WP-A2 §6.2 cited at 16.44 s against 8.43 s) — reproduced, same direction,
same factor:

| N | delaunay | structured | ratio |
|---|---|---|---|
| 512 | 1.426 s | 0.767 s | 1.86× |
| 1024 | 2.256 s | 0.968 s | 2.33× |
| 2048 | 2.933 s | 1.625 s | 1.81× |

**A covering-array fixture** (`test_audit2609_a15a_lens_covering_array.py`'s
curved-rear doublet with a 0.15 mm decenter so it routes to the 2-D remap),
whole `apply_real_lens` call:

| N | shipped (delaunay, 181) | structured, 181 | structured, 257 | structured, 513 |
|---|---|---|---|---|
| 64 | 0.240 s | 0.072 s (0.30×) | 0.175 s (0.73×) | 2.048 s (8.54×) |
| 256 | 0.296 s | 0.089 s (0.30×) | 0.274 s (0.92×) | 2.180 s (7.36×) |
| 512 | 0.384 s | 0.210 s (0.55×) | 0.305 s (0.79×) | 2.127 s (5.54×) |

**The p10 decentered singlet** (10 mm aperture, so the trace is 2.8× the rays
of the 6 mm covering-array fixture at the same lattice), the
scattered→field-grid step alone:

| N | n_side | delaunay | structured | ratio |
|---|---|---|---|---|
| 512 | 181 | 0.074 s | 0.167 s | 0.44× |
| 512 | 513 | 0.428 s | 0.213 s | 2.01× |
| 512 | 1025 | 1.798 s | 0.407 s | 4.42× |
| 1024 | 181 | 0.185 s | 0.617 s | 0.30× |
| 1024 | 1025 | 1.900 s | 0.837 s | 2.27× |
| 2048 | 181 | 0.607 s | 2.448 s | 0.25× |
| 2048 | 1025 | 2.581 s | 2.734 s | 0.94× |

and the same fixture end to end, old default against new
(shipped = delaunay at 181, new default = structured at 257):

| N | shipped | structured 181 | **structured 257 (new default)** | structured 513 | structured 1025 |
|---|---|---|---|---|---|
| 512 (dx 8 µm) | 0.166 s | 0.250 s (1.51×) | **0.323 s (1.95×)** | 1.435 s (8.66×) | 5.397 s (32.6×) |
| 1280 (dx 8 µm) | 0.699 s | 1.142 s (1.63×) | **1.225 s (1.75×)** | 2.294 s (3.28×) | 6.191 s (8.86×) |
| 3072 (dx 4 µm) | 4.587 s | 4.528 s (0.99×) | **4.532 s (0.99×)** | 5.423 s (1.18×) | 8.901 s (1.94×) |

**Honest summary.** The two costs scale differently and neither dominates
everywhere: the triangulation is built once over `n_side²` points and then
walked once per field point, so it wins on a coarse lattice over a large field
grid; the inversion pays ~12 bilinear gathers per field point and nothing for
the lattice, so it wins wherever the lattice is dense or the field grid is
modest — which is exactly the regime the raise moves into.  End to end at the
shipped defaults the new path is 0.30–0.92× the old on the covering-array
fixture (a 6 mm aperture), 1.75–1.95× on the p10 singlet (a 10 mm aperture, so
2.8× the rays at the same lattice) at N ≤ 1280, and 0.99× at N = 3072 where the
rest of the call dominates both.  WP-A2's "2× triangulation cost" prediction
holds for the cos-grid analogue it was measured on; it does not generalise to
the 2-D remap at a coarse lattice over a large grid, and that is recorded here
rather than quietly inherited.

---

### Migration

`surface_model='displaced'` on a decentered / tilted / `sag_callable` element
(the DEFAULT routing for such elements, and `displaced_mode='remap'`) returns a
different field. Three causes, in decreasing size:

1. **the default launch lattice moved 181 → 257.**  On a smooth input the
   observables move by ~1e-04 relative (measured above).  On an input the
   remap is already resampling near its Nyquist limit the move is large — up to
   0.38 of the peak on the covering-array fixture, whose 120 mm diverging
   source turns the residual phase by 3.7 rad per launch cell at 181.  Such a
   call was aliasing at the shipped default too; the fix is `conjugate=` (so
   the transported residual is flat) rather than any lattice.  Pass
   `displaced_n_side=181` to reproduce the previous sampling exactly.
2. **the exit field is rebuilt by inverting the map rather than triangulating
   it.**  Differences are O(h²) in the launch pitch and both paths converge to
   the same answer: 9e-04 peak-relative at 181 falling to 3e-05 at 1025 against
   the ray-exact oracle.  The rim changes more than the interior — the pupil
   edge now lands on the aperture rather than on the hull, so a truncated pupil
   gains back the ring the hull dropped and ~1 % of transmitted power with it.
3. **the carried envelope is cut at the grid's largest centred window**, so the
   outermost input row and column no longer contribute.  That is what makes the
   answer mirror-symmetric; it costs one pixel of the input rim.

`apply_real_lens_traced`, `apply_real_lens_maslov`, the pointwise obliquity
SCREEN (`displaced_obliquity='pointwise'`), the meridional LUT path, the 1-D
symmetric remap (`_apply_displaced_remap`) and every `surface_model='thin'` /
`'tangent_facet'` / `'tangent_facet_remap'` call are **unchanged** — verified
by `pytest tests/unit -k real_lens`, the WP-A2 a2 suites and
`validation/run_all.py test_lenses`.
