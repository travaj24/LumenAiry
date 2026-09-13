# WP-A26 -- WP-A1's exact conic intersection enlarged the decentred ray fit's data domain by 2.4x, and the D7 fit order was calibrated to the old one

Subject: `tests/unit/test_niche_d7_decentred_fit.py::test_the_off_centre_fit_order_raise_flattens_the_exit_wavefront[0.5|1.0]`
stepping from 2.162 / 1.958 urad of exit-slope error against its own analytic conic oracle to
44.457 / 31.556 urad at `f602b72c` (WP-A1, ray tracing) and staying there through HEAD `6345d99d`,
while the on-axis figure stays 41.089 urad to the last digit.

Everything below is a MEASUREMENT taken on this machine with
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, Python 3.14.6, against the fixture's
own analytic decentre-INVARIANT oracle (the `K = -n^2` Fermat conic's exit-vertex-plane path
`f_b - sqrt(x^2 + y^2 + f_b^2)`, which shares no code with the element) and against an inline exact
conic raytrace written for this work package.  The pre-A1 library was extracted READ-ONLY with
`git archive 0067d63b lumenairy` into the scratchpad and the CURRENT test module run against it, so
only the library varies.  Nothing was checked out, stashed or written to the repository; no git
write command was issued.

---

## 1. Summary

| finding | status | files:lines | tests | oracle | measured before -> after |
|---|---|---|---|---|---|
| **A26-1** the regression is a defect in WP-A1's ray tracing | **NOT A DEFECT IN A1 -- R4 is correct and the attribution is corrected** | no change to `lumenairy/raytrace/` | `test_audit2609_a26_decentred_exit_reference.py::test_a1s_resurrected_conic_rays_hit_the_conic`, `::test_the_fit_is_handed_data_out_to_the_launch_square_corner` | an inline exact conic raytrace (flat entrance, exact even-conic sag, gradient normal, vector Snell), no library code | the 294 244 rays R4 resurrected agree with it to **2.17e-19 m** in exit coordinate and **6.51e-19 m** in exit OPL out to 3.6062 mm -- at the ULP floor of the quantities themselves (`eps*max|x_out|` = 4.59e-19 m, `eps*max|OPL|` = 4.98e-19 m); the 111 525 rays that were already alive are **bit-identical** across A1 on x, y, z, L, M, N, opd and error_code (`max abs(d) = 0.0` on every field) |
| **A26-2** the DECENTRED fit order is calibrated to a ray set the tracer was truncating | **FIXED** -- `_DECENTRED_FIT_POLY_ORDER` re-derived 10 -> 16 | `lumenairy/elements/_lens_traced.py:4086` (the constant), `:4000-4052` (the data-domain half of its derivation: the ladder, the cost, the fold and conditioning invariance), `:8137-8193` (the public `decentred_fit_poly_order` doc), `:4256-4274` (the step-down arithmetic), `:4364-4377` (`_DECENTRED_FIT_SPECTRUM_ORDER`'s headroom statement) | `::test_the_decentred_exit_wavefront_returns_to_the_analytic_oracle[0.5|1.0]`, `::test_the_decentred_fit_takes_the_re_derived_order`, `::test_the_concentric_path_is_untouched`, `::test_a_caller_asking_for_more_still_gets_more`, `::test_the_step_down_still_protects_an_undersampled_disc` | the analytic Fermat sphere | decentred exit slope **44.457 / 31.556 -> 2.371 / 1.683 urad** at 0.5 w and 1.0 w of decentre (18.8x on both), against **41.089 urad on axis, unchanged to the digit and bit-identical**; the pre-A1 figure was 2.162 / 1.958 |
| **A26-3** `docs/audits/TRACED_LAYER_MAP.md` S2 records the shipped value | **TAKEN by the orchestrator** (section 5.1) | `docs/audits/TRACED_LAYER_MAP.md:60`, one cell; `lumenairy/elements/_traced_flags.py:194-197`, the `note=` string | `test_niche_c14_encapsulation.py::test_the_layer_map_shipped_column_matches_the_library` | the library's own constant | doc read 10 against a library at 16; both now read 16, test green |
| **A26-4** two tests pin "the applied order EQUALS the module constant" on a fixture whose disc can only constrain order 10 | **TAKEN by the orchestrator** (sections 5.2, 5.3) | `tests/unit/test_niche_c11_decentred_fit_arbiter.py:314-315`, `tests/unit/test_niche_c1_consolidation.py:240-241` (one assertion each, 4 tests) | the two tests themselves | the fixtures' own in-disc sample counts | the f/6 fixture holds 223 in-disc coarse samples, which the 3-samples-per-term step-down caps at order 10; the assertion read `== _DECENTRED_FIT_POLY_ORDER` and now reads `6 < o <= _DECENTRED_FIT_POLY_ORDER`, i.e. the RAISE rather than the constant.  Both files green |
| **A26-5** "a caller asking for MORE still gets more" asked for 14, which is no longer more | **TAKEN by the orchestrator** (section 5.4) | `tests/unit/test_niche_d7_decentred_fit.py:302-308`, now `newton_poly_order=20` asserting 20 | the ladder in section 3.4 | the analytic Fermat sphere | taking it is what lets the constant be 16 rather than 14: **2.371 / 1.683 urad** instead of 3.718 / 5.419, for 1.56x rather than 1.29x of the order-10 wall clock |
| **A26-6** the weighted restriction's strength is blind to the skirt's residual MAGNITUDE | **DEFERRED** (section 8) | -- | -- | -- | `_FIT_DISC_OUTSIDE_WEIGHT_REL` equalises the skirt's contribution to the Gram, not to the solution; the fit's exposure scales as `w^2 * sum(r_skirt^2)` and nothing bounds `r_skirt` |

**Headline.**  WP-A1's R4 is right, and nothing in `lumenairy/raytrace/` is changed by this work
package.  What it did was stop `_intersect_surface` from falsely killing conic marginal rays, and
on a conic prescription that enlarged the ray set the element's DECENTRED forward fit is handed
from `|h| <= |R| = 1.5106 mm` to the whole launch square, `|h| <= 3.6062 mm`.  D1's weighted
restriction keeps every one of those samples in the least squares by design -- deleting them is
what folds the fitted map -- so the total-degree-10 polynomial D7 sized against the truncated set
now has to follow 2.4x more territory on an f/0.88 conic, and what it trades away out there is
accuracy inside the beam.  The concentric branch restricts by a hard NaN mask instead, which is why
the on-axis figure did not move by one bit and why the bisect looked like an off-axis-only
mechanism.

The fix re-derives the order that raise was always a statement about, 10 -> 16.  Three cures that
do NOT work were measured first and are recorded in section 3, because each is the obvious one and
each fails for a reason worth knowing.

---

## 2. Diagnosis

### 2.1 The reproduction

The orchestrator's bisect table reproduces exactly, driving the CURRENT test module against each
archived tree (child process, `sys.meta_path` cleared of the editable-install finder so the archive
really is the imported library, asserted per probe):

```
tree        on axis   frac 0.5 pre-D7 / D7   frac 1.0 pre-D7 / D7
0067d63b    41.089    134.767 /   2.162      118.337 /   1.958      PASS
f602b72c    41.089    233.859 /  44.457      354.413 /  31.556      FAIL
```

### 2.2 The three arms share ONE `trace()` call, and it is the same call on axis

Instrumenting `raytrace.trace` over `_apply(0.0)`, `_apply(_X0)` and `_apply(_X0, pre_d7=True)`:
each call makes exactly ONE trace, of 405 769 rays (a 637^2 lattice spanning +-2.550 mm at
`ray_subsample=1`), through 2 surfaces -- and the md5 of every returned field is identical across
all three arms within a tree.  `launch_radius = 0.75 * aperture_diameter` does not depend on
`beam_centre`, so the ray data is literally the same data on axis and off; only the FIT's weights
follow the beam.  That is what makes the census below decisive: whatever A1 changed, it changed it
identically for the on-axis arm, and the on-axis arm did not move.

### 2.3 What A1 changed: 294 244 resurrected rays, and nothing else

| field | pre-A1 `0067d63b` | A1 `f602b72c` |
|---|---|---|
| rays traced | 405 769 | 405 769 |
| alive | **111 525** | **405 769** |
| error codes | `RAY_OK: 111 525`, `RAY_MISSED_SURFACE: 294 244` | `RAY_OK: 405 769` |
| `abs(h)` of the alive set | 0.0000 .. **1.5106 mm** | 0.0000 .. **3.6062 mm** |

`1.5106 mm` is `abs(R) = (n - 1) f` for this prescription -- the radius of the SPHERE whose
discriminant the pre-A1 Newton branch used as its miss test.  Of the resurrected rays, 29 638 are
inside the 1.700 mm clear-aperture radius and 264 606 are outside it, out to the launch square's
corner at 3.6062 mm (2.12 clear-aperture radii).

**Purely additive.**  On the 111 525 rays the pre-A1 tracer already kept alive, `max abs(A1 - pre)`
is `0.0e+00` on x, y, z, L, M, N and opd, and no error code differs.  Nothing A1 did moved a ray
that was already being traced; it only stopped killing the ones beyond `abs(R)`.

### 2.4 The resurrected rays are RIGHT

Scored against an inline exact conic raytrace written for this work package (flat entrance at
normal incidence, straight leg to the exact even-conic sag, vector Snell on the exact gradient
normal, straight exit leg to the vertex plane -- no library code), collimated rays from 0.30 mm to
3.6062 mm through `trace()` + `TraceResult.at_exit_vertex(1.0)`:

```
  h[mm]    x_out library [mm]   x_out oracle [mm]   abs(d)
  1.5106      1.257667978          1.257667978      0.0e+00   (last pre-A1 survivor)
  1.5500      1.281101956          1.281101956      0.0e+00   (resurrected)
  1.7000      1.366355096          1.366355096      0.0e+00   (resurrected)
  2.5500      1.750774700          1.750774700      2.2e-19
  3.6006      2.067186847          2.067186847      0.0e+00
  max abs(d x_out) = 2.17e-19 m,  max abs(d OPL) = 6.51e-19 m
```

against 8.10e-04 m of exit-coordinate travel across the resurrected band alone, and against a
float64 evaluation floor of `eps*max|q|` = 4.6e-19 / 5.0e-19 m -- i.e. the two evaluations agree to
the last bit they can.  R4 is verified correct on the
fixture that regressed, so restoring the miss test that killed these rays is not an option and is
not what this work package does.  (`test_a1s_resurrected_conic_rays_hit_the_conic` pins exactly
this, so a later "fix" that re-truncates the conic fails here first.)

### 2.5 Why the on-axis figure did not move by one bit

Instrumenting `_Cheb2DEvaluator.__init__` over the same three arms:

| arm | fits the Newton inversion is handed | order | weights | FINITE samples | `abs(h)` max |
|---|---|---|---|---|---|
| on axis, pre-A1 and A1 | 3 | 6 | no | **39 565** | **0.8999 mm** |
| decentred, pre-A1 | 3 (of 5 builds) | 10 | yes | **111 525** | **1.5106 mm** |
| decentred, A1 | 3 (of 5 builds) | 10 | yes | **405 769** | **3.6062 mm** |

The CONCENTRIC branch restricts by a hard NaN mask at the fit disc (`_fit_disc` -> `np.nan`), so its
sample set is the 0.900 mm disc whatever the tracer does outside it -- 39 565 samples before and
after, and a bit-identical returned field (`max abs(E_A1 - E_pre) = 0.0` over the whole grid).  The
DECENTRED branch is restricted by WEIGHTS instead (niche D1: a hard mask there leaves the fit's
remaining freedom unconstrained and the map FOLDS), so every finite sample enters the least
squares, and the set of finite samples is exactly the set of ALIVE rays.

The skirt -- the down-weighted out-of-disc rows -- therefore went from 71 938 rows spanning
0.900 .. 1.5106 mm to 366 182 rows spanning 0.900 .. 3.6062 mm, at an unchanged per-row weight of
`w_out = 3.288e-05` (the weight is normalised over the out-of-disc LATTICE, which did not change).

### 2.6 Why that costs the fit: the exit map over the enlarged domain

The forward map on this f/0.88 conic is strongly compressive, and the part A1 added is the part
where it bends hardest:

```
  h [mm]         0.30    0.90    1.50    2.00    2.55    3.40
  d(x_out)/dh   0.970   0.796   0.602   0.474   0.370   0.263
```

A total-degree-10 polynomial fitted over `abs(h) <= 1.51 mm` follows that; over `abs(h) <= 3.61 mm`
it does not, and a weighted least squares pays for the misfit where it has freedom to -- inside the
disc.  The exposure scales as `w_out^2 * sum(r_skirt^2)`, and A1 raised `sum(r_skirt^2)` by orders
of magnitude while `w_out` stayed put.

---

## 3. Cures measured, and three that do not work

All four were driven on BOTH fixtures the d7 file carries: the Fermat singlet that regressed, and
D1's own adversarial ghost geometry (`_ghost_apply`: a weak R = 32 mm singlet, 12 mm aperture,
0.40 mm beam at 5.6 mm of decentre, `amplitude_model='ray_density'`), scored the way
`test_no_fold_and_no_ghost_across_the_adversarial_geometries` scores it -- fold-caustic warnings,
off-beam amplitude fraction, and sign changes of `d(x_out)/dx` of the APPLIED forward-map fit over
the whole launch lattice.

### 3.1 Bound the fit's sample set at the clear aperture -- RESTORES THE ORACLE, AND FOLDS

Hard-bounding the traced set by entrance height (the element's own `aperture_diameter` is the only
physically meaningful edge: the lens is opaque beyond it):

| bound | frac 0.5 pre-D7 / D7 | frac 1.0 pre-D7 / D7 |
|---|---|---|
| unbounded (HEAD) | 233.859 / **44.457** | 354.413 / **31.556** |
| clear aperture, 1.700 mm | 134.731 / **2.290** | 122.252 / **4.042** |
| 1.2 clear apertures, 2.040 mm | 133.861 / 5.464 | 162.331 / 7.670 |
| launch radius, 2.550 mm | 130.785 / 11.934 | 314.436 / 18.920 |
| 0.99 launch radius, 2.525 mm | 129.981 / 11.631 | 309.390 / 18.219 |
| `abs(R)` = 1.511 mm (what pre-A1 did by accident) | 134.767 / 2.162 | 118.337 / 1.958 |

The clear-aperture bound recovers the oracle, and only it does -- trimming the launch square's
corners is not enough.  It was implemented as a ZERO WEIGHT on the out-of-aperture skirt (so the
OPL / exit-coordinate grids stay intact for the paraxial-magnification stencil, the process-pool
knot data and the direct-fit exit hull) and **refused on measurement**: with it in place,
**11 of the 13** `test_niche_d7_decentred_fit.py` ghost tests fail with
`amplitude_model='ray_density' detected a fold caustic`, and the same 13 pass with the bound
disengaged in the same process.  D1's skirt is not only regularisation: it is what keeps the fitted
map single-valued over the region the Newton inversion actually evaluates, which is the whole
launch square and not the aperture.

### 3.2 A two-class skirt (full weight inside the aperture, `FAR * w_out` outside) -- NO WINDOW

| FAR | frac 0.5 / 1.0 (D7 arm) | ghost folds | ghost off-beam | ghost sign changes |
|---|---|---|---|---|
| 1.0 (shipped) | 44.457 / 31.556 | 0 | 1.76e-04 | **0** |
| 1e-1 | 10.366 / 15.957 | 0 | 1.76e-04 | **0** |
| 1e-2 | 6.791 / 9.452 | 0 | 1.76e-04 | **212** |
| 1e-3 | 3.929 / 5.563 | 1 | 1.76e-04 | 878 |
| 1e-4 | 2.310 / 4.100 | 1 | 1.70e-04 | 1083 |
| 1e-6 | 2.290 / 4.042 | 1 | 1.70e-04 | 950 |
| 0.0 | 2.290 / 4.042 | 1 | 1.76e-04 | 950 |

The FAR that fixes the Fermat fixture is the FAR that folds D1's.  The fold is a RELATIVE
phenomenon -- what matters is that the far region is not much less constrained than the near one --
so any radially graded weight re-creates exactly the defect D1's restriction exists to prevent.

### 3.3 Lower `_FIT_DISC_OUTSIDE_WEIGHT_REL` uniformly -- fold-safe, but no plateau, and it breaks physics elsewhere

A uniform scaling does NOT fold (the balance within the skirt is preserved), and D1's ghost fixture
is flat across six decades of it:

| `_FIT_DISC_OUTSIDE_WEIGHT_REL` | frac 0.5 / 1.0 | ratio to on-axis | ghost folds / off-beam / sign changes |
|---|---|---|---|
| 1e-8 (shipped) | 44.457 / 31.556 | 1.0820 / 0.7680 | 0 / 1.76e-04 / 0 |
| 1e-10 | 9.840 / 16.224 | 0.2395 / 0.3949 | 0 / 1.70e-04 / 0 |
| 1e-12 | 7.073 / 5.779 | 0.1721 / 0.1406 | 0 / 1.70e-04 / 0 |
| 1e-14 | 3.450 / 2.331 | 0.0840 / 0.0567 | 0 / 1.75e-04 / 0 |

Refused for two reasons.  The Fermat fixture has no plateau -- it improves monotonically all the
way to the hard-mask limit, so any value is a judgement rather than a measured middle; and at
1e-13 the test surface shows a real physics regression,
`test_niche_d1_tilted_carrier.py::test_tilted_relay_reaches_the_on_axis_diffraction_limit` fails,
along with 13 others.  Weakening the skirt weakens what the skirt is FOR.

### 3.4 Raise the order the fit is given -- SHIPPED

The skirt's residual is large because the polynomial cannot follow the data it is handed.  Giving
it the terms to follow it fixes the cause rather than suppressing the symptom, and costs D1's
fixture nothing at all:

| order | basis terms | frac 0.5 | frac 1.0 | ratio to on-axis | ghost folds / off-beam / sign changes |
|---|---|---|---|---|---|
| **10 (pre-A26)** | 66 | **44.457** | **31.556** | 1.0820 / 0.7680 | 0 / 1.76e-04 / 0 |
| 12 | 91 | 10.841 | 11.829 | 0.2639 / 0.2879 | 0 / 1.76e-04 / 0 |
| 14 | 120 | 3.718 | 5.419 | 0.0905 / 0.1319 | 0 / 1.76e-04 / 0 |
| **16 (shipped)** | **153** | **2.371** | **1.683** | **0.0577 / 0.0410** | 0 / 1.76e-04 / 0 |
| 18 | 190 | 1.044 | 0.655 | 0.0254 / 0.0159 | 0 / 1.76e-04 / 0 |
| 20 | 231 | 0.321 | 0.301 | 0.0078 / 0.0073 | 0 / 1.76e-04 / 0 |
| 24 | 325 | 0.071 | 0.057 | 0.0017 / 0.0014 | 0 / 1.76e-04 / 0 |

**Why 16.**  The curve is monotone, so the value is a cost/accuracy choice and the choice is made
against the figure this element returned before the ray set stopped being truncated -- 2.162 urad at
0.5 w and 1.958 at 1.0 w (`0067d63b`).  16 is the LOWEST order on the ladder that reaches that scale
on BOTH decentres: 2.371 urad (1.10x of it) and 1.683 urad (0.86x).  14 is still 1.7x and 2.8x short
of it, and 18 / 20 / 24 buy more accuracy than the pre-truncation element ever had, at 190 / 231 /
325 terms.  16 also clears the D7 acceptance bar (`decentred < 0.25x on-axis`) by 4.3x and 6.1x,
against 2.8x and 1.9x at 14.  The cost is 1.56x of the order-10 wall clock on the off-centre branch
(section 4).

---

## 4. What the raise does NOT touch, measured

**Conditioning.**  D7's own note records that on design 121's last group "order 14 starts to LOSE
to conditioning (the normal-equations Gram matrix runs 1.0e10 -> 1.9e13 across the sweep)".  That
table predates niche C13's `LSTSQ_CONDITIONING_STEPDOWN` (2026-08-03, shipped `True`).  Re-measured
here with the census the C13 tests use -- Gram rcond and `||b - Ax||` against an independent QR
solve of the same system:

| fixture / order | min rcond | max `||b-Ax|| / ||b-Ax_qr||` | largest design matrix |
|---|---|---|---|
| Fermat singlet, 1.0 w, order 10 | 5.025e-15 | 1.000009 | (101761, 120) |
| Fermat singlet, 1.0 w, order 14 / **16** / 18 / 20 | 5.025e-15 | 1.000009 | (101761, 120 .. 231) |
| D1 ghost, order 10 | 1.802e-14 | 1.012192 | (203401, 120) |
| D1 ghost, order **16** / 18 / 20 | 1.802e-14 | 1.012192 | (203401, 153 .. 231) |

Identical at every order, on both fixtures -- and identical again with the step-down forced OFF
(2.114055 at every order on the Fermat singlet, 1.012192 on the ghost).  The worst solve of the
call is the inverse-characteristic model's own total-degree-14 exit fit (120 columns), not the
decentred forward fit, so the extra terms cost the solve nothing here.  Design 121's own fixture is
local-only (`validation/repro_traced_carrier_121/decentred_fit_defect.py` needs the .zmx and the
design-study runner) and is NOT re-measured; that is stated in the constant's note.

**The concentric path.**  Byte-identical: `apply_real_lens_traced(..., beam_centre=(0, 0))` and the
same call with `decentred_fit_poly_order=10` are `np.array_equal`, with and without a carrier.  The
raise is engaged only on the off-centre branch, which is what D7 shipped and what
`test_the_concentric_path_is_untouched` pins.

**D1's fold regularisation.**  Unmoved across the whole ladder (table in section 3.4) and green:
all 13 ghost tests in `test_niche_d7_decentred_fit.py` pass.

**The step-down.**  Unchanged in form; only its arithmetic moves (order 16 -> 153 terms -> 459
in-disc coarse samples).  It is what keeps `test_niche_c11_decentred_fit_arbiter.py` and
`test_niche_c1_consolidation.py` honest: their f/6 fixture holds 223 in-disc samples at
`ray_subsample=8`, so the applied order there is 10 whatever this constant says, and those two tests
now assert the RAISE (`6 < o <= _DECENTRED_FIT_POLY_ORDER`) rather than the constant.
`test_the_step_down_still_protects_an_undersampled_disc` pins the walk itself at
`ray_subsample=16`, where the disc holds too few samples and the order steps back down one
degree at a time rather than jumping to `newton_poly_order`.

**Niche D6's on-axis EE2 ratio -- the figure WP-A24 bisected to `f602b72c` as a -0.17 % move
(`0.971526 -> 0.969787`) -- does not move at all.**  Driven in one process with only
`_DECENTRED_FIT_POLY_ORDER` varying, on the d6 file's own fixture and inline Kirchhoff oracle:

```
  oracle EE2  on 0.783532630   off 0.716264664
  order 10    r_on 0.969786923   r_off 0.985517727   fwhm_on 3.1500 um
  order 16    r_on 0.969786923   r_off 0.985517727   fwhm_on 3.1500 um
```

identical to nine digits on both arms, and `r_on` reproduces WP-A24's 0.969787 exactly.
`tests/unit/test_niche_d6_exact_tilted_leg.py` is 38 passed.

**Cost.**  The Newton hot loop evaluates these fits per output pixel, so 153 terms against 66 is
not free.  Medians of 7 INTERLEAVED runs of one decentred `apply_real_lens_traced` (N = 512,
dx = 8 um, `ray_subsample=8`), box shared with other jobs: **248.2 ms at order 10 against 386.2 ms
at 16, i.e. 1.56x** (a 3-way run of the same fixture puts 14 between them at 320.0 ms / 1.29x).
Paid only on the off-centre branch; the concentric path is byte-identical and therefore unchanged in
cost.

---

## 5. Changes outside my ownership -- requested, and TAKEN by the orchestrator

Each is a number or an assertion FORM that encoded the old value of `_DECENTRED_FIT_POLY_ORDER`.
No claim in any of them changed.  All four have been applied by the orchestrator (they are not in my
file list, section 6), and the re-run in section 7 is on the tree with them in place; 5.4 is what
allows the shipped 16 rather than 14.

### 5.1 `docs/audits/TRACED_LAYER_MAP.md:60` -- one cell  *(TAKEN)*

`test_niche_c14_encapsulation.py::test_the_layer_map_shipped_column_matches_the_library` reported
`_DECENTRED_FIT_POLY_ORDER: doc says 10, library has 16`.  The cell now reads `16` and the test is
green.

That file's OPEN ITEM 1 at `:424` -- "the `_DECENTRED_FIT_POLY_ORDER` order-10 anomaly ... orders 6,
8 and 12 all close the `(-1,0)` chain residual and the shipped 10 does not" -- is an independent
measurement, taken before this work package, saying 10 was an unlucky value on design 121's chain;
it points the same way as the ladder in section 3.4 and it is NOT closed by anything measured here,
because design 121's fixture is local-only.  The orchestrator's `_traced_flags.py:194-197` note now
says exactly that: order 10 does not close the chain, and the shipped 16 is not yet measured on it.

### 5.2 `tests/unit/test_niche_c11_decentred_fit_arbiter.py:314-318`  *(TAKEN)*

```python
            assert applied and all(
                o == _lt._DECENTRED_FIT_POLY_ORDER for o, _w in applied), seen
```
The f/6 fixture (`_SLOW`, `ray_subsample=8`) holds 223 in-disc coarse samples, and the
3-samples-per-basis-term step-down caps that at order 10 (`3*11*12/2 = 198 <= 223 < 3*12*13/2 =
234`).  The claim -- "the weighted RAISED ORDER path still engages" -- is intact: the applied order
is 10 against a `newton_poly_order` of 6, and the weights arm passes.  What became too strong is
reading the CONSTANT where the test means the RAISE; the assertion silently assumed the step-down
never fires on this fixture.  Applied:

```python
            assert applied and all(
                6 < o <= _lt._DECENTRED_FIT_POLY_ORDER for o, _w in applied), seen
```

The sibling `test_the_scored_candidate_is_the_applied_candidate` does NOT need this; it compares the
applied fit to the winning TRIAL fit rather than to the constant, and it passes.

### 5.3 `tests/unit/test_niche_c1_consolidation.py:239-243` (3 parametrisations)  *(TAKEN)*

```python
            assert seen and all(o == _lt._DECENTRED_FIT_POLY_ORDER
                                for o, _w in seen), seen
```
Identical shape, identical fixture density (`ray_subsample=8`), identical restatement
(`6 < o <= _lt._DECENTRED_FIT_POLY_ORDER`).  The `all(o == 6 ...)` concentric arm three lines below
it is unaffected and stays as written.

*(A fourth failure seen in the same file while this work package was in progress,
`::test_the_focus_readout_whitelist_is_exactly_what_the_chain_consumes`, was never mine: it reported
`Extra items in the right set: 'replica_fill'` and reproduced with `_DECENTRED_FIT_POLY_ORDER`
forced back to 10.  It was WP-A25's new keyword, and it went away when WP-A25 committed at
`38bcc9c2`.)*

### 5.4 `tests/unit/test_niche_d7_decentred_fit.py:302-308` -- what decides 16 over 14  *(TAKEN)*

This is the edit the shipped value rests on.  The arm

```python
        # a caller asking for MORE still gets more
        seen.clear()
        _apply(_X0, newton_poly_order=14)
        seen[:] = seen[-3:]
        assert seen and all(o == 14 for o, _w in seen), seen
```
asked for 14 because 14 was more than the old constant of 10.  With the constant at 16 the
effective order is `max(14, 16) = 16` -- the documented semantics, and the test's own claim -- so
the number had to move above the constant.  Applied:

```
-        _apply(_X0, newton_poly_order=14)
+        _apply(_X0, newton_poly_order=20)
         seen[:] = seen[-3:]
-        assert seen and all(o == 14 for o, _w in seen), seen
+        assert seen and all(o == 20 for o, _w in seen), seen
```

(20 works on this fixture: its disc holds ~9 900 coarse samples at `ray_subsample=2` against the 693
an order-20 fit needs.)  With it, `_DECENTRED_FIT_POLY_ORDER = 16` reads **2.371 / 1.683 urad**
instead of 14's 3.718 / 5.419 -- at or below the pre-truncation figure on both decentres -- and
widens this report's own envelope bar from a 5.8x separation to 13x, for 1.56x rather than 1.29x of
the order-10 wall clock.  `tests/unit/test_niche_d7_decentred_fit.py` is 38 passed with the edit in
place.

### 5.5 Two local-only repro scripts state the old value in prose

`validation/repro_traced_carrier_121/imap_cost_121.py:265` and `imap_probe_121.py:459` say
"`_DECENTRED_FIT_POLY_ORDER` ... to 10".  Cosmetic; both scripts read the live constant.

---

## 6. Files touched

| file | what |
|---|---|
| `lumenairy/elements/_lens_traced.py` | `:4086` `_DECENTRED_FIT_POLY_ORDER` 10 -> 16; `:4000-4052` the second half of its derivation (the data-domain constraint, the order ladder and why 16, the measured cost, the fold and conditioning invariance); `:8137-8193` the `decentred_fit_poly_order` parameter doc; `:4256-4274` the step-down arithmetic; `:4364-4377` `_DECENTRED_FIT_SPECTRUM_ORDER`'s headroom statement |
| `docs/history/lumenairy.elements._lens_traced.md` | re-recorded fingerprints (`scripts/record_history_fingerprints.py`) |
| `tests/unit/test_audit2609_a26_decentred_exit_reference.py` | new, 10 tests |
| `docs/audits/.../fixes/WP-A26_REPORT.md`, `WP-A26_CHANGELOG.md` | new |

Nothing under `lumenairy/raytrace/` is changed: A1 is correct (section 2.4).  The four files in
section 5 are the orchestrator's, not mine.

---

## 7. Tests run

All with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, `-p no:randomly`.

Re-run end to end at the shipped `_DECENTRED_FIT_POLY_ORDER = 16`, on the tree with section 5's
four edits in place and WP-A25 committed at `38bcc9c2`, one process at a time:

| command | result | duration |
|---|---|---|
| `pytest tests/unit/test_niche_d7_decentred_fit.py` | **38 passed** | 276.74 s |
| `pytest tests/unit/test_audit2609_a26_decentred_exit_reference.py` (new) | **10 passed** | 58.68 s |
| `pytest` the four A1 files + `test_niche_c11_decentred_fit_arbiter.py` + `test_niche_c1_consolidation.py` + `test_niche_c14_encapsulation.py` | **190 passed** (the A1 set is 99 of them) | 179.13 s |
| `pytest tests/unit/test_niche_d6_exact_tilted_leg.py` | **38 passed** | 165.52 s |
| `pytest tests/unit -k real_lens` | **160 passed, 3 skipped** (PySide6, host-specific digests, numexpr) | 237.31 s |
| `pytest tests/unit -k "raytrace or exit_vertex or seidel or opd_fan"` | **591 passed, 2 skipped** (PySide6, rayoptics) | 248.09 s |
| wall-clock medians, 7 interleaved runs, decentred call | **248.2 ms at order 10, 386.2 ms at 16 (1.56x)** | 30 s |
| `ruff check` on `_lens_traced.py` and the new test file | **All checks passed** | 1 s |
| `python scripts/record_history_fingerprints.py --check` | **OK: every history document matches its module** | 3 s |

Carried over from the order-14 pass earlier in this work package, on the same tree minus section 5's
edits (re-run not repeated because nothing in them reads this constant):
`python validation/run_all.py test_raytrace test_lenses` **ALL 2 files passed** (raytrace 54/54,
24.8 s + 12.2 s), and `test_niche_c12_physics_fit_selection.py`,
`test_niche_c13_lstsq_conditioning.py`, `test_niche_c6_fit_guard.py`,
`test_fix_d5_fit_domain_basis.py`, `test_niche_d8_congruence_workers.py`,
`test_niche_d1_tilted_carrier.py` all green.

### 7.1 The refused cures, and how they were scored

The `-p a26_order16` / `-p a26_rel13` plugin runs in section 3 drove ten files with one constant
patched at `pytest_configure`, so that the tree itself never varied.  At the time (before section 5
was taken) order 16 gave 7 failed / 289 passed -- the four order pins of 5.1-5.4, plus WP-A25's
`replica_fill`, all of which are now resolved -- while `_FIT_DISC_OUTSIDE_WEIGHT_REL = 1e-13` gave
**14 failed / 282 passed** including
`test_niche_d1_tilted_carrier.py::test_tilted_relay_reaches_the_on_axis_diffraction_limit`, which is
a physics regression and is why that cure is refused rather than merely dispreferred.  The
clear-aperture bound of section 3.1 was driven the same way and failed 11 of the 13 d7 ghost tests
with a fold-caustic warning.

---

## 8. Deferred

**A26-6 -- the weighted restriction's strength is blind to the skirt's residual magnitude.**
`_FIT_DISC_OUTSIDE_WEIGHT_REL`'s formula, `w_out = sqrt(REL * n_in / n_out)`, makes the skirt's
contribution to the normal matrix `REL` of the disc's -- a statement about the GRAM, i.e. about the
design matrix alone.  What decides how far the skirt pulls the solution is
`w_out^2 * sum(r_skirt^2)`, and nothing in the design bounds `r_skirt`.  That is exactly why A26
happened: the residual grew (a 2.4x larger domain on a fast conic) while the weight, sized against
the row count, did not notice.  The constant's own note already flags the same blind spot from the
other direction ("treat 1e-14..1e-8 as a plateau MEASURED AT LOW NA with a small beam").

A regime-independent restriction would normalise on the fit's own residuals rather than on row
counts -- one extra scalar per fit, computed from a first pass, radially uniform so the fold cure is
untouched, and a no-op wherever the skirt residual already matches the disc's (which is D1's own
calibration fixture, so the shipped behaviour there would not move).  It is a design change with
its own calibration surface across `c11` / `c12` / `c13` / `d1` / `d5` and design 121, and it is not
a regression fix; recorded here rather than attempted.

**A26-7 -- the launch lattice is a SQUARE inside a DISC-shaped domain.**  `bound` and
`out_of_domain` define the element's forward-map domain as a disc of radius `~launch_radius`, yet
the fit is fed the square's corners at `sqrt(2) * launch_radius`, which the Newton inversion is
clamped never to return.  Trimming them is NOT a cure for A26 (measured: 11.934 / 18.920 urad,
section 3.1) and doing it by deletion folds the map, so nothing is proposed here -- but the
inconsistency is real and it is what puts the fit's worst data 1.41x beyond its own domain.
