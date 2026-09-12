# VERIFY-A7 -- independent adversarial re-verification of WP-A7 (`lumenairy/analysis/`)

Verifier: VERIFY-A7. Diff under test: commit `1ada5bc5`
(`fix(analysis): WP-A7 -- masked 2-D phase unwrap, exact-sphere Strehl reference, ...`),
base `8610e97e`, 16 source files + 5 new / 4 amended test files inside
`lumenairy/analysis/` and `tests/unit/`.

Everything below was **re-measured** on this machine (CPython 3.14.6, numpy 2.4.6,
scipy 1.17.1, jax 0.10.1, `OPENBLAS_NUM_THREADS=1`, 2026-09-12). Fixtures were chosen
to differ from the WP's wherever possible -- 532 nm and 1064 nm and 850 nm instead of
633 / 600 / 587.6 nm, odd and rectangular and anamorphic grids, annular and decentred
supports, three lens shapes -- so that a fixture-specific coincidence cannot carry both
sets of tests. Oracles are analytic or hand-written here; the A4 oracle is a complete
sequential ray trace that imports nothing from `lumenairy.raytrace`.

---

## 1. Verdicts

| ID | Verdict | Independent oracle | Re-measured |
|---|---|---|---|
| **A1** unwrap | **VERIFIED** (2 notes) | analytic Zernike pupils on annular / decentred / odd-N / anamorphic / float32 / vortex / disconnected supports | max abs OPD error **2.4e-15 .. 4.6e-7 waves** on 14 fixtures where the pre-fix code was wrong by **3-5 whole waves**; `repro/orch/verify_analysis.py` 1.000 waves / 53.2 % -> **0.000 / 0.0 %** |
| **A2** Strehl | **VERIFIED** | pupil aberration-free by construction; `strehl_phase_integral` (independent module); pre-fix reference restored in-process | Strehl **1.000000000** (9 figures) at f/50, f/10, f/5, f/2.5, f/2, both signs of f, decentred, odd N, rectangular, annular -- where the paraxial reference gives 1.107-1.468; `p6` best Strehl for a perfect f/2 lens **1.0000** |
| **A3** Shack-Hartmann | **VERIFIED-WITH-NOTES** | analytic tilt / defocus / astig-0/90 / **astig-45 (non-separable)** / coma, and exact synthetic slopes through `_reconstruct_wavefront` | reconstruction scale **0.9926 .. 1.0473 x slope gain** (pre-fix 0.5 x); solve exact to **1.7e-15 .. 3.4e-14 of span** on discs and annuli up to 64x64; NaN set == un-measured set on all three partial-illumination fixtures. Note A3-N1 below. |
| **A4** image-plane WFE | **VERIFIED-WITH-NOTES** | a full sequential ray trace written in the test file (exact sphere intersection, vector Snell, external Sellmeier) + textbook thick-lens BFL | `img_d_m` == BFL to **0 .. 3.9e-16 relative** on 3 lens shapes x 2 wavelengths; the whole OPD map reproduces the independent trace to **1.9e-11 waves on-axis**, and off-axis to **1.0e-4 / 4.1e-4 waves at 1 / 2 deg** whose entire content is a pure defocus term (shape residual **5.5e-9 / 3.0e-8**) on PVs of 3.59-6.31 waves. Notes A4-N1 (P2) and A4-N2 below. |
| **A5** GS / plotting | **VERIFIED** | exactly-solvable GS problem; pixel-centre identity | GS err **5.0e-30 / 1.9e-29 / 1.1e-28** at N = 32 / 64 / 128 against target energies 6.4e1 / 2.6e2 / 1.0e3; NumPy/JAX err ratio **1.000001** on a non-degenerate target (was `N_pix`); `_auto_extent` drawn centres coincide with the samples to **<= 3.6e-15 of dx** for even AND odd N, and now match `plot_psf` exactly |
| **A6** performance | **VERIFIED-WITH-NOTES** + **1 defect found and fixed** | per-plane `angular_spectrum_propagate`; `beam_centroid`/`beam_d4sigma`; the masked sum | recurrence engages on `linspace` (**2 vs 21 `np.exp` calls**, measured); 200-plane scan drift **4.5e-12** against 200 independent propagations (bound 5.2e-12); `single_plane_metrics` **bit-identical** (`==`) on 3 fixtures incl. anamorphic; `radial_power_bands` bit-identical at n=95, 5.5e-14 relative at n>=96. **Defect A6-D1 (NaN radius) found and fixed.** |
| **A7** P3 rows | **VERIFIED** | uniform-disc 1/sqrt(2); DFT bin of the y ramp; runtime export check; cache-key collision | ghost 50 %-EE **0.750 / 0.667 / 0.667 / 0.696 / 0.701 R** at 4 / 6 / 12 / 24 / 48 rings, converging on 0.7071, while the median ray is **0.500 R at every ring count**; the y fringe lands on the exact designed DFT bin with `dy` and on `dy/dx` times the wrong bin without it; all 7 new names resolve; the Zernike key now separates interior-warped grids and still separates `dy != dx` |

**No regression found.** 442 + 223 + 50 unit tests and all 3 validation topic files pass
(§5). Every item the audit listed as *verified correct* still measures the same (§4).

---

## 2. Per finding

### A1 [P0] -- masked 2-D unwrap

**Repro re-run.** `repro/orch/verify_analysis.py` (2): `max |OPD error| = 0.000 waves,
0.0 % of the pupil wrong`, at the same 0.27 rad/sample, against the audit's
`1.000 waves / 53.2 %`. `repro/ANALYSIS/p2c`: `0.000 waves` and the exact true `c8` at
every amplitude from 0.05 to 5.0 lambda rms. `p2b`: offset histogram `{0: N}` on all
five geometries. All three match the WP's claimed after-numbers.

> **One caveat on reading `p2c`.** Its last column still prints
> `fitted RMS-WFE = 0.4980 w (true 0.2000)` in the `f_ref` block, which looks unfixed.
> It is not: that column is `sqrt(sum c[1:]^2)` of the ABSOLUTE map, which legitimately
> contains the -0.45604-wave reference-sphere defocus. Decomposing the returned map and
> the truth map side by side, **every coefficient agrees to 5 decimal places** (piston
> -0.78989 both, defocus -0.45604 both, c8 +0.20000 both, everything else 0.00000) and
> `max |err| = 1.7e-15 waves`. The script's label is wrong, not the library.

**New adversarial fixtures (none used by the WP).** All at 532 nm, 0.9 waves rms coma +
0.5 waves rms spherical unless noted; the "pre-fix" column is the row-then-column
`np.unwrap` restored in-process on the identical field.

| fixture | pre-fix max abs err | post-fix |
|---|---|---|
| odd 257x257 | 3.0000 waves | 3.4e-15 |
| odd/rect 255x321 | 3.0000 | 4.6e-15 |
| odd + anamorphic dy = 2.45 dx | 4.0000 | 2.4e-15 |
| even/odd + anamorphic dy = 2.44 dx | 3.0000 | 5.0e-15 |
| annulus eps = 0.4 | 4.0000 | 3.6e-15 |
| annulus eps = 0.7 | 4.0000 | 2.8e-15 |
| decentred (+60, -40) um | 3.0000 | 5.0e-15 |
| decentred (+110, +90) um | 4.0000 | 4.4e-15 |
| complex64 field | 5.0000 | 4.6e-07 |
| F-contiguous / strided view | -- | bit-identical to the C-contiguous result |

Residue reporting fires on every genuine residue: charge +1, +2 and -1 vortices all warn
with `residue = 1.000 waves`; uniformly random phase warns; a well-sampled pupil does not
(checked with `simplefilter('error')`). A disconnected support warns with the component
count and each island is exact after its own piston removal (7.96e-16 and 5.97e-16 waves).
`unwrap='reliability'` lands on the identical branch as `'itoh'` on both a disc and an
annulus (max difference 6.8e-15 waves, identical piston), is 46.5x slower at N = 256 (the
docstring says ~50x), and refuses above 1.1e6 samples with the documented message.

**Robustness probes.** 1xN, Nx1, 1x1 fields; a single valid row or column inside a 2-D
grid; apertures of 1, 5 and 13 samples; an all-zero field -- all return without raising,
all exact. Pathological masks stress the Python BFS over runs without pathology: an
every-other-column mask at N = 512 (205 components) costs 177 ms and a checkerboard
(65 865 single-sample components) 97 ms, against 39 ms for the solid disc; both warn about
the disconnection, and the residual error is exactly the undetermined per-island piston.
`unwrap_phase_2d` returns out-of-mask samples unchanged, is congruent to the wrapped input
mod 2 pi to 2.8e-16, and rejects a 1-D input and a mismatched mask with the §2 prefix.

**Cost.** `wave_opd_2d` at N = 1024 is 1.18x the pre-fix unwrap-only cost (the WP says
1.10x; same direction, my baseline excludes the aperture/meshgrid work the WP's included)
and 0.79x at N = 512. `tracemalloc` peak 10.4 float64 grids at both sizes.

**Note A1-N1 [P3, documentation].** The `wave_opd_2d` Notes say the anchor makes "a pupil
carrying a known defocus come back with the right INTEGER WAVE COUNT". That holds for a
CENTRED pupil only -- the anchor is the valid sample nearest `x = y = 0`, which is the
wavefront's stationary point only then. Measured on a 1.2-waves-rms coma pupil: absolute
piston **0.0000** waves centred, **+1.0000** at (+60, -40) um, **+2.0000** at (+110, +90)
um. The offset is always an exact whole number (so the map stays congruent to the wrapped
phase -- it is a gauge, not an error), and the shape is exact in every case. Pinned both
ways in `test_offcentre_pupil_shape_is_exact_and_the_piston_gauge_is_documented`.

### A2 [P0] -- exact-sphere Strehl reference

`repro/ANALYSIS/p6`: `diffraction_limited_peak = 1.519335e+05` = the perfect sphere's own
peak, `through_focus_scan best Strehl for a PERFECT lens = 1.0000`, `Strehl at z = f
exactly = 1.0000` -- the WP's numbers to the digit. `p5` §C(ii) reads
`ASM paraxial/sphere peak = 1.00000` at both resolvable geometries; `p3` §F reads
`paraxial-ref peak / true-sphere peak = 1.00000` at f/25 down to f/2.
`repro/orch/verify_analysis.py` (1): `peak/reference = 1.000`.

**New fixtures at 1064 nm** (an aberration-free exact-sphere pupil, propagated by the same
ASM): Strehl `1.000000000` at f/50, f/10, f/5, f/2.5 and f/2. **At 532 nm, f/2.5 and
f/10, both signs of f, and on decentred / odd-N / rectangular / annular supports**:
`1.000000000` on every arm, where the pre-fix quadratic reference restored in-process
gives 1.127579 / 1.468475 / 1.126358 / 1.126121 / 1.106679 and 1.000063 at f/10. That is
5 decades of fail-before margin on the fast arms.

Cross-checks: the ASM Strehl of a known-aberrated pupil agrees with `strehl_phase_integral`
(a different, audit-verified module) to 0.3 % at lambda/14 and 0.2 % at lambda/7, and with
Marechal to 0.007 % at lambda/14. Every Strehl denominator in the library routes through
`diffraction_limited_peak` (23 references, 12 call sites; the only surviving paraxial
quadratic in `analysis/` is `wave_opd_2d`'s `f_ref` conditioning term, which is removed and
added back exactly).

*One weakness in the WP's own test.* `test_negative_focal_length_keeps_the_paraxial_sign`
re-derives `sign(f) (sqrt(r^2+f^2) - |f|)` inside the test and only asserts
`isfinite and > 0` of the library call, so it would pass against an `abs(f)`-only
implementation. My `test_strehl_is_one_for_both_signs_of_f` exercises the branch through
the library.

### A3 [P1] -- Southwell Shack-Hartmann reconstruction

`repro/ANALYSIS/p4` §A reproduces the WP's after-numbers exactly: 0.9481 / 0.9488 / 0.9450
for tilt (= the slope gain the same run reports), 0.9340 for defocus,
`max abs wf = 1.4351e-06 m` against a truth of 1.5000e-06 m.

**New fixtures at 850 nm.** Reconstruction scale against the analytic wavefront, divided by
the run's own measured slope gain:

| input | scale (1.0000 = correct) |
|---|---|
| tilt 0.3 mrad (x) | 1.0000 |
| tilt 0.3 mrad (x + 0.6 y) | 1.0000 |
| defocus | 0.9926 |
| astigmatism 0/90 | 0.9898 |
| **astigmatism 45 (W = 2c x y, NOT separable)** | **0.9986** |
| coma | 1.0473 |

The 45-degree astigmatism row matters: it is the case the pre-fix average of two one-sided
integrals could not even be *described* as halving. Isolated from the sensor and fed exact
analytic slopes, the solve is exact -- **1.7e-15 to 3.4e-14 of the wavefront span** over
172 to 3112 nodes on discs and annuli (n = 16, 17, 33, 64), for tilt, defocus,
astigmatism-45 and their sum; the residual 3.9e-3 on a cubic wavefront is the trapezoid
quadrature, identical for both methods, not the solve. Cost: 2 ms at 16x16, 15 ms at
64x64, **384 ms at 256x256** (the WP documented "~1 s", conservative).

Partial illumination: the NaN set of `wavefront` equals the un-measured set exactly on a
circular pupil (50 / 64 measured), an annulus (48 / 64) and two separated blocks (32 / 64).
A mask split into two components is gauged on each component's own lowest-index member
(`w[0,0] == 0` and `w[0,4] == 0`), the relative piston correctly undetermined.
Flat wavefront still returns bit-exact 0.

**Note A3-N1 [P3, documentation].** The `_reconstruct_wavefront` docstring says "Lenslets
with no measurement ... are excluded from the solve and returned as NaN rather than
integrated through as if they had measured zero slope." That paragraph reads as covering
both methods, and it is **false for `method='itoh'`**: `_itoh_wavefront` still does
`np.where(good, slopes, 0.0)` and integrates straight through a hole. Measured on the
annular fixture: southwell 8.7e-15 of the span, **itoh 4.9e-1**. Not a default-path defect
(`'southwell'` is the default), but the sentence should be moved under the Southwell
paragraph or qualified. Pinned by
`test_itoh_integrates_through_holes_where_southwell_does_not`, which fails loudly if the
behaviour is ever unified.

`shack_hartmann` also does not warn when the measured lenslet set is disconnected, where
`wave_opd_2d` does warn for the same situation. Consistency suggestion, not a defect.

### A4 [P1] -- `object_distance = inf`

`repro/ANALYSIS/p12` re-runs unchanged in the finite regime (chief z0 +0.0006 / +0.0238 /
-5.34 / +589.41 um at 1e3 / 1e4 / 1e5 / 1e6 m), as it must -- the fix adds an infinite
branch rather than repairing the finite one. The warning is silent at 1, 1e2, 1e3 m and
fires at 1e4, 1e5, 1e6 m, exactly as claimed. `best_rms` at infinity lands at
**47.4773 mm, PV 0.9425, RMS 0.2723** and at 1e6 m still at **96.6435 mm** on a 47.9 mm-BFL
lens -- the WP's numbers reproduce to the digit.

**Independent oracle.** I wrote a complete sequential ray trace (exact sphere intersection,
vector Snell, OPL as `n*t`, Schott's published N-BK7 Sellmeier, textbook thick-lens BFL)
that imports nothing from `lumenairy.raytrace`:

* `img_d_m` equals the thick-lens BFL to **0.00e+00 .. 3.88e-16 relative** for biconvex,
  plano-convex and meniscus singlets at 587.6 and 486.1 nm.
* The **whole OPD map**, ray by ray, reproduces the oracle to **1.9e-11 waves on-axis** on
  a map of PV 3.5944 waves -- nine decades better than the "1.8 %" the WP's transverse-ray
  oracle could show -- and it covers the off-axis `field_max_rad` launch the WP's own tests
  only check for finiteness.
* Off-axis the two disagree by 2.6e-5 / 1.0e-4 / 4.1e-4 waves at 0.5 / 1 / 2 deg, and
  fitting that difference to `piston + tilt + rho^2 + rho^3 + rho^4` shows it is
  **entirely a defocus term** (rho^2 coefficient -1.06e-4 at 1 deg, -4.24e-4 at 2 deg;
  everything else below 5.5e-9 and 3.0e-8 waves). That is the two sides taking the
  reference-sphere tangent point at O(theta^2) different places -- both legitimate
  readings of "the chief's path length from the last-surface vertex plane", a 46 nm
  difference in `r_sphere_m` -- and it is exactly what `image_plane='best_rms'` exists to
  remove. The test therefore asserts the raw agreement (5e-3 waves) AND the shape with
  piston/tilt/defocus projected out (1e-5 waves), so a real error in the collimated
  launch, the trace or the sphere cannot hide inside the focus term.
* A dissection confirms *why*: library and oracle produce identical exit-ray state
  (`y` to 0 nm, `M` to 1e-12, chief-relative OPL to 5 decimals in waves) and the identical
  reference-sphere radius, and the OPD differs only by the documented `rayoptics` sign.
  (My first oracle attempt took the FAR ray-sphere root and disagreed by 7 %; the library's
  smallest-`|t|` root is the correct one. Recording this because "the oracle disagreed"
  was, here, the oracle's fault.)

**Note A4-N1 [P2, convention trap -- orchestrator decision needed].** The same `field`
argument names **opposite physical field points** at the two conjugates, silently.
Measured directly on the launch bundle: at `object_distance = inf`, `field = (0, +1)` with
`field_max_rad = 1 deg` gives `M = +0.017452`; at `object_distance = 1e3 m`,
`field = (0, +1)` with the matching `field_max_m` puts the source at `y = +17.455 m` and
gives `M(chief) = -0.017452`. Consequently:

| comparison (biconvex N-BK7, 587.6 nm, 1e3 m vs inf) | 0.5 deg | 1 deg | 2 deg |
|---|---|---|---|
| inf `field=(0,+1)` vs finite `field=(0,-1)` | 0.0038 w (0.09 %) | 0.0028 w (0.06 %) | 0.0034 w (0.05 %) |
| inf `field=(0,+1)` vs finite `field=(0,+1)` | **0.854 w (23 %)** | **1.706 w (40 %)** | **3.404 w (59 %)** |

Neither branch is wrong in isolation: the infinite branch follows the library-wide field
ANGLE convention (`raytrace/ray_fan.py` and `analysis/field.py:420` both use
`M = +sin(theta)`), the finite branch scales an object HEIGHT, and those are opposite. But
one function argument now means two things, and **nothing in the suite would notice a flip**
because PV and RMS are sign-blind. I did **not** change the behaviour (that is an
orchestrator-level convention call, and the infinite branch matches `ray_fan`); I added an
explicit "Sign." paragraph to the `field_max_rad` docstring with the measured comparison, a
why-comment at the launch site, and a two-sided pin,
`test_field_sign_at_infinity_is_the_ray_fan_field_angle`, whose failure message tells
whoever unifies the convention to delete it. **Recommendation:** negate the infinite
branch (`Ld0 = -sin(th_x)`, `Md0 = -sin(th_y)`), because the function's own `field` is
documented as scaling an object position; one line, no released caller (the kwarg is new
in this commit).

**Note A4-N2 [P3].** The off-axis guard is `Nd0 = sqrt(1 - sin^2 x - sin^2 y) <= 0`, and
`sin` is not monotonic past pi/2, so `field_max_rad = 2.0` rad (114.6 deg) yields
`M = sin(2.0) = 0.909`, `N = +0.417` and is **silently launched at 65.4 deg** -- while the
message it never emits says "at or beyond 90 deg from the axis". Gate on the angle
(`abs(th) >= pi/2`) rather than on the direction cosine. Pinned by
`test_field_max_rad_beyond_ninety_degrees_is_not_caught` (which also asserts that a real
90 deg IS refused, so the fix will turn the first half of that test red on purpose).

Input validation is otherwise correct: `object_distance` of `None` -> infinity; `0`, `-1`,
`NaN` -> `ValueError` with the §2 prefix; a non-zero `field` at infinity without
`field_max_rad` -> `ValueError` naming the kwarg.

### A5 [P2] -- GS error scale, plotting extents

`repro/ANALYSIS/p4` §E: reported final error **3.255332e-29** (WP: 3.26e-29) against
`mean(target^2) = 3.896184e+02`, history no longer flat. On my own fixtures at N = 32 /
64 / 128 the err/target-energy ratio is 7.8e-32 / 7.4e-32 / 1.1e-31.

**Backend parity, x64 on.** On a non-degenerate (perturbed) target the NumPy and JAX errors
agree to **1.000001** and **1.000000** at N = 32 and 64 -- the `N_pix` factor is gone --
and the retrieved phases agree to 8.6e-5 and 3.6e-5 rad. On an *exactly* solvable target
both reach their own floor (numpy 4.5e-30, jax 2.3e-12) and the ratio is meaningless;
the jax floor is because `gerchberg_saxton_jax` still returns float32 (below).

`_auto_extent`: drawn pixel centres now coincide with the sample grid to
`<= 3.6e-15 of dx` for N = 8, 9 (odd), 32 and 64, and the `_auto_extent` and `plot_psf`
conventions produce identical ranges (checked at N = 8 and N = 33). The five
`_auto_extent(Ny, dy, unit_label)` sites correctly resolve y in the unit x picked.

**Note A5-N1 [P3, PRE-EXISTING, not WP-A7].** `gerchberg_saxton_jax(..., dtype=complex128)`
raises `TypeError: float() argument must be a string or a real number, not 'complex'`
at `phase_retrieval.py:868` (a line the WP did not touch: `dtype` is applied to the
amplitude arrays, so a complex dtype makes `err` complex). The default path also returns
**float32** even with `jax_enable_x64` on. Both predate this commit; flagged for whoever
owns `phase_retrieval.py`'s JAX side.

### A6 [P2] -- performance, and its numerics

**The recurrence engages.** The WP's tests check that the two paths *agree*; nothing checks
that the fast path is entered, so a uniformity gate that silently rejected `np.linspace`
would leave every test green and the speedup gone. Measured: `np.linspace` gives
`max |z - model| = 0.000e+00` exactly (so `k * that = 0 <= 1e-12`) for 21-, 200- and
wide-range scans, and a `np.exp` call count shows **2 calls for a 21-plane uniform scan vs
21 for a non-uniform one**. Pinned by `test_uniform_scan_really_takes_the_recurrence_path`.

**Accumulated rounding over a long scan.** 200 planes against 200 independent
`angular_spectrum_propagate` calls: worst relative `peak_I` drift **3.9e-13** over
`|kz z|max = 1.06e4` and **4.5e-12** over `|kz z|max = 1.18e4`, against the WP's derived
bound `2 |kz z| eps = 5.2e-12`. `d4sigma` drift 2.1e-13 / 3.9e-13. The bound is correct
and the `HOIST_RTOL = 1e-9` bar in `test_perf_v4_12_0_through_focus.py` -- the one bar the
WP relaxed -- carries its full derivation in the file and sits 2-3 decades above the
measurement and 6 below any real defect. The accompanying new pin (non-uniform z stays
`==` bit-identical) reproduces: I verified bit-identity of `peak_I` against
`angular_spectrum_propagate` on a 5-plane non-uniform scan.

**`single_plane_metrics` bit-identity.** `centroid_x/y`, `d4sigma_x/y` and `peak_I` compare
`==` against `beam_centroid` / `beam_d4sigma` on a focused sphere, a random complex field
and an anamorphic `dy = 2.3 dx` call; the ISO `background=`/`aperture=` route still goes
through the public pair. Guard parity holds for every odd input I could construct -- real
dtype, masked array, F-order, strided view, 3-D ensemble, list-of-lists -- the fast path and
the public pair agree or raise identically. The `_check_2d_scalar_field` census is **69**,
as the WP claims.

**`radial_power_bands` threshold.** Bit-identical (`array_equal`) at n = 1, 50, 95;
5.5e-14 relative at n = 96, 97, 128, 400 -- inside the documented associativity bound.
Order preserved, shape preserved for lists, tuples, 0-d, 2-D and rectangular/anamorphic
grids. Re-measured speedups on my fixture: **4.93x at 256 bands and 19.8x at 1024** (the WP
claims 2.99x / 9.63x -- better, not worse; the small-n rows of my table are dominated by the
function's own meshgrid + `|E|^2` setup, which my hand-rolled baseline hoists out).

**DEFECT A6-D1 [found by me, FIXED]: a NaN radius answered differently on the two sides of
the crossover.** `np.searchsorted` sorts NaN *above* every finite key, so the new sorted
path returned the **whole grid's power** for a NaN radius where the masked loop (and every
release before this commit) returns `0.0`. Measured: `2.513274e-09` (= the total) vs `0.0`.
Silent, and only reachable at `n_radii >= 96`, which is why no existing test saw it. Fixed
in `lumenairy/analysis/polychromatic.py` (`radial_power_bands`, 8 lines + a why-comment):
the sorted path now reproduces the masked loop's answer for NaN, with `+/- inf` left alone
because both paths already return the total there. Verified: NaN -> `0.0` on both sides of
the crossover, `+/- inf` -> the total on both sides, and the rest of the band vector still
matches the masked loop to 1e-11. Pinned by
`test_radial_band_nan_radius_answers_the_same_on_both_paths` and
`test_radial_band_infinite_radius_is_the_total_on_both_paths`.

**Note A6-N1 [P3].** The H-recurrence was not applied to the JAX twin
(`through_focus_scan_jax`). Not a correctness gap -- the JAX kernel re-derives `exp` per
plane, which is the *reference* behaviour, and the NumPy/JAX parity test still passes --
but COMMON.md §9's "fix the twin too" is unmet for the perf half of A6. Worth one line in
the changelog.

**Re-measured scan speedup.** 1.35x (N = 512) and 1.41x (N = 1024) per plane, uniform vs
non-uniform, on my fixture; the WP reports 1.57x on its own. Same direction, different
magnitude -- and, correctly, no test asserts any timing (TESTING_STANDARDS S1).

### A7 [P3] -- the remaining rows

* **Ghost 50 %-EE radius.** On a synthetic uniform-disc ring launch (12 rays/ring):
  0.7500 / 0.6667 / 0.6667 / 0.6956 / 0.7012 R at 4 / 6 / 12 / 24 / 48 rings against the
  analytic 0.70711 -- monotone convergence, always within one ring spacing -- while the
  plain median ray reads **0.5000 R at every ring count**, i.e. it does not converge at
  all. Weights sum to 1.000000.
* **`simulate_interferogram` `dy`.** With tilts designed to land on exact DFT bins
  (Ny = 64, Nx = 96, dx = 1 um, dy = 3 um), the fringe peak lands at bins `(-5, -3)` as
  designed; the pre-fix form (y ramp on `dx`) puts it at `(-2, -3)`, off by `dy/dx = 3`.
  Omitting `dy` reproduces the pre-fix output bit-for-bit, so back-compat holds. Output
  range `[0, 2*background]` confirmed at `background = 0.5` and `0.8`.
* **Exports.** All 7 names resolve on `lumenairy.analysis` and appear in its `__all__`;
  nothing in `__all__` fails to resolve. 4 of them (`unwrap_phase_2d`,
  `clear_meshgrid_cache`, `meshgrid_cache_bytes`, `zernike_basis_cache_bytes`) are still
  absent from the root package -- exactly the outside-ownership request in the WP report
  §5, confirmed rather than forgotten.
* **Zernike cache key.** Two grids differing only in the interior now hash differently and
  build different bases; the old mid-point sample is confirmed degenerate
  (`X.flat[N*N/2] == X.flat[0] == -1.000000`); `dy != dx` still does not alias.
* **`caustic_diagnostic`.** No warning on an axisymmetric singlet (`maslov_index = 0`), so
  the new warning is not a nuisance on the common path.
* **`strehl_phase_integral`** docstring warning present; `p3` §E still reproduces
  0.99996 vs 0.00069 on a 1-wave-rms tilt.

---

## 3. What I changed

Both inside WP-A7's ownership (`lumenairy/analysis/*`), plus two new test files.

1. `lumenairy/analysis/polychromatic.py` -- `radial_power_bands`: NaN-radius guard on the
   sorted path (defect A6-D1). +12/-1 lines including the why-comment.
2. `lumenairy/analysis/image_plane_wfe.py` -- documentation only, no behaviour change: a
   "**Sign.**" paragraph under `field_max_rad` carrying the measured cross-conjugate
   comparison, and a 6-line why-comment at the collimated-launch site. +22 lines.
3. `tests/unit/test_audit2609_verify_a7.py` -- **39 tests** (A1 grids / float32 / layout /
   annulus / decentred piston / vortices; A2 both signs of f and four support shapes;
   A3 Southwell on annular masks, the itoh-vs-southwell hole difference, disconnected
   gauging; A6 recurrence-engagement, 200-plane drift, crossover seamlessness, NaN and inf
   radii).
4. `tests/unit/test_audit2609_verify_a7_wfe.py` -- **11 tests** (A4: the independent ray
   trace on-axis and at 0.5 / 1 / 2 deg -- raw agreement and, separately, the shape with
   the reference-sphere defocus projected out -- the thick-lens BFL, the off-axis chief,
   the field sign pin, the >90 deg guard gap).

Every numeric bar carries its derivation, its measured value and the decades of gap on
both sides. Every bar was demonstrated to FAIL on the pre-fix behaviour, restored
in-process, with the numbers recorded in §2 and in the test docstrings
(scratch script: `scratchpad/v10_failbefore.py`). I touched no file outside
`lumenairy/analysis/` and made no git write of any kind.

---

## 4. The audit's "checked and found correct" list still holds

Re-ran `repro/ANALYSIS/p1, p2b, p2c, p3, p4, p5, p6, p11, p12` and
`repro/orch/verify_analysis.py`. Unchanged: Zernike OSA indexing, normalisation and the
21-mode round-trip (`max |Dc| = 1.059e-21`); the row/column axis convention; `dy != dx`
cache non-aliasing; `astigmatism_mag_angle` (0 / 45 / 22.5 deg); `mtf[0,0] = 8.596e-18`,
`max |MTF - pupil autocorr| = 6.661e-16`; Rayleigh / FWHM / Sparrow within 0.14 %; the
three Strehl definitions at lambda/14 (0.81451 / 0.81802 / 0.81669 / 0.81592);
`M2 = 1.000000` / `3.000000`; `beam_d4sigma = 2 w0` (80.0000 um); EE curve to 9.3e-5;
`apply_detector` flux 1.000003-1.000013; Shack-Hartmann slope gain 0.945-0.949 and the
bit-exact zero on a flat wavefront; `wave_opd_1d` exact (library OPL == a hand-written
geometric OPL to 4 decimal places in waves at every ray); the OPD sign convention;
`error_reduction` / `hybrid_input_output` (history 3.7117e+02 -> 1.7001e+00).

---

## 5. Tests run

| command | result | duration |
|---|---|---|
| `pytest tests/unit/test_audit2609_a7_{opd_unwrap,strehl_reference,detector_sh,image_plane_wfe,misc}.py test_analysis test_audit_analysis test_niche_s12_shack_hartmann_reference test_v4_16_1_agent_a test_perf_v4_12_0_through_focus` | **223 passed**, 13 warnings | 96 s |
| `pytest` over 18 analysis-adjacent files (`test_audit_w5_analysis`, `test_audit_w6_analysis_elements`, `test_audit_through_focus_jax_{s3_8,x64}`, `test_niche_audit_{a1_radial_metrics,a2_encircled_energy_radius,w4c_analysis_immersed,w4_input_kind}`, `test_ao_dm`, `test_perf_v4_12_0_zernike_cache`, `test_plot_lens_layout_ray_overlay`, `test_through_focus_{bucket_boundary,metric_parity}`, `test_v5_4_{make_shack_hartmann_wfs,retrace_ghost_path,zernike_normalization_weighting}`, `test_v5_4_6_wave6_analysis`, `test_v5_2_3_subaperture_image_plane`) | **442 passed, 3 skipped** | 66 s |
| `pytest tests/unit/test_audit2609_verify_a7.py tests/unit/test_audit2609_verify_a7_wfe.py` | **50 passed** | 6 s |
| `python validation/run_all.py test_analysis test_detector test_image_plane_wfe` | **ALL 3 files passed** (82.8 / 9.0 / 6.2 s) | 98 s |
| final confirming run AFTER my two source edits: the 5 WP files + my 2 + `test_analysis`, `test_audit_analysis`, `test_niche_s12_shack_hartmann_reference`, `test_v4_16_1_agent_a`, `test_perf_v4_12_0_through_focus`, `test_niche_audit_a1_radial_metrics`, `test_v5_4_6_wave6_analysis`, then `validation/run_all.py test_analysis test_detector test_image_plane_wfe` | **299 passed** + **ALL 3 validation files passed** | 39 s + 21 s |
| `repro/orch/verify_analysis.py`, `repro/ANALYSIS/{p1,p2b,p2c,p3,p4,p5,p6,p11,p12}` | all re-run, numbers in §2 | -- |

The 3 skips are the documented `jax_enable_x64`-is-process-wide skip in
`test_audit_through_focus_jax_x64.py`, pre-existing.

**Pre-existing failures found (not WP-A7).**
`tests/unit/test_niche_audit_w3_infra.py::TestA6EstimateAsmMemory::test_est_bounds_measured_first_call_peak`
fails on **two** arms (`[512-complex128]` and `[1024-complex128]`), not the one the WP
reported; the failure is inside `lumenairy/propagators/asm.py`, modified in this working
tree by another WP, and `analysis/` is not on the path. The second failure the WP reported
(`test_audit_misc.py::...StopIndexWarn::test_traced_emits_warning_for_stop_index_2`) is
**no longer present** -- the lens WP has since updated that test in the working tree
(`test_out_of_range_stop_index_now_raises` now passes), so that item can be struck from the
WP's list.

---

## 6. Open items for the orchestrator

| id | sev | item |
|---|---|---|
| **A4-N1** | **P2** | `field` names opposite field points at the finite and infinite conjugates of `eval_image_plane_wfe` (measured: inf `(0,+1)` == finite `(0,-1)` to 0.06 % of span; same sign differs by up to 59 %). Documented and pinned by me; the one-line behaviour fix (`Md0 = -sin(...)`) is a convention decision I deliberately did not take. |
| **A4-N2** | P3 | `field_max_rad > pi/2` is silently folded back by `sin` instead of being refused; gate on the angle. Pinned. |
| **A3-N1** | P3 | `_reconstruct_wavefront`'s NaN-exclusion sentence is false for `reconstruction='itoh'` (measured 0.49 of span on an annulus). Pinned; move or qualify the sentence. |
| **A1-N1** | P3 | The `wave_opd_2d` piston-anchor claim ("right integer wave count") holds only for a centred pupil; measured +1 and +2 whole waves on decentred pupils. Pinned; one clause in the Notes would close it. |
| **A5-N1** | P3 | PRE-EXISTING, `phase_retrieval.py:868`: `gerchberg_saxton_jax(dtype=complex128)` raises `TypeError`, and the default JAX path returns float32 even with x64 on. Owner: whoever owns the JAX phase-retrieval twin. |
| **A6-N1** | P3 | The H-recurrence was not applied to `through_focus_scan_jax`; perf only, no correctness gap. Changelog note. |
| **doc** | trivial | WP-A7's changelog says `test_audit2609_a7_detector_sh.py (11)`; the file collects **12**. The report's §1 (12) is right. Also `repro/ANALYSIS/p2c`'s "fitted RMS-WFE (true ...)" column compares a whole-map RMS against a single coefficient and reads as unfixed when it is not -- worth a line in the repro script if anyone re-runs it. |
| **prev** | P2 | `test_niche_audit_w3_infra.py::TestA6EstimateAsmMemory` fails on 2 arms, inside `propagators/asm.py` (another WP). |

Nothing in WP-A7 is NOT FIXED, and nothing regressed.

*(§7 below records how each of these was resolved after the coordinator's
rulings of 2026-09-12; §6 is left as written so the ruling has its
measurement behind it.)*

---

## 7. Open-item resolution (coordinator rulings, 2026-09-12)

All six rulings implemented in `lumenairy/analysis/*` and the analysis test
files, re-measured, and re-run (§7.7).

### 7.1 A4-N1 [P2] -- `field` now means one object point at both conjugates

**Ruling: negate the infinite-conjugate launch.** Implemented in
`lumenairy/analysis/image_plane_wfe.py`: `Ld0 = -sin(th_x)`,
`Md0 = -sin(th_y)`. A positive `field` is an object ABOVE the axis at either
conjugate, so its chief ray travels towards `-y` at either conjugate.

Re-measured on the biconvex N-BK7 singlet (587.6 nm, 61-ray pupil), same
script as §2:

| comparison (1e3 m vs inf) | 0.5 deg | 1 deg | 2 deg |
|---|---|---|---|
| inf `(0,+1)` vs finite `(0,+1)` -- **must now agree** | **0.0025 w (0.06 %)** | **0.0033 w (0.07 %)** | **0.0033 w (0.05 %)** |
| inf `(0,+1)` vs finite `(0,-1)` -- must now disagree | 0.854 w (23 %) | 1.706 w (40 %) | 3.404 w (59 %) |

Read off the launch bundle directly: `field = (0, +1)` with
`field_max_rad = 1 deg` now gives `M = -0.017452` at infinity, the same sign
and the same value as the finite launch's chief (`M = -0.017452`, source at
`y = +17.4551` m).

The "Sign." paragraph under `field_max_rad` now states the object-position
sense, the measured cross-conjugate agreement, and the relation to the
ray-direction convention: **`raytrace.ray_fan`'s `field_angle` is this
`field`'s negative**, `field_angle = -Hy * field_max_rad`. A why-comment at
the launch site says the same in three lines.

The pin is flipped and renamed:
`test_field_means_the_same_object_point_at_both_conjugates` asserts
`inf (0,+1) == finite (0,+1)` within 1 % of span (measured <= 0.07 %, one
decade of margin) AND that the reversed sign still disagrees by > 10 % of span
(measured 23-59 %), so the two arms are 2.5 decades apart and neither passes
by accident. `test_chief_at_infinity_travels_towards_minus_y_for_a_positive_field`
pins the mechanism on the launch bundle itself (`M == -sin(theta)` to 1e-12
relative). The independent ray-trace oracle in
`test_infinite_conjugate_opd_matches_an_independent_ray_trace` now takes
`-th`, since it is parametrised by the ray direction, and still reproduces the
library to 1.9e-11 waves on-axis.

### 7.2 A4-N2 [P3] -- the 90 deg guard is now on the angle

`if abs(th_x) >= 0.5*pi or abs(th_y) >= 0.5*pi: raise ValueError(...)`, before
the sines are taken, with the §2 `eval_image_plane_wfe: ` prefix. The
direction-cosine check is kept as a defensive second gate with its own
message. `test_field_max_rad_beyond_ninety_degrees_is_refused` now asserts the
refusal at 2.0 rad (114.6 deg -- the case that used to be launched silently at
65.4 deg), at exactly `pi/2`, and at 3.0 rad, and that an ordinary 2 deg field
is still accepted. All four arms are exact decisions, not tolerances.

### 7.3 A3-N1 [P3] -- the NaN-exclusion sentence is now attached to Southwell

`lumenairy/analysis/detector.py`: the sentence moved up into the Southwell
paragraph and now reads "excluded from the SOUTHWELL solve"; a new paragraph
states explicitly that `'itoh'` does NOT get it, why (a path integral has one
route to each node and cannot route around a hole), and what it costs
(measured 0.49 of the wavefront span on an annular mask against 8.7e-15 for
Southwell). The same caveat is now on `_itoh_wavefront` and under the public
`shack_hartmann(reconstruction=)` parameter, which is where a caller choosing
between the two actually looks. `test_the_itoh_caveat_is_documented` pins all
three sites structurally, and the behavioural
`test_itoh_integrates_through_holes_where_southwell_does_not` is unchanged.

### 7.4 A1-N1 [P3] -- the piston anchor names its condition

`lumenairy/analysis/opd.py`, `wave_opd_2d` Notes: the claim is now "a CENTRED
pupil carrying a known defocus comes back with the right integer wave count",
followed by what happens otherwise -- the anchor lands on the rim, where the
wavefront is not stationary, and the piston is an arbitrary but EXACT whole
number of waves (the measured +1.0000 and +2.0000 are quoted) -- and by what is
therefore unaffected (shape, PV, RMS, every Zernike coefficient above piston).
Pinned by `test_the_piston_anchor_note_names_the_centred_pupil_condition`
alongside the behavioural
`test_offcentre_pupil_shape_is_exact_and_the_piston_gauge_is_documented`.

### 7.5 A5-N1 [P3] -- `gerchberg_saxton_jax` dtype, both halves fixed

Not structural; both halves are fixed in
`lumenairy/analysis/phase_retrieval.py`.

* **Complex `dtype`.** A complex request now names the iteration's FIELD type
  and is mapped to its real counterpart for the amplitude and phase arrays, so
  `err` stays a real float and `float(err)` is well defined. Measured
  (x64 off, 10 iterations, N = 32, perturbed target): `np.complex128` ->
  phase dtype float64, `err = 1.502759e-01` == the NumPy twin's
  `1.502759e-01`; `np.complex64` -> float32, `err = 1.502756e-01`. Pre-fix
  both raised `TypeError: float() argument must be a string or a real number,
  not 'complex'`.
* **x64.** `dtype=None` now resolves to float64 when
  `jax.config.jax_enable_x64` is set and float32 otherwise, the same rule
  `jax.numpy` applies to its own default float type. Measured with x64 on:
  the JAX `err` is `1.5027588862e-01` against the NumPy `1.5027588862e-01` --
  agreement to every digit float64 carries, where the float32 default read
  `1.5027558804e-01` (2.0e-05 relative). On an exactly-solvable target the JAX
  floor drops from **2.3e-12 to 5.2e-29**. An explicit `dtype=np.float32`
  still pins the historical single-precision path regardless of the flag --
  which is the migration note, and which
  `test_gs_jax_default_precision_follows_x64` asserts as its second arm so the
  two precisions stay distinguishable.

Both pinned: `test_gs_jax_accepts_a_complex_dtype_and_returns_a_real_error`
(parametrised over complex64/complex128; fails with a `TypeError`, not a
tolerance, on the pre-fix code) and
`test_gs_jax_default_precision_follows_x64` (bar 1e-9 relative between the
backends: 4 decades above the `N_pix * eps = 2e-13` FFT-rounding floor and 4
below the float32 reading, so it decides between the two precisions rather
than tolerating either).

### 7.6 A6-N1 and the test-count typo -- changelog lines

* `WP-A7_CHANGELOG.md`, A6 performance section: the JAX twin
  `through_focus_scan_jax` deliberately keeps its per-plane `exp(1j kz z)` --
  it is the reference evaluation the NumPy path is checked against
  (`TestThroughFocusScanMatchesJAXTwin`), and re-deriving it per plane is what
  makes that check independent. The recurrence is a NumPy-path optimisation
  only.
* `tests/unit/test_audit2609_a7_detector_sh.py` reads **(12)**, not (11), in
  the A3 section (confirmed by `pytest --collect-only`: 18 / 11 / 12 / 13 / 24
  across the five new files, 78 total).
* The A6 `radial_power_bands` section gained the NaN-radius paragraph (defect
  A6-D1, §2), the A4 section the field-sign and 90 deg paragraphs, the A3
  section the `'itoh'` caveat, the A1 piston section the centred-pupil clause,
  and a new "Fixed -- analysis/phase_retrieval: `gerchberg_saxton_jax`'s
  `dtype` handling (A5 follow-up)" entry carries §7.5 with its migration note.

### 7.7 Re-run after the rulings

| command | result | duration |
|---|---|---|
| `pytest tests/unit/test_audit2609_verify_a7.py` | **45 passed** (39 + 1 new A2 arm + 5 resolution pins) | 8 s |
| `pytest tests/unit/test_audit2609_verify_a7_wfe.py` | **11 passed** | 1 s |
| `pytest` over the 5 WP A7 files + both VERIFY files + `test_analysis`, `test_audit_analysis`, `test_audit_misc`, `test_niche_s12_shack_hartmann_reference`, `test_v4_16_1_agent_a`, `test_perf_v4_12_0_through_focus`, `test_v5_2_3_subaperture_image_plane`, `test_niche_audit_a1_radial_metrics`, `test_v5_4_6_wave6_analysis`, `test_audit_through_focus_jax_x64` | **538 passed, 6 skipped** (3 cupy-absent, 3 the pre-existing x64 skip) | 230 s |
| `python validation/run_all.py test_analysis test_detector test_image_plane_wfe` | **ALL 3 files passed** (22.1 / 2.2 / 1.9 s) | 26 s |

### 7.8 Source files I now hold changes in

* `lumenairy/analysis/image_plane_wfe.py` -- field sign (behaviour), 90 deg
  angle gate (behaviour), "Sign." docstring paragraph, launch why-comment.
* `lumenairy/analysis/detector.py` -- docstrings only (three sites), no
  behaviour change.
* `lumenairy/analysis/opd.py` -- docstring only, no behaviour change.
* `lumenairy/analysis/phase_retrieval.py` -- `gerchberg_saxton_jax` dtype
  resolution (behaviour) + its docstring.
* `lumenairy/analysis/polychromatic.py` -- `radial_power_bands` NaN guard
  (behaviour, defect A6-D1 from §2).
* `tests/unit/test_audit2609_verify_a7.py` (45),
  `tests/unit/test_audit2609_verify_a7_wfe.py` (11).
* `docs/audits/.../fixes/WP-A7_CHANGELOG.md` -- the six edits listed in §7.6.

Two behaviour changes carry migration notes in the changelog (the
infinite-conjugate field sign, which has no released caller because the kwarg
is new in this commit; and the JAX GS default precision under x64, where
`dtype=np.float32` pins the old behaviour). No file outside
`lumenairy/analysis/`, the analysis test files, and the two audit documents
was touched, and no git write of any kind was made.
