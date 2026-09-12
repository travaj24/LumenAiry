# WP-A3 — `apply_real_lens_traced` and its siblings

Branch `audit-fixes-2026-09`.  Findings **S1** (P0), **T1–T16**, **S8–S11**,
and the new wave hand-off (§15.9, first bullet).  Every number below is
MEASURED on this box — by re-running the audit's own repro scripts before and
after, or by a probe written for the purpose — except where explicitly labelled
"pre-fix", which are the same scripts re-run on HEAD before any change.

Files owned and touched: `lumenairy/elements/_lens_traced.py`,
`_lens_traced_multibranch.py`, `_lens_traced_uniform.py`, `_lens_imap.py`,
`lumenairy/_math/chebyshev.py`, plus traced-lens test files and two new
`tests/unit/test_audit2609_a3_*.py`.  Nothing outside that set was modified.
`_traced_flags.py` needed no change (TR-SIBLINGS found no defect in it and I
did not either; the S8 cross-check it should have had is now in the imap key).

---

## 1. Summary table

| ID | status | files:lines | tests (path::name) | oracle | measured before → after |
|---|---|---|---|---|---|
| **S1** (P0) | **fixed** | `_lens_traced_multibranch.py:_trace_launch_grid`; `_lens_traced_uniform.py:_trace_meridional_fold`, `_trace_meridional_cusp` | `test_audit2609_a3_caustic_siblings.py::test_s1_launch_grid_lands_on_a_plane_not_on_the_sag` (4), `::test_s1_fail_before_*`, `::test_s1_multibranch_field_agrees_*` | independent vector-Snell trace + hand-written vertex transfer (`repro/TR-SIBLINGS/repro_vertex.py`, `oracle.py`) | R2=−25 mm/ap 20 mm: **476.8 µm / 3 501 waves → 1.7e-12 µm / 4.4e-12 waves**; field vs single-valued at the same plane **1.7984 → 0.0003 rad rms** (max π → 0.0049) |
| **T1** (P1) | **fixed** | `_lens_traced.py` `target_cdtype` (1 site, 4 consumers) + `_reference_input` | `test_audit2609_a3_traced_lens.py::test_t1_*` (4) | `apply_real_lens` on the same array | `AttributeError` → **complex128**, bit-comparable to the complex-input call (≤1e-12 rel) |
| **T2** (P1) | **fixed** | `_lens_traced_multibranch.py` census + two-sided tripwire, normalised by the launched power that reaches the grid | `::test_t2_multibranch_refuses_*`, `::test_t2_energy_tripwire_is_two_sided_*`, `test_v5_21_delta_audit.py::test_d3_air_focus_multibranch_runs_without_warning` | reconstructed grid power vs `Σ\\|E_launch\\|²h²` over launch nodes landing on the grid | at BFL **P/P_in = 0, 0 warnings → RuntimeError**; pre-focus band **4.78× silent → warned**; air-focus false positive **2.32× (and 12.3× on the shipped 10× bar) → 0.99×, silent** |
| **T3** (P1) | **fixed** | `_lens_traced.py` `_fill_dead_launch_nodes`, `_landed_on_filled_node`, the NaN-fill site, `_warn_newton_unconverged` | `::test_t3_*` (3) | the polynomial fit on the same rays | spline **P/P_in 0.0000, 0 non-zero px → 0.9501, 17 493 px** (polynomial 0.9754 / 21 601) |
| **T4** (P1) | **fixed** | `_lens_traced.py::_geometric_lens_phase` | `::test_t4_*` (4) | `angle(apply_real_lens(ones))` | `AttributeError` → runs; residual **0.00034 rad max at 1 µm of glass**; NaN leak → 0 + warning; float32 piston ulp 1.95e-3 rad → **< 1e-4 rad** |
| **T5** (P1) | **fixed** | `_lens_traced.py::_form_error_phase_screen` + both assembly paths + the caustic refusal | `::test_t5_*` (3) | the exact screen `−k0(n−1)·form_error` | applied-vs-exact **0.564 rad rms (100 % missing) → 0.0034 rad rms (0.6 %)**; analytic sibling 0.0145 |
| **T6** (P1) | **fixed** | `_lens_traced.py::_reverse_prescription`, `_negate_sag_callable`, `_negate_freeform` | `::test_t6_*` (5) | forward vs backward OPL (Fermat) + the forward sag negated | aspheric sag **+1.25e-06 m (2.13 waves) → 0.0e+00**; n-thickness OPL **Δ = 1.44e-01 m → 5.2e-18 m**, ray lands at +6.576 mm → +4.000 mm |
| **T7** (P2) | **fixed** | `_lens_traced.py::_sample_local_tilts`; `_compute_carrier` ndarray branch | `::test_t7_*` (2), `::test_t12_*` | the exact tilt of a plane / spherical wave | wrap **1.475e-01 → 1.232e-15**; half-pixel bias **+2.0000e-05 → +4.5e-21**; ndarray index mean offset **−0.4936 px → −0.0000** |
| **T8** (P2) | **fixed** | `_lens_traced.py` `_multi` signature-derived refusal; `_segmented` square-pixel + carriers | `::test_t8_*` (3) | `inspect.signature` of both entry points | opaque `TypeError` for 12 kwargs → **named `ValueError` with the remedy**; `segmented(dy=2dx)` returned a field → raises |
| **T9** (P2) | **fixed** | `_lens_traced.py`: `_invert_fit` chunk, whole-grid masking, `_ray_density_self_checks`, `_segment_field_by_angle` 1-D windows, `_Cheb2DEvaluator` numpy chunk | `::test_t9_*` (2) | tracemalloc peak; bitwise equality | NumPy Chebyshev fallback **1160 MB (145 float64/pt) → 120 MB (15.0/pt)**, bitwise identical; fit design 141 grid-units at N=1024 → chunked; masking ~7.1 → ~3.1 units |
| **T10** (P2) | **fixed** | `_lens_traced.py` numba kernel blocked; `_script_has_main_guard` / `_is_main_guard_test`; C13 domain probe | `::test_t10_*`, `test_fix_newton_pool_memory.py::test_the_guard_detector_*` (13 rows) | the pure-xp branch of the same evaluator; the audit's `guard/g*.py` corpus | kernel **346.4 → 213.4 ms (1.62×) / 236.4 → 152.7 ms (1.55×)**, `max\\|diff\\| = 0.0`; guard: 3 of 12 fixtures reclassified correctly |
| **T11** (P3) | **fixed** | `_lens_traced.py` + `_lens_imap.py` Chebyshev wrappers; 4 doc claims; `_opl_by_backward_trace` index | `::test_t11_*` (2) | `numpy.polynomial.chebyshev` + central FD | three copies → one, `max\\|Δ\\| = 0.0` at orders 0/1/6/8/12 |
| **T12** (P2) | **fixed** | `_lens_traced.py::_compute_carrier` ndarray branch | `::test_t12_*`, `test_niche_d1_tilted_carrier.py::test_tilted_carrier_beats_the_equivalent_ndarray_wavefront` (restated) | `carrier=<float>` (the same wavefront in closed form) | **9.914e-02 rad rms (20.67 nm) → 1.874e-03 rad (0.391 nm)**; float carrier 1.845e-03 |
| **T13** (P2) | **fixed** | `_lens_traced.py` `_sig` gate | `::test_t13_*` | the marginal ray at the aperture edge, traced independently | `na_exit` **0.78487 (3.14× over) → within 0.3 % of 0.24964** |
| **T14** (P2) | **fixed (doc + notice)** | `_lens_traced.py` `ray_subsample` doc; `_imap_domain_gate` notice | — (the notice is asserted by the T14 arm of `::test_t9_*` indirectly; see §2) | the audit's own f/7.5 measurement | silent 11.63 nm rms fallback → **named `RuntimeWarning` with the numbers** |
| **T15** (P2) | **partially fixed** | `_lens_traced.py` `parallel_amp` gate; strided `\\|E_analytic\\|` | — (both are perf; bit-identity covered by `::test_t9_final_masking_*` and the whole-file runs) | tracemalloc + wall medians | gate **48 GB flat → max(2 GB, 6·nbytes)**, so the measured 1.35× is reachable; full-grid abs 14.17 ms/8.4 MB → 0.114 ms strided.  `phase_analytic_lens` caching and the `_pip_residual_ri` band are DEFERRED (§6) |
| **T16** (P3) | **fixed** | `_lens_traced.py` delegate list + always-warn; `return_screen`+delegate refusal; `float(sum(thicknesses))`; the aperture-check swallow; two doc corrections | `::test_t8_*` (partly) | — | delegate with only defaults **0 warnings → always announced**; `return_screen`+delegate returned `E_in` → raises |
| **S8** (P2) | **fixed** | `_lens_imap.py` `_IMAP_KEY_TRACED_FLAGS` + `report_refusal` + GRAM provenance | `::test_s8_*` (2) | the key itself (`repro/TR-SIBLINGS/imap_cache.py`) | `key(det=True) == key(det=False)` **True → False**; same object served **True → False** |
| **S9** (P2) | **fixed** | `_lens_traced_multibranch.py` `_raster_batches` + vectorised Ludwig swap; `_lens_traced_uniform.py` dark-fill annulus | `::test_s9_*`; bit-behaviour by the k1/k4 suites | tracemalloc / bucket census; Airy decay | exact bbox + 4e6-entry budget (was unbounded: 7.74 GB traced / 12.2 GB RSS at N=4096); Ludwig swap 0.76–1.07 ms/pixel → one vectorised pass; dark fill 100.0 % of pixels → the 20·l_airy annulus (the 25.4 s → ~1.5 s headline is an **N = 2048** number; VERIFY-A3 re-measured only 1.1–1.2× at N = 384 / 512, where the dark fill is a small share of the call, and the FIELD is unmoved to 9.7e-27 of peak) |
| **S10** | **not mine** | `fga.py`, `lenses_maslov.py` | — | — | see §5 |
| **S11** (P3) | **fixed (4 of 8 items)** | `_lens_traced_multibranch.py` (KMAH roll, `L0`/`M0`, tripwire); `_lens_traced_uniform.py` (dead counter, dead `wavelength`); `_math/chebyshev.py` (round-trip claim) | `::test_s11_*` (3 + 4 rows) | closed-form edge-clamped shift; a flat-slope map | roll → edge-clamped; `n_turn` double-count on a flat sample 2 → 1; the other four items are in `_lens_jax` / `lenses_maslov` / `fga` — §5 |
| **WP-A2 §5.3** | **done** | `_lens_traced.py` stop-index read | `test_audit2609_a3_traced_lens.py::test_traced_stop_index_is_read_through_the_shared_normaliser` | `_lens_real._normalise_stop_index` (shared with `apply_real_lens`) | out-of-range / non-integer: `apply_real_lens:` from a worker thread, or `invalid literal for int()` → **`apply_real_lens_traced:` ValueError up front**; every in-range spelling bit-identical |
| **WP-A2 §5.5** | **done** | `_lens_traced.py` aperture-vs-grid notice | `::test_traced_aperture_notice_measures_the_narrow_axis` | the smaller of the two semi-extents | tall grid (Nx 64 / Ny 256, 2 mm aperture): **0 warnings → 1**, naming `N=64` |
| **§15.9 wave hand-off** | **added (new API)** | `_lens_traced.py` `caustic='wave'` | `::test_wave_*` (4) | `apply_real_lens` + band-limited ASM (a different phase model) | at d=0 **bit-identical** to the traced call; at the focus peak 8.5781 vs oracle 8.5781, EE(5/10/25 µm) 0.0408/0.1609/0.6348 vs 0.0408/0.1609/0.6348, where multibranch returns **nothing** |

Verified-correct items re-measured and unchanged: masked / non-converged pixels
exactly zero with no NaN or inf anywhere; `amplitude_model='screen'` energy
1.000000 against the aperture-transmitted input; `caustic='single'` bit-identical
to the default; `PreparedTracedLens` factorisation; the v5.44 banded assembly
byte-identical across its 7 configurations; the spectral partition of unity and
the segmented reconstruction; `_compute_carrier('auto')` on spheres; the
deterministic solve's thread-count byte-identity; the Newton inversion's
1e-15 m accuracy on a synthetic map; `_lens_imap`'s degree-14 model beating the
spline-Newton incumbent by 5.1e+03× on OPL (re-measured 3.60e-12 vs 1.84e-08
waves).  All of those are covered by the existing suites listed in §4, which
pass.

---

## 2. Per finding

### S1 (P0) — the caustic siblings evaluated on the last surface's SAG

**Wrong.**  `_trace_launch_grid` took `rt.trace(...).image_rays` — which sit at
`z = sag(rho)` of the last surface — and advanced them by
`t = output_plane_distance / N_z`.  That evaluates every ray at
`z = sag(rho) + d`: a RAY-DEPENDENT longitudinal position, while the public
docstring promises "the output plane `output_plane_distance` past the
prescription's exit vertex".  Worse, two branches arriving at nominally the same
`(x, y)` come from different `rho`, hence different `sag`, hence different `z`,
so the multi-branch interference was computed between fields at different
longitudinal positions.  `_lens_traced_uniform`'s two meridional traces repeated
the same three lines, so the fold radius `r_c`, `kappa` and the mean-eikonal fit
were resolved on the same wrong surface.

**Changed.**  All three sites call `TraceResult.at_exit_vertex()` (WP-A1's
shared operator, which resolves `n_exit` from the prescription and KILLS a
grazing ray rather than teleporting it), and then add the free-space leg
`t = d / N` in `output_plane_n` — so the leg really is `(d − z)/N` with `z = 0`.
`_kmah_free_leg`'s `Q0 = d(x_exit)/d(gamma)` is now taken at the vertex plane
too, which is the reference the `det Q(z) = det Q0 + z·trK + z²·detQd`
parametrisation assumes.

**Verified** against the auditor's own independent vector-Snell oracle
(`repro/TR-SIBLINGS/repro_vertex.py` + `oracle.py`), re-run before and after:

| fixture | before \|Δx\| | after \|Δx\| | before OPD | after OPD |
|---|---|---|---|---|
| R2 = ∞ (flat rear, CONTROL) | 4.337e-13 µm | 4.337e-13 µm | 2.95e-12 w | 2.95e-12 w |
| R1 = ∞, R2 = −25 mm, ap 20 mm, d = 0 | **476.8 µm** | **1.735e-12 µm** | **3 501 w** | **4.43e-12 w** |
| same, d = 40 mm | 476.8 µm | 1.041e-11 µm | 3 501 w | 3.54e-11 w |
| R1 = ∞, R2 = −100 mm, d = 0 | 24.59 µm | 1.735e-12 µm | 820.3 w | 1.18e-11 w |
| biconvex ±50 mm, ap 12 mm, d = 0 | 40.57 µm | 1.735e-12 µm | 568 w | 1.18e-11 w |

End to end on the public API (`repro/TR-SIBLINGS/repro_vertex_field.py`),
multibranch against the single-valued traced path at the SAME plane:

| fixture | before | after |
|---|---|---|
| flat rear (CONTROL) | 0.0050 rad rms, max 0.0207 | 0.0050 rad rms, max 0.0207 |
| curved rear R2 = −100 mm | **1.7984 rad rms (0.286 w), max π** | **0.0003 rad rms, max 0.0049** |
| biconvex ±60 mm | **1.8100 rad rms (0.288 w), max π** | **0.0045 rad rms, max 0.0197** |

1.80 rad rms with max = π is a fully decorrelated wrapped phase; after the fix
the curved-rear cases are BETTER than the flat-rear control, which is unmoved.

**Residual risk.**  `at_exit_vertex` kills a grazing ray (`|N| <= 1e-30`) where
the hand-written copies teleported it.  On these fixtures no ray is grazing; a
prescription that produced one would see that ray marked dead and dropped from
the triangulation, which is the intended semantics but is a behaviour change at
the boundary.

### T1 (P1) — a real-dtype `E_in`

**Wrong.**  `target_cdtype = E_in.dtype if iscomplexobj else np.complex128`
yields the numpy scalar TYPE for a real input, and `np.complex128.type` does not
exist — so the four `target_cdtype.type(0)` masking sites raised, two of them on
the banded path (the shipped default at N ≥ 4096).  The `!=` comparisons and
`.astype` calls happen to work with a bare type, which is why it survived.  The
validator accepts a real field and the whole trace, three fits and Newton
inversion ran before the failure.  `apply_real_lens` and `PreparedTracedLens`
accept the same array.

**Changed.**  `np.dtype(...)`, resolved once at the top of the function as
`target_cdtype_in` and reused by the multibranch dispatch, the carrier reference
and both assemblies.  The related `_reference_input()` cast is fixed too: for a
real `E_in.dtype` it silently discarded the imaginary part of `exp(i k0 W)`
behind a `ComplexWarning`, turning a unit-modulus phasor into `cos(k0 W)`.

**Verified.**  `repro/orch/verify_trmain2.py` item 1: `AttributeError` →
`complex128`.  The regression test additionally requires the real-input answer
to match the complex-input answer to 1e-12 relative, and exercises the banded
path explicitly with `sag_chunk_rows=32` rather than waiting for N ≥ 4096.

### T2 (P1) — the silent zero field at an axial focus

**Wrong.**  At and around the paraxial focus every mapped triangle either
collapses below `caustic_min_area_ratio` or compresses below one pixel, so the
rasteriser covers nothing.  The caller got an identically-zero field with **no
warning**: measured `P/P_in = 0.0000e+00`, 0 non-zero pixels, 0 warnings at the
25.478 mm BFL of the audit's own fixture.  The D5 tripwire fired only on a power
GAIN above 10×, so it saw neither that nor the 4–8× band before it.

**Changed.**  The rasteriser counts finite and degenerate triangles.  A
reconstructed power of exactly zero against a non-zero input aperture power now
raises `RuntimeError` naming the axial point-focus catastrophe and the remedies
(`caustic='wave'`, GBD, Maslov) and saying that lowering `min_area_ratio` does
not fix it (measured `P/P_in = 3.9e+09` at 1e-12).  A skip fraction above 25 %
warns.  The energy tripwire is two-sided: `_ENERGY_BLOWUP_FACTOR` 10.0 → 2.0 and
a new `_ENERGY_COLLAPSE_FACTOR = 0.5`.  `return_diagnostics` reports the census.

**Verified.**  `repro/orch/verify_trmain2.py` item 3 now raises with the named
message.  A scan toward the focus on an f≈25 mm singlet (N = 512, dx = 4 µm,
`caustic_ray_subsample=4`) measures, after the fix: 24.000 mm → 1.000 (0
warnings), 24.400 mm → 9.904 (1 warning), 24.550 mm → 16.98 (1 warning),
24.4833 mm → refusal.  Pre-fix the first three were 1.000 / 9.904 / 16.98 with
**0 / 1 / 1** warnings at the OLD threshold — i.e. the 4.78× point the
regression test finds by scanning was silent and now warns.

**Residual risk.**  The refusal is a hard `RuntimeError`, so a caller that
previously received zeros and carried on will now stop.  That is the intent
(zeros are not a physical answer at that plane) but it is a behaviour change on
an opt-in mode.  `apply_real_lens_traced_uniform` inherits it, since it runs the
multibranch first.

#### T2 follow-up — the tripwire's NORMALISER was wrong (found by VERIFY-A14)

VERIFY-A14 reported `test_v5_21_delta_audit.py::
test_d3_air_focus_multibranch_runs_without_warning` failing on my tree with

```
RuntimeWarning: apply_real_lens_traced_multibranch: reconstructed grid power is
2.32x the input aperture power (up to 1 branches coalesce on one pixel).
```

That is a **false positive**, and the give-away is in the message itself: *1*
branch per pixel.  There is no coalescence in that fixture at all, so there is
nothing for the point-focus catastrophe to blow up.

**Root cause (mine, exposed rather than introduced).**  The tripwire divided the
reconstructed GRID power by the input power inside the APERTURE circle measured
on the `E_in` grid.  Two ordinary geometries make that the wrong denominator, and
the D3 fixture has both:

* the launch sampler CLAMPS `E_in` at the grid edge
  (`fi = np.clip((xs_in/dx) + N/2, 0, N-1-1e-9)`), so an aperture wider than the
  `E_in` grid launches the edge amplitude over the whole rim annulus.  D3 is a
  6 mm aperture on a 48 × 25 µm grid — a 2.94 mm launch radius against a 0.6 mm
  grid half-width, i.e. ~24× the area at 0.61 of the edge intensity.  Far more
  power is launched than the aperture-circle sum counts;
* light that legitimately leaves the N × N output grid is off-screen, not lost.

Measured on the D3 fixture (`scratchpad/wpa3/p_energy_norm.py`):

| plane | shipped `P_out/P_aperture` | `P_out/P_launched-onto-grid` | branches |
|---|---|---|---|
| z = 38.000 mm (the D3 plane) | **2.325** | **0.9895** | 1 |
| z = 24.700 mm | **12.30** | **1.002** | 1 |
| z = 24.830 mm | **11.82** | **0.9626** | 1 |
| same lens, N = 256 (grid ⊃ aperture) | 0.9998 | 0.9999 | 1 |

So the old 10× bar was ALREADY firing falsely here (12.3 > 10) at two of those
planes; tightening it to 2× merely widened the false-positive band to the plane
the delta-audit test uses.  The energy is in fact conserved to 0.2 % / 3.7 %.

**Changed.**  `_lens_traced_multibranch.py` now normalises by the power the
launch congruence actually carries ONTO THE GRID — `|E_launch|²` summed over
alive launch nodes whose mapped exit point lands inside the grid, times the
launch cell area `h²`, against `dx²·Σ|E_out|²`.  Numerator and denominator are
then the same physical quantity, and both geometry effects cancel.  The three
messages say "the launched power that reaches this grid" instead of "the input
aperture power"; `return_diagnostics['power_ratio']` is now that ratio (1.0 =
energy conserved).

**Verified — the correction costs NO detection power.**  Re-running the audit's
own scan (`repro/TR-MAIN-2/p3b_scan.py` geometry: f≈25 mm singlet, N = 512,
dx = 4 µm, w₀ = 0.30 mm, `ray_subsample=4`) with both normalisers side by side
(`scratchpad/wpa3/p_energy_scan.py`).  On that fixture the beam is far inside
the grid, so the two agree to 4 digits everywhere and every real detection is
kept:

```
    z (mm)     P/Pgrid   P/Plaunch   nb   deg%  warns
    0.0000      0.9991      0.9991    1   0.00  0
   20.0000       0.999      0.9991    1   0.00  0
   24.0000      0.9996      0.9996    1   0.00  0
   24.7000       3.988       3.988    1   7.62  1      <- audit's silent 4x
   24.7400       8.101       8.102    1  10.87  1      <- audit's silent 8x
   24.7600       13.07       13.07    1  13.95  1
   24.8000       61.78       61.78    1  59.86  2      (+ the 25% skip warning)
   24.8200     REFUSED  (identically-zero field)
   24.8340     REFUSED  (identically-zero field)       <- the BFL
   24.8600       106.5       106.5    1   0.00  1
   26.0000      0.9991      0.9991    1   0.00  0
   30.0000       0.999      0.9991    1   0.00  0
```

**False-positive sweep.**  The boundary is approximate — a triangle can
straddle the grid edge, contributing to `p_out` while its nodes are excluded
from `p_in` — so I swept the pathological geometry itself: the D3 fixture
(aperture 24× wider than the grid) over 8 output planes × 4 launch densities
(`scratchpad/wpa3/p_energy_sweep.py`).  All 32 ratios land in **[0.70, 1.16]**
with **0 warnings**; the worst (0.7025, the coarsest launch at z = 80 mm) is
still well inside the [0.5, 2.0] band.

**Before/after on the reported test.**
`test_d3_air_focus_multibranch_runs_without_warning`: FAILED (`RuntimeWarning
… 2.32x`) → **passes**.  `test_multibranch_axial_focus_catastrophe_warns`
(v5.21 accuracy extensions, the fixture whose aperture fits inside its grid)
and both of my T2 regression tests pass unchanged — 21 passed in the combined
run, 188 passed across the caustic + a3 batch.

**Note for the audit trail.**  The `p_in` name is kept (it is the reference
power) but it no longer means "input aperture power".  A reader of the old
`power_ratio` diagnostic on an aperture-wider-than-grid fixture was reading a
geometry artefact, not physics.

### T3 (P1) — `newton_fit='spline'` with a vignetted ray

**Wrong.**  The launch lattice is a SQUARE of half-width `0.75*aperture`, so its
corners sit at `sqrt(2)*0.75 = 1.06` aperture radii — past any per-surface
`semi_diameter` of `aperture/2`, i.e. 2.12 clear-aperture radii.  Those rays die,
their forward-map entries become NaN, and `RectBivariateSpline` is an
INTERPOLATING (`s = 0`) FITPACK fit: one NaN sample makes ~90 % of the
coefficients NaN.  `valid = isfinite(opl_map)` is then all-False and the field
is identically zero.  The in-code comment claimed the spline would extrapolate
through the NaN; the only diagnostic was a 100 %-unconverged Newton warning
whose advice ("increase newton_max_iters") cannot help.

**Changed.**  Dead nodes are filled from their nearest LIVE neighbour
(`scipy.ndimage.distance_transform_edt`'s index output), so FITPACK gets a
NaN-free tensor grid; the filled mask travels in the Newton payload so the pool
and the serial closure mask identically; and any output pixel whose entrance
solution lands in a filled node's cell is NaN-ed exactly as an out-of-domain
pixel is.  A `RuntimeWarning` names the vignetting and points at
`newton_fit='polynomial'`; a prescription with no live launch ray raises.
`_warn_newton_unconverged` gains an `all_nan` arm that says "this is a BROKEN
FIT, not an iteration cap".

**Verified.**  `repro/orch/verify_trmain2.py` item 2: spline **0.0000 / 0
non-zero pixels → 0.9501 / 17 493**, against the polynomial control's unchanged
0.9754 / 21 601.  The 2.6 % power difference is the masked filled-node ring,
which is the honest outcome: those pixels have no ray physics behind them.

### T4 (P1) — `fast_analytic_phase=True`

**Wrong.**  `_geometric_lens_phase` did `from .. import raytrace as _rt` then
`_rt._surface_sag_xy(...)`; that name lives in `raytrace.surface` and the
package does not re-export it (and has no lazy `__getattr__`).  The loop
`continue`s only when `|n2 − n1| < 1e-15`, so every refracting prescription hit
the `AttributeError`.  A documented public kwarg and a GUI checkbox had never
worked.

**Changed.**  Imported from the module that defines it.  Three further
corrections in the same function: NaN outside the conic domain contributes zero
from that surface (with a warning naming the count) rather than propagating into
`delta_phase`; the bulk piston is accumulated as a float64 scalar and folded
modulo 2π before it touches a possibly-float32 accumulator; and the docstring's
"under 10 nm OPL on F/10+" is replaced by the measured thickness scaling.

**Verified.**  `repro/TR-INFRA/p3b_fastphase.py` / `TR-MAIN-1/p4_fastphase.py`
run clean.  Against `angle(apply_real_lens(ones))` on an N-BK7 biconvex
100/−100 at f/12.1 (N = 256, dx = 40 µm), piston removed: **3.3947e-04 rad max
at 1 µm of centre thickness, 3.3951e-02 at 100 µm, 6.8054e-01 at 2 mm** — linear
in thickness, not in f-number, which is why the old claim was 4× optimistic at
2 mm (14.1 nm rms / 41.6 nm PV).  float32-vs-float64 agreement **< 1e-4 rad** on
a 4 mm element at 1.31 µm, where the pre-fix float32 ulp at 2.9e4 rad is
1.95e-3 rad.

### T5 (P1) — `form_error` silently cancelled

**Wrong.**  `apply_real_lens` implements `form_error` as an additive sag map;
`raytrace.Surface` has no such field and `surfaces_from_prescription` never
reads the key.  The traced assembly is
`E_analytic * exp(i(k0·opl_traced − phase_analytic_lens))`, and BOTH analytic
legs carry `phi_form` while `opl_traced` carries none — so the two cancel and
the figure error vanishes from the answer.  Measured 254× suppression
(2.5698e-03 rad against the analytic model's 6.5410e-01 rad on a 250 nm PV
astigmatic map), with no warning and no mention in the docstring.

**Changed.**  I took option (b) of the audit's three, not (a): the screen
`phi_i = −k0(n_after − n_before)_i · form_error_i` — exactly what
`apply_real_lens` applies, same sign convention (CONVENTIONS §7) — is built once
from the prescription and added explicitly to `delta_phase`, on BOTH the banded
and whole-grid paths and on BOTH `preserve_input_phase` branches.  Refusal would
have been honest but would have left a documented prescription key unusable with
the accurate model; re-applying it makes the traced answer correct, and it is
verifiable against a closed form, which is why I did not think it "cannot be
subtly wrong" applied here.  The branch-enumeration modes, which have no
analytic leg to re-apply onto, DO refuse.

**Verified** against the exact screen over the bright support (N = 256,
dx = 25 µm, 250 nm PV astigmatic map on S1 of an N-BK7 100/−100 singlet,
`ray_subsample=8`):

| model | rms(applied − exact screen) | max | screen rms |
|---|---|---|---|
| traced, pre-fix | **0.564 rad (the whole screen missing)** | — | 0.564 rad |
| traced, after | **0.0034 rad (0.6 %)** | 0.0198 rad | 0.564 rad |
| `apply_real_lens` | 0.0145 rad (2.6 %) | 0.0640 rad | 0.566 rad |

The traced model now reproduces the screen more closely than the analytic
sibling does, because the analytic one additionally diffracts it through the
in-glass ASM legs.

**Residual risk.**  This is a DEFAULT CHANGE: a prescription carrying
`form_error` now returns a different (correct) field.  Migration note in the
changelog.  The screen is a thin-element treatment — it carries the figure
error's phase, not the transverse ray walk-off a perturbed surface would also
produce; that is the same idealisation `apply_real_lens` makes, and the
docstring now enumerates every phase-only feature and what happens to it.

### T6 (P1) — `_reverse_prescription`

**Wrong.**  Reversal is `z → -z`, so `sag → -sag` TERM BY TERM.  The code
negated `radius` / `radius_y` and left `conic` alone (both correct — the radius
flip supplies the conic term's sign) but passed `aspheric_coeffs`,
`aspheric_coeffs_y`, `freeform`, `tilt` and `sag_callable` through UNCHANGED.
Separately, `list(reversed(thicknesses))` is correct only under the
`len(thicknesses) == len(surfaces) − 1` convention;
`validate_prescription` accepts both, and under the other it made the reversed
GLASS gap the forward BFD.  `stop_index`, `elements` and every other top-level
key were dropped.

**Changed.**  Every sag term is negated (`freeform` coefficient-by-coefficient,
`sag_callable` wrapped); the thickness PAIRING is reversed after normalising
both conventions to "gap after surface i" and re-emitted in the caller's own
convention; `stop_index` is remapped to `n−1−i`, `elements` reversed, and every
unknown top-level key carried.

**Verified** with the auditor's own `repro/TR-INFRA/p4_reverse.py`:

| check | before | after |
|---|---|---|
| sag(reversed S1, h = 5 mm), `A4 = 1e3` | −2.500031447e-04 m (**err +1.25e-06 m = 2.13 waves**) | −2.512531447e-04 m (**err 0.0e+00**) |
| reversed glass gap, n-thickness form | 0.1 m (the forward BFD) | 0.005 m |
| forward vs backward OPL, n-thickness, h = 0/2/4 mm | Δ = −1.441e-01 m | **0.0 / 1.7e-18 / 5.2e-18 m** |
| back-traced ray vs launch height, h = 4 mm | +6.576e-03 m (err +2.58 mm) | +4.000000e-03 m (err 0.0) |
| top-level keys kept | `aperture_diameter` only | all, `stop_index` remapped, `elements` reversed |
| round trip `reverse(reverse(P))` | radii/glasses/thicknesses only | identical incl. `stop_index`, `elements`, `name` |

Bit-identical on the shipped path: `apply_real_lens` asserts the `n−1`
convention, where the new pairing produces the same list the old
`list(reversed(...))` did.

### T7 (P2) — `_sample_local_tilts` and the ndarray carrier index

**Wrong.**  `np.roll(E, -1, axis=1)` differenced the LAST column against column
0.  The docstring said the wrapped pixels "get low weights after the amplitude
mask", but the mask is `amp > 1e-3·amp.max()`, which any field that fills the
grid satisfies at the rim — and the shipped σ = 4 px amplitude-weighted Gaussian
then SPREAD the bad column 12 columns inward, because the amplitude weight is
uniform there.  Separately the forward difference estimates `dphi/dx` at `i+½`
and was stored at `i`, a `dx/(2R)` bias coherent across the pupil.  The
`_compute_carrier` ndarray branch truncated its grid index (`.astype(np.int64)`
after `+ N/2` is `floor`, not the "nearest neighbour" the sibling docstring
claims).

**Changed.**  Both estimators use the MIDPOINT construction the two sibling
estimators in the same file already use, with the half-pixel lookup offset
applied PER AXIS (L is midpointed in x only, M in y only) and an in-grid mask
preserving the documented "off-grid launch → zero tilt" policy.  The ndarray
carrier is sampled bilinearly.

**Verified** (`repro/TR-INFRA/p7_tilts.py`, `p8_misc.py`):

| quantity | before | after |
|---|---|---|
| whole-grid `max\|L − L0\|`, plane wave, σ = 0 | 1.475e-01 | **1.232e-15** |
| ...σ = 4 | 8.111e-02 | **2.220e-16** |
| mean bias vs the PIXEL-centre tilt, R = 0.10 m | +2.0000e-05 (= `dx/2R`) | **+4.48e-21** |
| ...R = 0.05 m | +4.0000e-05 | **−2.6e-22** |
| ndarray-carrier mean sampling offset | −0.4936 px | **−0.0000 px** |
| ndarray-carrier `max\|L − L_true\|` | 1.5800e-04 (1.3 % of span) | **2.67e-10** |

### T12 (P2) — `carrier=<ndarray>` accuracy

Same fix as T7's second half; reported separately because the audit measured it
end to end.  Against an independent exact-sphere trace launched along `grad W`
(`repro/TR-MAIN-1/p6b_ndarray.py`, `p6_carrier.py`), diverging 200 mm conjugate
through an f/32 singlet:

| carrier | before (rms vs oracle) | after |
|---|---|---|
| `None` | 6.97e-3 rad (1.45 nm) | unchanged |
| `200e-3` (float, exact sphere) | 1.845e-3 rad (0.385 nm) | unchanged |
| `'auto'` | 1.845e-3 rad | unchanged |
| `TiltedCarrier(...)` | 1.845e-3 rad | unchanged |
| **`ndarray` holding the SAME W** | **9.914e-2 rad (20.67 nm)** | **1.874e-3 rad (0.391 nm)** |

At the source, the eikonal error is now **0.269 / 0.066 / 0.015 nm rms** at
N = 256 / 512 / 1024 (was 301.4 / 150.9 / 69.0) and scales as `dx²` rather than
`dx`.

#### T12 follow-up — the D1 fail-before pin (found by VERIFY-A6)

VERIFY-A6 reported
`test_niche_d1_tilted_carrier.py::test_tilted_carrier_beats_the_equivalent_ndarray_wavefront`
failing at `assert 1.72e-08 > 1e-05`.  The pin is a FAIL-BEFORE arm: it
asserted the ndarray branch must be BAD, with a bar (`1e-5`) set by the
half-pixel nearest-neighbour error T12 removed.  The improvement is real and
intended, so the arm is restated — not deleted — as the property that now
holds, with a DERIVED bar.

The ndarray branch differentiates by `np.gradient` (central difference of step
`dx`) and then samples that derivative field bilinearly, so its error is the
sum of two textbook discretisations of the exact gradient
`g(x) = -uu/s`, `uu = x + R·L/N`, `s = √(uu² + R²)`, `g'' = 3R²·uu/s⁵`:

| term | bound |
|---|---|
| central difference | `(dx²/6)·max\\|g''\\|` |
| linear interpolation of a smooth function | `(dx²/8)·max\\|g''\\|` |
| **total** | **`(7/24)·dx²·max\\|g''\\|`** |

Measured on the fixture (R = −30 mm, L = 0.046, dx = 20 µm, queries at
0.37 dx and 4.5 dx — deliberately off-lattice):

| quantity | measured |
|---|---|
| `TiltedCarrier` gradient error | **0.0** (closed form) at every rung |
| ndarray gradient error | **1.7197e-08**, 1.6664e-08 |
| derived `(7/24)dx²max\\|g''\\|` | 1.7974e-08, 1.6919e-08 |
| ratio measured/derived | **0.957, 0.985** |
| the SAME branch pre-T12 (nearest-neighbour, truncating index) | **3.324e-04** |

The bar is met on a 4× refinement ladder (dx = 20 / 10 / 5 µm, ratios
0.57–1.00), which is the O(dx²) statement without a fragile measured ratio in
it.  The restated arm asserts, at each rung: the analytic gradient is exact to
`rtol=1e-12`; `err_ndarray <= 1.5·bar` (a return to nearest-neighbour
overshoots this by **4.3 decades** — 3.32e-04 against a 1.80e-08 bar, ~18 500×);
`err_ndarray >= 0.3·bar` (so the branch cannot silently become exact — e.g. by
being re-routed to the analytic path — without the test noticing); and
`err_analytic <= err_ndarray`, i.e. `TiltedCarrier` is still no worse.  Measured
by `scratchpad/wpa3/p_d1_carrier.py`; the whole file passes (**33 passed**).

### T8 (P2) — the `_multi` and `_segmented` contracts

**Wrong.**  `apply_real_lens_traced_multi(reuse_prepared=True)` raised an opaque
`TypeError: prepare_real_lens_traced() got an unexpected keyword argument` from
three frames down for twelve public kwargs — exactly the failure the v5.29
`_NO_SCREEN` block was written to close, which enumerated six of eighteen by
hand — and only on the DEFAULT reuse path, so the same call worked or crashed
depending on the carrier kind.  `apply_real_lens_traced_segmented` used `dy` for
the angular partition and `dx` alone for every traced pass, so the element's own
square-pixel refusal was never reached.

**Changed.**  The accepted set is DERIVED by comparing
`inspect.signature(apply_real_lens_traced)` against
`inspect.signature(prepare_real_lens_traced)` at call time, so a keyword added
to either signature is handled without editing a list; the refusal names the
keyword and `reuse_prepared=False`.  `_segmented` raises the same square-pixel
`ValueError` up front, and its single-segment path no longer forwards a
possibly-sequence `carriers` as a scalar `carrier=`.

**Verified.**  The regression test derives the missing set from the live
signatures and checks five representative keys; `segmented(dy=2*dx)` now raises
the same class the direct call does.

### T9 / T10 (P2) — memory and kernel work, all bit-verified

Measured with `time.perf_counter` medians of interleaved runs and tracemalloc
peaks, on a box shared with the other WPs.

| item | before | after | identity |
|---|---|---|---|
| numba Chebyshev kernel, order 6 (M = 28), 4 Mpt, 7 reps, min wall | 346.4 ms (86.6 ns/pt) | **213.4 ms (53.3 ns/pt), 1.62×** | `max\|diff\| = 0.000e+00` |
| ...order 10 (M = 66) | 236.4 ms (59.1 ns/pt) | **152.7 ms (38.2 ns/pt), 1.55×** | `max\|diff\| = 0.000e+00` |
| pure-NumPy Chebyshev fallback, n = 1e6, tracemalloc peak | 1160.00 MB (145.0 float64/pt) | **120.00 MB (15.0/pt)** | `np.array_equal` on all three outputs |
| ...n = 1e5 | 115.83 MB | 84.00 MB | bitwise |
| `inversion_method='fit'` design matrix | 141 full-grid units at N = 1024 (1.18 GB), 198 at N = 512 | chunked to `_CHEB_FIT_CHUNK_ENTRIES // M` | pointwise in the output pixel |
| whole-grid final masking | ≈7.1 units of `8N²` | ≈3.1 | same values (the banded path's own idiom) |
| `_ray_density_self_checks` | complex128 upcast of `E_in` + a full `\|E\|²` grid | `abs→float64` in place + `np.vdot` | agreement 2.2e-16 relative |
| segmented spectral windows | `(Kx+Ky+2)` full float64 grids (126 MB of 461 MB at N = 1024) | 1-D axes, product formed lazily; bins below `min_segment_power` skipped by Parseval before the inverse FFT | `_flattop_partition_1d` reads one coordinate; the 2-D form was `max\|Δ\| = 0.0` from the broadcast |
| `\|E_analytic\|` on the `sub>1` preserve path | full-grid float64 (14.17 ms / 8.4 MB at N = 1024) | strided (0.114 ms) | the only consumer is `amp[::sub,::sub]` |

The C13 step-down's blind spot is now MEASURED rather than argued: the solver
takes an optional `score_domain=` (the full-lattice design), and a residual TIE
hiding a difference above 1e-6 of the in-fit peak over the evaluation domain
warns.  At the shipped `newton_poly_order = 6` the measured difference is
3.9e-08, so the warning stays silent; at order 10 on a hard-mask disc it is
1.05e-02, which the file itself records as a measured-harmful configuration.
Which candidate is returned is unchanged, so no number moves.

`_script_has_main_guard` now asks the question the warning it feeds is about.
Against the audit's own `guard/g*.py` corpus, three verdicts change and the
other nine are unmoved:

| fixture | before | after | correct |
|---|---|---|---|
| `g6` inverted guard (`!=` + `raise SystemExit`) | True | **False** | False (a spawn child's `__name__` is `'__mp_main__'`) |
| `g9` `match __name__: case '__main__':` | False | **True** | True |
| `g12` decorative guard + unguarded 134 MB body | True | **False** | False — this is the 22.1 GB/worker failure the warning exists for |
| `g1`/`g2`/`g3` canonical, `g5` bare side effect, `g10`/`g11` text-only, `gbad` unparseable | — | unchanged | — |

`g7` (a guard through intermediate names) and `g8` remain False; both route to
SERIAL, which the audit calls a harmless false negative.

**Calibration.**  The audit's suggested predicate ("imports / simple constant
assignments / function+class definitions / `__future__`") refuses the two real
drivers in this repository -- `validation/repro_traced_carrier_121/
capstone_stageB.py` and `focus_scan_121.py` -- because both open with
`sys.path.insert`, an `os.environ.get` and the targeted `warnings.
filterwarnings` calls a runner installs at module scope precisely so its own
imports are covered.  Refusing there costs the 8-worker Newton pool on exactly
the drivers it exists for, and `test_verify_perf_fixes_2026_08_10.py::
test_capstone_stage_b_is_import_safe_and_blanket_free` pins that those files
ARE import-safe.  So the predicate additionally admits module-scope calls into
a NAMED set of four process-setup modules (`sys`, `os`, `warnings`, `logging`)
and the scalar-conversion builtins -- idempotent, allocation-free, and stated
as a named list in the code with its limitation.  Everything that calls
anything else at module scope stays unsafe, including `np.zeros((4096, 4096))`,
`load_config()` and `main()`.  Both drivers now read True and every g-fixture
verdict above is unchanged.

### T13 (P2) — the exit-NA guard

**Wrong.**  The trace runs on `pres_no_ap` (`aperture_diameter` popped), so
`surfaces_from_prescription` gives `semi_diameter = inf` unless the SURFACES
carry their own, while rays are launched out to `0.75·aperture` and the returned
field is masked to `aperture/2`.  The significance gate looked only at input
AMPLITUDE, which does nothing for a flat / top-hat / wide-Gaussian input — so
`na_exit` was the max over rays the output mask deletes.  It feeds
`propagate_traced_carrier_chain`'s `on_tilt_exact_grid`, whose default action is
`'error'`.

**Changed.**  `_sig` is intersected with the same aperture disc the output mask
uses (on the axis-centred launch lattice, which is why no origin term appears).

**Verified** on the file's own f/5 fixture (R = ±51.68 mm, ap 24 mm,
`E_in = ones`, N = 512) against the marginal ray at the aperture edge traced
independently: `na_exit` **0.78487 → within 0.3 % of the true 0.24964**
(pre-fix 3.144× overstated, which demanded `dx ≤ 0.83 µm` instead of 2.62 µm).

### T14 / T15 / T16 (P2/P3) — the doc-and-notice items

`ray_subsample`'s accuracy contract is restated with BOTH reconstructions'
measured numbers (the inverse-characteristic evaluator: 0.000 nm rms at sub 4,
8 and 16; the order-1 upsample: 3.40 / 11.63 / 44.18 nm rms), the list of
configurations that gate the evaluator off, and the `(sub·dx)²/(8·f_exit)`
bound.  Those configurations now emit the same accuracy notice an internal guard
refusal already emitted, so `use_gpu=True` no longer trades 0 nm for 11.6 nm rms
in silence.

`parallel_amp_min_free_gb` becomes an explicit FLOOR and the gate scales with
`E_in.nbytes` (`max(2 GB, 6·nbytes)`, which is the measured 26.0-vs-18.0
grid-unit cost).  `amp` is taken strided where that is all anyone reads.  The
`phase_analytic_lens` cache and the `_pip_residual_ri` band are deferred (§6).

`on_noncollimated='delegate'` now names the four physics-affecting knobs it
used to drop unreported (`newton_amp_mask_rel`,
`newton_mask_dilate_coarse_px`, `beam_centre`, `fast_analytic_phase`) plus
`origin` and the three `caustic_*` sub-knobs, and ALWAYS announces the model
swap — it used to be silent when only defaults were passed.  The pure policy
and resource knobs are deliberately NOT listed: none of them changes an answer,
and a caller who set one to keep the output quiet should not be answered with a
warning about it (an existing test,
`test_niche_audit_e_prepared_and_enums.py::test_l22_delegate_reports_the_discarded_physics_kwargs`,
pins exactly that, and it passes).
`return_screen=True` with `'delegate'` raises rather than returning a "screen"
that contains `E_in`.  `float(sum(thicknesses))` became a real validation with a
message; the `_warn_if_aperture_exceeds_grid` swallow now logs what it caught.
The `prefilter` comment, the trailing-band extrapolation, the
`_opl_by_backward_trace` validation claim and its on-axis reference index, and
the `on_noncollimated='off'` cost claim are all corrected against measurement.

### S8 (P2) — the inverse-map cache key and the GRAM guard's message

**Wrong.**  `_imap_key` hashed 12 scalars plus every input array but none of the
`_lens_traced` flags that change the arithmetic of the solve, while
`_det_traced()` reads `DETERMINISTIC_TRACED_FIT` at CALL time — so
`traced_flags(DETERMINISTIC_TRACED_FIT=False)`, the documented bit-for-bit
fail-before for the whole layer, was defeated by a cache HIT.  `report_refusal`
told the user a refusal "costs speed, never accuracy", contradicting the
module's own measurements three hundred lines above.

**Changed.**  Nine flags are in the key (`_IMAP_KEY_TRACED_FLAGS`, read by name
at build time for the same reason `_det_traced` reads its flag at call time).
`report_refusal` states what a refusal costs, with the module's own numbers; the
GRAM guard records whether its budget came from an explicit `set_max_ram` or
from psutil's live AVAILABLE reading and says so, because in the latter case the
returned FIELD depends on what else the machine is doing.

**Verified** with the auditor's own `repro/TR-SIBLINGS/imap_cache.py`:
`key(det=True) == key(det=False)` **True → False**, "SAME OBJECT served for
det=False" **True → False**; same for `LSTSQ_CONDITIONING_STEPDOWN`.  The
module's accuracy claim re-measured on the same fixture and unchanged:
`parity_map_opl_waves` 3.598e-12 against the incumbent's 1.845e-08.

### S9 (P2) — the rasteriser and the uniform dark fill

**Wrong.**  Triangles were batched by a POWER-OF-TWO bounding box with nothing
bounding `n_tri × 2^{2c}`.  At `output_plane_distance = 0` — the DEFAULT — the
map is near-identity, so every triangle got the same (padded) box and most of
the barycentric arithmetic was discarded: measured 144 s at N = 2048 and 882 s
at N = 4096 for the exit-vertex call against 0.85 s for the same grid near
focus, and 7.74 GB traced / 12.2 GB RSS with no warning and no model in
`estimate_lens_memory`.  The Ludwig band swap was a per-pixel Python loop
(0.76–1.07 ms per multi-branch pixel).  The uniform dark fill evaluated the CFU
Airy kernel on every pixel outside `r_c` (100.0 % of the grid at N = 2048) for a
tail that dies within ~15 Airy lengths.

**Changed.**  Batches share an EXACT `(wx, wy)` bounding box (falling back to
the power-of-two grouping above 96 distinct shapes, where the caller's range
mask is restored) and are capped at `_RASTER_CHUNK_ENTRIES = 4e6` entries; the
Ludwig swap is one padded-array pass with the same selection rule; the dark fill
covers `r_c < r < r_c + 20·l_airy`.  The `scipy.special.airy` import moves to
module scope.

**Verified.**  The contribution SET is unchanged by construction (the `inside`
test is the same predicate; only `np.add.at`'s summation order on a multi-branch
pixel can move, which this module already documents).  The k1 / k4 / R2 suites
pass unchanged — see §4 — and the S1 end-to-end field comparison above runs
through the new rasteriser.  I did NOT re-run the N = 4096 / 12 GB fixture: the
machine is shared and the audit's own measurement is the baseline.

**Residual risk.**  The exact-box path is the common case; the power-of-two
fallback is reached only above 96 distinct shapes, where the behaviour is the
pre-fix one plus chunking.  I did not construct a map with that many shapes.

### S11 (P3) — the small items in my files

* KMAH NaN-fill: `np.roll` → `_shift_clamped` (edge-repeat), with a no-progress
  break so a fully dead lattice cannot spin to the iteration cap.  Verified
  against the closed-form clamped shift for four shifts on both axes, and
  asserted to DIFFER from `np.roll` on the fixture (so the test can see it).
* `_count_interior_turning_points` is now called by `_trace_meridional_fold`
  (CORRECTED 2026-09-12 by VERIFY-A3 OI-6: an earlier draft of this line
  said "both meridional traces".  `_trace_meridional_cusp` still counts
  with the raw `np.diff(np.sign(dxo))` form, because it needs the turning
  POSITIONS and the count together and gates on `turns.size != 2`; that is
  a separate item, not something this change made).
  Verified: on `[0, 1, 2, 2, 1, 0]` the raw `diff(sign(diff))` form counts 2 and
  the robust counter 1 — the difference between routing a clean FOLD to the
  Airy completion and misrouting it to the Pearcey cusp path.
* `_build_pearcey_cusp_field`'s unread `wavelength` parameter is removed and the
  λ-freedom explained (the scaling is absorbed where the control coordinates are
  fitted to phases in radians).
* `L0`/`M0` renamed `p0x`/`p0y` etc. where they were rebound from launch
  direction cosines to per-vertex slowness components.
* `_math.chebyshev.chebyshev_fit_2d`'s unconditional round-trip claim is now
  conditional, with a `RuntimeWarning` on an off-centre grid whose fit carries a
  non-constant term (a `T_0 T_0` fit is shift-invariant and stays silent).

The other four S11 items are in `_lens_jax.py` (the 1.02 launch radius),
`lenses_maslov.py` (the four integrators, the dead `uniform_fold_airy` /
`pearcey`) and `fga.py` (`fold_split` and `dy`) — §5.

### §15.9 — `caustic='wave'`, the ray-to-wave hand-off

**Why.**  Geometric optics is EXACT at the exit pupil and singular at a caustic,
so the honest way to reach a caustic plane is to stop tracing at the pupil and
finish in wave optics.  The audit's own 25-line oracle did exactly this and ran
2× faster than the multibranch and 15× faster than the uniform path at N = 4096,
while reproducing the dark-side tail the multibranch drops to zero.

**How.**  ONE recursion with `caustic='single'`, `output_plane_distance=0` — so
every other keyword of the function applies verbatim, and the keyword set is
snapshotted from `locals()` against the live signature rather than a
hand-maintained list — plus one band-limited ASM leg at the exit medium's
in-medium wavelength (`resolve_exit_index` picks `n_exit` from the
prescription).  CPU only; `output_plane_distance != 0` is now accepted for this
mode as well as the two branch-enumeration ones.

**Verified** (f ≈ 25 mm biconvex `_A3GLASS`, N = 512, dx = 4 µm, w₀ = 0.30 mm,
BFL = 24.4833 mm from a ray trace of the prescription):

| plane | `caustic='wave'` | `caustic='multibranch'` | independent oracle (`apply_real_lens` + ASM) |
|---|---|---|---|
| d = 0 | **bit-identical to the traced call** (`array_equal` True) | — | — |
| d = 2 mm (no caustic) | P/P_in 1.000000, peak 1.119 | 0.999038, peak 1.119 | — (agreement 1.7 % over the bright core) |
| d = BFL/2 | 0.999999, peak 2.039 | 0.999074, peak 2.055 | — (12.5 % — the branch sum is already degrading) |
| **d = BFL** | **0.999999, peak 8.5781, EE(5/10/25 µm) 0.0408 / 0.1609 / 0.6348** | **RuntimeError (nothing to return)** | **1.000000, peak 8.5781, 0.0408 / 0.1609 / 0.6348** |

Peak pixel identical to the oracle's.  NOT made the default for
`output_plane_distance != 0`: the existing multibranch/uniform tests pin the
branch-sum field, and switching the default would move every one of those
numbers — the docstrings recommend it instead.

**Residual risk.**  Validity is the ASM's, not geometry's: the pupil must be
sampled at `dx <= λ/(2·NA_exit)` (the `on_undersample` guard already measures
`NA_exit`, and T13 makes that measurement trustworthy) and the grid must hold
the beam at the output plane.  Both are stated in the docstring.  The mode does
not produce a KMAH branch decomposition, which is the one thing the multibranch
is for.

### WP-A2 follow-ups landed in my files (their report §5, items 3 and 5)

WP-A2 asked for two changes in `_lens_traced.py` while I still own it.  Both are
bit-identical on every default path; each has a regression pin in
`tests/unit/test_audit2609_a3_traced_lens.py`.

**(a) `stop_index` is read through `_lens_real._normalise_stop_index`.**  An
out-of-range stop matches no surface AND suppresses the entrance aperture, i.e.
it silently removes every aperture mask.  The traced entry only tested
`int(stop_index) != 0`.

*Before* (measured by loading a copy of `_lens_traced.py` with the hunk reverted
as a sibling module — `scratchpad/wpa3/p_stop_before.py`), on a 2-surface lens:

```
  stop_index=2       : ValueError apply_real_lens: prescription['stop_index']=2 is out of range ...
  stop_index=5       : ValueError apply_real_lens: ...
  stop_index=-3      : ValueError apply_real_lens: ...
  stop_index='first' : ValueError invalid literal for int() with base 10: 'first'
```

i.e. the caller of `apply_real_lens_traced` was told about `apply_real_lens`,
from inside the analytic amplitude leg's worker thread, AFTER the ray trace had
run and after a "non-entrance stop, use apply_real_lens" RuntimeWarning had
already fired — and a non-integer produced a bare `int()` message.

*After* (`scratchpad/wpa3/p_stop_index.py`):

```
  stop_index=None : ok  identical=True   warn=[]
  stop_index=0    : ok  identical=True   warn=[]
  stop_index=1    : ok  identical=True   warn=['... specifies stop_index=1, ...']
  stop_index=-1   : ok  identical=True   warn=['... specifies stop_index=1, ...']
  stop_index=-2   : ok  identical=True   warn=[]
  stop_index=2       : ValueError apply_real_lens_traced: prescription['stop_index']=2 is out of range ...
  stop_index=5       : ValueError apply_real_lens_traced: ...
  stop_index=-3      : ValueError apply_real_lens_traced: ...
  stop_index='first' : ValueError apply_real_lens_traced: prescription['stop_index'] must be an integer ...
```

Every spelling of the SAME surface returns a **bit-identical** field --
`None` = `0` = `-2` (the entrance) and `1` = `-1` (the rear) on this
2-surface lens.  (CORRECTED 2026-09-12 by VERIFY-A3 OI-7: the sentence
used to read "every in-range spelling", which is over-general -- the two
CLASSES differ, as they physically must, because the stop moves through
3 mm of glass.)  Two incidental
improvements fall out of the normalisation: `-1` now reports the surface it
actually selects (`stop_index=1`) rather than the raw `-1`, and `-2` — which IS
the entrance on a 2-surface lens — correctly stops warning.  A prescription with
no `surfaces` key at all is left alone, so `validate_prescription` keeps
ownership of that diagnosis.  The sibling kernels (`_lens_traced_multibranch`,
`_lens_traced_uniform`, `_lens_imap`) do not read `stop_index` at all, so the
entry point is the whole surface.

**(b) the pre-flight aperture-vs-grid notice passes its own y extent.**
`_warn_if_aperture_exceeds_grid` gained `N_y=`/`dy=`; the traced call passed
`shape[0]` (Ny) together with `dx`, which describes a semi-extent that exists on
neither axis of an anamorphic grid.  On a TALL grid it reports the WIDE y
half-width and stays silent while the aperture over-fills x.  Measured
(`scratchpad/wpa3/p_anam_notice.py`) on Nx = 64 / Ny = 256 / dx = dy = 20 µm
with a 2 mm aperture (x half-width 0.64 mm vs a 1.00 mm semi-diameter):

```
OLD spelling (shape[0], dx)            : 0 warning(s)
NEW spelling (shape[1], dx, N_y, dy)   : 1 warning(s)  '... 3 prescription aperture(s) exceed the simulation grid (N=64, dx=20...'
```

The traced entry refuses a non-square grid a few lines later, so this only ever
changes what the caller is told on the way to that refusal — but it is now the
axis that actually truncates.  On a square grid the call is unchanged by
construction (`N_y = N`, `dy = dx`).

---

## 3. Files touched

Modified (all owned):

* `lumenairy/elements/_lens_traced.py` — T1, T3, T4, T5, T6, T7, T8, T9, T10,
  T11, T12, T13, T14, T15, T16, the three hand-written exit-vertex copies routed
  through `TraceResult.at_exit_vertex`, `caustic='wave'`, and WP-A2's two
  follow-ups (`_normalise_stop_index`; the `N_y`/`dy` aperture notice).
* `lumenairy/elements/_lens_traced_multibranch.py` — S1, T2 (including the
  tripwire's launched-power normaliser), S9, S11.
* `lumenairy/elements/_lens_traced_uniform.py` — S1, S9, S11.
* `lumenairy/elements/_lens_imap.py` — S8, T11 (`_cheb_dvander`).
* `lumenairy/_math/chebyshev.py` — S11 (`chebyshev_fit_2d` round-trip claim).
* `tests/unit/test_niche_k4_uniform_caustic.py`,
  `tests/unit/test_niche_r2_pearcey_cusp.py` — their `SimpleNamespace` trace
  stubs now offer the exit-vertex contract the modules use (the stub rays are
  already on the vertex plane, where the transfer is the identity, so the
  fixtures keep testing exactly what they tested).
  `test_niche_r2_pearcey_cusp.py` also drops the removed `wavelength` argument.
* `tests/unit/test_fix_newton_pool_memory.py` — the `__main__`-guard
  classification table (three shapes reclassified, two fixtures that used
  `x = 1` as "unguarded" now use a top-level call).  **This test asserted the
  OLD behaviour** — see §4.
* `tests/unit/test_niche_d14_deterministic_carrier_fit.py`,
  `tests/unit/test_niche_d15_deterministic_traced_fit.py` — four solver spies
  take `**kw` so a diagnostic-only keyword cannot turn the census into a
  `TypeError`.
* `tests/unit/test_niche_d1_tilted_carrier.py` — the ndarray-carrier
  fail-before arm is restated against a derived `(7/24)dx²max|g''|` bar (T12
  made that branch 4.3 decades more accurate, so the old `> 1e-5` bar measured
  a defect that no longer exists).  **Asserted the OLD behaviour** — see §4.
* `tests/unit/test_niche_c1_consolidation.py` — the exit-NA assertion reads both
  NAs out of the message and pins the RELATION instead of one build's string
  (T13 legitimately moved it).  **Asserted the OLD behaviour** — see §4.
* `tests/unit/test_niche_s10_sibling_patterns.py` — the multi-mode arm is
  restated as a degeneration bound instead of a monotone-decay trend.
  **Asserted the OLD behaviour** — see §4.

New:

* `tests/unit/test_audit2609_a3_traced_lens.py` (36 tests)
* `tests/unit/test_audit2609_a3_caustic_siblings.py` (19 tests)
* `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A3_REPORT.md`
* `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A3_CHANGELOG.md`

No git write commands were run.  `ruff check` is clean on every file above.

---

## 4. Tests run

All with `OPENBLAS_NUM_THREADS=1`.  I did NOT run the full suite (COMMON rule 5).

| command | result | duration |
|---|---|---|
| `pytest tests/unit/test_audit2609_a3_traced_lens.py tests/unit/test_audit2609_a3_caustic_siblings.py -q` | **53 passed** | 7.7 s |
| `pytest tests/unit/test_niche_c15_inverse_map.py test_niche_d15_deterministic_traced_fit.py test_banded_ray_density_and_inverse_map.py test_niche_p11_ray_density_amplitude.py -q` | **85 passed** | 492 s |
| `pytest tests/unit/test_niche_r2_pearcey_cusp.py test_v5_1_0_agent_a.py test_niche_k4_uniform_caustic.py -q` | **44 passed** | 25.6 s |
| `pytest tests/unit/test_fix_newton_pool_memory.py -q` | **51 passed, 1 failed, 3 skipped** — the failure is environmental (see below) | 15.8 s |
| `pytest tests/unit/test_niche_d14_deterministic_carrier_fit.py test_niche_c13_lstsq_conditioning.py test_niche_newton_pool_both_fits.py -q` | **61 passed** (with stdin redirected; see below) | 241 s |
| `pytest tests/unit/test_v5_4_chebyshev_fit_2d.py test_v5_2_chebyshev_extraction.py -q` | **20 passed** | 2.9 s |
| `pytest tests/unit/test_niche_audit_e_prepared_and_enums.py -q` | **41 passed** | 64.6 s |
| `pytest` over the fit-guard / origin / banding batch (c1, c6 ×2, c7, c8, s12, d9, tilt-quadratic, chunked-sag) | **156 passed** | 540 s |
| `pytest` over the new + edited test files together (a3 ×2, s10, r2, k4) | **102 passed** | 139 s |
| `pytest tests/unit/test_fix_newton_pool_memory.py test_verify_perf_fixes_2026_08_10.py -q` | **77 passed, 6 skipped**, 1 environmental failure | 16.6 s |
| `pytest` over the chain / carrier-eikonal / relay batch (d2, r6, r7, h3, h6, e4, p1) | **77 passed** | 854 s |
| `pytest` over the prepared / guards / perf batch (e-prepared-enums, d3, p2, upsample-lattice, perf-poly-locals, perf-round2, verify-perf, runner-oom) | **187 passed, 3 skipped** after the two restatements below | 536 s |
| `pytest` over the caustic / fit-domain batch (k1, k4, r2, v5_21_lens_accuracy_extensions, c14, c11, c12, d7) | **167 passed, 2 failed, 2 deselected** — the two failures are another WP's (proved below) | 622 s |
| `pytest tests/unit/test_v5_21_delta_audit.py::test_d3_air_focus_multibranch_runs_without_warning test_v5_21_lens_accuracy_extensions.py::test_multibranch_axial_focus_catastrophe_warns test_audit2609_a3_caustic_siblings.py -q` (after the T2 normaliser fix) | **21 passed** | 4.8 s |
| `pytest` over the caustic + a3 batch (v5_21_delta_audit, k1, k4, r2, s11, a3 ×2) after the T2 normaliser fix | **188 passed** | 95.8 s |
| `pytest tests/unit/test_audit2609_a3_traced_lens.py -q` (after the WP-A2 follow-ups) | **36 passed** | 5.9 s |
| `pytest` over the wide traced batch (audit_misc, carrier_referenced, g1_gate, h3, h6, w4, d14, **d15**, p1, p9, p11, v5_21_lens_accuracy_extensions) with `-s` | **501 passed, 3 skipped, 2 deselected** | 981 s |
| `pytest tests/unit/test_niche_d1_tilted_carrier.py -q -s` (after the restated T12 pin) | **33 passed** | 76.9 s |
| `python validation/run_all.py test_lenses` | **45/46 checks pass**; the one failure is `apply_real_lens_traced_jax` hitting WP-A4's new `jax_enable_x64` refusal in a validation script that does not enable x64 | 31 s |

The `scratchpad/wpa3/*.py` harnesses cited throughout are scratch probes in an
UNTRACKED `scratchpad/` directory — kept so every number here can be re-measured,
but not intended for the commit.

**Repro scripts re-run before and after** (each quoted in §2):
`repro/TR-SIBLINGS/repro_vertex.py`, `repro_vertex_field.py`, `imap_cache.py`;
`repro/orch/verify_trmain2.py`; `repro/TR-INFRA/p4_reverse.py`, `p7_tilts.py`,
`p8_misc.py`; `repro/TR-MAIN-1/p6_carrier.py`, `p6b_ndarray.py`.

**Existing tests that asserted the OLD behaviour, and were fixed.**

1. `tests/unit/test_fix_newton_pool_memory.py::test_the_guard_detector_reads_the_ast_not_the_text`
   parametrised `("x = 1\n", False)`, `("# if ...\nx = 1\n", False)` and
   `("def f():\n    if __name__ ...\n", False)` — i.e. it encoded the OLD
   predicate's semantics ("is there a top-level guard"), under which a module
   whose entire body is a literal assignment is "unguarded".  Under the audit's
   own recommended predicate ("every top-level statement other than imports /
   simple constant assignments / function+class definitions sits inside a
   guard") those three bodies are SAFE to re-run, which is the question the
   warning is about.  The table now carries the audit's classification plus the
   three shapes the old predicate got wrong (`g6`, `g9`, `g12`), and the two
   rows in `test_the_predicate_mirrors_multiprocessing` that used `x = 1` as
   "unguarded on purpose" use a top-level CALL instead — that row is about the
   `__spec__` / `__file__` dispatch, so it needs a body that is genuinely
   unguarded under either predicate.
2. The two `SimpleNamespace` `rt.trace` stubs (k4, r2) lacked `at_exit_vertex`.
   They stand in for a full `TraceResult`, so they now offer that contract; the
   synthetic rays are already on the vertex plane (`z = 0`, `N = 1`), where the
   transfer is the identity, so returning the same bundle is exactly what
   `TraceResult.at_exit_vertex` would return and each fixture keeps testing what
   it was written to test.
3. Four solver spies — two in `test_niche_d14_deterministic_carrier_fit.py`
   and two in `test_niche_d15_deterministic_traced_fit.py` — had fixed
   signatures `spy(A, b, deterministic=False)`; the C13 domain probe (T10) adds
   the keyword-only `score_domain` diagnostic argument, which turned the census
   into a `TypeError` inside the traced chain.  `**kw` pass-through in all four,
   with a note.  (The d15 pair is the one I missed on the first pass; it was
   caught by the wider regression batch and is fixed here.)
4. `test_niche_c1_consolidation.py::test_the_measured_na_guard_closes_the_paraxial_pre_checks_blind_spot`
   string-matched `'NA=0.34'` on the chain's high-NA warning.  T13 legitimately
   moves that statistic (0.3408 → 0.2205 on that fixture, by excluding rays the
   output mask deletes), and the guard's DECISION is unchanged — the grid still
   cannot carry the exit NA and 2.110 % of the exit power still aliases.  The
   assertion now reads both NAs out of the message and pins the RELATION
   `NA_measured > NA_grid > NA_paraxial` plus `NA_measured > 0.1`, which is
   what the message is claiming; a digit string that moves with a legitimate
   statistic is the `TESTING_STANDARDS` S5 shape.
5. `test_niche_s10_sibling_patterns.py::test_smooth_sigma_px_is_pixel_unit_but_core_tilt_is_dx_invariant`
   asserted a MONOTONE DECAY of the multi-mode tilt with finer pitch
   (`two[0] > 2 * two[-1]`), measured on the pre-fix estimator — whose readings
   carried the `np.roll` boundary wrap and the half-pixel storage offset T7
   fixes.  With both fixed the reading is 5–10× SMALLER at every pitch
   (9.4e-04 / 1.2e-03 / 2.0e-03 at N = 256 / 512 / 1024 against a pre-fix
   1.0e-02) and no longer monotone.  The DIRECTION of a trend in a quantity the
   test's own docstring says has no limit was never the property worth pinning;
   it now pins the documented DEGENERATION (the two-mode reading stays below
   0.1 × the physical fringe tilt at every pitch — measured 0.016 / 0.019 /
   0.033 of it, 3.0× below the bar, against a pre-fix 0.17 which is 1.7× ABOVE
   it) and keeps the single-mode dx-invariance arm untouched, which is the
   claim the F-B suspicion was actually about.
6. `test_niche_audit_e_prepared_and_enums.py::test_l22_delegate_reports_the_discarded_physics_kwargs`
   caught an OVER-report on my first pass: I had added every traced-only
   keyword to the delegate's dropped list, including the policy and resource
   knobs, and that test rightly pins that a defaults-only delegating call
   reports nothing DISCARDED (it passes `on_undersample='silent'`, which is a
   suppression knob, not physics).  The list is now the four physics-affecting
   keys plus `origin` and the three `caustic_*` sub-knobs, and the test passes
   UNCHANGED -- recorded here because the intermediate state is what the
   audit's own recommendation would have produced if taken literally.
7. `test_verify_perf_fixes_2026_08_10.py::test_capstone_stage_b_is_import_safe_and_blanket_free`
   caught the same class on the `__main__`-guard predicate; see the
   Calibration note under T10 in §2.  The test passes UNCHANGED.

8. `test_niche_d1_tilted_carrier.py::test_tilted_carrier_beats_the_equivalent_ndarray_wavefront`
   asserted `err_ndarray > 1e-5` — a fail-before pin whose bar was the
   half-pixel nearest-neighbour error T12 removed (measured `1.72e-08 > 1e-05`
   fails).  Restated, not deleted, against the derived O(dx²) discretisation
   bar on a 4× refinement ladder; full derivation and numbers under the
   "T12 follow-up" heading in §2.  Reported by VERIFY-A6.

**An existing test that asserted the RIGHT behaviour and caught a real defect
in my change.**
`test_v5_21_delta_audit.py::test_d3_air_focus_multibranch_runs_without_warning`
(reported by VERIFY-A14) was correct and my two-sided tripwire was wrong: its
normaliser could not tell a geometry artefact from an energy blow-up on a
fixture whose aperture is wider than its grid.  I did NOT touch the test.  The
denominator is fixed instead — full analysis, measurements and the audit-scan
re-run under "T2 follow-up" in §2.  Worth stating plainly: the SHIPPED 10× bar
was already firing falsely on that geometry (12.3× measured at a plane where the
launched energy is conserved to 0.2 %), so the defect predates my change and my
tightening to 2× only made it reachable by an existing test.

**Pre-existing / environmental failures found, with my judgement.**

* **`test_niche_d7_decentred_fit.py::test_the_off_centre_fit_order_raise_flattens_the_exit_wavefront[0.5]`
  and `[1.0]` — NOT WP-A3.**  Proved, not asserted: I loaded the PRE-WP-A3
  `_lens_traced.py` (`git show HEAD:...`) into a fresh module bound to the same
  package, rebound `lumenairy.apply_real_lens_traced` to it and re-measured with
  everything else current (`scratchpad/wpa3/bisect_d7b.py`).  The three numbers
  are IDENTICAL either way:

  | arm | 2026-07-29 (the docstring) | git HEAD `_lens_traced.py`, tree current | WP-A3 tree |
  |---|---|---|---|
  | on axis | 41.1 µrad | **41.089** | **41.089** |
  | decentred, pre-D7 | 134.8 µrad | **233.859** | **233.859** |
  | decentred, post-D7 | 2.4 µrad | **44.457** | **44.457** |

  The ON-AXIS arm is unmoved to four digits while both DECENTRED arms have
  moved — the signature of a change to the rays at LARGE heights, which an
  off-centre disc of radius `r` about a chief ray `|c|` off axis reaches
  (`|c| + r`) and the concentric disc does not.  The fixture is a `K = -n²`
  CONIC, so WP-A1's R4 (the exact conic quadratic replacing the ray-SPHERE
  discriminant as the intersection seed AND miss test, committed in `f602b72c`)
  is the obvious candidate; `_lens_real.py` and `lenses.py` are also dirty from
  another WP.  I did not chase it further: it is outside my ownership, and my
  own files demonstrably do not move it.  The owner of whichever change it is
  should re-derive this test's recorded 2.4 / 3.4 µrad.
* `test_fix_newton_pool_memory.py::test_a_spawn_worker_really_does_rerun_an_unguarded_main`
  fails under this agent shell with
  `OSError: [WinError 6] The handle is invalid` (or `WinError 50`) raised inside
  `subprocess.Popen._get_handles`, BEFORE any assertion.  The child is spawned
  with `capture_output=True` and stdin INHERITED, so `_get_handles` duplicates
  the parent's stdin — which under pytest's capture in this shell is not a real
  Windows handle.  Re-running with capture disabled (`pytest -s`) **passes**;
  so does redirecting stdin, depending on the shell.  Unrelated to WP-A3: it
  spawns a subprocess and never touches the predicate.  The same cause makes
  eight tests in `test_niche_d14_deterministic_carrier_fit.py` and three in
  `test_niche_d15_deterministic_traced_fit.py` fail in this shell; all of them
  pass with `-s` (d15: **20 passed in 63.7 s**).
* **Seven tests in `test_niche_d1_tilted_carrier.py` — NOT WP-A3, and NEWER
  than my last source edit.**  Six ERRORS (`test_chief_ray_closure_*`,
  `test_tilted_relay_lands_*`, `test_tilted_relay_reaches_*`,
  `test_scalar_chain_smears_*`, `test_energy_is_conserved_*`,
  `test_exact_final_leg_*`) plus
  `test_per_order_residual_stays_inside_the_documented_envelope`, all raising
  the SAME `RuntimeError` from `propagators/carrier.py:_check_focus_containment`
  via `_guard_dispose(action='error')`.  The guard's floor was a bare
  `_FOCUS_READOUT_CONTAINMENT_MIN = 1.0` radius and is now
  `max(1.0, _FOCUS_READOUT_CONTAINMENT_FRAC · _achievable_focus_margin(...))`
  — the in-code comment cites **VERIFY-A6 OI-1**.  The d1 relay fixture sits at
  a measured containment of **1.033 radii**, which cleared the old floor and is
  under the new **2.880**.  Timeline, from file mtimes: my last
  `_lens_traced*.py` edit 07:24:53, my d1 edit 08:25:58, the whole d1 file
  **33 passed** immediately after, `carrier.py` edited **08:29:25**, d1 then
  1 failed + 6 errors with nothing of mine changed in between.  The guard looks
  right and I did not touch it; its owner should either re-size the d1 relay
  fixture's standoff (the message computes the one that restores the margin:
  4.277574e-04 m against the resolved 5.822468e-04 m) or pass
  `on_focus_containment='warn'` in that fixture.
* `test_audit_lens_models_2026_07.py::test_a1_auto_n_v2_resolves_demanding_default_quadrature`
  — **NOT WP-A3.**  Reproduces in isolation with no traced code in the call
  graph at all: it is a pure `apply_real_lens_maslov` fixture, and the arm that
  fails is its FAIL-BEFORE half (`assert il2(old32, truth) > 0.5`, measured
  **1.33e-04**).  Someone's Maslov quadrature change has made the old fixed
  `n_v2=32` default accurate on this fixture too, so the test can no longer
  demonstrate the contrast it was written for.  `lenses_maslov.py` is WP-A4's;
  its owner should re-pick the demanding fixture or restate the arm.

**Confirming the new tests fail on the pre-fix code.**  A scratch harness
(`scratchpad/wpa3/prefix_check.py`) restores each pre-fix implementation in
process (monkeypatch of the shipped helper or, for T10, of the whole predicate)
and re-runs the matching test function.  Result: **25/25 arms behaved as
expected** — every regression test targeting a fix FAILS on the pre-fix code
(S1 × 5, T2, T3, T5, T6 × 4, T7 × 2, T12, S8, S11 × 2, T10 × 3), while the four
designated CONTROLS pass on BOTH sides:

* S1 on a FLAT rear surface (`sag ≡ 0`) — the fixture the entire shipped corpus
  uses, which cannot see the P0 at all;
* T6 thickness pairing under the `n−1` convention, where
  `list(reversed(...))` was already correct;
* S11 turning-point count on a map with no exactly-flat slope sample, where the
  raw and robust forms agree;
* T10 on a canonical `if __name__ == '__main__':` guard.

T1, T4, T9, T10's kernel and T11 are demonstrated directly instead: the pre-fix
T1 expression raises in one line (asserted in the test), `_surface_sag_xy` is
still absent from the `raytrace` package namespace (asserted in the test), and
the T9/T10/T11 claims are bitwise identities measured against the un-chunked /
un-blocked / shared implementations in the tests themselves.

---

## 5. Requested changes outside my ownership

(Requests in the other direction — WP-A2's §5 items 3 and 5, both INTO
`_lens_traced.py` — are **done**; see "WP-A2 follow-ups" at the end of §2.)

1. **`tests/conftest.py`** (audit S8, second half).  Add
   `lumenairy.elements._lens_imap.inverse_map_cache_clear()` to the per-test
   teardown, next to the existing `_TRACED_KWARG_DEFAULTS_CACHE.clear()` inside
   `_module_flag_leak_guard`.  The flag-leak guard restores `_lens_imap`'s
   scalars but never drains `_IMAP_CACHE`, so a map built inside
   `test_niche_d15_deterministic_traced_fit.py` (five direct flag assignments)
   survives into every later test in the shard.  **My side is done**: the flags
   are now part of the key, so a stale map can no longer be SERVED to a caller
   who asked for the other setting — what remains is only retained memory and
   any flag not in `_IMAP_KEY_TRACED_FLAGS`.
2. **`lumenairy/elements/_lens_jax.py`** (WP-A4; audit S7 and S11).  The two JAX
   entry points still read `image_rays` through hand-written copies (WP-A1 §5
   item 6) and launch at `0.5*aperture*1.02` under a comment saying "just inside
   the physical aperture", where the NumPy siblings use `0.98` for a stated
   reason (rays AT the rim die and NaN-poison their neighbours' FD Jacobians).
   Pick one policy across the family and name the constant.
3. **`lumenairy/elements/lenses_maslov.py`** (WP-A4; audit S3, S10, S11).  The
   exit-vertex transfer (`tr.image_rays` → `tr.at_exit_vertex()`, and
   `refocus(tr, d)` for the `output_plane_distance` leg); the four integrators
   integrating different integrands; and the dead `uniform_fold_airy` /
   `pearcey` that no integrator calls.  `_lens_traced_uniform` imports
   `_fold_airy_eval` and `pearcey` from there and is unaffected either way.
4. **`lumenairy/propagators/fga.py`** (audit S10, S11).
   `apply_real_lens_universal` routes a TILTED collimated high-NA plane at its
   focus to `phase_screen` (the collimation test reads a pure tilt as
   "non-collimated"), and `fold_split`'s legs drop `dy`.
5. **`lumenairy/propagators/carrier.py`** (WP-A6).  T13 changes what
   `_exit_na_out['na_exit']` reports (0.78487 → 0.24964 on the audit's
   fixture) and that value feeds `on_tilt_exact_grid`, whose default action is
   `'error'`.  Any threshold calibrated against the OVERSTATED number will now
   be 3× conservative in the other direction and should be re-checked.  Also:
   TR-MAIN-2's unverified suspicion that `_exit_na_out` "fails open" when
   `_sig.any()` is False is now slightly more reachable, because `_sig` is
   intersected with the aperture disc — a prescription whose entire launch
   lattice falls outside its own `aperture_diameter` would leave the dict
   unfilled and `carrier.py:6892`'s `float(_na_diag.get('na_exit') or 0.0) > 0`
   would SKIP the refusal.  I did not construct such a prescription; the guard
   on your side would be to treat an unfilled dict as "unknown", not "zero".
6. **`lumenairy/propagators/carrier.py` (WP-A6 / VERIFY-A6 OI-1).**  Raising
   `_check_focus_containment`'s floor from a bare 1.0 radius to
   `max(1.0, _FOCUS_READOUT_CONTAINMENT_FRAC · _achievable_focus_margin(...))`
   makes seven tests in `tests/unit/test_niche_d1_tilted_carrier.py` raise
   (measured containment 1.033 radii against the new 2.880 floor) — six chain
   tests error at setup and
   `test_per_order_residual_stays_inside_the_documented_envelope` fails.  The
   guard looks correct; the FIXTURE is now under-standing-off.  The message
   already computes the fix (a 4.277574e-04 m standoff against the resolved
   5.822468e-04 m).  I did not touch either file — `carrier.py` is yours and
   the d1 chain arms are not what WP-A3 changed.
7. **`lumenairy/analysis/through_focus.py` and any tolerancing helper** — T5
   makes `apply_real_lens_traced` honour `form_error`, so a study that swapped
   models to get the nominal answer will now get the perturbed one.  That is the
   fix, but it is a behaviour change worth knowing about downstream.
8. **`lumenairy/ui/lens_options_dialog.py`** — the `fast_analytic_phase`
   checkbox's tooltip says "~25% speedup with <10 nm OPL error".  The knob now
   works (T4), but the measured error is ~7 nm rms PER MM of glass, not a flat
   10 nm; the tooltip should say so.

---

## 6. Deferred, with designs

1. **Cache `phase_analytic_lens`** (audit T15 / TR-MAIN-1 perf #3).  It is
   input-INDEPENDENT (the file says so at its construction), and building it
   costs a full `apply_real_lens` pass on a `ones_like` placeholder — measured
   1.0 s of a 5.76 s call at N = 1024, and a 17.18 GB allocation at N = 32768.
   Design: a module-level bounded cache keyed on
   `(prescription content hash, wavelength, dx, N, _carrier_reuse_key(carrier),
   wave_propagator, sag_dtype, bandlimit, amp_use_gpu)` — `_carrier_reuse_key`
   already exists for exactly this key and correctly returns `None` (never
   share) for `'auto'` and ndarray carriers.  Must go through
   `register_cache_clearer` so `clear_all_registered_caches` reaches it, and
   must be keyed on prescription CONTENT (the S8 lesson), which needs a stable
   hash of a dict containing ndarrays.  ~4 h including the cache-key test.
2. **Row-band `_pip_residual_ri`'s carrier de-chirp** (audit T15).  The
   `_resid_eik` exponential three lines below is already banded (`_bd =
   4194304 // N`); the carrier de-chirp at the top of the same closure is
   whole-grid and builds four full-size arrays — ≈15.0 GB of transient at
   n_fine = 16384 for a 4.30 GB answer, on the chain's default configuration.
   Design: extend the existing `for _b0 in range(0, N, _bd)` loop upward to
   cover the de-chirp and the normalisation, writing the two float64 outputs
   band by band.  Bit-identical (pointwise).  ~2 h including a byte-identity
   sweep against the whole-grid form.
3. **The coarse→fine upsample's trailing band** (audit T16).  `coords` reaches
   `(N−1)/sub` while the coarse lattice covers `0 … (Ns−1)·sub`, so the last
   `N−1−(Ns−1)·sub` rows/columns are extrapolated, and `mode='nearest'` makes
   that extrapolation a CONSTANT — measured 69.2 nm (N = 256) / 140.8 nm
   (N = 512) against a 0.64 nm interior error on an f = 100 mm defocus.  The
   aperture mask and the Newton out-of-domain NaN kill those pixels on every
   shipped configuration, so this is a missing enforcement rather than a
   measured defect; it is documented in the code with the numbers.  Design: pad
   the coarse lattice (and its NaN companion) by one linearly-extrapolated
   row/column before interpolating and shift `coords` by 1 — removes the leading
   term of the constant extension without moving any interior value.  ~2 h,
   needs a byte-identity check on the interior.
4. **`mode='mirror'` for the order-3 OPL upsample** (audit T16).  The prefilter
   is an IIR filter (pole 0.268), so the 0-fill and the `mode='nearest'`
   constant extension bleed inward with a transient decaying 3.73× per coarse
   cell — measured order 3 at 7.09 / 1.90 / 0.51 / 0.04 nm at 0 / 1 / 2 / 4
   coarse cells of inset against order 1's flat 0.64 nm, i.e. cubic is WORSE
   than linear within ~2 cells.  `mode='mirror'` would remove the
   constant-extension half.  Not done because it moves every carrier-path
   number and the band is inside the aperture mask on every shipped
   configuration.  ~3 h including the carrier-chain re-validation.
5. **Score the C13 step-down over the evaluation domain** rather than only
   report it (audit T10).  The probe is in place and warns; making it CHOOSE
   would change the shipped selection rule on a quantity with byte-identity
   contracts around it.  Design: on a residual tie, prefer the QR answer when
   the two differ over the full lattice by more than
   `_LSTSQ_SCORE_DOMAIN_TOL` — QR is backward stable and is 58 000× more
   accurate in the COEFFICIENTS at order 10 on a hard-mask disc — behind a
   module flag defaulting to the current behaviour for one release.  ~4 h
   including a determinism sweep.
6. **Clenshaw recurrence for the Chebyshev evaluator** (audit alternative #1).
   The tensor-product sum factorises, so one Clenshaw pass in `v` per `kx` then
   one in `u` is `O(order²)` with ZERO scratch and no index gather — it makes
   the allocation problem the T10 blocking works around disappear by
   construction.  Bits change (different summation order), so it needs its own
   flag and a fail-before.  ~6 h.
7. **A Bessoid / Pearcey completion for the axial focus** (audit
   alternative, TR-MAIN-2).  `caustic='wave'` makes the plane REACHABLE, which
   is what the P1 asked for; the multi-arrival branch decomposition AT an axial
   focus still has no canonical completion here.  Recorded so it is not lost.

---

## 7. Changelog text

`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A3_CHANGELOG.md`
