# WP-A7 -- Analysis metrics (`lumenairy/analysis/`) -- fix report

Findings A1-A7 of `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11.md` §8,
the `ANALYSIS.md` partition report, and the `ORCHESTRATOR.md` rows on
`diffraction_limited_peak` and `wave_opd_2d`.

Every number below was measured on this machine (CPython 3.14.6, numpy 2.4.6,
scipy 1.17.1, jax 0.10.1, `OPENBLAS_NUM_THREADS=1`), before and after, on the
audit's own repro scripts or on independent oracles built in the tests.

---

## 1. Summary

| ID | Status | Files:lines | Tests | Oracle | Measured before -> after |
|---|---|---|---|---|---|
| **A1** (P0) | **fixed** | `analysis/opd.py:67-497` (new masked unwrap kernels), `:1064-1101` (`wave_opd_2d` body), `analysis/__init__.py` | `tests/unit/test_audit2609_a7_opd_unwrap.py` (18) | analytic Zernike pupils (coma, defocus, annular coma); `repro/orch/verify_analysis.py`; `repro/ANALYSIS/p2b,p2c` | max abs OPD error on 0.5 λ rms coma **1.000 waves / 53.2 % of the pupil -> 0.000 waves / 0.0 %**; 1 λ coma 4.000 -> 0.000; 5 λ coma 19.000 -> 0.000; fitted OSA c8 at 0.2 λ rms **+0.1024 -> +0.2000** (true 0.2000); absolute piston on a 51.8-wave defocus pupil **17.0000 waves -> 0.0000** |
| **A2** (P0) | **fixed** | `analysis/through_focus.py:225-299` | `tests/unit/test_audit2609_a7_strehl_reference.py` (11) | a pupil that is aberration-free by construction; `strehl_phase_integral` on the paraxial-minus-sphere error; `repro/ANALYSIS/p6,p5,p3`; `repro/orch/verify_analysis.py` | `through_focus_scan` best Strehl for a **perfect f/2 lens 11.3200 -> 1.0000**; `diffraction_limited_peak` 1.342173e+04 -> 1.519335e+05 (= the perfect sphere's own peak); Strehl of an aberration-free pupil at f/50 / f/10 / f/5 / f/2.5 / f/2: 1.0000 / 1.0000 / 1.0015 / 1.0988 / 1.4292 **-> 1.000000000 at every f/#** |
| **A3** (P1) | **fixed** | `analysis/detector.py:389-413` (signature/doc), `:836-985` (reconstruction) | `tests/unit/test_audit2609_a7_detector_sh.py` (12); `test_niche_s12_shack_hartmann_reference.py` (corrected) | analytic tilt / defocus / astigmatism, scaled by the sensor's own measured slope gain | wavefront / truth for tilt 0.2 / 0.5 / 1.0 mrad **0.4741 / 0.4744 / 0.4725 -> 0.9481 / 0.9488 / 0.9450** (= the slope gain); defocus **0.4626 -> 0.9340**; `max abs wf` 5.4263e-07 -> 1.4351e-06 m (truth 1.5000e-06) |
| **A4** (P1) | **fixed** | `analysis/image_plane_wfe.py:198-241` (precision gate), `:263-460` (entry + docs), `:598-660` (collimated launch) | `tests/unit/test_audit2609_a7_image_plane_wfe.py` (13) | transverse-ray-aberration integral + traced longitudinal SA, both built in the test from a hand-made collimated trace | `object_distance = inf` now accepted: **img_d = 47.87519 mm = BFL exactly, PV 3.5944 waves** vs oracle 3.6586 (-1.8 %) and LSA-derived W040 3.651 (-1.6 %); the 1e6 m stand-in read **PV 213.5098 / RMS 61.9051** with the chief 589 µm off the vertex and `best_rms` at 96.6435 mm on a 47.9 mm-BFL lens -- and now warns |
| **A5** (P2) | **fixed** | `analysis/phase_retrieval.py:181-199`, `:800-813` (JAX twin); `analysis/plotting.py:92-131`, `:620-630`, `:705-750` | `tests/unit/test_audit2609_a7_misc.py` (7); `test_audit_analysis.py` oracles corrected | an exactly-solvable GS problem; the pixel-centre identity for `imshow` extents | GS reported error for an exact solution **3.775380e+02 (97 % of the target energy, flat over 50 iterations) -> 3.26e-29**; NumPy/JAX `err` ratio **N_pix -> 1**; `_auto_extent` pixel centres now coincide with the samples they display (were `+dx/2`; `plot_psf` was `-dx/2`) |
| **A6** (P2) | **fixed** (4 of 7 items; 3 deferred with designs, §6) | `analysis/through_focus.py:193-216`, `:462-540`; `analysis/beam_stats.py:189-238`; `analysis/polychromatic.py:28-50`, `:120-155` | `tests/unit/test_audit2609_a7_misc.py` (9); `test_perf_v4_12_0_through_focus.py` (bar re-derived) | per-plane `angular_spectrum_propagate`; `beam_centroid`/`beam_d4sigma` for bit-identity; the masked sum for `radial_power_bands` | 21-plane scan at N = 1024: **202.33 -> 128.76 ms/plane (1.57x, 36 % of the scan)**; H rebuild 52.92 -> 10.97 ms/plane (4.83x); `single_plane_metrics` 81.44 -> 52.44 ms/plane (1.55x, bit-identical); `radial_power_bands` at 256/1024 bands 387.8/1245.7 -> 129.8/129.3 ms (2.99x / 9.63x), bit-identical below 96 bands |
| **A7** (P3) | **fixed** (6 of 6 items) | `analysis/ghost.py:610-690`, `:800-990`; `analysis/aberration.py:496-560`, `:592-606`; `analysis/interferometry.py:23-95`; `analysis/zernike.py:173-268`; `analysis/strehl.py:158-180`; `analysis/__init__.py` | `tests/unit/test_audit2609_a7_misc.py` (8) | uniform-disc 50 %-energy radius `1/sqrt(2)`; DFT bin of the y fringe ramp; runtime export check; cache-key collision construction | ghost 50 %-EE radius **0.500 R (29 % low) -> 0.667/0.667/0.708 R** at 6/12/24 rings (truth 0.7071); `simulate_interferogram` y fringe frequency off by `dx/dy` -> exact; 7 names added to `lumenairy.analysis.__all__`; Zernike cache key now separates grids that differ only in X's interior |

Nothing the audit listed as **verified correct** changed: `p1_zernike`,
`p2_opd`, `p3_psf_strehl`, `p5_detector_ref` and `p11_opl` re-run with the
same numbers (Zernike OSA indexing and normalisation, the `dy != dx`
cache-key check, `astigmatism_mag_angle`, the three Strehl definitions at
λ/14, `check_opd_sampling`'s Nyquist factor -- still 0.0000 waves up to
1.02·dx_max and 1.0000 at 1.2·dx_max, `wave_opd_1d`'s exactness, the OPD sign
convention, `depth_of_focus`, ghost transmittance, `error_reduction` /
`hybrid_input_output`).

---

## 2. Per finding

### A1 [P0] -- `wave_opd_2d` row-then-column unwrap

**What was wrong.** `phase = np.angle(field)` was taken over the whole grid,
zero-amplitude exterior included (`np.angle(0) == 0`), then unwrapped
`axis=1` and `axis=0`. The row pass anchors at column 0 -- outside the pupil
-- so every in-pupil row picks up its own `2π·k`; the column pass then
re-anchors INDEPENDENTLY IN EVERY COLUMN, at that column's first in-pupil row,
and the wrap of the entry jump injects a column-dependent whole-wave offset.
`valid` was applied only afterwards. Confirmed on HEAD before touching
anything: `repro/orch/verify_analysis.py` reported `max |OPD error| = 1.000
waves, 53.2 % of the pupil wrong by > 0.4 waves` at 0.27 rad/sample, and
`repro/ANALYSIS/p2c` reproduced the audit's whole threshold table to the
digit.

**What I changed and why.** `_unwrap_2d_itoh` in `analysis/opd.py`: the
wrapped column-to-column steps are built in one N² buffer with the
out-of-mask steps zeroed, `cumsum`'d along rows, and the resulting runs of
valid samples are then linked to each other through their vertical neighbours
by whole waves. Because the mask is applied BEFORE the integration, no path
crosses the exterior; because every path stays inside the mask, the result is
independent of the path and therefore exact for a residue-free field. The
run-pair reduction is an O(V) block scan rather than a sort (boolean indexing
yields links in row-major order and a run lives in one row, so each distinct
run pair is one contiguous block), and the offsets are propagated over the
run graph by BFS, which also labels connected components.

I also added the audit's named robust alternative as an opt-in:
`unwrap='reliability'` is the quality-guided Herráez, Burton, Lalor & Gdeisat
(*Appl. Opt.* **41** (2002) 7437) unwrap -- reliability from the local wrapped
second differences, edges merged in decreasing reliability order through a
weighted union-find. It returns the same answer on residue-free data and
confines the damage on noisy data. It is refused above 1.1e6 samples with a
message pointing at `'itoh'` (its merge is a Python loop, ~9 s at that size).

The piston anchor moved from "wherever the integration started" to the valid
sample nearest `x = y = 0`, which is what makes a known-defocus pupil come
back with the right integer wave count.

**How I verified.** `repro/orch/verify_analysis.py` (the orchestrator's own
independent cross-check) went from `1.000 waves / 53.2 %` to `0.000 waves /
0.0 %`. `repro/ANALYSIS/p2c` now reads 0.000 waves and the exact true c8 at
every amplitude from 0.05 to 5.0 λ rms; `p2b`'s offset histogram is `{0: N}`
for all five geometries, including the ones with no zero border and with
trefoil. The new tests fail on the pre-fix algorithm (restored in-process)
with exactly the audit's numbers -- 1.000 / 1.000 / 4.000 / 19.000 waves at
0.2 / 0.5 / 1.0 / 5.0 λ rms of coma -- and the absolute-piston test fails at
17.0000 waves.

**Residual risk.** A radially symmetric aliased wavefront wraps onto the
exactly self-consistent phase of a lower-frequency wavefront: it leaves no
residue (measured 0.0000 waves at 16.5, 32.9 and 98.8 rad per sample) and NO
unwrap of any kind can distinguish the two. That is covered by the separate,
pre-existing `focal_length` sampling gate, is pinned as such in
`test_symmetric_aliased_defocus_has_no_residue_and_is_caught_by_sampling`, and
is now spelled out in the docstring rather than implied. Cost: the call is
1.10x the old one at N = 1024 (203 vs 184 ms), the price of integrating inside
the mask; the kernel itself is *faster* than the two `np.unwrap` passes (72.4
vs 110.9 ms), the overhead is the residue self-check and the anchor.

### A2 [P0] -- `diffraction_limited_peak`'s paraxial reference

**What was wrong.** The aberration-free reference was
`exp(-i k0 r² / 2f)` while `angular_spectrum_propagate` is exact, so the
denominator was itself aberrated by `W040 = (D/λ)/(128 (f/#)³)` waves, its
peak depressed by its own Strehl, and every ratio inflated by `1/S_ref`.
Confirmed on HEAD: a PERFECT exact-sphere pupil at f/2.5 reported
`peak/reference = 1.468`, and `repro/ANALYSIS/p6` reported
`through_focus_scan best Strehl for a PERFECT lens = 11.3200`.

**What I changed and why.** One expression: the reference sag is now
`sign(f) * (sqrt(x² + y² + f²) - |f|)`, the exact converging (or, for `f < 0`,
diverging) sphere, which reduces to `r²/(2f)` in the paraxial limit for either
sign and costs one `sqrt` per sample. Every consumer -- `through_focus_scan`,
`tolerancing_sweep`, `monte_carlo_tolerancing` and its JAX and linearised
variants, `polychromatic_strehl`, `polychromatic_psf`, `optimize`'s
`StrehlMerit`, the GUI dock -- goes through this one function, so no other
site needed touching (verified by `grep`: 12 call sites, all of them
`diffraction_limited_peak(...)`).

**How I verified.** Three independent oracles. (1) A pupil that is
aberration-free by construction reports Strehl `1.000000000` at f/50, f/20,
f/10, f/5, f/3.9, f/2.5 and f/2 (nine figures), against 1.0000 / 1.0000 /
1.0000 / 1.0015 / 1.0067 / 1.0988 / 1.4292 before. (2) The size of the removed
inflation is predicted by `strehl_phase_integral` (a different module, audited
correct) applied to the paraxial-minus-sphere phase error: measured/predicted
1.000 at f/5, 1.005 at f/2.5, 1.10 at f/2. (3) At f/50 the new and old
references agree to 1e-6 relative, so the change is confined to the
non-paraxial regime. End-to-end, `repro/ANALYSIS/p6` now reads
`diffraction_limited_peak = 1.519335e+05` = the perfect sphere's own peak, and
`best Strehl for a PERFECT lens = 1.0000`; `repro/ANALYSIS/p5` §C(ii) reads
`ASM paraxial/sphere peak = 1.00000`.

**Residual risk.** The reference is evaluated at `z = f` exactly, as before;
the focal shift of a finite-aperture converging sphere is a second-order
effect that both numerator and denominator share, and the measured Strehl of
an aberration-free pupil is 1.000000000 with it. Users with recorded Strehl
numbers from f/# faster than ~f/10 will see them drop -- correctly (migration
note in the changelog).

### A3 [P1] -- Shack-Hartmann `wavefront` a factor of 2 too small

**What was wrong.** `wavefront = 0.5 * (wf_x + wf_y)` averaged two ONE-SIDED
integrals whose sum, for any wavefront separable in x and y, is `2(W - W_00)`
-- so the average is exactly half. Confirmed on HEAD via `repro/ANALYSIS/p4`
§A: 0.4741 / 0.4744 / 0.4725 for tilt and 0.4626 for defocus.

**What I changed and why.** The reconstruction is now a zonal least-squares
solve of the co-located Southwell geometry (*JOSA* **70** (1980) 998): each
in-mask neighbour pair contributes `(W[q] - W[p])/pitch = (s[q] + s[p])/2`,
and the whole set is solved through the normal equations (a graph Laplacian,
pinned one node per connected component, direct sparse factorisation -- no
iteration tolerance, deterministic). That was the audit's second-choice fix,
chosen over the bare path integral because it uses both slope components at
every lenslet and is correct for wavefronts that are NOT separable, which the
old code's own comment admitted it mis-reconstructed. The path integral is
retained as `reconstruction='itoh'` (now trapezoidal, which is what the
docstring always claimed).

**How I verified.** `repro/ANALYSIS/p4` §A now reads 0.9481 / 0.9488 / 0.9450
for tilt -- exactly the slope gain the same run reports -- and 0.9340 for
defocus, with `max abs wf` 1.4351e-06 m against a truth of 1.5000e-06 m. The
new tests separate the factor from the sensor's ~5 % sub-aperture truncation
deficit by dividing by the measured gain, and cover astigmatism (opposite-sign
x/y slopes, where averaging does not merely halve). The two reconstructions
agree to < 1 % of their own span on tilt, defocus and astigmatism, which is
what makes either a check on the other.

**Residual risk.** Lenslets with no measurement now return NaN instead of
being integrated through as zero slope (migration note). The Southwell solve
adds a scipy sparse factorisation per call: negligible at the lenslet counts
this sensor uses (16×16 = 256 unknowns; a 256×256 array would be 65 k
unknowns on a 5-point stencil, ~1 s, documented).

### A4 [P1] -- `eval_image_plane_wfe` and distant objects

**What was wrong.** A finite `object_distance > 0` was required and the bundle
was launched at the object, so an infinite conjugate had to be faked with a
large number -- and the object-side ray-surface quadratic then cancels: the
two roots are separated by only `~2|R|` while `b² ~ 4·object_distance²`, so
the float64 rounding of `b²` lands in the root as `~eps·object_distance²/|R|`.
Confirmed on HEAD via `repro/ANALYSIS/p12`, reproducing the audit's table
exactly (chief z0 +0.0006 / +0.0238 / -5.34 / +589.41 µm at 1e3 / 1e4 / 1e5 /
1e6 m).

**What I changed and why.** `object_distance = float('inf')` (or `None`) now
launches a collimated bundle: one common direction, each ray starting at its
own foot on a plane wavefront normal to that direction, a few aperture widths
before surface 0. Because every launch point is on one wavefront, the
object-side OPL difference across the pupil is carried exactly for any field
angle while the piston -- the only part that depends on how far back the plane
sits -- cancels in the chief-relative convention. All path lengths are
O(aperture), so none of the cancellation arises. The paraxial image distance
falls out of the existing index-threaded Gauss form with `u_pp = inf`. A new
`field_max_rad` carries the off-axis field, which at infinity is a direction
rather than a height.

The large-but-finite case now warns, gated on the mechanism
(`eps·object_distance²/min|R| > λ/10`) rather than a round number.

**How I verified.** Two independent oracles built in the test from a
hand-made collimated trace: the transverse-ray-aberration integral
(`W(ρ) = -(a/R)∫ε dρ`) gives PV 3.6586 waves and the traced longitudinal
spherical aberration (LSA -0.8144 mm) gives W040 3.651 waves, against the
library's 3.5944 -- agreement 1.8 % and 1.6 %, consistent with the ~3 % the
audit quotes for the same comparison in the finite regime. `img_d_m` comes out
at the BFL (47.87519 mm) to 1e-9 relative. The chief ray's OPD is 0 exactly.
The finite and infinite conjugates converge: at 1e3 m the per-ray OPD agrees
with the collimated one to 2 % of the map's span on identical pupil samples.
The warning is silent at 1, 1e2, 1e3 m -- where the reported PV is flat at
3.465-3.468 waves -- and fires at 1e4 m, where PV has already moved to 3.691.

**Exit-vertex census (ORCHESTRATOR F-O3 asked the ANALYSIS auditor to
confirm) -- see §8.1 for the measurement.** `analysis/image_plane_wfe.py`'s
read of `image_rays` is **NOT** part of the exit-vertex bug class, and this is
measured, not argued: the rays genuinely ARE left at the sag (`image_rays.z`
spreads over -242.084 … 0.000 µm on the audit's own singlet, whose last
surface is a curved refractor, R = -50 mm -- the surface list does not end in
an image plane), and yet the reference-sphere OPD computed from the sag state
and from `res.at_exit_vertex()` agree to **1.774e-11 waves over a PV of 3.5944
waves**, eleven decades below the signal. `_ray_sphere_opd` carries the ray's
actual `s2z` into the quadratic
(`c = (s2x-cx)² + (s2y-cy)² + (s2z-cz)² - R²`), so a transfer along the ray
adds `n·t` to the OPL and removes exactly the same `t` from the sphere
intersection. No transfer was added and no local copy exists.

**Residual risk.** The OPL seed of both launches is now explicit (see §8.2):
the point source takes `_make_bundle`'s default `opd_seed='plane'` (zero, the
correct convention for a bundle whose rays all leave one point), and the
collimated launch calls `raytrace.trace.seed_entrance_eikonal`, the library
helper, rather than a local copy.

### A5 [P2] -- `gerchberg_saxton`'s error scale, and `plot_stokes`

Both as stated in the row; see the changelog for the measured numbers. The
GS metric compared an unnormalised DFT against a target rescaled to the source
power, so the error carried a hard-wired `N_pix` and could not reach zero:
3.775380e+02 for an exact solution (97 % of the target energy, flat over 50
iterations) -> 3.26e-29. The JAX twin never rescaled at all and is fixed the
same way, restoring backend parity (`err` ratio N_pix -> 1). The retrieved
phase is unchanged to 1e-9 (both amplitude-replacement steps are scale
invariant), pinned against a reference re-run of the OLD scaling.

`plot_stokes` took `dy` and a real y extent. While in the file I also closed
the P3 half-pixel item (below) and the latent "two independent `'auto'` unit
resolutions on one figure" bug at five sites.

**Residual risk.** None measured; no plotting function is executed by the
test suite, so these are pinned structurally (signature, extent arithmetic)
rather than by rendering.

### A6 [P2] -- performance

Four of the seven items done, three deferred with designs (§6). The two the
WP brief named specifically:

* **`H(z+Δz) = H(z)·H(Δz)`.** 52.92 -> 10.97 ms/plane (4.83x) at N = 1024.
  Gated on z uniformity measured to better than 1e-12 rad of phase on the
  actual `z_values`, so a non-uniform scan keeps the direct `exp` and stays
  bit-identical to a per-plane `angular_spectrum_propagate` (pinned by
  `test_transfer_function_recurrence_engages_only_on_a_uniform_scan`, which
  asserts `==`, not `approx`).
  **Accumulated-rounding bound, derived:** both forms evaluate the same exact
  function, and both round the argument at `|kz z|·eps/2` -- the direct form
  as `fl(kz·z_n)`, the recurrence as `fl(kz·z_0) + n·fl(kz·dz) + n` multiply
  roundings. Their difference is therefore bounded by `2·|kz z|_max·eps`, and
  neither is the more accurate one. Measured `max |H_rec - H_dir|`: 6.2e-12 at
  `|kz z|_max` = 1.09e4 rad and 1.1e-10 at 2.23e5 rad, against predictions of
  4.8e-12 and 9.8e-11. `max | |H| - 1 |` after 21 steps is 1.8e-15, so no
  energy drifts. Worst relative metric drift vs a per-plane reference:
  9.5e-12.
* **`single_plane_metrics` single pass.** 81.44 -> 52.44 ms/plane (1.55x),
  **bit-identical** (`peak_I`, `centroid_x/y`, `d4sigma_x/y` compare `==`
  against `beam_centroid` / `beam_d4sigma`). The shared arithmetic lives in
  one new private helper used by both sides, so this is consolidation rather
  than a second copy.

Together: a 21-plane scan at N = 1024 goes **4248.9 -> 2703.9 ms (202.33 ->
128.76 ms/plane, 1.57x, 36 % of the scan removed)**, measured interleaved
against the pre-fix code restored in-process.

`radial_power_bands` (item 5) now switches to a sort-and-`searchsorted`
construction above a measured crossover of 96 bands. **The audit's claim that
the sorted form is a win at any count does not hold on this machine**: the
masked loop's marginal cost is 0.027 / 0.151 / 0.673 / 4.232 ms per radius at
N = 256 / 512 / 1024 / 2048 against a sort of 2.19 / 10.02 / 54.99 / 274.41 ms,
putting the crossover at 82 / 67 / 82 / 65 bands. Switching unconditionally
would have made the one-radius call -- which is what
`single_plane_metrics(bucket_radius=...)` issues per plane -- 3.9x slower. The
threshold is set above the whole measured band, the small-n path is
bit-identical, and the large-n path is 2.99x at 256 bands and 9.63x at 1024.

**Residual risk.** The one behaviour change is the H-recurrence tolerance,
which forced `test_perf_v4_12_0_through_focus.py`'s agreement bar from 1e-12
to 1e-9. That bar previously described a float64 coincidence (same argument,
same rounding) rather than a property; the new one is derived from the
float64 phase-argument floor, recorded in the file as `HOIST_RTOL` with the
full derivation, and still sits six decades below any real defect in the
transfer function. See §4 for the judgement.

### A7 [P3] -- the remaining rows

All six items done; see the changelog. Two are worth a note here:

* The ghost 50 %-EE radius is now the 50 % point of an area-weighted
  encircled-energy curve, not the median ray. On a uniform disc the median
  ray reads 0.500 R for **every** ring count (it does not converge at all);
  the new estimator reads 0.667 / 0.667 / 0.708 R at 6 / 12 / 24 rings against
  the analytic 0.7071 -- within one ring spacing, which is as fine as a ring
  launch can resolve the energy.
* `caustic_diagnostic`'s complex-eigenvalue clamp is *warned about*, not
  removed. Reporting a Maslov index for a rotating transverse map needs the
  complex-eigenvalue theory the function does not implement; silently
  collapsing to `tr/2` and then counting sign changes of `tr` invents
  caustics. An axisymmetric system never reaches the branch (pinned).

---

## 3. Files touched

**Modified (all inside `lumenairy/analysis/`, my ownership):**

- `lumenairy/analysis/opd.py` -- masked 2-D unwrap kernels, residue
  diagnostic, component anchoring, `unwrap_phase_2d`, `wave_opd_2d` body and
  docstring.
- `lumenairy/analysis/through_focus.py` -- exact-sphere Strehl reference;
  transfer-function recurrence + in-place band-limit; single-pass
  `single_plane_metrics`.
- `lumenairy/analysis/detector.py` -- Southwell / Itoh wavefront
  reconstruction, `reconstruction=` kwarg, docstring.
- `lumenairy/analysis/image_plane_wfe.py` -- infinite-conjugate launch,
  `field_max_rad`, precision warning, explicit `bundle.opd` seeding.
- `lumenairy/analysis/beam_stats.py` -- `_whole_grid_moments` /
  `_centroid_and_d4sigma` shared moment helpers.
- `lumenairy/analysis/phase_retrieval.py` -- GS error scale, NumPy and JAX.
- `lumenairy/analysis/plotting.py` -- `_auto_extent` pixel-edge convention,
  `plot_psf` matching convention, `plot_stokes(dy=...)`, shared y unit.
- `lumenairy/analysis/polychromatic.py` -- `radial_power_bands` crossover.
- `lumenairy/analysis/ghost.py` -- `_ring_area_weights`, `_weighted_median`,
  area-weighted 50 %-EE radius.
- `lumenairy/analysis/aberration.py` -- complex-eigenvalue count + warning.
- `lumenairy/analysis/interferometry.py` -- `dy`, output-range docstring.
- `lumenairy/analysis/zernike.py` -- diagonal-sampled basis cache key.
- `lumenairy/analysis/strehl.py` -- `strehl_phase_integral` tilt warning.
- `lumenairy/analysis/__init__.py` -- 7 exports added.

**Modified tests (analysis test files; two of them pinned the defect):**

- `tests/unit/test_niche_s12_shack_hartmann_reference.py` -- the `0.5 *` in
  `test_reconstruction_integrates_the_uniform_slope`'s oracle.
- `tests/unit/test_v4_16_1_agent_a.py` -- the `2.0 *` in
  `_measured_pitch_from_wf`, and the source-marker string for the v4.16.1
  on-grid-pitch fix (which moved with the integration into
  `_reconstruct_wavefront`; the marker still pins `pitch_actual`, not
  `lenslet_pitch`). Only the Shack-Hartmann assertions were touched; the
  `optimize/core.py` and `io/storage.py` parts of that file are untouched.
- `tests/unit/test_audit_analysis.py` -- `_gs_reference` and
  `test_3a_return_history_matches_reference` re-implemented the old GS
  scaling in their own oracles.
- `tests/unit/test_perf_v4_12_0_through_focus.py` -- `HOIST_RTOL` and its
  derivation (§4).

**New:**

- `tests/unit/test_audit2609_a7_opd_unwrap.py` (18 tests)
- `tests/unit/test_audit2609_a7_strehl_reference.py` (11)
- `tests/unit/test_audit2609_a7_detector_sh.py` (12)
- `tests/unit/test_audit2609_a7_image_plane_wfe.py` (13)
- `tests/unit/test_audit2609_a7_misc.py` (24)
- `docs/audits/.../fixes/WP-A7_REPORT.md`, `WP-A7_CHANGELOG.md`

Every new test was confirmed to FAIL on the pre-fix code by restoring the old
behaviour in-process (the paraxial reference, the halved reconstruction, the
row-then-column unwrap with no anchoring, the un-normalised GS metric, the
symmetric extent, the median-ray radius) and re-running them; the failure
messages carry the audit's own numbers. A4's tests need no such harness: the
pre-fix code raises `ValueError` on `object_distance = inf`.

---

## 4. Tests run

| Command | Result | Duration |
|---|---|---|
| `python -m pytest <48 analysis-related unit files> -q --no-header -p no:cacheprovider` (full list below) | **1717 passed, 8 skipped** | 115 s |
| `python -m pytest tests/unit/test_audit2609_a7_*.py -q` | 78 passed | 26 s |
| `python validation/run_all.py test_analysis test_detector test_image_plane_wfe test_ao test_coherence test_optimize test_features test_field` | **ALL 8 files passed** (14.2 / 1.2 / 2.0 / 1.0 / 2.4 / 4.7 / 1.5 / 2.0 s) | 29 s |
| `repro/orch/verify_analysis.py` | `(1) peak/reference = 1.000`, `(2) max OPD error 0.000 waves, 0.0 %` | 40 s |
| `repro/ANALYSIS/p1,p2,p2b,p2c,p3,p4,p5,p6,p7,p9,p10,p11,p12` | all re-run; see §2 | -- |

The 48 unit files: `test_audit2609_a7_{opd_unwrap,strehl_reference,detector_sh,image_plane_wfe,misc}`,
`test_analysis`, `test_audit_analysis`, `test_audit_w5_analysis`,
`test_audit_w6_analysis_elements`, `test_perf_v4_12_0_{through_focus,zernike_cache,jax_jit}`,
`test_audit_through_focus_jax_{s3_8,x64}`, `test_v5_1_0_agent_{e,g}_split`,
`test_niche_s12_shack_hartmann_reference`, `test_v5_4_{make_shack_hartmann_wfs,retrace_ghost_path,zernike_normalization_weighting}`,
`test_niche_audit_{a1_radial_metrics,a2_encircled_energy_radius}`, `test_ao_dm`,
`test_audit_g06_perf`, `test_v4_15_3_dispatcher_pin_2d_scalar_field`,
`test_niche_audit_w4_input_kind`, `test_v4_15_5_agent_a`, `test_v4_16_1_agent_a`,
`test_niche_audit_{w4c_analysis_immersed,w4d_folded_frames,r1_compute_pupils,w4_immersed_pupils}`,
`test_v5_21_2_subsystem_audits`, `test_niche_s11_sibling_deferred`,
`test_v5_4_7_audit_v5_4_6_gaps`, `test_v5_4_6_wave6_analysis`, `test_audit_optimize`,
`test_verify_perf_fixes_2026_08_10`, `test_audit_v5_24_2_g02`,
`test_niche_audit_p1_odd_n_freq_grid`, `test_v4_14_0_dispatcher_pin_welford_mirror`,
`test_v4_16_0_walker_xp_of_dispatch`, `test_validation_helpers`,
`test_g08_s4_15_cache_hygiene`, `test_niche_audit_w3_ui_deprecation`,
`test_v4_15_4_agent_d`, `test_v4_15_5_agent_b`, `test_v4_16_1_agent_d`,
`test_plot_lens_layout_ray_overlay` (the last four cover the plotting sites
touched by §8.3 and §8.5).

### Tests I changed rather than satisfied

Four, all documented above and in the changelog. Three were pinning the
defect (COMMON.md §6 / audit §15.2 pattern) and are strictly stronger now.
The fourth is a relaxed bar and deserves the explicit judgement:

`test_perf_v4_12_0_through_focus.py`'s `rtol=1e-12` vs a per-plane
`angular_spectrum_propagate` reference was achievable only because both sides
computed `exp(1j·kz·z_n)` from the *same* rounded argument. The recurrence
computes an equally accurate but differently rounded argument, and the gap
between any two such evaluations is `2|kz z|eps` -- 9.9e-11 on that fixture --
which no implementation choice can close. I moved the bar to a derived
`HOIST_RTOL = 1e-9` (one decade above the analytic bound, two above the
measured 9.5e-12 drift, and six below the 1e-3 that any real defect in the
band-limit, sign or shift convention would produce), recorded the whole
derivation in the file, and added a NEW pin that the non-uniform-z path is
still **bit-identical** (`==`) so the relaxation cannot hide a general
regression. Net: one bar loosened by three decades on a fixture-specific
coincidence, one exactness pin added.

### Pre-existing failures found (NOT mine)

Two, both in modules owned by other work packages and both reproducible with
`lumenairy/analysis/` untouched:

1. `tests/unit/test_niche_audit_w3_infra.py::TestA6EstimateAsmMemory::test_est_bounds_measured_first_call_peak[512-complex128]`
   -- `est/measured = 1.382` against a documented 1.06-1.09 band. The failure
   is inside `lumenairy/propagators/asm.py` (modified in this working tree by
   another WP); `analysis/` is not on the path.
2. `tests/unit/test_audit_misc.py::TestAuditFixesV4_12_1_coverage_StopIndexWarn::test_traced_emits_warning_for_stop_index_2`
   -- `lumenairy/elements/_lens_real.py:2406` now RAISES `ValueError` on an
   out-of-range `stop_index` where the test expects a warning. Another WP
   converted that warning into a refusal; the test needs updating on their
   side.

My judgement: neither is related to WP-A7. Both are flagged for the
orchestrator.

---

## 5. Requested changes outside my ownership

1. **`lumenairy/__init__.py` -- export `unwrap_phase_2d`.**
   Add `unwrap_phase_2d` to the `from .analysis import (...)` block (next to
   `wave_opd_2d`, line 67) and to `__all__` (next to `'wave_opd_2d'`, line
   1562). It is the new public masked 2-D unwrap kernel and is already in
   `lumenairy.analysis.__all__`. Without this it is reachable as
   `lumenairy.analysis.unwrap_phase_2d` only. Not a correctness issue; the
   root package is on the do-not-edit list.

2. **`lumenairy/__init__.py` -- optionally export the cache accessors.**
   `clear_meshgrid_cache`, `meshgrid_cache_bytes` and
   `zernike_basis_cache_bytes` were exported *nowhere* (A7); I added them to
   `lumenairy.analysis`. Their siblings (`clear_zernike_basis_cache`) are on
   the root, so the root probably wants these three too.

3. **`lumenairy/raytrace/__init__.py` -- re-export `seed_entrance_eikonal`.**
   `exit_vertex_transfer` is re-exported from the package; its R2 sibling is
   not, so `analysis/image_plane_wfe.py` has to reach into
   `..raytrace.trace`. One line, cosmetic. Owner: the `raytrace/` WP.
   *(The larger `_make_bundle` seeding concern I had raised is resolved:
   WP-A1 landed `opd_seed='plane'|'eikonal'` with `'plane'` -- opd = 0 -- as
   the default, which is the correct convention for a point-source launch,
   plus the functional `seed_entrance_eikonal` that carries the `N*z` term
   for off-`z=0` launches. Both of this module's launches now use the right
   one of the two.)*

4. **`tests/unit/test_audit_misc.py::...StopIndexWarn`** -- see §4, owned by
   the lens WP.

No source file outside `lumenairy/analysis/` was modified by me.

---

## 6. Deferred items (with designs)

All three are A6 performance rows; none is a correctness defect, and each is
gated behind a measurement I did not have time to complete safely.

1. **`compute_psf` materialises ~4 padded complex grids (1074 MB at
   oversample 4).** `psf_mtf_otf.py:161-172`.
   *Design:* replace `fftshift(fft2(ifftshift(a)))` with the chessboard
   identity `chess * fft2(chess * a)` where `chess = (-1)^(i+j)` -- separable
   as two 1-D sign vectors, applicable in place -- removing both shift
   copies, ~536 MB at oversample 4. Bit-identity must be demonstrated: the
   identity is exact for even N and differs by a cyclic shift for odd N, so
   the odd-N path needs either the explicit shifts or a derived tolerance.
   The larger win the audit names (`propagators/mft.py`'s
   `fraunhofer_propagate_mft`, Soummer 2007 -- arbitrary focal-plane pitch, no
   padding, ~0 extra memory) is an algorithm swap that changes the PSF grid
   contract and needs its own gated pass. *Effort:* 0.5 day for the
   chessboard + the odd-N proof; 2-3 days for MFT with a compatibility path.

2. **`encircled_energy_curve` / `encircled_energy_radius` re-run a full
   argsort each (499 + 487 ms at N = 2048 measured here).**
   `psf_mtf_otf.py:427-429`. A single call of either is already optimal --
   one sort -- so the waste is only in a caller doing BOTH. *Design:* the
   sound fix is API sharing, not a cache: expose
   `encircled_energy_profile(E, dx, dy=None, centroid=None) -> (r_sorted,
   p_cum, r_max)` and accept it as `profile=` on both functions. A cache keyed
   on array CONTENT would have to hash ~67 MB per call at N = 2048 (~60 ms),
   i.e. a 12 % tax on the far more common single-call path, and a key on
   anything less than the content violates audit §15.5. *Effort:* 0.5 day
   including docs and a "the radius is the exact inverse of the curve" pin.

3. **`_zernike_radial`'s factorial sum (one full-grid power per term; basis
   build 600 ms for 21 modes at N = 1024) and the DM influence-function cache
   (537 MB eager for a 16×16 DM on 512², measured 549.5 MB).**
   `zernike.py:88-95`, `ao.py:167-185`.
   *Design (Zernike):* Kintner (*Opt. Acta* **23** (1976) 679) or Prata-Rusch
   (*Appl. Opt.* **28** (1989) 749) recurrence in `n` at fixed `m`, stable to
   n ≈ 100 and one multiply-add per term instead of a `rho ** (n-2s)`; the
   audit confirmed the present code is *accurate* (Gram max off-diagonal
   1.5e-3, pure discretisation), so the bar is bit-near-identity against the
   factorial sum at every (n, m) in the shipped tables, which is the work.
   *Design (DM):* `ao.fit_phase` already builds the influence functions in
   bands; hoist that into a `_banded_IF_apply` and make `cache_basis` default
   to False above a byte budget, warning once. *Effort:* 1 day each.

Also deliberately NOT done, and why: `mtf_radial(..., wavelength, f)`,
`shack_hartmann(detector_pixels_per_lenslet=...)` and
`ghost_analysis(n_rays=...)` remain inert parameters (the audit lists them as
an observation, not a finding; `detector_pixels_per_lenslet` is documented as
RESERVED, validated, and wiring it would move every SH slope and the AO
closed-loop pins -- its own gated pass). `analysis/core.py` is still a
re-export shim and `through_focus.py` still mixes the scan with ~1100 lines of
tolerancing; both are the audit's "code organization observations", not
findings, and splitting them mid-remediation would collide with other WPs.

---

## 7. Changelog text

`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A7_CHANGELOG.md`

---

## 8. Coordinator follow-ups

Five items were routed to WP-A7 while it was in flight. All five are closed.

### 8.1 `TraceResult.at_exit_vertex` at `analysis/image_plane_wfe.py:508-511`

**Closed: the site is not in the exit-vertex class, and this is measured.**
The fixture the audit itself used (biconvex N-BK7, R = ±50 mm) does NOT end in
an image plane -- its last surface is a curved refractor, R = -50 mm -- and
`trace()` does leave the rays on its sag: `image_rays.z` spreads over
**-242.084 … 0.000 µm** across the pupil. The question is whether the OPD this
module derives from that state is wrong, and it is not:

```
reference-sphere OPD from image_rays  vs  from res.at_exit_vertex()
    max |difference| = 1.774e-11 waves   over a PV of 3.5944 waves
```

`_ray_sphere_opd` carries the ray's actual `s2z` into the quadratic, so a
signed transfer `t = -z/N` adds `n·t` to the OPL and removes exactly the same
`t` from the sphere intersection -- the two cancel identically. Adding the
transfer would be a no-op that costs a bundle copy, so it was not added. No
hand-rolled transfer exists in `analysis/` to delete (`grep -rn
"image_rays" lumenairy/analysis/` returns this one site).

### 8.2 Entrance eikonal (R2) at the `analysis/` launchers

**Closed: none of the six candidate sites needs it, measured not asserted.**
The eikonal seed only matters where OPL DIFFERENCES across a tilted bundle are
read as a wavefront error. `analysis/field.py` never reads `.opd` at all
(`grep` over the whole file returns nothing), `analysis/aberration.py`'s
caustic fan uses only ray POSITIONS to build the transverse Jacobian, and
`analysis/plotting.py`'s layout bundle is drawn, not integrated. Demonstrated
by seeding `seed_entrance_eikonal` on every one of those launchers in-process
and re-running them off-axis (3 deg, singlet):

| consumer | output | identical with the eikonal seeded? |
|---|---|---|
| `distortion_vs_field` | `distortion_pct` | **yes** (`array_equal`) |
| `distortion_grid` | `actual_x`, `actual_y` | **yes** |
| `field_aberration_sweep` | `sagittal_focus_shift`, `tangential_focus_shift`, `astigmatism` | **yes** |
| `caustic_diagnostic` | `det_J`, `maslov_index`, `caustic_z` | **yes** (0 -> 0, 0 -> 0) |

The two sites in `analysis/` that DO read exit OPL are the
`image_plane_wfe.py` launches, and they are handled: the point source keeps
`opd_seed='plane'` (zero -- every ray leaves one point, so the incident
wavefront is a sphere of zero radius and a plane-wave eikonal would add a
spurious `L·src_x + M·src_y` across the pupil, ~8 waves at 1 mm of object-side
field height), and the collimated launch calls the library's
`seed_entrance_eikonal`, which carries the `N*z` term the `_make_bundle` seed
omits for an off-`z=0` launch. No local copy of either.

### 8.3 `plot_opd_fan`'s PV/RMS labels after the R1 `opd_fan_data` fix

**Closed: nothing in `analysis/plotting.py` was calibrated on the old
values.** `_render_opd_fan_panel` (shared by `plot_opd_fan` and
`plot_opd_summary`) has no hard-coded axis limits, no diffraction-limit
reference line and no numeric threshold: it plots what it is given, autoscales,
and annotates `PV`/`RMS` computed from the same array via `_opd_pv_rms_1d`
(`max - min`, and the deviation about the in-aperture mean). The only fixed
element is `axhline(0.0)`, which is the OPD zero and is convention-independent.
So the labels simply report the new, correct numbers. The four test files that
exercise these functions (`test_v4_15_4_agent_d`, `test_v4_15_5_agent_b`,
`test_v4_16_1_agent_d`, `test_plot_lens_layout_ray_overlay`) assert structure
-- tuple shape, axis count, line count, finite-sample count -- not magnitudes,
and all 70 pass.

### 8.4 `_check_2d_scalar_field` census (`test_niche_audit_w4_input_kind.py`)

**Closed, and resolved without touching the shared census test.** The
intermediate state the coordinator saw was real: the first version of the
single-pass metrics helper carried its own guard, which took the census from
69 to 70. That was the wrong place for it -- the helper is PRIVATE, and the
census counts public entry points -- so the guard was removed from it and
`single_plane_metrics` now routes anything that is not a plain 2-D NumPy field
(an MCF, a 3-D ensemble, a CuPy/JAX array) to the unchanged
`beam_centroid` + `beam_d4sigma` pair, which carries the canonical guard. The
census is back at **69**, `_WIRED_SITES` is untouched, the rejection message
for a bad field is unchanged, and `tests/unit/test_niche_audit_w4_input_kind.py`
passes (239 tests). Verified after the fact:
`grep -rn "_check_2d_scalar_field(" lumenairy/ | grep -v "def " | wc -l` = 69.

### 8.5 Stale `LA1509-C 0.9883` census comment in `analysis/plotting.py`

**Closed: re-measured.** The comment records
`|image_z| / max(track, aperture, |efl|)` over every in-tree prescription and
is the derivation for `_LAYOUT_FOCUS_RUNAWAY_RATIO = 10.0`. Re-measured on the
corrected catalogue (LA1509-C now R1 = 51.5 mm, EFL 99.652 mm rather than the
200 mm-lens radius it carried):

| row | recorded | re-measured 2026-09-12 |
|---|---|---|
| LA1050-C | 0.9733 | 0.9729 |
| **LA1509-C** | **0.9883** | **0.9762** |
| LA1301-C | 0.9912 | 0.9910 |
| AC254-050-C | 0.8677 | 0.8664 |
| AC254-100-C | 0.9511 | 0.9508 |
| AC254-200-C | 0.9806 | 0.9805 |

The three synthetic rows (f/1 biconvex, f/40 plano-convex, f/100) keep their
2026-08-22 values because the exact fixture parameters are not recorded in the
comment; a reconstruction measures 0.8910 / 0.9980 / 0.9992, the same
envelope. Max over the census **0.9992**, so the comment's claim -- "every
focusing system sits at or below 1.0" -- still holds and the bar at 10 is
untouched. The comment now carries both measurement dates and says which rows
were re-measured, so the next catalogue correction cannot make it stale
silently.
