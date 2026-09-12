# WP-A2 changelog text — `apply_real_lens` analytic model

Findings L1–L20 and E4 of `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11.md`
(§2.1, §5, §14 V1–V2).  Files: `lumenairy/elements/_lens_real.py`,
`lumenairy/elements/lenses.py`.

---

### Fixed -- apply_real_lens: `seidel_correction=True` no longer injects ~90 µm·ρ² of spurious defocus (L1, V1)

Three compounding defects made the option 8×–3600× WORSE than leaving it off,
on the very prescriptions its docstring recommended it for.  All three are
fixed in `_lens_real.py`'s Seidel block:

* the 41-ray fan's OPL was read at the LAST SURFACE'S SAG, not on the exit
  vertex plane.  Exactly 0 nm of error on a plano rear face (every fixture in
  the suite) and −17.05 µm at h = 3.6 mm on a cemented doublet's R = −291 mm
  rear.  Now read through `TraceResult.at_exit_vertex()` (WP-A1);
* the model reference was `Σ (n2−n1)·sag_i(h)` — every surface at the same
  entrance height with no propagation — so the residual it fitted was
  dominated by the in-glass obliquity `n·t·θ²/2` that the split-step's own ASM
  legs already carry exactly (337 nm measured against a 341 nm prediction on a
  plano-convex whose true residual is 0.85 nm).  Replaced by
  `_split_step_fan_opl`, a thin-screen ray model of what the split step
  actually does: screen deflection `−∇[(n2−n1)·sag]` at each surface's vertex
  plane, then a straight ray through each glass gap accumulating `n·t/cosθ`;
* the fit basis started at ρ², i.e. it CONTAINED defocus, so every reference
  mismatch was absorbed as focus and imprinted (ρ² term 420 nm at the rim on a
  plano-convex, −5.5 µm on a doublet).  The fit now starts at ρ⁴, and the 5 nm
  gate is scored on the fitted ρ⁴+ part rather than the raw residual.

The ray OPL is also carried from the ray's landing point to the MODEL ray's
landing point at the exit momentum `p = n_exit·L` (they differ by up to
12.7 µm on an 8 mm doublet), because the screen multiplies the field at a fixed
exit coordinate.

Measured exit-plane OPD rms against an independent Newton-intersection +
vector-Snell ray oracle (`repro/RL-CORE/p6_seidel.py`), correction OFF / ON:

| prescription | OFF | ON before | ON after | after/before |
|---|---:|---:|---:|---:|
| plano-convex 4 mm | 0.848 nm | 88.87 nm (105× worse) | 0.848 nm (gate SKIPS) | 105× |
| biconvex R=±60 4 mm | 1.829 | 6599.4 (3608× worse) | 1.829 (gate SKIPS) | 3608× |
| biconvex R=±30 4 mm | 14.758 | 12128.7 (822× worse) | 0.141 (**104× better**) | 86 000× |
| cemented doublet 4 mm | 10.857 | 311.6 (28.7× worse) | 1.841 (**5.9× better**) | 169× |
| cemented doublet 8 mm | 173.47 | 1430.0 (8.2× worse) | 1.126 (**154× better**) | 1270× |

Through-focus, unwrap-free (`repro/RL-CORE/p6d_psf.py`,
`repro/orch/seidel_focus_shift.py`): the plano-convex is now bit-identical with
the flag on and off (peak 41 849.17 at z = 95.0657 mm both ways; before, ON lost
4.0 % of the peak and moved focus +2.0 %); the 4 mm doublet's peak goes
111 324.86 → 114 553.08 (before: −26.8 %) with focus unmoved; and the
AC254-100-like doublet's on-axis peak stays at z = 114.0 mm against a 114.85 mm
paraxial BFL (before it moved to z = 62.0 mm).

FOLLOW-UP (VERIFY-A2, 2026-09-12).  Two further changes to the same block,
both of which only improve the numbers above.  (1) The 41-ray fan is launched
across the FULL clear aperture (`±0.999·r_pupil`, was `±0.9`).  (2) The fitted
screen is now HELD CONSTANT beyond the largest radius the fan lands at instead
of being extrapolated: the transverse walk through the element is inward, so
the rays land short of the pupil edge (measured 0.897 of it on an f/2 singlet,
0.783 on the 8 mm doublet) and continuing a ρ⁴+ρ⁶ polynomial past its own data
put three waves of nothing on a third of the pupil area.  Re-measured: the
8 mm doublet 173.6 → **1.053 nm (165×)**, the f/2 singlet 402.6 → **1.504 nm
(268×)**, the 4 mm doublet's through-focus peak −0.24 % with the focus unmoved
(+0.146 %).  The plano-convex gate still SKIPS bit-identically.

KNOWN SCOPE LIMIT, measured and now in the docstring: the radial screen assumes
the exit field FILLS the pupil the fit is normalised to.  On a fast, thick
element it does not — on an f/2 singlet at converged sampling |E| falls 20×
between ρ = 0.85 and ρ = 0.93, and the model-vs-wave difference over the full
pupil is 3.7 µm against 1.5 nm over ρ ≤ 0.85.  Whatever the screen puts on that
annulus scatters its ~5 % of the energy out of the core: the exit wavefront
improves 268× while the focal PEAK drops 37 %.  Judge the option on a filled
pupil, and use `apply_real_lens_traced` (no radial screen) otherwise.  Also in
the docstring: any exit-OPD measurement on an apertured prescription needs
`dx ≈ 1.45·aperture/2048` or finer — coarser and the aperture edge aliases
through the in-glass ASM and reads hundreds of nm that are entirely the grid
(measured on a meniscus: 558 nm at N = 512, 0.03 nm at N = 2048).

### Fixed -- apply_real_lens: `slant_correction=True` used normal-referenced cosines and was 2.94× WORSE than the screen it corrects (L12)

The screen applied `(n2·cosθt − n1·cosθi)·sag`, with both cosines referenced to
the facet NORMAL.  The module's own axial-translation identity (equation (3) of
the SCREEN OBLIQUITY derivation) needs them referenced to the **Z axis** — a
facet is displaced along z, not along its own normal — which for the collimated
input this screen assumes is `(n2·cos(θi − θt) − n1)·sag`.  Expanded to O(θ²)
with n1 = 1: exact `(n2−1) − θ²(n2−1)²/(2n2)`, shipped
`(n2−1) + θ²(n2−1)/(2n2)` — opposite SIGN and 1/(n2−1) = 1.94× too large, so its
total error was n2/(n2−1) = 2.94× the paraxial screen's.  Fixed in both copies
(whole-grid and the banded `_slant_narrow_chunk` arm).

Measured exit OPD rms against an independent ray oracle
(`repro/RL-CORE/p5_slant.py`), default / slant:

| fixture | default | slant before | slant after |
|---|---:|---:|---:|
| plano-convex curved-first | 0.848 nm | 2.487 (0.34×) | **0.037 (23.2×)** |
| plano-convex flat-first | 1.177 | 2.282 (0.52×) | **0.017 (68.9×)** |
| parabolic asphere (k=−1, A4, A6) | 7.764 | 22.846 (0.34×) | **0.053 (147×)** |
| biconvex R=±60 | 1.829 | 1.133 (1.61×) | 1.641 (1.11×) |
| biconvex R=±25 | 25.591 | 17.588 (1.46×) | 23.907 (1.07×) |
| meniscus R=20/25 | 870.79 | 875.67 (0.99×) | 891.54 (0.98×) |

Against the exact one-facet eikonal on a single N-BK7 face the corrected screen
is 940× better than paraxial (0.0041 vs 3.83 nm rms).

**BEHAVIOUR CHANGE / migration.**  The flag's numbers move for every caller.
The biconvex and meniscus rows above show why the old form sometimes scored
better: its wrong-signed, oversized error partly cancelled a DIFFERENT error —
the identity is exact only for a COLLIMATED input, and on a thick symmetric
element the bundle at the second surface is already converging.  Measured on the
f/5 hammer fixture (R = ±51.68 mm, t = 5 mm, image r2m against a 65 µm
dual-oracle truth), at converged sampling dx ≤ 3 µm: paraxial 40.55 µm, old
slant 76.70 µm (overshoot), new slant 43.06 µm (undershoot).  If you turned
`slant_correction` on for a fast SYMMETRIC element, move to `carrier=`
(`screen_obliquity`), `surface_model='displaced'` / `'tangent_facet'`, or
`apply_real_lens_traced`: all three carry the true local ray angle, which is
what that residual is.  `slant_correction` now also refuses to combine with
`seidel_correction` (see below).

### Fixed -- apply_real_lens: `fresnel=True` applied |t|² instead of the power transmittance (L13)

`T_eff` was `0.5(|t_s|² + |t_p|²)` — the AMPLITUDE coefficients — where this
library's `Σ|E|²·dx·dy` IS the power (the ASM legs are Parseval-unitary and the
in-glass propagation adds no impedance factor).  Crossing an index step needs
the POWER transmittance `T = (n2 cosθt)/(n1 cosθi)·|t|²`.  Fixed in both copies.
Measured (`repro/RL-CORE/p3_fresnel_energy.py`), transmitted power fraction:

| configuration | before | after | correct |
|---|---:|---:|---:|
| single flat AIR→N-BK7 face | 0.632344 | **0.958057** | 0.958057 |
| prescription ENDING in glass | 0.632344 | **0.958057** | 0.958057 |
| bare cemented N-BK7→N-SF11 | 0.846390 | **0.993599** | 0.993599 |
| AIR→N-BK7→AIR plate | 0.917873 | 0.917873 | 0.917873 |

**BEHAVIOUR CHANGE / migration.**  An element that starts and ends in air is
UNCHANGED at normal incidence (the `n2/n1` factors telescope to
`n_last/n_first = 1`), which is why every plate fixture passed before.  Anything
that ends in glass, any bare cemented interface, and the `cosθt/cosθi` half at
finite NA all move — a single air-glass face by +51 %.  The shipped
per-interface loss is now the ~4.2 % the docstring has always claimed.

### Fixed -- apply_real_lens: `surface_frame=True` deleted the surface tilt (L2, F-O2, V2)

The rigid-body branch evaluated the sag at the rotated transverse FOOTPRINT and
discarded the rotated surface's own field-frame HEIGHT — which is where the tilt
lives.  A rotation re-expresses the ramp; it does not delete it.  The screen now
imprints `(n2−n1)·z_f` with
`z_f = R_zx·x_s + R_zy·y_s + R_zz·g(x_s, y_s)` and
`R_z· = (−cos θx sin θy, sin θx, cos θx cos θy)`.

Measured: a flat N-BK7 face tilted 5 mrad now deviates the beam by 2.537 mrad
against the thin-prism value (n−1)θ = 2.538 mrad, on both axes
(`repro/orch/surface_frame_tilt_check.py`); before, `surface_frame=True`
deviated **0.000 mrad** and a tilted flat face was a byte-identical no-op.
Against the closed-form rigid-body rotated sphere (R = 50 mm over ±2 mm), the
field-frame height error drops from 2.002 / 10.011 / 40.077 µm (1.63 / 8.15 /
32.6 waves) to 1.7 / 10.3 / 80.1 nm (0.0014 / 0.0084 / 0.065 waves) at 1 / 5 /
20 mrad — 1180× / 970× / 500×.  `decenter` is unaffected and remains
byte-identical between the two branches.

### Changed -- apply_real_lens: `surface_frame=True` now reads `tilt` the same way every other consumer does (L19)

With the ramp restored, the two branches had to agree about which axis a `tilt`
component acts about.  The surface-frame branch's rotation angles are now taken
from the library-wide reading of the key — `tilt = (t0, t1)` IS the linear sag
ramp `t0·x + t1·y`, i.e. `θx = t1`, `θy = −t0` — which is what the field-frame
branch, `_disp_surface_z_grad`, `raytrace`'s `field_tilt` and the lumenairy-free
geometric spot oracle all use.  Flipping `surface_frame` no longer re-points the
element.  This changes `surface_frame=True` results for a tilted surface (the
footprint rotation swaps which component scales which axis), on top of the L2
change above; `surface_frame=False` is untouched.

Note for the record: that shared reading is `(−θy, +θx)` relative to
CONVENTIONS §7's right-hand-rotation wording for `tilt=(theta_x, theta_y,
theta_z)`.  Re-spelling it to §7 would have to change `_lens_real`,
`raytrace/surface.py` and `validation/oracles/geom_spot_decenter_oracle.py`
together, plus every caller's `tilt` value; it is recorded as a cross-module
follow-up rather than taken unilaterally here.

### Fixed -- apply_real_lens: the `displaced` remaps discarded the input field's phase (L3)

`_apply_displaced_remap` and `_apply_displaced_remap_2d` sampled only
`np.abs(E_in)` and rebuilt the exit phase from the `conjugate` congruence
(default: collimated), so any phase the caller's field carried was thrown away
with no warning — and because `displaced_obliquity='auto'` routes any
decentered / tilted / `sag_callable` element to the 2-D remap, that was the
DEFAULT path for an asymmetric element.  Both now demodulate `E_in` by the
traced congruence (`_residual_input_field`), transport the complex residual
along the same rays, and re-apply it; `|F| = |E_in|`, so this replaces the old
real resample rather than adding one.

Measured (`repro/RL-MODELS/t5_phase_discard.py`, `t6_phase_discard2.py`): a flat
and a 35.4-wave-defocused input now separate by 1.974 (1-D remap) and 1.961
(2-D remap) of peak, against 1.978 for the thin reference and 1.977 for the
pointwise screen — they read **4.7e-16** before.  A 150 mm diverging source
through a decentered N-BK7 singlet with the DEFAULT `conjugate=None` now focuses
at 25.000 mm, matching the thin model and `conjugate='auto'`; before it focused
at 21.000 mm, the COLLIMATED focus, a 4 mm / 19 % error.  The `conjugate`
docstring, which asserted the opposite, is corrected.

### Fixed -- apply_real_lens: an out-of-range `stop_index` silently removed ALL aperture clipping (L14)

The entrance aperture ran only when `stop_index is None` and the per-surface
stop matched `i == stop_index`, so an out-of-range value matched nothing and
disabled both.  Measured transmitted power with a 3 mm stop on a 5.12 mm grid:
0.269 for `None`/0, 0.279 for 1, and **1.00000 for 2, 5, −1 and −2** with zero
warnings — 3.7× the energy.  `stop_index` is now normalised (`-1` means the last
surface, as everywhere else in Python: it transmits exactly what
`len(surfaces)-1` transmits) and anything outside `[0, len(surfaces))`, or not
an integer, raises a `ValueError` with the §2 prefix.  `prepare_real_lens` reads
the key through the same helper, so the two entry points diagnose a malformed
value identically before the former refuses a mid-train stop as it always did.

### Fixed -- apply_real_lens: `screen_obliquity` double-counted n1 for the `'auto'` and ndarray carriers (L4)

`_screen_obliquity_angle_field` multiplied by `n1` for every carrier vocabulary.
That is right for the two GEOMETRIC congruences (a `TiltedCarrier`, whose
`(L, M)` are unit-ray direction cosines, and a signed scalar conjugate, whose
`grad W = sin α`), and wrong for `'auto'` (which fits
`angle(E[:, 1:]·conj(E[:, :-1]))/(k0 dx)` — the field's phase is `k0·S` with `S`
the OPTICAL path, so the reading IS `p_x`) and for an explicit wavefront ndarray
(documented as "reference phase = k0·W").  Measured on an exact plane wave in
N-BK7 at 50 mrad, true `q = n1 sin θ = 0.075808`: `'auto'` and ndarray now read
0.075808 (before 0.114986, ×1.5168), `TiltedCarrier` still 0.075808.  On an
immersed R = 25 mm surface the inflated `q` made the "corrected" screen carry
0.0971 waves at 100 mrad — worse than the uncorrected 0.0784.

### Fixed -- apply_real_lens: two `displaced` caches returned stale results (L5, L6)

* `_DISPLACED_COS_GRID_CACHE` keyed a freeform `sag_callable` by object
  IDENTITY, which does not imply value equality for a MUTABLE callable — and
  the cache exists for exactly the design-iteration workload that mutates one.
  Measured: 164 % of peak amplitude of error (0.987 of peak in this build's
  fixture) from a stale grid.  The key now carries a VALUE fingerprint as well:
  the callable is probed on a fixed 24-point stencil spanning the traced extent
  and the float64 bytes go into the key, so a state change is a MISS.  A
  callable that cannot be probed makes the entry uncacheable rather than
  identity-keyed.
* `_DISPLACED_LUT_CACHE` (on by default) keyed glasses by registry NAME.
  `GLASS_REGISTRY` is a documented mutable extension point, so re-pointing an
  entry left the key unchanged and the cosines stale — measured 1.3 % of peak.
  Both keys now carry the RESOLVED index (`get_glass_index`, already memoised),
  which is strictly more correct and removes the name-aliasing hazard.

### Fixed -- apply_real_lens: the `displaced` ray maps referenced the exit leg through air (L7)

`_build_displaced_ray_map` and its 2-D twin walked the ray from the last
surface's sag back to the exit vertex plane with `opl += 1.0 * t_f`, but that
leg is travelled in `surfaces[-1]['glass_after']`.  Measured on an immersed-exit
singlet against a closed-form trace: 1.03e-5 m = **16.3 waves** of error before,
8.1e-19 m after.

### Fixed -- apply_real_lens: `tangent_facet_remap` refused ordinary padded grids (L8)

The fold determinant and the pull-back residual were reduced over EVERY pixel of
the grid, including the padding outside the clear aperture where the field is
identically zero and the sag grows without bound — so padding, which a
converging beam needs, CAUSED refusals.  Both reductions now score the
illuminated support (`|E| > 1e-6·max|E|`, which is 1e-12 of peak intensity and
is exactly the aperture for a hard-apertured pupil); the whole-grid minimum is
kept in the message as a diagnostic.  The pull-back loop also stops as soon as
the residual stops contracting (so real divergence is reported in 2–3 sweeps
instead of 64) and its ceiling rises from 64 to 256 sweeps, because the cap
exists to stop divergence, not to truncate a slow contraction.

Measured (`repro/RL-MODELS/t9_fold.py`), a 2 mm pupil at five pad/pitch
combinations: **5 of 5 accepted**, against 3 of 5 before (the 8.2× pads were
declined at `min det = −0.87` on dark corner pixels while the illuminated pupil
sat at 0.9986, three orders inside the 1e-4 bar).  A genuine fold inside the
support is still refused.

### Fixed -- apply_real_lens: `form_error` had no shape or dtype validation (L15)

An `(N,)` map was silently BROADCAST across every row — one row of a figure map
replicated N times, applied with no diagnostic; a mismatched 2-D map died with a
raw numpy broadcast error and an `(N, N, 1)` one survived the lens and died
inside the ASM.  All three now raise with the `apply_real_lens: surfaces[i]
['form_error'] ...` prefix, as does a complex map.  Value and sign are unchanged
(Δφ = −0.511441 rad for a uniform 100 nm map, exactly `−k0(n−1)·100 nm`).  The
docstring now states that it is a FIELD-frame map: it is added after the
surface-frame coordinates are consumed, so it is neither shifted by `decenter`
nor rotated by `surface_frame`.

### Fixed -- apply_real_lens: `absorption=True` attenuated by the AXIAL gap thickness (L19)

The glass a pixel crosses between surfaces i and i+1 is
`t_i + sag_{i+1} − sag_i`, not `t_i`.  Factorising that exponential puts
`exp(+k0 κ sag_i)` on surface i and `exp(−k0 κ sag_{i+1})` on surface i+1, so
each surface applies ONE local factor with the sag it already has — no second
sag grid, no halo, and the product over the element telescopes back to the true
local path.  Measured on a 6 mm-centre / 5.38 mm-edge N-BK7-like biconvex
against a Beer-Lambert ray-column oracle: max deviation 4.2e-4 against 1.52e-2
for the axial-only factor, **36×**, recovering 97 % of the apodisation depth.  A
flat plate and "no attenuation after the last surface" are unchanged exactly.

### Fixed -- elements: `surface_sag_general` dropped the whole aspheric polynomial for a non-C-contiguous `h_sq` (E4)

The numba kernel accumulates through `sag.ravel()`, which is a VIEW only when
`sag` is C-contiguous.  `sag` inherits its memory order from `h_sq`, so for an
F-ordered, transposed or strided `h_sq` the kernel added the polynomial into a
temporary copy that was then discarded — exactly zero aspheric contribution,
silently (measured 9.41e-6 m = 100 % of the term; 3.84e-5 m on this build's
fixture).  The buffer is made contiguous, accumulated into, and copied back.

### Fixed -- elements: `surface_sag_general(R=0)` returned an all-NaN grid behind anonymous numpy warnings

A zero radius is not a surface (the conic sag divides by `R**2` and then by
`R`).  It now raises a §2-prefixed `ValueError` naming the two spellings of a
FLAT surface (`np.inf`, `None`), both of which still return zeros.

### Performance -- elements: `surface_sag_general` rewritten in place; the phase screen built with cos/sin and applied in place; flat faces skipped

All three are BIT-IDENTICAL — asserted, not assumed: the banded/whole-grid
byte-identity matrix (5 prescriptions × 7 band sizes + the slant arm × 3),
`prepare_real_lens` vs `apply_real_lens` at complex128 and complex64, and a
direct comparison against the previous expressions all hold at `max|d| = 0.0`.

| change | before | after | gain |
|---|---|---|---|
| `surface_sag_general` conic chain, one grid + one bool instead of 7 fresh grids (N = 2048, medians of 5 interleaved runs) | 176.4 ms / 4.13 float64 grids | **56.5 ms / 1.13 grids** | 3.12× / 3.67× |
| phase screen: `cos`/`sin` into a preallocated complex array, then `E *= ph` (N = 2048, complex128, float64 geometry) | 253.1 ms / 3.00 complex128 grids | **172.2 ms / 2.50 grids** | 1.47× / 1.20× |
| flat-face early-out: screens BUILT per call | plano-convex 2, window 2 | **plano-convex 1, window 0** | one full complex `exp` + multiply per plano face |
| entrance aperture, TIR mask, clear aperture and stop mask as in-place boolean assignments instead of fresh `xp.where` grids | 4 full complex grids per element | 0 | — |
| end to end, the audit's 3-surface fixture, N = 2048 whole grid | 4.01 s / 16.13 float64 grids | **1.57 s / 14.00 grids** | **2.55× / 1.15×** |

The NaN-safe sense of the masks is preserved (`not (sin2_tt < 1.0)`, so a NaN
still lands on the zeroed side) and `complex64` input still returns `complex64`.

The cos/sin screen is restricted to float64 GEOMETRY (`set_lens_sag_dtype`'s
default).  At float32 the two forms are not the same arithmetic — numpy's
`complex64` exponential carries more than float32 through its own sine/cosine
while `np.cos(float32_arg, out=<float32 view>)` does not — and they diverge by
~8.4e-08 of unit modulus, which is enough to break `PreparedAnalyticLens`'s
byte-identity with `apply_real_lens` under `set_lens_sag_dtype(np.float32)`
(measured 1.2e-07 of peak field before the restriction).  `_screen_exp` falls
back to `xp.exp` for any narrower dtype, and the bit-identity test covers both.

### Performance -- apply_real_lens: the four geometric Newton loops stop at their fixed point (L9)

`_build_displaced_cos_luts`, `_build_displaced_cos_grid` and the two ray-map
builders each ran a FIXED 24 intersection sweeps, and each sweep costs three
`_surface_sag_general` evaluations over the whole fan.  They now break when `t`
reaches a BITWISE fixed point — not a tolerance: once another sweep provably
cannot change a bit, the remaining ones are pure cost, so the output is
unchanged by construction.

Re-measured (VERIFY-A2, 2026-09-12) on a spherical singlet and a
conic+aspheric one: the bitwise fixed point is reached after **13** sweeps in
both ray-map builders and 3 in `_build_displaced_cos_luts`, so
`_surface_sag_general` calls drop **150 → 84 (1.79x)** per two-surface build,
not the "2 sweeps / ~10x" an earlier revision of this entry (and RL-MODELS)
quoted — the residual is zero to a TOLERANCE after ~2 sweeps, but `t` keeps
moving in the last bits for a further ~11.  The property the change rests on is
unaffected and was verified directly: forcing the old 24 sweeps back (by making
the fixed-point test never fire) changes **no bit** of any of the four
builders' outputs across 2 prescriptions x 4 fan radii x 3 grid sizes
(`tests/unit/test_audit2609_a2_verify_lens_analytic.py::
TestVerifyL9NewtonEarlyExitIsBitwise`).

### Changed -- apply_real_lens: two `displaced` resolutions now scale with the grid (L9)

* the pointwise obliquity cos-grid spread a fixed 384 coarse samples over the
  FIELD extent, so its accuracy fell LINEARLY with pad factor (measured 24×
  worse over a 16× pad) — a numerical artefact of padding rather than of
  anything physical.  The count now scales with the pad factor so the pitch
  INSIDE the traced aperture is fixed at its unpadded value; measured error
  inside the pupil against a 4096-sample reference: **0.0** at every pad
  factor from 1× to 16×, against 1.5e-6 → 3.6e-5 before.
* the 2-D transverse-walk remap rebuilds the whole exit field from a fixed
  181 × 181 launch lattice whatever the field sampling, so structure finer
  than the launch pitch (a hard stop edge, an obscuration, an upstream DOE,
  speckle) is smoothed to the lattice — silently.  It now WARNS, naming the
  launch pitch, the field pitch and the two routes that do not smooth
  (`displaced_obliquity='pointwise'`, which lives on the field grid, and
  `apply_real_lens_traced`).  The lattice itself is NOT raised: measured on a
  decentered f/5 singlet at N = 512, scored by the mirror-symmetry residual of
  the image-plane intensity (+d vs −d, an exact symmetry of the physics, so any
  residual is the model's own artefact), it reads 7.9e-14 at n_side = 181,
  5.5e-14 at 257 and 4.1e-14 at 513 — but **4.1e-03 at 512 and 7.4e-03 at
  1025**.  The instability is QHull's, not the resolution's: a denser scattered
  set gives the Delaunay triangulation more near-degenerate cells to resolve
  arbitrarily, and which way it resolves them is not reflection-stable.
  Trading a documented smoothing limit for a measurable loss of an exact
  symmetry is a bad trade; the real fix is to replace `LinearNDInterpolator`
  with the structured Newton inversion this module already implements at
  `_interp2_structured` (which removes the ceiling AND the triangulation), and
  that is recorded as follow-up work rather than half-done.

### Changed -- apply_real_lens: `slant_correction=True` + `seidel_correction=True` is refused (L20)

Both flags replace the SAME per-surface coefficient and the Seidel block's model
reference is built from the screen the split step actually applies, so stacking
them double-counted the facet obliquity: measured 173.5 → 1488.6 nm rms exit OPD
on an 8 mm cemented doublet with both on.  `_check_apply_real_lens_kwarg_combination`
now raises.

### Fixed -- apply_real_lens: input-validation contracts and stale documentation (L10, L11, L18, L20)

* `assert len(thicknesses) == len(surfaces) - 1` is a `ValueError` with the §2
  prefix.  Under `python -O` the assert was stripped, so a short list surfaced
  as a bare `IndexError` from inside the loop and an over-long one was accepted
  silently; `prepare_real_lens` already raised properly for the same condition.
* `_warn_if_aperture_exceeds_grid` was called with `shape[0]` (Ny) paired with
  `dx`, describing a semi-extent that exists on NEITHER axis of an anamorphic
  grid.  It now takes both axes (`N_y=`, `dy=`, defaulting to the x values so
  every other caller is unchanged) and checks the smaller semi-extent.
* `lens_sag_float32_opd_error`'s `on_partial_aperture` guard could not fire on a
  default call: with `field_check_dx=None` the pitch is chosen so the aperture
  spans 80 % of the window, making `cover` 1.25 by construction, while the
  docstring said it warns by default and its own text said the PITCH is what the
  error depends on.  An auto-chosen pitch now also marks the run a proxy.
* the dead `_split_mode` arm in `_obl_gap_advance` is removed (a carrier is
  refused with every non-`'thin'` surface model, so it was unreachable), and the
  comment that described it is corrected.
* `_VALID_SCREEN_OBLIQUITY` was dead; it is now the single place the accepted
  set is spelled and is read by the error message, with a note on why membership
  testing is deliberately not used (`1 in ('auto', True, False)` is True).
* `sag_chunk_rows` was documented "wall-clock neutral" unqualified.  Measured
  against the whole-grid path with an explicit `sag_chunk_rows=256`: +28 % at
  N = 512, +5 % at N = 1024, +9 % at N = 2048, with the real payoff being memory
  (16.1 → 6.4–6.8 float64 grids).  The AUTO default only bands at N ≥ 4096, so
  shipped behaviour is unaffected; the sentence is now qualified with those
  numbers.
* the `screen_obliquity` cost table ("2.2× / 2.9× / 3.6× at N = 512/1024/2048",
  "+3 float geometry grids") was not reproducible and its TREND has the wrong
  sign — a fixed O(N²) addition measured against an O(N² log N) baseline must
  FALL with N.  Replaced with the re-measured table (3.88× / 5.09× / 2.38×,
  +11.13 grids).
* the numexpr-vs-numpy complex64 difference is documented on
  `PreparedAnalyticLens`: on a numexpr build at N ≥ 1024 the two paths differ by
  about one float32 ULP (1.05e-07 relative, 2.4e-08 rms), because numexpr
  narrows at the `out=` store while the prepared lens casts before the multiply.
* the `fresnel` parameter text now states that the refraction ANGLE comes from
  the real indices while the coefficients use the complex ones, that `θi` is the
  AOI of an AXIAL ray at the local facet normal, and that no path in the
  function has both a true local AOI and Fresnel.
* the `slant_correction` guidance ("helpful for an asymmetric meniscus, very
  steep asphere") and the `seidel_correction` recommendation ("turn on for
  AC254*-class cemented doublets") are rewritten against the measurements above.

### Changed -- tests: three test files that pinned the defects (V1, V2, L14)

* `tests/unit/test_audit_glass.py` — the repository's ONLY `seidel_correction`
  test wrapped the phase difference into (−π, π], divided by 2π (so the quantity
  was ≤ 0.5 by construction) and asserted it was < 50.0; a synthetic 10⁴-wave
  error scored 0.288 and passed.  Its fixture was also plano-REAR, where the
  exit-vertex defect is identically zero, and it scored exit-pupil phase, never
  focus.  Replaced by three falsifiable tests: the 5 nm gate must SKIP a
  well-corrected singlet (asserted as bit identity with the flag off); the
  correction must improve a CURVED-REAR doublet's exit wavefront by ≥ 3× against
  an independent exit-vertex ray oracle written from the surface equation; and
  it must not move the focus or cost peak intensity, measured unwrap-free by a
  parabola-refined through-focus scan.
* `tests/unit/test_v5_2_off_axis_conic_surface_frame.py` — the ±20 % slope band
  asserting that `surface_frame=True` has NO tilt ramp was the defect (L2); the
  two `array_equal(no_kwarg, default_kwarg)` assertions tested CPython's
  default-argument mechanism.  Replaced by a thin-prism deviation test (both
  branches must deviate by (n2−n1)θ, about the same axis, read from the exit
  field's spectral centroid) and a closed-form pin of the field-frame tilt-ramp
  convention plus the decenter bit-identity.
* `tests/unit/test_hammer_h1_slant_obliquity.py` — two of its three tests ran on
  under-sampled grids (dx = 12 µm against a 6.55 µm exit Nyquist, where the
  PARAXIAL arm itself reads 5.53 µm) and one was labelled "converged sampling"
  at dx = 6 µm where the same measurement moves 60 % on refinement.  Re-fixtured
  at Nyquist-compliant sampling and re-barred on the slant/paraxial RATIO, which
  converges to three digits two samplings before the absolute number does and
  brackets both defects (the pre-v5.25.0 cancellation below 1.0, the v5.25.0
  over-correction at 1.9–2.0).
* `tests/unit/test_audit_misc.py` — `test_traced_emits_warning_for_stop_index_2`
  used an out-of-range `stop_index = 2` on a two-surface singlet, which is now a
  `ValueError`.  Re-fixtured on the valid non-entrance `stop_index = 1` (which is
  what the warning is about) and joined by a pin of the new out-of-range
  contract.

### Added -- tests: `tests/unit/test_audit2609_a2_analytic_lens.py`, `tests/unit/test_audit2609_a2_displaced_models.py`

62 new regression tests (49 + 13) covering L3–L20 and E4, each with its oracle,
the oracle's own error floor, the measured value and the measured value of the
defect it guards.  No test asserts a wall clock or a speed-up, skips on a
resource precondition, or scores the code against another implementation in this
library.
