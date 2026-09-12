# WP-A2 — `apply_real_lens` analytic model (`elements/_lens_real.py`, `elements/lenses.py`)

Findings L1–L20 + E4 of `AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11.md` §2.1/§5,
plus V1/V2 of §14 and three coordinator additions.  Environment: CPython 3.14,
numpy 2.4.6, numba 0.65.1, **numexpr NOT installed** (so every measurement is on
the numpy fallback branch; the numexpr branch is desk-checked).
`OPENBLAS_NUM_THREADS=1` on every invocation.

---

## 1. Summary

| ID | Status | Files:lines | Tests | Oracle | Measured before → after |
|---|---|---|---|---|---|
| **L1** | fixed | `_lens_real.py:2156` (`_split_step_fan_opl`), `:7369-7502` (Seidel block) | `test_audit_glass.py::TestAuditFixesV4_11_2_track_a_SeidelCorrectionSignAgainstGroundTruth::{test_seidel_gate_skips_a_well_corrected_singlet, test_seidel_improves_a_curved_rear_doublet_against_a_ray_oracle, test_seidel_does_not_move_the_focus_and_does_not_cost_peak}` | independent Newton+Snell exit-vertex ray trace; unwrap-free through-focus scan | exit OPD rms, 8 mm doublet **1430.0 → 1.126 nm** (0.12× → 154× vs OFF); plano-convex gate now SKIPS (88.87 → 0.848 nm); focal peak −26.8 % → −0.27 %; focus −2.5 % → +0.15 % |
| **L2** | fixed | `_lens_real.py:6702-6746` (`_sf_active` map), `:6835-6862` (tilt) | `test_v5_2_off_axis_conic_surface_frame.py::TestTiltedSurfaceDeviatesTheBeam` (2) | thin-prism `(n2−n1)θ`; closed-form rigid-rotated sphere | tilted flat face deviation **0.000 → 2.537 mrad** (oracle 2.538); field-frame height error on R = 50 mm at 5 mrad **10.011 µm (8.15 waves) → 10.3 nm (0.0084 waves)** |
| **L3** | fixed | `_lens_real.py:1711` (`_residual_input_field`), `:1815-1829`, `:2097-2113`, `:2139-2150`, call sites `:5227/:5244/:5911/:5926` | `test_audit2609_a2_displaced_models.py::TestL3RemapsCarryTheInputPhase` (3) | thin + pointwise-screen peer models; ASM through-focus scan | flat vs 35.4-wave input separation **4.7e-16 → 1.974 / 1.961** of peak; 150 mm diverging source focus **21.0 → 25.0 mm** |
| **L4** | fixed | `_lens_real.py:2697` (`_carrier_is_geometric`), `:2771`, `:2951`, `:3019` | `test_audit2609_a2_displaced_models.py::TestL4CarrierMomentumUnits` (3); `test_lens_memory_levers.py::test_the_banded_seed_reproduces_the_whole_grid_momentum_field_exactly` | exact plane wave built in N-BK7 at a known in-glass angle | `'auto'` / ndarray `q` **0.114986 → 0.075808** (true 0.075808); TiltedCarrier unchanged; the banded row evaluators share the rule (`0.0162` vs `0.01` before) |
| **L5** | fixed | `_lens_real.py:807` (`_sag_callable_fingerprint`), `:1493-1502` | `test_audit2609_a2_displaced_models.py::TestL5L6CacheKeyCompleteness::test_a_mutated_sag_callable_is_a_miss_not_a_stale_hit` | cold rebuild of the same call | stale-hit error **0.987 of peak → 0.0 (byte-identical)** |
| **L6** | fixed | `_lens_real.py:789` (`_glass_key_value`), `:852-853`, `:1503-1504` | `…::test_repointing_a_glass_registry_entry_is_a_miss` | cold rebuild | stale-hit error **1.30e-2 of peak → 0.0** |
| **L7** | fixed | `_lens_real.py:1698-1706`, `:1998-2002` | `test_audit2609_a2_displaced_models.py::TestL7DisplacedRemapExitIndex` | closed-form single-surface trace with the correct exit index | exit-leg OPL error **1.03e-5 m (16.3 waves) → 8.1e-19 m** |
| **L8** | fixed | `_lens_real.py:3496` (`_TF_REMAP_SUPPORT_FRAC`), `:3470-3484`, `:3756-3781`, `:3799-3841`, `:3860-3876` | `test_audit2609_a2_displaced_models.py::TestL8RemapGuardsScoreTheSupport` (2) | the model's own acceptance + power closure | padded-grid acceptance **3 of 5 → 5 of 5** pad/pitch combinations |
| **L9** | partially fixed (see §2) | `_lens_real.py:1101-1119` (coarse grid), `:1857` / `:1860` (`_DISP_REMAP_2D_N_SIDE`, `_warn_if_remap_lattice_smooths`) + the call at `:5536`, `:612-623`/`:1190-1201`/`:1626-1637`/`:1932-1943` (Newton) | covered by the byte-identity + cache tests | cos grid at `n_coarse=4096`; image-plane mirror symmetry | pointwise cos-grid error inside the pupil **1.5e-6…3.6e-5 → 0.0** at every pad factor; Newton sweeps 24 → 2 (bitwise fixed point, output unchanged by construction); the remap's 181-lattice smoothing now WARNS — raising the lattice measured **4.1e-03 mirror-symmetry residual at 512 against 7.9e-14 at 181** and is deferred (§6.2) |
| **L10** | fixed (docs) | `_lens_real.py` `fresnel` parameter text | — | — | documented: axial-ray AOI, real-index refraction angle, and that no path has both true AOI and Fresnel |
| **L11** | fixed | `_lens_real.py:5147-5160` (ValueError), `:440-470` (`on_partial_aperture`), `:5887-5897` (dead `_split_mode`), `:5053-5088` (cost table) | `test_audit2609_a2_analytic_lens.py::TestL14StopIndexAndL11Contracts::test_thicknesses_contract_is_a_valueerror_not_an_assert` | — | `assert` → `ValueError` (survives `-O`); `on_partial_aperture` warnings on a default call **0 → 1**; cost table re-measured |
| **L12** | fixed | `_lens_real.py:6547-6559` (banded), `:6990-6996` (whole grid) | `test_audit2609_a2_analytic_lens.py::TestL12SlantCorrectionAxialTranslationIdentity` (2); re-fixtured `test_hammer_h1_slant_obliquity.py` | exact one-facet eikonal; independent ray oracle | single face **3.83 → 0.0041 nm** rms (940×); plano-convex curved-first 2.487 → **0.037 nm**; asphere 22.85 → **0.053 nm** |
| **L13** | fixed | `_lens_real.py:6611-6619` (banded), `:7222-7243` (whole grid) | `test_audit2609_a2_analytic_lens.py::TestL13FresnelPowerTransmittance` (4) | closed-form `4n1n2/(n1+n2)²` | single AIR→BK7 face **0.632344 → 0.958057**; bare cemented **0.846390 → 0.993599**; plate unchanged 0.917873 |
| **L14** | fixed | `_lens_real.py:2434-2472` (`_normalise_stop_index`), `:5162`, `:7664` | `…::TestL14StopIndexAndL11Contracts` (4); `test_audit_misc.py::…::test_out_of_range_stop_index_now_raises` | transmitted power | `stop_index=-1` P/P0 **1.00000 → 0.27880** (= `stop_index=1`); out-of-range now raises |
| **L15** | fixed | `_lens_real.py:6864-6904` | `test_audit2609_a2_analytic_lens.py::TestL15FormErrorValidation` (6) | closed-form `−k0(n2−n1)dz` | `(N,)` silently broadcast → `ValueError`; value/sign unchanged at −0.511441 rad |
| **L16** | fixed | `lenses.py:212-262` | `…::TestL16InPlaceSagIsBitIdentical` (7) | the previous expression, byte-for-byte | **176.4 ms / 4.13 grids → 56.5 ms / 1.13 grids** (3.12× / 3.67×), bit-identical |
| **L17** | fixed | `_lens_real.py:2291` (`_screen_exp`), `:6065-6082`, `:6597-6604`, `:7198-7208`, `:7251-7295` | `…::TestL17PhaseScreen` (6) | the previous expression, byte-for-byte | screen **253.1 ms / 3.00 → 172.2 ms / 2.50** complex128 grids; screens built per plano-convex call **2 → 1**, per window **2 → 0**; end-to-end N = 2048 **4.01 s / 16.13 → 1.57 s / 14.00** grids |
| **L18** | fixed (docs) | `_lens_real.py:188-190`, `:4786-4803` | — | `repro/RL-CORE/p17_perf.py` | "wall-clock neutral" qualified with +28 % / +5 % / +9 % at N = 512/1024/2048 |
| **L19** | fixed | `_lens_real.py:2327` (`_absorb_local_path`), `:6301-6308`, `:5954-5957`, `:6607-6610`, `:7246-7248`; tilt convention `:6702-6746` | `…::TestL19AbsorptionLocalPath` (3); `test_v5_2_off_axis_conic_surface_frame.py::TestFieldFrameTiltRampConvention` (2) | Beer-Lambert ray-column; closed-form ramp | absorption apodisation error **1.52e-2 → 4.2e-4** (36×, 97 % of the depth recovered); tilt axis unified across both branches |
| **L20** | fixed | `_lens_real.py:2417-2430` (slant+seidel), `:5396-5410` (anamorphic guard), `lenses.py:761-800`, `_lens_real.py:2630-2636` (`_VALID_SCREEN_OBLIQUITY`), `:7582-7598` (numexpr note) | `…::TestL20AnamorphicApertureGuard`, `…::test_slant_and_seidel_together_are_refused` | — | slant+seidel 173.5 → 1488.6 nm now refused; anamorphic guard reports the real semi-extent |
| **E4** | fixed | `lenses.py:297-318` | `…::TestE4AsphericKernelNonContiguousInput` (4) | the C-ordered answer, byte-for-byte | F-order/transposed/strided `h_sq`: aspheric term **dropped entirely (3.84e-5 m) → bit-identical** |
| **V1** | fixed | `tests/unit/test_audit_glass.py:67-355` | (the three tests above) | — | a vacuous assertion (`rms_waves < 50` on a quantity ≤ 0.5) replaced by three falsifiable ones |
| **V2** | fixed | `tests/unit/test_v5_2_off_axis_conic_surface_frame.py:187-411` | (the four tests above) | — | the defect-pinning slope band and two `array_equal(no_kwarg, default_kwarg)` tautologies replaced |
| **C1** (coordinator) | fixed | `lenses.py:218-230` | `…::TestL16InPlaceSagIsBitIdentical::test_zero_radius_raises_instead_of_returning_all_nan` | — | `R = 0` all-NaN + 4 anonymous numpy warnings → §2-prefixed `ValueError` |
| **C2** (coordinator) | not reproducible | — | — | — | `test_niche_c1_consolidation.py::…[no tilt ramp]` **passes** (3/3 in that parametrisation); it patches `propagators/carrier._tilt_ramp`, not `_lens_real`'s surface-frame branch |
| **C3** (coordinator) | fixed | `tests/unit/test_audit_misc.py:2224-2292` | `…::TestAuditFixesV4_12_1_coverage_StopIndexWarn` (3) | — | re-fixtured on a VALID non-entrance stop; new out-of-range contract pinned |
| **C4** (coordinator) | fixed | `_lens_real.py:2243-2265` | `…::TestSasRefusesAnAnamorphicPitch` (2) | — | `wave_propagator='sas'` with `dy != dx` silently square → `ValueError`; square-pitch sas still runs |

Not-regressed gates (all re-measured AFTER every change):
`repro/RL-CORE/p1b_thin_vs_rayoracle.py` — the default thin path is UNCHANGED
(0.848 / 1.177 / 1.829 / 10.857 nm rms on the four fixtures; the `sag·NA²`
prefactor table 0.0695 / 0.0486 / 0.0480 / 0.0492 / 0.0516 reproduces exactly).
`repro/RL-CORE/p10_banded_identity.py` — 44/44 OK: banded == whole-grid
byte-identical over 5 prescriptions × 7 band sizes plus the slant arm × 3 band
sizes, `accumulator_store='memmap'` == `'ram'`, `prepare_real_lens` ==
`apply_real_lens` byte-for-byte at complex128 AND complex64, complex64 vs
complex128 6.745e-07.

---

## 2. Per finding

### L1 — `seidel_correction=True` (P1, done LAST per the decision rule)

**Wrong.**  Three compounding defects, all confirmed on HEAD before touching
anything: (a) the 41-ray fan's OPL read at the last surface's SAG — measured
−6.87 µm and −27.5 µm of pure ρ² on the 4 mm and 8 mm doublets, exactly 0 on a
plano rear; (b) the model reference `Σ(n2−n1)·sag_i(h)`, which is not the
split-step's OPL — it omits the in-glass obliquity the ASM legs already carry,
so the "correction" double-counted 337 nm of it against a 341 nm prediction on a
lens whose true residual is 0.85 nm; (c) a fit basis starting at ρ², i.e.
containing defocus.

**Changed.**  (1) `res_fan.at_exit_vertex()` — the raytrace helper WP-A1 landed,
used directly, no private copy.  (2) A new private `_split_step_fan_opl` builds
the model's OWN exit-vertex OPL on the same fan: at each surface the ray meets
the screen on that surface's vertex plane (no axial motion), its OPL changes by
`−(n2−n1)·sag_i(x)` and its transverse momentum by `−(n2−n1)·∇sag_i(x)` — the
deflection a phase screen actually imparts; across each gap it walks straight
through the glass accumulating `n²t/pz = n·t/cosθ`, which is the stationary-phase
transport of the ASM leg.  Verified directly: `wave − model` = **1.001 nm rms**
on the 8 mm doublet, against `wave − ray` = 173.5 nm.  (3) The ray OPL is
carried from the ray's landing point to the MODEL ray's landing point at the
exit momentum `p = n_exit·L` (they differ by up to 12.7 µm), because the screen
multiplies the field at a fixed exit coordinate; without this the residual reads
114 nm where the true one is 173 nm and the correction under-corrects by a
third, making the wavefront WORSE (measured 0.60×).  (4) The fit starts at ρ⁴
and the 5 nm gate scores the fitted ρ⁴+ part.

**Verified.**  Decision rule: the plano-convex gate must SKIP — it does
(corr_rms 1.35 nm < 5 nm; the corrected call is bit-identical to the uncorrected
one).  The 8 mm doublet must improve ≥ 3× and not drop the through-focus peak —
**154×** (173.47 → 1.126 nm rms vs the independent oracle) with the peak up
2.9 % on the 4 mm doublet and the plano-convex bit-identical.
`repro/orch/seidel_focus_shift.py`: the AC254-100-like doublet's on-axis peak
stays at z = 114.0 mm against a 114.85 mm paraxial BFL (it moved to z = 62.0 mm
before).  Full table in the changelog.

**Residual risk.**  The 4 mm doublet gains 5.9× where the 8 mm one gains 154×;
the smaller pupil's residual (10.9 nm) is closer to the fan's own fit floor, so
the correction has proportionally less to work with.  The correction remains a
RADIAL screen fitted on a collimated on-axis fan: it is not valid for an
off-axis or non-collimated input, which the docstring now says.

### L2 + L19 (tilt) — `surface_frame=True`

**Wrong.**  The rigid-body branch evaluated the sag at the rotated transverse
FOOTPRINT and discarded the rotated surface's field-frame HEIGHT.  A rotation
re-expresses the ramp; it does not delete it.  Measured on HEAD: a tilted flat
N-BK7 face deviated the beam by exactly 0.000 mrad (thin prism: 2.575 mrad) and
was byte-identical to the untilted face.

**Changed.**  The screen now imprints `(n2−n1)·z_f` with
`z_f = R_zx·x_s + R_zy·y_s + R_zz·g(x_s, y_s)`,
`R_z· = (−cos θx sin θy, sin θx, cos θx cos θy)`.  The rotation angles are read
from the library-wide meaning of the `tilt` key (`tilt = (t0, t1)` IS the ramp
`t0·x + t1·y`, i.e. `θx = t1`, `θy = −t0`), so the two branches now deflect
about the SAME axis.

**Verified.**  `repro/orch/surface_frame_tilt_check.py`: both branches deviate
2.537 mrad against the 2.538 mrad thin-prism oracle, on both axes.  Against the
closed-form rotated sphere (R = 50 mm, ±2 mm): 1.7 / 10.3 / 80.1 nm at 1 / 5 /
20 mrad, from 2.002 / 10.011 / 40.077 µm.  Decenter-only remains byte-identical
between the branches.

**Residual risk / decision taken.**  The WP asked for the field-frame branch to
be re-spelled to CONVENTIONS §7 (`−θy·x + θx·y`) and noted "check whether
`raytrace.surfaces_from_prescription` reads the same `tilt` key — if so the fix
also makes ray and wave paths agree".  It does read the same key, but with the
SAME legacy convention (`raytrace/surface.py::_field_frame_sag_and_grad` applies
`tx*(x−dx) + ty*(y−dy)`), as does `validation/oracles/geom_spot_decenter_oracle`
— and `test_niche_p9_decenter_tilt.py::test_field_frame_geometry_matches_analytic_and_geom_oracle`
pins the three against each other to 1e-12/1e-13.  Flipping `_lens_real` alone
would therefore have DESYNCED the wave model from the ray models (a silent
physics divergence of exactly the class this audit is about) to fix a P3 naming
contradiction.  I unified the convention across the two `apply_real_lens`
branches instead — which is the ORCHESTRATOR's stated fix for F-O2 ("unify the
tilt-axis convention across both branches") — and list the §7 re-spelling as a
cross-module request in §5.  `test_niche_p9_decenter_tilt.py` passes (12/12).

### L3 — the `displaced` remaps discarded the input phase

**Wrong / changed / verified.**  See the changelog.  The key design point: the
demodulated field `F = E_in·exp(−i k0 W_conj)` carries `|E_in|` in its modulus,
so ONE complex resample replaces the old real one — no extra interpolation pass,
and for a real non-negative input with `conjugate=None` (`W = 0`) it reduces to
the old amplitude sample exactly.

**Residual risk.**  The residual is transported as a complex field through a
bilinear resample, so it must be smooth on the grid: the closer `conjugate` is
to the field's actual congruence, the better.  A wildly mismatched pair (a
20-wave-per-pixel residual) would be aliased rather than refused.  The
`conjugate` docstring now says so.  The audit's separate observation that the
2-D remap carries no in-glass diffraction at all is unchanged and still
documented in `displaced_mode`.

### L4 — carrier momentum units

**Wrong / changed / verified.**  See the changelog.  The branch test is on the
carrier's TYPE (`TiltedCarrier` or a real scalar → geometric, needs `n1`;
`'auto'` or ndarray → already optical).

**THREE SITES, one predicate.**  Fixing only `_screen_obliquity_angle_field`
broke the banded path: `_screen_obliquity_row_evaluator` and
`_screen_obliquity_rows_any` each carry their own `* n1`, and their bands must
be byte-identical to the corresponding row slice of the whole-grid field.
`test_lens_memory_levers.py::test_the_banded_seed_reproduces_the_whole_grid_momentum_field_exactly`
caught it immediately (`carrier='auto'`, n1 = 1.62: banded 0.0162 against
whole-grid 0.01 — exactly the factor).  All three now call one
`_carrier_is_geometric` predicate, so they cannot drift again.

**Residual risk.**  RL-MODELS' related, lower-exposure observation —
`_displaced_carrier_slope_fn` / `_displaced_carrier_dir_fn` treat the carrier
gradient as a TANGENT, which is right for a scalar conjugate and a third-order
error (~1.3e-3 relative at 50 mrad) for `'auto'` — is NOT fixed here; it is a
different function with a different consumer and was not in the WP's L4 line.
Recorded as deferred (§6).

### L5, L6 — cache keys

**Wrong / changed / verified.**  See the changelog.  The `sag_callable`
fingerprint is kept ALONGSIDE the object (not instead of it), so the entry still
holds a reference — an `id()`-only key could be recycled after GC — and two
distinct callables still miss.  Pre-fix behaviour confirmed in process by
reverting each keying function and re-running the same call (0.987 and 0.013 of
peak of stale-hit error respectively; both 0.0 after).

### L7 — the displaced remaps' exit referencing leg

**Wrong / changed / verified.**  See the changelog.  `idx[-1][1]` was already
resolved in both functions; the fix is one term.

### L8 — `tangent_facet_remap` fold / pull-back guards

**Wrong.**  Both reductions were whole-grid, so a converging beam's ordinary
padding caused refusals (measured: 8.2× pads declined at `min det = −0.87` on
dark corners while the pupil sat at 0.9986).

**Changed.**  Support = `|E| > 1e-6·max|E|` (1e-12 of peak intensity; exactly the
aperture for a hard-apertured pupil).  `det` is still computed from the
UNCLAMPED walk — an intermediate attempt that zeroed `W` outside the support
made `W` discontinuous and the fixed point `x = u − W(x)` oscillate with period 2
for every pixel within one walk of the support edge, which BROKE a grid that had
converged before (measured: a 2× pad stalled at a 0.303 px step).  Only the
CONVERGENCE TEST is support-restricted.  The pull-back loop additionally stops
as soon as the residual stops contracting (`_TF_REMAP_PROGRESS_FRAC = 0.999`) and
its ceiling moves 64 → 256, because the cap exists to stop divergence, not to
truncate a slow contraction.

**Verified.**  `repro/RL-MODELS/t9_fold.py`: 5/5 pad × pitch combinations
accepted (3/5 before), with a genuine in-support fold still refused.

**Residual risk.**  The support threshold is a fraction of the PEAK amplitude,
so a field with a hot spot far from the region of interest could shrink the
scored support.  The failure mode is permissive (accept where it should refuse)
rather than the reverse; the whole-grid minimum is kept in the message.

### L9 — resolutions and the Newton loops

**Changed.**  (a) The pointwise cos-grid's coarse sample count now scales with
the pad factor, so the pitch inside the traced aperture is fixed at its unpadded
value.  (b) The 2-D remap's 181 × 181 launch lattice now WARNS when its pitch
is coarser than twice the field pitch, naming both and the two routes that do
not smooth.  (c) All four fixed-24-sweep Newton loops break at a BITWISE fixed
point of `t` — chosen over a tolerance precisely so the output cannot move:
once another sweep provably cannot change a bit, the rest is pure cost.

**Verified.**  `repro/RL-MODELS/t14_coarse.py`: the pointwise cos-grid error
inside the pupil is now **0.0** at every pad factor from 1× to 16× (it ran
1.5e-6 → 3.6e-5 before, 24× degradation over a 16× pad).  The byte-identity
matrix and every cache test still pass, which is the Newton break's gate.

**(b) was implemented as a resolution scaling, then REVERTED on measurement.**
Scaling the launch lattice with the grid broke
`test_niche_p10_transverse_walk_remap.py::test_remap_sign_mirror_decenter`,
which asserts an EXACT symmetry of the physics: a +d and a −d decenter must
give x-mirrored PSFs.  Scored by the mirror residual of the image-plane
intensity on that fixture (N = 512, f/5 singlet, d = 0.6 mm): 7.9e-14 at
n_side = 181, 5.5e-14 at 257, 4.1e-14 at 513 — but **4.1e-03 at 512 and
7.4e-03 at 1025**, and EE80's own 1e-3 symmetry bar breaks with it.  The
instability is the Delaunay backend's, not the resolution's: a denser scattered
set hands QHull more near-degenerate cells to resolve arbitrarily, and which
way it resolves them is not reflection-stable.  Trading a documented smoothing
limitation for a measurable loss of an exact symmetry is a bad trade, so the
lattice stays at 181 and the finding's *silence* is what this pass fixes.  The
real fix — replacing `LinearNDInterpolator` with the structured Newton
inversion this module already implements at `_interp2_structured`, which
removes the resolution ceiling AND the triangulation — is §6.2.

**Residual risk.**  The remap still smooths sub-launch-pitch input structure;
it now says so on every call where that is true, which is the condition the
audit measured (a ripple at 2.2 launch samples/period returns at 0.51 of its
input contrast where the field-grid screen path resolves it at 1.26).

### L10, L11, L18, L20 — documentation, validation and dead code

See the changelog.  All verified by running the affected paths
(`on_partial_aperture` warnings 0 → 1 on a default call; the anamorphic guard
now reports `semi=0.512` for a 512×64 grid at dx = 2 µm, dy = 40 µm; the
slant+seidel combination raises).

### L12 — `slant_correction`

**Wrong / changed / verified.**  See the changelog, which carries the full
before/after table and the migration note.

**Residual risk — stated plainly.**  The new coefficient is the exact
axial-translation identity for a COLLIMATED input.  On an element whose LATER
surfaces see a strongly converging bundle the collimated assumption is what
limits it, and the OLD (wrong-signed, 1.94×-oversized) coefficient sometimes
compensated that other error.  Measured on the f/5 symmetric biconvex hammer
fixture at converged sampling (dx ≤ 3 µm), image r2m against a 65 µm dual-oracle
truth: paraxial 40.55 µm, old slant 76.70 µm, new slant 43.06 µm.  So on THAT
fixture the old form scored closer, by a cancellation the audit identified.  The
physics is not in doubt (the expansion, the orchestrator's independent
re-derivation, the 2.9414 ratio matched to 5 digits at four radii, and 940× on a
single face), but callers using `slant_correction` on a fast symmetric element
will see their numbers move away from a ray trace and must switch to
`carrier=` / `displaced` / `tangent_facet` / `traced`.  This is called out in
the changelog as a behaviour change with a migration note.

### L13 — `fresnel`

**Wrong / changed / verified.**  See the changelog.  Note for CONVENTIONS: the
library's power convention (`Σ|E|²·dx·dy` is the power, no impedance factor
anywhere) is now stated in the code but is not written down in `CONVENTIONS.md`
— requested in §5.

### L14, L15, L16, L17, L19, E4

See the changelog; every one is verified by a two-sided (byte-identity or
closed-form) assertion.

---

## 3. Files touched

Modified (owned):
* `lumenairy/elements/_lens_real.py`
* `lumenairy/elements/lenses.py`
* `tests/unit/test_audit_glass.py` — the V1 Seidel class only
* `tests/unit/test_v5_2_off_axis_conic_surface_frame.py` — the V2 classes
* `tests/unit/test_hammer_h1_slant_obliquity.py` — the two L12-affected tests +
  the module docstring
* `tests/unit/test_audit_misc.py` — `TestAuditFixesV4_12_1_coverage_StopIndexWarn`
  ONLY (WP-A10's D.2 hunk in the same file untouched)

Modified (owned; added after the coordinator's follow-ups):
* `validation/elements/test_lenses.py` -- the `all optional features together`
  case only (it passed the now-refused `slant_correction` +
  `seidel_correction` pair), plus a new case pinning the refusal

New:
* `tests/unit/test_audit2609_a2_analytic_lens.py` (49 tests)
* `tests/unit/test_audit2609_a2_displaced_models.py` (13 tests)
* `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A2_REPORT.md`
* `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A2_CHANGELOG.md`

No git write commands were run.  No other agent's files were edited.  No
processes were killed.

---

## 4. Tests run

All with `OPENBLAS_NUM_THREADS=1` and
`python -m pytest <file> -q --no-header -p no:cacheprovider`.

| command | result | duration |
|---|---|---|
| `test_audit2609_a2_analytic_lens.py` | **49 passed** | 3 s |
| `test_audit2609_a2_displaced_models.py` | **13 passed** | 38 s |
| `test_audit_glass.py` | **21 passed** | 88 s |
| `test_v5_2_off_axis_conic_surface_frame.py` | **5 passed, 1 skipped** (Optiland absent) | 4 s |
| `test_hammer_h1_slant_obliquity.py` | **3 passed** | 33 s |
| `test_audit_misc.py::TestAuditFixesV4_12_1_coverage_StopIndexWarn` | **3 passed** | 3 s |
| `test_niche_p10_transverse_walk_remap.py` + `test_audit2609_a2_displaced_models.py` + `test_niche_p2_displaced_extreme.py` + `test_g2_displaced_congruence.py` | **40 passed** | 451 s |
| `test_lens_memory_levers.py` + `test_audit2609_a2_displaced_models.py` | **155 passed** | 66 s |
| `test_lens_chunked_sag.py` + `test_slant_chunk_byte_identical.py` + `test_niche_audit_p2_fresnel_tf_buffer.py` + `test_obl_banded_halo.py` + `test_tf_banded_halo.py` + `test_hammer_h1_*` (batch) | **230 passed, 2 failed** — both the H1 tests, since re-fixtured and passing | 101 s |
| `test_elements_lens.py` + `test_sag_float32_production_window.py` + `test_audit_s4_3_waveoptics_biconic.py` | **27 passed, 2 skipped** (PySide6 absent) | 25 s |
| `test_niche_p9_decenter_tilt.py` + `test_niche_r1_cosgrid_cache.py` + `test_g2_displaced_congruence.py` | **38 passed** | 414 s |
| 18-file consolidated sweep over every module I touched, run LAST (`test_audit2609_a2_*`, `test_v5_2_off_axis_conic_surface_frame`, `test_hammer_h1_slant_obliquity`, `test_audit_glass`, `test_audit_misc`, `test_lens_chunked_sag`, `test_slant_chunk_byte_identical`, `test_niche_audit_e_prepared_and_enums`, `test_lens_memory_levers`, `test_obl_banded_halo`, `test_tf_banded_halo`, `test_screen_obliquity`, `test_elements_lens`, `test_sag_float32_production_window`, `test_niche_p9_decenter_tilt`, `test_niche_p3_pointwise_obliquity`, `test_niche_r1_cosgrid_cache`) | **838 passed, 4 skipped, 1 failed** — the failure is `test_l22_delegate_reports_the_discarded_physics_kwargs` (another WP's, see below) | 901 s |
| `python validation/run_all.py test_lenses` | 2 of my cases pass; 1 pre-existing failure remains (JAX x64, below) | 75 s |

Repro scripts re-run (before AND after):
`RL-CORE/p1b_thin_vs_rayoracle.py`, `p3_fresnel_energy.py`, `p5_slant.py`,
`p5c_slant_sign.py`, `p6_seidel.py`, `p6d_psf.py`, `p7b.py`, `p8_tilt_frames.py`,
`p9_formerr.py`, `p10_banded_identity.py`;
`RL-MODELS/t5_phase_discard.py`, `t6_phase_discard2.py`, `t9_fold.py`,
`t14_coarse.py`, `t19_units.py`;
`orch/surface_frame_tilt_check.py`, `orch/seidel_focus_shift.py`.
`RL-CORE/p4_absorption.py` does not run on this build (it imports
`lumenairy.glass.register_glass`, which does not exist); the absorption
measurement was reproduced through `GLASS_REGISTRY` instead and is in the
changelog.

**Validation.**  `validation/elements/test_lenses.py`'s
`all optional features together` case passed `slant_correction=True` AND
`seidel_correction=True`, which L20 now refuses; it is re-fixtured to exercise
the two one at a time, and a new case pins the refusal.  Both pass.

**Pre-existing failures found, unrelated to WP-A2.**  Each verified to be in a
file or a path I did not touch:
* `test_audit_lens.py::test_vectorised_matches_scalar_loop_harvey_shack` and
  `::test_vectorised_matches_scalar_loop_default_lambertian` — BSDF tests
  (`lumenairy/elements/bsdf.py`, modified in the working tree by a concurrent
  work package: E6's `make_bsdf` / Harvey-Shack items).
* `test_niche_audit_e_prepared_and_enums.py::test_l22_delegate_reports_the_discarded_physics_kwargs`
  — `apply_real_lens_traced`'s `on_noncollimated='delegate'` now reports
  `dy=8e-06, on_undersample='silent'` as DISCARDED on an all-default call.  The
  `('on_undersample', on_undersample)` entry is an ADDED line in that WP's
  1341-insertion diff of `_lens_traced.py`; neither kwarg reaches any code I
  changed.
* `validation/elements/test_lenses.py::apply_real_lens_traced_jax` —
  `RuntimeError: ... requires double precision, but jax_enable_x64 is
  disabled`, i.e. the §15.6 JAX-x64 policy landing in `_lens_jax.py`.

**Pre-existing failures I found and FIXED because my changes surfaced them:**
`test_niche_audit_e_prepared_and_enums.py::test_h3_prepared_analytic_lens_resolves_sag_dtype_and_dy`
(the float32-geometry screen, §2/L17),
`test_lens_memory_levers.py::test_the_banded_seed_reproduces_the_whole_grid_momentum_field_exactly`
(the banded carrier momentum, §2/L4) and
`test_niche_p10_transverse_walk_remap.py::test_remap_sign_mirror_decenter`
(the remap launch lattice, §2/L9).  All three now pass.

---

## 5. Requested changes outside my ownership

1. **`CONVENTIONS.md` — state the power convention.**  Add one sentence to the
   conventions section: *"`sum |E|**2 * dx * dy` IS the optical power: the
   propagators are Parseval-unitary and carry no impedance factor, so any
   element that crosses an index step must apply the POWER transmittance
   `T = (n2 cos theta_t)/(n1 cos theta_i) |t|**2`, not `|t|**2`."*  This is the
   convention L13's fix now depends on and it is currently written down only
   inside `_lens_real.py`.

2. **`lumenairy/raytrace/surface.py::_field_frame_sag_and_grad` +
   `validation/oracles/geom_spot_decenter_oracle.py::_surface_z_grad` +
   `lumenairy/elements/_lens_real.py::_disp_surface_z_grad` — the §7 tilt
   re-spelling (L19, P3).**  All three apply `tx*(x−dx) + ty*(y−dy)`, i.e.
   `tilt = (−θy, +θx)` relative to CONVENTIONS §7.  To make the key mean what §7
   says, all three must change to `−ty*(x−dx) + tx*(y−dy)` (and the gradients
   `dfdx += tx` / `dfdy += ty` become `dfdx -= ty` / `dfdy += tx`) **in the same
   commit**, together with `_lens_real`'s field-frame ramp and surface-frame
   angle mapping, and every stored `tilt` migrated as
   `new_tilt = (old_t1, −old_t0)`.  Doing it piecemeal desyncs the wave model
   from the ray models.  I have left the shipped reading in place and unified the
   two `apply_real_lens` branches onto it; the cross-model pin
   (`test_niche_p9_decenter_tilt.py`) is green.

3. **`lumenairy/elements/lenses_maslov.py` and `lumenairy/elements/_lens_traced.py`
   — `stop_index` validation.**  `apply_real_lens` now refuses an out-of-range
   `stop_index` (L14) through `_lens_real._normalise_stop_index`.
   `apply_real_lens_maslov` still accepts `stop_index=2` on a two-surface
   prescription and warns rather than raising (verified:
   `test_audit_misc.py::…::test_maslov_emits_warning_for_stop_index_2` passes
   unchanged).  Suggest importing and calling the same helper in both, so the
   family diagnoses a malformed key identically.

4. **`lumenairy/propagators/asm.py` — fold the fftshift pair into the transfer
   function (RL-CORE perf item 5, WP-A5).**  `_propagate_through_glass` triggers
   two full complex copies per in-glass gap (17.2 GB each at N = 32768 /
   complex128); `np.roll` shows 4 calls / 0.30 s at N = 2048 in the profile of a
   3-surface element.  Not touched here (asm.py is WP-A5's).

5. **`lumenairy/elements/lenses.py::_warn_if_aperture_exceeds_grid` — the new
   `N_y` / `dy` keywords.**  Defaulted so `lenses_maslov.py:1467` and
   `_lens_traced.py:8176` are unchanged, but both would benefit from passing
   their own y-axis extent for the same reason `apply_real_lens` now does.

---

## 6. Deferred

1. **`_displaced_carrier_slope_fn` / `_displaced_carrier_dir_fn` treat the
   carrier gradient as a TANGENT** (RL-MODELS, the "related, same class" note
   under L4).  Correct for a scalar conjugate (`g = h/s` is a tangent), a
   third-order error for `'auto'` (~1.3e-3 relative at 50 mrad) and, in glass,
   additionally off by `n1`.  Design: give both functions the same
   type-dispatch `_geometric` test `_screen_obliquity_angle_field` now uses, and
   convert the `'auto'`/ndarray gradients with `g = q/sqrt(n1² − q²)` before
   forming `(dz, dy) = (1, g)/sqrt(1+g²)`.  Needs an immersed-carrier fixture to
   pin.  ~0.5 d.

2. **Replace the 2-D remap's `LinearNDInterpolator` with the structured
   Newton inversion already implemented at `_interp2_structured`** (RL-MODELS'
   own recommendation), and only then raise the launch lattice.  This is the
   blocking item for the second half of L9: the launch fan is a REGULAR grid,
   so its exit map is a smooth curvilinear grid that can be inverted with
   `map_coordinates` instead of QHull.  That removes the 181-ray resolution
   ceiling, the 2× triangulation cost (measured 16.44 s Delaunay vs 8.43 s
   structured on the cos-grid analogue) and — the reason it must come first —
   the reflection instability measured above, which is what made raising the
   lattice alone a regression.  ~1.5 d, with
   `test_niche_p10_transverse_walk_remap.py`'s mirror / centroid / EE80
   symmetry trio as the acceptance test.  A public `displaced_n_side` kwarg
   (~0.25 d) only makes sense after it.

3. **The `tangent_facet_remap` pull-back could use a residual-scaled
   convergence bar.**  `_TF_REMAP_PULLBACK_TOL_PX = 1e-9` is a fixed pixel bar
   whose own docstring derives it from a PHASE budget (`k0 |p| tol dx <
   1e-6 rad`).  On a heavily padded fine grid the contraction rate approaches 1
   and the loop needs ~90 sweeps to reach it (it now gets up to 256).  Design:
   accept when `k0 · max|p| · step · max(dx, dy) < 1e-6 rad` as well as on the
   pixel bar, using the `pox`/`poy` already in hand.  ~0.25 d.  Not needed for
   any fixture measured here (all 5 now converge).

4. **Collapse the three surface bodies** (`_narrow_chunk`,
   `_slant_narrow_chunk`, whole grid) into one `for band in bands(...)`
   generator, as RL-CORE recommends.  I kept the three bodies and the
   byte-identity matrix as the gate — the L12 and L13 defects were each
   duplicated verbatim across two of them, and this pass had to fix both copies
   by hand twice more.  ~3 d, and the byte-identity matrix
   (`repro/RL-CORE/p10_banded_identity.py`, 44 configurations) is the ready-made
   acceptance test.

5. **CuPy.**  Not installed, so the GPU twins of every change here are
   desk-checked only.  `_screen_exp` falls back to `xp.exp` for CuPy (its
   elementwise kernels are already fused).  `_absorb_local_path`, the in-place
   masks and the in-place `surface_sag_general` use only `out=`, boolean-mask
   assignment and `logical_not(out=)`, all of which CuPy supports.  The in-place
   `E *= ph` and `E[mask] = 0` operate on `E`, which is a device copy on the GPU
   path.  RL-CORE's two latent CuPy observations (`_obl_q_whole()` missing the
   `xp.asarray` promotion, `sag_callable` using `np.asarray`) are unchanged and
   remain unreachable.

6. **JAX.**  `apply_real_lens` has no JAX twin (`_lens_jax.py` is the TRACED
   model's), so there is no parity to test for these findings.

---

## 7. Changelog text

`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A2_CHANGELOG.md`
