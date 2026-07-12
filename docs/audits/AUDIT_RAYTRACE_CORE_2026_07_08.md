# Raytrace-Core Audit — 2026-07-08

Scope: line-level reads of the geometric-trace foundation —
`raytrace/intersection.py` (528, full), `raytrace/paraxial.py` (254,
full), `raytrace/trace.py` (the sequential engine + prescription
conversion, lines 1–520 read in depth; the ray-generator /
element-bridge tail covered structurally, its conventions already
cross-checked via consumers in `analysis/field.py`, `hfpi.py`, and
`ghost.py`).  `raytrace/differential.py` (634) was fully audited in
`AUDIT_V5_21_DELTA_2026_07_07.md`.  Continuation of the
under-covered-subsystem sweep; the tracer underpins
`apply_real_lens_traced`, GBD, the multibranch solver, `image_plane_wfe`,
and every field-swept analysis.  Read-only single-context pass; the
vector optics re-derived.

---

## 1. Verdict

**The geometric kernel is correct.**  Independently verified:

* **Vector Snell** (`_refract`): `d_t = μ·d + (μcosθᵢ − cosθₜ)·n̂` with
  the normal flipped against the incident ray — exact standard form;
  TIR flagged with first-failure-wins error codes; degenerate-magnitude
  directions flagged `RAY_NAN` *and* dead (the 4.11.1 fix).
* **Reflection**: `d + 2cosθᵢ·n̂` under the same normal convention —
  exact.
* **Intersection**: the v5.4.1 direction-aware near-root pick (`|t₁| ≤
  |t₂|`) present on both the Newton-skip spherical fast path and the
  Newton initial guess (the Cassegrain-secondary reproducer's fix);
  tangent rays accepted (P3-3); Newton stuck-detection prevents the
  zero-step "false convergence"; the **signed** vertex-to-sag OPL leg
  (`opd += n·t`) with the explicit concave back-tracking rationale.
* **Coordinate breaks**: optical-convention rotations (the 3.7.1 sign
  fix), Zemax PARM-6 order semantics (the 4.10 swap fix), inverse-frame
  decenter.
* **Trace loop**: the intersect-in-n₁ / transfer-in-n₂ OPL split is
  consistent; the DOE kick applies the grating equation with the
  matching linear OPL term `m·λ·(x/Λx) + …` (gradient-consistent),
  sign-preserving N reconstruction, and evanescent-order kill; the
  `output_filter='last'` memory mode and coord-break history alignment
  are correct; P2-35 per-surface `semi_diameter` NumPy/JAX parity
  verified in `surfaces_from_prescription`; the `'MIRROR'`
  marker-glass inference and stop-index/per-surface-flag reconciliation
  are sound.
* **`paraxial.py`**: the OSA conversion factors (`c₂₀ = d/(2√3)`,
  `c₂₂ = a/√6`) re-derived; the FoV/invariant helpers are honestly
  documented approximations (F-25 formula fix in place).

---

## 2. Findings

### RT-1 (P4) — `_transfer` and `_intersect_surface` disagree on OPL sign convention
`_intersect_surface` deliberately accumulates the **signed** `n·t` for
the vertex→sag leg ("a negative t … corresponds to back-tracking, and
we should subtract the over-counted OPL"), but `_transfer` accumulates
`n·|t|`.  On any leg where a ray has already crossed the next vertex
plane before the transfer (t < 0 with the ray still forward-going —
overlapping-sag geometries where adjacent surfaces intersect), the
transfer **adds** `n·|t|` where the intersection convention would
subtract, over-counting the composite OPL by `2n|t|`.  Well-formed
prescriptions never produce this leg (post-mirror backward propagation
uses negative thicknesses and yields t > 0), so this is a
degenerate-geometry edge — but the two primitives implement opposite
conventions for the same quantity, and the trace's telescoping-OPL
model assumes they match.  **Fix**: use signed `n·t` in `_transfer`
(matching `_intersect_surface`), or assert `t >= 0` and flag violations
as prescription errors.

### RT-2 (P4) — `surface_sag_biconic` is separable, not the Zemax biconic
The library's biconic sag is the **separable per-axis conic sum**
`z = z_x(x) + z_y(y)` (explicitly documented with its formula), and
`_surface_sag_derivatives_xy`'s per-axis derivative sum is exactly
consistent with it — internal consistency verified.  However, the
docstring labels it "Biconic (Zemax 'Biconic')": Zemax's biconic uses a
**single shared square root**
`z = (c_x x² + c_y y²)/(1 + √(1 − (1+k_x)c_x²x² − (1+k_y)c_y²y²))` —
non-separable.  The two agree paraxially but diverge in the
fourth-order cross-terms, so imported Zemax BICONIC surfaces are
silently approximated at large aperture.  Document the deviation (or
implement the Zemax form and its exact gradient).

### RT-3 (P4) — dead `_paraxial_trace` carries a wrong refraction recursion
`seidel.py`'s `_paraxial_trace` / `_paraxial_refract` /
`_paraxial_transfer` are defined and exported in `__all__` but **never
called anywhere in the library**.  The trace's refraction update
`u ← u − yφ/n₂` omits the `(n₁/n₂)` rescaling of the incident angle
(correct: `u₂ = (n₁u₁ − yφ)/n₂`), wrong by `u₁(1 − n₁/n₂)` at every
glass transition with non-zero incident angle — and the inline comment's
equivalence claim is mathematically false.  The two helper siblings are
actually correct reduced-coordinate forms.  Since all three are
exported, any future caller inherits the broken trace.  Delete the trio
or fix the recursion.  (**The live paraxial machinery is unaffected**:
`system_abcd` is pure matrix composition and `seidel_coefficients` runs
its own — verified-correct — reduced-coordinate `(y, ν=nu)` trace.)

---

## 2b. `surface.py` + `seidel.py` (full/deep reads — same day)

Verified in `surface.py`: the conic sag derivative identity
`dz/dh = h/(R√(1−(1+k)h²/R²))`, the out-of-domain **NaN** discipline
(P3-2 / F-19 — no more silent flat normals on nonexistent geometry),
the `(−∂z/∂x, −∂z/∂y, 1)/|·|` normal, the freeform FD-gradient
fallback, and the P3-60 `_surface_copy_with` full-field propagation.

Verified in `seidel.py` (the live code): `system_abcd`'s Welford
mirror-parity matrix composition (`φ_mirror = −2n₁/R`, concave mirror
EFL positive — the 4.11.2 reconciliation, consistent with
`analysis/field.petzval_radius`); the Welford principal-plane formulas
`H = (D−1)/C`, `H′ = (1−A)/C`; the trailing-thickness strip in
`lens_abcd`; and — most importantly — **`seidel_coefficients`**:

* the reduced-coordinate `(y, ν = n·u)` marginal/chief trace is
  correct (refraction `ν ← ν − y(n₂−n₁)c`, transfer `y += (ν/n)t`);
* the Welford sums `S₁ = −A²hΔ(u/n)`, `S₂ = −AĀhΔ(u/n)`,
  `S₃ = −Ā²hΔ(u/n)` with the Abbe invariant `A = n(cy + u)` match the
  standard forms (the 4.9 `Δ(u/n)`-vs-`Δ(1/n)` fix verified in place);
* the S₄ summand `−c(n₂−n₁)/(n₁n₂)` is exactly the library's own
  Petzval-curvature summand (H² factored out by documented convention
  and consistently re-applied in the S₅ Schwarzschild relation and by
  `seidel_wfe`);
* the 4.9 flat-surface branch (S₁–S₃ ≠ 0 across a flat glass interface
  because Δ(u/n) ≠ 0) is correct physics;
* the stop-aware initial conditions (marginal `y₀ = r_stop/A_pre`,
  chief `y₀ = −B_pre ν₀/A_pre`, finite-conjugate lever
  `u_obj = r_stop/(A_pre d + B_pre n)`) — algebra re-derived, correct,
  including the pre-stop ABCD's missing-last-transfer composition;
* the 4.11.2 mirror-parity threading through the sums.

Nit: three computed-and-discarded `nu_c_after / n2` statements (lines
~1082 / 1140 / 1182) — the recurring refactor-residue class.

---

## 2c. Completion tranche (same day): world / analytics / bridges / JAX

Full reads of the remaining nine modules — `core.py` (re-export shell),
`layout.py`, `world_trace.py`, `bundles.py`, `world.py`,
`seidel_analysis.py`, `ray_fan.py`, `from_field.py`, `jax_trace.py`
(~4.6k lines).  Verified clean:

* **`trace_world`** — the world↔local rotation pair, the
  intersect-in-local / re-project loop, the DOE kick (grating equation
  plus linear OPL, sign-preserving N, evanescent kill — matching the
  legacy loop), and the aperture clip (inherited — it lives inside
  `_intersect_surface`, intersection.py:270, so the world path
  vignettes identically to `trace()`).
* **`bundles.py`** — `Q = −i/z_R` waist initialisation; the P2-33
  amplitude folding `exp(+ik₀·opd)·alive` is the exact inverse of
  `rays_from_field`'s `opd = angle(E)/k₀` seeding (phase wrap cancels
  in the exponential), consistent with the `exp(+ikz)` convention.
* **`seidel_field_sweep`** — the unit-field analytic scaling (S₁/S₄
  field-independent, S₂∝σ, S₃∝σ², S₅∝σ³, y_chief∝σ) is *exact* for
  the library's H²-factored Welford forms; the 4.13.0 hoist is a true
  no-loss optimisation.  **`seidel_wfe`** — expansion matches Welford
  eq. 7.11 including the 4.11.2 field-curvature DC term; the H² ladder
  (lagrange_invariant → f·σ via ABCD → warned σ² fallback) is sound.
* **`refocus` / `through_focus_rms`** — the closed-form
  `(Δz − z)/N` transfer with signed `n·t` OPL is exactly the operator
  the full retrace would apply; sag-start correction verified.
* **`from_field.py`** — the phase-ratio k-vector estimator (exact for
  plane waves within the ±π/2-per-sample window), the three placement
  modes with the v4.15.2/3 pixel-wise inclusive thresholding, and the
  evanescent flagging all verify.
* **`jax_trace.py`** — the sag/derivative kernels (incl. the sign(R)
  4.10/4.11.1 fixes), vector Snell with TIR double-where, the
  direction-aware near-root pick (P1-1/P3-1), the P3-59 residual kill
  with dtype-aware tolerance, the P3-58 tangency/evanescence parity,
  the P2-34/P2-35 guards in *both* entry points, and the aux-keyed LRU
  jit cache with correct lock discipline (compile outside the lock).
  For even aspheric powers the JAX and NumPy sags/derivatives agree
  exactly.

### RT-4 (P3) — `world.py` tilts use the opposite sign convention from the legacy coord-break path

> **REMEDIATION NOTE (2026-07-11): this finding is a PHANTOM and was NOT
> applied (an attempted fix was reverted).**  The claim that
> `world._apply_coord_break`'s `Rx_math(+θ)` disagrees with the legacy
> `trace()` path is **false**.  Verified empirically: a +90° tilt_x sends a
> `+z` ray to world `-y` in BOTH paths — `trace()`'s ray direction becomes
> `[0,-1,0]`, and the original `world_R[:,2]` (`_rot_x(+tx)` col 2) is
> `[0,-1,0]` too.  The `periscope` folded-design and `test_world_surfaces`
> validation oracles both pin `-y` as correct.  Flipping to `_rot_x(-tx)`
> INTRODUCED the disagreement (world → `+y`) and broke those oracles, so the
> change was reverted; `world.py` keeps `_rot_x(+tx)`.  (The audit's
> derivation confused the passive ray-coordinate transform with the
> `world_R` column direction.)

`intersection._apply_coord_break` (the 3.7.1 "optical convention"
fix, validated against the 2D layout) transforms ray coordinates by
`Rx_math(+θ)` for a `tilt_x = +θ` break — i.e. the new frame's
local-to-world rotation is **Rx_math(−θ)**.
`world._apply_coord_break` builds the frame's local-to-world as
`_rot_x(tx) @ _rot_y(ty) @ _rot_z(tz)` = **Rx_math(+θ)…** — every
tilt sign is flipped relative to the legacy path (the intrinsic
X→Y→Z composition order itself is correct; decenters agree).
`world.py` is 4.4.0 code that never absorbed the 3.7.1 sign
convention — the recurring unmirrored-fix pattern.  Consequences:
`trace()` and `trace_world()` fold the *same* prescription in
opposite angular directions; tilt-only systems come out as mirror
images (self-consistent for symmetric optics, but
`paraxial_focus_world` reports the fold direction wrong in world
coordinates — `(0, −1, 0)` where the design folds to `+y`); breaks
combining decenter *and* tilt are genuinely inconsistent, not merely
mirrored, because the decenter half does NOT flip.  The module
docstring's equivalence claim ("the two trace paths agree … on
straight-axis designs") covers exactly the case that sidesteps the
sign.  **Fix**: compose the frame from the legacy convention —
`tilt_R = _rot_x(−tx) @ _rot_y(−ty) @ _rot_z(−tz)` — and pin with a
single-fold `trace` vs `trace_world` oracle at ±45°.

### RT-5 (P3) — off-axis ray fans: the 4.11.2 chief fix left the fan itself decentred (and the world variants got no fix at all)
`make_fan` launches rays at z = 0 spanning ±semi_aperture with a
uniform tilt `M = sin(fa)`, so for off-axis fields every fan ray
crosses the entrance-pupil plane displaced by `ep_z·tan(fa)`.
The 4.11.2 fix in `ray_fan_data` / `opd_fan_data` moved only the
**reference chief** to the EP centre.  Post-fix, chief and fan
sample *different* pupil regions: (a) the fan is decentred at the
stop by `ep_z·tan(fa)` (asymmetric vignetting at large field, and
the `py = linspace(−1, 1)` abscissa no longer maps to the true pupil
zone); (b) the fan no longer passes through zero at `py = 0` — its
central ray is the *old* origin-launched chief, so `ey(0)` now
reads the image offset between the two launch conventions instead
of 0 (pre-4.11.2 the fan was decentred but at least self-consistent,
`ey(0) ≡ 0` exactly).  Same defect family as AN-4, but here the
inconsistency is *within* one function.  **Fix**: shift the fan
launch heights by the same `ep_y = −ep_z·tan(fa)` offset applied to
the chief (one added line after each `make_fan` call), restoring
`ey(0) = 0` with both chief and fan EP-centred.  Additionally, the
world-frame twins `ray_fan_data_world` / `opd_fan_data_world` never
received the chief fix at all — still `make_ray(0, 0, 0, …)`
(pre-4.11.2 semantics; unmirrored).  `through_focus_rms` /
`make_rings` share the vertex-launch class but are self-referenced
(RMS about the centroid), so their exposure is only the decentred
stop sampling.

### RT-6 (P4) — `_transfer_jax`'s "paraxial approximation" is not an approximation; the high-NA warning machinery guards a non-existent error
The transfer advances every ray by the parameter `t = thickness`
along its (unit) direction and shifts the frame by `thickness` — the
resulting state is a point **exactly on the ray line** in the next
surface's frame.  Every downstream consumer is invariant to where
along the line the state sits: the flat/quadratic/Newton intersect
solves from an arbitrary point (the sphere discriminant is a
property of the line, `disc/4 = R² − dist²(centre, line)`), the
refraction/aperture/DOE steps act at the intersection point, and
the OPL telescopes exactly (`n·(thickness + t_int) ≡ n·t_total`;
verified: a 30°-ray through a gap `d` lands at `d·tanθ` with OPL
`n·d/cosθ` — exact).  So the extensively documented per-surface
error claim (`thickness·NA²/2`, "OPL error scales similarly",
4.10.0–4.10.3 investigation, DEEP-4 MEDIUM-1) mischaracterises the
kernel, and the three-version warning apparatus built on it
(v4.16.1 threshold, v4.16.2 duck-typed gate, v4.16.3 per-surface
attribution) fires a **spurious** RuntimeWarning that tells
NA > 0.31 users to distrust correct results and fall back to the
NumPy path.  Residual genuine caveats, both degenerate: the
near-root pick can differ from NumPy's only when the advanced point
overshoots past the chord midpoint of a near-tangent intersection,
and Newton's convergence basin shifts (already guarded by the P3-59
residual kill).  **Fix**: re-derive, delete or demote the warning
to a docstring note, and correct the `_transfer_jax` /
`trace_jax` docstrings.  (No numerical output is wrong today; the
defect is that a correct path actively advertises itself as
inaccurate.)

### RT-7 (P4) — `_maybe_warn_transfer_jax_high_na` concretises a possibly-traced `thickness` outside its tracer guard
The try/except tracer probe wraps only the `state.N` inspection;
`delta_t = float(thickness) * …` and the `surf_clause` f-string sit
after it.  Differentiating w.r.t. **thickness only** (via
`trace_jax_with_params(thicknesses=<tracer>)`, or `trace_jax` with a
`JaxPrescription` whose `thicks` leaf is traced) leaves `state.N`
concrete through the first intersect/refract (radii are concrete),
so the probe passes, and if `min|N| < 0.95` the
`float(<tracer>)` raises `TracerArrayConversionError` — a hard
crash of the grad call, once per process (until the latch is set by
some other successful emission).  **Fix**: move the
`float(thickness)` conversions inside the existing try/except, or
gate on `hasattr(thickness, '__array__')`+try like `direction_n`.
(Moot if RT-6's recommendation to delete the warning is taken.)

### RT-8 (P4) — `trace_jax` silently ignores `surface_diffraction` for pre-built `JaxPrescription` inputs
When `prescription` is already a `JaxPrescription`, the
`surface_diffraction` kwarg is never folded in — both trace bodies
read the DOE spec exclusively from `jp.aux`'s `diff_aux` (baked at
build time), and the kwarg threaded into `_trace_body_static` /
`_trace_body_traced` / `_make_jit_kernel` is **dead** in all three
(refactor residue).  A power user following the docstring's
"pre-build a JaxPrescription" advice who passes `surface_diffraction`
gets a trace with no grating kick and no error.  **Fix**: raise (or
rebuild aux) when both a prebuilt `jp` and a non-None
`surface_diffraction` are supplied, and drop the dead parameters.

### RT-9 (P4) — `seidel_field_sweep` output can't drive `seidel_wfe`'s corrected Petzval path
The sweep result dict carries neither `'lagrange_invariant'` nor
`'abcd'` (the ABCD is returned *separately*), so
`seidel_wfe(sweep_result, field_index=k)` — the pairing the
docstrings advertise — always lands in the legacy bare-σ² fallback
(loud, via the RuntimeWarning, but degraded: the S₄ term is off by
the (y_pupil)² factor the 4.9 fix exists to correct).  **Fix**:
store the per-field `H = σ·H_ref` array (plus the abcd) in the sweep
result and teach `seidel_wfe`'s `_pick` to index it.

### Nits (completion tranche)
* `trace_summary` / `spot_diagram`: the `f_eff`-NaN fallback prints
  the pre-4.11.2 half-angle quantity (radians) labelled with length
  units — only on degenerate/afocal systems.
* `trace_world` comment "accumulates the full inter-surface OPL via
  |t|" — `_intersect_surface` uses **signed** `n·t` (comment-only;
  cf. RT-1).
* `paraxial_focus_world`: `np.linalg.solve` raises only on exactly
  singular systems; near-afocal traces return a garbage far-field
  "focus" instead of the documented ValueError.  Its "chief" ray is
  actually the axial ray (naming).  No alive-check on the two traced
  rays.
* `through_focus_rms`: an all-dead bundle yields `rms = inf`
  everywhere and `argmin` silently returns shift[0] as "best".
* Odd aspheric powers are validated nowhere: both backends document
  even-only, both compute sag as `h²^(p//2)` (flooring odd p), but the
  NumPy normal uses `p·h^(p−1)` and the JAX normal `p·h^(p−2)·x` —
  a sag/normal-inconsistent surface, differently inconsistent per
  backend.  One even-power guard in `validate_prescription` +
  `_reject_unsupported_jax_surfaces` closes it.
* `from_field.py`: odd-N fields place rays on the `ix − Nx//2` grid,
  a half-pixel off the library's `(arange−N/2)·dx` convention; the
  module docstring claims plane-wave exactness for `|kx·dx| < π`
  where the 2Δx phase ratio wraps at π/2 (the function docstring is
  correct); boundary pixels get a 2×-underestimated gradient
  (clipped neighbour over 2Δx); `_place_uniform`'s aspect correction
  assumes landscape orientation (anisotropic sub-grid on portrait
  fields); `_safe_sample`'s `clamp_floor/1e-300` can overflow to
  `inf` and make `0·inf = NaN` directions on alive rays for
  `|E|_max ≳ 1e12` beside hard-zero pixels (use
  `np.maximum(abs_v, clamp_floor)` instead).
* `_resolve_semi_diameters`: an explicit `'semi_diameter': None` in
  an `elements` entry raises `TypeError` on `None > 0` (the
  per-surface key guards `is not None`; the elements path doesn't —
  the NumPy twin shares the pattern, so at least the backends agree).

---

## 3. Coverage statement

**The raytrace subsystem is now fully line-audited.**  Fully read:
`intersection.py`, `paraxial.py`, `surface.py`, `core.py`,
`layout.py`, `world_trace.py`, `bundles.py`, `world.py`,
`seidel_analysis.py`, `ray_fan.py`, `from_field.py`, `jax_trace.py`;
`trace.py` lines 1–520 (the `trace()` engine, `validate_prescription`,
`surfaces_from_prescription`) plus the ray-generator block
(`make_ray`/`make_fan`/`make_ring`/`make_grid`/`make_rings`, read for
RT-5) — the remaining tail (`apply_doe_phase_traced`,
`trace_prescription`, `surfaces_from_elements`, `raytrace_system`)
covered structurally; `seidel.py` deep-read (`system_abcd`,
`lens_abcd`, `seidel_coefficients` — the physics core — plus the dead
paraxial-trace trio; `first_order_data`/`compute_pupils`/`find_lenses`
bodies at 551–853 covered structurally, their outputs already
cross-checked via `image_plane_wfe`/`field.py` consumers).
`differential.py` fully audited in the v5.21 delta round;
`raytrace/__init__.py` is re-exports (walker-symmetry-tested).  All
modules retain the 2026-06-10 structural coverage and the 07-01
deep-sweep remediations (9 raytrace findings, CHANGELOG-verified).

---

*Audit performed single-context against lumenairy v5.21, 2026-07-08.
Companion docs: `AUDIT_V5_21_DELTA_2026_07_07.md`,
`AUDIT_SOURCES_CORE_2026_07_07.md`,
`AUDIT_ANALYSIS_METRIC_CORE_2026_07_07.md`,
`AUDIT_PROPAGATORS_KERNELS_2026_07_07.md`.*
