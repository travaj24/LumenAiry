# LumenAiry Migration Guide

This guide collects user-facing breaking changes and non-obvious
default-shifts across LumenAiry versions.  The CHANGELOG is the
per-release record; this guide is the **actionable** version-spanning
migration recipe -- "I bumped from v4.X to v4.Y, what do I change?".

## Versions covered

This guide covers v4.x.  v5.0 will introduce its own migration
section when released.

Within v4.x, only behavior shifts that **require user code changes**
or **change numerical answers** are listed.  Pure additions (new
functions, new optional-dependency extras, performance improvements)
do not appear here -- consult the CHANGELOG for the full per-release
record.

---

## 4.13.0 -- `rcwa.py` -> `thin_grating.py` rename

The thin-grating subpackage was renamed to better reflect its scope
(the module never implemented full Rigorous Coupled-Wave Analysis;
it implemented a phase-only thin-grating model).

**Affected imports:**

```python
# Old (pre-4.13.0):
from lumenairy.elements.rcwa import apply_thin_grating, GratingProfile

# New (4.13.0+):
from lumenairy.elements.thin_grating import apply_thin_grating, GratingProfile
```

Behavior preserved bit-for-bit; only the import path moved.

## 4.13.0 -- `wavelength` now required on codegen helpers

Several prescription-builder helpers (`make_singlet`, `make_doublet`,
`make_zemax_singlet`, etc.) previously inferred wavelength from
ambient context; v4.13.0 requires an explicit `wavelength=` kwarg.
The context-inferred path was a frequent source of silent
substitution bugs when a user later switched the ambient context
without re-checking the prescription.

**Recipe:**

```python
# Old (pre-4.13.0):
pres = la.make_singlet(R1=50e-3, R2=-50e-3, glass='N-BK7', thickness=5e-3)

# New (4.13.0+):
pres = la.make_singlet(R1=50e-3, R2=-50e-3, glass='N-BK7',
                       thickness=5e-3, wavelength=587.6e-9)
```

## 4.15.1 -- Schell-model factories return ensemble, not MCF

The Schell-model partial-coherence factories
(`create_gaussian_schell_source`, `create_schell_model_source`,
`create_annular_incoherent_source` + their `Source.*` classmethod
wrappers) changed their default return shape from a Mutual Coherence
Function (MCF) object to a `(n_realizations, Ny, Nx)` ensemble
array.

v4.15.1 emitted a `DeprecationWarning` on the default-path call;
v4.16.1 retired the warning (the new default has had 4 releases of
exposure).

**Recipe -- if your code depended on the MCF return shape:**

The MCF return path was never landed beyond the v4.15.0 prototype;
v5.0 will introduce an `MCF` object with a `coherence_at(...)`
method.  In the interim, use the ensemble path + the new
`propagate_ensemble(...)` helper (added in v4.16.1):

```python
ensemble = la.create_gaussian_schell_source(
    Ny=256, Nx=256, dx=5e-6, n_realizations=64,
    w0=50e-6, sigma_g=15e-6, wavelength=633e-9,
)
I_partial = la.propagate_ensemble(
    ensemble, dx=5e-6, wavelength=633e-9,
    propagator='asm', z=0.10,
)
```

The Wolf `I_partial = <|E_k|^2>_k` formula is the canonical Schell-
model partial-coherence intensity (Goodman, *Statistical Optics*,
Sec 5.5).

## 4.16.1 -- `MultiWavelengthMerit` semantics: SUM -> AVG

`MultiWavelengthMerit.evaluate` changed from **summing** the
sub-merit value across wavelengths to **averaging** it (dividing by
`len(self.wavelengths)`).  The pre-v4.16.1 behavior contradicted
both the class docstring and the two sibling wrapper classes
(`MultiFieldMerit`, `ToleranceAwareMerit`), which already averaged.

**This change silently shifts numerical answers** for existing
3-wavelength configs by a factor of `len(wavelengths)`.  v4.16.2
adds a one-cycle `FutureWarning` on first call to alert users; the
underlying semantics change happened in v4.16.1.

**Recipe -- if your `design_optimize` recipe was tuned against
pre-v4.16.1 merit values:**

Re-scale `weight` by `len(wavelengths)`:

```python
# Old recipe (pre-4.16.1) that gave good results:
merit = la.MultiWavelengthMerit(
    sub_merit=spot_size_merit,
    wavelengths=[450e-9, 550e-9, 650e-9],
    weight=1.0,
)
# Equivalent v4.16.1+ recipe (preserves numerical answer):
merit = la.MultiWavelengthMerit(
    sub_merit=spot_size_merit,
    wavelengths=[450e-9, 550e-9, 650e-9],
    weight=3.0,   # was 1.0; multiply by len(wavelengths)
)
```

Alternatively: re-tune `weight` against the new AVG semantics.  The
new behavior is independent of wavelength count, so the same
`weight` value generalizes cleanly across configs with different
wavelength sets.

## 4.16.1 -- `refractiveindex` moved to optional `[glass]` extras

`refractiveindex>=1.0` was demoted from a hard runtime dependency
to an optional extras group.  Minimal installs no longer pull it,
and the library transparently falls back to bundled Sellmeier
coefficients (and v4.16.2+ formula-3 evaluator) for the ~46 glasses
with bundled fits.

**Recipe -- if your code uses Hikari E-/J-, Sumita K-, or formula-3
CDGM glasses:**

```bash
pip install lumenairy[glass]
```

The 46+ glasses with bundled Sellmeier coefficients (Schott N-,
Ohara S-, CDGM Sellmeier-2 entries, and v4.16.2+ formula-3 entries
as they're ingested) work on a minimal install without the
extras.

## 4.16.2 -- Default-config knobs (API-only; v5.1 rollout)

Three new library-wide setter functions land alongside the existing
`set_default_complex_dtype`:

* `set_default_real_dtype(np.float32 | np.float64)` -- real-array
  precision.  v4.16.3 wires this knob through `propagate_ensemble`'s
  no-input-dtype real-accumulator fallback (the canonical
  `in_dtype is None` path); full library-wide rollout follows in v5.0.
* `set_default_wave_propagator(name)` -- default `wave_propagator`
  for `propagate_through_system` / `apply_real_lens` / etc.
  **API ONLY in v4.16.2 through v5.0.x**: stored but not yet read by any
  library code.  Setter emits a one-shot `UserWarning` (v4.16.3+).
* `set_default_dy(value)` -- default anamorphic grid spacing.
  **API ONLY in v4.16.2 through v5.0.x**: same status as
  `set_default_wave_propagator`.

> **v4.16.2 / v4.16.3 / v5.0 / v5.0.1 limitation note.**  Two of the three new knobs
> (`set_default_wave_propagator`, `set_default_dy`) store the default
> but have **zero downstream consumers** in `lumenairy/` at v4.16.2
> ship.  Entry points like `apply_real_lens` continue to hardcode
> `wave_propagator='asm'` and accept `wave_propagator=...` / `dy=...`
> as per-call keyword arguments.  The library-wide resolver rollout
> that makes these setters actually steer the default at every entry
> point is staged for v5.1 alongside the file-split work.  Until
> then, the setters store the value the getter reads back, but
> downstream propagator dispatch is unaffected.  v4.16.3+ surfaces
> this honestly via a one-shot `UserWarning` from each setter.

**Recipe -- precision knob (the one with real consumers):**

```python
import numpy as np
import lumenairy as la

# Old (per-call dtype every entry point):
field = la.apply_real_lens(field, prescription=pres, wavelength=wl,
                            dx=dx, wave_propagator='fresnel')

# New (one-shot at app initialization).  `set_default_complex_dtype`
# is honored library-wide; `set_default_real_dtype` is honored at the
# `propagate_ensemble` no-input-dtype real fallback site (v4.16.3+):
la.set_default_complex_dtype(np.complex64)
la.set_default_real_dtype(np.float32)

# Per-call `wave_propagator=` still required until v5.0 (the setter
# is API-only at v4.16.2 / v4.16.3 / v5.0 / v5.0.1; see limitation note above):
field = la.apply_real_lens(field, prescription=pres, wavelength=wl,
                            dx=dx, wave_propagator='fresnel')
field = la.apply_real_lens_traced(field, ..., wave_propagator='fresnel')
```

The full library-wide resolver rollout (replacing hardcoded `'asm'`
/ `np.float64` / `dx` defaults at every entry point) is staged for
v5.0 alongside the file-split work.

---

## 5.0.0 -- Major structural release

v5.0 is the coordinated breaking-change release.  Each item below
documents what changed, why, and how to migrate user code.

### 5.0.0 -- Python 3.9 dropped (`requires-python = ">=3.10"`)

Python 3.9 reached end-of-life on 2025-10.  v5.0 bumps the floor to
3.10.  Users still on 3.9 see a `pip` install-time error and must
upgrade their interpreter.

No code change required if you're already on 3.10+.

### 5.0.0 -- `lumenairy/system.py` -> `lumenairy/propagators/system.py`

The sequential-propagation entry points (`propagate_through_system`,
`propagate_through_system_jax`, the JAX cache primitives) functionally
ARE a propagator -- they walk elements applying per-element
propagators.  v5.0 moves them under `lumenairy/propagators/` so the
package layout matches the conceptual role.

**Public-namespace users see no change.**  `import lumenairy as la`
and `la.propagate_through_system(...)` continue to work bit-for-bit.

**If you imported the private path directly:**

```python
# Old (pre-5.0):
from lumenairy.system import propagate_through_system
from lumenairy.system import _PROPAGATE_SYSTEM_JAX_CACHE  # private

# New (5.0+):
from lumenairy import propagate_through_system   # (preferred)
# or
from lumenairy.propagators.system import propagate_through_system
```

### 5.0.0 -- Back-compat shim removal

The following shims that had been carried 3-9 releases past their
deprecation cycle are removed in v5.0:

* `lumenairy.analysis.analysis` -- v4.7 rename shim.  **Removed.**
  Migrate: `from lumenairy.analysis.analysis import X` ->
  `from lumenairy.analysis import X` (or `lumenairy.analysis.core`).
  Old path now raises `ModuleNotFoundError`.
* `lumenairy.ao` -- v4.3 shim.  **Removed.**  Migrate:
  `from lumenairy.ao import DeformableMirror` ->
  `from lumenairy.analysis.ao import DeformableMirror` (or the
  top-level re-exports such as `lumenairy.DeformableMirror`).
  Old path now raises `ModuleNotFoundError`.
* `lumenairy.io.hdf5` -- shim.  **Removed.**  Migrate:
  `from lumenairy.io.hdf5 import save_field_h5` ->
  `from lumenairy.io.storage import save_field_h5` (or
  `from lumenairy import save_field_h5`).  Old path now raises
  `ModuleNotFoundError`.
* `propagate_through_system_jax` legacy aperture schema.
  **Removed.**  Pre-v4.12 aperture element params used
  `radius` / `half_width_x` / `inner_radius`; v4.12 deprecated
  them in favour of the canonical NumPy schema
  (`diameter` / `width_x` / `inner_diameter`).  v5.0 removes the
  legacy keys; they now raise `ValueError` with the migration
  recipe inline.  Migration: double the value and rename
  (`radius=r` -> `diameter=2*r`, `inner_radius=ri` ->
  `inner_diameter=2*ri`, etc.).
* `apply_detector(..., cosmic_ray_rate=...)` -- v4.9
  deprecated kwarg.  **Removed.**  The legacy `cosmic_ray_rate`
  did not scale with detector area or exposure time.  Migrate:
  `cosmic_ray_rate=R` -> `cosmic_ray_rate_per_m2_per_s=R/A/T`
  where `A = (n_pixels * pixel_pitch)^2` is the detector area
  and `T` is the exposure time.  Typical sea-level reference
  value ~1 /m^2/s.  Old kwarg now raises `TypeError`
  (unexpected keyword argument).

**Shims preserved as legitimate public API surface** (not removed
despite ROADMAP suggestion -- they're useful re-exports, not
deprecation shims):
* `lumenairy.elements.lenses.apply_*_lens` re-exports.  These
  provide a coherent one-stop import surface for lens-related
  functions; the underlying split into `_lens_thin.py` /
  `_lens_real.py` / `_lens_traced.py` is an internal
  organisational choice, not a user-facing API surface change.

---

## Items deferred from v5.0.0 to v5.1+

The ROADMAP originally scoped the following items into v5.0
alongside the breaking changes.  v5.0 ships the breaking changes
in isolation to keep the migration surface tight; the items below
follow in v5.1.x patch releases as time and review cycles allow.

* **6 large-file splits** (`raytrace/core.py`,
  `propagators/propagation.py`, `propagators/asymptotic.py`,
  `optimize/core.py`, `io/prescriptions.py`, `analysis/core.py`).
  No public API change -- mechanical reorganisation visible only
  to `git blame`.  Deferred so v5.0's diff stays reviewable.
* **Library-wide default-config knob resolver rollout.**  The 3
  v4.16.2 knobs (`set_default_wave_propagator`, `set_default_dy`,
  `set_default_real_dtype`) remain API-only at v5.0 ship.  The
  v4.16.3 one-shot `UserWarning` ("API-only; consumer wiring
  lands in v5.0") is **kept in place** for v5.0 and removed in
  v5.1 when the resolver rollout actually lands.
* **MCF `coherence_at(...)` object** -- the explicit-MCF
  alternative to the v4.16.1 `propagate_ensemble` path.
  Deferred to v5.1.  Current users continue with the ensemble
  helper.
* **Off-axis conic in surface frame** -- coordinate-frame
  transformation for tilted/displaced aspheres.  Deferred to
  v5.1.  Current `decenter` / `tilt` keys continue to apply in
  field frame as in v4.x.  **Landed in v5.2 as an opt-in
  `surface_frame=True` kwarg on `apply_real_lens` -- see the
  v5.2 section below.**
* **26 formula-3 glass coefficients** -- per-glass vendor-source
  ingestion of Hikari E-/J-, Sumita K-, 4 CDGM polynomial
  glasses.  `POLYNOMIAL_COEFFICIENTS = {}` remains empty at v5.0;
  the evaluator infrastructure shipped in v4.16.2 is unchanged.
  Per-glass ingestion is a v5.1 work item.
* **5 new examples** (multi-config / zoom, tolerancing,
  coronagraph workflow, AO closed-loop, ghost / stray-light).
  Deferred to v5.1.
* **57 audit-fix test-file consolidation.**  Mechanical merge of
  the `test_audit_fixes_v<X>_<Y>_*.py` files into topical
  homes.  No behaviour change.  Deferred to v5.1.

---

## 5.2.0 -- Off-axis conic in surface frame (`apply_real_lens`)

v5.2 closes the ROADMAP item "Off-axis conic in surface frame
(not just decenter+tilt)" deferred from v5.0.  The fix lands as
a **non-breaking opt-in kwarg** on
`lumenairy.elements._lens_real.apply_real_lens` (and the public
`lumenairy.apply_real_lens` re-export):

```python
E_out = la.apply_real_lens(
    E_in, prescription=prescription, wavelength=wl, dx=dx,
    surface_frame=True,            # v5.2 opt-in
)
```

The default `surface_frame=False` preserves v5.1 behavior
**bit-for-bit** -- existing callers see no numerical change.

### What changed

The per-surface `"decenter"` / `"tilt"` keys have always been
honored by `apply_real_lens`, but in v3.x -> v5.1 they were
applied as a **field-frame** coordinate shift plus a linear sag
ramp:

```
Xs = X - decenter_x
Ys = Y - decenter_y
sag(Xs, Ys) += tilt_x * Xs + tilt_y * Ys     # linear ramp
```

This is the textbook small-tilt / small-decenter alignment-
tolerance approximation but is **not** what Optiland and Zemax
do for a genuinely tilted / displaced asphere: those tools treat
the surface's sag as a rigid body and rotate / translate the
whole surface relative to the field's frame.  For a parabola
tilted by 5 mrad the difference is a fundamentally different
phase pattern -- a rotated parabola, not a parabola plus a
linear ramp.

The new `surface_frame=True` branch evaluates each surface's
sag in its own rigid-body-transformed local frame.  The field's
`(x, y)` grid is mapped to surface-frame coordinates via the
inverse rigid-body transform:

```
(x_s, y_s, z_s) = R^T @ (x_f - dcx, y_f - dcy, 0)
               where R = Rx(tilt_x) @ Ry(tilt_y)
```

The sag is then evaluated at `(x_s, y_s)` and contributes the
same `-k0 * (n2 - n1) * sag(x_s, y_s)` phase as the field-frame
branch -- only the coordinate at which sag is evaluated changes.
The full rotation matrix is used (no small-angle linearisation),
so arbitrary tilts are handled correctly.

### When to use each branch

* **Default `surface_frame=False`** -- use for small-tilt /
  small-decenter alignment-tolerance studies where the linear
  sag ramp is the textbook physics, and for backward
  compatibility with v5.1 and earlier results.

* **`surface_frame=True`** -- use for off-axis aspheres
  (OAP-style mirrors expressed as a refractive surface,
  decentered parabolic correctors, etc.) and any system where
  the surface's coordinate frame meaningfully differs from the
  field's grid frame.  Cross-checks against Optiland / Zemax
  results on the same prescription require this branch.

### Recipe -- no code change required (default preserved)

If your v5.1 code does not pass the new kwarg, nothing changes:

```python
# v5.1 -- still works in v5.2, bit-for-bit identical
E_out = la.apply_real_lens(
    E_in, prescription=presc, wavelength=wl, dx=dx)
```

### Recipe -- opt in to surface-frame physics

```python
# v5.2 -- evaluate sag in each surface's rigid-body frame
E_out = la.apply_real_lens(
    E_in, prescription=presc, wavelength=wl, dx=dx,
    surface_frame=True)
```

### Caveats and scope

* The opt-in lives on `apply_real_lens` only.
  `apply_real_lens_traced` already honors surface-frame
  transforms via its raytrace phase leg (Optiland-equivalent
  coord-break handling in `lumenairy.raytrace.world` /
  `intersection.py`), so no parallel kwarg is needed there.
  `apply_real_lens_maslov` predates this work and continues to
  use the field-frame approximation.

* The surface-frame branch evaluates sag at the surface-frame
  footprint of the field-plane normal, dropping `z_s`.  This is
  the same thin-element approximation the field-frame branch
  makes; full per-pixel intersection requires
  `apply_real_lens_traced`.

* `"form_error"` maps are still treated in the field frame
  (i.e. the measured figure error is applied to the same
  `(x_s, y_s)` grid as the sag).  A future enhancement could
  resample the form-error map under the rigid-body transform.

---

## 5.2.0 -- `output_grid` -> `output_shape` rename on GBD / HFPI / HF sub-propagators (AUDIT_V4_13_1 Part 2 P1-A)

The `output_grid` kwarg on the prescription-aware sub-propagators
collided with the dispatcher's `propagate(output_grid=...)` contract:

* Dispatcher: `output_grid = (N_out, dx_out)`  (grid spec)
* Sub-propagators: `output_grid = (Ny, Nx)`    (shape only)

v5.2 keeps the dispatcher contract as canonical and renames the
sub-propagator kwarg to `output_shape` for the shape-only meaning.
The legacy `output_grid` spelling on the sub-propagators still works
but emits a `DeprecationWarning` directing the caller to either the
new `output_shape=(Ny, Nx)` kwarg or the dispatcher path
`propagate(output_grid=(N_out, dx_out))`.

**Affected sub-propagators (5):**

* `lumenairy.propagators.gbd.propagate_gbd_freespace`
* `lumenairy.propagators.gbd.propagate_gbd_thin_lens`
* `lumenairy.propagators.gbd.propagate_gbd_through_prescription`
* `lumenairy.propagators.hfpi.propagate_hfpi_freespace_aperture`
* `lumenairy.propagators.hfpi.propagate_hfpi_through_prescription`
* `lumenairy.propagators.hf.propagate_huygens_fresnel_through_prescription`

**Recipe (legacy -> v5.2+):**

```python
# Old (pre-5.2, deprecated in v5.2):
out = la.propagate_gbd_freespace(
    E, dx=dx, z=z, wavelength=wl, output_grid=(Ny, Nx))

# New (v5.2+):
out = la.propagate_gbd_freespace(
    E, dx=dx, z=z, wavelength=wl, output_shape=(Ny, Nx))

# Or via the dispatcher (canonical (N_out, dx_out) form):
out = la.propagate(
    E, method='gbd', z=z, wavelength=wl, dx=dx,
    output_grid=(N_out, dx_out))
```

> **Known dispatcher-forwarding caveat (v5.2).**  The dispatcher
> forwards its `output_grid=(N_out, dx_out)` value directly to the
> sub-propagator's `output_grid=` legacy kwarg, which will emit a
> `DeprecationWarning` and then mis-interpret the tuple as
> `(Ny=N_out, Nx=dx_out)`.  Calling the dispatcher with
> `output_grid=(N, dx_out)` was already wrong physics pre-v5.2; the
> v5.2 rename surfaces the issue as a warning but does not yet fix
> the dispatcher's forwarding.  Until the dispatcher closure ships
> (v5.2.1 candidate), call the sub-propagators directly with
> `output_shape=(Ny, Nx)` for shape-only resampling.

---

## 5.2.0 -- `prescription_subdomain(method='maslov')` raises on grid mismatch (AUDIT_V4_13_1 Part 2 P1-C)

Pre-v5.2 `lumenairy.propagators.mhs.prescription_subdomain` with the
default `method='maslov'` silently returned the propagation on the
INPUT grid, ignoring the output Huygens surface's declared grid.
v5.2 raises `ValueError` at subdomain construction time when the
input and output grids differ, instead of silently dropping the
request.

**Recipe -- if you relied on the silent same-grid behaviour:** no
change needed (the input == output case still works).

**Recipe -- if you legitimately need a different output grid through
a `maslov` stage:**

```python
# Old (pre-5.2 -- silently returned input-grid output):
sub = la.prescription_subdomain(in_surf, out_surf, presc,
                                 wavelength=wl, method='maslov')

# v5.2 -- pick a method that natively supports output-grid
# resampling, or stage an explicit asm resampling step:
sub_gbd = la.prescription_subdomain(in_surf, out_surf, presc,
                                     wavelength=wl, method='gbd')

# OR: maslov on same-grid, then an asm resampling subdomain:
sub_maslov = la.prescription_subdomain(in_surf, in_surf, presc,
                                        wavelength=wl, method='maslov')
sub_resample = la.asm_subdomain(in_surf, out_surf, z=0.0,
                                 wavelength=wl)
```

---

## 5.2.0 -- `propagate_subaperture_asymptotic` UserWarning on non-unit magnification (AUDIT_V4_13_1 Part 2 P1-F)

Pre-v5.2 the partition-of-unity windows in
`lumenairy.propagators.subaperture.combine_patch_fields` were
centred on `patch_grid.centres`, which are SOURCE-plane positions.
This is only correct for unit-magnification, no-tilt geometries; a
system with magnification `|A| != 1` maps each source patch to an
image-plane footprint at `|A| * (cx, cy)` with corresponding scaled
half-widths, and the legacy code window tiles the wrong location.

v5.2 surfaces the limitation as a `UserWarning` when the system's
paraxial ABCD `|A - 1| > 0.05`, plus exposes two new optional kwargs
on `combine_patch_fields` -- `image_centres` and `image_half_widths`
-- so callers with knowledge of the system magnification + tilt can
supply the image-plane partition coordinates explicitly.

**Recipe -- if you ran subaperture decomposition on a magnifying
system before v5.2:** the result was unreliable at off-axis patches.
Either suppress the warning if you accept the limitation, or supply
image-plane centres:

```python
# Compute image-plane centres / half-widths from the system ABCD:
abcd = la.system_abcd_prescription(presc, wavelength)
M = abcd[0] if isinstance(abcd, tuple) else abcd
mag = abs(float(M[0, 0]))
img_centres = pg.centres * mag
img_half_widths = pg.half_widths * mag

# Build the per-patch fields the usual way (legacy
# propagate_subaperture_asymptotic doesn't yet route image_centres
# through; for v5.2 you must call combine_patch_fields directly):
out = la.combine_patch_fields(
    patch_fields, pg,
    output_grid_x=ox, output_grid_y=oy,
    image_centres=img_centres,
    image_half_widths=img_half_widths,
)
```

The full fix -- automatically computing the image-plane mapping
inside `propagate_subaperture_asymptotic` so the warning never
fires for legitimate magnifying systems -- is tracked as a v5.2.1
candidate per ROADMAP.

---

## 5.3.2 -- maintainers: CHANGELOG ship-time stamping is now a
pre-tag step

This is NOT a user-facing migration -- it's a maintainer-side
release-process note.  Library callers see no behaviour change.

Pre-v5.3.2, each CHANGELOG entry self-cited build-time empirical
numbers (test counts, file counts, line counts) that were always
at-write-time, never at-ship-time.  v5.3 surfaced this drift class
via the V17 walker
(`tests/unit/test_v5_3_walker_changelog_self_citation.py`).  v5.3.2
ships the FIX side: `scripts/stamp_changelog.py` stamps the topmost
CHANGELOG block with current empirical values just before tag commit.

**Maintainer recipe at tag time:**

```
1. Write the CHANGELOG entry with at-write-time placeholders.
2. ``git add CHANGELOG.md`` along with the other release files.
3. ``python scripts/stamp_changelog.py --quick --apply``
4. ``git add CHANGELOG.md`` again.
5. ``git commit ... && git tag ... && git push ...``
```

The script is dry-run by default; `--apply` is required to actually
rewrite `CHANGELOG.md`.  Full release-process documentation lives in
[`docs/release-process.md`](docs/release-process.md).

## 5.3.2 -- opt-in telemetry logging

v5.3.2 adds **opt-in** per-iteration telemetry logging to three
long-running paths the audit cited as lacking progress visibility:

* `apply_real_lens_traced` -- per-Newton-iteration markers
  (`apply_real_lens_traced: newton iter k/N residual_max=... m
  remaining=.../...`)
* `design_optimize` -- per-scipy-iteration markers
  (`design_optimize: iter k/N merit=... efl=...mm`)
* `monte_carlo_tolerancing` (+ its `_jax` twin) -- per-trial markers
  (`monte_carlo_tolerancing: trial k/N strehl_peak=...`)

This is **purely additive**.  No `warnings.warn` call was converted
to `logger.warning`; the deprecation / sampling-violation warning
surface is part of the public API contract and is unchanged.  No
default behaviour shifts.  No numerical answer changes.

**Default behaviour: SILENT.**  The library's `'lumenairy'` root
logger gets a `NullHandler` at import time, so a fresh program that
calls `apply_real_lens_traced` / `design_optimize` /
`monte_carlo_tolerancing` will see NO new log output -- the same as
in v5.3.1 and earlier.

**Opt-in recipe:** attach a handler to the `'lumenairy'` logger (or
a sub-logger) and set the level to `INFO`:

```python
import logging
import lumenairy as la

# Easiest: route lumenairy INFO records to stderr via the root
# logger's basicConfig.
logging.basicConfig(level=logging.INFO)

# Now any apply_real_lens_traced / design_optimize / MC call emits
# per-iteration progress as INFO records:
result = la.design_optimize(...)
# INFO:lumenairy.optimize.driver:design_optimize: entry method=L-BFGS-B ...
# INFO:lumenairy.optimize.driver:design_optimize: iter 1/100 merit=... efl=...mm
# INFO:lumenairy.optimize.driver:design_optimize: iter 2/100 merit=... efl=...mm
# ...
```

For larger Monte Carlo runs where per-trial output is noisy,
silence the per-trial logs by raising just the lumenairy logger to
`WARNING`:

```python
logging.getLogger('lumenairy').setLevel(logging.WARNING)
```

The entry-summary log on each of the three paths uses `INFO`, so
raising to `WARNING` silences both entry and per-iteration logs --
which is the right behaviour for n_trials > 100 cases the audit
specifically called out.

**No code changes required** to keep pre-v5.3.2 behaviour -- without
a user-attached handler, the `NullHandler` absorbs the records and
nothing prints.

