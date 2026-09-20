# LumenAiry Migration Guide

This guide collects user-facing breaking changes and non-obvious
default-shifts across LumenAiry versions.  The CHANGELOG is the
per-release record; this guide is the **actionable** version-spanning
migration recipe -- "I bumped from v4.X to v4.Y, what do I change?".

## Versions covered

v4.13 through v5.49.  Sections are in version order; the newest is
[5.49.0 -- the default flips (2026-09-20)](#5490----the-default-flips-2026-09-20)
at the end of this file, which is where the settings the 2026-09-11 audit
measured but did not move were decided; the 5.46.0 section is the largest
single batch of behaviour changes the library has shipped, and 5.47.0 is the
wave that implemented what it deferred.

Only behavior shifts that **require user code changes** or **change
numerical answers** are listed.  Pure additions (new functions, new
optional-dependency extras, performance improvements) do not appear
here -- consult the CHANGELOG for the full per-release record.

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

**Deprecated Zemax-loader aliases NOT removed in v5.0** (removal
realigned to v6.0; AUDIT_V5_24_2 S4-17): the v4.7 aliases
`load_zmx_prescription` (-> `load_zemax_zmx`) and
`load_zemax_prescription_txt` (-> `load_zemax_prescription_data_txt`)
were tagged `version_removed='5.0'` but were omitted from the v5.0
shim purge above and have shipped ever since.  Rather than break
callers 24 releases late, the removal target is moved to **v6.0** and
the emitted `DeprecationWarning` now reads "will be removed in v6.0".
Migrate at your convenience: call the canonical
`load_zemax_zmx(...)` / `load_zemax_prescription_data_txt(...)`.

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

> **Superseded in part by v5.46 (audit L2 / L19).**  From v5.2 to v5.45 the
> `surface_frame=True` branch discarded the rotated surface's field-frame
> height, so it deleted the tilt entirely (a tilted flat face deviated the beam
> by 0.000 mrad).  v5.46 fixes that and unifies the `tilt` reading across both
> branches, so the numbers below move.  See
> [5.46.0 -- adversarial audit remediation](#5460----adversarial-audit-remediation-2026-09-11).

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


---

## 5.30.0 -- Deprecation-shim removals (W5 wave)

Ten deprecation shims are **removed** in v5.30.  Eight had already blown
through a stated removal version while continuing to ship
(`version_removed='5.0'` at v5.29 — 29 minor releases late — or `'5.27'`
for the v5.25 kwarg renames); the horizons were re-scheduled to v5.32 and
then executed here rather than slipping a third time.

Every *modern* call path is unchanged, bit-for-bit: 73 captured arrays are
byte-identical to the pre-removal commit, and a 42-entry SHA-256 subset is
frozen into `tests/unit/test_niche_audit_w5_shim_removals.py`.

**Errors you may hit, and the fix**

| Old form | New form |
| --- | --- |
| `create_gaussian_beam(..., sigma=s)` | `create_gaussian_beam(..., w0=s*sqrt(2))` |
| `create_gaussian_schell_source(..., seed=k)` (and the other two Schell factories, `Source.gaussian_schell`, `Source.schell_model`) | `..., rng=k` — exactly equivalent |
| `Source.gaussian(w0, N, dx, wavelength)` | `Source.gaussian(*, N, dx, wavelength, w0)` |
| `Source.plane_wave(N, dx, wavelength)` | `Source.plane_wave(*, N, dx, wavelength)` |
| `Source.point_source(N, dx, wavelength)` | `Source.point_source(*, N, dx, wavelength)` |
| `Source.top_hat(diameter, N, dx, wavelength)` | `Source.top_hat(*, N, dx, wavelength, diameter)` |
| `Source.fiber_mode(mfd, N, dx, wavelength)` | `Source.fiber_mode(*, N, dx, wavelength, mode_field_diameter)` |
| `create_led_source(N, dx, diameter, divergence_angle, wavelength, ...)` | `create_led_source(N, dx, wavelength, *, diameter=..., divergence_angle=...)` |
| `create_gaussian_schell_source(..., return_kind=_RETURN_KIND_UNSET)` | omit `return_kind`, or pass `'ensemble'` / `'mcf'` |
| `makedammann2d(..., _legacy_units='auto')` | drop the kwarg (`'SI'` is the default) or pass `_legacy_units='um'` |
| `recommend_gbd_sampling(E, dx, wavelength=lam)` | `recommend_gbd_sampling(E, dx)` |
| `propagate_huygens_fresnel_with_opl_callable(..., wavelength=lam)` | drop the kwarg (`opl_fn` returns WAVES) |
| `design_optimize(..., wave_traced=True)` | `register_wave_propagator('real_lens_traced', fn)` + `wave_propagator='real_lens_traced'` |
| `MatchIdealSystemMerit(..., use_traced_lens=True, ray_subsample=n)` | an explicit `{'type': 'real_lens_traced', 'prescription': ..., 'ray_subsample': n}` entry in `real_elements` |
| `MatchIdealSystemMerit(..., focus_search=True, focus_search_range=r, focus_search_n=n)` | an explicit `{'type': 'propagate', 'z': dz}` offset in `ideal_elements` |

### `sigma` -> `w0` is a value conversion, not just a rename

`w0` is the **1/e² intensity radius** (the beam waist); the removed `sigma`
was the field *standard deviation*.  They relate as `w0 = sigma*sqrt(2)`:

```python
# Old (removed in 5.30):
E, x, y = la.create_gaussian_beam(N, dx, wavelength, sigma=50e-6)

# New -- same beam:
E, x, y = la.create_gaussian_beam(N, dx, wavelength,
                                  w0=50e-6 * np.sqrt(2))
```

Going the other way is exact: `w0=w` reproduces the old `sigma=w/sqrt(2)`
field bit-for-bit, because `w0/sqrt(2)` is literally the division the
factory performs.

### `_legacy_units='auto'` is gone because it was wrong, not just old

The v4.14.2 heuristic multiplied any `periodx` / `periody` / `waveln`
above 1 mm by `1e-6`, on the theory that such a value "must" be legacy
micrometres.  That is exactly backwards for a physical THz / MMW design:
an SI 8 mm period at 1.1 mm wavelength came back with 5e-10 m cells — a
factor 1e-6 wrong, and for five releases the only diagnostic was a
`DeprecationWarning`, which Python hides outside `__main__`.  v5.30 raises
`ValueError` instead.  If your inputs really are micrometres, say so with
`_legacy_units='um'`; mixed-unit calls must be normalised by the caller,
because no magnitude heuristic can tell which is which.

### The traced real-lens path is still available

`wave_traced` / `use_traced_lens` selected `apply_real_lens_traced` through
a boolean that also changed the meaning of `ray_subsample`.  R-17
grep-verified zero callers repo-wide, so CI never exercised those branches.
The propagator itself is untouched — register it explicitly:

```python
from lumenairy.optimize.driver import register_wave_propagator
from lumenairy import apply_real_lens_traced

def _traced(E0, pres, *, wavelength, dx, N, wp_kwargs, opts):
    return apply_real_lens_traced(
        E0, prescription=pres, wavelength=wavelength, dx=dx,
        ray_subsample=opts.get('ray_subsample', 4), n_workers=1,
        **wp_kwargs)

register_wave_propagator('real_lens_traced', _traced)
design_optimize(..., wave_propagator='real_lens_traced')
```

### Not removed

`propagate()`'s **return-contract transition** (P5) is a default *flip*, not a
shim removal — and it **executed in v5.30**: `propagate()` now returns a
`PropagationResult` by default.  Pass `return_result=False` for the legacy
shapes; that escape hatch is permanent and is not scheduled for removal.
`...with_opl_callable(chunk_output=)`, `rcwa_efficiency_1d_jax`,
`load_zmx_prescription` / `load_zemax_prescription_txt`, the
`output_grid` -> `output_shape` sub-propagator renames, `MultiFieldMerit`
scalar `field_angles` and `PMM2DStack` all keep working: none is at or past
its stated horizon.

---

## 5.46.0 -- adversarial audit remediation (2026-09-11)

The 2026-09-11 adversarial audit
([`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11.md`](docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11.md))
went through the library function by function with independent oracles.  The
remediation changes **numbers** in more places than any previous release, and
almost all of those changes are corrections: the old number was wrong.  This
section is the actionable list, grouped by area.  Each note names the audit
finding ID, what the call returned before, what it returns now, and the way
back where one exists.

Two general rules apply throughout:

* an option documented as *recommended* that the audit measured as broken has
  been fixed in place rather than removed, so the call site does not change --
  only the answer does;
* where the pre-fix behaviour is reproducible, the escape hatch is named in the
  note.  Where it is not (the answer was simply wrong), the note says so.

The per-finding detail, measurements and repro scripts live in
[`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/`](docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/);
the resolution table is
`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/RESOLUTION_STATUS.md`.
For the `apply_real_lens` family specifically, the living contract is
[`docs/subsystems/real_lens.md`](docs/subsystems/real_lens.md).

### Analytic lens -- `apply_real_lens`

* **`seidel_correction=True` (L1).**  The Seidel fan read the ray OPL at the
  last surface's SAG instead of at the exit vertex, injecting ~90 um*rho^2 of
  spurious defocus (the focus of a 115 mm doublet moved to 62 mm).  The fan now
  goes through the shared `at_exit_vertex()` transfer, is launched across the
  full clear aperture (+-0.999*r_pupil, was +-0.9) and the fitted screen is
  CLAMPED at the last radius the fan lands at rather than extrapolated.
  Measured on an 8 mm cemented doublet: exit OPD rms **173.6 -> 1.05 nm
  (165x)**; correction rms 34 221 -> ~1 118 nm.  *No way back and none wanted* --
  the flag's old output was the defect.  Note the scope limit in
  [`docs/subsystems/real_lens.md`](docs/subsystems/real_lens.md): on a fast
  element whose exit field does not fill the pupil the screen is normalised to,
  the wavefront still improves (267x on an f/2 biconvex) but the focal peak
  drops ~37 %.
* **`slant_correction=True` (L12).**  The screen used normal-referenced cosines
  and was 2.94x WORSE than the paraxial screen it is meant to improve, with the
  wrong sign of the O(theta^2) term.  It now implements the axial-translation
  identity `n2 cos(theta_i - theta_t) - n1`.  Against the exact one-facet
  eikonal on a single N-BK7 face: **940x better than paraxial** (0.0041 vs
  3.83 nm rms).  **The flag's numbers move for every caller**, and on some
  fixtures the old form scored *closer* by cancellation (the f/5 hammer fixture
  at dx = 6 um: old 14.7 um from truth, new 38.2 um) -- that cancellation was
  never a property you could rely on.  `slant_correction=False` (the default) is
  bit-identical to before.
* **`slant_correction=True` + `seidel_correction=True` is now refused (L20)**:
  both replace the same coefficient and stacking them double-counts.
* **`fresnel=True` (L13) -- power convention.**  The per-interface factor was
  `abs(t)**2`; it is now the POWER transmittance
  `T = (n2 cos theta_t)/(n1 cos theta_i) abs(t)**2`, which is the convention
  CONVENTIONS section 7 now states.  An element that starts AND ends in air at
  normal incidence is UNCHANGED (the `n2/n1` factors telescope) -- which is why
  every plate fixture passed before.  Anything that ends in glass, any bare
  cemented interface, and the `cos theta_t / cos theta_i` half at finite NA all
  move: a single AIR -> N-BK7 face **0.632344 -> 0.958057 (+51 %)**, a bare
  cemented N-BK7 -> N-SF11 interface 0.846390 -> 0.993599.  The shipped
  per-interface loss is now the ~4.2 % the docstring always claimed.
* **`surface_frame=True` (L2, L19).**  The rigid-body branch evaluated the sag at
  the rotated transverse footprint and DISCARDED the rotated surface's
  field-frame height, so a tilted flat face deviated the beam by exactly
  0.000 mrad.  It now imprints the rotated height and reads the `tilt` key the
  way every other consumer does.  Measured: tilted flat N-BK7 face **0.000 ->
  2.537 mrad** against a 2.538 mrad thin-prism oracle; field-frame height error
  on R = 50 mm at 5 mrad **10.011 um (8.15 waves) -> 10.3 nm (0.0084 waves)**.
  Flipping `surface_frame` no longer re-points the element.  **No stored `tilt`
  value needs migrating** -- the key's meaning is unchanged (`tilt = (t0, t1)`
  IS the sag ramp `t0*x + t1*y`); it is the surface-frame branch that was
  reading it wrongly.  See the updated row in CONVENTIONS section 7.  (The
  branch is no longer sold as the "more accurate" one: measured against the
  exact rigid-body height it is marginally *worse* than the default field-frame
  ramp -- 1.03e-08 vs 9.10e-09 m at 5 mrad, both under 0.13 waves.  Use it when
  "rigid body" is the geometry you have, not for accuracy.)
* **`surface_model='displaced'` (L3, L7).**  The remaps discarded the input
  field's phase (they resampled `abs(E)`) and referenced the exit leg through
  air.  A phase-carrying input now transports correctly; a real non-negative
  input with `conjugate=None` reduces to the old amplitude sample exactly.
* **`absorption=True` (L19).**  Attenuation was applied over the AXIAL gap
  thickness instead of along the local ray column; apodisation error
  1.52e-2 -> 4.2e-4 (36x).
* **An out-of-range `stop_index` now RAISES (L14).**  It used to remove ALL
  aperture clipping silently.  `stop_index=-1` now means the LAST surface, as
  Python indexing does.  The same validation is applied by
  `apply_real_lens_traced` and `apply_real_lens_maslov`, so the family
  diagnoses a malformed key identically.  If you relied on an out-of-range value
  to disable clipping, remove the `aperture_diameter` key instead.
* **`screen_obliquity` with `carrier='auto'` or an ndarray carrier (L4)** no
  longer double-counts `n1`.
* **Two `displaced` caches returned stale results (L5, L6)** -- keyed on a
  callable's identity and on a glass NAME.  Results change only where they were
  wrong.

### Traced lens and the caustic siblings -- `apply_real_lens_traced`

* **`caustic='multibranch'` / `'uniform'` (S1) evaluated on the last surface's
  SAG.**  Both now read the exit-vertex plane.  Against an independent
  vector-Snell trace on a plano-convex with `R2 = -25 mm`: transverse error
  **476.8 um -> 1.7e-12 um**, OPD **3 501 waves -> 4.4e-12 waves**.  Every
  physics fixture for this layer used a plano-rear singlet (`sag == 0`), which
  is why the pre-fix code passed 8/8.  `output_plane_distance = 0.0` -- the
  DEFAULT -- was affected as much as a through-focus plane.
* **`caustic='wave'` is new** (the ray-to-wave hand-off): single-valued traced
  field at the exit vertex, then one band-limited ASM leg.  At `d = 0` it is
  bit-identical to the ordinary traced call.  It is NOT the default for
  `output_plane_distance != 0` (that would move every pinned multibranch
  number), but it is the recommended mode for new work at any plane where the
  ray map is multi-valued.
* **A surface `form_error` is now honoured (T5).**  The traced model silently
  cancelled it out of the answer; `apply_real_lens_traced` on a prescription
  carrying `form_error` now returns a DIFFERENT (correct) field.  Code that
  relied on the traced model ignoring `form_error` should drop the key.
  `caustic='multibranch'` / `'uniform'` REFUSE a `form_error` prescription --
  they are a pure ray construction with no analytic leg to re-apply the screen
  onto.
* **Vignetting is now in the answer (VERIFY-A3 OI-11).**  A prescription that
  vignettes rays with a per-surface `semi_diameter` / `clear_aperture` now
  returns LESS power -- the vignetting used to be fitted over.  Nothing moves
  when no ray dies, nor when the dead rays are outside the clear aperture.  The
  CuPy branch (`use_gpu=True`, polynomial only) keeps the historical
  extrapolating behaviour, because the rejection kernel is NumPy.
* **The exit-NA undersample guard now measures the EXIT medium (VERIFY-A3
  OI-3).**  `n_exit == 1` on every air-ending prescription, where the test, the
  message and the reported fraction are BIT-IDENTICAL to before.  A prescription
  ending in glass (a cemented interface left open, an immersed sensor, a
  sub-assembly handed to another element) will now see the undersample
  `RuntimeWarning` fire where it did not, and will be asked for a grid `n_exit`
  times finer.  That is the physically required sampling; pass
  `on_undersample='silent'` to acknowledge it deliberately.
* **`newton_fit='spline'` + a vignetted ray, `fast_analytic_phase=True`,
  `carrier=<ndarray>` and `caustic='multibranch'` at an axial focus (T2, T3, T4,
  T12)** each returned an identically-zero field, raised, or sampled by nearest
  neighbour.  All four now work; none has an escape hatch to the broken
  behaviour.
* **`_reverse_prescription` did not negate the aspheric / freeform / tilt /
  `sag_callable` departures and mis-paired the thicknesses (T6)**, so any
  reversed-element call on a non-spherical prescription moves.
* **`on_noncollimated='delegate'` could swap models in silence (T16)**; it now
  says so.
* **Angular segmentation may return FEWER segments (VERIFY-A3 OI-4)** for a
  field whose spectrum has structure outside its own 0.995-power support.  At
  the default `min_segment_power=1e-3` the extra bins were dropped as empty
  anyway.

* **Decentred fits, `beam_centre` off axis (WP-A26, full-run follow-up).**  The
  polynomial order the off-centre ray fit is given (`_DECENTRED_FIT_POLY_ORDER`)
  is 16, was 10.  The 10 had been calibrated against a ray set the tracer was
  truncating at `|h| = |R|` on conic surfaces; WP-A1 stopped that truncation
  (correctly: the resurrected rays agree with an exact conic trace to 2e-19 m),
  the fit's data domain grew 2.4x on a fast conic, and at order 10 the decentred
  exit wavefront read **44.5 / 31.6 urad** of slope error against the analytic
  Fermat sphere at 0.5 / 1.0 beam radii of decentre.  At 16 it reads
  **2.37 / 1.68 urad** (2.16 / 1.96 before the truncation stopped).  Concentric
  calls are byte-identical; a decentred call pays ~1.56x at the fit.  *Way back:*
  `decentred_fit_poly_order=10`; a caller's value above the constant is honoured
  as before.

### Maslov / GBD / FGA and the asymptotic family

* **`apply_real_lens_maslov(normalize_output='none')` (S4)** now returns a
  physically-scaled field: its absolute amplitude changes by
  `1/(i lambda |det ds1/dv2|^(1/2))` relative to v5.45 (Van Vleck amplitude and
  the `k/(2 pi i)` prefactor).  The default `normalize_output='power'` is
  unaffected except for the small real shape change from the corrected density
  (0.44 % of the `abs(E)` profile on a benign converging chart, 0.30 % at
  NA 0.13, unbounded on an aberrated one).
* **`integration_method='auto'` no longer selects `local_quadrature` (S2).**  It
  falls back to `stationary_phase` and warns.  Pass
  `integration_method='local_quadrature'` explicitly to keep the old routing
  (the integrator itself is now correct -- principal axes, a window, and the
  window divided back out); pass `'quadrature'` with an explicit `n_v2` for an
  exit-plane field, or `output_plane_distance=` to move the observation plane to
  where the asymptotics belong.
* **The Maslov chart is built on the exit VERTEX plane (S3)** -- 31 waves of
  defocus on an f/19 singlet before.  `fold_split`'s free-space legs honour `dy`
  (S11).
* **`apply_real_lens_maslov` validates `stop_index` (WP-A2 hand-off)**: an
  out-of-range value now raises where it previously warned and continued.
* **`apply_real_lens_maslov_vector` applies ONE joint normalisation** to the
  Jones pair instead of normalising `E_x` and `E_y` independently (which forced
  the output polarization ratio back to the post-Fresnel input ratio).
* **GBD: the beamlet Gouy / Collins phase was conjugated (S5).**
  `gbd_asm_gouy_phase`, `gbd_field_to_asm` and `asm_field_to_gbd` are now no-ops
  (`0.0` and the identity), emit a `DeprecationWarning` and are scheduled for
  removal in v5.48.  **Migration: delete the call** -- a GBD free-space field
  already matches `angular_spectrum_propagate` in absolute phase.  For a general
  propagator-agnostic reconciliation use `match_global_phase`, unchanged.
* **`_lens_jax` (both JAX lens entry points) now REQUIRE
  `jax.config.update('jax_enable_x64', True)` (S7)** and refuse loudly without
  it.  With x64 off the pre-fix code returned complex64 for a complex128 input
  and its phase screen differed from the float64 one by 1.71e-05 waves rms /
  8.17e-05 waves p-v.  The NumPy `apply_real_lens_traced` /
  `apply_real_lens_maslov` are unaffected.
* **Every `L` entry of `aberration_tensor` and every pixel of
  `propagate_modal_asymptotic` changes scale by `1/(lambda^2 |det J|)` in
  `abs(.)^2` (Y2)** -- the Van Vleck--Maslov normalisation.  Measured factors on
  the two shipped fixtures: 8.362145e+16 (R1 = 60 mm, ap 12 mm) and
  1.486604e+17 (R1 = 500 mm, ap 4 mm), which reproduce the previously-pinned
  `abs(L_00)^2` values to 6 significant figures when divided back out.
  `AberrationTensorResult.van_vleck_weight` (new, `None` by default) exposes the
  applied factor, so a caller that needs the v5.45 scale can divide it back out.
* **The asymptotic family's v2-linear phase stays inside the integrand (Y1)**;
  `extract_linear_phase`'s default changed.
* **`LGAberrationMerit` / `make_lg_aberration_merit_jax` are now a
  dimensionless coupling** `abs(L)^2 / abs(L_ref(0,0))^2` referenced to the
  aberration-free twin of the same optic, on both backends (VERIFY-A4).  **Any
  absolute threshold, weight calibration, logged value or stopping tolerance
  tuned against the old merit must be re-derived**: the (0, 0) contribution
  moves from `-4.79e+14` (v5.46-pre) / `+0.9968` (v5.45) to a Strehl deficit in
  approximately `[-3e-03, 1]`.  A composite merit's RELATIVE channel weights are
  unchanged in meaning -- every channel is divided by the same per-field-point
  constant -- so an optimiser converges to the same design; only the printed
  numbers and any absolute stopping tolerance change.  `LGAberrationMerit`
  gained `strehl_branch` (default `'sigma'`).
  `lumenairy.aberration_free_reference_fit` is the new public helper that builds
  the reference.

### Propagator kernels

* **Richards--Wolf `E_z` had the wrong SIGN (K16).**  Any code that consumed
  `E_z` from `richards_wolf_focus` coherently -- a three-component
  superposition, a Stokes / spin calculation, an optical-force integral -- was
  wrong by the sign of that component and now is not.  Intensity-only consumers
  (`debye_wolf_psf`, `abs(E)^2` maps) are bit-identical.  The `pupil` coordinate
  convention is now documented (K10) and is in CONVENTIONS section 7: it is the
  physical exit-pupil coordinate, not the projected ray direction.
* **`rayleigh_sommerfeld_propagate` gained
  `kernel={'auto','transfer','spatial'}`, default `'auto'` (K9).**  The
  historical point-sampled Green's function CREATED energy whenever
  `z < 2 N dx^2 / lambda` (measured `P_out/P_in` = 21.44 at N = 64 / dx = 2 um).
  `'auto'` selects the exact frequency-domain RS-I transfer function below that
  threshold and the historical build at and above it, so **every call at or
  above the threshold is bit-identical to v5.45**; only the regime the audit
  measured as broken is re-routed.  `'spatial'` now RAISES inside its alias
  regime rather than returning the wrong field.  Reached by
  `propagate(method='rs')`, `propagate(method='hf')` and
  `propagate_huygens_fresnel`.  `rs_alias_free_distance(N, dx, wavelength)` is
  now public API for sizing a grid.
* **HFPI amplitudes change by orders of magnitude (K13, K18, and the cascaded
  case V1).**  The estimator is now the Huygens--Fresnel integral with the
  source-area normalisation; cascaded amplitudes were additionally low by
  `n_paths * z_to_aperture`.  Phase structure (fringe positions, interference
  contrast) is unaffected.  **Pass `normalisation='legacy'` to restore the raw
  path sum**; it MUST match the value passed to the accumulator, and the
  library's entry points thread one value to both.
  `apply_aperture_diffraction` and `apply_vector_aperture_diffraction` gained
  the same `normalisation` kwarg (`'physical'` default).  Code that
  re-normalised HFPI against an ASM reference should drop that step.
* **`rng=None` now draws fresh entropy on the HFPI family (K19)** instead of
  reusing a fixed stream.  **Pass an explicit `rng=` for reproducible runs**;
  every determinism test in the suite already did.
* **`wavelength` is keyword-required on `hfpi` / `vectorial_hfpi` (K12).**  Add
  `wavelength=` to any direct call.
* **`apply_vector_aperture_diffraction(vector_projection=...)` defaults to
  `True` (K17)** and the operation it names changed: the "vectorial" propagator
  previously measured identical to two scalar HFPI runs at twice the cost
  (`max abs(Ex_vec - Ex_scalar)` 1.9e-23, 45-degree `max abs(Ey/Ex - 1)` <=
  1.1e-16, i.e. zero depolarisation).  With the projection on, `abs(Ez)^2` =
  1.40e-2, cross-pol 9.3e-5, 45-degree 0.143, and `abs(E)^2` is conserved to
  `1 - 1.1e-15` against the old opt-in's 0.841.  **`vector_projection=False`
  reproduces the pre-v5.46 behaviour exactly.**  Pass `return_ez=True` for the
  longitudinal component; the historical 2-tuple return is unchanged by default.
* **`propagate(method='hf')` returns an ndarray like every other method (K20).**
* **`...with_opl_callable(chunk_output=)` is un-deprecated and functional
  again (K22)** -- it had been deprecated to a no-op, which is why the HF OPL
  quadrature took 119 minutes at `N_in = 256`.
* **Single-precision SAS (K3)** lost its ASM-Fresnel correction to
  cancellation; the rewrite also moves the complex128 default path.
* **`backend.scipy.jv(v, x)` computed `scipy.special.jv(x, v)` (K2)** -- the
  arguments were swapped.
* **The HF / MHS resample renormalisation fabricated energy across a crop
  (K11)**, and a 0.5 % pitch change was a silent no-op on `hf` (K21).
  `propagate_through_system` now warns when a Fresnel / SAS leg's resample-back
  CROPS the field (K6).

### Traced-carrier chain

* **The focal-peak readout no longer collapses when the reference carrier is a
  few percent off the beam (C1).**  The standoff resolver now carries a beam
  term.  The default standoff is UNCHANGED for every field whose envelope is
  flat (the beam term is exactly 0.0 there, verified over the resolver's own
  6 NA x 10 extent calibration matrix -- 60 cells, 0 with a non-zero beam term,
  and the resolved leg bit-identical).  On a field whose envelope carries
  residual curvature the leg is LONGER and the answer more accurate; **if a
  caller needs the old number, pass `standoff=` explicitly**.  The new
  containment guard REFUSES configurations that previously returned a
  silently-wrong spot; `on_focus_containment='warn'` downgrades it.
* **`aggregate` now sums at the members' own dtype.**  An all-complex64 fan is
  accumulated in complex64 where it used to be widened to complex128, so
  `aggregate(...).field.envelope.dtype` follows the inputs and the coherent sum
  carries float32 accumulation error (measured 7.6e-08 relL2 against the
  complex128 sum of the same 16 fields, inside the `sqrt(K)*eps32` = 4.8e-07
  random-walk bound).  That is the point of the change: 4.29 GB saved per grid
  at N = 16384.  Widen the inputs if you need the old accumulation.
* **A tilted complex64 chain stays complex64 (C3)** -- it used to promote to
  complex128 through a whole-grid chief-ray ramp.
* **`carrier_referenced_fit_radius` fits about the beam, not the grid origin
  (C2).**
* **`CarrierField` attribute assignment now emits a `DeprecationWarning`
  (C5).**  The class is scheduled to become `frozen=True` in **v5.48**.
  Assigning to a built field bypasses every `__post_init__` invariant --
  envelope shape vs the grid, complexity, wavelength, provenance
  canonicalisation.  The assignment still takes effect today, so no caller
  breaks; migrate to `with_provenance` / `re_reference` /
  `dataclasses.replace` / a fresh `CarrierField`, and for in-place accumulation
  to `np.add(acc.envelope, other, out=acc.envelope)` -- the same arithmetic bit
  for bit, with no rebind.

* **`focus_readout` / `output_grid` windows wider than one Bluestein period
  (WP-A25, full-run follow-up).**  A caller who waives the replica refusal
  (`on_replica='ignore'`) can now pass `replica_fill='zero'`: the samples the
  transform cannot measure come back zero instead of as periodic replicas of
  the core, which win every argmax / peak / encircled-energy reduction once the
  window is wider than TWO periods (the P2 design battery's unclipped doublet
  read FWHM 20.50 um / EE(2 waists) 0.495 against an analytic 17.41 um / 0.997
  that way; blanked, 18.50 um / 0.997).  The default `'repeat'` is unchanged, so
  no existing call moves; each chain stage dict carries a
  'readout_faithful_samples' entry saying how much of the window is
  measurement, and the refusal message states the
  two regimes instead of promising a safe peak.

### Analysis metrics

* **`wave_opd_2d` slipped whole waves on any aberrated pupil (A1)** -- the
  unwrap had no anchor.  It now anchors its piston at the pupil centre and
  reports when no unwrap can succeed.  **A pupil that does not straddle the grid
  origin is anchored on its rim instead, so a stored absolute `opd_map` piston
  from a strongly curved pupil may move by a whole number of waves** (measured
  +1.0000 and +2.0000 waves on a 1.2-waves-rms coma pupil decentred by
  (+60, -40) and (+110, +90) um, against 0.0000 centred).  The piston it stored
  was arbitrary.  Shape, PV, RMS, Zernike coefficients above piston and every
  nearly-flat map are unaffected.  `unwrap_phase_2d` is the new public masked
  2-D unwrap kernel.
* **Strehl used a PARAXIAL reference (A2).**  The denominator is now an exact
  spherical reference.  **Any recorded Strehl from an f/# faster than ~f/10 was
  inflated by `1/S_ref` and will now read LOWER -- correctly**; a Strehl above 1
  for a diffraction-limited pupil is no longer produced (f/2: reference 1.4292;
  f/2.5: 1.0988; f/50 unchanged to 1e-6 relative).  The "Strehl can exceed 1 in
  some edge cases" note described this defect, not a methodology consequence.
* **`shack_hartmann`'s `wavefront` was exactly HALF (A3)** and is now right.  It
  gains `reconstruction={'southwell','itoh'}` (`'southwell'` is the new
  default).  **A lenslet whose slopes are the NaN out-of-bounds sentinel now
  returns NaN in `wavefront` too**, instead of being integrated through as if it
  had measured zero slope -- the NaN pattern of `wavefront` matches `slopes_x`
  exactly.  Callers that reduce the map should mask (`np.nanmax`,
  `np.isfinite`).
* **`eval_image_plane_wfe` accepts `object_distance = float('inf')` (or
  `None`) (A4)**, which is the correct way to ask for an infinite conjugate --
  a large finite number cancels catastrophically in float64 (at 1e6 m the chief
  ray sat 589 um off axis and PV read 213.5 waves against the true 3.59).  A new
  `field_max_rad` kwarg (or `prescription['field_max_rad']`) carries the
  off-axis field at infinity, where a field point is a DIRECTION rather than a
  height; a non-zero `field` at infinity without it is refused, and a
  `field_max_rad` at or beyond 90 degrees is refused on the ANGLE.  **`field`
  keeps its object-position sense at BOTH conjugates**: `field = (0, +1)` is an
  object above the axis either way, so the chief travels towards `-y` and the
  direction cosines are `L = -sin(Hx * field_max_rad)`,
  `M = -sin(Hy * field_max_rad)` -- the NEGATIVE of `raytrace.ray_fan`'s
  `field_angle`, which is a ray-direction angle.  PV and RMS are sign-blind, so
  only a per-ray comparison sees the difference.  A finite `object_distance`
  that has lost the surface sag to cancellation now warns (gated on the
  mechanism: the intersection error exceeding a tenth of a wave).
* **`gerchberg_saxton`'s reported error was off by `N_pix` (A5)**, and
  `gerchberg_saxton_jax` now follows the x64 flag as `jax.numpy`'s own default
  float type does.  **A caller who wants the historical single precision under
  x64 passes `dtype=np.float32` explicitly.**
* **`plot_stokes` used the x extent for both axes (A5 / A7)**, and two
  half-pixel conventions disagreed.

### Thin elements, DOEs and materials

* **Three bundled Sellmeier rows held a DIFFERENT GLASS (E1)** and were the only
  dispatch path for those names.  Indices from those names change.  A VALUE
  cross-check over every bundled dispersion row now runs, so the class cannot
  recur.
* **A catalogue lookup outside the page's data range now RAISES instead of
  returning NaN (E2)** -- `get_glass_index('SILICON', 633e-9)` returned `nan`
  and `get_glass_index_complex` returned `nan + 0j`, with only a validity
  *warning*.  **A caller that relied on the NaN as an in-band "no data" signal
  must catch `ValueError`** (or test the range first); in-range lookups are
  bit-identical (SILICON 3.5003 at 1.31 um, 3.4757 at 1.55 um, 3.4150 at 10 um)
  and every other catalogue glass is unaffected.  Separately,
  `get_glass_index_complex` now honours the documented `kappa = 0` fallback
  instead of raising for exactly the common bulk materials.
* **`generate_turbulence_screen` delivered 2x the requested phase variance
  (E3).**  The spurious `sqrt(2)` is gone: every screen from this entry point is
  now smaller by exactly `sqrt(2)` on the same seed (`new * sqrt(2) == old` to
  5.5e-16 relative).  Runs tuned by eye against the old screen were running
  1.5x stronger turbulence than requested at small separations (effective
  `r0 = r0 / 2**(3/5)` = `r0/1.516`).  **To reproduce the old numbers exactly,
  ask for `r0_old_equivalent = r0 * 2**(-3/5)`.**  `subharmonics=N` (the Lane
  low-frequency correction) is new and is now forwarded by
  `propagate_through_system`'s `'turbulence'` element as
  `{'type': 'turbulence', ..., 'subharmonics': 3}`; the default 0 keeps a chain
  without the key bit-identical.
* **`apply_grin_lens` was 36 % wrong at the quarter pitch its own Notes
  recommend (E5).**  The screen applied the short-rod power `n0 g^2 d`; the
  rod's exact paraxial ABCD has `C = -n0 g sin(g d)`, so the two differ by
  `sin(gd)/(gd)` -- 0.9851 at `g d = 0.30`, **0.6366 at the quarter pitch
  pi/2**.  **`thin_form=True` reproduces the previous screen bit-identically**
  and warns above `g d = 0.2`, naming the measured factor.  On the exact path,
  `g d` beyond the quarter pitch warns that a single screen carries the rod's
  power only.
* **`make_bsdf` is strict about unknown keys (E6).**  ANY key the chosen `kind`
  does not consume is now a `ValueError`, so a surface `bsdf` spec that carried
  a free-text annotation (`'comment'`, `'source'`, `'notes'`) alongside its
  parameters used to be accepted and is now rejected.  Move such keys out of the
  spec dict -- the alternative, a silently-ignored key, is the defect this
  closes.  `A` / `B` / `C` are now accepted as aliases for `b0` / `l` / `s`, and
  supplying both spellings of one parameter raises.
* **BSDF sampling draws different random numbers for a given seed (E6)** --
  `sample()` on all three models and `sample_scatter_rays` use a vectorised
  batched path and an exact inverse CDF (113x lambertian / 327x gaussian /
  377x harvey_shack).  The sampled DISTRIBUTIONS are unchanged; the per-ray
  ordering is not.  A user subclass that implements only the ABC's abstract
  `sample` keeps working through the per-ray fallback.  Harvey-Shack TIS
  under-read by up to 32 % and is now right.
* **The `'air'` short-circuit now consults the registry (E7)** so an ambient
  model is implementable -- **by the canonical LOWER-CASE key only**.
  `get_glass_index('AIR')` and `'Air'` honour `GLASS_REGISTRY['air']`, but a
  callable stored under a mixed-case key (`GLASS_REGISTRY['Air'] = ...`) is
  still silently ignored and the lookup returns 1.0.  Register ambient models at
  `'air'`.  **No default changes**: `'air'` ships unregistered, `list_glasses()`
  is unchanged (77 names), and `'vacuum'` / `'__MIRROR__'` still raise.
* **`apply_aperture(edge='gray')`** is new: an area-weighted aperture edge.
  `create_periodic_phase_mask` no longer assumes a square cell without checking
  (E7).

### Polarization, coatings, sources, algebra, memory

* **`polarization='te'` applied the TM coefficient to a TE beam (Z1)**, and
  every unrecognised spelling did the same silently.  The accepted set is now
  `{'s','te','p','tm','avg'}`, case-insensitive, and **anything else raises**.
  Callers that meant `'p'` should say so; callers that meant `'te'` were getting
  the wrong physics and now get the right one.  `'s'`, `'p'` and `'avg'` are
  bit-identical (the TMM itself is untouched: the audit's Airy / Rouard oracles
  still agree to <= 6.7e-16 on R and <= 7.8e-16 on T).
* **Gaussian-Schell / Schell ensembles are no longer bit-identical for a given
  `rng` seed (Z2)**: they realised the grid-PERIODISED coherence kernel, so
  opposite edges of the grid came out ~99 % coherent (measured
  `max abs(mu - Gaussian)` 0.9949 -> 0.0081 at `sigma_g/L` = 0.313).  **The
  escape hatch `pad_sigma=0.0` reproduces the pre-fix path bit-for-bit.**  The
  factories warn when the grid is too short for the requested coherence length.
* **`FreeSpace(d)`'s default propagator is now `'asm'` (Z3)**, so the delivered
  grid agrees with the ABCD the operator reports.  **Pass `method='auto'` to
  restore the dispatcher-selected kernel and its resampling.**
  `algebra.from_prescription` keeps `method='auto'` (measured pitch-preserving
  on singlet / meniscus / doublet at N = 256, dx = 8 um).  `FreeSpace` now also
  reports when the pitch-preserving default has CLIPPED the beam at the window
  edge.  The PITCH CONTRACT is documented on `Operator.abcd`: a ray matrix
  describes `(height, angle)`, not sampling -- read the delivered pitch off the
  returned Source / tuple, never off `abs(A)`, unless every `FreeSpace` in the
  chain is pitch-preserving.
* **`estimate_lens_memory(lens_model=...)` is case-SENSITIVE and validates
  (Z3)**: a differently-cased or misspelled value was reserving the wrong budget
  and now raises.  `'traced'` (the default) and `'real'` are bit-identical.
  `parallel_amp` is inert for `lens_model='real'`, and the `'real'` estimate
  itself was under-predicting `apply_real_lens`'s peak by up to 2.8x.
* **`lumenairy.algebra.from_prescription` the MODULE no longer shadows the
  function of the same name (Z4).**  `Operator.from_prescription` remains the
  canonical spelling; the free function stays submodule-only.
* **Deprecation warnings name the CALLER, not lumenairy (Z4)**, so a
  `DeprecationWarning` filtered by module now behaves.

### Rigorous grating solvers (PMM / RCWA / BOR)

* **`PMMStack`'s default `min_feature` is now `period * 1e-3` (was
  `period * 1e-5`) (G2)**, above the measured sliver hazard band instead of at
  its bottom (0 of 11 degree-scattering rungs against 6 at the old default).
  **A caller who relied on the old value -- to keep a deliberate sub-nm
  cross-layer offset, or to reproduce a number measured before this release --
  gets it back with `min_feature=period*1e-5`.**  The snap moves walls by at
  most `min_feature/2`, so a stack WITH colliding cross-layer walls can now see
  its solved geometry differ from the requested one by up to 5e-4 of a period; a
  stack WITHOUT them is byte-identical (no pair inside the threshold ->
  identical grid -> identical answer).  The snap has always been
  cross-layer-pairs-only: a close pair a single layer owns (an intentional thin
  liner) is never thinned.  The union grid now WARNS when it snaps such a pair.
* **The differentiable `PMMStack.solve` twin ran NO guards (G1)** -- it returned
  before every one of them, so a gain superstrate that raises on the NumPy path
  returned silently-wrong answers on the JAX path.  All the concrete guards now
  run there too.
* **`PMM2DStackHybrid(formulation='li')` no longer depends on which axis you
  drew the grating along (G5).**  The per-slot Li operators were never routed,
  so a y-patterned layer got the y-axis inverse-rule operator on the Ex slot and
  Laurent on Ey -- both slots anti-Li.  One physical grating gave two different
  answers depending on the axis it was drawn along (T00 at `n_orders` 9:
  0.0334736588 vs 0.0322424723, a 5.3 % spread; now identical to 3.4e-14), and a
  one-layer stack disagreed with `pmm_efficiency_2d_cell`, which is documented
  as the same physics.  Numbers move for every `'li'` stack solve -- onto the
  correct answer.  Both `symmetry='auto'` and `symmetry=False` carried the
  defect; both are fixed.
* **`pmm_jones_2d` gained `formulation='auto'` (G6)** -- `'fff_nv'` on a
  separable in-plane cell, `'laurent'` otherwise, `'laurent'` on the JAX path --
  and `return_jones_transmission=True` (G13), the transmission Jones a
  transmissive metasurface QWP is actually designed against.  The DEFAULT stays
  `'laurent'`, so nothing moves unless you opt in; `'auto'` is the recommended
  setting for new code.  The docstring's accuracy ordering is the REFLECTION
  ordering -- on the TRANSMISSION retardance `'li'` is the most accurate of the
  three at `n_orders >= 9` (0.03 degrees against 0.17-0.34 for `'fff_nv'`).
* **A prepared 2-D sweep now folds, truncates and warns like the direct entry
  (G8).**  A sweep built at the defaults takes the even-parity fold, so its
  numbers move by ~1e-13 -- onto `pmm_efficiency_2d`'s default answer.  **Pass
  `symmetry=False` to `prepare_pmm_2d*` / `*_vs_wavelength` for the previous
  bits.**  A lossless sweep in an ill-conditioned `(degree, n_orders)` corner
  will now emit the closure warning it always should have.
* **The staggered 2-D family gained `max_pencil_dof` (default 12 000) and a
  REDUNDANCY warning.**  An unguarded ~1000x cost cliff (measured 498 s CPU /
  8.4 GB for a solve that needs 0.22 s through `pmm_efficiency_2d_cell`) now
  refuses on the pencil dimension, and a redundant SEGMENT grid warns naming the
  smallest uniform lattice that holds every wall on either axis -- the number
  you can actually follow, which for a shared-grid stack is the lcm over the
  layers' joint wall set.  `max_pencil_dof=` on any layer acknowledges and
  silences it.  It warns rather than refusing because splitting a region into
  more segments is a legal h-refinement.
* **The RCWA Rayleigh / Wood-anomaly nudge is now a symmetric bracket and it
  warns (H2).**  `rcwa_efficiency_1d`, `rcwa_jones_1d`,
  `rcwa_jones_1d_segments`, `rcwa_efficiency_2d`, `rcwa_efficiency_2d_shapes`,
  `PreparedRCWA2D.solve` (built by `prepare_rcwa_2d`; before v5.46.0 the warning
  named a class that does not exist, so grepping a captured pre-5.46 log for the
  real name finds nothing), `rcwa_jones_2d` and `RCWAStack.solve` return DIFFERENT
  numbers at an exact Wood anomaly than before (closer to the exact-wavelength
  answer) and warn there.  **Off-anomaly solves are bit-identical and
  unwarned.**  The effective wavelength actually solved is reported (`wl_eff`).
  `RCWAStack.solve(retain_internal=True)` keeps the one-sided nudge -- the
  retained per-layer partial S-matrices are not linear in the field, so
  averaging them is not defined -- and says so in its warning.  Under
  `jax.jit` the same anomaly used to return all-NaN; the grazing mode is now
  regularised in place and stays evanescent.
* **`rcwa_jones_2d(formulation='fff_nv')` manufactured form birefringence on a
  symmetric cell (H3).**  It now symmetrizes and emits a validated-scope warning
  on a curved cell; `allow_nonseparable_nv=True` acknowledges it.  **A cell that
  is not transpose-symmetric returns different numbers than before** -- measured
  `max abs(dJ)` 8.3e-04 at `n_orders` 4 falling to 1.2e-04 at 12, inside the
  truncation error, with both forms converging to the same limit.  A user
  pinning `fff_nv` Jones values will see the move.
* **`guided_modes` returned an EMPTY list for every weakly-guiding fiber (H1)**
  and now returns the modes; a SHORT result is no longer silent either.
* **`PMMStack.internal_field(pol=...)` takes the family's `'te'`/`'tm'`/`'s'`/
  `'p'` spellings (G4)**, and `set_source(angle=A, theta=T)` with `A != T` is no
  longer silent.  `PMMStack.prepare().solve()` refuses instead of silently
  dropping propagating orders.
* **`pmm_2d_order_drift` is new (G7)**: the per-order convergence signal that
  lossless energy closure cannot give you (closure improved monotonically while
  T00 swung 35 % between neighbouring truncations).  `energy_tol=` is accepted
  on the scalar 2-D entries.

### I/O, prescriptions and the optimiser

* **CODE V `DIM M` means MILLIMETRES, and `C` / `I` are recognised (I1).**
  Files written by lumenairy BEFORE this release with the default `units='M'`
  carry `DIM M` with SI-metre numbers.  The writer now stamps a format marker
  (`! LUMENAIRY-SEQ-FORMAT 2`) next to its generator banner, and
  `load_codev_seq` uses the pairing *banner present + marker absent + a `DIM M`
  line* to identify such a file exactly: it is read with the legacy metre scale
  (so its values are unchanged) and raises a `UserWarning` saying so.
  **Re-export it once to get a file CODE V can read, or pass the new
  `dim_units=` kwarg (`'M'`/`'C'`/`'I'`/`'SI'`) to force a reading.**  Files
  from CODE V itself, and files written from this release on, are unaffected.  A
  pre-release file written with `units='MM'` or `units='IN'` needs **no**
  migration and gets no warning: those tokens meant millimetres and inches then
  and still do.
* **HDF5 `compression` and `chunk_size` default to `'auto'` (I7).**  New files
  written by the four storage functions are larger by ~5.6 % for complex fields
  and are not gzip-filtered; existing files are unaffected and still read.
  Measured 24x faster writes / 3.3x faster reads at 4096^2, because gzip on
  incompressible complex data buys nothing.  **Pass
  `compression='gzip', compression_opts=4` to restore the old default.**
  `append_plane` on a `.zarr` store now warns that `compression` /
  `compression_opts` are dropped (the Zarr path has no compression parameter at
  all) instead of ignoring the request silently.
* **`THORLABS_CATALOG['LA1509-C']` was a 200 mm lens under a 100 mm part number
  (I6).**  Every catalogue row is now checked against its own part number.
* **CODE V mirrors, conics and aspheres are parsed instead of dropped (I3);**
  powered air-to-air surfaces survive the Zemax window auto-detect (I2);
  cylindrical / biconic surfaces no longer export as spheres in silence (I4);
  `scale_prescription` is self-similar for Forbes-Q, diffractives and the
  stored BFL (I7).
* **`design_optimize(method='newton')` now says that it drops bounds (I8)**, and
  `design_optimize_multi_objective` refuses an infeasible run instead of
  returning one.  `create_zoom_configs` no longer writes glass thicknesses or
  truncates silently.
* **`MinEdgeThicknessMerit` and `edge_thickness`** are new.

### Designer UI and the lens library

* **The point-source object distance comes from the element geometry (U3).**
  A design whose source form said e.g. 1000 mm while the first lens sat at
  200 mm previously under-filled the entrance pupil by 5x; it now fills it
  correctly.  The form field is honoured only when no optic has been placed yet,
  and its tooltip says so.  **To reproduce the old numbers, move the first
  element to the distance the form field named.**  An infinite object distance
  is exported as "at infinity".
* **`SystemModel.to_prescription()` exports the system the layout shows (U1,
  U2)** -- mirrors carry `is_mirror`, every surface carries its own
  `semi_diameter`, and the stop flag survives.  A per-surface `semi_diameter` is
  no longer clipped by the `elements` matcher, on either the NumPy or the JAX
  resolver.
* **A folded library entry saved before the U1/U2 fix now loads with a DIFFERENT
  (correct) EFL/BFL/aperture set** and says so once per entry with a
  `UserWarning`.  `load_lens` back-fills `is_mirror` / `semi_diameter` /
  `is_stop` from the entry's own `elements` list by position, and only when the
  two lists are provably consistent (same length, at least one mirror, no
  surface already carrying `is_mirror`, and every radius matching bit for bit).
  Measured on a concave fold mirror (R = -200 mm, semi-diameter 25 mm) followed
  by an N-BK7 singlet: -15.2 % EFL and +16.7 % BFL before, bit-identical to the
  layout after.  **Re-saving the design from the designer produces a
  prescription that needs no repair and silences the warning.**  Unfolded
  entries, modern entries and `.zmx`-shaped prescriptions are bit-identical to
  before.
* **A wave-optics run no longer disables pyFFTW / SciPy FFT for the whole
  process (U4)**, and `lumenairy.ui` no longer imports matplotlib at module
  scope.

### Packaging, configuration and public API

* **`threadpoolctl` is a new HARD dependency.**  `pip install lumenairy` now
  pulls it (pure Python, ~30 KB; environments with scikit-learn already have it
  as a transitive dependency).  Nothing breaks without it -- the library
  degrades exactly as before -- but the BLAS caps it advertises
  (`set_blas_threads`, `rcwa_blas_threads`, `@_with_blas_limit`) only take
  effect with it installed.  Measured on a 24-thread Windows OpenBLAS box: a
  1-D TM RCWA solve at `n_orders=81` takes 18.2 s unpinned against 0.13 s at one
  thread (140x).
* **`lumenairy.override(**knobs)` is the supported way to set a process
  global.**  Every registered knob is restored in reverse order on every exit
  path -- normal return, `break` or exception -- with a roll-back if a setter
  raises partway through entry, and an unknown name raises BEFORE the first
  setter runs.  17 knobs are registered today (12 in `fft_infra`, plus
  `max_ram`, `cache_budget`, `storage_backend`, `library_path`,
  `blas_threads`).  The bare `set_*` / `get_*` pairs still work and are
  unchanged; `lumenairy_context` is unchanged.  See CONVENTIONS section 10.1.

  ```python
  import lumenairy as la

  with la.override(fft_threads=1, pyfftw_planner='FFTW_ESTIMATE'):
      ...  # restored on every exit path, including an exception
  ```
* **15 names were promoted to the package root** (`lumenairy.__all__` 708 ->
  723): `unwrap_phase_2d`, `clear_meshgrid_cache`, `meshgrid_cache_bytes`,
  `zernike_basis_cache_bytes`, `MinEdgeThicknessMerit`, `edge_thickness`,
  `pmm_2d_order_drift`, `exit_vertex_transfer`, `EXIT_VERTEX_GRAZING_TOL`,
  `resolve_exit_index`, `vertex_plane_transfer_t`, `exit_vertex_transfer_jax`,
  `seed_entrance_eikonal`, `aberration_free_reference_fit`, `override`.  Plus
  `rs_alias_free_distance`, renamed from the private `_rs_alias_free_distance`
  (which is kept as an alias for existing importers).
* **`raytrace` gained one shared exit-vertex transfer** (audit section 15.1):
  `TraceResult.at_exit_vertex()`, `exit_vertex_transfer(bundle, n)`,
  `exit_vertex_transfer_jax(state, n)`, `vertex_plane_transfer_t`.  **Replace
  every hand-written `t = -z/N; opd += n*t; x, y += (L, M)*t; z = 0` block with
  it.**  `raytrace.trace()` leaves every ray at its intersection with the LAST
  surface, i.e. at `z = sag(rho)`, not on the exit-vertex plane; the audit found
  six hand-written corrections and five consumers that had forgotten it.  The
  helper differs from those copies in its grazing policy: a grazing ray
  (`abs(N) <= EXIT_VERTEX_GRAZING_TOL = 1e-30`, finite) dies with
  `RAY_MISSED_SURFACE` rather than being teleported.
* **`opd_fan_data` now subtracts a reference sphere (R1)** and off-axis fans no
  longer carry a launch-plane tilt (R2, through the new
  `opd_seed='plane'|'eikonal'` and the functional `seed_entrance_eikonal`).
  **Any hard-coded axis limit or threshold calibrated on the old OPD-fan numbers
  -- 3.1x too large and of the wrong sign -- needs re-checking.**
  `seidel_coefficients` now includes the conic and aspheric terms (R3); conic
  surfaces no longer falsely report `RAY_MISSED_SURFACE` (R4); the
  diffraction-order kick carries the medium index (R5); and `rays_from_field`
  no longer aliases at half the grid Nyquist (R6).
* **The test suite's own contract was corrected**, which matters only to
  contributors: the unit lane is ~3.63 h (the "fast (< 30 s) API-contract tests"
  claim was wrong by ~440x), the slow lane is re-marked at the > 2 min/file bar,
  `.test_durations` has a staleness gate, `ruff` is blocking and lints
  `lumenairy/ui/`, and CPython 3.14 is in the CI matrix.
* **Three configuration objects for the `apply_real_lens` family --
  `LensGeometry`, `LensNumerics`, `LensResources` -- and a `LensConfig` that
  holds all three.  NO MIGRATION IS REQUIRED: this is purely additive.**  The
  five entry points still take every keyword they took before, with the same
  defaults and the same answers; the objects are a second way to spell the same
  call, for the case the audit named -- a study that sweeps one axis while
  holding a dozen settings fixed, where the settings were previously a dozen
  loose keywords copied between five call sites.

  ```python
  import lumenairy as la

  numerics = la.LensNumerics(ray_subsample=2, newton_poly_order=8,
                             newton_fit='spline')
  resources = la.LensResources(n_workers=4, parallel_amp=False)

  E_out = la.apply_real_lens_traced(
      E_in, prescription=rx, wavelength=633e-9, dx=dx,
      numerics=numerics, resources=resources)
  ```

  which is bit-identical to passing `ray_subsample=2, newton_poly_order=8,
  newton_fit='spline', n_workers=4, parallel_amp=False` as loose keywords.  The
  objects are frozen dataclasses (hashable by value, comparable, `repr`-able),
  `LensConfig.from_kwargs(...)` / `.to_kwargs()` round-trip, and
  `.narrowed_to(fn_name)` drops the fields a given entry point does not accept
  so one config can drive all five.  Mixing an object with the loose keyword it
  also sets raises rather than silently picking one.  Full field tables, the
  role each object plays and the validation rules are in
  [`docs/lens_configuration.md`](docs/lens_configuration.md).
* **Version-history narrative moved out of the source into `docs/history/`
  (contributors only; no behaviour change).**  Several modules carried
  thousands of lines of `vN.M (audit X): pre-fix this did A, which was wrong
  because B, now it does C` commentary.  Those blocks now live in
  `docs/history/<module>.md`, reproduced verbatim under the source line they
  came from, so `git log -S` on any phrase still lands on the commit that wrote
  it.  Nothing the interpreter executes changed: each document's header records
  the SHA-256 of the module's AST (docstrings and positions removed) and of its
  token stream (comments and docstrings dropped), both taken from the
  pre-relocation file, and `tests/unit/test_audit2609_a17_history_relocation.py`
  recomputes them on every run.

  **The rule that follows from that pin: a commit that intentionally changes
  code in a module with a history document re-records that document's
  fingerprints in the same commit, and says why.**  Use
  `python scripts/record_history_fingerprints.py <module> --reason "..."`;
  `--check` reports drift without writing.  The `--reason` is mandatory on a
  write and is appended to the header as a `re_recorded:` line, so the header
  carries every baseline move and its justification.  Never hand-edit a hash.
  `CONTRIBUTING.md` has the same rule with the commands.
* **Three documentation and diagnostic corrections** worth knowing if you grep
  logs or read tooltips:
  * `scripts/check_doc_identifiers.py` is now committed beside
    `check_source_line_citations.py` and gated by
    `tests/unit/test_audit2609_a21_doc_identifiers.py`.  It resolves every
    backticked identifier in `README.md`, `ROADMAP.md`, `Migration-Guide.md`
    and `CONVENTIONS.md` against the installed package and exits non-zero on
    one that does not exist -- so a name in these documents is a name you can
    import.  It needs no network and takes ~4.5 s.
  * The staggered-PMM 2-D shared-grid advisory is caught by a **message**
    filter, not a module one:
    `-W "error:.*eps_cell is a SEGMENT grid:UserWarning"`, or the same string
    as a pytest filterwarnings entry.  A `module=` filter cannot work here at any
    *correct* `stacklevel`, and that is not a bug to fix: Python matches
    `module` against the frame `stacklevel` selects, and the whole point of a
    non-1 stacklevel is to point at YOUR code -- so a warning attributed
    correctly is by construction not attributed to the library module that
    raised it.  The exact spelling, the measurement table behind it and both
    entry points are documented beside the warning in
    `lumenairy/elements/pmm/twod_staggered.py`.
  * The `fast_analytic_phase` tooltip in the designer UI claimed "< 10 nm OPL
    error".  The measured figure is **~7 nm rms per mm of glass**, so on a
    10 mm element it is ~70 nm, not 10.  The tooltip and
    [`docs/subsystems/real_lens.md`](docs/subsystems/real_lens.md) section 4
    now both state the per-mm form.  Nothing computational changed -- only the
    claim about it.

---

---

---

## 5.47.0 -- adversarial audit remediation, Wave 4 (2026-09-14)

Wave 4 implements the performance and feature designs 5.46.0 deferred with a written plan
(the `WP-B*` reports under
[`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/`](docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/))
and closes the two findings 5.46.0 left partially fixed; thirteen work packages, each with an
independent verifier.  Almost everything is opt-in behind a
new keyword whose default reproduces 5.46.0 byte for byte.  The exceptions are listed first,
each with the finding ID, what changed, and the way back where one exists.

### Defaults that move

**Maslov lens, non-collimated input (S6; `apply_real_lens_maslov`).**  For a non-collimated
input with `integration_method` of `'stationary_phase'`, `'local_quadrature'`, or `'auto'`
where it resolves to `'stationary_phase'`, the returned field changes: the asymptotic saddle
now expands about the input field's local wavevector instead of the OPD-only stationary
point.  That is the fix.  Way back: `input_wavevector_saddle=False` on the call (the module
seam `lumenairy.elements.lenses_maslov._S6_INPUT_WAVEVECTOR_SADDLE` sets a process default).
`collimated_input=True` also pins the old saddle but re-sizes the pupil chart as well, so it
is not a way back to the old numbers.  `'quadrature'` and `'levin'` have no saddle and are
byte-identical.  Collimated inputs are unchanged everywhere.

**Analytic lens, displaced remap (L9; `apply_real_lens` with `surface_model='displaced'` on a
decentred, tilted or `sag_callable` element -- the default routing for such elements).**  The
returned field changes for three reasons, in decreasing size: the default launch lattice
moved from 181 to 257 rays a side (observables move about 1e-4 relative on a smooth input;
up to 0.38 of the peak on an input that was already aliasing at 181, where the cure is
`conjugate=`, not a lattice); the exit field is rebuilt by inverting the launch-to-exit map
instead of triangulating it (differences O(h^2) in the launch pitch; the pupil rim now lands
on the aperture rather than on the hull, so a truncated pupil gains back about 1 % of
transmitted power); and the carried envelope is cut at the grid's largest centred window,
which makes the answer mirror-symmetric at the cost of one input rim row and column.  Way
back for the lattice only: `displaced_n_side=181` (also `LensNumerics.displaced_n_side`).
The inversion and the window have no way back; the old answer was asymmetric by construction.
`apply_real_lens_traced`, `apply_real_lens_maslov`, the pointwise obliquity screen, the
meridional LUT path, the 1-D symmetric remap and every `'thin'` / `'tangent_facet'` /
`'tangent_facet_remap'` call are unchanged.

**RCWA, out-of-plane `fff_nv` (H3; `rcwa_jones_2d(formulation='fff_nv')` on a full 3x3 tensor
cell).**  The operator is now symmetrised over the two Li-2003 factorisation orders, as the
in-plane operator already was.  Numbers change by up to the spurious form birefringence a
single order manufactured (2e-4 to 5e-3 of the Jones scale on the audit's cells).  No way
back; the old answer broke the cell's own mirror symmetry.  In-plane cells are unchanged.

**The Fresnel leg of `propagate_through_system` (K6; `method='fresnel'` and per-element
`{'method': 'fresnel'}`).**  The leg now evaluates the Fresnel integral directly onto the chain
grid through the matrix Fourier transform instead of propagating onto the single-FFT natural grid
and cubic-interpolating back (which cropped the field outside `N dx`, paid the interpolator's MTF,
and on a non-square sample count scaled the y axis by the x ratio).  Against an explicit
double-sum oracle the leg reads 5e-16 to 5e-14 where it read 1e-4 to 4e-1.  Size of the move:
7e-15 where the natural grid already equalled the chain grid, about 1e-4 on a contained field,
about 1e-1 on a grid-filling one, 2.5e-1 on a non-square input.  A pinned `'fresnel'` chain result
must be re-baselined; there is no way back (the old answer was the cropped, interpolated one).
The leg's K6 crop warning is replaced by a window-loss warning of the same class that fires only
above the chirp-sampling bound `z = N dx^2 / lambda` (below it the matrix Fourier transform's own
faithful-zone warning covers the replica case); its under-sampled-chirp warning and its `z <= 0`
refusal are now prefixed by `fresnel_propagate_mft`.  The `'sas'` chain leg and the analytic lens's
in-glass `'sas'` / `'fresnel'` gap legs keep their resample-back and gate its method on window
against period: they move only where the requested window fits inside one chirp-Z reconstruction
period (in practice where the pitch coarsens), gaining a unit MTF there, and are byte-identical
everywhere else.

**Analytic lens, the 1-D symmetric displaced remap (L9 follow-up; `apply_real_lens` with
`surface_model='displaced'`, `displaced_mode='remap'` on a rotationally symmetric element).**  The
remap read the input at the entrance height through an interpolator that returns zero outside the
grid, so on a converging element the +x rim came back as a crescent of exact zeros while its -x
mirror carried the full envelope (paired mirror error 3-5 % of the field, worst pixel 0.65 of the
peak against an exact zero) -- pre-existing, on a CENTRED input, not only a decentred one.  The
read now uses the same centred window the 2-D remap uses; the answer is mirror-symmetric to 1e-16
at the price of the outermost input ring on both sides.  No way back.  The lattice-smoothing warning
now quotes the pitch the trace actually uses (the fan is thrown 3 % wider than the aperture) and
names a `displaced_n_side` that clears its own bar.

**Maslov propagator, non-collimated inputs (three Wave-4 corrections in `apply_real_lens_maslov` and its
JAX sibling).**  The pupil chart is sized from the input's MEAN launch direction plus its angular spread
instead of the second angular moment about zero, so a tilted input no longer inflates the chart threefold:
tilted inputs return a different (better) field -- at twice the lens NA fidelity against the exact integrator
goes from 0.000 to 0.91 -- and collimated, converging, diverging, speckled-about-zero and hard-apertured
inputs are byte-identical; to reproduce a 5.46 number on a tilted input pass the old sizing explicitly,
`input_na = 3 * sqrt(<v^2>)` of that field's own angular spectrum.  The S6 fallback additionally scores the
fitted wavevector's SLOPE error (`_K1_DERIV_RESIDUAL_MAX = 1.2`, derived on two charts) beside its value, so a
hard-edged or heavily speckled input above the bar now falls back to the OPD-only saddle with a warning that
names `integration_method='quadrature'` as the remedy, where it used to return an answer whose fidelity was
0.000; `input_wavevector_saddle=True` forces the previous behaviour.  `apply_real_lens_maslov_jax` carries the chief-ray displacement of a non-collimated input (its thin
screen landed a tilted input 2.7 % short; now 0.2 %), byte-identical on a collimated input;
`input_wavevector_saddle=False` restores the old screen.

**Universal dispatcher, a single-valued field at a caustic (WP-B7b; `apply_real_lens_universal(method='auto')`
at an `output_plane_distance` inside the caustic zone, when the field is single-valued AND the prescription is
inside the sag-screen aberration envelope).**  Those calls now route to `apply_real_lens` + exact angular
spectrum (`'phase_screen'`) instead of `apply_real_lens_fga`, because FGA converges to the WRONG field at a real
singlet focus: against a brute-force Rayleigh-Sommerfeld oracle on an exact conic raytrace its fidelity is
0.13-0.15 under fifteen sampling settings where the screen reaches 0.999, and the screen is closer at every NA
from 0.048 to 0.26 (rms spot error 7.2 um -> 0.02 um on the measured fixture; wall clock 7.3 s -> 0.7 s).  A
MULTI-VALUED field still routes to FGA, and so does an over-budget prescription (the H2 gate's own dual-oracle
measurement forbids the screen there; whether that condition should stay is escalated with the fixture that
decides it).  `apply_real_lens_auto` is not affected.  Way back: `method='fga'` (`caustic_pad_dof` only
narrows the zone; inside it the route is unchanged, so it is not a way back).  VERIFY-B7b traced the
FGA deficit itself to a reference-plane defect in the FGA transfer (the differential base-ray state is left
on the last surface while the image leg is added from the exit-vertex plane, a spurious phase of k times
the last surface's sag); with that repaired FGA scores 0.9998 at the same caustic, so this route is a
mitigation and the repair is the next work package.

**FGA on an even-aspheric prescription (WP-B7b; `apply_real_lens_fga`, `apply_real_lens_fga_vector`,
`apply_real_lens_universal(method='fga')` at `coarse_stride=1` with `exact_jacobian` at its `None` default or
`True`).**  The analytic-Jacobian predicate is now the analytic primitive's own domain, so an aspheric departure
takes the exact single-ray Jacobian instead of the finite-difference 9-ray bundle (the differential transfer
loses a 2.4e-9 relative FD truncation; trace count 9N -> N), and a field-decentred / tilted / sag-callable conic
FALLS BACK to FD where it raised `NotImplementedError`.  All-conic prescriptions are byte-identical.  Way back on
an asphere: `exact_jacobian=False`.

### Values that move in the last bits (no signature change)

* The single-layer RCWA entry points (`rcwa_efficiency_1d`, `rcwa_jones_1d`,
  `rcwa_jones_1d_segments`, `rcwa_efficiency_2d`, `rcwa_jones_2d`, `rcwa_efficiency_2d_shapes`,
  `PreparedRCWA2D.solve`) compute the two transmitted / reflected amplitude products through a
  closed form with one star inverse instead of two.  Worst measured movement 1.7e-15 absolute
  / 3.1e-15 relative; a test pinning one of them to more than about 13 significant figures
  needs its value re-recorded.  `RCWAStack`, `berreman_jones_1d`, EME and BOR are
  byte-identical.
* Zernike radial polynomials with `n >= 22` (`j >= 253`, beyond every shipped table) are
  evaluated by the Kintner recurrence instead of the alternating factorial sum, whose own
  error had reached 1.5e-9 at `n = 22` and 3e-3 at `n = 40`.  Every `n < 22` value is
  bit-identical.
* `jacobian='auto'` in the GBD propagator with an ASPHERIC prescription now receives the exact
  analytic ray-transfer Jacobian instead of the finite-difference fallback (the two agree to
  about 1e-8 relative; the analytic side is exact).  FGA's predicate followed in WP-B7b -- see
  "Defaults that move" above, since there it also changes a raise into a fallback.

### Calls that now refuse instead of answering

* `compute_psf(method='fft', N_psf=<less than the pupil size>)` raises.  It returned an
  `N_pupil x N_pupil` array while reporting the `N_psf` pitch (and mis-scaled `normalize='power'`
  by `(N_pupil/N_psf)^2`).  Ask for `N_psf >= N_pupil`, or use `method='mft'`, which samples
  exactly `N_psf` points.
* The Schell factories with `generator='modes'` refuse a zero, negative, NaN or infinite
  `coherence_length` (the `'fft'` generator's behaviour is unchanged).
* `apply_real_lens(wave_propagator='fresnel')` refuses an anamorphic pitch (`dy != dx`) and a non-square
  grid, as its `'sas'` sibling already did: the in-glass gap leg's resample-back reads one input pitch, so
  it scaled the y axis by the x ratio (a 40 % field error and a power ratio of exactly `dy/dx`), silently.
  Use `wave_propagator='asm'` (or `'rayleigh_sommerfeld'`) on such grids.
* `PMM2DStackHybrid.formulation`, `.cascade`, `.symmetry` and `.truncation` refuse an out-of-vocabulary
  assignment after construction (they were validated only in the constructor; `st.formulation = 'fff_nv'`
  silently behaved as `'laurent'` and `st.truncation = 'circle'` quietly solved the rectangular box).  `LensConfig.to_kwargs(strict=True)` is a new opt-in that raises where a requested field would
  be dropped; the default keeps dropping silently.
* `create_gaussian_beam(geometry_dtype=np.float32)` with a NumPy-scalar centre now actually runs
  its geometry in single precision (it silently promoted back to float64 before), so that opt-in
  call returns different values: within 1.2e-7 of the peak on axis and 3.2e-7 off axis of the
  float64 answer.

### Warning text

* The HFPI prescription walk's legacy-normalisation warning was rewritten (it now names the
  condition that failed).  A caller filtering on the 5.46 wording should match on
  `NOT photometric`.
* `scalable_angular_spectrum_propagate` (and so `apply_real_lens(wave_propagator='sas')` on an
  in-glass gap) emits a `RuntimeWarning` below its NEW near-field bound `z >= N dx^2 / lambda_medium`,
  where its Fresnel-sum step aliases the quadratic chirp exactly as `fresnel_propagate` does (the
  `'fresnel'` leg has warned there since the K1 guard).  Values are unchanged to the last bit; a
  caller who relied on the silence was getting an aliased answer (measured 6x to 85x the true power
  at 0.2 to 0.05 of the bound) and should move to `wave_propagator='asm'`, or filter
  `RuntimeWarning` from `lumenairy.propagators.sas`.
* `apply_real_lens_traced_uniform` (`caustic='uniform'`) emits a `RuntimeWarning` when the fold's `zeta` is
  extrapolated more than `_ZETA_EXTRAPOLATION_MAX = 8.0` times past the two-branch band it was fitted on
  (the returned field is unchanged; measured, the dark tail's energy is within 5 % below ~5x and +12.5 % /
  +22.8 % at 9.8x / 454x).  The two-branch band width and the extrapolation ratio join the
  diagnostics.  At such a fold
  `amplitude_model='ray_density'` is the better member; at a well-resolved fold with a wide band the uniform
  completion remains the best of the four.
* Every warning raised inside the lens family (`apply_real_lens`, `apply_real_lens_traced`,
  `prepare_real_lens_traced`, the Maslov, GBD, thin, image-map and multibranch bodies) is now
  attributed to the CALLER's frame -- the first frame outside `lumenairy` -- instead of a fixed
  depth that was one frame short on every configured (`config=`) call.  `warnings.filterwarnings(...,
  module=...)` keys on the attributed module, so a filter written against
  `lumenairy.elements._lens_real` or `..._lens_traced` no longer matches; filter on the category and
  message, or on the calling module.

### Opt-in additions, default byte-identical

Propagators: `propagate_hfpi_through_prescription(z_output=, normalisation='auto')` (K13),
`sampler='sobol'` (K22), `rayleigh_sommerfeld_propagate(kernel='spatial-integrated')` (K9; four
to five decades worse on a sampled smooth field, so only for cell-constant inputs),
`resample_field(method='chirpz')` (K6; its docstring now states that the unit MTF holds when
`N_out dx_out == N_in dx_in`).
Carrier chain: `propagate_traced_carrier_chain(transport='collins', on_collins_sampling=)` (and the
`propagate_traced_carrier_chain_multi` and `propagate_carrier_referenced` entry points; `dx_out` / `carrier_out` on the single step) --
the Collins / ABCD-Fresnel integral evaluated by a chirp-Z onto a freely chosen output pitch, so a
focus readout needs no standoff plane and no replica handling; it selects the transfer-function
quadrature wherever that one is sampled, so on ordinary legs it is the shipped arithmetic
bit for bit (each stage's diagnostics record which form ran).  `gap_kernel='exact'` is refused, not
downgraded, on a Collins leg whose exact-kernel refinement cannot be represented.
Analysis: `compute_psf(method='mft', dx_psf=)`, `encircled_energy_profile` and the `profile=`
keyword on the curve and radius functions, `create_gaussian_beam(geometry_dtype=)`, the
Schell factories' `generator='modes'` / `n_pseudo_modes=`.
Lenses: `input_wavevector_saddle=` (Maslov), `displaced_n_side=` / `LensNumerics.displaced_n_side`
(analytic), `fit_basis='zernike'` / `LensNumerics.fit_basis` (traced; a change of basis inside one
polynomial span cannot move the fitted map, so this buys conditioning only -- the concentric branch's
Gram goes from numerically singular to the identity, the decentred branch's advantage decays 0.95
decades per degree and is gone by the shipped order 16).
Ray tracing: `trace(sphere_normal='analytic')` (1.13x, within 4 ULP of a 60-digit oracle),
`trace(renormalize='exit')`, `make_rings(pattern='vogel')` / `ray_pattern='vogel'` /
`through_focus_rms(pattern=)`, the built `JaxPrescription` cache
(`lumenairy.raytrace.jax_trace.clear_jax_prescription_cache`).
PMM: the 2-D tensor operator cache (bit-identical; one assembly per geometry).
Asymptotic family: `_solve_envelope_stationary_batch(scale_relative_stop=)` (off by default: it moves the
field by 9e-11); the batched kernels evaluate one Chebyshev basis per sweep (1.7-2.4x, byte-identical);
`aberration_tensor` builds only the modes it reads and memoises its waist probe (1.6x, byte-identical); the
GBD FFT reconstruction clips its kernel to the beamlets' support (37.7x -> 6.1x the output grid in memory,
agreement 1e-15 with the windowed sum).
Hygiene: `LensConfig.to_kwargs(strict=True)`; `propagate_hfpi_freespace_aperture(sampling='stratified' | 'uniform',
sampler=)`, default uniform byte-identical; `glass.glass_registry_generation()`; `get_glass_index` memoised over
the whole catalogue resolution (24x on a catalogue name, bit-identical); one branch-band leaf for the four modal
engines and one row-band schedule for the chunked lens surfaces (bit-identical, kernel census unmoved);
`lumenairy.LensPhysics` (the nine `apply_real_lens` model-term switches: `fresnel`, `slant_correction`,
`absorption`, `seidel_correction`, `seidel_poly_order`, `surface_frame`, `displaced_mode`,
`displaced_obliquity`, `screen_obliquity`) with `apply_real_lens(physics=)` and `LensConfig.physics`, purely
additive and byte-identical to the keywords (`from_kwargs` / `to_kwargs` / `narrowed_to` reach it; a physics
request handed to a sibling entry point through `config=` raises and names `apply_real_lens` as the owner);
`PMM2DStackHybrid.truncation` as a validated property; `doe.create_fresnel_zone_plate`'s outside-the-aperture
fill written in the transmission's own dtype (bit-identical on every measured arm).

Measured and NOT shipped: the Gegenbauer nodal basis for the PMM wall corner (a Galerkin
no-op on the fixed polynomial space; the wall-corner cure is the hp mesh), Levinson solves for
the RCWA Toeplitz inverses (12 to 20 times slower and two decades less accurate than the
shipped inverse), the chessboard FFT-shift identity (bit-identical only on power-of-two
grids).

---

## 5.49.0 -- the default flips (2026-09-20)

The 2026-09-11 adversarial audit measured a set of numerical defaults that read
better than the shipped ones and shipped every one of them switchable with the
shipped default unchanged, recording the measurements in
`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/MAINTAINER_DECISIONS_2026_09.md`.
5.49.0 is the release in which the maintainer took those decisions.  Every item
below moves what an unmodified call returns, and every one of them leaves the
previous behaviour ONE keyword or ONE constant away, byte-identical under it
(proved against a `git archive` of the parent commit, on both builds).

### `gap_kernel='auto'` now falls back to `'fresnel'` near a carrier focus

**What moved.**  `lumenairy.propagators.carrier._GAP_KERNEL_ACCURACY_TAU`
defaults to `1e-4` instead of `None`.  On a Collins carrier leg, `'auto'` now
compares the exact-kernel refinement's PREDICTED departure from the paraxial
truth, `sqrt(3/2) * k * |z_eff| * theta_env^4 / 8`, against that tolerance, and
takes the paraxial kernel above it.  `theta_env` is the ENVELOPE's analytic
`1/e^2` half-angle, not the beam's.

**Who is affected.**  Only legs whose reduced frame `z_eff = B/A` is large,
i.e. legs close to the carrier's own `A = 0` plane.  The threshold is closed
form -- `|z_eff| > 8 tau / (sqrt(3/2) k theta_env^4)` -- so a design can be
checked without running anything.  Measured: on the VERIFY-B4 F3 fixture
(`w = 0.3 mm`, `lambda = 1.064 um`) the rule fires within **23.5 um** of the
focus and nowhere else; on the Wave-5 hygiene-2 fixture it fires within
**1.54 um** of that carrier's `A = 0` plane, which its published ladder never
reaches.  Entry points that can reach such a leg:
`propagate_carrier_referenced`, `propagate_traced_carrier_chain`,
`propagate_traced_carrier_chain_multi` and
`carrier_referenced_focus_readout`, each with `transport='collins'`.

**What it buys.**  On the two F3 rungs inside the band the relative L2 against
the analytic Gaussian goes from 2.3496e-03 to 1.4568e-14 (1 um) and from
2.3490e-04 to 1.1822e-14 (10 um).  The k4 wrap guard (the stats key of the same name) does not
fire at either
-- it bounds representability, not accuracy.

**Recipe -- keep 5.48.x exactly:**

```python
import lumenairy.propagators.carrier as _carrier
_carrier._GAP_KERNEL_ACCURACY_TAU = None      # the rule is not evaluated at all
```

**Recipe -- keep the exact kernel on one leg:** pass `gap_kernel='exact'`.  An
explicit `'exact'` is honoured over `tau`; only `'auto'` falls back.

**Caveat, stated both ways.**  The oracle behind the law is paraxial.  It
measures how far the exact kernel departs from the paraxial truth; it cannot
say which kernel is the more physical.  On a leg where the exact kernel is the
better physics this rule trades accuracy for agreement with that oracle, which
is why the opt-out is one line and an explicit request is never overridden.

### The dense GBD reconstruction counts its memory honestly

**What moved.**  `lumenairy.propagators.gbd.DENSE_MEM_BUDGET_ACCOUNTING`
defaults to `'measured'` instead of `'legacy'`.  The dense (`window=None`)
reconstruction sizes its beamlet chunk from `mem_budget_mb` using a per-cell
cost that was six times too small, so the budget was never a bound: measured at
`mem_budget_mb=512` with 1024 beamlets, the loop peaked at 3 074 MB on a
256-square grid (6.00x) and now peaks at 387 MB (0.756x).

**Who is affected.**  Any dense reconstruction whose `mem_budget_mb` actually
binds the chunk.  The returned field moves in the LAST BITS only (measured
2.1e-17 relative at N = 256, 1.8e-18 at N = 512), because the chunk boundary
sets the order the per-chunk reductions are summed in.  The windowed
(`window=5.0`) path, the FFT path and any call whose chunk was never bound are
untouched.  Entry points: `reconstruct_field_from_beamlets`,
`frame_completeness`, and `apply_real_lens_gbd` and its element family through
their own `mem_budget_mb`.

**New: the budget has a published floor, and below it the path is loud.**  The
chunk cannot go below one beamlet column, so
`lumenairy.propagators.gbd._dense_budget_floor_bytes(Ny, Nx)` --
`Ny*Nx*(48 + 128)` bytes, 11.534 MB at N = 256 and 46.137 MB at N = 512 -- is
the smallest budget that can be honoured.  Ask for less and the dense path now
emits a `RuntimeWarning` naming the floor and the two mitigations instead of
exceeding the request silently.  It warns rather than raises because at the
shipped `mem_budget_mb=512.0` default the floor binds on any square grid past
N = 1706, so refusing would break calls that complete today.

**Recipe -- keep 5.48.x exactly:**

```python
import lumenairy.propagators.gbd as _gbd
_gbd.DENSE_MEM_BUDGET_ACCOUNTING = 'legacy'   # byte-identical to 5.48.x
```

**Recipe -- if the floor notice fires:** raise `mem_budget_mb` to the floor the
message quotes, or pass `window=5.0` to `reconstruct_field_from_beamlets` for
the bounded-support scatter-add, whose accounting has no one-column floor at
these sizes.  `window=5.0` changes the returned field by its own truncation
(tail `exp(-25)`, about 1e-11), which is why it is not applied for you.

**Also:** an unrecognised `DENSE_MEM_BUDGET_ACCOUNTING` now raises instead of
falling through to `'legacy'`.  A typo would otherwise silently restore the
six-fold under-count the new default exists to remove.  The check runs only
where the budget arithmetic runs, so a caller who never passes `mem_budget_mb`
is unaffected.
