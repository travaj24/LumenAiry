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

## 4.16.2 -- Default-config knobs (API-only; v5.0 rollout)

Three new library-wide setter functions land alongside the existing
`set_default_complex_dtype`:

* `set_default_real_dtype(np.float32 | np.float64)` -- real-array
  precision.  v4.16.3 wires this knob through `propagate_ensemble`'s
  no-input-dtype real-accumulator fallback (the canonical
  `in_dtype is None` path); full library-wide rollout follows in v5.0.
* `set_default_wave_propagator(name)` -- default `wave_propagator`
  for `propagate_through_system` / `apply_real_lens` / etc.
  **API ONLY in v4.16.2/v4.16.3**: stored but not yet read by any
  library code.  Setter emits a one-shot `UserWarning` (v4.16.3+).
* `set_default_dy(value)` -- default anamorphic grid spacing.
  **API ONLY in v4.16.2/v4.16.3**: same status as
  `set_default_wave_propagator`.

> **v4.16.2/v4.16.3 limitation note.**  Two of the three new knobs
> (`set_default_wave_propagator`, `set_default_dy`) store the default
> but have **zero downstream consumers** in `lumenairy/` at v4.16.2
> ship.  Entry points like `apply_real_lens` continue to hardcode
> `wave_propagator='asm'` and accept `wave_propagator=...` / `dy=...`
> as per-call keyword arguments.  The library-wide resolver rollout
> that makes these setters actually steer the default at every entry
> point is staged for v5.0 alongside the file-split work.  Until
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
# is API-only at v4.16.2/v4.16.3; see limitation note above):
field = la.apply_real_lens(field, prescription=pres, wavelength=wl,
                            dx=dx, wave_propagator='fresnel')
field = la.apply_real_lens_traced(field, ..., wave_propagator='fresnel')
```

The full library-wide resolver rollout (replacing hardcoded `'asm'`
/ `np.float64` / `dx` defaults at every entry point) is staged for
v5.0 alongside the file-split work.

---

## v5.0 (planned)

v5.0 is the major-structural release.  Migration items expected:

* 6 large-file splits (no public API change; `git blame`-only
  reorg).
* Removal of 8 active back-compat shims (most are 4+ versions old).
* `lumenairy/system.py` -> `lumenairy/propagators/system.py`.
* `requires-python` bump from `>=3.9` to `>=3.10`.
* `MCF` object with `coherence_at(...)` (the explicit-MCF
  alternative to the ensemble path).
* Off-axis conic in surface frame (proper coordinate transformation;
  current `decenter`+`tilt` work in field frame).
* Full library-wide rollout of the v4.16.2 default-config knob
  resolvers.

When v5.0 ships, this guide will gain a v5.0 section with concrete
recipes for each migration point.
