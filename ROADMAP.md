# LumenAiry Forward Roadmap

**Last updated:** 2026-05-17 (post-v4.14.0 PyPI ship).

This file captures the next-release scope for LumenAiry and its
Designer GUI.  Items are grouped by release target and prioritised
within each.  Each entry has a short rationale + scope estimate so
a future plan-phase agent dispatch can lift items directly into a
work plan.

Historical per-module limitation notes from the v4.9.0-era ROADMAP
are preserved in git history; this file is forward-only.

---

## Current state

- **Library:** v4.14.0 on PyPI.  710 → 858 unit tests added across
  v4.13.0 → v4.14.0; 34/34 validation files passing.  Public API
  at ~365 symbols in `lumenairy.__all__`.
- **Designer GUI:** v3.7.10 (per in-code comments at `ui/main_window
  .py:2196` etc.).  No standalone release stream; the Designer
  ships co-versioned inside the library wheel.
- **Audit closure status:** AUDIT_V4_12_1, AUDIT_V4_13_0,
  AUDIT_V4_13_1 + cross-library survey Part 10 all closed (Tier-0
  + Tier-1).  AUDIT_V4_13_1 Tier-2/3/4 scoped to v4.15+/v5.0 as
  noted below.
- **Active back-compat shims:** 8 (catalogued in AUDIT_V4_13_1 Part
  5).

---

## v4.15.0 — Targeted minor (3 well-scoped items)

### 1. Modal asymptotic per-pixel vectorisation + test-pin update

The audit's biggest known perf opportunity (target 20-100×).  v4.14.0
shipped 6 private vectorised helpers (`_solve_envelope_stationary
_batch` et al.) but did NOT switch the public `propagate_modal
_asymptotic` body because of a real physics finding:

> The pre-v4.14 warm-started Newton chain lands in **wrong-saddle
> basins** at grid edges, where the overflow guard `|b_quad| > 700`
> silently zeros those pixels.  The cold-start batched Newton finds
> the physical saddle uniformly — producing strictly more non-zero
> pixels (physically more correct) but breaking the existing 16-test
> `1e-10 rel` bit-equal pin at
> `tests/unit/test_perf_v4_12_0_asymptotic.py`.

**Scope:** Switch the public body to use the batched helpers; update
the test pin to acknowledge the corrected physical behaviour at
grid edges; document the warm-saddle-basin behaviour as a known
v4.12.x → v4.14.x bug being fixed.  ~150 LOC code + ~50 LOC of test
adjustment + a coordinated CHANGELOG note about the bit-changing
output.

**Expected speedup:** 20-100× on the modal asymptotic path.
Cascades into LG aberration tensor evaluation and Maslov stationary-
phase chains.

### 2. Source factory signature normalisation

`Source.gaussian(w0, N, dx, wavelength)` puts beam-size first;
`Source.plane_wave(N, dx, wavelength)` / `Source.point_source(...)`
put N first; `Source.top_hat(diameter, N, ...)` and
`Source.fiber_mode(mode_field_diameter, ...)` put diameter first.
Mixed within `sources/core.py:1055-1150`.  Caught by AUDIT_V4_13_1
Part 4 #10.

**Scope:** Convert size-args to keyword-only across all 5 factories
with a one-release `DeprecationWarning` on the positional form.
Pick one canonical order (`Source.method(*, N, dx, wavelength,
<size_kwargs>)` is the natural fit).  ~25 LOC code + 5 tests.

### 3. `system.evaluate(prescription, source, ...)` ergonomic entry

`propagate_through_system(E_in, elements, ...)` takes an element
list, not a prescription dict.  Users loading a Zemax `.zmx` file
have to build the element list manually before propagating.
Optiland and prysm expose this as a one-liner.

**Scope:** New `lumenairy.system.evaluate(prescription, source, *,
output_grid=None, output_dx=None, ...)` that:
1. Builds the element chain from the prescription via the existing
   `surfaces_from_prescription` + element-handler dispatch.
2. Calls `Source.to_source()` to get `E_in`, `dx`, `dy`,
   `wavelength`.
3. Routes through `propagate_through_system` and returns
   `PropagationResult`.

~80 LOC code + 3 tests + Tier-1 `__all__` export.

---

## v4.16.0 — User-facing API expansion

The cross-library survey (Agent C, Part 10 of AUDIT_V4_13_1)
catalogued 12 user-facing API gaps.  v4.14.0 closed 5 (encircled
energy, MTF cutoff, beam diameter, depth of focus, plot_wavefront).
The remaining 7 are below.

### 4. Polychromatic encircled energy

`polychromatic_psf` exists; `encircled_energy_radius` exists; no
convenience helper `ee_polychromatic(prescription, wavelengths,
weights, radii)` chaining them.  ~30 LOC + 2 tests.

### 5. Polarisation-aware Strehl / coupling

All `strehl_*` helpers take a scalar field.  For vector imaging
(Richards-Wolf, polarization-ray-tracing) the user has to manually
`sqrt(|Ex|² + |Ey|²)` first.  Add an explicit `strehl_vector(Ex, Ey,
Ez=None, ...)` and `coupling_efficiency_vector(...)`.  ~50 LOC + 3
tests.

### 6. Optical resolution metrics (Sparrow / Rayleigh / FWHM)

Standard "two-point separability" definitions absent.  Add:
- `rayleigh_resolution(psf, dx, wavelength)` — Rayleigh diffraction-
  limit two-point separation.
- `sparrow_resolution(psf, dx)` — empirical Sparrow criterion
  (dip-just-vanishes).
- `fwhm_resolution(psf, dx)` — twice the FWHM half-radius.

~60 LOC + 4 tests.

### 7. Astigmatism magnitude+angle helper

`zernike_decompose` returns `(c5, c3)` = (vertical, oblique)
astigmatism but no `astigmatism_mag_angle(coeffs)` that converts to
`(|astig|, theta)` per Mahajan §8.2.  ~10 LOC + 1 test.

### 8. Off-axis parabola helper

Users currently roll their own OAP via tilt+decenter, which is
fragile.  Add `make_off_axis_parabola(focal_length, off_axis_angle,
clear_aperture, ...) -> prescription_dict` factory.  ~40 LOC + 2
tests.

### 9. Partial-coherence source helpers (Schell-model, Gaussian-Schell)

`create_led_source` returns suggested source angles for Köhler
integration but no first-class object represents source spectral
or spatial coherence.  Add:
- `create_gaussian_schell_source(N, dx, wavelength, w0, sigma_g)` —
  spatially-incoherent Gaussian-Schell beam.
- `create_schell_model_source(N, dx, wavelength, intensity_profile,
  coherence_length)` — generic Schell-model.

~120 LOC + 4 tests.

### 10. Q-type freeform (Forbes Q-bfs / Q-con)

`elements/freeform.py` only implements XY-polynomial, Zernike, and
Chebyshev.  Forbes Q-type is the standard at TI / Edmund / Zemax for
aspheric freeforms.  ~150 LOC + 5 tests.

### 11. Ring / annular incoherent source

`create_annular_beam` is monochromatic coherent.  Add an angular-
spectrum version with non-zero source size for partial-coherence
integration.  ~50 LOC + 2 tests.

---

## v4.17.0 — Optimisation framework expansion

### 12. Constrained optimisation

All Merits express constraints as soft penalties (`max(0, x -
threshold)²`).  scipy.optimize has `NonlinearConstraint` for hard
constraints (e.g. "BFL > 5 mm exactly").  Add `constraints:
Optional[Sequence[Constraint]]` kwarg to `design_optimize` that maps
to scipy's API.  ~80 LOC + 4 tests.

### 13. Checkpoint / resume on long `design_optimize` runs

A 4-hour optimisation run that crashes loses everything.
`plane_logger` saves per-iteration field but not the parameter
vector.  Add `state_file: Optional[str] = None` that persists
`(call_count, x_best, merit_best)` to JSON/H5 and resumes from disk
when present.  ~100 LOC + 3 tests.

### 14. Multi-objective (Pareto) optimisation

`CompositeMerit` collapses to scalar.  For "minimise spot size AND
match focal length" with no a priori weight balance, NSGA-II or
`scipy.optimize.differential_evolution` with vector merit is the
right tool.  Out-of-scope architecturally for v4.17 unless a clean
shim onto an external library (`pymoo`) is acceptable.

### 15. Hessian / Newton-step optimisation

L-BFGS-B is the default.  For small (< 30 free var) problems an
FD-Hessian-based Newton step converges in fewer evals.  Add
`method='newton'` to `design_optimize`.  ~60 LOC + 2 tests.

---

## v4.18.0 — Glass / materials expansion

### 16. CDGM / Hikari / Sumita glass catalogues

`glass.py` ships Schott + partial Ohara (S-LAH 64/79).  CDGM is the
dominant Chinese flint/crown catalogue cited in cellphone/telephoto
lens papers; Hikari and Sumita complete the major-vendor matrix.
Public Sellmeier-coefficient sources exist at refractiveindex.info.
~600 LOC of bulk data + a sweep through `GLASS_REGISTRY`.

### 17. Sellmeier validity ranges per glass

`_sellmeier_index` checks for resonance singularity and negative
n² but doesn't carry a `(lambda_min, lambda_max)` validity-window
per glass.  Asking N-BK7 for n at 200 nm returns a number that
bears no relation to physical N-BK7.  Add a `validity` field to
`GLASS_REGISTRY` entries; warn (or raise) when extrapolating.  ~30
LOC + 2 tests + bulk-data sweep.

---

## v5.0.0 — Major structural release (breaking changes coordinated)

Held back from incremental releases because they're cross-cutting
and break public-API or test-organisation contracts.

### Architecture / housekeeping

* **6 file splits** (no public API change; mechanical reorganisation
  visible only to git-blame):
  - `raytrace/core.py` (4422 LOC) → split surface/intersection/
    trace/world-trace/Seidel/ray-fan/layout.
  - `propagators/propagation.py` (3710 LOC) → split FFT-infra/
    ASM/Fresnel/RS/SAS/MFT.
  - `propagators/asymptotic.py` (3597 LOC) → split modes/canonical-
    fit/aberration-tensor/Maslov/JAX-twin.
  - `optimize/core.py` (3258 LOC) → split parameterizations/merit-
    terms/wrapper-merits/context/driver/JAX-merits.
  - `io/prescriptions.py` (2829 LOC) → split builders/Zemax/CODE V/
    Quadoa/transforms.
  - `analysis/core.py` (2196 LOC) → split beam-stats/Strehl/PSF-
    MTF-OTF/polychromatic/Zernike/OPD.

* **CI gates** (currently absent):
  - `pytest tests/unit -m "not integration"` job (fast PR feedback).
  - `ruff check` lint gate.
  - `mypy --strict` incremental adoption starting with small
    modules (`backend/`, `_deprecation.py`, `_context.py`,
    `progress.py`, `memory.py`).
  - `tests/test_public_api.py` smoke test asserting every `__all__`
    entry is `getattr(lumenairy, name)`-resolvable.

* **Remove 8 active back-compat shims** (catalogued in
  AUDIT_V4_13_1 Part 5):
  - `analysis/analysis.py` (v4.7 rename shim)
  - `lumenairy/ao.py` (v4.3 shim)
  - `lumenairy/io/hdf5.py`
  - `elements/lenses.py:938-983` (v3.5.5 — 9 versions old)
  - `elements/elements.py:862-875` (v4.3)
  - `system.py:550-599` legacy aperture schema
  - `_deprecation.py` shims (v4.7)
  - `analysis/detector.py:288` deprecated `cosmic_ray_rate`

* **Shared Chebyshev helpers** extraction.  `propagators/asymptotic
  .py` and `elements/lenses_maslov.py` both import private
  Chebyshev helpers from `elements/lenses.py` — an inverted
  dependency.  Move to `lumenairy/_math/chebyshev.py`.

* **Audit-fix test-file consolidation**.  37 of 55 unit-test files
  are named `test_audit_fixes_v<X>_<Y>_<id>.py` (audit accretion).
  Merge into topical homes (`test_lens.py`, `test_propagation.py`,
  etc.); delete obsolete proxy tests (the `inspect.getsource`
  substring-match pattern caught in AUDIT_V4_13_1 Part 6.1).

* **`lumenairy/system.py` → `lumenairy/propagators/system.py`**
  (it walks elements applying propagators — functionally a
  propagator, not a top-level peer).

* **Move audit `.md` files** from repo root to `docs/audits/`.
  Repo root has 14 `.md` files; audit docs (8 of them) belong in
  a subdir.

* **`__init__.py` reorganisation** so the import section's tier
  order matches `__all__`'s tier order (currently mismatched by
  ~5 tiers — see AUDIT_V4_13_1 Part 5 `__init__.py audit`).

* **Off-axis conic in surface frame** (not just decenter+tilt).
  `apply_real_lens` honours `decenter` and `tilt` keys but adds
  them to the field's coordinates, not the surface's frame.
  Tilted/displaced asphere with proper sag in surface frame
  requires a coordinate transformation Optiland and Zemax do
  natively.

* **Bump `requires-python` to >=3.10** (Python 3.9 EOL was
  2025-10).

### Config knobs

* `set_default_wave_propagator` — avoid passing
  `wave_propagator='gbd'` to every entry point.
* `set_default_real_dtype` — only complex has a knob today.
* `set_default_dy` — match v4.13.x anamorphic-threading direction.

### Documentation

* `validation/README.md` — currently absent; new contributors don't
  know whether to add tests to `tests/unit/` or `validation/`.
* `Migration-Guide.md` — currently missing; v4.13.0's `rcwa.py →
  thin_grating.py` rename and the `wavelength`-required-on-codegen
  are breaking changes that need migration prose.
* Convention table — one-stop summary of sign conventions
  (`exp(-i*omega*t)`, mirror radius wave-side `R>0=concave`,
  Welford signed-R for raytrace, OPD sign, etc.) — currently each
  is documented in the place it matters but no single doc page.
* Split README.md (3475 lines) — move deep cookbook to
  `docs/cookbook.md`.
* CHANGELOG.md (7000+ lines) — consider yearly / per-major-version
  archive splits.
* 5 missing examples: multi-config / zoom, tolerancing, coronagraph
  workflow, AO closed loop, ghost / stray-light.

---

## Designer GUI (separate version stream)

The Designer ships co-versioned inside the library wheel.  Current
in-code references: 3.7.6, 3.7.7, 3.7.9, 3.7.10 — latest is
**3.7.10**.  No active Designer plan since the prior `valiant-
hopping-dream` plan (3.6.1 + 3.7 + 3.8) was completed.

Possible v3.8+ scope (not yet planned):

* **Polarization plotting docks** — none currently surface Jones-
  pupil and Stokes maps from the library's `polarization.py` /
  Richards-Wolf paths.
* **Coronagraph workflow dock** — `analysis/coronagraph.py` has
  `coronagraph_contrast_curve` but no dedicated dock to set up
  the 4-stop chain (Lyot focal mask → Lyot stop → apodised pupil)
  interactively.
* **Tolerancing dock** — `monte_carlo_tolerancing` exists but the
  UI surface is limited; a dedicated "perturbation knobs + run MC"
  dock is the canonical Zemax pattern.
* **Multi-config / zoom dock** — `optimize/multiconfig.py` exists
  but no UI; users build configs in code.
* **Wavefront-map plot integration** — v4.14.0 added
  `plot_wavefront`; no dock surfaces it yet.
* **`CancellableProgress` UI button** — v4.13.1 added the
  cancellation protocol, wired into all 4 scipy callbacks; needs
  a Stop-button surface in the optimisation dock.

---

## Opportunistic / lower-priority items

Catalogued from prior audits + cross-library survey, not allocated
to a specific release:

* **`_deprecation.py` orphan helpers** — `warn_deprecated_kwarg`,
  `warn_renamed_function`, `warn_deprecated_default` are exported
  but never called.  Either delete or wire into kwarg renames
  introduced in v4.7-v4.13.
* **Duplicate `_xp_of`** in `elements/elements.py:27` and `analysis
  /core.py:52`.  Consolidate to direct `from ..backend import
  array_namespace`.
* **`backend/fft.py` → `propagators/propagation.py` inversion** —
  FFT plan caches + toggle flags live in `propagation.py`, but
  `backend/fft.py` imports from there.  Lift to `propagators/
  fft_infra.py`.
* **Source-text-proxy tests** in v4.13.1's Agent 3 file — 4 tests
  use `inspect.getsource` substring matching instead of behavioural
  assertions.  Replace with real behaviour pins.
* **`MultiFieldMerit` performance ceiling** — meshgrid cache hit
  1.19× at N=128; the per-field `np.exp` + `np.where` calls
  dominate and cannot be cached.  Future: JIT-compile the per-
  field tilt path via Numba or JAX.
* **`output_grid` parameter semantics inconsistency** (AUDIT_V4_13_1
  Part 2 P1-A) — dispatcher documents `(N, dx_out)` but 3 sub-
  propagators (gbd/hfpi/hf) interpret as `(Ny, Nx)`.  Pick one;
  rename the sub-propagator kwarg to `output_shape` for the
  `(Ny, Nx)` case.
* **MHS subdomain grid loss** (AUDIT_V4_13_1 P1-C) —
  `prescription_subdomain` with default `method='maslov'` silently
  outputs on input grid (compounds dispatcher P1).
* **Partition-of-unity convention bug** in `subaperture.py:140-148`
  (AUDIT_V4_13_1 P1-F).  Window centred on source-plane positions;
  only correct for unit-mag no-tilt.
* **`apply_doe_phase_traced`** discards sign of `N` but `trace()`
  inline DOE kick preserves it (AUDIT_V4_13_1 P1-G).  ~2 LOC fix.
* **`MultiPrescriptionParameterization.scale_floor`** not supported
  (AUDIT_V4_13_1 P1-1).  ~20 LOC + path-aware defaults per opt
  variable type (radii / thicknesses 1e-6, conics 1e-3, aspheric
  α_n 1e-3/factor).
* **`logging` adoption** — currently 1 use across the whole library
  (`propagators/propagation.py:592`); 42 `warnings.warn` calls
  across 22 files.  Long-running paths (`apply_real_lens_traced`,
  `design_optimize`, `monte_carlo_tolerancing`) have no per-
  iteration telemetry.

---

## Recommended sequencing

Stack v4.15 → v4.16 → v4.17 → v4.18 as a sequence of focused
minors landing over weeks, then plan v5.0 as a coordinated
breaking-change release when the v4.15+ minor sequence stabilises.

- **v4.15** is the smallest and most physics-significant — closing
  out the modal-asymptotic finding is a real engineering win and
  preconditions any future LG-aberration-tensor work.
- **v4.16** is the highest user-visible-value release — closing 7
  API gaps that experienced users routinely ask for.
- **v4.17** is the optimisation-framework focus — constrained opt
  + checkpoint/resume + Newton-step.
- **v4.18** is the glass / materials expansion — broaden vendor
  coverage and tighten extrapolation safety.
- **v5.0** is the major-structural release — file splits + CI
  gates + back-compat shim removal + Migration guide.  Coordinate
  timing with a Designer 4.0 if scope warrants.

---

## Shipped highlights (since v4.9.0)

(Brief; the full per-release breakdown is in
[`CHANGELOG.md`](CHANGELOG.md).)

- **v4.10–v4.11.x** — comprehensive multi-agent physics audit
  response (~100+ findings).  Welford-mirror convention,
  C-LR-1 saga, raytrace + GBD + HF + subaperture + sources +
  detector closures, Sellmeier registry expansion.
- **v4.12.x** — Tier-1 perf wins (ASM 4.3×, jit caches 36-163×),
  pre-PyPI audit closure, cache hygiene infrastructure (7 LRU
  caches + `lumenairy_context(clear_caches_on_exit=True)`).
- **v4.13.0** — S1/S2/S3 + L2/L3/L4/L6/L8 audit closure,
  `except Exception:` sweep (99 → 3 sites), Tier-2 perf bundle
  (188× BSDF TIS, 10× thin-grating, 10.8× SH FFT batching,
  4-72× Seidel field sweep), `rcwa.py` → `thin_grating.py`
  rename.
- **v4.13.1** — AUDIT_V4_13_0 closure (3 sibling-gap P1s + 9 P2 +
  6 P3); 3 new perf wins (SH scatter 9.5-25×, vec-acc 1.65×,
  GBD reconstruct 1.2-1.5×); `CancellableProgress` cancellation
  protocol wired into 4 scipy callbacks.
- **v4.13.2** — AUDIT_V4_13_1 Tier-0 (12 P1s + 5 cross-survey P0s
  + thin-lens sibling sweep + latent CuPy dispatch bug).
- **v4.14.0** — AUDIT_V4_13_1 Phase B (7 Tier-1 perf wins
  including 24.6× coatings + 77× LG-mode cache + 6.17×
  Multi*Merit cache; 6 new public functions: encircled energy,
  MTF cutoff, beam diameter, depth of focus, plot_wavefront;
  80 parametrized dispatcher pins closing the sibling-gap
  audit-meta-finding).

---

## How to update this file

When an item ships, move it from its release section to the
"Shipped highlights" section above with a one-line summary.  When
a new audit or cross-library survey adds items, append them under
the appropriate release target.  When v5.0 lands, archive this
file to `docs/roadmaps/ROADMAP_v4.md` and start fresh with the
v5.x forward plan.
