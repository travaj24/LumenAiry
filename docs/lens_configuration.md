# Configuring the `apply_real_lens` family

`lumenairy` ships seven real-lens entry points that share a contract —
`(E_in, *, prescription, wavelength, dx, …) -> field at the exit plane on the
input grid` — and, between them, about 110 distinct keyword arguments:

| entry point | keyword-only parameters |
|---|---|
| `apply_real_lens` | 31 |
| `apply_real_lens_traced` | 51 |
| `prepare_real_lens_traced` | 28 |
| `apply_real_lens_maslov` | 32 |
| `apply_real_lens_gbd` | 29 |
| `apply_real_lens_fga` | 24 |
| `apply_real_lens_traced_multibranch` | 14 |

(Measured 2026-09-12 from the live signatures; the counts include the four
configuration parameters described below, and exclude the positional `E_in`.)

Since v5.45.2 each of them also accepts **configuration objects** —
`LensGeometry`, `LensNumerics`, `LensResources`, and the `LensConfig` that
holds all three — so a set of settings can be built once, checked, stored,
compared and reused instead of being respelled at every call site.

**Nothing was deprecated.** Every keyword still works, with the same default.
A call that passes none of `geometry=` / `numerics=` / `resources=` / `config=`
runs byte-for-byte the code it ran before; the four new parameters cost four
`is not None` tests.

---

## The short version

```python
import lumenairy as la

numerics = la.LensNumerics(ray_subsample=2, newton_poly_order=8,
                           newton_fit='spline')
resources = la.LensResources(n_workers=4, parallel_amp=False)

E_out = la.apply_real_lens_traced(
    E_in, prescription=rx, wavelength=633e-9, dx=dx,
    numerics=numerics, resources=resources)
```

which is exactly (and bit-identically) the same call as

```python
E_out = la.apply_real_lens_traced(
    E_in, prescription=rx, wavelength=633e-9, dx=dx,
    ray_subsample=2, newton_poly_order=8, newton_fit='spline',
    n_workers=4, parallel_amp=False)
```

The three objects are **frozen dataclasses**: hashable-by-value, comparable,
`repr`-able, and safe to keep in a module-level table of study conditions.

---

## Why three objects, and where the line is drawn

The partition is by **role**, not by which function happens to take the
keyword:

* **`LensGeometry` — *what problem is being solved*.** Where the output plane
  is, what medium it sits in, which reference congruence the phase is measured
  against, which surface model, whether the aperture clips. Change one of
  these and the answer changes *because the problem changed*.
* **`LensNumerics` — *how that same problem is discretised*.** Sampling,
  fit orders, iteration caps, the diffraction kernel, band-limiting, the
  amplitude model, the caustic strategy. Change one of these and the answer
  should move only by its own truncation error.
* **`LensResources` — *what machine the call may use, and what it reports*.**
  GPU, workers, chunk sizes, memory stores, scratch directories, progress and
  verbosity. Nothing here may change the returned field. `sag_dtype` is the
  one deliberate exception and says so in its docstring: it trades ~1e-7
  relative surface departure for half the dtype-independent memory core, which
  is why it carries an accuracy warning rather than living in `LensNumerics`.

38 of the family's parameters are fields; the rest stay keyword-only, each for
a written reason (tabled below). The gate that keeps this honest is
`tests/unit/test_audit2609_a16_lens_config_round_trip.py`, which walks the live
signatures and fails if any parameter is neither a field, nor one of the four
contract names (`prescription`, `wavelength`, `dx`, `N`), nor a documented
exclusion. A parameter can be *excluded*; it cannot be *forgotten*.

---

## The precedence rule

A config field and an explicit keyword for the same setting **must agree, or
the call raises** (with the `CONVENTIONS.md` §2 `f"{fn_name}: …"` prefix).
Concretely, for each setting:

* the **field** is a REQUEST iff it differs from its dataclass default;
* the **keyword** is a REQUEST iff it differs from its signature default;
* two requests that disagree → `ValueError`;
* exactly one request → that value;
* neither → the entry point's own default, i.e. the historical behaviour.

The two defaults are identical by construction — a test asserts it for every
(entry point, field) pair — so the comparison is well defined.

Two consequences worth knowing:

1. **A config whose fields are all at their defaults is indistinguishable from
   passing no config at all.** That is what makes the configured path
   bit-identical for free, and it is asserted for every entry point.
2. **A field left at its default is therefore not a request, and cannot
   conflict.** Writing `LensNumerics(bandlimit=True)` next to `bandlimit=False`
   is not an error; the keyword simply wins. This is a deliberate trade
   (recorded in `test_a_field_left_at_its_default_is_not_a_request`): the
   alternative is tracking which fields were *passed* to `__init__`, which
   makes `dataclasses.replace` and unpickling ambiguous for no real gain.

### A setting the entry point does not have RAISES

```python
>>> cfg = la.LensConfig.from_kwargs(output_plane_distance=2e-3,
...                                 newton_poly_order=4)
>>> la.apply_real_lens_maslov(E_in, prescription=rx, wavelength=lam, dx=dx,
...                           config=cfg)
ValueError: apply_real_lens_maslov: numerics.newton_poly_order=4 is not a
setting apply_real_lens_maslov accepts (it applies to: apply_real_lens_traced,
prepare_real_lens_traced).  Passing it here would silently discard it.  Use
config.narrowed_to('apply_real_lens_maslov') to drop the fields this entry
point has no parameter for, or move the setting to the entry point that owns
it.
```

This is the point of the objects. The 2026-09-11 audit's headline finding for
this family was a *silently discarded knob*; a config that quietly dropped what
it could not apply would reproduce it at a larger scale.

`narrowed_to` is the explicit "yes, drop them":

```python
>>> cfg.narrowed_to('apply_real_lens_maslov').to_kwargs()
{'output_plane_distance': 0.002}
```

---

## Worked example — one output plane, four engines

The thing that was impractical with 25-to-52-parameter signatures: compare the
four wave models at the same plane, with one description of the plane.

```python
import numpy as np
import lumenairy as la

rx = la.load_prescription('AC254-050-A')          # any prescription
E_in, dx, lam = gaussian_input()                  # your source

cfg = la.LensConfig.from_kwargs(
    # geometry: where we are looking
    output_plane_distance=2.0e-3, output_plane_n=1.0,
    # numerics: how hard each engine should try
    newton_poly_order=8, amplitude_model='ray_density',
    caustic='multibranch', caustic_ray_subsample=1,
    # resources: this box
    n_workers=4,
)

fields = {}
for name, fn in (('analytic',    la.apply_real_lens),
                 ('traced',      la.apply_real_lens_traced),
                 ('maslov',      la.apply_real_lens_maslov),
                 ('gbd',         la.apply_real_lens_gbd),
                 ('fga',         la.apply_real_lens_fga)):
    fields[name] = fn(E_in, prescription=rx, wavelength=lam, dx=dx,
                      config=cfg.narrowed_to(fn.__name__))

for name, E in fields.items():
    print(f'{name:9s} P = {np.sum(np.abs(E)**2) * dx * dx:.6e}')
```

Each engine receives only the settings it understands — `apply_real_lens` gets
none of the caustic group and no `output_plane_distance` (it has no such
parameter), `apply_real_lens_fga` gets `output_plane_distance` but not
`output_plane_n`, and only the traced engine gets the Newton and caustic
settings. `cfg.narrowed_to(...).to_kwargs(entry_point=...)` shows exactly what
each one will see.

### Round trips

```python
>>> cfg = la.LensConfig.from_kwargs(ray_subsample=2, n_workers=4)
>>> cfg.to_kwargs()
{'ray_subsample': 2, 'n_workers': 4}
>>> la.LensConfig.from_kwargs(**cfg.to_kwargs()) == cfg
True
>>> cfg.to_kwargs(entry_point='apply_real_lens_traced')
{'ray_subsample': 2, 'n_workers': 4}
>>> cfg.to_kwargs(include_defaults=True)          # all 38 fields
{...}
```

`from_kwargs` and `to_kwargs` both take `entry_point=` and then speak that
entry point's own spelling, which matters for the two renamed settings below.

### Validation happens where you build the object

```python
>>> la.LensNumerics(remap_order=2)
ValueError: LensNumerics: remap_order=2 is not a valid choice.  Choose from
[1, 3, 5].
>>> la.LensResources(sag_dtype=np.float16)
ValueError: LensResources: sag_dtype=<class 'numpy.float16'> must be float32 or
float64 (the geometry lineage is real).
```

No field, no prescription, no call. Field-local range and enum checks run in
`__post_init__`; **cross-field and prescription-dependent rules stay in the
entry point** (`surface_model='displaced'` vs `fresnel`, `caustic` vs
`amplitude_model`, the undersample guard, the enum vocabularies that live
inline inside the 5 700-line analytic body). The config object does not restate
them — it would be a second implementation that could drift.

One place the config is *stricter* than the keyword: `sag_dtype`. The keyword
path resolves an unrecognised dtype silently to float64; the config refuses it,
because a config object that accepted `np.float16` and quietly gave float64
would be the discarded-setting failure this module exists to close.

---

## The field partition

`Y` = the entry point takes this setting under the same name;
`` `name` `` = it takes it under that different name (see "Naming mismatches"
below); `--` = it has no such parameter, and setting the field will raise if
the config is passed to it un-narrowed.

### `LensGeometry` (10 fields)

| field | default | analytic | traced | prepare | maslov | gbd | fga | multibranch |
|---|---|---|---|---|---|---|---|---|
| `dy` | None | Y | Y | -- | Y | Y | Y | -- |
| `output_plane_distance` | 0.0 | -- | Y | -- | Y | Y | Y | Y |
| `output_plane_n` | 1.0 | -- | -- | -- | Y | Y | -- | Y |
| `conjugate` | None | Y | -- | -- | -- | -- | -- | -- |
| `surface_model` | `'thin'` | Y | -- | -- | -- | -- | -- | -- |
| `clip_aperture` | True | -- | -- | -- | -- | Y | -- | -- |
| `carrier` | None | Y | Y | Y | -- | -- | -- | -- |
| `origin` | (0.0, 0.0) | -- | Y | -- | -- | -- | -- | -- |
| `beam_centre` | None | -- | Y | -- | -- | -- | -- | -- |
| `roi` | None | -- | -- | -- | Y | Y | -- | -- |

### `LensNumerics` (17 fields)

| field | default | analytic | traced | prepare | maslov | gbd | fga | multibranch |
|---|---|---|---|---|---|---|---|---|
| `bandlimit` | True | Y | Y | Y | -- | -- | -- | -- |
| `wave_propagator` | None | Y | Y | Y | -- | -- | -- | -- |
| `ray_subsample` | 8 | -- | Y | Y | -- | -- | -- | -- |
| `output_subsample` | 1 | -- | -- | -- | Y | Y | -- | -- |
| `remap_order` | 3 | Y | -- | -- | -- | -- | -- | -- |
| `min_coarse_samples_per_aperture` | 32 | -- | Y | Y | -- | -- | -- | -- |
| `fit_radius_beam_factor` | None | -- | Y | Y | -- | -- | -- | -- |
| `newton_fit` | `'auto'` | -- | Y | Y | -- | -- | -- | -- |
| `newton_poly_order` | 6 | -- | Y | Y | -- | -- | -- | -- |
| `newton_max_iters` | None | -- | Y | Y | -- | -- | -- | -- |
| `inversion_method` | `'newton'` | -- | Y | Y | -- | -- | -- | -- |
| `amplitude_model` | `'screen'` | -- | Y | Y | -- | -- | -- | -- |
| `caustic` | None | -- | Y | -- | -- | -- | -- | -- |
| `caustic_band` | `'ludwig'` | -- | Y | -- | -- | -- | -- | Y |
| `caustic_ray_subsample` | 2 | -- | Y | -- | -- | -- | -- | `ray_subsample` |
| `caustic_min_area_ratio` | 1e-06 | -- | Y | -- | -- | -- | -- | `min_area_ratio` |
| `inverse_map` | None | -- | Y | Y | -- | -- | -- | -- |

### `LensResources` (11 fields)

| field | default | analytic | traced | prepare | maslov | gbd | fga | multibranch |
|---|---|---|---|---|---|---|---|---|
| `use_gpu` | False | Y | Y | Y | Y | -- | -- | -- |
| `amp_use_gpu` | False | -- | Y | Y | -- | -- | -- | -- |
| `n_workers` | None | -- | Y | Y | -- | -- | -- | -- |
| `parallel_amp` | None | -- | Y | -- | -- | -- | -- | -- |
| `parallel_amp_min_free_gb` | 48.0 | -- | Y | -- | -- | -- | -- | -- |
| `sag_dtype` | None | Y | Y | Y | -- | -- | -- | -- |
| `sag_chunk_rows` | None | Y | Y | Y | -- | -- | -- | -- |
| `accumulator_store` | `'ram'` | Y | -- | -- | -- | -- | -- | -- |
| `scratch_dir` | None | Y | -- | -- | -- | -- | -- | -- |
| `progress` | None | Y | Y | Y | Y | Y | -- | -- |
| `verbose` | False | -- | -- | -- | Y | Y | -- | -- |

---

## Naming mismatches found while building the partition

Deriving the tables from the real signatures surfaced five places where the
family says the same thing in different words, or different things in the same
words. None of them was changed — renaming a shipped keyword is a migration,
not a refactor — but each is recorded here and encoded in the tables.

1. **`apply_real_lens_traced_multibranch.ray_subsample` is NOT
   `apply_real_lens_traced.ray_subsample`.** The multibranch one is the
   *caustic* launch spacing (default 2) and is the sibling of the traced
   engine's `caustic_ray_subsample` (also 2). The traced engine's own
   `ray_subsample` is the OPL coarse-grid spacing and defaults to 8. The config
   field is `caustic_ray_subsample`, and `from_kwargs(entry_point=
   'apply_real_lens_traced_multibranch', ray_subsample=…)` maps to it.
   A caller who moved a `ray_subsample=8` from one function to the other
   would silently quadruple the caustic launch density.
2. **`multibranch.min_area_ratio` is `traced.caustic_min_area_ratio`** (both
   1e-6). Same setting, two spellings.
3. **`multibranch.input_carrier` is NOT `carrier`.** It is a transverse carrier
   *wavevector* (`None | 'auto' | (kx, ky)` in rad/m); `carrier=` on the
   analytic and traced entry points is a reference *congruence*
   (`None | 'auto' | conjugate distance in m | wavefront ndarray |
   TiltedCarrier`). Mapping them onto one field would silently reinterpret
   metres as rad/m, so `input_carrier` stays keyword-only.
4. **`normalize_output` has two different defaults**: `'power'` on
   `apply_real_lens_maslov`, `'none'` on `apply_real_lens_gbd` and
   `apply_real_lens_fga`. A single field cannot carry two honest defaults, so
   it is keyword-only. Unifying it would be a behaviour change for one of the
   three and needs its own migration note.
5. **The work-chunk size is spelled three ways with three defaults**:
   `chunk_v2=64` (maslov), `chunk_beamlets=2048` (gbd), `chunk=None` (fga).
   Same reason, same resolution. `mem_budget_mb` is a fourth instance
   (`512.0` on gbd, `None` on fga).

---

## Deliberately keyword-only

| entry point | keyword | why |
|---|---|---|
| `analytic` | `absorption` | analytic-model physics option — see "Deferred: `LensPhysics`". |
| `analytic` | `displaced_mode` | legal only under `surface_model='displaced'` and validated against it by `_check_displaced_support`; an independent field could declare an illegal pair only the call can adjudicate. |
| `analytic` | `displaced_obliquity` | same — a sub-mode of `surface_model`. |
| `analytic` | `fresnel` | analytic-model physics option — see "Deferred: `LensPhysics`". |
| `analytic` | `on_screen_obliquity` | policy knob (`'warn'`/`'error'`/`'silent'`) for one call's diagnostics, not a setting of the optical problem. |
| `analytic` | `screen_obliquity` | legal only with `carrier=`; adjudicated by `_check_screen_obliquity_support` against the prescription. |
| `analytic` | `seidel_correction` | analytic-model physics option — see "Deferred: `LensPhysics`". |
| `analytic` | `seidel_poly_order` | governs `seidel_correction`, which is itself keyword-only; a numerics field whose enabling flag is not a field would be half-configurable. |
| `analytic` | `slant_correction` | analytic-model physics option — see "Deferred: `LensPhysics`". |
| `analytic` | `stream_transfer_function` | memory/streaming strategy of the analytic in-glass leg only; no sibling, no family-wide meaning. |
| `analytic` | `surface_frame` | analytic-model physics option — see "Deferred: `LensPhysics`". |
| `traced` | `_exit_na_out`, `_remap_launch_out`, `_imap_out` | private diagnostics sinks: underscore-prefixed MUTABLE out-parameters the call writes into. A frozen config must not carry one — sharing a config between two calls would make them share a sink. |
| `traced` | `on_undersample`, `on_noncollimated`, `on_aperture_beam`, `on_fit_domain_basis`, `on_pool_memory` | per-call diagnostic policy knobs. |
| `traced` | `preserve_input_phase`, `remap_sampling`, `tilt_aware_rays`, `decentred_fit_poly_order`, `newton_amp_mask_rel`, `newton_mask_dilate_coarse_px`, `fast_analytic_phase` | tuning constants of this model only; no sibling with the same name and semantics. |
| `traced` | `return_screen` | changes the RETURN TYPE (`field` → `(field, screen)`); a setting that changes what a function returns belongs at the call. |
| `prepare` | `on_undersample`, `on_noncollimated` | per-call diagnostic policy knobs. |
| `maslov` | `ray_field_samples`, `ray_pupil_samples`, `poly_order`, `n_v2`, `extract_linear_phase`, `use_numexpr`, `integration_method`, `stationary_newton_iter`, `stationary_newton_tol`, `local_n_samples`, `local_window_sigma`, `levin_tol`, `collimated_input`, `input_na`, `fold_split` | tuning constants of this model only. |
| `maslov` | `chunk_v2` | work-chunk size — see "Naming mismatches" item 5. |
| `maslov`, `gbd`, `fga` | `normalize_output` | DEFAULT CLASH — see "Naming mismatches" item 4. |
| `gbd` | `sample_step`, `beamlets_per_aperture`, `waist_factor`, `direction_sampling`, `reexpand`, `reexpand_carrier`, `reexpand_threshold`, `per_surface`, `jacobian`, `window` | beamlet-frame constants of this model only. |
| `gbd` | `chunk_beamlets` | work-chunk size — see item 5. |
| `gbd`, `fga` | `mem_budget_mb` | DEFAULT CLASH — see item 5. |
| `gbd` | `diagnostics` | MUTABLE out-parameter (the call fills the dict the caller passes). |
| `fga` | `w0_factor`, `dq_step`, `p_max`, `n_p`, `nsig`, `prune_frac`, `coeff_frac`, `separable`, `coarse_stride`, `exact_jacobian`, `cache_trace`, `momentum_sampling` | swarm tuning constants of this model only. |
| `fga` | `chunk` | work-chunk size — see item 5. |
| `multibranch` | `input_carrier` | NOT `LensGeometry.carrier` — see "Naming mismatches" item 3. |
| `multibranch` | `return_diagnostics` | changes the RETURN TYPE — see `traced.return_screen`. |

---

## Deferred: `LensPhysics`

Five `apply_real_lens` keywords — `fresnel`, `absorption`, `slant_correction`,
`seidel_correction`, `surface_frame` — are *physics-model* options: they change
which terms the per-surface screen carries. That is a fourth role, and the
audit's design named three. Rather than push them into the wrong one (they are
not geometry, and they are certainly not resources), they stay keyword-only in
this pass.

The design, if it is wanted: a fourth frozen dataclass `LensPhysics` with those
five fields plus `seidel_poly_order`, added to `LensConfig` as a fourth
component and to `_GROUPS` — the resolver, `from_kwargs`, `to_kwargs`,
`narrowed_to` and every test walk over `_GROUPS`, so nothing else changes.
The work is the docstrings and the validation (the five are mutually
constrained: `slant_correction` and `seidel_correction` replace the *same*
per-surface coefficient and stacking them double-counts the facet obliquity —
measured 173.5 → 1488.6 nm rms exit OPD on an 8 mm cemented doublet — and all
five are refused under `surface_model='displaced'`). Because those constraints
are *cross-field*, `LensPhysics.__post_init__` could carry the pairwise ones
and the entry point would keep the prescription-dependent ones. Effort ~4 h.

---

## Module layout

The lens family carries four genuine module↔module import cycles that execute
at import time (audit 2026-09-11, TESTS-ARCH "Import cycles"; re-measured
2026-09-12 with a module-level-only AST walk, which reproduces the audit's list
exactly):

```
_lens_real  <-> lenses
_lens_thin  <-> lenses
_lens_traced <-> lenses
lenses      <-> lenses_maslov
```

`lenses.py` is the family's public re-export hub: it imports all eight sibling
modules at module scope. The back-edges are small and completely enumerated:

| back-edge | what it imports from `lenses` |
|---|---|
| `_lens_real:146` | `_warn_if_aperture_exceeds_grid`, `surface_sag_biconic`, `surface_sag_general` |
| `_lens_traced:608` | `_warn_if_aperture_exceeds_grid` |
| `_lens_thin:39` | `CUPY_AVAILABLE` (plus a module alias for the lazy `cp`) |
| `lenses_maslov:118` | `NUMEXPR_AVAILABLE`, `_ensure_numexpr_loaded`, `_fit_normaliser`, `_multi_indices_total_degree`, `_warn_if_aperture_exceeds_grid` |

`lens_config.py` is deliberately a **leaf**: it imports nothing from
`lumenairy` at module scope (its validators borrow `_lens_real`'s vocabulary
tuples through an in-function import, cached), so the seven entry points can
depend on it without adding an edge in the other direction. Adding it created
no new cycle — verified by the same walk.

### The plan for the four cycles

Extract the shared leaf `elements/_lens_kernels.py` holding exactly the eight
names above, and repoint the four back-edges at it. `lenses.py` keeps
re-exporting all eight so every existing `from lumenairy.elements.lenses import
surface_sag_general` keeps working, and the shell-vs-canonical walker
(`tests/unit/test_v5_2_walker_shell_vs_canonical.py`) sees the same surface.

Not done in this pass, for two reasons, both concrete:

1. **It cannot be proved bit-identical by relocation alone.**
   `surface_sag_general` and `surface_sag_biconic` are several hundred lines
   each with their own accumulated guards; moving them is a large diff whose
   only defensible gate is the `-k real_lens` slice plus the WP-A2/A3/A4 files,
   and the audit itself budgets 3 days for it.
2. **It collides with the history relocation.** WP-A17 part 2 is moving the
   `v<N>.<N> (audit …)` narrative blocks out of exactly these files (the audit
   measures −10 500 lines, `_lens_traced.py` 13 228 → ~8 300). Two large
   mechanical diffs over the same regions in the same round is how a
   relocation loses a guard.

**One of the four is already a two-line fix and should be taken first:**
`_lens_thin` needs only `CUPY_AVAILABLE` and the lazy `cp` from `lenses`, and
both now live in `lumenairy/backend/_optional.py`. Repointing
`_lens_thin.py:39` at `..backend._optional` breaks `_lens_thin <-> lenses`
outright, with no code motion at all.

---

## See also

* `lumenairy/elements/lens_config.py` — the module, with a per-field docstring
  carrying units and defaults.
* `tests/unit/test_audit2609_a16_lens_config_round_trip.py` — the structural
  gates (table ↔ signature agreement, parameter classification, round trips,
  validation, precedence).
* `tests/unit/test_audit2609_a16_lens_config_bit_identity.py` — the configured
  call equals the keyword call, for every entry point, on the WP-A15a
  covering-array fixture.
* `tests/unit/test_audit2609_a15a_lens_covering_array.py` — the pairwise
  combination coverage the config objects make easy to extend.
