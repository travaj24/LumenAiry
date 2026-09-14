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

The four objects are **frozen dataclasses**: comparable by value, `repr`-able,
picklable, and safe to keep in a module-level table of study conditions. They
are also hashable by value — with the one exception every Python object shares,
that an instance holding an **ndarray** field (a wavefront `carrier`, an array
`beam_centre`) is unhashable, because `hash(ndarray)` is. Equality still works
on those: `==` compares field by field through the same `_same` helper the
resolver uses, so an array-valued field compares with `np.array_equal` instead
of raising "truth value is ambiguous".

---

## Why four objects, and where the line is drawn

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
* **`LensPhysics` — *which terms the model carries*.** The analytic screen's
  model-term switches: the Fresnel transmittances, bulk absorption, the
  angle-true refraction OPD, the Seidel residual, the surface frame, and the
  displaced / obliquity sub-modes. The line against `LensNumerics` is the one
  that needs stating, because both change the number that comes back: a
  `LensNumerics` field moves the answer by its own **truncation error** —
  refine it far enough and the answer stops moving — while a `LensPhysics`
  field moves it by a **term**, and no amount of refinement anywhere else
  produces that term.

47 of the family's parameters are fields; the rest stay keyword-only, each for
a written reason (tabled below). The gate that keeps this honest is
`tests/unit/test_audit2609_a16_lens_config_round_trip.py`, which walks the live
signatures and fails if any parameter is neither a field, nor one of the ten
contract names in `lens_config.CONTRACT_PARAMETERS` (`E_in`, `prescription`,
`wavelength`, `dx`, `N`, and the five configuration parameters themselves), nor
a documented exclusion. A parameter can be *excluded*; it cannot be
*forgotten*.

`physics=` is the one configuration parameter that is **not** on every entry
point, because every one of its fields is a parameter of `apply_real_lens` and
of nothing else: the traced / Maslov / GBD / FGA models build their screens
from a ray trace rather than from the thin-element OPD, so none of these terms
has a switch there to map onto. A physics request handed to a sibling through
`config=` still raises, naming `apply_real_lens` as the owner —
`test_the_physics_parameter_is_declared_exactly_where_it_applies` is the gate
that keeps the signatures and the field table in step, in both directions.

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

`sag_dtype` has three spellings — the per-call keyword, the process knob
`set_lens_sag_dtype`, and `LensResources.sag_dtype` — and all three enforce the
same rule: `None` or `np.float64` (the byte-identical default), or
`np.float32` (the opt-in that halves the geometry core). Anything else is
refused with the `CONVENTIONS.md` §2 prefix. The keyword path used to resolve
an unrecognised dtype to float64 in silence, which meant a caller who asked for
`np.float16` got the default and no way to find out; that is the
discarded-setting class the config objects exist to close, so it was closed at
the shared resolver (`_lens_real._resolve_sag_real`) rather than worked around
in the config.

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

### `LensNumerics` (19 fields)

| field | default | analytic | traced | prepare | maslov | gbd | fga | multibranch |
|---|---|---|---|---|---|---|---|---|
| `bandlimit` | True | Y | Y | Y | -- | -- | -- | -- |
| `wave_propagator` | None | Y | Y | Y | -- | -- | -- | -- |
| `ray_subsample` | 8 | -- | Y | Y | -- | -- | -- | -- |
| `output_subsample` | 1 | -- | -- | -- | Y | Y | -- | -- |
| `remap_order` | 3 | Y | -- | -- | -- | -- | -- | -- |
| `displaced_n_side` | None | Y | -- | -- | -- | -- | -- | -- |
| `min_coarse_samples_per_aperture` | 32 | -- | Y | Y | -- | -- | -- | -- |
| `fit_radius_beam_factor` | None | -- | Y | Y | -- | -- | -- | -- |
| `newton_fit` | `'auto'` | -- | Y | Y | -- | -- | -- | -- |
| `newton_poly_order` | 6 | -- | Y | Y | -- | -- | -- | -- |
| `fit_basis` | 'chebyshev' | -- | Y | Y | -- | -- | -- | -- |
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

### `LensPhysics` (9 fields)

Every field is an `apply_real_lens` parameter and nothing else's, so the table
is one column wide in practice; the rest are kept so the shape matches the
three above and so a sibling that grows one of these terms has an obvious row
to fill.

| field | default | analytic | traced | prepare | maslov | gbd | fga | multibranch |
|---|---|---|---|---|---|---|---|---|
| `fresnel` | False | Y | -- | -- | -- | -- | -- | -- |
| `slant_correction` | False | Y | -- | -- | -- | -- | -- | -- |
| `absorption` | False | Y | -- | -- | -- | -- | -- | -- |
| `seidel_correction` | False | Y | -- | -- | -- | -- | -- | -- |
| `seidel_poly_order` | 6 | Y | -- | -- | -- | -- | -- | -- |
| `surface_frame` | False | Y | -- | -- | -- | -- | -- | -- |
| `displaced_mode` | `'screen'` | Y | -- | -- | -- | -- | -- | -- |
| `displaced_obliquity` | `'auto'` | Y | -- | -- | -- | -- | -- | -- |
| `screen_obliquity` | `'auto'` | Y | -- | -- | -- | -- | -- | -- |

`__post_init__` checks only what each field can be judged on alone: the three
enums against `_lens_real`'s own live vocabulary tuples, the five flags as
strict `bool` (so `1` cannot masquerade as `True` here and then be refused by
the call), and `seidel_poly_order` as a positive `int`. The **cross-field**
rules stay where they already are and are not restated:

* `slant_correction` and `seidel_correction` replace the *same* per-surface
  coefficient, and stacking them double-counts the facet obliquity (measured
  173.5 → 1488.6 nm rms exit OPD on an 8 mm cemented doublet) —
  `_check_apply_real_lens_kwarg_combination` adjudicates it;
* every one of these terms is refused under `surface_model='displaced'`, and
  `displaced_mode` / `displaced_obliquity` are only legal *under* it —
  `_check_displaced_support` adjudicates that, against the prescription;
* `screen_obliquity=True` needs `carrier=` — `_check_screen_obliquity_support`.

`surface_model` lives on `LensGeometry` and `caustic` / `fit_basis` on
`LensNumerics`, so a config can state one half of each pair and the other half
is checked one level up. Re-checking any of it inside `LensPhysics` would need
it to see the sibling objects, which a component dataclass deliberately cannot,
and would be a second copy of a rule that already has an owner.

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
| `analytic` | `on_screen_obliquity` | policy knob (`'warn'`/`'error'`/`'silent'`) for one call's diagnostics, not a setting of the optical problem. |
| `analytic` | `stream_transfer_function` | memory/streaming strategy of the analytic in-glass leg only; no sibling, no family-wide meaning. |
| `traced` | `_exit_na_out`, `_remap_launch_out`, `_imap_out` | private diagnostics sinks: underscore-prefixed MUTABLE out-parameters the call writes into. A frozen config must not carry one — sharing a config between two calls would make them share a sink. |
| `traced` | `on_undersample`, `on_noncollimated`, `on_aperture_beam`, `on_fit_domain_basis`, `on_pool_memory` | per-call diagnostic policy knobs. |
| `traced` | `preserve_input_phase`, `remap_sampling`, `tilt_aware_rays`, `decentred_fit_poly_order`, `newton_amp_mask_rel`, `newton_mask_dilate_coarse_px`, `fast_analytic_phase` | tuning constants of this model only; no sibling with the same name and semantics. |
| `traced` | `return_screen` | changes the RETURN TYPE (`field` → `(field, screen)`); a setting that changes what a function returns belongs at the call. |
| `prepare` | `on_undersample`, `on_noncollimated` | per-call diagnostic policy knobs. |
| `maslov` | `ray_field_samples`, `ray_pupil_samples`, `poly_order`, `n_v2`, `extract_linear_phase`, `use_numexpr`, `integration_method`, `stationary_newton_iter`, `stationary_newton_tol`, `local_n_samples`, `local_window_sigma`, `levin_tol`, `collimated_input`, `input_na`, `fold_split` | tuning constants of this model only. |
| `maslov` | `input_wavevector_saddle` | which stationary point the asymptotic evaluators expand about (audit S6). RE-EXAMINED when `LensPhysics` landed and deliberately left keyword-only: **which saddle is the right one is a property of the INPUT FIELD's spectrum, not of the optic**. Every object in this module is designed to be built once and reused across fields; a field-dependent setting inside one would be silently wrong the first time the config outlived the field it was chosen for — the exact failure the config objects exist to prevent. |
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

## `LensPhysics`, and the three settings that did not move

`LensPhysics` is the fourth object. It carries the nine `apply_real_lens`
model-term switches that used to be keyword-only, tabled above. Adding it was
the mechanical change the deferral predicted — a dataclass, a `_PHYSICS_FOR`
table, one entry in `_GROUPS`, one component on `LensConfig`, one `physics=`
parameter — because the resolver, `from_kwargs`, `to_kwargs`, `narrowed_to`
and every structural test already walked `_GROUPS` rather than the three names.

```python
import lumenairy as la

E_out = la.apply_real_lens(
    E_in, prescription=rx, wavelength=633e-9, dx=dx,
    physics=la.LensPhysics(fresnel=True, absorption=True))
```

is the same call, byte for byte, as passing `fresnel=True, absorption=True`.

Three settings that fit the *role* definition were deliberately **left where
they are**:

| setting | lives on | why it did not move |
|---|---|---|
| `surface_model` | `LensGeometry` | It has been a shipped config field since the objects landed. Moving it would break a caller who wrote `LensGeometry(surface_model='displaced')`, to buy nothing a caller can do that they could not do before. |
| `caustic` | `LensNumerics` | Same — shipped, and it is genuinely on the line: the through-focus strategy is both a term selection and a discretisation of the branch sum. |
| `fit_basis` | `LensNumerics` | Same, and this one is on the *numerics* side of the line by its own docstring: Chebyshev and Zernike span the same polynomial space at the same total degree, so it changes the conditioning of the least squares and not the model. |

They are the partition's known ragged edge, recorded here rather than fixed,
because a field that moves between two shipped objects is a migration and this
pass ships none.

`input_wavevector_saddle` (`apply_real_lens_maslov`) was re-examined and stays
keyword-only — see the row in "Deliberately keyword-only" for the reason, which
is that it is a property of the **input field**, not of the optic, and every
object here is built to outlive the field it was configured against.

---

## Module layout

The lens family carries **two** genuine module↔module import cycles that execute
at import time (audit 2026-09-11, TESTS-ARCH "Import cycles" counted four;
re-measured with a module-level-only AST walk):

```
_lens_real   <-> lenses
lenses       <-> lenses_maslov
```

The audit's `_lens_thin <-> lenses` is **gone**: `_lens_thin` needed only
`CUPY_AVAILABLE` and the lazy `cp`, and both now come from the leaf
`lumenairy/backend/_optional.py`. That was the two-line quarter of this item.

`_lens_traced <-> lenses` is **gone** too: that back-edge carried exactly one
name, `_warn_if_aperture_exceeds_grid`, and the grid-versus-aperture
bookkeeping now lives in the leaf `elements/_lens_kernels.py`
(`_collect_semi_diameters`, `check_grid_vs_apertures`,
`recommend_grid_for_prescription`, `_warn_if_aperture_exceeds_grid`), which
imports `numpy` and `warnings` and nothing from `lumenairy`. `lenses.py`
re-exports all four, so every `lenses.check_grid_vs_apertures` and
`from .lenses import _warn_if_aperture_exceeds_grid` resolves unchanged and the
objects are identical (`is`).

`lenses.py` is the family's public re-export hub: it imports all eight sibling
modules at module scope. The remaining back-edges are small and completely
enumerated:

| back-edge | what it imports from `lenses` |
|---|---|
| `_lens_real` | `surface_sag_biconic`, `surface_sag_general` |
| `lenses_maslov` | `NUMEXPR_AVAILABLE`, `_ensure_numexpr_loaded`, `_fit_normaliser`, `_multi_indices_total_degree`, `_warn_if_aperture_exceeds_grid` |

`lens_config.py` is deliberately a **leaf**: it imports nothing from
`lumenairy` at module scope (its validators borrow `_lens_real`'s vocabulary
tuples through an in-function import, cached), so the seven entry points can
depend on it without adding an edge in the other direction. Adding it created
no new cycle — verified by the same walk.

### The plan for the two remaining cycles

Both close the same way the `_lens_traced` one did: move the name into
`elements/_lens_kernels.py` and repoint the back-edge at the leaf, leaving
`lenses.py` re-exporting it so every existing `from lumenairy.elements.lenses
import surface_sag_general` keeps working and the shell-vs-canonical walker
(`tests/unit/test_v5_2_walker_shell_vs_canonical.py`) sees the same surface.

**`_lens_real <-> lenses`** needs `surface_sag_general` and
`surface_sag_biconic` to move, and with them the optional-backend plumbing they
read from module scope: `_get_aspheric_sag_accum_numba`, `_NUMBA_KERNELS`,
`_load_numba` / `_njit` / `_prange` / `_numba`, `_NUMBA_AVAILABLE`,
`_is_cupy_array`, `_ensure_cupy_loaded` and the lazy `cp` alias. Two of those
are live state rather than definitions, which is the whole difficulty:

* `_NUMBA_AVAILABLE` is **monkeypatched to `False` by the test suite** to reach
  the pure-NumPy arm on a box that has numba, and `_load_numba` reads it at
  call time. If it moves, `lenses._NUMBA_AVAILABLE = False` stops being the
  gate and every such test silently exercises the numba arm instead.
* `cp` and `_ne` are populated on first use, so a plain
  `from ._lens_kernels import cp` in `lenses` would bind a **stale `None`**
  rather than a live view.

The mechanical answer to both is a PEP 562 `__getattr__` on `lenses.py` that
forwards unknown attributes to `_lens_kernels` (the shape `_lens_thin` already
uses for its `cp` forward), so `lenses._NUMBA_AVAILABLE = False` reaches the
leaf and `lenses.cp` stays live. That forward is what the move needs proving,
not the sag arithmetic, which is pointwise and relocates unchanged. Gate: the
`-k real_lens` slice plus the WP-A2/A3/A4/A16/B2/B10 files, and the
44-configuration banded byte-identity matrix.

**`lenses <-> lenses_maslov`** is a one-line edit --
`lenses_maslov.py:282`'s `from .lenses import (...)` becomes
`from ._lens_kernels import _warn_if_aperture_exceeds_grid` plus
`from .lenses import (NUMEXPR_AVAILABLE, _ensure_numexpr_loaded,
_fit_normaliser, _multi_indices_total_degree)`, and the edge closes entirely
once those four names follow into the leaf as well. `lenses_maslov.py` is owned
by another work package in this round, so the edit is written down here rather
than made.

`lens_config.py` adds no edge of its own to any of this, and
`tests/unit/test_audit2609_a16_verify_config_and_arch.py::test_lens_config_stays_a_leaf`
keeps it that way.

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
* `tests/unit/test_audit2609_a16_verify_config_and_arch.py` — the independent
  re-verification: a second fixture (singlet, 1064 nm, anamorphic `dy`,
  complex64, decentred+tilted), identical *refusal* through all three
  spellings, the `locals()` contract, the value-equality and `sag_dtype`
  contracts, and the falsifiability of the structural gates.
* `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/VERIFY_WP-A16.md`
  — what was re-measured and what is still open.
