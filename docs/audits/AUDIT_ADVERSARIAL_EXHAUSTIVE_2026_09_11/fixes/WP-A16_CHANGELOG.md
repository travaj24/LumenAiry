# WP-A16 -- changelog text

*(Orchestrator: assemble into `CHANGELOG.md` under the release entry.  Sections
are in the repository's CHANGELOG voice; finding IDs are the 2026-09-11
adversarial audit's.)*

---

### Added -- lens configuration objects: `LensGeometry` / `LensNumerics` / `LensResources` / `LensConfig`

The seven `apply_real_lens`-family entry points carry ~110 distinct keyword
arguments between them (31 / 51 / 28 / 32 / 29 / 24 / 14 keyword-only
parameters as measured 2026-09-12), with about twenty names shared between
siblings.  The 2026-09-11 audit (TESTS-ARCH section 14, item 13) measured 673
call sites in the corpus exercising only 177 distinct combinations, 68 % of
them passing zero or one optional keyword, and traced four of the five seeded
defects to *interaction* cases the suite had no power against -- because a
48-parameter signature is not a thing you can parametrise over.

New module `lumenairy/elements/lens_config.py` (leaf; imports nothing from
`lumenairy` at module scope) adds three frozen dataclasses, partitioned by
role and covering 38 of the family's parameters:

* **`LensGeometry`** (10 fields) -- what optical problem is being solved:
  `dy`, `output_plane_distance`, `output_plane_n`, `conjugate`,
  `surface_model`, `clip_aperture`, `carrier`, `origin`, `beam_centre`, `roi`.
* **`LensNumerics`** (17) -- how it is discretised: `bandlimit`,
  `wave_propagator`, `ray_subsample`, `output_subsample`, `remap_order`,
  `min_coarse_samples_per_aperture`, `fit_radius_beam_factor`, `newton_fit`,
  `newton_poly_order`, `newton_max_iters`, `inversion_method`,
  `amplitude_model`, `caustic`, `caustic_band`, `caustic_ray_subsample`,
  `caustic_min_area_ratio`, `inverse_map`.
* **`LensResources`** (11) -- what machine the call may use and what it
  reports: `use_gpu`, `amp_use_gpu`, `n_workers`, `parallel_amp`,
  `parallel_amp_min_free_gb`, `sag_dtype`, `sag_chunk_rows`,
  `accumulator_store`, `scratch_dir`, `progress`, `verbose`.
* **`LensConfig`** holds the triple and adds `from_kwargs()` / `to_kwargs()`
  (both accepting `entry_point=` so they speak that entry point's own
  spelling), `narrowed_to(entry_point)` and `requests()`.

All four are exported from `lumenairy` and `lumenairy.elements`.

`apply_real_lens`, `apply_real_lens_traced`, `prepare_real_lens_traced`,
`apply_real_lens_maslov`, `apply_real_lens_gbd`, `apply_real_lens_fga` and
`apply_real_lens_traced_multibranch` each gained `geometry=` / `numerics=` /
`resources=` / `config=`, all defaulting to `None`.

**This is purely additive and there is no deprecation.**  Every existing
keyword still works with the same default; a call that passes none of the four
runs exactly the code it ran before (four `is not None` tests and nothing
else).  Verified bit-identical, `np.array_equal`, across all seven entry points
on the WP-A15a covering-array fixture --- the configured call, the
three-component call and the keyword call return the same bytes, and an
all-default `LensConfig()` returns the same bytes as passing no config.

Precedence: a config field that differs from its default and a keyword that
differs from its signature default must AGREE or the call raises with the
`CONVENTIONS.md` section 2 prefix; a field that is set but that the entry point
has no parameter for also raises (naming the entry points that do take it, and
pointing at `narrowed_to`) rather than being silently discarded --- which is the
failure class the audit found.  Field-local validation moved into
`__post_init__`, so a bad setting is refused where it is built rather than
several hundred lines inside a 5 700-line call.

New: `docs/lens_configuration.md` (partition tables, precedence rule, worked
cross-engine example, the full list of deliberately keyword-only parameters
with reasons, and the module-layout plan).
New tests: `tests/unit/test_audit2609_a16_lens_config_round_trip.py` (84 ids),
`tests/unit/test_audit2609_a16_lens_config_bit_identity.py` (27).

### Changed -- lens family: one shared optional-dependency probe

`elements/_lens_real.py`, `elements/_lens_traced.py`, `elements/lenses.py`,
`elements/_lens_imap.py` and `propagators/fga.py` had seven hand-copied
implementations of `_ensure_cupy_loaded` / `_is_cupy_array` / `_load_numba`
between them (audit TESTS-ARCH P2-9); they now delegate to
`lumenairy/backend/_optional.py`.  No behaviour change: each module keeps its
own `cp` alias (the GPU branches read the module-level name), each keeps
`_NUMBA_AVAILABLE` as a module attribute that tests monkeypatch to reach the
pure-NumPy arm, and `apply_real_lens_fga` keeps its deliberate `ImportError`
with the `pip install lumenairy[numba]` hint --- it is the one consumer with no
NumPy fallback.

### Changed -- the three lens knobs are restorable

`lens_sag_dtype`, `lens_parallel_amp` and `pointwise_cos_grid_cache_budget` are
registered with `lumenairy._knobs`, so `lumenairy.override(...)`,
`snapshot()`/`restore()` and the suite's autouse fixture now reach them (audit
TESTS-ARCH P2-5).  That takes the registry to the audit's 20 and completes
`set_low_memory`'s coverage: `lens_parallel_amp` was the last of its four knobs
that a test could leak.  `pointwise_cos_grid_cache_budget` is registered
against a private byte/byte accessor pair, because the public setter takes
MEGABYTES while the public getter returns BYTES --- registering the public pair
would have multiplied the budget by 2**20 on every restore.

### Performance -- `import lumenairy` no longer loads `scipy.linalg`

`lumenairy/backend/__init__.py` forwards `scipy` through a PEP 562
`__getattr__` instead of `from . import scipy as scipy`, and
`elements/_lens_traced_multibranch.py`'s module-level
`from scipy.special import airy` became a cached first-use accessor.
`lumenairy.backend` has 41 module-level importers that want only
`array_namespace` / `is_*_array`, so every user was paying the rigorous-solver
scipy bill at import.

MEASURED on this box (Windows, CPython 3.14, `OPENBLAS_NUM_THREADS=1`, fresh
interpreters, medians of 9 interleaved same-build A/B pairs -- full table in
the WP-A16 report):
`import lumenairy` **745.0 ms -> 673.8 ms** (-71.2 ms / -9.6 %) on a quiet
box, and the interleaved same-build delta (which is immune to cross-run noise)
is **+0.1 ms before -> +70.8 ms after**; under load both numbers shrink to
~45 ms.  `lumenairy.backend`'s cumulative import cost goes from 567.3 ms
(WP-A15b's measurement) to **2.4 ms**, and `scipy.linalg` is no longer in
`sys.modules` after `import lumenairy` at all.
`la.backend.scipy.X`, `from lumenairy.backend import scipy` and
`import lumenairy.backend.scipy` are unchanged.

Smaller than the ~540 ms the earlier estimate suggested, for a measured reason
recorded in the report: `propagators/fft_infra.py` imports `scipy.fft` at
module scope, and `scipy.fft._fftlog_backend` imports `scipy.special` --- so
`scipy.special` and the heavy shared prefix (`scipy._lib._array_api` ->
`array_api_compat.numpy` -> `numpy.f2py` -> `charset_normalizer` ->
`numpy.testing`) are still charged to `import lumenairy` by a module outside
this work package (`scipy.fft` measures 426.5 ms cumulative).  What this change
removes is the marginal `scipy.linalg` cost.

### Fixed -- two `f`-string prefixes without placeholders (ruff F541)

`elements/_lens_real.py`'s unfolded-equivalent mirror warning carried two
`f"..."` continuation lines with no placeholders.  Dropped the prefixes; the
warning text is unchanged (re-measured: `ruff --isolated --select F541` reports
2 on the previous blob and 0 now).  The
`"lumenairy/elements/_lens_real.py" = ["F541"]` entry in
`[tool.ruff.lint.per-file-ignores]` is now unnecessary and should be deleted.

### Fixed -- one broad `except` narrowed

`elements/_lens_imap.py`'s `build_inverse_map` RAM-budget diagnostic caught
`Exception` around a bare `from .. import memory`; narrowed to `ImportError`
(audit TESTS-ARCH except budget, WP-A15a section 5.7).  `_lens_imap.py` now
carries zero broad except clauses.

New tests for the four items above:
`tests/unit/test_audit2609_a16_lens_arch.py` (27 ids).

### Migration note

None required.  Every existing call site is unchanged and bit-identical.  The
one thing worth knowing if you adopt the new objects: `LensConfig` refuses a
setting the entry point has no parameter for, rather than ignoring it, so a
config shared across engines needs `config.narrowed_to('apply_real_lens_…')`
at the call sites that do not take every field.  `docs/lens_configuration.md`
has the worked example.
