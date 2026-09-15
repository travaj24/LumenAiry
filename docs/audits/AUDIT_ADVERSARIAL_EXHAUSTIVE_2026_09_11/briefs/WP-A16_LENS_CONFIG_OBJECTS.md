# WP-A16 — New feature: configuration objects for the `apply_real_lens` family (runs AFTER WP-A2/A3/A4)

Read first: `COMMON.md`, then `TESTS-ARCH.md` §"Proposed config-object design for the lens family" and report §14
(V3, V6) in `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11.md`, plus the current signatures of `apply_real_lens`,
`apply_real_lens_traced`, `apply_real_lens_maslov`, `apply_real_lens_gbd`, `apply_real_lens_fga`,
`apply_real_lens_traced_multibranch`, `prepare_real_lens`, `PreparedTracedLens` (28–48 parameters each). Also read
`fixes/WP-A2_REPORT.md`, `fixes/WP-A3_REPORT.md`, `fixes/WP-A4_REPORT.md` and their `VERIFY_WP-*.md` — they changed
these signatures days ago and their "Requested changes outside my ownership" sections may name you.

## Files you own
New `lumenairy/elements/lens_config.py`; the entry-point signatures in `_lens_real.py`, `_lens_traced.py`,
`lenses_maslov.py`, `lenses_gbd.py`, `fga.py`, `_lens_traced_multibranch.py` (additive parameters only); for the
architecture items below also `_lens_imap.py`, `_lens_jax.py`, `gbd.py`, `lenses.py` (code); new tests
`tests/unit/test_audit2609_a16_*.py`; a docs page `docs/lens_configuration.md`; your report/changelog
`fixes/WP-A16_REPORT.md` / `fixes/WP-A16_CHANGELOG.md`. `lumenairy/__init__.py` / `elements/__init__.py` ONLY under the
rule in §11. `tests/unit/test_audit2609_a15a_lens_covering_array.py` may be IMPORTED (fixture factory) but not edited.

## Deliverable
Three frozen dataclasses — `LensGeometry` (~8 fields: `dy`, `output_plane_distance`, `output_plane_n`, `conjugate`,
`surface_model`, `clip_aperture`, …), `LensNumerics` (~12: `ray_subsample`, `output_subsample`, `remap_order`,
`newton_fit`, `newton_poly_order`, `newton_max_iters`, …), `LensResources` (~10: `sag_chunk_rows`, `accumulator_store`,
`scratch_dir`, `use_gpu`, `parallel_amp`, `*_min_free_gb`, …) — derive the exact field partition from the real
signatures and document it. Each entry point keeps `(E_in, prescription, wavelength, dx)` positionally and gains
`geometry=`, `numerics=`, `resources=`; the existing kwargs remain fully supported (NO deprecation in this pass);
precedence rule: an explicit kwarg and a config field for the same setting must agree or raise (§2 prefix); validation
moves into `__post_init__` (reuse the existing validators — do not duplicate their logic). Add `LensConfig.from_kwargs()`
/ `to_kwargs()` round-trips.

## Verification
- Bit-identity: for every entry point, the config-object call equals the kwarg call on the covering-array fixtures
  (reuse WP-A15a's `tests/unit/test_audit2609_a15a_lens_covering_array.py` fixture factory if it exists; otherwise
  build a small one and say so).
- A `dataclasses.fields()` round-trip test that detects a kwarg silently ignored by any entry point.
- Docstrings for every field with units and defaults; a worked example in the docs page.

## Addendum (orchestrator, 2026-09-12) — architecture items on the lens-family files, moved here from WP-A15
8. Switch the five lens-family optional-dependency helper copies (`_lens_real.py`, `_lens_traced.py`, `lenses.py`,
   `_lens_imap.py`, `fga.py`) to `lumenairy/backend/_optional.py` (created by WP-A15b: `ensure_cupy()`,
   `is_cupy_array()`, `load_numba()`). If that module does not exist when you start, STOP and tell the orchestrator
   (A15b may still be running) — do not create a second copy. Mechanical dedupe only; the lens test files gate it.
9. Register every process-global `set_*` knob in `_lens_real.py` / `_lens_traced.py` with `lumenairy/_knobs.py`
   (`register_knob(name, getter=..., setter=..., doc=...)` one line beside each setter) so `lumenairy.override(...)`
   and the autouse test fixture cover them; add a test that `override(<lens knob>=...)` restores on exit.
10. The `_lens_*`/`lenses` module-level import 2-cycles: extract the shared leaf (`elements/_lens_kernels.py`) ONLY if
    you can prove bit-identity on the lens test files (`-k real_lens` slice, 135 tests, plus the WP-A2/A3/A4 test
    files); otherwise document the plan in `docs/lens_configuration.md` §"Module layout".
11. `lumenairy/__init__.py` / `elements/__init__.py`: add the `LensGeometry`/`LensNumerics`/`LensResources`/`LensConfig`
    exports ONLY if the orchestrator's launch message says WP-A15b has finished; otherwise list them under "Requested
    changes outside my ownership".
12. Except-budget: WP-A15a will list the `except Exception:` sites in the lens modules that need narrowing; narrow
    the ones it names (its report `fixes/WP-A15a_REPORT.md` §4) if it exists when you start, else skip and say so.

## Addendum 2 (orchestrator, after WP-A15b landed)
13. WP-A15b has FINISHED: `lumenairy/backend/_optional.py` and `lumenairy/_knobs.py` exist with the contracts in `fixes/WP-A15b_REPORT.md` §2.1/§2.2; its §5.4 table gives the exact lines to replace at the five lens-family optional-dep sites (keep the module-level `cp` alias, keep `_NUMBA_AVAILABLE` a module attribute that tests monkeypatch, keep fga's deliberate raise) and §5.5 names the three lens knobs to register (`lens_sag_dtype`, `lens_parallel_amp`, `pointwise_cos_grid_cache_budget`; getters must be cheap and side-effect-free). The §11 rule is satisfied: you MAY add your exports to `lumenairy/__init__.py` / `elements/__init__.py` (read A15b's lazy tables first and follow their pattern).
14. Import time (A15b §5.1/§5.2, granted): make `lumenairy/backend/__init__.py` forward `scipy` lazily (PEP 562, the exact snippet is in A15b's report; the walker exemption `('lumenairy.backend', 'scipy')` already exists) and move `_lens_traced_multibranch.py:57`'s module-level `from scipy.special import airy` into the function that uses it (or a cached accessor). Measure `import lumenairy` before/after with A15b's interleaved same-build method (expected ~712 -> ~170 ms) and run `tests/unit/test_public_api.py` + the A15b test files + `-k real_lens`.
