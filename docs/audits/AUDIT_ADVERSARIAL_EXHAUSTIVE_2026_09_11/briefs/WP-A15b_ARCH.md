# WP-A15b — Architecture: optional-dep helper, `override()` knob context managers, lazy loading, layering, re-exports

Second half of the former WP-A15; runs concurrently with WP-A15a (tests/CI/packaging) on DISJOINT files. The
lens-family modules (`_lens_real.py`, `_lens_traced.py`, `_lens_imap.py`, `lenses.py`, `lenses_maslov.py`,
`lenses_gbd.py`, `gbd.py`, `fga.py`, `_lens_traced_multibranch.py`, `_lens_jax.py`) are WP-A16's — you create the
shared helpers and switch every NON-lens site; A16 switches the lens sites afterwards using your helpers, so their
contracts below are binding.

Read first: `COMMON.md`, then `TESTS-ARCH.md` §V6 and report §14 V6 / §15.6 in
`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11.md`; then every `fixes/WP-*_REPORT.md` and `fixes/VERIFY_WP-*.md`
"Requested changes outside my ownership" section — the re-export requests are yours (list at §5 below).

## Files you own
New `lumenairy/backend/_optional.py`, new `lumenairy/_knobs.py`; the non-lens optional-dep helper sites
(`lumenairy/propagators/fft_infra.py`, `lumenairy/sources/core.py`, `lumenairy/optimize/_merit_jit.py` — mechanical
dedupe only); the process-global `set_*` knob modules (`fft_infra.py`, `memory.py`, `cache.py`, `rcwa/_core.py`,
`io/storage.py`, `user_library.py` — additive API only); `lumenairy/__init__.py`, `lumenairy/elements/__init__.py`,
`lumenairy/raytrace/__init__.py`, `lumenairy/elements/pmm/__init__.py`, `lumenairy/analysis/__init__.py`,
`lumenairy/optimize/__init__.py` (exports / lazy loading); `lumenairy/raytrace/surface.py` (the import-time layering
violation only); `lumenairy/propagators/system.py` (one optional passthrough, §6); `tests/conftest.py` (autouse
snapshot/restore fixture); the walker tests `tests/unit/test_v4_16_0_walker_all_symmetry.py`,
`tests/unit/test_v4_14_1_dispatcher_pin_cache_clears.py`, `tests/unit/test_niche_audit_w3_infra.py`; new tests
`tests/unit/test_audit2609_a15b_*.py`; your report/changelog `fixes/WP-A15b_REPORT.md` / `fixes/WP-A15b_CHANGELOG.md`.
NOT: `pyproject.toml`, `.github/`, `tests/unit/test_public_api.py` (A15a — run it, don't edit it).

## Deliverables
1. **`lumenairy/backend/_optional.py`** exporting `ensure_cupy()`, `is_cupy_array(x)`, `load_numba()` (and whatever
   the five-fold copies actually share — read all eight copies first: `_lens_real.py`, `_lens_traced.py`, `lenses.py`,
   `fft_infra.py`, `sources/core.py`, `_lens_imap.py`, `_merit_jit.py`, `fga.py`; the helper must reproduce every
   copy's behaviour exactly, including the CONVENTIONS §10 absence semantics and any memoisation). Switch the THREE
   non-lens sites to it; leave the five lens sites for A16 but state in your report which lines they must replace.
2. **`lumenairy/_knobs.py`** — the generic knob registry: `register_knob(name, *, getter, setter, doc)`,
   `override(**knobs)` (a `contextlib.contextmanager` that sets each named knob and restores the previous values in
   reverse order even on exceptions; unknown names raise `ValueError("override: unknown knob 'x'; known: [...]")`),
   `snapshot() -> dict`, `restore(snapshot)`, `knobs() -> tuple[str, ...]`. Register every process-global `set_*` knob
   in the modules you own (one line beside each setter; registration at import time of the module that owns the knob —
   so a knob is only visible once its module is imported; document that). Export `override` from `lumenairy/__init__`.
   The autouse fixture in `tests/conftest.py` snapshots all registered knobs before each test and restores after (it
   must be cheap: dict copy; measure the per-test overhead and report it). A16 registers the lens knobs the same way.
3. **Lazy loading (PEP 562)** in `lumenairy/elements/__init__.py` for `rcwa`, `pmm`, `bor`, `berreman`, `eme` (and
   whatever else pulls `scipy.linalg`/`scipy.special` at import — measure with `python -X importtime -c "import
   lumenairy" 2> imp.log` on a quiet moment, before and after, and report the top-20 table). `from lumenairy.elements
   import rcwa`, `lumenairy.elements.rcwa.X`, `import lumenairy.elements.rcwa as r`, pickling of objects from those
   modules, `dir(lumenairy.elements)` and the repo's `__all__` walkers must all keep working; `__getattr__` must raise
   `AttributeError` (not ImportError) for unknown names. If `lumenairy/__init__.py` eagerly imports any of these,
   make that lazy too — but every name in `lumenairy.__all__` must still resolve (run `test_public_api.py`).
4. **Layering:** fix the import-time violation `raytrace/surface.py → elements.lenses` (move the import into the
   function that needs it or invert the dependency — say which and why; the `-k real_lens` slice and the raytrace test
   files gate it).
5. **Re-exports and walker reconciliation** (all requested by other WPs — verify each name exists first):
   top level (`lumenairy/__init__.py` + `__all__`): `from_prescription` (or decide to keep `Operator.from_prescription`
   as the only public spelling and document the walker exemption), `unwrap_phase_2d`, `clear_meshgrid_cache`,
   `meshgrid_cache_bytes`, `zernike_basis_cache_bytes`, `MinEdgeThicknessMerit`, `edge_thickness`,
   `rs_alias_free_distance`, `exit_vertex_transfer`, `pmm_2d_order_drift`, `override`; `lumenairy/raytrace/__init__.py`:
   `seed_entrance_eikonal` (sibling `exit_vertex_transfer` is already exported — check); `lumenairy/elements/pmm/__init__.py`:
   `pmm_2d_order_drift` (VERIFY-A12 already added it to `__all__` — confirm). Then make
   `test_v4_16_0_walker_all_symmetry.py` and `test_v4_14_1_dispatcher_pin_cache_clears.py` pass with the new `__all__`
   names (re-export or a documented exemption per name — no blanket exemptions), and re-derive the fence in
   `test_niche_audit_w3_infra.py` against `memory.py`'s `_ASM_FIRST_CALL_FIXED_BYTES` (the cold ASM peak fell to
   61.7 MB after WP-A5/A11 — re-measure, re-derive the constant in `memory.py` if that is where the truth lives, and pin
   a two-sided derived bar per TESTING_STANDARDS, not a 1.35× fudge).
6. **`propagators/system.py`** ≈ line 948: pass `subharmonics=elem.get('subharmonics', 0)` to
   `generate_turbulence_screen` (default identical — prove bit-identity on the existing system tests).
7. The `_lens_*`/`lenses` module-level 2-cycles are A16's; do not touch them.

## Verification specifics
- Touched-module test files + `tests/unit/test_public_api.py` + the walker files + `python validation/run_all.py`;
  import-time table before/after; bit-identity for every mechanical refactor (the existing tests of the touched
  modules are the gate; add a direct before/after digest test where a module has thin coverage).
- Threading: the knob registry is read from tests only, but `override()` must be safe to nest and to use under
  `ThreadPoolExecutor` (document that it is process-global, like the knobs themselves).

## Addendum (2026-09-12)
- `tests/conftest.py` ~lines 245-255: the comment cites `waveoptics_dock.py` clearing `USE_PYFFTW` unconditionally as the reason `fft_infra` is leak-guarded; WP-A9 fixed the dock (overrides are restored on every exit path). Keep the guard, update the sentence (WP-A9_REPORT.md §5.3).
