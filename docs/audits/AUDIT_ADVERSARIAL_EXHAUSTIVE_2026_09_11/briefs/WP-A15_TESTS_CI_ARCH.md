# WP-A15 — Tests, CI, packaging and architecture (cross-cutting; runs AFTER the physics work packages)

Read first: `COMMON.md`, then the partition report `TESTS-ARCH.md` (all of it) and report sections §14 (V3–V7; V1/V2 were
done by WP-A2) and §15.4–15.7 in `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11.md`. Also read every
`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-*_REPORT.md` section titled "Requested changes outside my
ownership" — several of them are addressed to you (e.g. WP-A14's `threadpoolctl` dependency line, CONVENTIONS sentences
are NOT yours — those go to WP-A18).

## Files you own (this WP runs alone on these, after the physics WPs finished)
`.github/workflows/*.yml`, `pyproject.toml`, `requirements*.txt`, `.gitignore`, `MANIFEST.in`, `tests/conftest.py`,
`tests/unit/test_public_api.py`, the test files you must edit for V5 (slow markers, timing-assertion conversions,
prose-test retirement), `lumenairy/backend/_optional.py` (new) and the five-fold helper sites it replaces
(`_lens_real.py`, `_lens_traced.py`, `lenses.py`, `fft_infra.py`, `sources/core.py`, `_lens_imap.py`, `_merit_jit.py`,
`fga.py` — mechanical dedupe only), the process-global `set_*` knob modules for the context-manager work
(`fft_infra.py`, `_lens_real.py`, `_lens_traced.py`, `memory.py`, `cache.py`, `rcwa/_core.py`, `io/storage.py`,
`user_library.py` — additive API only), `lumenairy/elements/__init__.py` and `lumenairy/__init__.py` (lazy loading),
`lumenairy/raytrace/surface.py` (the one import-time layering violation), `scripts/` (probe scratch relocation),
`lumenairy/_math/` if needed. New tests `tests/unit/test_audit2609_a15_*.py`.

## Deliverables
1. **V3 — combination coverage for the lens family.** A pairwise covering-array parametrisation over the ~12
   physics-affecting kwargs of `apply_real_lens` and `apply_real_lens_traced` (not the memory/perf knobs), on a
   curved-rear, non-collimated fixture, asserting energy conservation + finiteness + agreement with the kwarg-free call
   where the knob is at its default; and a property test that every kwarg at its default reproduces the no-kwarg call bit
   for bit ("knob silently discarded" detector). Keep the runtime bounded (document the budget; use small N).
2. **V4 — CI and packaging gates.** ruff `continue-on-error: false` (run `ruff check lumenairy` first and fix or
   explicitly ignore what it reports, per rule, with a reason); scope `lumenairy/ui/` back into ruff with a per-directory
   ignore list; update the stale mypy comment and grow the `[tool.mypy] files` whitelist by every module that passes
   `mypy --strict` today with zero errors (smallest first; list them); add CPython 3.14 to the `unit-tests.yml` matrix
   and restore the classifier, replacing the false accelerator-wheel comment; reconcile the `jax` floor with what is
   used locally (state the decision); add `threadpoolctl` to the core dependencies (WP-A14's request); add the test
   `importlib.metadata.version("lumenairy") == lumenairy.__version__` (skip-free — it must pass in an editable install,
   so re-run `pip install -e .` if the editable finder is stale and say so); collapse `test_public_api.py`'s 712
   parametrisations into a handful of assertions; `.gitignore`: `.benchmarks/`, `.mypy_cache/`, `.ruff_cache/`, and
   replace the blanket `*.png`/`*.dat`/`*.log` ignores with directory-scoped ones (list the two tracked benchmark JSONs
   for the orchestrator to untrack — no git writes yourself); correct `MANIFEST.in`'s "31 files" comment and exclude
   `validation/probe_*` from the sdist; move the 12 probe scratch files out of `scripts/` into
   `validation/probe_scripts_legacy/` (plain file moves; the orchestrator stages them).
3. **V5 — suite composition.** Re-mark the slow lane at the > 2 min/file bar (from `.test_durations`; add
   `pytestmark = pytest.mark.slow` to those files and list them); add a `.test_durations` staleness check (a test or CI
   step that fails when > 2 % of collected ids are missing); convert the 24 wall-clock/speedup assertions to
   operation-count / complexity-order assertions (TESTING_STANDARDS S1) — list each; retire the tests that assert on
   README/ROADMAP prose (keep the CHANGELOG fabrication walkers — they are release gates); correct the "fast (<30 s)"
   contract text in `pyproject.toml`.
4. **V6 — architecture.** (a) `@contextmanager override(...)` beside every process-global `set_*` knob (one generic helper
   + one line per knob) and an autouse conftest fixture that snapshots and restores all of them; (b) PEP 562 lazy
   `__getattr__` in `lumenairy/elements/__init__.py` for `rcwa`, `pmm`, `bor`, `berreman`, `eme` (and whatever else pulls
   `scipy.linalg` at import) — measure `python -X importtime -c "import lumenairy"` before/after on a quiet moment and
   verify every public name still resolves (the `test_public_api` loop); (c) `lumenairy/backend/_optional.py` exporting
   `ensure_cupy()`, `is_cupy_array()`, `load_numba()` and the five-fold sites switched to it; (d) fix the import-time
   layering violation `raytrace/surface.py → elements.lenses`; (e) the `_lens_*`/`lenses` module-level 2-cycles — extract
   the shared leaf (`elements/_lens_kernels.py`) ONLY if you can prove bit-identity on the lens test files; otherwise
   document the plan.
5. **V7 — hygiene** items not covered above (the future-dated audit doc name, `--maxfail`, `benchmarks/` gating note).

## Verification specifics
- Run the touched test files plus `python validation/run_all.py`; run `ruff check lumenairy` and `mypy` on the new
  whitelist; import-time measurement; the full `-k real_lens` slice (135 tests) after (c)/(d).
- No behaviour change anywhere: every mechanical refactor is gated by the existing tests of the touched modules.
