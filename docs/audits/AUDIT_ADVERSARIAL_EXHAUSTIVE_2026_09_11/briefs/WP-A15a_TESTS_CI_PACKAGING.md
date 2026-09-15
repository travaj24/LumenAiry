# WP-A15a — Tests, CI, packaging, suite composition and hygiene (cross-cutting; runs AFTER the physics WPs)

This is the first half of the former WP-A15. The second half (architecture: optional-dep helper, `override()`
context managers, lazy loading, layering, re-exports) is WP-A15b and runs concurrently on DISJOINT files — do not
touch its files (listed at the end). The lens-family library files belong to WP-A16 — never edit them.

Read first: `COMMON.md`, then the partition report `TESTS-ARCH.md` (all of it) and report sections §14 (V3–V5, V7)
and §15.4–15.7 in `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11.md`. Also read every
`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-*_REPORT.md` and `VERIFY_WP-*.md` section titled
"Requested changes outside my ownership" / "Open items" — the packaging/CI/test-composition requests are yours (e.g.
WP-A14's `threadpoolctl` dependency line); CONVENTIONS/README sentences are WP-A18's; re-exports are WP-A15b's.

## Files you own
`.github/workflows/*.yml`, `pyproject.toml`, `requirements*.txt`, `.gitignore`, `MANIFEST.in`,
`tests/unit/test_public_api.py`, `tests/unit/test_audit_except_budget.py`, the test files you must edit for V5 (slow
markers, timing-assertion conversions, prose-test retirement — list every one), `scripts/` (probe scratch relocation
into `validation/probe_scripts_legacy/` — plain file moves, the orchestrator stages them), new tests
`tests/unit/test_audit2609_a15a_*.py`, your report/changelog `fixes/WP-A15a_REPORT.md` / `fixes/WP-A15a_CHANGELOG.md`.
NOT: `tests/conftest.py`, any `__init__.py`, any library module (`lumenairy/**`) — those are A15b/A16.

## Deliverables
1. **V3 — combination coverage for the lens family (tests only).** A pairwise covering-array parametrisation over the
   ~12 physics-affecting kwargs of `apply_real_lens` and `apply_real_lens_traced` (read their CURRENT signatures — WP-A2
   and WP-A3 just changed them; not the memory/perf knobs), on a curved-rear, non-collimated fixture, asserting energy
   conservation + finiteness + agreement with the kwarg-free call where the knob is at its default; and a property test
   that every kwarg passed at its default reproduces the no-kwarg call bit for bit ("knob silently discarded" detector).
   Keep the runtime bounded (document the budget; use small N). WP-A16 will reuse this file for its config-object
   bit-identity test — name it `tests/unit/test_audit2609_a15a_lens_covering_array.py` and keep the fixture factory
   importable.
2. **V4 — CI and packaging gates.** ruff `continue-on-error: false` (run `ruff check lumenairy` first and fix or
   explicitly ignore what it reports, per rule, with a reason — but do NOT edit library modules: if a finding needs a
   code change, add a scoped per-file ignore with a `# TODO(audit-2609)` reason and list it in the report); scope
   `lumenairy/ui/` back into ruff with a per-directory ignore list; update the stale mypy comment and grow the
   `[tool.mypy] files` whitelist by every module that passes `mypy --strict` today with zero errors (smallest first;
   list them); add CPython 3.14 to the `unit-tests.yml` matrix and restore the classifier, replacing the false
   accelerator-wheel comment; reconcile the `jax` floor with what is used locally (state the decision); add
   `threadpoolctl>=3.1` to the core dependencies (WP-A14's request — check `pip show threadpoolctl` for the installed
   version); add the test `importlib.metadata.version("lumenairy") == lumenairy.__version__` (skip-free — it must pass in
   an editable install, so re-run `pip install -e .` if the editable finder is stale and say so); collapse
   `test_public_api.py`'s 712 parametrisations into a handful of assertions (same coverage, one collected id per
   property); `.gitignore`: `.benchmarks/`, `.mypy_cache/`, `.ruff_cache/`, and replace the blanket
   `*.png`/`*.dat`/`*.log` ignores with directory-scoped ones (list the tracked benchmark JSONs for the orchestrator to
   untrack — no git writes yourself); correct `MANIFEST.in`'s "31 files" comment and exclude `validation/probe_*` from
   the sdist; move the 12 probe scratch files out of `scripts/` into `validation/probe_scripts_legacy/`.
3. **V5 — suite composition.** Re-mark the slow lane at the > 2 min/file bar (from `.test_durations`; add
   `pytestmark = pytest.mark.slow` to those files and list them); add a `.test_durations` staleness check as a TEST
   (`tests/unit/test_audit2609_a15a_durations_staleness.py`: fails when > 2 % of collected ids are missing; collect with
   `--collect-only -q` in a subprocess with `OPENBLAS_NUM_THREADS=1`); convert the 24 wall-clock/speedup assertions to
   operation-count / complexity-order assertions (TESTING_STANDARDS S1) — list each with before/after; retire the tests
   that assert on README/ROADMAP prose (keep the CHANGELOG fabrication walkers — they are release gates); correct the
   "fast (<30 s)" contract text in `pyproject.toml`.
4. **Except budget.** `tests/unit/test_audit_except_budget.py` pins the non-UI `except Exception:` count at 48; the
   measured count was 51 at the audit base and is higher now after the fix WPs (A2 +2, A3/A4 `_lens_imap` +1 /
   `_lens_jax` +2, A1 `exit_vertex` +1, A5 `hfpi` −1, VERIFY-A8 narrowed glass.py). Re-measure, list every site with a
   one-line justification (or name it as a narrowing request for its owner — A15b for non-lens modules, A16 for lens
   modules), and set the budget to the justified count with the list IN the test so it fails on the next unexplained
   addition.
5. **Test isolation.** `test_propagation_asm_cache_lock_still_paired` passes standalone but failed in a shared session
   (VERIFY-A10's note) — find the pollution source and fix it in the TEST (fixture/reset), not the library; the
   subprocess tests that flake with `WinError 6/50` under machine load — make them robust (retry-on-OS-error helper or
   `subprocess.run(..., close_fds=...)` per the actual cause; measure, don't guess).
6. **V7 — hygiene** items not covered above (the future-dated audit doc name is deliberate — leave it; `--maxfail`;
   `benchmarks/` gating note).

## Verification specifics
- Run every touched test file plus `python validation/run_all.py`; `ruff check lumenairy` must be clean under the
  new configuration; `mypy` on the new whitelist must be clean; the `-k real_lens` slice (135 tests) after V3.
- No behaviour change anywhere in the library (you don't edit it). Report the collected-id count before/after V5.

## Files you must NOT touch (A15b / A16 own them)
`lumenairy/**` (all library modules incl. every `__init__.py`), `tests/conftest.py`,
`tests/unit/test_v4_16_0_walker_all_symmetry.py`, `tests/unit/test_v4_14_1_dispatcher_pin_cache_clears.py`,
`tests/unit/test_niche_audit_w3_infra.py`.

## Addendum (2026-09-12)
- `.gitignore`: the pre-existing untracked scratch directory `validation/repro_traced_carrier_122/` (August `.npz` assets, ~90 MB) must be covered by a directory-scoped ignore (`validation/repro_*/`) and excluded from the sdist; do not delete it.
- Also from WP-A14: `threadpoolctl` is still absent from `pyproject.toml` and `requirements.txt` (H5) — it is yours.
