# `validation/` -- physics-validation harness

This directory holds the **physics-validation suite**: end-to-end
tests that exercise full library workflows and compare the output
against either closed-form analytical references or independent
reference implementations (Optiland, OPDpy, Mahan, refractiveindex.info
cross-checks).  These are NOT unit tests -- they are slower, run
under `validation/run_all.py`, and are wired into CI separately via
`.github/workflows/validate.yml`.

The v5.2 ROADMAP closure note: this README closes the long-standing
"contributors don't know whether to add to `tests/unit/` or
`validation/`" gap.  Use the decision tree below.

---

## Decision: `tests/unit/` vs `validation/`

Use **`tests/unit/`** when:

* The test exercises a single public function or small interaction
  (one or two API calls).
* The expected output is a deterministic value, a shape, an error
  class, or an assertion about API contract.
* The runtime is `< 1 second`.
* The test is a regression pin for a specific commit / audit fix.
* The test is a structural walker (V1-V15) discovering all entry
  points.

Use **`validation/`** when:

* The test runs an end-to-end multi-stage workflow (e.g. source ->
  propagate -> lens -> propagate -> analyzer).
* The expected output is compared against an analytical reference
  (paraxial diffraction-limited spot size, Fraunhofer pattern
  closed-form, MTF / Strehl analytical formula) OR against an
  external reference (Optiland prescription, OPDpy ray trace,
  Mahan textbook example).
* The runtime is `~seconds` to `~minutes` (full propagation +
  analyzer chain).
* The test produces plots or large-data artifacts a developer
  reviews visually after a change.
* The test is the canonical "does the library still match physics
  at this magnification / NA / coherence regime" check.

When in doubt, `tests/unit/` is the right default.  Move to
`validation/` only when one of the bullets above is unambiguously
true.

---

## Layout

* `validation/_harness.py` -- the shared test harness: tolerance
  budgets, plot helpers, optional-dep skips, output-directory
  conventions.  Read this BEFORE writing a new validation file --
  most boilerplate is already provided.
* `validation/conftest.py` -- pytest configuration: fixture
  injection, marker registration, custom slow-test handling.
* `validation/run_all.py` -- script entry point: enumerates every
  `validation/*/t_*.py` file and runs them as subprocesses.  This
  is what CI `.github/workflows/validate.yml` invokes.
* Per-topic subdirectories (`propagators/`, `raytrace/`,
  `elements/`, `analysis/`, `sources/`, `optimize/`, `io/`,
  `backend/`, `gui/`, `through_focus_smoke/`, `real_lens_opd/`,
  `integration/`):  each contains `t_<descriptive_name>.py`
  validation files.  The `t_` prefix distinguishes these from
  unit tests and matches the legacy naming captured in
  `pyproject.toml`'s `python_files = ["test_*.py", "t_*.py"]`.

## Running

```bash
# Run the full suite (every file).
python validation/run_all.py

# Run a single file directly with pytest.
python -m pytest validation/propagators/t_asm_fraunhofer_crosscheck.py -v

# Run via the CI workflow locally.
gh workflow run validate.yml --ref main
```

## Adding a new validation file

1. Pick the right subdirectory (`propagators/`, `raytrace/`, etc.).
   If your test crosses topics, default to `integration/`.
2. Name the file `t_<descriptive_name>.py` -- the leading `t_`
   matters; the `run_all.py` enumerator filters on it.
3. Start from an existing file in the same subdirectory; copy its
   harness boilerplate verbatim and adapt.
4. Make the test self-contained: no shared global state between
   `t_*.py` files; assume the file runs in a fresh interpreter
   under `run_all.py`.
5. Set realistic tolerance budgets.  The library is `~1e-12`
   accurate at the propagation step, but typical end-to-end
   tolerances (Strehl, RMS WFE, OPD) accommodate sampling +
   floating-point accumulation -- `1e-3` to `1e-5` is the usual
   range.  Cite the budget in a docstring comment.
6. If your validation produces plots, route the output through
   the harness's plot directory convention (see `_harness.py`).
   PNGs are gitignored; CI uploads them as artifacts.

## Adding a new validation subdirectory

Only do this when an existing subdirectory genuinely doesn't fit.
Run `python validation/run_all.py` after creating it to verify
the enumerator picks it up.  Add a one-line description here
once the directory is committed.

---

## Why this isn't combined with `tests/unit/`

Two reasons:

* **Runtime budget**:  unit tests gate every PR via the
  `unit-tests.yml` workflow (3-5 minute budget).  Validation
  tests are slower (~3-5 minutes per file, ~30+ files) and run
  under `validate.yml` on a less-aggressive cadence.
* **Failure mode**:  unit tests catch "the function changed
  shape"; validation tests catch "the physics is wrong"
  (analytical or cross-implementation disagreement).  These
  failure modes need different debugging surfaces -- the
  validation output usually includes plots a developer
  reviews after a regression, while unit-test failures are
  pure pass/fail in CI logs.

See `tests/unit/README.md` for the unit-test side of this
documentation contract (if absent, add it as a v5.2.1 closure).
