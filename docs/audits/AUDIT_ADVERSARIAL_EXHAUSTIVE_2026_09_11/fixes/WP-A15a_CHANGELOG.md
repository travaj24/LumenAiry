# WP-A15a changelog text -- tests, CI, packaging, suite composition (audit 2026-09-11, findings V3, V4, V5, V7, H5)

No library module was modified by this work package; there is no behaviour change in
`lumenairy/`.  Everything below is the gate, the package metadata, or the suite.

### Added -- tests: combination coverage for the `apply_real_lens` family (V3)

The audit measured 673 `apply_real_lens*` call sites in the corpus exercising 177 distinct
kwarg combinations, **68 % of them passing zero or one optional kwarg**, against a family
whose traced entry point has 48 parameters.  Four of the five defects the orchestrator
seeded were *interaction* defects -- a flag combined with a geometry -- and the suite,
organised one file per past bug and testing one knob at a time against a default
background, had by construction almost no power against them.

`tests/unit/test_audit2609_a15a_lens_covering_array.py` adds a pairwise covering array over
the physics-affecting kwargs of both entry points -- **16 kwargs in 12 rows** for
`apply_real_lens`, **20 kwargs in 15 rows** for `apply_real_lens_traced` -- on the geometry
the suite never tested: an AC254-ish cemented doublet with a **curved rear**
(R3 = -291.07 mm; the only existing `seidel_correction` fixture is plano-rear, on which the
whole exit-vertex class is invisible) illuminated by a **diverging** spherical wave.  Each
arm asserts only what holds for every legal combination -- finiteness, output shape, and
that a passive lens cannot GAIN power -- because an oracle that held for every combination
would be a second implementation of the lens.

Measured over all 27 arms: the largest `P_out/P_in` is 0.996170598 (analytic) and
0.996162541 (traced), the smallest 0.438620794 and 0.354847608, and the largest excess over
1 is **-3.83e-03** -- no arm gains.  The bar is 1e-6 relative: seven decades above the
float64 accumulation floor for a 64x64 reduction and four below the smallest gain a real
defect produces (a double-applied Fresnel transmittance, a dropped obliquity cosine, an
un-normalised ray-density Jacobian).

A second property test passes **17 analytic and 28 traced kwargs each at its own signature
default** and requires the result to be bit-identical to omitting it -- the "knob silently
discarded" detector, which catches a kwarg whose unset sentinel and documented default
disagree and which nothing else in the suite can see.

Building the array surfaced the family's constraint lattice, measured by sweeping all 141
analytic and 308 traced factor-level pairs: 11 and 3 of them are refused outright
(`slant_correction` x `seidel_correction`; `surface_model='displaced'` against fresnel /
absorption / slant / seidel / surface_frame / carrier / non-ASM propagator;
`surface_model='tangent_facet_remap'` against slant / surface_frame / screen_obliquity /
non-ASM propagator; `amplitude_model='ray_density'` x `newton_amp_mask_rel`).  Each
exclusion is declared with the library's own refusal text and **re-measured on every run**,
so the table cannot become a hiding place.

45 collected ids, 3.34 s.  The fixture factory and the factor tables are module-level and
importable: WP-A16's config-object work reuses them.

### Fixed -- CI: `ruff` was advisory and blind to `lumenairy/ui/` (V4/P1-3)

The lint job carried `continue-on-error: true`, so the only always-on static gate on the
codebase reported green on a lint failure -- nothing in the merge-blocking path failed on an
unused import, a shadowed symbol or an **undefined name**.  Separately, `extend-exclude`
removed all of `lumenairy/ui/` (~30 modules including the 3 626-line `main_window.py`) from
linting entirely.

The job is now `continue-on-error: false`, its scope widens from `tests/unit/` to `tests/`,
and `lumenairy/ui/` is scoped back IN behind a per-directory ignore list rather than
excluded wholesale.  Findings on arrival: 13 in `lumenairy/` (9 x I001, 2 x F541, 2 x
F401), **331 in `lumenairy/ui/`** in exactly seven rules (F401 121, I001 116, E701 37, E702
37, F811 10, F841 7, F541 3), 37 in `tests/`.  After: `ruff check lumenairy/ tests/` ->
**All checks passed**.  What is now enforced in the previously-unlinted GUI: F821 undefined
name -- the exact class that shipped `carrier_field.py:466` to main this month -- plus F632,
E711/E712/E713/E714, E722 and E731.

Every ignore carries its reason.  `I001` is granted only to the files that use this
library's documented interleaved-import convention (the same convention `E402` is already
ignored project-wide for; `ruff --fix` on `raytrace/__init__.py` was measured to relocate a
commented import block and delete its commentary).  Four ignores are marked
`TODO(audit-2609)` because the finding needs a one-line code change in a module this work
package does not own; they are listed in the report as requests and should be deleted, not
kept.

### Fixed -- CI: `mypy --strict` covered 6 of 227 modules behind a comment 43 releases stale (V4/P1-4)

The `[tool.mypy]` block carried "No CI wiring at v5.0.1; `unit-tests.yml` keeps mypy unwired
until v5.1's cleanup lands".  `unit-tests.yml` has had a merge-blocking `mypy` job since
v5.2, and the repo is at 5.45.1 -- the comment described a state 43 minor versions old while
the gate it described was live and nearly empty.

The comment is corrected, and the whitelist is grown by measurement rather than by waiting
on a cleanup: one `mypy --strict --follow-imports=silent --ignore-missing-imports
--python-version 3.12` pass over all 170 non-`ui` candidate modules, attributing each error
to the file it was reported in.  20 modules come back clean; the 11 that are not
`__init__.py` are added, smallest first -- `elements/coronagraph.py` (54 lines),
`raytrace/core.py` (68), `analysis/core.py` (69), `io/prescriptions.py` (106),
`raytrace/layout.py` (190), `raytrace/bundles.py` (227), `algebra/from_prescription.py`
(230), `propagators/result.py` (244), `_cache_registry.py` (292), `analysis/strehl.py`
(557), `propagators/asymptotic_modes.py` (893).  The gate goes from 6 declared paths /
**11 source files** to 17 / **22**, and `mypy` reports `Success: no issues found in 22
source files`.  The nine clean `__init__.py` hubs are deliberately held back until the
lazy-loading rewrite lands, because a PEP 562 `__getattr__` needs its own annotations first.

### Fixed -- packaging: CPython 3.14 is the development interpreter and was in no CI leg (V4/P1-5)

The classifier block dropped 3.14 because "3.14 is too new for the optional accelerator
dependencies (numba, jax, pyfftw) to have published wheels against".  MEASURED on the dev
box: numba 0.65.1, jax 0.10.1 and pyfftw 0.15.1 are all installed on CPython 3.14.6, the
repo tracks `.benchmarks/Windows-CPython-3.14-64bit/*.json`, and `tests/unit/__pycache__`
holds `cpython-314` artefacts -- i.e. every line of this library was developed and
hand-tested on an interpreter no CI leg exercised.

3.14 joins the `unit-tests.yml` matrix (3.10-3.14) and the `publish.yml` release verify -- a
classifier is a promise and the release gate is what backs it -- the
`Programming Language :: Python :: 3.14` classifier is restored, and the false comment is
replaced by the measurement.  A new test asserts the two agree in both directions, so a
classifier no leg runs fails the gate.

### Fixed -- packaging: installed metadata was two releases behind the source (V4/P1-7)

`lumenairy.__version__` read 5.45.1 while `importlib.metadata.version('lumenairy')` read
**5.43.0**, with the editable finder in the `-X importtime` trace generated at 5.21.2 -- 24
releases stale.  Anything reading installed metadata (a user's `pip show`, a packaging
check, a plugin version gate) saw the wrong number and no test compared them.  After
`pip install -e .` both read 5.45.1, and
`tests/unit/test_public_api.py::test_installed_metadata_version_matches_source_version`
now pins the equality.  It is deliberately **skip-free**: every environment that runs this
suite installs the package, so a missing distribution is a broken environment and the
failure message names the command that fixes it.

### Changed -- tests: `test_public_api.py` 726 collected ids -> 7, same coverage (V4/P1-9)

A 104-line file parametrised over `lumenairy.__all__` and generated **726 collected ids**,
4.9 % of the whole suite, for what is one property -- inflating the headline test count
that is used as a proxy for physics coverage.  The parametrisation is now a loop that
reports every failing name at once (strictly more useful than 726 ids that each reported
one), so the file collects **7 ids, one per property, in 0.12 s** (was 0.65 s) while still
checking all 708 names on every run.  Two properties are added (`__all__` entries must be
identifier strings; installed metadata must match `__version__`) and the existing phantom
counter-pin is strengthened to assert the *aggregate* loop observes the injection -- the
half the parametrisation could not express.

### Added -- dependencies: `threadpoolctl>=3.1` is now a hard dependency (H5)

`set_blas_threads`, `rcwa_blas_threads` and the `@_with_blas_limit` wrapper on every public
RCWA entry point are **inert** without `threadpoolctl` -- they warn and pass through.  On an
oversubscribed many-core box that is catastrophic rather than a micro-optimisation:
MEASURED on a 24-thread Windows OpenBLAS 0.3.31 build, `inv()` of a 163x163 complex matrix
takes 2.29 s unpinned against 0.0057 s at one thread (**400x**), and a 1-D TM RCWA solve at
`n_orders=81` takes 18.2 s instead of 0.13 s (**140x**).  It is tiny and pure Python, so the
library's own remedy now works out of the box.  Declared in `pyproject.toml` and mirrored in
`requirements.txt`, with a test that both carry it.  Floor 3.1: `ThreadpoolController` (the
cached-enumeration path `rcwa/_core.py` prefers) landed in 3.0 and 3.1 is the first release
with the Windows OpenBLAS 0.3.x detection.

Two further `requirements.txt` drifts corrected in the same pass: `cupy>=11.0` -> `>=14.0`
(the floor `pyproject.toml` has carried since `cupy.linalg.eig` became a requirement) and
`jax>=0.4.20` -> `jax>=0.11.0 ; python_version >= "3.12"` (floor and PEP 508 marker both).

### Changed -- packaging: `.gitignore` stops blanket-ignoring binary fixtures (V4/P1-10)

Repo-wide `*.png` / `*.jpg` / `*.pdf` / `*.fits` / `*.dat` / `*.log` rules made every binary
of those types un-addable without `git add -f` -- the audit found 33 `.png` under
`validation/` in exactly that state.  They are replaced by directory-scoped rules derived
from `git ls-files --others --ignored` (113 files under `validation/repro_traced_carrier_121`,
26 under `docs/audits`, 22 under `validation/real_lens_opd`, 9 under `examples/output`, plus
`importtime.log`), so a fixture in `tests/` or `benchmarks/` is now addable normally while
the generated trees stay ignored.  `.mypy_cache/`, `.ruff_cache/` and `.benchmarks/` are
added -- all three appear in a working tree as soon as the tool runs -- and
`validation/repro_*/` covers the 89 MB untracked `repro_traced_carrier_122` scratch tree.
The untracked-file count is unchanged (27 -> 26), i.e. the rewrite adds no noise.

`.test_durations` remains **tracked**, deliberately, with a comment saying so: pytest-split
reads it to balance the CI shards.

### Changed -- packaging: `MANIFEST.in` no longer ships 808 probe-scratch files (V7)

The validation-suite comment claimed "31 files with ~370 physics-fidelity tests", a 4.4.0
count that is 28x off: MEASURED, the tree holds **876 `.py` across 70 sub-directories**, of
which 624 are under `validation/probe_*` and 184 under `validation/repro_*` -- per-round
audit scratch promoted to permanent tracked artefacts, several hard-coding absolute paths on
one dev box.  The comment now carries the measurement and both prefixes are pruned from the
sdist.  The 37 `t_*.py` / `test_*.py` entry points are the part of the old claim that
survives.

### Changed -- repo: `scripts/` holds only the four maintained tools (V7/P3-4)

Eleven one-off audit reproductions (`_d5_byteid`, `_d5_probe`, `_d5_probe2..6`, `_d5_spl`,
`_g8_c15lad`, `_g8_failbefore`, `_g8_probe`) sat in `scripts/` beside
`check_dep_metadata.py`, `check_source_line_citations.py`, `verify_changelog_closures.py`
and `stamp_changelog.py`, where an unqualified `.py` in a tools directory reads as a
maintained tool.  They move to `validation/probe_scripts_legacy/` -- recoverable, pruned
from the sdist, and documented -- with their repo-root arithmetic repointed one level up so
they still run.  A test keeps `scripts/` clean.

### Changed -- tests: the slow lane is re-marked at the > 2 min/file bar (V5/P2-1)

The suite's documented contract said `tests/unit/` is "fast (<30 s) API-contract tests".
MEASURED from the committed `.test_durations` against today's collection: **13 061 s =
3.63 h across 14 199 ids** -- wrong by a factor of ~440.  The `pyproject.toml` text is
corrected to the measurement, and the marker block now records what is actually applied
(`slow` in 54 files before this change, 65 after; `unit`, `regression` and `bench` zero
times each, kept as declared-but-unused so `--strict-markers` turns a future
`@pytest.mark.unit` into a collection error rather than a silent no-op).

Eleven files at or over the 2 min/file bar gain a module-level `pytest.mark.slow`, moving
**2 341.0 s** out of the fast gate: the fast lane goes 9 487.1 s -> 7 144.0 s (**31.6 ->
23.8 min per shard** on 5 shards, against a 45-min step cap) and the slow lane 3 573.9 s ->
5 915.0 s, which is 32.9 min per shard on 3 -- **over** the 30-min step cap -- and 19.7 min
on 5.  So the slow gate goes **3 -> 5 shards**; no timeout is raised.  Nine further files at
or over the bar are left marked as-is and listed in the report: five belong to other work
packages, and four are jax-guarded, where marking them slow would silently delete them from
the only CI leg that runs jax.

`test_fga.py` carried a comment explicitly declining the marker.  Both of its premises had
moved, and rather than leave a comment contradicting the code beneath it, the comment is
rewritten with the re-measurement: its claim that the file "measures 1791.7 s (~30 min)" is
**stale by 10x**, a pre-v5.31 unpinned-BLAS artefact -- the same 27 ids total **178.8 s** on
the regenerated file.

The release gate is widened to match.  `publish.yml` verified the **fast lane only**
(`-m "not integration and not slow"`), so every `slow`-marked test was outside tag
verification entirely -- a hole that predates this change and that the re-marking widens.  A
`verify-slow` job (5 shards, 3.12, single-threaded BLAS) now gates `build` and `publish`.

### Added -- tests: a `.test_durations` staleness gate (V5/P2-3)

`.test_durations` is load-bearing CI infrastructure, not a convenience file: pytest-split
weights every id it cannot find there at a default, so the shard split degrades toward a
count split as the gap grows.  The audit measured the difference -- a greedy 5-way split on
fresh durations is exact (max/min = **1.000**) where an alphabetical split is **3.85x**
imbalanced, and an over-weighted shard is the failure behind two 30-minute publish-verify
timeouts.  Nothing was watching it.

`tests/unit/test_audit2609_a15a_durations_staleness.py` collects the suite in a subprocess
and fails when more than **2 %** of collected ids carry no timing (derivation in the module
docstring: at the measured ~1.0 s mean, 2 % of ~14 000 ids is ~280 s of unmodelled work,
about 5 % of one fast shard, while 20 % is a whole shard and a guaranteed cap hit).  It also
bounds the reverse direction -- timings for ids that no longer collect -- and carries a
counter-pin that the two id sets really overlap, so a path-separator or rootdir change
reports as a gate failure rather than as data.

**This gate is RED on the audit-fixes branch, at 15.04 % (2 117 of 14 076), and that is the
gate working.**  The staleness is the fix campaign's own: the untimed ids are the tests the
fix and verify work packages added over the last two weeks.  Regenerating mid-campaign would
be stale again within the hour, so the regeneration belongs at campaign exit; the failure
message carries the exact procedure.

### Changed -- tests: eight wall-clock / speedup assertions converted to operation counts (V5, TESTING_STANDARDS S1)

Each of these read a `time.perf_counter()` delta and asserted on it -- the shape S1 forbids,
because the pass/fail boundary sits inside the cross-build spread of the quantity being
read.  Each is replaced by the operation count it was standing in for, which no machine, no
BLAS build and no CPU contention can move:

* **Gerchberg-Saxton** `elapsed_ms < 5000.0` (on a call the comment itself measured at
  600-900 ms) -> the routine performs **exactly `n_iter + 1` forward and `n_iter` inverse
  transforms**, measured 11/10, 26/25, 51/50 at `n_iter` = 10/25/50.
* **Pure-spherical doublet trace** `fast < 20 ms` and `speedup >= 1.2` (measured 1.78 /
  1.13 / 1.59 / 2.65 across four consecutive runs -- one outright failure) -> a 1k-ray trace
  through a pure-spherical doublet evaluates the surface sag **0 times** (the closed-form
  branch), a conic doublet **3 times** (one per surface), the Newton fallback 10 per
  surface.
* **`trace_jax` cache** `warm < 20 ms` and `first/warm >= 20x` (the docstring tabulates the
  ratio sliding 391-652x idle to 102-155x loaded, and the file logs it twice as a timing
  flake) -> `_TRACE_JAX_CACHE` holds exactly **one entry after each of 20 identical calls**
  and serves the **same object** every time.
* **through-focus JAX cache** `speedup >= 10.0` -> the same one-entry / same-object count.
* **numba multi-field kernel** `t_jit < 8 s` (against a measured 0.58-1.36 s) -> the kernel
  pair in `_NUMBA_KERNELS['multi_field']` is built once and keeps its identity across eight
  field angles.
* **RCWA analytic-shape overlap** `dt < 4.0` (against a measured 51.8-60.1 ms; the
  docstring's own stated intent is "to catch an accidental O(K^2)-in-the-predicate
  regression, not to race") -> a 1024-shape lattice reaches the exact overlap predicate
  **0 times** out of 523 776 pairs, while a genuinely overlapping pair reaches it once.

One further wall-clock bar -- the Shack-Hartmann `elapsed_ms < 5000.0` -- is **retired rather
than converted**: the per-lenslet gather it guarded is still a nested Python loop, so there
is no operation-count oracle that would not simply restate the loop.  It is replaced by the
contract the call owes its caller (all five returned arrays are `(n_lenslets, n_lenslets)`;
a uniform illumination yields measured, non-NaN slopes).  Three more sites live in files
owned by other work packages and are listed in the report with their suggested conversions.

### Removed -- tests: five assertions on README / ROADMAP / CHANGELOG prose (V5)

A regex over a 180 KB README is not a test of the library: it fails when someone rewords a
sentence and passes when the code underneath the sentence is wrong.  Five such tests are
deleted from `tests/unit/test_v4_16_2_agent_d.py` (13 -> 8 collected ids), with a block in
the file recording what went and why.  The drift they were written for -- `refractiveindex`
moving to the `[glass]` extra -- is pinned structurally by the `requirements.txt` tests kept
beside them and by `scripts/check_dep_metadata.py`, which compares `requirements.txt`
against `pyproject.toml` itself.

The **CHANGELOG fabrication walkers** are explicitly NOT part of this retirement: they check
a changelog claim against `git diff`, i.e. a fact rather than a wording, and they are
release gates.

### Changed -- tests: the broad-`except` budget is a per-file census, not a scalar (audit V4/L5 rule)

`tests/unit/test_audit_except_budget.py` pinned a single number, 48, which was already 3
short of the tree at the audit base and 7 short mid-campaign -- and a scalar cannot say
WHICH file grew, so one module could add a clause while another removed one and the gate
stayed green on the way past.

All **53** current non-`ui` `except Exception:` sites were read and justified individually,
and the scalar is replaced by a per-file census across 27 modules, grouped by the
justification that licenses each: 34 JAX tracer/concretization guards (the raised type is
untypeable without importing jax at module scope, and every fallback routes a traced input
to the exact path), 4 other untypeable optional-package boundaries (numba's compilation
errors; two `refractiveindex` lookups, one of which re-raises as `ValueError`), 9
user-code and best-effort probes, 2 teardown paths that must not raise, and **3 recorded as
narrowing requests** -- bare imports that want `except ImportError`.  Both the per-file and
the total bars are `<=`, so narrowing a clause never fails the gate while adding one does,
and a module absent from the census has an implicit allowance of zero.  A fourth test
bounds the census's own slack at 3 clauses (measured: **0**) so it cannot be set uniformly
generous.

### Fixed -- tests: `WinError 6 / 50` in subprocess tests was a stdio-handle defect, not machine load (V5)

Three work packages logged `OSError: [WinError 6] The handle is invalid` /
`[WinError 50] The request is not supported` from `subprocess.Popen._get_handles` as an
intermittent environment flake that "passes in isolation".  MEASURED under this suite's own
configuration: with `--capture=fd` (the default) a spawn that names only `stdout=PIPE`
raises WinError 6 before the child exists, while the same spawn with all three streams named
succeeds, and under `pytest -s` every shape succeeds.  The mechanism is CPython's
`Popen._get_handles` on Windows resolving **any stream left as `None`** through
`GetStdHandle()` and then `DuplicateHandle`: pytest's fd-capture reassigns the process's
file descriptors without calling `SetStdHandle`, so those Win32 handles can be stale.  The
all-`None` case is safe because CPython short-circuits it -- which is why only *partial*
spawns ever failed, and why it looked environmental.

Fixed deterministically by naming all three streams (`stdin=subprocess.DEVNULL` beside
`capture_output=True`; none of these children reads stdin) at **25 sites in 15 test files**
plus `scripts/verify_changelog_closures.py::_run_git` and `scripts/stamp_changelog.py::_run`,
which is where the surviving reproduction actually lived.  The reproducing batch went from
**1 failed / 31 passed to 32 passed, three runs in a row**.  A static AST gate now fails on
any new spawn that leaves some but not all streams inherited, with an exemption list that
may shrink but not grow.

### Fixed -- tests: the ASM cache-lock pin was order-dependent (V5)

`test_propagation_asm_cache_lock_still_paired` asserted
`'_ASM_CACHE_LOCK' in inspect.getsource(_clear_local_asm_caches)`, and VERIFY-A10 recorded
it passing standalone and failing inside a 577-test session.  `inspect.getsource` resolves
through `linecache` against the file **on disk** at the function's `co_firstlineno`, which
was fixed when the module was imported -- so any edit to `fft_infra.py` after import shifts
the definition and the lookup returns a neighbouring function's text.  A long session gives
that window; a 0.2 s standalone run does not.

Rewritten as a behavioural pin: recording locks are substituted for `_ASM_CACHE_LOCK` and
`_PYFFTW_PLAN_LOCK`, the clearer is called, and both must have been entered.  Strictly
stronger than what it replaces -- the old assertion passed on a body that merely mentioned
the name in a comment, and would have passed on an `acquire()` with no release.

### Changed -- CI: `--maxfail` 50 -> 10 (V7)

At 50 per shard across 5 shards x 5 interpreters the fast gate tolerated up to 1 250
failures before any job aborted, which is not an early-abort budget.  10 still shows the
pattern -- one shard's first ten failures name the class -- while a catastrophic break stops
the matrix in minutes instead of burning 45 of them per job.  Applied to `unit-tests.yml`
and to `publish.yml`'s release verify.

---

### Migration notes

* **`threadpoolctl` is a new hard dependency.**  `pip install lumenairy` now pulls it.  It is
  pure Python and ~30 KB; environments that already have scikit-learn have it as a transitive
  dependency.  Nothing breaks without it -- the library degrades exactly as before -- but the
  BLAS caps it advertises only take effect with it installed.

* **`lumenairy` now declares support for CPython 3.14** and the CI matrix runs 3.10-3.14.

* **`validation/probe_*` and `validation/repro_*` no longer ship in the sdist.**  A PyPI
  tarball still carries the 37 validation entry points, `tests/`, `examples/`, the workflows
  and the reference docs; it no longer carries ~808 per-audit-round probe scripts.

* **`scripts/_d5_*.py` and `scripts/_g8_*.py` moved** to
  `validation/probe_scripts_legacy/`.  Nothing in the repository referenced them.

* **The `slow` marker now means "at or over 2 min of file-total runtime"** and selects a
  5-shard CI lane on CPython 3.12.  Eleven files joined it; if you run
  `pytest tests/unit` locally without `-m`, nothing changes.
