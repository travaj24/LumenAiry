# TESTS-ARCH audit — tests, CI, packaging, architecture

Auditor partition: cross-cutting (test suite, validation, benchmarks, CI, packaging,
whole-library architecture). Repo `D:/Metacept/.../Lumenairy`, branch `main`, v5.45.1,
commit `e2ed280d release: 5.45.1`. Read-only; no repo file created, modified or deleted.

---

## Scope / what was measured

| Measurement | Method | Result |
|---|---|---|
| Full test collection | `pytest --collect-only -q` (whole suite) | **13 319 tests** collected (13 358 − 39 deselected) in **137 s**, 1 skip (PySide6 absent), 0 collection errors |
| Test corpus size | AST walk of `tests/` | 536 `.py` files (532 `tests/unit`, 2 `tests/integration`, `conftest.py`, `__init__.py`), **260 341 LOC**, 7 664 `def test*` functions |
| Library size | AST + `tokenize` over `lumenairy/` | 227 modules, **221 391 lines** = 106 827 code + 39 740 comment + 59 868 docstring; 4 110 functions |
| Suite runtime | parse `.test_durations` (12 941 entries) | **13 136 s = 3.65 h**; 26 tests > 60 s carry 25 % of it |
| Import graph | AST, module-level vs. in-function edges | 533 import-time edges, 415 lazy; 9 SCCs; 12 layering violations |
| Duplication | normalised-AST hash + `difflib` over bodies ≥ 8 lines | 9 exact clusters (~573 redundant lines); 5× `_ensure_cupy_loaded`, 5× `_is_cupy_array`, 5× `_load_numba` |
| `apply_real_lens` family | AST signature extraction + corpus grep per kwarg | 7 entry points, 28–48 params each; **673 call sites** in tests exercising **177 distinct kwarg combinations** |
| Adversarial classification | `random.seed(0)` sample of 70 collected test functions, read and hand-classified | see proportions below |
| Comment/history load | `tokenize` + block-extension regex for version/audit markers | `_lens_traced.py` **37.6 %** history, `carrier.py` **36.9 %** |
| CI | read all four `.github/workflows/*.yml` | 4 workflows, 5+3 shards, ruff non-blocking |
| Packaging | `pyproject.toml`, `MANIFEST.in`, `.gitignore`, `git ls-files` | 3 970 tracked files, 2 785 (70 %) under `validation/`; `.git` 550 MB |
| Representative run (part C) | `-k real_lens`, 3 smallest files | 2 of 3 complete, both **pass** |

**Correction to the brief:** `.github/workflows/unit-tests.yml` has **no uncommitted
modifications**. `git status --short` returns exactly one line — `?? validation/repro_traced_carrier_122/`
— and `git diff .github/workflows/unit-tests.yml` is empty. The working tree is otherwise clean.

---

## Findings

### P1 — tests that cannot catch real regressions on load-bearing physics; CI holes

---

**[P1-1] The only `seidel_correction=True` test in the entire suite has a mathematically
unfalsifiable assertion**
`tests/unit/test_audit_glass.py:92-152` (assertion at `:147`)

Evidence. `seidel_correction=` appears as a kwarg in exactly **one** `.py` file in `tests/`
(and nowhere in `validation/`, `benchmarks/` or `examples/` — 1 438 files scanned). That one
test ends:

```python
phase_diff = phase_diff - np.median(phase_diff[ap])
phase_diff = np.angle(np.exp(1j * phase_diff))          # <- wraps into (-pi, pi]
rms_waves = float(np.sqrt(np.mean(phase_diff[ap] ** 2))) / (2 * np.pi)
assert rms_waves < 50.0
```

`np.angle(np.exp(1j*x))` maps **any** real to `(-π, π]`. Therefore
`rms ≤ π` and `rms_waves = rms/2π ≤ 0.5`, unconditionally. The bar is 50.0 — a **100×
margin over the mathematical maximum**. I fed the expression a synthetic 10⁴-wave error and
recovered `rms_waves = 0.288`; the assertion passed. The test cannot fail for any input,
including a totally broken lens.

The test's own docstring states the intent — "Pre-v4.11.2 the disagreement was ~10⁴ waves;
post-fix it should be << 100. 50 lambda comfortably distinguishes a sign error" — which is
precisely the discrimination the wrap destroys. The 50-wave bar was chosen for an *unwrapped*
residual and then a wrap was inserted "to avoid 2-pi wraparound dominating the RMS" (`:137-141`),
silently converting a real gate into a no-op.

Three further independent weaknesses in the same test: (a) the prescription is
`R1=51.5e-3, R2=float('inf')` — a plano-convex singlet whose **last surface is flat**, so the
curved-last-surface case is untested; (b) the oracle is `apply_real_lens_traced`, i.e. another
implementation inside the same library, not independent truth; (c) it asserts on exit-pupil
**phase**, never on **focus position**.

Impact. `seidel_correction` is an opt-in physics correction with zero effective test coverage.
A sign flip, a scale error or a complete no-op would all pass.

Fix. (i) Delete the `np.angle(np.exp(1j·))` wrap and unwrap properly (`np.unwrap` on a radial
cut, or fit-and-subtract tilt+piston), then drop the bar to a value the fixed pipeline
actually achieves — measure it first and pin ~2× that. (ii) Add a curved-last-surface case
(`R2` finite). (iii) Add a focus-position test: sweep `output_plane_distance`, take the
intensity centroid/peak, and assert the shift against a Seidel-predicted longitudinal
aberration — an oracle independent of the traced path.

---

**[P1-2] `surface_frame=True` has no test that asserts beam deviation; its
"backward-compat" tests are tautologies**
`tests/unit/test_v5_2_off_axis_conic_surface_frame.py` (418 lines, the only file passing
`surface_frame=`)

Evidence. Three test functions pass `surface_frame=True` (`:116`, `:202`, `:397`). All three
mention tilt; **none** asserts on a centroid, deviation, chief-ray angle, direction or shift.
What they assert:

- `:230` `assert not np.allclose(opd_ff, opd_sf, atol=1e-15)` — only that the two branches
  *differ*. Any non-zero perturbation, correct or not, passes.
- `:415` `assert np.all(np.isfinite(E_out[slc, slc]))` — a finiteness smoke test.
- `:327` and `:372` `assert np.array_equal(E_no_kwarg, E_default_kwarg)` — asserts that
  passing `surface_frame=False` equals omitting it. Since `False` **is** the declared default,
  this tests CPython's default-argument mechanism, not the library. It is a tautology that
  cannot fail while the signature parses.
- The only quantitative check is `:255`/`:262`, a slope band at `rel=0.20` and
  `abs(slope_sf) < 0.2 * abs(slope_ff_expected)`.

Impact. The kwarg whose entire purpose is to change where a tilted surface sends the beam has
no test on where the beam goes.

Fix. Add a test that traces a collimated beam through a surface tilted by a known α, computes
the exit intensity centroid at a fixed downstream plane under both frames, and asserts the
`surface_frame=True` deviation against Snell at the tilted facet (closed-form for a single
tilted plane interface). Delete the two `array_equal(no_kwarg, default_kwarg)` tests — they
consume CI time and assert nothing.

---

**[P1-3] Ruff lint is non-blocking in CI, and 4 of the largest trees are excluded from it
anyway**
`.github/workflows/unit-tests.yml:214-236`, `pyproject.toml:296-312`

Evidence. The `Ruff lint` job carries `continue-on-error: true` (`:222`). A lint failure is
reported green. Separately, `[tool.ruff] extend-exclude` removes `validation/`, `docs/`,
`examples/` and **`lumenairy/ui/`** from linting — `lumenairy/ui/` alone is ~30 modules
including `main_window.py` (3 626 lines), `waveoptics_dock.py` (2 942), `model.py` (2 930).
The lint rule set is `["E","F","I"]` with `E501`, `E741`, `E402` all ignored.

Impact. The only always-on static gate on the codebase is advisory. Nothing in the
merge-blocking path fails on an unused import, an undefined name, or a shadowed symbol in
non-UI code, and nothing checks UI code at all.

Fix. Flip `continue-on-error` to `false` (the tree is presumably already clean — verify with
one run, then flip). Separately scope `lumenairy/ui/` in with a per-directory ignore list
rather than excluding it wholesale.

---

**[P1-4] `mypy --strict` covers 6 of 227 modules (2.6 %) and the comment claiming it is
unwired is three years of releases stale**
`pyproject.toml:344-383`, `.github/workflows/unit-tests.yml:238-273`

Evidence. `[tool.mypy] files` lists `lumenairy/backend`, `_deprecation.py`, `_context.py`,
`_validation.py`, `progress.py`, `memory.py`. `follow_imports = "silent"` and
`ignore_missing_imports = true` prevent it seeing anything else. The inline comment still
says *"No CI wiring at v5.0.1; `unit-tests.yml` keeps mypy unwired until v5.1's cleanup
lands"* — but the workflow **does** run mypy (job at `:238`, `continue-on-error: false`), and
the repo is at 5.45.1. The comment describes a state 45 minor versions old.

Impact. Type checking is real but nearly empty; the config comment actively misleads a reader
into thinking it is not running at all. None of `_lens_real.py`, `_lens_traced.py` or
`carrier.py` — the three load-bearing files — is checked.

Fix. Update the stale comment. Then grow the whitelist by the smallest-first rule: add every
module that passes `mypy --strict` today with zero errors (likely dozens of the small
`raytrace/`, `algebra/`, `analysis/` modules), rather than waiting on a big-bang cleanup.

---

**[P1-5] The dev interpreter (3.14) is in no CI leg, and the pyproject comment justifying
its exclusion is factually false on this machine**
`pyproject.toml:40-44`, `.github/workflows/unit-tests.yml:24`, `validate.yml:23`

Evidence. The classifier block says 3.14 is dropped because *"3.14 is too new for the
optional accelerator dependencies (numba, jax, pyfftw) to have published wheels against"*.
Measured in the live environment:

```
py 3.14.6 | numpy 2.4.6 | scipy 1.17.1 | numba 0.65.1 | jax 0.10.1 | pyfftw 0.15.1
```

All three named accelerators have 3.14 wheels and are installed. The CI matrix runs
3.10–3.13 (`unit-tests.yml`) and 3.11–3.13 (`validate.yml`). Corroborating evidence that 3.14
is the working interpreter: the repo **tracks** `.benchmarks/Windows-CPython-3.14-64bit/0001_v4_11_2.json`
and `0002_v4_11_2_zernike.json`, and `tests/unit/__pycache__/` holds `cpython-314` artefacts.

Impact. Every line of this library is developed and hand-tested on an interpreter that no CI
leg exercises. 3.14-specific behaviour changes (e.g. anything touching the free-threading
build, `ast` changes, or deprecation removals) reach `main` unverified.

Fix. Add `'3.14'` to the `unit-tests.yml` matrix and restore the classifier. Delete the
obsolete justification comment. If a specific optional dep is genuinely missing a 3.14 wheel,
name that one dep in the comment instead of the general claim.

---

**[P1-6] The declared `jax` floor excludes the jax that is actually installed and exercised
locally**
`pyproject.toml:148`

Evidence. `jax = ['jax>=0.11.0; python_version >= "3.12"']`, with a comment asserting the
S4-4 CI leg gates 0.11 as *"the supported baseline"*. The dev machine runs **jax 0.10.1 on
Python 3.14** — below the declared floor. So every jax-guarded test the author runs locally
exercises a version the package metadata declares unsupported, while CI exercises 0.11 on
3.12. Two disjoint jax configurations, neither of which matches the other.

Impact. A jax API difference between 0.10 and 0.11 produces a local green / CI red (or the
reverse) with no signal about which is right. The dep-drift workflow (`dep-drift.yml`) is
explicitly non-blocking (`:74` "not in the merge-blocking pipeline; informative only") so it
cannot catch this.

Fix. Either install jax ≥ 0.11 locally, or lower the floor to the version actually in use and
add a CI leg on it. Add an assertion to the jax CI job that the resolved jax version satisfies
the declared floor.

---

**[P1-7] Installed distribution metadata is two releases behind the source version**
measured at runtime

Evidence.
```
lumenairy.__version__      = 5.45.1
importlib.metadata.version = 5.43.0
```
The editable-install finder is older still — `__editable___lumenairy_5_21_2_finder` appears in
the `-X importtime` trace, i.e. the `.pth` shim was generated at **5.21.2**.

Impact. Anything reading installed metadata (packaging checks, telemetry, a user's
`pip show`, plugin version gates) sees 5.43.0. No test asserts the two agree, so the drift is
invisible.

Fix. Add a one-line test: `assert importlib.metadata.version("lumenairy") == lumenairy.__version__`
(skip when not installed). Re-run `pip install -e .` in the dev env.

---

**[P1-8] The `apply_real_lens` family's kwarg space is essentially uncovered in combination**
`lumenairy/elements/_lens_real.py:3773`, `_lens_traced.py:6731`, `lenses_maslov.py:1087`

Evidence. Across the whole test corpus there are **673** `apply_real_lens*` call sites
exercising **177 distinct kwarg combinations**. The distribution of non-trivial kwargs per
call:

| kwargs passed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 17 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| call sites | 188 | 272 | 100 | 36 | 23 | 19 | 14 | 11 | 4 | 2 | 1 | 1 | 2 |

**68 %** of all call sites (460/673) pass zero or one optional kwarg. `apply_real_lens_traced`
has **48 parameters**; exhaustive *pairwise* coverage alone would need ~1 000 combinations.
The top combinations are single knobs: `('carrier',)` ×38, `('sag_chunk_rows',)` ×37,
`('surface_model',)` ×21.

Impact. Interaction defects — which is what all five orchestrator findings are — are
structurally undetectable. The suite tests knobs one at a time against a default background.

Fix. Two cheap, high-yield additions: (1) a pairwise/covering-array parametrisation over the
~12 *physics-affecting* kwargs (not the memory/perf knobs), asserting only energy conservation
and finiteness — cheap invariants that catch most interaction breakage; (2) a property test
that every kwarg at its default reproduces the no-kwarg call bit-for-bit, which cheaply
catches "knob silently ignored" (the `test_niche_audit_e_prepared_and_enums.py:738`
"discarded physics kwargs" test suggests this class of bug is live).

---

**[P1-9] `tests/unit/test_public_api.py` generates 712 tests — 5.3 % of the entire suite —
all of which are `hasattr` / `__all__` tautologies**
`tests/unit/test_public_api.py` (104 lines)

Evidence. Measured directly: `pytest tests/unit/test_public_api.py -q` → **712 passed in 1.90 s**
from a 104-line file. `lumenairy.__all__` holds **708 symbols**; the file parametrises over
them. It has a dedicated CI job (`unit-tests.yml:275-292`).

Impact. Not harmful in itself (it is fast), but it inflates the headline test count by 5.3 %
and makes "13 319 tests" a poor proxy for physics coverage. Combined with the 41 % contract
share measured below, the suite's apparent size substantially overstates its physics reach.

Fix. Collapse to a handful of tests (one loop assertion, not 708 parametrisations) so the
test count reflects distinct behaviours. Keep the CI job.

---

**[P1-10] Machine-specific benchmark artefacts are committed and `.gitignore` does not
cover the cache directories**
`.gitignore`, `git ls-files`

Evidence. Tracked: `.benchmarks/Windows-CPython-3.14-64bit/0001_v4_11_2.json`,
`.../0002_v4_11_2_zernike.json`. `.gitignore` (47 lines) has **no** entry for `.benchmarks/`,
`.mypy_cache/`, `.ruff_cache/` — all three of which exist in the working tree. (`importtime.log`
is incidentally covered by the blanket `*.log`; `.test_durations` is tracked deliberately for
pytest-split, which is correct.)

A second hazard: `.gitignore` blanket-ignores `*.png`, `*.jpg`, `*.pdf`, `*.dat`, `*.log`.
Any legitimate binary fixture or doc asset with those extensions is silently un-addable —
`validation/` already holds 33 `.png` (11.6 MB) that can never be committed without `-f`.

Impact. Any developer running `pytest-benchmark`, `mypy` or `ruff` dirties the tree; the two
committed JSONs pin one Windows/3.14 machine's timings into version control.

Fix. Add `.benchmarks/`, `.mypy_cache/`, `.ruff_cache/` to `.gitignore`; untrack the two
benchmark JSONs. Replace the blanket `*.png`/`*.dat` ignores with directory-scoped ones
(`output/**/*.png`).

---

### P2 — fragile or wasteful design

---

**[P2-1] The suite's documented runtime contract is wrong by a factor of ~440**
`pyproject.toml:229-231` and `:249`

Evidence. The config says `tests/unit/` is *"fast (<30s) API-contract tests"* and the `unit`
marker is documented as *"fast API-contract tests, no external deps (<30s total)"*. Measured
from `.test_durations`: **13 136 s = 3.65 h**, with 26 individual tests over 60 s (max 308.7 s,
`test_audit_dynameta_consumer_api_2.py::test_b2_pure_amplitudes_vs_rcwa_conical`).

Secondary evidence that the marker system is vestigial: `mark.unit` is applied **0** times,
`mark.regression` **0** times, `mark.bench` **0** times, `mark.integration` 3 times. Only
`mark.slow` is used (108 applications in 54 of 536 files), accounting for at most 4 225 s
(32 %) — meaning the "fast" gate carries **≥ 8 900 s ≈ 2.5 h**, or ≥ 30 min per shard against
a 45-minute step timeout. That is a ~1.5× margin: a slow runner or one new heavy test
times a shard out.

Fix. Rewrite the two comments to the measured truth. Then either mark more files `slow`
(the six files over 5 min total carry 3 005 s alone) or raise the shard count. A `>5 s` bar
would move 522 tests (4 % of the suite) carrying 81 % of the total time.

---

**[P2-2] The test suite is dominated by contract and self-consistency checks, not physics
oracles**

Method. `random.seed(0)`, `random.sample(funcs, 70)` over the 7 664 AST-collected test
functions; each read and hand-classified (assertion lines dumped, not regex-guessed).

| Class | Count | Share |
|---|---|---|
| (iv) contract / signature / error-path | 29 | **41 %** |
| (i) physics oracle (independent truth) | 18 | **26 %** |
| (iii) tautology / self-consistency (code vs. itself) | 18 | **26 %** |
| (ii) regression pin to a stored literal | 3 | 4 % |
| (v) doc / source-text assertion | 2 | 3 % |

Whole-corpus regex corroboration over all 7 664: `NO-ASSERT` 928 (12.1 %), `SELF-CONSISTENCY`
763 (10.0 %), `SMOKE/FINITE` 513 (6.7 %), `DOC/SOURCE-TEXT` 376 (4.9 %), `MOCK` 268 (3.5 %),
`NEGATIVE-ONLY` (`assert not allclose`) 162 (2.1 %), `TRIVIAL` 62 (0.8 %).

Ten concrete weak tests, with file:line:

1. `tests/unit/test_audit_glass.py:147` — unfalsifiable bar (P1-1).
2. `tests/unit/test_v5_2_off_axis_conic_surface_frame.py:327` — `array_equal(no_kwarg, default_kwarg)`; tests Python defaults.
3. `tests/unit/test_v5_2_off_axis_conic_surface_frame.py:372` — same tautology with decentre/tilt.
4. `tests/unit/test_v5_2_off_axis_conic_surface_frame.py:230` — `assert not np.allclose(...)`; only asserts the branches differ.
5. `tests/unit/test_v5_2_off_axis_conic_surface_frame.py:415` — `np.all(np.isfinite(...))` smoke only.
6. `tests/unit/test_v4_15_agent_e.py:112` — `assert "N-LASF9" in src`, `assert "n = 1.85" in src or "n=1.85" in src`; asserts on **source text**, so reformatting breaks it and a wrong index passes.
7. `tests/unit/test_v4_16_2_agent_d.py:30,45,93,106,119` — reads `README.md`, `ROADMAP.md`, `CHANGELOG.md` and asserts on their prose.
8. `tests/unit/test_v5_2_walker_changelog_changeset.py:60` / `test_v5_3_walker_changelog_self_citation.py:70` — whole tests whose subject is the CHANGELOG text.
9. `tests/unit/test_v5_6_glass_memo.py:30` — `assert len(sentinels) >= 1`; near-vacuous.
10. `tests/unit/test_audit_propagation.py:3757`, `test_audit_raytrace.py:1092,1568`, `test_perf_v4_12_0_through_focus.py:714`, `test_v4_15_agent_a.py:351` — `assert speedup >= 5.0 / 10.0 / SPEEDUP_FLOOR`: **per-build wall-clock facts**, the exact shape `TESTING_STANDARDS.md` S1 forbids.

The suite **is** measured against its own S1–S5 standards, with the following violations
found (536 files scanned in-process):

| Fragile shape | Hits | Files |
|---|---|---|
| S1 assertions on measured time / speedup | 24 | 19 |
| S1b assertions on a measured-ratio bar | 6 | 5 |
| S3 `pytest.skip()` on env/resource preconditions | 203 | 84 |
| S3b resource probes inside tests (`psutil`, `cpu_count`, `free_gb`, RAM budget) | 444 | 121 |
| S4 `assert < <4+-digit float literal>` | 1 | 1 |
| S5 exact `len(...) == N` on machinery output | 426 | 157 |
| (v) tests reading `.md` files and asserting on prose | 22 | 10 |
| (v) tests asserting on `__doc__` / `inspect.getsource` | 211 | 80 |

Notably `tests/unit/test_niche_d3_guards.py:735` embeds a **CI timing table** in its docstring
(`CI py3.11 / numpy 2.4.6  21.0394  21.4420  0.98x  <- FAILED`) — the artefact of a previously
retired per-build pin, preserved in the test that replaced it.

Fix. (a) Convert the 24 timing assertions to operation-count or complexity-order assertions
(the pattern `tests/unit/test_pipeline.py:63` already documents — "The resume tests assert on
THIS rather than on a wall-clock"); (b) delete the 10 `.md`-prose tests and enforce doc
freshness in a docs job instead; (c) replace `inspect.getsource` proxies with behavioural
pins — `test_audit_glass.py:465` shows the team already knows this ("replace `inspect.getsource`
proxy with behavioral pin") but 211 sites remain.

---

**[P2-3] `.test_durations` is stale for 407 collected tests, and the CI relies on it for
shard balance**
`.test_durations` (1.7 MB, tracked), `unit-tests.yml:144-154`

Evidence. 12 941 timed ids vs. 13 319 collected today: **407 collected tests have no timing**
and 29 timed ids no longer exist. pytest-split assigns untimed tests a default, so balance
degrades as the gap grows. New untimed files include `test_ci_kernel_consistency.py`,
`test_backend_disable_jax.py`, `test_audit_s1_2_rcwa_lossless_tripwire.py`,
`test_bor_anisotropic.py`.

The balance itself is currently excellent — a greedy 5-way split on the committed durations
gives 2 627.3 s per shard, **max/min = 1.000**. For contrast, an alphabetical split (what you
get with no durations) gives 70.4 / 34.6 / 68.8 / 18.3 / 26.8 min — a 3.8× imbalance. So the
file is doing real work and its staleness is a real risk, not a cosmetic one.

Fix. Add a CI step that fails (or warns loudly) when > 2 % of collected ids are missing from
`.test_durations`, and regenerate on that signal. The workflow already documents the
regeneration procedure at `:125-132`.

---

**[P2-4] `_lens_traced.py` and `carrier.py` are ~37 % git history**

Evidence (`tokenize` comment tokens with block-extension for version/audit/date markers, plus
docstrings carrying ≥ 2 such markers):

| file | lines | comment lines | history comments | docstring lines | history-heavy docstrings | **total history** |
|---|---|---|---|---|---|---|
| `elements/_lens_traced.py` | 13 227 | 5 043 (38.1 %) | 3 330 (66 % of comments) | 2 996 | 1 639 (54.7 %) | **4 969 = 37.6 %** |
| `propagators/carrier.py` | 10 594 | 2 528 (23.9 %) | 1 533 (60.6 %) | 3 684 | 2 374 (64.4 %) | **3 907 = 36.9 %** |
| `elements/_lens_real.py` | 6 863 | 1 630 (23.8 %) | 802 (49.2 %) | 1 712 | 794 (46.4 %) | **1 596 = 23.3 %** |
| `elements/_lens_imap.py` | 1 840 | 554 (30.1 %) | 315 (56.9 %) | 423 | 243 (57.4 %) | **558 = 30.3 %** |
| `propagators/fft_infra.py` | 2 414 | 626 (25.9 %) | 389 (62.1 %) | 870 | 441 (50.7 %) | **830 = 34.4 %** |
| `elements/lenses_maslov.py` | 3 722 | 588 (15.8 %) | 369 (62.8 %) | 529 | 255 (48.2 %) | **624 = 16.8 %** |
| **6-file total** | **38 660** | | | | | **12 484 = 32.3 %** |

Judged by hand, these blocks document **history, not code** — the shape is consistently
"v5.xx (audit X): pre-fix this did A, which was wrong because B; now it does C". `git blame`,
`git log` and a 1.1 MB CHANGELOG already carry that. The same convention appears in
`pyproject.toml`, where lines 66–72 and 74–80 are a **verbatim duplicated** scipy-floor
rationale paragraph — an artefact of history-in-config that nobody re-reads.

Impact. A reader of `apply_real_lens_traced` (5 719 lines, cc 563) must skim ~2 000 lines of
narrative about what the function *used to* do to find what it *does*. This is a direct
contributor to P1-8: the code is too expensive to read, so tests are written against knobs
rather than behaviour.

Fix. Mechanical and safe: move every `v<N>.<N> (audit …)` block to the CHANGELOG entry it
already cites, leaving a one-line pointer where the rationale is load-bearing for the current
code. Budget ~1 week for the top 3 files; it removes ~10 500 lines with zero behaviour change,
and shrinks `_lens_traced.py` from 13 227 to ~8 300.

---

**[P2-5] 53 process-global configuration knobs, none with a context-manager form and none
with a reset**

Evidence. AST census of `set_*/get_*/reset_*/enable_*/disable_*/clear_*/configure_*` across
227 modules: 103 such functions, 22 of which write a module-level `global`. Grouped into
setter families, **53 have a `set_` verb; 0 have a context-manager form; 0 have a `reset_`**.
(One `@contextmanager` exists anywhere in that set.) Genuine process-global knobs (excluding
Qt property setters in `ui/`):

`fft_infra.py` — `asm_cache_size`, `default_complex_dtype`, `default_dy`, `default_real_dtype`,
`default_wave_propagator`, `fft_auto_promote`, `fft_double_buffer`, `fft_fallback`,
`fft_plan_cache_size`, `fft_plan_max_bytes_per_buffer`, `fft_threads`, `pyfftw_planner` (12);
`_lens_real.py` — `lens_sag_dtype`, `pointwise_cos_grid_cache_budget`;
`_lens_traced.py` — `lens_parallel_amp`; `memory.py` — `max_ram`, `low_memory`;
`cache.py` — `budget`, `cache_budget`; `rcwa/_core.py` — `blas_threads`;
`io/storage.py` — `storage_backend`; `user_library.py` — `library_path`.
Only 3 environment variables exist (`LUMENAIRY_MEM_BUDGET_MB`, `LUMENAIRY_CACHE_BUDGET_MB`,
`LUMENAIRY_DISABLE_JAX`) — good restraint there.

Test hygiene is partial. `tests/conftest.py` (679 lines) does carry 2 autouse guards
(`_module_flag_leak_guard`, `_module_glass_registry_guard`) and an `fft_state_ctx` fixture.
But per-knob restore coverage is uneven — e.g. `set_asm_cache_size` (1 file, 2 calls,
**0** `finally`), `set_fft_fallback` (1 file, **0** `finally`), `set_cache_budget` (2 files,
15 calls, **0** `finally`), `set_blas_threads` (2 files, 9 calls, 1 `finally`, 0 fixtures),
`set_fft_threads` / `set_fft_plan_max_bytes_per_buffer` (0 fixtures).

Impact. Cross-test leakage is possible in a suite that is deliberately run **serially** (the
CI comments say in-process xdist is unsafe), so a leaked global can silently change a later
test's result and will not reproduce under `-k`.

Fix. Add a `@contextmanager` `override(...)` beside every `set_*` (a 10-line generic helper
plus one line per knob) and an autouse conftest fixture that snapshots and restores all 20
process globals. Both are mechanical.

---

**[P2-6] `validation/` is 70 % of the tracked repo and mostly one-off audit probe scratch**

Evidence. `git ls-files` → **3 970** tracked files, of which **2 785 (70 %)** are under
`validation/`, including **1 493** tracked `.npz/.npy/.csv/.json`. On disk `validation/` holds
3 144 files in **128 subdirectories**: 8 134 MB of `.npz`, 3 222 MB of `.npy`, 59.5 MB of
`.json`, 21.8 MB of `.csv`. `.git` is **550 MB**. The subdirectory names are overwhelmingly
one-off probes — `probe_fix_bor_round2`, `probe_fix_bor_round3`, `probe_fix_sliver_round3`,
`probe_fix_sliver_round4`, `probe_pmm2d_mortar_round2`, `probe_fix_slant_anchor_v1v2o2`, … —
i.e. scratch from individual audit rounds, promoted to permanent tracked artefacts.

`MANIFEST.in:30-32` still says *"Validation suite — **31 files** with ~370 physics-fidelity
tests"* and ships `recursive-include validation *.py *.md *.txt`. The real count is **876 `.py`**
— so every PyPI sdist carries ~1 194 validation files, most of them dead probe scripts.

Fix. Move `probe_*` trees to a separate `probes/` directory excluded from `MANIFEST.in` and
(for the data) from git — or to a release-asset / DVC store. Correct the MANIFEST comment.
Expected effect: sdist shrinks by ~80 % of its validation payload; `.git` stops growing by
tens of MB per audit round.

---

**[P2-7] Import-time cost is structural: `import lumenairy` eagerly pulls `scipy.linalg` via
the RCWA/Berreman path**

Evidence (`python -X importtime -c "import lumenairy"`; absolute times are inflated ~250× by
the loaded machine, so read the **chain and the ratios**, not the seconds):

```
lumenairy                      cum 414 991 ms
 └ lumenairy.analysis          cum 414 574
    └ analysis.coherence       cum 338 014
       └ elements.lenses       cum 338 014
          └ elements           cum 338 014
             └ elements.berreman  cum 303 395
                └ elements.rcwa   cum 303 394
                   └ rcwa._core   cum 303 391
                      └ lumenairy.backend      cum 303 390
                         └ backend.scipy       cum 303 389
                            └ scipy.linalg     cum 271 584   (65% of total)
```

`lumenairy`'s own module bodies cost **877 ms across 162 modules** — i.e. essentially nothing.
**~65 % of import time is `scipy.linalg`**, reached because `analysis.coherence` needs
`elements.lenses`, which triggers the whole `elements` package `__init__`, which imports
`berreman` → `rcwa` → `backend.scipy`. A user who only wants `propagate_asm` pays for the full
rigorous-solver stack.

Fix. Make `lumenairy/elements/__init__.py` lazy for the heavy solver subpackages
(`rcwa`, `pmm`, `bor`, `berreman`, `eme`) via PEP 562 `__getattr__`. That is a ~30-line change
and, on the numbers above, removes ~65 % of import time. The library already uses lazy
in-function imports extensively (415 of 948 edges), so the idiom is established.

---

**[P2-8] Import cycles: 9 SCCs, 54 two-cycles, 12 layering violations**

Evidence (AST, module-level vs. in-function edges separated). The orchestrator's suspected
cycles are confirmed and I found their exact shape:

- `elements/_lens_traced.py:33` `from . import _lens_imap as _IMAP` (**module level**) while
  `_lens_imap.py:820, 877, 1464, 1544` do `from ._lens_traced import ...` (in-function). The
  source comment at `_lens_traced.py:31` claims *"No import cycle: `_lens_imap` reaches back"*
  — that is a cycle broken by laziness, not the absence of one; it is load-order-sensitive.
- `elements/_lens_traced.py:558` `from ._lens_real import apply_real_lens` (module level), and
  `_lens_traced.py:8140, 8353, 13189` import three more `_lens_real` symbols lazily.
- **Module-level** 2-cycles among real modules (not `__init__` re-export hubs):
  `_lens_real ↔ lenses`, `_lens_thin ↔ lenses`, `_lens_traced ↔ lenses`,
  `lenses ↔ lenses_maslov`, `lumenairy ↔ lenses_maslov`.
- Lazy-only 2-cycles: `eme/_jax_modes ↔ eme/eme_2d_vector`, `eme_2d ↔ eme_2d_vector`,
  `pmm/_jax_twod ↔ pmm/twod`, `pmm/stack2d_pure ↔ pmm/twod_staggered`.
- Largest SCC at module level: **43 modules** (rooted at the `lumenairy` package `__init__`,
  which both re-exports from and is imported by `analysis`, `elements`, `optimize`,
  `propagators`, `raytrace`).

Layering violations (deeper layer importing a more app-level one), 12 edges, **11 of 12 are
lazy** — so the layering is respected at import time and broken only at call time:

| from → to | edges | at module level |
|---|---|---|
| `elements → propagators` | 9 | 0 |
| `algebra → elements` | 2 | 0 |
| `elements → analysis` | 2 | 0 |
| `propagators → analysis` | 2 | 0 |
| `raytrace → propagators` | 2 | 0 |
| `raytrace → elements` | 2 | **1** (`raytrace/surface.py → elements.lenses`) |
| `sources → propagators` | 2 | 0 |
| `algebra → sources / raytrace / propagators` | 3 | 0 |
| `elements → io` | 1 | 0 |
| `io → propagators` | 1 | 0 |

Notably `lumenairy/propagators/carrier.py` and `lumenairy/elements/_lens_traced.py` each have
fan-out 16, the highest in the library after `lumenairy/__init__.py` (70) and
`ui/main_window.py` (51).

Fix. The one module-level layering violation (`raytrace/surface.py → elements.lenses`) is a
one-line fix. For the `_lens_*` cluster, the durable fix is to extract the shared types
(`_TracedExitSupport`, `_solve_lstsq_thread_safe`, sag helpers) into a leaf module
`elements/_lens_kernels.py` that all four import one-way.

---

**[P2-9] Duplicated backend-detection scaffolds across 5 modules each**
`difflib` + normalised-AST

| helper | definitions | modules |
|---|---|---|
| `_ensure_cupy_loaded` | 5 | `_lens_real.py:34`, `_lens_traced.py:38`, `lenses.py:46`, `fft_infra.py:46`, `sources/core.py:26` |
| `_is_cupy_array` | 5 | `_lens_real.py:42`, `_lens_thin.py:41`, `_lens_traced.py:46`, `lenses.py:119`, `fft_infra.py:54` |
| `_load_numba` | 5 | `_lens_imap.py:415`, `_lens_traced.py:74`, `lenses.py:85`, `_merit_jit.py:58`, `fga.py:95` |

`_load_numba` is **byte-identical** across `_lens_traced.py:76` / `_merit_jit.py:60`
(similarity 1.00) and 0.91 against the other two. Other cross-module near-duplicates:
`_resolve_output_shape` 0.96 across `propagators/hf.py:33`, `hfpi.py:53`, `gbd.py:200`;
`_resolve_incidence_checked` identical across `pmm/_core.py:133` and `rcwa/oned.py:286`.

Overall duplication is **lower than expected**: only 9 exact clusters (~573 redundant lines)
across the whole library, and the single largest is 40 copies of an 11-line Qt
`minimumSizeHint`. This is a mild finding, not a structural one.

Fix. One `lumenairy/backend/_optional.py` exporting `ensure_cupy()`, `is_cupy_array()`,
`load_numba()`. ~1 hour, removes ~13 duplicate definitions, and — more valuably — gives one
place to test the accelerator-absent path.

---

**[P2-10] Function-size and parameter-count hot spots**

| span | cc | params | location | function |
|---|---|---|---|---|
| **5 719** | **563** | **48** | `elements/_lens_traced.py:6731` | `apply_real_lens_traced` |
| **2 138** | 303 | 29 | `elements/_lens_real.py:4524` | `_apply_real_lens_impl` |
| 1 595 | 137 | 29 | `propagators/carrier.py:7126` | `propagate_traced_carrier_chain` |
| 1 289 | 117 | 29 | `elements/lenses_maslov.py:1087` | `apply_real_lens_maslov` |
| 1 159 | 149 | 23 | `optimize/driver.py:477` | `design_optimize` |
| 937 | 98 | **36** | `propagators/carrier.py:9658` | `propagate_traced_carrier_chain_multi` |
| 856 | 136 | 3 | `io/prescriptions_zemax.py:370` | `load_zemax_zmx` |
| 749 | 2 | 28 | `elements/_lens_real.py:3773` | `apply_real_lens` (pure delegator) |
| 682 | 86 | 1 | `ui/waveoptics_dock.py:510` | `WaveOpticsWorker._run_impl` |
| 655 | 8 | 3 | `ui/waveoptics_dock.py:1205` | `WaveOpticsDock.__init__` |
| 652 | 32 | 15 | `propagators/asymptotic_aberration_tensor.py:608` | `aberration_tensor` |
| 620 | 53 | 13 | `propagators/system.py:365` | `propagate_through_system` |
| 588 | 59 | 22 | `propagators/carrier.py:4382` | `carrier_referenced_exact_focus_readout` |
| 585 | 58 | 17 | `elements/_lens_imap.py:1213` | `build_inverse_map` |
| 546 | 47 | 9 | `analysis/image_plane_wfe.py:198` | `eval_image_plane_wfe` |
| 518 | 96 | 3 | `io/prescriptions_zemax.py:1232` | `load_zemax_prescription_data_txt` |
| 517 | 56 | 10 | `propagators/asymptotic.py:252` | `propagate_modal_asymptotic` |
| 517 | 28 | 16 | `elements/doe.py:538` | `makedammann2d` |
| 516 | 45 | 24 | `propagators/carrier.py:6407` | `_fine_trace_group_exit` |
| 479 | 94 | 3 | `elements/pmm/stack.py:2599` | `PMMStack.solve` |
| 474 | 75 | 4 | `elements/pmm/stack2d.py:1156` | `PMM2DStackHybrid.solve` |
| 455 | 42 | 18 | `elements/rcwa/twod.py:676` | `rcwa_efficiency_2d` |
| 447 | 33 | 10 | `elements/_lens_thin.py:69` | `apply_thin_lens` |
| 428 | 40 | 17 | `elements/rcwa/oned.py:321` | `rcwa_efficiency_1d` |
| 427 | 27 | 11 | `elements/_lens_traced_multibranch.py:330` | `apply_real_lens_traced_multibranch` |

58 functions exceed 300 lines; 236 exceed 150. Beyond the expected two, the genuinely
surprising entries are `io/prescriptions_zemax.py:370` (cc **136** in a file parser) and
`ui/waveoptics_dock.py:1205` (a 655-line `__init__`).

Files > 3 000 lines (13): `_lens_traced.py` 13 228, `carrier.py` 10 595, `pmm/_core.py` 7 515,
`_lens_real.py` 6 864, `pmm/stack.py` 5 051, `rcwa/_core.py` 4 416, `gbd.py` 3 819,
`lenses_maslov.py` 3 723, `pmm/twod_staggered.py` 3 628, `ui/main_window.py` 3 626,
`rcwa/stack.py` 3 264, `sources/core.py` 3 186, `fga.py` 3 078.

---

### P3 — quality

**[P3-1] 53 of 227 library modules are never named in any test file** (23 %). Excluding
`ui/` (which has no headless test harness — PySide6 is not installed), the substantive gaps
are: `lumenairy/_math`, `analysis/aberration`, `analysis/coronagraph`, `elements/_traced_flags`,
`elements/coronagraph`, `elements/materials`, `elements/bor/_inv_census`, `bor/_jax_bor`,
`bor/_jax_sem`, `bor/_orient`, `eme/_jax_modes`, `pmm/_jax_stack2d`, `pmm/_jax_twod_jones`,
`propagators/_bluestein`, `propagators/asymptotic_maslov`, `raytrace/paraxial`. Note
`raytrace/paraxial.py` is *also* the highest comment-to-code ratio in the library (3.81 — 198
comment/doc lines to 52 code lines).

Only 43 of 227 modules have a name-matched dedicated test file; 184 do not. Test files are
organised by **audit episode** (`test_niche_d7_decentred_fit.py`, `test_v5_21_delta_audit.py`,
`test_verify_pmmstack_sliver_round3.py`) rather than by module, which is why module-level
coverage is hard to reason about.

**[P3-2] Documentation load is large and heavily historical.** `docs/` holds **352 files,
10.9 MB**, of which **272 have AUDIT in the name** and **267 live in `docs/audits/`**; 88 are
named `FIX_*`/`VERIFY_*`/`GAP_*`/`PROBE_*`. Plus 62 `docs/release_notes/`. Plus a 1.1 MB
`CHANGELOG.md`, a 180 KB `README.md`, a 322 KB `docs/changelogs/v4.md`, and a 421 KB single
audit (`AUDIT_V5_17_0_2026_07_01_DEEP.md`). Doc mtimes cluster in bursts (2026-05: 102,
2026-08: 129, 2026-09: 55) — i.e. audit rounds, not steady documentation. One file is
**future-dated**: `docs/audits/SCOPE_BOR_MULTILAYER_GUARDS_2026_09_12.md` (today is
2026-09-11).

Discoverability check: I sampled 60 of the 2 263 distinct backtick identifiers in
`README.md` / `ROADMAP.md` / `Migration-Guide.md` / `CONVENTIONS.md` / `docs/**` (excluding
`docs/audits/`) with `random.seed(0)` and resolved each against `lumenairy` and its nine
subpackages: **13 resolve, 47 do not (78 %)**. *That headline overstates the problem* — many
non-resolvers are legitimately not API: exception names (`ValueError`), parameter names
(`wavelength`, `dtype`), and test-function citations inside `docs/release_notes/`. The
genuinely broken public-facing refs are the interesting subset:

- `README.md:714` cites `_decompose_prescription` (private, 4×)
- `README.md:2190` cites `focus_fixed_sampling` (2×) — does not resolve
- `ROADMAP.md:140` cites `_detect_backend` (6×) — does not resolve
- `Migration-Guide.md:711` cites `return_kind` (7×), `:714` `opl_fn`, `:560` `image_centres`
- README also cites `world_origin` (6×), `world_R` (5×), `rcwa_1d` (3×),
  `_PROPAGATE_SYSTEM_JAX_CACHE` (3×), `row_reset`, `fiber_mode`, `_spawn_rng`, `_fd_grad_pure`

Of the 250 most-cited README identifiers, **104 do not resolve** (again inflated by exception
names, but the private-symbol and stale-name cases above are real).

**[P3-3] `benchmarks/` is thin and ungated.** 7 `.py` files
(`test_bench_asymptotic`, `_baseline`, `_fft_infra`, `_jax_jit`, `_through_focus`,
`_zernike_cache`) plus a `conftest.py` and `README.md`. Results land in `.benchmarks/` — two
files, both from one Windows/CPython-3.14 machine, both **tracked**. No CI workflow references
benchmarks; `mark.bench` is applied **0 times** despite being a declared marker
(`pyproject.toml:253`). There is therefore **no performance-regression gate** at all.

**[P3-4] `scripts/` contains 12 probe scratch files.** Of 15 `.py`: `_d5_probe.py`,
`_d5_probe2..6.py`, `_d5_byteid.py`, `_d5_spl.py`, `_g8_c15lad.py`, `_g8_failbefore.py`,
`_g8_probe.py` — one-off audit scratch committed alongside the 4 real tools
(`check_dep_metadata.py`, `check_source_line_citations.py`, `stamp_changelog.py`,
`verify_changelog_closures.py`).

**[P3-5] `__pycache__` under `tests/unit/` holds artefacts for CPython 3.12, 3.13 and 3.14
and for pytest 9.0.3 / 9.1.1 simultaneously** — harmless (untracked, 0 `.pyc` tracked) but
another confirmation that multiple interpreters are in local use while CI pins 3.10–3.13.

---

## Metrics tables

### Test census

| metric | value |
|---|---|
| collected tests | 13 319 (of 13 358; 39 deselected) |
| collection time | 137.15 s |
| test `.py` files | 536 (532 unit, 2 integration, 2 infra) |
| `def test*` functions (AST) | 7 664 |
| test LOC | 260 341 (median file 371, mean 486) |
| largest test files | `test_audit_misc.py` 6 571; `test_audit_propagation.py` 4 506; `test_niche_audit_w3_oracles.py` 3 098; `test_audit_optimize.py` 2 909; `test_pmm_m2_window_contract.py` 2 554 |
| total runtime | 13 136 s (3.65 h) |
| median test | 0.0015 s; mean 1.015 s; max 308.7 s |
| `> 60 s` | 26 tests (0.20 %) = 3 314 s (25.2 %) |
| `> 30 s` | 85 (0.66 %) = 5 715 s (43.5 %) |
| `> 10 s` | 272 (2.10 %) = 8 910 s (67.8 %) |
| `> 5 s` | 522 (4.03 %) = 10 644 s (81.0 %) |
| `> 1 s` | 1 292 (9.98 %) = 12 541 s (95.5 %) |
| slowest test | 308.7 s `test_audit_dynameta_consumer_api_2.py::test_b2_pure_amplitudes_vs_rcwa_conical` |
| slowest file | 857.0 s (12 tests) `test_audit_dynameta_consumer_api_2.py` |
| `mark.slow` | 108 applications / 54 files (≤ 32 % of runtime) |
| `mark.unit` / `mark.regression` / `mark.bench` | 0 / 0 / 0 applications |
| `-k real_lens` selects | 135 tests (1.0 % of suite) across 23 files, 436 s |
| library modules never named in a test | 53 / 227 (23 %) |
| modules with a name-matched test file | 43 / 227 |

### Fast/slow split options (from `.test_durations`)

| bar | slow files | fast lane | slow lane |
|---|---|---|---|
| file total > 1 min | 61 | 3 231 s (53.8 min), 11 258 tests | 9 906 s, 1 683 tests |
| file total > 2 min | 27 | 6 249 s (104 min), 12 122 tests | 6 888 s, 819 tests |
| file total > 5 min | 6 | 10 132 s (169 min), 12 508 tests | 3 005 s, 433 tests |

Recommended: the **> 2 min** bar. It puts 819 tests (6 %) in a slow lane carrying 52 % of the
time, leaving a fast lane of 104 min ÷ 5 shards = **21 min/shard** (vs. ~30 today) — a
comfortable margin under the 45-min step timeout.

### 5-shard balance

| split method | shard loads |
|---|---|
| greedy on `.test_durations` | 2 627.3 s × 5 — **max/min = 1.000** |
| alphabetical (no durations) | 70.4 / 34.6 / 68.8 / 18.3 / 26.8 min — **3.85×** imbalance |

### Import-time top entries (loaded machine; read ratios, not absolutes)

| cumulative | self | module |
|---|---|---|
| 414 991 ms | 1.0 | `lumenairy` |
| 414 574 | 0.6 | `lumenairy.analysis` |
| 338 014 | 0.3 | `lumenairy.analysis.coherence` |
| 338 014 | 0.0 | `lumenairy.elements.lenses` |
| 303 395 | 0.7 | `lumenairy.elements.berreman` |
| 303 391 | 1.1 | `lumenairy.elements.rcwa._core` |
| 303 389 | 0.4 | `lumenairy.backend.scipy` |
| **271 584** | 1.5 | **`scipy.linalg` (65 % of total)** |
| 163 960 | 9 725 | `scipy.linalg._cythonized_array_utils` |
| 76 546 | 1.7 | `lumenairy.analysis.aberration` |
| 76 250 | 3.0 | `numpy` |
| 49 935 | 44 205 | `numpy.testing._private.utils` |
| 34 414 | 2.5 | `lumenairy.elements.lenses` |
| 34 389 | 12.0 | `lumenairy.elements._lens_real` |
| 34 191 | 2.3 | `lumenairy.propagators.carrier` |
| 34 188 | 2.2 | `lumenairy.propagators.fft_infra` |
| — | 189.0 | `lumenairy.elements.freeform` (highest lumenairy self) |
| — | 188.6 | `lumenairy.io.storage` |
| — | 173.4 | `lumenairy.propagators.carrier_field` |
| — | **877 total** | **all 162 lumenairy modules combined** |

### Import cycles (module-level = executed at import time)

| kind | count |
|---|---|
| module-level edges / lazy edges / total | 533 / 415 / 948 |
| SCCs with > 1 module (module-level) | 7 (largest 43 modules) |
| SCCs with > 1 module (incl. lazy) | 9 |
| 2-cycles (module-level) | 37 (32 are `__init__` ↔ submodule re-export) |
| 2-cycles (incl. lazy) | 54 |
| real module↔module 2-cycles at import time | `_lens_real ↔ lenses`, `_lens_thin ↔ lenses`, `_lens_traced ↔ lenses`, `lenses ↔ lenses_maslov`, `lumenairy ↔ lenses_maslov` |
| lazy-only real 2-cycles | `eme/_jax_modes ↔ eme_2d_vector`, `eme_2d ↔ eme_2d_vector`, `pmm/_jax_twod ↔ pmm/twod`, `pmm/stack2d_pure ↔ pmm/twod_staggered` |
| layering violations | 12 edges, 11 lazy, 1 module-level (`raytrace/surface.py → elements.lenses`) |
| highest fan-out | `lumenairy` 70, `ui/main_window` 51, `propagators/carrier` 16, `elements/_lens_traced` 16, `elements/lenses_maslov` 15 |
| highest fan-in | `elements` 41, `propagators` 27, `analysis` 21, `ui` 16, `elements.pmm` 14 |

### Comment/doc load — top 15 files where comments+docstrings exceed code

| ratio | comments+doc | code | lines | file |
|---|---|---|---|---|
| 3.81 | 198 | 52 | 268 | `raytrace/paraxial.py` |
| 3.31 | 886 | 268 | 1 228 | `elements/elements.py` |
| 3.17 | 380 | 120 | 570 | `elements/bor/_sem_contract.py` |
| 3.05 | 302 | 99 | 421 | `optimize/core.py` |
| 2.68 | 536 | 200 | 827 | `elements/bor/bor_solve.py` |
| 2.55 | 163 | 64 | 247 | `progress.py` |
| 2.50 | 145 | 58 | 221 | `propagators/result.py` |
| 2.37 | 372 | 157 | 587 | `optimize/context.py` |
| 2.17 | 537 | 247 | 804 | `analysis/detector.py` |
| 2.16 | 207 | 96 | 320 | `raytrace/_conic_core.py` |
| 2.15 | 258 | 120 | 406 | `raytrace/seidel_analysis.py` |
| 2.01 | 843 | 419 | 1 373 | `elements/_lens_thin.py` |
| 1.98 | 1 496 | 756 | 2 415 | `propagators/fft_infra.py` |
| 1.97 | 142 | 72 | 224 | `elements/thin_grating.py` |
| 1.93 | 1 090 | 565 | 1 743 | `raytrace/seidel.py` |
| 1.72 | **8 039** | **4 672** | **13 228** | `elements/_lens_traced.py` |

Library-wide: 106 827 code lines vs. 99 608 comment+docstring lines — a **0.93** overall ratio.

### Duplicate clusters (top, by removable lines)

| # | copies | ~lines | name(s) | locations |
|---|---|---|---|---|
| 1 | 40 | 11 | `minimumSizeHint` | 40 `ui/*_dock.py` (Qt boilerplate) |
| 2 | 5 | 13 | `_load_numba` | `_lens_imap:415`, `_lens_traced:74`, `lenses:85`, `_merit_jit:58`, `fga:95` |
| 3 | 5 | ~12 | `_is_cupy_array` | `_lens_real:42`, `_lens_thin:41`, `_lens_traced:46`, `lenses:119`, `fft_infra:54` |
| 4 | 5 | ~10 | `_ensure_cupy_loaded` | `_lens_real:34`, `_lens_traced:38`, `lenses:46`, `fft_infra:46`, `sources/core:26` |
| 5 | 3 | 39 | `_resolve_output_shape` (0.96) | `gbd:200`, `hf:33`, `hfpi:53` |
| 6 | 2 | 36 | `_resolve_incidence_checked` | `pmm/_core:133`, `rcwa/oned:286` |
| 7 | 2 | 15 | `_resolve_incidence` | `pmm/_core:116`, `rcwa/oned:276` |
| 8 | 6 | 10 | `clear_*_cache` family | `through_focus:92`, `zernike:255`, `_lens_real:734`, `fga:827`, `system:1287`, `jax_trace:1071` |
| 9 | 5 | 14 | `_draw_empty` | `coherence_dock:446`, `lg_aberration_dock:80`, `richards_wolf_dock:165`, `shack_hartmann_dock:100`, `thin_grating_dock:144` |
| 10 | 2 | 87 | `_cheb4d_vd9` / `_get_cheb4d_vd3_numba` (0.85) | `lenses_maslov:480`, `:322` |
| 11 | 2 | 12 | `_draw_message` (1.00) | `distortion_dock:243`, `spot_field_dock:266` |
| 12 | 3 | 11 | `_homog` | `pmm/_jax_stack2d:188`, `_jax_twod:370`, `_jax_twod_jones:204` |
| 13 | 3 | 9 | `_star` | `pmm/_core:3718`, `:4219`, `pmm/_jax_stack:366` |
| 14 | 2 | 18 | `_opd_vd3` / `_opd_vd9` (0.81) | `lenses_maslov:416`, `:573` |
| 15 | 2 | 16 | `_sample` / `_sample_bilinear_xp` (0.89) | `lenses_maslov:2777`, `:3538` |

Exact-duplicate clusters library-wide: **9**, ~573 redundant lines. The `t = -z/N` exit-vertex
idiom appears only **twice** (`raytrace/intersection.py:139`, `raytrace/jax_trace.py:235`) —
not a duplication problem.

### Global-config census (non-UI process globals)

| module | knobs | ctxmgr | reset |
|---|---|---|---|
| `propagators/fft_infra.py` | `asm_cache_size`, `default_complex_dtype`, `default_dy`, `default_real_dtype`, `default_wave_propagator`, `fft_auto_promote`, `fft_double_buffer`, `fft_fallback`, `fft_plan_cache_size`, `fft_plan_max_bytes_per_buffer`, `fft_threads`, `pyfftw_planner` (12) | 0 | 0 |
| `elements/_lens_real.py` | `lens_sag_dtype`, `pointwise_cos_grid_cache_budget` | 0 | 0 |
| `elements/_lens_traced.py` | `lens_parallel_amp` | 0 | 0 |
| `memory.py` | `max_ram`, `low_memory` | 0 | 0 |
| `cache.py` | `budget`, `cache_budget` | 0 | 0 |
| `elements/rcwa/_core.py` | `blas_threads` | 0 | 0 |
| `io/storage.py` | `storage_backend` | 0 | 0 |
| `user_library.py` | `library_path` | 0 | 0 |
| **totals** | **53 `set_*` families (incl. UI property setters)** | **0** | **0** |

Environment variables: 3 (`LUMENAIRY_MEM_BUDGET_MB`, `LUMENAIRY_CACHE_BUDGET_MB`,
`LUMENAIRY_DISABLE_JAX`).

Test-restore coverage (files using each knob / calls / files containing `finally` / files
containing a fixture): `set_fft_auto_promote` 17/45/10/16 (good);
`set_default_wave_propagator` 7/46/5/6; `set_default_complex_dtype` 8/44/6/4;
`set_max_ram` 5/23/5/1; **`set_cache_budget` 2/15/0/2**; **`set_asm_cache_size` 1/2/0/1**;
**`set_fft_fallback` 1/2/0/1**; `set_blas_threads` 2/9/1/0; `set_fft_threads` 1/4/1/0.

### Public API surface

| metric | value |
|---|---|
| `lumenairy.__all__` | **708** symbols |
| `__all__ +=` / `.extend` | 0 / 0 (single literal list) |
| tier comment markers in `__init__.py` | 10 |
| `__init__.py` size | 2 095 lines, fan-out 70 |
| functions with > 25 params | 6 (`apply_real_lens_traced` 48, `propagate_traced_carrier_chain_multi` 36, 3× 29, `apply_real_lens` 28) |
| functions with > 20 params | 22 |

---

## Untested `apply_real_lens`-family kwargs and combinations

Method: extracted every parameter name by AST from each entry point, then searched all 536
test files plus 902 `validation/` + `benchmarks/` + `examples/` files for `<name>=`.

**Never passed as a kwarg anywhere in the repo (tests, validation, benchmarks, examples):**

| function | never-exercised kwargs (default) |
|---|---|
| `apply_real_lens` (`_lens_real.py:3773`, 28 params) | `seidel_poly_order` (=6) |
| `_apply_real_lens_impl` (`_lens_real.py:4524`, 29) | `seidel_poly_order` (=6), `_accum_store` (=None) |
| `apply_real_lens_traced` (`_lens_traced.py:6731`, 48) | `newton_mask_dilate_coarse_px` (=2), `caustic_min_area_ratio` (=1e-06) |
| `apply_real_lens_maslov` (`lenses_maslov.py:1087`, 29) | `chunk_v2` (=64), `stationary_newton_iter` (=12), `stationary_newton_tol` (=1e-10) |
| `apply_real_lens_gbd` (`lenses_gbd.py:239`, 26) | `reexpand_threshold` |
| `apply_real_lens_fga` (`fga.py:1510`, 21) | — (none) |
| `apply_real_lens_traced_multibranch` (`_lens_traced_multibranch.py:330`, 11) | — (none) |

**Exercised in non-test code only:** `apply_real_lens_traced.fast_analytic_phase`,
`apply_real_lens_traced.amp_use_gpu`.

**Exercised by exactly ONE test file (single point of failure):**

- `apply_real_lens` / `_apply_real_lens_impl`: `absorption`, **`seidel_correction`**,
  **`surface_frame`**, `remap_order`, `accumulator_store`, `scratch_dir`,
  `stream_transfer_function`
- `apply_real_lens_traced`: `on_fit_domain_basis`, `parallel_amp_min_free_gb`,
  `newton_max_iters`, `return_screen`, `_exit_na_out` (and 2 files: `newton_poly_order`,
  `caustic_ray_subsample`, `_remap_launch_out`)
- `apply_real_lens_maslov`: `local_n_samples`, `local_window_sigma`, `levin_tol`, `input_na`
  (2 files: `output_plane_n`, `use_numexpr`, `collimated_input`, `fold_split`)
- `apply_real_lens_gbd`: `reexpand`, `reexpand_carrier`, `chunk_beamlets`
- `apply_real_lens_fga`: `w0_factor`, `nsig`, `prune_frac`, `separable`, `momentum_sampling`

**Combination coverage:** 673 call sites, **177 distinct combinations**, 68 % of sites passing
0–1 optional kwargs (full table in P1-8).

### Why the five orchestrator-found defects slipped through

| # | defect shape | what the suite has | verdict |
|---|---|---|---|
| **D1** | `seidel_correction=True`, curved last surface, assert on **focus position** | Exactly **1** test (`test_audit_glass.py:92`). Its prescription is `R2=inf` (**flat** last surface). It never mentions focus/`z_focus`/`bfd`/`argmax`. Its assertion is **unfalsifiable** (P1-1). | **0 tests could catch it.** Two independent reasons: wrong geometry, and a dead assertion. |
| **D2** | `surface_frame=True`, tilted surface, assert on **beam deviation** | Exactly **1** file, **3** tests. All mention tilt; **none** asserts centroid/deviation/chief-ray/direction/shift. Two of the three are `array_equal(default, explicit-default)` tautologies; one is a finiteness smoke test. | **0 tests could catch it.** The only quantitative check is a ±20 % slope band. |
| **D3** | `apply_real_lens_traced` with a **real-dtype** `E_in` | 107 traced call sites whose `E_in` assignment shows no complex marker — but these are `np.exp(-r²/w²)` Gaussians that NumPy makes `float64`, then the library presumably upcasts. The *only* dtype-intent test is `test_v4_14_0_dispatcher_pin_apply_lens.py:451` `test_complex64_input_preserves_complex64_output` — **complex**64, not real. No test asserts what a real input should produce. | **Untested by intent.** Real input is reached accidentally in ~107 places, so a wrong-but-plausible result is *pinned* by those tests rather than caught. |
| **D4** | `caustic='multibranch'` + `output_plane_distance` **at the paraxial focus** | 34 tests mention multibranch; 24 also pass `output_plane_distance`; only **6** evaluate at a focal/paraxial distance, and of those, `test_v5_21_delta_audit.py:127` is `..._runs_without_warning` and `test_v5_21_lens_accuracy_extensions.py:842` is `..._catastrophe_warns` — i.e. **warning-behaviour** tests, not field-correctness tests at the caustic. | **Near-miss.** The exact plane is visited, but every test there asserts on the *warning*, not the field. |
| **D5** | `newton_fit='spline'` with a **vignetted** ray | 18 tests pass `newton_fit='spline'`; 9 also mention vignetting/aperture/nan — but all 9 are in `test_fix_d5_fit_domain_basis.py` / `test_niche_audit_w3_elements.py` / `test_niche_d7_decentred_fit.py` and are about **the `fit_domain_basis` knob's announcement and error messages**, not about a ray that misses the aperture. Only 27 of 536 files mention "vignett" at all, none of them a spline-fit test. | **0 tests could catch it.** `newton_fit='spline'` is tested for knob-plumbing, never for degenerate ray geometry. |

**The common root cause.** Four of the five are *interaction* defects — a flag combined with a
geometry. The suite is organised by **audit episode** (one file per past bug) and tests **one
knob at a time** against a default background (68 % of call sites pass ≤ 1 kwarg). It therefore
has, by construction, almost no power against flag×geometry interactions. The fifth (D1) is a
*dead assertion*, which no amount of coverage fixes — it needs the wrap removed.

---

## Architecture recommendations (prioritised, with effort estimates)

| # | change | effort | payoff |
|---|---|---|---|
| 1 | Fix the `seidel_correction` assertion (unwrap properly, re-bar), add curved-last-surface + focus-position cases (P1-1) | **0.5 d** | Restores the only gate on an opt-in physics correction |
| 2 | Add a beam-deviation test for `surface_frame=True`; delete the 2 default-argument tautologies (P1-2) | **0.5 d** | First real test of the kwarg |
| 3 | Flip `continue-on-error: false` on the ruff job; scope `lumenairy/ui/` back in (P1-3) | **0.5 d** | Makes the only always-on static gate binding |
| 4 | Add Python 3.14 to the CI matrix; fix the stale classifier comment; reconcile the jax floor (P1-5, P1-6) | **1 d** | The dev interpreter becomes a tested interpreter |
| 5 | Pairwise covering-array parametrisation over the ~12 physics-affecting `apply_real_lens*` kwargs, asserting energy + finiteness only (P1-8) | **3 d** | Directly targets the interaction class that produced 4 of 5 defects |
| 6 | Re-mark the slow lane at the **> 2 min/file** bar; add a `.test_durations` staleness check (P2-1, P2-3) | **1 d** | Fast shard 30 → 21 min, restores CI headroom |
| 7 | `@contextmanager override(...)` + autouse snapshot/restore fixture for the 20 process globals (P2-5) | **2 d** | Removes a whole class of order-dependent flake |
| 8 | PEP 562 lazy `__getattr__` in `elements/__init__.py` for `rcwa`/`pmm`/`bor`/`berreman`/`eme` (P2-7) | **1 d** | ~65 % of import time |
| 9 | Extract `_ensure_cupy_loaded` / `_is_cupy_array` / `_load_numba` to `backend/_optional.py` (P2-9) | **0.5 d** | 13 duplicate defs → 3, one place to test the no-accelerator path |
| 10 | Extract shared `_lens_*` types to a leaf `elements/_lens_kernels.py`; fix `raytrace/surface.py → elements.lenses` (P2-8) | **3 d** | Breaks the 4-module `_lens_*` cycle cluster |
| 11 | Move `v<N>.<N> (audit …)` blocks from `_lens_traced.py` / `carrier.py` / `_lens_real.py` to the CHANGELOG (P2-4) | **5 d** | −10 500 lines, zero behaviour change; `_lens_traced.py` 13 228 → ~8 300 |
| 12 | Split `probe_*` out of `validation/`; correct `MANIFEST.in`; add cache dirs to `.gitignore` (P2-6, P1-10) | **2 d** | sdist ~80 % smaller; `.git` stops growing per audit round |
| 13 | Config-object design for the lens family (below) | **10 d** | The structural fix for 28/29/48-param signatures |
| 14 | Docs consolidation (below) | **3 d** | 352 files → a navigable set |

### Proposed config-object design for the lens family

The three sibling entry points share ~20 parameters with identical names and semantics
(`dy`, `output_plane_distance`, `output_plane_n`, `conjugate`, `surface_model`, `carrier`,
`ray_subsample`, `output_subsample`, `clip_aperture`, `use_gpu`, …). Proposal — three frozen
dataclasses, additive and backward-compatible:

```python
@dataclass(frozen=True)
class LensGeometry:      # ~8 fields: dy, output_plane_distance, output_plane_n,
    ...                  #            conjugate, surface_model, clip_aperture, ...
@dataclass(frozen=True)
class LensNumerics:      # ~12: ray_subsample, output_subsample, remap_order,
    ...                  #      newton_fit, newton_poly_order, newton_max_iters, ...
@dataclass(frozen=True)
class LensResources:     # ~10: sag_chunk_rows, accumulator_store, scratch_dir,
    ...                  #      use_gpu, parallel_amp, *_min_free_gb, ...
```

Each entry point keeps `(E_in, prescription, wavelength, dx)` positionally and gains
`geometry=`, `numerics=`, `resources=`, with the existing 28–48 kwargs retained as a
deprecated `**kwargs` shim routed through `_deprecation.py` (which already exists, 660 lines).
Three concrete wins: (1) the config objects are **testable in isolation** — validation moves
out of a 5 719-line function into a `__post_init__`; (2) a covering array over dataclass
fields is trivial to write, which is recommendation #5; (3) the "knob silently discarded" class
of bug (`test_niche_audit_e_prepared_and_enums.py:738`) becomes a `dataclasses.fields()`
round-trip assertion.

### Docs consolidation strategy

352 docs files / 10.9 MB, 272 named `AUDIT*`, 267 in `docs/audits/`, plus 62 release notes and
a 1.1 MB CHANGELOG. Recommendation, in order:

1. **Freeze `docs/audits/`** into `docs/audits/archive/<year>/` and stop adding to it. It is a
   historical record, not documentation — nothing in it is meant to be read twice.
2. **One living document per subsystem** (`docs/subsystems/real_lens.md`,
   `.../pmm.md`, …) holding the *current* contract, invariants and known limits. Each audit
   round updates its subsystem doc and archives its own report.
3. **Split `README.md`** (180 KB): keep a ~10 KB landing page, move the cookbook to
   `docs/cookbook/` where the 10 tests that currently assert on README prose can instead be
   doctests.
4. **Retire the doc-prose tests** (10 files, P2-2 items 6–8). Replace with a docs job that runs
   the cookbook as doctests — which tests the *code*, not the *prose*.
5. **Split the CHANGELOG** the way `docs/changelogs/v4.md` already demonstrates: one file per
   major version, `CHANGELOG.md` becomes an index.

On whether the process load is helping or hurting: **the audit machinery is finding real
defects** (the `docs/audits/` tree documents genuine physics fixes, and `TESTING_STANDARDS.md`
S1–S5 is a genuinely good piece of test-design thinking). But it is being **spent on the wrong
artefacts**. Each round produces a new `test_niche_*` / `test_verify_*` file pinned to that
round's specific reproduction, a new 70 KB audit doc, a new `probe_*` directory, and a new
history block in the source. What it does *not* produce is (a) an update to the *existing*
test for the same kwarg, (b) a combination test, or (c) a deletion. The result measured here:
13 319 tests, 3.65 h of runtime, and zero tests capable of catching five defects in the
library's flagship function. The single highest-leverage process change is a rule that an audit
round **may not add a new test file** — it must strengthen the existing test for that kwarg, or
explain why one does not exist.

---

## Unverified suspicions

- **`validation/`'s ~11 GB of `.npz`/`.npy` may be partly tracked.** I confirmed 1 493 tracked
  `.npz/.npy/.csv/.json` under `validation/` and `.git` at 550 MB, but did not break that down
  by extension or measure the packed size of the binary blobs specifically. If the 8 GB of
  `.npz` were mostly tracked, `.git` would be far larger than 550 MB, so most are probably
  untracked — but I did not prove it.
- **`validation/oracles/` independence.** The four oracle modules
  (`caustic_fold_truth.py` 426 lines, `debye_oracle_v3.py` 518, `geom_spot_decenter_oracle.py`
  339, plus `conftest.py`) import only `numpy`, `scipy.special`, `json`, `sys` — **none imports
  `lumenairy`**, which is exactly right and is a genuine strength. What I could *not* verify
  in the time available is the `REAL_LENS_CHANGES.md` §3 concern — whether the
  **traced-vs-geometric** comparison still uses the same tracer on both sides. That comparison
  lives outside `validation/oracles/` and I did not locate it.
- **The 78 % doc-ref non-resolution rate is inflated** by exception names, parameter names and
  test-function citations in release notes. The real public-API breakage is the ~12 named
  identifiers in P3-2; I did not build a clean denominator.
- **`--maxfail=50` per shard** (`unit-tests.yml:145`) with 5 shards × 4 Pythons = 20 jobs
  permits up to 1 000 failures before any job aborts. I flagged this as generous but did not
  establish it has ever mattered.
- **Whether `xfail_strict = true` is actually load-bearing.** The config comment says every
  applied xfail is already `strict=True`; I did not audit the xfail sites.
- **The `elements/_lens_traced.py:31` "No import cycle" comment** — I showed the cycle exists
  and is broken only by laziness, but did not determine whether a legal import order actually
  fails today.

## Not reached

Explicitly not completed, listed rather than finished per the coordinator's instruction:

1. **`validation/run_all.py` was not executed.** It exists (3 502 bytes) and is wired into
   `validate.yml` with a 15-minute timeout on ubuntu+windows × Python 3.11–3.13. I did not run
   it — the machine was CPU-saturated throughout and the brief capped test execution.
   (Note: `run_validation.py`, named in the brief, **does not exist**; the runner is
   `validation/run_all.py`.)
2. **`ruff check lumenairy` was not run.** I read the full ruff config and the CI job but did
   not execute the linter, so I cannot report the current error count — only that a failure
   would not block CI.
3. *(completed after the deadline — see part C below; all three files pass.)*
4. **Type-hint coverage was not measured** (fraction of public functions with annotations).
   Only the mypy *whitelist* size (6/227) was established.
5. **`requirements.txt` / `requirements-gui.txt` vs. `pyproject.toml` drift** was not
   diffed — both files were located (2 721 and 1 206 bytes) but not compared.
6. **`tests/integration/`** (2 files) was inventoried but not opened; `mark.integration` is
   applied 3 times and `addopts` excludes it by default, so it is de-facto dormant, but I did
   not confirm what it contains.
7. **`examples/` (19 `.py`)** was counted but not checked for executability.
8. **`.benchmarks/` result content** was not parsed (2 JSON files, tracked).

## Checked and found sound

- **Collection is clean.** 13 319 tests collect in 137 s with **zero** collection errors and
  exactly one skip (PySide6 absent). For a 536-file, 260 KLOC test corpus with heavy optional
  dependencies, that is genuinely good hygiene.
- **`validation/oracles/` are truly independent.** All four oracle modules import only
  `numpy`/`scipy.special`/`json`/`sys` — no `lumenairy` import anywhere. Independent oracles
  are the hardest thing to get right in a physics library and these are right.
- **Shard balance is exact.** A greedy 5-way split on the committed `.test_durations` gives
  max/min = **1.000** (2 627.3 s per shard) versus 3.85× for an alphabetical split. The
  committed durations file is doing real, measurable work.
- **The fast/slow CI split is correctly constructed** — `-m "not integration and not slow"` on
  5 shards plus a separate `-m "slow and not integration"` job on 3 shards, both balanced by
  the same durations file, both with per-step timeouts, `fail-fast: false`, log artefacts on
  failure, and `FAILED`-line GitHub annotations. The *shape* of this CI is better than most.
- **`validate.yml` genuinely runs the validation suite** on every push to `main` and every PR,
  across ubuntu+windows × Python 3.11–3.13, with correct `PIPESTATUS` propagation through the
  `tee` (a subtlety most projects get wrong).
- **`publish.yml` gates PyPI on a CI re-run of the release tag** (`verify` job, 7 shards) before
  `build` and `publish` — a release cannot go out on red CI. `twine check dist/*` runs.
- **The warnings policy is well-reasoned.** `filterwarnings = ["default",
  "error::pytest.PytestReturnNotNoneWarning", "error::SyntaxWarning"]` with a written
  justification for *not* using a blanket `error` (legitimate physics diagnostics fire in the
  green suite). `xfail_strict = true` is on. Both are the right calls and both are documented.
- **`MANIFEST.in` is thorough** (modulo the stale "31 files" comment) — it ships tests,
  validation, examples, workflows and the reference docs so a PyPI tarball reproduces the
  repo, with correct `global-exclude`/`prune` for caches.
- **Dependency floors are adjudicated, not guessed.** The `numpy>=2.0` / `scipy>=1.13`
  rationale (first SciPy line built against the numpy 2.0 ABI) is correct and the PEP 508
  environment markers on `pyfftw`, `zarr` and `jax` correctly solve real resolver failures on
  Python 3.10/3.11. This is unusually careful packaging work.
- **`.gitignore` correctly tracks `.test_durations`** rather than ignoring it — necessary for
  pytest-split, and a mistake many projects make in the other direction.
- **No `.pyc` files are tracked** (`git ls-files | grep -c '\.pyc$'` → 0) despite
  `__pycache__` directories for three CPython versions existing in the tree.
- **Duplication is low.** Only 9 exact-duplicate function clusters (~573 redundant lines) in a
  221 KLOC library, and the largest is unavoidable Qt boilerplate. The suspected `t = -z/N`
  exit-vertex duplication appears exactly twice. For a codebase this size, that is a good result.
- **`lumenairy.__all__` is a single literal list** (708 symbols, no `+=` or `.extend`), which
  makes the public surface statically analysable — and `test_public_api.py` does analyse it.
- **`tests/conftest.py` has real autouse leak guards** (`_module_flag_leak_guard`,
  `_module_glass_registry_guard`) plus an `fft_state_ctx` fixture and a `process_state_dump`.
  The global-state problem in P2-5 is partially, deliberately addressed — it just needs to
  cover all 20 knobs instead of a subset.
- **Part C — representative real-lens slice.** `-k real_lens` selects **135 tests across 23
  files (436 s)**. Ran the smallest files:

  | file | lines | result | test time | wall (incl. import) |
  |---|---|---|---|---|
  | `tests/unit/test_v5_4_6_wave4_polarization.py` | 92 | **4 passed**, 2 warnings | 7.17 s | 59 s |
  | `tests/unit/test_public_api.py` | 104 | **712 passed** | 1.90 s | 146 s |
  | `tests/unit/test_folded_design_guard.py` | 136 | **9 passed**, 3 warnings | 141.18 s | 535 s |

  All three pass. The suite is **green today**; its problem is not breakage but discrimination.
  Two incidental observations from the run: (a) `test_folded_design_guard.py` is a 136-line,
  9-test file that takes **141 s** of test time — even allowing ~4× for the loaded machine,
  ~35 s nominal for one small file sits in the "fast" lane and reinforces P2-1; (b) the
  warnings surfaced are a credit to the library — `polarization.py:466` and the
  `apply_real_lens` aperture guard emit detailed, actionable truncation warnings naming the
  offending surfaces, the shortfall in mm, and the fix.
