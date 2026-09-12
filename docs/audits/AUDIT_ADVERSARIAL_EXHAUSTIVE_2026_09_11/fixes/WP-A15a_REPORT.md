# WP-A15a — tests, CI, packaging, suite composition and hygiene

Branch `audit-fixes-2026-09`.  All measurements 2026-09-12 on the dev workstation:
CPython 3.14.6, numpy 2.4.6, scipy 1.17.1, numba 0.65.1, jax 0.10.1, pyfftw 0.15.1,
ruff 0.16.0, mypy 1.x (installed for this work package).  Every python invocation ran
with `OPENBLAS_NUM_THREADS=1`.  No git write commands were run; no file under
`lumenairy/` was edited.

---

## 1. Summary

| Finding | Status | Files:lines | Tests (path::name) | Oracle | Measured before → after |
|---|---|---|---|---|---|
| **V3** (P1) kwarg space uncovered in combination | **fixed** | `tests/unit/test_audit2609_a15a_lens_covering_array.py` (new, 673 lines) | `::test_analytic_covering_array_is_finite_and_loses_no_energy` (12 ids), `::test_traced_covering_array_is_finite_and_loses_no_energy` (15), `::test_every_declared_exclusion_is_really_refused` (14), `::test_analytic_kwargs_at_their_default_reproduce_the_bare_call`, `::test_traced_kwargs_at_their_default_reproduce_the_bare_call`, 2 counter-pins | energy (a passive lens cannot gain power), finiteness, bit-identity | combination coverage 0 rows → **27 pairwise rows over 36 physics kwargs**; worst `P_out/P_in` **0.996170598** / **0.996162541**, no arm gains (largest excess −3.83e-03); 45 collected ids in **3.34 s** |
| **V4** (P1) ruff advisory, `ui/` unlinted | **fixed** | `pyproject.toml:343-353, 366-455`; `.github/workflows/unit-tests.yml:213-250` | `test_audit2609_a15a_packaging.py::test_the_lint_job_is_blocking`, `::test_ruff_lints_the_ui_package` | `ruff check` | `continue-on-error: true` → **false**; lint scope `lumenairy/ tests/unit/` (ui excluded) → `lumenairy/ tests/` (**ui included, ~30 modules**); findings 13 + 331 + 37 → **0** |
| **V4** (P1) mypy covers 2.6 % behind a stale comment | **fixed** | `pyproject.toml:456-488, 495-521` | `::test_the_mypy_whitelist_only_grows` | `mypy --strict` | 6 declared paths / **11 source files** → 17 paths / **22 files**; `Success: no issues found in 22 source files`; the "no CI wiring at v5.0.1" comment (43 minor versions stale) replaced |
| **V4** (P1) 3.14 in no CI leg; false accelerator comment | **fixed** | `pyproject.toml:40-48`; `unit-tests.yml:23-34`; `publish.yml:54-60` | `::test_every_declared_python_classifier_is_in_the_unit_test_matrix` | matrix-vs-classifier | matrix `3.10-3.13` → **`3.10-3.14`** (release verify too); classifier restored; comment replaced with the measurement (numba 0.65.1 / jax 0.10.1 / pyfftw 0.15.1 all present on 3.14.6) |
| **V4** (P1) declared `jax` floor excludes the local jax | **decided, gated in CI** | `pyproject.toml:151-167`; `unit-tests.yml` "Assert the resolved jax satisfies the declared floor" | — (a workflow step, deliberately not a test — see §2.4) | `packaging.Requirement` vs `jax.__version__` | floor **kept at >= 0.11** (the version every jax gate is measured against); CI now FAILS if the resolved jax violates it; the local 0.10.1 recorded as knowingly below-floor |
| **V4** (P1) installed metadata 2 releases behind | **fixed** | `tests/unit/test_public_api.py` | `::test_installed_metadata_version_matches_source_version` | `importlib.metadata` | `importlib.metadata.version('lumenairy')` **5.43.0 → 5.45.1** (`__version__` 5.45.1) after `pip install -e . --no-deps` |
| **V4** (P1) `test_public_api.py` = 5 % of the suite in tautologies | **fixed** | `tests/unit/test_public_api.py` | whole file | collected-id count | **726 ids / 0.65 s → 7 ids / 0.12 s**, same 708 names checked |
| **V4** (P1-10) `.gitignore` / tracked benchmark JSONs | **fixed** (ignores) + **requested** (untracking) | `.gitignore` | `::test_gitignore_has_no_blanket_binary_rules`, `::test_gitignore_covers_the_tool_caches` | `git check-ignore`, `git status` | blanket `*.png/*.jpg/*.pdf/*.fits/*.dat/*.log` → directory-scoped; `.benchmarks/ .mypy_cache/ .ruff_cache/` added; `validation/repro_*/` added (covers the 89 MB `repro_traced_carrier_122`); untracked-file count **27 → 26** (no new noise) |
| **V5** (P2) "fast (<30 s)" contract wrong by ~440x | **fixed** | `pyproject.toml:227-247, 265-278` | — (comment) | `.test_durations` | text now states the measured **13 061 s = 3.63 h / 14 199 ids**; marker inventory documented (`slow` 54→65 files; `unit`/`regression`/`bench` 0 applications, kept as declared-but-unused with the reason) |
| **V5** (P2) slow lane mis-marked at the 2 min/file bar | **partially fixed** (11 of 20 files; 9 deferred by ownership — §5) | 11 test files + `unit-tests.yml` + `publish.yml` | `-m "slow"` collection | `.test_durations` | slow lane **277 → 551 ids, 3 573.9 s → 5 915.0 s**; fast lane **9 487.1 s → 7 144.0 s**, i.e. **31.6 → 23.8 min/shard**; slow shards **3 → 5** (32.9 → 19.7 min/shard) |
| **V5** (P2-3) `.test_durations` staleness unwatched | **gate added — and it is RED** | `tests/unit/test_audit2609_a15a_durations_staleness.py` (new) | 4 ids | subprocess `--collect-only` vs the committed JSON | **2 117 of 14 076 collected ids (15.04 %) carry no timing**, against the 2 % bar.  See §2.6 — this is the gate firing correctly on a real, campaign-wide staleness; the remedy is one regeneration run at campaign exit |
| **V5** (S1) wall-clock / speedup assertions | **8 of 11 converted**, 3 not mine | 6 test files | see §2.7 | operation counts | e.g. GS `elapsed_ms < 5000` → **exactly `2·n_iter+1` transforms**; `speedup >= 1.2` → **0 sag evaluations on a pure sphere** |
| **V5** (v) tests asserting on `.md` prose | **5 retired** | `tests/unit/test_v4_16_2_agent_d.py` | — | — | file 13 → 8 ids; CHANGELOG fabrication walkers kept (release gates) |
| **Except budget** | **fixed** | `tests/unit/test_audit_except_budget.py` | 4 ids | AST/regex census | budget **48 (scalar, already 3 short at the audit base)** → **53, as a per-file census of 27 modules**, every site justified; 3 flagged as narrowing requests |
| **Isolation** `test_propagation_asm_cache_lock_still_paired` | **fixed** | `tests/unit/test_v4_16_1_agent_c.py` | `::test_propagation_asm_cache_lock_still_paired` | behavioural lock capture | source-text proxy (`inspect.getsource`, order-dependent) → **lock actually entered**; fail-before demonstrated |
| **Isolation** `WinError 6/50` in subprocess tests | **fixed** | 15 test files (25 sites) + 2 `scripts/` tools | `test_audit2609_a15a_packaging.py::test_every_subprocess_spawn_names_all_three_stdio_streams` | direct probe under pytest capture | 28 partial spawns in 17 files → **3** (all in files this WP does not own); the reproducing batch went **1 failed / 31 passed → 32 passed, three runs in a row** |
| **H5** (WP-A14's request) `threadpoolctl` undeclared | **fixed** | `pyproject.toml:81-103`; `requirements.txt:26-33` | `::test_threadpoolctl_is_a_hard_dependency_in_both_places` | — | absent from both files → `threadpoolctl>=3.1` in both |
| **V7** (P3) `--maxfail=50` × 20 jobs | **fixed** | `unit-tests.yml`, `publish.yml` | — | — | 50 → **10** per shard (fast matrix budget 1 250 → 250 failures before any abort) |
| **V7** (P3-4) 12 probe scratch files in `scripts/` | **fixed** | `scripts/` → `validation/probe_scripts_legacy/` | `::test_scripts_holds_only_the_maintained_tools` | — | `scripts/` 15 `.py` → **4** (the maintained tools only); 11 probes moved, their `_ROOT` arithmetic repointed, README added |

**Not reproducible / corrections to the brief.**  The brief says "the 12 probe scratch files"
in `scripts/`; there are **11** (`_d5_byteid`, `_d5_probe`, `_d5_probe2..6`, `_d5_spl`,
`_g8_c15lad`, `_g8_failbefore`, `_g8_probe`) beside the 4 real tools — 15 `.py` total, which
matches the TESTS-ARCH count of "15 `.py`" and its "4 real tools".

---

## 2. Per deliverable

### 2.1 V3 — combination coverage for the lens family

**New file** `tests/unit/test_audit2609_a15a_lens_covering_array.py` (45 collected ids,
**3.34 s**).  Nothing else was touched for this item.

**Signatures read from the source, not the audit.**  Both entry points are now
**keyword-only after `E_in`** (WP-A2 `b97c0b6e`, WP-A3 `fix(lens-traced): WP-A3`):
`apply_real_lens(E_in, *, prescription, wavelength, dx, ...)` — 28 parameters;
`apply_real_lens_traced(E_in, *, ...)` — 48.  Worth flagging on its own: every positional
call site outside the repo is now a `TypeError`.

**Fixture** (`lens_covering_array_fixture`, module-level and importable — WP-A16 reuses it):
AC254-ish cemented doublet with a **curved rear** (R1 33.3 mm, R2 −22.28 mm,
**R3 −291.07 mm**, N-BAF10/N-SF6HT, 6 mm aperture) illuminated by a Gaussian-apodised
**diverging spherical wave** from −120 mm.  N = 64, dx = 1.125e-4 m (1.2 aperture diameters
on the grid).  This is the geometry the audit says the suite never tests — the only
`seidel_correction` fixture in the repo has R2 = ∞, on which the exit-vertex class is
invisible, and essentially every call site is collimated.

**Arrays.**  Factors are *groups* of kwargs, because the constraint lattice is real:

| | factors | kwargs exercised | rows | refused pairs (measured) |
|---|---|---|---|---|
| `apply_real_lens` | 10 | **16** | **12** | 11 of 141 |
| `apply_real_lens_traced` | 13 | **20** | **15** | 3 of 308 |

The exclusions were found by **sweeping every factor-level pair** and recording which raise,
not by reading the docs.  The array found the first one on its second row:

* `slant_correction` × `seidel_correction` — *"the two flags replace the SAME per-surface
  coefficient, so stacking them double-counts the facet obliquity (measured 173.5 → 1488.6 nm
  rms exit OPD …)"*;
* `surface_model='displaced'` × {`fresnel`, `absorption`, `slant_correction`,
  `seidel_correction`, `surface_frame`, `carrier`, `wave_propagator='rs'`};
* `surface_model='tangent_facet_remap'` × {`slant_correction`, `surface_frame`,
  `screen_obliquity=True`, non-ASM `wave_propagator`};
* `amplitude_model='ray_density'` × `newton_amp_mask_rel` (all three ray-density levels).

`test_every_declared_exclusion_is_really_refused` (14 ids) **re-measures all 14 on every run**
and also asserts the refusal names the function (CONVENTIONS §2), so the exclusion table
cannot quietly grow into a hiding place.

**Invariants asserted** — finiteness, output shape, `P_out <= P_in·(1+1e-6)`, `P_out > 0`.
Energy is one-sided on purpose: loss is combination-dependent (0.354847608 … 0.996170598 over
the 27 arms), gain is not.  Bar derivation: measured largest excess over 1 is **−3.83e-03**
(no arm gains); the float64 accumulation floor for a 64×64 reduction is ~1e-13 relative; the
smallest gain a real defect produces (double-applied Fresnel, dropped obliquity cosine,
un-normalised ray-density Jacobian) is ≳1e-2.  1e-6 sits seven decades above the floor and
four below the signal.

**"Knob silently discarded" detector** — 17 analytic and 28 traced kwargs, each passed at its
own signature default, must reproduce the omitted call **bit for bit**.  All 45 do today.
`min_compared=` guards keep the loop from silently skipping.

**Fail-before demonstrations** (in-process, recorded here):

```
FAIL-BEFORE energy   -> output power 5.088793311304e-06 EXCEEDS input power 5.088742423879e-06 by 1.000e-05  (1e-5 gain injected, 10x the bar)
FAIL-BEFORE finite   -> output carries 1 non-finite samples out of 4096
FAIL-BEFORE identity -> fresnel=True -> max |diff| = 8.1255e-01   (pretending a default is honoured)
```

**Runtime budget.**  Warm analytic call 1–7 ms, warm traced call 20–45 ms; the slowest arm is
`displaced_obliquity='pointwise'` at 0.37 s.  `ray_subsample=1` is pinned in the traced base
call because at N = 64 the shipped default of 8 leaves 5 coarse samples across the aperture and
the undersample guard **correctly refuses** (`... gives only 5.0 coarse samples across the
6.00-mm aperture (threshold 32)`).

### 2.2 V4 — ruff

`ruff check lumenairy` on arrival: **13 findings** — I001 ×9 (`lumenairy/__init__.py`,
`analysis/__init__.py`, `analysis/image_plane_wfe.py`, `elements/__init__.py`,
`propagators/fft_infra.py`, `raytrace/__init__.py`, `elements/rcwa/oned.py`, `rcwa/stack.py`,
`rcwa/twod.py`), F541 ×2 (`elements/_lens_real.py:2634-2635`), F401 ×2
(`rcwa/oned.py:26`, `rcwa/twod.py:24`).  `lumenairy/ui` (excluded until now): **331** findings
in exactly 7 rules — F401 121, I001 116, E701 37, E702 37, F811 10, F841 7, F541 3.
`tests/unit`: **37** — I001 34, F841 3.

No library module was edited.  Disposition, **every ignore with its reason**:

| ignore | rules | class | reason |
|---|---|---|---|
| `lumenairy/__init__.py` | +`I001` | convention | this library interleaves imports with the multi-line comments that explain their placement; `E402` is already ignored project-wide for the same reason.  MEASURED: `ruff --fix` on `raytrace/__init__.py` relocates the whole commented `from .exit_vertex import (...)` block, deleting the commentary |
| `lumenairy/analysis/__init__.py` | +`I001` | convention | as above |
| `lumenairy/elements/__init__.py` | +`I001` | convention | as above |
| `lumenairy/raytrace/__init__.py` | +`I001` | convention | as above |
| `lumenairy/propagators/fft_infra.py` | `I001` | convention | the deliberately-late `from .._knobs import register_knob` and `from ..memory import available_cpus`, each under its own explanatory comment |
| `lumenairy/analysis/image_plane_wfe.py` | `I001` | convention | same shape |
| `tests/**/*.py` | +`I001`, `E701`, `E702` | convention | tests import inside test bodies to isolate optional deps; `a = 1; b = 2` in a parametrise table and `if x: return` in a helper are not defects.  Both still fire in library code |
| `lumenairy/elements/rcwa/oned.py` | `F401`, `I001` | **TODO(audit-2609)** | imports `_grazing_safe_wavelength` and no longer calls it (WP-A14 routed both entry points through `_grazing_safe_wavelength_pair`).  **Request to the RCWA WP**: drop the import or re-export it deliberately |
| `lumenairy/elements/rcwa/twod.py` | `F401`, `I001` | **TODO(audit-2609)** | same |
| `lumenairy/elements/rcwa/stack.py` | `I001` | **TODO(audit-2609)** | import block introduced un-sorted by WP-A14; re-sort and delete this line |
| `lumenairy/elements/_lens_real.py` | `F541` | **TODO(audit-2609)** | `:2634-2635`, two `f"…"` continuation lines in the mirror-guard warning carry no placeholders.  **Request to the analytic-lens WP**: drop the `f` prefix |
| `lumenairy/ui/**/*.py` | the 7 measured rules | scoped-in | replaces a wholesale `extend-exclude`.  Now ENFORCED in ~30 previously-unlinted GUI modules: **F821 undefined name** (the exact class that shipped `carrier_field.py:469` this month), F632, F502/F522, E711/E712/E713/E714, E722, E731.  A test pins the list so it can shrink but not grow |

Three genuine findings in *test* files were fixed rather than ignored (dead stores):
`test_audit2609_a7_misc.py:74` and `test_audit2609_a7_opd_unwrap.py:251` (`rng` assigned,
never used) and `test_audit2609_verify_a7.py:573` (`jax = pytest.importorskip('jax')` → bare
call).

`ruff check lumenairy/ tests/` → **All checks passed**.  Job flipped to
`continue-on-error: false` and its scope widened from `tests/unit/` to `tests/`.

### 2.3 V4 — mypy

Method: one `mypy --strict --follow-imports=silent --ignore-missing-imports
--python-version 3.12` pass over **all 170 non-`ui` candidate modules**, attributing every
error to the file it was reported in.  150 modules have ≥ 1 error; **20 are clean**.

Added (the 11 clean non-`__init__` modules, smallest first):

| lines | module |
|---|---|
| 54 | `lumenairy/elements/coronagraph.py` |
| 68 | `lumenairy/raytrace/core.py` |
| 69 | `lumenairy/analysis/core.py` |
| 106 | `lumenairy/io/prescriptions.py` |
| 190 | `lumenairy/raytrace/layout.py` |
| 227 | `lumenairy/raytrace/bundles.py` |
| 230 | `lumenairy/algebra/from_prescription.py` |
| 244 | `lumenairy/propagators/result.py` |
| 292 | `lumenairy/_cache_registry.py` |
| 557 | `lumenairy/analysis/strehl.py` |
| 893 | `lumenairy/propagators/asymptotic_modes.py` |

**Deliberately NOT added yet** — the 9 clean `__init__.py` hubs (`_math`, `sources`,
`elements/pmm`, `elements/rcwa`, `elements/eme`, `elements/bor`, `io`, `analysis`, and the
root `lumenairy/__init__.py`).  WP-A15b is rewriting them for lazy loading this round, and a
PEP 562 `__getattr__` needs its own annotations before it is strict-clean.  They are clean
*today*; adding them now would hand A15b a gate failure for doing its job.

`mypy` (driven by the pyproject whitelist) → **`Success: no issues found in 22 source files`**
(was 11 files).  The stale "No CI wiring at v5.0.1; `unit-tests.yml` keeps mypy unwired until
v5.1's cleanup lands" comment is replaced with the true state (a merge-blocking job since
v5.2) plus the exact command that regenerates the whitelist.

### 2.4 V4 — Python 3.14, the jax floor, metadata, `test_public_api`

**3.14.**  `unit-tests.yml` matrix `['3.10','3.11','3.12','3.13']` →
`['3.10','3.11','3.12','3.13','3.14']`, and the same in `publish.yml`'s release verify (a
classifier is a promise; the release gate is what backs it).  The classifier
`Programming Language :: Python :: 3.14` is restored and the comment that justified dropping
it — *"3.14 is too new for the optional accelerator dependencies (numba, jax, pyfftw) to have
published wheels against"* — is replaced by the measurement that falsifies it (numba 0.65.1,
jax 0.10.1, pyfftw 0.15.1, all installed on CPython 3.14.6).  Pinned by
`test_every_declared_python_classifier_is_in_the_unit_test_matrix`, which is two-sided: a
classifier no leg runs fails it.

**jax floor — the decision.**  Kept at `jax>=0.11.0; python_version >= "3.12"`.  It is the
version every jax-guarded gate in the suite is measured against; lowering it would declare
support for a line no leg exercises.  What changes is that the `jax-unit` leg now parses the
declared requirement out of `pyproject.toml` and fails if the resolved `jax.__version__` does
not satisfy it, so the two can no longer drift silently (`dep-drift.yml` cannot catch this —
it is explicitly non-blocking).  This is a **workflow step, not a test**, on purpose: the dev
box runs jax 0.10.1, so a test form would be red locally for an environment reason, which is
precisely the S3 shape the standards forbid.  The local box is recorded in the pyproject
comment as knowingly below-floor, with jax results from it advisory; raising it is a
follow-up (§6) because upgrading jax mid-campaign changes numerics for every concurrent WP.

**Installed metadata.**  `importlib.metadata.version('lumenairy')` read **5.43.0** against
`__version__` **5.45.1**.  Ran `pip install -e . --no-deps` (deliberately `--no-deps`, so no
dependency in the shared environment moved — in particular `threadpoolctl` was **not**
installed, which would have changed `rcwa/_core.py`'s BLAS-cap behaviour under other agents
mid-run).  Both now read **5.45.1**.  Pinned skip-free by
`test_public_api.py::test_installed_metadata_version_matches_source_version`; a missing
distribution raises with the exact remedy rather than skipping.

**`test_public_api.py`.**  `@pytest.mark.parametrize('name', la.__all__)` → one loop that
reports every failing name at once.  **726 collected ids / 0.65 s → 7 ids / 0.12 s**, the same
708 names checked, one collected id per property.  Two properties added
(`__all__` entries are identifier strings; installed-metadata equality) and the existing
phantom counter-pin strengthened to also assert the *aggregate* loop sees the injection — the
half the 726-id parametrisation could not express.  Collected ids kept stable in meaning:
`test_every_all_entry_is_resolvable` is the same name, minus its `[param]` suffix.

### 2.5 V4/V7 — `.gitignore`, `MANIFEST.in`, `scripts/`

**`.gitignore`.**  Added `.mypy_cache/`, `.ruff_cache/`, `.benchmarks/` (all three exist in a
tree as soon as the tool runs) and a note that `.test_durations` stays tracked deliberately.
Replaced the blanket `*.png` / `*.jpg` / `*.pdf` / `*.fits` / `*.dat` / `*.log` with
directory-scoped rules, derived from `git ls-files --others --ignored`: 113 files under
`validation/repro_traced_carrier_121`, 26 under `docs/audits`, 22 under
`validation/real_lens_opd`, 9 under `examples/output`, plus `importtime.log`.  Added
`validation/repro_*/` (the 2026-09-12 addendum — it covers the 89 MB untracked
`validation/repro_traced_carrier_122`; an ignore does not untrack, so
`repro_traced_carrier_121`'s 430 tracked files are unaffected) and `/scratchpad/`.  Verified
with `git check-ignore -v`: `tests/unit/fixture.png` is now **addable without `-f`**, which
was the point.  Untracked-file count 27 → 26, i.e. **no new noise for the other agents**.

**Benchmark JSONs to untrack** (no git writes from me — for the orchestrator):

```
git rm --cached .benchmarks/Windows-CPython-3.14-64bit/0001_v4_11_2.json
git rm --cached .benchmarks/Windows-CPython-3.14-64bit/0002_v4_11_2_zernike.json
```

Those are the only two tracked files under `.benchmarks/` (`git ls-files .benchmarks/`), and
they pin one Windows/3.14 machine's timings into version control.

**`MANIFEST.in`.**  The "Validation suite — 31 files with ~370 physics-fidelity tests" comment
(a 4.4.0 count, 28× off) is replaced by the measurement: **876 `.py` across 70
sub-directories, of which 624 are under `validation/probe_*` and 184 under
`validation/repro_*`**; the 37 `t_*.py`/`test_*.py` entry points are the part of the old claim
that survives.  `prune validation/probe_*` + `prune validation/repro_*` (plus
`recursive-exclude` for the non-`.py` payload) keep 808 of those 876 files out of the sdist.

**Probe files moved** — `scripts/` → `validation/probe_scripts_legacy/` (11 files, plain
`mv`; the orchestrator stages the rename):

```
_d5_byteid.py  _d5_probe.py  _d5_probe2.py  _d5_probe3.py  _d5_probe4.py
_d5_probe5.py  _d5_probe6.py  _d5_spl.py    _g8_c15lad.py  _g8_failbefore.py  _g8_probe.py
```

All eleven computed the repo root as `dirname(dirname(__file__))`, which the move breaks, so
each got a two-line repoint (`_HERE` + `_ROOT` one level further up) and `_d5_spl.py`'s
`sys.path.insert(1, os.path.join(_ROOT, 'scripts'))` became `sys.path.insert(1, _HERE)`.
Verified: all 11 compile and resolve `_ROOT` to the repository root.  A README records what
they are and that `scripts/` now holds only the four maintained tools
(`check_dep_metadata.py`, `check_source_line_citations.py`, `verify_changelog_closures.py`,
`stamp_changelog.py`).  `validation/probe_*` is pruned from the sdist, so they no longer ship.

### 2.6 V5 — suite composition

**Slow lane.**  27 files sit at or over the **2 min/file** bar on the committed
`.test_durations` restricted to currently-collected ids.  Five already carry partial
`@pytest.mark.slow` on their heavy tests (fast residue 5–37 s each) and two are already
module-level.  Of the 20 fully-unmarked files:

*Marked with `pytestmark = pytest.mark.slow` (11 files, 2 341.0 s moved):*

| file | measured s | tests |
|---|---|---|
| `test_fix_bor_multilayer_guards.py` | 520.6 | 73 |
| `test_v5_6_rcwa_convergence.py` | 284.5 | 23 |
| `test_niche_d5_dx_flatness_gate.py` | 194.0 | 13 |
| `test_niche_d7_decentred_fit.py` | 191.5 | 38 |
| `test_niche_d3_guards.py` | 187.5 | 41 |
| `test_v5_14_0_pmm2d_cell.py` | 187.2 | 8 |
| `test_fga.py` | 178.8 | 27 |
| `test_fix_pmm2d_mortar_round3.py` | 172.2 | 14 |
| `test_lens_gbd.py` | 144.5 | 5 |
| `test_hammer_h7_gbd_diverging.py` | 141.5 | 9 |
| `test_bor_sem.py` | 138.7 | 23 |

`test_fga.py` carried a comment explicitly declining the marker.  Both of its premises had
moved and the comment is rewritten in place rather than left contradicting the code: the slow
job is no longer near its cap (see below), and its "this file measures 1791.7 s (~30 min)"
reading is **stale by 10×** — a pre-v5.31 unpinned-BLAS artefact.  **Re-measured: the same 27
ids total 178.8 s**, heaviest `test_fga_beats_gbd_at_spherical_aberration_caustic` at 40.1 s.

*Not marked — ownership (5 files, 1 654.4 s), listed for the orchestrator:*
`test_audit_lens_models_2026_07.py` 641.0 s (VERIFY-A4), `test_niche_d2_chain_multi.py`
338.8 s (VERIFY-A6 carrier), `test_audit_misc.py` 320.6 s (verifiers),
`test_pmm2d_staggered_mortar.py` 229.3 s (VERIFY-A13), `test_niche_d6_exact_tilted_leg.py`
124.7 s (VERIFY-A6 carrier).

*Not marked — a coverage side effect that needs a coordinated change (4 files, 619.7 s):*
`test_niche_r3_gbd_mem_lstsq.py` 174.8 s, `test_v5_11_0_rcwa_fff_nv_2d.py` 172.6 s (fast
residue), `test_gbd_feature_complete.py` 148.9 s, `test_bor_sem_jax.py` 123.7 s.  All four are
**jax-guarded**, and the `jax-unit` CI leg selects `-m "not integration and not slow"` — so
marking them slow would silently remove them from the only leg that runs jax at all.  Moving
them needs the jax leg to opt the slow files in (a one-line `-m` change plus a cap/shard
review).  Designed in §6.

**Lane arithmetic** (committed `.test_durations`, today's collection):

| | ids | timed s | per shard |
|---|---|---|---|
| fast, before | 14 389 | 9 487.1 | **31.6 min** on 5 (38.3 with untimed at the mean) |
| fast, after | 13 648 | 7 144.0 | **23.8 min** on 5 (31.4 with untimed) |
| slow, before | 277 | 3 573.9 | 19.9 min on 3 |
| slow, after | 551 | 5 915.0 | **32.9 min on 3 — over the 30-min step cap**; **19.7 min on 5** |

So the slow gate goes **3 → 5 shards**; no timeout was raised.  `publish.yml` gains a
`verify-slow` job (5 shards, 3.12, single-threaded BLAS) that `build` and `publish` now depend
on: the release gate verified the *fast lane only*, so every `slow`-marked test was outside
tag verification entirely — a hole that predates this change and that the re-marking widens.

**`.test_durations` staleness gate** — `tests/unit/test_audit2609_a15a_durations_staleness.py`
(4 ids, 26.4 s: one `--collect-only` in a subprocess with BLAS pinned and all three stdio
streams named).  Three of the four pass.  The fourth is **RED, by design**:

```
.test_durations is STALE: 2117 of 14076 collected ids (15.04%) carry no timing, above the 2% bar.
Worst files: test_v4_14_2_dispatcher_pin_zero_plus_zeroj.py (232), test_lens_memory_levers.py (144),
test_audit2609_a11_polar_sources_infra.py (103), test_audit2609_a6_verify_carrier.py (85), ...
```

**This is the gate working, not a defect I introduced.**  The staleness is the campaign's own:
~2 100 of the untimed ids are tests the fix and verify work packages added over the last two
weeks.  It cannot be fixed *now* and should not be: a regeneration run is only valid once the
last WP has landed, and regenerating mid-campaign would be stale again within the hour.  The
remedy is one run at campaign exit, and the test's failure message carries the exact
procedure (the same one `unit-tests.yml` documents — serial, idle machine,
`OMP/OPENBLAS/MKL_NUM_THREADS=1`, both lanes, merged).  **Orchestrator action, §5 item 1.**
The bar itself is derived, not picked: at the measured ~1.0 s mean, 2 % of ~14 000 ids is
~280 s of unmodelled work — about 5 % of one fast shard, inside the headroom the shard count
is chosen for — while 20 % is a whole shard, i.e. a guaranteed cap hit.

**"fast (<30 s)" contract text** corrected in `pyproject.toml`, with the measured
13 061 s / 3.63 h and the two-lane split spelled out, and the marker block rewritten to record
what is actually applied (`slow` 54 → 65 files; `unit` / `regression` / `bench` 0 applications
each, kept as declared-but-unused because `--strict-markers` would turn a future
`@pytest.mark.unit` into a collection error rather than a silent no-op).

### 2.7 V5 — wall-clock / speedup assertions converted

An AST sweep (functions that read a clock, assertions on names derived from that read) plus a
grep over `assert` lines found **11 live sites** in 8 files — not 24; the audit's regex count
included docstring tables and comments, several of which record previously-retired pins.

| # | site (before) | after | oracle |
|---|---|---|---|
| 1 | `test_audit_analysis.py::test_3a_gs_speedup_smoke` — `elapsed_ms < 5000.0` on a call measured at 600–900 ms (6–8× margin) | `::test_3a_gs_does_two_transforms_per_iteration_and_no_more` | wraps `phase_retrieval._fft2/._ifft2`: **exactly `n_iter+1` forward and `n_iter` inverse**, measured at n_iter = 10/25/50 → 11/10, 26/25, 51/50 |
| 2 | `test_audit_analysis.py::test_3b_shack_hartmann_speedup_smoke` — `elapsed_ms < 5000.0` on "a few hundred ms" | `::test_3b_shack_hartmann_output_is_sized_by_the_lenslet_grid` — **RETIRED, not converted** | the per-lenslet gather is still a `for iy: for ix:` loop in `detector.py`, so there is no operation-count oracle that would not just restate the loop.  Replaced by the contract the call owes: all five returned arrays are `(n_lenslets, n_lenslets)`, and a uniform illumination yields measured (non-NaN) slopes |
| 3–4 | `test_audit_raytrace.py::test_doublet_trace_is_faster_than_the_legacy_newton_path` — `fast < 20 ms` **and** `speedup >= 1.2` (measured 1.78 / 1.13 / 1.59 / 2.65 over four runs — one outright failure) | `::test_pure_spherical_doublet_never_evaluates_the_surface_sag` + `::test_a_non_spherical_surface_does_evaluate_the_sag` | wraps `intersection._surface_sag_xy`: pure-spherical doublet → **0 evaluations**, conic doublet → **3 (one per surface)**, Newton fallback bound → 10/surface.  The class's own timing helpers (`_time_interleaved`, `_make_legacy_tracer`) are deleted with it |
| 5–6 | `test_audit_raytrace.py::test_warm_call_is_cheap_and_much_faster_than_the_first` — `warm < 20 ms` **and** `first/warm >= 20x` (docstring tabulates 391–652× idle vs 102–155× loaded; logged twice as a timing flake) | `::test_warm_calls_are_served_from_the_cache_not_recompiled` | `_TRACE_JAX_CACHE` holds **exactly 1 entry after each of 20 identical calls** and serves the **same object** (`is`) every time.  No counter-pin added — `…_AuxKeying` in the same file already pins the miss direction on glass, wavelength and radius |
| 7 | `test_perf_v4_12_0_through_focus.py::test_warm_call_at_least_10x_faster_than_first` — `speedup >= 10.0` | `::test_warm_calls_are_served_from_the_cache_not_recompiled` | `_THROUGH_FOCUS_SCAN_JAX_CACHE`: 1 entry and the same object across 8 calls; the miss direction is already pinned by `test_cache_misses_on_different_bandlimit` in the same file |
| 8 | `test_v5_3_multi_field_merit_jit.py::test_numba_jit_speedup_on_large_grid` — `t_jit < 8 s` against a measured 0.58–1.36 s | `::test_the_jit_kernel_pair_is_built_once_not_per_call` | `_merit_jit._NUMBA_KERNELS['multi_field']` is populated once and keeps **the same identity** across 8 field angles — the per-call recompile the ceiling was proxying |
| 9 | `test_niche_audit_w9_overlap_exact.py::test_w9d_performance_envelope` — `dt < 4.0` against a measured 51.8–60.1 ms (≈70× margin); its own docstring says the intent is "to catch an accidental O(K²)-in-the-predicate regression, not to race" | `::test_w9d_the_cheap_prefilter_decides_every_disjoint_pair` + `::test_w9d_the_predicate_counter_sees_a_pair_that_needs_it` | wraps `rcwa._core._shapes_overlap`: a 1024-shape lattice reaches the exact predicate **0 times** (measured at k = 16 and k = 32 for all three shape kinds) out of 523 776 pairs, while a genuinely overlapping pair reaches it **1 time** |

**Not converted — not this work package's files** (listed for the orchestrator to assign):

| site | assertion | suggested conversion |
|---|---|---|
| `tests/unit/test_audit_propagation.py:2678` | `assert t_fused < 60.0` | count the transforms the fused path performs vs the per-plane path (the file's own `test_fused_path_does_the_fused_work` name says the claim is about *work*) |
| `tests/unit/test_audit_propagation.py:3787` | `assert speedup >= 5.0` | same shape as #1: count the FFTs / kernel invocations on each leg |
| `tests/unit/test_audit_optimize.py:905` | `assert dt < 180.0` | count merit evaluations (the optimiser's own call counter), not seconds |

`tests/unit/test_ci_kernel_consistency.py:492` (`assert elapsed < 20.0`) is left as-is
deliberately: that file's whole subject is a cross-arm decision census, and the 20 s budget is
the *stated promise of the file* rather than a proxy for an operation count.  Converting it
means re-deriving what the census should cost in decisions, which belongs with whoever owns
the census.  Recorded in §6.

### 2.8 V5 — prose tests retired

`tests/unit/test_v4_16_2_agent_d.py`, 13 → 8 collected ids.  Deleted:
`test_readme_does_not_cite_refractiveindex_as_required`,
`test_readme_pip_install_command_uses_extras_or_omits_refractiveindex`,
`test_roadmap_claims_correct_meta_pin_count`, `test_roadmap_enumerates_v10_v11`,
`test_changelog_high_na_transfer_jax_uses_runtimewarning`.  A block at the top of the file
records what was deleted and why, so the deletion is not mistaken for a loss of coverage: the
README-vs-reality drift they were written for is pinned *structurally* by the kept
`requirements.txt` tests and by `test_v5_2_3_dep_drift_check.py` /
`scripts/check_dep_metadata.py`, which compare `requirements.txt` against `pyproject.toml`
itself.

**Kept deliberately**: the CHANGELOG fabrication walkers
(`test_v5_2_3_walker_changelog_content.py`, `test_v5_3_walker_changelog_self_citation.py`,
`test_v5_3_2_walker_source_line_citation.py`) — they check a changelog claim against
`git diff`, i.e. a fact, not a wording, and they are release gates.  Also kept: the
`requirements.txt` tests (a machine-consumed file) and the four 10th-walker AST tests.

The corpus still has ~20 other files that read a `.md`; they are a mix of structural
doc-consistency walkers and genuine prose assertions and were not swept, because a blanket
deletion across files owned by five other work packages is not a safe mid-campaign change.
Listed as deferred, §6.

### 2.9 Except budget

`tests/unit/test_audit_except_budget.py` rewritten.  The scalar 48 (already 3 short of the
tree at the audit base, 7 short mid-campaign) is replaced by a **per-file census of 53 sites
across 27 modules**, every one read and justified, plus the scalar total as a second bar.  Both
are `<=` bars, so narrowing never fails the gate; adding does, and a module absent from the
census has an implicit allowance of zero.

**The 53 sites.**

*(1) JAX tracer / concretization guards — 34 sites.*  `try: <materialize> except Exception:
<conservative fallback>`, where the raised type is `TracerArrayConversionError` /
`ConcretizationTypeError` / `TypeError` depending on the jax version and cannot be named
without importing jax at module scope.  Each routes a traced input to the general/exact path
instead of a concrete-only fast path, so the fallback is conservative by construction.
`elements/_berreman_jax.py` 4 (internal_field thicknesses, (3,3)-tensor inspectability,
is-traced, isotropy probe) · `elements/pmm/_core.py` 5 (`_resolve_incidence_checked`,
`_jpmm_concrete_incidence_guard`, 2×`_re_or_none`, `_n_or_none`) · `pmm/_jax_stack.py` 3 ·
`pmm/_jax_stack2d.py` 1 · `pmm/_jax_twod.py` 3 · `pmm/oned.py` 2 (`_off_mag`,
`pmm_jones_1d`'s scale probe) · `pmm/stack.py` 3 (`add_layer` traced thickness, `solve`'s OOP
probe, `_slices_consensus_check` — a failed clone solve is recorded as *infinite* disagreement,
which is conservative) · `pmm/stack2d.py` 1 · `elements/rcwa/_core.py` 5 (`_is_traced`,
`_require_jax_x64`, all-zero tensor probe, `_tensor_offplane_or_traced`,
`_reject_jax_offplane`) · `rcwa/oned.py` 2 · `rcwa/stack.py` 2 · `rcwa/twod.py` 1 ·
`optimize/jax_merits.py` 1 · `propagators/asymptotic_jax_twin.py` 1 · `propagators/gbd.py` 1.

*(2) Other untypeable optional-package boundaries — 4 sites.*
`raytrace/differential.py` 1 (numba build errors: TypingError / LoweringError / raw LLVM;
falls back to the exact NumPy dual) · `elements/_lens_real.py` 2 (`_glass_key_value` —
`get_glass_index` reaches the optional `refractiveindex` package and the fallback is an
`('unresolved', name)` cache key, never a wrong HIT; `_sag_callable_fingerprint` — probing a
**user-supplied callable**) · `raytrace/exit_vertex.py` 1 (`resolve_exit_index` —
**re-raises as ValueError**, the sanctioned third tier).

*(3) User code and best-effort probes — 9 sites.*  `optimize/context.py` 2 (pickle-ability
probe, which **warns** before passing; the merit-function probe call) · `io/storage.py` 3
(h5py `swmr_mode`, dataset re-entry, zarr append — backend-specific raise types) ·
`analysis/psf_mtf_otf.py` 2 (`CubicSpline` fit → nan, `brentq` bracket → continue) ·
`elements/bor/coupled_radial_eigensolver.py` 1 (step-index root census) · `_knobs.py` 1
(`_same` compares two knob values of arbitrary type; `ndarray == ndarray` is not a bool).

*(4) Teardown paths that must not raise — 2 sites.*  `_context.py` atexit global-restore ·
`optimize/driver.py` `__del__` dtype-restore.

*(5) **Narrowing requests** — 3 sites, counted but NOT endorsed.*  `memory.py` ×2
(`set_low_memory`'s two `from .propagators.fft_infra import …`) and `elements/_lens_imap.py`
×1 (`build_inverse_map`'s `from .. import memory`).  All three wrap a bare import where
`except ImportError` is the exact type set.  **Requests to their owners** (§5); lower the
census entry in the same change.

Four tests: the total bar, the pre-sweep ratchet (99 → 53), the per-file census, and a
counter-pin that the census carries **≤ 3 clauses of unearned slack** (measured slack: **0**)
so it cannot be set uniformly generous.  Fail-before demonstrated three ways — a file absent
from the census, a file gaining one clause, and an inflated total:

```
FAIL-BEFORE 1 ok -> ... clauses than the justified census allows (file: actual vs allowed): elements/rcwa/twod...
FAIL-BEFORE 2 ok -> ... (file: actual vs allowed): elements/pmm/_core...
FAIL-BEFORE 3 ok -> The per-file census allows 60 broad-excepts but the tree only has 53: 7 clauses of unearned slack.
```

### 2.10 Test isolation

**`test_propagation_asm_cache_lock_still_paired`** (VERIFY-A10 recorded it passing standalone
and failing inside a 577-test session).  Root cause, diagnosed rather than guessed: the test
asserted `'_ASM_CACHE_LOCK' in inspect.getsource(prop_mod._clear_local_asm_caches)`.
`inspect.getsource` resolves through `linecache` against the file **on disk** at the function
object's `co_firstlineno`, which was fixed when the module was imported.  Any edit to
`fft_infra.py` after import — routine while several work packages are in flight, and that file
has been dirty in the working tree all week — shifts the definition and the lookup returns a
neighbouring function's text.  A long session gives that window; a 0.2 s standalone run does
not.  The audit prescribes exactly this remedy ("replace `inspect.getsource` proxies with
behavioural pins", P2-2 (c)).

Rewritten as a behavioural pin: substitute recording locks for `fft_infra._ASM_CACHE_LOCK` and
`_PYFFTW_PLAN_LOCK`, call the clearer, assert both were entered, restore both in `finally`.
It also asserts the shell still re-exports the canonical object, which is the other half of the
pairing claim.  **Strictly stronger** than what it replaces: the old assertion passed on a body
that merely *mentioned* the name in a comment, and would have passed on an `acquire()` with no
release.  Fail-before demonstrated with an unlocked clearer:

```
FAIL-BEFORE ok -> v4.14.2 cache-lock-pairing regression: _clear_local_asm_caches ran without ever entering _ASM_CACHE_LOCK ...
```

**`OSError: [WinError 6] / [WinError 50]` in subprocess tests** (reported by WP-A3, WP-A5 and
WP-A11 as an environment flake).  Measured, not guessed, under this suite's own configuration:

```
pytest --capture=fd  (the default)
  HANDLES: fd0=0x21C fd1=0x268 fd2=0x24C | sys.stdin=<DontReadFromInput>
  GetStdHandle(STD_INPUT)=0x268  GetStdHandle(STD_OUTPUT)=0x1994  GetStdHandle(STD_ERROR)=0x1e8
  OK    capture_output=True, stdin inherited
  OK    capture_output=True, stdin=DEVNULL
  FAIL  stdout=PIPE only          -> OSError: [WinError 6] The handle is invalid
  OK    all three named explicitly
  OK    nothing named at all
pytest -s  (capture disabled)
  every shape OK
```

Mechanism: CPython's `Popen._get_handles` on Windows resolves **any stream left as `None`**
through `GetStdHandle()` and then `DuplicateHandle`.  pytest's fd-capture reassigns the
process's file descriptors **without** calling `SetStdHandle`, so the Win32 std handles it
duplicates can be stale.  The all-`None` case is safe because CPython short-circuits it before
touching a handle — which is why the failure only ever hit *partial* spawns, and why it looked
like an intermittent environment problem.

Fix: name all three streams.  **28 partial spawns in 17 files** at the time of measurement;
**25 fixed** in 15 files (`stdin=subprocess.DEVNULL` beside `capture_output=True`; none of
these children reads stdin), plus **2 in `scripts/`** — `verify_changelog_closures.py::_run_git`
and `stamp_changelog.py::_run`, which is where the surviving reproduction actually lived.
Before/after on the reproducing batch (7 files, same command, three consecutive runs):

```
before:  1 failed, 31 passed   (test_v16_synthetic_fabrication_is_caught, OSError WinError 6 at subprocess.py:1431)
after:   32 passed  /  32 passed  /  32 passed
```

Pinned by `test_audit2609_a15a_packaging.py::test_every_subprocess_spawn_names_all_three_stdio_streams`,
a static AST gate over `tests/` with a three-entry exemption list for the files this work
package does not own — plus a counter-pin that each exemption still points at a real partial
spawn, so the holes close as their owners fix them.

### 2.11 V7 — hygiene

* `--maxfail=50` → **10** in `unit-tests.yml` and `publish.yml`.  At 50 across 5 shards × 5
  pythons the gate tolerated 1 250 failures before any job aborted; 10 still shows the pattern
  while a catastrophic break stops the matrix in minutes.
* **`benchmarks/` gating note.**  `mark.bench` is applied 0 times and no workflow references
  `benchmarks/`, so there is no performance-regression gate at all.  The marker is kept and the
  fact is now written down in the `markers` block rather than implied.  Wiring a gate needs a
  decision this work package cannot make alone (which metric, what envelope, on whose
  hardware) — designed in §6.
* The **future-dated audit doc** (`docs/audits/SCOPE_BOR_MULTILAYER_GUARDS_2026_09_12.md`) is
  left alone: today is 2026-09-12 and the name is now simply correct.

---

## 3. Files touched

**Configuration / CI / packaging (5)** — `pyproject.toml`, `.gitignore`, `MANIFEST.in`,
`requirements.txt`, `.github/workflows/unit-tests.yml`, `.github/workflows/publish.yml`.

**Repository tools (2)** — `scripts/verify_changelog_closures.py`, `scripts/stamp_changelog.py`
(one kwarg + comment each).

**Moved (11 + 1 new)** — `scripts/_d5_*.py`, `scripts/_g8_*.py` →
`validation/probe_scripts_legacy/`, plus a new `validation/probe_scripts_legacy/README.md`.

**New tests (3)** — `tests/unit/test_audit2609_a15a_lens_covering_array.py`,
`tests/unit/test_audit2609_a15a_durations_staleness.py`,
`tests/unit/test_audit2609_a15a_packaging.py`.

**Rewritten tests (2)** — `tests/unit/test_public_api.py`,
`tests/unit/test_audit_except_budget.py`.

**Timing conversions / prose retirement / isolation (8)** — `test_audit_analysis.py`,
`test_audit_raytrace.py`, `test_perf_v4_12_0_through_focus.py`,
`test_v5_3_multi_field_merit_jit.py`, `test_niche_audit_w9_overlap_exact.py`,
`test_v4_16_2_agent_d.py`, `test_v4_16_1_agent_c.py`, and (ruff dead stores)
`test_audit2609_a7_misc.py`, `test_audit2609_a7_opd_unwrap.py`,
`test_audit2609_verify_a7.py`.

**Slow markers (11)** — listed in §2.6.

**Subprocess stdio (15)** — `tests/integration/test_validation_files.py`,
`test_audit_w6_propagators.py`, `test_backend_disable_jax.py`, `test_fix_newton_pool_memory.py`,
`test_niche_d15_deterministic_traced_fit.py`, `test_niche_r3_gbd_mem_lstsq.py`,
`test_v4_15_2_agent_a.py`, `test_v4_15_4_agent_b.py`, `test_v5_0_1_agent_a.py`,
`test_v5_2_3_dep_drift_check.py`, `test_v5_2_3_walker_changelog_content.py`,
`test_v5_3_2_stamp_changelog.py`, `test_v5_3_2_walker_source_line_citation.py`,
`test_v5_3_walker_changelog_self_citation.py`, `test_v5_4_1_stamp_changelog_net_loc.py`.

**Report / changelog (2)** — this file and `WP-A15a_CHANGELOG.md`.

No file under `lumenairy/` was modified.  `tests/conftest.py`, every `__init__.py`, and the
three A15b-owned walker files were not touched.

---

## 4. Tests run

| command | result | duration |
|---|---|---|
| `ruff check lumenairy/ tests/` | **All checks passed** (was 13 + 331 + 37) | 1 s |
| `mypy` (pyproject whitelist) | **Success: no issues found in 22 source files** (was 11) | 40 s |
| `pytest tests/unit/test_audit2609_a15a_lens_covering_array.py` | **45 passed** | 3.34 s |
| `pytest tests/unit/test_audit2609_a15a_packaging.py` | **11 passed** | 0.52 s |
| `pytest tests/unit/test_audit2609_a15a_durations_staleness.py` | **3 passed, 1 failed** (the staleness gate — §2.6) | 26.4 s |
| `pytest tests/unit/test_public_api.py` | **7 passed** (was 726) | 0.12 s |
| `pytest tests/unit/test_audit_except_budget.py` | **4 passed** | 0.47 s |
| `pytest` × the 16 directly-touched test files | **373 passed, 1 skipped, 1 failed** (pre-existing, see below) | 51.9 s |
| `pytest tests/unit/test_audit_raytrace.py` | **62 passed** | 24.8 s |
| `pytest tests/unit/test_v5_3_multi_field_merit_jit.py` | **54 passed, 1 skipped** | 2.60 s |
| `pytest tests/unit/test_niche_audit_w9_overlap_exact.py` | **20 passed** | 5.53 s |
| `pytest tests/unit/test_v4_16_1_agent_c.py` | **20 passed** | 0.29 s |
| `pytest` × the 7-file WinError reproduction batch, ×3 | **32 passed, 4 skipped** each time (was 1 failed / 31 passed) | ~18 s each |
| `pytest tests/unit --collect-only -q` | **14 199 ids** (was 14 666) | 17.5 s |
| `pytest tests/unit -m "slow and not integration" --collect-only -q` | **551 ids** (was 277) | 15.6 s |
| `pytest tests/unit -k real_lens -m "not integration"` (the V3 slice) | **129 passed, 3 skipped, 0 failed** (14 070 deselected) | 296 s |
| `pytest tests/unit/test_lens_gbd.py test_v5_14_0_pmm2d_cell.py -m slow` (two newly-marked files, proving the marker selects them) | **13 passed** | 566 s |
| `python validation/run_all.py` (BLAS pinned to 1 thread) | **ALL 37 files passed**, exit code 0 | ~9 min |

**Collected-id accounting**, per file, before → after:

```
test_public_api.py                               726 ->    7   -719
test_audit2609_a15a_lens_covering_array.py         0 ->   45    +45
test_audit2609_a15a_packaging.py                   0 ->   11    +11
test_audit2609_a15a_durations_staleness.py         0 ->    4     +4
test_v4_16_2_agent_d.py                           13 ->    8     -5
test_audit_except_budget.py                        2 ->    4     +2
test_audit_raytrace.py                            61 ->   62     +1
test_niche_audit_w9_overlap_exact.py              19 ->   20     +1
(test_audit_analysis / through_focus / merit_jit / agent_c: unchanged counts)
TOTAL for this work package                      982 ->  322   -660
whole unit suite                              14 666 -> 14 199   -467
attributable to other WPs landing concurrently                  +193
```

**Pre-existing failure found, NOT mine.**
`tests/unit/test_audit2609_a7_misc.py::test_transfer_function_recurrence_engages_only_on_a_uniform_scan`
fails on a **1-ULP** bit-identity drift (`0.2570220974913172` vs `0.2570220974913171`).
Established by measurement rather than assertion: the **HEAD copy** of that test file
(`git show HEAD:… > copy.py`) fails identically on the current working tree, and my only edit
to that file is the deletion of one dead `rng = np.random.default_rng(2)` line in a different
test.  Cause is an in-flight library change on the ASM/through-focus path
(`lumenairy/propagators/*` is dirty from several work packages).  For its owner.

---

## 5. Requested changes outside my ownership

1. **Orchestrator — regenerate `.test_durations` at campaign exit, then re-run
   `tests/unit/test_audit2609_a15a_durations_staleness.py`.**  It is red at 15.04 % today and
   will stay red until the regeneration; that is the gate doing its job.  Procedure (the one
   `unit-tests.yml` already documents): serially, on an idle machine, with
   `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`, for **both** lanes —
   `pytest tests/unit -m "not integration and not slow" --store-durations --durations-path .test_durations`
   and `pytest tests/unit -m "slow and not integration" --store-durations --durations-path .test_durations_slow`,
   then merge the two JSON objects.  Do it **after** the last WP lands, not before.

2. **Orchestrator — untrack the two benchmark JSONs** (I make no git writes):
   `git rm --cached .benchmarks/Windows-CPython-3.14-64bit/0001_v4_11_2.json` and
   `…/0002_v4_11_2_zernike.json`.  `.gitignore` already covers the directory.

3. **Orchestrator — stage the `scripts/` → `validation/probe_scripts_legacy/` rename** (11
   files, moved on disk; `git status` shows them as 11 deletions + one new directory).

4. **RCWA work package — `lumenairy/elements/rcwa/oned.py:26` and `twod.py:24`.**
   `_grazing_safe_wavelength` is imported and no longer called (F401).  Drop the import or add
   it to an `__all__` as a deliberate re-export, and re-sort the import block in `oned.py`,
   `twod.py` and `stack.py` (I001).  Then **delete** the three `TODO(audit-2609)` entries from
   `[tool.ruff.lint.per-file-ignores]`.

5. **Analytic-lens work package — `lumenairy/elements/_lens_real.py:2634-2635`.**  Two `f"…"`
   continuation lines in the mirror-guard warning carry no placeholders (F541).  Drop the `f`
   prefix and delete that per-file ignore.

6. **`lumenairy/memory.py` owner — narrow two broad excepts.**  `set_low_memory`'s two
   `try: from .propagators.fft_infra import … except Exception: pass` should be
   `except ImportError:`.  (They are counted in the census as narrowing requests, so the gate
   is green either way; lower the entry when they land.)

7. **`lumenairy/elements/_lens_imap.py` owner (WP-A3/A16) — same.**  `build_inverse_map`'s
   `try: from .. import memory as _mem except Exception:` → `except ImportError:`.

8. **VERIFY-A4 / VERIFY-A6 / VERIFY-A13 — three timing assertions and five slow markers.**
   The timing conversions are tabulated in §2.7; the slow-marker candidates with their
   measured seconds are in §2.6.  Both lists are one-line changes in files I must not touch.

9. **WP-A15b — three subprocess spawns.**  `tests/unit/test_niche_audit_w3_infra.py:753` and
   `tests/unit/test_niche_d14_deterministic_carrier_fit.py:217, :415` need
   `stdin=subprocess.DEVNULL` beside `capture_output=True` (see §2.10 for the measured
   mechanism).  They are on the exemption list in
   `test_audit2609_a15a_packaging.py::_STDIO_EXEMPT`; delete the entry when the kwarg lands.

10. **WP-A15b — nine `__init__.py` modules are `mypy --strict` clean today** (`_math`,
    `sources`, `elements/pmm`, `elements/rcwa`, `elements/eme`, `elements/bor`, `io`,
    `analysis`, and the root `lumenairy/__init__.py`).  They are deliberately not in the
    whitelist because the lazy-loading rewrite will add a PEP 562 `__getattr__`.  Please add
    them to `[tool.mypy] files` once that lands **with the `__getattr__` annotated**, and raise
    the `>= 17` floor in
    `test_audit2609_a15a_packaging.py::test_the_mypy_whitelist_only_grows` in the same change.

11. **WP-A18 (CONVENTIONS/README) — one sentence each**, if wanted: that the suite's unit lane
    is 3.63 h split into a fast matrix and a sharded slow lane (the "fast (<30 s)" claim is
    corrected in `pyproject.toml` but repeated in prose elsewhere), and that `threadpoolctl` is
    now a hard dependency.

---

## 6. Deferred, with designs

1. **Move the four jax-guarded slow-lane candidates** (`test_niche_r3_gbd_mem_lstsq.py`,
   `test_v5_11_0_rcwa_fff_nv_2d.py`, `test_gbd_feature_complete.py`, `test_bor_sem_jax.py`;
   619.7 s).  *Design*: the `jax-unit` leg selects `-m "not integration and not slow"`, so
   marking them slow deletes them from the only leg that runs jax.  Add a second step to that
   job running the same guard-selected file list with `-m "slow and not integration"` under its
   own `timeout-minutes`, then mark the four files.  *Gate*: the job's measured wall time after
   the change must stay under the 45-min step cap.  *Effort*: ~1 h, but it needs one CI run to
   measure, so it belongs to whoever can read the runner logs.

2. **Raise the dev box to `jax >= 0.11`.**  The declared floor and the local install disagree
   (§2.4).  Not a mid-campaign action: a jax upgrade changes numerics for every concurrent work
   package, and several are pinning jnp/np parity bars right now.  *Design*: after the campaign,
   `pip install 'jax>=0.11'`, re-run the jax-guarded file set (the workflow's own grep selects
   it), and re-derive any parity bar that moves — with the new measurement dated in the comment.
   *Effort*: ~2 h plus whatever the parity bars need.

3. **Convert `test_ci_kernel_consistency.py:492`.**  *Design*: the file promises a budget for a
   cross-arm decision census; the honest conversion asserts on the **number of decisions
   probed** (and that every cheap row is censused — which the file already checks) rather than
   on 20 s.  That needs the census's own owner to say what the decision count should be.
   *Effort*: ~1 h with that number in hand.

4. **Sweep the remaining `.md`-prose tests.**  ~20 files still read a `.md`; they are a mix of
   structural doc-consistency walkers (keep) and prose assertions (delete).  *Design*: classify
   by whether the assertion would survive a re-wording — if it would not, it is prose.  Replace
   the surviving *need* with the audit's own suggestion: run the README cookbook as doctests in
   a docs job, which tests the code rather than the sentence.  *Effort*: ~1 d, and it should be
   one owner in one pass rather than five work packages touching each other's files.

5. **Gate `benchmarks/`.**  No CI job references it and `mark.bench` is applied 0 times, so
   there is no performance-regression gate.  *Design*: a non-blocking workflow that runs
   `pytest benchmarks/ --benchmark-only --benchmark-json=…` on a fixed runner label, compares
   against the previous run's artefact (not against a committed JSON — that is what P1-10 is
   about), and annotates a regression beyond a *measured* cross-run envelope.  It must be
   advisory until that envelope is measured over at least ten runs, or it becomes the exact S1
   shape this work package spent its day removing.  *Effort*: ~1 d, mostly measuring.

6. **Shrink the `lumenairy/ui/` ignore list.**  331 findings in 7 rules, 250 of them
   `--fix`-able.  *Design*: one mechanical `ruff check lumenairy/ui --fix` pass (F401 121 +
   I001 116 + F811 10 + F541 3 = 250) behind a PySide6-capable smoke run, then remove those
   rules from the per-file ignore.  E701/E702/F841 (81) need hand edits.  *Effort*: ~0.5 d, but
   it needs a GUI runner to verify, which this box does not have (PySide6 is not installed).

---

## 7. Changelog

`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A15a_CHANGELOG.md`
