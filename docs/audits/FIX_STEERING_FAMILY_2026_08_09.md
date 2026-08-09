# set_default_* steering test family -- two sightings, adjudicated

Date: 2026-08-09
Branch: feat/pmm-per-layer-roadmap
Family: `tests/unit/test_v5_1_0_agent_a.py` (the `set_default_wave_propagator`
/ `set_default_dy` / `set_default_real_dtype` consumer-wiring pins)

Two tests in this one file were flagged failing in two different
environments on the same day.  The working hypothesis handed to this
investigation was a PROCESS-GLOBAL STATE interaction -- the steering
knobs are process globals, and this branch added pool payload / dispatch
code to `_lens_traced.py` that could plausibly have leaked one.

**Neither sighting was a steering-global leak.  Neither was a library
defect.  They are two unrelated false failures with two unrelated
causes, and the state hypothesis is refuted for both.**

A real (previously unguarded) leak in the family was found while
refuting it, and is fixed here.

---

## S1. Sighting 2 (CI) -- NOT A FAILURE. Log-annotation artifact.

**Claim.** `TestSetDefaultWavePropagatorSteersPropagateThroughSystem::
test_unsupported_default_raises_value_error` carried an `##[error]`
annotation in the GitHub run; the shard reached 99% and the job was
suspected of hitting its 45-minute cap, leaving open whether the test
FAILED, HUNG, or errored in teardown.

**Adjudication: it PASSED.**  It did not fail, did not hang, and did not
error in teardown.  The job did not hit the cap either.

Evidence, from the saved log (`ci2_failed.log`, all three shard-1 jobs
present -- py3.10, py3.12, py3.13):

```
02:46:11 ##[error]tests/unit/test_v5_1_0_agent_a.py::TestSetDefaultWave
         PropagatorSteersPropagateThroughSystem::
         test_unsupported_default_raises_value_error PASSED [ 86%]
```

The annotation text ENDS IN `PASSED`.  The `##[error]` prefix is not
pytest's -- it is the workflow's own log-summary step.
`.github/workflows/unit-tests.yml:179`:

```yaml
grep -E "^FAILED |^ERROR |error |Error:" unit_test_output.txt \
  | head -n 20 \
  | while IFS= read -r line; do echo "::error::${line}"; done
```

Three of the four alternatives are anchored (`^FAILED `, `^ERROR `,
`Error:`); the fourth, `error `, is **not anchored and not
word-bounded**, so it matches anywhere in a line.  Every pytest node-id
whose test name ends in `..._error` is followed by a space and `PASSED`,
producing the substring `error ` -- so every such PASSING test is
re-emitted as a GitHub `::error::` annotation.  In the py3.10 shard-1
job that is 7 passing tests annotated as errors, ahead of the 2 real
`FAILED` lines; `head -n 20` then truncates, so the annotation list is
mostly noise.

Timing and outcome, same job:

```
= 2 failed, 3374 passed, 200 skipped, 7351 deselected, 208 warnings
  in 2325.50s (0:38:45) =
```

38:45 of pytest inside a job that ran 02:06:42 -> 02:46:11 (39.5 min):
under the 45-minute cap, exited normally on `exit code 1`.  The two real
failures are `test_niche_newton_pool_both_fits::
test_pool_result_is_bit_identical_to_serial[polynomial]`
(pool-vs-serial `max|delta| = 1.358e-11`) and
`test_pmm_m3_efficiency::test_t34_guard_is_silent_on_the_cured_ladder`
-- **neither in this family**.  The other jobs in the same run fail on
`test_m1_conditioning_guard` and `test_pmm_m2_window_contract`, also
unrelated.

Local hang probe, for completeness (the class, under `pytest-timeout`):

```
tests/unit/test_v5_1_0_agent_a.py::TestSetDefaultWavePropagatorSteers
    PropagateThroughSystem   3 passed in 0.77s
```

0.77 s against a 120 s timeout.  Nothing to hang.

**No fix is applied here.**  The defect is in
`.github/workflows/unit-tests.yml`, which is outside the writable set
for this task.  Recommended one-character-class fix, for whoever owns
the workflow: anchor the loose alternative the way its three siblings
are anchored, e.g. `^FAILED |^ERROR |^E   |Error:` (dropping bare
`error ` entirely), or at minimum `[^_]error `.  Until then, an
`##[error]` annotation on this repo is NOT evidence that the named test
failed -- read the line's own verdict word.

---

## S2. Sighting 1 (local) -- NOT A LEAK. Source-vs-bytecode line skew.

**Claim.** In a local serial full-suite preflight,
`TestSetDefaultRealDtypeSteersGeometricLensPhase::
test_geometric_lens_phase_honours_real_dtype` FAILED; it passes in
isolation and in file context on the settled tree (17 passed).  The
preflight was run while the tree was being edited.

**Adjudication: a false failure caused by editing the tree mid-run.  No
steering global is involved -- the failing assertion does not read one.**

The test, as written before this fix:

```python
def test_geometric_lens_phase_honours_real_dtype(
        self, restore_default_real_dtype):
    import inspect
    from lumenairy.elements import _lens_traced as _lt
    src = inspect.getsource(_lt._geometric_lens_phase)
    assert 'get_default_real_dtype' in src
```

It is a SOURCE-TEXT pin.  Its only two inputs are the module object in
`sys.modules` and the bytes of `lumenairy/elements/_lens_traced.py` on
disk.  `DEFAULT_REAL_DTYPE` -- the global the enclosing class is named
after -- is never read.  **No value of any steering knob can change this
assertion's outcome**, so the process-global hypothesis is refuted for
this test on inspection, before any reproduction.

What CAN change its outcome is the file on disk:

* `func.__code__.co_firstlineno` is baked in when the module is
  IMPORTED.  For `_geometric_lens_phase` that is line 2550 of an
  ~8 kLOC file, with the pinned `get_default_real_dtype` call ~53 lines
  further down at 2603.
* `inspect.getsource` -> `getsourcelines` -> `findsource` calls
  `linecache.checkcache(file)` FIRST, which drops the cached lines
  whenever the file's `(size, mtime)` moved, so `getlines` RE-READS from
  disk.
* Fresh lines + stale line number = `inspect.getblock(lines[2549:])`
  returns whatever function now begins at line 2550.

Any edit landing anywhere ABOVE line 2550 shifts the target out from
under the pin.  This branch's work -- the new pool payload / dispatch
code -- is in that same file.

Demonstrated in a controlled repro (a synthetic module with a decoy
function above the pinned one; import, then rewrite the file with nine
lines prepended, changing nothing else):

```
BEFORE EDIT: co_firstlineno = 10
BEFORE EDIT: 'get_default_real_dtype' in src -> True
AFTER EDIT : co_firstlineno = 10 (unchanged -- baked into the code object)
AFTER EDIT : 'get_default_real_dtype' in src -> False
AFTER EDIT : getsource returned:
    def _decoy_function(a):
        ...
```

The pin fails while reading a completely different function's body, with
no library change, no global touched, and no test-order dependence --
and passes again the moment the tree settles.  That is the observed
signature exactly.

**Reproduction attempts that came back GREEN**, as expected once the
cause is understood (all single-process, `-p no:randomly`, on the
settled tree):

| run | result |
| --- | --- |
| `test_v5_1_0_agent_a.py` alone | 17 passed |
| shard-1-ish order: `test_audit_optimize.py test_fix_d5_fit_domain_basis.py test_v5_1_0_agent_a.py` | 144 passed |
| the named class under `--timeout=120` | 3 passed in 0.77 s |

The polluter search was not widened past this, because the pin cannot
read a polluted global.

### The fix

`_function_source(func)` in `tests/unit/test_v5_1_0_agent_a.py` replaces
`inspect.getsource` for this pin: read the file fresh, find
`^(\s*)def <name>\s*\(` BY NAME, slice to the next line at or below that
indentation.  No line number is consulted, so the answer is a property
of the file as it currently is.  A second assertion guards the guard --
the returned block must start with `def _geometric_lens_phase(` -- so a
future extraction bug reports itself instead of silently making the
marker assertion meaningless.

Same-scenario proof, `inspect.getsource` against `_function_source` on
an identically edited module:

```
settled tree  : getsource pin        -> True
settled tree  : _function_source pin -> True
edited mid-run: getsource pin        -> False   <-- the false failure
edited mid-run: _function_source pin -> True    <-- skew-proof
edited mid-run: block starts with    -> def _geometric_lens_phase(...)
```

In-repo precedent for reading from disk rather than through `inspect`:
`tests/unit/test_v5_1_0_agent_f_split.py::test_prescriptions_shell_is_thin`
("Reads the source from disk directly (not `inspect.getsource`) so the
result doesn't depend on `linecache` state pre-warmed by sibling
tests").

**Scope note.** `inspect.getsource` appears at 111 sites across 46 files
in `tests/unit`.  Every one of them is a member of this false-failure
class whenever a preflight overlaps an edit of the file it pins.  Only
the one in the assigned family is changed here; `_function_source` is
written to be liftable if the class is ever swept.

---

## S3. The real leak, found while refuting the hypothesis

The state hypothesis was tested rather than argued away, with a
per-test state-diff harness (a pytest plugin snapshotting
`DEFAULT_REAL_DTYPE`, `DEFAULT_COMPLEX_DTYPE`, `DEFAULT_WAVE_PROPAGATOR`
and `DEFAULT_DY` across `fft_infra`, `propagation`, `_lens_traced` and
`carrier` before and after every test, comparing by IDENTITY and never
restoring, so a leak is left visible).  Run over all 12 files in
`tests/unit` that call a `set_default_*` setter (919 tests):

```
LEAK test_v5_1_0_agent_a.py::TestSetDefaultRealDtypeSteersGeometric
     LensPhase::test_geometric_lens_phase_honours_real_dtype ::
     fft_infra.DEFAULT_REAL_DTYPE : <class 'numpy.float64'>
                                  -> dtype('float64')
LEAK test_audit_misc.py::...::test_warns_when_plane_logger_raises ::
     fft_infra.DEFAULT_COMPLEX_DTYPE : <class 'numpy.complex128'>
                                     -> dtype('complex128')
```

Two independent findings behind that:

### S3.1 The save/restore fixtures were not identity-preserving

`restore_default_real_dtype` restored with
`set_default_real_dtype(get_default_real_dtype())` -- which looks like a
no-op and is not.  `set_default_real_dtype` normalises its argument
through `np.dtype(...)` (`fft_infra.py:372`), while the SHIPPED value is
the numpy scalar TYPE `np.float64`.  So the "restore" rewrote
`DEFAULT_REAL_DTYPE` from `np.float64` to `np.dtype('float64')`.

Harm today: none that could be found.  Every library consumer of these
two knobs wraps them (`np.dtype(_state.DEFAULT_COMPLEX_DTYPE)` in
`asm.py`, `fresnel.py`, `mft.py`, `rs.py`, `sas.py`, `carrier.py`;
`.astype(...)` in `_lens_real.py`), and `np.dtype('float64') ==
np.float64` is True.  It is nonetheless a change this file made to the
process and did not undo, and identity leaks are precisely what a future
`is`-comparison trips over.

Fixed: the three fixtures now go through one `_restored(attr)`
contextmanager that saves and `setattr`s back the `fft_infra` attribute
itself.  `fft_infra` is the single source of truth -- `propagation`
deletes its import-time bindings and forwards through `__getattr__`
(`propagation.py:293-305`) -- so one write restores every view.
`_restored` restores on the exception path too, so a RED steering test
cannot hand a steered knob to the next test.

Pinned by the new `TestSteeringKnobSaveRestoreIsExact` (identity bar for
all four knobs, exception-path restore, and that `propagation` still
forwards to `fft_infra` so a single-target restore stays complete).

### S3.2 The conftest leak guard had a hole at exactly these two knobs

`tests/conftest.py` `_module_flag_leak_guard` (autouse, module scope) is
the house-style order-independence guard: it discovers every upper-case
module-level knob in `_lens_traced`, `carrier` and `fft_infra`, then
restores any that moved.  Its type filter was

```python
_LEAK_GUARD_TYPES = (bool, int, float, str, type(None))
```

`DEFAULT_REAL_DTYPE` and `DEFAULT_COMPLEX_DTYPE` ship as `np.float64` /
`np.complex128`, which are `type` objects, and become `np.dtype`
instances after any setter call.  Neither spelling is in that tuple, so
**the guard never looked at the two dtype steering knobs at all**.  A
test that called `set_default_real_dtype(np.float32)` and forgot to
restore would have handed every later test in the process a float32 OPL
accumulator, silently, with the guard whose entire purpose is that class
standing by.  `DEFAULT_WAVE_PROPAGATOR` (str) and `DEFAULT_DY`
(float/None) were always covered; only the dtype pair was exposed.

Fixed: `_LEAK_GUARD_TYPES = (bool, int, float, str, type(None), type,
np.dtype)`.  Both spellings are listed because either can be the value
in flight (shipped = type, post-setter = dtype).  Measured against the
three guarded modules, this admits the two dtype knobs and nothing else.
It does not make the guard fussy about the type-vs-dtype SPELLING: the
guard's comparison falls back to `==`, and `np.dtype('float64') ==
np.float64` is True, so benign renormalisation is tolerated while a real
float32 / complex64 leak is caught and reverted.

Harness re-run after the fix: the `test_v5_1_0_agent_a.py` leak is gone
(925 tests, 0 leaks from this file).  The two residual entries are the
same benign type -> dtype renormalisation in `test_audit_misc.py` and
`test_v4_16_3_agent_b.py`, now caught by the guard's snapshot and left
alone by its `==` comparison; those files were not edited, being outside
this task's family.

---

## S4. Files changed, and why each was in scope

| file | why |
| --- | --- |
| `tests/unit/test_v5_1_0_agent_a.py` | the family under repair: `_function_source` (kills the S2 false-failure mechanism), `_restored` + the three fixtures (kills the S3.1 leak), `TestSteeringKnobSaveRestoreIsExact` (pins the fix) |
| `tests/conftest.py` | the S3.2 guard hole is in the shared guard, is in this family (the dtype steering knobs), and cannot be fixed from the test file |

No `lumenairy/**` file was changed.  The two sightings were not library
defects, and the one library asymmetry found (S3.1: the public setter's
`np.dtype` normalisation makes a save/restore round trip through the
public API non-identity-preserving) is harmless to every in-tree
consumer and would be a public-surface change to alter -- it is
documented above rather than "fixed".

## S5. Green evidence (all single-process; a serial preflight was live
on the box, so no full-suite run was started)

| run | before | after |
| --- | --- | --- |
| `test_v5_1_0_agent_a.py` isolated, `-p no:randomly` | 17 passed | **23 passed** (+6 new pins) |
| shard-1-ish order (`test_audit_optimize.py test_fix_d5_fit_domain_basis.py test_v5_1_0_agent_a.py`) | 144 passed | **150 passed** |
| all 12 `set_default_*` caller files, with the leak harness | 919 passed, 4 leaks | **925 passed, 2 leaks** (both benign, both outside this family) |
| 8 flag-heavy files exercising the edited leak guard (`test_audit_w2_fft_state`, `test_context_manager`, `test_audit_g06_perf`, `test_audit_s5_8_perf_noloss`, `test_lens_chunked_sag`, `test_niche_audit_p2_fresnel_tf_buffer`, `test_audit_v5_24_2_g05_seams`, `test_memory_guardrail`) | -- | **121 passed** |
| `test_v5_1_0_agent_a.py` under RANDOM order, 3 repeats | -- | **23 / 23 / 23 passed** |
| `ruff check` on both edited files | -- | **All checks passed** |

No `xfail`, no `skip`, no rerun-to-green.

## S6. Open leads (found here, out of scope, for whoever owns them)

1. **The CI annotation grep** (`unit-tests.yml:179`, and the same
   pattern at `:427` for the JAX job).  `error ` is unanchored; it
   annotates passing tests as errors and pushes the real `FAILED` lines
   out of the `head -n 20` window.  This is what made Sighting 2 look
   like a failure.  See S1 for the suggested pattern.

2. **Pool workers do not inherit the FFT / precision steering knobs.**
   `fft_infra.snapshot_fft_state` / `restore_fft_state` exist for
   exactly this (v5.4.6 audit P3-16, extended v5.17.1 P3-54) and
   `_FFT_STATE_KEYS` correctly names `DEFAULT_COMPLEX_DTYPE`,
   `DEFAULT_REAL_DTYPE`, `DEFAULT_WAVE_PROPAGATOR`, `DEFAULT_DY` -- but
   **no in-library pool calls them.**  `carrier._multi_worker_init`
   carries `_WORKER_STATE_MODULES` state, and that tuple is
   `(glass, _lens_traced, carrier, asm)` -- `fft_infra` is not in it,
   and `_WORKER_STATE_SCALARS = (bool, int, float, str, bytes,
   NoneType)` would exclude the dtype knobs even if it were (the same
   type-filter hole as S3.2, in the library rather than the tests).  The
   `_lens_traced` persistent Newton `ProcessPoolExecutor`
   (`_get_persistent_pool`, spawn) passes no initializer at all.
   Consequence: a caller who sets `set_default_complex_dtype(
   np.complex64)` and then uses `congruence_workers>1` gets workers
   computing at complex128 -- different physics, silently, which is the
   exact failure `_multi_capture_worker_state`'s own docstring says it
   exists to prevent.  Latent for the Newton pool (that payload does not
   read the dtype knobs); live for the congruence pool, whose chain runs
   through `asm.py`'s `np.dtype(_state.DEFAULT_COMPLEX_DTYPE)`.
   Worth pricing against the still-open
   `test_niche_newton_pool_both_fits::
   test_pool_result_is_bit_identical_to_serial[polynomial]` failure
   (`max|delta| ~ 1.3e-11`) -- though that delta is
   BLAS-reduction-order sized, not dtype-flip sized (~1e-7), so it is
   probably a different cause.

3. **The other 110 `inspect.getsource` pins** in `tests/unit` share the
   S2 mid-run-edit fragility.  `_function_source` is the drop-in.
