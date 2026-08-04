# The P2 `DID NOT WARN`: the physics is excluded on every buildable axis, and the fixture was sitting 0.87 % from a routing cliff

Niche P2 diagnosis, 2026-08-03.  Branch `feat/d121-final-closure`, measured on
the working tree at `ea6d4c2` plus the in-flight C13 edit to
`_lens_traced.py` (read-only for this study; `lumenairy/**` was not modified).

Predecessor: `C11_PHYSICAL_DECENTRE_GATE_2026_08_03` S9.3 (which located the
case and adjudicated the fixture), S9.5/S9.6 (the state-leak class and the
self-reporting pattern this study reuses).

---

## 0. Headline

CI at `5af1edf` failed
`tests/unit/test_niche_p2_guards.py::test_self_check_dx_flags_a_non_convergent_chain`
with `DID NOT WARN` on `pytest.warns(RuntimeWarning, match='NOT dx-STABLE')`.
C11 S9.3 then measured the fixture's drift at 52.5 % against a 5 % tolerance --
a 10.5x margin, identical on two BLAS builds -- and could not explain the
failure.

**This study varied eleven axes to completion across forty distinct runs of the
failing fixture, and the guard FIRED in every single one** -- including inside
a one-process reconstruction of CI's own shard 2/3, at the test's real position
(1744 of 3424, with 1742 tests ahead of it), where it PASSED.  Python 3.12 /
3.13 / 3.14, numpy 2.4.4 /
2.4.6, scipy 1.17.1 / 1.18.0, numba 0.65.1 / 0.66.0, pytest 9.0.3 / 9.1.1,
MKL and OpenBLAS, RAM budgets from 60 MB to 67 GB, BLAS thread counts 1 / 2 /
4 / 24, both `final_leg` branches, four wave propagators, the pyFFTW and SciPy
FFT dispatch flags, and the glass-catalog source: **the drift reads
52.4574 % / 50.0264 % / 3.5842 % to six significant figures in all of them.**

So the DID-NOT-WARN is **not reproducible on any physics axis that exists off
CI**, and this document ends in an exclusion table rather than a culprit.  What
it does deliver is four things that were not known before and that are worth
more than the attribution:

1. **The fixture was a coin toss on an axis nobody had measured.**  Under the
   `final_leg='auto'` default its measured exit NA is **0.14870 against
   `na_exact_threshold` = 0.15** -- 0.87 % below a routing cliff the library
   itself warns about.  C11's 10.49x margin is the PARAXIAL branch's margin.
   **On the EXACT branch the same fixture measures 1.78x, with the `peak`
   metric already INSIDE tolerance.**  The route is now pinned.
2. **Warning dedup cannot silence the TEST -- but it DOES silence the
   PRODUCTION warning.**  Measured: two identical chain calls from one source
   line under stock CPython filters deliver the `NOT dx-STABLE` flag **once**.
   That is a library defect, not a test artefact.
3. **`_run_chain_dx_self_check` has three silent-pass holes**, one of them
   measured to make this very fixture read as dx-stable (0.09 % drift) with no
   warning at all.
4. **The test is now self-reporting.**  A DID-NOT-WARN now prints the guard's
   own per-metric margin, so the next occurrence discriminates "the guard
   stopped detecting" from "the guard fired and the capture lost it" in one
   line instead of two studies.  Verified by poisoning, both miss modes.

---

## 1. The event, and the two readings it admits

The failure report contains a `DID NOT WARN`, a regex, and a list of bystander
warnings (the `ray_density` energy band at `P_out/P_ap` = 0.8757).  It does not
contain the margin.  That is the whole difficulty: exactly two readings fit the
same report and they need opposite responses.

* **reading A -- the guard did not fire.**  The chain's drift fell under
  `self_check_tol`; the physics moved.  Response: find what moved it.
* **reading B -- the guard fired and the capture lost it.**  Response: the
  physics is irrelevant and the warning plumbing is the defect.

`pytest.warns` is binary, so the report cannot separate them, and every
measurement below was chosen to attack one or the other.

For the record, the bystander text is fully accounted for: the fixture emits
`P_out/P_ap` = **0.875777** on the primary run and **0.875764** on the refined
one -- both print as "0.8757", so its presence in the log says only that the
chain ran, not how many times.  C11 S9.4 was right to call it a bystander.

---

## 2. The exclusion table

Every row is a measurement made for this document unless attributed to C11.
"drift" is the guard's own `m1` vs `m2` comparison, read out of its own INFO
log, on the failing fixture (N=768, dx=4 um, `r_in`=+3 mm).

| # | axis | how it was varied | drift (power / peak / r50, %) | guard fires | builds |
|---|---|---|---|---|---|
| 1 | **python + wheel BLAS** | 3.14.6/MKL/np 2.4.4/scipy 1.17.1; 3.13.14/OpenBLAS/np 2.4.6/scipy 1.18.0/numba 0.66.0 (a purpose-built CI-line proxy); 3.12/OpenBLAS (C11) | 52.4574 / 50.0264 / 3.5842 in all three | YES | both |
| 2 | **warning registry / `default` dedup** | same text, same line, same module, emitted in an earlier test then demanded by `pytest.warns` | n/a | YES (probes 1a/1b, 2, 5) | both |
| 3 | **leaked `ignore` filter** | an earlier test calls `warnings.simplefilter('ignore')` and never restores | n/a | YES (probe 3) | both |
| 4 | **RAM budget** | `set_max_ram` 60 MB, 0.25, 1, 2, 4, 8, 13, 16, 32, 64 GB and auto (67 GB) | 52.4574 / 50.0264 / 3.5842, bit-identical at every budget; **no `MEMORY-LIMITED` warning at any of them** | YES | both |
| 5 | **BLAS thread count** | `OMP/OPENBLAS/MKL/NUMEXPR/VECLIB` = 1, 2, 4, and unpinned (24 cores) | identical | YES | Windows + Linux unpinned |
| 6 | **glass catalog source** | `refractiveindex` is installed locally and is NOT in CI's install line; forced the bundled-Sellmeier fallback | N-BK7 @ 1.31 um = 1.503582905410 from **both** sources, delta 0.000e+00 | YES | Windows |
| 7 | **dispatch globals** (the C11 leak-guard blind spot at `5af1edf`) | `USE_PYFFTW=False`, `FFTW_MIN_SIZE=2**31`, `USE_SCIPY_FFT=False`, `SCIPY_FFT_WORKERS=1`, `DEFAULT_WAVE_PROPAGATOR` = `asm`/`sas`/`fresnel`/`rs` | 52.4574 for six of eight; `sas` 47.258 / 47.081 / 5.901; `fresnel` 56.297 / 53.492 / 3.131 -- **identical to every digit on both builds** | YES, 8 of 8 | both |
| 8 | **`final_leg` route** | `paraxial` and `exact` pinned explicitly | paraxial 52.4574 / 50.0264 / 3.5842; **exact 8.8836 / 4.3574 / 0.4212** | YES both, but see S3 | both |
| 9 | **library tree** | C11-era tree (C11 S9.3's own instrument) vs this tree (C12 arbiter + C13 lstsq step-down in flight) | 52.457 vs 52.4574 | YES | both |
| 10 | **cache warmth** (C11's stated residual: the module-level CONTAINERS its scalar guard cannot compare) | the fixture run 3x in ONE process, `_H_CACHE` / `_FREQ_GRID_CACHE` / `_BANDLIMIT_CACHE` / `_PYFFTW_PLAN_CACHE` / `_PYFFTW_BAD_SHAPES` / `_TRACED_KWARG_DEFAULTS_CACHE` counted after each | 52.457393 / 50.026380 / 3.584207 (Windows) and 52.457393 / 50.026382 / 3.584208 (Linux) on ALL THREE runs, to every digit; caches fill on run 1 (4 / 2 / 2 / 8 / 0 / 0 entries) and do not change the answer | YES, 3 of 3 on each | both |
| 11 | **shard composition** | CI's own `--splits 3 --group 2 --splitting-algorithm least_duration` reconstructed on py3.13/Linux and run in ONE process; the p2 file sits at positions 1743-1746 with **1742 tests ahead of it** | all four p2 tests **PASSED** in that process -- including after six traced-lens niche tests FAILED immediately upstream (S6) | YES | Linux |

Two rows deserve emphasis because they overturn assumptions that were being
carried:

* **row 4 kills the most attractive hypothesis before it was written down.**
  `get_ram_budget()` returns *currently available physical memory*, this box
  reports 40-67 GB and a GitHub runner has 16 GB total, and
  `_memory_bounded_n_fine` caps the exact readout's fine grid at
  `2**floor(log2(sqrt(budget)/16))` -- so the cap really is 16384 here and 4096
  there.  It is inert anyway: this fixture never requests a fine grid large
  enough to hit the cap, at any budget, on either branch.  The cap never binds
  and never warns.
* **row 7 is the strongest available negative on the state-leak class.**  C11
  closed that class by construction (an autouse guard restoring 91 flags) but
  never named a leaker.  Sweeping the flags the guard did NOT cover at
  `5af1edf` -- the whole `fft_infra` dispatch set, including
  `DEFAULT_WAVE_PROPAGATOR`, which changes *every propagation in the process* --
  moves the drift by at most 5 points and never below 5 %.  **No leakable
  dispatch global can silence this test.**

---

## 3. What IS fragile: `final_leg='auto'` is a 0.87 % coin toss under this fixture

The library says it itself, in a `RuntimeWarning` this fixture emits on every
run under the shipped defaults:

> `propagate_traced_carrier_chain`: the final group's measured exit NA
> **0.14870** sits within 20 % of `na_exact_threshold=0.15` -- BELOW (routing
> PARAXIAL).  `final_leg='auto'` flips between the exact and the PARAXIAL focus
> readout at that threshold **with no other symptom** ... **A beam-size change
> of 0.9 % would flip it.**  Pass `final_leg='exact'` explicitly to pin the
> route (the recommended production setting whenever the exit NA is anywhere
> near the threshold).

And the two branches are not equivalent for this test:

| branch | power % | peak % | r50 % | max | margin vs 5 % | metrics that trip |
|---|---|---|---|---|---|---|
| **PARAXIAL** (shipped route, C11's numbers) | 52.4574 | 50.0264 | 3.5842 | 52.4574 | **10.49x** | 2 of 3 |
| **EXACT** | 8.8836 | 4.3574 | 0.4212 | 8.8836 | **1.78x** | **1 of 3** |

Identical to every printed digit on Windows/MKL py3.14 and Linux/OpenBLAS
py3.13.

So C11 S9.3's "10.5x margin, five-figure-identical, every knob inert" is
**true and branch-conditional**, and the branch was never a knob it swept.  On
the other side of a decision the fixture sits 0.87 % from, the margin is 1.78x
and `peak` -- one of the three metrics -- is already inside tolerance.  That is
precisely the regime C11 itself called "a coin toss between BLAS builds"
(S9.3: "one that sits at 6 % is a coin toss ... and must be strengthened").

**This does not explain the CI failure** -- the exact branch still fires, on
both builds -- and it is not being offered as one.  It is offered as the reason
the test should not have been carrying that decision at all: the test is about
the dx self-check, not about `final_leg='auto'`'s routing, and a guard test
that inherits an unmeasured 0.9 %-margin branch decision is one beam-size
change away from measuring something other than what its docstring says.

**Changed:** the fixture now passes `final_leg='paraxial'` explicitly, which is
the branch every recorded number in its docstring was measured on.  Pinning
also drops the two NA-proximity bystanders (one per chain run) from the
recorder: 11 warnings under `'auto'`, 9 pinned.

---

## 4. What IS defective, guard-side (`lumenairy/**` was NOT edited -- these are recommendations)

### 4.1 The convergence flag is deduped in PRODUCTION.  Measured.

The brief asked whether warning dedup could silence a production warning,
"which would be a real library defect the user cares about far more than the
test".  **It can, and it does.**

`_run_chain_dx_self_check` emits with `warnings.warn(..., RuntimeWarning,
stacklevel=3)`.  `stacklevel=3` attributes the warning to the CALLER of
`propagate_traced_carrier_chain` -- which is correct for blame, and fatal for
delivery: under CPython's stock filters an unmatched `RuntimeWarning` takes the
`"default"` action, which is **once per (text, category, module, lineno)**.  A
batch loop calls the chain from ONE line.

`p2diag_prod_dedup.py` runs the real chain twice from one source line in a
plain interpreter with untouched filters and counts what reaches a handler:

```
total warnings delivered : 15
NOT dx-STABLE delivered  : 1        <-- two calls, ONE flag
```

**Every later non-converged result in that loop returns unflagged.**  For a
guard whose entire purpose is "no silent cliffs; a non-converged result is
never returned as if it were converged", once-per-process is the wrong
delivery contract.

Recommendation (owner of `carrier.py`): make the flag `always`-delivered rather
than registry-deduped -- either emit under a module-scoped
`warnings.catch_warnings()` + `simplefilter('always')` around the `warn`, or
give the warning its own category and document
`filterwarnings('always', category=...)`, or return the drift on the result
object so a caller can assert on it without relying on the warnings machinery
at all.  The third is the only one that is robust to a caller's own filter
configuration.

### 4.2 Three silent-pass holes in the self-check

```python
m1 = _chain_result_metrics(res)
if not m1:
    return                      # (a) SILENT
...
m2 = _chain_result_metrics(res2)
for key in sorted(set(m1) & set(m2)):   # (b) empty intersection -> SILENT
```

* **(a)** `_chain_result_metrics` returns `{}` whenever the field's total
  intensity is non-finite or `<= 0`.  A primary run that degenerated therefore
  reads as "dx-stable" and the check never runs -- and the refined chain is
  never even executed.
* **(b)** if the REFINED run degenerates, `m2` is `{}`, the key intersection is
  empty, `bad` stays empty, and the guard returns without warning **after
  paying for both chains**.  A self-check that cannot compare should say so.
* **(c) the `res.R is not None` branch is blind.**  Without a focus readout the
  guard compares `w_env`, `power` and `R` -- quantities that are dx-invariant by
  construction.  Measured on the SAME beyond-Nyquist fixture that reads 52.5 %
  through the readout:

  ```
  NO focus_readout | res.R = 0.124328 | FIRED = False
    w_env 0.028368673855 vs 0.028393267902   (0.0867 %)
    power 1.112856759e-06 vs 1.112840320e-06 (0.0015 %)
    R     0.124328099825 vs 0.124328099825   (0 %)
  ```

  `self_check='dx'` on a chain without a focus readout is very nearly a no-op
  that costs 2x runtime.  It should either compare something dx-sensitive or
  refuse the mode.

None of these is the CI failure (the failing call supplies a readout, and
`res.R` is `None` on every readout path -- verified by reading all four return
sites).  All three are silent passes in a guard that exists to prevent silent
passes.

---

## 5. The only capture mechanism that survives, and the C11 sentence it corrects

C11 S9.6 states: *"Warnings-filter leakage is already contained by pytest's own
per-item `catch_warnings`, so it is not the mechanism for S9.3 either."*

That is **correct for filters and registries, and now measured** rather than
asserted (`p2diag_capture.py`, 8 passed + 1 xfailed, identical on both builds):

* the repo's `filterwarnings = ["default", ...]` really is the once-per-location
  dedup action -- the CONTROL probe shows a same-key repeat swallowed inside one
  filter epoch;
* but `pytest.warns` enters `catch_warnings` and calls
  `warnings.simplefilter("always")`, which bumps `_filters_version` and
  invalidates every module `__warningregistry__` on the next `warn_explicit`.
  Probes 1a/1b (warm the registry in one test, demand it in the next), probe 2
  (same test, bare then wrapped) and probe 5 all PASS.  **Axis 2 is closed: no
  registry state and no leaked `ignore` filter can produce this failure.**

It is **incomplete in one way that matters**, and probe 4 demonstrates it:

> `warnings.catch_warnings` is process-global and not thread-safe.  A thread
> that entered it BEFORE `pytest.warns` and exits DURING the block restores
> `warnings.showwarning` / `_showwarnmsg_impl` to the value it saved on entry,
> **ripping out the recorder `pytest.warns` installed**.  Warnings emitted
> before the restore are recorded; ones after are lost.

That is the exact shape of the CI report -- early chain warnings present, the
LAST warning of the call missing -- and `p2diag_capture.py::test_probe4_...`
reproduces `DID NOT WARN` from it in 0.14 s, with a non-matching bystander
preserved in the recorder just as in the log.

**This is a mechanism, not an attribution.**  The failing call passes
`traced_kwargs=dict(parallel_amp=False, ...)` and `n_workers=1`, so
`_use_parallel_amp` is False and this test's own path is single-threaded; the
library's five `catch_warnings` sites live in four modules -- `pmm/stack.py`,
`rcwa/oned.py`, `rcwa/twod.py` (x2) and `fga.py` -- none of which the chain
touches, and there is no `threading.Thread` anywhere outside
`lumenairy/ui/`.  For probe 4 to
be the CI cause, a thread from an EARLIER test in the shard would have to still
be inside a `catch_warnings` block -- and every `ThreadPoolExecutor` in the
library is used as a context manager, which joins.  Recorded as the one
surviving capture mechanism, with its prerequisite unmet on the evidence
available.

---

## 6. Axis 3 -- shard composition

The composition is genuinely untested at CI's geometry, and C11's negative
result does not transfer: **C11 measured the Windows/py3.14 split, where the
failing test sits at position 291 with 12 files ahead of it.  On the CI-line
py3.13/Linux collection it sits at position 1744 of 3424, with ~100 files
ahead** (`--splits 3 --group 2 --splitting-algorithm least_duration` against the
committed `.test_durations`).  Those are different experiments.

**RESULT: all four p2 tests PASSED in that process.**

```text
#1743  PASSED  test_niche_p2_guards.py::test_self_check_rejects_unknown_modes
#1744  PASSED  test_niche_p2_guards.py::test_self_check_dx_flags_a_non_convergent_chain
#1745  PASSED  test_niche_p2_guards.py::test_self_check_tolerance_is_honoured
#1746  PASSED  test_niche_p2_guards.py::test_fit_radius_too_small_falls_back_and_says_so
```

And the negative is stronger than a clean prefix would have been, because the
prefix was **not** clean.  Seven tests were non-passing ahead of position 1743,
six of them traced-lens niche tests running the same machinery, immediately
upstream:

```text
#527   test_g08_s4_20_packaging_ui::test_manifest_ships_reference_docs      (copy artefact)
#1629  test_niche_d1_tilted_carrier::test_off_centre_fit_disc_does_not_ghost_the_exit_field
#1652  test_niche_d3_guards::test_the_separation_survives_the_c10_residual_degree_and_is_caused_by_it
#1682  test_niche_d6_exact_tilted_leg::test_exact_beats_paraxial_for_a_tilted_congruence_against_the_oracle
#1683  test_niche_d6_exact_tilted_leg::test_decentred_carrier_decentre_penalty_envelope
#1696  test_niche_d7_decentred_fit::test_the_order_steps_down_when_the_disc_cannot_constrain_it
#1702  test_niche_d7_decentred_fit::test_the_hard_mask_arm_ghosts_on_every_build
```

(Those six belong to the C12/C13 in-flight tree on Linux/OpenBLAS -- the
build-divergence family C11 S9.2/S9.5 and C13 own -- and are outside this
study's scope.  What matters here is that **a shard in which six traced-lens
tests have just failed still does not silence the P2 guard.**)

Getting this number cost three attempts: the first two runs died when the
harness stopped unrelated background wrapper tasks, at 40.6 % and 8.9 %
respectively (no OOM in `dmesg` -- the WSL process tree went with the wrapper).
The third was launched with `setsid` and polled only from the foreground.
The reconstruction command is in S8.

**Reading a running shard.**  `pytest -q` prints one CHARACTER per test and
names the failures only in the end-of-run summary, so a multi-hour shard is
opaque until it finishes.  `p2diag_shardmap.py` maps the progress characters
back onto the collected node IDs, which makes any position adjudicable while
the run is still going -- including position 1744.  (Two failures it reports,
`test_g08_s4_20_packaging_ui::test_manifest_ships_reference_docs` and
`test_g10_s5_9_layerspec_tracked::test_roadmap_tracks_s5_9_layerspec_gap`, are
artefacts of the WSL-native working copy the run uses, which initially lacked
`MANIFEST.in` and `ROADMAP.md`; they are not shard-order findings.)

This is consistent with rows 7 and 10, which had already emptied the search
space a shard-order cause would need: C11's residual hypothesis was a leaked
module-level CONTAINER (the six caches its scalar guard cannot compare), and
running the fixture three times in one process fills all six and changes the
answer by **zero digits** (row 10, both builds); no dispatch-global VALUE moves
it below tolerance either (row 7, both builds).  A shard-order cause would have
had to act through something that is neither a flag, nor a warm cache, nor the
1742 tests that actually precede it -- which, on the evidence in S5, leaves the
warning plumbing.

**Caveat, stated plainly.**  This is CI's shard GEOMETRY reproduced against
CI's dependency set, not CI's shard.  The collection differs from `5af1edf`'s
(the C12 and C13 test files did not exist then, which shifts every position),
and CI's own per-version collection differs again.  It is the closest
experiment available without CI log access, and it is negative.

---

## 7. What changed

One test file, three changes.

1. **`_expect_dx_warning` + `_dx_selfcheck_margin` + `_margin_report`.**  The
   contract is unchanged -- still `pytest.warns(RuntimeWarning, match='NOT
   dx-STABLE')` -- but a miss now carries the guard's OWN `m1` vs `m2`
   comparison, rendered as a per-metric margin table with a `TRIPS`/`inside`
   verdict and the multiple of tolerance, plus `describe_process_state()` from
   `tests/conftest.py`'s `process_state_dump` fixture.  The report ends with the
   discriminator spelled out: *if every metric TRIPS the guard DID fire and the
   capture lost it; if none does, the physics moved.*
2. **`final_leg='paraxial'` pinned** on the non-convergent fixture (S3), with
   both branches' numbers recorded in the docstring.
3. **Docstrings re-measured** on both builds and dated.

**The instrument was verified by poisoning it** (C11's standard: "an instrument
that stays silent under the condition it exists to report is worse than none").
`p2diag_instrument.py` forces both miss modes:

* *guard runs, decides stable* (`self_check_tol=10.0`) -> prints the three
  metrics with `0.05x tol ... inside` -- i.e. correctly reports "the physics
  moved";
* *guard never runs* (`self_check` omitted) -> prints `DIAGNOSIS: the guard
  logged NO self-check line -- it returned before comparing anything`.

### Files

```text
tests/unit/test_niche_p2_guards.py                     the instrument + the routing pin
docs/audits/P2_DIDNOTWARN_DIAGNOSIS_2026_08_03.md      this document
validation/repro_traced_carrier_121/
    p2diag_route.py        one instrumented run: NA, route, per-metric drift, fired
    p2diag_ram_axis.py     axis 4 -- the RAM budget and the fine-grid cap
    p2diag_state.py        axis 7 -- every leakable dispatch global
    p2diag_capture.py      capture mechanics -- registry, filters, the thread clobber
    p2diag_prod_dedup.py   the production dedup measurement (S4.1)
    p2diag_instrument.py   poisons the instrument to prove it reports
    p2diag_shardmap.py     reads a RUNNING shard's per-test verdict (axis 11)
```

No library file, no `CHANGELOG.md`, no `pmm/**` was touched.  `ruff check`
passes on `tests/unit/test_niche_p2_guards.py` (the CI lint scope);
`validation/` is in `[tool.ruff] extend-exclude`, and the `p2diag_*` scripts
carry only `I001` (the deliberate `sys.path` setup before imports, matching
`c11_p2dx_recon.py`).

---

## 8. Reproduction

```bash
# the fixture's margin, the route, and the NA -- one run, fully instrumented
python validation/repro_traced_carrier_121/p2diag_route.py
NA_THR=0.14 python validation/repro_traced_carrier_121/p2diag_route.py   # EXACT branch
WHICH=stable python validation/repro_traced_carrier_121/p2diag_route.py  # the sibling

# axis 4 -- the RAM budget (and the fine-grid cap it drives)
python validation/repro_traced_carrier_121/p2diag_ram_axis.py
BUDGETS=13,1 python validation/repro_traced_carrier_121/p2diag_ram_axis.py

# axis 7 -- every dispatch global an earlier test could leave dirty
python validation/repro_traced_carrier_121/p2diag_state.py

# capture mechanics -- registry dedup, leaked filters, the thread clobber
python -m pytest validation/repro_traced_carrier_121/p2diag_capture.py -v

# the PRODUCTION dedup measurement
python validation/repro_traced_carrier_121/p2diag_prod_dedup.py

# the instrument's own poison test
python validation/repro_traced_carrier_121/p2diag_instrument.py

# axis 11 -- CI's shard, reconstructed (py3.13 / Linux / no jax, as CI installs)
python -m pytest tests/unit -m "not integration and not slow" \
  --splits 3 --group 2 --splitting-algorithm least_duration \
  --durations-path .test_durations -p no:cacheprovider -q | tee shard.log

# ... and read its per-test verdict WHILE IT RUNS (-q gives one char per test;
# names appear only in the end-of-run summary)
python -m pytest tests/unit -m "not integration and not slow" --collect-only \
  -q -p no:cacheprovider --splits 3 --group 2 \
  --splitting-algorithm least_duration --durations-path .test_durations \
  | grep '::' > ids.txt
python validation/repro_traced_carrier_121/p2diag_shardmap.py ids.txt shard.log
```

A CI-faithful Python 3.13 proxy was built for this study with
`uv venv -p 3.13` and CI's own dependency set (no `jax`, no
`refractiveindex`, matching `pip install -e ".[fft,perf,numba,hdf5,zarr,dev]"`):
numpy 2.4.6, scipy 1.18.0, numba 0.66.0, pytest 9.1.1.

### Both-builds evidence

| | Windows / MKL | Linux / OpenBLAS |
|---|---|---|
| python / numpy / scipy / pytest | 3.14.6 / 2.4.4 / 1.17.1 / 9.0.3 | 3.13.14 / 2.4.6 / 1.18.0 / 9.1.1 |
| fixture drift (paraxial) | 52.4574 / 50.0264 / 3.5842 | 52.4574 / 50.0264 / 3.5842 |
| fixture drift (exact) | 8.8836 / 4.3574 / 0.4212 | 8.8836 / 4.3574 / 0.4212 |
| sibling fixture (tol 1e-4) | 0.1019 / 0.1052 / 0.0008 | 0.1019 / 0.1052 / 0.0008 |
| `test_niche_p2_guards.py` | **12 passed** in 70.25 s | **12 passed** in 633.94 s |
| `p2diag_capture.py` | **8 passed, 1 xfailed** | **8 passed, 1 xfailed** |
| in CI's shard 2/3, one process, position 1743-1746 | n/a | **4 of 4 PASSED** |

---

## 9. What remains open

1. **The attribution.**  All eleven rows are complete and all eleven are
   negative, shard composition included.  **The honest statement is that the
   `5af1edf` DID-NOT-WARN is not reproducible off CI on any axis this study
   could build**, and the next CI occurrence -- which will now print its own
   margin -- decides it in one line (S10).
2. **The one command that will name a state leak** if that is what it is, from
   C11 S9.6 and still the right instrument:
   `LUMEN_TEST_FLAG_LEAK_STRICT=1 pytest <the shard>`.  Note it covers scalars
   only, and row 7 already shows that no dispatch-global VALUE can silence this
   test, so a leak would have to act through a module-level CONTAINER (one of
   the six caches the guard cannot compare) rather than a flag.
3. **The three guard-side holes in S4** and the **production dedup in S4.1**
   belong to `carrier.py`'s owner.  S4.1 in particular is a shipped-behaviour
   defect with a two-line reproduction, independent of anything CI did.
4. **The `--maxfail=5` interaction was not investigated.**  Which tests get to
   report at all depends on how many failed before them, so "3.13 and 3.10 but
   not 3.11/3.12" may be a reporting artefact of the abort rather than a
   version signal.  Nothing here depends on that being true.

---

## 10. Reading the NEXT occurrence (the whole point of S7)

The failure text will now contain a margin table.  Three lines decide it:

1. **Every metric says `TRIPS`.**  The guard fired and the capture lost the
   warning.  Do not touch the physics.  Go to S5: look for anything in that
   shard that leaves a thread inside `warnings.catch_warnings`, and treat the
   emission itself as the defect (S4.1's recommendation is the durable fix --
   return the drift on the result object and assert on the number, not on a
   warning).
2. **No metric says `TRIPS`.**  The physics moved.  The table names WHICH
   metric and by how much, and the `full guard log` line carries both raw
   metric dicts -- compare them against S8's both-builds table to see what
   changed.  Re-run `p2diag_route.py` on the CI python to get the route and the
   measured NA in the same breath.
3. **`DIAGNOSIS: the guard logged NO self-check line`** or **`NO SHARED METRIC
   KEYS`.**  The guard never compared anything: this is S4.2 (a) or (b), a
   library-side silent pass, and it is a `carrier.py` bug regardless of what CI
   was doing.

The attached `describe_process_state()` dump covers the flag/cache half of the
state question at the moment of failure, which rows 7 and 10 already show is
unlikely to be the carrier -- so if it prints clean, that is confirmation, not
a dead end.
