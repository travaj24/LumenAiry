# WP-B14 -- the known reds, the CI matrix, and the warning-attribution sweep

Wave 5 item D (`PLAN_WAVE5_LEFTOVERS_2026_09_14.md`), branch
`fix/known-reds-and-stacklevels`, base `96cb2096` (the 5.47.0 release commit
`4bf26c5e` plus the Wave-5 plan).  Closes handoff item 4.6 (the known reds) and
the mechanical half of 4.4 (warning attribution), and takes the CI-matrix
remediation for run 34914295323.

Every number below was measured on this branch.  The rule applied throughout is
the maintainer's: **a test that is red on one arm and green on another is a
defect to be root-caused by measurement, never masked, never rerun-to-green,
never loosened without a derived two-sided bar; a decision that depends on the
kernel, the thread count, the wheel or the interpreter is a library defect.**

## 0. The environment the numbers were taken on

| item | value |
|---|---|
| box | Windows 11 Pro 10.0.26200, 127.9 GB RAM |
| interpreter | CPython 3.14.6 (Windows), 3.12.3 (WSL `~/lumvenv`), 3.13.13 (Windows, no pytest) |
| numpy | 2.4.4 |
| BLAS | `libscipy_openblas64_` 0.3.31.188.0, `pthreads`, architecture **Haswell** by default |
| kernel ladder | `OPENBLAS_CORETYPE` in {HASWELL, NEHALEM, KATMAI, SANDYBRIDGE} x `*_NUM_THREADS` {1, 4}, confirmed per arm with `threadpoolctl` (ZEN aliases Haswell here; SKYLAKEX crashes and was not used) |
| tree pinning | every invocation carries `PYTHONPATH=/c/tmp/lum_reds`; every probe asserts `lumenairy.__file__` contains `lum_reds` (the `pip -e` install points at `D:\...\Lumenairy`, a different branch) |
| invocation | `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` on the command line, pytest with `--capture=sys` (handoff 4.1b) |

Probes and their per-arm JSON: `validation/probe_known_reds/`.

---

## 1. RED 1 -- the c7 / c8 halo pins: **a real regression, not box state**

### 1.1 What the handoff said, and why it was wrong

Handoff section 1 records four ids -- `test_niche_c7_ray_density_halo_check.py::
test_fires_on_the_fit_guard_regression` and three in
`test_niche_c8_inverse_support_bound.py` -- as failing "on this box since
2026-09-14 ... alone, in a `git archive` tree of the 5.46.0 base `81d5b586`,
with a fresh numba cache, with `NUMBA_DISABLE_JIT=1` and with `--capture=sys`,
so neither a Wave-4 change nor the capture interaction; the fixture's
manufactured lobe depends on something in the box state that was not traced".

It is not box state.  Two facts settle it:

1. **CI reproduces it, on Linux, to twelve digits.**  Run 34914295323 fails the
   same four ids on py3.12 / 3.13 / 3.14 with
   `the fail-before arm stopped manufacturing light (1.522e-04)` and
   `assert 0.00015217194665917948 < (0.00015217194665917948 / 100.0)`.  This box
   reads `0.00015217194665917584`.  A box-state effect does not agree with a
   fleet of Linux runners to twelve significant figures.
2. **It bisects to one commit.**  The reproduction was run against
   `git archive`d `lumenairy` trees (never the shared working tree), each
   imported in a child process with `PYTHONPATH` set to the archive and
   `lumenairy.__file__` asserted under it.

### 1.2 The bisection

`validation/probe_known_reds/probe_c8_guard_reach.py` + `at_commit.sh`.  The
quantity is the fixture's manufactured halo: `max|E|` beyond `3 w`, over the
peak, on the `_GHOST` cell with the C6 launch and the C6 fit guard on and the C8
support bound off.

| tree | halo beyond 3 w, bound OFF | bound ON | verdict |
|---|---|---|---|
| `v5.45.1` (`adff3652`) | **4.594672922387141e-02** | 8.913411901995892e-04 | the lobe is manufactured; matches the fixture's own docstring (`4.593e-02 -> 8.911e-04`, "51.5x") |
| `bbb6c02d^` | **4.594672922387141e-02** | 8.913411901995892e-04 | same, bit for bit |
| `bbb6c02d` | **1.5217194665917584e-04** | 1.5217194665917584e-04 | the lobe is gone; the bound has nothing to remove |
| `4bf26c5e` (5.47.0) | 1.5217194665917584e-04 | 1.5217194665917584e-04 | same |

`bbb6c02d` is **WP-A26**, `fix(lens-traced): the decentred ray fit's order is
re-derived against the full conic ray set (10 -> 16)` -- a Wave-1..3 (5.46.0)
commit that moved `_DECENTRED_FIT_POLY_ORDER` from 10 to 16 for an unrelated and
well-measured reason (niche D7's decentred exit-slope error, 44.5 / 31.6 urad at
order 10 against 2.4 / 1.7 at order 16).

### 1.3 The mechanism, and the intermediate wrong answer that was ruled out

The first hypothesis -- that `REMAP_STATIONARY_PHASE_FIT_GUARD` had stopped
reaching the code (a monkeypatch-shaped flag that no longer lands) -- was
**refuted by measurement**: guard on vs off differs by `max|dE| = 7.010e-02`, so
the flag reaches.  A second candidate, the G8 inverse-characteristic refusal
that fires on the guard-on cells ("off-lattice entrance-position error
1.0523e-07 m against the incumbent Newton path's 7.4904e-08, 1.40x, bar 1.00x"),
was also ruled out: with the inverse map forced off the halo is bit-identical to
the refused case (`1.5217194665917584e-04` both ways), so the refusal is a
passenger, not the cause.

The cause is the fit order itself.  Walking it
(`validation/probe_known_reds/probe_c8_fit_order.py`, halo beyond `3 w`):

| `decentred_fit_poly_order` | 6 | 8 | **10** | 12 | 14 | 16 (default) | 18 |
|---|---|---|---|---|---|---|---|
| bound OFF | 1.463e-04 | 1.463e-04 | **4.595e-02** | 1.463e-04 | 1.533e-04 | 1.522e-04 | 1.537e-04 |
| bound ON | 1.463e-04 | 1.463e-04 | **8.913e-04** | 1.463e-04 | 1.533e-04 | 1.522e-04 | 1.537e-04 |
| ratio | 1.00 | 1.00 | **51.55x** | 1.00 | 1.00 | 1.00 | 1.00 |

The defect class these two files exist to catch -- an extrapolated inverse-map
value handing real amplitude to an exit pixel no ray reached -- is reachable on
this geometry **at order 10 and at order 10 only**.  At 10 the numbers reproduce
the fixtures' own documented readings exactly, including the 51.5x the C8
docstring quotes.

### 1.4 The fix, and the layer

**Test layer, and it is a restatement rather than a relaxation.**  The stimulus
is now STATED by the fixture instead of inherited from a library default that
moved: `decentred_fit_poly_order=10` is added to `_BASE_KW` in both files, with
the measured ladder and the reason in the comment.  This is the same pattern
`test_pmm_m2_window_contract.py::_mk` already uses for `min_feature` ("It has to
be stated rather than inherited: the library default moved").

No bar was loosened, no assertion was weakened, and the fail-before assertion
(`h_before > 1.0e-02`) stays a HARD assertion -- the premise is now fully
determined by parameters the test states, and it reproduces bit-identically
across Windows and the Linux CI fleet, so gating it would delete the test's
meaning rather than protect it.

WP-A26 itself is **not** reverted and its order-16 default is not challenged:
its derivation is sound and its own acceptance bar is cleared by 4.3x / 6.1x.
What WP-A26 failed to do was restate the two fixtures that depended on the old
default -- the same "stale pin" class the 5.47.0 gate found four more of.

### 1.5 Evidence

```
before:  4 failed, 24 passed in 55.29s
after :  28 passed in 72.82s
```
Per-arm: section 7's ladder table, 8/8 arms green.

---

## 2. RED 2 -- `test_pmm_m2_window_contract` T3-1, the BLAS-build classification

### 2.1 Reproduction

The test passes on this box at the default kernel, which is why it had not been
pinned down.  Under the mandated ladder it reproduces on exactly one arm:

| arm | outcome (before) |
|---|---|
| HASWELL 1 / 4 thr | pass |
| NEHALEM 1 / 4 thr | pass |
| KATMAI 1 / 4 thr | pass |
| SANDYBRIDGE 1 thr | pass |
| **SANDYBRIDGE 4 thr** | **fail** |

```
AssertionError: every cell of both devices was screened out as classification-
unsound, so T3-1 was not measured at all. ... screened: [('uncoated ns=3', 10, 0, 2, 0),
('25 nm coat ns=8', 6, 0, 0, 1), ('25 nm coat ns=8', 8, 1, 2, 2), ('25 nm coat ns=8', 10, 2, 2, 2)]
assert 0 >= 1
```

### 2.2 Root cause, with numbers

The message overstates its own case.  On that arm the uncoated device's degree-6
and degree-8 cells **passed** the screen and their contract assertions ran and
held; only degree 10's halfwidth-2 run put 2 growing modes into a forward set.
So the test did not measure "nothing" -- it measured two of three rungs and then
failed because `full_ladders >= 1` demands a COMPLETE three-rung ladder.

Which cell the flux cut classifies by round-off is, in the module's own words at
`_MODE_CUT_CENSUS`, "a per-build, per-thread-count fact", and the sibling T3-4
census documents the same thing.  So the failing assertion was the one thing in
the test that was a *reading* rather than a *decision*: the existence of a
complete ladder on this particular build.

### 2.3 The fix

Three claims, separated by what they depend on:

* **Unconditional** -- the comparative contract (`|dJ| < 0.5 dJ_deg`,
  `|dR| < 0.5 dJ_deg`) on every cell that passes the screen on the running arm.
  Unchanged.
* **Unconditional, and now stronger** -- the spectral-decay claim is asserted
  over whatever rungs were measured (`len(dJs) >= 2`) rather than only over a
  complete three-rung ladder.  On SANDYBRIDGE / 4 threads that means the decay
  claim from degree 6 to degree 8 is now asserted where previously it was
  dropped.
* **Hard failure, unconditional** -- if NO cell of either device passed the
  screen, the test measured nothing and says so.  That protection is kept.
* **Premise-gated** -- the existence of a complete three-rung ladder is measured
  and, when it does not hold on the running arm, `pytest.skip`s **carrying the
  reading**: the cells that were measured with their numbers, and the full
  screened census.

The eight-arm ladder and the reason are written into the test body.

### 2.4 Evidence

```
HASWELL     t1  1 passed      SANDYBRIDGE t1  1 passed
HASWELL     t4  1 passed      SANDYBRIDGE t4  1 skipped
NEHALEM  t1/t4  1 passed      KATMAI   t1/t4  1 passed
```
and the skip message on the failing arm carries what the arm did measure:

```
SKIPPED: T3-1 spectral-decay ladder: no device yielded a complete three-rung
(degree 6/8/10) ladder on THIS build ... The window contract itself IS asserted
above on the 2 cell(s) that passed the screen here: uncoated ns=3 deg=6:
|dJ|=1.298e-04 vs degree residual 1.232e-03; uncoated ns=3 deg=8: |dJ|=3.123e-05
vs degree residual 4.849e-04.  Screened (device, degree, n_grow at hw=1 / hw=2 /
degree+2): [('uncoated ns=3', 10, 0, 2, 0), ...]
```

Both measured cells reproduce the table in the test's own docstring (ratios
0.105 and 0.064), which is the independent check that the premise gate did not
quietly change what is being measured.

---

## 3. RED 3 -- the order-dependent glass-validity one-shot pin

### 3.1 Reproduction

```
pytest tests/unit/test_audit_w4_glass_registry_meshgrid.py \
       tests/unit/test_v4_16_0_agent_d_validity_ranges.py
-> FAILED ...::test_validity_warning_is_one_shot_per_pair
   AssertionError: Out-of-range warning must be one-shot per (glass, wavelength)
   pair; got 0 warnings.  assert 0 == 1
```
Green in the reverse order and green alone.

### 3.2 Root cause -- a PARTIAL reset of coupled state, in the test

The first hypothesis (the warn-once set `_validity_warned` leaking) is wrong: an
instrumented run of the meshgrid file alone shows that set empty at every
teardown.

The state that actually leaks is `_glass_value_cache`.  `get_glass_index`
memoises the whole `(name, wavelength)` evaluation and **returns on a hit before
it reaches `_maybe_warn_outside_validity`** (`lumenairy/glass.py`, the value-memo
block above the validity call).  The memo's own rationale says this is
warning-neutral only because the two are emptied together: *"`_clear_glass_caches`
empties both together, so a cleared cache re-warns exactly as it did."*

`test_audit_w4_glass_registry_meshgrid.py` calls
`get_glass_index('N-BK7', 200e-9)`, leaving that pair memoised.  The consuming
file's `autouse` fixture `_reset_validity_warned` cleared **only**
`_validity_warned` -- producing precisely the state the library's design
excludes: the warn-once set says "not yet warned" while the value cache answers
before the warning site is reached.  Five calls, zero warnings.

The library is self-consistent; the test was doing a partial reset of coupled
state.  **Fix layer: test.**

### 3.3 The fix

The `autouse` fixture drains through the library's own registered drain,
`la.clear_asm_caches()` (which empties the value cache, both warn-once sets and
the reloadable catalogue entries), before and after each test, with the
mechanism and the failing order written into the docstring.

### 3.4 Evidence

```
meshgrid THEN validity : 24 passed in 7.85s   (was: 1 failed, 23 passed)
validity THEN meshgrid : 24 passed in 7.23s
validity alone         :  8 passed in 1.88s
```

---

## 4. RED 4 -- the interpreter crash: bounded, with a mechanism and an opt-in fix

### 4.1 What was to be found

Handoff section 1: "the first full fast lane of the day died after 4 h with
`Windows fatal exception: access violation` (its traceback was lost to a
filtered log)".  Section 5 places one of the two recorded instances "in the dense
`reconstruct_field_from_beamlets` path of `lumenairy/propagators/gbd.py`, which
passes alone in 30 s", and notes the box has 127 GB so "it is not memory".

### 4.2 What was measured

That dense loop contains no native code of its own -- no numba, no numexpr -- so
a native-level fault in it, appearing only beside other heavy jobs, is an
allocation story.  The question that CAN be answered by measurement is whether
the loop's `mem_budget_mb` bounds what it allocates.  It does not.

`validation/probe_known_reds/probe_gbd_dense_budget.py`, `tracemalloc`, a
64/128/192/256 grid ladder x {512, 64} MB budgets x {512, 1024} beamlets:

| N | beamlets | budget | effective chunk | **measured peak** | overrun | B per cell-col |
|---|---|---|---|---|---|---|
| 64 | 512 | 512 MB | 512 | 196.5 MB | 0.38x | 93.7 |
| 64 | 512 | 64 MB | 512 | 151.1 MB | **2.36x** | 72.1 |
| 128 | 1024 | 512 MB | 1024 | 1208.5 MB | **2.36x** | 72.0 |
| 192 | 512 | 512 MB | 512 | 1360.1 MB | **2.66x** | 72.1 |
| 192 | 512 | 64 MB | 108 | 384.1 MB | **6.00x** | 96.5 |
| 256 | 512 | 64 MB | 61 | 387.1 MB | **6.05x** | 96.8 |
| 256 | 1024 | 512 MB | 488 | **3073.5 MB** | **6.00x** | 96.1 |

Declared: **16.0 B** per (output cell x beamlet-column).  Measured: **72.0 to
96.8 B**.  The overrun saturates at 6.0x = 96/16 once the chunk is the binding
constraint.

The arithmetic error is plain once it is looked at.  The comment reads "bytes per
beamlet-column of the dense working set ~ Ny*Nx*16 (the dX/dY/rho2/phase
buffers)" -- but 16 B is the size of ONE complex128 element, not the sum of three
float64 buffers (`dX`, `dY`, `rho2`) and one complex128 (`phase`), let alone the
`dX*dX` / `dY*dY` temporaries.  The WINDOWED sibling in the same file does the
accounting correctly and says so: `_WINDOWED_CELL_BYTES = 32.0`, with a written
per-array tally of ~26 B/cell measured and the margin stated.

### 4.3 The fix, and why it is opt-in

Correcting the constant changes the chunk boundary, which changes the order the
per-chunk `einsum` reductions are summed in, which -- floating-point addition not
being associative -- changes the output bytes on a **default** path.  Under the
house rule a default moves only with a Migration note and a measurement beside
it; everything else ships opt-in behind a switch whose default reproduces the
previous release exactly.  So:

* `_DENSE_CELL_BYTES_LEGACY = 16.0` -- the shipped figure, with the measurement
  that shows it is not a bound recorded against it;
* `_DENSE_CELL_BYTES_MEASURED = 128.0` -- the honest figure.  **Two-sided**: the
  lower side is the measurement (below 96.8 the budget stops bounding the loop);
  the margin is the one the windowed sibling already carries (32.0 against ~26,
  1.23x), rounded up to a power of two, 1.32x the worst measurement; the upper
  side is cost, because a larger constant only shrinks the chunk and buys wall
  time for nothing (at 128 the chunk is 8x smaller than at 16);
* `DENSE_MEM_BUDGET_ACCOUNTING = 'legacy'` -- the switch, defaulting to the
  byte-identical state.  An unrecognised value falls to `'legacy'`, so a typo
  cannot silently move a user's bytes.

Verified on the worst cell (N=256, 1024 beamlets, 512 MB budget):

```
legacy   peak 3118.9 MB   overrun 6.09x
measured peak  387.1 MB   overrun 0.76x   (UNDER budget)
max|legacy - measured| = 2.1e-17 relative   -- summation-order round-off only
```

### 4.4 What is NOT claimed

**No access violation was reproduced.**  What is established is that the
transient is up to six times the size the caller asked for, which is a defect on
its own terms and is consistent with a fault that appears only under memory
pressure.  The user-facing mitigations that exist today without flipping the
switch are `window=5.0` (the bounded-support scatter-add, whose accounting IS
correct) or dividing `mem_budget_mb` by six.

New pins: `tests/unit/test_wave5_gbd_dense_mem_budget.py` (5 tests) -- the
default is `'legacy'` and byte-identical; the overrun is asserted as a
fail-before (bar 3.0x, half the measured 6.0x and three times the repaired
state); `'measured'` puts the peak under the budget (bar 1.0x, measured 0.76x);
the two arms differ only by round-off (bar 1e-12 against a measured 2.1e-17);
an unknown value is treated as `'legacy'`.

**Reserved for the maintainer:** whether `DENSE_MEM_BUDGET_ACCOUNTING` becomes
`'measured'` by default.  It is a default-path byte move and belongs on the
handoff 4.7 list.

---

## 5. Warning attribution (handoff 4.4, mechanical part)

### 5.1 The fail-before, measured

`validation/probe_known_reds/probe_carrier_attribution.py` wraps
`warnings.warn` and, for every warning raised from inside the package, computes
two things from the live stack: the frame the call's own `stacklevel` actually
names, and the first frame outside the package (what `caller_stacklevel()`
returns).  A site is misattributed when they disagree.  This needs no guess about
how deep any entry point is.

The decisive pair is one warn site reached at two library depths:

* **A** -- `propagate_carrier_referenced(..., gap_kernel='fresnel', tilt=(0.12, 0))`
  called directly: `stacklevel=3` names the caller.  Correct.
* **B** -- `carrier_referenced_focus_readout(...)` with the same kwargs, which
  forwards them into the SAME entry point one library frame deeper
  (`carrier.py` line 3987 as it then was): `stacklevel=3` names **`carrier.py` itself**.

| | warnings from `carrier.py` | misattributed |
|---|---|---|
| before the sweep | 4 | **2** (both arms of the focus-readout cell) |
| after the sweep | 4 | **0** |

### 5.2 The sweep

`caller_stacklevel()` walks out to the first frame outside the package, so it is
correct at every depth.  `lumenairy/elements/_lens_kernels.py` imports nothing
from `lumenairy`, so importing it from the propagators introduces no cycle
(verified by importing the three modules in a pinned child process).

| module | `warnings.warn` literals | threaded literals | after |
|---|---|---|---|
| `lumenairy/propagators/carrier.py` | 14 | 21 (helper signature defaults and call sites: `_guard_dispose`, `_warn_undeduped`, `_check_collins_sampling`, `_collins_transport`, `_check_focus_containment`, `_check_readout_replica`) | 0 / 0 |
| `lumenairy/propagators/system.py` | 6 | 0 | 0 |
| `lumenairy/propagators/carrier_field.py` | 1 | 2 | 0 / 0 |

The three warning helpers now take `stacklevel=None` meaning "compute it", and
an explicit integer is still honoured with exactly its old meaning, so the
external callers (`_lens_real.py`, `carrier_field.py`, and
`test_niche_c14_encapsulation.py`, which passes `stacklevel=1`) are unaffected.
`_warn_undeduped` counts from its own CALLER rather than from its own frame, so
its computed default is `caller_stacklevel() - 1`; that one-frame offset is
stated in the code.

### 5.3 Why these three modules and not all nineteen

The ratchet's rationale is that a literal encodes ONE call depth, so it is wrong
wherever a warn site is reachable at more than one.  A static reachability screen
(`validation/probe_known_reds/probe_stacklevel_reachability.py`) flags **19 of
the propagator modules**, which is too permissive to act on blindly -- it is a
possibility test, not a measurement.

The three swept are the ones where multi-depth reach is the DOCUMENTED shape and
was MEASURED: `carrier.py`'s public entry points call one another and the traced
chain calls `propagate_carrier_referenced` twice more; `system.py`'s chain runs
its legs through the same helpers from `propagate_through_system` and from
`evaluate`; `carrier_field.py` reports through `carrier._guard_dispose` from its
own entry points.

The remaining 16 files are recorded, with their census, as open work rather than
swept silently: each needs its own two-caller measurement first, which is not a
mechanical change.  Library-wide census:
`validation/probe_known_reds/stacklevel_census_base.json` (23 literals in
`io/prescriptions_zemax.py`, 9 in `sources/core.py`, 9 in `user_library.py`,
8 in `elements/pmm/stack.py`, 8 in `optimize/driver.py`, 8 in `propagators/gbd.py`, ...).

### 5.4 The ratchet and the proof

`tests/unit/test_audit2609_b11_hygiene.py`:

* the literal-stacklevel scanner is factored into one `_literal_stacklevel_lines`
  helper (one implementation per kernel);
* `test_no_literal_stacklevel_is_left_in_the_swept_lens_bodies` keeps its id and
  its eight files;
* **new** `test_no_literal_stacklevel_is_left_in_the_swept_chain_bodies` covers
  the three chain modules, with the measurement and the scope reasoning in its
  docstring;
* **new** `test_the_same_warn_site_names_the_caller_at_two_depths` is the
  two-caller fixture of section 5.1 as a test: the same warn site reached
  directly and one library frame deeper must name the caller's file in both
  arms.

```
tests/unit/test_audit2609_b11_hygiene.py: 85 passed in 137.03s   (was 82)
```

History fingerprints re-recorded by the recorder for `carrier`, `carrier_field`,
`lumenairy.propagators.system` and `lumenairy.propagators.gbd`, each with the
reason; `record_history_fingerprints.py --check` reads
`OK: every history document matches its module.`

### 5.5 RESERVED FOR THE MAINTAINER -- not taken here

Handoff 4.4's second bullet, **whose diagnostic the in-glass gap-leg warnings
are**: the `wave_propagator='sas'` / `'fresnel'` notices on `apply_real_lens`
name `_lens_real.py` because the propagator warns and its caller is the lens.
Making them name the user needs the lens to CATCH and RE-EMIT at its own entry
point, which is a design decision about ownership of the diagnostic, not a
mechanical substitution.  It is deliberately untouched and stays on the handoff
4.7 list (WP-B11 section 4b, request 1).

---

## 6. The CI matrix (run 34914295323)

34 jobs, 30 red, on the 5.47.0 release commit.  The matrix now runs Python
3.10-3.14 x 5 shards, a 5-shard slow lane, a JAX job and a strict mypy job.
Downloaded logs: `C:/tmp/ci_5470/<job>/`.  Job-level outcome as found:

| lane | outcome as found |
|---|---|
| py3.10 x 5 | **every shard aborted at collection** -- `18 skipped, 1 error`; the lane tested nothing |
| py3.11 x 5 | **every shard aborted at `--maxfail=10`** -- 10 failed / 178-223 passed each |
| py3.12 x 5, py3.13 x 5, py3.14 x 5 | 1-8 failures per shard |
| slow lane x 5 | **all five timed out** at the 30-minute step cap (1847-1863 s each; shard 2 reached 65 %, shard 5 reached 98 %, no summary line) |
| mypy strict | 1 error |
| JAX | 2 failures, both already in the classes above |

Because 3.11 aborted at 10 per shard, **the 3.11 failure set is a lower bound**
and the census of what is broken on 3.11 is unknowable from that run -- which is
itself the argument for the `--maxfail` item below.

### 6.1 py3.10 -- the whole lane aborted at collection

`tests/unit/test_audit2609_a15a_packaging.py` imports `tomllib`
unconditionally; `tomllib` is 3.11+, and CI installs `tomli` on 3.10 for exactly
this reason (the install step says so).  Fixed at the test layer with the
standard fallback, so the lane runs at all.  Consequence worth stating: **py3.10
has been running zero of the 2 800-odd ids per shard**, so the first green 3.10
lane will be the first time several gates -- the a17 history gates among them --
are exercised on that interpreter.

### 6.2 py3.11 -- the a17 history token digest (PEP 701)

**49 distinct ids** of
`test_the_module_token_stream_is_unchanged_since_the_history_move` failed on
3.11 and **0** on 3.12 / 3.13 / 3.14.  The sibling AST test was **123/123 green
on 3.11**, which is the first hard datum: the module sources are identical on
every arm, so the thing that moved with the interpreter is the digest's own
definition.

Mechanism, measured: PEP 701 landed in 3.12, so `x = f"a{b}c"` tokenises as
**seven** records on 3.12+ (`FSTRING_START` / `FSTRING_MIDDLE` / `OP` / `NAME` /
`OP` / `FSTRING_MIDDLE` / `FSTRING_END`) and **one** `STRING` record before it.
**110 of the 123** registered modules contain an f-string (3 445 `JoinedStr`
nodes); **13** do not; the 7 ids that PASSED on 3.11 are all in the 13 and the
49 that failed are all in the 110, both directions clean.  So the true count of
affected modules is **110**, not the 49-id lower bound the aborted run showed.

The decisive step, taken without a 3.11 interpreter on the box: the actual
3.11-computed digests were scraped out of the 49 CI failure blocks, and the new
scheme -- collapsing each `FSTRING_START..FSTRING_END` run to one `STRING`
record carrying the f-string's exact **source slice** -- reproduces **49 of 49**
of them bit-for-bit on 3.12.3, 3.13.13 and 3.14.6 alike.  That leaves no
residual for any other tokenizer difference (NEWLINE/NL/INDENT sequencing and
the rest), so the mechanism is established rather than assumed.

Fixed by making the digest version-independent (the brief's option A), not by
skipping below 3.12, because the claim the test exists for **can** be kept: the
source slice is FINER than the token run (which normalises `{{` to `{` and says
nothing about spacing inside a replacement field).  Six new falsifiability cases
prove the digest still moves on a value-preserving re-spelling of an f-string --
quote character, prefix case, spacing inside `{...}`, a conversion, a format
spec, an implicit concatenation -- and three companions prove the first three
are invisible to the AST fingerprint, which is what keeps the two fingerprints
independent.  A registry-wide sweep asserts no tokenizer-specific record name
ever reaches the digest, which is the guard that catches the next PEP 701.

**110 of 123** history documents re-recorded by the recorder (never by hand),
each exactly +2/-1 lines; `ast_sha256` moved on **0 of 123**, which is the
arithmetic proof that this was a digest-scheme change and not a code change.
`record_history_fingerprints.py --check` is green, and the recorder and the gate
were confirmed to share ONE implementation of the digest (no second copy to
drift).

Kernel ladder 8/8 arms: 123/123 both fingerprints, rc=0 -- no build dependence,
as expected for a digest that touches neither numpy nor BLAS.  The dependence
was on the interpreter alone.

### 6.3 The a8 glass tests -- the library was right, the tests were not

Seven CI reds, all in `test_audit2609_a8_glass.py` and
`test_audit2609_a8_verify.py`, from CI deliberately not installing the glass
extra.  The brief's hypothesis was that the first two --
`get_glass_index_complex raised ... instead of falling back to kappa = 0` --
were a LIBRARY contract violation.  **They are not, and `lumenairy/glass.py` is
byte-identical to the base commit** (md5 checked both sides).

The reasoning is worth keeping.  `refractiveindex` IS installed on this box, so
its absence had to be made a FIXTURE: a blocker that wraps
`importlib.util.find_spec` and sets `sys.modules['refractiveindex'] = None`
before `lumenairy.glass` is first imported (the module decides availability once
at import).  Against a pristine `git archive` tree that fixture reproduces
**exactly the seven CI ids and nothing else**.

With the package blocked, of 49 tuple-registered glasses **exactly one --
`SILICON` -- has no bundled row**; the other 48 return `n + 0j` with one
warn-once each.  And the `ImportError` does not come from the extinction path at
all: `get_glass_index` itself cannot produce a real index for `SILICON`, and
`get_glass_index_complex` reaches `kappa` only after the real index.  Catching
it and "falling back to kappa = 0" would mean **fabricating a real index** --
precisely the silent-wrong shape WP-A8's own E2 second arm exists to kill.  The
module docstring settles it ("Only the tuple-style entries that lack a Sellmeier
fallback will raise"), and so does an internal contradiction: a sibling test in
the same family REQUIRES `get_glass_index_complex` to raise `ValueError` out of
page range, so "never raises" was never literal.

`pytest.importorskip` was ruled out for all of them -- not by preference but by
`docs/TESTING_STANDARDS.md` rule 4 ("Never `pytest.skip` on a resource check --
two skips silently removed five tests from the gate on exactly the runners that
mattered"), which both files already cite.  Every test instead asserts the
documented no-package fact as a two-sided partition: the exempt set must EQUAL
the independently computed "tuple entry with no bundled row on this install"
set, each exemption must be an `ImportError` naming both remediations, the
premise is re-measured, and a hard floor keeps the sweep from shrinking.
`test_e2_missing_kappa_warns_once_and_returns_zero` is split PER PARAMETER, so
the three glasses that do exercise the fallback still run on both installs; a
blanket skip would have deleted them.
`test_e2_pages_that_do_carry_k_keep_their_value_and_sign` is split by half --
the real-index half is bit-identical on both installs
(`0x1.802abb7771dacp+0`) and stays unconditional.

Not weakened: with the package present, restoring the pre-WP-A8 catch tuple
turns the rewritten gate red naming the right glasses, so the exemption cannot
swallow an extinction regression.

```
a8_glass + a8_verify, package PRESENT : 64 passed, 0 skipped
a8_glass + a8_verify, package BLOCKED : 64 passed, 0 skipped
```
Same collection count on both arms, no coverage deleted.

---

### 6.4 `--maxfail` 10 -> 50, and what the lower cap actually cost

The 2026-09-12 change to 10 argued that "50 per shard across 5 shards x 5
pythons tolerated up to 1 250 failures before any job aborted".  That sums the
MATRIX, but nothing waits on the matrix sum: a job's wall clock is bounded by
its own budget, and a catastrophically broken job reaches 50 in its first
minutes exactly as it reaches 10, so the early-abort property survives the
revert.

What it cost, measured on run 34914295323: all five py3.11 shards stopped at
exactly `10 failed`, having executed 188-233 of ~2 933 selected ids -- **6.4 to
7.9 %**, so **more than 92 % of the 3.11 lane never ran**.  49 of the 50
failures are ONE parametrised gate with 123 cases, and the same case ids pass on
the other interpreters in the same run.  Independent measurement (6.2) puts the
true size of that class at **110 of 123**, so the run reported 45 % of it.  At
~22 cases per shard, 50 enumerates the class whole in one run and 10 truncates
every shard at the same 45 %.

The cap costs a healthy lane nothing: the 3.12 shards reported 3 and 2 failures,
3.13 reported 4 / 2 / 4, 3.14 reported 8 and 1 -- all below 10, so the cap never
fired there.  It fires precisely on the lane whose census is worth having.  The
comment also records that this is a FAILURE budget, not an error budget: a
collection error aborts the shard whatever `--maxfail` says, which is the 3.10
mechanism of 6.1.

### 6.5 The slow lane -- the timeout was predictable before the run started

All five slow shards hit the 30-minute step cap (1 847-1 863 s; shard 2 stopped
at 65 %, shard 5 at 98 %, neither with a summary line).  The brief's hypothesis
was a balance problem from files moved into the lane without durations.  **That
is not what it was**: measured against the committed `.test_durations`, the slow
selection is **935 ids with 935 entries -- a 100 % coverage, zero gap**, so
nothing needed regenerating and the file was not touched.

The real cause is size.  The lane now totals **9 630.0 s** against the
**5 914.9 s** the 2026-09-12 note recorded -- **+62.8 %**, because later waves
kept moving files over the two-minute bar without re-running the sum.  At
`--splits 5`, `least_duration` balances that to **1 926.0 s per shard, +/-0.0 %**,
against an 1 800 s step cap: the lane was predicted to time out on every shard
before the run started.

Both changes are needed, and the arithmetic is in the workflow comment: at 8
splits the per-shard budget is 1 203.8 s; the runners scale the recorded seconds
by 0.98x (shard 5) to 1.48x (shard 2), so sizing on the worst factor gives
~1 782 s -- against the old 1 800 s cap that is a 1 % margin, not a margin, and
raising the cap alone leaves 1 926.0 x 1.48 = ~2 850 s against 2 700 s, still
over.  Together: ~1 782 s against 2 700 s, 34 % headroom.  Shipped as `--splits`
5 -> 8 with the shard list in step, the slow step cap 30 -> 45 min and the job
cap 35 -> 50, plus an explicit "`--splits` must equal the length of the `shard:`
list" warning at both sites.

Open, and flagged in the comment for an owner: the 0.98-1.48x spread itself.
100 % duration coverage rules out the missing-entry explanation; the leading
candidate is that `.test_durations` was captured with BLAS unpinned while the
slow lane pins `OMP/OPENBLAS/MKL_NUM_THREADS=1` in its `env` block, so eig-bound
entries are on a different scale there than where they were timed.

### 6.6 mypy strict -- one error, fixed at the expression

`decompose_lg` normalised its `only` argument with
`tuple(tuple(k) for k in only)`.  `tuple(iterable)` types as
`tuple[_T_co, ...]`, so `tuple(k)` of a `(p, ell)` pair **widens**
`tuple[int, int]` to `tuple[int, ...]`: the length information was destroyed by
the constructor, at the call site.  The callee's annotation is the truth, not
the lie -- it consumes `only` as `for (p, ell) in only`, twice.  Unpacking by
name instead (`tuple((int(p), int(ell)) for (p, ell) in only)`) types exactly as
`tuple[tuple[int, int], ...]`, keeps the normalisation the re-wrap existed for
(a caller handing `[[0, 1]]` or numpy scalars still arrives hashable for the
frozenset and the cache key) and keeps the loud failure, one frame earlier.
Neither the whitelist nor an `ignore` was touched.

Before: `Found 1 error in 1 file (checked 33 source files)`.  After:
`Success: no issues found in 33 source files`, on Windows py3.14 and WSL py3.12.
Behaviour proved bit-identical against the full decomposition for tuple, list
and numpy-scalar inputs, with the same exception class and message shape for
wrong-length input.  Reported and NOT edited: the identical redundant re-wrap in
`asymptotic_aberration_tensor.py`, which is outside the mypy whitelist today and
will raise the same error the moment that module joins the ratchet.

### 6.7 The platform / kernel bit pins

Each was root-caused before any bar moved, and in three of the six the cause was
NOT rounding.

* **`test_d3_offplane_fff_nv_keeps_the_cells_own_mirror`** (b5) -- the fixture's
  precondition, an exact `np.array_equal` mirror test, failed on Linux and
  passed on Windows.  Cause: `uniaxial_tensor` builds the x and y legs from
  different expressions in `cos(phi)` and `sin(phi)`, so at `phi = 45 deg` the
  fixture's symmetry is only as exact as the platform libm's
  `sin(pi/4) == cos(pi/4)`.  MSVC returns both as `0x3fe6a09e667f3bcd`; glibc
  returns `sin` one ULP low, and the tensor's `exz`/`eyz` and `ezx`/`ezy` pairs
  then differ by 3.33e-16.  **Fixed in the construction, not the bar**: the
  fixture is averaged with its own mirror, which is exact on every IEEE-754
  platform (addition is commutative and multiplication by 0.5 is exact), so
  `array_equal` is kept.  A derived 8-ULP guard bounds how far the raw tensor
  may sit from its mirror before the averaging would be manufacturing a symmetry
  rather than repairing round-off.

* **`test_h4_bor_pencil_eigh_accuracy`** (a14) -- 1.326e-12 against a 1e-12 bar.
  The bar was one arm's reading.  It is now derived from the pencil: for a
  symmetric-definite pencil reduced by Cholesky, LAPACK's backward-error result
  bounds the relative error on `sqrt(lam_i)` by `p(n) eps lam_max / (2 lam_i)`,
  worst at the smallest eigenvalue, with `lam_max` read off the solver's own
  spectrum and pinned to its mesh-determined range so a broken solve cannot
  inflate its own bar.  The ladder shows an **11x swing on one machine from the
  BLAS kernel alone** (HASWELL 2.758e-13, NEHALEM 8.882e-14, KATMAI 9.924e-13,
  SANDYBRIDGE 5.732e-13), with KATMAI landing 0.8 % under the old bar; WSL reads
  2.758e-13, bit-identical to the Windows HASWELL arm, which is the control that
  says the swing is the kernel and not the OS.  Fail-before by mutating the
  discretisation through the public API: degree 8 -> 5 puts the residual 2-4
  decades outside on every fixture, and dropping the `m^2` axis term puts it 11
  decades outside.

* **`test_the_tf_step_is_bit_identical`** and
  **`test_the_on_axis_answer_moves_by_at_most_two_ulp`** (a6) -- the CI reading
  of exactly `0.0` is not rounding.  `_fit_carrier_inv` evaluates the moment as
  `sum(xm * (wgt*slope))` when it projects and as `sum((wgt*xm) * slope)` when
  it does not: a different ASSOCIATION of the same three factors, which
  `carrier.py`'s own comment there already said "moves the answer by a few ulp".
  Multiplication is commutative but not associative, so byte equality was never
  an invariant of that pair.  Measured, the projection correction is 3e-21 to
  2e-19 ULP of the moment it corrects -- it cannot move any bit -- and the whole
  difference is the re-association, 0-4 ULP across the eight-arm ladder and
  0-2 ULP on Linux.  Bar 16 ULP, unconditional: 4x over the worst arm measured
  and 14 decades under the defect it guards (a genuine centring error is
  `1 + 2 x0^2/w^2`, ~2e15 ULP).

* **`test_z3_stokes_and_dop_peak_arrays`** (a11) and
  **`test_b8_apply_jones_matrix_peak_full_grid_arrays`** (b8) -- both are
  allocation counts, not timings, and both moved because NumPy's `temp_elide.c`
  rewrite of `a*X + b*Y` / `Ex * conj(Ey)` into an unreferenced temporary's own
  buffer is a **BUILD** property (it needs `backtrace()` and a stack walk that
  can confirm the temporary came from the interpreter).  It does not follow the
  operating system: on this run the add form was elided on the py3.14 Linux
  runner and not on py3.12 or py3.13, while the conj form was elided on py3.11.
  So each file MEASURES its own elision premise on the running arm, with a
  companion arm that binds the temporary to a name (lifting its refcount out of
  elision's reach) asserted at the un-elided count, so a reading of "no elision
  here" can never come from an instrument that measured nothing.  The slack
  above a whole number of grids is derived two-sided: tracemalloc's own
  bookkeeping measured 448-7 712 B over 9 repeats on two arms, at most 4.6e-04
  grids, and the slack is 0.05 -- two decades above that spread and 1.3 decades
  below the 1.0 that separates one allocation count from the next.

* **`test_z3_estimate_lens_memory_real_bounds_apply_real_lens`** (a11, found on
  this box's own ladder rather than in the CI logs) -- read 0.88 against a
  fail-safe bar of 1.0, i.e. the pre-flight estimate under-reserving.  It was
  neither load nor a real under-reservation: the test's warm-up call ran on a
  64x64 block, BELOW the grid size at which `apply_real_lens` takes a
  deferred-import branch, so ~11.2 MB of one-time module imports landed inside
  the measured region (16.78 MB from `fft_infra`, 9.55 MB from
  `importlib._bootstrap_external`, 4.19 + 4.19 MB from `asm.py`).  The
  Windows-minus-Linux excess is 13.7 MB and is EXACTLY constant across N and
  dtype -- bytecode, not lens arithmetic.  Warming at 256x256 makes all four
  (N, dtype) cells byte-identical on both arms (est/peak 1.064 / 1.239 / 1.072 /
  1.323), with both bars unchanged, plus a new unconditional guard that the
  measured call must RETAIN 5.5-7.0 complex grids (clean 6.01-6.09; contaminated
  8.38 under pytest and 9.49 standalone -- 15 % above the worst clean and 16 %
  below the lowest contaminated).

* **`test_w3_t3b_*`** (w3 oracles) -- `val_a` swings 2.7x across BLAS kernels on
  one box while `val_b` holds to 1.6e-3 and the response to 3.2e-05, because the
  merit's aberration-free reference is COLLAPSED on that chart, so `val_a`'s
  scale is not a measurement at all.  The physics pins keep their bars; `val_a`
  gets a derived order-of-magnitude band (4.8x under the lowest reading, 18x
  over the highest, with the smallest rescale it must still catch nine decades
  away); and the collapse itself becomes a POSITIVE pin -- the warning must fire
  and name a coupling far outside [0, 1] -- so the band can never quietly become
  a band on a Strehl.  The library-side repair (a saddle-local reference) is
  VERIFY-A4 O-3b and is not taken here.

### 6.8 A LIBRARY DEFECT found while root-causing the a6 bit pin -- named, not fixed

Chasing the "bit-identical" transfer-function pin (6.7, third bullet) produced a
finding that is bigger than the test it came from, and it is recorded here
because it is the next wave's, not this one's.

**With the shipped pyFFTW two-buffer ping-pong ON, `_ifft2(_fft2(E) * H)` in
`lumenairy/propagators/fft_infra.py` is not a function of its input values
alone.**  Two call sites handed bit-identical `E` and bit-identical `H` return
results that differ.  Measured on WSL (Linux, numpy 2.4.6, pyFFTW 0.15.1), and
only at `n >= FFTW_MIN_SIZE` (256):

| mode | n = 128 | n = 256 | n = 512 |
|---|---|---|---|
| shipped (ping-pong on) | 0 | **1.448e-15 / 1.798e-15** (79 % of doubles differ) | **1.790e-15 / 1.897e-15** |
| `set_fft_double_buffer(False)` | 0 | **0** | **0** |
| `USE_PYFFTW=False` | 0 | 0 | 0 |

Identical at 1, 2 and 8 FFTW threads, under both ESTIMATE and MEASURE planning,
and across repeats; Windows reads exactly 0 in all three modes on all eight
ladder rungs.  Neither route is the "right" one -- both sit 2.5e-15 to 3.1e-15
from pocketfft -- so this is a reproducibility defect, not an accuracy one, and
the physics is unaffected at ~1 ULP.

What makes it a defect rather than a curiosity: **it contradicts
`set_fft_double_buffer`'s own docstring**, which states that values are
byte-identical either way.  A byte-identity contract that the shipped default
cannot honour is exactly the kind of claim the audit exists to find.

Not fixed here, and deliberately: the ping-pong is a documented 256 MB - 1 GB
per call saving with its own tests, there is no minimal correct edit visible
from outside the module, and the mechanism below the A/B was NOT established --
slot-plan parity, buffer stability across an intervening allocation, operand and
output alignment (swept 0 / 16 / 32 / 48 mod 64 on the real values), thread
count and planner effort each test clean in isolation.  Reproducer:
`validation/probe_known_reds/probe_c4_double_buffer.py`.  **Handed to the owner
of `fft_infra.py`.**

The a6 test itself was fixed without waiting for that: the byte claim now runs
where the change under test lives -- the whole shipped function against the
oracle with the ping-pong off, restored in a `finally` -- unconditionally on
every arm and every shape, measured 0.000e+00, with a second unconditional
claim on the shipped dispatch at a derived 1e-13 relative bar (250x over the
worst arm's 3.9e-16 and nine decades under the 1e-4 exact-versus-Fresnel kernel
gap).  No skip was needed.

### 6.9 One more BLAS-build classification, found by the wave's own ladder

`test_niche_audit_w3_oracles.py::test_w4_t1_pure_lg00_has_no_sigma_grid_and_is_unchanged`
was not in the CI failure set -- CI never reached it, because `--maxfail`
aborted first.  The kernel ladder run for the two briefed w3 pins found it: six
arms `181 passed`, **SANDYBRIDGE at one and at four threads `1 failed,
180 passed`**, on an exact-equality pin against the SAME frozen `L(0,0)`
literal as the briefed sibling, reading `rel 1.223e-08` against its own 1e-8.

The cause is the same and is worth stating plainly, because it is why an exact
pin was never going to hold: `L(0,0)` is a saddle amplitude with a stationary
phase of `|Phi| = 9.8883e+05` rad, so ONE ULP on a fit coefficient reaches the
answer as `eps |Phi| = 2.196e-10` -- and `md5(coef_phi)` differs on every
`OPENBLAS_CORETYPE` rung.  Measured `rel` against the frozen literal: HASWELL
1.838e-09, NEHALEM 6.653e-09, KATMAI 7.518e-09, **SANDYBRIDGE 1.223e-08**, WSL
4.398e-09, CI 8.190e-09 (the same digits on py3.14 and on the py3.12 JAX job,
so deterministic per build, not noise).  The constants were baked on one arm,
and the test was red on Windows at the 5.47.0 commit AND at the commit that
introduced the pin.

Both tests now read ONE shared bar derived once beside the constant
(`_Y2_L00_REL_BAR = 1e-6`, 82x over the worst arm), with the DERIVATION --
`want == old * (-1j) / (lambda sqrt|det J|)` -- kept as a separate
frozen-to-frozen claim at 1e-11 where the worst arm reads 9.86e-13.  Post-fix:
SANDYBRIDGE t1 and t4 and KATMAI t1 and t4 all `181 passed`.

That this was found at all is the argument for the ladder being mandatory: it
was invisible to CI (hidden behind `--maxfail`) and invisible on the default
local arm.

## 7. Runs

All with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` on the
command line, `--capture=sys`, `PYTHONPATH=/c/tmp/lum_reds`, and every tail
grepped for `passed|failed|error|no tests ran`.

### 7.1 The two builds

| run | Windows py3.14.6 / OpenBLAS Haswell | WSL py3.12.3 (Linux) |
|---|---|---|
| the seven affected files (c7, c8, M2 window contract, validity ranges, glass-registry meshgrid, b11 hygiene, gbd budget) | see 7.2 | **162 passed, 1 skipped in 690 s** |

The single WSL skip is a pre-existing `importorskip('refractiveindex')` in the
meshgrid file -- the glass extra is not in that venv.  The Linux arm is the
important one for RED 1: it is a different libm and a different BLAS, and the
restored order-10 stimulus manufactures the lobe there too, so the fix is not a
Windows artefact.

### 7.2 Per-file, Windows

| file | before | after |
|---|---|---|
| `test_niche_c7_ray_density_halo_check.py` + `test_niche_c8_inverse_support_bound.py` | 4 failed, 24 passed (55 s) | **28 passed** (73 s) |
| `test_pmm_m2_window_contract.py` (T3-1 id) | 7 arms pass, SANDYBRIDGE/4thr **fail** | 7 arms pass, SANDYBRIDGE/4thr **skip with the reading** |
| `test_v4_16_0_agent_d_validity_ranges.py` after the meshgrid file | 1 failed, 23 passed | **24 passed** (both orders, and alone) |
| `test_audit2609_b11_hygiene.py` | 82 passed | **85 passed** (137 s; +1 chain ratchet, +1 two-caller fixture, +1 shared scanner) |
| `tests/unit/test_wave5_gbd_dense_mem_budget.py` (new) | -- | **5 passed** (40 s) |

### 7.3 The kernel ladder

`OPENBLAS_CORETYPE` in {HASWELL, NEHALEM, KATMAI, SANDYBRIDGE} x threads {1, 4},
the kernel confirmed per arm with `threadpoolctl`.  ZEN aliases Haswell on this
box and SKYLAKEX crashes, so neither is used.

**Before the fix.**  The complete eight-arm ladder run against the unmodified
T3-1 test (`validation/probe_known_reds/ladder_t31/`): seven arms pass, and
**SANDYBRIDGE at four threads FAILS** with `assert 0 >= 1` and the screened
census quoted in section 2.1.  That is the red, and it is one arm in eight.

**After the fix.**  Three selections were run, and a reading is attributable to
its selection by its size alone, which is what makes the combined table safe to
read: 54 ids for the c7 / c8 / T3-1-file / GBD-budget selection, 163 for the
full seven-file selection, and 1 for the single T3-1 id.

| arm | T3-1 after | 54-id selection | 163-id selection |
|---|---|---|---|
| HASWELL 1 | pass | 54 passed | -- |
| HASWELL 4 | pass | 54 passed | 163 passed |
| NEHALEM 1 | pass | 54 passed | 163 passed |
| NEHALEM 4 | pass | 54 passed | 163 passed |
| KATMAI 1 | pass | -- | 163 passed |
| KATMAI 4 | -- | -- | 163 passed |
| SANDYBRIDGE 1 | pass | -- | 163 passed |
| **SANDYBRIDGE 4** | **skip with the reading** | **53 passed, 1 skipped** | -- |

Every one of the eight arms is covered, and the failing arm is covered twice:
as the single id (`ladder_t31_after/`) and inside the 54-id selection
(`ladder_itemD/SANDYBRIDGE_t4.log`), which reads

```
53 passed, 1 skipped
SKIPPED [1] tests\unit\test_pmm_m2_window_contract.py:2178: T3-1 spectral-decay
ladder: no device yielded a complete three-rung ...
```

-- the premise gate firing on exactly the arm it was derived for, with the
census in the message, while the other 53 ids of that selection pass.  Nothing
else in the selection changes behaviour on that arm.

A caveat on provenance, because it is the honest thing to record: two ladder
loops of this session outlived the shells that launched them (section 8) and
one of them was re-running the seven-file selection into the same directory, so
`ladder_itemD/` and `ladder_waveD/` each had two writers.  The readings above
are still attributable because the three selections have different sizes and a
pytest summary line names its own total -- a `163 passed` line can only have
come from the seven-file run and a `53 passed, 1 skipped` line only from the
54-id one.  The single-id `ladder_final/` and `ladder_t31_after/` logs had one
writer each.

**c7 / c8 / the GBD budget specifically**, from the uncontested
`ladder_final/` logs: HASWELL 1, HASWELL 4, NEHALEM 1, NEHALEM 4 and KATMAI 1
all read `33 passed`, and the 54-id and 163-id selections above cover the rest.
Plus the WSL (Linux, py3.12, numpy 2.4.6) run of 7.1 -- a different libm and a
different BLAS, and the arm that matters most for RED 1 since CI is Linux.  The
restored order-10 stimulus manufactures the lobe there too.

**What the ladder was NOT used for.**  `test_v4_16_0_agent_d_validity_ranges`,
`test_audit_w4_glass_registry_meshgrid` and `test_audit2609_b11_hygiene` are
registry-state, warn-once and AST checks with no floating-point kernel in them,
so a BLAS ladder measures nothing there; they are inside the 163-id selection
above and were also run on the default arm and under WSL.  The GBD budget test
reads `tracemalloc` allocation counts, which are kernel-independent by
construction -- it reads the same on every arm, as the table shows.

### 7.4 The consumer sweep of the swept modules

Every test file that imports `propagators.carrier`, `propagators.system`,
`propagators.gbd` or `propagators.carrier_field` was enumerated (105 files; 3
excluded because another agent held them mid-edit).  The subset that pins the
warning helpers directly -- `test_niche_c14_encapsulation.py` (which calls
`_warn_undeduped` with an explicit `stacklevel=1`),
`test_fix_grid_intent_override_2026_08_10.py` (which pins the `_guard_dispose`
route), `test_carrier_referenced.py` and `test_carrier_field.py` -- reads

```
109 passed, 5 warnings in 384.04s
```

which is the check that the `stacklevel=None` seam kept the explicit-integer
contract exactly.  A 20-file core sweep of the same consumer set then read

```
414 passed, 2 skipped, 49 deselected in 2391.91s (0:39:51)
```

(the two skips are pre-existing: CuPy is importable on this box but has no
functional CUDA device).  The broader 102-file sweep was started and then
stopped for CPU: see section 8.

### 7.5 The census / walker / dispatcher-pin / public-API / doc-consistency sweep

26 walker and pin files plus `test_audit_except_budget.py` and
`test_niche_audit_w4_input_kind.py`:

```
1 failed, 864 passed, 11 skipped in 470.44s
```

The single failure is `test_public_api.py::
test_installed_metadata_version_matches_source_version`, and it is
**pre-existing and environmental**: it compares the box's INSTALLED
distribution metadata (3.7.8, from the stale editable install at
`D:\...\Lumenairy`) against the source `__version__` (5.47.0).  It reproduces
identically on a pristine `git archive` tree of the base commit `96cb2096`, and
the test's own message says what it means ("The editable install is stale:
re-run `pip install -e .`").  Nothing in this branch touches it.

The eleven skips are all pre-existing documented exemptions (the cache-lock
exemption list) or walkers that correctly decline because the topmost versioned
CHANGELOG block carries no claim of the kind they check.

### 7.6 Walkers on the written documents

```
python scripts/check_source_line_citations.py    ok=107  drift=0  total=107
python scripts/record_history_fingerprints.py --check
                                                 OK: every history document matches its module.
python scripts/check_doc_identifiers.py          OK: every API-claiming backticked identifier resolves.
tests/unit/test_audit2609_a17_history_relocation.py + _a17_history_lint.py
                                                 757 passed in 146 s
wsl ruff check lumenairy/ tests/                 All checks passed!
```

The 13 citations that drifted are all in `carrier.py` and `system.py` and all
drifted because THIS branch moved those lines; they were re-anchored against the
base commit `96cb2096` with the repository's own `reanchor_since.py` (content
matching, not arithmetic), and the V18 walker then reads 107 of 107.

### 7.7 The CHANGELOG

A `## [Unreleased]` block was created above `## [5.47.0]` -- the file had none,
and there is exactly one now.  It carries no `path.py:N` citations, so it adds
nothing for V18 to chase, and the changelog walkers continue to read the topmost
VERSIONED block, which is unchanged.

## 8. What could not be established, and what was deliberately not done

1. **The access violation itself was not reproduced.**  Section 4 bounds it --
   the dense GBD loop's transient is up to 6.0x the budget the caller asked for,
   3 073 MB against a 512 MB request -- and gives the switch that makes the
   budget a bound.  It does not prove that this is what faulted.  A reproduction
   would need the fault to be caught under `faulthandler` with the allocation in
   the traceback, which did not happen in this window.

2. **The 102-file consumer sweep did not finish.**  It was started, then stopped
   to free CPU when the box reached 33-48 concurrent python processes (the
   maintainer's own multi-day `q2b_qwp.py` run, four other agents' test runs and
   this session's ladders).  The targeted 4-file subset that actually pins the
   swept helpers did finish, green (7.4).  Two of this session's own ladder
   shells outlived the shells that launched them and kept writing into
   `ladder_waveD/` and `ladder_itemD/`; reaping them was not permitted by the
   environment, so the final ladder was routed to a directory they do not know
   about and those two directories' logs must not be read.

3. **One transient red was observed and resolved, and is recorded so it is not
   re-discovered.**  A concurrent 78-file sweep read
   `tests/unit/test_niche_d8_congruence_workers.py` as 3 failed (pool clamp /
   spawn / drain order) while green on a pristine archive of the base commit --
   the signature of a real regression.  It was reading `carrier.py`
   MID-SWEEP, between the `warnings.warn` rewrite and the helper-signature
   rewrite, when the threaded `stacklevel` arguments and the helpers that
   consume them disagreed.  On the finished tree the same file reads
   **36 passed in 99.90 s**, and its own warning output names
   `test_niche_d8_congruence_workers.py` line 564 -- the caller's frame,
   through the congruence-worker path, which is the sweep working.  The lesson
   for the next wave is the one the handoff already records: a shared worktree
   with several agents in it makes any cross-file sweep a reading of a moving
   tree unless it is run from an archive.

4. **The sixteen other propagator modules were not swept** (5.3).  A static
   reachability screen flags them, but a screen is a possibility test; each needs
   its own two-caller measurement, and doing that blind is not the mechanical
   half of 4.4.  Their census is in `stacklevel_census_base.json`.

5. **No 3.10 or 3.11 interpreter exists on this box.**  `py -0p` lists 3.14.6 and
   3.13.13; WSL has 3.12.3.  The 3.11 digest fix (6.2) is therefore established
   by reproducing, on 3.12/3.13/3.14, the exact digests CPython 3.11 itself
   computed on the CI runners (49 of 49, bit for bit) rather than by running
   3.11.  The 3.10 arm is inferred from the same pre-PEP-701 tokenizer and is
   NOT measured.  Only a re-run of the un-masked matrix closes either.

6. **Two `test_audit2609_a9_verify_ui.py` failures** were observed in passing and
   **reproduce on a pristine `git archive` tree of the base commit** (`2 failed,
   47 passed`), so they are pre-existing and outside every file set in this
   package.  They do not appear in any downloaded CI log for run 34914295323,
   which may mean they are Windows-specific or may mean the py3.11 `--maxfail`
   abort hid them.  Flagged for whoever owns `lumenairy/ui/`.

7. **Decisions deliberately NOT taken, and reserved for the maintainer:**
   * whether `DENSE_MEM_BUDGET_ACCOUNTING` becomes `'measured'` by default (it
     moves default-path bytes by summation order; the measurement is in 4.3);
   * **whose diagnostic the in-glass gap-leg warnings are** (handoff 4.4's second
     bullet) -- making them name the user needs the lens to catch and re-emit at
     its own entry point, which is an ownership decision, not a substitution;
   * whether `SILICON` should gain a bundled Sellmeier row so it is usable
     without the glass extra (raised by the a8 work; a data/default change, not a
     red fix).
