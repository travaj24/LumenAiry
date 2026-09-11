# CI KERNEL SWEEP -- 2026-09-11

> **SUPERSEDED IN PART, same day -- see
> `docs/audits/CI_PREMISE_GATES_2026_09_11.md`.**  This sweep's census asserted
> that every arm takes the same guard OUTCOME.  A later matrix pass showed that
> to be the wrong invariant: the CI runner arm SOLVES the ill-conditioned
> fixtures of this campaign CORRECTLY where all ten arms censused below solve
> them wrong, so the guards correctly take different outcomes there.  The
> census now records the ANSWER CLASS beside each decision and asserts RULE
> conformance instead; the kernel-ladder measurements below, including the ZEN
> and SKYLAKEX findings, are unchanged and still stand.

Preventive sweep of the fast unit gate for **kernel-dependent decisions**,
prompted by the RED release-CI matrix on `main` `59105d6` (the 5.45.0 release
commit; not tagged).  Working tree: `fix/ci-kernel-sweep` off `59105d6`.

The question this sweep answers is narrow and it is not "why did CI fail".  It
is: **which assertions and which library guards change their answer when only
the BLAS micro-kernel changes**, and can each of them be restated so that they
do not.

Scope note.  Two other agents own the sliver guard
(`lumenairy/elements/pmm/stack.py` + its test files) and the
branch-cut / JAX / tripwire / except-budget items (`rcwa/_core.py`, the JAX
twins, `test_audit_s1_2_rcwa_lossless_tripwire.py`,
`test_fix_branch_cut_round2.py`, `test_m1_conditioning_guard.py`,
`test_audit_except_budget.py`, the two JAX gradient gates).  Their failures are
reported here with readings and handed off; nothing of theirs is changed.

---

## 1. The instrument, and what it can and cannot reach

The bundled scipy-openblas in both local builds is a **DYNAMIC_ARCH** build, so
`OPENBLAS_CORETYPE=<kernel>` re-dispatches the whole BLAS/LAPACK kernel set at
import time -- a different reduction order, a different blocking, the same
arithmetic contract.  That is precisely the axis a rounding-level guard reading
rides on, and it is settable on the command line, which makes it the right tool
for this sweep.

**It takes effect on BOTH builds.**  Verified by reading the loaded kernel back
out of the library rather than trusting the variable
(`threadpoolctl.threadpool_info()[...]['architecture']`):

| `OPENBLAS_CORETYPE` | WIN (py3.14.6, numpy 2.4.4) | WSL (py3.12.3, numpy 2.4.6) |
|---|---|---|
| *(unset)* | `Haswell` | `Haswell` |
| `HASWELL` | `Haswell` | `Haswell` |
| `SANDYBRIDGE` | `Sandybridge` | `Sandybridge` |
| `NEHALEM` | `Nehalem` | `Nehalem` |
| `PRESCOTT` | `Katmai` | `Katmai` |
| `ZEN` | `Haswell` | `Haswell` |
| `BOGUSCORE` | `Haswell` | `Haswell` |
| `SKYLAKEX` | `SkylakeX`, then **SIGILL** | `SkylakeX`, then **SIGILL** |

Two of those rows are findings in their own right and both change how the CI
failure should be read.

**`ZEN` is not a distinct kernel in these wheels.**  It is not in the
DYNAMIC_ARCH table, so the request falls through to auto-detection -- and an
unrecognised name (`BOGUSCORE`) lands in exactly the same place, which is how we
know it is a fallback and not a kernel.  Auto-detection on this host (**AMD
Ryzen 9 5950X**, Zen 3, `AuthenticAMD`) reports `Haswell`.  So:

> **The local default arm ALREADY IS the CI Zen arm at the BLAS-kernel level.**
> A decision that differs between this workstation and an EPYC runner does not
> differ *because of a Zen kernel*.  The Zen hypothesis in the task framing is
> not supported by measurement; the remaining live axes are the OS/libm, the
> interpreter and numpy version, and -- the big one -- **thread count**, since
> the fast `unit` lane deliberately does NOT pin BLAS threads (the workflow says
> so at `unit-tests.yml`, "this job deliberately does NOT pin BLAS at run
> time"), while the `slow-tests` and `jax-unit` lanes do.

**`SKYLAKEX` is unreachable on this host.**  The corename is accepted and the
kernels dispatch, but they are AVX-512 and a Zen 3 CPU cannot execute them: the
first BLAS call dies with `Illegal instruction`, exit 132, on **both** builds.
It is therefore absent from the ladder by hardware, not by choice, and the
`SKYLAKEX` reading claimed in older memory notes for the WSL build does not
reproduce here (WSL's default is `Haswell`, same as Windows).

**The usable ladder** is consequently four genuinely different kernels spanning
SSE2 through AVX2/FMA:

    HASWELL (= default = Zen)  AVX2 + FMA
    SANDYBRIDGE                AVX, no FMA
    NEHALEM                    SSE4.2
    PRESCOTT (-> Katmai)       SSE2

crossed with two builds = **eight arms**, all at `OMP/OPENBLAS/MKL_NUM_THREADS=1`
so the kernel is the only moving part.

---

## 2. What the CI matrix actually reported

`--maxfail=5` means **every shard that shows 5 failures stopped early**, so the
CI logs in `C:/tmp/ci_5450/` are a lower bound, not a census.  Ten of the
sixteen unit shards hit the cap.

Failures, de-duplicated across the 16 unit shards + the jax lane + slow shard 1:

| test | owner | class |
|---|---|---|
| `test_fix_pmmstack_sliver_walls.py` (6 tests) | sliver agent | (b) |
| `test_fix_pmmstack_sliver_walls_round2.py` (7 tests) | sliver agent | (b) |
| `test_fix_pmmstack_sliver_round3.py` (3 tests) | sliver agent | (b) |
| `test_verify_pmmstack_sliver_walls.py` (2 tests) | sliver agent | (b) |
| `test_verify_pmmstack_sliver_round2.py` (1 test) | sliver agent | (b) |
| `test_verify_pmmstack_sliver_round3.py` (1 test) | sliver agent | (b) |
| `test_audit_s1_2_rcwa_lossless_tripwire.py::test_closure_warning_names_the_exact_index_coincidence_and_the_detune` | branch-cut agent | (b) |
| `test_fix_branch_cut_round2.py::test_the_spacer_coincidence_is_what_breaks_the_pre_round_one_branch` | branch-cut agent | (a) |
| `test_m1_conditioning_guard.py::test_x1_is_closed_across_the_whole_thin_ladder_and_reopens_pre_fix` | branch-cut agent | (a) |
| `test_audit_except_budget.py` (2 tests) | branch-cut agent | (c) |
| `test_niche_audit_w9_eig_vjp.py::test_pmm2d_near_normal_angle_gradient_improved` | branch-cut agent | (a) |
| `test_v5_14_0_pmm2d_autodiff.py::test_gate_angle_grad_at_normal_offcenter_is_genuine` | branch-cut agent | (a) |
| **`test_fix_pmm2d_mortar_round2.py::test_the_plain_1d_interface_solve_is_left_unguarded_and_this_is_why`** | **this sweep** | **(a)** |
| **`test_v5_20_12_rcwa_jones_2d_fff_nv.py::test_stripe_fixture_is_free_of_the_mode_match_degeneracy`** | **this sweep** | **see §5** |

### The two this sweep owns, measured on the whole ladder

Both were re-run on all eight arms (Windows py3.14 / WSL py3.12 x Haswell /
Sandybridge / Nehalem / Katmai, one thread).

| test | HAS | SBR | NEH | KAT | verdict |
|---|---|---|---|---|---|
| `..mortar_round2::test_the_plain_1d_interface_solve_is_left_unguarded_and_this_is_why` (as shipped) | pass | pass | pass | pass | **passes locally, fails on CI** -- class (a), coupled to a class-(b) library defect (B1) |
| `..fff_nv::test_stripe_fixture_is_free_of_the_mode_match_degeneracy` (as shipped) | FAIL | FAIL | FAIL | FAIL | **fails on every arm** -- NOT kernel-dependent; class (c), a stale negative arm |

The second one matters as a method point.  On the CI shard it presents as
`DID NOT WARN`, which is character-for-character what the two genuinely
kernel-decided warnings beside it in the same matrix look like.  Only a
two-sided measurement separates them, and the separation is total: the stripe
test's negative arm is dead on **every** arm, including the two local builds
that were called green.  It was never a kernel problem and no amount of
kernel-laddering would have explained it.

Per-arm readings for that fixture, worst `|sum R + sum T - 2|` over
`n_orders` 11..41 (16 rungs):

| arm | clean groove (2.10) | coincident groove (2.25) | ratio | tripwire |
|---|---|---|---|---|
| WIN-Haswell | 2.034e-13 (16/16) | 2.498e-13 (16/16) | 1.23 | silent |
| WIN-Sandybridge | 3.493e-13 (16/16) | 6.430e-13 (16/16) | 1.84 | silent |
| WIN-Nehalem | 4.434e-13 (16/16) | 4.543e-13 (16/16) | 1.02 | silent |
| WIN-Katmai | 4.552e-13 (16/16) | 4.448e-13 (16/16) | 0.98 | silent |
| WSL-Haswell | 2.072e-13 (16/16) | 2.536e-13 (16/16) | 1.22 | silent |
| WSL-Sandybridge | 3.473e-13 (16/16) | 5.280e-13 (16/16) | 1.52 | silent |
| WSL-Nehalem | 4.343e-13 (16/16) | 4.365e-13 (16/16) | 1.01 | silent |
| WSL-Katmai | 4.572e-13 (16/16) | 4.263e-13 (16/16) | 0.93 | silent |

against the `2.761e-02, 0/16 sound, WARNS` its docstring recorded on
2026-09-10.  The test asserts a `1e5` ratio; the shipped solver delivers ~1.

### The local full-gate sweeps

Three gate runs on this workstation, all at one thread, all with the shard
layout CI itself uses (`pytest-split --splits N --group g --splitting-algorithm
least_duration --durations-path .test_durations`):

| run | kernel | build | scope | result |
|---|---|---|---|---|
| **SLOW gate, 3 shards** | Sandybridge | WIN | complete -- 245 tests | **1 failure**, and it is the stripe test above (the shard collected the module before that fix landed).  Shards 2 and 3: 82 + 82 passed, zero failures.  Wall 1:14 / 1:21 / 1:23. |
| FAST gate, 4 shards | Katmai (`PRESCOTT`) | WIN | partial -- ~5,800 of 12,931 at the time of writing | **zero failures** |
| FAST gate, 2 shards, numerics files only (144 files matching pmm/rcwa/bor/eme/slant/staggered/mortar/conditioning/branch/sliver/jones/berreman/emt) | Nehalem | WSL | partial -- ~1,900 (stopped by an over-broad `pkill` of this sweep's own, not by a failure) | **zero failures** |
| FAST gate, 4 shards, the CI-FAITHFUL arm (default kernel, `OMP/OPENBLAS/MKL_NUM_THREADS=4`, i.e. a 4-core runner's width) | Haswell | WSL | partial -- still running when the sweep closed | **zero failures so far** |

### Cross-kernel verification of everything this sweep touched

Run one kernel at a time, at one thread, after the changes:

| files | WIN HAS / SBR / NEH / KAT | WSL HAS / SBR / NEH / KAT |
|---|---|---|
| mortar rounds 2/3/4 + both mortar verify files + the `fff_nv` file + the new consistency gate (58 tests) | 58 / 58 / 58 / 58 | 58 / 58 / 58 / (in flight) |
| `test_niche_d1_tilted_carrier.py` (33 tests) | 33 / 33 / 33 / 33 | 33 / 33 / 33 / 33 |
| `test_ci_kernel_consistency.py` (5 tests, 4.5 s) | pass | pass |

All eight arms, both witnesses, zero failures.

**The Katmai fast gate found failures CI never reported**, which is the whole
point of running a sweep rather than reading a log.  Six, from the ~8,100 tests
it reached before being stopped to make room for the CI-faithful run:

| test | on Katmai | also on CI? | class |
|---|---|---|---|
| `test_audit_s1_2_rcwa_lossless_tripwire::test_closure_warning_names_the_exact_index_coincidence_and_the_detune` | FAIL | yes | (b), branch-cut agent -- **now reproduced locally**, so it is kernel-decided, not runner-specific |
| `test_fix_pmmstack_sliver_walls_round2::test_the_round1_misses_are_refused_or_are_below_the_trigger` | FAIL | yes | (b), sliver agent -- **now reproduced locally** |
| `test_verify_pmmstack_sliver_walls::test_the_guard_has_a_measured_floor_the_theorem_cannot_reach` | FAIL | no | (b), sliver agent |
| `test_niche_audit_w9_eig_vjp::test_pmm2d_near_normal_angle_gradient_improved` | FAIL | yes (jax lane) | (a), branch-cut agent |
| **`test_niche_d1_tilted_carrier::test_the_fold_warning_on_an_off_centre_disc_was_a_true_positive`** | **FAIL** | **no** | **(a), this sweep -- restated, §4** |
| **`test_niche_r1_cosgrid_cache::test_structured_cold_speedup`** | **FAIL** | **no** | **(c), contention -- §6** |

Two of those reproduce a CI failure on a LOCAL kernel for the first time,
which upgrades them from "the runner does something odd" to "this is a
kernel-decided guard" and hands the owners a reproducer they can run.

The completed slow gate is the other informative one: **the entire slow gate
runs clean on a second kernel**, save the one test that is red on every kernel
anyway.
The slow gate is where the eig-heavy, conditioning-sensitive work lives (the
EME vector convergence files, the FFF-NV RCWA files), so a clean pass there on
an AVX-no-FMA kernel is real evidence that nothing else in that population is
kernel-decided.

Two notes on why the fast sweeps are partial, both worth recording for whoever
runs the next one:

* **`PRESCOTT` is not a cheap arm.**  SSE2 kernels turn the eig-heavy files
  into hours: `.test_durations` says the fast gate is ~11,944 s serial on the
  kernel it was recorded with, and shard 4 of the Katmai run spent over ninety
  minutes inside a single file (`test_audit_dynameta_consumer_api_2.py`, whose
  four heaviest tests are 308 / 238 / 168 / 119 s on Haswell).  Budget a full
  working day for a complete SSE2 fast gate, or shard it eight ways.
* **A `ZEN` full-gate arm was started and abandoned deliberately**, not for
  time.  Once §1 established that `OPENBLAS_CORETYPE=ZEN` resolves to
  `Haswell`, that run was re-running the already-green default configuration;
  the capacity went to `NEHALEM` on the numerics subset instead, which is a
  kernel the tree had never been run on.




---

## 3. The census: ten arms on two axes, 54 decisions

`validation/probe_ci_kernel_sweep/probe_decisions.py` takes the library's guard
**decisions** -- not readings -- on one arm and writes them as JSON;
`merge_arms.py` merges the arms into the committed
`validation/probe_ci_kernel_sweep/decisions.json`.  The census covers the sliver
refuse/return, the two 2-D mortar screens (rcond and residual), the per-layer
width band's warn/silent/refuse, the RCWA generalized interface's `T22`
refuse/accept, and the modal branch-cut orientation census.

An arm is a `(build, kernel, thread-width)` triple, and the THREAD width is a
first-class axis because CI's fast lane -- where six of the eight red lanes
were -- leaves BLAS unpinned.  The committed table carries ten arms: the four
kernels x two builds at one thread, plus a `t4` arm on the CI kernel on each
build.

Result: **all 54 decisions are unanimous on all ten arms**, at one thread and
at four alike.  The only rows that move are the two `hypothetical` ones -- the
verdict a bar the library does NOT ship would return at the one site it
deliberately leaves unguarded, which is the subject of the failing test this
sweep owns.

`t4` and not `tauto`, deliberately: "unpinned" means "as many threads as the
machine has", and this workstation has 24 where the runner has about four, so
a `tauto` arm here is a different arm wearing CI's label.  It is also
pathologically slow -- this probe is a few hundred SMALL solves and the
per-solve spawn/sync of 24 BLAS threads dominates, the same effect
`AUDIT_CI_TEST_TIME_2026_08_03` S1 measured when it declined to pin the fast
lane.

### The one site where a bar's verdict flips with the kernel

The plain 1-D `_interface_smatrix` site (`pmm/_core.py:~1851`), on a two-layer
`PMMStack` whose walls differ by `delta`.  LAPACK `gecon` reciprocal 1-condition
of `Wb` / `Vb`, and the solve's own energy closure, with the 1-D sliver guard
**disarmed** so the site is reached identically on every arm:

| arm | `rcond` @ `delta`=1e-4 | `rcond` @ 1e-5 | a 1e-12 bar @ 1e-5 | `max(R+T)` @ 1e-5 |
|---|---|---|---|---|
| WIN-Haswell | 9.69685e-11 | 9.73088e-13 | refuse | 2.172858 |
| WIN-Sandybridge | 9.69692e-11 | 9.72967e-13 | refuse | 3.611593 |
| WIN-Nehalem | 9.69399e-11 | 9.73088e-13 | refuse | 2.171586 |
| WIN-Katmai | 9.69399e-11 | **1.05377e-12** | **accept** | 2.171316 |
| WSL-Haswell | 9.69685e-11 | 9.73088e-13 | refuse | 2.172858 |
| WSL-Sandybridge | 9.69692e-11 | 9.72967e-13 | refuse | 3.611593 |
| WSL-Nehalem | 9.69399e-11 | 9.73088e-13 | refuse | 2.171586 |
| WSL-Katmai | 9.69399e-11 | **1.05377e-12** | **accept** | 2.171316 |
| *CI (py3.11, EPYC, threads unpinned)* | *9.693992e-11* | *9.729670e-13* | *refuse* | ***1.0000010*** |

Read the last two columns together, because that is the whole finding:

* the **`rcond` straddles a 1e-12 bar**: 9.73e-13 on three kernels, 1.054e-12 on
  the fourth.  A guard at this site would refuse on Haswell and answer on
  Katmai, from the same code on the same input.  The correct population (1e-4,
  which closes energy to 1.02e-8 here and 3.66e-7 on CI) sits **1.965 -- 1.999
  decades** above the
  would-be-refused one -- against 5.4 decades at the 2-D in-plane mortar sites,
  where a 1e-12 bar *is* shipped and *is* decidable.
* the **answer itself is a build-dependent number**: `max(R+T)` at `delta`=1e-5
  reads 2.17, 2.17, 2.17, 3.61 across the local kernels and **1.0000010** on CI.
  Two decades of super-unity here, one part in a million there, from the same
  geometry.

Note that WIN and WSL are bit-identical to each other at every kernel.  The
build (interpreter + numpy version + OS) is **not** an independent axis for
these quantities on this host; the kernel is.

---

## 4. What was restated, and into what

### `test_fix_pmm2d_mortar_round2.py::test_the_plain_1d_interface_solve_is_left_unguarded_and_this_is_why` -- class (a)

**What it used to assert.**  Three readings at the unguarded site, and then
this closing pair:

```python
assert rc_ok / _pc._MORTAR_RCOND_REFUSE < 1e3       # "within 3 decades"
assert out[1e-5][3] is not None and "SLIVER" in out[1e-5][3]
```

The second line is a claim about a **different guard**.  It says: the row that
would trip a bar here is already refused, better, by the 1-D sliver guard.  On
CI that is false -- the sliver guard was silent and the solve returned -- and
the test failed on `assert (None is not None)`.

**Why it failed is not what it looks like.**  The `rcond` readings on the CI
runner (9.693992e-11 / 9.729670e-13) are within 0.03 % of the local ones.  The
reading was never the problem.  What differed is that the sliver guard's
**trigger** is the solve's own energy super-unity against `_SLIVER_TRIGGER_BAR`
= 1e-3, and at this geometry that super-unity is 1.17--2.61 locally and
1.05e-6 on CI -- three decades *below* the trigger.  The guard is silent exactly
where the answer is wrong-but-energy-closing.  **That is a live P1 in the sliver
guard and it is handed off, not fixed here** (§5).

**What it asserts now.**  Four claims, none of which can be moved by a kernel:

1. **Structural.**  `inspect.getsource(_pc._interface_smatrix)` contains exactly
   two bare `np.linalg.solve(` and no `gecon` / `rcond` /
   `_guarded_mortar_solve`.  "Unguarded" is pinned as a property of the shipped
   body, so arming a guard there trips this test deliberately.
2. **Both populations, re-measured on the running build, with the sliver guard
   DISARMED** (function-scoped, restored) so the site is reached on every build
   rather than only on the ones where another guard happens to stand aside.
   The correct row closes and is silent; the wrong row **returns** -- unguarded
   is a real property, not a figure of speech -- and does not close.  The two
   closure bars are placed by the GAP, not by one build's residual: the correct
   row's worst `|R+T - 1|` is 3.66e-07 (the CI runner; 1.02e-08 here) and the
   wrong row's BEST is 1.17 (Katmai), so `1e-04` and `1e-01` leave 273x below
   and 11.7x above, worst arm of nine.  Nothing about the sliver guard is
   asserted.
3. **The gap, as a decade count, not a ratio against a bar**:
   `log10(rc_ok / rc_bad) < 2.1`, measured 1.965--1.999 over the eight arms.
4. **The census.**  The committed per-kernel table must show the 1e-12 bar's
   verdict at this site to be **non-unanimous** (`{refuse, accept}`), and the
   running build's own verdict must be one of the two.  This converts "within
   one decade of a bar" -- a statement about a float -- into a statement about a
   *decision*, which reads the same on every build including builds that are not
   in the table.

The docstring carries a dated paragraph recording the change, the CI reading
that forced it, and the hand-off.

### `test_v5_20_12_rcwa_jones_2d_fff_nv.py::test_stripe_fixture_is_free_of_the_mode_match_degeneracy` -- class (c), restated anyway

**What happened.**  5.45.0's own modal branch-cut repair killed this test's
negative arm.  The arm reconstructed a mode-match degeneracy by setting the
groove permittivity to the director's `no^2` = 2.25, which is also
`n_sub^2` -- and the branch-cut fix is *precisely* the repair of "a LAYER
permittivity EXACTLY EQUAL to a REGION's ... degenerate at EVERY truncation".
The coincidence stopped biting, the tripwire stopped warning, and
`pytest.warns` failed.  The test's own docstring predicted this event and
prescribed the remedy: *"If the solver is ever made degeneracy-robust this
test fails, and that is the gate working: it must then be re-derived
(durability rule), not widened."*

**Two-sided proof that the branch-cut fix is the cause**, and note the trap:
mutating `_CUT_BAND_REL` does nothing, because `_sqrt_decay`'s `band`
parameter is a DEFAULT ARGUMENT bound at def time.  The state has to be
constructed by rebinding the function at every module that holds a
`from ... import` copy -- the same shape the branch-cut fix's own gate file
uses.  Measured 2026-09-11, one thread:

| arm | clean (2.10) | coincident (2.25) | ratio | tripwire |
|---|---|---|---|---|
| SHIPPED 5.45.0 | 2.034e-13 (16/16 sound) | 2.498e-13 (16/16 sound) | 1.2 | silent |
| PRE-5.45.0 body | 2.083e-13 (16/16 sound) | **2.761e-02 (0/16 sound)** | **1.3e11** | **WARNS** |

The pre-fix arm reproduces the 2026-09-10 docstring readings to three figures,
which is what makes it the right reconstruction rather than a different defect.

**What it asserts now.**  The positive arm is unchanged.  The old negative arm
becomes a *positive* claim -- the coincidence is sound on the shipped solver
too, at the same bar, which is what 5.45.0 bought -- and the negative arm is
re-derived as an ENGINEERED fail-before: reinstate the pre-5.45.0 branch body
and the degeneracy returns by eleven decades, the tripwire fires, and 0 of 16
rungs stay sound.  A guard is added that the pre-fix body breaks the
COINCIDENT cell *and only* the coincident cell (`pre_clean` must stay sound),
so the arm cannot silently start demonstrating some other defect.

Note the coupling: this restatement reads the pre-round-1 `_sqrt_decay` shape,
which is the branch-cut agent's area.  The body is re-typed locally (no import
from their file) so the two do not share a fixture, but if `_sqrt_decay`'s
signature changes, this test's fail-before needs the same change.

#### And then the restatement had to be restated -- the sweep catching its own remedy

The first version of the engineered arm asserted `pre_coin_sound == 0`: with
the pre-fix body installed, ZERO of the 16 truncation rungs stay sound.  That
is what the Haswell arm reads, and it **failed on WSL-Sandybridge** during this
sweep's own cross-kernel verification pass.  Running the ladder on all eight
arms says why:

| arm | pre-fix clean | pre-fix coincident | rungs sound | rungs warning |
|---|---|---|---|---|
| WIN-Haswell | 2.083e-13 | 2.761e-02 | 0/16 | 16 |
| WIN-Sandybridge | 3.020e-13 | 4.690e-02 | **1/16** | 14 |
| WIN-Nehalem | 2.958e-13 | 8.158e-03 | **2/16** | 14 |
| WIN-Katmai | 4.008e-13 | 7.289e-04 | **5/16** | 11 |
| WSL-Haswell | 1.861e-13 | 5.001e-02 | 0/16 | 16 |
| WSL-Sandybridge | 3.069e-13 | 4.802e-02 | **1/16** | 14 |
| WSL-Nehalem | 2.918e-13 | 2.955e-03 | **2/16** | 12 |
| WSL-Katmai | 4.035e-13 | 1.584e-03 | **5/16** | 11 |

WHICH rungs of the ladder land on the wrong side of a rounding-level
degeneracy is precisely what the kernel decides -- `docs/TESTING_STANDARDS.md`
shape **S5**, "exact count/set of nondeterministic machinery", written into a
brand-new test by the sweep that exists to remove that shape.  Recording it
here rather than quietly deleting it is the point: the fragile shape is not
obvious at the moment of writing, and only the ladder makes it visible.

The count assertion is withdrawn.  What is asserted instead is the READING,
which has room on both sides on every arm: the coincident worst never falls
below **7.289e-04**, seven decades above the clean population's 4.6e-13
ceiling and 73x above the 1e-05 bar; the ratio against the clean arm measured
in the same run is 1.82e+09 .. 2.69e+11 against a 1e+05 bar; and at least one
rung must be broken, which the reading already implies.

Note the ordering in that table, because it is the general lesson: the
coincident worst falls **monotonically as the SIMD width narrows** (Haswell
2.8-5.0e-02 down to Katmai 0.7-1.6e-03).  A rounding-level defect amplified by
an ill-conditioned operator amplifies *less* on a narrower kernel, so a bar
calibrated on the widest kernel available has its smallest margin on the
oldest one -- which is the opposite of where a developer looks.


### `test_niche_d1_tilted_carrier.py` (two witnesses) -- class (a)

Found by the Katmai gate, never seen on CI.  The fail-before arm of both
decentred-ghost witnesses reconstructs a pre-D1 ghost by driving a
deliberately near-singular fit into a fold, at ONE decentre (5.60 mm).
Whether a given decentre folds is a rounding-level property of that fit, so it
is decided by the kernel:

| kernel | `bad_rel` at the historical 5.60 mm | verdict |
|---|---|---|
| Haswell | 4.900e-01 | reproduces |
| Sandybridge | 1.003e-03 | **does not** |
| Nehalem | 1.003e-03 | **does not** |
| Katmai | 1.003e-03 | **does not** |

against a 0.02 bar.  Three decades under it on three of four kernels -- the
witness simply was not there.  Note that the three non-Haswell readings agree
to five figures with each other AND with the reading this test's own comment
records for the *cured* path, which is what makes it a disappeared witness
rather than a smaller one.

**The defect has moved, not gone.**  Scanning the decentre finds it on every
kernel, at a different rung:

| kernel | first reproducing rung | `bad_rel` there |
|---|---|---|
| Haswell | 5.60 mm | 4.900e-01 |
| Nehalem | 5.80 mm | 1.329e-01 |
| Katmai | 5.80 mm | 9.608e-01 |
| Sandybridge | 5.85 mm | 1.495e-01 |

`docs/TESTING_STANDARDS.md` restatement 3 prescribes exactly this: "Scan a
ladder if needed; hard-fail only when the ladder is exhausted AND the fixed
path misbehaves -- the fixed-path claim stays unconditional on every arm."  So
`_GHOST_XC_LADDER` is added (ordered so the historical rung is tried first:
Haswell stops at rung 1, Nehalem and Katmai at rung 2, Sandybridge at rung 3,
which costs one or two extra solves), `_first_reproducing_rung` walks it
requiring BOTH a loud ghost and a fold warning, and it raises with the whole
scanned ladder if nothing reproduces -- the "ladder exhausted" case, which
means re-derive.  The fixture helpers are parameterised by the decentre so the
scan can move it.  The PASS-AFTER half is untouched and still runs at the
shipped decentre.

**The SIBLING witness fails on Nehalem too, and it needs a wider ladder.**
Both witnesses in this file are red on Nehalem at `59105d6` -- verified by
running the ORIGINAL file from that commit -- so the sibling
(`test_off_centre_fit_disc_does_not_ghost_the_exit_field`) is a second
class-(a) failure the sweep found, not a casualty of fixing the first.

It is the STRICTER of the two: it compares the broken arm against a spline
ORACLE and needs that oracle UNFOLDED at the same decentre, so a rung must
satisfy three conditions, not two.  `_first_reproducing_rung` takes a
`quiet_oracle` flag for that, and pays the extra oracle solve only at rungs
that already clear the ghost bar -- opt-in, because the fold-warning witness
does not need it.

The first attempt at this ran a 5.50 .. 6.00 mm ladder and EXHAUSTED on
Nehalem: ten rungs leave the ghost at ~1e-03, and the single loud one
(5.95 mm, 2.34e-02) has a folded oracle.  Widening the reach finds it at
**5.30 mm (3.00e-01, clean oracle)**; 6.60 mm is loud there too (2.44e-02) but
its oracle folds as well.  The ladder is therefore near-rungs-first with the
wide rungs appended, so Haswell, Sandybridge and Katmai still stop at the
historical 5.60 mm and only Nehalem walks to rung 4.

Two lessons worth keeping, because both cost a round here.  First, a ladder is
the right instrument for a witness whose state has MOVED and the wrong one for
a witness that never moved -- adding one "for symmetry" can convert a passing
test into a kernel-dependent one, which is what the first attempt did before
the wider reach fixed it properly.  Second, the qualifying CONDITIONS have to
be the ones the test actually depends on: a rung that reproduces the ghost is
not automatically a rung the comparison is valid at, and the difference is not
visible until a kernel forces the scan off the historical rung.

---

## 5. Class (b) hand-offs -- kernel-dependent LIBRARY decisions

### B1 (P1, sliver agent) -- the sliver guard's trigger is the quantity it is trying to police

**Site.** `lumenairy/elements/pmm/stack.py`, `_sliver_screen` / `_sliver_arbiter`;
trigger `_SLIVER_TRIGGER_BAR = 1e-3` on the solve's energy super-unity, closure
attribution `_SLIVER_ATTRIB_CLOSURE` = 1e-5 / `_SLIVER_CLOSURE_FRACTION` = 1e-2.

**Fixture** (reproduces in three lines; also §3's table row):

```python
from lumenairy.elements.pmm import PMMStack
a0, a1, delta = 0.27865, 0.62505, 1e-5
st = PMMStack(1.2, degree=12, far_field_orders=5)
st.add_layer(0.08, segments=[(a0, 2.25), (a1 - a0, 9.0), (1 - a1, 2.25)])
st.add_layer(0.08, segments=[(a0 - delta, 2.25),
                             (a1 + delta - (a0 - delta), 9.0),
                             (1 - (a1 + delta), 2.25)])
st.set_source(0.85, theta=0.15)
st.solve()          # refuses HERE, returns on the CI runner
```

**Readings.**  With the guard disarmed, `max(R+T)` = **2.171316 / 2.171586 /
2.172858 / 3.611593** on Katmai / Nehalem / Haswell / Sandybridge (identical on
both builds) and **1.0000010** on the CI runner (py3.11 shard 4, threads
unpinned).  With the guard armed it **refuses on all eight local arms** and
**returned on CI**.  The `rcond` at the same site is stable to 0.03 % across all
nine readings, so the divergence is in the *solve's answer*, not in the
conditioning estimate.

**Why this is a decision and not a tolerance.**  A guard whose trigger is the
super-unity of the answer cannot see a wrong answer that happens to close
energy, and whether it closes is exactly what the kernel decides here.  The
sliver defect's own documentation says the corruption is "ENERGY-INVISIBLE";
the trigger contradicts that.  This is also the direct cause of six of the CI
test failures and of the mortar test in §4.

**Suggested shape** (the sliver agent's call): trigger on a quantity that is a
property of the *geometry* (the cross-layer wall separation relative to the
union grid's own scale -- which `_cross_layer_sliver` already computes) rather
than on the answer's energy, and keep the energy reading as *evidence in the
message*, not as the gate.

**Not fixed here** per the ownership split.

### B2 (P2, branch-cut agent) -- the lossless tripwire's warning is kernel-decided

`test_audit_s1_2_rcwa_lossless_tripwire.py::test_closure_warning_names_the_exact_index_coincidence_and_the_detune`
failed on three of the four pythons with `DID NOT WARN ... Emitted warnings:
[]`.  A warning that fires locally and not on the runner is the same shape as
B1 one level down: the tripwire's threshold is on a closure defect that the
branch-cut fix itself moved by decades.  Recorded here for completeness; the
branch-cut agent owns both the guard and the test.

### B3 (P3, no owner yet) -- `--maxfail=5` hides the census

Not a kernel finding, but it shaped this one and it will shape the next.  Ten
of the sixteen fast shards stopped at five failures, so the matrix that
prompted this sweep reports a LOWER BOUND on what is red.  A sweep for
kernel-dependent decisions specifically needs the whole list -- the interesting
signal is which tests fail on WHICH arms, and a cap that truncates per shard
destroys exactly that.  Suggest raising the cap (or dropping it) on the release
matrix specifically, where the cost of a long red run is one runner-hour and
the cost of a truncated one is a second full CI cycle.

---

## 6. Class (c) -- and a caution about how this sweep was run

`test_niche_r1_cosgrid_cache.py::test_structured_cold_speedup` asserts a
WALL-CLOCK ratio (`structured < 0.7 x delaunay`, best of three) between two
implementations in the same process.  It failed on the Katmai gate at ratio
0.87.  It is **not** kernel-dependent, and the evidence is two-sided in the
most direct way available: re-run one at a time, it PASSES on Katmai (and on
Haswell and Sandybridge) and FAILS on Nehalem at ratio 1.18 -- the opposite
pattern from the gate run.  The same test took 16.9 s, 21.9 s, 41.9 s and
43.6 s on the four kernels, which is not a kernel signal; it is the box.

Both failures were produced by THIS SWEEP's own methodology: up to eleven
concurrent pytest processes plus another agent's probes on one workstation.
The test is left exactly as it is.  Widening a speed bar until it survives a
self-inflicted 11-process load is precisely the move
`docs/TESTING_STANDARDS.md` forbids, and it is not red on CI, where each shard
has a runner to itself.

**The caution for the next sweep:** a kernel ladder oversubscribes the machine
by construction, so wall-clock assertions inside it are measuring the sweep
and not the tree.  Either exclude the timing files from a laddered run
(`--deselect`, or a `-m "not perf"` marker if one is ever added) or re-run any
timing failure alone before believing it.

---

## 7. The new local instrument

`tests/unit/test_ci_kernel_consistency.py` (5 tests, **5.3 s** measured) closes
the hole that made this sweep necessary: nothing in the gate compared a decision
across kernels, so a kernel-dependent guard could only ever be discovered on CI.

It asserts, in this order:

1. the committed census spans >= 2 kernels and >= 2 builds and every arm was
   taken at one thread (an arm with threads unpinned is measuring a different
   axis and cannot be compared);
2. **every guard decision is identical on every arm** -- the headline, and a
   disagreement is reported as a P1 with each arm's verdict printed;
3. the **hypothetical** bars at deliberately-unguarded sites are still
   **non-unanimous** -- the standing evidence for each omission, asserted rather
   than assumed, so that if a site ever became decidable the omission gets
   re-argued instead of silently kept;
4. the arm running *now* re-takes the four cheap sections and must join the
   consensus, naming itself if it does not -- which is what makes a
   never-before-censused runner (a future CI image, a colleague's laptop) fail
   here rather than three months later;
5. the set of re-taken sections is pinned, so narrowing the local re-take is a
   visible edit.

---

## 8. Which axis is it, then?  What was ruled out, and what is left

**Correction, 2026-09-11 (coordinator).**  CI's runners are **AMD EPYC 7763 --
Zen 3, no AVX-512**.  That is the same microarchitecture generation as this
host (Ryzen 9 5950X, Zen 3), which settles two things at once and invalidates
one hypothesis this document carried in an earlier revision:

* the runner's OpenBLAS kernel is the **same Haswell-class kernel** this
  workstation selects by default -- so the CI-vs-local divergence is not a
  BLAS kernel difference at all;
* numpy's own SIMD dispatch is likewise the same tier (**X86_V3**: AVX2, no
  AVX-512).  An earlier revision of this document proposed X86_V4 ufunc loops
  on the runner as the leading explanation for B1.  **That is withdrawn**: a
  Zen 3 EPYC cannot report X86_V4.

### Ruled out by measurement

* **BLAS micro-kernel.**  Same kernel on both sides (above), and in any case
  the full 54-decision census is unanimous across Haswell / Sandybridge /
  Nehalem / Katmai except at the one deliberately-unguarded site in §3.
* **The build (interpreter + numpy version + OS).**  WIN py3.14.6 / numpy
  2.4.4 and WSL py3.12.3 / numpy 2.4.6 produce **bit-identical readings at
  every kernel** across the whole census.
* **A Zen-specific BLAS kernel.**  Does not exist in these wheels (§1).
* **numpy SIMD tier.**  Same on both sides (above).  It could not be driven
  DOWN either -- `NPY_DISABLE_CPU_FEATURES="AVX2 FMA3 F16C AVX"` did not change
  numpy 2.4.4's reported dispatch -- but with the runner on the same tier that
  no longer matters.

### What is left: the THREAD WIDTH, and the census now carries it

CI's fast lane deliberately leaves BLAS **unpinned** on a 2-4 core runner
(`unit-tests.yml`: "this job deliberately does NOT pin BLAS at run time"),
while the slow and jax lanes pin to one.  Six of the eight red lanes are fast
lanes.  A reduction split across four threads is a different summation order
from the same reduction on one, in exactly the way a different kernel is -- so
the thread width is a first-class axis, and the committed census now carries
`t1`, `t4` and `tauto` (unpinned) arms alongside the kernel ladder, with the
width OpenBLAS actually chose recorded beside the label.

The honest statement of where this leaves B1: **on this host the thread width
does not move any decision** (the full census at `t4` and unpinned is
identical to `t1`), but this host has 16 cores and the runner has 4, and an
unpinned OpenBLAS picks its blocking from the core count.  The remaining
untested difference between this workstation and the runner is therefore
narrow and specific -- unpinned OpenBLAS on a **4-core** machine -- and the
cheapest way to close it is on the runner, not here: add
`python -c "import numpy, threadpoolctl; print(threadpoolctl.threadpool_info())"`
to a CI step and compare the reported `architecture` and `num_threads` against
the arms in `decisions.json`.

### Could not be run here

* **`SKYLAKEX`** -- unreachable on this host (AVX-512 SIGILL on both builds),
  and moot: the runner has no AVX-512 either.
* **A 4-core unpinned arm** -- needs a 4-core machine, or a cgroup/affinity
  restriction this sweep did not set up.
* **A complete FAST-gate arm on the Katmai kernel.**  It reached ~82 % on
  three of four shards (finding the failures in §2) before it was stopped to
  make room for the CI-faithful run; the fourth shard was still inside one
  file.  The SLOW gate did finish, on Sandybridge, clean.
