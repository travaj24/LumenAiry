# CI KERNEL SWEEP -- 2026-09-11

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
| FAST gate, 2 shards, numerics files only (144 files matching pmm/rcwa/bor/eme/slant/staggered/mortar/conditioning/branch/sliver/jones/berreman/emt) | Nehalem | WSL | partial -- ~1,900 | **zero failures** |

The completed one is the most informative: **the entire slow gate runs clean
on a second kernel**, save the one test that is red on every kernel anyway.
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

## 3. The census: eight arms, 56 decisions

`validation/probe_ci_kernel_sweep/probe_decisions.py` takes the library's guard
**decisions** -- not readings -- on one arm and writes them as JSON;
`merge_arms.py` merges the arms into the committed
`validation/probe_ci_kernel_sweep/decisions.json`.  The census covers the sliver
refuse/return, the two 2-D mortar screens (rcond and residual), the per-layer
width band's warn/silent/refuse, the RCWA generalized interface's `T22`
refuse/accept, and the modal branch-cut orientation census.

Result over the eight arms: **54 of 56 decisions are unanimous.**  The two that
are not are both the same site, and they are the subject of the failing test
this sweep owns.

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

## 6. The new local instrument

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

## 7. Axes that were ruled OUT, and what could not be run

### Ruled out by measurement (so the CI divergence is none of these)

* **BLAS thread count.**  The fast `unit` lane runs BLAS unpinned; the sweep
  pinned every arm to one thread.  So the whole 56-decision census was re-taken
  at **2 and 4 threads** on WIN-Haswell: **zero decisions and zero hypothetical
  verdicts differ** from the 1-thread arm, and `max(R+T)` at the B1 fixture
  moves from 2.172858 to 2.172861 -- six figures, not two decades.  A separate
  1/2/4/8-thread ladder on the sliver decision alone also stays `refuse`
  throughout.  Threading is not the explanation for B1 on this host.
* **The build (interpreter + numpy version + OS).**  WIN py3.14.6 / numpy 2.4.4
  and WSL py3.12.3 / numpy 2.4.6 produce **bit-identical readings at every
  kernel** across all 56 decisions and every float in the census.  On this host
  the build is not an independent axis; the kernel is.
* **A Zen-specific BLAS kernel.**  Does not exist in these wheels (§1).

### Could not be run here

* **`SKYLAKEX`** -- unreachable on this host (AVX-512 SIGILL on both builds).
  A true SkylakeX arm needs Intel hardware.
* **numpy's OWN SIMD dispatch above X86_V3.**  OpenBLAS is not the only
  CPU-dispatched arithmetic in the stack: numpy dispatches its ufunc loops
  separately, and this host reports `baseline X86_V2, found X86_V3` -- i.e. AVX2
  but no AVX-512.  A CI EPYC of the Zen 4 generation or later reports
  **X86_V4** and therefore runs *different numpy loops* for every reduction,
  `abs`, `sum` and comparison in the closure computation -- including the
  `sum R + sum T` that the sliver guard's trigger is taken from.  This axis was
  NOT reachable here in either direction: the host cannot go up, and
  `NPY_DISABLE_CPU_FEATURES="AVX2 FMA3 F16C AVX"` did not change numpy 2.4.4's
  reported dispatch (still `found: X86_V3`), so it could not be driven down
  either.

  **This is now the leading hypothesis for B1**, having ruled out threads,
  build and BLAS kernel above.  It is also testable cheaply by whoever owns the
  runner image: print `np.show_runtime()`'s `simd_extensions` from a CI step and
  compare against `X86_V3`.  If the runner reports `X86_V4`, the sliver guard's
  energy trigger is being computed by a different set of ufunc kernels than any
  arm in this census, which is exactly the shape B1 needs.
* **A complete FAST-gate arm on a second kernel.**  The SLOW gate finished on
  Sandybridge (clean); the fast gate reached ~45 % on Katmai and the numerics
  subset ~68 % on Nehalem, both with zero failures, before the sweep's window
  closed.  See "The local full-gate sweeps" in §2 for the numbers and for why
  a full SSE2 fast gate needs a day.
