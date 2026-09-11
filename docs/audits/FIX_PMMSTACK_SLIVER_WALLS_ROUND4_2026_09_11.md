# O-11 ROUND 4 -- the `PMMStack` sliver guard decides on ANSWERS, not on an energy total (2026-09-11)

Subject: the near-coincident-wall (sliver) guard in
`lumenairy/elements/pmm/stack.py`, as it stood at the 5.45.0 release commit
`59105d6` (rounds 1-3, all merged, the release NOT yet tagged).

Trigger: the release CI matrix for that commit was RED. Twenty distinct tests
of the sliver family failed, on different pythons for different tests, and
every one of them failed on the same shape of claim.

Branch: `fix/pmmstack-sliver-guard-round4`, worktree `C:/tmp/lum_sliver4`.
Probes and per-arm JSON: `validation/probe_fix_sliver_round4/`.

---

## S1. Vocabulary, stated once

Every term below is used in exactly this sense for the rest of the document.

**Manufactured cell.** `PMMStack` solves every layer on the UNION of all
layers' wall coordinates. A union cell is MANUFACTURED when its two walls
share no owning layer -- geometry no single layer asked for. A thin feature
inside one layer owns both its walls and is never manufactured.

**Sliver.** A manufactured cell at least `_SLIVER_OWN_SCALE_RATIO` = 100 times
finer than the finest wall spacing any single layer did ask for. Two adjacent
slices of a taper whose walls differ by `delta` of the period put a sliver of
exactly that width on the union grid.

**The mechanism.** The spectral element carries the Jacobian `J = w P / 2`, so
the nodal `Kx^2` grows as `1/w^2`, the layer's modal spectrum acquires
spurious wavenumbers `|q| ~ 0.65 N(N+1)/4 / (k0 J)`, and the interface
mode-match conditions as `1/w^2`. Past a degree-dependent onset the cascade
returns a wrong answer.

**The continuity classification.** The structure is continuous in `delta`, so
the exact answer moves linearly with it. Scored against the exact `delta -> 0`
solve of the same device: `err <= 10 delta` is RIGHT, `err > 100 delta` is
WRONG, between is GREY. Round 4 takes `err` over BOTH incident polarizations;
see S6.4 for why that correction matters and what it moves.

**The reading.** `max R+T - 1` of a solve. Rounds 1-3 gated the guard on it
(`_SLIVER_TRIGGER_BAR`) and attributed with it (`_SLIVER_ATTRIB_CLOSURE`,
`_SLIVER_CLOSURE_FRACTION`). This document's subject is that it is not a
property of the library.

**An arm.** One `(build, OpenBLAS kernel)` pair. `build` is Windows
python 3.14.6 / numpy 2.4.4 or WSL python 3.12.3 / numpy 2.4.6; `kernel` is
what OpenBLAS actually dispatched to, which is not always what
`OPENBLAS_CORETYPE` asked for (S4).

**`d0`, `d12`.** The two quantities the round-4 arbiter compares.
`d0 = |A_sliver - A_M|` is the largest per-order efficiency difference between
the solve as given and the solve on the PRESCRIBED grid
`min_feature = 2 w_wide P` -- the shipped `_sliver_answer_move`, unchanged.
`d12 = |A_L - A_L'|` is the same statistic between two solves that carry NO
sliver: the colliding walls CLOSED onto one coordinate, and that closed wall
DISPLACED by one widest manufactured cell. `d12` is therefore this device's
own answer change for a wall displacement of exactly the sliver's size.

---

## S2. The evidence: what the 5.45.0 CI matrix measured

Sixteen shards (python 3.10-3.13 x four shards), logs in `C:/tmp/ci_5450/`.
Twenty distinct sliver-family tests failed. They divide into three classes,
and all three are the same defect.

### S2.1 The premise class: a named row does not read what the test says

| test | what it asserted | what CI measured |
|---|---|---|
| `..._round2::test_the_round1_misses_are_refused_or_are_below_the_trigger` | `(20, 1.71299e-06)` reads `R+T <= 1.01` | **3.49325** (py3.10), **2.11822** (py3.11), **2.21342** (py3.12 and py3.13) |
| `..._walls::test_fail_before_the_pre_fix_path_returns_a_wrong_energy_violating_answer` | `(14, 1e-4)` is far from the physical shift AND `R+T - 1 > 1.0` | `err` = **1.148e-04**, i.e. **1.15x the shift -- CORRECT**, at `R+T` = **1.000115** |
| `..._walls::test_fail_before_the_pre_fix_path_only_warns_it_does_not_refuse` | that row emits `energy not conserved` | **no warning at all** (the empty list in the assertion) |
| `..._round2::test_the_guard_now_reaches_a_liquid_crystal_sliver` | the in-plane director row reads above the trigger | **1.0000001290987883** |
| `..._verify_walls::test_a_thin_feature_owned_by_one_layer_is_exempt...` | some degree of the owned liner reads SUB-unity | min over degrees **1.1298** |

### S2.2 The did-not-raise class: the plumbing test fails on another row's physics

`..._round2::test_the_sweep_arbitrates_at_its_own_wavelength_not_a_stale_set_source`,
`..._round2::test_the_sweep_arbitrates_the_same_way_at_any_worker_count`,
`..._round2::test_the_prepared_path_arbitrates_at_the_wavelength_it_was_given`,
`..._walls::test_the_refusal_names_the_geometry_and_the_two_remedies`,
`..._walls::test_the_wavenumber_the_message_quotes_is_the_one_the_solve_actually_has`,
`..._verify_walls::test_the_m1_module_disarm_is_function_scoped_and_restores`
all assert `pytest.raises(...)` on the NAMED row `(14, 1e-4)`. Every one of
them is really about plumbing -- which wavelength the arbiter is handed, what
the message says, whether a module fixture restores -- and every one of them
failed because that row is CORRECT on CI's kernel and is therefore returned.

### S2.3 The wrong-answer class: the guard returned answers it should not have

This is the half that is a defect in the LIBRARY and not only in the tests.

* `..._round3::test_the_d5_rows_are_refused_and_the_refusal_names_min_feature`
  failed on py3.12 and py3.13 shard 1 with `[(6.8726e-06, 1475.4362439811316,
  0.0019313052743061148)]`: the row is **1475x the physical wall shift as
  returned** and **0.0019x on the prescribed grid**, and round 3 RETURNED it.
  It did so because its criterion is the DROP factor
  `(worst - 1) / su_snapped`, and on that kernel the drop fell under the 100
  the criterion demands. The same row's drop is 5,166 on this box.
* `..._verify_round2::test_the_closure_criterion_is_relative_not_absolute`
  recorded the same row from the other side: `R+T` = 1.0101734731052092
  returned against 1.0000372944595628 snapped.
* `..._round3::test_the_closure_fraction_separates_the_two_drop_populations`
  failed with `[272.78778736769146]`: a CORRECT row whose drop is 272.8, i.e.
  ABOVE the bar the correct population is supposed to sit below.

So the closure crosses the correct population on one kernel and the wrong
population on another. Both directions, in one matrix.

### S2.4 The arbitration class: the verdict at a named row is a kernel fact

`..._round2::test_the_arbiter_separates_the_two_causes_on_this_build` and
`..._round3::test_the_round2_fixtures_arbitrate_identically_under_the_relative_closure`
both failed at `(14, 1e-4)` with verdict `truncation` and evidence
`snapped_super_unity` between 2.2e-16 and 3.1e-14, `move` between 3.07e-05 and
3.12e-05 against `w_wide` = 1e-04 -- i.e. `move/w_wide` about **0.31**, four
decades under the 100x bar. That verdict is CORRECT on that kernel: the sliver
did not move that answer there. The test was wrong to name the row.

---

## S3. Why the reading is a property of the kernel, measured

The interface mode-match conditions as `1/w^2`, so the answer in the hazard
band is round-off amplified by a condition number that reaches 1.2e+09 at
`w` = 1e-4. Which side of the onset a given row lands on is therefore a
property of the arithmetic, not of the geometry. Measured on this box with
`OPENBLAS_CORETYPE`, the O-11 fixture (period 1.2 um, wl 0.85 um, theta 0.15,
walls 0.27865 / 0.62505, degree 14, `delta` = 1e-4 of the period, guard
disarmed, one thread):

| requested | kernel dispatched | `max R+T` | `err / delta` | continuity |
|---|---|---|---|---|
| HASWELL | Haswell | 2.17298 | 4789 | WRONG |
| ZEN | Haswell (aliased, S4) | 2.17298 | 4789 | WRONG |
| PRESCOTT | Katmai | 2.17297 | 4789 | WRONG |
| NEHALEM | Nehalem | 3.61242 | 9301 | WRONG |
| SANDYBRIDGE | Sandybridge | 1.00000 | **1.155** | **RIGHT** |
| CI's own kernel | (GitHub `ubuntu-latest`) | 1.000115 | 1.148 | **RIGHT** |

and the same table at `(20, 1e-4)` runs the other way -- Haswell RIGHT
(`err/delta` = 1.153), Prescott WRONG (4789). The full five-kernel x
three-degree x five-delta grid is `validation/probe_fix_sliver_round4/`'s
first measurement and is reproduced in S7.

Two consequences, and they point in opposite directions:

1. A test that NAMES a row and asserts a reading, a verdict or a refusal is
   pinning a kernel. That is the whole of S2.1, S2.2 and S2.4, and it is a
   defect in the tests.
2. A GUARD that keys on the reading inherits the same property. That is S2.3,
   and it is a defect in the library.

It also bounds what any repair can promise. In the conditioning-collapse band
the ANSWER is a property of the kernel, so no guard that looks at the answer
can make an identical decision on every kernel -- and a guard that made one
would have to be ignoring the answer, which is the defect being repaired.
What round 4 can guarantee, and what S7 pins, is that the guard adds no kernel
dependence OF ITS OWN: wherever two arms agree about the answer they agree
about the decision.

---

### S3.1 The runners are not on this box's curve at all

A separate CI-conditions sweep, run against the same commit, measured the
fixture this whole family is named after and it is worth quoting exactly.
Two-layer `PMMStack(1.2e-6, degree=12, far_field_orders=5)`, walls differing
by `delta` = 1e-5, `set_source(0.85e-6, theta=0.15)`, guard DISARMED:

| arm | `max R+T` |
|---|---|
| Katmai | 2.1713 |
| Nehalem | 2.1716 |
| Haswell | 2.1729 |
| Sandybridge | 3.6116 |
| **CI** (py3.10-3.13, EPYC 7763, unpinned) | **1.0000010** |

The four local readings are identical on BOTH builds and at one and at four
threads. The CI reading is not a fifth point on that spread: on the runners
that solve is (near) CORRECT, and no local arm reproduces it. The `rcond` of
the interface system is stable to **0.03 %** across every arm, CI included --
so what differs is the ANSWER, not the conditioning, and the axis that has
not been reproduced here is an unpinned OpenBLAS on a four-core VM together
with the runner's own `libm`.

That is the strongest form of this round's premise, and it forces three
things, all of which the repair already does:

1. **The trigger must be GEOMETRIC.** A wrong answer on this class can read
   `1.000001`. Any gate on the energy reading -- rounds 1, 2 and 3's --
   cannot see it. S5.1.
2. **The arbiter must RETURN when the sliver answer agrees with the
   sliver-free grids.** That is exactly what CI will measure on this fixture:
   `d0` small, verdict `truncation`, answer returned. So no test may assert
   that this fixture is refused, and none does -- S8 records the audit of
   every refusal assertion in the family and the search-or-skip shape they
   now have. `tests/unit/test_fix_pmmstack_sliver_round4.py::
   test_an_answer_that_agrees_with_the_sliver_free_grids_is_returned` pins
   the contract directly, on the running build, at three different readings
   of the same row.
3. **Every sliver test must be a decision-consistency claim**: a RETURNED
   answer agrees with the sliver-free reference within the continuity bar,
   while REFUSALS are allowed to differ by arm, and the committed per-arm
   decision tables carry the rest. S7, S8.

---

## S4. The measurement arms, and what they are not

The round was first asked to show every decision identical under HASWELL,
SKYLAKEX, ZEN and PRESCOTT on both builds -- eight arms. Two of those four
names are not realisable on this hardware. That was measured, reported, and
the brief was then CORRECTED to a two-ladder sweep: the reachable kernels
{HASWELL, NEHALEM, KATMAI, PRESCOTT} crossed with the thread counts
{1, 2, 4, unpinned}, on both builds. S4.1 records the kernel ladder and its
substitutions; S4.2 records the thread ladder and why it is the ladder that
actually reaches CI.

### S4.1 The kernel ladder

**`OPENBLAS_CORETYPE` works, and is verified per run.** Every probe records
`threadpoolctl.threadpool_info()[0]["architecture"]` -- OpenBLAS's own
`corename` -- next to what was requested, so a request that did not take
effect cannot be mistaken for an arm.

**ZEN is not a distinct arm on this build.** The bundled
`scipy-openblas 0.3.31.188.0` reports `architecture` = `Haswell` for
`OPENBLAS_CORETYPE=ZEN` on BOTH builds, and the dispatch is bit-identical:
a 257x257 complex `A @ B`, `eigvals(A)` and `solve(A, B[:,0])` hash to the
same SHA-256 prefixes under ZEN and under HASWELL (`cfe52bef61aeff38` /
`356e4943346d34ef` / `9cd0619aae611bc6` on Windows). `SAPPHIRERAPIDS` aliases
the same way. So CI's AMD EPYC kernel cannot be reproduced by name here; what
CAN be reproduced is a SPREAD of kernels at least as wide as the one CI
exhibits, which is what S3's table does.

**SKYLAKEX cannot run at all on this CPU.** `OPENBLAS_CORETYPE=SKYLAKEX`
loads, and `threadpool_info()` even reports `SkylakeX`, but the first real
GEMM dies with `Illegal instruction` (exit 132) on both builds: the CPU has no
AVX-512. `COOPERLAKE` behaves the same way.

**What the eight arms actually are.** Four DISTINCT kernel families are
reachable per build, and the requested names collapse onto them as follows:

| requested | kernel | distinct? |
|---|---|---|
| HASWELL, ZEN, SAPPHIRERAPIDS, (default) | Haswell | yes (1) |
| PRESCOTT, CORE2, NORTHWOOD, OPTERON, DUNNINGTON | Katmai | yes (2) |
| SANDYBRIDGE | Sandybridge | yes (3) |
| NEHALEM, BARCELONA, ATOM | Nehalem | yes (4) |
| SKYLAKEX, COOPERLAKE | -- | SIGILL |

`KATMAI` was requested explicitly as well, and dispatches to `Katmai` --
the same kernel `PRESCOTT` reaches, on both builds. The two names are one
arm here, and `PRESCOTT` is the one carried in the filenames.

Both aliases are carried as named arms anyway, at one thread on each build,
so that the claim is checked where the round actually makes it: `ZEN` against
`HASWELL` and `KATMAI` against `PRESCOTT` decide **0 of 123 rows** differently
and differ in `err/delta` on **0 of 123 rows** -- four comparisons, all exact
(S7.2 (c)). Hashes say the arithmetic is the same; these say the DECISIONS
are.

So **4 distinct kernels are reachable per build**, and SKYLAKEX is replaced
by SANDYBRIDGE and NEHALEM, which are genuinely different reduction orders.
That is strictly more distinct arithmetic than the requested set, not less.

**The two builds are also a real arm.** At the same Haswell kernel the two
builds' `eigvals` hashes differ (`356e4943346d34ef` on Windows against
`62e193de730b9bb2` on WSL) while GEMM and `solve` agree, i.e. the LAPACK
paths differ between them. Note also that the WSL build's DEFAULT kernel is
now **Haswell**, not the SkylakeX that rounds 1-3 recorded: that build's numpy
has been upgraded since, so the "two builds, two kernels" evidence those
rounds rest on no longer holds as stated. This is recorded as open item R4-D.

### S4.2 The thread ladder, and why it is the one that reaches CI

**CI's runners are the same kernel family as this box.** The 5.45.0 matrix
runs on GitHub-hosted `ubuntu-latest`, i.e. AMD EPYC 7763 (Zen 3). Zen 3 has
no AVX-512, so `scipy-openblas` dispatches it to the SAME Haswell-class
kernel that is this box's default -- exactly the aliasing S4.1 measured for
`OPENBLAS_CORETYPE=ZEN`. The conclusion that follows is uncomfortable and is
stated plainly: **the kernel ladder alone does not reproduce CI**, because CI
is not on a different kernel. What CI has that every local measurement in
rounds 1-3 did not is a different THREAD COUNT -- its fast unit lane leaves
BLAS unpinned on a 4-core runner, while every sliver test file and every
probe in this family pins `OMP/OPENBLAS/MKL_NUM_THREADS=1`.

**The thread count is a known hazard in this very module.** It is not a
speculative axis: `tests/unit/test_m1_conditioning_guard.py` records a
closure residual moving from `6.65e-06` to `2.14e+01` -- seven decades -- on
the SAME cell between one and two OpenBLAS threads. Any guard that keys on a
numeric reading is therefore exposed to the thread count in the same way it
is exposed to the kernel, and the fact that rounds 1-3 never varied it is a
gap in their evidence, not a property of the code.

**So every decision is measured on both ladders.** The decision tables are
regenerated for every `(build, requested coretype, threads)` triple over
kernels {HASWELL, PRESCOTT (= Katmai), NEHALEM, SANDYBRIDGE} and threads
{1, 2, 4, unpinned}: 4 x 4 x 2 = **32 arm files**, plus the ZEN and KATMAI
aliasing arms kept as named evidence. Each file records what was REQUESTED
and what OpenBLAS actually RAN with (`threadpoolctl`'s `num_threads`), never
only the request -- unpinned resolves to 24 threads on this box against CI's
4, so the local unpinned arm is a strictly harder stress than CI's, not a
softer one. S7 gives the results on both axes.

**One trap, recorded because it silently produced a wrong arm.** The first
unpinned run reported `threads_requested = UNPINNED` and `num_threads = 1`:
`g_fixtures` imported round 3's `f_fixtures` BEFORE numpy, and `f_fixtures`
re-applies its own unconditional `os.environ.setdefault(..., "1")` ahead of
ITS numpy import, so `libopenblas` loaded pinned anyway. The env variables
are read once, at library load. `g_fixtures` now imports numpy immediately
after clearing them and snapshots `THREADS_REQUESTED` before anything can
put the pin back. An arm that mis-reports itself is worse than a missing
arm, which is why the readback is asserted rather than the request.

---

## S5. The repair

### S5.1 The trigger is the geometry, not the reading

`_warn_stack_energy` used to return before doing anything whenever
`worst <= 1 + _SLIVER_TRIGGER_BAR`. It no longer does. What decides whether
the guard arbitrates is `_sliver_screen`, which is pure geometry: a
manufactured cell at least `_SLIVER_OWN_SCALE_RATIO` times finer than the
input geometry, on the grid the cascade actually ran on. That is a
deterministic function of the wall coordinates and `min_feature`, identical on
every kernel.

`_sliver_screen` gained a keyword, `require_passive`, defaulting to the
rounds-1-3 behaviour. The arbiter calls it with `False`; see S5.4.

One exclusion had to become EXPLICIT because of that. Rounds 1-3 kept a
JAX-TRACED stack out of the guard for free -- `_stack_provably_passive` cannot
resolve a traced index and answered False, so the screen never fired. Round 4
asks the geometric question without passivity, so `_sliver_screen` now returns
`None` for a stack whose `_holds_traced()` is true: the arbiter's three
re-solves under a trace would be neither cheap nor meaningful.

### S5.2 The attribution is a comparison of answers

The arbiter runs three extra solves and compares efficiencies with
efficiencies. No energy total enters the decision.

```
d0  = |A_sliver - A_M|          A_M  : min_feature = 2 w_wide P (remedy 1)
d12 = |A_L      - A_L'|         A_L  : walls CLOSED onto one coordinate
                                A_L' : that closed wall DISPLACED by w_wide

attributed  <=>  d0 > _SLIVER_MOVE_FACTOR * w_wide       (the geometric floor)
             AND d0 > _SLIVER_WALL_RATIO  * d12          (the measured slope)
```

with three verdicts instead of two:

| verdict | condition | behaviour |
|---|---|---|
| `sliver` | both criteria | REFUSE on a provably passive stack, WARN otherwise |
| `wall` | floor met, ratio not | RETURN with a warning naming the wall sensitivity |
| `truncation` | floor not met | RETURN; note the sliver where the plain warning fires |
| `unknown` | a probe solve cannot be run | round 1's behaviour, unchanged |

**Why the second denominator exists.** `_SLIVER_MOVE_FACTOR` compares an
EFFICIENCY DIFFERENCE with a PERIOD FRACTION. That is only a criterion if the
device's `dR/dx` is of order one, and both earlier verifications refuted that:
the round-2 verification measured a guided-mode resonance at
`dR/d(duty)` = 169.2 whose CORRECT rows reach `move/w_wide` = **833.78**, and
the round-3 verification measured a many-slice taper whose CORRECT rows reach
**906.56** because `move` saturates while `w_wide` vanishes. Both rounds kept
the bar and deleted its published margin, leaving the CLOSURE arm to carry
those rows -- and the closure reads the energy total. Round 4 MEASURES
`dR/dx` instead of assuming it, so the bar it applies is scale-free in the
device.

The measurement is visible in the numbers. `d12 / w_wide` IS the device's
answer change per sliver width, and over the round-4 population it spans
**0.0193 to 29.65** -- three decades -- across the fourteen devices:

| device | `d12 / w_wide` | device | `d12 / w_wide` |
|---|---|---|---|
| `tensor:oop` | 0.0193 | `ladder:C_nir` | 1.003 - 1.008 |
| `tensor:in_plane` | 0.0260 | `d5:wood_deg8` | 1.021 |
| `d5:graze_deg6` | 0.1071 | `ladder:B_vis` | 3.139 - 3.211 |
| `d5:graze_deg8` | 0.1078 | `ladder:O11` | 4.517 - 4.522 |
| `d5:gmr_deg6` | 0.1468 | `census` | 0.4205 - 4.765 |
| `d5:gmr_deg8` | 0.1481 | `ladder:D_tele` | 8.895 - 8.984 |
| `d5:fp_deg8` | 0.7604 - 0.7610 | `ladder:S_steep` | 25.53 - 29.65 |

and on the many-slice taper it tracks `w_wide` the way the move does: at
degree 8 the O-11 taper reads `d12/w_wide` = 2.73 at `nl` = 8 and **45.1** at
`nl` = 16, so its CORRECT rows read `d0/d12` = 0.63 - 1.34 at BOTH slice
counts while `d0/w_wide` climbs from 3.6 to 25.6. That is the 906.56 class,
neutralised by the denominator rather than by a fixed bar.

### S5.3 What was refuted along the way

The round's brief proposed a "two-snap consistency" test: solve on the
prescribed `min_feature = 2 w_wide P` snap and on a SECOND snap, e.g.
`4 w_wide P`, and compare. Measured, and refuted, twice:

* **A coarser `min_feature` is the same grid.** Snapping at `2 w_wide P` and
  at `4 w_wide P` merges the same wall pairs to the same midpoints, so
  `|A_snap1 - A_snap2|` is **exactly 0.0** on all 309 screened rows of the
  first ladder measurement (`p1_ladder_win_DEFAULT.json`).
* **Closing left and closing right is a pure TRANSLATION.** On a SYMMETRIC
  wall opening -- which is the whole `vstair` / `vgmr` / O-11 family -- the two
  closures produce the same ridge WIDTH at two positions, and a translation
  leaves every efficiency unchanged. Measured over the 57 rows the two probes
  share, `|A_L - A_R|` against `|A_L - A_L'|`: **1.31e-06 .. 7.50e-06** on the
  O-11 fixture, **2.80e-06** on the D-5 guided-mode resonance, and never above
  **0.445** anywhere (the multi-slice census box, where a chain of flagged
  walls does change width with the side). In absolute terms on the O-11 row
  the guard refuses: the two closures differ by **1.99e-09** while the
  displacement moves the answer by **4.52e-04**. Using the SIDE as the
  denominator gives `d0/d12` of 1e+05 to 1e+08 on CORRECT rows -- it
  attributes everything.

So the second sliver-free grid has to differ from the first in a way the
DEVICE responds to, and that is what the `shift` argument of
`_sliver_collapsed_segments` does: it displaces the closed wall of the WIDEST
flagged group only, which changes a ridge width rather than a position. The
refuted variants are kept in `p1_ladder.py` / `p2_candidates.py` /
`p2_run.py` as the record.

Two further candidate discriminants were measured on the same rows and
refuted (`p2_*.json`):

* **the bracket excursion** -- how far the sliver answer sits outside the
  componentwise interval the two closures span -- reads within 0.05 % of `d0`
  on every row, because the interval is round-off wide;
* **degree stability**, `|A(deg) - A(deg-2)|` for the sliver solve against the
  same for the snapped one, overlaps: on `d5:graze_deg6` a CORRECT row reads
  32.44 and a WRONG row reads 2.01.

### S5.4 Passivity is required for the REFUSAL, not for the arbitration

Round 4's arbitration never reads an energy total, so it is meaningful on a
stack for which `R + T <= 1` is not a theorem. The geometric screen therefore
no longer requires passivity; the REFUSAL does, and its scope is unchanged.
A stack that is not provably passive and that the arbiter attributes is
WARNED with the same attribution text, marked
`WARNING (the answer is RETURNED, see below)`.

This closes the reachable half of verification defect R3-C. A keyed
`prepare()` stack -- `_segment_passive` answers False for a `str` payload --
is now screened and the arbiter is ATTEMPTED on it; on that path the re-solve
cannot materialise the keys, so the verdict is `unknown` and the behaviour is
unchanged. A non-Hermitian `eps_xy = 0.2` payload, which no exact argument
makes passive, IS arbitrated and IS warned.

### S5.5 The within-layer arm loses its trigger too

`_within_layer_hazard` was reachable only above the trigger, which is open
item R2-B: the measured 1e-6-of-a-period liner that is 1.06e-03 wrong reads
`R+T` = 0.999221, i.e. SUB-unity, and the arm was silent on it. It is now
reached on every solve; its own bar, `_SLIVER_Q_EXCESS`, is a ratio of a
predicted spurious `|q|` to the stack's physical index ceiling and is
unchanged.

---

## S6. The bars, derived two-sided

### S6.1 The populations

Measured with the guard DISARMED so nothing the library decides feeds back,
on `validation/probe_fix_sliver_round4/p3_bars.py`, over fourteen devices in
two runs:

* the five round-3 staircases (continuity slopes 0.47 .. 31.4) x three degrees
  x 18 wall steps; the 648-configuration realistic staircase box, sub-sampled;
  the five D-5 mounts (guided-mode resonance at two degrees, grazing staircase
  at two degrees, Fabry-Perot, near-Wood); the two liquid-crystal directors --
  **510 rows, 471 screened and measured**;
* the round-3 verification's degree-4 V-4 mount; the ROUND-2 verification's
  guided-mode resonance AT ITS OWN RESONANCE (period 1.0 um, `dR/d(duty)` up
  to 169, wl 1.74892 / 1.74900 / 1.74904 um, degrees 16 and 20); and the
  many-slice tapers at `nl` = 6, 12 and 24 -- **87 rows, all measured**.

**558 measured rows: 258 RIGHT / 78 GREY / 222 WRONG.**

| population | `n` | `d0 / w_wide` | `d0 / d12` | attributed |
|---|---|---|---|---|
| RIGHT | 258 | 0.0705 .. **13.5531** | 0.1819 .. **35.1773** | **0** |
| GREY | 78 | 1.48 .. 223.995 | 0.0556 .. **83.1195** | **0** |
| WRONG | 222 | 101.535 .. 1.57e+08 | 3.9733 .. 1.537e+08 | 188 |
| ATTRIBUTED | 188 | **128.57** .. 1.57e+08 | **110.649** .. 1.537e+08 | -- |

WSL, on the first run's 471 rows: RIGHT 13.5531 / 35.1785, WRONG floor
101.5340 / 3.9733, attributed floor 128.57 / 110.647 -- agreeing to five to six
significant figures, and **0 of 357 rows common to both builds differs in
either the continuity class or the analytic verdict**.

### S6.2 `_SLIVER_MOVE_FACTOR` = 100, the geometric floor

Unchanged, and unchanged in meaning: it is the campaign's own
`err > 100 delta` WRONG rule applied to the answer move in units of the widest
manufactured cell. It carries **7.38x** over the correct population's measured
envelope (13.5531) and sits 1.015x under the wrong population's floor
(101.535). The thin side is deliberate: this is a FLOOR, not the
discriminant -- its job is to keep the arbiter off answers the sliver did not
move at all, and 0 of 251 correct rows reaches it.

### S6.3 `_SLIVER_WALL_RATIO` = 100, the measured-slope criterion

Also the campaign's 100x rule, with `delta` replaced by the MEASURED answer
change rather than by the wall step. Two-sided over the 558 rows:

* the CORRECT population's envelope is **35.1773**, so the bar carries
  **2.843x**;
* including GREY it is **83.1195** -- one degree-4 V-4 row -- i.e. **1.203x**;
* the ATTRIBUTED population's floor is **110.649**, i.e. the bar sits 1.106x
  under it;
* the gap BETWEEN the correct-plus-grey envelope and the attributed floor is
  **110.649 / 83.1195 = 1.33x**, and against the CORRECT envelope alone
  **3.15x**. That is the honest margin. It is not decades and is not claimed
  to be: the WRONG population reaches DOWN to `d0/d12` = 3.973, so the two
  populations OVERLAP and no value of this constant separates them completely
  (open item R4-A).

What the value buys: **0 false positives** -- 0 of 258 RIGHT and 0 of 78 GREY
attributed, including the two classes the earlier verifications built to break
the move bar. Six correct-or-grey rows DO pass the geometric floor
(`d0/w_wide` = 114.7 .. 224.0) and every one is held out by the measured
slope:

| row | kind | `d0 / w_wide` | `d12 / w_wide` | `d0 / d12` |
|---|---|---|---|---|
| round-2 GMR at resonance, deg 16, `delta` 3e-5 | GREY | **223.995** | 138.5 | **1.617** |
| V-4 mount deg 4, `delta` 1.1183e-05 | GREY | 170.1 | 2.046 | **83.12** |
| V-4 mount deg 4, `delta` 1.2690e-05 | GREY | 150.0 | 2.040 | 73.51 |
| V-4 mount deg 4, `delta` 1.5538e-05 | GREY | 122.7 | 2.046 | 59.96 |
| V-4 mount deg 4, `delta` 1.6622e-05 | GREY | 114.7 | 2.046 | 56.08 |
| `S_steep` deg 12, `delta` 6.5788e-06 | GREY | 115.6 | 25.55 | 4.522 |

and the many-slice taper -- the round-3 verification's 906.56 class -- never
reaches the floor at all once its own slope is in the denominator: at degree 8
its CORRECT rows read `d0/w_wide` = 3.6 at `nl` = 8 and 25.6 at `nl` = 16
while `d12/w_wide` climbs 2.73 -> 45.1, so `d0/d12` stays at 0.63 .. 1.34.

**False negatives: 34 in 222 WRONG rows**, and where they sit matters more
than the count. Twenty-six are on the round-2 guided-mode resonance sitting on
its own resonance and four on the 12-slice taper -- the two devices whose
measured `dR/dx` is 26 .. 176, i.e. the devices on which the ABSOLUTE
continuity rule is at its own limit. The remaining four are ordinary ladder
rows:

| row | `err/delta` (both pol) | `d0/w` | `d0/d12` | `R+T - 1` |
|---|---|---|---|---|
| `S_steep` deg 20, `delta` = 4.10780e-06 | 1037 | 1012 | 39.60 | 6.292e-03 |
| `D_tele` deg 12, `delta` = 2.56492e-06 | 439.5 | 445.1 | 50.04 | 7.754e-04 |
| `B_vis` deg 20, `delta` = 4.10780e-06 | 177.0 | 179.9 | 57.32 | 1.209e-03 |
| `S_steep` deg 14, `delta` = 4.10780e-06 | 127.1 | 101.5 | 3.973 | 6.404e-04 |

identical on both builds to four significant figures. Rounds 2 and 3 also
left four false negatives on their own grid, and returned them SILENTLY;
these thirteen of the thirty-four are `wall` verdicts and are returned UNDER A
WARNING. Lowering the bar to 50 would recover two of the four ladder rows and
would leave the correct population 1.42x of headroom and the grey population
NONE, which is the shape round 2's verification refuted by 42x; the bar is
left at the campaign's own value.

### S6.3.1 What is returned in silence, and the rule it satisfies

Twenty-one of the 558 rows are WRONG by the absolute rule and returned with
nothing said (verdict `truncation`, and below the plain super-unity warning
bar). Every one is on the guided-mode resonance or the 12-slice taper, and
every one is RIGHT by the SLOPE-NORMALISED rule -- round 3's own -- with room:

| quantity | value |
|---|---|
| absolute `err/delta` of the 21 silent rows | 106.7 .. 221.4 |
| their measured `dR/dx` = `d12 / w_wide` | 26.49 .. 145.7 |
| **worst slope-normalised score `err / (delta * dR/dx)`** | **4.03** |
| the same score, floor over the 188 ATTRIBUTED rows | **15.34** |
| the same score, worst over ALL 34 returned wrong rows | 99.54 |

so the statement the cross-arm test asserts is not "nothing wrong is ever
silent" -- which is false, and was false of every earlier round too -- but the
one that is true and is checkable: **a row is returned in silence only if it
is RIGHT by the slope-normalised continuity rule at the campaign's own 10x
cutoff**, which the 21 clear by 2.48x while the attributed population starts
at 15.34. Recorded as open item **R4-E**.

### S6.4 The classification is taken on BOTH polarizations

`_sliver_answer_move` has always taken its maximum over both incident
polarizations; the campaign's `err` convention scores polarization 1 only.
Where they disagree, the tests were scoring something the guard does not look
at. The round-2 verification already measured 0.0106x the physical shift on
pol 1 against 316x on pol 0 for one out-of-plane-director row. Round 4 applies
the correction everywhere, and it MOVES A PUBLISHED CONCLUSION:

verification defect **V-4** -- "three CORRECT degree-4 answers are refused,
and the named remedy makes them slightly worse" -- does not survive it.
Measured on the verification's own mount and rows:

| `delta` | `err/delta` pol 1 | `err/delta` BOTH | snapped, both | round-4 verdict |
|---|---|---|---|---|
| 1.662224e-05 | 1.051 (RIGHT) | **57.40 (GREY)** | 2.042 | `wall` -> returned + warned |
| 1.268961e-05 | 1.065 (RIGHT) | **75.03 (GREY)** | 2.042 | `wall` -> returned + warned |
| 7.395466e-06 | 0.9791 (RIGHT) | **128.3 (WRONG)** | 2.042 | `sliver` -> REFUSED |

so not one of the three is correct on the statistic the guard decides with,
the prescribed remedy is 28-63x CLOSER to the truth on all three, and the
"slightly worse" finding came from comparing against a DEGREE-16 solve of the
sliver-free device -- across a truncation error this degree-4 mount carries
anyway. Measured: the returned and the snapped answers sit 0.05925 and
0.05927 from that solve, so what separates them (2.7e-05) is 2,000x smaller
than what separates both from it. Against the reference the campaign actually
classifies with -- the exact `delta -> 0` limit at the SAME degree, where the
truncation cancels -- the snap is unambiguously better.

V-4 is CLOSED as a false-refusal finding and restated as a scoring correction.

### S6.5 What round 4 closes and what it leaves

| item | status |
|---|---|
| **R3-A** (a restorable wrong answer left returned because the drop cannot reach 100 on a mount whose floor is near the trigger) | **CLOSED**. The limit was arithmetic in the closure (`drop >= trigger / floor`) and round 4 does not use the drop. The grazing row round 3 returned SILENTLY at `R+T` = 1.0023512 is now REFUSED. |
| **R2-A** (the theorem is a one-sided detector: 11 sub-unity WRONG rows, 29 of 878 at or below the trigger) | **CLOSED as a gate**. Nothing is gated on the reading any more; a sub-unity row is arbitrated like any other. |
| **R2-B** (the within-layer arm inherits the same floor) | **CLOSED**. S5.5. |
| **R2-D** (the move bar assumes `dR/dx = O(1)`) | **CLOSED**. S5.2 measures it. |
| **R3-C** (a keyed `prepare()` stack is outside the guard entirely) | **HALF CLOSED**. The screen and the arbiter now reach it; the re-solve still cannot materialise keys, so the verdict is `unknown`. |
| **V-4** (three correct answers refused) | **CLOSED**, S6.4. |
| **R3-B** (one thin OWNED feature lowers the global own-scale and disarms the cross-layer screen) | **OPEN, untouched**. |
| **R2-E** (`stack2d.py` still passes `stack=None`) | **OPEN, untouched**. |

---

## S7. The decision tables, per arm

`validation/probe_fix_sliver_round4/p4_decisions.py` runs a fixed
123-row decision set -- the five staircases x three degrees, the census box,
the five D-5 mounts, the V-4 mount, the round-2 guided-mode resonance at its
own resonance, two many-slice tapers and the two directors -- through
`PMMStack.solve` end to end and records what the library DID, what the exact
`delta -> 0` reference says about what it RETURNED, and the arbiter's own
quantities. One JSON per arm, committed.

`tests/unit/test_fix_pmmstack_sliver_round4.py` reads all of them and asserts:

1. the arms cover at least two builds, four DISTINCT kernels and the thread
   ladder {1, 2, 4, unpinned}, carry the same row set, and any arm that does
   not DECLARES its subset;
2. **no arm returns a WRONG answer in silence**;
3. **no arm refuses a RIGHT answer**;
4. **wherever every arm agrees about the answer, every arm agrees about the
   decision** -- with the count of answer-unstable rows reported in the
   failure message so a regression that widens it is visible;
5. the same statement again with the BUILD and the dispatched KERNEL held
   fixed and only the THREAD COUNT varied (S7.3);
6. there EXIST rows whose `max R+T` moves by more than two decades between
   arms while the decision is identical -- and rows where it does so across
   the thread ladder ALONE. The round's own premise, as a positive statement,
   on each ladder separately.

### S7.1 What the arms measured

**33 arms.** 28 of them carry the whole 123-row table under a PINNED thread
count: two builds x four dispatched kernels x threads {1, 2, 4}, plus the two
aliasing arms (`ZEN`, `KATMAI`) at one thread on each build. Four more are
the UNPINNED rung on WSL, and one is the UNPINNED rung on Windows, reduced
and declared (S4.2, R4-H). Regenerated from the committed JSON by
`validation/probe_fix_sliver_round4/_arms_table.py`:

| build | requested | kernel dispatched | threads asked / run | rows | python / numpy | refused | wall | returned | wrong returned silently | worst slope-normalised score |
|---|---|---|---|---|---|---|---|---|---|---|
| win | HASWELL | Haswell | 1 / 1 | 123 | 3.14.6 / 2.4.4 | 48 | 6 | 69 | 2 | 1.25 |
| win | HASWELL | Haswell | 2 / 2 | 123 | 3.14.6 / 2.4.4 | 46 | 6 | 71 | 2 | 1.25 |
| win | HASWELL | Haswell | 4 / 4 | 123 | 3.14.6 / 2.4.4 | 49 | 5 | 69 | 2 | 1.25 |
| win | HASWELL | Haswell | UNPINNED / 24 | 15 (subset ladder:O11) | 3.14.6 / 2.4.4 | 10 | 0 | 5 | 0 | 0 |
| win | KATMAI | Katmai | 1 / 1 | 123 | 3.14.6 / 2.4.4 | 55 | 5 | 63 | 2 | 1.25 |
| win | NEHALEM | Nehalem | 1 / 1 | 123 | 3.14.6 / 2.4.4 | 54 | 3 | 66 | 2 | 1.3 |
| win | NEHALEM | Nehalem | 2 / 2 | 123 | 3.14.6 / 2.4.4 | 54 | 3 | 66 | 2 | 1.3 |
| win | NEHALEM | Nehalem | 4 / 4 | 123 | 3.14.6 / 2.4.4 | 54 | 3 | 66 | 2 | 1.3 |
| win | PRESCOTT | Katmai | 1 / 1 | 123 | 3.14.6 / 2.4.4 | 55 | 5 | 63 | 2 | 1.25 |
| win | PRESCOTT | Katmai | 2 / 2 | 123 | 3.14.6 / 2.4.4 | 55 | 5 | 63 | 2 | 1.25 |
| win | PRESCOTT | Katmai | 4 / 4 | 123 | 3.14.6 / 2.4.4 | 54 | 5 | 64 | 2 | 1.25 |
| win | SANDYBRIDGE | Sandybridge | 1 / 1 | 123 | 3.14.6 / 2.4.4 | 48 | 4 | 71 | 2 | 1.31 |
| win | SANDYBRIDGE | Sandybridge | 2 / 2 | 123 | 3.14.6 / 2.4.4 | 50 | 4 | 69 | 2 | 1.31 |
| win | SANDYBRIDGE | Sandybridge | 4 / 4 | 123 | 3.14.6 / 2.4.4 | 47 | 2 | 74 | 2 | 1.31 |
| win | ZEN | Haswell | 1 / 1 | 123 | 3.14.6 / 2.4.4 | 48 | 6 | 69 | 2 | 1.25 |
| wsl | HASWELL | Haswell | 1 / 1 | 123 | 3.12.3 / 2.4.6 | 48 | 6 | 69 | 2 | 1.25 |
| wsl | HASWELL | Haswell | 2 / 2 | 123 | 3.12.3 / 2.4.6 | 46 | 6 | 71 | 2 | 1.25 |
| wsl | HASWELL | Haswell | 4 / 4 | 123 | 3.12.3 / 2.4.6 | 49 | 5 | 69 | 2 | 1.25 |
| wsl | HASWELL | Haswell | UNPINNED / 4 | 123 | 3.12.3 / 2.4.6 | 49 | 5 | 69 | 2 | 1.25 |
| wsl | KATMAI | Katmai | 1 / 1 | 123 | 3.12.3 / 2.4.6 | 55 | 5 | 63 | 2 | 1.25 |
| wsl | NEHALEM | Nehalem | 1 / 1 | 123 | 3.12.3 / 2.4.6 | 54 | 3 | 66 | 2 | 1.3 |
| wsl | NEHALEM | Nehalem | 2 / 2 | 123 | 3.12.3 / 2.4.6 | 54 | 3 | 66 | 2 | 1.3 |
| wsl | NEHALEM | Nehalem | 4 / 4 | 123 | 3.12.3 / 2.4.6 | 54 | 3 | 66 | 2 | 1.3 |
| wsl | NEHALEM | Nehalem | UNPINNED / 4 | 123 | 3.12.3 / 2.4.6 | 54 | 3 | 66 | 2 | 1.3 |
| wsl | PRESCOTT | Katmai | 1 / 1 | 123 | 3.12.3 / 2.4.6 | 55 | 5 | 63 | 2 | 1.25 |
| wsl | PRESCOTT | Katmai | 2 / 2 | 123 | 3.12.3 / 2.4.6 | 55 | 5 | 63 | 2 | 1.25 |
| wsl | PRESCOTT | Katmai | 4 / 4 | 123 | 3.12.3 / 2.4.6 | 54 | 5 | 64 | 2 | 1.25 |
| wsl | PRESCOTT | Katmai | UNPINNED / 4 | 123 | 3.12.3 / 2.4.6 | 54 | 5 | 64 | 2 | 1.25 |
| wsl | SANDYBRIDGE | Sandybridge | 1 / 1 | 123 | 3.12.3 / 2.4.6 | 48 | 4 | 71 | 2 | 1.31 |
| wsl | SANDYBRIDGE | Sandybridge | 2 / 2 | 123 | 3.12.3 / 2.4.6 | 50 | 4 | 69 | 2 | 1.31 |
| wsl | SANDYBRIDGE | Sandybridge | 4 / 4 | 123 | 3.12.3 / 2.4.6 | 47 | 2 | 74 | 2 | 1.31 |
| wsl | SANDYBRIDGE | Sandybridge | UNPINNED / 4 | 123 | 3.12.3 / 2.4.6 | 47 | 2 | 74 | 2 | 1.31 |
| wsl | ZEN | Haswell | 1 / 1 | 123 | 3.12.3 / 2.4.6 | 48 | 6 | 69 | 2 | 1.25 |

### S7.2 The findings

**(a) No arm returns a wrong answer outside the slope-normalised allowance,
and no arm refuses a right one.** 0 false positives on every arm; the rows
returned WRONG and silent are the two guided-mode-resonance rows of S6.3.1, on
every full arm, with a worst slope-normalised score of 1.25-1.31 against the
bar of 10.

**(b) The decision is a function of the ARITHMETIC, not of the BUILD.** At
every one of the twelve (kernel, thread-count) configurations the two builds
give the SAME three counts, and **0 of the 123 rows decides differently
between the two builds** at any of them. The builds differ in python (3.14.6
against 3.12.3), in numpy (2.4.4 against 2.4.6) and in their LAPACK paths
(their `eigvals` hashes differ at the same kernel), and none of that reaches
the decision.

**(c) The two aliased coretypes are aliases in the DECISIONS too.** `ZEN`
against `HASWELL`, and `KATMAI` against `PRESCOTT`, at one thread on each
build: **0 of 123 rows differ in decision, and 0 differ in `err/delta` at
all** -- four comparisons, all exact. S4.1 established the aliasing from GEMM
and `eigvals` hashes; this is the same statement at the level the round
cares about, and it is why "eight arms, one per requested coretype" would
have been four arms wearing eight names.

**(d) UNPINNED IS NOT A THIRD AXIS -- it is a thread COUNT the library picks
for itself.** On Linux an unpinned OpenBLAS sizes its pool from the CPUs the
process can see, so narrowing the affinity to four reproduces CI's runner
(read back: `num_threads` = 4). Against the `OPENBLAS_NUM_THREADS=4` arm at
the same kernel, the unpinned arm is **bit-identical**: 0 of 123 rows differ
in `decision`, and 0 differ in `err/delta` at all, at Haswell, Katmai,
Nehalem and Sandybridge alike. Removing the variables is not a different
configuration from pinning the same number -- which is why S7.3 varies the
COUNT and does not treat "unpinned" as a category of its own.

**(e) Where the arms agree about the ANSWER they agree about the DECISION.**
"Agree about the answer" is not "same continuity class" -- wrong by 180x the
wall shift and wrong by 259,000x are both WRONG and are not the same answer --
so the window is the same class AND distances from the exact `delta -> 0`
limit within 10x of one another. On that definition **68 of the 123 rows are
answer-stable across all 32 full arms, and the decision is identical on every
arm for every one of them**; over all 33 arms, 7 of the 15 rows they share are
answer-stable, likewise with no disagreement.

The rows that are NOT answer-stable are the conditioning collapse itself. Six
of them agree in CLASS across every arm while the magnitude does not, so a
naive class-only comparison would have called them a guard defect; four are
quoted here:

| row | `err/delta` per kernel | decision |
|---|---|---|
| `ladder:B_vis` deg 12, `delta` 3e-06 | 179.6 (Haswell) / 242.6 (Katmai) / **2.589e+05** (Nehalem) | `wall` / `wall` / refused |
| `ladder:S_steep` deg 12, `delta` 3e-06 | 305.5 / 161.7 / **5.431e+05** | `wall` / `wall` / refused |
| `ladder:S_steep` deg 14, `delta` 3e-06 | 2329 / **5.087e+05** / 387.2 | `wall` / refused / `wall` |
| `ladder:S_steep` deg 20, `delta` 3e-06 | **5.282e+05** / 4865 / 2557 | refused / refused / `wall` |

Every one is WRONG on every kernel, and wrong by three decades more on one
kernel than on another. The guard follows the answer it was given; it is not
the source of the disagreement.

**(f) The energy reading moves by decades on rows the guard decides
identically.** Worst spread of `max R+T - 1` over the full arms, restricted to
rows every arm decided the same way: **3961**, then 3056, 1873, 1490 and 1309.
That is the round's own premise as a positive statement -- rounds 1-3 keyed
both the trigger and the attribution on that quantity, so on those rows they
could not have decided identically.

### S7.3 The thread ladder

Held at one build and one dispatched kernel, with only the thread count
varied (the aliasing arms sit in their kernel's group, which is why some
groups carry more arms than thread settings):

| build | kernel | arms | thread settings | rows compared | answer differs | DECISION differs | worst err/delta spread |
|---|---|---|---|---|---|---|---|
| win | Haswell | 5 | 1,2,4,UNPINNED | 15 | 15 | 2 | 1018 |
| win | Katmai | 4 | 1,2,4 | 123 | 101 | 1 | 423.9 |
| win | Nehalem | 3 | 1,2,4 | 123 | 117 | 0 | 2.794 |
| win | Sandybridge | 3 | 1,2,4 | 123 | 117 | 9 | 2.031e+04 |
| wsl | Haswell | 5 | 1,2,4,UNPINNED | 123 | 123 | 11 | 3.368e+04 |
| wsl | Katmai | 5 | 1,2,4,UNPINNED | 123 | 102 | 1 | 423.9 |
| wsl | Nehalem | 4 | 1,2,4,UNPINNED | 123 | 117 | 0 | 2.71 |
| wsl | Sandybridge | 4 | 1,2,4,UNPINNED | 123 | 117 | 9 | 2.031e+04 |

`answer differs` counts any numerical difference in `err/delta`, including the
last bit; `DECISION differs` is the NAIVE comparison that ignores whether the
arms agree about the answer, and is printed here so that what the pinned claim
covers is explicit. Under the same rule the kernel pin uses -- same continuity
class, and distances from the exact limit within 10x -- every one of those 33
rows is in the answer-unstable set, and across the eight ladders there are
**828 answer-stable rows, 0 decision differences, and 48 answer-unstable
rows**.

Three things follow.

**(i) The thread count moves the ANSWER, on nearly every row.** 101 to 123 of
the 123 rows of every ladder differ numerically between one, two and four
threads, and in the collapse band they differ by a long way: `err/delta`
spreads to **3.4e+04** at Haswell and **2.0e+04** at Sandybridge on a fixed
build and kernel. That is a property of the library, not of the guard, and it
is recorded as open item R4-G.

**(ii) It moves the ENERGY READING rounds 1-3 decided on by more than two
decades, on rows round 4 decides identically.** Worst spread of `max R+T - 1`
across the thread ladder alone: **1307x** on `ladder:C_nir` at degree 20
(Sandybridge), **452.6x** on `ladder:B_vis` at degree 14 (Haswell), then
263.4x, 143.1x and 140.2x -- the same values on both builds to four figures.
A guard keyed on that reading could not have been thread-independent, and the
5.45.0 fast lane is where that showed.

**(iii) It does not move the DECISION.** 0 differences over 828 answer-stable
rows. Nehalem is the one kernel whose classification does not move with the
thread count at all (0 answer-unstable rows of 123); Haswell and Sandybridge
carry 13 and 15.

---

## S8. The restated tests

Every restatement below is a change of SHAPE, not a relaxed bar. The rule
applied throughout: a test may name a row to build a fixture, and may not name
a row to assert a reading, a verdict or a refusal.

| file | what changed |
|---|---|
| `test_fix_pmmstack_sliver_walls.py` | `_solve` keeps BOTH polarizations and `_classify` scores both; the named `(14, 1e-4)` row is replaced by `_hazard_band()`, which classifies a ladder at run time; the two fail-before tests assert what the pre-fix path DOES (returns every band row, bit for bit) instead of what it reads; a new `test_the_mechanism_is_a_deterministic_geometric_fact_not_an_energy_reading` fits the `1/w^2` interface conditioning (exponent 2 to within 10 %) and the `1/w` spurious `|q|`, which is what the fail-before tests key on now; the message tests take their refusal from `_first_refusal()`. |
| `test_fix_pmmstack_sliver_walls_round2.py` | `_err` defaults to both polarizations; a new `_outcome()` returns `refused` / `warned` / `silent`, and the two-sided tests assert "a WRONG row is never SILENT" instead of "a WRONG row is refused"; `test_the_round1_misses_...` becomes `test_no_wrong_row_of_the_round1_miss_set_is_returned_in_silence`; `test_the_trigger_sits_above_...` becomes `test_the_move_floor_sits_above_the_correct_populations_own_envelope`; `test_the_arbiter_costs_one_solve_...` becomes `test_the_arbiter_costs_three_solves_and_only_on_a_screened_stack` and states the cost change; the three path tests take their row from `_a_refused_row()`; new tests: `test_the_arbiter_runs_on_a_stack_that_reads_no_super_unity_at_all`, `test_the_closed_grid_carries_no_sliver_and_is_the_same_device`, `test_a_not_provably_passive_sliver_is_warned_and_never_refused`.  Two population counts were loosened after the Katmai arm measured them smaller (the arbitrated-correct count 10 -> 9) and one named-row assertion on the conical path was replaced by the screen, because that row is WRONG-and-warned on Haswell and GREY-and-silent on Katmai. |
| `test_fix_pmmstack_sliver_round3.py` | the D-5 decision is kept and its ASSERTIONS move to the round-4 criteria; `test_the_closure_fraction_separates_the_two_drop_populations` becomes `test_the_closure_fraction_is_evidence_and_is_not_a_separator` and states the CI numbers on both sides; new `test_the_d5_class_is_reached_without_reading_an_energy_total_at_all` hands the arbiter a reading of exactly 1.0; the truncation-note test follows the note's new wording. |
| `test_verify_pmmstack_sliver_walls.py` | the owned-liner ladder no longer asserts that some degree reads SUB-unity (the CI matrix read 1.1298 / 2.0084 / 1.5745 / 1.1618 -- none is) and asserts instead that the four degrees do not agree with each other; the false-negative floor and the V-1 false-positive reproducer score the reading where the sentence is about it and never assert it of a named row; `test_the_guard_has_a_measured_floor_the_theorem_cannot_reach` demanded `returned >= 2` and now demands `refused >= 2` -- measured on this box the Haswell kernel returns 3 of its 60 rows and the KATMAI kernel returns none, so the old demand failed on a STRONGER guard. |
| `test_verify_pmmstack_sliver_round2.py` | `test_a_keyed_prepared_stack_is_outside_the_guard_entirely` re-pinned against S5.4 -- it was written as "re-pin it if keyed stacks are later resolved", and they now reach the arbiter. |
| `test_verify_pmmstack_sliver_round3.py` | `test_the_relative_closure_leaves_a_restorable_wrong_answer_returned` re-pinned as `test_round4_closes_the_restorable_wrong_answer_r3a_left_returned` (R3-A closed); `test_a_correct_row_past_the_move_bar_is_still_returned` keeps its DECISION and takes the new verdict name; `test_a_correct_answer_is_refused_when_the_snap_leaves_the_superunity_regime` re-pinned against S6.4. |
| `test_fix_pmmstack_sliver_round4.py` | NEW. The cross-arm pin on BOTH ladders, the two bars two-sided, the sensitivity-denominator contract, the trigger removal, and the refusal message's two criteria. Its arms are keyed `(build, kernel, requested, threads)`, the coverage test requires the thread ladder {1, 2, 4, UNPINNED} to be present at every distinct kernel on both builds, and `test_the_decision_does_not_move_with_the_thread_count` holds the build and the dispatched kernel FIXED and varies only the thread count -- judged by the same answer-agreement window the kernel pin uses, so neither ladder is graded more leniently than the other. |
| `test_m1_conditioning_guard.py` | untouched. Its module-scope `PMM_SLIVER_GUARD` disarm already covers round 4. |
| `test_pmm_m2_window_contract.py` | NOT a sliver file, and changed anyway, by the precedent its sibling already set: it gains the same module-scope `PMM_SLIVER_GUARD` disarm. Its subject IS the near-coincident-wall collision -- the window's only cross-layer separation is the per-slice taper offset (3.61 / 1.80 / 0.90 nm at `n_slices` = 3 / 6 / 12) and several arms deliberately leave it UNSNAPPED to measure the silent-wrong draw. Rounds 1-3 reached those arms only by accident (the solves read exactly `R+T` = 1, so the trigger never opened); round 4 decides on the geometry and REFUSES them, correctly -- measured on the uncoated `ns` = 6 ladder, 10 manufactured cells, `move` = **0.536** in per-order efficiency = **207.9x** the widest manufactured cell against a measured device wall sensitivity of **0.00217** (ratio **247.2x**). That is the R2-A class closing on a real device: an answer half the efficiency scale away from its sliver-free value that no earlier round could see. |

**What the per-arm sweep itself found, and it is the point of running one.**
Three kernels surfaced assertions that were still kernel facts after the first
pass, and each was restated rather than relaxed:

| kernel | what failed | what it now asserts |
|---|---|---|
| Katmai | `test_the_guard_has_a_measured_floor_the_theorem_cannot_reach` demanded `returned >= 2` of its 60-row ladder | `refused >= 2` -- Katmai refuses ALL 60, which is a STRONGER guard, not a broken ladder |
| Katmai | the arbitrated-correct count `n >= 10` read 9 | `n >= 6`, with the two measured counts quoted |
| Katmai | the conical `delta` = 3e-5 row was asserted never SILENT | the screen is asserted; that row is WRONG-and-warned on Haswell and GREY-and-silent on Katmai |
| Sandybridge | three D-5 existence tests demanded `>= 2` rows meeting the "wrong as returned, right on the prescribed grid" premise | `>= 1`; three rows meet it on Haswell, one on Sandybridge (the others read 24.4x and 61.4x) |
| Sandybridge | the R3-A re-pin asserted round 3's drop CANNOT reach the row | recorded, not asserted -- the drop is 5.205 on Haswell and 149.6 on Sandybridge, which is precisely why R3-A was a limit that moved |
| Sandybridge | the returned row vs its remedy was asserted 100x apart | a DECADE; measured 93.6x on Sandybridge and 1.1e+03x on Haswell |
| Sandybridge | `test_the_m1_module_disarm_is_function_scoped_and_restores` raised on the NAMED `(14, 1e-4)` row | the refused row is searched for; the claim is about the SWITCH |
| Nehalem | the LC out-of-plane row asserted pol 0 is WRONG by the absolute rule (it reads 316x on Haswell and **73.9x** on Nehalem, i.e. GREY) | the two polarizations DISAGREE by a decade -- which is the reason the move is taken on both, and does not move |
| Nehalem | the D-1 owned-liner reproducer and the keyed-`prepare()` CONTROL both raised on the NAMED `(12, 3e-5)` row (refused on three kernels, RETURNED on Nehalem) | the refused delta is searched for; the claims are about the SCREEN and about the KEY |

**Every refusal assertion in the family was audited, and each one now either
searches or skips.** The CI-conditions sweep of S3.1 makes this sharp: on the
runners the O-11 fixture is solved CORRECTLY, so a test that requires a
refusal is asserting the ARITHMETIC and not the guard. There are sixteen
places in the seven files that assert a refusal happened. Ten of them are
already conditional on the row being measured WRONG on the running build
(`test_the_hazard_band_is_still_refused_and_the_outside_is_untouched` asserts
`refused` under `if kind == "wrong"` and `not refused` under `if kind ==
"right"`; the D-5 loop only enters its body for rows with `err > 100` and
`err_snapped < 1`), and are correct on CI by construction. The other six went
through a helper that searched for a refused row and RAISED when it found
none. Those helpers now do three things instead:

1. assert that the GEOMETRIC SCREEN fired on every row they searched -- a
   deterministic function of the wall coordinates and the `min_feature`, so
   it must hold on every build and a failure there is a guard defect;
2. `pytest.skip` with a message that names the reason -- "what is missing is
   a WRONG ANSWER, not the guard", with CI's 1.0000010 against this box's
   2.17 / 3.61;
3. keep the claim they were making about PLUMBING (which wavelength the
   arbiter re-solves at, whether the module switch is armed, whether an owned
   liner silences the screen) rather than about physics.

The files touched: `test_fix_pmmstack_sliver_walls.py` (`_first_refusal`),
`test_fix_pmmstack_sliver_walls_round2.py` (`_a_refused_row`, the three
`solve_vs_wavelength` / `prepare()` path tests),
`test_verify_pmmstack_sliver_walls.py` (the M1 module-disarm test),
`test_verify_pmmstack_sliver_round2.py` (the owned-liner disarm and the keyed
`prepare()` CONTROL) and `test_fix_pmmstack_sliver_round3.py` (the D-5
existence count). Each gained a `_screened(...)` builder that runs the screen
WITHOUT solving.

Counts and wall times, on every arm, are in S11.1.  Nothing in the restated
files asserts a reading, a verdict at a named row, or a refusal at a named
row; the two things that ARE asserted of a named fixture are its GEOMETRY (the
screen, the own-scale ratio, the widths the collapse produces) and the
MECHANISM (the spurious ``|q|`` and the `1/w^2` conditioning exponent), both
of which are deterministic functions of the wall coordinates and the degree.

---

## S9. Cost

Round 4 pays MORE, and pays it on MORE stacks. Both halves are deliberate and
both are measured (`validation/probe_fix_sliver_round4/p5_cost.py`, best of
five reps, one thread, `OPENBLAS_CORETYPE=HASWELL`).

**What is paid, and where.** Round 2 paid ONE extra solve, and only on a stack
that already read super-unity above `_SLIVER_TRIGGER_BAR` -- it measured that
at 0.26x of the solve it guarded and fired on 0 of 600 converged correct rows.
Round 4 pays THREE, on every stack the GEOMETRIC screen fires on, whatever the
solve reads. The three probe grids are COARSER than the sliver grid (the snap
merges cells and the closure removes them), so each probe is cheaper than the
solve it guards.

| stack | screened | guard OFF | guard ON | ratio |
|---|---|---|---|---|
| `census_deg6_nl4` | yes | 102.8 ms | 137.5 ms | **1.338** |
| `o11_deg14_d0` | no | 25.4 ms | 27.3 ms | **1.073** |
| `o11_deg14_d1e-4` | yes | 93.4 ms | 180.0 ms | **1.928** |
| `o11_deg14_d3e-3` | no | 102.8 ms | 96.5 ms | **0.939** |
| `o11_deg20_d1e-5` | yes | 255.9 ms | 423.8 ms | **1.657** |
| `taper16_deg8` | yes | 25114.8 ms | 20850.0 ms | **0.830** |

**READ THE RATIO, NOT THE MILLISECONDS, AND NOT THE LAST ROW.** These were
taken on a box at 100 % CPU from unrelated concurrent work (19-21 python
processes), which is why a 17-layer degree-8 solve reads 25 SECONDS and why
its ratio comes out below 1: the scheduler noise on that row is larger than
the quantity being measured.  The four rows whose solves are 25-260 ms are the
usable ones, and they bracket the cost at **1.07x** with nothing screened and
**1.34 - 1.93x** with a sliver present -- i.e. the three probe solves together
cost 0.3 to 0.9 of the solve they guard, consistent with round 2's measured
0.26x for ONE of them on a coarser grid.  A quiet-box re-measurement is left
as an open item.

**How much of a real population pays it.** The screen is the whole gate, so
the firing rate IS the exposure:

* `census_box`: **120 of 144** (83.3 %) stacks are screened.
* `ordinary`: **0 of 400** (0.0 %) stacks are screened.

**What it does NOT cost.** On a stack with no manufactured cell the guard runs
two union-grid rebuilds and a passivity check and nothing else -- the rows
marked `screened=False` above.

**The honest summary.** On a tapered staircase fine enough to manufacture a
sliver, a `PMMStack` solve now costs about **1.3 - 1.9x** what it did,
and a `solve_vs_wavelength` sweep over such a stack pays that at every
wavelength. That is the price of a decision that does not move with the BLAS
kernel, and it is charged only to geometry that is already the subject of a
refusal.

---

## S10. Open items

**R4-A (MEDIUM).** The two populations overlap on `d0/d12`: WRONG reaches down
to 3.973 while CORRECT reaches up to 35.177 and GREY to 83.120, so the bar is
a gap between envelopes (1.33x against CORRECT+GREY, 3.15x against CORRECT
alone) and not a separation. 34 wrong rows in 222 are returned -- 13 of them
under the `wall` warning and 21 in silence (R4-E). A detector with a real
separation would have to come from the CONDITIONING rather than from the
answers -- the round-3 verification's own recommendation, and still the
highest-value change this campaign points at.

**R4-B (LOW).** A WRONG row whose move is under the GEOMETRIC floor is
returned by the `truncation` arm, and is silent unless the plain super-unity
warning fires. That is how the 21 rows of R4-E come back. On the ordinary
(non-resonant) families the floor holds with 1.015x -- the wrong population's
`d0/w_wide` floor is 101.535 against a bar of 100 -- which is thin, and is a
sample property.

**R4-C (LOW).** `d12` is measured by displacing ONE wall group -- the one
carrying the widest manufactured cell. On a stack with many flagged groups
that is a local measurement of a global sensitivity. It has not been shown to
matter on any device measured here.

**R4-E (MEDIUM).** The ABSOLUTE continuity rule `err > 100 delta` and the
guard's criterion cannot both be satisfied on a device whose measured `dR/dx`
approaches 100: at `dR/d(duty)` = 169 the rule calls a row WRONG at 132x the
wall shift when displacing that wall by one sliver width already moves the
answer 113 .. 146x. Twenty-one such rows are returned in silence (S6.3.1).
The round-2 verification adjudicated the same class against an independent
`RCWAStack` oracle and found the guard's decisions correct on all 14 rows it
could score; no such adjudication was run this round.

**R4-D (LOW, methodological).** The WSL build's default OpenBLAS kernel is now
Haswell, not the SkylakeX that rounds 1-3 record, so the "two builds, two
kernels" evidence in those documents no longer holds as stated. Round 4 drives
the kernel explicitly with `OPENBLAS_CORETYPE` and verifies the dispatch; the
earlier rounds' cross-build agreement should be re-read as cross-BUILD, not
cross-kernel.

**R4-F (LOW, methodological).** The cost table in S9 was taken on a box at
100 % CPU from unrelated work; the two rows whose solves exceed a second are
scheduler noise and one of them reads a ratio below 1. The four usable rows
bracket the cost at 1.07x unscreened and 1.34-1.93x screened. A quiet-box
re-measurement would tighten that and has not been run.

**R4-G (MEDIUM, and new with the thread ladder).** In the
conditioning-collapse band the library's ANSWER is a function of the BLAS
THREAD COUNT, not only of the kernel, and by more than the kernel moves it:
at a fixed build and a fixed dispatched kernel, `err/delta` spreads by up to
**3.4e+04** between one, two and four threads on the same row, and the energy
reading `max R+T - 1` spreads by **452.6x** on rows the guard decides
identically. Round 4 neither causes this nor fixes it -- the guard follows
the answer it is given, and S7.3 shows it adds no thread dependence of its own
-- but it is the correct diagnosis of why the 5.45.0 fast lane failed where a
pinned local run passed, and it means any future criterion keyed on a NUMBER
rather than on a comparison of answers will have the same failure mode. The
conditioning-based detector R4-A asks for is the change that would close it.

**R4-H (LOW, methodological).** The UNPINNED rung of the thread ladder is run
on a declared SUBSET of the decision table (`subset` in the JSON), not on all
123 rows. The reason is measured: this box has 24 hardware threads, so
"unpinned" here means OpenBLAS taking 24 threads on spectral-element
eigenproblems a few hundred wide, and the first full unpinned arm had not
finished its FIRST case group after 17 minutes against 2 minutes for the whole
pinned table. CI's unpinned lane is a FOUR-core runner -- i.e. exactly the
`t4` arm, which IS measured in full at every kernel on both builds -- so what
the reduced rung adds is the extreme, not the CI configuration. A box with
fewer cores, or a `taskset`-limited run, would let the full unpinned table be
taken.

**R3-B, R2-E** carried forward untouched (S6.5).

---

## S11. Runs

Every run below is pinned at one thread (`OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` on the command line) with the kernel
named explicitly by `OPENBLAS_CORETYPE`. The box was heavily loaded throughout
by unrelated concurrent work (19-21 python processes at 100 % CPU), so the
WALL times below are load figures and not performance figures; the round-2
audit records the same hazard.

### S11.1 The eight sliver test files, both ladders, two builds

Every table in this section is regenerated from the run logs by `_fill_runs`.
The LOGS themselves are not committed -- the repository ignores `*.log`, and
round 3 committed probe scripts and JSON only -- so what ships is the scripts
that produce them (`_run_tests_par.sh`, `_run_tests_threads.sh`,
`_run_ladder_par_*.sh`, `_run_ci_emul_wsl.sh`, `_run_unpinned_*.sh`), the
per-arm decision tables, and these numbers.

| build | requested | kernel dispatched | BLAS threads | sweep | result |
|---|---|---|---|---|---|
| win | HASWELL | Haswell | 1 | kernel sweep | `117 passed, 1 warning in 108.18s (0:01:48)` |
| win | HASWELL | Haswell | 1 | thread ladder | `117 passed, 1 warning in 86.56s (0:01:26)` |
| win | HASWELL | Haswell | 2 | thread ladder | `117 passed, 1 warning in 89.82s (0:01:29)` |
| win | HASWELL | Haswell | 4 | thread ladder | `117 passed, 1 warning in 87.58s (0:01:27)` |
| win | NEHALEM | Nehalem | 1 | kernel sweep | `117 passed, 1 warning in 170.93s (0:02:50)` |
| win | PRESCOTT | Katmai | 1 | kernel sweep | `117 passed, 1 warning in 183.54s (0:03:03)` |
| win | SANDYBRIDGE | Sandybridge | 1 | kernel sweep | `1 failed, 116 passed, 1 warning in 163.16s (0:02:43)` |
| win | ZEN | Haswell | 1 | kernel sweep | `117 passed, 1 warning in 107.33s (0:01:47)` |
| wsl | HASWELL | Haswell | 1 | kernel sweep | `117 passed, 1 warning in 103.47s (0:01:43)` |
| wsl | HASWELL | Haswell | 1 | thread ladder | `117 passed, 1 warning in 84.66s (0:01:24)` |
| wsl | HASWELL | Haswell | 2 | thread ladder | `117 passed, 1 warning in 73.66s (0:01:13)` |
| wsl | HASWELL | Haswell | 4 | thread ladder | `117 passed, 1 warning in 69.78s (0:01:09)` |
| wsl | NEHALEM | Nehalem | 1 | kernel sweep | `117 passed, 1 warning in 167.35s (0:02:47)` |
| wsl | PRESCOTT | Katmai | 1 | kernel sweep | `1 failed, 116 passed, 1 warning in 180.22s (0:03:00)` |
| wsl | SANDYBRIDGE | Sandybridge | 1 | kernel sweep | `117 passed, 1 warning in 158.69s (0:02:38)` |
| wsl | ZEN | Haswell | 1 | kernel sweep | `117 passed, 1 warning in 103.99s (0:01:43)` |

The thread ladder of those same eight files, at the kernel CI dispatches to:
**117 passed** at one, two and four BLAS threads, on BOTH builds
(`tests_<build>_HASWELL_t<n>.log`). Nothing in the family is thread-dependent.

Per-test wall times are captured with `--durations=0` and spliced into
`.test_durations` by `_splice_durations.py` (removed 8 node ids this round
renamed away, added 26, updated 161; **12,776 entries**, sorted, JSON-valid).
The SLOWEST test of the eight files is **11.48 s**
(`test_the_guard_has_a_measured_floor_the_theorem_cannot_reach`), then 6.97 s
and 5.73 s -- every one comfortably inside the 60 s ceiling, on a loaded box.
The two tests round 4 adds are 0.9 s and 0.4 s: they read committed JSON.

**One failure in that sweep is NOT round 4's and is NOT a sliver test.**
`test_m1_conditioning_guard.py::test_x1_is_closed_across_the_whole_thin_ladder_and_reopens_pre_fix`
fails with "the worst pre-fix `sum(R)` is only **7.30x** (WSL/Katmai) /
**7.44x** (Windows/Sandybridge) from the converged 2.094088e-04: the
engineered arm is not reproducing the 152x error the defect is documented
at". It failed the 5.45.0 CI matrix the same way, at **0.68x**, on python
3.10 shard 4.

The interesting part is WHICH arms it fails on, and it is reproducible: it
fails on Windows/Sandybridge and WSL/Katmai and PASSES on WSL/Sandybridge and
Windows/Katmai -- the same four outcomes in two independent sweeps, one
sequential and one with the arms running concurrently. So it is not a
property of the kernel, nor of the build, but of the PAIR, and the two
failing readings sit within 2 % of each other. That is what a fail-before
pinned to the MAGNITUDE of an ill-conditioned draw looks like when the draw
is marginal against the bar. It is also thread-independent: the same file
passes at one, two and four threads on Haswell on both builds. That file
disarms
`PMM_SLIVER_GUARD` for the whole module -- its own comment says "which of them
trip it is a BLAS fact" -- so round 4's guard is inert there and cannot be the
cause; with the switch off `_warn_stack_energy` is bit-identical to the
pre-round-4 version. It is the SAME DEFECT SHAPE in a different guard's test:
a FAIL-BEFORE that asserts the MAGNITUDE of an ill-conditioned pre-fix draw
rather than its mechanism or the decision it forces -- which is precisely
what S8 restated out of all seven sliver files. It belongs to whoever owns
the M1 conditioning guard and is reported, not fixed, here.

**One CI failure OUTSIDE this family is fixed by the trigger removal.**
`test_fix_pmm2d_mortar_round2.py::test_the_plain_1d_interface_solve_is_left_unguarded_and_this_is_why`
failed the matrix on python 3.11 shard 4 because at `delta` = 1e-5 its 1-D
interface solve read `R+T` = 1.0000010 -- under `_SLIVER_TRIGGER_BAR` -- so
the guard said nothing and the expected SLIVER text was absent. Round 4
arbitrates on the geometry, so the guard speaks there; the three sliver-facing
tests of that file (`test_a_requested_sliver_is_refused_and_the_message_names_the_cure`,
`test_fail_before_the_sliver_the_guard_refuses_is_measurably_wrong`,
`test_the_plain_1d_interface_solve_is_left_unguarded_and_this_is_why`) pass
**3/3** on Windows/Haswell, 74.63 s. That file belongs to the mortar rounds and
was not touched.

### S11.2 The `PMMStack` regression

`validation/probe_verify_sliver_round2/_pmmstack_test_files.txt`, 43 files,
every test file that imports `PMMStack`:

| build | requested | kernel dispatched | result |
|---|---|---|---|
| win | HASWELL | Haswell | `824 passed, 1 skipped, 71 warnings in 1499.50s (0:24:59)` |
| wsl | PRESCOTT | Katmai | `1 failed, 823 passed, 1 skipped, 71 warnings in 2784.70s (0:46:24)` |

### S11.3 The census / walker / dispatcher-pin / public-API sweep

25 files -- `test_eme_census_determinacy.py`, the fourteen `*walker*` and
`*dispatcher_pin*` files, and `test_public_api.py`: **1284 passed, 12
skipped**, Windows, one thread, re-run after the last documentation edit.
This is the sweep that reads `CHANGELOG.md` and the docs, so it is what
checks the round-4 entry.

### S11.4 `ruff`

`wsl -e bash -lc 'cd /mnt/c/tmp/lum_sliver4 && ~/lumvenv/bin/ruff check
lumenairy/ tests/ validation/probe_fix_sliver_round4/'` -> **All checks
passed!**  (Seven import-ordering findings in the new probe files were fixed
with `--fix`; `lumenairy/` and `tests/` were clean throughout.)

### S11.5 The probes

| probe | what it produced |
|---|---|
| `p1_ladder.py` | 330 rows, 309 screened, on `win/DEFAULT` -- the first refutation (the second `min_feature` snap is the same grid; `d12` exactly 0.0 on 309 of 309). |
| `p2_candidates.py` + `p2_run.py` | 453 + 87 rows on `win/DEFAULT` -- the second and third refutations (left/right closure is a translation; the bracket excursion is `d0`; degree stability overlaps). |
| `p3_bars.py` | 510 rows on `win/DEFAULT` and `wsl/DEFAULT` (471 screened, 0 of 357 common rows differing in class or verdict) + 87 rows of the V-4 / resonance / taper families on `win/DEFAULT`. |
| `p4_decisions.py` | the 123-row decision table on TEN `(build, requested-coretype)` arms. |
| `p5_cost.py` | the cost ratios and the two firing rates. |
| `_arms_table.py`, `_splice_durations.py` | the cross-arm summary S7 quotes, and the `.test_durations` splice. |

Every JSON carries an `arm` block naming the build, the requested coretype,
the kernel OpenBLAS actually dispatched to, the python and the numpy.
