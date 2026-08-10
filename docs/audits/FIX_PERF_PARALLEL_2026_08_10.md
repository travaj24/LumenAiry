# FIX -- audit item #5 (ORDER-LEVEL PARALLELISM) plus the adjudication's loud-choice remedies

**2026-08-10.  Branch `perf/traced-hotpath`, checked out at `0097e5a` and
modified in the working tree only -- no commit, no push, no tag, no CHANGELOG
entry.**

Three things ship here, and they are one thing:

1. **The two design-121 runners are import-safe.**  `fan_multi_121.py` and
   `focus_scan_121.py` put their bodies behind `if __name__ == '__main__':`.
   Under `spawn` a congruence worker RE-IMPORTS `__main__`, so an unguarded
   runner ran its whole acceptance again in every child -- which is why
   `congruence_workers` could not be used from the fan AT ALL
   (`AUDIT_TRACED_SPEED_2026_08_09.md` sec 3.2).  The library's own refusals
   stay exactly where they are; they are the backstop, not the fix.
2. **The record-run grid choice is LOUD.**  Both runners take an explicit
   RAM-budget intent (`RAMB`), PROVE before the chain starts that the RAM
   clamp cannot degrade the `n_fine_cap` they were configured with, refuse
   with exit 2 if it would, and re-check it afterwards from the warnings the
   run actually raised -- the workers' included.  This is the direct remedy
   for `ADJUDICATION_NFC_8192_2026_08_10.md`'s headline finding: on this box
   the shipped `NFC=16384` ALREADY RAN AT 8192, silently.
3. **The clamp's cost model is re-derived from measurement.**  Not `frac`:
   the term that scales.  `_FINE_GRID_WORK_ARRAYS` was 16 and is measurably
   **19.1**, and there is a **2.3 GB** per-process floor the model did not
   have at all.  The shipped model was therefore 1.20x OPTIMISTIC on the term
   that grows -- it priced an `NFC=8192` congruence worker at 17.55 GB against
   a measured 24.0 GB and approved FIVE workers where four fit.

And then the thing all three exist for: **the k-ladder, measured.**

---

## 0. VERDICT

> **k = 3 is the operating point on this box at `NFC = 8192`: chain B
> 805.2 s -> 422.6 s, a MEASURED 1.905x at 63.5 % parallel efficiency
> (1.575x / 78.8 % at k=2), with every per-order output BIT-IDENTICAL to
> serial at every k -- sha256 of the accumulated field, sha256 of all six
> readout tiles, thirteen scalars per order at `rel=0 abs=0`, and
> `np.array_equal` on two full 1024^2 tiles.  The library's own clamp, with
> the corrected model, CHOOSES 3: asked for four workers it refused the
> fourth and ran three, and that arm reproduced the k=3 arm to 1.0 %.**
>
> **At `NFC = 16384` the answer is k = 1, and it is the box that says so, not
> the clamp: one order peaks at a MEASURED 84.6 GB, so two need ~170 GB
> against 137.4 GB of physical memory.  The "k=2 at 16384" confirmation is
> therefore a REFUSAL -- from both guards, before anything is spent.**
>
> **The silent degradation reproduces end to end and is now fatal.**  The
> acceptance runner `focus_scan_121.py`, run at a budget that binds, printed
> `FWHM=3.350um EE3=90.3% EE6=99.7% EE12=99.8%` -- **digit for digit the
> undegraded banner** -- on HALF the grid, in 86 s instead of 164 s, with
> **ZERO warnings emitted**, because line 34 was a bare
> `filterwarnings('ignore')`.  The only trace was the peak intensity, which
> nobody scores: 5.505e+03 against 5.529e+03.  After this change the same
> command exits **2** in 0.9 s.

---

## 1. BOX, BUILD

```
Windows 11 Pro 10.0.26200        AMD Ryzen 9 5950X, 24 logical CPUs
137.4 GB physical (128 GiB)      get_ram_budget() = 104.5 .. 105.7 GB at launch
python 3.14.6   numpy 2.4.4   scipy 1.17.1   psutil present
lumenairy 5.33.1 (working tree = perf/traced-hotpath @ 0097e5a + this change)
```

Memory is sampled at 1 Hz over the WHOLE process tree (parent + every child),
so a sub-second allocation transient can be missed; every plateau reported
here lasts tens of seconds.

---

## 2. THE SILENT DEGRADATION -- FAIL-BEFORE AND AFTER

Four fail-befores, all on the CURRENT tree before any edit
(`scratchpad/par/fail_before.py` -> `out_fail_before.txt`).

### 2.1 FB1 -- neither runner has a `__main__` guard

Asserted with the library's OWN predicate, not by reading the file:

```
  fan_multi_121.py         _script_has_main_guard = False
  focus_scan_121.py        _script_has_main_guard = False
```

`_lens_traced._script_has_main_guard` is what forces the Newton pool to
`n_workers = 1`, and `carrier._multi_looks_like_spawn_bootstrap` is what turns
the resulting `BrokenProcessPool` into a message naming the fix.  Both stay.
After the change both predicates return `True`, and `import focus_scan_121`
succeeds without running a single line of science -- which is the property a
spawn child needs and the property the guard is FOR.

### 2.2 FB2 -- the blanket filter swallows the clamp

`focus_scan_121.py:34` was a bare `warnings.filterwarnings('ignore')` with no
comment.  Driving `_memory_bounded_n_fine` at a budget that binds, under three
ambient filter states:

| ambient filter state | cap returned | clamp warnings seen |
|---|---|---|
| `simplefilter('always')` | 1024 | **1** ("the fine grid is MEMORY-LIMITED to 1024x1024 ...") |
| **`filterwarnings('ignore')`** -- focus_scan's own | 1024 | **0** |
| `fan_multi_121.py`'s three targeted ignores | 1024 | **1** |

So the degraded grid was returned identically in all three cases and only the
runner that never blanket-silenced could see it.  The fan's three ignores --
`prescription aperture`, `residual transverse`, `under-sampled` -- are the
campaign's own documented suppressions and are the ones copied into
`focus_scan_121.py`; nothing else is suppressed there any more.  Re-measured
after the change, under the runner's own filter state (i.e. by importing the
module, which the guard now makes possible): **1 warning, message present.**

### 2.3 FB3 -- the silent 8192, on this box, today

`get_ram_budget() = 105.68 GB`, no explicit budget anywhere:

| `n_fine_cap` | box budget | `ram_budget=inf` | clamp warnings |
|---|---|---|---|
| 4096 | 4096 | 4096 | 0 |
| 8192 | 8192 | 8192 | 0 |
| 12288 | **8192** | 12288 | 1 |
| **16384** | **8192** | 16384 | 1 |

That is `ADJUDICATION_NFC_8192_2026_08_10.md` sec 2.1 reproduced directly:
the runner of record asks for 16384, gets 8192, and prints an acceptance
banner that passes.  The adjudication had to inject `ram_budget=inf` from
outside to make its reference arm real, and recorded the un-overridden clamp
in a SHADOW column that read 8192 on all 64 rows.

**PASS-AFTER.**  The same configuration now refuses before the chain:

```
GRID INTENT -- design-121 fan, 32 orders, NFC=16384 CW=1
  n_fine_cap requested : 16384
  ram_budget           : auto (the box: get_ram_budget() = 104.94 GB)
  congruence workers   : 1
  clamp ceiling        : 8192  (the largest grid a 104.94 GB budget approves;
                                16384 needs 171.8 GB of budget)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
REFUSED -- design-121 fan, 32 orders, NFC=16384 CW=1
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  the RAM clamp would DEGRADE the fine grid from 16384 to 8192.
  * this run would have produced a 8192x8192 answer wearing a 16384 label ...
  Remedies:
    RAMB=inf   -- disable the clamp, having read that one order peaks at ...
    RAMB=<GB>  -- pin the budget (>= 171.8)
    NFC=8192   -- ask for the grid the box can hold
```

exit code **2**.  The distinction the exit codes carry: **2** = this box
cannot do what you asked; **3** = the pre-flight said it could and the run
degraded anyway (i.e. the model and the clamp disagree, which is a defect
report, not a configuration problem).

### 2.3b The same thing END TO END, in the acceptance runner

The probes above are arithmetic.  This one is the runner, four times, through
`scratchpad/par/run_focus.py` (which only pins `set_max_ram` and samples RSS):

| run | script | budget | banner | `pk` | wall | peak RSS | warnings | exit |
|---|---|---|---|---|---|---|---|---|
| 1 | `HEAD` | auto | 3.350 / 90.3 / 99.7 / 99.8 | 5.529e+03 | 200.8 s | 19.65 GB | -- | 0 |
| 2 | **fixed** | auto | 3.350 / 90.3 / 99.7 / 99.8 | 5.529e+03 | 200.0 s | 19.71 GB | -- | 0 |
| 3 | `HEAD` | **`set_max_ram(30)`** | **3.350 / 90.3 / 99.7 / 99.8** | **5.505e+03** | **86 s** | **6.42 GB** | **0** | **0** |
| 4 | **fixed** | `set_max_ram(30)` | -- REFUSED -- | -- | 0.9 s | 0.07 GB | -- | **2** |

Row 3 is the failure this campaign is named for, in the acceptance runner
itself.  The fine grid was clamped 8192 -> 4096; the run was **2.3x faster and
3.1x lighter**; the four scored numbers are IDENTICAL to the un-degraded run
to the precision the banner prints; and `grep -ci warning` on the whole
transcript returns **0**.  The only quantity that moved is `pk`, by 0.44 %,
and `pk` is not a scored quantity.

Rows 1 and 2 are the other half of the guard's fail-after: adding
`if __name__ == '__main__':` makes this runner's `n_workers=8` Newton pool
REACHABLE for the first time (`_script_has_main_guard` was forcing it serial),
so the answer had to be re-checked rather than assumed.  It does not move --
identical to every printed digit including `pk` -- and the wall is 200.0 s
against 200.8 s.  (It does not move because the pool still declines to
dispatch: 262 144 Newton points clears the cold bar but the polynomial route's
per-step cost does not clear `_POOL_PROMOTE_MIN_SECONDS`, exactly as
`AUDIT_TRACED_SPEED_2026_08_09` sec 6.1 measured.)

### 2.4 FB4 -- the worker clamp over-approves

At the fan's own shapes (`shape0=(1024,1024)`, `K=32`, 8 GB reserve, 105.7 GB
free):

| `n_fine_cap` | model per worker | requested 2/3/4/6/8/32 -> APPROVED |
|---|---|---|
| 8192 | 17.55 GB | 2/3/4/**5**/5/5 |
| 16384 | 69.09 GB | 1/1/1/1/1/1 |

Five workers at a MEASURED 24.0 GB each is 120 GB of a box with 105.7 GB free.
Section 3 prices it properly; section 4 runs the ladder that proves the new
number.

---

## 3. THE CLAMP, RE-DERIVED FROM MEASUREMENT

`FIX_PERF_CACHES_BLUESTEIN_2026_08_09.md` sec 7 item 4 left this open: "`frac
= 0.5` in `_memory_bounded_n_fine` was not re-derived.  With a 16-array model
it now demands 137.4 GB of budget for `n_fine = 16384` against a MEASURED
whole-process peak of 98.85 GB -- conservative by ~1.4x."

Two things had to be established before that could be answered, and both moved
the answer:

* **the 98.85 GB is stale.**  It was measured before items #2/#6/#7.  On this
  branch the same order peaks at **84.6 GB**.
* **the model's error is not a single ratio.**  A census counts arrays at ONE
  grid, so it cannot tell what scales from what does not.

### 3.1 The measurement

`validation/repro_traced_carrier_121/kladder_121.py` at `CW=1`, two orders
(`(0,0)`, `(+1,0)`), everything pinned -- `RN=1024 RS=4 NW=1 DXO=0.2 um
NOUT=8192 TILE=1024 WF=4.0 LEG=auto`, `RAMB=inf` so the grid is the one asked
for and not the one the clamp allows -- with the whole process tree sampled at
1 Hz.  `NFC=4096` needs `OTEG=warn` (its retrace pitch trips the D6 refusal);
the other two run the shipped `'error'`.

| `n_fine` | chain-B s/order | **peak RSS** | implied work arrays at zero intercept |
|---|---|---|---|
| 4096 | 45.2 | **7.123 GB** | 26.5 |
| 8192 | 126.3 | **23.968 GB** | 22.3 |
| 16384 | 544.4 | **84.589 GB** | 19.7 |

The implied count FALLS as the grid grows.  That is the signature of a fixed
cost, not of a smaller array set, and it is why one constant could not be
right at both ends.  A straight line in `n_fine^2` fits all three to within
3.5 %:

```
   peak(n) = 305.9 B/px * n^2  +  2.635 GB
             = 19.12 complex128-equivalents           intercept 2.635 GB
   pairwise slopes: 20.92 (4096-8192), 19.24 (4096-16384), 18.82 (8192-16384)
```

### 3.2 What was set, and why not `frac`

| constant | was | now | why |
|---|---|---|---|
| `_FINE_GRID_WORK_ARRAYS` | 16 | **20** | the measured slope is 19.1; 20 is the round-up, so the grid term is an upper bound at every measured point |
| `_FINE_GRID_BASE_BYTES` | (did not exist) | **2.3 GB** | the measured intercept 2.635 GB less the 0.369 GB the `_MULTI_WORKER_GRID_FACTOR` term already charges for that measurement's own 1024^2 input |
| `_FINE_GRID_RAM_FRAC` | 0.5 | **0.5** | unchanged -- see below |

**`frac` was NOT moved, and that is a decision with a reason, not an
omission.**  `tests/unit/test_fix_newton_pool_memory.py::
test_the_budget_rule_matches_the_sibling_pool` pins
`_lens_traced._NEWTON_POOL_RAM_FRAC == carrier._FINE_GRID_RAM_FRAC` -- "the
two clamps that meet on the exact final leg must speak the same language" --
and `_lens_traced.py`'s pool internals are outside this change's scope.  More
to the point, `frac = 0.5` now has a MEASURED justification it did not have
before: the rule prices the GRID against half the budget, and the other half
has to cover the process floor, which is now known to be 2.3 GB.  Half a
budget covers a 2.3 GB floor for any budget above ~4.6 GB, so the reserve is
sufficient rather than merely cautious -- that is the statement the old `frac`
could not make.

**Where each term is spent, and where it is not.**  The floor is charged by
`_fine_grid_peak_bytes` -- i.e. by `_multi_resolve_workers`, where k copies of
it exist at once, and by the runners' pre-flight.  It is deliberately NOT
charged by `_fine_grid_ceiling`: that rule prices ONE process's grid against
`frac * budget`, the remaining fraction IS the allowance for the floor, and
charging it twice would refuse every grid down to `_FINE_GRID_MIN` on a box
whose budget comfortably holds one.  The floor is also a design-121-CLASS
figure, not a universal python floor, and the ceiling is called by everything.

### 3.3 The model against the measurement

`_fine_grid_peak_bytes(n, n_px=1024^2)` with (20, 2.3 GB):

| `n_fine` | measured | model | model / measured |
|---|---|---|---|
| 4096 | 7.123 GB | 8.038 GB | **1.128** |
| 8192 | 23.968 GB | 24.144 GB | **1.007** |
| 16384 | 84.589 GB | 88.568 GB | **1.047** |

A tight upper bound at all three, which is the property a worker clamp needs:
under-pricing is an OOM, over-pricing is only a lost worker.

### 3.4 What the clamp now approves

```
box budget 105.06 GB          ceiling = 8192
  n= 4096 -> 4096      clamp needs   10.7 GB of budget
  n= 8192 -> 8192      clamp needs   42.9 GB of budget
  n=12288 -> 8192      clamp needs   96.6 GB of budget
  n=16384 -> 8192      clamp needs  171.8 GB of budget

D8 worker clamp, fan shape (1024^2, K=32, 8 GB reserve, 105.1 GB free):
  NFC= 8192  per-worker MODEL 24.14 GB  requested 2/3/4/6/8/32 -> 2/3/4/4/4/4
  NFC=16384  per-worker MODEL 88.57 GB  requested 2/3/4/6/8/32 -> 1/1/1/1/1/1
```

The `NFC=8192` row moved from 5 to **4**, and section 4 measures whether 4 is
right.  `16384` stays at 1, now for a measured reason: one order really does
peak at 84.6 GB.

**`16384` still needs an explicit budget on this box, and should.**  The clamp
demands 171.8 GB where the run measurably peaks at 84.6 GB -- a 2.03x margin
against the shipped model's 1.62x, i.e. this change made the DEFAULT more
conservative, not less.  That is the correct direction here: 84.6 GB is 80 %
of everything currently free on this box, and a default that hands 80 % of
free memory to one leg without being asked is the same class of decision as
the one that made the clamp approve six workers.  What was wrong was never the
conservatism -- it was that the refusal was silent.  It is now an exit code
with four named remedies (sec 2.3).

---

## 4. THE K-LADDER, MEASURED

### 4.1 Configuration

`validation/repro_traced_carrier_121/kladder_121.py`, which runs
`fan_multi_121.py` UNMODIFIED under an owned namespace and records the wall,
the memory and a sha256 of every output.  SIX orders, the same six at every k,
chosen as a compact block so the common grid stays small enough that the
parent is not the experiment:

```
KEEP='-1,-1;0,-1;1,-1;-1,0;0,0;1,0'   NOUT=8192   DXO=0.2 um   TILE=1024
RN=1024  RS=4  NW=1  LEG=auto  WF=4.0  OTEG=error (the shipped default)
NFC=8192   RAMB=48   RAMRES=4
```

`RAMB=48` is not decoration.  The worker budget is DIVIDED, not copied
(`_multi_capture_worker_state`), so at `RAMB=auto` the leg would see
`105 / k` GB and the clamp would silently degrade the grid from k=3 upward --
the arms would then not be comparable at all.  With the budget pinned, all
four arms provably run the same 8192 grid, and each one asserts it afterwards.
Four of these six orders are in the adjudication's DEGRADED class, i.e. the
ones most sensitive to the grid; that is deliberate.

### 4.2 The ladder

| k | chain B | s/order | speed-up | efficiency | peak tree | largest child | acceptance |
|---|---|---|---|---|---|---|---|
| **1** | 805.2 s | **134.20** | 1.000 | 100 % | 23.59 GiB (25.3 GB) | -- | PASS |
| **2** | 511.1 s | **85.18** | **1.575** | **78.8 %** | 46.32 GiB (49.7 GB) | 24.21 GiB (26.0 GB) | PASS |
| **3** | 422.6 s | **70.44** | **1.905** | **63.5 %** | 67.59 GiB (72.6 GB) | 24.20 GiB (26.0 GB) | PASS |
| 4 -> **3** | 427.0 s | 71.17 | 1.886 | (see below) | 67.57 GiB (72.6 GB) | 24.00 GiB (25.8 GB) | PASS |

**The k=4 row is a k=3 row.**  The library's own clamp refused the fourth
worker, with the corrected model, in its own words:

```
propagate_traced_carrier_chain_multi: congruence_workers=4 would need
~96.6 GB (24.1 GB per worker at 1024^2 complex128) but only 104.2 GB is
available with a 8 GB reserve; running 3 worker(s) instead.
```

so that arm is an unplanned REPRODUCIBILITY check on k=3 rather than a fourth
point: 427.0 s against 422.6 s, **1.0 % apart**, with identical output hashes.
The shipped clamp would have approved five.

`k=2`'s 78.8 % reproduces `AUDIT_TRACED_SPEED_2026_08_09` sec 3.4's 81 %
(measured there on two different orders at the same cap).  The efficiency
falls because each congruence keeps the library's own 8 FFTW threads on a
24-thread box and the orders compete for memory bandwidth; `fft_infra` is not
in `_WORKER_STATE_MODULES`, so nothing coordinates them.

### 4.3 The acceptance: bit-identity across k

`kladder_compare.py`, on the artifacts rather than on the banner:

```
  BIT-IDENTITY vs k1 (the D8 contract)
    k2         IDENTICAL
    k3         IDENTICAL
    k4         IDENTICAL

  FULL-ARRAY CHECK on the dumped tiles (np.array_equal)
    k2  tile_0_0  array_equal=True  max|delta|=0.000e+00  sha=d7fc7c2f3b8619ca
    k2  tile_1_0  array_equal=True  max|delta|=0.000e+00  sha=0b7ef51f19266129
    k3  tile_0_0  array_equal=True  max|delta|=0.000e+00  sha=d7fc7c2f3b8619ca
    k3  tile_1_0  array_equal=True  max|delta|=0.000e+00  sha=0b7ef51f19266129
    k4  tile_0_0  array_equal=True  max|delta|=0.000e+00  sha=d7fc7c2f3b8619ca
    k4  tile_1_0  array_equal=True  max|delta|=0.000e+00  sha=0b7ef51f19266129

  VERDICT: BIT-IDENTICAL AT EVERY k
```

"IDENTICAL" there covers, per arm: the sha256 of the whole accumulated field,
the sha256 of all SIX readout tiles, and thirteen per-order scalars
(`power_in`, `power_exit`, `power_out`, `throughput`, `capture`, `cellP`,
`field_pct`, `fwhm_um`, `ee3`, `ee6`, `ee12`, `chief_x_um`, `chief_y_um`)
compared at `rel=0 abs=0`.  The two `np.array_equal` rows are the D8 contract
verified on two full 1024^2 complex128 tiles, as asked.

All four arms printed `VERDICT: PASS` with the same
`max |share/design - 1| = 0.00045   rms 0.00029`, and all four printed the
post-run grid check.

### 4.4 The k the CLAMP chooses, at both caps

| `n_fine_cap` | model per worker | clamp's k | measured basis |
|---|---|---|---|
| **8192** | 26.34 GB | **3** | largest child 26.0 GB; 3 workers held a 72.6 GB tree with 32.5 GB still free |
| **16384** | 90.77 GB | **1** | one order peaks at 84.6 GB; two would need ~170 GB against 137.4 GB of physical memory |

The runners' pre-flight, which additionally prices the parent's common-grid
accumulator, agrees at both caps:

```
  32 orders NFC=16384 RAMB=inf CW=1 (NOUT=32768) : 1 x 90.8 + 0.0 =  90.8 GB  ACCEPTED
  32 orders NFC=16384 RAMB=inf CW=2              : 2 x 90.8 + 21.7 = 203.2 GB REFUSED
   6 orders NFC=8192  RAMB=48  CW=3 (NOUT=8192)  : 3 x 26.3 +  5.6 =  84.6 GB ACCEPTED
   6 orders NFC=8192  RAMB=48  CW=4              : 4 x 26.3 +  5.6 = 110.9 GB REFUSED
```

**So the "k=2 at 16384" confirmation is a REFUSAL, and it is the honest
answer.**  It was not run because it cannot be: 2 x 84.6 GB measured is
~170 GB on a 137.4 GB box, and the run would page rather than compute.  Both
guards say so before spending anything -- the library clamp returns 1, the
pre-flight exits 2 -- and the box that would hold it needs
`2 x 90.8 + 21.7 + 8 = 211 GB`.  The accumulator is charged only at `k > 1`
because at k=1 the leg peak and the accumulator are SEQUENTIAL; that is
measured (see `_grid_intent.py`, and the adjudication's own arm B peaked at
77-86 GB with a 17.2 GB accumulator live in the same process).

### 4.5 What it is worth on the fan of record

Per-order, at `NFC=8192` on this box: 134.20 s serial -> **70.44 s at k=3**.
Over the 32-order fan that is **1.19 h -> 0.63 h**.  At `NFC=16384`, k is 1
and the fan stays serial on this box; the parallel win there needs a bigger
box or the mesh (`AUDIT_TRACED_SPEED_2026_08_09` sec 3.7), not a better clamp.

---

## 5. TESTS AND SUITES

### 5.1 A RED suite was already on the branch

`tests/unit/test_niche_p2_guards.py` -- the file that pins the memory clamp --
was **failing at `0097e5a`, before this change**, on two tests.  Commit
`6464384` re-measured `_FINE_GRID_WORK_ARRAYS` from 4 to 16 and left the
suite's cap ladder at its 4-array values.  Proved rather than asserted, by
transcribing `0097e5a`'s own expression and evaluating it at both counts:

| budget | the suite expects | HEAD's formula at `n_work=16` | at `n_work=4` |
|---|---|---|---|
| 0.25 GiB | 1024 | **512** | 1024 |
| 1 GiB | 2048 | **1024** | 2048 |
| 4 GiB | 4096 | **2048** | 4096 |
| 16 GiB | 8192 | **4096** | 8192 |
| 34 GiB | 16384 | **8192** | 16384 |
| 136 GiB | 32768 | **16384** | 32768 |

The refactor here reproduces `0097e5a`'s arithmetic exactly (the `n_work=16`
column, row for row), so the failure is inherited, not caused.  It is fixed
with the ladder re-derived at the new constant AND with the constant asserted
FIRST, so the next re-measurement fails with the formula in the message
instead of an unexplained `512 != 1024`.

### 5.2 New pins

`tests/unit/test_niche_d8_congruence_workers.py` gains four, all about the
model rather than the physics:

* `test_the_cost_model_is_an_upper_bound_on_the_measured_peak` -- carries the
  three MEASURED peaks and requires the model to sit above every one of them
  and within 1.5x.  Under-pricing is an OOM; over-pricing costs a worker.
* `test_the_worker_clamp_and_the_grid_clamp_share_one_model` -- source-level,
  deliberately: the approved worker count is a function of live free memory,
  so pinning one would be a test that fails on what else the box is doing.
* `test_the_grid_ceiling_is_the_pure_form_of_the_ram_clamp` -- the pre-flight's
  proof is only worth something if `_fine_grid_ceiling` and
  `_memory_bounded_n_fine` agree at every budget.
* `test_the_grid_ceiling_does_not_charge_the_process_floor` -- the small-box
  regression this design deliberately avoids.

### 5.3 Suite results

Windows (python 3.14.6, numpy 2.4.4), `-q -p no:randomly`:

| suite | result |
|---|---|
| `test_niche_tight_focus_readout.py` + `test_carrier_referenced.py` | **33 passed** |
| `test_niche_d2_chain_multi.py` | **38 passed** |
| `test_niche_d6_exact_tilted_leg.py` + `test_niche_d9_grid_origin.py` | **61 passed** |
| `test_niche_d8_congruence_workers.py` + `test_fix_newton_pool_memory.py` + `test_niche_newton_pool_both_fits.py` + `test_memory_guardrail.py` + `test_niche_p2_guards.py` | **139 passed** |
| `test_niche_p2_guards.py` + `test_niche_d3_guards.py` (after the constant change) | **53 passed** |

WSL CI proxy (python 3.12.3, `~/lumvenv`), the representative arm:

```
test_niche_p2_guards.py + test_niche_d8_congruence_workers.py
+ test_niche_d3_guards.py                             87 passed in 223 s

fan_multi_121.py     guard=True        work arrays 20  base 4.5 GB  frac 0.50
focus_scan_121.py    guard=True        budget 104.42 GB -> ceiling 8192
kladder_121.py       guard=True        D8 clamp NFC=8192, requested 8 -> 3
  model( 8192) 26.344 GB vs measured 25.996 -> 1.013
  model(16384) 90.768 GB vs measured 84.589 -> 1.073
```

i.e. the guard predicate, the constants and the clamp's choice are the same on
both builds.

---

## 6. WHAT THIS DOES NOT DO

1. **`k=2` at `NFC=16384` was NOT run.**  It cannot be on this box (sec 4.4):
   2 x 84.6 GB measured against 137.4 GB physical.  What is reported instead
   is the refusal from both guards and the arithmetic behind it.  A 211 GB box
   would settle it.
2. **The ladder is SIX orders, not thirty-two.**  Per-order wall and per-worker
   peak are order-scale invariants (the four arms agree to 1 %), but the
   32-order projection in sec 4.5 is arithmetic, not a run.
3. **`_FINE_GRID_RAM_FRAC` is still 0.5 and is still not independently
   derived.**  It is now MEASURED to be sufficient (it must cover a 2.3-4.5 GB
   floor out of half the budget), but whether 0.5 is the RIGHT reserve rather
   than merely a sufficient one is a question about the Newton pool's copy of
   the same constant, and that file is out of scope here.  The consequence is
   live: `NFC=16384` needs an explicit `RAMB` on any box under 172 GB.
4. **The 2.3 GB / 4.5 GB floor is a design-121-class figure.**  It was
   measured on this leg, this input shape and this branch, and it grew by 1.8x
   between a two-order and a six-order process because the leg's caches grow
   with the orders a process runs.  A 32-order-per-worker run may exceed it;
   the caches are byte-capped since item #6, so it should saturate, but that
   was not measured here.
5. **Peak RSS is sampled at 1 Hz.**  A sub-second allocation transient is
   invisible to it.  Every plateau reported here lasts tens of seconds.
6. **Nothing was done about the efficiency roll-off.**  63.5 % at k=3 is
   measured, not explained: `fft_infra` is not in `_WORKER_STATE_MODULES`, so
   k workers each start 8 FFTW threads on a 24-thread box with nothing
   coordinating them.  A per-worker thread budget is the obvious next probe
   and is not attempted here.
7. **`_lens_traced.py`'s Newton-pool internals, `pmm/**` and the fan's own
   physics are untouched.**  The only library change is the clamp
   constants/model region of `carrier.py`.

---

## APPENDIX -- artifacts

In-repo:

| path | what |
|---|---|
| `validation/repro_traced_carrier_121/_grid_intent.py` | the loud-choice module: `resolve_ram_budget`, `preflight`, `record_warnings`, `assert_no_grid_degradation` |
| `validation/repro_traced_carrier_121/kladder_121.py` | the k-ladder driver (import-safe; runs the fan unmodified) |
| `validation/repro_traced_carrier_121/kladder_compare.py` | the bit-identity + speed comparison |

Session scratchpad `scratchpad/par/`:

| file | what |
|---|---|
| `fail_before.py`, `out_fail_before.txt` | FB1-FB4 |
| `calib/calib_nfc{4096,8192,16384}.json`, `calib_nfc*.log` | the three-grid scaling measurement (sec 3.1) |
| `kl8192/k{1,2,3,4}.json`, `k*_tiles.npz`, `kl8192_k*.log` | the k-ladder arms and their tiles |
| `kl8192_compare.txt` | the bit-identity transcript |
| `kl8192_k2_UNGUARDED_DRIVER.log` | the k=2 arm that died because the DRIVER was unguarded, and then hung |
| `demo_D{1,2,3}.log` | the three refusal transcripts |
| `run_focus.py`, `focus_{HEAD,NEW,HEAD_ram30,NEW_ram30}.log` | the end-to-end silent-degradation reproduction (sec 2.3b) |
| `wsl_suite.txt` | the WSL arm |
| `orig/*_HEAD.py` | `git show HEAD:` copies of both runners, used as the fail-before |
