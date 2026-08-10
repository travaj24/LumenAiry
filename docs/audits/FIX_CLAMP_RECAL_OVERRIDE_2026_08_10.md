# FIX -- the fine-grid memory model RE-MEASURED on the final tree, and the record-run intent allowed to proceed

**2026-08-10.  Branch `perf/traced-hotpath`, checked out at `0b4a385` and
modified in the WORKING TREE only -- no commit, no push, no tag, no CHANGELOG
entry.**

Two things ship, and the first one inverts the premise it was started on.

1. **`_FINE_GRID_WORK_ARRAYS` / `_FINE_GRID_BASE_BYTES` are re-measured on the
   FINAL round-2 tree: (22, 2.6 GB) -> (24, 1.8 GB).**
2. **`_grid_intent.preflight` no longer refuses an explicit intent it cannot
   approve.**  A binding BOX check warns and proceeds; the CLAMP check, and a
   run the box physically cannot allocate, still refuse.

---

## 0. VERDICT

> **The shipped (22, 2.6 GB) split is UNDER a measured worker child on this
> tree.**  Re-measured at the same shapes with the same harness, an
> `n_fine = 8192` congruence worker at k=3 peaks at **26.737 GB** against a
> model of **26.591 GB** -- **0.995x**.  The envelope had stopped being an
> envelope.  Under-pricing a worker is an OOM, which is the side of this trade
> that costs a run rather than a warning.
>
> **The premise this round was started on is wrong in direction.**  Round 2
> removed a measured 8.6 GB coords transient at `n_fine = 16384`, so the
> constants were expected to be stale-CONSERVATIVE.  They are stale-OPTIMISTIC
> at the binding end: the 16384 whole-process peak **ROSE 84.589 -> 87.925 GB
> (+3.9 %)** and the 8192 k=3 child rose **+2.9 %**, while only the small end
> fell (4096 child **-2.6 %**).  Round 2 moved this leg's footprint in BOTH
> directions -- it also added ~1.05 GB of resident pyFFTW plan buffers by
> routing two more call sites through the dispatcher -- and at 8192 and above
> the additions won.  **A removed transient is not a lower peak until it is
> measured.**
>
> **Re-derived: (24, 1.8 GB), worst ratio 1.274, tightest 1.045, an upper bound
> at all seven measured points** (was worst 1.313 and tightest 0.995, i.e. not
> a bound).  Both constants had to move, because the set did not move one way:
> bounding the k=3 child with 2 % of margin at slope 22 needs a 2.9 GB floor,
> and that floor prices the 4096 child at 1.35x.  The floor is smaller for a
> MEASURED reason -- the four whole-process peaks now fit a line with intercept
> 2.102 GB, i.e. a **1.733 GB** per-process floor once the input-grid term is
> removed, and 1.8 GB is that rounded up.
>
> **`n_fine = 16384` does NOT fit unrefused, and the refusal does not
> dissolve.**  It is more expensive than believed, not less.  The measured
> 87.9 GB does fit inside 94.4 GB free -- by 6.5 GB, which is less than the
> 8 GB reserve -- and the honest model prices it at **105.2 GB**, 18.8 GB
> beyond what the pre-flight will approve.  So the record grid reaches the box
> through item 2's warning path, which is therefore **the normal path for the
> grid of record on tesla-ryzen, not a rare-box fallback**.  The 16384
> measurement in this document was itself taken through it.
>
> **The D8 clamp's choices are unchanged at both caps**: k=3 at `NFC=8192`,
> k=1 at `NFC=16384`, and the six-rung cap ladder is identical rung for rung.
> The model moved; nothing it decides did.

---

## 1. WHAT WAS RE-MEASURED, AND WHY IT HAD TO BE

`FIX_PERF_ROUND2_2026_08_10.md` sec 3.2(b) replaced the `map_coordinates`
coordinate build -- `np.indices((N, N), float64)` plus `np.array([ii / sub,
jj / sub])`, a 12.9 GB transient for a 4.295 GB answer at `n_fine = 16384` --
with one `arange(N) / sub` broadcast into a pre-allocated `(2, N, N)`.  That
takes **8.6 GB** of transient off the priced path at 16384, AFTER every peak
the `(22, 2.6 GB)` envelope was fitted to had been taken.  Round 2 also routed
`_fourier_upsample_crop` and `_shift_envelope` through the FFT dispatcher and
measured the plan cache saturated at 8 of 8 entries with **+1.05 GB** of
resident pyFFTW buffers (its sec 7.3).

The constants were therefore calibrated against a tree that no longer exists,
and the NET direction is not knowable from the arithmetic.  That is why this is
a re-measurement.

### 1.1 Box and build

```
Windows 11 Pro 10.0.26200        AMD Ryzen 9 5950X, 24 logical CPUs
137.4 GB physical (128 GiB)      90.2 .. 106.1 GB free at the arms' launches
python 3.14.6   numpy 2.4.4   scipy 1.17.1   psutil present
lumenairy 5.33.1, branch perf/traced-hotpath @ 0b4a385 (working tree)
OMP / OPENBLAS / MKL / NUMEXPR_NUM_THREADS = 1
```

### 1.2 Harness and configuration -- the SAME ones, deliberately

`validation/repro_traced_carrier_121/kladder_121.py`, which runs
`fan_multi_121.py` UNMODIFIED under an owned namespace and samples the WHOLE
PROCESS TREE at 1 Hz, reporting the tree total, the parent, and the largest
single CHILD.  Configuration pinned exactly as `FIX_PERF_PARALLEL` sec 3.1:

```
RN=1024  RS=4  NW=1  DXO=0.2 um  NOUT=8192  TILE=1024  WF=4.0  LEG=auto
```

`NFC=4096` needs `OTEG=warn` (its retrace pitch trips the D6 refusal); the
others run the shipped `'error'`.  `RAMB=inf` on the two-order arms so the grid
is the one asked for and not the one the clamp allows; `RAMB=48` with
`RAMRES=4` on the six-order k>1 arms, because the worker budget is DIVIDED and
`auto` would silently degrade the grid at k >= 3 and make the arms
incomparable.

Peaks are reported by the harness in GiB under a field named `peak_*_gb`; every
byte figure below is that value x 2^30.  Reading one as decimal GB is what
produced the retracted 1.540x claim of `VERIFY_PERF_BRANCH` D4, so the raw GiB
is carried in every label.

### 1.3 How the 16384 arm was made to happen

The pre-flight refuses `NFC=16384` on this box -- the finding round 2 recorded
and could not get past (its sec 1.3).  Rather than bypassing the runner (the
adjudication's `output_grid['ram_budget'] = inf` monkeypatch), the arm was run
through the runner's OWN path with `RAMB=inf` and item 2's override in place.
The pre-flight printed its `PROCEEDING OVER A BINDING MEMORY MODEL` warning and
ran.  Verbatim, from `n16384_k1.log`:

```
GRID INTENT -- design-121 fan, 2 orders, NFC=16384 CW=1
  ram_budget           : inf (RAM clamp DISABLED, explicitly)
  clamp ceiling        : unbounded  (the largest grid an unbounded budget
                                     approves; 16384 needs 189.0 GB of budget)
  modelled peak        : 1 x 97.5 GB per worker + 0.0 GB parent = 97.5 GB
                         (free 99.1 GB, reserve 8 GB)
~~~~~~~~~~~~~~~~~~~~~~ (74 cols) ~~~~~~~~~~~~~~~~~~~~~~
PROCEEDING UNDER WARNING -- design-121 fan, 2 orders, NFC=16384 CW=1
  97.5 GB modelled  >  99.1 GB free - 8 GB reserve   (137.4 GB physical)
  the requested grid is UNCHANGED at 16384 -- nothing here degrades it.
  ONBOX=error refuses instead; ONBOX=warn (default) is what you have.
```

(that transcript is at the PRE-recalibration constants, which is what the arm
ran under -- it is evidence, so it is kept as it was emitted).  The run
completed, exit 0, `min_avail` 11.4 GiB at its worst.  **The override is not
only shipped but exercised end to end by the measurement this document rests
on.**

---

## 2. TASK 1 -- THE RE-MEASURED SET AND THE RE-DERIVED ENVELOPE

### 2.1 MEASURED, on the final tree

Seven points, five arms plus two batch-2 arms.  `was` is the corresponding
`FIX_VERIFY_PERF_2026_08_10.md` sec 4.3 row.

| `n_fine` | what | GiB | **now** | was | move |
|---|---|---|---|---|---|
| 4096 | 2 orders, whole process | 6.574 | **7.059 GB** | 7.123 GB | -0.9 % |
| 4096 | 2 orders, whole process, re-run | 6.603 | **7.090 GB** | 7.120 GB | -0.4 % |
| 4096 | 2 orders, largest CHILD at k=2 | 6.296 | **6.760 GB** | 6.937 GB | **-2.6 %** |
| 8192 | 2 orders, whole process | 22.927 | **24.618 GB** | 23.968 GB | **+2.7 %** |
| 8192 | 6 orders, largest CHILD at k=2 | 24.377 | **26.175 GB** | 26.001 GB | +0.7 % |
| 8192 | 6 orders, largest CHILD at k=3 | 24.901 | **26.737 GB** | 25.985 GB | **+2.9 %** |
| 16384 | 2 orders, whole process | 81.887 | **87.925 GB** | 84.589 GB | **+3.9 %** |

The two 4096 whole-process arms agree to **0.44 %**, which is the
reproducibility figure this set carries.

**The set did not move one way, and that is the whole result.**  Only the
small end fell.  Everything at 8192 and above rose.  The 8.6 GB coords
transient that round 2 removed at 16384 was evidently not what set that peak;
the plan-buffer growth and whatever else round 2 changed more than covered it.

### 2.2 The shipped model against it

| point | measured | model (22, 2.6 GB) | ratio |
|---|---|---|---|
| 4096, whole process | 7.059 GB | 8.875 GB | 1.257 |
| 4096, whole process, re-run | 7.090 GB | 8.875 GB | 1.252 |
| 4096, CHILD k=2 | 6.760 GB | 8.875 GB | **1.313** |
| 8192, whole process | 24.618 GB | 26.591 GB | 1.080 |
| 8192, CHILD k=2 | 26.175 GB | 26.591 GB | 1.016 |
| **8192, CHILD k=3** | **26.737 GB** | **26.591 GB** | **0.995  <- UNDER** |
| 16384, whole process | 87.925 GB | 97.458 GB | 1.108 |

Worst **1.313**, tightest **0.995**.  Both bars are broken: the 4096 child is
outside the 1.3x this round targets, and the k=3 child is not bounded at all.

### 2.3 What a LINE says now

Least squares through the four WHOLE-PROCESS peaks (a per-process quantity; a
k>1 TREE total is not one and is not fitted):

```
   peak(n) = 320.5 B/px * n^2  +  2.102 GB
           = 20.03 complex128-equivalents      (was 19.12)
   per-process FLOOR = 2.102 - 0.369 = 1.733 GB
       (0.369 GB is the _MULTI_WORKER_GRID_FACTOR term this measurement's own
        1024^2 input already pays)
   residuals  +5.9 %  +5.5 %  -4.1 %  +0.2 %
   pairwise slopes  21.78 (4096-8192)  20.08 (4096-16384)  19.65 (8192-16384)
```

The implied count at ZERO intercept still FALLS with the grid -- 26.4 at 4096,
22.9 at 8192, 20.5 at 16384 -- which is the same fixed-cost signature the
original derivation found.  The 8192 **k=3 CHILD** implies **24.9**, above
every whole-process figure at that grid: a worker's own peak sits above the
whole-process line, and by more than it used to.

### 2.4 The re-derivation, and why BOTH constants had to move

Same rule as `FIX_VERIFY_PERF` sec 4.3: minimise the worst model/measured ratio
subject to (a) bounding EVERY measured point, whole-process and child, (b) at
least 2 % of margin over the binding child, (c) an integer slope and a floor on
a 0.1 GB grid that clears the process's own commit.

Constraint (b) at the k=3 child and the 1.3x bar at the 4096 child are the two
that bind, and together they fix the slope:

```
  bound the 4096 child at <= 1.30x  :  0.26844*n_work + base <=  8.419 GB
  bound the 8192 k=3 child at >=1.02:  1.07374*n_work + base >= 26.903 GB
  subtract                          :  0.80526*n_work        >= 18.484
                                    ->  n_work >= 22.95
```

So **slope 22 is infeasible**, not merely suboptimal: at 22 the margin
requirement needs `base >= 2.9 GB`, which prices the 4096 child at **1.35x**.
Slope 23 leaves a base window of `[2.207, 2.245] GB`, which is empty on a
0.1 GB grid.  **Slope 24** leaves `[1.8, 1.976] GB`, and 1.8 GB is the measured
floor (sec 2.3) rounded up.

| constant | was | **now** |
|---|---|---|
| `_FINE_GRID_WORK_ARRAYS` | 22 | **24** |
| `_FINE_GRID_BASE_BYTES` | 2.6 GB | **1.8 GB** |
| `_PARAXIAL_BASE_BYTES` | 1.0 GB | 1.0 GB (not re-measured -- sec 6) |
| `_FINE_GRID_RAM_FRAC` | 0.5 | 0.5 |

`_fine_grid_peak_bytes(n, n_px=1024^2)` at **(24, 1.8 GB)**:

| point | measured | model | ratio |
|---|---|---|---|
| 4096, whole process | 7.059 GB | 8.612 GB | 1.220 |
| 4096, whole process, re-run | 7.090 GB | 8.612 GB | 1.215 |
| 4096, CHILD k=2 | 6.760 GB | 8.612 GB | **1.274** |
| 8192, whole process | 24.618 GB | 27.939 GB | 1.135 |
| 8192, CHILD k=2 | 26.175 GB | 27.939 GB | 1.067 |
| 8192, CHILD k=3 | 26.737 GB | 27.939 GB | **1.045** |
| 16384, whole process | 87.925 GB | 105.248 GB | 1.197 |

**Worst 1.274, tightest 1.045.**  An upper bound at every point, with 4.5 % of
headroom over the row the clamp is actually decided by (was 1.3 %, then 1.6 %,
then negative).  The headroom is stated rather than minimised on purpose:
under-pricing is an OOM, over-pricing costs a worker or a warning.

**The 1.5x bar in `test_niche_d8_congruence_workers.py` is left at 1.5**, not
tightened to the 1.3 this derivation targets.  1.274 against a 1.3 bar is 2 %
of slack, and a pin that tight fails on the next re-measurement for reasons
that are not defects.  The bar that matters -- and that caught this -- is
`model >= measured`, which the k=3 row now fails on the old constants.

**A limitation of that pin, stated because it is the one that let this
through.**  It compares the model against the rows the FILE carries, so it can
only catch drift when the rows are refreshed.  It was green at `0b4a385` on
stale rows while the model was already under a real worker.  Nothing but a
re-measurement finds that, which is the argument for re-measuring after any
change to the fine leg -- now written at the constant.

### 2.5 The trade that was DROPPED

`FIX_VERIFY_PERF` sec 4.3 rejected a steeper slope specifically to keep the
16384 price under what the pre-flight would approve for one worker
("(23, 2.0 GB) reads 1.232 worst -- better -- but prices `n_fine = 16384` at
101.2 GB per worker, which is where `_grid_intent.preflight` stops approving").
That constraint is deliberately dropped here, and item 2 is why: **the
pre-flight no longer refuses what it cannot approve.**  Buying a cheaper 16384
price with an envelope that under-prices a worker child was the trade on offer,
and it is the wrong one.

---

## 3. TASK 2 -- ALLOW WITH WARNINGS, on an explicit intent

### 3.1 What was wrong with refusing

`preflight` runs two checks and, until this change, failed both the same way:
`SystemExit(2)`.  They are not the same kind of failure.

* **The CLAMP check is about a WRONG ANSWER.**  If `_fine_grid_ceiling` is
  below the requested `n_fine_cap`, the leg returns a smaller grid under the
  label the runner configured -- the silent-8192 failure of
  `ADJUDICATION_NFC_8192_2026_08_10.md` sec 2.1, where a 4.7-hour reference run
  and a 1.3-hour exploration run were the same run with different labels.
* **The BOX check is about CAPACITY**, decided by a model that is deliberately
  an UPPER-BOUND ENVELOPE.  "Modelled need > free" does not mean "will not
  fit"; it means "the envelope says it might not".  Refusing an EXPLICIT intent
  on that basis is an envelope over-ruling an operator who has already read
  what the run costs -- and it had a measured cost: round 2 could not profile
  the fan's own shipped default at all.

### 3.2 What ships

`preflight` gains `on_box_budget` -- the library's own `on_*` action pattern
(`on_ram_cap`, `on_mem_budget`, `on_tilt_exact_grid`), dispatched through
`carrier._guard_dispose` rather than a private `warnings.warn` so it cannot
drift from the family.  `ONBOX=<action>` sets it for a whole run;
`preflight(on_box_budget=...)` for one call.

| condition | disposition |
|---|---|
| the CLAMP would bind | **refuse, exit 2** -- no disposition, ever |
| modelled need `>=` TOTAL physical memory | **refuse, exit 2** -- no disposition |
| modelled need `>` free - reserve, `<` total | **`'warn'` (default): RuntimeWarning naming modelled need / free / the MEASURED reference peak, and PROCEED at the requested grid** |
| the same, with `'error'` | refuse, exit 2 (the pre-change behaviour) |

`'ignore'` is deliberately NOT in the vocabulary -- the module exists to make
the grid choice loud, and a silent over-commit is the failure it prevents.  The
two-word vocabulary is pinned, and the rejection says why.

**The grid is never what gives way.**  `preflight` returns a RAM BUDGET, never
a grid, so there is no path by which warn-and-proceed hands the runner a
smaller `n_fine_cap`.  The post-run `assert_no_grid_degradation` -- the
shadow-`n_fine` detector -- is unchanged and still exits 3 on a clamp that
actually fired.

**One thing that had to be got right rather than assumed.**  The override's own
warning is teed into the same recorder that post-run check reads.  Had its text
carried any of `MEMORY-LIMITED` / `COUNT-LIMITED` / `RESOLUTION-LIMITED`, every
warn-and-proceed run would have failed its own post-run check with exit 3 --
a false defect report on a correct run.  The exclusion is pinned against
`_DEGRADE_MARKS` itself, not a transcription of it.

**Also charged to the TOTAL check**: `k=2 at NFC=16384` stays refused
(2 x 105.2 GB modelled, and 2 x 87.9 GB MEASURED, against 137.4 GB physical).
Before this change a binding FREE check refused it; that is now the TOTAL
check's job alone, and it is pinned as such.

### 3.3 FAIL-BEFORE / FAIL-AFTER, executed

Three arms on a SCALED config (`n_fine_cap = 1024` against synthetic box
memory), run against the module before and after the edit:

| arm | fixture | BEFORE (`0b4a385`) | AFTER |
|---|---|---|---|
| **A** binding on FREE, explicit `RAMB=inf`, 4 GB free / 64 GB total | the case this changes | **exit 2**, 0 warnings | **exit 0**, returns `inf`, **1 RuntimeWarning** |
| **B** modelled >= TOTAL, 1 GB free / 2 GB total | physically impossible | exit 2, 0 warnings | **exit 2**, 0 warnings, "PHYSICALLY cannot allocate" |
| **C** `RAMB=0.02`, 900 GB free | the CLAMP would degrade 1024 -> 128 | exit 2, 0 warnings | **exit 2**, 0 warnings, unchanged text |

**As a suite**: the new module run against the pre-change `_grid_intent.py`
(restored with `git show 0b4a385:`, then restored and verified byte-identical
by sha256) reads **17 failed, 0 passed**.  That count is honest but blunt --
`_box_memory` does not exist on the pre-change module, so the fixture cannot
even be built there.  **The BEHAVIOURAL fail-before is the A/B/C table above**,
which drives the real module with `psutil.virtual_memory` patched.

---

## 4. WHAT THE CLAMP CHOOSES NOW

Measured on this box with the re-derived constants (`get_ram_budget()` =
104.67 GB at the probe; the answers below are identical evaluated at the
94.4 GB free the round-2 refusal was recorded against, which was checked
arithmetically rather than assumed).

```
GRID CLAMP (one process)
  ceiling(104.67 GB) = 8192                                   UNCHANGED
  n= 4096 ->  4096      clamp needs   12.9 GB of budget       (was  11.8)
  n= 8192 ->  8192      clamp needs   51.5 GB of budget       (was  47.2)
  n=12288 ->  8192      clamp needs  116.0 GB of budget       (was 106.3)
  n=16384 ->  8192      clamp needs  206.2 GB of budget       (was 189.0)

D8 WORKER CLAMP, fan shape (1024^2, K=32, 8 GB reserve)
  NFC= 8192  per-worker MODEL  27.94 GB  requested 2/3/4/6/8/32 -> 2/3/3/3/3/3
  NFC=16384  per-worker MODEL 105.25 GB  requested 2/3/4/6/8/32 -> 1/1/1/1/1/1
  paraxial   per-worker MODEL   1.37 GB  requested 2/3/4/6/8/32 -> 2/3/4/6/8/32

CAP LADDER (tests/unit/test_niche_p2_guards.py), re-derived rung by rung at 24
  0.25/1/4/16/34/136 GiB -> 512/1024/2048/4096/4096/8192       UNCHANGED
```

**Every decision is unchanged**: k=3 at `NFC=8192`, k=1 at `NFC=16384`, the
same grid ceiling, the same six rungs.  The ladder surviving two consecutive
re-derivations unchanged is exactly how a stale ladder gets shipped, so the
asserted-first constant moves to 24 and the comment says the survival is the
OUTPUT of the re-derivation, not a licence to skip the next one.

### 4.1 The `n_fine = 16384` verdict, at 94.4 GB free

| question | answer |
|---|---|
| does the measured run fit in 94.4 GB free? | **yes, by 6.5 GB** -- 87.925 GB measured.  It does NOT fit with the 8 GB reserve intact (87.9 > 86.4) |
| does the honest model approve it? | **no.**  105.2 GB modelled against 86.4 GB of approvable free |
| so is it refused? | **no** -- item 2's warn path.  `PROCEEDS UNDER WARNING`, at the grid it asked for |
| did the refusal dissolve because the model got cheaper? | **no.  The opposite.**  16384 costs MORE than the branch believed, on both the model (97.5 -> 105.2 GB) and the measurement (84.6 -> 87.9 GB) |
| is the warning path a rare-box fallback? | **no.**  It is the normal path for the grid of record on this box |
| `k=2` at 16384? | still **REFUSED, exit 2**, now by the TOTAL-physical check |

The run that produced the 87.925 GB figure started with 99.7 GB free and its
sampler recorded `min_avail` **11.4 GiB** at the worst instant.  It fits, and
it fits with less margin than anyone should schedule unattended -- which is
precisely the judgement the warning hands to the operator instead of making
it for them.

---

## 5. GREEN

All at `OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = MKL_NUM_THREADS =
NUMEXPR_NUM_THREADS = 1`, `-q -p no:randomly`.  Nothing was xfailed, skipped or
deselected by this change.

**Windows**, python 3.14.6 / numpy 2.4.4 / scipy 1.17.1:

| suite | result |
|---|---|
| `test_niche_p2_guards` (the cap ladder) + `test_niche_d8_congruence_workers` + `test_verify_perf_fixes_2026_08_10` + **`test_fix_grid_intent_override_2026_08_10` (NEW: 17)** + `test_niche_perf_round2_2026_08_10` + `test_niche_perf_poly_locals` + `test_perf_v4_12_0_fft_infra` + `test_audit_w2_fft_state` + `test_niche_k3_perf` + `test_fix_newton_pool_memory` + `test_niche_newton_pool_both_fits` + `test_memory_guardrail` + `test_niche_d3_guards` | **294 passed**, 5:52 |

**WSL** (the repo's CI proxy), python 3.12.3 / numpy 2.4.6, `~/lumen_venv`:

| suite | result |
|---|---|
| `test_niche_p2_guards` + `test_niche_d8_congruence_workers` + `test_fix_grid_intent_override_2026_08_10` + `test_verify_perf_fixes_2026_08_10` + `test_niche_d3_guards` | **133 passed**, 4:37 |

`ruff check` on every changed library and test file: **All checks passed**.
Every changed file decodes as cp1252 and **0 of the 455 added lines** contains
a non-ASCII byte.

`validation/repro_traced_carrier_121/fan_multi_121.py` and `focus_scan_121.py`
carry **10 pre-existing ruff findings** (`E401`/`I001`/`E701`/`E702`, all on
lines this change does not touch); the identical 10 are reported by
`ruff check` on `git show 0b4a385:` copies of both, so they are inherited, not
caused.  This change edits only comment blocks in those two files.

### 5.1 Fail-before, as a set

| # | claim | fail-before | pass-after |
|---|---|---|---|
| 1 | the cap ladder is arithmetic on the constant | moving 22 -> 24 fails **2 tests** in `test_niche_p2_guards` with the formula in the message (`assert 24 == 22`, "Re-derive it: cap = 2**floor(...)") | ladder re-derived, rungs unchanged, `_LADDER_WORK_ARRAYS = 24` |
| 2 | the shipped model is not an upper bound on the re-measured set | the d8 upper-bound assertion, run with the RE-MEASURED rows against the shipped constants restored at runtime: **1 of 7 rows UNDER** (8192 k=3 child, 0.995x) | 0 of 7 under; worst 1.274, tightest 1.045 |
| 3 | the override changes the binding-box outcome | the A/B/C probe: exit 2 / exit 2 / exit 2, **zero warnings** | exit 0 + 1 RuntimeWarning / exit 2 / exit 2 |
| 4 | the override module bites | 17 of 17 fail on the pre-change module (structural -- see sec 3.3) | 17 pass |
| 5 | warn-and-proceed does not reopen the silent-shrink hole | `test_a_real_clamp_bind_still_trips_the_post_run_check` drives the LIBRARY's own clamp after a warn-and-proceed pre-flight and requires exit 3 | it exits 3; and `test_proceeding_never_returns_a_smaller_grid` pins that the return value is a budget, never a grid |

### 5.2 A concurrent session was editing this checkout

While this work was in the working tree, another agent on the same checkout
modified `lumenairy/elements/_lens_traced.py` (one docstring reference) and
`tests/unit/test_niche_audit_e_prepared_and_enums.py`, and added
`docs/audits/FIX_H2_DLASCL_2026_08_10.md`.  **None of them is this change**,
none was edited here, and they are recorded so the diff is not read as one
piece of work.  The green counts above were taken with those edits present.

---

## 6. WHAT THIS DOES NOT DO

1. **The re-measured set is smaller than the one it replaces** -- seven points
   where v5.33.3 had eleven.  The dropped rows were re-runs and k-variants that
   all sat INSIDE the ones kept.  Covered: both ends (4096, 16384), both
   readings at the binding grid (whole process, and worker child at two
   different k), and a reproducibility pair at 4096 agreeing to 0.44 %.
2. **`_PARAXIAL_BASE_BYTES` was NOT re-measured.**  Round 2 touched
   `_fourier_upsample_crop`, `_shift_envelope`, the coords build and `_poly`;
   a `final_leg='paraxial'` worker's floor is not dominated by any of them.  It
   stays at its measured 1.0 GB with its own, narrower envelope note (no
   `N_out` term; an UNTILED readout measured 11.2 GB).  A paraxial
   re-measurement is a separate round, and the fact that the exact leg moved
   3 % in three days is the argument for doing it.
3. **`_FINE_GRID_RAM_FRAC` is still 0.5 and still not independently derived.**
   It is measured to be SUFFICIENT: half a budget must cover a 1.8 GB floor,
   i.e. any budget above ~3.6 GB (it was ~5.2 GB at the old floor).
4. **Peak RSS is sampled at 1 Hz over the tree.**  A sub-second transient is
   invisible to it; every plateau reported here lasts tens of seconds.  Same
   instrument and same limitation as the measurement it replaces, which is what
   makes the two comparable.
5. **One order shape, one input grid, two or six orders.**  1024^2 in, design
   121's post-DOE chain, exact final leg.  The floor is a design-121-CLASS
   figure and says so at the constant.  A 32-order-per-worker run may still
   exceed the envelope.
6. **Wall times are NOT claimed.**  The arms report 45.5 / 138.3 / 589.0 s per
   order at 4096 / 8192 / 16384 against 45.2 / 126.3 / 544.4 s previously, but
   these are one rep each, and unit suites were deliberately run on the same
   box during two of the arms.  Memory peaks are insensitive to that
   contention; walls are not.  Nothing here should be read as a speed
   regression or a speed win.
7. **The override is a policy change on a VALIDATION harness, not on the
   library.**  `carrier._multi_resolve_workers` still clamps k downward and
   never over-commits; nothing here changes that.  What changed is that a
   runner stating an explicit intent is no longer refused by the harness's box
   model.
8. **`format_bytes` labels binary units as decimal.**  Observed while reading
   the clamp's own warning: `format_bytes(1.8e9)` prints `1.68 GB`, because the
   helper divides by 1024 and labels 'GB' (its own doctest pins that).  It is
   the same units confusion that produced the retracted 1.540x claim; it is
   PRE-EXISTING and public, and it is recorded rather than changed under a
   memory-model round.
9. **The new test module carries one guard that never fires**: it skips at
   module level if `validation/` is absent (the repo's existing pattern, from
   `test_verify_perf_fixes_2026_08_10.py`).  On a full checkout it does not
   trigger -- both green runs above report 0 skipped.

---

## APPENDIX -- artefacts

Session scratchpad (outside the repo, so the tree stays clean):
`%LOCALAPPDATA%\Temp\claude\C--Users-Tesla\372a2d1f-...\scratchpad\`

| file | what |
|---|---|
| `recal_runs.py` | the seven-arm driver (subprocess per arm, `kladder_121.py` unmodified) |
| `recal/n{4096,8192,16384}*.json`, `*.log` | the arms, their peaks and their transcripts |
| `recal_fit.py` | the envelope search: the measured table, the shipped model against it, and the constrained candidate ranking |
| `fail_before_override.py` | the A/B/C probe (sec 3.3), run before and after the edit |
| `failbefore_suite.ps1` | the new module against `git show 0b4a385:_grid_intent.py`, with a sha256-verified restore |
| `d8_probe.py` | the sec 4 clamp / pre-flight table |
