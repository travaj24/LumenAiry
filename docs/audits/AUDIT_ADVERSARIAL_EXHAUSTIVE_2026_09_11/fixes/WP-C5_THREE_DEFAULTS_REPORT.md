# WP-C5 -- three maintainer decisions taken: the near-focus kernel switch, honest dense-GBD accounting, and zeroed replicas

Branch `feat/c5-three-defaults`, parent `49ddf4bd`.  Built 2026-09-20 for the
5.49.0 default-flip release.

The three decisions are ledger items **1.5 / 4.3** (the near-focus exact-kernel
fallback), **1.8** (`DENSE_MEM_BUDGET_ACCOUNTING`) and **1.7**
(`replica_fill`) of `MAINTAINER_DECISIONS_2026_09.md`.  The brief numbered them
0.1 / 0.2 / 0.3; that document has no section 0, and its items 1.5, 1.8 and 1.7
are the ones whose evidence the brief cites, so they are what is built here.

Every item below moves what an unmodified call returns, and every one of them
leaves the previous behaviour **one keyword or one constant away, byte-identical
under it**, proved archive-to-archive from a `git archive 49ddf4bd` on both
builds.  Nothing was re-pinned by pasting a number: every bar in every test
added or restated here is derived at runtime from a quantity the running build
measures.

## The two builds, and how every run was pinned

| | Windows | WSL |
|---|---|---|
| python | 3.14.6 (MSC v.1944) | 3.12.3 (GCC 13.3.0) |
| numpy | 2.4.4 | 2.4.6 |
| interpreter | `python` | `~/lumvenv/bin/python` |
| tree pin | `PYTHONPATH=C:/tmp/lum_c5` | `PYTHONPATH=/mnt/c/tmp/lum_c5` |
| baseline pin | `PYTHONPATH=C:/tmp/c5_arch/base` | `PYTHONPATH=/mnt/c/tmp/c5_arch/base` |

Every command carried `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1` on the command line, every GBD reading additionally carried
`LUMENAIRY_MEM_BUDGET_MB=2048`, and every probe writes the resolved
`lumenairy.__file__` into its JSON so a mis-pinned run is visible in the output
and not only in a terminal that has scrolled away.  The baseline is a `git
archive 49ddf4bd` extracted to `C:/tmp/c5_arch/base`; it was diffed against the
clean worktree before use and differs only by `__pycache__`.  The comparison is
ARCHIVE-TO-ARCHIVE: two processes, two `PYTHONPATH`s, one comparison module
that imports nothing from the package.

Probes and JSON: `validation/probe_c5_three_defaults/`.

---

## Item 1 -- the near-focus kernel switch, ON

### What moved

One constant.  `carrier._GAP_KERNEL_ACCURACY_TAU` goes from `None` to `1e-4`.
The rule it arms was built and measured in Wave 5 hygiene round 2 and shipped
switched off; nothing about it changed here.  With it armed, a Collins leg that
the `k4` representability gate has already resolved to `'exact'` computes

    departure_relL2 = sqrt(3/2) * k * |z_eff| * theta_env^4 / 8

with `theta_env` the envelope's own ANALYTIC `1/e^2` half-angle (read from its
sampled spectrum as `2 sqrt(<theta^2>)`, exact for a Gaussian), and resolves to
`'fresnel'` when that exceeds `tau`.  An explicit `gap_kernel='exact'` is never
overridden; `None` restores 5.48.x bit for bit.

The rule lives inside `_collins_transport`, which WP-C3 is editing
concurrently.  Nothing in that function was touched: the decision is the
module-level constant, which is what "one line to arm" meant when the switch
was built.

### The ladder

Windows readings; WSL agrees to every digit shown
(`item1_{head,base}_{win,wsl}.json`).

**Wave-5 hygiene-2 fixture** (`w0` 15.915 um, `theta` 20 mrad, `f` 20 mm,
`theta_env` 7.9512e-04 rad), `gap_kernel='auto'`, distances short of the WAIST:

| d to waist | `z_eff` (m) | k4 | departure | resolved |
|---|---|---|---|---|
| 1 um | 12.27 | 2.23e-05 | 4.7161e-06 | exact |
| 3 um | 11.56 | 2.10e-05 | 4.4435e-06 | exact |
| 10 um | 9.612 | 1.75e-05 | 3.6956e-06 | exact |
| 30 um | 6.488 | 1.18e-05 | 2.4944e-06 | exact |
| 100 um | 3.028 | 5.51e-06 | 1.1641e-06 | exact |
| 300 um | 1.190 | 2.16e-06 | 4.5748e-07 | exact |
| 1 mm | 0.3689 | 6.71e-07 | 1.4185e-07 | exact |
| 3 mm | 0.1123 | 2.04e-07 | 4.3188e-08 | exact |
| 5 mm | 0.05972 | 1.09e-07 | 2.2960e-08 | exact |

**The ladder is inert**: every rung still resolves `'exact'`, and all nine
fields are byte-identical to the baseline.  The worst departure, 4.72e-06, is
1.3 decades under `tau`.

**VERIFY-B4 F3 fixture** (`w` 0.3 mm, N 1024, `dx` 4 um, `lambda` 1.064 um,
`R` -40 mm, `theta_env` 1.1289e-03 rad), distances short of the geometric
focus, relative L2 against the analytic Gaussian (whole-function `q` form,
piston included):

| dz | `z_eff` (m) | k4 | `'auto'` resolves | `'auto'` vs oracle | `'fresnel'` | `'exact'` |
|---|---|---|---|---|---|---|
| 1 um | 1600 | 9.11e-03 | **fresnel** | **1.4568e-14** | 1.4568e-14 | 2.3496e-03 |
| 10 um | 160 | 9.11e-04 | **fresnel** | **1.1822e-14** | 1.1822e-14 | 2.3491e-04 |
| 100 um | 15.96 | 9.09e-05 | exact | 2.3438e-05 | 8.8343e-15 | 2.3438e-05 |
| 1 mm | 1.560 | 8.89e-06 | exact | 2.2910e-06 | 1.2074e-14 | 2.2910e-06 |
| 5 mm | 0.280 | 1.59e-06 | exact | 4.1120e-07 | 2.6360e-14 | 4.1120e-07 |

The 1 um and 10 um rungs fall back and the 100 um rung does not, which is what
`tau = 1e-4` was chosen to do.  At both rungs that fall back, `k4` sits two to
three decades under its bar of 1 -- the gap this rule closes is exactly the one
the wrap guard cannot see.

### The band

The departure is linear in `|z_eff|`, so the rule has a closed-form threshold:

    it fires where   |z_eff| > 8 tau / (sqrt(3/2) k theta_env^4)

| fixture | `theta_env` | threshold in `z_eff` | as a distance to `A = 0` | ladder's reach |
|---|---|---|---|---|
| VERIFY-B4 F3 | 1.1289e-03 rad | 68.10 m | **inside 23.48 um** | reaches `z_eff` 1600 m, so it fires |
| hygiene-2 | 7.9512e-04 rad | 260.09 m | **inside 1.5427 um** | walks to the WAIST; its carrier focus is 31.66 um further on, so `z_eff` caps at 12.27 m and it never fires |

Both edges were found by bisection on the running build, not computed from the
formula and quoted: the bisected F3 edge is 23.4823 um and the bisected
hygiene-2 edge (walked to its own `A = 0` plane instead of its waist) is
1.5427 um, against the ledger's independently measured 1.5431e-04 departure at
1 um from that plane -- agreement to four digits.
`test_c5_three_defaults.py::test_the_rule_fires_only_inside_the_band_the_law_predicts`
asserts the observed switch-over against the closed form to 1 %.

### Blast radius

**Archive-to-archive, 48 digested legs per build:**

| build | identical | moved | which |
|---|---|---|---|
| Windows py3.14 | 46 | 2 | `f3/auto/1e-06`, `f3/auto/1e-05` |
| WSL py3.12 | 46 | 2 | the same two |

The 14 opt-out keys (`tau = None` set at runtime) are identical on both builds,
including the two that moved under the default -- so `None` really is 5.48.x,
and the `stats_out` dict does not grow the `kernel_departure` key under the
opt-out either.

**Over the suite** -- a pytest plugin
(`validation/probe_c5_three_defaults/c5_kernel_census.py`) wraps
`_collins_exact_kernel_departure` and reads the decision out of the caller's
frame, so every leg the `k4` gate resolved to `'exact'` across the 27
carrier / traced-chain test files is censused with its `z_eff`, its
`theta_env`, its departure and whether the rule moved it.  Results in
`item1_census_win.json`; the table is in the "Runs" section below.

### Fixtures restated

| id | what it said | what it says now |
|---|---|---|
| `test_wave5_h2_near_focus_table.py::test_the_shipped_default_does_not_evaluate_the_accuracy_rule_at_all` | the shipped tau is `None` and the rule does not run | renamed `test_tau_none_does_not_evaluate_the_accuracy_rule_at_all`; sets the OPT-OUT explicitly and keeps every assertion |
| (new beside it) | -- | `test_the_shipped_default_arms_the_rule_at_tau_1e_4`: reads the running module's tau and immediately exercises F3's 1 um rung, with a two-sided derived bar on the departure |
| `test_verify_hyg2_round2.py::test_the_accuracy_rule_is_never_executed_while_tau_is_none` | asserted the shipped constant, then that the rule's two functions are never called | sets the opt-out explicitly; the claim it makes (never CALLED) is unchanged and is now the load-bearing one |

Neither restatement pastes a number: the first reads `tau` from the module and
derives its bar from the run, the second asserts a call count of zero.

---

## Item 2 -- honest accounting

### What moved

`gbd.DENSE_MEM_BUDGET_ACCOUNTING` goes from `'legacy'` to `'measured'`, and
three things are added beside it:

* `_DENSE_FIXED_CELL_BYTES = 48.0` -- the per-output-cell cost the loop carries
  OUTSIDE the chunk arithmetic, which is why the budget cannot be met below one
  beamlet column;
* `_dense_budget_floor_bytes(Ny, Nx)` -- ONE helper, `Ny*Nx*(48 + 128)` bytes;
* a `RuntimeWarning` from the dense path when `mem_budget_mb` is below that
  floor, naming the floor, the budget, the ratio and both mitigations.

`_dense_cell_bytes()` now validates the mode: an unrecognised value is REFUSED
by name rather than falling through to `'legacy'`.

### The two constants, derived

The loop's live peak is affine in the chunk.  Fitting
`peak/(Ny Nx) = fixed + c*chunk` over a 1/2/4/8/16/32 chunk ladder at a budget
far too large to bind (`item2_base_win.json`, and identically on the branch
tip):

| grid | build | `fixed` (B/cell) | `c` (B/cell-column) | worst deviation from the affine model |
|---|---|---|---|---|
| 512 x 512 | Windows py3.14 | 48.551 | 96.000 | 5.8e-07 |
| 256 x 256 | Windows py3.14 | 50.114 | 96.000 | 7.0e-06 |
| 512 x 512 | WSL py3.12 | 48.047 | 88.000 | 1.2e-06 |
| 256 x 256 | WSL py3.12 | 48.098 | 88.000 | 4.8e-06 |

Windows' two `fixed` readings differ by 1.56 B/cell, which at N = 256 is
102 KB -- a grid-INDEPENDENT offset divided by a smaller cell count, not a
second per-cell term.  On Windows `c = 96.000` is the same constant the WP-B14
ladder read as "72.0 to 96.8, saturating at 96.0 once the chunk binds",
measured here a second way.

**`c` IS BUILD-DEPENDENT** -- 96 on Windows, 88 on WSL, one 8-byte-per-cell
temporary the two numpy versions differ over.  That is precisely why
`_DENSE_CELL_BYTES_MEASURED` is 128 and not either reading: it has to sit above
the LARGER of them with margin, which it does on both.  Nothing in this item is
pinned to one build's number.

The FIRST tracemalloc window in a process also captures whatever the import
graph allocates lazily on first use; it landed on whichever chunk was measured
first and read 838 B/cell at N = 256 against a model 144.5.  One discarded
warm-up run removes it and the fit goes from 80 % scatter to 7e-06.  That is
recorded because it is the kind of thing that reads as a result.

### The floor, and both sides of it

`_dense_budget_floor_bytes` publishes `Ny*Nx*(48 + 128)`:

| grid | published floor | measured one-column peak | published / measured |
|---|---|---|---|
| 256 x 256, Windows | 11.534 MB | 9.576 MB | **1.205x** |
| 512 x 512, Windows | 46.137 MB | 37.893 MB | **1.218x** |
| 256 x 256, WSL | 11.534 MB | 8.919 MB | **1.293x** |
| 512 x 512, WSL | 46.137 MB | 35.664 MB | **1.294x** |

The floor is an UPPER bound on the loop's own one-column peak, which is what
makes "at or above the floor the budget is honoured" true rather than hopeful.
It deliberately does not take the accounting mode: it is what the LOOP costs,
and a `'legacy'` floor would read 4.19 MB on a grid where the loop measurably
cannot go below 9.58.

**Above the floor the budget bounds the loop** -- swept over ten multiples,
1024 beamlets, `tracemalloc`, Windows:

| multiple of the floor | N = 256 ratio | N = 512 ratio |
|---|---|---|
| 1.00x | 0.830 | 0.821 |
| 1.25x | 0.664 | 0.657 |
| **1.50x** | **0.917** | **0.911** |
| 2.00x | 0.688 | 0.683 |
| 2.50x | 0.768 | 0.765 |
| 3.00x | 0.822 | 0.819 |
| 4.00x | 0.753 | 0.751 |
| 6.00x | 0.775 | 0.773 |
| 8.00x | 0.786 | 0.785 |
| 12.0x | 0.751 | 0.750 |

and no floor notice fires anywhere on that sweep.  WSL reads the same shape
lower throughout (worst 0.849 at N = 256 and 0.849 at N = 512, both at 1.5x),
so Windows is the binding build and is what the table above shows.  The worst
cell is 1.5x the floor, where the chunk has just stepped to two; the ratio
settles to `c`/128 as the budget grows (0.75 on Windows, 0.6875 on WSL) --
the per-column margin, with the fixed term amortised away.  The 9 % margin at the worst cell is the honest number and
is stated as such in the test: the budget arithmetic models `chunk * 128`
B/cell while the loop also holds 48 B/cell outside the chunk, so just after a
chunk increment the slack is only what the 128-against-96 margin has left over.
This is why the claim is asserted over a SWEEP and not at one budget.

**Below the floor it is loud, and the notice is accurate** -- at 0.5x the
floor:

| grid | build | budget | peak | ratio | notice |
|---|---|---|---|---|---|
| 256 x 256 | Windows | 5.767 MB | 9.578 MB | 1.661x | fires, names 11.5343 MB and `window=5.0` |
| 512 x 512 | Windows | 23.069 MB | 37.894 MB | 1.643x | fires, names 46.1373 MB |
| 256 x 256 | WSL | 5.767 MB | 8.922 MB | 1.547x | fires |
| 512 x 512 | WSL | 23.069 MB | 35.665 MB | 1.546x | fires |

and under `'legacy'` at the same budgets nothing is emitted (peaks 6.008x /
6.006x on Windows, 5.516x / 5.546x on WSL -- the old under-count, unchanged).

### Warn, not refuse -- and why, measured

The brief allows either, chosen "by measurement of what a caller can do about
it".  The measurement:

* At the shipped `mem_budget_mb=512.0` default the floor binds from
  `Ny*Nx > 512e6/176 = 2.909e+06` cells, i.e. **any square grid past
  N = 1706**.  A refusal would therefore turn a dense reconstruction that
  completes today into a hard error on a DEFAULT path, for a memory request
  the library merely cannot meet -- while the field it would have returned is
  correct.
* The one mitigation that keeps the grid is `window=5.0`, whose own accounting
  has no one-column floor at these sizes.  It is not equivalent: it changes the
  returned field by its own truncation (`exp(-25)`, about 1e-11).  The library
  cannot apply it on the caller's behalf without moving the answer, so the
  caller has to choose.
* The remaining remedies -- raise the budget, shrink the grid -- change the
  request, not the code path.

So the caller has an action for every case and none of them can be taken
automatically.  The honest move is to run, and to say the budget was not met
with the floor and the remedies in the message.  Under `'legacy'` nothing is
emitted at all: that mode's arithmetic never claimed to bound the loop and a
caller who selects it has opted out of the claim.

### What the flip costs, and byte identity

| grid | `'legacy'` vs `'measured'`, max relative difference |
|---|---|
| 256 x 256 | 2.117e-17 |
| 512 x 512 | 1.824e-18 |

-- identical to all printed digits on BOTH builds, which is what says the
difference is the summation order and not anything the allocator or the BLAS
is doing.

-- i.e. the last bits, and the two are NOT identical (the chunk did move).  The
same budget run twice is bit-identical under both modes.

### Blast radius

Two digest sets, because they answer two different questions.

**Explicit-mode keys** (24 per build: `'legacy'` and `'measured'` at N = 64 /
128 / 256 x 512 / 64 / 8 MB, plus the windowed sibling and the repeat-run
keys): **24 identical, 0 moved**, on both builds.  That is the statement
"`'legacy'` is byte-identical to `49ddf4bd`" -- and, incidentally, that
`'measured'` is unchanged too: the item moved the DEFAULT, not the mode.

**Default-path keys** (22 per build, no switch touched at all, from
`c5_item2_default.py`): **8 identical, 14 moved**, both builds agreeing on the
set.  The eight that do not move are the windowed path, `frame_completeness`
(which is windowed) and the three budgets at which both accountings already
floor the chunk at one column (`N192/1MB`, `N256/1MB` and `N64/512MB`) -- where
the constant cannot change anything because there is nothing left to divide.

The floor notice fires on exactly four of those 22 default calls
(`N128/1MB`, `N192/1MB`, `N256/1MB`, `N256/8MB`) and on none of the others; on
the parent archive it fires on none, because the helper does not exist there.

### Consumers of `mem_budget_mb`

Grepped across `lumenairy/`, `tests/`, `examples/`, `benchmarks/` and
`scripts/`: `reconstruct_field_from_beamlets` and `frame_completeness` in
`gbd.py`, and `elements/lenses_gbd.py` (four call sites, `mem_budget_mb`
default 512.0), which `lens_config.py` exposes.  `carrier.py`'s
`mem_budget_mb` is the multi-chain orchestrator's own budget and never reaches
this loop; `fga.py`'s is the momentum swarm's, likewise.  No example or
benchmark calls either GBD entry point.

The element family's blast radius is narrower than the grep suggests: all four
of its call sites pass `window=window`, whose default is `5.0`, so they take
the WINDOWED path -- whose accounting was always correct and is untouched --
unless a caller explicitly passes `window=None` or the bundle is non-uniform
on JAX / CuPy.  Every one of them still works; the tests that exercise them
are in the Runs section.

---

## Item 3 -- zeroed replicas, and the proof that the zeroing is confined

### What moved

`replica_fill` goes from `'repeat'` to `'zero'` on
`carrier_referenced_focus_readout`,
`carrier_referenced_exact_focus_readout` and `_collins_focus_readout` (the
private readout `propagate_traced_carrier_chain` uses on
`transport='collins'`).  The chain and multi entry points forward the keyword
unchanged, so they follow.  `_fill_readout_replicas` now publishes
`out['replica_fill']` beside `out['faithful_samples']`, and
`_publish_readout_containment` copies it to each stage as
`readout_replica_fill`.

### (a) A faithful window is untouched

The region the fill governs is the exact complement of the replica guard's own
condition, so it is EMPTY whenever `2|centre_out| + N_out dx_out <= period`.
Measured: the two fills return the SAME OBJECT, not merely equal arrays, and
the default call's digest equals the explicit `'repeat'` call's digest on both
builds (`faithful/default`, `faithful/repeat`, `faithful/zero`,
`exact_faithful/*` -- all identical to the baseline).

### (b) On an oversized window, only the replicas move

256-sample windows at three multiples of the period, both builds identical:

| window / period | faithful samples | `2*floor(p/2/dx_out)+1` | inside byte-identical to `'repeat'` | max abs outside, `'zero'` | max abs outside, `'repeat'` |
|---|---|---|---|---|---|
| 1.10 | (233, 233) | 233 | yes | **0.0** | 0.0464 |
| 1.60 | (161, 161) | 161 | yes | **0.0** | 0.0851 |
| 2.20 | (117, 117) | 117 | yes | **0.0** | 36.18 |

The boundary is the period the transform reports through `_period_out`, and the
faithful count is that reported period's own arithmetic -- not a stored number
and not a heuristic.  At 2.20 periods the `'repeat'` maximum outside the
faithful zone is 36.18 against a faithful-zone peak of the same order: that is
the core's own replica inside the window, the regime in which a peak or a width
found by an argmax is no longer safe.

The exact readout, on its own period (the fine crop window): 4096 samples at
1.71 periods, faithful (2401, 2401), inside byte-identical, outside exactly 0,
and the default equals the explicit `'zero'`.

**Off axis** -- the case a heuristic gets wrong.  With `centre_out` 0.30
periods off axis the faithful count is still (161, 161) and the surviving band
is NOT centred on the window: `E(u + period) == E(u)` holds in ABSOLUTE output
coordinates, so pushing the window off axis SPENDS the period rather than
carrying it.  A fill keyed on the window's own centre would pass every on-axis
reading in this report and fail here;
`test_the_faithful_zone_is_centred_on_the_field_not_on_the_window` asserts the
band's centre index explicitly.

### (c) What the result says

`_period_out['faithful_samples']` (existing) is the `(nx, ny)` samples per axis
that carry measurement; `_period_out['replica_fill']` (new) names the fill that
was applied.  Both are published on BOTH settings and on faithful as well as
oversized windows, so neither has to be inferred from the other's absence, and
the traced chain publishes both per stage (`readout_faithful_samples`,
`readout_replica_fill`).

### (d) The refusal is unchanged

The guard is evaluated before any fill runs, in all three readouts.  Measured
as two censuses over the same ladder rather than asserted about the code:

| window / period | `'repeat'` | `'zero'` |
|---|---|---|
| 0.50 | served | served |
| 0.98 | served | served |
| 1.00 | served | served |
| 1.02 | refused (ALIASES) | refused (ALIASES) |
| 1.60 | refused | refused |
| 2.20 | refused | refused |

Identical cell for cell, on both builds: **no fixture that was refused is now
served, and none that was served is now refused.**  `test_the_refusal_is_taken_before_the_fill_is_reached`
makes the ordering a demonstration rather than a reading: with
`_fill_readout_replicas` replaced by a raising sentinel, an oversized window at
`on_replica='error'` still comes back as the refusal, and the counter-pin (the
same window with the guard waived DOES reach the sentinel) is what stops that
arm from being vacuous.

### Blast radius

Archive-to-archive, 20 digested readouts per build:

| build | identical | moved |
|---|---|---|
| Windows py3.14 | 16 | 4 |
| WSL py3.12 | 16 | 4 |

The four are `oversized/{1.10,1.60,2.20}/default` and `exact/default` -- every
one a default-fill call on a window that reaches past one period.  Every
explicit `'repeat'` key, every explicit `'zero'` key, every faithful window
(default included) and both off-axis keys are byte-identical.

### (e) The three demonstrations, rewritten as demonstrations of the opt-in

| id | how it reads now |
|---|---|
| `test_fix_v1_v8_readout_guard_and_standoff.py::TestV3ChainScope::test_a_walking_chief_ray_gives_a_full_amplitude_ghost` | asks for `replica_fill='repeat'` and keeps its original claim verbatim (the ghost's peak is bit-identical to the real spot's); GAINS two arms -- under the shipped default the ghost window is exactly 0, and the ON-chief-ray window is bit-identical to the `'repeat'` one, so the fill costs the measurement nothing.  Docstring says why the default moved. |
| `test_niche_tight_focus_readout.py::test_the_refused_window_really_would_have_been_corrupt` | asks for `'repeat'` (the corrupt wing IS the replicas) and keeps the `> 5x` asymmetry; GAINS the arm that the same waived window under the default reads the beam again (`< 2x` the faithful wing metric).  The two arms sit on opposite sides of the same number. |
| `test_niche_d2_chain_multi.py::test_k1_keeps_the_requested_field_of_view` | asks for `'repeat'` on the K = 1 equivalence and keeps "the whole requested window is live"; GAINS the arm that under the default the same call keeps the requested grid size and tile, differs from `'repeat'` only outside one period, and is therefore NOT the 2026-08-06 silent shrink-and-zero the docstring records -- which is still refused. |

`TestV3ReplicaGuardSeesCentreOut::test_the_verifiers_own_ghost_case_is_refused`
took the same treatment for the same reason.  None of the four is weakened:
each keeps its original assertion under the opt-in and gains a two-sided arm.

### (f) The mutation matrix

| mutation | caught by |
|---|---|
| one sample INSIDE the period zeroed (an off-by-one in the comparison) | `test_zeroing_one_sample_inside_the_period_would_be_caught` -- blanks the last faithful column of a real answer and asserts the confinement helper raises, with a premise gate that the column was not already zero |
| the fill keyed on the WRONG period (half, or one and a half) | `test_keying_the_fill_on_the_wrong_period_would_be_caught` -- builds both mutants through the library's own `_fill_readout_replicas`, asserts both are rejected, and asserts the RIGHT period passes the same helper |
| the fill applied on the REFUSAL path | `test_the_refusal_is_taken_before_the_fill_is_reached` -- raising sentinel, plus the counter-pin above |

All three aim at one helper, `_assert_confined`, which is also what the
non-mutated ids assert, so a mutation that slipped past the helper would slip
past the real ids too and the matrix would be measuring nothing.

### Public entry points whose returned array can change

`carrier_referenced_focus_readout`, `carrier_referenced_exact_focus_readout`,
`propagate_traced_carrier_chain` (via `focus_readout=`) and
`propagate_traced_carrier_chain_multi` (via `output_grid=` and its own
`on_replica=`).  Grepped: nothing else in the package calls either readout.
All four are named in the Migration paragraph, with the condition that is
necessary as well as sufficient -- the window must reach outside one period AND
the replica refusal must be waived.

---

## Files touched

| file | what |
|---|---|
| `lumenairy/propagators/carrier.py` | `_GAP_KERNEL_ACCURACY_TAU = 1e-4` and its note (item 1); `replica_fill='zero'` on the three readouts, the fill published through `_fill_readout_replicas` / `_publish_readout_containment`, and the prose that called `'repeat'` the default (item 3) |
| `lumenairy/propagators/gbd.py` | `DENSE_MEM_BUDGET_ACCOUNTING = 'measured'`, `_DENSE_FIXED_CELL_BYTES`, `_DENSE_MEM_BUDGET_ACCOUNTINGS`, `_dense_cell_bytes`, `_dense_budget_floor_bytes`, the floor notice, and the notes for all of it |
| `docs/history/carrier.md`, `docs/history/lumenairy.propagators.gbd.md` | fingerprints re-recorded, one `re_recorded:` line per item, in the item's own commit |
| `tests/unit/test_c5_three_defaults.py` | new, 25 ids |
| `tests/unit/test_wave5_h2_near_focus_table.py` | the opt-out id restated + a new shipped-default id (item 1) |
| `tests/unit/test_verify_hyg2_round2.py` | D-6 sets the opt-out explicitly (item 1) |
| `tests/unit/test_wave5_gbd_dense_mem_budget.py` | the two default-reading ids restated (item 2) |
| `tests/unit/test_fix_v1_v8_readout_guard_and_standoff.py`, `tests/unit/test_niche_tight_focus_readout.py`, `tests/unit/test_niche_d2_chain_multi.py`, `tests/unit/test_audit2609_a25_carrier_focus_readout.py`, `tests/unit/test_audit2609_b4_collins_transport.py` | the demonstrations named the opt-in, with their reasons (item 3) |
| `CHANGELOG.md` | three `### Changed` entries in `## [Unreleased]`, each with a **Migration** paragraph; plus the `[5.47.0]` block's source-line citations re-anchored (mechanical -- `carrier.py` gained lines) |
| `Migration-Guide.md` | a `## 5.49.0 -- the default flips (2026-09-20)` section with one recipe-carrying subsection per item |
| `validation/probe_c5_three_defaults/` | the probes, the comparison module, the census plugin and every JSON from both builds |

## Runs

### Windows py3.14.6 / numpy 2.4.4

Every command carried `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1 LUMENAIRY_MEM_BUDGET_MB=2048`, `PYTHONPATH=C:/tmp/lum_c5`,
`--capture=sys` and `-p no:randomly`.

| what | result |
|---|---|
| the 27 carrier / traced-chain files + `test_c5_three_defaults.py` (with the census plugin) | **768 passed, 2 skipped** in 2887 s |
| `test_c5_three_defaults.py` alone | **25 passed** in 101 s |
| `test_verify_b14_known_reds.py` + `test_gbd*.py` + `test_verify_wave5_e.py` | **36 passed** in 303 s |
| `test_wave5_gbd_dense_mem_budget.py` | **5 passed** in 31 s |
| `test_niche_d2_chain_multi.py` | **38 passed** in 693 s |
| `test_niche_p2_design_battery.py` + `test_niche_c1_consolidation.py` + `test_niche_d8_congruence_workers.py` | **100 passed** in 936 s |
| the census / walker / dispatcher-pin / public-API / doc-consistency sweep, 28 files incl. `test_audit_except_budget.py` and `test_niche_audit_w4_input_kind.py` | 866 passed, 14 skipped, **3 failed** on the first pass -> two repaired here, one pre-existing |
| `test_public_api.py` + `test_v5_3_2_walker_source_line_citation.py` after the repairs | **19 passed, 1 skipped, 1 failed** (the pre-existing one) |

The two repaired sweep failures were both this branch's:

* `test_no_shipped_source_claims_a_version_the_package_has_not_reached` -- the
  first draft of the notes said "since 5.49.0" while `__version__` is 5.48.1.
  The gate is right (release numbering moves), so the shipped source now names
  no version at all and the CHANGELOG carries it.
* `test_v18_5_the_5_47_0_block_citations_name_the_right_lines` -- `carrier.py`
  gained lines, so ten `[5.47.0]` CHANGELOG citations pointed at the wrong
  ones.  Re-anchored with the sanctioned tool
  (`scripts/reanchor_citations.py --base f4f18851 --block "[5.47.0]"`, content
  based and idempotent).

The third, `test_installed_metadata_version_matches_source_version`, is
environmental and pre-existing: see "What could not be measured".

No test in any of these runs was skipped for a resource reason; the two skips
in the big run are `test_niche_k2_carrier_backends.py`'s CuPy ids on a box with
no functional CUDA device.

### WSL py3.12.3 / numpy 2.4.6

Same pins, `PYTHONPATH=/mnt/c/tmp/lum_c5`, `~/lumvenv/bin/python`:

| what | result |
|---|---|
| `test_c5_three_defaults.py` + `test_wave5_h2_near_focus_table.py` + `test_verify_hyg2_round2.py` + `test_wave5_gbd_dense_mem_budget.py` + `test_verify_b14_known_reds.py` + `test_audit2609_a25_carrier_focus_readout.py` + `test_niche_tight_focus_readout.py` | **93 passed** in 489 s |

`test_audit2609_b4_collins_transport.py` is deliberately absent, and the first
attempt at this run is why: it hung for over an hour on an idle
`multiprocessing` spawn pool with six live worker children.  That is the
PRE-EXISTING WSL stall VERIFY-WAVE5-HYGIENE2 round 2 section 11 records -- "it
stalls at id #57 of the file, reproducible 3 of 3, and identical on the
`112c3049` archive ... the b4 file cannot be run whole on a WSL CI lane".  The
file's 162 ids pass on Windows in this branch, and the stalled run was killed
and its workers swept.  Nothing here changes it and nothing here could have
measured around it.

### Walkers and gates

```
python scripts/check_doc_identifiers.py            624 -> 630 API-claiming tokens, 0 unresolved
python scripts/record_history_fingerprints.py --check    no drift
python scripts/check_source_line_citations.py      declines: the v5.48.1 block carries no source:line citation
python -m ruff check lumenairy/ tests/   (WSL)     All checks passed!
python -m mypy                                      Success: no issues found in 33 source files
```

`docs/history/carrier.md` was re-recorded twice -- once in the item-1 commit
and once in the item-3 commit, each with its own `re_recorded:` reason --
and `docs/history/lumenairy.propagators.gbd.md` once, in the item-2 commit.
The later gbd edits are comments and docstrings only and move neither
fingerprint, which is why no third re-record appears.

### The item-1 census, over the suite

The plugin ran inside the 768-test carrier / chain run.  **167 legs** reached
the accuracy rule (i.e. the `k4` gate had already resolved them to `'exact'`)
across **48 test ids**; **24** of them fell back.  Twenty-three are in the
three ids that exist to exercise the rule -- twenty of those are the bisection
inside
`test_c5_three_defaults.py::test_the_rule_fires_only_inside_the_band_the_law_predicts`,
which walks the threshold on purpose and whose last ten rungs read a departure
of 1.0000e-04 against a tau of 1e-04.

**One leg outside the tau fixtures changes kernel**, and it is worth naming:

| where | `z_eff` | `theta_env` | departure | threshold `8 tau / (sqrt(3/2) k theta^4)` |
|---|---|---|---|---|
| the `mismatch_matrix` fixture of `test_audit2609_b4_collins_transport.py` (WP-A6's own fixture, lambda 1.31 um, NA 0.05) | -0.180 m | 5.5712e-03 rad | 1.2733e-04 | 0.1414 m |

-- inside the band by 1.27x, which is exactly `departure / tau`.  It is a
LARGE-ANGLE leg rather than a near-focus one (`theta_env` is 4.9x the F3
fixture's, and the departure is quartic in it), which is the useful reminder
that the band is a band in `k |z_eff| theta^4` and not in distance alone.  The
file is green: the gate that scores that fixture is
`test_collins_reads_unity_at_every_mismatch`, whose bar is `|peak - 1| < 1e-4`
against a column reading 1.000000 to 0.999938.

One recorded leg carries `kernel_asked = None`: that is
`test_the_departure_law_predicts_what_the_refinement_actually_changes` calling
`_collins_exact_kernel_departure` DIRECTLY to compare the law against the
measurement, so there is no transport frame to read and it is not a leg at
all.  Counted out, the leg total is 24.

### A note on the census instrument, because it was wrong once

The first version of `c5_kernel_census.py` wrapped BOTH
`_collins_exact_kernel_departure` and `_collins_exact_kernel_correction` and
inferred the decision from the call ORDER (a departure with no correction after
it = a fallback).  Two things went wrong and both are worth recording:

* it mis-attributed two legs in `test_the_departure_law_predicts_what_the_refinement_actually_changes`,
  where an explicit `gap_kernel='exact'` is honoured over `tau` -- the pairing
  had no way to see the asked-for kernel;
* wrapping `_collins_exact_kernel_correction` broke
  `test_wave5_h2_collins_jax.py::test_the_threaded_helpers_default_to_the_numpy_build[_collins_exact_kernel_correction]`,
  which inspects that helper's SIGNATURE.  A census that changes the thing it
  is counting is not a census.

The shipped version wraps one function, with `functools.wraps`, and reads
`_kernel_asked` and the resolved `kernel` out of the caller's frame, so the
decision is recorded rather than inferred.  The numbers below are from that
version.

## What could not be measured

* **Which kernel is the more physical (item 1).**  The oracle behind the
  departure law is the paraxial whole-function `q` form.  It can say how far
  the exact-kernel refinement departs from the paraxial truth and it cannot say
  which of the two is closer to Maxwell on a given leg.  On a leg where the
  exact kernel IS the better physics this rule trades accuracy for agreement
  with that oracle.  That is why an explicit `gap_kernel='exact'` is never
  overridden and why `None` stays one assignment away; it is stated in the
  constant's own note, in the CHANGELOG and in the Migration Guide, and it is
  not something this work package can settle.

* **What `tau` should be for a caller who is not this campaign.**  `1e-4` is
  the maintainer's number, chosen because it leaves the hygiene-2 ladder inert
  and catches VERIFY-B4 F3's two near rungs.  It is a relative-L2 budget on the
  field; a caller with a different budget has to set a different `tau`, and
  nothing here measures what the right one is for an arbitrary design.

* **RSS, as opposed to Python-level allocation (item 2).**  Every memory
  reading here is `tracemalloc`, which counts allocations the CPython allocator
  made and is reproducible across arms in a way RSS is not.  It does not see
  arena fragmentation, the BLAS's own workspaces, or anything a native library
  allocates outside Python.  So "the budget bounds the loop" is a statement
  about the loop's Python-level transient, which is the quantity the chunk
  arithmetic is about; it is not a promise about the process's resident set.

* **Whether the dense path caused the access violation.**  Handoff section 5's
  `Windows fatal exception: access violation` is still unreproduced.  What is
  established -- and was already established by WP-B14 -- is that the transient
  was up to six times the size the caller asked for.  This item removes that
  overrun; it does not demonstrate that the fault goes with it.

* **A CuPy device (items 1 and 3).**  `test_niche_k2_carrier_backends.py`
  skipped two ids on this box with "CuPy present but no functional CUDA
  device", so neither the kernel rule nor the fill was exercised on a GPU
  array.  Both are structurally host-side -- the kernel rule reads a NumPy
  spectrum and `_collins_transport` refuses the measured decisions under a JAX
  trace by name, and the fill builds its mask on the field's own backend and
  multiplies -- but "structurally" is not "measured", and this box could not
  measure it.

* **The merge with WP-C3.**  That work package is making `transport='collins'`
  the default and adding a CuPy arm to `_collins_transport` on its own branch.
  Nothing here touches that function, its helpers or the `transport=` plumbing,
  so the two diffs should not overlap -- but the two branches were never merged
  in this session and the combined blast radius was not measured.  It is worth
  noting that the flip WP-C3 is making ENLARGES this item's own blast radius:
  every leg that moves from `'sziklas'` to `'collins'` becomes a leg the
  near-focus rule can see, because the rule lives in the Collins transport.

* **The release number.**  `Migration-Guide.md` files these under a
  `## 5.49.0` heading and the CHANGELOG entries sit in `## [Unreleased]`.  If
  the release lands under another number that heading needs restamping; the
  shipped source deliberately names no version at all
  (`test_public_api.py::test_no_shipped_source_claims_a_version_the_package_has_not_reached`
  caught the first draft doing so, and it was right to).

* **The b4 file, whole, on WSL.**  It stalls there on an idle
  `multiprocessing` spawn pool, which VERIFY-WAVE5-HYGIENE2 round 2 section 11
  measured as pre-existing and identical on a `112c3049` archive.  The file's
  ids were therefore run whole on Windows only (162 passed).  The one leg in
  its `mismatch_matrix` fixture that this item moves is a Windows reading; the
  other build could not be asked.

* **One pre-existing red, environmental.**
  `test_public_api.py::test_installed_metadata_version_matches_source_version`
  fails on this box because the editable install's metadata reads 5.47.0
  against a source `__version__` of 5.48.1.  It reproduces identically on a
  pristine `git archive 49ddf4bd` tree, and WP-B14 section 7.5 records the same
  failure for the same reason.  Nothing in this branch touches it.
