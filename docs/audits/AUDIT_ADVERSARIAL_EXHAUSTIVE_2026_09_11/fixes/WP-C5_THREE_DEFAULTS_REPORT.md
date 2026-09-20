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
is doing.  The figure is GRID-SPECIFIC and has to be quoted with its grid
(VERIFY-WP-C5 D8): on the verification's own grids the same quantity reads
1.8e-16 at N = 320 and 7.3e-17 at N = 512, identical on both builds -- the
same last-bit character an order of magnitude above the two rows above.  The
claim this table supports is "the last bits", not "2e-17"; the branch's test
bar is 1e-12 and is unaffected either way.

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

The first two aim at one helper, `_assert_confined`, which is also what the
non-mutated ids assert, so a mutation that slipped past the helper would slip
past the real ids too and the matrix would be measuring nothing.  The third
aims at a CALL SITE rather than at the helper, and there are three of them
(carrier.py 2772 / 4493 / 7235); this id exercises the paraxial readout's
(4493).  VERIFY-WP-C5 D9 measured the other two per call site and found the
C5 file set caught the Collins one only through the verification's own id and
the exact one not at all, so round 2 gives each site a named id in this file:
`test_the_refusal_precedes_the_fill_on_the_collins_readout` (2772) and
`test_the_refusal_precedes_the_fill_on_the_exact_readout` (7235), each with
the same raising sentinel and the same counter-pin.  The LIBRARY was covered
on all three throughout -- the Collins site by
`test_audit2609_b4_collins_transport.py::TestKellyGuard::test_the_period_is_the_input_grid_s_and_the_replica_guard_sees_it`
and the exact site by
`test_fix_v1_v8_readout_guard_and_standoff.py::TestV3ExactReadout::test_one_period_off_the_chief_ray_is_refused`
and
`test_niche_tight_focus_readout.py::test_the_exact_readout_guards_the_same_way_on_its_own_period`
(VERIFY-WP-C5, measured per call site) -- so this was a reach statement about
this file's own matrix, not a hole in the guards.  It mattered because
`_collins_focus_readout` is the readout WP-C3 is about to make the default
route.

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
four ids that exist to exercise the rule -- nineteen of those are the
bisection inside
`test_c5_three_defaults.py::test_the_rule_fires_only_inside_the_band_the_law_predicts`,
which walks the threshold on purpose and whose last ten rungs read a departure
of 1.0000e-04 against a tau of 1e-04.  (Corrected in round 2, VERIFY-WP-C5 D7:
the prose read "three ids ... twenty of those" against this branch's own
`item1_census_win.json`, which records four and nineteen -- `nodes_with_fallback`
holds six entries, of which `test_wave5_h2_near_focus_table.py::test_the_departure_law_predicts_what_the_refinement_actually_changes`
carries `kernel_asked = None` and is the direct call counted out below, and
the sixth is the `mismatch_matrix` leg named next.  The independent census
reproduces both figures.)

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

---

# Round 2 (VERIFY-WP-C5) -- 2026-09-20

The independent verification `fixes/VERIFY_WP-C5.md` recommends SHIPPING all
three items and raises nine non-blocking defects, D1-D9.  All nine are closed
here, on the branch `feat/c5-three-defaults-round2` (from
`verify/c5-three-defaults` at `df6c73f3`).  Nothing in this round moves a
returned byte: every source edit is a comment, a docstring or a name in prose,
and that is measured rather than asserted (see **Neutrality**).

Both builds, pinned on every command line with
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`:

| | Windows | WSL |
|---|---|---|
| python | 3.14.6 (MSC v.1944) | 3.12.3 (GCC 13.3.0) |
| numpy / scipy | 2.4.4 / 1.17.1 | 2.4.6 / 1.17.1 |
| tree pin | `PYTHONPATH=C:/tmp/lum_c5b` | `PYTHONPATH=/mnt/c/tmp/lum_c5b` |
| PRE pin | `PYTHONPATH=C:/tmp/c5b_arch` -- this round's own `git archive df6c73f3` | the same path under `/mnt/c` |

Probes added: `validation/probe_c5_round2/`.  Every probe writes and prints
its resolved `lumenairy.__file__`.

## What closed, and with what

| defect | what it was | closed by | measured, Windows | measured, WSL |
|---|---|---|---|---|
| **D1** | `_collins_transport`'s comment said the rule was OFF BY DEFAULT while the shipped tau is `1e-4` | comment rewritten; `test_the_shipped_source_never_says_the_rule_is_off_while_it_is_armed` greps the running module | id passes; against the PRE tree it FAILS, naming this site | identical |
| **D2** | the same claim in `_collins_exact_kernel_departure`'s docstring | docstring rewritten; the same id collects both sites into one message | id passes; the PRE failure names D1 and D2 together | identical |
| **D3** | one docstring paragraph at column 0 inside an indented docstring | indented; the same id's second arm reads `inspect.cleandoc` | **0 of 66** non-blank lines indented after cleandoc, against **53 of 55** on the PRE tree | identical (0 of 66 / 53 of 55) |
| **D4** | "any square grid past N = 1706" in four places | all four corrected, each now carrying both readings | floor 511.636400 MB at N = 1705, 512.236736 MB at N = 1706 | identical to every digit |
| **D5** | the fill justified by `E(u+p) == E(u)`, which the Collins readout does not obey | docstring restated (and the CHANGELOG / Migration sentences that repeated it); `test_the_collins_replicas_are_aliases_up_to_a_known_phase` | complex residual 6.957743e-03 of peak, modulus 2.012754e-14, closed form 7.169836e-09 over **377** pairs; the paraxial readout reads 1.131721e-13 complex | 6.957743e-03 / 2.013022e-14 / 7.173703e-09 over 377 pairs |
| **D6** | Migration and CHANGELOG stated the scope as a distance to a focus | both rewritten to the criterion, the rule renamed in source, `test_a_wide_envelope_leg_far_from_any_focus_falls_back` added | see the next section | identical to every printed digit |
| **D7** | the report's prose said "three ids ... twenty of those" | corrected to four and nineteen against this branch's own `item1_census_win.json` | 167 legs / 48 ids / 24 fallbacks, 19 of them the bisection | n/a (a document) |
| **D8** | "2e-17 relative" quoted without its grid | restated in the report and the CHANGELOG with both sets of grids | 2.1e-17 (N = 256) / 1.8e-18 (N = 512) here, 1.8e-16 (N = 320) / 7.3e-17 (N = 512) there | identical |
| **D9** | the refusal-before-the-fill demonstration covered one of three call sites | a named id per uncovered site; the report's "(f)" paragraph corrected | the mutation matrix below reads **12 of 12** caught, against 11 of 12 | identical, cell for cell and set for set |

## D6, re-measured here

The verification's two examples, rebuilt from their own definitions on this
tree (`validation/probe_c5_round2/r2_d6_band_{win,wsl}.json`) and **identical
on both builds to every printed digit**:

| fixture | distance to its own `A = 0` plane | `theta_env` (rad) | departure | resolves |
|---|---|---|---|---|
| relay, `fr = 0.85`, readout AT the beam's focus | **3.00 mm** | 7.085714e-03 | **1.772948e-04** | `fresnel` |
| relay, `fr = 0.90` | 2.00 mm | 4.487030e-03 | 4.528060e-05 | `exact` |
| relay, `fr = 0.95` | 1.00 mm | 2.193737e-03 | 5.461675e-06 | `exact` |
| the same relay read out a QUARTER of the way to the focus, `fr = 0.70` | **9.00 mm**; 48.6 Rayleigh ranges from the beam's focus, beam 0.600 mm wide | 1.715395e-02 | **4.179416e-04** | `fresnel` |
| the same, `fr = 0.80` -- FURTHER from `A = 0` | 11.0 mm | 1.001900e-02 | 4.547770e-05 | `exact` |
| `test_audit2609_b4_collins_transport.py`'s `mismatch_matrix`, `fr = 0.90`, rebuilt from that fixture's own definition (1.31 um, N 1024, `w_in` 1 mm, NA 0.05) | 2.00 mm | 5.571183e-03 | **1.273287e-04** | `fresnel` |

and the arm this round adds, which removes the last way of reading the band as
a distance -- ONE leg, with only the envelope moved:

| input beam radius | `z_eff` | distance to `A = 0` | `theta_env` (rad) | departure | resolves |
|---|---|---|---|---|---|
| 0.80 mm | 7.777778e-03 m | 9.00 mm | 1.715395e-02 | 4.179416e-04 | **`fresnel`** |
| 0.60 mm | 7.777778e-03 m | 9.00 mm | 1.288341e-02 | 1.329790e-04 | `fresnel` |
| 0.40 mm | 7.777778e-03 m | 9.00 mm | 8.659722e-03 | 2.714408e-05 | **`exact`** |
| 0.30 mm | 7.777778e-03 m | 9.00 mm | 6.635604e-03 | 9.357961e-06 | `exact` |

`z_eff` is bit-identical down the column, and
`(theta_wide/theta_narrow)^4 / (dep_wide/dep_narrow)` reads
**1.000000000** on both builds: the quartic, with the distance cancelled.
That identity is what
`test_c5_three_defaults.py::test_a_wide_envelope_leg_far_from_any_focus_falls_back`
asserts, to 1e-6, beside the decision itself and beside premise gates on the
distance (> 1 mm from `A = 0`) and on the Rayleigh ranges (> 20).

One correction to the verification's own D6 table, since it is being quoted:
its `fr = 0.80` row prints `theta_env` 8.79e-03, which is inconsistent with
the departure printed in the same row.  Measured here on both builds it is
**1.001900e-02**, and `sqrt(3/2) k |z_eff| theta^4 / 8` at that angle
reproduces that row's own 4.5478e-05 exactly.  Nothing else in the table
moves, and the table's conclusion is unaffected -- `fr = 0.80` is further from
`A = 0` than `fr = 0.70` and does not fire.

## The kernel question, and how big the answer is

The verification answered the WP-C5 report's "could not be measured" on the
one suite leg this flip moves, against a closed-form NON-paraxial oracle.
Recorded here because it belongs with the decision and because its SIZE
matters as much as its sign:

| on `mismatch_matrix`'s `fr = 0.90` row | relative L2 against the exact scalar field |
|---|---|
| `gap_kernel='fresnel'` -- what `'auto'` now returns | **8.212599e-02** |
| `gap_kernel='exact'` -- what `'auto'` returned before | **8.222558e-02** |

The rule moves that leg TOWARD the truth.  It also changes the answer's error
by 9.96e-05 against a modelling error of 8.21e-02 -- a factor of 825.  Both
kernels are 8.2e-02 from the true field because the refinement lives in the
REDUCED frame on the ENVELOPE's angle while the leg's own non-paraxiality is
set by the BEAM's NA (0.05 at that fixture).  In plain language: the rule is
arbitrating one ten-thousandth of an eight-per-cent modelling error.  It is
the right call, and it is not what makes such a leg accurate -- a leg that is
genuinely non-paraxial needs a non-paraxial propagator, not a different
kernel.  This is now stated in `_GAP_KERNEL_ACCURACY_TAU`'s own note, in the
CHANGELOG and Migration caveats, and in ledger section 0.1.

## Merge note for WP-C3

WP-C3 makes `transport='collins'` the chain's default.  Every leg that moves
from `'sziklas'` to `'collins'` becomes a leg the kernel-departure rule can
see, because the rule lives inside `_collins_transport`.  The verification
quantified that by forcing the transport default over the same 27 carrier /
traced-chain files plus `test_c5_three_defaults.py` and
`test_wave5_h2_collins_jax.py`; the census is recorded here for the merge.

| | shipped (`'sziklas'` default) | forced `'collins'` |
|---|---|---|
| legs the `k4` gate resolved to `'exact'` -- the legs the rule can see | 167 | **284** |
| legs the rule MOVED | 24 | **96** |
| test ids that reach the rule | 48 | **73** |
| test ids with at least one fallback | 6 | **15** |
| pytest outcome of that run | 1036 passed, 1 skipped | 950 passed, 64 failed, 22 errors, 1 skipped |

**The nine ids that gain a fallback.**  None of them is near a focus:
`|z_eff|` runs 0.014 to 0.33 m (against VERIFY-B4 F3's 1600 m) while
`theta_env` runs to 1.26e-01 rad, 112x F3's, and the departures reach 9.33 --
93 000 tau.

| test id (all `tests/unit/`) | fallbacks, shipped -> collins | departure, min..max | `theta_env` (rad) | `\|z_eff\|` (m) |
|---|---|---|---|---|
| `test_niche_d3_guards.py::test_the_separation_survives_the_c10_residual_degree_and_is_caused_by_it` | 0 -> **25** | 1.448e-04 .. **2.396e+00** | 4.96e-03 .. 9.32e-02 | 0.0433 .. 0.3256 |
| `test_niche_d3_guards.py::test_c13_makes_the_d3_separation_build_independent` | 0 -> **15** | 1.448e-04 .. 2.396e+00 | 4.96e-03 .. 9.32e-02 | 0.0433 .. 0.3256 |
| `test_niche_d3_guards.py::test_the_guarded_input_really_is_the_wrong_answer` | 0 -> **15** | 1.448e-04 .. 2.396e+00 | 4.96e-03 .. 9.32e-02 | 0.0433 .. 0.3256 |
| `test_niche_d3_guards.py::test_the_residual_degree_moves_the_multiplexed_route_only_through_c6` | 0 -> **8** | 1.013e+00 .. 2.396e+00 | 4.54e-02 .. 9.32e-02 | 0.0433 .. 0.3256 |
| `test_niche_d3_guards.py::test_the_verdict_is_identical_through_a_slow_and_a_fast_chain` | 0 -> **4** | 1.637e-04 .. 2.364e-03 | 1.11e-02 .. 2.17e-02 | 0.0144 .. 0.0263 |
| `test_niche_gap_frame_observable.py::test_arm_c_has_its_own_knob_and_does_not_silence_arms_a_b` | 0 -> **2** | 9.331e+00 | 1.262e-01 | 0.0501 |
| `test_niche_gap_frame_observable.py::test_arm_c_catches_what_the_carrier_na_proxy_misses` | 0 -> **1** | 9.331e+00 | 1.262e-01 | 0.0501 |
| `test_niche_gap_frame_observable.py::test_gap_env_phi_tol_zero_disables_the_trip_but_keeps_the_number` | 0 -> **1** | 9.331e+00 | 1.262e-01 | 0.0501 |
| `test_niche_d2_chain_multi.py::test_memory_budget_is_honoured` | 0 -> **1** | 1.367e-04 | 6.92e-03 | 0.0811 |

The six ids that already fall back are unchanged:
`test_audit2609_b4_collins_transport.py` (1), `test_c5_three_defaults.py`
(1 + 19) and `test_wave5_h2_near_focus_table.py` (2 + 1 + 1).

**Recommendation for the merge.**

1. **Run C3's archive-to-archive comparison TWICE**, once with
   `carrier._GAP_KERNEL_ACCURACY_TAU = 1e-4` and once with it set to `None`,
   and report both.  Every one of the nine new ids is a CHAIN call, so a
   merged C3 changes them twice -- once by the transport and once by the
   kernel rule -- and only the transport change is visible in C3's digests
   unless the rule is disarmed for the comparison.  With `tau = None` the
   condition is not evaluated at all, so the second run isolates the
   transport exactly.
2. **Re-DERIVE the five `test_niche_d3_guards.py` bars; do not re-record
   them.**  What those ids score is a numerical SEPARATION between two
   routes.  A kernel change on both sides of such a comparison can move the
   separation without moving either side's correctness, so a bar that is
   simply re-recorded to the new reading stops testing what it was written to
   test.  The other four ids (`test_niche_gap_frame_observable.py` x3,
   `test_niche_d2_chain_multi.py::test_memory_budget_is_honoured`) score
   single quantities and can be re-recorded with their new values dated.
3. **Treat the census as a LOWER bound.**  The forced run had 64 failures and
   22 collection / fixture errors -- ids that assert the `'sziklas'` route,
   which is exactly what C3 will be re-pinning.  A leg inside a test that
   errors before reaching the transport is not counted, so the figure after
   C3 re-pins its fixtures can only grow.
4. **Nothing has scored the new legs against an oracle**, and at departures of
   1e+00 to 9.33 the exact-kernel refinement is far outside the regime the
   departure law was fitted in, so even the departure figure is an
   extrapolation there.  Falling back is almost certainly right at those
   readings; "almost certainly" is the honest word until something measures
   it.

The same three points are recorded in the ledger, section 0.1, as a merge
consequence in plain language.

## The mutation matrix, re-run on this tree

Twelve source mutations, one at a time, on a scratch copy of this branch tip
(`C:/tmp/c5b_mut` and `/mnt/c/tmp/c5b_mut_wsl`, never the worktree; the driver
restores the file in a `finally`), each scored against BOTH C5 test files --
48 ids now, the five this round adds included.  Driver:
`validation/probe_verify_c5/v_mutations.py`; results
`validation/probe_c5_round2/r2_mutations_{win,wsl}.json`.

Control on both builds: **48 passed** (the 43 the verification scored plus the
five this round adds).  **The two builds agree cell for cell AND on the SET of
ids that caught each mutation, not only on the counts.**

| mutation | ids that caught it (Windows) | ids that caught it (WSL) |
|---|---|---|
| item 1: `_GAP_KERNEL_ACCURACY_TAU` back to `None` | **10** | **10** |
| item 1: the rule keyed on the CONTAINMENT half-angle instead of the analytic one | **5** | **5** |
| item 2: `_dense_budget_floor_bytes` loses its fixed term | **4** | **4** |
| item 2: the floor notice suppressed | **2** | **2** |
| item 2: `_dense_cell_bytes` falls through to `'legacy'` on an unknown mode | **1** | **1** |
| item 2: `DENSE_MEM_BUDGET_ACCOUNTING` back to `'legacy'` | **2** | **2** |
| item 3: the fill keyed on the WINDOW's centre (`centre_out` dropped) | **9** | **9** |
| item 3: the fill keyed on HALF the period | **16** | **16** |
| item 3: `replica_fill` back to `'repeat'` on all three readouts | **6** | **6** |
| item 3: the refusal waived on `_collins_focus_readout` (carrier.py 2772) | **2** | **2** |
| item 3: the refusal waived on `carrier_referenced_focus_readout` (4493) | **2** | **2** |
| item 3: the refusal waived on `carrier_referenced_exact_focus_readout` (7235) | **1** | **1** |

**12 of 12 now caught by the two C5 files**, against 11 of 12 before this
round -- D9's finding closed.  The three per-readout rows read, by name:

| call site | readout | ids that caught it |
|---|---|---|
| 2772 | `_collins_focus_readout` | `test_verify_c5_three_defaults.py::test_an_off_axis_window_is_refused_where_the_same_ratio_is_served`, `test_c5_three_defaults.py::test_the_refusal_precedes_the_fill_on_the_collins_readout` |
| 4493 | `carrier_referenced_focus_readout` | `test_the_refusal_is_taken_before_the_fill_is_reached`, `test_the_replica_refusal_is_unchanged_by_the_fill` |
| 7235 | `carrier_referenced_exact_focus_readout` | `test_c5_three_defaults.py::test_the_refusal_precedes_the_fill_on_the_exact_readout` |

The two item-1 rows also gain one id each (9 -> 10 and 4 -> 5): the new
wide-envelope id reads the rule's decision off the running build, so both
disarming the rule and keying it on the wrong angle move it.

## Neutrality -- this round moves no byte

Every source edit in this round is a comment, a docstring or a name in prose.
That is not asserted; it is measured three ways.

**1. Digests, archive to archive.**  The verification's own three probes were
re-run unchanged against `PYTHONPATH=C:/tmp/c5b_arch` (this round's own
`git archive df6c73f3`, the branch tip round 2 started from) and against this
tree, on both builds, and the digest maps compared key by key
(`validation/probe_c5_round2/r2_n_item{1,2,3}_cmp_{win,wsl}.json`).  A digest
is SHA-256 over the array's exact bytes plus its shape and dtype.

| digest set | keys | identical | moved | only in one |
|---|---|---|---|---|
| item 1 -- the kernel ladders (hygiene-2 x9, F3, `mine`, b4, `wide_far`), Windows | 52 | **52** | **0** | 0 |
| item 1, WSL | 52 | **52** | **0** | 0 |
| item 2 -- 6 `'legacy'` + 6 `'measured'` explicit keys, 2 windowed, 4 default, Windows | 18 | **18** | **0** | 0 |
| item 2, WSL | 18 | **18** | **0** | 0 |
| item 3 -- the replica battery (16 oversized windows, the faithful set, both fills, the exact readout), Windows | 63 | **63** | **0** | 0 |
| item 3, WSL | 63 | **63** | **0** | 0 |

**133 keys per build, 266 in all, every one byte-identical to `df6c73f3`** --
including the six item-1 keys and the seventeen item-3 keys that the flips
themselves moved against `49ddf4bd`, which is the arm that says the
comparison is sensitive.  The fill's behaviour (D5) is inside that statement:
the item-3 set is the one the verification used to measure the flip, and it
does not move here.

**2. The history-document fingerprint gate.**
`python scripts/record_history_fingerprints.py --check` reads **OK: every
history document matches its module** with no re-record.  That gate hashes
the module's AST with docstrings and positions removed, and its
meaning-carrying token stream with comments and docstrings dropped -- so a
fingerprint that does not move IS the statement "this edit was documentation
only", made by machinery that was not written for this branch.  `carrier.md`
and `gbd.md` therefore carry no round-2 `re_recorded:` line, because nothing
they pin changed.

**3. The mutation matrix's control**, 48 passed on both builds, and every
mutation's caught-set identical between them.

## One red this round found, and what it was

Running `test_verify_c5_three_defaults.py` inside the 34-file WSL gate rather
than beside eight other files turned
`test_the_flip_bounds_the_resident_set_and_not_only_tracemalloc` red: its
`'measured'` arm read a sampled RSS delta of **exactly 0.0 MB** against a
197.2 MB `tracemalloc` peak, and its own premise gate refused -- correctly --
to conclude anything from two instruments that disagree by more than 2x.  Run
alone, the same id passed on both builds.

The mechanism was isolated rather than assumed
(`validation/probe_c5_round2/r2_rss_arena.py`, both builds).  glibc serves an
allocation above `M_MMAP_THRESHOLD` with `mmap` -- returned to the OS on free,
so the next one maps fresh pages and the resident set grows -- and anything
below it out of the retained heap, which grows nothing.  That threshold is
DYNAMIC: freeing an mmap'd block raises it, up to 32 MB.  Measured on WSL
py3.12, 200 MB of work in 2 MB blocks (the shape of this loop's chunk arrays):

| arm | cold process | after a larger block has been freed | ratio |
|---|---|---|---|
| 200 MB in 2 MB blocks, WSL | 2.0 MB of RSS growth | **0.0041 MB** | **481x** |
| 200 MB in ONE block, WSL | 192.5 MB | 196.0 MB | 1.0x |
| 200 MB in 2 MB blocks, Windows | 1.9 MB | 2.0 MB | 0.9x |

-- which is the 4 KB reading the id's own docstring already recorded, and why
the failure was WSL-only.  The id's precondition was therefore inherited from
whatever ran before it in the session.  Each arm now runs in a CHILD
interpreter that pays its first-touch page cost on one discarded warm-up call
and then measures, reporting both instruments as JSON, with the parent
asserting the child resolved the same `gbd.py`.  No bar changed and the claim
is unchanged; re-measured on Windows the id reads 0.713x / 0.712x
(`'measured'`) and 6.00x / 5.96x (`'legacy'`), separation 8.37x, against the
verification's 0.7135 / 0.7146, 5.9983 / 5.9924 and 8.39x.

This is `docs/TESTING_STANDARDS.md` rule 3 applied to an instrument rather
than to a physical quantity, and it is the only test change in this round that
is not an added id.

## Gates

All Windows runs carried `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1 LUMENAIRY_MEM_BUDGET_MB=2048 PYTHONPATH=C:/tmp/lum_c5b`,
`--capture=sys`, `-p no:randomly`; WSL the same with
`PYTHONPATH=/mnt/c/tmp/lum_c5b` and `~/lumvenv/bin/python`.

| what | Windows py3.14.6 / numpy 2.4.4 | WSL py3.12.3 / numpy 2.4.6 |
|---|---|---|
| the 34 carrier / traced-chain / readout files, both C5 files included (b4 whole on Windows, excluded on WSL) | **1133 passed, 3 skipped** in 2241 s | **1006 passed, 4 skipped** in 2416 s |
| `test_audit2609_b4_collins_transport.py` PER CLASS (16 classes) | run whole, inside the row above | **124 passed across 15 classes**; `TestGateCTwoGroupChain` alone hits the 900 s guard (rc 124) |
| the GBD budget files (`test_gbd_feature_complete.py`, `test_wave5_gbd_dense_mem_budget.py`, `test_verify_b14_known_reds.py`) | **36 passed** in 259 s | inside the 1006 row |
| the census / walker / dispatcher-pin / public-API / doc-consistency sweep, 35 files including `test_audit_except_budget.py` and `test_niche_audit_w4_input_kind.py` | **1739 passed, 14 skipped** in 565 s | -- |
| both C5 files alone, after the durations splice | **48 passed** in 152 s | **48 passed** (the mutation control) |
| `test_audit2609_a15a_durations_staleness.py` after the splice | **4 passed** in 41 s | -- |
| the mutation matrix, 12 mutations on a scratch copy | control **48 passed**; 12/12 caught | control **48 passed**; 12/12 caught, every set identical |
| the neutrality digests, three probes x two trees | 133 keys, **133 identical, 0 moved** | 133 keys, **133 identical, 0 moved** |

The three skips on Windows are `test_niche_exact_gap_kernel.py`'s `m == 1`
mathematical premise and two pre-existing ones; the 14 in the sweep are
pre-existing and none is a resource skip on a C5 path.
`test_public_api.py::test_installed_metadata_version_matches_source_version`
is GREEN here, as it was for the verification -- the WP-C5 report's "one
pre-existing red" remains a property of that agent's environment.

Walkers and tools:

```
python scripts/record_history_fingerprints.py --check   OK: every history document matches its module
python scripts/reanchor_citations.py --base f4f18851 --block "[5.47.0]"   10 re-anchored
python scripts/check_source_line_citations.py           declines: the v5.48.1 block carries no source:line citation (exit 2, as on the branch)
python scripts/check_doc_identifiers.py                 630 API-claiming tokens, 0 unresolved
python -m ruff check lumenairy/ tests/   (WSL)          All checks passed!
python -m mypy                                          Success: no issues found in 33 source files
```

`validation/` is in `pyproject.toml`'s `extend-exclude`, so the round-2 probes
are outside the ruff gate's scope; named explicitly they report 2
import-ordering notes, which is the directory's existing convention (the
verification's own probes report 3 and the WP-C5 probes 5).

`.test_durations` was re-captured for BOTH C5 files in one serial run with
BLAS pinned (48 ids, 151.55 s): **30 ids added** -- the whole of
`test_c5_three_defaults.py`, which had never been spliced -- and 18 replaced,
taking the file from 16 614 to 16 644 entries.  The re-captures include the
RSS id, whose child-process fixture moves it from 12.19 s to 26.40 s.  The
staleness gate's four ids pass.

No forward version token is in `lumenairy/`: the round-2 edits name 5.48.x
and the CHANGELOG's `## [Unreleased]`, never a release number the package has
not reached.

## What this round could not measure

* **The C3 merge itself.**  The census forces the `transport` DEFAULT; it does
  not merge C3's diff, which also adds a CuPy arm to `_collins_transport`.
  The table above is what the merge should be checked against, not a
  measurement of the merge.
* **The nine new Collins legs, physically.**  Recorded, not scored -- see
  recommendation 4.
* **Which kernel is right in general.**  The non-paraxial oracle is the exact
  scalar field of a Gaussian with a parabolic carrier.  It settles the b4 leg
  and the wide-envelope legs; it does not generalise to an arbitrary envelope
  and it says nothing about a vector field.
* **A CuPy device.**  Neither the kernel rule nor the fill was exercised on a
  GPU array in this round either.
* **`TestGateCTwoGroupChain` on WSL.**  The b4 file is run per class there;
  fifteen of its sixteen classes complete (124 ids, all passing) and that one
  hits the guard, which is the pre-existing spawn-pool condition
  VERIFY-WAVE5-HYGIENE2 round 2 section 11 records.  Nothing in this round
  touches it.
* **The floor as a bar on the RSS peak.**  Still measured, reported and
  asserted nowhere: even with the fresh-process fixture the reading is one
  sample of a quantity that spread 19 % run to run at N = 1024 on WSL.  The
  id asserts the flip's 8.4x separation instead, which is decades wide.
* **The durations on a quiet box.**  The `.test_durations` entries spliced
  below were measured serially with BLAS pinned, but on a workstation that
  also runs four sibling agents; they are weights for a shard balancer, and
  the gate they feed reads them as such.
