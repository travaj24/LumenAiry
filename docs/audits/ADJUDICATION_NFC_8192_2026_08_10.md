# ADJUDICATION -- is `n_fine_cap = 8192` safe as the production setting on the design-121 32-order fan?

Branch `perf/traced-hotpath` @ `6464384` (checked out, not modified).
Question, as posed: **does `n_fine_cap=8192` lower accuracy on ANY of the 32
fan orders, or not?**

---

## 0. VERDICT

> **YES -- on 16 of the 32 orders, and the previously published bound is too
> small by 11.7x.**
>
> The degradation is confined to **power bookkeeping** (`throughput`,
> `power_exit`, and the delivered per-frame cell power), where 16 orders
> breach the campaign's 4e-5 energy-honesty bar, the worst reading
> **-3.52e-04** at order `(+2,-1)`.
>
> **Spot quality never degrades on any order.**  FWHM is identical on all 32
> (delta exactly 0.000 um), and the largest encircled-energy move anywhere is
> **0.0079 points** against a 0.1-point bar -- 12.7x inside it.
>
> `AUDIT_TRACED_SPEED_2026_08_09.md` sec 7.4 concluded "the accuracy risk of
> the `8192` opt-in is not unknown: it is **bounded and measured at 3.0e-5**".
> That number is reproduced here exactly -- it is order `(-2,+0)`, which reads
> **-3.07e-05**.  But `(-2,+0)` is the *mildest* member of the affected class.
> The two-order sample happened to draw the gentlest available example; over
> all 32 orders the bound is **3.52e-04**.
>
> **The most actionable finding is separate from all of that: on this 128 GB
> box the shipped `n_fine_cap=16384` ALREADY runs at 8192, silently.**  The
> honest clamp needs 137.4 GB of budget to approve `n_fine=16384` (189.0 GB
> at the re-measured `_FINE_GRID_WORK_ARRAYS = 22`; see sec 2.1) and
> `get_ram_budget()` returns 102.9 GB, so without an explicit `ram_budget`
> override the "reference" setting degrades with a warning that a production
> log will swallow.  Every arm-B row in this document required that override
> to exist at all; the SHADOW column records that the un-overridden clamp
> would have returned **8192 in both arms**.

**Recommendation: keep 16384 for runs of record, adopt 8192 for exploration
-- and in either case set the budget explicitly.**  Detail in section 7.

---

## 1. BOX, BUILD, CONFIGURATION

```
Windows 11 Pro 10.0.26200        AMD Ryzen 9 5950X, 24 logical CPUs
127.9 GB physical RAM            get_ram_budget() = 102.9 GB at launch
python 3.14.6   numpy 2.4.4   scipy 1.17.1
lumenairy 5.33.1 (working tree = perf/traced-hotpath @ 6464384, clean)
```

Both arms run `validation/repro_traced_carrier_121/fan_multi_121.py`
**unmodified**, on the runner's own defaults and the capstone's own full-fan
common grid:

```
RN=1024  RS=4  DXO=0.2 um  NOUT=32768  TILE=1024  LEG=auto  WF=4.0
NW=1 (Newton pool serial -- CAPSTONE_D121 sec 5: the shipped 8-worker pool
      does not fit this 128 GB box)
OTEG=warn (see 2.3)
```

| | arm A | arm B (reference) |
|---|---|---|
| `n_fine_cap` | **8192** | **16384** |
| `output_grid['ram_budget']` | `inf` | `inf` |

`n_fine_cap` is the ONLY difference between the arms.

---

## 2. THE THREE THINGS THAT HAD TO BE GOT RIGHT BEFORE ANY NUMBER COUNTED

### 2.1 Without the RAM-budget override, arm B is arm A

`FIX_PERF_CACHES_BLUESTEIN_2026_08_09.md` sec 2.4 prices the honest clamp:
with `_FINE_GRID_WORK_ARRAYS = 16` and `_FINE_GRID_RAM_FRAC = 0.5`, approving
`n_fine = 16384` needs `16 * 16 B * 16384^2 / 0.5` = **137.4 GB of budget**.
(**SUPERSEDED**: that figure is the `_FINE_GRID_WORK_ARRAYS = 16` model this document ships.  The constant has since been re-measured twice -- 20 in `FIX_PERF_PARALLEL_2026_08_10.md` sec 3.2, then **22** in `FIX_VERIFY_PERF_2026_08_10.md` sec 4 -- and the budget `n_fine = 16384` needs is now **189.0 GB**, not 137.4.  The conclusion is unchanged and stronger: the shipped `NFC = 16384` is further out of reach on this box, not nearer.)
This box reports `get_ram_budget() = 102.9 GB`.  Measured directly, before any
chain was run:

```
  n_req= 8192 -> box-budget  8192 | ram_budget=inf  8192 | warnings 0
  n_req=16384 -> box-budget  8192 | ram_budget=inf 16384 | warnings 1
```

**So on this box the shipped `NFC=16384` silently degrades to 8192.**  A naive
dual-run would have compared 8192 against 8192 and reported "identical" --
the exact silent-wrong-answer class this campaign exists to catch.

Remedy #1 of sec 2.4 is used: `output_grid={'ram_budget': inf}`.  `ram_budget`
is an accepted `_OUTPUT_GRID_PASSTHROUGH` key, so it reaches BOTH
`_fine_trace_group_exit` and `carrier_referenced_exact_focus_readout`.  It is
applied to **both** arms, so the comparison isolates the COUNT cap and nothing
else.  `set_max_ram` was rejected as the vehicle: it is process-global and also
moves `asm.py` and `_lens_traced.py` batching decisions, which would have put a
second variable in the experiment.

**Verified end to end, per order, from the run's own trace:**

* arm A: leg `n_fine` = **8192** on all 32 orders, readout `n_fine` = 4096;
* arm B: leg `n_fine` = **16384** on all 32 orders, readout `n_fine` = 8192;
* SHADOW (what the un-overridden clamp would have returned): **8192 in both
  arms, on all 64 rows.**

Note the readout grid is not capped in either arm -- its own requirement is
4096 (arm A) / 8192 (arm B), both under their respective caps.  It moves only
because it follows the leg's pitch.  So `n_fine_cap` binds on the RETRACE LEG
and nowhere else, and the readout tracks it.

### 2.2 The leg's actual `n_fine` is asserted per order, not assumed

`carrier._memory_bounded_n_fine` is wrapped by a recorder that forwards the
call unchanged and logs `(n_req, label, ram_budget, window, nyquist_dx, n_out)`
plus the SHADOW value.  Every row below carries the grid the leg actually ran
on.

### 2.3 `on_tilt_exact_grid='warn'`, deliberately

The shipped fan default is `'error'`, which REFUSES an order whose retrace
pitch is coarser than the exit sphere's Nyquist pitch.  A refusal aborts the
whole shard, so an order that tripped at 8192 would vanish from the table
instead of appearing in it.  Both arms run `'warn'` and every warning is
captured and attributed to the order that raised it.

**As it turns out the guard never fires in either arm** (section 6.3), so
`'warn'` and `'error'` are numerically identical here -- `_guard_dispose` only
changes the disposition when the condition is met.

---

## 3. HARNESS

`validation/repro_traced_carrier_121/adjudicate_nfc_8192.py` (new; no library
edit, no edit to `fan_multi_121.py`).  It:

1. injects `output_grid['ram_budget']`;
2. installs the `_memory_bounded_n_fine` recorder (2.2);
3. forces the Newton pool to `n_workers=1`;
4. wraps the chain-multi `progress` hook for per-order wall times;
5. un-silences warnings (`simplefilter('always')` BEFORE the runner prepends
   its own three targeted `'ignore'`s) and tags each warning with the order;
6. `exec`s the runner with `__name__ = '__main__'` in a namespace the wrapper
   owns, so the runner's own arrays (`res`, `exact_c`, `cellP`, `fwhms`,
   `ee3s`, ...) survive its terminal `raise SystemExit` -- **every number in
   the table below is the runner's own, not a reimplementation of it**;
7. appends per-order rows to a CSV and dumps a per-shard JSON.

`validation/repro_traced_carrier_121/adj_rebuild_csv.py` rebuilds the CSV from
the JSONs.  Full 64-row output:
`validation/repro_traced_carrier_121/_adj_nfc_8192_rows.csv`.

**Sharding.**  16 shards of 2 orders each, in the runner's own lexsort order
(`my` then `mx`), `NOUT` pinned to 32768 so every shard runs on the same
full-fan common grid the capstone used.  Shards run serially, arms interleaved,
extreme orders first; a shard whose JSON exists is skipped, so the run is
resumable.  Two orders per shard rather than one because several of the
runner's own diagnostics (`np.diff` of the frame-centre lattice, `np.corrcoef`
of the share vectors) are undefined for a single order.

**Every one of the 32 shards exited 0** -- i.e. the fan acceptance's own five
checks passed in both arms, at every order.

### 3.1 Two instrument defects, found and fixed before they reached the table

Recorded because both were the silent kind:

1. **CSV column shift.**  The first writer joined on `,` by hand while two
   columns (`order` = `(-4,-2)`, `keep` = `-4,-2;-3,-2`) carry commas, which
   shifted every column to their right *without failing*.  Caught on the
   calibration shard by reading the row back; the writer now uses
   `csv.writer`, and the JSONs (never affected) are the source of truth.
2. **Wrong stage selected for the NA diagnostics.**  There are TWO stages
   flagged `exact_final`: the leg (`carrier.py:7783`, carries
   `na_exit_measured` / `exit_power_above_nyquist`) and a `'<target>'`
   bookkeeping stage appended after it (`carrier.py:7871`, no diagnostics).
   Taking `[-1]` silently returned `None` for every NA column.  Selection is
   now by the diagnostic, not by position.

Shards `s00` and `s12` were run before fix 2 and were re-run afterwards.

### 3.2 Determinism, and one contaminated shard

**Reps: one per order per arm**, justified by direct measurement rather than
assumption.  Shard `s00` and shard `s15` were each re-run from scratch
**6.5 hours apart, under completely different box load**, and reproduced
**bit-identically on all 13 scored fields** (`fwhm`, `ee3`, `ee6`, `ee12`,
`throughput`, `capture`, `power_exit`, `power_out`, `cellP`, `peak_I`,
`halo_amax`, `dx_fine`, `n_fine`) in both arms.

That re-run was not optional.  A background task holding the first shard
driver hit a 1-hour cap and was killed, but the kill orphaned its child and
left its shell alive, so for about six minutes **two processes ran shard 15
concurrently**.  Shard 15 carries `(+2,+1)`, one of the orders that moves
between arms, so it was re-run clean rather than argued about.  It reproduced
bit-identically; the contention moved wall time and RSS, not bits.  (It could
not have: `ram_budget=inf` removes the only branch in this path that reads
free RAM.)  The calibration shard also reproduced
`CAPSTONE_D121_2026_08_06.md` sec 6.3's published `(-4,-2)` row exactly.

---

## 4. SCORING BARS

| quantity | bar | source |
|---|---|---|
| encircled energy EE3 / EE6 / EE12 | 0.1 point | campaign bar |
| energy honesty (throughput, capture, per-frame power, `power_exit`) | 4e-5, RELATIVE | chain's own energy honesty |
| banner digits | exact string match on the frame table's own format specs | `fan_multi_121.py:528` |
| FWHM | 1 nm printed (metric itself quantised to `2*dx_out` = 0.4 um) | banner |
| halo amax | 5 % relative | this document |

`VERDICT` per order: **IDENTICAL** (all banner digits match and every raw delta
is 0) / **IDENTICAL\*** (banner-identical, sub-printed-digit deltas only) /
**BOUNDED** (inside every bar, but a printed digit moves) / **DEGRADED**
(arm A worse than arm B by more than a bar on any scored quantity).

"Halo amax" is the library's own `amax_halo` convention (max AMPLITUDE beyond a
radius as a fraction of the peak amplitude) applied to the readout tile about
its own peak, at r > 12 / 20 / 40 um.

`cell_in_pct` is a percent (~2.75), so its 4e-5 bar is applied RELATIVELY.
Applying it as percentage-points would have understated every cell delta by
36x and hidden the whole finding.

---

## 5. RESULTS

### 5.1 The 32 x 2 table

`dX = arm A (8192) - arm B (16384)`.  `dEE` in POINTS, `dcell` / `dthru` / `dcap` RELATIVE.

| order | dFWHM um | dEE3 pt | dEE6 pt | dEE12 pt | dcell | dthru | dcap | dhalo12 | banner | VERDICT |
|---|---|---|---|---|---|---|---|---|---|---|
| `(-4,-2)` | 0.000 | -0.0006 | -0.0003 | -0.0000 | +8.97e-07 | +9.15e-07 | -2.41e-08 | +0.41% | same | IDENTICAL* |
| `(-3,-2)` | 0.000 | +0.0001 | +0.0000 | -0.0000 | +5.76e-07 | +3.84e-07 | +1.89e-07 | +0.22% | same | IDENTICAL* |
| `(-2,-2)` | 0.000 | +0.0000 | +0.0001 | -0.0000 | +1.93e-07 | +3.65e-07 | -1.75e-07 | +0.10% | DIFF | BOUNDED |
| `(-1,-2)` | 0.000 | -0.0031 | -0.0007 | +0.0003 | -2.28e-04 | -2.33e-04 | +6.56e-06 | +3.21% | DIFF | **DEGRADED** |
| `(+0,-2)` | 0.000 | -0.0079 | -0.0036 | +0.0003 | -2.61e-05 | -3.07e-05 | +4.89e-06 | +1.72% | DIFF | BOUNDED |
| `(+1,-2)` | 0.000 | -0.0033 | -0.0021 | +0.0003 | -4.73e-05 | -5.54e-05 | +8.47e-06 | +2.61% | DIFF | **DEGRADED** |
| `(+2,-2)` | 0.000 | +0.0001 | +0.0001 | -0.0000 | +2.75e-07 | +3.77e-07 | -1.05e-07 | -0.06% | same | IDENTICAL* |
| `(+3,-2)` | 0.000 | -0.0001 | -0.0000 | -0.0000 | +1.92e-07 | +4.06e-07 | -2.16e-07 | +0.04% | same | IDENTICAL* |
| `(-4,-1)` | 0.000 | -0.0004 | -0.0001 | +0.0000 | -2.24e-06 | -3.89e-06 | +1.68e-06 | -0.19% | DIFF | BOUNDED |
| `(-3,-1)` | 0.000 | -0.0002 | -0.0001 | -0.0000 | -1.64e-07 | -5.51e-07 | +3.90e-07 | +0.39% | same | IDENTICAL* |
| `(-2,-1)` | 0.000 | -0.0031 | -0.0007 | +0.0003 | -2.28e-04 | -2.33e-04 | +6.56e-06 | +3.21% | DIFF | **DEGRADED** |
| `(-1,-1)` | 0.000 | -0.0050 | +0.0005 | +0.0003 | -2.91e-04 | -3.06e-04 | +1.72e-05 | +4.68% | DIFF | **DEGRADED** |
| `(+0,-1)` | 0.000 | -0.0020 | +0.0017 | +0.0003 | -7.31e-05 | -9.05e-05 | +1.80e-05 | +4.97% | DIFF | **DEGRADED** |
| `(+1,-1)` | 0.000 | -0.0047 | -0.0007 | +0.0003 | -1.09e-04 | -1.27e-04 | +1.87e-05 | +4.49% | DIFF | **DEGRADED** |
| `(+2,-1)` | 0.000 | +0.0037 | +0.0040 | +0.0003 | -3.35e-04 | **-3.52e-04** | +1.94e-05 | +2.98% | DIFF | **DEGRADED** |
| `(+3,-1)` | 0.000 | -0.0001 | -0.0001 | +0.0000 | +5.70e-07 | +4.62e-07 | +1.05e-07 | +0.05% | DIFF | BOUNDED |
| `(-4,+0)` | 0.000 | -0.0006 | -0.0006 | -0.0001 | +1.50e-07 | +8.91e-08 | +6.07e-08 | +0.64% | same | IDENTICAL* |
| `(-3,+0)` | 0.000 | -0.0003 | -0.0002 | -0.0001 | +3.39e-07 | +3.42e-07 | -5.61e-09 | +0.44% | same | IDENTICAL* |
| `(-2,+0)` | 0.000 | -0.0079 | -0.0036 | +0.0003 | -2.61e-05 | -3.07e-05 | +4.89e-06 | +1.72% | DIFF | BOUNDED |
| `(-1,+0)` | 0.000 | -0.0020 | +0.0017 | +0.0003 | -7.31e-05 | -9.05e-05 | +1.80e-05 | +4.97% | DIFF | **DEGRADED** |
| `(+0,+0)` | 0.000 | -0.0068 | -0.0008 | +0.0003 | +1.42e-04 | +1.23e-04 | +1.86e-05 | +4.87% | DIFF | **DEGRADED** |
| `(+1,+0)` | 0.000 | -0.0047 | -0.0010 | +0.0002 | +1.05e-04 | +8.52e-05 | +1.94e-05 | +3.59% | DIFF | **DEGRADED** |
| `(+2,+0)` | 0.000 | +0.0039 | +0.0032 | +0.0002 | -1.15e-04 | -1.33e-04 | +1.82e-05 | +1.35% | DIFF | **DEGRADED** |
| `(+3,+0)` | 0.000 | -0.0001 | -0.0001 | -0.0000 | +3.54e-07 | +4.36e-07 | -8.45e-08 | +0.18% | DIFF | BOUNDED |
| `(-4,+1)` | 0.000 | -0.0001 | -0.0000 | -0.0000 | +7.40e-07 | +6.43e-07 | +9.23e-08 | +0.36% | same | IDENTICAL* |
| `(-3,+1)` | 0.000 | -0.0001 | -0.0000 | -0.0000 | +3.39e-07 | +2.87e-07 | +5.06e-08 | +0.06% | same | IDENTICAL* |
| `(-2,+1)` | 0.000 | -0.0033 | -0.0021 | +0.0003 | -4.73e-05 | -5.54e-05 | +8.47e-06 | +2.61% | DIFF | **DEGRADED** |
| `(-1,+1)` | 0.000 | -0.0047 | -0.0007 | +0.0003 | -1.09e-04 | -1.27e-04 | +1.87e-05 | +4.49% | DIFF | **DEGRADED** |
| `(+0,+1)` | 0.000 | -0.0047 | -0.0010 | +0.0002 | +1.05e-04 | +8.52e-05 | +1.94e-05 | +3.59% | DIFF | **DEGRADED** |
| `(+1,+1)` | 0.000 | -0.0043 | -0.0018 | +0.0002 | +7.15e-05 | +5.09e-05 | +2.03e-05 | +4.28% | DIFF | **DEGRADED** |
| `(+2,+1)` | 0.000 | +0.0041 | +0.0028 | +0.0003 | -1.55e-04 | -1.75e-04 | +2.18e-05 | +1.62% | DIFF | **DEGRADED** |
| `(+3,+1)` | 0.000 | -0.0000 | -0.0000 | +0.0000 | +2.44e-07 | +3.43e-07 | -1.01e-07 | +0.13% | DIFF | BOUNDED |

**VERDICT COUNTS: IDENTICAL\* = 9, BOUNDED = 7, DEGRADED = 16 (of 32).**

### 5.2 The worst order, characterised

`(+2,-1)`:

| quantity | arm A (8192) | arm B (16384) | delta | bar | inside/outside |
|---|---|---|---|---|---|
| `throughput` | 0.9931477 | 0.9934998 | **-3.52e-04** | 4e-5 | **8.8x OUTSIDE** |
| `power_exit` | -- | -- | **-3.54e-04** | 4e-5 | **8.9x OUTSIDE** |
| cell power / P_in | -- | -- | **-3.35e-04** | 4e-5 | **8.4x OUTSIDE** |
| `capture` | 0.9999960 | 0.9999766 | +1.94e-05 | 4e-5 | inside |
| FWHM | 3.400 um | 3.400 um | **0.000 um** | 1 nm | identical |
| EE3 | 90.802 % | 90.799 % | +0.0037 pt | 0.1 pt | 27x inside |
| EE6 | 99.859 % | 99.855 % | +0.0040 pt | 0.1 pt | 25x inside |
| EE12 | 99.994 % | 99.994 % | +0.0003 pt | 0.1 pt | 333x inside |
| halo amax (r>12 um) | -- | -- | +2.98 % | 5 % | inside |

The other four worst:

* `(-1,-1)` -- dthru **-3.06e-04**, dcell -2.91e-04, dEE3 -0.0050 pt, dFWHM 0.000 um, halo +4.68 %
* `(-1,-2)` -- dthru **-2.33e-04**, dcell -2.28e-04, dEE3 -0.0031 pt, dFWHM 0.000 um, halo +3.21 %
* `(-2,-1)` -- dthru **-2.33e-04**, dcell -2.28e-04, dEE3 -0.0031 pt, dFWHM 0.000 um, halo +3.21 %
* `(+2,+1)` -- dthru **-1.75e-04**, dcell -1.55e-04, dEE3 +0.0041 pt, dFWHM 0.000 um, halo +1.62 %

Aggregates over all 32 orders:

| | value | at | bar | verdict |
|---|---|---|---|---|
| max abs `dthru` | **3.52e-04** | `(+2,-1)` | 4e-5 | 8.8x OUTSIDE |
| max abs `dcell` (relative) | **3.35e-04** | `(+2,-1)` | 4e-5 | 8.4x OUTSIDE |
| max abs `dEE3` | **0.0079 pt** | `(+0,-2)`, `(-2,+0)` | 0.1 pt | 12.7x inside |
| max abs `dFWHM` | **0.0000 um** | -- | 1 nm | never moves |
| max halo amax increase | **+4.97 %** | `(-1,+0)` | 5 % | just inside |

**The sign of the power delta is not systematic** -- 12 of the 16 degraded
orders lose power in arm A, 4 gain it (`(+0,+0)` +1.23e-04, `(+1,+0)` and
`(+0,+1)` +8.52e-05, `(+1,+1)` +5.09e-05).  This is aliased outer-NA content
being REDISTRIBUTED, not a systematic vignetting loss.  That is consistent with
`apply_real_lens_traced`'s own warning, which says the beyond-Nyquist annulus
"ALIASES: far-halo energy lands at wrong radii" rather than disappearing.

### 5.3 Wall time and memory

| arm | orders | per-order s (min..max, mean) | sum | shards | peak RSS GB | min avail GB |
|---|---|---|---|---|---|---|
| A (8192) | 32 | 121.9 .. 164.9 (141.6) | 4531.4 s = **1.26 h** | 16 | 32.39 .. 40.09 | 56.05 |
| B (16384) | 32 | 465.4 .. 603.3 (526.7) | 16855.5 s = **4.68 h** | 16 | 71.89 .. 80.46 | 20.26 |

**Arm B costs 3.72x the wall time and 2.1x the peak RSS of arm A.**  Total
campaign wall (including per-shard setup, the two re-run shards and the
contaminated-shard redo): arm A 1.33 h, arm B 4.75 h.

Both arms are materially faster than the capstone's 910.9 s/order at 16384,
because `6464384` carries the `_ResidualEikonal._poly` power-cache (audit item
2).  The measured 526.7 s/order at 16384 is a **1.73x** speed-up on the
capstone's own configuration -- a by-product of this campaign, not its subject.

---

## 6. NYQUIST-MARGIN TABLE -- AND THE TWO REFERENCES THAT DISAGREE

The task asked for "each order's exit-sphere Nyquist pitch vs the retrace
pitch ... the physics predictor of where 8192 could bite".  There are **two**
exit-sphere Nyquist pitches on this leg and they do not agree:

| | NA | Nyquist pitch `lambda/(2 NA)` | what uses it |
|---|---|---|---|
| **SIZED** | paraxial `na_exit` = 0.4051 | ~1.615 um | `_fine_trace_group_exit`'s F-D warning and the D6 `on_tilt_exact_grid` refusal |
| **MEASURED** | `na_exit_measured` = 0.4620 .. 0.5349 | 1.22 .. 1.42 um | `apply_real_lens_traced`'s `on_undersample` warning, and the physics |

### 6.1 Per-order margins

`pAN` = `exit_power_above_nyquist`, the library's own `|E|^2`-weighted fraction
of exit power above the grid's Nyquist NA -- what is actually thrown away.

| order | win mm | dx_fine A um | dx_fine B um | margin SIZED A | margin SIZED B | NA measured | margin MEASURED A | pAN A | pAN B |
|---|---|---|---|---|---|---|---|---|---|
| `(-4,-2)` | 12.4874 | 1.5243 | 0.7622 | 1.0608 | 2.1216 | 0.5321 | 0.8075 | 1.438e-03 | 0.000e+00 |
| `(-3,-2)` | 12.4874 | 1.5243 | 0.7622 | 1.0597 | 2.1193 | 0.5211 | 0.8245 | 1.753e-03 | 0.000e+00 |
| `(-2,-2)` | 12.4874 | 1.5243 | 0.7622 | 1.0588 | 2.1175 | 0.5012 | 0.8573 | 1.590e-03 | 0.000e+00 |
| `(-1,-2)` | 12.5539 | 1.5325 | 0.7662 | 1.0527 | 2.1053 | 0.4999 | 0.8549 | 1.681e-03 | 0.000e+00 |
| `(+0,-2)` | 12.5539 | 1.5325 | 0.7662 | 1.0527 | 2.1053 | 0.5001 | 0.8547 | 1.733e-03 | 0.000e+00 |
| `(+1,-2)` | 12.5539 | 1.5325 | 0.7662 | 1.0527 | 2.1053 | 0.5000 | 0.8549 | 1.650e-03 | 0.000e+00 |
| `(+2,-2)` | 12.4874 | 1.5243 | 0.7622 | 1.0588 | 2.1176 | 0.5012 | 0.8573 | 1.518e-03 | 0.000e+00 |
| `(+3,-2)` | 12.4874 | 1.5243 | 0.7622 | 1.0597 | 2.1193 | 0.5201 | 0.8262 | 1.582e-03 | 0.000e+00 |
| `(-4,-1)` | 12.4874 | 1.5243 | 0.7622 | 1.0603 | 2.1206 | 0.5349 | 0.8033 | 1.421e-03 | 0.000e+00 |
| `(-3,-1)` | 12.4874 | 1.5243 | 0.7622 | 1.0591 | 2.1183 | 0.5199 | 0.8264 | 1.905e-03 | 0.000e+00 |
| `(-2,-1)` | 12.5539 | 1.5325 | 0.7662 | 1.0527 | 2.1053 | 0.4999 | 0.8549 | 1.681e-03 | 0.000e+00 |
| `(-1,-1)` | 12.5539 | 1.5325 | 0.7662 | 1.0522 | 2.1043 | 0.4811 | 0.8883 | 1.969e-03 | 0.000e+00 |
| `(+0,-1)` | 12.5539 | 1.5325 | 0.7662 | 1.0522 | 2.1043 | 0.4804 | 0.8896 | 2.006e-03 | 0.000e+00 |
| `(+1,-1)` | 12.5539 | 1.5325 | 0.7662 | 1.0522 | 2.1043 | 0.4813 | 0.8880 | 1.943e-03 | 0.000e+00 |
| `(+2,-1)` | 12.5539 | 1.5325 | 0.7662 | 1.0527 | 2.1053 | 0.4994 | 0.8559 | 1.627e-03 | 0.000e+00 |
| `(+3,-1)` | 12.4874 | 1.5243 | 0.7622 | 1.0591 | 2.1183 | 0.5192 | 0.8275 | 1.753e-03 | 0.000e+00 |
| `(-4,+0)` | 12.4874 | 1.5243 | 0.7622 | 1.0603 | 2.1206 | 0.4623 | 0.9292 | 1.482e-03 | 0.000e+00 |
| `(-3,+0)` | 12.4874 | 1.5243 | 0.7622 | 1.0591 | 2.1183 | 0.5184 | 0.8287 | 1.970e-03 | 0.000e+00 |
| `(-2,+0)` | 12.5539 | 1.5325 | 0.7662 | 1.0527 | 2.1053 | 0.5001 | 0.8547 | 1.733e-03 | 0.000e+00 |
| `(-1,+0)` | 12.5539 | 1.5325 | 0.7662 | 1.0522 | 2.1043 | 0.4804 | 0.8896 | 2.006e-03 | 0.000e+00 |
| `(+0,+0)` | 12.5539 | 1.5325 | 0.7662 | 1.0522 | 2.1043 | 0.4620 | 0.9250 | 2.039e-03 | 0.000e+00 |
| `(+1,+0)` | 12.5539 | 1.5325 | 0.7662 | 1.0522 | 2.1043 | 0.4802 | 0.8896 | 1.980e-03 | 0.000e+00 |
| `(+2,+0)` | 12.5539 | 1.5325 | 0.7662 | 1.0527 | 2.1053 | 0.4997 | 0.8557 | 1.684e-03 | 0.000e+00 |
| `(+3,+0)` | 12.4874 | 1.5243 | 0.7622 | 1.0591 | 2.1183 | 0.5189 | 0.8290 | 1.818e-03 | 0.000e+00 |
| `(-4,+1)` | 12.4874 | 1.5243 | 0.7622 | 1.0603 | 2.1206 | 0.5349 | 0.8033 | 1.384e-03 | 0.000e+00 |
| `(-3,+1)` | 12.4874 | 1.5243 | 0.7622 | 1.0591 | 2.1183 | 0.5199 | 0.8264 | 1.855e-03 | 0.000e+00 |
| `(-2,+1)` | 12.5539 | 1.5325 | 0.7662 | 1.0527 | 2.1053 | 0.5000 | 0.8549 | 1.650e-03 | 0.000e+00 |
| `(-1,+1)` | 12.5539 | 1.5325 | 0.7662 | 1.0522 | 2.1043 | 0.4813 | 0.8880 | 1.943e-03 | 0.000e+00 |
| `(+0,+1)` | 12.5539 | 1.5325 | 0.7662 | 1.0522 | 2.1043 | 0.4802 | 0.8896 | 1.980e-03 | 0.000e+00 |
| `(+1,+1)` | 12.5539 | 1.5325 | 0.7662 | 1.0522 | 2.1043 | 0.4812 | 0.8879 | 1.916e-03 | 0.000e+00 |
| `(+2,+1)` | 12.5539 | 1.5325 | 0.7662 | 1.0527 | 2.1053 | 0.4994 | 0.8559 | 1.597e-03 | 0.000e+00 |
| `(+3,+1)` | 12.4874 | 1.5243 | 0.7622 | 1.0591 | 2.1183 | 0.5192 | 0.8275 | 1.753e-03 | 0.000e+00 |

Three facts fall straight out of that table:

1. **`pAN_B = 0.000e+00 on all 32 orders.**  The 16384 grid captures the entire
   measured exit NA; nothing is discarded.  Arm B is a genuine converged
   reference, not merely a finer arm.
2. **`pAN_A = 1.38e-03 .. 2.04e-03 on all 32 orders.**  The 8192 grid discards
   0.14 - 0.20 % of the exit power on EVERY order, degraded or not.
3. **`pAN` does not predict which orders degrade.**  The largest `pAN_A`
   (2.039e-03, `(+0,+0)`) is degraded, but the second largest (1.970e-03,
   `(-3,+0)`) is IDENTICAL\*, and `(-3,-1)` at 1.905e-03 is IDENTICAL\* too.
   *How much* is discarded does not matter; *how it aliases* does.

### 6.2 The separator that DOES work -- the retrace-window quantisation

`win = n_crop * cur_dx` with `n_crop = 2*round((window_factor*w / cur_dx)/2)`,
so the retrace window is quantised to `2*cur_dx` and lands on one of exactly
two values across the fan.  That, not the NA, is the discriminant:

| class | n | retrace window | dx_fine at 8192 | margin SIZED | abs `dthru` range | breach 4e-5 |
|---|---|---|---|---|---|---|
| **SMALL window** | 14 | 12.4874 mm | 1.5243 um | 1.0588 - 1.0608 | 8.91e-08 .. **3.89e-06** | **0 / 14** |
| **LARGE window** | 18 | 12.5539 mm | 1.5325 um | 1.0522 - 1.0527 | **3.07e-05** .. 3.52e-04 | **16 / 18** |

**The two classes do not overlap: 3.89e-06 (small-class max) to 3.07e-05
(large-class min) is a 7.9x gap.**  Every order that moves at all is in the
large-window class; every order in the small-window class is three orders of
magnitude inside the bar.

The two large-window orders that do NOT breach -- `(+0,-2)` and `(-2,+0)`, both
at 3.07e-05 -- are inside the bar by 1.3x, not by a margin worth relying on.
The honest statement is that the class is **at risk**, and 16 of its 18 members
realise the risk.

### 6.3 Neither library guard fires on the orders that degrade

This is the part worth carrying forward:

* The **D6 `on_tilt_exact_grid` refusal** and the **F-D warning** both test the
  SIZED margin, which is `> 1` on all 32 orders in BOTH arms (1.0522 .. 1.0608
  at 8192).  **They are silent on all 32 orders, including all 16 that
  degrade.**  This reproduces `AUDIT_TRACED_SPEED_2026_08_09.md` sec 7.1's
  observation that no `on_tilt_exact_grid` message is emitted -- and shows that
  the silence is not evidence of safety.
* The **`apply_real_lens_traced` `on_undersample` warning** tests the MEASURED
  margin, which is `< 1` on all 32 orders at 8192 (0.8033 .. 0.9292).  **It
  fires on all 32, including the 14 that are provably fine.**

So one guard never fires and the other always fires; neither separates the 16
degraded orders from the 16 that are not.  The quantity that does separate them
-- the retrace-window quantisation step -- is not exposed by any guard.

---

## 7. RECOMMENDATION

### 7.1 The answer to the question as asked

**`n_fine_cap=8192` DOES lower accuracy, on 16 of the 32 orders, but only in
power bookkeeping and only at the 1e-4 level.**  Whether that matters is
entirely a question of which bar governs:

| bar | reading at 8192 | pass? |
|---|---|---|
| fan acceptance, `max abs(share/design - 1) < 0.02` | worst per-order perturbation 3.35e-04 | **PASS, ~60x inside** (all 32 shards exited 0 in both arms) |
| spot quality, EE within 0.1 pt / FWHM to 1 nm | worst 0.0079 pt; FWHM never moves | **PASS, 12.7x inside** |
| chain energy honesty, 4e-5 | worst 3.52e-04 | **FAIL on 16 / 32 orders, worst by 8.8x** |

### 7.2 What I recommend

**Keep `n_fine_cap = 16384` for runs of record, and set the budget explicitly.**
Reasons, in order:

1. Arm B is the *converged* arm in a strong sense, not merely the finer one:
   `exit_power_above_nyquist = 0` on all 32 orders.  Arm A discards 0.14-0.20 %
   of exit power on every order.  Reporting per-order power from a grid that
   provably discards outer-NA content, when the converged grid is affordable,
   is the kind of thing this campaign's own energy-honesty bar exists to stop.
2. The failure is **unpredictable from anything a caller can see**: neither
   guard separates the affected orders, and the discriminant is an internal
   window quantisation.
3. The cost is real but bounded: **4.68 h vs 1.26 h** of chain-B time for the
   fan, and **80.5 GB vs 40.1 GB** peak RSS.  4.68 h still clears the
   capstone's 24 h bar by 5.1x, and it is 1.73x faster than the capstone's own
   projection because of the `_poly` power-cache in this commit.

**RAM needed to hold 16384 honestly: 137.4 GB of `get_ram_budget()`** (16 work
arrays x 16 B x 16384^2 / 0.5; **189.0 GB** at the re-measured 22 arrays --
`FIX_VERIFY_PERF_2026_08_10.md` sec 4).  This box has 102.9 GB, so the setting must be
made reachable one of two ways:

* `output_grid={'ram_budget': float('inf')}` (used here -- targeted, affects
  only the readout path), having read that the measured peak is 80.5 GB; or
* `lumenairy.set_max_ram(140e9)` (process-global; also moves `asm` and
  `_lens_traced` batching, which is why it was not used here).

**Adopt 8192 for exploration, sweeps and optimisation loops**, where the 3.7x
speed and 2.1x memory saving are worth a 1e-4 power perturbation and the spot
metrics are exact anyway.  This is also what makes order-level parallelism
reachable (`AUDIT_TRACED_SPEED` sec 3).

### 7.3 The hybrid, stated because the margin table does separate

The task asked for a per-order rule if the margin table separates cleanly.  It
does (6.2), so here is the rule:

> Run the leg at `n_fine_cap = 16384` when the retrace window lands on the
> upper quantisation step -- equivalently when the SIZED margin
> `lambda/(2*na_exit) / (win/8192) < 1.055` -- and at 8192 otherwise.

On design 121 that selects 18 of 32 orders, and costs
`18*526.7 + 14*141.6 = 11463 s = 3.18 h` against 4.68 h for all-16384: a
**32 % saving**.

**I do not recommend adopting it.**  18 of 32 orders need the fine grid anyway,
so the saving is modest; the threshold 1.055 sits in a 0.006-wide gap between
1.0527 and 1.0588 and is calibrated on one design at one `window_factor`; and
`n_fine_cap` is a per-CALL knob, so using it per-order means the orchestrator
must size the window before choosing the cap -- a library change, not a
configuration change.  The rule's value here is diagnostic: it identifies which
orders to check first on any future design.

### 7.4 Documentation that is now stale

Not changed by this document (it makes no library or runner edits), but flagged:

* `AUDIT_TRACED_SPEED_2026_08_09.md` sec 7.3-7.4 -- "the audit recommends the
  `n_fine_cap=8192` path" and "bounded and measured at 3.0e-5".  The bound is
  3.52e-04 over the full fan; the 3.0e-5 figure is order `(-2,+0)`, the
  mildest member of the affected class.  Its own stated limit ("two orders, not
  thirty-two") is exactly where it went wrong, and it said so.
* `fan_multi_121.py:455` -- "with NFC >= 16384 on design 121" is CORRECT and
  should stay; this document supplies the evidence it previously lacked.
* Anything that treats a silent `on_tilt_exact_grid` as evidence that the
  retrace grid is adequate (6.3).

---

## APPENDIX -- artifacts

* Per-order rows, 64 x 65 columns:
  `validation/repro_traced_carrier_121/_adj_nfc_8192_rows.csv`
* Harness: `validation/repro_traced_carrier_121/adjudicate_nfc_8192.py`
* CSV rebuild: `validation/repro_traced_carrier_121/adj_rebuild_csv.py`
* Per-shard JSON dumps (clamp records, warning inventory, progress marks),
  shard logs, and the v1 copies of the two re-run shards: session scratchpad
  `nfc_adj/`.
