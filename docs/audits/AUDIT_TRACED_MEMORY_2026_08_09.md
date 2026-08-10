# AUDIT -- memory footprint of the traced propagation path (design 121)

**2026-08-09.  Branch `main`, commit `c8bcbcb` (v5.33.1).  REPORT-ONLY: no
file under `lumenairy/**` or `validation/**` was modified, no git command was
run, `CHANGELOG.md` was not touched.  Every probe lives in the session
scratchpad.**

Scope: items 1-6 of the memory-footprint assignment, against the workload
`validation/repro_traced_carrier_121`.  Prior art read first:
`CAPSTONE_D121_2026_08_06.md` (Stage B 21 GB serial, C-2 92 GB at the full-fan
grid), `HANDOFF_TRACED_EXACT_2026_08_05.md` sec 4 (the ~8*N^2 c128 chain
working set; complex64 UNMEASURED through a 6-group chain),
`FIX_POOL_MEMORY_2026_08_06.md` (the per-worker model).

---

## 0. VERDICT

> **One design-121 order, at the shipped production configuration
> (`RN=1024, RS=4, NFC=16384, WF=4.0, TILE=1024, DXO=0.2 um`, exact final leg,
> serial Newton) peaks at **98.85 GB RSS / 99.88 GB peak working set /
> 101.74 GB commit** -- measured from OUTSIDE the process, with no
> instrumentation thread inside it -- and leaves a 137 GB box with **18.4 GB**
> free.  At the peak plateau **69.26 GB** of that is live
> full-grid ndarrays, and **at least 25.8 GB of those 69.26 GB** are held by
> implementation accident, not by the physics: five consumed phase factors that
> `_fine_trace_group_exit` never releases (21.5 GB) and full 2-D meshgrids
> where a broadcast would do (4.3 GB there, 6.4 GB more at the absolute peak
> instant).  A further **24.8 GB** is retained pyFFTW plan-cache
> double-buffers, which one documented, byte-identical library call halves, and
> **4.3 GB** is the pre-leg field pinned alive through the whole readout.**

> **complex64: the ledgered halving is NOT AVAILABLE on this path as shipped,
> and this is a measurement, not an inference.  Requesting it end-to-end
> (`set_default_complex_dtype(np.complex64)` + a complex64 source) saves
> 0.0 GB -- measured peak RSS 64.56 GB against the complex128 run's
> 64.45 GB, i.e. 0.2 % HIGHER, inside the 1.1 % run-to-run spread.  The reason
> is visible in the dtype trace: complex64 survives exactly ONE chain leg and
> is upcast at the first group hand-off, so the entire memory-dominant exact
> final leg runs complex128 either way.  With only one leg at c64 the
> acceptance banner is unchanged to every printed digit and EE3 agrees to
> 1e-8, which is an accuracy verdict about ONE LEG and must not be read as a
> verdict about a complex64 chain.**

> **window_factor: the handoff's "quadratic in `window_factor`; `n_fine_cap`
> does NOT bound that dimension" is half right, and the half that is wrong is
> the expensive one.  `n_fine_cap` DOES bound the exact leg's retrace grid
> (`_fine_trace_group_exit` takes `min(n_fine_req, n_fine_cap)` and then the
> RAM clamp), so at the shipped `NFC=16384` every `wf` in 3..7 lands on the
> SAME 16384^2 grid -- `window_factor` is not the memory lever there.  It is
> exactly right about `carrier_referenced_exact_focus_readout`, which is
> handed no `n_fine_cap` at all and whose internal grid is bounded ONLY by
> `_memory_bounded_n_fine`.  And that clamp's cost model prices 4 complex128
> work arrays against a MEASURED 16.1 -- a 4.0x under-price, which is why a
> box the clamp says can afford `n_fine=16384` (it asks for 34.4 GB free)
> ends the run having touched 98.85 GB.**

> **The `wf = 4` no-op claim HOLDS at the acceptance bars** (`wf = 7`
> reproduces every banner digit; EE3 differs by 0.028 points against a
> 0.1-point bar), **but `window_factor` is not a memory lever at the shipped
> `n_fine_cap`**: peak RSS moves 0.9 % across `wf` 2 -> 4 while EE3 moves
> 35.7 points.  And below 4 the penalty is 4x what the library's own docstring
> table records for this class of case.

Three recommendations, in section 8.

---

## 1. BOX, BUILD, METHOD

```
Windows 11 Pro 10.0.26200        AMD Ryzen 9 5950X, 24 logical CPUs
137.4 GB physical RAM            244.3 GB commit limit
python 3.14.6                    numpy 2.4.4    scipy 1.17.1   psutil 7.2.2
lumenairy 5.33.1 (working tree, main @ c8bcbcb)   numba present
```

**Protocol.**  `psutil` peak RSS, peak working set (`peak_wset`, a kernel
high-water counter, not sampled) and peak commit (`peak_pagefile`, which is
trim-immune and is the quantity Windows fails allocations on), sampled at
0.25 s; >= 2 reps on every headline number; a live
big-ndarray census walked from `sys._current_frames()` so it sees the locals
of whatever function is executing AT the peak (a wrapper-based hook cannot --
its callee's frame is already gone by the time the wrapper regains control);
and RSS-step attribution by monkeypatching the library's own call sites.
tracemalloc was used only to confirm that numpy array data IS tracked (it is,
via numpy's tracemalloc domain) before choosing the cheaper census.

**METHOD FINDING, and it changed how the headline was taken.**  An in-process
sampler thread on this workload inflates the measured peak working set by
**2.5x** -- bisected to that single variable in 4.5, with the monkeypatches
ruled out.  So: the headline production run (2.1) carries **no in-process
thread at all** and is sampled from a separate process; the live-array
censuses, which need the thread, are reported as LOGICAL array sizes (which
the artefact cannot touch) and are cross-checked against the thread-free peak.
Anyone repeating this work should measure from outside the process.

**One harness caveat, stated up front.**  The in-flight census walk is
depth- and order-limited, so it under-reports arrays reachable only through
long module->module->cache->entry->list chains; module-level cache footprints
in this document therefore come from a direct enumeration
(`_H_FFT_CACHE`, `_H_CACHE`, `_PYFFTW_PLAN_CACHE` walked by name) and from the
post-run census, not from the in-flight one.  Frame-held arrays -- which is
what sections 2 and 5 turn on -- are reported correctly by both.

**Contention caveat.**  An unrelated 78-91 GB `prof_run.py` (another session)
was resident on the box for part of this audit.  Its effect was not cosmetic
and is itself a finding: the first full-configuration run had its fine grid
silently clamped from 16384 to 8192 by `_memory_bounded_n_fine`, because that
clamp reads `psutil.virtual_memory().available` AT CALL TIME
(`memory.get_ram_budget()` returns exactly that when `set_max_ram` is unset).
The headline run in section 2 was therefore taken with `set_max_ram(105)`
pinned, so the grid choice is deterministic and reproducible rather than a
function of what else the box was doing.  Both runs are reported.

**Configuration measured.**  A single congruence of the 32-order fan, driven
through `propagate_traced_carrier_chain` with exactly the arguments
`fan_multi_121.py` gives it (order `(-4,-2)`, the historical worst case;
`focus_readout = {dx_out: 0.2 um, N_out: 1024, centre_out: <lattice-snapped
chief ray>, n_fine_cap: 16384, window_factor: 4.0, on_replica: 'error'}`;
`final_leg='auto'` -> exact at `na_exit` 0.405).  The 32768-square common grid
enters only the orchestrator's accumulator, which is priced separately and
exactly (17.18 GB complex128), so a single-congruence run reproduces the
per-order cost without paying it.

The probe deliberately has **no** `if __name__ == '__main__':` guard, because
neither `fan_multi_121.py` nor `focus_scan_121.py` has one -- so the shipped
Newton pool rule 1 routes SERIAL in production today, and the probe must too.
Verified directly: `_script_has_main_guard` returns False for all five d121
runners, and the serial-routing `RuntimeWarning` fires in every probe run.

---

## 2. ITEM 1 -- WHERE THE MEMORY GOES

### 2.1 The run

```
order (-4,-2), N=1024, rs=4, NFC=16384, WF=4.0, tile 1024, serial Newton
set_max_ram(105 GB)   (so the fine-grid clamp is deterministic, not a
                       function of what else the box is doing -- see 1.)

exact leg              n_fine = 16384   dx_fine = 0.7622 um   window 12.487 mm
                       na_exit 0.4051 (paraxial, sized the grid)
                       na_exit_measured 0.5321
                       exit_power_above_nyquist 0.0
readout                N_fine = 8192    Bluestein L = 9216
                       readout period 4.7346 mm   tile used 204.8 um (23x margin)
```

| | HEADLINE: no in-process thread, sampled externally | with the 0.25 s in-process sampler (attribution + census run) |
|---|---|---|
| peak RSS | **98.853 GB** (t = 789 s) | 110.546 GB (t = 917 s) |
| peak working set (kernel high-water) | **99.884 GB** | 110.666 GB |
| peak commit | **101.742 GB** | 119.661 GB |
| min free system RAM | **18.4 GB** | 9.599 GB |
| wall | **918 s** | 998 s |

The two differ by 11 % -- the measurement artefact of 4.5, which is 2.5x on
the Stage-B configuration and only 1.12x here (this leg's peak is dominated by
long-lived frame locals, not by rapid allocate/free churn).  **The headline
column is the thread-free one.**  The attribution and census below come from
the instrumented column and are reported as LOGICAL array sizes, which the
artefact cannot touch; they account for 96.6 GB of the thread-free 98.85 GB
(2.3), i.e. the two columns close on each other to ~2 GB.

### 2.2 RSS-step attribution

Deltas are RSS across the call; nesting is the real call nesting.

| call | RSS in (GB) | RSS out (GB) | delta (GB) | wall (s) |
|---|---|---|---|---|
| 5 x coarse `apply_real_lens_traced` | 0.63 -> 5.79 | | +0.68 .. +1.16 each | 3.4 - 5.5 each |
| 5 x `_carrier_step_fast` | | | +0.02 .. +0.10 each | 0.07 - 1.39 |
| **`_fine_trace_group_exit`** | **5.978** | **87.925** | **+81.947** | **877.0** |
| ...`_fourier_upsample_crop` 1024 -> 16384 | 5.978 | 18.025 | +12.048 | 17.7 |
| ...`_exact_sphere_eikonal` | 22.316 | 24.464 | +2.148 | 4.6 |
| ...**`apply_real_lens_traced`** (on 16384^2) | 33.054 | 87.925 | **+54.870** | **794.1** |
| **`carrier_referenced_exact_focus_readout`** | **98.662** | 96.038 | -2.624 | **82.7** |
| ...`_exact_sphere_eikonal` | 34.986 | 43.576 | +8.590 | 4.3 |
| ...`_fourier_upsample_crop` -> 8192 | 69.374 | 78.286 | +8.912 | 8.5 |
| ...`_exact_sphere_eikonal` | 78.286 | 80.434 | +2.148 | 1.0 |
| ...`angular_spectrum_propagate_mft` | 76.676 | 96.038 | +19.362 | 8.4 |
| ......`_bluestein_2d` | 85.400 | 96.004 | +10.604 | 3.3 |

The five COARSE groups together cost 5.8 GB and 22 s.  **The exact final leg
costs 82 GB and 960 s.**  Every number below is about that leg.

### 2.3 What is simultaneously live at the peak plateau

Census taken by the sampler thread at t = 890 s (RSS 104.6 GB), 23 arrays,
**69.26 GB owned**:

| bytes | shape | dtype | held by |
|---|---|---|---|
| 4.295 GB x 6 | (16384, 16384) | complex128 | `_fine_trace_group_exit`: `env_f`, `E_full`, `_ph`, `_cf`, `_rp`, `_xf` |
| 4.295 GB x 5 | (16384,16384) / (2,16384,16384) | complex128 / float64 | `apply_real_lens_traced`: `_unit`, `E_out`, `_coords`, `E_analytic`, `_rd_resid_map` |
| 2.147 GB x 10 | (16384, 16384) | float64 | `apply_real_lens_traced`: `_pip_remap_W`, `_ard`, `_absE`, `_nan_rd`, `_a_rd`, `ard_map`, `amp`, `_mag0`, `Y`, `X` |
| 0.268 GB | (16384, 16384) | bool | `apply_real_lens_traced`: `valid` |

and at the ABSOLUTE peak instant (t = 917 s, RSS 110.546 GB) the innermost
library frame is **`carrier.py:1163 _envelope_amp_radius`**, holding `I`, `X`,
`Y` -- three (16384, 16384) float64 arrays, 6.44 GB -- on top of the 4.295 GB
`E_exit_fine`, **to return one scalar** (the beam radius).

Retained after the run (direct cache enumeration):

| cache | entries | bytes | cap |
|---|---|---|---|
| `fft_infra._PYFFTW_PLAN_CACHE` | fwd+inv @16384 (4 bufs), fwd+inv @9216 (4), fwd @8192 (2) | **24.77 GB** | 8 KEYS x 2 buffers, **no byte cap** |
| `_bluestein._H_FFT_CACHE` | 1 | 1.359 GB (9216^2), **hits = 0** | 16 ENTRIES, **no byte cap** |
| `fft_infra._H_CACHE` | 7 | 1.174 GB | 8 entries, 2.1 GB/entry, 8.6 GB total (**has** byte caps) |

**Accounting closes.**  69.26 (frame-live) + 24.77 (plan buffers) + 1.36
(`_H_FFT_CACHE`) + 1.17 (`_H_CACHE`) = **96.6 GB** of live data against the
thread-free peak RSS of **98.85 GB** -- a 2.3 GB (2.3 %) residual of allocator
retention.  Every gigabyte of this run is named.

### 2.4 Necessary vs accident

**By necessity** (the physics of an exact retrace on a 16384^2 grid):

* `E_full` -- the field being traced (4.295 GB);
* `apply_real_lens_traced`'s `E_out`, `E_analytic`, the OPL/amplitude maps and
  the Newton coordinate stack -- these are the traced element's actual state.
  They can be STREAMED (2.5 below), but at any instant some of them must exist.

**By implementation accident** (dead, or avoidable at identical arithmetic):

| what | bytes at n_fine = 16384 | why it is accident |
|---|---|---|
| `env_f`, `_ph`, `_cf`, `_rp`, `_xf` in `_fine_trace_group_exit` | **21.48 GB** | Each is built, multiplied into `E_full`, and then **never released**.  The code writes `E_full = np.asarray(E_full) * _cf` etc., so both the operand and the product are live, and every local stays bound until the function returns -- i.e. across the entire **794 s** `apply_real_lens_traced` call.  `E_full *= <expr>` with no name (or an explicit `del`) is the same IEEE arithmetic. |
| `X`, `Y` meshgrids in `apply_real_lens_traced` (line 9764), `_exact_sphere_eikonal` (2822), `_envelope_amp_radius` (1162), `_radial_carrier_phase` (743) | **4.29 GB** at the plateau, **6.44 GB** at the absolute peak | `np.meshgrid(y, x)` materialises two full 2-D arrays where `x[None,:]`/`y[:,None]` broadcasting materialises none.  Measured bitwise-identical (2.5). |
| `E_exit_fine` in the chain frame during the readout | **4.295 GB** | The caller's local pins the pre-leg field alive for the whole 82.7 s readout, which is building its own 8192^2 grids at the same time. |
| pyFFTW plan-cache SECOND buffers | **12.4 GB** of the 24.77 GB | The two-buffer ping-pong is a speed optimisation with a documented, byte-identical opt-out (`set_fft_double_buffer(False)`) that the library's own `set_low_memory()` classes as SAFE. |
| `_H_FFT_CACHE` entry | 1.359 GB, **0 hits** | Section 6. |

**At the t = 890 s plateau specifically, 25.8 GB of the 69.26 GB frame-live
total is accident** (21.48 GB of dead factors + 4.29 GB of meshgrid).
`E_exit_fine`'s 4.295 GB and `_envelope_amp_radius`'s 6.44 GB are accident at a
LATER instant (the readout phase, which is where the absolute peak sits), and
12.4 GB of the retained plan buffers is accident throughout.  **None of it
changes a single arithmetic operation.**

### 2.5 The degraded run, and why it matters

The first full-configuration attempt ran while another process held 78 GB.
`_memory_bounded_n_fine` clamped `n_fine` 16384 -> 8192.  Same order, same
arguments:

| | n_fine = 8192 (clamped) | n_fine = 16384 (pinned) |
|---|---|---|
| peak RSS (instrumented column, like for like) | **27.83 GB** | **110.55 GB** |
| chain B wall | **333.1 s** | **993.1 s** |
| readout peak intensity | 5.4687e+03 | 5.3959e+03 |
| readout window power | 6.780494e-08 | 6.779109e-08 |
| dx_fine vs measured-NA Nyquist (1.231 um) | 1.5243 um -- **COARSER, outer NA discarded** | 0.7622 um -- resolved, `exit_power_above_nyquist = 0.0` |

Three things follow.

1. **The 4x memory is buying real physics.**  At `n_fine = 8192` the retrace
   pitch is coarser than the Nyquist pitch of the exit sphere at the
   element's own MEASURED exit NA (0.5321), so outer-NA content is discarded;
   at 16384 it is not.  You cannot buy this memory back by lowering
   `n_fine_cap`.  You have to buy it back from 2.4.
2. **The clamp is contention-sensitive and the choice is invisible.**  The
   grid the exact leg builds -- and hence the resolution of the answer -- is a
   function of free RAM at the instant of the call.  `focus_scan_121.py:34` is
   a blanket `filterwarnings('ignore')`, so on the acceptance runner the
   `RESOLUTION-LIMITED (non-converged)` warning that announces the degradation
   is not printed.  (This is the capstone's own open item 2, now with a
   measured consequence attached: a 1.3 % shift in readout peak intensity
   between two runs of the identical command.)
3. **The clamp's cost model is 4.0x optimistic.**  `_FINE_GRID_WORK_ARRAYS = 4`
   prices four complex128 arrays of the fine grid.  MEASURED at the plateau:
   69.26 GB / 4.295 GB = **16.1** complex128-equivalents live in frames alone,
   21.9 including the plan-cache buffers, 25.7 counting peak RSS.  With
   `frac = 0.5` the shipped model approves `n_fine = 16384` whenever
   ~34.4 GB is available; the run then touches **98.85 GB** -- 2.9x.

### 2.6 Cross-check against the capstone, and an open discrepancy

The capstone measured 75.67 GB (C-1, on-axis order, `N_out` 8192) and 92.14 GB
(C-2, two orders, `N_out` 32768) on the same tree content.  This audit's
single tilted order at a pinned 105 GB budget peaks at 98.85 GB.  The
difference is consistent with the RAM clamp having selected a SMALLER `n_fine`
in the capstone's runs (C-2 carries a 17.18 GB accumulator resident, which
reduces `available` at exactly the moment `_memory_bounded_n_fine` reads it),
which would mean the capstone's 8.12 h 32-order projection rests partly on
RAM-degraded fine grids.  **That is not established here** -- it needs the
capstone's runs re-taken with the fine-grid warning un-silenced and `n_fine`
logged -- but the mechanism is now measured (2.5) and the check is cheap.

The capstone's Stage-B figure, separately, is CONFIRMED by an external control
in section 4.5 -- there is no memory regression in 5.33.

---

## 3. ITEM 2 -- complex64 THROUGH THE CHAIN: THE VERDICT

### 3.1 What was run

`focus_scan_121.py`'s construction and its `metrics()` function reproduced
verbatim (same groups, same source beam, same `dx_out = 0.05 um / N_out =
2048` readout, same encircled-energy definition), stopping at the AT-PLANE
banner line -- the +/-80 um dz scan is post-processing on the readout field and
does not exercise the chain.  Shipping defaults otherwise
(`N=2048, rs=4, NFC=8192, WF=4.0`, `final_leg='auto'` -> exact, 6 post-DOE
groups + the exact leg).  complex64 was requested the only way the library
offers: `fft_infra.set_default_complex_dtype(np.complex64)` plus a complex64
source envelope.  Two reps each.

### 3.2 The numbers

| | complex128 rep 1 | complex128 rep 2 | complex64 rep 1 | complex64 rep 2 |
|---|---|---|---|---|
| FWHM (um) | 3.350000 | 3.350000 | 3.350000 | 3.350000 |
| EE3 (%) | 90.348891 | 90.348891 | **90.348890** | **90.348890** |
| EE6 (%) | 99.699520 | 99.699520 | **99.699520** | **99.699520** |
| EE12 (%) | 99.796669 | 99.796669 | **99.796669** | **99.796669** |
| peak | 5.528622557e+03 | 5.528622557e+03 | **5.528622546e+03** | **5.528622546e+03** |
| P_window / P_in | 0.998014353 | 0.998014353 | **0.998014351** | **0.998014351** |
| halo outside 12 um | 4.766594e-05 | 4.766594e-05 | **4.766594e-05** | **4.766594e-05** |
| halo outside 20 um | 2.175495e-05 | 2.175495e-05 | 2.175495e-05 | 2.175495e-05 |
| halo outside 40 um | 3.905364e-06 | 3.905364e-06 | **3.905364e-06** | **3.905364e-06** |
| **peak RSS (GB)** | **64.447** | **65.147** | **64.555** | **64.512** |
| peak commit (GB) | 66.230 | 66.936 | 66.353 | 66.406 |
| wall (s) | 365 | 367 | 359 | 410 |

Both dtypes reproduce themselves to every digit across reps; the two dtypes
differ from each other only in the 8th significant figure and agree bitwise on
every halo fraction.

Against the campaign's own bars: banner digits IDENTICAL
(`3.350um / 90.3 / 99.7 / 99.8`); EE3 within **1e-6 points** of the 0.1-point
bar; energy `P_window/P_in` differs by **2.0e-9**, four orders under the 4e-5
honesty bar; halo fractions identical to 7 significant figures.

**And the memory saving is 0.0 GB.**  64.555 GB against 64.447 / 65.147 GB --
the complex64 run is 0.2 % ABOVE the first complex128 run and 0.9 % BELOW the
second, i.e. entirely inside the 1.1 % run-to-run spread.

### 3.3 Why: complex64 survives exactly ONE leg

The dtype trace through the complex64 run, taken at the library's own call
sites:

```
_carrier_step_fast          in=complex64    out=complex64     (2048,2048)
_carrier_step_fast          in=complex128   out=complex128    (2048,2048)   <- upcast
_carrier_step_fast x6       in=complex128   out=complex128    (2048,2048)
_fourier_upsample_crop      in=complex128   out=complex128    (8192,8192)   +6.461 GB
_fine_trace_group_exit      in=complex128   out=complex128    (8192,8192)  +31.402 GB
_fourier_upsample_crop      in=complex128   out=complex128    (4096,4096)   +2.376 GB
exact_focus_readout         in=complex128   out=complex128    (2048,2048)  +14.827 GB
```

The upcast is at the FIRST group hand-off, and the sites are identifiable.
Probed directly with a complex64 field on a real design-121 prescription:

| helper | preserves complex64? |
|---|---|
| `_carrier_step_fast` / `propagate_carrier_referenced` | YES (explicit `astype` back) |
| `carrier_referenced_reconstruct` / `..._envelope` | YES (casts the phase to the field dtype) |
| `apply_real_lens_traced` | YES |
| **`_sphere_parab_conversion`** | **NO -- returns complex128** |
| **`_tilt_ramp`** | **NO -- returns complex128** |
| **`_tilt_exactness_phase`** | **NO -- returns complex128** |
| **`_radial_carrier_phase`** | **NO -- returns complex128** |
| **`_fourier_upsample_crop`** | **NO -- `pad = np.zeros(..., dtype=np.complex128)`** (`carrier.py:3206`) |
| **`carrier_referenced_exact_focus_readout`** | **NO -- `E_fine = (...).astype(np.complex128)`** (`carrier.py:4198`, `:4200`) |

`carrier_reference='sphere'` is the shipped default, so
`_sphere_parab_conversion`'s complex128 factor multiplies the envelope at the
first group and NumPy promotes; from there the chain is complex128 forever.
Even had that been fixed, `_fourier_upsample_crop`'s hard complex128 pad and
the readout's explicit `.astype(np.complex128)` sit directly on the
memory-dominant stage, so **no dtype flag reachable from the public API can
halve the exact final leg today.**

### 3.4 VERDICT

**complex64 end-to-end: NOT MEASURED, because it does not happen.**  What was
measured is a chain whose FIRST leg ran at complex64 and whose remaining
seven stages ran at complex128.  For that -- one leg -- the accuracy verdict
is **NONE (bounded at 1.1e-9 relative on the peak, 2.0e-9 on captured energy,
1e-6 EE3 points)**, and the memory verdict is **0.0 GB saved**.

The handoff's ledgered halving is therefore **unclaimed, not disproved**.
Claiming it requires fixing the six dtype leaks above, and the accuracy
question it was flagged for -- "accumulated error through a 6-group chain,
exactly where it could bite" -- has NOT been answered by this run, because
five of the six groups never saw complex64.  Anyone who reads the identical
banner digits above as evidence that a complex64 chain is safe will be
reading a complex128 chain's numbers.

**If mixed precision is pursued, the measured shape of it is:** storage in
complex64 for the ENVELOPE (which is smooth by construction -- that is the
whole point of the carrier reference), with complex128 retained at (a) every
reference-phase build, since `exp(i k S)` with `k S` up to ~1e6 rad has no
float32 representation at all (this is the same trap `mft.py` already
documents for JAX's float32 default -- "losing ~26 dB of phase accuracy"), and
(b) the Bluestein chirp, which the code ALREADY computes in float64 before
casting (`_bluestein.py`: "avoids float32 chirp-phase precision loss at large
indices").  In other words: **c64 storage, c128 phase construction and c128
reduction** -- and the leak list in 3.3 is exactly the list of places where
that boundary is currently drawn on the wrong side.

Before any of that is worth doing, note the ordering: the leg is 16 complex128-
equivalents live (2.3), of which at least 7 are accident (2.4).  Removing the
accident is a 43 % cut with a NONE verdict; halving the dtype is a 50 % cut
that costs a precision campaign.  **Do the accident first.**

---

## 4. ITEM 3 -- window_factor^2, n_fine_cap, AND TILING

### 4.1 Which grid `window_factor` actually sizes

Two different grids, and they behave differently.

**The retrace grid (`_fine_trace_group_exit`, carrier.py:5896-5909):**

```
win        = min(window_factor * w_entrance, N * cur_dx)
n_crop     = 2*round((win/cur_dx)/2)
n_fine_req = 2**ceil(log2(max(win/dx_fine, n_crop)))
n_fine     = min(n_fine_req, n_fine_cap)          <- n_fine_cap DOES bound it
n_fine     = _memory_bounded_n_fine(n_fine, ...)  <- then the RAM clamp
dx_fine    = win / n_fine
```

At design 121's last group (`w_entrance = 3.1315 mm`, `cur_dx = 51.2334 um`,
`dx_fine target = lambda/(3*NA) = 1.0755 um`), the staircase is:

| `wf` | window (mm) | `n_fine_req` | `n_fine` at `NFC=16384` | leg array size |
|---|---|---|---|---|
| 2 | 6.250 | 8192 | 8192 | 1.074 GB |
| 3 | 9.427 | 16384 | 16384 | 4.295 GB |
| **4 (shipped)** | **12.501** | **16384** | **16384** | **4.295 GB** |
| 5 | 15.677 | 16384 | 16384 | 4.295 GB |
| 6 | 18.751 | 32768 | **16384 (capped)** | 4.295 GB |
| 7 | 21.928 | 32768 | **16384 (capped)** | 4.295 GB |

**So on the shipped configuration `window_factor` is not a memory lever at
all** across 3..7: the count cap has already bound.  What `wf` changes above 3
is `dx_fine = win/n_fine`, i.e. it spends the SAME memory on a wider window at
a coarser pitch -- which is the opposite of what a memory knob should do, and
is why the F-D / `on_tilt_exact_grid` guards exist.

**The readout grid (`carrier_referenced_exact_focus_readout`, carrier.py:4169-
4182)** is the one the handoff describes correctly:

```
N_fine = 2**ceil(log2(max(win/dx_fine, n_crop)))
N_fine = _memory_bounded_n_fine(N_fine, ...)      <- RAM clamp ONLY
```

`n_fine_cap` is not a parameter of this function and is not in the
`exact_kw` list the chain forwards to it (`carrier.py:7607-7611`).  Its window
is `window_factor * w_exit`, so its area IS quadratic in `wf` with nothing but
the RAM clamp between it and an OOM.  MEASURED on the production order:
`w_exit = 1.1835 mm`, `wf = 4` -> window 4.7346 mm -> `N_fine = 8192`
(4.295 GB/array); at `wf = 7` the same geometry gives 8.285 mm ->
`N_fine = 16384`, i.e. **4x the readout's memory for the same physics**.  That
is the handoff's 8712^2 regime, and it is still uncapped.

### 4.2 Can the leg be TILED or streamed?

**The readout's `readout_tile` is not the equivalent.**  `readout_tile` is an
OUTPUT-side window: it shrinks `N_out` per congruence (32768 -> 1024) so each
frame is read out on a small tile and accumulated.  MEASURED, it does not
touch the input-side cost: with `tile = 1024` the Bluestein length is still
`L = next_fast_len(N_fine + N_out - 1) = 9216`, set by `N_fine = 8192`.

**The final leg has no input-side equivalent -- but the library already ships
the pattern.**  `apply_real_lens` (the non-traced sibling) has
`sag_chunk_rows`: a row-band path, AUTO-enabled at `N >= 4096`, documented and
tested BYTE-IDENTICAL to the whole-grid path, whose stated purpose is exactly
this ("the full-grid meshgrids (~26 GB at N=32768) AND the full-grid sag
(~43 GB float64 transient) NEVER materialise -- only a (chunk_rows x Nx) band
is live at once", `_lens_real.py:2859-2870`).  The census in 2.3 shows that
15 of the 16 full-grid arrays alive inside `apply_real_lens_traced` on the fine
leg are POINTWISE in (x, y) -- coordinates, amplitude, OPL map, NaN masks, the
remap weights, the ray-density maps.  The only non-pointwise operation on that
path is the element's internal ASM, one global FFT.  So the traced element is
directly amenable to the same treatment, with an in-library precedent for the
byte-identity claim.

`_fourier_upsample_crop` is a global FFT and cannot be row-banded, but it CAN
be done in place (section 5) and it is separable.

### 4.3 Should the internal grid be bounded by the replica-guard-safe window?

**No -- and it is worth saying why, because it is a plausible-looking wrong
answer.**  MEASURED on the production order: the readout's Bluestein period is
4.7346 mm and the tile actually used is 204.8 um -- a **23x margin**.  The
replica guard is nowhere near binding, so sizing the internal window from it
would permit `wf ~ 0.2`.  But `window_factor` sets the CROP that decides how
much of the beam survives into the readout; shrinking it to the
replica-safe bound would simply truncate the beam.  The binding constraint on
`wf` is beam containment, not replicas.  The correct memory levers on this leg
are (a) removing the accidental retention (2.4), (b) row-banding the pointwise
work (4.2), and (c) giving the readout's `N_fine` the `n_fine_cap` it never
got.

### 4.4 The `wf = 4` no-op claim -- VERIFIED, with a caveat below it

Swept on the acceptance configuration (`N=2048, rs=4, NFC=8192, NOUT=2048`,
on-axis, everything else at library defaults), one full chain run per point:

| `wf` | leg `n_crop` -> `n_fine` | readout `n_crop` -> `N_fine` | FWHM (um) | EE3 (%) | EE6 (%) | EE12 (%) | capture | halo > 12 um | peak RSS (GB) | wall (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 378 -> 8192 | 2678 -> 4096 | **5.050** | **54.641** | 78.594 | 81.083 | **0.83120** | **2.037e-02** | 63.778 | 307 |
| 3 | 566 -> 8192 | 3050 -> 4096 | 3.650 | 88.051 | 98.810 | 99.004 | 0.99108 | 1.039e-03 | 63.448 | 302 |
| **4 (shipped)** | 754 -> 8192 | 3092 -> 4096 | **3.350** | **90.349** | 99.700 | 99.797 | 0.99801 | 4.767e-05 | 64.447 / 65.147 | 365 / 367 |
| 7 (library default) | 1320 -> 8192 | 3094 -> 4096 | **3.350** | **90.377** | 99.723 | 99.814 | 0.99815 | 1.198e-05 | 67.988 | 308 |

**The no-op claim holds at the acceptance bars.**  `wf = 7` reproduces `wf = 4`
to every banner digit (`3.350um / 90.3 / 99.7 / 99.8`), and the EE3 difference
is **0.028 points** against the campaign's 0.1-point bar.  So design 121's
`WF = 4.0` is not leaving accuracy on the table at the printed precision.

Three things the sweep adds that the docstring does not say.

* **Below 4 it is not a gentle taper.**  `_fine_trace_group_exit`'s docstring
  records the compound crop as "0.880 at `window_factor=2` (worth 8.6 EE3
  points), 0.987 at 3, and an exact no-op from 4 upward".  MEASURED on this
  chain, `wf = 2` costs **35.7 EE3 points** (54.6 vs 90.3) and 1.70 um of
  FWHM, with capture falling to 0.831 -- 4x the documented penalty.  `wf = 3`
  costs 2.30 EE3 points and 0.30 um of FWHM, where the docstring's 0.987
  factor implies a near-no-op.  Whatever case that table was measured on, it
  is optimistic for design 121.
* **`wf` is not a memory lever here.**  Peak RSS across `wf` 2 -> 4 moves
  63.8 -> 64.4 GB (**0.9 %**) because `n_fine_cap` pins the leg grid at 8192
  for every one of them, and the readout's `N_fine` lands on 4096 at every
  point.  Halving `wf` from 4 to 2 buys 0.9 % of memory and costs 35.7 EE3
  points.  Going the other way, `wf = 7` costs **5.5 %** more memory for
  0.028 EE3 points -- though it does buy a genuine **4x reduction in the halo
  beyond 12 um** (1.198e-05 vs 4.767e-05 of P_in), which matters for a
  stray-light budget even though it is invisible in EE.
* **The readout's uncapped dimension is latent, not realised, at these two
  configurations.**  Its `N_fine` sizing formula is quadratic in `wf` with no
  `n_fine_cap` in the path (4.1), but the power-of-two staircase plus the
  `_avail = N*dx` clamp held it at 4096 across `wf` 2..7 here, and at 8192 on
  the production order at `wf = 4`.  The exposure is real (nothing bounds it
  but the RAM clamp, whose model is 4.0x optimistic -- 2.5) and simply did not
  bite at these two points.  That is an argument for capping it, not for
  assuming it is safe.

### 4.5 THE OBSERVER CHANGES THE OBSERVABLE -- a 2.5x measurement artefact

This started as an apparent 3x memory regression and ended as a methodological
finding that anyone measuring this library's memory needs to know.

The section-3 acceptance harness reproduces `focus_scan_121.py`'s construction
and metrics verbatim and returns the banner to every printed digit and the peak
intensity to ten significant figures -- but it peaked at **63.4-65.1 GB** across
five runs, against the capstone's recorded **20.86 GB**.  The bisect:

| run | peak RSS | peak WSET (kernel high-water) | wall | banner |
|---|---|---|---|---|
| `focus_scan_121.py` UNMODIFIED, sampled from OUTSIDE, rep 1 | **24.569 GB** | -- | 346 s | `3.350 / 90.3 / 99.7 / 99.8` |
| `focus_scan_121.py` UNMODIFIED, sampled from OUTSIDE, rep 2 | **24.506 GB** | **25.252 GB** | 346 s | identical |
| capstone record (5.32.1 / 5.33.0, three days earlier) | 20.86 / 21.37 GB | -- | 346.1 / 346.6 s | identical |
| this audit's harness, in-process sampler @ 0.25 s (5 runs) | 63.358-65.147 GB | 63.361-65.163 GB | 303-367 s | identical |
| ...same, sampled from OUTSIDE as well | 63.591 GB | 63.594 GB | 309 s | identical |
| ...same, every library monkeypatch REMOVED | 63.358 GB | 63.361 GB | 303 s | identical |
| **...same, in-process SAMPLER THREAD removed** | **25.044 GB** | **25.247 GB** | 310 s | identical |

**The single variable is the in-process sampler thread.**  Remove the
monkeypatches: 0.0 % change.  Remove one daemon thread that wakes 4x/second to
call `psutil.Process().memory_info()` and `psutil.virtual_memory()`: the peak
working set falls **63.36 -> 25.25 GB**, landing on the unmodified runner's
25.25 GB to within 0.1 %.

**Consequences, in order of importance.**

1. **There is no memory regression in 5.33.**  The capstone's Stage-B figure is
   confirmed: 24.5-25.2 GB measured externally on the unmodified runner against
   its recorded 20.86 GB (the residual is the external sampler covering the
   whole process tree plus the +/-80 um dz-scan tail the capstone timed
   separately).
2. **Any in-process memory sampler on this workload can inflate peak RSS by
   2.5x.**  The mechanism is not proven here, but the shape is consistent with
   GIL hand-off: numpy releases the GIL for large array operations, and a
   thread that takes it 4x/second delays the main thread's DECREFs -- so
   multi-GB temporaries that would have been freed before the next allocation
   are still live when it happens, and the OS high-water climbs.  The
   library's own hot loops allocate and free 2-4 GB arrays in tight
   succession, which is exactly the regime where that matters.
3. **Measure this library's memory from OUTSIDE the process**, or accept a
   2.5x ceiling.  The capstone's own samplers were in-process
   (`capstone_stageB.py` / `capstone_stageC.py` "adds a memory sampler"); their
   numbers agree with an external measurement at Stage B, so their sampling
   period was evidently long enough not to bite -- but C-1's 75.67 GB and
   C-2's 92.14 GB have not been re-taken externally and inherit the question.
4. **Section 2's headline was re-taken with no in-process thread** (2.1), and
   section 3's complex64 comparison is unaffected because it compares two runs
   of the SAME harness -- its relative result (0.0 GB saved, 0.2 % apart) is
   sound, and its absolute 64 GB is an observer-inflated figure that is quoted
   nowhere else.

---

## 5. ITEM 4 -- IN-PLACE OPPORTUNITIES ON THE CHAIN

The handoff's "~8*N^2 complex128 working set" is an under-count for the exact
leg: MEASURED, the leg holds **16.1 complex128-equivalents** (2.3).  Of those,
these are avoidable, with the accuracy verdict measured rather than argued.

### 5.1 `_fine_trace_group_exit`: five dead full-grid factors -- 21.5 GB

`carrier.py:6015-6038` builds `env_f`, then `_ph`, `_cf`, `_rp`, `_xf` and
folds each into `E_full` with `E_full = np.asarray(E_full) * _x`.  Every one of
those names stays bound until the function returns, i.e. **through the entire
794 s `apply_real_lens_traced` call**.  The census in 2.3 shows all six alive
simultaneously at 4.295 GB each.

* Reduction: **-21.48 GB** at `n_fine = 16384` (five of six arrays).
* Accuracy: **NONE.**  `E_full *= f` performs the identical IEEE operations as
  `E_full = E_full * f`; the change is `del`/in-place, not arithmetic.
* Size: **S** (about ten lines, one function).

### 5.2 Full 2-D meshgrids where broadcasting suffices -- measured bitwise

`np.meshgrid(y, x, indexing='ij')` materialises two full 2-D float64 arrays.
Four sites on this path do it: `_exact_sphere_eikonal` (carrier.py:2822),
`_envelope_amp_radius` (1162), `_radial_carrier_phase` (743),
`apply_real_lens_traced` (`_lens_traced.py:9764`).

Measured at 4096^2, `_exact_sphere_eikonal` shipped vs a broadcast + in-place
rewrite:

```
shipped   (np.meshgrid)      peak 0.671 GB = 5.0 full float64 arrays   0.23 s
broadcast (in-place)         peak 0.134 GB = 1.0                       0.07 s
max|delta| = 0.000e+00       bitwise-equal = True
```

* Reduction: **-8.6 GB per call** at 16384^2; the leg calls it 3 times
  (measured deltas +2.148, +8.590, +2.148 GB).  The `_envelope_amp_radius`
  site is the one holding the ABSOLUTE peak (2.3): -4.3 GB there, bitwise.
* Accuracy: **NONE (bitwise-equal, measured).**
* Size: **S**.

### 5.3 `_fourier_upsample_crop`: 6 full-size arrays -> 2 -- measured

`carrier.py:3196-3220` holds, at `n_fine`: the zero-padded spectrum, the
`ifftshift` copy, the `ifft2` output, the `fftshift` copy, and the `out *
scale` product.  Measured at `n_crop=512 -> n_fine=4096` (one array =
0.268 GB):

| variant | peak above baseline | full-size arrays | max delta vs shipped |
|---|---|---|---|
| shipped | 1.619 GB | 6.0 | -- |
| in-place quadrant swaps + numpy `ifft2` | **0.976 GB** | 3.6 | **0.000e+00 (bitwise)** |
| in-place quadrant swaps + `scipy.fft.ifft2(overwrite_x=True)` | **0.490 GB** | 1.8 | 2.670e-15 |

(For even `n`, `fftshift == ifftshift ==` a quadrant swap, which costs one
quarter-size temporary instead of a full copy.)

* Reduction: **-40 % byte-identical**, **-70 %** at 2.7e-15.  On the real leg
  the measured call delta is +12.048 GB (16384) and +8.912 GB (8192).
* Accuracy: **NONE** for the byte-identical variant; **bounded at 2.7e-15
  max|delta|** for the scipy variant.
* Size: **S**.

### 5.4 pyFFTW plan-cache double buffers -- 24.77 GB retained

`_PYFFTW_PLAN_CACHE` keeps up to 8 KEYS, each holding TWO aligned full-grid
buffers (the ping-pong), with **no byte cap** -- unlike `_H_CACHE`, which has
both a 2.1 GB/entry and an 8.6 GB total cap.  MEASURED retained after one
order: fwd+inv at 16384^2 (17.18 GB), fwd+inv at 9216^2 (5.44 GB), fwd at
8192^2 (2.15 GB).

* Reduction: `set_fft_double_buffer(False)` halves it (**-12.4 GB**);
  `set_fft_plan_cache_size(4)` bounds the key count.  Both are in the
  library's own `set_low_memory()` **safe set** ("all byte-identical to a
  default run -- memory/speed trade only"), the cost being one array copy per
  FFT.  Verified at 4096^2: 1.074 GB -> 0.537 GB per fwd/inv pair.
* Accuracy: **NONE** (documented byte-identical values).
* Size: **S** as a caller-side call; **S/M** to give the plan cache the byte
  cap `_H_CACHE` already has.

### 5.5 The Bluestein 2-D pad -- separable, and it is also faster

`_bluestein_2d` pads BOTH axes to `L = next_fast_len(N_in + N_out - 1)` and
convolves in 2-D, so every working array is `L^2`.  The transform is EXACTLY
separable -- the code already builds the kernel as `h_y[:,None] * h_x[None,:]`
-- so the same sum can be taken as two 1-D chirp-Z passes whose largest array
is `(N_in x L)`.  Measured against the shipped path on a tapered beam:

| N_in / N_out | shipped peak | separable peak | reduction | rel L2 delta | power ratio | time |
|---|---|---|---|---|---|---|
| 2048 / 256 (L=2304) | 0.854 GB | **0.255 GB** | **70 %** | 8.6e-16 | 1.000000000000 | 0.15x |
| 4096 / 1024 (L=5120) | 3.412 GB | **1.343 GB** | **61 %** | 9.1e-16 | 1.000000000000 | 0.42x |

* Reduction: 61-70 % of the transform's peak (the real leg's `_bluestein_2d`
  delta is +10.604 GB), and it eliminates the `L^2` chirp-kernel cache
  entirely -- two length-`L` vectors (0.15 MB) replace a 1.359 GB array.
* Accuracy: **bounded at rel L2 9.1e-16, max|delta|/max|F| 1.2e-15, power
  ratio 1.000000000000** -- round-off only, but **NOT byte-identical** (a
  different association order for the same sum).
* Size: **M**.

### 5.6 Release the pre-leg field before the readout -- 4.295 GB

`carrier.py:7655` calls the readout with `E_exit_fine` while the chain frame
still holds it; the readout immediately builds its own grids.  Handing it
through a one-element container the callee can clear (or restructuring so the
chain drops its reference) frees 4.295 GB for the 82.7 s of the readout.

* Accuracy: **NONE.**  Size: **S**.

### 5.7 The `preserve_input_phase='remap'` sampler -- a row-band candidate

Seen in the census at t = 854 s: `_pip_sample_residual` holds `row` and `col`,
two (16384, 16384) float64 arrays (**4.29 GB**), and then builds
`np.vstack([row.ravel(), col.ravel()])` -- a (2, 2.7e8) float64 (**4.29 GB**
transient) -- to hand `scipy.ndimage.map_coordinates` one call.  The retained
`_pip_res_ri` real/imag pair (2 x 2.147 GB) is genuinely reused and is not the
issue.  `map_coordinates` is trivially separable by rows, and the sibling
`_pip_residual_ri` in the same closure ALREADY row-bands its own exponential
(`_bd = 4194304 // N`) with `np.broadcast_to` instead of materialising
coordinates.  Applying the same idiom one function down would remove ~8.6 GB
of transient at no arithmetic cost.

* Accuracy: **NONE** (`map_coordinates` on a row band gives identical values;
  order-1 sampling is local).  Size: **S/M**.

### 5.8 What is NOT avoidable

`apply_real_lens_traced`'s 43.2 GB is not a temporaries problem -- the element
genuinely needs the coordinate stack, the OPL map, the amplitude map and the
two complex fields at once.  The lever there is streaming (4.2), not `del`.
`fft` `overwrite_x` on the pyFFTW path is already effectively in use (the
plans are in-place, `b, b`); the memory there is the RESIDENT plan buffers
(5.4), not per-call temporaries.

---

## 6. ITEM 5 -- CACHE FOOTPRINTS AND PER-ORDER DUPLICATION

### 6.1 Chain-A and the design-side caches: negligible

```
_chainA_v2_n1024_rs4_<12 hex>.npz     16.23 MB on disk (x3 present, 3 key variants)
_chainA_1024_2000nm_rs2/rs4.npz       15.03 / 15.46 MB   (schema-v1, unreachable)
_chainA_2048_1000nm_rs4.npz           60.38 MB           (schema-v1, unreachable)
_dammann_121_4x8_128.npy               0.26 MB
                                     -------
total on disk                        139.8 MB
chain-A envelope in RAM              (1024,1024) complex128 = 16.78 MB
```

All 32 congruences in `fan_multi_121.py` are handed the SAME `env_doe` object
and `propagate_traced_carrier_chain_multi` does not copy it, so the serial
orchestrator holds ONE copy.  A K-way `congruence_workers` pool pickles it per
worker: 32 x 16.78 MB = 537 MB -- real but two orders below anything else
here.  `groups_k` is `groups` itself when no `doe_order` is set (identity, not
a copy).  **Nothing on the design side is worth memory-mapping.**

### 6.2 What IS duplicated per order

| cache | per-order behaviour | cap |
|---|---|---|
| `_bluestein._H_FFT_CACHE` | keyed on `(alpha_x, alpha_y, N_in, N_out, sign, dtype)`.  `alpha = dx_out/(N_in*dx_in)` and `dx_in = dx_fine` differs per order (C-2 measured per-order readout periods 4734.6..4738.3 um, a 0.08 % spread), so **every order writes a NEW entry and hits none**.  MEASURED: `hits = 0` after a full production order; demonstrated directly with two orders differing by 0.08 % -> 2 entries, 0 hits. | **16 ENTRIES, no byte cap** |
| `fft_infra._PYFFTW_PLAN_CACHE` | keyed on shape; shapes are constant across orders, so this one is SHARED -- but at 8 keys x 2 full-grid buffers with no byte cap (5.4). | 8 KEYS x 2 bufs, **no byte cap** |
| `fft_infra._H_CACHE` | keyed on `(N, dx, lambda, z, ...)`; `dx_fine` differs per order, so new entries per order -- but the 2.1 GB/entry and 8.6 GB total byte caps bound it.  MEASURED 1.174 GB after one order. | 8 entries, **byte-capped** |

**The `_H_FFT_CACHE` is the one that matters, and it is pure cost on a fan.**
Priced at the shipped shapes:

| leg `n_fine` | readout `N_fine` | tile | `L` | one entry | 16 entries |
|---|---|---|---|---|---|
| 16384 | 8192 | 1024 | 9216 | **1.359 GB** | **21.7 GB** |
| 16384 | 16384 (`wf`=7) | 1024 | 17424 | **4.858 GB** | **77.7 GB** |
| 8192 | 8192 | 2048 | 10240 | 1.678 GB | 26.8 GB |

A 32-order fan fills the 16 entries and retains 21.7 GB of chirp kernels it
will never read again (and 77.7 GB at `wf = 7`).  The same module already
knows the fix -- `_H_CACHE` two files away carries exactly the byte caps this
one lacks, with a comment recording the identical lesson ("At N=32768 each H is
16 GB complex128; without this cap, an 8-entry cache can hold up to 128 GB").
**A byte cap here is a strict win with no accuracy consequence, and section
5.5's separable rewrite removes the array from existence.**

### 6.3 Consequence for k-way concurrent orders

Per-order concurrency multiplies the frame-live 69.26 GB, NOT the caches
(module globals are per-process, and a `congruence_workers` pool gets a fresh
copy of each per worker anyway).  So the concurrency ceiling on this box, at
the shipped `n_fine_cap = 16384`, is **k = 1**: one order already peaks at
98.85 GB of a 137.4 GB box.  Every reduction in section 5 raises that ceiling
directly -- removing just 5.1 + 5.2 + 5.4 + 5.6 (42.7 GB, all NONE-verdict)
takes one order to ~56 GB and makes **k = 2** fit on this box with margin.
Nothing else about the parallel path changes that arithmetic: the memory, not
the CPU, is the binding constraint.

---

## 7. ITEM 6 -- THE NEWTON WORKER'S 1.75 GB IMPORT INTERCEPT

Method: a FRESH interpreter per row (which is what a `spawn` worker is),
reading `psutil` `peak_pagefile` (Windows peak commit) and `peak_wset` -- the
same method `FIX_POOL_MEMORY` sec 3.2 used.

| process state | peak commit | peak wset |
|---|---|---|
| bare python | 0.0116 GB | 0.020 GB |
| `import numpy` | 0.8306 GB | 0.034 GB |
| `import numpy, numba` | 0.8594 GB | 0.072 GB |
| `import numpy, scipy.fft` | **1.6263 GB** | 0.063 GB |
| `import numpy, scipy.linalg` | 1.6257 GB | 0.063 GB |
| `import numpy, scipy.interpolate` | 1.6443 GB | 0.086 GB |
| **`import lumenairy.elements._lens_traced`** (the worker's entry module) | **1.6430 GB** | 0.080 GB |

**The intercept decomposes as: numpy 0.83 GB + scipy 0.80 GB + numba 0.03 GB +
lumenairy 0.00 GB.**  151 lumenairy submodules and 664 modules total are
imported, and they cost essentially nothing beyond what numpy and scipy
already reserved.

**Which import pulls scipy, traced with a meta-path finder:**

```
FIRST-SCIPY-IMPORT  lumenairy/elements/rcwa/_core.py:95   from ...backend import (
                    lumenairy/backend/__init__.py:30      from . import scipy as scipy
                    lumenairy/backend/scipy.py:21         import scipy.linalg as _sp_linalg
```

`lumenairy.backend.__init__` eagerly imports its scipy backend shim, which
eagerly imports `scipy.linalg`.  Nothing on the Newton chunk path
(`_newton_invert_chunk` -> the Chebyshev evaluator -> numpy/numba kernels)
uses it.

* **Trimmable: 0.795 GB per worker, 45 % of the 1.75 GB intercept**, by making
  `lumenairy.backend.__init__` lazy (PEP-562 `__getattr__`), a pattern this
  repo already uses and already has CI lessons about.
* Accuracy verdict: **NONE** (import order only).
* **Does it matter at the shipped worker counts?  On design 121, no.**  The
  per-worker model at Stage B's actual shape (32 768 points/chunk, 531^2 fit)
  is 2.00 GB; trimming 0.795 takes it to 1.21 GB, so 8 workers go from 16.0 GB
  to 9.7 GB of commit against a 244 GB limit -- and `FIX_POOL_MEMORY` sec 6
  measured **zero** memory-cap warnings on that run, i.e. the memory arm never
  binds there.  Two corners where it DOES matter: (a) the fit-grid-heavy case
  that fix's own sec 5.4 records (24 CPUs x 1125^2 fit = 2.83 GB/worker, 67.8
  GB against a 56.8 GB budget, clamped 24 -> 20); trimming the intercept takes
  that to 2.04 GB/worker and the clamp stops binding; (b) any future
  `congruence_workers > 1`, where the K chain processes each pay it again.
* Note the commit/wset split: the worker's RESIDENT cost after import is
  0.080 GB.  The 1.75 GB is address-space commit, which is free on Linux
  overcommit and charged in full on Windows -- so this is a
  Windows-and-`congruence_workers` item, not a universal one.
* Size: **M** (an import-graph change with a CI surface).

---

## 8. RANKED TABLE AND THE THREE I WOULD DO FIRST

All footprints MEASURED on the production order (`n_fine = 16384`) unless
noted.  "Size" is implementation size, not risk.

| # | item | measured footprint today | projected reduction | accuracy verdict | size |
|---|---|---|---|---|---|
| 1 | `_fine_trace_group_exit` never releases 5 consumed full-grid phase factors (`env_f`, `_ph`, `_cf`, `_rp`, `_xf`), held across the 794 s element call | **21.48 GB** | **-21.48 GB** | **NONE** (in-place `*=` + `del`; identical IEEE ops) | S |
| 2 | pyFFTW plan cache: 8 keys x 2 full-grid buffers, count-capped only | **24.77 GB retained** | **-12.4 GB** (`set_fft_double_buffer(False)`); more with a byte cap | **NONE** (library's own documented byte-identical "safe set") | S |
| 3 | `apply_real_lens_traced` full-grid working set on the fine leg (5 complex128 + 10 float64 + 1 bool) | **43.2 GB** | **-20 to -30 GB** by row-banding the pointwise stages | **NONE** if banded ops stay pointwise (precedent: `apply_real_lens`'s `sag_chunk_rows`, documented byte-identical) | L |
| 4 | `_fourier_upsample_crop` holds 6 full-size arrays | **+12.05 GB** (16384 call), +8.91 (8192 call) | **-40 %** byte-identical / **-70 %** | **NONE** bitwise (measured); 2.7e-15 for the scipy variant | S |
| 5 | `np.meshgrid` in `_exact_sphere_eikonal`, `_envelope_amp_radius`, `_radial_carrier_phase`, `apply_real_lens_traced` | **5.0 full arrays** where 1.0 suffices; 6.44 GB at the absolute peak instant | **-8.6 GB/call** (3 calls/order) | **NONE (bitwise-equal, measured)** | S |
| 6 | `_bluestein_2d` pads both axes to `L^2` | **+10.60 GB** transient | **-61 to -70 %** of the transform peak | **bounded: rel L2 9.1e-16, power ratio 1.000000000000; NOT byte-identical** | M |
| 7 | `_H_FFT_CACHE`: 16 entries, no byte cap, **0 hits** on a fan | 1.359 GB/order -> **21.7 GB** over 32 orders (77.7 GB at `wf`=7) | **-21.7 GB** | **NONE** (a cache) | S |
| 8 | `E_exit_fine` pinned in the chain frame across the 82.7 s readout | **4.295 GB** | **-4.295 GB** | **NONE** | S |
| 9 | `_FINE_GRID_WORK_ARRAYS = 4` vs a MEASURED 16.1 complex128-equivalents | clamp under-prices **4.0x** | (correctness, not a saving) | **UNSAFE as shipped**: the clamp asks for 34.4 GB free for a run that touches 98.85 GB, and the runner silences the warning | S |
| 10 | `carrier_referenced_exact_focus_readout`'s `N_fine` takes no `n_fine_cap`, only the RAM clamp whose model is 4x optimistic; sizing is quadratic in `wf` | 4.295 GB/array (`N_fine`=8192) on the production order; **latent, did not bite at either configuration measured** | bound it, as the retrace leg already is | **UNSAFE as shipped** (the dimension has no cap; it was not exercised here) | S |
| 11 | Newton worker import intercept | **1.75 GB/worker commit**, of which 0.795 GB is `scipy.linalg` via `lumenairy.backend.__init__` | **-0.795 GB/worker (-45 %)** | **NONE** | M |
| 12 | complex64 through the chain | **0.0 GB saved** (64.56 vs 64.45 GB measured) | -50 % of the leg IF six dtype leaks are fixed | see sec 3.4 -- **UNCLAIMED, and the 6-group accumulation question is still open** | M |
| 13 | chain-A npz + Dammann caches, per-order geometry | 139.8 MB disk / 16.78 MB RAM, shared across all 32 orders | -- | NONE | -- |

### The three I would implement first

**(1) Free the consumed phase factors in `_fine_trace_group_exit` (row 1).**
21.5 GB, one function, ten lines, a NONE verdict, and it is the single largest
recoverable block on the path.  It is also the cleanest: the arrays are
provably dead -- each is multiplied into `E_full` and never read again -- yet
they are held across the longest call in the whole chain.  Pair it with row 8
(release `E_exit_fine`) and row 5 (the meshgrids), both S and both bitwise, and
one order drops from 98.9 GB to roughly 64 GB with no number moving.

**(2) Fix the fine-grid cost model and cap the readout's `N_fine` (rows 9 and
10).**  These are the only two rows with an UNSAFE verdict, and they are the
mechanism behind both the capstone's OOM class and the silent 1.3 %
resolution shift measured in 2.5.  `_FINE_GRID_WORK_ARRAYS` should carry the
measured count (16, not 4), and
`carrier_referenced_exact_focus_readout` should receive the same `n_fine_cap`
the retrace leg already honours -- today the chain forwards eleven keys to it
and that is not one of them.  Note the consequence, which is the point: with a
calibrated model, design 121's shipped `NFC = 16384` does NOT fit a 137 GB box,
which is exactly what the measurement says (18.4 GB free at peak).  The model
being 4x optimistic is the only reason the run completes, and that is not a
safety margin -- it is the absence of one.

**(3) Give `_H_FFT_CACHE` a byte cap and turn off the pyFFTW double buffer on
this path (rows 7 and 2).**  Two one-line-class changes, both NONE, worth
12.4 GB immediately and 21.7 GB across a 32-order fan, and both remove a
retained cost that produces **zero** cache hits on the fan workload
(`hits = 0`, measured).  `_H_CACHE`, in the same subpackage, already carries
exactly the byte caps `_H_FFT_CACHE` and `_PYFFTW_PLAN_CACHE` lack; this is
bringing two caches up to a standard the third already meets.

Rows 3 (row-band the traced element) and 6 (separable Bluestein) are the
larger structural wins -- together another 30-40 GB -- but they are M/L and,
in the Bluestein's case, not byte-identical.  They should follow, not lead.

---

## 9. WHAT THIS AUDIT DID NOT ESTABLISH

1. **The MECHANISM of the 2.5x sampler-thread artefact** (4.5).  The variable
   is isolated (one daemon thread, 4 wakeups/second) and the effect is
   reproducible and workload-dependent -- 2.5x on the Stage-B configuration,
   1.12x on the production order -- but the causal chain (GIL hand-off
   delaying DECREFs of multi-GB temporaries) is a hypothesis consistent with
   the shape of the data, not a proven mechanism.  Nothing in this document
   depends on it: the headline was re-taken thread-free, and section 3's
   verdict is a same-harness relative comparison.
2. **Whether the capstone's C-1 / C-2 runs used a RAM-degraded `n_fine`**
   (2.6).  The mechanism is measured; those specific runs were not re-taken.
3. **complex64 through a genuinely complex64 6-group chain** (3.4).  Not
   possible through the public API today; the accumulated-error question the
   handoff flagged remains open.
4. **The projected reductions are projections.**  Rows 1, 5, 8 are exact
   arithmetic on measured array sizes; rows 2, 4, 6 are measured on a smaller
   grid and scaled; row 3 is an estimate from the pointwise-array inventory,
   not a prototype.  None of them has been implemented, and this audit changed
   no library file.
5. **Single order, single design, single box.**  Everything here is design 121
   at `n_fine_cap = 16384` on one Windows box.  The array-size arithmetic
   scales exactly with `n_fine^2`; the allocator-retention component of peak
   RSS (2.3 GB, 2.3) does not, and is platform-dependent.
