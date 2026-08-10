# AUDIT -- SPEED and PARALLELISM of the traced propagation path

**2026-08-09.  Branch `main`, `c8bcbcb` (v5.33.1).  REPORT ONLY: nothing under
`lumenairy/**` or `validation/**` was modified, no git command was run, no
CHANGELOG entry.  Every probe, profile and memory log lives in the session
scratchpad; the paths are named beside each measurement.**

Workload of record: `validation/repro_traced_carrier_121` -- chain A plus 32
INDEPENDENT DOE orders through `propagate_traced_carrier_chain_multi`, ~911 s
each, run SERIALLY today (`CAPSTONE_D121_2026_08_06.md` sec 6.5: 8.12 h
projected).

---

## 0. VERDICT

> **The 8.12-hour fan is not FFT-bound, not Newton-bound, and not
> `map_coordinates`-bound.  57.8 % of one order's wall is a single Python
> polynomial loop -- `_ResidualEikonal._poly` (`_lens_traced.py:5108-5121`) --
> which recomputes `u ** i` and `v ** j` from scratch for every one of the six
> accumulators of every term, and whose only consumer on that path
> (`.value()`) never reads three of the six.  A power-cached, Hessian-free
> rewrite measured 3.81x on the real shapes and is BIT-IDENTICAL
> (`np.array_equal` True).  That alone is 1.74x on the fan: 8.12 h -> 4.65 h.**
>
> **Order-level parallelism -- the untouched elephant -- is already in the
> library (`congruence_workers`, niche D8) and is blocked by MEMORY, not by
> cores: ONE order at the shipped `n_fine_cap=16384` peaks at 80.6 GB measured,
> against the clamp's own 17.55 GB model.  The clamp therefore approves SIX
> workers where ONE fits, and `fan_multi_121.py` has no `__main__` guard, so
> the knob cannot be used at all today.  Once the grid is halved it works and
> it pays: MEASURED 1.61x on two real orders at k=2, with an acceptance
> printout identical to serial row for row.**
>
> **And `n_fine_cap=8192` -- which this audit ran with the D6 pre-check ARMED,
> and which was NOT refused -- is 3.37x faster and 3.36x lighter at a largest
> measured deviation of 3.0e-5, inside the chain's own 4e-5 energy honesty.**

Two instruments agree on the headline (sec 2): an in-process wall-clock
sampler at 100 Hz and `py-spy 0.4.2 --nonblocking --idle` at 25 Hz, attached to
the same run, put `_poly` at 57.75 % and 59.05 % of the attributed samples.

Compounded, on measurements only: **8.12 h -> 0.89 h** (items #1 + #2 + k=2).

---

## 1. BOX, BUILD, AND THE HONESTY CAVEATS

```
Windows 11 Pro 10.0.26200        AMD Ryzen 9 5950X, 24 logical CPUs
127.9 GB physical RAM            227.5 GB commit limit
python 3.14.6   numpy 2.4.4   scipy 1.17.1   numba 0.65.1   pyfftw 0.15.1
lumenairy 5.33.1 (working tree, clean)
```

Timing protocol: quiet box (checked before each block), >= 3 reps, MINIMUM
reported with the [min-max] spread beside it.

**Three caveats, stated up front rather than in a footnote.**

1. **Run P1's WALL is contaminated; its SHARES and its PEAK RSS are not.**  A
   second heavy python (a helper probe launched by a parallel investigation)
   started 640 s into the 1154 s profile run and drove free RAM to 0.08 GB.
   It was killed.  The measured peak RSS 80.60 GB was reached at t = 296 s,
   i.e. BEFORE the contamination, and the profile SHARES are corroborated by a
   second instrument that sampled the same run.  Where a wall-clock number is
   needed for the projection, the capstone's clean **910.9 s/order** is used,
   not P1's 1154.6 s.  P1's physics output is byte-for-byte the capstone's:
   `(-4,-2)  FWHM 3.800 um  EE3 90.1  EE6 99.9  EE12 100.0  throughput
   0.99331  capture 0.99993` against `CAPSTONE_D121_2026_08_06.md` sec 6.3's
   identical row -- so the run itself is sound.
2. **The in-process sampler under-counts pure-Python phases** (it must take the
   GIL to walk the frame).  That biases AGAINST the finding in sec 2, which is
   a pure-Python loop -- so 57.8 % is a floor, and py-spy, which does not take
   the GIL, reads it higher (59.1 %).
3. **The `_poly` microbenchmark's ABSOLUTE seconds rest on two assumed
   geometry constants** (`scale` = 3.0 mm, freeze radius 3.6 mm, fine pitch
   from `window_factor * w / n_fine`).  Its RATIOS -- which is what the
   recommendation rests on -- do not: they are measured on the shipped class,
   with the shipped term list, at the shipped degree, and are asserted
   bit-identical against the shipped method's own output.

---

## 2. ITEM 2 -- FRESH PROFILE OF ONE REPRESENTATIVE ORDER

Configuration: exactly `fan_multi_121.py`'s own defaults on the full-fan common
grid -- `RN=1024 RS=4 DXO=0.2 um NOUT=32768 TILE=1024 LEG=auto NFC=16384
WF=4.0 OTEG=error`, order `(-4,-2)` (the historical worst case, and the order
the capstone timed at 912.1 s), `NW=1`.  Driver
`scratchpad/traced_speed/prof_run.py` runs `fan_multi_121.py` UNMODIFIED
through `runpy`; profiles `scratchpad/traced_speed/_prof_p1.txt` (internal,
71 694 samples) and `_prof_p1_spy.txt` (py-spy, 82 434 samples).

### 2.1 By phase -- the exact final leg IS the order

| phase | share |
|---|---|
| **EXACT final leg, fine retrace (`_fine_trace_group_exit`)** | **87.30 %** |
| EXACT final leg, readout (`carrier_referenced_exact_focus_readout` -> ASM + Bluestein) | 7.88 % |
| the SIX coarse chain groups + all gap ASM legs | **3.62 %** |
| runner recombination / acceptance metrics | 0.99 % |

Of the 87.30 %, 81.33 % is inside the single `apply_real_lens_traced` call the
retrace makes on the 16384-square fine grid (`carrier.py:7573`).

### 2.2 TOP 10 SITES (self time, innermost Python frame)

| # | % of order | site | what it is |
|---|---|---|---|
| 1 | **15.07** | `_lens_traced.py:_poly:5111` | `P = P + c * u ** i * v ** j` |
| 2 | **10.88** | `_lens_traced.py:_poly:5115` | `Pv` term |
| 3 | **10.68** | `_lens_traced.py:_poly:5113` | `Pu` term |
| 4 | **7.12** | `_lens_traced.py:_poly:5117` | `Puu` term |
| 5 | **6.98** | `_lens_traced.py:_poly:5119` | `Puv` term |
| 6 | **6.85** | `_lens_traced.py:_poly:5121` | `Pvv` term |
| 7 | 5.68 | `scipy/ndimage/_interpolation.py:map_coordinates:474` | all resampling |
| 8 | 2.25 | `numpy/fft/_pocketfft.py:_raw_fft:101` | the RAW `np.fft` calls |
| 9 | 1.98 | `carrier.py:_tilt_exactness_phase:2940` | complex `exp` over n_fine^2 |
| 10 | 1.65 | `carrier.py:_tilt_ramp:4615` | complex `exp` over n_fine^2 |

Rows 1-6 are ONE function.  Summed: **57.75 %**, all of it reached by exactly
one call chain (measured, not inferred):

```
_pip_residual_ri (_lens_traced.py:7480)            57.75 %
  -> _ResidualEikonal.value  (:5163)
     -> _ResidualEikonal._eval (:5136)
        -> _ResidualEikonal._poly (:5108-5121)
```

The next-largest caller of `_poly` on the whole run is 0.04 %.

**py-spy cross-check**, excluding its own sampler/idle frames: `_poly`
**59.05 %**, `map_coordinates` **5.74 %**, everything else 35.22 %.  Two
independent instruments, 1.3 points apart.

### 2.3 By work bucket

| bucket | % |
|---|---|
| residual-eikonal polynomial (`_poly` / `_eval`) | **59.26** |
| `scipy.ndimage.map_coordinates` (every site) | 5.67 |
| `carrier._tilt_exactness_phase` | 3.64 |
| remap residual de-chirp (`exp`/`abs`/`where`, `_pip_residual_ri`) | 3.19 |
| raw `numpy.fft` pocketfft | 2.25 |
| pyFFTW dispatcher (`_fft2`/`_ifft2`, incl. the `copyto` into the workspace) | 2.10 |
| exact-sphere phasor (`_sphere_parab_conversion` + `_exact_sphere_eikonal`) | 1.71 |
| `carrier._tilt_ramp` | 1.65 |
| `carrier._radial_carrier_phase` | 0.98 |

### 2.4 What this REFUTES

* **The capstone's "~23 % in `map_coordinates` resampling"
  (`FIX_D1_POOL_2026_08_06.md` sec 6) does not hold at this configuration.**
  Measured 5.67 % (internal) / 5.74 % (py-spy).  That 23 % was a same-shape
  profile of a CHAIN GROUP; on the real fan order the fine retrace dominates
  and dilutes it.  Sec 4 measures what is left of it.
* **The handoff's stale profile was pointing at the right function for the
  wrong reason.**  `HANDOFF_TRACED_EXACT_2026_08_05.md` sec 5 lists
  `_pip_sample_residual 42.7 %` / `_poly 39.6 %` and says the numbers must be
  re-derived because `_poly` "is gone from the CPU path now" with the spline
  default.  Two corrections: (a) `newton_fit` is `'polynomial'` on today's tree
  (capstone pre-flight (e)), and (b) **this `_poly` is not the Newton fit at
  all** -- it is `_ResidualEikonal._poly`, the residual-eikonal potential of the
  `remap` launch (niche C6), which no `newton_fit` setting touches.  The site
  survived every default flip and has GROWN from 39.6 % to 57.8 %.
* **The Newton inversion is not the cost.**  Measured on this run: the six
  coarse groups invert 65 536 points each and the exact leg's fine retrace
  inverts **9 025**.  `FIX_D1`'s "1.5 % of a group's wall" stands, and a group
  is 0.6 % of an order.

### 2.5 The fix, measured on the shipped class

`scratchpad/traced_speed/probe_poly.py` and `probe_poly2.py`.  The shipped
`_ResidualEikonal` is imported and driven directly: degree 6, its own 27-term
`[(i, d-i) for d in 1..6 for i in 0..d]` list, one 256 x 16384 row band (4.19
Mpt -- the band size `_pip_residual_ri` itself uses, `_lens_traced.py:7477`),
64 bands to the fine grid.  Min of 3, [min-max].

| variant | one band | vs shipped | bit-identical | projected, whole fine grid |
|---|---|---|---|---|
| **SHIPPED `_poly`** (all 6 outputs) | 9.499 s [9.499-10.283] | -- | -- | 607.9 s |
| V1 power-cached + in-place | 3.746 s [3.746-3.883] | **2.54x** | **YES** | 239.7 s |
| V3 repeated-multiply power table | 3.157 s [3.157-3.369] | 3.01x | no (max abs delta 1.07e-14, rel 2.9e-10) | 202.0 s |
| **SHIPPED `.value()`** end to end | 9.970 s [9.970-9.994] | -- | -- | **638.1 s** |
| **V4 = power-cached AND no Hessian** | **2.620 s** [2.620-2.786] | **3.81x** | **YES** | **167.7 s** |

Two independent corroborations that this is the same 57.8 % the profile sees:

* the synthetic projection of the SHIPPED `.value()` over the whole fine grid,
  638.1 s, sits between the profile's 57.75 % of P1's 1154.6 s (667 s) and
  57.75 % of the capstone's clean 910.9 s order (526 s) -- i.e. it reproduces
  the measured absolute cost without being fitted to it;
* the same six leaf lines carry **54.82 %** at `n_fine_cap=8192` (profile
  `_prof_p2b.txt`), where the fine grid has a quarter of the points -- the
  share is a property of the code, not of one grid.

`V1` and `V4` are bit-identical because they issue the SAME `np.power` calls
(one per distinct exponent instead of one per term per accumulator) in the same
operand order, and `+=` rounds identically to `x = x + y`.  `V3` replaces
`np.power` with repeated multiplication and is NOT bit-identical; it is listed
because it is faster still, and its deviation (2.9e-10 relative) is quantified
rather than assumed.

---

## 3. ITEM 1 -- ORDER-LEVEL PARALLELISM

### 3.1 The knob already exists

`propagate_traced_carrier_chain_multi(..., congruence_workers=K)` (niche D8,
`carrier.py:8339-8828`) already does exactly what this item asks for: K
independent chain calls in a `spawn` `ProcessPoolExecutor`, input fields
deduplicated by identity and shipped ONCE through the pool initializer
(`_multi_worker_init`, `carrier.py:8537`), results returned as the readout TILE
(1024^2 = 16 MB) not the common grid, guards and accumulation kept in the
parent in ascending k so the complex sum is **FP-identical to serial**
(`carrier.py:8352-8356`), and per-worker RAM clamped by `_multi_resolve_workers`
(`carrier.py:8568`).

`fan_multi_121.py` never passes it.  So the fan is serial by omission.

### 3.2 The prerequisite: `fan_multi_121.py` has no `__main__` guard

Under `spawn`, every worker re-imports `__main__`; an unguarded module body
therefore runs the WHOLE runner in every child.  The library detects this and
refuses with a message that names the fix (`carrier.py:8801-8813`), and the
Newton pool independently forces `n_workers = 1` for the same reason
(`_spawn_reexecuted_main_script` / `_script_has_main_guard`,
`_lens_traced.py:1417-1493`; the mechanism is `FIX_POOL_MEMORY_2026_08_06.md`
sec 4).  `fan_multi_121.py` has no top-level `if __name__ == '__main__':`, so
`congruence_workers > 1` cannot be used from it as written.

The prerequisite is one of:
* put the runner's body behind a guard, or
* keep `__main__` a thin import-safe shim that `runpy`s the science script
  under another `run_name` -- which is what this audit's own driver
  (`scratchpad/traced_speed/prof_run.py`) does, and it is why the pool paths
  were reachable at all here.

### 3.3 The binding constraint is MEMORY, and the clamp mis-prices it by 4.6x

**Measured, one order, full-fan configuration, `n_fine_cap=16384`:**

```
peak RSS 80.60 GB at t = 296 s      (whole process, chain B, order (-4,-2);
                                     the N_out=32768 accumulator is NOT yet
                                     allocated at that instant, so this is the
                                     per-CONGRUENCE cost -- i.e. what ONE
                                     congruence_workers worker would hold)
```
`scratchpad/traced_speed/_mem_p1.tsv`.  Consistent with the capstone's 92.14 GB
whole-process peak for the two-order run on the same grid (80.6 + the 17.2 GB
common-grid accumulator, which does not fully co-occur with the chain peak).

**What the clamp believes** (`_multi_resolve_workers`, probed directly with the
real arguments -- `shape0=(1024,1024)`, `K=32`, `congruence_worker_min_free_gb`
= 8.0, 119.9 GB free):

| `n_fine_cap` | library per-worker model | requested 2 / 4 / 8 / 32 -> APPROVED |
|---|---|---|
| 16384 | **17.55 GB** | 2 / 4 / 6 / **6** |
| 8192 | 4.66 GB | 2 / 4 / 8 / **23** |

The model is `_MULTI_WORKER_GRID_FACTOR (22) * N_in^2 * 16 B` +
`_FINE_GRID_WORK_ARRAYS (4) * 16 B * n_fine_cap^2` = 0.37 + 17.18 GB.  Against
the measured 80.60 GB that is **4.59x optimistic**, so at the shipped
`n_fine_cap` the clamp would approve **six** workers -- ~484 GB on a 128 GB
box.  This is the same class of failure the clamp's own comment records having
already caused once (`carrier.py:8597-8603`), one layer further down: the fix
priced the fine grid, but priced it at the library's own 4-array model, and
the leg does not hold four arrays.

Why 4 arrays is wrong (source read, `carrier.py` / `mft.py` / `_bluestein.py`,
each item corroborated by the cache census in sec 3.6):

* the leg runs `_fourier_upsample_crop` **twice** -- retrace `carrier.py:6015`,
  readout `carrier.py:4185/4190` -- and each is a 4-array n_fine^2 peak in its
  own right (`carrier.py:3203-3220`);
* the reference-phase locals `_ph`, `_cf`, `_rp`, `_xf`, `env_f` are never
  freed and stay live across `apply_real_lens_traced` and across the transform
  (`carrier.py:6019-6038`, `:4075-4095`) -- **6 x 4.295 GB = 25.8 GB** at
  `n_fine = 16384`;
* the Bluestein zoom pads to `next_fast_len(N_fine + N_out - 1)` and holds
  eight such arrays (`_bluestein.py:215-216`, `:247-286`);
* the pyFFTW plan cache holds **two** aligned workspaces per
  `(direction, shape, dtype, threads)` key under an LRU of 8, and the
  Bluestein chirp cache `_H_FFT_CACHE` is evicted by COUNT (16), not by bytes.
  **Measured: 9.23 GB of library-resident cache after a two-order run at the
  SMALL cap** (sec 3.6), none of it in any cost model.

### 3.4 MEASURED: `congruence_workers=2` on the same two orders

Two orders, `(-4,-2)` and `(-2,0)`, `n_fine_cap=8192`, everything else at
`fan_multi_121.py`'s defaults, run twice back to back on a quiet box through
the same import-safe driver -- once serial, once with `congruence_workers=2`.

| | serial (P3a) | `congruence_workers=2` (P3b) | ratio |
|---|---|---|---|
| chain B wall (both orders) | **582 s** | **361 s** | **1.61x** |
| whole-run wall | 603.6 s | 383.8 s | **1.57x** |
| per-order wall (serial) | 291.7 s, 290.4 s (spread 0.4 %) | -- | -- |
| peak total RSS | 41.47 GB (one process, incl. the 17.2 GB accumulator) | **48.29 GB** = parent 0.62 GB + 2 children, **largest child 24.97 GB** | -- |
| **per-WORKER peak RSS** | (23.996 GB, from the single-order run) | **24.97 GB** | agrees |

**Parallel efficiency 81 % at k = 2.**  It is not 2.00x because the two workers
share memory bandwidth and each independently starts 8 FFTW threads on a
24-thread box (`fft_infra` is not in `_WORKER_STATE_MODULES`, so nothing
coordinates them).

**The answer does not move.**  The acceptance printout is identical row for
row:

```
frame table: (m,n)  design%  FIELD%  ratio  exact x,y (um)  throughput capture  FWHM   EE3   EE6   EE12
serial  (-4,-2)  50.091  50.078  0.9998  (-1920.4, -960.2)  0.99331  0.99993  3.800  90.1  99.9  100.0
        (-2,+0)  49.909  49.922  1.0002  ( -960.0,   +0.0)  0.99375  0.99998  3.400  90.6  99.9  100.0
cw=2    (-4,-2)  50.091  50.078  0.9998  (-1920.4, -960.2)  0.99331  0.99993  3.800  90.1  99.9  100.0
        (-2,+0)  49.909  49.922  1.0002  ( -960.0,   +0.0)  0.99375  0.99998  3.400  90.6  99.9  100.0
```

plus the same `max abs(share/design - 1) = 0.00025`, the same throughput
0.99353 (per-order spread 4.39e-04), the same capture 0.99996, the same
`design->measured correlation 1.00000`, and the same `VERDICT: PASS`.  That is
the library's documented FP-identity (`carrier.py:8352-8356`) observed rather
than assumed.

### 3.5 How many orders fit, and what the fan would cost

| configuration | per-order peak RSS | k on a 128 GB box (8 GB reserve) | 32-order fan |
|---|---|---|---|
| shipped `n_fine_cap=16384`, serial | **80.60 GB** (measured) | **1** | 8.12 h |
| `n_fine_cap=16384`, clamp's own belief | 17.55 GB (model) | 6 -- **would need ~484 GB** | -- |
| `n_fine_cap=8192`, serial | **24.0 GB** (measured) | **4** | 2.40 h |
| `n_fine_cap=8192`, `congruence_workers=2` | 24.97 GB/worker | 2 measured | **1.49 h** (2.40 / 1.61) |
| `n_fine_cap=8192`, k=4, at the measured 81 % efficiency | ~25 GB/worker | 4 | ~0.9 h (**projected, not measured**) |

At the shipped `n_fine_cap` order-level parallelism is **not memory-feasible on
this box at all**: k = 1.  It becomes feasible through item 6 (sec 7) or
through the memory hygiene of sec 8.3 -- which is the whole reason those two
items rank where they do.

### 3.6 CACHE CENSUS -- measured, after a serial two-order run at `n_fine_cap=8192`

Emitted by the driver from the parent process at the end of run P3a:

```
_bluestein._H_FFT_CACHE : 2 entries, 0.78 GB   (maxsize 16, evicted by COUNT)
    key (4.224209674578248e-05, 4.224209674578248e-05, 4096, 4096, 1024, 1024, 1, 'complex128')
    key (4.2208845952639814e-05, 4.2208845952639814e-05, 4096, 4096, 1024, 1024, 1, 'complex128')
fft_infra._PYFFTW_PLAN_CACHE : 7 entries, 6.12 GB of aligned workspaces
    ('fwd', (1024,1024)) 0.03 GB  calls 30      ('inv', (1024,1024)) 0.03 GB  calls 30
    ('fwd', (8192,8192)) 2.00 GB  calls  4      ('inv', (8192,8192)) 2.00 GB  calls  4
    ('fwd', (4096,4096)) 0.50 GB  calls  2
    ('fwd', (5120,5120)) 0.78 GB  calls  4      ('inv', (5120,5120)) 0.78 GB  calls  2
fft_infra._H_CACHE (ASM H) : 8 entries, 2.33 GB
                              (per-entry cap 2.0 GB, so a 16384^2 H at 4.0 GiB
                               is NEVER stored and is rebuilt every order)
```

Three things this settles, none of which was previously counted anywhere:

1. **`_H_FFT_CACHE` misses on every order, confirmed.**  Two orders produced
   TWO entries whose keys differ only in the leading float
   (`alpha = dx_out / (N_fine * dx_fine)`, re-derived per order from a beam
   radius re-measured per order): `4.2242e-05` vs `4.2209e-05`.  The chirp
   kernel is mathematically the same object to 8 significant figures.  Over 32
   orders that is 32 extra full-size FFTs and, because eviction is by COUNT,
   **16 retained entries** -- 6.2 GB at this cap, and ~4x that at
   `n_fine_cap=16384`.
2. **The pyFFTW plan cache alone holds 6.12 GB** of aligned workspaces at the
   SMALL cap, with double buffering on; the four fine-leg keys are the whole of
   it.  Nothing prices this.
3. **The ASM `H` cache is live but structurally excluded at the big cap**
   (`_H_CACHE_MAX_BYTES_PER_ENTRY = 2 GB`, `fft_infra.py:1237`; a 16384^2 `H`
   is 4.0 GiB).  It holds 2.33 GB here and contributes 0 at 16384.

**9.23 GB of library-resident cache after two orders at the small cap** is the
measured floor of what a k-way scheme pays per worker on top of its own working
set, and it is why the clamp's 17.55 GB model cannot be repaired by a constant.

### 3.7 Distribution across the Tailscale mesh

Roster (from the `mesh-run` skill; RAM figures are measured, not nameplate).
"Orders that fit" uses THIS audit's measured per-order peaks (80.60 GB at
`n_fine_cap=16384`, 24.97 GB per worker at 8192) with the skill's rule that
shard capacity is RAM-bound, not core-bound:

| box | logical CPUs | RAM | orders at NFC 16384 (80.6 GB) | orders at NFC 8192 (25.0 GB) |
|---|---|---|---|---|
| tesla-ryzen (A, hub) | 24 | 127 GB | 1 | **4** |
| austinoffice-2 (B) | 20 | 127 GB | 1 | **4** |
| athena (C, FLAKY) | 24 | **34 GB** | **0** | **1** |
| maxarch | 16 | **30 GB** | **0** | **1** |
| aj-asus (laptop, sleeps) | -- | unprofiled | -- | -- |

i.e. **2 concurrent orders across the mesh at the shipped cap, 10 at 8192.**

**A per-order shard split is a natural fit** -- the orders are independent, the
job list is 32 long, and the shard pattern (`SHARD/NSHARDS` stride, per-shard
result file, `RESUME=1`, glob-and-dedup combine) needs no barrier until the
recombination.  What it needs beyond the pattern:

1. **Ship the inputs, not the cache.**  Each shard needs the `.zmx`
   (`tx4designstudy121/20260707 dll Tx02-MSOP16.zmx`), the design-study runner
   `run_poc_119_120_v518.py` (only for its Schott Sellmeier block) and
   `tx_design_study_sim`, plus the Dammann cell cache
   `_dammann_121_4x8_128.npy` (262 kB).  These live under
   `Free_Space_Optics/Reverse_Symmetric_ASM`, NOT under the mirrored
   `Metasurface_QWP` tree -- **so the mirror does not carry them today** and
   that is the one real porting item.
2. **Ship chain A's output, not chain A.**  `env_doe` is 1024^2 complex128 =
   **16.8 MB**, and chain A costs 10 s; either is cheap, but shipping the array
   also removes any risk of two boxes building a different chain A.  The
   schema-2 cache key already covers the configuration
   (`_chainA_v2_n1024_rs4_<digest>.npz`, 16.2 MB).
3. **Return only the tile.**  Per-order result = the 1024^2 readout tile
   (16.8 MB) + the per-order scalars.  32 orders = 537 MB total across the
   mesh -- minutes of Resilio, not hours.  The 32768^2 common grid (17.2 GB) is
   built ONCE, by the combining box, and never crosses the network.
4. **Per-box durations.**  A shard's cost is `orders_on_that_box x
   per-order-wall`.  Only tesla-ryzen has a measured per-order wall (910.9 s at
   NFC 16384, 270-292 s at 8192); austinoffice-2 (Zen 3, 10C/20T) should be
   within ~20 %, but **it has never been measured and the split must not assume
   it** -- run one order per box first and weight the chunks by the measured
   rate, which is what the skill's throughput-weighted split is for.  athena is
   FLAKY by the skill's own note and RAM-bound at 34 GB, so it should get the
   SMALLEST contiguous range and be allowed to drop.
5. **Do NOT pin BLAS/FFT threads to 1 here.**  The skill's default
   (`OMP_NUM_THREADS=1`, one shard per core) is for many small jobs; these are
   32 huge ones, and this audit measured 81 % parallel efficiency at k=2 with
   each order keeping the library's own 8 FFTW threads.  Note `fft_infra` is
   NOT in `_WORKER_STATE_MODULES` (`carrier.py:8370-8375`), so FFT thread
   settings set in a parent do not cross into congruence workers -- a per-shard
   env var is the only lever that works on every box.
6. **Per-shard RAM guard armed on athena and maxarch**, per the skill: a single
   order at the shipped cap is 3x either box's total RAM, so an accidental
   `NFC=16384` shard there is an OOM-reboot, not a slow run.

**Verdict on the mesh:** at `n_fine_cap=16384` only the two 127 GB boxes can
hold ONE order each, so the mesh buys **2x** (8.12 h -> ~4.1 h) and
athena/maxarch cannot participate at all.  At `n_fine_cap=8192` the mesh holds
**10 concurrent orders**, and with the sec-10 item #1 fix on top the fan is a
**~10-minute** job.  The mesh is therefore the LAST thing to reach for, not the
first: items #1 and #2 of sec 9 are worth 5.7x on a single box before any
network is involved, and they are what make the small boxes usable at all.

---

## 4. ITEM 3 -- `map_coordinates`

Measured share on the real order: **5.67 %** (internal) / **5.74 %** (py-spy),
NOT the ~23 % the capstone recorded at a chain-group shape.

The shipped OPL upsample (`_lens_traced.py:9712-9760`) builds
`np.indices((N,N))`, forms a `(2, N, N)` float64 coordinate stack, runs
`map_coordinates(order=3, prefilter=True)` for the OPL and a SECOND
`map_coordinates(order=1)` for the NaN mask.  Benchmarked at the real shapes
(`scratchpad/traced_speed/probe_mapcoords.py`, min of 5 [min-max]):

| variant | N=1024, sub=4 (no NaN) | N=1024, sub=4 (12 % NaN) | N=4096, sub=16 | bit-identical |
|---|---|---|---|---|
| **A shipped** | 92.9 ms [92.9-107.4] | 96.2 ms [96.2-101.0] | 1554.6 ms | -- |
| B coordinate stack cached across calls | 84.5 ms (**1.10x**) | 84.6 ms (**1.14x**) | 1304.3 ms (**1.19x**) | **YES** |
| B' + spline prefilter hoisted | 83.3 ms (1.12x) | 82.4 ms (1.17x) | 1335.9 ms (1.16x) | no |
| **D skip the NaN pass when the coarse OPL has no NaN** | 53.6 ms (**1.73x**) | n/a | n/a | **YES when applicable** |
| E float32 coordinate stack | 90.0 ms (1.03x) | 90.4 ms (1.06x) | 1406.9 ms (1.10x) | no (0.000e+00 here, not guaranteed) |

Reading:

* **D is the only large one and it is free when it applies.**  When
  `opl_coarse` carries no NaN, `nan_full` is identically zero and
  `np.where(nan_full > 0.5, np.nan, opl_map)` is the identity -- so guarding
  the second `map_coordinates` on `np.isnan(opl_coarse).any()` is bit-identical
  BY CONSTRUCTION, and worth 1.73-1.83x of the upsample.  Whether it applies on
  design 121 is a property of the ray-fit hull mask and is NOT established here.
* **B is bit-identical and unconditional but small** (1.10-1.19x of 5.67 % =
  0.6-0.9 % of the order).
* E buys almost nothing and is not exact-preserving; **reject**.

**Bottom line for item 3: <= 1 % of the order's wall for the bit-identical
options at this configuration, ~2 % if D applies.  It is not where the time
is.**  Every honest win here is far inside the chain's own 4e-5 energy
honesty.

---

## 5. ITEM 4 -- FFT STRATEGY

`scratchpad/traced_speed/probe_fft.py`, min of 3.

```
PYFFTW_AVAILABLE True   USE_PYFFTW True   FFTW_MIN_SIZE 256
FFTW_THREADS 8 of 24 cpus          SCIPY_FFT_WORKERS -1
planner ('FFTW_ESTIMATE',)   auto_promote False
plan cache size 8   double buffer True
backend for a 1024^2 complex128: pyfftw
```

| shape | pyFFTW cold (plan+exec) | pyFFTW warm | plan overhead | numpy.fft | scipy.fft(-1) | warm vs numpy / scipy |
|---|---|---|---|---|---|---|
| 1024^2 | 0.329 s | 0.008 s | 0.320 s (97.4 % of the first call; includes the lazy pyFFTW import) | 0.029 s | 0.019 s | 3.50x / 2.27x |
| 2048^2 | 0.050 s | 0.038 s | 0.013 s (25.2 %) | 0.146 s | 0.092 s | 3.90x / 2.45x |
| 4096^2 | 0.169 s | 0.142 s | 0.027 s (15.9 %) | 0.613 s | 0.380 s | 4.32x / 2.67x |
| 8192^2 | 0.714 s | 0.539 s | 0.175 s (24.5 %) | 2.714 s | 2.007 s | 5.04x / 3.72x |

**Answers to the item as posed.**

* **Is pyFFTW / plan reuse active on the chain's ASM legs?  YES.**  The gap
  legs go `_carrier_step_fast` -> `_envelope_tf_step` ->
  `_exact_envelope_tf_step` -> `_ifft2(_fft2(E) * H)` (`carrier.py:1051`) at
  1024^2 complex128, which clears `FFTW_MIN_SIZE` and lands on pyFFTW.  The
  plan cache is process-global, keyed `(direction, shape, dtype, threads)`,
  strict LRU of 8 (`fft_infra.py:613-615`, `:1118`, `:1189`), so every leg
  after the first is a hit.
* **Wisdom caching?  NONE, anywhere.**  `export_wisdom` / `import_wisdom`
  appear only in a test fixture; nothing is persisted to disk.  Every fresh
  process re-plans.  Under the shipped `FFTW_ESTIMATE` that costs the table's
  "plan overhead" column ONCE per shape per process -- 0.013-0.175 s at these
  sizes, i.e. **< 0.1 % of an order**.  Wisdom would matter only if the
  planner were promoted to `MEASURE`; the library records that promotion as
  worth 2.04x at 1024^2 and 4.55x at 4096^2 (`fft_infra.py:662-665`) but ships
  it OFF and it is **not** bit-reproducible across processes.
* **Plan overhead share of the workload: negligible.**  Total FFT time on the
  order is 2.25 % (raw `np.fft`) + 2.10 % (pyFFTW dispatcher, including the
  full-size `np.copyto` into the aligned workspace) = **4.35 %**.
* **The one real FFT finding is that a hot path bypasses all of the above.**
  `carrier._fourier_upsample_crop` calls **raw `np.fft.fft2` / `ifft2`**
  (`carrier.py:3203`, `:3214`) at the fine-grid size, so it gets
  single-threaded pocketfft rather than the 8-thread pyFFTW the rest of the
  library uses.  It runs TWICE per exact leg (retrace `carrier.py:5964/6015`,
  readout `carrier.py:4185/4190`).  Measured cost on the order: the pocketfft
  bucket is **2.25 % = ~26 s**, and the 8192^2 row says pyFFTW is 5.04x faster
  at the neighbouring size -- so routing those four transforms through
  `_fft2`/`_ifft2` recovers ~**2 % of the order**.
  *Accuracy:* NOT bit-identical (different FFT implementation), bounded at FFT
  round-off, ~1e-16 relative -- eleven orders inside the chain's own 4e-5
  energy honesty.  It would, however, break any byte-identity contract pinned
  on this leg, which is why it is ranked below the free wins.

**Also worth recording (memory, not speed):** with `double_buffer=True` the
plan cache holds two aligned full-size workspaces per key.  MEASURED after a
two-order run at `n_fine_cap=8192` (sec 3.6): **7 entries, 6.12 GB**, of which
the fine leg's four keys are 6.06 GB.  Shape scaling puts that at ~24 GB at
`n_fine_cap=16384`.  No cost model in the library counts it.
`set_fft_double_buffer(False)` halves it for one copy per FFT.

---

## 6. ITEM 5 -- NEWTON POOL INITIALIZER

`scratchpad/traced_speed/probe_pool_dispatch.py`.

### 6.1 On the workload of record the pool NEVER dispatches

```
_POOL_MIN_PIXELS      (cold) = 200 000
_POOL_MIN_PIXELS_WARM (warm) =   8 000
_POOL_PROMOTE_MIN_SECONDS    = 0.35     _POOL_PROMOTE_MIN_SAMPLES = 2

  fan_multi_121 at RN=1024 / RS=4:  65 536 Newton pts per coarse group
                                    -> below the 200k COLD bar
  the EXACT leg's fine retrace:      9 025 Newton pts   (measured, this run)
                                    -> below it by 22x
```

Both tiers then depend on the cost gate, and the shipped default path
(polynomial + numba) measures 0.048 s per group against the 0.35 s bar
(`FIX_D1_POOL_2026_08_06.md` sec 3), so it refuses.  **Measured share of a
Newton dispatch in today's design-121 wall: zero, at any `NW`.**  (Additionally
`fan_multi_121.py` is unguarded, so `_newton_resolve_workers` forces
`n_workers = 1` regardless.)

### 6.2 The pickle share, measured

Payload = `_spline_data`, which carries the five ray-fit grids:

| fit grid | payload | `pickle.dumps` | `pickle.loads` |
|---|---|---|---|
| 387^2 | 5.99 MB | 3.59 ms [3.59-3.90] | 1.33 ms [1.33-1.72] |
| 465^2 | 8.65 MB | 5.05 ms [5.05-6.66] | 1.85 ms [1.85-2.21] |
| **531^2** (design 121's largest) | **11.28 MB** | **6.27 ms** | 2.24 ms |

Real pool round-trip, `spawn`, warm pool, chunks of 65 536/`n_cpu` points:

| workers | bare round-trip | with the REAL payload | **payload share** |
|---|---|---|---|
| 4 | 0.7 ms [0.7-0.9] | 86.7 ms [86.7-88.6] | **99.2 %** |
| 8 | 1.5 ms [1.5-2.0] | **173.1 ms** [173.1-183.0] | **99.2 %** |

That reproduces `FIX_D1`'s ~0.22 s/dispatch constant and **identifies it: 99.2 %
of it is re-pickling `_spline_data` once per chunk.**  Note the payload is
11.3 MB, not the "~1.9 MB of grids" quoted in
`FIX_POOL_REBUILD_2026_08_08.md` sec 3.

### 6.3 Projected win, and whether it moves the cost gate

Shipping `_spline_data` once per worker (pool initializer, as
`_multi_worker_init` already does for the congruence pool) removes the 99.2 %:
**0.173 s -> ~0.002 s at 8 workers.**

* **It DOES change the cost-gate arithmetic.**  Break-even is
  `t * (1 - 1/n) > dispatch_cost`; at n=8 that moves from **t > 0.25 s** to
  **t > 0.0023 s**, i.e. ~100x.  `_POOL_PROMOTE_MIN_SECONDS = 0.35` would go
  from a 1.4x margin over break-even to a 150x one, and the default polynomial
  path's 0.048 s step would clear break-even by 21x.  The constant should be
  re-derived, not kept.
* **It does NOT move design 121.**  The Newton inversion is 1.5 % of a coarse
  group's wall, and the six coarse groups are 3.62 % of an order, so a perfect
  8x on the Newton step is `0.015 * (7/8) * 3.62 %` = **0.05 % of the order**.
  For a workload whose Newton lattices are 65 536 and 9 025 points there is
  nothing here.

**Recommendation: implement it for the LIBRARY (it is a real 100x on a real
constant, and it unblocks pooling for callers whose Newton work is large), but
do NOT count it in any design-121 speed plan.**

---

## 7. ITEM 6 -- `n_fine_cap = 8192` AS AN OPT-IN

The item asks what the D6 pre-check's refusal protects on design 121
specifically.  It was run both ways on ONE order, everything else held at
`fan_multi_121.py`'s own defaults.

### 7.1 The refusal does not happen on this configuration

**Run P2a: `NFC=8192` with `on_tilt_exact_grid='error'` -- the shipped fan
default, guard fully ARMED -- COMPLETED.**  No `on_tilt_exact_grid` message of
any disposition was emitted, and the acceptance printed `VERDICT: PASS`.

So on `fan_multi_121.py`'s configuration (`RN=1024 RS=4 DXO=0.2 um TILE=1024
WF=4.0 LEG=auto`, order `(-4,-2)`) the pre-check **accepts** `n_fine_cap=8192`.
The handoff's "D6's paraxial pre-check refuses 8192" is not reproducible here;
it was recorded against a different runner and a different shape
(`HANDOFF_TRACED_EXACT_2026_08_05.md` sec 4 quotes 84 s vs 312 s, an order of
magnitude below this leg's cost, i.e. `focus_scan_121.py`'s geometry, not the
fan's).  **The claim should be re-scoped to the configuration it was measured
on, and it should not be used to refuse `8192` on the fan.**

Run P2b repeated it with `on_tilt_exact_grid='warn'` and the profiler on;
identical metrics, so the guard's disposition changes nothing here either.

### 7.2 Head to head, same order, same everything else

| | `n_fine_cap=16384` (shipped) | `n_fine_cap=8192` | ratio |
|---|---|---|---|
| chain-B wall, order `(-4,-2)` | **910.9 s** (capstone sec 6.5, clean box) | **270 s** [270-291] | **3.37x faster** |
| peak RSS DURING chain B | **80.60 GB** | **23.996 GB** | **3.36x smaller** |
| peak RSS whole run (incl. the 32768^2 accumulator) | (92.14 GB, capstone, 2 orders) | 38.89 GB | -- |
| **FWHM** | 3.800 um | **3.800 um** | identical |
| **EE3** | 90.1 % | **90.1 %** | identical |
| **EE6 (halo)** | 99.9 % | **99.9 %** | identical |
| **EE12 (halo)** | 100.0 % | **100.0 %** | identical |
| chain throughput `power_exit/power_in` | 0.99331 | **0.99331** | identical |
| readout capture `power_out/power_exit` | 0.99993 | **0.99993** | identical |
| centroid vs predicted chief ray | 0.286 um | **0.286 um** | identical |
| cell power vs library `power_out` | 4.441e-16 | **4.441e-16** | identical |
| library chief ray vs exact skew trace | 0.000 um | **0.000 um** | identical |
| acceptance `max abs(share/design - 1)` | 0.00000 | **0.00000** | identical |
| acceptance verdict | PASS | **PASS** | -- |

"Identical" means identical to the acceptance banner's own printed precision:
FWHM to 1 nm, encircled energy to 0.1 point, throughput/capture to 1e-5,
centroid to 1 nm.  Both runs cleared every one of the five acceptance checks.
The two `NFC=8192` runs differ from each other by 270 vs 291 s (7.8 %), which
is this box's own run-to-run spread with the sampler on; the 3.37x is far
outside it.

### 7.3 Verdict

**The audit recommends the `n_fine_cap=8192` path, with evidence.**  On the fan
configuration it is 3.37x faster and 3.36x lighter at metrics that are
identical to the oracle-grade bars the acceptance itself uses, with the D6
pre-check left ARMED.  Projected on the fan alone: **8.12 h -> 2.4 h**, and the
per-order footprint drops from "one order per 128 GB box" to four -- which is
what makes sec 3's order-level parallelism reachable at all.

### 7.4 A SECOND order, against the capstone's own 16384 run

Run P3a (sec 3.4) put BOTH capstone C-2 orders through at `n_fine_cap=8192` on
the same full-fan common grid the capstone used.  Its frame table can therefore
be compared line for line with `CAPSTONE_D121_2026_08_06.md` sec 6.3, which is
the same two orders at `n_fine_cap=16384`:

| order | quantity | capstone, NFC 16384 | here, NFC 8192 | delta |
|---|---|---|---|---|
| (-4,-2) | FIELD % | 50.077 | 50.078 | 1e-5 |
| (-4,-2) | throughput | 0.99331 | 0.99331 | **0** |
| (-4,-2) | capture | 0.99993 | 0.99993 | **0** |
| (-4,-2) | FWHM / EE3 / EE6 / EE12 | 3.800 / 90.1 / 99.9 / 100.0 | 3.800 / 90.1 / 99.9 / 100.0 | **identical** |
| (-2,+0) | FIELD % | 49.923 | 49.922 | 1e-5 |
| (-2,+0) | throughput | 0.99378 | 0.99375 | **3.0e-5** |
| (-2,+0) | capture | 0.99998 | 0.99998 | **0** |
| (-2,+0) | FWHM / EE3 / EE6 / EE12 | 3.400 / 90.6 / 99.9 / 100.0 | 3.400 / 90.6 / 99.9 / 100.0 | **identical** |

**Largest deviation anywhere: 3.0e-5 in throughput** -- inside the chain's own
**4e-5** energy honesty, and 670x inside the acceptance's tightest bar
(`max abs(share/design - 1) < 0.02`, which reads 0.00025 on both).  Spot
quality (FWHM, EE3, EE6, EE12) does not move at all, on either order.

So the accuracy risk of the `8192` opt-in is not "unknown": it is **bounded and
measured at 3.0e-5**, on two orders including the historical worst case.

Two limits stated rather than buried:

* **Two orders, not thirty-two.**  `(-4,-2)` and `(-2,0)` are the capstone's
  own subset (worst case and mid order); the on-axis order and the other 29
  were not re-run at 8192 here.
* **This is a GRID-CONVERGENCE claim, and the repo's open item on exactly that
  is still open** (`HANDOFF_TRACED_EXACT_2026_08_05.md` sec 5 item 2: nothing
  establishes the chain grid is sufficient).  What is shown here is that 8192
  and 16384 AGREE to the acceptance's precision -- which is the standard
  convergence argument, and is the same argument the library's own
  `NFC 12288 vs 16384: EE3 65.26 vs 65.26` row makes one step further up.

---

## 8. ITEM 7 -- WHAT ELSE THE PROFILE SURFACED

### 8.1 `value()` builds a Hessian it never reads (folded into sec 2's fix)

`_eval` (`_lens_traced.py:5125-5160`) returns `(a, ax, ay)`.  Outside the
radial freeze, `a = P + (r - r1) * b` with `b = gx*ux + gy*uy` -- so the VALUE
needs `P`, `gx`, `gy`.  The three Hessian accumulators `Puu`/`Puv`/`Pvv` feed
`ex_x`/`ex_y` only, which are the GRADIENT outputs `ax`/`ay`.  `value()`
(`:5162`) takes `[0]`.  Half of `_poly`'s arithmetic is therefore discarded on
the path that costs 57.8 % of the order.

### 8.2 Order-INDEPENDENT quantities recomputed per order

On the D9 chief-ray-centred path the retrace's reference build is centred at
`(0, 0)`, so several n_fine^2 arrays depend only on `(n_fine, dx_fine, lambda,
R)` and not on the order's tilt:

| quantity | site | keyed on |
|---|---|---|
| `_radial_carrier_phase` | `carrier.py:6019` | `(n_fine, dx_fine, lambda, R_in)` |
| `_sphere_parab_conversion` | `carrier.py:6024` | same |
| `_exact_sphere_eikonal` (readout, fine) | `carrier.py:4196` | takes no `centre` at all |
| ASM transfer function `H` | `mft.py:370-408` | key has no tilt and no `centre_out` |
| Bluestein `H_FFT` | `_bluestein.py:270-285` | key has no tilt and no `centre_out` |

**But the time in them is small**: `_radial_carrier_phase` 0.98 %,
exact-sphere phasor 1.71 %, `_asm_H_from_kz` 0.20 % -- and the whole readout
phase is 7.88 %.  So cross-order caching is worth **~3 % of the order at most**,
not the double-digit number the structure suggests.  (`_tilt_ramp` 1.65 % and
`_tilt_exactness_phase` 3.64 % ARE order-dependent and cannot be shared.)

One free algebraic identity is worth naming anyway: `_radial_carrier_phase(...,
+1)` builds `exp(+i k r^2/2R)` and `_sphere_parab_conversion(..., +1)` builds
`exp(+i k (S - r^2/2R))`; their product is exactly `exp(+i k S)`.  The leg
allocates two n_fine^2 complex128 arrays, evaluates two 268 M-point `exp`s and
does two full-grid multiplies (`carrier.py:6022`, `:6027`) to form one
exact-sphere phasor.  Collapsing them saves ~1.7 % of time and 4.3 GB of peak.

### 8.3 The memory hygiene that gates sec 3

Independent of speed, and the reason 80.6 GB is 4.6x the model: the
reference-phase locals in `_fine_trace_group_exit` (`_ph`, `_cf`, `_rp`, `_xf`,
`env_f` -- `carrier.py:6019-6038`) and in the readout
(`carrier.py:4075-4095`) are never released and stay live across
`apply_real_lens_traced` (`carrier.py:6106`) and across the ASM+Bluestein
transform (`carrier.py:4248`).  At `n_fine=16384` that is **6 x 4.295 GB =
25.8 GB of dead weight**, four-sixths of which is already folded into
`E_full`.  `del` at the point of last use is a mechanical change with no
numerical content, and it is what makes k-way order parallelism affordable.

### 8.4 Guard surface

`fan_multi_121.py` does NOT blanket-silence warnings (unlike
`focus_scan_121.py:34`), and the run surfaced them: 3.0 % / 36.8 % / 21.6 %
Newton non-convergence on different groups, an `NA_exit = 0.3566`
exit-wavefront under-sampling, and a chief ray 1.079 beam radii off axis at the
fine retrace.  None is a speed finding; all of them stay true of any faster
variant, which is the point of listing them next to a speed number.

---

## 9. RANKED TABLE

Ranked by expected wall-clock win on the workload of record (32 orders,
910.9 s each, 8.12 h serial).  "measured today-cost" is this audit's own
measurement unless a source is named.  Sizes: **S** = a contained edit inside
one function; **M** = a new cache / plumbing across two or three call sites;
**L** = a structural change.

| # | item | measured today-cost | projected win | accuracy risk | size |
|---|---|---|---|---|---|
| **1** | `_ResidualEikonal._poly` recomputes `u**i`/`v**j` per term AND `value()` builds a Hessian it never reads (`_lens_traced.py:5097-5163`) | **57.75 %** of an order (py-spy cross-check 59.05 %); 638 s of a 911 s order | **3.81x** on the site -> **42.6 % of the order** -> per-order 911 -> 523 s, fan **8.12 h -> 4.65 h** (1.74x) | **NONE** -- `np.array_equal` True against the shipped method, same `np.power` calls, same operand order | **S** |
| **2** | `n_fine_cap` 16384 -> 8192 (opt-in; the D6 pre-check does NOT refuse it on the fan) | fine leg at 16384: 910.9 s / 80.6 GB per order | **3.37x** wall and **3.36x** RSS -> fan **8.12 h -> 2.40 h**, and 1 -> 4 orders per 128 GB box | **bounded, MEASURED 3.0e-5** (largest deviation, in throughput, against the capstone's 16384 run on TWO orders; FWHM/EE3/EE6/EE12 identical) -- inside the chain's own 4e-5 energy honesty | **S** |
| **3** | order-level parallelism (`congruence_workers`, niche D8) never used by the fan runner | 32 orders serial; per-order peak **80.6 GB** at NFC 16384 vs the clamp's **17.55 GB** model (**4.59x optimistic**) | **MEASURED 1.61x at k=2** (81 % efficiency, per-worker peak 24.97 GB at NFC 8192); k=4 fits on this box at NFC 8192 | **NONE, OBSERVED** -- the k=2 acceptance printout is identical to serial row for row, including `max abs(share/design-1)=0.00025` and both frame rows | **S** (knob + a `__main__` guard) + **M** (fix the clamp's per-worker model) |
| **4** | the exact leg never frees its reference-phase locals (`_ph`, `_cf`, `_rp`, `_xf`, `env_f`; `carrier.py:6019-6038`, `4075-4095`) | **25.8 GB** of the 80.6 GB peak is dead weight at `n_fine=16384` | no wall-clock win by itself; it is what buys the k in #3 | **NONE** -- `del` at last use has no numerical content | **S** |
| **5** | `carrier._fourier_upsample_crop` calls RAW `np.fft` (`carrier.py:3203`, `:3214`), bypassing the pyFFTW dispatcher, twice per exact leg | pocketfft bucket **2.25 %** = ~26 s/order | pyFFTW warm is **5.04x** numpy at 8192^2 -> ~**2 % of the order** | **bounded**: FFT round-off, ~1e-16 relative; NOT bit-identical, and it would break any byte-identity contract pinned on this leg | **S** |
| **6** | the OPL upsample runs a second full-grid `map_coordinates` for the NaN mask even when the coarse OPL has no NaN (`_lens_traced.py:9748-9750`) | `map_coordinates` total **5.67 %** of the order | **1.73-1.83x** on the upsample when applicable -> up to ~2 % of the order | **NONE** -- with no NaN, `nan_full == 0` and the `np.where` is the identity, so the guard is bit-identical by construction | **S** |
| **7** | the `(2, N, N)` coordinate stack is rebuilt on every upsample call | inside the same 5.67 % | **1.10-1.19x** on the upsample -> **0.6-0.9 %** of the order | **NONE** (`np.array_equal` True) | **S** |
| **8** | `_radial_carrier_phase * _sphere_parab_conversion` is algebraically `exp(+i k S)` but is built as two n_fine^2 phasors and two full-grid multiplies (`carrier.py:6019-6027`) | 0.98 % + 0.72 % of the order, and **8.6 GB** of peak | ~**1.7 %** of the order + 4.3 GB | **bounded** ~1e-16 relative (it changes the multiply order); not bit-identical | **M** |
| **9** | order-INDEPENDENT n_fine^2 quantities rebuilt per order (ASM `H` -- never cached at 16384 because a 4.0 GiB entry exceeds `_H_CACHE_MAX_BYTES_PER_ENTRY` = 2 GB, `fft_infra.py:1237`; exact-sphere eikonal; Bluestein `H_FFT`, whose key carries a float `alpha` re-measured per order) | ASM `H` 0.20 %, exact-sphere 1.71 %, whole readout phase 7.88 %.  **`_H_FFT_CACHE` MEASURED at 1 entry per order** (2 orders -> 2 keys differing only in `alpha`'s 5th digit) | **<= 3 %** of the order in TIME -- much smaller than the structure suggests.  The MEMORY side is the real one: eviction is by COUNT (16), not bytes | **NONE** if keyed exactly; **bounded** if `alpha` is rounded to make the keys hit | **M** |
| **10** | Newton pool re-pickles `_spline_data` once per CHUNK (`_lens_traced.py:9359-9361`) | **0 dispatches** on this workload; the constant itself is **173.1 ms at 8 workers, 99.2 % of it pickle** | break-even 0.25 s -> **0.0023 s** (a real 100x on a real constant) but **0.05 %** of a design-121 order | **NONE** (payload identity is unchanged) | **M** |
| -- | pyFFTW planner `ESTIMATE` -> `MEASURE`, wisdom persistence | plan overhead 0.013-0.175 s per shape per process, **< 0.1 %** of an order | not worth it here; and `MEASURE` is not bit-reproducible across processes (`fft_infra.py:645-649`) | bounded, non-reproducible | **M** |
| -- | pyFFTW `double_buffer` (two aligned workspaces per key, LRU 8) | **6.12 GB MEASURED** at NFC 8192 (sec 3.6); ~24 GB at 16384 by shape scaling | memory only; halves with `set_fft_double_buffer(False)` at one copy per FFT | **NONE** | **S** |

**Compounding** (the two free ones plus the opt-in, all on the same order):

```
  today                                    910.9 s/order   8.12 h fan
  + #1  _poly (bit-identical)              522.9 s/order   4.65 h   1.74x
  + #2  n_fine_cap 8192 (measured equal)   270   s/order   2.40 h   3.37x
  #1 and #2 together                       ~161  s/order   1.43 h   5.66x
       (the _poly share holds at the smaller grid: MEASURED 54.82 % at
        n_fine_cap=8192, against 57.75 % at 16384)
  + #3  k=2 (MEASURED 1.61x)                               0.89 h   9.1x
  + #3  k=4 on this box (projected at the
        measured 81 % efficiency)                          ~0.5 h   ~16x
```

The k=2 arm is not an extrapolation: 582 s -> 361 s of chain B was measured on
two real orders, with an acceptance printout identical to serial (sec 3.4).

---

## 10. THE THREE I WOULD IMPLEMENT FIRST

### 1. Power-cache `_ResidualEikonal._poly` and give `value()` a Hessian-free path

`lumenairy/elements/_lens_traced.py:5097-5163`.  Hoist `u ** i` / `v ** j` into
per-exponent tables (the SAME `np.power` calls, so the bits do not move),
accumulate with `+=`, and let `_eval` ask for only the outputs its caller
needs -- `value()` needs `P`, `gx`, `gy`; the three Hessian accumulators feed
`ax`/`ay` alone.

*Why first:* it is the largest single item by a factor of ten, it is
**bit-identical** (asserted, not argued: `np.array_equal` True against the
shipped method on the real term list at the real degree), it is confined to one
class, and it needs no new configuration surface.  Measured **3.81x** on
`.value()` at the real shapes; **1.74x end to end on the fan**.

*Fail-before / pass-after:* re-run this audit's profile and require the `_poly`
leaf group to drop from ~58 % to < 25 %, plus an existing byte-identity niche
(C1 / C6 / C8) green.

### 2. Free the exact leg's reference-phase locals, then re-price the congruence-worker clamp, then turn `congruence_workers` on

`carrier.py:6019-6038` and `:4075-4095` (`del` at last use), then
`_multi_resolve_workers`' per-worker model (`carrier.py:8596-8607`), then a
`__main__` guard in `fan_multi_121.py` plus a `CW` env knob.

*Why second:* it is the only item that scales with the number of orders rather
than the cost of one, and it is currently blocked by three independent things
none of which is physics -- 25.8 GB of unreleased temporaries, a clamp that is
4.59x optimistic (it would approve six workers where one fits), and a missing
five-character guard in the runner.  The parallelism itself is already written,
tested, and FP-identical to serial.

*Fail-before / pass-after:* assert the clamp's model against a measured
per-order peak (this audit's 80.6 / 24.0 GB are the two calibration points),
and require the k-way fan to reproduce the serial acceptance banner exactly --
which the library's ascending-k accumulation guarantees.

### 3. Adopt `n_fine_cap=8192` on the fan, and re-scope the refusal claim

`validation/repro_traced_carrier_121/fan_multi_121.py:311` (`NFC` default) plus
the sentence in `HANDOFF_TRACED_EXACT_2026_08_05.md` sec 4.

*Why third and not first:* it is the biggest single number (3.37x wall, 3.36x
RSS) and it passed every acceptance check with the D6 pre-check ARMED -- but it
is a DISCRETISATION change, so unlike #1 it can only ever be "equal to the
precision we measured at", and it was measured on one order.  Doing #1 first
means the grid question is decided on physics rather than under time pressure.

*Fail-before / pass-after:* run `(0,0)`, `(-2,0)` and `(-4,-2)` at both caps and
require the frame table to agree row for row; that is ~25 min of box time at
8192 and ~46 min at 16384.

---

## 11. WHAT THIS AUDIT DOES NOT CLOSE

1. **The full 32-order fan has still never been run end to end** -- this audit
   measured one order at 16384, one order twice at 8192, and the same two
   orders twice more (serial and k=2) at 8192.  Every projection is per-order
   arithmetic, exactly as the capstone's was.
2. **`n_fine_cap=8192` is established at TWO of 32 orders** -- the capstone's
   own subset, including the historical worst case.  Sec 7.3, 7.4.
3. **The `_poly` microbenchmark's absolute seconds use assumed geometry
   constants** (sec 1 caveat 3).  Its ratios and its bit-identity do not, and
   the ratio is what the recommendation uses.
4. **P1's wall is contaminated** by a concurrent process from t = 640 s (sec 1
   caveat 1).  Its shares and its pre-contamination peak RSS are sound; its
   wall is not used anywhere in this document.
5. **Whether the NaN-pass guard (#6) applies on design 121 was not measured** --
   it depends on whether the ray-fit hull leaves NaN in `opl_coarse`, which
   this audit did not instrument.
6. **No library edit was made and no test was written.**  Every item above is a
   measurement plus a proposal; the fail-before/pass-after for the top three is
   named but not executed.
7. **The k=4 projection is a projection.**  k=2 was measured (1.61x, 81 %); the
   k=4 arithmetic assumes the same efficiency, and on a 24-thread box running
   four workers that each start 8 FFTW threads it may not hold.  One run
   settles it.
8. **The cache census was taken in the PARENT only.**  Under
   `congruence_workers` the fine-leg caches live in the workers, so the k=2
   census (sec 3.4's run) reads near-empty; the numbers in sec 3.6 are from the
   SERIAL two-order run, which is the right process to read them in.  A
   per-worker census would need the census code inside `_multi_worker_run`.
9. **No cross-order `_H_FFT_CACHE` key-rounding was attempted.**  The miss is
   confirmed (1 entry per order, keys differing in `alpha`'s 5th significant
   digit); whether rounding `alpha` to a tolerance is SAFE -- i.e. whether two
   orders' chirp kernels really are interchangeable at that tolerance -- is a
   physics question this audit did not answer.

---

## APPENDIX -- MEASUREMENT ARTIFACTS

All under the session scratchpad
`.../372a2d1f-acbe-4b57-a148-eeae3fe1d729/scratchpad/traced_speed/`.  Nothing
in the repo was written except this document.

| file | what |
|---|---|
| `prof_run.py` | the driver: runs `fan_multi_121.py` UNMODIFIED via `runpy` under an import-safe `__main__`, with a 1 Hz per-process RSS sampler, a 100 Hz wall-clock stack sampler, an optional `congruence_workers` injection (`CW`) and an end-of-run cache census |
| `anaprof.py` | folds either profile format into phase / leaf / bucket tables |
| `probe_poly.py`, `probe_poly2.py` | `_ResidualEikonal._poly` variants + bit-identity assertions (sec 2.5) |
| `probe_mapcoords.py` | OPL-upsample variants (sec 4) |
| `probe_fft.py` | backend, plan reuse, plan overhead, pyFFTW vs numpy vs scipy (sec 5) |
| `probe_pool_dispatch.py` | Newton-pool census + dispatch cost decomposition (sec 6) |
| `_out_p1.txt`, `_prof_p1.txt`, `_prof_p1_spy.txt`, `_mem_p1.tsv` | P1: one order, `n_fine_cap=16384` |
| `_out_p2.txt`, `_prof_p2b.txt`, `_mem_p2a.tsv`, `_mem_p2b.tsv` | P2a/P2b: one order, `n_fine_cap=8192`, guard armed / downgraded |
| `_out_p3.txt`, `_mem_p3a.tsv`, `_mem_p3b.tsv` | P3a/P3b: two orders, serial vs `congruence_workers=2` |
| `_probes.txt`, `_probe_fft.txt` | probe transcripts |

Run index, with the numbers each one produced:

| run | configuration | wall | peak RSS | what it settled |
|---|---|---|---|---|
| P1 | 1 order `(-4,-2)`, NFC 16384, NW 1 | 1154.6 s (CONTAMINATED after t=640 s) | **80.60 GB** at t=296 s (clean) | the profile (sec 2); the per-order memory (sec 3.3) |
| P2a | 1 order, NFC 8192, `on_tilt_exact_grid='error'` | chain B **270 s**, wall 293.1 s | 23.996 GB during chain B | the D6 pre-check does NOT refuse (sec 7.1); the 3.37x (sec 7.2) |
| P2b | same, guard `'warn'`, profiler on | chain B 291 s, wall 314.4 s | 39.56 GB | `_poly` share holds at 54.82 % on the smaller grid |
| P3a | 2 orders, NFC 8192, serial | chain B **582 s** (291.7 + 290.4) | 41.47 GB | the k=1 baseline; the cache census (sec 3.6); the 2nd order for item 6 (sec 7.4) |
| P3b | same 2 orders, `congruence_workers=2` | chain B **361 s**, wall 383.8 s | 48.29 GB total, **24.97 GB per worker** | **1.61x at k=2, acceptance identical to serial** (sec 3.4) |
