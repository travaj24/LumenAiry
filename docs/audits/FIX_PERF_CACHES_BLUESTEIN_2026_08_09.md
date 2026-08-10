# FIX -- audit items #4, #6, #7 from AUDIT_TRACED_MEMORY_2026_08_09

**2026-08-09.  Branch `perf/traced-hotpath`, on top of `c8bcbcb` (v5.33.1).
No git command was run beyond `status` / `diff` / `log`; `CHANGELOG.md` was not
touched.  Every probe and every recorded output lives in the session
scratchpad (`perf/`), inventoried in section 7.**

Implements three of the audit's rows:

| this doc | audit rows | what |
|---|---|---|
| **#4** | 9 and 10 -- the audit's **only two UNSAFE rows** | price the fine grid at the MEASURED census (`_FINE_GRID_WORK_ARRAYS` 4 -> 16); give the exact readout's `N_fine` the `n_fine_cap` semantics it never had |
| **#6** | 7 and 2 | byte-cap `_bluestein._H_FFT_CACHE`; byte-cap the pyFFTW plan cache's ping-pong |
| **#7** | 6 | the separable Bluestein readout, default ON, shipped 2-D path one flag away |

Files changed: `lumenairy/propagators/carrier.py`,
`lumenairy/propagators/_bluestein.py`, `lumenairy/propagators/fft_infra.py`,
`lumenairy/propagators/mft.py`, `lumenairy/memory.py`, and one pinned
enumeration in `tests/unit/test_niche_c1_consolidation.py`.
`lumenairy/elements/_lens_traced.py` belongs to a concurrent agent and was not
touched by this work -- but it IS modified in the working tree, which is a
measurement confound and is called out wherever it matters (5.2).

---

## 0. VERDICT

> **All three items land.  On the design-121 acceptance configuration every
> banner digit is IDENTICAL to the pre-change reference
> (`3.350000 / 90.348891 / 99.699520 / 99.796669 / 5.528622557e+03`), captured
> energy is identical to every printed digit (`0.998014353`), and the readout
> field differs by `rel L2 = 4.59e-16` with `power ratio = 1.000000000000` --
> against a 4e-5 honesty bar, i.e. four orders of margin on a difference that
> is round-off.  With item #7's flag OFF (the fail-before switch) the field is
> BYTE-IDENTICAL to the pre-change reference, which is the proof that items #4
> and #6 move no bits at all and that #7 is the only thing that does.**

> **Measured reductions.**  On the design-121 production order's shapes:
> retained pyFFTW plan buffers **24.76 -> 10.74 GB (-14.03 GB)**; the
> `_bluestein_2d` step's own peak **13.32 -> 3.72 GB (-72.1 %)** at **1.74x**
> the speed; the chirp-kernel cache **1.359 GB -> 0.000 GB per order**, and
> across a 32-order fan **21.7 GB -> 0.00 GB** on the shipped route with a hard
> **8.59 GB** ceiling for any consumer still on the 2-D one (it was unbounded:
> 21.7 GB at `wf = 4`, 77.7 GB at `wf = 7`).  **Total -35.7 GB on the
> production order / fan.**

> **Two behaviour changes that are the POINT of item #4, and that a production
> operator must know about.**  (a) With the clamp priced at the measured 16
> arrays, `_multi_resolve_workers` approves **1** congruence worker at
> `n_fine_cap = 16384` where it used to approve **6** -- and the measured
> per-order peak is 98.85 GB, so six of them was ~484 GB on a 128 GB box.
> (b) By the same arithmetic, `_memory_bounded_n_fine` now needs **137.4 GB of
> budget** to approve `n_fine = 16384` (it used to need 34.4 GB), so on a
> 137.4 GB box design 121's shipped `NFC = 16384` will be CLAMPED to 8192 and
> announce `RESOLUTION-LIMITED (non-converged)` unless the operator raises the
> budget deliberately.  That is exactly the audit's finding -- "the model being
> optimistic was the only reason the run completed; that is the absence of a
> safety margin, not the presence of one" -- but it changes what the shipped
> runner does, and `focus_scan_121.py:34`'s blanket `filterwarnings('ignore')`
> means the announcement is invisible there.  Remedies in 2.4.

---

## 1. THE PARTIAL DIFF, ADJUDICATED

A prior attempt was killed mid-task and left `carrier.py` with 45 uncommitted
added lines in two hunks.  **Both were KEPT and completed** rather than
reverted; the decision, hunk by hunk:

| hunk | content | verdict | why |
|---|---|---|---|
| 1 | `_FINE_GRID_WORK_ARRAYS = 4 -> 16` plus a ~45-line comment carrying the audit's live-array census | **KEEP, one correction** | Checked line by line against `AUDIT_TRACED_MEMORY_2026_08_09` sec 2.3 / 2.5 / row 9 and `AUDIT_TRACED_SPEED_2026_08_09` sec 3.3.  The census rows (6 + 5 c128 + 10 f64 + 1 bool = 69.26 GB / 23 arrays), the 16.1 ratio, the 34.4 GB approval floor, the 98.85 GB outcome and the "SIX workers ~484 GB" cross-reference all reproduce.  ONE arithmetic slip: the comment attributed the audit's `25.7` complex128-equivalents to "the thread-free peak RSS of 98.85 GB", but `98.85 / 4.295 = 23.0`; `25.7` is against the **110.55 GB instrumented** peak the census itself was taken inside (audit sec 4.5's observer artefact).  Corrected in place, with the distinction spelled out. |
| 2 | `n_fine_cap` / `on_n_fine_cap` added to `carrier_referenced_exact_focus_readout`'s SIGNATURE | **KEEP, complete it** | Correct names, correct defaults (`None` / `'warn'`), and unambiguous: they are exactly what audit row 10 prescribes.  But they were **dead** -- no entry validation, no docstring, no use in the body, no forwarding from the chain.  Verified dead before completing: `perf/out_fail_before_4.txt` records `'n_fine_cap' in readout signature: True` alongside `'n_fine_cap' forwarded: False` and the chain's eleven-key list. |

Nothing was ambiguous enough to warrant `git checkout --`; both hunks were
verifiable against the audit text, and hunk 1 is a large hand-transcribed
census that would have been wasteful to re-derive.  The correction in hunk 1
is the only pre-existing content that changed meaning.

---

## 2. ITEM #4 -- THE TWO UNSAFE ROWS

### 2.1 (a) `_FINE_GRID_WORK_ARRAYS` 4 -> 16

`carrier.py:3700-3800`.  The constant now carries the audit's frame-live
census -- 69.26 GB across 23 live full-grid arrays at `n_fine = 16384`, where
one complex128 grid is 4.295 GB, i.e. **16.1 complex128-equivalents** -- and
the comment records the method (census walked from `sys._current_frames()`,
measured from OUTSIDE the process) so the next person to re-measure does not
repeat the 2.5x sampler-thread artefact.

16 is the FRAME-LIVE count, deliberately not the 21.9 that includes the
process-global pyFFTW plan buffers: those are shared across legs, so charging
them per fine grid would double-count when two grids of different size are
sized in one process.

**FAIL-BEFORE (`perf/probe_fail_before_4.py` -> `out_fail_before_4.txt`).**
`_multi_resolve_workers` probed with the audit's own arguments
(`shape0 = (1024,1024)`, `K = 32`, `min_free_gb = 8.0`, 119.9 GB free):

| `n_work` | `n_fine_cap` | library per-worker model | requested 2 / 4 / 8 / 32 -> APPROVED |
|---|---|---|---|
| **4 (shipped 5.33.1)** | 16384 | 17.55 GB | 2 / 4 / **6** / **6** |
| 4 | 8192 | 4.66 GB | 2 / 4 / 8 / 23 |
| **16 (this fix)** | 16384 | 69.09 GB | **1 / 1 / 1 / 1** |
| 16 | 8192 | 17.55 GB | 2 / 4 / 6 / 6 |

Against a MEASURED per-order peak of 98.85 GB, the shipped model approved
**six** workers -- the clamp approving 6 where 1 fits, which is the fail-before
this row exists for.  It now approves 1.

### 2.2 (b) the readout's `N_fine` gets `n_fine_cap`

`carrier.py`, `carrier_referenced_exact_focus_readout`.  Two new keyword-only
parameters, applied in the RE-TRACE LEG'S OWN ORDER -- count cap first, RAM
clamp second, exactly `_fine_trace_group_exit`'s `min(n_fine_req, n_fine_cap)`
then `_memory_bounded_n_fine`:

* `n_fine_cap : int, optional` -- default `None` = **no count cap**, i.e.
  byte-identical to every pre-v5.33.2 direct call.  A direct caller who passes
  nothing keeps exactly the behaviour they had.
* `on_n_fine_cap : {'warn', 'error', 'ignore'}, default `'warn'`` -- mirrors
  `on_ram_cap`, including raising `MemoryError` on `'error'` (the same
  exception class, since it is the same class of fault), and validated AT
  ENTRY next to `on_replica` so a typo cannot ride through the fine trace.

`propagate_traced_carrier_chain` forwards `focus_readout['n_fine_cap']`
(default 16384) **explicitly**, not through the `if kk in fr` comprehension, so
the default travels too and both grids cap at the same number.  Eleven keys
reached the readout before; `on_n_fine_cap` joins them and
`_OUTPUT_GRID_PASSTHROUGH` (hence `_FOCUS_READOUT_KEYS`, hence the multi entry
point).  The paraxial `_par_kw` whitelist drops it, as it already drops the
other exact-only keys.

The guard message names, in house style: the capped size, the un-degraded
requirement AND its pitch, the resulting `dx_fine` measured against the exit
sphere's Nyquist pitch (with the discarded-NA number when it is coarser), the
memory both grids cost **at the measured 16-array count**, and three remedies.

**FAIL-AFTER (`perf/probe_after_4.py` -> `out_after_4.txt`):**

```
  n_fine_cap=8192 (does not bind): warnings 0  byte-identical to uncapped: True
  n_fine_cap=256  (BINDS):         warnings 1  field changed: True
  on_n_fine_cap='ignore':          warnings 0  same field as 'warn': True
  on_n_fine_cap='error':           MemoryError
  on_n_fine_cap='nope' -> ValueError;  n_fine_cap=1 -> ValueError
  ordering: ram_budget=inf   n_fine_cap=256  -> guards ['COUNT']
            ram_budget=2 GB  n_fine_cap=256  -> guards ['COUNT']
            ram_budget=2 GB  n_fine_cap=None -> guards []
```

The `'warn'` and `'ignore'` fields are identical, which is the contract: the
disposition changes what is SAID, never what is computed.

### 2.3 What this changes for a chain caller

A chain caller who sets a small `n_fine_cap` now gets the readout capped too.
This is visible in the existing suite and is the intended fix:
`test_niche_d6_exact_tilted_leg.py::test_the_refusal_can_be_downgraded_and_then_it_is_worse`
(`n_fine_cap = 256`) now emits

```
carrier_referenced_exact_focus_readout: the readout's internal fine grid is
COUNT-LIMITED to 256x256 by n_fine_cap.  The un-degraded requirement was
1024x1024 -- the 2.1771 mm window (window_factor=4.0 x exit beam radius
544.4852 um) at dx_fine=2.1261 um -- so the readout runs at dx_fine=8.5044 um
instead, COARSER than the exit sphere's Nyquist pitch lambda/(2*NA)=3.6089 um
at NA=0.1815: every spatial frequency above NA=0.0770 is silently DISCARDED
...  Remedies: raise n_fine_cap to 1024 (RAM permitting); shrink
window_factor (currently 4.0); or pass n_fine_cap=None ...
```

and still passes.  Before this fix that readout ran at 1024 while its own leg
ran at 256 -- the two halves of one `n_fine_cap` disagreeing, silently.

### 2.4 The consequence for design 121, stated plainly

With `frac = 0.5` unchanged and `n_work = 16`, approving `n_fine = 16384`
needs `16 * 16 B * 16384^2 / 0.5` = **137.4 GB of budget** (it needed 34.4 GB).
On the 137.4 GB box the audit measured, the shipped `NFC = 16384` therefore
degrades to 8192 with the `RESOLUTION-LIMITED (non-converged)` warning, and
audit sec 2.5 measured what that costs: `dx_fine` 1.5243 um against a
measured-NA Nyquist pitch of 1.231 um, outer-NA content discarded, readout peak
intensity 1.3 % different.

Three remedies, in order of honesty:

1. `focus_readout={'ram_budget': float('inf')}` -- accept the uncapped
   behaviour deliberately, having read that the order touches 98.85 GB.
2. `lumenairy.set_max_ram(...)` above the true peak.
3. Take the audit's rows 1 / 5 / 8 (the dead phase factors, the meshgrids, the
   pinned pre-leg field -- 30 GB, all NONE-verdict, none of them in this
   change), after which the calibrated model fits the box for real.

The acceptance configuration (`NFC = 8192`) is unaffected: it needs 34.4 GB of
budget and the runs in section 5 were approved at `n_fine = 8192`, confirmed
from the run's own stage trace.

---

## 3. ITEM #6 -- THE TWO UNCAPPED CACHES

### 3.1 `_bluestein._H_FFT_CACHE` gets `_H_CACHE`'s byte caps

`_bluestein.py`.  The cache was bounded by **16 ENTRIES and nothing else**,
while one entry is `L^2` complex128 with `L = next_fast_len(N_in + N_out - 1)`
-- a size set by the CALLER's grid.

**FAIL-BEFORE (`perf/probe_fail_before_6.py` -> `out_fail_before_6.txt`):**

```
  _H_FFT_CACHE_MAXSIZE = 16 entries      byte caps present: NONE
  two orders whose alpha differs in the 5th digit -> 2 entries, 0 hits
  N_fine= 8192 N_out=1024 -> L= 9216: one entry 1.359 GB, 16 entries 21.7 GB
  N_fine=16384 N_out=1024 -> L=17424: one entry 4.858 GB, 16 entries 77.7 GB
```

Now carries `_H_FFT_CACHE_MAX_BYTES_PER_ENTRY = 2 GiB` and
`_H_FFT_CACHE_MAX_TOTAL_BYTES = 8 GiB` -- `fft_infra._H_CACHE`'s own numbers,
deliberately, since this is bringing a sibling up to a standard the library
already sets -- through a `_h_fft_cache_store` helper that mirrors
`_h_cache_store`: an over-cap entry is not stored at all (the transform still
returns it), and after any store the oldest entries are evicted until both
bounds hold.  The existing `_cache_registry` enrollment (`'bluestein_h_fft'`)
is untouched, so `clear_asm_caches()` still drains it.

**FAIL-AFTER (`perf/probe_after_6.py` -> `out_after_6.txt`):**

```
  MAX_BYTES_PER_ENTRY = 2.147 GB (_H_CACHE: 2.147 GB)
  MAX_TOTAL_BYTES     = 8.590 GB (_H_CACHE: 8.590 GB)
  20 orders under a 3-entry total cap -> 3 entries, 19.7 MB retained
    (the count cap alone would have retained 104.9 MB)
  per-entry cap below the entry size    -> 0 entries, 0.0 MB retained
  byte-identical outputs across all three cap regimes: True
  hits after 3 DISTINCT orders (the fan case): 0
  hits after 2 IDENTICAL calls (the cache still works): 1, byte-identical True
```

The last two lines are the pair that matters: the fan gets zero value from
this cache (audit sec 6.2's `hits = 0`, reproduced), and a workload that
repeats a key still gets its hit.

### 3.2 The pyFFTW plan cache gets a per-workspace byte cap

`fft_infra.py`.  8 KEYS x TWO full-grid aligned workspaces, priced only in
KEYS -- 24.77 GB retained after ONE design-121 order, of which 12.4 GB is the
second buffer of each key.  `set_fft_double_buffer(False)` is the documented,
byte-identical opt-out (in `set_low_memory()`'s stated SAFE set), but it is
all-or-nothing and process-global.

Now `_plan_entry_n_bufs(shape, dtype)` is the single source of truth for how
many workspaces a key may hold: `2` only while the ping-pong is on AND one
workspace fits `_PYFFTW_PLAN_MAX_BYTES_PER_BUFFER` (default **2 GB**,
decimal).

**A correctness hazard had to be closed to do this at all, and it is worth
recording.**  `_fft2` / `_ifft2` / `_fft2_nd` / `_ifft2_nd` decided
copy-or-not by reading the GLOBAL `_PYFFTW_DOUBLE_BUFFER`, not the entry.  A
per-key cap that built ONE workspace while that global was still True would
have handed the caller a live workspace the next call at the same key
overwrites -- a wrong ANSWER, not a memory regression.  All four sites now
branch on the entry's buffer COUNT, returned as a fourth element from
`_get_or_make_plan`.  `_PYFFTW_PLAN_MAX_BYTES_PER_BUFFER` also joins
`_FFT_STATE_KEYS` / `_FFT_STATE_SETTER_KEYS`, so a spawn worker inherits it
rather than silently rebuilding double-buffered plans at the shapes the parent
capped -- the same failure the v5.17.1 P3-54 note records for its siblings.

**Why 2 GB and not 1 GB, measured.**  This module's v5.16.2 comment prices the
single-buffer copy at "~1-3% of a large transform".  It is not:

| N (complex128) | workspace | `_fft2` double-buffered | `_fft2` single-buffered | cost |
|---|---|---|---|---|
| 4096 | 0.268 GB | 184.0 ms | 160.3 ms | (not capped -- noise) |
| 8192 | 1.074 GB | 597.2 ms | 985.0 ms | **+387.8 ms, 1.65x** |
| 16384 | 4.295 GB | 6328.8 ms | 6853.0 ms | **+524.2 ms, 1.08x** |

The copy is a ~2.8 GB/s first-touch page-fault cost, so it is the same ORDER
as the transform at every large shape -- 65 % of it at N = 8192.  A 1 GB cap
would have taxed the library's most common large shape 1.65x to buy 1.07 GB.
A 2 GB cap binds at 11586^2 and above -- in practice the 16384^2 family, where
the design-121 order does ~3 transforms and pays ~1.6 s of a ~900 s order
(0.2 %) to give back 8.59 GB of a 98.85 GB peak (8.7 %).

**FAIL-AFTER, and byte-identity asserted per leg
(`perf/out_after_6_final.txt`, `out_after_6_byteid.txt`):**

```
  cap = 2.000 GB per workspace
    N= 8192: workspace 1.074 GB -> n_bufs=2     N=12288: 2.416 GB -> n_bufs=1
    N= 9216: workspace 1.359 GB -> n_bufs=2     N=16384: 4.295 GB -> n_bufs=1

  shapes (4096, 16384), fwd AND inv:
    uncapped resident 18.254 GB   capped resident 9.664 GB   (-8.590 GB, 47.1 %)
    N= 4096 fwd/inv: BYTE-IDENTICAL True   (max|d| 0.0e+00)
    N=16384 fwd/inv: BYTE-IDENTICAL True   (max|d| 0.0e+00)
```

### 3.3 What #6 is worth on the production order

With item #7 shipped (the separable readout never plans a 9216^2 transform and
never builds an `L^2` chirp kernel):

| retained | 5.33.1 | this fix |
|---|---|---|
| plan buffers, fwd+inv @ 16384 | 17.18 GB | **8.59 GB** (capped) |
| plan buffers, fwd+inv @ 9216 | 5.44 GB | **0.00 GB** (never planned) |
| plan buffers, fwd @ 8192 | 2.15 GB | 2.15 GB (under the cap) |
| **plan-buffer total** | **24.76 GB** | **10.74 GB (-14.03 GB)** |
| `_H_FFT_CACHE`, per order | 1.359 GB | **0.000 GB** |
| `_H_FFT_CACHE`, 32-order fan | 21.7 GB (77.7 at `wf`=7), **unbounded in bytes** | **0.00 GB** shipped route; **8.59 GB** hard ceiling otherwise |

**-35.7 GB across the production order / fan**, none of it moving a value.
The `_H_FFT_CACHE` line is directly visible in the acceptance runs' cache
report: **1 entry / 0.604 GB / hits=0** with item #7's flag off, **0 entries**
with it on (5.2).

---

## 4. ITEM #7 -- THE SEPARABLE BLUESTEIN READOUT

### 4.1 What was built

`_bluestein.py` gains `_bluestein_axis_1d` (one chirp-Z pass along one axis)
and `_bluestein_2d_separable` (two of them), plus a `separable` keyword on
`_bluestein_2d` and `_bluestein_centred_2d`.  The 2-D primitive pads BOTH axes
to `L`, so every working array is `L^2`; the transform is exactly separable --
the shipped code already builds its kernel as `h_y[:,None] * h_x[None,:]` --
so two 1-D passes take the same sum with a largest array of `(N_in x L)`, and
the `L^2` chirp-kernel cache entry ceases to exist (two length-`L` vectors,
0.15 MB, replace 1.359 GB).

Design decisions, and why:

* **`separable` defaults to False on the primitive and on
  `angular_spectrum_propagate_mft` (as the private `_bluestein_separable`).**
  The route is NOT byte-identical, and 101 test files reach the MFT
  propagators.  Nothing measured says the win is needed for
  `fresnel_propagate_mft` / `fraunhofer_propagate_mft` consumers, so their bits
  do not move.
* **The default-ON flag lives on the one consumer the audit measured**:
  `carrier._EXACT_READOUT_SEPARABLE_BLUESTEIN = True`, with an immutable
  `_..._SHIPPED` companion (the `_PYFFTW_AUTO_PROMOTE_SHIPPED` pattern) so a
  pin can assert the shipped contract regardless of what a process last set.
  Setting it `False` is the fail-before switch every comparison here is taken
  against.
* **NumPy only.**  The 1-D transforms go through `_fft_1d`, which mirrors
  `_scipy_or_numpy_fft2`'s backend choice (threaded SciPy pocketfft when
  `USE_SCIPY_FFT`, NumPy otherwise).  pyFFTW is deliberately not consulted: a
  1-D plan family would double the resident workspace this route exists to
  remove.  With any other `xp` the 2-D path runs unchanged.
* Chirps are still built in float64 and cast to `target_cdtype` before
  multiplying -- the float32 chirp-phase trap the 2-D path documents is not
  reintroduced -- and the kernel product is in-place, because a second
  `(rest x L)` array is exactly what this route exists to avoid.

### 4.2 Measured (`perf/probe_after_7.py`, `out_after_7_prod_shape.txt`)

| `N_in / N_out` (L) | 2-D peak | separable peak | reduction | 2-D time | sep. time | speedup | rel L2 | power ratio |
|---|---|---|---|---|---|---|---|---|
| 2048 / 256 (2304) | 0.855 GB | 0.213 GB | **75.1 %** | 1.15 s | 0.24 s | **4.81x** | 8.56e-16 | 1.000000000000 |
| 4096 / 1024 (5120) | 2.575 GB | 1.024 GB | **60.2 %** | 1.90 s | 1.24 s | **1.53x** | 9.14e-16 | 1.000000000000 |
| **8192 / 1024 (9216)** -- the production readout shape | **13.322 GB** | **3.717 GB** | **72.1 %** | 5.86 s | 3.37 s | **1.74x** | 1.02e-15 | 1.000000000000 |

`max|delta| / max|F|` is 1.04e-15 / 1.22e-15 / 1.31e-15 across the three.  The
audit's sec 5.5 prototype measured 70 % / 61 % and 0.15x / 0.42x time on the
first two rows; the shipped implementation reproduces the reductions and lands
inside the same speed band, with the third row (the shape that actually runs in
production) added.

**End to end, one route per FRESH PROCESS** so the two do not share a warm plan
cache, chirp cache or allocator arena (`perf/probe_after_7b.py`,
`out_after_7b.txt`, 2 reps each):

| | 2-D (fail-before switch) | separable (shipped) |
|---|---|---|
| readout peak RSS | 25.333 / 25.333 GB | **14.238 / 14.172 GB** |
| retained `_H_FFT_CACHE` | 1.359 GB | **0.000 GB** |
| retained plan buffers | 3.792 GB | **1.074 GB** |
| readout period | identical | identical |
| peak intensity | 2.029892662e+02 | 2.029892662e+02 |
| field | -- | rel L2 4.53e-16, max\|d\|/max\|F\| 7.7e-16, power ratio 1.000000000000 |

**-11.1 GB (-44 %) for the whole readout call**, reproducible across reps.

### 4.3 The acceptance bars the task set

* **banner digits identical** -- section 5.
* **energy <= 4e-5** -- captured energy identical to every printed digit
  (`P_window/P_in = 0.998014353` both sides); on the field itself the power
  ratio is `1.000000000000`, delta **2.2e-16**.
* **replica-guard boundary tests unchanged** -- `perf/out_after_7.txt`, three
  geometries each straddling ITS OWN period (the period is read from the
  readout, not assumed): on-axis (347.2 um PASS / 372.8 um REFUSE of a 360.0 um
  period), chief-ray-centred readout of a decentred congruence (428.0 PASS /
  453.6 REFUSE of 441.0 um), and the V3 case -- an AXIS readout of a decentred
  congruence, where the residual spends period (368.0 PASS / 393.6 REFUSE of
  441.0 um).  **The guard outcome AND the full message text are
  character-identical under both routes.**
* **V3 chief-ray-residual semantics VERBATIM** -- not merely unchanged but
  UNREACHABLE from this change: the period, the residual `centre_out - centre`
  and `_check_readout_replica` are all computed BEFORE the transform is called,
  and the only line item #7 touches is the `angular_spectrum_propagate_mft`
  call itself.  The residual wording appears in 2 of the 3 refusals (the
  on-axis refusal has no residual to report, correctly).

---

## 5. ACCEPTANCE -- DESIGN 121

### 5.1 Harness

`scratchpad/probe_accept.py`, which reproduces `focus_scan_121.py`'s
construction and its `metrics()` VERBATIM (same groups, same source beam, same
readout, same encircled-energy definition) and stops at the AT-PLANE banner.
Configuration `N=2048, rs=4, NFC=8192, WF=4.0, NOUT=2048, DT=c128`, no
`__main__` guard (the production runners have none, so Newton routes serial as
it does in production).  The reference column is the same harness run on `main`
@ `c8bcbcb` before this work, 2 reps, recorded in
`scratchpad/out_accept_c128_r1.txt` / `_r2.txt` with the readout field saved.

### 5.2 Result

| | reference (`c8bcbcb`, 2 reps) | **#7 flag OFF** (fail-before switch) | **shipped** (#4 + #6 + #7) |
|---|---|---|---|
| FWHM (um) | 3.350000 | 3.350000 | **3.350000** |
| EE3 (%) | 90.348891 | 90.348891 | **90.348891** |
| EE6 (%) | 99.699520 | 99.699520 | **99.699520** |
| EE12 (%) | 99.796669 | 99.796669 | **99.796669** |
| peak | 5.528622557e+03 | 5.528622557e+03 | **5.528622557e+03** |
| `P_window/P_in` | 0.998014353 | 0.998014353 | **0.998014353** |
| halo > 12 / 20 / 40 um | 4.766594e-05 / 2.175495e-05 / 3.905364e-06 | identical | **identical** |
| readout field vs reference | -- | **BYTE-IDENTICAL** | rel L2 **4.59e-16**, power ratio **1.000000000000** |
| `_H_FFT_CACHE` after the run | -- | 1 entry, 0.604 GB, **hits=0** | **0 entries** |
| peak RSS (instrumented harness) | 64.447 / 65.147 GB | 63.332 GB | **62.424 GB** |
| leg `n_fine` | 8192 | 8192 | 8192 (not degraded) |

**The middle column is the load-bearing one.**  With item #7's flag off and
items #4 and #6 fully in, the readout field is byte-identical to the
pre-change reference -- so #4 and #6 are byte-identical on this configuration,
and #7 is the only thing in this change that moves a bit.  It moves them by
4.59e-16.

**Confound, stated.**  The working tree also carries a concurrent agent's
uncommitted `lumenairy/elements/_lens_traced.py` changes.  Their effect on
these runs is bounded by the same byte-identity: the fail-before column
reproduces the reference field exactly, so their edits move no bits either --
but the wall time (365 / 367 s reference -> 260-280 s here) is **theirs, not
this change's**, and no timing claim in this document rests on the acceptance
runs.  The RSS column is likewise a mixture; only the 63.332 -> 62.424 GB step
(one tree, one flag) is cleanly attributable, and it is item #7's.

---

## 6. TESTS

All green.  Windows (python 3.14.6, numpy 2.4.4, pytest 9.0.3) unless noted.

| suite | result |
|---|---|
| `test_niche_tight_focus_readout`, `test_carrier_referenced`, `test_niche_exact_gap_kernel`, `test_fix_v3_mft_centre_window`, `test_fix_v1_v8_readout_guard_and_standoff` | **198 passed, 1 skipped** |
| `test_niche_d2_chain_multi`, `test_niche_d6_exact_tilted_leg` | **76 passed** |
| fga / gbd readout consumers (14 files: `test_fga*`, `test_gbd_feature_complete`, `test_lens_gbd`, `test_niche_p1/p4/r3/r4/r5*`, `test_hammer_h7*`, `test_v5_21_gbd_*`) | **184 passed** |
| FFT infra + plan-cache locks + registry (`test_perf_v4_12_0_fft_infra`, `test_audit_w2_fft_state`, `test_v4_14_2_*`, `test_v4_16_1_*`, `test_audit_w4_rcwa_homog_fftlock`, `test_niche_audit_p2_fresnel_tf_buffer`) | **132 passed, 4 skipped** |
| memory / cache / API surface (`test_memory_guardrail`, `test_audit_s5_8_perf_noloss`, `test_v5_1_0_agent_c_split`, `test_niche_audit_w3_infra`, `test_niche_r0_byte_budgeted_cache`, `test_public_api`) | **954 passed** |
| `n_fine_cap` consumers (`test_niche_d3_guards`, `_s12_`, `_d5_`, `_c1_`, `_c5_`, `_d9_`, `_r9_`) | **165 passed** (see below) |
| `test_niche_d8_congruence_workers` | **30 passed** |
| re-run at the final plan-buffer cap (`_perf_v4_12_0_fft_infra`, `_audit_w2_fft_state`, `_niche_c1_consolidation`, `_memory_guardrail`) | **76 passed** |
| **WSL CI proxy** (python 3.12.3, numpy 2.4.6, `~/lumen_venv`): `test_perf_v4_12_0_fft_infra`, `test_audit_w2_fft_state`, `test_niche_tight_focus_readout`, `test_fix_v3_mft_centre_window`, `test_niche_c1_consolidation` | **74 passed** |
| `ruff check` on all changed files, Windows AND WSL (ruff 0.16.1) | **All checks passed** |

**One test file changed, and it had to be.**
`test_niche_c1_consolidation.py::test_the_focus_readout_whitelist_is_exactly_what_the_chain_consumes`
asserts `set(sample) == _FOCUS_READOUT_KEYS` against a hand-written `sample`
enumerating every accepted `focus_readout` key.  Adding `on_n_fine_cap`
legitimately fails it.  The sample now carries `'on_n_fine_cap': 'warn'` (and
the `<=` assertion above it names the key), which additionally exercises the
paraxial `_par_kw` drop path, since that fixture's final leg is paraxial.  No
`xfail`, no `skip`, no tolerance was widened anywhere.

---

## 7. WHAT THIS DOES NOT DO

1. **Audit rows 1, 3, 4, 5, 8 are untouched** -- the five dead full-grid phase
   factors in `_fine_trace_group_exit` (21.48 GB), row-banding
   `apply_real_lens_traced` (-20 to -30 GB), `_fourier_upsample_crop`'s six
   full-size arrays, the `np.meshgrid` sites, and the pinned `E_exit_fine`.
   Rows 1, 5 and 8 are the audit's own "do these first" and are all
   NONE-verdict; they remain open.
2. **The 98.85 GB production order was NOT re-run.**  It is 918 s at a pinned
   105 GB budget, and section 2.4 explains why it would now be clamped anyway.
   Every production-order figure here is either the audit's own measurement or
   this fix's arithmetic on the audit's measured shapes, priced exactly and
   labelled as such.
3. **The separable route is not byte-identical and is not claimed to be.**  It
   is a different association order for the same sum: rel L2 <= 1.02e-15,
   power ratio 1.000000000000 across every configuration measured.  Anything
   pinning readout bits must set
   `carrier._EXACT_READOUT_SEPARABLE_BLUESTEIN = False`.
4. **`frac = 0.5` in `_memory_bounded_n_fine` was not re-derived.**  With a
   16-array model it now demands 137.4 GB of budget for `n_fine = 16384`
   against a MEASURED whole-process peak of 98.85 GB -- conservative by ~1.4x.
   Whether the fraction should move with the calibrated count is a separate
   question and needs its own measurement.
5. **The plan-cache byte cap is a per-WORKSPACE bound, not a total-bytes one.**
   `_H_CACHE` has both; a total bound here would evict plans and cause
   re-planning churn, which is a different trade and was not measured.

### Probe inventory (session scratchpad, `perf/`)

| file | what |
|---|---|
| `probe_fail_before_4.py` / `out_fail_before_4.txt` | #4 fail-before: the worker clamp at `n_work` 4 vs 16; the readout's missing forward |
| `probe_after_4.py` / `out_after_4.txt` | #4 fail-after: cap semantics, dispositions, validation, count-then-RAM ordering |
| `probe_fail_before_6.py` / `out_fail_before_6.txt` | #6 fail-before: both caches unbounded in bytes; per-order key drift, 0 hits |
| `probe_after_6.py` / `out_after_6.txt` | #6 fail-after: `_H_FFT_CACHE` bounds, values unchanged, cache still hits |
| `out_after_6_copycost.txt`, `out_after_6_final.txt`, `out_after_6_byteid.txt` | #6b: the measured copy cost that set the 2 GB threshold; production-order saving; byte-identity per leg |
| `probe_after_7.py` / `out_after_7.txt` | #7: primitive at the audit's shapes; replica-guard boundary sweep under both routes |
| `out_after_7_prod_shape.txt` | #7 at the production readout shape (L = 9216) |
| `probe_after_7b.py` / `out_after_7b.txt` | #7 end to end, one route per fresh process |
| `../probe_accept.py`, `../out_accept_v5332_{sep0,sep1,final}.txt` | design-121 acceptance, both routes, against the recorded `c8bcbcb` reference |
