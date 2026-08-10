# FIX -- the six defects `VERIFY_PERF_BRANCH_2026_08_10.md` found on `perf/traced-hotpath`

**2026-08-10.  Branch `perf/traced-hotpath`, checked out at `7890b7d` and
modified in the WORKING TREE only -- no commit, no push, no tag, no CHANGELOG
entry.**

Claims list under repair: `VERIFY_PERF_BRANCH_2026_08_10.md` D1-D6, its
section 3 arithmetic mismatches, and its section 5 sibling sweep.

---

## 0. VERDICT

> **All six are fixed, and one of them was worse than reported.**  D1 (the
> `np.dtype(f'c{2 * cb}')` plan-buffer branch) is not only a 43 % Linux
> under-estimate of `estimate_asm_memory` at N = 8192 complex128 -- it is a
> **9.66 GB under-estimate on WINDOWS too**, at N = 12288 complex64, because
> `'c16'` is a perfectly valid dtype there.  And it had already turned one of
> the repo's own pins RED on the CI-proxy platform:
> `test_niche_audit_w3_infra.py::TestA6EstimateAsmMemory::
> test_documented_band_vs_steady_state` read 6.364 on Windows (green) and
> 4.364 on Linux (RED) at `7890b7d`.  Both arms now read 4.364, which is the
> correct value.
>
> **D4 is half a unit error and half a real finding, and the real half is
> worse.**  `kladder_121.py` reports peaks in GiB under a field named
> `peak_*_gb`; the verifier read 6.648 GiB as 6.648 GB, which is where the
> "1.540x, outside the 1.5x bound" came from -- the true ratio at that
> measurement is **1.434**, and three independent 4096 runs agree to 0.27 %
> (7.123 / 7.138 / 7.120 GB), not the 7 % spread reported.  BUT: the 4096
> WORKER CHILD had never been measured, and it is the smallest peak in the
> whole set (**6.937 GB**).  Against it the shipped (20, 4.5 GB) split reads
> **1.476x** -- inside its own bar by 1.6 %, and with only 1.3 % of margin
> over the 8192 child it must not go under.  The split cannot be tight at both
> ends with a 19.1-class slope, so it is re-derived: **(22, 2.6 GB)**, worst
> ratio **1.279** over eleven measured points, tightest bound **1.023**.
>
> **D2's fix is not the one the report implies.**  Dropping the thread-local
> attribute (`del _numexpr_last.l`) does NOT free the array: since numexpr
> 2.11 the record is a `ContextDict` whose payload lives in a `contextvars`
> ContextVar, and the weakref probe still read STILL ALIVE after that.  The
> drain has to `.clear()` it.  With that, all twelve `del` sites free and the
> census is real.
>
> **D5 is fixed by measuring the thing nobody measured**: a design-121
> PARAXIAL congruence worker peaks at **1.174 GB** (k=2 child, shipped tiled
> readout), so it gets its own 1.0 GB floor rather than the exact leg's.
>
> **The traced-niche-set row the branch names as a merge precondition is
> filled** (sec 7).

---

## 1. D1 -- the wrong dtype code in `estimate_asm_memory`  (P1)

`lumenairy/memory.py`.  `cb` is already `np.dtype(complex_dtype).itemsize`, so
`np.dtype(f'c{2 * cb}')` asks the plan-buffer predicate about a dtype of twice
the element size.  Fixed to pass `np.dtype(complex_dtype)` -- the predicate
reads only `.itemsize`, so the caller's dtype is both correct and incapable of
raising for anything `_as_complex_itemsize` accepted.

**FAIL-BEFORE, on this box (Windows, numpy 2.4.4), measured before the edit:**

| N | dtype | keys | true `n_bufs` | pre-fix estimate | correct | error |
|---|---|---|---|---|---|---|
| 8192 | complex128 | 8 | 2 | 19.762 GB | 19.762 GB | 0 (`'c32'` raises, swallowed to 2) |
| 16384 | complex128 | 2 | 1 | 27.332 GB | 18.742 GB | **+8.59 GB over** |
| 11000 | complex128 | 8 | 2 | 35.584 GB | 35.584 GB | 0 |
| **12288** | **complex64** | 8 | 2 | **12.984 GB** | **22.648 GB** | **-9.66 GB UNDER** |

The complex64 row is the one the report attributes to Linux only: `2 * cb` is
16 there, and `'c16'` IS `complex128` on MSVC, so the workspace is priced at
16 B/element, fails the 2 GB cap that 8 B/element passes, and the estimator
reports one buffer where two are built.  **The defect was never
Windows-immune; it was complex128-immune on Windows.**

**FAIL-AFTER, both platforms, exact to the byte** (`estimate_asm_memory` vs
its own formula evaluated at the TRUE `n_bufs`):

```
                          WINDOWS py3.14.6 / numpy 2.4.4   WSL py3.12.3 / numpy 2.4.6
  np.dtype('c32')         TypeError                        complex256
  N= 8192 c128 keys=8     19.762 GB  (n_bufs 2)            19.762 GB  (n_bufs 2)
  N=11000 c128 keys=8     35.584 GB  (n_bufs 2)            35.584 GB  (n_bufs 2)
  N=12288 c64  keys=8     22.648 GB  (n_bufs 2)            22.648 GB  (n_bufs 2)
  N=16384 c128 keys=2     18.742 GB  (n_bufs 1)            18.742 GB  (n_bufs 1)
  N= 1024 c128 keys=2      0.165 GB  (n_bufs 2)             0.165 GB  (n_bufs 2)
  ALL EXACT               True                             True
```

**A pin that was already RED on Linux.**  `test_niche_audit_w3_infra.py::
TestA6EstimateAsmMemory::test_documented_band_vs_steady_state` asserts the
large-N ratio `estimate / (N^2 * 16)`.  At `7890b7d` that ratio was **6.364 on
Windows** (the swallowed `TypeError` gave `n_bufs = 2`) and **4.364 on Linux**
(`'c32'` = complex256 > the 2 GB cap gave `n_bufs = 1`), against a pin of
6.35: green on one arm, RED on the other.  4.364 is the correct value -- the
v5.33.2 per-key cap really does leave ONE workspace at N = 16384 -- so the pin
and the `estimate_asm_memory` docstring both move to it, and the asymptote in
that test is now DERIVED from `_plan_entry_n_bufs` instead of a hard-coded
101.6 B/px.

**Tests** (`test_verify_perf_fixes_2026_08_10.py`): six parametrised shapes
compared byte-for-byte against the formula at the true `n_bufs`; the 19.762 /
22.648 GB absolute pins; a cap-flip test asserting the estimate moves by
exactly one workspace per key (this is the one that fails on pre-fix Windows,
where the branch was a constant); and a source pin on the dtype expression.

---

## 2. D2 -- numexpr held `E_analytic` past every `del`  (P2)

`numexpr.evaluate` is `validate` + `re_evaluate`, and `validate` parks its
kwargs -- `out` included -- in `necompiler._numexpr_last`.  `apply_real_lens`
is the last numexpr caller in `apply_real_lens_traced`, so the field it
returns stayed reachable to the end of the CHAIN.

`_lens_real._drop_numexpr_out_retention()` is called immediately after each of
the module's three `out=` evaluates (the whole-grid phase screen and the two
row-band ones).

**The obvious fix does not work, and that is worth recording.**  `del
_nc._numexpr_last.l` drops the THREAD-LOCAL attribute, but since numexpr 2.11
`l` is a `ContextDict` backed by a `contextvars.ContextVar`, so the payload
survives in the context.  Measured with the verifier's own probe after that
version of the fix:

```
  line 10135  E_analytic       67.109 MB (2048, 2048) complex128  -> STILL ALIVE
        referrer dict keys=['out'] (dict has 4 keys, sample ['out', 'order', ...])
```

`.clear()` empties the ContextVar and is also correct for the plain dict older
numexpr used.

**FAIL-BEFORE / FAIL-AFTER**, the verifier's `p2c_weakref.py` unmodified
(weakref taken at each `del` line, checked at the element's return, D6-class
singlet, decentred + tilted, ray_density + remap + lattice, `n_fine = 2048`):

| | `E_analytic` at element return | other 11 sites |
|---|---|---|
| `7890b7d` | **STILL ALIVE**, referrer = numexpr's 4-key kwargs dict | freed |
| after the drain | **freed** | freed |

**The census, re-run and honest.**  An object-level probe
(`fix_d2_census.py`) weakrefs every object ever bound to the fourteen names
the verifier's census credits as freed, and checks each at the element's
return:

```
  VERIFIABLY FREED   478.183 MB = -14.25 float64 grid equivalents
  STILL ALIVE          0.000 MB = +0.00 float64 grid equivalents
  module globals holding a >=16 MB ndarray: NONE
  numexpr _numexpr_last kwargs['out'] type: NoneType
  VERDICT: the census is REAL -- every name it credits is gone from the process
```

`E_analytic` is 2.00 of those equivalents.  So the verifier's frame census of
**-16.25** equivalents is now entirely real, where it was over-stated by
exactly this one complex128 full grid (its honest figure was -14.25, i.e.
-30.6 GB rather than -34.9 GB at `n_fine = 16384`).  **Item #2's headline
`-34.9 GB` stands as written.**

Stated limitation: this probe reaches 14 named objects totalling 478.18 MB,
where the verifier's frame census measured 545.26 MB; the ~67 MB difference is
one more grid its census counted by id and its prose did not name.  Nothing in
either accounting is now alive.

**And the drain changes no VALUE**, which matters because it is inserted into
the phase-screen loop of every real-lens call.  `apply_real_lens` run against
a PRE-FIX package snapshot (`git show 7890b7d:` over the changed files, one
file differing) and against the tree, same input, three routes:

```
                              PRE-FIX (7890b7d)                  FIXED
  N=1024 whole-grid    3cbfcfe6d4004d8f...              3cbfcfe6d4004d8f...
  N=1024 row-band(128) 3cbfcfe6d4004d8f...              3cbfcfe6d4004d8f...
  N=2048 whole-grid    fd9e27acaa404ef2...              fd9e27acaa404ef2...
```

sha256 identical on all three, and the row-band route is byte-identical to the
whole-grid one both before and after -- the contract
`test_chunked_sag_byte_identical` exists for.

**Tests**: the retention pinned three ways -- an end-to-end `apply_real_lens`
at 1024^2 (the size gate is 1 Mi elements) whose result must die when its last
name is dropped; a process-wide scan for module globals holding a grid; a
source pin that every `out=` site is followed by the drain AND that the drain
`.clear()`s rather than `del`s; and the drain against numexpr directly.

---

## 3. D3 -- the over-cap chirp entry was copied before it was rejected  (P2)

`_bluestein._h_fft_cache_store` now reads `H_FFT.nbytes` BEFORE the
`np.copy`, exactly as `fft_infra._h_cache_store` reads `_entry_bytes(H)`.

**FAIL-BEFORE / FAIL-AFTER** (`fix_d3_copyorder.py`, which carries the
`7890b7d` body transcribed from `git show` rather than re-typed, so the
before-arm is the code under test; per-entry cap set to 1 B so every store is
rejected; `tracemalloc` around the call):

```
per-entry cap = 1 B, so EVERY store is rejected:
  BEFORE   entry  67.109 MB   retained   0.000 MB   traced peak  +67.109 MB
  AFTER    entry  67.109 MB   retained   0.000 MB   traced peak   +0.000 MB

  retained equal (both reject): True
  under-cap entry still stored, byte-identical, and COPIED: True / True / True
```

At the `window_factor = 7` geometry the cap's own comment works through
(`L = 17424`) that transient is **4.858 GB**, allocated for an entry thrown
away one line later, on the run whose peak is the thing being defended.

---

## 4. D4 -- the clamp model, re-derived  (P2)

### 4.1 The unit error, stated plainly

`kladder_121.py` divides by `2**30` and names the field `peak_*_gb`.  Every
byte figure in `FIX_PERF_PARALLEL` sec 3 is that value x 2^30.  Reading one as
decimal GB moves a ratio by 7.4 %:

| measurement | as reported by the harness | in bytes | model / measured |
|---|---|---|---|
| 4096, verifier's fresh run | 6.648 GiB | 7.138 GB | 1.434 (**not** 1.540) |
| 8192, verifier's fresh run | 22.764 GiB | 24.443 GB | 1.078 |
| 8192, verifier's k=3 child | 24.169 GiB | 25.951 GB | 1.015 |

The verifier converted correctly for the child row ("24.17 GiB = 25.95 GB")
and not for the tree rows, which is what produced both the 1.540 ratio and the
"7 % run-to-run spread".  In consistent units three independent 4096 runs read
**7.123 / 7.138 / 7.120 GB** -- a spread of **0.27 %**.

### 4.2 The real finding: the 4096 worker CHILD

Nobody had measured a worker child at the small grid, and it is the smallest
peak in the whole set. Fresh runs at the doc's own sec 3.1 configuration
(`KEEP='0,0;1,0' NFC=4096 RAMB=inf NOUT=8192 TILE=1024 DXO=0.2um RN=1024 RS=4
NW=1 WF=4.0 LEG=auto OTEG=warn`):

| arm | peak tree | largest child | field sha256 |
|---|---|---|---|
| k=1 | 6.631 GiB = **7.120 GB** | -- | `39e09deb4b8dcf93...` |
| k=2 | 12.692 GiB = 13.628 GB | 6.461 GiB = **6.937 GB** | `39e09deb4b8dcf93...` |

(Both arms produce the same field hash as the verifier's own 4096 arm -- an
unplanned bit-identity check at k=2/4096 that the k-ladder had only run at
8192.)

Against the 6.937 GB child the shipped (20, 4.5 GB) model reads **1.476x**.
Inside its own 1.5x bar, by 1.6 %.  And it cannot simply be lowered: the same
constant is what bounds the 8192 child at 26.001 GB, with only **1.3 %** of
margin.  The feasible window for `_FINE_GRID_BASE_BYTES` at slope 20 is
[4.157, 4.668] GB, and across all of it either the child margin is under 2 %
or the 4096 ratio is at the bar.  **A 19.1-class slope cannot be an upper
bound at both ends.**  That is the real defect, and it is structural.

### 4.3 The re-derivation

The split is what it actually is -- a constrained UPPER-BOUND ENVELOPE over
every measured point, not a decomposition -- chosen to minimise the worst
model/measured ratio subject to: bounding all eleven points, keeping >= 2 % of
margin over the binding 8192 child, and not pushing the 16384 price past what
the runners' pre-flight approves for ONE worker on this box.

| constant | was | now |
|---|---|---|
| `_FINE_GRID_WORK_ARRAYS` | 20 | **22** |
| `_FINE_GRID_BASE_BYTES` | 4.5 GB | **2.6 GB** (the three-grid fit's measured 2.3 GB intercept, rounded up; still clear of the 1.75 GB import commit) |
| `_PARAXIAL_BASE_BYTES` | (did not exist) | **1.0 GB** (sec 5) |

`_fine_grid_peak_bytes(n, n_px=1024^2)`, all eleven measured points:

| `n_fine` | measured | what it is | model | ratio |
|---|---|---|---|---|
| 4096 | 7.123 GB | 2 orders, whole process | 8.875 GB | 1.246 |
| 4096 | 7.120 GB | 2 orders, whole process, re-run | 8.875 GB | 1.246 |
| 4096 | **6.937 GB** | 2 orders, largest CHILD at k=2 | 8.875 GB | **1.279** |
| 8192 | 23.968 GB | 2 orders, whole process | 26.591 GB | 1.109 |
| 8192 | 24.443 GB | 2 orders, whole process, re-run | 26.591 GB | 1.088 |
| 8192 | 25.331 GB | 6 orders, whole process | 26.591 GB | 1.050 |
| 8192 | 25.951 GB | 2 orders, largest CHILD at k=3 | 26.591 GB | 1.025 |
| 8192 | **26.001 GB** | 6 orders, largest CHILD at k=2 | 26.591 GB | **1.023** |
| 8192 | 25.985 GB | 6 orders, largest CHILD at k=3 | 26.591 GB | 1.023 |
| 8192 | 25.772 GB | 6 orders, largest CHILD at k=4->3 | 26.591 GB | 1.032 |
| 16384 | 84.589 GB | 2 orders, whole process | 97.458 GB | 1.152 |

Worst **1.279** (was 1.476), tightest **1.023** (was 1.013).  Every point
bounded.

**What did NOT move**, checked rather than assumed:

```
  cap ladder (test_niche_p2_guards):  0.25/1/4/16/34/136 GiB -> 512/1024/2048/4096/4096/8192   UNCHANGED
  _fine_grid_ceiling(105.06 GB) = 8192      _fine_grid_ceiling(104.42 GB) = 8192               UNCHANGED
  D8 worker ladder, 105.1 GB free, 8 GB reserve:
      NFC= 8192  26.59 GB/worker  2/3/4/6/8/32 -> 2/3/3/3/3/3                                  UNCHANGED
      NFC=16384  97.46 GB/worker  2/3/4/6/8/32 -> 1/1/1/1/1/1                                  UNCHANGED
  budget a grid needs:  4096 11.8 GB   8192 47.2 GB   12288 106.3 GB   16384 189.0 GB
```

**The trade that was made deliberately.**  (23, 2.0 GB) reads 1.232 worst --
better -- but prices `n_fine = 16384` at 101.2 GB per worker, which is where
`_grid_intent.preflight` stops approving a SINGLE 16384 worker on a ~105 GB
box.  (22, 2.6 GB) keeps that configuration approvable at 97.5 GB.  The note
is at the constant so the next re-tune meets it.

---

## 5. D5 -- the exact leg's floor is no longer charged to a paraxial worker  (P2)

`_FINE_GRID_BASE_BYTES`'s own ENVELOPE note says it is a design-121-CLASS
figure; `_multi_resolve_workers` charged it to every worker, `final_leg=
'paraxial'` included.  `_fine_grid_peak_bytes` now charges
`_PARAXIAL_BASE_BYTES` when `n_fine == 0`.

**MEASURED, because nothing in the three fix documents measures a paraxial
worker.**  Row 1 is the design-121 fan's own paraxial worker with the shipped
tiled readout (`kladder_121.py`, `LEG=paraxial CW=2 RN=1024 NOUT=8192
TILE=128`); the rest are one-process-per-point stand-ins (fresh interpreter,
one chain, peak read both as Windows `peak_wset` and as a 20 Hz RSS sample)
that separate the floor from the input-grid term:

| what | peak | grid term | implied floor | model | ratio |
|---|---|---|---|---|---|
| design-121 fan, largest CHILD at k=2 (1024^2 in) | **1.174 GB** | 0.369 GB | **0.805 GB** | 1.369 GB | **1.166** |
| stand-in, 1024^2 in | 0.951 GB | 0.369 GB | 0.582 GB | 1.369 GB | 1.440 |
| stand-in, 512^2 in | 0.571 GB | 0.092 GB | 0.478 GB | 1.092 GB | 1.913 |
| stand-in, 256^2 in | 0.470 GB | 0.023 GB | 0.447 GB | 1.023 GB | 2.177 |
| stand-in, 128^2 in | 0.435 GB | 0.006 GB | 0.429 GB | 1.006 GB | 2.312 |

The same stand-in puts the EXACT leg at 0.51-1.68 GB, so the two floors are
genuinely different quantities and 1.0 GB is not merely the interpreter.

**Effect on approval**, the table D5 reports, recomputed:

| leg / input | before | after | free 16 GB | free 32 GB | free 128 GB |
|---|---|---|---|---|---|
| paraxial, 128^2 | 4.5058 GB | **1.0058 GB** | 1 -> **7** | 5 -> 23 | 26 -> 119 |
| paraxial, 1024^2 | 4.8691 GB | **1.3691 GB** | 1 -> **5** | 4 -> 17 | 24 -> 87 |

**Envelope, and it is narrower than the exact leg's.**  The paraxial readout's
cost scales with `N_out`, which NO term in this model prices.  Measured on the
same fan with the readout NOT tiled (`TILE=none`, so it runs on the full
8192^2 common grid) a paraxial worker peaked at **11.221 GB** -- ten times the
floor, all of it the untiled readout.  The shipped runners tile the readout;
a caller who does not must size that window itself.  Recorded at the constant.

---

## 6. D6 and the arithmetic mismatches

| where | was | now | evidence |
|---|---|---|---|
| `fft_infra.py` plan-cap comment | "a 2 GB cap binds at 11586^2 and above" | "binds at **N >= 11181**" | `11180^2 x 16 = 1.99988e9`, `11181^2 x 16 = 2.00024e9`; 11586 is the 2 **GiB** crossover, which the same comment says it deliberately did not use.  Pinned by a test that flips `_plan_entry_n_bufs` between 11180 and 11181 |
| `fft_infra.py` copy-cost comment | "~4.5 s of a ~900 s order (0.5 %)" | "~1.6 s ... (0.2 %)" | 3 transforms x the MEASURED +524.2 ms at 16384 = 1.57 s.  The doc's number was right and the code comment was not |
| `carrier.py` `ram_budget` docstring | "`_FINE_GRID_WORK_ARRAYS` = 16 complex128 arrays" | "= 22" | it read 16 through two re-measurements; now pinned by a test that reads the constant |
| `memory.py` docstring | "runs ~6.4x [the steady state] asymptotically" | "~6.4x where a key still holds TWO workspaces, ~4.4x once the v5.33.2 cap drops it to one" | see sec 1 |
| `FIX_PERF_PARALLEL` sec 3.2/3.3/3.4 | 2.3 GB-era figures | regenerated at the shipped (22, 2.6 GB) | sec 4.3 |
| `FIX_PERF_PARALLEL` sec 4.2 clamp transcript | "~96.6 GB (24.1 GB per worker)" | kept VERBATIM (it is a real run's output) + annotated with the shipped arithmetic | 4 x 26.591 = 106.4 GB against 96.2 GB available still returns 3; the decision does not move |
| `FIX_PERF_PARALLEL` sec 4.4 model table | 26.34 / 90.77 GB | 26.59 / 97.46 GB | sec 4.3 |
| `FIX_PERF_CACHES` sec 0 / 2.4 / 7.4 and `ADJUDICATION` sec 0 / 2.1 / 7.2 | "137.4 GB of budget" | annotated: 137.4 is the `n_work = 16` figure those documents ship; **189.0 GB** at the shipped 22 | the conclusion is unchanged and stronger -- 16384 is further out of reach on this box, not nearer |

`focus_readout={'n_fine_cap': None}` (D6's third bullet) is left alone: the
verifier itself classes it as PRE-EXISTING (`carrier.py:7905`, unchanged by
this branch), and it is a chain-level forward, not a defect of this work.
Recorded here so it is not lost.

---

## 7. THE TRACED NICHE SET -- the row the branch calls a merge precondition

`FIX_PERF_POLY_LOCALS_2026_08_09.md` sec 4 leaves this row "STILL RUNNING ...
**This row must be filled in before the branch is proposed for merge**".  The
set is regenerated by its own stated rule -- every `tests/unit` module that
references `apply_real_lens_traced` / `propagate_traced_carrier_chain` /
`prepare_real_lens_traced` / `_lens_traced` -- which now selects **98** files
(97 plus this fix's own module).

**2587 passed, 5 skipped, 0 failed** -- 32:31 wall, Windows py3.14.6 /
numpy 2.4.4, `-p no:randomly -n 5 --dist loadfile`, everything at
`OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS=1`.

**Run TWICE**, because the first pass overlapped a `carrier.py` comment edit
and a suite that reads its own source with `inspect.getsource` cannot be
trusted across one.  Both passes report the identical `2587 passed, 5
skipped`; the second is on a FROZEN tree (45:35 and 32:31 -- the difference is
box load, not content).

`--dist loadfile` (one FILE per worker, so intra-module order and
module-global state are preserved) rather than serial: serial measured ~5 %
of the set per hour on this box, i.e. ~18 h, which is not a merge gate anyone
will run.  Distribution is by file, `-p no:randomly` is still set, and the
count matches across two independent runs.

The 5 skips are all PRE-EXISTING documented exemptions, none touched by this
change:

```
  test_v4_14_2_dispatcher_pin_cache_locks.py  _REGISTRY_LOCK        (exemption list)
  test_v4_14_2_dispatcher_pin_cache_locks.py  _PERSISTENT_POOL_LOCK (exemption list)
  test_v4_14_2_dispatcher_pin_cache_locks.py  _BLAS_CONTROLLER_LOCK (exemption list)
  test_v4_14_2_dispatcher_pin_cache_locks.py  _ZARR_MKDIR_PATCH_LOCK (documented)
  test_niche_exact_gap_kernel.py              m == 1 identically for a collimated carrier
```

**`FIX_PERF_POLY_LOCALS_2026_08_09.md` sec 4's open row is therefore
CLOSED**, and it is closed on a tree that also carries the two newly capped
caches, the numexpr drain and the re-derived clamp -- i.e. it covers this
fix, not only the branch it was written for.

---

## 8. SIBLING SWEEP (VERIFY sec 5)

### 8.1 Class B -- caches with a count cap and no byte cap

Both members now carry the `fft_infra._H_CACHE` / `_bluestein._H_FFT_CACHE`
policy verbatim: 2 GiB per entry inside 8 GiB total, size tested BEFORE the
entry is inserted, eviction to both bounds.

| cache | entry | exposure before | after |
|---|---|---|---|
| `analysis/beam_stats._MESHGRID_CACHE` | a TUPLE OF TWO full `(Ny,Nx)` float64 grids | 8 x 2 x 2.147 GB = **34.4 GB at N = 16384**, unbounded in bytes | per-entry cap binds at `Ny*Nx >= 1.25e8` (N >= 11181 square); total <= 8 GiB |
| `analysis/zernike._ZERNIKE_BASIS_CACHE` | `(basis, mask)`, basis `(n_modes, Npix)` float64 | **4.8 GB per entry** at 36 modes on 4096^2, x 32 | per-entry cap binds at `n_modes*Npix >= 2.5e8`; total <= 8 GiB |

A cache is a cache: an over-cap entry is not retained, the caller still gets
the value it asked for, and the next call rebuilds it.  Both are pinned for
value-identity under the cap, for total-cap eviction, and (meshgrid) for the
count cap that was already there.

`fft_infra._FREQ_GRID_CACHE` and `_BANDLIMIT_CACHE` are NOT members, as the
verifier says: they store 1-D `kx_sq`/`ky_sq` and `bl_x`/`bl_y`, so their
exposure is O(N).

### 8.2 Class A -- `capstone_stageB.py`

Hardened, being the one the verifier named and the capstone's own runner.  It
runs `focus_scan_121.py` under `run_name='__main__'`, i.e. it BECOMES that
runner's `__main__`, and that runner asks for `n_workers=8` -- so unguarded,
`_script_has_main_guard` inspected `capstone_stageB.py`, found no guard, and
silently forced the Newton pool serial.  **The capstone measured a knob that
was never applied.**  It now has a top-level `__main__` guard (verified with
the library's own predicate) and the campaign's three targeted filters instead
of relying on the runner's; the `CAPSTONE_WARN=1` blanket-neutraliser is kept
as a backstop and its comment now says so.

### 8.3 LEDGER -- the 15 unguarded runners, NOT edited

Recorded rather than changed: they are one-off probes, not the acceptance
path, and editing 15 harnesses in a fix round is how an unrelated regression
arrives.  `guard` from `_lens_traced._script_has_main_guard`; `blanket` from an
AST scan for a single-argument `filterwarnings('ignore')` / `simplefilter(
'ignore')`; `n_workers` from literal keyword values.

| file | guard | blanket ignore | `n_workers` |
|---|---|---|---|
| `carrier_chain_121.py` | UNGUARDED | YES | 8 |
| `repro_dx_scaling.py` | UNGUARDED | YES | 8 |
| `review_real_chain_convention.py` | UNGUARDED | YES | 8 |
| `traced_group_dx_probe.py` | UNGUARDED | YES | 8 |
| `traced_group_oracle.py` | UNGUARDED | YES | 8 |
| `review_carrier_convention_2x2.py` | UNGUARDED | YES | 8 |
| `stigmatic_control_121.py` | UNGUARDED | YES | 8 |
| `ablate_exitna_transpose.py` | UNGUARDED | YES | 1 |
| `p2diag_capture.py` | UNGUARDED | YES | -- |
| `wfe_probe_common.py` | UNGUARDED | -- | 8 |
| `_c14_pre_baseline_lens_traced.py` | UNGUARDED | -- | -- |
| `_d121_common.py` | UNGUARDED | -- | -- |
| `approx_common.py` | UNGUARDED | -- | -- |
| `adjudicate_nfc_8192.py` | UNGUARDED | -- | -- |
| `capstone_stageC.py` | UNGUARDED | -- | -- |

The seven carrying BOTH a blanket filter and `n_workers=8` are the ones able
to reproduce the silent-8192 failure exactly as `focus_scan_121.py` did.  Four
of the fifteen (`_d121_common`, `approx_common`, `wfe_probe_common`,
`_grid_intent`) are import-only helpers for which a guard is meaningless; the
rest are single-purpose probes.  `fan_multi_121.py`, `focus_scan_121.py`,
`kladder_121.py`, `capstone_d121.py` and now `capstone_stageB.py` -- the
acceptance path -- are all GUARDED.

---

## 9. GREEN

All at `OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = MKL_NUM_THREADS =
NUMEXPR_NUM_THREADS = 1`, `-q -p no:randomly`.

**Windows**, python 3.14.6 / numpy 2.4.4 / scipy 1.17.1:

| suite | result |
|---|---|
| `test_verify_perf_fixes_2026_08_10.py` (NEW: 28 tests) | **28 passed**, 6.1 s |
| MEMORY + CACHE: `test_memory_guardrail` + `test_niche_audit_w3_infra` + `test_audit_g06_perf` + `test_perf_v4_12_0_zernike_cache` + `test_v4_14_2_dispatcher_pin_cache_locks` + `test_v5_21_2_subsystem_audits` + `test_audit_misc` + `test_audit_optimize` | **605 passed, 4 skipped**, 3:02 |
| THE PERF MODULE + READOUT: `test_niche_perf_poly_locals` + `test_niche_tight_focus_readout` + `test_carrier_referenced` + `test_niche_audit_w3_infra` + `test_memory_guardrail` | **155 passed**, 1:58 |
| THE CLAMP (incl. the p2_guards cap ladder): `test_niche_p2_guards` + `test_niche_d8_congruence_workers` + the new module | **75 passed**, 1:47 |
| LENS BYTE-IDENTITY + POOLS: `test_lens_chunked_sag` + `test_slant_chunk_byte_identical` + `test_elements_lens` + `test_audit_lens` + `test_v4_14_0_dispatcher_pin_apply_lens` + `test_niche_d3_guards` + `test_fix_newton_pool_memory` + `test_niche_newton_pool_both_fits` | **280 passed**, 8:25 |
| `test_public_api.py` (the API surface the two new cache helpers sit next to) | **697 passed**, 0.8 s |
| **THE TRACED NICHE SET** -- 98 modules, 2592 tests (sec 7) | **2587 passed, 5 skipped**, 32:31 (`-n 5 --dist loadfile`; sec 7) |

**WSL** (the repo's CI proxy), python 3.12.3 / numpy 2.4.6,
`~/lumen_venv`:

| suite | result |
|---|---|
| `test_verify_perf_fixes_2026_08_10.py` | **28 passed**, 5.4 s |
| the D1 arm + `test_niche_audit_w3_infra` + `test_memory_guardrail` + `test_niche_p2_guards` + `test_niche_d8_congruence_workers` + `test_niche_d3_guards` | **218 passed**, 12:47 |

**Three WSL failures were seen on an earlier arm and all three are
accounted for**, because a green-on-retry is not evidence:

* `test_pool_uses_spawn_never_the_platform_default_fork` and
  `test_futures_are_drained_by_completion_not_submission_order` are pure
  `inspect.getsource` assertions on `carrier._multi_parallel_results`, and
  `carrier.py` was EDITED (one comment line added) while that run was in
  flight -- `getsource` slices the file on disk using the imported module's
  `co_firstlineno`, so it returned the wrong lines.  Self-inflicted by
  editing the tree under a running suite; both pass on the frozen tree
  (`test_niche_d8_congruence_workers.py` alone on WSL: **35 passed**).
* `test_the_drain_actually_releases_the_out_array` was a real, platform-split
  TEST defect, not a library one: its precondition used
  `gc.get_referrers`, and numexpr's kwargs dict holds only strings, a bool
  and a numeric ndarray -- none GC-tracked -- so CPython leaves the dict
  itself untracked and `get_referrers` never reports it.  MEASURED:
  `gc.is_tracked(kwargs)` is `False` on py3.12/Linux (invisible) and the dict
  IS reported on py3.14/Windows.  The precondition is now a weakref, which
  sees the retention on both.  This is also why the verifier's own probe used
  weakrefs and said `gc.get_objects()` was useless here.


`ruff check` on all changed library, test and validation files: **All checks
passed**.  Every changed file decodes as cp1252 and **0** added lines contain
a non-ASCII byte.  Nothing was xfailed, skipped or deselected by this change.

### 9.1 Fail-before, as a suite

The new module run against a PRE-FIX package snapshot -- a full copy of
`lumenairy/` with the seven changed files replaced by `git show 7890b7d:` and
nothing else altered:

```
  18 failed, 9 passed, 1 skipped        (the skip is capstone_stageB.py,
                                         absent from the package snapshot)
  ...with validation/ added:            19 failed
```

i.e. **19 of the 28 new tests fail on `7890b7d`**.  The nine that pass either
way are structural checks that were already true (the meshgrid count cap, the
under-cap chirp store) or generic contracts.

---

## 10. WHAT THIS DOES NOT DO

1. **`_FINE_GRID_RAM_FRAC` is still 0.5 and still not independently derived.**
   Untouched here, as on the branch.  It is now measured to be SUFFICIENT (it
   must cover a 2.6 GB floor out of half the budget, i.e. from ~5.2 GB up).
2. **The paraxial model still has no `N_out` term.**  Measured and bounded
   (sec 5), not fixed: adding one would change the exact leg's pricing too and
   nothing here measures that interaction.
3. **The 32-order fan was not re-run.**  The constants moved; the ADJUDICATION
   arms were not re-executed.  The clamp's CHOICES at both caps are unchanged
   (sec 4.3), which is what those arms depend on.
4. **`focus_readout={'n_fine_cap': None}`** is documented, not fixed (sec 6).
5. **Concurrency.**  `_h_fft_cache_store`, the two newly capped caches and the
   numexpr drain were exercised single-threaded.  The drain clears a
   THREAD-LOCAL record from the thread that filled it, which is the only
   ordering that can matter, but no test races it.
6. **CuPy / JAX.**  The numexpr route is CPU-only by construction (`xp is np`
   gates it), so the drain cannot reach a device array; not verified on one.
7. **The 15 unguarded runners are ledgered, not fixed** (sec 8.3).
