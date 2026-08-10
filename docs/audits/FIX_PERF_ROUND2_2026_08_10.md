# FIX -- post-`_poly` RE-PROFILE and the round-2 speed items

**2026-08-10.  Branch `perf/traced-hotpath`.  Re-profiles the design-121 fan
order on TODAY's tree -- the 2026-08-09 profile in
`AUDIT_TRACED_SPEED_2026_08_09.md` is obsolete, because the site it measured at
57.8 % (`_ResidualEikonal._poly`) has since been made 3.78x faster
(`FIX_PERF_POLY_LOCALS_2026_08_09.md`) -- and then implements the items the
FRESH profile ranks.  Library files edited: `lumenairy/propagators/carrier.py`
and `lumenairy/elements/_lens_traced.py`.  One new test module
(`tests/unit/test_niche_perf_round2_2026_08_10.py`); two source-text pins in
`tests/unit/test_fix_newton_pool_memory.py` re-anchored (their invariant is
unchanged -- see sec 6.2).  `CHANGELOG.md` was not touched.  Out of scope by
instruction and untouched: the MFT propagators' `_bluestein_separable` default,
`pmm/**`, the just-re-derived clamp constants, and anything complex64.**

---

## 0. HEADLINE

> **The profile has changed shape, and the audit's ranking with it.**  On
> today's tree one design-121 fan order at `n_fine_cap=8192` spends **23.5 %**
> in `_poly` (was 57.8 %) and **9.9 % in `scipy.ndimage.map_coordinates`, which
> is now the #2 site** -- 1.7x the share the audit measured, purely because the
> denominator shrank.  The two RAW-`np.fft` sites the audit found are together
> **5.5 %** (`_raw_fft` 4.1 % + the `fftshift`/`np.roll` around them).  The
> ranking is confirmed at a second grid: at `n_fine_cap=12288` every one of the
> top six moves by less than 0.8 points.
>
> **`n_fine_cap=16384` could not be run at all** -- not by choice, but because
> THIS BRANCH's own measured box guard refuses it on this box: 97.5 GB modelled
> against 94.4 GB free.  The confirmation arm is 12288, the largest grid the
> guard approves.  Sec 1.3.
>
> **The new #1 is `_poly` again, and it is CAMPAIGN-SIZED, not an easy win.**
> The free structural fix (one `np.power` per distinct exponent, no Hessian on
> the value path) has already been taken; what remains is 27 terms x 3
> accumulators of full-band multiply-add over the whole fine grid, and every
> route past it (Horner/Clenshaw, a fused numba kernel, a polar reduction on
> the frozen annulus) MOVES BITS.  Sec 1.4 prices each.  One byte-identical
> sliver inside it -- the exponent-0/1 table entries -- was measured and is
> taken as item 4b.
>
> **The audit's one open question about item 6 is answered, and the answer is
> NO.**  It could not say whether the NaN-mask guard would ever fire on design
> 121 (its sec 11 item 5).  A `map_coordinates` census over a whole order says
> the coarse OPL and the coarse ray-density amplitude carry NaN at **every one
> of the eight sites that consult them -- all seven coarse groups and the fine
> retrace**.  The guard is correct, bit-identical and worth a measured
> **1.53-1.57x** of the upsample where it applies, and it is worth **exactly
> zero on the workload of record**.  Sec 3.1 -- including the retraction of a
> first census that measured the wrong array and read the opposite.
>
> **COMPOUNDED, measured on the same order: chain B 160 s -> 144 s (min of 2
> reps per arm, PROF=0, arms interleaved against a pre-change package
> snapshot) = 1.111x; and 170 s -> 149 s = 1.141x at matched sampler
> overhead.**  Every AFTER run beat every BEFORE run.  The per-site arithmetic
> predicts -9.7 % and the wall delivers -10.0 / -12.4 %.  Green: 276 Windows /
> 179 WSL, zero xfail or skip; the WSL arm caught a real defect -- in this
> round's own TEST, which compared two different inputs and passed on Windows
> by luck (sec 6.1).

---

## 1. PHASE A -- THE FRESH PROFILE

### 1.1 Box, build, and what was run

```
Windows 11 Pro 10.0.26200        AMD Ryzen 9 5950X, 24 logical CPUs
127.9 GB physical RAM
python 3.14.6   numpy 2.4.4   scipy 1.17.1   pyfftw 0.15.1   py-spy 0.4.2
lumenairy 5.33.1, branch perf/traced-hotpath at ebc248f (working tree)
```

Workload: `validation/repro_traced_carrier_121/fan_multi_121.py` run UNMODIFIED
through `runpy` by `scratchpad/perf_round2/prof2.py`, `KEEP="-4,-2"` (the
historical worst-case order, and the one both the capstone and the audit
timed), `NW=1`, everything else at the runner's own defaults -- `RN=1024 RS=4
DXO=0.2um NOUT=32768 TILE=1024 LEG=auto WF=4.0 OTEG=error`.  The runner now
carries its own `__main__` guard, so the driver runs it under `run_name`
`'__main__'` and keeps spawn-safety through its OWN guard.

| run | grid | chain B | whole-run wall | peak RSS | samples |
|---|---|---|---|---|---|
| **r2a** | `n_fine_cap=8192` | **170 s** | 187.5 s | 37.20 GB | 11 485 @ 100 Hz |
| **r2c** | `n_fine_cap=12288` | **378 s** | 396.5 s | 43.31 GB | 24 201 @ 100 Hz |
| r2b | `n_fine_cap=16384` | -- | REFUSED at 8.5 s | -- | -- |

Both completed runs printed `VERDICT: PASS` and the frame row
`(-4,-2) ... 0.99331 0.99993 3.800 90.1 99.9 100.0` -- identical to the
capstone's and to the audit's, so the profiled run is the physics run.

*Honesty note on the second instrument.*  py-spy was attached to r2a for the
last 62 s only (it was started mid-run and the process exited); that window is
the readout tail, not the order, so **it is not used as a cross-check here**.
The audit's two-instrument agreement (57.75 % vs 59.05 % on `_poly`) is what
established that this sampler does not bias the ranking; nothing about the
sampler changed.

### 1.2 THE NEW TOP 10  (self time, innermost Python frame, lines merged)

`n_fine_cap=8192`, 11 342 attributed samples.  "self" means the sample's
innermost Python frame, so native time (numpy / scipy / FFT) lands on the
Python call site that entered it.

| # | % of order | site | what it is |
|---|---|---|---|
| **1** | **23.52** | `_lens_traced.py:_ResidualEikonal._poly` | the residual-eikonal potential, already power-cached and Hessian-free |
| **2** | **9.87** | `scipy/ndimage/_interpolation.py:map_coordinates` | ALL resampling (9 call sites -- sec 3.1) |
| 3 | 6.73 | `_lens_traced.py:apply_real_lens_traced` (self) | the element's own full-grid arithmetic |
| 4 | 6.52 | `carrier.py:_tilt_exactness_phase` | complex `exp` over n_fine^2 |
| **5** | **4.07** | `numpy/fft/_pocketfft.py:_raw_fft` | the RAW `np.fft` calls -- **items 1 + 4a** |
| 6 | 3.90 | `_lens_traced.py:_pip_residual_ri` | the de-chirp + unit-phasor build |
| 7 | 3.54 | `fan_multi_121.py:<module>` | the runner's own recombination |
| 8 | 2.85 | `carrier.py:_tilt_ramp` | complex `exp` over n_fine^2 |
| 9 | 2.66 | `_lens_traced.py:_carrier_residual_rms` | the F1 collimation guard |
| 10 | 2.55 | `lenses.py:surface_sag_general` | ray-trace sag |
| -- | 2.18 | `numpy/lib/_function_base_impl` -> `numeric.py:roll` | `fftshift`/`ifftshift`, 0.92 of it under `_fourier_upsample_crop` |

By phase: **fine retrace 69.05 %**, exact readout 12.18 %, everything else
(chain A, the six coarse groups, the gap ASM legs, the acceptance metrics)
18.76 %.

### 1.3 The 16384 arm: REFUSED by this branch's own guard

`n_fine_cap=16384` is still `fan_multi_121.py`'s default, and on this box the
branch's `_grid_intent.preflight` now refuses it:

```
clamp ceiling : 8192  (16384 needs 189.0 GB of budget; the box gives 94.48 GB)
modelled peak : 1 x 97.5 GB per worker + 0.0 GB parent = 97.5 GB
                (free 94.4 GB, reserve 8 GB)
REFUSED -- the box cannot hold this run
```

With `RAMB=inf` the *clamp* is disabled but the *box* check still refuses, and
no `RAMRES` value can satisfy it (97.5 > 94.4 for any reserve >= 0).  That is
the re-priced, MEASURED model of `FIX_PERF_PARALLEL_2026_08_10` sec 3 doing
exactly what it was built to do -- but it means **the fan's shipped default is
not runnable on tesla-ryzen**, which is a finding this document records and
does not resolve.

The confirmation arm is therefore **12288**, the largest grid the guard
approves.  It answers the question the 16384 arm was for -- does the profile's
SHAPE hold at a larger grid:

| site | 8192 | 12288 | delta |
|---|---|---|---|
| `_poly` | 23.52 | 23.47 | -0.05 |
| `map_coordinates` | 9.87 | 10.56 | +0.69 |
| `_tilt_exactness_phase` | 6.52 | 7.29 | +0.77 |
| `apply_real_lens_traced` (self) | 6.73 | 6.57 | -0.16 |
| `_raw_fft` | 4.07 | 4.29 | +0.22 |
| `_pip_residual_ri` | 3.90 | 3.75 | -0.15 |
| `roll` (fftshift) | 2.18 | 2.43 | +0.25 |
| fine-retrace phase | 69.05 | 74.31 | +5.3 |

Same ranking, every top-six share inside 0.8 points.  The share structure is a
property of the code, not of the grid -- which is the same conclusion the audit
reached comparing its 16384 and 8192 arms (57.75 % vs 54.82 %).

### 1.4 CLASSIFYING THE NEW #1

`_poly` is reached by exactly one path, measured: `_pip_residual_ri` (line
7590) -> `value` -> `_eval` -> `_poly` carries **25.09 %** of the order
inclusive; the next-largest caller of `value` is 0.09 %.  It is evaluated ONCE
per order, over the whole fine grid, in 4.19 Mpt row bands.

**Verdict: CAMPAIGN-SIZED (report only), with one byte-identical sliver taken
as item 4b.**  The reasoning, priced:

* **The free win is already taken.**  57.75 % / 3.78 = 15.3 % of the OLD
  order's wall, which is 26.6 % of the new (shorter) order; measured 23.5 %.
  The two agree, so nothing has been left on the table by the power cache.
* **What remains is arithmetic, not overhead.**  At degree 6 the shipped term
  list is 27 terms and the value path accumulates three of them (`P`, `Pu`,
  `Pv`) -- ~5.4 G multiply-adds over a 8192^2 grid, plus 14 `np.power` passes.
  The leaf lines confirm it: `:5180` (P) 6.49 %, `:5182` (Pu) 5.40 %, `:5184`
  (Pv) 5.25 %, `:5161`/`:5162` (the power tables) 6.00 %.
* **Every route past it moves bits.**  Horner / Clenshaw in `u` and `v` changes
  the summation order (the audit measured its own repeated-multiply variant V3
  at 2.9e-10 relative -- fine as a bound, fatal to the byte-identity contract
  `test_niche_perf_poly_locals.py` now pins).  A fused numba kernel changes
  FMA contraction.  Restricting the gradient to the frozen annulus is
  bit-identical in principle but does not pay: **74 % of the fine grid IS
  outside the freeze radius** on this configuration (r_fit ~ 3.6 mm inside a
  half-width-6.25 mm square), so the gather would cover three quarters of the
  points and add scatter on top.
* Each of those is a design decision plus a new identity contract plus its own
  fail-before -- i.e. a campaign, not an item in this round.

---

## 2. ITEM 1 -- `_fourier_upsample_crop` through the FFT dispatcher

### 2.1 What was wrong

`carrier._fourier_upsample_crop` called RAW `np.fft.fft2` / `np.fft.ifft2`
(`carrier.py:3203`, `:3214` pre-change), i.e. single-threaded pocketfft, while
every other transform in the library goes through `_fft2` / `_ifft2` (pyFFTW
with a cached plan on `FFTW_THREADS=8`, scipy.fft next, numpy last).  It runs
TWICE per exact final leg -- retrace and readout -- at the FINE grid.

MEASURED share on the fresh profile (`n_fine_cap=8192`): the function is
**3.69 % inclusive** (2.50 % under `_fine_trace_group_exit:6426`, 1.19 % under
`carrier_referenced_exact_focus_readout:4595`), of which **2.51 %** is
`_raw_fft` and 0.92 % the surrounding `fftshift`.

### 2.2 MEASURED -- A/B against the pre-change snapshot

`scratchpad/perf_round2/probe_items.py`, two package snapshots differing in the
two edited files only (`base/` restored from `ebc248f` with `git show`).  Min
of 3, plan warmed, quiet box.

| n_in / n_crop / n_fine | BEFORE (raw `np.fft`) | AFTER (dispatcher) | ratio |
|---|---|---|---|
| 1024 / 1024 / **8192** | 5.175 s [5.175-5.517] | **2.802 s** [2.802-2.914] | **1.85x** |
| 1024 / 1024 / **12288** | 10.599 s [10.599-11.283] | **6.354 s** [6.354-6.464] | **1.67x** |
| 1024 / 512 / **8192** | 4.889 s [4.889-5.093] | **2.768 s** [2.768-2.870] | **1.77x** |

(The ratio is below the audit's 5.04x pyFFTW-vs-numpy row because that row
timed the transform alone; here the `fftshift` pair, the spectral pad and the
value-preserving rescale are all inside the measurement and none of them got
faster.)

### 2.3 ACCURACY -- bounded, and measured, not asserted

This is the one class of change in this round that is not bit-identical:
pyFFTW and pocketfft are different implementations.  Same input, both arms:

| shape | rel L2 | max abs (rel to peak) | power ratio |
|---|---|---|---|
| n_crop 1024 -> n_fine 8192 | **4.353e-16** | 1.090e-15 | **1.000000000000000** |
| n_crop 1024 -> n_fine 12288 | **4.719e-16** | 1.237e-15 | **1.000000000000000** |
| n_crop 512 -> n_fine 8192 | **2.776e-16** | 9.986e-16 | **1.000000000000000** |

Envelope-accept: the bound is <= **5e-16 relative**, i.e. eleven orders inside
the chain's own 4e-5 energy honesty and inside the 1e-15 the instruction set.

### 2.4 The identity that IS asserted

Not bit-identity of the answer -- bit-identity of everything except the
backend.  `test_upsample_crop_byte_identical_with_the_dispatcher_pinned_to_numpy`
pins `fft_infra.USE_PYFFTW = USE_SCIPY_FFT = False` (the documented escape
hatch, which routes the dispatcher back to `numpy.fft`) and requires
`tobytes()` equality against a VERBATIM copy of the pre-change expression, at
three crop/fine combinations including the DOWNSAMPLE branch.  That is what
proves the edit moved the backend and nothing else -- no shift, no scale, no
re-ordered spectrum.

Two further pins, because the first two would each pass on a function that had
quietly reverted:

* `test_upsample_crop_actually_calls_the_dispatcher` counts `_fft2` / `_ifft2`
  and requires exactly one of each;
* `test_upsample_crop_result_does_not_alias_a_plan_workspace` issues three more
  transforms at the same key afterwards and requires the first answer to be
  unchanged -- the dispatcher can return one of two ping-pong workspaces, and
  the reason this call site does not need a defensive copy is that `fftshift`
  is `np.roll` and always allocates.

### 2.5 One behaviour that had to be PRESERVED, not inherited

numpy's FFT is double-only: `np.fft.fft2` returned complex128 for every input
dtype.  The dispatcher's pyFFTW / scipy backends PRESERVE complex64, so a
straight substitution would silently narrow a complex64 caller's output.  The
function now promotes to complex128 before the transform (a no-op on the
shipped complex128 path), pinned by
`test_upsample_crop_keeps_numpys_double_only_output_dtype`, which also checks
the no-transform branch still returns the input dtype.

---

## 3. ITEM 2 -- the `map_coordinates` site

### 3.1 THE CENSUS -- which sites, and does the NaN guard apply at all?

The audit could not say whether the NaN-mask guard would ever fire on design
121: "whether the NaN-pass guard applies on design 121 was not measured"
(its sec 11 item 5).  A `map_coordinates` census over a whole order, taken by
patching `scipy.ndimage` in the driver (`prof3.py`), settles it -- **and the
answer is NO.**

**A RETRACTION FIRST, because the first census read the opposite.**  It
recorded, per call, whether the ARRAY PASSED TO `map_coordinates` contained a
NaN, and reported `False` at all nine sites.  That answers the wrong question:
the OPL site is handed `np.where(np.isnan(opl_coarse), 0.0, opl_coarse)` and
the mask site is handed `np.isnan(opl_coarse).astype(float64)` -- both are
NaN-FREE BY CONSTRUCTION whatever `opl_coarse` holds.  The census was
re-taken recording each input's MAX instead: for a 0/1 mask, `max == 0.0`
means the coarse array it came from was NaN-free (the guard fires) and
`max == 1.0` means it was not.

**Re-taken, `n_fine_cap=8192`, whole order, PRE-change arm** (`_out_r2e.txt`;
line numbers are the post-edit ones):

| site | input | order | mask MAX | reading |
|---|---|---|---|---|
| `:10094` **OPL NaN mask**, coarse groups (256^2) | 7 calls | 1 | **1.0** | NaN PRESENT |
| `:10094` **OPL NaN mask**, fine retrace (95^2 -> 8192^2) | 1 call | 1 | **1.0** | NaN PRESENT |
| `:10177` **ray-density NaN mask**, coarse groups | 7 calls | 1 | **1.0** | NaN PRESENT |
| `:10177` **ray-density NaN mask**, fine retrace | 1 call | 1 | **1.0** | NaN PRESENT |
| `:10084` OPL upsample (not a mask) | 8 calls | 3 | 1.2e-3 .. 7.3e-5 | -- |
| `:10168` ray-density amplitude (not a mask) | 8 calls | 1 | 0.033 .. 0.177 | -- |

So `opl_coarse` and `ard_coarse` DO carry NaN on design 121 -- from
`_invert_newton`'s own out-of-domain policy (`np.where(out_of_domain, np.nan,
...)`), which on the fine retrace fires on part of a 95^2 lattice whose Newton
solve the run itself reports as 21.6 % non-converged.

**Confirmed independently by the AFTER profile.**  Both mask sites still
execute on the changed tree and at unchanged absolute cost: `:10094` 1.36 %
of a 149 s order (1.23 % of a 170 s order before) and `:10177` 1.35 % (1.20 %
before) -- i.e. 2.03 s and 2.01 s after against 2.09 s and 2.04 s before.  If
the guard had fired, those rows would be zero.

**Therefore item 2's NaN half is worth 0 on the workload of record.**  It is
shipped anyway: it is bit-identical by construction, it costs one
`.any()` on a 95^2 array, and it IS worth 1.53-1.57x of the upsample on the
configurations where the ray fit covers its own hull -- which the round-2 test
fixture is, and which is why that test can assert the guard fires at all.  It
is excluded from sec 7's compounded number.

### 3.2 What was implemented, and what was DELIBERATELY NOT

**(a) The NaN pass is skipped when the coarse array is NaN-free.**  With no
NaN, the interpolated mask is identically zero, `nan_full > 0.5` is identically
False, and the consuming `np.where` is the IDENTITY -- so the skip is
bit-identical BY CONSTRUCTION, not to a tolerance.  Applied at all three
routes that carry a mask: the whole-grid OPL upsample, the ray-density
amplitude, and the row-banded assembly (so banded and whole-grid stay
element-identical to each other as well as to their pre-guard selves).

MEASURED on the site, at the REAL shapes the census reports (min of 3).  This
is what the guard is worth WHERE IT APPLIES -- which, per sec 3.1, is not
design 121:

| coarse -> fine | WITH the NaN pass | WITHOUT | ratio | byte-identical |
|---|---|---|---|---|
| 94^2 -> 8192^2 (sub 87.15) | 6.016 s | **3.936 s** | **1.53x** | **True** |
| 94^2 -> 12288^2 (sub 130.72) | 13.523 s | **8.592 s** | **1.57x** | **True** |
| 256^2 -> 1024^2 (sub 4) | 0.091 s | **0.059 s** | **1.53x** | **True** |

(The audit projected 1.73-1.83x; it measured on a synthetic whose coarse array
was `N/sub` square.  At the real 94^2 the cubic prefilter is negligible and the
two passes are closer in cost, so the honest number is 1.53-1.57x.)

**(b) The coordinate stack is built straight into its buffer.**
`np.indices((N, N), float64)` + `np.array([ii / sub, jj / sub])` materialised
the (2, N, N) index pair, then two more full-grid quotients, then the result --
a ~12.9 GB transient for a 4.295 GB answer at `n_fine = 16384`.  It is now one
`arange(N)/sub` broadcast into a pre-allocated `(2, N, N)`.  BIT-IDENTICAL:
`np.indices(..., float64)` holds exact integer-valued float64s, so this is the
same IEEE division of the same two operands.

| N, sub | old | new | ratio | byte-identical |
|---|---|---|---|---|
| 8192, 87.149 | 797.2 ms | **200.6 ms** | **3.97x** | **True** |
| 12288, 130.723 | 1877.8 ms | **465.5 ms** | **4.03x** | **True** |
| 1024, 4 | 10.8 ms | **3.4 ms** | 3.18x | **True** |

**(c) The audit's cross-call coords CACHE is NOT taken, on measurement.**  Its
ranked row 7 offered 1.10-1.19x of the upsample.  On the real order the ENTIRE
build measures **0.30 %** of the wall (profile leaf `:9833`), so the cache's
ceiling is 0.3 % -- while a live cache would retain **4.295 GB** of full-grid
float64 at `n_fine = 16384` for the rest of the order, on a branch whose
companion item just FREED 34.9 GB of exactly that shape
(`FIX_PERF_POLY_LOCALS_2026_08_09` sec 3.3).  Item (b) takes 3/4 of the same
time at negative memory cost and keeps bit-identity, so it is strictly better
here.  Recorded as a deviation from the instruction, with the number that
drove it.

### 3.3 The identity, asserted at the element

`test_nan_pass_guard_is_byte_identical_and_actually_fires` runs the whole
element on a design-121-like fixture (N = 1024, carrier sphere, ray-density,
`preserve_input_phase='remap'`, `remap_sampling='full'`, sub 4) TWICE: once as
shipped, once with the guard DEFEATED by a shim whose `isnan` result reports
`.any() == True` while its VALUES stay all-False -- i.e. exactly the
pre-change work.  The exit field must be `tobytes()`-equal, AND the shipped
run must have made strictly fewer `map_coordinates` calls, so the identity
cannot be vacuous.

---

## 4. ITEM 3 -- the Newton pool payload, shipped once per worker

### 4.1 The defect, restated with the audit's measurement

`_invert_newton_parallel` built one arg tuple per chunk with `_spline_data`
INSIDE each, so the executor pickled the same payload once per chunk.  The
audit measured the consequence: a bare 8-worker round trip is 1.5 ms; the same
round trip with the real payload is 173.1 ms, i.e. **99.2 % of the dispatch
constant is re-pickling the payload**.

### 4.2 What was implemented

Two independent halves, plus one thing that had to be got right.

1. **The parent pickles ONCE.**  `_newton_payload_blob` does a single
   `pickle.dumps`; what crosses the wire per chunk is a `bytes`, whose own
   pickling is a memcpy.
2. **Workers KEEP the last payload**, in a module-level registry cleared by the
   pool `initializer` (`_newton_pool_init`, now passed to
   `ProcessPoolExecutor`).  A later dispatch whose payload digests the same
   sends the KEY ALONE.
3. **The key is a CONTENT DIGEST** (blake2b-128 of the wire bytes), not an
   identity or a counter.  This is the safety argument, and it is not
   cosmetic: the payload dict is rebuilt per `apply_real_lens_traced` call but
   MUTATED per dispatch -- `cheb_backend` and `cheb_fit` are stamped in
   immediately before every dispatch.  An `id()`- or counter-keyed cache would
   let a worker answer from a payload whose pinned backend or built fit had
   since changed, which is a silently different floating-point order -- the
   exact failure class v5.32.3 and v5.33.0 exist to remove (measured then at
   5.2e-14 local / 1.358e-11 CI).  A digest of the bytes cannot do that.

**Residency is never load-bearing for correctness.**  `ProcessPoolExecutor`
does not promise that every worker takes a chunk from a given dispatch, so a
worker asked for a key it does not hold raises `NewtonPayloadNotResident` and
the parent re-submits THAT chunk with the blob attached.  The worst case is
therefore the pre-change behaviour for one chunk.

**Every named semantic is preserved.**  Spawn is still forced; the pool is
still persistent and still rebuilt only on a worker-count change; the atexit
registration is still once per process; the cost gates
(`_POOL_MIN_PIXELS` / `_POOL_MIN_PIXELS_WARM` / `_POOL_PROMOTE_MIN_SECONDS`
/ `_POOL_PROMOTE_MIN_SAMPLES` / `_POOL_PROMOTE_SIZE_RATIO`) are untouched, as
is the memory clamp and the `__main__`-guard refusal; the backend pin and the
shipped-fit stamp still happen BEFORE the payload is frozen (re-pinned, sec
6.2); and a worker holds AT MOST ONE payload, so per-worker resident cost is
bounded by the largest payload -- which a chunk held for its own duration
anyway.

### 4.3 MEASURED

`scratchpad/perf_round2/probe_pool2.py`: the audit's own method -- an ECHO
worker, so what is timed is the arg tuple's journey and not the Newton work --
driven through the LIBRARY's `_newton_payload_blob` / `_newton_worker_payload`.
65 536 points split across the workers, warm pool, min of 5.

| payload | workers | bare | BASE (payload per chunk) | payload share | NEW, first dispatch | NEW, resident |
|---|---|---|---|---|---|---|
| 3.599 MB | 4 | 0.8 ms | 25.0 ms | 96.6 % | 23.2 ms (1.08x) | **7.7 ms (3.24x)** |
| 3.599 MB | 8 | 1.7 ms | 46.8 ms | 96.3 % | 38.0 ms (1.23x) | **7.6 ms (6.15x)** |
| 6.773 MB | 4 | 0.8 ms | 43.8 ms | 98.2 % | 42.5 ms (1.03x) | **12.4 ms (3.53x)** |
| **6.773 MB** | **8** | 1.4 ms | **86.4 ms** | **98.4 %** | 70.5 ms (1.23x) | **12.7 ms (6.80x)** |

Read honestly:

* **6.80x on the steady-state dispatch constant** at 8 workers, and 1.23x even
  on a cold one (the single `dumps` replaces eight).
* The residual 12.7 ms is **the parent's own one `dumps` + digest**, not
  transport -- which also bounds what any further caching could buy here.  It
  is left in deliberately: removing it would require trusting that the payload
  has not changed since the last dispatch, which is the assumption sec 4.2
  item 3 refuses to make.
* **This is worth ZERO on design 121**, and the audit already said so: the fan
  dispatches the pool **never** (65 536 pts/group is below the 200k cold bar
  and the default polynomial path's 0.048 s step is below the 0.35 s promote
  bar; the fine retrace inverts 9 025 points).  It is shipped for the LIBRARY
  -- optimisation and tolerancing runs that call the element hundreds of times
  are where a per-dispatch constant matters -- and it is excluded from sec 7's
  compounded number.

### 4.4 Bit-identity

Unchanged by construction: the worker receives a payload that unpickles to the
same values it was previously handed directly, and nothing else about
`_newton_invert_chunk` moved.  Observed rather than argued by the repo's own
pins, green on this tree:
`test_niche_newton_pool_both_fits.py::test_pool_result_is_bit_identical_to_serial`
for BOTH fits, plus its warm-tier sibling.  The round-2 module adds the
protocol pins: the key moves when any field moves (six fields checked
individually), a worker refuses a key it does not hold, a second payload evicts
the first, and the 3-tuple and 4-tuple arg shapes give `tobytes()`-equal
answers.

---

## 5. ITEM 4 -- the two the FRESH profile named

The instruction allowed up to two additional items, each < ~50 lines and
byte-identical or bounded <= 1e-15.  The profile named both.

### 5.1 Item 4a -- `carrier._shift_envelope`, the OTHER raw-`np.fft` site

The `_raw_fft` leaf splits by caller: 2.51 % under `_fourier_upsample_crop`
and **1.37 % under `_shift_envelope`** (`carrier.py:5013` pre-change), which is
the band-limited sub-pixel translation `_crop_about_centre` uses to place the
chief-ray-centred readout window.  Inclusive, `_shift_envelope` is 1.80 % of
the order (1.18 % via `_crop_about_centre`, 0.62 % via the chain's two
axis-recentring calls).  Same defect, same fix, same accuracy statement:

| n | BEFORE | AFTER | ratio | rel L2 | power ratio |
|---|---|---|---|---|---|
| 2048 | 0.483 s | **0.273 s** | **1.77x** | 4.257e-16 | 1.000000000000000 |
| 4096 | 2.003 s | **1.056 s** | **1.90x** | 4.558e-16 | 1.000000000000000 |
| 8192 | 8.037 s | **4.069 s** | **1.98x** | 4.996e-16 | 1.000000000000000 |

Pinned the same three ways (numpy-pinned byte-identity, bounded deviation, no
workspace aliasing).  One extra care: the pre-change return was
`out.astype(e.dtype, copy=False)`, which for an already-complex128 input
returned the transform's own buffer -- harmless when that buffer came from
`np.fft`, an aliasing bug when it can come from the plan cache.  The function
now copies explicitly, and the test issues three more transforms at the same
key and requires the first answer to be unchanged.

### 5.2 Item 4b -- `_poly` still builds `u**0` and `u**1`

The power cache that shipped on 2026-08-09 issues one `np.power` per DISTINCT
exponent -- including exponent 0 (a whole full-grid array of ONES, whose only
use is a multiply by exactly 1.0) and exponent 1 (a whole full-grid COPY of
the operand).  On the shipped degree-6 list that is 12 of 27 terms carrying a
redundant full-array multiply in `P` alone, plus two allocations per axis.

Eliding both is **BIT-IDENTICAL**: multiplying a float64 by exactly 1.0 is
exact for every finite operand and preserves inf, nan and the sign of zero,
and `np.power(x, 1)` returns `x`'s bits.  Left-to-right association is
preserved: the shipped expression was `((c*i) * UP[p]) * VP[q]`, and dropping
a factor that is identically 1.0 cannot move a bit.

MEASURED on the real band shapes the fine leg uses (4.19 Mpt bands, 27 terms,
degree 6, value path):

| band | BEFORE | AFTER | ratio | byte-identical |
|---|---|---|---|---|
| 512 x 8192 | 2.405 s | **2.039 s** | **1.179x** | **True** |
| 256 x 16384 | 2.584 s | **1.976 s** | **1.307x** | **True** |

*(measured with a WSL test suite running on the same box, so both arms carry
the same contention and the RATIO is what is claimed, not the absolute
seconds.)*

Against `_poly`'s 23.52 % that is **3.6-5.6 % of the order** -- the largest
single item in this round, and the reason it was taken rather than merely
reported.  The bit-identity is asserted twice: by the round-2 module against a
verbatim copy of the PRE-ELISION loop (2-D band and scalar query, `hess` both
ways), and for free by `test_niche_perf_poly_locals.py`, whose `_RefEikonal`
reference predates both the power cache AND the elision and compares with
`np.array_equal` across degrees 1-6, sparse and all-zero coefficient vectors,
a decentred model and the cold 1-D / scalar callers.

---

## 6. GREEN

All at `OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = MKL_NUM_THREADS =
NUMEXPR_NUM_THREADS = 1`, `-p no:randomly`.  Nothing was xfailed, skipped or
deselected by this change.

### 6.1 The suites

| suite | where | result |
|---|---|---|
| **`test_niche_perf_round2_2026_08_10.py`** (new: 26 tests) + `test_niche_perf_poly_locals.py` + both Newton-pool modules + `test_perf_v4_12_0_fft_infra.py` + `test_audit_w2_fft_state.py` + `test_niche_k3_perf.py` + `test_verify_perf_fixes_2026_08_10.py` + **`test_niche_tight_focus_readout.py`, `test_carrier_referenced.py`, `test_niche_d2_chain_multi.py`, `test_niche_d6_exact_tilted_leg.py`** | Windows, py 3.14.6 / numpy 2.4.4 | **276 passed**, 12:07 |
| the same set minus the four heavy traced modules | **WSL**, `/home/travaj/lumen_venv`, py 3.12.3 / **numpy 2.4.6** (plus `test_carrier_referenced.py`) | **179 passed**, 2:32 |
| `ruff check` on both changed library files and both test files | Windows | All checks passed |

Two intermediate greens are recorded because they bound WHICH change each one
covers: 109 passed on the four heavy traced modules and 158 passed on WSL
after items 1-3 + 4a but BEFORE item 4b; the rows above are the full set after
all five.

**THE WSL ARM EARNED ITS KEEP, AND IT CAUGHT A TEST BUG.**  The first run of
`test_poly_exponent_elision_is_byte_identical` passed on Windows and FAILED on
WSL / numpy 2.4.6 at `max|d| = 4.337e-19`.  The library was not at fault: the
test drove the shipped method with ``u * s`` while driving its reference with
``u``, so the method's own ``ex / s`` made the two arms see DIFFERENT inputs --
``(u * s) / s != u`` for 1 892 of 12 707 random float64s on this box, measured.
Windows passed by luck.  The test now derives ``u = ex / s`` in both arms from
one ``ex``.  Recorded rather than quietly fixed, because "passed on one
platform" was exactly the wrong conclusion to draw.

### 6.1b FAIL-BEFORE, executed

| # | claim | fail-before | pass-after |
|---|---|---|---|
| 1 | the FFT items changed only the backend | with the dispatcher pinned to `numpy.fft`, `tobytes()` equality against a verbatim pre-change expression at 3 crop/fine combinations incl. the downsample branch | equal; with the shipped backend, rel L2 <= 5e-16 |
| 2 | the identity pins are not vacuous | a 1-ulp perturbation inside `_mul` (`* 1.0000000000000002` on one branch) makes **17 tests FAIL** across `test_niche_perf_round2` and `test_niche_perf_poly_locals` -- executed, then reverted | unperturbed: all pass |
| 3 | the NaN guard is not vacuous | the element-level test requires the shipped run to have made strictly FEWER `map_coordinates` calls than the guard-defeated run | it does, and the exit field is `tobytes()`-equal |
| 4 | the dispatcher is really being called | `_fft2` / `_ifft2` call counts == 1 each | 1 each |
| 5 | the per-item speeds | every BEFORE row in secs 2, 3, 5 is measured on the `base/` snapshot in-session, not quoted from the audit | AFTER rows on the working tree |

The new module's own teeth, listed because a green count says nothing about
whether the assertions bite:

* the dispatcher tests fail if the function reverts to `np.fft` (an explicit
  `_fft2` / `_ifft2` call count);
* the NaN-guard test fails if the guard saves no calls on its fixture
  (`cnt_off.n > cnt_on.n`), so its byte-identity cannot be vacuous;
* the elision test compares against a VERBATIM copy of the pre-elision loop,
  and `test_niche_perf_poly_locals.py`'s `_RefEikonal` compares against the
  pre-power-cache one -- two independent frozen references over the same code;
* the payload-key test asserts SIX different fields each move the key
  individually, which is the whole safety argument for reuse.


### 6.2 The two re-anchored source pins, and why that is not a weakening

`test_fix_newton_pool_memory.py` carried two ordering pins written as
`body.index("_spline_data['cheb_backend']") < body.index('args_list = [')` and
the same for `cheb_fit`.  `args_list = [` was the point at which the payload
was frozen for the wire -- the comprehension that embedded the dict in every
chunk tuple.  Item 3 replaces that comprehension, so the anchor had to move to
the new freeze point, `_newton_payload_blob(_spline_data)`.

The invariant is **unchanged and, if anything, tightened**: a stamp landing
after the blob is taken is invisible to every worker, exactly as a stamp
landing after the chunks were built used to be.  The re-anchored pins also
assert the anchor EXISTS (`'_newton_payload_blob(_spline_data)' in body`), so
a future edit that removes the freeze cannot make the ordering assertions
vacuously true -- which the old form could not detect.

---

## 7. COMPOUNDED -- the per-order wall, before and after

### 7.1 The A/B

Same order, same configuration, same box, `PROF=0` (no sampler), two reps per
arm, arms interleaved.  `ARM=base` runs the pre-change package snapshot; the
two arms differ in exactly two files.

| arm | rep 1 | rep 2 | MIN | spread |
|---|---|---|---|---|
| **BEFORE** (`ebc248f`) | 160 s | 167 s | **160 s** | 4.4 % |
| **AFTER** (this round) | 158 s | **144 s** | **144 s** | 9.7 % |

**MIN to MIN: 160 s -> 144 s = 1.111x, -10.0 % of chain B.**  Every AFTER run
beat every BEFORE run (AFTER max 158 s < BEFORE min 160 s), which is what the
overlap check has to say when the arms' own spreads are 4-10 %.

A third, independent pair at MATCHED sampler overhead (`PROF=1`, one rep each
-- runs r2a and r2d): **170 s -> 149 s = 1.141x, -12.4 %**.

Both are consistent with the per-site arithmetic, which predicts **-9.7 %**
from items 1 + 4a + 4b + the coords build (sec 7.2) -- item 2's NaN half
contributing nothing here (sec 3.1) and item 3 nothing at all (sec 4.3).

### 7.2 Where the wall went -- BEFORE and AFTER profiles, same instrument

Runs r2a and r2d, both `PROF=1` at 100 Hz, whole-run walls 187.5 s and 157.9 s.
Shares are of the whole run; the absolute column is share x that wall.

| site | BEFORE share | AFTER share | absolute | ratio |
|---|---|---|---|---|
| `_raw_fft` (RAW `np.fft`) | **4.07 %** | **0.19 %** | 7.63 s -> **0.30 s** | 25x |
| `_fourier_upsample_crop` (incl.) | 3.69 % | 1.95 % | 6.92 s -> **3.08 s** | 2.25x |
| `_shift_envelope` (incl.) | 1.80 % | 0.98 % | 3.38 s -> **1.55 s** | 2.18x |
| `_poly` (+ its new `_mul` frame) | **23.52 %** | 10.55 + 9.75 = **20.30 %** | 44.10 s -> **32.05 s** | **1.38x** |
| coords build (`np.indices` + list) | 0.30 % | absent | 0.56 s -> ~0.14 s | ~4x |
| `map_coordinates` | 9.87 % | 11.32 % | 18.51 s -> 17.87 s | **~1** (sec 3.1) |

Summed over the four items that moved: **-18.1 s of a 187.5 s run = -9.7 %**,
against a measured -10.0 % (PROF=0, min of 2, chain B) and -12.4 % (PROF=1,
chain B).  The per-site arithmetic and the end-to-end wall agree.

Note the in-situ `_poly` ratio (1.38x) is at the top of the microbenchmark's
1.18-1.31x range, and the in-situ `_fourier_upsample_crop` / `_shift_envelope`
ratios (2.25x / 2.18x) are ABOVE the probe's 1.85x / 1.98x.  That is a
one-rep-each comparison of two sampled runs, so a few points of it is
sampling; it is reported as it was measured rather than trimmed to the probe.

The only raw-`np.fft` left anywhere on the order is
`_gap_envelope_angular_spread` at 0.19 %, which is a diagnostic, not
transport.

### 7.3 Memory and the plan cache, measured

| | BEFORE | AFTER |
|---|---|---|
| whole-run peak RSS (`n_fine_cap=8192`) | 37.78 GB | **35.40 GB** |
| pyFFTW plan-cache entries | 5 | **8** (the LRU's own limit) |
| pyFFTW resident buffers | 4.56 GB | **5.61 GB** (+1.05) |

Two things to state plainly rather than bury:

* the RSS drop is not claimed as a result -- it is one rep of a high-water
  mark, and the coords build's 8.6 GB transient at `n_fine = 16384` is
  arithmetic on the shape, not a measurement (sec 8 item 1);
* **routing two more sites through the dispatcher fills the plan cache.**  It
  now holds 8 of 8 entries and the `('fwd', (1024, 1024))` key -- the chain's
  gap ASM legs -- is evicted at least once in a run.  The cost is a re-plan
  plus a 2 x 16 MB aligned allocation, far below the 6.6 s the two items buy,
  and it is recorded here because the LRU is now saturated: the NEXT site
  routed through the dispatcher will thrash it.


---

## 8. WHAT THIS DOES NOT CLOSE

1. **`n_fine_cap=16384` was never run.**  The branch's own box guard refuses it
   on this box (sec 1.3), so the profile's grid-independence is established
   across 8192 and 12288, not 8192 and 16384.  Every 16384-scaled figure in
   this document (memory transients, plan-buffer sizes) is arithmetic on the
   shape, not a measurement.
2. **One order, not thirty-two.**  Same limit the audit and the capstone
   carry: `(-4,-2)`, the historical worst case.
3. **The fan's shipped default is not runnable on this box.**  Recorded in sec
   1.3; not resolved here, and it interacts with
   `ADJUDICATION_NFC_8192_2026_08_10`'s "16384 stays the record default".
4. **Item 3 is worth zero on design 121** and is measured on a transport
   probe, not on a real Newton dispatch -- because the fan makes none (sec
   4.3).  Its bit-identity rests on the repo's existing pool==serial pins,
   which are green, not on a new end-to-end comparison.
5. **The `_poly` campaign is priced, not attempted** (sec 1.4).  The three
   routes named there each need their own identity contract; none is in this
   round.
6. **The plan cache grows.**  Routing two more call sites through the
   dispatcher adds pyFFTW plan-buffer keys at the crop and fine sizes.  The
   measured parent-side census is in sec 7; at `n_fine_cap=16384` the same
   keys are 4x larger by shape, and that is arithmetic, not a measurement.
7. **A CONCURRENT SESSION COMMITTED THIS WORK.**  While these edits were in the
   working tree, another agent on the same checkout made commit `a6bcc46`
   ("fix(ci): enroll `_PYFFTW_PLAN_MAX_BYTES_PER_BUFFER` ..."), which swept in
   295 lines of `_lens_traced.py` and 80 of `carrier.py` belonging to THIS
   round alongside its own two-line change.  No git command that writes was run
   by this work; the collision is recorded so the branch history is not read as
   deliberate.  Every A/B here is measured against a package SNAPSHOT restored
   from `ebc248f` (`scratchpad/perf_round2/base/`, verified to contain zero
   occurrences of this round's marker), so the measurements are unaffected.

---

## 9. ARTEFACTS

All under the session scratchpad `scratchpad/perf_round2/`.

| file | what |
|---|---|
| `prof2.py` | the driver: runs `fan_multi_121.py` UNMODIFIED via `runpy` under an import-safe `__main__`, with a 1 Hz RSS sampler, a 100 Hz stack sampler, an `ARM=base` package switch and an end-of-run cache census |
| `prof3.py` | `prof2.py` + the `map_coordinates` census of sec 3.1 |
| `ana2.py`, `ana3.py` | fold a profile into phase / self-time / leaf tables; attribute a named leaf to its library call sites |
| `probe_items.py` | the `_fourier_upsample_crop` / `_shift_envelope` A/B and bound, the NaN-pass site, the coords build |
| `probe_pool2.py` | the Newton dispatch constant, BASE vs NEW transport |
| `probe_poly3.py` | the exponent-0/1 elision (item 4b) |
| `base/lumenairy/` | the PRE-CHANGE package snapshot (`git show ebc248f:` for the two edited files) |
| `_out_r2a.txt`, `_prof_r2a.txt`, `_mem_r2a.tsv` | r2a: one order, `n_fine_cap=8192` |
| `_out_r2c.txt`, `_prof_r2c.txt`, `_mem_r2c.tsv` | r2c: one order, `n_fine_cap=12288`, with the census |
| `_out_r2b.txt` | the refused 16384 arm |
| `_out_ab_{base,new}_{1,2}.txt` | the compounded A/B, two reps per arm |
