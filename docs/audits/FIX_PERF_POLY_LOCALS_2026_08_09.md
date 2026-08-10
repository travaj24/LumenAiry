# FIX -- `_ResidualEikonal._poly` power cache, and the traced element's full-grid local lifetimes

**2026-08-09.  Branch `perf/traced-hotpath`, from `c8bcbcb` (v5.33.1).
Implements items #1 and #2 of `AUDIT_TRACED_SPEED_2026_08_09.md` and
`AUDIT_TRACED_MEMORY_2026_08_09.md` -- the share of them that lives in
`lumenairy/elements/_lens_traced.py`.  ONE library file was edited
(`lumenairy/elements/_lens_traced.py`) plus one new test module
(`tests/unit/test_niche_perf_poly_locals.py`).  `carrier.py` was NOT touched:
the memory item's other half -- the five consumed phase factors in
`_fine_trace_group_exit` (`env_f`, `_ph`, `_cf`, `_rp`, `_xf`, 21.48 GB) and
the `_exact_sphere_eikonal` / `_envelope_amp_radius` / `_radial_carrier_phase`
meshgrids -- is a `carrier.py` change and is out of this document's scope.  No
git command that writes was run; `CHANGELOG.md` was not touched.**

---

## 0. HEADLINE

> **Item #1 (speed).  `_ResidualEikonal._poly` now issues ONE `np.power` per
> distinct exponent instead of one per term per accumulator, and `value()` --
> the consumer the audit measured at 57.8 % of a design-121 fan order's wall --
> takes a Hessian-free path.  MEASURED on the audit's own probe, the same box,
> the same shapes: `.value()` 9.709 s -> 2.568 s per 4.19 Mpt band,
> **3.78x** (the audit projected 3.81x).  Whole fine grid, 64 bands:
> 621.4 s -> 164.4 s.  The cold `_poly` / `grad()` callers, whose contract is
> unchanged, get **2.72x** for free.  BIT-IDENTICAL: `np.array_equal` True
> against a verbatim copy of the pre-change implementation, on the real
> design-121 term list at the shipped degree cap, in and out of the radial
> freeze.**
>
> **Item #2 (memory).  The element builds its wave-grid coordinate pair with
> `np.broadcast_to` instead of `np.meshgrid`, and frees TWELVE consumed
> full-grid locals at their last use.  MEASURED, exactly and deterministically: the fine
> leg's frame holds 40 arrays / 718.08 MB at return before, 26 arrays /
> 198.08 MB after, at `n_fine = 2048` -- and 2854.09 MB -> 774.09 MB at
> `n_fine = 4096`.  Both are **16.25 full-grid float64 equivalents**, i.e. the
> reduction is exactly quadratic in `n_fine`: **-34.9 GB at the shipped
> `n_fine = 16384`**, out of the 43.2 GB the memory audit measured this
> function holding.  The peak of LIVE bytes across the whole exact leg drops
> 1.4462 -> 1.2779 GiB at `n_fine = 2048` and 5.1374 -> 4.7467 GiB at 4096,
> and the memory-time integral drops 10-12 %.  BITWISE: the group exit field
> `_fine_trace_group_exit` returns has the SAME sha256 before and after, at
> both grids, and so does the chain's readout field.**

Nothing in the physics moved.  Both items are lifetime / call-count changes;
neither alters an arithmetic operation, an operand order, or a rounding.

---

## 1. BOX, BUILD, METHOD

```
Windows 11 Pro 10.0.26200        AMD Ryzen 9 5950X, 24 logical CPUs
python 3.14.6   numpy 2.4.4   scipy 1.17.1   psutil 7.2.2
lumenairy 5.33.1 working tree, branch perf/traced-hotpath
threads pinned: OMP_NUM_THREADS = OPENBLAS = MKL = NUMEXPR = 1
```

**The A/B is isolated from the concurrent `carrier.py` work.**  Every
before/after number below is taken between two package SNAPSHOTS that differ in
exactly one file:

```
scratchpad/perf_traced_lens/base/lumenairy/   git HEAD (c8bcbcb), all files
scratchpad/perf_traced_lens/new/lumenairy/    the same, with THIS branch's
                                              elements/_lens_traced.py only
```

`diff -rq` between the two reports exactly one differing file.  Both arms
therefore run the identical `carrier.py`, so nothing here is confounded by the
edits another agent is making to it.

Timing protocol: >= 3 reps, MINIMUM reported with the [min-max] spread.  Memory
is reported three ways, because they answer different questions and only one of
them is exact:

1. **Frame census at return** (`probe_census.py`) -- `sys.settrace` on the
   `'return'` event of `apply_real_lens_traced`, where `f_locals` is still
   populated, summing DISTINCT array bases.  Deterministic, exact, and
   sampler-artefact-proof; this is the audit's own "logical array size"
   method.
2. **Live traced bytes** (`probe_ab.py --tracemalloc`) -- `tracemalloc`, whose
   numpy-domain tracking was verified in-session (a 134.2 MB `np.zeros` moves
   the counter by 134.2 MB and back).  `reset_peak()` at entry to
   `_fine_trace_group_exit` gives the LEG's peak.
3. **Process peak RSS** -- reported for completeness and, on this fixture,
   *unchanged* (sec 3.5 says why, and why that is expected rather than a
   refutation).

---

## 2. ITEM #1 -- `_ResidualEikonal._poly`

### 2.1 What was wrong, restated from the code

`_poly` (pre-change, `_lens_traced.py:5097-5123`) held six accumulators and,
for every term, recomputed `u ** i` and `v ** j` from scratch in each of them:

```python
P   = P   + c * u ** i * v ** j
Pu  = Pu  + c * i * u ** (i - 1) * v ** j
Pv  = Pv  + c * j * u ** i * v ** (j - 1)
Puu = Puu + c * i * (i - 1) * u ** (i - 2) * v ** j
Puv = Puv + c * i * j * u ** (i - 1) * v ** (j - 1)
Pvv = Pvv + c * j * (j - 1) * u ** i * v ** (j - 2)
```

numpy has no fast path for integer exponents above 2, so each of those is a
libm `pow()` per element.  At the shipped degree cap 6 the term list is
`[(i, d - i) for d in 1..6 for i in 0..d]` -- 27 terms -- so the loop issues
~130 whole-array `pow` passes where **14 distinct exponents** exist.

And `value()`, which is the only consumer on the hot path
(`_pip_residual_ri`, `_lens_traced.py:7590`), reads `_eval(...)[0]`.  Outside
the radial freeze `a = P + (r - r1) * (gx*ux + gy*uy)`; inside it `a = P`.
Either way the three Hessian accumulators feed `ax`/`ay` alone.  Half the
arithmetic was discarded.

### 2.2 What it is now

* one `np.power` per DISTINCT exponent, into a dict keyed by exponent, with the
  exponent census taken from the same `c == 0`-skipped term set the
  accumulation loop uses (so a sparse or low-degree fit builds a smaller
  table, and `hess=False` never builds the exponents only the Hessian reads);
* accumulation with `+=`;
* `_poly(ex, ey, hess=False)` returns `(P, Pu/s, Pv/s, None, None, None)` --
  `None`, not a stale interior Hessian, so a consumer that reads the slot fails
  loudly instead of silently using the wrong thing;
* `_eval(xq, yq, need_grad=False)` returns `(a, None, None)` and skips both the
  Hessian and the whole tangential-Hessian temporary chain;
* `value()` calls that path.  `grad()` and the default `_poly(ex, ey)` /
  `_eval(x, y)` signatures are untouched.

**Why it is bit-identical, not "identical to round-off".**  The same
`np.power` calls on the same operands return the same bits; the operand order
inside each term is unchanged (`c * i * U[i-1] * V[j]` associates left to right
exactly as `c * i * u ** (i-1) * v ** j` did); `x += y` rounds identically to
`x = x + y`; and `a` in `_eval` is built by the same expressions from the same
operands whether or not the gradient is also built.

### 2.3 MEASURED -- the audit's own probe, before and after

`scratchpad/traced_speed/probe_poly.py` and `probe_poly2.py`, unmodified, on
the real band shape (256 x 16384 = 4.19 Mpt, the band `_pip_residual_ri`
itself uses), degree 6, 27 terms, min of 3, quiet box:

| | BEFORE | AFTER | ratio | whole fine grid (64 bands) |
|---|---|---|---|---|
| **`.value()`** -- the hot consumer | **9.709 s** [9.709-9.915] | **2.568 s** [2.568-2.586] | **3.78x** | 621.4 s -> **164.4 s** |
| `_poly(...)`, all 6 outputs | 9.346 s [9.346-9.473] | 3.433 s [3.433-3.458] | 2.72x | 598.1 s -> 219.7 s |
| `_poly(..., hess=False)` | (not available) | 2.229 s [2.229-2.338] | -- | 142.6 s |
| `.grad()` -- cold caller, contract unchanged | 10.007 s [10.007-10.112] * | 4.051 s [4.051-4.108] | 2.47x | -- |

\* the pre-change `grad()` and `value()` shared ONE `_eval` code path, so the
`probe_poly.py` row "SHIPPED `.value()` end to end" -- 10.007 s -- is that path
timed; `grad()` differs from it only in which of the tuple's three slots it
returns.

The audit projected 3.81x for this variant (its V4) and measured 3.83x in its
own probe harness; the shipped implementation lands at **3.78x**.  The two
probe rows are consistent: `probe_poly2.py`'s standalone V4 prototype read
2.535 s, the shipped method reads 2.568 s.

**Projection onto the workload of record**, using the audit's arithmetic and
its clean 910.9 s/order: `_poly` was 57.75 % of an order, now 57.75/3.78 =
15.28 %, so the order retains 0.4225 + 0.1528 = 0.5753 of its wall -- an order
goes **910.9 s -> 524.0 s** and the 32-order fan **8.12 h -> 4.67 h (1.74x)**.  That is a PROJECTION -- the whole fan has still
never been run end to end, exactly as the audit says (sec 11 item 1).  What is
MEASURED here is the site, at the real shapes, on the shipped class.

### 2.4 BIT-IDENTITY -- asserted, not argued

`tests/unit/test_niche_perf_poly_locals.py` keeps the pre-change `_poly` /
`_eval` / `value` / `grad` VERBATIM as `_RefEikonal` (copied, not imported, so
the reference stays frozen when the shipped one is edited again) and asserts
`np.array_equal` -- never `allclose` -- on:

| what | cases |
|---|---|
| `_poly`'s full 6-tuple | inside the freeze and outside it |
| `value()` | inside / outside |
| `grad()` | inside / outside |
| both | every degree 1..6 |
| both | a 50 %-zero coefficient vector, and the all-zero fit |
| both | a decentred model (`cx, cy != 0`; the niche D9 case) |
| `value` / `grad` | 1-D launch heights and a scalar query (the cold callers) |
| contract | `_poly(ex, ey)` still returns 6 populated arrays; `_eval(x, y)` still returns 3; `hess=False` returns `None` in the Hessian slots, and its first three outputs equal the `hess=True` ones bit for bit |

**These assertions bite.**  Substituting the audit's V3 variant (a
repeated-multiply power table, which the audit measured at max|delta| 1.07e-14,
rel 2.9e-10 -- i.e. *far* inside any tolerance) makes
`test_value_bit_identical` and `test_poly_six_outputs_bit_identical` FAIL, as
required.  Recorded in sec 5 (fail-before table, row 2).

### 2.5 Contract check -- every caller

`_poly` has exactly ONE caller in the library (`_eval`, `_lens_traced.py:5136`
pre-change) and `_eval` exactly two (`value`, `grad`).  The model's own
consumers are three, all cold and all on small 1-D launch-height arrays:

| site | call | affected? |
|---|---|---|
| `_lens_traced.py:7590` (`_pip_residual_ri`) | `.value(band, band)` | HOT -- this is the 57.8 % |
| `_lens_traced.py:8358` | `.grad(h_x, h_y)` | no: `grad` still asks for the Hessian |
| `_lens_traced.py:8498` | `.value(h_x, h_y)` | value-only, bit-identical (pinned) |

The default arities are preserved, so a caller that never heard of `hess` /
`need_grad` sees no change at all.

---

## 3. ITEM #2 -- the traced element's full-grid local lifetimes

### 3.1 Broadcast instead of meshgrid

`apply_real_lens_traced` built its wave-grid coordinate pair as
`X, Y = np.meshgrid(x, _y_ax)` -- two MATERIALISED full 2-D float64 arrays,
2 x 2.147 GB at `n_fine = 16384`, held for the whole call.  The memory audit's
census caught both live at the peak plateau (sec 2.3 / row 5).

They are now `np.broadcast_to(x[None, :], (N, N))` and
`np.broadcast_to(_y_ax[:, None], (N, N))` -- zero-copy read-only views with the
same elements.  Every consumer reads only: `X[::sub, ::sub]` (the coarse Newton
lattice), `X[mask_full]`, `X ** 2 + Y ** 2` (the aperture mask),
`np.asarray(Xw).ravel()` inside `_invert_fit` / `_invert_newton_parallel`, and
`X.shape` / arithmetic inside `_compute_carrier`.  Nothing writes through them,
and nothing could have: the `sub > 1` path already handed those same consumers
the STRIDED VIEW `X[::sub, ::sub]`, so a write would have corrupted `X` long
before this change.

The second, defensive `np.meshgrid` in the `_use_fit` full-grid fallback was
converted the same way for consistency (that branch is unreachable today --
`_chunk_assembly` implies `sub > 1`, which the preceding `elif` already
consumed).

### 3.2 Free at last use

Twelve full-grid locals were consumed and then held to the end of the call
(fourteen names counting `X` and `Y`, which no longer materialise at all).  Each
`del` / `= None` below is at the last read, verified by enumerating every
occurrence of the name in the function:

| local | shape / dtype | bytes at `n_fine = 16384` | last read |
|---|---|---|---|
| `X`, `Y` | (N, N) float64 x2 | 4.295 GB | (now views: never materialised) |
| `_mag0` | (N, N) float64 | 2.147 GB | the bright-support mask |
| `_bright0` | (N, N) bool | 0.268 GB | the carrier peak-eikonal reduction |
| `_coords` | (2, N, N) float64 | 4.295 GB | stashed into `_rd_upsample_coords`; the LOCAL name kept it alive for the rest of the call after the stash had been cleared |
| `_a_rd`, `_nan_rd` | (N, N) float64 x2 | 4.295 GB | the `np.where` that builds `ard_map` |
| `ard_map` | (N, N) float64 | 2.147 GB | the `np.where` that builds `_ard` |
| `E_analytic` | (N, N) complex128 | 4.295 GB | the exit assembly (both branches) |
| `valid` | (N, N) bool | 0.268 GB | the ray-coverage mask |
| `_absE` | (N, N) float64 | 2.147 GB | the unit-phasor divide |
| `_unit`, `_ard` | complex128 + float64 | 6.442 GB | `E_out = _ard * _unit` |
| `_rd_resid_map` | (N, N) complex128 | 4.295 GB | the residual-phasor multiply |

`_coords` is worth naming separately because it was not merely late, it was a
leak of exactly the shape the audit's census reported: the ray-density branch
stashed the array into `_rd_upsample_coords` and freed only the *alias*
(`del _coords_rd`) plus the stash (`_rd_upsample_coords = None`), while the
original binding stayed live to function exit.

### 3.3 MEASURED -- exact frame census at return

`probe_census.py`, `sys.settrace` on the element's `'return'` event, distinct
array bases only:

| | `n_fine = 2048` | `n_fine = 4096` |
|---|---|---|
| BEFORE: arrays / bytes held at return | 40 / **718.08 MB** | 40 / **2854.09 MB** |
| AFTER: arrays / bytes held at return | 26 / **198.08 MB** | 26 / **774.09 MB** |
| reduction | **-520.00 MB (-72.4 %)** | **-2080.00 MB (-72.9 %)** |
| in full-grid float64 equivalents | **-16.25** | **-16.25** |

The two grids give the SAME 16.25 equivalents, which is the point: the freed
set is exactly quadratic in `n_fine`, so it can be scaled without fitting.

**At the shipped `n_fine = 16384`, one full-grid float64 array is 2.147 GB, so
this is `-34.9 GB` held by that one frame.**  For scale, the memory audit
measured `apply_real_lens_traced`'s full-grid working set on the fine leg at
**43.2 GB** (its row 3) and classed it "not a temporaries problem".  It is
partly a lifetime problem: 34.9 GB of the 43.2 GB was still bound when the
function returned.

What remains bound at return, after the change, is exactly the four arrays that
are genuinely alive: `E_in` (the caller's field), `E_out` (the return value),
`_cW` (the carrier eikonal) and `amp` (read by the niche-D9 zero-set check).

### 3.4 MEASURED -- live bytes during the exact leg

`probe_ab.py --tracemalloc`, `tracemalloc.reset_peak()` at entry to
`_fine_trace_group_exit`:

| | `n_fine = 2048` BEFORE | AFTER | `n_fine = 4096` BEFORE | AFTER |
|---|---|---|---|---|
| PEAK live bytes during the leg | 1.4462 GiB | **1.2779 GiB** | 5.1374 GiB | **4.7467 GiB** |
| reduction | -- | **-0.1683 GiB** (-5.4 grid-equivalents) | -- | **-0.3907 GiB** (-3.1 grid-equivalents) |
| p90 of live bytes over the run | 1.0003 GiB | 0.8715 GiB | 4.5271 GiB | 4.1216 GiB |
| mean live bytes (memory-time) | 0.7738 GiB | 0.6791 GiB (-12.2 %) | 2.9588 GiB | 2.6605 GiB (-10.1 %) |

**Read this honestly.**  The PEAK is a single instant, and only the frees that
happen before that instant can lower it; on this small fixture the peak lands
before the ray-density tail, so the tail frees (which are the largest ones)
show up in the p90 and the mean rather than in the peak.  On the real
design-121 order the audit's census caught `X`, `Y`, `_coords`, `_a_rd`,
`_nan_rd`, `ard_map`, `_absE`, `_unit`, `_ard` and `E_analytic` ALL live
simultaneously at the peak plateau (its sec 2.3), so there the peak reduction
is bounded below by these 3-5 equivalents and above by the census's 16.25.
**That upper end has not been measured at `n_fine = 16384`, and this document
does not claim it.**

### 3.5 The process peak RSS does NOT move, and that is expected

Whole-process peak RSS is 6.4103 GB before and 6.4104 GB after at
`n_fine = 4096` (two reps each, interleaved).  Two reasons, both mundane:

* RSS is a high-water mark that does not fall when memory is freed -- freeing
  early lets LATER allocations reuse the same pages instead of growing the
  process, which is a real win that this metric cannot show; and
* the run's global peak sits in the READOUT (`carrier.py`), after
  `apply_real_lens_traced` has returned and its locals are gone regardless.
  Lowering that peak is the `carrier.py` half of the audit item -- the five
  consumed phase factors and `_envelope_amp_radius` -- which is not this file's
  change to make.

The whole-run `tracemalloc` peak is likewise identical (6.6535 GiB both arms),
for the same reason and with the same conclusion.

### 3.6 BITWISE -- the group exit field

`probe_ab.py` drives `propagate_traced_carrier_chain(..., final_leg='exact')`
on a design-121-like configuration (N = 1024 coarse, `dx = 4 um`,
`w = 0.9 mm`, a strong biconvex N-BK7 singlet giving a high-NA exit,
`ray_subsample = 4`, `amplitude_model='ray_density'`,
`preserve_input_phase='remap'`, `remap_sampling='full'`, `window_factor = 4`)
and captures the `(E_exit_fine, dx_fine)` that `_fine_trace_group_exit`
returns:

| | sha256 of `E_exit_fine.tobytes()` | sha256 of the chain readout |
|---|---|---|
| `n_fine = 2048` BEFORE | `8669411f8acc599de36f22a20bf803fc93ded57b7d911cbc9b2acac4414d8a71` | `27d54c2bdeb029d24cd82db2aca793a1b387026d1c15cb9be7f0730832584b93` |
| `n_fine = 2048` AFTER | **identical** | **identical** |
| `n_fine = 4096` BEFORE | `9cc565157a6d8d5ac45c375101a9011d17226b97c202f881911a3ac8f181731c` | `9b1ba6abc16fa03c94778d81724a14b8782e955b0437ee4327f026464e8ef436` |
| `n_fine = 4096` AFTER | **identical** | **identical** |

Byte-for-byte, on the array the audit item is about, at two grids, through the
public chain entry point.

### 3.7 BITWISE -- eleven configurations, not one

The fixture above exercises ONE carrier kind and one amplitude model.  The same
two snapshots were compared across the matrix that reaches the changed code by
different routes (`_carriers_case.py`, N = 256, sha256 of the returned exit
field):

| configuration | before vs after |
|---|---|
| scalar carrier, ray-density, `remap_sampling='full'`, sub 4 | BYTE-IDENTICAL |
| the same at `remap_sampling='lattice'` | BYTE-IDENTICAL |
| screen amplitude, sub 4 | BYTE-IDENTICAL |
| screen amplitude, sub 1 (the FULL-GRID Newton path, where `X.ravel()` changes from a view to a copy) | BYTE-IDENTICAL |
| `carrier='auto'` (the fitted-carrier path) | BYTE-IDENTICAL |
| `carrier=None, tilt_aware_rays=True` (the implicit auto-carrier) | BYTE-IDENTICAL |
| `carrier=None` (plane-wave reference) | BYTE-IDENTICAL |
| `carrier=<ndarray>` (user wavefront) | BYTE-IDENTICAL |
| `carrier=TiltedCarrier(...)` (niche D1) | BYTE-IDENTICAL |
| `preserve_input_phase=False` (the `amp * phase_exp` assembly branch) | BYTE-IDENTICAL |
| `sag_chunk_rows=64` (row-band assembly; `X = Y = None`, so the broadcast build is skipped entirely) | BYTE-IDENTICAL |

Eleven of eleven.  The `sub = 1` and `TiltedCarrier` rows are the two that
matter most for the meshgrid substitution: the first is the only path that
hands a consumer the whole `X` rather than a strided slice of it, and the
second is the only one that does arithmetic on `X`/`Y` inside
`_compute_carrier`.

The primary comparison is also pinned IN THE SUITE without needing two package
snapshots: `test_group_exit_field_byte_identical_to_materialised_grids` runs
the element on an N = 1024 design-121-like fixture twice -- once as shipped,
once with `_lens_traced.np.broadcast_to` shimmed to return
`np.ascontiguousarray(...)`, i.e. the MATERIALISED memory behaviour the
meshgrid form had -- and requires `tobytes()` equality.  It also asserts the
C6 residual fit engaged at degree 6 / 27 terms, so the fixture is provably on
the path this change touches, and every `del` site is executed (verified by
line tracing: 10 of 12 sites are hit by this one fixture; the two that are not
are the all-dark-input `_mag0` branch, where `del` on a just-bound name cannot
fail, and the unreachable `_use_fit` full-grid fallback).

---

## 4. GREEN

All at `OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = MKL_NUM_THREADS =
NUMEXPR_NUM_THREADS = 1`, `-p no:randomly`.  Nothing was xfailed, skipped or
deselected by this change.

**One honesty note on attribution.**  The suites below run against the WORKING
TREE, which at the time of the run also carried another agent's in-progress
edits to `carrier.py`, `mft.py`, `fft_infra.py`, `_bluestein.py` and
`memory.py` (the other half of the same two audit items).  So a green suite
here is evidence that the COMBINED tree is green, not that this file's change
alone is.  The isolation evidence for this file is sec 3.6 / 3.7 -- the two
package snapshots that differ in exactly one file -- and the identity tests of
sec 2.4, which compare against a reference carried inside the test module and
therefore do not depend on the rest of the tree at all.

| suite | where | result |
|---|---|---|
| `test_niche_perf_poly_locals.py` (new: 20 tests) | Windows, py 3.14.6 / numpy 2.4.4 | 20 passed, 17.7 s |
| c6 (fit guard + stationary-phase launch), c11, c12, d2, d6, tight_focus_readout, carrier_referenced, + the new module | Windows | **204 passed**, 18:24 |
| the TRACED NICHE SET -- every `tests/unit` module that touches `apply_real_lens_traced` / `propagate_traced_carrier_chain` / `prepare_real_lens_traced` / `_lens_traced` (97 files, list in `_traced_tests.txt`) | Windows | STILL RUNNING at the time this document was written; it passed the pool/`spawn` modules (`d8_congruence_workers`, `newton_pool_both_fits`) without failing.  Its nine heaviest traced modules are the named set above, already green twice. **This row must be filled in before the branch is proposed for merge.** |
| the same named set | **WSL**, `/home/travaj/lumen_venv`, py 3.12.3 / **numpy 2.4.6** | **204 passed**, 24:51 -- the same 204, on a different Python AND a different numpy |
| `test_niche_perf_poly_locals.py` under the DEFAULT (randomised) test order | Windows | 20 passed, 30.2 s |
| `ruff check` on both changed files | Windows | All checks passed |

The two element-level tests carry `@pytest.mark.slow` (10.2 s and 6.6 s at
N = 1024): the repo's fast unit job runs `-m "not integration and not slow"`
and a dedicated job runs `-m "slow and not integration"`
(`.github/workflows/unit-tests.yml:496`), so they run in CI -- they are routed,
not skipped.

---

## 5. FAIL-BEFORE, RECORDED

| # | claim | fail-before evidence | pass-after |
|---|---|---|---|
| 1 | `_poly` is the hot site and the rewrite is 3.8x | audit probe on the PRE-change tree, run in-session before any edit: `.value()` **9.709 s** / band, `_poly` 9.346 s; `scratchpad/perf_traced_lens/_poly_before.txt` | `.value()` **2.568 s**, **3.78x**; `_poly_after.txt` |
| 2 | the identity tests are not vacuous | with the audit's V3 `_poly` (max|delta| 1.07e-14) substituted, `test_value_bit_identical` and `test_poly_six_outputs_bit_identical` FAIL | with the shipped `_poly`, all 20 pass |
| 3 | the byte-compare test is not vacuous | with `broadcast_to` shimmed to perturb one coordinate element by 1 ulp, the exit fields differ (max|delta| 2.02e-11) | unperturbed: `tobytes()` equal |
| 4 | the element held 16.25 grid-equivalents to return | `probe_census.py` on the HEAD snapshot: 40 arrays, 718.08 MB @2048 / 2854.09 MB @4096 | 26 arrays, 198.08 MB / 774.09 MB |
| 5 | the group exit field does not move | sha256 of `E_exit_fine` on the HEAD snapshot at both grids | identical sha256 after |

---

## 6. WHAT THIS DOES NOT CLOSE

1. **The fan has still never been run end to end.**  The 1.74x order-level
   projection is arithmetic on the audit's own measured share and this
   document's measured site ratio, exactly as the audit's 1.74x was.
2. **The 34.9 GB is a frame census at return, not a peak-RSS reduction at
   `n_fine = 16384`.**  It is exact and it scales exactly; what it buys at the
   real peak depends on where that peak sits, which is decided in `carrier.py`
   (the readout and the five consumed phase factors) and is measured there,
   not here.
3. **`carrier.py`'s half of the memory item is untouched** -- `env_f`, `_ph`,
   `_cf`, `_rp`, `_xf` (21.48 GB) and the three `carrier.py` meshgrid sites.
   Until those land, the whole-process peak of a design-121 order is set in the
   readout and will not move.
4. **The 3-5 grid-equivalent peak reduction was measured at `n_fine` 2048 and
   4096**, on a fixture whose peak instant is not the design-121 order's peak
   instant.  The 16.25-equivalent census is the grid-independent number; the
   peak number is not.
5. **One suspected latent `NameError` was CHASED AND CLEARED**, recorded here
   because the next person to add a `del` to this function will suspect it
   too.  The element already frees `amp` on the `sub > 1` +
   `preserve_input_phase` path (`_lens_traced.py:9758` / `:9773`), yet `amp`
   is read again at the niche-D9 zero-set check (`:10174`) whenever `origin`
   is set and `ORIGIN_AMP_SUPPORT_CHECK` is not `'silent'` (it defaults to
   `'error'`).  That looks reachable and is not: a decentred `origin` is
   REFUSED unless `amplitude_model='ray_density'` AND
   `preserve_input_phase='remap'`, and `'remap'` normalises
   `preserve_input_phase` to `False` (`:6937`), which is exactly the flag the
   two `del amp` sites are gated on.  Verified by running the combination
   against the HEAD snapshot: it raises the documented `NotImplementedError`
   from the origin guard, never a `NameError`.  No `del` added by this change
   is on that path.

---

## 7. ARTEFACTS

All under `scratchpad/perf_traced_lens/` (session scratchpad; nothing was
written into the repo except this document, the edited library file and the new
test module).

| file | what |
|---|---|
| `base/lumenairy/`, `new/lumenairy/` | the two package snapshots the A/B runs against (differ in one file) |
| `probe_ab.py` | drives the chain's exact leg, captures the group exit field's sha256, peak RSS, `tracemalloc` leg peak and a 50 Hz live-bytes series |
| `probe_census.py` | the exact `sys.settrace` frame census at the element's return |
| `_poly_before.txt`, `_poly_after.txt` | the audit's own `_poly` probes, pre- and post-change |
| `_ab_mem.txt`, `_ab_tm2048.txt`, `_ab_tm_series.txt`, `_ab_census.txt` | the memory A/B transcripts |
| `_series_*.tsv` | the live-bytes time series |
| `_carriers_case.py` | the eleven-configuration byte-compare of sec 3.7 |
| `_traced_tests.txt` | the 97-file traced niche set the green run used |
| `_wsl_suite.txt` | the WSL arm |
| `_grad_ab.txt` | a cold-caller `grad()` A/B across the two snapshots.  CONTAMINATED (taken while a test suite was running -- both arms read ~2x high) and therefore NOT used in sec 2.3; kept because its RATIO, 3.64x on `value()` and 2.36x on `grad()`, corroborates the quiet-box rows |

Upstream, unmodified and reused rather than re-run:
`scratchpad/traced_speed/probe_poly.py`, `probe_poly2.py` (the audit's own
item-1 probes).
