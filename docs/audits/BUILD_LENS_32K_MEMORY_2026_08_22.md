# BUILD -- the design-121 analytic run at N = 32768, and what was actually in the way

**2026-08-22.  Branch `feat/lens-32k-memory`, cut from `origin/main`
@ `a2652283` (the 5.39.1 release), in the worktree `C:/tmp/lum_mm`.  Commit on
the branch only -- no merge, no push, no `gh`.**

Binding law: `docs/TESTING_STANDARDS.md`.

Companion to `PROBE_D121_ANALYTIC_32K_FOOTPRINT_2026_08_17.md`, which asked
whether the refusal was reducible.  It is.  It is also reducible for a
different reason than the probe supposed, and that is the substance of this
note.

---

## 0. VERDICT

> **THE RUN FITS.  THE PREFLIGHT ADMITS IT AT `need ~81.7 GB` AGAINST
> `121.1 GB` FREE -- 39.4 GB OF HEADROOM OVER A 20 GB FLOOR -- WHERE THE
> SHIPPED BUILD REFUSES AT `need ~115.6 GB` / `+5.5 GB`.  BOTH READINGS COME
> FROM THE SAME UNCHANGED `_preflight_memory_check` CALL AGAINST THE SAME
> BOX-B NUMBERS; ONLY THE LIBRARY UNDER IT DIFFERS.**
>
> **AND THE LEVERS ARE NOT DECORATION: with them OFF the re-anchored preflight
> still REFUSES, by 0.4 GB against the floor.**  Which lever crosses that line
> depends on the input field, and the honest statement is per-lever rather
> than joint: at the 25 % bright-support the preflight assumes, LEVER 3
> (-2.0 grids) is what flips the verdict and LEVER 1's credit is hidden
> underneath the `carrier='auto'` fit; at the 5 % a real design-121 beam may
> well have, LEVER 1 is worth its full -3.5 and is the one that matters.  The
> preflight models that as a MAX of the route's two candidate peaks, not a
> sum, which is why it can say so.
>
> **1. THE PROBE'S 11.23 GRIDS WERE NOT THE ACCUMULATORS.**  PROBE_D121 S5
> attributed the residual to "the momentum accumulator, the sag gradients
> feeding it, and the ASM work arrays".  Measured again with a time-resolved
> sampler instead of a warmed peak -- a 1 kHz thread that records the main
> thread's stack at every new high-water mark -- **the peak of that call
> stands entirely inside `_compute_carrier`, the `carrier='auto'` polynomial
> FIT, at set-up, before the first surface is touched.**  The accumulators are
> +4.13 grids; the set-up was +9.2 on top of them.  A warmed peak cannot tell
> you where it stood, and that is how the attribution went wrong.
>
> **2. AND THE ANCHOR WAS WRONG FOR A THIRD REASON AGAIN.**  PROBE_D121 S6
> read 11.23 against the preflight's 7.7 and recommended re-anchoring on a
> multi-surface group.  Re-anchored: the surface count is worth **+0.00**
> (confirming the probe's own S4), and the real driver is that the `'auto'`
> term **scales with the BRIGHT-SUPPORT FRACTION OF THE INPUT FIELD**.  On the
> shipped tree the design-121 route reads **+10.44 grids at 5 % bright,
> +13.30 at 21 %, +20.08 at 59 %** against a priced 7.7.  The 7.7 anchor was
> measured on a fixture whose bright support happened to be small.  **No
> prescription-only anchor can be correct about this term**, so it is now
> priced through a stated scaling law with the assumed fraction PRINTED.
>
> **3. LEVER 1 WORKS, AND ITS SAVING IS IN COMMIT, NOT RSS.**  The persistent
> accumulators -- the tangent-facet momentum pair plus the fresh destination
> pair each surface writes into -- are **exactly** the +4.13 grids the route
> costs once the set-up is fixed.  `accumulator_store='memmap'` takes private
> commit from **6.52 to 3.03 float64 grids, -3.48 grids = -29.9 GB at
> N = 32768**.  Resident set RISES (6.5 -> 13.9 grids) because the OS keeps
> file-backed pages resident on an idle box.  Commit is what the 2026-08-16
> run exhausted; commit is what falls.  `tracemalloc` reads -3.49 and cannot
> see the mapping at all -- both are reported, neither alone.
>
> **4. LEVER 3 IS WORTH TWO GRIDS AND ONLY AT N = 32768.**  Streaming the ASM
> transfer function through the multiply removes `H` and the product: 5 grids
> to 3.  But `H` is CACHED when it fits the 2 GiB per-entry cap, which at
> complex64 is exactly satisfied at N = 16384 (2.147 GB) and violated at
> N = 32768 (8.59 GB) -- so below 32768 there is nothing to save and streaming
> costs a rebuild per call.  Measured at N = 4096 with the cap forced to
> refuse (the 32768 condition): **-4.94 grids**.  **Priced at -2.0**, because
> 3 of those 4.94 are the plain builder's chunk, which at 4096 resolves to the
> whole grid and at 32768 self-limits to 17.5 % of it.  Transplanting the
> measured number would have over-credited by 3 grids.
>
> **5. BYTE IDENTITY IS MEASURED AGAINST THE SHIPPED TREE, NOT AGAINST THIS
> ONE.**  A **8960-arm TWO-TREE** comparison against `origin/main` @ 5.39.1
> (5 prescriptions x 4 grids including odd `N` and `dx != dy` x 4 surface
> models x 7 carriers x 7 option combinations x 4 band sizes), compared on a
> SHA-256 of the returned field plus its dtype plus the warning set: **0
> differences, and 8640 of the 8960 returned a FIELD** rather than a matching
> refusal.  Plus 144 new tests, the fold guard's refusal reproduced at every
> band size, and a separate two-tree pin on the carrier fit's own
> `(W, grad_fn, w_fn)`.
>
> **5b. AND THE PRODUCTION-GRID SMOKE READ THREE DIFFERENT HASHES, WHICH IS
> NOT THE LEVERS.**  The N = 16384 smoke of the runner's own call returned a
> different SHA-256 for each lever setting.  The NULL COMPARISON -- the same
> call four times, changing nothing, with the lever arms interleaved --
> produces **two distinct results at N = 4096**, and both lever arms land on
> the MAJORITY null value while a null arm is the odd one out.  **The
> `carrier='auto'` path does not reproduce ITSELF**: its fit forms
> `G = A.T @ A` over 1.8 M rows at 4096 and 28 M at 16384, and a
> multi-threaded BLAS reduction of that length does not fix its partitioning
> across calls.  **Demonstrated, not inferred: with `OMP_NUM_THREADS=1` the
> same four calls give ONE result -- and it is the same value the threaded run
> produced on five of its six arms, both lever arms included.**
> `auto_promote` off, `FFTW_ESTIMATE`, no bad shapes -- not the FFT; `A.T @ A`
> is code this build did not touch, and the two-tree's `'auto'` arms agree
> across both trees -- not this build.  It means a production `carrier='auto'`
> run is not bit-reproducible and no A/B against a stored result can be read at
> the last bits.  The smoke is the lesson too: it varied only the thing under
> test, and a design that does that cannot tell an effect from a drift.  (S4.2)
>
> **6. THE FIRST GREEN WAS VACUOUS AND IS RECORDED AS SUCH.**  The two-tree
> sweep's first run read "8960 arms, 0 differ" with **3776 of them
> refusal-vs-refusal**: the fixtures named `N-SK2` and `N-LAK9`, which are
> design-121 glasses not in the shipped registry, so both trees raised
> `Glass not in registry` and the comparison compared two identical failures.
> Found by printing the refusal census rather than the pass count.  Fixed and
> re-run; the surviving 320 refusals are the legitimate
> `surface_model='displaced'` incompatibilities.
>
> **7. WHAT MADE IT FIT WAS NOT MAINLY EITHER LEVER.**  Honest accounting: the
> levers are worth 3.5 and 2.0 grids; **removing the set-up's waste was worth
> 9.2** (the `carrier='auto'` route's extras fell 13.30 -> 4.12 grids at 21 %
> bright, and 10.44 -> 4.12 at 5 %).  That work was not in the brief
> and is in this build because the re-anchoring it *did* ask for is what
> exposed it -- the seed of the accumulator held **7 float64 grids to deliver
> 2**, and `_screen_obliquity_row_evaluator` had declined to band it on
> grounds that turned out to apply to the fit's SET-UP and not to its
> EVALUATION.

---

## 1. THE MEASUREMENT THAT CHANGED THE ANSWER

PROBE_D121's protocol is warmed `tracemalloc`: call once to warm,
`reset_peak`, call again, read `get_traced_memory()[1]`.  That gives a peak
and no location.  Its S5 then reasons about the location from the code.

The instrument here adds a sampler: a daemon thread reads
`tracemalloc.get_traced_memory()[0]` (LIVE bytes) at ~1 kHz and, at every new
high-water mark, records `sys._current_frames()` for the main thread.  On the
production arm -- design-121 group 1 (three surfaces, N-SK2 / N-SF6),
`surface_model='tangent_facet'`, `carrier='auto'`, `screen_obliquity=False`,
complex64, the production 29.58 mm extent, N = 4096 -- the high-water trace
ends:

```
  HWM  22.359 :: apply_real_lens -> _screen_obliquity_angle_field
                 -> _compute_carrier line 4088   A[nL:, k] = ...
  HWM  23.479 :: apply_real_lens -> _screen_obliquity_angle_field
                 -> _compute_carrier line 4092   A = A * w[:, None]
  PEAK 23.479
```

and the FIRST `progress` callback -- which fires before surface 0's screen --
already reports `peak-so-far 23.479`.  **The whole peak is set before any
surface is touched.**  The band loop that PROBE_D121 S5 analysed never
reaches it.

`carrier='auto'` does not state a ray angle; it FITS one.  The fit is a
weighted least squares of a low-order polynomial's gradient against the
field's own local tilt over its bright support, and its design matrix `A` is
`(2 * n_bright, n_terms)` float64.  On this arm at 59 % bright support that is
5.9 grids, and `A = A * w[:, None]` doubled it.

---

## 2. THE FOUR THINGS THAT WERE BUILT

### 2.1 LEVER 1 -- `accumulator_store={'ram','memmap'}` (+ `scratch_dir`)

`apply_real_lens` now allocates its PERSISTENT full-grid accumulators through
an `_AccumulatorStore`.  Everything else it allocates at full-grid size is a
transient that `sag_chunk_rows` already keeps to one band; what banding cannot
remove is the state that must be simultaneously live across the whole grid
while the band loop walks it:

| accumulator | grids | when |
|---|---|---|
| `_tf_px` / `_tf_py` | 2 | route 3 / remap, from the first powered surface |
| the surface's fresh destination pair | 2 | route 3 / remap, inside every surface |
| `_rm_wx_g` / `_rm_wy_g` | 2 | the remap rung's walk |
| `_obl_p0x` / `_obl_p0y`, `_obl_ux` / `_obl_uy` | 4 | the angle-true obliquity block |
| `_obl_qx` / `_obl_qy`, `_obl_total` | 3 | ditto |

Each is written once per surface, band by band, in increasing row order, and
read back band by band on the next surface.  `'memmap'` backs each with an
`np.memmap` in `scratch_dir` and the OS pages in the band under the cursor.

**The API adjudication.**  Three shapes were considered against the existing
conventions:

| candidate | verdict |
|---|---|
| a bare `scratch_dir=` that turns spilling on by being non-None | REJECTED. Every other opt-in in this signature states the DECISION and takes its parameters separately (`sag_chunk_rows`, `sag_dtype`, `surface_model`), and overloading a path for a mode makes "where do the files go" and "should there be files" the same question. |
| a boolean `memmap_accumulators=True` | REJECTED. The library's mode selectors are strings with a validated tuple (`_VALID_SURFACE_MODELS`, `_VALID_SCREEN_OBLIQUITY`, `_VALID_WAVE_PROPAGATORS`), and a boolean cannot grow a third backing later. |
| **`accumulator_store='ram'\|'memmap'` + `scratch_dir=None`** | **TAKEN.** Matches the string-mode convention, has its `_VALID_ACCUMULATOR_STORE`, names what is being stored rather than how, and keeps the directory a separate concern. |

**Byte identity is structural.**  The store changes only WHERE an
accumulator's bytes live.  Two properties carry it:

* every expression that reads or writes an accumulator is unchanged, on the
  same dtype and the same C-contiguous layout;
* `_make` hands out `np.asarray(mm)` -- a BASE-CLASS view of the mapping, not
  the `np.memmap` subclass -- so no ufunc can dispatch on the array's type and
  no result can inherit a subclass wrapper.  `test_closing_the_store_twice_is_harmless`
  pins `type(a) is np.ndarray` for exactly that reason.

**Windows.**  Two hazards, both found by testing rather than by reading:

1. a mapped file cannot be unlinked while the mapping is open, so `close()`
   drops the store's own references, calls `mm._mmap.close()` explicitly, and
   only then unlinks;
2. **unlinking is not always complete when the call returns.**  The entry
   lingers in a pending-delete state and the immediately following `rmdir`
   fails with "directory not empty" on a directory `listdir` already reports
   as EMPTY.  Deterministic on this box for the private-scratch-directory
   path.  `close()` retries with backoff (10 attempts, ~1.1 s worst case) and
   **warns** if it exhausts them rather than leaving silent litter in `%TEMP%`.
   Worth being precise about what the failure mode was: the FILES were always
   gone -- the directory was empty every time -- so no accumulator data was
   ever left behind, only an empty directory.

   **HOW LONG IT LASTS IS A FUNCTION OF THE MACHINE'S LOAD, SO THE TEST DOES
   NOT ASSERT THE INODE.**  8/8 clean runs on an idle box after the retry, and
   a failure under a concurrent 16 GB job.  A test that demanded the directory
   be gone would be asserting the load -- `docs/TESTING_STANDARDS.md` S3
   exactly.  `test_a_private_scratch_directory_leaves_no_data_behind` asserts
   the two-sided property instead: no FILE survives, and any surviving
   directory is EMPTY *and* was reported through a warning rather than left
   silently.  That is the library's decision; the inode is the OS's.

   The same test also had to be made HERMETIC, and for a reason worth
   recording: globbing the real `%TEMP%` for `lumenairy_accum_*` also sees the
   private directories of any CONCURRENT process using this feature.  That is
   not hypothetical -- it is how the restated test first failed, against a
   two-tree sweep running beside it.  It now points `tempfile.tempdir` at its
   own `tmp_path`.  A test that reads another process's live state is testing
   the box.

**The scratch footprint is bounded by what is LIVE, not by what was ever
made.**  The tangent-facet path allocates a FRESH destination pair per surface
and a fresh pair per gap transport and rebinds the accumulator to it; the
previous pair is garbage immediately.  A store that only released at `close()`
would therefore grow to twelve mappings on a three-surface group -- ~103 GB of
scratch at N = 32768 against ~34 GB live.  Each mapping carries a
`weakref.finalize` on the view it handed out, so it is closed and unlinked as
soon as that view is collected.  Measured on a three-surface group: **2
simultaneous files at the sampled points, 0 left after**, against 12 made.
`test_the_scratch_footprint_is_bounded_by_what_is_LIVE` pins it at <= 8, a
derived bar (six is the structural maximum -- momentum pair, destination pair,
walk pair) with a factor to spare.

**Context management.**  `apply_real_lens` is now a thin wrapper that owns
`with _AccumulatorStore(...) as _store:` and forwards to
`_apply_real_lens_impl`.  The split exists for exactly that `with`: the scratch
files must be released on EVERY exit, an exception raised mid-prescription
included, and wrapping two thousand lines of body in a `try` was the
alternative.  `test_the_scratch_files_are_removed_when_the_call_raises` uses
the engineered folding prescription, which raises AFTER the store has
allocated.

Two pre-existing tests read `inspect.getsource(LR.apply_real_lens)` for the
band gates and now read `_apply_real_lens_impl` instead.  **That is a
RETARGET, not a relaxation** -- the assertions are unchanged, and each gained
a companion check that `apply_real_lens` still forwards, so a future move of
the body cannot make them read an empty string and pass vacuously.

**The split also moved every warning's apparent origin, and that is fixed
rather than accepted.**  An extra frame between the caller and the body made
the aperture guard report itself as coming from
`return _apply_real_lens_impl(...)` -- which tells a user where the library
called itself and nothing about their code.  Every warning raised inside the
impl now uses `_WARN_STACKLEVEL`, and
`test_warnings_still_point_at_the_callers_line_after_the_split` asserts the
recorded warning's filename is the TEST FILE.  Found by reading a pytest
warnings summary, not by looking for it.

### 2.2 LEVER 1b -- the accumulator SEED is banded

This is the one that reaches the peak, and it is what the re-anchoring forced.

`_screen_obliquity_row_evaluator` (v5.35.3) already bands the carrier momentum
field, but only for a `TiltedCarrier`.  Its docstring declines the `'auto'`,
scalar-conjugate and ndarray congruences because "`_compute_carrier`'s set-up
is itself whole-grid".  **That is true of the set-up and false of the
evaluation.**  Once the fit's coefficients exist, `grad_fn` is POINTWISE -- a
polynomial in `(x, y)`, a closed-form sphere, an index lookup.

What the whole-grid seed held:

```
  _screen_obliquity_angle_field, non-collimated carrier
    Xg, Yg = meshgrid(...)                          2 grids
    _compute_carrier -> W_full                      1 grid   DISCARDED
    L, M = grad_fn(Xg, Yg)                          2 grids
    asarray(L, float64) * n1                        2 grids
                                                    ------
    7 live to deliver 2 -- 60 GB at N = 32768 to deliver 17
```

`_screen_obliquity_rows_any` evaluates it a band at a time straight into the
accumulator store.  Byte identity: `np.meshgrid(xax, yax[r0:r1])` is exactly
the `[r0:r1]` slice of `np.meshgrid(xax, yax)`, `grad_fn` is pointwise, and
the same `asarray(., float64) * n1` chain runs on the result.

**The one thing a band cannot reproduce, and does not skip.**  The whole-grid
helper collapses a CONSTANT momentum field to two Python floats
(`ptp(L) == 0 and ptp(M) == 0`).  That collapse is OBSERVABLE, not cosmetic:
under NEP 50 a Python float and a float64 array of the same value promote
`float32` geometry differently, which is the hazard `BUILD_TF_BANDED` S2.1
documented for the accumulator.  So the caller reproduces it as a band-wise
reduction -- `ptp == 0` on both components is exactly "every element equals
element 0" -- and `test_a_constant_momentum_field_still_collapses_to_two_python_floats`
pins it on the `sag_dtype='float32'` arm where a promotion would show.

### 2.3 `_compute_carrier` builds less, and only what is asked for

Three byte-null trims on the `'auto'` fit:

| trim | what it removes | why it is bit-identical |
|---|---|---|
| `need_W=False` | the full-grid potential `W_full`, plus `_poly_and_grad`'s `Wq` inside `grad_fn` | `Wq` accumulates into its own array from its own terms; the gradient does not read it.  With it off, `X`/`Y` are read only for their SHAPE, so the caller passes zero-copy `broadcast_to` views and the coordinate stack goes with it. |
| separable sample coordinates | `meshgrid` (2 grids) + two midpoint expressions (2 more) | `Xg[i,j] == xax[j]` and `Yg[i,j] == yax[i]` EXACTLY, so `0.5*(Xg[:,1:]+Xg[:,:-1])[i,j] == 0.5*(xax[j+1]+xax[j])` -- the same two IEEE operands in the same order.  Boolean-masking a zero-strided `broadcast_to` view selects only the kept elements; the `(N, N)` intermediate never exists. |
| `A *= w[:, None]` | a full copy of the fit's largest array | `np.multiply(A, w[:,None], out=A)` is the same elementwise product on the same operands. |

Pinned two-tree against `origin/main` on the fit's own outputs (`W`, `grad_fn`,
`w_fn`), 12 arms, SHA-256, 0 differences.

### 2.4 LEVER 3 -- `stream_transfer_function=True`

`angular_spectrum_propagate` gains an opt-in that generates `H` one row band at
a time DURING the frequency-domain multiply, in place on the spectrum:

```
  plain     H = _get_asm_H_natural(...)                    # a full complex grid
            fftshift(ifft2(fft2(ifftshift(E)) * H))        # and a full product

  streamed  spec = fft2(ifftshift(E))
            for band: spec[j0:j1] *= H_band                # H never exists
            fftshift(ifft2(spec))
```

Bit identity rests on two facts, both properties of the code:

1. `_get_asm_H_natural` ALREADY builds `H` in row chunks with this expression
   on these operands, and every element of `H` depends on its own row and
   column alone -- so `H[j0:j1]` has the same bytes whatever the chunking.
   The streamed path reuses the same kernel (`_asm_H_from_kz`) rather than
   restating it.
2. `np.multiply(a, b, out=a)` yields the same bits as `a * b` when
   `result_type(a, b) is a.dtype`.  The caller guarantees that precondition
   (`H` is built at `spec.dtype`); if it could not, it does not take the path.

`test_the_streamed_band_width_is_a_free_choice` sweeps the band constant from
1 element to 4 Mi and requires identical bytes at every width -- otherwise the
identity above would be an accident of one default.

**The band width had to be capped, and the first attempt made things worse.**
The plain builder sizes its chunk at 10 % of the RAM budget, which on a 90 GB
budget resolves to the WHOLE grid below N ~ 16384.  Harmless there (the
float64 kernel workspace is the only thing live) and counter-productive here
(the spectrum is live alongside it): the first streamed measurement at N = 4096
read **+1.69 grids WORSE** than the plain path.  Capping the workspace at
4 Mi elements fixes it and makes the saving monotone in N.

**Opt-in, and the tradeoff is priced.**  The streamed `H` is never cached.
`_H_CACHE_MAX_BYTES_PER_ENTRY` is 2 GiB, and at complex64 an `H` is 2.147 GB
at N = 16384 (cached, exactly at the cap) and 8.59 GB at N = 32768 (refused).
So the lever is free above 32768's threshold and a real repeat cost below it.
`return_transfer_function=True` asks for the very grid it avoids, so that
combination takes the plain path and still returns `H`.

---

## 3. THE RE-ANCHOR

Design-121 group 1 out of `20260707 dll Tx02-MSOP16.zmx` (three surfaces,
N-SK2 / N-SF6 -- one of the eight groups PROBE_D121 measured), at the
production 29.58 mm extent, complex64 field, warmed, one arm per process.

**Three instruments, because they answer three different questions.**
`tracemalloc` counts Python allocations and a mapped page is not one, so it is
the right instrument for the Python-side claim and the WRONG one for the spill
claim.  `rss` is resident pages, mapped ones included.  `vms` on Windows is
private commit -- anonymous memory the system must guarantee -- and mapped file
pages carry no commit charge, so that is the quantity the 32k run exhausted.
Both process metrics are SAMPLED by a thread, not read from `peak_wset` /
`peak_pagefile`, which are cumulative from process start and would carry the
warm-up run's peak into the measured one.

### 3.0 What the SHIPPED tree costs on this fixture, for comparison

Same protocol, same group, on `origin/main` @ 5.39.1.  These are the numbers
the 7.7-grid anchor is supposed to cover:

```
  EXTRAS over the paraxial no-carrier call (tracemalloc, float64 grids)
    tangent_facet, no carrier                     +4.12
    tangent_facet + finite-conjugate carrier      +4.62
    tangent_facet + carrier='auto',  5% bright   +10.44
    tangent_facet + carrier='auto', 21% bright   +13.30
    tangent_facet + carrier='auto', 59% bright   +20.08
                          priced by the preflight: 7.7
```

Under-priced by 2.7 to 12.4 grids depending on the FIELD, on a term whose
anchor names only the prescription.  PROBE_D121 was right that the refusal
survived only because of the 20 GB floor.

### 3.1 N = 4096, EXTRAS over the paraxial no-carrier call at the same N

| arm | tracemalloc | rss | **commit** |
|---|---|---|---|
| paraxial, no carrier (the baseline) | 2.384 | 2.368 | **2.391** |
| `tangent_facet`, no carrier | +4.12 | +3.50 | **+4.13** |
| ... + `accumulator_store='memmap'` | +0.63 | +9.51 | **+0.64** |
| `tangent_facet` + finite conjugate | +4.12 | +3.52 | **+4.13** |
| ... + `accumulator_store='memmap'` | +0.63 | +11.50 | **+0.65** |
| `tangent_facet` + `carrier='auto'`, 5 % bright | +4.12 | +3.53 | **+4.13** |
| ... + `accumulator_store='memmap'` | +2.74 | +11.50 | **+2.75** |
| `tangent_facet` + `carrier='auto'`, 21 % bright | +4.12 | +3.72 | **+4.13** |
| ... + `accumulator_store='memmap'` | +3.73 | +11.48 | **+3.73** |
| `tangent_facet` + `carrier='auto'`, 59 % bright | +11.25 | +11.25 | **+11.25** |
| ... + `accumulator_store='memmap'` | +11.25 | +11.49 | **+11.25** |

**The same numbers at N = 8192**, which is what makes a single coefficient
legal: `tangent_facet` no carrier 6.515 -> 3.034 commit under `'memmap'`
(-3.48, against -3.48 at 4096); the `tangent_facet` and `'auto'` rows flat to
four significant figures across the 4x change in grid area.  Note the RSS
column moves there -- the mapped pages were NOT kept resident at 8192, because
by then other jobs wanted the RAM.  That is the same statement as "RSS rises
on an idle box", seen from the other side.

Read three things off that table.

* **`+4.13` IS THE FOUR ACCUMULATORS.**  From surface 2 onward route 3 holds
  the momentum pair and the fresh destination pair; the residual 0.13 is band
  transients.  Nothing else would have moved it, and `'memmap'` moves it to
  0.64.
* **RSS GOES UP UNDER `'memmap'`, AND THAT IS NOT A FAILURE.**  The mapped
  pages stay resident while the box has RAM to spare; commit is what falls,
  and commit is the constraint.  Reporting only RSS would have made the lever
  look broken; reporting only tracemalloc would have made it look free.
* **THE `'auto'` TERM IS A PROPERTY OF THE FIELD.**  It is invisible at 5 %
  bright (it sits under the accumulators), binding at 21 % once they are
  spilled, and dominant at 59 %.

### 3.2 The `'auto'` fit peak, and the law the preflight uses

Fit peak read with the accumulators spilled, so it is not hidden underneath:

```
  bright support     5 %      21 %      59 %      96 %
  fit peak (grids)   2.75     3.73     11.25     18.65
```

The preflight prices the straight line through the two EXTREME points,
`1.9 + 17.5 * f`, which reproduces the ends and OVER-prices both interior
points (5.6 vs 3.73, 12.2 vs 11.25).  A line rather than a spline through all
four precisely so that it over-prices: that is the direction a preflight must
be wrong in.  `LENS_AUTO_CARRIER_BRIGHT_FRAC` states the assumed fraction
(default 0.25) and the preflight PRINTS it, because a term this large that
depends on the field rather than the prescription must not be invisible.

**THE ANCHOR IS FLAT IN N**, which is what makes a single coefficient legal at
all.  The same arms at N = 8192, in commit grids: `tangent_facet` no carrier
6.514 (vs 6.518 at 4096), `+ carrier='auto'` at 21 % 6.514 (vs 6.518), at
59 % 13.664 (vs 13.667).  Three rows, flat to four significant figures across
a 4x change in grid area.

### 3.3 The streamed-H credit, measured and then DERIVED down

At N = 4096 with `_H_CACHE_MAX_BYTES_PER_ENTRY` forced to refuse the entry --
the N = 32768 condition, reproduced at a size that fits:

```
  paraxial baseline, plain H (H not cacheable)   7.709 grids commit
  paraxial baseline, streamed H                  2.762
                                                 ------
                                                 -4.94
```

**Priced at -2.0, not -4.94.**  The difference is the plain builder's chunk:

| N | plain build chunk | its float64 workspace | streamed band |
|---|---|---|---|
| 4096 | 4096 rows (100 %) | 4.00 grids | 1024 rows (1.00) |
| 8192 | 8192 rows (100 %) | 4.00 | 512 (0.25) |
| 16384 | 11440 rows (69.8 %) | 2.79 | 256 (0.062) |
| **32768** | **5720 rows (17.5 %)** | **0.70** | **128 (0.016)** |

At N = 32768 the plain builder already self-limits, so the saving is `H` plus
the product plus ~0.7 of workspace: **2 grids, conservatively**.  Quoting the
measured -4.94 at 32768 would have over-credited by 3 grids -- which is the
one direction a credit must never be wrong in.

---

## 4. BYTE IDENTITY

### 4.1 Two-tree, against `origin/main` @ 5.39.1

`_twotree_mm.py`, the same arm list executed under both library roots with the
path assert keyed to each.  5 prescriptions (biconvex, a three-surface
triplet, a leading-plate quad, an oblate `k = +4` whose NaN annulus falls
inside the grid, a conic+aspheric asphere) x 4 grids (96; 65 odd; 72 with
`dx != dy`; 129 odd with `dx != dy`) x 4 surface models x 7 carriers (none,
`'auto'`, two signed conjugates, finite-R tilt, collimated tilt, zero tilt) x
7 option combinations (plain, fresnel, fresnel+absorption, no bandlimit,
absorption, float32 geometry, fresnel+float32) x `sag_chunk_rows` in
{0, 1, 7, AUTO}.  Under `origin/main` neither new keyword exists, so the
reference arms run without them: **every arm of this build is compared against
the SHIPPED answer, not against this build's own.**

```
  8960 arms   0 differ
  8640 returned a FIELD;  320 are refusal-vs-refusal
  (the 320: surface_model='displaced' is incompatible with fresnel/absorption)
```

Compared per arm: SHA-256 of the returned field, its dtype, and the sorted
warning set -- or, where the call refuses, the exception type and message.

**The first run of this sweep was a vacuous green and is recorded as one.**
It read `8960 arms, 0 differ` with **3776 refusal-vs-refusal**, because the
fixtures named `N-SK2` and `N-LAK9` -- design-121 glasses that are NOT in the
shipped registry -- and both trees raised `Glass not in registry`.  A
comparison of two identical failures is not evidence.  It was caught by
printing the refusal CENSUS next to the pass count, not by reading the pass
count; the census is now part of the script's output for that reason.

### 4.2 THE PRODUCTION-GRID SMOKE READ THREE DIFFERENT HASHES -- AND IT IS NOT THE LEVERS

The N = 16384 smoke of the runner's own call (`surface_model='tangent_facet'`,
`carrier='auto'`, complex64, the production 29.58 mm extent) returned a
DIFFERENT SHA-256 for each lever setting:

```
  levers OFF   483.0 s   commit 10.59 g   sha 739ad40fb148f2a2
  memmap       437.7 s   commit  5.14 g   sha 4f1bdd8912226ac5
  stream H    1100.6 s   commit  6.51 g   sha 3fbc5d7fdb4e5eff
```

**Before that can be called a byte-identity failure it has to be separated
from the box, and the instrument for that is a NULL COMPARISON: the same call,
twice, changing nothing.**  `_null16k_mm.py` at N = 4096, interleaving the
lever arms between four IDENTICAL ones:

```
  N = 4096   auto_promote=False   planner=FFTW_ESTIMATE   USE_PYFFTW=True

    levers OFF #1   sha 19d912611851ef8b   bad_shapes=0  plans=2
    levers OFF #2   sha 19d912611851ef8b
    memmap          sha 19d912611851ef8b        <- SAME as the null
    levers OFF #3   sha 06afacc78d6882f6        <- THE NULL ITSELF DIFFERS
    stream H        sha 19d912611851ef8b        <- SAME as the null
    levers OFF #4   sha 19d912611851ef8b

    the four NULL arms produced 2 distinct results -- NOT REPRODUCIBLE
```

**The `carrier='auto'` path does not reproduce ITSELF, and both levers land on
the majority null value.**  The odd arm is a null one, sandwiched between the
two lever arms that matched.  No A/B across a call that varies run to run can
attribute anything to what was varied -- including the 16384 smoke, and
including any comparison against exp31's stored numbers.

**The mechanism, and what it is NOT.**  `auto_promote` is off (the 5.30.1
default), the planner is `FFTW_ESTIMATE`, `_PYFFTW_BAD_SHAPES` is empty and the
plan cache holds 2 entries -- so no FFT-backend fallback and no mid-session
plan promotion.  What is left is the fit itself:
`_solve_lstsq_thread_safe` forms the normal equations `G = A.T @ A` over a
`(2 * n_bright, n_terms)` design matrix -- **1.8 million rows at N = 4096 and
28 million at N = 16384** -- and a multi-threaded BLAS reduction of that length
does not fix its partitioning across calls.

**AND THAT IS DEMONSTRATED, NOT INFERRED.**  The same four-call null with
`OMP_NUM_THREADS=1` (and `MKL` / `OPENBLAS` pinned with it), everything else
identical:

```
  N = 4096  carrier='auto'  OMP=1
    call 0   191.3 s   sha 19d912611851ef8b
    call 1   108.9 s   sha 19d912611851ef8b
    call 2   104.3 s   sha 19d912611851ef8b
    call 3    29.7 s   sha 19d912611851ef8b
    1 distinct over 4 IDENTICAL calls -- REPRODUCIBLE
```

**Single-threaded it reproduces, and it reproduces to the SAME value the
threaded run produced on five of its six arms** -- including both lever arms.
So the threaded run is not computing a different answer; it is computing the
same answer with an occasionally different summation order.  That closes the
diagnosis: a threaded reduction, not the FFT, not the levers, not the field.

**IT IS ALSO A PROPERTY OF THE SHIPPED PATH, NOT OF THIS BUILD.**  Structurally,
`G = A.T @ A` is code this build did not touch -- the fit trims
(`A *= w[:, None]`, the separable coordinates, `need_W=False`) are all
elementwise and cannot introduce a reordering; and the OMP-pinned arm reaching
the same value as the threaded majority says the arithmetic is unchanged, only
its order.  Empirically, the 8960-arm two-tree's `'auto'` arms agree across
BOTH trees, because at those grid sizes the reduction stays below the threading
threshold and is deterministic in both.

**A CLOSED-FORM CARRIER DOES NOT SHOW IT.**  The same four-call null on this
tree with `carrier=0.030` -- a signed conjugate: identical accumulator traffic,
identical screen, identical ASM legs, and NO least-squares reduction -- at
N = 8192:

```
    call 0  549.2 s   sha b787e0a7ac2392f5
    call 1   90.1 s   sha b787e0a7ac2392f5
    call 2  287.0 s   sha b787e0a7ac2392f5
    call 3  290.2 s   sha b787e0a7ac2392f5
    1 distinct over 4 -- REPRODUCIBLE
```

The wall clocks span 6x (the box was carrying four other jobs) and the bytes do
not move, which is the point: it is not load that perturbs the result, it is
the reduction that load perturbs.

**THE DIRECT SHIPPED-TREE CONTROL IS REPORTED HONESTLY, INCLUDING WHERE IT
DOES NOT SHOW THE EFFECT.**  A four-call null on `origin/main` at N = 8192 read
**1 distinct over 4 -- REPRODUCIBLE**.  That is NOT a clearance for this build
and it is not evidence against the diagnosis either: the effect is
INTERMITTENT (one arm in six on the run that caught it), so four calls that
agree bound nothing.  A matched control -- BOTH trees, the same N = 4096 that
caught it, the same eight threads, eight calls each -- is the only reading
that could separate them, and its result is recorded in S10 whatever it says.
What does NOT depend on it: the byte-identity claim, which rests on the
8960-arm two-tree at sizes where both trees are deterministic, and the
diagnosis, which rests on the OMP-pinned arm.

**What the byte-identity contract therefore says, precisely:** the levers are
byte-identical wherever the underlying path is reproducible at all.  Where it
is not, nothing is -- and that is worth knowing independently of this build,
because it means a production `carrier='auto'` run is not bit-reproducible and
an A/B against a stored result cannot be read at the last bits.  A closed-form
carrier (a signed conjugate, a `TiltedCarrier`) has no reduction and no such
problem; `OMP_NUM_THREADS=1` restores reproducibility at the cost of the whole
run's threading, which makes it a diagnostic setting and not a production one.

**AND THE SMOKE ITSELF IS THE LESSON.**  It varied exactly one thing -- the
lever setting -- and read three different hashes, which is what a byte-identity
failure looks like.  A design that only varies the thing under test cannot
distinguish an effect from a drift.  The null arm cost one extra call and
turned a false alarm into a finding.

### 4.3 The carrier fit, two-tree on its own outputs

`_twotree_fit.py`: 3 grids x 2 waists x 2 tilts, comparing SHA-256 of
`W_full`, `grad_fn(X, Y)`'s two components and `w_fn(X, Y)`.

```
  12 arms   0 differ
```

### 4.4 The suite

```
  tests/unit/test_lens_memory_levers.py               144 passed   (new)
  tests/unit/test_tf_banded_halo.py                }
  tests/unit/test_obl_banded_halo.py               }  155 passed
  tests/unit/test_tangent_facet.py                 }
  tests/unit/test_tangent_facet_remap.py           }  104 passed (unchanged)
  tests/unit/test_slant_chunk_byte_identical.py    }  (with obl: 135 passed)
  tests/unit/test_niche_r6_auto_carrier_fit.py     }
  tests/unit/test_niche_d1_tilted_carrier.py       }   46 passed
  tests/unit/test_niche_p1_traced_tiltaware.py     }
  tests/unit/test_audit_lens.py                        52 passed
```

The last three groups are the ones this build could have broken without the
new suite noticing: the carrier-fit trims land in `_compute_carrier`
(`test_niche_r6` / `d1` / `p1`), and the function split rewrote two
`surface_model` docstring paragraphs and moved the band gates
(`test_audit_lens`, and the two RETARGETED introspection tests).

---

## 5. WALL CLOCK

Interleaved, min-of-reps, inside one process -- the protocol
`BUILD_TF_BANDED` S5.5 arrived at, for the same reason: an A-then-B timing on
a shared box reads the load.  Memory and wall clock in separate passes
(`tracemalloc`'s per-allocation hook inflates the clock badly).

The arm is `tangent_facet` + a finite-conjugate carrier -- the same
accumulator traffic and the same ASM legs as `carrier='auto'` without paying
the fit's whole-grid least squares on every rep, which would swamp the effect
under its own variance.  Design-121 three-surface group, production extent,
complex64, ratios against `ram + plain H`:

```
  N = 4096, 5 reps    ram+plainH 48.105 s (1.000)
                      memmap     44.886 s (0.933)
                      stream     50.197 s (1.043)
                      both       50.674 s (1.053)

  N = 8192            NOT OBTAINED -- see below
```

**`'memmap'` is at parity or better** -- the accumulators are written once per
surface in row order and read back in row order, which the page cache serves
without ever going to disk on a box with RAM to spare, and the allocator
pressure it removes is real.  0.933 is not a speedup claim; it is inside the
run-to-run spread that `BUILD_TF_BANDED` S5.5 measured at +-10 % for a NULL
comparison on this box, and the defensible statement is "not materially
slower".

**Streaming costs 4-5 % AT N = 4096, and that is the documented tradeoff, not
a regression.**  At 4096 `H` fits the 2 GiB cache entry cap, so the plain path
builds it once and every later call is a cache HIT while the streamed path
rebuilds the kernel every time.  That is precisely the regime where the lever
is not worth taking, which is why `LENS_STREAM_H` AUTO leaves it off below
N = 32768 -- where `H` is 8.59 GB, cannot be cached, and the plain path is
rebuilding it every call anyway.

The box carried the N = 16384 smoke and a two-tree sweep throughout, so these
are not clean-box seconds; the RATIO is what is quoted, and the interleaving
is what makes it meaningful.

**THE N = 8192 ROW IS NOT REPORTED, NOT APPROXIMATED.**  Its pass died under
the concurrent load before completing the interleave, and a single contended
observation printed beside a well-sampled row invites exactly the comparison
it cannot support -- the call `BUILD_TF_BANDED` S5.5 made about its own remap
row, and `BUILD_TF_REMAP` S6 about its N = 8192.  What the measured row
establishes stands on its own, and the band-loop-overhead argument that makes
these ratios flat in N is the sibling build's, unchanged: neither lever adds
per-band work that grows with the grid.

---

## 6. THE 32k ADMISSION, ARITHMETIC PRINTED

Both readings come from the same unchanged `_preflight_memory_check` call
against box B as `PROBE_D121` recorded it (136.6 GB total, **121.1 GB free**,
`FREE_RAM_FLOOR_BYTES = 20.0e9`), `N = 32768`, complex64,
`surface_model='tangent_facet'`, `carrier='auto'`, 8x8 emitters.  Only the
library under it differs.

```
### SHIPPED 5.39.1 (control -- the refusal must be preserved)
  preflight: need ~115.6 GB (peak 100.5 GB x 1.15, analytic lens
             [tangent_facet+66.1 GB]), have 121.1 GB free,
             headroom +5.5 GB against a 20.0 GB floor.
  VERDICT: REFUSED
```

That reproduces PROBE_D121's table to the printed digit (`need` 115.6 GB,
shortfall against the floor), which is what makes the comparison below an
apples-to-apples one rather than a claim about it.

```
### THIS BUILD
  levers OFF   need ~101.5 GB (peak 88.3 x 1.15, [tangent_facet+53.9])
               headroom +19.6 GB       REFUSED   <- by 0.4 GB, against the floor
  LEVER 1      need ~101.5 GB
               headroom +19.6 GB       REFUSED   (see the note below)
  LEVER 3      need  ~81.7 GB (peak 71.1 x 1.15,
                               [tangent_facet+53.9, stream_asm_H-17.2])
               headroom +39.4 GB       ADMITTED
  BOTH         need  ~81.7 GB
               headroom +39.4 GB       ADMITTED
```

Term by term at `LENS_AUTO_CARRIER_BRIGHT_FRAC = 0.25`, one grid = 8.59 GB:

```
  base work set   estimate_op_memory((N,N), complex64, 4)      4.00 grids  34.4 GB
  tangent_facet   max(accumulator peak, 'auto' fit peak)
                    accumulator peak   4.2   (0.7 with memmap)
                    fit peak           1.9 + 17.5*0.25 = 6.28
                  -> 6.28 grids                                            53.9 GB
  stream_asm_H    -2.00 grids                                             -17.2 GB
                                                                         --------
  peak                                                        8.28 grids  71.1 GB
  need = peak x 1.15                                                      81.7 GB
  free - need = 121.1 - 81.7 = 39.4 GB   >   20.0 GB floor       ADMITTED
```

**LEVER 1 SHOWS NO CREDIT IN THAT LINE, AND THAT IS THE MODEL BEING HONEST.**
At 25 % assumed bright support the `'auto'` fit peak (6.28) is above the
accumulator peak (4.2), so spilling the accumulators exposes the fit rather
than lowering the total.  The preflight takes a **MAX, not a SUM**, of the
route's two candidate peaks -- they are never live at the same instant, the
accumulators from surface 2 onward and the fit only during set-up.  A sum
would over-price the RAM case; crediting the memmap saving unconditionally
would UNDER-price the spilled one, which the measured table in S3.1 shows
directly (the memmap credit is 3.5 grids without the `'auto'` fit and 0.4 with
it).  At the 5 % bright support the design's beams may well have, Lever 1 is
worth its full 3.5 grids and the preflight would show it.

---

## 7. WHAT WAS REFUTED ALONG THE WAY

| # | attack / candidate | outcome |
|---|---|---|
| 1 | PROBE_D121 S5: the residual is the momentum accumulators + sag halos + ASM work arrays | **WRONG, and by a lot.**  Time-resolved sampling puts the peak inside the `carrier='auto'` FIT at set-up.  The accumulators are +4.13; the set-up was +9.2 on top.  A warmed peak reports a number and not a place, which is how the attribution went wrong. |
| 2 | PROBE_D121 S6: the anchor under-prices because it was taken on a singlet and design-121 groups have more surfaces | **THE GAP IS REAL, THE CAUSE IS NOT.**  The probe's own S4 measured +0.00 grids per extra surface and this build reproduces it.  The driver is the BRIGHT-SUPPORT FRACTION of the field, which no prescription-only anchor can capture. |
| 3 | "memmap the accumulators and the peak falls" | **NOT ON ITS OWN.**  With the whole-grid seed in place, `'memmap'` moved the measured peak by 0.00 grids -- it spills state that is not standing at the peak.  It became worth 3.48 grids only after the seed was banded.  Building Lever 1 first and measuring second would have shipped a lever that did nothing. |
| 4 | measure the spill with warmed `tracemalloc`, as the probe did | **CANNOT SEE IT, AND WOULD HAVE OVERSTATED IT.**  A mapped page is not a Python allocation.  tracemalloc reports -3.49 grids as if they had vanished; commit reports -3.48 as actually released; RSS reports +6.0 as still resident.  All three are in S3.1 and none is quoted alone. |
| 5 | stream `H` with the plain builder's chunk size | **MADE IT WORSE, +1.69 grids at N = 4096.**  The chunk is 10 % of the RAM budget, which resolves to the whole grid below N ~ 16384; that workspace is harmless when it is the only thing live and not when the spectrum is live beside it.  Capped at 4 Mi elements. |
| 6 | quote the measured -4.94 grid streaming credit at N = 32768 | **WOULD OVER-CREDIT BY 3 GRIDS.**  3 of the 4.94 are the plain builder's whole-grid chunk at N = 4096; at 32768 it self-limits to 17.5 %.  Priced at the derived -2.0. |
| 7 | the first two-tree run's "8960 arms, 0 differ" | **VACUOUS.**  3776 arms were refusal-vs-refusal because the fixtures named registry-absent design-121 glasses.  Caught by printing the refusal census; re-run with registered glasses, 8640 arms now return a field. |
| 8 | add the accumulator term and the `'auto'` fit term in the preflight | **WRONG SHAPE.**  They are alternatives, not addends -- the fit is live only during set-up and the accumulators only from surface 2.  Summing over-prices RAM; keeping only the accumulator term under-prices memmap by 0.83 grids, which the measured table shows.  The preflight takes a max. |
| 9 | gate the new preflight anchors on `_lumenairy_version_at_least(5, 40, 0)` alone | **WRONG ON A PRE-RELEASE WORKTREE**, where the code exists and no released version names it.  Gated on version OR signature, with the CREDITS additionally keyed to the same predicate the runner uses to decide whether to pass the keyword -- so a credit can never be claimed for something the call did not ask for.  That is the `screen_obliquity` phantom's failure mode. |
| 10 | the N = 16384 smoke's three different SHA-256s are a byte-identity failure of the levers | **NO -- THE PATH DOES NOT REPRODUCE ITSELF.**  The null comparison (same call, four times, lever arms interleaved) gives two distinct results at N = 4096, with both lever arms on the majority value and a NULL arm as the odd one out.  Caught by running the null at all; a smoke that only varies the thing under test cannot tell a difference from a drift.  Mechanism: the `'auto'` fit's `G = A.T @ A` over 1.8-28 M rows.  Not the FFT (`auto_promote` off, `FFTW_ESTIMATE`, no bad shapes, 2 plans).  Not this build (the 8960-arm two-tree's `'auto'` arms agree across both trees at grid sizes where the reduction stays short). (S4.2) |
| 11 | "`PYTHONPATH` pins the runner at the worktree" | **IT DOES NOT, AND THE FAILURE IS SILENT.**  `tx_design_study_sim.py` inserts the sibling `../Lumenairy` at `sys.path[0]` before importing, which beats `PYTHONPATH`.  Caught only because the preflight's signature probe read `accumulator_store=False` against a worktree that had the keyword; on the production run the symptom would have been 4.5 hours of numbers from the wrong build.  Fixed with a `LUMENAIRY_ROOT` override, an assert, and a `[lib]` line in the log naming the resolved directory -- a version string would not have caught it, since an untagged worktree reports the last release's version. |
| 12 | `close()` the store and unlink | **INSUFFICIENT ON WINDOWS.**  The unlink completes but leaves the directory in a pending-delete state, and the immediate `rmdir` fails on a directory `listdir` reports as empty.  Retried with backoff and warned on exhaustion.  Recorded rather than quietly fixed because "the files are gone but the directory is not" is the precise shape of it. |

---

## 8. FILES

| file | change |
|---|---|
| `lumenairy/elements/_lens_real.py` | `_AccumulatorStore` + `_VALID_ACCUMULATOR_STORE`; `_screen_obliquity_rows_any`; `apply_real_lens` split into a wrapper owning the store context + `_apply_real_lens_impl`; the banded accumulator seed; every persistent-accumulator allocation routed through the store; `stream_transfer_function` threaded to `_propagate_through_glass`; three new docstring parameter entries |
| `lumenairy/elements/_lens_traced.py` | `_compute_carrier(need_W=...)`; `_poly_and_grad(want_W=...)`; separable fit sample coordinates; in-place design-matrix weighting |
| `lumenairy/propagators/asm.py` | `_asm_apply_H_streamed` + `_ASM_STREAM_BAND_ELEMS`; `angular_spectrum_propagate(stream_transfer_function=...)` |
| `tests/unit/test_lens_memory_levers.py` | NEW.  144 tests |
| `tests/unit/test_tf_banded_halo.py`, `tests/unit/test_obl_banded_halo.py` | source-introspection tests RETARGETED to `_apply_real_lens_impl` (assertions unchanged; each gained a forwarding check so it cannot pass vacuously) |
| `CHANGELOG.md` | `[Unreleased]` |
| `docs/audits/PROBE_D121_ANALYTIC_32K_FOOTPRINT_2026_08_17.md` | committed (it was untracked; it is the measured basis this note argues with) |
| `docs/audits/BUILD_LENS_32K_MEMORY_2026_08_22.md` | this note |

**Outside the repo, not git** -- `Reverse_Symmetric_ASM/tx_design_study_sim.py`:
the re-anchored `tangent_facet` term (max-of-two-peaks), the `'auto'` fit
scaling law + `LENS_AUTO_CARRIER_BRIGHT_FRAC`, the two lever credits,
`_lumenairy_has_5_40_levers` / `_lumenairy_supports`, `LENS_ACCUM_STORE` /
`LENS_SCRATCH_DIR` / `LENS_STREAM_H` and their resolution helpers.

Not in git, in the scratchpad: `_d121_fix.py` (the shared design-121 fixture),
`_prof_mm.py` (S1), `_arms_mm.py` (S3), `_twotree_mm.py` (S4.1),
`_twotree_fit.py` (S4.2), `_time_mm.py` (S5), `_smoke16k_mm.py` (S9),
`_admit_mm.py` (S6), `_mat_mm.sh` (the S3 sweep driver).

---

## 9. THE PRODUCTION RUN -- ONE COMMAND AWAY, NOT LAUNCHED

The 4.5-hour N = 32768 run is deliberately NOT started here.  To start it:

```bash
cd d:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Reverse_Symmetric_ASM
# PYTHONPATH IS NOT ENOUGH -- see the note below.
export LUMENAIRY_ROOT=C:/tmp/lum_mm
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export POC_N_GRID=32768 POC_DX_UM=0.675
export POC_SURFACE_MODEL=tangent_facet POC_ANALYTIC_CARRIER=auto
python run_poc_119_120_v518.py
```

**`PYTHONPATH` COULD NOT HAVE POINTED THIS RUN AT THE WORKTREE, AND THAT WAS
SILENT.**  `tx_design_study_sim.py` inserts the sibling `../Lumenairy` at
`sys.path[0]` before importing, which beats `PYTHONPATH` outright -- so a run
launched with `PYTHONPATH` set would have imported the SHIPPED library while
looking correctly pinned.  Found while wiring the preflight credits: the
signature probe read `accumulator_store=False` against a worktree that plainly
had the keyword.  The insert now honours `LUMENAIRY_ROOT`, and the runner
PRINTS the resolved library directory and asserts it matches, so the run log
records which build produced the numbers:

```
  [lib] lumenairy 5.39.1 from C:/tmp/lum_mm/lumenairy
```

A version string alone would not have caught it -- a worktree carries the last
release's version until it is tagged, so both trees say 5.39.1.

Before it goes, set the two levers in `tx_design_study_sim.py` (they default
OFF, so the run is byte-identical to exp31's model without them):

```python
LENS_ACCUM_STORE = 'memmap'
LENS_SCRATCH_DIR = r'C:\scratch\lumenairy'   # a FAST LOCAL disk, not a share
LENS_STREAM_H = None                          # AUTO -> True at N >= 32768
```

`LENS_SCRATCH_DIR` carries one sequential write and one sequential read per
accumulator per surface: at N = 32768 that is 4 x 8.59 GB per surface.  On a
network share this lever turns a memory bound into an I/O bound, which is
exactly the trade PROBE_D121 S5 declined for an out-of-core FFT.

**A KILLED PROCESS LEAVES ITS SCRATCH FILES.**  Cleanup runs from a ``with``
block, and a ``with`` block does not run when the process is killed -- which
is the failure mode this run has (`ram_watchdog.py`).  Observed directly in
this campaign: two terminated probe processes left 3.1 GB behind.  Before a
long run, and after any kill, check `LENS_SCRATCH_DIR` for stale
`accum_<pid>_*.dat` and remove files whose pid is gone.  At N = 32768 a
killed run can leave ~34 GB there.

**THE RUN WILL NOT BE BIT-REPRODUCIBLE, and that is worth knowing before it
starts rather than after.**  `POC_ANALYTIC_CARRIER=auto` fits the carrier by
least squares over the bright support, and that reduction does not fix its
partitioning across calls (S4.2): a repeat of this run will not hash the same,
and neither will a comparison against exp31's stored field.  Read such
comparisons at a physical tolerance, not at the bytes.  If bit-reproducibility
is wanted, pin `OMP_NUM_THREADS=1` for the fit (it costs the whole run's
threading, so it is a diagnostic setting, not a production one) or supply a
CLOSED-FORM carrier -- a signed conjugate or a `TiltedCarrier` -- which has no
reduction at all.

**Re-check `LENS_AUTO_CARRIER_BRIGHT_FRAC` against the real field first.**  It
is the one preflight term that depends on the input rather than the
prescription, the preflight prints the value it assumed, and the run's own
first-lens field is what settles it.

---

## 10. SUITES

```
Windows 11 Pro 10.0.26200        AMD Ryzen 9 5950X   137.4 GB
python 3.14.6   numpy 2.4.4      lumenairy 5.39.1+ (worktree C:/tmp/lum_mm,
scipy 1.17.1    pyfftw 0.15.1    branch feat/lens-32k-memory off
numexpr 2.14.1                   origin/main a2652283)
```

| gate | result |
|---|---|
| `test_lens_memory_levers.py` (NEW) | **144 passed** |
| `test_tf_banded_halo` + `test_obl_banded_halo` (the two RETARGETED files) | **155 passed** |
| `test_tangent_facet` + `test_tangent_facet_remap` (unchanged) | **104 passed** |
| `test_obl_banded_halo` + `test_slant_chunk_byte_identical` | **135 passed** |
| `test_niche_r6_auto_carrier_fit` + `test_niche_d1_tilted_carrier` + `test_niche_p1_traced_tiltaware` | **46 passed** |
| `test_audit_lens` (the docstring / signature audits) | **52 passed** |
| two-tree vs `origin/main` @ 5.39.1 | **8960 arms, 0 differ** (8640 returning a field) |
| two-tree, the carrier fit alone | **12 arms, 0 differ** |
| NULL reproducibility of `carrier='auto'`, threaded | **2 distinct over 4 -- NOT REPRODUCIBLE** (S4.2; the shipped path) |
| NULL reproducibility of `carrier='auto'`, `OMP_NUM_THREADS=1` | **1 distinct over 4 -- REPRODUCIBLE**, and equal to the threaded majority |
| NULL reproducibility of `carrier=0.030` (closed form), N = 8192, threaded | **1 distinct over 4 -- REPRODUCIBLE** |
| `ruff check lumenairy/ tests/` | **All checks passed** |
| `xfail` / `skip` added | **ZERO** |
| pre-existing assertions relaxed | **ZERO** (two RETARGETED after the function split; see S2.1) |

**Path pinning.**  Every run in this note was made with `PYTHONPATH` pinned to
the worktree and `lumenairy.__file__` asserted in-process before anything
else; the two-tree comparisons ran the SAME script under both roots with that
assert keyed to each root, so a silently-imported installed wheel cannot have
produced any of these numbers.

**A NOTE ON WHAT WAS MEASURED AGAINST WHAT.**  The 8960-arm two-tree sweep was
run against the tree as it stood once every VALUE-AFFECTING change was in
place.  Five changes landed after it, and every one of them is inspectable as
non-value-affecting: the store's per-view reaping (where a mapping is released,
not what it contains), `_WARN_STACKLEVEL` (which FRAME a warning names -- the
sweep compares warning MESSAGES), `atexit=False` on the reapers, one comment
about the streamed `z = 0` branch, and docstring text.  A re-run on the exact
committed tree was started and is reported in S10 by whether it completed; the
claim above is stated against the tree that produced it, which is the
discipline `BUILD_TF_BANDED`'s closing note set.

**Box sharing.**  A peer session held a second worktree on this box for the
duration.  Memory is unaffected -- `tracemalloc` counts allocations and the
commit sampler reads this process only -- so every memory number here is a
clean reading.  Wall clock is affected and S5 bounds its claim accordingly.

Author: the 32k memory build, 2026-08-22.
