# BUILD -- the angle-true screen, row-banded with a halo

**2026-08-13.  Branch `fix/obl-banded-halo`, cut from `origin/main` @ `98abd37`
= v5.35.1, in the worktree `C:/tmp/lum_halo`.  Commit on the branch only -- no
merge, no push, no `gh`.**

Closes the memory hole `BUILD_SCREEN_OBLIQUITY_2026_08_11` opened.  That build
shipped the angle-true screen behind `carrier=`; this one makes it affordable,
without moving a single bit of its output.

---

## 0. VERDICT

> **THE TWO ROW-BAND DISQUALIFIERS ARE GONE, AND THE BANDED PATH IS
> BYTE-IDENTICAL TO THE WHOLE-GRID PATH.**
>
> **1. THE COST IS PAID.**  Warmed `tracemalloc` peak, biconvex fast singlet,
> sphere+tilt carrier, in float64 grids (`8*N*N` bytes), flat across
> N = 4096 / 8192:
>
> ```
>                                  v5.35.1   this build
>   banded-slant baseline (no carrier)  7.32       7.32     <- untouched
>   + carrier, applied, warn           32.31      16.13     -16.18
>   + carrier, applied, silent         31.31      11.77     -19.54
>   + carrier, estimator only          30.31      16.13     -14.18
>   extras over the baseline           24.99       8.80
> ```
>
> At N = 32768 one grid is 8.59 GB, so the term drops from **+215 GB to
> +76 GB** -- the 32k angle-aware run that OOMed the box on 2026-08-12
> now costs less than the pre-fix run did at half the grid.
>
> **2. AND NOTHING MOVED.**  9600 whole-grid-vs-banded configurations
> compared with `np.array_equal` on the returned complex field, on its
> dtype, on the guard's message text AND on the estimator accumulator the
> guard scores: **0 differences.**  86 tests in
> `tests/unit/test_obl_banded_halo.py`, all green.
>
> **3. THE WHOLE-GRID PATH IS PROVABLY UNPERTURBED.**  Below the N >= 4096
> auto-band threshold the same warmed probe reads the SAME seven peaks to
> the last printed digit before and after (14.13 / 18.13 / 32.25 / 31.25 /
> 30.25 / 28.25 / 30.25).  Nothing was traded away to buy the banded path.
>
> **4. ONE REAL WRONG ANSWER WAS CAUGHT IN THE BUILD, BY THE BUILD'S OWN
> ADVERSARIAL SWEEP.**  The first draft promoted the scalar momentum seed
> to a full grid at band 0, so bands 1..n read a float32 ARRAY of zeros
> where band 0 and the whole grid read a PYTHON float -- which under NEP 50
> drops the whole momentum triangle from float64 to float32.  Invisible at
> the float64 default, **5e-6 of field error at `sag_dtype='float32'`**.
> Fixed structurally (the source is pinned per surface) and pinned by
> `test_dtype_matrix_byte_identical`.

---

## S1. THE CLAIM, AND WHY IT WAS NOT ALREADY TRUE

`apply_real_lens` has row-banded its sag paths since v5.17.0
(`_narrow_chunk`, and `_slant_narrow_chunk` since v5.17.x): the per-surface
sag / OPD / refraction pipeline is evaluated `chunk_rows` at a time, so the
full-grid float64 transients never materialise.  v5.35.0's angle-true screen
disqualified both, with one clause each:

```python
    and not surface_frame and not _displaced and not _obl_active     # _narrow_chunk
    and not surface_frame and not _obl_active                        # _slant_narrow_chunk
```

so `carrier=` forced the whole-grid path.  That is not a small tax: it voids
the banded credit AND adds the obliquity block's own whole-grid arrays on
top.  Measured (2026-08-13, first probe, biconvex fast singlet, sphere+tilt
carrier, `'warn'`): 32.31 grids against a 17.32 banded-slant baseline --
+15.0 grids, +129 GB at N = 32768.  A 32k angle-aware run OOMed the box.

**The claim of this build:** the obliquity block can be banded exactly, with a
halo, and the banded output is byte-identical.

---

## S2. WHAT IS NOT POINTWISE, AND THE HALO EACH PIECE NEEDS

A band can be evaluated in isolation iff every operation is pointwise.  Five
things are not.  The build enumerates them rather than assuming:

| operation | where | halo |
|---|---|---|
| `xp.gradient(sag)` | the obliquity delta's own facet normal | 1 row |
| `xp.gradient(e_err)` inside `_screen_drift_opd` | the R1 drift term | **2 rows** of sag (it differentiates a quantity that is itself built from a gradient) |
| `xp.gradient(p0)` | the inter-surface re-imaging step (eq. 6) | 1 row on the persistent momentum grids |
| `bool(xp.any(sag))` | the flat-face skip | none -- a band scan with a short circuit |
| `bool(xp.all(q == 0))` | the zero-carrier structural skip | none -- same |

The 2-row case is the one a "1-row halo, like the slant path" reading gets
wrong.  Writing the dependency out: `_screen_drift_opd` needs `grad(e_err)` at
band rows `[r0, r1)`, so `e_err` at `[r0-1, r1]`, so `grad(sag)` at
`[r0-1, r1]`, so **`sag` at `[r0-2, r1+1]`**.  The build takes the halo width
from the live state -- `2 if (_obl_apply and _obl_drift_live) else 1` -- and
`_obl_drift_live` is fixed for the duration of a surface, so a band loop never
changes its mind mid-surface.

`np.gradient`'s interior stencil does not know how large the array is, so
widening the halo does not perturb the band's own rows; and the halo is
clipped at the true array edges, so rows 0 and `Ny-1` keep their natural
one-sided stencil in the first and last band.  That is the same argument
`test_slant_chunk_byte_identical` already pinned for the refraction leg, and
`test_odd_n_anisotropic_grid_byte_identical` re-pins it here at `cr=1`, where
EVERY row is simultaneously a band interior and a band boundary.

---

## S3. THE THREE PIECES OF STATE, AND HOW EACH IS BANDED

### S3.1 The carrier momentum field `q` -- evaluated, not stored

`_screen_obliquity_angle_field` builds two full-grid float64 arrays for any
carrier that is not a collimated tilt (+2 grids, +17 GB at N = 32768).  For a
finite-radius `TiltedCarrier` that is pure waste: `_tilted_carrier_parts` is
ANALYTIC and pointwise in `(x, y)`.

New `_screen_obliquity_row_evaluator` returns a `rows(r0, r1)` closure for
exactly that case and `None` for every other carrier vocabulary (scalar
conjugate, `'auto'` fit, explicit wavefront ndarray -- all of whose
`_compute_carrier` set-up is itself whole-grid).  Those keep the materialised
field and are simply sliced: no memory win, but the byte-identity claim is
unconditional, and `test_other_carrier_vocabularies_byte_identical` covers
them.

Byte-identity of the band: the y axis is rebuilt as `arange(r0, r1) - Ny/2`,
which is `(arange(Ny) - Ny/2)[r0:r1]` exactly in IEEE terms, and the SAME
`_tilted_carrier_parts` -> `asarray(float64) * n1` chain runs on it.
`test_row_evaluator_declines_non_analytic_carriers` asserts three bands
against the whole-grid field with `np.array_equal`.

If a surface later falls through to the whole-grid path (a decentered or
tilted face in the same prescription), `_obl_q_whole()` materialises the field
once, at that point, and the row evaluator retires.  Mixed prescriptions are
covered by `test_masks_and_mixed_paths_byte_identical`.

### S3.2 `_obl_p0*` and `_obl_u*` -- full-grid STORAGE, band UPDATE

These are genuinely full-grid state (the momentum a field point has
accumulated, and the drift its ray has accumulated); banding their storage
would be a different model, not a leaner one.  They stay full-grid float64 --
the honest floor -- and are updated a band at a time.

Two hazards, both real, both structural rather than numeric:

**(a) The band must read the PRE-SURFACE momentum.**  The R1 halo reads `p0`
at row `r0-1`, which belongs to the PREVIOUS band and has already been
overwritten by an in-place band update.  Fixed by holding each band's write
back one band (`_obl_p0_pending`), flushed at the next band and at
`_obl_end_surface`.

**(b) The scalar seed must not promote mid-loop.**  `_obl_p0*` starts as the
Python float `0.0` and becomes a grid at the first powered surface.  Promoting
in place at band 0 makes bands 1..n read a float32 array of zeros where band 0
and the whole grid read a Python float, and under NEP 50 that is not a
no-op: in `_facet_axial_momenta`, `p_sq = px*px + py*py` becomes a float32
ARRAY instead of a Python float, so `pz1 = sqrt(max(n1**2 - p_sq, 0))` is
computed in float32 instead of float64.

This was a genuine wrong answer, found by the build's own float32 sweep --
20 of 60 configurations differed, all and only the `sag_dtype='float32'`
ones, at 2e-6 to 5e-6 of field amplitude.  Fixed by pinning the source per
surface (`_obl_p0_src`) and writing a scalar-sourced surface into a FRESH
destination, so every band of that surface reads the same scalar the whole
grid reads.  `_obl_u*` gets the same treatment (`_ux_src`) for the same
reason.

### S3.3 `_obl_total` -- the guard's estimator accumulator

Stays a full grid (`Ny, Nx`), band-updated with `_obl_total[r0:r1] += _d`,
and the guard block at the end of `apply_real_lens` is untouched.  In-place
`+=` with a mixed dtype uses the same `same_kind` cast the whole-grid `+=`
uses, so it is elementwise identical.  `_run` in the test file spies on
`_screen_obliquity_rms_waves` and compares the accumulator itself with
`np.array_equal`, not just the warning text.

### S3.4 The inter-surface gap step

Pre-fix, the eq.-6 drift advance lived INLINE in the whole-grid surface body,
after the point at which both banded paths `continue` -- which is a second,
independent reason `carrier=` had to disqualify them.  It is now
`_obl_gap_advance(i, n2r)`, called from all three paths, banded internally
when `sag_chunk_rows` is live.  The `_pbx`/`_pby` re-imaging semantics
(`p0 - (ux * grad_x p0 + uy * grad_y p0)`, and the `getattr(p0, 'ndim', 0)`
scalar guard) are preserved verbatim; only the gradient is halo-banded.

### S3.5 The flat-face skip

`if _obl_active and bool(xp.any(sag))` is observable, not an optimisation: it
is what keeps `_obl_p0*` a pair of floats through a leading plate.  Banded, it
becomes `_obl_any_sag`, which walks bands and returns on the first non-zero --
so a powered surface pays one band and a genuinely flat one pays the scan and
then skips the whole block.  `_planoconvex` and `_leading_plate` cover both.

---

## S4. WHAT WAS DELIBERATELY LEFT ALONE

* **The numexpr gate.**  Both banded paths keep the whole-`E.size` decision
  (`E.size >= _NUMEXPR_MIN_SIZE`), because numexpr's `exp` differs from
  numpy's in the last bit and a per-band gate would break byte-identity at
  the threshold.  This is the `_slant_narrow_chunk` precedent, unchanged.
* **The NaN-sentinel zeroing of `opd`.**  Still `any -> where`, per band,
  after the obliquity term is added -- the same order as the whole grid.
  The obliquity block's OWN `_sag_ok` zeroing is applied to the HALO, not the
  band, or the halo rows would feed NaN back into the interior stencil.
  `test_nan_sentinel_at_band_boundaries` uses an oblate conic (k = +4) whose
  domain edge falls inside the grid, at `cr` in {1, 3, 7, 17} so the annulus
  crosses every band boundary.
* **The guard block.**  `_ensure_full_grids()` + `_screen_obliquity_rms_waves`
  at the end of the call are unchanged: the estimator sums over the whole
  pupil through `xp.sum`, so banding it would change the summation order and
  break byte-identity.  It is priced honestly in the envelope table instead.
* **The GPU path.**  Both gates still require `xp is np`, and the row
  evaluator is only built when `_chunk_grids` (which itself requires
  `xp is np`).  A CuPy field keeps taking the whole-grid path.
  `test_gpu_namespace_still_falls_through_to_whole_grid` pins the gates
  structurally.

One lifetime change WAS made outside the band loop: `_obl_p0*`, `_obl_u*` and
`_obl_q*` are set to `None` after the surface loop, before the Seidel block
and the guard.  They are dead there.  This is a pure lifetime edit -- no
arithmetic reads them again -- and the N < 4096 probe row confirms it does not
move the whole-grid peak (which is reached inside the surface loop).

---

## S5. EVIDENCE

### S5.1 The adversarial byte-identity matrix (the load-bearing one)

`_adv.py`, run once against the finished build: 10 prescriptions x 3 grids
(including odd N and `dx != dy`) x 2 carriers x 4 screen combinations x 2
policies x 2 apply/estimate x 5 band sizes (1, 3, 7, N, N+7) x 2 complex
dtypes.

```
  total 9600   fails 0
```

Compared per configuration: the returned field (`np.array_equal`), its dtype,
and the sorted set of guard / near-grazing warnings.

### S5.2 The test file

`tests/unit/test_obl_banded_halo.py` -- **86 passed**.

* the core matrix: {paraxial, slant, fresnel} x {plane, sphere carrier} x
  {warn, silent} x {applied, estimator-only} x {biconvex, one flat face};
* NaN sentinel at band boundaries (oblate conic, `cr` 1/3/7/17);
* clear aperture, aperture stop, leading flat plate, a decentered face that
  falls through mid-prescription;
* odd N (65 and 515), `dx != dy`, band sizes 1 / 3 / N / N+7;
* complex64 and complex128 fields, float64 and float32 geometry;
* the non-analytic carrier vocabularies;
* the AUTO threshold on both sides (with the threshold monkeypatched down for
  the banded arm, so the test does not need a multi-GB whole-grid reference);
* proof the band loop is TAKEN: no full-grid meshgrid, and `np.gradient` is
  called once per band per surface rather than once per surface;
* the GPU fall-through gates;
* the memory ratio.

### S5.3 The supplementary option sweep

The option axes that do NOT touch the sag block but do share the call --
`seidel_correction` (which runs after the loop, downstream of the accumulator
release), `absorption`, `bandlimit`, and the `sas` / `fresnel` /
`rayleigh_sommerfeld` propagators -- crossed with both carriers, both screens
and `cr` in {1, 13}: **48 configurations compared, 0 differences** (8 more
combinations refused by the kwarg validator before reaching either path, as
they do without a carrier).

### S5.4 The suites

```
  tests/unit/test_obl_banded_halo.py                86 passed   (new)
  tests/unit/test_screen_obliquity.py       )
  tests/unit/test_slant_chunk_byte_identical.py )  111 passed
  tests/unit/test_lens_chunked_sag.py       )
  tests/unit/test_hammer_h1_slant_obliquity.py  )   16 passed
  tests/unit/test_niche_p3_pointwise_obliquity.py )
  tests/unit -k lens                              532 passed, 2 skipped
                                                  (both skips pre-existing:
                                                   host-specific W5 digests,
                                                   a lock-exemption pin)
  ruff check lumenairy/elements/_lens_real.py
              tests/unit/test_obl_banded_halo.py   All checks passed
```

### S5.5 The envelope, re-measured

`obl_mem_probe3.py` (the 2026-08-13 probe plus a warm-up pass and two extra
columns).  The warm-up matters: the FIRST `apply_real_lens` of a process also
pays FFT-plan and lazy-import allocations -- about 10 grids at N >= 4096 --
which land in the `tracemalloc` peak.  The original probe measured its
baseline first (warm-up included) and the obliquity arms afterwards, so its
`B - A = 15.0` UNDER-states the same build's true delta by that much.  Warmed,
both builds, float64 grids:

```
  N       P      A      B     Bs     Bp      C      D
  ------- paraxial/slant baselines, then carrier arms ------------------
  v5.35.1 (whole-grid obliquity)
  2048  14.13  18.13  32.25  31.25  30.25  28.25  30.25
  4096   6.38   7.32  32.31  31.31  30.31  28.31  30.31
  8192   6.38   7.32  32.31  31.31  30.31  28.31  30.31

  this build (banded-with-halo obliquity)
  2048  14.13  18.13  32.25  31.25  30.25  28.25  30.25   <- identical
  4096   6.38   7.32  16.13  11.77  16.13  15.19  16.13
  8192   6.38   7.32  16.13  11.77  16.13  15.19  16.13

    P  paraxial, no carrier            Bp slant, PLANE carrier, warn
    A  slant, no carrier               C  paraxial, sphere carrier, warn
    B  slant, sphere carrier, warn     D  slant, sphere carrier, warn,
    Bs the same, 'silent'                 screen_obliquity=False
```

Read off the new coefficients, as extras over the corresponding no-carrier
baseline, at N >= 4096:

```
  carrier, correction applied, estimator live (warn/error)   +8.80 grids
  carrier, correction applied, 'silent'                      +4.7 / +4.45
  carrier, estimator only (screen_obliquity=False, warn)     +8.80 grids
```

and the two things that USED to matter and no longer do: the term is now the
same for the paraxial and slant screens (`C - P == B - A == 8.80`), and the
same for a plane and a finite-radius carrier (`Bp == B`) -- the old +2.0
sphere-carrier surcharge is gone because the field is row-evaluated.

Below the auto-band threshold the whole-grid block runs exactly as before, and
the N = 2048 row is identical pre- and post-fix.  Re-measured warmed there, the
extras are **+14.13** (warn) / **+13.13** ('silent') / **+12.13**
(estimator-only) -- which is what the runner preflight now prices for that
branch, since the shipped 12/9+2+1 was differenced against an unwarmed
baseline.

### S5.6 The runner preflight

`Reverse_Symmetric_ASM/tx_design_study_sim.py` (outside the repo, not git) --
`_preflight_memory_check`'s `screen_obliquity` extras term is now
VERSION-AWARE via a new `_lumenairy_version_at_least(major, minor, patch)`
helper reading `la.__version__`:

* **<= 5.35.2** keeps the 2026-08-13a coefficients verbatim
  (`(12.0 if slant/fresnel else 9.0) + 2.0`, `+1.0` for the estimator);
* **> 5.35.2** uses 2026-08-13b: `8.8` / `4.7` grids when row-banded
  (N >= 4096), `14.2` / `13.2` when not;
* an unparseable version ('unknown', a dev checkout) is treated as OLD, i.e.
  the LARGER envelope -- the preflight exists to refuse runs that will not
  fit, so an unknown build must never be priced cheap;
* the over-budget hint now says which regime it priced, and tells a
  <= 5.35.2 user that upgrading is cheaper than dropping the carrier.

Both anchors are dated in the docstring; the 2026-08-13a text is kept for
history, with a note that its baseline was unwarmed.

---

## S6. REFUTATION ATTEMPTS

Each of these was an attempt to BREAK the byte-identity claim, not to confirm
it.

| # | attack | outcome |
|---|---|---|
| 1 | float32 geometry (`sag_dtype='float32'`) | **BROKE IT.** 20/60 configs differed at 2e-6..5e-6. Root cause = the scalar-seed promotion (S3.2b). Fixed; re-run clean. |
| 2 | `cr=1` -- every row is a band boundary, so no gradient stencil is ever purely interior to a band | clean, all screens, all carriers |
| 3 | NaN sentinel on an annulus crossing every band boundary (oblate k=+4 conic) | clean; the zeroing had to move to the HALO, which it did from the first draft |
| 4 | the R1 drift term's gradient-of-a-gradient | forced the 2-row halo; a 1-row halo would have been silently wrong only when the drift is live, i.e. never on surface 0 -- caught by construction, pinned by the `cr=1` and leading-plate cases |
| 5 | band-boundary staleness of `p0` (row `r0-1` already overwritten) | forced the one-band deferred write; a `cr=1` run with the deferral removed is the failing control |
| 6 | leading FLAT plate -> drift goes live while `p0` is still a scalar | clean (the R1 term runs against a scalar momentum; `_obl_band_of` passes scalars through) |
| 7 | decentered / tilted face falling through to the whole-grid path MID-prescription | clean -- the accumulators hand back and forth; `_obl_q_whole()` materialises the carrier field at that point |
| 8 | estimator-only path (`screen_obliquity=False`) -- the correction is not applied but `_obl_total` and `p0` still accumulate | clean, and the accumulator itself is compared, not just the message |
| 9 | odd N (65, 515), `dx != dy`, band wider than the grid (N+7) | clean |
| 10 | complex64 + fresnel (which promotes the field to complex128 mid-surface) | clean, dtype included in the assert |
| 11 | GPU namespace | still falls through: both gates keep `and xp is np`, the row evaluator is gated on `_chunk_grids`; pinned structurally |
| 12 | does the band loop actually RUN, or is it silently falling back? | `np.gradient` count = whole-grid count x n_bands; zero full-grid meshgrids on a single-surface banded angle-aware run |
| 13 | the AUTO (`sag_chunk_rows=None`) resolution | unchanged on both sides of the threshold; asserted directly |

The one that fired (#1) is the reason the build does not simply promote
accumulators in place, and is the finding this document exists to record: an
accumulator's STORAGE dtype is part of the arithmetic, not just of the memory
bill.

### S6.1 Controls -- each guard REMOVED, and the failure it lets through

A guard that is never observed to fail is indistinguishable from a guard that
does nothing.  Each of the three structural guards was deleted in turn and the
byte-identity probe re-run (12 configurations: {float64, float32 geometry} x
{paraxial, slant} x `cr` in {1, 3, 13}, N = 96, sphere carrier, applied,
`'silent'`).

```
  as shipped                            0/12 differ   worst |dE| = 0
  A  halo forced to 1 row              12/12 differ   worst |dE| = 2.221e-05
  B  p0 band written immediately       12/12 differ   worst |dE| = 1.971e-03
  C  scalar p0 promoted in place        6/12 differ   worst |dE| = 1.849e-06
  restored                              0/12 differ   worst |dE| = 0
```

C fails on exactly the six float32-geometry configurations and no others,
which is the signature the diagnosis predicts: at float64 the promoted zeros
array and the Python float give the same arithmetic, and only the narrower
storage exposes the difference.  That is also why this failure could sit in a
shipped build undetected -- the default geometry dtype is float64.
