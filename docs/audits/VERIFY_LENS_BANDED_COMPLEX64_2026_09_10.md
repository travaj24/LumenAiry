# VERIFY -- the two 5.44.0 traced-lens changes, re-measured independently

**Date** 2026-09-10 **Subject** the two `### Changed` entries of the
5.44.0 CHANGELOG block (library commit `4e8ea24`, released at `50824e9`):

1. the traced ROW-BAND assembly now also serves `amplitude_model='ray_density'`
   AND the inverse-characteristic evaluator
   (`AUDIT_TRACED_MEMORY_2026_08_09` row 3, declared closed);
2. complex64 THROUGH the carrier chain (that audit's sec 3 / row 12, declared
   closed).

Both shipped without prior independent re-measurement.  Nothing below is read
off the shipped tests: every number is re-measured with the probes in
`validation/probe_verify_lens_5440/` (see its README), on fixtures built for
this verification, except where a shipped fixture is deliberately re-run to
check a recorded constant.

## Arms

| arm | tree | `lumenairy.__file__` asserted | build |
|---|---|---|---|
| WITH | `C:/tmp/lum_vlens` @ `50824e9` (branch `verify/lens-banded-complex64`) | yes, every script | Windows 11, py 3.14.6, numpy 2.4.4, scipy 1.17.1 |
| WITHOUT | `C:/tmp/lum_v5430` @ `v5.43.0` (`78e4091`, worktree, removed at the end) | yes | same |
| SECOND BUILD | `\\wsl$` `~/lumvenv` on `/mnt/c/tmp/lum_vlens` | yes | WSL Ubuntu, py 3.12.3, numpy 2.4.6, scipy 1.17.1 |

Thread caps `OMP/OPENBLAS/MKL_NUM_THREADS=1` on every measurement that could
depend on them, except V3, where the thread count IS the measurement.  (V13 /
V16 / V17 are pure elementwise / logic probes with no BLAS in them and were
run uncapped.)

---

## 1. Verdict table

| # | claim (as written) | verdict | key numbers |
|---|---|---|---|
| C1 | non-banded traced calls are unchanged across the release | **CONFIRMED** | 28/28 calls (14 fixtures x `sag_chunk_rows` {0, None}) bit-identical v5.43.0 vs 50824e9; every diagnostic equal (V1) |
| C2 | the named exception: a banded call at the shipped default used to select the coarse-Newton incumbent, 2.19e-02 on the S10 carrier fixture | **CONFIRMED, exactly** | v5.43.0 reproduces **2.1886e-02** (carrier) and **2.4216e-01** (`carrier=None`) on the S10 fixture; on my own fixtures 1.354e-01 (screen), 4.324e-02 (screen+carrier), and **1.1909e-01 at the true `sag_chunk_rows=None` default, N=4096** (V8, V10) |
| C3 | the 5.44.0 banded answer IS the evaluator's answer, not a third one | **CONFIRMED, by hash** | 5.44.0 banded hash == 5.44.0 whole-grid hash == **v5.43.0 whole-grid hash**, bit for bit, on every fixture incl. N=4096 AUTO (V8, V10) |
| C4 | banded == whole-grid, byte-identical, field AND diagnostics, both inversion routes | **CONFIRMED, on two builds** | **648 pairwise comparisons, 0 mismatches** -- 432 on Windows (18 case/route pairs x up to 7 band heights x 4 threshold regimes) and 216 on WSL (py 3.12.3 / numpy 2.4.6) -- on field hash, `n_out_of_domain`, `gate_open`/`engaged` and the full warning list (V2) |
| C5 | the census median / min / max / sign scan is order-independent | **CONFIRMED, made load-bearing** | with the caustic floor forced to 1.5x the median (so the median enters EVERY pixel) the banded field still equals the whole-grid field bit for bit at band heights 1, 3, 7, 32, 128, N (V2 `--medianbite`); the sign scan's two algorithms agree on 5868 engineered comparisons (V13) |
| C6 | D14/D15 determinism preserved on the banded path | **CONFIRMED** | identical field hash at `OPENBLAS_NUM_THREADS` 1 / 2 / 4, in separate processes, on 6 cases incl. banded ray-density + evaluator (V3) |
| C7 | "the model is evaluated 7/4 times" | **CONFIRMED, exactly** | 7.000 channel-evaluations per exit pixel banded vs 4.000 whole-grid on the ray-density branch; 3.000 vs 3.000 on the screen branch (V4) |
| C8 | the row-band assembly cuts memory (22.6 -> 10.2 grids at N=512) | **CONFIRMED** | 22.63 -> 10.21 grids (Windows), 22.50 -> 10.21 (WSL); at N=4096 **22.27 -> 9.05 grids (2850.6 -> 1158.3 MiB, -59 %)** (V4, V9) |
| C9 | "wall time ... ~10 % higher on the evaluator route (the 7/4 evaluation)" | **REFUTED as stated** | the ~10 % holds for the SCREEN branch (band/whole **1.077** at N=4096) but NOT for the ray-density branch the 7/4 belongs to: **1.385** at N=4096 sub=32, **1.324** at N=2048 sub=16.  And the banded SCREEN call at the shipped default -- the production one -- went **10.529 s -> 21.890 s (+108 %)** across the release, unpriced (V11, V15, V20) |
| C10 | complex128 carrier chain byte-identical | **CONFIRMED** | 9/9 cases -- 2- and 3-group chains, parabola reference, final free leg, exact focus readout, the readout called directly, both crop branches -- bit-identical across builds, including every per-stage `dx`/`R` (V14) |
| C11 | complex64 phasor is one float32 rounding from complex128, FLAT in the argument | **CONFIRMED** | max \|dz\| **4.21e-08**, flat from 2.0e+02 to 2.0e+05 rad of argument, on all four helpers; analytic ceiling sqrt(2)·eps32/2 = 8.43e-08 (V5) |
| C12 | the float32-ARGUMENT control grows with the argument | **CONFIRMED (physics), recorded numbers REFUTED** | control 4.2e-08 -> 7.8e-03 over 2.0e+02 -> 2.0e+05 rad; but on the SHIPPED fixtures the arguments are 22.3 / 820 rad (not 3.6e+02 / 1.3e+04) and the control reads 9.87e-07 / 3.05e-05 (not 1.5e-05 / 4.9e-04) (V5, V9) |
| C13 | `dtype=None` / `complex128` byte-identical to the shipped `np.exp` | **CONFIRMED** | all four helpers, all seven ladder rungs (V5); and `_phasor_rows` is band-size invariant AND exactly equals a whole-grid `np.exp(float64).astype(complex64)` at band sizes 16 .. N rows (V16) |
| C14 | a complex64 chain stays complex64 end to end | **CONFIRMED for the dtype** | 2-group chain returns complex64; v5.43.0 returned complex128 for the same input (V6, V7) |
| C15 | six-leg chain error grows ~linearly at ~1.9e-07/leg, 1.1e-06 after six | **CONFIRMED in kind, coefficient is fixture-dependent** | my own six-leg chain: linear, **4.58e-08 per leg, 2.75e-07 after six** (V6) |
| C16 | the mixed-precision chain stays under the 4e-05 energy honesty bar on a REAL prescription | **CONFIRMED** | real two-group traced chain, N=512: relative L2 of the field **1.158e-07**, relative total-power change **7.69e-09** -- 3 and 4 decades under the derived bars (V6) |
| C17 | "requesting complex64 saved 0.0 GB" is closed | **BOUNDED -- partially closed** | v5.43.0: **0.0 MiB saved (0.0 %)**, field complex128.  50824e9: **44.1 MiB of 376.9 MiB (11.7 %)**, field complex64 -- but **5 full-grid complex128 phasors per two-group chain survive** at `carrier.py:1663` (V7) |
| C18 | `_fourier_upsample_crop` "transforms in complex128 and narrows back on return" | **REFUTED** | on numpy >= 2.0 `np.fft.fft2(complex64)` returns complex64: the transform pair now runs in SINGLE precision.  Measured 2.6 x eps32 x peak vs a complex128 transform of the same samples (a narrow-on-return would be <= 0.5 x); rel L2 1.7e-07 (V6, `V17`) |

No **silent-wrong** was found in either change.  The byte-identity claims that
carry the risk all hold, on both builds.

---

## 2. WITHOUT-ARM identity (task 1)

### 2a. Non-banded traced calls -- `V1`

14 fixtures on a biconvex N-SF11 singlet at 1.064 um (a different glass,
wavelength, f/# and grid from every shipped fixture), each run at
`sag_chunk_rows=0` and `sag_chunk_rows=None` (N=320 < 4096, so AUTO resolves
to whole-grid): both amplitude models, `preserve_input_phase` True and
`'remap'` in both samplings, a spherical-carrier input, a decentred `origin`,
a `tilt_aware_rays` decentred congruence, both inversion routes
(`inverse_map` default and `False`), a caustic-bearing strong singlet whose
map the evaluator's own G2 guard refuses, and `ray_subsample=1`.

**28 / 28 field hashes bit-identical** across v5.43.0 and 50824e9.  Every
other recorded field -- `sum |E|^2`, `max |E|`, `engaged`, `gate_open`,
`refused`, `n_out_of_domain`, the warning list -- identical as well.

One difference the hash comparison cannot see, found by `V11` and reported as
defect **D1** below: the *attribution* of two of the ray-density self-check
warnings moved from the caller to `_lens_traced.py`.

### 2b. The complex128 carrier chain -- `V14`

| case | hash (both builds) |
|---|---|
| `chain2_sphere` | `577e209a0757a57bab89857f` |
| `chain2_parab` (`carrier_reference='parabola'`) | `ae7983bcf984d93609759cf1` |
| `chain3_sphere` (three groups) | `c6dda90c08bd2f8830a348b9` |
| `chain2_final_leg` (`final_distance=6 mm`) | `4cd9552cefd80598e5a04fd6` |
| `chain2_readout` (exact focus readout) | `ab751a2265db2c814047852c` |
| `readout_direct` | `65b759bab27be8cac1cecd2c` |
| `crop_128_256` / `crop_256_128` / `crop_256_256` | `5baf422b...` / `4c5d9fd1...` / `b3d0a0b2...` |

Bit-identical across the release, including every per-stage `dx` and `R_out`
the chain reports.  The complex128 path is untouched.  **CONFIRMED.**

### 2c. The named exception, quantified -- `V8`, `V10`

On the S10 fixture itself (`test_niche_s10_sibling_patterns.py`'s singlet,
N=256, dx=4 um, carrier 30 mm):

| build | whole-grid (`rows=0`) | banded (`rows=32`) | rel |
|---|---|---|---|
| v5.43.0, carrier | `f7680b0c…` eng=True | `800baab3…` eng=False gate=False | **2.1886e-02** |
| v5.43.0, `carrier=None` | `0caa26d9…` eng=True | `8a669460…` eng=False gate=False | **2.4216e-01** |
| 50824e9, carrier | `f7680b0c…` eng=True | **`f7680b0c…`** eng=True | 0 |
| 50824e9, `carrier=None` | `0caa26d9…` eng=True | **`0caa26d9…`** eng=True | 0 |

The CHANGELOG's 2.19e-02 reproduces to five figures, and the 5.44.0 banded
field is the **v5.43.0 whole-grid evaluator field, bit for bit** -- not a
third answer.  On my own fixtures the pre-change route difference is 1.354e-01
(screen), 4.324e-02 (screen + spherical carrier); at the true production
default (`sag_chunk_rows=None`, N=4096, AUTO banding) it is **1.1909e-01**,
and after the change the AUTO-banded field equals `91da5e59c46beb1d967b28b3`
on both builds' whole-grid arm.

---

## 3. The banded claim (task 2) -- `V2`, `V3`, `V13`

Nine fixtures x two inversion routes, band heights {0, 7, 32, 128, N} (plus
{1, 3} in the two forced regimes), compared pairwise against the whole-grid
arm on the field hash and on `n_out_of_domain`, `gate_open`, `engaged` and the
FULL warning list:

| regime | what it drives | comparisons | mismatches |
|---|---|---|---|
| plain | shipped thresholds | 90 | **0** |
| `--forced` | the three ray-density self-checks and the niche-D9 origin verdict driven over their thresholds (2-3 warnings fire per case) | 90 | **0** |
| `--fold` | caustic floor 0.999 x median, max/min 1+1e-7 -- the **fold-caustic warning fires** on every ray-density case | 126 | **0** |
| `--medianbite` | caustic floor **1.5 x median**, which exceeds max \|det J\|, so the capped amplitude is `\|E_in\|/sqrt(1.5 x median)` at **every pixel** -- the census median is now IN THE FIELD | 126 | **0** |

Repeated on the SECOND build (WSL, py 3.12.3 / numpy 2.4.6): plain 90 and
`--medianbite` 126 comparisons, **0 mismatches** -- 648 in total across the
two builds.

The `--medianbite` regime is the load-bearing one: it changes the field hash
(`eecefe42…` -> `f0d4c499…`), so the median genuinely reaches the output, and
the banded and whole-grid fields are still bit-equal at band heights down to
**one row**.

Fixtures include a decentred/tilted group, `origin` set, `remap_sampling`
`'lattice'` and `'full'`, an automatic OPL piston on every call (the
prescription is a real singlet), an aperture from the prescription, and a
caustic-bearing strong singlet.

### The two-row-halo sign scan -- `V13`

The only non-pointwise piece besides the median.  Both algorithms transcribed
from the library (the closure's whole-grid scan and the band loop's per-band
scan with its one-row halo) and run against each other on 489 fields x 12 band
heights = **5868 comparisons, 0 mismatches**.  The fields are engineered for
the failure mode: a single sign flip placed at every row boundary in turn,
with the horizontal and vertical neighbour pairs selectively masked out, and a
sparse two-pixel case whose only finite pair straddles the boundary.

### Determinism (D14 / D15) -- `V3`

Each thread count in its own subprocess (`OPENBLAS/OMP/MKL/NUMEXPR_NUM_THREADS`
are read at process start):

| case | threads 1 | 2 | 4 |
|---|---|---|---|
| `rd_imap_band32` | `eecefe42276fe6b7b4c67d2a` | same | same |
| `rd_imap_band7` | same | same | same |
| `rd_imap_whole` | same | same | same |
| `rd_newton_band32` | `6bd918bc340591460ca8c503` | same | same |
| `rd_remapfull_band32` | `4da384eec1ee92020afb1efe` | same | same |
| `screen_band32` | `203aee7468521ed5bbee545d` | same | same |

**No regression.**  The band loop introduces no thread-dependent reduction,
and the banded hash equals the whole-grid hash at every thread count.

### The one quantity that is NOT bit-identical -- `V12`

The niche-D9 origin support measurement accumulates its two sums band by band
(the CHANGELOG names this).  Measured:

* the percentage the two paths PRINT is identical to six decimals at band
  heights 0, 1, 7, 32, 128, N;
* a synthetic bound on the summation-order difference for arrays of this shape
  and magnitude: **<= 3.78e-16 relative** (worst at one-row bands; exactly 0 at
  32, 128, N).

The D9 tolerance is 1e-9, so the accumulated fraction would have to sit within
~1e-15 of the tolerance for the decision to differ.  **BOUNDED, not a defect.**

---

## 4. The 7/4 evaluation and the memory claim (task 3) -- `V4`

`InverseCharacteristic.eval_into` counted by monkeypatch, N=1024, sub=16,
dx=6 um:

| route | calls | channel-evaluations per exit pixel | channel sets |
|---|---|---|---|
| ray-density, whole-grid | 1 | **4.000** | `(0,1,2,3)` |
| ray-density, band 32 | 64 | **7.000** | `(0,1,2,3)` then `(0,1,2)` |
| ray-density, band 7 | 294 | **7.000** | same |
| screen, whole-grid | 1 | 3.000 | `(0,1,2)` |
| screen, band 32 | 32 | 3.000 | `(0,1,2)` |

**Exactly 7/4 on the ray-density branch, one pass elsewhere.  CONFIRMED.**
(On v5.43.0 the same script reads 4.000 for every ray-density arm -- the band
path was the whole-grid path -- and **0 calls** for `screen_band32`, the
incumbent.)

Whole-call `tracemalloc` peak, warm, one grid = `8 N^2` bytes:

| N, sub, dx | route | 50824e9 whole | 50824e9 banded | v5.43.0 whole | v5.43.0 banded |
|---|---|---|---|---|---|
| 1024, 16, 6 um | ray-density | 47.06 grids (376.5 MiB) | **20.15 (161.2 MiB)** | 47.06 | 47.06 (band refused) |
| 1024, 16, 6 um | screen | 46.06 (368.5 MiB) | **11.41 (91.3 MiB)** | 46.06 | 9.20, `engaged=False` |
| 4096, 32, 1.5 um | ray-density | 22.27 (2850.6 MiB) | **9.05 (1158.3 MiB)** | 22.27 | 22.27 (band refused) |
| 4096, 32, 1.5 um | screen | 17.13 (2192.1 MiB) | **9.05 (1158.3 MiB)** | 17.13 | 9.05, `engaged=False` |
| 512, 16, 12 um (the shipped test's own fixture) | ray-density | 22.63 | **10.21** | -- | -- |

The shipped test's recorded 22.6 -> 10.2 grids reproduces (22.63 -> 10.21 on
Windows, 22.50 -> 10.21 on WSL).  At the production threshold N=4096 the
saving is **13.2 grids = 1692 MiB, 59 % of the whole-grid peak**.

Two honest riders:

* the banded SCREEN route at the shipped default costs **+2.2 grids** at
  N=1024 (9.20 -> 11.41) and is level at N=4096 (9.05 -> 9.05) relative to
  v5.43.0 -- because it now runs the evaluator instead of the incumbent.  The
  memory is not worse; the *inversion* is different (and better).  What it
  costs is time (section 5).
* the CHANGELOG's prototype numbers (24 -> 1.2 and 30 -> 3.7 grids for the
  assembly's own transient, and the ~64 GB -> ~8 GB extrapolation to the 16384
  fine leg) are from `../Lumenairy_prototypes/`, which is not in this
  repository.  **NOT VERIFIED**; my whole-call peaks are a different quantity
  and are given above instead.

---

## 5. Wall time -- `V11`, `V15`, `V20`

CHANGELOG: *"Wall time is neutral on the coarse route and ~10 % higher on the
evaluator route (the 7/4 evaluation)."*

Best of three, no `tracemalloc`, warm, single-threaded, AUTO band height
(`max(256, N//16)`), the machine otherwise idle:

**N=4096, sub=32, dx=1.5 um -- the production regime (`V15`):**

| build | route | whole-grid | AUTO-banded | band / whole | banded arm's inversion |
|---|---|---|---|---|---|
| 50824e9 | screen | 20.323 s | **21.890 s** | **1.077** | evaluator |
| v5.43.0 | screen | 20.209 s | **10.529 s** | 0.521 | coarse-Newton incumbent |
| 50824e9 | ray-density | 28.472 s | **39.427 s** | **1.385** | evaluator |
| v5.43.0 | ray-density | 21.121 s | 30.054 s | 1.423 | evaluator (traced assembly not banded; only the ANALYTIC leg bands) |

**N=2048, sub=16, dx=3 um, uncontended (`V11`):** ray-density band/whole =
**1.324** (`rows=32`) and **1.310** (AUTO `rows=256`); on v5.43.0 the same
comparison reads 0.989 (there is no traced banding on that route there).

Readings:

* the **~10 % is right for the SCREEN branch only** (1.077 measured), which
  runs the evaluator in ONE pass.  On the ray-density branch -- the branch the
  bullet actually names, because that is where the 7/4 evaluation lives -- it
  is **+32 % to +39 %**.  The bullet conflates the two routes.
* the number that matters for the runner is the **cross-build banded screen
  call at the shipped default: 10.529 s -> 21.890 s, +108 %**, and it is well
  controlled (the two builds' whole-grid screen arms agree to 0.6 %).  That is
  the price of the route change, not of banding, and it buys the evaluator's
  answer instead of the incumbent's; it is not priced in the entry at all.
* cross-build banded ray-density, same bits out on both builds
  (`08f9aa00b567ed87658cf8ca`, `V8 --auto`): 30.054 s -> **39.427 s, +31 %**.
* the two builds' whole-grid RAY-DENSITY arms in the table above differ
  (21.1 s vs 28.5 s) on a byte-identical path.  `V20` isolates it in a fresh
  process with ray-density run FIRST: **28.333 s (50824e9) vs 27.391 s
  (v5.43.0), +3.4 %** -- inside this box's ~15 % run-to-run spread.  The V15
  discrepancy is an ORDERING artifact (whichever model runs second in a
  process is slower); both builds were run in the same order, so the
  cross-build ratios above stand, but the absolute seconds should be read with
  that spread in mind.  The whole-grid ray-density path did NOT get slower.

## 6. The complex64 chain (task 4)

### 6a. The phasor ladder -- `V5`

N=512, dx=1.9 um, R chosen per rung so the radial argument hits the target.
Error = max \|c128 - c64 widened\|:

| true max argument (rad) | `_radial_carrier_phase` c64 | float32-ARGUMENT control | ratio |
|---|---|---|---|
| 2.0e+02 | 4.202e-08 | 4.202e-08 | 1.0 |
| 6.0e+02 | 4.192e-08 | 4.292e-08 | 1.0 |
| 2.0e+03 | 4.179e-08 | 4.236e-08 | 1.0 |
| 6.0e+03 | 4.195e-08 | 2.442e-04 | 5 821 |
| 2.0e+04 | 4.176e-08 | 9.766e-04 | 23 385 |
| 6.0e+04 | 4.207e-08 | 1.953e-03 | 46 424 |
| 2.0e+05 | 4.180e-08 | 7.813e-03 | 186 891 |

The other three helpers read 3.66e-08 .. 4.20e-08 over the same sweep.
`dtype=None` and `dtype=complex128` are `np.array_equal` at every rung, for
every helper.

**FLAT at 4.2e-08, one float32 rounding (analytic ceiling 8.43e-08).
CONFIRMED.**  The control's growth is confirmed too -- but note it only
separates above ~2e+03 rad, and the shipped test's two fixtures sit at 22 and
820 rad (defect **D4**).

`_phasor_rows`' band size is inert for values (`V16`): all four helpers give
the same hash at 16, 61 and 1024 rows per band, and each equals a whole-grid
`np.exp(float64 argument).astype(complex64)` exactly.

### 6b. Dtype at every hand-off, and the residual upcast -- `V6`, `V7`

Every one of the six helpers plus `_fourier_upsample_crop` and
`carrier_referenced_exact_focus_readout` wrapped, on a REAL two-group traced
carrier chain:

| build | input | returned field | full-grid complex128 phasors built |
|---|---|---|---|
| v5.43.0 | complex128 | complex128 | 10 |
| v5.43.0 | **complex64** | **complex128** | **10** (the audit's finding, reproduced) |
| 50824e9 | complex128 | complex128 | 10 |
| 50824e9 | **complex64** | **complex64** | **5** |

The five survivors are all the same call site:
`_build_carrier_phase` -> `_radial_carrier_phase` at
`lumenairy/propagators/carrier.py:1663`, which takes no `dtype=` and is reached
from `carrier_referenced_envelope` / `carrier_referenced_reconstruct` (both of
which narrow the finished phasor with `.astype(E.dtype, copy=False)`, so the
VALUES are right and only the transient is wide).  Each is a full grid:
16 MiB at N=1024, **1.07 GB at N=8192, 4.29 GB at N=16384**.

Whole-call `tracemalloc` peak of the same two-group chain, N=1024, dx=13 um:

| build | complex128 | complex64 | saving |
|---|---|---|---|
| v5.43.0 | 376.9 MiB | 376.9 MiB | **0.0 MiB (0.0 %)** |
| 50824e9 | 376.9 MiB | 332.7 MiB | **44.1 MiB (11.7 %)** |

So the audit's row 12 moved from "0.0 GB" to a real but partial saving.
Defect **D2**.

### 6c. The energy honesty bar, derived and applied -- `V6`

The campaign bar (`ADJUDICATION_NFC_8192_2026_08_10`) is **4e-05 RELATIVE**
on chain energy readouts -- throughput, capture, per-frame power,
`power_exit`.  Power is quadratic in the field, so a field relative-L2 of `e`
moves a power ratio by at most ~`2e`; the field-side bar is therefore
**2e-05**.

Measured on a REAL two-group traced carrier chain (N-BK7 biconvex + N-SF11
plano, 12 mm gap, `r_in=55 mm`, N=512, dx=26 um), complex64 envelope against
the complex128 arm:

* returned dtype complex64 (c128 arm: complex128);
* **relative L2 of the field 1.158e-07** -- 173x under the 2e-05 field bar;
* **relative total-power change 7.69e-09** -- 5200x under the 4e-05 bar.

Synthetic six-leg chain (de-chirp, band-limited resample round trip, re-chirp,
tilt ramp, x6): relative L2 grows **linearly**, 6.50e-08, 1.09e-07, 1.51e-07,
1.92e-07, 2.33e-07, **2.75e-07** -- **4.58e-08 per leg**.  The CHANGELOG's
1.9e-07/leg and 1.1e-06-after-six are a different synthetic chain (the
prototypes'); linearity and the decade of headroom are confirmed, the
coefficient is fixture-dependent and mine is 4x smaller.  Either way the
mixed-precision chain sits 2-4 decades under the honesty bar.

### 6d. `_fourier_upsample_crop` -- `V6`, `V17`

The test docstring and the CHANGELOG say a complex64 input is "transformed in
complex128 and NARROWED back on return".  On the shipped numpy that is false:

```
np.fft.fft2(np.ones((8,8), dtype=np.complex64)).dtype  ->  complex64
```

numpy >= 2.0 has a single-precision FFT.  So for a complex64 envelope BOTH
transforms of the pair now run in single precision (and the complex64 `pad` is
a no-op narrowing, since `F` is already complex64).  Measured against a
complex128 transform of the SAME complex64 samples:

| crop | out dtype | max abs diff vs single-rounding reference | eps32 x peak | rel L2 vs the complex128-input result |
|---|---|---|---|---|
| 256 -> 512 (upsample) | complex64 | 3.577e-07 | 1.364e-07 | 1.745e-07 |
| 512 -> 256 (downsample) | complex64 | 2.541e-07 | 1.288e-07 | 1.850e-07 |
| 512 -> 512 (no transform) | complex64 | 0 | -- | 2.394e-08 |

**2.6x and 2.0x eps32 x peak** -- a narrow-on-return would be <= 0.5x.  The
extra is the single-precision transform (a size-512 FFT accumulates
~sqrt(log2 N) x eps32).  Harmless at these sizes (2 decades under the field
bar) but the recorded rationale is wrong, and the precision boundary the
change advertises -- "float64 CONSTRUCTION of every reference-phase ARGUMENT"
-- does not extend to the transform pair.  Defect **D3**.

### 6e. The exact readout's dtype

complex64 in -> **complex64 out** (v5.43.0: complex128 out).  Relative L2
against the complex128 arm on the shipped fixture: **9.90e-08** (Windows),
**9.88e-08** (WSL).

---

## 7. Fail-before (task 5)

Both fail-befores are measured on the v5.43.0 worktree, not simulated by
reverting files.

**Change 1.**  A banded traced call at the shipped default on v5.43.0:

| fixture | v5.43.0 banded | rel from the whole-grid evaluator |
|---|---|---|
| S10 carrier | `engaged=False`, `gate_open=False` | **2.1886e-02** |
| S10 `carrier=None` | `engaged=False` | **2.4216e-01** |
| my screen fixture, N=384 rows=64 | `engaged=False` | 1.3541e-01 |
| my screen + spherical carrier | `engaged=False` | 4.3243e-02 |
| **N=4096, `sag_chunk_rows=None` (the AUTO production default)** | `engaged=False` | **1.1909e-01** |
| any ray-density fixture | banding silently OFF (whole-grid path; `eval_into` called once with 4 channels, peak identical to whole-grid) | 0 (same path) |

**Change 2.**  On v5.43.0 the same two-group chain fed a complex64 envelope
returns a **complex128** field, builds **10** full-grid complex128 phasors,
and saves **0.0 MiB** against the complex128 run -- the audit's finding
("the dtype survived exactly one leg", "saved 0.0 GB"), reproduced exactly.

---

## 8. Durability audit (task 6)

Every constant the two new test files assert, re-measured on the SHIPPED
fixtures, on both builds (`V9`).

| test / constant | asserted | measured (Win) | measured (WSL) | margin | durable? |
|---|---|---|---|---|---|
| `_C64_PHASOR_TOL = 2.5e-7`, R=45.9 mm | `err <= 2.5e-7` | 4.203e-08 | 4.203e-08 | 5.9x | **yes** -- a float32 rounding, build-invariant, and the analytic ceiling 8.43e-08 is stated |
| `_C64_PHASOR_TOL`, R=5 mm | same | 4.212e-08 | 4.212e-08 | 5.9x | **yes** |
| control ratio bar 10x, R=45.9 mm | `e_naive >= 10 e_good` | **23.5x** | 23.5x | **2.3x** | **thin** -- docstring claims 360x; see D4 |
| control ratio bar 100x, R=5 mm | `e_naive >= 100 e_good` | **727x** | 727x | 7.3x | ok -- docstring claims 11 600x |
| `upsample_crop` rel L2 bar 1e-5 | `<= 1e-5` | 1.715e-07 | 1.715e-07 | 58x | yes; docstring records 2.0e-07 "at N=2048" but the test runs N=512 |
| exact readout rel L2 bar 1e-4 | `<= 1e-4` | 9.90e-08 | 9.88e-08 | **1010x** | **too loose**; docstring records no number at all |
| one-group chain rel L2 bar 1e-4 | `<= 1e-4` | 9.443e-08 | 9.443e-08 | **1059x** | **too loose** |
| memory bar `>= 6` grids | `p_whole - p_band >= 6 grids` | **12.41** (22.63 -> 10.21) | **12.29** (22.50 -> 10.21) | 2.1x / 2.0x | **yes** -- the docstring's 22.6 / 10.2 reproduce on both builds |
| `np.array_equal` across band heights | same-build two-arm | -- | -- | -- | **yes** -- a two-arm claim on one build; exactly the right bar |
| `not np.array_equal` (model vs incumbent) | same-build two-arm | rel 2.25e-03 on this file's fixture | -- | -- | **yes** |
| band heights 32 / 128 / 7 | fixture parameters | -- | -- | -- | yes |

Cross-build test run of the two new files on the SECOND build:

```
wsl.exe -e bash -lc "cd /mnt/c/tmp/lum_vlens && OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 ~/lumvenv/bin/python -m pytest \
  tests/unit/test_banded_ray_density_and_inverse_map.py \
  tests/unit/test_mixed_precision_carrier_helpers.py -q"
-> 19 passed in 41.54s        (py 3.12.3, numpy 2.4.6)
```

Touched files:

* `test_lens_chunked_sag.py` -- docstring only; its pins keep
  `inverse_map=False` and remain a same-build two-arm byte-identity claim.
  Durable.
* `test_niche_s10_sibling_patterns.py` -- the gate assertion flipped from
  "banded refuses" to "both engage" and gained `np.array_equal(w_d, b_d)`.
  Structural, same-build, two-arm.  Durable, and it is the right place for it.
* `test_niche_perf_round2_2026_08_10.py::test_upsample_crop_keeps_the_envelope_dtype`
  -- contract inverted deliberately (complex128-for-everything -> envelope
  dtype).  A dtype assertion, build-free.  Durable; its docstring carries D3.

---

## 9. Test suites and lint (task 7)

`ruff check lumenairy/ tests/` -> **All checks passed!**  (`validation/` is
`extend-exclude`d by `pyproject.toml`, as it is for every other probe
directory in the repo.)

Full run of the four lens test files plus every test file naming
`_fourier_upsample_crop` / `_radial_carrier_phase` / `_tilt_ramp` /
`ray_density`, plus `test_niche_d15_deterministic_traced_fit.py`,
`test_niche_d14_deterministic_carrier_fit.py`,
`test_fix_runner_oom_2026_08_13.py` and `test_niche_d7_decentred_fit.py`
(38 files):

```
python -m pytest <38 files> -q -p no:randomly
-> 840 passed, 148 warnings in 2273.50s (0:37:53)
```

Zero failures.  The 148 warnings are the pre-existing aperture:beam and
evaluator-refusal notices these fixtures have always raised; none is new.
The two new files also pass on the SECOND build (19 passed, WSL, py 3.12.3 /
numpy 2.4.6).

No test was added by this verification, so `.test_durations` needs no splice.

---

## 10. Defects

| id | severity | where | what | reproducer |
|---|---|---|---|---|
| **D1** | cosmetic | `lumenairy/elements/_lens_traced.py`, the three `warn(...)` calls inside `_ray_density_self_checks` | the block moved into a nested closure but kept `stacklevel=2`, so it is now one frame short: on the WHOLE-GRID path too, the energy / halo / retained-band warnings are attributed to `_lens_traced.py` instead of the caller (v5.43.0 attributed all three to the caller).  Beyond the reported location, the default warning filter's per-location dedup registry moves into the library module, so a second call from a different caller module no longer re-warns.  Fix: `stacklevel=2` -> `3`. | `V11` -- v5.43.0 prints `v11_side_effects.py:52` for all three; 50824e9 prints `_lens_traced.py:12236` (banded) / `:12350` (whole-grid) for two of them.  The change DID bump the other two closures correctly: `_warn_ray_density_fold` and `_origin_amp_support_verdict` both carry `stacklevel=3` and are attributed to the caller on both builds and on both paths (`V19`, `V11`) -- so this is an omission in one block, not a systematic slip |
| **D2** | cosmetic by the correctness taxonomy (values are right), PRODUCTION-RELEVANT by memory; the entry's claim is overstated and row 12 is not fully closed | `lumenairy/propagators/carrier.py:1663` (`_build_carrier_phase`) | the seventh reference-phase construction was not given `dtype=`, so a complex64 chain still materialises one full-grid complex128 phasor per `carrier_referenced_envelope` / `carrier_referenced_reconstruct` call -- 5 per two-group chain, 16 MiB each at N=1024, **4.29 GB each at N=16384**.  Values are correct (narrowed after `exp`); the memory prescription is half-implemented, and complex64 saves 11.7 %, not the "THROUGH the chain" the entry implies.  Fix: thread `dtype=` through `_build_carrier_phase` / `_axis_carrier_phase` from the two public helpers' input dtype. | `V7` -- caller frame logged for every surviving complex128 phasor |
| **D3** | cosmetic (recorded rationale wrong; small real precision change) | `tests/unit/test_niche_perf_round2_2026_08_10.py::test_upsample_crop_keeps_the_envelope_dtype` docstring, and the CHANGELOG entry | "numpy's FFT is still double-only, so a complex64 input is transformed in complex128 and NARROWED back on return" is false on numpy >= 2.0: both transforms of the pair now run in SINGLE precision.  Measured 2.6x eps32 x peak against a complex128 transform of the same samples (a narrow-on-return is <= 0.5x); rel L2 1.7e-07 at N=512, and it grows with N. | `V6`, `V17` |
| **D4** | cosmetic (durability) | `tests/unit/test_mixed_precision_carrier_helpers.py` docstrings + the CHANGELOG | four recorded measurements do not reproduce: the two fixtures' phase arguments are **22.34** and **820.2** rad, not "3.6e+02" and "1.3e+04" (16x); the control errors are **9.87e-07** and **3.05e-05**, not "1.5e-05" and "4.9e-04" (15x); the claimed ratios "360x and 11 600x" are actually **23.5x and 727x**.  Consequence: the 10x bar has a **2.3x** margin, not the 36x the docstring implies.  Identical on both builds, so this is a wrong record, not build drift.  The PHYSICS the test asserts is sound -- an independent ladder over a genuine 2e+02 .. 2e+05 rad confirms flatness and the control's growth. | `V9` (Win + WSL), `V5` |
| **D5** | cosmetic (durability) | same file, `test_exact_focus_readout_keeps_complex64` and `test_one_group_chain_keeps_complex64_end_to_end` | bars of 1e-4 against measurements of 9.9e-08 / 9.44e-08 -- **three decades** of slack, so a 1000x precision regression passes; and the readout test's docstring records no number ("recorded in the assertion message on first run"), which is not a derivation.  Both reproduce to 3 figures on the second build, so a bar of ~1e-6 would be safe and two-sided. | `V9` (Win + WSL) |
| **D6** | cosmetic by the correctness taxonomy, PRODUCTION-RELEVANT by runtime | CHANGELOG, change 1, "Measured" bullet | "~10 % higher on the evaluator route (the 7/4 evaluation)" is right for the SCREEN branch (measured 1.077 at N=4096) but the 7/4 lives on the RAY-DENSITY branch, where it reads **1.324 (N=2048) to 1.385 (N=4096)**.  Worse, the bullet does not price the banded SCREEN call at the shipped default at all -- the one AUTO banding gives every N >= 4096 traced call -- which went **10.529 s -> 21.890 s, +108 %** across the release (whole-grid controls agree to 0.6 %), buying the evaluator's answer instead of the incumbent's.  The whole-grid ray-density path is unchanged in time (+3.4 %, inside noise, `V20`). | `V11`, `V15`, `V20` |

| **D7** | cosmetic (pre-existing, blast radius grew) | `_lens_traced.py`, the niche-C15 private `_imap_out['probe_rc']` block | the probe block sits AFTER the row-band assembly's `return`, so a banded call never fills `probe_opl` / `probe_ard`.  That was inert before v5.44 (a banded call was always the incumbent, and the probe exists to compare inversions); now a banded call at the shipped default IS the evaluator, so a C15-style comparison run at N >= 4096 with AUTO banding gets silence instead of the model's OPL.  Private, opt-in, diagnostic-only.  Measured: `rows=0` -> `probe_opl` present; `rows=32` -> absent, both with `engaged=True` and the same `n_out_of_domain`. | `V18` |

Nothing found is **silent-wrong**.  D2 is the only one with a production
consequence at scale (memory at N >= 8192 on a complex64 chain); D6 is the
only one with a production consequence in time.

Two rewrites that COULD have moved bits and do not, checked directly rather
than trusted:

* the shared `_ray_density_self_checks` closure rewrote the energy check's
  aperture term from `X ** 2 + Y ** 2` to `x[None, :] ** 2 + _y_ax[:, None] ** 2`
  (the band path has no `X`).  `X` / `Y` are `np.broadcast_to` views of exactly
  those axes, so the two forms are bitwise equal -- verified directly
  (`max |a - b| = 0.0` on a decentred 384-square grid), and the 28 cross-build
  hashes of `V1` cover it end to end.
* `_phasor_rows`' band size (`_PHASOR_BAND_BYTES = 32e6`, an untested magic
  constant) is inert for values: all four helpers hash identically at 16, 61
  and 1024 rows per band, and each equals a whole-grid
  `np.exp(float64 argument).astype(complex64)` exactly (`V16`).

---

## 11. Recommendation

**A 5.44.1 is NOT required for correctness.**  Both changes do what their
central claims say, and the claims that carry the risk -- byte-identity of the
non-banded and whole-grid paths, byte-identity of the banded field and its
diagnostics at the same inversion, the D14/D15 determinism contract, the
complex128 carrier chain -- are confirmed bit for bit against v5.43.0, on
every fixture, on two builds.  The route change the entry names is exactly the
evaluator's answer and reproduces the recorded 2.19e-02 to five figures.

**A 5.44.1 is RECOMMENDED as a small, low-risk follow-up**, in this order:

1. **D2** -- thread `dtype=` through `_build_carrier_phase` (carrier.py:1663)
   so `carrier_referenced_envelope` / `carrier_referenced_reconstruct` stop
   building a full-grid complex128 phasor on a complex64 chain.  This is the
   one item with a GB-scale production consequence and it is the residual of
   the very leak row 12 declared closed.  Pin it with the caller-frame probe
   in `V7` (a test that asserts zero complex128 phasor returns on a complex64
   chain).
2. **D1** -- `stacklevel=2` -> `3` on the three self-check warnings.  One
   character each; restores v5.43.0's attribution and the per-caller dedup.
3. **D3 / D4 / D5** -- correct the recorded numbers in the two test docstrings
   and in the CHANGELOG, and tighten the two 1e-4 bars to ~1e-6 (both
   measurements reproduce to three figures across builds, so the tightened bar
   is two-sided).
4. **D6** -- restate the wall-time bullet: separate the SCREEN branch (where
   ~10 % is right) from the RAY-DENSITY branch that the 7/4 evaluation belongs
   to (1.32x-1.39x), and price the banded SCREEN route explicitly (+108 %
   across the release), since AUTO banding at N >= 4096 makes it the
   production default for every traced call the runner drives.

## 12. What could not be verified

* the CHANGELOG's prototype figures (`../Lumenairy_prototypes/`: 2496 + 336
  byte-identical runs; 24 -> 1.2 and 30 -> 3.7 grids of assembly transient at
  N=4096; the six-leg 1.9e-07/leg) -- the prototypes are not in this
  repository.  Independent equivalents are measured above.
* the audit's "one evaluation at 1.9 s on the 8192-square design-121 last
  group against a 96.9 s element" and the ~64 GB -> ~8 GB extrapolation to the
  16384 fine leg -- both need the design-121 `.zmx`, which this worktree does
  not carry.  The N=4096 whole-call peaks are given instead.
* accumulated error through the real design-121 chain at complex64 (the
  CHANGELOG explicitly does not claim it either).  A real two-group
  prescription at N=512 is measured instead (section 6c).
* GPU (`use_gpu=True`) and JAX arms of either path -- out of scope for both
  changes (`_imap_domain_gate` still excludes `use_gpu`, unchanged).
* whether the fold-caustic warning can fire on the evaluator route with the
  SHIPPED census thresholds: the evaluator's own G2 guard refuses a folded
  map, so on every fixture I could build, a natural fold sends the call to the
  coarse-Newton route.  The banded census was therefore exercised with the
  thresholds forced (`V2 --fold`, `--medianbite`), which drives the identical
  code path and puts the median into the field.
