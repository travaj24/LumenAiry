# VERIFY -- the seven 5.44.0 lens follow-ups (D1-D7), independently

**Date** 2026-09-11 **Branch** `verify/lens-followups` (worktree
`C:/tmp/lum_vlens2`, off `24651c8` = the integration branch `wave2/pmm2d`
with `fix/lens-5440-followups` merged at `a1954c0`)
**Subject** every claim of `docs/audits/FIX_LENS_5440_FOLLOWUPS_2026_09_11.md`
(commits `e55d0c3..dc8fcef`), which are the defects D1-D7 of
`docs/audits/VERIFY_LENS_BANDED_COMPLEX64_2026_09_10.md`.

Everything below is **re-measured**, not re-read.  The fixtures are my own --
`validation/probe_verify_lens_followups/_vf.py` carries an N-BAF10 positive
meniscus, a fast N-LAK22 biconvex and a cemented N-LAK22 / N-SF6 doublet at
1.55 um, none of which is the builder's N-SF11 singlet or the shipped tests'
N-BK7 one, plus a speckled beam so no hash can be satisfied by a smooth-field
coincidence.  The two deliberate exceptions are named where they occur: the
durability probe measures the SHIPPED TESTS' constants and must therefore use
the shipped tests' fixtures, and `run_builder_probe.py` re-runs the builder's
own probes UNCHANGED whenever the question is whether their RECORD reproduces
rather than whether the physics does.

## Arms

| arm | tree | build |
|---|---|---|
| NEW | `C:/tmp/lum_vlens2` @ `verify/lens-followups` | Windows 11, py 3.14.6, numpy 2.4.4, scipy 1.17.1 |
| v5.44.0 | `C:/tmp/lum_v5440c` @ `v5.44.0` (`9af9376`) | as NEW |
| v5.43.0 | `C:/tmp/lum_v5430v` @ `v5.43.0` (`78e4091`) -- D6 attribution only | as NEW |
| SECOND BUILD | WSL Ubuntu `~/lumvenv` on `/mnt/c/tmp/lum_vlens2` | py 3.12.3, numpy 2.4.6, scipy 1.17.1 |

Every script asserts `lumenairy.__file__` before it measures anything and
REFUSES a tree it was not pointed at; the JSON carries the same block.
`OMP/OPENBLAS/MKL_NUM_THREADS=1` throughout.  Both read-only worktrees are
removed at the end.

`wave2/pmm2d` moved on during this verification (`24651c8` -> `f2371e0`, the
`verify/sliver-round2` and `fix/pmmstack-sliver-guard-round3` merges).  Those
merges touch **no** file this verification measures -- `git diff
24651c8..f2371e0 -- lumenairy/elements/_lens_traced.py
lumenairy/propagators/carrier.py lumenairy/elements/_lens_imap.py` and the
three test files is **empty** -- so nothing here was re-run against them.

**Box load.**  Three other agents ran compute on this box throughout: 24
logical CPUs, `LoadPercentage` 62-83 % at the readings I took, 10-11 resident
python processes, 75-82 GB free of 128 GB.  That is the same condition the
builder measured under and it is stated again here for the same reason: the
D6 wall-clock seconds are not comparable to an idle box, and none of the D6
conclusions rest on them.  What they rest on are FIELD HASHES, `domain_mask`
CALL and PIXEL counts, and within-run ratios, all of which are load-free.

---

## 0. Verdict table

| id | claim | verdict | the number that decides it |
|---|---|---|---|
| **T1** | every complex128 carrier chain / readout / crop is bit-identical to v5.44.0 | **CONFIRMED** | 395 leaves compared, **0 differing** |
| **T1** | every whole-grid and banded traced call is bit-identical to v5.44.0 | **CONFIRMED** | 11 fixtures x 5 band heights = 55 calls, 2503 leaves, **0 differing** outside the free-RAM reading, the wall clock and D1's attribution |
| **D1** | the three ray-density self-check warnings name the CALLER; 6 of 9 -> 0 of 9 | **CONFIRMED** | 9 ray-density notices over 3 band heights: **6 library-attributed -> 0**; field hash `314e073156398b735b904b85` on all six arms |
| **D1'** | (the VERIFY doc's secondary rationale) the library attribution also breaks the default filter's per-location dedup | **REFUTED as observable** | 2 caller modules, `simplefilter('default')`: **10 notices on BOTH arms**.  One `apply_real_lens_traced` call moves the warnings filter version **10 -> 25**, which invalidates every `__warningregistry__` -- the dedup is defeated on this path regardless of stacklevel |
| **D2** | scalar carrier, N=2048: 224.03 -> 189.03 MiB; N=4096: 896.06 -> 640.06 | **CONFIRMED** | 224.032 -> 189.033 and 896.063 -> 640.063 MiB, byte-identical between my probe and the builder's own, on both arms |
| **D2** | astigmatic, N=2048: **128.00** -> 64.00 (saved 64.00, -50.0 %); N=4096: **512.00** -> 256.00 (saved 256.00, -50.0 %) | **REFUTED (recorded number)** | the AFTER values are right; the BEFORE values are **96.00** and **384.00** -- so the saving is **32.00 MiB (-33.3 %)** and **128.00 MiB (-33.3 %)**.  The builder's OWN `p1_before.json` says 96.00 / 384.00; the doc's table copied the complex128 column |
| **D2** | "the complex64 arm's peak was EXACTLY the complex128 arm's before the fix, **on every case**" | **REFUTED** | true on the 8 scalar cases, false on all 8 astigmatic ones (96.00 vs 128.00 at N=2048, 384.00 vs 512.00 at N=4096): complex64 already saved 25 % there |
| **D2** | "below N ~ 1414 the band IS the grid and there is no transient saving at all" | **REFUTED** | it is not zero, it is a **PENALTY**: the complex64 scalar peak rises **14.3 %** against v5.44.0 at every N in that regime (56.017 -> 64.017 MiB at N=1024), break-even between **N=1600 and N=1700**, and the complex64 call peaks ABOVE the complex128 call it is supposed to undercut |
| **D2** | full-grid complex128 phasors per two-group chain 5 -> 0 | **CONFIRMED** | my own chain: **5 -> 0** (helper returns 15 -> 20); the builder's V6 re-run agrees |
| **D2** | the complex64 chain's field is unchanged to 16 digits: rel L2 `1.157936782559319e-07`, rel power `7.688986734224147e-09` | **CONFIRMED** | identical to all 16 digits on v5.44.0, on NEW, and to the builder's record |
| **D2** | every field hash unchanged | **CONFIRMED** | 36 + 24 + 16 = **76 carrier-helper field hashes**, 0 differing |
| **D3** | the shipped single-precision pair is not worse than a forced complex128 pair narrowed once | **CONFIRMED** | my chain: A 2.4573e-07 / 1.3392e-07 vs B 2.4711e-07 / 1.3118e-07 -- B's L2 is **worse**, its power marginally better, exactly the shape they report |
| **D3** | a plain chain calls the crop 0 times | **CONFIRMED** | `final_leg='paraxial'`: **0 calls**; with a focus readout: 2 |
| **D3** | the crop's error grows as ~`sqrt(log2 N)` | **BOUNDED** | over four octaves mine grows **1.282x** against the `sqrt(log2 N)` prediction of 1.173x (theirs 1.151x).  Same order, slow, and the extrapolation holds -- but "~`sqrt(log2 N)`" is a shape, not a fit |
| **D6** | the banded SCREEN doubling IS the route change, proved by a same-build control that returns v5.43.0's bits | **CONFIRMED** | v5.43.0's banded screen answer and 5.44.0's `inverse_map=False` control are the same field on BOTH fixtures (`66b9384477cf8a9ca1844c6a` on theirs, `7cdb324a33738b60d36ff2a2` on mine), and 5.44.0's banded screen answer is the whole-grid EVALUATOR answer |
| **D6** | the banded screen call costs ~1.6x what it did on v5.43.0 | **CONFIRMED on their fixture, REFUTED as a general statement** | their fixture on my box: 17.241 / 10.803 = **1.596x**.  MY fixture, same N / sub / dx: 5.44.0 banded screen **11.817 s** vs the same-build incumbent control **12.382 s** = **0.954x** -- no penalty at all |
| **D6** | the price is the whole-grid DOMAIN TEST (~8.9 s), not the channel evaluations (~0.9 s) | **BOUNDED to their fixture** | on MY fixture `domain_mask` is **0.41 s** of a 13 s call (1.00 grids, same counts), and the evaluator and incumbent routes cost the same.  The MECHANISM is right; the 8.9 s is one geometry's number |
| **D6** | banding is free (0.97x-1.08x at the same inversion) | **CONFIRMED** | my fixture 1.044 / 1.059 (screen, v5.44.0 / NEW), theirs 0.900 / 1.036; the ray-density pair is 1.174 -> 1.100 |
| **D6 fix** | the duplicated pass-2 domain mask: 2.00 -> 1.00 grids of pixels, 32 -> 16 calls | **CONFIRMED** | N=4096 / 256-row bands: `domain_mask` **32 calls / 2.000 grids -> 16 / 1.000**, `eval_into` untouched at 32 calls / 7.000 ch per pixel |
| **D6 fix** | the removed computation was byte-identical | **CONFIRMED, directly** | every mask hashed: v5.44.0's pass-1 sequence **equals** its pass-2 sequence, and the NEW arm's single sequence equals BOTH, at 12 and at 55 bands |
| **D6 fix** | ray-density banded/whole 1.285 -> 1.110 at N=4096 | **CONFIRMED (direction and size)** | their fixture on my box **1.410 -> 1.058**; mine **1.174 -> 1.100**.  The seconds are load-bound; the counts are not |
| **D6 fix** | the field is bit-identical before and after | **CONFIRMED** | `08f9aa00b567ed87658cf8ca` (their fixture) and `769e48c6f8d010a6daa7f977` (mine) on both arms, plus 55 more calls in T1 |
| **D7** | the C15 probe is filled on the band path and equals the whole-grid arm | **CONFIRMED, and wider** | 5 routes x 3 band heights = **15 banded calls**: absent on all 15 at v5.44.0, present and bit-equal on all 15 after -- `probe_opl`, `probe_ard`, `probe_opl_piston` and the **NaN pattern** (82 of my 112 probe pixels are out of domain), with the field unmoved by asking |
| **D4** | 22.34 / 820.19 rad, controls 9.8719e-07 / 3.0523e-05, ratios 23.5x / 727x | **CONFIRMED, both builds** | 22.3363 / 820.1891 rad; 9.87193e-07 / 3.05233e-05; 23.486x / 727.038x -- identical on Windows and WSL to every figure printed |
| **D4** | `_C64_PHASOR_TOL` sits 3.95x below the control | **CONFIRMED** | control / bar = **3.949x**; the bar is 5.94x above the measurement |
| **D5** | readout 9.8995e-08 (Win) / 9.8841e-08 (WSL), chain 9.4427e-08, margins 10.1x / 10.6x | **CONFIRMED, both builds** | 9.899520e-08 / 9.884099e-08 (spread 0.156 %) and 9.442663e-08 on both; margins **10.101x** and **10.590x** |
| **D5** | the two 1e-6 bars are two-sided | **CONFIRMED, with a caveat** | no bar in these files is within a decade of a build-moving quantity; the tight ones are tight against a DESIGN quantity (the float32-argument control), not against build drift |
| **task 6** | the band-memory bar's "above anything a build can move" | **CONFIRMED, now measured** | 12.414612 grids (Windows) / 12.290180 (WSL) = **1.00 % spread** against a 2.07x margin, and 10.78 grids on a second fixture |

Two REFUTED rows are recorded numbers in the audit document, not defects in
the shipped code (D2's astigmatic "before" column and its "on every case"
sentence).  One REFUTED row is a real, measured behaviour change the document
mis-states in the safe direction (D2 below the band/grid crossover).  One is a
generalisation that does not hold outside the fixture it was measured on (D6's
1.6x and its 8.9 s).  Nothing in this branch returns a wrong answer, and no
bit that was supposed to stay still moved.

---

## 1. Task 1 -- bit identity against the released 5.44.0

### 1a. The complex128 side (`q1_c128_chain_identity.py`)

Two- and three-group traced carrier chains on the meniscus + doublet, both
carrier references (`sphere`, `parabola`), both final legs, a speckled input;
the two public carrier helpers at a scalar, an astigmatic and BOTH
single-finite-axis carriers; two exact focus readouts; and all three crop
branches (`n_crop < n_fine`, `n_crop > n_fine`, `n_crop == n_fine`) at both
dtypes.

| what | cases | result |
|---|---|---|
| chains (field hash, `dx`, `R`, every stage record) | 6 | identical |
| carrier helpers (envelope, reconstruct, and the factor itself) | 4 x 3 | identical |
| exact focus readouts | 2 | identical |
| crops | 12 | identical |
| **leaves compared** | **395** | **0 differing** |

Chain hashes, for the record: `d4a386bf5624d1d27aa87411` (2-group sphere),
`27fafbf4ca6d4b87a223c70f` (parabola), `fd112f5f04c8ed7040427b1f` (speckled),
`fd9cc102eca0e416c06b5f59` (3-group), `9b79353de1d3551a27c281bc` /
`6ec6b78649a67ea03e7fcdcd` (exact readout, 2 and 3 groups).

This is the pin that matters for D2: `_build_carrier_phase` now takes
`dtype=`, and on `None` / `complex128` it must be the shipped whole-grid
`np.exp`.  It is.

### 1b. The traced side (`q2_traced_identity.py`)

11 fixtures -- screen / ray-density, preserve / remap x {lattice, full}, a
decentred `origin`, a caustic-bearing fold, a tilt-aware `sub=1` call, both
inversion routes -- at band heights **{None, 0, 7, 32, 128}**, i.e. 55 traced
calls per arm, each hashed together with `sum|E|^2`, `max|E|`, the count of
non-finite pixels, the whole `_imap_out` record and the full warning list.

```
leaves compared: 2503   differing: 0   allowed: 160
```

The 160 allowed leaves are three classes and only three:

* `rec.build_budget_gb` -- the evaluator records the FREE RAM it saw
  (43.4-44.6 GB across the runs).  An environment reading, not a value.
* `rec.build_seconds` -- wall clock.
* `w_attr` -- warning `filename:lineno`.  Every message text is identical;
  what moves is (i) D1's intended attribution change and (ii) a pure line
  shift, `_lens_traced.py:8888 -> :8901` and `:8917 -> :8930`, of the two
  notices the NESTED `apply_real_lens` call raises, caused by the 13-line
  docstring the branch adds at line 7524.  Same file, same function, same
  frame; the line moved because the file did.

Field hashes are equal across the two arms **and** across all five band
heights within each arm -- which re-confirms the original 5.44.0 banding
claim on 11 fixtures the 5.44.0 verification did not use.

*Observation, not a defect and not this branch's:* those two
`apply_real_lens:` notices are attributed to `_lens_traced.py` on both arms.
That is correct in its own terms -- `apply_real_lens`'s caller genuinely is
`apply_real_lens_traced` -- but a user who passes an oversized aperture sees
the same sentence twice, once against their own line and once against the
library's.  Pre-existing, cosmetic, out of D1's scope.

---

## 2. D1 -- the self-check warnings

`q3_d1_warn_attr.py`, on a decentred beam on the meniscus with all five
ray-density thresholds driven over, at band heights 0 / 32 / 7.

| arm | notices per call | from the caller | from the library |
|---|---|---|---|
| v5.44.0 | 5 | 2 | **3** (the nested `apply_real_lens` aperture notice + the ENERGY and SUPPORT-BAND self-checks) |
| NEW | 5 | 4 | 1 (the nested `apply_real_lens` notice only) |

Counting only the notices `apply_real_lens_traced` itself raises -- the origin
verdict, the energy self-check and the support-band self-check, 3 per band
height, 9 in total, which is exactly the population the claim is about:

**9 -> 15 total notices, of which library-attributed 6 of 9 -> 0 of 9.**
CONFIRMED, on my fixture, at the counts claimed.

Per-arm detail, v5.44.0: `_lens_traced.py:12350` (whole-grid) and `:12236`
(banded) for the energy and support-band notices; NEW:
`q3_d1_warn_attr.py:67` for all three.  The field hash is
`314e073156398b735b904b85` on all six arms -- a `stacklevel` cannot move a
value, and it did not.

The halo self-check did not fire on my fixture either, which independently
reproduces the builder's open item 4: two of the three checks in that closure
are pinned only by argument (they share one frame depth) and by
`p2_d1_attr.py`, not by a test.

### 2a. The dedup half of the defect does not reproduce

`VERIFY_LENS_BANDED_COMPLEX64_2026_09_10` D1 also says: *"the default warning
filter's per-location dedup registry moves into the library module, so a
second call from a different caller module no longer re-warns"*, and the
`_ray_density_self_checks` docstring the branch adds repeats it.  Measured, by
`exec`-ing the same caller source under two different filenames and calling
with `simplefilter('default')`:

| arm | notices after caller 1 | after caller 2 |
|---|---|---|
| v5.44.0 | 5 | **10** |
| NEW | 5 | **10** |

No suppression on EITHER arm.  The reason is measurable: a single
`apply_real_lens_traced` call moves `warnings`' internal filter version from
**10 to 25** (15 mutations), and every `__warningregistry__` is invalidated
whenever that version changes.  So on this call path the per-location dedup is
already defeated by the library's own filter handling, independently of
`stacklevel`.

**Verdict: the printed-location half of D1 is CONFIRMED; the dedup half is
REFUTED as observable through the public entry point.**  It costs nothing --
the fix is right for the `filterwarnings(module=...)` reason alone -- but the
sentence should not be left in a docstring as a measured consequence.

---

## 3. D7 -- the C15 probe on the band path

`q4_d7_probe_rc.py`, 112 probe pixels chosen to straddle every band boundary
(rows 0, 1, 6, 7, 8, 31, 32, 33, 63, 64, 127, 128, 129, N/2, N-2, N-1) and
both grid edges, on five routes.

| route | v5.44.0, banded | NEW, banded |
|---|---|---|
| screen + evaluator | `probe_opl` **absent** at rows 7 / 32 / 128 | present, `array_equal` to the whole-grid arm |
| ray-density + evaluator | absent | present, and `probe_ard` equal too |
| screen + coarse-Newton | absent | present |
| ray-density + coarse-Newton | absent | present, `probe_ard` equal |
| ray-density + remap + evaluator | absent | present, `probe_ard` equal |

15 banded calls; 15 absent before, 15 equal after, with the **NaN pattern**
identical (82 of the 112 pixels are out of domain on the evaluator routes,
which is what makes the gather a real test rather than an interpolation of a
smooth map).  `probe_opl_piston` agrees.  The whole-grid arm's 45 recorded
leaves are identical between the two arms -- asking on the whole-grid path is
unchanged -- and every field hash is identical with and without `probe_rc`,
on both arms: **asking costs no bits.**

This is strictly wider than the builder's V18 (one route, one band height) and
than the shipped test (three routes, all-finite pixels).

---

## 4. D2 -- the seventh reference phase

### 4a. The per-call transient (`q5_d2_memory.py`, my fixture, both arms)

Whole-call `tracemalloc` peak of ONE public helper call, warm.  `tracemalloc`
counts REQUESTED sizes, so these are fixed by shapes and dtypes; envelope and
reconstruct agree to the byte in every row, so only one is shown.

| N | carrier | complex128 (both arms) | complex64 **v5.44.0** | complex64 **NEW** | change |
|---|---|---|---|---|---|
| 1024 | scalar | 56.02 | 56.02 | **64.02** | **+8.00 MiB (+14.3 %)** |
| 1024 | astigmatic | 32.00 | 24.00 | 24.28 | +0.28 MiB |
| 1024 | single-axis | 16.14 | 8.07 | 8.07 | 0 |
| 2048 | scalar | 224.03 | 224.03 | **189.03** | -35.00 MiB (-15.6 %) |
| 2048 | astigmatic | 128.00 | **96.00** | **64.00** | -32.00 MiB (-33.3 %) |
| 2048 | single-axis | 64.16 | 32.08 | 32.08 | 0 |
| 4096 | scalar | 896.06 | 896.06 | **640.06** | -256.00 MiB (-28.6 %) |
| 4096 | astigmatic | 512.00 | **384.00** | **256.00** | -128.00 MiB (-33.3 %) |
| 4096 | single-axis | 256.19 | 128.09 | 128.09 | 0 |

The two scalar rows the FIX doc leads with reproduce exactly:
**224.03 -> 189.03** and **896.06 -> 640.06**.

The astigmatic rows do not.  The FIX doc records the complex64 BEFORE column
as 128.00 and 512.00 -- the complex128 values -- and prices the saving at
64.00 MiB / -50.0 % and 256.00 MiB / -50.0 %.  Re-measured, the BEFORE is
**96.00** and **384.00** and the saving is **32.00 MiB / -33.3 %** and
**128.00 MiB / -33.3 %**.

This is not a fixture difference.  The **builder's own probe**, run unchanged
on the v5.44.0 worktree (`run_builder_probe.py p1_d2_transient.py`), prints
`2048/astig/envelope/c64 96.00 MiB = 3.00 c64-grids` and
`4096/astig/envelope/c64 384.00 MiB = 3.00`, and their recorded
`results/p1_before.json` says the same.  **The probe data is right; the table
in the document is wrong**, in four of its eight "before" cells and four of
its eight "saved" cells, and so is the sentence *"the complex64 arm's peak was
EXACTLY the complex128 arm's before the fix, on every case"*.

The mechanism is plain once measured: on the shipped astigmatic path
`phase = phase.astype(E.dtype, copy=False)` drops the complex128 product as
soon as the narrowed copy exists, so the peak was already 3 grids and not 4.

### 4b. The band/grid crossover -- the claim that is wrong in the safe direction

The FIX doc says *"Below N ~ 1414 the band IS the grid and there is no
transient saving at all (only the narrower output)"*.  Measured straight
through the crossover (`q5b_band_crossover.py`, complex64 peak, MiB):

| N | bands | scalar v5.44.0 | scalar NEW | change | astig v5.44.0 | astig NEW |
|---|---|---|---|---|---|---|
| 512 | 1 | 14.009 | **16.009** | **+14.3 %** | 6.000 | 6.268 |
| 1024 | 1 | 56.017 | **64.017** | **+14.3 %** | 24.000 | 24.283 |
| 1414 | 1 | 106.802 | **122.056** | **+14.3 %** | 45.763 | 46.023 |
| 1415 | 2 | 106.953 | **122.143** | **+14.2 %** | 45.827 | 46.045 |
| 1448 | 2 | 111.999 | **125.035** | **+11.6 %** | 47.990 | 46.776 |
| 1600 | 2 | 136.744 | 139.186 | +1.8 % | 58.594 | 50.344 |
| 1700 | 2 | 154.370 | **149.234** | **-3.3 %** | 66.147 | 52.816 |
| 1800 | 2 | 173.063 | 159.935 | -7.6 % | 74.158 | 55.510 |
| 2048 | 3 | 224.032 | **189.033** | **-15.6 %** | 96.000 | 64.000 |

So on the scalar (radial) branch the change is a **14.3 % INCREASE** in
per-call peak for every N at or below the crossover, and the break-even is
**between N=1600 and N=1700**, not at 1414.  At N=1024 the complex64 call now
peaks 14 % ABOVE the complex128 call it exists to undercut (64.02 vs 56.02
MiB).  The mechanism is that a single-band `_phasor_rows` holds the complex64
output AND the whole-grid complex128 `exp` transient simultaneously, where the
shipped whole-grid build released the complex128 array as it narrowed.

**It is inherited, not invented** (`q5c_phasor_rows_penalty.py`).  The four
helpers that already took `dtype=` at 5.44.0 read the same complex64 /
complex128 peak ratios at N=1024 on **both** versions, to the byte:

| helper | N=1024 c64/c128, v5.44.0 | N=1024, NEW | N=2048, both |
|---|---|---|---|
| `_radial_carrier_phase` | 1.143 | 1.143 | 0.844 |
| `_tilt_ramp` | 1.250 | 1.250 | 0.727 |
| `_tilt_exactness_phase` | 1.091 | 1.091 | 0.853 |
| `_sphere_parab_conversion` | 1.166 | 1.166 | 1.000 |

What D2 does is extend that pre-existing sub-crossover penalty to the two
PUBLIC carrier helpers, which is a new cost on the public call path.  It is
small in absolute terms (8 MiB at N=1024, 15 MiB at N=1414), it is dwarfed by
the GB-scale win the fix is for, and the shipped test pins N=2048, in the win
regime.  But "no saving at all" understates it, and the shipped bar is not
scale-free.  Recorded in the test docstring on this branch.

### 4bb. The second build agrees where the test pins, and not everywhere else

`q5_d2_memory.py` on WSL (py 3.12.3 / numpy 2.4.6):

* **N=2048 scalar: 224.03 -> 189.03 MiB, identical to Windows to the byte** --
  the shipped bar's regime, and the FIX doc's "the same figures to the byte,
  because tracemalloc counts REQUESTED sizes" holds there;
* N=1024 scalar: 56.02 (complex128) vs **64.02** (complex64) -- the
  sub-crossover penalty of 4b is build-independent;
* N=4096 scalar: **573.06 MiB on WSL against 640.06 on Windows** (4.477 vs
  5.000 complex64 grids).  So "identical to the byte on both builds" is true
  at N=2048 and false at N=4096; the shipped test pins N=2048.

**And one finding that bounds every complex128 hash in this campaign,
including the builder's.**  The complex128 phasor is NOT bit-portable across
these two numpy versions, while the complex64 one is.  Isolated directly:

| quantity | Windows, numpy 2.4.4 | WSL, numpy 2.4.6 |
|---|---|---|
| the float64 argument `k r^2 / 2R` | `fcd684998f5de9cf2f5422cf` | **same** |
| `np.exp(1j * arg)` -> complex128 | `a66bf6d5a3caa8dc6beacdf0` | **`14cf483a5b3e81c6117241ee`** |
| `np.cos(arg) + 1j*np.sin(arg)` | `a66bf6d5a3caa8dc6beacdf0` | `14cf483a5b3e81c6117241ee` |
| the same narrowed to complex64 | `30d1a53f57fbc49eb83c5a42` | **same** |

Two consequences.  First, every bit-identity claim in this document and in the
two it verifies is a WITHIN-BUILD claim -- which is what they were measured as,
and the arms compared here are always same-build.  Second, this is direct
evidence FOR the boundary D2 implements: a float64 argument narrowed after
`exp` is portable at complex64 precision, where the complex128 phasor it
replaces is not.

### 4c. Values: nothing moved

* 36 transient cases (3 N x 3 carriers x 2 helpers x 2 dtypes) -- **36 field
  hashes, 0 differing** between the arms;
* the crossover sweep and the helper sweep add **24 + 16 = 40** more, 0
  differing;
* the two-group chain census: complex128 arm **10 helper returns, 10 full-grid
  complex128, field hash equal**; complex64 arm **15 returns / 5 full-grid
  complex128 -> 20 returns / 0**, field hash equal.  **5 -> 0, CONFIRMED**;
* the builder's `v6_c64_chain.py`, re-run unchanged on both arms: rel L2
  **1.157936782559319e-07** and rel power **7.688986734224147e-09**, identical
  to all 16 digits on v5.44.0, on NEW, and to their recorded JSON.  The only
  JSON leaves that differ between the arms are the upcast call log -- 5
  `_radial_carrier_phase -> complex128` entries replaced by
  `_phasor_rows -> complex64`.

---

## 5. D6 -- the banded route's price

### 5a. The attribution, on the builder's fixture (`p3_d6_time.py`, unchanged)

N=4096, sub=32, dx=1.5 um, AUTO band height 256, best of 2, cache cleared per
call, on a ~80 % busy box.

| arm | route | wall | `eval_into` calls (ch/px) | `map_coordinates` | field hash |
|---|---|---|---|---|---|
| 5.44.0 | screen, whole-grid | 19.161 s | 1 (3.000) | 0 | `91da5e59c46beb1d967b28b3` |
| 5.44.0 | screen, AUTO-banded | **17.241 s** | 16 (3.000) | 0 | `91da5e59c46beb1d967b28b3` |
| 5.44.0 | screen, banded, `inverse_map=False` | **10.492 s** | 0 | 32 | **`66b9384477cf8a9ca1844c6a`** |
| v5.43.0 | screen, whole-grid | 17.754 s | 1 (3.000) | 0 | `91da5e59c46beb1d967b28b3` |
| v5.43.0 | screen, AUTO-banded (its default) | **10.803 s** | 0 | 32 | **`66b9384477cf8a9ca1844c6a`** |
| v5.43.0 | screen, banded, `inverse_map=False` | 10.580 s | 0 | 32 | `66b9384477cf8a9ca1844c6a` |
| NEW | screen, AUTO-banded | 19.543 s | 16 (3.000) | 0 | `91da5e59c46beb1d967b28b3` |
| NEW | screen, banded, `inverse_map=False` | 11.051 s | 0 | 32 | `66b9384477cf8a9ca1844c6a` |
| 5.44.0 | ray-density, whole-grid | 21.912 s | 1 (4.000) | 1 | `08f9aa00b567ed87658cf8ca` |
| 5.44.0 | ray-density, AUTO-banded | **30.896 s** | 32 (7.000) | 16 | `08f9aa00b567ed87658cf8ca` |
| NEW | ray-density, whole-grid | 23.074 s | 1 (4.000) | 1 | `08f9aa00b567ed87658cf8ca` |
| NEW | ray-density, AUTO-banded | **24.409 s** | 32 (7.000) | 16 | `08f9aa00b567ed87658cf8ca` |

The hashes reproduce the builder's record exactly, including
`66b9384477cf8a9ca1844c6a` and `08f9aa00b567ed87658cf8ca`, on a different
worktree and a different day.  Read them first:

* v5.43.0's banded SCREEN answer **is** the incumbent's, and 5.44.0's
  `inverse_map=False` control returns those same bits -- the control is the
  old route, not a model of it;
* 5.44.0's and NEW's banded screen answers are the whole-grid EVALUATOR
  answer, on both versions.

So the ROUTE CHANGE is confirmed bit for bit, and on this fixture the price
is confirmed too: **17.241 / 10.803 = 1.596x** against v5.43.0's banded
default, and **17.241 / 10.492 = 1.643x** against the same-build control.

The ray-density ratio, which is what the D6 FIX moves:
**30.896 / 21.912 = 1.410 (v5.44.0) -> 24.409 / 23.074 = 1.058 (NEW)**,
with `eval_into` untouched at 32 calls and 7.000 channels per pixel.  The
claimed 1.285 -> 1.110 is confirmed in direction and size; the exact seconds
are load-bound (their unchanged whole-grid arm moved 24.747 -> 20.179 s
between their own two runs, mine 21.912 -> 23.074 between arms).

### 5b. The same measurement on MY fixture -- where the 1.6x is not there

`q7_d6_route.py`, N=4096 / sub=32 / dx=1.5 um, AUTO band height, best of 3,
on my N-BAF10 meniscus with a spherical carrier.

| arm | screen whole | screen banded | screen banded, incumbent | rd whole | rd banded | rd banded, incumbent |
|---|---|---|---|---|---|---|
| v5.43.0 | 10.931 | **12.075** (= incumbent) | 11.955 | 15.754 | 15.796 | 15.828 |
| v5.44.0 | 11.315 | **11.817** (= evaluator) | **12.382** | 14.948 | 17.545 | 17.520 |
| NEW | 12.771 | 13.523 | 13.512 | 17.185 | 18.903 | 17.492 |

The hashes behave exactly as on their fixture -- v5.43.0's banded screen
answer is `7cdb324a33738b60d36ff2a2` (the incumbent), 5.44.0's and NEW's is
`41fb3a2c9e0a807995d84733` (the evaluator), and every `inverse_map=False`
control returns the incumbent's bits -- so the ROUTE claim generalises.

The COST does not.  On this fixture 5.44.0's banded screen call at the
shipped default is **0.954x** the same-build incumbent control (11.817 /
12.382) and **0.979x** v5.43.0's banded default (11.817 / 12.075).  There is
no 1.6x, and no doubling.

### 5c. Why: the domain test is a geometry-dependent cost

The instrumented split (`q7_d6_route.py --mode stages`), N=4096, my fixture:

| arm | route | total | `eval_into` | **`domain_mask`** | `build_inverse_map` | `map_coordinates` |
|---|---|---|---|---|---|---|
| v5.44.0 | screen, whole | 13.198 s | 0.89 (3.00 ch/px) | **0.40 s, 1 call, 1.000 grids** | 0.24 | 0.00 |
| v5.44.0 | screen, banded | 12.962 s | 1.06 (3.00) | **0.41 s, 16 calls, 1.000 grids** | 0.10 | 0.00 |
| v5.44.0 | screen, banded incumbent | 13.651 s | 0.00 | 0.00, 0 calls | 0.00 | 2.31 / 32 |
| v5.44.0 | rd, whole | 17.060 s | 1.22 (4.00) | **0.41 s, 1 call, 1.000 grids** | 0.16 | 0.85 / 1 |
| v5.44.0 | rd, banded | 18.579 s | 2.18 (7.00) | **0.73 s, 32 calls, 2.000 grids** | 0.09 | 0.88 / 16 |
| NEW | rd, banded | 17.758 s | 2.19 (7.00) | **0.33 s, 16 calls, 1.000 grids** | 0.13 | 0.88 / 16 |

`domain_mask` is **0.41 s** here, not 8.9 s.  The evaluator's channel
evaluation, its model build and its domain test together are ~1.6 s against
the incumbent's 2.3 s of `map_coordinates` -- which is why the two routes cost
the same on this fixture, and why they differ by 6.7 s on the builder's.

The difference is in `InverseCharacteristic.domain_mask` itself: it takes the
cheap SCREENED path (`hull_mask_grid`, a separable axis test) only when
`self.hull is not None and axes is not None and self.hull_c`, and otherwise
falls back to a whole-grid `_TracedExitSupport.signed_distance` against the
hull polygon.  Which branch a call takes is a property of the exit-support
GEOMETRY, i.e. of the prescription and the beam -- not of N, not of banding
and not of the release.

**Verdict.**  D6's mechanism (the evaluator's domain test dominates the
evaluator route) is right where it was measured, and the attribution by hash
is airtight everywhere.  But "the banded SCREEN call at the shipped default
costs about 1.6x what it did on v5.43.0" and "that price is the evaluator's
whole-grid DOMAIN TEST (~8.9 s)" are now in a PUBLIC DOCSTRING
(`apply_real_lens_traced`, `sag_chunk_rows`) and in the CHANGELOG as
unqualified statements, and a second fixture at the same N, sub and dx reads
0.95x and 0.41 s.  They should be qualified as one geometry's measurement.

### 5d. The D6 fix, proved directly (`q8_d6_mask_identity.py`)

Every mask `domain_mask` returns is hashed, in order, on the ray-density +
evaluator route at N=384.

| arm | band height | `domain_mask` calls | pixels tested | pass 1 == pass 2 | field |
|---|---|---|---|---|---|
| v5.44.0 | whole-grid | 1 | 1.000 grids | -- | `4b91391b27b3cfb775adc5b3` |
| v5.44.0 | 32 (12 bands) | **24** | **2.000 grids** | **True** | `4b91391b27b3cfb775adc5b3` |
| v5.44.0 | 7 (55 bands) | **110** | **2.000 grids** | **True** | `4b91391b27b3cfb775adc5b3` |
| NEW | 32 | **12** | **1.000 grids** | -- | `4b91391b27b3cfb775adc5b3` |
| NEW | 7 | **55** | **1.000 grids** | -- | `4b91391b27b3cfb775adc5b3` |

and the NEW arm's single sequence of band-mask hashes equals **both** of
v5.44.0's sequences, band for band, at 12 bands and at 55.  The screen branch
is untouched: 12 and 55 calls, 1.000 grids, identical hashes on both arms.

That is the claim proved at its own level rather than inferred from the
field: what pass 2 used to recompute was bit-for-bit what pass 1 had already
computed, and the cache hands back exactly that.  The counts at N=4096 /
256-row bands are the ones the fix is priced on: **32 calls / 2.000 grids ->
16 / 1.000**, `eval_into` unchanged at 32 calls / 7.000 channels per pixel
(the residual duplication the branch deliberately leaves).

### 5e. The CHANGELOG correction text

Checked line by line against my numbers.  Every structural statement holds:
the table's route/inversion assignment, the two field hashes it cites, the
`inverse_map=False` remedy, "banding is free", "the 7/4 evaluation is not the
price", and the ray-density attribution.  Two qualifications are owed:

1. the 1.6x and the 8.9 s are one fixture's, as measured above -- the entry
   already says "read the ratios, not the seconds", but the RATIO 1.6x is
   itself fixture-dependent, and the docstring states it as the cost of the
   feature;
2. "the screen branch evaluates 3.000 channels per exit pixel in ONE pass,
   banded and whole-grid alike" is **CONFIRMED** on both fixtures (16 calls,
   3.000 ch/px banded; 1 call, 3.000 whole-grid).

---

## 6. D3 -- the single-precision transform pair

`q9_d3_fft.py`, my own two-group chain with an exact focus readout (which is
what reaches the crop), three arms, identical on both builds of the library
because D3 changed only comments.

| chain | arm | crop calls | rel L2 vs the complex128 chain | rel total power |
|---|---|---|---|---|
| 2 groups + exact readout | A: complex64, SHIPPED single-precision pair | 2 | **2.4573e-07** | **1.3392e-07** |
| | B: complex64, pair forced to complex128, narrowed ONCE | 2 | 2.4711e-07 | 1.3118e-07 |
| | C: complex128 | 2 | -- | -- |
| 2 groups, paraxial (no readout) | A | **0** | 1.1822e-07 | 8.5174e-09 |
| | B | **0** | identical to A | identical |

* **the premise**: `np.fft.fft2(complex64).dtype` is `complex64` on numpy
  2.4.4.  The rationale D3 corrects was indeed false.
* **the decision**: B is **not better** -- its L2 is 0.6 % worse and its power
  2 % better, i.e. the two arms are indistinguishable and the error is
  dominated by the complex64 storage, exactly as claimed.  A is 81x under the
  2e-05 field bar and 299x under the 4e-05 energy bar on my chain (64x / 159x
  on theirs).
* **the crop-call count**: a `final_leg='paraxial'` chain calls
  `_fourier_upsample_crop` **0 times**.  CONFIRMED.

The ladder, against a narrow-once reference:

| n_crop -> n_fine | max abs diff vs narrow-once | eps32 x peak | ratio | rel L2 vs the complex128-input result |
|---|---|---|---|---|
| 128 -> 256 | 3.5771e-07 | 1.4618e-07 | 2.45 | 1.7623e-07 |
| 256 -> 512 | 3.5794e-07 | 1.5669e-07 | 2.28 | 1.9243e-07 |
| 512 -> 1024 | 3.7697e-07 | 1.7525e-07 | 2.15 | 2.1077e-07 |
| 1024 -> 2048 | 4.2982e-07 | 1.7583e-07 | 2.44 | 2.2599e-07 |

The "2.6x eps32 x peak" reads 2.15-2.45 here, and the growth over four
octaves is **1.282x** against the `sqrt(log2 N)` prediction of 1.173x (the
builder measured 1.151x).  The claim's PURPOSE -- that extrapolating to a
16384 fine leg stays around 2e-07, two decades under the bar -- survives on
both fixtures; the functional form is a shape argument that my fixture fits
9 % worse than theirs.  **BOUNDED.**

---

## 7. D4 / D5 -- the recorded numbers, both builds

`q10_durability.py`, on the shipped tests' own fixtures, Windows and WSL.

| quantity | asserted | Windows | WSL | margin | the FIX doc records |
|---|---|---|---|---|---|
| `_C64_PHASOR_TOL`, R=45.9 mm / N=512 (max of the 4 helpers) | `<= 2.5e-7` | **4.203244e-08** | **4.203244e-08** | 5.948x | 4.2032e-08, 5.94x |
| `_C64_PHASOR_TOL`, R=5 mm / N=1024 (max of the 4 helpers) | `<= 2.5e-7` | **4.211665e-08** | **4.211665e-08** | 5.936x | 4.2117e-08, 5.94x |
| max argument, R=45.9 mm | -- | **22.3363 rad** | same | -- | 22.336 |
| max argument, R=5 mm | -- | **820.1891 rad** | same | -- | 820.19 |
| control error, R=45.9 mm | -- | **9.87193e-07** | same | -- | 9.8719e-07 |
| control error, R=5 mm | -- | **3.05233e-05** | same | -- | 3.0523e-05 |
| control ratio, R=45.9 mm | `>= 10x` | **23.486x** | same | **2.349x** | 23.486x, 2.35x |
| control ratio, R=5 mm | `>= 100x` | **727.038x** | same | 7.270x | 727.04x, 7.27x |
| control / `_C64_PHASOR_TOL` | -- | **3.949x** | same | -- | 3.95x |
| upsample crop rel L2 (512->256) | `<= 1e-5` | 1.715123178672e-07 | 1.715123178673e-07 | 58.3x | 1.71512e-07 |
| upsample crop rel L2 (256->512) | `<= 1e-5` | 1.688898531211e-07 | 1.688898531212e-07 | 59.2x | not recorded |
| exact readout rel L2 | `<= 1e-6` | **9.899520e-08** | **9.884099e-08** | **10.101x** / 10.117x | 9.8995e-08 / 9.8841e-08, 10.1x |
| one-group chain rel L2 | `<= 1e-6` | **9.442663e-08** | **9.442663e-08** | **10.590x** | 9.4427e-08, 10.6x |
| band memory saving (test fixture) | `>= 6` grids | **12.414612** | **12.290180** | 2.069x / 2.048x | 12.414831 / 12.290180 |
| band memory saving (MY fixture) | `>= 6` grids | 10.7778 | 10.7763 | 1.796x | -- |
| D2 tracemalloc gap at N=2048 | `>= 4 N^2` = 16.00 MiB | 35.00 MiB | -- | 2.188x | 35.00 MiB, 2.19x |

Every recorded figure reproduces.  The build spread is 0.156 % on the readout,
0 (10 figures) on the chain, 1.00 % on the band memory, and exactly 0 on
everything the float64 argument determines.

### 7a. Are the bars two-sided, and is any within a decade of build drift?

| bar | pass side | fail side | stated in the docstring? |
|---|---|---|---|
| `_C64_PHASOR_TOL` 2.5e-7 | 5.94x above the measurement | **3.95x below the float32-argument control** | yes, verbatim |
| control ratio 10x | 2.349x | (a design ratio, not a measured population) | yes, with the reason |
| control ratio 100x | 7.270x | as above | yes |
| crop 1e-5 | 58.3x | none -- one-sided | the number, yes; the one-sidedness, no |
| readout 1e-6 | 10.101x | none measured; build spread 0.156 % | yes |
| chain 1e-6 | 10.590x | none measured; build spread 0 | yes |
| band memory 6 grids | 2.069x | pre-banding = 0 saving | now yes, with the 1.00 % spread |
| D2 gap 16.00 MiB | 2.188x | **pre-fix gap exactly 0** | yes |

**No bar in these three files is within a decade of a build-moving
quantity.**  The two tight ones (`_C64_PHASOR_TOL` at 3.95x, the 10x control
ratio at 2.349x) are tight against a DESIGN quantity that is identical on both
builds to six figures, which is the honest reading and is what the docstrings
now say.  The three genuinely two-sided bars are `_C64_PHASOR_TOL` (a control
above it), the D2 memory gap (zero below it) and the band-memory saving.  The
crop, readout and chain bars are one-sided: they pin a ceiling with no
measured floor, which is acceptable for a precision bar but should not be
called two-sided.

### 7b. Durability changes made on this branch (test-only, docstrings)

| file | what was restated |
|---|---|
| `test_mixed_precision_carrier_helpers.py` | the D2 memory bar records that **N=2048 IS the regime**: the crossover table above, the break-even between 1600 and 1700, and the inherited `_phasor_rows` ratios |
| `test_banded_ray_density_and_inverse_map.py` | the band-memory bar's **measured** two-build spread (1.00 %) and a second fixture's 10.78 grids, replacing an argued "above anything a build can move" |
| `test_banded_ray_density_and_inverse_map.py` | the D7 probe test records that `np.array_equal` is FALSE on NaN, that its four pixels are interior by construction, and where the NaN-bearing case is covered |

No assertion, bar, fixture or parametrisation was changed.

---

## 8. Tests, lint, timings

### 8a. The three touched files, re-timed after the docstring restatements

```
OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS=1 python -m pytest \
  test_mixed_precision_carrier_helpers.py \
  test_banded_ray_density_and_inverse_map.py \
  test_niche_perf_round2_2026_08_10.py -q -p no:randomly
-> 61 passed, 1 warning in 44.82s
```

The one warning is `_lens_traced.py:873` `divide by zero encountered in
divide` from `test_the_chunk_worker_accepts_both_arg_shapes_identically`, a
pre-existing notice of that fixture's degenerate Jacobian, not this branch's.

### 8b. The named run

The four lens test files (`test_banded_ray_density_and_inverse_map`,
`test_lens_chunked_sag`, `test_mixed_precision_carrier_helpers`,
`test_niche_s10_sibling_patterns`), `test_niche_perf_round2_2026_08_10`,
`test_niche_d15_deterministic_traced_fit`,
`test_niche_d14_deterministic_carrier_fit`, `test_fix_runner_oom_2026_08_13`,
`test_niche_d7_decentred_fit`, `test_niche_d8_congruence_workers`, and the
census / walker / dispatcher-pin / public-API / doc-consistency sweep
(`ls tests/unit | grep -iE 'walker|census|dispatcher_pin|public_api|doc_consistency'`,
25 files) -- **35 files** in total, list in
`validation/probe_verify_lens_followups/results/_testfiles.txt`:

```
OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS=1 python -m pytest <35 files> -q -p no:randomly
-> 1515 passed, 12 skipped, 146 warnings in 912.53s (0:15:12)
```

Zero failures.  All 12 skips are pre-existing "nothing to verify" skips, none
of them touched by this branch: 4 in `test_v4_14_2_dispatcher_pin_cache_locks`
(exempted locks that guard operations rather than caches) and 8 CHANGELOG
walkers reporting that the topmost `## [5.44.0]` block cites no audit IDs, has
no audit-closure bullets and carries no pass/skip, "N files touched" or
"CHANGELOG.md: X -> Y lines" claim.  The 146 warnings are the aperture:beam,
evaluator-refusal and HFPI-undersampling notices these fixtures have always
raised.

SECOND BUILD, the three touched files plus the two that keep an
`inverse_map=False` scoping (`test_lens_chunked_sag`,
`test_niche_s10_sibling_patterns`):

```
wsl.exe -e bash -lc "cd /mnt/c/tmp/lum_vlens2 && OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS=1 \
  PYTHONPATH=/mnt/c/tmp/lum_vlens2 ~/lumvenv/bin/python -m pytest <5 files> -q -p no:randomly"
-> 105 passed, 80 warnings in 125.98s     (py 3.12.3, numpy 2.4.6)
```

### 8c. Lint

`python -m ruff check lumenairy/ tests/` -> **All checks passed!**
(`validation/` is `extend-exclude`d by `pyproject.toml`, as for every other
probe directory.)

### 8d. `.test_durations`

**Nothing to splice.**  This verification adds no test node ids -- the three
test-file changes are docstrings only (`git diff --stat` on
`tests/`: 2 files, docstring hunks), so pytest-split's store is unchanged and
the builder's splice at `72776b5` still describes the tree.

---

## 9. Defects

| id | severity | where | what | evidence |
|---|---|---|---|---|
| **V1** | cosmetic (recorded number, in an audit document) | `FIX_LENS_5440_FOLLOWUPS_2026_09_11.md` sec 1a, table, and the sec 0 verdict row | the ASTIGMATIC "complex64 BEFORE" column reads 128.00 / 512.00 MiB where the measurement is **96.00 / 384.00**, so the four "saved" cells read 64.00 / 256.00 MiB and -50.0 % where the truth is **32.00 / 128.00 MiB and -33.3 %**; and the sentence "the complex64 arm's peak was EXACTLY the complex128 arm's before the fix, **on every case**" is false on all eight astigmatic cases | the builder's own `p1_before.json`, their probe re-run unchanged on the v5.44.0 worktree, and my independent probe -- three sources, identical to the byte |
| **V2** | cosmetic by the correctness taxonomy, PRODUCTION-RELEVANT by memory below N ~ 1640 | `carrier.py`, `_build_carrier_phase` -> `_phasor_rows`; recorded wrongly in `FIX_LENS_5440_FOLLOWUPS` sec 1a | "below N ~ 1414 ... there is no transient saving at all" understates a **14.3 % INCREASE** in the complex64 per-call peak against v5.44.0 (56.017 -> 64.017 MiB at N=1024), which persists to a break-even between N=1600 and N=1700 and leaves the complex64 call peaking above the complex128 one.  Inherited from `_phasor_rows` (identical ratios on both versions for the four helpers that already had `dtype=`), so this is a NEW instance of an OLD cost on the public call path, not a new mechanism | `q5b_band_crossover.py` (11 N x 2 carriers x 2 dtypes x 2 arms, tracemalloc byte-exact), `q5c_phasor_rows_penalty.py` |
| **V3** | cosmetic (over-general statement, now in a PUBLIC docstring and the CHANGELOG) | `_lens_traced.py`, `apply_real_lens_traced`'s `sag_chunk_rows` docstring; CHANGELOG "CORRECTION 2026-09-11" items 2 and 3 | "a banded SCREEN call at the shipped default costs about 1.6x what it did on v5.43.0 ... That price is the evaluator's whole-grid DOMAIN TEST (~8.9 s)" is one fixture's measurement stated as the feature's cost.  A second fixture at the same N / sub / dx reads **0.954x** against the same-build incumbent control and `domain_mask` = **0.41 s** of a 13 s call.  Which branch `domain_mask` takes (the separable screened hull test vs the whole-grid signed distance) is a property of the exit-support geometry | `q7_d6_route.py` wall + stages on both fixtures, `p3_d6_time.py` re-run unchanged on three arms |
| **V4** | cosmetic (unobservable rationale in a docstring) | `_lens_traced.py`, `_ray_density_self_checks` docstring; `VERIFY_LENS_BANDED_COMPLEX64_2026_09_10` D1 | "the default filter's per-location dedup registry is indexed by [the reported module]" is true of CPython but INERT here: one `apply_real_lens_traced` call moves the warnings filter version 10 -> 25, invalidating every registry, so two different caller modules each warn on both arms (10 notices, both) | `q3_d1_warn_attr.py` `_dedup` block; a direct filter-version probe |
| **V5** | cosmetic (test durability, pre-existing on this branch's new test) | `test_banded_ray_density_and_inverse_map.py::test_the_c15_probe_is_filled_on_the_band_path` | `np.array_equal` without `equal_nan=True` is FALSE wherever a probe pixel lands out of the model's domain, which is the normal state for most of an exit grid; the four fixture pixels are interior, so moving them would fail the test for a reason that is not the defect | recorded in the test docstring on this branch; the NaN case is covered in `q4_d7_probe_rc.py` (82 of 112 pixels out of domain, equal including the NaN pattern) |
| **V6** | cosmetic, pre-existing, NOT this branch's | `_lens_traced.py`, the nested `apply_real_lens` call | the two `apply_real_lens: ... aperture(s) exceed the simulation grid` notices a traced call raises are attributed to `_lens_traced.py`, so a user sees the same sentence twice, once against their own line and once against the library's | `q2_traced_identity.py` (55 calls), `q3` |

Nothing above is a wrong answer, and nothing above blocks the release.

---

## 10. Recommendation

**SHIP in 5.45.0, with three documentation corrections and no code change.**

The seven follow-ups do what they claim where it matters: every bit that had
to stay still stayed still (3,000+ compared leaves over 55 traced calls, 24
carrier chains / readouts / crops, 76 carrier-helper calls and 15 probe
routes, on my own fixtures and on both arms), the two memory and runtime
changes are real and are proved at their own level rather than inferred (the
5 -> 0 phasor census; the mask-hash sequences that show the removed pass-2
computation was bit-for-bit redundant), the warning attribution is fixed, the
private probe now works on the path that needed it, and every recorded number
in the three touched test files reproduces to the digit on two builds.

Before tagging:

1. **correct the astigmatic "before" column of `FIX_LENS_5440_FOLLOWUPS`
   sec 1a** (96.00 / 384.00 MiB, saving 32.00 / 128.00 MiB, -33.3 %) and drop
   the "EXACTLY the complex128 arm ... on every case" sentence.  The builder's
   own probe JSON already disagrees with the table;
2. **qualify the sub-crossover statement** -- "no transient saving" ->
   "a 14.3 % penalty below the crossover, break-even near N=1650".  The test
   docstring on this branch now carries the measurement;
3. **qualify the 1.6x and the 8.9 s** in `apply_real_lens_traced`'s docstring
   and in the CHANGELOG correction as one geometry's measurement.  A public
   docstring is the wrong place for a fixture-specific constant; naming
   `inverse_map=False` as the escape hatch is right and should stay.

Optionally, drop the dedup-registry sentence from the
`_ray_density_self_checks` docstring (V4) -- the `filterwarnings(module=...)`
reason alone justifies the fix and is observable.

---

## 10a. Commits

Branch `verify/lens-followups`, off `24651c8` (`wave2/pmm2d` with
`fix/lens-5440-followups` merged).  No library file, no CHANGELOG entry, no
version bump; the only files outside `validation/` are two test docstrings and
this document.

| commit | what |
|---|---|
| `aa3d779` | `test(probes)`: task-1 bit identity vs v5.44.0 -- own fixtures, both arms (D1, D7).  `_vf.py` (fixtures + the arm banner that refuses a mismatched tree), `qdiff.py`, `q1`, `q2`, `q3`, `q4` and their JSON |
| `0066adb` | `test(probes)`: D2 memory (`q5`, `q5b`, `q5c`), D6 (`q7`, `q8`), D3 (`q9`), durability (`q10`), `run_builder_probe.py` and the README |
| `e63833a` | `test(probes)`: the measured JSON for D2, D3, D6 and the durability table -- both Windows arms, the WSL second build, the v5.43.0 arm, and the builder's own p1 / p3 / v6 / v9 probes re-run unchanged per arm |
| `8921e73` | `test(durability)`: state the regime and the build spread of the three tight bars.  Docstrings only, no assertion / bar / fixture change; 61 passed in 44.8 s |
| (this file) | `docs(audits)`: VERIFY_LENS_5440_FOLLOWUPS_2026_09_11 |

## 11. What could not be verified

1. **The idle-box absolute seconds.**  Every wall-clock number here was taken
   with three other agents on the box (62-83 % CPU).  Ratios, hashes and
   counts are load-free and are what the D6 conclusions rest on; the
   verification's own idle-box V15 table remains the reference for absolute
   seconds and was not re-taken.
2. **The GPU / JAX arms** of either path -- no GPU on this box, and neither
   change touches the JAX twin (`bld is np` guards every new branch).
3. **The 16384-square extrapolations** (D2's 4.29 GB per call, D3's ~2e-07 at
   a 16384 fine leg).  Both are arithmetic on measured slopes; N was capped at
   4096 here.
4. **The halo self-check's attribution** -- like the builder, I could not make
   it fire on a fixture that also fires the other two, so its `stacklevel` is
   verified only by shared frame depth with the two that do fire.
5. **The prototypes' figures** and the design-121 `.zmx` arms cited in the
   original 5.44.0 entry, unchanged from the verification's own section 12.
6. **CROSS-BUILD bit identity of anything complex128.**  It does not hold and
   was not claimed: `np.exp(1j*arg)` on the same float64 argument returns
   different bits on numpy 2.4.4 and 2.4.6 (sec 4bb).  Every identity result
   here -- and in the two documents this verifies -- compares two arms of the
   SAME build, which is the only comparison those hashes support.  The
   complex64 narrowing IS portable across the two builds, measured.
7. **`_PHASOR_BAND_BYTES` at other values.**  `_narrow_rows` inherits the
   32 MB constant; its value-inertness is argued (an elementwise product on a
   row slice) and confirmed at the shipped value on 76 hashes, but not swept
   over band sizes.
