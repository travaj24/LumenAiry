# FIX -- the seven follow-ups of VERIFY_LENS_BANDED_COMPLEX64_2026_09_10

**Date** 2026-09-11 **Branch** `fix/lens-5440-followups` (worktree
`C:/tmp/lum_lensfix`, off `824994c` = integration branch `wave2/pmm2d`)
**Subject** D1-D7 of `docs/audits/VERIFY_LENS_BANDED_COMPLEX64_2026_09_10.md`.

The verification confirmed every correctness claim of the two 5.44.0
traced-lens changes bit for bit, on two builds, and recommended a small
low-risk follow-up.  These are those follow-ups.  Nothing here repairs a wrong
answer: two are memory / runtime, one is a warning's attribution, one fills a
private diagnostic, and three are recorded numbers that did not reproduce.

Every number below is re-measured on this tree.  Probes live in
`validation/probe_fix_lens_5440/`; each asserts `lumenairy.__file__` before it
measures anything.  Where a quantity the verification hashed is involved, its
probe (`validation/probe_verify_lens_5440/`) is re-run UNCHANGED on this tree
and the JSON diffed against the recorded result -- so bit-identity is checked
against the verification's own fixtures, not against a re-derivation.

## Arms

| arm | tree | build |
|---|---|---|
| FIX | `C:/tmp/lum_lensfix` @ `fix/lens-5440-followups` | Windows 11, py 3.14.6, numpy 2.4.4, scipy 1.17.1 |
| SECOND BUILD | WSL Ubuntu `~/lumvenv` on `/mnt/c/tmp/lum_lensfix` | py 3.12.3, numpy 2.4.6, scipy 1.17.1 |
| v5.43.0 (D6 only) | `C:/tmp/lum_v5430b` @ `v5.43.0` (`78e4091`, worktree, removed at the end) | as FIX |

`OMP/OPENBLAS/MKL_NUM_THREADS=1` on every measurement.

**Box-load caveat, and it applies only to the wall-time tables (D6).**  Three
other agents ran compute on this box throughout (24 logical CPUs, ~83 % busy).
Cross-build ABSOLUTE seconds are therefore not comparable to the
verification's idle-box V15 table, and the D6 conclusion does not rest on
them.  What it rests on is the WITHIN-BUILD control -- 5.44.0's banded screen
call forced back onto the incumbent with `inverse_map=False`, measured in the
same process and the same minutes as the arm it is compared against -- which
returns v5.43.0's banded FIELD HASH bit for bit and lands on v5.43.0's time.

---

## 0. Verdict table

| id | what shipped | headline number |
|---|---|---|
| **D2** | `dtype=` threaded through `_build_carrier_phase`; new `_narrow_rows` for the astigmatic product; both public helpers pass the field's dtype | per-call transient at N=4096 on a complex64 field: **896.06 -> 640.06 MiB** (scalar carrier) and **512.00 -> 256.00 MiB** (astigmatic); full-grid complex128 phasors per two-group chain **5 -> 0**; every field hash unchanged |
| **D1** | `stacklevel=2 -> 3` on the three ray-density self-check warnings | warnings attributed to the library **6 of 9 -> 0 of 9**; field hash unchanged |
| **D6** | CHANGELOG bullet corrected with a dated addendum; `apply_real_lens_traced` docstring priced | the doubling IS the route change, **confirmed by a same-build control that reproduces v5.43.0's exact bits**; but the *reason* the bullet gives is wrong -- the screen branch evaluates **3/3 in ONE pass** and the evaluations are **0.88 s of 18.52 s** |
| **D3** | measured and KEPT; three rationale comments and one test docstring corrected | complex64 chain with the single-precision transform pair: rel L2 **3.112e-07**, rel power **2.520e-07** -- 64x / 159x under the bars, and NOT worse than a forced complex128 pair (3.156e-07 / 2.334e-07) |
| **D4** | four recorded measurements corrected | arguments **22.34 / 820.19 rad** (not 3.6e+02 / 1.3e+04); control **9.8719e-07 / 3.0523e-05** (not 1.5e-05 / 4.9e-04); ratios **23.5x / 727.0x** (not 360x / 11 600x) |
| **D5** | two 1e-4 bars tightened to 1e-6; the readout test now records its number | readout **9.8995e-08** (Win) / **9.8841e-08** (WSL); one-group chain **9.4427e-08** on both; margins 1010x/1059x -> **10.1x / 10.6x** |
| **D7** | the niche-C15 probe is gathered band by band and handed back | banded `probe_opl` **absent -> present**, and `np.array_equal` to the whole-grid arm's; field unchanged |

---

## 1. D2 -- the seventh reference phase

`_build_carrier_phase` (`carrier.py:1663` in the shipped file) took no
`dtype=`, so `carrier_referenced_envelope` / `carrier_referenced_reconstruct`
built a full-grid complex128 phasor even on a complex64 chain -- 5 per
two-group chain, 16 MiB each at N=1024 and 4.29 GB each at N=16384.  It now
takes `dtype=` exactly as the four helpers the 5.44.0 change did give one:
`None` / `complex128` is the shipped whole-grid `np.exp`, and `complex64`
takes the `_phasor_rows` route on the radial branch.

The ASTIGMATIC branch is a PRODUCT of two already-exponentiated per-axis
factors rather than one `exp`, so it gets a value-side sibling, `_narrow_rows`,
which stores `px[j] * py[i]` one row band at a time.  The multiply is
elementwise either way, so the stored complex64 array is bit for bit what
`(phase * py).astype(complex64)` returns -- the full-grid complex128 transient
is the only thing that leaves.

### 1a. The per-call transient (`p1_d2_transient.py`)

Whole-call `tracemalloc` peak of ONE public-helper call, warm.  "before" is
the shipped code measured on this tree before the edit; "after" the same
script after it.  One `c64 grid` = `8 N^2` bytes.

| N | carrier | helper | complex128 arm | complex64 BEFORE | complex64 AFTER | saved |
|---|---|---|---|---|---|---|
| 2048 | scalar | envelope | 224.03 MiB (7.00 grids) | 224.03 (7.00) | **189.03 (5.91)** | 35.00 MiB, -15.6 % |
| 2048 | scalar | reconstruct | 224.03 | 224.03 | **189.03** | 35.00 MiB |
| 2048 | astigmatic | envelope | 128.00 (4.00) | 128.00 (4.00) | **64.00 (2.00)** | 64.00 MiB, -50.0 % |
| 2048 | astigmatic | reconstruct | 128.00 | 128.00 | **64.00** | 64.00 MiB |
| 4096 | scalar | envelope | 896.06 (7.00) | 896.06 (7.00) | **640.06 (5.00)** | 256.00 MiB, -28.6 % |
| 4096 | scalar | reconstruct | 896.06 | 896.06 | **640.06** | 256.00 MiB |
| 4096 | astigmatic | envelope | 512.00 (4.00) | 512.00 (4.00) | **256.00 (2.00)** | 256.00 MiB, -50.0 % |
| 4096 | astigmatic | reconstruct | 512.00 | 512.00 | **256.00** | 256.00 MiB |

Two readings worth keeping:

* the complex64 arm's peak was EXACTLY the complex128 arm's before the fix, on
  every case -- which is the audit's "requesting complex64 saved 0.0 GB", still
  true of this call site at 5.44.0;
* the scalar saving at N=2048 is 35 MiB rather than 64 because
  `_PHASOR_BAND_BYTES` = 32 MB of complex128 scratch replaces the full-grid
  phasor with a 32 MB band; at N=4096 the band is 32 MB against a 256 MiB
  phasor, so the saving approaches the whole factor.  Below N ~ 1414 the band
  IS the grid and there is no transient saving at all (only the narrower
  output) -- which is why the shipped memory pin for this item runs at N=2048.

### 1b. Values: nothing moved

* all **16** p1 cases (2 N x 2 carriers x 2 helpers x 2 dtypes) return the
  same field hash before and after -- complex128 AND complex64;
* the complex128 carrier chain is bit-identical on all **9** fixtures the
  verification hashed (`v14_c128_chain_identity.py` re-run; full JSON diff =
  wall-clock seconds only):

  | case | hash (unchanged) |
  |---|---|
  | `chain2_sphere` | `577e209a0757a57bab89857f` |
  | `chain2_parab` | `ae7983bcf984d93609759cf1` |
  | `chain3_sphere` | `c6dda90c08bd2f8830a348b9` |
  | `chain2_final_leg` | `4cd9552cefd80598e5a04fd6` |
  | `chain2_readout` | `ab751a2265db2c814047852c` |
  | `readout_direct` | `65b759bab27be8cac1cecd2c` |
  | `crop_128_256` / `crop_256_128` / `crop_256_256` | `5baf422b957ea82d0de43dd5` / `4c5d9fd17d5ebab8ede34b1c` / `b3d0a0b2f94beb33ea1cab24` |

  every per-stage `dx` and `R_out` identical as well;
* the complex64 two-group chain's field is unchanged to all 16 digits
  (`v6_c64_chain.py` re-run): relative L2 against the complex128 arm
  **1.157936782559319e-07** before and after (field bar 2e-05), relative total
  power **7.688986734224147e-09** (energy bar 4e-05).  The upsample-crop and
  six-leg blocks of that probe are byte-identical JSON.

### 1c. The leak itself, counted by caller frame (`v7_c64_memory.py`)

| arm | full-grid complex128 phasor returns | bytes |
|---|---|---|
| complex128 chain, before and after | 10 | 160.0 MiB |
| complex64 chain, BEFORE | **5** (all `_build_carrier_phase -> _radial_carrier_phase`) | 80.0 MiB |
| complex64 chain, AFTER | **0** | 0.0 MiB |

By call signature, on the complex64 arm:

```
('_radial_carrier_phase', dtype=None,      -> complex128)   5 -> 0
('_radial_carrier_phase', dtype=complex64, -> complex64)    0 -> 5
('_phasor_rows',          -> complex64)                     5 -> 10
('_sphere_parab_conversion', dtype=complex64 -> complex64)  5 -> 5
```

The whole-CHAIN peak at N=1024 is unchanged (376.9 MiB complex128 arm,
332.7 MiB complex64 arm, saving 44.1 MiB = 11.7 %) because at that size the
chain's peak is a traced-element stage, not the carrier phasor.  The win this
fix buys is the per-call transient in 1a, and it is a GB-scale one only where
the audit said it was: 4.29 GB per call at N=16384.

### 1d. Test

`tests/unit/test_mixed_precision_carrier_helpers.py`, 10 new cases, two-sided:

* `test_build_carrier_phase_builds_no_complex128_grid_on_a_complex64_field`
  (6 cases: 2 public helpers x scalar / astigmatic / one-finite-axis) -- no
  reference-phase helper returns a full-grid complex128 array, the factor is
  complex64, and it EQUALS the whole-grid complex128 factor narrowed once
  (`np.array_equal`), on both branches;
* `test_build_carrier_phase_complex128_is_the_shipped_whole_grid_build`
  (3 cases) -- `dtype=None`, `dtype=complex128` and the public helper are all
  `np.array_equal`;
* `test_the_complex64_carrier_call_no_longer_pays_a_full_grid_complex128` --
  the memory teeth at N=2048, bar `p128 - p64 >= 4 N^2` bytes (16.00 MiB)
  against a measured 35.00 MiB on BOTH builds (identical to the byte:
  tracemalloc counts REQUESTED sizes, fixed by shapes and dtypes) and a
  pre-fix gap of exactly 0.

---

## 2. D1 -- the self-check warnings name the caller again

The three `warn(...)` calls inside `_ray_density_self_checks` kept
`stacklevel=2` when the block moved into a nested closure, which is one frame
short there.  `2 -> 3`; the two sibling closures already carried 3.

`p2_d1_attr.py` drives all five ray-density notices over their thresholds on a
decentred `origin` + `remap_sampling='full'` fixture and records
`w.filename` / `w.lineno`:

| rows | before | after | field hash |
|---|---|---|---|
| 0 (whole-grid) | 1/3 from the caller; energy + support-band -> `_lens_traced.py:12350` | **3/3 from the caller** | `a78a7ff1533ade9e24e9f84a` |
| 32 | 1/3; the same two -> `_lens_traced.py:12236` | **3/3** | `a78a7ff1533ade9e24e9f84a` |
| 7 | 1/3; `:12236` | **3/3** | `a78a7ff1533ade9e24e9f84a` |

**6 of 9 -> 0 of 9** attributed to the library; the field hash is identical on
every arm before and after.  The origin/D9 verdict was already correct (it
carries `stacklevel=3`), which is what makes the fixture a two-sided control:
the same call shows one notice landing correctly and two not.

Test: `test_ray_density_self_check_warnings_name_the_caller` (3 cases, whole /
band32 / band7).  Fail-before verified by reverting the three characters:
**3 failed at `stacklevel=2`, 3 pass at 3**.

---

## 3. D6 -- what the banded SCREEN route's doubling actually is

The CHANGELOG prices the change as *"Wall time is neutral on the coarse route
and ~10 % higher on the evaluator route (the 7/4 evaluation)"*, and the
verification refuted that as stated: the banded SCREEN call at the shipped
default -- the one AUTO banding gives every `N >= 4096` traced call -- went
10.529 s -> 21.890 s across the release, and the bullet does not price it.  It
ATTRIBUTED that to the route change (the banded screen call now runs the
evaluator where v5.43.0 silently ran the coarse-Newton incumbent) without
measuring the attribution.

### 3a. The attribution, confirmed -- and it is airtight by HASH, not by time

`p3_d6_time.py`, N=4096 / sub=32 / dx=1.5 um, AUTO band height (256 rows),
best of 3, `tracemalloc` off, inverse-map cache cleared before every timed
call.  The third row of each block is the CONTROL this test adds: 5.44.0's
banded screen call forced back onto the incumbent with `inverse_map=False`.

| build | route | wall (best of 3) | `eval_into` calls | channel-evals / px | field hash |
|---|---|---|---|---|---|
| 5.44.0 | screen, whole-grid | 19.006 s | 1 | 3.000 | `91da5e59c46beb1d967b28b3` |
| 5.44.0 | screen, AUTO-banded | **18.524 s** | 16 | 3.000 | `91da5e59c46beb1d967b28b3` |
| 5.44.0 | screen, AUTO-banded, `inverse_map=False` | **11.856 s** | 0 | -- | **`66b9384477cf8a9ca1844c6a`** |
| v5.43.0 | screen, whole-grid | 18.148 s | 1 | 3.000 | `91da5e59c46beb1d967b28b3` |
| v5.43.0 | screen, AUTO-banded (the shipped default there) | **10.950 s** | 0 | -- | **`66b9384477cf8a9ca1844c6a`** |
| v5.43.0 | screen, AUTO-banded, `inverse_map=False` | 10.283 s | 0 | -- | `66b9384477cf8a9ca1844c6a` |

Read the hashes first.  v5.43.0's banded screen answer and 5.44.0's
forced-incumbent control are **the same field, bit for bit**, and 5.44.0's
banded and whole-grid evaluator answers are the same field as v5.43.0's
whole-grid evaluator answer.  So the control is not an approximation of the
old route -- it IS the old route, on the new build, in the same process as the
arm it is compared against.  It lands on v5.43.0's time (11.856 s against
10.950 / 10.283 s) while the evaluator arm sits at 18.524 s.  **The doubling
is the route change, confirmed.**

Two things fall out that the CHANGELOG bullet gets wrong:

* **banding is free on this branch.**  18.524 / 19.006 = **0.975** -- the
  banded screen call is not slower than the whole-grid one at the same
  inversion.  At N=2048 the same pair reads 5.314 / 4.914 = **1.081**, which
  is where the entry's "~10 %" comes from;
* **the 7/4 evaluation is not what it costs.**  The screen branch evaluates
  **3.000 channels per exit pixel in ONE pass**, banded and whole-grid alike
  (the 7/4 belongs to the ray-density branch), and `eval_into` accounts for
  **0.88 s of the banded call's 18.52 s**.

### 3b. Where the evaluator route's time actually goes

`p6_d6_stages.py` splits the call one level finer.  N=4096 / sub=32, best of 2,
instrumented (the wrappers perturb the total slightly; read the split, not the
total).

| route | total | `eval_into` | **`domain_mask`** | `build_inverse_map` | `map_coordinates` | rest |
|---|---|---|---|---|---|---|
| screen, whole-grid | 21.977 s | 0.81 (3.00 ch/px) | **9.97 s (1.00 grids)** | 0.43 | 0.00 | 10.77 |
| screen, AUTO-banded | 20.196 s | 0.91 (3.00) | **8.89 s (1.00 grids)** | 0.41 | 0.00 | 9.98 |
| screen, banded incumbent | 11.233 s | 0.00 | **0.00 (0 grids)** | 0.00 | 1.72 | 9.52 |
| ray-density, whole-grid | 24.747 s | 1.03 (4.00) | **8.83 s (1.00 grids)** | 0.38 | 0.90 | 13.61 |
| ray-density, AUTO-banded | 31.208 s | 1.87 (7.00) | **15.05 s (2.00 grids)** | 0.38 | 0.80 | 13.11 |
| ray-density, banded incumbent | 12.925 s | 0.00 | 0.00 | 0.00 | 2.51 | 10.41 |

The evaluator route's price is `domain_mask` -- the screened landing-hull test
plus the entrance-radius test, over every exit pixel -- at **8.9-10.0 s**,
against 0.8-0.9 s of channel evaluation and 0.4 s of model build.  The
incumbent runs no domain test at all (0 grids) and pays 1.7 s of
`map_coordinates` instead.  Evaluator-specific work on the banded screen call
is 8.89 + 0.91 + 0.41 = 10.21 s against the incumbent's 1.72 s: a difference
of 8.5 s, against the 8.96 s the two totals differ by.  The attribution
closes.

### 3c. The part that was NOT the evaluator, and is now fixed

The ray-density row above shows `domain_mask` running over **2.00 grids** of
pixels on the banded arm against **1.00** on the whole-grid arm.  That is the
two-pass band loop recomputing an IDENTICAL mask: pass 1 evaluates
`CH_X_IN` / `CH_Y_IN` and takes the mask for the census, pass 2 evaluates the
same two channels again and re-derived the same mask from them.  It cost
**15.05 - 8.83 = 6.22 s of the banded ray-density route's 6.46 s penalty**;
the 7/4 evaluation cost 0.84 s of it.

**Fixed**: pass 1 stores its mask in one bool grid (`N^2` bytes = 1/8 of a
float64 grid) and pass 2 takes it (`_eval_band(..., ok=...)`).  Byte-identical
by construction -- `eval_into` writes each channel independently, which is the
same property that makes the banded field bit-identical in the first place --
and verified as such:

| ray-density route | `domain_mask` | calls | pixels tested | `eval_into` | total | banded / whole |
|---|---|---|---|---|---|---|
| whole-grid, BEFORE run | 8.83 s | 1 | 1.00 grids | 1.03 s (4.00 ch/px) | 24.747 s | -- |
| banded, BEFORE | **15.05 s** | 32 | **2.00 grids** | 1.87 s (7.00 ch/px) | 31.208 s | **1.261** |
| whole-grid, AFTER run | 6.81 s | 1 | 1.00 grids | 0.93 s (4.00) | 20.179 s | -- |
| banded, AFTER | **6.61 s** | **16** | **1.00 grids** | 1.87 s (7.00) | 21.306 s | **1.056** |

The **pixel count** is the load-free statement: 2.00 grids -> 1.00, 32
`domain_mask` calls -> 16, with the 7/4 channel evaluation untouched at
1.87 s.  The seconds moved between the two runs because the box got quieter
(the UNCHANGED whole-grid arm reads 24.747 s and 20.179 s across them), which
is exactly why the ratio is quoted: **1.261 -> 1.056**.

The same comparison on the UN-instrumented best-of-3 wall clock (`p3`), which
is the number a caller sees:

| N, sub | ray-density banded / whole-grid, BEFORE | AFTER |
|---|---|---|
| 4096, 32 | 32.486 / 25.285 = **1.285** | 21.000 / 18.914 = **1.110** |
| 2048, 16 | 7.462 / 6.604 = **1.130** | 5.760 / 5.628 = **1.023** |

and the SCREEN branch, which the fix does not touch, stays where it was:
15.819 / 15.287 = 1.035 at N=4096 and 4.871 / 5.630 = 0.865 at N=2048 (the
0.97x-1.08x band, now with a sub-unity reading -- this box's run-to-run spread
on a ~5 s call).  Every field hash in both runs is unchanged
(`91da5e59c46beb1d967b28b3` / `08f9aa00b567ed87658cf8ca` at N=4096,
`218be92f7c6d9087c27b4369` / `14a892c7b3ccb66ad600cae0` at N=2048, and the
incumbent controls `66b9384477cf8a9ca1844c6a` / `b0a55957c3feaadc66f87bba` /
`f7393dea35d4d3af77cf37a1` / `c0d2d94647d54c8c6329b905`).

Byte-identity, checked against the verification's OWN fixtures rather than a
re-derivation -- `v2_banded_claim.py` re-run in all four regimes on this tree:

| regime | comparisons | mismatches | band hashes vs the verification's record |
|---|---|---|---|
| plain | 180 | **0** | all equal |
| `--forced` (the three self-checks + the D9 verdict driven over threshold) | 180 | **0** | all equal |
| `--fold` (the fold-caustic warning fires on every ray-density case) | 252 | **0** | all equal |
| `--medianbite` (caustic floor 1.5x median -- the census median is IN the field) | 252 | **0** | all equal |
| **total** | **864** | **0** | |

`sum |E|^2`, `engaged`, `gate_open`, `n_out_of_domain` and the full warning
list are compared alongside the hash in every row.

### 3d. What the CHANGELOG now says

The 5.44.0 "Measured" bullet is left as written and a dated **CORRECTION**
bullet is appended under it (history is not rewritten): banding is free
(0.97x-1.08x at the same inversion), the banded SCREEN call at the shipped
default costs ~1.6x what it did on v5.43.0 because it runs the evaluator, the
control that proves it returns v5.43.0's bits, the price is the whole-grid
domain test and not the 7/4 evaluation, and the ray-density branch's remaining
penalty was the duplicated mask now fixed.  `apply_real_lens_traced`'s
`sag_chunk_rows` docstring carries the same in three sentences, and names
`inverse_map=False` as the way to buy the old speed back at the old answer.

---

## 4. D3 -- the transform pair runs in single precision, and that is accepted

On numpy >= 2.0 `np.fft.fft2(complex64)` returns complex64, so BOTH transforms
of `_fourier_upsample_crop`'s pair run in single precision for a complex64
envelope.  The recorded rationale ("numpy's FFT is still double-only, so a
complex64 input is transformed in complex128 and NARROWED back on return") has
been false since numpy 2.0.

The decision needed a number on a REAL chain, so `p4_d3_fft.py` runs a
two-group traced carrier chain WITH an exact focus readout (`final_leg='exact'`
-- the plain chain never reaches the crop at all: 0 calls, which is itself
worth recording) in three arms:

| chain | arm | rel L2 vs the complex128 chain | rel total power |
|---|---|---|---|
| 2 groups + exact readout (2 crop calls) | A: complex64, SHIPPED single-precision pair | **3.112e-07** | **2.520e-07** |
| | B: complex64, pair forced to complex128, narrowed ONCE | 3.156e-07 | 2.334e-07 |
| | A vs B (the transform's own share) | 2.662e-07 | -- |
| 2 groups, no readout (0 crop calls) | A | 1.158e-07 | 9.669e-09 |
| | A vs B | 0 (identical -- the crop is not reached) | -- |

**Decision: KEEP the single precision.**  A is 64x under the 2e-05 field bar
and 159x under the campaign's 4e-05 energy bar, and B is not better -- on a
complex64 chain the error is dominated by the complex64 STORAGE of the
envelope, not by the transform.  Promoting would cost a full-grid complex128
transform pair on the memory-dominant stage to buy nothing measurable.

Directly on the crop, against a narrow-once reference, the cost is 2.6x
`eps32 x peak` (a narrow-on-return would be <= 0.5x) and grows as
~`sqrt(log2 N)`:

| n_crop -> n_fine | max abs diff vs narrow-once | eps32 x peak | rel L2 vs the complex128-input result |
|---|---|---|---|
| 128 -> 256 | 3.600e-07 | 1.355e-07 | 1.454e-07 |
| 256 -> 512 | 3.604e-07 | 1.370e-07 | 1.472e-07 |
| 512 -> 1024 | 4.214e-07 | 1.393e-07 | 1.545e-07 |
| 1024 -> 2048 | 4.815e-07 | 1.437e-07 | 1.674e-07 |

15 % over four octaves -- so the extrapolation to a 16384 fine leg stays
around 2e-07, two decades under the bar.

Corrected: the two rationale comments in `carrier.py` (the crop's DTYPE PARITY
note and `_crop_about_centre`'s), and the docstring of
`test_upsample_crop_keeps_the_envelope_dtype`.  No behaviour change; the
26 tests of `test_niche_perf_round2_2026_08_10.py` pass unchanged.

---

## 5. D4 / D5 -- the recorded numbers, on both builds

`v9_durability.py` re-run on Windows and on WSL.  Every value is identical
across the two builds except the two marked, and every value except the
tracemalloc row is identical to what the verification recorded -- so these are
re-measurements, not new fixtures.

| test / constant | asserted | Windows | WSL | margin | docstring said |
|---|---|---|---|---|---|
| `_C64_PHASOR_TOL`, R=45.9 mm | `<= 2.5e-7` | 4.2032e-08 | 4.2032e-08 | 5.94x | -- |
| `_C64_PHASOR_TOL`, R=5 mm | `<= 2.5e-7` | 4.2117e-08 | 4.2117e-08 | 5.94x | -- |
| max argument, R=45.9 mm | -- | **22.336 rad** | same | -- | "3.6e+02 rad" (16x) |
| max argument, R=5 mm | -- | **820.19 rad** | same | -- | "1.3e+04 rad" (16x) |
| control error, R=45.9 mm | -- | **9.8719e-07** | same | -- | "1.5e-05" (15x) |
| control error, R=5 mm | -- | **3.0523e-05** | same | -- | "4.9e-04" (16x) |
| control ratio, R=45.9 mm | `>= 10x` | **23.486x** | same | **2.35x** | "360x" |
| control ratio, R=5 mm | `>= 100x` | **727.04x** | same | 7.27x | "11 600x" |
| upsample crop rel L2 | `<= 1e-5` | 1.71512e-07 | 1.71512e-07 | 58.3x | "2.0e-07 at N=2048" (test runs N=512) |
| exact readout rel L2 | was `<= 1e-4`, now **1e-6** | **9.8995e-08** | **9.8841e-08** | 1010x -> **10.1x** | no number at all |
| one-group chain rel L2 | was `<= 1e-4`, now **1e-6** | **9.4427e-08** | **9.4427e-08** | 1059x -> **10.6x** | "2.8e-7/leg, 1.1e-6 after six" (a different chain) |
| band memory saving | `>= 6` grids | 12.4147 (22.627 -> 10.212) | 12.2902 (22.503 -> 10.213) | 2.07x / 2.05x | 22.6 / 10.2 -- reproduces |

Notes on the two bars that changed and on the one that did not:

* the two 1e-6 bars are two-sided: 10.1x and 10.6x above BOTH builds, and the
  build spread is 0.16 % and 4e-08 % respectively, so the bar cannot be reached
  by build drift but is reached by any real widening of the complex64 path;
* the 10x control-ratio bar keeps its 2.35x margin, and the docstring now SAYS
  so, together with the reason: the float32-argument control only separates
  above ~2e+03 rad (independent ladder: ratios 1.0, 1.0, 1.0, 5.8e+03,
  2.3e+04, 4.6e+04, 1.9e+05 over 2.0e+02 .. 2.0e+05 rad), and both shipped
  fixtures sit below that.  Tightening the FIXTURES would change what the test
  is about; stating the regime is the honest fix;
* `band_memory_saving_grids` moved by 0.0001 grids on Windows (12.414588 ->
  12.414717, ~270 bytes at N=512) -- the closure object D7 adds per banded
  call.  Against a 2.07x margin on a 6-grid bar this is noise, and it is the
  ONLY tracemalloc number this branch moves.

---

## 6. D7 -- the niche-C15 probe on the band path

`_imap_out['probe_rc'] = (rows, cols)` asks for the finalised `opl_map` (and
`ard_map`) at named pixels; it is C15's independent oracle for deciding which
INVERSION is faithful.  The block that fills it sits after the row-band
assembly's `return`, so a banded call filled nothing.  The band path has no
full-grid map to sample, so the same M pixels are now gathered band by band,
in probe order, from each band's final OPL and its ray-density amplitude, and
handed back under the same three keys.

`v18_c15_probe_on_band.py` (N=384, sub=4), re-run unchanged:

| rows | before | after |
|---|---|---|
| 0 | `probe_opl` present: `[-7.064440854777966e-05, -1.4704248941185464e-05, -5.333147337716495e-07]` | **identical** |
| 32 | `probe_opl` ABSENT (`engaged=True`, `n_out_of_domain=4125`) | **present, and equal to the whole-grid arm's to the bit** |

Test: `test_the_c15_probe_is_filled_on_the_band_path`, 3 routes
(screen+evaluator, ray-density+evaluator, ray-density+coarse-Newton), each at
band heights 32 and 7, asserting the banded values equal the whole-grid
values, that `probe_ard` and `probe_opl_piston` agree, and that the returned
FIELD is unmoved by asking (compared against the same call without
`probe_rc`).

---

## 7. Bit-identity ledger

Everything the verification hashed, re-measured on this tree AFTER all seven
changes:

| verification probe | what it hashes | result |
|---|---|---|
| `v1_without_identity` | 14 fixtures x `sag_chunk_rows` {0, None} -- the NON-banded traced calls, on field hash, `sum |E|^2`, `max |E|`, `engaged`, `gate_open`, `refused`, `n_out_of_domain` and the warning list | 28 cases, **224 fields compared, 0 mismatches** |
| `v2_banded_claim` (all four regimes) | the banded field hash + `sum |E|^2`, `engaged`, `gate_open`, `n_out_of_domain` and the full warning list at band heights 0 / 1 / 3 / 7 / 32 / 128 / N, on 9 fixtures x 2 inversion routes | **864 comparisons, 0 mismatches**, every band hash equal to the verification's record |
| `v14_c128_chain_identity` | 9 complex128 carrier-chain / readout / crop fixtures + every per-stage `dx`, `R` | **identical**, full JSON diff = wall-clock only |
| `v16_phasor_band_invariance` | all four phase helpers at 16 / 61 / 1024 rows per band against a whole-grid `np.exp(float64).astype(complex64)` | **identical JSON** to the verification's record |
| `v5_c64_ladder` | the complex64 phasor error and its float32-argument control over 2.0e+02 .. 2.0e+05 rad, all four helpers, 7 rungs | **identical JSON** to the verification's record |
| `v3_determinism` (D14 / D15) | one field hash per case at `OPENBLAS/OMP/MKL/NUMEXPR_NUM_THREADS` 1 / 2 / 4, each in its own subprocess, on 6 cases including the banded ray-density + evaluator route the D6 fix touches | identical at every thread count AND equal to the verification's record (`eecefe42276fe6b7b4c67d2a`, `6bd918bc340591460ca8c503`, `4da384eec1ee92020afb1efe`, `203aee7468521ed5bbee545d`) |
| `v10_s10_route` | the CHANGELOG's named S10 fixture, banded vs whole-grid at the shipped default | `f7680b0c2ef005db9605beb2` (carrier) and `0caa26d97c50bad02e361743` (`carrier=None`), banded == whole-grid, rel 0.0000e+00 -- the verification's recorded values |
| `v6_c64_chain` | the complex64 two-group chain's rel L2 / rel power, the six-leg ladder, the crop's double rounding | **identical to 16 digits**; the only change is the upcast list, 5 -> 0 |
| `v7_c64_memory` | whole-chain peak and every complex128 phasor return by caller frame | complex128 arm identical (376.9 MiB, 10 returns); complex64 arm 5 -> 0 returns |
| `v9_durability` | every constant the two new test files assert | identical on both builds except the +270-byte tracemalloc row above |
| `v18_c15_probe_on_band` | the whole-grid probe values | identical; the banded arm goes absent -> equal |
| `p2_d1_attr` | the field hash on 3 band heights with all self-checks firing | `a78a7ff1533ade9e24e9f84a` on all six (3 arms x before/after) |
| `p3_d6_time` | the field hash of 12 traced calls, two builds | banded == whole-grid on the evaluator (`91da5e59c46beb1d967b28b3` screen, `08f9aa00b567ed87658cf8ca` ray-density at N=4096); 5.44.0's forced-incumbent control == v5.43.0's banded answer (`66b9384477cf8a9ca1844c6a`) |
| `p1_d2_transient` | 16 carrier-helper calls | identical before / after on both dtypes |

---

## 8. Tests, lint, durations

### 8a. What changed in the tests

| file | change |
|---|---|
| `tests/unit/test_mixed_precision_carrier_helpers.py` | +10 cases (D2), and the D4/D5 docstring / bar corrections |
| `tests/unit/test_banded_ray_density_and_inverse_map.py` | +3 cases (D1) and +3 cases (D7) |
| `tests/unit/test_niche_perf_round2_2026_08_10.py` | docstring only (D3) |

No test was loosened.  Two bars were TIGHTENED (1e-4 -> 1e-6, D5); one new bar
was added with a derivation and a two-build measurement (the D2 tracemalloc
gap); the rest of the changes are recorded numbers and stated margins.

### 8b. The run

<!-- TESTS-RUN -->

### 8c. Lint

`ruff check lumenairy/ tests/` -> **All checks passed!** (`validation/` is
`extend-exclude`d by `pyproject.toml`, as for every other probe directory.)

### 8d. `.test_durations`

pytest-split's own store, one quiet single-threaded run per changed file
(`--store-durations --clean-durations --durations-path <scratch>/<file>.json`,
`OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS=1`), spliced line-based per
CHORE_TEST_HYGIENE_2026_08_16 (d)/(e):

```
measured 35 node ids over 2 files
retained 12552, removed 19, added 35 -> 12587 entries
re-parsed 12587 entries; 12552 retained lines byte-identical
```

35 entries, 30.47 s in total; formatting unchanged (2-space indent, sorted
keys, CRLF in the working tree / LF in the blob).

---

## 9. Open items

Recorded, not fixed:

1. **The evaluator route's dominant cost is untouched.**
   `InverseCharacteristic.domain_mask` -- the screened landing-hull test plus
   the entrance-radius test -- is 8.9-10.0 s of a ~20 s call at N=4096, i.e.
   ~45 % of every whole-grid evaluator call, banded or not.  It is not banding's
   and not this branch's, but it is where the traced element's time goes at
   production N, and nothing has profiled it below the `hull_mask_grid` level
   (the ring reduction against the screened interior).  A separate item.
2. **The banded ray-density branch still evaluates `CH_X_IN` / `CH_Y_IN`
   twice** -- 0.84 s at N=4096, the residual 7/4.  Caching them would cost TWO
   full float64 grids (2.1 GB at N=16384), which is precisely what the band
   assembly exists to avoid, so it is left as it is.  The bool mask was worth
   caching at 1/8 of one grid; these are not.
3. **The D6 wall times were taken on a box ~80 % busy** with three other
   agents' work.  Ratios and pixel counts are load-free and are what the
   conclusions rest on; the verification's idle-box absolute seconds (V15)
   were not re-taken and remain the reference for those.
4. **Two of the three ray-density self-checks could not be made to fire on the
   shipped test fixture.**  The energy check fires and is pinned in
   `test_banded_ray_density_and_inverse_map.py`; the halo and support-band
   checks need a different exit-support geometry, and their attribution is
   measured only in `p2_d1_attr.py` (which does fire the support-band one on
   both routes).  All three sit in one closure at one frame depth, so the
   pinned one covers the contract, but the test is one fixture short of
   pinning all three directly.
5. **`_PHASOR_BAND_BYTES = 32e6` is still an untested magic constant** (the
   verification's own note).  `_narrow_rows` inherits it.  Its VALUE-inertness
   is pinned for `_phasor_rows` (V16: identical hashes at 16 / 61 / 1024 rows
   per band); the new sibling's band size is inert for the same reason -- an
   elementwise product on a row slice -- but that is argued, not measured at
   several band sizes.
6. **A plain traced carrier chain never calls `_fourier_upsample_crop`**
   (measured: 0 calls).  The crop is reached from
   `carrier_referenced_exact_focus_readout` and `_fine_trace_group_exit`, i.e.
   only under `final_leg='exact'` or a fine retrace.  D3's decision therefore
   rests on the exact-readout chain, which is the production shape for a
   high-NA final leg; a `final_leg='paraxial'` chain never sees the
   single-precision transform at all.
7. Everything in the verification's own section 12 ("what could not be
   verified") is unchanged: the prototypes' figures, the design-121 `.zmx`
   arms, and the GPU / JAX arms of either path.
