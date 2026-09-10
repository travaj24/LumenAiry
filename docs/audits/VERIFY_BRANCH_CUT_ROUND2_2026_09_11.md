# Independent verification: the modal branch cut, round 2

Verifier's report on `docs/audits/FIX_BRANCH_CUT_ROUND2_2026_09_11.md` (the
round-2 fix) in the context of `FIX_RCWA_EVEN_SECTOR_WSL_2026_09_11.md`
(round 1) and `VERIFY_RCWA_EVEN_SECTOR_2026_09_11.md` (round 1's verification).
Every number below was **re-measured on fixtures built here**, on both builds;
none is read from the fix document.

Worktree `verify/branch-cut-round2` at `C:/tmp/lum_vbc2`, cut from `664a84b`
(the `wave2/pmm2d` tip carrying round 2).  The PRE tree is `2898767` (round 1 +
its verification, without round 2), mounted detached at `C:/tmp/lum_vbc2_pre`
and removed at the end; probes were run from BOTH trees unchanged, so for the
PMM claims the ARM IS THE TREE rather than an in-process patch.  For the RCWA
claims (X-1, the band) both trees carry round 1, so there the arm is
ENGINEERED: the pre-ROUND-1 body reinstalled at every module binding.

Builds: **WIN** = Windows 11, python 3.14.6, numpy 2.4.4, scipy-openblas
dispatching Haswell.  **WSL** = Ubuntu on the same box, python 3.12.3,
numpy 2.4.6, SkylakeX.  `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and
`MKL_NUM_THREADS` are on the command line of every run; where a thread count is
the variable, the ladder 1 / 2 / 4 / 8 was run and is reported.

Probes and JSON: `validation/probe_verify_branch_cut_round2/`.

---

## 1. Task 0 -- the merged-tree gate, both builds

Pinned (`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`),
`-p no:randomly`, the five files the brief names:

```
WIN  pytest tests/unit/test_fix_rcwa_even_sector_wsl.py
            tests/unit/test_verify_rcwa_even_sector.py
            tests/unit/test_fix_branch_cut_round2.py
            tests/unit/test_m1_conditioning_guard.py
            tests/unit/test_v5_14_2_backlog_batch.py
     86 passed, 4 warnings in 92.35s (0:01:32)

WSL  the same five files
     86 passed, 2 warnings in 95.38s (0:01:35)
     ** On entry to DLASCL parameter number  4 had an illegal value      (x2)
```

The two `DLASCL` lines are the deliberate NaN matrix of
`test_guarded_lstsq_stands_aside_on_a_non_finite_system`, exactly as round 1's
verification section 10 attributes them; they are `zgelsd`'s Fortran unit-6
output, not this solve's.

The JAX-guarded PMM / RCWA / Berreman files on Windows, pinned:

```
WIN  pytest tests/unit/test_audit_w3_pmm_jax_guards.py
            tests/unit/test_v5_14_2_jax_stacks.py
            tests/unit/test_v5_20_2_pmm_jones_2d_jax.py
            tests/unit/test_v5_20_1_rcwa_2d_oop_jax.py
            tests/unit/test_v5_20_3_rcwa_1d_oop_jax.py
            tests/unit/test_v5_14_5_emt_and_berreman_jax.py
            tests/unit/test_v5_20_10_berreman_internal_jax.py
     71 passed, 7 warnings in 148.33s (0:02:28)
```

**No failure in the gate.  Nothing here is a first finding.**

---

## 2. Verdict table

| # | claim | verdict | my numbers |
|---|---|---|---|
| 1a | ONE `_sqrt_decay`, five copies deleted, six call sites routed | **CONFIRMED** | AST: 1 definition (`rcwa/_core.py:1284`, args `x, xp, band`), 23 call sites, 8 `from`-imports, 6 module-level bindings.  PRE tree: 6 definitions. |
| 1b | refactor bit-identical to round 1 | **CONFIRMED** | `max abs diff = 0`, bit-identical `True`, over 4,037 engineered values at 8 array sizes plus 4 dominated / sub-unit spectra, BOTH builds.  `min Re(lam) = 0.0`, 0 roots with `Re < 0`. |
| 1c | the JAX twins trace it | **CONFIRMED, and widened** | Under `jax.jit`: explicit-`jnp` vs NumPy max **2.84e-16** relative over 4,015 values, **0** branch disagreements, `jax.grad` identical to all digits (851406297.1649544, both builds).  There is a **FOURTH** JAX caller that does NOT pass `jnp` -- `elements/_berreman_jax.py:396` -- which works only because `array_namespace` recognises a Tracer; the auto-detecting call is BITWISE identical to the explicit one under `jit`.  Twins vs NumPy on a coincidence fixture: 1.3e-15 / 1.3e-14 / 5.6e-17. |
| 2a | the hybrid PMM's answer was WRONG | **CONFIRMED, and larger** | 5 coincident-spacer stacks of my own.  Worst PRE per-order distance to an independent `RCWAStack`: **7.4844e+00** (`S5`), 2.5890e-01 (`S2`), 1.0690e-03 (`S4`, conical) -- against the fix document's 2.0035e-03.  POST: 5.5e-10 at normal incidence, 4.3e-07 at oblique (= that mount's own hybrid Fourier floor). |
| 2b | `sum R + T` up to 110 pre-fix; 2.000000000 post | **CONFIRMED** | Worst PRE `sum R + T` over 6 (build, thread) samples: 2.8603 (WIN 1), 3.1608 (WIN 2), 2.1609 (WIN 4), **17.707** (WIN 8), **28.945** (WSL 1), 2.8508 (WSL 4).  POST: **2.000000000** on all six, to nine decimals. |
| 2c | `cond(a+b)` 1.96e9 -> 14.5 | **BOUNDED** | PRE 5.25e8 .. 1.56e10, POST **2.58e1 .. 9.97e7** -- fixture-dependent.  On 3 of my 5 spacer fixtures the post-fix PMM interface still reads 1.5e7 .. 1.0e8, against 3.8e2 on the no-spacer control.  A detune ladder shows it is the coincidence (`cond` falls exactly as 1/detune: 5.9e7 / 9.5e5 / 9.8e4 / 9.8e3 / 9.6e2 at d = 0 / 1e-4 / 1e-3 / 1e-2 / 1e-1) but it is a NEAR-DEGENERACY, not the singularity: closure is FLAT at 1.63e-10 across the whole ladder.  The fix removes the wrong answer; it does not universally restore conditioning. |
| 2d | a relative 1e-6 detune returns silently wrong; only 1e-3 cures | **CONFIRMED / RESTATED** | 1e-6 does NOT cure: PRE closure 7.90e-05, per-order 2.52e-04.  The cure point on MY fixture is between 3e-6 (1.98e-07) and **1e-5** (7.30e-10), not 1e-3.  There is no fixture-independent cure detune, which is the reason D6's correction is right. |
| 2e | no-spacer control identical pre/post | **CONFIRMED at the closure level, RESTATED bitwise** | Closure identical to all printed digits on all 12 samples (1.0074e-10 / 2.0202e-10 / 8.4546e-11); per-order motion is **1.7e-15 .. 3.1e-15**, not zero. |
| 2f | PURE STAGGERED and 1-D PMM make ZERO calls and are bit-identical | **CONFIRMED** | 33 converse fixtures (pure stack shared / per-layer grids / per-layer walls / tensor / conical / region-both / eps 4 / strong modulation; staggered cell scalar and Jones, conical; 1-D `PMMStack` and `pmm_efficiency_1d` at M = 6..28, TE and TM): **0 `_sqrt_decay` calls on either arm**, **33 of 33 bit-identical**, worst motion **0.0000e+00**.  Closures 4.4e-16 .. 3.3e-13 (per-layer mortar 4.5e-12, strong modulation 2.2e-09 = its Fourier floor).  Cross-build spread WIN vs WSL 2.0e-12. |
| 2g | the LOSSY census exemption | **REFUTED as stated** | The discriminator is loss in the **PATTERNED** layer, not loss in the stack.  Lossy CELL (1e-2 or 1e-6, normal and conical, spacer lossy or not): **0.0000e+00**, bit-identical.  Lossy SPACER with a LOSSLESS cell: **NOT** bit-identical -- 2.0e-13 at `Im = 1e-2` and **1.3353e-03 per order at `Im = 1e-6`** (PRE closure +3.57e-04).  Counted on the eigenproblem: the acted-on layer population is 0 with a lossy cell and 4..8 with a lossy spacer alone, on both builds.  The task's specific requirement -- "the hybrid at oblique with a lossy spacer must be bit-identical pre/post" -- **is not met**. |
| 2h | the defect is visible only in a modulation window around 1e-6 | **REFUTED** | PRE closure at relative modulation **1e-10**: 3.29e-05 (WIN, `n_orders` 4) -- the LOUDEST rung of my ladder; worst pre/floor over `n_orders` 3/4/5: 4.98e+10 (WIN) / 1.92e+11 (WSL) at 1e-10 and 1.05e+11 / 6.46e+10 at 1e-8.  "Below 1e-08 the layer is numerically uniform ... where the old pin was always correct" does not hold on my fixture. |
| 3a | ARRAY-MAX has the larger gap on all three populations, both builds | **REFUTED as stated** | RCWA ordinary: AM **13.88** (WIN) / **14.01** (WSL) against PER-MODE **14.13 / 14.26** -- per-mode WINS.  Hybrid PMM: AM 7.94 / 7.93 against PM 7.80 / 7.79 -- array-max wins narrowly.  Layer cutoff: AM **-0.29** (WIN) / **+0.01** (WSL), PM the same -- both shapes' populations OVERLAP.  The DECISION (keep array-max) is still defensible; the stated basis for it does not reproduce. |
| 3b | margins 6.19/6.10, 6.75/6.46, 4.04/3.34 decades | **REFUTED as stated** | RCWA ordinary: **7.34 / 6.53** (WIN), 7.47 / 6.53 (WSL).  Hybrid PMM: 6.94 above the noise side but only **1.00 decade** below the signal side (nearest signal-side ratio 1.0036e-07, both builds) -- not 6.46.  Layer cutoff, driven to `min\|lam^2\| = 3.1722e-16`: the bar sits 0.41 decades above the noise side and **BELOW** the signal side (1.9954e-09 on WIN), i.e. a signal-side mode is conjugated. |
| 3c | "no wrong answer follows from either choice on any fixture measured" | **CONFIRMED, sharpened** | 18 deepest mounts x 2 builds x 4 branch rules (shipped / band 0 / per-mode / pre-round-1 pin): every mount on which the shipped rule returns **SILENTLY** moves by at most **8.75e-11** per order between rules; every mount where they diverge materially (up to **9.5104e-02**) is LOUD.  And the band is LOAD-BEARING: with `band = 0` the solve RAISES on **16 of 18** mounts. |
| 3d | deeper cutoff than 6.5341e-10 unmeasured | **CLOSED here** | `min \|lam^2\|` driven to **3.1722e-16** (WIN) / 4.0185e-16 (WSL) by golden-section over the ridge index -- 6.3 decades below the fix document's ladder and 1.2 below round 1's verification.  Worst `\|Re r\|/\|r\|` conjugated: **0.9976** (WIN) / **0.99993** (WSL) at `\|r\| = 4.7e-08` / 4.1e-08, three decades past the verification's 2.0751e-03. |
| 4 | X-1 closed: 7-8 / 14 / 152.60x -> 0 / 0 / <= 1.4e-15 | **CONFIRMED to every printed digit** | TE PRE **7** raises (WIN) / **8** (WSL), **14** flagged both, worst closure **3.1956e-02** both, min equilibrated `rcond` **3.331e-19** (WIN) / **3.614e-19** (WSL).  TM PRE 5 / 9 / **2.6165e-04** (WIN), 5 / 9 / **1.5404e-04** (WSL).  POST **0 / 0** with worst closure **1.3323e-15** (WIN) / **1.4433e-15** (WSL) and min `rcond` 6.250e-02 / 5.603e-02.  Pinned cells, `sum(R)`: M=12 2.016454e-04 -> 2.015824e-04; M=19 1.838764e-02 (WIN) / RAISED (WSL) -> 2.053766e-04 on both; M=20 2.088570e-04 / 2.262156e-04 -> 2.053491e-04; M=21 3.216567e-02 -> 2.095174e-04.  Every one of those matches the fix document exactly. |
| 4b | the restated pinning tests fail on the PRE tree | **RESTATED** | On the pre-round-2 tree the ROUND-2 file fails 11 of 17 (`4 uniform-spacer / manufactured-energy / reference / incoming-root / JAX-twin / consolidation` gates) -- a clean file-level fail-before, both builds.  But `test_m1_conditioning_guard.py` reads **28 passed, 0 skipped on the PRE tree too**: the restated X-1 tests are ROUND-1 assertions.  They are correct and worth having; they do not discriminate round 2. |
| 5a | M1 motivating calls 7/110 -> 0/110 | **CONFIRMED for the POST half** | My 26-fixture sweep: **0 of 106** motivating post-fix on both builds; PRE (engineered) 1 of 106, max raw residual 4.525e+00 (WIN) / 5.934e+00 (WSL), max raw/equilibrated ratio 3.280e+05 / 1.587e+05. |
| 5b | "post-fix no call where the two instruments disagree by more than 2.2x" | **REFUTED** | Max raw/equilibrated ratio POST on my sweep: **4.737e+02** (WIN), 4.530e+02 (WSL).  Both instruments are far below the 1e-8 bar, so no call is motivating and nothing is wrong -- but 2.2x is sample-scoped by two and a half decades. |
| 5c | the instrument is KEPT, not dead | **CONFIRMED** | Synthetic scaled operands (`cond` 1e19 .. 1e31 from row/column scaling alone): raw residual 3.86e-08 / 1.26e-03 / 1.14e-01, all above the 1e-8 refusal bar, equilibrated residual **8.1e-16 .. 8.4e-16** -- rescued, every time.  Both halves of the instrument execute: `_rcond_1_equilibrated` on the scaled operand (rcond_raw 1.4e-27 vs rcond_eq 6.7e-03, so the free screen correctly does NOT fire) and `_equilibrated_inverse_residual` on a genuinely near-singular one (rcond_eq 2.85e-16 < the 1e-8 screen, residual 2.29e-03 computed). |
| 6 | census: lossy 11/11, off-coincidence <= 6.6e-13, on-coincidence up to 1.26e-04 | **CONFIRMED except the lossy class** | Off-coincidence on my fixtures: 1.7e-15 .. 3.1e-15.  On-coincidence: up to 7.48e+00 per order.  Lossy: see 2g. |

---

## 3. Task 1 -- consolidation, in detail

### 3.1 The AST census (`v1_consolidation.py`)

Exactly **one** definition in the whole of `lumenairy/elements`:
`rcwa/_core.py:1284`, signature `(x, xp, band)`.  On the PRE tree there are
**six** (`pmm/twod.py:411`, `pmm/twod_staggered.py:2286`,
`pmm/_jax_stack2d.py:175`, `pmm/_jax_twod.py:341`,
`pmm/_jax_twod_jones.py:191`, `rcwa/_core.py:1224`).

The five former copy sites and what they became:

| module | calls now | `xp` |
|---|---|---|
| `pmm/twod.py` | 3 (lines 668, 755, 778) | auto-detect |
| `pmm/_jax_stack2d.py` | 2 (192, 210) | `jnp` |
| `pmm/_jax_twod.py` | 2 (374, 402) | `jnp` |
| `pmm/_jax_twod_jones.py` | 1 (208) | `jnp` |
| `pmm/twod_staggered.py` | 0 -- the dead copy is gone, and nothing was routed in its place | -- |

Plus the sites that always used the shared body: `rcwa/_core.py` (8),
`rcwa/oned.py` (4), `rcwa/stack.py` (1), `elements/berreman.py` (1),
`elements/_berreman_jax.py` (1).  23 call sites, 8 `from`-imports, 6
module-level bindings.

The "no exact-zero branch pin anywhere" claim survives scrutiny: the twelve
`Im(...) != 0.0` comparisons my AST pass finds are all LOSSINESS tests on USER
INPUT (`eps_sup`, an `eps_cell`, a segment spec), where an exact zero is exact
by construction, not branch decisions on an `eig` output.

### 3.2 Bit-identity, on my own value sets

4,037 values: 4,000 pseudo-random over fifteen decades plus signed zeros, both
denormals, `lam^2` exactly on the cut, `nan`, `+-inf`, `1e300 + 1e300j`,
`1e-300`, and three values placed exactly AT the band on a unit spectrum.  Eight
array sizes (1, 2, 3, 7, 64, 243, 1024, 4037), plus four spectra whose `max|r|`
is dominated by ONE evanescent mode (`1e8`, `1e16`, `1e24`) and one whose top is
below 1 so the `1.0` floor engages.

`bit_identical = True`, `max |diff| = 0.000e+00`, on **both builds**, against a
transcription of round 1's body read out of `2898767`.  `min Re(lam) = 0.0` and
zero roots with `Re < 0` over the whole set, so the `|X| <= 1` contraction holds.

**One robustness observation, shared with round 1 and therefore not a round-2
regression.**  The scale is `max(max|r|, 1)` over the WHOLE array, so a single
non-finite root decides the verdict on every other mode of that layer.  A `NaN`
anywhere makes `scale = nan`, `on_cut` False everywhere, and the pin **silently
disarms for the whole layer**: 14 conjugations become 0 on a 24-mode synthetic
spectrum.  `inf` and `1e300` do not do this.  I found no solve path that reaches
it (`eig` fails first on the fixtures tried), so this is INFO, not a defect.

### 3.3 The JAX twins

Under `jax.jit`, over 4,015 finite non-denormal values (XLA flushes subnormals
and numpy's own `sqrt(0 + 5e-324j)` overflows to `inf`, so those are reported
separately rather than mixed into a parity claim):

| | WIN (jax 0.11.0) | WSL (jax 0.10.2) |
|---|---|---|
| explicit-`jnp` vs NumPy, max relative | 2.8434e-16 | 2.8434e-16 |
| branch decisions that differ | 0 | 0 |
| auto-detect (`xp=None`) vs explicit, bitwise | identical | identical |
| `jax.grad` through the body | 851406297.1649544 | 851406297.1649544 |

Twin public surfaces, traced vs NumPy on the coincidence fixture: 1.33e-15
(`pmm_efficiency_2d_cell`), 1.27e-14 (`pmm_jones_2d`), 5.6e-17
(`PMM2DStackHybrid`); `jax.grad` through the traced twin -0.006314087369544095
(WIN) / -0.006314087369543981 (WSL).

---

## 4. Task 2 -- the hybrid reproducer, on my own fixtures

### 4.1 The fixtures

Five coincident-spacer `PMM2DStackHybrid` stacks, none of them the fix
document's: `S1` an 8x8 cell of `eps = 2.25` with a 3x2 block at
`2.25(1 + 1e-6)`, periods 0.62 / 0.58 um, spacers 0.12 / 0.09 um, `n_sub`
1.63; `S2` a 6x6 L-shaped cell at `eps = 4.0`, `rel = 2e-6`, period 0.44 um,
`n_sub` 1.71; `S3` `S1` at theta = 12 deg; `S4` `S1` at theta = 15 deg,
phi = 33 deg (conical); `S5` a 10x10 cell at `eps = 2.89`, `rel = 5e-7`,
asymmetric spacers, `n_sub` 1.42.  Two no-spacer controls and two lossy
fixtures.  All at `n_orders` 3 / 4 / 5.

Reference: an independent `RCWAStack` solve of the SAME device with the cell
block-replicated 8x, with its own convergence reported (`n_orders` 4 / 6 / 8
and replication 4 / 8 / 12).

### 4.2 The reading, over 12 (tree, build, thread) samples

Worst `|sum R + T - 2|` per sample, spacer fixtures against controls:

| sample | worst spacer `sum R + T` | worst control |
|---|---|---|
| PRE WIN 1 | **2.8603121** | 2.0202e-10 |
| PRE WIN 2 | **3.1608437** | 2.0202e-10 |
| PRE WIN 4 | **2.1609307** | 2.0202e-10 |
| PRE WIN 8 | **17.7071027** | 2.0202e-10 |
| PRE WSL 1 | **28.9447995** | 2.0202e-10 |
| PRE WSL 4 | **2.8507918** | 2.0202e-10 |
| POST, all six | **2.0000004** | 2.0202e-10 |

The PRE spread is four and a half decades with the BLAS thread count and the
build, on a conservation law; the POST column is identical to all printed digits
on every sample, and the 4e-07 in it is the oblique mounts' own hybrid Fourier
truncation error (the RCWA reference reads the same there).  The control is
arm-, build- AND thread-independent at the closure level -- 2.0202e-10 on all
twelve.

Worst per-order distance to the independent reference:

| fixture | PRE worst | POST worst |
|---|---|---|
| `S1_spacer_8px` | 8.3116e-04 | 5.5300e-10 |
| `S2_spacer_eps4` | **2.5890e-01** | 2.0850e-13 |
| `S3_spacer_oblique` | 5.2349e-04 | 4.0462e-07 |
| `S4_spacer_conical` | 1.0690e-03 | 4.2847e-07 |
| `S5_spacer_10px` | **7.4844e+00** | 2.9283e-10 |

### 4.3 The detune and modulation ladders

Spacer detuned by a relative `d`, everything else fixed (`S1`, `n_orders` 4,
WIN 1 thread; PRE closure / PRE per-order vs the reference):

| `d` | 0 | 1e-8 | 1e-7 | **1e-6** | 3e-6 | **1e-5** | 1e-4 | 1e-3 |
|---|---|---|---|---|---|---|---|---|
| PRE closure | 1.00e-05 | 1.17e-04 | 6.25e-05 | **7.90e-05** | 1.98e-07 | **7.30e-10** | 1.69e-10 | 1.63e-10 |
| PRE vs ref | 2.67e-05 | 5.32e-04 | 2.91e-04 | **2.52e-04** | 8.17e-07 | 1.90e-09 | 5.44e-10 | 5.53e-10 |

So the fix document's central point holds -- a relative 1e-6 detune leaves the
answer silently wrong -- but its "only 1e-3 cures it" is one fixture's: mine is
cured at 1e-5.  There is no fixture-independent cure detune, and the POST
closure is flat at 1.63e-10 across the whole ladder, which is the durable form
of the claim (gate 3 of the new test file).

Modulation ladder, PRE closure at `n_orders` 4 (WIN 1 thread):

| relative pillar-host | 1e-10 | 1e-8 | 1e-7 | 1e-6 | 1e-5 | 1e-4 |
|---|---|---|---|---|---|---|
| PRE, spacer | **3.29e-05** | -1.29e-07 | -1.52e-08 | 1.00e-05 | 3.69e-07 | 1.14e-07 |
| PRE, no spacer | 1.90e-10 | 1.90e-10 | 1.91e-10 | 2.02e-10 | 3.12e-10 | 1.48e-09 |

`1e-10` is not "a window around 1e-06", and it is the loudest rung.  Scanning
`n_orders` 3/4/5 as well (because which truncation manifests is per-build), the
worst PRE/floor ratio is 4.98e+10 (WIN) / 1.92e+11 (WSL) at 1e-10 and
1.05e+11 / 6.46e+10 at 1e-8.

### 4.4 The converse: the pure staggered engine and the 1-D PMM

I tried to break them: `PMM2DStackPure` with a uniform spacer of exactly the
patterned layer's background, on SHARED and PER-LAYER grids, with explicit
per-layer walls, in-plane scalar and with a `mu` cell, at normal and conical
incidence, with the region coincidence added (`n_sub = n_sup = 1.5`), at
`eps = 4`, and at strong modulation; `pmm_efficiency_2d_staggered` and
`pmm_jones_2d_staggered` on and off the region coincidence; and the 1-D
`PMMStack` in the X-1 shape ONE LEVEL UP -- a uniform spacer of exactly the
groove index with BOTH half-spaces also at the groove index -- at
`n_orders` 6 / 11 / 19 / 21 / 28, plus `pmm_efficiency_1d` TE and TM at the
same rungs.

**All 33 are immune.**  `_sqrt_decay` call count: **0** on every one, on BOTH
arms (`_forward_branch_flip` 3-5 calls each, `_select_forward_flux` 0).
Pre-vs-post: **33 of 33 bit-identical, worst motion 0.0000e+00.**  Closures:

| surface | WIN | WSL |
|---|---|---|
| `pure_spacer_shared` | 2.66e-15 | 7.11e-15 |
| `pure_spacer_perlayer` / `perlayerwalls` | 4.49e-12 | 6.45e-12 |
| `pure_spacer_region_both` | -2.89e-15 | 3.11e-15 |
| `pure_spacer_conical` | 1.60e-14 | 2.44e-14 |
| `pure_spacer_tensor` | 3.11e-15 | 7.11e-15 |
| `pure_nospacer_ctrl` | 4.44e-16 | 4.44e-15 |
| `pure_spacer_strong` (rel 1e-1) | -2.17e-09 | -2.17e-09 |
| 1-D `PMMStack` spacer, M = 6..28 | 2.66e-15 (every rung) | -- |
| `pmm_efficiency_1d` TE/TM, M = 6..28 | 4.4e-16 .. 4.7e-15 | -- |

The largest cross-build per-order spread over the whole converse set is
**2.0142e-12**; the largest closure on either build is 2.1665e-09 (the strong
modulation rung's own Fourier floor).  The 1-D coincident-spacer stack -- the
geometry whose RCWA sibling manufactured 3.2e-02 pre-round-1 -- closes at
2.66e-15 at every truncation.

### 4.5 The lossy claim: the discriminator is the PATTERNED layer

`v7_followups.py`, PRE tree against POST tree, same fixture, both mounts:

| fixture | PRE closure | POST closure | max per-order motion | bit-identical |
|---|---|---|---|---|
| lossless (control) | 1.0031e-05 | 1.6299e-10 | 5.02e-05 | no |
| lossy CELL 1e-2, normal | -3.536518e-02 | -3.536518e-02 | **0.0000e+00** | **yes** |
| lossy CELL 1e-2, conical | -3.570996e-02 | -3.570996e-02 | **0.0000e+00** | **yes** |
| lossy CELL 1e-6, normal | -3.572619e-06 | -3.572619e-06 | **0.0000e+00** | **yes** |
| lossy CELL 1e-6, conical | -3.292707e-06 | -3.292707e-06 | **0.0000e+00** | **yes** |
| lossy CELL + lossy SPACER, both mounts | -6.646669e-02 / -6.701592e-02 | same | **0.0000e+00** | **yes** |
| **lossy SPACER only, 1e-2, normal / conical** | -3.164495e-02 / -3.186305e-02 | same | 2.03e-13 / 5.52e-13 | **no** |
| **lossy SPACER only, 1e-6, normal / conical** | +3.568750e-04 / +1.704007e-05 | -3.20e-06 / -2.90e-06 | **1.3353e-03** / 9.5330e-05 | **no** |

The mechanism, counted on the eigenproblem so the count is arm-independent
(`v10_acted_population.py`, `n_orders` 3/4/5):

| fixture | acted-on LAYER modes, WIN | WSL |
|---|---|---|
| fully lossless | 5 / 6 / 7 | 8 / 4 / 4 |
| lossy CELL (1e-2 or 1e-6) | **0 / 0 / 0** | **0 / 0 / 0** |
| lossy CELL + lossy SPACER | **0 / 0 / 0** | **0 / 0 / 0** |
| lossy SPACER only (1e-2 or 1e-6) | 5 / 6 / 7 | 8 / 4 / 4 |
| lossy SPACER only, conical | 6 / 6 / 6 | 6 / 6 / 6 |

A lossy spacer does not empty the acted-on population, because the PATTERNED
layer's propagating modes are still exactly on the cut.  The round-2 file's
gate 6 is sound (its cell IS lossy), but the SCOPE STATEMENT the census and the
fix document carry -- "where the sign is physics, nothing moves" -- reads as
"loss anywhere is safe" and is false by three decades.

---

## 5. Task 3 -- the band scale, on my own populations

`v4_band.py`, taps `_sqrt_decay`'s RAW input, so every ratio is a property of
the eigenproblem and identical on both arms.  Classification is made on
`lam^2` AND on the fixture: NOISE = a mode of a provably LOSSLESS fixture with
`Re(lam^2) < 0` and `|Im(lam^2)| <= 1e3 eps_mach max|lam^2|`; SIGNAL =
everything else in the acted-on population.  Region arrays (`Im` exactly zero)
are excluded.

| population | build | AM noise max | AM signal min | AM gap | PM gap |
|---|---|---|---|---|---|
| RCWA ordinary (72 fixtures, 668 / 691 acted-on) | WIN | 4.5442e-16 | 3.4235e-02 | **13.88** | **14.13** |
| | WSL | 3.3802e-16 | 3.4235e-02 | **14.01** | **14.26** |
| hybrid PMM `P@Q` (24 fixtures, 1818 / 1711) | WIN | 1.1482e-15 | 1.0036e-07 | **7.94** | 7.80 |
| | WSL | 1.1742e-15 | 1.0036e-07 | **7.93** | 7.79 |
| RCWA layer cutoff (54 mounts, 1998 modes) | WIN | 3.9195e-09 | 1.9954e-09 | **-0.29** | -0.29 |
| | WSL | 3.9195e-09 | 4.0244e-09 | **+0.01** | +0.01 |

Three things this changes.

* **Per-mode is not uniformly worse.**  On the ordinary RCWA population it has
  the LARGER gap on both builds (14.13 / 14.26 against 13.88 / 14.01).  The fix
  document's "ARRAY-MAX has the larger two-sided gap on all three populations
  and on both builds" does not reproduce.
* **The hybrid PMM's signal-side margin is 1.00 decade, not 6.46.**  The nearest
  signal-side mode reads 1.0036e-07 on both builds -- ten times the bar -- and it
  is an ordinary healthy mode (`|lam^2| = 1.796`, `|r| = 1.340`).  It is one of
  the 50 genuinely COMPLEX modes a lossless hybrid cell carries; counting those
  as noise instead (the looser convention) puts the "noise side" at 3.0918e-04,
  i.e. **4.49 decades ABOVE the shipped bar**.  Which convention one picks
  changes the answer by six decades, which is the real content of the fix
  document's own section 5.5 warning, one population over.
* **At my deepest cutoff the two populations overlap, on both shapes.**  The
  band ACTS ON a signal-side mode: worst `|Re r| / |r|` conjugated is
  **0.9976** (WIN) / **0.99993** (WSL), at `|r| = 4.66e-08` with
  `|Im r| = 3.26e-09`.  The verification's D4 reading of 2.0751e-03 is three
  decades short of what a deeper ladder finds.

### 5.1 Is the corner harmless?  Measured, not argued

`v6_cutoff_consequence.py`: 18 mounts per build, each solved under FOUR branch
rules (shipped / `band = 0` / per-mode / the pre-round-1 exact-zero pin).

| | WIN | WSL |
|---|---|---|
| mounts where the SHIPPED rule returns SILENTLY | 11 of 18 | 11 of 18 |
| worst per-order motion between rules on those | **8.75e-11** | **8.75e-11** |
| worst per-order motion on the LOUD mounts | 9.5104e-02 | 9.5104e-02 |
| mounts where `band = 0` RAISES `_EnergyError` | **16 of 18** | 16 of 18 |

So the fix document's "no wrong answer follows" is right, and the sharper
statement it supports is: **the cutoff corner is never SILENTLY wrong, and the
band is load-bearing.**  Both halves are pinned by gate 4 of the new test file.

**Closest approaches.**  A lossless propagating mode gets within 0.41 decades of
the bar (band ratio 3.9195e-09, `|lam^2| = 1.21e-15`), on both builds.  A
lossy / evanescent (signal-side) mode gets to 1.9954e-09 on WIN -- i.e. INSIDE
the band -- and 4.0244e-09 on WSL.  Both are cutoff modes whose own magnitude
has collapsed to ~2e-08.

---

## 6. Task 6 -- test durability

Every constant in `tests/unit/test_fix_branch_cut_round2.py` and the restated
X-1 test, re-measured on ITS OWN fixture over six (build, thread) samples
(`v8_durability.py`; WIN 1/2/4/8, WSL 1/4).

| bar | value | floor it must clear (measured envelope) | defect it must catch | margins | verdict |
|---|---|---|---|---|---|
| `_CLOSURE_BAR` (gate 3) | 1e-9 | POST 3.11e-15 .. 6.66e-15 | PRE 4.89e-06 .. 3.26e-03 | 5.18 / 3.71 dec | **sound** |
| `_PASSIVITY_BAR` (gate 4) | 1e-8 | POST 8.85392e-10 .. 8.85394e-10 (spread 9e-16 over six samples) | PRE per-sample worst 6.6385e-04 .. 33.67 | **1.05** / 4.18 dec | **sound but tight**; see below |
| `_REFERENCE_BAR` (gate 5) | 1e-9 | POST 3.997e-15 .. 6.217e-15 | PRE 2.93e-05 .. 2.00e-03 | 5.21 / 4.47 dec | **sound** |
| `_X1_CLOSED_CLOSURE` | 1e-8 | POST 1.3323e-15 .. 1.4433e-15 | PRE smallest manifesting 2.41e-06 | 6.84 / 2.38 dec | **sound** |
| `worst_ratio > 10.0` (X-1 fail-before) | 10x | -- | PRE worst relative `sum(R)` **29.27 .. 158.28** over six samples | **0.47 dec** on the worst sample | **S1 shape, flag** |

Three notes.

1. **`_PASSIVITY_BAR`'s floor really is invariant.**  I reproduce
   `8.8539e-10` on all six samples to within 9e-16, on both arms, which is the
   claim the comment rests on.  The 1.05-decade headroom is the whole margin,
   though, and it is headroom over a FIXTURE's Fourier truncation error rather
   than over an arithmetic floor: sibling fixtures of the same family (my `S3`,
   `S4`) have a post-fix truncation error of **4.3e-07**, 1.6 decades ABOVE this
   bar.  The bar is scoped to `_pixel_stack`, not to the property.  Not a defect;
   worth stating in the comment so a later fixture edit does not silently cross
   it.
2. **The comment's "smallest PRE SPACER reading is 6.6385e-04" means the
   smallest PER-SAMPLE WORST, not the smallest mount.**  I reproduce 6.6385e-04
   exactly as the WSL 1-thread worst; individual PRE spacer mounts read as low as
   **1.60e-10** (WIN 2 threads, WSL 1 thread), i.e. below the bar.  The test is
   fine -- it asserts on the ladder's worst -- but the sentence reads as a
   per-mount envelope and is not one.
3. **The X-1 fail-before's `> 10x` is an S1-shape pin** (`docs/TESTING_STANDARDS.md`
   S1: "magnitude-ratio defect pin"), and its worst sample is 29.27x -- 0.47
   decades of headroom, not the 1.2 decades the quoted 152.60x suggests.  It
   passed on all six of my samples; it is the thinnest margin in the file.

`test_m1_conditioning_guard.py` reads **28 passed, 0 skipped** on the round-2
tree AND on the pre-round-2 tree, both builds: the restated X-1 tests pin
round 1's repair, not round 2's.

---

## 7. Defects

**V1 -- MEDIUM (documentation / scope).  The lossy exemption is stated as
"loss", and the discriminator is loss in the PATTERNED LAYER.**
A hybrid stack with a lossy SPACER (`Im eps = 1e-6`) and a lossless patterned
cell was wrong by **1.3353e-03 per order** before round 2 and is NOT
bit-identical across the arms; the acted-on layer population there is 4..8, not
0, on both builds.  The census row "LOSSY 11/11 bit-identical", the fix
document's "where the sign is physics, nothing moves", and gate 6's docstring
("for a lossy layer the acted-on population is EMPTY") all read as "loss
anywhere is safe".
*Severity:* the LIBRARY is correct; the scope statement a later reader would act
on is not.
*Reproducer:* `PYTHONPATH=. python validation/probe_verify_branch_cut_round2/v7_followups.py out.json`
on both trees, the `spacer_only_weak_normal` row; and
`v10_acted_population.py`, the `lossy_spacer_*` rows.
*Remedy:* say PATTERNED LAYER wherever the exemption is stated.  Gate 1 of
`tests/unit/test_verify_branch_cut_round2.py` now pins both sides.

**V2 -- MEDIUM (documentation).  The band-scale decision's stated basis does not
reproduce.**
"ARRAY-MAX has the larger two-sided gap on all three populations and on both
builds -- 12.35 / 7.38 / 13.71 against 12.09 / 7.16 / 12.58."  On my populations
per-mode has the LARGER gap on the ordinary RCWA population (14.13 / 14.26
against 13.88 / 14.01) and the two shapes are equally bad at a deep cutoff
(both -0.29 on WIN, both +0.01 on WSL).  The DECISION to keep array-max is still
defensible -- it wins on the hybrid PMM population and the cutoff corner is
never silently wrong under it -- but the sentence claims a clean sweep that is
not there.
*Severity:* no shipped behaviour is wrong; a later reader deciding whether to
change the shape would be misled about the evidence.
*Reproducer:* `v4_band.py` on both builds, the `pop1_rcwa_ordinary` row.
*Remedy:* re-state as "array-max wins on the hybrid PMM population, ties at a
cutoff, and loses narrowly on the ordinary RCWA one; it is kept because the
cutoff corner is never silently wrong under it".

**V3 -- LOW (documentation).  `_CUT_BAND_REL`'s per-population margins are
sample-scoped, in the direction that matters.**
The docstring states 6.75 / **6.46** decades for the hybrid PMM population.
Measured over 24 fixtures, the margin BELOW the signal side is **1.00 decade**
(nearest signal-side ratio 1.0036e-07, both builds), because a lossless hybrid
cell carries genuinely COMPLEX modes that sit close to the cut.  Under the
looser classification the fix document's own section 5.5 warns about, the
"noise side" is 3.0918e-04 -- 4.5 decades ABOVE the bar.
*Severity:* the bar is not crossed on any fixture measured; the stated envelope
is off by five decades on one side.
*Reproducer:* `v4_band.py`, `pop2_hybrid_pmm`, the `closest_signal_to_bar` and
`LOOSE_arraymax_noise_max` fields.
*Remedy:* record the strict and loose readings side by side, as this report does.

**V4 -- LOW.  Two headline numbers are one fixture's and read as general.**
(a) "only 1e-3 brings it to 8.7e-14": on my fixture the pre-round-1 arm is cured
at **1e-5** (7.30e-10), and at 3e-6 it is already at 1.98e-07.  (b) "the defect
is visible in a window around 1e-06, and only there": my fixture's LOUDEST rung
is a relative modulation of **1e-10** (PRE closure 3.29e-05).  (c) "post-fix the
sweep contains no call where the two instruments disagree by more than 2.2x":
my sweep reaches **4.737e+02**.
*Severity:* none of these changes a decision; all three would be quoted as
properties of the defect by a later reader.
*Reproducer:* `v2_hybrid.py` (detune and modulation ladders), `v5_x1_m1.py`
(`m1_post.max_ratio`).

**V5 -- LOW.  `cond(a+b)` does not universally recover.**
On 3 of my 5 coincident-spacer fixtures the post-fix PMM interface mode-match
still reads 1.5e7 .. 1.0e8, against 3.8e2 on the no-spacer control, and a detune
ladder shows it scaling exactly as 1/detune.  It is a NEAR-DEGENERACY of order
1/(cell modulation) -- a uniform layer whose permittivity nearly equals a weakly
modulated layer's background genuinely has nearly the same modes -- not the
singularity round 2 removed, and the closure is flat at 1.63e-10 across the
whole ladder.  The fix document's "1.959e+09 -> 1.453e+01" is that fixture's.
*Severity:* informational; the answers are correct and thread/build-independent.
*Reproducer:* `v7_followups.py`, the `cond_vs_detune` block.

**V6 -- INFO.  There is a fourth JAX caller, and it does not pass `jnp`.**
`elements/_berreman_jax.py:396` calls `_sqrt_decay(...)` with no `xp`; it works
because `array_namespace` recognises a JIT Tracer, and the auto-detecting call
is bitwise identical to the explicit one under `jit` (measured).  The fix
document says "the three JAX twins call it with jnp", which would let a later
change add `if xp is None: raise` and break a traced path no PMM test covers.
Pinned by gate 5 of the new test file.

**V7 -- INFO.  A `NaN` in the eigenvalue array silently disarms the pin for the
whole layer.**
`scale = max(max|r|, 1)` becomes `nan`, `on_cut` is False everywhere, and 14
conjugations become 0 on a synthetic 24-mode spectrum.  Shared with round 1
bit-for-bit, so not a round-2 regression, and I found no solve that reaches it.

**Not defects, recorded so they are not re-found:** the no-spacer control moves
1.7e-15 .. 3.1e-15 per order across the arms (the fix document says "identical",
which is true of its closure and not of its bits); and the restated X-1 / M1
tests pass on the pre-round-2 tree, i.e. they gate round 1.

---

## 8. Ship recommendation for 5.45.0

**SHIP.**

The change is a strict improvement and its central claim survives independent
re-measurement more strongly than it was stated:

* the hybrid PMM's pre-round-2 answer was wrong by up to **7.48 per order**
  against an independent method on fixtures I built, and returned **28.94** times
  the incident power on a passive lossless stack (WSL, one thread) -- against
  the fix document's own 2.0e-03 and 110;
* every one of those mounts now reads `sum R + T = 2.000000000` and agrees with
  the independent reference to its own truncation floor, identically on both
  builds at every thread count measured;
* X-1 reproduces to every printed digit, on both builds, in both arms;
* the refactor is bit-identical on 4,037 engineered values at eight array sizes,
  on both builds, and the JAX twins trace it with zero branch disagreements;
* the pure staggered engine and the 1-D PMM are bit-identical on 33 fixtures I
  built to break them, with zero calls to the function on either arm;
* the merged-tree gate is 86 passed on both builds and the JAX battery 71 passed.

Five conditions, all documentation or test-comment work, none blocking:

1. **Fix the lossy scope statement (V1).**  Say PATTERNED LAYER in the census
   row, in section 4.2's "where the sign is physics" sentence, and in gate 6's
   docstring.  This is the one place a reader could act on the text and be wrong
   by three decades.
2. **Re-state the band-scale comparison (V2) and the hybrid-PMM margin (V3)**
   in `_CUT_BAND_REL`'s docstring, with the strict and loose classifications
   side by side and the 1.00-decade signal-side margin named.
3. **Mark V4's three numbers as fixture-scoped** where they appear.
4. **Add the 1.05-decade note to `_PASSIVITY_BAR`** and the S1 note to the X-1
   `> 10x` fail-before, so the two thinnest margins in the file carry their own
   warning.
5. **Land `tests/unit/test_verify_branch_cut_round2.py`** in the same gate; it
   is 11 tests, 21.4 s pinned, green at 1 / 2 / 4 / 8 threads and unpinned on
   Windows and at 1 / 4 on WSL, and 4 of its 11 fail on the pre-round-2 tree on
   BOTH builds.

---

## 9. What I could NOT verify

* **CuPy / GPU.**  Every measurement is NumPy or JAX on CPU.  The shared body is
  `xp`-generic and `array_namespace` routes CuPy the same way, but no GPU arm
  was run -- the same gap the fix document records.
* **The armed `T22` refusal population.**  My M1 sweep reached the armed site on
  0 of 106 guarded inverses, exactly as the fix document's reached it on 0 of
  110.  Round 1's verification reading (minimum equilibrated `rcond` 2.792e-02,
  eight decades above its own bar) stands un-re-measured by anyone since.
* **The wide batteries on WSL.**  I ran the five-file gate and the whole probe
  set on both builds and the JAX battery on Windows; the 728-test PMM/RCWA
  battery and the 1284-test census/walker sweep were not re-run on either build
  (they are the fix document's, and the box is shared).
* **Whether the hybrid PMM has coincidence partners beyond the region, the
  superstrate and the uniform layer.**  I added no new partner class and found
  none; "no other exists" is not supported by anything here.
* **`stabilize=True`.**  Not characterised; whether its retry schedule can now be
  narrowed is still open.
* **The deepest reachable cutoff.**  My golden-section reached
  `min |lam^2| = 3.1722e-16`, which is within two decades of `eps_mach` on a
  spectrum of order 1, so it is close to the floor -- but a different device
  family could go deeper, and the noise side has no floor there for the
  structural reason `_CUT_BAND_REL`'s docstring already gives.
* **Whether the residual `cond(a+b) ~ 1/detune` (V5) has a consequence at a
  modulation much weaker than 1e-6.**  The closure is flat at every detune I
  ran; a modulation of 1e-12 was not tried against a reference.

---

## 10. Reproduction

```
git worktree add -b verify/branch-cut-round2 C:/tmp/lum_vbc2 664a84b
git -C C:/tmp/lum_vbc2 worktree add --detach C:/tmp/lum_vbc2_pre 2898767
cp C:/tmp/lum_vbc2/validation/probe_verify_branch_cut_round2/*.py \
   C:/tmp/lum_vbc2_pre/validation/probe_verify_branch_cut_round2/

# sys.path[0] is the SCRIPT's directory, and this box carries an editable
# install of lumenairy pointing at a DIFFERENT tree, so every probe calls
# _vcommon.pin_tree() FIRST: it puts the worktree at the head of sys.path and
# then ASSERTS that lumenairy.__file__ came from it, refusing to produce a
# number if anything got in first.  PYTHONPATH=. is belt and braces.
cd C:/tmp/lum_vbc2 && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 PYTHONPATH=. python -u \
  validation/probe_verify_branch_cut_round2/v2_hybrid.py out.json
```

| probe | what it measures |
|---|---|
| `_vcommon.py` | the tree pin, the arm detector (reads the LIVE source), the JSON stamp |
| `v1_consolidation.py` | AST census, refactor bit-identity, the JAX body and twins, scale contamination |
| `v2_hybrid.py` | five coincident-spacer stacks + controls + two lossy, `cond(a+b)`, the RCWA reference, the detune and modulation ladders |
| `v3_converse.py` | the pure staggered stack, the staggered cells and the 1-D PMM, with call counters |
| `v4_band.py` | ARRAY-MAX vs PER-MODE on three populations, plus the cutoff hunt |
| `v5_x1_m1.py` | the X-1 `THIN` ladder both arms, the M1 motivating population, M1 reachability |
| `v6_cutoff_consequence.py` | the deepest mounts under four branch rules |
| `v7_followups.py` | the loss partition, and `cond(a+b)` against the spacer detune |
| `v8_durability.py` | every bar in the round-2 test file, on its own fixture, per thread count |
| `v9_ladder_scan.py` | the two-build data behind the new test file's bars |
| `v10_acted_population.py` | the acted-on population counted on the eigenproblem, by loss placement |

Every JSON carries a `_stamp` naming the `lumenairy.__file__` measured, the ARM
detected from the live source, the interpreter, numpy, the platform and the
three thread environment variables.
