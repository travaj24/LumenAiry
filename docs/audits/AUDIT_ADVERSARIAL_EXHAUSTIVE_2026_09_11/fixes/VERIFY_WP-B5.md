# VERIFY-WP-B5 -- independent adversarial re-verification of WP-B5 (RCWA / EME / BOR)

Verifier: VERIFY-WP-B5 (did not write the fixes).  Subject: commit `4ea16066`
(`perf(rcwa): WP-B5 -- the two-interface closed form ... the off-plane fff_nv
operator symmetrised ... the Toeplitz solves measured and refused`), diff base
`4ea16066^` = `2680c24e`.  Items D1 / D2 / D3 of `fixes/WP-A14_REPORT.md`
section 6 (findings H4 and H3).  Branch `audit-fixes-2026-09`.

Every number below was re-MEASURED on this machine with
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, one process at a
time.  Nothing is quoted from the WP report without an independent
re-measurement; where my number differs from the WP's, both are given.

**Byte identity was taken ARCHIVE-TO-ARCHIVE, never against the working tree.**
`git archive 4ea16066^ lumenairy` and `git archive 4ea16066 lumenairy` were
extracted read-only into
`scratchpad/verify_b5/base/` and `scratchpad/verify_b5/live/`; every probe runs
in a child process whose cwd AND `PYTHONPATH` are that archive, asserts
`lumenairy.__file__` is under it, and prints it.  Nothing goes through pytest.
The two archives differ in exactly three files (`diff -rq`):
`rcwa/_core.py`, `rcwa/oned.py`, `rcwa/twod.py` -- so `eme/`, `bor/`,
`berreman.py`, `pmm/` and `rcwa/stack.py` are source-identical before a single
number is computed.

Three instruments were written for this pass and share no code with the library
or with the WP's gate file:

* a **double-double (2 x float64, ~2^-106) matvec and iteratively refined
  solve** (`scratchpad/verify_b5/work/dd.py`) -- this box has no `mpmath` and
  its `np.longdouble` is float64, so the extended-precision reference the D1
  and D2 claims need is built with Dekker splitting.  It scores every float64
  candidate on FORWARD error against a reference solution, which is strictly
  more than the WP could do by comparing candidates to each other;
* a **parent-arithmetic emulation** of the star site -- `_redheffer_star_rt`
  replaced, on the LIVE tree, by the parent's own two lines (assemble the star,
  then `S11 @ cinc` / `S21 @ cinc`).  Proved **byte-identical to the
  `4ea16066^` archive over 162 arrays** (`work/validate_emu.py`) before it was
  used, so both formulations can be measured in ONE process, on the same box,
  the same second, the same BLAS;
* a **leaky guided-mode fixture** that drives the star denominator
  `I - B11 A22` near-singular FROM THE PUBLIC API with `|X| <= 1` everywhere --
  the fixture that falsifies D2's residual-risk statement (section 3, D-1).

---

## 1. Summary

| # | Claim | Verdict | My oracle | My numbers |
|---|---|---|---|---|
| **D2-a** | the closed form reproduces the star | **VERIFIED** | the defining coupled `2n` system, solved whole with double-double refinement | agreement at the arithmetic floor wherever `I - B11 A22` is well conditioned: **6.6e-17 .. 2.9e-16** relative on six off-resonance rungs of my own fixture; on the resonance rungs BOTH formulations sit at the SAME distance from the reference (below) |
| **D2-a2** | it keeps the star's stability on THICK layers with strongly evanescent orders, and reduces to a single interface as the thickness vanishes | **VERIFIED** | the same operands through both formulations; the bare Fresnel interface | 5 / 10 / 20 wavelengths of Ag at `n_orders` 200, both polarizations: closed vs assembled **2.8e-17 .. 1.1e-16**, all finite, `sum R + T` identical to 12 digits.  `depth = 1e-18` m on a uniform `n = 3.5` layer: **\|R0 - Fresnel\| <= 9.7e-17**, residual then growing as `depth^2` |
| **D2-b** | `RCWAStack`, `berreman`, `eme`, `bor` byte-identical | **VERIFIED** | archive-to-archive, child process | **12/12 + 18/18 + 15/15 + 16/16 arrays byte-identical, max abs 0.0** |
| **D2-c** | `pmm/` byte-identical (`_symmetric_cascade_rt` keeps its assembled star) | **VERIFIED** | archive-to-archive, child process | **48/48 arrays byte-identical, 0.0**, incl. `pmm_jones_2d` / `pmm_efficiency_2d_cell` at `symmetry` False AND auto and `PMM2DStackHybrid`; the PMM census still records BOTH star inverses with bit-identical `rcond` |
| **D2-d** | the default moves `<= 1.665e-15` abs / `3.114e-15` rel | **VERIFIED-WITH-NOTES -- it is not a bound** | archive-to-archive over 829 arrays incl. thick / grazing / Wood / extreme-contrast / deep-truncation configurations | well-conditioned groups agree with the WP (worst abs **4.996e-16**); four groups exceed the stated envelope -- extreme contrast `n = 20` (**2.692e-15** abs), the ordinary 1-D Jones (**7.96e-15** rel) and 2-D efficiency (**7.74e-15** rel) families, and decisively the guided-mode-resonance family (**8.941e-09** abs / **9.060e-09** rel) |
| **D2-e** | the guard is preserved on the retained inverse; census 4 rows -> 3, same site string | **VERIFIED** -- this is D2's own acceptance criterion (WP-A14 section 6: a closed form "would have to reproduce all three" of the census, the residual probe and the refusal path) | archive-to-archive census diff | site strings identical, the three retained `rcond` readings **bit-identical**, `refused` False on both trees, 4 -> 3 rows on every 1-D single-layer solve; the refusal-armed `rcwa generalized interface (T22)` site is unchanged in count and `rcond` on the out-of-plane path |
| **D2-f** | the RETAINED denominator is the tighter equilibrated `rcond` (0.340/0.167/0.108/0.0695 vs 0.523/0.670/0.671/0.671) | **VERIFIED** (exactly), with one note | my own census on the same ladder | all eight numbers reproduced to the printed digits.  NOTE: on the **TE** rung of the same ladder the retained one is marginally the LOOSER reading (0.9877 vs 0.9862, 0.15 %); the gate admits 5 % and passes, but "the tighter reading at every rung" is the TM ladder's property, not a universal one |
| **D2-g** | `np.linalg.inv` per solve 6 -> 5 | **VERIFIED** | counting wrapper, both archives | TM `n_orders` 50 and 100: **6 -> 5**; TE 50: **5 -> 4** |
| **D2-h** | 1.20x / 1.23x / 1.37x at `n_orders` 50 / 100 / 200 | **VERIFIED-WITH-NOTES** | interleaved in one process (emulation) | **1.10x / 1.12x / 1.22x**, star share **13.8/17.7/25.4 % -> 4.7/6.1/6.5 %** (WP: 20.2/18.7/21.6 -> 5.0/6.8/6.7).  The after-shares reproduce; the whole-solve ratio is lower on a box shared with three other work packages.  Neither the report nor the gates assert a clock |
| **D2-i** | **residual risk: the two associations disagree by O(1) ONLY in the growing-propagator regime, which the branch cut makes unreachable from the public API** | **NOT CORRECT AS STATED -- defect found and FIXED** | a leaky guided-mode resonance, plus the double-double reference | `cond(I - B11 A22)` reaches **1.75e+13** from `rcwa_efficiency_1d` alone with `max|A22| = 1.000000` (no growing propagator).  There the two formulations are **4.2e-04** apart and **equally far (9.7e-04 each)** from the reference.  End to end the shipped per-order efficiency moves up to **6.404e-06** absolute between the two formulations |
| **D3-a** | the symmetrisation removes the spurious form birefringence | **VERIFIED** | the x<->y mirror of Maxwell's equations, on cells and tensors the WP did not use (BIAXIAL, `ezz != exx`, a cross) | own mirror `\|Jxx-Jyy\|` **7.3e-16 .. 8.5e-15** after vs **9.7e-05 .. 8.3e-03** before; mirror covariance **2.6e-15 .. 1.7e-14** after vs **1.5e-04 .. 4.8e-03** before |
| **D3-b** | a uniform rotated-director cell still matches conical Berreman, and the symmetrisation is inert there | **VERIFIED** | `elements/berreman.py` (called, not edited) on 7 tensors x 4 incidence pairs x M = 1, 2 | worst **7.29e-15** on `\|J\|` 0.24 .. 0.51, and the symmetrized and single-order answers agree to the printed digits on every row |
| **D3-c** | reduces bit-identically to the in-plane 2x2 operator; off-plane blocks exactly zero | **VERIFIED** | direct operator comparison at M = 2, 4, 7 (WP used 3, 5) | **0.0** on all four in-plane blocks and **0.0** on the four off-plane blocks at every M |
| **D3-d** | **the `l3-` Schur fold runs AFTER the mean, and that ordering matters** | **VERIFIED as behaviour, NOT GATED -- defect found and FIXED** | the algebra, plus a mutation | the shipped entry returns the RAW mean to **0.0**; the two orderings differ by **2.37e-05** abs / **1.69e-05** rel on the real operator at M = 3.  A mutation that swaps them left **all 38 tests green** |
| **D3-e** | the in-plane path is untouched | **VERIFIED** | archive-to-archive | `rcwa_jones_2d` at `'li'` / `'laurent'` on the SAME out-of-plane cell moves 4.4e-16 / 5.2e-15 (D2 only); only `'fff_nv'` + out-of-plane moves (1.57e-05 / 1.08e-03, which is the intended change) |
| **D3-f** | the symmetrisation is "measurably more expensive at large `n_orders`" | **VERIFIED-WITH-NOTES -- the direction is wrong** | direct timing | the operator costs **2.2x** the single order (5.9 -> 13.1 ms at M = 4, 57.0 -> 119.7 at M = 8) but its share of the whole solve **FALLS** with M: **7.7 % -> 3.7 % -> 2.9 %** at M = 4 / 6 / 8 |
| **D1-a** | `solve_toeplitz` costs 12x .. 20x the shipped composite | **VERIFIED** | my own metals (Al, Cu) at `n_orders` 100 and 300 | site A **11.7x / 16.4x** (Al), **11.4x / 15.0x** (Cu); site B **11.1x / 19.5x** (Al), **9.4x / 14.5x** (Cu) |
| **D1-b** | Levinson lands "two decades" further from its own equation | **VERIFIED-WITH-NOTES** | the row-equilibrated backward residual | **12.9x .. 163x** (1.1 .. 2.2 decades).  "Two decades" is the Ag / Cu reading; on Al it is 12.9x at `n_orders` 100 and 19.2x at 300 |
| **D1-c** | Levinson's ANSWER moves the operator above the package's 1.4e-13 closure; the LU route stays a decade below | **VERIFIED, and strengthened** | forward error against a double-double reference (the WP compared candidates to each other) | Levinson **4.51e-13 / 1.78e-12** (Al 100/300) and **7.52e-12 / 2.79e-12** (Cu); the shipped `inv` **3.53e-15 .. 8.59e-15** and an LU `solve` **2.36e-15 .. 8.75e-15**.  The shipped route is the ACCURATE one, not merely the different one |
| **D1-d** | `cond([[1/eps]])` over the metallic ladder is 2.51e+02, not 1e13 | **VERIFIED** (for metals) | direct | worst over Al / Cu x duty 0.1 / 0.5 / 0.9 x `n_orders` 100 / 300 is **1.60e+02**.  NOTE: the conditioning is set by the permittivity RATIO, so an epsilon-near-zero ridge (`n = 1e-6`) reaches **1.00e+12** with the inverse's own equilibrated residual at **5.1e-04** -- outside the claim's metal scope, but the 1e6 gate bar is not a universal fact about this matrix |
| **D1-e** | both sites still form the explicit inverse | **VERIFIED**, and the gate is real | mutation of the source the gate reads | swapping either site to a Levinson spelling turns `test_d1_the_two_sites_still_form_the_explicit_inverse` red |
| **pins** | the new gates are real | **VERIFIED**, with one gap | five in-memory mutations | drop the guard -> 1 red; revert D2 -> 1 red; revert D3 -> 5 red; Levinson at the D1 sites -> 1 red; **swap the Schur/mean ordering -> 0 red** (D-2, now 1 red) |
| **CI census** | "no decision may move" | **VERIFIED** | `tests/unit/test_ci_kernel_consistency.py` | **7 passed** |
| **suite** | no collateral damage | **VERIFIED** | `pytest tests/unit -k "rcwa or eme or bor"` | **2039 passed, 8 skipped, 1 xfailed, ZERO failures** in 1:21:34 -- the WP's five reds are gone (three were another engineer's in-flight `_lens_traced` edit, since re-recorded; two were this WP's and the orchestrator's patches fixed them) |

**Two defects found, both fixed inside WP-B5's own ownership** (section 3).
**Collateral damage attributable to WP-B5: none.**

---

## 2. Per item

### D2 -- the two-interface closed form

**The algebra.**  `_redheffer_star_rt` computes `C11 c = A11 c + A12 z` and
`C21 c = B21 (u + A22 z)` with `u = A21 c`, `z = D B11 u`,
`D = (I - B11 A22)^-1`.  The push-through identity `F = I + A22 D B11`
(`F = (I - A22 B11)^-1`) is what removes the second inverse, and it is exact:
`(I - A22 B11)(I + A22 D B11) = I + A22 D B11 - A22 B11 - A22 B11 A22 D B11
= I + A22 (I - B11 A22) D B11 - A22 B11 = I`.  The two zero-block shortcuts
match `_redheffer_star`'s on the same concrete `.any()` tests behind the same
`is_jax_array` backend guard.  Checked at the eight wired call sites: in
`rcwa_jones_2d` the two source columns are the UNIT polarizations
(`(ex0, ey0) = (1, 0)` then `(0, 1)`, `twod.py:2047`), so replacing
`xp.concatenate([ex0 * delta, ey0 * delta])` by the stacked pair is exact and
not merely equivalent.

**Byte identity and movement, archive to archive** (`work/move.py`, 833 keys ->
829 numeric arrays + 4 exception strings identical on both trees;
`work/cmp.py`):

| group | arrays | byte-identical | max abs | max rel |
|---|---|---|---|---|
| `berreman_jones_1d` (3 mounts) | 12 | **12** | 0 | 0 |
| `elements/eme` (`strip_x_modes` x4, lossy, `strip_vector_modes`, `layer_modes`, `cell_smatrix`) | 18 | **18** | 0 | 0 |
| `elements/bor` (`radial_spectrum` x6, `guided_modes`, `fiber_modes`, `radial_coupled_modes`, `BORStack`) | 15 | **15** | 0 | 0 |
| `RCWAStack` 1-D, 1 and 2 layers, `symmetry` off and auto (R, T, Jr, Jt) | 16 | **16** | 0 | 0 |
| `rcwa_efficiency_1d` (5 media x 2 pol x 2 formulations x 4 `n_orders` x 2 angles) | 480 | 190 | **4.996e-16** | 3.980e-15 |
| 1-D thick layers (1 / 5 / 10 / 20 wavelengths, `n_orders` 100) | 24 | 13 | 1.110e-16 | 9.632e-16 |
| 1-D near-grazing (80 / 89 / 89.9 / 89.99 deg) | 24 | 10 | 1.804e-16 | 1.605e-15 |
| 1-D Wood anomalies (superstrate and substrate side) | 36 | 17 | 5.551e-16 | 1.854e-15 |
| 1-D extreme contrast (`n` = 10 / 20 / 0.05+30j) | 18 | 6 | **2.692e-15** | **2.921e-13** |
| 1-D deep truncation (`n_orders` 300 / 400, Ag and Al) | 12 | 5 | 2.220e-16 | 1.160e-15 |
| **1-D guided-mode resonance** | 48 | 22 | **8.941e-09** | **9.060e-09** |
| `rcwa_jones_1d` (+ full 3x3) and `rcwa_jones_1d_segments` | 45 | 9 | 1.776e-15 | **7.963e-15** |
| `rcwa_efficiency_2d` (2 cells x `'laurent'`/`'li'` x 2 angles) | 24 | 8 | 8.327e-16 | **7.741e-15** |
| `rcwa_jones_2d` (+ out-of-plane) | 48 | 16 | **1.575e-05** | **1.079e-03** |
| `rcwa_efficiency_2d_shapes` (`symmetry` off and auto) | 6 | 2 | 2.776e-16 | 4.525e-15 |
| `PreparedRCWA2D.solve` (3 wavelengths) | 3 | 0 | 8.882e-16 | 2.473e-15 |
| **`elements/pmm`** (separate harness, `work/move2.py`) | 48 | **48** | 0 | 0 |

Four rows want a sentence each.

*The four MUST-NOT-MOVE groups are exactly 0.*  That half of the claim is
unconditionally VERIFIED, and so is the PMM half the report argued for
separately: `_symmetric_cascade_rt` still assembles its star, and the PMM
census proves it (`pmm interface mode-match (a+b) x2 | rcwa Redheffer star
(I - A22 B11) x1 | rcwa Redheffer star (I - B11 A22) x1`, with the four `rcond`
readings bit-identical on the two trees).

*`rcwa_jones_2d (+ out-of-plane)` moving 1.08e-03 is D3, not D2*, and the
changelog's migration note covers it.  It is broken out here because the
report's D2 movement table carries a row with that exact label and the number
`3.114e-15`, which cannot include `formulation='fff_nv'` on an out-of-plane
cell.  On the SAME cell `'li'` and `'laurent'` move 4.4e-16 / 5.2e-15, so the
scoping is right and only the label is misleading (requested correction,
section 5(c)).

*The relative envelope `3.114e-15` is a property of the WP's fixture set.*  Four
of my ordinary well-conditioned groups reach 4.0e-15 .. 8.0e-15 relative on
absolute movements of 1e-16 .. 2e-15 -- the same arithmetic floor, divided by a
smaller efficiency.  The ABSOLUTE envelope 1.665e-15 survives everywhere except
extreme contrast (`n_ridge = 20`, 2.692e-15) and the resonance family.

*The guided-mode-resonance family is the one that matters*, and it is D-1.

**The guard, against D2's own acceptance criterion.**  WP-A14 deferred D2
because "`_guarded_inverse` carries the M1 conditioning census, its residual
probe and its refusal path, and a closed form would have to reproduce all three
or lose the guard on the default path of every single-layer solve".  Diffed
archive to archive (`work/move2.py`, the census armed around each solve):

| solve | parent census | `4ea16066` census |
|---|---|---|
| `rcwa_efficiency_1d` TM, `n_orders` 11 / 50 / 100 / 200 | `mode-match (a+b)` x2, `(I - B11 A22)` x1, `(I - A22 B11)` x1 | `mode-match (a+b)` x2, `(I - B11 A22)` x1 |
| `rcwa_efficiency_1d` TE, `n_orders` 50 | same 4 | same 3 |
| `rcwa_jones_1d` out-of-plane | `generalized interface (T22)` x2 + the 2 stars | `generalized interface (T22)` x2 + 1 star |
| `rcwa_jones_2d` out-of-plane `'fff_nv'` | 4 rows | 3 rows |
| `pmm_jones_2d` | `pmm mode-match (a+b)` x2 + BOTH stars | **unchanged, all four `rcond` bit-identical** |

The retained rows' `rcond` values are bit-identical on the two trees on every
solve, the `refused` flag is `False` everywhere on both, and the one site where
a refusal is ARMED (`_interface_smatrix_general`'s `T22`) is unchanged in count
and in reading.  So the census records it identically, the residual probe is
the same instrument on the same matrix, and the refusal path is untouched --
all three of the design's conditions, measured rather than argued.

**The near-singular star denominator, reached from the public API.**  The
report's residual-risk paragraph names one regime in which the re-association
is not neutral (an exponentially GROWING layer propagator) and says
`_sqrt_decay`'s branch cut makes it unreachable.  The branch cut does close
that door.  It does not close the regime: what the re-association needs is a
near-singular `I - B11 A22`, and a HIGH-Q CAVITY RESONANCE produces one with
`|X| <= 1` throughout.

Construction (`work/d2_resonance.py`, `work/d2_scan.py`): a weakly modulated
high-index slab in air, period 0.45 um at 633 nm, `n = 2.0 + dn` / `2.0`,
duty 0.5, `n_sup = n_sub = 1.0`, `n_orders` 11 .. 41.  The `+-1` order has
`kx/k0 = 1.4067`: EVANESCENT in both half-spaces (total internal reflection,
`|r| = 1` at both faces) and PROPAGATING in the layer -- a leaky guided mode.
An eigenvalue of `B11 A22` then has modulus `1 - O(dn^2)` and a phase the
thickness tunes, so the denominator can be driven as close to singular as
float64 allows.  Confirmed directly: on the UNIFORM slab `max|eig(B11 A22)| =
1.000000` exactly at every thickness, and `max|A22| = 1.000000` -- there is no
growing propagator anywhere in this fixture.

The resonance is located at runtime (a scan then golden section on
`min |1 - eig(B11 A22)|`).  At the located thickness, on the SAME captured
`(SA, SB, cinc)`:

| `dn` | pol | `cond(I - B11 A22)` | `max\|A22\|` | closed vs reference | star vs reference | closed vs star | `50 cond eps` |
|---|---|---|---|---|---|---|---|
| 3e-02 | te | 6.391e+12 | 1.000000 | 3.263e-04 | 3.263e-04 | 3.591e-04 | 7.1e-02 |
| 3e-02 | tm | **1.750e+13** | 1.000000 | **9.679e-04** | **9.679e-04** | 4.249e-04 | 1.9e-01 |
| 1e-02 | te | 6.755e+03 | 1.000000 | 6.168e-13 | 6.172e-13 | 3.555e-16 | 7.5e-11 |
| 1e-02 | tm | 2.389e+12 | 1.000000 | 1.568e-05 | 1.896e-05 | 4.130e-06 | 2.7e-02 |
| 1e-03 | te | 4.214e+06 | 1.000000 | 7.649e-10 | 7.649e-10 | 7.904e-16 | 4.7e-08 |
| 1e-03 | tm | 1.553e+12 | 1.000000 | 6.461e-06 | 6.471e-06 | 3.045e-07 | 1.7e-02 |

(reference = the defining coupled `2n` system solved whole; the same rows OFF
resonance read **6.6e-17 .. 2.9e-16** on every column, which is the two-sided
contrast.)  With a double-double refined reference instead
(`work/d2_resonance.py`, 601-point locator) the worst reading in the family is
**closed 2.586e-04, star 2.586e-04, closed-vs-star 4.429e-04** at `dn = 3e-02`
TE, reference residual 1.9e-16.

Three conclusions, and they are not the same conclusion:

1. **the regime IS reachable from the public API.**  `rcwa_efficiency_1d`, no
   monkeypatching, `|X| <= 1`, `cond` 1.75e+13;
2. **the closed form is not at fault.**  Both formulations are the same
   distance from an extended-precision reference on every rung, and both sit
   two and a half decades inside `50 * cond * eps`.  This is the conditioning
   of the cavity denominator, which the ASSEMBLED star pays identically;
3. **but the stated envelope is not a bound.**  End to end, over `dn` 1e-1 ..
   1e-6 x `n_orders` 11 / 15 / 25 / 41 x both polarizations (80 located
   resonances, `work/d2_scan.py`, both arms in one process), the shipped
   per-order efficiency moves between the two formulations by up to
   **6.404e-06 absolute / 7.760e-06 relative** on a solve the library RETURNS
   (`sum R + T = 1.0137`: under the 1.05 gross tripwire, so `_check_energy`
   warns on the lossless clause and hands the answer back).  The rungs that
   say the most are the SILENT ones -- below the 1e-6 lossless-closure warning
   threshold, so no signal of any kind reaches the caller:

   | `dn` | `n_orders` | pol | `sum R + T` | signal | movement |
   |---|---|---|---|---|---|
   | 1e-1 | 15 | tm | 1.000000000007 | none | **1.843e-14** |
   | 1e-2 | 15 | tm | 1.000000000466 | none | **2.949e-13** |
   | 1e-2 | 25 | tm | 1.000000004609 | none | **2.014e-12** |
   | 3e-3 | 25 | te | 1.000000012520 | none | **1.206e-11** |
   | 1e-3 | 41 | tm | 0.999998889558 | warns (1.1e-06) | **1.959e-10** |
   | 1e-6 | 11 | tm | 1.013749727525 | warns (1.4e-02) | **6.404e-06** |

   At the STAR's own output the disagreement tracks `cond * eps` and is
   monotone in it (the table above: 3.6e-16 at `cond` 6.8e+03 through 4.1e-06
   at 2.4e+12 to 4.2e-04 at 1.75e+13).  END TO END it is that disagreement
   projected onto the propagating orders, so rungs of similar `cond` differ by
   decades -- which is why the family has to be scanned rather than a single
   worst case quoted.  The first row alone -- a solve whose energy closes to
   7e-12 and which warns about nothing -- is already an order of magnitude
   outside the stated absolute envelope.

The honest statement of the residual risk is therefore: *the re-association is
neutral while `I - B11 A22` is well conditioned; where it is not, neither
association is the better one and the answer is not determined to the quoted
precision on any build.*  That is what the docstring now says (section 3, D-1)
and what the new gate pins.

A side observation the next reader may want: the library's own equilibrated
instruments are BLIND to this failure mode.  Over sixteen located resonances
whose `cond` runs to 1.14e+13, the site's `_rcond_1_equilibrated` reads
**0.040 .. 0.489** and `_equilibrated_inverse_residual` reads
**6.3e-17 .. 4.6e-16**.  They are not wrong
-- the inverse IS computed accurately in its own scaling; the amplification is
in the ANSWER, not in the inverse -- but no conditioning guard placed on this
matrix would see the resonance.  Follow-up B5-D2c.

**The three edges the brief names** (`work/edge.py`, both arms in one process
through the validated emulation):

*Thick layers with strongly evanescent orders.*  A 1-D Ag grating 5, 10 and 20
wavelengths thick at `n_orders` 200 (`N` = 401), both polarizations -- the
regime where `A22 = X S22 X` spans ~300 decades and underflows into subnormals.
The closed form keeps the star's stability exactly: closed vs assembled
**2.776e-17 .. 1.110e-16**, every entry finite, and `sum R + T` identical to
twelve printed digits on all six rungs (0.944558468957 / 0.899036845884 /
0.908280645535 / 0.829326380389 / 0.840659872209 / 0.728885402101).  No
overflow, no digit loss: the push-through never forms a growing exponential.

*Thickness -> 0 reduces to a single interface.*  A UNIFORM `n = 3.5` layer
between air and `n = 1.5`, so the exact answer at zero thickness is the bare
Fresnel interface.  At `depth = 1e-18` m the solver returns it to
**|R0 - Fresnel| = 2.8e-17 .. 9.7e-17** and **|T0 - Fresnel| = 5.6e-16 ..
8.9e-16** at both polarizations and at normal and 0.35 rad incidence, and the
residual then grows as `depth^2` exactly as a thin film must (1.7e-15 at 1e-15
m, 1.7e-09 at 1e-12 m, 1.7e-03 at 1e-09 m).  Closed vs assembled over all
sixteen rows: **0.0 .. 4.441e-16**.

*A fixture that trips the armed refusal.*  My own scan (out-of-plane
`rcwa_jones_1d` and `rcwa_jones_2d` at index coincidences, `n_e` up to 9, large
period, `n_orders` 3 .. 9) found NONE -- the round-3 branch-cut fix repaired
that class, which is the library working.  The refusal path is therefore gated
where it already lives: `test_fix_slant_anchor_v1_v2_o2.py` and
`test_verify_slant_anchor_v1_v2_o2.py` (which are NOT in the `-k "rcwa or eme
or bor"` selection, so nobody's routine run covers them) drive
`_interface_smatrix_general`'s `T22` guard to raise and pin the message text
clause by clause: **30 passed** on this tree.  Together with the census diff
above -- the `T22` site unchanged in count and reading -- that is the whole of
the claim "a fixture that trips the guard trips it identically".

**Cost, interleaved** (`work/perf.py`, medians of five, Ag TM ladder, one BLAS
thread, both arms in one process):

| `n_orders` | `N` | assembled | closed | speed-up | star share before -> after |
|---|---|---|---|---|---|
| 50 | 101 | 15.9 ms | 14.5 ms | 1.10x | 13.8 % -> 4.7 % |
| 100 | 201 | 80.8 ms | 72.2 ms | 1.12x | 17.7 % -> 6.1 % |
| 200 | 401 | 471.4 ms | 387.9 ms | 1.22x | 25.4 % -> 6.5 % |

The star's collapse to ~5-7 % of the solve reproduces the WP's figure almost
exactly; the whole-solve ratio is lower than the WP's 1.20x / 1.23x / 1.37x on
a box shared with three other work packages.  Reported, not asserted, on both
sides.

### D3 -- the off-plane (full 3x3) `fff_nv` operator

`work/d3.py`.  Six probes; the first four read the answer against something the
library did not produce (a Berreman 4x4, the x<->y mirror of Maxwell's
equations, and the algebra itself), on cells and tensors the report did not
use.

*(a) A conical Berreman 4x4 on a laterally uniform cell*, at tensors and angles
the report did not use: uniaxial at (polar, azimuth) = (40, 10), (65, 70),
(80, 135) deg, a LOSSY uniaxial (`n_e = 2.6 + 0.1j`) at (33, 200), two BIAXIAL
tensors (`1.5/1.7/2.0` and `2.0/2.2/2.6`, general Euler rotations) and a
`ezz != exx` tilted tensor -- each at incidence `(theta, phi)` = (0, 0),
(0.30, 0.70), (0.60, 2.10) and (0.95, 4.00) rad, at M = 1 and 2.  Worst
`max|dJ|` = **7.288e-15** on a Jones of scale 0.24 .. 0.51, and the symmetrized
and single-order answers agree to the printed digits on all 56 rows -- which is
what must happen on a laterally uniform cell, and gates the off-plane path's
absolute correctness and the symmetrisation's inertness at once.

*(b) Mirror covariance on cells with no symmetry of their own* (rectangle, an
L, a disk; biaxial and `ezz != exx` tensors), M = 3 and 4:
`max|J_mirror - P J P|` = **2.597e-15 .. 1.669e-14** after, against
**1.463e-04 .. 4.804e-03** before, on `|J|` 0.15 .. 0.20.

*(c) The cell's own mirror, with a BIAXIAL director in the `x = y` plane*
(built by symmetrising a general biaxial tensor under `P eps P`, verified
`0.00e+00`), on a square, a disk and a CROSS: `|Jxx - Jyy|` =
**7.297e-16 .. 8.500e-15** after, **9.652e-05 .. 8.310e-03** before.

*(d) The `l3-` fold ordering.*  The shipped entry returns the RAW mean of the
two orders to **0.0** (checked against a mean recomputed here from
`_li_tensor_full_l2l1` and its transposed run), so the caller's fold runs after
it.  The ordering is not cosmetic: on the real operator
`max|Schur(mean) - mean(Schur)|` = **2.373e-05** (relative **1.694e-05**) at
M = 3 and **2.246e-05** / **1.597e-05** at M = 4.  A hand-built 2x2-block pair
puts the same difference at 1.01e+00.  See D-2: nothing gated this.

*(e) Reduction to the in-plane operator*, at M = 2, 4, 7 on a cell with
`ezz != exx` and no off-plane components: the four in-plane blocks are
**0.0** from `_li_convolutions_2d_tensor` and the four off-plane blocks are
**0.0**.

*(f) Cost.*  Operator build, single order -> symmetrized: **5.9 -> 13.1 ms**
(M = 4), **26.2 -> 40.6** (M = 6), **57.0 -> 119.7** (M = 8), against whole
solves of 171 / 1112 / 4060 ms -- so the symmetrized operator is **7.7 % /
3.7 % / 2.9 %** of the solve.  The extra work is real but its share FALLS with
truncation, which is the opposite of the report's "measurably more expensive at
large `n_orders`" (requested correction, section 5(d)).

### D1 -- the two Toeplitz inverses

`work/d1.py`, on ALUMINIUM (`1.3399 + 7.3441j`) and COPPER
(`0.2130 + 3.6700j`) at 633 nm, duty 0.5, `n_orders` 100 and 300, medians of
five (three for the Levinson arms), one BLAS thread.  Site A is the composite
the planar TM fast path needs (`inv` + two products, against two `solve`s,
`lu_factor` + two `lu_solve`, and two `solve_toeplitz`).

| metal | `n_orders` | `N` | `cond(T)` | ships | 2x`solve` | `lu`+2 | 2x`toeplitz` | ratio |
|---|---|---|---|---|---|---|---|---|
| Al | 100 | 201 | 1.595e+02 | 5.16 ms | 6.50 | 4.94 | **60.5** | **11.7x** |
| Al | 300 | 601 | 9.691e+01 | 97.2 ms | 117.5 | 86.0 | **1591** | **16.4x** |
| Cu | 100 | 201 | 1.243e+02 | 4.32 ms | 8.78 | 4.61 | **49.5** | **11.4x** |
| Cu | 300 | 601 | 3.345e+01 | 94.2 ms | 87.8 | 91.5 | **1417** | **15.0x** |
| Ag | 100 | 201 | 8.377e+01 | 4.42 ms | 6.93 | 4.80 | **58.9** | **13.3x** |
| Ag | 300 | 601 | 3.239e+01 | 112.1 ms | 107.2 | 94.9 | **1586** | **14.1x** |

Site B (`inv(EPS)`, consumed only elementwise) -- `inv` / `solve` /
`solve_toeplitz`: Al **3.28 / 3.82 / 36.6** ms at 100 and **41.8 / 50.8 /
816.0** at 300; Cu **3.74 / 6.85 / 35.0** and **57.6 / 55.7 / 835.2**.  So
9.4x .. 19.5x, and the plain `solve` is a wash.  The WP's 12x .. 20x band is
confirmed on metals it did not use.

**Accuracy, with a reference the WP did not have.**  Row-equilibrated backward
residual `r`, and FORWARD error `fe` against a double-double iteratively
refined solution of the same system:

| metal | `n_orders` | `r_inv` | `r_lu` | `r_lev` | `fe_inv` | `fe_lu` | `fe_lev` |
|---|---|---|---|---|---|---|---|
| Al | 100 | 1.24e-14 | 1.47e-14 | **1.60e-13** | 6.87e-15 | 8.03e-15 | **4.51e-13** |
| Al | 300 | 2.14e-14 | 2.36e-14 | **4.11e-13** | 7.06e-15 | 8.26e-15 | **1.78e-12** |
| Cu | 100 | 4.86e-15 | 4.60e-15 | **5.89e-13** | 3.53e-15 | 2.36e-15 | **7.52e-12** |
| Cu | 300 | 6.93e-15 | 7.67e-15 | **2.68e-13** | 8.23e-15 | 8.75e-15 | **2.79e-12** |
| Ag | 100 | 3.47e-15 | 4.58e-15 | **5.64e-13** | 4.28e-15 | 6.35e-15 | **8.64e-12** |
| Ag | 300 | 6.00e-15 | 5.67e-15 | **8.93e-13** | 8.59e-15 | 8.42e-15 | **2.11e-12** |

The backward-residual ratio is 12.9x .. 163x (1.1 .. 2.2 decades), so "two
decades" is the Ag / Cu reading and not universal -- but the FORWARD error
settles the question the WP could only argue: Levinson's answer is
**4.5e-13 .. 8.6e-12** from the true solution while both LAPACK routes are at
**2.4e-15 .. 8.6e-15**.  The shipped route is the accurate one; the refusal is
right, and it is right for a stronger reason than the report gives.

**The correction of record holds, with a scope.**  Over Al / Cu x duty
0.1 / 0.5 / 0.9 x `n_orders` 100 / 300 the worst `cond([[1/eps]])` is
**1.60e+02** and the worst `cond([[eps]])` **1.60e+02** -- nowhere near 1e13,
exactly as the report says.  The conditioning of this matrix is set by the
permittivity RATIO across the step, so it is a fact about METALS, not about the
matrix: an epsilon-near-zero ridge (`n = 1e-3`) reads **1.00e+06**, exactly the
gate's bar, and `n = 1e-6` reads **1.00e+12** with the explicit inverse's own
equilibrated residual at **5.1e-04**.  The gate
(`test_d1_the_inverse_rule_toeplitz_is_not_the_ill_conditioned_matrix`) is
parametrized over four metals and is correct as written; anyone widening it
should know where the bar comes from.  Follow-up B5-D1b.

---

## 3. Defects found and fixed

### D-1.  `_redheffer_star_rt`'s documented boundary was wrong (`rcwa/_core.py`)

**What it said.**  That the re-association is non-neutral only where a LAYER
mode carries an exponentially growing propagator, and that
`_sqrt_decay`'s `Re(lam) >= 0` branch "is what makes that unreachable" from the
public API.

**Why that is a defect and not a quibble.**  The same paragraph is the safety
argument for a deliberate default change, it is quoted in the report
(section 2 and section 5(a) point 2) as the reason two test helpers may record
an `inf`, and a reader auditing the next change to this site will use it to
decide how far the `1.7e-15 / 3.1e-15` envelope can be trusted.  It is
falsifiable and false: a high-Q cavity resonance reaches the same near-singular
denominator with `|X| <= 1`, from `rcwa_efficiency_1d` alone.

**Fail-before, measured** (all numbers in section 2, D2): `cond(I - B11 A22)`
1.75e+13 with `max|A22| = 1.000000`; the two formulations 4.2e-04 apart and
9.7e-04 each from the reference; the shipped per-order efficiency moving
6.404e-06 end to end.

**The fix** is in the docstring only -- **no line of executable code changed,
and no default moves.**  The envelope paragraph is now scoped ("WHILE
`I - B11 A22` IS WELL CONDITIONED that is the whole of the difference") and the
residual-risk paragraph names both routes into the regime, says which one the
branch cut closes and which one does not, gives the measured numbers for the
open one, and points at the gate.

**The gate**, new in
`tests/unit/test_audit2609_b5_rcwa_eme_bor.py::test_d2_a_near_singular_star_denominator_is_reachable_and_neither_form_is_better`:
the resonance is located at RUNTIME on the running build (coarse scan + golden
section on the eigenvalue gap of `B11 A22`), the premise is ASSERTED rather
than skipped (`cond >= 1e10` on at least one rung, `max|A22| <= 1` on all of
them), and the claim is two-sided -- the star output is undetermined by more
than 1e-9 (measured 6.5e-06 .. 9.7e-04, against 6.6e-17 .. 2.9e-16 off
resonance) AND both formulations sit inside `50 * cond * eps` (measured 9.7e-04
against 1.9e-01).  Verified to fail when the premise is removed: with the
locator returning an off-resonance thickness the test fails at the premise
assertion with all six rungs at `cond` 1.08 .. 1.59.

### D-2.  The `l3-` fold ordering was asserted everywhere and gated nowhere (`twod.py`)

**What was missing.**  `_li_convolutions_2d_tensor_full`'s docstring, the call
site's comment, the report and the changelog all state that the mean is taken
on the RAW `ehat` blocks so the `l3-` `E_z` fold runs AFTER it, "the mean of
two Schur complements is not the Schur complement of the mean".  That is
correct and load-bearing -- the caller also hands the raw cross-blocks and
`ehat^{33}` to the generalized generator's own `inv(EZZ)`, so folding first
would give it two quantities from different operators.  It is also an explicit
requirement of the design WP-B5 was implementing: `fixes/WP-A14_REPORT.md`
section 6, D3 -- "the `l3-` Schur fold then has to be applied after the mean,
not before, since the mean of two Schur complements is not the Schur complement
of the mean".  Nothing tested it.

**Fail-before, measured.**  A mutation that computes the mean of the two
per-order Schur complements and hands the caller blocks that reproduce it
(`scratchpad/verify_b5/mut/mut_schur_first.py`) leaves
`tests/unit/test_audit2609_b5_rcwa_eme_bor.py` at **38 passed** -- every D3
gate included.  It cannot be otherwise: the x<->y mirror is a symmetry of BOTH
orderings, so no mirror test can separate them, and on a cell with no off-plane
components the Schur complement is the identity, so the reduction test cannot
either.  The two orderings differ by **2.373e-05** absolute / **1.694e-05**
relative on the real operator.

**The gate**, new:
`::test_d3_the_symmetrisation_is_the_raw_mean_so_the_l3_fold_runs_after_it`.
Tolerance-at-0.0 on all nine blocks against the mean recomputed in the test
from `_li_tensor_full_l2l1` and its transposed run, plus a measured
significance bar (`Schur(mean)` vs `mean(Schur)` must differ by more than 1e-8
relative, three decades under the measurement) so the fixture can never become
one that does not distinguish the two.  Under the mutation it is **red**
(1.4255e-05 against the 0.0 tolerance), and it is the only test in the file
that is.

**No library code changed for D-2.**

---

## 4. Tests run

All with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`,
`-q --no-header -p no:cacheprovider`, one process at a time.  The box was
shared with other work packages throughout, so durations are upper bounds.

| command | result | duration |
|---|---|---|
| `pytest tests/unit/test_audit2609_b5_rcwa_eme_bor.py` (before my additions) | **38 passed** | 18.14 s |
| `pytest tests/unit/test_audit2609_b5_rcwa_eme_bor.py` (after, 2 added) | **40 passed** | 17.47 s, re-run at the end 16.61 s |
| `pytest ... -p mut_noguard` (the retained star inverse loses `_guarded_inverse`) | **1 failed**, 37 passed -- `test_d2_a_single_layer_solve_records_one_star_inverse_not_two` | 18.77 s |
| `pytest ... -p mut_assembled` (D2 reverted in memory) | **1 failed**, 37 passed -- the same gate, on the re-appearing `(I - A22 B11)` row | 18.79 s |
| `pytest ... -p mut_nosym` (D3 reverted in memory) | **5 failed**, 33 passed -- both mirror gates (x2 params) + the in-plane reduction | 11.32 s |
| `pytest ... -p mut_d1_source` (both D1 sites spelled as Levinson) | **1 failed**, 37 passed -- `test_d1_the_two_sites_still_form_the_explicit_inverse` | 17.72 s |
| `pytest ... -p mut_schur_first` (fold before the mean), BEFORE my gate | **38 passed** -- the gap | 18.62 s |
| `pytest ... -p mut_schur_first`, AFTER my gate | **1 failed**, 39 passed -- `test_d3_the_symmetrisation_is_the_raw_mean_so_the_l3_fold_runs_after_it` | 19.90 s |
| `pytest ...::test_d2_a_near_singular...` `-p mut_offres` (premise removed) | **1 failed** at the premise assertion, as designed | 1.36 s |
| `pytest tests/unit/test_audit2609_a14_rcwa_eme_bor.py tests/unit/test_audit2609_a14_verify.py` | **59 passed** | 133.57 s |
| `pytest tests/unit/test_v5_20_12_rcwa_jones_2d_fff_nv.py tests/unit/test_audit_s1_2_rcwa_lossless_tripwire.py tests/unit/test_v5_11_0_rcwa_fff_nv_2d.py tests/unit/test_m1_conditioning_guard.py tests/unit/test_audit2609_a17_history_lint.py` | **60 passed, 1 skipped** (the documented `test_m1_conditioning_guard.py:1354` "premise absent on this arm") -- the two arms the orchestrator patched are GREEN | 312.78 s |
| `pytest tests/unit/test_ci_kernel_consistency.py` | **7 passed** -- no guard decision moved | 2.46 s |
| `pytest tests/unit -k "rcwa or eme or bor"` (2047 selected, 13877 deselected) | **2039 passed, 8 skipped, 1 xfailed, ZERO failures** | 4894.06 s (1:21:34) |
| `pytest tests/unit/test_fix_slant_anchor_v1_v2_o2.py tests/unit/test_verify_slant_anchor_v1_v2_o2.py` (the only tests that make the ARMED `T22` refusal actually fire; not in the `-k` selection) | **30 passed** | 46.61 s |
| `python validation/run_all.py test_rcwa` | **PASS** (1 of 1) | 1.4 s |
| `ruff check lumenairy/elements/rcwa lumenairy/elements/eme lumenairy/elements/bor tests/unit/test_audit2609_b5_rcwa_eme_bor.py` | **All checks passed** | < 1 s |
| `python scripts/record_history_fingerprints.py --check` | every `rcwa` / `eme` / `bor` document **OK**; the tree's only `DRIFT` on the final run is `lumenairy.propagators.asymptotic_aberration_tensor.md`, another work package's in-flight edit (the tree was drift-free earlier in this pass, so it appeared while I was running) | 3 s |
| `python scripts/check_doc_identifiers.py` | **OK**, 598/598 resolve | 6 s |

The big selection is the WP's own verification set and it is **clean on this
tree**, where the WP recorded five failures: the three `_lens_traced` ones were
another engineer's in-flight edit and have since been re-recorded (the
fingerprint check now reads OK on every document), and the two that WERE this
work package's are green because the orchestrator applied the section-5(a)
patches (`_EnergyError` imported and caught in both files -- verified present
at `test_v5_20_12_rcwa_jones_2d_fff_nv.py:35,423` and
`test_audit_s1_2_rcwa_lossless_tripwire.py:31,318`).  The 8 skips are the
pre-existing set (PySide6 x2, the two `test_v5_6_rcwa_convergence.py`
"no measure-zero large-period blow-up on this LAPACK build", the
`threadpoolctl installed` arm, and three dispatcher / glass ones) and the
`xfail` is the documented `test_verify_bor_guards_round3.py` D-V1.

`docs/history/` carries `lumenairy.elements.rcwa.stack.md` and nothing for
`_core.py`, `oned.py` or `twod.py` (`ls docs/history | grep -i rcwa` returns
that one file), and `stack.py` is not in my diff -- so there is **no history
document to re-record** for this pass.

Measurement scripts, run once each and NOT tests:
`scratchpad/verify_b5/work/dd.py` (the double-double instrument),
`d2_probe.py`, `d2_eig.py`, `d2_resonance.py`, `d2_scan.py`, `move.py`,
`move2.py`, `cmp.py`, `validate_emu.py`, `perf.py`, `d1.py`, `d3.py`,
`edge.py`, `show_rows.py`; mutation plugins under `scratchpad/verify_b5/mut/`; the two
read-only archives under `scratchpad/verify_b5/base/` and `.../live/`.

---

## 5. Requested changes outside my ownership

**(a) `fixes/WP-B5_REPORT.md` section 2, "Residual risk, stated exactly".**
The paragraph is the safety argument for the change and is falsified as
written.  Requested replacement:

> **Residual risk, stated exactly.**  The closed form is a RE-ASSOCIATION, and
> it is neutral exactly while `I - B11 A22` is well conditioned.  Where that
> denominator is near-singular the answer is a difference of terms far larger
> than the difference, and both formulations carry their own `cond * eps`.
> Two things reach that regime.  An exponentially GROWING layer propagator
> does (`A22` far above 1) and `_sqrt_decay`'s `Re(lam) >= 0` branch keeps it
> off the public API (`|X| <= 1` always); it IS reachable by monkeypatching
> that branch, and one existing test does so -- see section 5.  A HIGH-Q
> CAVITY RESONANCE does it with `|X| <= 1` throughout and IS reachable: on a
> weakly modulated high-index slab whose `+-1` order is evanescent in both
> half-spaces, `cond(I - B11 A22)` reaches 1.75e+13 from `rcwa_efficiency_1d`
> alone, the closed form and the assembled star land 4.2e-04 apart and
> EQUALLY far (9.7e-04 each) from the coupled system solved whole, and the
> shipped per-order efficiency moves up to 6.4e-06 between them.  The
> `<= 1.665e-15 / 3.114e-15` envelope above is a statement about the
> well-conditioned population it was measured on, not a bound on the entry
> points.  Both regimes are documented in `_redheffer_star_rt`'s docstring and
> gated by
> `test_audit2609_b5_rcwa_eme_bor.py::test_d2_a_near_singular_star_denominator_is_reachable_and_neither_form_is_better`.

**(b) `fixes/WP-B5_REPORT.md` section 5(a), point 2 of "Why the fixture itself
does NOT have to change", and `fixes/WP-B5_CHANGELOG.md` migration bullet 1.**
Both say the regime is unreachable / that only the last bits move.  The
conclusion (the fixtures stay, the arms record the refusal) is unaffected --
the engineered arm IS unreachable and the shipped arms ARE unchanged -- but the
supporting clause needs the scope.  Requested: in the report, replace "the
regime the arm creates is unreachable from the public API" with "the regime the
arm creates -- a layer mode with `Re(lam) < 0` -- is unreachable from the public
API; the near-singular denominator it produces is reachable by other means (a
cavity resonance) and is documented in `_redheffer_star_rt`, but not with
`|A22|` far above 1, which is what makes THIS arm's disagreement O(1)".  In the
changelog, replace "A test that pins one of these to more than ~13 significant
figures will need its value re-recorded; nothing else is affected" with "A test
that pins one of these to more than ~13 significant figures will need its value
re-recorded.  The envelope is the well-conditioned population's: at a high-Q
cavity resonance, where `I - B11 A22` reaches `cond` 1e13, the two formulations
differ by up to 6.4e-06 and neither is the better one -- see
`_redheffer_star_rt`."

**(c) `fixes/WP-B5_REPORT.md` section 2, the 382-array movement table.**  The
row labelled `rcwa_jones_2d` **(+ out-of-plane)** | 84 | 38 | 1.665e-15 |
3.114e-15 cannot include `formulation='fff_nv'` on an out-of-plane cell, which
D3 moves by 1.08e-03 (measured here).  Requested: rename the row to
`rcwa_jones_2d` (out-of-plane cell, `'laurent'` / `'li'`) or add "(`'fff_nv'`
on an out-of-plane cell is D3's intended change -- see the migration note)".

**(d) `fixes/WP-B5_REPORT.md` section 2, D3 "Residual risk".**  "an
out-of-plane `fff_nv` build is measurably more expensive at large `n_orders`"
has the direction backwards: measured 7.7 % / 3.7 % / 2.9 % of the whole solve
at M = 4 / 6 / 8 (operator 5.9 -> 13.1 / 26.2 -> 40.6 / 57.0 -> 119.7 ms
against solves of 171 / 1112 / 4060 ms).  Requested: "doubles the operator
build, which is 7.7 % of the solve at `n_orders` 4 and falls to 2.9 % by 8 --
the eigensolve outgrows it".

**(e) One correction of record, requested in no file.**  WP-B5's section 5(c)
says WP-A14's outstanding request -- `threadpoolctl` as a hard dependency -- "is
still open and ... still not installed on this box (the H5 warning fires in the
suite by design)".  It is CLOSED: `pyproject.toml:97` and `requirements.txt:32`
both carry `threadpoolctl>=3.1`, version 3.6.0 is installed here, and the
selection's skip list now reads `threadpoolctl installed: the cap is effective
here` (`test_niche_audit_m4_m5_m6_rcwa.py:389`).  Nothing to change in WP-B5's
files -- the statement was true when written -- but the next reader should not
carry it forward.

**(f) Nothing else.**  No change is requested in `pmm/`, `berreman.py`,
`rcwa/stack.py`, `pyproject.toml` or any propagator / lens module.

---

## 6. Follow-up

**B5-D2c (new) -- the star denominator has no instrument that can see a
resonance.**  `rcond_refuse` is `None` at both star sites, which is right (the
report proves the refusal path is untouched), but the reason it would not help
is worth recording: over sixteen located resonances reaching
`cond(I - B11 A22) = 1.14e+13` the site's own `_rcond_1_equilibrated` reads
0.040 .. 0.489 and `_equilibrated_inverse_residual` reads
6.3e-17 .. 4.6e-16.  Equilibration is
exactly the right scaling for judging the INVERSE, and the inverse is fine; the
amplification is in the answer.  A guard for this class would have to read
`min |1 - eig(B11 A22)|` (or the smallest singular value of `I - B11 A22`
UNequilibrated), which costs an eigensolve or an SVD per star -- so it is a
design question, not a patch.  The population it would have to separate is
sketched in section 2: closure defects from 4.6e-09 to 1.4e-02 as the
modulation falls from 1e-2 to 1e-6, all under the 1.05 gross tripwire.

**B5-D1b (new) -- the `cond([[1/eps]])` gate's bar is a metal fact.**
`test_d1_the_inverse_rule_toeplitz_is_not_the_ill_conditioned_matrix` asserts
`cond < 1e6` over four metals, which is right and has 3.8 decades of margin
(worst measured 1.60e+02 on Al / Cu).  The conditioning is the permittivity
ratio across the binary step, so the bar is reached exactly at an
epsilon-near-zero ridge (`n = 1e-3` reads 1.00e+06, `n = 1e-6` reads 1.00e+12,
with the inverse's equilibrated residual at 5.1e-04 there).  If the gate is
ever widened past metals, the bar has to be re-derived from the ratio.

**B5-D1a, B5-D2a, B5-D2b (the WP's own).**  Unchanged by this pass and still
worth what the report says.  One number to carry into B5-D2b if it is taken:
the subnormal tail is what makes the `n_orders = 400` speed-up superlinear, and
my interleaved measurement at 50 / 100 / 200 (1.10x / 1.12x / 1.22x under
contention) is consistent with that being the whole of the extra factor at 400.

**D4 / D5 (WP-A14).**  Untouched, as the brief directs.

**Not reproducible: none.**  Every claim above was either reproduced or
falsified with numbers; nothing failed to reproduce for want of a fixture.
