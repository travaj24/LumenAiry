# VERIFY -- round 2 of the PURE staggered 2-D PMM per-layer grids (L2 mortar)

**Date** 2026-09-11 · **Worktree** `C:/tmp/lum_vmortar2`, branch
`verify/mortar-round2`, HEAD `fbf493c` (the integration branch `wave2/pmm2d`
merged in during the audit; its library delta over the `8be752b` the audit
started on is `lumenairy/elements/_lens_traced.py` ALONE -- nothing on the PMM
path, so every measurement below stands) · **`without` arm** `24651c8`
(`C:/tmp/lum_prem2`, read-only, removed at the end)

**Under verification** `docs/audits/FIX_PMM2D_MORTAR_ROUND2_2026_09_11.md`
(D1 / D2 / D3) and the merged tree it now shares with the 1-D sliver round-2
arbiter (`docs/audits/FIX_PMMSTACK_SLIVER_WALLS_ROUND2_2026_09_11.md`).

**Method** every number below was RE-MEASURED with my own scripts and my own
fixtures (`validation/probe_verify_mortar_round2/`, eight probes + a README);
none is read from the fix doc.  Every script asserts which `lumenairy` it
imported.  Both directions: claimed successes were attacked, and the claimed
limitations were prosecuted for whether they are worse than stated.

**Binding** `docs/TESTING_STANDARDS.md`.

---

## S0. Verdicts

| # | claim | verdict | key numbers (mine) |
|---|---|---|---|
| C1 | the merged tree is green | **CONFIRMED** | WIN 50 files **1033 passed / 1 skipped / 0 failed** (3535 s); WSL 16 files **390 passed / 0 failed** (1583 s); the twelve required suites collect **310**, matching |
| C2 | bit-identity vs `24651c8` on shared, per-layer, 1-D and the integer projector | **CONFIRMED** | 30 fixtures / **165 hashes, 165 identical, 0 warning-set differences** |
| C3 | `lu_factor`+`lu_solve` is bit-identical to `np.linalg.solve` | **CONFIRMED** | **60/60** solves (35 REAL mortar operands + 25 synthetic incl. near- and exactly-singular); cost **1.00x** WIN / 1.09x WSL |
| C4 | D1 is a FLOOR under the `n_modes` ladder, not a wander | **CONFIRMED** | 7->8 rung collapses **10.33x -> 3.05x**; `M`=8 error saturates 5.23e-04 -> 3.07e-03 (**5.9x**) |
| C5 | the floor is ENERGY-INVISIBLE, closure pinned | **CONFIRMED** | closure **3.87e-07 at every rung of every `delta`**, 5.1 decades under `_STAG_CLOSURE_TOL`; **0 warnings** on 45 solves |
| C6 | the mortar's ALGEBRA is exact: all-host layers on three grids = the analytic slab | **CONFIRMED** | err_s **2.7e-12 .. 6.8e-12 at `M`=6 at every wall separation**, flat in `delta` within **2.5x**; closure 2.3e-14 .. 5.6e-14 |
| C7 | `\|gamma\|/k0 = c(M) M(M+1)/(4 k0 J)` with `c` = 0.9165 / 0.9536 | **CONFIRMED, and it is UNIVERSAL** | reproduced to **4 digits** on MY fixture AND on BOTH of the builder's; stable over 3 decades of `delta` |
| C8 | a SPECTRAL bar would refuse fine uniform lattices | **CONFIRMED** | uniform `N`=8 (0.125 d) reads **15.74** while a `delta`=1e-1 sliver reads **11.67** -- the populations CROSS |
| C9 | exponents: `W_b` flat, `V_b` ~2, E row ~2, H row > 2 | **CONFIRMED / RESTATED** | mine (sliver on ONE axis): `W_b` **-0.003 / -0.006**, `V_b` **1.892 / 1.872**, E row **1.01 / 1.006**, H row (sliver in the `A` slot) **2.30 / 2.06**, `\|gamma\|` **0.998 / 0.999`.  The builder's E-row 2.0 is the TWO-AXIS value; the exponent is **1.0 per axis carrying the sliver** |
| C10 | the 1e-3 width bar has 2.10 decades to the narrowest ordinary geometry | **CONFIRMED** | my own census: worst ordinary **1.250e-01** (nested refinement) = **125x** the bar |
| C11 | the 1e-9 slack: exactly 1e-3 accepted, 1e-3 - 2e-9 refused | **CONFIRMED** | and the slack is RELATIVE: `1e-3*(1-5e-10)` accepted, `1e-3*(1-2e-9)` refused; the shipped `(0.28572-0.28452)/1.2` = 9.99999999999982457e-04 (1.756e-17 under) is ACCEPTED |
| C12 | the integer path is exempt and cannot reach the bar | **CONFIRMED** | `Basis1D(1.0, 2000, 3)` builds; the same lattice as an ARRAY is refused at segment 576; `N`=1001 at `M`=4 is a 1.804e+07 region eig |
| C13 | the shared path refuses `x_walls` / `y_walls` | **CONFIRMED** | both refused by name |
| C14 | the closing taper crosses "at about 250 slices" | **CONFIRMED to the slice** | 250 -> 1.0000e-03 ACCEPTED, **256 -> 9.7656e-04 REFUSED** |
| C15 | above the bar the cost is CONTINUOUS and NOT restored (1.0..5.0x) | **CONFIRMED, and 5.9x on my fixture** | `M`=8: 1.00 / 2.09 / 4.02 / 5.31 / **5.69** / 5.82 / **5.86x** at `delta` = 3e-1 / 1e-1 / 3e-2 / 1e-2 / 4.1e-3 / 2e-3 / 1e-3, all SILENT |
| C16 | D2 healthy population 2.6e-07 .. 3.8e-04, bar 5.4 decades below | **REFUTED as stated** | my own healthy population reaches **1.502e-11** and the population of the THIRD guarded site runs **2.64e-14 .. 1.48e-04** -- see **DEFECT V1** |
| C17 | the conditioning backstop independently refuses the sliver | **REFUTED as a general claim** | on a 2-layer stack whose sliver is in the LAST layer and on ONE axis, NOTHING is refused down to `delta` = **1e-7** (`rcond` ~1.1e-11) |
| C18 | `LinAlgWarning` in the `except` tuple keeps the named refusal under `-W error` | **CONFIRMED** | `_ConditioningError` on BOTH builds under both warning filters |
| C19 | the `~1850` plain 1-D site is rightly left unguarded | **CONFIRMED** | correct population's closest approach **1.948e-10** = 2.29 decades (builder: 1.0); every reading below is refused by the shipped 1-D guard at `rcond` = 1.951e-12; and the WIN/WSL spread on those readings is **8 %** |
| C20 | D3 `nq` rule: slope predictor, envelope, 720/720, 457/720 | **CONFIRMED** | slope **0.6293 .. 0.6366 (1.2 % spread)**, intercept `9.30 + 0.431 M`; shipped `(0.72, 0.5, 10.0)` is an upper envelope; **720/720**; candidate **457/720** exactly |
| C21 | D3 kernel error 7.5e-04 -> 4.7e-15 | **CONFIRMED (AFTER), fixture-dependent (BEFORE)** | AFTER **4.68e-15** (matches); BEFORE on my grid **1.29e-03**; at the ENFORCED order cap the reachable BEFORE is **1.15e-04** (0.96 d, `M`=8) -> **1.01e-14** |
| C22 | D3 moves no ordinary per-layer answer | **CONFIRMED (to round-off)** | `nq` moves on 21 of 30 (geometry, `M`) cells at the enforced cap, but the far field moves **<= 5.6e-15 absolute / 5.0e-14 relative** with vs pre |
| C23 | the durability of the 15 gates | **CONFIRMED with one structural hole** | every constant re-measured; the D2 population gate never builds a stack that reaches the third guarded site -- which is exactly where the bar is wrong |

**DEFECT V1 (P1, ship-blocking) -- the D2 `rcond` bar refuses ORDINARY
out-of-plane per-layer stacks.**  Details in S6.3.

---

## S1. The two builds

| | Windows (WIN) | WSL (Ubuntu) |
|---|---|---|
| python | 3.14.6 | 3.12.3 |
| numpy | 2.4.4 | 2.4.6 |
| scipy | 1.17.1 (scipy-openblas) | 1.17.1 (scipy-openblas) |
| threads | `OMP/OPENBLAS/MKL_NUM_THREADS = 1` | same |
| `lumenairy` | `C:\tmp\lum_vmortar2\lumenairy\__init__.py` v5.44.0 | `/mnt/c/tmp/lum_vmortar2/...` v5.44.0 |

Same BLAS family on both, so every cross-build spread here is a LOWER bound --
the same caveat the build, the fix and the verification carry.  The one place
that caveat bit is S6.4: the plain 1-D `rcond` readings differ by **8 %**
between the builds, which is a lot for a quantity a 1-decade bar would sit on.

---

## S2. THE COMBINED TREE (task 1)

### 2.1 Windows, one thread, the union of the required suites and every
`PMMStack` importer

`grep -rl 'PMMStack' --include='*.py' tests/` returns **43** files at
`8be752b` (the builder's 42 plus `test_fix_pmmstack_sliver_walls_round2.py`,
which arrived with the 1-D arbiter).  Union with the twelve required suites =
**50 files**.

```
1033 passed, 1 skipped, 0 failed, 101 warnings in 3535.58s (0:58:55)
```

The single skip is `test_niche_audit_m4_m5_m6_rcwa.py:387`
("threadpoolctl installed: the cap is effective here") -- an environment
branch, not a masked test.  **Zero failures: no merge defect.**

The twelve required suites collect **310** tests, matching the fix doc's 310
exactly:

| file | tests |
|---|---|
| `test_fix_pmm2d_mortar_round2.py` | 15 |
| `test_pmm2d_staggered_mortar.py` | 31 |
| `test_pmm2d_staggered_nonuniform.py` | 16 |
| `test_verify_pmm2d_perlayer_slant.py` | 4 |
| `test_pmm2d_staggered_slant.py` | 70 |
| `test_pmm2d_staggered_oop.py` | 35 |
| `test_pmm2d_staggered_anisotropic.py` | 41 |
| `test_pmm2d_staggered_magnetic.py` | 35 |
| `test_v5_12_0_pmm2d_staggered.py` | 13 |
| `test_v5_21_pmm2d_staggered_oblique.py` | 8 |
| `test_v5_14_0_pmm2d_stack.py` | 8 |
| `test_p2c_pmm2d_stack_cascade.py` | 34 |
| **total** | **310** |

### 2.2 WSL

The twelve suites + the three contract-touching files + the arbiter's own
round-2 file:

```
390 passed, 0 failed, 27 warnings in 1583.22s (0:26:23)
```

(310 + 15 + 19 + 27 + 19 = 390.)

### 2.3 After the later `wave2/pmm2d` merge

`fbf493c` brings `tests/unit/test_verify_pmmstack_sliver_round2.py` (new, 450
lines) and edits to `test_fix_pmmstack_sliver_walls_round2.py`.  Re-run on WIN
together with the restated `test_verify_pmmstack_sliver_walls.py`:

```
40 passed, 0 failed in 55.15s
```

**No expectation from either branch fails against the other's code.**  The two
branches touch the same file (`test_verify_pmmstack_sliver_walls.py`), and the
one gate whose premise the width contract supersedes
(`test_the_mortar_carries_a_within_layer_sliver_without_a_silent_wrong_answer`)
is restated and passes on both builds.

---

## S3. BIT-IDENTITY vs `24651c8` (task 2)

`v1_bitid.py` + `v1_compare.py`.  MY OWN fixtures, sha256 over
`dtype|shape|tobytes` of every returned array, plus the WARNING SET.

| family | fixtures | hashes |
|---|---|---|
| SHARED grid: scalar 3-layer, scalar at normal incidence, in-plane tensor, OUT-OF-PLANE tensor, magnetic (`mu_cell`), slanted, `retain_internal` + `layer_absorption`, `per_order_amplitudes` (both ports, 24 arrays), `jones=False`, `M`=6/`N`=4 | 10 | 59 |
| PER-LAYER, ordinary non-uniform walls: conforming, non-conforming, nested (5 segments), mixed uniform/patterned, per-layer `n_modes`, `add_tapered_pillar` (midpoint AND `rule='bottom'`), `add_tapered_pillars`, out-of-plane tensor, slant, magnetic, `retain_internal`, uniform-ARRAY spelling, INTEGER spelling | 14 | 58 |
| 1-D `PMMStack`: shared, per-layer, conical, slanted | 4 | 16 |
| the PROJECTOR directly, 8 `(d, N, M, alpha0, m_max)` points on both global stencil sets: INTEGER-`N` and the same lattices as explicit uniform ARRAYS | 2 | 32 |
| **TOTAL** | **30** | **165** |

```
IDENTICAL 165 / 165 hashes; 0 mismatches; 0 warning-set differences
with: C:\tmp\lum_vmortar2\lumenairy\__init__.py   py3.14.6 np2.4.4
pre : C:\tmp\lum_prem2\lumenairy\__init__.py      py3.14.6 np2.4.4
```

**A trap worth recording, because I fell into it and it produces a FALSE
PASS.**  `_stag_fourier_projection` returns a CLOSURE, and
`per_order_amplitudes` returns a DICT.  `np.asarray` of either is a 0-d object
array whose `tobytes()` is a POINTER: the first pass hashed process addresses
and reported 18 "mismatches" that were memory reuse.  A harness that hashes the
wrong object can equally report 0 mismatches for no reason at all.  The fixed
arms apply the closure to `basis.B` / `basis.Btilde` and hash each dict entry.

### 3.1 The `nq` claim on integer lattices, on MY OWN set

`_stag_quad_order` returns exactly `2 M + 8` on **720/720** cells
(`M` = 3..14 x `N` = 1..60) under the order cap the library ENFORCES --
`n_orders <= (q - 1) // 2`, `q = N (M - 1)`, which `PMM2DStackPure.solve`
raises on (measured: `n_orders=7` on a 3-segment `M`=4 stack is refused by
name).  The builder's REJECTED candidate `max(2M+8, ceil(0.75 w) + M + 8)`
moves **457** of the same 720 -- the fix doc's number, digit for digit.

With MY first (wrong) cap `(M-1) N // 2` the shipped rule violates 5 cells at
`M`=3, `N`=1..3; that is my cap being wrong, not the rule.  Recorded because
the "720/720" claim is CONDITIONAL on the enforced cap and I re-derived the cap
from the library's own refusal message rather than from the fix doc.

### 3.2 The mortar solves' `lu_factor` path

`v4_d2.py bitid`.  The probe monkeypatches `_guarded_mortar_solve`, captures
the ACTUAL `(A, B)` pairs the 14 per-layer fixtures build, and re-solves each
both ways:

* **35 real mortar operands** (sizes 128, 162, 288, 324, 450, 800):
  **35/35 byte-identical**, on BOTH builds;
* **25 synthetic complex operands** 17..900 wide, including 8 driven to
  `sigma_min/sigma_max = 1e-13` and 8 made EXACTLY singular (a repeated
  column): **25/25 byte-identical** -- so the identity is not a property of
  well-conditioned inputs;
* cost on a 450x450 complex pair: `np.linalg.solve` 0.0398 s vs
  `lu_factor` + `gecon` + `lu_solve` 0.0397 s = **1.00x** (WIN); 0.0411 vs
  0.0449 = 1.09x (WSL).

**60/60 solves, 0 differing.**

---

## S4. D1 RE-DERIVED (task 3)

My fixture is deliberately NOT the builder's: `P` = 1.05 um, `wl` = 0.71,
`theta` = 0.26, `phi` = 0, `eps` = 7.5 / 2.0, `t` = 0.12, patterned neighbours
at 0.18/0.61 and 0.29/0.74, y walls 0.27/0.73 identical on all three layers
(so the ONLY non-conformity is in `x`), and the sliver centred **off-centre**
at 0.44.  Oracle: the exact 1-D `PMMStack` at degree 14, whose own 12->14
self-gap measures **1.459e-05**.

### 4.1 (a) THE FLOOR -- CONFIRMED

`v2_d1.py floor`, worst per-order deviation over `\|m\| <= 1`, `n` = 0, TE row:

| `delta` | `M`=4 | 5 | 6 | 7 | **8** | 7->8 | closure at `M`=8 |
|---|---|---|---|---|---|---|---|
| **3e-01** (ordinary) | 4.854e-02 | 4.707e-01 | 1.338e-02 | 5.400e-03 | **5.228e-04** | **10.33x** | 3.87e-07 |
| 1e-01 | 4.268e-02 | 4.074e-01 | 1.567e-02 | 6.458e-03 | **1.090e-03** | 5.92x | 3.89e-07 |
| 3e-02 | 4.120e-02 | 2.876e-01 | 1.687e-02 | 7.937e-03 | **2.101e-03** | 3.78x | 3.88e-07 |
| 1e-02 | 4.103e-02 | 2.314e-01 | 1.726e-02 | 8.763e-03 | **2.777e-03** | 3.16x | 3.87e-07 |
| 3e-03 | 4.107e-02 | 1.978e-01 | 1.765e-02 | 9.171e-03 | **3.013e-03** | 3.04x | 3.87e-07 |
| 1e-03 | 4.116e-02 | 1.766e-01 | 1.804e-02 | 9.354e-03 | **3.065e-03** | 3.05x | 3.87e-07 |
| 1e-04 | 4.136e-02 | 1.473e-01 | 1.886e-02 | 9.573e-03 | **3.032e-03** | 3.16x | 3.87e-07 |
| 1e-05 | 4.147e-02 | 1.338e-01 | 1.936e-02 | 9.691e-03 | **2.983e-03** | 3.25x | 3.87e-07 |
| 1e-06 | 4.151e-02 | 1.288e-01 | 1.958e-02 | 9.741e-03 | **2.959e-03** | 3.29x | 3.87e-07 |

* the per-rung improvement collapses **10.33x -> 3.05x** and the `M`=8 reading
  SATURATES at ~3.0e-03 while the ordinary arm's is 5.2e-04: a **5.9x floor**.
  The fix doc's 11.71x -> 2.32x and 6.3x, on a different fixture.  **CONFIRMED.**
* **CLOSURE PINNED at 3.87e-07 on every one of the 45 solves**, 5.1 decades
  under `_STAG_CLOSURE_TOL` = 5e-2, with **0 warnings** anywhere.  **CONFIRMED**
  -- the fix doc's 8.0e-08 is its fixture's value; the CLAIM is the pinning, and
  the pinning reproduces.
* **one restatement.**  "The error grows smoothly and MONOTONELY in
  `log(delta)`" holds at `M` = 6, 7, 8 and is FLAT at `M` = 4 (4.85e-02 ->
  4.15e-02); at `M` = 5 my fixture moves the OTHER WAY (4.71e-01 -> 1.29e-01).
  `M`=5 is an outlier rung on this fixture (its own convergence has not
  started), so the floor is visible only at rungs where the ordinary arm has
  converged below it.  That is a property of the instrument, not a defect, but
  "monotone" is not universal across `M`.

**ORDINARY-wall control** (the same layer count, an ordinary partition instead
of a sliver), `M`=8: 9.392e-04 / 6.060e-04 / 5.483e-04 / 8.688e-04 / 1.138e-03
at widths 0.50 / 0.30 / 0.20 / 0.12 / 0.08.  So an ordinary 0.08 partition
costs 2.2x while the sliver costs 5.9x and saturates -- the fix doc's point
that the effect is not simply "a less balanced partition is less accurate".

**NO-MORTAR control** (every layer on the sliver's own grid, so every interface
is a plain square match): `M`=4/5/6 read 0.5918/0.5750/0.5258 at `delta`=1e-3
and 0.5919/0.5755/0.5270 at 1e-6 -- **delta-independent to 1.6e-04 / 4.9e-05 /
1.2e-03 absolute over three decades**, against the mortared arm's 5.9x.  The
attribution reproduces.

### 4.2 (b) ALL-HOST LAYERS ON THREE GRIDS = THE ANALYTIC SLAB -- CONFIRMED

`v2_d1.py allhost`.  Three ALL-HOST layers on three DIFFERENT non-uniform
grids; the device is a homogeneous slab of thickness `3 t` whose reflectance is
the symmetric-slab Airy form built from the interface coefficient alone.  My
analytic formula is itself validated against the solver's own SHARED uniform
path: **err_s 9.393e-10, err_p 2.124e-09** at `M`=5.

Analytic `R_s` = 0.113240309396, `R_p` = 0.093904795505.

| `delta` | `M`=4 | 5 | 6 | 7 | closure at `M`=7 |
|---|---|---|---|---|---|
| 1e-01 | 1.152e-07 | 7.329e-10 | 2.725e-12 | 3.511e-14 | 5.1e-14 |
| 3e-02 | 1.737e-07 | 1.149e-09 | 5.609e-12 | 5.880e-14 | 4.8e-14 |
| 1e-02 | 1.920e-07 | 1.208e-09 | 6.574e-12 | 6.295e-14 | 5.6e-14 |
| 3e-03 | 1.963e-07 | 1.135e-09 | 6.776e-12 | 1.009e-13 | 4.6e-14 |
| 1e-03 | 1.951e-07 | 1.028e-09 | 6.739e-12 | 8.002e-14 | 5.4e-14 |
| 1e-04 | 1.884e-07 | 8.174e-10 | 6.380e-12 | 1.674e-14 | 2.3e-14 |

**The mortar reproduces the analytic slab to 6.8e-12 at `M`=6 at every wall
separation, flat in `delta` within a factor 2.5** (the fix doc: 9.6e-13 and a
factor 3.3, on its own fixture).  The mortar's ALGEBRA is not what D1 breaks.
**CONFIRMED.**

### 4.3 (c) THE SPECTRAL PREDICTOR -- CONFIRMED, and it is UNIVERSAL

`v2_d1.py spectrum`.  `\|gamma\|/k0 = c(M) M (M+1) / (4 k0 J)`:

| `delta` | `k0 J_min` | `\|gamma\|/k0`, `M`=4 | `c`, `M`=4 | `c`, `M`=6 |
|---|---|---|---|---|
| 3e-01 | 1.347e+00 | 5.822e+00 | 1.5688 | 1.6324 |
| 1e-01 | 4.646e-01 | 1.167e+01 | 1.0840 | 1.1134 |
| 3e-02 | 1.394e-01 | 3.466e+01 | 0.9662 | 0.9974 |
| 1e-02 | 4.646e-02 | 1.004e+02 | 0.9332 | 0.9679 |
| 1e-03 | 4.646e-03 | 9.882e+02 | 0.9182 | 0.9550 |
| 1e-04 | 4.646e-04 | 9.865e+03 | **0.9167** | **0.9537** |
| 1e-05 | 4.646e-05 | 9.864e+04 | **0.9165** | **0.9536** |
| 1e-06 | 4.646e-06 | 9.864e+05 | **0.9165** | **0.9536** |

**0.9165 / 0.9536 -- the fix doc's constants to four digits, on a fixture with
a different period, wavelength, angle, contrast and an off-centre sliver.**
And re-measured on BOTH of the builder's fixtures as a cross-check, all three
agree to four digits:

| fixture | `c`(`M`=4) | `c`(`M`=6) |
|---|---|---|
| mine (1.05 / 0.71 / eps_h 2.0) | 0.9165 | 0.9536 |
| builder `alt` (1.2 / 0.85 / 2.25) | 0.9165 | 0.9536 |
| builder `r5` (0.9 / 0.60 / 2.25) | 0.9165 | 0.9536 |

`c` is a UNIVERSAL constant of the basis, not a fixture reading.  That is a
STRONGER result than the fix doc claims, and it is what makes the hard-coded
`0.93` in `_raise_stag_sliver`'s message defensible.

**The uniform-lattice overlap that kills a spectral bar -- CONFIRMED, and
harder than stated.**  Physical ceiling `sqrt(eps_max)` = 2.74.

| grid | `\|gamma\|/k0` at `M`=4 |
|---|---|
| uniform `N`=3 (segments 0.333 d) | 5.447 |
| uniform `N`=5 (0.200 d) | **9.601** |
| uniform `N`=8 (0.125 d) | **15.738** |
| sliver `delta` = 1e-1 (0.100 d) | **11.665** |
| sliver `delta` = 3e-2 | 34.661 |

A uniform `N`=8 lattice reads HIGHER than a `delta`=1e-1 sliver on a
1.3x FINER segment.  The two populations do not merely approach -- they
**cross**.  Any spectral bar between them refuses a uniform lattice for being
fine.  (The fix doc's own reading, 9.84 vs 39.8 at 6x, is the same argument
from a weaker angle.)

### 4.4 (d) THE EXPONENTS -- CONFIRMED, with the per-axis restatement

`v2_d1.py exponents`, fitted in `1/delta` over the tail `delta <= 1e-2`, my
fixture (sliver on ONE axis, y identical on all layers) vs the fix doc's (sliver
on BOTH axes):

| operator | mine `M`=4 / `M`=6 | fix doc (two axes) |
|---|---|---|
| `\|gamma\|max` of the sliver grid | **0.998 / 0.999** | 0.999 |
| `W_b` (the E-field modal matrix) | **-0.003 / -0.006 (FLAT)** | 0.029 / -0.075 (FLAT) |
| `V_b` (the H-field modal matrix) | **1.892 / 1.872** | 2.028 / 1.950 |
| `MassE_B W_B` (the E row) | **1.010 / 1.006** | 1.995 / 1.994 |
| `MassH_?` with the SLIVER in that slot (the H row) | **2.300 / 2.055** | 3.218 / 3.003 |
| cross-mass `C1_x` | 2.939 / 1.224 | 1.025 / 1.284 |
| cross-mass `C2_x` | 2.000 / 2.316 | 2.303 / 2.200 |

* `W_b` FLAT and `\|gamma\|` exactly `1/w`: **CONFIRMED on both fixtures.**
* `V_b` ~ 2: **CONFIRMED** (1.87-1.89 vs 1.95-2.03).
* **The E row is 1.0 PER AXIS carrying the sliver.**  My one-axis fixture reads
  1.01; the fix doc's two-axis fixture reads 1.995 -- exactly twice.  The "~2.0"
  in the fix doc is therefore right for its fixture and should be stated as
  `1.0 x (number of axes carrying the sliver)`.  This is a **correction to the
  doc's wording, not to its arithmetic**.
* The H row is FASTER than the E row on the same slot on both fixtures (2.06-2.30
  vs 1.01 for me; 3.0-3.2 vs 2.0 for the doc), which is the claim it supports.
  The exact exponent is **BOUNDED, not pinned**: the doc's own last rung sits at
  the float64 ceiling and mine is a different axis count.
* the cross-mass exponents are NOT reliable on either fixture: mine reach
  3.6e+18 (saturated) and disagree between `M`=4 and `M`=6 by 2.4x.  Nothing in
  the fix rests on them; they should be marked saturated in the doc.
* **A NEW observation with consequences (S6.2).**  With the sliver in grid `B`,
  the operator the shipped guard screens at the `MassH_A V_A` site is
  `delta`-INDEPENDENT: measured **1.384e+05 at every `delta` from 3e-1 to
  1e-6**.  The H-row screen sees a sliver only at the interface where the
  sliver's grid is in the `A` slot.

---

## S5. THE WIDTH GUARD -- BOTH BREAK ATTEMPTS

`v3_guard.py`.  All census numbers are read off the shipped `_STAG_SEG_CENSUS`
hook, i.e. off what `Basis1D` actually receives.

**A surface note first.**  `add_layer` only RECORDS the wall array; the contract
is reached at `solve()` time.  Measured: a 1e-4 grid is `add_layer` ACCEPTED and
`solve` REFUSED.  That is what the shipped gate means by "the public surface",
and it is worth stating in the docstring, because a user building a 400-slice
taper pays the whole build before the refusal.

### 5.1 The false-positive census -- my own geometries

| geometry | narrowest / period | x the bar | refused |
|---|---|---|---|
| single interior wall at 0.4 | 4.0000e-01 | 400.0 | no |
| duty-1/3 pair | 3.3333e-01 | 333.3 | no |
| conforming 0.2371/0.6183 | 2.3710e-01 | 237.1 | no |
| non-conforming 0.3117/0.7402 | 2.5980e-01 | 259.8 | no |
| axes carrying different walls | 2.1000e-01 | 210.0 | no |
| `add_tapered_pillar`, 4 / 8 / 16 / 32 / 64 slices | 2.3375e-01 .. 2.2086e-01 | 233.7 .. 220.9 | no |
| `add_tapered_pillars`, 4 / 6 / 16 slices | 1.3750e-01 .. 1.0937e-01 | 137.5 .. 109.4 | no |
| **nested refinement 0.125/0.25/0.75/0.875** | **1.2500e-01** | **125.0** | no |
| pillar edge 0.002 from the period end | 2.0000e-03 | 2.0 | no |
| pillar edge 0.0011 | 1.1000e-03 | 1.1 | no |
| pillar edge 0.0010 | 1.0000e-03 | 1.0 | no |
| pillar edge 0.0009 | 9.0000e-04 | 0.9 | **YES** |

**2.10 decades to the worst ORDINARY geometry: CONFIRMED** (1.250e-01 = 125x,
the same worst case and the same number the fix doc states).  Identical to the
digit on WSL.

**CLOSING TAPER (a pillar closing from half the period to a point):**

| `n_slices` | 4 | 8 | 32 | 64 | 128 | 200 | **250** | **256** | 300 | 512 |
|---|---|---|---|---|---|---|---|---|---|---|
| narrowest / period | 6.25e-02 | 3.13e-02 | 7.81e-03 | 3.91e-03 | 1.95e-03 | 1.25e-03 | **1.000e-03** | **9.766e-04** | 8.33e-04 | 4.88e-04 |
| refused | no | no | no | no | no | no | **no** | **YES** | YES | YES |

**The crossing is between 250 and 256 slices -- the fix doc's "about 250",
confirmed to the slice, on both builds.**

### 5.2 BREAK ATTEMPT 1 -- a legitimate geometry the bar REFUSES

Three classes found.  Only the third is a real problem.

1. **The closing taper past 250 slices.**  Documented by the fix, named in the
   message ("lower n_slices or stop the taper before its tip closes"), and the
   refusal names the offending slice.  Refusing is RIGHT: at 256 slices the
   sampled tip is 0.98 nm on a 1 um period, below anything the method resolves.
   The message is genuinely actionable -- I reproduced it in full and it names
   the segment index, the absolute width, the fraction, the bar, the whole wall
   array, the spurious `\|gamma\|` in units of `G`, and four remedies, and it
   explicitly does NOT offer "raise n_modes".

2. **A thin ridge below 1e-3 of the period.**  On a 1 um period the refused
   widths are 0.8 nm and below.  Refusing is RIGHT.  **But the bar is a pure
   PERIOD FRACTION with no wavelength in it**, so on a 40 um period (a DOE or a
   large metasurface supercell) the same bar refuses a **40 nm** feature.  The
   conditioning that motivates the bar really is governed by `w/d` (it is a
   cross-grid projection on one period), so this is defensible -- but it is not
   stated anywhere, and a user with a large period will meet it.  **Flagged as
   a documentation gap, not a defect.**  The remedy the message offers is
   itself arithmetic-checkable and I did check it: "carry the fine feature on
   the SHARED lattice with an `N` that resolves it" needs `N >= 500 / 1000 /
   2000 / 10000` for features of 2e-3 / 1e-3 / 5e-4 / 1e-4 -- and the shared
   lattice's own segments are then `d/N`, i.e. the SAME width the per-layer path
   was refused for.  The remedy works (a uniform lattice's mortar is the
   identity), but the message reads as though the width itself were the
   problem, which it is not.

3. **A CONFORMING per-layer stack -- the sharpest false positive.**  The fix's
   own attribution (S2.4) is that the mortar's algebra is exact and what a
   degenerate grid loses is a PATTERNED NEIGHBOUR's trace across a CROSS-GRID
   projection.  A per-layer stack whose layers all carry the SAME wall array has
   no cross-grid projection at all -- every interface is the plain square modal
   match.  `Basis1D` cannot know that, so it refuses anyway.  Measured
   (`v3_guard.py nomortar_fp`, guards lifted): such a stack's answer moves by
   **1.5e-05 / 4.9e-05 / 1.2e-04** (at `M` = 4 / 5 / 6) between `delta` = 1e-4
   and 1e-6, i.e. it is `delta`-independent to ~1e-4 while the mortared stack
   moves 5.9x.  **The refusal of a conforming per-layer stack is a false
   positive on a whole class.**  It is mild (the class is rare, and the user's
   fix is one keyword), and the fix doc already met its shipped instance
   (S3.2b, the single-layer sliver-verify gate).  **Severity P3**, logged as
   DEFECT V2.

### 5.3 The 1e-9 slack, the switch, the exemption, the shared path

| request | actual fraction | vs bar | outcome |
|---|---|---|---|
| exactly 1e-3 | 9.9999999999999568e-04 | -4.34e-18 | **ACCEPTED** |
| `1e-3 * (1 - 5e-10)` | 9.9999999950000805e-04 | -5.00e-13 | ACCEPTED |
| `1e-3 * (1 - 1e-9)` | 9.9999999899999395e-04 | -1.00e-12 | REFUSED |
| `1e-3 * (1 - 2e-9)` | 9.9999999799999222e-04 | -2.00e-12 | REFUSED |
| `1e-3 - 2e-9` (ABSOLUTE) | 9.9999799999998850e-04 | -2.00e-09 | **REFUSED** |
| 1.001e-3 | 1.0009999999999969e-03 | +1.00e-06 | ACCEPTED |
| the SHIPPED fixture `(0.28572-0.28452)/1.2` | 9.99999999999982457e-04 | **-1.756e-17** | **ACCEPTED** |

**CONFIRMED, with the clarification that the slack is RELATIVE (1e-9 relative =
1e-12 absolute at this bar).**  The at-threshold defect the fix found is real:
the shipped fixture's own arithmetic lands 1.756e-17 strictly below 1e-3, and
without the slack it would be a coin flip.  With it, it is deterministic.

`PMM2D_STAG_MIN_SEG_GUARD` = `True` refuses a 2e-4 grid by name; `False`
accepts it and returns `N`=3.  Identical on both builds.

INTEGER exemption: `Basis1D(1.0, 2000, 3)` builds (`uniform=True`, segments
5.0e-04 -- 0.3 decades UNDER the bar); the SAME lattice spelled as an ARRAY is
refused at segment 576.  Reaching the bar on the integer path needs `N > 1000`,
i.e. `N`=1001 at `M`=4 -> `q` = 3003 and a **1.804e+07**-dimension region eig.
**CONFIRMED, and the exemption costs nothing.**

SHARED path: `x_walls` and `y_walls` are both refused by name; the shared
lattice is `linspace(0, d, N+1)` at every `N` checked (3/5/8/13/24/100).
**CONFIRMED -- there is no API path to a degenerate shared cell.**

### 5.4 BREAK ATTEMPT 2 -- the DEGRADED ANSWER above the bar (the load-bearing
question)

Guard ARMED throughout; every row below is an answer the SHIPPED library
RETURNS.  Scored against the exact 1-D oracle on the device that CANNOT depend
on the wall separation.

| narrowest segment | `M`=6 err | x ordinary | `M`=8 err | **x ordinary** | closure | warnings |
|---|---|---|---|---|---|---|
| 3e-01 (ordinary) | 1.3379e-02 | 1.00 | 5.2275e-04 | **1.00** | 3.87e-07 | 0 |
| 1e-01 | 1.5673e-02 | 1.17 | 1.0904e-03 | 2.09 | 3.89e-07 | 0 |
| 3e-02 | 1.6870e-02 | 1.26 | 2.1013e-03 | 4.02 | 3.88e-07 | 0 |
| 1e-02 | 1.7256e-02 | 1.29 | 2.7766e-03 | 5.31 | 3.87e-07 | 0 |
| **4.1e-03** (the 64-slice closing taper) | 1.7541e-02 | 1.31 | 2.9770e-03 | **5.69** | 3.87e-07 | 0 |
| 3e-03 | 1.7647e-02 | 1.32 | 3.0129e-03 | 5.76 | 3.87e-07 | 0 |
| **2e-03** (a user just above the bar) | 1.7789e-02 | 1.33 | 3.0430e-03 | **5.82** | 3.87e-07 | 0 |
| 1.5e-03 | 1.7893e-02 | 1.34 | 3.0558e-03 | 5.85 | 3.87e-07 | 0 |
| 1.1e-03 | 1.8008e-02 | 1.35 | 3.0636e-03 | 5.86 | 3.87e-07 | 0 |
| **1.0e-03** (AT the bar) | 1.8044e-02 | 1.35 | 3.0650e-03 | **5.86** | 3.87e-07 | 0 |

**ANSWER: a user at 2e-3 gets a SILENTLY DEGRADED answer, not one within the
physical shift.**  On this fixture the device is provably `delta`-independent,
so 5.82x is pure numerical damage; the closure is pinned at 3.87e-07, five
decades from `_STAG_CLOSURE_TOL`, and there is **no warning of any kind**.
Raising `n_modes` does not remove it -- that is what "floor" means, and the
ladder itself flattens, so a user converging in `n_modes` would read the
flattening as convergence.

The fix doc states this cost plainly (S3.2a, open item C) and quantifies it as
1.0..5.0x; I measure **1.0..5.9x** and confirm it is continuous, unrefused and
unwarned.  **The claim is CONFIRMED; the CONSEQUENCE is worse than a reader of
"the bar does not restore accuracy above 1e-3" is likely to assume**, because
the whole band from 1e-2 down to the bar already carries 5.3-5.9x and the
engine's own tripwires cannot see it.

**Recommendation (not shipped, not a defect):** emit a `UserWarning` -- not a
refusal -- for a narrowest segment between the bar and ~3e-2, naming the
measured floor and saying that `n_modes` will not remove it.  The census hook
that makes it cheap already exists.

---

## S6. D2 RE-MEASURED (task 4)

### 6.1 The healthy population -- REFUTED as stated

`v4_d2.py pop`, over MY OWN 14 per-layer fixtures (non-conforming, conforming,
2 %-fine features, the taper, a mixed 1-/3-/5-segment stack and an out-of-plane
stack, at `M` = 4, 5, 6), read off the shipped `_MORTAR_SOLVE_CENSUS`:

| site | solves | `rcond` range |
|---|---|---|
| `... mortar interface (MassE_B W_B)` | 17 | 3.957e-06 .. 1.078e-04 |
| `... mortar interface (MassH_A V_A)` | 17 | 3.178e-08 .. 4.596e-05 |
| **`... GENERALIZED mortar interface`** | 1 | **1.502e-11** |
| whole population | 35 | **1.502e-11 .. 1.078e-04** |

WIN and WSL agree on every one of those to **five significant figures**
(1.5021e-11 / 1.0783e-04 on both).  The IN-PLANE pair reproduces the fix doc's
population (its 2.61e-07 floor vs my 3.18e-08 -- both comfortably 4+ decades
above the bar).  **The GENERALIZED site does not**, and that is DEFECT V1.

Modal ladder on the shipped taper (worst `rcond` over 6 solves): **1.647e-05
(`M`=4) / 7.359e-07 (6) / 4.664e-07 (7) / 1.419e-07 (8)** -- the fix doc's
1.62e-05 / 5.98e-07 / 2.64e-07 / 4.67e-08, same shape, mine slightly higher.
**Still 4.9 decades clear at `M`=8 on the IN-PLANE pair.**

### 6.2 The refusal at `delta` = 1e-7 -- REFUTED as a general claim

`v4_d2.py refuse`: a two-layer stack, patterned first layer, all-host second
layer whose two walls sit `delta` apart on the `x` axis only, `M`=6, WIDTH
contract lifted:

| `delta` | 1e-3 | 1e-4 | 1e-5 | 1e-7 |
|---|---|---|---|---|
| outcome | RETURNED | RETURNED | RETURNED | **RETURNED** |

**Nothing is refused, on either build.**  The reason is S4.4's new observation:
the `MassH_A V_A` operator is built from the FIRST grid, so with the sliver in
the LAST layer it never enters the `A` slot, and only the E row sees the sliver
-- at exponent 1.0 per axis, with the sliver on one axis.  Measured on the
matching fixture, `rcond(MassE_B W_B)` at `M`=6 runs 1.678e-05 / 1.373e-07 /
1.420e-08 / 1.275e-09 / **1.116e-10** at `delta` = 3e-1 / 1e-3 / 1e-4 / 1e-5 /
1e-6, i.e. still **2 decades above the bar at `delta` = 1e-6**.

The fix doc's "independently by the CONDITIONING backstop ... (`rcond` =
3.168e-15 on that fixture)" is TRUE ON ITS FIXTURE (sliver on BOTH axes,
patterned neighbours on both sides, so the sliver's grid does occupy the `A`
slot).  As a general statement -- "the backstop earns its keep above `M ~ 6`
where a fixed width contract cannot follow" -- it is **REFUTED**: on a one-axis
sliver in the last layer the backstop is 2-4 decades from firing at every
`delta` a user can reach.  **This does not weaken the round-2 fix** (the WIDTH
contract fires on all of these); it weakens only the claim that the backstop is
an independent second line.  **Severity P3 (documentation), logged as DEFECT V3.**

### 6.3 **DEFECT V1 (P1, SHIP-BLOCKING)** -- the bar refuses ordinary
out-of-plane per-layer stacks

`v7_generalized.py`, `v8_oop_regression.py`.

`_guarded_mortar_solve` is wired at THREE sites.  The bar `1e-12` was
calibrated on the IN-PLANE pair.  The THIRD site,
`_interface_smatrix_general_mortar_2d`, is the `4 qq x 4 qq` block solve an
OUT-OF-PLANE tensor or a SLANTED per-layer layer takes, and its healthy
population is several decades lower.  Measured over 24 ORDINARY, healthy
per-layer stacks (narrowest segment **0.237 of the period = 237x the width
bar**, no sliver anywhere), `n_orders`=2:

| stack | `M`=4 | `M`=5 | `M`=6 | `M`=7 |
|---|---|---|---|---|
| BOTH layers out-of-plane | 1.32e-04 | 6.53e-06 | 1.26e-06 | 5.77e-07 |
| BOTH layers slanted | 1.07e-04 | 5.04e-06 | 1.01e-06 | 3.28e-07 |
| **ONE out-of-plane layer + ONE ordinary SCALAR layer** | **1.41e-11** | **8.40e-13** | **6.05e-14** | **2.64e-14** |
| outcome of that row | returns | **REFUSED** | **REFUSED** | **REFUSED** |

`_ConditioningError: pmm2d staggered GENERALIZED mortar interface: the 576x576
mortar operator is numerically singular -- LAPACK reciprocal 1-condition
8.396e-13 against a 1e-12 bar ...`

**The refusal is a FALSE POSITIVE, and this is the proof.**  `v8` builds ONE
device two ways: (a) per-layer NON-CONFORMING (walls 0.2371/0.6183 and
0.3117/0.7402 -- takes the generalized mortar), and (b) both layers on the
COMMON REFINEMENT of those walls (takes no mortar), which is therefore an
oracle for (a).  Zeroth-order `R` for incident `E_y`:

| `M` | per-layer, **`24651c8`** | per-layer, **round 2** | union (oracle) | per-layer closure |
|---|---|---|---|---|
| 4 | 0.06072107 | 0.06072107 | 0.06436131 | 1.084e-03 |
| 5 | **0.06365828** | **REFUSED** | 0.06435761 | 1.139e-03 |
| 6 | **0.06396188** | **REFUSED** | 0.06460829 | 3.038e-04 |
| 7 | **0.06443418** | **REFUSED** | 0.06460797 | 2.942e-04 |

* the pre-round-2 per-layer arm **CONVERGES MONOTONICALLY** toward the union
  oracle's 0.064608 (0.06072 -> 0.06366 -> 0.06396 -> 0.06443), and its own
  lossless closure IMPROVES with `M` (1.08e-03 -> 2.94e-04).  These are ordinary
  converging answers;
* they are **BUILD-STABLE**: WIN and WSL agree to all 8 printed digits at
  `M` = 4, 5 and 6 (0.06072107 / 0.06365828 / 0.06396188 on both).  The guard's
  stated justification -- "the answer would be a build-dependent number rather
  than a solution" -- is **measurably false at this site**;
* the geometry is completely ordinary: two layers, ordinary non-uniform walls,
  narrowest segment 237x the width contract.  The `wide` variant
  (0.21/0.55 vs 0.30/0.70) fails identically from `M`=6.

**Consequence.**  Any user of `layer_grids='per-layer'` with ONE out-of-plane
(or `slant`ed) layer next to an ordinary scalar layer loses their solve at
`n_modes >= 5`.  On `24651c8` it worked.  This is a hard regression on a
supported path.

**Why the shipped gates missed it.**  `test_the_mortar_rcond_bar_has_decades_of_gap_on_both_sides`
builds only in-plane fixtures; nothing in the 15 gates reaches
`_interface_smatrix_general_mortar_2d` with a mixed in-plane / out-of-plane
pair.  The suites that DO exercise out-of-plane per-layer work
(`test_verify_pmm2d_perlayer_slant.py`, 4 tests) use SLANTED stacks, whose
generalized population is 1.07e-04 .. 3.28e-07 -- 5 to 8 decades away from the
bar.  It is the MIXED case that is pathological, and it is untested.

**Mechanism (my reading, not measured to closure).**  When one side of a
generalized mortar is an in-plane region promoted to the 6-tuple general form,
its forward and backward mode sets are related by the `+/-q` symmetry, so the
`[[E1,E2],[H1,H2]]` block acquires a near-null space by CONSTRUCTION.  The
right-hand side lies in the range (which is why the answer is well determined
and build-stable), so `rcond` is the wrong instrument here for exactly the
reason the fix doc gives for rejecting the free lower bound: the ill-conditioning
is never excited by this `B`.

**Fixes, in order of preference.**

1. Give the generalized site its OWN bar, derived from ITS OWN healthy
   population (which must include the MIXED in-plane / out-of-plane pair).
   From the 24 stacks above, that population is 2.6e-14 .. 1.5e-04, so a bar
   there would have to sit near 1e-17 -- i.e. essentially only "exactly
   singular".
2. Leave `_interface_smatrix_general_mortar_2d` UNGUARDED, exactly as the plain
   1-D site is left unguarded and for the same measured reason (the correct
   population comes within a decade of any bar).
3. Screen on the RESIDUAL at that site instead: the operand is rank-deficient
   but the system is consistent, and a residual screen distinguishes those two.

Option 1 or 2 is a two-line change; the round-2 work is otherwise sound.
**Until it lands, 5.45.0 should not ship.**

### 6.4 The `~1850` plain 1-D decision -- CONFIRMED

`v4_d2.py plain1d`, a two-layer `PMMStack` at degree 12 whose walls differ by
`delta`, with `rcond(Wb)` / `rcond(Vb)` instrumented at every
`_interface_smatrix` call (patched in BOTH `_core` and `stack`, because
`stack.py` imports the symbol by name -- patching `_core` alone intercepts
nothing, which is how my first pass measured "0 interface calls"):

| `delta` | min `rcond(Wb, Vb)` WIN / WSL | `R+T` | warnings | outcome |
|---|---|---|---|---|
| 1e-02 | 2.9538e-06 / 2.9538e-06 | 1.000000 | 0 | correct |
| 1e-03 | **2.0946e-08 / 2.2594e-08** | 1.000000 | 0 | correct |
| **1e-04** | **1.9482e-10 / 1.9482e-10** | **1.000000** | **0** | **correct** |
| 1e-05 | 1.9512e-12 | -- | -- | REFUSED by the shipped 1-D sliver guard |
| 1e-06 | 1.9543e-14 | -- | -- | REFUSED |
| 1e-07 | 1.9554e-16 | -- | -- | REFUSED |

* the CORRECT population's closest approach to a 1e-12 bar is **1.948e-10 =
  2.29 decades** (the fix doc measures 1.0 on its fixture).  Either way it is
  1-2 decades against the mortar path's 5+, so **leaving it unguarded is the
  right call and I agree with it**;
* better: at `delta` = 1e-5 the reading is **1.9512e-12, ABOVE a 1e-12 bar** --
  a 1e-12 screen would NOT have refused it, while the shipped 1-D sliver guard
  does.  The 1-D guard is strictly stronger there, which is the fix doc's
  argument, now measured from a second fixture;
* and the **cross-build spread on those readings is 8 %** (2.0946e-08 vs
  2.2594e-08 at `delta` = 1e-3).  A bar with one decade of margin sitting on a
  quantity with an 8 % two-build spread and only a same-family BLAS pair
  sampled is precisely the S4 shape `TESTING_STANDARDS.md` forbids.

### 6.5 `LinAlgWarning` under `-W error` -- CONFIRMED

An exactly-singular 6x6 operand (zero pivot):

| filter | outcome |
|---|---|
| `simplefilter("always")` | `_ConditioningError: probe site: the 6x6 mortar operator is numerically singular ...` |
| `simplefilter("error")` | `_ConditioningError: ...` (identical) |

Identical on both builds.  The `except` tuple does what the doc says.

### 6.6 The refuted free lower bound

Not independently re-derived; the shipped gate
`test_the_free_lower_bound_on_the_condition_number_is_refuted_here` re-measures
it on the running build and passed on both builds in S2.  Accepted on the
gate's evidence, flagged in S10 as not independently re-measured.

---

## S7. D3 RE-MEASURED (task 5)

`v5_d3.py`.  All readings below are IDENTICAL on WIN and WSL to every digit
printed.

### 7.1 The requirement

Smallest `nq` reaching the reference rule's own floor, the bar derived at each
point from that rule's 37-node self-drift (my bar is 3x the drift, normalized
per `omega`; the shipped gate uses 20x normalized at `omega`=0):

| `M` \ `omega` | 0 | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 2 | 9 | 18 | 12 | 16 | 23 | 34 | 53 | 90 |
| 4 | 2 | 18 | 10 | 13 | 17 | 23 | 34 | 53 | 90 |
| 6 | 3 | 18 | 11 | 14 | 18 | 24 | 35 | 54 | 91 |
| 8 | 4 | 18 | 12 | 15 | 19 | 25 | 36 | 55 | 92 |
| 12 | 6 | 18 | 14 | 17 | 21 | 27 | 38 | 57 | 93 |

Fitted `need = a . omega + b(M)`: **slope 0.6293 .. 0.6366 across `M` = 3..12,
a spread of 1.2 %**; intercept `9.30 + 0.431 M`.  (The fix doc: 0.6341..0.6455,
1.8 %, `9.38 + 0.47 (M-3)` = `7.97 + 0.47 M`.)  The shipped constants
`(0.72, 0.5, 10.0)` are an UPPER envelope of my measurement on all three terms.
**CONFIRMED.**

**One caveat, stated because the fix doc's "worst margin 2 nodes" is
bar-dependent.**  Against MY tighter bar the rule falls **4 nodes short in 2 of
72 cells** -- (`M`=4, `omega`=1) and (`M`=3, `omega`=2).  Both are the
small-`omega` corner where the bar is ~5e-15 RELATIVE, i.e. at the double
precision floor: at the `nq` the rule hands out, the actual error is
**5.53e-15 and 5.44e-15 relative (9.99e-15 / 7.11e-15 absolute)**.  The
shortfall is round-off, not under-resolution.  **CONFIRMED in substance;** the
"worst margin 2 nodes" should be stated with its bar.

### 7.2 The kernel ladder

Relative kernel error against an 8x-refined rule, `alpha0` = 0.37 G, on MY grid
(a 3-segment grid with one long central segment):

| longest segment | BEFORE (`\|m\|<=7`) | AFTER | BEFORE at the ENFORCED cap | AFTER at the cap |
|---|---|---|---|---|
| 0.33 d, `M`=4 | 7.00e-15 | 8.37e-15 | 9.09e-15 (`M`=4 cap 4) | 6.19e-15 |
| 0.62 d, `M`=4 | 2.36e-08 | 6.04e-15 | 9.09e-15 | 6.19e-15 |
| 0.91 d, `M`=4 | 3.85e-04 | 5.35e-15 | 5.69e-10 | 7.80e-15 |
| **0.96 d, `M`=4** | **1.29e-03** | **4.68e-15** | 2.51e-09 | 5.51e-15 |
| 0.96 d, `M`=6 | 2.09e-06 | 2.26e-15 | 2.09e-06 (cap 7) | 2.26e-15 |
| **0.96 d, `M`=8** | 1.38e-09 | 8.61e-15 | **1.15e-04** (cap 10) | **1.01e-14** |

* the AFTER reading at the fix doc's own cell, **4.68e-15**, matches its
  4.7e-15 to two digits;
* the BEFORE reading is fixture-dependent (mine 1.29e-03 vs its 7.5e-04) --
  expected, different grid;
* **the defect IS reachable through the shipped API.**  The order cap the
  library enforces (`n_orders <= (q-1)//2`) admits `\|m\| <= 10` on a 3-segment
  `M`=8 grid, and there the pre-round-2 kernel error on a 0.96 d segment is
  **1.15e-04**, now **1.01e-14** -- a **10-decade** improvement on a path a user
  can actually take.  This is worth stating, because D3 is graded P3 and reads
  as defensive.

### 7.3 What D3 moves

`nq` changes on **21 of 30** (geometry, `M`) cells at the enforced cap,
including the shipped taper slice, the nested refinement and the non-conforming
pair.  But the FAR FIELD barely moves: with vs `24651c8`, on ordinary per-layer
stacks at the cap,

| fixture | max `\|with - pre\|` |
|---|---|
| non-conforming `M`=4, `n_orders`=4 | 3.331e-16 |
| non-conforming `M`=6, `n_orders`=7 | 5.551e-16 |
| taper slice `M`=6, `n_orders`=7 | 1.665e-15 |
| 0.96 d long segment `M`=6, `n_orders`=3 | 5.551e-15 |
| uniform ARRAY / INTEGER spellings | 0.000e+00 |

**D3 is behaviour-preserving to round-off on every ordinary geometry, and the
uniform-array and integer spellings do not move at all.**  The claim's SCOPE
(integer lattices bit-identical) is correct and conservative.

---

## S8. DURABILITY AUDIT (task 6)

`tests/unit/test_fix_pmm2d_mortar_round2.py` (15 gates) and the restated
`tests/unit/test_verify_pmmstack_sliver_walls.py` (15 gates).  Re-timed on WIN,
one thread: **30 passed in 113.74 s** (slowest:
`fail_before_the_sliver...` 49.92 s, `the_spurious_spectrum...` 15.15 s,
`the_guard_has_a_measured_floor...` 13.90 s).

| bar | value | measured population | gap | verdict |
|---|---|---|---|---|
| `worst_ordinary > 100 * bar` (census) | 100x | 125.0x (nested refinement) | **1.25x** | PASS -- exact arithmetic, no build spread; brittle only to a NEW ordinary fixture narrower than 0.1 of the period |
| closing-taper rate `0.7 < got/pred < 1.6` | ratio | measured ratio | 1.4-2.3x | PASS, tracks the slicing rule |
| `n_cross > 200` | 200 | 250-256 measured | 1.25x | PASS, arithmetic |
| `min(closing) > 3 * bar` | 3x | 3.91x at 64 slices | 1.3x | PASS, arithmetic |
| `c` spread `< 1.02` | 1.02 | 1.007 measured; `c` is UNIVERSAL to 4 digits across 3 fixtures and 2 builds | ~7 decades over the build spread | PASS |
| **`uniform_c > 3 * sqrt(eps_p)`** = 9.0 | 9.0 | **9.84** on the gate's fixture | **1.09x -- the tightest bar in the file** | PASS: `\|gamma\|max` agreed to 4+ digits across my two builds, so 9 % is ~1e12 x the build spread.  It is tight to a FIXTURE change, not to a build |
| `sliver_c / uniform_c < 10` | 10 | 4.04 | 2.5x | PASS |
| `ordinary > 100 * selfgap` | 100x | measured | large | PASS |
| `sliver > 3 * ordinary` | 3x | 6.84x | 2.3x | PASS (my fixture: 5.9x) |
| closure `< 0.01 * _STAG_CLOSURE_TOL` | 5e-4 | 1.53e-05 / 7.94e-06 | 30-60x | PASS |
| all-host `e < 1e-6` | 1e-6 | 1.17e-10 / 5.53e-10 | ~2000x | PASS |
| **`worst_healthy > 1e4 * bar`** = 1e-8 | 1e-8 | 3.55e-06 on the gate's fixture | 355x | **PASS but the POPULATION IS INCOMPLETE** -- it never reaches the third guarded site, whose healthy readings are 2.6e-14 (DEFECT V1) |
| `rcs[1e-7] < 0.01*bar`, `rcs[1e-5] < bar` | -- | measured | -- | PASS |
| `c_bad/c_ok > 1e8`, `g_bad/g_ok < 10` | ratios | measured | -- | PASS |
| `rc_ok < 1e-9` and `rc_ok/bar < 1e3` | 1e-9 / 1e3 | 9.697e-11 (doc) / 1.948e-10 (mine) | 5-10x / 5-10x | PASS, but see S6.4: the two-build spread on this quantity is **8 %**, so the 1e-9 side has ~10x of margin against an 8 % spread on a SAME-FAMILY BLAS pair.  Acceptable; a third BLAS family would settle it |
| `n == 12` (projector hash count) | exact | deterministic | -- | PASS |
| 720-cell `assert not bad` | exact | 720/720 reproduced | -- | PASS |
| `rule >= need` | -- | reproduced with the gate's bar | -- | PASS (see S7.1 for the bar dependence) |
| `e_new < 1e-13`, `e_old > 100 * e_new` | 1e-13 | 4.68e-15 | 21x | PASS |

**Every numeric constant in the file has a stated origin, and every one I
re-measured came back within its stated gap.**  No fabricated-adjacent constant
found.  The two gates I would restate:

* `test_the_mortar_rcond_bar_has_decades_of_gap_on_both_sides` -- add a MIXED
  in-plane / out-of-plane per-layer fixture to the healthy population.  That is
  the gate that should have caught DEFECT V1 and it is a six-line addition;
* `test_the_quadrature_rule_clears_the_measured_requirement` -- state the bar
  (20x the reference's self-drift, normalized at `omega`=0) in the assertion's
  comment, since the "worst margin" is a function of it.

Sub-decade bars: none of the 15 gates asserts a bar with less than a decade of
gap on a build-movable quantity.  The three arithmetic bars with ~1.25x margin
(`100 * bar`, `n_cross > 200`, `3 * bar`) sit on exact geometry, which has no
build spread; they are fixture-brittle rather than build-brittle, and that is
the intended behaviour of a census gate.

---

## S9. `ruff` and durations

```
ruff check lumenairy/ tests/        ->  All checks passed!
ruff check validation/probe_verify_mortar_round2/  ->  All checks passed!
```

No test was added, so `.test_durations` needs no splice.  (Had DEFECT V1's
reproducer been added as a gate it would need one; it is shipped as a probe
instead, since a failing gate would break the suite.)

---

## S10. DEFECTS

| # | severity | what | reproducer |
|---|---|---|---|
| **V1** | **P1 -- ship-blocking** | `_MORTAR_RCOND_REFUSE` = 1e-12, calibrated on the two IN-PLANE mortar sites, is applied unchanged to `_interface_smatrix_general_mortar_2d`, whose healthy population on a MIXED in-plane / out-of-plane per-layer stack is 1.4e-11 .. 2.6e-14.  Ordinary two-layer stacks (narrowest segment 237x the width bar) are REFUSED from `n_modes` = 5 up.  The refused solves converge to the union-grid oracle and are build-stable to 8 digits, so the refusal is a false positive.  Hard regression vs `24651c8`. | `validation/probe_verify_mortar_round2/v7_generalized.py`, `v8_oop_regression.py` |
| **V2** | P3 | The width contract refuses a CONFORMING per-layer stack (all layers on one wall array), which has no cross-grid projection and is measurably insensitive to the sliver (`delta`-independent to ~1e-4 over three decades). `Basis1D` cannot know its neighbours, so the refusal is structural. | `v3_guard.py nomortar_fp` |
| **V3** | P3 -- documentation | The conditioning backstop is claimed as an independent second line; on a sliver that occupies ONE axis in the LAST layer it does not fire down to `delta` = 1e-7 (`rcond` ~1.1e-10 at 1e-6, 2 decades above the bar), because the `MassH_A V_A` operator is built from the FIRST grid and is `delta`-independent there (measured 1.384e+05 at every `delta`). | `v4_d2.py refuse`, `v2_d1.py exponents` |
| **V4** | P3 -- documentation | The E-row exponent "~2.0" is the TWO-AXIS value; it is **1.0 per axis carrying the sliver** (measured 1.01 on a one-axis fixture).  The cross-mass exponents `C1_x`/`C2_x` are saturated at the float64 ceiling on both fixtures and disagree between `M` = 4 and 6 by 2.4x; they should be marked as such. | `v2_d1.py exponents` |
| **V5** | P4 -- documentation | The 1e-3 bar is a pure PERIOD FRACTION with no wavelength in it, so on a large-period cell it refuses features that are physically ordinary (40 nm on a 40 um period).  Defensible (the mortar conditioning really is governed by `w/d`) but unstated.  Related: the message's remedy (2) points at a shared lattice whose own segments are the same width the per-layer path was refused for. | `v3_guard.py falsepos` |
| **V6** | P4 | The contract is reached at `solve()` time, not at `add_layer` -- a 400-slice taper is built in full before it is refused.  The fix doc says "at the grid's entry point", which is true of `Basis1D` but not of the user's call site. | `v3_guard.py boundary` (`public_surface`) |

Nothing found that makes the shipped ANSWERS wrong: bit-identity holds on 165
hashes and 60 solves, D1's floor and its energy invisibility reproduce, the
mortar's algebra reproduces against an analytic oracle, and D3 moves ordinary
per-layer answers only in the last bits.

---

## S11. SHIP RECOMMENDATION for 5.45.0

**DO NOT SHIP as it stands.  Fix DEFECT V1 first.**

V1 is a two-line change (either give the generalized site its own bar or leave
it unguarded, as the plain 1-D site already is) plus one fixture added to
`test_the_mortar_rcond_bar_has_decades_of_gap_on_both_sides`.  It is a hard
regression on a supported path -- an out-of-plane or slanted per-layer stack
next to an ordinary scalar layer -- that `24651c8` solved correctly and
reproducibly, and no shipped gate covers it.

Everything else in round 2 verifies.  D1's contract, its derivation and both
sides of its bar reproduce on an independent fixture; D3 is a real, API-reachable
10-decade improvement that moves no ordinary answer; D2's `lu_factor` path is
bit-identical over 60 solves at 1.00x cost and its named refusal behaves under
`-W error`.  The width guard's documented cost above the bar is real and I
measure it slightly worse than stated (1.0..5.9x rather than 1.0..5.0x), silent,
and not removable by `n_modes`; a `UserWarning` in the 1e-3..3e-2 band would
close the last silent-wrong-answer surface, and I would ask for it before 5.45.0
as well, though it is not blocking.

With V1 fixed and re-run, **ship**.

---

## S12. WHAT I COULD NOT VERIFY

1. **A THIRD BLAS FAMILY.**  Both builds link scipy-openblas, as in the build,
   the fix and the first verification.  Every cross-build spread here is a lower
   bound.  This matters most for S6.4 (`rc_ok < 1e-9`, 8 % spread on a
   same-family pair) and for DEFECT V1's build-stability evidence (WIN/WSL agree
   to 8 digits; a different LAPACK might not).
2. **The free-lower-bound refutation (S6.6)** was not independently
   re-derived; it is accepted on its shipped gate, which passed on both builds.
3. **The 106-solve population and the 87-hash harness of the fix doc** were not
   re-run; my own 35-solve population and 165-hash harness were built instead.
   Where the two overlap (the in-plane pair, the integer projector, the taper
   ladder) they agree.
4. **DEFECT V1's mechanism** (a structural near-null space from promoting an
   in-plane region to the 6-tuple general form) is my reading of the operator's
   construction; I measured the SYMPTOM and its build-stability, not the null
   space itself.
5. **`M` > 8** on the D1 ladder and `M` > 7 on the generalized-site population.
   The 3-segment `M`=8 fixture is already 175 s per `delta` rung on this box.
6. **The two pre-existing noises** the fix doc logs as open item B (the
   `RuntimeWarning: invalid value encountered in multiply` and the LAPACK
   `DLASCL` stderr) were observed on WIN in the gate run and NOT checked against
   the pristine tree by me; the fix doc states it did check them.

---

## S13. Probes

`validation/probe_verify_mortar_round2/` -- `README.md` and eight scripts, each
asserting which `lumenairy` it imported, each writing its JSON beside itself:

| script | what it measures |
|---|---|
| `v1_bitid.py` + `v1_compare.py` | 30 fixtures / 165 hashes + warning sets, `with` vs `24651c8` |
| `v2_d1.py` | `floor` / `allhost` / `spectrum` / `exponents` -- D1 re-derived on my own fixture |
| `v3_guard.py` | `census` / `boundary` / `exempt` / `shared` / `falsepos` / `degraded` / `nomortar_fp` -- both break attempts |
| `v4_d2.py` | `bitid` / `pop` / `refuse` / `warnerr` / `plain1d` -- D2 re-measured |
| `v5_d3.py` | `need` / `rule` / `kernel` / `integer` / `candidate` / `nqmap` -- D3 re-measured |
| `v6_d3_impact.py` | how much D3 moves an ordinary per-layer answer, both arms |
| `v7_generalized.py` | the THIRD guarded site's healthy population -- DEFECT V1 |
| `v8_oop_regression.py` | DEFECT V1's two-way reproducer against a union-grid oracle |
