# VERIFY -- adversarial verification of Stage A (in-plane anisotropic pure staggered 2-D PMM)

Date: 2026-09-09.  Branch `feat/pmm2d-staggered-anisotropic`, worktree
`C:/tmp/lum_aniso`, build HEAD at the start of this verification `78dca9a`
(build commits `7f75ae7`, `eeb1c6f`, `5f4e831`, `228103b`, `0dc027b`; the
merge `78dca9a` adds only Stage-B probe documents -- confirmed by
`git diff 0dc027b..78dca9a -- lumenairy/ tests/`, which is EMPTY).

Verifier: an agent that did not build this.  Every number below was
re-MEASURED with scripts under `validation/probe_verify_staggered_aniso/`
(see its `README.md`); nothing was accepted from a comment or a table.

Machine for every measurement: tesla-ryzen, Windows 11, CPython 3.14.6,
NumPy 2.4.4 on scipy-openblas 0.3.31.188.0, `OMP_NUM_THREADS =
OPENBLAS_NUM_THREADS = MKL_NUM_THREADS = 1` unless a row says otherwise.
Every probe asserts `lumenairy.__file__`; the main-clone arm asserts the
`D:/Metacept/.../Lumenairy` path.

---

## 0. Verdict summary

| gate | claim | verdict | key numbers |
|---|---|---|---|
| task 1 -- scalar path untouched | the shipped scalar answers did not move | **CONFIRMED** | 9 fixtures, 0 field mismatches, sha256 identical across the two CODE VERSIONS |
| G1 reduction | scalar cell == tensor `e*I` cell, bit for bit | **CONFIRMED** | max\|diff\| = 0.0 on `Lmat/Rmat/Stt/Schur/Et_blocks` on two FRESH cells; public entries bit-identical |
| G2 published oracle | Granet 2023 Table 2 not reproducible | **REFUTED (the negative is wrong)** | all SIX of Li 2003 Table 1's reflected orders reproduced, max dev **8.74e-05** at M=8 vs the 5e-4 bar |
| G3 Berreman | uniform tensor slab, R/T/Jones | **CONFIRMED** | worst M=7 residual **1.17e-13** on fresh fixtures (build 9.33e-14); bar 1e-11 |
| G3b fail-before | every new tensor term load-bearing | **CONFIRMED** | doc's four rows reproduced to 4 s.f. |
| G4 1-D reduction | y-uniform stripe vs the 1-D engines | **CONFIRMED** | fresh fixture ladder 1.34e-02 -> 6.96e-05 (M=5..8); doc's ladder reproduced exactly |
| G5 three engines | staggered / hybrid / RCWA agree; no Fourier floor | **CONFIRMED, bar is fixture-calibrated** | fresh spread **4.82e-03** vs the test's 5e-3 bar (1.04x); no-floor 5.00e-16 vs hybrid 1.03e-02 |
| G6 closure | Hermitian closes, non-Hermitian absorbs | **CONFIRMED** | fresh LC 3.90e-08 / gyro 3.44e-08 at M=8; lossy deficit 0.332 |
| G7 symmetry | transpose / mirror exact; swapped `e12`/`e21` breaks | **CONFIRMED** | fresh transpose 1.10e-14 (LC) / 1.74e-14 (gyro); swapped gyro **1.19e-02** |
| G8 stack | split == single; uniform multilayer vs Berreman | **CONFIRMED** | fresh split 1.12e-15; multilayer 5.87e-14 at M=7 |
| G9 absorption | `sum A == 1 - R - T` for a lossy tensor stack | **CONFIRMED** | fresh M=8 dev 1.48e-06, lossless layers 1.45e-14 |
| task 3 -- Jones gauge | PUBLIC, no conjugation; columns = incident | **CONFIRMED with fail-befores** | matched 1.4e-14..4.9e-14 vs Berreman; CONJUGATED arm 2.1e-02..5.8e-02; TRANSPOSED arm (gyro) 6.1e-02..6.9e-02 |
| task 5 -- tripwire | never fires on a shipped scalar fixture; fail-before real | **CONFIRMED, and restated** | 0 firings in 140 shipped tests; 1.82x margin reproduced exactly, replaced by a 13.1x engineered arm |
| task 6 -- pre-existing red | `test_fff_nv_stripe_reduces_to_rigorous_1d` fails pre-build | **CONFIRMED** | main clone f70628d: `1 failed, 9 passed` in 106.9 s |

Library defects found: **none.**  Two documentation defects in the build doc
(G2's negative verdict and the conjugation-bridge conclusion drawn from it),
both corrected here and in the test file.  Six durability observations, three
of which were fixed as test-only changes (section 7).

---

## 1. Task 1 -- the SCALAR path is untouched, proven across CODE VERSIONS

The build's G1 (table T1) compares the scalar arm against the tensor `e*I` arm
*inside the new build*.  That shows the dispatch is internally consistent; it
cannot show the shipped scalar answer did not move.  Probe
`v1_scalar_identity.py` runs the same fixtures under two `PYTHONPATH`s on one
machine, one process each:

* worktree `C:\tmp\lum_aniso\lumenairy\__init__.py` (5.42.1)
* read-only main clone `D:\Metacept\...\Lumenairy\lumenairy\__init__.py`
  (5.42.1, HEAD `f70628d`, which is the parent of the plan commit `9f212dd`
  and therefore library-identical to the pre-build tree)

**Result: 0 field mismatches out of 9 fixtures x 3-7 hashed fields.**

| fixture | R sha256[:16] | T sha256[:16] | sum R | sum T |
|---|---|---|---|---|
| `(2,2)` pillar, TE, normal | `eb2f34c10c2f1064` | `2bcee02d409b77bb` | 0.11460120879702795 | 0.885400466896066 |
| `(2,2)` pillar, TM, normal | `a243ef290a2565d6` | `847905e78965cde3` | 0.11460120879703058 | 0.8854004668960492 |
| `(3,3)` two-pillar, TE | `36f446df320afa1a` | `3cebe93fe2cabc0b` | 0.07234406819552247 | 0.9276392464921797 |
| `(2,2)` pillar, CONICAL `theta 0.31 / phi 0.62` | `79a6a31574c160f1` | `1bd5697fa67773a3` | 0.11594586454926592 | 0.8826278112024988 |
| `(2,2)` LOSSY, TM, `theta 0.22` | `ed5fdd1153b60e05` | `64723e85c4b9003e` | 0.10638759286375293 | 0.5439052539475118 |

| assembled operators | `Lmat` | `Rmat` | `Stt` | `Schur` | `Et_blocks[0]` | `Et_blocks[1]` | sorted `lam` |
|---|---|---|---|---|---|---|---|
| pillar `(2,2)` M=6, Bloch 0.31 / -0.17 | `674b3ce872ca3e1f` | `95330a5328a4f2aa` | `8c7fafec3a854d69` | `b2fb481d61447040` | `17f0c6e0018e82a6` | `1f247e6f94a68fa2` | `7d3ee1055d5b12b8` |
| two-pillar `(3,3)` M=5, normal | `1be7da942b70bb3c` | `e601dcab557497e0` | `4a528b6c01360917` | `d40d1b13bd9cc05b` | `665c4da06b7024a0` | `0a04dda580850eba` | `4796fd1c3cdf60e9` |
| lossy `(2,2)` M=6, Bloch 0.11 / 0.29 | `5391cc1253438c0b` | `4b629bbcdb48e3db` | `af9a05bf9a624ebe` | `303310d7e8255226` | `5d4d65842bddbc12` | `b2d7fbab1f9bfe9d` | `02a934e55eb8afff` |

| pure stack (scalar patterned + uniform, conical) | R | T | jones | sum R+T |
|---|---|---|---|---|
| | `867ec3a3284040a9` | `9add0848cedb2c2b` | `7908f3c3c117df8e` | 0.9993316491245817 / 1.0006965940969919 |

The eigen-SPECTRUM hashes are identical too, which is the strongest of these:
a dense generalized eig is where a code-layout change would first show.

Cross-check with the build doc: this probe's `Rmat` and `Stt` hashes for the
`(2,2)` M=6 assembly (`95330a5328a4f2aa`, `8c7fafec3a854d69`) are the SAME
strings the build doc's T1 reports for its own `(2,2)` M=6 assembly -- those
two operators are eps-free, so they must match across different cells, and
they do.  That independently corroborates T1's hashing method.

**Verdict: CONFIRMED.**  The scalar dispatch is not merely internally
consistent, it is byte-identical to the pre-build library.

---

## 2. Gates re-measured on FRESH fixtures (task 2)

`v2_gates.py`.  New tensors (`uniaxial_tensor(1.45, 1.75, pi/2, 0.90)`; a
gyrotropic `e12 = -e21 = 0.35i` with `e11 = e22 = 2.60`, `e33 = 2.40`; an
isotropic `5.0 I`), new periods / wavelengths / depths / substrate indices,
new angles.  Nothing from the build's parameter set.

### G1 -- reduction (CONFIRMED)

| fresh fixture | `Lmat` | `Rmat` | `Stt` | `Schur` | `Et_blocks` | `Et_offdiag` |
|---|---|---|---|---|---|---|
| `[[3.24, 1.0], [2.10, 7.29]]`, `(2,2)`, M=7, Bloch -0.44 / 0.23 | 0.0 | 0.0 | 0.0 | 0.0 | bit-equal | scalar `None`, tensor exactly 0 |
| a fully asymmetric `(3,3)` cell (2..9), M=5, Bloch 0.05 / -0.61 | 0.0 | 0.0 | 0.0 | 0.0 | bit-equal | as above |

Public entries on a fresh geometry (period 0.95 um, depth 0.37 um, wl 0.71 um,
`n_sub` 1.45, M=7, `n_orders` 4): `max|dR| = max|dT| = 0.0`, and a SCALAR
`(Nx, Ny)` map handed to the tensor entry is bit-identical to the promoted
`e*I` cell (R and Jones both).  `Curl / Kzt / Ktz / G3 / Meps33` absent on
both arms.

### G3 -- Berreman (CONFIRMED)

24 fresh combinations (2 tensors x 2 grids x 3 incidences x 2 modal counts),
period 0.33 um, wl 0.90 um, depth 0.62 um, `n_sub` 1.6.

| | worst M=5 | worst M=7 |
|---|---|---|
| residual over R, T and the complex Jones | **3.95e-12** | **1.17e-13** |

The test's bar is `1e-11`.  On the build's own fixtures the worst M=7 value is
9.33e-14 (107x of margin); on these it is 1.17e-13 (**85x**, 1.93 decades).
Still a gap, but the number to know is 85x, not 107x.

Two-sided ladder on the fresh gyrotropic `(2,2)` slab at 18 deg: 2.58e-12 ->
3.74e-14, a factor **69** against the test's 10x bar (the build's fixture gave
5.8e3).  At NORMAL incidence the M=5 -> M=7 residual RISES (6.17e-15 ->
7.25e-14): the roundoff plateau, exactly as the build doc says, and the reason
the ladder claim is made at oblique incidence only.

### G4 -- 1-D reduction (CONFIRMED)

Fresh stripe: period 1.10 um, wl 0.63 um, depth 0.42 um, `theta` 0.35, ridge =
the fresh LC tensor, groove `2.10 I`, `n_sub` 1.45.

| M | per-order max\|dR\|,\|dT\| | max\|dJones\| | y-forbidden | closure |
|---|---|---|---|---|
| 5 | 1.340e-02 | 1.713e-03 | 8.72e-29 | 6.29e-03 |
| 6 | 3.271e-03 | 1.112e-03 | 6.09e-28 | 1.73e-04 |
| 7 | 4.993e-04 | 2.199e-04 | 1.58e-26 | 2.46e-05 |
| 8 | **6.960e-05** | **2.642e-05** | 1.47e-26 | 2.37e-06 |

The two 1-D oracles (`pmm_jones_1d` degree 16 vs `rcwa_jones_1d` 81 orders)
agree with each other to 9.77e-08 (R/T) and 1.83e-07 (Jones) on this fixture,
so the residual is the staggered arm's discretization error, not the oracle's.
The M=8 residual is **16x larger than on the build's fixture** -- see the
durability note in section 7.

### G5 -- three engines + no floor (CONFIRMED; the bar is fixture-calibrated)

Fresh cell: period 0.62 um, wl 0.50 um, depth 0.31 um, `n_sub` 1.45.

| cell | worst \|dR\| | worst \|dT\| | worst \|dJones\| | staggered closure |
|---|---|---|---|---|
| iso pillar in LC host | 8.29e-04 | 2.80e-03 | 8.23e-03 | 5.68e-07 |
| LC pillar in iso host | 8.44e-04 | **4.82e-03** | 4.85e-03 | 7.94e-08 |

against hybrid (degree 9/11) and `rcwa_jones_2d` (`n_orders` 9/13).  The
staggered arm closes energy 3-4 decades better than the hybrid on both cells
(5.7e-07 / 7.9e-08 vs 1.0e-03 / 3.9e-04), which is the evidence that the
spread is the Fourier engines' floor.

No-floor, `n_orders` 4 -> 8: staggered **5.00e-16**, hybrid **1.03e-02** --
a ratio 2.1e13, reproducing the build's 2.5e12 in order of magnitude.

### G6 -- closure (CONFIRMED)

| fresh cell | M=6 | M=8 | drop |
|---|---|---|---|
| LC host + iso pillar | 1.519e-05 | **3.897e-08** | 390x |
| gyrotropic host + iso pillar | 1.451e-05 | **3.443e-08** | 421x |
| LC host + `eps = 5 + 0.9i` pillar | deficit 0.3318 | deficit **0.3324** | -- |

Bar 1e-5 at M=8: 257x over the worse fresh measurement.  The gyrotropic cell
closes as well as the LC one, confirming the Hermitian-lossless classification
on a fresh tensor.

### G7 -- symmetries (CONFIRMED)

Fresh cells, periods `(0.62, 0.899) um`, M=6.

| claim | per-order | Jones |
|---|---|---|
| x<->y transpose, LC | 1.096e-14 | 1.496e-14 |
| x<->y transpose, GYROTROPIC | 1.737e-14 | 1.107e-14 |
| y-mirror, LC | 6.314e-15 | 1.031e-14 |
| CONTROL: `e12`/`e21` swapped, LC | 1.096e-14 (**bit-identical to the correct arm**) | 1.496e-14 |
| CONTROL: `e12`/`e21` swapped, GYROTROPIC | **1.187e-02** | 6.686e-02 |

The LC swap is not merely small, it produces the identical float -- the swap
is a literal no-op there.  That is the sharpest possible statement of why the
gyrotropic cell is the discriminator, and it reproduces the build's finding.

### G8 -- stack (CONFIRMED)

Fresh: split vs single at `theta 0.27 / phi 0.83`, M=6 -> `dR` 3.19e-16,
`dT` 8.33e-16, `dJ` 1.12e-15 (bar 1e-12).
Fresh uniform tensor multilayer (LC 0.19 / gyro 0.11 / mirrored-LC 0.23 um,
period 0.31 um, wl 0.90 um, 18 deg / 65 deg, `n_sub` 1.6) vs Berreman:
M=5 **1.238e-13**, M=7 **5.873e-14**.  Both far inside the 1e-11 bar -- but
the M=5 -> M=7 drop is only **2.1x** on this fixture, because M=5 is already
at the plateau here.  See section 7 item 2: this is why the test's
`r7 < r5/10` was restated.

### G9 -- absorption budget (CONFIRMED)

Fresh three-layer stack (LC+iso 0.10 um, LC + `5 + 0.9i` pillar 0.31 um,
uniform gyrotropic 0.08 um), `theta 0.19 / phi 0.44`:

| M | \|sum A - (1-R-T)\| (Ex / Ey) | A(lossy) | max \|A(lossless)\| | wall |
|---|---|---|---|---|
| 6 | 5.297e-04 / 5.948e-05 | 0.4123 / 0.3897 | 8.10e-15 | 0.62 s |
| 7 | 2.779e-05 / 4.579e-06 | 0.4115 / 0.3898 | 1.83e-14 | 1.74 s |
| 8 | **1.482e-06 / 1.359e-07** | 0.4115 / 0.3898 | 1.45e-14 | 4.05 s |

Bar 1e-5: 6.7x on this fresh fixture (122x on the build's).  The eps-free
block-Gram flux does work for tensor modes, and the lossless layers really do
carry zero.

### Two extras the build claimed but did not measure

* **The no-floor property holds for the JONES, not just R/T.**  T5 measures
  `n_orders` 4 -> 8 on order-0 R and T only.  Measured here on the complex
  order-0 reflection Jones of the same cell: **5.20e-16**.
* **T12's retained-operator inventory reproduces.**  Walking every ndarray
  attribute of a `(2,2)` M=6 assembly: scalar 2 896 064 bytes, tensor
  3 216 576 bytes, ratio **1.1107** (the doc claims 1.111).  The tensor arm's
  attribute set differs from the scalar one by exactly `Et_offdiag` -- no
  `Curl`, `Kzt`, `Ktz`, `G3` or `Meps33` on either.

### The build doc's own tables re-run (`v6_doc_numbers.py`)

Every entry of T3 (13 values), T4 (12), T5-no-floor (2), T6 (4), T7 (8),
T8b (2) and T9 (6) reproduces to **four significant figures** on this machine.
No "right conclusion, wrong numbers" defect in those tables.

---

## 3. Task 3 -- cross-engine JONES (magnitudes AND phases)

`v3_jones.py`, `v3b_jones_power.py`.  This is the gap no efficiency gate can
cover: `|J|^2` is invariant under a global CONJUGATION and under a TRANSPOSE.

### (a) vs `berreman_jones_1d` (exact oracle, contract "columns = incident lab [Ex; Ey]")

Uniform slab, period 0.33 um, wl 0.90 um, depth 0.62 um, `n_sub` 1.6, M=7.

| tensor | incidence | max\|J - jr\| | max\|\|J\|-\|jr\|\| | CONJUGATED arm | TRANSPOSED arm | `J01/J10` |
|---|---|---|---|---|---|---|
| LC | normal | 4.890e-14 | 4.252e-14 | 2.691e-02 | 4.890e-14 (no-op) | +1.0000000000002 |
| LC | 0.30 rad | 1.354e-14 | 1.338e-14 | 4.459e-02 | 3.425e-03 | +0.9127 |
| LC | 0.30 / 0.75 conical | 2.664e-14 | 2.442e-14 | 4.964e-02 | 1.015e-03 | 0.9601 - 0.0071i |
| gyro | normal | 3.566e-14 | 3.014e-14 | 5.794e-02 | **6.857e-02** | **-1.0000000000003** |
| gyro | 0.30 rad | 1.353e-14 | 1.277e-14 | 5.277e-02 | **6.139e-02** | -0.9127 |
| gyro | 0.30 / 0.75 conical | 1.357e-14 | 1.055e-14 | 5.140e-02 | **6.139e-02** | -0.4784 - 0.4064i |
| diagonal (`e12 = 0`) | normal | 4.203e-14 | 4.174e-14 | 3.123e-02 | 4.203e-14 (no-op) | -- |

* the PUBLIC-gauge / no-conjugation claim is confirmed with a **12-decade**
  fail-before on every fixture;
* the row/column contract is confirmed against Berreman's documented one,
  with a **12-decade** fail-before -- but ONLY on the gyrotropic slab, where
  reciprocity does not force `J01 = J10`.  On the LC and diagonal slabs at
  normal incidence the transposed arm is bit-identical to the matched one.
  That blind spot is now itself asserted in the test file.

### (b) vs `pmm_jones_2d` (hybrid; conjugates back) and `rcwa_jones_2d`

Patterned LC-host + iso-pillar cell, period 0.62 um, wl 0.50 um, depth 0.31 um.

| incidence | stag vs hybrid | stag vs rcwa | hybrid vs rcwa | max phase diff | CONJ arm vs hybrid | CONJ arm vs rcwa |
|---|---|---|---|---|---|---|
| normal | 4.882e-03 | 4.713e-03 | 6.449e-04 | 1.22 deg | **1.646e-01** | **1.646e-01** |
| 0.30 / 0.70 | 3.119e-03 | 4.306e-03 | 1.194e-03 | 3.25 deg | **6.961e-02** | **6.904e-02** |

The three engines' Jones agree in magnitude AND phase to the Fourier engines'
own floor, and the conjugated arm is 1.5 decades outside it.  (The transposed
arm is invisible here for the same reciprocity reason -- another argument for
the gyrotropic fixture.)

### (c) `|jones|^2` <-> efficiency consistency

At oblique incidence the reconstruction needs the longitudinal components the
`2x2` Jones does not carry, on BOTH sides: the reflected wave's `E_z` from
`k . E = 0`, and the INCIDENT wave's, which normalizes each column
differently (`kx` for column 0, `ky` for column 1).

| host | incidence | R00 rebuilt vs engine | same identity with `J.T` | \|J01 - J10\| |
|---|---|---|---|---|
| LC | normal | 2.776e-17 | 4.163e-17 | 4.8e-15 |
| LC | 0.30 / 0.70 | 5.551e-17 | **1.068e-05** | 9.3e-04 |
| LC | 0.55 / 1.30 | 6.939e-18 | **4.796e-03** | 2.4e-02 |
| gyro | normal | 4.163e-17 | 5.551e-16 | 4.4e-02 |
| gyro | 0.30 / 0.70 | 5.551e-17 | **5.745e-04** | 2.4e-02 |
| gyro | 0.55 / 1.30 | 0.000e+00 | **3.981e-03** | 4.0e-02 |

Machine-precision consistency at every incidence, and the identity is a
SECOND column-convention discriminator that works even on the symmetric-Jones
LC cell.  (This also settles a first-pass confusion: without the incident
normalization the identity misses by 6.8e-03 at 0.30 / 0.70 -- the missing
factor is `1 + |k_t . e / kz|^2`, not a solver defect.)

**Verdict: CONFIRMED, with fail-befores.**  Both properties are now gated in
`test_g3_jones_gauge_and_row_column_convention_are_load_bearing` and
`test_g5_order0_reflected_power_rebuilds_from_the_jones`.

---

## 4. Task 4 -- G2 reconciliation: the negative is REFUTED

`v4_g2_li2003.py`.

### What the build got wrong

Granet 2023 Sec. 4.B states that his second-example FMM values "come from an
in-house code and correspond perfectly with those reported by Li [1]", i.e.
L. Li, *J. Opt. A* **5**, 345 (2003).  Li's **Example 1** (p. 352) states the
same grating without ambiguity:

> `zeta = Theta = Phi = 0`, `d1 = 2.4 lam0`, `d2 = 1.4 lam0`, `h = lam0`,
> `w1/d1 = w2/d2 = 0.5`, `n^(+1) = 1.0`, **`n^(-1) = 1.0 + i5.0`**,
> `eps_a = 2.25(xx+yy) + i0.5(xy - yx) + 2.0 zz`,
> `eps_b = 2.25(xx+yy) - i0.5(xy - yx) + 2.0 zz`, `theta = phi = 0`, and the
> incident polarization is in the `Oxz` plane.

Granet's restatement carries two transcription errors, and the build followed
both:

1. **`n^(-1)` is a refractive INDEX, not a permittivity.**  Granet writes
   "deposited on a lossy medium with a complex relative permittivity
   `eps_r = 1 - i5`" -- Li's index, conjugated into Granet's `exp(+i w t)`
   and relabelled.  The substrate permittivity is `n^2 = -24 + 10i`.
2. **Li's Table 1 lists the REFLECTED orders**, not the transmitted ones.
   Li's Fig. 3 caption is explicit ("Convergence of the REFLECTED (0, 0) order
   efficiency for the grating in example 1"), it converges on the tabulated
   0.2980, and Fig. 7 magnifies the same quantity.  Decisively, the tabulated
   order set (columns `m = 0, +1, +2`; rows `n = -1, 0, +1`) is EXACTLY the
   propagating set of the VACUUM superstrate at `d1 = 2.4 lam`, `d2 = 1.4 lam`
   -- 11 orders, half of them plus (0,0) listed, "since this grating has space
   reversal symmetry".  Granet's running text says "transmitted".

Because the oracle is the reflected set into VACUUM, the build's entire
efficiency-DEFINITION sweep (its definitions A/B/C/D/E for a lossy substrate)
is moot: at order 0 in vacuum every definition coincides.

Li's numbers are already in the PUBLIC `exp(-i w t)` convention (his substrate
index has `Im n > 0` for loss), so **the tensors are used AS PRINTED, with no
conjugation** -- the opposite of what the build's bridge did.

### The measurement

Pillar = Li's `eps_b`, surround = `eps_a`, `n_sup = 1`, `n_sub = 1 + 5i`,
`h = lam`, normal incidence, `E` along `x`, `(2,2)` cell, `n_orders = 4`.
REFLECTED efficiencies, incident-`E_x` row:

| arm | (0,0) | (1,0) | (2,0) | (0,-1) | (1,-1) | (1,1) | max dev | sum R |
|---|---|---|---|---|---|---|---|---|
| **Li 2003 table 1** | 0.2980 | 0.1195 | 0.0222 | 0.0619 | 0.0269 | 0.0137 | -- | 0.7864 |
| staggered M=5 | 0.300126 | 0.122847 | 0.016530 | 0.059439 | 0.029141 | 0.013597 | 5.67e-03 | 0.783234 |
| staggered M=6 | 0.297975 | 0.119532 | 0.021876 | 0.062174 | 0.026937 | 0.013725 | 3.24e-04 | 0.786464 |
| staggered M=7 | 0.298164 | 0.119679 | 0.022049 | 0.061921 | 0.026807 | 0.013714 | 1.79e-04 | 0.786505 |
| **staggered M=8** | 0.297932 | 0.119539 | 0.022239 | 0.061987 | 0.026826 | 0.013709 | **8.74e-05** | 0.786533 |
| centred `(4,4)` pillar, M=6 | 0.297962 | 0.119522 | 0.022233 | 0.061972 | 0.026855 | 0.013701 | 7.15e-05 | -- |
| hybrid `pmm_jones_2d` d11/n13 | 0.297744 | 0.119327 | 0.022346 | 0.062202 | 0.026861 | 0.013742 | 3.02e-04 | -- |

Space-reversal symmetry `(m,n) -> (-m,-n)` holds to 5e-15 on every listed
order.  `sum R + sum T = 0.999884` at M=7 (the layer is Hermitian; the loss is
all in the substrate).

Li's own precision bounds the comparison: four tabulated decimals is `+/- 5e-05`
of rounding alone, at truncation order 23, and his Fig. 7 shows the (0,0) order
still spanning 0.2980..0.2988 across the three Fourier representations of
`eps` at that truncation.  **The plan's 5e-4 bar is the oracle's own floor**,
and this build sits 5.7x inside it.

### The sign discriminator, now published and two-sided

Li's Table 1 gives a SECOND row per order: "the same except the signs of the
cross terms of the permittivity tensors are reversed, i.e. `eps_a` and
`eps_b` are interchanged".  The swap moves only the `(1,-1)`/`(1,1)` pair
(0.0269 <-> 0.0137).  Measured at M=7 with the pillar/surround interchanged:

| | max dev vs Li row 1 | max dev vs Li row 2 |
|---|---|---|
| pillar = `eps_b` (as defined) | **1.79e-04** | 1.32e-02 |
| pillar = `eps_a` (interchanged) | 1.32e-02 | **1.79e-04** |

74x apart, both directions, against a published labelled pair.

### Controls (the readings the build tried)

| reading | max dev vs Li table 1 |
|---|---|
| `eps_sub = 1 + 5i` (the build's) | 2.65e-01 |
| air substrate ("suspended in air", the Table 2/3 footnote) | 2.43e-01 |
| axes swapped (`d_x = 1.4`, `d_y = 2.4`) | 1.08e-02 |

So the reading is uniquely pinned; the footnote's "suspended in air" is not
the computed configuration.

### What this means for the build's conclusions

* Build doc section 2, **T2 VERDICT ("the absolute Table-2 values are NOT
  reproduced ... by ~2 to ~230 times the 5e-4 bar")**: **REFUTED.**  Outcome
  (a) of the brief: reproduced within 5e-4, with the reading documented.  The
  G2 test is now a real published-oracle gate
  (`test_g2_li2003_table1_reflected_orders`), commit `d87fee4`.
* Build doc section 5 open item 1 ("G2 is not closed", the speculation about
  fill fractions and normalization): **CLOSED.**  Nothing was tuned; the fill
  fractions are Li's stated 0.5/0.5 and were never varied.
* Build doc T2's **gyrotropic-sign conclusion** ("this build reproduces that
  sense with the conjugation bridge and reverses it without -- so the bridge
  and the `e12`/`e21` signs are confirmed") is **not supported by the evidence
  it cites.**  Granet's Table-2 order LABELS are mirrored on one axis relative
  to Li's (his (1,1) 0.0268 / (-1,1) 0.0139 are Li's (1,-1) / (1,1)), so
  comparing *our* (1,1) vs (-1,1) against *his* labels leaves the sign
  ambiguous by exactly the flip under test; the build's conjugation and that
  label mirror cancel.  The conclusion "the `e12`/`e21` placement is right"
  is correct -- G7's swapped-block control and the Li two-row test both
  establish it -- but the stated mechanism was a coincidence of two flips.
  The correct statement is that Li is ALREADY `exp(-i w t)`, so his tensors
  are used unconjugated, and a build that conjugates them lands on his second
  row.  The test now says exactly that.

---

## 5. Task 5 -- tripwire audit

`v5_tripwire.py`, `regress_staggered.log`.

### (a) must-not-fire: CONFIRMED

The eight shipped staggered / pure suites (`test_v5_12_0_pmm2d_staggered`,
`test_v5_21_pmm2d_staggered_oblique`, `test_staggered`,
`test_audit_p1_staggered_guard`, `test_p2c_pmm2d_stack_cascade`,
`test_p2t_pmm2d_tree_cascade`, `test_pmm2d_lossless_closure_two_sided`,
`test_audit_s1_3_pmm2d_lossless_tripwire`) run with `-W always`:
**140 passed in 297.66 s, and `grep -c "closure violated"` = 0.**

### (b) the 1.82x fail-before margin: CONFIRMED, and it is the ceiling

The build's fixture (gyrotropic host, `eps = 100` pillar, period 0.70 um,
depth 0.28 um, wl 0.55 um, `n_sub` 1.5) reproduces **exactly**: 9.084329e-02
at M=3 against the 5e-02 window, ratio **1.8169**.  Extending the sweep beyond
what the build tried:

| `eps_pillar` | M=3 | M=4 | M=5 | M=8 |
|---|---|---|---|---|
| 4 | 4.775e-02 (silent) | 2.462e-04 | 1.159e-04 | 3.927e-08 |
| 16 | 8.261e-02 (fires) | 2.934e-04 | 3.540e-04 | 1.074e-07 |
| 36 | 7.682e-02 (fires) | 4.432e-03 | 3.314e-04 | 2.750e-07 |
| **100** | **9.084e-02 (fires)** | 2.532e-04 | 3.882e-04 | 1.508e-07 |
| 400 | 7.703e-02 (fires) | 1.538e-03 | 4.563e-04 | 1.671e-08 |
| 1600 | 6.729e-02 (fires) | 1.648e-03 | 4.990e-04 | 3.058e-08 |

Contrast does not help beyond `eps = 100` -- it hurts.  The build's claim that
the basis "degrades gracefully, there is no blow-up to exploit" is confirmed,
and 9.08e-02 really is the strongest physical violation available.

The M=3 value is **bit-identical** between `OPENBLAS_NUM_THREADS` 1 and 4
(9.084329e-02 both), so the 1.82x decision is not near a *build* edge.  The
risk the build doc flagged (open item 7) is different and real: an algorithm
improvement that pushes the M=3 closure under 5e-02 turns the test red.

### (c) restatement with a decade of margin: DONE

Two engineered breaks were tried.

**A rescaled modal `V` does NOT work, and that is a finding of its own.**
Monkeypatching `_region_modes` to return `V * 1.5` changes the order-0
reflectance by **0.209** and the Jones by 0.223 -- a grossly wrong answer --
while `sum R + sum T` stays at **1.3e-07**.  A uniform modal-admittance error
is energy-conserving, so no closure gate, this one included, can see it.
(The classic lossless trap: energy conservation is not per-order correctness.)

**A non-unitary layer dispersion does work.**  Monkeypatching `_region_modes`
to return `lam * (1 + i s)` -- an artificial imaginary part on every
propagation constant, i.e. the wrong-branch / mis-signed-`Im gamma` class of
defect the guard exists to catch -- leaves every permittivity exactly
Hermitian, so the guard's own losslessness predicate is untouched:

| `s` | 0 | 1e-4 | 1e-3 | 3e-3 | 1e-2 | 3e-2 | 0.1 |
|---|---|---|---|---|---|---|---|
| \|sum R+T - 1\| | 3.93e-08 | 1.87e-03 | 1.83e-02 | 5.26e-02 | 1.52e-01 | 3.40e-01 | **6.56e-01** |
| fires | no | no | no | yes | yes | yes | **yes** |

The test now asserts the `s = 0.1` arm (13.1x OUTSIDE the window) against the
SAME cell at the SAME `M` with `s = 0` (1.3e6x INSIDE it): the pair brackets
the tolerance over 7 decades without pinning its value.  The physical
under-resolution arm survives as its own test, restated as a ladder
(`d3 > 1e-2`, `d8 < 1e-5`, `d3 > 1e4 d8`; measured 9.08e-02, 1.51e-07,
6.0e5) so that no bar sits at the 1.82x crossing.  Commit `6470ad6`.

### (d) non-Hermitian silence: CONFIRMED

A lossy tensor cell with a 0.227 deficit (0.332 on the fresh fixture) emits
nothing, because `_stack_is_lossless` returns False.  The guard is not
tautological.

---

## 6. Task 6 -- the pre-existing red

`tests/unit/test_v5_20_12_rcwa_jones_2d_fff_nv.py::test_fff_nv_stripe_reduces_to_rigorous_1d`,
run **on the read-only main clone** (`D:/Metacept/.../Lumenairy`, HEAD
`f70628d`, `PYTHONPATH` pinned there):

```
1 failed, 9 passed, 18 warnings in 106.93s
AssertionError: no truncation in 11..41 gave the rigorous 1-D solver its own
exact lossless closure on this build, so there is no sound reference to
compare against; the fixture, not the formulation, is the problem
```

with `_EnergyWarning` from `rcwa_jones_1d_segments` at every truncation in the
ladder.  **CONFIRMED pre-existing and unrelated**: the failure is inside the
test's own `_sound_1d_reference` ladder in `rcwa_jones_1d_segments`, which is
not on the staggered path's call graph, and it reproduces on a library tree
that predates every build commit.

---

## 7. Task 7 -- durability audit of `tests/unit/test_pmm2d_staggered_anisotropic.py`

Against the five fragile shapes S1-S5 of `docs/TESTING_STANDARDS.md`.  "origin
measured?" means: is the constant traceable to a measurement recorded in the
build doc (or, for the rows this verification changed, in this document)?

| # | constant | test | stated origin | origin measured? | margin measured here | shape risk | action |
|---|---|---|---|---|---|---|---|
| 1 | exact equality (`array_equal`) | `g1_*` (2) | T1: dispatch takes the identical arithmetic path | yes -- and reproduced on two FRESH cells and across two CODE VERSIONS | exact | none (a decision, not a reading) | keep |
| 2 | `5e-4` | `g2_li2003_table1_reflected_orders` | the ORACLE's floor: Li's 4-digit rounding (5e-5) + his fig. 7 cross-representation spread | yes (this doc, section 4) | 5.7x over 8.74e-05; 26x under the 1.32e-02 wrong-sign arm | by design (rule 2: bounded by the oracle's own floor) | added by this verification |
| 3 | `5e-4` / `1e-3` | `g2_li2003_cross_term_sign_is_the_discriminator` | Li table 1 rows 1 and 2 | yes | 1.79e-04 vs 5e-4 (2.8x); 1.32e-02 vs 1e-3 (13x) | two-sided, published both ways | added by this verification |
| 4 | `1e-9` | `g2_control_no_gyrotropy_no_asymmetry` | isotropic control is exactly mirror-symmetric | yes (3.21e-15) | 5.5 decades | none | keep |
| 5 | `1e-11` | `g3_uniform_tensor_slab_matches_berreman` (x6), `g3_multisegment`, `g3_jones_..._load_bearing` | 1e2 x the worst measured M=7 residual | yes (T3, re-measured to 4 s.f.) | 107x on the build's fixtures, **85x** on fresh ones | S4-adjacent (a roundoff-plateau bar) -- but the plateau moved only 14% between BLAS thread paths, so 85x is ~1 decade of true margin | keep, margin recorded |
| 6 | `10.0` (ladder ratio) | `g3_convergence_is_two_sided_in_M` | T3's 5.8e3 drop | yes | 5.8e3 (build fixture), **69x** (fresh) | S1 magnitude-ratio pin; still >= 6.9x on a fresh fixture | keep |
| 7 | `1e-4` | `g3_each_new_tensor_term_is_load_bearing` (x3) | T3b: smallest break 5.70e-03, reference 5.24e-14 | yes (T3b reproduced) | 57x under the smallest break, 10 decades over the reference | none | keep |
| 8 | `1e-2` | `g3_jones_gauge_and_row_column_convention` | measured conj/transpose arms 1.47e-01 .. 1.77e-01 | yes (this doc, section 3) | 14.7x under the smallest break, 9 decades over the matched arm | none | added by this verification |
| 9 | `1.3e-5` / `2.7e-5` | `g4_stripe_grating_matches_the_1d_engines` | 3x the M=8 residual (T4) | yes (T4 reproduced exactly) | **3.0x** -- under a decade | S1/S4.  MEASURED envelope: the residual moves by a relative 2.1e-09 between `OPENBLAS_NUM_THREADS` 1 and 4, so 3x is ~9 decades above the last-bit spread.  It is FIXTURE-sensitive though: 6.96e-05 (16x larger) on a fresh stripe | kept, with the envelope measurement written into the docstring (`4ffcbaf`) |
| 10 | `10.0` (ladder) | `g4_convergence_ladder_is_monotone` | T4's 506x | yes (506.35 measured; 506.35 at 4 threads) | 50x | none | keep |
| 11 | `1e-20` | `g4_y_momentum_is_conserved` | T4's 4.86e-27 roundoff floor | yes | 6.3 decades (fresh fixture 1.47e-26, still 5.8 decades) | none | keep |
| 12 | `5e-3` | `g5_three_engines_agree` | ~8x the measured 5.82e-04 three-engine spread | yes (T5) | **8.6x** -- under a decade; a FRESH cell spreads to 4.82e-03, i.e. 1.04x inside | S4.  The quantity is the Fourier oracles' truncation floor, which is fixture-dependent; against the last-bit envelope it is safe (hybrid arm stable to 1e-12 across the thread change) | kept, with the fixture-sensitivity written into the docstring (`4ffcbaf`) |
| 13 | `1e-10` / `1e-4` | `g5_no_fourier_floor_two_sided` | T5: 3.886e-15 vs 9.687e-03 | yes (reproduced exactly) | 4.4 decades / 2.0 decades | none | keep |
| 14 | `1e-12` / `1e-5` | `g5_order0_reflected_power_rebuilds_from_the_jones` | measured 1.39e-17 / 3.77e-04 | yes (this doc) | 5 decades / 37x | none | added by this verification |
| 15 | `1e-5` | `g6_hermitian_tensor_cell_closes` (x2) | 1e2 x the worse M=8 closure (T6) | yes | 255x (build), 257x (fresh) | none | keep |
| 16 | `10.0` / `0.05` | `g6_closure_improves_..._absorbs` | T6: 298x drop; 0.227 deficit | yes | 29.8x / 4.5x | the 0.05 is a physical absorption magnitude (0.332 on a fresh cell), deterministic | keep |
| 17 | `1e-11` | `g7_xy_transpose`, `g7_y_mirror`, `g7_..._control` (correct arm) | 1e2 x the measured 2.07e-14 | yes (T7 reproduced) | 484x (build), 575x (fresh) | S4-adjacent; the transpose residual moved 60% across the thread change, still 3 decades inside | keep |
| 18 | `1e-7` | `g7_mixed_block_placement_control_gyrotropic` | T7's 3.635e-03 break | yes (3.6349e-03; 1.19e-02 fresh) | 4.6 decades under the break, 4 decades over the correct arm; the break is stable to 3e-13 across threads | none | keep |
| 19 | `1e-12` | `g8a_split_tensor_layer_equals_the_single_layer` (x3) | T8a's 8.74e-16 | yes | 480x (the value moved 58% across the thread change -- 8.74e-16 -> 2.08e-15 -- so quote 480x, not 1144x) | none | keep |
| 20 | `1e-11` | `g8b_uniform_tensor_multilayer_matches_berreman` | 1e2+ x T8b's 4.21e-14 | yes | 238x (build), 170x (fresh) | none | keep |
| 21 | `10.0` (M5->M7 ratio) | same test | T8b's 57x | yes | **5.7x** -- under a decade, and the M=6 residual (2.09e-14) is BELOW the M=7 one, so the ratio references the roundoff plateau; it moved 10% across the thread change and is only 2.1x on a fresh fixture | S1 | **RESTATED** to an M=3 -> M=7 span >= 1e5 (measured 1.32e9, 4 decades; M=3 bit-identical across threads) (`6470ad6`) |
| 22 | `1e-5` / `10.0` / `0.05` / `1e-9` | `g9_absorption_budget` | 1e2 x T9's 8.21e-08; 263x drop; A(lossy) 0.254; A(lossless) 5.59e-14 | yes (T9 reproduced exactly) | 122x / 26x / 5x / 4.3 decades.  Fresh fixture: 6.7x on the 1e-5 | none, but note the 1e-5 is fixture-calibrated like G4/G5 | keep |
| 23 | `(3, 2)` shape assertion | `g9_absorption_budget` | the stack has 3 layers, 2 polarizations | n/a | exact | NOT an S5 census -- a deterministic API shape | keep |
| 24 | `0.0 < off < 1e-12 * max` | `g10_offplane_test_is_relative_not_strict` | `cos(pi/2) = 6.1e-17` float noise, measured 5.17e-17 vs floor 2.97e-12 | yes (T10) | 4.8 decades on the upper side; the LOWER side is an exact-nonzero assertion on roundoff | S5-adjacent: `np.cos(np.pi/2)` is 6.123233995736766e-17 in IEEE double on every conforming build, so the risk is small, but a library that ever returned an exactly-diagonal rotated tensor would turn this red for the right reason | **NOT CHANGED** -- this is one of the out-of-plane guard functions reserved for the integration agent.  Reported here |
| 25 | exception types / `match=` strings | `g10_*` (5) | T10 | yes | decisions, not readings | none.  `_NAMES_HYBRID = r"Use pmm_jones_2d \(the hybrid"` correctly defeats the `pmm_jones_2d` / `pmm_jones_2d_staggered` prefix trap | keep |
| 26 | `1e-5` | `tripwire_silent_on_a_well_resolved_hermitian_tensor_stack` (implicit), `..._fires_...` silent arm | T6/T11 | yes | 256x | none | keep |
| 27 | `0.95` | `tripwire_silent_on_a_non_hermitian_tensor` | T11's 0.227 deficit -> R+T = 0.773 | yes | 0.177 absolute | a physical magnitude | keep |
| 28 | `2e-1` + `pytest.warns` | `tripwire_fires_on_an_engineered_lossless_violation` | measured 6.56e-01 at `s = 0.1` | yes (this doc) | 3.3x on the deviation, 13.1x on the window crossing, silent arm 1.3e6x inside | two-sided, 7-decade bracket | added by this verification |
| 29 | `1e-2` / `1e-5` / `1e4` | `tripwire_underresolved_closure_ladder` | 9.08e-02 / 1.51e-07 / 6.0e5 | yes (this doc) | 9.1x / 66x / 60x | none | added by this verification |

**S3 (env-dependent precondition): none.**  The file's only environment
interaction is `os.environ.setdefault` of the three thread caps at import,
which forces a state rather than reading one, and there is **no `pytest.skip`
anywhere in the file** -- the shape that silently removed five tests from the
gate in the v5.35 campaign.

**S5 (exact count/set of nondeterministic machinery): none.**  No mode census,
no eigenvalue count, no length comparison.  The only exact assertions are the
G1 bit-identity pair (a same-build, same-process, same-BLAS dispatch claim,
verified on four fixtures here) and the `(3, 2)` array shape.

**S2 (pre-fix-referencing arm): none.**  Every fail-before constructs its
broken state in-process by monkeypatch; nothing references a prior version.

---

## 8. Task 8 -- suites and lint

| run | result |
|---|---|
| `tests/unit/test_pmm2d_staggered_anisotropic.py` (after this verification's changes) | **41 passed, 64.26 s** |
| slowest tests | 10.65 s `g5_three_engines`, 6.93 s `g3_multisegment`, 4.64 s `g9_absorption`, 3.45 s `g6_closure_improves`, 3.14 s `tripwire_fires`, 2.02 s `g3_jones_gauge...` |
| the 8 shipped staggered / pure suites, `-W always` | **140 passed, 297.66 s**, zero tripwire firings |
| the 17-file regression slice (this file + `test_v5_14_0_pmm2d_stack`, `test_v5_12_0_pmm2d_loss`, `test_v5_14_0_pmm2d_cell`, `test_audit_w3_entry_validation`, `test_audit_w6_pmm_rcwa`, `test_v5_14_0_pmm_jones_2d`, `test_public_api`, `test_v4_16_0_walker_all_symmetry`, and the 8 staggered / pure suites), `-W always` | **978 passed, 571.08 s**, zero tripwire firings (the 32 warnings are 14 Deprecation, 12 User and 4 `_EnergyWarning`, none of them `PMM2DStackPure.solve`) |
| `ruff check lumenairy/ tests/` | **All checks passed** |
| `ruff check validation/probe_verify_staggered_aniso/` | **All checks passed** |

The file is inside the plan's budget (< 3 min per file, < 40 s per test,
grids <= (3,3), M <= 8): it grew from 37 tests / 53.8 s to 41 / 64.3 s.

---

## 9. Library defects found

**None.**  Every claimed behaviour re-measured as claimed, on the build's
fixtures and on fresh ones.  Two limitations worth recording (neither is a
defect against a stated contract):

1. **The closure tripwire is blind to a modal-admittance error.**  Rescaling
   the `V` block of every region's modes by 1.5 changes the order-0
   reflectance by 0.209 and the Jones by 0.223 while `sum R + sum T` stays at
   1.3e-07 (section 5c).  No closure gate can catch that class; G3/G4/G5 are
   what cover it.  Worth a line in the guard's docstring if anyone is tempted
   to read a silent tripwire as correctness.
2. **The Wood-anomaly `eps` list is inconsistent between the scalar and
   tensor paths, with a measured consequence** -- the build doc's own open
   item 6, now with a reproducer
   (`validation/probe_verify_staggered_aniso/v8_wood_anomaly_path_split.py`).
   `PMM2DStackPure.solve` feeds TENSOR layers' diagonals into
   `_grazing_safe_wavelength` and deliberately omits SCALAR layers, so a
   scalar cell and its `e*I` promotion take different wavelength nudges when
   an order sits on a LAYER's own Rayleigh cutoff.  Put order `m = 1` exactly
   there (uniform `eps = 4` layer, normal incidence, `wl/px = 2 = sqrt(4)`):

   | `px` | `wl` | max\|dR\| | max\|dT\| | bit-identical |
   |---|---|---|---|---|
   | 0.5 um | 1.0 um (ON the cutoff) | **4.591e-08** | **4.591e-08** | **no** |
   | 0.5 um | 1.0 um x (1 - 1e-9) | 0.0 | 0.0 | yes |
   | 0.7 um | 0.55 um (far from any cutoff) | 0.0 | 0.0 | yes |

   So G1's "a scalar map is promoted to `e*I`" is exact away from cutoffs and
   differs by ~5e-08 on one.  The nudged (tensor) arm is arguably the more
   correct of the two, and unifying the rule would move shipped scalar
   results, so leaving it is the right call -- but the asymmetry is NEW with
   this build (previously neither 2-D staggered entry included layer eps) and
   the promotion claim should say "away from a layer cutoff".

---

## 10. Changes committed on this branch by the verification

| commit | what |
|---|---|
| `d87fee4` | **G2 becomes a real published-oracle gate** -- Li 2003 example 1 / table 1, six reflected orders to 8.74e-05 at M=8, plus the two-row cross-term sign discriminator.  Replaces the "not reproduced" record and a sign claim whose reference was ambiguous.  Tests only |
| `6470ad6` | tripwire fail-before restated with an engineered non-unitary dispersion (13.1x outside the window vs 1.3e6x inside, replacing 1.82x); the physical under-resolution arm kept as its own ladder test; G8b's M5->M7 ratio (5.7x of headroom, plateau-referencing) restated as an M3->M7 span >= 1e5 (1.32e9 measured).  Tests only |
| `7fab15c` | two new tests for the Jones properties every efficiency gate is blind to: the PUBLIC gauge + row/column contract vs Berreman on a GYROTROPIC slab (with the LC no-op recorded), and the `|jones|^2` <-> order-0 efficiency identity including the incident longitudinal component.  Tests only |
| `3e15470` | `validation/probe_verify_staggered_aniso/` -- eight probes and their recorded readings |
| `4ffcbaf` | the measured last-bit envelope and fixture sensitivity written into the two docstrings whose bars have under a decade of margin (G4's 3x, G5's 8.6x).  No bar changed |
| (this file) | the verification report |

No library file was touched.  No merge, push, tag or version bump.

## 11. What could not be verified

* **Cross-platform / cross-LAPACK.**  Everything here is one machine.  The
  partial substitute is the `OPENBLAS_NUM_THREADS` 1 vs 4 probe (`v7`), which
  exercises different OpenBLAS kernels and reduction orders; it is reported
  per-bar in section 7 but it is NOT a second wheel.  The build doc's open
  item 10 stands.
* **The JAX legs and the full library suite** were not run (the staggered path
  is NumPy-only, and the plan scopes the suite list).
* **Li's own error.**  The 8.74e-05 residual at M=8 is comparable to Li's own
  convergence spread (his fig. 7 spans 8e-4 on the (0,0) order across Fourier
  representations at truncation 23), so the comparison bounds this build's
  error by the oracle's, and no tighter statement is available from a
  published table.
* **Stage B (out-of-plane)** is out of scope here; the guard tests for it are
  reserved for the integration agent and were not modified.
