# BUILD -- OUT-OF-PLANE anisotropy in the PURE (no-floor) staggered 2-D PMM

Date: 2026-09-09.  Branch `feat/pmm2d-staggered-oop`, worktree
`C:/tmp/lum_aniso_oopint`, on top of `78dca9a` (Stage A in-plane build merged
with the Stage B prototype).  Stage B *integration* of
`docs/audits/PLAN_PMM2D_STAGGERED_ANISOTROPIC_2026_09_09.md` Section 4, taking
the GO route of `docs/audits/EXPERIMENT_PMM2D_STAGGERED_OOP_2026_09_09.md`
(candidate (a), the first-order staggered generator on `[E1; E2; G1; G2]`).

**Mount.**  Windows 11, python 3.14.6, numpy 2.4.4, scipy 1.17.1
(scipy-openblas), tesla-ryzen.  Every probe and test exported
`OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = MKL_NUM_THREADS = 1` **before**
python started and asserted `lumenairy.__file__` under
`C:\tmp\lum_aniso_oopint`, version 5.42.1.  Every number below is printed by a
script in `validation/probe_pmm2d_staggered_oop/` (logs in its `logs/`, JSON in
its `results/`).

---

## 0.  Summary

The out-of-plane path is **shipped**.  A uniform out-of-plane slab reproduces
the exact Berreman 4x4 to `1e-14` on R, T and both Jones matrices at normal,
oblique and conical incidence for lossless, lossy and NON-RECIPROCAL tensors;
a y-uniform out-of-plane stripe matches the two 1-D engines per order to
`1.6e-05` (their own mutual spread being `3.2e-06`) with y-momentum conserved
to `1e-27`; a genuinely 2-D out-of-plane cell with a RE-ENTRANT corner sits at
the two Fourier oracles' own mutual spread; the cascade closes to `1e-13` at
depths up to three wavelengths with a forward growth factor of exactly
`1.0000e+00`; and the no-floor property survives (the result moves `2.6e-15`
when the far-field order count nearly triples).  Cost is **1.33 - 2.03x** the
in-plane region solve and **~3.0x** its peak working set.

**GATE 0 changed the story of the prototype's one open item.**  The 2.20e-03
departure the experiment attributed jointly to "the out-of-plane coupling AND a
re-entrant corner" is neither: it is a **180-degree pattern-orientation
convention in the probe's own driver**, and it does not exist in the library
gauge.  Section 1 is the measurement; Section 2 is what the integration had to
carry because of it.

---

## 1.  GATE 0 -- adjudicating the prototype's open item

The brief's gate: three ladders on the M9 `(3,3)` L cell (an L of three
tilted-uniaxial pixels, `px = py = 1.2 lam`, `depth = 0.4 lam`, `n_sub = 1.5`,
tensor `uniaxial(1.5, 1.7, tilt 35, azim 25)`), and STOP if the staggered arm
does not converge or converges somewhere the Fourier arms do not approach.

`validation/probe_pmm2d_staggered_oop/g0_corner_gate.py`, normal incidence.

### T0.1  is the staggered arm self-convergent?  (YES)

`sum R`, both incident polarizations, out-of-plane tensor:

| M | dim | sum R | step \|d\| | own \|R+T-1\| | t [s] |
|---|---|---|---|---|---|
| 4 | 324 | 0.03766 / 0.03864 | -- | 1.94e-04 | 0.2 |
| 5 | 576 | 0.03766 / 0.03864 | 1.65e-05 | 2.02e-06 | 1.1 |
| 6 | 900 | 0.03766 / 0.03864 | 1.02e-05 | 1.32e-07 | 3.5 |
| 7 | 1296 | 0.03765 / 0.03864 | 2.11e-06 | 5.73e-10 | 10.6 |
| 8 | 1764 | 0.03765 / 0.03864 | 5.51e-07 | 3.60e-11 | 26.5 |
| 9 | 2304 | 0.03765 / 0.03864 | **4.80e-07** | **1.32e-14** | 63.6 |

Per order, M=8 -> M=9: R moves `1.10e-06`, T `3.15e-06`, Jones `1.57e-06`.
**Self-convergent.**

### T0.2  how far does each Fourier arm move? (own drift at the ladder top)

| arm | ladder | top step on `sum R` | own \|R+T-1\| at the top |
|---|---|---|---|
| `pmm_jones_2d` laurent | 7, 9, 11, 13 | 4.41e-05 | 1.16e-06 |
| `pmm_jones_2d` li | 7, 9, 11, 13 | 8.95e-05 | 1.53e-06 |
| `rcwa_jones_2d` | 5, 7, 9 | 2.38e-05 | 2.33e-13 |
| staggered | 4 .. 9 | 4.80e-07 | 1.32e-14 |

Cross-arm on `sum R`: hyb-laurent(13) vs hyb-li(13) `4.06e-05`, hyb-laurent
vs rcwa(9) `1.20e-04`, hyb-li vs rcwa `1.31e-04`; staggered(9) vs the three
`2.20e-03 / 2.18e-03 / 2.30e-03`.

### T0.3  the in-plane control on the SAME walls -- and the smoking gun

The prototype doc's 2-D comparisons (M5, M8, M9) are all on **totals**.  Per
order the picture is different, and the IN-PLANE control -- the shipped
discretization, same walls, same corner -- fails too:

per-order T (incident Ex), top of each ladder, IN-PLANE control:

| order | staggered | hyb-laurent | hyb-li | rcwa |
|---|---|---|---|---|
| (0,0) | 0.7275733 | 0.7273961 | 0.7280813 | 0.7265545 |
| (1,0) | **0.0449443** | 0.0314476 | 0.0311695 | 0.0313119 |
| (-1,0) | **0.0315975** | 0.0448193 | 0.0446129 | 0.0447521 |
| (0,1) | **0.0500264** | 0.0562677 | 0.0563543 | 0.0565478 |
| (0,-1) | **0.0561526** | 0.0502810 | 0.0501758 | 0.0507558 |

The probe's orders are **SWAPPED with `(-m, -n)`** -- `per-order dR 2.86e-03`,
`dT 1.80e-02` against the oracles, on the IN-PLANE arm, at NORMAL incidence,
where the totals still agree to 3.5e-05 because a total is mirror-invariant.
The out-of-plane arm is not a pure mirror (per-order `dR 4.47e-03`,
`dT 2.69e-02`) because a 180-degree rotation acts on the TENSOR too.

### T0.4  attribution, four ways

* **`g0b_per_order_2d.py`** -- every stripe (x- or y-patterned, director
  azimuth 0 / 90 / 25) and the (2,2) single-pixel pillar are BLIND to it,
  because each is 180-degree symmetric up to a lattice shift; there the
  out-of-plane and in-plane arms agree with each other to the printed digits
  (e.g. x-stripe azim25: out-of-plane `dR 2.73e-04` vs in-plane `2.93e-04`
  against `rcwa(9)`, whose own 7 -> 9 drift is `5.7e-05`).  The (3,3) L is the
  only cell in the whole probe that is not.
* **`g0f_basis_phase_slope.py`** -- the SHIPPED basis carries
  `exp(-i alpha0 x)`: the reconstructed order-m field's phase slope measures
  `-2.65539` against `alpha0 = +2.65539` for m=0, `-6.35139` vs
  `-(alpha0 + G) = -6.35138` for m=+1, `+1.04060` vs `-(alpha0 - G)` for m=-1.
* **`g0e_gauge_vs_oracle_1d.py`** -- on a CHIRAL 3-segment cell
  (`eps = [1, 2.25, 4]`, not a translate of its own mirror) at 25 degrees, the
  SHIPPED path reproduces `rcwa_jones_1d_segments` order for order
  (`1.0e-07` TE, `5.2e-06` TM) while the PROBE reproduces the oracle at
  `(-theta, -m)` instead, `1.03e-02` away from the correct answer.
* **`g0c_rho_attribution.py`** -- feeding the PROBE the 180-degree-ROTATED L
  cell collapses its disagreement:

| probe arm, M=6 | per-order dR vs rcwa(9) | dT | dJones |
|---|---|---|---|
| cell as-is | 4.51e-03 | 2.68e-02 | 5.41e-03 |
| **rho(pattern)** | **7.24e-05** | **1.17e-03** | **1.67e-03** |
| rho(tensor) only | 2.61e-03 | 2.81e-02 | 1.67e-03 |
| rho(pattern) + rho(tensor) | 1.95e-03 | 2.41e-02 | 5.41e-03 |
| *oracles' own spread* hyb(13) vs rcwa(9) | *4.84e-05* | *9.52e-04* | *1.00e-03* |
| *rcwa's own 7 -> 9 drift* | *3.54e-05* | *4.12e-04* | *4.12e-04* |

* **`g0d_shipped_orientation.py`** -- the SHIPPED in-plane path does NOT have
  it: on the same L cell (in-plane tensor) it agrees with `rcwa(9)` per order
  at `1.74e-03` (normal, M=6) and `2.12e-03` (conical 20/35), against
  `2.44e-02` / `5.47e-02` for the mirror.

### VERDICT

**GATE 0 PASSES, and the open item is CLOSED as a convention, not a corner.**
The staggered arm is self-convergent (4.8e-07 at the top of its ladder against
Fourier drifts of 2.4e-05 .. 1.2e-04), and once the pattern orientation is
right it lands inside the oracles' own mutual spread.  The re-entrant corner is
not implicated: it was simply the only shape in the prototype whose
180-degree image is not itself.  The library gauge does not carry the defect,
but the gauge IS load-bearing for the tensor, which is what Section 2 is about.

---

## 2.  The rotation gauge, and what the integration carries

`Basis1D`'s glue `tau = exp(-i alpha0 p)` makes the basis run as
`exp(-i alpha0 x)` (T0.4), while `_stag_fourier_projection`'s kernel and the
`eps_cell` indexing run the other way.  Composed, the shipped solve is the
180-degree rotation about `z` (`rho`) of the physical one, reported in
`rho` coordinates.

* Every IN-PLANE tensor component (`e11, e12, e21, e22, e33`) is INVARIANT
  under `rho` -- which is exactly why the isotropic solver and the Stage-A
  in-plane path never had to know about it, and why every Stage-A gate passes.
* The four OUT-OF-PLANE components change sign under `rho`.

So the assembly negates them once (`twod_staggered._OOP_ROT_SIGN = -1.0`).  The
equivalent statement inside the generator -- proved as a discrete identity and
recorded in `_assemble_oop`'s docstring -- is that negating the four
single-derivative blocks (`P13, P23, CwE1, CwE2`) AND the four out-of-plane
entries leaves the pencil invariant term for term, so "the basis' transverse
derivative runs backwards" and "the tensor is rotated 180 degrees" are one
statement.

### T2  the sign, arbitrated against `berreman_jones_1d`

`g1_rot_sign_berreman.py`: uniform slab, `px = py = 0.9 lam`,
`depth = 0.35 lam`, `n_sub = 1.5`; director azimuth **25 deg**, conical
incidence azimuth **40 deg** (the S6 degeneracy is avoided deliberately).
`max |dR|` and `max |dJones|` over both incident polarizations:

| tensor | mount | M | rot = -1 (shipped) | rot = +1 |
|---|---|---|---|---|
| lossless | normal | 8 | 3.5e-15 / 2.1e-14 | 3.5e-15 / 2.1e-14 |
| lossless | oblique 25 | 6 | 7.93e-12 / 1.92e-11 | 1.09e-05 / 2.478e-03 |
| lossless | oblique 25 | 8 | **7.97e-15 / 2.11e-14** | 1.09e-05 / 2.478e-03 |
| lossless | conical 25/40 | 6 | 2.36e-13 / 1.21e-12 | 7.80e-06 / 1.398e-03 |
| lossless | conical 25/40 | 8 | **1.99e-14 / 8.43e-14** | 7.80e-06 / 1.398e-03 |
| lossy 0.08 | conical 25/40 | 8 | **1.99e-14 / 8.11e-14** | 2.82e-05 / 1.391e-03 |
| NON-RECIPROCAL | oblique 25 | 8 | **7.78e-15 / 2.32e-14** | 1.969e-03 / 5.738e-03 |
| NON-RECIPROCAL | conical 25/40 | 6 | 2.38e-13 / 1.20e-12 | 1.677e-03 / 4.490e-03 |
| NON-RECIPROCAL | conical 25/40 | 8 | **2.06e-14 / 8.54e-14** | 1.677e-03 / 4.490e-03 |

Order leak (any non-(0,0) order of a uniform cell) is `1e-27 .. 4e-26` in every
row.  The lossy rows reproduce Berreman's own absorption deficit
(`|R+T-1| = 1.06e-01` normal, `1.13e-01` oblique, `1.12e-01` conical) to the
same `1e-14`.

**Reading.**  `-1` is right by 9 to 11 decades wherever the measurement can
see the sign at all, and at NORMAL incidence on a uniform cell BOTH signs agree
-- the rotation maps that configuration onto itself, so the normal mount cannot
gate the sign.  That asymmetry is itself asserted, two-sided, in
`test_g7_rotation_gauge_is_two_sided`.

### T3  the assembled generator's dispersion, and the sum-of-roots discriminator

Max over the fundamental harmonic's four exact quartic roots of the distance to
the nearest generator eigenvalue, `(2,2)` grid, `px = py = 0.9 lam`, M=8:

| tensor | mount | at `+k_t` (physical) | at `-k_t` (the wrong gauge) | `sum` of the four (0,0) roots |
|---|---|---|---|---|
| tilt35 azim25 | normal | **2.37e-14** | 2.37e-14 | 0 (symmetric at `k_t = 0`) |
| tilt35 azim25 | conical 25/40 | **1.49e-14** | 9.16e-02 | -0.091625 |
| NON-RECIPROCAL | normal | **2.37e-14** | 2.37e-14 | 0 |
| NON-RECIPROCAL | conical 25/40 | **2.98e-14** | 8.41e-02 | -0.091625 |

Higher harmonics converge spectrally, as the prototype found: at conical,
harmonic `(-1,0)` is `6.6e-08` at M=6 and `3.8e-12` at M=8.  The sum of the
fundamental's roots is exactly zero for any in-plane tensor and non-zero only
through the out-of-plane coupling, so it is the discriminator the factor-i
audit asks for -- and it is identical for the symmetric and the non-reciprocal
tensor, which is the transpose-blindness theorem (S5 T3) reproduced here: only
the FIELDS (T2) can see an `e13`/`e31` swap.

### T4  the in-plane reduction, and the H gauge

`g2_inplane_reduction.py`.  The dispatch is forced by patching the relative
off-plane test, so the SAME cell runs both arms.  Cross terms EXACTLY zero.

| cell | mount | M | dR | dT | dJones |
|---|---|---|---|---|---|
| in-plane uniaxial pillar | normal | 5 | 3.75e-15 | 1.58e-14 | 8.48e-15 |
| in-plane uniaxial pillar | normal | 6 | 3.75e-14 | 1.08e-14 | 4.49e-14 |
| in-plane uniaxial pillar | conical 20/35 | 6 | 3.32e-14 | 2.51e-14 | 4.11e-14 |
| gyrotropic pillar | conical 20/35 | 6 | 2.13e-14 | 2.17e-14 | 2.70e-14 |
| lossy diagonal pillar | normal | 6 | 3.17e-14 | 6.06e-14 | 3.99e-14 |
| uniform in-plane uniaxial | conical 20/35 | 6 | 6.76e-15 | 9.99e-15 | 1.27e-14 |

Worst over all 16 rows: **4.49e-14**.  Spectra (max distance from the
generator's `4q^2` eigenvalue set to the E-form's `{+q, -q}`), M=6, dim 400:
`1.38e-12` / `1.12e-12` (in-plane uniaxial, normal / oblique), `7.82e-13` /
`7.03e-13` (gyrotropic) -- the growth with dimension of an eigenvalue
conditioning number, not of a discretization error.

This table is also the ONLY gate on `_OOP_H_GAUGE = -1j`, the constant relating
the generator's `G = i Z0 H` state to the Eq.-25 partner the ISOTROPIC
half-spaces carry: it cancels in a pure out-of-plane stack and does not cancel
in a mixed one, which is what this comparison is.

Dispatch floor: a `1e-16` stray in an `xz` slot leaves `offplane` False and
gives **byte-identical** R, T and Jones to the clean cell; `1e-3` sets it True.

---

## 3.  Validation tables the test bars derive from

`g3_integration_tables.py` unless stated.

### T5  y-uniform out-of-plane STRIPE, per order, both polarizations

Ridge = `uniaxial(1.5, 1.7, 35, azim 25)`, groove = air, `px = py = 1.2 lam`,
`depth = 0.4 lam`, duty 0.5 -- a cell with four DIELECTRIC CORNERS.  Oracles:
`rcwa_jones_1d_segments(n_orders=41)` and `pmm_jones_1d(degree=18,
far_field_orders=31, stabilize=False)`, two independent engines.

**The oracles' own spread is the bar:** normal `dR 3.16e-06`, `dT 3.84e-05`,
`dJones 7.30e-06`; oblique 25 `3.30e-06`, `3.21e-05`, `1.29e-05`.

| mount | M | dim | dR vs rcwa1d | dT | dJones | y-leak R / T | \|R+T-1\| |
|---|---|---|---|---|---|---|---|
| normal | 5 | 256 | 1.20e-04 | 8.29e-04 | 6.73e-04 | 1.1e-29 / 1.4e-28 | 2.66e-04 |
| normal | 6 | 400 | 4.64e-05 | 2.23e-04 | 2.20e-04 | 6.6e-28 / 1.5e-27 | 7.48e-06 |
| normal | 7 | 576 | 2.23e-05 | 1.09e-04 | 2.08e-04 | 1.0e-27 / 4.5e-27 | 2.81e-07 |
| normal | 8 | 784 | **1.59e-05** | **1.04e-04** | **9.51e-05** | 2.6e-27 / 1.2e-26 | 1.28e-08 |
| oblique 25 | 5 | 256 | 7.93e-04 | 6.64e-03 | 2.87e-03 | 1.4e-29 / 4.6e-29 | 1.51e-03 |
| oblique 25 | 6 | 400 | 1.59e-05 | 2.97e-04 | 2.39e-04 | 6.8e-28 / 6.7e-28 | 7.53e-05 |
| oblique 25 | 7 | 576 | 1.14e-05 | 1.40e-04 | 1.25e-04 | 1.1e-27 / 5.4e-28 | 5.29e-06 |
| oblique 25 | 8 | 784 | **7.38e-06** | **1.15e-04** | **8.77e-05** | 3.9e-27 / 6.0e-27 | 1.16e-07 |

Monotone, and the residual floor is the shipped basis's documented CORNER CAP
(the prototype measured 1.4e-05 on the same cell, and the same cell with the
cross terms zeroed caps the same way -- GATE 0b) rather than an out-of-plane
defect.  **Y-momentum is conserved to 1e-27**: a y-invariant cell scatters
nothing into `(m, n != 0)`, which the y-carrying blocks `A23`/`A32` would break
if mis-placed.

### T6  the (3,3) RE-ENTRANT-corner cell, PER ORDER, library gauge

Oracles: `rcwa_jones_2d` at `n_orders` 7 and 9 (pixel-replicated cell) and
`pmm_jones_2d(degree=9, n_orders=13)` in BOTH `E_z` rules.

| mount | comparison | dR | dT | dJones |
|---|---|---|---|---|
| normal | rcwa 7 -> 9 (its own drift) | 3.46e-05 | 5.56e-04 | -- |
| normal | hyb-laurent(13) vs rcwa(9) | 4.84e-05 | 9.52e-04 | 1.00e-03 |
| normal | hyb-li(13) vs rcwa(9) | 6.40e-05 | 1.73e-03 | 9.64e-04 |
| normal | **staggered M=5** | 6.62e-05 | 1.10e-03 | 1.78e-03 |
| normal | **staggered M=6** | 7.24e-05 | 1.17e-03 | 1.67e-03 |
| normal | **staggered M=7** | 7.58e-05 | 1.17e-03 | 1.67e-03 |
| conical 20/35 | rcwa 7 -> 9 | 5.89e-05 | 9.65e-04 | -- |
| conical 20/35 | hyb-laurent(13) vs rcwa(9) | 9.28e-05 | 1.35e-03 | 1.22e-03 |
| conical 20/35 | hyb-li(13) vs rcwa(9) | 9.02e-05 | 2.02e-03 | 1.34e-03 |
| conical 20/35 | **staggered M=7** | 1.63e-04 | 1.90e-03 | 1.98e-03 |

The staggered arm's own self-move over M=6 -> 7 is `9.26e-06` (normal) /
`5.59e-06` (conical) and its energy closure reaches `5.73e-10` / `6.32e-08`,
while no Fourier arm is converged: the two `E_z` elimination rules of the SAME
hybrid, both rigorous, disagree with each other by `7.73e-04` (normal) and
`6.73e-04` (conical) at `n_orders = 13`.  So the honest statement is that on a
re-entrant-corner out-of-plane cell the staggered result sits AT the oracles'
mutual spread, and no engine in the suite currently pins that regime better
than ~1e-4 on R per order.

**NO-FLOOR, two-sided.**  Far-field `n_orders` 3 -> 8 moves the staggered
result by `2.55e-15` (normal) and `3.94e-09` (conical); the hybrid's two `E_z`
rules differ by `7.73e-04` / `6.73e-04`.  Eleven and six decades of contrast.

### T7  cascade stability and closure vs DEPTH

`px = py = 1.2 lam`, M=7, depths 0.25 / 1 / 3 wavelengths.  `max fwd growth` is
`max exp(-Re(lam_f) k0 L)` at 3 wavelengths -- any value above 1 means a
growing mode was classified forward.

| cell | mount | 0.25 lam | 1 lam | 3 lam | fwd/bwd | max fwd growth |
|---|---|---|---|---|---|---|
| uniform Hermitian | normal | 6.84e-14 | 8.26e-14 | 2.05e-13 | 288/288 | 1.0000e+00 |
| uniform Hermitian | conical 25/40 | 8.82e-14 | 1.25e-13 | 2.24e-13 | 288/288 | 1.0000e+00 |
| pillar Hermitian | normal | 7.23e-08 | 2.43e-07 | 2.01e-07 | 288/288 | 1.0000e+00 |
| pillar Hermitian | conical 25/40 | 7.24e-07 | 8.30e-07 | 2.69e-06 | 288/288 | 1.0000e+00 |
| pillar LOSSY (absorbed) | normal | 2.38e-02 | 1.61e-01 | 4.72e-01 | 288/288 | 8.6802e-01 |
| pillar LOSSY (absorbed) | conical 25/40 | 2.95e-02 | 1.68e-01 | 4.34e-01 | 288/288 | 8.8946e-01 |

Closure does NOT grow with depth (a growing-mode leak would multiply by
`exp(2 * 3 * k0 * something)` between the first and last column); the split is
exactly `2q^2 / 2q^2` in every row; absorption rises monotonically and stays in
`[0, 1]`.

### T8  stacks

| claim | mount | measurement |
|---|---|---|
| one 0.4-lam out-of-plane layer == two 0.2-lam layers, per order / Jones | normal | 8.88e-16 / 4.06e-16 |
| same | conical 20/35 | 3.33e-16 / 2.81e-16 |
| uniform out-of-plane MULTILAYER (lossless + non-reciprocal + lossy) vs `berreman_jones_1d`, dR / dT / dJones, M=6 | oblique 25 | 8.26e-12 / 4.30e-11 / 2.34e-11 |
| same, M=8 | oblique 25 | **7.47e-15 / 1.27e-13 / 3.10e-14** |
| same, M=6 | conical 25/40 | 3.06e-13 / 1.57e-12 / 1.25e-12 |
| same, M=8 | conical 25/40 | **1.25e-14 / 1.21e-13 / 6.50e-14** |

`retain_internal` / `layer_absorption` on the generalized cascade (out-of-plane
over in-plane over uniform, conical 20/35):

| stack | M | \|R+T-1\| | \|sum A - (1-R-T)\| | max per-layer \|A_i\| |
|---|---|---|---|---|
| LOSSLESS mixed (3 layers) | 5 | 5.75e-04 | 5.75e-04 | 6.63e-15 |
| LOSSLESS mixed | 6 | 7.29e-05 | 7.29e-05 | 8.44e-15 |
| LOSSLESS mixed | 7 | 4.32e-06 | 4.32e-06 | 2.86e-14 |
| LOSSLESS mixed | 8 | 1.78e-07 | 1.78e-07 | 5.30e-14 |
| LOSSY out-of-plane + in-plane | 5 | -- | 9.14e-04 (rel 2.5e-02) | absorbed 3.57e-02 |
| LOSSY | 6 | -- | 7.62e-05 (rel 2.1e-03) | absorbed 3.57e-02 |
| LOSSY | 7 | -- | 4.14e-06 (rel 1.1e-04) | absorbed 3.57e-02 |
| LOSSY | 8 | -- | **2.09e-07 (rel 5.6e-06)** | absorbed 3.57e-02 |

The budget residual falls two decades per degree, i.e. it is
DISCRETIZATION-limited, not a formula error; on the lossless stack every
per-layer absorption is at roundoff.  So `layer_absorption` WORKS on the
out-of-plane path and does not need the "refuse loudly" fallback the brief
allowed.

### T9  fail-before controls (uniform slab vs Berreman, conical 25/40, M=8)

Director azimuth 25 deg, incidence azimuth 40 deg -- chosen so the S6
degeneracy is not hit.

| tensor | control | dR | dJones | decades above the reference |
|---|---|---|---|---|
| lossless azim25 | *reference* | *1.99e-14* | *8.43e-14* | -- |
| lossless azim25 | drop OOP | 1.55e-04 | 1.54e-03 | 10 / 10 |
| lossless azim25 | negate OOP | 7.80e-06 | 1.40e-03 | 8 / 10 |
| lossless azim25 | transpose OOP | 1.99e-14 | 8.43e-14 | none -- `e13 = e31` here, a NO-OP |
| NON-RECIPROCAL | *reference* | *2.06e-14* | *8.54e-14* | -- |
| NON-RECIPROCAL | drop OOP | 1.04e-03 | 3.22e-03 | 11 / 11 |
| NON-RECIPROCAL | negate OOP | 1.68e-03 | 4.49e-03 | 11 / 11 |
| **NON-RECIPROCAL** | **transpose OOP** | **1.67e-03** | **4.49e-03** | **11 / 11** |

Both prototype degeneracies are reproduced and both are handled: the transpose
is invisible on a symmetric tensor (trivially), and the negate would be
invisible if the director azimuth equalled the incidence azimuth.

### T10  cost -- dimension, wall time, peak traced allocation

`(3,3)` grid, uniform tilted-uniaxial cell, region assembly + mode solve, single
threaded.  In-plane arm = the SAME cell with the cross terms zeroed, so the two
arms differ only in the formulation.

| M | in-plane dim / t / peak | out-of-plane dim / t / peak | t ratio | peak ratio |
|---|---|---|---|---|
| 5 | 288 / 0.29 s / 15.4 MB | 576 / 0.58 s / 46.3 MB | 2.03x | 3.01x |
| 6 | 450 / 0.97 s / 37.3 MB | 900 / 1.55 s / 112.9 MB | 1.60x | 3.03x |
| 7 | 648 / 2.71 s / 77.1 MB | 1296 / 3.95 s / 234.0 MB | 1.46x | 3.03x |
| 8 | 882 / 7.05 s / 142.7 MB | 1764 / 9.38 s / 433.4 MB | 1.33x | 3.04x |

The time ratio FALLS with size and is far below the `(4/2)^3 = 8x` a dimension
count predicts, because the in-plane path pays a QZ (`scipy.linalg.eig(L, G)`)
while the out-of-plane path pays a Cholesky-whitened standard eig on a matrix
twice the size.  The peak allocation ratio is flat at ~3.0x (the generator is
`4q^2` square plus the two `2q^2 x 4q^2` elimination solves).

---

## 4.  Equations to code

| object | experiment (S1/S2) | code |
|---|---|---|
| `Ggram1`, `Ggram2`, `Gw` | `M1`, `M2`, `Mw` | `_assemble_oop`, `kron` of `Mtt/Mbb` |
| `CwE1 = <Vw\|D2\|V1>`, `CwE2 = <Vw\|D1\|V2>` | the strong curl | `kron(dbt_y, Mbb_x)`, `kron(Mbb_y, dbt_x)` |
| `P13 = <V1\|D1\|V3>`, `P23 = <V2\|D2\|V3>` | the strong mimetic gradient (`= -Ktz`) | `kron(Mtt_y, dbt_x)`, `kron(dbt_y, Mtt_x)` |
| `A11..A33` (nine eps-weighted masses) | Appendix-A Eq. 40/41 + the four NEW out-of-plane blocks | `self._eps_weighted(...)` with one component map each |
| `g3 = Gw^-1 (CwE2 e2 - CwE1 e1)` | STRONG elimination of `H3` | `G3S` |
| `e3 = A33^-1(P23^H g1 - P13^H g2 - A31 e1 - A32 e2)` | WEAK elimination of `E3`, tested in V3 | `E3S` |
| the four rows of the pencil | S2 | `row0 .. row3`, `Bgen = blkdiag(G1, G2, G2, G1)` |
| Cholesky-whitened standard eig | S13.2 | `_region_modes_oop` |
| flux split on the whitened blocks | S13.3 (PMM_ROADMAP C-FLUX) | `_region_modes_oop`, `_select_forward_flux` |
| `[[W, W], [V, -V]]` for symmetric regions | S13.4 | `_modes_as_general` |
| generalized cascade | S13.4 | `PMM2DStackPure.solve`, `any_oop` branch |
| `_OOP_ROT_SIGN` | NEW (Section 2 here; the experiment's S0/S13.6 flagged the `tau` trap but recommended flipping `tau`, which would move the in-plane path) | module constant |
| `_OOP_H_GAUGE` | S13.5 / M0.4 (`V_shipped = -i V_strong`) | module constant |

`K11 .. K23` -- the six eps-weighted div-D blocks -- are NOT used by candidate
(a): it eliminates `E3` through the longitudinal curl-H row, not through
`div D = 0`.  Only the in-plane path builds them.

---

## 5.  Tests

`tests/unit/test_pmm2d_staggered_oop.py` (new) -- see Section 6 for counts and
durations.  Every bar cites the table above and the date.  Also updated: the
two OUT-OF-PLANE guard functions of
`tests/unit/test_pmm2d_staggered_anisotropic.py`, which asserted the Stage-A
`NotImplementedError`; they now assert the positive behaviour (the dispatch
fires, the solve runs and closes) and, for the relative floor, that a
sub-floor stray gives BYTE-IDENTICAL R/T -- the two-arm form of "the in-plane
path does no additional work".

---

## 6.  Test results, durations, and the suites re-run

Command (from the worktree, every run):

```
PYTHONPATH=/c/tmp/lum_aniso_oopint OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1   MKL_NUM_THREADS=1 python -m pytest -q -p no:randomly <files>
```

| file | tests | duration |
|---|---|---|
| **`tests/unit/test_pmm2d_staggered_oop.py`** (new) | **35 passed** | **90.6 s** |
| `tests/unit/test_pmm2d_staggered_anisotropic.py` (Stage A, two guard functions updated) | 37 passed | 53.9 s |

Slowest tests in the new file: `test_g5_no_fourier_floor_two_sided` 25.5 s (it
runs the hybrid at two truncations so the "the hybrid's answer DOES move" half
of the claim is measured on this build, not remembered),
`test_g8b_uniform_oop_multilayer_matches_berreman_multilayer` 9.6 s,
`test_g6_hermitian_oop_closes_at_three_depths_and_does_not_grow` 6.9 s,
`test_g5_2d_reentrant_corner_cell_matches_the_fourier_oracles` 5.1 s.  All
under the 40 s per-test rule; the file is under the 4 min rule; grids are
(2,2) and (3,3) and `M <= 8` throughout.

Regression suites re-run green on the integrated build:

| file | tests |
|---|---|
| `test_v5_12_0_pmm2d_staggered.py`, `test_v5_21_pmm2d_staggered_oblique.py`, `test_staggered.py`, `test_audit_p1_staggered_guard.py`, `test_p2c_pmm2d_stack_cascade.py`, `test_p2t_pmm2d_tree_cascade.py`, `test_pmm2d_lossless_closure_two_sided.py`, `test_audit_s1_3_pmm2d_lossless_tripwire.py`, `test_v5_14_0_pmm2d_stack.py` | 148 passed, 306.2 s |

`ruff check lumenairy/ tests/` clean.

---

## 7.  Open items -- honest list

1. **No converged reference exists for a re-entrant-corner OUT-OF-PLANE 2-D
   cell.**  T6 shows the staggered arm at the two Fourier oracles' mutual
   spread, but both oracles route their tensor layer through
   `rcwa._core._layer_eigenmodes_tensor`, so their agreement is not independent
   about the out-of-plane blocks, and neither is converged there (the hybrid's
   own two `E_z` rules differ by 7.7e-04).  The staggered arm is internally
   converged (self-move 9.3e-06, closure 5.7e-10) and, by the lossless-trap
   rule, that still proves nothing about per-order correctness.  What WOULD
   settle it: a high-order `rcwa_jones_2d` ladder taken far enough to stop
   moving, or a staircase-refined staggered cell.
2. **The rotation gauge is a compensation, not a fix.**  `_OOP_ROT_SIGN`
   cancels a 180-degree rotation that the shipped basis, far-field kernel and
   `eps_cell` indexing carry between them.  It is measured, gated two-sided and
   documented, but the cleaner end state is to flip `Basis1D`'s `tau` and
   `_stag_fourier_projection`'s kernel together and delete the constant -- a
   change that touches the in-plane path's bytes and therefore belongs in its
   own commit with its own re-validation of every staggered suite.
3. **The normal-incidence involution accelerator was NOT integrated.**  The
   prototype measured the parity-times-sign structure holding on the staggered
   generator (`||R A R + A|| / ||A|| = 2.6e-15 .. 2.8e-14` on uniform and
   centro-symmetric cells, and FAILING at 0.36 - 0.73 on an off-centre pillar,
   which is the two-sided half).  A `2 q^2` eig instead of `4 q^2` at normal
   incidence is available; it needs the same verify-then-use gate
   `_generator_block_eig` already applies, plus a staggered parity map.  Left
   as documented future work rather than shipped unvalidated.
4. **Anisotropic HALF-SPACES** remain out of scope (the Rayleigh match is
   scalar), as in the hybrid.
5. **SLANT x out-of-plane** stays refused, as in `_layer_eigenmodes_tensor`.
6. **The JAX twin**: the staggered path is NumPy-only; nothing here changes
   that.
7. **Non-square grids** (`Nx != Ny`) remain out of scope.
8. **The prototype's `solve_slab` driver is mirrored at oblique and on
   non-180-degree-symmetric patterns** (Section 1).  Its uniform,
   straight-walled and convex measurements (M1, M2, M4, M6, M7 and the M5
   pillar) are unaffected -- those cells are blind to it -- but M8's and M9's
   2-D numbers should be read with Section 1 in hand.  The probe is left as
   run; the correction is one line (`eps_cell[::-1, ::-1]`) and is exercised by
   `g0c_rho_attribution.py`.
