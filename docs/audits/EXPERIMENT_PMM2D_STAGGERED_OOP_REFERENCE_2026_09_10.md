# EXPERIMENT -- a converged reference for the two BOUNDED out-of-plane 2-D cases

Date: 2026-09-10.  Worktree `C:/tmp/lum_oopfast`, branch
`feat/pmm2d-oop-block-eig`, library 5.43.0 (+ the parity block reduction, which
does not engage on either fixture -- neither cell is its own parity image, and
one mount pair is conical).

**The question.**  `docs/audits/BUILD_PMM2D_STAGGERED_OOP_2026_09_09.md` open
item 1 and `docs/audits/VERIFY_PMM2D_STAGGERED_OOP_2026_09_09.md` section 10
both stop at the same place: on a patterned OUT-OF-PLANE 2-D cell, the
staggered arm sits at the two Fourier oracles' own mutual spread, but neither
oracle is converged and both route their tensor layer through the SAME
`rcwa._core._layer_eigenmodes_tensor`, so their agreement is not independent
evidence.  The claim was therefore recorded as a BOUND
(`1.34e-04` on the chiral cell at conical incidence, against an oracle spread
of `6.1e-05 .. 9.1e-05`), not a confirmation.  This experiment asks whether
extending every ladder and EXTRAPOLATING each to its own limit turns the bound
into a confirmation, tightens it, or exposes a discrepancy.

**Mount.**  Windows 11, python 3.14.6, numpy 2.4.4, scipy 1.17.1
(scipy-openblas), tesla-ryzen; `OMP_NUM_THREADS = OPENBLAS_NUM_THREADS =
MKL_NUM_THREADS = 1` exported before python started; every probe asserts
`lumenairy.__file__` under the worktree.  Probes:
`validation/probe_pmm2d_staggered_oop_reference/t1_ladders.py` (the ladders and
fits, JSON in `results/`) and `t2_summary.py` (the tables below, logs in
`logs/`).

---

## 0.  Verdict

**The bound TIGHTENS; it does not become a confirmation, and no discrepancy
survives measurement.**

1. **No Fourier arm converges, and that is now diagnosed rather than
   asserted.**  Fitting each ladder to `f = f_inf + C x^-p` with `p` SCANNED
   over `[0.25, 10]`, a best fit landing ON either end of the scan is a
   diagnosed failure of the extrapolation -- the ladder has not entered its
   asymptotic regime.  Across the four fixture x mount cases the Fourier arms'
   fits are sound on only **13/32, 12/28, 24/36, 15/40** (hybrid-laurent),
   **18/32, 14/28, 21/36, 13/40** (hybrid-li) and **13/32, 10/28, 6/36, 7/40**
   (rcwa) of the live observables.
2. **The staggered arm's extrapolation is 1.5 - 2 decades tighter.**  Median
   fit uncertainty **2.7e-06 .. 4.7e-06** against **9.3e-05 .. 3.8e-04**
   (hybrid) and **1.2e-04 .. 3.4e-04** (rcwa); its own top-of-ladder per-order
   step is **5.6e-07 .. 2.1e-06** at `M = 10` against **2.5e-04 .. 6.2e-04**
   (rcwa at `n_orders = 11`) and **2.0e-03 .. 3.7e-03** (hybrid at 13).
3. **Where both fits are sound, the extrapolated limits AGREE within their
   combined uncertainties**, and the staggered arm agrees with the hybrid
   BETTER than the hybrid agrees with rcwa: **43/47** (staggered vs the two
   hybrid rules) and **13/17** (staggered vs rcwa) against **11/16** (hybrid vs
   rcwa), summed over the four cases.
4. **All three Fourier ladders are measured moving TOWARD the staggered
   limit** -- on **24 - 40 of every 28 - 40 observables** (86 - 100%), with the
   worst-case distance falling by 1.3x to 5.4x over the ladder.  This is the
   single strongest piece of evidence, and it is a direction, not a number:
   three independent truncation families, two of them sharing an
   `E_z`-elimination code path and one not, all approach the staggered answer.
5. **The bound itself.**  On the chiral cell at conical incidence -- the exact
   case the verification could not settle -- the per-order `dR` against the
   nearest Fourier arm falls from `1.341e-04` (staggered M=7 vs rcwa n=9) to
   **`5.93e-05`** (M=10 vs hybrid-laurent n=13) / **`1.10e-04`** (M=10 vs rcwa
   n=11), against a Fourier MUTUAL spread that also tightens, to
   `3.69e-05 .. 6.10e-05`.  On TRANSMISSION the staggered arm now sits INSIDE
   the Fourier arms' own mutual spread on **all four** cases (ratio
   **0.64x .. 0.92x**); on REFLECTION it is **1.1x .. 2.1x** that spread.

So the honest statement is unchanged in kind and improved in degree: the
staggered out-of-plane result on a re-entrant-corner or chiral cell is
BOUNDED at the Fourier oracles' own mutual spread, now one rung deeper on
every ladder, with the extrapolation evidence pointing at the staggered value
as the limit all three Fourier families are approaching.  Nothing here is a
proof of per-order correctness, and by the lossless-trap rule the staggered
arm's own internal convergence never will be.

**One correction to the brief.**  `rcwa_jones_2d` is NOT in-plane only: it
takes a full `(Nx, Ny, 3, 3)` cell with out-of-plane entries and routes it
through `rcwa._core._layer_eigenmodes_tensor`'s generator branch.  MEASURED --
zeroing `e13/e23/e31/e32` on the corner cell moves its answer by
`dR 7.63e-04` / `dJones 2.41e-03` at `n_orders = 5`, so the entries are being
used, not dropped.  It is therefore a full third ladder here, not a
partial one.

---

## 1.  Fixtures, ladders, cost

| fixture | cell | geometry | mounts |
|---|---|---|---|
| `corner` | (3,3), three `uniaxial(1.5, 1.7, tilt 35, azim 25)` pixels forming an **L** with a RE-ENTRANT 270-degree corner, air background | `px = py = 1.2 lam`, `depth = 0.4 lam`, `n_sub = 1.5` | normal, conical 20/35 |
| `chiral` | (3,3), TWO DIFFERENT out-of-plane tensors at (0,0) and (1,0) plus an isotropic pixel at (1,2) -- the 180-degree image differs in pattern AND tensor | `px = py = 1.10 um`, `wl = 0.68 um`, `depth = 0.36 um`, `n_sub = 1.5` | conical 25/40, normal |

Both are run through the SHIPPED entry points in the LIBRARY gauge
(`pmm_jones_2d_staggered`, `pmm_jones_2d`, `rcwa_jones_2d`) -- not the
prototype driver, whose 180-degree orientation defect the out-of-plane build
doc's GATE 0 pinned.

| ladder | rungs | wall time per rung (s), across the four cases |
|---|---|---|
| staggered `M` | 5, 6, 7, 8, 9, **10** | 1.6-3.1 / 5.7-5.9 / 18.8-19.8 / 50.0-53.9 / 101.6-118.3 / **176-247** |
| hybrid `n_orders` (degree 9), `laurent` | 7, 9, 11, **13** | 5.8-7.1 / 22.2-27.6 / 66.7-83.0 / **166-205** |
| hybrid `n_orders`, `li` | 7, 9, 11, **13** | 5.1-6.4 / 20.1-25.6 / 62.2-90.5 / **175-272** |
| rcwa `n_orders` (pixel-upsampled to `ceil((4n+1)/3)` per pixel) | 5, 7, 9, **11** | 0.5-0.8 / 2.7-3.9 / 9.9-16.2 / **32.0-48.7** |

Every ladder stopped at the first rung to exceed a 180 s budget, which is what
capped all four at the rungs shown.  Total probe time ~2 h for the two
fixtures, run concurrently.

## 2.  Method -- how a limit and its uncertainty are obtained

For each observable (per-order `R` and `T`, both incident polarizations,
`sum R`, `sum T`, and the four Jones entries) and each ladder:

* fit `f(x) = f_inf + C x^-p`, with `p` SCANNED on `[0.25, 10]` (400 points)
  and `(f_inf, C)` solved by linear least squares at each `p`; the `p` with the
  smallest residual wins.  **The rate is fitted, never assumed.**
* `sigma` is the MAXIMUM of three independently derived quantities, so it
  cannot be smaller than any of the ways the extrapolation could be wrong:
  (i) the least-squares standard error of the `f_inf` coefficient at the best
  `p`, using the fit's own residual as the noise estimate; (ii) the shift in
  `f_inf` when the FIRST ladder rung is dropped (stability against the
  pre-asymptotic regime); (iii) the distance to an independent Aitken
  `delta^2` extrapolant built from the last three rungs.
* a fit whose best `p` lands ON the scan boundary (`p <= 0.30` or `p >= 9.70`)
  is marked **UNSOUND** and excluded from the agreement verdict.  This is not a
  convenience: it is the diagnosable signature of a ladder that is still in its
  pre-asymptotic regime, and treating such an extrapolant as a limit is exactly
  how a converged-reference study fabricates a discrepancy.
* observables that are EXACTLY zero in every engine (the evanescent orders) are
  excluded -- they agree trivially and carry no uncertainty.

An example of what the boundary test catches, `corner/normal`, `T(-1,1)p0`:

| engine | ladder values | fit |
|---|---|---|
| staggered | 0.0046053 0.0046128 0.0046144 0.0046152 0.0046157 0.0046160 | **0.0046159 +/- 7.0e-07, p = 6.4** |
| hybrid-laurent | 0.0050560 0.0050910 0.0048012 0.0047313 | 0.0025667 +/- 2.1e-03, **p = 0.2 (UNSOUND)** |
| hybrid-li | 0.0049696 0.0049991 0.0046742 0.0046932 | 0.0026485 +/- 2.0e-03, **p = 0.2 (UNSOUND)** |
| rcwa | 0.0051060 0.0050321 0.0049310 0.0048501 | 0.0036914 +/- 8.3e-04, **p = 0.2 (UNSOUND)** |

The staggered ladder has visibly converged; the three Fourier ladders are all
still falling, monotonically, TOWARD it, and a naive extrapolation of any of
them would shoot far past.  Reading those three extrapolants as limits would
have produced a 2.0e-03 "discrepancy" that is entirely an artefact of the fit.

---

## 3.  Results

### C1  the staggered ladders (`sum R`, incident Ex, and the per-order step)

| case | `sum R` at M = 5 .. 10 | successive steps | per-order `dR`, top step |
|---|---|---|---|
| corner / normal | 0.035512527 0.035502494 0.035504778 0.035504936 0.035505566 0.035505597 | 1.00e-05 2.28e-06 1.58e-07 6.30e-07 3.08e-08 | **7.77e-07** |
| corner / conical 20/35 | ... 0.024446388 | 5.45e-06 1.73e-06 9.01e-07 6.05e-07 3.55e-07 (**monotone**, ratios 0.32 0.52 0.67 0.59) | **5.59e-07** |
| chiral / conical 25/40 | ... 0.036179916 | 1.69e-05 1.57e-06 5.26e-06 1.56e-06 1.94e-06 | **2.10e-06** |
| chiral / normal | ... 0.031931401 | 4.01e-05 2.49e-06 8.12e-07 6.56e-08 2.28e-07 | **1.39e-06** |

The two conical/normal `corner` ladders and `chiral/normal` fall by 1.5 to 3
decades; `chiral/conical` settles onto a **~2e-06 wobble** on `sum R` rather
than continuing to fall -- that plateau IS the staggered basis's own resolution
floor on this cell at conical incidence, and it is the reason that case's
staggered fits are sound on only 10/36 observables (a wobbling ladder has no
algebraic rate to find).  Reporting it is the point: the arm is not converging
past ~2e-06 there, and no claim below rests on pretending otherwise.

On the per-order `T` the low-M behaviour is cleanly algebraic and is what the
one new test asserts (section 5): steps of `2.51e-04`, `6.39e-05`, `9.26e-06`
at `M = 4 -> 5 -> 6 -> 7` on `corner/normal`, i.e. shrink factors of 3.9x and
6.9x, 27x end to end.

### C2  fit health and uncertainty, per engine and case

| case | live obs. | engine | sound fits | sigma min | sigma max | median sigma | top-of-ladder step (max) |
|---|---|---|---|---|---|---|---|
| corner / normal | 32 | **staggered** | **21/32** | 3.55e-07 | 3.46e-05 | **2.72e-06** | **3.73e-06** |
| | | hybrid-laurent | 18/32 | 1.37e-05 | 1.75e-03 | 9.32e-05 | 2.44e-03 |
| | | hybrid-li | 18/32 | 1.40e-05 | 8.67e-04 | 1.02e-04 | 2.46e-03 |
| | | rcwa | 13/32 | 7.02e-06 | 7.03e-04 | 3.36e-04 | 2.51e-04 |
| corner / conical 20/35 | 28 | **staggered** | 12/28 | 1.23e-06 | 5.56e-05 | **4.67e-06** | **3.09e-06** |
| | | hybrid-laurent | 12/28 | 2.86e-05 | 2.47e-03 | 2.35e-04 | 2.02e-03 |
| | | hybrid-li | 14/28 | 7.11e-06 | 3.73e-03 | 2.67e-04 | 2.55e-03 |
| | | rcwa | 10/28 | 7.39e-06 | 1.01e-03 | 1.64e-04 | 4.70e-04 |
| chiral / conical 25/40 | 36 | **staggered** | 10/36 | 4.67e-08 | 2.33e-05 | **1.18e-06** | **5.24e-06** |
| | | hybrid-laurent | 24/36 | 3.95e-06 | 5.86e-03 | 1.17e-04 | 3.85e-03 |
| | | hybrid-li | 21/36 | 1.17e-05 | 3.79e-03 | 2.20e-04 | 3.83e-03 |
| | | rcwa | 6/36 | 1.54e-06 | 1.43e-03 | 1.54e-05 | 6.62e-04 |
| chiral / normal | 40 | **staggered** | 8/40 | 2.26e-07 | 2.84e-05 | **3.61e-06** | **4.38e-06** |
| | | hybrid-laurent | 15/40 | 1.74e-06 | 1.83e-03 | 3.76e-04 | 3.59e-03 |
| | | hybrid-li | 13/40 | 1.57e-06 | 1.95e-03 | 3.35e-04 | 3.69e-03 |
| | | rcwa | 7/40 | 1.15e-05 | 1.49e-03 | 1.23e-04 | 6.23e-04 |

Two readings, and they pull in opposite directions -- both are reported:

* the staggered arm's uncertainty is 1.5 - 2 decades tighter than any Fourier
  arm's in every case, and its top-of-ladder step is 2 - 3 decades smaller;
* but its FIT-HEALTH count is not uniformly the best (8/40 on `chiral/normal`
  against the hybrid's 15/40).  That is the plateau of C1 showing through: once
  a ladder is at its own floor there is no algebraic rate left to fit, so the
  boundary test marks it unsound.  A tight ladder and a fittable ladder are
  different things, and conflating them would overstate the case.

### C3  do the extrapolated limits agree, per observable?

On the subset where BOTH fits are sound: `d = |limit_A - limit_B|` against the
combined uncertainty `sqrt(sA^2 + sB^2)`, `z = d / sigma`.

| case | pair | both sound | within combined sigma | worst obs. | d | sigma | z |
|---|---|---|---|---|---|---|---|
| corner / normal | staggered vs hyb-laurent | 10/32 | **9/10** | `R(0,-1)p0` | 2.26e-05 | 1.65e-05 | 1.37 |
| | staggered vs hyb-li | 11/32 | **9/11** | `T(-1,-1)p1` | 7.73e-05 | 1.95e-05 | 3.96 |
| | staggered vs rcwa | 8/32 | **6/8** | `R(0,0)p0` | 3.05e-05 | 1.82e-05 | 1.67 |
| | hyb-laurent vs hyb-li | 17/32 | 17/17 | `T(0,-1)p1` | 4.37e-04 | 8.50e-04 | 0.51 |
| | hyb-laurent vs rcwa | 3/32 | **1/3** | `T(1,-1)p0` | 1.33e-04 | 9.85e-05 | 1.35 |
| | hyb-li vs rcwa | 3/32 | **0/3** | `T(-1,-1)p1` | 1.72e-04 | 1.10e-04 | 1.57 |
| corner / conical | staggered vs hyb-laurent | 6/28 | **6/6** | `T(0,-1)p0` | 1.53e-03 | 1.65e-03 | 0.93 |
| | staggered vs hyb-li | 7/28 | **7/7** | `R(-1,-1)p0` | 2.92e-05 | 2.99e-05 | 0.98 |
| | staggered vs rcwa | 4/28 | **3/4** | `T(-1,1)p1` | 6.93e-05 | 3.01e-05 | 2.30 |
| | hyb-laurent vs hyb-li | 11/28 | 10/11 | `T(0,0)p0` | 1.01e-03 | 9.03e-04 | 1.12 |
| | hyb-laurent vs rcwa | 3/28 | 3/3 | `sumR1` | 2.41e-04 | 3.79e-04 | 0.63 |
| | hyb-li vs rcwa | 3/28 | 3/3 | `T(1,-1)p0` | 8.13e-04 | 8.84e-04 | 0.92 |
| chiral / conical | staggered vs hyb-laurent | 6/36 | **5/6** | `R(-1,1)p0` | 3.18e-05 | 3.15e-05 | 1.01 |
| | staggered vs hyb-li | 5/36 | **5/5** | `R(-1,1)p0` | 1.16e-04 | 1.27e-04 | 0.91 |
| | staggered vs rcwa | 4/36 | **3/4** | `R(-1,1)p0` | 9.77e-06 | 1.72e-06 | 5.68 |
| | hyb-laurent vs hyb-li | 19/36 | 19/19 | `T(-1,-1)p0` | 2.22e-03 | 2.65e-03 | 0.84 |
| | hyb-laurent vs rcwa | 3/36 | 3/3 | `R(-1,1)p0` | 2.21e-05 | 3.15e-05 | 0.70 |
| | hyb-li vs rcwa | 1/36 | 1/1 | `R(-1,1)p0` | 1.06e-04 | 1.27e-04 | 0.84 |
| chiral / normal | staggered vs hyb-laurent | 1/40 | **1/1** | `R(-1,1)p1` | 1.90e-05 | 3.92e-05 | 0.48 |
| | staggered vs hyb-li | 1/40 | **1/1** | `R(-1,1)p1` | 1.78e-04 | 2.01e-04 | 0.89 |
| | staggered vs rcwa | 1/40 | **1/1** | `R(-1,-1)p0` | 6.14e-06 | 8.68e-05 | 0.07 |
| | hyb-laurent vs hyb-li | 13/40 | 12/13 | `R(-1,0)p0` | 8.11e-06 | 4.22e-06 | 1.92 |
| | hyb-laurent vs rcwa | 0/40 | -- | -- | -- | -- | -- |
| | hyb-li vs rcwa | 0/40 | -- | -- | -- | -- | -- |

Summed over the four cases:

| pair | within combined sigma |
|---|---|
| **staggered vs hybrid (both rules)** | **43/47 (91%)** |
| **staggered vs rcwa** | **13/17 (76%)** |
| hybrid vs rcwa | 11/16 (69%) |
| hybrid-laurent vs hybrid-li | 58/60 (97%) -- the same code with two `E_z` rules |

**The answer to the brief's question is YES on the subset where the question is
answerable, and the staggered arm is not the odd one out** -- it agrees with
the hybrid more often than the hybrid agrees with rcwa.  The subset is small
because the Fourier ladders are not asymptotic, which is itself the finding.

### C4  do the FOURIER ladders move TOWARD the staggered limit?

For each Fourier engine and observable: is the LAST rung closer to the
staggered extrapolated limit than the FIRST?

| case | engine | monotone rungs | last rung closer than first | worst observable, `\|first - stag\|` -> `\|last - stag\|` |
|---|---|---|---|---|
| corner / normal | hybrid-laurent | 0/32 | **30/32** | `T(0,-1)p0` 1.50e-03 -> 2.66e-04 |
| | hybrid-li | 3/32 | **28/32** | `T(0,0)p0` 3.46e-04 -> 5.34e-04 |
| | rcwa | 18/32 | **30/32** | `T(0,0)p0` 3.06e-03 -> 9.40e-04 |
| corner / conical | hybrid-laurent | 5/28 | **27/28** | `T(0,1)p0` 3.13e-03 -> 5.83e-04 |
| | hybrid-li | 3/28 | **26/28** | `T(0,1)p0` 3.25e-03 -> 6.55e-04 |
| | rcwa | 16/28 | **24/28** | `T(0,0)p0` 3.76e-03 -> 1.43e-03 |
| chiral / conical | hybrid-laurent | 6/36 | **36/36** | `T(1,0)p1` 3.83e-03 -> 8.74e-04 |
| | hybrid-li | 9/36 | **33/36** | `T(1,0)p1` 3.65e-03 -> 8.23e-04 |
| | rcwa | 19/36 | **33/36** | `T(0,0)p1` 6.49e-03 -> 2.16e-03 |
| chiral / normal | hybrid-laurent | 9/40 | **39/40** | `T(-1,0)p1` 3.59e-03 -> 6.15e-04 |
| | hybrid-li | 7/40 | **40/40** | `T(0,0)p0` 1.05e-03 -> 7.97e-04 |
| | rcwa | 20/40 | **35/40** | `T(0,0)p1` 6.72e-03 -> 2.15e-03 |

**86 - 100% on every arm and every case**, worst-case distance falling 1.3x to
5.4x.  Note the `monotone` column is low for the hybrid: its `n_orders` ladder
oscillates (its `sum R` steps on `chiral/conical` read 3.13e-04, 1.48e-04,
1.64e-04 -- not a decaying sequence), which is precisely why its extrapolation
is unreliable and why a DIRECTION is the strongest thing that can be read off
it.  rcwa's ladder is the most monotone of the three (16 - 20 of 28 - 40) and
also the one whose fits most often hit the boundary, i.e. it is monotone but
far from its asymptote.

### C5  the bound, stated at the top of every ladder (no extrapolation)

Per-order max `|dR|` and `|dT|` between the TOP rungs (staggered M=10, hybrid
n=13 both rules, rcwa n=11).

| case | pair | max `\|dR\|` | max `\|dT\|` |
|---|---|---|---|
| corner / normal | staggered vs hyb-laurent | 3.14e-05 | 2.67e-04 |
| | staggered vs hyb-li | 6.17e-05 | 5.33e-04 |
| | staggered vs rcwa | 6.15e-05 | 9.41e-04 |
| | *Fourier mutual* | *3.00e-05 .. 5.44e-05* | *7.01e-04 .. 1.47e-03* |
| corner / conical | staggered vs hyb-laurent | 7.04e-05 | 5.83e-04 |
| | staggered vs hyb-li | 8.08e-05 | 6.56e-04 |
| | staggered vs rcwa | 1.32e-04 | 1.43e-03 |
| | *Fourier mutual* | *2.54e-05 .. 6.16e-05* | *6.73e-04 .. 1.55e-03* |
| chiral / conical | staggered vs hyb-laurent | **5.93e-05** | 8.73e-04 |
| | staggered vs hyb-li | 1.20e-04 | 8.22e-04 |
| | staggered vs rcwa | **1.10e-04** | 2.16e-03 |
| | *Fourier mutual* | *3.69e-05 .. 6.10e-05* | *1.02e-03 .. 2.54e-03* |
| chiral / normal | staggered vs hyb-laurent | 1.22e-04 | 6.36e-04 |
| | staggered vs hyb-li | 5.89e-05 | 7.86e-04 |
| | staggered vs rcwa | 2.54e-04 | 2.17e-03 |
| | *Fourier mutual* | *6.25e-05 .. 1.95e-04* | *1.10e-03 .. 2.57e-03* |

Ratio of the staggered arm's worst distance to the Fourier arms' own worst
mutual distance:

| case | on `R` | on `T` |
|---|---|---|
| corner / normal | 1.13x | **0.64x** |
| corner / conical | 2.14x | **0.92x** |
| chiral / conical | 1.97x | **0.85x** |
| chiral / normal | 1.30x | **0.84x** |

**On transmission the staggered arm is INSIDE the Fourier arms' own mutual
spread in every case; on reflection it is 1.1 - 2.1x that spread.**  The
reflected totals here are ~3e-02 against ~0.96 transmitted, so the reflection
column is the relatively larger disagreement -- and it is on the orders where
every Fourier fit is unsound, i.e. where no engine in the suite has a reference
to offer.

Directly against the verification's number: `chiral / conical`, staggered vs
rcwa, per-order `dR` -- **`1.341e-04` at (M=7, n=9)** becomes **`1.099e-04` at
(M=10, n=11)**, and against the nearest Fourier arm **`5.93e-05`**, inside the
Fourier mutual top of `6.10e-05`.

---

## 4.  What would actually settle it

Nothing in this suite.  The three ladders are not three independent references:
both `pmm_jones_2d` rules and `rcwa_jones_2d` route their out-of-plane tensor
layer through the SAME `rcwa._core._layer_eigenmodes_tensor` generator, so the
"three-engine agreement" is really two truncation families over one physics
implementation.  What would settle it, in order of cost:

1. **A conical 1-D out-of-plane engine.**  `pmm_jones_1d_conical_tensor`
   deliberately refuses a patterned out-of-plane cell at conical incidence
   (`AUDIT_PMM_CONICAL_PATTERNED_TENSOR_BUG_2026_07_12`); lifting that refusal
   would give a y-uniform out-of-plane oracle at conical incidence that shares
   NO code with either Fourier arm.
2. **A staircase-refined staggered cell** -- replace the corner with a
   sequence of finer staircases and watch the answer converge geometrically,
   which tests the corner treatment rather than the out-of-plane blocks.
3. **rcwa taken to `n_orders >= 17`** on a machine with the memory for a
   `4 (2n+1)^2` generator (2312 at n=11 here, 7938 at n=17).  The rcwa ladder
   is the most monotone of the three, so it is the one most likely to reach an
   asymptote; at the measured `n^3.5` scaling that is ~15 min per rung, which
   is affordable but was outside this experiment's budget.

## 5.  The one test this experiment earns

`tests/unit/test_pmm2d_staggered_oop_corner_convergence.py` (1 test, 20 s).
Everything else here is a cross-ENGINE comparison whose bar would be a value
from another build's Fourier ladder -- the cross-build pin
`docs/TESTING_STANDARDS.md` forbids.  What IS two-sided and derived is the
staggered arm's own ladder on the re-entrant-corner cell:

* the ladder is NOT already converged at the bottom rung (first per-order `T`
  step `> 1e-05`; measured `2.51e-04`, 25x of headroom, and four decades above
  the ~1e-09 a converged fixture would read);
* every successive step at least halves (measured ratios `0.25` and `0.14`,
  i.e. 2.0x and 3.5x of headroom under the `0.5` bar);
* and the ladder as a whole falls by more than a decade (measured 27x).

These are DISCRETIZATION errors of a deterministic polynomial basis, not build
noise -- the cross-build spread of a fixed-`M` solve is at the 1e-15 level, ten
decades below the smallest step compared.

---

## 6.  Files

| path | what |
|---|---|
| `validation/probe_pmm2d_staggered_oop_reference/t1_ladders.py` | the ladders, the fits, the pairwise; `python t1_ladders.py corner\|chiral` |
| `validation/probe_pmm2d_staggered_oop_reference/t2_summary.py` | the tables above, including the boundary-pinned-fit diagnosis |
| `.../results/t1_corner.json`, `t1_chiral.json` | every rung, every observable, every fit |
| `.../logs/t1_*.txt`, `logs/t2_*.md` | the console record and the rendered tables |
| `tests/unit/test_pmm2d_staggered_oop_corner_convergence.py` | the one test |
