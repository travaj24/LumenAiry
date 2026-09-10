# VERIFY -- the normal-incidence PARITY block reduction for the PURE staggered OUT-OF-PLANE eig, and its converged-reference companion

Date: 2026-09-10.  Independent adversarial verification of

* `docs/audits/BUILD_PMM2D_STAGGERED_OOP_BLOCK_EIG_2026_09_10.md` (the
  accelerator, commits `1f39746 .. c37bc7a`), and
* `docs/audits/EXPERIMENT_PMM2D_STAGGERED_OOP_REFERENCE_2026_09_10.md` (the
  converged-reference study, commit `e141a04`),

both merged into `wave2/pmm2d` at `9203843`.

**Where this was done.**  Worktree `C:/tmp/lum_vacc`, branch
`verify/oop-block-eig`, HEAD `9203843`.  The READ-ONLY comparison arm is the
main clone
`D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy` at
`fb3fd93` (5.43.0 plus one backend commit, i.e. WITHOUT the accelerator).
Every script asserts `lumenairy.__file__` for its own arm before it computes
anything (`vfix.assert_arm`), and the two-arm scripts were run with the
neutral-cwd trick so `PYTHONPATH` actually decides which copy is imported.

**Mount.**  Windows 11, python 3.14.6, numpy 2.4.4, scipy 1.17.1
(scipy-openblas), tesla-ryzen; `OMP_NUM_THREADS = OPENBLAS_NUM_THREADS =
MKL_NUM_THREADS = 1` exported before python started, every run.

**What "verify" meant here.**  Nothing below is read off the build's own
probes.  Every number is produced by a script in
`validation/probe_verify_oop_block_eig/` (README there; JSON in `results/`,
console record in `logs/`) using fixtures written from the physics -- a
different tensor zoo, a different mount (`px != py`, `wl = 0.633 um`), a
different cell-construction rule with an assertion that each fixture really is
(or is not) its own parity image -- and, where the build used an index trick,
a second independent route to the same quantity (an explicit dense `R`, a
re-assembly of the twelve pencil blocks from the public primitives, a
from-scratch re-implementation of the extrapolation).  Both directions are
exercised: claimed successes are attacked, and claimed failures (refusals,
fail-before) are proved out.

---

## 0.  Verdicts

| # | claim | verdict |
|---|---|---|
| 1 | default ON moves nothing outside its domain | **CONFIRMED**, and stronger than claimed: bit-identical on 16/16 fixtures ON vs OFF, and on 14/14 across the two BUILDS |
| 2 | ON vs OFF agree on parity-symmetric OOP cells; exact `2q^2/2q^2` split | **CONFIRMED** on 13 of my own (cell, tensor) combinations: dR 2.0e-14, dT 9.1e-14, dJones 4.9e-14, eig-set 2.3e-12, split 13/13 |
| 3 | the involution `R = diag(I,I,-I,-I) . blkdiag(P1,P2,P2,P1)` | **CONFIRMED** by re-derivation: `J^2 = I` EXACTLY (20/20), 36 EVEN blocks and 12 ODD blocks at 8e-17 .. 2.3e-16, and exactly **2 of 64** (permutation, sign) patterns work -- the shipped one and its own global negation |
| 4a | fail-before: disarming the gate is wrong by decades | **CONFIRMED** (dR 2.4e-04 .. 9.6e-02, eig-set 7.7e-02 .. 1.1e+01), with one caveat found: `tol = 1.0` is **not** a full disarm |
| 4b | the bar `_STAG_BLOCK_TOL = 1e-10` has "8.8 decades below the smallest real violation" | **BOUNDED / RESTATED** -- that gap is a property of the four violating fixtures chosen, not of the gate (an engineered 1e-7-relative parity break reads 2.5e-09, only 1.4 decades above the bar).  The bar is nevertheless **SAFE by ~10 decades more than claimed**, because the observable error is QUADRATIC in the residual: dR ~ 0.56 dA^2, so the worst error the gate can accept is ~6e-21 |
| 4c | "squares the spectrum": backward-error ratio 0.28 .. 7.20, test bar `1e2` | **CONFIRMED** (my 19 combinations: 0.435 .. 5.390); and the `1e2` bar IS two-sided -- a forced reduction reads **5.8e12 .. 2.0e13** |
| 5a | region solve 3.3 - 4.2x | **CONFIRMED** (3.16 .. 4.14x; my low end is 3.16x at (2,2) M=6) |
| 5b | whole solve 1.49 - 1.55x single layer, 1.80 - 1.87x on three layers | **CONFIRMED with a measurement-protocol caveat**: 1.40 .. 1.58x and 1.68 .. 1.90x in-process; in a COLD subprocess the single-layer figure falls to 1.12 .. 1.51x because ~0.7 s of fixed per-process warm-up dilutes it |
| 6a | the reference study's C5 bound, incl. 1.099e-04 / 5.93e-05 | **CONFIRMED EXACTLY** -- all 24 pair numbers, the four Fourier-mutual ranges and all eight ratios reproduce to the printed digits |
| 6b | sound-fit counts, and the pairwise 43/47, 13/17, 11/16 | **CONFIRMED EXACTLY** by an independent re-implementation of the fit |
| 6c | "the Fourier ladders move toward the staggered limit on 86 - 100%" | **CONFIRMED EXACTLY** (30/32, 28/32, 30/32 \| 27/28, 26/28, 24/28 \| 36/36, 33/36, 33/36 \| 39/40, 40/40, 35/40) |
| 6d | the UNSOUND-exclusion rule / "the staggered arm is not the odd one out" | **BOUNDED**: the headline survives with the unsound fits INCLUDED, but the RANKING sentence does not -- see S7 |
| 6e | `rcwa_jones_2d` genuinely uses the OOP entries (dR 7.63e-04) | **CONFIRMED EXACTLY** (7.627e-04 / dJones 2.406e-03) |
| 7 | durability of the two new test files | **SOUND** -- every constant has an origin and a measured two-sided gap; one sub-decade bar identified and restated; no exact-count assertion on nondeterministic machinery |

Three DOC defects are recorded (S8); none is a library defect.  **No library
defect was found.**

---

## 1.  Task 1 -- the default change moves nothing outside its domain

`validation/probe_verify_oop_block_eig/v1_default_on.py`
(`results/v1_tip.json`, `results/v1_main.json`).  Three arms, hashed with
sha256 over `(orders, R, T, Jones)` including dtype and shape:

* **A** = this tip, `symmetry='auto'` (the shipped default),
* **B** = this tip, `symmetry=False`,
* **C** = the main clone at `fb3fd93` (which has no `symmetry` keyword).

`A vs B` isolates the accelerator, `B vs C` isolates everything ELSE the merge
carries (the Wood-list unification, the magnetic weights), `A vs C` is the
composite a consumer sees.

| fixture class | cases | A = B | B = C | A = C |
|---|---|---|---|---|
| SCALAR cell, (2,2)/(3,3), normal + conical 25/40 | 3 | 3/3 | 3/3 | 3/3 |
| IN-PLANE tensor (e13 = e23 = e31 = e32 = 0 exactly), centro / off-centre / oblique 20 | 3 | 3/3 | 3/3 | 3/3 |
| OUT-OF-PLANE, off-centre pillar (2,2) + (3,3) | 2 | 2/2 | 2/2 | 2/2 |
| OUT-OF-PLANE, parity-BREAKING tensor (2,2) + (3,3) | 2 | 2/2 | 2/2 | 2/2 |
| OUT-OF-PLANE centro, OBLIQUE 25 deg and 5 deg | 2 | 2/2 | 2/2 | 2/2 |
| OUT-OF-PLANE centro, CONICAL 25/40 and 1/90 | 2 | 2/2 | 2/2 | 2/2 |
| MAGNETIC (scalar mu on an in-plane tensor; tensor mu on a scalar eps, conical) | 2 | 2/2 | n/a | n/a |
| **total** | **16** | **16/16** | **14/14** | **14/14** |

`dR = dT = dJones = 0.0` EXACTLY on every A-vs-B row -- these are sha256
identities, not tolerances.  The magnetic pair has no `fb3fd93` arm because
`mu_cell` is new on this branch.

Two things worth stating beyond the claim:

* **`B = C` on 14/14 as well.**  So the whole merged tip -- Wood-list
  unification and magnetic weights included -- is bit-identical to `fb3fd93` on
  every non-magnetic fixture here.  The default-ON change is therefore not
  merely "invisible net of the other changes"; nothing in the merge moves these
  answers at all.
* **The keyword has exactly one consumption site.**
  `grep -rn --include='*.py' "_region_modes_oop" lumenairy/` gives one call,
  `stack2d_pure.py:636`, and `self.symmetry` is read at exactly one place,
  `stack2d_pure.py:637`.  The in-plane pencil cannot see it.

### the structure residual on both sides of the bar (my fixtures)

`v2_on_off.py` section A (`results/v2_on_off.json`).  `dA = max|R A R + A| /
max|A|`, `dB = max|R B R - B| / max|B|`, computed with an EXPLICIT dense
`R = P diag(r)` rather than the library's row-blocked index arithmetic -- two
independent routes to the gate's own quantity.

| class | rows | `dA` | `dB` | `\|R^2 - I\|` | engages |
|---|---|---|---|---|---|
| 13 CARRYING cells x M 5,6,7,8 | 52 | **1.251e-16 .. 8.331e-15** | 5.931e-17 .. 2.373e-16 | **0.0 exactly** | 52/52 YES |
| off-centre pillar lc (2,2) / generic (3,3) | 2 | 7.309e-01 / 6.499e-01 | 2.1e-16 | 0.0 | no |
| parity-breaking lc/lc2 (2,2) / lc/nonrec (3,3) | 2 | 9.945e-02 / **2.395e-02** | 2.1e-16 | 0.0 | no |
| parity-breaking at 1e-3 relative (2,2) | 1 | 5.072e-05 | 2.1e-16 | 0.0 | no |
| **parity-breaking at 1e-7 relative (2,2)** | 1 | **2.535e-09** | 2.1e-16 | 0.0 | no |
| centro, OBLIQUE 25 / CONICAL 25/40 / OBLIQUE 1e-6 rad | 3 | *gauge refuses* | -- | -- | no |

The carrying set is deliberately harder than the build's: a generic dense
tensor with every out-of-plane entry populated and no symmetry at all, a
non-reciprocal Hermitian tensor, a strongly-lossy metal (`n = 0.6 + 4.2i`),
(3,3) cells with an INTERIOR feature on the parity-fixed centre pixel, and
4-pixel orbits carrying two DIFFERENT tensors.  All 52 engage; `1e-10` sits
**4.1 decades** above the worst.

The `dB` column never moves off 2e-16 -- `B` is the eps-free block Gram, so it
is parity-invariant on any uniform wall grid, exactly as the build states.  It
is not a discriminating quantity; `dA` is.

**The refutation on the other side.**  The build's claim that the bar sits
"8.8 decades below the smallest real violation" is true of ITS four violating
fixtures and of the two extra classes I added, but it is a fixture property,
not a gate property: an engineered 1e-7-relative parity break (one tensor entry
of one mirror pixel) reads `dA = 2.535e-09`, i.e. **1.4** decades above the bar,
and the family is continuous.  What actually makes the bar safe is measured in
S4.

---

## 2.  Task 2 -- ON vs OFF on parity-symmetric OOP cells at normal incidence

`v2_on_off.py` section B.  Thirteen (cell, tensor) combinations built here, not
copied; engagement is asserted BEFORE any agreement number is read, so no row
can pass by quietly refusing.  Through the shipped entry point, both incident
polarizations, per order; `M = 7` on (2,2) and `M = 6` on (3,3).

| cell | grid | M | `4q^2` | dR | dT | dJones | d(eig set) | split | `min\|q\|/max\|q\|` |
|---|---|---|---|---|---|---|---|---|---|
| uniform tilted uniaxial | (2,2) | 7 | 576 | 8.81e-16 | 4.77e-15 | 2.31e-15 | 9.27e-13 | ok | 3.24e-02 |
| uniform negative uniaxial | (3,3) | 6 | 900 | 1.74e-15 | 5.24e-14 | 6.29e-15 | 4.99e-13 | ok | 2.27e-02 |
| centro pair lc | (2,2) | 7 | 576 | 1.36e-15 | 8.99e-15 | 7.58e-15 | 2.34e-12 | ok | 1.17e-02 |
| centro pair GENERIC dense tensor | (2,2) | 7 | 576 | 3.61e-15 | 2.09e-14 | 1.38e-14 | 1.14e-12 | ok | 1.92e-02 |
| centro pair NON-RECIPROCAL | (2,2) | 7 | 576 | 1.81e-15 | 1.33e-14 | 8.52e-15 | 1.81e-12 | ok | 1.03e-02 |
| centro pair LOSSY | (2,2) | 7 | 576 | 9.51e-16 | 1.82e-14 | 8.64e-15 | 1.09e-12 | ok | 2.12e-02 |
| centro pair strongly-lossy METAL | (2,2) | 7 | 576 | 5.51e-15 | 1.14e-14 | 2.05e-14 | 1.09e-12 | ok | 3.52e-02 |
| (3,3) ring lc + INTERIOR lossy centre | (3,3) | 6 | 900 | 8.40e-16 | 1.52e-14 | 7.98e-15 | 6.34e-13 | ok | 2.07e-02 |
| (3,3) ring generic + INTERIOR non-reciprocal centre | (3,3) | 6 | 900 | 4.20e-15 | 9.08e-14 | 1.58e-14 | 5.74e-13 | ok | 1.82e-02 |
| (3,3) ring metal + INTERIOR lc2 centre | (3,3) | 6 | 900 | **2.03e-14** | 8.15e-14 | **4.90e-14** | 9.72e-13 | ok | 2.21e-02 |
| (3,3) cross orbit lc / nonrec | (3,3) | 6 | 900 | 1.65e-15 | 3.64e-14 | 1.28e-14 | 8.85e-13 | ok | 1.54e-02 |
| (2,2) cross orbit lossy / generic | (2,2) | 7 | 576 | 2.13e-15 | 3.05e-14 | 6.72e-15 | 1.27e-12 | ok | 1.92e-02 |
| (3,3) pair lc2 + centre metal | (3,3) | 6 | 900 | 5.38e-15 | 6.93e-14 | 1.83e-14 | 1.24e-12 | ok | 2.04e-02 |
| **worst over the 13** | | | | **2.03e-14** | **9.08e-14** | **4.90e-14** | **2.34e-12** | **13/13** | 1.03e-02 .. 3.52e-02 |

* the eigenvalue set is compared by SYMMETRIC HAUSDORFF distance, not by
  sorting -- a plain sort pairs a near-degenerate doublet wrongly and
  manufactures a mismatch; this is the build's own choice and it is the right
  one;
* the forward/backward split is **exactly `2 q^2 / 2 q^2` on BOTH arms in all
  13 rows** (and the counts are also equal arm to arm);
* `min|q|/max|q|` never approaches `_STAG_GAM_FLOOR = 1e-13` -- 11 decades of
  margin -- so the reconstruction's `1/q` is never near its guard;
* no row is bit-identical (0/13), which is the expected reading: these are two
  different algorithms on the same pencil, not the same code twice.

The build's envelope (dR 4.9e-15, dT 1.6e-13, dJones 2.6e-14, eig-set 6.9e-12)
and mine (2.0e-14, 9.1e-14, 4.9e-14, 2.3e-12) agree in kind and in order; my
dR and dJones are ~4x and ~2x larger because the metal-tensor fixtures are
harder than any in the build's set.  **The test bar of `1e-11` still clears my
worst by 2.0 decades.**

---

## 3.  Task 3 -- the involution, re-derived

`v3_involution.py` (`results/v3_involution.json`).

### 3.1  the 1-D maps (20 combinations, `Nx` 2..5, `M` 4..8)

| quantity | envelope | the WRONG reading |
|---|---|---|
| `\|P^2 - I\|`, both sets | **0.0 EXACTLY on 20/20** | -- |
| `\|P^T Mtt P - Mtt\| / max`, and the `B` twin | 5.25e-17 .. 1.89e-16 | 2.00 exactly |
| `\|Pb^T <B\|d\|Btilde> Pt + <.>\| / max` (parity-ODD) | 6.59e-17 .. 1.39e-16 | 2.00 exactly |

`J^2 = I` is exact, not small, because a signed permutation composed with
itself squares its signs -- and the measurement says `0.0`, on every
combination, not `~1e-16`.

### 3.2  the 2-D blocks, re-assembled here

The twelve pencil blocks of `_assemble_oop` were rebuilt in the probe from the
same public primitives (`Basis1D.mass`, `.mixed`, `_eps_weighted`,
`_OOP_ROT_SIGN`) rather than read off the solver, then hit with the four space
parities `P1 = kron(Pt_y, Pb_x)`, `P2 = kron(Pb_y, Pt_x)`,
`P3 = kron(Pt_y, Pt_x)`, `Pw = kron(Pb_y, Pb_x)` in the module's `kron(y, x)`
order.  Three cells (a generic dense tensor, a (3,3) with an interior feature,
a non-reciprocal), `M` 5/6/7.

| class | blocks | expected | measured `\|P M P - s M\| / max\|M\|` | the OPPOSITE reading |
|---|---|---|---|---|
| the three block Grams `Ggram1, Ggram2, Gw` + the NINE eps-weighted masses `A11 .. A33` | 36 rows | EVEN | **8.11e-17 .. 2.27e-16** | 2.00 exactly |
| the FOUR single-derivative blocks `P13, P23, CwE1, CwE2` | 12 rows | ODD | **1.10e-16 .. 2.20e-16** | 2.00 exactly |

### 3.3  the two eliminations carry the DERIVED signs

Pushing `e1 -> P1 e1`, `e2 -> P2 e2`, `g1 -> -P2 g1`, `g2 -> -P1 g2` through
the two elimination solves (rebuilt here) must give `E3 -> +P3 E3` and
`G3 -> -Pw G3`.  Measured on the elimination MATRICES themselves, before the
pencil is assembled:

| cell | `E3 -> +P3 E3` | (the `-P3` reading) | `G3 -> -Pw G3` | (the `+Pw` reading) |
|---|---|---|---|---|
| centro pair generic (2,2), M=6 | **7.06e-14** | 2.00 | **1.37e-13** | 2.00 |
| (3,3) ring lc + interior lossy, M=5 | **3.47e-14** | 2.00 | **2.69e-14** | 2.00 |

These carry the `A33^-1` and `Gw^-1` solves, which is why they sit at 1e-14
rather than 1e-16; they are still twelve decades from the opposite reading.

### 3.4  every WRONG pattern fails, by decades

Four block-permutation patterns x all 16 block sign vectors = **64**
combinations, on `(3,3) ring lc + interior lossy`, `M = 5`:

| pattern | passes `dA <= 1e-10` | smallest FAILING `dA` |
|---|---|---|
| **SHIPPED** `blkdiag(P1, P2, P2, P1)` | **2 / 16** -- `(+,+,-,-)` at `dA = 2.329e-15`, and `(-,-,+,+)`, which is the same `R` negated (`R A R` is quadratic in `R`) | 2.420e-01 |
| `blkdiag(P1, P2, P1, P2)` | 0 / 16 | 1.079e+00 |
| `blkdiag(P1, P1, P2, P2)` | 0 / 16 | 1.079e+00 |
| `blkdiag(P3, P3, P3, P3)` | 0 / 16 | 7.660e-01 |

**Exactly the shipped sign pattern works, and nothing else does.**  The
smallest failing residual over all 62 failures is `2.420e-01` -- **9.4 decades**
above the bar.

Corrupting the 1-D map instead of the block signs fails equally hard (same
cell, shipped block signs):

| corruption of `_stag_parity_1d` | `dA` | `dB` | `\|R^2 - I\|` |
|---|---|---|---|
| *shipped* | 2.329e-15 | 8.11e-17 | 0.0 |
| bubbles carry NO sign (`(-1)^a -> +1`) | 3.309e-01 | 2.778e-01 | 0.0 |
| bubble sign FLIPPED (`(-1)^(a+1)`) | 9.888e-01 | 8.333e-01 | 0.0 |
| the `B`-set half-hats do NOT swap | 4.186e-01 | 2.778e-01 | 0.0 |
| identity map (no parity at all) | 6.954e-02 | 0.0 | 0.0 |

Note the last row: the identity map leaves `dB = 0` (the Gram is symmetric
under it) and still fails on `dA` at 6.95e-02 -- a gate that only tested `B`
would accept it.  The shipped gate tests both and refuses.

### 3.5  the sector dimensions

`tr R = 0` is not an accident of a square grid: blocks 1 and 4 carry the SAME
permutation `P1` with signs `+1` and `-1`, and blocks 2 and 3 carry `P2` the
same way, so the fixed-point traces cancel in pairs identically.  With
`R^2 = I` that forces `dim(U+) = dim(U-) = 2 q^2` exactly, and
`_stag_block_eig` re-checks the two sector counts anyway.  Measured exact on
all 52 carrying rows of S1 (`len(plus) == len(minus) == 2 q^2`).

---

## 4.  Task 4 -- fail-before, the bar's own safety, and the backward error

`v4_failbefore.py` (`results/v4_failbefore.json`).

### 4.1  FAIL-BEFORE

The runtime verification is disarmed and the SHIPPED entry point is run on
cells that do not carry the structure.

| cell | M | `dA` | d(eig set) | dR | dT | dJones | shipped == dense |
|---|---|---|---|---|---|---|---|
| off-centre pillar lc (2,2) | 6 | 7.309e-01 | 2.884e-01 | 1.127e-02 | 1.916e-01 | 5.782e-02 | YES |
| off-centre pillar generic (3,3) | 6 | 6.499e-01 | 2.130e-01 | 7.122e-03 | 1.359e-01 | 5.153e-02 | YES |
| parity-breaking lc/lc2 (2,2) | 6 | 9.945e-02 | 8.270e-02 | 1.672e-03 | 1.043e-02 | 1.421e-02 | YES |
| parity-breaking lc/nonrec (3,3) | 6 | 2.395e-02 | 7.732e-02 | 2.387e-04 | 3.832e-03 | 8.737e-04 | YES |
| off-centre METAL (2,2) | 7 | **1.033e+00** | 1.124e+01 | 9.609e-02 | 3.397e-01 | 2.370e-01 | YES |
| *accepted agreement, for contrast (S2)* | | *<= 8.3e-15* | *<= 2.3e-12* | *<= 2.0e-14* | *<= 9.1e-14* | *<= 4.9e-14* | |

**Ten to twelve decades** between the forced answer and the accepted one, and
the shipped answer is the dense one on 5/5.  CONFIRMED.

**Caveat found: `tol = 1.0` is not a full disarm.**  `dA` is bounded by 2, not
by 1, and the off-centre METAL cell reads `dA = 1.033`, so it is STILL REFUSED
when `_STAG_BLOCK_TOL` is monkeypatched to `1.0` -- the value
`test_forcing_the_reduction_without_the_structure_is_wrong_by_decades` uses.
`np.inf` is the actual disarm (used for the table above).  This is not a defect
in the shipped test: its two fixtures read 6.95e-01 and 2.89e-01, and if a
future fixture crossed 1.0 the test would FAIL LOUDLY (`forced == ref`, so the
`> 1e-4` assertion fails) rather than pass vacuously.  It is worth a one-word
change (`1.0 -> np.inf`) the next time the file is touched.

### 4.2  THE BAR'S OWN SAFETY -- the question the build doc does not ask

The violation is made CONTINUOUS: the mirror pixel's `e13`/`e31` are scaled by
`(1 + d)`, sweeping `dA` from 1.3e-15 to 5.3e-03 straight through
`_STAG_BLOCK_TOL`.  At each rung the gate's DECISION is recorded, and the
reduction is separately FORCED so the error it *would* have produced is visible
on both sides.

| `d` | `dA` | gate | dR (forced) | dT (forced) | dJones (forced) |
|---|---|---|---|---|---|
| 0 | 1.341e-15 | **accept** | 1.599e-15 | 4.663e-15 | 5.973e-15 |
| 1e-13 | 4.951e-15 | accept | 2.741e-16 | 4.996e-15 | 2.078e-15 |
| 1e-12 | 5.047e-14 | accept | 7.737e-16 | 4.552e-15 | 4.283e-15 |
| 1e-11 | 5.069e-13 | accept | 6.696e-16 | 4.136e-15 | 4.139e-15 |
| 1e-10 | 5.070e-12 | accept | 1.506e-15 | 1.343e-14 | 6.845e-15 |
| 1e-09 | **5.070e-11** | **accept** (the last one) | 1.422e-15 | 6.106e-15 | 7.153e-15 |
| 3e-09 | **1.521e-10** | **refuse** (the first one) | 8.223e-16 | 1.443e-14 | 3.091e-15 |
| 1e-08 | 5.070e-10 | refuse | 2.248e-15 | 1.399e-14 | 8.295e-15 |
| 1e-07 | 5.070e-09 | refuse | 2.359e-16 | 1.210e-14 | 2.424e-15 |
| 1e-06 | 5.070e-08 | refuse | 1.135e-15 | 1.521e-14 | 3.913e-15 |
| 1e-05 | 5.070e-07 | refuse | 1.437e-13 | 3.260e-13 | 5.368e-13 |
| 1e-04 | 5.070e-06 | refuse | 1.428e-11 | 3.285e-11 | 5.363e-11 |
| 1e-03 | 5.072e-05 | refuse | 1.428e-09 | 3.284e-09 | 5.362e-09 |
| 1e-02 | 5.095e-04 | refuse | 1.426e-07 | 3.275e-07 | 5.346e-07 |
| 1e-01 | 5.323e-03 | refuse | 1.411e-05 | 3.176e-05 | 5.184e-05 |

Three readings:

1. **The crossing is exactly at the bar**: the largest accepted `dA` is
   5.070e-11, the smallest refused is 1.521e-10.
2. **Everything the gate accepts is exact.**  The worst observable error over
   the six accepted rungs is `dR = 1.599e-15` -- indistinguishable from the
   exact-symmetry rungs, four decades under the tests' `1e-11` bar.  Every
   refused rung is bit-identical to `symmetry=False`.
3. **The error law is QUADRATIC in the residual, which is what actually makes
   `1e-10` safe.**  Over the clean part of the sweep
   (`dA = 5.1e-07 .. 5.3e-03`) the ratio `dR / dA^2` is constant at
   **0.50 .. 0.56**:

   | `dA` | 5.070e-07 | 5.070e-06 | 5.072e-05 | 5.095e-04 | 5.323e-03 |
   |---|---|---|---|---|---|
   | `dR` | 1.437e-13 | 1.428e-11 | 1.428e-09 | 1.426e-07 | 1.411e-05 |
   | `dR / dA^2` | 0.559 | 0.556 | 0.555 | 0.549 | 0.498 |

   Extrapolating that law to the bar gives a worst accepted error of
   `0.56 * (1e-10)^2 ~ 6e-21` -- **ten decades better than the linear reading**
   a "decades of gap" argument would suggest.  So the build's headline gap
   claim is fixture-dependent (S1) but its CONCLUSION is safer than it argues,
   for a reason it did not measure.

### 4.3  the "squares the spectrum" caveat

`v4_failbefore.py` section C, 19 combinations (7 cells, `M` 5/6/7), normwise
eigenpair backward error `|A x - q B x| / ((|A| + |q| |B|) |x|)` on the same
pencil.

| quantity | this verification | the build's claim |
|---|---|---|
| dense `zgeev` | **4.453e-16 .. 4.608e-15** | 1.13e-15 .. 5.27e-15 |
| factored | **3.768e-16 .. 1.680e-14** | 4.49e-16 .. 2.11e-14 |
| RATIO factored / dense | **0.435 .. 5.390** | 0.28 .. 7.20 |
| forward counts equal | **19/19** | equal in every row |

CONFIRMED: the reduction is not uniformly better than dense (unlike the Fourier
twin), it is within a decade, and it never misclassifies a mode.  The worst
ratios are on the (3,3) rows (4.78 and 5.39 at `M = 6`), consistent with the
`Xh Yh` squaring argument.

**The `1e2` bar IS two-sided**, which the build did not measure: forcing the
reduction onto a structure-violating cell gives a backward-error ratio of
**5.85e12 .. 2.03e13** (`v7_test_durability.py`).  So the bar has 1.27 decades
above the worst legitimate ratio and **10.8 decades** below the smallest real
violation.

---

## 5.  Task 5 -- speed

`v5_speed.py` (`results/v5_speed.json`), plus `results/v5b_inproc.json` and
`results/v5c_multilayer.json`.

### 5.1  region solve (in-process, min over 3 timed calls after a warm-up)

| grid | M | `4q^2` | OFF [s] | ON [s] | ratio |
|---|---|---|---|---|---|
| (2,2) | 6 | 400 | 0.218 | 0.069 | **3.16x** |
| (2,2) | 7 | 576 | 0.605 | 0.175 | **3.46x** |
| (2,2) | 8 | 784 | 1.263 | 0.373 | **3.39x** |
| (3,3) | 6 | 900 | 1.817 | 0.566 | **3.21x** |
| (3,3) | 7 | 1296 | 5.130 | 1.339 | **3.83x** |
| (3,3) | 8 | 1764 | 12.146 | 2.931 | **4.14x** |

**3.16 .. 4.14x** against the claimed **3.3 - 4.2x**.  CONFIRMED; the low end
is 3.16x rather than 3.3x on the smallest grid.

### 5.2  whole solve

| instrument | single layer | three DISTINCT layers |
|---|---|---|
| in-process, min of 2, after warm-up | **1.40 .. 1.58x** (6 configs) | **1.68 .. 1.90x** (4 configs) |
| separate COLD subprocess, alternating, min of 4 rounds | 1.12 .. 1.51x | 1.53 .. 1.80x |
| build doc | 1.49 - 1.55x | 1.80 - 1.87x |

The two configurations the build timed on three layers give **1.87x** ((2,2)
M=7) and **1.90x** ((3,3) M=7) here, against its 1.87x and 1.80x.

The cold-subprocess row is the honest caveat.  My whole-solve subprocess timer
starts after the imports but still pays ~0.7 s of first-call BLAS / allocator
warm-up: `(2,2) M=6` dense reads **0.441 s** in-process and **1.196 s** in a
fresh subprocess.  That overhead is pure serial time, so it dilutes the ratio
(at `(2,2) M=6` the measured 1.12x is exactly what Amdahl predicts from the
diluted region share).  Where the build's own B4b table gives a region SHARE of
42 - 52% and a whole-solve `0.366 s` at `(2,2) M=6`, my in-process numbers give
a share of **41.7 - 53.7%** and `0.441 s` -- i.e. the build's subprocess timer
was effectively warm.  With that protocol matched, the claim reproduces:

| grid | M | whole OFF [s] | whole ON [s] | measured | region share | Amdahl prediction |
|---|---|---|---|---|---|---|
| (2,2) | 6 | 0.441 | 0.285 | 1.55x | 50.7% | 1.52x |
| (2,2) | 7 | 1.192 | 0.753 | 1.58x | 53.7% | 1.58x |
| (2,2) | 8 | 2.566 | 1.746 | 1.47x | 49.7% | 1.55x |
| (3,3) | 6 | 4.020 | 2.878 | 1.40x | 45.3% | 1.44x |
| (3,3) | 7 | 11.959 | 7.999 | 1.50x | 43.8% | 1.47x |
| (3,3) | 8 | 28.120 | 18.758 | 1.50x | 41.7% | 1.46x |

The Amdahl column is computed from MY OWN share and ratio, and tracks the
measurement to within 0.08x -- so the build's mechanism (the region solve is
~half a single-layer solve; a multi-layer stack pays it once per distinct
layer) is confirmed, not just its numbers.

---

## 6.  Task 6 -- the converged-reference study, independently re-fitted

`v6_refit.py`, `v6b_refit_detail.py`, `v6c_rcwa_oop.py`.  The ladder DATA is
the committed
`validation/probe_pmm2d_staggered_oop_reference/results/t1_{corner,chiral}.json`;
the EXTRAPOLATION is re-implemented here from the method description in the
study's S2, and a SECOND, wholly different extrapolant (a pure Aitken/Shanks
transform with no model and no fitted rate) is run alongside it.

### 6.1  the committed ladders are reproducible

`v6c_rcwa_oop.py` section 1.  The corner fixture was rebuilt from the study's
prose description (S1), and the rcwa arm's pixel-upsampling rule
(`ceil((4n+1)/3)` per pixel) re-derived from the same sentence.

| rung | wall time | observables matched | worst `\|mine - committed\|` |
|---|---|---|---|
| staggered `M = 7`, corner/normal | 11.2 s | 48/48 | **4.441e-16** |
| rcwa `n_orders = 5` (upsample x7) | 0.5 s | 48/48 | **0.000e+00** |

So the data being re-fitted is not an artefact of the study's driver.

### 6.2  C5 -- the top-of-ladder bound (no extrapolation): EXACT

Every number in the study's C5 table reproduces to the printed digits.

| case | staggered vs hyb-laurent | vs hyb-li | vs rcwa | *Fourier mutual* | ratio R | ratio T |
|---|---|---|---|---|---|---|
| corner / normal | dR 3.144e-05 | 6.167e-05 | 6.146e-05 | *3.002e-05 .. 5.435e-05* | **1.13x** | **0.64x** |
| corner / conical | 7.039e-05 | 8.079e-05 | 1.319e-04 | *2.538e-05 .. 6.155e-05* | **2.14x** | **0.92x** |
| chiral / conical | **5.927e-05** | 1.203e-04 | **1.099e-04** | *3.691e-05 .. 6.101e-05* | **1.97x** | **0.85x** |
| chiral / normal | 1.215e-04 | 5.893e-05 | 2.540e-04 | *6.254e-05 .. 1.951e-04* | **1.30x** | **0.84x** |

and the headline tightening, restated from the same data:

| case | staggered M=7 vs rcwa n=9 | staggered M=10 vs rcwa n=11 |
|---|---|---|
| corner / normal | 7.5794e-05 | 6.1458e-05 |
| corner / conical | 1.6280e-04 | 1.3194e-04 |
| **chiral / conical** | **1.3410e-04** | **1.0986e-04** |
| chiral / normal | 3.4211e-04 | 2.5403e-04 |

`1.341e-04 -> 1.099e-04`, and `5.93e-05` against the nearest Fourier arm,
inside the Fourier mutual top of `6.10e-05`.  **CONFIRMED EXACTLY**, including
the "inside on T in all four cases, 1.1 - 2.1x on R" statement.

### 6.3  the fits: sound-fit counts and the pairwise verdict

Re-implementing the study's method as described (`f = f_inf + C x^-p`, `p`
scanned on `[0.25, 10]` with **400** points as stated, `(f_inf, C)` by linear
least squares, `sigma = max(LS standard error, drop-first-rung shift, distance
to an Aitken extrapolant)`, `p` on `[0.30, 9.70]` or UNSOUND):

| case | staggered | hybrid-laurent | hybrid-li | rcwa |
|---|---|---|---|---|
| corner / normal (32) | **21/32** | **18/32** | **18/32** | **13/32** |
| corner / conical (28) | **12/28** | **12/28** | **14/28** | **10/28** |
| chiral / conical (36) | **10/36** | **24/36** | **21/36** | **6/36** |
| chiral / normal (40) | **8/40** | **15/40** | **13/40** | **7/40** |

Every one of these sixteen counts matches the study's C2 table exactly, and so
do the four staggered `sigma` envelopes and medians (corner/normal
3.55e-07 .. 3.46e-05, median 2.72e-06; corner/conical 1.23e-06 .. 5.56e-05,
median 4.67e-06; chiral/conical 4.67e-08 .. 2.33e-05, median 1.18e-06;
chiral/normal 2.26e-07 .. 2.84e-05, median 3.61e-06) -- so the `sigma`
definition ("the MAXIMUM of three independently derived quantities") was
reconstructed correctly from the prose, not guessed.  Two notes:

* the study's **live-observable counts (32/28/36/40) exclude the eight Jones
  components**, although its S2 says the observable set includes "the four
  Jones entries"; its own stated exclusion rule ("exactly zero in every
  engine") applied to the committed data gives 40/36/44/48.  This is a
  DOCUMENTATION mismatch, not a numerical one -- the tables are internally
  consistent on the no-Jones set, and adding the Jones components does not
  change any verdict (S6.5).
* the `chiral/conical` staggered count is scan-resolution-sensitive by one:
  **400** scan points give 10/36 (the study's value), 1201 or 4001 give 11/36.
  Worth knowing if that number is ever quoted as a bar.

And the C3 pairwise table -- all 24 rows and all six totals:

| pair | this verification | the study |
|---|---|---|
| staggered vs hybrid (both rules) | **43/47 (91.5%)** | 43/47 |
| staggered vs rcwa | **13/17 (76.5%)** | 13/17 |
| hybrid vs rcwa | **11/16 (68.8%)** | 11/16 |
| hybrid-laurent vs hybrid-li | **58/60 (96.7%)** | 58/60 |

**CONFIRMED EXACTLY.**  The per-case rows (10/32 -> 9/10, 11/32 -> 9/11,
8/32 -> 6/8, 17/32 -> 17/17, 3/32 -> 1/3, 3/32 -> 0/3, and the other three
cases likewise) all match.

The study's S2 example table also reproduces exactly, including the fitted
rates:

| engine | corner/normal, `T(-1,1)p0` ladder | fit |
|---|---|---|
| staggered | 0.0046053 0.0046128 0.0046144 0.0046152 0.0046157 0.0046160 | 0.0046159 +/- 6.9e-07, p = 6.4 |
| hybrid-laurent | 0.0050560 0.0050910 0.0048012 0.0047313 | 0.0025667 +/- 2.1e-03, p = 0.2 UNSOUND |
| hybrid-li | 0.0049696 0.0049991 0.0046742 0.0046932 | 0.0026485 +/- 2.0e-03, p = 0.2 UNSOUND |
| rcwa | 0.0051060 0.0050321 0.0049310 0.0048501 | 0.0036914 +/- 8.3e-04, p = 0.2 UNSOUND |

as do the C1 staggered `sum R` ladders and their step sequences on all four
cases, and the C4 note (hybrid-laurent `sum R` steps on chiral/conical:
3.13e-04, 1.48e-04, 1.64e-04).

### 6.4  C4 -- do the Fourier ladders move TOWARD the staggered limit?

| case | engine | last rung closer than first | monotone (ladder VALUES) |
|---|---|---|---|
| corner / normal | hyb-laurent / hyb-li / rcwa | **30/32 / 28/32 / 30/32** | 0/32 / 3/32 / 18/32 |
| corner / conical | | **27/28 / 26/28 / 24/28** | 5/28 / 3/28 / 16/28 |
| chiral / conical | | **36/36 / 33/36 / 33/36** | 6/36 / 9/36 / 19/36 |
| chiral / normal | | **39/40 / 40/40 / 35/40** | 9/40 / 7/40 / 20/40 |

**86 - 100% -- CONFIRMED EXACTLY**, on all twelve rows, and the `monotone`
column reproduces exactly once it is read as "the ladder VALUES are monotone"
(rather than "the steps decay").  The claim survives the independent Aitken
extrapolant unchanged (the staggered limit moves by less than the distances
being compared).

### 6.5  the UNSOUND-exclusion rule -- the cherry-picking check

The study excludes a fit whose best `p` lands on the scan boundary.  That is
methodologically right (a boundary-pinned `f_inf` is not a limit), but it does
select the subset the verdict is read on, so the verdict is repeated here with
those fits INCLUDED (their huge `sigma` then makes "within combined sigma"
easy, which is why including them inflates every count):

| pair | sound-only (the study's rule) | unsound INCLUDED | pure Aitken (no model) |
|---|---|---|---|
| staggered vs hybrid-laurent | 21/23 (91.3%) | **124/136 (91.2%)** | 105/136 (77.2%) |
| staggered vs hybrid-li | 22/24 (91.7%) | **123/136 (90.4%)** | 97/136 (71.3%) |
| staggered vs rcwa | 13/17 (76.5%) | **105/136 (77.2%)** | 77/136 (56.6%) |
| hybrid-laurent vs hybrid-li | 58/60 (96.7%) | **134/136 (98.5%)** | 111/136 (81.6%) |
| hybrid-laurent vs rcwa | 7/9 (77.8%) | **123/136 (90.4%)** | 98/136 (72.1%) |
| hybrid-li vs rcwa | 4/7 (57.1%) | **127/136 (93.4%)** | 101/136 (74.3%) |

**What survives, and what does not.**

* The study's HEADLINE -- "the bound tightens; it does not become a
  confirmation, and no discrepancy survives measurement", and "the staggered
  arm is not the odd one out" -- **SURVIVES both robustness checks**: with the
  unsound fits included, staggered-vs-hybrid sits at 91.2 / 90.4%, squarely
  inside the band the Fourier arms occupy among themselves (90.4 .. 98.5%).
* The RANKING sentence in S0 item 3 -- "**it agrees with the hybrid more often
  than the hybrid agrees with rcwa**" -- is **exclusion-dependent and does not
  survive**.  With the unsound fits included, hybrid-li vs rcwa reads 93.4%,
  ABOVE staggered vs hybrid-laurent (91.2%); under the pure-Aitken extrapolant
  the two bands overlap (77.2 / 71.3% against 72.1 / 74.3%).  The ordering is a
  property of the sound subset, not a robust fact.
* Under BOTH robustness checks the **staggered-vs-rcwa pair is the weakest of
  the six** (77.2% with unsound included, 56.6% under Aitken).  That is the
  same pair the study's own C5 identifies as the largest top-of-ladder
  distance, and the same one the tightened bound is quoted on -- so the
  finding is consistent with, not contrary to, the study's conclusion that this
  remains a BOUND.

Verdict on this claim: **BOUNDED.**  The exclusion is not cherry-picking (the
verdict's direction is unchanged with the excluded fits restored, and the
excluded fits are independently diagnosable), but one sentence of the summary
over-reads the subset.

### 6.6  `rcwa_jones_2d` really does consume the out-of-plane entries

`v6c_rcwa_oop.py` section 2 -- zero `e13/e23/e31/e32` on the corner cell and
compare:

| engine | knock-out `dR` | `dT` | `dJones` | bit-identical |
|---|---|---|---|---|
| `rcwa_jones_2d`, `n_orders = 5` (upsample x7) | **7.627e-04** | 1.257e-02 | **2.406e-03** | no |
| `rcwa_jones_2d`, `n_orders = 7` (upsample x10) | 7.462e-04 | 1.236e-02 | 2.417e-03 | no |
| `pmm_jones_2d_staggered`, `M = 6` | 7.400e-04 | 1.247e-02 | 2.420e-03 | -- |

The claimed `dR 7.63e-04` / `dJones 2.41e-03` are **CONFIRMED to the printed
digits**, and the staggered arm's own knock-out agrees on the SIZE of the
out-of-plane contribution to within 3% -- a small extra cross-check the study
did not take.

---

## 7.  Task 7 -- durability of the two new test files

Every constant re-measured on the tests' own fixtures
(`v7_test_durability.py`, `results/v7_test_durability.json`).

| test / bar | quantity | MEASURED here | gap ABOVE | the smallest REAL signal | gap BELOW |
|---|---|---|---|---|---|
| `test_parity_map...` `np.array_equal(P@P, I)` | `\|P^2 - I\|` | **0.0 exactly**, 12/12 | -- (exact) | -- | -- |
| ... masses `<= 1e-12 * max\|M\|` | `\|P^T M P - M\|/max` | 8.02e-17 .. 1.80e-16 | 3.7 dec | a real parity failure reads 2.4e-02 .. 1.0e+00 | 10 dec |
| ... derivative `<= 1e-12 * max\|C\|` | `\|Pb^T C Pt + C\|/max` | 6.59e-17 .. 1.39e-16 | 3.9 dec | the EVEN reading is 2.00 exactly | 12 dec |
| ... wrong sign `> 1e-3 * max\|C\|` | `\|Pb^T C Pt - C\|/max` | 2.00 exactly | 3.3 dec | -- | -- |
| `test_reduction_engages...` `struct_dA <= 1e-10` | `max\|R A R + A\|/max\|A\|` | 2.31e-15 .. 4.28e-15 | 4.4 dec | file's violating fixtures 6.17e-02 .. 6.95e-01 | 8.8 dec (**but only 1.4 dec on an ENGINEERED 1e-7 break -- see S1/S4.2**) |
| ... sector counts `== 2 q^2`, `sum(r[fixed]) == 0.0` | signed-permutation arithmetic | exact | -- | -- | not build-dependent (integer counting, no eig) |
| `test_reduction_matches...` `<= 1e-11` | `max\|ON - OFF\|` | dR 2.32e-15, dT 4.25e-14, dJones 1.20e-14 | 2.4 dec | forced 2.4e-04 .. 3.4e-01 | 7.4 dec |
| `test_..._eigenvalue_set...` `<= 1e-9` | symmetric Hausdorff | 1.94e-12 | 2.7 dec | forced 7.7e-02 .. 1.1e+01 | 7.9 dec |
| ... counts `== 2 q^2` EXACT | forward/backward split | exact | -- | -- | **not an S5 shape**: `_region_modes_oop` RAISES if the split is not `2q^2/2q^2`, so the assertion restates an invariant the library enforces, and `min\|q\|/max\|q\|` is 1.0e-02 .. 3.5e-02, eleven decades off the guard |
| `test_factored_eigenpair...` `bf <= 1e2 * bd` | backward-error ratio | 1.22 .. 3.56 (file's params); 0.435 .. 5.390 (19 of mine) | 1.27 dec | **forced: 5.85e12 .. 2.03e13** | **10.8 dec** |
| ... `bd <= 1e3 * 4 q^2 * eps` | dense backward error | 1.47e-15 .. 4.62e-15 vs its own 8.9e-11 .. 2.0e-10 | 4.6 dec | -- | sanity bar, one-sided by nature |
| `test_structure_bar_has_a_gap...` tol ladder | the shipped gate at `tol/1e2`, `tol*1e3`, `1.0` | `struct_dA(ok) = 4.28e-15 < 1e-10 < struct_dA(bad) = 2.89e-01` | -- | -- | the `tol=1.0` arm needs `struct_dA(bad) < 1`; my off-centre METAL fixture reads 1.033.  **Fails loudly, never vacuously** |
| `test_forcing...` `> 1e-4` | forced `max(dR,dT,dJ)` | 1.71e-02 .. 2.22e-01 | -- | accepted agreement <= 4.9e-14 | 2.2 dec below the measurement, 7 dec above the accepted agreement |
| corner `steps[0] > 1e-5` | `max\|T(M=5) - T(M=4)\|` | **2.512e-04** (25x) | 1.40 dec | a converged fixture reads ~1e-09 | 4 dec |
| corner `steps[i+1] <= 0.5 * steps[i]` | successive step ratios | **0.254 and 0.145** | **0.29 dec -- SUB-DECADE** | the quantity's own cross-build spread is ~1e-10 RELATIVE | ~10 dec |
| corner `steps[-1] <= 0.1 * steps[0]` | end-to-end fall | **0.0368** (27x) | 0.43 dec | same | ~10 dec |

**The one sub-decade bar, restated (test-only).**
`steps[i+1] <= 0.5 * steps[i]` has 1.97x of headroom at its tightest rung
(0.254 against 0.5), i.e. 0.29 decades.  It is nevertheless NOT a per-build
bar: the quantity is a ratio of discretization steps of a deterministic
polynomial basis, whose cross-build spread is ~1e-15 ABSOLUTE against steps of
1e-5 .. 1e-4, i.e. ~1e-10 relative -- about ten decades below the headroom.
What the bar really encodes is a convergence-RATE assumption about this one
fixture ("the corner ladder's error at least halves per rung between M = 4 and
M = 7"), and that is a physics statement, not a numerical one.  It should be
read that way, and if a future basis change slows the corner's convergence the
bar firing is the gate working, not a flake.  The same applies to (c), which
has 2.71x of headroom.  The two are also not independent: (c) follows from (b)
by two applications, so (c) adds margin, not information.

**Everything else checks out:** every numeric constant carries a stated origin
and a date, every comparison is two-arm and same-build (nothing pins a value
from another build), the exact-equality assertions are all on genuinely exact
quantities (a signed permutation squared, a sha256, an integer count enforced
by a library raise), and there is no `pytest.skip` on any resource or
capability check.  The three "MEASURED" envelopes quoted in the docstrings
(B0, B1, B2, B5b, B6) all re-measure inside the values my own fixtures give.

### 7.1  one test added

The shipped suite tests the gate's bar by moving the BAR past fixtures whose
residual is 6e-02 .. 7e-01 -- 8.8 decades away.  Nothing tested the gate where
it actually sits.  One test was added to
`tests/unit/test_pmm2d_staggered_oop_block_eig.py`:

`test_the_bar_is_walked_across_and_everything_it_accepts_is_still_exact` walks
a family of cells whose two mirror pixels differ by a relative `d` across the
bar (`dA` from ~1e-13 to ~1e-07), asserts that the family really SPANS the bar
(a factor-2 guard band on each side, asserted rather than hoped), that the
gate's decision follows its own predicate outside that band, that **every cell
the gate ACCEPTS still reproduces the dense path to `1e-11`**, and that a
refused one is bit-identical.  Every quantity is measured at runtime; nothing
is pinned.

### 7.2  re-timed

| file | tests | duration (this mount, `-p no:randomly`) | build doc |
|---|---|---|---|
| both files, AS SHIPPED (36 + 1) | 37 | **79.99 s** | 101 - 117 s + 20 s |
| both files, after the addition (37 + 1) | 38 | **84.51 s** | -- |
| `test_pmm2d_staggered_oop_corner_convergence.py` alone | 1 | **13.7 - 14.1 s** (two runs) | 20 s |
| the added test alone | 1 | **1.93 s** | -- |

Slowest single test 14.1 s -- well under the 40 s per-test rule; each file is
under the 3 min rule.  `.test_durations` was spliced with the one new entry
(1.93 s), leaving the file's existing `indent=2`, key-sorted formatting
untouched (`git diff --stat` shows `+1` line).

---

## 7.3  Task 8 -- the suites

Command, every run:
`PYTHONPATH=/c/tmp/lum_vacc OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q -p no:randomly <files>`

| files | tests | result | duration |
|---|---|---|---|
| `test_pmm2d_staggered_oop_block_eig.py` + `test_pmm2d_staggered_oop_corner_convergence.py` (with the added test) | **38** | **passed** | 84.5 s |
| `test_pmm2d_staggered_oop.py` (35) + `test_pmm2d_staggered_anisotropic.py` (41) + `test_pmm2d_staggered_magnetic.py` (32) + `test_pmm2d_staggered_wood_list.py` (18) | **126** | **passed** | 237.8 s |
| **total** | **164** | **164 passed, 0 failed** | ~5.4 min |

`ruff check lumenairy/ tests/` -- **clean**.  `ruff check
validation/probe_verify_oop_block_eig/` -- clean.

---

## 8.  Defects found

**No library defect.**  Three documentation defects and two test-hygiene notes:

| # | where | defect | severity |
|---|---|---|---|
| D1 | `EXPERIMENT_..._REFERENCE` S0 item 1 | the hybrid-laurent sound-fit list is given as "13/32, 12/28, 24/36, 15/40"; the first entry is **18/32** (its own C2 table says 18/32, and my independent re-fit says 18/32).  `13/32` is rcwa's corner/normal value. | cosmetic, but it is a quoted count |
| D2 | `EXPERIMENT_..._REFERENCE` S0 item 4 / C4 | "with the worst-case distance falling by **1.3x to 5.4x** over the ladder".  Under the document's OWN selection rule (the observable with the largest LAST distance) the twelve rows read **0.65x .. 5.84x**: one row -- corner/normal, hybrid-li, `T(0,0)p0` 3.463e-04 -> 5.336e-04, which the C4 table itself prints -- moves AWAY by 1.54x, and the largest fall is 5.84x (chiral/normal, hybrid-laurent), not 5.4x.  The summary silently drops the rising row and rounds the top of the range down. | the direction claim (86 - 100% of observables move closer) is unaffected and CONFIRMED; only the "worst-case falls" sentence is wrong |
| D3 | `EXPERIMENT_..._REFERENCE` S2 / C2 | the observable set is described as including "the four Jones entries", but the live counts (32/28/36/40) exclude all eight Jones components; the rule as stated gives 40/36/44/48. | documentation only -- no verdict changes when they are included |
| D4 | `test_..._block_eig.py::test_forcing_...` | `_STAG_BLOCK_TOL` is monkeypatched to `1.0`, which is **not** a full disarm (`dA` is bounded by 2, and a cell with `dA > 1` -- e.g. an off-centre strongly-lossy metal pillar, 1.033 -- is still refused).  The test's two fixtures are under 1, and the failure mode is loud, so nothing is wrong today. | suggest `np.inf` next time the file is touched |
| D5 | `BUILD_..._BLOCK_EIG` S0 gate (2) / `_STAG_BLOCK_TOL` docstring | "8.8 decades below the smallest real violation" reads as a property of the bar; it is a property of the four violating fixtures.  A continuous violation family crosses the bar smoothly, and an engineered 1e-7-relative parity break sits 1.4 decades above it. | the CONCLUSION is safe -- safer than argued, by the quadratic law of S4.2 -- but the phrasing over-claims.  The new test of S7.1 closes the gap |

---

## 9.  What could NOT be verified

* **Per-order correctness of the staggered out-of-plane arm itself.**  Unchanged
  from the build's own statement and the study's S4: the three Fourier ladders
  are not three independent references (both `pmm_jones_2d` rules and
  `rcwa_jones_2d` route their out-of-plane tensor layer through the same
  `rcwa._core._layer_eigenmodes_tensor`), so this remains a BOUND.  Nothing in
  this verification changes that, and by the lossless-trap rule the staggered
  arm's own internal convergence never will.
* **Off-normal incidence.**  Provably 1.00x and provably refused; verified as a
  refusal (bit-identity at oblique 25 deg / 5 deg and conical 25/40 / 1/90,
  including a 1e-6 rad polar angle where the gauge still refuses), not as a
  capability.
* **Cross-BUILD durability.**  Everything here is one mount (py3.14.6 / numpy
  2.4.4 / scipy-openblas / tesla-ryzen).  The durability argument in S7 is that
  each bar's headroom exceeds the quantity's own cross-build spread by decades,
  which is an argument, not a second-BLAS measurement.  The two-arm structure of
  every test (same build, two algorithms) is what makes that argument safe.
* **The `chiral` fixture's exact tensors** were not reconstructed from prose
  (the study gives them only descriptively); the chiral re-fit therefore uses
  the committed ladder values, whose reproducibility was established only on the
  `corner` fixture (S6.1).
* **The study's ladder rungs above what is committed** (rcwa at `n >= 13`,
  staggered at `M >= 11`) were outside this verification's budget, as they were
  outside the study's.

---

## 10.  Files and commits

| path | what |
|---|---|
| `validation/probe_verify_oop_block_eig/README.md` | how to re-run every arm |
| `.../vfix.py` | fixtures, hashing, metrics; asserts which library copy is imported |
| `.../v1_default_on.py` | the three-arm bit-identity matrix (tip auto / tip False / fb3fd93) |
| `.../v2_on_off.py` | structure residuals with an explicit dense `R`; ON vs OFF on 13 cells |
| `.../v3_involution.py` | the 1-D maps, the twelve re-assembled blocks, the two eliminations, 64 wrong patterns, 4 map corruptions |
| `.../v4_failbefore.py` | fail-before, the continuous bar sweep, the backward-error table |
| `.../v5_speed.py` | interleaved subprocess timings + in-process region timings |
| `.../v6_refit.py`, `v6b_refit_detail.py` | the independent re-fit (power law + pure Aitken), C3/C4/C5, the unsound-included check |
| `.../v6c_rcwa_oop.py` | ladder-rung reproduction and the out-of-plane knock-out |
| `.../v7_test_durability.py` | every test constant re-measured on the tests' own fixtures |
| `.../results/*.json`, `.../logs/*.txt` | every number above |
| `tests/unit/test_pmm2d_staggered_oop_block_eig.py` | +1 test (S7.1) |
| `.test_durations` | +1 entry |
