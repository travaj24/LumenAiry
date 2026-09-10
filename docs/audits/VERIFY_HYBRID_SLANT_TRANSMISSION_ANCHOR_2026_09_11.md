# VERIFICATION -- the HYBRID 2-D PMM frame-anchor fix, and the JAX slant refusal

**Date:** 2026-09-11 - **Worktree:** `C:/tmp/lum_vhyb`, branch
`verify/hybrid-slant-anchor` off `wave2/pmm2d` @ `39bd54e`.
**Under verification:** `docs/audits/FIX_HYBRID_SLANT_TRANSMISSION_ANCHOR_2026_09_11.md`
-- commits `0ebd632` (the per-order frame anchor on the transmitted
amplitudes) and `5dba5ee` (the JAX refusal), merged at `0002d76`.
**Probes:** `validation/probe_verify_hybrid_anchor/` (own README, sixteen
scripts, every run's JSON on both arms and both builds).
**Binding law:** `docs/TESTING_STANDARDS.md`.
**Method:** every number below was RE-MEASURED here on fixtures written from
the geometry, at a different period, wavelength, cell, walk fraction and
staircase ladder than the fix's own probes; nothing was taken from the fix's
JSONs.

---

## VERDICT UP FRONT

| # | claim | verdict |
|---|---|---|
| 1 | the WITHOUT arm is bit-identical except the transmission accessors on a slanted PATTERNED layer | **CONFIRMED** -- 186/200 hashes identical over 29 of my own fixtures, 14 moved, **0 unexpected**; plus 15 further surfaces in `q1b` |
| 1a | the zeroth-order Jones is EXEMPT at normal incidence (`P_0 = 1`) | **CONFIRMED and REFINED** -- the exemption is WALK-INDEPENDENT; the brief's "quarter-walk non-exemption" is not a property of `jones_transmission`, it is the `+/-` conjugate non-degeneracy of the PER-ORDER arm |
| 2 | `A_lab(m) = exp(+i k0 alpha_m . W)`, `W = sum slant_j d_j` over the layers that enter a frame | **CONFIRMED** on three references and three mounts; the three wrong arms fail by 49x .. 104x |
| 2a | the shipped arm converges toward a refined oracle while the un-anchored one stands still | **CONFIRMED** -- shipped 2.38x / 1.53x / 1.78x over K = 5 -> 25; un-anchored 0.998x / 1.000x / 1.002x |
| 3 | the walks ADD; a film above or below contributes nothing; no round-trip phase on reflection | **CONFIRMED** -- every row, both mounts |
| 3a | the layer-split identity is sha-exact and BLIND to the sum | **CONFIRMED** -- `0.000e+00`, and both arms move together under a wrong sum, to the digit |
| 3b | SCOPE: a PATTERNED layer below a slanted one rides the shear (O1) | **CONFIRMED** -- `+walk` wins by 7.1x .. 7.9x on transmission AND 3.7x .. 6.9x on the anchor-free REFLECTION |
| 4 | every traced scalar returned the VERTICAL answer silently on 5.44.0, and now refuses | **CONFIRMED** -- SEVEN routes, all silent on 5.44.0 (`dR 3.46e-03`, `dT 6.89e-02`, `dJones 1.54e-02`, no warning), all refused on the fix; both controls solve, identically on both arms |
| 5 | O2 -- the slanted-over-film cascade blow-up | **BOUNDED, and its ATTRIBUTION REFUTED** -- reproduced at 6.34e+30 (worse than the 2.6e+27 reported, and on a SINGLE layer, which O2 says was fine); the cause is `cond(T22) >= 6e+14` in the generalized interface solve, controlled by the SHEAR DISCRETIZATION and not by the `eps = 1.0` coincidence or a near-cut-off order -- both refuted by measurement.  LOUD (96/96 warned).  Severity **P2**.  See S6 |
| 6 | the 27-test file's bars are durable | **CONFIRMED with four restatements** -- cross-build spread `<= 7.0e-08` relative over 49 re-measured quantities |
| O1 | should the hybrid adopt the pure engine's refusal? | **RECOMMEND NO** -- the numbers and the three reasons are in S10 |

**Three defects found here, none of them closed by this fix.**

| | severity | what |
|---|---|---|
| **V1** | **P1, silent-wrong, OPEN** | `pmm_jones_2d(..., slant=...)` with ANY traced input returns the **VERTICAL** answer, silently -- the exact shape commit `5dba5ee` closed one function away.  Measured: bit-close to the NumPy vertical solve (`dR 8.4e-15`), wrong against the NumPy slanted answer by `dR 3.85e-03 / dT 5.29e-02 / dJones 1.66e-02`.  Reads IDENTICALLY on 5.44.0 and on the fix branch. |
| **V2** | **P1, silent-wrong, OPEN** | The **1-D** engine has the SAME frame-anchor defect: `PMMStack.jones_transmission()` on an `add_sheared_grating` layer (`factorization='convection'`, the only 1-D path that retains amplitudes on a slanted stack) is FRAME-referenced.  Against the 1-D engine's own z-staircase the as-returned Jones is flat at `9.28e-01 / 9.27e-01 / 9.28e-01` over `n_slices` 6/12/24 while `x P_0` converges `8.2e-03 -> 2.1e-03 -> 1.3e-03`, below the staircase's own last step. |
| **V3** | doc, minor | The fix audit's S3.1 / `test_d2` line "either round-trip factor is 55x .. 83x above it" has the wrong antecedent: 56x .. 83x is the ratio to the AS-RETURNED reflection; against the oracle's own step the round-trip arms are 15.8x .. 23.4x, which is what the `10 step_J` bar actually tests.  Restated in the test file. |

---

## 0. ARMS

| | WIN | WSL | WITHOUT |
|---|---|---|---|
| tree | `C:/tmp/lum_vhyb` | same via `/mnt/c` | `C:/tmp/lum_v5440` (tag `v5.44.0` = `50824e9`), READ-ONLY |
| interpreter | CPython 3.14.6 | CPython 3.12.3 | as the build using it |
| numpy / scipy | 2.4.4 / 1.17.1 | 2.4.6 / 1.17.1 | as above |
| jax | 0.11.0, x64 | 0.10.2, x64 | as the build using it |
| threads | `OMP` = `OPENBLAS` = `MKL` = 1 | `OMP` = `OPENBLAS` = 1 | as above |

`lumenairy.__file__` is asserted on every arm by `_lib.arm()`, which decides
`fix` / `v5440` from the path alone and stamps it, with the interpreter,
numpy, scipy and the thread caps, into every JSON.

**The `v5.44.0` arm is legitimate for THIS fix, checked rather than assumed.**
`v5.44.0` resolves to `50824e9` (main's tip, which already carries a
`wave2/pmm2d` merge), and `git diff v5.44.0 9b36ded -- lumenairy/` -- `9b36ded`
being the commit the fix branched from -- touches only `elements/_lens_traced.py`
and `propagators/carrier.py`.  **The whole `lumenairy/elements/pmm/` tree is
byte-identical between `v5.44.0` and the pre-fix branch point**, so a hash that
moves between the two arms is attributable to this fix and to nothing else in
`pmm/`.

The branch tip does carry four other agents' `pmm/` work (`stack2d_pure.py`,
`twod_staggered.py`, `pmm/_core.py`, `pmm/stack.py`).  Of those the hybrid's
import graph reaches exactly one changed symbol -- `stack.py`'s
`_warn_stack_energy`, which gained a `stack=None` keyword whose
`_sliver_refusal(None, ...)` returns immediately; `pmm/_core.py`'s changes are
in `_pmm_union_grid` and a new mortar interface, neither of which `stack2d.py`
imports.  The 186 identical hashes are the empirical confirmation.

---

## 1. TASK 1 -- the WITHOUT-arm bit-identity, on my own fixtures

`q1_fixtures.py` builds **29 fixtures** and hashes seven keys on each
(`orders`, `R`, `T`, `jones_reflection`, `jones_transmission`,
`per_order_amplitudes('transmission')`, `per_order_amplitudes('reflection')`)
plus a four-key `solve_vs_wavelength` sweep -- **200 sha256 hashes**.  The set
covers vertical scalar at three mounts, a vertical in-plane tensor, a vertical
OUT-OF-PLANE tensor (the 4N generator with no slant), a three-layer stack, a
uniform-film stack, the even-parity fold at `symmetry='auto'` AND
`symmetry=False`, one and many tapered pillars, the `fused` / `tree` /
`monolithic` cascades, slanted UNIFORM x-only and diagonal, a CONSTANT-tile
slanted cell, a vertical pattern over a slanted film, a lossy cell, and eight
slanted PATTERNED rows (scalar at three mounts at a QUARTER walk, a HALF walk
at normal, an in-plane tensor, two DIFFERENT slants, one over a reflecting
film, one with a `y` slant component).

| comparison | identical | moved | UNEXPECTED |
|---|---|---|---|
| `v5.44.0` vs the fix, WIN | **186** | **14** | **0** |
| `v5.44.0` vs the fix, WSL | **186** | **14** | **0** |

The two builds' hash VALUES differ throughout, as two BLAS builds must; the
identity claim is same-build, and the SET of fixtures and keys that moves is
the same on both -- including both exemptions of S1.1.

Every one of the 14 is a transmission accessor on a fixture whose stack holds a
slanted PATTERNED (non-constant-tile) layer:

| fixture | key |
|---|---|
| `slanted_patterned_quarter_normal` | `per_order_transmission` |
| `slanted_patterned_quarter_oblique25` | `jones_transmission`, `per_order_transmission` |
| `slanted_patterned_quarter_conical25_40` | `jones_transmission`, `per_order_transmission` |
| `slanted_patterned_half_normal` | `per_order_transmission` |
| `slanted_tensor_inplane_oblique25` | `jones_transmission`, `per_order_transmission` |
| `two_slanted_layers_conical` | `jones_transmission`, `per_order_transmission` |
| `slanted_over_film_oblique25` | `jones_transmission`, `per_order_transmission` |
| `slanted_patterned_xy_conical` | `jones_transmission`, `per_order_transmission` |

`R`, `T`, `orders`, `jones_reflection` and `per_order_amplitudes('reflection')`
are byte-identical on **every one of the 29 fixtures**, slanted rows included.

### 1.1 The `P_0 = 1` exemption -- CONFIRMED, and the brief's reading REFINED

Two rows in the table above are conspicuous by their absence:
`slanted_patterned_quarter_normal.jones_transmission` and
`slanted_patterned_half_normal.jones_transmission` are **both identical**
across the arms.

At normal incidence `kx0 = ky0 = 0`, so the zeroth order has `alpha_0 = 0` and
`P_0 = exp(i k0 * 0 * W) = 1` **for any walk whatsoever**.  The exemption is a
property of the MOUNT, not of the walk fraction.  The brief's parenthetical
("...and NOT `jones_transmission` at normal incidence for a HALF-period walk --
reproduce the `P_0 = 1` exemption and the QUARTER-walk non-exemption") reads as
if a quarter walk would un-exempt `jones_transmission`; measured, it does not.
What a quarter walk does un-degenerate is a different thing, and it is
reproduced in S2.2: at normal incidence with a HALF walk `P_m = exp(i pi m)` is
REAL, so the `x P_m` and `x conj(P_m)` arms coincide identically; at a QUARTER
walk they separate (`8.439e-01` against `9.803e-01`).  The fix audit's own
statement of the exemption ("at normal incidence the ZEROTH order alone is
exempt: `P_0 = 1` when `kx0 = ky0 = 0`") is exactly right; only the brief's
restatement of it conflates the two degeneracies.

`per_order_transmission` moves at normal incidence at BOTH walks, because every
`m != 0` order carries a real `alpha_m` and a non-trivial phase.

### 1.2 The two NO-OP shapes, as a CROSS-FIXTURE identity

Anchoring a slanted UNIFORM film, or a slanted patterned cell whose tile is
constant, would corrupt an answer that is currently exact.  Measured as a
same-build identity against the plain vertical film of the same `eps` and
thickness, **all seven keys, on BOTH arms**:

| fixture vs `vertical_film_2p20_oblique25` | identical keys | differing |
|---|---|---|
| `slanted_uniform_x_oblique25` (`slant = (0.7, 0)`) | **7 / 7** | none |
| `slanted_consttile_oblique25` (`slant = (0.7, 0)`) | **7 / 7** | none |

and `q3`'s two-sided arm prices the alternative: putting either shape into the
walk sum costs `5.870e-01` (oblique) / `5.331e-01` (conical) on the per-order
transmitted amplitudes against `0.000e+00` as shipped.

### 1.3 The surfaces the hash table does not reach (`q1b_extra.py`)

| surface | fix vs `v5.44.0` |
|---|---|
| `internal_field(z)` on a VERTICAL 3-layer stack -- `Ex Ey Ez Hx Hy Hz x y z` | **9 / 9 identical** |
| `layer_absorption()` on the same | identical |
| `cascade_stats()` | identical |
| `jones_field_from_orders` bridge, VERTICAL stack, both ports (4 arrays) | identical |
| `jones_field_from_orders` bridge, SLANTED stack, REFLECTION port (2 arrays) | identical |
| `jones_field_from_orders` bridge, SLANTED stack, TRANSMISSION port (2 arrays) | **MOVED** -- correct, it consumes the per-order dict |

**MAGNETIC stacks are NOT APPLICABLE to the hybrid**, checked rather than
assumed: `PMM2DStackHybrid.add_layer`'s parameters are
`(thickness, eps, eps_cell, eps_tensor_cell, region_layout, slant)` -- there is
no `mu` surface at all (the PURE engine has `mu` / `mu_cell`; the hybrid does
not).  The brief's "magnetic stacks" row cannot be built on this class.

---

## 2. TASK 2 -- THE ANCHOR, derived here and measured against three references

### 2.1 The formula, from the geometry

A slanted PATTERNED region is solved in `u = x - t_x z`, `v = y - t_y z`,
`w = z`.  In that frame the structure is `w`-invariant, so the state the
cascade carries is the FRAME Fourier coefficient: the field is
`sum_m F_m(w) exp(i k0 alpha_m . (u, v))`.  Two consequences:

1. the frame is anchored at the layer's TOP face, where `z = 0` gives
   `(u, v) = (x, y)` -- frame and lab coincide, so the SUPERSTRATE side (`R`,
   `T`, the reflection Jones, the reflected per-order amplitudes) needs
   nothing;
2. at the layer's BOTTOM, `w = d`, the same plane is `z = d` with
   `(u, v) = (x - t_x d, y - t_y d)`, so against the substrate's own lab basis
   `exp(i k0 alpha_m . (x, y))`

   ```
       A_lab(m) = exp(+i k0 alpha_m . t d) A_frame(m)
   ```

   in the PUBLIC (`exp(-iwt)`) convention, and the frames of successive sheared
   layers simply continue downward, so

   ```
       A_lab(m) = exp(+i k0 alpha_m . W) A_frame(m),
       W = sum over the layers that ENTER A FRAME of slant_j * d_j.
   ```

`alpha_m` is real for every order, propagating or evanescent, so the factor is
UNIMODULAR and no efficiency can move -- which is the structural reason the
omission was silent.  **Measured** (`q2_anchor.py`, conical 25-40,
`n_orders = 7`, a QUARTER walk): over **225 orders of which 210 are
EVANESCENT**, `max ||P_m| - 1| = 1.110e-16` and `max |arg P_m| = 2.403 rad`.

The sign is not a matter of taste and is settled below by measurement: the
conjugate arm is WORSE than applying nothing at all, in every row, on every
reference.

### 2.2 (a) Against the engine's OWN fine staircase

Fixture (mine, not the fix's): `px = py = 0.90 um`, `wl = 0.62 um`,
`n_sup = 1.0`, `n_sub = 1.5`, a 6 x 4 x-ASYMMETRIC y-varying cell
(`eps` 1.30 .. 3.05), `d = 0.45 um`, `slant = (0.5, 0)` -- `t d = px/4`, a
**QUARTER-period walk**, so `P_m = exp(i arg P_0) i^m` is four-valued and
neither the whole-period nor the half-period degeneracy can hide anything.
Oracle: the SAME engine's z-staircase of the SAME solid at `K = 5 / 15 / 25`,
MIDPOINT rule, on 600-column cells -- `_lib.staircase_upsample_factor` picks
the smallest upsample for which every rung's roll is an EXACT integer number of
pixels and `staircase_cells` refuses anything else.  `n_orders = 5`; the
residual is the relative Frobenius norm over the 121 shared orders and both
incident polarizations.

Per-order TRANSMITTED amplitudes, `K = 25`:

| mount | shipped | no anchor | conjugate | best single GLOBAL phase |
|---|---|---|---|---|
| normal (quarter walk) | **9.422e-03** | 8.4394e-01 (**89.6x**) | 9.8027e-01 (**104.0x**) | 8.2502e-01 (**87.6x**) |
| oblique 25 | **1.9200e-02** | 1.1029e+00 (**57.4x**) | 1.5897e+00 (**82.8x**) | 9.4796e-01 (**49.4x**) |
| conical 25-40 | **1.4051e-02** | 9.8718e-01 (**70.3x**) | 1.3682e+00 (**97.4x**) | 8.8110e-01 (**62.7x**) |

The zeroth-order `jones_transmission()` on the same oracle:

| mount | shipped | no anchor | conjugate |
|---|---|---|---|
| normal | 8.0004e-04 | **8.0004e-04** | **8.0004e-04** |
| oblique 25 | **3.3235e-03** | 9.2673e-01 (279x) | 1.6412e+00 (494x) |
| conical 25-40 | **3.2249e-03** | 7.2244e-01 (224x) | 1.3460e+00 (417x) |

-- the normal-incidence row is the `P_0 = 1` exemption of S1.1, at a QUARTER
walk, reading identically on all three arms to every digit.

**The bound.**  On this fixture the shipped residual is limited by the FOURIER
TRUNCATION, not by the staircase: the oracle's own `K15 -> K25` step is
`1.45e-03 / 1.30e-03 / 1.31e-03`, while the HYBRID's own `n_orders 5 -> 7` step
is `1.505e-02 / 2.961e-02 / 2.095e-02`.  The shipped reading sits **1.60x /
1.54x / 1.49x BELOW the engine's own truncation step**, while the un-anchored
one sits **56x / 37x / 47x ABOVE it**.  Raising the truncation confirms it:
at oblique 25 against a `K = 25` oracle, shipped falls `1.920e-02 -> 1.004e-02
-> 7.890e-03` over `n_orders` 5 / 7 / 9 while the un-anchored arm reads
`1.1029 / 1.0902 / 1.0907`.

### 2.3 (a2) The decision that identifies a FRAME rather than an accuracy gap

Refine the oracle and watch which arm follows:

| mount | K = 5 | K = 15 | K = 25 | first/last |
|---|---|---|---|---|
| normal, shipped | 2.2403e-02 | 9.7894e-03 | 9.4221e-03 | **2.378x** |
| normal, no anchor | 8.4192e-01 | 8.4378e-01 | 8.4394e-01 | **0.9976x** |
| oblique 25, shipped | 2.9396e-02 | 1.9669e-02 | 1.9200e-02 | **1.531x** |
| oblique 25, no anchor | 1.1033e+00 | 1.1029e+00 | 1.1029e+00 | **1.0004x** |
| conical, shipped | 2.4968e-02 | 1.4496e-02 | 1.4051e-02 | **1.777x** |
| conical, no anchor | 9.8932e-01 | 9.8734e-01 | 9.8718e-01 | **1.0022x** |

The un-anchored arm moves by at most 0.24% over a 5x refinement of the oracle.
The shipped arm's improvement flattens because it hits the `n_orders = 5`
truncation floor, exactly as S2.2's ladder shows.

### 2.4 (b) Against the PURE engine

`q2b_pure.py`, a SQUARE 6 x 6 cell both engines express, hybrid `n_orders = 5`
against `PMM2DStackPure(n_modes = 4 and 5, n_orders = 3)`:

| | oblique 25 | conical 25-40 |
|---|---|---|
| per order, shipped (`n_modes = 4` / `5`) | **4.282e-02 / 4.309e-02** | **3.235e-02 / 3.247e-02** |
| the same with the anchor REMOVED | 1.0799e+00 (**25.2x**) | 9.3730e-01 (**29.0x**) |
| the same CONJUGATED | 1.6227e+00 (**37.9x**) | 1.3588e+00 (**42.0x**) |
| the BEST single GLOBAL phase on the un-anchored arm | 8.2584e-01 (**19.3x**) | 7.5076e-01 (**23.2x**) |
| the hybrid's OWN `n_orders` 5 -> 7 step (the bound) | 2.637e-02 | 2.135e-02 |
| zeroth-order Jones, shipped | **8.897e-03** | **7.647e-03** |
| ... no anchor / conjugate | 9.191e-01 (103x) / 1.6359e+00 | 7.151e-01 (93.5x) / 1.3409e+00 |
| the hybrid's OWN Jones `n_orders` 5 -> 7 step | 1.322e-03 | 1.806e-03 |

The two engines sit **1.62x / 1.51x** of the hybrid's own truncation step apart
on the per-order amplitudes and **6.7x / 4.2x** on the zeroth-order Jones --
discretization, not phase.  The pure arm is converged: the reading moves by
**0.6% / 0.4%** between `n_modes` 4 and 5.

(The pure engine's cost is `(segments x n_modes)` per axis per component;
`n_modes = 7` on a 6 x 6 cell is a ~7000-dof dense complex eig and ran for over
an hour before it was cut.  Recorded in the probe README.)

### 2.5 (c) Against the 1-D engine -- and V2, a THIRD instance of the defect

The brief asks for `pmm_jones_1d_slanted`; that function returns
`(orders, R, T, jones_REFLECTION)` and exposes no transmitted field at all, so
it cannot arbitrate an anchor.  The 1-D surface that CAN is
`PMMStack.add_sheared_grating` + `jones_transmission()`, and asking it the
question first produced a defect rather than an oracle.

`q2c_oned.py`, `P = 0.90 um`, `wl = 0.62 um`, `d = 0.45 um`,
`eps_ridge = 3.05`, `eps_groove = 1.30`, `duty = 0.5`, `shear = 0.25` (a
QUARTER-period walk), `theta = 25 deg`, `degree = 14`, `n_orders = 11`:

| `factorization` | transmitted field on a sheared grating |
|---|---|
| `auto` | **no per-order amplitudes retained** |
| `covariant` | **no per-order amplitudes retained** |
| `convection` | **retained** |

and against the 1-D engine's OWN z-staircase of the identical parallelogram
(`add_tapered_grating(duty_top == duty_bottom, shear=0.25, n_slices=N)`), which
is lab-referenced by construction:

| `n_slices` | `dJ_t` as returned | `dJ_t x P_0` | `dJ_t x conj(P_0)` | `dJ_reflection` | staircase's own step |
|---|---|---|---|---|---|
| 6 | 9.2951e-01 | **8.2212e-03** | 1.6451e+00 | 7.6075e-03 | -- |
| 12 | 9.2729e-01 | **2.0737e-03** | 1.6426e+00 | 2.1403e-03 | 6.5324e-03 |
| 24 | 9.2754e-01 | **1.3461e-03** | 1.6428e+00 | 1.1733e-03 | 1.9286e-03 |

`P_0 = exp(+i k0 alpha_0 W)`, `arg P_0 = 0.9636 rad`.  The as-returned arm is
**FLAT** to 0.24% over a 4x refinement while the `x P_0` arm converges through
the staircase's own step; the REFLECTION Jones converges as returned
(`7.6e-03 -> 1.2e-03`), exactly as the derivation says it must.  A `centre`
convention mismatch cannot explain it: a lateral translation of the whole
structure leaves the ZEROTH-order Jones invariant (order `m` picks up
`exp(i(alpha_0 - alpha_m) delta)`, which is 1 at `m = 0`), and the reflection
Jones -- same order, same solve -- agrees.

**V2: `PMMStack.jones_transmission()` / `per_order_amplitudes('transmission')`
on a slanted 1-D stack is FRAME-referenced -- the same silent-wrong as D1, one
dimension down, still open.**  Same signature: `R` (`1.2e-04`), `T`
(`8.7e-04`) and the reflection Jones all converge to the staircase; only the
transmission carries the un-removed unimodular phase.  Reproducer:
`validation/probe_verify_hybrid_anchor/q2c_oned.py`.

**The cross-engine arm itself.**  Running the same grating through the 2-D
hybrid as a y-uniform stripe and comparing its zeroth-order transmitted Jones
against the 1-D `n_slices = 24` staircase:

| arm | reading |
|---|---|
| the hybrid AS SHIPPED | **2.7547e-02** |
| the hybrid with the anchor removed | 9.0719e-01 (**32.9x**) |
| the hybrid conjugated | 1.6319e+00 (**59.2x**) |
| the hybrid's own `n_orders` 7 -> 9 step (the bound) | 1.2655e-02 |

i.e. the anchored 2-D hybrid agrees with a fully INDEPENDENT engine's
lab-referenced staircase to **2.18x its own truncation step**, and the
un-anchored one does not.

---

## 3. TASK 3 -- COMPOSITION

`q3_composition.py` / `q3b_split.py`, `n_orders = 5`, oracles = 15-rung
midpoint staircases with exact integer rolls, `W = px/4` unless stated.
Residuals are per-order transmitted amplitudes over 121 shared orders.

| case | stack | shipped | no sum | conjugate | oracle's own K5 -> K15 step |
|---|---|---|---|---|---|
| **A** | slanted PATTERNED over a UNIFORM film | **1.991e-02 / 1.436e-02** | 1.1024 / 9.829e-01 (55.4x / 68.4x) | 1.5903 / 1.3681 | 1.100e-02 / 1.063e-02 |
| **B** | a UNIFORM film ABOVE the slanted layer | **9.237e-03 / 1.060e-02** | 1.0264 / 1.0127 (111x / 95.5x) | 1.5900 / 1.3681 | -- |
| **E** | a slanted UNIFORM film BELOW | **0.000e+00** vs the vertical film (sha-equal) | -- | -- | putting it in the sum: 5.870e-01 / 5.331e-01 |
| **F** | a CONSTANT-tile slanted cell BELOW | **0.000e+00** (sha-equal) | -- | -- | 5.870e-01 / 5.331e-01 |

(each pair is oblique 25 / conical 25-40.)

### 3.1 TWO slanted layers at DIFFERENT slants and thicknesses -- the SUM rule

`d1 = 0.30 um` at `slant = 0.75` (walk `0.225 um = px/4`) over
`d2 = 0.15 um` at `slant = 0.60` (walk `0.090 um = px/10`); total
`W = 0.315 um = 0.35 px`.  Oracle: a 15-rung staircase of each, the lower one
built at the accumulated `px/4` offset (the frame continues -- S3.3).

| arm | oblique 25 | conical 25-40 |
|---|---|---|
| the FULL sum (shipped) | **3.013e-02** | **2.064e-02** |
| only layer 1's walk in the sum | 5.255e-01 (**17.4x**) | 4.492e-01 (**21.8x**) |
| only layer 2's walk in the sum | 1.1566 (**38.4x**) | 9.545e-01 (**46.2x**) |
| no sum at all | 1.4259 (**47.3x**) | 1.1712 (**56.7x**) |
| the conjugate sum | 1.6778 (**55.7x**) | 1.6258 (**78.8x**) |
| the oracle's own K5 -> K15 step | 1.059e-02 | 8.649e-03 |

### 3.2 The LAYER-SPLIT identity, and its BLINDNESS -- reproduced explicitly

One slanted layer of `d` against TWO slanted halves of `d/2` (`q3b_split.py`,
a 48-column cell so half the walk is an exact 6 pixels):

| | oblique 25 | conical 25-40 |
|---|---|---|
| `sha256(jones_transmission)` equal | **True** | **True** |
| `sha256(per_order_transmission)` equal | **True** | **True** |
| `dJones`, `d(per order)` | **0.000e+00 / 0.000e+00** | **0.000e+00 / 0.000e+00** |
| one layer vs the staircase | 1.9669e-02 | 1.4496e-02 |
| two halves vs the staircase | **1.9669e-02** | **1.4496e-02** |
| **one layer with HALF the sum** | **6.1055e-01** | **5.6189e-01** |
| **two halves with HALF the sum** | **6.1055e-01** | **5.6189e-01** |
| one layer / two halves with NO sum | 1.1029e+00 / 1.1029e+00 | 9.8734e-01 / 9.8734e-01 |

**That is the blindness, to the digit**: give both arms a wrong sum and they
read the SAME wrong number, so the split identity alone can never detect a
mis-summed walk.  What detects it is the comparison against the lab-referenced
staircase, where the half-sum arm is **31.0x / 38.8x** and the no-sum arm
**56.1x / 68.1x** worse than the shipped one.

The same probe also prices the alternative reading of the split: passing the
lower half's cell PRE-ROLLED by half the walk (i.e. assuming the frame restarts
rather than continues) reads **3.679e-01 / 3.553e-01** against the staircase,
18.7x / 24.5x worse.  The frame CONTINUES; that is the same statement as S3.4.

Re-run on WSL: both sha equalities hold, and every reading agrees with WIN to
nine significant figures (`1.9668813227e-02` vs `1.9668813213e-02`;
`6.105517938e-01` on both).

### 3.3 The REFLECTION round trip -- there is none

A reflecting `eps = 3.6`, `0.12 um` film under the sheared layer, so a strong
return comes back up through it (`q3_composition.py` case G):

| quantity | oblique 25 | conical 25-40 |
|---|---|---|
| `dR` vs the staircase + the same film | 1.802e-03 | 1.139e-03 |
| `dJones_reflection`, AS RETURNED | **1.525e-02** | **7.980e-03** |
| reflection `x P_0` (a one-way round trip) | 9.295e-01 (**60.9x**) | 7.192e-01 (**90.1x**) |
| reflection `x P_0^2` (a full round trip) | 1.6399e+00 (**107x**) | 1.3428e+00 (**168x**) |
| the staircase's own K5 -> K15 step on `Jones_refl` | 4.322e-03 | 5.940e-03 |
| per-order REFLECTED amplitudes, as returned | **9.698e-02** | **4.315e-02** |
| ... `x P_m` / `x P_m^2` | 1.6378 (16.9x) / 1.5397 (15.9x) | 1.2164 (28.2x) / 1.3989 (32.4x) |
| ... the oracle's own K5 -> K15 step | 2.686e-02 | 3.676e-02 |

Either round-trip factor is 61x .. 168x worse than applying nothing.  **The
frame is anchored at the sheared layer's TOP, which is where the reflected wave
leaves it, so no offset ever accumulates on that side.**  CONFIRMED.

### 3.4 SCOPE -- a PATTERNED layer below a slanted one (audit O1)

Test article: the hybrid, `[slanted BASE at px/4] + [vertical 12-column LOWER
cell as written]`.  Oracle: a 15-rung staircase of the upper layer, with the
lower cell placed three ways.  A QUARTER walk makes `+walk` and `-walk`
DIFFERENT translations of a 12-pixel cell (3 pixels vs 9).

| lower layer placed | per-order T | `dR` | `dJones_reflection` | per-order R |
|---|---|---|---|---|
| **oblique 25** | | | | |
| AS WRITTEN | 3.2088e-01 | 1.2514e-02 | 3.2662e-01 | 6.5689e-01 |
| translated by **+walk** | **4.0445e-02** | **6.324e-04** | **4.7197e-02** | **8.8478e-02** |
| translated by -walk | 3.2554e-01 | 9.482e-03 | 1.7505e-01 | 5.6882e-01 |
| **conical 25-40** | | | | |
| AS WRITTEN | 2.9178e-01 | 6.195e-03 | 2.6284e-01 | 3.9096e-01 |
| translated by **+walk** | **4.1367e-02** | **1.266e-03** | **7.1191e-02** | **8.2777e-02** |
| translated by -walk | 3.4885e-01 | 6.996e-03 | 1.8735e-01 | 5.8518e-01 |

The `+walk` arm wins by **7.9x / 7.1x** on the transmission, **6.9x / 3.7x** on
the REFLECTION Jones, **7.4x / 4.7x** on the per-order reflection and **19.8x /
4.9x** on `dR`.  The reflection carries no anchor at all, so those columns
identify the GEOMETRY and not the phase: **the hybrid solves the
shear-CONTINUED solid, in which the lower layer is translated by the
accumulated walk.**  CONFIRMED, on my own fixture, at both mounts.

---

## 4. TASK 4 -- the JAX refusal (fix 2), and V1

`q4_jax.py`, jax 0.11.0 with `jax_enable_x64`, a slanted PATTERNED layer
(`slant = (0.5, 0)`) at oblique 25, `n_orders = 5`.  **Seven** traced inputs
reach `_jax_stack2d` -- the brief names four; the fix's own comment names six;
the dispatch condition
(`is_jax_array` on `n_sub`, `n_sup`, `wavelength`, `theta`, `phi`, any layer
`t`, any layer `eps`, or a traced cell) admits seven distinct scalars, and all
seven were tested.

| traced input | `v5.44.0` | the fix |
|---|---|---|
| layer THICKNESS | SOLVED SILENTLY | **NotImplementedError** |
| WAVELENGTH | SOLVED SILENTLY | **NotImplementedError** |
| `theta` | SOLVED SILENTLY | **NotImplementedError** |
| `phi` | SOLVED SILENTLY | **NotImplementedError** |
| `n_substrate` | SOLVED SILENTLY | **NotImplementedError** |
| `n_superstrate` | SOLVED SILENTLY | **NotImplementedError** |
| a traced UNIFORM `eps` on a second layer | SOLVED SILENTLY | **NotImplementedError** |

On `v5.44.0` every one of the seven returned, with **zero warnings** and a
closure of `1.0146593453` (identical to the correct NumPy solve's, because the
super-unity is the `n_orders = 5` truncation residue and not a symptom):

| arm | reading |
|---|---|
| the traced solve vs the NumPy **VERTICAL** stack | `dR 1.06e-14`, `dT 1.10e-14`, `dJones 4.65e-14` |
| the traced solve vs the correct NumPy **SLANTED** answer | **`dR 3.462e-03`, `dT 6.889e-02`, `dJones 1.535e-02`** |
| the two NumPy answers against each other | `dR 3.462e-03`, `dT 6.889e-02`, `dJones 1.535e-02` |

-- the last two rows agreeing to `1e-14` is the proof that the JAX arm WAS the
vertical answer, not merely close to it.  The fix's own reading
(`dR 1.839e-02 / dJones 3.227e-02`) is the same statement on its own fixture.

**Controls, identical on BOTH arms to every digit** (so the refusal is about
the shear, not the API):

| control | reading |
|---|---|
| traced THICKNESS on a VERTICAL stack | SOLVES; vs NumPy `dR 1.06e-14`, `dT 1.10e-14`, `dJones 4.65e-14`; closure 1.0146593453 |
| traced THICKNESS on a CONSTANT-TILE "slanted" layer | SOLVES; vs the NumPy vertical film `dR 6.9e-18`, `dT 1.1e-16`, `dJones 0.0`; closure 0.9999999999999997 |
| `jax.grad` of `sum T` w.r.t. a traced thickness, VERTICAL | AD `-3.516093e+05` vs central FD (`h = 1e-10`) `-3.516090e+05`, **rel 1.20e-06** |

**All of S4 re-run on WSL** (CPython 3.12.3, jax 0.10.2, both arms): the same
seven SILENT rows on `v5.44.0` (`dR 3.4615883597e-03`, `dT 6.8890393006e-02`,
`dJones 1.5351005655e-02` -- WIN to ten significant figures), the same seven
refusals on the fix branch, the same two controls solving on both arms, and
`jax.grad` AD-vs-FD `rel 1.19e-06`.

### 4.1 The OTHER slant-carrying JAX route -- **V1, still open**

`grep --include='*.py' -rn 'slant' lumenairy/elements/pmm/_jax_stack2d.py
lumenairy/elements/pmm/stack2d.py` finds the string **zero** times in the jnp
twin, and `_jax_stack2d` has exactly ONE call site
(`stack2d.py:1285`), which the fix now guards.  But the same silent shape
exists in the SINGLE-LAYER 2-D Jones entry, whose JAX dispatch is a different
twin:

`lumenairy/elements/pmm/twod_jones.py`, `pmm_jones_2d(..., slant=...)` -- the
`if any(is_jax_array(a) for a in _jx)` branch at line 513 never looks at
`slant`, and hands off to `_pmm_jones_2d_cell_jax(...)` **without passing it**;
`slant = _norm_slant_pair(slant, ...)` is not even reached until after the
branch.  MEASURED, a slanted `(6, 4, 3, 3)` cell with a TRACED `depth`,
oblique 25, `n_orders = 3`, `degree = 5`:

| arm | `dR` | `dT` | `dJones` |
|---|---|---|---|
| the traced slanted call vs the NumPy **VERTICAL** call | 8.393e-15 | 7.883e-15 | 7.426e-14 |
| the traced slanted call vs the NumPy **SLANTED** call | **3.847e-03** | **5.285e-02** | **1.661e-02** |
| the two NumPy calls against each other | 3.847e-03 | 5.285e-02 | 1.661e-02 |

Silently, energy-conserving, no warning, and **the reading is identical on
`v5.44.0` and on this branch, on BOTH builds** (WSL: `dR 3.847171587e-03`,
`dT 5.285006357e-02`, `dJones 1.660890724e-02`) -- the fix did not close it.
`R` and `T` are wrong here too, so no unimodularity protects it.

Severity **P1**: same class as the defect `5dba5ee` closed, one function away,
on a public entry point.  `pmm_efficiency_2d_cell` has no `slant` parameter and
is not exposed; `PMM2DStackPure` has no JAX surface.  `pmm_jones_2d` is the
only other reachable route, and it is the one that is open.
Reproducer: `validation/probe_verify_hybrid_anchor/q4_jax.py`, section
`pmm_jones_2d_route`.

---

## 5. TASK 6 -- DURABILITY of the 27-test file

`q6_durability.py` imports the test module and re-measures every numeric bar
THROUGH THE FILE'S OWN FIXTURES, on both builds.  **49 quantities.**

### 5.1 Cross-build spread -- the number the bars must clear

| | value |
|---|---|
| quantities re-measured on WIN and WSL | **49** |
| worst RELATIVE WIN <-> WSL gap on any physically meaningful row | **7.02e-08** (`T none/shipped, film, oblique25`: 1.468415e+02 vs 1.468416e+02) |
| second worst | 6.78e-08 (`dJones(refl) vs step_J, oblique25`) |
| rows agreeing to better than 1e-9 relative | 20 of 49 |
| the only larger row | `||P_0| - 1|` at conical: 1.110e-16 (WIN) vs 0.0 (WSL) -- machine epsilon against zero, not a spread |

The fix audit's claim that WIN and WSL "print identically to every digit" is
CONFIRMED at four significant figures and quantified here at seven.  Every bar
in the file has a margin of at least 1.05x, i.e. **>= 7e5 times the measured
cross-build spread**.

### 5.2 The bar table (WIN; WSL identical to within S5.1)

| test | bar | measured | margin | note |
|---|---|---|---|---|
| a1 | `shipped < oracle K3->K15 step` | 7.7782e-03 / 7.3806e-03 | **5.29x / 3.56x** | step 4.1185e-02 / 2.6291e-02 |
| a1 | `none/shipped > 20x` | 143.15 / 121.07 | 7.16x / 6.05x | |
| a1 | `conj/shipped > 20x` | 159.62 / 176.53 | 7.98x / 8.83x | |
| a1 | `conj > none` | 1.1150 / 1.4581 | **1.12x / 1.46x** | **RESTATED** -> `conj/none > 1.05` |
| a2 | `shipped first/last > 2x` | 4.9973 / 3.6273 | 2.50x / 1.81x | ladder 3.887e-02 / 1.490e-02 / 7.778e-03 |
| a2 | `none first/last in (0.9, 1.1)` | 1.00000 / 0.99656 | 29x (deviation 0.0% / 0.34% of a 10% allowance) | |
| a3 | `best global / shipped > 10x` | 49.13 / 60.64 | 4.91x / 6.06x | |
| a4 | `||P_0| - 1| < 1e-14` | 0.0 / 1.110e-16 | 90x .. inf | |
| a4 | `|arg P_0| > 1.0 rad` | 1.9525 / 1.4957 | 1.95x / 1.50x | fixture property, deterministic |
| a4 | `jones none/shipped > 20x` | 186.21 / 121.07 | 9.31x / 6.05x | |
| a5 | `1.0 <= closure < 1.05` | 1.011002 / 1.008581 (slanted), 1.012042 / 1.010179 (stair), 1.011813 / 1.009474 (film) | upper **4.2x .. 5.8x**, lower **8.6e-03 absolute** | **RESTATED** -> lower bar 0.99 |
| a6 | `max ||P|-1| < 1e-13` | 0.0 | inf | |
| a6 | `ptp(arg P) > 3.0` | 3.14159 (`= pi` exactly) | **1.05x** | **RESTATED** -- see S5.3 |
| b1 | `cross-engine < 10x own step` | 2.1720 / 2.1793 | 4.60x / 4.59x | own step 5.981e-03 / 4.568e-03 |
| b1 | `pre-fix/shipped > 20x` | 85.71 / 89.53 | 4.29x / 4.48x | |
| c2 | `dR < 2 step_R` | 9.7727e-04 / 1.1960e-03 | 4.11x / 3.05x | step_R 2.006e-03 / 1.821e-03 |
| c2 | `dJones(refl) < step_J` | 2.4714e-03 / 1.8887e-03 | 3.37x / 4.07x | step_J 8.333e-03 / 7.686e-03 |
| c3 | `cost of anchoring a uniform film > 1.0` | 1.3710 | **1.37x** | deterministic; 2.05x the amplitude scale 6.694e-01 |
| d1 | `half-sum/full > 20x` | 81.46 / 65.56 | 4.07x / 3.28x | |
| d1 | `no-sum/full > 20x` | 143.15 / 121.07 | 7.16x / 6.05x | |
| d2 | `refl as returned < step_J` | 3.9088e-03 / 3.8594e-03 | 2.95x / 3.56x | step_J 1.154e-02 / 1.374e-02 |
| d2 | `x P_0 > 10 step_J` | 2.2822e-01 / 2.1690e-01 | **1.98x / 1.58x** | = 19.8x / 15.8x the step; **58.4x / 56.2x the AS-RETURNED reading** -- **RESTATED** |
| d2 | `x P_0^2 > 10 step_J` | 2.5825e-01 / 3.2109e-01 | 2.24x / 2.34x | = 22.4x / 23.4x the step; 66.1x / 83.2x the as-returned reading |
| d2 | `T none/shipped > 5x` | 146.84 / 14.34 | 29.4x / 2.87x | |

Everything else in the file is STRUCTURAL (sha256 equality, exact `(0.0, 0.0)`
tuples, `is False` / `is True` on the two helpers, `pytest.raises`) and carries
no numeric bar.  `e1`'s bars (3x on the transmission, 2x on `dR` and the
reflection Jones) were re-measured independently in S3.4, where the same
decision reads 7.1x .. 7.9x and 3.7x .. 6.9x.

### 5.3 The four restatements (test-only, this branch)

1. **`a1`'s `conj > none`** was an ORDERING with no bar; its smallest
   separation is 11.5%.  Now `conj / none > 1.05`, with the measured 1.1150 /
   1.4581 and the 7.0e-08 cross-build spread stated at the site.
2. **`a5`'s lower closure bar** was `1.0`, i.e. it pinned the SIGN of a
   truncation residue with 8.6e-03 .. 1.2e-02 of room.  Nothing in the physics
   requires the residue to land above one.  Now `0.99`, which still refuses a
   silently-lossy answer; the upper bar (4.2x .. 5.8x of room) is unchanged.
3. **`a6`'s non-vacuity check** `ptp(arg P_m) > 3.0` read `3.14159` -- exactly
   `pi`, a 4.7% margin -- and for a reason the comment did not state: at the
   file's HALF walk `P_m = exp(i arg P_0) exp(i pi m)`, so over all 121 orders
   it takes exactly **TWO** values.  `ptp` was measuring the half-walk
   degeneracy, not a spread.  Both facts are now asserted directly, and a
   QUARTER walk -- where `P_m = exp(i arg P_0) i^m` takes **FOUR** values with
   `ptp > 3.0` -- is evaluated on the SAME solve (the factor depends only on
   `kx` and the walk), so the genuine spread costs no extra runtime.
4. **`d2`'s round-trip bars** keep `> 10 step_J` and gain the better-margined
   form of the same statement, `x P_0 / as_returned > 10` (measured 58.4x /
   56.2x) and `x P_0^2 / as_returned > 10` (66.1x / 83.2x).  The docstring's
   "55x .. 83x above it" is corrected: that range is the ratio to the
   AS-RETURNED reflection, not to the oracle's step (against which the arms are
   15.8x .. 23.4x).  **V3.**

No test was added or removed, so `.test_durations` needs no splice.

### 5.4 Timing

| | WIN | WSL |
|---|---|---|
| the 27-test file, alone | **85.11 s** | **100.15 s** (under concurrent load) |
| slowest test | `test_b1_...[oblique25]` **22.30 s** | `test_b1_...[conical25_40]` **26.61 s** |
| second slowest | `test_b1_...[conical25_40]` 21.08 s | `test_b1_...[oblique25]` 21.57 s |
| third | `test_e1_...rides_the_shear` 10.06 s | -- |

The fix audit records 63.5 s / slowest 15.3 s on WIN, which this machine does
not reproduce even idle: the file reads **85.11 s** here with a slowest test of
**22.30 s**, and 100.15 s / 26.61 s on WSL under a co-running suite.  That is
not a discrepancy in the fix's favour or against it -- it is a machine and load
difference -- but it matters for the file's own budget: the headroom against
the 40 s per-test cap falls from the recorded 2.6x to **1.79x (WIN, idle)** and
**1.50x (WSL, loaded)**, and the 3-minute file budget from 2.8x to 2.1x / 1.8x.

`test_b1` is the pure engine's `n_modes = 4` solve, whose cost is super-linear
in `n_modes` and in the cell's segment count (S2.4: `n_modes = 7` on a 6 x 6
cell runs for over an hour).  **That margin should not be spent.**  Noted, not
changed.

---

## 6. TASK 5 -- audit item O2, the cascade blow-up: REPRODUCED and EXPLAINED

`q5_o2_blowup.py` (two fixtures of my own), `q5b_o2_exact.py` (the EXACT
fixture the fix's probes rejected, 96 solves), `q5c_mechanism.py` (the
mode-level instrumentation) and `q5d_conditioning.py` (the interface-level
one).  The exact fixture, from
`validation/probe_fix_hybrid_slant_anchor/p1_census.py` plus `p4_test_bars.py`'s
film: `px = py = 1.20 um`, `wl = 0.68 um`, `n_sup = 1.0`, `n_sub = 1.5`, a
6 x 6 cell whose x-profile is `[4, 4, 2, 1, 1, 1]` on a ground of `1.0` -- so
**the cell contains `eps = 1.0`, the superstrate's own permittivity** --
`d = 0.50 um` at `slant = (0.5, 0)`, over a `0.25 um` `eps = 3.6` film.

Fifth probe: `q5e_discriminators.py`, the one-axis-at-a-time scans that decide
what the blow-up actually depends on.

### 6.1 REPRODUCED -- and worse than reported

96 solves (4 stack shapes x 4 mounts x `n_orders` 3/5/7/9/11/13):

| shape | mount | `n_orders` | `max(sum R + T)` |
|---|---|---|---|
| slant + film | oblique 25 | 3 | **4.5408e+28** |
| slant + film | oblique 25 | 5 | **5.8448e+29** |
| slant + film | oblique 25 | 7 | **1.6264e+29** |
| **slant ALONE** | oblique 25 | 3 | **3.7610e+27** |
| **slant ALONE** | oblique 25 | 5 | **1.2570e+30** |
| **slant ALONE** | oblique 25 | 7 | **6.3422e+30** |
| everything else (90 rows) | | | `<= 1.0801` |

and the truncation ladder on the worst row (slant ALONE, oblique 25):

| `n_orders` | 3 | 5 | 7 | 9 | 11 | 13 |
|---|---|---|---|---|---|---|
| `sum R + T` | 3.761e+27 | 1.257e+30 | 6.342e+30 | **1.00651** | **1.00063** | **1.00196** |

**The audit's framing is REFUTED in one respect and confirmed in the rest.**
O2 says "while the SINGLE-layer rows on the same fixture were fine".  They are
not: the SINGLE slanted layer with no film at all blows up on the same three
`(mount, n_orders)` pairs, and harder (6.34e+30 against 1.63e+29).  It is not a
two-layer cascade effect.  Everything else O2 says reproduces: it is only some
`(n_orders, mount)` pairs (6 of 96), it is the `eps = 1.0` cell against a
near-cut-off order, and `n_orders = 9` cures it.

The blow-up is confined to **oblique 25**, the only mount on this fixture where
an order sits within about 1% of a half-space cut-off:

| mount | nearest order to the `eps = 1.0` cut-off | relative gap |
|---|---|---|
| **oblique 25** | `(+1, 0)` at `abs(alpha) = 0.989285` | **-1.07e-02** |
| conical 25-40 | nothing within 5% of 1.0 | -- |
| oblique 40 | nothing within 5% of 1.0 | -- |
| normal | nothing within 5% of 1.0 | -- |

and because the CELL contains `eps = 1.0`, that same `abs(alpha) = 0.989285` is
simultaneously within 1.07e-02 of the layer's OWN `eps = 1.0` region cut-off.
That is the degenerate layer <-> region mode match, named.

### 6.2 The MECHANISM, by measurement -- it is the INTERFACE SOLVE

Two candidate mechanisms were instrumented.  One is refuted, the other is
decisive.

**REFUTED: the forward/backward split.**  `q5c_mechanism.py` patches
`_layer_modes_projected` and counts, per layer, the modes placed on the wrong
side (a FORWARD mode with `Re(gam) < 0`, or a BACKWARD mode with
`Re(gam) > 0` -- either puts a GROWING `exp(|Re gam| k0 d)` into the
propagation S-matrix).  Such modes do exist, and their growth factors reach
9.40 -- but they do not predict the blow-up:

| mount, `n_orders` | worst growth rate | growth factor | `sum R + T` |
|---|---|---|---|
| oblique 40, M = 3 | 0.4851 | **9.40** | **1.0626** (fine) |
| conical 25-40, M = 5 | 0.4630 | **8.49** | **1.0446** (fine) |
| oblique 25, M = 7 | 0.1080 | 1.65 | **6.3422e+30** |

A growing exponential of 9.4 in the propagation S-matrix is survivable; a
factor of 1.65 accompanies a 1e+30.  (The modes concerned sit just under
`_select_forward_flux`'s hard-coded `|Re gam| > 0.5` "stability band", which is
worth knowing, but it is not this defect.)

**CONFIRMED: `cond(T22)` in `_interface_smatrix_general`.**
`q5d_conditioning.py` records, per interface, `cond` of the mode matrix the
interface solves against (`Mb`) and `cond` of the
`T22 = (inv(Mb) Ma)[2N:, 2N:]` block that `_guarded_inverse` inverts
EXPLICITLY.  The correlation is total:

| mount, M | `cond(Mb)` | **`cond(T22)`** | max prop factor | `sum R + T` |
|---|---|---|---|---|
| oblique 25, M = 3 | 171.4 | **2.22e+15 / 4.49e+15** | 6.91 | **4.54e+28 / 3.76e+27** |
| oblique 25, M = 5 | 171.4 | **4.48e+15** | 8.65 | **5.84e+29 / 1.26e+30** |
| oblique 25, M = 7 | 171.4 | **6.14e+14 / 1.19e+15** | 1.65 | **1.63e+29 / 6.34e+30** |
| oblique 25, M = 9 | 171.4 | **33.8 / 53.8** | 1.53 | 1.0068 |
| conical 25-40, M = 5 | 68.7 | 1.63e+03 | 8.49 | 1.0446 |
| oblique 40, M = 3 | 29.0 | 47.8 / 56.1 | 9.40 | 1.0626 |
| normal, M = 5 | 87.9 | 212.0 | 1.53 | 1.0394 |

`cond(T22) >= 6e+14` on every blown-up row and `<= 1.6e+03` on every healthy
one -- **eleven orders of magnitude, with nothing in between**.  `cond(Mb)` is
171.4 on ALL FOUR oblique-25 rows, the cured `M = 9` included, so the
half-space mode matrix is not the problem: the singularity is in the LAYER <->
HALF-SPACE match, precisely where the near-degenerate `eps = 1.0` order sits.
`4.5e+15` is at the edge of double precision, so `T22` is numerically singular
and its explicit inverse is noise.

**Why the SLANT meets it and a vertical layer does not.**  Measured on the same
solid, same mount, same truncation: the VERTICAL layer never enters
`_interface_smatrix_general` at all -- `q5c`'s instrumentation records
**0 generalized layers** on every vertical row -- because a vertical in-plane
layer keeps the `[W; -V] <-> -lam` symmetry and uses `_interface_smatrix`,
whose inverted block is differently structured.  Across all 16 vertical
`(mount, n_orders)` rows the worst reading is `sum R + T = 1.0796`.  The slant
is what forces the 4N generalized cascade, and the generalized cascade is what
carries the singular `T22`.

**The PURE engine is stable on the identical solid**, at every pair tested:
`sum R + T = 1.0000015 / 1.0000058 / 1.0000010 / 0.9999987` (oblique 25,
conical 25-40, oblique 40, normal).  It does not share the hybrid's Rayleigh
interface match.

### 6.2b What CONTROLS `cond(T22)` -- and the two attributions that do NOT

`q5e_discriminators.py` varies one axis at a time on the exact fixture at
`n_orders = 5`.  All 30 rows keep the S6.2 correlation exactly: `cond(T22)`
above `8e+14` on every blown-up row, below `1.4e+03` on every healthy one.

**REFUTED -- "the documented near-degenerate layer <-> region mode match".**
That is the attribution O2 gives, and it is the failure class the library's own
`_EnergyWarning` describes, whose stated remedy is to "DETUNE one of the
coincident permittivities by a relative ~1e-6".  Measured, the detune does not
work:

| arm | `sum R + T` | `cond(T22)` |
|---|---|---|
| as written (cell contains `eps = 1.0` = the superstrate) | 5.845e+29 | 4.48e+15 |
| cell's `eps = 1.0` detuned by **1e-6** | **3.460e+28** | 2.15e+15 |
| cell's `eps = 1.0` detuned by **1e-4** | **2.162e+28** | 1.88e+15 |
| the SUPERSTRATE detuned by 1e-6 instead | **2.196e+28** | 1.68e+15 |

Removing the coincidence entirely leaves the blow-up 28 decades out.  The
`eps = 1.0` pixel is not what triggers this.

**REFUTED -- the near-cut-off order.**  Scanning `theta` moves the `+1` order
across the `|alpha| = 1` cut-off; the blow-up does not follow it:

| `theta` | 15 | 20 | **22** | 24 | 25 | 26 | 28 | 30 | **35** |
|---|---|---|---|---|---|---|---|---|---|
| `alpha_+1` | 0.825 | 0.909 | **0.941** | 0.973 | 0.989 | 1.005 | 1.036 | 1.067 | **1.140** |
| `sum R + T` | 7.7e+28 | 7.7e+29 | **1.058** | 1.1e+31 | 5.8e+29 | 5.4e+29 | 6.2e+28 | 2.5e+28 | **1.068** |
| `cond(T22)` | 4.1e+16 | 7.0e+15 | **1.4e+03** | 1.5e+16 | 4.5e+15 | 5.3e+15 | 2.4e+15 | 1.9e+15 | **50.5** |

`theta = 15` is 17% BELOW the cut-off and blows up; `theta = 22` is 6% below it
and is fine; `theta = 30` is 7% ABOVE it and blows up.  There is no monotone
relation to the cut-off gap at all.

**CONFIRMED -- the three DISCRETIZATION knobs of the sheared generator.**

| the SLANT magnitude (wall tilt) | `sum R + T` | `cond(T22)` |
|---|---|---|
| vertical (no generalized cascade at all) | 1.0412 | -- |
| 1e-4 .. 0.10 (0.006 .. 5.7 deg) | 1.0412 .. 1.0421 | 25 .. 39 |
| **0.25 (14.0 deg)** | **1.0431** | **49** |
| **0.35 (19.3 deg)** | **6.333e+27** | **3.57e+15** |
| 0.50 (26.6 deg) | 5.845e+29 | 4.48e+15 |
| 1.00 (45.0 deg) | 1.701e+29 | 1.28e+16 |

| the spectral `degree` (at slant 0.5) | 7 | 9 | 11 | 13 | 15 |
|---|---|---|---|---|---|
| `sum R + T` | **1.051** | 3.5e+28 | 5.8e+29 | 5.8e+28 | 2.8e+27 |
| `cond(T22)` | **44.4** | 3.0e+15 | 4.5e+15 | 8.7e+14 | 7.8e+14 |

| `n_orders` (at slant 0.5, degree 11) | 3 | 5 | 7 | **9** | **11** | **13** |
|---|---|---|---|---|---|---|
| `sum R + T` | 3.8e+27 | 1.3e+30 | 6.3e+30 | **1.00651** | **1.00063** | **1.00196** |

So the trigger is the DISCRETIZATION of the sheared generator -- shear
magnitude above roughly a 15-20 degree wall tilt, together with a `degree` and
an `n_orders` in the wrong window -- and not any coincidence in the physical
structure.  That is a materially different statement from O2's, and it changes
the remedy: detuning does nothing, while raising `n_orders` (or, here, lowering
`degree`) does.

### 6.3 Is it LOUD?  Yes -- but not DISCRIMINATING

Every one of the 6 blow-ups raises `_warn_stack_energy`'s
`UserWarning: PMMStack.solve: energy not conserved (max R+T = ... > 1) -- a
near-singular interface mode-match in the cascade`.  **Zero silent blow-ups in
96 solves** (`q5b`'s `silent_blowups` list is empty), so this is NOT a
silent-wrong: nothing here can be mistaken for an answer without a warning
having been printed, and the warning text names the right cause.

The qualification is that the SAME warning fires on 17 perfectly ordinary rows
whose only sin is `sum R + T = 1.05 .. 1.08` at `n_orders = 3` -- VERTICAL rows
included.  A user who has learned to ignore the tripwire (as the fix's own test
module does, at module scope, for exactly that reason) would not notice a 1e+30
among them.  The tripwire is a single threshold at `1.0 + 1e-2` with no second
band.

Two further facts a fix would need:

* `_guarded_inverse`'s advertised "screen-and-refuse guard" is **dormant by
  default**: its body returns `xp.linalg.inv(A)` immediately unless
  `_INV_CENSUS` is not `None`, and `_INV_CENSUS = None` at module scope
  (`lumenairy/elements/rcwa/_core.py:641`).  Nothing screened
  `cond(T22) = 4.5e+15`.
* `stabilize=` is genuinely unavailable on `PMM2DStackHybrid.__init__`
  (parameters: `period_x, period_y, n_superstrate, n_substrate, degree,
  elements_per_strip, grade, n_orders, formulation, symmetry, max_nodal_dof,
  cascade, cache_max_bytes, tree_max_bytes`), as O2 says.  The remedy measured
  here is `n_orders = 9`, or the PURE engine.

### 6.4 SEVERITY

**P2, pre-existing, LOUD, MIS-ATTRIBUTED, with a remedy that is not the
documented one.**

Not P1: it cannot return a plausible-looking wrong answer silently -- the
failure is 27 to 31 decades out and warned every time, in 96 of 96 solves on
the exact fixture and 40 of 40 on two of my own.

Not P3, and this is where it is worse than O2 reads.  The blow-up is NOT gated
by a permittivity coincidence or a near-cut-off order -- S6.2b refutes both --
so "do not put the superstrate's `eps` in the cell" and "keep the orders clear
of a cut-off" do not protect a caller.  What gates it is the SHEAR
DISCRETIZATION: on this fixture ANY wall tilt above about 19 degrees at
`n_orders` 3 / 5 / 7 and `degree` 9 .. 15 blows up, and a 14-degree tilt at the
same settings does not.  A wall tilt above 19 degrees is an ordinary slanted
grating; `degree = 11` is the class default; `n_orders = 5 .. 7` is an ordinary
truncation.  Worse, the library's OWN advice for this warning class (the
`_EnergyWarning` on `pmm_jones_2d`: "DETUNE one of the coincident
permittivities by a relative ~1e-6") is MEASURED HERE TO NOT WORK, and it is
the advice a user who hits this will find.

The remedies that do work, measured: `n_orders >= 9` (1.0065 / 1.0006 /
1.0020 at 9 / 11 / 13), `degree = 7` (1.051), a slant at or below 14 degrees,
or the PURE engine (1.0000015).  `stabilize=` is unavailable on the class, as
O2 says.

A `cond(T22)` band in `_interface_smatrix_general` -- refuse above, say,
`1e+12`, with a message naming `n_orders` / `degree` / the shear -- would turn
this from a warning among warnings into a refusal, and the eleven decades of
separation measured over 30+ rows say such a band would be unambiguous.  Not
attempted here (no library edits in a verification).

Reproducers: `validation/probe_verify_hybrid_anchor/q5b_o2_exact.py` (the map),
`q5c_mechanism.py` (the refuted split mechanism and the vertical control),
`q5d_conditioning.py` (the confirmed conditioning mechanism, the cut-off table
and the pure control), `q5e_discriminators.py` (the four one-axis scans).

---

## 7. TASK 7 -- SUITE RUNS

WIN, `OMP` = `OPENBLAS` = `MKL` = 1, `-p no:randomly`: the five files the brief
names plus every test file importing `PMM2DStackHybrid`
(`grep -rl --include='*.py' PMM2DStackHybrid tests/`).  That grep now returns
**twelve** files, one more than the fix audit's eleven --
`test_pmm2d_staggered_mortar.py` has since started importing the class.

| file | tests |
|---|---|
| `test_fix_hybrid_slant_transmission_anchor.py` | 27 |
| `test_pmm2d_slant_metric.py` | 25 |
| `test_pmm2d_staggered_slant.py` | 70 |
| `test_v5_14_0_pmm2d_stack.py` | 8 |
| `test_p2c_pmm2d_stack_cascade.py` | 34 |
| `test_p2t_pmm2d_tree_cascade.py` | 57 |
| `test_audit_dynameta_consumer_api_2.py` | 12 |
| `test_niche_audit_w7_pmm.py` | 137 |
| `test_pmm2d_oop_block_eig.py` | 11 |
| `test_pmm2d_staggered_mortar.py` | 31 |
| `test_pmm_m3_efficiency.py` | 46 |
| `test_v5_21_delta_audit.py` | 14 |
| **total** | **472 passed, 0 failed, in 2143.89 s (35:43)** |

72 warnings, all pre-existing (the `PMM2DStack is a TRANSITIONAL alias`
deprecation and `_pmm_union_grid`'s near-coincident-wall snap notice, both
raised on purpose by their own tests).  The slowest tests are
`test_audit_dynameta_consumer_api_2.py::test_c3_pure_lossless_zero_and_contract`
(466.24 s) and `::test_b2_pure_amplitudes_vs_rcwa_conical` (453.88 s), both
pre-existing and unrelated.

WSL, the three files this branch touches: **122 passed in 296.88 s**
(27 + 25 + 70).

`ruff check lumenairy/ tests/` -- **All checks passed!**
`ruff check validation/probe_verify_hybrid_anchor/` -- **All checks passed!**
(`validation/` is in `pyproject.toml`'s `extend-exclude`, so the probe
directory is out of the repository lint's scope by configuration; it is linted
here anyway.)

`.test_durations`: **not spliced, and it should not be** -- this verification
adds no test.  The four restatements of S5.3 change assertions inside existing
test ids; `a6` gained a second, solve-free arm and `d2` two extra ratio
assertions, neither of which changes a test id or measurably its duration
(`a6` 0.9 s, `d2` unchanged at ~2 s per parameter).

---

## 8. COMMITS

On `verify/hybrid-slant-anchor`, off `wave2/pmm2d` @ `39bd54e` (the merge that
picked up the two lint-only commits was a fast-forward).  Nothing merged,
pushed, tagged or version-bumped; **no library file was edited**.

| commit | what |
|---|---|
| `a57e7a8` | `test(probes)`: `validation/probe_verify_hybrid_anchor/` -- the first 13 scripts, the README and every run's JSON on both arms and both builds |
| `2b285fd` | `test(pmm2d hybrid anchor)`: the four durability restatements of S5.3 |
| `487f465` | `test(probes)`: the O2 mechanism probes -- `q5c` (mode level), `q5d` (interface level), `q5e` (the four discriminators) |
| `24b16a0` | `docs`: this document |

---

## 9. WHAT COULD NOT BE VERIFIED, AND WHAT IS DELIBERATELY OUT OF SCOPE

1. **A MAGNETIC hybrid stack does not exist.**  The brief's bit-identity list
   names "magnetic stacks"; `PMM2DStackHybrid.add_layer` has no `mu` surface at
   all (S1.3).  Not verifiable, and nothing to verify.
2. **`pmm_jones_1d_slanted` cannot arbitrate the anchor.**  It returns
   `(orders, R, T, jones_REFLECTION)` and exposes no transmitted field, so the
   brief's arm (c) was taken on `PMMStack.add_sheared_grating` +
   `jones_transmission()` instead -- which is what turned up **V2**.  The
   comparison against the 1-D FUNCTION was therefore done only on the
   anchor-free quantities.
3. **The fix's own 21 fixtures and 144 hashes were not re-run.**  This
   verification re-measured with 29 fixtures and 200 hashes of its own; the two
   sets agree structurally (their 7 moving hashes are 4 slanted patterned rows
   x 2 transmission keys minus the 1 normal-incidence `jones_transmission`
   exemption, which is exactly the pattern S1.1 reproduces) but the hash VALUES
   are not comparable across fixture sets.
4. **The `O1` decision is not taken here.**  S3.4 supplies the numbers a
   decision needs and S10 states the recommendation, but adopting or refusing a
   `PMM2DStackPure`-style refusal in the hybrid is a behaviour change, not a
   measurement.
5. **V1 and V2 are reported, not fixed.**  Both are library changes; this is a
   verification branch and no library file was edited.
6. **The O2 blow-up was not traced below `cond(T22)`.**  S6.2 / S6.2b establish
   the proximate cause and the three knobs that control it, and refute two
   attributions, but *why* the truncated sheared mode set becomes rank-deficient
   against the Rayleigh half-space set above a ~19-degree tilt is a question
   about the generator's own conditioning that this bounded investigation did
   not open.
7. **Cross-build coverage is WIN + WSL only.**  Both are OpenBLAS builds
   (scipy-openblas); a genuinely different BLAS (MKL, Accelerate) was not
   available here, so the 7.0e-08 spread of S5.1 is a two-OpenBLAS number.

---

## 10. THE O1 DECISION -- the numbers, and a recommendation

O1 asks whether the hybrid should adopt `PMM2DStackPure._check_stack_slant`'s
refusal of a PATTERNED layer below a slanted one.  The measurements a decision
needs (S3.4, my own fixture, both mounts, reproducing the fix's finding on a
different cell):

| | oblique 25 | conical 25-40 |
|---|---|---|
| the solid the hybrid actually solves | the lower layer TRANSLATED by `+walk` | the same |
| how strongly the reflection says so (`dJones_refl`, as-written / `+walk`) | **6.9x** | **3.7x** |
| the same on `dR` | **19.8x** | **4.9x** |
| the same on the per-order reflection | **7.4x** | **4.7x** |
| the same on the anchored transmission | **7.9x** | **7.1x** |
| how strongly it says the answer is NOT the as-written solid | `-walk` is 4.9x .. 8.4x worse than `+walk` too |

**Recommendation: do NOT adopt the refusal; document and gate it, which is what
this fix already did.**  Three reasons, each a measurement rather than a
preference:

1. **The answer is not wrong, it is a different structure.**  The hybrid
   returns the exact solution of a realizable solid -- the shear continued
   through the lower layer -- and the anchor-free REFLECTION identifies that
   geometry by 3.7x .. 19.8x.  A refusal would remove a correct answer.
2. **The two engines' refusal is not the same object.**  The pure engine
   refuses because its NODAL basis cannot represent a real lateral translation
   of one grid against another unless the offset is a whole number of cells;
   the hybrid's Fourier projection has no such constraint, so the pure engine's
   ground for refusing does not exist in the hybrid.  Mirroring the refusal
   would import a limitation rather than a safeguard.
3. **The cost of refusing is unbounded and the cost of documenting is zero.**
   Any stack with a patterned layer under a slanted one runs today; a refusal
   breaks all of them at once.  The shipped `add_layer` docstring now states
   the scope and `test_e1_...rides_the_shear` pins it, so it cannot change
   unnoticed -- which is the whole safety benefit a refusal would have bought.

If the cross-engine inconsistency is judged unacceptable on its own terms, the
cheaper direction is the opposite one: RELAX the pure engine's refusal for the
case its own basis can represent exactly (an offset that is a whole number of
grid cells), rather than tighten the hybrid's.  That is a build task, not a
verification one.

The MIXED-SLANT half of O1 (patterned layers at DIFFERENT slants, which the
pure engine also refuses) is a separate question.  S3.1's case D is evidence
for the same answer -- its oracle places the second cell at the ACCUMULATED
walk and the shipped stack matches it to 2.8x / 2.4x the oracle's own step,
which a wrong placement would not do -- but the alternative placements were not
scanned for that shape as they were for S3.4, so it is one arm rather than
three.
