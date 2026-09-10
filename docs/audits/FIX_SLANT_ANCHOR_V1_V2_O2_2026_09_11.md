# FIX -- V1, V2 and O2 of the hybrid-slant-anchor verification

**Date:** 2026-09-11 - **Worktree:** `C:/tmp/lum_slfix`, branch
`fix/slant-anchor-v1-v2-o2` off `wave2/pmm2d` @ `4a987e3`.
**Under repair:** the three items
`docs/audits/VERIFY_HYBRID_SLANT_TRANSMISSION_ANCHOR_2026_09_11.md` left open --
**V1** (P1, silent-wrong), **V2** (P1, silent-wrong) and **O2** (P2, loud but
unscreened).
**Probes:** `validation/probe_fix_slant_anchor_v1v2o2/` -- own `_lib.arm()`,
which decides the ARM from `lumenairy.__file__`, refuses any tree but the two
named in S0, and stamps the tree, interpreter, numpy, scipy and thread caps into
every JSON; `summarize.py` re-reads those JSONs and prints the tables below, so
no number here is transcribed by hand.
**Binding law:** `docs/TESTING_STANDARDS.md`.
**Gate:** `tests/unit/test_fix_slant_anchor_v1_v2_o2.py`.

---

## 0. ARMS

| | WIN | WSL |
|---|---|---|
| tree (POST-FIX) | `C:/tmp/lum_slfix` | the same tree via `/mnt/c` |
| tree (PRE-FIX) | `C:/tmp/lum_slfix_pre` | the same via `/mnt/c` |
| interpreter | CPython 3.14.6 | CPython 3.12.3 |
| numpy / scipy | 2.4.4 / 1.17.1 | 2.4.6 / 1.17.1 |
| jax | 0.11.0, x64 | 0.10.2, x64 |
| threads | `OMP` = `OPENBLAS` = `MKL` = 1 | `OMP` = `OPENBLAS` = 1 |

"PRE-FIX" means `C:/tmp/lum_slfix_pre` -- a READ-ONLY `git archive` of
`lumenairy/` at the branch point `4a987e3`, extracted so the fail-before arm
could run WITHOUT flipping the working tree under a concurrent suite run.
"POST-FIX" means the branch tip.  `_lib.arm()` decides which is which from
`lumenairy.__file__` alone, stamps it into every JSON, and refuses any third
tree.  BOTH ARMS RAN ON BOTH BUILDS.

## SUMMARY

| item | what shipped | commit |
|---|---|---|
| **V1** | `pmm_jones_2d`'s JAX dispatch REFUSES a slanted PATTERNED cell, naming the traced input(s) that routed the call | `c8cb563` |
| **V2** | `PMMStack`'s TRANSMITTED per-order amplitudes are re-referenced from the sheared FRAME to the substrate's LAB basis | `006c031` |
| **O2** | `_interface_smatrix_general`'s explicit `T22` inverse carries an ARMED screen-and-refuse guard, derived from this site's own measured population | `b678747` |

---

## 1. V1 -- `pmm_jones_2d(..., slant=...)` on the JAX dispatch

### 1.1 The defect, re-measured here

`pmm_jones_2d`'s branch
`if any(is_jax_array(a) for a in _jx)` hands off to `_pmm_jones_2d_cell_jax`
**without `slant`**, and the normalization `slant = _norm_slant_pair(slant, ...)`
is not reached until AFTER the branch.  The jnp twin takes no `slant` argument
at all.  `_jx` is a SEVEN-tuple, and all seven were exercised.

Fixture (mine, not the verification's): `px = py = 0.85 um`, `wl = 0.58 um`,
`n_sup = 1.0`, `n_sub = 1.5`, a 6 x 4 x-ASYMMETRIC y-varying `(6, 4, 3, 3)` cell
(`eps` 1.30 .. 3.05), `depth = 0.42 um`, `slant = (0.5, 0)`, oblique 25,
`n_orders = 3`, `degree = 5`.

| traced input | PRE-FIX | POST-FIX |
|---|---|---|
| `eps_tensor_cell` | SOLVED SILENTLY | **NotImplementedError** |
| `n_substrate` | SOLVED SILENTLY | **NotImplementedError** |
| `n_superstrate` | SOLVED SILENTLY | **NotImplementedError** |
| `depth` | SOLVED SILENTLY | **NotImplementedError** |
| `wavelength` | SOLVED SILENTLY | **NotImplementedError** |
| `theta` | SOLVED SILENTLY | **NotImplementedError** |
| `phi` | SOLVED SILENTLY | **NotImplementedError** |

Every pre-fix row read IDENTICALLY, with **zero warnings**:

| arm | WIN | WSL |
|---|---|---|
| the traced solve vs the NumPy **VERTICAL** call | `dR 1.344e-15`, `dT 4.240e-15`, `dJones 1.497e-14` | `dR 2.331e-15`, `dT 2.953e-14`, `dJones 1.669e-14` |
| the traced solve vs the NumPy **SLANTED** call | **`dR 3.6700921434790e-03`, `dT 5.5135402311188e-02`, `dJones 1.5457253102440e-02`** | **`dR 3.6700921434787e-03`, `dT 5.5135402311190e-02`, `dJones 1.5457253102438e-02`** |
| the two NumPy calls against each other | `dR 3.6700921434803e-03`, `dT 5.5135402311189e-02`, `dJones 1.5457253102443e-02` | as above |

The last two rows agreeing to twelve significant figures is the proof that the
JAX arm WAS the vertical answer, not merely close to it -- and the two builds
agree with each other to twelve figures as well, so the defect was not a
per-build artefact.  Unlike the 2-D hybrid's frame anchor this is NOT a
unimodular phase: `R` and `T` are wrong here too, so nothing protects it.

### 1.2 What shipped

The refusal uses the same decision the frame anchor uses -- "does this cell get
SOLVED IN A SHEARED FRAME" -- factored as
`_slanted_cell_is_a_frame_noop(eps_tensor_cell)`, which mirrors
`stack2d._layer_enters_slant_frame`'s own uniform-tile test
(`max|tile - tile.flat[0]| < 1e-12`) so the two entry points can never disagree
about which slanted cells are genuine no-ops.  A TRACED cell answers `False`
(it cannot be inspected -- that is data-dependent control flow), which is the
same "be loud, not silently anchor-free" policy `_layer_enters_slant_frame`
takes for its own traced tiles.

The message names the traced input(s), the twin, the measured magnitude, and
both remedies; two different routes produce two different messages, which the
gate asserts.

### 1.3 The controls -- bit-identical on both builds

sha256 over `(orders, R, T, jones)`, PRE-FIX against POST-FIX, same build:

| control | WIN | WSL |
|---|---|---|
| a VERTICAL stack on the same traced `depth` | **identical** (`R` `7bf932e277ce3d1f`) | **identical** (`R` `b75027780038a607`) |
| a CONSTANT-tile slanted cell on the same traced `depth` | **identical** (`R` `15963161d3b37b94`) | **identical** (`R` `4da7756d1472d7d0`) |
| `jax.grad(sum T)` vs central FD (`h = 1e-11`), vertical | `rel 1.5184403291e-08` pre AND post | `rel 1.5251240221e-08` pre AND post |

The constant-tile no-op is a MEASUREMENT, not an assumption: the NumPy slanted
and vertical calls on that cell agree to `dR 4.86e-17 / dT 2.08e-14 /
dJones 5.36e-16` (WIN) and `8.33e-17 / 2.36e-14 / 2.07e-16` (WSL) -- so
refusing it would have removed an answer that is currently exact, and anchoring
it (the 2-D hybrid's own lesson) would have corrupted one.

---

## 2. V2 -- the 1-D frame anchor

### 2.1 The derivation, from the 1-D shear convention

A sheared 1-D layer is solved in `u = x - z tan(phi)`, `w = z`.  In that frame
the structure is `w`-invariant, so the state the cascade carries is the FRAME
Fourier coefficient of `sum_m F_m(w) exp(i k0 alpha_m u)`.  Two consequences,
identical in shape to the 2-D rule and specialised here to the x-only shear:

1. the frame is anchored at the layer's TOP face, where `z = 0` gives `u = x`;
   so the SUPERSTRATE side -- `R`, `T`, the reflection Jones, the reflected
   per-order amplitudes -- needs nothing;
2. at the bottom of the sheared run the same plane is `u = x - W`, so against
   the substrate's own lab basis `exp(i k0 alpha_m x)`

   ```
       A_lab(m) = exp(+i k0 alpha_m W) A_frame(m),
       W = sum_j tan(phi_j) d_j   over the layers that enter a sheared frame
   ```

The SIGN comes from the 1-D convention and is then checked by measurement:
`add_sheared_grating` stores `slant_angle = arctan(shear * period / thickness)`
and lays the LAB ridge centre at `centre + shear * (zeta - 0.5)` with `zeta = 0`
at the TOP, so a POSITIVE `slant_angle` walks the structure toward `+x` with
depth and `tan(phi) d = shear * period`.

`alpha_m = kx_m / k0` is REAL for every order, propagating or evanescent, so the
factor is UNIMODULAR -- which is the structural reason the omission was silent.
Measured over the nine retained orders of the fixture below,
`max ||P_m| - 1| = 1.110e-16` on BOTH builds, and `ptp(arg P_m) = 5.6549 rad`
over NINE distinct values (a 0.30-period walk, so neither the whole-period nor
the half-period degeneracy can hide a wrong phase).

### 2.2 The SIGN, against the engine's own z-staircase

Fixture: `P = 0.80 um`, `wl = 0.55 um`, `d = 0.40 um`, `eps_ridge = 4.20`,
`eps_groove = 1.45`, `duty = 0.45`, `shear = 0.30` (walk `0.240 um` = 0.30 P),
`theta = 25 deg`, `n_sup = 1.0`, `n_sub = 1.6`, `degree = 12`, `n_orders = 7`.
Oracle: the SAME engine's z-staircase of the identical parallelogram
(`add_tapered_grating(duty_top == duty_bottom, shear=0.30, n_slices=N)`), which
is LAB-referenced by construction.  Residuals are relative Frobenius norms.

**Which 1-D path even exposes a transmitted field on a sheared stack** (both
builds): `factorization='auto'` -- no per-order amplitudes retained;
`'covariant'` -- none; `'convection'` -- **retained**.  That is the surface
under test.

| arm | `n_slices` 6 | 12 | 24 |
|---|---|---|---|
| transmission Jones, PRE-FIX (as returned) | 1.08624 | 1.08783 | 1.09364 |
| transmission Jones, POST-FIX (as returned) | **3.65707e-02** | **1.42889e-02** | **5.09869e-03** |
| ... POST-FIX `/ P` (i.e. the anchor removed) | 1.08624 | 1.08783 | 1.09364 |
| ... PRE-FIX `x conj(P)` | 1.81885 | 1.82284 | 1.82916 |
| per-order amps, PRE-FIX (as returned) | 1.47032 | 1.47150 | 1.47241 |
| per-order amps, POST-FIX (as returned) | **3.49314e-02** | **1.37293e-02** | **5.08297e-03** |
| ... PRE-FIX `x conj(P)` | 1.29252 | 1.29548 | 1.29608 |
| the staircase's OWN last step (Jones) | -- | 2.44572e-02 | 1.03789e-02 |
| the staircase's OWN last step (amps) | -- | 2.22108e-02 | 8.97283e-03 |
| the REFLECTION Jones, as returned | 1.92424e-02 | 9.30418e-03 | 4.62204e-03 |
| `dR` | 2.97158e-03 | 9.97063e-04 | 5.11155e-04 |
| `dT` | 7.32874e-03 | 4.82587e-03 | 2.09199e-03 |

Read three ways:

* **the shipped arm CONVERGES**, 3.657e-02 -> 5.099e-03 (7.17x over the
  ladder), and at `n_slices = 24` it sits **2.04x BELOW the oracle's own last
  step**;
* **the un-anchored arm STANDS STILL** -- 1.08624 -> 1.09364, a 0.68% spread
  over a 4x refinement of the oracle.  That is the decision that identifies a
  FRAME rather than an accuracy gap: a phase that were merely a discretization
  error would follow the oracle;
* **the conjugate is WORSE THAN APPLYING NOTHING**, in every row.

The POST-FIX `/ P` column reproducing the PRE-FIX `as returned` column to every
printed digit is the arithmetic identity `A_lab / P = A_frame`; it is recorded
because it also lets the pre-fix reading be recovered from a post-fix run.

**All four arms were run DIRECTLY** -- pre-fix and post-fix, WIN and WSL -- and
every one of the twelve numbers above is identical across all four to the five
figures printed (`summarize.py derive`).  The pre-fix arms imported
`C:/tmp/lum_slfix_pre`, a read-only `git archive` of `lumenairy/` at `4a987e3`;
`_lib.arm()` stamps which tree answered into every JSON and refuses any third
one.

### 2.3 WHICH layers enter a frame -- and where the 1-D rule DIFFERS from the 2-D

The 2-D hybrid does NOT anchor a uniform (or constant-tile) slanted layer,
because `_mode_key` and `_build_layer_modes` short-circuit it to
`_homogeneous_modes` BEFORE the slant is read.  The 1-D metric generator has no
such short-circuit: `_build_generator_metric` adds the convection
`tan_conv * Dopx` whenever `|tan_conv| > 1e-14`, patterned or not.  So the
question had to be MEASURED, not inherited.

Test article: a UNIFORM `eps = 2.60` layer, `d = 0.40 um`, `tan(phi) = 0.60`
(the same 0.30-period walk), oblique 25, against the VERTICAL film of the same
`eps`.  These are the SAME physical solid -- a shear of a homogeneous medium is
a pure coordinate change -- so every lab-referenced observable must agree to
machine precision.

| quantity | PRE-FIX | POST-FIX (WIN) | POST-FIX (WSL) |
|---|---|---|---|
| transmission Jones | 1.09497 | **2.87e-14** | **1.26e-14** |
| per-order transmitted amplitudes | 1.09497 | **3.68e-14** | **2.82e-14** |
| the same, `x conj(P)` | 1.83258 | 1.09497 | 1.09497 |
| REFLECTION Jones (never anchored) | 1.51e-14 | 1.51e-14 | 1.07e-14 |
| `dR` / `dT` | 1.93e-15 / 7.44e-15 | unchanged | 1.23e-15 / 3.11e-15 |

A uniform slanted 1-D layer therefore DOES enter the walk sum.  This is the one
place the 1-D and 2-D rules genuinely differ, and it is also the cheapest
ANALYTIC oracle the anchor has: an exact no-op at 1e-14, with no staircase and
no convergence argument.  It is `test_v2_a_uniform_slanted_film_is_the_vertical_film_of_the_same_eps`.

### 2.4 The reflection side carries no anchor

Third row from the bottom of S2.2: `1.92424e-02 -> 9.30418e-03 -> 4.62204e-03`
AS RETURNED, i.e. converging with no factor applied at all, and identical on
both trees.  `dR` and `dT` converge too.  That is the derivation's claim (1) --
the frame is anchored at the stack's TOP, which is where the reflected wave
leaves it -- measured rather than argued.

### 2.5 The ZEROTH-order exemption at normal incidence

At `theta = 0`, `kx0 = 0`, so `alpha_0 = 0` and `P_0 = exp(0) = 1` EXACTLY, for
any walk whatsoever.  Measured: `alpha_0 = 0.0`, `|P_0 - 1| = 0.0` (an exact
float identity, on both builds), and

| quantity, normal incidence, vs an `n_slices = 12` staircase | PRE-FIX | POST-FIX |
|---|---|---|
| `jones_transmission` | 3.15118e-02 | **3.15118e-02** (unchanged, bit for bit) |
| `per_order_amplitudes('transmission')` | 1.68861 | **1.09826e-02** |
| the same `x conj(P)` | 1.56479 | 1.68861 |

so the ZEROTH-order Jones is exempt while the 154x-wrong per-order arm is not:
every `m != 0` order carries a real `alpha_m`, and the walk takes NINE distinct
values over the retained set.

### 2.6 COMPOSITION -- the walks add, and the split is blind to the sum

`v2_compose.py`.  Two SHEARED layers at DIFFERENT shears, thicknesses, duties
and permittivities -- `d1 = 0.24 um` at shear 0.25 over `d2 = 0.16 um` at shear
0.10, total walk `0.280 um` = 0.35 P -- against a z-staircase that places the
LOWER layer at the ACCUMULATED walk, which is the statement that the frame
CONTINUES rather than restarting.  Every wrong-sum arm is CONSTRUCTED from the
shipped answer (`/ P(W)` then times whatever the arm claims), so the probe reads
the same on either tree; the table is the `degree = 12`, `n_orders = 7` arm
(`results/v2_compose_smoke.slfix.win.json`), `K` rungs per layer:

| arm | K = 8 | K = 16 | ratio to the full sum at K = 16 |
|---|---|---|---|
| the FULL sum (shipped) | **9.469e-03** | **3.694e-03** | -- |
| only layer 1's walk in the sum | 5.535e-01 | 5.542e-01 | **150x** |
| only layer 2's walk in the sum | 1.1646 | 1.1643 | **315x** |
| no sum at all | 1.3591 | 1.3590 | **368x** |
| the CONJUGATE sum | 1.6317 | 1.6335 | **442x** |
| the oracle's own K8 -> K16 step | -- | 6.065e-03 | the shipped arm is 1.64x BELOW it |

and on the zeroth-order Jones the same decision reads full `2.954e-03` against
the oracle's own `4.112e-03` step, with only-layer-1 at 129x, no-sum at 423x and
the conjugate sum at 661x.

The `degree = 8`, `n_orders = 5`, `K = 4 / 6 / 8` arm was run on BOTH builds and
BOTH arms and reads the same decision at a coarser resolution -- per-order
amplitudes at `K = 8`: full **8.933e-03**, only-layer-1 5.538e-01 (**62x**),
only-layer-2 1.1649 (**130x**), no sum 1.3595 (**152x**), conjugate sum 1.6318
(**183x**), against the oracle's own `5.533e-03` step; WIN and WSL agree to five
significant figures on every entry.

The PRE-FIX run of that same probe is the FAIL-BEFORE, and it is legible without
any extra arithmetic: because every arm is built FROM the shipped answer, the
pre-fix "full sum" column IS the frame answer, and it reads **1.3626 / 1.3601 /
1.3595** where the post-fix one reads 2.896e-02 / 1.420e-02 / 8.933e-03 -- a
factor of **152** at `K = 8`, flat where the shipped arm converges.  The
layer-split row moves the same way: `one_layer_vs_staircase` 1.3762 pre-fix
against 1.0367e-02 post-fix, with the one-layer and two-halves arms equal to the
digit on BOTH trees.

**The LAYER-SPLIT identity, and its BLINDNESS.**  In the frame the structure is
`z`-invariant, so one sheared layer of `d` at shear `s` is the same solid as TWO
halves of `d/2` at shear `s/2` (same `tan(phi)`) whose own-frame ridge centre is
the ORIGINAL top-face centre.  Measured:

| | one layer | two halves |
|---|---|---|
| against each other, per-order amplitudes | **1.078e-15** | (the same pair) |
| against the staircase, the RIGHT sum | **3.7081e-03** | **3.7081e-03** |
| against the staircase, a HALF sum | 9.1619e-01 | **9.1619e-01** |
| against the staircase, NO sum | 1.3750 | **1.3750** |

-- both arms move together under a wrong sum, to the digit.  The split identity
alone can therefore never detect a mis-summed walk; what detects it is the
lab-referenced staircase, against which the half-sum arm is 247x and the no-sum
arm 371x worse than the shipped one.

### 2.7 The CROSS-ENGINE arm

The staircase is the same engine as the thing under test, so it settles the sign
but not the CONVENTION.  `v2_cross.py` hands the SAME parallelogram to two
INDEPENDENT engines as a y-uniform 2-D cell and compares the ZEROTH-order
transmitted Jones -- which is invariant under a lateral translation of the whole
structure (order `m` picks up `exp(i(alpha_0 - alpha_m) delta)`, which is 1 at
`m = 0`), so the engines' differing `centre` conventions cannot contaminate it.

Fixture: `P = 0.80 um`, `wl = 0.55 um`, `d = 0.40 um`, `duty = 0.50`,
`shear = 0.25`, `centre = 0.625` (so the TOP-face ridge centre is 0.500, exactly
where the 2-D cells put it), oblique 25.

| reference | POST-FIX (as returned) | POST-FIX `/ P_0` (= the pre-fix reading) | POST-FIX `x conj(P_0)` | the reference's OWN step |
|---|---|---|---|---|
| `PMM2DStackPure(n_modes = 6, n_orders = 3)` | **9.0086e-03** | 9.3571e-01 (**104x**) | 9.3571e-01 | `6.06e-04` (`n_modes` 6 -> 8) |
| `PMM2DStackPure(n_modes = 8)` | **8.9961e-03** | 9.3562e-01 | 9.3562e-01 | -- |
| `PMM2DStackHybrid(n_orders = 7)` | **8.6808e-02** | 9.2303e-01 (**10.6x**) | 9.2303e-01 | `3.85e-02` (`n_orders` 7 -> 9) |
| `PMM2DStackHybrid(n_orders = 9)` | **5.4242e-02** | 9.1843e-01 (**16.9x**) | 9.1843e-01 | as above |

The PURE arm is CONVERGED -- its own `n_modes` 6 -> 8 step is `6.06e-04`, so the
`9.0e-03` agreement is 15x that step and is the two engines' discretization gap,
not a phase.  The HYBRID arm agrees to **1.41x its own `n_orders` 7 -> 9 step**,
which is the same decision at a coarser resolution: a y-uniform stripe is the
Fourier hybrid's worst case, so it is reported and not leaned on.

Both 1-D grid routes were run and read IDENTICALLY to every digit:
`layer_grids='shared'` (the general shared-grid cascade, modal site 1) and
`'per-layer'` (the general per-layer cascade, modal site 2).  WIN and WSL agree
to five significant figures on every entry (WIN pre-fix `x P_0` = WSL post-fix
`as returned` = `9.0086e-03`).

### 2.8 The CENSUS -- every transmission-side surface, and the bit-identity set

`v2_census.py` hashes ELEVEN surfaces on TWENTY-SIX `PMMStack` fixtures:

```
orders  R  T  Jrefl        solve() returns; anchor-free
Jtrans                     jones_transmission()
perT_Ex perT_Ey            per_order_amplitudes('transmission')
perR_Ex perR_Ey            per_order_amplitudes('reflection')
bridgeT bridgeR            jones_field_from_orders on each port's dict
internal absorb            internal_field / layer_absorption (retain_internal)
```

Eighteen fixtures hold NO shear anywhere -- vertical at three mounts,
`retain_internal`, `stabilize='slices'`, per-layer grids, a three-layer stack, a
two-film stack, two taper staircases, tapered ridges, an in-plane tensor, an
OUT-OF-PLANE tensor (the generalized cascade with no slant) on both grid routes,
conical, lossy -- and eight are sheared.

**27 fixtures, 303 hashes, PRE-FIX against POST-FIX, same build.**

| comparison | identical | moved | UNEXPECTED |
|---|---|---|---|
| WIN | **272** | **31** | **0** |
| WSL | **272** | **31** | **0** |

The moved SET is the same on both builds, and every member of it is a
TRANSMISSION-side surface on a fixture that holds a sheared layer:

| fixture | keys that moved |
|---|---|
| `shear_ob25` | `Jtrans`, `perT_Ex`, `perT_Ey`, `bridgeT` |
| `shear_neg_ob25` (shear -0.30) | the same four |
| `shear_perlayer_ob25` (`layer_grids='per-layer'`) | the same four |
| `shear_over_film` | the same four |
| `shear_over_pattern` | the same four |
| `two_sheared` (different shears, one NEGATIVE) | the same four |
| `uniform_slanted` (the S2.3 no-op) | the same four |
| **`shear_normal`** | **`perT_Ex`, `perT_Ey`, `bridgeT` ONLY** |

`shear_normal` is the `P_0 = 1` exemption of S2.5 appearing as a structural
fact: its zeroth-order `Jtrans` is BYTE-IDENTICAL across the arms while every
`m != 0` order moves.

`orders`, `R`, `T`, `Jrefl`, `perR_Ex`, `perR_Ey` and `bridgeR` are
byte-identical on **every one of the 27 fixtures**, sheared rows included, and
all EIGHTEEN unsheared fixtures are byte-identical on all eleven surfaces --
vertical at three mounts, `retain_internal` (so `internal_field` and
`layer_absorption` are hashed too), `stabilize='slices'`, per-layer grids, a
three-layer stack, a two-film stack, two taper staircases, tapered ridges, an
in-plane tensor, an OUT-OF-PLANE tensor on both grid routes, conical, lossy.

Two structural facts underwrite that:

* `internal_field` and `layer_absorption` are UNREACHABLE on a sheared stack --
  `solve(retain_internal=True)` raises "all-vertical in-plane stacks only" --
  so there is no internal-field surface to anchor.  Checked, not assumed.
* the COVARIANT (uniform-slant) cascade retains NO amplitudes at all
  (`jones_transmission` raises), so it is untouched; and
  `solve_vs_wavelength` clears `_modal` and returns only anchor-free
  quantities.

---

## 3. O2 -- the `T22` screen in the generalized interface

### 3.1 The population, and the threshold derived from it

`o2_census.py` instruments EVERY consumer of `_interface_smatrix_general` (the
binding is imported by name into seven modules, so each module's binding is
patched) and records, per interface, `cond(T22)`, the FREE equilibrated
`rcond(T22)` the library already computes, and the confirming equilibrated
residual.

A solve counts as **BROKEN** when it returns (or raises on) `sum R + T` above
`1.10` per incident state -- the library's own energy criterion, not the
instrument being calibrated.  **The two populations are the SAME 18 / 53 solves
on both builds.**

| population | solves / interfaces | equil. `rcond(T22)` | equil. residual | `cond(T22)` |
|---|---|---|---|---|
| **BROKEN**, WIN | 18 / 39 | 6.66e-18 .. 9.96e-02 | 7.9e-17 .. 2.55e-01 | 10.0 .. 3.56e+16 |
| **BROKEN**, WSL | 18 / 39 | 1.65e-17 .. 9.96e-02 | 7.9e-17 .. 1.13e-01 | 10.0 .. 1.58e+16 |
| **HEALTHY**, WIN | 53 / 114 | **2.971e-05** .. 1.0 | 0.0 .. **1.47e-14** | 1.0 .. 1630 |
| **HEALTHY**, WSL | 53 / 114 | **2.971e-05** .. 1.0 | 0.0 .. **1.53e-14** | 1.0 .. 1630 |

A solve is refused when ANY of its interfaces trips, so the bar must separate
each solve's WORST interface:

| statistic | WIN | WSL |
|---|---|---|
| the WORST interface of the BEST broken solve | 2.359e-16 | **7.120e-16** |
| the WORST interface of the WORST healthy solve (the hybrid at `n_orders` 11) | **2.971e-05** | **2.971e-05** |
| the gap | 11.10 decades | **10.62 decades** |
| its geometric middle | 8.37e-11 | 1.45e-10 |
| the smallest BROKEN per-solve residual | 4.182e-03 | 3.756e-03 |
| the largest HEALTHY residual | 1.469e-14 | 1.525e-14 |
| the residual's gap | 11.45 decades | 11.39 decades |

`_INV_T22_RCOND_REFUSE = 1e-10` is that middle, rounded, and it sits inside the
gap on BOTH builds: **5.15 decades above every broken reading and 5.47 below
every healthy one** (worst case over the two).  The CONFIRMING residual keeps
the existing `_INV_RESID_REFUSE = 1e-8`, which is 5.4 / 5.8 decades from each
side of its own 11.39-decade gap.

**The two-sided check.**  With the guard armed, the set of solves that RAISE is
EXACTLY the 18 that were broken -- on BOTH builds, with no healthy solve refused
and no broken one let through:

| | WIN | WSL |
|---|---|---|
| refused post-fix | **18** | **18** |
| still solving | 56 | 56 |
| `refused set == BROKEN set` | **True** | **True** |

Two findings the audit item did not have.  **The blow-up is NOT slant-specific**:
`hyb_OOP_vert_ob25_M5` -- an OUT-OF-PLANE VERTICAL hybrid layer on the same
fixture -- reads `sum R + T = 2.766e+26` and is caught by the same screen.  What
carries the defect is the GENERALIZED (4N) cascade, which slanted AND
out-of-plane layers both reach and a vertical in-plane layer never does.  And
the class extends to the SINGLE-LAYER entry: `pmm_jones_2d` contributes three of
the eighteen.

### 3.2 What shipped

`_guarded_inverse` gains one keyword, `rcond_refuse=`, dormant for every caller
but `_interface_smatrix_general`.  When armed, the free screen decides whether
the confirming residual is computed at all, and the REFUSAL needs BOTH: the
equilibrated `rcond` below `_INV_T22_RCOND_REFUSE` AND the equilibrated residual
above the existing `_INV_RESID_REFUSE`.  Anything not refused is returned
unchanged, bit for bit -- this adds a raise, never a different number -- and the
JAX / CuPy backends keep the historical arithmetic.

The M1 withdrawal of the GLOBAL inverse refusal STANDS: there is still no bar
that works across every explicit inverse in the library (the 2-D hybrid's
healthy `_interface_smatrix` reads sit inside the 1-D broken band on that
instrument).  What is new is that this ONE site's own population was measured
and is decidable.

`_ConditioningError` subclasses `_EnergyError`, so every existing `stabilize=`
retry ladder steps `n_orders` around the refusal unchanged.

### 3.3 The FALSE-ALARM rows, which must NOT be refused

The verification's S6.3 names them as the reason the 1e+30 rows were invisible:
`PMMStack.solve`'s energy tripwire warns at `sum R + T = 1.03 .. 1.08` on
ordinary low-`n_orders` truncations, in text identical to the warning a 1e+30
raises.  This census carries **22** such rows (more than the verification's 17,
because it adds out-of-plane vertical fixtures), the SAME 22 on both builds:

| | WIN | WSL |
|---|---|---|
| benign warned rows | 22 | 22 |
| their `sum R + T` | 1.0130 .. 1.0726 | 1.0130 .. 1.0726 |
| their worst `rcond(T22)` | 9.346e-05 .. 3.752e-03 | 9.346e-05 .. 3.752e-03 |
| their worst residual | <= 1.469e-14 | <= 1.525e-14 |

so the closest any of them comes to the `1e-10` bar is **six decades**, and to
the `1e-8` residual bar **six decades** the other way.  None is refused, and the
gate pins three of them (`test_o2_the_benign_energy_warning_rows_are_not_refused`)
as still SOLVING and still WARNING.

### 3.4 Bit-identity of every healthy generalized-cascade fixture

`o2_identity.py` re-runs the same fixture set and hashes `(orders, R, T, jones)`
on each.  A screen that only ADDS a raise must leave every row it does not
refuse byte-for-byte where it was:

| | WIN | WSL |
|---|---|---|
| fixtures that produced a solve PRE-FIX | 66 | 66 |
| ... of which the guard now refuses | 12 | 12 |
| ... compared PRE-FIX vs POST-FIX | 54 | 54 |
| **hashes identical** | **216** | **216** |
| **hashes moved** | **0** | **0** |

(The census's 18 refusals against this run's 12 differ because six of the
blown-up fixtures already raised `_EnergyError` on the PRE-FIX tree, so they
produced no hash to compare.)

Zero moved over 216 hashes on both builds is the bit-identity claim, measured
rather than argued from the source.  It is corroborated structurally by
`test_o2_the_guard_is_site_scoped_and_default_off`, which shows an UNARMED
`_guarded_inverse` returning `np.linalg.inv` sha-identically on a matrix the
ARMED one refuses.

### 3.5 The two neighbouring paths

**The PURE engine's `_modes_as_general` route.**  `_modes_as_general` packs a
six-tuple; it takes no inverse.  The pure engine's conforming-grid interface
calls the SAME `_interface_smatrix_general` (`stack2d_pure.py:1423-1426` and the
`same` bypass at 1782), so it is COVERED by the armed guard rather than needing
a second one -- and the census's own readings say it is genuinely fine rather
than merely unmeasured.

**The MORTAR interfaces.**  `_interface_smatrix_general_mortar` and
`_interface_smatrix_general_mortar_2d` do NOT take an explicit inverse at all:
both end in `np.linalg.solve(A, B)`, so the `_guarded_inverse` shape does not
exist there.  Measured rather than read off the source --

`o2_identity.mortar_conditioning()` patches both mortar functions, spies on the
`np.linalg.solve(A, B)` inside them, and records `cond(A)`:

| fixture | reached the 2-D general mortar | `sum R + T` | `cond(A)` (WIN) | `cond(A)` (WSL) |
|---|---|---|---|---|
| `pure_mortar2d_M3_g1.0` (the BLOW-UP cell, ground 1.0) | yes, 936 x 936 | 0.999704 | **3.1400e+03** | 3.1400e+03 |
| `pure_mortar2d_M3_g2.1` (benign ground) | yes | 1.00015 | 1.2112e+04 | 1.2112e+04 |
| `pure_mortar2d_M5_g1.0` | yes | 0.999704 | 3.1400e+03 | 3.1400e+03 |
| `pure_mortar2d_M5_g2.1` | yes | 1.00015 | 1.2112e+04 | 1.2112e+04 |
| `pure_mortar2d_M3_nm6_g1.0` | yes | 1.0 | 1.2282e+04 | 1.2282e+04 |
| six 1-D `layer_grids='per-layer'` sheared stacks | **no** | 1.0 | -- | -- |

So on the blow-up fixture CLASS the mortar's own solve conditions at
`3.14e+03 .. 1.23e+04` -- twelve decades below the singular `T22` readings the
refusal exists for -- and it is a `solve`, not an explicit inverse, so there is
nothing for `_guarded_inverse` to screen.  The two builds agree to twelve
significant figures.

The six 1-D rows are recorded as a NEGATIVE result rather than dropped: on
`layer_grids='per-layer'` the window grids of every 1-D fixture written here
CONFORM, so `_ifc_g` takes the square `_interface_smatrix_general` bypass, which
IS guarded.  The 1-D general mortar was therefore not exercised (S7 item 6).

---

## 4. TESTS

`tests/unit/test_fix_slant_anchor_v1_v2_o2.py` -- **22 tests**.  Every numeric
bar is a DECISION with measured room on both sides, and every constant quoted in
a docstring was re-measured on both builds on 2026-09-11.

| # | test | what it decides |
|---|---|---|
| 1 | `v1_all_seven_traced_routes_refuse_a_slanted_patterned_cell` | all SEVEN dispatch members raise; the fixture's own slanted-vs-vertical gap is re-derived first so the test cannot go vacuous |
| 2 | `v1_the_refusal_names_the_traced_input_and_the_remedy` | two routes produce two DIFFERENT messages, each naming its own input, the NumPy alternative and the z-staircase |
| 3 | `v1_the_vertical_control_still_solves_and_matches_numpy` | the refusal is about the SHEAR, not the API |
| 4 | `v1_a_constant_tile_slanted_cell_is_a_measured_no_op_and_still_solves` | the no-op is MEASURED (`< 1e-11`) and the traced call is let through; `_slanted_cell_is_a_frame_noop` is two-sided |
| 5 | `v1_the_vertical_traced_solve_still_differentiates` | AD vs central FD, bar derived from the FD's OWN floor (`eps * abs(f) / h ~ 1e-6`), not from the reading |
| 6 | `v2_the_walk_helpers_decide_by_the_generator_not_by_the_keyword` | the routing literal; the exact `0.0` on a vertical stack; the sum; opposite slants cancelling exactly |
| 7 | `v2_the_anchor_is_the_identity_at_zero_walk_and_is_unimodular` | at `walk = 0.0` the SAME array objects come back (bytes that are never touched cannot move); `abs(P) - 1 < 1e-14`; `P_0 == 1+0j` exactly at `alpha = 0` |
| 8 | `v2_a_uniform_slanted_film_is_the_vertical_film_of_the_same_eps` | THE ANALYTIC ORACLE (`< 1e-11`), two-sided (removing the anchor breaks it by `> 1e6`), plus the reflection side |
| 9 | `v2_the_transmission_converges_to_the_lab_referenced_staircase` | converging (first/last `> 3x`) AND inside the oracle's own resolution (`< 2x` its last step) |
| 10 | `v2_the_unanchored_and_conjugate_arms_do_not_converge` | flat inside `(0.8, 1.25)`; `none/shipped > 20`; `conj/shipped > 20`; `conj/none > 1.3` |
| 11 | `v2_the_reflection_side_carries_no_anchor` | the reflection converges AS RETURNED |
| 12 | `v2_the_zeroth_order_is_exempt_at_normal_incidence` | `P_0 == 1` exactly, the Jones is sha-identical under it, the per-order arm is not (`> 20x`) |
| 13 | `v2_the_walks_add_over_two_sheared_layers` | four wrong-sum arms each `> 10x`, the full sum inside the oracle's own step |
| 14 | `v2_the_layer_split_is_exact_and_blind_to_a_wrong_sum` | the identity (`< 1e-12`) AND the blindness (both arms read the same wrong number under a half sum) |
| 15 | `v2_the_cross_engine_arm_agrees_with_the_independent_pure_engine` | the no-floor PURE engine, `none/shipped > 20`, `conj/shipped > 20` |
| 16 | `o2_the_generalized_cascade_blow_up_is_refused` | the slanted AND the OUT-OF-PLANE row, so the defect is not read as slant-specific |
| 17 | `o2_the_refusal_names_both_instruments_and_the_measured_remedies` | message content, and that it does NOT repeat the detune advice |
| 18 | `o2_the_measured_cures_still_solve` | every clause of the remedy line, re-measured |
| 19 | `o2_the_benign_energy_warning_rows_are_not_refused` | three benign rows still SOLVE and still WARN |
| 20 | `o2_the_threshold_sits_in_the_measured_gap_on_both_sides` | the census is armed and the two populations are re-derived: broken `rcond < 1e-15` and `resid > 1e-3`, healthy `rcond > 1e-5`, and each side is `> 1e4` from the bar |
| 21 | `o2_the_guard_is_site_scoped_and_default_off` | an UNARMED `_guarded_inverse` on a matrix that WOULD be refused returns `np.linalg.inv` sha-identically and does not raise |
| 22 | `o2_the_mortar_interfaces_take_no_explicit_inverse` | both mortar functions end in `np.linalg.solve`, neither calls `_guarded_inverse` |


## 5. SUITE RUNS AND LINT

### 5.1 Timing and the budget

The gate file ALONE, `OMP` = `OPENBLAS` = `MKL` = 1, `-p no:randomly`, on an
otherwise idle box:

| | WIN | WSL |
|---|---|---|
| the 22-test file | **66.96 s** | **69.52 s** |
| slowest test (`o2_the_measured_cures_still_solve`, whose `n_orders = 9` cure is a 4N generator solve) | **15.46 s** | **16.17 s** |
| second / third | 9.75 s / 9.52 s | 11.28 s / 10.09 s |
| headroom against the 40 s per-test cap | **2.59x** | **2.47x** |
| headroom against the 4-minute file budget | **3.58x** | **3.45x** |

`.test_durations` was SPLICED, not rewritten: `splice_durations.py` merges the
22 measured ids into the existing 12 575, giving 12 597 (`git diff --stat` reads
23 insertions, 1 deletion).  pytest-split's own `--store-durations` would have
replaced the file with only this run's tests.

### 5.2 The named files

`grep -rl --include='*.py' PMMStack tests/` (42 files) plus
`grep -rl --include='*.py' _interface_smatrix_general tests/` (6) plus the seven
the brief names, deduplicated: **53 files** (`results/suite_files.txt`).

**WIN, `OMP` = `OPENBLAS` = `MKL` = 1, `-p no:randomly`: 1236 passed, 1 skipped,
0 failed, in 3740.05 s (1:02:20).**  The one skip is pre-existing
(`test_niche_audit_m4_m5_m6_rcwa.py:387`, "threadpoolctl installed: the cap is
effective here").  102 warnings, all pre-existing energy-tripwire and
deprecation notices raised on purpose by their own tests.

The four slowest tests are `test_audit_dynameta_consumer_api_2.py`'s
(555 / 525 / 321 / 244 s), all pre-existing and unrelated; the run's next
slowest is 68 s.

`ruff check lumenairy/ tests/` -- **All checks passed!**
`ruff check validation/probe_fix_slant_anchor_v1v2o2/` -- **All checks passed!**
(`validation/` is in `pyproject.toml`'s `extend-exclude`, so the probe directory
is out of the repository lint's scope by configuration; it is linted here
anyway.)

**WSL, `OMP` = `OPENBLAS` = 1: 1236 passed, 1 FAILED, in 3329.96 s (0:55:29)** --
the same 1236 passes, and ONE failure that is **PRE-EXISTING, per-build, and not
this branch's**, established by measurement rather than by argument.

### 5.3 The WSL failure, attributed

`tests/unit/test_v5_14_2_backlog_batch.py::test_jones_2d_even_sector_matches_full`
compares `rcwa_jones_2d(..., symmetry=False)` against `symmetry=True` on a
rotated-director cell and asserts they agree to `1e-8`.  On WSL it reads
`1.625072e-02`.  The same run's LAPACK also prints
`** On entry to DLASCL parameter number 4 had an illegal value`.

`rcwa_jones_2d` calls NONE of the three functions this branch changed, and the
`_guarded_inverse` edit is a no-op for every caller that passes no
`rcond_refuse` while `_INV_CENSUS is None`.  But attribution by reading is
exactly what `docs/TESTING_STANDARDS.md` forbids, so the test body was lifted
verbatim into `wsl_flake_check.py` and run against BOTH TREES on BOTH BUILDS on
the same interpreters:

| build | tree | `dR` | `dJones` | vs the `1e-8` bar |
|---|---|---|---|---|
| WIN | POST-FIX (branch tip) | 2.812111e-10 | 7.097420e-10 | PASS |
| WIN | PRE-FIX (`4a987e3`) | **2.812111e-10** | **7.097420e-10** | PASS |
| WSL | POST-FIX (branch tip) | 1.625072e-02 | 4.356731e-02 | FAIL |
| WSL | PRE-FIX (`4a987e3`) | **1.625072e-02** | **4.356731e-02** | FAIL |

The branch tip and the branch point read IDENTICALLY, to every digit, on each
build.  The failure is therefore a PRE-EXISTING per-build defect in
`rcwa_jones_2d`'s even-parity fold on this OpenBLAS/LAPACK build -- eight
decades of disagreement between the folded and the full solve, with a LAPACK
argument complaint beside it -- and it is recorded as OPEN ITEM 7 rather than
touched here (`rcwa/twod*.py` is outside this brief's file list, and the fix
would be a per-build investigation of its own).

`ruff check lumenairy/ tests/` on WSL as well -- **All checks passed!**

## 6. COMMITS

On `fix/slant-anchor-v1-v2-o2`, off `wave2/pmm2d` @ `4a987e3`.  Nothing merged,
pushed, tagged or version-bumped.

| commit | what |
|---|---|
| `c8cb563` | **V1** -- `fix(pmm2d)`: `pmm_jones_2d`'s JAX dispatch never read `slant` and silently returned the VERTICAL answer (`twod_jones.py`, `_jax_twod_jones.py`) |
| `006c031` | **V2** -- `fix(pmm 1-D)`: `PMMStack`'s transmitted amplitudes were FRAME-referenced on a sheared stack (`pmm/stack.py`) |
| `b678747` | **O2** -- `fix(rcwa cascade)`: arm the dormant conditioning screen on the GENERALIZED interface's `T22` (`rcwa/_core.py`) |
| (this one) | `test` + `docs`: the gate, the probes with every run's JSON on both arms and both builds, this document, and the CHANGELOG entries |

Files touched in `lumenairy/`: `elements/pmm/twod_jones.py`,
`elements/pmm/_jax_twod_jones.py`, `elements/pmm/stack.py`,
`elements/rcwa/_core.py`.  Nothing in `pmm/stack.py`'s sliver-guard code, the
lens files, `twod_staggered.py`, `stack2d_pure.py` or the new mortar functions
in `pmm/_core.py` was edited -- those are other agents' worktrees.


## 7. WHAT IS OUT OF SCOPE, AND WHAT IS STILL OPEN

1. **The 2-D hybrid's own anchor is not re-derived here.**  It shipped at
   `0ebd632` and was independently verified; V2 is its 1-D twin and reuses its
   rule, with the one measured DIFFERENCE recorded in S2.3 (uniform layers).
2. **`pmm_jones_1d_slanted` still exposes no transmitted field.**  It returns
   `(orders, R, T, jones_REFLECTION)`, all anchor-free, so it neither carries
   the defect nor can arbitrate it.  Unchanged.
3. **A slant on the 2-D JAX surface is REFUSED, not implemented.**  Making
   `_pmm_jones_2d_cell_jax` carry the shear is a build task; the module
   docstring now says so, and says the refusal upstream is what keeps the two in
   step.
4. **O2's mechanism is not traced below `cond(T22)`.**  The verification
   established the proximate cause and refuted two attributions; this fix
   screens it.  *Why* the truncated sheared (or out-of-plane) mode set becomes
   rank-deficient against the Rayleigh half-space set above a ~19-degree tilt
   remains a question about the generator's own conditioning.
5. **Cross-build coverage is WIN + WSL only.**  Both are OpenBLAS builds; a
   genuinely different BLAS was not available here.
6. **The 1-D general MORTAR interface was not reached** by any fixture written
   here: on `layer_grids='per-layer'` the window grids of the fixtures tried
   conform, so `_ifc_g` takes the square `_interface_smatrix_general` bypass --
   which IS guarded.  The 2-D general mortar was reached and measured (S3.5).
7. **OPEN, and NOT this branch's.**
   `tests/unit/test_v5_14_2_backlog_batch.py::test_jones_2d_even_sector_matches_full`
   FAILS on WSL and passes on WIN, reading `dR 1.625072e-02` against a `1e-8`
   bar, with a LAPACK `** On entry to DLASCL parameter number 4 had an illegal
   value` beside it in the same run.  S5.3 measures the branch tip and the
   branch point `4a987e3` reading IDENTICALLY to every digit on each build, so
   it is a PRE-EXISTING per-build defect in `rcwa_jones_2d`'s even-parity fold
   (or in what its `symmetry=True` path asks of this LAPACK).  `rcwa/twod*.py`
   is outside this brief's file list, and the assertion is exactly the `S5`
   shape `docs/TESTING_STANDARDS.md` warns about -- a fold-vs-full comparison
   pinned at `1e-8` on a build-dependent eigenbasis -- so it is REPORTED here,
   not touched.  Reproducer:
   `validation/probe_fix_slant_anchor_v1v2o2/wsl_flake_check.py`.
