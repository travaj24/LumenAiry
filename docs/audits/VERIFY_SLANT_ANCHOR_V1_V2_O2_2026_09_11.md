# VERIFY -- an independent re-measurement of the V1 / V2 / O2 slant-anchor fix

**Date:** 2026-09-11 - **Worktree:** `C:/tmp/lum_vslant`, branch
`verify/slant-anchor-v1-v2-o2` at the `wave2/pmm2d` tip `2ec4359`.
**Under verification:** `docs/audits/FIX_SLANT_ANCHOR_V1_V2_O2_2026_09_11.md`
-- items **V1** (`pmm_jones_2d`'s JAX dispatch), **V2** (the 1-D `PMMStack`
transmitted-amplitude frame anchor) and **O2** (the `T22` conditioning screen).
**Binding law:** `docs/TESTING_STANDARDS.md`.
**Probes:** `validation/probe_verify_slant_anchor/` -- five scripts plus their
own `_lib.arm()`, which decides the TREE from `lumenairy.__file__` alone,
refuses any tree but the three named in S1, and stamps tree, interpreter,
numpy, scipy, jax and thread caps into every JSON.
**New gate:** `tests/unit/test_verify_slant_anchor_v1_v2_o2.py` (8 tests).

Nothing in `lumenairy/` was edited by this verification.

---

## 0. TERMS, BEFORE ANY NUMBER

**Frame.** A sheared 1-D layer of wall tilt `phi` is solved in the coordinate
`u = x - z tan(phi)`, in which the structure is z-invariant.  The quantities
the cascade carries inside such a layer are FRAME Fourier coefficients.

**Walk.** `W = sum_j tan(phi_j) d_j` over the layers that enter a sheared
frame -- the total lateral offset between the frame at the bottom of the
sheared run and the lab.

**Anchor.** `A_lab(m) = P_m A_frame(m)` with `P_m = exp(+i k0 alpha_m W)` --
the re-referencing V2 shipped, applied to the TRANSMITTED per-order
amplitudes only.

**alpha_m.** `kx_m / k0`, the order's dimensionless x-wavevector.  Real for
every order (see S4.8), which is why `P_m` is unimodular and why the omission
moved no efficiency.

**Broken / refused (O2).** A solve is BROKEN when, on a tree WITHOUT the
guard, it returns (or raises the library's own `_EnergyError` on)
`sum R + T > 1.10` per incident state.  It is REFUSED when, on the tree WITH
the guard, it raises `_ConditioningError`.  The two are measured by different
instruments on purpose: the first is the library's energy criterion, the
second is the screen being calibrated.

**Bit-identity.** sha256 over the raw C-contiguous bytes of an array, compared
between two TREES on the SAME build.  A digest that moves is a byte that
moved; no tolerance is involved.

---

## 1. ARMS -- three trees, two builds

| | WIN | WSL |
|---|---|---|
| interpreter | CPython 3.14.6 | CPython 3.12.3 |
| numpy / scipy | 2.4.4 / 1.17.1 | 2.4.6 / 1.17.1 |
| jax | 0.11.0 (x64 on) | 0.10.2 (x64 on) |
| BLAS | scipy-openblas (Haswell kernels) | scipy-openblas (SkylakeX kernels) |
| threads | `OMP` = `OPENBLAS` = `MKL` = 1 | `OMP` = `OPENBLAS` = 1 |

| tree | commit | what it is |
|---|---|---|
| `post` `C:/tmp/lum_vslant` | `2ec4359` | the wave2 tip -- HOLDS the fix |
| `pre` `C:/tmp/lum_vslant_pre` | `4a987e3` | the fix's BRANCH POINT |
| `rev` `C:/tmp/lum_vslant_rev` | `2ec4359` minus `c8cb563`, `006c031`, `b678747` | the SURGICAL isolation |

**Why three and not two.**  `wave2/pmm2d` moved between the fix's branch point
and the tip for reasons that are not this fix: the sliver round-2 and mortar
round-2 merges both edited `pmm/stack.py` (532 insertions / 52 deletions
between the fix branch's tip and `HEAD` in that file alone).  `pre` therefore
cannot attribute a moved byte to this fix.  `rev` is `HEAD` with ONLY the
three fix commits reverted -- `rcwa/_core.py`, `twod_jones.py` and
`_jax_twod_jones.py` restored from `4a987e3` (nothing else touched them) and
`006c031` reverted from `stack.py` (a clean auto-merge).  Every identity claim
below was checked against BOTH `pre` and `rev`, on BOTH builds, and the two
comparisons agree exactly.

---

## 2. VERDICTS

| item | claim | verdict | this verification's numbers |
|---|---|---|---|
| **V1** | all seven traced routes SOLVED SILENTLY with the vertical answer pre-fix | **CONFIRMED** | own fixture, both builds: all seven identical to the NumPy VERTICAL call at `dR 1.34e-13`, and wrong against the correct SLANTED one by `dR 2.880903e-02`, with ZERO warnings; all seven sha-identical to each other |
| **V1** | all seven REFUSE post-fix, naming the traced input | **CONFIRMED** | seven `NotImplementedError`s on both builds, each naming its own input, the NumPy alternative and the z-staircase |
| **V1** | the constant-tile slanted cell is a MEASURED no-op | **CONFIRMED and WIDENED** | `dR <= 1.97e-14` over five mounts including CONICAL and an ANISOTROPIC constant tile, and at a large `slant = (1.7, 0.9)`, not only the fix's one oblique isotropic row |
| **V1** | no eighth route | **CONFIRMED** (22 candidates) | jit / vmap / grad / y-only slant all route through the same tuple and are refused; a traced `slant` raises `ConcretizationTypeError`; the pure staggered engine has no jnp twin and raises under jit; `PMM2DStackHybrid` was already closed at `4a987e3` |
| **V2** | the SIGN, against a z-staircase | **CONFIRMED** on an independent hand-built oracle | shipped `4.52e-02 -> 1.54e-02 -> 6.30e-03` at 4/8/16 rungs (7.2x) against a FLAT `1.1427 -> 1.1406` un-anchored arm; shipped sits **1.54x below the oracle's own last step** |
| **V2** | the analytic uniform-film oracle | **CONFIRMED** | at `tan(phi)` = +0.60 / -0.60 / +1.35: `3.69e-14` / `3.68e-14` / `3.42e-14` (WIN) and `3.41e-14` / `4.10e-14` / `3.67e-14` (WSL); un-anchored `1.186` / `1.186` / `1.980` on both |
| **V2** | the CROSS-ENGINE arm | **CONFIRMED on TWO independent engines** | vs `PMM2DStackPure` at `n_modes = 8`: shipped `3.4536e-03`, un-anchored `0.9487` (**275x**), conjugate `1.6691` (**483x**), against that engine's own `n_modes` 6 -> 8 step of `1.5565e-03`; vs `RCWAStack` at 32 rungs: shipped `1.635e-02`, un-anchored `0.9572` (**58.6x**), conjugate `1.6745` (**102x**).  Both builds agree to seven figures |
| **V2** | negative shear, opposite shears, net-zero walk | **CONFIRMED** | the negative-shear ladder converges `5.58e-02 -> 6.13e-03`; the net-zero stack's walk is EXACTLY `0.0` and the anchor returns the same array objects, while one layer's own walk moves the amplitudes by `> 0.5` |
| **V2** | the walks ADD over two sheared layers | **CONFIRMED, with a restatement** | `1.827e-02 -> 8.037e-03` against the frame-continued oracle, every wrong-sum arm `101x .. 193x` worse, and the full sum sits 1.33x BELOW the oracle's own step -- but ONLY when the second layer is placed at the ACCUMULATED walk; see **DEFECT D1** |
| **V2** | conical incidence is unreachable | **CONFIRMED** | all six (factorization x grid) combinations raise |
| **V2** | `retain_internal` / `internal_field` unreachable on a sheared stack | **CONFIRMED** | all four combinations plus a MIXED vertical+sheared stack raise |
| **V2** | `solve_vs_wavelength` carries nothing | **RESTATED (stronger)** | it does not "clear `_modal`" -- it REFUSES a sheared stack outright |
| **V2** | the two anchored sites are the ONLY ones a sheared stack reaches | **CONFIRMED** | all FIVE `self._modal = modal` sites enumerated and each exercised: the two conical ones refuse the shear, the vertical per-layer one is unreachable by the routing literal, and `prepare().solve()` raises "all-vertical stacks only" (S4.8) |
| **V2** | `P_m` unimodular for evanescent orders too | **CONFIRMED** | `max\|Im(alpha_m)\| = 0.0` EXACTLY and `max\|\|P_m\|-1\| = 1.11e-16` across a Rayleigh cut-off with 4-6 evanescent orders, and under a lossy superstrate |
| **V2** | the census: transmission-side movers only | **CONFIRMED on 46 + 9 own fixtures** | 585 identical / 55 moved / **0 unexpected**, the same set on both builds and against both comparison trees |
| **O2** | the bar separates the populations | **CONFIRMED on an independent population** | 134 solves / 229 interfaces; refused `rcond` `4.7e-18 .. 4.8e-16`, healthy `rcond >= 2.08e-05`; `1e-10` is **5.3 decades above** every refused reading and **5.3 below** every healthy one |
| **O2** | refused set == broken set | **CONFIRMED** | see S6.3 |
| **O2** | bit-identity of every healthy solve | **CONFIRMED** | see S6.4 |
| **O2** | the benign 1.01..1.07 warning rows are untouched | **CONFIRMED** | 1.0350 / 1.0726, both still WARNING |
| **O2** | "the free screen ... not one extra flop" | **BOUNDED** | true for unarmed callers (the early return is unchanged); the ARMED site pays `1.10x .. 1.51x` of the inverse itself at `n = 66 .. 722`, falling with `n` -- see **NOTE N1** |

**Defects found (none of them silent-wrong in the shipped answer):**

| id | severity | what |
|---|---|---|
| **D1** | **P2** | `add_sheared_grating`'s `centre` is documented as a LAB position and is interchangeable with the z-staircase -- true for ONE sheared layer, FALSE for the second and later ones, which the cascade places at `+ W1`.  PRE-EXISTING (not introduced by V2), but V2's composition claim rests on it and no public docstring states it. |
| **D2** | **P3** | On the library DEFAULT (`factorization='auto'`) a single in-plane sheared grating routes to the COVARIANT cascade, which retains no amplitudes -- so the anchored surface does not exist at all.  V2 is reachable only via `'convection'` or a mixed-slant / out-of-plane stack.  Recorded in the fix's S2.2 as a table row; not pinned by any test until now. |
| **D3** | **P4** | A CONCRETE (non-tracer) `jnp` cell that is constant-valued is refused although it is inspectable and a measured no-op, because `is_jax_array` cannot distinguish it from a tracer.  Loud, with a named remedy; a scope boundary rather than a wrong answer. |
| **N1** | note | the armed `T22` screen's cost, measured. |
| **N2** | note | `ptp(arg P_m)` as a diagnostic is a WRAPPED angular range and cannot exceed `2 pi`; the fix's "5.6549 rad over NINE distinct values" is a wrapped reading, not evidence of the walk's size. |

**Ship recommendation: SHIP for 5.45.0.**  All three items do what they claim,
on this verification's own fixtures, on both builds, against three trees.  D1
is a docstring correction on a pre-existing cascade property and D2/D3 are
scope statements; none of them blocks.

---

## 3. TASK 1 -- BIT-IDENTITY, on 46 + 9 fixtures of this verification's own

`t1_identity.py` + `fixtures.py`.  Forty-six `PMMStack` mounts and nine 2-D /
single-layer entries, hashed on up to thirteen SURFACES each:

```
orders R T Jrefl          solve() returns; anchor-free by the derivation
Jtrans                    jones_transmission()
perT_Ex perT_Ey           per_order_amplitudes('transmission')
perR_Ex perR_Ey           per_order_amplitudes('reflection')
bridgeT bridgeR           polarization.jones_field_from_orders on each port
internal absorb           internal_field / layer_absorption (retain_internal)
```

The vertical members: binary gratings at normal / 25 / 40 degrees, a two-layer
stack, a uniform film, an IN-PLANE tensor layer, an OUT-OF-PLANE tensor layer
on BOTH grid routes, plain per-layer grids, `retain_internal`, a lossy stack
with `retain_internal`, conical incidence, `stabilize='slices'`, a taper
staircase, tapered ridges, and a three-layer Bragg stack.
The sheared members: POSITIVE and NEGATIVE shear at two mounts and at normal
incidence, per-layer grids, a large shear, a lossy shear, two sheared layers
of the SAME sign, two of OPPOSITE sign, a NET-ZERO walk pair, a sheared layer
ABOVE and BELOW a vertical patterned one, a UNIFORM slanted film, a sheared
layer stacked with an out-of-plane one, the covariant route, the two refusal
mounts, a slant BELOW the `1e-12` routing literal, and a near-Wood wavelength
-- each of the single-slant members also in a `factorization='convection'`
twin, because `'auto'` does not reach the anchored cascade (D2).

**Result, `post` against BOTH `pre` and `rev`, on BOTH builds -- four
comparisons, all four identical:**

| | identical | moved | **UNEXPECTED** |
|---|---|---|---|
| WIN, post vs pre | 585 | 55 | **0** |
| WIN, post vs rev | 585 | 55 | **0** |
| WSL, post vs pre | 585 | 55 | **0** |
| WSL, post vs rev | 585 | 55 | **0** |
| **`pre` vs `rev`** (the two comparison trees against each other), WIN | **640** | **0** | -- |
| **`pre` vs `rev`**, WSL | **640** | **0** | -- |

The last two rows are worth their own line: the OTHER work that landed on
`wave2/pmm2d` between the fix's branch point and the tip -- the sliver and
mortar round-2 merges, the lens follow-ups -- moves NOT ONE of these 640
digests, so the two comparison trees are interchangeable here and every
moved byte below is attributable to the three fix commits alone.

(640 comparable digests per build: 46 one-dimensional fixtures x up to 13
surfaces plus the 2-D entries.)  Every one of the 55 moved DIGESTS is a
TRANSMISSION-side surface on a SHEARED fixture -- 14 fixtures in all -- and the
moved set is the same on both builds and against both comparison trees:

| fixture | keys that moved |
|---|---|
| `shear_pos_ob25_conv`, `shear_neg_ob25_conv`, `shear_pos_ob40_conv` | `Jtrans`, `perT_Ex`, `perT_Ey`, `bridgeT` |
| `shear_big_conv`, `shear_lossy_conv`, `shear_perlayer_conv` | the same four |
| `shear_wl_near_wood_conv`, `uniform_slanted_film_conv` | the same four |
| `shear_two_same_sign`, `shear_two_opposite`, `shear_with_oop` | the same four |
| `shear_over_vertical_pattern`, `shear_under_vertical_pattern` | the same four |
| **`shear_pos_norm_conv`** | **`perT_Ex`, `perT_Ey`, `bridgeT` ONLY** |

Six structural facts fall out of the table rather than being argued:

* **`orders`, `R`, `T`, `Jrefl`, `perR_Ex`, `perR_Ey`, `bridgeR`, `internal`
  and `absorb` are byte-identical on every one of the 55 fixtures** (46
  one-dimensional + 9 two-dimensional), sheared rows included.  The anchor is a phase on one port; no efficiency and no
  energy sum can move, and none did.
* **`shear_pos_norm_conv` moves only three of its four transmission
  surfaces.**  At normal incidence `alpha_0 = 0` exactly, so `P_0 = 1` exactly
  and the ZEROTH-order `jones_transmission` is byte-identical across the
  trees while every `m != 0` order moves.  The exemption appears as a
  structural fact rather than as a tolerance.
* **`shear_net_zero_walk` does NOT move at all.**  Two layers at `+phi` and
  `-phi` of equal thickness give `W` exactly `0.0`, the anchor short-circuits,
  and not one byte of any surface changes -- the strongest form the
  cancellation claim can take.
* **`shear_pos_ob25` (the `'auto'` default) does not move either**, because
  that route retains no amplitudes at all (D2).
* **`shear_tiny_below_bar`** (`slant_angle = 5e-13`) does not move: below the
  `1e-12` literal the stack takes the all-vertical symmetric cascade, and the
  walk helper agrees with the routing (S4.6).
* the 2-D entries -- `pmm_jones_2d` vertical / slanted / conical-slanted /
  constant-tile, `PMM2DStackHybrid` vertical and slanted, `PMM2DStackPure`
  vertical and slanted -- are byte-identical on every surface.  V1 adds a
  refusal to the JAX branch and touches no NumPy arithmetic.

---

## 4. TASK 2 -- V2's SIGN, by five independent means

`t2_v2_sign.py`.  **Fixture (mine, none of it the fix's):** `P = 0.66 um`,
`wl = 0.52 um`, `d = 0.34 um`, `eps_ridge = 3.80`, `eps_groove = 1.25`,
`duty = 0.38`, `shear = 0.24` (a 0.24-period walk -- neither the half- nor the
quarter-period degeneracy), `theta = 31 deg`, `n_sub = 1.45`, `degree = 11`,
`n_orders = 7`.  Every arm below is reconstructed FROM the tree's own returned
answer (`shipped`, `none = shipped / P`, `conj`, `half`, `double`), so the
same script is legible on a pre-fix tree, where "as returned" IS the frame
answer.

### 4.1 The oracle is built HERE, not called

`ladder()` does not use `add_tapered_grating`.  It lays each rung of the
parallelogram down with `add_layer(segments=...)` from segment widths this
probe computes (`binary_segments`, wrap-aware), so the reference is an
ALL-VERTICAL stack whose lab geometry is under the probe's own control and
which enters no frame at all.  Two controls bound that choice:

* the oracle on `layer_grids='shared'` against the same oracle on
  `'per-layer'` (used for cost): **`2.16e-05`**, i.e. 290x below the residual
  being read;
* the LIBRARY's own `add_tapered_grating` staircase at 16 rungs against the
  hand-built one: shipped reads **`6.320e-03`** vs **`6.297e-03`** -- the two
  constructions are the same solid.

### 4.2 The ladder, both signs (per-order transmitted amplitudes)

| arm | ns 4 | ns 8 | ns 16 |
|---|---|---|---|
| **shipped (POSITIVE shear)** | **4.518e-02** | **1.536e-02** | **6.297e-03** |
| un-anchored (`/ P`) | 1.1427 | 1.1407 | 1.1406 |
| conjugate | 1.5121 | 1.5132 | 1.5132 |
| half walk | 0.6627 | 0.6626 | 0.6634 |
| double walk | 1.1341 | 1.1381 | 1.1394 |
| the oracle's OWN step | -- | 3.199e-02 | 9.685e-03 |
| the REFLECTION, as returned | 1.047e-01 | 3.164e-02 | 1.047e-02 |
| **shipped (NEGATIVE shear)** | **5.582e-02** | **1.744e-02** | **6.132e-03** |

Read four ways:

* the shipped arm **CONVERGES** (7.2x positive, 9.1x negative over the
  ladder) and at 16 rungs sits **1.54x BELOW the oracle's own last step**;
* the un-anchored arm **STANDS STILL** -- 1.1427 to 1.1406, 0.18% over a 4x
  refinement.  That is the decision that identifies a FRAME rather than an
  accuracy gap;
* **every** wrong re-referencing is flat and worse: conjugate 1.51 (240x),
  half 0.66 (105x), double 1.14 (181x);
* the REFLECTION converges AS RETURNED, with no factor applied, on both
  trees -- the derivation's claim that the frame is anchored at the stack's
  TOP, measured.

On the zeroth-order Jones the same ladder reads shipped `1.019e-02 ->
3.103e-03 -> 2.618e-03` against an oracle step of `1.326e-03` (1.97x), with
un-anchored `0.947` flat (362x) and conjugate `1.667` (637x).

### 4.2b THE FAIL-BEFORE, on the guard-free tree

The same script on `rev` -- where "as returned" IS the frame answer, so the
arms simply relabel.  Per-order transmitted amplitudes against the same
hand-built staircase:

| arm on `rev` (guard-free) | ns 4 | ns 8 | ns 16 |
|---|---|---|---|
| **as returned** | **1.142709** | **1.140710** | **1.140598** |
| its `x P` arm (the probe's `double` label) | **4.518061e-02** | **1.536151e-02** | **6.297491e-03** |
| the POST tree's `shipped` arm, for comparison | 4.518061e-02 | 1.536151e-02 | 6.297491e-03 |
| the POST tree's `none` arm, for comparison | 1.142709 | 1.140710 | 1.140598 |

The two pairs agree to EVERY PRINTED DIGIT: on `rev` the returned answer IS
the frame answer, and `frame x P` reproduces the post-fix reading exactly.
The pre-fix arm is not merely less accurate -- it STANDS STILL (1.1427 ->
1.1406, 0.18%) under a 4x refinement of an oracle the post-fix arm converges
through.  The analytic uniform-film oracle reads the same way on `rev`:
**1.18598 / 1.18598 / 1.97971** as returned at `tan(phi)` = +0.60 / -0.60 /
+1.35, and `3.69e-14 / 3.68e-14 / 3.39e-14` once multiplied by `P`.

(The probe carries its own two-line copy of `_layer_enters_slant_frame_1d` and
`_slant_frame_walk_1d` for this arm -- they SHIPPED with V2 and do not exist
on the guard-free trees -- and stamps `walk_helper_source: "probe-local"` into
the JSON so the substitution is visible rather than silent.)

### 4.3 The ANALYTIC oracle -- a uniform slanted film

A UNIFORM `eps = 2.60` layer at `tan(phi)` against the VERTICAL film of the
same `eps`: the same physical solid, so every lab observable must agree to
machine precision, with no staircase and no convergence argument.

| `tan(phi)` | walk | shipped | un-anchored | conjugate |
|---|---|---|---|---|
| **+0.60** | 0.204 um | **3.688e-14** | 1.186 | 1.512 |
| **-0.60** | -0.204 um | **3.683e-14** | 1.186 | 1.512 |
| **+1.35** | 0.459 um | **3.421e-14** | 1.980 | 1.585 |

-- and the REFLECTION side reads `1.04e-13` as returned, with `dR`, `dT` and
the reflection Jones at `1e-14`.  A uniform slanted 1-D layer therefore DOES
enter the walk sum, for both signs and at a 53-degree wall tilt.

### 4.4 The CROSS-ENGINE arm, with its sign adjudicated first

The staircase is the same engine as the thing under test, so it settles the
sign but not the CONVENTION.  `PMM2DStackPure` (the staggered engine, which is
an independent discretisation with its own solve-level walk sum) is handed the
SAME parallelogram as a y-uniform 2-D cell on a per-layer grid whose walls are
the ridge edges, and the ZEROTH-order transmitted Jones is compared -- which
is invariant under a lateral translation of the whole structure, so the two
engines' `centre` conventions cannot contaminate it.

| pure-engine slant | `n_modes` | shipped | un-anchored | conjugate | half | double |
|---|---|---|---|---|---|---|
| `+t` | 4 | 1.202e-01 | 0.8710 | 1.6162 | 0.4110 | 1.0227 |
| `+t` | 6 | **3.571e-03** | 0.9477 (**265x**) | 1.6683 (**467x**) | 0.4894 | 0.9450 |
| `+t` | 8 | **3.454e-03** | 0.9487 (**275x**) | 1.6691 (**483x**) | 0.4905 | 0.9443 |
| `-t` | 6 | 2.127 | 2.428 | 2.321 | 2.324 | 1.581 |

The pure engine's OWN `n_modes` 6 -> 8 step is **`1.557e-03`**, so the
`3.45e-03` agreement is 2.2x its own resolution -- the two engines'
discretisation gap, not a phase.  The WRONG 2-D sign (`-t`) leaves NO arm
below 1.58, which is what makes the `+t` reading a sign statement rather than
a coincidence.


### 4.5 A THIRD engine -- RCWA

`t2b_rcwa_cross.py`.  The staircase is the same engine as the thing under
test and `PMM2DStackPure` is a different discretisation of the same
(spectral-element) family.  `RCWAStack` is neither: a Fourier-modal solver
with its own z-staircase, its own factorisation rules and its own far-field
bookkeeping.  The SAME parallelogram, `n_orders = 15`, `raster='area'`,
`n_x = 512`, zeroth-order transmitted Jones -- **identical on both builds to
six figures**:

| RCWA rungs | shipped | un-anchored | conjugate | half | double | RCWA's own step |
|---|---|---|---|---|---|---|
| 8 | **1.978e-02** | 0.9600 | 1.6762 | 0.5029 | 0.9333 | -- |
| 16 | **1.749e-02** | 0.9580 | 1.6750 | 0.5007 | 0.9352 | 2.561e-03 |
| 32 | **1.635e-02** | 0.9572 | 1.6745 | 0.4998 | 0.9361 | 1.200e-03 |

The shipped arm is **58.6x** closer than the un-anchored one and **102x**
closer than the conjugate, on an engine that shares no code with the 1-D PMM
cascade below `numpy`.  Its own floor (`1.6e-02`, still falling at 32 rungs)
is RCWA's Fourier truncation plus its staircase, which is why this arm settles
the DECISION and is not read as an accuracy statement.

### 4.6 NEGATIVE shear, OPPOSITE shears, and the NET-ZERO walk

* the negative-shear ladder is in S4.2 and the negative uniform film in S4.3;
* **net zero:** two layers of equal thickness at `+phi` and `-phi` give
  `_slant_frame_walk_1d(...) == 0.0` EXACTLY on both builds (`np.tan` is odd
  to the bit), the anchor short-circuits and returns the SAME array objects,
  and the whole fixture is byte-identical across the trees (S3).  Two-sided:
  ONE layer's own walk applied to those same amplitudes moves them by
  **`> 0.5`** relative, and its `P_m` spans a full wrapped `5.65 rad`;
* **opposite-sign pair with a non-zero net walk** (`+0.30 P` over `-0.12 P`,
  `W = 0.18 P`), against the hand-built two-run staircase:

| arm | 6 rungs/layer | 12 rungs/layer | ratio to full at 12 |
|---|---|---|---|
| **the FULL sum (shipped)** | **1.8273e-02** | **8.0368e-03** | -- |
| only layer 1's walk | 8.0867e-01 | 8.0943e-01 | **101x** |
| only layer 2's walk | 1.4862 | 1.4873 | **185x** |
| no sum at all | 1.1213 | 1.1206 | **139x** |
| the CONJUGATE sum | 1.5474 | 1.5494 | **193x** |
| the oracle's OWN 6 -> 12 step | -- | 1.0667e-02 | shipped is **1.33x below** it |

  (WSL; the WIN arm of this table is in `t2_v2_sign.rev.win.json`'s companion
  run and agrees to seven figures.)  The walk helper returns `1.1880e-07 m`
  where `(s1 + s2) P = 1.1880e-07 m` -- the sum, to the digit.

### 4.7 The routing literal, two-sided

| `slant_angle` | `_layer_enters_slant_frame_1d` | walk |
|---|---|---|
| 0.0 | False | 0.0 |
| 5e-13 | False | 0.0 |
| **1e-12** | **False** (the comparison is strict `>`) | 0.0 |
| 1.0000001e-12 | True | 3.4e-19 m |
| 1e-9 | True | 3.4e-16 m |

The walk helper's `1e-12` is the SAME literal `PMMStack.solve` uses to promote
the stack to the general cascade (three sites, all `abs(L[2]) > 1e-12`), so
the anchor turns on exactly where the sheared frame does.  At the first
`True` the phase is `4e-12 rad`, so the discontinuity at the literal is below
what any observable can see.

### 4.8 The break attempts

| attempt | result |
|---|---|
| conical incidence on a sheared stack, 3 factorizations x 2 grid routes | **all six raise** `NotImplementedError` ("not available for SLANTED layers") |
| the conical cascade's own modal dict, vertical stack | walk `0.0`; `max\|ky\| = 0.549` (so the conical path is genuinely 2-D and genuinely unanchored) |
| `retain_internal=True` on a sheared stack, 2 factorizations x 2 grid routes | **all four raise** ("all-vertical in-plane stacks only") |
| `retain_internal=True` on a MIXED vertical + sheared stack | **raises** -- the case that could plausibly slip past a "the sheared cascade keeps nothing" rule |
| `solve_vs_wavelength` on a sheared stack | **raises** ("all-vertical in-plane stacks only") -- stronger than the fix's claim that it "clears `_modal`" |
| Wood-anomaly sweep, 5 wavelengths across the substrate's `m = -2` cut-off | `max\|Im(alpha_m)\| = 0.0` EXACTLY at every point; `max\|\|P_m\|-1\| = 1.11e-16`; 4 to 6 EVANESCENT orders present at each point |
| lossy superstrate `n_sup = 1 + 0.02j` and `1 + 0.4j` | both SOLVE; `alpha_m` still exactly real (`kx0` is built from `Re(n_sup)`), `P_m` still unimodular to `1.11e-16` |
| a sheared layer ABOVE and BELOW a vertical patterned layer | both give walk `= shear * P` exactly; both are transmission-only movers in S3 |
| a slanted layer under a vertical patterned layer, per-layer grids | same |

### 4.9 EVERY site that can produce a frame-referenced amplitude

The anchor is only complete if the fix reached every `self._modal = modal`
site a sheared stack can arrive at.  There are FIVE in `pmm/stack.py`, and
each was checked by behaviour rather than by reading:

| site | cascade | can a SHEARED stack reach it? | anchored? |
|---|---|---|---|
| `stack.py:1883` | conical, nodal patterned | **no** -- `solve` raises "conical incidence (phi != 0) is not available for SLANTED layers" (measured on all six factorization x grid combinations) | n/a |
| `stack.py:2051` | conical, uniform Fourier | **no**, same refusal | n/a |
| `stack.py:2517` | general fwd/back, SHARED grid | yes | **YES** |
| `stack.py:2730` | all-vertical in-plane, PER-LAYER grids | **no** -- `solve` routes any `abs(slant_angle) > 1e-12` to `_solve_general_perlayer` instead (measured: the routing literal, S4.6) | walk is `0.0` by construction |
| `stack.py:2869` | general fwd/back, PER-LAYER grids | yes | **YES** |

and the three OTHER entries that could plausibly produce one:

| entry | on a sheared stack |
|---|---|
| `PMMStack.prepare().solve(...)` (the assemble-once material sweep) | **raises** `NotImplementedError: PMMStack.prepare: all-vertical stacks only.` -- measured, not read |
| `PMMStack.solve_vs_wavelength(...)` | **raises** "all-vertical in-plane stacks only" |
| `_solve_covariant` (the spectral uniform-slant cascade) | retains NO amplitudes at all: `jones_transmission()` and `per_order_amplitudes()` both raise (this is D2) |

So the two sites the fix anchored are the only two a sheared stack can reach,
and the claim is closed rather than merely unfalsified.

### 4.10 DEFECT D1 -- where the SECOND sheared layer actually sits

**The finding.**  The cascade matches successive sheared layers' FRAME
coefficients directly (`_interface_smatrix_general(Mls[i], Mls[i+1])`, with no
re-referencing), so layer 2 is solved in layer 1's frame `u = x - W1`.  The
`centre` handed to `add_sheared_grating` for the second sheared layer
therefore describes a ridge that stands at **`centre + W1`** in the LAB.

`add_sheared_grating`'s own docstring states the opposite: "the ridge centre
is `centre + shear * (zeta - 0.5)` in period fractions, so this and the
staircase describe the SAME structure and the two are interchangeable at the
call site."  That is true for ONE sheared layer and false for the second.

**Measured**, per-order transmitted amplitudes of a two-sheared-layer stack
against this verification's hand-built staircase:

| where the oracle puts layer 2 | 6 rungs | 12 rungs |
|---|---|---|
| at `c_top2 + W1` (**what the cascade models**) | **1.827e-02** | **8.037e-03** |
| at `c_top2` (**what the docstring says**) | 1.139 | 1.142 |

-- a factor 142, and the discrimination is a convergence statement, not a
tolerance: the frame-continued arm converges 2.3x over the doubling while the
lab-placed one stands still (1.003x).

**Attribution.**  PRE-EXISTING.  The anchor is applied once, at the end, to a
cascade whose relative layer placement it does not touch; the same table reads
identically on `rev`.  The fix's own composition test encodes the correct
placement (its staircase puts the lower ridge at `0.75 P` for a `0.25 P` first
walk) without saying why, so the property was known and is simply not written
down anywhere a caller would look.

**Severity P2** -- a caller who builds a two-sheared-layer stack from the
documented centre law gets a structure displaced by `W1`, silently.
**Reproducer:** `test_verify_v2_a_second_sheared_layer_sits_at_the_ACCUMULATED_walk`.
**Remedy:** one paragraph in `add_sheared_grating`'s docstring (and the same in
`add_layer`'s `slant_angle` note), stating that `centre` is a LAB position for
the FIRST sheared layer and a FRAME position (`+ sum of the walks above it`)
for later ones.

### 4.11 DEFECT D2 -- the default factorization has no anchored surface

`factorization='auto'` sends an IN-PLANE stack whose slant is UNIFORM to the
COVARIANT (spectral) cascade, which retains no per-order amplitudes at all.
On the library DEFAULT, therefore, a single `add_sheared_grating` layer
exposes neither `jones_transmission()` nor `per_order_amplitudes()` -- both
raise -- and V2 is reachable only through `factorization='convection'`, or
through a stack whose slants are MIXED (or which carries an out-of-plane
layer), which `'auto'` routes to convection by itself.

This is why 11 of this verification's 30 sheared census fixtures had to be
duplicated as `_conv` twins before they moved anything at all.  It is a row in
the fix's S2.2 table; it is a scope statement a user needs, and nothing pinned
it.  **Severity P3.** Pinned by
`test_verify_v2_the_default_factorization_exposes_no_transmitted_field`.

---

## 5. TASK 3 -- V1, the JAX dispatch

`t3_v1_jax.py`.  **Fixture (mine):** a `(5, 4, 3, 3)` x-ASYMMETRIC, y-varying
in-plane tensor cell (`eps` 1.30 .. 3.05 with off-diagonal `exy` up to 0.14)
over `px = 0.92 um`, `py = 0.80 um`, `wl = 0.61 um`, `n_sub = 1.55`,
`depth = 0.37 um`, `slant = (0.45, 0)`, oblique 28 degrees, `n_orders = 3`,
`degree = 5`, jax x64 ON.

The fixture's own defect magnitude, re-derived first so the test cannot go
vacuous -- the two NumPy calls (vertical vs slanted) against each other:
**`dR 2.880903302808535e-02`, `dT 9.4448758931444e-03`,
`dJones 8.257670740900952e-02`**, identical on both builds to every printed
digit.

### 5.1 The seven routes

| arm | PRE (`pre` and `rev`) | POST |
|---|---|---|
| `eps_tensor_cell`, `n_substrate`, `n_superstrate`, `depth`, `wavelength`, `theta`, `phi` | **all seven SOLVED** | **all seven `NotImplementedError`** |

Every pre-fix row read IDENTICALLY, and this is the load-bearing part:

* against the NumPy **VERTICAL** call: `dR 1.3437e-13`, `dT 6.6833e-15`,
  `dJones 1.0410e-13`;
* against the NumPy **SLANTED** call: `dR 2.880903e-02` -- the SAME number as
  the two NumPy calls' own gap, to seven figures;
* **all seven produced the SAME sha256** (`8590d3364d777b5f` on WIN), i.e. one
  answer, not seven near-answers;
* **`warnings` was EMPTY on all seven.**

Post-fix each of the seven raises, and the message names its own traced input,
the NumPy alternative and the z-staircase remedy.

### 5.2 The constant-tile no-op, widened

The fix measured one row (oblique, isotropic).  Measured here at FIVE mounts,
NumPy slanted vs NumPy vertical on the identical constant tile:

| mount | `dR` | `dT` | `dJones` | at `slant = (1.7, 0.9)` |
|---|---|---|---|---|
| isotropic, oblique 28 | 3.59e-15 | 4.90e-15 | 3.79e-15 | `dR 1.13e-14` |
| isotropic, NORMAL | 1.70e-16 | 4.63e-16 | 2.62e-16 | `dR 1.70e-16` |
| isotropic, CONICAL (phi = 37) | 4.85e-15 | 7.14e-15 | 8.12e-15 | `dR 1.97e-14` |
| **ANISOTROPIC** (`exx/eyy/ezz` 2.9/2.1/2.5, `exy = 0.42`), oblique 28 | 1.84e-14 | 1.59e-14 | 1.29e-14 | `dR 1.13e-14` |
| ANISOTROPIC, CONICAL | 4.10e-15 | 5.74e-15 | 1.39e-14 | `dR 3.59e-15` |

So the no-op is a code-path fact (the constant tile short-circuits to the
homogeneous modes before the slant is read), it survives conical incidence, a
full anisotropic tile and a 60-degree wall tilt, and it is not a small-angle
accident.

### 5.3 The EIGHTH-ROUTE HUNT -- twenty-two candidates, none silent

| candidate | PRE | POST |
|---|---|---|
| `jax.jit` over `depth` | SOLVED (silent, vertical) | **refused** |
| `jax.vmap` over `wavelength` | SOLVED | **refused** |
| `jax.grad` over `depth` | SOLVED | **refused** |
| a y-ONLY slant `(0, 0.45)` on a traced call | SOLVED | **refused** |
| an EXPLICIT zero slant on a traced call | SOLVED | SOLVED (the control) |
| a tile varying by `5e-13` (inside `_SLANT_NOOP_TOL`) | SOLVED | **refused** (the cell is a jax array) |
| a tile varying by `5e-11` (outside it) | SOLVED | **refused** |
| a TRACED `slant`, everything else NumPy | `ConcretizationTypeError` | `ConcretizationTypeError` |
| a CONCRETE `jnp` slant pair, everything else NumPy | SOLVED, and it is the correct NumPy SLANTED answer | same |
| `PMM2DStackHybrid.solve` with a traced thickness / wavelength / theta / half-space index | already refused at `4a987e3` | refused |
| `PMM2DStackHybrid` with a traced thickness and a SCALAR cell | already refused | refused |
| `pmm_jones_2d_staggered` under `jax.jit` | `ConcretizationTypeError` | same (that engine has NO jnp twin -- the string `jax` does not occur in `twod_staggered.py` or `stack2d_pure.py`) |
| `PMM2DStackPure.add_layer` with a concrete `jnp` thickness | SOLVED (numeric coercion, correct slanted answer) | same |
| a CONSTANT **NumPy** cell with a traced `depth` | SOLVED | **SOLVED**, `dR < 1e-9` against the NumPy VERTICAL call |
| the private twin `_pmm_jones_2d_cell_jax`'s signature | no `slant` parameter | unchanged; the module docstring now says so |

**No eighth silent route exists.**  Every candidate either routes through the
same seven-tuple (and is now refused), fails loudly on a coercion, or lands on
an engine with no jnp twin at all.

### 5.4 DEFECT D3 -- the concrete-`jnp` boundary

`_slanted_cell_is_a_frame_noop` answers `False` for ANY `is_jax_array(cell)`,
on the stated ground that "a TRACED cell cannot be inspected".  A CONCRETE
`jnp` array is not a tracer and IS inspectable (`np.asarray` on it is exact),
so a constant-valued concrete `jnp` cell is refused although the answer would
be the vertical film's to `1.7e-16`.  The refusal is loud and names the remedy
("use NumPy inputs"), and the fix's own control -- a constant NumPy cell with
a traced scalar -- does still solve.  **Severity P4**, recorded so that a
future relaxation (`isinstance(cell, jax.core.Tracer)` instead of
`is_jax_array`) is a deliberate decision rather than a discovery.

---

## 6. TASK 4 -- O2, the `T22` bar re-derived on an independent population

`t4_o2_census.py`.  The census is armed through `rcwa._core._INV_CENSUS`,
which `_guarded_inverse` reads as a module global at call time, so every
consumer that imported `_interface_smatrix_general` by name is instrumented
from one place (verified: the recorded interface counts match the cascades'
own structure, 2 per single-layer solve).

### 6.1 The population

**134 solves / 229 interfaces**, `T22` block sizes 2 .. 1058, spanning every
consumer of `_interface_smatrix_general`:

* the 1-D `PMMStack` convection-slant cascade at four shears x five order
  counts, both grid routes, `degree` 7 / 11 / 15, grazing 78 degrees, a dense
  superstrate, a high-index substrate, a 3.2 um period at `n_orders = 11`, and
  the covariant route;
* the 1-D out-of-plane vertical cascade at four order counts, both grid routes;
* the native-conical `PMMStack` cascade;
* the 2-D hybrid, tensor cells, three slants x four order counts + `M = 11`,
  out-of-plane vertical, `degree` 7 / 15, the `tree` cascade, grazing, dense
  superstrate, high contrast, a theta scan across the `|alpha| = 1` cut-off,
  a y-only slant and an xy slant;
* **the 2-D hybrid, SCALAR cells** -- three high-contrast x profiles against a
  half-cell of uniform ground, three slants x three order counts, plus
  `M` = 9 / 11, `degree` 7 / 15, four mounts including normal and 70/80
  degrees, a dense superstrate, a high-index substrate, a detuned ground, and
  two wavelengths at and beside a substrate Rayleigh cut-off;
* `pmm_jones_2d` (the single-layer entry), tensor and scalar, slanted and
  out-of-plane;
* the 2-D PURE staggered engine (slanted, out-of-plane, magnetic) and
  `pmm_jones_2d_staggered`;
* the Berreman 4x4 planar cascade at four angles plus a lossy and a thick film.

**A finding the fix's census did not have.**  The blow-up is excited by a
SCALAR high-contrast cell standing against a half-cell of uniform ground.  The
first 88 fixtures of this census -- tensor cells of comparable contrast at
every slant, order count, degree and mount -- were **all healthy**; not one
broken solve appeared until the scalar family was added.  Anyone re-deriving
this bar from tensor cells alone will measure a one-sided population and
conclude there is nothing to screen.

### 6.2 The two populations, this verification's numbers

Measured with the census armed on the POST tree.  `rcond` is each solve's
WORST interface, which is the statistic the refusal actually decides on (a
solve is refused when ANY of its interfaces trips).

| population | solves | equil. `rcond(T22)` | equil. resid | | |
|---|---|---|---|---|---|
| | | **WIN** | **WIN** | **WSL** | **WSL** |
| **REFUSED** | 21 | `4.678e-18 .. 4.780e-16` | `6.519e-03 .. 7.965e-01` | `3.851e-18 .. 4.524e-16` | `5.837e-03 .. 7.181e-01` |
| **SOLVED** | 110 | `2.079e-05 .. 1.0` | -- | `2.079e-05 .. 1.0` | -- |
| unreachable (conical + OOP raises) | 3 | -- | -- | -- | -- |

| statistic | WIN | WSL |
|---|---|---|
| the worst REFUSED reading | 4.780e-16 | 4.524e-16 |
| the worst SOLVED reading | **2.079e-05** | **2.079e-05** |
| the gap | **10.64 decades** | **10.66 decades** |
| `1e-10` above every refused reading | **5.32 decades** | **5.34 decades** |
| `1e-10` below every healthy reading | **5.32 decades** | **5.32 decades** |
| largest `sum R + T` among the SOLVED | **1.09520** | 1.09520 |

**Is there a HEALTHY solve within two decades of `1e-10`?  No.**  The hunt
included grazing incidence (1-D at 78 degrees, the hybrid at 70 and 80), a
dense superstrate (`n_sup = n_sub = 2.6`), a high-index substrate (3.9), a
3.2 um period at `n_orders = 11`, `n_orders = 11` on both the tensor and the
scalar families, a detuned ground, and two wavelengths at and beside a
substrate Rayleigh cut-off.  The three tightest healthy readings are
`2.079e-05` (`hyb_slant0.5_theta15_M5`), `2.971e-05` (`scalA_sl0.5_M11`) and
`4.098e-05` (`j2d_slant0.5_M9`) -- **5.3 decades** from the bar.  The middle
one reproduces the fix's own worst-healthy reading (`2.971e-05`, "the hybrid at
n_orders 11") to four figures on an independently written fixture.

### 6.3 The two-sided check

BROKEN is measured on the `rev` tree, which holds no guard, by the library's
own energy criterion; REFUSED is measured on `post`.  The two are independent
instruments, and the fixture list is identical.

| | WIN | WSL |
|---|---|---|
| fixtures | 134 | 134 |
| ... that never reach `_interface_smatrix_general` at all | 9 | 9 |
| **BROKEN on `rev`** (`sum R + T > 1.10`, or raised) | **21** | **21** |
| **REFUSED on `post`** (`_ConditioningError`) | **21** | **21** |
| **`refused set == BROKEN set`** | **True** | **True** |
| false refusals | **0** | **0** |
| broken solves let through | **0** | **0** |

The nine that reach no guarded interface are the three vertical conical
solves, the three conical out-of-plane ones (which raise
`NotImplementedError`), the magnetic pure-staggered layer and the two VERTICAL
in-plane scalar cells -- all of which take the symmetric
`_interface_smatrix` instead, and all of which solve cleanly
(`sum R + T <= 1.038`).  None of them is a hidden broken solve.

### 6.4 Bit-identity of every healthy solve

The same fixture set, `(orders, R, T, jones)` hashed on each, `post` against
`rev` on the SAME build.  A screen that only ADDS a raise must leave every row
it does not refuse byte-for-byte where it was:

| | WIN | WSL |
|---|---|---|
| solves that BOTH trees produced | **110** | **110** |
| **hashes identical** | **110** | **110** |
| **hashes moved** | **0** | **0** |

(110 rather than 104 + 9: the count includes every fixture that returned a
solve on both trees, guarded interface or not.)  Corroborated structurally by
this verification's own unarmed-guard control: `_guarded_inverse` called
WITHOUT `rcond_refuse` on a matrix whose singular values run `1` down to
`1e-20` -- one the ARMED call refuses at `rcond = 1.59e-18` -- returns
`np.linalg.inv` with the IDENTICAL sha256 and does not raise.

### 6.4b THE CENSUS HOOK ITSELF -- a decision the fix's gate does not make

`_guarded_inverse` computes the confirming residual when `rcond` falls below
`thr = max(_INV_RCOND_SCREEN if the census is armed else 0, rcond_refuse)` --
`1e-8` with the census armed, `1e-10` without.  Two different thresholds, two
different amounts of work, and the fix's gate exercises the armed path (its
threshold test) and the unarmed path (its blow-up test) but never both on the
same fixture.  Measured here over the whole census, run TWICE per arm:

| | WIN | WSL |
|---|---|---|
| solves whose bytes are identical census-armed vs census-off | **110 / 110** | **110 / 110** |
| refusals that fire in BOTH modes | **21 / 21** | **21 / 21** |

So the instrument does not change the verdict and does not change the answer.
Pinned by `test_verify_o2_the_refusal_is_the_same_with_the_census_armed_or_not`.

### 6.5 The benign warning rows

The fix's own false-alarm fixture: `sum R + T` = **1.0350** (normal incidence)
and **1.0726** (oblique 40), both still SOLVING and both still WARNING with the
library's "energy not conserved" text, on both builds; their `T22` reads
`rcond >= 7.29e-04`, 6.9 decades on the safe side of the bar.

This verification's census carries a much LARGER false-alarm population than
the fix's 22 rows -- **53 solves that warn** "energy not conserved" (or the
pure engine's "lossless energy closure violated"), `sum R + T` from `0.97202`
to **`1.09520`**, with `T22` `rcond` from **`2.079e-05`** to `5.962e-03`.  Not
one of them is refused, and the closest any of them comes to the `1e-10` bar
is **5.32 decades**.  Separating those from a `1e+31` is the whole point of
the screen, and it does.

### 6.5b An unrelated observation, recorded so it is not lost

All six `berreman_jones_1d` fixtures in this census raise numpy's
`ComplexWarning: Casting complex values to real discards the imaginary part`
somewhere inside that solver.  It reads IDENTICALLY on `post` and on `rev`
(six fixtures each), so it is PRE-EXISTING and not this fix's; `berreman.py`
is outside this brief's file list and it is reported here rather than chased.

### 6.6 NOTE N1 -- what the armed screen costs

The fix says the screen is "free" and "not one extra flop".  The second half
is exactly true for every caller that does not arm it (the early return is
unchanged), but the ARMED site now pays `_rcond_1_equilibrated` -- three O(n^2)
reductions, a gemv and two `isfinite` sweeps -- on EVERY generalized interface,
where before the census-off path returned immediately.  MEASURED (best of
seven, after warm-up, WIN):

| `n` (the `T22` block) | plain (WIN) | armed (WIN) | ratio WIN | ratio WSL |
|---|---|---|---|---|
| 66 | 0.00023 s | 0.00034 s | **1.51x** | **1.46x** |
| 242 | 0.00600 s | 0.00688 s | 1.15x | 1.14x |
| 450 | 0.03535 s | 0.04346 s | 1.23x | 1.22x |
| 722 | 0.12153 s | 0.15247 s | 1.26x | 1.10x |

(Block sizes 66 .. 722 are the ones this census actually produced; the full
range it saw was 2 .. 1058.)  The O(n^2) instrument is asymptotically free
against the O(n^3) inverse and the ratio falls with `n`, but at the smallest
blocks it is a measurable ~50% of the inverse -- and a much smaller fraction
of the whole solve, which also pays the modal eigenproblems.  Recorded so the
word "free" is bounded rather than taken on trust.  Not a defect.

---

## 7. TASK 5 -- TEST DURABILITY

`t5_durability.py` and `t5b_durability_rest.py` import the gate module itself
and call ITS helpers, so every number below is the one the assertion actually
reads.  "Margin" is the factor by which the reading clears its bar.

### 7.1 Every numeric bar, its origin, and its margin on both builds

| # | bar | origin | WIN reading (margin) | WSL reading (margin) |
|---|---|---|---|---|
| 1 | `gap > 1e-3` | DERIVED AT RUNTIME from the fixture's own two NumPy calls -- the test cannot go vacuous | 5.5135e-02 (**55.1x**) | 5.5135e-02 (**55.1x**) |
| 3 | `dR, dT, dJ < 1e-11` | measured 1.3e-15..3.0e-14; bar 3 decades above | 1.344e-15 (7438x) / 4.240e-15 (2359x) / 1.497e-14 (668x) | 2.331e-15 (4291x) / 2.953e-14 (**339x**) / 1.669e-14 (599x) |
| 4 | `dR, dT, dJ < 1e-11` (constant tile) | measured | 4.857e-17 (2.06e5x) / 2.076e-14 (482x) / 5.361e-16 (1.87e4x) | 8.327e-17 (1.20e5x) / 2.365e-14 (423x) / 2.066e-16 (4.84e4x) |
| 5 | `rel < 1e-5` | DERIVED from the FD's own floor `eps*\|f\|/h` = 4.44e-05 | 1.5184e-08 (**659x**) | 1.5251e-08 (**656x**) |
| 5 | `\|g\| > 1.0` | non-vacuity | 3.3210e+05 | 3.3210e+05 |
| 6 | `\|one - W\| < 1e-18` | ABSOLUTE, on a 2.4e-07 m quantity (4e-12 relative) | error EXACTLY 0.0 | EXACTLY 0.0 |
| 6 | opposite slants `== 0.0` | IEEE oddness of `tan` | exactly 0.0 | exactly 0.0 |
| 7 | `\|\|P\|-1\| < 1e-14` | measured 1.11e-16 | 1.110e-16 (**90.1x**) | 1.110e-16 (**90.1x**) |
| 8 | uniform film `< 1e-11` (x2), ratio `> 1e6` | ANALYTIC oracle | 6.369e-15 (1570x) / 1.122e-14 (892x) / 1.72e+14 (1.7e8x) | 7.214e-15 (1386x) / 1.175e-14 (851x) / 1.52e+14 (1.5e8x) |
| 9 | `r4/r12 > 3`; `r12 < 2 x step` | the oracle's own resolution | 6.0613 (**2.02x**) / 0.4884 (**2.05x**); amps 5.6996 (**1.90x**) / 0.4830 (**2.07x**) | 6.0613 / 0.4884 / 5.6996 / 0.4830 -- the same to 10 figures |
| 10 | flat inside (0.8, 1.25) | | J 1.0110 (**1.24x** to the edge), amps 0.9994 | identical to 10 figures |
| 10 | `none/shipped > 20` (x2), `conj/shipped > 20`, `conj/none > 1.3` | | 99.54 (4.98x) / 135.16 (6.76x) / 166.09 (8.30x) / 1.6686 (**1.28x**) | identical to 10 figures |
| 11 | `r4/r12 > 2`; `r12 < 2 x step` | | 3.1411 (**1.57x**) / 0.4321 (2.31x) | identical to 10 figures |
| 12 | `kx0 == 0.0`, `P_0 == 1+0j`, `>= 5` distinct phases, `none/shipped > 20` | exact identities + a count | (see 7.2) | (see 7.2) |
| 13 | `full < 2 x step`; four arms `> 10`; `K4/K6 > 1.5` | | (see 7.2) | (see 7.2) |
| 14 | split `< 1e-12`; agreement `< 1e-6` (x2); `> 10` (x2) | | (see 7.2) | (see 7.2) |
| 15 | `shipped < 0.05`; `none/shipped > 20`; `conj/shipped > 20` | | (see 7.2) | (see 7.2) |
| 19 | benign rows `1.0 < tot < 1.10` and WARNED | | 1.0350 / 1.0726, both warned | 1.0350 / 1.0726, both warned |
| 20 | broken `rcond < 1e-15` | derived from the fix's 71-solve census, ASSERTED on ONE fixture | 3.902e-17 (**25.6x**) | 7.125e-17 (**14.0x**) |
| 20 | broken `resid > 1e-3` | same | 0.08082 (80.8x) | 0.04509 (45.1x) |
| 20 | healthy `rcond > 1e-5` | same, asserted on THREE fixtures | 7.291e-04 (72.9x) | 7.291e-04 (72.9x) |
| 20 | `len(healthy) >= 4` | a COUNT of census rows | 6 (**1.5x**) | 6 (**1.5x**) |

### 7.2 The four remaining gate bars

`t5b_durability_rest.py`, split off only for runtime (these re-run the gate's
three most expensive fixtures).  WSL readings in the second column.

| # | bar | WIN reading (margin) | WSL reading (margin) |
|---|---|---|---|
| 12 | `kx[i0] == 0.0` | exactly `0.0` | exactly `0.0` |
| 12 | `P_0 == 1+0j` | exactly `(1.0, 0.0)` | exactly `(1.0, 0.0)` |
| 12 | `>= 5` distinct phases | 9 (**1.8x**) | 9 (**1.8x**) |
| 12 | `none/shipped > 20` | 85.9955 (4.30x) | 85.9955 (4.30x) |
| 13 | `full < 2 x step` | 0.45949 (**2.18x**); full 1.4199e-02 vs step 1.5451e-02 | 0.45949 (**2.18x**) |
| 13 | `only1/full > 10` | 39.011 (3.90x) | 39.011 (3.90x) |
| 13 | `only2/full > 10` | 82.104 (8.21x) | 82.104 (8.21x) |
| 13 | `none/full > 10` | 95.785 (9.58x) | 95.785 (9.58x) |
| 13 | `conj/full > 10` | 114.796 (11.48x) | 114.796 (11.48x) |
| 13 | `K4/K6 > 1.5` | 2.03945 (**1.36x**) | 2.03945 (**1.36x**) |
| 14 | split identity `< 1e-12` | 6.4509e-16 (1550x) | 6.4721e-16 (1545x) |
| 14 | `abs(full1-full2)/full1 < 1e-6` | 1.8742e-14 (5.3e7x) | 2.9786e-14 (3.4e7x) |
| 14 | `abs(half1-half2)/half1 < 1e-6` | 2.4215e-16 (4.1e9x) | 4.8430e-16 (2.1e9x) |
| 14 | `half/full > 10` | 88.454 (8.85x) | 88.454 (8.85x) |
| 14 | `none/full > 10` | 132.748 (13.27x) | 132.748 (13.27x) |
| 15 | `shipped < 0.05` | 1.5655e-02 (**3.19x**) | 1.5655e-02 (**3.19x**) |
| 15 | `none/shipped > 20` | 59.914 (**3.00x**) | 59.914 (**3.00x**) |
| 15 | `conj/shipped > 20` | 105.412 (5.27x) | 105.412 (5.27x) |

Every RATIO in this block reads identically on the two builds to six
significant figures; only the three machine-precision residuals differ at all,
and each of those clears its bar by more than nine decades.

Every figure the gate's docstrings quote for these four tests reproduces:
`6.45e-16` for the split identity, `88x` and `133x` for the half-sum and
no-sum arms, `1.42e-02` for the full sum at `K = 6` against its `1.55e-02`
oracle step.  The thinnest of them are test 13's `K4/K6 > 1.5` (**1.36x**) and
test 15's `none/shipped > 20` (**3.00x**).

**A finding, from getting this wrong first.**  The gate's test 14 hands BOTH
halves the SAME `centre` (`0.5 - 0.0625`).  Reproducing it with the halves at
their LAB positions (`+0.125 P` apart) reads `0.6728` on the split identity
instead of `6.45e-16` -- because the cascade continues the frame, so the second
half already stands at `+W1` in the lab.  That is DEFECT D1 appearing a second
time inside the fix's own test file, again without being stated anywhere a
caller would look.

### 7.3 What is FLAGGED

1. **SAMPLE-SCOPED, and the thinnest bar in the file: test 20's
   `max(broken rcond) < 1e-15`.**  Its constant is derived from the fix's
   71-solve census but ASSERTED on a single fixture, and it is the only bar
   in the file whose reading moves appreciably between builds
   (`3.90e-17` WIN vs `7.12e-17` WSL, a 1.83x spread) -- leaving **14.0x
   (1.15 decades)** on the tighter build.  This verification's own 134-solve
   census puts the WORST refused reading in a wider population at
   **4.78e-16**, which is only **0.32 decades** below the bar.  A refusal
   whose `rcond` happened to land there would fail the test while the guard
   was working correctly.
   **Recommendation:** move that bar to `1e-13`, which still sits 2.3 decades
   above the worst refused reading measured anywhere here and 8.3 decades
   below the healthy floor.  Not blocking.

2. **A COUNT bar: `assert len(healthy) >= 4`** (reading 6 = 3 solves x 2
   interfaces).  Structurally fixed today, but it is the `S5` shape
   `docs/TESTING_STANDARDS.md` names -- an exact count of machinery -- and it
   would break if interface dedup or caching were ever introduced on that
   path.  Restating it as "at least one healthy interface was recorded"
   removes the exposure at no cost.

3. **Tight (1.28x .. 2.07x) but BUILD-FREE: tests 9, 10 and 11.**  Every one
   of their readings agrees between the two builds to TEN significant figures,
   so their tightness is exposure to an intentional algorithm change, not to
   build spread -- which is what an envelope bar is for.  Not flagged as
   per-build; recorded so a future failure is read as "re-derive once", not
   "flaky".

4. **An ABSOLUTE bar on a dimensional quantity: test 6's
   `abs(one - V2_W) < 1e-18`** on a `2.4e-07 m` walk (4e-12 relative).  It
   reads EXACTLY `0.0` on both builds because `tan(arctan(x)) * d` round-trips
   exactly here, so there is no exposure today; a relative form would survive
   a change of the fixture's units.

5. **No bar in the file is per-build in the `S1`-`S5` sense.**  Every one was
   re-measured on both builds; the only reading with a cross-build spread
   above 2x is test 3's `dT` (4.24e-15 WIN vs 2.95e-14 WSL, a 7.0x spread),
   and its bar sits 339x above the worse of the two.

### 7.4 RUNTIME, under load

The gate file ALONE, `-p no:randomly`, thread caps 1, on a box that was
concurrently running this verification's own census and a 22-file regression:

| | WIN | WSL |
|---|---|---|
| the 22-test file | **78.43 s** | **78.83 s** |
| slowest test (`o2_the_measured_cures_still_solve`) | 15.12 s | 15.09 s |
| headroom against the 40 s per-test cap | **2.65x** | **2.65x** |
| headroom against the 4-minute file budget | **3.06x** | **3.04x** |

**22 passed, 0 failed, on both builds.**  The fix reported 66.96 s / 69.52 s on
an idle box; the 1.17x / 1.13x here is the concurrent load, and the budget
holds either way.

This verification's own file, `test_verify_slant_anchor_v1_v2_o2.py`, under
the same load: **8 passed in 25.30 s (WIN) and 22.00 s (WSL)** (the transcripts in
`results/verify_gate_*.txt`; three runs each spread 19.1-25.3 s under varying
load), slowest test 9.21 s / 13.55 s -- **2.4x / 2.7x** inside the 60 s budget
the brief sets and **4.3x / 2.9x** inside the 40 s per-test cap.


---

---

## 8. SUITE RUNS AND LINT

### 8.1 The 22 named files, Windows

`tests/unit/test_pmm*.py tests/unit/test_fix_pmm*.py
tests/unit/test_verify_pmm*.py tests/unit/test_rcwa*.py` -- 22 files,
`OMP` = `OPENBLAS` = `MKL` = 1, `-p no:randomly`:

**582 passed, 0 failed, 0 errors, 50 warnings, in 2245.06 s (0:37:25).**

The 50 warnings are pre-existing energy-tripwire and lossy-incidence notices
raised on purpose by their own tests.  The four slowest are
`test_pmm2d_lossless_closure_two_sided.py::test_solver_warning_matches_the_predicate_on_every_arm`
(232.74 s), `test_pmm2d_staggered_oop.py::test_g5_no_fourier_floor_two_sided`
(65.95 s) and the two `test_rcwa.py::test_analytic_energy_and_clean_convergence`
parametrisations (63.6 / 62.3 s), none of them this branch's.

### 8.2 The two gate files, both builds

| file | WIN | WSL |
|---|---|---|
| `test_fix_slant_anchor_v1_v2_o2.py` (the fix's, 22 tests) | **22 passed, 78.43 s** | **22 passed, 78.83 s** |
| `test_verify_slant_anchor_v1_v2_o2.py` (this one, 8 tests) | **8 passed, 25.30 s** | **8 passed, 22.00 s** |

Both under concurrent load (this verification's own 134-solve census and the
22-file regression were running alongside).

### 8.3 Lint

```
$ ruff check lumenairy/ tests/ validation/probe_verify_slant_anchor/
All checks passed!
```

(ruff 0.15.16, on WSL.)  `lumenairy/` was not edited by this verification.
Every transcript above is reproduced verbatim in
`validation/probe_verify_slant_anchor/results/RUNS.md`, because `*.log` is
gitignored (`.gitignore:43`).

### 8.4 `.test_durations`

SPLICED, not rewritten: `splice_durations.py` merges the eight measured node
ids into the existing 12 655, giving 12 663.  `git diff --stat` reads **8
insertions, 0 deletions**, and the file stays sorted.  pytest-split's own
`--store-durations` would have replaced the file with only this run's tests.

## 9. WHAT COULD NOT BE VERIFIED

1. **Cross-build coverage is WIN + WSL only, and both are OpenBLAS.**  The
   same limitation the fix records.  Nothing here distinguishes a genuinely
   different LAPACK; the six readings with any cross-build spread at all are
   listed in S7.1 and none of them crosses a bar.

2. **The 2-D hybrid's OWN frame anchor is not re-derived here.**  It shipped
   at `0ebd632`, before this fix's branch point, and appears in this
   verification only as a bit-identity control (its surfaces do not move).
   V2's 1-D rule was verified on its own terms, not inherited from it.

3. **The 1-D general MORTAR interface was not reached** by any fixture written
   here either: on `layer_grids='per-layer'` the window grids of every 1-D
   fixture in this census CONFORM, so `_ifc_g` takes the square
   `_interface_smatrix_general` bypass, which IS guarded.  The fix records the
   same negative result.  What WAS re-checked, by source and by measurement:
   neither `_interface_smatrix_general_mortar` nor
   `_interface_smatrix_general_mortar_2d` forms an explicit inverse (the first
   ends in `np.linalg.solve`, the second in `_guarded_mortar_solve`), and
   `_guarded_mortar_solve` is BIT-IDENTICAL to `np.linalg.solve` on this
   verification's own 40x40 and 200x200 complex pairs (`maxdiff` exactly 0.0,
   byte-equal).  So there is nothing at those sites for `_guarded_inverse` to
   screen.

4. **WHY the truncated sheared or out-of-plane mode set becomes rank-deficient
   against the Rayleigh half-space set is not traced here either.**  This
   verification adds one empirical constraint the fix did not have: on 88
   TENSOR-cell fixtures spanning every slant, order count, degree and mount
   tried, not one solve broke; every broken solve in this census carries a
   SCALAR high-contrast cell standing against a half-cell of uniform ground.
   That is a lead, not a mechanism.

5. **The `_guarded_inverse` guard is NumPy-only by construction**
   (`xp is not np` returns early), so the JAX and CuPy backends of any future
   generalized-cascade caller remain unscreened.  No such caller exists today
   -- the generalized cascade is NumPy-only on every route this census reached
   -- so the exposure is latent rather than live, and it is documented in the
   function.  Not measured further.

6. **The regression run is the 22 files the brief names, not the full
   matrix.**  A release still needs the un-masked main-CI matrix on the merge,
   per `docs/TESTING_STANDARDS.md`'s process rules.

7. **`test_v5_14_2_backlog_batch.py::test_jones_2d_even_sector_matches_full`,
   the fix's OPEN ITEM 7, was not re-adjudicated.**  It is outside the named
   file set and the fix already measured the branch tip and the branch point
   reading identically on each build.
