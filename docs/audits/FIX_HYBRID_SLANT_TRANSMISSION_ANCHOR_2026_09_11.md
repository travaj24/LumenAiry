# FIX -- the HYBRID 2-D PMM's transmitted amplitudes were FRAME-referenced on a slanted PATTERNED layer

**Date:** 2026-09-11 - **Branch:** `fix/hybrid-slant-transmission-anchor`
off `wave2/pmm2d` @ `9b36ded`.
**Worktree:** `C:/tmp/lum_hyb`.
**Defect:** **D1** of `docs/audits/VERIFY_PMM2D_STAGGERED_SLANT_2026_09_10.md`
S3.4 / S10 -- silent-wrong, pre-existing since
`docs/audits/BUILD_PMM2D_SLANT_METRIC_2026_08_16.md`.
**Probes:** `validation/probe_fix_hybrid_slant_anchor/` (own README, five
scripts, every run's JSON on both builds).
**Binding law:** `docs/TESTING_STANDARDS.md`.

---

## VERDICT UP FRONT

`PMM2DStackHybrid.jones_transmission()` and
`per_order_amplitudes('transmission')` omitted the per-order frame-anchor
phase.  **Reproduced on both builds to every printed digit, fixed, and gated.**

| | before | after | the wrong-sign arm |
|---|---|---|---|
| per-order transmitted amplitudes vs the engine's own fine staircase, oblique 25 | **5.914e-01** | **3.122e-02** | 4.841e-01 |
| the same, conical 25-40 | **5.610e-01** | **1.636e-02** | 6.955e-01 |
| the same, normal | **8.156e-01** | **2.380e-02** | -- (a half walk makes `P_m` real at `kx0 = 0`) |
| normal, QUARTER walk (where it does not degenerate) | **8.163e-01** | **1.725e-02** | 9.025e-01 |
| zeroth-order transmission Jones, oblique 25 | 4.909e-01 | 1.421e-02 | 3.777e-01 |
| the best possible SINGLE GLOBAL phase (oblique / conical) | 4.631e-01 / 4.447e-01 | -- | -- |

`R`, `T`, the reflection Jones and `per_order_amplitudes('reflection')` are
**bit-identical** before and after; 137 of 144 sha256 hashes over 21 fixtures
match the read-only main clone, and the 7 that move are exactly the transmission
accessors on the four slanted PATTERNED rows.

**Two shapes carry a slant and must NOT be anchored**, and getting that wrong
would have been a new silent-wrong of the same family: a UNIFORM layer and a
CONSTANT-tile patterned cell are both solved as the plain vertical film, so the
sum skips them (anchoring either costs `1.371e+00` on an answer exact at
`0.000e+00`).

**One scope statement is now documented and pinned rather than left implicit:**
with a PATTERNED layer BELOW a slanted one, the hybrid solves the
shear-CONTINUED solid -- the lower layer riding along with the walk -- which is
the shape `PMM2DStackPure` refuses outright.

---

## 0. Arms

| | WIN | WSL | BASE (the pre-fix reference) |
|---|---|---|---|
| tree | `C:/tmp/lum_hyb` | same, via `/mnt/c` | `D:/Metacept/.../Lumenairy`, READ-ONLY |
| interpreter | CPython 3.14.6 | CPython 3.12.3 | CPython 3.14.6 / 3.12.3 |
| numpy / scipy | 2.4.4 / 1.17.1 | 2.4.6 / 1.17.1 | as the build using it |
| threads | `OMP` = `OPENBLAS` = `MKL` = 1 | `OMP` = `OPENBLAS` = 1 | as above |

The BASE clone's `stack2d.py`, `stack2d_pure.py`, `twod.py`, `twod_jones.py`
and `rcwa/_core.py` are **byte-identical** (`diff -q`, five files, no output) to
the pre-fix worktree, so it is a legitimate independent "without" arm *and* the
reference for every bit-identity hash -- and it stays valid after the worktree
is edited, which is why every pre-fix arm below was taken there.

Every probe calls `_lib.arm()`, which decides `fix` / `base` from
`lumenairy.__file__` itself -- never from a flag -- and stamps the resolved
path, interpreter, numpy, scipy and thread caps into its JSON.

**Cross-build spread.** Of the 34 scalar quantities re-measured on both builds
at the test fixture's size, **all 34 print identically to every digit**; of the
26 measured at the census fixture's size, 26 do.  The only quantity that moves
is a wall clock (S8).

---

## 1. THE DEFECT, reproduced

`validation/probe_verify_slant/v8_hybrid_anchor.py` (the verification's own
reproducer), run unchanged in this worktree on both builds:

| mount | hybrid raw | hybrid `x P_m` | hybrid `x conj(P_m)` | best single GLOBAL phase | staircase own step (dR / dJ) |
|---|---|---|---|---|---|
| oblique 25 | 5.914e-01 | **3.122e-02** | 4.841e-01 | 4.631e-01 | 1.35e-02 / 8.00e-02 |
| conical 25-40 | 5.610e-01 | **1.636e-02** | 6.955e-01 | 4.447e-01 | 1.34e-02 / 6.04e-02 |

WIN and WSL agree on every digit, and every digit reproduces the verification's
table.  **CONFIRMED as reported.**

### 1.1 The derivation, done here rather than taken

A slanted region is solved in the frame `u = x - t_x z`, `v = y - t_y z`,
`w = z`, in which the structure is `w`-invariant.  The solver's state is
therefore the FRAME Fourier coefficient: the field is
`sum_m F_m(w) exp(i alpha_m . (u, v))`.  Two consequences follow immediately:

1. the frame is anchored at the layer's TOP face, so at `w = 0` frame and lab
   coincide -- **the superstrate interface and the reflection need nothing**;
2. at the layer's BOTTOM, `w = d`, the same plane is `z = d` with
   `(u, v) = (x - t_x d, y - t_y d)`, so against the substrate's own basis
   `exp(i alpha_m . (x, y))`

   ```
       A_lab(m) = exp(-i alpha_m . t_int d) A_frame(m)
   ```

`alpha_m` is real for every order, propagating or evanescent, so the factor is
UNIMODULAR and no efficiency can move.  **That is the whole reason the omission
was silent**: `R + T` closes to the same value with or without it, so neither
the library's own energy tripwire nor any existing test could see it.

The hybrid's cascade runs in the INTERNAL conj gauge and its public amplitudes
are the conjugate of the cascade output (`amp["tx"] = conj(tx)`), so the factor
applied to the PUBLIC amplitudes is the conjugate one:

```
    A_lab(m) = exp(+i k0 (alpha_m . slant) d) A_frame(m)
```

which is *the same public expression* `PMM2DStackPure.solve` applies (it spells
the same number as `exp(-i k0 (kx _shx + ky _shy))` with
`_shx = -sum(slant_x d)`).  The sign is settled by measurement, not by taste:
the conjugate arm is `4.841e-01 / 6.955e-01` against `3.122e-02 / 1.636e-02` --
and it is WORSE than applying nothing at all, so the factor cannot be a fudge
absorbing an arbitrary residual.

### 1.2 The trap that had to be avoided twice

With a **WHOLE-period** walk `exp(2 pi i m t d / px) = 1` for every order: the
per-order structure vanishes and the anchor degenerates into a single global
phase that no measurement can separate from any other.  Every probe here walks
HALF a period -- and at **NORMAL incidence a half walk degenerates one level
further**, because `P_m = exp(i pi m)` is then REAL and `x P_m` equals
`x conj(P_m)` identically (measured: `2.380e-02` for both).  `p1_census.py`
therefore adds a QUARTER-walk normal-incidence arm, where `P_m = i^m`:

| normal, quarter walk | before | after | `x conj` after |
|---|---|---|---|
| per-order transmitted amplitudes | 8.163e-01 | **1.725e-02** | 9.025e-01 |

---

## 2. THE CENSUS -- which public outputs were frame-referenced

`p1_census.py`, both builds, `n_orders = 9`, a 6-pixel x-asymmetric cell, a
HALF-period walk, oracle = the hybrid's own 15-slice staircase of the same
solid (60-pixel cells, lab-referenced by construction).  **Every row below is
identical on WIN and WSL to every printed digit.**

| public output | normal | oblique 25 | conical 25-40 | verdict |
|---|---|---|---|---|
| `solve()` -> `R` | 1.732e-03 | 3.886e-03 | 1.796e-03 | **unaffected** (staircase's own step 1.80e-02 / 1.35e-02 / 1.34e-02) |
| `solve()` -> `T` | 1.207e-02 | 1.025e-02 | 9.927e-03 | **unaffected** |
| `solve()` -> `jones_reflection` | 5.850e-03 | 5.928e-03 | 5.269e-03 | **unaffected** (staircase step 8.15e-02 / 8.00e-02 / 6.04e-02) |
| `per_order_amplitudes('reflection')` | 2.007e-02 | 1.864e-02 | 1.749e-02 | **unaffected** -- and anchoring it makes it WORSE (1.680e-01 / 2.885e-01 / 1.607e-01) |
| `per_order_amplitudes('transmission')` | **8.156e-01 -> 2.380e-02** | **5.914e-01 -> 3.122e-02** | **5.610e-01 -> 1.636e-02** | **WAS WRONG, FIXED** |
| `jones_transmission()` | 8.390e-03 (unchanged) | **4.909e-01 -> 1.421e-02** | **1.368e-01 -> 9.053e-03** | **WAS WRONG, FIXED** (at normal incidence the ZEROTH order alone is exempt: `P_0 = 1` when `kx0 = ky0 = 0`) |

and the surfaces that cannot serve a frame-referenced field at all, checked
rather than assumed:

| surface | on a slanted stack | so |
|---|---|---|
| `solve(retain_internal=True)` | raises `NotImplementedError` (a slanted layer is a `'gen'` modal set) | -- |
| `internal_field(z)` | raises `ValueError` (nothing retained) | no anchor decision to make |
| `layer_absorption()` | raises `ValueError` | -- |
| `solve_vs_wavelength(...)` then `jones_transmission()` | raises `ValueError` -- the sweep solves on private clones and leaves `self._modal = None` | -- |
| `pmm_jones_2d(..., slant=)` | returns `(orders, R, T, jones_REFLECTION)`; no transmission surface exists | **the single-layer entry never had the defect** |

`jones_field_from_orders(...)` consumes the `per_order_amplitudes` dict, so it
is fixed by the same change with nothing of its own to do.

### 2.1 The two NULL rows a fix must not disturb

| row | before | after | what anchoring it would cost |
|---|---|---|---|
| slanted UNIFORM layer vs the vertical film, `jones_transmission` | `0.000e+00` (sha256-identical) | `0.000e+00` (sha256-identical) | **1.525e+00** |
| CONSTANT-tile slanted PATTERNED layer vs the vertical film | `0.000e+00` (sha256-identical) | `0.000e+00` | **1.525e+00** |

The reason is in the source and is deliberate: `add_layer`'s uniform branch does
not even STORE the slant, and `_build_layer_modes` / `_mode_key` short-circuit a
constant-valued tile to `_homogeneous_modes` **before** the slant is read.
Neither layer ever enters a frame.  A naive "sum over every layer carrying a
slant keyword" breaks the second row, which is why the shipped helper
`_layer_enters_slant_frame` **mirrors `_build_layer_modes`'s own uniform-tile
test verbatim**, and why the test file asserts the helper and the modal build
agree on both sides (`test_c4_...`).

---

## 3. THE COMPOSITION RULE

**Stated.** Each layer that enters a sheared frame contributes
`exp(+i k0 (alpha_m . slant_j) d_j)` to everything transmitted below it, and the
frames simply continue downward, so the offsets ADD:

```
    A_lab(m) = exp(+i k0 alpha_m . W) A_frame(m),
    W = sum over layers that ENTER A FRAME of slant_j * thickness_j
```

with "enters a frame" = a PATTERNED layer (scalar with a non-constant tile, or
tensor) carrying a non-zero slant.  That is exactly `_slant_frame_walk`.

**Measured** (`p2_composition.py`, `n_orders = 7`; `p4_test_bars.py`,
`n_orders = 5`; both builds identical to every printed digit).  The "pre-fix"
column is the read-only main clone; "shipped" is this branch.

| case | stack | pre-fix, as returned | shipped | the wrong sums |
|---|---|---|---|---|
| A | slanted PATTERNED over a UNIFORM film | 1.061e+00 / 8.548e-01 | **6.185e-03 / 2.904e-02** | conjugate 1.185e+00 / 1.249e+00 |
| B | UNIFORM film ABOVE the slanted layer | 1.003e+00 / 8.224e-01 | **6.199e-03 / 5.909e-03** | -- (the film above contributes nothing, as it must) |
| C | TWO slanted halves of `d/2` | 1.111e+00 / 8.919e-01 | **5.872e-03 / 6.123e-03** | only ONE half in the sum: 6.313e-01 / 4.816e-01 |
| E | a slanted UNIFORM film BELOW | `0.000e+00` vs the vertical film | `0.000e+00` | putting it in the sum: 4.292e-01 / 3.242e-01 |
| F | a CONSTANT-tile slanted cell BELOW | `0.000e+00` | `0.000e+00` | putting it in the sum: 4.292e-01 / 3.242e-01 |

(each pair is oblique 25 / conical 25-40).  Case C also satisfies the LAYER-SPLIT
identity exactly -- two halves of `d/2` are sha256-identical to one layer of `d`
on `jones_transmission`, `0.000e+00` -- but that identity alone cannot see the
SUM (both arms would be equally wrong), which is why the gate is C's comparison
against the lab-referenced staircase with the half-sum arm beside it (**81.5x /
65.6x** separation at the test size).

### 3.1 Does the REFLECTION from below a slanted layer need a round-trip phase?

**No -- measured.**  Put a reflecting `eps = 3.6` film under the sheared layer
so a strong return comes back up through it (`p4_test_bars.py`, both builds):

| quantity | oblique 25 | conical 25-40 |
|---|---|---|
| `dR` vs the staircase + the same film | 8.570e-04 | 1.664e-03 |
| `dJones_reflection`, AS RETURNED | **3.909e-03** | **3.859e-03** |
| the staircase's own K5 -> K15 step | 1.15e-02 | 1.37e-02 |
| reflection `x P0` (a one-way round trip) | 2.282e-01 | 2.169e-01 |
| reflection `x P0^2` (a full round trip) | 2.582e-01 | 3.211e-01 |

The reflection as returned sits **2.9x / 3.6x BELOW the oracle's own step**,
while either round-trip factor is **20x / 16x ABOVE it**.  The derivation says
why: the frame is anchored at the sheared layer's TOP, which is where the
reflected wave leaves it, so no offset ever accumulates on that side.

### 3.2 SCOPE -- a PATTERNED layer BELOW a slanted one

The far-field factor is exact only while every region below a sheared one is
HOMOGENEOUS (a lateral offset is a pure gauge there, since a homogeneous
half-space maps to itself under translation) or continues the SAME shear.  With
a PATTERNED layer below, the cascade matches that layer's LAB coefficients
directly to the sheared region's FRAME coefficients, which is the physically
realizable solid in which **the lower layer is TRANSLATED by the accumulated
walk** -- the frame simply continuing downward.

Measured with a QUARTER-period walk (so `+walk` and `-walk` are DIFFERENT
translations of the 12-pixel lower cell; at a half walk that cell is its own
image), the lower layer placed three ways against a 15-slice staircase of the
upper one:

| lower layer placed | per-order T (anchored) | `dR` | `dJones_reflection` |
|---|---|---|---|
| AS WRITTEN | 3.752e-01 | 1.082e-02 | 5.054e-02 |
| translated by **+walk** (riding the shear) | **1.290e-02** | **1.037e-03** | **9.284e-03** |
| translated by -walk | 1.637e-01 | 8.369e-03 | 2.771e-02 |

(`n_orders = 7`, oblique 25; at `n_orders = 5` the same rows read 3.715e-01 /
**1.554e-02** / 1.649e-01 and dJones(refl) 4.995e-02 / **7.462e-03** /
2.765e-02, and at conical 25-40 3.493e-01 / **1.521e-02** / 1.443e-01.)  The
`+walk` arm wins by **23.9x .. 29.1x** on the transmission and **5.4x .. 6.7x**
on the REFLECTION -- and the reflection carries no anchor at all, so it is the
reading that identifies the GEOMETRY rather than the phase.

`PMM2DStackPure._check_stack_slant` REFUSES this shape outright (its "MIX of
vertical and slanted regions above a pattern" branch).  The hybrid has no such
refusal and accepts it silently.  **No refusal was added here** -- that is a
behaviour change beyond a phase fix, and it would break stacks that work today
-- but the behaviour is now stated in `add_layer`'s docstring and pinned by
`test_e1_a_patterned_layer_below_a_slanted_one_rides_the_shear`, so it cannot
change unnoticed.  See O1.

---

## 4. WHAT SHIPPED

`lumenairy/elements/pmm/stack2d.py`, three sites, nothing else:

1. **`_layer_enters_slant_frame(L)`** (new, module level) -- does this layer
   actually get solved in a sheared frame?  Mirrors `_build_layer_modes`'s own
   uniform-tile test (`max|tile - tile.flat[0]| < 1e-12`) so the two can never
   drift apart, with the measured cost of getting it wrong in the docstring.
2. **`_slant_frame_walk(layers)`** (new, module level) -- the accumulated
   `(sum t_x d, sum t_y d)` in metres over those layers.  Returns `(0.0, 0.0)`
   for any stack with no sheared region, which is the branch that keeps the
   pre-fix bit-identity STRUCTURAL rather than tolerance-based.
3. **`solve()`** -- computes `tphase = exp(+i k0 (kx W_x + ky W_y))` once (only
   when the walk is non-zero) and multiplies it into the PUBLIC transmitted
   amplitudes `amp["tx"] / amp["ty"]` alone.  `R`, `T`, `rz/tz` and the
   reflection Jones are computed from the untouched cascade output ABOVE that
   line, which is why they stay bit-identical by construction and not by
   accident.

A NEW private helper was added rather than sharing the pure engine's inline
computation, because `stack2d_pure.py` is another agent's file this round.

### 4.1 A SECOND silent-wrong, found while censusing the first

`add_layer` already refused `slant=` on a TRACED `eps_cell`, on exactly the
right ground -- "the slanted layer runs the 4N generator and the generalized
cascade, which the 2-D JAX surface does not implement".  But the JAX dispatch in
`solve()` is reached by **six other traced inputs**: a layer THICKNESS, the
wavelength, `theta`/`phi`, a half-space index, a traced uniform `eps`.  And
`pmm/_jax_stack2d.py` contains the string `slant` **zero times**.

MEASURED before the guard (jax 0.11.0, x64), a slanted patterned layer with a
TRACED THICKNESS:

| arm | reading |
|---|---|
| the traced-thickness solve vs the VERTICAL stack | `dR = 0.000e+00`, `dJones = 0.000e+00` -- **bit-identical** |
| the same vs the correct NumPy SLANTED answer | `dR 1.839e-02`, `dJones 3.227e-02` |
| energy closure | conserved; no warning |

It returned the VERTICAL answer, silently, with `R` and `T` wrong this time --
the same shape as D1 one dispatch away, and worse, because no unimodularity
protects the efficiencies here.  It now raises `NotImplementedError`, using the
SAME `_layer_enters_slant_frame` decision as the anchor, so:

| arm | behaviour |
|---|---|
| traced THICKNESS + slanted patterned layer | **raises** |
| traced WAVELENGTH + slanted patterned layer | **raises** |
| traced thickness, VERTICAL layer (control) | solves -- the refusal is about the shear, not the API |
| traced thickness + CONSTANT-tile "slanted" layer | solves (closure 1.000000) -- a genuine no-op is not refused |

Gate: `test_f2_a_slanted_patterned_layer_refuses_the_JAX_path`.

Docstrings: `add_layer` gained a FRAME ANCHOR paragraph (the factor, which
layers count, and the scope statement of S3.2); the anchor site carries the
derivation, the sign measurement and the "which layers count" rule inline.

---

## 5. BIT-IDENTITY

`p3_bit_identity.py` / `p3_compare.py`: **21 fixtures**, each hashed on
`orders`, `R`, `T`, `jones_reflection`, `jones_transmission`,
`per_order_amplitudes('transmission')` and `per_order_amplitudes('reflection')`
(the sweep fixture has four) -- **144 sha256 hashes**, against the READ-ONLY
main clone.

| comparison | identical | differing | unexpected |
|---|---|---|---|
| pre-fix worktree vs BASE clone, WIN | **144** | 0 | 0 |
| pre-fix worktree vs BASE clone, WSL | **144** | 0 | 0 |
| **post-fix worktree vs BASE clone, WIN** | **137** | **7** | **0** |
| **post-fix worktree vs BASE clone, WSL** | **137** | **7** | **0** |

(the post-fix rows were re-taken AFTER the S4.1 JAX guard as well, unchanged --
that guard lives inside the `_jax_in` branch, which none of these 21 NumPy
fixtures enters).

The seven that move, all EXPECTED:

| fixture | key | before | after |
|---|---|---|---|
| `slanted_patterned_oblique` | `jones_transmission` | `3f9d8fae37ed1fcb` | `4cca0071873633b5` |
| `slanted_patterned_oblique` | `per_order_transmission` | `76a393bb17dc86cc` | `4d51e5b8d52c798d` |
| `slanted_patterned_conical` | `jones_transmission` | `05ac8fa79b371361` | `894f20a10d30521c` |
| `slanted_patterned_conical` | `per_order_transmission` | `219b60fb1ffe0ecb` | `3aeea9f3d11a9a14` |
| `slanted_patterned_normal` | `per_order_transmission` | `a2543b439413c6c1` | `923840cfaaacb719` |
| `slanted_tensor_inplane_oblique` | `jones_transmission` | `df4fcc6406b868ec` | `ea8cfce55fa7eaa4` |
| `slanted_tensor_inplane_oblique` | `per_order_transmission` | `fceeb11254b2952c` | `4314099472ad1e6b` |

Note `slanted_patterned_normal.jones_transmission` is **NOT** in the list: at
normal incidence the zeroth order has `alpha_0 = 0`, so `P_0 = 1` and the
ZEROTH-order Jones alone is exempt while its neighbours are not.  A gate that
looked only at `jones_transmission()` at normal incidence would have seen
nothing.

The 21 fixtures: vertical scalar at normal / oblique / conical; a vertical
in-plane tensor; a vertical OUT-OF-PLANE tensor (the 4N generator with no
slant); a three-layer vertical stack; a uniform-film stack; the even-parity fold
`symmetry='auto'` AND `symmetry=False` on a centro cell; tapered pillars; the
`fused` and `tree` cascade strategies; slanted UNIFORM x-only and diagonal; a
CONSTANT-tile slanted cell; a vertical pattern OVER a slanted uniform layer;
four slanted PATTERNED rows (scalar oblique / conical / normal, in-plane
tensor); and a three-wavelength `solve_vs_wavelength` sweep.

The same 21 rows on WSL are internally consistent (144/144 pre-fix) with
different hash values, as two BLAS builds must be; the identity claim is
same-build throughout.

---

## 6. TWO-SIDED, and the fail-before

The two-sided arms are in the numbers above; collected:

| claim | right arm | "no anchor" | "conjugate" | best GLOBAL phase |
|---|---|---|---|---|
| census fixture, oblique 25 | **3.122e-02** | 5.914e-01 (18.9x) | 4.841e-01 (15.5x) | 4.631e-01 (14.8x) |
| census fixture, conical | **1.636e-02** | 5.610e-01 (34.3x) | 6.955e-01 (42.5x) | 4.447e-01 (27.2x) |
| census fixture, normal quarter walk | **1.725e-02** | 8.163e-01 (47.3x) | 9.025e-01 (52.3x) | 4.470e-01 (25.9x) |
| test fixture, oblique 25 | **7.778e-03** | 1.113e+00 (143x) | 1.242e+00 (160x) | 3.821e-01 (49.1x) |
| test fixture, conical | **7.381e-03** | 8.936e-01 (121x) | 1.303e+00 (176x) | 4.476e-01 (60.6x) |

and the decision that identifies the defect as a REFERENCE FRAME rather than an
accuracy gap -- refine the oracle and watch which arm follows:

| staircase K | 3 | 5 | 15 | first/last |
|---|---|---|---|---|
| shipped, oblique | 3.887e-02 | 1.490e-02 | 7.778e-03 | **5.00x** |
| no anchor, oblique | 1.113e+00 | 1.114e+00 | 1.113e+00 | **1.000x** |
| shipped, conical | 2.677e-02 | 1.275e-02 | 7.381e-03 | **3.63x** |
| no anchor, conical | 8.905e-01 | 8.944e-01 | 8.936e-01 | **0.9965x** |

**The fail-before runs on this build's own bytes.**  `_slant_frame_walk` IS the
shipped decision point, so monkeypatching it to `(0.0, 0.0)` reproduces the
pre-fix library exactly -- verified two ways: the patched arm's transmitted
amplitudes equal this file's `_rephase(w_alt = 0)` reconstruction to `< 1e-15`,
and its per-order residual against the staircase reads the pre-fix number.  On
that arm `sha256(R)`, `sha256(T)`, `sha256(jones_reflection)` and
`sha256(orders)` are byte-identical to the shipped ones while
`sha256(jones_transmission)` differs -- the defect's whole signature, asserted.

**Cross-engine.**  The claim "a layer moves between the engines unchanged" now
holds on the transmission side too:

| | oblique 25 | conical 25-40 |
|---|---|---|
| pure (`n_modes = 4`, `n_orders = 3`) vs hybrid (`n_orders = 5`), per order | **1.299e-02** | **9.955e-03** |
| the same with the hybrid's anchor removed | 1.113e+00 (85.7x) | 8.912e-01 (89.5x) |
| the hybrid's OWN `n_orders` 5 -> 7 step (the bound) | 5.981e-03 | 4.568e-03 |
| zeroth-order Jones, pure vs hybrid | 1.044e-02 | 5.750e-03 |

i.e. the two engines now sit **2.17x / 2.18x** of the hybrid's own truncation
step apart -- discretization, not phase.

---

## 7. TESTS

### 7.1 `tests/unit/test_fix_hybrid_slant_transmission_anchor.py` (NEW, 26 tests)

Fixture: `px = py = 1.0 um`, `wl = 0.68 um`, a 6-pixel x-ASYMMETRIC cell
(`eps` 1.15 .. 3.24 on a 1.44 ground), `d = 0.5 um`, `slant = (1.0, 0)` -- a
HALF-period walk -- at oblique 25 and conical 25-40, `n_orders = 5`.  Oracle:
the same engine's z-staircase at K = 3 / 5 / 15 of 60-pixel cells.

| test | what it decides |
|---|---|
| `a1_transmitted_amplitudes_are_lab_referenced` [2] | shipped inside the oracle's own coarsest-rung uncertainty; `none/shipped` and `conj/shipped` > 20x; conj worse than none |
| `a2_the_unanchored_arm_does_not_improve_when_the_oracle_refines` [2] | shipped improves > 2x first-to-last; the un-anchored arm moves < 10% |
| `a3_no_single_global_phase_can_replace_the_per_order_factor` [2] | best global / shipped > 10x |
| `a4_zeroth_order_jones_transmission_carries_it_too` [2] | `none/shipped` > 20x; `|P0| = 1`; `|arg P0| > 1` rad |
| `a5_the_fixture_closes_where_it_is_documented_to` | `1.0 <= sum R+T < 1.05` on six stacks |
| `a6_the_anchor_is_unimodular_on_every_order` | `max ||P_m| - 1| < 1e-13` over 121 orders, some evanescent |
| `b1_pure_and_hybrid_transmitted_jones_now_agree` [2] | cross-engine gap < 10x the hybrid's own `n_orders` step; pre-fix arm 20x worse |
| `c1_pre_fix_arm_moves_transmission_only` [2] | THE FAIL-BEFORE: R / T / reflection sha256-identical, transmission not |
| `c2_efficiencies_and_reflection_track_the_staircase` [2] | R and the reflection Jones inside the oracle's own step |
| `c3_a_slanted_uniform_layer_is_byte_identical_to_the_vertical_film` | sha256 equality + the `> 1.0` cost of anchoring it |
| `c4_a_constant_tile_slanted_layer_is_byte_identical_too` | the helper AND the modal build agree, on both sides |
| `c5_a_vertical_stack_takes_no_anchor_at_all` | structural zero |
| `d1_two_slanted_layers_sum_their_walks` [2] | the split identity + the half-sum and no-sum arms at 20x |
| `d2_a_uniform_film_below_needs_no_round_trip_phase` [2] | the reflection inside the step, both round-trip factors 10x above it |
| `d3_the_walk_is_the_layer_sum_and_nothing_else` | a five-layer stack mixing every shape; a pure decision, no solve |
| `e1_a_patterned_layer_below_a_slanted_one_rides_the_shear` | the SCOPE statement of S3.2, pinned |
| `f1_the_frame_referenced_surfaces_are_only_the_transmission_ones` | the census as a gate, with a VERTICAL control proving the refusals are about the slant |
| `f2_a_slanted_patterned_layer_refuses_the_JAX_path` | S4.1's second silent-wrong, three arms |

**27 passed in 63.5 s (WIN)**; the 26-test version before `f2` read 63.9 s
(WIN) / 57.8 s (WSL).  Slowest test **15.3 s** against a 40 s cap; file budget
3 min.  `.test_durations` spliced by dict union: 12443 -> **12470** keys, **27
added, 0 dropped**, and the file re-sorted so the whole-branch diff is 27
insertions and no deletions.

The file filters `_warn_stack_energy`'s `R + T > 1` tripwire at module scope --
`n_orders = 5` leaves the closure at 1.008 .. 1.014 on this fixture -- and pins
that reading as an assertion instead (`a5`), so the warning is accounted for
rather than buried.

### 7.2 `tests/unit/test_pmm2d_staggered_slant.py` -- the four restated bars

Task 4 of the brief; `p5_restated_bars.py` imports the fixtures FROM the test
file, so these re-measure the file's own bars.  WIN and WSL agree to every
printed digit on all of them except the wall clock.

| # | bar | was | now | measurement |
|---|---|---|---|---|
| 1 | `cost_min_ratio` docstring range | "0.96x .. 1.07x" (an UNLOADED snapshot; the verification read 1.167 / 1.105 under load) | a table of four arms and their load states, envelope **0.87x .. 1.17x** | WIN idle 0.969 / 0.932 / 0.937 / 1.094 (M 4..7); WSL idle 0.986 / 0.869 / 0.915 / 0.939; WIN under a co-running suite 1.006 / 0.905 / 0.929 / 0.922; under heavy load 1.167 (WIN) / 1.105 (WSL).  **Assertion unchanged** (min of two rungs vs 2.0x, which the loaded envelope clears by 1.7x) |
| 2 | `m4c_*_against_ratio` | `against[1] > 0.8 * against[0]` -- 0.87511 / 0.98581, a **9%** margin on a staircase's geometric error ratio | an ORDERING with no constant (`against_ratio > with_ratio`) PLUS a 1.5x separation | with 0.35814 / 0.45288, against 0.87511 / 0.98581, separation **2.443 / 2.177** (margin 1.63x / 1.45x) |
| 3 | `m4_*_plus_monotone` | `plus[0] > plus[1] > plus[2]` -- rung-by-rung strictness on a CROSS-ENGINE ladder whose rungs are 0.7e-02 apart while the hybrid's own `n_orders` step is 1.26e-02 .. 1.73e-02 | first-to-last > 1.5x plus the unchanged endpoint floor | first/last **2.1777 / 2.1957**; per-rung steps 1.396, 1.560 / 1.492, 1.472 |
| 4 | `b3_conj_over_none` | docstring said "about TWICE" while the bar is 1.5 and the SMALLEST row is 1.79 | the measured RANGE stated, bar unchanged | conj/none over all 12 rows = **1.7878 .. 1.9920**; shipped worst 2.637e-05; none smallest 1.434e-01 |

Plus the **D3 scope note** in `test_b2_uniform_layer_at_any_slant_is_a_noop`:
its `1e-04` bar is a `<= 35 deg` statement -- the verification measured
`1.471e-04` at 60 degrees on the transmission Jones, with a ladder falling
6.88e-03 -> 1.47e-04 -> 1.59e-06 -> 1.21e-08 -> 7.01e-11 over M 4..8.  Recorded
rather than papered over by raising the bar.

**70 passed in 117.3 s** after the restatements (unchanged count).

### 7.3 `lumenairy/elements/pmm/stack2d_pure.py` -- D3 and D4

Module docstring only, no code: the `<= 35 deg` scope of the uniform-null bar,
and D4 -- a slanted UNIFORM layer is MATERIALISED on the union grid and takes
its own `4 q^2` solve while a vertical one rides the shared eps-free geometric
eig, so slanting a uniform spacer costs about two decades of accuracy at fixed
`M` (`3.4e-04` against the single-layer null's `4.1e-06`) and converges only
algebraically.

---

## 8. SUITE RUNS

WIN, `OMP` = `OPENBLAS` = `MKL` = 1, on the eleven files that touch this code --
the five named in the brief plus every test file importing `PMM2DStackHybrid`
(`grep -rl --include='*.py' PMM2DStackHybrid tests/`):

| file | tests |
|---|---|
| `test_fix_hybrid_slant_transmission_anchor.py` (NEW) | 27 |
| `test_pmm2d_slant_metric.py` | 25 |
| `test_pmm2d_staggered_slant.py` | 70 |
| `test_v5_14_0_pmm2d_stack.py` | 8 |
| `test_p2c_pmm2d_stack_cascade.py` | 34 |
| `test_audit_dynameta_consumer_api_2.py` | 12 |
| `test_niche_audit_w7_pmm.py` | 137 |
| `test_p2t_pmm2d_tree_cascade.py` | 57 |
| `test_pmm2d_oop_block_eig.py` | 11 |
| `test_pmm_m3_efficiency.py` | 46 |
| `test_v5_21_delta_audit.py` | 14 |
| **total** | **441 passed, 0 failed, in 1580.32 s (26:20)** |

The same eleven files on the tree BEFORE the S4.1 JAX guard read **440 passed
in 1591.32 s** -- the one extra test is `f2`, and no other count or result
moved.  The 72 warnings are all pre-existing: the `PMM2DStack is a TRANSITIONAL
alias` deprecation and `_pmm_union_grid`'s near-coincident-wall snap notice,
both raised on purpose by their own tests.

WSL, the three files this branch touches:
**122 passed in 234.80 s** (27 + 25 + 70).

The slowest test in the sweep is 416.6 s
(`test_audit_dynameta_consumer_api_2.py::test_b2_pure_amplitudes_vs_rcwa_conical`),
pre-existing and unrelated; the slowest test this branch adds is **15.3 s**
(`test_b1_pure_and_hybrid_transmitted_jones_now_agree`), against a 40 s cap.

`ruff check lumenairy/ tests/` -- **All checks passed!**  (`validation/` is in
`pyproject.toml`'s `extend-exclude`, so the probe directory is out of lint scope
by configuration.)

---

## 9. COMMITS

On `fix/hybrid-slant-transmission-anchor` (off `9b36ded`).  Nothing merged,
pushed, tagged or version-bumped.

| commit | what |
|---|---|
| `0ebd632` | `fix(pmm2d hybrid)`: the per-order frame-anchor phase on the transmitted amplitudes; `_layer_enters_slant_frame` + `_slant_frame_walk` + the `solve()` site + the `add_layer` docstring |
| `275fefa` | `test(pmm2d hybrid)`: `tests/unit/test_fix_hybrid_slant_transmission_anchor.py` (26 tests) + the `.test_durations` splice |
| `4f1ce7d` | `test(pmm2d staggered slant)`: the four restated bars + the D3 scope note |
| `02c6ab3` | `docs(pmm2d pure)`: D3 / D4 in the module docstring |
| `81a01f3` | `test(probes)`: `validation/probe_fix_hybrid_slant_anchor/` -- five probes, the README, every JSON, both builds |
| `5dba5ee` | `fix(pmm2d hybrid)`: refuse a SLANTED patterned layer on the JAX dispatch (S4.1) + its three-arm gate |
| `4400df7` | `docs`: this document + the CHANGELOG `[Unreleased]` section + the `.test_durations` re-splice |

---

## 10. OPEN ITEMS

| # | severity | what |
|---|---|---|
| **O1** | scope, cross-engine inconsistency | The hybrid ACCEPTS stacks `PMM2DStackPure._check_stack_slant` refuses -- a PATTERNED layer below a slanted one, and PATTERNED layers at different slants -- and for the first of those it silently solves the shear-CONTINUED solid (S3.2, measured, pinned by test).  A refusal mirroring the pure engine's would make the two engines agree, but it is a behaviour change beyond this fix and would break stacks that run today.  **Decision needed**, not a measurement. |
| **O2** | stability, pre-existing | The slanted-layer-over-a-film cascade blew up to `sum R + T = 2.6e+27` on the FIRST fixture tried here (`px = 1.2 um`, a cell containing `eps = 1.0` = the superstrate, a `+1` order at `\|alpha\| = 0.989` against a cut-off of 1.0) -- and only at some `(n_orders, mount)` pairs, while the SINGLE-layer rows on the same fixture were fine.  A second, lower-contrast fixture blew up at a different pair.  This is the documented near-degenerate layer<->region mode match, but a slanted layer's generalized cascade appears to meet it more readily than a vertical one, and `stabilize=` is not available on the stack entry.  Not investigated here. |
| **O3** | ~~open~~ **FIXED here** | A slanted patterned layer reaching the JAX dispatch (through a traced thickness, wavelength, angle or half-space index) got the VERTICAL answer, silently and energy-conserving, with `R` and `T` wrong.  Measured, refused and gated -- S4.1.  Listed here because it is a SECOND defect this fix uncovered rather than part of D1, and a reviewer may want it in its own commit (`5dba5ee`). |
| **O4** | open, inherited | The parity accelerator on slanted cells (verification O1) and the `_slant_is_zero` exact-zero usability note (verification O2) are untouched by this fix. |
