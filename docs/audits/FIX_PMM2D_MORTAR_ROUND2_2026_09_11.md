# FIX -- round 2 of the PURE staggered 2-D PMM per-layer grids (L2 mortar)

**Date** 2026-09-11 · **Worktree** `C:/tmp/lum_mortar2`, branch
`fix/pmm2d-mortar-round2` off `wave2/pmm2d` `1a142dd` (the mortar VERIFICATION
merge) · **Scope** `lumenairy/elements/pmm/twod_staggered.py`,
`lumenairy/elements/pmm/_core.py`,
`lumenairy/elements/pmm/stack2d_pure.py` (docstrings)

**Fixes** defects **D1**, **D2** and **D3** of
`docs/audits/VERIFY_PMM2D_STAGGERED_MORTAR_2026_09_11.md`, and applies that
audit's **D4** corrections to `BUILD_PMM2D_STAGGERED_MORTAR_2026_09_11.md`.
**D5** was already closed on the verification branch.

**Reproducers** `validation/probe_pmm2d_mortar_round2/` (seven probes, a
README and the JSON every table below is read from).  Every script asserts
which `lumenairy` it imported.

**Binding** `docs/TESTING_STANDARDS.md`.

---

## S0. Summary

| | |
|---|---|
| **D1** | An intra-layer SLIVER under a mortar is **not a wandering wrong answer and has no onset** -- it is a **FLOOR under the `n_modes` ladder**, energy-invisibly.  Fixed by a MINIMUM SEGMENT WIDTH **contract** at `Basis1D.__init__`: `_STAG_MIN_SEG_FRAC` = **1e-3 of the period**, 2.10 decades below the narrowest segment any ordinary shipped geometry asks for and landing on the same width as the independent conditioning bar. |
| **D2** | The three 2-D mortar `np.linalg.solve` calls now go through `_guarded_mortar_solve` -- **bit-identical** (`lu_factor` + `lu_solve` is the same LAPACK pair, measured on 106 solves at 0.96x wall time) with a free LAPACK `gecon` screen.  Bar `_MORTAR_RCOND_REFUSE` = **1e-12**, healthy population **2.61e-07 .. 3.77e-04**. |
| **D3** | `_stag_fourier_projection`'s Gauss rule is now sized PER SEGMENT from that segment's own half-phase.  Long-segment kernel error **7.5e-04 -> 4.7e-15**; the INTEGER path bypasses the formula and is bit-identical by construction. |
| **the ~1850 site** | **LEFT UNGUARDED, by measurement.**  Reachable with a near-singular operand, but every reading below the bar is already refused by the shipped 1-D sliver guard, and the CORRECT 1-D population comes within **1.0 decade** of a 1e-12 bar against 5.4 decades on the mortar path. |
| **Bit-identity** | **33 fixtures / 87 sha256 hashes**, each build against a pristine `git archive HEAD` tree of the SAME build: **0 mismatches on WIN, 0 on WSL**. |
| **Tests** | `tests/unit/test_fix_pmm2d_mortar_round2.py`, **15 tests**, 83.5 s (WIN, uncontended); with `test_pmm2d_staggered_nonuniform.py`, 31 passed in 188.5 s (WSL). |
| **Regression** | the twelve required suites: **310 passed, 0 failed on BOTH builds** (S7.1). |

---

## S1. The two builds

Every measured table below was taken on both, and the reading of each is
stated where they differ.

| | Windows (WIN) | WSL (Ubuntu) |
|---|---|---|
| python | 3.14.6 | 3.12.3 |
| numpy | 2.4.4 | 2.4.6 |
| scipy | 1.17.1 (scipy-openblas) | 1.17.1 (scipy-openblas) |
| threads | `OMP/OPENBLAS/MKL_NUM_THREADS = 1` | same |

The BLAS family is the same on both, so every cross-build spread quoted here
is a LOWER bound -- the same caveat the build and the verification carry.

---

## S2. D1 -- what the defect actually is

### 2.1 The verification's instrument confirms the defect but not its shape

The verification's reproducer (S6.3 there) is a 2-D pillar sandwich at
`n_orders = 1`, scored against its own `delta = 3e-1` reading.  Re-run here
(`r1_onset.py ladder`, JSON `r1_onset_ladder.json`) that fixture's OWN
reference -- the same stack with the middle layer on a ONE-segment grid, which
is the same device -- has an `M`-ladder self-gap of **9.156e-04** between
`M = 7` and `M = 8`, and its `M = 4..7` errors at a HEALTHY `delta = 0.30`
read 1.04e-03 / 6.46e-03 / 1.34e-03 / 1.62e-03: **non-monotone, and within an
order of magnitude of the effect being measured.**

Two things follow, and both matter.  The 7.8e-03 excursion is 8.5x that
self-gap, so **a real component IS present** -- the verification's defect is
not an artefact.  But the excursion's SHAPE -- "wandering", "not even
monotone at `M = 8`" -- is not separable on that fixture from the device's own
slow, oscillatory convergence, which is non-monotone at a healthy wall
spacing too.  So the fixture was rebuilt to make the attribution exact, and
the rebuilt one says the shape is a FLOOR rather than a wander (S2.3), and
that the size is 3.4-6.3x rather than 8 %.

### 2.2 The fixture that does separate them

`r3_ladder.py`, `r5_conv.py`.  Three x-strip layers, **constant along `y`**,
with the MIDDLE one **ALL HOST** on its own grid whose two walls sit `delta`
apart.  Then:

* the device **cannot depend on `delta` at all** -- the walls are element
  boundaries in a continuous medium;
* being y-uniform, the **exact 1-D pure PMM** (`PMMStack` at degree 14, whose
  own degree-12 self-gap is measured at each fixture) is the truth for the
  WHOLE 2-D answer, exactly as the shipped `test_n4_*` gate uses it;
* the y grids are IDENTICAL on all three layers, so the only non-conformity is
  in `x`.

### 2.3 THE ONSET MAP -- there is no onset; there is a FLOOR

`r5_conv.py conv`, fixture (`P` = 0.9 um, `wl` = 0.6, `theta` = 0.20,
`eps` = 6.0 / 2.25, `t` = 0.10, walls 0.21/0.68 and 0.33/0.79), oracle
self-gap **1.690e-06**:

| `delta` (of the period) | `M`=4 | 5 | 6 | 7 | **8** | 7->8 | closure at `M`=8 |
|---|---|---|---|---|---|---|---|
| **3e-01** (ordinary) | 2.368e-02 | 2.424e-02 | 4.942e-03 | 1.380e-03 | **1.179e-04** | **11.71x** | 7.99e-08 |
| 1e-02 | 2.408e-02 | 3.730e-02 | 8.091e-03 | 1.543e-03 | **4.047e-04** | 3.81x | 8.0e-08 |
| 1e-03 | 2.438e-02 | 3.997e-02 | 9.130e-03 | 1.652e-03 | **5.314e-04** | 3.11x | 8.01e-08 |
| 1e-04 | 2.451e-02 | 4.321e-02 | 1.012e-02 | 1.691e-03 | **6.365e-04** | 2.66x | 8.03e-08 |
| 1e-05 | 2.457e-02 | 5.016e-02 | 1.074e-02 | 1.711e-03 | **7.088e-04** | 2.41x | 8.04e-08 |
| 1e-06 | 2.459e-02 | 5.374e-02 | 1.099e-02 | 1.719e-03 | **7.420e-04** | 2.32x | 8.05e-08 |

**Three readings, and they change what D1 is.**

1. **There is no onset.**  The error grows smoothly and MONOTONELY in
   `log(delta)` and SATURATES: 3.4x / 4.5x / 5.4x / 6.0x / 6.3x the ordinary
   arm at `M = 8`.  Nothing is non-monotone once the fixture's own
   convergence is not part of the measurement.
2. **It is a FLOOR, not a wander.**  The per-rung improvement collapses from
   **11.71x to 2.32x**; the `M = 8` reading saturates at ~7.4e-04 while the
   ordinary arm's is still falling.  Raising `n_modes` still helps in
   absolute terms and never removes the floor -- which is a measurement, and
   it is why "raise the modal count" is NOT offered as a remedy in the
   refusal.
3. **It is ENERGY-INVISIBLE.**  The closure is PINNED at 8.0e-08 across the
   whole ladder -- 5.8 decades under the engine's own `_STAG_CLOSURE_TOL`
   (5e-2) -- so `_warn_stag_closure` and the 1-D sliver fix's `R+T` screen
   have nothing to see.  This is exactly the shape `TESTING_STANDARDS.md`
   calls the most dangerous.

The second fixture (`r3_ladder.py`, `P` = 1.2, `wl` = 0.85, `theta` = 0.15,
`eps` = 9.0/2.25, walls 0.2371/0.6183 and 0.3117/0.7402) is more sensitive:
at `M = 6` its error runs **7.294e-03 -> 4.990e-02** over `delta` 3e-1 -> 1e-5
(**6.8x**), against a HEALTHY control at ordinary wall spacings of 0.50 / 0.30
/ 0.20 / 0.12 that reads 2.423e-02 / 7.294e-03 / 8.114e-03 / 1.114e-02.  That
control matters: it says the effect is NOT simply "a less balanced partition
is less accurate" -- an ordinary 0.12 partition costs 1.5x, the sliver 6.8x
and rising.

### 2.4 ATTRIBUTION -- the mortar's ALGEBRA is exact; what it loses is STRUCTURE

`r7_allhost.py pure`, and this is the sharpest measurement in the round.
Three **ALL-HOST** layers on three DIFFERENT non-uniform grids: the device is
a homogeneous slab whose reflectance is **analytic** (the symmetric-slab Airy
form, which needs only the interface reflection coefficient and so is free of
the `t = 1 + r` convention that is false for TM), and a homogeneous layer is
**exactly representable on any element grid**, so every digit of the deviation
is the mortar's cross-grid projection.

| `delta` | `M`=4 | 5 | 6 | 7 | closure at `M`=7 |
|---|---|---|---|---|---|
| 3e-01 | 7.597e-08 | 1.169e-10 | 1.287e-13 | 8.826e-14 | 8.0e-14 |
| 1e-01 | 1.380e-07 | 3.239e-10 | 3.272e-13 | 5.707e-14 | 7.3e-14 |
| 3e-02 | 2.013e-07 | 4.963e-10 | 6.770e-13 | 1.086e-13 | 7.4e-14 |
| 1e-02 | 2.317e-07 | 5.529e-10 | 8.687e-13 | 8.349e-14 | 6.0e-14 |
| 1e-03 | 2.467e-07 | 5.805e-10 | 9.602e-13 | 1.187e-13 | 7.8e-14 |
| 3e-04 | 2.476e-07 | 5.828e-10 | 9.733e-13 | 9.570e-14 | 4.0e-14 |

**The mortar reproduces the analytic slab to 9.6e-13 at `M` = 6 at EVERY wall
separation, and the residual is flat in `delta` to within a factor 3.3.**  So
the mortar's algebra is not what D1 breaks.  What a degenerate grid loses is
the **structured trace of a PATTERNED neighbour** -- which is why the
fail-before fixture needs patterned outer layers to show the effect at all,
and why the all-host arm of the verification's own S6.3 (a middle layer that
is all host BETWEEN PATTERNED NEIGHBOURS) does show it.

The NO-MORTAR control on the same fixture (all three layers on the sliver's
own grid, so every interface is a plain square match) reads
**5.1e-09 / 7.7e-12 / 1.1e-11** at `M` = 4 / 5 / 6 down to `delta` = 1e-06 --
confirming the verification's 4.1e-13 `delta`-independence on the shipped
library, from a different direction.

### 2.5 THE MECHANISM, measured -- and why the guard is a WIDTH bar

`r1_onset.py cond`.  The sliver's spurious spectrum is FREE: with
`J = w / 2`,

```
    |gamma| / k0  =  c(M) . M (M + 1) / (4 k0 J)
```

| `delta` | `k0 J_min` | `c` at `M` = 4 | `c` at `M` = 6 |
|---|---|---|---|
| 3e-01 | 1.331e+00 | 1.5316 | 1.5970 |
| 1e-01 | 4.435e-01 | 1.0772 | 1.1078 |
| 3e-02 | 1.331e-01 | 0.9647 | 0.9963 |
| 1e-02 | 4.435e-02 | 0.9327 | 0.9675 |
| 1e-03 | 4.435e-03 | **0.9181** | **0.9550** |
| 1e-04 | 4.435e-04 | **0.9167** | **0.9537** |
| 1e-05 | 4.435e-05 | **0.9165** | **0.9536** |
| 1e-06 | 4.435e-06 | **0.9165** | **0.9536** |

`c` is stable to **four digits over five decades** while `|gamma|` itself
moves 1e+05x.  **That is what settles the guard's shape.**  The spectrum is a
pure function of the wall array and `M`, so a SPECTRAL bar carries no
information a WIDTH bar does not -- and it costs a region eig to read.  Worse,
it would refuse legitimately fine UNIFORM lattices: measured here, a uniform
`N = 5` lattice (segments 0.20 of the period) reads `|gamma|/k0` = **9.84** at
`M` = 4 against a physical ceiling of 3.0, while a `delta` = 3e-02 sliver -- a
SIX-TIMES finer geometry -- reads **39.8**.  Four times apart on geometries
six times apart: any bar placed between them refuses a uniform lattice for
being fine, which is precisely how the 1-D `|q|max` bar failed
(`FIX_PMMSTACK_SLIVER_WALLS_2026_09_11.md` S3.3, where the populations
OVERLAP across degrees).

**Which operator carries the `1/w`**, fitted exponents in `1/delta` over the
tail `delta <= 1e-2`, on TWO fixtures (`v_s63` / `alt`) and TWO builds:

| operator | fx 1, WIN / WSL | fx 2, WIN / WSL |
|---|---|---|
| `\|gamma\|max` of the sliver grid | 0.999 / 0.999 | 0.999 / 0.999 |
| `W_b` (the E-field modal matrix) | **0.029 / 0.029** | **-0.075 / -0.069** (FLAT) |
| `V_b` (the H-field modal matrix) | **2.028 / 2.028** | **1.950 / 1.951** |
| `MassE_B W_B` (the E-row solve) | 1.995 / 1.995 | 1.994 / 1.994 |
| `MassH_A V_A` with the SLIVER in `A` (the H-row solve) | **3.218 / 3.003** | **3.214 / 3.103** |
| cross-mass factor `C1_x` | 1.025 / 1.025 | 1.284 / 1.284 |
| cross-mass factor `C2_x` | 2.303 / 2.303 | 2.200 / 2.200 |

Every exponent agrees between the builds to the digits printed EXCEPT the
H-row one, and the reason is stated rather than smoothed: that fit's
`delta = 1e-6` point sits at the float64 ceiling (`cond_2` reads 2.3e+19 and
7.4e+19 on the two fixtures), so its last decade is measuring the arithmetic
rather than the operator.  The claim it supports -- that the H row grows
FASTER than `1/w^2` because it multiplies the sliver grid's own mass by the
sliver grid's own `V` -- is carried by every rung above that one and by both
builds.

**The `1/w^2` lives in the `V` half**, exactly as the 1-D verification located
it (`VERIFY_PMMSTACK_SLIVER_WALLS_2026_09_11.md` claim 2b: `cond(V)` fits
p = 2.000, `cond(W)` is flat).  The H row carries `1/w^3.2` because it
multiplies the sliver grid's own mass by the sliver grid's own `V`.  The
CROSS-MASS is NOT innocent here either (`C2_x` fits 2.2-2.3), which is the
difference from the cross-layer case the 1-D fix calls safe: there the
near-coincident walls appear only in the integration mesh
(`cond(C_bb^x)` measured CONSTANT at 9.43e+01), here the sliver IS a spectral
element of one of the two grids.

### 2.6 The SHARED-grid path CANNOT be driven into the band

The verification could not finish its shared-lattice control.  It is settled
by construction, and measured (`r1_onset.py shared`): `PMM2DStackPure` with
`layer_grids='shared'` **REFUSES both `x_walls` and `grid`** --

```
ValueError: PMM2DStackPure.add_layer: x_walls, y_walls are only meaningful
with layer_grids='per-layer' ...
```

-- and the shared lattice is `np.linspace(0, d, N+1)`, whose aspect ratio is
exactly 1.0 at every `N` (measured at `N` = 3/5/8/13/24).  There is no API
path to a non-uniform cell on the shared grid, so the guard does not apply
there and nothing on that path changes.

---

## S3. D1 -- what ships

### 3.1 The contract

`twod_staggered.py`, in `Basis1D.__init__`'s ARRAY branch only:

```python
PMM2D_STAG_MIN_SEG_GUARD = True      # fail-before switch
_STAG_MIN_SEG_FRAC       = 1.0e-3    # of the period
_STAG_SEG_CENSUS         = None      # instrument
```

A non-uniform wall array whose narrowest segment is below the bar raises,
naming the offending SEGMENT INDEX, its width (absolute and as a fraction of
the period), the whole wall array, the spurious spectrum in units of
`G = 2 pi / d`, and FOUR remedies:
merge the walls; carry the fine feature on `layer_grids='shared'`; use
`PMM2DStackHybrid` (Fourier-projected, no element grid); or lower `n_slices` /
stop a taper before its tip closes.  **"Raise `n_modes`" is explicitly NOT
offered**, and the message says so -- S2.3 reading 2.  (Naming the OTHER
grid is the conditioning backstop's job, S4.4: `Basis1D` is built before its
mortar partner exists.)

The INTEGER path is EXEMPT and that costs nothing: a uniform lattice's
segments are all `d/N`, so reaching the bar needs `N > 1000`, i.e. `q >= 3000`
at `M = 4` and a `2 q^2 = 1.8e+07`-dimension region eigenproblem.  The
exemption is also what keeps gate N1's bit-identity claim unconditional.

### 3.2 THE BAR, both sides measured

**FALSE-POSITIVE side.**  Every per-layer geometry class the library builds,
through the public API, with the narrowest segment read off the stack's own
wall arrays (test `test_the_minimum_segment_bar_clears_every_geometry_the_library_builds`,
which re-measures this on the running build):

| geometry | narrowest segment / period | x the bar |
|---|---|---|
| a single interior wall at `0.4` | 4.0000e-01 | 400.0 |
| duty-1/3 pair | 3.3333e-01 | 333.3 |
| conforming / non-conforming (`0.2371, 0.6183` vs `0.3117, 0.7402`) | 2.3710e-01 | 237.1 |
| axes carrying different walls (`0.21, 0.55` x `0.33, 0.78`) | 2.1000e-01 | 210.0 |
| the mortar suite's taper, `n_slices` = 4 / 8 / 16 / 32 / 64 | 2.0035e-01 .. 1.8812e-01 | 200.3 .. 188.1 |
| `add_tapered_pillars`, 6 slices | 1.7333e-01 | 173.3 |
| **nested refinement (`0.125, 0.25, 0.75, 0.875`) -- the WORST ordinary one** | **1.2500e-01** | **125.0** |
| closing taper, `n_slices` = 8 | 3.1438e-02 | 31.4 |
| closing taper, `n_slices` = 32 | 8.0094e-03 | 8.0 |
| closing taper, `n_slices` = 64 | 4.1047e-03 | 4.1 |

**2.10 decades** of gap to the narrowest ORDINARY geometry.  The ONE surface
that walks toward the bar is a taper whose tip CLOSES -- the midpoint rule's
narrowest SAMPLED width is `~ w_bottom / (2 n_slices)`, so a pillar closing
from half the period crosses at about **250 slices**.  That is reported, not
hidden: it is remedy (4) in the message, it is in `add_tapered_pillar`'s
docstring, and the shipped test asserts the RATE (so it tracks any change to
the slicing rule) rather than the numbers.

**TRUE-POSITIVE side, and it is a consistency check.**  The bar was derived
from ACCURACY (S2.3).  The mortar's H-row operator with the sliver in the `A`
slot reads LAPACK `rcond` = **7.34e-13 at exactly `w/d` = 1e-03, `M` = 6** --
below the 1e-12 refusal of the INDEPENDENTLY derived conditioning bar of S4.
Two derivations, one from digits and one from error against an exact oracle,
land on the same width:

| `w/d` | `rcond(MassE_B W_B)`, sliver = B | `rcond(MassH_A V_A)`, sliver = A |
|---|---|---|
| 1e-02 | 2.610e-07 | 7.158e-10 |
| 3e-03 | 2.754e-08 | 2.000e-11 |
| **1e-03** | 2.989e-09 | **7.342e-13** |
| 3e-04 | 2.601e-10 | 2.058e-14 |
| 1e-04 | 3.113e-11 | 7.471e-16 |
| 3e-05 | 3.072e-12 | 2.153e-17 |

(`M` = 6; the `M` = 4 and `M` = 5 rows are in `r4_screen.json` and the probe
log.)  Below the bar the operator collapses fast, and at 1e-07 the unguarded
`solve` raised outright.

**The bar is `M`-independent and the conditioning is not**, and the layering
is deliberate: `rcond` falls about 3x per modal rung (S4.2), so above `M ~ 6`
the conditioning backstop is the one that fires first on a marginal grid.  A
fixed WIDTH contract cannot know `M`, the wavelength or the contrast; it is
the cheap, documented, solve-free half, and the backstop names the same
remedies.

### 3.2a WHAT THE GUARD DOES NOT DO, stated plainly

The accuracy cost is CONTINUOUS in the wall width (S2.3), so a bar that
refuses at 1e-3 does not restore accuracy above 1e-3 -- it refuses where the
operator runs out of digits, and the band between an ordinary partition and
the bar carries a real, unrefused cost:

| narrowest segment | 3e-01 | 1e-01 | 3e-02 | 1e-02 | 3e-03 | 1e-03 (the bar) |
|---|---|---|---|---|---|---|
| err at `M` = 6, fixture 2 | 7.29e-03 | 1.24e-02 | 1.95e-02 | 2.59e-02 | 3.14e-02 | 3.61e-02 |
| x the ordinary arm | 1.0 | 1.7 | 2.7 | 3.6 | 4.3 | 5.0 |

and the ORDINARY control at 0.12 -- a perfectly reasonable partition nobody
would refuse -- already reads 1.114e-02, i.e. 1.5x.  **So part of that band is
just "a less balanced partition is less accurate", which is a property of the
method and not a defect**, and the rest shades continuously into the sliver
regime.  There is no width at which the two can be separated, which is why the
bar is placed on DIGITS rather than on a chosen accuracy multiple, and why the
`n_modes` ladder -- not this guard -- remains the user's instrument.

The concrete case a user meets is the closing taper: at `n_slices` = 64 its
narrowest sampled segment is 4.1e-03, **above the bar and therefore returned**,
with a floor of roughly 4x on its `n_modes` ladder.  That is a documented cost
of slicing a taper to its tip, not something the guard removes; it is why
`add_tapered_pillar`'s docstring now carries the arithmetic.

### 3.3 FAIL-BEFORE, two-sided

`test_fail_before_the_sliver_the_guard_refuses_is_measurably_wrong`.  With
BOTH round-2 guards disarmed (they land on the same width, so a demonstration
of what round 2 prevents has to lift both), on fixture 2 at `M` = 6:

| arm | err vs the exact oracle | closure | warnings |
|---|---|---|---|
| ordinary (`delta` = 0.30) | 7.294e-03 | 1.5e-05 | none |
| **sliver (`delta` = 1e-05)** | **4.990e-02** | 7.9e-06 | **none** |

**6.84x worse, with the closure 3.5 decades UNDER the engine's own tripwire
and no warning on either arm.**  With the guards armed the sliver arm is
refused -- by the WIDTH contract (which needs no solve), and independently by
the CONDITIONING backstop with the width contract lifted (`rcond` = 3.168e-15
on that fixture) -- while the ordinary arm returns the identical number.

---

## S4. D2 -- the mortar's own solves

### 4.1 Two instruments tried, one REFUTED

| instrument | verdict | measurement |
|---|---|---|
| residual `\|\|AX-B\|\|` | not usable (M1's finding, restated) | a backward-stable `gesv` leaves ~eps whatever the conditioning; the build doc's argument for it is right, and does not cover a SINGULAR operator |
| the FREE rigorous lower bound `\|\|A\|\|_F \|\|X\|\|_F / (sqrt(n) \|\|B\|\|_F) <= cond_2(A)` | **REFUTED** | over 106 mortar solves it reads **0.79 .. 24.0** while the true `cond_2` runs 3.9e+02 .. 7.9e+15.  The right-hand side is the OTHER grid's trace, which carries none of the sliver's spurious content, so the ill-conditioning is never excited by this `B` |
| LAPACK `gecon` on the LU the solve already forms | **SHIPPED** | tracks the exact `cond_2` within a factor 21 across that whole range (`rcond * cond_2` in [0.047, 0.595]) |

### 4.2 Why it costs nothing, and is bit-identical

`np.linalg.solve` calls LAPACK `gesv`, which IS `getrf` + `getrs`.
`scipy.linalg.lu_factor` + `lu_solve` calls the same pair, and `gecon` is
`O(n^2)` on factors that already exist.  MEASURED:

* **bit-identity**: `lu_solve(lu_factor(A), B) == np.linalg.solve(A, B)` byte
  for byte on **all 106 mortar solves** of the probe population, **on BOTH
  builds** (`r4_screen.py`, `lu_solve_bitidentical: True` in
  `r4_screen.json` and `r4_screen_wsl.json`), plus five synthetic shapes
  (72 / 162 / 200x120 / 450 / 288x450 complex);
* **cost**: 0.0436 s vs 0.0455 s for `np.linalg.solve` on a 450x450 complex
  pair, i.e. **0.96x** -- the `gecon` call is inside the noise.

### 4.3 THE BAR, both sides measured

`_MORTAR_RCOND_REFUSE = 1e-12`.

**HEALTHY population** -- every grid pair the shipped fixtures build (the
taper's three adjacent slices at `M` = 4/5/6, non-uniform vs uniform,
non-uniform vs non-uniform, uniform 2 vs 3 and 3 vs 5, a nested refinement,
conforming, a single interior wall, and ordinary fine features down to 2 % of
the period), 106 solves: **`rcond` = 2.61e-07 .. 3.77e-04**, so the bar sits
**5.4 decades below it**.  The two builds agree to six significant figures on
every summary of that population (worst healthy `rcond` 2.6100830e-07 WIN /
2.6099070e-07 WSL; worst sliver 1.2885149e-17 on both; the `gecon`-vs-`cond_2`
ratio bracket [0.0467144056374, 0.5953032635510] on both), so the bar is not
sitting on one machine's arithmetic.

Pushed up the modal ladder on the shipped taper, where this operator
conditions worst:

| `M` | 4 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|
| worst `rcond` over the 3 interfaces | 1.623e-05 | 5.975e-07 | 2.638e-07 | 4.668e-08 | 3.894e-08 | **2.809e-08** |

Still **4.6 decades clear at `M` = 10**, which is a 1800-dimension region eig
per interface, and the readings FLATTEN rather than continuing to fall (3x per
rung to `M` = 8, then 1.2x and 1.4x).

**WRONG population**: the S3.2 table.  The bar is crossed at a 1e-03 wall
separation on the H row and the operator reaches 2.15e-17 at 3e-05.

### 4.4 The refusal

`_ConditioningError` (a subclass of `_EnergyError`, so every existing
`stabilize=` retry ladder routes around it unchanged), naming the site, the
matrix size, the measured `rcond` against the bar, the healthy population it
sits outside, **both grids' `M`, `q`, wall arrays and narrowest segments**,
and the remedies.  Wired at all three sites the verification named:
`_interface_smatrix_mortar_2d`'s E row and H row, and
`_interface_smatrix_general_mortar_2d`'s single solve (the OUT-OF-PLANE /
SLANTED per-layer path).

### 4.5 The `~1850` site -- LEFT UNGUARDED, and this is the measurement

`_interface_smatrix`'s two solves are shared by the 1-D `PMMStack`, the 2-D
pure SHARED path and the 2-D per-layer CONFORMING bypass.  Measured on a
two-layer `PMMStack` at degree 12 whose walls differ by `delta`, with the
LAPACK reciprocal condition of `Wb` / `Vb` instrumented at every call:

| `delta` | min `rcond(Wb, Vb)` | `R+T` | warnings | outcome |
|---|---|---|---|---|
| 1e-02 | 1.094e-06 | 1.000000 | 0 | correct |
| **1e-04** | **9.697e-11** | **1.000000** | **0** | **correct** |
| 1e-05 | 9.731e-13 | -- | -- | **already REFUSED by the 1-D sliver guard** |
| 1e-06 (`min_feature=0`) | 9.741e-15 | -- | -- | already REFUSED |
| 1e-07 (`min_feature=0`) | 9.744e-17 | -- | -- | already REFUSED |

Two facts settle it:

1. **Every reading below a 1e-12 bar is already refused**, and refused by a
   guard that can say MORE than a conditioning bar can -- it names
   `min_feature` and the exact manufactured cell.
2. **The CORRECT population comes within 1.0 decade of the bar** (9.697e-11,
   `R+T` = 1.000000, no warning), against 5.4 decades on the mortar path.  A
   bar with one decade of margin on a site three engines share is the S4 shape
   `TESTING_STANDARDS.md` forbids.

Pinned by `test_the_plain_1d_interface_solve_is_left_unguarded_and_this_is_why`,
which re-measures both readings on the running build.

---

## S5. D3 -- the far-field projector's quadrature order

### 5.1 The requirement, MEASURED rather than assumed

`r2_quad.py need`.  The smallest `nq` whose kernel `INT Ltilde_a(u)
e^{i omega u} du` matches a refined rule, with the bar derived AT EACH POINT
from the reference rule's own 37-node self-drift (1.4e-14 .. 3.8e-14) rather
than pinned -- because at `omega = 96` the true integral is ~1e-09 and asking
for 14 relative digits there is asking for 23 absolute ones:

| `M` \ `omega` | 0 | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 2 | 8 | 9 | 12 | 15 | 22 | 33 | 52 | 89 |
| 4 | 2 | 8 | 10 | 12 | 16 | 22 | 33 | 53 | 89 |
| 6 | 3 | 9 | 11 | 13 | 17 | 23 | 34 | 53 | 90 |
| 8 | 4 | 10 | 12 | 14 | 18 | 24 | 35 | 55 | 91 |
| 12 | 6 | 12 | 14 | 16 | 20 | 26 | 37 | 55 | 92 |

Fitted `need = a . omega + b(M)`:

| `M` | 3 | 4 | 5 | 6 | 7 | 8 | 10 | 12 |
|---|---|---|---|---|---|---|---|---|
| slope `a` | 0.6455 | 0.6441 | 0.6450 | 0.6425 | 0.6434 | 0.6413 | 0.6341 | 0.6357 |
| intercept `b` | 9.38 | 9.71 | 9.97 | 10.69 | 10.88 | 11.66 | 12.67 | 13.62 |

**A slope spread of 1.8 % across eight modal counts** -- that is what makes it
a predictor rather than a fit.  The intercept is `9.38 + 0.47 (M - 3)`.

### 5.2 The rule

```python
_STAG_QUAD_OMEGA, _STAG_QUAD_M, _STAG_QUAD_CONST = 0.72, 0.5, 10.0

def _stag_quad_order(M, omega):
    base = 2 * M + 8
    need = _STAG_QUAD_OMEGA * omega + _STAG_QUAD_M * M + _STAG_QUAD_CONST
    return base if need <= base else int(ceil(need))
```

with `omega_n = max_m |m G + alpha0| . J_n` -- the segment's OWN half-phase.
The constants are an UPPER ENVELOPE of S5.1 (0.72 against a measured 0.645,
`0.5 M + 10` against a measured `0.47 M + 8.0`).

Scored two-sided in the probe (`r2_quad.py rule`):

* it clears the MEASURED requirement of S5.1 everywhere, **worst margin 2
  nodes** over the 112 `(omega, M)` cells;
* **the INTEGER path does not reach the function at all** --
  `_stag_fourier_projection` branches on `basis.uniform` and keeps the single
  `2 M + 8` rule, one `leggauss`, one `_modleg_value_deriv`, the same doubles,
  so the bit-identity is structural.  The FORMULA additionally returns exactly
  `2 M + 8` on every uniform lattice `M = 3..14 x N = 1..60` the shipped order
  cap allows with `|alpha0| <= G/2` -- **720 cells, 0 violations** -- which is
  what keeps an explicitly-passed uniform ARRAY ULP-close to the integer
  spelling (gate N2).

**A REJECTED candidate, kept in the probe because its failure is the design
argument.**  `max(2M + 8, ceil(0.75 omega) + M + 8)` also clears the
requirement (worst margin 1 node) but returns something other than `2 M + 8`
on **457 of the same 720 cells**: its `M`-linear term is too large relative to
its constant, so the `2 M + 8` reserve runs out at large `N`, where
`omega -> pi (M - 1) / 2`.  The shipped form carries `0.5 M` and a larger
constant, which is exactly what lets it fit under the reserve everywhere.

### 5.3 The result

Relative kernel error against an 8x-refined rule, `alpha0 = 0.37 G`, the
verification's own grid (S2.6 there).  **BEFORE** is the shipped fixed rule;
**AFTER** is what ships:

| longest segment | `M`=4, `m`<=3 | `M`=4, `m`<=7 | `M`=6, `m`<=3 | `M`=6, `m`<=7 | `M`=8, `m`<=7 |
|---|---|---|---|---|---|
| 0.33 d (three EQUAL segments, spelled as an ARRAY) BEFORE | 7.7e-15 | 7.7e-15 | 6.3e-15 | 6.3e-15 | 6.7e-15 |
| 0.33 d AFTER | 7.0e-15 | 8.9e-15 | 6.2e-15 | 6.4e-15 | 1.1e-14 |
| 0.62 d BEFORE | 7.1e-15 | 2.2e-08 | 6.0e-15 | 2.0e-12 | -- |
| 0.62 d AFTER | 6.2e-15 | **6.0e-15** | 6.1e-15 | **3.5e-15** | 1.3e-14 |
| 0.91 d BEFORE | 1.0e-13 | 2.4e-04 | 6.1e-15 | 3.0e-07 | -- |
| 0.91 d AFTER | 6.2e-15 | **5.3e-15** | 6.4e-15 | **2.5e-15** | 8.1e-15 |
| **0.96 d BEFORE** | 4.6e-13 | **7.5e-04** | 6.7e-15 | 1.4e-06 | 5.6e-11 |
| **0.96 d AFTER** | 9.3e-15 | **4.7e-15** | 5.3e-15 | **2.3e-15** | 8.6e-15 |

`nq` at the worst cell goes **16 -> 29**.  Every cell is back at round-off,
and the two builds agree closely enough that the ladder is a statement about
the rule rather than about a machine: over all **36 cells**, `nq` matches
EXACTLY (0 mismatches) and the kernel errors agree to **1.24e-16**, with the
worst reading on either build **1.27e-14**.

The first row needs its label read carefully: those three segments are EQUAL,
but they are spelled as an explicit ARRAY, so they take the formula, which
hands out `nq` = 18 rather than 16 at `m <= 7` -- and the row moves by nothing
but that reordering.  **The INTEGER spelling of the same lattice is byte-
identical**, because it never reaches the formula; that is proved separately
and unconditionally in S6.

---

## S6. BIT-IDENTITY

`r6_bitid.py` + `r6_compare.py`.  ONE file, run against this tree and against
a pristine `git archive HEAD` tree in the scratchpad; every arm asserts which
`lumenairy` it imported.  Hashes are sha256 over `dtype + shape + tobytes`.

| family | fixtures | hashes |
|---|---|---|
| `_stag_fourier_projection` on **INTEGER-`N`** grids -- `(d, N, M, tau, alpha0, m_max)` = (1.2,3,5,1,0,4), (0.9,4,4,e^-0.41i,2.08,5), (1.4,6,6,e^0.77i,-3.1,7), (0.7,2,8,e^1.13i,1.7,6), (1.0,12,3,e^-0.2i,0.4,8), (1.0,1,7,e^-0.9i,5.5,2), each on the `B` and `Btilde` sets | 6 | 12 |
| the same lattices spelled as explicit UNIFORM ARRAYS | 6 | 12 |
| STACK solves -- shared path (normal / conical / `M`=6 `N`=3); per-layer conforming, non-conforming, nested, normal-incidence, `M`=6, `jones=True`, mixed uniform/patterned; both taper builders; slant; in-plane tensor; out-of-plane tensor; magnetic; `retain_internal`; `layer_absorption`; three 1-D `PMMStack` arms (shared, per-layer, conical) | 21 | 63 |
| **TOTAL** | **33** | **87** |

| comparison | result |
|---|---|
| **WIN `with` vs WIN `pre`** | **87/87 identical, 0 mismatches** |
| **WSL `with` vs WSL `pre`** | **87/87 identical, 0 mismatches** |
| WIN `with` vs WSL `with` | 20/87 identical -- and that is the EXPECTED reading, not a defect: it is the ordinary cross-BLAS last-bit spread on quantities that pass through a `zgeev` and a cascade.  The 20 that do agree are the integer ORDER arrays.  The claim bit-identity makes is per build, against that build's own pre-change tree, and it holds on both. |

The arm is not vacuous: **21 of the 33 fixtures (63 of the 87 hashes) are full
stack solves** that take the mortar, both taper builders, the generalized
cascade (slant + out-of-plane) and the 1-D cascade, and the remaining 12
fixtures (24 hashes) are the projector D3 touched directly.  Running the pristine arm from a
`git archive HEAD` copy (rather than from an in-process reimplementation) is
what makes it a statement about the SHIPPED library.

---

## S7. TESTS

`tests/unit/test_fix_pmm2d_mortar_round2.py` -- 15 tests.

| test | what it decides |
|---|---|
| `..._bar_clears_every_geometry_the_library_builds` | the FALSE-POSITIVE census, re-measured; the closing taper's RATE |
| `a_requested_sliver_is_refused_and_the_message_names_the_cure` | the refusal fires at the grid's entry point and through the public surface; the message carries the width and all four remedies and does NOT offer "raise n_modes" |
| `the_spurious_spectrum_is_a_function_of_the_wall_array_alone` | WHY a width bar and not a spectral one: `c` constant to 1.007 over three decades, and the uniform-lattice overlap |
| `fail_before_the_sliver_the_guard_refuses_is_measurably_wrong` | the pre-fix arm is 6.8x worse with the closure pinned and NO warning; the fixed arm refuses, by each guard independently; the ordinary arm is unmoved |
| `the_mortars_own_algebra_is_exact_at_every_wall_separation` | the ATTRIBUTION: against the ANALYTIC slab the mortar is exact at every wall separation |
| `the_integer_lattice_is_exempt_and_cannot_reach_the_bar` | the exemption costs nothing (arithmetic, so it tracks the constant) |
| `the_guarded_mortar_solve_returns_the_numpy_solve_bit_for_bit` | D2 moves no bit of a healthy answer |
| `the_mortar_rcond_bar_has_decades_of_gap_on_both_sides` | both D2 populations, re-measured |
| `a_singular_mortar_operator_is_refused_by_name_not_by_LinAlgError` | the named refusal, its message content, and that it is an `_EnergyError` |
| `the_free_lower_bound_on_the_condition_number_is_refuted_here` | the refuted instrument, so nobody re-proposes it |
| `the_plain_1d_interface_solve_is_left_unguarded_and_this_is_why` | the ~1850 DECISION, pinned by its two readings |
| `the_integer_lattice_projector_is_bit_identical_to_the_fixed_rule` | D3 bit-identity against an in-process reimplementation of the pre-change rule |
| `the_rule_returns_the_shipped_order_on_every_uniform_lattice` | 720 cells, 0 violations |
| `the_quadrature_rule_clears_the_measured_requirement` | the requirement re-measured with the oracle's own floor as the bar |
| `a_long_segment_kernel_is_back_at_round_off` | 7.5e-04 -> 4.7e-15, and the improvement is DECADES |

### 7.1 Runs

| suite | WIN | WSL |
|---|---|---|
| `test_fix_pmm2d_mortar_round2.py` | **15 passed**, 83.5 s | included below |
| `test_fix_pmm2d_mortar_round2.py` + `test_pmm2d_staggered_nonuniform.py` | -- | **31 passed**, 188.5 s |
| `test_pmm2d_staggered_mortar.py` + `test_verify_pmm2d_perlayer_slant.py` | **35 passed**, 298.5 s | -- |
| `test_pmm2d_staggered_nonuniform.py` | **16 passed**, 74.7 s | included above |
| **the twelve required suites together** | **310 passed, 0 failed**, 1333.8 s | **310 passed, 0 failed** (31 + 279, split across the two runs above and below), 188.5 + 1217.4 s |
| the ten of those not in the rows above | -- | **279 passed, 0 failed**, 1217.4 s |
| every test file that imports `PMMStack` (42 files) | -- | (see below) |

The two builds agree on the count exactly (310 = 310), which is the reading
that matters for a change whose whole claim is that it moves nothing on a
healthy path.
| `ruff check lumenairy/ tests/` | **clean** (also over `validation/probe_pmm2d_mortar_round2/`) | -- |

---

## S8. What was NOT done, and why

1. **A THIRD BLAS family.**  Both builds link scipy-openblas, as in the build
   and the verification.  Every cross-build spread here is a lower bound.
2. **Beyond `M = 10` on the shipped taper's conditioning ladder.**  Measured
   to `M = 10` (**2.809e-08**, 509 s, a 1800-dimension region eig per
   interface -- 4.6 decades clear of the bar).  Nothing above that was run;
   the readings flatten (4.67e-08 / 3.89e-08 / 2.81e-08 at `M` = 8 / 9 / 10)
   rather than continuing to fall 3x per rung, so the extrapolation is
   conservative, but it is an extrapolation.
3. **An `M`-DEPENDENT width contract.**  The conditioning degrades with the
   modal count and the width bar does not follow it (S3.2).  A bar of the form
   `f(M) / period` is derivable from the same data; it is not shipped, because
   a contract a user can state in one number is worth more than one they have
   to compute, and the conditioning backstop covers the remainder with an
   actionable message.  Logged as **open item A**.
4. **The `RuntimeWarning: invalid value encountered in multiply`** that the
   `4 q^2` generator emits on a grid inside the refused band (reachable only
   with the guard disarmed).  It is pre-existing, it is not silent, and the
   round-2 guards mean no supported input reaches it.  Logged as **open
   item B**.
5. **The verification's own S12 items** (JAX per-layer, `prepare()`,
   `tau`-keyed cache cost) are untouched -- they are the build's open items,
   not defects.
6. **The accuracy cost in the band ABOVE the bar is NOT removed** (S3.2a).
   It is continuous, it shades into ordinary partition quality, and the
   `n_modes` ladder remains the user's instrument for it.  Logged as **open
   item C**, with the concrete case named: a 64-slice closing taper at
   4.1e-03 of the period is returned, carrying a ~4x floor.
7. **The verification's own S6.3 fixture was not re-run at `M = 10`** (its
   S12 item 4).  It was SUPERSEDED rather than extended: its `M`-ladder
   self-gap is within an order of magnitude of the effect, so another rung on
   it cannot settle the shape.  What replaced it is an exact-oracle fixture
   run to `M = 8` and an analytic-oracle attribution run to `M = 7`.

---

## S9. Commands

```
# probes (both builds; WSL via wsl.exe -e bash -lc "...")
PYTHONPATH=$PWD OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python validation/probe_pmm2d_mortar_round2/r1_onset.py cond shared
  ... r2_quad.py need rule ladder
  ... r3_ladder.py ladder healthy taper
  ... r4_screen.py pop
  ... r5_conv.py conv
  ... r7_allhost.py pure nomortar taper

# bit-identity: the pristine arm
git archive HEAD lumenairy | tar -x -C <scratch>/pre
R6_TAG=with PYTHONPATH=$PWD                python .../r6_bitid.py
R6_TAG=pre  PYTHONPATH=<scratch>/pre \
            R6_EXPECT_ROOT=<scratch>/pre   python .../r6_bitid.py
python .../r6_compare.py with pre

# the suites
python -m pytest -q tests/unit/test_fix_pmm2d_mortar_round2.py \
  tests/unit/test_pmm2d_staggered_mortar.py \
  tests/unit/test_pmm2d_staggered_nonuniform.py \
  tests/unit/test_verify_pmm2d_perlayer_slant.py \
  tests/unit/test_pmm2d_staggered_slant.py tests/unit/test_pmm2d_staggered_oop.py \
  tests/unit/test_pmm2d_staggered_anisotropic.py \
  tests/unit/test_pmm2d_staggered_magnetic.py \
  tests/unit/test_v5_12_0_pmm2d_staggered.py \
  tests/unit/test_v5_21_pmm2d_staggered_oblique.py \
  tests/unit/test_v5_14_0_pmm2d_stack.py \
  tests/unit/test_p2c_pmm2d_stack_cascade.py

# and, because _core.py is SHARED, every file that imports PMMStack
python -m pytest -q $(grep -rl 'PMMStack' --include='*.py' tests/unit/)
```

### 9.1 Commits on `fix/pmm2d-mortar-round2`

| commit | what |
|---|---|
| `3386d74`, `b82995f` | the seven probes, the bit-identity harness, the README |
| `9989d77` | **D1** the minimum-segment contract + **D3** the per-segment quadrature order (`twod_staggered.py`) |
| `e5f7982`, `bc6e005` | **D2** `_guarded_mortar_solve` at the three sites (`_core.py`) |
| `726c81d`, `da54888`, `69652ae`, `3514c22` | the 15 gates |
| `86fdec6` | this doc, the **D4** corrections to the build doc, the CHANGELOG and the public-surface contracts |
| `a9c6367` | `.test_durations` |
| the remainder | corrections made by re-measuring: the census (125x not 187x), the D3 after-reading (4.7e-15 not 5.6e-15), the two-build exponents and populations, and what the guard does NOT do |

No merge, push, tag or version bump was made on this branch.
