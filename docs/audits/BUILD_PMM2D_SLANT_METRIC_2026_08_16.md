# BUILD -- a native 2-D slant metric for the 2-D PMM solver (PMM2D roadmap item 3)

**Date:** 2026-08-16 - **Branch:** `feat/pmm2d-slant-metric` off `origin/main` @ `fc3ac52`
(the 5.37.0 release commit; ancestry verified) - **Worktree:** `C:/tmp/lum_sl`
**Status:** PHASE A -- derivation + minimal prototype. Nothing under `lumenairy/` is
modified; the prototype lives in `validation/slant2d_*.py` and injects the slant by
monkeypatching, so the shipped package is untouched.
**Binding law:** `docs/TESTING_STANDARDS.md`.

Tags used below: **[A]** analysis/derivation, no run. **[M]** measured, on the build
recorded in S0. **[H]** hypothesis, consistent with evidence but NOT established.

---

## VERDICT UP FRONT

**Phase A VALIDATES. The formulation is correct and every gate passes** (S4), at normal,
oblique and conical incidence, on both a 1-D-oracle-backed degenerate geometry and a
genuinely 2-D slanted pillar.

**But the scoping question outranks the verdict, and it is not a technical question:**

> **A native 2-D slant metric solves a SLANTED pillar -- tilted axis, CONSTANT
> cross-section. It does NOTHING for a TAPERED pillar -- shrinking cross-section.**
> M5 measured the shear's contribution to a taper at **1.00x** (no gain at all, even on
> the taper carrying the maximum possible shear content). These are different geometries
> requiring different mathematics, and the metasurface pillar program is historically
> **tapered**.

So Phase B is a decision, not a task. If the target devices are **slanted**, this work is
ready to productionize and buys *the same answer 1.5-3.6x cheaper*, plus 2-10x better
accuracy at a matched budget (S4.1). If the target devices are **tapered**, this work
does not help them and the correct item is **T3-6**, which is a different and harder
project:

**[A] T3-6's prerequisites, in one paragraph.** The taper's coordinate map is
`x = u + (z - h/2) S(u)` with `S` piecewise affine, so `X_u = 1 + (z-h/2)S'(u)` is
z-dependent and `sqrt(g) != 1`; the operator picks up a **dilation generator** with a
`<v|S'|phi>` term that is invisible to the slant-limit test (M5 wrote it wrong on the
first pass and the shear test passed anyway). Worse, the resulting frozen pencil is
**non-normal with a `q -> -conj(q)` symmetry**, so its fundamental is complex on a
lossless cell and **no shipped forward/backward selector classifies its modes** -- M5's
prototype cascade diverged at degree 12. And the far field lives in a **distorted frame**
(the interface map is not a rigid translation), so the Rayleigh projection becomes a
quadrature rather than a diagonal phase. Two of these three are research-class with no
precedent in this repo. None of them arise for a shear, because a shear leaves `w = z`
untouched (S1.1).

**Recommendation: do not start Phase B until the target geometry is confirmed.**

---

## 0. Build pin

| | |
|---|---|
| interpreter | CPython 3.14.6 |
| numpy / scipy | 2.4.4 / 1.17.1 |
| `lumenairy.__file__` | `C:\tmp\lum_sl\lumenairy\__init__.py` (asserted in every script) |
| version | 5.37.0 |
| thread caps | `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=4` set before python |

Single-build campaign. Every number below is therefore a **one-build sampling**, not a
cross-build envelope; per `TESTING_STANDARDS.md` rule 5 no test bar may be derived from
these numbers until they are re-measured on a second build. They are used here only to
adjudicate the Phase A go/no-go, which is a decision about *shape* (convergence
direction, null-control exactness, sign uniqueness), not about a reading.

---

## 1. THE DERIVATION

### 1.1 Scope: shear is not taper (the decision that frames everything)

A wall moving linearly with depth decomposes exactly into two independent parts
(`PMM_M5_2D_FEASIBILITY_2026_08_04.md` S3.1):

* **shear / centre walk** -- the whole feature translates with depth. The substitution
  `u = x - t_x z` absorbs it **exactly, at any magnitude**, because the resulting metric
  is z-invariant. This is what ships in 1-D as `PMMStack.add_sheared_grating` -- ONE
  exact slanted layer, no staircase.
* **dilation / duty change** -- the walls separate. **No shear absorbs any of it.**

M5 measured this in 1-D and the result was unambiguous: on a 2 deg taper, scanning the
shear over 13 values improved the single-layer error by a factor of **1.00x**, even on
the one-wall taper that carries the maximum shear content a taper can have.

**[A] Consequence for this campaign, stated up front because it corrects the charter.**
A native 2-D slant metric solves a **slanted** pillar -- tilted axis, *constant*
cross-section -- in one layer. It does **not** solve a **tapered** pillar (shrinking
cross-section); that is the T3-6 covariant-taper item, which M5 left gated on two
research-class prerequisites (a mode selector for a non-normal `q -> -conj(q)` pencil,
and a distorted far-field quadrature). This document is the shear; the taper is not in
scope and does not follow from it.

**[A] Both of M5's T3-6 blockers vanish for pure shear**, which is precisely why the
shear is tractable and the taper is not:

| M5 blocker (taper) | status for a pure 2-D shear |
|---|---|
| non-normal pencil, `q -> -conj(q)`, no valid mode selector | **absent** -- M5's own NULL CONTROL 1 measured the pure-shear cascade at `\|S(K=1) - S(K=8)\| = 9.49e-16`, and 1-D ships the shear at 49 deg through the standard flux selector |
| far field in a distorted frame (needs a quadrature) | **absent** -- at any fixed `z` the map `x = u + t_x z` is a **rigid translation**, so the Rayleigh projection picks up a diagonal phase, not a quadrature |
| `sqrt(g)` carries the z-dependence (the dilation generator) | **absent** -- `det J = 1` exactly (S1.2), so there is no dilation generator and M5's `<v\|S'\|phi>` TRAP cannot arise: `S' = 0` identically for a shear |

**[A] The root reason all three vanish: `w = z` exactly.** A shear moves only the
transverse coordinates; the longitudinal coordinate is untouched. So the frame's
constant-`w` surfaces ARE the lab's constant-`z` surfaces, which means (i) the
z-Poynting flux through an interface plane is the same quantity in both frames -- so the
shipped flux-based forward/backward selector remains valid, and M5's BLOCKER 1 cannot
arise; (ii) the interface planes themselves are undistorted, so the far field stays a
diagonal phase. A taper distorts the transverse map *as a function of z*, which is what
breaks both. This single fact is the dividing line between the tractable item and the
blocked one.

### 1.2 The 2-D slanted coordinate transformation

Let the structure translate with depth by the **slant vector** `t = (t_x, t_y)`
(dimensionless tangents; `t_x = tan(phi_x)`). Following the shipped 1-D convention, the
frame is anchored at the layer TOP (`stack.py:690-692`):

```
u = x - t_x z ,   v = y - t_y z ,   w = z
```

so the inverse map is `x = u + t_x w`, `y = v + t_y w`, `z = w`, with Jacobian

```
        d(x,y,z)         [ 1   0   t_x ]
  A =  ----------   =    [ 0   1   t_y ]        det A = 1   (unit upper-triangular)
        d(u,v,w)         [ 0   0    1  ]
```

**[A] `det A = 1` EXACTLY, for any slant vector, in 2-D as in 1-D.** The matrix is
unit-triangular. (This corrects a claim made during the survey that a 2-D shear would
give `sqrt(g) != 1`; it does not. The volume-preserving property is what makes the
mass matrix `A2 = <v|phi>` completely unchanged from the vertical layer, and it is the
structural reason the 2-D case is no harder than the 1-D case.)

Covariant and contravariant metrics, `g = A^T A`:

```
            [  1    0    t_x ]                    [ 1+t_x^2   t_x t_y   -t_x ]
  g_ij  =   [  0    1    t_y ]        g^ij  =     [ t_x t_y   1+t_y^2   -t_y ]
            [ t_x  t_y  1+|t|^2]                  [  -t_x      -t_y       1  ]
```

with `sqrt(g) = |det A| = 1`.

### 1.3 The operator: metric form

Substituting into the covariant Helmholtz form
`(1/sqrt g) d_i (sqrt g g^ij d_j E) + k0^2 eps E = 0` with `sqrt g = 1` and `g^ij`
constant, and taking `E = phi(u,v) exp(i q k0 w)`, gives the quadratic pencil
`A1 phi - q Ac phi - q^2 A2 phi = 0` with

```
  A1 = <v| eps |phi>  -  (1/k0^2) <grad v| G_t |grad phi>,    G_t = I + t t^T
  Ac = (2i/k0) <v| (t . grad) phi>                            <-- z-FREE, EXACTLY
  A2 = <v| phi>                                               <-- UNCHANGED from vertical
```

Three structural facts, each load-bearing:

1. **[A] `G_t = I + t t^T` is a constant rank-one update of the identity.** It has no
   `(u,v)` dependence, so the metric contributes **nothing** to the Fourier
   factorization problem -- no Li rule is stressed by the slant. (This is the 2-D
   analogue of the 1-D `sec^2` scalar.)
2. **[A] `Ac` is the directional derivative along the slant vector, and is exactly
   z-free.** All z-dependence of the exact operator is gone, so ONE eigensolve does the
   whole layer at any slant magnitude -- this is exact, not a first-order expansion.
3. **[A] `A2` is untouched**, because `sqrt(g) = 1`.

### 1.4 The 1-D limit reduces to the shipped Edee-Granet form -- ANALYTICALLY

Set `t_y = 0` and drop `y`-dependence. Then `G_t -> (1 + t_x^2) = sec^2(phi)` and
`Ac -> (2 i t_x / k0) <v| d_u phi>`, so the pencil becomes

```
  A1 = Peps - sec^2 * L / k0^2 ,   Ac = (2i t/k0) C ,   A2 = S0
```

which is **verbatim** the shipped `_sem_modes_slant` TE pencil, `_core.py:3843-3847` /
`5300-5327`:

```python
        # quadratic pencil (A1 - q Ac - q^2 A2) phi = 0:
        #   A1 = Peps - sec^2 Lop/k0^2,  Ac = (2 i t/k0) C,  A2 = S0.
```

The reduction is exact and symbolic -- no tolerance is involved. This is the gate the
derivation had to pass before any code was written.

### 1.5 The equivalent first-order (convection) form -- what the code actually implements

The shipped 1-D solver does **not** assemble the metric fold; it carries the slant as a
first-order convection on the 4n generator (`_core.py:5772-5787`), because the static
`ezz*tan^2` fold caps per-order accuracy at ~1e-2 while the convection reaches the
~1e-4 wall-normal floor uniformly. The entire slant is these four lines
(`_core.py:5959-5962`):

```python
    if abs(tan_conv) > 1e-14:
        for _b in range(4):
            _sl = slice(_b * n, (_b + 1) * n)
            L[_sl, _sl] += tan_conv * Dopx
```

**[A] The two forms are algebraically identical.** In the frame, at fixed `(u,v)`,
`d/dz|_x = d/dz|_u - (t . grad)`. Substituting into the lab Helmholtz operator:

```
  d_u^2 + d_v^2 + (d_z - t.grad)^2 + k0^2 eps
     = [ (1+t_x^2) d_u^2 + 2 t_x t_y d_u d_v + (1+t_y^2) d_v^2 ]   <-- the metric G_t
       + d_z^2  - 2 (t.grad) d_z                                    <-- the convection Ac
       + k0^2 eps
```

so the `G_t` stiffness and the `Ac` convection are the *same* term, split two ways. The
first-order form is the one to implement: it adds to the generator instead of rebuilding
it.

**[A] The 2-D generalization is therefore one line.** With the 2-D solver's first-order
generator `G = [[A, P], [Q, B]]` (`rcwa/_core.py:3239-3248`) acting on the 4N state
`[E_x; E_y; u_x; u_y]`, and `Kx = GxF`, `Ky = GyF` the projected (dimensionless,
`k/k0`-normalized) derivative operators:

```
  G_slant = G_vertical + c * kron(I_4, t_x*Kx + t_y*Ky)
```

i.e. the convection `t . grad` added to **each of the four diagonal field blocks** --
exactly the 1-D `L[_sl,_sl] += tan_conv * Dopx` with the scalar `tan * d/dx` promoted to
the vector `t_x d/du + t_y d/dv`.

**[A] The sign `c` is predicted to be `-i`.** The 2-D solver propagates as
`exp(-lam k0 z)`, so its ODE is `dX/d(k0 z) = -G X`; in the frame
`dX/dz|_u = dX/dz|_x + (t.grad) X` and `t.grad -> i k0 (t_x Kx + t_y Ky)`, giving
`G_slant = G - i (t_x Kx + t_y Ky)`. **This prediction is not assumed** -- all four
candidates `{+1, -1, +i, -i}` are scanned against the 1-D oracle in gate A1, and the
failure of the other three is the evidence.

**[A] Why the sign must be pinned empirically anyway.** The shipped 1-D tree carries
*two mutually inconsistent* slant sign conventions: the scalar path builds
`t = -np.tan(slant_angle)` with the comment `u = x + tan(phi) z` (`_core.py:5285-5289`)
while the 4n metric generator uses `tan_conv = +np.tan(slant_angle)`
(`_core.py:5785`). Both are validated against the same RCWA staircase, so a
compensating convention exists elsewhere in one of them. Inheriting either by reading is
unsafe.

### 1.6 How the metric composes with the Rayleigh projection

This is where the 2-D hybrid is genuinely **cleaner than 1-D**, and it is worth stating
because it was an open question in the charter.

The hybrid projects every layer into a shared Rayleigh basis via `O_F = T O T^+`
(`twod.py:415-450`), so it has **no union-grid constraint** -- each layer keeps its own
exact spectral-element grid. Three consequences for the slant:

1. **[A] The metric never reaches the projection.** `G_t` is a constant tensor, so it
   commutes with `T` and `T^+`; the convection acts on the already-projected `GxF`,
   `GyF`. Nothing about the slant interacts with the nodal->Fourier bridge.
2. **[A] The 1-D union-grid pathology cannot occur here, and neither can the reason for
   it.** In 1-D, a staircased slant piles up near-coincident walls across slices, and
   the union grid's sliver elements drive `cond ~ w_max/w_min`
   (`AUDIT_PMM_ELEMENT_SIZE_SCALING_2026_06_03.md`). **The 2-D hybrid has no union
   grid at all**, so the `degree>=8` conditioning pathology named in the charter *does
   not exist in `PMM2DStackHybrid`* -- it belongs to the 1-D `PMMStack` and to
   `PMM2DStackPure`. **This removes one of the charter's three stated motivations.**
   (Retiring it remains a real motivation for `PMM2DStackPure`, which does have a hard
   union-grid constraint, `stack2d_pure.py:174-180`.)
3. **[A] The far field stays diagonal.** At any fixed `z` the frame-to-lab map is a
   rigid translation, so a frame-anchored layer needs at most a diagonal phase per
   order -- no quadrature (contrast M5's BLOCKER 2 for the taper).

### 1.7 What the slant costs, structurally

The slant breaks the `[W; -V] <-> -lam` symmetry, exactly as an out-of-plane tensor does
(`twod_jones.py:641-653`). So a slanted layer must return the **generator 6-tuple**
`(W, V, lam, Wb, Vb, lam_b)` and promote the whole stack to the generalized cascade. That
machinery **already exists and is plumbed end to end** -- this is the single largest
piece of good news in the survey; the slant reuses it rather than inventing anything.

The cost is the 4N generator eig in place of the 2N symmetric `eig(P@Q)`.
**[M] Measured overhead: 3.4x** (y-uniform cell, `n_orders=15`, `Nf=961`: 15.1 s
vertical -> 51.2 s slanted). Any speed claim against the staircase must clear this 3.4x
first -- a staircase of fewer than ~4 slices is *cheaper* than one slanted layer.

---

## 2. Phase A gates -- design

Every gate is a **per-order** comparison against an independent oracle. Energy closure is
recorded but **never** used as a pass criterion: the lossless trap (energy conservation
does not prove per-order correctness) is the named hazard for this campaign, and M5's S4
finding -- a tapered PMM stack silently wrong by up to 10x with `|R+T-1| <= 1e-6` --
is the in-repo proof that closure is blind here.

| gate | claim | oracle |
|---|---|---|
| A1 | the convection sign is pinned by the oracle, uniquely | shipped `pmm_jones_1d_slanted` |
| A2 | a UNIFORM cell with any slant is a NO-OP | the same cell with slant 0 |
| A3 | a y-uniform cell sheared ALONG Y is a NO-OP; `(t_x,t_y)` == `(t_x,0)` there | the same cell, `t_y=0` |
| A4 | y-uniform slanted grating == the 1-D slant solver, at the hybrid's own floor | shipped `pmm_jones_1d_slanted` |
| A5 | the N-slice staircase CONVERGES TOWARD the single metric layer | the metric layer itself |

**The bar for A4 is the VERTICAL CONTROL, not an absolute number.** The 2-D hybrid has a
documented Fourier-truncation floor (`twod.py:19-36`) that the 1-D PMM does not have, so
the 2-D-vs-1-D residual is floor-limited even with zero slant.

**[M] Measured vertical control floor** (y-uniform binary grating, `eps` 4/1, `d=300nm`,
`duty=0.5`, `degree=11`; residual = max per-order `|R,T|` difference vs `pmm_jones_1d`):

| period | theta | n=7 | n=11 | n=15 |
|---|---|---|---|---|
| 500 nm | 0.00 | 9.58e-03 | 5.77e-03 | 2.02e-03 |
| 500 nm | 0.20 | 2.20e-02 | 6.98e-03 | 3.18e-03 |
| 700 nm | 0.00 | 8.84e-03 | 4.97e-03 | 1.57e-03 |
| 700 nm | 0.20 | 1.05e-02 | 6.41e-03 | 3.45e-03 |
| 1000 nm | 0.00 | 1.48e-02 | 5.56e-03 | 1.47e-03 |
| 1000 nm | 0.20 | 1.82e-02 | 6.00e-03 | 3.53e-03 |

At `degree=11` with 2 walls, `n_orders >= 21` is refused with

```
ValueError: pmm_jones_2d: n_orders=21 too large for the x-axis grid: need
2*n_orders+1 (43) <= per-axis nodes (33 = 3 elements x degree 11);
raise degree / elements_per_strip or lower n_orders
```

so the ceiling is a **grid** limit that can be lifted (raise `degree` or
`elements_per_strip`), not a hard wall. What actually caps the probe is **cost**: the
order set is the full `(2n+1)^2` even for a y-uniform cell (`Nf = 961` at `n=15`), so the
slanted 4N generator is a dense `3844 x 3844` eig, and `n=21` would be `7396 x 7396`.
**The floor reachable at practical cost on this probe is therefore ~1.5e-3**, and A4
asserts *tracking of the vertical control*, not an absolute tolerance.

[A] Note this also means a y-uniform validation cell is intrinsically wasteful here --
the hybrid spends the whole `y` order dimension on a structure that does not vary in `y`.
That is a property of the solver, not of the slant, but it is what makes the exact 1-D
oracle expensive to reach in 2-D.

---

## 3. Phase A results

Probe geometry unless stated: `P_x = P_y = 700 nm`, `wl = 633 nm`, `d = 300 nm`,
binary x-grating `duty = 0.5`, `eps` 4/1, `n_sub = 1.5`, `degree = 11`, cell sampled on
480 x 1 pixels with the ridge an INTEGER pixel count so every wall is represented
exactly. Residuals are max per-order `|R,T|` differences over both polarizations.

### 3.1 [M] A2 -- a UNIFORM cell with any slant is a NO-OP

A shear of a homogeneous medium is a pure coordinate change, so it must not move any
observable. Includes a genuinely 2-D **diagonal** slant (`t_y = 0.7 t_x`).

| theta | slant | x-only | x AND y |
|---|---|---|---|
| 0.00 | 5 / 20 / 40 deg | `0.00e+00` (all) | `0.00e+00` (all) |
| 0.20 | 5 deg | 5.65e-14 | 1.22e-13 |
| 0.20 | 20 deg | 8.13e-14 | 3.53e-14 |
| 0.20 | 40 deg | 7.17e-14 | 1.29e-14 |

**PASS at machine precision.** (The exact `0.0` at normal incidence is expected and is
the *weaker* arm: only the `(0,0)` order propagates there and its `k_x = 0`, so the
convection shift is identically zero. The oblique row is the informative one.)

### 3.2 [M] A3 -- a y-uniform cell sheared ALONG Y is a NO-OP

Shearing an invariant direction does nothing; and adding a `t_y` component to an
x-slant on such a cell must not perturb it. This is the gate that tests the **vector**
structure of the 2-D convection -- it has no 1-D analogue.

| theta | claim | residual |
|---|---|---|
| 0.00 | `t_y` = 10 / 30 deg alone | `0.00e+00` |
| 0.00 | `(t_x, t_y=10 deg)` vs `(t_x, 0)` | `0.00e+00` |
| 0.00 | `(t_x, t_y=30 deg)` vs `(t_x, 0)` | `0.00e+00` |
| 0.20 | `t_y` = 10 deg alone | 9.96e-14 |
| 0.20 | `t_y` = 30 deg alone | 7.09e-14 |
| 0.20 | `(t_x, t_y=10 deg)` vs `(t_x, 0)` | 6.89e-14 |
| 0.20 | `(t_x, t_y=30 deg)` vs `(t_x, 0)` | 6.06e-14 |

**PASS at machine precision.**

**[M] A2/A3 are also a cross-path check worth more than their stated claim.** The
baselines run the shipped **symmetric 2N** path (`eig(P@Q)`), while every slanted arm
runs the prototype's **4N generator** with a non-zero convection term. That the two
agree to ~1e-13 *on a patterned cell* (A3) says the generator assembly, the flux-based
forward/backward selection, and the generalized cascade all reproduce the shipped
symmetric algorithm exactly where they must.

### 3.3 [M] A1 -- the convection sign is UNIQUELY pinned by the oracle

y-uniform slanted grating, 20 deg slant, `theta = 0.20`, vs
`pmm_jones_1d_slanted`. The winner must **track the vertical control**; the
others must not.

| n_orders | vertical ctrl | `c=+1` | `c=-1` | `c=+i` | `c=-i` |
|---|---|---|---|---|---|
| 7 | 1.05e-02 | `_EnergyError` | `_EnergyError` | 4.19e-01 | **1.05e-02** |
| 11 | 6.41e-03 | `_EnergyError` | `_EnergyError` | 4.04e-01 | **7.48e-03** |
| 15 | 3.45e-03 | `_EnergyError` | `_EnergyError` | 4.11e-01 | **3.57e-03** |

**PASS, and two-sided.** `c = -i` tracks the control to within 15 % at every
truncation and falls with it (1.05e-2 -> 3.57e-3 as the control goes
1.05e-2 -> 3.45e-3). `c = +i` is wrong by two decades and does **not** improve with
`n_orders`; `c = +/-1` are not merely inaccurate, they violate energy hard enough to
raise. This is the sign predicted analytically in S1.5 (`G_slant = G - i(t.K)`), but the
prediction was **not** assumed -- the other three arms are the evidence.

### 3.4 [M] A4 -- the 1-D oracle, and the SILENT-WRONG it exposed

First run (prototype as originally written), residual vs `pmm_jones_1d_slanted`:

| theta | phi | n=7 ctrl / slant | n=11 ctrl / slant | n=15 ctrl / slant |
|---|---|---|---|---|
| 0.20 | 10 deg | 1.0e-02 / 1.1e-02 | 6.4e-03 / 7.0e-03 | 3.4e-03 / 3.6e-03 |
| 0.20 | 20 deg | 1.0e-02 / 1.0e-02 | 6.4e-03 / 7.5e-03 | 3.4e-03 / 3.6e-03 |
| 0.20 | 35 deg | 1.0e-02 / 1.1e-02 | 6.4e-03 / 9.5e-03 | 3.4e-03 / 2.9e-03 |
| **0.00** | **10 deg** | 8.8e-03 / **9.2e-02** | 5.0e-03 / **8.3e-02** | 1.6e-03 / **8.5e-02** |
| **0.00** | **20 deg** | 8.8e-03 / **1.7e-01** | 5.0e-03 / **1.6e-01** | 1.6e-03 / **1.6e-01** |
| **0.00** | **35 deg** | 8.8e-03 / **2.5e-01** | 5.0e-03 / **2.4e-01** | 1.6e-03 / **2.4e-01** |

Oblique tracks the control at every angle and truncation. **Normal incidence was wrong by
up to 2.5e-01, growing with the slant angle and NOT converging with `n_orders`** -- the
signature of a bypassed code path, not of a discretization error.

**[M] Diagnosis, confirmed by direct experiment.** At normal incidence
`kt = hypot(kx0, ky0) < 1e-12` triggers the **F2 even-parity fold**
(`twod_jones.py:614-627`), which calls `_tensor_layer_modes(return_ops=True)` and
`_symmetric_cascade_rt` -- it **never reaches `_layer_eigenmodes_tensor`**, where the
convection lives. A shear destroys the `x -> -x` flip symmetry the fold assumes, so with
the fold left on **a slanted layer silently returns the VERTICAL answer**. Re-run with
`symmetry=False`:

| phi | n | ctrl | fold ON (`symmetry='auto'`) | fold OFF (`symmetry=False`) |
|---|---|---|---|---|
| 10 deg | 7 | 8.84e-03 | 9.23e-02 | **9.06e-03** |
| 10 deg | 11 | 4.97e-03 | 8.34e-02 | **4.81e-03** |
| 20 deg | 7 | 8.84e-03 | 1.70e-01 | **9.65e-03** |
| 20 deg | 11 | 4.97e-03 | 1.61e-01 | **4.34e-03** |
| 35 deg | 7 | 8.84e-03 | 2.52e-01 | **1.09e-02** |
| 35 deg | 11 | 4.97e-03 | 2.43e-01 | **5.43e-03** |

**A4 PASSES at normal incidence once the fold is disabled**, tracking the control at
every angle. The prototype now disables it (`slant2d_proto.py`, `_tensor_layer_modes`
wrapper).

**This is a load-bearing Phase B finding, not merely a harness bug.** It is exactly
insertion site #2 (S4B.1), and the failure mode is the worst class in this repo's
taxonomy: **no warning, energy conserved, deterministic, and wrong by 25 %**. Whoever
implements Phase B must return `None` from `return_ops` for slanted layers -- and the
gate for it must be a *normal-incidence* slanted case, because every oblique test passes
with the bug present.

### 3.5 [M] A gotcha found while building the staircase baseline

`pmm_jones_2d` defaults to `formulation='laurent'`; `PMM2DStackHybrid` defaults to
`formulation='li'`. Comparing them without matching gives a **1.09e-02** discrepancy at
`n_orders=7` on identical vertical geometry -- large enough to be mistaken for a slant
defect. Matched (either value), `stack(ns=1)` vs the single-layer solver is **exactly
`0.0`**. Recorded because any future staircase-vs-metric comparison will hit it.

---

### 3.6 [M] A5 -- the staircase ladder, and the turnaround it produced

y-uniform grating, 25.02 deg slant, `theta = 0.20`, `n_orders = 11`. Distance from the
N-slice staircase to the single metric layer:

| ns | `\|staircase - metric\|` | ratio | cost |
|---|---|---|---|
| 1 | 2.316e-01 | -- | 4.4 s |
| 2 | 7.556e-02 | 3.07 | 7.0 s |
| 4 | 2.161e-02 | 3.50 | 13.8 s |
| 8 | 1.021e-02 | 2.12 | 30.7 s |
| 16 | 1.538e-02 | **0.66** | 51.9 s |
| 32 | 1.809e-02 | **0.85** | 108.2 s |

The staircase closes on the metric answer by 1.4 decades and then **turns around**. Two
readings are consistent with that -- the metric layer is wrong at ~1e-2, or the
staircase's own limit is displaced -- and energy closure cannot separate them (M5 S4:
25 of 26 wrong cells conserved energy). **This was treated as a hard gate and
adjudicated with an independent oracle, not explained away.**

### 3.7 [M] THE ARBITER -- both routes measured against the exact 1-D oracle

Same geometry; oracle is the shipped `pmm_jones_1d_slanted` at `degree=20`, independent
of both 2-D routes.

| n_orders | route | vs 1-D oracle | cost | `\|R+T-1\|` |
|---|---|---|---|---|
| 7 | **metric layer** | **1.066e-02** | 1.20 s | 9.80e-03 |
| 7 | staircase ns=1 | 2.189e-01 | 0.38 s | 4.44e-03 |
| 7 | staircase ns=4 | 3.051e-02 | 1.36 s | 6.16e-03 |
| 7 | staircase ns=8 | 2.467e-02 | 2.69 s | 6.69e-03 |
| 7 | staircase ns=32 | 2.280e-02 (plateau) | 10.49 s | 6.88e-03 |
| 11 | **metric layer** | **7.945e-03** | 10.54 s | 1.65e-04 |
| 11 | staircase ns=1 | 2.296e-01 | 3.18 s | 7.33e-03 |
| 11 | staircase ns=4 | 2.109e-02 | 11.14 s | 1.51e-03 |
| 11 | staircase ns=8 | 6.438e-03 | 21.58 s | 1.98e-03 |
| 11 | staircase ns=16 | **1.055e-02** | 39.80 s | 7.51e-03 |
| 11 | staircase ns=32 | **1.327e-02** | 78.88 s | 8.69e-03 |

**The turnaround indicts the staircase, not the formulation.** At `n_orders=7` the
staircase converges to a limit **2.28e-02** from the exact oracle while the metric layer
sits at **1.07e-02** -- the metric answer is more than 2x closer to truth than the
*fully converged* staircase, so `|staircase - metric|` must stop falling and settle at
the gap between them. That is precisely the observed turnaround.

At `n_orders=11` both routes reach the hybrid's own truncation floor (vertical control
6.41e-03, S2): metric 7.95e-03, staircase 6.44e-03 at ns=8. **Neither route is
"more accurate" in the limit -- both are floor-limited** -- and the staircase begins to
edge ahead at high ns. This is the same pattern the shipped 1-D path documents for
`add_sheared_grating` (an exact-layer plateau that the staircase overtakes around
ns ~ 12-16). The metric layer's value is therefore **cost at fixed accuracy**, not a
lower asymptote.

**[M] And past ns=8 the staircase DIVERGES from the exact oracle.** At `n_orders=11` the
ladder reads `2.11e-02 (ns=4) -> 6.44e-03 (ns=8) -> 1.06e-02 (ns=16) -> 1.33e-02
(ns=32)`: refining the staircase past its sweet spot makes it **monotonically worse**,
by 2.1x over two doublings, with energy closure degrading in step
(`1.98e-03 -> 7.51e-03 -> 8.69e-03`). The metric layer sits stably at 7.95e-03 across
the whole range. **Without an external oracle there is no way to locate that sweet
spot** -- closure does not mark it, and "refine until it stops moving" walks the wrong
way. This is **M5 S4's pathology reproduced**, and it is worth
recording that it appears here in **`PMM2DStackHybrid`**, whereas M5 measured it on the
1-D `PMMStack` and the per-layer paths. It is an independent reason not to treat "a
converged staircase" as a trustworthy PMM-side reference for slanted or tapered stacks
without an external arbiter -- and it is the second time in this document that energy
closure moved in the same direction as the error without being able to detect it.

### 3.8 [M] B1 -- a GENUINELY 2-D slanted pillar, at normal / oblique / conical

Rectangular pillar, 0.35 x 0.35 of the cell, `eps` 4/1, translating along the **diagonal**
slant vector `t = (0.4667, 0.2333) = (25.0, 13.1) deg`. `n_orders = 7`; the Fourier
truncation is common-mode between the two routes and cancels in their difference. No 1-D
oracle exists for this geometry (and the shipped 1-D slant path refuses conical outright),
so the claim is the **direction** of convergence.

| ns | normal | oblique (th=0.2) | conical (th=0.2, ph=0.6) |
|---|---|---|---|
| 1 | 5.770e-02 | 1.062e-01 | 1.034e-01 |
| 2 | 1.245e-02 (4.64) | 1.855e-02 (5.72) | 1.265e-02 (8.18) |
| 4 | 2.794e-03 (4.45) | 6.289e-03 (2.95) | 4.678e-03 (2.70) |
| 8 | 1.031e-03 (2.71) | 2.583e-03 (2.43) | 1.868e-03 (2.50) |
| 12 | 1.130e-03 | 2.073e-03 | 2.051e-03 |
| 24 | 1.252e-03 | 1.830e-03 | 2.235e-03 |

**PASS in all three incidence cases, conical included.** The staircase converges toward
the single metric layer by 1.5-2 decades and then plateaus at the shared `n_orders=7`
floor. Energy closure sits at 4.6e-03 - 5.6e-03 for **both** routes across every cell --
i.e. the envelope is set by the truncation, is the same for both, and (per the lossless
trap) is reported as context, never as a pass criterion.

---

## 4. Verdict -- PHASE A VALIDATES (GO)

Every gate passes. The formulation is correct, it is the exact 2-D generalization of the
shipped 1-D machinery, and it is validated at normal, oblique and conical incidence on
both a degenerate (1-D-oracle-backed) and a genuinely 2-D structure.

| gate | claim | result |
|---|---|---|
| A1 | convection sign pinned uniquely | **PASS** -- `c=-i` tracks the control at 3 truncations; `+i` wrong by 2 decades; `+/-1` raise on energy |
| A2 | uniform cell + any slant is a no-op | **PASS** -- <= 1.2e-13 |
| A3 | y-uniform cell sheared along y is a no-op; `(t_x,t_y)==(t_x,0)` | **PASS** -- <= 1.0e-13 |
| A4 | y-uniform slanted == shipped 1-D slant solver | **PASS** at oblique as written; **PASS** at normal after disabling the even-parity fold |
| A5 + arbiter | staircase converges toward the metric layer | **PASS** -- and the observed turnaround is the staircase's displaced limit, proven with an independent oracle |
| B1 | genuinely 2-D, normal / oblique / conical | **PASS** -- 1.5-2 decades of convergence in all three |

### 4.1 [M] Accuracy per unit cost -- the number that should decide Phase B

Genuinely 2-D pillar, `n_orders = 7`. "Metric bound" is its distance to the most
converged staircase rung (ns=24), i.e. an upper bound on its own residual.

| case | metric cost / bound | at MATCHED cost | typical ns=8 | typical ns=12 |
|---|---|---|---|---|
| normal | 2.12 s / 1.25e-03 | ns=4: 2.79e-03 (1.74 s) | 1.03e-03 at **1.49x** cost | 1.13e-03 at **2.29x** cost |
| oblique | 1.42 s / 1.83e-03 | ns=2: 1.86e-02 (0.90 s) | 2.58e-03 at **2.48x** cost | 2.07e-03 at **3.57x** cost |
| conical | 1.57 s / 2.23e-03 | ns=2: 1.26e-02 (0.99 s) | 1.87e-03 at **2.14x** cost | 2.05e-03 at **3.15x** cost |

**Read this two ways, and both are modest.**

* **At matched wall-clock the metric layer wins**, by 2.2x (normal), 10x (oblique) and
  5.6x (conical) in accuracy -- because the staircase can only afford ns=2-4 at the
  metric layer's price.
* **At typical practice (ns=8-16) the staircase reaches the same accuracy** -- both are
  at the truncation floor -- **for 1.5x to 3.6x the cost.** So the honest headline is
  *"same answer, 1.5-3.6x cheaper"*, plus a genuine accuracy win only when the budget is
  tight.

This is a real but **not** transformational win, and it is materially smaller than the
charter's framing. The 4N generator overhead (3.4x, S1.7) is what eats most of the
"no N eigensolves" saving.

---

## 4B. Phase B plan -- the insertion set, and the economics that should gate it

### 4B.1 The insertion set (surveyed, not yet implemented)

Ten sites, all already located. Nothing in `_stack2d_cache.py` needs to change -- it is
key-agnostic; the keys are built in `stack2d.py`.

| # | site | change |
|---|---|---|
| 1 | `twod_jones.py:117-320` `_tensor_layer_modes` | accept the slant vector; add the convection to the generator; decide the uniform / separable / crossed branch behaviour |
| 2 | `twod_jones.py:312-318` `return_ops` | return `None` when slanted (kills the even-parity fold, as out-of-plane already does -- the shear destroys the flip symmetry) |
| 3 | `rcwa/_core.py:3153` `_layer_eigenmodes_tensor` | route slanted through the 4N generator -- **already exists**, reuse the 6-tuple |
| 4 | `stack2d.py:466-476` `_append_patterned` | carry `slant=(t_x, t_y)` in the layer dict |
| 5 | `stack2d.py:333-337` `_geom_key` | **add the slant vector to the cache key** |
| 6 | `stack2d.py:729-742` `_mode_key` | add to `common`; **guard the uniform-valued-tile collapse at 737-741** -- a patterned slanted layer whose tile happens to be constant would otherwise collapse to `('uniform', ...)` and silently drop the slant |
| 7 | `stack2d.py:744-788` `_build_layer_modes` | return `('gen', ...)` for slanted |
| 8 | `stack2d.py:1162-1164` symmetry gate | exclude slanted layers |
| 9 | `stack2d.py:542-584` | new `add_slanted_pillar(..., shear=)` mirroring `PMMStack.add_sheared_grating` |
| 10 | tests | per `TESTING_STANDARDS.md`; see 4B.3 |

**Two correctness traps specific to this integration**, both found in the survey:

* **Site 6 is a silent-wrong hazard**, not a nicety: the `_mode_key` early return
  collapses a constant tile to the uniform key *before* `_geom_key` is consulted.
* **`_cascade_sequence`'s run-merge** (`stack2d.py:868-888`) sums the thicknesses of
  adjacent layers sharing a modal key. For slanted layers that is exact **only if the
  slant frame is continuous across the joint** -- two slanted layers with the same
  slant vector stack correctly, but the merge must not fire across a slant
  discontinuity. Needs its own two-sided test.

### 4B.2 Scope decision -- OOP-slant

The 1-D tree leaves slant x out-of-plane only partially closed: `'auto'` reroutes
slanted OOP to convection, and the metric-generator `gen2` prototype was never
integrated (it hit the lossless trap). **The 2-D derivation does NOT give OOP-slant for
free**: the prototype carries the OOP cross-blocks `A`/`B` through unchanged, but that
composition -- a constant shear metric on top of the pointwise `e_zz`-Schur reduction
(`twod_jones.py:143-160`) -- is exactly the ordering the 1-D `gen2` got wrong, and it is
untested here. **Recommendation: refuse `slant + out-of-plane` in 2-D with an explicit
`NotImplementedError`**, mirroring the 1-D gate wording, and treat it as a separate item.
Do not inherit the open corner silently.

### 4B.3 What Phase B must measure before it can claim anything

Per `TESTING_STANDARDS.md`, no bar below may be set from this document's single-build
numbers; each must be re-measured on two builds with the derivation recorded.

1. **`slant -> 0` reduction**, byte-identical to the shipped vertical path (the 1-D
   precedent `test_jones_slant_out_of_plane_slant0_reduces_to_vertical` asserts exact
   `0.0`). The prototype already delegates to the original at `t = 0`, so this is
   structural, not tolerance-based.
2. **y-uniform cell vs `pmm_jones_1d_slanted`**, asserted as *tracking the vertical
   control at the same `n_orders`* -- never an absolute floor bar (S2: the 2-D hybrid's
   own truncation floor dominates and moves with `n_orders`).
3. **A2/A3 null controls** -- these are the strongest available and are decisions, not
   readings: "a shear of an invariant direction changes nothing".
4. **The cascade-merge two-sided test** (4B.1).
5. **Cost, honestly.** See 4B.4.

### 4B.4 [M] The economics -- and why they should gate Phase B

The charter states three motivations. The survey and Phase A retire or shrink two of
them:

| charter motivation | status |
|---|---|
| "retires the `degree>=8` staircase-conditioning pathology" | **VOID for `PMM2DStackHybrid`.** That pathology is a *union-grid* sliver-element effect and the hybrid has no union grid (S1.6). It is real for `PMM2DStackPure`, which does. |
| "accuracy gain (no staircase discretization)" | **REAL, but bounded by the hybrid's own Fourier floor** (~1.5e-3 on the Phase A probe, S2), which is far above the staircase error at modest slice counts. |
| "speedup (no N eigensolves)" | **REAL but reduced by 3.4x.** The slant forces the 4N generator in place of the 2N symmetric eig: measured 51.2 s vs 15.1 s at `Nf=961`. One slanted layer costs ~3.4 vertical slices, so the break-even is ~4 slices, not 1. |

**And the target geometry matters more than any of this.** The charter names the
tapered-pillar metasurface program. A *tapered* pillar gets **nothing** from this work
(S1.1, measured in 1-D by M5 at 1.00x). Only a *slanted* pillar -- tilted axis, constant
cross-section -- benefits. **Phase B should not start until it is confirmed that the
target devices are slanted rather than tapered**, because for a taper the correct item
is T3-6, which is blocked on different and harder prerequisites.

---

## 5. Side finding -- an undocumented Jones sign discontinuity in the SHIPPED 1-D slant path

Found while characterizing the Phase A oracle; **not caused by this work**, and it
constrains what the oracle can be compared on.

**[M] The 1-D slanted solver's p-row (row 0) Jones sign disagrees with `pmm_jones_1d`,
and flips discontinuously at `slant -> 0` under oblique incidence.** Probe: `P=1um`,
`wl=633nm`, `d=300nm`, `duty=0.45`, `eps` 4/1, `n_sub=1.5`, `degree=12`,
`factorization='convection'`.

| theta | `J00` from `pmm_jones_1d_slanted(phi=0)` | `J00` from `pmm_jones_1d` | ratio |
|---|---|---|---|
| 0.00 | `+0.2608730132718605+0.0243370852944442j` | `-0.2608730132718612-0.0243370852944423j` | **-1** |
| 0.25 | `+0.1802692384599169-0.0413939531930273j` | `-0.1802692384599139+0.0413939531930267j` | **-1** |

and within the slant solver itself, at `theta=0.25`, `J00(phi)/J00(phi=0)`:

| phi | 1e-6 deg | 1e-3 deg | 1 deg | 5 deg | 20 deg |
|---|---|---|---|---|---|
| ratio | **-1.000000** | -1.000000 | -0.999991 | -0.999745 | -0.990318 |

so an infinitesimal slant flips the row-0 sign, then varies continuously from there. At
`theta=0` the same ratios are `+1` -- the discontinuity is oblique-only. `|J|` and all
efficiencies are unaffected (a uniform slab is slant-invariant in `R`/`T` to 5.8e-10 over
0-40 deg), so this is a **sign/gauge convention**, not an energy defect -- but the
relative s/p phase is physically observable (polarization conversion), so it is not free.

**Why it was never caught:** the reduction test
`test_jones_slant_reduces_to_pmm_jones_at_phi0_stabilized`
(`tests/unit/test_v5_12_0_pmm_slant_and_convergence.py:414`) compares
`np.abs(J) - np.abs(JJ)` -- **magnitudes only** -- and its docstring does not mention a
sign convention. `test_slant_zero_reduces_to_vertical` is on efficiencies. So no shipped
test constrains the slanted Jones sign.

**[H]** The most likely mechanism is the discrete forward/backward selection in
`_split_modes_flux_metric` re-pairing when the convection term is switched on at
`abs(tan_conv) > 1e-14` (`_core.py:5959`), which would make the flip a mode-labelling
artifact rather than a physics error. Not established -- it is recorded as a finding, not
a diagnosis.

**Consequence for this campaign:** Phase A compares **per-order efficiencies**, which are
convention-free, and does not use the 1-D Jones sign as an oracle.

### 5.1 Reproduction

```
cd validation
PYTHONPATH=<repo>:<repo>/validation python slant1d_jones_sign.py
```

`validation/slant1d_jones_sign.py` prints, for `theta` in `{0.0, 0.25}`, the row-0 and
row-1 Jones entries from `pmm_jones_1d_slanted(phi=0)` against `pmm_jones_1d`, then the
ratio `J(phi)/J(phi=0)` for `phi` in `{1e-6, 1e-3, 1, 5, 20}` deg. The signature is
`J00` ratio `= -1.000000` at `phi = 1e-6` deg when `theta != 0`, and `+1` when
`theta = 0`. `validation/slant1d_floor_probe.py` is the companion probe that
reproduces the shipped `add_sheared_grating` accuracy table (2.93e-03 at 49.4 deg,
matching the docstring) and measures the 1-D slant floor vs tilt:
3.0e-4 (9.5 deg) / 4.2e-4 (18.4 deg) / 1.2e-3 (33.7 deg) / 2.9e-3 (49.4 deg).

### 5.2 Status

**Candidate defect, NOT fixed in this campaign** (out of scope; it is a 1-D path and this
branch changes no library file). Recommended as a separate small fix, whose gate would be
a sign-sensitive reduction assertion -- compare `J` itself, not `|J|` -- at oblique
incidence across `slant = 0` and `slant = 1e-6`. Note that a fix must first decide which
of the two conventions is canonical; `pmm_jones_1d` and the slant path currently disagree
at `theta = 0` as well, so "make the slant path match the vertical path" is a real
behaviour change to the vertical path's sign or the slant path's, not a no-op.
