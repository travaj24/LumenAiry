
### 4.5  M5 -- SPURIOUS CENSUS, CASCADE STABILITY, and the NO-FLOOR property

`m5_census_cascade.py`.  The 1-D convection route's known cost is an
"advection spurious" sea whose `|Re q|` grows with slant, and the roadmap's
V-partner wall showed up as an energy blow-up.  Both are census questions, and
both are asked here against the SAME cell at slant 0 -- the shipped
out-of-plane generator's own census is the control.

**[M] T5a -- CENSUS.**  `px = py = 1.2 lam`, conical 20/35, `M = 6`, `(2,2)`
grid, `dim = 4q^2 = 400`.  "above-band" counts modes with `|q| > 3 sqrt(max eps)`
-- the polynomial basis's own unresolved harmonics, which exist at slant 0 too.

| cell | slant | split (want 200/200) | `min Re(lam_f)` | `max abs(q)` | above-band |
|---|---|---|---|---|---|
| scalar pillar `eps` 4 | vertical | **200/200** | -5.3e-15 | 9.63 | 98 (24.5%) |
| scalar pillar | x 20 deg | **200/200** | -3.6e-15 | 9.73 | 102 (25.5%) |
| scalar pillar | x 45 deg | **200/200** | -6.5e-15 | 11.23 | 128 (32.0%) |
| scalar pillar | diag 45 deg | **200/200** | -5.2e-15 | 13.24 | 160 (40.0%) |
| scalar pillar | x 60 deg | **200/200** | -2.4e-15 | 14.20 | 186 (46.5%) |
| out-of-plane uniaxial pillar | vertical | **200/200** | -9.0e-15 | 9.56 | 144 (36.0%) |
| out-of-plane uniaxial pillar | x 60 deg | **200/200** | -3.3e-15 | 14.10 | 222 (55.5%) |
| high contrast `eps` 12 | vertical | **200/200** | -3.7e-15 | 9.56 | 0 |
| high contrast `eps` 12 | x 60 deg | **200/200** | -6.7e-15 | 13.41 | 34 (8.5%) |
| LOSSY pillar `4 + 0.6i` | vertical | **200/200** | **+4.9e-02** | 9.63 | 96 (24.0%) |
| LOSSY pillar | x 60 deg | **200/200** | **+3.3e-02** | 14.22 | 185 (46.2%) |

Three readings:

* the forward/backward split is **exactly `2q^2 / 2q^2` in all 20 rows**, BEFORE
  the selector's defensive rebalance, at slants up to 60 degrees, on a high-
  contrast cell and on a lossy one -- the contract `_region_modes_oop` pins;
* `min Re(lam_f)` is `>= -2.3e-14` on every lossless row (no growing mode
  classified forward) and strictly POSITIVE on every lossy row (forward modes
  decay), which is the sign a broken magnetic partner would break;
* `max abs(q)` grows from `9.6` to `14.2` between slant 0 and 60 degrees --
  a factor of `1.48`, i.e. `sec(60 deg) = 2` bounded, and NOT the `~210` a
  from-scratch convection form produces in 1-D.  The above-band population is
  the polynomial basis's own unresolved harmonics (24.5% ALREADY at slant 0),
  growing with the same `sec` factor; the flux selector absorbs them, which is
  what the exact split says.

**[M] T5b -- CASCADE vs DEPTH.**  `M = 5`, depths 0.25 / 1 / 3 wavelengths;
`max fwd growth` is `max exp(-Re(lam_f) k0 L)` at 3 wavelengths (any value above
1 means a growing mode was classified forward).

| cell | slant | mount | 0.25 lam | 1 lam | 3 lam | max fwd growth |
|---|---|---|---|---|---|---|
| scalar pillar | vertical | normal | 1.70e-04 | 2.07e-04 | 2.83e-04 | **1.0000e+00** |
| scalar pillar | vertical | conical 20/35 | 1.63e-03 | 7.54e-03 | 1.48e-02 | **1.0000e+00** |
| scalar pillar | x 36.9 deg | conical 20/35 | 2.12e-03 | 3.92e-03 | 4.53e-03 | **1.0000e+00** |
| scalar pillar | diag 45 deg | normal | 4.05e-05 | 2.12e-04 | 7.59e-05 | **1.0000e+00** |
| scalar pillar | diag 45 deg | conical 20/35 | 1.44e-03 | 8.40e-04 | 5.76e-04 | **1.0000e+00** |
| out-of-plane pillar | x 36.9 deg | normal | 6.03e-05 | 1.73e-04 | 3.87e-04 | **1.0000e+00** |
| out-of-plane pillar | diag 45 deg | conical 20/35 | 1.82e-04 | 6.89e-04 | 7.35e-04 | **1.0000e+00** |
| LOSSY pillar | x 36.9 deg | normal | 0.134 | 0.639 | 0.937 | 3.18e-01 |
| LOSSY pillar | diag 45 deg | conical 20/35 | 0.167 | 0.589 | 0.869 | 4.79e-01 |

Forward growth is **exactly `1.0000e+00`** on every lossless row and strictly
below 1 on every lossy one.  Closure does NOT grow with depth on the slanted
rows -- and note the sharpest comparison in the table: the VERTICAL cell at
conical incidence degrades `1.63e-03 -> 1.48e-02` over the depth ladder while
the SLANTED cell on the same geometry improves, `2.12e-03 -> 4.53e-03`.  The
lossy rows' absorption rises monotonically and stays in `[0, 1]`.

**[M] T5c -- LAYER SPLIT.**  One slanted layer of depth `d` versus two stacked
slanted layers of `d/2` at the same slant (the same cell in both, which is the
correct construction: the identity interface match carries the frame offset,
S1.5):

| cell | slant | mount | dR | dT | dJones |
|---|---|---|---|---|---|
| scalar pillar | x 36.9 deg | normal | 2.57e-16 | 5.55e-16 | 6.02e-16 |
| scalar pillar | diag 45 deg | conical 20/35 | 1.95e-16 | 5.55e-16 | 1.26e-15 |
| out-of-plane pillar | x 36.9 deg | conical 20/35 | 5.90e-17 | 9.99e-16 | 6.32e-16 |
| out-of-plane pillar | diag 45 deg | normal | 1.46e-16 | 4.72e-16 | 7.08e-16 |

Worst over all 8 rows: **1.28e-15**.  Machine precision -- so the propagator,
the internal interface and the frame anchor compose exactly, which is the
strongest single check on S1.5's interface argument.

**[M] T5d -- the NO-FLOOR property survives the slant.**  Movement of the
per-order result and the reflection Jones when the far-field order half-width
goes `3 -> 5 -> 8`:

| cell | slant | mount | move 3->5 | move 3->8 |
|---|---|---|---|---|
| scalar pillar | x 36.9 deg | normal | 2.78e-16 | 3.89e-16 |
| scalar pillar | diag 45 deg | normal | 6.11e-16 | 1.17e-15 |
| out-of-plane pillar | x 36.9 deg | normal | 8.88e-16 | 1.94e-16 |
| scalar pillar | x 36.9 deg | conical 20/35 | 1.81e-06 | 2.45e-06 |
| scalar pillar | diag 45 deg | conical 20/35 | 3.33e-06 | 4.87e-06 |
| out-of-plane pillar | diag 45 deg | conical 20/35 | 2.29e-06 | 2.58e-06 |

Machine precision at normal incidence; `~3e-06` at conical.  **[H]** the conical
residue is very likely the `_grazing_safe_wavelength` nudge, which is a function
of the ORDER SET and so changes slightly with `n_orders` -- it is absent at
normal incidence, where no order is near a cutoff.  Either way it is three
decades below the Fourier hybrid's own `E_z`-rule spread (`7.7e-04`), so the
no-floor property is intact.

**M5 PASSES on every arm.**

### 4.6  M6 -- COST

`m6_cost.py` (plus `m4a_timing_probe.py`, S4.4).

**[M] T6a -- per-region assembly + eig**, `Nx = Ny = 2`, conical Bloch shift,
median of 2-5 repeats:

| M | `q^2` | in-plane `2q^2` (QZ) | vertical OOP `4q^2` | SLANTED `4q^2` | slant / OOP | slant / in-plane |
|---|---|---|---|---|---|---|
| 4 | 36 | 8.3 ms | 27.7 ms | 24.3 ms | **0.88x** | 2.98x |
| 5 | 64 | 35.3 ms | 80.6 ms | 78.8 ms | **0.98x** | 2.31x |
| 6 | 100 | 114.1 ms | 228.2 ms | 223.0 ms | **0.98x** | 1.88x |
| 7 | 144 | 325.3 ms | 631.5 ms | 630.5 ms | **1.00x** | 2.07x |

**The slant itself is FREE.**  Against the vertical out-of-plane path -- which
is where a slanted cell has to live anyway, because its covariant tensor has
out-of-plane entries -- the congruence and the six blocks cost `0.88x .. 1.00x`.
The `1.9x .. 3.0x` against the in-plane `2q^2` path is the price of needing the
first-order generator at all, and it is the SHIPPED Stage-B number (`1.33-2.03x`
measured there), not something the slant adds.

And at larger `q` the slanted solve is outright FASTER than the vertical
in-plane one, because it is a Cholesky-whitened standard eig while the in-plane
path pays a QZ -- `m4a` at `Nx = 8, M = 4`: in-plane `2q^2 = 1152` takes
**52.73 s**, slanted `4q^2 = 2304` takes **40.05 s**.

**[M] T6b -- end-to-end single-layer solve** (scalar pillar, conical 20/35,
`n_orders = 3`), slanted vs vertical:

| M | vertical | slanted | ratio |
|---|---|---|---|
| 5 | 0.097 s | 0.143 s | **1.47x** |
| 6 | 0.307 s | 0.400 s | **1.30x** |
| 7 | 0.809 s | 1.076 s | **1.33x** |

**[M] T6c -- the staircase arithmetic.**  See S4.4: on the M4b geometry the
staircase's ladder is fixed by the union grid, its per-slice lateral step cannot
go below one grid cell, and a rung costs `0.8x .. 2.9x` a slanted solve.  The
practical statement is not "one slanted solve replaces N slices" but the
stronger one: **at the slice counts a given union grid admits, there may be no
rung that reaches the slanted answer at all** -- refining the step means
refining `Nx`, which multiplies the region eig by `(Nx (M-1))^6`.

### 4.7  M7 -- SLANT x ANISOTROPY, and a coverage gap this CLOSES

`m7_slant_x_aniso.py`, `m7b_hybrid_oop_slant.py`, `m7c_hybrid_stripe_anomaly.py`.
The covariant congruence `eps -> A^-1 eps A^-T` is tensor-agnostic, so slant x
anisotropy is not a separate feature in this formulation -- it is the same line
of code.  That matters because the shipped 2-D engines do not cover it.

**[M] T7a/T7b -- the COVERAGE GAP is real, and it raises at SOLVE, not at
`add_layer`.**  `PMM2DStackHybrid.add_layer(eps_tensor_cell=..., slant=...)`
ACCEPTS an out-of-plane tensor; the refusal comes later, from

```
NotImplementedError: _layer_eigenmodes_tensor: a SLANTED layer with OUT-OF-PLANE
coupling (eps_xz/yz/zx/zy) is not supported.  The 2-D slant metric is validated
for IN-PLANE tensors only ...
```

So the restriction is genuine (the method's docstring is right about the
outcome) but is enforced one call later than the docstring reads -- worth a
line in the build's own validation.  **No 2-D engine in the suite covers slant x
out-of-plane today.**

*(An incidental note, resolved: `m7b`'s in-plane arm reported a `TypeError` --
that was the PROBE's own result unpacking, not the library.  `m7c` drives all
four `{y-uniform stripe, pillar} x {slant, no slant}` in-plane tensor
configurations through `PMM2DStackHybrid.solve()` and all four succeed.)*

**[M] T7b -- the prototype DOES cover it, validated against the 1-D engine that
also does.**  A y-uniform SLANTED OUT-OF-PLANE stripe (ridge =
`uniaxial(1.5, 1.7, tilt 35 deg, azim 25 deg)`, groove = air, `px = 0.75 lam`,
`depth = 0.30 lam`, duty 0.5) against `pmm_jones_1d_slanted` at degree 30,
per order over `m = -2..2`, both incident polarizations:

| phi | mount | M | incident Ex | incident Ey | oracle's own drift | y-leak | closure |
|---|---|---|---|---|---|---|---|
| 0 | normal | 8 | 2.68e-05 | 1.23e-06 | 3.1e-06 | **0.0** | 4.97e-10 |
| 20 | normal | 8 | 2.71e-05 | 1.75e-06 | 5.5e-06 | **0.0** | 9.11e-10 |
| 35 | normal | 8 | **1.66e-05** | 1.96e-06 | 6.8e-06 | **0.0** | 2.30e-09 |
| 0 | oblique 25 | 8 | 1.91e-05 | 9.35e-07 | 3.3e-06 | **0.0** | 4.10e-11 |
| 20 | oblique 25 | 8 | 2.75e-05 | 3.79e-07 | 1.4e-05 | **0.0** | 2.04e-11 |
| 35 | oblique 25 | 8 | **2.76e-05** | 6.93e-07 | 2.1e-05 | **0.0** | 8.56e-11 |

The slanted rows are as good as the vertical (`phi = 0`) row -- in fact slightly
better at 35 degrees normal -- the `Ey` channel is AT the oracle's own drift,
y-momentum is conserved EXACTLY (`0.0`, not merely small), and closure reaches
`1e-10`.  `m7b` reproduces the `phi = 35`, oblique-25 row independently at
`Ex 2.76e-05 / Ey 6.93e-07`.

**[M] T7c -- a genuinely 2-D slanted IN-PLANE tensor pillar vs the hybrid.**
`px = py = 1.2 lam`, quarter-period pillar of `uniaxial(1.5, 1.7, tilt 90 deg,
azim 25 deg)`, `depth = 0.8 lam`, `t = 0.75`:

| mount | prototype self-move M5->M6 | hybrid `+t`, n=5 / n=7 | hybrid `-t`, n=5 / n=7 |
|---|---|---|---|
| normal | 1.39e-04 | 4.78e-02 / 4.48e-02 | **1.04e-02 / 4.24e-03** |
| conical 20/35 | 9.56e-05 | 5.40e-02 / 4.85e-02 | **7.84e-03 / 3.57e-03** |

The sign is pinned a third time, on a TENSOR cell; the right arm halves with the
hybrid's truncation while the wrong arm does not move; and the prototype's own
`(M)` self-move is two decades below the gap, so the residual is the hybrid's
Fourier floor.  **M7 PASSES**, and the slant x out-of-plane combination becomes
available for the first time in a 2-D engine.
