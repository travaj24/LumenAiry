
### 4.4  M4 -- a genuinely 2-D SLANTED PILLAR, three ways

`m4_pillar.py`, `m4a_timing_probe.py`, `m4b_staircase_ladder.py`,
`m4c_hybrid_ladder.py`.  Pillar `[0.3, 0.6] x [0.3, 0.6]` (a quarter-period
square) in `px = py = 1.2 lam`, `eps` 4 / 1, `depth = 0.8 lam`, `n_sub = 1.5`,
`wl = 1`.  The pillar translates by `t . depth = 0.6` = HALF a period over the
layer -- a steep case, chosen because every staircase wall then lands on the
uniform `Nx = Ny = 8` grid.  Comparison is per order on `(0,0)`, `(+-1, 0)`,
`(0, +-1)`, `(+-1, +-1)`, both incident polarizations, on R and T.

**[M] (a) The prototype metric layer -- one solve, and it is GRID-INVARIANT.**
The same physical layer solved on two different union grids and two modal
counts:

| block | `Nx=4, M=5` | `Nx=4, M=6` | `Nx=8, M=3` | `Nx=8, M=4` (reference) |
|---|---|---|---|---|
| normal, diagonal slant -- self-move vs the reference | 1.82e-04 | 1.40e-04 | 5.23e-04 | -- |
| normal -- cost / closure | 8.1 s / 5.6e-08 | 28.4 s / 4.8e-10 | 7.0 s / 1.6e-05 | 98.2 s / 1.2e-07 |
| conical 20/35, x slant -- self-move | 4.43e-04 | 3.13e-04 | 5.38e-03 | -- |
| conical -- cost / closure | 8.8 s / 9.0e-06 | 38.6 s / 2.2e-07 | 9.2 s / 7.1e-04 | 133.9 s / 5.5e-06 |

The `Nx = 4` and `Nx = 8` grids describe the SAME pillar (the extra walls are
redundant), so the `1.4e-04 .. 4.4e-04` spread is the solver's own
discretization, not a slant artifact -- and it is what the reference row below
is worth.

**[M] (b) The shipped HYBRID slant metric -- an INDEPENDENT formulation, and the
sign, two-sided.**  `PMM2DStackHybrid.add_layer(slant=...)`: a Fourier basis, a
tensor fold, and a lab-Cartesian convection on its 4N generator -- nothing in
common with the prototype except the physics.  Distance to the prototype's
`Nx=8, M=4` reference:

| block | `t_hybrid = +t_probe` | `t_hybrid = -t_probe` |
|---|---|---|
| normal, diagonal, `n_orders = 5` | 1.811e-01 | **1.196e-02** |
| normal, diagonal, `n_orders = 7` | 1.761e-01 | **5.436e-03** |
| conical 20/35, x, `n_orders = 5` | 3.189e-01 | **5.270e-02** |
| conical 20/35, x, `n_orders = 7` | 3.190e-01 | **4.230e-02** |

**The hybrid's public `slant` is the NEGATIVE of the prototype's internal `t`**
(S6.1) -- and the wrong sign does not merely sit further away, it does not
IMPROVE with `n_orders` (1.811e-01 -> 1.761e-01, 3.189e-01 -> 3.190e-01) while
the right one does (1.196e-02 -> 5.436e-03).  That is the two-sidedness: a
truncation ladder separates a discretization gap from a wrong structure.

**[M] (c) A z-STAIRCASE from the PURE solver itself -- and the structural
obstruction.**  Distance to the same reference, `Nx = Ny = 8`, `M = 3`:

| block | n=1 | n=2 | n=4 |
|---|---|---|---|
| normal, diagonal | 2.415e-01 | 1.649e-01 | 1.685e-01 |
| conical 20/35, x | 2.599e-01 | 2.942e-01 | 3.070e-01 |

and the `M = 4` CONTROL on the normal/diagonal block, which rules out an
`M = 3` artifact completely -- **every rung reproduces its `M = 3` value to four
digits**, at 5-10x the cost:

| n | `M = 3` | `M = 4` | `M = 3` cost | `M = 4` cost |
|---|---|---|---|---|
| 1 | 2.415e-01 | **2.415e-01** | 8.8 s | 104.8 s |
| 2 | 1.649e-01 | **1.649e-01** | 13.6 s | 146.2 s |
| 4 | 1.685e-01 | **1.685e-01** | 22.3 s | 213.3 s |

So the staircase is fully converged in the modal count and simply GEOMETRICALLY
wrong: at the only slice counts this grid admits it sits `1.7e-01` from the
answer, while an INDEPENDENT engine (the hybrid, S4.4(b)) agrees with the
prototype to `5.4e-03` -- 30x closer -- and is still improving with its own
truncation.

Two incidental confirmations fall out.  MIDPOINT and LEADING-EDGE sampling give
IDENTICAL efficiencies at every `n` where both are admissible (`2.415e-01` and
`1.649e-01` in both columns) -- the two differ by a RIGID lateral translation of
the whole stack, and the pure staggered solver's position invariance makes that
a no-op.  And closure is excellent throughout (`1.6e-04 .. 2.5e-05`) while the
answer is `1.7e-01` wrong: **the lossless trap, reproduced on this cascade.**

**[M] Why the ladder stops at n = 4, and why that is the finding.**  `Basis1D`'s
segments are a `linspace`, so every slice must place its walls on ONE uniform
union grid of spacing `h = px / Nx`.  Two consequences:

1. the admissible slice counts are exactly the DIVISORS of `S / h` (`S` = the
   total lateral walk).  Here `S = px/2 = 4h`, so `n in {1, 2, 4}` and nothing
   else -- the ladder is fixed by the geometry and the grid, not chosen;
2. the smallest non-zero per-slice lateral STEP is `h` itself.  At `Nx = 8` that
   is `0.15`, i.e. **half the pillar width**.  A staircase of this pillar can
   therefore never have a step finer than half the feature until the union grid
   is refined -- and refining it costs `(Nx (M-1))^6` in the region eig.

`m4a_timing_probe.py` prices that: one region solve, `px = py = 1.2`, same cell.

| grid | in-plane `2q^2` (QZ) | slanted `4q^2` (whitened) |
|---|---|---|
| `Nx=4, M=5` (`q^2 = 256`) | 512, **3.51 s** | 1024, **4.78 s** |
| `Nx=4, M=6` (`q^2 = 400`) | 800, **13.55 s** | 1600, **13.74 s** |
| `Nx=8, M=3` (`q^2 = 256`) | 512, **3.34 s** | 1024, **3.76 s** |
| `Nx=8, M=4` (`q^2 = 576`) | 1152, **52.73 s** | 2304, **40.05 s** |

Note the last row: the SLANTED `4q^2` solve is *faster* than the vertical
in-plane `2q^2` one at the same `q`, because the slanted pencil is
Cholesky-whitened to a standard eig while the in-plane path pays a QZ.  Doubling
`Nx` from 4 to 8 at fixed `M` multiplies the region solve by ~8x per axis pair;
that is the price of every extra staircase rung.
