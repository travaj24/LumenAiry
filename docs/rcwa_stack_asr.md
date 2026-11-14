# Adaptive spatial resolution for `RCWAStack` (1-D)

`RCWAStack(..., asr=True)` solves a 1-D multilayer in one stretched lateral
coordinate whose lines carry the walls of every patterned layer, with the
Fourier harmonics clustered at the walls.  On TM metal-strip stacks, where the
uniform basis converges only as about `N^-1.1`, it reaches 5e-6 at
`n_orders = 20` and 1e-7 (the accuracy of the reference) from about 70.  The
implementation is `lumenairy/elements/rcwa/_asr_stack.py`; its module
docstring carries the derivation.  Tests: `tests/unit/test_rcwa_stack_asr.py`.

```python
import lumenairy as lum
st = lum.RCWAStack(370e-9, n_superstrate=1.0, n_substrate=1.447, n_orders=30,
                   asr=True)                      # or asr=dict(eta=1.0)
st.add_layer(50e-9, segments=[(175 / 370, -74 + 6j), (195 / 370, 1.0)])
st.add_layer(21e-9, eps=3.3)
st.add_layer(70e-9, eps=-160 + 30j)
res = st.set_source(1300e-9, theta=0.0).solve()
r_tm = res.jones_reflection()[0, 0]
```

## API and why it is a stack property

`asr` is a constructor argument (`None` default = off, bit-for-bit the
existing solve; `True`; a stretch `eta` in `(0, 1]` or `'auto'`; or
`dict(eta=..., profile='sine4' | 'sine')`), read back by `RCWAStack.asr`.  The
one coordinate map is built from all the layers, so it belongs to the stack,
and as a stack attribute it reaches every solve path without plumbing:
`solve`, `solve(stabilize=True)`, `solve_vs_wavelength` (dispersive layers are
materialized per wavelength; walls are geometry and do not move) and
`rcwa_convergence` (which bumps `n_orders` in place).

## The formulation, and the measurements behind each choice

References: Granet, JOSA A 16, 2510 (1999); Vallius and Honkanen, Opt.
Express 10, 24 (2002); Li, JOSA A 13, 1870 (1996); Granet and Guizal, JOSA A
13, 1019 (1996); Lalanne and Morris, JOSA A 13, 779 (1996); Weiss et al., Opt.
Express 17, 8051 (2009); Essig and Busch, Opt. Express 18, 23258 (2010).

1. **One map for the whole stack** from the union of the transition points of
   all patterned layers (Essig and Busch; Weiss et al.), not per-layer maps
   with a projection at each interface (Vallius and Honkanen).  A projection
   between differently mapped bases has to represent plane waves of one basis
   in the other: that is the truncated bridge `G` of item 4, which floors at
   1e-5 even on a dielectric grating.  On a thin-layer stack the patterned and
   uniform layers couple through exactly the high-order evanescent content
   such a projection truncates.  Uniform layers lose their diagonal modes, but
   all of them (and both half-spaces) share one Hermitian eigendecomposition
   per solve, and their modes are exact eigenvectors of the truncated
   operator in classical and conical incidence (an all-uniform stack forced
   through these modes reproduced the plain solve to 3e-13 at theta 0.3; the
   shipped solve runs such a stack the standard way).  Two patterned layers
   with different walls (union map, four sub-intervals) converge as well:

   | `n_orders` | 10 | 20 | 30 | 40 | 60 | 80 |
   |---|---|---|---|---|---|---|
   | dielectric bars (eps 4), TM, `'li'` | 2.9e-4 | 7.1e-5 | 3.0e-5 | 1.7e-5 | 7.4e-6 | 4.1e-6 |
   | same, ASR | 2.0e-4 | 6.9e-7 | 1.6e-8 | 1.4e-8 | 1.2e-9 | 1.6e-10 |
   | metal bars (eps -20+2i), TM, `'li'` | 2.2e-2 | 6.0e-3 | 8.4e-3 | 6.3e-3 | 4.1e-3 | 2.5e-3 |
   | same, ASR | 2.1e-2 | 1.2e-3 | 1.1e-4 | 2.8e-5 | 2.7e-6 | 4.4e-7 |

   (period 1000 nm, 150 nm bars 0.4-wide over 30 nm of eps 2 over a
   0.25 / 0.5 / 0.25 three-segment 150 nm layer, n = 1.5 substrate, 1310 nm,
   theta 0.2; error against ASR at 120, which agrees with single-element
   `PMMStack` degree 80 to 5e-8 (dielectric; PMM's own 60-80 spread 8e-8) and
   6e-5 (metal; PMM spread 6e-5).)

2. **Covariant form** (Weiss et al. eqs. 39-40): `eps' = eps diag(1/f, f, f)`,
   `mu' = diag(1/f, f, f)`; inverse rule `[[f/eps]]^-1` for `eps'_uu`, direct
   rule on `eps f` and `f` elsewhere.  Only `f` enters, never `1/f`, which is
   what allows `f = 0` at the walls.  The single-layer ASR's chain-rule form
   (`[[1/f]] K`) cannot reach `f = 0`.  Fourier coefficients are exact
   (closed form; no sampling).

3. **The map**: per sub-interval `f(s) = 1 - eta sum_k a_k cos(2 pi k s)`.
   `'sine'` (`a = (1,)`) is Granet's; `'sine4'` (`a = (4/3, -1/3)`,
   `f = (8/3) sin^4(pi s)` at `eta = 1`) is flat to fourth order at the walls,
   so `x - a ~ s^5` and a corner field `r^tau` becomes `s^(5 tau)`: the
   singularity is removed, not just resolved more finely.  Fixture A (below),
   TM, `eta = 1`, error against PMM degree 20 x 12 (whose own error is
   5e-7):

   | `n_orders` | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 100 |
   |---|---|---|---|---|---|---|---|---|
   | `'sine'` | 7.6e-4 | 2.4e-4 | 3.3e-5 | 4.1e-5 | 9.3e-5 | 3.8e-6 | 1.9e-6 | 6.9e-6 |
   | `'sine4'` | 4.2e-6 | 2.6e-6 | 2.2e-6 | 1.0e-6 | 1.6e-6 | 6.6e-7 | 5.7e-7 | 5.5e-7 |

   Any `eta < 1` leaves a linear zone at the wall where the singularity keeps
   its exponent: `'sine4'` at `eta = 0.999` sits at 2.5e-6 / 1.7e-6 at 150 /
   200 where `eta = 1` is at 9e-8.

4. **Half-spaces through an oversampled outgoing admittance.**  This is the
   cause of the single-layer ASR's ~1e-4 TM floor
   (`rcwa_efficiency_1d(asr_eta=...)`, which bridges with `G^-1`), and the
   central fix.  A half-space enters the S-matrix only through the relation
   between the tangential `H` and `E` of outgoing waves.  Dielectric lamellar
   grating (eps 4 / air, duty 0.5, 300 nm, period 1000 nm, 800 nm, normal
   incidence), TM, `'sine'` at `eta = 0.9`, error against plain `'li'` at
   `n_orders` 400 (whose own error is about 5e-7):

   | half-space treatment | 10 | 20 | 40 | 60 | 80 |
   |---|---|---|---|---|---|
   | plain `'li'` (no ASR) | 4.1e-4 | 9.5e-5 | 2.2e-5 | 9.2e-6 | |
   | truncated plane-wave bridge `G` | 4.0e-6 | 1.1e-6 | 2.4e-5 | 9.3e-6 | 1.2e-2 |
   | the half-space's own truncated modes | 7.4e-4 | 5.0e-4 | 2.5e-4 | | 1.3e-4 |
   | own modes, propagating ones exact (Essig and Busch) | 7.4e-4 | 5.0e-4 | 2.5e-4 | | 1.3e-4 |
   | admittance from a 2x basis, restricted (shipped) | 3.8e-6 | 7.6e-7 | 3.7e-8 | 9.2e-8 | |

   The bridge `G` (the old single-layer construction) is good at low order
   and then floors and blows up: high-order plane waves are not representable
   in `N` stretched harmonics.  The half-space's own truncated modes carry
   distorted high-order eigenvalues, and in TM the resulting outgoing
   admittance costs an `O(N^-1)` error (slower than no ASR at all).  The
   shipped admittance is computed in a `u` basis twice as long and
   restricted to the retained harmonics (a projection of an accurate operator
   instead of an operator of a projection); oversampling 2 and 3 give equal
   results, and it equals the admittance built from exact plane waves on a
   fine `x` grid to round-off.  Finite uniform layers keep their own modes:
   with both directions present their basis is complete (replacing them by
   exact projected operators changed nothing, and treating the 70 nm mirror
   of fixture A as a half-space instead of a layer left the error
   unchanged).  Rayleigh amplitudes are
   the exact `x`-Fourier coefficients of the truncated field.

5. **Balanced first-order eigensolve.**  With `f -> 0` the local wavenumber
   `k/f` near a wall is huge (largest modal eigenvalue 5e7 at `n_orders`
   100).  The squared operator `P Q` then has norm 1e12 and more, and the
   eigensolver's absolute error swamps the `O(1)` eigenvalues.  Fixture A,
   `'sine'`, `eta = 1`:

   | `n_orders` | 80 | 90 | 100 | 101 | 110 / 120 |
   |---|---|---|---|---|---|
   | eig of `P Q` | 2.0e-6 | 6.9e-6 | **4.4e-4** | 3.3e-5 | 2.0e-5 |
   | balanced first-order (shipped) | 1.9e-6 | 5.0e-6 | 6.9e-6 | 1.0e-6 | 6.0e-7 |

   The layer is solved as `[[0, P'], [Q', 0]]` in the frame of the uniform
   eigenvectors with a per-mode diagonal scaling that balances each `2 x 2`
   block; its norm is the largest eigenvalue, not its square.  It stays clean
   to `n_orders` 300 (150 / 200 / 300 agree to 1.3e-8).

**Default stretch** (`eta='auto'`): `eta = clip((n_orders - 5) / 10, 0, 1)`.
The full stretch resolves the middle of each segment 2.7 times more coarsely
than the plain basis; at `n_orders` 10 a fixed `eta = 1` was less accurate
than the plain basis on the dielectric grating (TE 2.1e-3 against 2.9e-4),
while the ramp (0.5 there) gives 1.9e-5.  With the ramp, ASR beat or
matched the plain basis at 62 of 66 points (`n_orders` 6..16; TE and TM on the
dielectric grating, fixture A and fixture B); the exceptions are the
dielectric TE at 7 (8.2e-4 against 7.3e-4), fixture A TM at 7 and 8 (both more
than 6 % off) and fixture B TM at 11 (6.5e-3 against 5.0e-3, a dip of the
plain basis's own scatter).  At 5 and below the solve is the standard
one.  A fixed `eta < 1` is for studies only (item 3).  The guard against the
ill-conditioned extreme: the Toeplitz matrix of `f` (whose inverse square root
builds every uniform mode) is refused above condition number 1e13 (measured
2.7e3 / 1.7e7 / 2.7e8 at `n_orders` 50 / 150 / 300); the eigensolve itself is
balanced (item 5).

## Convergence tables

**Fixture A**: a 175 nm strip (eps -74+6i, 50 nm) over 1 / 7 / 5 / 7 / 1 nm of
eps 3.05 / 3.53 / 2.5+0.5i / 3.53 / 3.05 over a 70 nm mirror (eps -160+30i),
period 370 nm, air / n = 1.447, 1300 nm.  **Fixture B**: the same stack at
period 1200 nm with a 500 nm strip (orders +/-1 propagate in the substrate).
References: `PMMStack` with geometric grading (ratio 0.17), degree 24 x 12
elements (fixture A TM: 0.05715711017 + 0.13122118072i; degree 20 is 5.2e-7
from it, and ASR at 150 / 200 / 300 within 9e-8, so its error is about 1e-7 and
smaller errors are not resolved).  TE references: single-element PMM degree 60
(agrees with degree 80 to 5e-12 on A, with degree 40 to 9e-12 on the
dielectric grating).  Error `|r - r_ref|`, `r` the zeroth-order co-polarized
`jones_reflection`; `asr=True` (the `'auto'` ramp: `eta` 0.5 at 10, 1 from 15).

| `n_orders` | 10 | 15 | 20 | 25 | 30 | 40 | 50 | 60 | 70 | 80 | 100 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A TM normal, `'laurent'` | 7.6e-1 | 3.4e-1 | 7.6e-1 | 3.4e-1 | 9.8e-2 | 1.8e-1 | 2.6e-1 | 2.8e-1 | 2.2e-2 | 9.0e-2 | 1.7e-1 |
| A TM normal, `'li'` | 3.8e-1 | 2.3e-2 | 4.2e-2 | 3.3e-2 | 2.4e-2 | 1.9e-2 | 3.9e-2 | 1.8e-2 | 1.2e-2 | 1.1e-2 | 1.1e-2 |
| A TM normal, ASR | 5.8e-2 | 3.5e-4 | 4.7e-6 | 3.2e-5 | 3.1e-6 | 2.0e-6 | 1.1e-6 | 1.9e-6 | 5.3e-7 | 1.3e-7 | 1.3e-7 |
| A TM theta 0.3, `'li'` | 3.8e-1 | 2.4e-2 | 4.5e-2 | 3.4e-2 | 2.5e-2 | 1.9e-2 | 3.9e-2 | 1.8e-2 | 1.2e-2 | 1.1e-2 | 1.1e-2 |
| A TM theta 0.3, ASR | 5.8e-2 | 3.5e-4 | 4.2e-6 | 3.2e-5 | 3.1e-6 | 1.8e-6 | 1.1e-6 | 1.9e-6 | 5.2e-7 | 1.3e-7 | 1.5e-7 |
| A TE normal, `'li'` | 8.3e-5 | 2.9e-5 | 1.1e-5 | 6.5e-6 | 3.8e-6 | 1.7e-6 | 8.2e-7 | 4.8e-7 | 3.1e-7 | 2.1e-7 | 1.1e-7 |
| A TE normal, ASR | 9.5e-6 | 5.6e-10 | 5.0e-11 | 4.9e-11 | 4.8e-11 | 4.7e-11 | 5.2e-11 | 5.1e-11 | 1.7e-11 | 1.2e-10 | 2.9e-10 |
| B TM normal, `'li'` | 1.2e-2 | 5.8e-3 | 4.5e-3 | 3.1e-3 | 6.4e-3 | 1.2e-2 | 6.9e-3 | 9.5e-4 | 1.4e-3 | 1.8e-3 | 8.4e-4 |
| B TM normal, ASR | 2.5e-3 | 8.3e-5 | 5.6e-7 | 2.0e-7 | 3.2e-7 | 1.7e-7 | 1.0e-7 | 9.1e-8 | 1.0e-7 | 8.2e-8 | 7.6e-8 |
| B TM theta 0.2, `'li'` | 1.3e-2 | 8.8e-3 | 5.5e-3 | 2.1e-3 | 3.3e-3 | 6.8e-3 | 3.7e-3 | 7.9e-4 | 9.4e-4 | 1.2e-3 | 4.5e-4 |
| B TM theta 0.2, ASR | 3.0e-3 | 7.5e-5 | 6.8e-7 | 1.0e-7 | 1.5e-7 | 3.7e-8 | 5.9e-8 | 7.4e-8 | 5.3e-8 | 6.9e-8 | 7.5e-8 |

The ASR TM floor is the reference's (about 1e-7); ASR's own self-agreement is
1.3e-8 (150 / 200).  The rise of the TE readings past 70 (1e-10 .. 3e-10) is
round-off from the stretch.  Odd `n_orders` read about 3x worse than their
even neighbours (25 above; the scatter table below).

**Lossless lamellar grating** (eps 4 / air, duty 0.4, 300 nm, on n = 1.5,
period 1000 nm, 800 nm, theta 0.2).  TE against single-element PMM degree 60;
TM against geometric PMM degree 24 x 12 (single-element PMM degree 40 and 60
differ by 1.3e-6 in TM, so TM errors below about 1e-7 are not resolved); the
closure is `|1 - sum R - sum T|` of the ASR solve (the plain basis closes to
round-off at every truncation).

| `n_orders` | 10 | 15 | 20 | 25 | 30 | 40 | 60 | 80 | 100 |
|---|---|---|---|---|---|---|---|---|---|
| TE, `'li'` | 2.9e-4 | 8.7e-5 | 3.7e-5 | 1.9e-5 | 1.1e-5 | 4.6e-6 | 1.4e-6 | 5.8e-7 | 2.9e-7 |
| TE, ASR | 1.9e-5 | 1.0e-5 | 1.7e-8 | 5.8e-12 | 2.1e-11 | 3.5e-11 | 8.0e-11 | 7.8e-10 | 7.2e-10 |
| TM, `'li'` | 4.3e-4 | 1.9e-4 | 1.0e-4 | 6.5e-5 | 4.5e-5 | 2.4e-5 | 1.0e-5 | 5.7e-6 | 3.6e-6 |
| TM, ASR | 9.7e-5 | 1.6e-6 | 9.1e-8 | 8.0e-8 | 7.7e-8 | 7.6e-8 | 7.5e-8 | 7.6e-8 | 7.5e-8 |
| ASR closure | 7.1e-9 | 2.2e-10 | 9.1e-11 | 4.0e-12 | 2.7e-11 | 9.5e-11 | 3.2e-10 | 2.6e-9 | 1.5e-9 |

Lossless closure is truncation-limited, not round-off-limited, under ASR (the
truncated `x`-projection of the reflected field is not exactly the truncated
system's flux).  On a three-layer dielectric stack with two patterned layers
it reads 1.5e-4 / 4e-6 / 1.3e-7 / 2.5e-10 at 10 / 15 / 20 / 40; below about
12 the library's lossless-closure warning fires, correctly flagging an
unconverged truncation.  Conical incidence (phi 0.3, same grating) closes to
6e-12 at 30.

**Neighbouring-N scatter**, fixture A TM, `n_orders` 40 .. 70 (31 consecutive
values): `'li'` spans 1.2e-2 .. 4.5e-2 (a factor 3.7 between neighbours);
ASR spans 3.6e-7 .. 5.0e-6, every value 2400x or more below the best `'li'`
value of the window.  The remaining ASR scatter is an odd/even alternation
(odd `n_orders` 2-5x worse), from 2.0e-6 / 5.0e-6 at 40 / 41 to 4e-7 / 1.4e-6
near 70.

**Cost** (fixture A, one solve, median of 3, BLAS 2 threads, shared machine):

| | `n_orders` | time | error |
|---|---|---|---|
| `'li'` | 100 / 200 / 400 | 0.32 / 1.87 / 9.9 s | 1.1e-2 / 4.9e-3 / 2.3e-3 |
| ASR | 15 / 20 / 40 / 100 | 0.02 / 0.04 / 0.17 / 1.6 s | 3.5e-4 / 4.7e-6 / 2.0e-6 / 1.3e-7 |

At equal `n_orders` an ASR solve costs 3-5x a `'li'` solve (0.041 s against
0.012 s at 20, 1.75 s against 0.36 s at 100: two first-order eigensolves per
patterned layer, the oversampled half-space basis and the plane-wave columns).
At equal accuracy it is far cheaper: `'li'`'s best value here (2.3e-3 at 400,
9.9 s) is beaten by ASR at 15 (3.5e-4, 0.02 s), and `'li'` would need of the
order of 1e4 orders for ASR's 5e-6 at 20 (0.04 s).

## What composes, what is refused

Composes: uniform layers and `'li'` `shapes=` / `segments=` layers (several
patterned layers with different walls); classical and conical incidence;
dispersive materials through `solve_vs_wavelength`; `stabilize=True`;
`rcwa_convergence`; `retain_internal=True` with `RCWAResult.internal_field`
(fields from the `u` harmonics at `u(x)`; `E_x` from the continuous `D_x`
divided by `eps(x)` -- the mean of the two sides on a wall --, `H_x` and
`H_z` through `[[f]]^-1`, never a division by `f`) and
`RCWAResult.layer_absorption` (every layer's face-flux difference, summed
exactly over the `u` harmonics).  The even-parity fold does not engage; the
solve is the full one (`symmetry` has no effect).  A stack without patterned
layers, or `eta='auto'` at `n_orders <= 5`, runs the standard solve bit for
bit.

Refused with a `ValueError`: 2-D stacks; `eps_cell` rasters (no exact wall
positions to put on coordinate lines -- pass the layer as `segments=`);
tensor layers; patterned layers solved with `'laurent'`; JAX-traced inputs;
`use_gpu=True`.  Bad `asr` values raise at construction.

## Limits and follow-ups

* `rcwa_efficiency_1d(asr_eta=...)` (single layer) still uses the `G^-1`
  bridge and the chain-rule form, i.e. it keeps its documented ~1e-4 TM
  floor.  Routing it through this engine (a one-layer stack) would remove it;
  not done in this change.
* Wall positions closer than 1e-9 of the period are merged onto one
  coordinate line.  A very narrow segment (a 1 nm liner in a 370 nm period)
  gets its share of the `u` axis in proportion to its width, like the plain
  basis; a `u` partition that widens narrow segments is a possible
  extension.
* `eps_cell` rasters could be supported by reading walls off the pixel
  boundaries, at the price of a different pixel semantics than the plain
  path's point samples.
