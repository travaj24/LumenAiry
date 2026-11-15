# Adaptive spatial resolution for `RCWAStack` (1-D)

`RCWAStack(..., asr=True)` solves a 1-D multilayer in one stretched lateral
coordinate whose lines carry the walls of every patterned layer, with the
Fourier harmonics clustered at the walls.  On TM metal-strip stacks, where the
uniform basis converges only as about `N^-1.2`, it reaches 3e-6 at
`n_orders = 25` (2.5e-5 at 20) and 2e-7 (the accuracy of the reference) from
about 60 -- measured on the strip / spacer / mirror stack of the convergence
tables below.  That holds for stacks whose patterned segments are all wide on
the scale of the period: the stretched axis is shared among the segments in
proportion to their widths, so a narrow segment is not widened and converges
much more slowly (the same stack with a 1 nm wide metal segment added at the
strip's edge still errs 1.7e-5 at `n_orders` 80 and 1.2e-5 at 150; see the
limits at the end).  The implementation is
`lumenairy/elements/rcwa/_asr_stack.py`; its module docstring carries the
derivation.  Tests: `tests/unit/test_rcwa_stack_asr.py`.

```python
import lumenairy as lum
st = lum.RCWAStack(500e-9, n_superstrate=1.0, n_substrate=1.5, n_orders=30,
                   asr=True)                      # or asr=dict(eta=1.0)
st.add_layer(40e-9, segments=[(180 / 500, -90 + 7j), (320 / 500, 1.0)])
st.add_layer(19e-9, eps=3.6)
st.add_layer(80e-9, eps=-130 + 20j)
res = st.set_source(1500e-9, theta=0.0).solve()
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
   singularity is removed, not just resolved more finely.  The strip /
   spacer / mirror stack (below), TM, `eta = 1`, error against PMM degree
   24 x 12 (whose own error is about 1.5e-7):

   | `n_orders` | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 100 |
   |---|---|---|---|---|---|---|---|---|
   | `'sine'` | 4.0e-4 | 1.1e-4 | 3.3e-4 | 2.4e-5 | 9.8e-6 | 3.8e-5 | 4.6e-7 | 3.8e-6 |
   | `'sine4'` | 2.5e-5 | 2.9e-6 | 2.6e-6 | 2.3e-6 | 3.2e-7 | 4.3e-7 | 3.1e-7 | 2.1e-7 |

   Any `eta < 1` leaves a linear zone at the wall where the singularity keeps
   its exponent: `'sine4'` at `eta = 0.999` sits at 3.9e-6 / 2.8e-6 at 150 /
   200 where `eta = 1` is at 1.8e-7 / 1.5e-7 (the reference's accuracy).

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
   exact projected operators changed nothing, and treating the mirror of the
   strip / spacer / mirror stack as a half-space instead of an 80 nm layer
   left the error unchanged: 2.55e-6 either way at `n_orders` 40, each
   against its own PMM reference).  Rayleigh amplitudes are
   the exact `x`-Fourier coefficients of the truncated field.

5. **Balanced first-order eigensolve.**  With `f -> 0` the local wavenumber
   `k/f` near a wall is huge (largest modal eigenvalue at `n_orders` 100 on
   the strip / spacer / mirror stack: 1e6 with `'sine'`, 3e8 with
   `'sine4'`).  The squared operator `P Q` then has norm 1e12 (`'sine'`) to
   1e17 (`'sine4'`), and the eigensolver's absolute error swamps the `O(1)`
   eigenvalues.  On that stack the two solves still agree with `'sine'` at
   `n_orders` 80 .. 120; with the shipped `'sine4'` (`eta = 1`; `P Q` taken
   in the same uniform-eigenvector frame, TM block):

   | `n_orders` | 40 | 60 | 70 | 80 | 90 | 100 | 120 |
   |---|---|---|---|---|---|---|---|
   | eig of `P Q` | 2.5e-6 | 3.4e-7 | **2.5e-5** | **8.5e-5** | 5.0e-7 | 7.9e-7 | 1.7e-7 |
   | balanced first-order (shipped) | 2.6e-6 | 3.2e-7 | 4.3e-7 | 3.1e-7 | 2.1e-7 | 2.1e-7 | 1.8e-7 |

   The layer is solved as `[[0, P'], [Q', 0]]` in the frame of the uniform
   eigenvectors with a per-mode diagonal scaling that balances each `2 x 2`
   block; its norm is the largest eigenvalue, not its square.  It stays clean
   to `n_orders` 300 (150 / 200 / 300 agree to 8e-8).

**Default stretch** (`eta='auto'`): `eta = clip((n_orders - 5) / 10, 0, 1)`.
The full stretch resolves the middle of each segment 2.7 times more coarsely
than the plain basis; at `n_orders` 10 a fixed `eta = 1` was less accurate
than the plain basis on the dielectric grating (TE 2.1e-3 against 2.9e-4),
while the ramp (0.5 there) gives 1.9e-5.  With the ramp, ASR beat the plain
basis, or came within 5 % of it, at 63 of 66 points (`n_orders` 6..16; TE and
TM at normal incidence on the dielectric grating, the strip / spacer / mirror
stack and its wide-period variant); the exceptions are the dielectric TE at
7 (8.2e-4 against 7.3e-4), the strip stack's TM at 12 (1.2e-1 against
8.4e-2) and the wide-period TM at 10 (4.3e-2 against 2.8e-2), both at a dip
of the plain basis's own scatter.  At 5 and below the solve is the standard
one.  A fixed `eta < 1` is for studies only (item 3).  The guard against the
ill-conditioned extreme: the Toeplitz matrix of `f` (whose inverse square root
builds every uniform mode) is refused above condition number 1e13 (measured
2.3e5 / 1.6e7 / 2.5e8 at `n_orders` 50 / 150 / 300 on the strip stack's map,
`'sine4'` at `eta = 1`); the eigensolve itself is balanced (item 5).

## Convergence tables

**The strip / spacer / mirror stack**: a 180 nm strip (eps -90+7i, 40 nm)
over 2 / 6 / 3 / 6 / 2 nm of eps 2.8 / 4.2 / 2.2+0.6i / 4.2 / 2.8 over an
80 nm mirror (eps -130+20i), period 500 nm, air / n = 1.5, 1500 nm; TM sits on
the gap resonance (|r| = 0.20).  **The wide-period variant**: the same stack
at period 1600 nm with a 600 nm strip (orders +/-1 propagate in the
substrate).  References: `PMMStack` with geometric grading (ratio 0.17),
degree 24 x 12 elements (strip stack TM: -0.10122424808 - 0.17630843363i;
degree 20 is 1.3e-7 from it and degree 16 2.9e-7, and ASR at 150 / 200 / 300
sits 1.1e-7 .. 1.8e-7 from it while agreeing among themselves to 8e-8, so its
error is about 1.5e-7 and smaller errors are not resolved; on the wide-period
variant degree 20 is 2.6e-8 from degree 24).  TE references: single-element
PMM degree 60 (agrees with degree 80 to 9e-13 on the strip stack, with degree
40 to 9e-12 on the dielectric grating).  Error `|r - r_ref|`, `r` the
zeroth-order co-polarized `jones_reflection`; `asr=True` (the `'auto'` ramp:
`eta` 0.5 at 10, 1 from 15).

| `n_orders` | 10 | 15 | 20 | 25 | 30 | 40 | 50 | 60 | 70 | 80 | 100 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| strip TM normal, `'laurent'` | 7.0e-1 | 1.4e-1 | 2.6e-1 | 3.0e-1 | 3.8e-1 | 5.5e-1 | 1.4e-1 | 2.6e-1 | 2.3e-1 | 1.2e-1 | 2.8e-2 |
| strip TM normal, `'li'` | 2.7e-1 | 5.4e-2 | 3.0e-1 | 6.3e-2 | 1.3e-1 | 3.1e-2 | 1.9e-2 | 2.4e-2 | 2.6e-2 | 1.3e-2 | 1.5e-2 |
| strip TM normal, ASR | 6.0e-2 | 2.7e-4 | 2.5e-5 | 3.0e-6 | 2.9e-6 | 2.6e-6 | 2.3e-6 | 3.2e-7 | 4.3e-7 | 3.1e-7 | 2.1e-7 |
| strip TM theta 0.3, `'li'` | 2.6e-1 | 5.6e-2 | 3.0e-1 | 6.3e-2 | 1.3e-1 | 3.1e-2 | 1.9e-2 | 2.3e-2 | 2.6e-2 | 1.3e-2 | 1.5e-2 |
| strip TM theta 0.3, ASR | 5.8e-2 | 2.7e-4 | 2.6e-5 | 3.1e-6 | 2.9e-6 | 2.5e-6 | 2.1e-6 | 3.8e-7 | 5.1e-7 | 3.9e-7 | 3.0e-7 |
| strip TE normal, `'li'` | 9.6e-5 | 3.5e-5 | 1.7e-5 | 9.3e-6 | 5.2e-6 | 2.2e-6 | 1.2e-6 | 6.7e-7 | 4.4e-7 | 2.9e-7 | 1.5e-7 |
| strip TE normal, ASR | 1.1e-5 | 2.6e-8 | 2.5e-10 | 9.0e-12 | 7.9e-13 | 7.1e-12 | 3.7e-11 | 3.9e-11 | 2.0e-11 | 1.4e-10 | 7.6e-10 |
| wide TM normal, `'li'` | 2.8e-2 | 4.1e-2 | 1.0e-2 | 2.0e-2 | 6.9e-3 | 3.2e-2 | 6.0e-3 | 2.1e-2 | 1.9e-2 | 1.7e-3 | 2.6e-3 |
| wide TM normal, ASR | 4.3e-2 | 5.7e-3 | 5.7e-5 | 9.9e-7 | 2.1e-6 | 1.1e-6 | 1.3e-7 | 8.8e-8 | 1.2e-7 | 8.9e-8 | 4.3e-8 |
| wide TM theta 0.2, `'li'` | 2.1e-2 | 6.0e-2 | 1.1e-2 | 1.6e-2 | 1.1e-2 | 3.3e-2 | 6.6e-3 | 1.9e-2 | 1.9e-2 | 1.9e-3 | 2.8e-3 |
| wide TM theta 0.2, ASR | 5.8e-2 | 5.6e-3 | 5.8e-5 | 8.4e-7 | 1.8e-6 | 9.5e-7 | 1.7e-7 | 9.6e-8 | 1.1e-7 | 8.0e-8 | 3.8e-8 |

The strip stack's ASR TM floor is the reference's (about 1.5e-7); ASR's own
self-agreement is 2.4e-8 (150 / 200).  The rise of the TE readings past 70
(1e-10 .. 8e-10) is round-off from the stretch.  At `n_orders` 10 (`eta`
0.5) ASR is not yet ahead on the wide-period variant; from 15 on it is ahead
everywhere.  Consecutive `n_orders` scatter by up to about 6x (the scatter
paragraph below).

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

**Neighbouring-N scatter**, the strip / spacer / mirror stack, TM, `n_orders`
40 .. 70 (31 consecutive values): `'li'` spans 1.4e-2 .. 9.1e-2 (a factor 3.6
between neighbours); ASR spans 1.8e-7 .. 4.5e-6, every value 3100x or more
below the best `'li'` value of the window.  The remaining ASR scatter is up to
a factor 6 between neighbours, falling from 1.5e-6 .. 4.5e-6 near 40 to
2e-7 .. 8e-7 near 70.

**Cost** (the strip / spacer / mirror stack, one solve, median of 3, BLAS 2
threads, shared machine):

| | `n_orders` | time | error |
|---|---|---|---|
| `'li'` | 100 / 200 / 400 | 0.35 / 1.78 / 9.6 s | 1.5e-2 / 7.8e-3 / 3.0e-3 |
| ASR | 15 / 20 / 40 / 100 | 0.02 / 0.04 / 0.16 / 1.45 s | 2.7e-4 / 2.6e-5 / 2.6e-6 / 2.1e-7 |

At equal `n_orders` an ASR solve costs 3-5x a `'li'` solve (0.037 s against
0.010 s at 20, 1.45 s against 0.35 s at 100: two first-order eigensolves per
patterned layer, the oversampled half-space basis and the plane-wave columns).
At equal accuracy it is far cheaper: `'li'`'s best value here (3.0e-3 at 400,
9.6 s) is beaten by ASR at 15 (2.7e-4, 0.02 s), and `'li'` (about `N^-1.2`
here) would need of the order of 1e5 orders for ASR's 2.6e-6 at 40 (0.16 s).

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
* Wall positions closer than a billionth of the period
  (`_ASR_MERGE_TOL`) are merged onto one coordinate line.
* NARROW SEGMENTS ARE NOT WIDENED.  A very narrow segment gets its share of
  the `u` axis in proportion to its width, like the plain basis, so the
  accuracy quoted above holds only when every segment is wide on the scale of
  the period.  Measured on the strip / spacer / mirror stack with a 1 nm wide
  segment of the mirror's permittivity added at the strip's edge (the strip
  179 nm; reference PMM degree 24 x 12, degree 20 within 1.4e-7 of it): ASR
  errs 2.8e-4 / 2.4e-5 / 1.7e-5 / 1.2e-5 at `n_orders` 20 / 40 / 80 / 150,
  against 2.5e-5 / 2.6e-6 / 3.1e-7 / 1.8e-7 without that segment -- still
  about 1000x below the plain basis (2.9e-1 / 2.7e-2 / 1.3e-2 / 1.1e-2), but
  converging much more slowly.  A `u` partition that widens narrow segments
  is a possible extension.
* `eps_cell` rasters could be supported by reading walls off the pixel
  boundaries, at the price of a different pixel semantics than the plain
  path's point samples.
