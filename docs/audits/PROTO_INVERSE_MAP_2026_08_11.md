# PROTOTYPE -- the INVERSE (mixed) characteristic for the design-121 traced fan

**2026-08-11.  Working tree on branch `fix/tilt-quadratic-opl` @ `a185cfc`, two
commits ahead of `main` @ `755ad99` (v5.34.0): `63e6905` (the tilt-quadratic OPL
piston fix -- the uncommitted working-tree edit `PROTO_HAMILTON_MAP_2026_08_11`
ran on, committed mid-session) and `a185cfc` (`carrier._exact_sphere_eikonal`
rewritten into its cancellation-free algebraic equivalent, also mid-session).
Both are inert for everything measured here: `63e6905`'s own commit message pins
`opl_grid` as expression-identical, so the fits / Newton / masks it feeds are
bit-for-bit, and `_exact_sphere_eikonal` is not on any path this prototype
exercises.  NO file under `lumenairy/**` was touched by this prototype; no
`git`, no `gh`, no `CHANGELOG`.  Everything added is
`validation/repro_traced_carrier_121/imap_*.py` plus this note.**

Answers the one item `PROTO_HAMILTON_MAP_2026_08_11` S8.3 left unmeasured:

> *"REDIRECT the speed work to the INVERSE (mixed) characteristic, which is
> where the 419 s actually is. ... It is UNMEASURED here -- in particular the
> Chebyshev degree needed in the EXIT coordinates."*

---

## 0. VERDICT

> **The redirect's PREMISE is false.  Its CONCLUSION survives anyway, on a
> different mechanism, at a smaller but genuinely material size -- and the bar
> it proposed sizing the map to would have shipped a 24x silent regression.**
>
> **1. THE PREMISE, refuted by direct measurement.**  S8.3 sized the redirect
> on *"the Newton pull-back of every one of `n_fine**2 = 67.1e+06` exit
> pixels"*.  `apply_real_lens_traced` does not do that.  `sub = ray_subsample`,
> and the OPL inversion runs on `X[::sub, ::sub]`: at the shipped retrace
> configuration (`n_fine = 8192`, `sub = rs_fine = 87`) that is a
> **95 x 95 = 9 025-point** lattice, Newton'd twice (once for the OPL, once for
> the ray-density amplitude) and then INTERPOLATED to the wave grid.  Read out
> of the library's own per-iteration INFO log: **18 050 Newton points =
> 0.0269 % of the 67 108 864 exit pixels, in 0.010 s = 0.010 % of a 96.9 s
> element.**  `_invert_newton` does not appear in the cProfile top-45 at all.
> There is no 419 s of Newton; removing the Newton entirely buys 0.010 s per
> order.  The per-pixel iteration count is **2.05-2.73, not 12** -- the
> paraxial-magnification initial guess is already good.
>
> **2. THE CONCLUSION, upheld on the UPSAMPLE.**  The per-exit-pixel cost the
> element really pays is the coarse lattice's interpolation.  Measured with a
> spy on `scipy.ndimage.map_coordinates`: **8 full-grid calls, 17.886 s =
> 15.6 % of the element** (cProfile independently reads the same bucket at
> 16.4 %), of which **six -- 14.767 s -- are exactly the four channels G would
> produce EXACTLY, in one pass**: the OPL (order 3), its NaN mask, the
> ray-density amplitude, its NaN mask, and the entrance coordinates `xe`, `ye`
> that `remap_sampling='full'` pulls the input residual back through.
> Measured against them: **one numba evaluation of a total-degree-14 Chebyshev,
> 4 channels, over 6.71e+07 points = 1.910 s** -- against **4.035 s for the
> single order-3 `map_coordinates` it replaces**, and against **39.9-45.8 s**
> for the per-pixel Newton the redirect thought it was replacing.
>
> **3. THE BAR IS THE FINDING.**  On this group the shipped path already
> delivers **1.11e-04 waves** (forward fit + Newton 6.65e-05, measured; cubic
> upsample 4.44e-05, measured inside the landing hull) -- **lambda/9 000**.
> The requested `lambda/100` bar is **90x looser than the incumbent**.  G sized
> to `lambda/100 with 3x margin` is `7 x 6 = 42` nodes x exit degree 10 =
> **2.61e-03 waves**, which is **24x WORSE than the path it replaces**: a
> plausible-looking wrong answer of exactly the class this campaign exists to
> remove.  **The node/degree budget must be set by PARITY, not by
> lambda/100.**  Parity is `9 x 8 = 72` nodes x exit degree **14** ->
> **3.84e-05 waves**, 2.9x better than the incumbent, **276.5 kB**, **6.00 s**
> to build.
>
> **4. NET.**  Projected from measured parts: **12.05 s per order = 10.5 % of
> the instrumented element** (12.4 % of the clean one), the element being
> 61.1 % of an order, so **~6.4-7.6 % of a 32-order run** -- against a
> ONE-OFF 6.00 s build.  Contrast the forward map's 0.0006 %.
>
> **Recommendation: NO-GO on the redirect as stated (there is no per-pixel
> Newton to remove, and its lambda/100 sizing is a regression).  GO on the
> inverse characteristic as an EXACT per-pixel evaluator replacing the
> coarse-lattice-plus-upsample chain, sized to parity at 72 nodes x degree 14,
> with three NEW guards (G6-G8) the inverse side needs and the forward side did
> not.**  Design in S6.

---

## 1. BOX, BUILD, CONFIGURATION

```
Windows 11 Pro 10.0.26200        AMD Ryzen 9 5950X, 24 logical CPUs
137.4 GB physical RAM            83.8 GB free at launch
python 3.14.6   numpy 2.4.4   scipy 1.17.1   numba 0.65.1
lumenairy 5.34.0                 branch fix/tilt-quadratic-opl @ a185cfc
numba gate _lens_traced._NUMBA_AVAILABLE = True
_NEWTON_MAX_ITERS 12   _DECENTRED_FIT_POLY_ORDER 10
_FIT_RADIUS_BEAM_FACTOR_DEFAULT 2.0
```

**LOAD CAVEAT, UP FRONT.**  The box carried other campaigns throughout (29
python processes, CPU at 100 % when this study started).  Every verdict is
either a RATIO taken inside one process or a best-of-3 timing; absolute seconds
are upper bounds on an idle box.  The SAME element call was measured four times
under four instrumentations -- **96.9 s** (clean + log handler), **92.9 s**
(cProfile), **114.7 s** (`map_coordinates` spy, while a stale 11 GB job from
this study's own earlier step was still resident), **91.4 s** (evaluator spy).
The two independent measurements of the `scipy.ndimage` bucket agree at 15.6 % /
16.4 %, and that agreement is what the verdict rests on.

The element under test is driven exactly as `_fine_trace_group_exit` drives it
at `n_fine_cap = 8192`, with `propagate_traced_carrier_chain`'s own `base_kw`:

```
prescription  design 121's LAST post-DOE group (doublet SK2/SF57)
n_fine 8192   dx 1.5324 um   window 12.5534 mm   ray_subsample 87
carrier       order (-4,-2)'s own TiltedCarrier at that group's entrance
amplitude_model='ray_density'   preserve_input_phase='remap'
remap_sampling='full'           fit_radius_beam_factor=2.0
n_launch 229 (52 441 rays)      COARSE Newton lattice 95 x 95 = 9 025
```

---

## 2. WHAT G IS

```
G(x_out, y_out ; x_src, y_src)  ->  (x_in, y_in, OPL, det J)
```

* **Source axes** -- tensor Chebyshev-Lobatto NODES on the source-label box.
  This is the parametrisation `PROTO_HAMILTON_MAP_2026_08_11` S2.1 proved
  EXACT: `_tilted_carrier_parts` depends on `(x0, y0, L, M)` only through
  `(x_src, y_src) = (x0 - R L/N, y0 - R M/N)`, and design 121's last-group
  entrance carries ONE order-independent `R = -21.139185 mm`, so the 32-order
  fan is a two-parameter family and a node in `(x_src, y_src)` hits every order
  with no parametrisation residual.  Box padded by the niche-C6 widening
  `max|grad a_fit| x |R| = 0.6884 mm`: half-widths **2.4216 / 1.4330 mm**
  against the unpadded **1.7332 / 0.7446 mm**.
* **Exit axes** -- total-degree Chebyshev LEAST SQUARES of degree `d` from that
  node congruence's scattered ray landings.  Same basis the shipped
  `inversion_method='fit'` path (`_invert_fit`) uses, so "degree" here means
  what `newton_poly_order` means there.
* **ONE COMMON exit normalisation box** shared by every node -- measured centre
  `(-0.0900, -0.0929) mm`, half-widths `(4.7347, 4.3683) mm`.  Without a common
  box the source-axis contraction is not a contraction of one function.  This
  is a structural requirement the forward map did not have, and it is what
  makes guard G6 (S5.3) necessary.
* **Build domain** -- each node congruence is traced on the SHIPPED launch
  lattice (`linspace(-15.2974, +15.2974 mm, 229)`, square, 52 441 rays) and the
  exit fit takes the 16 237 samples inside the UNION entrance pupil
  `r <= 9.6490 mm`.  `det J` comes from `np.gradient` on that regular lattice
  -- no extra rays.
* **Per-order use** -- contract the source axes at the order's own
  `(x_src, y_src)` (`nx * ny * P * 4` multiply-adds, microseconds), then ONE
  polynomial evaluation per exit pixel.

**Scoring.**  Against the DIRECT RAY TRACE at OFF-NODE exit pixels (the test
entrance lattice is offset half a build cell, so no test exit point is a fit
sample) and OFF-NODE source labels (the 32 orders' own, including the extreme
`(-4,-2)`): 22 624 exit pixels over the 32 orders.  The INCUMBENT column is the
shipped per-pixel Newton, reproduced verbatim from `_invert_newton` on the
library's own `_Cheb2DEvaluator` forward fits, over the SAME ray data and the
SAME sample domain -- so the only difference between the arms is
forward-fit-then-invert versus direct inverse fit.

---

## 3. MEASUREMENT 1 -- THE EXIT DEGREE AND THE NODE COUNT

`imap_probe_121.py degree` (~75 s), `imap_cost_121.py fitorder` (~91 s).

### 3.1 The exit degree alone (ONE congruence, no source interpolation)

Maximum least-squares residual over the 16 237 union-pupil samples of one node:

| exit degree | terms | `x_in` (nm) | OPL (waves) |
|---|---|---|---|
| 4  | 15  | 5.87e+03 | 2.15e+00 |
| 6  | 28  | 2.39e+02 | 1.24e-01 |
| 8  | 45  | 1.45e+00 | 6.25e-03 |
| 10 | 66  | 1.23e+00 | 3.01e-04 |
| 12 | 91  | 1.45e-01 | 2.41e-05 |
| 14 | 120 | 1.24e-02 | 3.24e-06 |
| 16 | 153 | 8.82e-04 | 2.97e-07 |
| 18 | 190 | 4.88e-05 | 2.09e-08 |

Geometric, ~1 decade per two degrees.  **The shipped path's degree 6 in
ENTRANCE coordinates is nowhere near enough in EXIT coordinates** -- 0.124
waves against the 6.65e-05 the same 66-term budget buys on the forward side
(S3.4).  That asymmetry is the central structural fact of this study.

### 3.2 The full 4-D map, C6-PADDED (the production box)

OPL error, waves, maximum over the 32 orders x 22 624 off-node exit pixels.
**Bold** = clears `lambda/100` with 3x margin (3.33e-03).

| nodes | deg 6 | deg 8 | deg 10 | deg 12 | deg 14 |
|---|---|---|---|---|---|
| 4 x 4 = 16  | 1.93e+00 | 1.84e+00 | 1.83e+00 | 1.83e+00 | 1.83e+00 |
| 5 x 4 = 20  | 3.66e-01 | 3.26e-01 | 3.24e-01 | 3.24e-01 | 3.24e-01 |
| 6 x 5 = 30  | 1.32e-01 | 2.66e-02 | 2.48e-02 | 2.46e-02 | 2.46e-02 |
| 7 x 6 = 42  | 1.21e-01 | 7.73e-03 | **2.61e-03** | **2.43e-03** | **2.44e-03** |
| 8 x 7 = 56  | 1.22e-01 | 7.65e-03 | **9.60e-04** | **2.60e-04** | **3.24e-04** |
| 9 x 8 = 72  | 1.22e-01 | 7.50e-03 | **8.12e-04** | **8.97e-05** | **3.84e-05** |
| 10 x 9 = 90 | 1.22e-01 | 7.50e-03 | **8.17e-04** | **8.99e-05** | **1.75e-05** |

Entrance-position error tracks it (7 x 6 / deg 10: 3.77 nm; 9 x 8 / deg 14:
0.032 nm).  Storage runs 88.7 kB (7 x 6 / 10) to 345.6 kB (10 x 9 / 14).

**Two axes, both real, and they bind in different places.**  Reading down a
column gives the SOURCE-axis convergence (geometric, ~1 decade per node added
to each axis, saturating once the exit degree binds); reading across a row
gives the EXIT-degree floor of S3.1.  The total is the max of the two.  At
`9 x 8` and beyond the source axis is converged and the degree column IS the
answer: 7.50e-03 / 8.1e-04 / 9.0e-05 / 1.8e-05 for degrees 8 / 10 / 12 / 14.

### 3.3 The niche-C6 pad costs exactly ONE node per axis

Unpadded control (`mode_degree(pad=0.0)`), same table:

| nodes | deg 8 | deg 10 | deg 12 | deg 14 |
|---|---|---|---|---|
| 5 x 4 = 20  | 3.70e-02 | 3.46e-02 | 3.45e-02 | 3.45e-02 |
| 6 x 5 = 30  | 7.98e-03 | **2.49e-03** | **2.41e-03** | **2.40e-03** |
| 7 x 6 = 42  | 7.50e-03 | **8.16e-04** | **1.72e-04** | **1.60e-04** |
| 8 x 7 = 56  | 7.50e-03 | **8.15e-04** | **8.83e-05** | **2.34e-05** |
| 9 x 8 = 72  | 7.50e-03 | **8.15e-04** | **8.81e-05** | **8.35e-06** |

The padded `7 x 6` reads what the unpadded `6 x 5` reads, the padded `8 x 7`
what the unpadded `7 x 6` reads, and so on down.  **The C6 widening applies to
the inverse map exactly as it applied to the forward one, and its price is one
node on each axis** -- the same shape of answer as
`PROTO_HAMILTON_MAP_2026_08_11` S3.4 (20 -> 30 nodes there).

### 3.4 THE INCUMBENT, and why `lambda/100` is the wrong bar

Same rays, same sample domain, same 22 624 test points; the only variable is
the forward fit's order.

| forward fit order | terms | OPL max (waves) | `x_in` max (nm) | amplitude rel max | Newton iters mean / max |
|---|---|---|---|---|---|
| 6 (`newton_poly_order` default) | 28 | 1.31e-02 | 88.1 | 1.76e-04 | 2.05 / 3 |
| 8 | 45 | 1.44e-03 | 9.86 | 2.66e-05 | 2.05 / 3 |
| **10 (`_DECENTRED_FIT_POLY_ORDER`)** | **66** | **6.65e-05** | **0.381** | **1.03e-06** | 2.05 / 3 |
| 12 | 91 | 2.62e-06 | 0.0145 | 4.22e-08 | 2.05 / 3 |

**Which one is actually in force is not a guess.**  Wrapping the module-global
`_Cheb2DEvaluator` and running the real element (`imap_cost_121.py fitorder`)
shows **five fits built, four of them at order 10 and weighted** -- the
niche-C11 arbiter takes the OFF-CENTRE branch, because the launch lattice is
axis-centred while the beam sits at the chief ray, and
`_decentred_fit_restriction` then raises the order to
`_DECENTRED_FIT_POLY_ORDER = 10`:

```
0  order 10  (66 terms)  52441 samples  weighted=True
1  order  6  (28 terms)  52441 samples  weighted=False
2  order 10  (66 terms)  52441 samples  weighted=True
3  order 10  (66 terms)  52441 samples  weighted=True
4  order 10  (66 terms)  52441 samples  weighted=True
```

So the incumbent's error budget on this group is

```
forward fit + Newton      6.65e-05 waves   (S3.4, measured)
cubic upsample            4.44e-05 waves   (S5.1, measured in-hull)
                          --------
worst-case sum            1.11e-04 waves = lambda/9 000
```

against which `lambda/100` is **90x looser**.  Sizing G to `lambda/100 with 3x
margin` gives `7 x 6 = 42` nodes x degree 10 = 2.61e-03 waves -- **24x worse
than the code it replaces, with no warning and no energy to show for it.**

> **NODE COUNT OF RECORD: 72 nodes, `nx = 9` x `ny = 8`, exit degree 14**, on
> the C6-padded source-position box.  **3.84e-05 waves**, i.e. 2.9x INSIDE the
> incumbent's own 1.11e-04, 260x inside lambda/100.  **276.5 kB**, **6.00 s**
> to build.  `10 x 9` x 14 buys another 2.2x for 69 kB and 1.3 s and is the
> margin option; `8 x 7` x 12 (2.60e-04, 163 kB) is 2.3x OUTSIDE parity and is
> the row to refuse.

### 3.5 The amplitude is never binding -- and its floor is the ESTIMATOR's

The amplitude error saturates at **1.58e-05** relative for EVERY node count and
EVERY exit degree in S3.2.  A quantity that moves with neither axis of the
interpolant is not an interpolation error, so `imap_probe_121.py ampctl` asks
what it is:

```
np.gradient on the 229-point launch lattice (134.2 um pitch)
  vs the 5 um central stencil        amplitude rel  max 3.17e-05  rms 2.11e-05
ANALYTIC gradient of the order-10 forward Chebyshev fit
  vs the same                        amplitude rel  max 1.58e-06  rms 1.67e-07
```

**The floor is this prototype's `det J` build recipe, not the map.**  The
production recipe -- fit `det J` from the analytic Chebyshev gradient, which is
what `_ray_density_amp_grid` already does -- is a decade better, and the
incumbent's own amplitude at order 10 is 1.03e-06.  Either way the amplitude is
3-4 decades easier than the OPL and never sets the node count, which is the
same ordering the forward map found (S6.5 there) and worth having measured,
because `amplitude_model='ray_density'` consumes exactly this Jacobian.

---

## 4. MEASUREMENT 2 -- COST

`imap_cost_121.py newtonsite / profile / mapcoords / evalcost / newtonpp /
build`.

### 4.1 The measurement that refutes the premise

One real element call, clean, with a handler attached to the module's own
INFO log (which already emits one record per Newton iteration carrying the
point count -- attaching a handler only stops them being discarded):

```
n_fine 8192   dx 1.5324 um   ray_subsample 87   order (-4,-2)
n_launch 229 (52 441 rays);  exit pixels 6.711e+07
COARSE Newton lattice 95 x 95 = 9 025 points
element wall 96.9 s   peak working set 20.5 GB
  Newton call 1:  9 025 points, 12 iterations, 0.006 s, 596 active at the cap
  Newton call 2:  9 025 points, 12 iterations, 0.004 s, 596 active at the cap
TOTAL Newton points 18 050 = 0.0269 % of the 67 108 864 exit pixels
TOTAL Newton seconds 0.010  = 0.010 % of the element
```

The mechanism is `sub = max(1, int(ray_subsample))` and
`Xs = X[::sub, ::sub]` -- the same F-C pitch-preserving `ray_subsample` rescale
that pinned `n_launch` at 229 in the forward prototype also pins the Newton
lattice at `ceil(8192/87)**2`.  The second call is
`_ray_density_amp_grid`, which Newtons the same coarse lattice again.

### 4.2 Where the element's seconds actually are

cProfile, same call (92.9 s wall WITH profiler overhead; the shares are the
result):

| site | ncalls | tottime | share | cumtime |
|---|---|---|---|---|
| `scipy.ndimage` C entry points | 9 | 15.249 | **16.4 %** | 15.249 |
| `_ResidualEikonal._poly` | 18 | 13.410 | 14.4 % | 25.391 |
| `_ResidualEikonal._mul` | 1287 | 11.717 | 12.6 % | 11.717 |
| `apply_real_lens_traced` (inline assembly) | 1 | 9.447 | 10.2 % | 92.873 |
| `_TracedExitSupport.signed_distance` | 2 | 5.893 | 6.3 % | 7.020 |
| `_pip_residual_ri` | 1 | 5.107 | 5.5 % | 33.876 |
| `_carrier_residual_rms` | 1 | 4.062 | 4.4 % | 6.028 |
| `surface_sag_general` (inside `apply_real_lens`) | 48 | 3.657 | 3.9 % | 3.910 |
| `_ray_density_amp_grid` | 1 | 0.162 | 0.17 % | **0.261** |
| `_Cheb2DEvaluator.__init__` (the forward fits) | 5 | 0.132 | 0.14 % | **0.471** |
| `_invert_newton` | -- | -- | -- | **absent from the top-45 (< 0.033 s)** |
| `raytrace.trace` | -- | -- | -- | **absent from the top-45 (< 0.033 s)** |

`_pip_sample_residual` carries 37.511 s cumulative (40.4 %) -- the
`preserve_input_phase='remap'` + `remap_sampling='full'` machinery.  **The
entire traced-map apparatus the redirect targeted -- trace, five Chebyshev
fits, two Newtons, the Jacobian -- is 0.744 s, 0.8 % of the element.**

### 4.3 The `map_coordinates` census -- what G would actually replace

`imap_cost_121.py mapcoords` spies the scipy entry point (the module imports it
inside the function, so patching `scipy.ndimage` reaches the call site).  One
element call, 114.7 s wall:

| # | input | output points | order | s | replaceable by G? |
|---|---|---|---|---|---|
| 0 | (95, 95) | 6.711e+07 | 3 | 4.035 | **YES** -- the OPL upsample |
| 1 | (95, 95) | 6.711e+07 | 1 | 2.396 | **YES** -- the OPL NaN mask |
| 2 | (8192, 8192) | 9 025 | 1 | 0.002 | changes -- `\|E_in\|` moves to the fine entrance points |
| 3 | (95, 95) | 6.711e+07 | 1 | 1.875 | **YES** -- the ray-density amplitude |
| 4 | (95, 95) | 6.711e+07 | 1 | 2.316 | **YES** -- its NaN mask |
| 5 | (95, 95) | 6.711e+07 | 1 | 1.838 | **YES** -- the entrance `xe` |
| 6 | (95, 95) | 6.711e+07 | 1 | 2.307 | **YES** -- the entrance `ye` |
| 7 | (8192, 8192) | 6.711e+07 | 1 | 1.612 | no -- residual phasor (real) |
| 8 | (8192, 8192) | 6.711e+07 | 1 | 1.506 | no -- residual phasor (imag) |
| | | | | **17.886 (15.6 %)** | replaceable **14.767** |

### 4.4 The evaluation cost, MEASURED

One total-degree Chebyshev evaluation over 6.711e+07 points, numba kernel of
exactly the shape of the shipped `_cheb2d_val_grad_numba` (3-term recurrence on
the stack, `prange`, `fastmath`), best of 3:

| exit degree | terms | 3 channels | 4 channels |
|---|---|---|---|
| 8  | 45  | 0.601 s | 0.806 s |
| 10 | 66  | 0.866 s | 1.144 s |
| 12 | 91  | 1.128 s | 1.581 s |
| **14** | **120** | **1.546 s** | **1.910 s** |

For scale, on the same points:

```
scipy map_coordinates order-3 (the shipped OPL upsample)      3.498 s
scipy map_coordinates order-1                                 1.764 s
numpy chunked design-matrix @ coef, degree 10 (the shape
  shipped _invert_fit uses)                                  38.356 s
per-pixel NEWTON on the real forward fit, 2.39-2.73 iters
  mean:  0.379-0.683 us/pt  ->  at 6.711e+07 px            25.4-45.8 s
```

Two readings worth keeping.  **The kernel shape matters as much as the
degree**: the shipped `_invert_fit` form is 39x slower than the recurrence
kernel at the same degree, and its `(6.71e+07, 66)` float64 design matrix is
35 GB unchunked.  And S8.3's own "one polynomial evaluation instead of ~24" is
**real as a ratio (13-24x) and irrelevant as a saving**, because the shipped
path evaluates neither -- it interpolates.

### 4.5 Build cost and storage

Cold builds, no cache, each node traced on the shipped 229-point launch lattice:

| nodes | exit deg | terms | rays | trace | fit | **cold total** | storage | at-node OPL residual |
|---|---|---|---|---|---|---|---|---|
| 7 x 6 = 42 | 10 | 66 | 2.20e+06 | 0.61 s | 1.40 s | **2.02 s** | 88.7 kB | 2.31e-03 w |
| 8 x 7 = 56 | 12 | 91 | 2.94e+06 | 0.80 s | 2.87 s | **3.69 s** | 163.1 kB | 2.48e-04 w |
| **9 x 8 = 72** | **14** | **120** | 3.78e+06 | 0.95 s | 5.04 s | **6.00 s** | **276.5 kB** | **2.69e-05 w** |
| 10 x 9 = 90 | 14 | 120 | 4.72e+06 | 1.03 s | 6.27 s | **7.32 s** | 345.6 kB | 2.69e-05 w |

The build is dominated by the least-squares fits, not the rays (0.95 s of 6.00 s
traces 3.78 million of them).  Storage is a non-issue at any row.

### 4.6 THE COST TABLE, per order, and the 32-order projection

Measured items in bold; the G column is a PROJECTION from measured parts (no
library edit was made, so no end-to-end G run exists).

| stage | shipped | with G (deg 14, 4 channels) |
|---|---|---|
| ray trace, 52 441 rays | **0.012 s** | 0 |
| 5 x `_Cheb2DEvaluator` forward fits | **0.471 s** | 0 |
| 2 x coarse Newton, 9 025 pts | **0.010 s** | 0 |
| ray-density Jacobian + `\|E_in\|` at 9 025 pts | **0.261 s** (incl. above) | -- |
| G contraction at the order's label | -- | ~0 (5 760 mult-adds x 4) |
| upsamples #0, #1, #3, #4, #5, #6 | **14.767 s** | 0 |
| G evaluation over 6.711e+07 px | -- | **1.910 s** |
| `\|E_in\|` sampled at 6.711e+07 entrance points | **0.002 s** (coarse) | ~1.55 s (the measured cost of #7 / #8) |
| residual phasor #7 + #8 | **3.118 s** | **3.118 s** (unchanged) |
| domain mask | `signed_distance`, **5.893 s**, already full-grid | reused, 0 |
| **total in this bucket** | **18.63 s** | **6.58 s** |

```
NET SAVING           12.05 s per order
  as a share of the instrumented element (114.7 s)       10.5 %
  as a share of the clean element (96.9 s)               12.4 %
  the element is 61.1 % of an order (419.2 / 686.3 s,
  PROTO_HAMILTON_MAP_2026_08_11 S6.1)           -> 6.4-7.6 % of an order
32 orders                                          ~386 s of element time
ONE-OFF build (9 x 8 x deg 14)                          6.00 s, 276.5 kB
break-even                                    0.50 orders
```

Compare `PROTO_HAMILTON_MAP_2026_08_11` S6.3's forward map: break-even at 30
orders, saving 0.0006 %.  **The inverse side is four decades better because it
attacks per-PIXEL work instead of per-RAY work.**

---

## 5. MEASUREMENT 3 -- EDGE HONESTY AND SINGLE-VALUEDNESS

### 5.1 What the coarse-lattice upsample actually costs in accuracy

`imap_probe_121.py upsample`.  Newton the coarse `X[::87, ::87]` lattice,
cubic-upsample it exactly as the library does (`_opl_up_order = 3` whenever a
carrier is set, plus the order-1 NaN pass), and difference against the SAME
Newton run directly at 1 048 576 fine pixels.

**The mask is not optional, and getting it wrong is instructive.**  Unmasked,
the difference reaches **1.12e+04 waves** in the outer bins.  That is not the
upsample: outside the convex hull of the ray landings there is no ray, both
arms are EXTRAPOLATING the same degree-10 fit, and they extrapolate
differently.  The shipped path never uses those pixels either
(`_TracedExitSupport` tapers them out).  Inside the hull, with the library's own
`sqrt(2) * sub * dx = 188.5 um` plateau inset (281 054 pixels, reach 6.195 mm):

| region | fit order 6 | fit order 10 |
|---|---|---|
| all in-hull, max | 4.42e-05 w | **4.44e-05 w** |
| all in-hull, rms | 1.33e-05 w | 1.34e-05 w |
| `r/w` 0.0-0.5 | 4.10e-05 | 4.12e-05 |
| `r/w` 0.5-1.0 | 4.42e-05 | 4.44e-05 |
| `r/w` 1.0-1.5 | 4.03e-05 | 4.03e-05 |
| `r/w` 1.5-2.0 | 1.67e-05 | 1.67e-05 |

**The R7 cubic upsample is doing its job: 4.4e-05 waves, flat in radius, and
independent of the forward fit's order (it is a property of the 133.3 um coarse
pitch, not of the fit).**  So there is **no accuracy case** for an exact
per-pixel inverse -- only a cost case.  That is the second half of why S3.4's
parity framing is the right one: G is not buying accuracy, so it must not
spend any.

### 5.2 Error growth toward the fit-domain rim

`imap_probe_121.py edge`, two arms per order: test points from the ORDER's own
pupil (the production domain) and from the whole UNION pupil, which walks them
out to the map's own fit-domain rim.  Ten radial deciles of the exit footprint,
7 x 6 nodes:

| deg | order | arm | rim reach | `\|u\|` max | decile 0.0-0.1 | decile 0.9-1.0 |
|---|---|---|---|---|---|---|
| 8  | (-4,-2) | order pupil | 2.487 mm | 0.920 | 5.72e-03 | 1.28e-02 |
| 8  | (-4,-2) | union pupil | 3.915 mm | 0.951 | 6.82e-03 | 1.20e-02 |
| 10 | (-4,-2) | order pupil | 2.487 mm | 0.920 | 2.00e-03 | 1.58e-03 |
| 10 | (-4,-2) | union pupil | 3.915 mm | 0.951 | 2.09e-03 | 1.45e-03 |
| 10 | (0,0)   | union pupil | 3.925 mm | 0.884 | 1.20e-03 | 1.07e-03 |
| 12 | (-4,-2) | union pupil | 3.915 mm | 0.951 | 1.59e-03 | 1.68e-03 |

**PASS, and the frbf / skirt pattern does NOT reproduce here.**  At the node
count of record the error is FLAT to the rim -- the outermost decile is if
anything BETTER than the centre, because a Chebyshev least-squares fit
distributes its error toward the endpoints by construction rather than
concentrating it there.  Only degree 8 -- already below parity -- shows a 2x
outer-decile uptick.  Every order's exit footprint sits at `|u| <= 0.951` of the
common exit box, so there is no Chebyshev extrapolation in the normalised
coordinate anywhere in production use.

**The guard is nevertheless mandatory and it is the S5.1 number that sizes it**:
one plateau outside the landing hull, the answer is 1e+04 waves wrong.  The
guard already exists in the file -- `_TracedExitSupport.half_planes` /
`signed_distance`, which `_invert_fit` applies and which
`apply_real_lens_traced` already evaluates on the full grid (5.893 s in S4.2).
G inherits it at zero cost.

### 5.3 Node-hull coverage -- the guard the INVERSE needs and the forward did not

Contracting the source axes BLENDS the node polynomials, so an exit pixel
outside ONE node's landing hull mixes that node's EXTRAPOLATION into the answer
even though the contracted order's own support is fine.  Measured, worst over
32 orders x 42 node hulls (positive = outside):

```
node hulls cut to the r <= r_union FIT samples : (-4,-2) at +1.7829 mm  EXTRAPOLATES
node hulls from EVERY ALIVE launch-square ray  : (-4,-2) at -2.8687 mm  INSIDE
```

**The failure is real and the fix is free.**  The `r <= r_union` restriction is
this prototype's choice for the FIT, not the library's;
`apply_real_lens_traced` traces the whole 229 x 229 launch square anyway, so
widening each node's landing HULL out to every alive ray costs no rays at all
and turns a +1.78 mm violation into a 2.87 mm margin.  Coverage is a property of
the hull, not of the fit domain, and the two must be allowed to differ.  This is
guard **G6**.

### 5.4 Single-valuedness, from the INVERSE side

`imap_probe_121.py single`, 7 x 6 nodes, degree 10: differentiate G's own
entrance-coordinate channels analytically with respect to the EXIT coordinates
and census `det d(x_in, y_in)/d(x_out, y_out)` over each order's own footprint.

```
sign-consistent on every one of the 32 orders:              True
worst |det| dynamic range WITHIN one order:                 1.2627
worst |det_inv * det_fwd - 1| against the traced forward
  Jacobian (reciprocity):                                   1.3728e-05
```

**PASS.**  The 1.2627 reproduces the forward census's 1.2615
(`PROTO_HAMILTON_MAP_2026_08_11` S5) to three digits, which is the consistency
check that matters: the inverse is as single-valued as the forward map said it
would be, on the same domain, and the two Jacobians are reciprocal to 1.4e-05.
The entrance -> exit map is a diffeomorphism over the whole union domain and the
interpolant is interpolating one smooth branch.

---

## 6. GO / NO-GO AND THE PRODUCTION DESIGN

### 6.1 Verdicts

| question | measured | verdict |
|---|---|---|
| Is the 419 s a per-exit-pixel Newton over 6.71e+07 px? | 2 calls x 9 025 pts, 0.010 s, 0.010 % of the element | **NO -- premise refuted** |
| Does removing the Newton save anything? | 0.010 s per order | **NO-GO** |
| Exit-coordinate degree for `lambda/100` with 3x margin? | degree 10 x 42 nodes -> 2.61e-03 w (3.8x) | GO on the bar as posed |
| ...is `lambda/100` the right bar? | the shipped path delivers 1.11e-04 w | **NO -- 90x too loose; sizing to it is a 24x silent regression** |
| Degree / nodes for PARITY with the shipped path? | **9 x 8 = 72 nodes x exit degree 14 -> 3.84e-05 w** (2.9x inside) | **GO** |
| Does the C6 pad apply? | yes; costs exactly ONE node per axis | **applies** |
| Amplitude bound? | 1.58e-05 rel, and that is the `np.gradient` ESTIMATOR floor (control 3.17e-05); analytic-Chebyshev recipe 1.58e-06 | **GO -- never binding** |
| Error growth toward the fit-domain rim? | flat; outermost decile no worse than the centre at deg >= 10 | **GO** |
| Behaviour outside the landing hull? | 1.1e+04 waves (the fit's extrapolation) | **guard MANDATORY** -- and it already exists |
| Is the inverse single-valued? | sign-constant on 32 orders, range 1.2627, reciprocity 1.37e-05 | **GO** |
| Node-hull coverage under contraction? | fails at +1.78 mm with fit-cut hulls; passes at -2.87 mm with full-square hulls | **GO with G6** |
| Evaluation cost at 6.71e+07 px? | 1.910 s vs 4.035 s for the one order-3 upsample it replaces, vs 25-46 s for a per-pixel Newton | **GO** |
| Net saving? | 12.05 s per order = 10.5-12.4 % of the element, 6.4-7.6 % of an order; ~386 s over 32 orders | **GO** |
| Build / storage / break-even? | 6.00 s, 276.5 kB, break-even 0.50 orders | **GO** |

### 6.2 Where G plugs in

NOT at the Newton call.  At the `sub > 1` branch of the OPL inversion
(`_lens_traced.py` ~9985-10188) and the ray-density branch (~10196-10276), which
together are:

```
coarse Newton on X[::sub, ::sub]                        ->  opl_coarse   (9 025)
map_coordinates(order 3) + order-1 NaN pass             ->  opl_map      (6.7e7)
_ray_density_amp_grid on X[::sub, ::sub]  (a 2nd Newton)->  ard_coarse   (9 025)
map_coordinates(order 1) x 2                            ->  ard_map      (6.7e7)
map_coordinates(order 1) x 2 on (xe, ye)                ->  the entrance
                                                            pullback for
                                                            remap_sampling='full'
```

replaced by

```python
cvec = imap.at(carrier)                              # (P, 4), microseconds
x_in, y_in, opl, det_j = imap.eval(X, Y, cvec)       # ONE numba pass, 6.7e7 px
```

Everything downstream is untouched: the `_opl_ref` conditioning,
`_TracedExitSupport` (which already runs on the full grid and hands G its domain
mask for free), `_pip_sample_residual` (which CONSUMES `(x_in, y_in)` and is not
replaced), the field assembly.

API shape -- element-level, opt-in, no default change:

```python
imap = lumenairy.elements.build_inverse_map(
    prescription, wavelength, R,                  # G1: ONE radius, asserted
    src_box=((xs_lo, xs_hi), (ys_lo, ys_hi)),     # the fan's labels + C6 pad
    nodes=(9, 8),                                 # S3.4 -- parity, not lambda/100
    exit_degree=14,                               # S3.4
    launch_radius=None, ray_subsample=None,       # default to the element's own
) -> InverseCharacteristic   # 276.5 kB; .nodes .src_box .exit_box .node_resid
                             #           .det_j_range .alive .hulls

cvec = imap.at(TiltedCarrier(R, L, M, x0, y0))    # G4 + G5 checked here
```

### 6.3 The guard set

G1-G5 carry over from `PROTO_HAMILTON_MAP_2026_08_11` S5 unchanged.  G6-G8 are
NEW and are what the inverse side needs and the forward side did not.

| guard | when | refuses on | measured on design 121 |
|---|---|---|---|
| G1 one radius | build | any congruence's `R` off the map's by > tol | passes by construction (`R` closure is order-independent) |
| G2 Jacobian sign + floor | build | `det J` sign change on any node | 0 flips; inverse-side range 1.2627 |
| G3 alive census | build | any node ray dead | passes |
| G4 C6 budget | evaluate | `max\|grad a_fit\| x \|R\|` outside the built pad | pad 0.6884 mm; costs 1 node per axis |
| G5 label in-box | evaluate | label outside the source box | all 32 orders inside |
| **G6 node-hull coverage** | build | any evaluable exit pixel outside ANY node's landing hull | +1.78 mm FAIL with fit-cut hulls, -2.87 mm PASS with full-square hulls |
| **G7 exit-degree adequacy** | build | per-node least-squares residual above a stated fraction of lambda | free -- it IS the fit residual; deg 14 -> 2.69e-05 w |
| **G8 parity with the incumbent** | build | the map's budget looser than the Newton path it replaces | THE one that matters (S3.4): lambda/100 would have been a 24x regression |

### 6.4 What stays Newton

**Newton stays the default and the build-time fallback.**  There is no
per-pixel fallback to write: an exit pixel outside the landing hull has no ray,
and BOTH paths already refuse it -- Newton via
`xe^2 + ye^2 > (launch_radius * 0.99)^2` plus the `_TracedExitSupport` taper, G
via the same taper.  The fallback lives at BUILD time: if G1, G2, G3, G6, G7 or
G8 fails -- a design with an order-dependent or chromatic `R`, a fan that
outgrows the box, a group near a fold, a node whose hull does not cover the
evaluated support, a degree that cannot reach parity -- the builder refuses and
the caller keeps the shipped Newton path unchanged.  Refuse, never degrade.

---

## 7. WHAT IS NOT CLAIMED

* **No field was propagated through G.**  Every accuracy number is
  map-vs-direct-ray-trace on the characteristic itself.  The step from
  "the OPL interpolates to 3.84e-05 waves" to "the readout tile is unchanged"
  is not taken here; the prior probes' null-control pattern (arm A vs arm B on
  the same aperture field) is the way to take it.
* **The G column of the S4.6 cost table is a PROJECTION**, assembled from
  measured parts (the numba evaluation at 6.71e+07 points, the `map_coordinates`
  census, the `|E_in|` sampler's own measured cost).  No library edit was made,
  so no end-to-end G run exists.  The one term that is an ESTIMATE rather than
  a measurement is the fine-grid `|E_in|` sampler at ~1.55 s, taken from the
  measured cost of the two identically-shaped residual samplers (#7 / #8).
* **The incumbent control uses a HARD union-disc sample mask**; the shipped
  order-10 fit uses `_decentred_fit_score_weight` weights over the same 52 441
  samples.  Both are order 10 / 66 terms on the same rays.  A weighted fit
  concentrates its accuracy on the beam, so the shipped incumbent is if
  anything BETTER than the 6.65e-05 measured here -- which makes the parity bar
  tighter, not looser.
* **Only the LAST post-DOE group is mapped**, and only at `n_fine_cap = 8192`.
  At the shipped 16384 the replaceable upsample bucket grows with the pixel
  count while the build does not, so S4.6 gets stronger -- an inference, not a
  measurement.
* **The C6 pad is the forward prototype's scalar bound** (0.6884 mm, a max over
  the whole launch lattice including edges carrying negligible amplitude), so
  the padded node counts are conservative and the unpadded ones optimistic.
* **`det J` is built by `np.gradient` on the launch lattice** in this
  prototype, which S3.5 shows sets a 1.58e-05 amplitude floor that the
  production recipe does not have.  The amplitude column of S3.2 is therefore a
  lower bound on G's achievable amplitude accuracy, not a measurement of it.
* **The load caveat of S1 applies to every absolute second.**  Ratios inside
  one process carry the verdicts.

---

## 8. FILES

| file | what |
|---|---|
| `validation/repro_traced_carrier_121/imap_probe_121.py` | `degree` / `edge` / `single` / `ampctl` / `upsample` / `newton` -- every accuracy, edge and single-valuedness measurement, plus the G builder and the reproduced shipped Newton |
| `validation/repro_traced_carrier_121/imap_cost_121.py` | `newtonsite` / `profile` / `mapcoords` / `fitorder` / `evalcost` / `newtonpp` / `build` -- the cost decomposition |

Both are `__main__`-guarded and import-safe; both import
`hmap_probe_121` (the forward prototype) for the shared geometry, so that
note's `_d121_common` chain-A cache and its guards apply unchanged.  Results of
record are `_imap_*.json` (a few kB each).

Reproducing, in the order the sections appear:

```
python imap_cost_121.py newtonsite      # S4.1, ~2 min, ~21 GB
python imap_cost_121.py profile         # S4.2, ~3 min, ~21 GB
python imap_cost_121.py mapcoords       # S4.3, ~3 min, ~21 GB
python imap_cost_121.py fitorder        # S3.4, ~2 min, ~21 GB
python imap_cost_121.py evalcost        # S4.4, ~3 min, ~7 GB
python imap_cost_121.py newtonpp        # S4.4 Newton row, ~1 min
python imap_probe_121.py degree         # S3.2 + S3.4, ~75 s
python -c "import imap_probe_121 as I; I.mode_degree(pad=0.0)"   # S3.3
python imap_probe_121.py ampctl         # S3.5, ~1 s
python imap_probe_121.py upsample       # S5.1, ~4 s
python imap_probe_121.py edge           # S5.2 + S5.3, ~5 s
python imap_probe_121.py single         # S5.4, ~3 s
```

**Housekeeping:** the directory's `.gitignore` covers `_*.npz` only, so the
`_imap_*.json` artefacts are not ignored.  They total under 200 kB -- a naming
note, not a hazard -- unlike the 1.07 GB `_sumap_ap_*.npy` files an earlier
probe left, which are still present and still unignored.
