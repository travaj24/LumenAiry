# PROTOTYPE -- a shared-pupil Hamilton characteristic map for the design-121 traced fan

**2026-08-11.  Working tree on branch `fix/tilt-quadratic-opl` off `main` @
`755ad99` (v5.34.0), with that branch's one library edit
(`lumenairy/elements/_lens_traced.py`, the tilt-quadratic OPL piston) in place
and NOT further modified.  No file under `lumenairy/**` was touched by this
prototype; no `git`, no `gh`, no `CHANGELOG`.  Everything added is
`validation/repro_traced_carrier_121/hmap_*.py` plus this note.**

Answers the blocking item `PROBE_SUM_AT_APERTURE_2026_08_11` S9.1 named:

> *"A shared-pupil formulation of the last group's ray map (one map reused by
> every order, instead of one per congruence) is the only route that could
> reach the claimed factor."*

---

## 0. VERDICT

> **GO on the physics and the geometry.  NO-GO on the cost, by four decades.**
>
> The map is not merely feasible, it is **exact in its parametrisation**.  At
> the last post-DOE group's entrance every DOE order shares ONE carrier radius
> (`R = -21.139185 mm`, measured order-independent), and
> `_tilted_carrier_parts` depends on `(x0, y0, L, M)` only through the source's
> transverse projection `(x0 - R L/N, y0 - R M/N)`.  **The 32-order fan is
> therefore a TWO-parameter family of congruences**, labelled by the source
> position -- so a 4-D map `F(x, y; x_src, y_src)` represents every order with
> no parametrisation residual at all.
>
> Measured on that parametrisation, at the 32 orders' own labels, over the
> shipped retrace pupil, with the niche-C6 residual-eikonal widening included:
> **30 Chebyshev nodes (6 x 5) hold the OPL to 1.16e-03 waves** -- lambda/100
> with **8.6x** of margin, against a 3x bar.  Zero rays of the union domain
> clip anything (12.4 % clearance on the tightest surface).  `det J` is
> sign-constant on all 32 congruences and all node congruences, with a dynamic
> range of **1.26** inside an order -- no fold, no caustic, single-valued.  The
> amplitude (`|det J|^-1/2`) interpolates 3-4 decades more easily than the OPL.
> The whole map is **47 kB** of Chebyshev coefficients and **0.36 s** to build.
>
> And that is exactly the problem.  **The ray trace the map amortizes is
> 0.012 s of a 457.6 s retrace -- 0.003 %.**  The launch lattice is pinned at
> `n_launch = 229` (52 441 rays) by the F-C pitch-preserving `ray_subsample`
> rescale, which sets the retrace's ray pitch from the CHAIN's physical pitch
> (`4 x 33.2112 um`) and the group's aperture, **not from `n_fine`** -- so it
> does not grow with the fine grid, and it is 52 k rays whether `n_fine` is
> 2048 or 8192 (measured: 51 529 / 54 289 / 52 441).  Break-even is at
> **30 orders**; design 121 has 32; the saving is **0.14 s of a ~22 000 s
> 32-order run = 0.0006 %**.
>
> **The 76.4 % the sum-at-aperture probe attributed to "the fine re-trace" is
> not tracing.**  It is the Chebyshev fit, the Newton pull-back of every one of
> `n_fine**2 = 67.1e+06` exit pixels, and the field assembly -- all per-order,
> per-output-pixel, and untouched by any angular interpolation.
>
> **Consumer 2 is the real result.**  Evaluating `apply_real_lens` at the
> carrier's own incidence angle instead of at normal incidence is worth, on
> design 121's last group at the extreme order, **0.212 waves** of wavefront
> (piston and tilt removed) at a 3 mm pupil radius -- **21x** lambda/100 --
> plus a **14.97-wave** piston and a **25.68-wave** tilt.  That is a real
> accuracy gap in a shipped, widely-used element model, and the map closes it
> for 0.36 s and 47 kB.
>
> **Recommendation: NO-GO as a speed change for the traced fan; GO as an
> accuracy feature for `apply_real_lens`; and REDIRECT the speed work to the
> INVERSE (mixed) characteristic, which is where the 419 s actually is.**
> Design sketch in S8.

---

## 1. BOX, BUILD, CONFIGURATION

```
Windows 11 Pro 10.0.26200        AMD Ryzen 9 5950X, 24 logical CPUs
137.4 GB physical RAM            79.7 GB free at launch
python 3.14.6   numpy 2.4.4      lumenairy 5.34.0
numba gate _lens_traced._NUMBA_AVAILABLE = True
working tree = fix/tilt-quadratic-opl (one library file differs from main)
```

Ray-trace measurements (S2-S5, S7) run in seconds and are re-runnable freely.
The ONE end-to-end chain run (S6) uses the sum-at-aperture probe's own arm-A
configuration verbatim:

```
N=1024  dx0=2.0 um  RS=4  NW=1  DXO=0.2 um  TILE=1024
NFC=8192  WF=4.0  LEG='auto'  ram_budget=inf   order (0,0)
```

**A LOAD CAVEAT, STATED UP FRONT.**  The box carried other campaigns' jobs
throughout (~20 GB and a full worker pool).  The chain run measured
**686.3 s** per order against `PROBE_SUM_AT_APERTURE_2026_08_11` S7.1's
**138.3 s** mean on an idle box -- 5.0x.  Every verdict here is a RATIO taken
inside one process (trace vs retrace vs wall), and the ratio is what the
conclusion rests on.  For the avoidance of doubt: even at the LARGEST trace
fraction observed anywhere in this study (**0.146 %**, the `n_fine = 4096`
scaling point), the map's best case saves 0.055 % of an order.  The sign of
the verdict does not depend on the load.

---

## 2. WHAT THE MAP IS, AND THE PARAMETRISATION THAT MAKES IT EXACT

`apply_real_lens_traced` launches an axis-centred lattice
`xs_in = linspace(-launch_radius, launch_radius, n_launch)` along
`grad(W + a_fit)` -- ONE congruence -- traces it through the group's surfaces,
applies the exit-vertex correction, and keeps

```
P(x, y; theta_x, theta_y) -> (x_out, y_out, OPL)      [+ exit cosines]
```

for that congruence and no other.  Each of the 32 orders re-traces the whole
lattice because each has its own congruence.  The prototype's `characteristic()`
reproduces that trace step verbatim, including the exit-vertex correction, and
deliberately does NOT add the H6 carrier eikonal `W` or the niche-C6 residual
`a_fit`: both are ANALYTIC and PER-ORDER, so the shared map stores the group's
GEOMETRIC path only and each order adds its own entrance eikonal afterwards.
That factorisation is exact.

### 2.1 The fan is a two-parameter family, exactly

`_tilted_carrier_parts` evaluates

```
uu = (x - x0) + R L / N ,   vv = (y - y0) + R M / N
grad W = sign(R) * (uu, vv) / sqrt(uu^2 + vv^2 + R^2)
```

so the launch-direction field depends on the four congruence parameters ONLY
through the source's transverse projection `(x_src, y_src) = (x0 - R L/N,
y0 - R M/N)`.  At a plane where every order shares one `R` -- which design
121's last-group entrance does exactly, because the chain's `R` closure
(`_paraxial_group_r_out` + gaps) carries no order dependence -- the entire fan
is a **two-parameter family of congruences**.

Consequence, and it is the whole design: a map sampled at Chebyshev nodes in
`(x_src, y_src)` hits **every order exactly in its parametrisation**.  There is
no residual from labelling a congruence by a local angle that varies across the
pupil.  Measured, it buys 2.9x on the interpolation error at the same node
count (S3.3).

### 2.2 Three angular domains, and why only one of them is usable

| domain | what it is | half-widths | usable? |
|---|---|---|---|
| **absolute angle** `theta` | index the map by the launch direction itself | **0.363 / 0.329 rad** | **NO** -- 44-50 % of the node rays leave the group entirely (S3.2) |
| **reduced angle** `s = theta - grad S_R(x,y)` | shear out the ONE sphere every order shares; the node lattice in `theta` is then sheared across the pupil | 0.0817 / 0.0371 rad | yes, 24-30 nodes |
| **source position** `(x_src, y_src)` | the congruence's own label (S2.1) | **1.733 / 0.745 mm** (equivalently 0.082 / 0.035 rad) | yes, **exact**, 20-30 nodes |

The absolute box is 4.4x / 8.9x wider than the reduced one because it contains
the beam's own NA, which is not a fan property at all.  A tensor node grid
placed on it fires rays at up to 0.36 rad from pupil points 9.6 mm off axis;
those rays miss the group.  **The shear (or, better, the source-position
label) is not an optimisation -- it is the difference between a map that
exists and one that does not.**

---

## 3. MEASUREMENT 1 + 2 -- THE ANGULAR DOMAIN AND THE NODE COUNT

`hmap_geom_121.py` (census, ~15 s), `hmap_probe_121.py smooth` (~1 s),
`hmap_probe_121.py src` (~23 s).

### 3.1 The domain -- and the number the sum-at-aperture probe did NOT measure

At the LAST post-DOE group's ENTRANCE (group 5 of 6; `gap_before` 3.3233 mm):

```
R at the entrance          -21.139185 mm        (order-INDEPENDENT)
chief-ray positions        x_c in [-3.0162, +2.2620] mm   (0.75405 mm lattice)
                           y_c in [-1.5081, +0.7540] mm
chief-ray reach            3.3723 mm
CHIEF-RAY TILT SPREAD      93.326 mrad          L in [-36.65, +49.07] mrad
                                                M in [-24.54, +24.54] mrad
```

**The exit spread the earlier probe reported is 7.68e-04 rad; the ENTRANCE
spread is 9.33e-02 rad -- 122x larger.**  That is not a discrepancy: the last
group is what converts the fan's angular separation into the spatial separation
the back aperture shows.  It is the entrance number a shared map lives on, and
it is the reason the map needs real node counts rather than a handful.

In reduced angle the 32 orders sit on a **near-lattice of 8 columns x 4 rows**:

```
s_x ~ {-92.3, -69.8, -46.8, -23.5, 0.0, +23.5, +46.8, +69.8} mrad
s_y ~ {-46.8, -23.5, 0.0, +23.5} mrad
```

which is why the two angular axes must carry DIFFERENT node counts
(`Hx/Hy = 2.20`) and why an isotropic grid wastes ~20 % of its nodes.

### 3.2 Smoothness -- ABSOLUTE angle (the control that fails)

Every tensor node grid on the absolute box, at every node count tried, kills
44-50 % of its own node rays:

```
3x3   36/81 dead      5x5  104/225 dead      7x7  206/441 dead
4x4   72/144 dead     6x6  143/324 dead      9x9  316/729 dead
```

No interpolant can be built.  Recorded as `domain: absolute` in
`_hmap_smooth.json`.

### 3.3 Smoothness -- the two usable parametrisations

Error is the MAXIMUM over 9 pupil probes (centre, 4 at half the union radius,
4 at the union radius) x the 32 orders' OWN labels, against a direct trace.
Bar: `lambda/100 = 0.01` waves; the 3x-margin bar is `3.33e-03` waves.

| nodes | reduced angle, unpadded | **source position, unpadded** | **source position, C6-PADDED** |
|---|---|---|---|
| 3 x 3 = 9  | 1.47e+00 | 4.28e-01 | 1.26e+00 |
| 4 x 4 = 16 | 5.94e-02 | 4.99e-02 | 2.14e-01 |
| 5 x 4 = 20 | 4.77e-03 | **2.23e-03** | 2.92e-02 |
| 6 x 4 = 24 | **2.87e-03** | **9.91e-04** | 2.35e-02 |
| 5 x 5 = 25 | 3.53e-03 | **1.45e-03** | 7.81e-03 |
| 6 x 5 = 30 | **4.03e-04** | **1.07e-04** | **1.16e-03** |
| 7 x 5 = 35 | **9.05e-05** | **2.28e-05** | **4.86e-04** |
| 6 x 6 = 36 | **3.51e-04** | **9.27e-05** | **8.34e-04** |
| 7 x 6 = 42 | -- | **4.09e-06** | **7.96e-05** |
| 7 x 7 = 49 | **3.72e-05** | **4.11e-06** | **5.45e-05** |
| 8 x 7 = 56 | -- | **6.22e-07** | **7.86e-06** |
| 9 x 9 = 81 | **7.28e-07** | **2.88e-08** | **6.03e-07** |

(waves; **bold** = clears lambda/100 with 3x margin.  Exit-POSITION error
tracks it: 3.4 nm at 20 nodes, 0.12 nm at 30, unpadded.)

Two structural readings:

* **`ny = 4` saturates.**  `6x4`, `7x4` and `8x4` all read 2.87-2.99e-03 waves
  in the reduced-angle arm -- adding x nodes buys nothing once y binds.  Node
  budgets have to be spent on the SHORT axis first, which is the opposite of
  the intuition that the wide axis is the hard one.
* **The source-position label is worth ~2.9x** at the same node count
  (`6x4`: 9.91e-04 vs 2.87e-03), because it removes the pupil-dependence of the
  congruence label entirely.

### 3.4 The niche-C6 widening -- why the padded column is the production one

`apply_real_lens_traced` launches along `grad(W + a_fit)`, not `grad W`.
`a_fit` is fitted to THIS order's own field, so it is order-dependent and
**cannot be absorbed into a shared congruence label**: it is an extra angular
offset the shared domain must cover.  Measured on order (-4,-2), all six
post-DOE groups (`hmap_cost_121.py afit`, 320 s):

```
group      0          1          2          3          4          5
|grad a|   1.57e-04   1.13e-04   1.12e-04   1.23e-02   1.21e-02   3.257e-02  rad
```

The last group's `3.257e-02` reproduces
`PROBE_CHAIN_LADDER_PISTON_2026_08_11` S3.4's own table to four digits.  As a
source displacement that is `|grad a| x |R|` = **0.6884 mm**, which grows the
box from 1.733 / 0.745 mm to **2.422 / 1.433 mm** (1.40x / 1.92x) -- and the
node count with it.

> **NODE COUNT OF RECORD: 30 nodes, `nx = 6` x `ny = 5`,** on the
> source-position label over the C6-padded box.  `1.16e-03` waves = lambda/100
> with **8.6x** of margin.  25 nodes (`5x5`) is inside lambda/100 but only
> 1.3x, i.e. it fails the 3x bar; 20 nodes fails outright once C6 is carried.

### 3.5 The `theta^2` coefficient specifically

The piston-fix territory (`FIX_TILT_QUADRATIC_OPL_2026_08_11`): fit
`c0 + c1 s + c2 s^2` to the OPL along the reduced-x axis at the pupil centre.

```
direct trace   c2 = +6.529944384e-03 m/rad^2   (= 33.27 waves across the box)
```

| nodes | `c2` relative error | as waves across the box |
|---|---|---|
| 3 x 3 | 4.16e-05 | 1.39e-03 |
| 5 x 4 | 2.35e-06 | 7.80e-05 |
| 6 x 5 | 1.77e-06 | 5.89e-05 |
| 7 x 5 | 1.70e-08 | 5.65e-07 |
| 9 x 9 | 3.44e-11 | 1.15e-09 |

**The tilt-quadratic term is the EASY part.**  Even a 3x3 grid -- which is 147
waves wrong on the OPL itself -- carries `c2` to 1.4e-03 waves.  The node count
is set by the higher-order angular structure (the field-angle dependence of the
group's aberration), not by the quadratic the piston fix repaired.  Anything
that samples the box at all gets the `theta^2` law right.

---

## 4. MEASUREMENT 3 -- VIGNETTING

`hmap_probe_121.py vig`.  180 325 rays: 7 213 pupil points on the union disc
`r <= 9.6490 mm` x 25 angular nodes spanning the full reduced box.

```
dead rays (the tracer's own semi_diameter clip)          0   (0.0000 %)
group aperture_diameter        20.3966 mm  ->  semi-aperture 10.1983 mm
  surface 0 (S1)  max hit radius 8.9321 mm   +1.2662 mm clear  (+12.42 %)
  surface 1 (S2)  max hit radius 6.7151 mm   +3.4832 mm clear  (+34.15 %)
  surface 2 (S3)  max hit radius 4.6906 mm   +5.5077 mm clear  (+54.01 %)
exit landing radius max                                  4.8930 mm
union PUPIL radius 9.6490 mm vs semi-aperture 10.1983 mm  ->  +5.39 %
```

**PASS.**  The tightest margin is the union pupil against the declared
semi-aperture, **+5.39 %**; the tightest ray-vs-glass margin is **+12.42 %** at
the first surface.

Three clips exist and they are not the same thing; the census reports all
three because conflating them is how a "no vignetting" claim goes wrong:

1. the per-surface `semi_diameter` the TRACER enforces -- infinite on all three
   surfaces of this group, so it can never kill a ray;
2. the prescription's `aperture_diameter`, which `apply_real_lens_traced`
   **pops** before building surfaces (so it never kills a ray either) but
   **reads** to size the launch disc -- the physically meaningful bound;
3. the launch disc itself, `0.75 x aperture_diameter` = **15.2974 mm**,
   axis-centred and ORDER-INDEPENDENT.

Item 3 is a finding in its own right for the cost table: because
`aperture = lens_prescription.get('aperture_diameter')` is set, the
grid-and-origin-derived launch radius branch is dead here, so **every order's
retrace already launches over the same 15.2974 mm axis-centred disc**.  The
shared map needs 9.6490 mm -- **36.9 % inside it**.  A shared-pupil map costs
NO pupil the shipped per-order path does not already pay for.

---

## 5. MEASUREMENT 5 -- CAUSTIC GUARD / SINGLE-VALUEDNESS

`hmap_probe_121.py caustic`.  `det J = det d(x_out, y_out)/d(x_in, y_in)`
evaluated ALONG each congruence (the angle moves with the pupil point, exactly
as the shipped launch does), central difference `h = 5 um`, 3 209 pupil points
per order on that order's own support.

```
sign-consistent on every one of the 32 orders:            True
worst |det J| dynamic range WITHIN one order:             1.2615
|det J| range across all orders:            1.4239e-01 .. 1.7963e-01
node congruences, 3x3 sheared:   16 137 samples,  0 sign flips
node congruences, 5x5 sheared:   44 825 samples,  0 sign flips
```

**PASS, with room that is not marginal.**  `det J` never approaches zero: its
global minimum over all 32 orders and 102 688 samples is 0.1424 against a
median of 0.1498, so the quantity barely varies at all, let alone approaches
the zero a fold would need.  It never changes sign, and its full variation
inside one order is 26 %.  The last group is far from any caustic on this
design, the entrance->exit map is a diffeomorphism over the whole union domain,
and the interpolant is interpolating a single smooth branch.

**The guard the production version needs is nevertheless not optional**, and it
is not the census above -- a census proves the current design, not the next
one.  The C6 lesson (`FIX_C13_BUILD_SPREAD_2026_08_06`: a degenerate launch is
silent, and the failure is a plausible-looking wrong answer) says the guard has
to be a runtime precondition:

* **G1 -- one radius.**  Refuse to build the map unless every congruence's `R`
  at the map plane agrees to a stated tolerance.  Without that the family is
  not two-parameter and the source-position label is not a label.  Design 121
  passes by construction (the chain's `R` closure is order-independent); a
  design with a chromatic or per-order `R` does not.
* **G2 -- Jacobian sign and floor.**  Compute `det J` at every node
  congruence on the node pupil lattice (it is already traced -- this is a
  reduction, not a new trace), refuse on any sign change, and warn below a
  stated `|det J|` floor relative to the median.  Measured cost: zero extra
  rays.
* **G3 -- alive census.**  Refuse if any node ray dies.  This is what catches
  the absolute-angle parametrisation (S3.2) and any domain that has grown past
  the aperture; it is one boolean over the node trace.
* **G4 -- the C6 budget.**  Refuse (or re-pad and rebuild) if any consumer's
  `max |grad a_fit| x |R|` exceeds the pad the box was built with.  This is the
  one guard whose input is per-ORDER and only known at evaluation time, so it
  has to be checked at evaluation, not at build.
* **G5 -- in-box.**  Refuse an evaluation whose label falls outside the built
  box.  Chebyshev extrapolation past the endpoints diverges like the degree,
  and at `nx = 6` that is fast.

---

## 6. MEASUREMENT 4 -- COST, AND WHY THE PREMISE FAILS

`hmap_cost_121.py run` -- one order, end to end, with pass-through timing
spies on `lumenairy.raytrace.trace`,
`lumenairy.elements.apply_real_lens_traced`,
`carrier.._fine_trace_group_exit` and `carrier..carrier_referenced_exact_focus_readout`.

### 6.1 The measurement that decides it

```
order (0,0)   wall 686.3 s   retrace 457.6 s (66.7 %)   readout 49.1 s
retrace grid   n_fine = 8192  at  dx_fine = 1.5325 um ;  chain cur_dx = 33.2112 um
the ELEMENT inside the retrace          419.2 s
   of which trace()                       0.0118 s   over 52 441 rays
   n_launch = 229 ,  rs_fine = 87
peak working set                         22.8 GB
```

```
trace share of the RETRACE     0.003 %
trace share of the ORDER       0.002 %
```

Every `trace()` call in the whole chain, for scale:

```
    47 961 rays x 2 surfaces    0.008 s
    50 625 rays x 2 surfaces    0.008 s
    53 361 rays x 3 surfaces    0.088 s
    70 225 rays x 2 surfaces    0.053 s
    70 225 rays x 2 surfaces    0.049 s
    52 441 rays x 3 surfaces    0.012 s      <-- the fine retrace
```

The whole chain traces 345 k rays in 0.22 s.  **The ray tracing is not the
cost of the traced path.  It never was.**

### 6.2 Why `n_launch` does not grow with the fine grid

This is the structural point, and it is a consequence of an existing, correct
design decision (audit `AUDIT_TRACED_CHAIN_DX_SCALING_2026_07_22` F-C):

```
rs_fine   = round(ray_subsample * cur_dx / dx_fine)  = round(4 * 33.2112 / 1.5325) = 87
n_launch  = 2 * launch_radius / (dx_fine * rs_fine)
          = 2 * 15.2974 mm / (1.5325 um * 87)
          = 30.5948 mm / 133.3 um  =  229
```

`dx_fine * rs_fine` IS the chain's physical ray pitch (`4 x 33.2112 um =
132.8 um`), by construction.  So the launch lattice is a function of the
CHAIN's ray pitch and the GROUP's aperture -- and of nothing else.  Confirmed
by driving `apply_real_lens_traced` directly at three output grids
(`hmap_cost_121.py scaling`):

| `n_fine` | `dx` (um) | element (s) | trace (s) | rays | trace share |
|---|---|---|---|---|---|
| 1024 | 12.2592 | 73.12 (JIT warm-up) | 0.018 | 51 529 | 0.0247 % |
| 2048 | 6.1296  | 24.77 | 0.015 | 51 529 | 0.0617 % |
| 4096 | 3.0648  | 70.47 | 0.103 | 54 289 | 0.1459 % |
| 8192 | 1.5325  | 419.17 | 0.012 | 52 441 | 0.0028 % |

(The 1024 row carries the numba JIT compile and the 4096/8192 rows were taken
under different concurrent load, so the element column is not a clean scaling
law -- `AUDIT_TRACED_SPEED_2026_08_09`'s own variance caveat applies.  The RAY
COUNT column is the point, and it is flat: **51.5 k -- 54.3 k rays at every
output grid from 1024 to 8192**.)

### 6.3 The cost table

Per order, at the configuration of record.  "Map" = 30 nodes on the shared
pupil, amortised over 32 orders.

| stage | shipped, per order | with the map | shareable? |
|---|---|---|---|
| coarse chain (6 groups) | 179.6 s | unchanged | no -- per-congruence by construction |
| **retrace: `trace()`** | **0.0118 s** (52 441 rays) | **0.0111 s** (30/32 x) | **YES -- this is the whole prize** |
| retrace: carrier + `a_fit` fit | inside 419.2 s | unchanged | no -- fitted to THIS order's field |
| retrace: Chebyshev fit of the map | inside 419.2 s | tensor contraction, ~5.9e+03 mults | yes, and it is already microseconds |
| **retrace: Newton pull-back, 8192^2 = 6.71e+07 px** | **dominates the 419.2 s** | **unchanged** | **NO -- per exit pixel, per order** |
| retrace: amplitude + assembly, 6.71e+07 px | inside 419.2 s | unchanged | no |
| exact focus readout | 49.1 s | unchanged | no (19 % is an irreducible per-frame Bluestein -- prior probe S8.3) |
| **TOTAL** | **686.3 s** | **686.3 s - 0.0004 s** | |

Build cost and break-even, from the measured 0.0118 s per lattice trace:

| nodes | map build | 32 direct traces | saved | as % of a 32-order run |
|---|---|---|---|---|
| 20 | 0.237 s | 0.379 s | +0.142 s | +0.0006 % |
| 24 | 0.284 s | 0.379 s | +0.095 s | +0.0004 % |
| 25 | 0.296 s | 0.379 s | +0.083 s | +0.0004 % |
| **30 (of record)** | **0.355 s** | **0.379 s** | **+0.024 s** | **+0.0001 %** |
| 36 | 0.426 s | 0.379 s | -0.047 s | -0.0002 % |
| 49 | 0.580 s | 0.379 s | -0.201 s | -0.0009 % |

**Break-even order count = the node count = 30.**  Design 121 runs 32 orders,
so the map is 6 % cheaper on the trace and 0.0001 % cheaper on the run.  A fan
of 24 orders would be SLOWER with the map than without it -- and it would still
be slower by 0.0002 %, which is the real point: **the quantity is irrelevant in
both directions.**

### 6.4 Storage

| form | size |
|---|---|
| raw node samples: 229^2 pupil x 30 nodes x 4 outputs x 8 B | **50.3 MB** |
| Chebyshev coefficients at the shipped pupil degree (`newton_poly_order = 6` -> 7 x 7) x 6 x 5 x 4 outputs x 8 B | **47.0 kB** |

Storage is a non-issue in either form.  The coefficient form is also what makes
the per-order evaluation trivial: contracting the 4-D tensor at the order's
`(x_src, y_src)` costs `7*7*6*5*4 = 5 880` multiply-adds and yields exactly the
2-D pupil Chebyshev the shipped path fits -- so the map drops into the existing
Newton pull-back with no interface change at all.  It also does not make that
pull-back any faster.

### 6.5 Amplitude / Jacobian interpolation accuracy

`hmap_probe_121.py amp`: interpolate the map, then differentiate the
INTERPOLANT along each order's congruence on a 4-point pupil stencil
(`h = 20 um`), and score the ray-tube amplitude `|det J|^-1/2` against the same
quantity from direct traces.  Square reduced-angle box (conservative in y).

| nodes/axis | amplitude relative error, max | rms |
|---|---|---|
| 3 | 4.98e-04 | 1.67e-04 |
| 4 | 4.83e-05 | 1.84e-05 |
| 5 | 1.65e-06 | 4.39e-07 |
| 6 | 1.59e-07 | 5.52e-08 |
| 7 | 5.08e-08 | 1.37e-08 |

**The amplitude is 3-4 decades easier than the OPL and is never the binding
constraint.**  At the node count the OPL demands (30), the ray-tube amplitude
is at ~1e-07 relative.  That is the expected ordering -- the OPL carries the
`k0` multiplier and the amplitude does not -- but it is worth having measured,
because `amplitude_model='ray_density'` consumes exactly this Jacobian and a
map that was accurate in phase and wrong in amplitude would be a silent-wrong
result.

---

## 7. MEASUREMENT 6 -- CONSUMER 2: THE ANGLE-BLIND ELEMENT MODEL

`hmap_probe_121.py consumer2` (~10 s).

`apply_real_lens` builds its element phase as a sum of per-surface sag screens,
`phi += -k0 (n2 - n1) sag(x, y)` (`_geometric_lens_phase`), with ASM steps
between surfaces.  The screen is a function of `(x, y)` **alone**: it carries
no incidence angle.  An angle-aware version would evaluate the map at the
carrier's own local incidence angle instead.

Measured as the change in the group's own traced OPL map when the launch
congruence is tilted to that order's actual chief-ray direction at that group,
decomposed by least squares over a disc into PISTON / TILT / RESIDUAL, and
again with a defocus term.  The free-space obliquity through a homogeneous gap
is exactly a piston at fixed entrance point, so the RESIDUAL column is the
refraction-geometry error that no angle-blind screen can represent.

Worst order per group (all 32 scanned), waves:

| group | prescription | worst order | tilt (mrad) | pupil r | piston | tilt | **RESIDUAL** | resid + defocus |
|---|---|---|---|---|---|---|---|---|
| 0 | plate, N-SF1 25.40 mm | (-3,-2) | 41.52 | 3 mm | 9.910 | 0.000 | **0.0000** | 0.0000 |
| 1 | plate, N-BK7 3.20 mm | (-3,0) | 34.55 | 3 mm | 0.970 | 0.000 | **0.0000** | 0.0000 |
| 2 | doublet PK52A/SF57 | (-4,-2) | 51.50 | 3 mm | 8.651 | 3.484 | **0.0041** | 0.0040 |
| 3 | singlet LAK8 | (-4,-2) | 46.69 | 3 mm | 4.275 | 5.932 | **0.0070** | 0.0037 |
| 4 | singlet LAK9 | (-4,-2) | 7.38 | 3 mm | 0.083 | 1.156 | **0.0014** | 0.0014 |
| **5** | **doublet SK2/SF57, R = 19.6 / -27.4 / 12.65 mm** | **(-4,-2)** | **54.87** | **3 mm** | **14.713** | **25.682** | **0.2118** | **0.1221** |

and the last group against pupil radius:

| pupil radius | piston | tilt | **RESIDUAL** | resid + defocus |
|---|---|---|---|---|
| 1 mm | 14.971 | 8.665 | **0.0200** | 0.0102 |
| 2 mm | 14.879 | 17.258 | **0.0849** | 0.0454 |
| 3 mm | 14.713 | 25.682 | **0.2118** | 0.1221 |

**Sizing consumer 2.**  On the last group at the extreme order and a 3 mm pupil
(the entrance beam radius there is ~3.14 mm, so this is ~1 beam radius), the
angle-blind screen is wrong by **0.212 waves rms** after piston and tilt are
removed -- **21x** lambda/100 -- and still **0.122 waves** if a defocus is
allowed to absorb what it can.  It grows as the square of the pupil radius
(0.020 / 0.085 / 0.212 at 1 / 2 / 3 mm), which is the field-angle-dependent
astigmatism-and-coma signature, and it is dominated by ONE group: the other
four contribute 0.0125 waves between them.

Two collateral readings worth recording:

* **Groups 0 and 1 return EXACTLY zero residual** -- they are plane-parallel
  plates (both radii infinite), for which a tilted ray's OPL change at fixed
  entrance point is a pure constant.  Their **9.910- and 0.970-wave pistons**
  are precisely the tilt-quadratic path the branch's own
  `FIX_TILT_QUADRATIC_OPL_2026_08_11` restores; the traced element carries it,
  the angle-blind screen does not.  A plate is the cleanest possible fixture
  for that fix and it is already in this prescription.
* The plates also shift the ray transversely, which an angle-blind screen
  misses entirely; that is NOT in the residual column above (the comparison is
  at fixed entrance point) and is a separate, larger, unmeasured term.

---

## 8. GO / NO-GO AND THE PRODUCTION DESIGN

### 8.1 Verdicts

| question | measured | verdict |
|---|---|---|
| Is the map's angular domain bounded and small? | reduced 0.082 / 0.037 rad; absolute 0.363 / 0.329 rad, unusable | **GO** with the shear / source-position label; **NO-GO** on absolute angle |
| Node count for lambda/100 with 3x margin? | **30 (6 x 5)** C6-padded, 1.16e-03 waves (8.6x) | **GO** |
| Is the `theta^2` (piston-fix) term reproduced? | 1.77e-06 relative at 30 nodes = 5.9e-05 waves | **GO** |
| Does anything vignette? | 0 dead rays; +5.39 % pupil, +12.42 % glass | **GO** |
| Is the characteristic single-valued? | `det J` sign-constant, range 1.26 within an order, 0 flips in 61 k node samples | **GO** |
| Does the amplitude interpolate? | 1.6e-07 relative at the OPL's node count | **GO** |
| Does it make the traced fan faster? | trace = 0.003 % of the retrace; break-even 30 orders; saving 0.0001 % | **NO-GO** |
| Does an angle-aware `apply_real_lens` gain accuracy? | 0.212 waves (21x lambda/100) on group 5 | **GO** |

### 8.2 Why the cost premise failed, stated once

`PROBE_SUM_AT_APERTURE_2026_08_11` S7.1 measured "the fine RE-TRACE" at 76.4 %
of an order and inferred that a shared ray map would attack it.  The inference
does not hold, and the reason is a naming collision: `_fine_trace_group_exit`
is called a re-TRACE, but tracing is 0.003 % of it.  The 76.4 % is the Fourier
upsample, the carrier reconstruction, the residual-eikonal fit, the Chebyshev
fit, **the Newton pull-back of `n_fine**2` exit pixels**, the amplitude and the
assembly -- and of those, only the Chebyshev fit is a function of the ray map
at all, and it is already microseconds.

The design decision that pins this is F-C's pitch-preserving `ray_subsample`
rescale, and it is CORRECT: it keeps the retrace's ray lattice at the chain's
own physical ray pitch instead of densifying it by `cur_dx / dx_fine` (which
the pre-F-C code did, and which reached 84.7 GiB on a production chain).  A
consequence of that correctness is that there is no ray-trace cost left to
amortize.

### 8.3 The redirect -- the map that WOULD attack the 419 s

The same two-parameter congruence label, applied to the **inverse (mixed)
characteristic** instead of the point characteristic:

```
G(x_out, y_out ; x_src, y_src)  ->  (x_in, y_in, OPL, |det J|)
```

Per order, contract at that order's `(x_src, y_src)` -- 5 880 multiply-adds --
and the result is a 2-D Chebyshev in EXIT coordinates that returns the entrance
point directly.  That replaces a 12-iteration 2-D Newton (each iteration
evaluating a degree-6 Chebyshev and its gradient, i.e. ~24 polynomial
evaluations per pixel) with **one** polynomial evaluation per pixel, over
`6.71e+07` pixels.  Everything this prototype measured supports it:

* the congruence family is the same two-parameter family (S2.1);
* the forward map is a diffeomorphism over the whole union domain, `det J`
  sign-constant with a 1.26 dynamic range (S5), so the inverse exists, is
  single-valued and is as smooth as the forward map;
* the union EXIT support is 4.8930 mm (S4), comfortably bounded;
* the node count would be set by the same angular structure.

**It is UNMEASURED here** -- in particular the Chebyshev degree needed in the
EXIT coordinates (the shipped path fits degree 6 in ENTRANCE coordinates and
inverts; a direct exit-coordinate fit is a different function and may need a
higher degree, and the Newton's per-pixel convergence tolerance
`5.1e-07 m` is a bar the fit would have to clear outright rather than iterate
to).  The 2.1-26.8 % Newton non-convergence the COARSE legs logged at 12 iterations
(the retrace's own inversion logged none) is a reason to look, not a reason to
assume.  **That is the experiment worth
running next, and this prototype's machinery is most of it.**

### 8.4 Production design sketch -- if consumer 2 is pursued

API shape (element-level, opt-in, no default change):

```python
# build once, per (group, wavelength, R, box)
hmap = lumenairy.elements.build_congruence_map(
    prescription, wavelength, R,                 # G1: ONE radius, asserted
    src_box=((xs_lo, xs_hi), (ys_lo, ys_hi)),    # the fan's labels + C6 pad
    nodes=(6, 5),                                # S3.3; 30 nodes
    launch_radius=None,                          # defaults to the element's own
    ray_subsample=None,                          # defaults to the element's own
)  -> CongruenceMap        # 47 kB; .nodes, .box, .det_j_range, .alive

# evaluate, per congruence
screen = hmap.at(TiltedCarrier(R, L, M, x0, y0))   # G4 + G5 checked here
```

Where it plugs in:

* **`apply_real_lens_traced`** -- `hmap.at(carrier)` returns exactly the
  `(x_out_grid, y_out_grid, opl_grid)` triple the function currently builds at
  line ~8755 from `final.x / final.y / final.opd`, on the same
  `xs_in` lattice.  Substitution point is one block; everything downstream
  (the `_opl_ref` conditioning, `_TracedExitSupport`, the fit-domain
  restriction, the Newton pull-back) is untouched.  **This buys nothing on
  speed** (S6) and is only worth doing as the shared foundation for the two
  below.
* **`_fine_trace_group_exit`** -- pass the map through `call_kw`; the element
  consumes it as above.  Same conclusion.
* **`apply_real_lens`** (the actual win) -- it currently takes `conjugate=`, a
  SCALAR on-axis conjugate distance that can express neither a tilt nor a
  decentre; give it a congruence label (a `TiltedCarrier`, the same object the
  traced element already consumes) and, when given, replace
  `_geometric_lens_phase`'s angle-blind sag sum with the map evaluated at that
  label.  Worth 0.212 waves on design 121's last group
  (S7).  Note `_geometric_lens_phase` is documented NOT origin-aware
  (niche D9); the map is origin-aware by construction, so this closes that
  restriction rather than inheriting it.

The guard set is G1-G5 of S5, all cheap, all runtime, and all refusing rather
than degrading:

| guard | when | refuses on |
|---|---|---|
| G1 one radius | build | any congruence's `R` off the map's by > tol |
| G2 Jacobian | build | `det J` sign change on any node; warn below a floor |
| G3 alive | build | any node ray dead |
| G4 C6 budget | evaluate | `max|grad a_fit| * |R|` outside the built pad |
| G5 in-box | evaluate | label outside the built box |

---

## 9. WHAT IS NOT CLAIMED

* **The cost run is ONE order on a LOADED box** (686.3 s against the prior
  probe's 138.3 s idle).  Ratios inside one process carry the verdict; absolute
  seconds do not.  The `n_fine` scaling table (S6.2) is likewise
  load-contaminated in its element column, and only its flat RAY-COUNT column
  is used.
* **`n_fine_cap = 8192`**, per the affordability convention of the prior probe.
  At the shipped 16384 the Newton/assembly share grows with the pixel count
  while `n_launch` does not, so S6's conclusion gets stronger -- an inference,
  not a measurement.
* **The C6 pad is a scalar bound**, `max|grad a_fit| x |R| = 0.6884 mm`, taken
  from the coarse-leg fit of order (-4,-2) (`_fine_trace_group_exit`'s own
  `a_fit` is fitted on the finer grid and was not separately captured).  It is
  a max over the whole launch lattice, including edges carrying negligible
  amplitude, so the padded node count (30) is conservative and the unpadded one
  (20) is optimistic.  An amplitude-weighted bound would land between them.
* **Only the LAST post-DOE group is mapped.**  The `R`-sharing argument of
  S2.1 is checked at that plane only; earlier groups have their own `R` and
  their own fan geometry, and consumer 2's per-group numbers (S7) are traced
  directly rather than through a map.
* **The inverse-characteristic redirect (S8.3) is unmeasured.**  No Chebyshev
  degree, no error, no timing.
* **No field was propagated through a map.**  Every accuracy number here is
  map-vs-direct-trace on the characteristic itself.  The step from
  "the OPL interpolates to 1.16e-03 waves" to "the readout tile is unchanged"
  is one this prototype does not take; the prior probe's null-control pattern
  (arm A vs arm B on the same aperture field) is the way to take it.
* **Consumer 2's residual is measured at fixed ENTRANCE point** and therefore
  excludes the transverse ray walk-off, which an angle-blind screen also
  misses and which is a separate, larger term.

---

## 10. FILES

| file | what |
|---|---|
| `validation/repro_traced_carrier_121/hmap_geom_121.py` | the angular-domain census: 32 orders at the last group's entrance, the three domains, the union pupil |
| `validation/repro_traced_carrier_121/hmap_probe_121.py` | `smooth` / `src` / `vig` / `caustic` / `amp` / `consumer2` / `trace` -- every accuracy and geometry measurement |
| `validation/repro_traced_carrier_121/hmap_cost_121.py` | `run` (instrumented chain) / `afit` / `scaling` / `model` -- the cost decomposition |

All three are `__main__`-guarded and import-safe.  Results of record are
`_hmap_*.json` (a few kB each) plus `_hmap_cost_run.log` and `_hmap_afit.log`.

Reproducing, in the order the sections appear:

```
python hmap_geom_121.py                       # S3.1, ~15 s
python hmap_probe_121.py all                  # S3.2/3.3/3.5, S4, S5, S6.5, S7; ~50 s
python -c "import hmap_probe_121 as P; P.mode_src(pad=0.6884e-3)"   # S3.3 padded column
HM_M=-4 HM_N=-2 python hmap_cost_121.py afit  # S3.4, ~320 s
python hmap_cost_121.py run                   # S6.1, ~11 min loaded, ~23 GB
python hmap_cost_121.py scaling               # S6.2, ~3 min
```

Every measurement except `run` / `afit` / `scaling` is pure ray tracing and
reproduced bit-for-bit across separate processes during this study (the
`trace` throughput mode is the sole exception and is load-bound by design).

**Housekeeping:** the directory's `.gitignore` covers `_*.npz` only, so the
`_hmap_*.json` / `_hmap_*.log` artefacts of this prototype are NOT ignored.
They total under 100 kB, so this is a naming note rather than a hazard --
unlike the 1.07 GB `_sumap_ap_*.npy` files the prior probe left, which are
still present and still unignored.
