# WP-A26 changelog text (the decentred ray fit's order, under WP-A1's ray set)

Assembled by the orchestrator into `CHANGELOG.md`.  Finding IDs are from
`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A26_REPORT.md`.

### Fixed -- traced lens: the off-centre ray fit's order, re-derived against the conic ray set (A26)

`apply_real_lens_traced` launches its ray lattice over a SQUARE of half-width
`launch_radius = 0.75 * aperture_diameter` -- 1.5 clear-aperture radii on the
axes, 2.12 at the corners -- and pops `aperture_diameter` before
`surfaces_from_prescription` so that margin is traced unvignetted and no field
energy is clipped.  On a DECENTRED beam, niche D1's restriction keeps every one
of those samples in the least squares and only DOWN-WEIGHTS the out-of-disc
ones, because a hard mask there leaves the fit's remaining freedom
unconstrained and the map FOLDS.  The order that fit is given therefore has to
be enough to follow the whole launch square, not just the fit disc.

Until the 2026-09-11 audit's R4 it never had to.  `_intersect_surface` seeded
its Newton branch from the ray-SPHERE quadratic and used THAT discriminant as
the miss test, so on a conic every ray beyond `h = |R|` came back
`alive=False, error_code=RAY_MISSED_SURFACE, t=0` although it genuinely hits
the surface.  R4 replaced the seed and the miss test with the exact conic
quadratic -- correctly: measured against an inline exact conic trace (flat
entrance, exact even-conic sag, gradient normal, vector Snell, no library
code), the resurrected rays agree to **2.17e-19 m** in exit coordinate and
**6.51e-19 m** in exit OPL out to 2.12 clear-aperture radii, which is the ULP
floor of the quantities themselves; and the rays that were already alive come
back bit-identical (`max |d| = 0.0` on x, y, z, L, M, N, opd and error_code).
What changed is the fit's DATA DOMAIN.

On the `K = -n^2` Fermat singlet of `tests/unit/test_niche_d7_decentred_fit.py`
(N-BK7, f = 3 mm, 3.40 mm aperture, 0.60 mm beam, `ray_subsample=1`), the
weighted fit's finite sample set went from 111 525 rows spanning
`|h| <= 1.5106 mm` (= `|R| = (n-1) f`, the sphere the old miss test was really
testing) to 405 769 rows spanning `|h| <= 3.6062 mm`, and the exit-slope error
of the returned field against that fixture's ANALYTIC, decentre-INVARIANT conic
oracle went from 2.162 / 1.958 urad to 44.457 / 31.556 urad at 0.5 and 1.0 beam
radii of decentre.  The on-axis figure did not move by one bit -- 41.089 urad,
bit-identical field -- because the CONCENTRIC branch restricts by a hard NaN
mask and its sample set is the fit disc whatever the tracer does outside it.

`_DECENTRED_FIT_POLY_ORDER` is re-derived 10 -> 16 against the ray set the
tracer now produces.  The ladder, same fixture, same process, exit-slope rms
over the beam core at 0.5 / 1.0 beam radii of decentre:

| order | basis terms | 0.5 w | 1.0 w | vs the 41.089 urad on-axis figure |
|---|---|---|---|---|
| 10 | 66 | **44.457** | **31.556** | 1.0820 / 0.7680 |
| 12 | 91 | 10.841 | 11.829 | 0.2639 / 0.2879 |
| 14 | 120 | 3.718 | 5.419 | 0.0905 / 0.1319 |
| **16** | **153** | **2.371** | **1.683** | **0.0577 / 0.0410** |
| 18 | 190 | 1.044 | 0.655 | 0.0254 / 0.0159 |
| 24 | 325 | 0.071 | 0.057 | 0.0017 / 0.0014 |

Monotone: more terms are strictly better here, so the value is a cost/accuracy
choice and not a plateau, and the choice is made against the figure this
element returned before the ray set stopped being truncated -- 2.162 urad at
0.5 w and 1.958 at 1.0 w.  **16 is the LOWEST order on the ladder that reaches
that scale on both decentres** (2.371 urad, 1.10x of it, and 1.683 urad, 0.86x);
14 is still 1.7x and 2.8x short.  Niche D1's own
adversarial ghost geometry is untouched across the whole ladder -- 0
fold-caustic warnings, 0 sign changes of `d(x_out)/dx` over the launch square,
off-beam amplitude 1.76e-04 of peak at every order from 10 to 24.  Niche D6's
on-axis EE2 ratio against its inline Kirchhoff oracle does not move either:
`r_on = 0.969786923` and `r_off = 0.985517727` at order 10 and at 16, identical
to nine digits.

THE COST is in the Newton hot loop, which evaluates these fits per output
pixel.  Medians of 7 interleaved runs of one decentred `apply_real_lens_traced`
(N = 512, dx = 8 um, `ray_subsample=8`), box shared with other jobs:
**248.2 ms at order 10 against 386.2 ms at 16, i.e. 1.56x**.  Paid only on the
off-centre branch; the concentric path is byte-identical and unchanged in
cost.

CONDITIONING re-measured rather than inherited.  D7's sizing note records that
on design 121's last group "order 14 starts to LOSE to conditioning"; that
table predates niche C13's `LSTSQ_CONDITIONING_STEPDOWN` (shipped `True`).  The
census the C13 tests use reads IDENTICALLY at order 10 and at 14 / 16 / 18 / 20
on both fixtures in the tree: Fermat singlet Gram rcond 5.025e-15 with a
returned-fit residual ratio 1.000009 against an independent QR, D1's ghost
1.802e-14 and 1.012192 -- and identical again with the step-down forced off.
The worst solve of the call is the inverse-characteristic model's own
total-degree-14 exit fit, not this one.  Design 121's own fixture is local-only
and is not re-measured.

Unchanged: the concentric / on-axis path (byte-identical -- the raise is
engaged only on the off-centre branch, exactly as D7 shipped it), the
`decentred_fit_poly_order=<newton_poly_order>` fail-before switch,
`newton_fit='spline'` (which takes no fit-domain restriction at all), and the
"3 samples per basis term" step-down, whose arithmetic follows the new order
(153 terms -> 459 in-disc coarse samples, i.e. the full raise survives while
`fit_radius_beam_factor * w / (dx * ray_subsample) >~ 12.1`).  A caller asking
for more still gets more.

Migration note: a decentred call now builds a 153-term fit where it built a
66-term one, and a coarse ray grid silently takes the highest order its disc
can constrain -- design 121's last group clears 459 samples by 3.8x at the
default `ray_subsample=8` and takes order 15 at 16; the `_lens_traced.py`
synthetic f/6 example holds 223 and takes order 10, which is why the two
"routes to the weighted raised order path" pins in `test_niche_c11_*` and
`test_niche_c1_*` now assert the RAISE (`6 < o <= _DECENTRED_FIT_POLY_ORDER`)
rather than the constant.

### Added -- tests

* `tests/unit/test_audit2609_a26_decentred_exit_reference.py` (10 tests) -- the
  decentred exit wavefront against the analytic Fermat sphere as a two-sided
  envelope with the pre-A26 order as an in-process fail-before; the applied fit
  order; that the fit really is handed data out to the launch square's corner;
  that R4's resurrected rays hit the conic (so re-truncating them fails here
  first); the concentric path's byte identity; and the step-down.
