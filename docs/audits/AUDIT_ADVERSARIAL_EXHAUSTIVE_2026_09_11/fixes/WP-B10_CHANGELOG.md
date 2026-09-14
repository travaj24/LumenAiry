# WP-B10 changelog text (a disc-orthogonal basis for the traced ray fits)

Assembled by the orchestrator into `CHANGELOG.md` for 5.47.0.  Finding IDs are from
`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-B10_REPORT.md`.

### Added -- traced lens: `fit_basis='zernike'`, a disc-orthogonal design basis for the ray fits (audit sec. 15.9, B10)

`apply_real_lens_traced` and `prepare_real_lens_traced` take a new
`fit_basis` keyword -- `'chebyshev'` (default, byte-identical to every prior
release) or `'zernike'`.  On the opt-in basis the entrance-plane forward and
OPL fits the Newton inversion is handed are expressed in the Zernike
polynomials, orthonormal on the RAY-FIT DISC (beam-centred on the decentred
branch, concentric otherwise), at the same total degree.

**What it changes is the conditioning of the solve, and nothing else.**  The
Zernike set of total degree `<= order` and the tensor-Chebyshev total-degree
set are two bases of the SAME space -- `(order+1)(order+2)/2` terms either way,
verified by rank and by projection (`_lens_traced.py:3756`) -- so the same
samples with the same weights minimise the same residual over the same space
and return the same polynomial.  Measured end to end on niche D7's `K = -n^2`
Fermat singlet against its analytic decentre-invariant oracle, the decentred
exit-slope error is the SAME NUMBER on both bases at every order from 6 to 20
and at both decentres:

| order | terms | 0.5 w, cheb / zern | 1.0 w, cheb / zern |
|---|---|---|---|
| 6 | 28 | 233.859 / 233.859 | 354.413 / 354.413 |
| 10 | 66 | 44.457 / 44.457 | 31.556 / 31.556 |
| 14 | 120 | 3.718 / 3.718 | 5.419 / 5.419 |
| **16 (shipped)** | 153 | **2.371 / 2.371** | **1.683 / 1.683** |
| 20 | 231 | 0.321 / 0.321 | 0.301 / 0.301 |

(urad of exit-slope rms over the beam core; the full eight-order ladder is in
the report, and its Chebyshev column reproduces WP-A26's own digit for digit)

and the niche-C11 arbiter's two candidate residuals agree across the bases to
all seven printed digits, so its verdict cannot move either.  This is niche
D7's affine-invariance refusal generalised: least squares depends on the SPAN,
not on the basis.

**Where it helps.**  Conditioning is a joint statement about the basis and the
sample measure, so a disc-orthogonal basis pays where the retained samples ARE
the disc -- which is the CONCENTRIC branch, whose fit-domain restriction is a
hard NaN mask.  On the on-axis call of the same fixture the equilibrated Gram
rcond of the applied fits goes **1.415e-11 -> 9.785e-01** and niche C13's
conditioning step-down, which fires on **3 of 3** solves there today, fires on
**0 of 3**; the returned field's exit slope is 41.089 urad either way.  The
decentred arbiter's concentric trial fit -- also a hard mask -- goes
**3.108e-08 -> 9.863e-01** in the same call.

**Where it does not.**  On the DECENTRED branch niche D1's weighted skirt keeps
every launch sample in the least squares, out to 4.01 fit-disc radii on this
fixture, and a disc-normalised column grows as `(r/R)^n` out there.  The
advantage is still large at low degree (**1.70e-10 -> 1.31e-01** at order 6)
and decays by about 1.2 decades per degree (a disc-normalised column gains one
power of the skirt's reach per shell at low order; the measured rate saturates
near 1.19 decades per degree from a data-to-disc ratio of about 3.4 upward and
does NOT scale with that ratio, which moves the low-order offset instead --
VERIFY-B10 section 8) until it crosses the square basis's roughly flat `~1e-11`
near the shipped order 16.  The ladder for both
fixtures is in the report; this is why the basis is opt-in and why the default
does not move.

The section 15.9 line this answers is that "the entire fit-radius / arbiter /
predictor apparatus exists because a square Chebyshev basis couples marginal
rays into defocus on a disc".  Measured, it does not.  Marginal rays DO couple
into defocus -- with D1's skirt in force, the fitted map's `Z(2,0)` on the fit
disc shifts by 2.083e-10 m of exit coordinate at a total degree of 6 and by
1.934e-13 m at the shipped 16, a factor of 1077 -- but the shift is the same
number in both bases (to 1.3e-10 of it at order 6), and what removes it is the
ORDER (WP-A26's 10 -> 16), not the basis.

**Unchanged, and proved so.**  `fit_basis='chebyshev'` is the default and is
byte-identical: fifteen configurations of the element -- concentric, decentred
at one beam radius, the pre-D7 order, an order-20 caller, three of those again
on the forward path where the fit really does reach the returned field, three
at `ray_subsample=1`, `newton_fit='spline'`, `inverse_map=False`,
`inversion_method='fit'`, D1's ray-density ghost geometry and a prepared
screen -- return fields whose md5s match a `git archive` extraction of the tree
without this change, driven in child processes with `lumenairy.__file__`
asserted (report section 7).  Those fifteen cases carry **12 distinct fields**,
so the battery can see a change: three of them differ only in the forward fit's
order.  The default path passes no
new keyword to the fit at all, which is also what keeps the fixed-signature
fit spies in `test_niche_c1_consolidation` and `test_niche_d7_decentred_fit`
working.  D1's weighted restriction, `_FIT_DISC_OUTSIDE_WEIGHT_REL`,
`_DECENTRED_FIT_POLY_ORDER` and its sample-count step-down, and the niche-C11
arbiter are untouched in both bases -- `fit_basis` changes columns, never rows.

**Cost**, measured as interleaved medians of one decentred call (the report has
the table): the opt-in path has no numba kernel -- the Chebyshev evaluation
drops into a `@njit(parallel=True)` recurrence per sample and the Zernike one
runs a chunked column generator in NumPy -- so it is **412.5 -> 463.9 ms,
1.12x**, on the same fixture WP-A26 priced its order raise with (N = 512,
dx = 8 um, `ray_subsample=8`, order 16), and 1.69x on a `ray_subsample=1` call
where the per-pixel evaluation of the fits dominates.  Paid only when asked
for.

**Refused rather than silently ignored**: `fit_basis='zernike'` with
`newton_fit='spline'` (no design matrix to express) or with
`inversion_method != 'newton'` (the direct inverse-map fit lives over the EXIT
coordinates, a different domain with a different disc) raises, and the gate runs
on every call rather than only on the calls that build a polynomial fit.

### Added -- tests

* `tests/unit/test_audit2609_b10_zernike_fit_basis.py` (30 tests, 5 s) --
  the two bases' term counts, their equal span by rank and by projection, the
  radial recurrence against the textbook Zernikes, orthonormality on the disc
  by exact quadrature, the gradient against a central difference inside and
  outside the disc; the two oracles against each other (the inline exact conic
  trace and the closed-form Fermat sphere, to the float64 floor) and the fitted
  OPL against them on a HELD-OUT lattice; that a change of basis does not move
  the fitted polynomial while two degrees of order does; that the arbiter's two
  candidates score the same on both bases; the marginal-ray -> defocus
  coupling as a fail-before,
  measured in both bases and falling with the ORDER; the conditioning in both
  directions, including the `(r/R)^n` decay that makes this opt-in; the
  default's fit-state payload; and the refusals.  No wall clock is asserted:
  where the cost matters the tests count step-down firings instead.
