# WP-A24 -- changelog text

### Fixed -- traced carrier chain: the shipped decentre calibration was stale, and its ordering had inverted

`propagate_traced_carrier_chain`'s `decentre_fit_frac` warning
(`lumenairy/propagators/carrier.py`, `_check_decentred_fit`) quoted a
2026-07-29 measurement of the `K = -n^2` conic stand-in -- "0.00 w -> 0.997;
0.25 w -> 1.002; 0.50 w -> 1.005; 0.75 w -> 0.977; 1.00 w -> 0.983;
1.50 w -> 0.923" (chain / independent ray-trace + Kirchhoff oracle, EE2
ratio).  Re-measured on the same stand-in, the same oracle and the same metric
on 2026-09-12: **0.970 / 1.010 / 1.008 / 1.002 / 0.986 / 0.903**.  The fall
past one beam radius reproduces, but the on-axis and 1.0 w rows have **crossed
over**, so the message stated the opposite of the measured ordering to every
user whose fan trips the guard.  Both the message and `_check_decentred_fit`'s
docstring now carry the re-measured six points, the date, the oracle's own
floor on the ratio (6.4e-04, measured by sweeping the oracle's pupil patch
2.2 -> 4.0 beam radii and halving its quadrature pitch), and the reason the
ordering moved: it is set by the terminal fine retrace's v5.35
inverse-characteristic evaluator, not by the decentre.  With
`traced_kwargs={'inverse_map': False}` the same stand-in reads 0.997 on axis
and 0.971 at 1.0 w -- the 2026-07-29 ordering -- and the evaluator improves the
WORSE of the two arms (field-fidelity defect 1.90e-03 -> 1.61e-03, decentred
FWHM ratio 1.0952 -> 1.0000), so it is a trade and not a loss.  No behaviour
changed; the numbers a user is told did.

### Changed -- `test_niche_d6_exact_tilted_leg`'s on-axis EE2 bar is a derived two-sided envelope

`test_decentred_carrier_decentre_penalty_envelope`'s `assert r_on > 0.97` was a
per-build number: a one-sided threshold parked 0.0266 under a single
2026-07-29 reading of 0.9966, with no oracle error floor, no defect scale and
no upper arm (`docs/TESTING_STANDARDS.md` S5).  It read **0.969787** and went
red on a 2.2e-04 margin.  It is now `0.95 < r_on < 1.02`, derived: the oracle's
own floor on this ratio is 6.4e-04; the 0.0268 between 0.9698 and 0.9966 is the
inverse-characteristic evaluator's documented model choice (42x the floor, so
both readings are real, and a bar inside that band measures the build); the
lower arm sits 1.66x under the measured shortfall from 1 and 9.7x over the
mildest on-axis defect the fixture has ever shown (a mis-placed readout centre,
ratio 0.516; the paraxial route reads 0.217); the upper arm sits 0.010 over the
largest value the metric takes anywhere on the stand-in's decentre curve
(1.010 at 0.25 w).  The docstring's premise -- "the chain tracks the oracle on
axis and slightly worse when the same beam is decentred" -- is corrected to the
measured ordering, and the `r_off` arm's recorded value is updated
(0.9828 on 2026-07-29, 0.9855 on 2026-09-12; its bar, already two-sided, is
unchanged).

### Added -- `tests/unit/test_audit2609_a24_decentre_calibration.py` (7 tests)

Pins the corrected warning text (fail-before: the superseded 0.997 / 0.983 pair
is asserted absent), the attribution -- `final_leg='exact'` never reaches
`_default_focus_standoff` or the beam-referenced term WP-A6's C1 added to it,
with the paraxial leg as its falsifier -- and the oracle floor the restated
envelope is derived against (EE(2 um) moves 6.8e-04 between a 2.2 w and a
3.2 w pupil patch, against a 5e-03 bar).

### Note -- WP-A6 is NOT the cause of the d6 failure, and the bisect that said so is corrected

`WP-A16_REPORT.md` section 8.3 attributed the d6 crossing to `a18ab074`
(WP-A6, "focus readout sized from the BEAM").  Re-measured read-only with
`git archive`: **`a18ab074^` already reads 0.969787**, bit for bit HEAD's
value, and C1's resolver is not on that fixture's code path at all.  The step
is `a4e8e855` 0.996575 -> `4e8ea247` 0.971526 (**-0.0251**, the
inverse-characteristic evaluator, reproduced at HEAD to six digits by
`inverse_map=False`) plus `0067d63b` 0.971526 -> `f602b72c` 0.969787
(**-0.0017**, WP-A1's raytrace corrections), bit-stable through every commit
since.  No change was made to the C1 resolver or its guard.
