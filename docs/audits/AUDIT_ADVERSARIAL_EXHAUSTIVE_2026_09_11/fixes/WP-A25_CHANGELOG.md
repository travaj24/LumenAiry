# WP-A25 -- CHANGELOG text (paraxial focus readout)

Finding A25 of `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A25_REPORT.md`.
File: `lumenairy/propagators/carrier.py`.

---

### Added -- `replica_fill='zero'`: keep an oversized readout window, drop the periodic replicas (A25)

Both public focus readouts finish on `angular_spectrum_propagate_mft`, whose
reconstruction obeys `E(u + period) == E(u)` identically in the absolute output
coordinate, so only `|u| <= period/2` about the transform's own origin carries
measurement.  `on_replica='error'` has refused a wider request since D3
(2026-08-06); with the refusal waived, the only answer available out there was
the periodic replicas the transform writes.  That is not a degraded reading of
the field: a replica is a FULL-AMPLITUDE image of the core laid down where the
real field is weak, so it wins every max / argmax / centroid /
encircled-energy reduction taken over the window -- including the one a spot
budget uses to decide where the spot IS.

`carrier_referenced_focus_readout` and `carrier_referenced_exact_focus_readout`
take `replica_fill={'repeat', 'zero'}`, default `'repeat'`
(`lumenairy/propagators/carrier.py:3841`), reachable through
`propagate_traced_carrier_chain`'s `focus_readout` dict and
`propagate_traced_carrier_chain_multi`'s `output_grid`
(`lumenairy/propagators/carrier.py:9531`, `:9805`).  `'zero'` blanks the part
of the window that lies outside one period; `'repeat'` is the historical
answer and stays the default, because a caller deliberately reading the
periodic reconstruction needs it -- the multi-congruence chain's `K == 1`
field-of-view contract requires the whole requested grid live, and the V3
off-axis ghost fixtures exist to show that a window one whole period off the
chief ray returns a full-amplitude copy.

The region the knob governs is the EXACT complement of the replica guard's own
condition, so it is empty precisely when
`2|centre_out| + N_out*dx_out <= period` holds on both axes: a faithful window
is returned by IDENTITY on either setting, and the two settings are
bit-identical inside one period.  Neither moves the leg -- the standoff is
still the accuracy-optimal one `_default_focus_standoff` resolves from the
beam.

Measured on the P2 design battery's unclipped doublet cell (a 2 mm Gaussian
through a 50 mm achromat at a 2.5x aperture; readout 512 x 0.5 um = 256.000 um
against a 124.113 um period = 2.063 periods; `tests/unit/test_niche_p2_design_battery.py::test_battery_through_focus_unclipped_doublet_matches_gaussian`
with `replica_fill='zero'`, 2026-09-13), against the analytic Gaussian focus of
the chain's own exit beam:

| | `'repeat'` | **`'zero'`** | standoff 768 um | standoff 1536 um |
|---|---|---|---|---|
| best-focus FWHM | 20.500 um | **18.500 um** | 18.500 um | 18.500 um |
| FWHM / analytic (17.413 um) | 1.1773 | **1.0624** | 1.0624 | 1.0624 |
| EE inside 1 / 2 / 3 waists | 0.3531 / 0.4953 / 0.5032 | **0.8585 / 0.9970 / 0.9980** | 0.8585 / 0.9970 / 0.9980 | 0.8585 / 0.9970 / 0.9980 |
| best-focus plane | +0.3934 mm | **+0.1311 mm** | +0.1311 mm | +0.1311 mm |
| returned / stop-plane power | 5.7004 | **0.99873** | 0.99800 | 0.99800 |
| peak of the best plane | pixel (0, 0) | **(256, 256)** | (256, 256) | (256, 256) |

-- the fixture had been scoring a replica sitting in the window's CORNER, where
three quarters of the encircled-energy disc falls off the grid.  The last two
columns are the same readout taken at a leg long enough for one period to cover
the window: three independent geometries with no replicas in them, agreeing to
the digit, and agreeing with the fixture's own 2026-07-25 record (18.5 um,
1.062x, EE1w 86.0 %, EE2w 99.7 %, EE3w 99.8 %).

The line between "the wings are wrong" and "everything is" is exactly TWO
periods -- the nearest replica's centre sits one period from the origin, the
window's edge at half its span -- and it is what the same fixture's own history
turns on.  Same fixture, same fill, only the leg varied:

| standoff | period | window / period | FWHM | EE2w |
|---|---|---|---|---|
| 6.0 z_R = 3147.166 um (the pre-2026-08-06 default) | 1049.606 um | 0.244 | 18.500 um | 0.9970 |
| 0.8 z_R = 419.622 um | 139.948 um | 1.829 | 18.500 um | 0.9970 |
| 337.468 um (the extent-following law, pre-C1) | 112.548 um | 2.275 | 16.500 um | 0.9527 |
| 372.144 um (C1, HEAD) | 124.113 um | 2.063 | 20.500 um | 0.4953 |

At 1.829 periods 1.2348x of the window's power is already replicas and the
reading is still exact, because everything the fixture measures is inside the
core; past two periods an image of that core is in the window and the
`argmax` finds it.

### Fixed -- the replica refusal no longer promises that a peak or a width still reads correctly (A25)

`_check_readout_replica`'s message and both readouts' `on_replica`
documentation said that past one period "the spot CORE is unaffected -- so a
width or a peak still looks right -- while second-moment / r^2-weighted /
large-radius encircled-energy / centroid metrics read wildly wrong".  That
holds up to 1.5 periods and fails beyond TWO, where the core's own replica
lands inside the window: on the battery cell at 2.063 periods the peak of the
scan's best plane IS a replica, an argmax-led width reads 20.50 um against an
analytic 17.41 um, and the encircled energy about it reads 49.5 % against
99.70 %.  The battery's own waiver cites exactly the superseded premise ("every
metric it takes ... is confined to the core").

The refusal now reports whichever regime the request is in -- it knows the
ratio -- with the measured counter-example, and names `replica_fill='zero'` as
the way to keep the window without the replicas
(`lumenairy/propagators/carrier.py:3769`).  The measured overshoot, the alias
count per edge, the largest safe `N_out` and the `ALIASES` / `REPLICAS` tokens
it already carried are unchanged.

### Added -- how much of a readout window is measurement (A25)

`_period_out['faithful_samples']` on both public readouts, on either fill, and
`readout_faithful_samples` on the chain's stage dict beside `readout_period` /
`readout_containment` / `readout_window_energy`
(`lumenairy/propagators/carrier.py:3917`, `:3529`): the `(nx, ny)` samples per
axis that lie inside one period.  `(N_out, N_out)` whenever the window is
faithful; `(249, 249)` of 512 on the battery cell above.  The number was
already computed inside the refusal message, where a waiving caller never saw
it.

### Added -- `tests/unit/test_audit2609_a25_carrier_focus_readout.py` (17 tests)

Pins the battery cell against the analytic Gaussian as a derived two-sided
envelope (the truth 1.0624x measured on three independent replica-free
geometries, the reading quantised at one radial bin = 2 `dx_out` = 0.0574 in
ratio units, the defect it catches 0.4953 EE2w); the two-period criterion
two-sidedly on the same leg (`N_out = 480`, 1.934 periods, reads 18.500 um /
0.9970 with `'repeat'`; `N_out = 512`, 2.063 periods, reads 20.500 um /
0.4953 -- the arms bracket the criterion by eight output samples each side);
the knob two-sidedly (the two fills bit-identical inside one period, `'zero'`
exactly zero outside it, `'repeat'` non-zero there, a faithful window returned
by identity on both, the vocabulary validated, complex64 preserved, the key
reaching the readout through the chain's `focus_readout` dict); the mechanism
as a DECISION (with `'repeat'` the scan's peak sits more than a quarter period
from the window centre and reduces to within one focal waist of the origin --
that is what makes it an image of the core); that WP-A6/C1's beam-referenced
leg is still the one this cell runs on (372.144 um against the
carrier-referenced 337.468 um, re-measured from the fixture's own exit
envelope); and the new stage diagnostic.

### Note -- WP-A6/C1 is not the cause of the battery step, and it is not undone

The bisect that routed this to `a18ab074` is right about the commit and wrong
about the fault.  C1 lengthened the readout leg 337.468 -> 372.144 um because
the envelope handed to it carries a fitted residual curvature of +0.01762 /m
against a carrier `1/R` of -15.1596 /m -- the beam's own focus really does sit
past the carrier's, which the through-focus scan confirms independently (best
focus at +0.131 mm needs +0.0300 /m).  The Bluestein period followed the leg by
the same 10.27 %, 112.548 -> 124.113 um, and that moved the brightest replica
from output pixel (30, 30) -- 113.0 um off centre, reducing to 0.452 um from the
origin, far enough inside the window that the encircled-energy disc still fitted
and 0.9527 looked plausible -- to the corner at (0, 0).  Both readings were
artefacts; only the second was loud.  The leg, the containment (3.1893 measured
/ 3.2000 modelled) and the period are bit-identical before and after this
change.
