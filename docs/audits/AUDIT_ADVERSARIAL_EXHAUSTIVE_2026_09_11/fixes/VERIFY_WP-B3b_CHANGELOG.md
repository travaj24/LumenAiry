# VERIFY-WP-B3b -- CHANGELOG text (independent re-verification of the K6 call sites)

Release 5.47.0, folded into WP-B3b's own entry.  Finding **K6** of
`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11.md`: the adversarial
re-verification of WP-B3b (commit `908c02d6`) found one diagnostic
regression and one wrong documented rule, both in files that package
owned.  Files: `lumenairy/propagators/system.py`,
`lumenairy/propagators/mft.py` (docstring only).  Evidence and every
number below: `fixes/VERIFY_WP-B3b.md`.

---

### Fixed -- a `method='fresnel'` chain step warns again when the chain window holds only part of the beam (K6)

WP-B3b retired the `'fresnel'` leg's resample, and the K6 crop warning
went with it (`_warn_system_resample_crop`, `system.py:359`).  Its
changelog recorded that `fresnel_propagate_mft`'s own faithful-zone
warning takes over, so no diagnostic is lost.  It does not: on the chain
grid the two conditions are **disjoint**.  `fresnel_propagate_mft` warns
when `N_out*dx_out > lambda*|z|/dx_in`, which with the chain's
`dx_out = dx_in = dx` and `N_out = N` is

    N*dx > lambda*z/dx   <=>   z < N*dx^2/lambda

-- exactly the K1 under-sampled-chirp band -- while a beam outgrows the
chain's window in the other direction, at `z` **above** that bound.  No
geometry can trip both, so everywhere above `z = N*dx^2/lambda`, where
the beam progressively outgrows the grid, the chain returned a field
missing most of its power and said nothing.  Measured (lambda = 633 nm,
dx = 2 um; `P_wide` is the same field evaluated by
`fresnel_propagate_mft` on an 8x wider window):

| fixture | z / (N dx^2/lambda) | `P_out/P_in` | `P_wide/P_in` | 5.46.0 | WP-B3b |
|---|---|---|---|---|---|
| top-hat, radius 3 px, N = 64 | 30x | **0.031674** | 0.786324 | warned | silent |
| top-hat, radius 4 px, N = 128 | 20x | **0.114889** | 0.879506 | warned | silent |
| Gaussian, w0 = 2.6 px, N = 128 | 10x | **0.334870** | **1.000000** | warned | silent |
| top-hat, radius 5 px, N = 256 | 10x | **0.537900** | 0.973707 | warned | silent |
| top-hat, radius 4 px, N = 128 | 6x | **0.699974** | 1.094883 | warned | silent |
| grid-filling top-hat, N = 512 | 2x | 0.995833 | -- | warned | silent |

The third row is the decisive one: `P_wide` reads exactly 1.000000, so
the step conserves the power and two thirds of it is simply outside the
chain's window.  (The rows where `P_wide` exceeds 1 are the 8x window's
own replica regime at that pitch, so they bound the loss rather than
measure it.)

The leg now calls `_warn_system_fresnel_window`
(`lumenairy/propagators/system.py:415`, called at `:928`), which measures
the power the chain window keeps and raises the same `RuntimeWarning`
class, at the same `1e-6` retained-power bar, as
`_warn_system_resample_crop` -- naming the retained percentage, `z`
against `N*dx^2/lambda`, and the same three remedies (a larger `N`, a
coarser chain pitch, or `method='asm'`).  It fires only for
`|z| > N*dx^2/lambda`, which is exactly the band the retired crop warning
covered on this leg (`dx_new = lambda*z/(N*dx) > dx` is the same
inequality), so the partition between the three diagnostics is enforced
rather than described: below the bound a short window is not a crop --
the natural grid is finer than the chain's, so the reconstruction
replicates rather than truncates -- and the faithful-zone and K1 warnings
already cover it.  The bar has decades on both sides: contained Gaussians
at 1x, 2x and 3x the bound read `P_out/P_in = 1.000000000` for N = 64,
65, 128 and 256, worst departure **3.1e-8**.

**Values are unchanged.**  Archive-to-archive, 72 of 72 probe arrays are
byte-identical to WP-B3b's tree -- the MFT propagators, both
`resample_field` legs, every `'asm'` chain (bare, un-band-limited,
three-element, lens+aperture, anamorphic, tilted element, tilted chain),
`propagate_through_system_jax`, `apply_real_lens` on every propagator,
and every guard text.  Exactly one warning record changes, and it is the
probe where 5.46.0 emitted the crop warning.

`_warn_system_resample_crop`'s docstring (`system.py:359`) and the
`'fresnel'` leg's comment now say which of the three diagnostics covers
which condition, instead of describing one as the other's replacement.

### Fixed -- `resample_field`'s exact-window rule is a condition on `N_in`, not on the scale factor alone

The F6 paragraph WP-B3b added to `resample_field`'s docstring said the
extent-preserving default `N_out = round(N_in*dx_in/dx_out)` lands on one
reconstruction period "x0.5, x1, x2 and x4 at **any** `N_in`".  The
condition it states one clause earlier -- `N_in/scale` whole -- is a
condition on `N_in` as much as on the scale: **x2 needs an even `N_in`
and x4 an `N_in` divisible by 4**, alongside the x1.5-by-3, x1.25-by-5
and x1.7-by-17 rules it already gave.  Checked over nine `N_in` against
eight scale factors, the divisibility rule holds in every cell and the
"any `N_in`" claim fails at `N_in` = 127, 65, 63, 51, 34 and 17.

At `N_in = 65` the x2 default rounds to `N_out = 32` -- a 64-`dx_in`
window against a 65-`dx_in` period -- and the power ratio reads
**0.995181** on a rim-filling envelope and 0.999995 on a contained one,
against **0.999907** at `N_in = 128` on that same rim-filling envelope,
where the default lands on the period exactly.  The fixture the paragraph quotes is
`N_in = 128`, where all four of its "any `N_in`" scales happen to be
exact, which is why it read as true.

Corrected in place with that reading added
(`lumenairy/propagators/mft.py:611-620`).  Docstring only:
`scripts/record_history_fingerprints.py --check` reports
`lumenairy.propagators.mft` OK.

### Changed -- two load-bearing properties of the chirp-Z gate are now pinned

WP-B3b's gate is `method=('chirpz' if N_out*dx_out <= min(N_in)*dx_in *
(1 + 1e-9) else 'spline')`, and its report calls out two deliberate
details: the **per-axis `min`** (because `resample_field` reads one input
pitch for both axes, so the shorter input extent sets the period) and the
**`1e-9` slack** (which is `_warn_mft_output_window`'s own tolerance, so
the chain takes chirp-Z on exactly the windows that resampler would not
warn about).  Both claims are true and neither could fail: deleting the
`min`, or the slack, left all 39 of WP-B3b's pins green.

`tests/unit/test_audit2609_b3b_resample_call_sites.py` gains 21 tests
(39 -> 60), none of the existing ones weakened:

* `TestV3ThePerAxisPeriodIsLoadBearing` drives the in-glass `'fresnel'`
  gap -- the only one of the three call sites a non-square grid can reach,
  since `scalable_angular_spectrum_propagate` refuses one outright -- on
  32x64, 48x64 and 16x64 grids where the per-axis rule picks the spline
  and a bare x-axis rule would pick chirp-Z, with a 64x32 grid as the
  two-sided arm and an arm that forces chirp-Z there and requires
  `resample_field`'s faithful-zone warning to name the **y** axis.
  Deleting the `min` now turns 4 red;
* `TestV4TheGateSharesTheResamplersOwnTolerance` measures the resampler's
  tolerance directly (silent at one period + 5e-10, warns at + 2e-9) and
  walks each owner function's AST to require the same `1e-9` literal in
  the `method=` selector.  Deleting the slack now turns 1 red;
* `TestV1TheFresnelLegStillReportsAWindowLoss` (4) and
  `TestV2TheExactPeriodScaleFactorsDependOnNin` (3) pin the two fixes
  above; both carry an arm that fails on WP-B3b's tree.

### Migration

**No default moved and no value moved.**  `resample_field`'s `method`
default is still `'spline'`, the chain's `method` default is still
`'asm'`, `apply_real_lens`'s `wave_propagator` default is unchanged, and
every array this verification measured is bit-for-bit what WP-B3b
shipped.

One new diagnostic: a `propagate_through_system(..., method='fresnel')`
step at `|z| > N*dx^2/wavelength` whose chain window holds less than
`1 - 1e-6` of the input power now emits a `RuntimeWarning` beginning
`"propagate_through_system: the fresnel leg evaluated the integral on the
chain window"`.  5.46.0 warned on the same geometries with the K6 crop
text (`"the fresnel leg returned its natural output grid ... which CROPS
it"`), so code that filtered the old text and had nothing to match in
WP-B3b has something to match again -- with different wording.  Code that
asserted a `'fresnel'` chain step is warning-free on a beam that outgrows
its grid was asserting the regression.
