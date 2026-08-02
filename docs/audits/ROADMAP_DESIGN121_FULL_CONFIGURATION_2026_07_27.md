# Design 121 (full configuration) — what the library still needs

Consumer-side requirements note, written against **v5.31.0** (verified at that
tag, not inferred). Companion to `AUDIT_TRACED_CHAIN_DX_SCALING_2026_07_22.md`
(F-A/F-C/F-D shipped; see §7 on F-B's record) and
`AUDIT_TRACED_PRODUCTION_READINESS_2026_07_24.md`.

**Premise: the refractive half is done.** The v5.29 default flip
(`carrier_reference='sphere'` + `preserve_input_phase='remap'` +
`amplitude_model='ray_density'` + `remap_sampling='full'`) took the design-121
8-group relay from EE6 69.7% to **EE6 99.6 / EE3 88.8 / FWHM 3.450 um** at
N=2048, dx0=1.0 um — at the measured ideal-field ceiling for that readout.
Nothing below asks for more accuracy on that path.

**What this note is about:** design 121 is not a bare relay. The shipping
device is a **Dammann DOE feeding an 8x4 order fan (480 um pitch, +-46 mrad)
from an emitter array**. Every item below is a blocker or a guard rail for
running the *full* configuration, and none of them moved in v5.30 or v5.31.

Priorities are consumer-side (what blocks the study), not library-internal.

---

## P1 — A first-class "N independent congruences through one chain" facility

**This is the single blocker.** It has two faces that are the same feature:

* **Per-order (DOE).** A post-DOE field superposes 32 comparable-power beams at
  well-separated angles. `apply_real_lens_traced`'s entrance->exit map assumes
  one congruence per exit pixel, so the fan cannot go through the chain
  multiplexed — the library says so itself (`_lens_traced.py:2170-2179`, which
  names "comparable-power beams at well-separated angles (post-DOE at large
  split)" as the excluded case).
* **Per-emitter (array).** Design 121 is a 4x4 / 8x8 emitter array. The
  existing multi-emitter entry point `apply_real_lens_traced_multi` **cannot
  express the validated configuration**: it fixes `preserve_input_phase=True`,
  and `amplitude_model='ray_density'` / `fit_radius_beam_factor` now *raise* on
  the default `reuse_prepared=True` (`_lens_traced.py:5111-5137`). So the array
  case is currently locked out of the v5.29 physics that made the single-beam
  case work.

Both want: **run K clean single-congruence chains and recombine coherently at
the image plane.**

### What's missing mechanically

1. **The chain carrier is scalar-only.** `propagate_traced_carrier_chain`
   reduces every hand-off to `R = float(R_carrier)` (`carrier.py:703`, `:933`),
   so a *tilted* carrier `W = S(r) + (L x + M y)` is not expressible chain-level
   — even though the ELEMENT already accepts an arbitrary `ndarray` `W(x,y)`
   via `carrier=` (`_lens_traced.py:2156-2157`). A +-46 mrad order therefore has
   no way to be carried as its own congruence through the chain.
2. **No per-order/segmented route exists from the chain.** `segmented`,
   `multivalued`, `per_order` and `order_split` appear **zero** times in
   `propagators/carrier.py` (grep, v5.31). `apply_real_lens_traced_segmented`
   exists but is element-level only.
3. **`max_segments: int = 32`** (`_lens_traced.py:5428`) is *exactly* saturated
   by an 8x4 fan, leaving no headroom for zero-order leak or a stray order —
   and its multi-segment path routes through `_multi`, i.e. back into the
   contract in the bullet above.

### Requested

Either of these unblocks the study; the first is smaller, the second is what a
consumer actually wants to call:

* **(a) Tilted-carrier hand-offs** — let the chain carry `(R, L, M)` (sphere +
  linear tilt) instead of scalar `R`. Minimal, and makes per-order runs
  first-class since each order becomes a clean congruence.
* **(b) A per-congruence orchestrator** — e.g.
  `propagate_traced_carrier_chain_multi(fields_or_orders, groups, ..., recombine='coherent')`
  that runs each congruence through the shipped-default chain and sums complex
  amplitudes on a common image grid. Serves per-order and per-emitter with one
  implementation.

**Preferred variant for 121:** decenter each order to on-axis for its own chain
run and re-offset at the image plane — that keeps the +-46 mrad out of the
residual entirely, so each run sits inside the validated envelope rather than
2.3x outside it (see P3). If (b) does the decentre/re-offset internally, the
consumer never has to think about it.

**Acceptance:** the 32-order 121 fan reconstructs with per-frame power matching
the Dammann design uniformity (design 2.78%/frame, uniformity ~0.996) instead
of the 0.47 +/- 0.51%/frame scramble measured when the fan was pushed through
the chain multiplexed at v5.28.

> **MEASURED NOTE (niche D6, 2026-07-29) -- the exact high-NA final leg now
> carries a TILTED congruence.**  D1/D2 shipped (a) and (b) but left one gap:
> `propagate_traced_carrier_chain` RAISED `NotImplementedError` whenever a
> tilted congruence routed onto the EXACT high-NA final leg, so the 121 fan had
> to run `final_leg='paraxial'` and every per-order spot was capped far below
> the single-beam acceptance.  That is closed.
>
> **The obstacle, re-measured before it was fixed.**  `_fine_trace_group_exit`
> crops its pre-readout re-trace to `window_factor * w_entrance` about the GRID
> CENTRE.  The chain stores its envelope in the chief-ray-TRACKING frame, so
> that crop was already chief-ray-centred in absolute terms -- the real
> blocker is that `apply_real_lens_traced` builds its grid symmetrically about
> the OPTICAL AXIS (it traces the beam through the surfaces at the beam's
> physical transverse position), so ONE axis-centred window has to hold the
> axis AND the displaced beam.  On design 121, order (-4,-2), N=1024: chief ray
> **3.373 mm** off axis at the final group entrance against an entrance beam
> radius **3.126 mm** and `na_exit = 0.4053` -- the window grows **12.50 ->
> 18.54 mm (1.482x)** and the fine grid must grow with it.
>
> **The fix.**  `_fine_trace_group_exit` and
> `carrier_referenced_exact_focus_readout` both take `centre` (the chief ray)
> and `tilt` `(L, M)`, defaulted and short-circuited so the on-axis path is
> untouched.  The RETRACE widens its axis-centred window to
> `2*max(|x_c|,|y_c|) + window_factor*w`, band-limit-shifts the envelope to the
> beam's physical position, reconstructs against the DECENTRED sphere plus the
> tilt ramp and hands the element the matching `TiltedCarrier` -- the same
> hand-off the chain already performs on its coarse legs.  The READOUT
> references sphere AND tilt about the chief ray, takes its `window_factor`
> crop ABOUT THE CHIEF RAY (so its internal fine grid costs exactly what an
> on-axis beam of the same radius costs, however far off axis it sits) and asks
> the band-limited ASM Bluestein zoom for the window at `centre_out - chief`;
> the ASM is translation-covariant, so the tilt's own transverse advance and
> path piston are carried EXACTLY instead of being reimposed by paraxial
> bookkeeping the way `carrier_referenced_focus_readout` has to.
>
> **HEADLINE (design 121, post-DOE 6-group chain, N=1024, dx0 = 2.0 um, rs 4,
> `n_fine_cap` 12288, `window_factor` 4.0, common grid dx_out 0.4 um, tile
> 1024 px), through `propagate_traced_carrier_chain_multi` with
> `final_leg='auto'` -- it RUNS, no `NotImplementedError`:**
>
> | order | field angle | leg | FWHM (um) | EE3 % | EE6 % | EE12 % |
> |---|---|---|---|---|---|---|
> | (+0,+0) | 0 | paraxial | 6.800 | 31.7 | 66.0 | 82.9 |
> | (+0,+0) | 0 | **exact** | **4.400** | **87.6** | **99.5** | **99.7** |
> | (-1,+0) | 11.5 mrad | paraxial | 7.600 | 25.4 | 59.0 | 82.0 |
> | (-1,+0) | 11.5 mrad | **exact** | **4.400** | **81.7** | **98.5** | **99.6** |
> | (-4,+0) | 46.1 mrad | paraxial | 8.400 | 22.8 | 54.6 | 78.7 |
> | (-4,+0) | 46.1 mrad | **exact** | **4.400** | **64.8** | **90.1** | **98.0** |
> | (-4,-2) | 46.1 + 23.0 mrad | paraxial | 8.400 | 21.9 | 53.2 | 77.7 |
> | (-4,-2) | 46.1 + 23.0 mrad | **exact** | **4.400** | **60.2** | **87.8** | **97.1** |
>
> (FWHM is quantised by the 0.4 um common-grid pitch; measured at dx_out =
> 0.05 um on a single-order run the same two orders read **3.650 um** (0,0) and
> **4.150 um** (-4,-2) at the plane, **3.650 / 3.950 um** at best focus.)  The
> m=0 order is single-beam class against the shipped 3.450 / 88.8 / 99.6
> acceptance.  What the gate asked for is measured on every row: the exact leg
> is reachable and materially better -- **EE6 +33.5 points on axis and
> +34.6 points at the extreme order**, peak intensity 4.4x.
>
> > **RETRACTION (2026-07-29, adversarial verification).**  An earlier revision
> > of this note read "the +-46 mrad orders keep GENUINE field-angle
> > aberration, which is the physics."  **That was false, and the per-order
> > rows above are a LOWER BOUND on design 121, not its performance.**  An
> > independent oracle -- exact skew ray trace of the same post-DOE .zmx
> > surfaces (`lumenairy.raytrace` only: no propagator, no carrier machinery,
> > no FFT, no `window_factor`) plus an energy-conserving Debye / local-plane-
> > wave sum -- says the design is EQUALLY diffraction-limited at every order.
> > Per order (0,0)/(-1,0)/(-4,0)/(-4,-2): rms wavefront about the geometric
> > centroid **0.0820/0.0807/0.0784/0.0794 waves**, geometric rms spot radius
> > 0.427/0.402/0.296/0.422 um, PSF FWHM 3.508/3.510/3.521/3.547 um, **EE3
> > 90.73/90.77/90.75/90.37 %**, **EE6 99.90/99.90/99.92/99.91 %** (ray-count
> > converged, 161^2 vs 321^2 identical to 3 decimals; corroborated on axis by
> > `focus_scan_121.py`'s own printed ideal-field ceiling 3.45-3.55 um / 90.3 /
> > 99.8).  The chain's monotone loss with chief-ray offset is a LIBRARY
> > DEFECT, not the design.  **That much still stands.  Where it was localised
> > does not -- see the D7 correction.**
> >
> > > **CORRECTION (niche D7, 2026-07-29).  The localisation was wrong, and so
> > > was its calibration.**  This block used to say the defect was
> > > `apply_real_lens_traced` under a DECENTRED carrier, quoting a
> > > 3.7 -> 408 urad exit-slope curve (0.029 -> 3.147 um of blur) from
> > > `validation/repro_traced_carrier_121/decentred_fit_defect.py`, with the
> > > mechanism "the ray-FIT region follows the beam while the fit BASIS domain
> > > stays the axis-centred launch square, so the residual lands as COMA".
> > > Three independent measurements kill all of it:
> > >
> > > 1. **The curve is the repro script's own measurement artefact.**  That
> > >    script extracted the exit slope with a GLOBAL FFT derivative of the
> > >    de-chirped field.  Fed a SYNTHETIC field whose exit-slope error is
> > >    **0.36 urad by construction** (amplitude = the library's, phase =
> > >    `k0 * OPL` from an order-12 fit + tight Newton), the same oracle
> > >    reports **400.51 urad** at 0.97 w -- against the library field's
> > >    401.68.  The whole curve is the oracle, not the element.
> > > 2. **The element was already accurate.**  Measured aliasing-free (local
> > >    wrapped nearest-neighbour phase differences; per-pixel steps are
> > >    << pi, so nothing can fold), design 121's last group under the same
> > >    decentred carrier reads **1.28 urad on axis -> 7.16 urad at 0.97 beam
> > >    radii** pre-D7, i.e. 0.055 um of blur against a 3.5 um FWHM.  The
> > >    fitted maps' own exit cosines against the exact ray trace agree:
> > >    0.50 -> 6.91 urad at order 6, and **0.36 urad at order 12**.
> > > 3. **The prescribed fix is a no-op.**  Re-mapping the tensor-Chebyshev
> > >    BASIS DOMAIN onto the off-centre disc cannot change the answer: the
> > >    total-degree polynomial space is affine-invariant, so the weighted
> > >    least-squares solution is the same polynomial.  Measured: OPL residual
> > >    over the beam 2.5076 nm (launch-square domain) vs 2.5076 nm (disc-bbox
> > >    domain), normal equations and SVD alike, while cond(Gram) falls
> > >    1.0e10 -> 3.2e4.  It is also a liability -- the Newton loop evaluates
> > >    the same fit over the WHOLE launch square, where the re-mapped basis
> > >    reaches `max|T_k| = 5.7e8` at order 12 and the two identical fits then
> > >    differ by **9.9e-7 m** of `x_out` at the launch corners (5.2e-9 m at
> > >    order 6).  REFUSED, with those numbers, in
> > >    `_DECENTRED_FIT_POLY_ORDER`'s note.
> > >
> > > **What is real, and what D7 shipped.**  An off-centre fit disc of radius
> > > `r` about a chief ray `|c|` off axis covers the aperture out to `|c| + r`
> > > instead of `r`, so the SAME total-degree budget buys a worse fit over
> > > strictly more aberrated territory.  Design 121's last group, OPL residual
> > > over `r <= w`: order 6 reads **0.177 nm on axis and 2.508 nm at 0.97 w**
> > > (14x), recovering to 0.667 / 0.121 / 0.114 nm at orders 8 / 10 / 12 and
> > > degrading again to 0.199 nm at 14 (conditioning).  So
> > > `apply_real_lens_traced` now RAISES the fit order to
> > > `_DECENTRED_FIT_POLY_ORDER = 10` on exactly the off-centre branch (the
> > > same branch `_FIT_DISC_OUTSIDE_WEIGHT_REL` regularises), stepping back
> > > down whenever the disc holds fewer than 3 samples per basis term, and
> > > exposes it as `decentred_fit_poly_order`.  The concentric / on-axis path
> > > is untouched and **byte-identical** (21 configurations, max |dE| = 0.0
> > > against a pristine pre-D7 tree, including `newton_fit='spline'`,
> > > `inversion_method='fit'`, a tilted-but-CENTRED carrier and
> > > `beam_centre=(0,0)`); `decentred_fit_poly_order=<newton_poly_order>`
> > > reproduces pre-D7 exactly (max |dE| = 0.0 on the three changed cases) and
> > > is the tests' fail-before switch.
> > >
> > > **Measured payoff.**  Element, design 121 last group, aliasing-free exit
> > > slope: **7.16 -> 0.90 urad at 0.97 w** (on axis 1.28 urad, unchanged), so
> > > the decentred figure now sits below the on-axis one *in the untilted
> > > sweep* -- the qualifier matters, because the repro's decentre sweep runs
> > > entirely at `tilt_L = 0`, and against the TILTED on-axis control (48.7
> > > mrad, the regime 121's orders are actually in) the same 0.90 urad sits
> > > **1.4x ABOVE** a 0.64 urad baseline.  The payoff and every conclusion
> > > drawn from it are indifferent to the choice -- 0.90 urad is 0.007 um of
> > > blur on a 3.5 um FWHM either way -- but the unqualified "sits BELOW"
> > > claim, as first written here, is true only of the untilted column.
> > > Conic stand-in
> > > (`K = -n^2`, decentre-invariant truth), chain / oracle EE2 ratio:
> > >
> > > | decentre | 0 w | 0.25 w | 0.50 w | 0.75 w | 1.0 w | 1.5 w |
> > > |---|---|---|---|---|---|---|
> > > | pre-D7 | 0.9966 | 0.9743 | 0.9608 | 0.9554 | 0.9498 | 0.8385 |
> > > | D7 | 0.9966 | 1.0024 | 1.0046 | 0.9765 | **0.9828** | **0.9225** |
> > >
> > > (FWHM ratio at 1.0 w: 1.0952 -> 1.0000.)  Design 121 per order, exact
> > > final leg, `n_fine_cap` 12288, `window_factor` 4.0, dx_out 0.4 um,
> > > N_out 1024, RN 1024, rs 4 -- FWHM um / EE3 / EE6 / EE12 %:
> > >
> > > | order | pre-D7 | D7 | oracle EE3 / EE6 |
> > > |---|---|---|---|
> > > | (+0,+0) | 4.400 / 87.62 / 99.49 / 99.65 | 4.400 / 87.62 / 99.49 / 99.65 | 90.73 / 99.90 |
> > > | (-1,+0) | 4.400 / 81.73 / 98.51 / 99.65 | 4.400 / **86.01** / 99.15 / 99.62 | 90.77 / 99.90 |
> > > | (-4,+0) | 4.400 / 64.01 / 90.54 / 98.08 | 4.400 / **68.13** / 91.52 / 97.76 | 90.75 / 99.92 |
> > > | (-4,-2) | 4.400 / 60.48 / 87.54 / 97.07 | 4.400 / **65.26** / 87.88 / 96.50 | 90.37 / 99.91 |
> > >
> > > **The honest residual.**  That closes 47 % of the (-1,0) gap and only
> > > 15-16 % of the (-4,0) / (-4,-2) gap.  Most of design 121's per-order loss
> > > is still unexplained, and it is NOT: the element's ray fit (0.90 urad of
> > > exit slope = 0.007 um of blur cannot cost 25 EE3 points), the fine
> > > retrace grid (`n_fine_cap` 12288 vs 16384: EE3 65.26 vs 65.26), the
> > > Newton cap (`newton_max_iters` 12 vs 40: 65.26 -- the 37 % "unconverged"
> > > pixels are out-of-domain edge pixels), the readout window
> > > (`window_factor` 4.0 / 6.0 / 8.0: 65.26 all three), or the coarse grid
> > > (D6: RN 1024/2048/4096 move EE3 by 0.15 points).  By elimination it lies
> > > in the chain's TILTED-CONGRUENCE TRANSPORT across the coarse legs --
> > > where the recorded `sin`-vs-`tan` chief-ray caveat below also lives.
> > > That is the next phase's target, not this one's claim.
> >
> > **What the library does about it now:** `propagate_traced_carrier_chain`
> > and `..._multi` carry `on_decentred_fit` (default `'warn'`), which fires
> > at every traced hand-off whose chief ray exceeds `decentre_fit_frac`
> > (default 0.5) beam amplitude radii off the element grid centre and says in
> > as many words that the resulting per-order metric is a LOWER BOUND.  D7
> > KEPT the guard -- the per-order metric IS still a lower bound (EE3 65.3 %
> > against the oracle's 90.4 % at the extreme order) -- and re-pointed its
> > calibration at the two decentre-INVARIANT measurements above.  Pinned by
> > `tests/unit/test_niche_d6_exact_tilted_leg.py::`
> > `test_decentred_carrier_decentre_penalty_envelope` (the post-fix envelope;
> > it was `test_decentred_carrier_fit_defect_envelope`, deliberately written
> > to fail loudly the day the defect moved -- which it did) and by
> > `tests/unit/test_niche_d7_decentred_fit.py`.  Repro:
> > `validation/repro_traced_carrier_121/decentred_fit_defect.py`, whose
> > oracle D7 replaced with the aliasing-free one and which now prints BOTH so
> > the artefact stays reproducible.
>
> **Cost, and the honest refusal.**  The 1.482x window means the same `dx_fine`
> needs 1.482x the pixels: measured 75 s/order at `n_fine_cap` 8192 (untilted)
> against 172 s at 12288 and 316 s at 16384 (tilted).  12288 and 16384 agree to
> **0.03 EE points** (EE3 60.27 vs 60.29, EE6 86.53 vs 86.56), i.e. the tilted
> leg is grid-converged at 12288.  At `n_fine_cap = 8192` the capped
> `dx_fine = 2.262 um` is COARSER than the exit sphere's Nyquist pitch
> `lambda/(2 NA) = 1.616 um`, so the new `on_tilt_exact_grid` guard (default
> **`'error'`**) REFUSES, naming the chief-ray offset, the beam radius, the
> window and its ratio, the required `n_fine`, the cap that bound it, the NA
> that would be discarded (0.2895) and the three remedies.  `'warn'` accepts
> the degraded leg; nothing ever falls back to paraxial silently.  The other
> refusal is geometric: a chief ray the co-moving grid cannot hold raises
> before any work.
>
> **Not regressed.**  `focus_scan_121.py` (the shipped single-beam acceptance,
> pure library defaults, N=2048/NFC=8192/WF=4.0) still reads best focus
> **3.450 um / 88.8 / 99.6, on-axis**, to the digit.  An untilted
> `TiltedCarrier(R, 0, 0)` is `np.array_equal` to the scalar carrier.  D2's
> power bookkeeping is untouched: the full 32-order fan at `final_leg='paraxial'`
> still reads `max |share/design - 1| = 3.0e-4` (VERDICT PASS), and the same
> acceptance on the exact leg reads **4e-4** (orders (0,0)+(-4,-2)) and
> **2e-5** (orders (-1,0)+(-4,0)), capture 0.998, throughput spread 6e-4.
> Bonus: the exact readout's Bluestein period is **4762-4808 um** against the
> paraxial route's 483-492 um, so the D2 replica guard gains a 10x margin.
>
> **One caveat, ~~recorded not fixed~~ NOW FIXED** (niche C3, 2026-07-30; it
> belonged to D1's convention).  The chain carries `(L, M)` as DIRECTION
> COSINES -- advancing the chief ray by `z L / cos(theta)`, exact for a free
> leg -- but USED TO obtain them from the group's PARAXIAL ABCD, where they are
> slopes.  Where the exit tilt was large the two mixed `tan` with `sin`: on the
> D6 synthetic stand-in (`L_out = -0.20`, `f = 3 mm`) the predicted chief ray
> sat **12.4 um** from the Fermat focus while the exact leg's spot landed on
> the FERMAT focus (measured centroid within 0.002 um), i.e. not on the
> predictor.  Design 121's final group leaves `L_out = 8.3e-5`, where the same
> term is **6 pm** -- invisible, which is why nothing in this study saw it
> before.
>
> **The fix, and the refutation that shaped it.**  `_group_chief_transfer` no
> longer linearises the group at all: the chief ray is TRACED through that
> group's own surfaces, front vertex -> back vertex, with the same engine the
> tests use as their oracle (apertures removed, since a purely geometric
> predictor must not change regime by vignetting).  The obvious repair was
> tried FIRST and REFUTED.  Measured on the D1 two-singlet relay at 46 mrad,
> against an exact meridional ray trace at **1.783248056 mm**:
>
> | chief-ray predictor | image-height residual |
> |---|---|
> | old lumped paraxial ABCD (raw cosines) | **+0.1214 um** |
> | cosine<->slope conversion of that ABCD | **+1.1208 um** (9x WORSE) |
> | EXACT chief-ray trace (shipped) | **0.0000 um** |
>
> The conversion loses because a lumped group ABCD is not a single angle
> convention at all: refraction at a surface is linear in SINES (Snell) while
> free transfer inside the group is linear in TANGENTS, and a group of this
> class is refraction-dominated (`B = t/n ~ 2 mm` against `GAP`/`fd` legs two
> orders larger, which already used the exact `tan`).  So no scalar convention
> can be right for it, and the predictor had to stop being linear.  The ABCD
> survives in two documented roles only -- the exact transfer's paraxial limit
> (checked to 1e-9 relative at 1e-7 rad) and the fall-back for a group the ray
> engine cannot build -- and an untilted, undecentred ray still short-circuits
> to zeros, so the on-axis path is byte-identical.
>
> On the D6 stand-in the predictor now lands on the Fermat focus to **1e-19 m**,
> **0.0005 um** from the measured spot centroid (1/6700 of the 3.15 um FWHM).
> Pinned by `tests/unit/test_niche_d6_exact_tilted_leg.py` (38 tests, no
> `.zmx`, ~135 s, RAM-guarded), whose oracle is a lumenairy-FREE inline exact
> conic raytrace plus a Rayleigh-Sommerfeld surface integral: the exact leg
> tracks it to **0.0 % on FWHM / 0.982x EE2 / 0.995x EE4** (was 9.5 % / 0.951x
> / 0.991x -- the readout's `centre`/`tilt` are exact now too), while the
> paraxial leg it replaces is **1.857x** wide (EE2 10.6 % against 71.6 %).
> That fail-before ratio was **3.19x** before the fix and moved for a good
> reason: the paraxial leg's wavefront is still ~200 rad wrong, but its spot is
> now correctly PLACED, so less of it falls off the readout window.  The
> discrimination is carried by two numbers that did not weaken -- the paraxial
> leg keeps 14.8 % of the oracle's EE2 and puts its brightest pixel on the
> window edge, 8.25 um from the Fermat focus, with 2.8 % of its power within
> 2 um of it against the exact leg's 70.2 %.
>
> **Still open (2026-07-30).**  `_chain_chief_ray_at_target` -- the helper that
> places `propagate_traced_carrier_chain_multi`'s per-congruence readout tiles,
> and which `test_niche_d4_dgrating.py` pins as "the library's own predictor
> agrees with the chain to the digit" -- was NOT converted and still applies the
> lumped ABCD.  On the D6 stand-in it therefore now reports a chief ray
> **12.372 um** from the one the chain itself lands on, and a tile centred on it
> clips the spot (EE2 0.363 against 0.703).  Design 121's `L_out = 8.3e-5`
> keeps this at 6 pm there, so the shipped fan is unaffected, but the helper
> should be routed through `_group_chief_transfer` too.

---

## P2 — Let the full design be expressed as ONE object

`DGRATING` surfaces are imported as **flat optical surfaces** — the diffractive
behaviour is dropped (`io/prescriptions_zemax.py:829`). Consequence: the 121
prescription that the chain sees has never contained the DOE, and the consumer
must hand-build the grating, hand-split the chain at the DOE plane, and
hand-fold the 51.539 mm DOE gap into a neighbouring group's gap. That manual
fold is exactly the kind of bookkeeping that produced a wrong answer once
already in this study.

**Requested:** import `DGRATING` parameters (period, orders, and the existing
`PARM 1` lines/um already parsed at `:474`) into the prescription, and let the
chain's `groups` list accept a DOE entry between refractive groups. The
*propagation* can still be per-order under P1; this is about the design being
expressible and the gaps being bookkept by the library.

> **MEASURED NOTE (niche D4, 2026-07-28).** Shipped as asked, in both halves.
> `load_zemax_zmx` now imports `DGRATING` surfaces into `rx['diffractives']`
> (geometry import unchanged: still a flat/conic surface, still no aspheric
> coefficients, still nothing on the lens-only `'surfaces'` list), and
> `propagate_traced_carrier_chain`'s `groups` list accepts
> `{'doe': rx['diffractives'][k]}` as an entry.  On the real 121 `.zmx` the
> two DGRATINGs read period **113.7656 um** (PARM 1 = ±0.00879 lines/um, which
> is per MICROMETRE regardless of `UNIT`) and design orders **-4 / -2**, i.e.
> **±46.06 / ±23.03 mrad** — and the imported period implies a frame pitch of
> **479.96 um** against the design's 480, so the .zmx's own number is the
> design number.  The **51.5393 mm** gap arrives with the DOE (`gap_before`),
> as does the **7.000 mm** `gap_after`, both measured to the neighbouring
> POWERED elements with the STOP/reference dummy planes collapsed — identical
> to what the consumer folds by hand today, so the fold is now the library's.
> The `COORDBRK` **+90 deg z-roll** between the pair is folded into the second
> grating's azimuth (270 deg, the extra 180 from the negative lines/um), which
> is what makes the crossed pair actually crossed.
>
> Two measured results worth carrying. (i) **Expressing the DOE cannot move
> the validated relay.** Inserting both DOE entries at order 0 into the
> shipped 8-group 121 chain (which replaces the hand-folded 58.5393 mm gap
> with 51.5393 + 7) is **BITWISE identical** to the shipped arrangement,
> exact final leg included. That required *not* interrupting the carrier leg
> at the DOE plane: the order's whole action on the envelope in its own
> tracking frame is a complex CONSTANT, which commutes with the transport
> exactly, so deferring the transport is what turns "agrees to 1e-11" into
> "bitwise". (ii) For a DEFLECTED
> order the entry reproduces the manual hand-split that
> `fan_multi_121.py` performs today: on the real 121 at order (-4,-2),
> N=1024, dx0 = 2.0 um, `final_leg='paraxial'` — re-measured 2026-07-28 —
> `max|dE|/max|E|` = **5.5e-7**, power to 6.9e-9 relative, chief ray
> identical at (-1919.686, +959.843) um, a 479.92 um implied frame pitch.
> The per-order fan is now one `groups` list plus a per-congruence
> `'doe_order'` on `propagate_traced_carrier_chain_multi`.
>
> Note the DOE entry's order DEFAULTS to the .zmx's design order (-4 / -2 for
> the 121), not to 0 — a bare `{'doe': rx['diffractives'][k]}` reproduces the
> order Zemax's sequential trace follows. It is reported in `stages`, and on
> design 121 it is loud rather than silent.
>
> **STALE AS WRITTEN (superseded by niche D6, 2026-07-29).** This paragraph
> originally ended "a deflected congruence makes the exact high-NA final leg
> raise (D1's documented limit), naming `final_leg='paraxial'`". That refusal
> no longer exists: D6 carries a TILTED congruence through the exact leg, and
> the 32-order fan runs there. What can still refuse on that route is narrower
> and differently named — `on_tilt_exact_grid` (a `RuntimeError`, default
> `'error'`) when the fine grid is too coarse to sample the widened,
> axis-centred retrace window, and the trailing-DOE `NotImplementedError`
> (D4), which is about a screen after the last group, not about deflection.
>
> **CORRECTION (2026-07-28, adversarial re-measure).** The first revision of
> this note justified the deferred transport with "the carrier step is
> path-dependent wherever it crosses the carrier's own focus — which this leg
> does (one 58.539 mm step and a 51.539 + 7 pair land on co-moving pitches
> 5.5x apart)". **That attribution is wrong for design 121 and is withdrawn.**
> Measured at the pre-DOE group exit (`Lens S5-S7`, N=1024, shipped defaults):
> the carrier there is `R = +703591.2 mm` — the DOE sits in **collimated**
> space, 703 m from the carrier's focus — and the one-step and split routes
> land on the **same** co-moving pitch (51.23386 um, ratio 1.000000), agreeing
> to `max|dE|/max|E|` = **2.1e-11**. Independently: the deflected (-4,-2)
> integrated chain and the hand-split agree to 5.5e-7, which a 5.5x pitch
> split could not produce. The *mechanism* is real but lives only in the
> near-focus corner: on a synthetic 58.5393 mm leg split at 51.5393 mm the
> pitch ratio is 5.4x / 49x / 278x for a carrier focus at 51.0 / 51.6 /
> 51.55 mm, and **1.000000000000** everywhere the split plane is clear of the
> focus — including a focus *inside* the leg but away from the split
> (R = -3 to -45 mm, field agreement 3e-12 to 6e-11). Both halves are now pinned synthetically in
> `tests/unit/test_niche_d4_dgrating.py`. So: deferring the transport is
> justified for design 121 by the **bitwise** result alone (1e-11 -> 0), and
> for the general case by the near-focus corner — not by anything about this
> design's own leg.
>
> Same pass, one real defect fixed: a DOE entry's order `amplitude` did not
> reach `propagate_traced_carrier_chain_multi`'s `power_exit` when the screen
> was the LAST entry in `groups` (the DOE stage reported no power, so the
> accounting fell back to the last lens group's, measured before the screen
> scaled the field). `capture` then read |amplitude|^-2 too small — 0.2497
> instead of 0.9988 at amplitude 0.5 — and `on_readout_clip` fired on
> bookkeeping rather than on a clipped halo (a hard failure at
> `on_readout_clip='error'`, which is what a production fan run should use).
> The DOE stage now reports `power` across the screen; `capture` is
> amplitude-independent at both placements, and `throughput` scales as
> |amplitude|^2 as it should.
>
> **CORRECTION 2 (2026-07-28, second adversarial re-measure).** Three more
> defects, all in the *drop-in* contract this note advertises, all now fixed
> at root and pinned synthetically in `tests/unit/test_niche_d4_dgrating.py`.
> Design 121 was immune to all three, which is why the acceptance above
> reproduced while the feature was still wrong for other designs.
>
> 1. **The loader recorded the inter-DOE leg twice.** A diffractive is an
>    anchor for the neighbour scan, so for two DGRATINGs `d` apart the loader
>    wrote `d` as DOE_k's `gap_after` *and* as DOE_(k+1)'s `gap_before`, while
>    the chain transports `gap_before + gap_after` per entry. Measured on a
>    synthetic `lensA -20mm- DGRATING -10mm- DGRATING -7mm- lensB`: the
>    documented drop-in transported **47.000 mm** across a **37.000 mm** file,
>    moving the chief ray at the target from -1369.311 um to -1802.549 um
>    (**433 um, 31.6%**) and giving a different field at order 0
>    (`max|dE|/max|E|` = 2.13). Nothing warned. Design 121's crossed pair sits
>    at `DISZ 0.0`, so its inter-DOE leg is zero and the double count is
>    invisible — as it was to every test here, all of which used `DISZ 0.0`
>    too. **Fixed** in `_collect_diffractives`: `gap_after` is 0 when the next
>    optical element is another DGRATING, whose `gap_before` carries the leg.
>    Each axial leg is now recorded exactly once; re-measured, the drop-in
>    transports 37.000 mm, the chief ray matches the hand-typed chain
>    **exactly**, and the order-0 field is **bitwise** the hand-folded gap.
>    The one join the library still cannot own is to the neighbouring *lens
>    groups* — a group's own `gap_before` is indistinguishable from the DOE's
>    `gap_after` — so "give the group after the last DOE `gap_before=0`" is
>    now stated in the loader docstring, the per-DGRATING warning and the
>    chain's `groups` docstring.
> 2. **A DGRATING outside the glass span was silently dropped.** The
>    lens-window auto-detect built its `active` list from glass/mirror
>    surfaces only; a DGRATING is an air-to-air flat, so any DOE outside the
>    glass span was discarded *before* the diffractive collector ran.
>    Measured: `collimated -> DGRATING -> singlet` and
>    `singlet -> DGRATING -> image` — both ordinary fan-out layouts — returned
>    `len(rx['diffractives']) == 0` with **zero** warnings, i.e. exactly the
>    "the prescription the chain sees has never contained the DOE" state P2
>    exists to end, now silent instead of warned. **Fixed**: the auto-detect
>    counts a DGRATING that carries usable grating data as active (a DGRATING
>    with no `PARM 1` still does not, so it stays the plain flat plane it is).
>    The `+1` window extension is now conditioned on the last active surface
>    carrying glass, which is the same test as before for every
>    non-diffractive file. An explicit `surface_range` that excludes a
>    DGRATING is still honoured, but now warns and names the surfaces.
> 3. **"An order-0 DOE is bitwise inert" was float-association luck.** The
>    chain accumulated `(gb1+ga1) + (gb2+ga2)` while the hand fold it replaces
>    is the axial-order `gb1+ga1+gb2+ga2`; float addition is not associative,
>    so gaps 0.02/0.0/0.01/0.007 folded to 0.037 but accumulated to
>    0.037000000000000005 — one ulp — and the "bitwise" claim became
>    1.44e-7. Design 121's own gaps happen to re-associate exactly
>    (0.0515393280925041 + 0 + 0 + 0.007 == 0.058539328092504096), which is
>    why the acceptance still reproduced. **Fixed**: the chain accumulates the
>    deferred gaps one leg at a time, left to right, which is bit-identical to
>    the axial-order fold for *any* gaps. Worth recording alongside it: one
>    ulp on a 37 mm gap (6.9e-18 m, a 3e-11 rad phase) is not below the noise
>    — the traced pipeline's roundoff floor is ~1e-7 relative and a few ulp
>    reach it (+1 ulp measured 6.5e-11 on one relay and 1.4e-7 on another,
>    +10 ulp 8.1e-8). That floor is *why* bitwise is the property worth
>    having: it is the only claim that separates "provably unchanged" from
>    "unchanged as far as this pipeline can resolve".
>
> The shipped 121 acceptance was re-run after all three fixes and has not
> moved: the 8-group chain with both DOE entries at order 0 is still
> `max|dE|/max|E|` = **0.000e+00**, `np.array_equal` True, against the
> hand-folded arrangement (N=1024, dx0 = 2.0 um, `final_leg='auto'`), and the
> 121 import is unchanged (23 elements, `gap_before` 51.53932809250411 mm /
> `gap_after` 0.0 and 0.0 / 7.0 mm, period 113.76564277588169 um, orders
> -4/-2, azimuths 0/270 deg).

Related, already fixed and worth a consumer note: `bba1bc4` closed a
`create_periodic_phase_mask` cell-lookup bug (`clip` where a modulo was needed)
that put a full 0<->pi flip on 2816/65536 mask pixels in the measured case,
leaking 11.3% of power off the order lattice with 2.95% into nominally
forbidden even orders. **Any DOE fan built through that helper before v5.30
should be re-measured before its non-uniformity is attributed to physics.**

---

## P3 — Refuse (or shout about) multi-congruence input

The highest-value guard here, because the failure mode is a *plausible-looking
wrong answer*: at v5.28 the 32-order fan went through the chain and produced a
populated, credible-looking frame lattice whose per-frame power was scrambled
(0.47 +/- 0.51% vs a design 2.78%). Nothing raised, nothing warned.

The library already has the threshold — residual transverse angular spread
above **~0.02 rad** is documented as invalidating the carrier-referenced traced
correction (`_lens_traced.py:1032-1038`). A +-46 mrad fan is ~2.3x outside it on
a single order and worse on the fan rms.

**Requested:** have `propagate_traced_carrier_chain` measure the input's
angular spread / multi-valuedness at entry and, above the documented envelope,
either raise or emit a `RuntimeWarning` that names the multi-congruence route
(P1). Note the measurement caveat already recorded in the wavefront-aware
audit: a wrapped nearest-neighbour gradient estimator **under-reports** when the
content aliases, so the detector should not be built on the aliased gradient
alone. `apply_real_lens_universal` already does multi-valuedness routing
(`fga.py:2859-2878`) — the chain could reuse that measurement.

> **MEASURED NOTE (niche D3, 2026-07-28).** The request and its acceptance are
> unchanged and were met by reusing `fga._tilt_dispersion` as asked. But the
> caveat quoted above is only *half* of that estimator's grid dependence, and
> the other half lands directly on P4. `_tilt_dispersion`'s raw reading also
> **under-reports as the grid gets FINER** — it falls as `sqrt(dx)`, because two
> equal beams crossing at ±θ superpose to a *real* cosine times one carrier, so
> the wrapped increment is ±π only across the amplitude nulls, whose weight
> falls as `(dx/fringe)²`. Measured on design 121's own 8x4 fan at a fixed
> 2.048 mm window: raw 2.97e-2 / 2.28e-2 / 1.66e-2 / 1.19e-2 / **8.36e-3** /
> 5.92e-3 rad at dx0 = 4 / 2 / 1 / 0.5 / **0.25** / 0.125 um — i.e. the naive
> gate was **silent at dx0 = 0.25 um / N = 8192**, the exact production row P4
> below names, while the multiplexed answer stays 36-86% wrong at every pitch.
> The shipped gate multiplies by `sqrt(lambda/dx)`, which cancels the law
> analytically and leaves a score `~3.5 theta^1.5` that depends only on the
> crossing angle: 1.70e-2 - 1.92e-2 rad flat across that whole sweep, firing at
> every pitch. Two consequences worth carrying: (i) any *other* detector built
> on a wrapped nearest-neighbour gradient inherits the same `sqrt(dx)` law and
> must be canonicalised the same way; (ii) the gate's floor is now stated in
> ANGLE (~19 mrad between interfering pairs, i.e. the nearest-neighbour order
> spacing for a dense fan, not its total span) rather than in grid pitch — an
> 8x8 fan spanning ±23 mrad sits on that boundary and is not reliably caught.

---

## P4 — Close the validation envelope where production actually runs

* **dx-flatness is published only over N = 1024-4096** (dx0 2.0 -> 0.5 um). The
  original F-B evidence matrix's *worst* row — dx0 = 0.25 um / N = 8192, which
  read EE6 46.5% pre-flip — has never been re-published under the shipped
  defaults. Closing that one row would make the convergence claim airtight
  against its own original counter-evidence.
* **Nothing above N=4096 has been re-validated post-flip at all**, while the
  pre-flip N=28672 runs are what produced both the >100% energy reading (F-A)
  and the divergence. If large-N is intended to be supported, it needs a row;
  if it is not, saying so explicitly would be just as useful — the consumer
  lesson from this study is that grid size stopped being the accuracy lever, and
  that is worth stating where people will read it.
* **No CI gate asserts design-121 dx-flatness** (the `.zmx` can't ship). The
  existing `self_check='dx'` gate runs on one synthetic N=512 singlet with a
  single sqrt(2) step at 5% tolerance on window-power/peak/r50 — it would catch
  a gross regression but, by my estimate, not a subtle one, and it does not
  cover EE3/EE6/FWHM. A synthetic multi-group stand-in with the 121's NA
  progression would close this without shipping the prescription.

> **MEASURED NOTE (niche D5, 2026-07-29).** All three bullets closed; full
> record in `AUDIT_TRACED_PRODUCTION_READINESS_2026_07_24.md` §0.
>
> 1. **The worst row is closed.** `N = 8192 / dx0 = 0.25 um` re-run under the
>    shipped defaults (no chain kwargs), ray pitch held at the reference 4 um
>    (`ray_subsample=16`), NFC 8192 / WF 4.0: best focus **FWHM 3.4265 um /
>    EE3 88.832 % / EE6 99.580 % / EE12 99.762 %**, on-axis, at-plane window
>    **99.796 %**, per-stage power constant to 6 digits. That is **identical
>    in every digit to the N = 4096 row** (3.4266 / 88.832 / 99.580 / 99.762)
>    and in three to the N = 2048 acceptance (3.4156 / 88.829 / 99.583). The
>    row read EE6 **46.5 %** / FWHM 8.85 um pre-flip, so the convergence claim
>    is now airtight against its own original counter-evidence. Measured
>    systematic across 2048 -> 8192: **0.001 EE6 points per octave** (the F-B
>    era figure was ~15).
> 2. **Large N: stated as NOT supported, on measurement.** The limit is
>    **cost, not accuracy**. Chain wall time 82 / 104 / 191 / 556 s at
>    N = 1024 / 2048 / 4096 / 8192 (~2.9x per octave, 24-thread box); at
>    N = 16384 the **first two of eight groups alone exceeded 600 s**, i.e.
>    > 40 min for a 4-digit-converged answer. Memory scales the same way:
>    one complex128 grid is 1.0 / 4.0 / 12.25 GiB at N = 8192 / 16384 /
>    28672 and the chain holds several. The accuracy return is zero
>    (bullet 1).
>    The fine-grid failure that motivated the worry is gone: the F-A trigger
>    `n_crop > n_fine_cap` (130.8 % window at N = 28672 pre-fix) was
>    re-created cheaply at N = 2048 with cap 512 / 256 and now reads window
>    **99.747 % / 98.832 %** with the F-D warning naming the discarded NA.
>    The position is written where a consumer reads it — the
>    `wavelength, dx` entry of `propagate_traced_carrier_chain`'s docstring —
>    with the one caveat that large N is only ever meaningful
>    PITCH-PRESERVING.
> 3. **CI gate shipped:** `tests/unit/test_niche_d5_dx_flatness_gate.py`
>    (13 tests, measured **137 s / 3.16 GiB peak RSS** on Windows and
>    167 s / 3.34 GiB on the WSL Linux proxy, no `.zmx`, no data file, every
>    chain-running test RAM-guarded at 4 GiB available). A four-group
>    synthetic with the 121's structure — small-waist diverging launch,
>    collimate, focus, re-collimate, fast final group; per-group exit NA
>    0.000 / 0.078 / 0.000 / 0.189, final-leg `na_exit` **0.2021** so the
>    SAME exact high-NA leg runs — and, crucially, a **CORRECTED RELAY**:
>    G3's flat exit carries an `r^4` pre-shaping term (2.0 rad at r = w) that
>    G4's flat entrance compensates 2 mm downstream, so the chain must HAND A
>    RESIDUAL ACROSS A GROUP BOUNDARY exactly as design 121 does (S12: 9.2 rad
>    on the 121 final leg). The correction is proved in CI by an exact
>    meridional raytrace through the actual conic + aspheric sags (transverse
>    error <= 0.102 um out to 1.25 launch radii against a 2.743 um oracle
>    FWHM; removing G4's compensating term alone moves the same rays to
>    +10.15 um). Gate = dx-flatness on FWHM/EE across N = 512 / 768 / 1024 at
>    pitch-preserving rs 2/3/4 (measured spread FWHM 0.0014 %, EE <= 0.011
>    pts, window 0.007 pts) **plus an ABSOLUTE level** against
>    `validation/oracles/debye_oracle_v3.py`, the lumenairy-FREE oracle
>    already in the tree — it shares no readout, no `window_factor`, no FFT
>    grid and no wave model with the propagator under test. Measured level:
>    FWHM +0.437 % of the oracle (tolerance 2 %), EE2 0.790x and EE4 0.867x
>    (tolerances 0.70 / 0.80), window 99.883 % of the LAUNCHED power
>    (tolerance 99.0). Teeth verified from the SAME gate function: 4
>    violations each for `carrier_reference='parabola'`,
>    `preserve_input_phase=True`, `preserve_input_phase=False`,
>    `final_leg='paraxial'` and the full legacy configuration; none for the
>    shipped defaults.
>
> **CORRECTION (2026-07-29, adversarial re-measure).** The first revision of
> bullet 3 shipped a gate with a *stigmatic-groups* stand-in and a level
> "oracle" built from a perfect sphere pushed through the LIBRARY'S OWN
> `carrier_referenced_exact_focus_readout`. Three things were wrong and are
> now fixed at root; full record in
> `AUDIT_TRACED_PRODUCTION_READINESS_2026_07_24.md` S0.3.
>
> * **It had ONE tooth.** `preserve_input_phase=True` and `False` both passed
>   with an EMPTY violation list (0.03 / 0.45 EE2 points) — while the same two
>   flips on the real design 121 cost **35.7 / 33.1 EE3 points**. The
>   mechanism is in the element docstring: "for a pure carrier-sphere input
>   the residual is ~0 and 'remap' coincides with False", so a stand-in whose
>   groups are each individually stigmatic **structurally cannot** test that
>   leg of the v5.29 triple. The corrected relay above fixes it: the same two
>   flips now cost 23 % and 52 % of the FWHM and 10 / 24 EE2 points.
> * **The level half was three RATIOS to a reference sharing the chain's last
>   stage**, so any defect common to both cancelled: `window_factor` 4.0 ->
>   3.0 broadened the chain's FWHM by 6.1 % *and the reference in lockstep*,
>   and the gate returned zero violations. It is now an independent absolute
>   anchor (above), with a two-sided FWHM tolerance.
> * **The stated memory was wrong by 140x** — "peak single array 2048^2
>   complex128 = 64 MiB" against a measured 8.9 GiB peak RSS, with no RAM
>   guard on a file that CI runs on 7 GiB runners. The rebuilt file measures
>   3.16 GiB and guards every chain-running test.
>
> **Three findings, all pinned.** (i) **dx-flatness alone is not a sufficient
> gate** — `carrier_reference='parabola'` is dx-FLAT to 0.005 % on the
> stand-in while sitting **3.67x** wide of the independent oracle (FWHM
> 10.062 vs 2.743 um, EE2 6.19 vs 80.90 %, window 75.08 vs 99.88 %), so a
> flatness-only gate — which is all `self_check='dx'` is — passes it silently.
> The ABSOLUTE level half carries the teeth. (ii) **`remap_sampling` has no
> teeth on this stand-in and the reason is quantitative** (S12's `r_alias` is
> 3.7w at the carried residual this geometry can support); recorded on its own
> test rather than assumed covered. (iii) **A NEW library finding:** the
> aperture:beam ray-FIT cliff is not closed at its shipped chain default when
> the final group is fast AND its input is COLLIMATED (so no carrier engages).
> On one stigmatic conic singlet at exit NA 0.20 the traced exit wavefront is
> **1.122 rad** wrong at r=w with `fit_radius_beam_factor=2.0` (the chain
> default) and 0.087 rad at 1.5, while `apply_real_lens_gbd` reproduces the
> exact Fermat-spherical exit wavefront to 0.031 rad. It costs ~15 % of the
> peak intensity and ~11 EE2 points end-to-end, is **dx-INDEPENDENT**, and
> leaves the exit AMPLITUDE right to 0.1 % — so neither a flatness gate nor an
> energy check can see it. Design 121 is not obviously exposed (its final
> group is fed non-collimated, so a carrier does engage), but "fast final
> group after a collimated space" is a common architecture; see
> `AUDIT_TRACED_PRODUCTION_READINESS_2026_07_24.md` S0.4 and the P8 row below.
>
> Tolerances were calibrated on a real second platform (WSL Ubuntu-24.04,
> CPython 3.12.3 / numpy 2.4.6 / scipy-openblas 0.3.31, vs Windows CPython
> 3.14.6 / numpy 2.4.4): agreement <= 4e-6 um on FWHM and <= 5e-5 points on
> EE across the whole ladder, ~200x below the dx spread the gate tolerates,
> and the oracle itself is identical to every printed digit.

---

## P5 — Guard rails on the two silent-degradation paths

* **`na_exact_threshold = 0.15` vs design 121's exit NA 0.152** — a **1.3%
  margin** (`carrier.py:2629`). `final_leg='auto'` routes correctly today, but
  one beam-size change drops this design silently onto the paraxial readout,
  which is ~200 rad wrong at this NA. Requested: warn when the exit NA lands
  within ~20% of the threshold, so the near-miss is visible. (Consumer-side
  mitigation is `final_leg='exact'` explicitly, which the 121 runners should
  do regardless — but the trap is set for the next design.)

  > **CORRECTION (niche D3, measured 2026-07-28).** The "0.152 / 1.3% margin"
  > premise above is **wrong**, and the requested guard was built anyway
  > because the *mechanism* is real. `final_leg='auto'` does not branch on the
  > geometric aperture/EFL system NA (0.152); it branches on
  > `na_exit = _envelope_amp_radius(last group entrance) / |R_out|`. Measured
  > on the real `.zmx`-backed 121 chain (N=2048, dx0=1.0 um, shipped
  > defaults): **`na_exit` = 0.4053** (w = 3.126 mm over R_out = -7.712 mm on
  > `Lens S25-S27`) — **170% ABOVE** the 0.15 default, not 1.3% below it, and
  > it would take a ~2.7x beam-radius shrink to flip. This agrees with
  > `AUDIT_TRACED_CARRIER_CHAIN_2026_07_21.md:88`, which already recorded the
  > last leg at "NA ~ 0.46 (R_out = -7.71 mm, w = 3.53 mm)". So **the P5a trap
  > is not armed on design 121** with the shipped configuration. The guard
  > ships regardless, and `TracedCarrierChainResult.stages[last]['na_exit']`
  > now reports the branch quantity directly so any design's margin is visible
  > without catching a warning.
* **RAM-capped readout** currently emits a `RuntimeWarning` and continues with
  a metric computed on a degraded grid, correctly labelled
  "RESOLUTION-LIMITED (non-converged)". For unattended/batch production runs
  that warning is easy to lose. Requested: an `on_ram_cap='error'` option so a
  production run fails loudly rather than reporting a degraded number.
* **`rs_fine` clamp degenerate corner** — when the memory/Nyquist-capped
  `dx_fine` is coarser than `ray_subsample * cur_dx`, the rescale clamps to
  `rs_fine = 1` and the F-C pitch-preservation contract silently stops holding
  (measured 5.25x mismatch at the N=28672 / `n_fine_cap`=16384 121 condition).
  Warn-only today; an opt-in strict mode would be better for production.

---

## P6 — Doc hygiene (cheap, and it actively misleads today)

`AUDIT_WAVEFRONT_AWARE_RAY_LAUNCH_2026_07_23.md` still carries **"F-B
(dx-scaling audit, still open)"** at `:89` and "Left open" at `:191`. F-B was
in fact root-caused and closed by the frozen-amplitude work
(`AUDIT_TRACED_FROZEN_AMPLITUDE_2026_07_24.md` §2-§4, §6.7) plus the
`0a743a6` upsample-lattice fix — one day *after* that audit was written. It was
never closed under the label "F-B", and `AUDIT_TRACED_CHAIN_DX_SCALING_2026_07_22.md`
was never amended, so both docs still read as open. Its §"Validity map" also
still says re-attributing the DOE fan scramble "needs the F-B fix first" — that
prerequisite is now met, and the re-attribution lands on P1/P2 instead.

Also stale, same class: `Migration-Guide.md:770-775` says the P5 return-contract
transition "remains scheduled for v5.32" (it executed in v5.30);
`carrier.py:2856-2861` calls `remap_sampling='full'` "opt-in only for
byte-compatibility" (it is a chain default since `a9dc454`); and
`validation/repro_traced_carrier_121/focus_scan_121.py`'s footer prints the
pre-S12 acceptance (3.550 / 88.4 / 99.3) as the shipping-default line.

---

## P7 — Known-and-accepted, listed so it isn't re-discovered

* **Inter-group transport is still paraxial** (Sziklas-Siegman). Under
  `'sphere'` the (S - parabola) term rides inside the transported envelope
  (~7 rad at r=w on the 121 final gap); measured end-to-end cost <= 0.2 EE3
  points, core agreement 0.019 rad rms vs an exact-ASM prototype. Fine for 121
  — but there is **no high-NA-gap guard**, so the next design finds the edge
  the hard way.
* `window_factor` is consumed twice on the exact path and compounds below
  wf ~ 3 (at wf=2, 8.6 EE3 points lost). Keeping the 121 acceptance's wf=4.0 is
  fine; a note at the call site would save the next consumer the bisection.

---

## P8 — NEW (raised by D5, 2026-07-29): the ray-FIT cliff at a fast, CARRIER-FREE final group

Not a consumer request from the original note — a defect surfaced while
building P4's CI gate, recorded here because it is the only open accuracy
item this study has that design 121 does *not* already exercise.

`fit_radius_beam_factor` (the P2 aperture:beam cliff guard, chain default
**2.0**) is **not sufficient** when a group is fast AND its input is
COLLIMATED, because then no carrier engages (a collimated `carrier=inf`
eikonal is NaN, so the R7 carrier-gated fit restriction never runs) and the
OPL fit has to represent the whole exit sphere rather than a small residual.

Measured on ONE plano-convex conic singlet, `K = -n^2`, f = 4.83 mm,
collimated Gaussian in, exit NA 0.20 — stigmatic, so the exact exit wavefront
is a sphere by Fermat and needs no diffraction model:

| route | exit-wavefront error at r = w |
|---|---|
| `apply_real_lens_traced`, no fit restriction | 4.428 rad |
| ... `fit_radius_beam_factor=2.0` (chain default) | **1.122 rad** |
| ... `fit_radius_beam_factor=1.5` | 0.087 rad |
| `apply_real_lens_gbd` (independent propagator) | 0.031 rad |

End-to-end on the D5 stand-in it costs ~15 % of the peak intensity and ~11
EE2 points (chain 63.88 % vs an independent-oracle 80.90 %) as a HALO between
r = 3 and 7 um, at a FWHM cost of only +0.44 %. It is **dx-independent**
(1.240 / 1.238 / 1.233 rad at N = 1024 / 2048 / 4096, unchanged at
`ray_subsample` 1 vs 4 and at `n_fine_cap` 2048 / 4096 / 8192) and leaves the
exit AMPLITUDE correct to 0.1 %, so neither a dx-flatness gate nor an energy
check can detect it. It scales ~NA^5 (1.238 / 0.276 / 0.050 / 0.031 rad at
exit NA 0.20 / 0.15 / 0.10 / 0.05), i.e. it switches on essentially where
`na_exact_threshold` = 0.15 does.

**Design 121 is not obviously exposed** (its final group is fed
non-collimated, so a carrier engages and the fit is referenced to a small
residual, consistent with the acceptance sitting at its measured ideal-field
ceiling) — but "a fast final group after a collimated space" is the most
common free-space relay architecture, so the next design finds this the hard
way.

**Requested:** either auto-tighten the fit disc when no carrier engages and
the exit NA is above `na_exact_threshold`, or warn naming
`fit_radius_beam_factor`; and correct the `fit_radius_beam_factor` docstring's
"the recovery is flat for 1.5-2.5", which is an E4-case statement (here 2.0
leaves 1.12 rad where 1.5 leaves 0.087). Consumer-side mitigation today:
`traced_kwargs={'fit_radius_beam_factor': 1.5}` on such a group, cross-checked
against `apply_real_lens_gbd`. Pinned by
`tests/unit/test_niche_d5_dx_flatness_gate.py::test_the_level_gap_is_the_traced_fit_radius_cliff`;
full record in `AUDIT_TRACED_PRODUCTION_READINESS_2026_07_24.md` S0.4.

---

## Summary

| # | Item | Blocks | Size |
|---|------|--------|------|
| P1 | Per-congruence chain (tilted carrier and/or multi orchestrator) | **the full 121 config: DOE fan AND emitter array** | large |
| P2 | DGRATING import + DOE as a chain group | manual, error-prone bookkeeping | medium |
| P3 | Multi-congruence refusal/warning at chain entry | silent wrong answers | small |
| P4 | dx-flat row at N=8192 + a 121-class CI gate | confidence at production pitch | small-medium — **DONE** (D5) |
| P5 | NA-threshold proximity warning, `on_ram_cap='error'`, strict `rs_fine` | silent degradation in batch runs | small |
| P6 | Stale F-B / migration / footer text | actively misleads readers today | trivial |
| P8 | Ray-FIT cliff at a fast CARRIER-FREE group (`fit_radius_beam_factor`) | ~11 EE2 points, dx-invisible, on the commonest relay architecture | small-medium — **NEW** (D5) |

P1 and P3 together are what turn "design 121 runs" into "design 121 runs
*smoothly, with the DOE, and tells you when it can't*". P3 is small and would
have saved this study a full experiment cycle.

---

## Niche C2 (2026-07-30) — claim corrections and disclosure pins

C1 closed the five *code* findings the D1–D7 verifiers left. C2 closes the
*claim* findings: places where the shipped text asserted more than the
measurement supported. Every item below was re-measured before being rewritten,
and two of the original claims turned out to be **wrong**, not merely loose.

| # | Claim | Verdict |
|---|---|---|
| 1 | "A deflected congruence makes the exact high-NA final leg raise" | **STALE** — D6 removed that refusal; corrected |
| 2 | `_FIT_DISC_OUTSIDE_WEIGHT_REL` plateau 1e-14..1e-8 | **NARROWED** — evidence is low-NA/small-beam only |
| 3 | Off-centre branch is "strictly a superset of the mask" | already retired by C1 item 1 |
| 4 | Detector B's score is "set by the finest fringes / nearest-neighbour order spacing" | **REFUTED** — it tracks the total SPAN |
| 5 | "`gap_before` is always the true distance" | **FALSE** for a grating on glass — now flagged + refused |
| 6 | "the decentred figure sits BELOW the on-axis one" | **TRUE ONLY UNTILTED** — 1.4x above the tilted control |
| 7 | D7's raised fit order | **can go inert silently**; boundary documented |
| 8 | The order-vs-residual calibration table | ray grid **pinned** (`ray_subsample=4`) |

**Item 4 — the substantive refutation.** The multi-congruence detector's
documented envelope claimed a dense fan hides far below its span, because "the
score is set by the finest fringes". Re-measured with the shipped helper (the
harness reproduces the 8×8 row to 3 digits, so it is the same measurement):

| construction | canonical rad, dx0 = 4 / 2 / 1 µm | equivalent PAIR |
|---|---|---|
| 8×8 fan, span ±23 mrad, NN 6.571 | 7.83e-3 / 8.41e-3 / 8.93e-3 | 17.1–18.7 mrad |
| PAIR at that NN spacing, ±3.286 | 7.37e-4 / 6.79e-4 / 6.46e-4 | 3.2–3.5 mrad |
| PAIR at that span, ±23.0 | 1.19e-2 / 1.22e-2 / 1.22e-2 | 22.5–23.0 mrad |

The fan reads **5.3× above** the nearest-neighbour rule and 0.8× of an
equal-span pair, and densifying at fixed span moves the score **down**
(4 / 8 / 16 orders across ±23 read like 16.7 / 14.2 / 12.8 mrad) — the opposite
of the old rule's direction. Corrected rule: **score a fan by its total span,
derated ~20 %**. The old wording was over-conservative rather than unsafe, and
both concrete verdicts it reported still stand.

**Item 2 — why the envelope could not simply be widened.** The plateau was
measured entirely at aperture:beam = 30:1 and low NA. Extending it to design
121's regime (3.26:1, exit NA 0.405) is not merely a different answer — the
oracle fails first. On 121's last group at N=1024 / dx = 33.211 µm the exit NA
is 0.356 against a grid Nyquist direction cosine of 0.0197 (**18× short**), and
`newton_fit='spline'` — the fit-domain-free reference the note leans on — fails
to converge for **100.0 %** of 65536 pixels and returns an **all-zero field**
(the polynomial path fails for 81.4 % and still returns a usable one). Under
`on_undersample='silent'` that zero is returned without a word. The envelope is
therefore narrowed and labelled, not extended. **The shipped 121 chain is
unaffected** — it re-traces the final leg at `n_fine_cap` 12288 and its
acceptance is unchanged (3.450 µm / EE3 88.8 / EE6 99.6 / EE12 99.8).

**Item 5 — the only behaviour change.** `gap_before` / `gap_after` are raw
axial thicknesses and the chain transports them through **air**, so a grating
ruled on a substrate would be placed at the wrong optical distance (`t - t/n`
per glass leg; 1.0 mm for a 3 mm N-BK7 plate) with no symptom. The importer now
records a `gap_media` marker and warns; the refusal itself lives at the **point
of use** (`_normalise_doe_entry`), because `load_zemax_zmx` serves far more than
the DOE drop-in and the rest of such a file imports correctly. Design 121 is
untouched (both gaps free space; markers absent, gaps still 51.5393 / 7.0000 mm).
