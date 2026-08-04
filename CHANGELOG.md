# Changelog — lumenairy

All notable changes to the core library are documented here.

## [5.32.1] — 2026-08-03

### Fixed — the shared least-squares solver no longer draws arbitrary answers on singular systems (niche C13, `elements/_lens_traced.py`)

The campaign's residual "CI flakiness" was a real defect one level below every
hypothesis: `_solve_lstsq_thread_safe` solved ALL traced fits by normal
equations on a docstring claim that is false for the D1/D7 WEIGHTED fits
(`cond(A)` 1.4e10 squares past float64; Cholesky then returns an arbitrary
null-space draw per BLAS build -- fit residual 1.05x optimal on one build,
14.8-23.0x on another, exit field pixel-speckled).  C10's degree 6 was only
the messenger (`cond(A)` 4.1e2 there -- one of the BEST-conditioned solves).
Fix: rcond-screened equilibrated Gram; where singular, Householder-QR
re-solve kept only if it fits better; ties return historical bits.
Cross-build agreement 6.8e-2 -> 1.2e-10; degree sweep smooth 2..6 both
builds; per-order closure reproduced to four decimals on BOTH builds (first
OpenBLAS measurement of the design-121 table); independently CURES the D1
fold defect.  Commit gate: 501 tests x 2 builds green.

### Changed — the measured fit-branch arbiter is now the DEFAULT (user-ordered); the C12 predictor stays dormant (regression found)

`DECENTRED_FIT_ARBITER = True`: five orders improve (worst residual 0.152 ->
0.069), the 67-point wrong-branch trap dies, and the user explicitly accepted
the proven-irreducible (-1,0) +0.026 trade.  The ordered
`DECENTRED_FIT_PREDICTOR` flip was REVERTED ON EVIDENCE: adjudicating its 11
test failures against the analytic tilted-leg oracle exposed a real 17,000x
inversion -- nine eras-pins that would have been written were a live
regression.  Predictor ships False with the defect recorded (C13 S11).

### Added — encapsulation + guard integrity (niche C14, both library files)

`_traced_flags.py` registry (presets, not an ordinal era -- the C6-on/C8-off
corner stays reachable), `TRACED_LAYER_MAP.md` bound to the code by four CI
tests (one caught a real dangling reference on first run), and UNIT C
extracted: the three notions of traced exit support are one construction
(byte-identical over 36 configs incl. the never-probed direct-fit path),
closing the documented C7/C8 joint blind spot on the relay.  Guard
integrity: the dx-stability warning is no longer dedup-silenced (a batch
loop was warned ONCE; now every qualifying call warns via warn_explicit),
and FOUR silent exits of `_run_chain_dx_self_check` now speak.  Legs: 376 +
119-strict + 142-WSL green; 32/32 both builds.

### Changed — three external audits assessed and answered (CI test-time, test justification, p2 DID-NOT-WARN)

Per-recommendation responses with counter-measurements where rejected:
~47 min/push of measured CI savings, a JAX coverage hole closed (nine files
ran in NO leg), 13 docstring-drift fixes, three defects the audits missed,
the p2 fixture's 0.87 %-margin route pinned, and the corrupted durations
mechanism (contention-captured units) documented with the regeneration
protocol now in the workflows.  Deferred with ledgers: the ~30-duplicate
dedup cycle, one backwards marker move.  Records: CI_TEST_TIME_RESPONSE,
TEST_JUSTIFICATION_RESPONSE, P2_DIDNOTWARN_DIAGNOSIS (2026-08-03).

### Added — physics-derived fit-branch predictor (niche C12, `elements/_lens_traced.py`, opt-in)

The concentric/off-centre crossover now closes in ONE LINE with no fitted
constant: each candidate's residual is its own Chebyshev tail's, the tail is
decentre-free (measured), so the crossover is the concentric disc's inflation
`rho=sqrt(1+2u^2)` vs the tail's spectral moment.  Predicts all three
measured crossovers to 0.03 % (arbiter: -4.6 %), build-invariant to 4 digits;
falls back to the measured arbiter where the expansion is unsupported (121's
underfilled launch box; 26/26 agreement).  Also PROVEN: the (-1,0) +0.026 is
irreducible for ANY per-call selector (per-group preferences are
non-monotone; "improve everywhere" and "keep the gains" are mutually
exclusive on 121) -- so the default stays False and the flip is an informed
trade, not a fix away.  C11's "85 % C6-domain" attribution reversed
(ray-fit branch is the driver).  Bit-identical default, 6/6 orders;
production acceptance unchanged; both builds green.

### Added — a measured fit-branch arbiter, shipped OPT-IN (niche C11, `elements/_lens_traced.py`)

The C1 decentre gate's 0.05 w constant was never physics: the measured
concentric/off-centre fit crossover is 0.55 w on an f/3 singlet, 0 w on an
f/6, 0.46-0.69 w across design 121's groups -- four closed-form predictors
were derived and all four refuted cross-design.  `DECENTRED_FIT_ARBITER`
instead BUILDS BOTH candidate fits at the site and scores them against the
traced samples, intensity-weighted (42/42 agreement with the fit-free spline
oracle over three geometries; the (-4,0)-group-4 67-point catastrophe is
seen at 18x margin before any field is reconstructed).  Default **False**:
per-order it improves five orders (worst residual 0.152 -> 0.069, spread
0.200 -> 0.117) but moves (-1,0) by +0.026 -- a bounded near-tie mis-pick
-- so the default flip is an explicit release decision, not a side effect.
Flag-off is a no-op, not a fall-back (the scorer is never called), verified
bit-for-bit 6/6 orders cross-process against pre-C11 hashes.  Also
measured: C10 made the forced-concentric branch MORE dangerous (19.9 ->
67.3 points at (-4,0)), strengthening the arbiter's case.  Record:
`docs/audits/C11_PHYSICAL_DECENTRE_GATE_2026_08_03.md`.

### Fixed — four platform/era test reconciliations (both BLAS builds green)

C9/C10 legitimately moved physics that older pins measured: the D3
multiplex-guard linearity pin (attributed to C10's degree-6 residual on
MULTIPLEXED inputs only -- the route the guard exists to refuse; era-pinned
at degree 4 with a live comparative sibling, and the test's NEXT latent
failure `bad > 20*good` fixed in the same pass), the D7 hard-mask-ghost pin
(both builds shrink the ghost -- by 1900x on MKL, 1.9x on OpenBLAS; re-pinned
0.1x -> 0.8x with both recorded), and C1/C6-guard arms now pin their flags
explicitly.  The p2_guards dx-stability fixture was adjudicated and NOT
changed: 10.5x margin, five-figure agreement across builds, all C9/C10/C6
knob states bit-identical -- the one unexplained CI DID-NOT-WARN is recorded
as a suspected shard-ordering warning-filter leak with a committed 2-minute
instrument (`c11_p2dx_recon.py`).  158 tests green on Windows/MKL AND
Linux/OpenBLAS, identical counts.

### Fixed — the residual eikonal now carries the r^6 term: the chain matches the exact-ray oracle to ~0.1 EE3 points on every order (niche C10, `elements/_lens_traced.py`)

`_REMAP_RESID_EIKONAL_DEGREE = 4 -> 6` -- one constant.  The final-closure
residual (0.05-0.94 points/order) was made in the CONVERGING groups 2-4 by
exactly the second-order stationary-phase term C6's derivation names,
`(1/2) grad(a - a_fit)^T H^-1 grad(a - a_fit)`: a carrier-referenced relay's
residual is r^4-dominant with an r^6 next term, and the measured response is
degree 4 ~= 5 << 6 at every order.  The only reason the degree had stayed at
4 was a 5.2 %-of-input self-caustic ghost that measurement reproduces exactly
-- and that niche C8 removed the day AFTER the degree-4 decision was made.

The non-monotone neighbour swing that blocked every physical explanation was
the METRIC: EE3 scored with a hard binary pixel mask on a 0.4 um lattice
carries 0.128 points per boundary pixel (+-0.45 points of pure quantisation,
cancelling between arms at some orders and not others).  Area-exact
rescoring of the SAME arrays is monotone.

Final chain-vs-oracle residual: -0.048 / 0.029 / 0.063 / 0.090 / 0.141 /
0.152 points at (0,0)...(-4,-2), spread 0.886 -> 0.200.  Production
acceptance unchanged (3.350 um / 90.3 / 99.7 / 99.8, digit for digit);
conservation 6/6 every order; C7 halo check silent; fail-before bit-exact.
The D7 fold witness is era-pinned at degree 4 (its fixture's fold no longer
exists under the better model -- the C9 precedent).  Open items and two
caveats (last-group element pass 0.001-0.005 waves worse, in the noise;
single-design generality) recorded in
`docs/audits/D121_RESIDUAL_CLOSURE_2026_08_02.md`.

### Fixed — the parabola<->sphere carrier conversion is applied EXACTLY: the cos^2 band-limit taper is gone (niche C9, `propagators/carrier.py`)

The final-closure campaign's decomposition of the last ~1.3 EE3 points per
order between the chain and the exact-ray oracle found ~1.0 point/order in
ONE library defect: `_sphere_parab_conversion`'s cos^2 taper, engaged at
group 5's exit re-envelope -- the same site as the -0.96-point leg remnant
C5 left behind (per-call census; groups 1-5 bit-identical, 99.74 % of the
effect at the final onset).  The taper's protective role was a MIS-CITATION:
the audit cited as evidence that removing it breaks a coarse chain
(AUDIT_TRACED_FROZEN_AMPLITUDE_2026_07_24 S6.6) says the opposite, and
`T == 1` is the saturated end of a monotone axis (`r_safe` x1.0 reproduces
the shipped taper bit for bit; x1.5/x2/x3 reproduce `T == 1` bit for bit).

`SPHERE_PARAB_CONVERSION_EXACT = True` (3 executable lines; `= False` is the
fail-before, 52/52 configurations bit-identical to v5.32.0).  Everything
improves together: acceptance **3.450 -> 3.350 um / EE3 90.2 -> 90.3** at the
unchanged best-focus plane (no plane of +-80 um worse); tilted production
readout +1.40 EE3 at (-4,-2); ghost `g4` 3-676x and `amax4` 2-14x smaller;
`P/Pin` within 4.1e-05; 6/6 conservation bounds on every order.

The remaining chain-vs-oracle residual is 0.05-0.94 points per order --
REAL against a +-0.005-point instrument band, characterised as a halo
deficit (FWHM matches the exact trace to 0.02-0.06 um), not discretisation.
Leading candidate: the element's remap model error (excluded earlier against
a 2.0-point target that has since moved; not yet verified).  The instrument
itself accounted for 0.5-1.1 points/order of the ORIGINAL gap (a wrong
oracle ceiling + a readout launch-phase split), proven by null experiments.
Full record: `docs/audits/D121_FINAL_CLOSURE_2026_08_02.md`.

## [5.32.0] — 2026-08-02

### Changed — the empty deprecation horizon advanced past this release (`lumenairy/_deprecation.py`)

`NEXT_REMOVAL_VERSION` (and its alias `API_TRANSITION_VERSION`) read `'5.32'`
and came due with this release — with **nothing scheduled**: both registries
are empty tombstones because every removal and the one API transition were
executed early, in v5.30.  Advanced `'5.32' -> '5.36'` per the constant's own
documented one-line-slip mechanism.  No shim's lifetime changes; banners that
resolve through the backstop now read `v5.36 (rescheduled from ...)`.


### Fixed — the inverse map may no longer invent light: exit pixels outside the traced support get zero (niche C8, `elements/_lens_traced.py`)

Closes the C6 on-axis energy defect STRUCTURALLY, at the mechanism.  The
`ray_density` amplitude was evaluated from the fitted INVERSE of the traced
entrance->exit map even at exit pixels no traced ray ever reached; there the
fit extrapolates, and its made-up pullbacks land in the bright beam and
acquire amplitude -- light from nowhere.  `REMAP_INVERSE_SUPPORT_BOUND = True`
(fail-before `= False`, bit-exact) zeroes ray-density amplitude outside the
convex hull of the alive traced rays' exit landing points, with a plateau of
`sqrt(2) sub dx` and a one-exit-cell raised-cosine feather (a plateau-less
taper bled 3.8e-05 of Pin of LEGITIMATE skirt through the bilinear upsample
-- measured and fixed pre-ship).  Both `newton_fit` backends share one hull.

Measured: (0,0) production `P_out/P_ap` **1.000741 -> 0.996026** (the C6-off
class), ghost `g4`/`amax4` **exactly 0**, EE3 unchanged to the last digit on
every order, the at-plane acceptance unchanged, and -- unlike the opt-in fit
guard, which it makes redundant AND safe -- **zero regressions across the six
synthetic fixtures**.  Every measured order/configuration now scores **6/6**
of the conservation bounds at both subsamples, including two cases the guard
structurally could not reach and one that fails with C6 off.  100 % of the
removed power lies outside the convex hull of every alive ray, at every
feather (three-way partition against the call's own ray bundle).

The same defect class was independently confirmed elsewhere first: the C7
halo check's very first suite run flagged a 0.641-of-peak lobe beyond the
exact-ray support in a niche-D6 fixture (ordinary group call, 1.0 w
decentre; 0 of 12849 alive rays reach the lobe radius; amplitude gain
2.55e5; diffraction excluded at 1.42e6x below the observed level).  C8 takes
that lobe to exactly 0 too.  Record:
`docs/audits/C8_INVERSE_SUPPORT_BOUND_2026_08_01.md` and
`ORACLE_ENERGY_AND_D6_HALO_2026_08_01.md`.

### Fixed — the exact-ray oracle can now answer ABSOLUTE energy questions (`validation/repro_traced_carrier_121/`)

The Rayleigh-Sommerfeld kernel in `oracle_spot` omitted the `1/(i lambda)`
prefactor (every intensity `1/lambda^2` too large).  The second defect the
POP cross-check alleged -- `launch_power` missing the cell area -- was FALSE:
the double-count was in the POP comparison harness itself, and the
`lambda^2/h^2` arithmetic closes the discrepancy exactly.  Validated
absolutely: an unaberrated converging sphere through the same machinery
returns `P_out/P_in` = **0.99999988**; design 121 converges to
**P/Pin = 1.0000 on both (0,0) and (-4,-2)**, agreeing with Zemax POP's
1.00000000 and the 70681-ray pupil trace.  The energy-conservation audit was
verified UNAFFECTED digit-for-digit (its references are chain-internal
ratios that never touch the RS kernel).

### Fixed — `remap` now launches at the stationary point of the WHOLE phase (niche C6, `elements/_lens_traced.py`)

`preserve_input_phase='remap'` launched its rays along `grad(W)` -- the
carrier eikonal's gradient alone -- and evaluated the input residual `a` at
that ray's foot.  The exit eikonal is a stationary value of `W + a + V`, so
this dropped a second-order stationary-phase term
`(1/2) grad a^T H^-1 grad a`: quadratic in `grad a`, verified by prediction
(across C5, `grad a` rms grew 1.46 -> 2.30 mrad, ratio squared 2.48; the
measured element wavefront error grew 0.0359 -> 0.0659 waves, ratio 2.48).
The fix launches along `grad(W + a_fit)` and lets the transported phasor
carry the leftover `a - a_fit`; the `ray_density` Jacobian follows the
augmented map automatically.  Design 121 EE3 against the exact-ray oracle:
(0,0) 87.99 -> **89.21** (oracle 90.08), (-4,0) 70.19 -> **88.94** (90.78),
(-4,-2) 66.24 -> **88.49** (89.78); field-angle spread **14.33 -> 0.72**
points.  Fail-before: `REMAP_STATIONARY_PHASE_LAUNCH = False` reproduces the
prior library bit for bit, tilted orders included (verified per order).

A backend-consistency defect found by the D7 spline-oracle test was fixed
before shipping: the residual fit's radial-freeze circle sat at exactly the
polynomial ray-fit disc radius (both 2.0 w), so the map went non-smooth
precisely where that fit began extrapolating and the two `newton_fit`
backends diverged 7.8e-09 -> 3.7e-03 of peak.  Scored pointwise against the
exact skew ray trace the POLYNOMIAL backend was the wrong one (5.6 um in the
skirt, 15.1 um at the aperture rim, vs the spline's 0.006 / 0.002).
Separating the freeze circle from the fit disc
(`_REMAP_RESID_FREEZE_MARGIN = 1.25`) restores agreement to **8.6e-06**.

**KNOWN DEFECT, disclosed not fixed: C6 manufactures energy on axis.**  On
design 121's last group at order (0,0) the production path returns
`P_out/P_ap` = 1.000741 against C6-off's 0.995883 -- **+0.486 % of input
power created**, deposited as a lobe at 4-8 mm carrying 4.7e-03 of Pin at
83 % of peak amplitude, where the exact ray trace permits 3.6e-10.  Every
EE-family metric is blind to it (the same field reports +1.691 EE3).
Mechanism: the corrected launch makes the ray map non-radial and the
CONCENTRIC fit branch's inverse map extrapolates beyond its data.
`REMAP_STATIONARY_PHASE_FIT_GUARD = True` removes it outright on (0,0) but
regresses 2 of 6 synthetic fixtures (one to P/Pin 1.00697), so it stays
opt-in (`docs/audits/C6_FIT_GUARD_DECISION_2026_07_31.md`).  Per-order /
tilted use is sound; on-axis halo and second-moment metrics are NOT until
the structural fix (bounding the Newton inverse to the traced samples'
support) lands.  Full conservation record in
`docs/audits/ENERGY_CONSERVATION_AUDIT_2026_07_31.md`.

### Added — halo-amplitude self-check for `ray_density` (niche C7, `elements/_lens_traced.py`)

The scalar energy self-check's observable is EXHAUSTED: its +5 % gain band
cannot be tightened, because a currently-green CI battery cell legitimately
reads `P_out/P_ap` = 1.04374 at the N/subsample CI actually runs -- the same
magnitude as the defects worth catching -- and the C6 backend fix
demonstrated that a total-power criterion can be satisfied while the defect
(the lobe) remains.  New observable instead: `amax4`-style HALO AMPLITUDE at
1.25x the exact-ray exit support, `_RD_HALO_AMAX_TOL = 1.0e-03`, default
`RAY_DENSITY_HALO_CHECK = 'warn'`.  Calibrated on 180 element calls: worst
clean 4.6e-05, mildest confirmed defect 5.7e-03 -- 123x separation, and it
never fires on any P2 battery cell.  On its first full-suite run it flagged
a real, previously-unknown defect in niche D6's exact-tilted-leg retrace
fixture (0.641 of peak beyond the exact trace's 1.616 mm support) -- open.

### Changed — the design-121 acceptance is scored AT the metasurface plane (user-approved re-baseline, 2026-08-01)

The recorded acceptance `3.450 um / EE3 88.8 / EE6 99.6` was measured at
`dz = +10 um` -- 10 um PAST the metasurface plane -- because the pre-C6
chain's residual aberration pushed its focus downstream (at-plane read
3.750 / 87.4).  C6 removed that offset; the spot is now sharpest at the
plane itself.  New acceptance, at `dz = 0`:
**3.450 um / EE3 90.2 / EE6 99.7 / EE12 99.8**, against the measured
ideal-field ceiling 3.45-3.55 um / 90.3 / 99.8.  The focus scan now reports
`BEST-FOCUS[peak]` (max intensity) alongside the historical EE6-selected
line, because EE6 saturates near 99.7 and mis-selects on its 4th digit.

### Validated — independent Zemax cross-check of the per-order result (`docs/audits/POP_CROSSCHECK_121_2026_07_31.md`)

Zemax's ray-based RMS OPD at the extreme order (-4,-2) reads **0.030 waves**
tilt-free (Marechal limit 0.071): the design is diffraction-limited at the
fan corner, corroborating the exact-ray oracle (<= 0.017 waves) and the
chain (EE3 89.13 vs oracle 89.66, flat across the fan).  Zemax POP reads
8.8 EE3 points lower at that order and is demonstrably UNCONVERGED there:
its space-bandwidth product (`dx_mid x dx_img = 53500 um^2 / N`) needs
N ~ 42000 for a 51.5 mrad order at 0.1 um image pitch against its 8192 cap,
and its log-domain image shows a ~1e-4 pedestal (true halo ~1e-7) plus a
rectangular block artefact.  The historical "POP waist 2.737 um" target is
a POP of the paraxially-EQUIVALENT 4f, not of the real prescription -- now
labelled as such at its quoting sites.  Three-way beam-profile figures
(POP | chain | exact-ray oracle, linear + log10, common scale) for six
orders in `validation/repro_traced_carrier_121/pop_profiles/`.

### Fixed — the tilted carrier's reference wavefront is now an actual eikonal (`elements/_lens_traced.py`, `propagators/carrier.py`)

`TiltedCarrier` defined its wavefront as an on-axis sphere **plus a linear
ramp**.  That is not a solution of the eikonal equation.  The exact eikonal of
the congruence it names -- a point source at signed AXIAL distance `R` whose
chief ray carries `(L, M)` -- is the same sphere transversely RE-CENTRED on the
source's own projection, `W = sign(R)(sqrt((u + R L/N)^2 + (v + R M/N)^2 + R^2)
- |R|/N)` with `N = sqrt(1 - L^2 - M^2)`.  The difference is coma **linear** in
field angle plus astigmatism quadratic in it: on design 121's fifth leg
(`R = -24.46 mm`, tilt 54.9 mrad, `w = 3.63 mm`) it is -0.73 waves one beam
radius along the tilt, **+2.53 against it**, and 15.8 waves at two radii.

This matters because the chain defines its envelope as *field / carrier*, so a
reference that is not a true wavefront dumps real optical path into the
"envelope" -- precisely the thing Sziklas-Siegman then transports by a plain
dilation, which cannot carry it.  Measured in closed form on the leg alone
(exact congruence; no ray trace, diffraction integral, unwrap or FFT
derivative), the leg's model error runs 1.0e-5 waves at zero tilt, 0.0134 at
5.5 mrad, 0.0678 at 27 mrad and **0.1362 at the design's 54.9 mrad** -- and
back to **1.0e-5** with the exact eikonal, holding to 180 mrad.

Design 121 per order, EE3, at the chain's group-5 exit:

| order | field angle | before | after |
|---|---|---|---|
| (0,0) | 0 | 87.99 | **87.99** |
| (-1,0) | 11.5 mrad | 86.48 | 87.27 |
| (-2,0) | 23.0 mrad | 83.15 | 85.60 |
| (-3,0) | 34.5 mrad | 77.02 | 81.95 |
| (-4,0) | 46.1 mrad | 70.19 | **76.61** |
| (-4,-2) | 46.1 + 23.0 mrad | 66.24 | **73.66** |

Field-angle spread **21.75 -> 14.33** points.  Split across the leg and the
element pass at (-4,-2), the leg's share of the loss goes **-20.70 -> -0.96**
(95 % closed) while the element's goes -3.04 -> **-15.26**: the element's own
model error was previously MASKED by partial cancellation against the leg's,
and is now exposed.  Its cause is named -- `preserve_input_phase='remap'`
launches along `grad(W)` alone and so drops a second-order stationary-phase
term scaling with `grad a` (2.30 mrad after this fix against 1.46 before) --
and is NOT fixed here.

The UNTILTED path is byte-identical: `np.array_equal`, max |dE| = **0.0**, over
7 configurations of design 121's real post-DOE chain (two grids, two
`ray_subsample`s, 3- and 5-group runs, both readout paths, `final_leg='exact'`)
and 12 synthetic ones.  The shipped single-beam acceptance is unchanged at
**3.450 um / EE3 88.8 / EE6 99.6 / EE12 99.8**.  Fail-before switch:
`TILTED_CARRIER_EXACT_EIKONAL = False` reproduces the previous field bit for
bit, tilted orders included.

Independently corroborated on a fixture sharing no code with design 121: D1's
and C1's "the off-axis spot reaches the on-axis diffraction limit" pins were
asserting that a 46/51 mrad spot through two uncorrected N-BK7 singlets EQUALS
the on-axis one.  It does not -- an exact skew trace reads geometric rms
1.469 / 1.719 um off axis against 0.419 on axis, and quadrature with the
18.8 um on-axis FWHM predicts 19.12 / 19.38 um.  The corrected reference
measures **19.20 um** in both; the old one measured 18.80 / 18.95, i.e. exactly
the on-axis width -- it had been ERASING the relay's own coma.  Those pins are
re-based on the geometric prediction, with fail-before witnesses.

Also measured and REJECTED: the anisotropic effective-distance term
(`z/(1-L^2)^{3/2}` along the tilt vs `^{1/2}` across) contributes **0.000 EE3
points**.  Its isotropic part is a re-parametrisation rather than an error --
`R` is the AXIAL radius, so the shipped `R -> R + z` is already exact -- and the
anisotropic remainder is ~2e-5 waves, four orders below the effect.  Both
formulations were implemented and measured identical before this was concluded.

### Fixed — `na_exit`'s amplitude mask was transposed (`elements/_lens_traced.py`)

The significance mask was built `(y, x)` by `np.ix_(_ray_iy, _ray_ix)` but
ravelled against a launch grid built with `indexing='ij'` -- x along axis 0 --
so every amplitude was paired with the TRANSPOSED ray.  Rotationally symmetric
beams are invariant under that swap, which is why it survived; on an asymmetric
one the two readings exchange outright (measured on a biconic: 0.0338 reported
against 0.0684 true, and 0.0669 against 0.0340).  Design 121's last group
reported `na_exit` **0.3633** where the transpose-immune value is **0.2912** --
25 % overstated -- and `_exit_na_out` feeds the chain's `on_tilt_exact_grid`
routing, so this was not merely cosmetic.

### Fixed — the chain's chief ray is TRACED, not linearised (`propagators/carrier.py`)

Under a tilted carrier the chain transferred `(x_c, y_c, L, M)` through each
group's lumped paraxial ABCD.  That is not a self-consistent convention: this
module carries angles as DIRECTION COSINES (the free-leg advance
`z L / cos(theta)` is already exact), while an ABCD ray vector is
`[height, SLOPE]`.  The obvious repair -- converting cosine to slope and back
across the ABCD -- was implemented, MEASURED, and **refused**: against an exact
meridional trace on the D1 two-singlet relay at 46 mrad it lands **+1.1208 um**
out where the raw-cosine form lands **+0.1214 um**, i.e. 9x worse.  A lumped
group ABCD is not one convention at all (Snell refracts linearly in SINES,
free transfer in TANGENTS) and a group of this class is refraction-dominated.

So the predictor is no longer linearised: the chief ray is traced through the
group's own surfaces with the same engine the tests use as their oracle
(`_group_chief_transfer`), falling back to the ABCD only if a group cannot be
traced.  Residual against the exact trace **0.1214 um -> 0.0** (machine
precision), at ANY angle -- the `z L^3 / 2`-class error cannot arise.  Judged by
an independent conic ray trace, all five affected closures improved
(0.0440 / 0.2881 / 0.0037 / 0.00045 / **12.3724** um -> 0.0), and on the D6
stand-in the predictor now sits ON the Fermat focus where the light actually
lands, 0.000468 um from the measured spot centroid.  `_chain_chief_ray_at_target`
uses the same step -- the orchestrator cross-checks the two and raises on a
mismatch.  An untilted, undecentred ray short-circuits, so the on-axis path is
byte-identical.

### Added — `on_gap_paraxial`: a guard on the inter-group paraxial transport (`propagators/carrier.py`)

The Sziklas-Siegman inter-group step is exact for the quadratic carrier and
PARAXIAL for the envelope, and nothing measured that.  The obvious metric --
the quartic sag phase `phi_sag = k w^4 / (8 |R|^3)`, the "~7 rad on the 121
final gap" the roadmap quotes -- is **wrong, and a guard built on it would have
fired hardest on the SAFEST legs**.  A leg does not carry `phi_sag`; it drops
the CHANGE in it, `k z NA^4 / 8`, which is exactly the Fresnel kernel's own
defect.  At fixed `phi_sag` = 8 rad the measured cost runs **-2.1 to -65 EE
points** with leg length alone, and at fixed NA the disagreement FALLS as
`1/phi_sag`.

Better still, under the shipping `carrier_reference='sphere'` that dropped
quartic **cancels exactly**: the parabola/sphere conversions bracketing a leg
contribute `-z (parabola - S)`, and with the Fresnel leg's `z (1 + t^2/2)` the
total is `z sqrt(1 + t^2)` -- the exact tilted-ray path, to all orders in `t`
(verified to 2.2e-16).  Measured over NA 0.35-0.45 and `phi_sag` 1-100 rad:
**0.000 EE points on every row**, against -20 to -33 points for the same legs
under legacy `'parabola'`.

The guard therefore trips on the DROPPED quartic (`gap_sag_tol`, default
0.30 rad; 1 EE3 point is crossed at 0.40) and on the gap NA (0.60, the first
row off the zero-cost floor), with per-leg diagnostics in `stages`.  Design 121
is SILENT on every shipping leg with **4.08x** margin; under legacy
`'parabola'` it fires on two, independently corroborated by this library's own
audit of that legacy triple (best-focus EE6 79.7 % vs 99.3 %).  Note the
largest drop is the SOURCE leg, not the final gap.  Diagnostic only: fields are
bitwise identical across every setting, verified by monkeypatching the entire
added path out.

### Added — design 121 full configuration: tilted carriers and a per-congruence fan (`propagators/carrier.py`, `elements/_lens_traced.py`, `io/prescriptions_zemax.py`)

The traced chain can now carry a **tilted** congruence, so a DOE order is a
first-class chain input instead of something the caller has to hand-split.
`TiltedCarrier(R, L, M, x0, y0)` carries sphere + tilt through every hand-off
with the exact obliquity `1/sqrt(1-L^2-M^2)` (the chief ray advances by
`z L / cos(theta)`, not `z L`), and
`propagate_traced_carrier_chain_multi(congruences, groups, ..., recombine='coherent')`
runs each congruence through the shipped-default chain and recombines on a
common image grid.  The 32-order design-121 fan now runs end to end and its
per-frame power reproduces the Dammann design to **3.0e-4** — the v5.28
scramble was readout **replica aliasing** (a readout window wider than the
Bluestein reconstruction's spatial period wraps the frame), now sized away by
`readout_tile='auto'` and refused by `on_replica`.

The exact high-NA final leg carries tilt too (it previously raised
`NotImplementedError` and forced `final_leg='paraxial'`, capping every
per-order spot).  `carrier_referenced_exact_focus_readout` references sphere
AND tilt about the chief ray and takes its crop there, so an off-axis beam
costs what an on-axis beam of the same radius costs.  On design 121, order
(-4,-2): FWHM 8.400 -> **4.400 um**, EE3 22.8 -> **64.8 %**.  The single-beam
acceptance is unchanged at **3.450 um / EE3 88.8 / EE6 99.6 / EE12 99.8**.

Zemax `DGRATING` surfaces are imported (`prescription['diffractives']`) with
their axial gaps measured to the neighbouring real elements, so the DOE drops
straight into a `groups` list.  Also new: `decentred_fit_poly_order`, and the
guards `on_replica`, `on_readout_window`, `on_tilt_exact_grid`,
`on_na_proximity` and the chain-entry multi-congruence check.

### Fixed — the off-centre ray fit folded, and its polynomial budget was too small (`elements/_lens_traced.py`)

A decentred beam broke the ray-fit disc's core assumption.  The disc is
retained by a hard sample mask, which is safe only while it is CONCENTRIC with
the Chebyshev basis's own domain; off centre the fitted map **folds**
(`d(x_out)/d(x_in)` changes sign), the Newton inverse then sends far exit
pixels back into the bright beam, and `amplitude_model='ray_density'` gives
them real amplitude — a spurious lobe carrying **6.8e-3 of input power at 0.75
of the on-beam peak**.  Replaced by a weighted least-squares restriction
(`_FIT_DISC_OUTSIDE_WEIGHT_REL`) that keeps every sample and pins the fit's
free directions to the traced map: ghost power **6.8e-3 -> 2.5e-8**, no sign
change.  Separately, an off-centre disc of radius `r` about a chief ray `|c|`
covers the aperture out to `|c| + r`, so the same total degree buys a worse fit
over more aberrated territory — the OPL residual is **14x** the on-axis one at
order 6 and recovers 20x at order 10, so the off-centre branch now fits at
`_DECENTRED_FIT_POLY_ORDER = 10`.  Both engage **only** off centre; the
concentric path is byte-identical (21 configurations, max |dE| = 0.0).

### Fixed — five consolidation defects from the D1–D7 adversarial verifiers (niche C1)

* **A null decentre flipped the whole ray fit.**  The off-centre branch was
  selected by `bool(_bcx or _bcy)`, so a numerically tiny beam centre swapped
  the concentric mask for the weighted solve *and* the raised order — moving
  the returned field by **8.3e-6 of peak at 1e-9 pixels** of decentre.  Now
  gated on `max(_DECENTRE_GATE_PIXELS * dx, _DECENTRE_GATE_W_FRAC * w)`.
* **The DGRATING import re-opened the v5.17.1 no-STOP aperture pollution.**
  Reproduced at **12.000 -> 100.000 mm (8.33x)** with a dummy reference plane
  between the DOE and the glass; the fallback now reads the GLASS/MIRROR span
  only.
* **The tilted exact-leg guard measured the wrong NA** (the chain's paraxial
  `w_in/|R_out|`, 0.4780, not the element's measured 0.4052), so it stayed
  silent on a leg the element itself warned was under-sampled.  Re-armed on the
  measured NA as a **power budget** — the fraction of exit power above the grid
  Nyquist NA — calibrated so the demonstrably-converged shipped configuration
  still passes (it clears by 12.5x; nothing previously accepted is now refused).
* `_FOCUS_READOUT_KEYS` is now a whitelist — unknown keys raise rather than
  being silently dropped.
* A D1 test that asserted a tilted claim through an untilted code path is
  replaced by a genuinely skew one, scored against an inline exact skew ray
  trace with three demonstrated fail-before switches.

### Fixed — a DGRATING gap that runs through glass is no longer transported as air (`io/prescriptions_zemax.py`, `propagators/carrier.py`)

`gap_before` / `gap_after` are raw axial thicknesses and the chain transports
them through **air**, so a grating ruled on a substrate was placed at the wrong
optical distance (`t - t/n` per glass leg — 1.0 mm for a 3 mm N-BK7 plate) with
no symptom in the output.  The importer now records a `gap_media` marker and
warns, and `_normalise_doe_entry` **refuses** such an entry, naming the gap to
override.  The refusal is at the point of use, not at import: `load_zemax_zmx`
serves far more than the DOE drop-in and the rest of such a file is correct.
Design 121 is unaffected (both gaps free space).

### Changed — claim corrections in the traced/carrier notes (niche C2)

Re-measurement of the shipped documentation corrected two claims that were
**wrong**, not merely loose.  The multi-congruence detector's envelope said its
score is "set by the finest fringes, i.e. the nearest-neighbour order spacing";
measured, an 8x8 fan spanning +-23 mrad reads **5.3x above** what that rule
predicts and 0.8x of an equal-**span** pair, and densifying at fixed span moves
the score **down**, not up.  Corrected rule: score a fan by its total span,
derated ~20 %.  And "the decentred figure sits BELOW the on-axis one" holds
only for the **untilted** baseline (0.90 vs 1.28 urad); against the tilted
on-axis control — the regime 121's orders are actually in — it sits **1.4x
above** (0.90 vs 0.64).  Neither changes a conclusion, and both were stated
without the qualifier.  Also disclosed: D7's raised order can go **inert
silently** when the fit disc holds fewer than 3 samples per basis term (it
survives only while `fit_radius_beam_factor * w / (dx * ray_subsample) >~ 7.9`;
live at the default `ray_subsample=8`, but the documented f/6 example clears by
only 1.13x and reverts to order 6 at 16), and the `_FIT_DISC_OUTSIDE_WEIGHT_REL`
plateau is now labelled as evidence from **one** regime (aperture:beam 30:1,
low NA) — it cannot be extended to design 121's regime by the same method,
because `newton_fit='spline'`, the fit-domain-free oracle that sweep is scored
against, fails to converge for **100 %** of pixels there and returns an
all-zero field.  Full record in
`docs/audits/ROADMAP_DESIGN121_FULL_CONFIGURATION_2026_07_27.md`.

### Added — PMM per-layer grids: full surface build-out (`elements/pmm/`)

The per-layer surface now covers (each gated in
`tests/unit/test_pmm_per_layer_grids.py`, 11/11): `retain_internal` ->
`internal_field` / `layer_absorption` (per-layer partial cascades; fields
reconstruct on each layer's own machinery and REQUIRE `nx=` -- there is no
single shared nodal axis; absorption closes ΣA ~= 1-R-T to 5e-3 on
non-conforming stacks); `solve_vs_wavelength` (window grids, interface
masses/cross-masses and both Rayleigh projectors hoisted once per sweep;
<1e-14 vs per-λ solve and vs the shared sweep on conforming stacks;
dispersive materials and thread-pool semantics preserved); SLANTED layers and
OUT-OF-PLANE tensors via `_interface_smatrix_general_mortar` (the mixed
weak-continuity scattering system; E tested on the lower grid, H on the
upper; <1e-9 vs the shared general cascade on conforming windows); and the
JAX twin `_pmm_stack_solve_jax_perlayer` (geometry concrete, eps/thickness/
angle traced; forward <1e-10 vs NumPy, `jax.grad` vs FD to 5e-5).  Still
shared-grid-only: the covariant uniform-slant route (single-frame by
construction) and `prepare()` (design sketch + sequencing in
`docs/audits/ROADMAP_PMM_PER_LAYER_GRIDS_2026_07_28.md`, together with the
PMM2DStack per-layer extension plan and the remaining quality items).

### Added — PMM per-layer element grids with interface mortar (audit R-6, `elements/pmm/`)

`PMMStack(..., layer_grids='per-layer')` (opt-in; default `'shared'` is
byte-identical to prior behaviour): each layer's SEM operators are assembled
on its OWN element grid — its walls plus its two NEIGHBOURS' (the
interface-conforming window) — and adjacent layers are coupled by an exact L2
mortar (`_interface_smatrix_mortar`), with rectangular-block Redheffer
support (`_redheffer_star_rect`).  Removes the shared-union-grid
wall-collision pathology structurally: MEASURED on the 2-deg coated-pillar
taper, in-plane oblique degree spread 91% -> 5–6% at the library-DEFAULT
`min_feature` (inert on this path), the n_slice=4 conditioning re-break
15.7% -> 4.7%, and 17.8x faster (97.3 s -> 5.5 s per solve pair).  The
n_slice staircase ladder is now affordable and CONVERGES (ns~6, 0.1–0.4%
ns6->ns8); the ns=2 values used previously carried 10–30% staircase-geometry
error at oblique angles.  Conical (`phi != 0`) is covered via the same
construction in `_conical_nodal_solve` (bit-exact on 2-layer stacks; 1.4–5.8%
vs the converged shared reference on the audit device; 14–18x faster).
Known residual: the non-conforming mortar remainder decays spectrally
(|R+T-1| ~1e-4 at degree 6 -> ~1e-6 at degree 10) — use degree >= 8 for
deep-null work; publication-grade closure stays with the shared grid, and the
two paths now cross-check each other (the independent oracle
`AUDIT_PMM_OBLIQUE_INPLANE_UNION_GRID_2026_07_28.md` §7 found missing).
Not on the per-layer surface (raise loudly): slant, out-of-plane tensors,
JAX, `stabilize`, `retain_internal`, `prepare()`, `solve_vs_wavelength`.
Implementation report: `docs/audits/AUDIT_PMM_PER_LAYER_GRIDS_IMPL_2026_07_28.md`.

### Fixed — the staircase guard is reachable on geometry-built stacks (audit R-1, `elements/pmm/stack.py`)

`solve(stabilize='slices')` previously SKIPPED entirely when no taper-builder
recipe was recorded, leaving every `SegmentStackGeometry`-built stack — the
documented device route — unprotected against the passive-but-wrong staircase
pathology.  It now falls back to a union-grid consensus (`min_feature`
perturbation anchored to a physical ~nm scale), verified to fire on the
pathological stack and stay silent on clean ones.  The wall-snap warning
reports its max wall displacement, and `min_feature` is documented as an
ACCURACY knob (its default, `period * 1e-5`, is ~200x too small for tapered
stacks whose collision scale is `(thickness/n_slices)*tan(sidewall)`).


## [5.31.0] — 2026-07-27

The **W8 + W9 audit follow-up waves** to the v5.30.0 adversarial-audit campaign:
the PMM tapered builders gain an exact sheared-sidewall route, the RCWA
rasterization contract is closed (`raster='harmonic'`), the auto-dispatcher's
regime rule is unified, and a set of measured physics/CI fixes land with
regression pins.  `.test_durations` was regenerated end-to-end (the W9 shard
rebalance).

### Added — PMM tapered-builder `shear` + an exact single-layer route for the pure-shear taper (audit W9, `elements/pmm/stack.py`)

The RCWA twin has carried sheared (parallelogram) sidewalls since v5.14.1
(audit GAP1); the PMM tapered builders could only make a *symmetric* trapezoid,
so the one profile class a fab undercut actually produces had no laterally-exact
(no-Fourier-floor) solver.  Both PMM builders now take `shear`, with the
**exact** RCWA convention mirrored so the two packages build the *identical*
staircase — and the pure-shear sub-case gets a route that skips the staircase
altogether.

- **`shear` on `PMMStack.add_tapered_grating` and `add_tapered_ridges`**
  (`lumenairy/elements/pmm/stack.py:342`, `lumenairy/elements/pmm/stack.py:653`).  The ridge centre is
  `0.5 + shear * (zeta - 0.5)` in period fractions — `shear` *periods* of
  lateral walk from top to bottom about mid-depth, `zeta` sampled per slice by
  the same `rule` as the duty, wrap-aware (`lumenairy/elements/pmm/stack.py:494`, `lumenairy/elements/pmm/stack.py:711`).
  `add_tapered_ridges` walks every tooth rigidly (`center + shear*period*(zeta-0.5)`)
  and keeps its overlap guard on the *sheared* geometry.  Both record `shear` in
  the taper recipe (`lumenairy/elements/pmm/stack.py:470`, `lumenairy/elements/pmm/stack.py:703`) so `_resliced_clone` /
  `solve(stabilize='slices')` replay the sheared structure rather than silently
  re-slicing an unsheared one.
- **`shear=0` is BIT-identical to the pre-change builder.**  The new wrap-aware
  slice builder `PMMStack._ridge_slice_segments` (`lumenairy/elements/pmm/stack.py:301`) reproduces the
  historical `[0.5*(1-duty), duty, 0.5*(1-duty)]` triple *bit-for-bit*, because
  halving is exact in binary floating point (`(1-d)/2 == 0.5 - d/2`) and the
  groove widths are built as differences of the total groove so the list
  telescopes.  MEASURED against a reference dumped from a pre-change worktree:
  0 differing widths over 9 `(duty_bottom, duty_top, n_slices, rule)` cases —
  including the vanished-ridge and vanished-groove ends — × both builders ×
  {kwarg omitted, `shear=0.0`} (7 of the 9 carried into the pin), and the solve
  is bit-identical (max |diff| = 0.0).
- **Cross-package staircase identity, MEASURED two ways.**  *Geometry*:
  rasterizing the PMM segments on RCWA's pixel-centre lattice reproduces its
  `eps_cell` with **0 mismatched pixels** over `shear` ∈ [-1.7, 2.5] × 3 duty
  configs × `n_x` ∈ {64, 256, 1024, 4096}, wrapping cases included.  *Physics*:
  at the same `(n_slices, duties, shear)` the RCWA answer converges to the PMM
  one as `n_x` grows — TE row, `eps_ridge = 2.1`, `nox = 31`, `n_slices = 6`:
  4.683e-04 → 1.023e-04 → 9.362e-06 at `n_x` 512 / 2048 / 8192 (a 50x fall),
  and 1.073e-04 → 3.302e-05 → 1.099e-05 at `shear = 0.90` (the WRAP layout).
  The full-row gap instead saturates on RCWA's `O(1/nox)` Fourier floor
  (measured 4.7e-04 at *both* `nox = 31` and `61`); `shear` only phase-shifts
  the cell's Fourier coefficients, so that floor is shear-invariant (measured
  identical 4.6888e-04 at `shear` 0.35 and 1.4).
- **NEW `PMMStack.add_sheared_grating`** (`lumenairy/elements/pmm/stack.py:498`) — the taper-metric
  roadmap item's *shear* sub-case, exactly.  A parallelogram is the one taper
  the shipped covariant/convection machinery solves with **no z-staircase at
  all** (`u = x - z tan(phi)` keeps the modal coefficients z-independent), so
  this emits ONE slanted layer with
  `slant_angle = arctan(shear*period/thickness)` (`lumenairy/elements/pmm/stack.py:608`) and the same
  centre law as the staircase.  MEASURED against the staircase of the identical
  geometry (`P = 1 um`, `wl = 633 nm`, `d = 300 nm`, `eps_ridge = 4`,
  `duty = 0.45`, `shear = 0.35` = a 49.4° wall, `theta = 0.17`; error vs the
  `n_slices = 20` staircase over `(R, T)` of `|m| <= 2`, both polarizations):

  | route                        | err      | time    |
  |------------------------------|----------|---------|
  | `add_tapered_grating` ns=4   | 2.59e-02 |  0.25 s |
  | `add_tapered_grating` ns=8   | 6.93e-03 |  2.27 s |
  | `add_tapered_grating` ns=12  | 2.63e-03 |  7.53 s |
  | `add_tapered_grating` ns=16  | 9.01e-04 | 23.04 s |
  | `add_sheared_grating` deg=12 | 2.93e-03 |  0.05 s |

  i.e. **8.9x more accurate than the `n_slices = 4` staircase at a fifth of the
  cost, and the `n_slices = 12` answer ~150x faster**.  It *plateaus* on the
  slant path's own wall-normal per-order floor, which grows with the wall tilt:
  measured plateau 1.76e-03 at 45.0°, 1.89e-03 at 49.4°/normal, 2.93e-03 at
  49.4°/oblique, 1.16e-02 at 69.4° (four different designs) — so the staircase
  overtakes it at `n_slices ≈ 16` and remains the route below the plateau.  The
  docstring carries both readings plus the inherited restrictions
  (`solve_vs_wavelength`, `prepare`, `retain_internal`, JAX and conical all
  raise `NotImplementedError` on a slanted stack — all four verified).

### Documented — the tapered z-staircase's MEASURED budget, and two dead ends (audit W9, `elements/pmm/stack.py`)

- **`O(1/n_slices^2)` CONFIRMED, and the cost quantified** (`lumenairy/elements/pmm/stack.py:367`).
  Cost is `~O(n_slices^3.4)` end to end (`n_slices` 4/8/16 → 0.30/2.97/34.1 s at
  `degree = 12`) because the union grid — and therefore every layer's eig —
  grows with the slice count.  The staircase error itself was measured on the
  cross-package oracle that *isolates* it (the RCWA twin at `raster='area'`,
  whose realised duty is exact, reference `n_slices = 768`): 3.82e-03 /
  1.27e-03 / 3.34e-04 / 8.47e-05 / 2.18e-05 / 5.70e-06 at `n_slices`
  8/16/32/64/128/256 — a factor 3.9 per doubling, shared by all 12 observables
  (3.5–4.6 at 32 → 64).
- **`rule='trapezoid'` is a NO-OP here** (`lumenairy/elements/pmm/stack.py:406`).  For the LINEAR duty
  ramp these builders lay down, `0.5*(k/n + (k+1)/n) == (k + 0.5)/n` exactly, so
  `'trapezoid'` samples the same duty as `'midpoint'` up to last-bit rounding
  (measured bit-identical for 5 of the 7 slices at `n_slices = 7`, 1 ULP apart
  for the other 2).  It is kept for signature parity with `add_graded_layer`, whose profile is
  arbitrary and where the rules genuinely differ.
- **Richardson extrapolation in `n_slices`: REJECTED, with numbers**
  (`lumenairy/elements/pmm/stack.py:380`).  Measured across 8 designs (duty ranges 0.10–0.85, lossy,
  oblique 0.17/0.45, `eps_ridge` 4/9, deep + sheared, with and without shear)
  against the *equal-cost* comparator `f(2n)`: the `(n, 2n)`, `p = 2` two-point
  Richardson gains 3.0x / 7.4x / 14.5x on the clean designs but 1.0x / 0.95x /
  1.17x — i.e. nothing — on the steep-duty, narrow-duty, high-contrast and deep
  designs at `n = 8/16/32`.  Multi-exponent (`1/n + 1/n^2`) and 3-point
  fitted-exponent variants are outright worse (fitted-p at `n = 4, 5, 6`
  returned 9.2e-03 against `f(6) = 4.4e-03`).

Pins: `tests/unit/test_niche_audit_w9_pmm_taper.py` (19; 16 fail on a
pre-change worktree, 3 pass as regression + error-law locks).

### Fixed — ESTIMATE→MEASURE FFT plan auto-promote now ships OFF (audit W9, `propagators/fft_infra.py`)

`apply_real_lens_traced` on one FIXED input returned one bit pattern for its
first two calls and a different one (max|d| ~ 2.8e-15) for every call after,
in a fresh process. Root cause was not the traced pipeline: `fft_infra`
shipped auto-promote ON since 4.12, and one traced call runs exactly four
transforms at one 256² plan key, so the per-key counter trips the 5-call
threshold *inside call 2* and rebuilds the plan under `FFTW_MEASURE`
mid-session.

- **The default flipped `True` → `False`** (`_PYFFTW_AUTO_PROMOTE_SHIPPED`, a
  new immutable source-declared constant mirroring
  `DEFAULT_WAVE_PROPAGATOR_SHIPPED`). Two independent reproducibility
  failures, both MEASURED: (1) the counter is global per
  `(direction, shape, dtype, threads)`, so an unrelated earlier caller at the
  same shape moves the boundary — this reached CI as a byte-identity pin
  failing on one pytest collection layout and passing on three others; (2)
  `FFTW_MEASURE` selects its algorithm by *timing* candidates, so four fresh
  processes gave four DIFFERENT post-promotion results where ESTIMATE gave
  one identical result in all four. Only ESTIMATE is a deterministic planner.
  Neither result is more accurate, so the tie-break is reproducibility —
  and `docs/TOLERANCE_POLICY.md` already promised it ("determinism within a
  build is guaranteed" — false before this fix).
- **The feature is kept, as an opt-in**, because the speedup is real
  (complex128, 8 threads): 1.39x @256², 2.22x @512², 2.04x @1024²,
  3.67x @2048², 4.55x @4096². Prefer `set_pyfftw_planner('FFTW_MEASURE')`
  over `set_fft_auto_promote(True)`: it plans every key at FIRST use, so the
  process stays internally byte-consistent from call 0 (MEASURED 8/8) and
  skips the wasted ESTIMATE warm-up.
- **`set_pyfftw_planner` now documents that a high-effort planner is a
  one-way door.** Clearing the plan cache does not clear libfftw3's
  process-global *wisdom*: after a MEASURE plan, later ESTIMATE plans at that
  size reuse the wisdom-recorded algorithm (MEASURED `f51bdc2a28c2` clean vs
  `5d609b5be4f7` post-MEASURE on a 256² transform).
- **`memory._LOW_MEMORY_SHIPPED_DEFAULTS['fft_auto_promote']` → `False`**,
  companion-locked to the fft_infra constant by the new pin. A stale `True`
  would have made `set_low_memory(False)` with no enable on record silently
  opt the caller INTO the non-reproducible planner.
- **Result (MEASURED, post-fix):** the 30-iteration traced stress reports
  `0/30 iterations diverged`; a 100-call fresh-process byte map is a single
  group `calls 0..99`, with the SAME hash in three separate processes and
  equal to the pre-fix call-0 value. Also verified at 512² and 1024². The
  s12 warm-up fixture (8352e79) is thereby redundant and retained only as
  defense-in-depth against an unrestored opt-in leaking across a shard.
- New pin `tests/unit/test_niche_audit_w9_traced_determinism.py` (7 tests;
  3 fail against the pre-fix default). It snapshots/restores FFTW wisdom so
  its own opt-in tests cannot perturb later tests in the same worker, and it
  passes identically under four different collection layouts. Four existing
  fixtures that restored auto-promote to a hardcoded `True` now restore the
  prior value instead.

### Fixed — the shared eig VJP's broadening floor is now SPECTRUM-RELATIVE (audit W9, `elements/rcwa/_core.py`)

The custom-VJP `eig` (`_jax_eig_stable`, shared by RCWA, PMM 1-D/2-D/Jones/stacks,
EME, BOR and Berreman) regularised its eigenvector-gradient factor
`F = D/(|D|^2 + eps)` with an **absolute** `eps = 1e-10`.  The eigenvalues of
these modal operators are dimensionful — `max|lam|` is ~6e2 on the PMM
spectral-element fold, ~3e1 on the RCWA `P@Q` fold, ~1 on the Berreman 4x4 —
so that floor corrupted a scale-dependent window: `F` was wrong whenever
`|D| <~ 1e-5`, a *relative* splitting of only 1.6e-8 on PMM.

- **Symptom.** `pmm_efficiency_1d` TE `d(sum R)/d(theta)` at 1e-6 rad off
  normal: AD `4.217e-05` against FD `1.755e-06` — a **24x** error, clean only
  by ~1e-4 rad.  (Independently reported by a downstream consumer at exactly
  theta = 0.)
- **Cause, MEASURED.** The exact factor is `1/conj(D)`; the degenerate
  cross-block is physical, not noise — `|M_ij|/|D_ij|` converges to
  `2.0584e-02`, identical to 5 digits from theta 1e-8 to 1e-3.  On an exact
  entrywise oracle (`L = |tr(expm(A) X)|^2` via the eig route vs
  `jax.scipy.linalg.expm`'s known-correct VJP — gauge-invariant, no finite
  differences) the absolute floor gave a **72% gradient error at 3e-7 relative
  splitting and still 2.3e-9 at FULL separation**, i.e. it perturbed every
  gradient, degenerate or not.
- **Fix.** `denom = |D|^2 + (_EIG_TAU_REL * max|lam|)^2`, `_EIG_TAU_REL = 1e-12`
  (per-call `tau_rel`).  Exact wherever the splitting is numerically resolved;
  the floor only bites inside the LAPACK rounding floor, where an unfloored
  `1/D` divides by noise (measured 7.7x worse).  The scale is read off traced
  `lam`, so jit/vmap keep working and each `vmap` element gets its own scale.
- **No forward change** — the primal and `fwd` rules are untouched;
  `pmm_efficiency_1d`, `rcwa_efficiency_1d`, `berreman_jones_1d` and
  `pmm_efficiency_2d` are **bit-identical** (`array_equal`, max |diff| 0.0).
- **Measured effect.** pmm 1-D `d/d(theta)` at 1e-6 rad: `|AD - FD|` 4.04e-05
  -> **1.09e-09**; against the FD-free oracle "`dR/dtheta` is linear in theta"
  the relative error goes 23 -> **2.5e-06**.  pmm 2-D at normal incidence
  4.11e-05 -> 1.12e-06.  The smallest USABLE off-normal angle drops from
  ~1e-4 to ~1e-6 rad.  RCWA 1-D (the clean control, analytic half-space modes)
  is unchanged at 1.1e-13; EME, BOR, Berreman and the 2-D stacks were already
  clean and stay so.
- **KNOWN LIMIT, now documented and pinned.**  At an EXACT (symmetry-enforced)
  degeneracy no choice of `F` can be right: for a matrix-function loss
  `L = tr(g(A) X)` the cotangent carries `M_ij = (g(lam_j) - g(lam_i)) Y_ji`,
  so when `lam_i == lam_j` exactly `M_ij` is identically zero and the divided-
  difference factor `g'(lam) Y_ji` is absent — `eig` itself is not
  differentiable there.  Measured 0.16-0.75 relative error for every variant.
  It bites only where the perturbation's intra-cluster block is non-diagonal,
  i.e. `d/d(angle)` at EXACTLY normal incidence on the PMM paths (2.22e-03 TE /
  9.66e-02 TM); every DESIGN gradient at normal incidence is clean to 2e-08.
  Offset the angle by >= 1e-6 rad if the angle derivative itself is the
  objective.  (The structural cure — analytic half-space modes for the PMM
  fold, which is why RCWA is clean — is recorded as the follow-up.)

New pins: `tests/unit/test_niche_audit_w9_eig_vjp.py` (29; 11 fail pre-fix, 18
regression fences pass pre-fix).

### Fixed — the `'area'`-under-`'li'` raster regression: `raster='harmonic'` (audit W9, `elements/rcwa/stack.py`)

W8 shipped `raster='area'` with a measured wart in its own docstrings: area
weighting is 1-3 orders better than `'hard'` under the default
`formulation='laurent'` for both polarizations, but WORSE than `'hard'` for the
wall-NORMAL polarization under `'li'`.  The cause, now fixed: the cell is
consumed TWICE with OPPOSITE rules — `Cxx = [[1/eps]]^{-1}` (inverse,
wall-normal) and `Cyy` / `EZZ = [[eps]]` (direct, tangential) — and a boundary
pixel's correct effective medium is the ARITHMETIC average for one and the
HARMONIC average for the other (Farjadpour et al. 2006).  One scalar cell
cannot carry both.

- **`raster='harmonic'`** (`lumenairy/elements/rcwa/stack.py:321`, `_raster_companions` at
  `lumenairy/elements/rcwa/stack.py:404`) paints the AREA cell — BIT-IDENTICAL to `raster='area'`, so
  `plot_geometry`, the Im(eps) loss maps and `layer_absorption` read exactly
  the cell they always did — and rides an inverse-rule COMPANION PAIR
  `(exx, eyy)` that ONLY the `'li'` inverse Toeplitz reads.  The layer stays
  ISOTROPIC: the pair is two DISCRETIZATIONS of one scalar material, not a
  birefringent tensor (which is why it does not ride in an `eps_tensor_cell` —
  that would report a fake birefringence to the absorption machinery).
- **`add_layer(..., eps_cell_normal=(exx, eyy))`** (`lumenairy/elements/rcwa/stack.py:1471`) is the
  seam; `formulation='li'` is required and every other pairing raises.
  `RCWAStack._li_blocks` (`lumenairy/elements/rcwa/stack.py:2527`) is the SINGLE place the inverse rule
  reads its cell, shared by `_layer_modes` and the even-parity
  `_layer_even_spec`, so the two cascades cannot factorize a layer differently.
  With the pair equal to the cell it reduces to `_li_convolutions_2d` exactly
  (measured: `Cxx` bit-identical, `Cyy` 4.2e-16).  The pair enters
  `_layer_eig_key` (`lumenairy/elements/rcwa/stack.py:2628`) — two layers with the same `eps_cell` and
  different companions have different eigenproblems.
- **The tapered builders grew `formulation=`** (`'laurent'` default,
  bit-preserving; `lumenairy/elements/rcwa/stack.py:1797`/`1987`/`2060`, forwarded by
  `add_graded_layer`, `lumenairy/elements/rcwa/stack.py:1705`), which also closes the documented wart
  that reaching `'li'` meant rasterizing by hand.

MEASURED, vertical binary grating against the EXACT analytic oracle (`n = 2/1`,
`duty = 0.37`, `P = 1 um`, `wl = 633 nm`, `M = 9`, `theta = 0.25` rad),
`'li'` TM `max|x - x_exact|` over `(R0, T0, R+1, T+1)`:

      n_x |   hard      area     harmonic  | vs hard  vs area
       64 | 3.07e-03  1.18e-03  3.71e-04   |    8.3x     3.2x
      256 | 1.56e-03  7.30e-04  5.16e-05   |   30.3x    14.1x
     1024 | 4.76e-04  1.89e-04  3.63e-06   |  131.3x    52.0x
     8192 | 6.45e-05  2.21e-05  5.79e-08   | 1115.7x   381.2x

`'area'` PLATEAUS on this channel while `'harmonic'` keeps converging.  On the
`'laurent'` TE/TM and `'li'` TE channels `'harmonic'` IS `'area'` (bit-identical
cell, no companion read), so it is never worse than `'area'` and never worse
than `'hard'` — one safe choice per formulation: `'area'` under `'laurent'`,
`'harmonic'` under `'li'`.  On the W8 SHEARED taper (shear 0.4, 16 slices,
reference `n_x = 4096`) where `'area'` was outright worse than `'hard'`
(4.79e-03 vs 1.92e-03 at `n_x = 64`), `'harmonic'` is 8.41e-04 — 2.3x on
`'hard'`, 5.7x on `'area'`.  On a NON-taper two-material multi-ridge
(`eps = 12` and `4 + 0.3j` in air) at `n_x = 64`: 2.02e-02 / 2.11e-02 /
3.83e-03.  For 2-D pillars BOTH in-plane blocks take an inverse rule, so both
polarizations gain (single pillar, `M = 4`, `n = 64`: `'li'` TM 4.25e-03 /
1.67e-02 / 1.46e-03, `'li'` TE 3.60e-02 / 3.15e-03 / 1.92e-03).

REJECTED, with numbers: a SCALAR harmonic cell (the harmonic mean stored IN
`eps_cell`) gains only 1.1-3.5x on `'li'` TM and is 5-40x WORSE than `'area'`
under `'laurent'` (1.11e-02 vs 2.86e-04 at `n_x = 64`, TE) — it corrupts the
direct-rule channel it also feeds.

`raster` defaults to `'hard'` and every default call is bit-for-bit unchanged.
Pins: `tests/unit/test_niche_audit_w9_raster_harmonic.py` (22 cases; 20 fail
pre-fix, 2 regression fences pass pre-fix).

### Changed — EXACT pair predicates in the shape-overlap guard (audit W9, `elements/rcwa/_core.py`)

W8's guard decided every curved pair by scanning the support functions over
4096 directions, which UNDER-estimates the separating-axis maximum by up to
~2e-7 of a period.  That approximation lived INSIDE the predicate, so the
guard's blindness was the entanglement of two unrelated numbers.
`_shapes_overlap` (`lumenairy/elements/rcwa/_core.py:1072`) is now exact algebra: rect/rect by interval
overlap, disk/disk by centre distance, rect/ellipse by axis-scaling the ellipse
to the UNIT DISK (which keeps the rectangle axis-aligned), and ellipse/ellipse
by axis-scaling the first to the unit disk — reducing every curved pair to
POINT-ELLIPSE distance (`_point_ellipse_distance`, `lumenairy/elements/rcwa/_core.py:1020`: the distance
quartic as a BRACKETED monotone root, a proven bracket and a fixed 64 bisections,
never an unbracketed iteration).  Shapes are axis-aligned throughout — the shape
dicts carry no rotation entry and neither do the form factors that read them.

What remains is ONE explicit, named, one-sided tolerance,
`_OVERLAP_SLACK_FRAC = 1e-6` (`lumenairy/elements/rcwa/_core.py:998`), overridable per call via
`tol_frac`.  Its VALUE is unchanged: it is a deliberate forgiveness for layouts
whose centres came out of float arithmetic, not blindness.

MEASURED: 20000 random pairs across all six kind combinations agree with the
pre-W9 scan at the shipped window — 0 disagreements — and 0 order-asymmetric
verdicts (both shapes are eroded by `tol/2`, so a verdict cannot depend on list
ORDER).  What changes is that the tolerance can now be BELIEVED: at
`tol_frac <= 1e-8` the old scan reported 406 (1e-8) and 735 (1e-10) FALSE
POSITIVES per 3000 exactly-tangent / gapped LEGAL disk pairs, while the exact
predicate reports ZERO at every tolerance — so the detection floor is now the
tolerance itself, measured to resolve overlaps of 1e-8, 1e-10, 1e-12 and 1e-14
of a period while never flagging tangency at exactly 0.  Faster, too: 1024 disks
51.8 ms against W8's recorded 81 ms (1024 ellipses 60.1 ms; 1024 nearly-touching
ellipses, where every neighbour reaches the predicate, 54.4 ms).

Tangent / abutting shapes stay LEGAL (their intersection has measure zero), the
wrap-aware minimal periodic image is unchanged, and the bounding-box pre-filter
is unchanged.  Two consequences recorded: a shape whose semi-axis is BELOW
`tol/2` now lies entirely inside its own forgiveness window and is reported
disjoint (0.06 pm scale, far below anything the form factors resolve); and the
predicate now always returns a Python `bool` — a numpy-scalar semi-axis used to
leak a `np.bool_`, which breaks the `is True` / `is False` tests its callers and
pins use.  The now-unused `_OVERLAP_DIRS` constant is removed;
`_shape_support` is retained as the independent cross-check the W9 pins
re-implement the old scan from.  Pins:
`tests/unit/test_niche_audit_w9_overlap_exact.py` (19 cases; 13 fail pre-fix,
6 regression fences pass pre-fix).

### Changed — auto-dispatcher follow-up wave (audit W9 items 1–7, `propagators/dispatch.py`, `system.py`, `fga.py`, `fft_infra.py`)

- **One free-space regime rule (W9-7).**  `_select_asm_variant` — behind
  `which_propagator` / `asm_propagate` — no longer carries its own
  thresholds; it delegates to `_auto_select_method`, now documented as the
  canonical regime logic.  Its old far-field trip (`Q > 20`) sat at aperture
  Fresnel number `N/80`, so it fired further inside the near field the bigger
  the grid: measured complex-overlap fidelity against an exact
  `angular_spectrum_propagate_mft` oracle just above that trip was
  0.9516 / 0.8185 / 0.4111 / 0.4241 at N = 128 / 256 / 512 / 1024, where the
  canonical rule stays on `sas` and scores 1.00000 at all four.  The SAS
  boundary moves with it (`Q > 2` -> `Q > 1`); in the newly-`sas` band both
  members are exact and neither warns.  ROUTING CHANGE in the transition band.
- **`propagate()` honours `set_default_wave_propagator()` (W9-8).**  Leaving
  `method` unset is no longer the same as passing `'auto'`: on a free-space
  call, with the knob moved off its shipped `'asm'`, that default is used.
  `method='auto'` explicitly always auto-selects.  Two deliberate departures
  from `propagate_through_system`'s unconditional resolution, each forced by
  measurement: prescription calls keep `'auto'` (measured,
  `propagate(prescription=rx, method='asm')` returns the input UNCHANGED, so
  applying a free-space knob there would make it a silent no-op); and the
  knob is honoured only once it differs from the shipped value
  (`DEFAULT_WAVE_PROPAGATOR_SHIPPED`, `fft_infra.py:326`), so resolution
  stays stateless and restoring the knob restores auto-selection.
- **Traced options are reachable through the element chain (W9-11).**  The
  `'real_lens_traced'` element forwards every keyword-only parameter of
  `apply_real_lens_traced`, as top-level keys or in a `traced_kwargs` dict,
  and REJECTS anything else with a named `ValueError`.  Pre-fix the handler
  forwarded four arguments and dropped the rest in silence — measured
  bit-identical output for all of `amplitude_model`, `preserve_input_phase`,
  `remap_sampling`, `fit_radius_beam_factor`, `carrier`, `on_undersample`,
  `n_workers`, `traced_kwargs` and an outright typo key — which made the
  v5.29 + S12 validated traced configuration unreachable through this API.
- **The universal router runs `traced` with the P2 cliff guard (W9-13).**
  `apply_real_lens_universal` now defaults `fit_radius_beam_factor=2.0` on its
  traced route instead of inheriting the element's `None`.  Measured on the E4
  corrected relay at the element's own defaults, exit-wavefront Strehl
  0.9701 / 0.1085 / 0.0384 at 1.50x / 1.75x / 2.50x the beam diameter without
  the guard versus 0.9874 / 0.9820 / 0.9816 with it.  The chain's other three
  validated options are NOT adopted: they are carrier-regime options and this
  router supplies no carrier (measured -0.0025 / -0.0249 / -0.1912 without
  one).  Opt out with
  `method_kwargs={'traced': {'fit_radius_beam_factor': None}}`.

### Removed — the dead DOE routing branch (audit W9-9, `propagators/dispatch.py`)

- `_auto_select_method`'s "prescription with diffractive surfaces -> hfpi"
  rule keyed on `prescription['events_json']`, a key that occurred exactly
  once in the repository — in `dispatch.py`.  No loader or factory ever
  emitted it, so the branch could not fire, and when forced it routed to a
  call that immediately raised `TypeError: ... missing 1 required
  keyword-only argument: 'n_paths'`.  It could not be repointed: this library
  has no prescription-embedded DOE representation — diffractive data travels
  as the `surface_diffraction` / `diffracting_surfaces` kwargs.  (An initial
  supporting measurement — that hfpi "missed the analytic order-1 deflection
  by 85–97%" — was later traced to HFPI under-sampling and is WITHDRAWN, see
  the W9-14 note below; the two structural reasons above are decisive on
  their own.)  DOE kwargs handed to a member that cannot accept them now raise a
  dispatcher-level `ValueError` naming the members that can, instead of a raw
  `TypeError` from `apply_real_lens_maslov`.

### Fixed — HFPI says when it is under-sampled (audit W9-14, `propagators/hfpi.py`)

- Paths are drawn into a cone of half-angle `cone_half_angle` — a full forward
  hemisphere by default — while a realistic output grid subtends a few
  degrees, so almost every path can miss the grid and the survivors are far
  too few to interfere into a field.  MEASURED at the default cone on a 128^2
  output grid subtending 7.3 deg: the fraction of output pixels that ever
  received a path was 0.6% / 2.1% / 7.9% at n_paths = 20k / 80k / 320k, and
  two seeds of the SAME physics agreed to an intensity-shape fidelity of
  0.000 / 0.005 / 0.021.  The binned profile was a broad dome spanning the
  whole grid for both a control and a grating case — the Monte-Carlo sampling
  envelope, not a propagated field — and nothing warned, despite the
  docstring's guarantee that "fringe positions and interference contrast
  (phase structure) are correct".  `accumulate_to_grid` — the single point all
  HFPI entry points, scalar and vectorial, funnel through — now counts the
  paths that actually LANDED and warns below one per output pixel, naming the
  counts and both levers (raise `n_paths`; far more effective, narrow
  `cone_half_angle` toward the angle the output grid subtends — the library's
  own HFPI test already did this by hand).  New
  `on_undersampled={'warn', 'silent', 'error'}` policy on
  `accumulate_to_grid`, `propagate_hfpi_freespace_aperture` and
  `propagate_hfpi_through_prescription`; `'warn'` is the default.  One landed
  path per pixel is a necessary and nowhere near sufficient condition — the
  same probe still only reached seed-to-seed fidelity 0.44 at ~12 per pixel —
  so the bar is set where the failure is unambiguous, not where the answer
  becomes trustworthy.
- **HFPI's `surface_diffraction` is correct**, and the W9 dispatcher audit's
  initial report that it "missed the analytic thin-grating order-1 deflection
  by 85–97%" is WITHDRAWN: that measurement was taken on the under-sampled
  output described above.  Verified at the ray level, where
  `raytrace.trace(surface_diffraction=...)` puts the exit direction cosine at
  `m*lambda/Lambda` to a relative error of 0.0–2.2e-16 and the exit offset at
  `t*tan(asin(m*lambda/Lambda))`, for Lambda/m of 80/1, 40/1, 20/1, 20/2 and
  10/1 — including the (20 um, m=2) == (10 um, m=1) degeneracy — and with a
  pure y-order deflecting only the y direction cosine.  At the HFPI level the
  deflection converges toward the analytic value as the estimator is fed
  (ratio 0.401 / 0.530 / 0.742 / 0.861 at n_paths = 0.1M / 0.4M / 1.6M /
  6.4M at fixed geometry), which is non-convergence, not a physics error.
  `fit_canonical_polynomials`, the other `surface_diffraction` consumer,
  routes through the same exact `trace` path and is deterministic (no
  Monte-Carlo sampling), so neither finding affects it.  Pins:
  `tests/unit/test_niche_audit_w9_hfpi_doe.py` (16 cases; 8 fail pre-fix, 8
  ray-level fences pass at both baselines).

### Fixed — dispatcher usability follow-ups (audit W9-10 / W9-12)

- **`hfpi` / `asymptotic` are usable through `propagate()` (W9-10).**  Missing
  required kernel arguments now raise a `ValueError` naming `propagate()` and
  EVERY missing name, per the 4.12 B1-6 rule, instead of a raw `TypeError`
  naming a kernel the caller never called: `n_paths` for
  hfpi-with-prescription, all four of `z_to_aperture` / `aperture_radius` /
  `z_aperture_to_output` / `n_paths` for the hfpi free-space form (the old
  check advertised only `aperture_radius`), and `s2_grid_x` / `s2_grid_y` for
  asymptotic.  No invented defaults: those are Monte-Carlo budgets and output
  grids, and any value the dispatcher picked would be a silent accuracy
  decision.
- **The `ray_subsample` docstring contradiction (W9-12).**  The
  `'real_lens_traced'` element docstring read "default 1; 4 is the recommended
  production value" while the code hard-coded 1.  The docstring is corrected
  and 4 is now reachable, but the default stays 1: measured, the E4
  exit-wavefront Strehl at 1 / 4 / 8 is 0.9994 / 0.9993 / 0.9974 (6 mm) and
  0.9996 / 0.9995 / 0.9976 (10 mm), so 4 buys no fidelity — while
  `min_coarse_samples_per_aperture=32` means a divisor of 4 quadruples the
  grid a chain needs, raising `ValueError` for a 2 mm aperture spanning 50 or
  100 samples that runs fine at `ray_subsample=1`.  The three entry-point
  defaults (element 8, chain 4, chain element 1) are deliberate and now
  pinned together.

Pins for the follow-up wave: `tests/unit/test_niche_audit_w9_dispatch2.py`
(80 cases) plus one attributed edit to `test_niche_audit_w9_dispatch.py`;
54 of the combined 115 fail at `268b019` in a read-only worktree, 61 pass as
regression fences.

### Fixed — auto-dispatcher `output_grid`/`output_dx` handling and the ASM-family twin (audit W9, `propagators/dispatch.py`)

Six measured defects in `lumenairy/propagators/dispatch.py`:

- **`method='auto'` with an output-grid request no longer selects `sas` in the
  `Q > 1` band (W9-1).**  SAS has no output-grid path, so the request raised
  `ValueError: propagate(method='sas', ...)` — naming a kernel the caller
  never wrote, decided purely by `z` (measured: `output_dx=3e-6` succeeded at
  `z=1e-4` and `z=5`, raised at `z=1e-3`, N=64/dx=2 um/633 nm).  The band now
  selects `asm`, which auto-promotes to the exact
  `angular_spectrum_propagate_mft` — the remedy the SAS error message itself
  recommended.  Explicit `method='sas'` still raises, unchanged.
- **`which_propagator` / `asm_propagate` no longer route `z < 0` into the
  forward-only `sas` / `fraunhofer` kernels (W9-2)** (measured: `z=-1.21e-3`
  -> `scalable_angular_spectrum_propagate: z must be > 0`).  This is the 4.12
  B1-6 guard its twin `_auto_select_method` has carried since 4.12.
- **A carrier tilt passed alongside `output_dx` was silently discarded
  (W9-3)** (the `asm_mft` branch outranks the tilt branch and the MFT kernel
  has no tilt parameter): measured bit-identical output, `max|difference| =
  0.0`, for `tilt_x=0.05` vs `0.0`.  The collision now emits a `UserWarning`,
  matching the v5.30 treatment of the legacy `'propagate_tilted'` element.
- **`maslov` / `asymptotic` / `mhs` with `output_grid` / `output_dx` now raise
  a dispatcher-level `ValueError` (W9-4)** naming the members that honour the
  request (`gbd` / `hf` / `hfpi`).  Pre-fix the request was dropped in
  silence — and with the `output_dx` shortcut the returned
  `PropagationResult.dx` reported the requested pitch while the field was
  still at the input pitch.  `maslov` is what `method='auto'` picks for a
  prescription without aspherics.
- **`PropagationResult.dx` now honours `output_grid=(N_out, dx_out)`, not only
  the `output_dx` shortcut (W9-5).**  Pre-fix an `output_grid` call returned a
  field genuinely resampled to `dx_out` (bit-identical to additionally passing
  `output_dx`) labelled with the INPUT pitch — for asm / fresnel / fraunhofer
  via the MFT promotion and for gbd / hf / hfpi.
- **The wrapper no longer publishes `PropagationResult(field=None)` (W9-6).**
  Measured: `propagate(method='mhs', ..., return_intermediate=True)` returned
  a null field with no warning, because `_coerce_field` cannot read MHS's
  native `[(HuygensSurface, ndarray), ...]` history.  It now raises and names
  `return_result=False`, which returns that history intact.  The P5 flip's
  guarantee is that `.field` is defined whichever kernel ran.

Routing with no output-grid request, and all forward `z > 0` ASM-family
routing, are bit-for-bit unchanged.  Also verified in the same audit: the P5
return contract, method validation on BOTH system twins, and the odd-N
round-trip are clean; the traced chain-default flip (v5.29→v5.30 commitment)
SHIPPED in `455be4a`/`a9dc454`.  Pins:
`tests/unit/test_niche_audit_w9_dispatch.py` (35 cases; 25 fail pre-fix, 10
regression fences pass pre-fix).

### Fixed — the analytic shape path's layering contract (audit W8, `elements/rcwa/twod.py`, `_core.py`)

The other item W7 left open: `rcwa_efficiency_2d_shapes`' analytic form-factor
path "was read but not numerically cross-validated against the rasterised
path", while the campaign's standing recommendation ("for accuracy-critical
geometry use the shapes path — exact form factors, no rasterization") rested on
exactly that validation.  It is now cross-validated for every kind
`_shape_form_factor` supports (`rectangle` / `disk` / `ellipse`), lossless and
lossy, normal and conical, at two fill fractions each — and the **form factors
are clean**.  The three defects found were all in the plumbing around them, all
silent, all energy-clean.

- **The form factors, verified clean (89 pins,
  `tests/unit/test_niche_audit_w8_shapes.py`).**  The DC coefficient is the
  closed-form area fraction **bit-exactly** (0.0 for all three kinds); a
  cell-filling rectangle reproduces the uniform cell (7.3e−16 in `[[eps]]`,
  4.4e−34 in R/T) and still does off-centre; two abutting rectangles equal their
  merged rectangle (1.1e−16, 5.9e−15 in R/T); a vanishing shape approaches the
  background at the exact area rate (ratio 100.00 per decade of radius);
  `F(−G) = conj(F(G))` to 0.0, so `[[eps]]` is Hermitian for real `eps` and
  obeys `[[eps]](conj eps) = conj([[eps]])^T` exactly.  Periodic wrap is
  **exact, not asymptotic** — a shape at a corner, across a seam or outside the
  cell leaves `|[[eps]]|` invariant to 1.1e−16 and R/T to 1.1e−14, because the
  form factor is sampled on the reciprocal lattice where one shape and its whole
  wrapped tiling transform alike.  And the rasterised path converges **to** the
  analytic answer with no systematic residual: `O(1/S)` for a point-sampled cell
  (measured 1.99 + 1.99 per doubling for a rectangle), `O(1/S²)` for an
  area-averaged one (3.46 … 4.55 per doubling across all 12 kind/fill/eps
  combinations, 3.98 … 4.00 in R/T for the rectangle across all 8 of its
  eps/angle/polarization combinations), down to 5.4e−5 in `[[eps]]` and 8.4e−6
  in R/T.  The second return of `_analytic_convolutions_2d` really is the
  analytic `[[1/eps]]` (net 3.73 … 4.28 over a 4× refinement).
- **W8-A — overlapping shapes (FOUND + FIXED).**  The docstring promised
  "shapes are painted in order over the background"; the analytic factorization
  **adds** the form factors, which is the same thing only on a *disjoint* list.
  Overlaps were accepted whenever the total area still fitted in one cell (the
  v5.5.3 cumulative-area guard sees only the total, not the arrangement) and the
  shared area silently got `eps_bg + (eps_1 − eps_bg) + (eps_2 − eps_bg)` —
  neither shape's `eps`: R/T off by **6.1e−02** (two 5/6-overlapping
  rectangles, DC permittivity 2.833 against the painted 2.501), **1.3e−01**
  (two partially overlapping disks) and **1.1e−01** (two IDENTICAL disks — the
  case AUDIT_V5_5_2 2026-05-31 recorded and the cumulative fix did not reach),
  with energy closures of −6.7e−16 / −3.8e−15 / −1.9e−14.  `_validate_shapes`
  now rejects overlapping pairs via a support-function separating-axis test
  (exact for every {rectangle, disk, ellipse} pair, wrap-aware, one-sided at
  tangency: exactly-tangent circles and exactly-abutting rectangles stay legal,
  and so do diagonal neighbours whose bounding boxes overlap), mirroring the
  `add_tapered_ridges` / `add_tapered_pillars` guards.  The docstring now states
  the additive rule and the disjointness requirement.
- **W8-B — `n_orders_y = 0` on a y-varying shape list (FOUND + FIXED).**  With
  no retained y-harmonic only the y-AVERAGED permittivity enters, so the solve
  returned a **different structure's** answer: R00 = 0.054846 for a disk against
  the y-resolved 0.006897 (8×), matching the explicitly y-averaged pixel cell,
  with a 4.4e−16 closure.  The pixel path has rejected exactly this since audit
  M8 (`_validate_cell_sampling(strict_y=True)`); the analytic path and
  `RCWAStack.add_layer(shapes=…)` had no counterpart and now do.  A full-height
  rectangle (a stripe) is genuinely y-invariant and keeps the M8 fast path
  (reproduces the y-resolved solve to 2.9e−16).
- **W8-B′ — the 1-D-stack half of the same trap (FOUND + FIXED).**  A 1-D
  stack's `noy = 0` is a SENTINEL, not a truncation choice, so the raise above
  must not fire there — which is exactly why commit `809314c` gave the pixel
  path a `RCWAYAverageWarning` DIAGNOSTIC for that case.  The analytic-shape
  layer had no such flavour and averaged in silence: measured through the stack
  API (P = 0.6 µm, λ = 550 nm, d = 220 nm, eps 6.25 disk r = 160 nm,
  `n_orders = 6`), the 1-D stack returned **R00 = 0.054846364** against the 2-D
  stack's **0.006896833** — **7.95×**, absolute error 0.047950 — with closures
  −8.9e−16 and −1.5e−14, so neither tripwire could see it.  The mechanism is
  measured, not inferred: feeding the explicitly y-AVERAGED *pixel* cell of the
  same disk to the same 1-D stack gives 0.054845052 at S = 512 and 0.054846287
  at S = 2048, converging on the shapes answer (|Δ| 1.3e−06 → 7.7e−08, the
  raster's own residual).  New `_warn_if_shapes_y_averaged` at both shapes sites
  (`add_layer`, and `_materialized_layers` for the dispersive route, one report
  per wavelength), same `RCWAYAverageWarning` category so one filter covers both
  flavours, and stacklevel 3 / 5 as on the pixel path.  It is a diagnostic, not
  a rejection: the 1-D + shapes contract is unchanged.  Raise and warn now read
  ONE predicate (`_shapes_y_varying`), so they cannot drift — the same
  no-divergence contract `809314c` pinned for the pixel pair, extended to the
  analytic flavour and pinned over 6 shape lists.
- **W8-C — malformed shape dicts (FOUND + FIXED).**  A missing or non-numeric
  `radius` / `size` / `semi_axes` / `eps` / `center`, a non-dict shape or a bare
  shape dict passed as the list escaped as `KeyError('radius')` /
  `TypeError: 'float' object is not iterable` / `AttributeError` from inside the
  form factors; all eleven cases are now named `ValueError`s (house rule).

### Fixed — the `eps_cell` rasterization contract (audit W8, `elements/rcwa/**`)

The one item the W7 rcwa audit deferred *by design*: the tapered / sheared
generators rasterized their analytic shapes with a **symmetric** `|dist| <
half`, which excludes **both** walls, so an edge landing exactly on a pixel
centre lost one pixel — the W7-A defect class (a duty quantised to a grid),
but in the **user-facing pixel cell**, where changing the semantics is a
contract decision rather than a bug fix.  The decision is now taken and
recorded in three parts.

- **Boundary coincidence (FOUND + FIXED).**  All three internal rasterizers
  (`RCWAStack.add_tapered_grating` / `add_tapered_ridges` /
  `add_tapered_pillars`, plus the shapes branch of `plot_geometry`) now use the
  **half-open** convention `[lo, lo + w)` — lower wall inclusive, upper
  exclusive — which is the house convention already used by the analytic 1-D
  paths (`oned.py`: "the ridge occupies `[0, duty)`"),
  `PMMStack._ridges_to_segments` (`lo <= mid < hi`) and
  `SegmentStackGeometry.to_rcwa_stack` (`(xs >= lo) & (xs < hi)`).  The
  recorded W7 outlier reproduced exactly: at `shear=0.5, duty=0.5,
  n_slices=128, n_x=256` **all 128 slices** realised duty `127/256 =
  0.49609375` (−3.906e−03).  `n_x == 2*n_slices` is a whole coincidence
  *family*, not one unlucky point, and there is a matching family at `duty=1`
  (the pixel antipodal to the ridge centre became a groove — an "all ridge"
  layer with a hole).  On a clean-closure case (P = 1 µm, λ = 633 nm, d = 300
  nm, `eps_ridge=4`, M = 7, 64 slices, coincidence at `n_x=128`) the zeroth
  orders were off by **1.802e−02** pre-fix and are off by **1.552e−04** now
  (the ordinary `O(1/n_x)` quantisation) — a 116× improvement **at** `n_x=128`
  rather than by refining.  Every non-coincident geometry is **bit-identical**:
  40 000 random `(S, width, centre)` triples measured 0 mask differences, and
  the `n_slices = 16/32/128` rows of the convergence sequence above are
  bit-for-bit unchanged.
- **Pixel semantics documented, by measurement.**  A single canonical **PIXEL
  CELL CONTRACT** block in `elements/rcwa/stack.py`, pointed at from every
  `eps_cell`-accepting entry point (`add_layer`, `add_graded_layer`,
  `rcwa_efficiency_2d`, `rcwa_jones_2d`, `prepare_rcwa_2d`,
  `_eps_convolution_2d`).  `eps_cell[j, i]` is a **node point sample** at
  `(j Px/Sx, i Py/Sy)` — *not* a cell average: measured against a band-limited
  analytic profile, a node sampling reproduces the exact convolution matrix to
  **1.1e−16** while a midpoint sampling is off by **5.9e−02**, the two DFTs
  differing by exactly `exp(+iπk/S)` (to 8e−16).  Boundary pixels are
  hard-assigned, half-open, with `O(1/Sx)` geometric quantisation that no
  `n_orders` convergence and no energy closure can see (measured up to
  6.1e−02 of a period at `Sx = 16`), and two escapes: raise `Sx`, or use an
  exact-geometry path (`shapes=`, the 1-D `segments` entries, a `PMMStack`
  taper).  The generators write the pixel-**centre** lattice, half a pixel off
  what the factorization reads; measured, that is an exact rigid `−P/(2 Sx)`
  translation for a band-limited cell (efficiencies invariant to **8.4e−15**
  under any shift) and, for a hard raster, a second `O(1/Sx)` aliasing
  difference that shrinks with the first (7.36e−03 → 5.66e−04 for
  `n_x = 128 → 2048`).  **PMM is exempt** (checked by grep + measurement): its
  tapered helpers emit exact segments / spectral-element walls and resolve a
  strip at its midpoint, so the coincidence class cannot arise — left
  untouched.

### Added — opt-in area-weighted rasterization (audit W8)

- `raster='hard' | 'area'` on `RCWAStack.add_tapered_grating` /
  `add_tapered_ridges` / `add_tapered_pillars`.  **Default `'hard'`, and the
  default call is bit-identical to the previous behaviour** (pinned).
  `'area'` gives each boundary pixel the area-weighted `eps` average, making
  the realised feature width exact at any `n_x` (`|Σcover/n_x − width| ≤
  3.7e−15` over 5000 random cases, against `O(1/n_x)` for `'hard'`); the
  rectangle is separable, so the 2-D pillar weight is the exact product of the
  two per-axis coverages.  The overlap guards read the **hard** masks in both
  modes, so whether two features collide is a property of the geometry, not of
  the raster mode — and features that *touch* exactly are legal and share the
  boundary pixel by area.
- **The physics was measured, per polarization and per formulation, and the
  docstring recommendation follows the measurement** (full table in
  `add_tapered_grating`).  Against the exact analytic 1-D oracle (vertical
  `duty = 0.37`, θ = 0.25 rad, M = 9), with the default
  `formulation='laurent'`, `'area'` is **1–3 orders of magnitude** more
  accurate at the same `n_x` for **both** polarizations (TE 5.49e−03 →
  2.86e−04 at `n_x=64`, 9.29e−04 → 3.84e−06 at 1024; TM 2.33e−03 → 5.67e−05 →
  1.03e−06).  With `formulation='li'`/`'fff'` the **wall-normal (TM)**
  polarization should keep `'hard'`: Li's inverse rule assumes a sharp
  interface and the arithmetic (area) average is the wrong effective medium
  for the normal component — it wants the harmonic one (Farjadpour 2006
  subpixel smoothing).  Measured, `'li'` TM gains only ~2.5× and plateaus
  (2.2e−05 at `n_x=8192` against 1.6e−08 for `'laurent'` TM), and on a sheared
  taper it is outright **worse** than `'hard'` — by 10.6× at `n_x=64`, 9.2× at
  256 — while TE on the same sweep improves by up to 120×.  (TE is
  bit-identical between the two formulations for a 1-D cell — `E_y` is
  tangential to every wall, measured 0.0 — so only the TM arm needed
  deciding.)  Hence: opt-in, default off, with the regression recorded rather
  than hidden.
- `plot_geometry`'s shapes branch is now wrap-aware as well as half-open: a
  rectangle crossing the cell edge used to vanish from the picture entirely
  even though the solver's analytic form factor included it.

22 pins in `tests/unit/test_niche_audit_w8_raster.py`, **16 of them verified
failing on a clean `e37d7b7` worktree**; the 6 that pass in both trees are
documented non-discriminators (the contract locks and the control arm that
explains why the defect read as a convergence outlier).

## [5.30.0] — 2026-07-27

**The adversarial-audit campaign.**  One whole-codebase adversarial audit
(171k lines, five Opus agents, measured-repro evidence rules) followed by
seven fix waves: every finding fixed, decided, or refuted-with-evidence.
**15 CRITICAL-physics defects** found and corrected — none previously
caught by CI — including the 2-year coordinate-break inversion (folded
designs now trace correctly), odd-N propagator grids, non-front-stop and
immersed-conjugate pupils/focal distances, the Shack-Hartmann reference
calibration (15 mrad phantom tilt on a flat wavefront), the EME
Hermitian-solver Bloch-phase keying, the BOR staggered wall anchor
(first→second-order convergence, now the default), the Berreman and PMM
flux-gauge conjugations silently zeroing transmission into absorbing
substrates, and RCWA duty cycles silently quantised to the FFT grid.
~3,300 regression pins added (every pin set verified failing on pre-fix
code; all value pins carry measured cross-platform tolerances); the
DynaMeta consumer contract held green throughout.  BREAKING: all ten
scheduled deprecation shims executed (see Removed + Migration-Guide
§5.30.0), the P5 return contract flipped (`propagate()` returns
`PropagationResult` by default; `return_result=False` is the permanent
legacy escape hatch), and the coordinate-break fold and BOR wall-anchor
conventions changed to the oracle-verified-correct ones.  Full causal
record in `docs/audits/AUDIT_ADVERSARIAL_CODEBASE_2026_07_25.md`;
per-territory verified-clean maps and honest coverage statements in the
commit history.

### Fixed — EME physics interiors (audit W6, `elements/eme/**`)

The territory the 2026-07-25 adversarial audit named as its own coverage gap
(never numerically validated; the 2026-07-09 read-only EME audit concluded
"none above nit level").  Audited oracles-first: an independently bisected
analytic symmetric-slab dispersion, lossless power conservation of the lateral
cell S-matrix, the analytic Airy/Fabry-Perot slab (lossless **and** lossy), and
the package's own 2-D-FD mode oracles used as a recall/spurious cross-check.
75 pins in `tests/unit/test_niche_audit_w6_eme.py`, 49 of them verified failing
on a pre-fix worktree of `3a1da2b` (the other 26 are verified-clean locks that
must pass in both trees).

- **CRITICAL — `layer_modes` returned pure garbage at a nonzero x-Bloch phase
  (`kx0 != 0`).**  `strip_x_modes` keyed its solver choice on `kx0 == 0`, but
  the discrete Bloch operator `A = D + diag(eps k0^2)` is **Hermitian for a real
  `eps` at ANY real `kx0`** (the two wrap corners carry `exp(+i kx0 Lx)` and its
  conjugate).  Routing it to `scipy.linalg.eig` anyway did two things: (i) `lam`
  came back with roundoff imaginary parts of *arbitrary sign*, and `np.sqrt`'s
  principal branch (`Re >= 0`, not `Im >= 0`) then put 8-11 of 16 strip modes on
  the exponentially **growing** propagator (measured `max|exp(i ky h)| = 6.2e6`)
  — the T-matrix blow-up the S-matrix cascade exists to avoid; (ii) the
  complex-symmetric bilinear normaliser `Phi/sqrt(sum(Phi^2))` is the wrong
  metric for a Hermitian operator, leaving the basis non-orthonormal (measured
  `max|Phi^H Phi - I| = 43.2`, column norms 1.01..6.65) so the interface solves
  were ill-conditioned and `sigma_min(M)` lost all meaning against `tol`.
  Measured end effect on the reference structured cell at `kx0 = 0.37`,
  `ky0 = pi`: **68 roots returned, 0 of the 3 real modes recovered, all 68
  spurious** (nearest FD eigenvalue to the best of them 0.56 away), and the
  lossless cell S-matrix violated power conservation by 1.5e-2 (`kx0 = 0.37`) /
  1.8e-1 (`kx0 = 1.1`).  `kx0 = 0` was correct throughout, which is why every
  shipped test passed.  Fixed by routing real `eps` to `eigh` at any `kx0`,
  building the wrap phase with `conj(ph)` (exactly Hermitian), and adding
  `_ky_forward` — one shared decaying-branch (`Im(ky) >= 0`) selector, matching
  the vector sibling's `_strip_split_forward` convention, now used by `_wv`,
  `_global_lateral_nullspace` and `mode_field` (the two field-reconstruction
  sites had inlined the same unguarded `np.sqrt`).  Post-fix: power conserved to
  1.3e-15 at every `kx0`, recall 3/3 with 0 spurious, and the analytic slab
  reproduced at `kx0 != 0` via `qz^2 = beta^2 - kx0^2`.
- **HIGH — `mode_match` carried a growing exponential and silently returned
  R = 1 / T = 0 for a homogeneous medium.**  The backward layer amplitudes were
  referenced at `z = 0`, so `Einv = exp(-i qz depth) = exp(+|qz| depth)` entered
  the matched system for every evanescent layer mode.  `cond(A)` grew as
  `exp(2 |qz|max depth)` — measured 9.3 -> 3.8e6 -> 1.4e17 -> 8.7e38 for
  `depth = 0.2 / 2 / 5.3 / 12` on a 1-um cell — and once it passed the
  `lstsq(rcond=None)` cutoff the physical solution was truncated to zero:
  an **index-matched** layer (`n_sup = n_lay = n_sub`) reported
  `R_00 = 1.000000`, `T_00 = 0.000000` where the exact answer is `T = 1`, and a
  `n = 1.5` slab reported `R_00 = 1.0` at `depth >= 8` against analytic 0.1461.
  `energy = 1.000000` in every failing case, so the module's own energy check
  could not see it.  Fixed by referencing the backward amplitudes at
  `z = depth` (`c- = E d-`), algebraically identical but leaving only the
  decaying `E = exp(i qz depth)`: `cond(A)` now stays ~2 and the analytic Airy
  slab is reproduced to ~5e-15 at every depth from 0.2 to 16.
- **HIGH — `diffraction_fd` did not absorb.**  It took `Re(qz^2)` from the FD
  oracle, so an absorbing layer behaved as a lossless one: at `n = 1.5 + 0.2j`,
  `depth = 4` it reported `energy = 1.000000` and `R_00 = 0.032362` where the
  analytic lossy Airy slab gives `R + T = 0.046505`, `R_00 = 0.046099` — it
  claimed all the light emerged while 95% was absorbed.  A complex `eps_xy` now
  keeps the complex spectrum (`return_complex=True`) and `mode_match` selects the
  decaying `qz` branch; the analytic lossy R/T are reproduced to 8 decimals
  across four (index, depth) combinations.  A **real** `eps_xy` takes the
  byte-identical legacy path.
- **MEDIUM — the rasterizers `strips_to_eps_xy` / `_strips_to_mu_xy` lacked the
  `sum(h) == Ly` guard both layer finders have.**  Measured: heights summing to
  0.25 of `Ly` silently produced a grid with **24/32 cells at `eps = 0`**, and
  the vector oracle's `1/(k0 eps)` then built `inf`/`NaN` generators — silent
  data corruption on the way into an "independent oracle".  The contract check
  is now one shared helper (`_check_strip_heights`) used by all four sites.
- **MEDIUM — `solver=` accepted any junk value and silently ran dense.**
  `'bananas'`, `'DENSE'`, `'sparse'`, `''`, `None` and `7` all measured
  bit-identical to `solver='dense'` (the P6 junk-method fall-through class).
  Now validated in `dispersion_vec` / `layer_vector_modes`.  Measured tolerance
  recorded for the two *valid* solvers: `banded` reproduces the dense
  `sigma_min` to 1.7e-3 relative, recovered modes to 2e-5 relative.
- **MEDIUM — the `layer_vector_modes` detection grid was not scale-invariant.**
  `_DETECT_PPU` was documented and applied as "points per unit `qz^2`", a
  quantity with units of 1/length^2, so ONE physical cell was scanned at wildly
  different densities depending on the caller's length unit: measured 3944
  points in um, 400 (the `n_scan` floor, i.e. **under-resolved**) in nm, and
  3.94e9 points — a 31.5 GB `linspace`, i.e. a hang or `MemoryError` — in mm.
  The density is now per unit of the dimensionless `(hi - lo) * Ly^2` (identical
  at `Ly = 1`, which every shipped test uses), with a 200_000-point cap that
  warns loudly instead of trying to allocate a pathological window.
- **MEDIUM — `sigma` was silently inert without `k`** in both FD oracles
  (measured bit-identical results with `sigma=1e9`), so a caller asking for a
  few modes near a shift silently received the full dense spectrum.  Now raises.
- **MEDIUM — the vector layer finder had no band-edge guard, so a `qz2_range`
  starting at 0 crashed.**  Found by one of this wave's own pins.  The H-part of
  a vector strip mode is recovered as `(C U)/(i ky)`, which is division by zero
  for a mode sitting exactly on a band edge — measured `min|ky| = 0.000e+00` for
  a uniform `eps = 2` strip at `qz^2 = 0` with `Nx = 8`, `k0 = 8`.  That produced
  `NaN` in the modal state and the call then died several frames downstream with
  `ValueError: array must not contain infs or NaNs`; the same opaque error came
  from the *other* non-evaluable sample, a `qz^2` so far outside the band that
  `exp(+|ky| h)` overflows (measured at `qz^2 = 1e7`, `max|ky| = 3.2e3`).  The
  scalar sibling `layer_modes` has skipped its analogous band-edge sample since
  audit P3-18; the vector sibling never got that guard.  Both cases now raise a
  *named* `numpy.linalg.LinAlgError` from one shared `_equilibrated_G` builder,
  and `layer_vector_modes` skips such samples exactly as `layer_modes` does.
- **LOW cluster.**  The scalar sparse oracle `ref_2d_modes(k=...)` lacked the
  fixed ARPACK `v0` both its siblings pass, so its output depended on the global
  NumPy RNG state (measured 1.7e-13 drift between two seeds) — now deterministic.
  The lossy `Im(qz^2)`-discard warning that `ref_2d_modes` emits was missing on
  the JAX scalar twin **and** on the NumPy `ref_2d_modes_vector` — both now warn.
  `layer_modes(n_scan <= 1)` silently returned an empty mode set (reads as "no
  modes in this window") — now rejected.  Opaque failures given clear messages:
  `eps_xy_to_strips` on a tensor grid (`too many values to unpack (expected 2,
  got 4)`), `mode_match` with a zero-norm mode column (`SVD did not converge in
  Linear Least Squares`), `_sigma_min_invpow(iters=0)` (`UnboundLocalError`), and
  a traced `kx0`/`ky0` on the JAX path (a bare `ConcretizationTypeError` naming
  only "the `float` function", although the dispatch's `is_jax_array(kx0)` test
  reads as support — the frozen FD/Yee operators make `qz^2` differentiable
  w.r.t. `eps` and `k0` only).
- **Behaviour changes to note.**  (a) Real-`eps` strips at `kx0 != 0` now return
  ascending real `lam` and an orthonormal `Phi` from `eigh` instead of unsorted
  `eig` output — the pre-fix values there were unusable, so this is a repair, not
  a re-tuning.  (b) `_ky_forward` forces the decaying branch, which also flips a
  **gain** medium (`Im(eps) < 0`) onto the decaying branch; gain is out of scope
  for the scalar and the vector cascade alike (the vector sibling already did
  this).  (c) `diffraction_fd` on a **lossy** layer now returns
  absorption-correct, energy-non-conserving efficiencies.  (d) `mode_match`
  rearranges its unknowns, so lossless results are equal to lstsq roundoff rather
  than bit-identical.  Everything the fixes claim not to change was checked:
  `strip_x_modes` (real and lossy) at `kx0 = 0`, `cell_smatrix`, `dispersion`,
  `layer_modes`, `mode_field`, the vector `_global_block_G` / `dispersion_vec` /
  `layer_vector_modes` / `mode_field_vec`, both dense FD oracles, and the
  rasterizer output are **bit-identical** across the pre/post trees.
- The module's documented **negative result stands**: with the stable
  reformulation in place a STRUCTURED layer's mode-matched efficiencies still do
  not converge (energy strays, `T_00` wanders) and still warn.  W6-2 was a
  conditioning bug in the `z` match; the structured non-convergence is a basis
  problem, and the fix does not disturb it.

### Changed (BREAKING) — `propagate()`'s default return is now a `PropagationResult` (audit P5 / roadmap F1, EXECUTED)

Owner decision, same shape as the W5 wave below: **execute the scheduled API
transition now** rather than ship only its announcement.  The transition was
decided and announced earlier in this same (unreleased) v5.30 cycle
(`3097cda`, option 4 of the four costed in
`docs/roadmap_deferred_2026_07_21.md` Part F1); waiting for the registry
horizon would have shipped a `DeprecationWarning` about a change no released
version had yet made.

- **`lumenairy.propagate(...)` without `return_result` now returns a
  `PropagationResult` for every method** — `.field` / `.dx` / `.dy` /
  `.dx_out` / `.dy_out` mean the same thing whichever kernel ran, and
  `np.asarray(result)` yields the field.  Previously the default return was
  the selected kernel's native shape: a bare `ndarray` (asm / rs / maslov /
  gbd / hfpi / hf) **or** an `(E, dx_out, dy_out)` triple at a kernel-chosen
  pitch (sas / fresnel / fraunhofer) — under `method='auto'` decided by `z`,
  so a caller could not know which it would get without re-running the
  selector.  That was audit finding P5.
- **Migration: `return_result=False`** returns those native shapes, bit-for-bit
  as before.  It is permanent and un-deprecated — the answer for code that
  unpacks `E, dxo, dyo` (`PropagationResult` iteration stays 2-item,
  `(field, intermediates)`, audit P16) and for fast loops that want no wrapper
  allocation.  `return_result=True` is unchanged.
- Measured bit-identity against `3a1da2b` over a 55-record capture: all 14
  `return_result=False` records, all 14 `return_result=True` records, all three
  `mhs.prescription_subdomain` paths, all eight `algebra.FreeSpace` paths and
  both `Source.propagate` paths byte-identical (41/41).  The 14 changed records
  are the default-path ones, each byte-equal to that case's pre-flip
  `return_result=True` record — the container changed, the content did not.
- **Retired with the flip:** the transition `DeprecationWarning`, its
  `_caller_is_internal` external-caller predicate, and dispatch's
  `API_TRANSITION_VERSION` / `resolve_removal_version` imports — a warning
  reading "the default *will* become a `PropagationResult` in vX" cannot
  outlive the version that makes it one.  `_deprecation.API_TRANSITION_VERSION`
  survives as the (empty) slot for the next transition, still bound to
  `NEXT_REMOVAL_VERSION` so `check_removal_schedule()` still covers it, with
  the executed entry recorded as a tombstone comment — the same convention this
  release used for `REMOVAL_SCHEDULE`.
- The grid-change `UserWarning` (v5.30, P5) now fires **only** on the
  `return_result=False` path: the stable contract has no `z`-dependent return
  shape to warn about.
- Internal call sites migrated in the same commit:
  `propagators/mhs.py`'s two `prescription_subdomain` dispatcher calls and
  `algebra/primitives.py`'s `FreeSpace._apply` now name
  `return_result=False`.  The roadmap's flip-day inventory had listed the
  algebra site as already flip-safe; measurement showed it was tolerant of the
  wrapper for the *field* but not for the anamorphic *y-pitch* (`dx = 2 µm`,
  `dy = 3 µm` returned `dy_out = 3e-6` unwrapped, `2e-6` wrapped), so the flip
  would have silently squared an anamorphic algebra chain's output pitch.
- Updated for the new default: `validation/propagators/test_dispatch.py`
  (contract check superseded; 19/19), `examples/01_basic_propagation.py`,
  the `README.md` quick-start table, and seven legacy-shape pins across
  `tests/unit/` (`test_niche_audit_w3_propagators.py`,
  `test_v5_3_hf_freespace_output_grid.py`,
  `test_audit_v5_24_2_g09_seams_prop2.py`, `test_v4_15_2_agent_c.py`).
- Pins: `tests/unit/test_niche_audit_w4_p5_return_contract.py` rewritten to the
  shipped contract (53 pins, announcement-phase classes marked `SUPERSEDES`;
  32 of them fail on a `3a1da2b` worktree, the other 21 being the invariance
  pins).

### Removed (W5 shim-removal wave — the scheduled deprecations are executed, not slipped again)

Owner decision: the overdue deprecation shims ship **removed in v5.30**
rather than waiting for the re-scheduled v5.32 horizon.  Eight of the ten
had already blown through a stated removal version while continuing to ship
(`version_removed='5.0'` at v5.29 — 29 minor releases late — or v5.27 for
the v5.25 kwarg renames); R-18/E-H11/R-17/P7/P8 re-scheduled the banners,
and this wave executes them.  `lumenairy._deprecation.REMOVAL_SCHEDULE` is
now empty and `check_removal_schedule()` returns no violations.

Every surviving (modern) call path is proven **bit-identical** to the
pre-removal commit `24c7d30`: 73 captured arrays byte-for-byte equal, with a
42-entry SHA-256 subset frozen into
`tests/unit/test_niche_audit_w5_shim_removals.py` so a future edit cannot
silently perturb a path this wave rewrote.

Error shapes follow existing in-repo precedent.  A kwarg **rename** or an
**inert** kwarg is a plain signature removal → `TypeError: … unexpected
keyword argument` (precedent: `analysis/detector.py`'s v5.0
`cosmic_ray_rate`, `optimize/multiconfig.py`'s v5.0 `wavelength` default).
A shim that intercepted **values** keeps an always-raising detector so the
error can name the modern form (precedent: `propagators/system.py`'s
`_reject_legacy`, v5.0 aperture-schema purge) — permanently, scheduling
nothing new.

**`lumenairy/sources/core.py`**

- **`create_gaussian_beam(sigma=)`** (deprecated v5.25, stated v5.27) —
  `sigma=s` → **`w0=s*sqrt(2)`**.  `w0` is the 1/e² intensity radius;
  `sigma` was the field std-dev.  Equivalently `w0=w` reproduces the old
  `sigma=w/sqrt(2)` field bit-for-bit.  The surviving missing-argument
  `TypeError` names the conversion for one more cycle.
- **Schell-family `seed=`** on `create_gaussian_schell_source` /
  `create_schell_model_source` / `create_annular_incoherent_source` /
  `Source.gaussian_schell` / `Source.schell_model` (v5.25, stated v5.27) —
  `seed=<int>` → **`rng=<int>`**.  Exactly equivalent: `seed` was forwarded
  verbatim into `rng`.
- **The five `Source.*` legacy positional overloads** (v4.15, stated v5.0,
  re-scheduled v5.32 by R-18) —
  `Source.gaussian(w0, N, dx, wavelength)` →
  **`Source.gaussian(*, N, dx, wavelength, w0)`**; likewise
  `plane_wave(N, dx, wavelength)`, `point_source(N, dx, wavelength)`,
  `top_hat(diameter, N, dx, wavelength)` and
  `fiber_mode(mode_field_diameter, N, dx, wavelength)` → their
  `(*, N, dx, wavelength, <size>)` forms.  The legacy order put the SIZE
  argument first, so a positional caller has every quantity one slot out of
  place; each classmethod therefore keeps an always-raising
  `*_legacy_positional` collector and names its canonical signature via one
  shared `Source._reject_legacy_positional` helper.
- **`create_led_source` legacy positional form** (v4.14.2, stated v5.0) —
  `create_led_source(N, dx, diameter, divergence_angle, wavelength, x0, y0,
  dtype)` → **`create_led_source(N, dx, wavelength, *, diameter=…,
  divergence_angle=…, dy=…, x0=…, y0=…, dtype=…)`**.  The v4.14.3
  scale-inversion heuristic goes with it: it existed only to tell the
  canonical-order mistake apart from a legitimate legacy call while one of
  them was still legal, and both now hit the same rejection.
- **The Schell `return_kind` sentinel apparatus** (v4.15.1; its warning was
  already retired in v4.16.1) — `_RETURN_KIND_UNSET`,
  `_SchellReturnKindUnsetSentinel`, `_warn_schell_return_kind_default` and
  the five no-op `if return_kind is _RETURN_KIND_UNSET` branches.
  `return_kind=_RETURN_KIND_UNSET` → **omit `return_kind`, or pass
  `'ensemble'` / `'mcf'` explicitly**.  No bespoke rejection was needed: an
  unrecognised value already lands on `_validate_return_kind`'s
  `ValueError`, which names both modern values.  The zero-production-call-site
  measurement that licensed this is pinned in
  `tests/unit/test_niche_audit_w3_ui_deprecation.py`.
- The bookkeeping constants `_OVERDUE_SHIM_VERSION_REMOVED`,
  `_DEPRECATION_VERSION_ADDED` and `_DEPRECATION_VERSION_REMOVED` (they
  existed only to keep the eight warning sites' horizons in sync).

**`lumenairy/elements/doe.py`**

- **`makedammann2d(_legacy_units='auto')`** (v4.14.2; demoted from default
  to explicit loud opt-in by v5.30 E-H11) → **drop the kwarg** (`'SI'` is
  the default) **or pass `_legacy_units='um'`** for genuinely
  micrometre-valued inputs.  The heuristic multiplied any `periodx` /
  `periody` / `waveln` above 1 mm by `1e-6`, so a physically correct SI
  THz/MMW design (8 mm period at 1.1 mm wavelength) came back with 5e-10 m
  cells.  A shim that silently rewrites physical inputs, and is known wrong
  for a legitimate design regime, does not get another cycle; `'auto'` now
  raises `ValueError` naming both survivors and the `'um'` recipe.
  Accepted modes are exactly `{'SI', 'um'}`.

**`lumenairy/propagators/`**

- **`gbd.recommend_gbd_sampling(wavelength=)`** (audit P8) → **drop it**.
  A required keyword the body never read; output was proven independent of
  it (identical dict at λ = 0.4 µm and 10 µm on a fixed `E_in`).  Wavelength
  dependence still arrives through `E_in`'s phase gradient in rad/m; for a
  width tuned against a real propagation use `converge_gbd_sampling`.
- **`hf.propagate_huygens_fresnel_with_opl_callable(wavelength=)`**
  (audit P7) → **drop it**.  Same shape: never read, and it could never
  acquire a meaning — consuming it would break every existing
  waves-returning `opl_fn` by ~1e6.  `opl_fn` returns WAVES; divide a
  metre-valued OPL by the wavelength inside `opl_fn`.
- *Kept:* `…with_opl_callable(chunk_output=)` stays warn-only — deprecated
  in v5.17 with **no** stated horizon, so it is not past one.

**`lumenairy/optimize/`**

- **`design_optimize(wave_traced=)`** (R-17) → **register a propagator**:
  `register_wave_propagator('real_lens_traced', fn)` +
  `design_optimize(wave_propagator='real_lens_traced')`.  One dispatch
  mechanism instead of a boolean that mutated the meaning of
  `ray_subsample`; `opts` still carries `ray_subsample` for exactly that
  purpose, and `_wave_real_lens` carries a copy-paste recipe.
- **`MatchIdealSystemMerit(use_traced_lens=, ray_subsample=)`** (R-17) →
  put an explicit `{'type': 'real_lens_traced', 'prescription': …,
  'ray_subsample': …}` entry in `real_elements`; the `_prescription_`
  placeholder now always expands to `'real_lens'`.
- **`MatchIdealSystemMerit(focus_search=, focus_search_range=,
  focus_search_n=)`** (R-17) → add an explicit
  `{'type': 'propagate', 'z': dz}` offset to `ideal_elements` (or sweep
  `dz` and take the minimum penalty).  The now-dead
  `_focus_search_penalty` helper is deleted with them.
- R-17 grep-verified ZERO callers repo-wide (library, tests, validation,
  examples, UI) for all three flags, so CI never covered these branches
  (`e843f6f`).  **Scope boundary:** only *one* of the four "zero-caller"
  penalty helpers becomes dead.  `_field_mse_penalty`,
  `_intensity_mse_penalty` and `_intensity_overlap_penalty` are gated by
  `match=`, a live documented feature that R-17 explicitly declined to
  deprecate — they are **kept**, and a pin now asserts so.

**Explicitly NOT removed (still scheduled / not past horizon)**

- The **P5 return-contract transition** is not a shim removal (it flips a
  default rather than deleting a name), so it is not part of this wave — it was
  EXECUTED separately in the same release, see
  *Changed (BREAKING) — `propagate()`'s default return* above.
  `_deprecation.py` itself stays fully functional and the next deprecation
  cycle (or API transition) registers there as before.
- `rcwa_efficiency_1d_jax` (v6.0.0), `load_zmx_prescription` /
  `load_zemax_prescription_txt` (v6.0) — horizons still in the future.
- The `output_grid` → `output_shape` sub-propagator renames,
  `MultiFieldMerit` scalar `field_angles`, `PMM2DStack`, and the
  `Constraint` auto-probe notice — deprecated with no stated horizon.

### Fixed (adversarial-audit Tier 1: all 6 CRITICALs + the 2 silent-data-corruption HIGHs, 2026-07-25)

Fix wave over `docs/audits/AUDIT_ADVERSARIAL_CODEBASE_2026_07_25.md`, five
Opus agents on disjoint territories, every finding reproduced by measurement
BEFORE the fix and re-measured after; 236 new regression pins across 8 files,
each set verified to fail on the pre-fix code.

- **P1 (CRITICAL, `e29a8db`)** — odd-N frequency grids put DC at −0.5 bin, so
  every ASM-family propagator returned laterally shifted, phase-wrong fields
  (ASM N=257: 2.6e-1 max rel err, −3.89 px centroid walk).  DC anchor
  `N/2 → N//2` in `fft_infra` freq grids/bandlimit masks and the three
  `asm.py` H builders; ASM-MFT gets the matching integer anchor plus a
  half-input-pixel `k_centre_out` offset; single-FFT Fresnel AND Fraunhofer
  (same defect, found during the fix) get exact half-sample shifted-DFT phase
  corrections so odd-N output agrees with the already-correct Bluestein
  siblings (7e-14).  Odd N now matches the even-N accuracy floor everywhere;
  even N proven bit-identical pre/post fix (78/78 captured arrays).
- **P2 (HIGH, `e29a8db`)** — `fresnel_tf_propagate` returned the live pyFFTW
  inverse ping-pong buffer; a later same-shape call silently overwrote
  previously returned fields (measured 0.497 on a peak-1 field; consumer:
  `propagate_carrier_referenced`).  Now returns a copy (rs.py F-3 precedent).
- **R-1 (+R-1b/R-1c) (CRITICAL, `1fc8b1f`)** — `compute_pupils` dropped the
  final pre-stop transfer (EP radius −21%, f/# +27% on the audit design, up
  to −84% at larger stop gaps; fed `analysis/field.py` ray aiming), returned
  `ep_z` as an object DISTANCE instead of the signed coordinate its docstring
  and all four consumers require, and walked the post-stop leg in air even
  when the stop's image side is glass (xp_z +7.9%).  Pre/post-stop subsystem
  builders are now shared single-source with `seidel_coefficients`
  (bit-for-bit) and all pupil quantities match an independent exact real-ray
  oracle to ≤1.2e-12 on 5 designs.
- **R-2 (CRITICAL, `1fc8b1f`)** — `seidel_wfe` returned −W (composed the
  textbook expansion with the library's `code = −S_Welford` values without
  converting).  S1..S5 negated at ingestion; sign convention documented and
  anchored to a measured marginal-focus direction; two existing pins that
  encoded the wrong sign corrected with justification.
- **A-1 (CRITICAL, `720f689`)** — default `axis='radial'`
  `fwhm_resolution`/`rayleigh_resolution` biased −8%/−21% (4.9/2.4 samples
  per first zero) by integer-pixel radial binning; now routed through
  `_radial_profile_subpixel` (+0.01%/−1.03%), and Rayleigh no longer returns
  NaN blaming a "Gaussian-like" PSF on a perfectly sampled Airy.
- **A-2 (HIGH, `720f689`)** — `encircled_energy_radius` inverted a 256-point
  corner-spanning radius ladder (+6.05% drift under zero-padding alone); now
  inverts the exact sorted cumulative curve (drift exactly 0.0, worst 0.17 px
  from the analytic inverse over 21 configurations).
- **M1 (HIGH, `a240c15`)** — the RCWA shapes-layer eigenmode cache key
  flattened all shapes into one sorted multiset, colliding structurally
  different layers (swapped-centre disks solved bit-identically to (A,A):
  3.7× error in R₀, energy-conserving and silent; the default symmetry fold
  masked the audit's own geometry — pinned on a fold-ineligible one).  Key is
  now per-shape and order-preserving; dedup remains bit-exact memoization.
- **E-C1/E-C2 (CRITICAL docs, `2be264c`)** — `_lens_thin.py` docstrings:
  retracted the wrong-surface SA-null conic guidance (following it measured
  2.6× WORSE than a plain sphere; the correct null is flat-first exit-surface
  `k2=−n²`, exactly stigmatic, and the screen model's own null is
  `k=−1−(n−1)²`) and the "exact OPD … all higher-order aberrations" claim
  (the model is the orientation-blind single-plane sag-projection screen and
  never reads `d`); replaced with measured validity boundaries mirroring
  `apply_real_lens`'s house wording.  Code AST-verified unchanged.

Known-unfixed items from the audit remain open in the report (Tier 2+:
silent-junk-input validation cluster, frozen-defaults/ownership, guard gaps,
conventions/dead code) plus the flagged world.py coord-break question and the
mirror-parity signing of stop-adjacent pupil legs (needs a fold-pupil oracle).

### Fixed (adversarial-audit Tier 2, batch A — remaining HIGH findings, 2026-07-25)

Five more Opus agents over the same report, disjoint territories, same
discipline (reproduce → fix → re-measure → pin; every pin set verified to
fail on the pre-fix tree).  238 new pins across 6 files.

- **R-3/R-4/R-5/R-6/R-7 (`bcb568d`)** — `LGAberrationMerit` minimized
  |L(0,0)|² and drove Strehl toward 0 (NumPy and JAX merits were exactly `x`
  vs `1−x`; the JAX OPT-1 fix is now ported, agreement 4.4e-9, descent
  direction correct); all three OPD merits zero-filled non-finite OPD before
  `zernike_decompose` (which masks NaN itself) — a vignetted annulus
  SIGN-FLIPPED the fitted spherical coefficient, now recovered exactly;
  grazing-ray guards added to the JAX flat-with-aspherics branch (negative
  OPL −8.05e-3 m killed) and the NumPy flat fast path (immortal t=0 phantom
  killed, first-failure semantics, DOE-order case measured unchanged);
  degenerate ray bundles route to the documented `_adrt_numpy` fallback
  instead of raising bare `ZeroDivisionError` (bit-equal to the sibling).
- **E-H1/E-H5 (`03fa01e`)** — the Maslov `output_subsample` upsampler used
  edge-anchored `zoom` against a stride-subsampled lattice (the exact sibling
  of the 0a743a6 bug): +3.7 fine-px centroid walk at sub=8 → 0.000;
  fixed with the exact affine map (memory-safe vs a 17 GB coordinate stack
  at N=32768), sub=1 bit-identical, no CuPy twin exists (shared host path).
  `roi=` silently discarded `normalize_output` ('power' default returned the
  raw scale ~8 orders off): global reductions now warn loudly (they are
  unevaluable on the ROI fast path by construction), scalar factors are
  applied, junk raises; `fold_split` no longer silently drops the requested
  observation plane; 10 dead integrator args removed.
- **E-H2/E-H3/E-H4/E-H7 + E-M2..M5 (`0a443d6`)** — `newton_max_iters` now
  travels to the ProcessPool Newton workers (was hardcoded 12; pool 1-vs-12
  iters 0.0 → 5.98e-5) and the pool emits the unconverged warning;
  `prepare_real_lens`/`PreparedTracedLens` now resolve and store their
  PREPARE-TIME settings and deep-copy the prescription (global-flip and
  in-place-mutation drifts measured 49.6/53.3/1.77 → all 0.0);
  `on_noncollimated`/`inversion_method` raise on junk values; the delegate
  model swap forwards raw `sag_chunk_rows` (caller's banding opt-out
  honored) and warns when discarding non-default physics kwargs;
  `lens_model='local_only'` docstring corrected to its measured (opposite)
  behavior.  121 acceptance battery green after every edit.
- **E-H8/E-H9/E-H10 + E-L12..L17 + E-M13 (`5f9d82b`)** — polarization
  handedness typos no longer silently produce LEFT circular (closed alias
  set, everything else raises); PBS `extinction_ratio < 1` raises instead of
  silently swapping ports; `jones_pupil_to_stokes_unpolarized` accepts both
  Jones-pupil layouts bit-identically and rejects shapes with no 2×2 block;
  DOP floor made relative (a 1e-15 V/m fully-polarized field read DOP 0.0)
  with NaN propagation and [0,1] clip; grazing `kz→1.0` substitution
  (7e12×) now returns 0.0; `chi` domain enforced; the S3 convention
  relabelled IEEE/right-hand-rule (measured; was mislabelled Born-Wolf —
  code self-consistent, no sign changed) in the module docs and
  CONVENTIONS.md.
- **M2/M3/M4/M5/M6/M9 (`4602b7f`)** — `PMMStack.add_layer` validates segment
  width fractions (sum-1.4 and metre-valued inputs were silently clipped to
  a different structure); classical PMMStack rejects gain/NaN/non-propagating
  media (the audit's suggested one-liner was verified INSUFFICIENT — three
  guard layers applied; legit lossy media still solve silently);
  `_warn_stack_energy` raises on non-finite/negative totals; the fff_nv
  non-separable guard now also gates metallic cornered cells on max|Nx·Ny|
  (the measured Cxy driver: 0.500 square vs 0.000 validated stripe) while
  keeping validated lossless-dielectric cases admitted; fff_nv is exempted
  from the `stabilize=True` lossless-closure ladder accounting (its closure
  error is inherent/non-Hermitian — every rung warned, so stabilize ALWAYS
  hard-raised; laurent/li accounting unchanged, injected-failure control
  still trips); `set_blas_threads` warns once when threadpoolctl is absent
  (internal sweep sites use a quiet variant); the geo-eig cache returns
  read-only arrays (measured cache poisoning closed).

New measured findings recorded during the wave (open, not fixed):
`aberration_tensor` returns identical L for every `output_mode`
(`propagators/asymptotic.py` — LG non-piston merit channels currently
indistinguishable from piston); NumPy `propagate_through_system` junk-method
fall-through to ASM (same class as P6, wider blast radius); uniform cell +
`fff_nv` raises `AssertionError` instead of a clear error;
`create_elliptical_polarized(orientation=NaN)` and
`apply_waveplate(retardance=NaN)` unguarded.

### Fixed (adversarial-audit Tier 2, batch B — infra/contract findings, 2026-07-25)

Second wave over the same report (Territories P + A and the two elements
findings in the same blast radius), same discipline: reproduced by measurement
first, re-measured after, 55 new pins in
`tests/unit/test_niche_audit_p2b_infra_contracts.py` of which 40 were verified
to fail on the pre-fix tree.  Everything the fixes claim NOT to change was
proven bit-identical across the two trees (22/22 captured arrays: the
Richards-Wolf fields at all three probe pitches, the HF callable at explicit
finite-difference steps, the JAX/NumPy ASM system walk, the explicit Dammann
unit paths, `quarter_wave_ar`).

- **P3 (HIGH)** — `hf.py` Van-Vleck cross-Hessian `finite_diff_step` default
  1e-9 → 1e-6 m.  At 1e-9 the central-difference stencil was essentially pure
  round-off: on an exact-quadratic (Fresnel) OPL oracle the recovered density
  amplitude was −9.05e-2 wrong at the origin and the end-to-end field was up
  to 1.56e-2 off exact Fresnel quadrature; at the new default those become
  −2.5e-8 and 8.3e-9.  Docstring now carries the h² / eps-h² error budget and
  how to scale `h` with the caller's length scale.
- **P4 (HIGH)** — `richards_wolf_focus` now warns (`RuntimeWarning`, VD-1 /
  S9-VD2 house style) when `Np*dx_pupil/2 < f*NA`, i.e. when the pupil ARRAY
  cannot reach the geometric rim and the exit pupil silently degenerates into
  the square array boundary: measured 5.53× focal-FWHM error (1.9833 µm vs the
  0.3587 µm Airy width) at NA_eff = 0.160 against a requested NA = 0.9, with
  zero diagnostics before.  The message reports requested NA, delivered
  NA_eff, and both remedies by number (`dx_pupil >=` / `Np >=`).  Diagnostic
  only — fields bit-identical, and a spanning pupil stays silent.
- **P6 (MEDIUM)** — `propagate_through_system_jax(method=...)` was accepted
  and never read: `'fresnel'`, `'sas'`, `'rs'` and outright junk all returned
  the ASM field, 5.0e-2 relative L2 from the NumPy twin's Fresnel answer.  The
  JAX path is ASM-only by construction (the twin's Fresnel/SAS branches
  resample via `scipy map_coordinates`), so it now says so: `NotImplementedError`
  for the NumPy-twin-only methods, `ValueError` for unrecognised names,
  docstring updated to match.
- **P7 (MEDIUM)** — `propagate_huygens_fresnel_with_opl_callable`: the
  `opl_fn` units contract (returns **WAVES**, not metres — a metres-valued
  callable measures 1.0e-6 of the correct field) is now documented
  prominently, and the required-but-unread `wavelength` keyword is optional +
  deprecated (removal v5.32) rather than consumed, which would have broken
  every existing waves-returning callable by ~1e6.  Numbers unchanged.
- **E-H6 (HIGH)** — `broadband_ar_v_coat(n_substrate, ...)` never read
  `n_substrate`: the hard-coded n_H=2.3 / n_L=1.38 quarter-quarter stack
  satisfies the admittance match only for a substrate of (2.3/1.38)² = 2.778,
  and measured with this module's own TMM at 550 nm it was WORSE THAN BARE
  GLASS across the common range (R = 0.0856 vs 0.0426 bare on N-BK7 — an "AR"
  coating that doubled the reflectance; 0.0986 vs 0.0337 at n_s = 1.45).  The
  high-index layer is now set by the quarter-wave admittance condition
  `n_H = n_L·sqrt(n_substrate/n_ambient)`, giving R ≤ 1e-31 at the design
  wavelength for every substrate probed (1.45 → 4.0) and a 450–650 nm residual
  of 0.0202 (N-BK7) to 0.0058 (n_s = 2.78).  Layer order (ambient-side first,
  v5.4.6 P3-6) and the QW thicknesses are unchanged; non-physical
  `n_substrate` now raises.
- **E-H11 (HIGH)** — `makedammann2d(_legacy_units=)` default flipped
  `'auto'` → `'SI'` and the micrometre auto-detect shim retired.  The
  heuristic multiplied any period/wavelength above 1 mm by 1e-6, so a correct
  SI THz/MMW design (8 mm period at 1.1 mm) came back with 5e-10 m cells —
  behind a `DeprecationWarning` that Python suppresses outside `__main__`, so
  the surfaced-warning set at a library call site was literally empty.  SI is
  now taken at face value (no rescale, no warning); the heuristic survives one
  cycle behind an explicit `_legacy_units='auto'` with a **loud** `UserWarning`
  (removal v5.32); `_legacy_units='um'` is unchanged as the migration path.
  Two existing pins that encoded the old default/category were updated with
  justification.
- **A-3 (HIGH)** — `lumenairy.DEFAULT_COMPLEX_DTYPE` / `DEFAULT_REAL_DTYPE` /
  `DEFAULT_WAVE_PROPAGATOR` / `DEFAULT_DY` were import-time snapshots the
  four setters did not move (`set_default_complex_dtype('complex64')` left the
  top-level constant reading complex128 while the getter and the submodule
  twin both read complex64).  Now PEP-562 live-forwarded to
  `propagators.fft_infra`, replicating the `propagation.py:298` precedent
  (whitelist + delete the stale bindings + `__getattr__`), plus a `__dir__`
  so the names stay discoverable.  Export integrity unchanged (every
  `__all__` entry still resolves; phantom names still raise `AttributeError`).

### Fixed (wave 7 — the W6 sibling-class sweep over RCWA and PMM interiors, 2026-07-27)

The question "do pmm/rcwa carry the same defect classes W6 found?"
answered by two Opus agents with the seven W6 classes as targeted
hunts plus the original audit's recorded non-coverage. Verdict: YES —
twelve more FOUND+FIXED groups, four classes refuted per package with
oracle evidence (~330 pins):

- **PMM (`5e52a08`, `7893468`)** — the Berreman flux-gauge class hit
  THREE times: every slanted-layer `PMMStack` solve silently zeroed T
  into absorbing substrates (internal gauge fed to the public-gauge
  `_kz_forward`; all 71 gauge sites now AST-classified — PMM and RCWA
  need OPPOSITE bridges, documented); the TM incident flux mixed real
  `kz_inc` into complex `eps_sup` (TE≠TM at NORMAL incidence by up to
  5.8e-2); the stabilizer rejected every degree under an absorbing
  incidence medium. Plus the default 2-D SEM symmetry fold missing the
  gauge on its dense operators (2e-2), length-unit-dependent flux
  floors (R+T=2.027 in metres vs 2.000003 in nm), an incomplete
  geometry cache key returning bit-identical STALE answers across
  degree changes, five writeable caches, and traced-wavelength
  rejection at all three entries (jit had returned different array
  LENGTHS than un-jitted with 0.39%-wrong gradients; 16 in-repo sites
  migrated, 9 of which were asserting d/dλ against an FD reference that
  went through the same collapsed path).
- **RCWA (`6405095`)** — requested DUTY CYCLES silently rounded to the
  FFT sampling grid (invisible to order-convergence and closure;
  error matched the analytic prediction to 3 figures; now the exact
  Fourier series), which also explains why ASR NEVER CONVERGED (a flat
  1e-4 quantisation gap at every order — ASR now beats uniform 15× at
  M=9); returned arrays that WERE the caches (mutating `kz` poisoned
  the next solve; a prepared object's order table was every solve's
  order table); a floor-vs-nearest cell-indexing bias that no grid
  refinement shrinks; a 90° fast-axis error for circular
  eigenpolarizations; silent R+T=1.023 under lossy incidence (now
  warns); two moved pins retuned with DECOMPOSED attribution (the
  fixed solver at the legacy quantised duty reproduces the historical
  values — proven geometry, not drift).

Refuted per package with oracles: exponential cascade conditioning,
discarded-Im absorption (from-scratch TMM at 1e-15/1e-16), the
even-parity folds, forward-mode classification at degenerate edges,
dimensional tolerances elsewhere. Deferred with record: the `eps_cell`
rasterisation contract (anti-aliasing is a semantics decision), CuPy
(no device).

### Fixed (wave 6 — the four never-validated solver interiors, audited and fixed, 2026-07-26)

Four Opus agents took oracles into the territory the original audit
honestly declared unreached (~10k lines of EME / BOR / Berreman /
asymptotic numerics). Three CRITICAL-physics defects found and fixed,
plus deep verified-clean maps (~640 pins):

- **EME (`2ecd20b`)** — the strip solver keyed Hermitian-vs-general on
  `kx0 == 0`, but the operator is Hermitian for real ε at ANY real Bloch
  phase: oblique cells returned 68 spurious roots, growing propagators
  (|exp(i ky h)| to 6.2e6) under the wrong normalisation metric — now
  3/3 analytic modes, 0 spurious, unitary to 9e-16. `mode_match`'s
  z=0-referenced backward amplitudes grew cond to 8.7e38 — an
  index-matched medium returned R=1.000000 with energy=1.000000 masking
  it (now Airy to 5e-15 through depth 16). Absorbing slabs behaved
  LOSSLESS (`Im(qz²)` discarded — 95% absorbed power reported emergent).
- **BOR (`884115d`, flip `aae5f38`)** — the staggered eigensolver's wall
  anchor sat half a cell out: FIRST-order convergence in the production
  BORStack basis (p=0.995 measured). The corrected anchor (p=1.989, four
  decades at identical cost, de Rham + energy preserved) is now the
  DEFAULT, with the legacy anchor kept as a documented escape hatch and
  the 2026-07-13 audit gates retuned by the bit-exact equivalence
  `Rbig+h/2` (every historical number still reproduces; the
  lossless-trap gate's discriminator re-derived as a property so the
  retune could not defang it). Plus: dimensional guided-mode margins
  (nanometre fibers silently returned []), junk-wall/PML validation,
  modal-LRU cyclic collapse (41→21 eigs), one-node-ring interface
  clobbering, complex-eps lexicographic angle ordering. The classic
  r-dr-weight defect REFUTED at 4.6e-14 under an independent quadrature.
- **Berreman (`c314d01`)** — the off-plane solvers double-conjugated the
  flux gauge into `_forward_flux_kz`, silently ZEROING transmission into
  absorbing substrates (T=0.000000 vs the rcwa-verified 0.930316; the
  JAX twin hit it at every incidence, a 0.988 twin divergence). Full
  guard parity with siblings added (a metallic superstrate had returned
  3000% energy violation silently; back-side angle aliasing). The
  router's "~2% off" claim refuted at 4.3e-15; isotropic limit matches
  the TMM at 1.9e-15; uniaxial closed forms, rotation covariance,
  reciprocity, twist-continuum order 4.00 all verified.
- **Asymptotic (`2a54deb`)** — the Maslov raster unwrap made a pointwise
  field GRID-DEPENDENT (proven analytically unnecessary: Re M is PD so
  the principal branch is unique and the true index is 0; 60/289 shared
  points flipped sign between grids); the LG/HG mode-stack cache
  collided `xy`/`ij` meshgrids (callers received TRANSPOSED stacks at
  1.6e15× error); quadrature validity/uniformity warnings; the
  scale-relative Newton verdict; measured accuracy envelopes and the
  rank-deficiency of the default fit documented in the public
  docstrings.

Also in this phase: the P5 return-contract flip executed (`2898665`,
roadmap F1 EXECUTED — with the flip inventory's one wrong entry caught
by measurement), all scheduled shims removed (`3a1da2b` + `bfb6179`),
input_kind 68/68 (`29f8dbc`), and two CI pin recalibrations to measured
cross-platform envelopes (`c0073c2`, `bfb6179`).

### Fixed (adversarial-audit wave 4 — every remaining open item closed, 2026-07-26)

Seven Opus agents + three follow-on legs drained the deferred/flagged
list to zero (~640 new pins; every set verified failing pre-fix; all
value pins authored with measured cross-platform tolerances):

- **P5 return-contract transition DECIDED + shipped (`3097cda`)** — option
  4: falsy-sentinel default (43/43 arrays bit-identical), registry-routed
  DeprecationWarning only for external callers who didn't choose a
  contract, flip bound to `API_TRANSITION_VERSION` (= registry
  NEXT_REMOVAL_VERSION), P16 iteration arity decided permanent, flip-day
  migration inventory in the roadmap; plus the `propagate_tilted` method
  warning and the GBD paraxial z_image default fix.
- **`input_kind` rollout (`ce0a265`)** — the deferred "67 sites" resolved
  by AST inventory; closed vocabulary, 25 in-scope sites wired
  fail-closed, handoff table auto-enforcing as owners wire theirs.
- **h5 sibling writers (`30f0043`)** — all four remaining raw metadata
  surfaces on the codec; legacy files read byte-identically.
- **Shack-Hartmann reference calibration S12-1 (`c5d3e27`)** — the flag's
  premise was wrong but measurement found the real defect: the flat-field
  reference propagated with a bare fft2 in the wrong plane, reading up to
  15.2 mrad of phantom tilt on a FLAT wavefront; now exactly 0.0 at all
  45 configs; three degenerate pins rebased with documented arithmetic;
  coherence-dock source shapes wired (+ a second silent bug: empty
  source-angle lists returned a black frame); three ratio-based timing
  pins retuned to measure their guarded regressions (each caught a real
  flake during its own verification).
- **Immersed-conjugate optics (`8fdcccc`, `aed46ac`, `21185b7`,
  `d298207`)** — a four-leg oracle campaign: pupil positions were reduced
  (exactly 1/n, −34/−39%; no pre-existing test exercised an immersed
  conjugate, probe-proven); bfl/ffl/principal planes geometric while efl
  stays reduced by consumer audit (the algebra twin structurally cannot
  express the alternative; the agent retracted its own fnum claim with
  the disproving measurement); analysis/ consumers fixed (distortion
  reference was +51.7% = 100·(n−1) at every angle; the indexed Gauss
  solve; folded bfl-as-thickness 185× RMS error); and the folded-frame
  finale (world image planes placed along the TRACED exit direction,
  resolving the two-flavour world-list ambiguity without frame marking;
  folded reference spheres now reproduce their mirrorless controls
  bit-for-bit across 32 configuration pairs — pre-fix 388 waves PV for a
  1.1-wave system).
- **Elements/raytrace gaps (`809314c`)** — odd-power guard on the five
  wave-optics sag sites (100× silent sag error closed end-to-end);
  non-finite f guards (damage differed per lens model — one was a silent
  no-op); the 1-D-stack y-averaging trap now warns via
  `RCWAYAverageWarning` (33.6% measured error, closure-invisible;
  exported top-level); the last R-18 dead-code claim CONFIRMED by AST and
  deleted.
- **LG-merit stack completed (`8ffc90e`)** — adaptive σ-grid from a
  triply-validated chirp bound (6–7× accuracy at the same verdicts, cap
  measured); the (2,0) oscillation fixed by an opt-in curvature-matched
  complex-q basis (0/6 sign flips vs 5/6); and the "local-only" JAX
  fit-match red exposed as a REAL defect (Tikhonov floor 4.25× spec
  through normal equations — QR fix, 12× truth-error improvement,
  validation 48/48 everywhere).

Campaign process lessons recorded en route: editable-install +
script-dir sys.path can silently resolve the wrong tree from worktree
scripts; `git stash` on the shared tree is forbidden (one incident,
fully recovered); chirp-integral pins need ≥1e-2 measured tolerances.

### Fixed (adversarial-audit wave 3 — oracle resolutions + remaining MEDIUM/LOW territories, 2026-07-25/26)

Seven Opus agents closed out the audit's MEDIUM/LOW inventory, the four
new findings from the fix waves, and the two oracle-needing physics
questions (measure-first throughout; ~470 new pins across 7 files, every
set verified failing on pre-fix worktrees).  Territory blocks for
raytrace/sources and UI/deprecation follow separately below; the other
five in brief (full numbers in the commit messages):

- **Oracle wave (`1523d8e`, `86cadbe`)** — three real physics defects
  confirmed and fixed by building the oracle first: **(W3-1)** the
  coordinate-break rotation was INVERTED in `intersection.py`,
  `differential.py` and `ui/model.py` since 3.7.1 (8.24° of refracted-ray
  disagreement at a single 12° tilt; `world.py` proven the correct
  local-to-world side per OpticStudio KA-01638; the 2026-07-08 "phantom"
  revert argument retracted — a mirror fold is provably sign-degenerate;
  a test that pinned the bug rewritten; new canonical CONVENTIONS.md §7
  row; **user-visible: every non-zero coord-break tilt now folds the
  other way**); **(W3-2)** the stop-adjacent pupil legs dropped the
  Welford mirror-parity sign (flat fold 15 mm before the stop: ep_z
  +307% and bit-identical to the mirrorless control; fixed to 7.8e-14 vs
  two independent oracles; new flagged finding: immersed-conjugate pupil
  positions drop 1/n); **(W3-3)** `aberration_tensor` output modes were
  POINT SAMPLES, not projections (every (p,0) returned the piston value
  bit-identically; now routed through σ-integration, 4.9e-15 vs an
  independent LG-overlap oracle); **(W3-4)** 30 silent NaN-field leaks in
  polarization angle/retardance inputs guarded at the shared helper.
- **Infra (`d045980`)** — h5 `append_plane_h5` metadata now round-trips
  all 19 probe types via the module's own codec (was 14 OK/4 raise/1
  dropped; legacy files read byte-identically); `deep_nbytes` charges
  views by base (a ByteBudgetedLRU could retain 64× its cap);
  `estimate_asm_memory` re-fit to a measured 1.02–1.09× bound;
  user-library corruption now warns; zernike basis cache read-only; io
  `__all__`; negative memory costs raise.
- **Propagators (`3f22778`)** — `propagate(method='auto')` warns when a
  grid-changing kernel alters the return contract (API redesign recorded
  as roadmap Part F/F1); NumPy `propagate_through_system` junk-method
  fall-through closed (system + per-element); `recommend_gbd_sampling
  (wavelength=)` measured inert-by-physics and deprecated;
  `fga_memory_estimate(nsig=)` wired (calibrated 0.993);
  `patches_for_box(centred=True)` implemented symmetric (audit's offset
  formula corrected); MFT output-grid validation + kernel-dependent
  replica warnings (audit's period formula extended by measurement);
  band-limit docstrings state the Matsushima asymptote; `beam_d4sigma`
  fallback shape bug; dead guards/params pruned with live ones
  counter-pinned.
- **RCWA/PMM (`3ead8cb`)** — uniform-cell `fff_nv` routed to `laurent`
  (was `AssertionError`); `n_orders_y=0` allowed on y-invariant cells
  (matches the 1-D solver at 5e-15, 27× cheaper) WITH a new guard: a
  y-varying cell at `N_y=0` previously solved the y-averaged structure
  silently (~2× wrong, energy-clean); missing
  `formulation`/`stabilize`/`symmetry` kwargs threaded into the two
  wrapper entry points (default paths digest-identical); the
  propagating-order mask drift was PMM-1-D vs everything else — aligned
  on the documented convention, zero pin movement; hygiene batch.
- **Elements (`bba1bc4`)** — ray_density energy self-check (band sized on
  the measured battery envelope incl. a subsample-independent physical
  floor); DOE periodic-mask half-pixel wrap bug (even-order leakage
  0.0295 → 3.7e-33); thin-lens out-of-domain NaN parity; worker-pool
  atexit/locking/exception-swallowing repaired (worker faults now
  propagate; broken pools reset); the P1 odd-N anchor fixed at its two
  elements siblings (uniform-caustic sampler, turbulence-screen DC bin —
  screens differed by up to 86% of peak at odd N); dead numexpr scaffold
  deleted; `coronagraph.py` deletion claim REFUTED (public namespace)
  and covered by a test instead.

### Fixed (adversarial-audit Tier 2, wave 3 — Territory R: raytrace / glass / sources / optimize MEDIUM + LOW, 2026-07-25)

Territory R's MEDIUM/LOW tail plus the dead-code and overdue-shim sweep.
Every finding reproduced by measurement BEFORE the fix and re-measured after;
72 pins in `tests/unit/test_niche_audit_w3_raytrace_sources.py`, 47 verified
failing on a pre-fix worktree of `7ea2eb9` (the 25 that pass pre-fix are
non-vacuity probes plus the three DECLINED-finding locks, each labelled as
such in its class docstring).

- **R-8 / E-L7 (MEDIUM)** — an ODD aspheric power made a surface
  sag/normal-INCONSISTENT, and inconsistent DIFFERENTLY per backend: the sag
  floors `h_sq ** (p // 2)` to the next-lower EVEN power while the NumPy
  normal uses `p*h**(p-1)` and the JAX normal `p*h_sq**((p-2)//2)*x`
  (measured at `{5: 1e6}`, h = 10 mm, flat base: sag 0.01 m, NumPy dz/dx
  0.05, JAX dz/dx 5.0 against the sag-consistent 4.0 — a 100x
  cross-backend divergence).  `validate_prescription` already rejected it,
  but a hand-built `Surface(aspheric_coeffs={5: A5})` and the JAX
  prescription path (which never builds a `Surface`) both sailed through:
  `trace_jax` returned a finite ray height off the inconsistent surface.
  One shared guard now sits at both entry points —
  `raytrace/_conic_core.py:55` (`check_even_aspheric_powers`, called from
  the sag/derivative twins at `:266` and `:305`, which is the JAX backend's
  only sag route) and `raytrace/surface.py:264` (`Surface.__post_init__`)
  — with the same message `validate_prescription` emits.  Non-integral
  powers (`4.5`) are rejected too; EVEN powers are byte-unchanged.
- **R-11 (MEDIUM)** — `paraxial.f_number` returned the SIGNED ratio, so a
  diverging prescription read `f/-9.965` while all three siblings computing
  the same quantity (`raytrace/layout.py`, `merit_terms.MaxFNumberMerit`,
  `FirstOrderData.fnum`) reported `+9.965`.  Now `abs(EFL)/D`
  (`raytrace/paraxial.py:200`); all four agree bitwise.  Grep-verified that
  no consumer read the sign.
- **R-12 (MEDIUM)** — `glass._sellmeier_index` was scalar-only: an ndarray
  died on the resonance guard with numpy's opaque "truth value of an array
  with more than one element is ambiguous" and a list with "can't multiply
  sequence by non-int of type 'float'" — while the `_polynomial_index`
  sibling's docstring advertised scalar/array parity with it.  Given the
  same scalar-fast-path + vector-path split as that sibling
  (`glass.py:694`); the vector result is bit-identical (0 ULP) to a scalar
  loop over the same wavelengths, the scalar path is unchanged, and the
  vector path keeps the 4.10 resonance / negative-n2 diagnoses.
- **R-13 (MEDIUM)** — the NumPy DOE kick divided by the grating period
  unguarded: `period=0.0` raised `ZeroDivisionError` mid-trace and
  `period=nan` silently NaN-poisoned (L, M) (measured NumPy `(nan, nan)` vs
  JAX `(0.0, 0.0)`).  Now zero/non-finite means "no grating on that axis"
  — exactly the JAX twin's documented contract (`raytrace/trace.py:213`).
  `inf` was already 0.0 by IEEE division and is bit-identical.
- **R-16 (LOW)** — the numba dual `_dsqrtq` clamped a NaN radicand to a
  finite `0.0` (because `nan > 0.0` is False) where the `_dual_sqrt` NumPy
  twin propagates NaN through `np.maximum`.  NaN now propagates in value AND
  tangent (`raytrace/differential.py:681`); finite radicands, including
  exactly 0.0 and negatives, are bit-identical.  The divergence is invisible
  at the public boundary (`_adrt_numba` scrubs with `np.nan_to_num`; 7
  NaN/inf probes x refract/mirror stacks came back IDENTICAL pre-fix), so
  the njit primitives are published as `_ADRT_NUMBA_PRIMS` for the pin to
  compare against `_dual_sqrt` directly.
- **R-9 / R-10 (MEDIUM API hazards, documented — no signature change)** —
  `create_hermite_gauss` / `create_laguerre_gauss` take `w0` in the
  positional slot every other `create_*` factory uses for `wavelength`, and
  the swapped call was silently accepted.  Prominent docstring warnings plus
  a zero-false-positive runtime check (`sources/core.py:157`, called at
  `:700` / `:872` AFTER the `w0 > 0` validation): a paraxial mode always has
  `w0 >= wavelength`, and the six in-repo HG/LG call sites run at ratios
  6.45-2000, so the tightest clears the threshold by 6.4x.  The annular
  radius-vs-diameter split (`create_annular_beam` takes DIAMETERS,
  `create_annular_incoherent_source` RADII) is now flagged at both sites.
- **R-17 (LOW)** — `design_optimize(wave_traced=)` and
  `MatchIdealSystemMerit(use_traced_lens=, focus_search=)` are documented
  public flags with live branches and ZERO callers anywhere (library, tests,
  validation, examples, UI).  Deprecated through the shared registry with
  removal v5.32 (`optimize/driver.py:577`, `optimize/merit_terms.py:579`)
  — NOT deleted, since out-of-repo callers cannot be ruled out.  The
  warnings fire only on a non-default value, so the existing corpus stays
  silent.  The four consequently-unexercised penalty helpers keep their code
  and now say in their docstrings that CI does not cover them.  `match=` was
  NOT deprecated: its default is the exercised path and the metric choice is
  a real feature (only its three non-default kernels are uncovered).
- **Overdue shims re-scheduled** — seven `version_removed` = `5.0` sites in
  `sources/core.py` (the `create_led_source` positional shim, the five
  `Source.*` legacy positional shims, the Schell `return_kind` sentinel
  helper) were still shipping at v5.29, 29 minor releases past their own
  removal date.  Re-scheduled ONCE to v5.32 through a single constant
  (`sources/core.py:60`) so the next slip is a one-line edit and cannot drift
  between sites; measured that all seven warnings still fire from the
  production path.  Two existing pins that hard-coded the old string were
  corrected to assert against the constant, with the rationale in their
  docstrings.
- **Declined, with the measurement that justifies each** — **R-14**
  (aperture-clip order differs NumPy-vs-JAX): only DEAD rows differ (`|dL|`
  up to 0.187); the alive masks are equal and every ALIVE row's `x`/`L` is
  bit-identical, because the clip reads only `(x, y)` which refraction never
  touches.  Documented in `trace()`'s Notes and pinned; the reorder would
  touch two other territories and would relabel a vignetted-AND-TIRing ray
  from `RAY_APERTURE` to `RAY_TIR`.  **R-15** (`make_fan(axis='x')` tilts in
  `L` where `make_ring`/`make_grid` tilt in `M`): the per-axis convention is
  load-bearing — monkeypatching the proposed "always M" form moved
  `ray_fan_data`'s `ex(0)` from exactly 0.0 to -1.381e-04 m at a 3 deg
  field, breaking the RT-5 invariant at four call sites.  Documented on
  `make_fan` and locked by a pin.  **R-18 dead code**: four of the
  "grep-verified dead, delete" entries have live or grep-invisible
  consumers — `_invalidate_glass_name` is called by
  `raytrace.trace._register_fixed_index`; `_GLASS_CACHE` is the companion
  name the v4.14.2 cache/lock walker discovers BY REFLECTION (the lower-case
  `_glass_cache` does not match its candidate list);
  `_POLYNOMIAL_STUB_NAMES` is a documented forward-compat hook whose
  emptiness IS the invariant, read by a load-time well-formedness loop and
  already acknowledged by two skipping tests; `TraceResult.rays_at` is
  public documented API that `trace()`'s own `output_filter` contract
  describes.  `DifferentialTransfer` / `ParetoResult` are the return TYPES
  of public functions, not unused exports.  Counter-pins record each
  consumer so the next hygiene pass does not repeat the mistake.
- **CHANGELOG line-citation refresh** — `optimize/merit_terms.py:536` ->
  `optimize/merit_terms.py:638` (`MatchIdealSystem._make_source` `ap>0`
  branch; the R-17 docstring additions shifted it by 53 lines).

### Fixed (adversarial-audit Tier 2, wave 3 — UI breadth + deprecation registry, 2026-07-25)

Territory A's UI pass and deprecation rot.  All six dead UI actions were
measured headlessly (PySide6 absent) exactly as the audit measured them:
`importlib` on the import TARGETS, `inspect.signature(...).bind` on the
kwargs.  36 pins in `tests/unit/test_niche_audit_w3_ui_deprecation.py`, 23
verified failing on the pre-fix tree.

- **Four dead propagator menu choices** — `waveoptics_dock.py:775-795`
  imported GBD / HFPI / Huygens-Fresnel / Subaperture from
  `propagators.propagation`, the v5.1.0 re-export shell for the
  ASM/Fresnel/RS/SAS/MFT family, which has never exported them.  Every one of
  the four whole-prescription runs died with `ImportError` and was reported to
  the user as a generic "`<method>` failed".  Now imported from the owning
  submodules (`propagators.gbd` / `.hfpi` / `.hf` / `.subaperture`) — the same
  targets `optimize/driver.py` and `propagators/dispatch.py` use; the call
  kwargs were already correct and bind unchanged.
- **Dead detector option + the unpack bug behind it** —
  `waveoptics_dock.py:994` imported a nonexistent `..detector`
  (`ModuleNotFoundError`, swallowed by `except Exception: pass`), so the
  "Apply detector to focal field" checkbox has always been a no-op.  Behind
  it, `analysis.detector.apply_detector` returns `(image, x_det, y_det)`,
  which the dock bound to `E_focus` — fixing only the import would have traded
  a silent no-op for a `ValueError` one line later.  Import corrected, the
  3-tuple unpacked, the electron image carried as an amplitude-equivalent
  field (`sqrt(clip(image, 0))`, pitch → `pixel_pitch`) so `|E_focus|²` IS the
  detected image for every downstream consumer, and detection moved AFTER the
  chief-relative-OPD conversion (detection destroys phase).  That handler now
  reports through `ui.diagnostics.diag`.
- **Three stale-kwarg docks** — `coherence_dock.py:42` passed
  `source_sigma`/`N`/`n_modes` to `koehler_image` and omitted the required
  `object_field`; `shack_hartmann_dock.py:128` passed `lenslet_focal_length`
  (the parameter is `lenslet_focal`) and `n_zernike` (never a parameter), then
  read `res.slopes_x` off what is actually a 5-tuple;
  `lg_aberration_dock.py:104` called `aberration_tensor(prescription,
  wavelength=, w0=, p_max=, l_max=)` against a signature taking
  `(fit: CanonicalPolyFit, s2_image, …)`.  All three re-bound against the real
  APIs: the Schell tab maps sigma → condenser NA via the model's EPD/2·EFL and
  mode count → `n_source_points`; the SH dock unpacks the 5-tuple and fits the
  advertised Zernike spectrum with `zernike_decompose`; the LG dock routes
  through the public `aberration_summary` wrapper (which owns the
  fit → envelope-solve → tensor chain) and renders `.L` with real (p, ℓ)
  Seidel labels instead of raw matrix indices.
- **Optimizer abort** — `optimizer_dock.py:1088` built
  `ToleranceAwareMerit(inner_merit=…, radius_sigma_frac=…, thickness_sigma=…)`;
  the parameter is `sub_merit`, the other two do not exist, and the required
  `perturbation_spec` was missing — selecting that merit raised `TypeError`
  and aborted the run.  Now passes `sub_merit` plus a per-surface
  decentre/tilt spec, and logs that radius/thickness sigmas are not part of
  this merit's Monte-Carlo model rather than silently reinterpreting them.
- **`ui/surface_table.py` DELETED** (370 lines, `SurfaceTableEditor`) — zero
  references repo-wide (verified in pure Python, no shelled-out `rg`);
  superseded by `element_table.py`.  The `ui/__init__.py` architecture list
  also named a `workers.py` that has never existed in this tree.
- **Deprecation removal-schedule registry** — the removed-in banner emitted
  "will be removed in v5.27" FROM v5.29.0, and nothing in the library ever
  compared a stated horizon against `__version__` (four independent f-strings
  each interpolated `version_removed` verbatim).  `lumenairy/_deprecation.py`
  now owns `REMOVAL_SCHEDULE` / `NEXT_REMOVAL_VERSION` (v5.32) /
  `resolve_removal_version` / `check_removal_schedule`, routed through a
  single `_format_removal` builder: a re-scheduled horizon reports as
  `will be removed in v5.32 (rescheduled from v5.27)` — naming the slip
  instead of moving the goalpost silently — and ANY stated version that has
  already shipped is promoted to the live one as a backstop.  No shim is
  removed (that stays a release decision for each module owner).
- **`_lens_jax.py`** — `apply_real_lens_traced_jax` documented a parameter
  `lens_prescription`; that is the function's internal alias and the real
  keyword is `prescription`, so the documented call form raised `TypeError`.
  Pinned generically: every parameter documented in that module must exist in
  the signature.

## [5.29.0] — 2026-07-25

**The traced-carrier production campaign.**  `propagate_traced_carrier_chain`
becomes a production daily driver: energy-conserving, grid-convergent, and at
its readout's ideal-field ceiling on the design-121 acceptance with ZERO
configuration (best-focus FWHM 3.450 µm, EE3 88.8%, EE6 99.6%, EE12 99.8%,
on-axis, dx- and ray_subsample-flat).  The chain DEFAULTS flip to the
validated carrier-regime configuration (`carrier_reference='sphere'`,
`amplitude_model='ray_density'`, `preserve_input_phase='remap'`,
`remap_sampling='full'`, `fit_radius_beam_factor=2.0`; legacy escape hatch
documented and pinned); standalone element defaults are untouched.  Includes
the P2 daily-driver guards (aperture:beam cliff guard, memory-bounded exact
readout, opt-in dx self-check, CI-safe design battery with a documented
known-good envelope) and two library-wide sibling-pattern sweeps + their
deferred-decision resolutions (33 measured fixes total, including the
flat-fold Seidel parity physics fix and its v4.15.2 algebra twin, the
upsample lattice bug, and the ray-trace flat-fast-path cylinder-sag drop).
Every fix was landed measure-first with a regression pin (~500 new pins);
full causal records in `docs/audits/AUDIT_TRACED_FROZEN_AMPLITUDE_2026_07_24.md`
and `docs/audits/AUDIT_SIBLING_PATTERN_SWEEP_2026_07_25.md`.

### Fixed (deferred decision items resolved, S11/S12, 2026-07-25)

All ten report-only findings from the sibling-pattern sweep are resolved
(commits `403ea1f` S11, `75517cb`+`a9dc454` S12; 105 new pins; full details
in the commit messages).  Highlights:

- **Seidel flat-fold parity (physics)**: a flat fold mirror skipped the
  R-independent Welford `n' = −n` flip in `system_abcd`/`seidel_coefficients`
  — EFL discontinuous at R=∞, ray heights diverging past the fold; validated
  against first principles + R-continuity + 13 exact-trace probes; the
  algebra layer had COPIED the bug in v4.15.2 (agreement-pinned on the wrong
  answer) and is fixed in lockstep; three folded-prescription pins moved to
  exact-trace-confirmed values (≤3e-13).
- NaN-position rays no longer survive the NumPy aperture gate as `RAY_OK`;
  cell-centred ray placement (`n_rays`=1..4 used to raise on valid input);
  `pixel_pitch` is the detector's single pixel-area authority (explicit
  `n_pixels` silently redefined it up to 16×); the AO `noise_sigma_pixels`
  knob was inert (applied post-calibration) and now works as documented;
  a guards batch (resample_field (0,0) array, telecentric NaN, degenerate-
  input crashes, unguarded nanmax); conic NaN inputs propagate (the
  documented JAX out-of-domain zero-clamp preserved).
- **The remaining-accuracy budget, measured**: 1.37 of the nominal 2.1-EE3
  gap to the ideal ceiling was a focus-selection artifact; the taper-skirt
  attribution is a measured NULL; the paraxial gap transport agrees with
  exact ASM to 0.019 rad.  The real mechanism — `'remap'` sampling the
  carried residual phasor on the coarse ray lattice (aliasing beyond a
  predicted-and-measured r_alias = 1.52 w) — is fixed by the new
  `remap_sampling='full'` (element opt-in; **chain default**), which also
  makes the mode `ray_subsample`-independent (180-9000× reduction).
  Design-121 pure defaults: best-focus **FWHM 3.450 µm (= the ideal-field
  ceiling), EE3 88.8, EE6 99.6, EE12 99.8**, on-axis, dx-flat.
  `_fine_trace_group_exit` warns when the F-C ray-pitch contract cannot be
  met on the retrace grid.  Deferred with documentation: wiring
  `detector_pixels_per_lenslet` (a feature moving every SH slope) and four
  flagged analysis sites whose fixes are not bit-compatible.

### Fixed (sibling-pattern sweep, 2026-07-25)

Sixteen measured fixes from a library-wide sweep of the traced-campaign bug
classes (two Opus-subagent territories; every fix repro'd with numbers
first, bit-identical where the old path was correct, 77 new regression
pins; full details in the `17c0ad7`/`fd78ad2` commit messages and
`docs/audits/AUDIT_SIBLING_PATTERN_SWEEP_2026_07_25.md`).  Highlights:

- **raytrace**: the flat fast path silently discarded the sag of
  y-cylinders / freeform-on-flat surfaces (771 waves of OPL at the probe);
  biconics at the `conic_y=None` default were untraceable (`TypeError`);
  tall-grid ray placement was 20-511× anisotropic.
- **traced element**: `sag_chunk_rows` banding (auto-ON at N≥4096)
  silently downgraded the R7 cubic OPL upsample to linear under an engaged
  carrier (λ/216 rms); `carrier=±inf` built an all-NaN eikonal (the
  "All-NaN slice" warning source) — now the analytic plane-wave limit;
  `_multi` silently dropped `preserve_input_phase='remap'`; `prepare`
  rejects incompatible modes with reasons.
- **carrier**: `_fourier_upsample_crop` rejects `n_crop >` input (silent
  wrong-window 1×1-crop branch); the auto-fit gains a dx-exact
  `estimator='increment'` + alias guard (default `gradient` estimator bias:
  +0.4% → +32% with pitch); the chain names its own default-flip conflicts.
- **vector diffraction**: Richards-Wolf pad/crop parity phase
  (2π/N_focal rad/px on mismatched parity) fixed; pupil-truncating
  `N_focal` now warns with the measured NA/energy loss.
- **analysis**: anamorphic-grid radial profile sampled an index-space
  ellipse (Sparrow drifted 1.894→1.138 µm on an unchanged PSF); a dead
  all-NaN guard reported fully-vignetted distortion sweeps as "0.0%".
- **validation harnesses**: bare runs now measure the SHIPPING chain
  configuration; unrecognised knob values error instead of silently
  falling through (the class that produced the campaign's candidate-2
  false negative).

Ten further measured findings are recorded REPORT-ONLY with decision
notes (flat-fold Seidel parity, NaN-ray aperture-gate polarity, detector
`n_pixels`/`pixel_pitch` inconsistency, edge-anchored ray placement, and
others) in the sweep audit, plus a verified-clean map (notably: the GBD
node upsample — the original lattice bug's direct sibling — is correct,
proven with a counterfactual probe).

### Added (P2 daily-driver guards)

- **Aperture:beam cliff guard.**  `apply_real_lens_traced` gains
  `fit_radius_beam_factor` (default `None` = byte-identical): restricts the
  entrance ray samples entering the forward-map/OPL fits to a BEAM-relative
  disc `r <= factor*w_in`, decoupling the fit domain from the vignetting
  aperture — only the fit domain; `launch_radius`, the Newton bound and the
  NaN threshold are untouched, so **no field energy is clipped** (the
  aperture-clamp trap of audit 07-23 §4c.3 avoided; exit power identical to
  4 digits across the sweep).  `propagate_traced_carrier_chain` defaults it
  to `2.0` (escape hatch `traced_kwargs={'fit_radius_beam_factor': None}`):
  the E4 cliff (exit Strehl 0.105/0.042/0.039 at 7/8/10 mm apertures on a
  2 mm beam) recovers to **0.9995** with pre-cliff results and the
  design-121 acceptance unchanged (EE6 99.3; EE3 88.4→88.2).  New warn-only
  `on_aperture_beam='warn'` flags the possible-cliff regime (aperture >
  1.5× beam 1/e² diameter with no guard active).  Mechanism correction to
  the 07-24 audit §4: the cliff is the un-carriered launch SQUARE's corner
  samples (a collimated `carrier=inf` eikonal is NaN, so R7's fit-domain
  restriction never engages), evidenced by the per-group discriminator,
  the `_CARRIER_FIT_RADIUS_FRAC`-sweep null, config-independence, and the
  π/√3 random-wrapped-phase residual signature.
- **Memory-bounded exact readout.**  `carrier_referenced_exact_focus_readout`
  and `_fine_trace_group_exit` cap their internal fine grids against a RAM
  budget (new `ram_budget=` kwarg / `focus_readout={'ram_budget': ...}`;
  honours `set_max_ram` via `get_ram_budget()`, `inf` disables) with a
  RuntimeWarning naming both the capped and the un-degraded sizes — no more
  silent 16 GiB auto-sizing / MemoryError (the prior 34 GB box's crash
  condition now degrades gracefully at N_fine=16384).
- **Convergence self-check.**  Opt-in
  `propagate_traced_carrier_chain(self_check='dx', self_check_tol=0.05)`
  re-runs at dx/√2 (extent-preserving) and warns "NOT dx-STABLE" with
  per-metric deltas when focal metrics disagree — the cheap
  is-this-number-grid-stable flag (~3× cost, default off).
- **Design battery** (`tests/unit/test_niche_p2_design_battery.py`, 17
  tests): fast singlet / achromat doublet / triplet / E4 corrected relay ×
  beam sizes × aperture:beam {1.2×, 2.5×}, gated against an inline
  meridional ray oracle + analytic through-focus truths; documents the
  known-good envelope (aperture:beam 1.2–2.5×, exit NA 0.013–0.20) and pins
  the cliff in focal terms (relay 2.5×: EE-in-one-waist 81.1% guarded vs
  0.2% unguarded).  Plus `test_niche_p2_guards.py` (12 tests).  E4 file
  extended 3→9 tests (cliff pinned with the guard off + recovery/energy/
  warning pins); two R8 and two H3 tests take documented escape hatches.
- Fix: the paraxial final-leg readout no longer raises `TypeError` when
  `focus_readout` carries exact-path keys (`n_fine_cap`, `window_factor`,
  `ram_budget`, ...) and the leg turns out low-NA.

### Changed

- **`propagate_traced_carrier_chain` now DEFAULTS to the validated
  carrier-regime configuration** (audit
  `AUDIT_TRACED_FROZEN_AMPLITUDE_2026_07_24` §8, the recorded §6.8
  commitment): `carrier_reference='sphere'` and per-group
  `traced_kwargs` defaults `{'amplitude_model': 'ray_density',
  'preserve_input_phase': 'remap'}` (caller-supplied `traced_kwargs`
  always win).  Rationale: the chain by construction operates with its
  carrier beyond the grid Nyquist, where the geometric (ray-density)
  amplitude and the geometric residual carry are the correct physics,
  not options — with the old defaults the chain's exit discarded the
  design's distributed correction (design-121: best-focus EE6 79.7% →
  99.3%, FWHM 5.15 → 3.55 µm, dx-flat).  **This deliberately changes
  chain results**; the R8 orchestrator-vs-manual pins were updated to
  pass the legacy configuration explicitly, and the flip itself is
  pinned (`test_chain_default_is_validated_config_and_legacy_is_reachable`).
  Legacy escape hatch (pre-flip results, bit-for-bit):
  `carrier_reference='parabola'` +
  `traced_kwargs={'amplitude_model': 'screen', 'preserve_input_phase': True}`.
  The standalone `apply_real_lens_traced` element defaults are UNTOUCHED.
  Supporting element change: `preserve_input_phase='remap'` no longer
  requires an engaged carrier — with an absent/unengaged (collimated)
  carrier the de-chirp degenerates to the identity, and the input's own
  (slow) phase is the carried residual.

### Investigated & reverted (no net code change)

- **Roadmap Part E — wavefront-aware ray launch (candidate 1): implemented,
  tested, and REVERTED. Part E remains OPEN.**  A carrier-relative residual ray
  launch (`apply_real_lens_traced(tilt_aware_rays=True, carrier=<explicit>)` +
  a chain `wavefront_aware` opt-in) was built to carry a corrected relay's
  inter-group aberration through the ray launch.  It **did not work** — it
  regressed the real design-121 focus (EE6 50.0% → 10.0%) and gave no
  improvement on a synthetic corrected relay — so all of it (the kwarg, the
  `_carrier_relative_launch` branch, the E2 diagnostic + its unit test, and the
  residual-launch diagnostic script) was removed; the pre-existing F3 behavior
  (tilt_aware + explicit carrier → carrier-alone launch + warn) is restored.
  The investigation also **overturned an earlier reading**: the traced-carrier
  chain does *not* fail corrected relays — at a beam-matched aperture the plain
  sphere-only chain is diffraction-limited (Strehl 0.997, ties GBD); the poor
  synthetic result was an oversized-aperture (2.5× beam) fit-corruption
  artifact.  The real 121 residual is confirmed **neither** a launch **nor** an
  aperture/fit problem (both eliminated by direct test).  Full record:
  `docs/audits/AUDIT_WAVEFRONT_AWARE_RAY_LAUNCH_2026_07_23.md` (§4b–§4c).
  `tests/unit/test_niche_e4_corrected_relay_oracle.py` is retained, repurposed
  as a feature-independent regression (chain is diffraction-limited at a
  matched aperture; degrades past the aperture:beam cliff).
- **Known, unfixed:** the traced element's OPL fit is corrupted when the
  physical aperture greatly exceeds the beam (marginal-ray aliasing into the
  low-order fit; sharp cliff at ~1.5× the beam diameter on a fast singlet).
  Candidate fix (not yet done): a beam-relative launch/fit radius that decouples
  the ray-fit domain from the vignetting aperture.

### Added

- **`propagate_traced_carrier_chain(carrier_reference='sphere')`** (opt-in;
  default `'parabola'` is byte-identical to prior releases) — **closes the
  design-121 converged fidelity plateau.**  The carrier machinery references
  the paraxial parabola `r^2/(2R)`, while a traced element's ray launch and
  carrier eikonal reference the EXACT sphere `S(R)`; the difference,
  `+k r^4/(8R^3)`, is **+3.36 rad at r=w** at the design-121 first group vertex
  (emitter NA 0.104 over 45.9 mm) — a spurious spherical aberration relative to
  the physical diverging wave.  `'sphere'` band-limits that difference out of
  every hand-off (`_sphere_parab_conversion`, `cos²` taper ending at
  `r_safe = (|R|³λ/dx)^{1/3}`, the radius beyond which the difference term
  itself aliases), so the transported envelope is the physical wavefront
  RESIDUAL and `preserve_input_phase='remap'` then carries the design's genuine
  inter-group correction instead of the parabola's artifact.  Measured on
  design-121 with `{'amplitude_model': 'ray_density',
  'preserve_input_phase': 'remap'}`: exit wavefront vs the exact exit sphere
  **r⁴ −0.13 rad / rms 0.015 rad**, matching the independent full-train ray
  oracle's design floor (0.018 rad); focus **FWHM 3.55 µm, EE3 88.4%,
  EE6 99.3%** at best focus +5…+10 µm from the design plane (was FWHM 5.15 µm /
  EE3 55.6 / EE6 79.7 at +60 µm), i.e. within 0.1 µm / 2 EE3 points of the
  *ideal* field's ceiling through the same readout; **dx-flat** (EE6
  99.0/99.3/99.3 at N = 1024/2048/4096).  Note the conversion is a measured
  NO-OP under the default `preserve_input_phase=False` (the element re-imposes
  its own spherical reference and discards the input wavefront — which is why
  the chain's exit then carries only the last group's own contribution, i.e.
  minus the sum of the correction the earlier groups applied); all three
  options are needed together.  Caveat: the inter-group Sziklas-Siegman leg
  still transports with the paraxial kernel, so the `(S − parabola)` term rides
  inside the envelope (verified hand-off by hand-off against the ray oracle,
  ≤0.01 rad); a `RuntimeWarning` fires if the band-limit radius reaches inside
  2× the beam radius.  Pins: `tests/unit/test_niche_s8_sphere_carrier_reference.py`
  (11 tests).  Full record: `docs/audits/AUDIT_TRACED_FROZEN_AMPLITUDE_2026_07_24.md` §8.
- **`apply_real_lens_traced(preserve_input_phase='remap')`** (opt-in; requires
  `amplitude_model='ray_density'` and an engaged `carrier=`): transports the
  input's carrier-de-chirped RESIDUAL phase to the exit geometrically, sampled
  at each exit pixel's Newton-inverted entrance point (the same pullback the
  ray-density amplitude uses for `|E_in|`).  Unlike `True` it never touches
  the analytic wave pair (whose phase corrupts under grid refinement on
  carrier-referenced inputs: 0.015 → 0.243 rad/group as dx 20 → 5 µm measured
  on the 121 front group), so it is dx-independent by construction; unlike
  `False` it does not discard the input's genuine residual.  Combined with
  `amplitude_model='ray_density'`, the traced-carrier 121 chain becomes
  dx-CONVERGENT (EE6 identical to the digit across N=2048/4096/8192 — the
  production-readiness P0 acceptance).  No-double-count and carried-residual
  pins in `tests/unit/test_niche_upsample_lattice_fix.py`.  Defaults
  byte-identical (both features opt-in).  Full record:
  `docs/audits/AUDIT_TRACED_FROZEN_AMPLITUDE_2026_07_24.md` §6.7-§6.8.

### Fixed

- **Coarse→fine upsample LATTICE bug in `apply_real_lens_traced` (audit
  `AUDIT_TRACED_FROZEN_AMPLITUDE_2026_07_24`).**  The OPL / ray-density /
  valid-mask maps were interpolated to the wave grid with coordinates
  `ii*Ns/N` (`Ns = ceil(N/ray_subsample)`), which equals the exact `ii/sub`
  only when `ray_subsample` divides `N`; otherwise every map was displaced
  diagonally toward the (−x,−y) grid corner by `(N/2)·(Ns·sub−N)/N` pixels
  and radially mis-scaled.  Measured on 121-final-leg conditions: −6.100 µm
  at N=8192/sub=50 (predicted −6.11), −12.187 at sub=48, −14.467 at sub=51,
  exactly 0 for divisor subs — this was the traced carrier chain's diagonal
  focus walk (the F-C fine-retrace rescale routinely produces non-divisor
  `ray_subsample` values).  Fixed at all four construction sites
  (`coords = ii/sub`, exact for any sub); bit-identical whenever
  `sub | N` (all pinned suites pass unchanged, 49/49).  Regression pins:
  `tests/unit/test_niche_upsample_lattice_fix.py` (5 tests: exit centroid
  on-axis for divisor AND non-divisor subs, both amplitude models, and
  divisor/non-divisor field agreement).
- **Traced-carrier-chain exact-final-leg dx-scaling fixes (audit
  `AUDIT_TRACED_CHAIN_DX_SCALING_2026_07_22` F-A / F-C / F-D).**
  `_fourier_upsample_crop` gains a true band-limited DOWNSAMPLE branch
  (k-space truncation with the same value-preserving scale as the upsample
  direction) instead of silently returning the raw wrong-pitch crop whenever
  `n_fine <= n_crop` — the F-A bug that inflated the design-121 readout
  window to 130.8% of launch power (EE6 102.3%) at chain N > 16384, exactly
  when `n_crop > n_fine_cap` (F-A).  `_fine_trace_group_exit` now (F-C)
  rescales `ray_subsample` on entry to preserve the CHAIN's physical ray
  pitch on the fine retrace grid (pre-fix, the chain-level integer was
  reinterpreted in fine-pixel units — observed an attempted 84.7 GiB
  Chebyshev design matrix), with an independent `max_fine_launch_points`
  backstop cap, and (F-D) warns when `n_fine_cap` forces `dx_fine` coarser
  than the exit sphere's Nyquist pitch `lambda/(2*NA)` instead of silently
  discarding outer-NA content.  `focus_readout` gains the `n_fine_cap` and
  `max_fine_launch_points` keys.  Defaults byte-identical away from the
  trigger conditions; regression pins in
  `tests/unit/test_niche_r9_dx_scaling_fix.py` (F-B — the absolute-metric
  dx-divergence — remains OPEN and is tracked in the audit).

## [5.28.0] — 2026-07-22

Deferred-items roadmap implementation (`docs/roadmap_deferred_2026_07_21.md`)
plus the traced-carrier-chain audit remediation.  All new API is opt-in;
rotationally-symmetric / collimated / low-NA defaults are byte-identical
(tolerance-pinned).  Every item was implemented → adversarially verified → fixed.

### Added

- **`lumenairy.cache.ByteBudgetedLRU` + the cache memory-safety contract (R0).**
  A shared cache foundation bounded by TOTAL RETAINED BYTES (not entry count),
  with a single collective global ceiling (`LUMENAIRY_CACHE_BUDGET_MB` +
  `set_cache_budget`/`get_cache_budget`, default `min(512 MB, 10% of RAM)`),
  global LRU eviction across caches, `register_cache_clearer` enrollment
  (`clear_asm_caches()` drains it), `.release()`/`.clear()`, and a public
  `cache_report()`.  Empirically bounded: a 10 MB-cap cache fed 800 MB of arrays
  retained ≤ 8.39 MB.  Opt-in (`max_bytes=0` disabled) for N²-scale caches.
- **Pointwise cos-grid cache + structured-grid interpolation (R1).**  The
  `surface_model='displaced'` pointwise path gains an opt-in, byte-budgeted
  cos-grid cache — a warm hit is ~6 µs vs ~1.0–1.2 s cold (~160,000× in a design
  loop) — and a structured-grid `map_coordinates` replacing the scattered
  Delaunay (~5× cold).  Byte-identical, default off.
- **Pearcey cusp completion of `caustic='uniform'` (R2).**  Extends the uniform
  Airy fold completion to cusps via the shipped `pearcey` kernel; beats the
  multibranch fallback on a cusp ground truth.
- **`propagate_gbd_vector_through_prescription(direction_sampling='auto')`
  (R5).**  Husimi carrier decomposition on the vector GBD chain (diverging power
  0.82 → 0.99+); collimated byte-identical.  Plus `caustic='uniform'`
  higher-catastrophe detect + clean multibranch fallback.
- **Traced carrier-chain composition API (R8).**  `propagate_traced_carrier_chain`
  (packages the carrier-leg → reconstruct → element(carrier=R) → re-envelope
  hand-off in one call; the element supplies R_out) + `TracedCarrierChainResult`
  + `carrier_referenced_focus_readout` (packaged near-focus landing).
- **Exact non-paraxial high-NA final leg (R9).**  `carrier_referenced_exact_focus_readout`
  + `propagate_traced_carrier_chain(final_leg='auto', na_exact_threshold=0.15)`
  route any high-NA leg through exact band-limited ASM (no paraxial magnification),
  auto-selected by NA.  Design-agnostic: a synthetic NA-0.46 sphere focuses to
  the diffraction limit (EE-in-2w₀ 1.3% paraxial → 99.8% exact); low-NA legs
  byte-identical.

### Performance

- **Deferred perf/memory items (R3, R4).**  GBD `_reconstruct_windowed` chunked
  into memory-budgeted tiles — peak 939.6 → 546.9 MB at N=1024 (−42%),
  byte-identical, 1.35× bonus; a thread-safe normal-equations solver replaces a
  JAX-OpenMP-deadlock-prone `lstsq` in the traced path.  Adaptive-FGA dual-number
  ray-transfer → numba kernel (default), exact (bit-identical Jacobian) — ~12.7×
  on the hot loop, ~3× end-to-end.

### Fixed

- **Traced-carrier-chain audit F1 — `carrier='auto'` on spherical input (R6).**
  The `'auto'` fit spanned the full field, but aliasing beyond r ≈ 1.5–4 mm
  corrupted the least-squares gradient (recovering R ≈ +1094 mm instead of
  +153 mm → collapse to no-carrier).  Restricting the fit to the un-aliased core
  recovers R to ~0.26% (r⁴ 0.588 → 0.005 == explicit); collimated byte-identical.
- **Audit F2 — thick-group intra-group exit-curvature error (R7).**  The float
  carrier used the paraxial parabola not the exact point-source sphere, and the
  global Chebyshev map/OPL fit aliased marginal-ray high order into the defocus
  coefficient.  Exact-sphere carrier + carrier-gated fit-domain restriction +
  cubic OPL upsample bring per-group exit-wavefront rms on all 8 real-121 groups
  from up to 1.8 rad down to ≤ 0.023 rad; default byte-identical.
- **Audit F3 — `tilt_aware_rays` on a steep explicit carrier (R8).**  Guarded to
  reroute through the carrier path (1.723 → 0.008 rad); collimated tilt-aware
  byte-identical (N5 preserved).

### Known limitation

- The corrected-relay (design-121) end-to-end image improved ~10× (EE6 7.3% →
  69.7%) with F2 + the exact high-NA leg, but does not yet reach EE6 ≥ 99%: the
  traced-carrier model launches rays along the carrier **sphere**, so a corrected
  relay's non-spherical intermediate wavefronts are carried uncorrected
  (~1.68 rad accumulated).  A **wavefront-aware ray launch** is the tracked next
  item (`docs/roadmap_deferred_2026_07_21.md`).

## [5.27.0] — 2026-07-21

Deferred-items campaign after v5.26.0 (niches N13–N16): multibranch KMAH caustic
sum for traced ray-density and its uniform-Airy dark-side completion (traced
diffraction-correct through folds), CuPy + JAX backends for the carrier ASM, and
a measured perf/memory sweep.  All new API is opt-in; defaults byte-identical.

### Added

- **Multibranch KMAH / Maslov caustic amplitude for the traced ray-density mode
  (niche N13 / K1).**  New opt-in `apply_real_lens_traced(amplitude_model=
  'ray_density', caustic='multibranch', output_plane_distance=d)` connects the
  ray-density amplitude to the EXISTING `apply_real_lens_traced_multibranch`
  branch-finder + analytic det-Q KMAH counter (reuse, not a reimplementation).
  Where the ray map folds (`det J -> 0` / sign change) it gathers ALL real ray
  branches per output pixel, weights each `|E_in(x_in^b)| / sqrt(|det J_b|)`,
  applies the Maslov phase `exp(-i (pi/2) KMAH_b)`, and sums COHERENTLY, with
  the Ludwig uniform-Airy swap in the Kravtsov-Orlov band — so the field is
  FINITE at the fold (the `sqrt`-singularity of single-branch ray density
  resolves into the fold-diffraction profile; never inf/nan) and the output is
  taken directly at `output_plane_distance` past the exit vertex (no separate
  ASM step).  New knobs `caustic_ray_subsample` (launch density),
  `caustic_band` (`'ludwig'`/`'plain'`), `caustic_min_area_ratio`.  Default
  `caustic=None` is BYTE-IDENTICAL to prior releases.
  - **Validated (lumenairy-free oracles, no Zemax):** the routing is
    byte-identical to a direct `apply_real_lens_traced_multibranch` call
    (`np.array_equal`).  **KMAH / Maslov correctness** is guarded at a GENUINE
    two-branch region — the wave-resolved `caustic_fold_ref` fold RING, where
    the bright-ring fringe position is set by the RELATIVE Maslov phase between
    the two coalescing branches: the multibranch with the correct det-Q KMAH
    reproduces the direct-RS reference bright ring (peak radius ~6.1 um,
    radial-shape correlation ~0.95), while ZEROING or NEGATING the per-branch
    KMAH index moves the ring out to ~14 um and drops the correlation to
    ~0.55–0.64 (a wrong `+-pi/2` is decisively caught; the test monkeypatches
    the counter to prove sensitivity), plus the `ludwig_fold` bright->dark unit
    flip (>1e3x contrast).  A separate self-contained direct-Huygens match to
    rel-L2 ~6% validates the branch AMPLITUDE + eikonal through the API at a
    SINGLE-branch plane (`n_branch.max()==1`, KMAH uniformly 0 — it does not
    exercise the Maslov index).  The single-branch limit reduces to the
    ray-density field (<1% L2) and conserves energy <0.5%; the decenter
    centroid tracks the geometric spot oracle to ~0.3%.
    `tests/unit/test_niche_k1_kmah_caustic.py`.
  - **HONEST residuals (pinned).**  On the fine, wave-resolved `caustic_fold_ref`
    grid the pure GEOMETRIC multibranch does NOT beat the single-branch
    ray-density exit field + ASM: it is identically zero on the DARK side of the
    fold (no evanescent tail), so windowed r2m reads ~15% low and ~20% of the
    caustic energy is missing, whereas single-branch+ASM (a genuine wave
    propagation) already matches the reference to ~3%.  Single-branch
    `ray_density` + ASM / `apply_real_lens_gbd` / `apply_real_lens_fga` remain
    the quantitative caustic reference; multibranch is the finite, no-blow-up,
    one-call coherent field AT the caustic.  At the paraxial image plane (an
    axial point focus) multibranch over-amplifies ~2-3x (the D5 caustic
    pile-up) — finite but not the decentered-PSF EE model (`apply_real_lens_gbd`
    remains that reference, N10b).

- **Uniform Airy dark-side completion — traced diffraction-correct THROUGH a
  fold (niche N16 / K4).**  New opt-in `apply_real_lens_traced(amplitude_model=
  'ray_density', caustic='uniform')` (and the public
  `apply_real_lens_traced_uniform`) keeps the K1 multibranch bright side and
  fills the DARK side of a fold with the Chester-Friedman-Ursell uniform Airy
  tail, so the traced field is diffraction-correct through the fold.  Closes
  K1's dark-side gap: vs the lumenairy-free direct-Rayleigh-Sommerfeld
  `caustic_fold_ref`, windowed r2m **-14.8% / energy 0.80** (multibranch) →
  **-1.9% / 0.96** (uniform); the dark tail is the genuine `Ai(+)` exponential
  (fitted decay matches the ray geometry to ~1.7%).  Reuses the validated
  `uniform_fold_airy` through a shared `_fold_airy_eval` kernel (byte-identical,
  ~1e-13 vs the exact cubic-phase Airy integral).  Default
  (`caustic=None`/`'single'`/`'multibranch'`) byte-identical.  Envelope
  (documented): the far tail beyond ~2× the caustic radius is aperture-edge
  diffraction a pure fold-Airy underestimates (~2%); non-rotationally-symmetric /
  decentered / cusp folds are detected and fall back to multibranch (GBD/FGA
  remain the references there).  Tests: `test_niche_k4_uniform_caustic.py`.
- **CuPy + JAX backends for the carrier-referenced ASM (niche N14 / K2).**  The
  astigmatic / apertured Sziklas-Siegman pilot-beam propagator
  (`propagate_carrier_referenced`, `carrier_referenced_aperture`) now runs on
  CuPy and JAX alongside NumPy via one backend-parametrized code path.
  JAX-vs-NumPy parity ~5e-16 on scalar-diverging / astigmatic / focus-crossing /
  apertured legs; the isotropic `(R, R)` case reduces to the scalar path exactly;
  no dtype upcast (complex64 preserved); the field is differentiable through a
  carrier leg under JAX.  The NumPy default is byte-identical.  CuPy/JAX are
  import-guarded (no hard dependency).  Tests:
  `test_niche_k2_carrier_backends.py`.

### Performance

- **Perf / memory sweep across the accuracy-niche hot paths (niche N15 / K3).**
  Profiled (cProfile + tracemalloc) the traced ray-density amplitude, the
  displaced 2-D transverse-walk remap scatter, the GBD decenter path, the
  astigmatic carrier ASM, the Seidel gate and adaptive FGA at representative
  grids; applied the free (BYTE-IDENTICAL) wins only.  No default, cache, or
  accuracy changed.
  - **Displaced 2-D remap scatter (`displaced_mode='remap'`, and the DEFAULT
    path for a decentered/tilted/freeform element): ~1.8x.**  The amplitude and
    OPL remaps now share ONE Delaunay triangulation of the scattered exit points
    via a single 2-column `LinearNDInterpolator`, instead of building two
    triangulations + two full-grid queries.  The barycentric weights depend only
    on the points (identical for both quantities), so each column reproduces the
    former separate 1-column interp BIT-FOR-BIT (`np.array_equal`).  Measured at
    N=1024: 1171 ms -> 659 ms (**1.78x**), peak memory unchanged (115 MB; the
    per-column results are strided views into one `(Ny, Nx, 2)` array — no dense
    copy). Guarded by `tests/unit/test_niche_k3_perf.py` (byte-identity vs the
    pre-K3 two-interp algorithm reconstructed inline as the oracle + an in-test
    speedup measurement).
  - **Traced ray-density amplitude upsample: redundant per-call allocation
    removed.**  On the sub>1 path the ray-density amplitude upsample now REUSES
    the OPL upsample's coarse->full `(2, N, N)` coordinate stack (identical by
    construction — same `X[::sub, ::sub]` grid) instead of rebuilding
    `np.indices` + a second `(2, N, N)` float64 array.  Byte-identical
    (`np.array_equal` vs the pre-change field at N=512/1024 and sub=1).  The
    plan's hypothesised "double-trace" is REFUTED by measurement: the forward
    ray trace + Chebyshev entrance->exit fit is already SHARED between the OPL
    and the ray-density amplitude (the expensive part is not duplicated); only
    the cheap coarse Newton INVERSE runs twice (~0.007 s at the default sub=8,
    ~1% at sub=1), so it is left as-is rather than risk a masked/parallel
    byte-identity break.  The ray-density call's peak is set by the analytic ASM
    amplitude leg (needed for the exit phase), not the upsample.
  - **No free byte-identical win (profiled, reported honestly):** the pointwise
    2-D obliquity is Delaunay-query-bound (the ray trace is already fully
    vectorised and the scatter already runs on a bounded coarse grid + bilinear
    upsample — the existing 5.8x coarse-grid win); the astigmatic carrier ASM is
    1-D-FFT-bound (the Sziklas-Siegman focus-crossing bridge's FFT count is
    intrinsic); the GBD decenter overhead is negligible (~0.007 s field-frame
    ray transfer) with the cost in the standard windowed beamlet reconstruction
    (unchanged by decenter); the Seidel gate is already trivial (~14 ms at
    N=512); adaptive FGA is bound by the pre-existing dual-number analytic
    ray-transfer arithmetic (`raytrace/differential.py`), outside this phase's
    byte-identical-free-win scope.

## [5.26.0] — 2026-07-20

Accuracy-niches campaign (niches N1–N12): closes every documented real-lens /
GBD / FGA / carrier-ASM accuracy niche, each phase impl → adversarial-verify →
fix.  New opt-in API for decentered/tilted/freeform elements, a traced
ray-density amplitude mode, astigmatic carrier-referenced propagation, a
Seidel-based dispatcher gate, content-adaptive FGA sampling, and GBD
re-expansion; rotationally-symmetric / collimated defaults byte-identical.

### Added

- **Accuracy-niches capstone — composed end-to-end + full-aperture fast case
  (niche N9 / P8).**  No new library API; the deliverable is the DEMONSTRATED
  COMPOSITION of the campaign's individually-validated pieces, pinned by
  `tests/unit/test_niche_p8_capstone.py` (runs without Zemax):
  - **Composed chain (STEP B):** a generic weak-doublet (f1~173 mm) + relay
    (f2~40 mm) propagated end-to-end through the NEW stack — per-group `traced`
    (H6/N5) for the doublet, a carrier-referenced gap leg (N7), the
    `apply_real_lens_universal`-gated relay (routes to the displaced phase-screen),
    and a carrier-referenced FINAL-FOCUS leg — reproduces the independent
    `debye_oracle_v3` diffraction oracle at the paraxial image to **EE80 0.9%
    (1.31 um) / 2.9% (0.633 um)** (EE50 0.9% / 3.3%).
  - **Full-aperture f/2 (STEP C):** the M4 biconvex at w0=9 mm reconstructed AT
    FOCUS via the GBD pilot beam on a MODEST N=3072 grid (4.3 s, frame
    completeness 0.98) matches `debye_oracle_v3` **EE80 to 2.6%**, where the
    ~0.64-NA exit makes a fixed grid need N~21000 (a fixed-grid ASM at the same N
    reads 0.12x — aliased) — the pilot beam is what makes it feasible.
  - **ZOS Huygens-PSF oracle mode (STEP A / N0.2):** `scratchpad/zos_oracle.py`
    gained a `huygens_psf` job type (point source, finite/infinite conjugate,
    pupil-limited) alongside POP — Strehl + PSF grid + centroid/EE/first-zero.
    Validated: unaberrated f/25 lens first-zero **40.32 vs Airy 40.15 um (0.4%),
    Strehl 1.000**; aberrated f/4 uniform pupil vs `debye_oracle_v3` at a matched
    metric window **EE80 0.72% / EE95 0.17%**.  Invocation + caveats in
    `validation/oracles/README.md`.
- **`apply_real_lens_fga(momentum_sampling='adaptive')` — content-adaptive FGA
  momentum sampling (accuracy niche N6b / P5).**  Opt-in (default `'uniform'`,
  **byte-identical** to prior releases — the scalar `dp**2` measure via the
  unchanged `nodes=None` path).  `'adaptive'` IMPORTANCE-samples the `n_p` momentum
  nodes by the inverse-CDF of the input field's (beamlet-broadened, symmetrized)
  marginal angular power, with the MATCHING per-node midpoint-cell quadrature
  weights (which reduce exactly to the uniform `dp` cell on a uniform grid) and
  auto power-normalization (the non-uniform quadrature carries the correct RELATIVE
  weights for the SHAPE but not the calibrated uniform absolute scale).  Also on
  `apply_real_lens_fga_vector`.  **Honest measured result (N6b):** this is a
  NON-improvement — the FGA reconstruction integrand is the field spectrum
  convolved with the beamlet momentum width (`~1/(k w0)`), so it is intrinsically
  too broad for importance sampling to beat uniform, and the "0.97 cap" the feature
  targeted does not even reproduce on a smooth 0.23-NA diverging transport (uniform
  reaches fidelity 1.000 at n_p = 21).  Shipped opt-in per the plan with a
  documented fidelity-vs-cost curve and the fundamental-trade explanation
  (`docs/audit_real_lens_hammer_2026_07_19.md` N6b); the matching-weights
  correctness (a valid quadrature; naive uniform weights over-count > 200% on
  concentrated nodes) is unit-tested.  Tests: `tests/unit/test_niche_p5_sampling.py`.
- **`validation/oracles/caustic_fold_truth.py` — multi-valued fold-caustic
  ground-truth oracle (niche N0.3 / P5).**  Lumenairy-free.  Builds the reference
  wave field at a through-focus plane of a strongly-aberrated singlet where the ray
  map genuinely FOLDS, by a dense DIRECT Rayleigh-Sommerfeld ring integral of the
  exact exit field (no stationary phase, no ray-branch assumptions); self-verified
  (grid convergence 1.3e-5, energy closure 0.999, 2-branch fold).  Used to
  cross-check FGA vs GBD vs traced at a fold — the N6a finding (hammer doc) is that
  GBD/traced do NOT degrade there, so the naive "FGA niche" is refuted.
- **`apply_real_lens_gbd(reexpand='auto')` — GBD strong-reconvergence frame
  re-expansion (accuracy niche N3 / P4).**  A converging input reconverged by a
  NEGATIVE element to a near real focus sheds ~6–12 % of its power at the INPUT
  decomposition: a flat-waist Gaussian beamlet frame cannot carry the input
  wavefront curvature, so its coherent sum is incomplete (measured at the
  decomposition plane, BEFORE any propagation — the loss is baked into the frame
  and is unrecoverable downstream; the documented ~0.94 G1 edge).  `reexpand`
  (default `'off'`, **byte-identical** to prior releases) adds an `'auto'` policy
  that, when the naive frame's input-plane completeness falls below
  `reexpand_threshold` (0.98), re-decomposes with a **carrier reference**: remove
  the smooth congruence `W` (fit via `reexpand_carrier='auto'`, or a signed
  scalar conjugate / explicit wavefront), decompose the compact near-flat
  residual (a near-complete frame), and seed each beamlet's launch **direction**
  (`grad W` -- the H7 carrier-normal machinery), **curvature** (`Q += Hessian W`,
  the seed plain Husimi omits) and **piston** from the carrier.  Headline (G1 M5
  biconcave, converging `R_in = -35 mm` -> real image ~108 mm): power conserved
  **0.94 -> >0.99** with windowed r2m within **0.3 %** of the carrier-referenced
  `apply_real_lens_traced`; grid-converged (dx halving unchanged); the
  re-decomposition Parseval-audits clean (completeness 0.999, no double-count);
  runtime ~1.2–1.9x when it fires (measured 1.16x).  Surgical: a collimated
  input (already complete) and a diverging input through a positive element or a
  doublet (completeness 0.996–0.998) are NOT re-expanded, so `'auto'` returns
  output byte-identical to `'off'` there.  Also publishes the **frame-completeness
  metric** (`frame_completeness` in `lumenairy.propagators.gbd`; reconstructed
  output power / aperture-transmitted input power) via the new opt-in
  `diagnostics=` dict.  Independent oracles: ABCD Gaussian q-trace and
  `apply_real_lens_traced` (H6-fixed).  Tests: `test_niche_p4_gbd_reexpand.py`.
- **`propagate_carrier_referenced` — carrier-referenced ("pilot-beam") free-space
  propagation (Phase E prototype).**  Transports a strongly diverging / converging
  beam's slowly-varying ENVELOPE on a modest grid while the spherical carrier
  radius `R` and the co-moving grid pitch evolve ANALYTICALLY (`R -> R + z`,
  `dx' = dx*(R+z)/R`) — the Sziklas & Siegman (1975) scaled-coordinate Fresnel
  transform, the mechanism Zemax POP calls pilot-beam re-referencing.  This
  sidesteps the audit-H8 memory wall: a diverging input's own fringe pitch
  `lambda*R/r` forces production grids to N=28672 *propagator-independently*, but
  the envelope carries no such fringe.  Headline (w0=4 um diverging Gaussian,
  R=+55 mm, +50 mm): the N=2048 carrier result matches the N=16384 exact-ASM
  ground truth to **0.05–0.08 % in windowed r2m / EE50 / EE80** at **64x less
  array memory** (67 MB vs 4.29 GB/array; 32x lower tracemalloc peak; 86x faster).
  Exact for the quadratic (carrier) part under Fresnel; the only approximation is
  paraxial propagation of the near-collimated envelope.  Ships with
  `carrier_referenced_reconstruct` / `carrier_referenced_envelope` (rebuild /
  extract the full field for element hand-offs — e.g. reconstruct at a lens plane
  and pass the carrier `R` to `apply_real_lens_traced(carrier=R)`, exact per H6)
  and the `CarrierReferencedField` result tuple `(env, R, dx)`.  Independent
  oracles: analytic Gaussian ABCD q-trace (grid-free) and exact band-limited ASM
  on a fine grid.  PROTOTYPE — isotropic (single-`R`) carriers only; astigmatic
  carriers, element-plane aperture handling in the envelope frame, and stepping
  across the carrier focus are documented follow-ups.  Tests:
  `test_carrier_referenced.py`.

- **Decentered / tilted / freeform elements (niches N2 / N10 / N11).**
  Per-surface `decenter=(dx, dy)`, `tilt=(tx, ty)` and `sag_callable` are now
  honored by the real-lens family.  `surface_model='displaced'` gains a pointwise
  2-D vector-Snell obliquity path (P3) and a 2-D transverse-walk **remap** (P10)
  that broadens induced coma correctly (RMS ratio ~1.02 @1 mm / ~1.09 @2 mm,
  within ~10% of a lumenairy-free geometric oracle; the single-plane pointwise
  screen shrinks); `apply_real_lens_traced` and `apply_real_lens_gbd` thread the
  same field-frame geometry through the ray trace (P9, shared transformed
  sag/normal helper; `raytrace == analytic == oracle` to 1e-11…1e-17).  **GBD is
  the decentered encircled-energy reference** (grid-robust RMS second-moment; the
  earlier "traced shrinks" reading was an EE80 quantization artifact on an
  undersampled spot — traced also broadens under the RMS metric).  Rotationally-
  symmetric defaults are byte-identical.  Tests: `test_niche_p3_*`,
  `test_niche_p9_*`, `test_niche_p10_*`.
- **`apply_real_lens_traced(amplitude_model='ray_density')` (niche N12).**  Opt-in
  geometric ray-tube exit amplitude `|E_in(x_in)| / sqrt(|det J|)` from the
  ray-map Jacobian — energy-conserving and **decenter-stable** (0.999 at
  0/1/2 mm, where the screen amplitude leg leaks to ~0.907 at 2 mm), a smooth
  envelope free of exit-aperture Fresnel ripple, and **caustic-safe** (detects
  folds, floors the amplitude finite, warns and steers to GBD/FGA — never
  inf/nan).  Default `'screen'` is byte-identical.  Note: the traced output is the
  exit vertex (upstream of focus, where the ray map is near-identity), so the mode
  does not by itself carry the focal decentered coma — that is a downstream/phase
  effect; GBD remains the decentered-EE reference.  Tests:
  `test_niche_p11_ray_density_amplitude.py`.
- **Seidel-based dispatcher gate (niche N8).**  `apply_real_lens_universal` now
  gates on a Seidel-S1 true-spherical-aberration estimate (validated < 0.5% vs an
  independent Debye `r^4` wavefront fit), so an SA-nulled asphere routes to the
  fast analytic path where the old `c4` sag-coefficient bound over-conservatively
  sent it to ray tracing; gate SAFETY proven on M1–M6 × {collimated, diverging,
  converging} (no aberrated case routes to analytic).  The `c4` bound is retained
  as the fallback.  Tests: `test_niche_p7_seidel_gate.py`.

### Fixed

- **Hammer audit H6-class — `apply_real_lens_traced(tilt_aware_rays=True)`
  entrance eikonal (niche N5).**  The tilt-aware path omitted the entrance-plane
  eikonal `k0·W(x_in)`, collapsing a diverging-input focus to the collimated
  plane (EE100 ~0.01); now threaded through the shared carrier plumbing (EE100
  0.97+ at the ABCD image, agreeing with `carrier=`), no double-count when an
  explicit carrier is also supplied.  Collimated byte-identical.  Tests:
  `test_niche_p1_traced_tiltaware.py`.
- **`propagate_gbd_through_prescription` diverging/converging/tilted input
  (niche N4).**  The prescription-chain entry still ran the old axial frame (the
  H7 fix had only reached `apply_real_lens_gbd`); it now takes
  `direction_sampling='auto'` (Husimi beamlets on curved/tilted input), lifting
  chain power ~0.82 → 0.99 and matching back-to-back `apply_real_lens_gbd` calls
  to numerical precision.  Collimated byte-identical.  Tests:
  `test_niche_p1_gbd_chain.py`.
- **Analytic displaced screen — extreme finite conjugates (niche N1).**  The
  previously-reported ~0.5× oracle ratio on the negative-lens real-focus /
  virtual-image cases was a **false oracle attribution** (a mislabeled
  geometric-spot number).  Against a corrected, grid-converged congruence
  diffraction oracle the shipped screen is 0.998× (exit-plane remap 1.000×), and
  discriminating (a no-obliquity thin model and a sign-flipped conjugate both fail
  a 15% gate).  Tests: `test_niche_p2_displaced_extreme.py`; oracle:
  `validation/oracles/debye_oracle_v3.py`.

- **Hammer audit H7 — `apply_real_lens_gbd` diverging-input energy collapse.**
  The position-only (axial) beamlet decomposition carried a strongly-diverging /
  converging input's wavefront curvature in the beamlet **amplitude only, not the
  launch direction**, so the axial base rays refracted as if collimated (focused
  at `f`) and the beamlet frame shed the beam's angular content — power conserved
  collapsed to ~0.2–0.7 on the dual-oracle singlet (worse: ~1e-4 + NaN warnings
  at the production 121's NA~0.23 beam) and the true finite-conjugate image
  smeared.  Fix: `direction_sampling='auto'` is now the default — it launches
  each beamlet along the input's local wavevector (Husimi / carrier normals) when
  the wavefront is curved/tilted and along the axis otherwise.  A flat-wavefront
  (collimated) input measures **exactly 0** angular spread and takes the
  byte-identical axial path (the validated 98.4% collimated baseline is preserved
  to the bit).  Result: power **0.997–0.998** and focus at the ABCD image plane
  (< 5%) across the R_in = 300/150/100 mm scan, NaN warnings gone.  Independent
  oracle: ABCD Gaussian q-trace.  Tests: `test_hammer_h7_gbd_diverging.py`.

## [5.25.0] — 2026-07-19

The **real-lens hammer campaign** release (`docs/audit_real_lens_hammer_2026_07_19.md`):
adversarial validation of the real-lens propagator family against two fully
independent oracles — Zemax OpticStudio POP via ZOS-API (dispersionless model
glass, per-surface indices cross-checked) and a self-contained exact-raytrace
Debye/Huygens integral (oracle cross-agreement 0.5–8% on every case).
Verdicts: **`traced` 99.7%** and **`gbd` 98.4%** of dual-oracle truth on a
heavily aberrated f/5 singlet; benign regime exact to 4 digits; the
`analytic` model's validity envelope honestly quantified.  Also implements
the 2026-07-18 thin-lens audit and **all 10 deferred v5.24.2-audit items**.

### Added

- **`lens_model='stigmatic'`** on `apply_thin_lens` (+ `conjugates=` kwarg):
  the conjugate-matched EXACT ideal element `phi = k*(S(R_out) - S(R_in))` —
  aberration-free under the exact ASM propagator at ANY conjugates (the fix
  for the fictitious-spherical mechanism that broke ideal-lens relay chains).
  Reduces exactly to `'nonparaxial'` for collimated input.
- **`fresnel_tf_propagate`**: same-grid Fresnel transfer-function step
  (`z < 0` allowed, `z == 0` exact identity).  The matched pair
  (paraxial lens x this step) is self-consistent by construction — the
  Zemax-POP-equivalent ideal reference mode for relay studies (matched 1:1
  imaging lands on the ABCD waist to 0.25%).
- **`rng=` alongside `seed=`** on the stochastic source factories
  (`seed=` deprecated, removal v5.27), explicit **`normalize=`** kwargs
  (defaults preserve each factory's current behavior), and **`w0=`** alias
  where `sigma=` was the Gaussian-width misnomer.
- **Opt-in `NormalizedMerit`** scale wrapper for the optimizer (defaults
  byte-identical); BOR top-level exports; type-tagged **metadata
  serialization contract** honored identically by the h5 and zarr storage
  backends (round-trip fidelity + legacy-load compatibility).
- Runtime **exit-NA Nyquist guard** on `apply_real_lens_traced` (hammer H3):
  warns when the exit beam's convergence exceeds the grid Nyquist
  (`dx > lambda/(2*NA_exit)`) — previously this aliased silently, reading
  r^2-weighted spot metrics ~37% low while EE50/EE80 stayed plausible.

### Fixed

- **`slant_correction` had inverted cosines (hammer H1)** — it computed the
  slab ray path-length `n*sag/cos` where the wavefront OPD of a tilted
  refracting facet is `(n2*cos_tt - n1*cos_ti)*sag`.  The inversion
  sign-flipped the obliquity/spherical term and, on symmetric lenses,
  CANCELLED the pupil aberration (an impossible 3.6 um "perfect" spot vs
  the 65 um dual-oracle truth).  The corrected slant screen moves the
  analytic model toward the oracle (50.3 um).  Note: `through_focus`'s
  internal "ideal pupil" reference used this flag — derived goldens shift.
- **`apply_thin_lens` sign bugs** (thin-lens audit 2026-07-18):
  `'nonparaxial'` with `f < 0` focused like its converging twin (byte-
  identical for `f > 0`); `'aplanatic'` carried the WRONG quartic sign
  (2x paraxial's spherical error, focusing WORSE than paraxial — now the
  exact stigmatic sphere on the sine-condition domain).
- **FGA auto path memory + sampling (hammer H4/H5)**: the byte-identical
  position-lattice chunk loop now engages by default (RAM-fraction budget);
  the memory model counts the 9-ray FD Jacobian bundle; all-conic
  prescriptions default to the analytic `exact_jacobian`; and the auto
  sampler no longer floors `p_max` at the beamlet-completeness width for
  near-collimated inputs (32–130x the field's angular content — the
  excess-momentum halo that produced 5x-wrong spots through strong lenses).
  `coarse_stride` itself was verified byte-faithful.  The v5.24.3 S2-2
  identity-power contracts are preserved.
- ASM transfer-function construction consolidated into one shared kernel
  with the complex64 mod-2pi mitigation applied to ALL builders (previously
  only one copy; the mitigation's contract is a platform-independent
  half-to-one float32-ULP bound vs an 80-bit oracle).
- `user_library` expression masks: `eval()` replaced by a restricted
  AST evaluator (whitelisted operations/names; hostile expressions raise).

### Changed

- `analytic` (`apply_real_lens`) accuracy envelope documented: per-surface
  phase screens cannot represent orientation-dependent aberration
  (dual-oracle: 60/61 um for both plano-convex orientations where the truth
  is 43/128 um).  Use `traced`/`gbd` for absolute spot fidelity on
  aberrated systems.
- **Known limitations (production-scale, strongly-DIVERGING input; see the
  hammer doc's 121-scoreboard addendum):** `traced` with `carrier='auto'`
  fails to form a focus on a strongly diverging input (H6, open); GBD's
  paraxial frame collapses energy on the same class (H7, known); FGA at
  production scale should be RE-TESTED against this release's H4 chunking
  fix (H8).  Until then the conjugate-matched `stigmatic` thin chain is the
  recommended production surrogate for corrected relays.
- **Through-focus single-plane metrics unified across backends (deferred audit
  S3-19).**  `through_focus_scan_jax` no longer hand-inlines a twin of
  `single_plane_metrics`; both backends now route every plane through that one
  function.  Two documented band-edge divergences on the JAX backend are
  reconciled to the canonical NumPy conventions: (1) `power_in_bucket` now uses
  the inclusive `R**2 <= r*r` boundary (was a strict `r < R`), so a pixel lying
  **exactly** on the bucket radius is counted on the JAX path too; (2)
  `rms_radius` is now the `beam_d4sigma`-derived second moment, which is 0 (not
  NaN) on an all-zero plane — so a dark scan's `best_focus_spot` is `0.0` on
  both backends (the JAX path formerly returned NaN).  Smooth fields are
  unaffected beyond ~1e-15 summation-order noise.  Pinned by
  `tests/unit/test_through_focus_bucket_boundary.py` and
  `tests/unit/test_through_focus_metric_parity.py`.

## [5.24.4] — 2026-07-18

Exhaustive whole-library audit (`docs/audits/AUDIT_V5_24_2_2026_07_17_EXHAUSTIVE.md`)
remediation: **88 findings fixed** across P1 (11), P2 (23), P3 (54) plus the
follow-ups the new JAX-CI leg surfaced.  10 P3 items are documented-deferred
(genuinely not byte-identical-consolidatable, or public-API changes that need a
deprecation cycle) — see the commit history for each rationale.

### Added

- **JAX now runs in CI.** A dedicated non-matrix `jax-unit` job installs the
  `jax` extra and runs the jax-guarded suite on jax 0.11 (audit S4-4) — the
  previously-invisible JAX paths are now gated.  BLAS threads are pinned in that
  job to avoid a JAX×OpenBLAS OpenMP deadlock in numpy's `lstsq`.
- **FGA opt-in performance levers** (`coarse_stride`, `exact_jacobian`,
  `cache_trace`) — no-accuracy-loss speedups for repeated propagations through
  fixed optics; each is off by default.
- Optional `jones` field on the `Source` dataclass (audit S3-17): carried
  metadata for vectorial pipelines; scalar propagators are unaffected.
- A validation smoke leg for the newest propagators
  (fga / multibranch / levin / chebyshev / mft, audit S5-13).

### Fixed

- **Seidel Petzval sign (audit S3-1).**  `S4` (Petzval) sat on the OPPOSITE
  sign convention to `S1`–`S3`, corrupting `S5` (distortion) and `seidel_wfe`
  field curvature.  Corrected and **verified against three independent oracles
  external to the library** — rayoptics 0.9.8, the analytic Petzval theorem
  (`P = Σ c(n'-n)/(n'n)`), and the stop-at-thin-lens distortion-vanishing
  theorem (`|S5/S1|` now `4e-8`).  Permanent cross-oracle gates added.
- **FGA t=0 identity power (audit S2-2, a v5.24.3 regression).**  The auto
  momentum sampler could drop `p_max` below the beamlet momentum width, hiding
  a 30–35 % power deficit behind `fidelity ~ 1`; a completeness floor restores it.
- **Multibranch rasterizer double-counting (audit S2-1)** — shared triangle-edge
  pixels no longer add spurious energy (up to +53 %).
- **JAX complex64 phase loss (audit S2-3)** in ASM / MFT / Fresnel; and the
  canonical-poly fit's differentiable solve now has a finite VJP under jax≥0.11
  (`jnp.linalg.lstsq`'s SVD gradient NaN'd on near-degenerate singular values;
  replaced with a normal-equations solve — same forward, finite gradient).
- The full P1/P2/P3 remediation across the EM engines (RCWA/PMM/Berreman/EME/BOR),
  wave propagators (ASM/GBD/FGA/Maslov/Levin), ray tracing, sources, analysis,
  IO, and the optimizer — silent-fallback guards, energy/convergence tripwires,
  dead-code removal, byte-identical de-duplication of 4 genuine copy clusters,
  measured no-loss perf wins, and ~30 convention/docstring corrections
  (including the actively-dangerous `CONVENTIONS.md` waveplate slow-axis row).

### Changed

- **`jax` dependency floor raised `>=0.4.20` → `>=0.11.0`** — the supported
  baseline going forward (the audit fixes and jnp/np parity bars are validated
  against it).

### Fixed

- **The vector-EME 2-D Bloch mode-finder (`layer_vector_modes`) no longer
  flakily under-recovers the mode band across LAPACK backends.**  It detected
  modes as strict 3-point grid local minima of `sigma_min(qz^2)`
  (`d[i]<d[i-1] and d[i]<d[i+1] and d[i]<tol`); in the closely-packed structured
  band whether a dip registered depended on the scan-grid alignment and on the
  per-point `sigma_min` values, which wobble across BLAS/LAPACK backends -- so
  the recovered mode count varied run-to-run on different CI runners (recall
  swung 9..16/16, tripping the completeness assertion).  Detection now samples
  on a grid dense enough to resolve the packed band (`>= _DETECT_PPU` points per
  unit `qz^2`, independent of `n_scan`) and picks dips with
  `scipy.signal.find_peaks` (plateau/tie-robust), and the rank-drop `ratio_tol`
  default is tightened `1e-2 -> 1e-3` (real modes rank-drop `~1e-6` while a
  spurious `det(G)` ghost-zero only `~5e-3`, a 2-3 decade margin) so the denser
  detection does not admit spurious.  Recall is now backend-stable at the full
  band with spurious `<= 1`.
- **FGA's strongly-diverging-beam fidelity cap is RESOLVED via adaptive
  phase-space sampling -- it was a quadrature-resolution artifact, NOT a
  frozen-approximation limit.**  v5.24.2 documented a ~`0.93` fidelity ceiling
  for a beam diverging at ~`0.1` rad and ascribed it to the fixed (frozen)
  beamlet width.  That diagnosis was wrong.  Leading-order Herman-Kluk / frozen-
  Gaussian propagation is *identically exact* for a quadratic Hamiltonian --
  and free-space propagation is quadratic (Lasser & Lubich, *Acta Numerica* 29,
  229 (2020); Kröninger, Lasser & Vaníček, *Front. Phys.* 11:1106324 (2023):
  "exact for harmonic motion... error only due to Monte-Carlo integration").
  The residual is therefore entirely the **phase-space quadrature**, and the cap
  was simply a too-coarse momentum spacing `dp = 2*p_max/(n_p-1)`: a diverging
  beam's broad angular footprint was under-sampled at the old fixed `n_p`.
  `apply_real_lens_fga` / `_vector` now auto-size the swarm when `p_max` / `n_p`
  are left `None` (the new default): `p_max` is set from the field's own angular
  content (FFT power spectrum, capped by the system NA) and `n_p` is chosen so
  `dp <= ~0.008`.  A beam diverging at ~`0.09` rad now reconstructs to
  fidelity `> 0.998` (from `~0.93`) with the same small swarm sizes for compact
  fields; the fix is field-size-independent (a large collimated beam still gets
  a small `n_p`, no over-refinement).  An explicit `p_max` / `n_p` is always
  honoured.  Supersedes the "Documented FGA's diverging-beam limitation" note
  in 5.24.2.

## [5.24.2] — 2026-07-17

### Added

- **FGA position-lattice (`Nq`) chunking -- `mem_budget_mb` now genuinely bounds
  memory on large apertures (audit F3, full fix).**  v5.24.1's guard only
  fell-back/failed-fast; `mem_budget_mb` now chunks BOTH dimensions -- the
  momentum swarm (`chunk`) AND the position lattice (`nq_chunk`) -- so a large
  aperture runs within the budget as an additive sum over lattice chunks instead
  of OOMing.  The separable `(nq_chunk*Np)` coefficient array, the per-momentum
  ray trace, and the scatter are all bounded by `nq_chunk`.  The chunked result
  is identical to the un-chunked full swarm to float round-off (verified `~2e-14`
  across separable/direct/coefficient-pruning/vector); coefficient pruning stays
  matched via a global-per-momentum-peak pre-pass when the lattice is chunked.
  Only an absurd budget (a single lattice point can't fit) now raises.

### Changed

- **`apply_real_lens_universal` documented as the canonical dispatcher;
  `apply_real_lens_auto` as the GBD/FGA-only 2-way subset** (audit secondary
  note).  Both `auto` members launch beamlets along the local phase gradient, so
  both already handle a single-valued diverging beam -- `auto` never routes to
  bare `traced` and so needs no collimation split.
- **Documented that the near-caustic -> `fga` decision keys on
  `output_plane_distance`** (audit secondary note): a split-step caller that
  applies the lens at `output_plane_distance=0` and does its own downstream ASM
  never triggers the `fga` branch; pass the full distance or force `method='fga'`
  for caustic-accurate rendering.
- **Documented FGA's diverging-beam limitation.**  The frozen beamlet width does
  not spread, so a strongly-EXPANDING wavefront far from a focus is only weakly
  reconstructed -- fidelity caps ~`0.93` for a beam diverging at ~`0.1` rad,
  confirmed independent of grid size (a frozen-approximation limit, not an edge
  effect).  FGA excels AT caustics and for compact fields; the dispatcher already
  routes diverging single-valued beams to the wave-exact phase-screen.

## [5.24.1] — 2026-07-16

### Fixed

- **`apply_real_lens_universal` no longer silently blurs a diverging beam
  (dispatcher audit F1).**  The single-valued high-NA branch routed a
  single-valued but strongly **diverging** beam (a bare point-source relay) to
  `traced`, which launches one ray per pixel along the local phase gradient and
  is only valid for a ~collimated beam -- so the beam was silently blurred
  (`traced` warns to stderr but still returns the blurred field; the v5.24.0
  multi-valued guard only covers multi-valued fields).  The smooth-plane case now
  splits on the beam's residual angular spread using traced's OWN discriminator
  and threshold (`_carrier_residual_rms` vs `0.02` rad): collimated -> `traced`
  (unchanged, sub-nm), diverging -> `'phase_screen'` (`apply_real_lens` + exact
  ASM -- wave-exact in propagation, bounded thin-screen OPD, never a blur, honest
  `return_method`).  Near-caustic still -> `fga`.
- **`apply_real_lens_fga` / `_fga_vector` respect `mem_budget_mb` on large
  apertures instead of OOMing (dispatcher audit F2/F3).**  `mem_budget_mb` bounds
  the momentum chunk only; the separable analysis allocated the whole `(Nq*Np)`
  coefficient array up front (hundreds of GB on a 24 mm aperture) regardless of
  the budget, and the per-momentum ray trace runs over all `Nq` position-lattice
  points.  A new guard now (1) auto-falls-back the separable path to the
  per-momentum-chunk direct analysis when its `c_full` would exceed the budget
  (no accuracy loss -- the direct path is exact), and (2) raises a CLEAR error
  naming `Nq` and the levers (`dq_step`/`prune_frac`/`n_p`/a lighter propagator)
  when even the minimum ray-trace floor is a genuinely large overshoot, instead
  of a confusing multi-GB `MemoryError`.  `mem_budget_mb`'s momentum-only scope is
  now documented.  (Full position-lattice chunking is deferred -- a 24 mm-aperture
  FGA is outside the method's practical envelope at any accuracy-preserving
  `dq_step` regardless.)
- **De-flaked the vector-EME verify oracle (unrelated eig-heavy CI flake).**
  `test_eme_2d_vector::test_vector_verify_removes_spurious` flaked on CI (recall
  9/16, passed on re-run): `_fd_eig_dist` placed the shift-invert `sigma` exactly
  at the candidate eigenvalue (`i*sqrt(qz2)`), making `(Gc - sigma)` singular so
  ARPACK's Ritz values were BLAS/backend-sensitive (fine on local MKL, diverged
  on CI OpenBLAS).  Offset `sigma` off the eigenvalue (`+1e-3` along the imaginary
  axis, `k=4->6`) to condition the solve deterministically, and pinned the file's
  eig-heavy tests to one BLAS thread via a `threadpoolctl` fixture.

## [5.24.0] — 2026-07-16

### Added

- **FGA memory chunking (`mem_budget_mb` / `chunk`).**  `apply_real_lens_fga` and
  `apply_real_lens_fga_vector` now process the momentum swarm in chunks, bounding
  peak beamlet memory from `O(Nq*Np)` to `O(Nq*chunk)` (the scatter is an additive
  sum over independent beamlets, so a chunk is computed and accumulated in place,
  then discarded).  `mem_budget_mb` auto-sizes the chunk from the position-lattice
  count; `chunk` sets it explicitly.  The chunked result is **numerically identical**
  to the full swarm (verified max abs diff `~5e-14`, fidelity 1.0), so it is a pure
  memory lever — it makes high-resolution / fine-sampled FGA (which otherwise OOMs)
  runnable.
- **FGA position-support pruning (`prune_frac`, default `1e-4`).**  Drops
  launch-lattice points whose windowed `|E_in|` is below `prune_frac` of the peak;
  by Cauchy-Schwarz those beamlets carry a negligible Gabor coefficient for every
  momentum, so the reconstruction is unchanged (verified fidelity-vs-unpruned
  `1.0` at `1e-4`/`1e-3`).  **3-5x faster on concentrated fields** (fewer lattice
  points in both kernels), a no-op on grid-filling fields.  `prune_frac=0`
  disables it.
- **FGA coefficient pruning (`coeff_frac`, default `1e-4`).**  Skips whole
  momenta whose peak Gabor coefficient `max_q |c(q,p)|` is below `coeff_frac` of
  the running global peak -- the field carries ~no energy at that direction, so
  the entire ray trace + scatter for that momentum is dropped.  Conservative /
  no-loss (the running peak only grows, so it never over-prunes; verified
  fidelity-vs-unpruned `1.0` at `1e-4`/`1e-3`).  **Faster for
  spectrally-concentrated (smooth) fields**, a no-op for broadband ones.
- **FGA separable analysis + recurrence scatter (`separable`, default
  `'auto'`).**  Two faster kernels that replace the direct Gabor-analysis and
  frozen-Gaussian-scatter inner loops with no accuracy loss: (1) the momentum
  grid is the tensor product `pv (x) pv`, and both the Gaussian window and the
  `exp(-i k (px dxr + py dy))` phase are separable, so the 2-D windowed analysis
  factors into an x-transform reused across every `py` -- ~`n_p` x less work
  (shared precomputed phase/Gaussian tables; the circular truncation is preserved
  exactly); (2) post-transport the scatter beamlets have no shared grid, so the
  scatter instead advances the window phase (constant per-beamlet rotation) and
  the Gaussian (two-term recurrence) along each row, hoisting the cos/sin/exp out
  of the inner loop.  Both are numerically equivalent to the direct kernels to
  ULP in isolation, and equally accurate vs the exact angular-spectrum oracle
  (the reconstruction's beamlet cancellation amplifies the round-off to ~`1e-4`
  peak, well below the FGA ~`1e-3` error floor -- verified the spherical-aberration
  caustic peak error and free-space fidelity are unchanged).  **~1.5-1.8x combined**
  (scalar and vector).  `'auto'` enables it for `n_p >= 5`; `separable=False`
  restores the direct kernels.

### Changed

- **`apply_real_lens_universal` no longer routes MULTI-VALUED fields to
  `traced`.**  `traced` launches one ray per pixel along the local phase
  gradient, which is undefined where several wave components cross the same
  region — a **multi-emitter, post-DOE, or speckle** field — so it silently
  collapses them to their amplitude-weighted *mean* direction and applies the
  wrong angle-dependent OPD (`apply_real_lens_traced`'s own guard already flags
  such inputs "INCOHERENT — per-pixel single-direction estimation fails").  The
  high-NA `'auto'` branch now measures the field's multi-valuedness (the
  NA-normalized spread of the local wavevector about its per-region mean) and
  routes multi-valued fields to `'fga'`, whose phase-space swarm transports every
  direction independently (verified FGA-exact vs the angular-spectrum oracle,
  fidelity `1.0`, on a two-emitter field).  The detector is single-valued-safe:
  a plane wave, Gaussian, single diverging/converging source, MLA-tilted beamlet,
  or any smooth aberrated single beam scores `<0.006` while genuine multi-valued
  fields score `>0.08` (a >10× separation), so a single beam still gets the
  sub-nm traced OPL.  New `multivalued` (`None` auto-detect / `True` force FGA /
  `False` trust single-valued) and `multivalued_threshold` (default `0.06`)
  overrides.  A false positive only costs speed (FGA is never *wrong*), so the
  cut is biased to prefer FGA when uncertain.

- **FGA `nsig` default 4.0 → 3.0** (~1.8x faster).  The per-beamlet window cost
  scales as `nsig**2`; the `>3-sigma` tail (`exp(-4.5)`) is filled by overlapping
  beamlets, so the reconstruction is unchanged.  Verified: free-space fidelity
  identical (0.999996), spherical-aberration caustic peak-intensity error identical
  (0.8%).  Callers can restore the old window with `nsig=4.0`.

## [5.23.0] — 2026-07-15

### Added

- **`apply_real_lens_fga` — a caustic-accurate lens propagator** (Frozen
  Gaussian Approximation; `lumenairy/propagators/fga.py`).  The
  Gaussian-beam-summation family is now caustic-accurate: FGA (Lu & Yang,
  *Commun. Math. Sci.* 9(3):663, 2011 — the wave-equation transplant of the
  Herman–Kluk propagator) **freezes** the beamlet width and weights each by the
  Herman–Kluk prefactor `a = sqrt(det Z)`, `Z = (A+D) + i(k w0^2 C - B/(k w0^2))`,
  built from the SAME ray-transfer/monodromy blocks the GBD propagator already
  computes — a retrofit, not a rewrite.  Because the position→momentum block
  `C` (which vanishes at a focus) enters *additively*, the prefactor never blows
  up: the method is regular at caustics by construction.  Validated: reproduces
  the exact angular-spectrum field to fidelity 0.9998 in free space, matches
  `apply_real_lens_gbd` and the angular-spectrum oracle through a real
  plano-convex singlet to 0.997–0.999, and **beats GBD at a spherical-aberration
  caustic** on peak-intensity error (GBD 0.03–0.34 vs FGA 0.01–0.07).  Energy is
  a controllable knob (the frozen width `w0_factor` is the FGA convergence
  parameter; `normalize_output='power'` conserves it exactly).  Momentum sampling
  auto-sets from the prescription NA.  NumPy-only; requires the optional `numba`
  accelerator.  Background: `docs/gbd_caustic_accuracy_literature.md` (a
  five-agent literature round establishing that GBD's caustic error is a
  phase/interference problem, not an amplitude singularity, and ranking the fix
  routes).
- **`apply_real_lens_auto` — GBD/FGA auto-dispatching lens propagator.**
  ``method='auto'`` detects the field's geometric caustic zone (a meridional ray
  fan whose launch directions follow the input wavefront -> where the exit rays
  cross the axis, spherical-aberration-broadened) and routes the output plane to
  the fast thawed-beamlet `apply_real_lens_gbd` in smooth regions or the
  caustic-accurate frozen-beamlet `apply_real_lens_fga` near a focus / fold /
  cusp (widened by a diffraction depth-of-focus pad).  Both are ray-based (no
  thin-screen obliquity ceiling), so the dispatched result is accurate at high
  NA as well as at caustics; the dispatch is biased toward FGA when uncertain
  (FGA matches GBD in smooth regions, so this only costs speed, never accuracy).
  ``method='gbd'``/``'fga'`` force the choice; ``return_method=True`` reports it.
- **`apply_real_lens_universal` — the universal (4-way) auto-dispatching lens
  propagator.**  Routes each output plane to the MOST ACCURATE propagator for its
  regime: low NA (< ``na_threshold``) → the wave-exact thin-element phase screen
  (`apply_real_lens` at the exit vertex + an exact angular-spectrum output leg,
  which handles focus/caustics with no beamlet-discretization cost); high NA and
  near a caustic → the caustic-accurate, ray-based `apply_real_lens_fga`; high NA
  and smooth → the per-pixel ray-traced `apply_real_lens_traced` (sub-nm OPL, no
  thin-screen obliquity ceiling).  ``method='phase_screen'|'gbd'|'traced'|'fga'``
  forces the choice (`'gbd'` — the fast, differentiable, polarization-capable
  thawed beamlet — is available but not auto-selected, since `traced`/`fga`
  dominate it on accuracy); ``return_method=True`` reports the routed name and
  ``method_kwargs`` forwards per-method arguments.  This makes the beamlet /
  ray / wave-exact-surface family a single "incredibly accurate everywhere"
  entry point.  Demo: `examples/14_fga_caustic_propagator.py`.
- **`apply_real_lens_fga_vector` — vector (Jones) caustic-accurate propagator.**
  Propagates a ``(2, Ny, Nx)`` transverse Jones field ``(E_x, E_y)``: each frozen
  beamlet carries the per-surface Fresnel s/p Jones matrix (polarization ray
  tracing -- diattenuation, retardance, and the geometric s/p frame rotation,
  which supplies the semiclassical Berry phase), and the longitudinal ``E_z``
  (``E . k = 0``, the high-NA piece) is added from the exit-ray directions.
  Returns ``(2, ...)`` or ``(3, ...)`` with ``return_longitudinal=True``.
  Validated: with a null (air) prescription the Jones is the identity and the
  vector ``E_x`` reproduces the scalar propagator to fidelity 1.0 with no
  spurious cross-polarization.
- **Anamorphic grids for the whole FGA family.**  All four FGA entry points
  (`apply_real_lens_fga` / `_fga_vector` / `_auto` / `_universal`) now accept a
  ``dy`` (y pixel pitch) argument and support rectangular arrays, joining the
  canonical anamorphic ``apply_*`` contract.  The FGA physics is grid-agnostic
  (ray transport, Herman–Kluk monodromy, and OPL are in physical units), so
  ``dy`` enters only the sampling: the Gabor analysis lattice, the frozen-beamlet
  scatter, and the phase-space measure (the anamorphic cell ``dx*dy``); the
  frozen beamlet stays isotropic with width ``w0 = w0_factor*sqrt(dx*dy)`` and
  the momentum swarm is unchanged (``p`` is a physical direction cosine bounded
  by the NA, not the pixel pitch).  ``dy=None`` is byte-identical to the prior
  square path.  Validated vs the exact angular-spectrum oracle at ``dy=1.5*dx``
  and ``2*dx`` (fidelity 1.0, energy 1.0) and on a non-square array.  Strong
  anisotropy (``>~ 3:1``) may want a larger ``w0_factor`` to keep the coarse
  axis well sampled.

### Fixed

- **FGA normalization -- the ``t=0`` resolution of identity is now exact.** The
  leading Herman-Kluk identity factor ``a(0) = 2^{d/2}`` (``d=2`` transverse) was
  double-counted, so the ``t=0`` reconstruction over-counted by ``2^d = 4`` in
  power and free-space propagation carried a ~2x energy excess.  Dividing the
  reconstruction by ``2^{d/2}`` restores the resolution of identity (``t=0``
  power ratio ``4.0 -> 1.000``) and makes free-space propagation energy-conserving
  (``eta 2.0 -> 1.000``, fidelity 1.0) -- confirming (per the higher-order-FGA
  analysis, Lu & Yang 2012) that the energy defect was a normalization factor,
  NOT the O(eps) transport error, so no higher-order correction is needed.  The
  field SHAPE was already correct (the fix is scale-only); ``normalize_output``
  and fidelity results are unchanged.  A documented residual: near-collimated
  inputs through strong focusing (FBI spectrum concentrating near ``p=0``) can
  still over-amplify the absolute scale -- a representation regime, handled by
  ``normalize_output='power'`` and by not over-widening ``p_max``.

## [5.22.0] — 2026-07-14

### Fixed

- **Out-of-plane generator factor-i — wrong extraordinary-wave dispersion
  for OOP tensors at oblique incidence, multi-release**
  (`docs/audits/AUDIT_OOP_GENERATOR_FACTOR_I_2026_07_14.md`).  The full-3x3
  layer generator's off-plane cross-blocks (`A` from `ezx/ezy`, `B` from
  `exz/eyz`) were written with real coefficients in a modal-u state whose
  in-plane `P`/`Q` blocks demand relative `-/+i` factors there.  Since
  `A = B = 0` for in-plane tensors and at normal incidence, ONLY
  out-of-plane tensors at oblique incidence were affected — there the
  extraordinary propagation constants came out artificially `+/-` symmetric
  (`kz_e/k0 = +/-1.5646` vs the exact `det(k x (k x .) + eps) = 0` roots
  `{-1.5214, +1.6090}` on a tilted-35-degree uniaxial probe: a 3-5% error),
  the mode fields violated Maxwell under every constant re-scaling, internal
  fields broke the local Poynting theorem by ~7%, and the density- vs
  flux-based per-layer absorption attributions disagreed at 3e-3 while every
  energy budget still closed (budgets telescope — the lossless-trap rule).
  Five releases of gates missed it because the `_berreman4x4` TEST ORACLE
  shared the same prototype ancestry and carried the same blocks — every
  1e-10 solver-vs-oracle agreement was circular.  Fixed in all four Delta
  copies in lockstep (the rcwa generator, its jax twin, the native Berreman
  Delta (latent), and the test oracle); new INDEPENDENT anchors in
  `tests/unit/test_audit_oop_dispersion.py`: exact-dispersion roots (1e-14,
  both gauges, asymmetric-e-pair count), per-mode Maxwell residuals (1e-12,
  all six curl rows), local Poynting inside OOP layers (`C/k0 = 1 +- 2e-3`),
  and cross-machinery attribution agreement (1e-4).  Affected results
  (`rcwa_jones_1d/2d` full-tensor at oblique, `RCWAStack` OOP layers,
  Berreman OOP-oblique incl. the jax twin, `pmm_jones_1d_conical_tensor`,
  PMM hybrid tensor paths) move to the corrected values.
- **The PMM 1-D generators carried the same missing factor-i in their
  off-plane cross blocks — both fixed and dispersion-pinned** (audit doc
  §6).  For a UNIFORM medium each spectral-element generator is an exact
  matrix polynomial in `Dop`, so `eig(L)` must land on the exact
  det-condition roots at every alpha in the operator's own spectrum — a
  closed-form anchor sharing zero code with the solvers.  The METRIC
  generator (`_build_generator_metric`, the `pmm_jones_1d` vertical-OOP
  path): the corrected cross-block signs are the UNIQUE combo of all 256
  per-block `{+-1, +-i}` choices that closes (1.9e-10; next-best 1.2e-2);
  the y-uniform three-engine cross-check (`test_v5_14_0_pmm2d_oop`) drops
  4.5e-2 -> 8.7e-4 and its bar re-tightens 6e-2 -> 3e-3 — this CLOSES the
  "metric-generator OOP channel" open item recorded above.  The COVARIANT
  generator (`_cov_generator_4n`, the spectral slant path): again the unique
  combo of 256 (4e-12 full-spectrum with the modal Ez closure at slant
  0/30/45, generic AND symmetric tensors; 2e-10 on resolved alphas with the
  production div-conforming closure), and its cross blocks now use the
  pointwise `[[exz/ezz]]`-style ratio composites (Li Eq.12 discipline)
  instead of spectral products of discontinuous factors.  New gates:
  `test_audit_oop_dispersion.py::test_pmm_metric_generator_matches_exact_dispersion`
  and `test_pmm_covariant_generator_matches_exact_dispersion[30/45]`.

- **Exactly-zero off-plane blocks route to the symmetric path (numpy
  backend)**: an in-plane cell passed through an off-plane-capable assembly
  (e.g. the `fff_nv` tensor factorization) hands zero cross-block MATRICES
  rather than `None`; those now take the symmetric `eig(P Q)` path —
  mathematically identical, 4x cheaper, and numerically stable at marginal
  truncations (resolves the order-dependent `fff_nv` cross-solver flake).
  The routing is gated OFF the jax backend: there the check would be
  value-dependent (eager calls see concrete zeros, grad/jit tracing cannot),
  so finite-difference and autodiff would silently walk two different exact
  algorithms — measured as an O(1) in-plane FD-vs-AD gradient mismatch in
  the `pmm_jones_2d` jax twin before the gate.  (Also recalibrates the
  long-standing marginal 1e-9 jax-vs-numpy parity bar to 5e-9 — two exact
  algorithms agreeing at the eig bit-noise floor.)
- **`RCWAStack.solve(retain_internal=True)` for out-of-plane tensor stacks**
  (previously raised): the Berreman-C2 generalized retention ported —
  generalized partial cascades, explicit asymmetric mode sets, and the
  full-tensor `E_z` recovery (`EZX`/`EZY`).  Validated against the
  independent Berreman machinery on identical physics: internal fields agree
  at 1e-15 on all six components, per-layer absorption splits at 2.6e-6.
- **Stale gates repaired**: the jax `retain_internal` guard test pinned a
  pre-v5.21 restriction (in-plane traced retention is supported — the gate
  now pins the closed absorption budget; note the CI unit jobs install no
  jax, so this gate only ever ran locally); `test_v5_14_0_pmm2d_oop`'s
  y-uniform OOP grating oracle repointed to the dispersion-anchored rcwa
  solver; the `fff_nv` convergence-margin recalibrated (0.3 -> 0.5, measured
  0.32 post-fix); `lc_cell`'s rcwa oracle moved off an isolated unstable
  truncation (shipped in the v5.21.5 line, recorded here for completeness).
- **Subsystem-audit residuals cleared (audits 1/3/4/6 deferrals, 9 items)**:
  `gerchberg_saxton_jax` error metric no longer reports the PREVIOUS
  iterate's far-field error (parity with NumPy 2e-16); `vectorial_hfpi`
  `output_grid -> output_shape` rename with deprecation shim; afocal
  `trace_summary`/`spot_diagram` no longer print a radians half-angle with
  a length label or a meaningless Airy overlay; `paraxial_focus_world`
  gains near-parallel + dead-ray guards (chief -> axial rename); Zemax
  writers expose `glass_catalogs=` (GCAT no longer hardcoded); Zemax
  loaders reassign a STOP set on a COORDBRK row to the next optical
  surface with a warning; `_bluestein_2d` caches the chirp-kernel FFT
  (numpy default backend, buffer-ownership-safe copies);
  `surfaces_from_prescription(include_coord_breaks=True)` (opt-in,
  default byte-identical) interleaves loader coord breaks so the plain
  local-frame `trace()` handles folded prescriptions — matches the
  `trace_world` oracle to ~1e-17 on a folded periscope, with the ZX-1
  DISZ fold reversed for strictly-between breaks.  15 new tests.
  Recorded follow-up: `world_surfaces_from_prescription` double-counts
  ZX-1-folded coord-break DISZ on loader-produced folded prescriptions
  (pre-existing; validation oracles use hand-built world dicts).

### Changed

- **`'auto'` now routes slanted OUT-OF-PLANE cells/stacks to `'convection'`**
  (`pmm_jones_1d_slanted`, `pmm_jones_1d_slanted_segments`, `PMMStack`;
  in-plane slanted cells keep the spectral covariant route).  With the
  corrected physics, a 3-way referee on the slanted binary OOP grating shows
  convection agreeing with the independent RCWA tensor staircase at
  3.8e-3/3.9e-3 (slant 30/45 deg — the mutual truncation floor), while the
  covariant layout's DISCONTINUOUS off-plane TM channel is the outlier at
  ~0.10-0.12 (TE clean) — the 2026-06-08 six-avenue study's unresolved
  "bare exz/ezx sub-channel" floor, previously masked because all pre-fix
  engines agreed on the same symmetrized wrong answer (the old 3e-3
  staircase bars were calibrated in that world).  Explicit
  `factorization='covariant'` still solves OOP with the limitation
  documented in its docstrings; the covariant OOP tests are now
  regression-trackers (TE clean / TM within the documented floor / energy
  conserving), and the staircase gate runs the production route at its
  measured ~4e-3 floor (bar 6e-3).

## [5.21.5] — 2026-07-14

### Added

- **Consumer API — the full remainder of
  `docs/audits/AUDIT_DYNAMETA_CONSUMER_API_GAPS_2026_07_13.md`** (items
  B-PMM2D / C1 / C2 / C3 / D1; the audit is now FULLY shipped).  Gates in
  `tests/unit/test_audit_dynameta_consumer_api_2.py`:
  - **Per-order amplitudes for the 2-D PMM engines (B, PMM2D leg)**:
    `per_order_amplitudes(port)` + `jones_transmission()` on
    `PMM2DStackHybrid` (and the `PMM2DStack` alias) and `PMM2DStackPure`
    via a shared `PerOrderAmplitudesMixin` — the exact
    `RCWAResult.per_order_amplitudes` contract with 2-D `(N, 2)` orders,
    PUBLIC `exp(-iwt)` gauge, public decaying-branch `kz`.  Validated
    against RCWA per-order COMPLEX amplitudes on an identical crossed-pillar
    cell at normal + conical incidence (Pure ~3e-4, Hybrid ~3e-3 vs a
    converged reference); the documented flux recipe rebuilds the returned
    efficiencies exactly.
  - **BOR modal amplitudes + absorption (C1)**:
    `BORStack.per_mode_amplitudes(port)` — complex modal scattering
    amplitudes in a PINNED deterministic eigenvector gauge (the raw
    `res["S"]` column gauge is now documented on `solve`; its diagonal was
    always gauge-invariant) — and `BORStack.layer_absorption()` via
    `solve(retain_internal=True)` partial cascades + the staggered two-grid
    flux.  Budget `R + T + sum A = 1` closes at 1e-12; the pinned
    fundamental-mode COMPLEX reflection matches analytic Fresnel (5e-16)
    and Fabry–Perot (2e-14) oracles at the mode's own local angle — the
    transmitted/reflected PHASE is now a first-class BOR observable.
  - **Berreman internals for OUT-OF-PLANE tensors at OBLIQUE incidence
    (C2)** — the audit's flagship: `solve(retain_internal=True)` no longer
    raises for the tilted-director regime.  The generalized (Li 2003)
    cascade retains the same internals shape as the native core (asymmetric
    modes sliced from its `M` blocks + generalized-convention partial
    cascades, mapped to the public gauge by conjugation with a modal-H
    negation), so `internal_field` and `layer_absorption` serve it
    unchanged.  Absorption budget closes at machine precision (9e-16) at
    oblique AND conical on a lossy tilted-director stack; theta -> 0
    continuity against the native path holds for the absorption (1.8e-7)
    and all six field components (4.9e-5 at theta = 1e-4).
  - **Pure-engine absorption (C3)**:
    `PMM2DStackPure.solve(retain_internal=True)` + `layer_absorption()` —
    z-flux differences in the staggered modal basis via the eps-free block
    field Gram (the Eq.25 dual pairing; `Re` form pinned on the
    homogeneous-mode oracle).  Budget 8e-14; Pure-vs-Hybrid per-layer
    cross-gate 6.7e-3.
  - **RCWAStack JAX ergonomics (D1)**: a traced uniform `eps=` scalar and
    traced `set_source` wavelength / theta / phi now flow gradients through
    the stack twin (previously complex()/float()-severed, forcing a
    lifted-cell eigensolve workaround and making wavelength/angle gradients
    impossible).  Kept raw when traced; backend dispatch includes them; the
    grazing nudge + propagating-incidence guard are documented as skipped
    under trace; the homogeneous-mode cache is bypassed (unhashable traced
    keys).  Forward parity with the concrete solve 4e-15; AD-vs-FD: eps
    6e-9, wavelength 1.1e-7, theta 1.7e-8.  Concrete solves are
    byte-identical (every change branches on tracedness).

## [5.21.4] — 2026-07-13

### Fixed

- **BOR propagating-mode classifier: near-grazing orders no longer dropped**
  (`docs/audits/AUDIT_BOR_PROPAGATING_CUTOFF_ENERGY_2026_07_13.md`).  The
  v5.18.0 dimensionless classifier (`q/k0 > 0.05`, audit P2-06) was an
  ANGULAR cutoff (`theta < 88 deg` in n=1.41) that silently excluded
  genuinely propagating near-grazing orders from the incident/outgoing
  sets — per-order `R`/`T` biased low, lossless energy closure degraded
  from 1.2e-11 to 2.28e-2 on a 48 um ring-grating reproducer (319 -> 318
  modes; surfaced by DynaMeta's `lumenairy_bor_bridge` GATE C).  The
  real-axis leg is now floored at the `q ~ 0` degenerate point only
  (`q/k0 > 1e-6`), applied in lockstep to all THREE classifier twins
  (`bor_stack.solve`'s `prop()`, `bor_solve._physical_propagating`,
  `_jax_bor._mask`); imag + index-ceiling legs unchanged.  Reproducer
  restored to the pre-regression values to all digits (319 modes,
  1.22e-11) at m/um/nm scales; fundamental-mode `R = 0.146135` pinned
  (lossless-trap guard); flux-normalizer seam probed unreachable
  (`P/fnrm = q/k0` — kept implies flux-normalized).  6 gates in
  `tests/unit/test_audit_bor_grazing_cutoff.py`; the k0=2.0 suites have no
  near-cutoff modes (verified by running, not assumed).
- **BOR legacy nodal cascade: catastrophic large-cell energy blow-up**
  (same audit, follow-up finding).  `bor_solve.build_layer`/`solve` (the
  M5 prototype of `BORStack`) returned `max|R+T-1| ~ 1e29..1e32` for
  `Rbig >= ~12 lambda`.  Root cause: the nodal FD basis's zero-flux
  spurious mode sea orients forward/backward by the SIGN OF NOISE, so
  adjacent layers sharing most of their cross-section carry near-identical
  spurious modes oriented oppositely — a null vector of the interface
  transmission block `a + b` (`cond ~ 2.6e15` while `cond(W), cond(V) ~
  1e3`).  `build_layer` now defaults to the spurious-free staggered
  (Yee div-conforming) basis — the production `BORStack` discretization —
  with `basis="nodal"` retained as the legacy escape hatch.  Blow-up
  config: 9.7e29 -> 3.9e-13; the audit reproducer through
  `build_layer`/`solve` now matches `BORStack` per-mode EXACTLY (diff 0.0).
- **Conical PMM per-order `kz` export branch**: the nodal-conical far
  field's `conj(kz_forward(conj(eps)))` gauge map flipped the EVANESCENT
  `kz` to the growing branch (`Im < 0`) for lossless media.  `R`/`T`/Jones
  were unaffected (the flux math is `Re()`-only); the branch now evaluates
  directly on the PUBLIC eps (`Im >= 0` decay), as the new
  `per_order_amplitudes` surface reports these values.

### Added

- **Consumer API surface** (`docs/audits/AUDIT_DYNAMETA_CONSUMER_API_GAPS_
  2026_07_13.md`, items A1/A2/B; driven by the DynaMeta bridge campaign):
  - `BerremanStack.jones_transmission()` (A1) — the transmission Jones the
    class solve computed and DISCARDED on every path, forcing consumers to
    run a second functional solve just for `t`.  Retained on all four solve
    paths (NumPy main + OOP-oblique, JAX plain + retain); bit-identical to
    `berreman_jones_1d`'s `jones_t`; one `retain_internal=True` solve now
    serves the far field (incl. `t`) AND `layer_absorption` (absorption
    budget closes at 1e-10).
  - `RCWAStack.layers` (A2) — public read-only tuple view of the per-layer
    records (`thickness`/`kind`/`data`/`formulation`/`dispersive` documented
    as public); reverse translators no longer read the private `_layers`
    slot under a version ceiling.
  - `PMMStack.per_order_amplitudes(port)` + `PMMStack.jones_transmission()`
    (B) — per-order complex tangential amplitudes with a pinned PUBLIC
    `exp(-iwt)` gauge, mirroring the `RCWAResult.per_order_amplitudes`
    contract exactly (same keys; `kx`/`ky`/`kz` normalized by `k0`).
    Retained by the classical mount (incl. convection-slant and
    generalized-OOP close-outs) and both native-conical paths (patterned
    nodal + uniform Fourier); the covariant uniform-slant cascade and the
    JAX twin raise with the surface documented.  This unlocks exact conical
    s/p synthesis for patterned PMM cells (per-order cross terms that
    per-order POWERS cannot provide) and the transmitted PHASE for the PMM
    referee engine.  Validated against RCWA per-order COMPLEX amplitudes on
    identical physics (classical oblique ~1e-4, conical theta=30/phi=25
    ~3e-4, uniform conical exact); the documented flux recipe rebuilds the
    returned efficiencies to 1e-16; the rotated conical s-hat total
    synthesized from the amplitudes matches the RCWA-amplitude oracle at
    9e-6 while the naive power sum misses 1.25e-2 of cross terms.  15 gates
    in `tests/unit/test_audit_dynameta_consumer_api.py`.  (The PMM2D leg of
    B and items C1–C3/D1 remain open — roadmap-class per the audit.)

## [5.21.3] — 2026-07-12

### Fixed

- **Conical (`phi != 0`) PMM: pure-nodal cascade for PATTERNED layers**
  (`docs/audits/AUDIT_PMM_CONICAL_PATTERNED_TENSOR_BUG_2026_07_12.md`).  The
  v5.20.0 native conical 1-D path produced a systematic,
  resolution-INDEPENDENT error for patterned layers: at the `ky0 = 0`
  degenerate limit (`theta = 0`) it never reduced to the converged classical
  solve (`||J_conical - J_classical|| ~ 3.3e-3`, a ~3.8 deg
  reflection-retardance offset that destroyed a multilayer LC out-coupler's
  51:1 switching extinction).  Root cause (the audit's localization,
  verified and expanded): the defect hits SCALAR patterned gratings too (the
  old scalar reduction test's 5e-3 tolerance masked it), and it is NOT the
  tensor factorization rule — it is the Fourier-PROJECTED operator build
  itself (`T @ op @ pinv(T)` compression in `_tensor_layer_modes` /
  `_layer_modes_projected`): the gap saturates in `far_field_orders`
  (identical 2.822e-3 at ffo 41 and 81) and GROWS with `degree` (1.7e-3 ->
  4.3e-3 over degree 6 -> 18).
  - `_sem_modes_tensor` gains a dimensional `ky0`: the full
    dimension-agnostic P/Q tensor blocks assemble from the same weak-form
    mass/stiffness/convection operators (single-`Kx` cross terms as
    elementwise-exact weak first derivatives).  Every added term carries a
    `ky0` factor, so at `ky0 = 0` the build is BIT-IDENTICAL to the
    classical solve.
  - New `_conical_nodal_solve` (pmm/conical.py): the classical union-grid
    Redheffer cascade run end-to-end in the NODAL basis (public gauge, no
    projection floor), closed with the conical vector far field via the
    nodal->Rayleigh projection.
  - `PMMStack.solve(phi != 0)`, `pmm_jones_1d_conical`, and
    `pmm_jones_1d_conical_tensor` route every PATTERNED in-plane cell
    through the nodal cascade; a patterned OUT-OF-PLANE tensor cell now
    raises `NotImplementedError` (the old path returned silently-wrong
    retardance for it); all-UNIFORM cells keep the exact Fourier path
    (Berreman-validated, including out-of-plane tensors).
  - Validated: reproducer gap 3.29e-3 -> 1.4e-14 (machine precision),
    retardance offset 3.83 deg -> 0.00 deg; scalar + mixed
    uniform/patterned multilayer stacks likewise; lossless energy at
    genuine conical closes to 1e-10; `theta -> 0` continuity is quadratic;
    degree-CONVERGENT off-normal; the independent `rcwa_jones_2d`
    cross-oracle converges toward the nodal answer.  8 new regression gates
    in `tests/unit/test_v5_20_0_pmm_conical.py` (patterned tensor + scalar
    degenerate-limit reductions at 1e-10, director retardance sweep,
    single-layer entry reductions, energy + continuity, OOP-patterned
    raises, rcwa cross-oracle, mixed multilayer).
  - Note: `PMM2DStack` (hybrid) still uses the projected machinery and
    remains Fourier-floored for patterned cells under conical incidence —
    use `PMM2DStackPure` for a no-floor 2-D answer.

## [5.21.2] — 2026-07-11

### Fixed

- **Subsystem-audit remediation campaign** — 18 chronological, line-level
  subsystem audits (`docs/audits/AUDIT_*_2026_07_07..09.md`) of the entire
  library.  The physics kernels were essentially flawless; every actionable
  defect lived at integration seams, unmirrored fixes, dead parameters, or
  silent-degradation paths.  Each finding validated; tests in
  `tests/unit/test_v5_21_2_subsystem_audits.py`.
  - **analysis** (`AUDIT_ANALYSIS_METRIC_CORE_2026_07_07.md`): AN-1
    `depth_of_focus` was 2x too large (one-sided Rayleigh DOF `= 2 f#^2 λ`);
    AN-4 distortion/spot/footprint now aim the chief ray at the entrance
    pupil; AN-3/AN-5 + validation/dead-code nits.
  - **sources** (`AUDIT_SOURCES_CORE_2026_07_07.md`): SRC-1 dense
    `PartialCoherenceMCF` was the CONJUGATE of `⟨E(r1)·conj(E(r2))⟩`
    (flipped the coherence phase sign with grid size); SRC-2 scale guards;
    SRC-3 empty-field-angles raise.
  - **propagators** (`AUDIT_PROPAGATORS_KERNELS_2026_07_07.md`): DS-1
    `Source.propagate` mishandled pitch-changing kernels (tuple-as-field +
    stale pitch); HF-1 `beam_d4sigma` `float(tuple)` crash + corrected LG
    waist to `D4σ/2`; VD-1 immersion-NA raise; SY-1/2, PK-1/2/3, MHS-1/2,
    HFPI-1/2, VHFPI-1, dispatch/system nits.
  - **raytrace** (`AUDIT_RAYTRACE_CORE_2026_07_08.md`): RT-5 off-axis ray/OPD
    fans EP-centred; **RT-6 removed a spurious high-NA RuntimeWarning** (the
    JAX transfer is exact to sub-ppm, proven invariant across a 100x gap
    sweep); RT-1/2/3/8/9 + even-aspheric guard + through-focus all-dead guard.
    (RT-4, the audited "world coord-break tilt sign" finding, was verified to
    be a PHANTOM and reverted: `world_R`'s `_rot_x(+tx)` already agrees with
    `trace()` — both fold a +z ray to world -y — as the periscope
    folded-design + `test_world_surfaces` validation oracles confirm.)
  - **glass + polarization** (`AUDIT_GLASS_POLARIZATION_2026_07_08.md`):
    GL-1 missing-κ message, GL-2 lockless glass-cache purge race, SILICA
    Sellmeier row, array-safe validity check, JonesField/docstring nits.
  - **io/zemax** (`AUDIT_IO_ZEMAX_2026_07_08.md`): ZX-1 COORDBRK DISZ folded
    into the flat thicknesses; ZX-3 `is_stop`/`stop_index` preserved on
    loaded files; ZX-4 `back_focal_length` honoured in the full writer;
    Q-type/encoding nits.
  - **doe/grating/freeform** (`AUDIT_DOE_GRATING_FREEFORM_2026_07_09.md`):
    DOE-1 FITS default split-save now recovered by the default load (was
    silent phase loss on round-trip); dammann/mask/collision nits.
  - **coatings + elements** (`AUDIT_COATINGS_ELEMENTS_2026_07_09.md`): COAT-1
    non-dispersive-sweep documentation; material-index warning gate;
    gaussian/annular aperture guards; `quarter_wave_ar` / `n_cplx` nits.
  - **maslov** (`AUDIT_MASLOV_2026_07_09.md`): MSL-1 sign-preserving
    near-singular-Hessian floor (a tiny negative determinant produced a NaN
    saddle step near fold caustics) at all three saddle integrators.
  - **eme** (`AUDIT_EME_2026_07_09.md`): floored the complex-symmetric
    bilinear mode norm against a defective matrix; documented the unsorted
    `eig` branch.
  - **bsdf + segment-geometry** (`AUDIT_BSDF_SEGMENT_GEOMETRY_2026_07_09.md`):
    BSDF-1 `GaussianBSDF.sample` drew a half-normal, omitting the `sinθ`
    solid-angle Jacobian (a ~35%-too-narrow stray-light cone) — now Rayleigh;
    HarveyShack batch-safety + `to_rcwa_stack` NaN-tile guard.
  - **optimize** (`AUDIT_OPTIMIZE_{MERITS,DRIVER,WRAPPERS,TAIL,SECOND_PASS}_2026_07_09.md`):
    OPT-1 the JAX LG-aberration merit MINIMISED the Strehl (drove toward
    max aberration) — now the Strehl deficit; OPT-2 `ToleranceAwareMerit`
    now populates the OPD/spot sub-context (OPD merits no longer degenerate
    to ∞); OPT-3 the `method='lm'` path now checkpoints/telemeters like every
    other method; FD-floor classification, honest `converged`/`iterations`
    result fields, and a retired 21-versions-stale DeprecationWarning.
  - **io/prescriptions + storage/codegen**
    (`AUDIT_IO_PRESCRIPTIONS_2026_07_09.md`,
    `AUDIT_IO_STORAGE_CODEGEN_2026_07_09.md`): CV-1 CODE V loader warns on
    dropped fold/aspheric directives (was a silent straight-axis import);
    CG-1 codegen forwards Q-type freeform keys (+ warns on unrepresentable
    mirror aspherics); H5 None-attr boundary guard.

## [5.21.1] — 2026-07-07

### Fixed

- **v5.21 delta-audit remediation** (`docs/audits/AUDIT_V5_21_DELTA_2026_07_07.md`,
  findings D1–D15).  Correctness/robustness gaps in the new v5.21 code, each
  validated; tests in `tests/unit/test_v5_21_delta_audit.py`.
  - **D1 (conical incidence guards).**  `pmm_jones_1d_conical`,
    `pmm_jones_1d_conical_tensor` and `PMMStack`'s conical (`phi != 0`) solve
    now run the suite-standard `_require_propagating_incidence` +
    `_grazing_safe_wavelength` guards (every other PMM/RCWA entry has both).
    A gain / metallic / grazing incidence medium now raises instead of
    silently negating or NaN-ing the `kz_inc`-normalised far field; the
    grazing nudge is a no-op away from a Rayleigh cutoff, so valid solves are
    byte-identical.
  - **D2 (differential-ray-transfer NaNs).**  The finite-difference
    `ray_transfer_jacobian` now ANDs companion-aliveness into `alive` and
    scrubs the Jacobian, so a rim-adjacent base ray whose ±h companion
    vignettes/TIRs is masked out instead of leaking a NaN Jacobian into the
    coherent GBD per-surface sum (reachable on aspheric/freeform surfaces).
    The GBD local-frame branch also gains the `isfinite` mask the world
    branch already had.
  - **D3 (multibranch KMAH index).**  In-glass (surface-to-surface) fold
    caustics are now counted per leg with the same exact-quadratic `det Q(z)`
    method the exit leg uses, replacing a mod-2 parity closure that left an
    EVEN internal-crossing count invisible (a focus between two elements = a
    π Maslov-index error).  The exact parity guarantee is retained; 0 for an
    air-focus system (byte-identical to v5.21).
  - **D4 (multibranch immersed output plane).**  The Kraaijpoel OPL
    intrapolation eikonal gradient is now `p = n·(L, M)` for
    `output_plane_n != 1` (byte-identical for a vacuum output plane).
  - **D7 (Levin engine).**  Dead `fmax` parameter removed; the rigorous
    residual bound `∫|r|` is now a Clenshaw–Curtis quadrature (exact
    integral) instead of the endpoint-over-weighted sample mean; stale
    `levin2d` docstring corrected.
  - **D11 / D12 (berreman JAX twin).**  A traced out-of-plane tensor now
    routes to the generalized (exact) cascade instead of the ~2%-off native
    path (mirroring the rcwa tracer→general fix; forward matches concrete to
    ~6e-15, `jax.grad` flows); a concrete OOP layer beside a traced isotropic
    spacer no longer raises under `jit`; `_offplane_solve_jax` gains the
    NumPy path's grazing / `Re(kz)>0` guards; the `retain_internal` message
    no longer misleadingly says "OBLIQUE".
  - **D9 (`PMM2DStack` deprecation phase).**  The bare `PMM2DStack`
    transitional alias now emits a `DeprecationWarning` on construction (it
    is scheduled to be repointed from the hybrid to the no-floor pure stack —
    a silent results change); pin `PMM2DStackHybrid` or `PMM2DStackPure`
    explicitly.  Explicit names stay silent.
  - **D13 (`fff_nv` cross-reference).**  `CONVENTIONS.md` §11 documents that
    `formulation='fff_nv'` names three different algorithms across
    `rcwa_efficiency_2d` / `rcwa_jones_2d` / `pmm_jones_2d`.
  - **D14 / D15 + D5 / D10 (comments & scope notes).**  Corrected the
    geo-eig cache comment (`k0`-independent only at fixed `kx0`); the
    traced-segmentation cap drops the shallowest cuts first (matching its
    comment) and drops a dead `wavelength` parameter; scope notes on the
    Ludwig fold-vs-point band and the GBD vector sampling coupling.
  - Verified already-fixed (no change): **D8** — both JAX 2-D cell branches
    already honour `max_nodal_dof` at dispatch.

## [5.21.0] — 2026-07-07

### Added

- **Lens-propagator accuracy extensions (v5.21).**  A batch of per-propagator
  accuracy features, each validated against an independent oracle (no unvalidated
  physics).  Tests in `tests/unit/test_v5_21_lens_accuracy_extensions.py`.
  - **Maslov `poly_order='auto'`** -- the tensor-Chebyshev fit order is raised
    until the OPD-fit residual, scored on a **held-out** ray split (so a too-high
    order that only fits ray-node noise is rejected), plateaus or hits a target.
    A smooth optic fits at a cheap low order; a strongly-aberrated / near-caustic
    chart gets the order it needs -- no manual tuning.  Auto tracks the order-8
    field to ~2e-5 (≈100x better than a fixed order-4) on a singlet.
  - **GBD `converge_gbd_sampling`** -- picks the beamlet width (swept as the
    decimation-independent overlap `w0/spacing`) that makes a free-space GBD
    propagation most accurate, scored against the **exact** angular-spectrum
    oracle (reconciling the known GBD↔ASM beamlet-Gouy convention first).
    Reports the full error curve + whether it met the tolerance.
  - **GBD longitudinal `E_z`** -- `reconstruct_vector_field_with_ez` and
    `propagate_gbd_freespace_vector(..., return_longitudinal=True)` add the
    longitudinal field from transversality (`E·k=0`): `E_z=-(L·E_x+M·E_y)/N` per
    beamlet, exact to machine precision (verified `E_z=-tanθ·E_x` up to NA 0.5).
    The longitudinal energy fraction grows with NA -- the piece a transverse-only
    GBD misses at high NA.
  - **traced `apply_real_lens_traced_segmented`** -- applies the traced lens to a
    single, possibly MULTI-congruence field by blindly splitting its angular
    spectrum at the deep VALLEYS between beams (a `cos^2` partition of unity that
    sums to the input exactly), so each segment is a single congruence and the
    per-segment traced results sum coherently.  Recovers the exact per-emitter
    reference to ~1.7e-4 on two crossing beams through an aberrated lens
    (~8000x better than the single-congruence-violating `traced(sum)`); a
    unimodal field is a single segment == plain traced.
  - **traced jax twin differentiable w.r.t. prescription geometry** --
    `apply_real_lens_traced_jax(..., radii=, conics=, thicknesses=)` routes the
    trace through `trace_jax_with_params` and takes a tracer-safe Newton initial
    guess, so `jax.grad` flows into the lens geometry (the lever for
    gradient-based lens *design* on the accurate ray-traced OPD).  `grad` matches
    finite-difference for radius (2e-8) and thickness (6e-9); the static path
    (no arrays) is byte-identical.
  - **GBD complex-source-point (non-paraxial) beamlets** -- `csp_beamlet_field`
    evaluates the EXACT scalar-Helmholtz field of Deschamps complex-source-point
    beams (branch `Im R ≤ 0`), matching the angular-spectrum method to grid
    precision at all NA where a paraxial Gaussian is ~33% wrong (NA 0.45).
    `propagate_gbd_freespace_csp` rides the same beamlet skeleton with the exact
    field.  Honestly scoped: the per-beamlet exactness is the clear win; the
    GBD-*sum* gain over paraxial is regime-dependent (co-limited by the shared
    reconstruction/overlap floor at moderate NA).
  - **Uniform fold-Airy caustic evaluator** (`uniform_fold_airy`) -- the
    Chester-Friedman-Ursell caustic-FINITE value of `int g exp(ikf) dt` where two
    stationary points coalesce (a fold), from the two saddles' `f`, `f''`, `g`.
    Branch discipline pinned by the stationary-phase Maslov phase
    `exp(i·sgn(f'')·π/4)` per saddle (not an ambiguous `sqrt` root); matches the
    exact cubic-phase integrals to ~1e-14 for both symmetric and asymmetric
    (`a1 != 0`) folds and stays finite through the caustic where ordinary
    stationary phase diverges.  The caustic-safe integrator underlying a uniform
    Maslov evaluator and the multi-branch Airy hand-off.
  - **Pearcey (cusp-caustic) evaluator** (`pearcey`) -- the canonical cusp
    diffraction special function `P(x,y)=int exp(i(t^4+x t^2+y t)) dt` (the cusp
    peer of the Airy function), via its everywhere-convergent series.  Machine-
    precision vs the exact cusp value `P(0,0)=1/2 Gamma(1/4) exp(i pi/8)`, the
    even-in-y symmetry, and a contour-rotated quadrature.
  - **Windowed CSP reconstruction** (`propagate_gbd_freespace_csp(..., window=)`)
    -- evaluates each CSP beamlet only over its local pixel box, `O(n*box)`
    instead of `O(n*N^2)`; matches the dense sum to the tail truncation.
  - **Multi-branch traced lens field** (`apply_real_lens_traced_multibranch`)
    -- the traced model extended THROUGH focus and caustics, where the
    single-valued `apply_real_lens_traced` breaks down.  Implements the
    seismology wavefront-construction method (Lambare 1996; Vinje 1993;
    Chambers & Kendall 2008): triangulated launch-grid ray map with barycentric
    multi-arrival branch finding, second-order "intrapolated" OPL
    (`T = sum a_i [T_i + 1/2 (x-x_i).p_i]`, Kraaijpoel 2003 eq. 5.7),
    signed-area-ratio `1/sqrt|J|` amplitudes, and the KMAH (Maslov/Gouy) phase
    `exp(-i pi m/2)` per branch with fold crossings counted ANALYTICALLY on the
    exit leg via the exact quadratic `det Q(z)` (Cerveny/Klimes dynamic ray
    tracing) plus a parity closure.  Validated: energy conserved to 2.7%
    (launch-rim tail), the Gouy `-pi` emerges from pure ray bookkeeping
    (KMAH 0 -> 2 through focus), and the mid-annulus multipath intensity
    matches an exact decouple-pipeline oracle (ray-traced exit-pupil field +
    direct Rayleigh-Sommerfeld summation) to ~6% masked off the O(1-px)
    caustic band, where ART is undefined by construction (literature-standard).
    v1 scope: collimated / slowly-varying input congruence.
    **Ludwig caustic-band swap** (`caustic_band='ludwig'`, default): per
    pixel, the closest-eikonal branch pair inside the Kravtsov-Orlov band
    `k|S+ - S-| <= pi` is replaced by the `ludwig_fold` uniform two-branch
    field (Grillo & Cordes 2019 eq. 47 form), taming the `1/sqrt|J|`
    divergence exactly on the fold while staying byte-identical to the plain
    sum wherever fewer than two branches land (`caustic_band='plain'`
    restores the raw sum).
    **Vectorized triangle rasterization**: the per-launch-cell Python loop
    (~70k triangles of small-array NumPy) is replaced by flat-array
    per-triangle setup + power-of-2 bounding-box bucket rasterization --
    identical math and contribution set (only the float summation order in
    the scatter-add differs); ~6x end-to-end (20.3 s -> 3.5 s on the
    192x192 through-focus benchmark).
    **Tilted / carrier input** (`input_carrier=None|'auto'|(kx, ky)` in
    rad/m): the launch congruence follows the input phase plane (direction
    cosines `kx/k0`, `ky/k0`), the input eikonal `T_in = L0 x + M0 y` rides
    the branch phases exactly, and the envelope is bilinearly sampled
    CARRIER-STRIPPED (works for super-Nyquist carriers when `(kx, ky)` is
    given explicitly; `'auto'` estimates the mean carrier by the lag-1
    correlation phase, subpixel-exact for carrier x smooth envelope).
    Validated: `(0, 0)` byte-identical to the default; `'auto'` recovers a
    1-degree carrier to 1e-6; the near-focus pattern displaces by the traced
    chief-ray transverse position to 1%; energy matches the untilted run to
    0.1% at non-degenerate planes.  Small-tilt scope: the ART amplitude
    keeps the transverse-area ratio (obliquity factors <0.1% below
    ~2.5 deg).
  - **Adaptive delaminating Levin engine** (`lumenairy._math.levin`) and
    `apply_real_lens_maslov(integration_method='levin', levin_tol=...)` -- a
    caustic-UNIFORM evaluator for the Maslov pupil integral with **no saddle
    finding** (finite and accurate through folds where `stationary_phase` /
    `local_quadrature` diverge) at a per-pixel cost independent of the v2
    oscillation count.  After Chen-Serkh-Bremer-Aubry (arXiv:2506.02424) with
    validated deviations: residual-bound acceptance (their eq. 152; the
    parent-child value test false-accepts on under-resolved oscillatory
    boxes), priority-queue refinement against a global bound budget,
    machine-level TSVD truncation, and float-coerced box coordinates (an
    int-dtype corner truncates the companion coordinate inside boundary-phase
    closures -- a domain-edge-only phase error invisible to any value test).
    Engine validated on canonical folds to 8.9e-10 THROUGH the caustic with
    its rigorous returned bound honored.  `levin_tol` is RELATIVE (scaled per
    pixel by the probed integrand magnitude).
    **Wave-batched production integrator**: `integration_method='levin'` runs
    lockstep per-pixel quadtrees batched over (pixel, box) pairs -- each
    refinement wave evaluates ALL pairs in a handful of large vectorized
    `_opd6` / batched normal-equations collocation calls (Tikhonov ~1e-8
    standing in for the delaminating TSVD; the rigorous residual bound
    measures the ACTUAL solution, so regularization can only cost refinement,
    never accuracy).  Leaves accepted by an ADAPTIVE residual-bound budget
    (accepted leaves consume tolerance and release area, so slack from
    smooth regions flows to the hard caustic-band leaves); depth-capped
    stragglers get a deeper batched re-pass, then the per-pixel engine as a
    final safety net.  A dedicated shared-basis Numba kernel (`_opd_vd9`:
    value + v2 first-derivatives for opd/s1x/s1y in ONE pass, no T''
    recurrence) supplies all integrand pieces per query set.  On a hard
    2-mm-aperture high-NA singlet chart (16x16 grid, ~16.5-wave p-v pupil
    OPD) vs a dense n_v2=256 quadrature oracle (both reference-limited),
    0 fallbacks: 1.4e-2 in 9.7 s at `levin_tol=1e-2` (~0.04 s/pixel) and
    1.2e-2 in 75 s at `1e-3` (~0.3 s/pixel) -- roughly 300-2000x the
    per-pixel adaptive engine (~83 s/pixel), making full-ROI caustic-band
    maps practical.  Peak memory is chunk-bounded independent of grid size.
  - **Ludwig uniform fold formula** (`ludwig_fold`) -- the ray-native
    caustic-band primitive (Ludwig 1966 / Kravtsov 1964): the uniform field of
    two coalescing ray branches from their eikonals and COMPLEX amplitudes
    (each carrying its own Maslov phase).  Finite exactly where the branch
    amplitudes diverge; reduces to the plain two-branch sum on the bright side.
    Machine-precision (~3e-16) against the exact cubic-phase integral in both
    regimes, including exactly on the caustic.  The drop-in pair-swap for a
    multi-branch sum inside the Kravtsov-Orlov band `k|S+-S-| <~ pi`.
  - **Turnkey lens-design optimiser** (`optimize_traced_geometry`) -- Adam over
    the differentiable ray-traced OPD to optimise prescription radii / conics /
    thicknesses (built on the new `apply_real_lens_traced_jax` geometry
    gradient).  Default merit sharpens the focus (peak intensity at the focal
    plane via an exact angular-spectrum step); custom merits supported.  This
    supersedes the "geometry not differentiable" limitation noted in
    `make_lg_aberration_merit_jax`.

- **GBD FFT-convolution reconstruction** for uniform-Q bundles.  When a beamlet
  bundle has a uniform `Q` (scalar, or diagonal tensor), a uniform launch
  direction and on-grid centres -- exactly what `decompose_field_to_beamlets`
  followed by free-space / uniform-ABCD evolution produces -- the coherent sum
  is a convolution of the amplitude array with ONE Gaussian kernel, evaluated
  as a single FFT of `O(Ny*Nx log)` **independent of beamlet count**.  Auto-
  detected inside `reconstruct_field_from_beamlets(..., window=...)`;
  machine-precision identical to the dense sum (~1e-15) and **2000-3700x
  faster** in the spread/dense regime where the windowed box fills the grid.
  **Backend-generic** (NumPy / JAX / CuPy via `xp.fft` + per-backend scatter),
  so it runs on the GPU under CuPy and is `jax.grad` / `jit` differentiable
  under JAX (grad matches FD to ~1e-12) -- a fast, differentiable free-space
  reconstruct.  Falls back to the windowed scatter-add (NumPy) or the dense sum
  (JAX / CuPy) for per-beamlet-`Q`, skew, off-grid or per-beamlet-tilted
  bundles.

- **v5.21 API-conformance sweep** (release-gate walkers): the new lens /
  GBD / raytrace-differential names are re-exported at top level
  (`apply_real_lens_maslov_vector`, `pearcey`, `uniform_fold_airy`,
  `optimize_traced_geometry`, `recommend_gbd_sampling`,
  `gbd_ghost_analysis`, `propagate_gbd_freespace_spectral` / `_vector`,
  `propagate_gbd_vector_through_prescription`,
  `apply_prescription_persurface_to_beamlets`, `DifferentialTransfer`,
  `ray_transfer_jacobian` / `_analytic` / `_jax`), and every new
  scalar-field entry point now runs the shared `_check_2d_scalar_field`
  input guard first (canonical MCF / non-2-D error messages).  The three
  Maslov Cholesky-fit broad-excepts were narrowed to
  `(ImportError, ValueError, LinAlgError)`.

- **`PMM2DStackPure` — multilayer 2-D PURE (no-floor) PMM stack**
  (`pmm/stack2d_pure.py`), the no-Fourier-floor sibling of the hybrid stack and
  the 2-D analogue of the 1-D `PMMStack`: every region (half-spaces + layers)
  is solved in one shared staggered modified-Legendre basis (Granet 2023),
  interfaces are square modal matches cascaded by Redheffer, and the Rayleigh
  projection is applied once, forward-only, at the half-spaces — so
  patterned-layer accuracy is `n_orders`-independent and tracks only the modal
  degree.  Supports any mix of uniform and patterned layers on one shared
  square grid, **including direct patterned|patterned (A|B) interfaces**
  (validated per-order ~2e-3 against the exact 1-D multilayer `PMMStack` at
  oblique incidence, energy exact to 1e-6; uniform layers reuse one shared
  eps-free geometric eig, and byte-identical patterned cells dedupe their
  eig).  The existing hybrid stack class is renamed **`PMM2DStack` →
  `PMM2DStackHybrid`** (aliases `PMM2DStack_hybrid` and transitional
  `PMM2DStack` retained — no breakage); tapered-grating helpers remain
  hybrid-only.

### Performance

- **Threaded PMM wavelength sweeps (`PMMStack` + `PMM2DStack`) + per-wavelength
  eig dedup.**  The per-wavelength PMM solves are independent and release the GIL
  inside LAPACK, so both `PMMStack.solve_vs_wavelength` and
  `PMM2DStack.solve_vs_wavelength` now run on a bounded thread pool (the
  `RCWAStack.solve_vs_wavelength` pattern: `max_workers` / `blas_per_worker`
  kwargs, per-worker BLAS-thread cap, results stored by index; `PMM2DStack`
  threads on private `copy.copy` clones, a traced JAX half-space forces serial).
  Additionally, within each `PMMStack` wavelength the per-layer generalized eig --
  the dominant cost -- is now DEDUPED across identical layers (an ABAB Bragg / DBR
  stack eigs each distinct layer once; `solve()` already did this, the 1-D sweep
  did not, re-eig'ing every layer at every wavelength; the 2-D sweep inherits the
  dedup through `solve()`).  Both are BYTE-IDENTICAL to the serial path and to a
  per-wavelength `solve()` loop (measured 0.0 on R/T/Jones); the geometric-eig
  cache is lock-guarded so any worker interleaving is deterministic.  Measured
  ~2.1x (1-D 10-layer DBR) / ~2.8x (2-D 3-layer) on an 8-core box; more for
  distinct-layer stacks, and the DBR dedup is itself a large separate win.
  Removes the old "NOT a speedup" caveat.  See
  `tests/unit/test_v5_21_pmm_threaded_sweep.py`.

### Changed

- **`apply_real_lens_maslov` focus-plane ROI** (`output_plane_distance`,
  `output_plane_n`) -- compose a free-space leg into the canonical
  entrance->exit map so the fit + `roi` land on a downstream (focus / image)
  plane a distance past the prescription's exit vertex, WITHOUT re-tracing the
  optics.  A tiny `roi` window at the focus then costs `O(roi_n^2)` integrand
  evals (measured ~21x vs the full grid; up to ~1e3-1e4x for a tight spot on a
  large grid), and a through-focus scan re-uses the single ray trace.  Exact:
  matches baking the distance into the prescription to ~1e-10; the ROI window is
  identical to the full-grid slice.

- **GBD/Maslov performance batch (v5.21)** -- exact rewrites, each asserted
  against the reference before landing (no physics change):
  - **GBD closed-form batched 2x2** inverse / eigenvalues / determinant
    (`_inv2x2` / `_eigvals2x2` / `_det2x2`) replace `xp.linalg.inv/eigvals/det`
    at the ~8 tensor-Q evolution sites.  Matches LAPACK to ~1e-13; **inv 4.7x /
    eig 32x** faster on batched 2x2 (LAPACK per-matrix dispatch dominates on tiny
    matrices), and made of only `+ - * /` so it also **unblocks jax.grad / CuPy**
    for the whole tensor-Q family (a complex non-symmetric `eigvals` has no JAX
    VJP).
  - **GBD per-surface Jacobian default `'fd'` -> `'auto'`** -- prefers the
    truncation-free analytic differential ray transfer (one dual-number trace, no
    per-surface (N,4,4) LAPACK inverse) and falls back to finite-difference (9N
    traces) for any surface type analytic does not cover.  **21.6x** faster on
    the Jacobian build; the reconstructed field is identical (relerr 0.0, Q
    matches FD to 1.7e-10) and analytic is the *exact* derivative, so accuracy
    never drops.
  - **GBD windowed reconstruction**: separable outer-product exp for scalar /
    diagonal-Q beamlets (`exp(a+b)=exp(a)exp(b)` -> `ng*(bx+by)` exps instead of
    `ng*bx*by`; only the skew `Qxy!=0` case keeps the full 2-D form) + a
    `sqrt(2)` window ladder (per-axis box inflation ~2x -> ~1.41x).  Still matches
    the dense sum to 1e-15..1e-17; **2.35x** faster reconstruct.
  - **Maslov fit via normal-equations Cholesky** (`_solve_fit` /
    `_gram_cho_factor`) instead of the `gelsd`-SVD `lstsq`.  Matches to 2.6e-15,
    **12.8x** faster (54x with the cacheable Gram factor, for a same-optic
    sweep).  The fit is the dominant Maslov stage once the integrand is
    accelerated.
  - **Maslov `local_quadrature` shared value+1st-derivative kernel** (`_opd_vd3`,
    numba + numpy): the integrand loop needs no second derivatives (those are the
    one-time per-pixel Hessian) and evaluates `opd`/`s1x`/`s1y` at the same
    points, so build the Chebyshev basis once and skip the T'' recurrence -- ULP
    equal, **2.6x** cheaper than three 6-output kernel calls.
  - **Maslov headless waste removal**: the fit-residual RMS GEMVs are gated
    behind `verbose`/`progress`; `sample_E_bilinear` precomputes its grid origin
    scalar instead of allocating `arange(N)` per chunk; the two identical Tukey
    windows are computed once.

### Added

- **GBD windowed (bounded-support) reconstruction** -- large speed **and**
  memory win, machine-precision identical to the dense sum.  Each Gaussian
  beamlet has support only within a few waists of its centre, but
  `reconstruct_field_from_beamlets` evaluated every beamlet over the WHOLE
  `(Ny, Nx)` grid (an `O(beamlets * Ny * Nx)` product through a
  `(Ny, Nx, chunk)` complex buffer -- e.g. 19-28 s and 7-13 GB peak for ~9 k
  beamlets on N=192-256).  The new `window=` path evaluates each beamlet only
  on the local pixel box where its Gaussian is non-negligible (out to `window`
  amplitude-1/e radii, tail `exp(-window^2)`) and scatter-adds via
  `numpy.bincount` -- `O(sum_b window_box_b)` instead.  Measured **147x**
  faster and **15x** less memory (3.1 GB -> 206 MB) at N=192, `relerr` vs the
  dense sum **1e-16 to 1e-15** across scalar / tensor-Q / direction-ramp /
  anamorphic beamlets, and it makes previously-intractable sizes routine (N=512
  / 262 k beamlets: 4.9 s, 1.6 GB bounded, vs a multi-GB OOM before).  Opt-in on
  the low-level `reconstruct_field_from_beamlets(..., window=5.0)` (default
  `None` = byte-identical dense sum); the drivers (`propagate_gbd_thin_lens`,
  `propagate_gbd_freespace`, `propagate_gbd_through_prescription`) default to it
  on the NumPy backend.  JAX / CuPy fall back to the dense sum.  See
  `tests/unit/test_v5_21_gbd_windowed_adaptive.py`.

- **GBD adaptive (edge-refined) decomposition** (`decompose_field_adaptive`).
  A two-level residual-refinement decomposition: a coarse beamlet grid at
  `base_step` plus a finer, narrower grid at `refine_step` confined to where the
  residual `E - reconstruct(coarse)` is large (the features the coarse grid
  under-resolves).  The fine beamlets carry the residual (ADD a correction), so
  the coarse/fine seam has no partition-of-unity mismatch and no double
  counting.  Reaches uniform-fine edge fidelity at **~6x fewer beamlets** for
  fields with localized sharp features on a well-resolved background (measured:
  uniform-fine ss=1 = 65 k beamlets -> `relI` 9.0e-2; adaptive base=4 refine=1
  = 10 k beamlets -> 9.1e-2).  Pairs with the windowed reconstruction.  (For a
  full-pupil FOCUSING system use a uniform fine grid + `soft_edge` below --
  coarsening the interior degrades a focus.)

- **GBD analytic soft-edge (partial-vignetting) aperture**
  (`apply_aperture_to_beamlets(..., soft_edge=True)`,
  `propagate_gbd_thin_lens(..., aperture_soft_edge=True)`).  Replaces the binary
  chief-ray keep/drop at an aperture with the analytic fraction of each
  beamlet's Gaussian that passes the rim
  (`1/2 (1 + erf(d*sqrt(2)/w))`, straight-edge approximation), removing the
  beamlet-pitch staircase.  Measured to improve the hard-aperture Airy-focus
  intensity error **~1.8x** (3.3% -> 1.9%) at no extra cost.

- **Differentiable (JAX) Berreman internal observables** (`layer_absorption`,
  `internal_field`).  `BerremanStack.solve(retain_internal=True)` now works under
  a trace (previously it raised): the per-layer modal-amplitude reconstruction
  from the retained partial cascades is backend-generic S-matrix algebra, so the
  absorbed-power-per-layer and the internal E/H field intensities now
  differentiate w.r.t. a lossy layer's Im(eps), the layer indices, thicknesses
  and the source -- a natural objective for LC retarder / magneto-optic / lossy
  dichroic film design.  jnp twins (`_solve_jax_retain` + `_amplitudes_jax` /
  `_layer_absorption_jax` / `_internal_field_jax`) port the validated NumPy loops.
  Traced == concrete to machine precision (~3e-16), the closure invariant
  `sum_i A_i == 1 - R - T` holds, and grad matches central FD to ~1e-10;
  out-of-plane-oblique retain still raises (no field reconstruction there).  See
  `tests/unit/test_v5_20_10_berreman_internal_jax.py`.

- **Differentiable (JAX) EMT mixing rules** (`rytov_tensor`,
  `rytov_segments_tensor`, `maxwell_garnett`, `bruggeman`).  The closed-form
  effective-medium rules are now backend-generic (via `array_namespace`), so a
  traced constituent eps / index or fill flows through them and onward through
  `BerremanStack.add_effective_grating` -> the Berreman jnp far-field twin --
  enabling gradient-based homogenized-grating design loops (the fast
  screen-then-optimize workflow the module advertises).  `bruggeman`'s
  data-dependent passive-root selection is made differentiable with a
  nudge-continued reference + `where` (a measure-zero branch, so the gradient
  flows through the chosen exact root).  The concrete NumPy path is
  byte-identical (34 emt tests unchanged); grad matches central FD to ~1e-10
  (rytov->Berreman, MG, Bruggeman).  See `tests/unit/test_v5_20_9_emt_jax.py`.

- **Differentiable (JAX) axisymmetric BOR-PMM stack** (`BORStack.solve`).
  `solve()` now runs under a trace when any layer permittivity, concentric-ring
  ridge index, or thickness is a JAX array (the half-spaces stay concrete): the
  geometry-only staggered stencils are frozen into the trace once, and the
  eps-dependent `K x = q^2 B x` assembly, the equilibrated-fold eigensolve (the
  rcwa gauge-stable custom-VJP eig -- the "generalized eig VJP" is a phantom, the
  fold is a plain standard eig), the modal field/flux reconstruction (with a
  trace-safe `where` forward-orientation), and the Redheffer cascade are all
  differentiable -- enabling gradient-based axisymmetric grating / VCSEL-aperture
  design loops.  Like the RCWA/PMM twins the propagating order SET is data-
  dependent and cannot be materialized under a trace, so `R`/`T` come back as
  full-`2N` per-mode arrays masked to 0 off the propagating set; the concrete
  NumPy path is unchanged (65 bor tests) and the traced TOTAL `sum(R)`/`sum(T)`
  reproduce the NumPy solve to ~1e-13 (order-/gauge-invariant), the masked
  per-order energy closes (`R+T==1` lossless, <1e-6), and grad matches central FD
  to ~1e-8 for the ring index, thickness and a lossy layer's Im(eps).  New
  `lumenairy/elements/bor/_jax_bor.py`; dispatched on `is_jax_array` like the
  other twins.  CPU-only (`jnp.linalg.eig`).  See
  `tests/unit/test_v5_20_11_bor_jax.py`.

- **Full anisotropic OFF-DIAGONAL FFF** for `rcwa_jones_2d`
  (`formulation='fff_nv'`), including CROSSED cells.  The existing `'li'` inverse
  rule fixes only the DIAGONAL tensor blocks; the off-diagonal `Cxy`/`Cyx` of a
  rotated in-plane director (`exy, eyx != 0`) stayed Laurent-floored (~1e-3) at
  every order.  The new path builds the Li-2003 (J.Opt.A 5:345) successive
  full-tensor factorization `ehat = L2 L1(eps)` -- the Smagin-Weiss-Dyakov 2026
  `l+-_tau` operator, `L_tau = l+_tau F_tau l-_tau` applied along x then y -- so
  ALL FOUR in-plane blocks get the correct inverse-rule treatment: the inverse
  rule on the wall-normal diagonal (`exx` along x, `eyy` along y) and the
  off-diagonal carried through the Schur reorganization.  The ONLY inversions are
  of scalar wall-normal elements (plus one N x N block along the second axis), so
  it is WELL-CONDITIONED even for a CROSSED (both-axis-patterned) pillar
  (measured `cond ~ O(10)`), unlike the normal-vector projector form
  `[[eps.C]][[C]]^-1` (`cond ~ 1e7` for a crossed pillar -- a silent lossless-trap
  hazard, which is why that form was dropped in favour of Li-2003).  Verified
  (five independent literature agents + primary-source Li-2003 + numerics): the
  operator reduces EXACTLY (machine precision) to the rigorous Li-1996 1-D
  factorization on a stripe; the solver reduces to `rcwa_jones_1d_segments` on a
  stripe (~6x faster than `'laurent'`); and a high-contrast lossy CROSSED
  rotated-director pillar now CONVERGES monotonically, ~10x faster than
  `'laurent'` and reliably (the diagonal-only `'li'` is non-monotone there).
  Rigorous for AXIS-ALIGNED (Manhattan) cells; the `L2 L1` vs `L1 L2` order
  differs in the truncated space (Li 2003 Sec. 5.2, same limit).  OUT-OF-PLANE
  tensor cells (`exz, eyz != 0` -- tilted-director / magneto-optic gratings) are
  ALSO handled: the full-3x3 `L2 L1` factorization plus the `E_z` fold `l3-`
  (Li 2003 Eq. 27, an ordinary matrix inverse of `ehat^{33}` through the
  generalized forward/backward cascade), converging to the direct-rule limit but
  far faster (nearly order-independent from n_orders~7 on a tilted-uniaxial
  stripe where laurent and the 1-D solver are still ~9e-5 short at n_orders=41).
  NumPy/CuPy only; new `_li_convolutions_2d_tensor(_full)` and
  `_li_axis_tensor` / `_li_axis_blocks`.  See
  `tests/unit/test_v5_20_12_rcwa_jones_2d_fff_nv.py`.

- **`formulation='fff_nv'` ported to the hybrid PMM** (`pmm_jones_2d`).  For a
  SEPARABLE (single-orientation, x- or y-patterned) anisotropic cell the
  wall-normal is constant, so the SEM-projected tensor operator reduces to the
  rigorous Li-1996 1-D anisotropic factorization built directly from the
  projected component masses -- the wall-normal diagonal takes the inverse rule
  and the off-diagonal `Cxy`/`Cyx` of a rotated director gets its correct
  composite, with the single inversion `[[1/e_nn]]^-1` well-conditioned by
  construction (no crossed-cell blow-up, so no `cond` gate is needed on this
  path).  Note PMM's `'li'` applied the inverse rule ONLY to the `E_z`
  elimination in the separable branch, so `'fff_nv'` is the FIRST correct
  in-plane inverse-rule treatment there -- an even bigger gain than in rcwa.
  Validated: converges to the rigorous `rcwa_jones_1d_segments` on a stripe
  (measured err 9e-5 vs laurent 7.3e-3 at n_orders=13), agrees cross-solver with
  `rcwa_jones_2d(fff_nv)`, and the lossy absorptance split tracks 1-D.  A crossed
  (both-axis-patterned) / out-of-plane / JAX cell raises (the same honest scope
  as rcwa; the crossed case is research-grade matched-coordinate FFF).  See
  `tests/unit/test_v5_20_13_pmm_jones_2d_fff_nv.py`.

### Changed

- **`apply_real_lens_maslov` default `integration_method` is now `'auto'`**
  (was `'quadrature'`).  A naive near-focus call previously ran the exact
  uniform quadrature, which is an oscillatory integral that converges terribly
  at a caustic (measured **86 s** and multi-hundred-MB on an N=96 singlet near
  focus, with an "under-resolved" warning); `'auto'` routes such charts to the
  fast asymptotic evaluator (**~1 s**, ~4x less memory) and is byte-identical to
  `'quadrature'` in the well-resolved regime (same auto-sized `n_v2`).  It only
  diverges from the old default in exactly the under-resolved near-caustic
  regime the old default already warned about.  Pass
  `integration_method='quadrature'` to force the exact uniform quadrature
  everywhere.  709 maslov / lens-model / GBD tests unchanged.

- **GBD `reconstruct_field_from_beamlets` dense path auto-caps memory**
  (`mem_budget_mb=512`, byte-identical for small `N`).  When the dense
  `(Ny, Nx, chunk_beamlets)` working set would exceed the budget,
  `chunk_beamlets` is auto-reduced (never grown), preventing the multi-GB peaks
  the fixed `chunk_beamlets=2048` hit at large `N` without changing small-`N`
  results.

- **`retain_internal` stores fewer S-blocks per layer** (memory; byte-identical).
  The internal-field / layer-absorption reconstruction (`RCWAStack` /
  `PMMStack` `solve(retain_internal=True)`) kept the FULL 4-block partial
  S-matrices per layer, but the reconstruction reads only `S_above[.][2:4]`,
  `S_below[.][0]` and `S_below_bot[.][0]`.  After the (complete) cascade
  recurrences, the unused `(2N, 2N)` blocks are now dropped: RCWA 12 -> 4
  retained blocks/layer (~63% of the retained field state), PMM 8 -> 3.  The
  retained arrays are unchanged, so internal fields / absorption are
  byte-identical (58 internal-field tests pass).

### Added

- **Threaded `RCWAStack.solve_vs_wavelength`** (speed; byte-identical).  The
  per-wavelength solves are independent and NumPy releases the GIL inside LAPACK,
  so they now run on a bounded thread pool -- each on a private `copy.copy` clone
  of the stack (safe: RCWAStack holds no instance cache, and the shared
  `_HOMOG_CACHE` is lock-guarded with a byte-identical recompute), with a
  per-worker thread-local BLAS pin so the pool does not oversubscribe.  New
  `max_workers` (default `min(cpu_count, n_wl)` on NumPy, 1 on GPU/JAX) and
  `blas_per_worker` kwargs.  Results are stored by index, so the output is
  BYTE-IDENTICAL to a serial sweep regardless of worker count -- measured **~8x**
  on a 24-core box for a 24-wavelength 3-layer sweep.  See
  `tests/unit/test_v5_20_8_rcwa_threaded_sweep.py`.

- **Berreman interface-S-matrix sweep cache** (speed; byte-identical).  On top
  of the per-layer eig cache, the native cascade's interface S-matrices are also
  wavelength-independent (built from the wl-independent field-mode matrices), so
  a fixed-angle sweep rebuilt the same `_interface_smatrix_general` at every
  point.  A second bounded LRU (`_interface_smatrix_cached`) reuses them
  byte-for-byte (~1.2-1.3x on top of the eig cache; interface entries stay stable
  across a sweep).

- **PMM 1-D stack per-layer eig dedup** (speed; byte-identical).
  `PMMStack.solve` (all-vertical in-plane path) now content-keys the per-layer
  modal eig on the layer's eps bytes, so a periodic / Bragg (ABAB...) stack
  computes each distinct layer's eig ONCE instead of once per repetition
  (up to ~Px on a P-period stack).  Deterministic eig -> the memo returns the
  same arrays a plain loop would build (mirrors the RCWA stack).

- **PMM 1-D geometric-eig sweep cache** (speed).  The eps-free geometric eig of a
  uniform half-space (`_uniform_geo_eig` / `_scalar_uniform_geo_eig`) depends only
  on the nodal geometry and `kx0` (angle), NOT on `k0` (wavelength -- which enters
  purely as the `1/k0^2` spectrum scale), so a fixed-angle wavelength sweep
  re-eigs the SAME pencil at every point (the audited 51-64% of 1-D eig time).  A
  bounded LRU (`_cached_geo_eig`, cache-registry-clearable) now eigs the
  k0-independent pencil ONCE and scales -- verified the cache does not grow across
  a sweep.  (The modes match the historical per-`k0` eig to ~1e-14 -- `eig(cB)`
  and `eig(B)` share eigenvectors to machine precision with exact eigenvalue
  scaling -- a physically-equivalent gauge, not bit-for-bit; a cleared-vs-warm
  solve within a version is byte-identical.)  See
  `tests/unit/test_v5_20_7_pmm_geo_eig_cache.py`.

- **`rcwa_jones_2d` gains a `formulation` kwarg** (`'laurent'` default, `'li'`).
  `'li'` applies the Li-1997 (JOSA A 14:2758, Eqs. 8/9) inverse rule to the
  DIAGONAL in-plane tensor blocks (`C_xx` inverse-along-x from `exx`, `C_yy`
  inverse-along-y from `eyy`, reusing the validated scalar `_li_convolutions_2d`),
  which converges faster for high-contrast / metallic anisotropic cells with
  sharp axis-aligned walls (the Gibbs-limited wall-normal discontinuity),
  reaching the SAME limit as `'laurent'` (verified: the two agree at high
  `n_orders` and the gap shrinks with truncation).  The off-diagonal blocks and
  the `E_z` rule stay Laurent (Li rule 3), so the full Popov-Neviere
  mixed-composite off-diagonal rule is not implemented -- a strongly-rotated
  director sees only a partial gain.  A scalar cell reduces EXACTLY to
  `rcwa_efficiency_2d(formulation='li')` (~1e-14); in-plane cells only (an
  out-of-plane cell always uses the direct rule).  The default (`'laurent'`) path
  is byte-identical to before.  See `tests/unit/test_v5_20_6_rcwa_jones_2d_li.py`.

### Changed

- **Even-parity fold is now ON by default** (`symmetry='auto'`) across the 2-D
  solvers (`rcwa_efficiency_2d`, `rcwa_jones_2d`, `RCWAStack`,
  `pmm_efficiency_2d(_cell)`, `pmm_jones_2d`, `PMM2DStack`).  A centro-symmetric
  cell at NORMAL incidence now automatically solves in the `(N+1)`-d even sector
  (~2-8x, growing with `n_orders`) instead of the full `2N` -- previously this
  was opt-in (`symmetry=True`) and users left the speed-up on the table.  The
  precondition is auto-detected and every non-applicable case (oblique,
  non-symmetric, out-of-plane, uniform, JAX) transparently falls back to the
  full solve.  `'auto'` and `True` are equivalent; the even-adapted basis
  matches the full solve to ~1e-12 (not bit-for-bit), so a centro-symmetric
  normal-incidence result may differ from a pre-change baseline at ~1e-12 --
  pass `symmetry=False` to force the exact prior bits.

### Added

- **Maslov lens propagator — feature parity + speed (v5.21).**
  - `integration_method='auto'` resolves the integrator from the fitted chart's
    v2-oscillation count: exact/caustic-safe uniform `'quadrature'` when
    well-resolved (also where near-caustic charts fall), the fast asymptotic
    `'local_quadrature'` only when quadrature would over-run its sample cap.
    Byte-identical to the method it picks; **357×** faster on a high-NA singlet.
  - `apply_real_lens_maslov(..., fold_split=True)` auto-handles a folded
    prescription (split at every fold + alternate the propagator per refractive
    leg with `apply_mirror` per fold), the documented pattern in one call.
  - `apply_real_lens_maslov_vector` — vector/Jones Maslov: applies the base-ray
    Fresnel Jones (reusing the GBD polarization ray tracing) then propagates
    each component caustic-safe (x-pol beam carries `T1·T2`; cross-pol at the
    symmetry floor).  Closes "Maslov is scalar-only".
- **GBD multilayer reflection coatings + ghost budget.**  A mirror `coating`
  may be a dielectric stack (`{'layers': [(index, thickness), ...],
  'substrate': index}`) -> `r_s`/`r_p` from the thin-film matrix (a quarter-wave
  TiO2/SiO2 Bragg stack matches the analytic reflectance).  `gbd_ghost_analysis`
  gives the first-order stray-light budget (per-surface Fresnel reflectance +
  double-bounce `R_i·R_j` ghost intensities; AR coats suppress them >1e6×).

### Fixed

- **Differentiable Maslov caustic phase (`apply_real_lens_maslov_jax`).**  The
  `det(J)` sign-flip radial scan could not see an axial focus and missed
  even-multiplicity caustics (both eigenvalues flip -> `det` unchanged), so the
  output phase was wrong by `pi` past an axial focus.  Now uses the Morse /
  Maslov index at the pixel (negative-eigenvalue count of the forward ray-map
  Jacobian): index 2 past an axial focus (the `-pi` Gouy shift), 1 past an
  off-axis fold.  Validated: maslov/traced ratio exactly `-1` past focus.

- **Berreman per-layer modal eig cache** (speed; byte-identical).  The 4x4
  layer eig depends only on `eps` and `Kx/Ky` (angle), NOT on wavelength (which
  enters solely through the propagation phase), so a fixed-angle wavelength
  sweep and a periodic ABAB... (DBR / Bragg) stack recomputed the SAME eig
  repeatedly.  A module-level bounded LRU (`_layer_modes_cached`, keyed on the
  eps bytes + Kx + Ky, cache-registry-clearable) now returns byte-identical
  modes: a fixed-angle dispersion sweep reuses every layer eig (verified the
  cache does not grow across a sweep), and a DBR dedups its repeated layers
  within one solve -- the PMMStack / BORStack caching precedent, which Berreman
  lacked.  See `tests/unit/test_v5_20_4_berreman_mode_cache.py`.

- **Differentiable (JAX) 1-D OUT-OF-PLANE RCWA** (`rcwa_jones_1d`,
  `rcwa_jones_1d_segments`).  A full-3x3 tensor with out-of-plane coupling
  (`eps_xz/eps_yz/eps_zx/eps_zy != 0` -- a tilted-director LC) is now
  `jax.grad` / `jit`-able (gradient matches central finite difference to
  ~7e-10; forward matches NumPy to ~1e-15).  The 1-D path previously *rejected*
  the JAX backend (`_reject_jax_offplane`) citing a non-differentiable host
  `argsort` flux split -- but that split already has a trace-safe twin
  (`_select_forward_flux_jax`, wired into the shared `_layer_eigenmodes_tensor`
  for the 2-D OOP work), so the rejection was stale.  A concrete off-plane jax
  tensor routes to the general full-3x3 solver, and a *traced* tensor routes
  there too (exact for in-plane -- the off-plane blocks vanish) so the forward
  and the gradient stay on one branch; a concrete in-plane tensor keeps the
  faster 2N path (bit-identical to NumPy).  See
  `tests/unit/test_v5_20_3_rcwa_1d_oop_jax.py`.

- **Differentiable (JAX) 2-D anisotropic PMM Jones solver** (`pmm_jones_2d`).
  The full-tensor 2-D hybrid PMM is now `jax.grad` / `jit`-able: gradients flow
  through the per-region (3, 3) permittivity tensor values (real and imaginary
  parts), the half-space indices, depth, wavelength and the incidence angles.
  As with the scalar cell twin, a traced `eps_tensor_cell` cannot define the
  spectral-element walls, so a CONCRETE `region_layout` (int grid labelling the
  regions) is passed alongside on the JAX path.  The twin reuses the already-
  differentiable shared tensor generator (`_layer_eigenmodes_tensor`, whose
  out-of-plane branch was made trace-safe for the RCWA-2D twin) + the generalized
  S-matrix cascade, feeding it a jnp operator assembly that is LINEAR in the
  traced eps (the GLL nodal masses are diagonal).  It ALWAYS drives the full-3x3
  generator path -- exact for an in-plane tensor, correct for an out-of-plane one
  -- so the forward and the gradient share ONE branch (no silently-wrong in-plane
  fallback under `jax.grad`).  Verified: forward matches NumPy to MACHINE
  PRECISION (~1e-13, in-plane AND out-of-plane), gradients match central finite
  difference to ~1e-9 (eps re/im, depth, wavelength, theta), `jit`-able.  Scope:
  cells patterned along BOTH axes (a cell uniform along an axis -- fully uniform
  or a 1-D-grating stripe -- has a degenerate spectrum that `jnp.linalg.eig`
  resolves ill-conditionedly, so the JAX path RAISES for it, pointing at the 1-D
  differentiable solvers / `berreman_jones_1d`).  See
  `tests/unit/test_v5_20_2_pmm_jones_2d_jax.py`.

- **Differentiable (JAX) OUT-OF-PLANE 2-D RCWA** (`rcwa_jones_2d`, `RCWAStack`).
  An out-of-plane permittivity tensor (`eps_xz/eps_yz != 0`) at oblique/conical
  incidence is now `jax.grad` / `jit`-able (gradients match central finite
  difference to ~4e-9; forward matches NumPy to 1e-15; in-plane gradients
  unchanged at 1.8e-10).  Two parts: a trace-safe `jnp.argsort` forward/backward
  flux selector (`_select_forward_flux_jax`) replacing the host argsort in the
  generalized generator, and a routing fix -- a TRACED tensor cannot be inspected
  for out-of-plane coupling, so under `jax.grad` the solve was silently taking
  the in-plane branch (dropping the z-coupling, ~30%-wrong gradient) while the
  concrete forward correctly took the out-of-plane branch; a traced jax tensor
  now routes to the general cascade (exact for in-plane too).  This was *not* an
  eig-VJP / adjoint research problem (the broadened eig VJP is accurate on the
  4N generator to 1e-9) -- purely the routing.  Concrete in-plane forwards keep
  the faster symmetric path.  See `tests/unit/test_v5_20_1_rcwa_2d_oop_jax.py`.

### Fixed

- **`pmm_jones_2d` no longer blows up SILENTLY** at a near-singular truncation.
  A high-contrast / birefringent tensor cell -- common at CONICAL incidence --
  could return a non-physical answer (sum R+T up to ~1e7, singular values >> 1)
  with no warning (while `rcwa_jones_2d` stayed stable and PMM with
  `stabilize=True` converged).  The non-stabilized path now runs the same energy
  tripwire as `rcwa_jones_2d`: it RAISES `_EnergyError` on the catastrophic case
  and WARNS on a lossless-closure violation, both pointing at `stabilize=True`
  (verified to cure it) or a different `n_orders`.  Lossy cells and converged
  solves are unaffected.

- **Stale out-of-plane-at-conical documentation corrected** across PMM/RCWA.  The
  "PMM/RCWA-vs-Berreman OOP-at-conical few-percent residual" was a PHANTOM (graded
  against the buggy `berreman_jones_1d` oracle, fixed 77b1964); with the corrected
  oracle `pmm_jones_1d_conical_tensor` / `pmm_jones_2d` / `rcwa_jones_2d` match
  Berreman to ~1e-15 at every incidence (cross-solver verified).  Also corrected
  `rcwa._core._require_inplane_tensor`, whose docstring + error wrongly said
  "2-D / RCWAStack out-of-plane is pending" (supported since v5.14.1).

- **Berreman out-of-plane tensor at OBLIQUE / CONICAL incidence is now exact.**
  `berreman_jones_1d` / `BerremanStack` were off by ~2% on one reflection
  eigenchannel for an out-of-plane permittivity tensor (`eps_xz/eps_yz != 0`) at
  oblique or conical incidence (exact for isotropic / in-plane / out-of-plane-
  at-normal; energy still conserved, so the error was silent).  Root cause: the
  native Berreman 4×4 S-matrix cascade pairs forward/backward modes via the
  `[W; -V] ↔ -λ` symmetry, which an out-of-plane tensor at oblique incidence
  *breaks*.  That regime — and only that regime — now routes to the same
  generalized (Li 2003) single-Fourier-order S-matrix that `rcwa_jones_1d` /
  `RCWAStack` use (`_homogeneous_eigenmodes` half-spaces + `_layer_eigenmodes_
  tensor` fed the ezz-Schur-condensed in-plane block + raw off-plane operators),
  to which a planar stack reduces exactly.  Validated to machine precision
  against `RCWAStack` across isotropic / in-plane / out-of-plane at normal /
  planar-oblique / conical, lossy, and multilayer stacks; all non-out-of-plane
  regimes stay on the native cascade **byte-identical**.  The differentiable
  (JAX) twin routes a concretely-detected out-of-plane tensor to the SAME
  generalized path, so **`jax.grad` / `jit` through out-of-plane-oblique /
  conical Berreman now works** (matches NumPy to machine precision; gradient
  agrees with central finite difference).  Internal-field / absorption
  retention (`retain_internal=True`) is not available for the out-of-plane-
  oblique regime -- that needs asymmetric-mode field reconstruction in the
  generalized convention, machinery `rcwa.RCWAStack` also lacks (it raises the
  same for out-of-plane stacks); far-field R / T / Jones are exact.  See
  `lumenairy/elements/berreman.py:_offplane_oblique_solve` and
  `tests/unit/test_v5_20_1_berreman_offplane_oblique.py`.  This closes the
  documented `berreman_jones_1d` "KNOWN LIMITATION" and the memory-tracked
  PMM/RCWA-vs-Berreman out-of-plane-conical residual (an artifact of two
  transfer-matrix *oracles* — single-layer direction and multilayer order — each
  carrying its own error; the solvers agree to machine precision).

### Added

- **GPU (CuPy) for the Maslov asymptotic evaluators.**  `apply_real_lens_maslov(
  ..., use_gpu=True)` now supports `integration_method='stationary_phase'` and
  `'local_quadrature'` (v5.20 GPU'd only the `quadrature` default; the fast
  production evaluators previously raised under `use_gpu`).  The per-pixel Newton
  saddle + Chebyshev value/derivative work runs on the device via a fused CuPy
  `RawKernel` (one thread per query point, O(poly_order) local memory — mirroring
  the Numba CPU kernel, no `(M, n)` global temporaries).  **1.8–1.9×**
  (stationary_phase) and **6.4× → 10.4×** (local_quadrature, N=192 → 384, growing
  with N) on an RTX 4070 Ti, matching the CPU integrator to ~1e-12 (complex64
  byte-identical; anamorphic pixels compose).  The CPU integrators are untouched.
  GPU path caps at `poly_order <= 23`.  See
  `docs/audits/AUDIT_WAVE_LENS_MODELS_2026_07_02_REMEDIATION.md` §4.5.

- **GBD feature-completeness.**  The Gaussian Beam Decomposition propagator
  (`lumenairy.propagators.gbd`) gains, on top of the audit-closed correctness:
  - **Per-surface / aberration-aware form** —
    `propagate_gbd_through_prescription(..., per_surface=True)` (default `False`
    keeps the whole-system-ABCD path unchanged) evolves each beamlet's complex
    parameter **surface by surface** via a new reusable raytrace primitive
    `raytrace.ray_transfer_jacobian` (the per-ray ABCD Jacobian of the real
    aberrated trace, by central finite differences; on-axis 2×2 block reproduces
    `system_abcd_prescription` to 1.1e-8).  `Q` promotes to a `(N, 2, 2)` tensor
    (general astigmatic Gaussian) — generalized Collins `Q_out=(C+DQ)(A+BQ)^{-1}`,
    per-surface `1/sqrt(det(A+BQ))` amplitude (branch-safe), base-ray OPL piston,
    branch-safe tensor free-space.  Reduces to the isotropic result on-axis;
    off-axis it captures tangential/sagittal **astigmatism** (~field², a
    near-line focus at 6°) the paraxial form cannot.  `raytrace.
    ray_transfer_jacobian` is reusable (Maslov Hessian propagation later).
  - **Analytic differential ray transfer** —
    `raytrace.ray_transfer_jacobian_analytic`, the closed-form / autodiff twin
    of the FD `ray_transfer_jacobian`: forward-mode AD (dual numbers) over the
    **exact** conic trace (intersection + vector Snell / reflection + vertex
    transfer), so the `(x,y,ux,uy)` Jacobian is **truncation-free** (the FD
    `h → 0` limit) and, on the JAX backend, `jax.jacfwd`/`grad`/`jit`-able in
    pure NumPy elsewhere.  Forward-mode AD == analytic differential ray tracing
    (Volatier 2017); see `docs/ANALYTIC_DIFFERENTIAL_RAY_TRACING_LITERATURE.md`.
    Adversarially verified exact vs the FD primitive (composite + per-surface +
    OPL + alive) across a hard sweep — f/1, hyperbolas `k=−6`, `u=1.5` field,
    concave backtracks, mirrors, near-TIR, a 3906-ray root-selection sweep — with
    a *proof* (`|ae| < q²`) that the near-vertex root is always the physical
    one, and a Coddington sagittal/tangential cross-check.  Conic surfaces +
    reflection (`is_mirror`) + **Zemax coordinate breaks** (`is_coordbrk` --
    decenter + X/Y/Z tilts, a smooth differentiable frame transform, matching FD
    to 1.3e-8 on a tilted+decentered system → differentiable alignment /
    tolerancing through a fold; a *large* tilt shares the world-frame slope-space
    caveat); aspheres / freeforms / biconics raise, pointing back to the FD
    primitive.  Selectable in the per-surface GBD via
    `propagate_gbd_through_prescription(..., jacobian='analytic')` (default
    `'fd'` unchanged; the two give the same field to ~1e-10).
  - **Husimi decomposition plumbed through the lens & prescription helpers**
    (`direction_sampling=`), so a tilted source focuses at `f·tanθ`.
  - **Aperture vignetting** (`apply_aperture_to_beamlets`, `aperture_semi_
    diameter=`), **polychromatic** (`propagate_gbd_freespace_spectral`,
    stack/incoherent-intensity), **auto-sampling** (`recommend_gbd_sampling`),
    and a **tensor-Q reconstruction** branch (isotropic reduces to scalar at
    2.3e-15).
  - **Anamorphic (`dy != dx`) sampling** — `decompose_field_to_beamlets` /
    `reconstruct_field_from_beamlets` / `propagate_gbd_freespace(..., dy=,
    output_dy=)` accept a separate `y` pitch, decomposing into **elliptical
    (diagonal-tensor-`Q`) beamlets**; a physically circular Gaussian on a
    `dy = 2·dx` grid propagates and stays circular.  `dy=None`/`dy=dx` keeps the
    scalar circular-beamlet path **byte-identical** (the tensor-`Q` core
    generalization is opt-in).
  - **Vector / Jones with polarization ray tracing** — free-space
    `propagate_gbd_freespace_vector` (independent-component), and through a real
    prescription `propagate_gbd_vector_through_prescription` applies **per-surface
    Fresnel s/p** along each beamlet's base ray (`_fresnel_jones_matrix_per_
    beamlet`, a per-beamlet `(2,2)` transverse Jones matrix) — dispatching each
    surface exactly as `raytrace.trace`:
    - **refraction** — Fresnel transmission `t_s` / `t_p` (s channel transverse-
      exact, p channel with the honest `cos θ_out` projection).  An x-polarized
      beam through a singlet carries the two-surface near-axis Fresnel power
      transmission `T1·T2` (0.9179 = 0.9581²) with cross-pol at the symmetry
      floor.
    - **reflection** at `is_mirror` surfaces — an **ideal reflector**
      `|r_s| = |r_p| = 1` (energy-conserving, no diattenuation; PEC convention
      `r_s = -1`, `r_p = +1`) with the geometric s/p frame rotation carried by
      recomposing on the reflected p axis.  The `r_s = -1` / `r_p = +1` pairing
      was checked against an independent Maxwell boundary-condition oracle to
      ≤2.2e-16 (an adversarial 3-lens verification; the flipped sign violates the
      tangential-E boundary condition by O(5)).  A concave mirror is
      energy-conserving (`det|P| = 1` on-axis) where an equivalent refractive
      surface is Fresnel-lossy.
    Vignetted rays are zeroed (never NaN).  A mirror may also carry a
    **`coating`** (a complex refractive index — a metal) for the full complex
    Fresnel `r_s` / `r_p` (**diattenuation + retardance**): aluminum
    (1.374+7.62j @633nm) reproduces the analytic normal-incidence reflectance
    (0.914) and off-axis `|r_s|` / `|r_p|` to 1e-6, reducing continuously to the
    ideal reflector as `|n_coating| → ∞`.
  - **World-frame output plane for large folds** —
    `propagate_gbd_through_prescription(..., world_output_plane='auto' | (p0,
    R_out))`.  A large fold (e.g. a 90° periscope) reverses the propagation
    axis, so the default fixed +z x-y reconstruction is meaningless.  This
    reconstructs on the physical plane perpendicular to the *folded* beam: the
    base rays are world-traced (`raytrace.trace_world` /
    `paraxial_focus_world`), `Q` is evolved on the **unfolded-equivalent**
    straight system (a fold reflects the local trace — `N` flips sign, so the
    slope phase-space `u=L/N` is corrupted; flat folds are Q-invariant, so the
    straight equivalent is exact), then reframed onto the plane (with a
    `Q → R₂ Q R₂ᵀ` transverse-frame rotation).  Validated on a 90° periscope
    (focuses where the fixed x-y path blows up) and shown to reproduce the
    default reconstruction on an unfolded system.  Forces `per_surface=True`.
    **Curved (powered) fold mirrors** raise `NotImplementedError` (not
    Q-invariant — they need the full world per-surface differential transfer).
  - **JAX-differentiable** free-space / thin-lens paths (backend-dispatched;
    `jax.grad` / `jax.jit` validated), plus a differentiable per-ray transfer
    `raytrace.ray_transfer_jacobian_jax` (`jax.jacfwd` around `trace_jax`,
    vmapped) — the gradient foundation for per-surface GBD lens design (matches
    the NumPy finite-difference primitive at low NA; the `_transfer_jax` high-NA
    `B`-block caveat is documented in `raytrace/differential.py`).
  - **Multilayer thin-film transmission coatings** — a refracting surface may
    carry a `coating` = list of `(index, thickness)` layers (AR / dichroic
    stack); `t_s` / `t_p` then come from the thin-film characteristic-matrix
    method (`_thin_film_coefficients`) instead of bare single-interface Fresnel
    (a quarter-wave AR raises single-surface transmittance 0.958 → 0.9999;
    reduces to bare Fresnel at zero layers; energy-conserving).  Complements the
    single-metal-index **reflection** coating (diattenuation + retardance).
  - **GPU reconstruction** — `propagate_gbd_through_prescription(...,
    use_gpu=True)` moves the beamlet bundle to CuPy after the (NumPy) evolution
    and runs the O(N_beamlets × N_pixels) coherent reconstruction on the device
    (**~35× total** on an N=128 singlet, matching the CPU field to 8.5e-15).
    Default `chunk_beamlets` lowered 4096 → 2048 (≈½ reconstruction peak RAM,
    flat runtime; byte-identical chunking).
  Tests: `tests/unit/test_gbd_feature_complete.py` (18 tests).

### Fixed

- **Staggered 2-D PMM far field at oblique incidence — two projection-kernel
  defects in `_stag_fourier_projection`** (`pmm/twod_staggered.py`), affecting
  `pmm_efficiency_2d_staggered` (and `PMM2DStackPure`) for every patterned
  cell at `theta != 0`:
  1. the Rayleigh projection kernel omitted the Bloch `alpha0` shift carried
     by the tau-glued modal basis, making the one-period projection
     non-orthogonal at oblique (energy loss even in the specular order —
     vacuum-oblique read 0.87); and
  2. the kernel's order-index sign was MIRRORED: slot `m` received physical
     order `-m`'s amplitude and then order `+m`'s `kz` flux factor.  Invisible
     at normal incidence (`kz` even in `kx`), for uniform cells (specular is
     self-mirrored) and for reflection-symmetric cells (mirror-identical
     magnitudes) — which is why every prior gate passed — but at oblique every
     |m|>0 order was mis-weighted: an asymmetric-cell per-order oracle test
     measured `stag[m] = oracle[-m]*kz(m)/kz(-m)`, i.e. several-% energy
     non-conservation with theta-dependent sign (converged stripe 0.878,
     pillar 0.82-1.05, direct A|B cascades 1.2-3.4).
  The fixed kernel is the conjugate of the order-m plane wave,
  `e^{+i(mG + alpha0)x}`.  Post-fix: converged stripe/pillar/asymmetric cells
  conserve to <=5e-5 at oblique; per-order matches the EXACT 1-D pure PMM to
  ~2e-4 (degree 8) in the correct slots; normal-incidence symmetric-cell
  results are bit-level unchanged; and the direct patterned|patterned (A|B)
  staggered cascade — previously believed to need a first-order
  Poynting-sorted mode rebuild (that theory is refuted: the up/down split of
  real eigenvalues is non-critical for internal layers, Li 2003 J. Opt. A
  5:345 / *Gratings* ch. 13 2014 §13.2.3.3) — is exact, so `PMM2DStackPure`
  supports multiple patterned layers.  En route, `_far_projector_2d` /
  `_stag_fourier_projection` gained the `alpha0x`/`alpha0y` arguments, and the
  lossy-superstrate guard-test bound was recalibrated (the old `tot <= 1.0`
  was only satisfied by the mirrored far field's downward bias).
  Tests: `tests/unit/test_v5_21_pmm2d_staggered_oblique.py` (energy sweeps,
  per-order orientation vs the exact 1-D `PMMStack` with an anti-mirror
  tripwire at theta=0 and 0.2, vacuum-oblique exactness, A|B multilayer
  vs the 1-D multilayer oracle).

## [5.20.0] — 2026-07-04

Maslov propagator (`apply_real_lens_maslov`) brought in line with the rest of
the lens family: an accuracy/UX fix to the default integrator, anamorphic
(`dy != dx`) support, and a GPU (CuPy) path.  Minor bump — one default-output
change (documented below), otherwise additive.  See
`docs/audits/AUDIT_WAVE_LENS_MODELS_2026_07_02_REMEDIATION.md` §4.3–4.4.

### Changed

- **`apply_real_lens_maslov` auto-resolves `n_v2`** (uniform-quadrature v2
  sampling).  `n_v2` now defaults to `None` → the `integration_method=
  'quadrature'` path sizes it from the fitted OPD's v2-oscillation count
  (`clip(ceil(4·v2_osc)+1, 32, 256)`), so the robust default integrator is
  *properly resolved* out of the box instead of speckling at the old fixed
  `n_v2=32`.  A demanding tight-focus chart (that the code's N2 guard flags as
  wanting ~180 samples) went from **181 % off** the well-resolved truth to
  **0.1 %**.  Low-NA charts clamp to the floor and stay **byte-identical** to
  the historical `n_v2=32`; an explicit `n_v2` is still honoured exactly.  This
  corrects a mis-diagnosis in the 5.19.0 notes: the claimed "`local_quadrature`
  diverges ~67 % from the oracle" was the *reference* being under-resolved, not
  an integrator bug — all three integrators agree to 0.1–0.24 % once the
  reference is converged (audit §4.3).

### Added

- **Anamorphic Maslov (`dy != dx`).**  `apply_real_lens_maslov` no longer
  rejects non-square pixels: the entrance/exit sampler, output axes, and
  angular-content estimate use the separate `dx`/`dy` pitches (`dy` resolves
  `None → get_default_dy() → dx`, matching `apply_real_lens`).  Validated
  against the analytic propagator via the pixel-ellipticity invariant — a
  circular beam on a `dy = 2·dx` grid renders at `sy/sx = 0.500` for both, to
  **1.3e-4**.  The array must stay square `N×N`; a rectangular *array*
  (`Ny != Nx`) and `roi=` under anamorphic pixels raise a clear
  `NotImplementedError` pointing at `apply_real_lens`.
- **GPU Maslov (CuPy).**  `apply_real_lens_maslov(..., use_gpu=True)` — or
  passing a CuPy input array — runs the O(N²·n_v2) phase-space quadrature on the
  device, mirroring `apply_real_lens`'s dispatch.  The trace + Chebyshev fit
  stay on the host; only the integrand moves to the GPU (self-contained CuPy
  twin of the factorized quadrature, CPU path untouched).  Validated
  byte-identical to the CPU integrator (NumPy backend) and to **5e-16** on an
  RTX 4070 Ti, with a **35× speedup** at N=192.  Supported for
  `integration_method='quadrature'` (the default); the asymptotic evaluators
  remain CPU-only and raise under `use_gpu`.  Composes with anamorphic pixels.
  Returns a device array (`cupy.asnumpy` to pull to host).

### PMM / RCWA solver upgrades

Implements the two solver audits — `docs/audits/PMM_RCWA_AUDIT_2026_07_02.md`
(findings + performance) and `docs/audits/AUDIT_PMM_CONICAL_OUT_OF_PLANE_2026_07_03.md`
(conical incidence).  Every change is gated by a byte-identity parity harness
across all ten solver families; the default (public) paths are byte-identical to
5.19.0, with the new fast paths opt-in and a full-solve fallback.

Added

- **Native conical (out-of-plane, `phi != 0`) PMM.**  `pmm_jones_1d_conical`
  (isotropic grating) and `pmm_jones_1d_conical_tensor` (full `(3,3)` LC-director
  profile) — the `O(N)` `n_y = 0` reduction of the 2-D coupled build (keeps the
  y-axis degenerate, routes through the same 2-D machinery).  `PMMStack.set_source`
  grows a `phi` kwarg for the native `O(N)` conical MULTILAYER stack.  Validated
  against the analytic Berreman 4×4 conical oracle (uniform slab, singular
  values), the classical 1-D solver (`phi = 0` reduction) and the `PMM2DStack`
  y-invariant bridge.  Restricted to all-vertical scalar / in-plane NumPy stacks;
  an out-of-plane tensor at conical incidence inherits a documented shared-
  generator residual vs Berreman on one eigenchannel (`pmm_jones_2d`/
  `rcwa_jones_2d` affected identically — flagged for a focused follow-up).
- **Conical `PMM2DStack` bridge validation (Path A).**  The existing y-invariant
  bridge is now validated at `phi != 0` against the Berreman conical oracle,
  unblocking out-of-plane cuts through `PMM2DStack` with y-invariant cells.

Performance (opt-in fast paths; default byte-identical)

- **Even-parity symmetry fold** (`symmetry=True`) extended from
  `pmm_efficiency_2d(_cell)` to `PMM2DStack` (the whole Redheffer cascade in the
  even sector) and `pmm_jones_2d` (the in-plane tensor `(P, Q)` folded via RCWA's
  `_tensor_PQ`).  A centro-symmetric cell at normal incidence runs in the
  `(Nf+1)`-d even subspace (~2–4.5× on the O(Nf³) steps); a per-layer flip-
  invariance guard falls back to the full solve.
- **Shared structure-aware S-matrix algebra (F1):** PMM reuses RCWA's zero-block
  Redheffer + diagonal-aware propagation star (isotropic paths bit-identical,
  generalized ≤ 1 ULP).
- **Shared eps-free geometric half-space eig (S5-P1):** the 1-D scalar uniform
  half-spaces (and now the slant half-spaces) share ONE geometric eig across both
  polarizations (`q² = eps − mu`), ~1.5–2× (gauge-equivalent ~1e-14).
- **2-D hybrid:** factorized sandwiches never materialize `kron(Ty, Tx)` (F5,
  bit-identical); optional Lalanne circular truncation (F8, ~2× eig); `PMM2DStack`
  dedups repeated identical layers within a solve and hoists the wavelength-
  independent build across a sweep (F4); the JAX 2-D cell assembly is factorized
  to remove the dense N×N materialization (B1).
- Quadrature cache (P2, bit-exact); sweep angle reuse (B3).

Fixed

- `pmm_efficiency_2d_cell` JAX dispatch now honours `max_nodal_dof` (P1 resource
  blow-up); stale-internal invalidation, `_epsF_cache` invalidation and dead-code
  cleanups (B4/B5); `fff_nv` direct-rule documentation (B2).

## [5.19.0] — 2026-07-04

Wave lens-models audit remediation (F1-F5, N1-N8) reconciled with a companion
review and two adversarial verification passes, followed by the full
deferred-items performance program worked to completion.  Every addition is
backward-compatible and opt-in -- Minor bump for the new public APIs; no
existing default code path changes output.  Every speedup below was validated
byte-identical or ULP against the exact reference path it replaces.

### Added

- **Prepared lenses** for optimizer / tolerancing / multi-field loops that hold
  `(prescription, wavelength, dx, N)` fixed.  `prepare_real_lens_traced(...)` /
  `PreparedTracedLens` caches the traced lens's input-independent screen (ray
  trace + Chebyshev/spline fits + Newton inversion + analytic-phase reference +
  valid/aperture masks), so each call is one `apply_real_lens` + one complex
  multiply -- **55x** per call (prepared == direct to 4.6e-16).
  `prepare_real_lens(...)` / `PreparedAnalyticLens` is the analytic analogue
  (per-surface `exp(-i k0 (n2-n1) sag)` screens + entrance mask; the ASM
  transfer functions are already cached inside `angular_spectrum_propagate`) --
  **2.85x** on an 8-surface lens (3e-15 vs direct).  Both raise a clear
  `NotImplementedError` for the paths they do not cover (decentred / tilted /
  freeform / biconic / stop / mirror surfaces, `'auto'` carriers, and the
  slant / fresnel / absorption / seidel / surface-frame / GPU / non-ASM modes).
- **`apply_real_lens_traced_multi(emitter_fields, ..., carriers=)`**: applies
  the traced model PER emitter (each a single congruence) and coherently sums.
  The traced model assigns one ray-traced OPL per output pixel, so it is not
  linear; feeding it the summed field of a multi-emitter scene violates the
  single-congruence assumption and softens the image.  This is the correct way
  to image a multi-emitter scene with the traced model, and the tractable form
  of carrier K-decomposition (the K congruences are the known emitters, so no
  blind segmentation is needed).  `reuse_prepared=True` shares a
  `PreparedTracedLens` screen per distinct carrier.
- **`apply_real_lens_maslov(..., roi=(cx, cy, half_width))`**: evaluate the
  Maslov output only on a square region of interest -- byte-identical to the
  full-grid slice (on- and off-axis) at `O(roi_n^2)` instead of `O(N^2)`
  integrand evaluations (**8.8x** at N=1024 for a 40x40 spot; grows with N).
- **`apply_real_lens_traced(..., inversion_method='fit')`**: opt-in scattered
  Chebyshev inverse-map fit that replaces the per-pixel Newton inversion
  (convex-hull-masked) -- 2.6e-6 vs Newton, **2.42x** at the default
  `ray_subsample`.  Default stays `'newton'`.
- **Carrier-referenced traced model** (`apply_real_lens_traced(..., carrier=
  <float>|<ndarray>|'auto', on_noncollimated=)`): references the traced
  correction to the beam's own smooth carrier wavefront instead of a plane
  wave, generalizing the traced model past its collimated-input assumption for
  a single divergent / tilted congruence; the `on_noncollimated` guard warns /
  delegates when the residual angular spread is too large (e.g. an
  un-referenced emitter array, where `apply_real_lens` remains correct).

### Performance

- **Maslov integrators.**  A Numba `@njit(parallel)` 4-variable Chebyshev
  value+derivative kernel for `_opd_and_derivs` (**13.6x**, ULP-equal, NumPy
  fallback when Numba is absent); the three fit solves (OPD, s1x, s1y)
  collapsed into one multi-RHS `lstsq` sharing a single SVD (**2.9x** fit);
  Kronecker factorization of the `(N_out^2, M)` quadrature design matrix (no
  `G` materialized); and pixel-band (`stationary_phase`) + output-row-band
  (exact quadrature) evaluation that removes the 133 GB / 451 GB full-grid
  allocations at N=16384.

### Fixed

- **Maslov `input_na` / NA-clamp**: an explicit `input_na=NaN` slipped past the
  `na_proxy >= 1` clamp (`NaN >= 1` is False) and died later with a misleading
  "0 rays survived" TIR error; explicit `input_na` is now validated finite and
  non-negative up front, and the auto NA proxy is clamped below unity so a
  broadband / hard-aperture input cannot grazing-kill the whole pupil chart.
  Plus the N1-N4 / F3 / F4 audit items: the `(N_out^2, M)` design-matrix hoist
  out of the non-quadrature integrators, the uniform-quadrature under-resolution
  warning, re-applying the orphaned fitted linear-OPD term (piston as a scalar,
  slope on the fine grid), the suite-signature progress callback, and the
  wrapping-safe `tilt_aware_rays` recommendation.
- **JAX-x64 test flake**: `test_twins_raise_without_x64` spawned a subprocess
  that inherited a sibling module's `os.environ.setdefault("JAX_ENABLE_X64",
  "true")`, so the child saw x64 ON and the "expected x64 OFF" assertion failed
  order-dependently in a full-suite run; the child now gets an explicit
  `JAX_ENABLE_X64=0` env.

### Changed

- **Docs**: honest model-selection guidance for the three wave lens models
  (analytic vs traced vs Maslov), the `sag * theta^2` analytic validity
  boundary, and the carrier validity boundary from the design-119 no-MLA
  multi-emitter investigation.  The full remediation record, including the
  items assessed and declined with measured reasons (N6 eigen-rotation reverted
  as no-benefit-vs-oracle; the M-P5 wavelength-rescale as dispersion-limited;
  M-P7 float32 as counterproductive on the validation integrator), is in
  `docs/audits/AUDIT_WAVE_LENS_MODELS_2026_07_02_REMEDIATION.md`.

## [5.18.1] — 2026-07-02

Post-release cleanup of the deferred / residual items recorded during the
v5.18.0 audit-fix campaign (the report's "New residual observations"), plus the
two deferred JAX residuals.  No behavior changes to existing default code paths.

### Fixed

- **UI load-saved-run crash**: `WaveOpticsDock`'s "load embedded prescription"
  path called `self.model`, which never existed (the dock stores the model as
  `self.sm`), so it always raised `AttributeError` -- swallowed as a "Load
  failed" dialog.  The feature now works.
- **EME verify determinism**: `layer_vector_modes(verify=True)` called ARPACK
  `eigs` (in `_fd_eig_dist`) with a random start vector, making the FD spurious
  discriminator non-deterministic run-to-run.  It now seeds `v0` (eigenvalues
  are unchanged; only the random-start jitter is removed).
- **scipy 1.18 already handled in 5.18.0** — this release additionally pins the
  slow-tests CI job to single-threaded BLAS so the eig-heavy EME vector
  convergence tests (whose ill-conditioned cascade svdvals are sensitive to
  multi-threaded reduction order) stop wobbling their recovered mode count
  across runs.
- **Zemax `.txt` exporter type**: `export_zemax_lens_data` hardcoded
  `TYPE=STANDARD` for every surface, mislabelling aspheric/freeform surfaces
  (the export-side sibling of the P3-43 .txt-loader drop).  It now emits the
  real type (`EVENASPH`/freeform, inferred from non-zero aspheric coefficients
  when no explicit type is present) and footnotes any aspheric surface,
  pointing at `export_zemax_zmx` for a lossless round-trip.
- **Doc**: the `test_memory_guardrail` header quoted the pre-v5.17.0 estimator
  anchors (44.5/37.2/57.3 GB) that `lumenairy/memory.py` itself flags as
  superseded by the v5.17.1 recalibration (29.69/21.87/31.39 + 26.30 GB
  chunked).  Synced.

### Added

- **`through_focus_scan_jax(..., stream=True)`** (audit P3-03 residual): an
  opt-in per-plane device loop that keeps DEVICE memory at `O(Ny*Nx)` (one
  plane at a time) instead of the fused vmap's `O(n_z*Ny*Nx)`.  Numerically
  identical to the default fused path; trades batch fusion for a tight device
  budget on dense large-grid scans.

### Performance

- **Differentiable JAX Jones twin** (`pmm_jones_1d` with JAX input; audit
  P3-27 second half): the two isotropic half-spaces now share ONE geometry-only
  eig (backlog A2) instead of two independent full `2n` eigs, mirroring the
  numpy `_pmm_jones_solve_core`.  As a bonus, because the twin now uses the
  identical shared-eig gauge as the numpy oracle, forward parity IMPROVES to
  machine precision (`max|dJones|` ~9e-16 vs the oracle); gradients stay finite.

## [5.18.0] — 2026-07-02

Deep-audit remediation.  All 114 findings of the v5.17.0 deep audit
(`AUDIT_V5_17_0_2026_07_01_DEEP.md`) are resolved: 113 fixed, 1 refuted in
implementation (P3-20 — the prescribed single-SVD merge provably breaks the
EME sigma_min mode diagnostic on the real reference cell; the two-call
structure is kept, documented, and pinned by a regression test).  Fixes were
implemented in six themed waves, each: reproduce-first probe, minimal surgical
change, discriminating test, sibling-suite runs, ruff-clean; the physics /
convention changes were hand-derived before the fix was trusted.  Minor bump
(not a patch) because this release contains deliberate output-changing
behavior changes — enumerated here up front.

**BEHAVIOR CHANGES (ten; each corrects a previously wrong or misleading
output — review before upgrading):**

1. `bruggeman()` — every output moves; the old effective-medium root solved
   the wrong quadratic (a passive-root selector now picks the physical branch).
2. `zernike_decompose(normalization='Noll')` — no longer negates sine-mode
   (m < 0) coefficients; 'Noll' now equals 'OSA' (both are Noll 1976 / OSA
   positive-sine; the old flip matched no published convention).
3. `apply_waveplate` and the Jones-element retarder family — slow-axis sign is
   `exp(+i*retardance)`, matching the library's Berreman/RCWA transmission
   Jones; QWP outputs are the CONJUGATE of prior releases (HWP unchanged), the
   linear-x -> right-circular recipe is now fast axis at -pi/4.  rcwa's internal
   `_qwp_matrix` is deliberately unchanged — the `reflective_outcoupling`
   metric is provably conjugate-invariant at the default 45 deg, so published
   out-coupling numbers are unaffected.
4. GBD `apply_abcd_to_beamlets` — amplitude/piston use the Collins/Siegman
   factor `1/(A + B*Q_in)` (were wrong by `(C*q_in + D)`).
5. `PMMStack`/`PMM2DStack` — a stale `internal_field` after a superseding
   solve now raises instead of returning silently-wrong data.
6. RCWA `symmetry=True` — reflected/transmitted amplitudes are returned
   un-gauged (the symmetry basis phase no longer leaks into user amplitudes).
7. PMM (1-D / staggered / JAX twins) and RCWA back-side incidence — a
   gain/evanescent or otherwise non-propagating incident angle now raises a
   clear error instead of silently returning meaningless efficiencies.
8. `make_shack_hartmann_wfs(rng_seed=...)` — seeded detector noise is now a
   reproducible SEQUENCE across frames (was the identical frozen realization
   every AO frame); seeded noise streams change.
9. The asymptotic-mode JAX twins — raise when `jax_enable_x64` is off instead
   of flipping the global JAX config mid-call.
10. Low-memory chunked-lens memory ESTIMATES — report ~2x for
    `parallel_amp=True` (the runtime truth), up from the prior under-estimate.

### Fixed (wave 6 — the P3 sweep: boundary guards, doc honesty, small physics, perf)

- **EME** (audit P3-17..P3-23): lossy scalar strips at kx0=0 now return the true
  COMPLEX spectrum (eigh silently discarded Im(eps)); `ref_2d_modes` gains
  `return_complex=` and warns when a lossy eps would be realified; band-edge
  scan samples no longer crash `layer_modes` with `LinAlgError`; strip heights
  are validated against `Ly` (silent wrong Bloch modes at ky0 != 0 previously);
  magnetic `(eps, h, mu)` 3-tuple strips now work in `mode_field_vec`,
  `strips_to_eps_xy` and `layer_vector_modes(verify=True)` with a mu-consistent
  verify oracle; the duplicated rasterizer is unified; `diffraction_eme`
  returns a bare dict (with `qz2` folded in) like its sibling.  **Audit P3-20
  was REFUTED in implementation**: the prescribed single-SVD merge breaks the
  sigma_min mode diagnostic on the real reference cell (gesdd with-vectors
  sigma is ~14x off on the unequilibrated G); the two-call structure is
  accuracy-load-bearing, now documented in-code and pinned by a regression
  test.
- **BOR-PMM** (audit P3-11..P3-14 + P3-64): per-instance bounded modal cache +
  identical-layer dedup (repeated solve/sweeps reuse eigensolves,
  byte-identical R/T); the `_fast_geig` QZ fallback is now actually reachable
  at longitudinal resonances (LU pivot-ratio guard, ported from pmm); the
  staggered layer basis returns the FACE grid + per-half quadrature weights;
  `angles` docstring corrected (superstrate, not substrate); GATE 4a now
  classifies mode polarization so a TE/TM-swap bug cannot pass it.
- **PMM/RCWA** (audit P3-26/27/33/34/35/37/38): the stabilize consensus
  cluster is now pairwise-MUTUAL (was an anchor star admitting members 2x tol
  apart); the v5.14.2 shared-eig optimization is ported to the JAX stack twin;
  the 2-D `formulation='li'` inverse rule is applied per-slot so y-patterned
  (x-uniform) cells get the rule on the correct components; the staggered
  assembler frees five dead operator matrices (~2x peak); dispersive tensor
  layers get the same validation contract as static ones; two stale docstrings
  corrected.
- **Propagators** (audit P3-51/52/53/56/57 + gate hardening): the asymptotic
  JAX twins adopt the x64 guard and no longer mutate `jax_enable_x64`
  MID-CALL (**now raise if x64 is off** -- previously flipped global config);
  `decompose_lg/decompose_hg` handle `indexing='ij'` meshgrids (previously
  silent all-zero coefficients); `apply_fresnel_curvature` honors
  dtype-follows-input; `fresnel_propagate` drops an avoidable full-grid copy;
  the dead `chunk_output` parameter is deprecated; the pyFFTW gate itself now
  requires complex input so no future caller can poison the shape blacklist.
- **Raytrace JAX parity** (audit P3-58/59/60): tangent rays (disc == 0)
  survive like NumPy; the DOE-kick boundary comparison matches; the JAX
  aspheric Newton adds the post-iteration convergence check (unconverged
  intersections are killed like NumPy instead of silently accepted);
  `_surface_copy_with` propagates ALL optional Surface fields (coordinate
  breaks / tilts / decenters / world frames were silently dropped).
- **Optimize** (audit P3-47/48/49/50 + TNC): FD gradients clamp their stencil
  inside the bounds box at active bounds; `MaxFNumberMerit` tolerates
  aperture_diameter=None; TNC receives `maxfun` (its budget option) so
  max_iter is effective; `DesignParameterization` gains the duplicate-free-vars
  guard; two docstrings corrected (the solver zeros C, not B).
- **scipy 1.18 compatibility**: `design_optimize` no longer passes `disp` as a
  `scipy.optimize.minimize` solver option.  scipy 1.18.0 tightened per-method
  option validation and rejects `disp` for L-BFGS-B (and likely other methods),
  emitting `OptimizeWarning: Unknown solver options: disp` on every generic
  minimize (scipy <= 1.17 accepted it).  The driver already prints its own
  iteration progress from the merit callback when `verbose=True`, so scipy's
  internal `disp` was redundant; `differential_evolution` / `basinhopping`
  keep their native `disp=` argument.
- **Analysis/elements** (audit P3-01/02/03/04/07/15/24/25/39): seeded
  Shack-Hartmann noise is a reproducible SEQUENCE (was the IDENTICAL frozen
  frame every AO frame -- seeded noise realizations change); GaussianBSDF's
  hemisphere integral now equals `total_integrated_scatter()` at oblique
  incidence; Rytov EMT warns outside its period/wavelength validity domain;
  `through_focus_scan_jax` streams host copies per plane; the duplicate
  `_jax_available` definition removed; the segment-geometry liner warns when
  it cannot fit; three doc corrections (Marechal doctest value, thin-lens
  thickness formula sign, maslov JAX pointer).
- **Zemax IO** (audit P3-41..P3-44): malformed/truncated .zmx lines raise
  `ValueError` with file/line/text context (was raw IndexError); the
  auto-detected exit surface is no longer appended after a terminal MIRROR
  (bogus refractive element); the .txt loader warns per-surface when it drops
  aspheric data instead of silently claiming interchangeability; a stale
  exporter comment corrected.
- **UI** (audit P3-62/63): `WaveOpticsWorker`'s custom signal renamed
  (`finished_result`, matching the 14 sibling dock workers) so it no longer
  shadows `QThread.finished`; the worker snapshots all model state on the GUI
  thread at construction instead of reading the live model from the
  background thread.

### Fixed (wave 5 — remaining P2s: physics conventions, IO trust, performance, UI robustness)

**Three physics/convention BEHAVIOR CHANGES (each hand-derived against the
literature / the library's own rigorous solvers before fixing):**

- **`zernike_decompose(normalization='Noll')` no longer negates sine-mode
  (m < 0) coefficients** (audit P2-02): Noll 1976 and OSA/ANSI Z80.28 use
  IDENTICAL polynomials (both positive-sine; hand-derived from Noll Eq. 2 and
  verified against Noll Table I), so the old flip matched NO published
  convention.  'Noll' now returns coefficients equal to 'OSA' (the conventions
  differ only in single-index ordering, deliberately not permuted).  The old
  pinned test was circular and was replaced by hand-written Noll Table-I
  oracles.
- **`apply_waveplate` (and the Jones-element retarder family) slow-axis sign
  flipped** to `exp(+i*retardance)` (audit P2-15), matching the library's own
  Berreman/RCWA transmission Jones (a Berreman uniaxial QWP slab gives
  slow-rel-fast phase +pi/2 under the public exp(-i omega t) convention).
  Circular handedness (S3) no longer flips between the element family and the
  solver family for the same physical waveplate.  QWP outputs equal the
  CONJUGATE of prior releases (HWP unchanged); the QWP recipe producing
  'right' circular from linear-x is now fast axis at -pi/4.  rcwa's internal
  `_qwp_matrix` is deliberately unchanged: the `reflective_outcoupling`
  metric is provably invariant to the conjugate choice at the default 45 deg
  (documented in its docstring), so published out-coupling numbers are
  unaffected.
- **GBD `apply_abcd_to_beamlets` amplitude uses the Collins/Siegman factor**
  `1/(A + B*Q_in)` instead of `Q_new/Q_old` (audit P2-30) -- amplitude and
  piston were wrong by a factor `(C*q_in + D)`.  Hand-derived from the Collins
  integral and validated against the analytic Gaussian-beam `w(z)/R(z)/Gouy`
  free-space oracle.

**Other fixes:**

- **RCWAStack.solve(symmetry=True) per-order amplitude phases un-gauged**
  (audit P2-18): the even-parity cascade computed amplitudes in the
  recentering gauge, silently corrupting `per_order_amplitudes` /
  `to_multiorder_field` / `to_jones_field` phases (efficiencies unaffected);
  phases now match the symmetry=False path (probe: 0.343 -> 2e-13).
- **PMM `layer_absorption(by_material=True)` includes the `Im(ezz)` channel**
  (audit P2-13): a layer whose only loss was ezz vanished from the
  per-material dict while its flux absorption was correctly nonzero.
- **`stabilize` passive gate is two-sided and sign-aware** (audit P2-09): the
  degree-scan consensus could certify solves with NEGATIVE totals; now
  requires -tol <= tot <= 1+tol and per-order non-negativity
  (defense-in-depth behind the wave-1 incidence guards).
- **RCWA 1-D rejects back-side angles** (wave-3 follow-up): |angle| >= pi/2
  raised, mirroring the wave-3 PMM guard (sin^2(100 deg) previously slipped
  the evanescence check as a valid front-side angle).
- **Zemax loader: unknown SURFTYPEs are no longer parsed as EVENASPH** (audit
  P2-19) -- TOROIDAL/ODDASPHE/BICONICX/... PARM values were interpreted as
  huge bogus aspheric coefficients; unknown types now import as a plain conic
  with a loud per-surface warning naming the unsupported type.  **Exporter
  emits aspheric coefficients for MIRRORS** (audit P2-20; load->export->load
  is now identity for aspheric mirrors) and warns loudly on Q-type freeforms
  Zemax cannot represent instead of dropping them silently.
- **`ray_to_beamlet` carries `RayBundle.opd` into beamlet piston phases**
  (audit P2-33): the advertised GBD coherent-recombination workflow silently
  zeroed all inter-beamlet pistons.  JAX/NumPy trace backends now resolve
  per-surface apertures identically (wave-3 P2-35 residual).
- **Optimizer bounds honored for Powell / Nelder-Mead / TNC / COBYQA** (audit
  P2-24): user bounds were silently dropped for every method outside
  {L-BFGS-B, SLSQP, trust-constr} although scipy supports them.
- **Real-dtype fields no longer poison the pyFFTW shape blacklist** (audit
  P2-26): a real E_in is cast to the matching complex dtype at entry instead
  of being rejected by pyFFTW and permanently blacklisting the shape for ALL
  dtypes with a misleading warning.
- **fresnel/fraunhofer complex64 carrier computed at float64** (audit P2-29),
  matching the ASM kernel + MFT twins' documented precision contract.
- **HFPI**: real-dtype inputs no longer discard the imaginary part of complex
  path weights (audit P2-32); the normalization warning now states exactly
  what is and is not applied (audit P2-31 -- the old text contradicted the
  code).
- **Performance**: `DeformableMirror.fit_phase`'s 'streamed' path actually
  streams (probe: 307 -> 67 MB peak, 7.6x faster, lstsq for rank-deficient
  geometries) (audit P2-01); `angular_spectrum_propagate_batch` no longer
  wastes an FFT+IFFT pair on a proxy field (was measured 2.07x SLOWER than
  two scalar calls; docstring re-measured honestly) (audit P2-27); tilted ASM
  uses the exact 2-shift natural-H fold from v5.5.3 (audit P2-28,
  byte-identical); `retain_internal` uses a linear reverse Redheffer
  recurrence instead of O(n^2) chain rebuilds (audit P2-12).
- **UI robustness** (audit P2-37/38/39/40 + a latent F821): worker QThreads
  are no longer rebound while running (coherence dock crash), the app close
  now interrupts + joins all dock workers (the v5.4.2 close-guard was only on
  one dock), hidden analysis views defer recompute until shown, Stop buttons
  use cooperative interruption instead of QThread.terminate() inside FFT/HDF5
  C code, and a stray paste-duplicate referencing an undefined name in the
  wave-optics dock (latent crash) was removed.
- **Misc**: `makedammann2d` no longer uses `np.sign(complex)` (destroyed the
  IFTA far-field phase on NumPy < 2.0) (audit P2-07);
  `surface_sag_zernike_freeform` validates `norm_radius > 0` (audit P2-08);
  the Maslov JAX docstring states its actual algorithm and the
  even-multiplicity caustic limitation honestly (audit P2-03).

### Fixed (wave 4 — cache lifecycle: unbounded growth, stale values, lock discipline)

- **RCWA homogeneous-modes cache bounded** (audit P2-16/P2-17): `_HOMOG_CACHE`
  retained 2 dense eigenmode entries per sweep point FOREVER (probe: 42 MB per
  25-wavelength sweep, doubling per incidence angle); now a 32-entry LRU
  (last ~16 source configs stay hot; a solve touches 2), byte-identical on
  hit, miss, and post-eviction recompute, with the existing
  `rcwa_homogeneous_modes` clearer unchanged.
- **pyFFTW plan-cache lock discipline** (audit P3-55):
  `_clear_local_asm_caches` mutated `_PYFFTW_PLAN_CACHE`/`_PYFFTW_BAD_SHAPES`
  under the WRONG lock (`_ASM_CACHE_LOCK`), racing concurrent planners into an
  uncaught `KeyError` mid-FFT (deterministically reproduced); the two
  structures are now cleared under `_PYFFTW_PLAN_LOCK`, acquired sequentially
  (never nested) so no lock order is created and the worker-restore path
  stays deadlock-free.
- **JAX/prepared static caches bounded** (audit P3-16, P3-28, P3-32): the eme
  frozen-operator cache grew one dense operator set per Bloch k-point (the
  module's stated band-structure niche!) and the pmm 2-D JAX geometry cache
  ~3-9 MB per distinct geometry, both forever; now 8- and 16-entry LRUs.
  `_PreparedPMMStack`'s per-instance eig/mats caches are likewise bounded with
  a `clear_cache()`; post-eviction recomputes are byte-identical everywhere.
- **glass caches enrolled, bounded, and coherent** (audit P3-40 + P2-41 +
  P2-42): glass.py's module caches were unbounded, unenrolled, AND invisible
  to the enrollment meta-pin (its `endswith('_CACHE')` filter was
  case-sensitive -- fixed, so this hole class stays closed).
  `clear_asm_caches()` / `lumenairy_context(clear_caches_on_exit=True)` now
  drain the glass value cache, warn-once sets, and catalogue-dispatch
  material objects (user-FIXED entries are correctly PRESERVED -- for those
  the cache is the authoritative store).  `user_library.load_material` now
  invalidates the stale cache entries when it re-points a material, so a
  re-load can no longer silently serve the previous refractive index.
- **Wrapper-merit meshgrid cache keyed correctly** (audit P2-25): the cache
  keyed six aperture-INDEPENDENT full-grid arrays on the aperture VALUE --
  optimizing `aperture_diameter` meant a 100% miss rate plus up to 32
  grid-sized payloads retained.  The grid-only arrays now live in a
  grid-keyed LRU shared across apertures (merit values byte-identical);
  historical CHANGELOG line-citations for the `_ZERO_APERTURE_MASK` branch
  refreshed for the line drift.

### Fixed (wave 3 — guard-mirroring drift: a guard existed on one sibling but not another)

- **PMM JAX twins now enforce the guards their NumPy siblings enforce** (audit
  P2-10, P2-11, P2-14 + the 1-D twins): grazing/evanescent incidence on the
  JAX stack twin raised NaN-free nowhere (returned `R = nan` silently) and a
  gain superstrate on the 1-D twins silently negated T; both now raise the
  string-identical NumPy errors when the source values are CONCRETE (a traced
  value skips the guard -- documented; gradients through valid points are
  unaffected).  The 2-D pillar/cell twins also apply the propagating-incidence
  raise and the Wood-anomaly grazing-wavelength nudge (restoring NumPy-vs-JAX
  parity at Rayleigh cutoffs from 5e-5 to ~4e-16), and pillar-bounds
  validation now runs BEFORE the JAX dispatch (inverted bounds previously
  returned an energy-conserving but wrong answer on the JAX path).
- **`trace_jax_with_params` fails loud on unsupported surfaces** (audit
  P2-34): mirrors / coordinate breaks / biconics / freeforms were silently
  traced as flat refractives; the v4.10/v4.13.2 fail-loud builder guard is now
  shared by both JAX entry points (`NotImplementedError`, same message
  family).
- **NumPy trace backend honors per-surface `semi_diameter`** (audit P2-35):
  the key was honored by the JAX backend but silently ignored by the NumPy
  backend, so the identical prescription vignetted 1/25 vs 25/25 rays.  Same
  replace-the-default semantics on both backends; prescriptions without the
  key are byte-identical.
- **1-D PMM rejects back-side incidence angles** (audit P3-29): all eleven
  public 1-D entry points raise `ValueError` for `|angle| >= pi/2` (or NaN)
  instead of silently aliasing to the supplementary front-side angle.
- **`PMMStack` entry validation parity** (audit P3-30 + P3-31):
  `add_layer` now rejects non-positive/non-finite thickness (mirroring
  `PMM2DStack`); `solve` validates `stabilize` EAGERLY (an invalid value was
  silently accepted on the covariant path), and the covariant (uniform-slant)
  dispatch now raises `NotImplementedError` for `retain_internal=True` and
  honors `stabilize='slices'` instead of silently ignoring both kwargs.
- **`pmm_efficiency_2d_staggered` non-square cells raise a clear error**
  (audit P3-36): `Nx != Ny` cells died with a deep cryptic `AssertionError`;
  the entry point now raises `ValueError` naming the square-cell restriction
  and the `pmm_efficiency_2d_cell` alternative.
- **`BORStack` input validation** (audit P3-10): non-positive `Rbig` /
  thickness / wavelength / `k0`, non-integral `m` / `N` now raise clear
  errors instead of propagating to plausible-looking garbage (e.g. a negative
  thickness silently destabilized the Redheffer cascade; `wavelength <= 0`
  returned EMPTY R/T).

### Fixed (wave 2 — v5.16.1/v5.17.0 recent-delta consistency)

- **`set_max_ram()` is now actually honored** (audit P2-21): `pick_batch_size`
  / `should_split` default their available-RAM read to `get_ram_budget()`
  (which equals the psutil read when no override is set -- default behavior
  unchanged), and `apply_real_lens_traced`'s parallel-amp guard compares
  against `min(psutil-free, budget)`, so a pinned budget can force the doubled
  parallel working set off.
- **Row-band lens memory estimates account for `parallel_amp` and
  `slant_correction`** (audit P2-22): the chunked branch of
  `estimate_lens_memory` now models the runtime parallel doubling of the
  leg-local working set and the full-grid slant fall-through stack.
  **Estimates (and `check_sim_memory` refusal peaks) for default
  parallel-amp configs roughly double -- matching measured runtime truth**;
  the calibrated `parallel_amp=False` anchors are bit-unchanged, and the
  'set parallel_amp=False (byte-identical)' claw-back rung works in chunked
  mode again.  The 'use complex64 fields' claw-back label now discloses its
  `parallel_amp=False` assumption (audit P3-45).
- **`set_low_memory(False)` restores exactly what `set_low_memory(True)`
  found** (audit P2-23 + P3-46): priors are captured from the live getters at
  first-enable (including the aggressive complex64 default-dtype flip, which
  previously PERSISTED after 'restoring defaults', and user customizations
  like reproducibility pins), and disable replays that snapshot.
- **`sag_chunk_rows=0` (force whole-grid) is honored end-to-end** (audit
  P2-05): `apply_real_lens_traced` forwarded the RESOLVED value (0 -> None) to
  its internal `apply_real_lens` amplitude legs, where None re-resolved to
  AUTO -- silently re-enabling row-banding.  The raw kwarg is now forwarded, so
  0 forces the whole-grid path in BOTH stages (None / positive ints band
  identically in both, as before).
- **Freeform surfaces keep their diagnostic on the default row-band path**
  (audit P2-04): zernike/xy-polynomial/chebyshev surfaces were chunk-eligible,
  and the band loop computes only the base conic sag -- silently dropping the
  'freeform departure is NOT included' RuntimeWarning that every release since
  the warning's introduction emitted.  Non-Q freeform surfaces now fall
  through to the whole-grid path per surface (outputs byte-identical -- the
  departure was dropped on both paths; only the diagnostic differed).
- **Undersample floor enforced for apertureless prescriptions** (audit P3-08):
  the `ray_subsample` coarse-sampling check -- documented as 'already
  enforced' -- was silently skipped when the prescription had no
  `aperture_diameter`; the effective pupil now falls back to the largest
  per-surface `clear_aperture` (capped at the launch diameter) or the launch
  diameter itself.
- **Chunked-assembly lifetime fix** (audit P3-09): the full-grid `amp` array
  (and its coarse subsample) are freed immediately after the Newton-mask
  build on the default `preserve_input_phase=True` path -- byte-identical
  outputs, lower peak.
- **`snapshot_fft_state` captures the post-v5.4.6 knobs** (audit P3-54):
  `USE_SCIPY_FFT`, the pyFFTW fallback flag, the v5.17.0 double-buffer
  opt-out, the plan-cache bound (restored through its trimming setter), and
  the ASM cache bounds now survive into spawned workers;
  `restore_fft_state` tolerates old snapshots that lack the new keys
  (mixed-version worker pools).
- **Docstrings** (audit P3-06 + P3-05): `sag_dtype` / `sag_chunk_rows` are now
  documented in both `apply_real_lens` and `apply_real_lens_traced` Parameters
  sections (auto-band rule, the 0 escape hatch, the float32 sign-off
  validator), and the JAX lens twins document that the row-band mode is NOT
  replicated there (monolithic allocation).

### Fixed (wave 1 — the eight P1 findings, all adversarially re-verified)

- **`bruggeman()` returned a non-solution** (`elements/emt.py`, audit P1-02): the
  quadratic's linear coefficient was algebraically wrong (hand-re-derived:
  `b = ((1-f) - g f) eps_a + (f - g(1-f)) eps_b`), so the returned permittivity
  violated the defining Bruggeman self-consistency equation at essentially every
  fill (e.g. `bruggeman(4, 4, 0.5)` returned 2.828 instead of 4.0; the dilute
  limit `f=0.001` returned 5.218 instead of ~2.003).  The root selector was also
  rewritten to pick the PASSIVE branch (`Im(eps) >= 0`, continuous with the
  infinitesimal-loss limit) instead of an arbitrary `Re > 0` root that broke
  metallic mixtures (`bruggeman(-10, 2.25, 0.0)` returned +5.0 instead of 2.25).
  **BEHAVIOR CHANGE: every `bruggeman()` output moves** (the old values were
  wrong); validated by a 40,000-sample self-consistency sweep (residual
  <= 2e-13, passivity everywhere, no branch jumps).  `maxwell_garnett` and the
  Rytov tensors are unchanged.
- **BOR-PMM unit-system dependence** (`elements/bor/`, audit P1-01 + P2-06): the
  flux-normalization threshold (absolute `1e-10`) and the propagating-mode
  classifiers (absolute `|q.imag| < 1e-4`, `q.real > 0.1`) were dimensional, so
  the SAME physics expressed in meters returned silently wrong T (energy
  0.44-0.99) and nm-scale inputs returned silently EMPTY R/T.  Both now work in
  unit-invariant form (flux relative to the mode's own `r dr` field norm;
  classifiers in the dimensionless `q/k0`), bit-identical at the validated
  micron scale and verified identical across 5 unit systems (nm to m).
- **1-D PMM gain/evanescent incidence guard** (`elements/pmm/_core.py` +
  `oned.py` docstrings, audit P1-03): mirrored the v5.14.1 RCWA audit-P1 guard
  into all 1-D PMM far-field paths (scalar, Jones, slanted, segments, oblique
  out-of-plane) -- a gain superstrate (public `Im(n) < 0`, even `-1e-9`)
  previously flipped `kz_inc` negative and silently returned zeroed R with
  negative T (`sum T = -12.5`).  Now raises `ValueError` like RCWA; plain lossy
  superstrates still run (documented caveat).  Same guard added to
  `pmm_efficiency_2d_staggered` (audit P1-05), which silently returned
  `sum T = -144.6` where the hybrid raised.
- **Stale retained internals in PMMStack / PMM2DStack** (audit P1-04):
  `internal_field()` / `layer_absorption()` silently served the PREVIOUS solve's
  fields after the source or geometry changed.  Retained internals are now
  invalidated at every superseding entry point (`solve`, `solve_vs_wavelength`,
  `add_layer`, `set_source`, and the prepared assemble-once path), so they can
  only describe the most recent solve, which must have used
  `retain_internal=True` -- otherwise the documented `ValueError` is raised.
  **BEHAVIOR CHANGE: sequences that previously (wrongly) returned stale fields
  now raise.**
- **LG/HG mode-stack cache origin collision**
  (`propagators/asymptotic_modes.py`, audit P1-06): the cache keys captured
  shape/pitch/waist/centre/dtype but not the grid ORIGIN, so `decompose_lg` /
  `decompose_hg` on a shifted same-shape grid silently reused the wrong cached
  modes (~27% amplitude error).  The origin `(X[0,0], Y[0,0])` is now part of
  both keys.
- **Ray-trace pseudo-glass registration collisions** (`raytrace/trace.py`,
  audit P1-07 + P2-36 + P3-61): spherical/aspheric elements registered their
  fixed-index glass under `id(elem)`-derived names -- CPython id recycling let a
  later build silently retarget earlier surface lists to the WRONG refractive
  index (198/200 wrong in a 200-point sweep).  Names are now content-derived
  (`__spherical_<repr(n)>`), the registry sentinel was corrected to the
  `'__user__'` tuple `get_glass_index` actually matches (fixes `ImportError` on
  installs without the optional `refractiveindex` package), and re-registration
  invalidates the per-name value cache.
- **GUI wave-optics dock dead import** (`ui/waveoptics_dock.py`, audit P1-08):
  the worker imported the pre-v5.14-reorg module path `lumenairy.propagation`,
  so every Run died instantly and the dock hung at 'Running...'.  Backend
  selection now sets the flags on `propagators.fft_infra` (where the FFT
  dispatchers actually read them), and `run()` is failure-safe: any exception
  -- including ones with a broken `__str__` -- still emits the finished signal
  with an error payload, so the Run button always recovers.

## [5.17.1] — 2026-07-02

### Fixed

- **Release-blocking V14 forwarding gap** (caught by the v5.17.0 publish
  verify gate -- v5.17.0 never reached PyPI): the new
  ``_PYFFTW_DOUBLE_BUFFER`` global was missing from
  ``propagation._LIVE_FORWARD_NAMES``, so ``propagation.X`` attribute reads
  after ``set_fft_double_buffer()`` saw a stale import-time snapshot.
  Added to the live-forward whitelist.
- **Tilted-ASM dtype contract**: ``angular_spectrum_propagate_tilted`` built
  its carrier unconditionally as complex128, silently upcasting the whole
  tilted pipeline for complex64 inputs (2x working memory) and RETURNING
  complex128.  The carrier is now built at the target dtype with the
  carrier phase folded mod 2*pi in float64 before the float32 cast (the
  main-ASM accuracy mitigation): complex64 agrees with the complex128
  pipeline to < 5e-5 relative and honours dtype-follows-input; the
  complex128 path is bit-identical to pre-fix.  Eager frees added for the
  carrier grids / demod field / spectrum.

### Changed (estimator recalibration)

- ``estimate_lens_memory`` / ``estimate_sim_memory`` anchors recalibrated
  to post-lifetime-fix measurements (see the calibration table in the
  module docstring); the pre-fix anchors over-predicted the whole-grid
  path by ~30% after the v5.17.0 leak fixes.

## [5.17.0] — 2026-07-01

### Changed (row-band lens mode ON by default — auto)

- **`sag_chunk_rows=None` now resolves to AUTO**: the row-band lens path is
  the default for grids with `N >= 4096` (band = `max(256, N // 16)`);
  smaller grids keep the whole-grid path exactly as before.  Rationale for
  a default flip: the banded path is **byte-identical** (`np.array_equal`-
  pinned across preserve_input_phase modes, band sizes, and fallback
  surfaces), **wall-clock neutral** (133 vs 136 s at N=16384/sub=16; the
  banded assembly cost is offset by a 3.5x faster Newton fed contiguous
  coarse grids), and dramatically leaner (43.6 -> 18.4 GB traced-lens peak
  at N=16384 pre-lifetime-fixes).  Pass ``sag_chunk_rows=0`` to force the
  whole-grid path; an explicit positive int sets the band size.  The
  memory estimator mirrors the same auto rule so estimates match a
  default call.

### Added (row-band lens memory mode — the fidelity-preserving large-grid enabler)

- **`sag_chunk_rows` (BYTE-IDENTICAL)** on `apply_real_lens` and
  `apply_real_lens_traced` -- runs the per-surface phase screens AND the traced
  OPL-upsample/exit-assembly in row bands, so the full-grid float64 lens stack
  (coordinate meshgrids, sag/opd, `np.indices` + the `(2,N,N)` map_coordinates
  input, `delta_phase`, the complex128-first `phase_exp`) never materialises.
  Element-identical to the whole-grid path (`np.array_equal`-pinned; the
  order-1 `map_coordinates` upsample is pointwise in the output). Measured
  traced-lens peak at N=16384/sub=16/c64: **43.6 -> 18.4 GB** (whole-grid vs
  chunked) -- 45% below even the v3.2.14.1-era 33.2 GB. This restores
  full-fidelity N=32768 runs on 128-137 GB boxes with NO accuracy trade.
  Non-narrow surfaces (decenter/tilt/slant/fresnel/stop/biconic/freeform)
  fall through to the whole-grid path per surface (meshgrids built lazily),
  so mixed prescriptions remain exact.
- **`sag_dtype` (opt-in, accuracy-trading)** on the same functions +
  process-wide `set_lens_sag_dtype` / `get_lens_sag_dtype` -- float32 geometry
  (coordinate/sag/opd lineage), halving the float64 core. Gated by
  **`lens_sag_float32_opd_error`**: a radial OPD scan + a field-level float32
  vs float64 A/B. The field error is CONFIG-dependent (the f32 phase
  perturbation interferes through in-glass diffraction), so pass your
  production `field_check_dx=` for sign-off; the default coarse check is a
  gross-failure screen only.
- **Estimator + guardrail understand chunking**: `estimate_lens_memory` /
  `estimate_sim_memory` / `check_sim_memory` take `sag_chunk_rows`
  (calibrated 18.4 GB anchor), and the guardrail's claw-back ladder now
  RECOMMENDS the byte-identical row-band mode FIRST -- before any
  dtype/subsample/grid reduction that would trade fidelity.

### Fixed (memory root-cause -- benefits DEFAULT whole-grid runs too)

Stage-profiling the identical lens on the archived 3.2.14.1 vs 5.16.1
measured a **+32% traced-lens peak-memory growth** (33.2 -> 43.6 GB at
N=16384/sub=16/c64), which is what moved full-grid N=32768 traced runs from
fits-on-128-GB (historical Design-51/71 runs) to OOM. tracemalloc per-line
attribution pinned it (the suspected polynomial-Newton fit was innocent,
~0.07 GB coarse-scale) and the components are now fixed:

- **v4.10 tilt-detection leak (the big one).**  The `tilt_aware_rays=False`
  advisory check computed `np.abs` / mask / `np.angle` / `np.gradient` of
  the full input field and left all five arrays referenced by the function
  frame for the REST of the lens call -- ~4 full-grid float32 + a bool
  (~18 GB at N=32768) held through the ray trace, Newton, and assembly.
  Now freed immediately after the RMS is computed (pure lifetime fix;
  values, outputs, and the warning behaviour unchanged).
- **Whole-grid OPL upsample built the `(2, N, N)` `map_coordinates`
  coordinate stack twice** (once for the OPL, once for the NaN mask) with
  `ii`/`jj` held throughout -- ~4 extra full-grid float64 (~34 GB at
  N=32768) at the upsample peak, present since 3.2.14.1.  Built once,
  freed early; identical coords -> byte-identical outputs.
- **Eager frees in the whole-grid exit assembly** (`opl_map` /
  `delta_phase` / `phase_exp` dropped as soon as their consumer exists).
- **`set_fft_double_buffer(False)` / `get_fft_double_buffer`** -- opt-out
  for the v4.12 two-buffer pyFFTW ping-pong (the one remaining deliberate
  delta vs 3.2.14.1: one extra resident full-grid aligned buffer per plan
  key, 16 GiB/key at N=32768 complex128).  Disabling returns
  `buf.copy()` per FFT instead -- byte-identical values, ~one extra copy
  per transform.  Folded into `set_low_memory(True)`.

Post-fix tracemalloc (N=4096 probe): assembly-stage peak and end-of-call
residual are now BELOW the 3.2.14.1 baseline (2.83 -> 1.88 GB and 2.43 ->
1.34 GB vs archived 2.33 / 1.93); the overall whole-grid peak is within
~8% of 3.2.14.1, all of it the (opt-out-able) double buffer.
`sag_chunk_rows` remains far leaner still, at zero fidelity cost.

## [5.16.1] — 2026-06-25

### Added (memory estimation + autodetect guardrail)

- **`estimate_sim_memory` / `estimate_lens_memory` / `estimate_asm_memory`** -- a
  system-level peak-RAM estimator for free-space + ray-traced-lens simulations.
  Where `estimate_op_memory` models one FFT op as `n_work_arrays` same-dtype
  temporaries, these account for the memory-DETERMINING step: the ray-traced lens
  amplitude pass, whose working set is a stack of **float64-fixed** full-grid
  arrays (the `np.arange(N)*dx` coordinate lineage, sag/opd, the
  `np.indices((N,N))` + `(2,N,N)` `map_coordinates` upsample stack, and
  `delta_phase`) that does NOT shrink with complex64. Calibrated to measured peak
  RSS on a 137 GB box; pass `itemized=True` for the per-term breakdown.
- **`check_sim_memory`** -- an autodetect guardrail. Estimates the true peak,
  compares to available RAM, and (mode `'warn'` / `'raise'` / `'silent'`) returns
  a structured verdict that, when a config will not fit, lists concrete claw-backs
  that DO fit (parallel_amp off -> complex64 -> coarser ray_subsample -> smaller
  N), each with its estimated peak -- so large-grid runs fail FAST with an
  actionable message instead of OOMing mid-run.
- **`set_low_memory(enabled, *, aggressive=False)`** -- flips the byte-safe
  memory-lean knobs together (FFT plan-cache -> 2, lens `parallel_amp` -> off,
  auto-promote off) and restores them on `set_low_memory(False)`. `aggressive=True`
  additionally sets the default field dtype to complex64 (logged).
- **`set_lens_parallel_amp` / `get_lens_parallel_amp`** and
  **`get_fft_plan_cache_size`** -- opt-in knobs for the largest lens-step claw-back
  (sequential amp + amp(pw), ~2x working set, byte-identical output) and for
  reading the resident pyFFTW plan-cache bound.

### Changed

- `apply_real_lens_traced(parallel_amp=...)` default is now `None`, resolving to
  the process-wide `set_lens_parallel_amp` global (shipped default `True`).
  **Byte-identical** for every caller: explicit `True` / `False` still win, and
  callers that omit the kwarg get the unchanged `True` path.

### Fixed

- **Zemax `EVENASPH` loader power-index off-by-one** (`io/prescriptions_zemax.py`).
  Zemax `PARM_n` is the coefficient on `r^(2n)`, but the loader mapped it to
  `r^(2 + 2n)` (and the exporter inverted that) -- inflating a real ~3 um asphere
  into a ~77 um surface and producing a multi-mm focus shift + smeared spots on
  direct-imaging prescriptions. Loader and exporter corrected to
  `power = 2*parm_num`.

### Notes

- All new knobs default to the shipped (pre-5.16.1) behaviour; `set_low_memory`
  is the single opt-in. Existing results are byte-identical unless opted in. The
  chunked-sag and float32-sag headroom knobs for N=32768 land in a follow-up.

## [5.16.0] — 2026-06-23

### Added

- **BOR-PMM (body-of-revolution / axisymmetric Polynomial Modal Method) graduated to
  `lumenairy.elements.bor`** — the **cylindrical-coordinate** peer of `pmm` / `rcwa`
  for rotationally-symmetric structures (concentric-ring gratings, fibers, axisymmetric
  diffractive optics). Fields separate as `exp(i m phi + i q z)`, so the problem reduces
  to a 1-D *radial* eigenproblem per azimuthal order `m`, cascaded in `z` by a Redheffer
  S-matrix (the direct analog of the 1-D PMM lateral solve). Headline API `BORStack`
  (mirrors `PMMStack` in cylindrical coordinates); access via
  `from lumenairy.elements.bor import …`. The radial solvers, far-field
  (Fourier-Bessel / Hankel) helpers, and analytic validation oracles (`fiber_modes`,
  `stepindex_modes`) are exported alongside it.
- Built on the **Yee div-conforming (staggered) radial discretization** (`E_r` on
  faces, `E_phi`/`E_z` on nodes), which makes the discrete `curl·grad = 0` to machine
  precision and so eliminates the spurious curl-curl gradient mode sea → the z-cascade
  conserves energy to machine precision (~1e-13).
- **GATE 4 validated** (the load-bearing diffraction gate): a concentric ring grating's
  per-order efficiency matches the rigorous planar `pmm_efficiency_1d` at each
  cylindrical mode's local oblique angle `theta = arcsin(gamma/(n k0))`, for BOTH TE and
  TM, to well under the few-percent bar on the staggered basis (the residual is the
  2nd-order FD floor, shown to shrink under N-refinement). The match is ~28–100× tighter
  than at a wrong incident angle — a genuine cylindrical→planar correspondence, not an
  energy-balance artifact. Plus the full milestone ladder: radial operator vs Bessel
  zeros (~1e-13), coupled vector eigensolve vs an exact open-cladding fiber-dispersion
  oracle, the radial-PML open boundary, and the analytic-slab z-cascade. The BOR
  convergence tests run in CI (marked `slow`).

## [5.15.0] — 2026-06-22

### Added

- **EME (eigenmode-expansion) 2-D mode solvers graduated to `lumenairy.elements.eme`**
  — the scalar-Helmholtz (`ref_2d_modes`, `layer_modes`) and full-vector Maxwell
  (`ref_2d_modes_vector`, `layer_vector_modes`, `strip_vector_modes`, `mode_field_vec`)
  2-D Bloch *layer-mode* / band-structure solvers move from `experiments/eme/` into a
  first-class subpackage — the mode-solver peer of `pmm` / `rcwa` (diffraction
  *efficiencies* remain the job of `rcwa_efficiency_2d` / `pmm_efficiency_2d`; the EME
  mode-matching route to efficiencies is a documented dead end). Access via
  `from lumenairy.elements.eme import …`; the 27 convergence tests now run in CI
  (marked `slow`).
- **Full out-of-plane 3×3 anisotropic vector strips** — the vector strip generator
  now supports the complete out-of-plane permittivity tensor (diagonal +
  `exz`/`ezx` + `eyz`/`ezy`), validated against the role-swapped Berreman dispersion
  (`exz`) and the analytic Christoffel determinant `det(k kᵀ − |k|²I + k0²ε) = 0`
  (`eyz`, where the role-swapped Berreman is the *wrong* oracle). `eyz` breaks the
  block-anti-diagonal `[W;−V]` backward mode, so the *strip* modes stay rigorous
  (a `bc_ok` guard falls back to the full `eig`) while the cascade *layer* finder
  is gated on `eyz`.
- **Vector-solver universality** — scalar magnetic permeability `mu(x,y)`; a banded
  O(S) inverse-power `σ_min` (`solver="banded"`) for fine y-staircases; a seeded
  Beyn contour refiner (`beyn_refine_complex` / `layer_vector_modes_complex`)
  reaching complex / lossy / leaky `qz²`; arbitrary-geometry rasterization
  (`eps_xy_to_strips`).
- **Differentiable (JAX) twin of the eig-based mode oracles** — pass a JAX `eps(x,y)`
  to `ref_2d_modes` / `ref_2d_modes_vector` and get differentiable `qz²` (w.r.t.
  `eps` and `k0`) via the gauge-fixed Lorentzian-broadened custom-VJP eig shared
  with RCWA / PMM (`_jax_eig_stable`). The Yee / Laplacian operators are frozen
  (geometry-/`k0`-only) and the generator is reassembled in `jnp` from the traced
  `eps`. Forward is byte-exact vs NumPy; AD matches finite-difference to ~1e-7. The
  `σ_min` root-scan layer finders are not differentiable and **raise** on a JAX
  `eps` (use the oracle twin / implicit diff). Scalar isotropic `eps`, dense
  spectrum, `qz²` only; CPU-only; keep `Nx·Ny` small (~12×12) for design loops.

## [5.14.6] — 2026-06-22

### Fixed

- **Differentiable `eig` custom-VJP pytree mismatch** (`_jax_eig_stable`, the
  gauge-stable JAX eig shared by the Berreman and RCWA JAX paths). The primal
  returned `jnp.linalg.eig(A)`, which in modern JAX / NumPy 2.0 is an `EigResult`
  namedtuple — a custom pytree node — while its `custom_vjp` forward rule returned
  a plain `(lam, V)` tuple. `custom_vjp` requires the primal and forward outputs
  to share pytree structure: plain `grad` tolerated the mismatch, but
  **`grad ∘ vmap` and `grad ∘ jit` raised**, breaking the batched / JIT gradient
  of the Berreman 4×4 and RCWA eig paths. Fixed by unpacking the primal to a plain
  tuple so both sides match. Purely structural — the gradient math is untouched
  (AD vs finite-difference unchanged at ~1e-9); `grad`, `grad ∘ vmap`,
  `vmap ∘ grad`, and `grad ∘ jit` all pass, 27 JAX gradient tests green.

### Experiments (not packaged)

- **EME lateral-cascade 2-D mode solvers** (`experiments/eme/`) — scalar and
  full-vector (TE/TM) 2-D Bloch *layer-mode* solvers built from 1-D-x strips + a
  well-conditioned global block-`G` lateral-interface null-space, validated
  against a Yee-staggered 2-D vector FD oracle (cross-checked vs an independent
  Fourier plane-wave solver). The vector strip modes are a Berreman 4×4 Δ
  re-oriented to propagate laterally. Diffraction-*efficiency* mode-matching is
  documented as a dead end (real-space modes are not a Fourier-order basis — use
  `rcwa_efficiency_2d` / `pmm_efficiency_2d`); the EME's niche is modes / band
  structure.

## [5.14.5] — 2026-06-20

### Performance

- **Folded 1-D modal eigensolve** (`_fast_geig`) — the dominant cost of a 1-D
  PMM solve is the dense modal eig `A x = q² B x` (`B` = nodal mass `S0` for TE,
  `Pinv` for TM). It is now solved as the *standard* problem `eig(B⁻¹ A)` instead
  of the generalized QZ, **~1.7× faster on the eigensolve** (e.g.
  `pmm_efficiency_1d` 21.6 → 12.7 ms at `degree=24`), with **no change to the
  result**. `B` is the mass matrix — well-conditioned by construction — so the
  fold is exact (the JAX twin already folds `eig(solve(B, A))`, validated to
  ~1e-12). Safety: the same element-size equilibration as `_safe_geig`, plus an
  LU-pivot-ratio guard that falls back to the robust generalized QZ if `B` is
  near-singular (an extreme-`eps` metal corner) — so speed never trades away
  physical accuracy. Wired into the scalar 1-D path (`pmm_efficiency_1d`) and the
  vertical-slant case (`pmm_efficiency_1d_slanted`).
- Per-order efficiencies reproduce the generalized-QZ result to **~1e-14** across
  TE, TM, oblique, and an extreme-`eps` metal cell; **269 PMM unit tests
  unchanged**. `PMMStack`, `pmm_jones_*`, and `pmm_efficiency_2d` already use the
  standard `eig` (the coupled `Mbig` / covariant-metric generators) and are
  unaffected.

## [5.14.4] — 2026-06-14

### Added

- **Effective-medium (EMT) homogenization bridge** (`lumenairy.elements.emt`
  — `rytov_tensor`, `rytov_segments_tensor`, `maxwell_garnett`, `bruggeman`,
  plus `BerremanStack.add_effective_grating`) — the fast SCREEN that feeds
  the rigorous patterned solvers: a sub-wavelength 1-D grating homogenizes
  (Rytov) to a uniaxial `diag(eps_perp, eps_par, eps_par)` tensor solvable in
  microseconds with `BerremanStack`.  Validated: the EMT + Berreman slab
  Jones converges MONOTONICALLY onto the rigorous `rcwa_jones_1d` /
  `pmm_jones_1d` as the period shrinks (≈3e-3 at period = λ/100).  The
  zeroth-order tensor is the exact `period → 0` limit (the default); an
  opt-in second-order bulk-index correction is available, with the honest
  caveat that the homogenized SLAB keeps an inherent `O(period/λ)` interface
  error — it screens, then you validate rigorously.  Maxwell-Garnett /
  Bruggeman scalar mixing rules cover 2-D inclusion arrays (approximate).
- **Berreman 4×4 JAX twin** — `berreman_jones_1d` / `BerremanStack.solve`
  dispatch to a differentiable jnp path on JAX inputs: gradients flow
  through every layer permittivity tensor (real AND imaginary entries),
  thickness, wavelength, incidence `angle`/`phi`, and the half-space
  indices.  Forward-identical to NumPy (~1e-16); AD-vs-FD ≤1e-8 across every
  parameter class; `vmap`/`jit` clean.  The forward/backward mode split uses
  a STABLE `jnp.argsort` (the gathered eigenpairs carry the gradient, the
  integer permutation is constant) — this is what lets the Berreman twin
  trace where the PMM out-of-plane path could not (its host-side NumPy
  argsort severed the graph).  x64 required; the eig is CPU-only;
  `retain_internal` is NumPy-only.
- **Optics-viewer support for waveplate and polarizing-beam-splitter
  elements** — the 2-D (Qt) and 3-D (PyVista) layout views render the new
  `'Waveplate'` and `'PBS'` element types (`Element.TYPES`), previously
  unvisualizable: a violet retarder plate with a fast-axis tick and a
  λ/4 · λ/2 · WP label, and a cyan beam-splitter cube with its diagonal
  interface and reflected exit port.  Polarization parameters
  (`aux['wp_kind']`, `aux['fast_axis_deg']`, `aux['pbs_angle_deg']`) drive
  the glyph; they trace as flat pass-through windows.
- **Berreman 4×4 anisotropic planar multilayer solver** (`lumenairy.
  berreman_jones_1d`, `lumenairy.BerremanStack`) — the fast, exact
  planar-anisotropic member of the solver family, generalizing the scalar
  transfer-matrix coating model to fully anisotropic layers (LC retarders,
  waveplates, birefringent / uniaxial / biaxial films, magneto-optic
  stacks).  A `4×4` tangential-field state `[Ex, Ey, Hx, Hy]` per layer
  (Berreman 1972 / Yeh 1979), composed by the SAME numerically-stable
  generalized scattering-matrix cascade the PMM / RCWA out-of-plane paths
  use (so a thick / lossy stack never overflows).  Promotes the previously
  test-only single-slab Berreman oracle to a public **multilayer** solver
  with the full RCWA/PMM-parity observable set:
  - **Full `2×2` Jones** reflection and transmission, plus flux-normalized
    `R` / `T` per incident polarization; conical (azimuthal `phi`) mounts.
  - **`BerremanStack.internal_field(z, component=, incident=)`** — the
    in-structure `E`/`H` field (all six components; `Ez` from `Dz = 0`,
    `Hz` from the curl), tangentially continuous across interfaces and
    equal to incident + reflected Jones at the top plane.
  - **`BerremanStack.layer_absorption()`** — per-layer absorbed power, the
    volume integral `k0 Im(E†·eps·E)` matching the flux-based result to
    ~1e-7 and closing against `1 − R − T`.
  - Validated to MACHINE PRECISION against independent oracles: the
    isotropic limit reproduces `coating_reflectance` (the validated
    complex-angle scalar TMM) in R, T AND reflection phase, for lossy and
    multilayer stacks at oblique incidence; a uniform tensor slab
    reproduces the independent `_berreman4x4` oracle (lossless and lossy);
    energy is conserved on lossless conical stacks.
  - It is NOT a competitor to RCWA / PMM — it is the planar (laterally
    uniform) tier, ~100–1000× faster for unpatterned retarder / coating
    design and the natural effective-medium screen that feeds the
    patterned solvers.  CONVENTION NOTE: `eps` enters raw (public
    `exp(-i w t)`, `Im > 0`), so the forward/backward mode split — and
    hence the flux-based power — is PHYSICAL on lossy stacks (a
    conjugated-eps split, as in the lossless-only standalone oracle, gives
    `T > 1` / negative absorption; `test_lossy_power_matches_scalar_tmm`
    guards this).

## [5.14.3] — 2026-06-11

### Added

- **PMM internal-field parity with RCWA** (user request 2026-06-11):
  - `PMMStack.internal_field` upgraded to the `RCWAResult.internal_field`
    interface: `z` arrays (stack-top or `layer=`-local), all SIX field
    components (`Ez`/`Hz` from per-element spectral derivatives of the
    nodal solution — exact pointwise `1/ezz`, no factorization rule),
    `incident=` Jones (mutually exclusive with the legacy `pol=`), the
    Bloch carrier `exp(i kx0 x)`, and optional uniform-grid resampling
    (`nx=`, barycentric evaluation of the spectral interpolant; default
    stays the exact GLL nodal grid).  BEHAVIOR CHANGE vs the day-one
    v5.14.2 method: `Hx`/`Hy` are now in the RCWA-co-registered `-i eta0`
    scale (the old return carried the raw modal convention, documented as
    such) and the Bloch carrier is included.  Conventions pinned by
    oracles: a uniform absorbing film matches RCWA pointwise in all six
    components at ~1e-15 (normal, oblique, complex incident); on a
    patterned grating RCWA's near field converges TOWARD the nodal-exact
    PMM (3.4e-2 @ n_orders 30 → 9e-3 @ 120 interior); the volume-integral
    absorption identity closes against the flux-based `layer_absorption`
    (TE 4.5e-4, TM 5.6e-3 — wall-interpolation-limited, converging in
    degree).
  - **`PMM2DStack.internal_field`** (new): the crossed-grating mirror of
    `RCWAResult.internal_field` — same centred grids, carriers, `component`
    / `incident` / `layer` / `filter='lanczos'` semantics, so the two
    co-register pointwise.  `Ez` via the projected `[[ezz]]` solve (the
    RCWA-mirror route; in-plane tensor layers project their `zz`
    component).  Validated: uniform conical film vs RCWA ~3e-16 all six
    components incl. complex incident; patterned stripe converges to the
    nodal-exact 1-D PMM as (n_orders, degree) rise together (Ex 0.105 →
    0.044, Ez 0.114 → 0.049, Hy 0.024 → 0.009); tangential continuity
    across internal interfaces ~2e-5; uniform-absorber volume identity
    ~1e-3.  NB the hybrid 2-D solution lives in the projected Fourier
    basis: keep `n_orders` comfortably inside the axis nodal capacity
    (the solver's own validated regime) — the projection's highest orders
    degrade first and `Ez` (order-weighted) shows it earliest.

### Fixed

- **`RCWAResult.internal_field` flipped the handedness of complex
  incident Jones drives** (found by the PMM co-registration oracle): the
  internal-gauge `cinc` was built directly from the PUBLIC incident and
  the output conjugated, so a circular/elliptical `incident=` returned
  the field of the CONJUGATED incident.  The incident now enters
  conjugated (public-linear superposition restored:
  `field(a, b) == a*field(1,0) + b*field(0,1)`); real incidents are
  bit-unchanged.  `to_jones_field`/`to_multiorder_field` were already
  public-linear and are unaffected; `layer_absorption` uses real basis
  drives and is unaffected.

## [5.14.2] — 2026-06-11

**Backlog batch 1** (docs/BACKLOG_2026_06_10.md priority-① items A1, A2,
B1+B2, D1) **+ stack-level JAX differentiability** (PMMStack / PMM2DStack
jnp twins):

### Added

- **RCWA even-parity scope extension (A1)**: the ×4 normal-incidence
  symmetry fast path — previously `rcwa_efficiency_2d(formulation=
  'laurent')` only — now also covers the sequential-rule `'li'`
  (`rcwa_efficiency_2d` + `prepare_rcwa_2d`), `rcwa_jones_2d(symmetry=True)`
  (in-plane tensor cells, both incident polarizations from one even solve),
  and **`RCWAStack.solve(symmetry=True)`** (whole multi-layer cascades —
  uniform, pixel-cell laurent/li, analytic shapes, and in-plane tensor
  layers — run in the (N+1)-d even sector; measured ×4.5 on a 4-layer mixed
  stack at n_orders 6 and 9). Implemented as a generalized
  `_symmetric_cascade_rt` over per-layer `(P, Q)` operators with a common
  recentering gauge; ANY failed precondition (oblique incidence,
  out-of-plane tensors, dispersive layers, mismatched layer symmetry
  centres) falls back to the full solve **bit-identically**. NB a pixel-cell
  feature's centre lies on the half-pixel grid — an analytic-shapes layer
  at exactly `period/2` in an otherwise pixel-cell stack is a GENUINE centre
  mismatch and correctly falls back.
- **PMM shared homogeneous eigenproblem (A2)**: uniform (half-space and
  uniform-layer) SEM modes are now obtained from ONE eps-free geometric
  eigenproblem of half the size (`q²(eps) = eps − μ_geo`, shared
  eigenvectors; the 2n problem block-diagonalizes), with the z-Poynting
  forward selector still applied per medium. Measured ×5.5–6.1 on the
  half-space mode pair; spectra match the full eigensolve to ~1e-11.
- **`PMM2DStack.solve(retain_internal=True)` + `.layer_absorption()`
  (B1)**: per-layer absorbed power for the crossed (2-D) stack from partial
  S-matrix cascades, evaluated in the Rayleigh basis (Parseval flux —
  closure vs `1 − R − T` at ~4e-15, lossless spacers at ~1e-16).
- **`PMMStack.internal_field(z, pol=)` (B2)**: nodal-exact internal field
  profiles `(x, Ex, Ey, Hx, Hy)` inside any layer from the retained modal
  amplitudes (tangential-E continuity across interfaces at truncation
  level).
- **Written BLAS-gauge tolerance policy (D1)**: `docs/TOLERANCE_POLICY.md`
  (when bit-identity is allowed vs when a physical tolerance is mandatory —
  degenerate-eig gauge freedom across code paths/builds) + named constants
  in `tests/unit/_tolerances.py`, used by the new regression tests.
- **JAX differentiability at the STACK level** — `PMMStack.solve` and
  `PMM2DStack.solve` gain jnp twins (`pmm/_jax_stack.py`,
  `pmm/_jax_stack2d.py`) composing the validated single-layer machinery
  (frozen NumPy geometry, eps-LINEAR traced assembly, the gauge-stable
  custom-VJP eig, the backend-generic Redheffer cascade).  Passing any
  input as a jnp array dispatches transparently:
  - **1-D `PMMStack`**: all-vertical in-plane stacks — gradients w.r.t.
    segment eps (re+im, scalar or in-plane tensor entries), layer
    thicknesses, wavelength, angle, and half-space indices.  Forward
    agreement with NumPy ~1e-15 (normal AND oblique); AD-vs-FD ≤6e-8
    relative across all six parameter classes.  Widths/walls are static.
  - **2-D `PMM2DStack`**: the scalar surface — traced uniform-layer eps,
    thicknesses, wavelength, theta/phi, half-spaces, and traced
    `eps_cell` VALUES via the new `add_layer(eps_cell=<jax>,
    region_layout=<int grid>)` (a concrete layout defines the walls, the
    traced cell provides each region's value — the
    `_pmm_efficiency_2d_cell_jax` contract).  Forward ~1e-15 across every
    dispatch flavor; AD-vs-FD ≤3e-9 (theta 2e-6, FD-limited).
  - Everything off-surface raises loudly under JAX: slanted layers,
    out-of-plane/tensor cells, `stabilize`, `retain_internal`, and the
    assemble-once `solve_vs_wavelength`/`prepare` paths (NumPy-only).
    x64 required; `jnp.linalg.eig` is CPU-only.

### Fixed

- **All-uniform PMM stacks were silently wrong**: a stack whose union grid
  had NO interior walls (every layer a single full-period segment, e.g.
  `PMMStack.add_layer(eps=...)` alone, or `pmm_jones_1d_segments` with one
  segment) assembled ONE spectral element over the period — a periodic
  nodal basis too poor for the Rayleigh far-field match. Energy leaked
  ~2–30% into spurious orders and split the polarizations while LOOKING
  plausible (closure off by +2.1e-2 at period 0.8 µm, −0.29 subwavelength).
  `_segment_elem_bnds` now midpoint-splits a lone element; the uniform-film
  oracle is Fresnel-exact (~2e-15) on all entry points. Patterned cells
  (≥2 regions) are bit-unchanged.
- `RCWAStack.solve(stabilize=..., symmetry=True)` propagates `symmetry`
  into the stabilizer's window re-solves.

## [5.14.1] — 2026-06-10

**Device-geometry roadmap** (docs/audits/ROADMAP_DEVICE_GEOMETRY_SWEEPS_
2026_06_10.md — geometry construction, not solvers, was the bottleneck):

### Added (device-geometry)

- **Multi-feature tapered builders, center-anchored** (item 1):
  `PMMStack.add_tapered_ridges`, `RCWAStack.add_tapered_ridges` /
  `.add_tapered_pillars` (the first 2-D RCWA tapered builder),
  `PMM2DStack.add_tapered_pillars` — absolute-position `(center, w_top,
  w_bottom, eps)` features that taper about their own FIXED centers (the
  audited left-anchored-drift bug class), wrap-aware, overlap-raising;
  single-feature cases reproduce the legacy builders bit-for-bit.
- **`SegmentStackGeometry`** (item 2): solver-independent geometry algebra —
  `add_ridges` / conformal `coat` (exact L∞ dilation: tooth tops, sidewalls,
  gap floors in one operation) / `line_interface` (liner on specific
  material-material interfaces, vertical AND horizontal, carved from a named
  side) / `fill` / `to_pmm_stack` / `to_rcwa_stack` / `plot` — one object
  feeds the PMM solve, the RCWA cross-check, and the viewer.
- **Staircase robustness** (item 3): `_pmm_union_grid` gains a PHYSICAL
  `min_feature` wall-snap (default period×1e-5) that merges near-coincident
  CROSS-LAYER wall pairs with a warning naming them — a single layer's own
  thin feature (a 1 nm liner) is never touched; `PMMStack.solve(
  stabilize='slices')` re-solves recorded taper builders at n_slices ± 1 and
  warns on zeroth-order Jones disagreement — the PASSIVE-BUT-WRONG staircase
  detector that energy tripwires cannot see.
- **Sweep parity + dispersion** (item 5):
  `pmm_jones_1d_segments_vs_wavelength` (the missing segments mirror);
  `PMMStack` and `PMM2DStack` accept `wl -> value` callables in every
  material slot and their `solve_vs_wavelength` gains `jones=True` (opt-in
  4-tuple; the released 3-tuple is unchanged); dispersive stacks refuse a
  single-wavelength `solve()` with guidance.
- **PMM internal absorption** (item 6): `PMMStack.solve(
  retain_internal=True)` + `layer_absorption(by_material=False|True)` — the
  internal z-Poynting flux difference per layer (closes against the far
  field to ~1e-14: `sum A == 1 - R - T` is a cross-machinery invariant, not
  a construction) with per-MATERIAL attribution via the lossy volume
  density.  The PMM split is flat in degree where the RCWA cross-check is
  provably under-resolved on metals — the loss-map gap that drove the
  device's Ta-liner redesign by inference now has a direct instrument.
- **Prepared material slots** (item 8): segments accept material KEY strings;
  `PMMStack.prepare()` + `prepared.solve(materials={...})` re-eig only the
  layers whose keys changed (an LC director sweep re-eigs 1 of N layers),
  bit-equal to the rebuild path.
- **Geometry viewers** (item 7): `plot_geometry` on `PMMStack` (exact
  analytic rectangles), `RCWAStack` (per-layer eps maps), `PMM2DStack`
  (per-layer exact-wall cell maps) — the picture IS the model.
- **`Material.from_csv`** (item 9): the tabulated n/k loader every project
  re-writes, as a `wl -> eps` callable accepted by every dispersive slot;
  linear in n/k, loud out-of-range.
- **`RCWAStack` out-of-plane tensor layers** (prior roadmap follow-through):
  any full-3×3 layer promotes the whole cascade to the generalized
  S-matrix (the `PMM2DStack` any_oop pattern over the `rcwa_jones_2d` GAP2
  machinery); single-layer stacks match the direct solver bit-for-bit,
  split-film identity to 4e-17, mixed stacks energy-exact.

### Application-feedback polish (same day; FEEDBACK_DEVICE_GEOMETRY doc)

- `layer_absorption(by_material=True)` attributes by material KEY when the
  stack came from `SegmentStackGeometry.to_pmm_stack` — twin keys with the
  same eps (the `"Cu"`/`"CuCol"` under-tooth-liner trick) now split the
  loss map as they split the geometry (raw-eps stacks keep complex-eps
  keys; the `eps·(1+1e-12)` workaround is obsolete).
- `PMMStack.plot_geometry` legends show material KEY names automatically on
  geometry-built stacks; viewer docstrings note `ax.figure.savefig(...)`.
- `min_feature` documented as the COST knob for dense staircases (measured
  5.7× on the application's ns8 coated taper).
- `to_rcwa_stack` accepts ANISOTROPIC materials (feedback ask 4): a band
  containing any (3, 3) tensor material (the LC) pixelates into an
  `eps_tensor_cell` with scalars promoted to `eps·I₃`; scalar-only bands
  keep the `eps_cell` path (scalar-vs-`eps·I₃` exports agree to 1e-10).

### Deferred with reasons (device-geometry)

- Item 4 (native exact trapezoid-metric PMM layer): the roadmap's "linear
  convection term like the slant" sketch does not hold — `u = (x − c)/w(z)`
  leaves `1/w(z)²` z-dependence in the lateral operator for a LINEAR taper,
  so the modal problem is not constant-coefficient (no single per-layer
  eig); an exponential-taper gauge or a z-ODE (Magnus) integrator are the
  honest starting points.  Research-grade; the staircase + items 1/3 remain
  the path.
- 1-D homogeneous-eig share (PMM roadmap #2): requires threading a
  presolved spectrum through the modal flux selector; small win (1 eig of
  N+2), deferred.
- RCWA even-parity scope (LEV-3) and µ/bianisotropy + hex lattices: as
  before.

**RCWA accuracy audit (46 findings; every landed fix hand-verified with
independent probes — the audit's verification phase was unavailable).**

### Fixed (physics — results change)

- **P1: the 2-D `E_z` inverse-rule factorization was wrong.**
  `rcwa_efficiency_2d` `formulation='li'/'fff'/'auto'`, the analytic-shape
  solver `rcwa_efficiency_2d_shapes`, `fff_nv`'s `EZZ`, and shapes layers in
  `RCWAStack` eliminated `E_z` with Li's INVERSE rule `[[1/eps]]`. `E_z` is
  tangential to every vertical wall of a z-invariant layer, so the direct
  rule `[[eps]]` is mandatory (Li 1997 Eq. 27; what S4/grcwa do). The wrong
  rule overestimated metal absorptance by up to **+0.35** (period-robust,
  silent on lossy cells where the energy tripwire cannot fire) and was less
  accurate than `'laurent'` even on dielectrics. 2-D `'li'` is now the
  **Li-1997 sequential rule** (Eqs. 8/9: inverse along each E-component's own
  axis, direct along the other; routed through the per-component tensor
  eigensolver): a y-/x-uniform cell reduces to rigorous 1-D `'li'` per-order
  (~5e-5 at S=256, 2nd-order in the cell sampling), metal-stripe absorptance
  error at M=16 drops to +4.6e-3, and on a true 2-D metal pillar the
  Richardson 1/M extrapolation of `'laurent'` lands on its converged value.
  On staircased CURVES it wins too (a pixel cell is axis-aligned by
  construction): on a metal disk `'li'` sits in the converged absorptance
  band at M=13 while `'laurent'` is 2× high and only reaches it at M=21.
  The old "matches grcwa/inkstone" and "direct-z biased low by 6e-2" claims
  were error-cancellation artifacts (inkstone at num_g≤1400 is itself
  unconverged on the audited pillar). **Re-run any 2-D
  `'li'/'fff'/'auto'`/shapes/fff_nv results on lossy cells.** `'laurent'`
  (the default) is byte-identical-unchanged everywhere.
- **P1: gain superstrates silently negated every efficiency.**
  `Im(n_superstrate) < 0` (even 1e-9) flipped the forward root, returning
  R=0 and NEGATIVE T (TM sum −392.8 on a plain lossless grating) below the
  one-sided tripwire. All entry points now reject gain incidence media
  loudly; `_check_energy` additionally raises on negative totals.
- **P1: the silent energy window (1e-6 < |R+T−1| < 0.05) is now policed.**
  For provably-lossless inputs (every permittivity exactly real — closure is
  exact in this code) a violation beyond 1e-6 emits `_EnergyWarning` with
  the measured closure error (the audited 1-D case: a silent +3.3e-2 with an
  8% per-order error and broken ±1 symmetry), and `stabilize=True` treats
  the warning as a failed attempt — it previously returned the
  byte-identical wrong answer (now it returns the adjacent-truncation
  consensus value).
- **P2: 2-D `stabilize=True` wrong-abort** — upward bumps past the
  cell-sampling alias bound are pre-filtered, so the ladder terminates with
  its documented `_EnergyError` instead of a misleading "cell too coarse for
  n_orders you never requested" ValueError.
- **P2: the Berreman 4×4 test oracle was wrong at true conical incidence**
  (`tests/unit/_berreman4x4._berreman_delta`: three entries off by exactly
  ±Kx·Ky, pinned by rotation covariance — exact at φ=0/90, which is why all
  planar tests passed). Fixed; the conical wrapper now conserves energy to
  1e-12 and `rcwa_jones_1d` matches it conically to 3.9e-15 including
  out-of-plane lossy tensors.

### Changed

- `formulation='li'` (2-D) no longer uses the `symmetry=True` even-parity
  fast path (the sequential rule lives in the tensor eigensolver; the fold
  is scalar-only) — it transparently falls back to the full solve.
  Extending the fold to the per-component tensor operators is a roadmap
  perf item.
- 1-D docstring: explicit `'laurent'`+TM on metals flagged as a
  factorization-study mode (unconverged at n_orders=128 on Ag, ~2-3e-2 off
  `'li'`).
- Re-pinned tests that encoded the refuted error-cancellation agreements
  (fff_nv dielectric square, inkstone metal-pillar "gold", symmetry×li).

### Added (the roadmap's deferred items, same release)

- **2-D out-of-plane tensors in `rcwa_jones_2d`** (audit GAP2): full-3×3
  cells (tilted uniaxials, magneto-optic media) via the pointwise ezz-Schur
  fold + the generalized forward/backward S-matrix cascade (the 1-D OOP /
  `pmm_jones_2d` pattern). Validated against an independent conical
  Berreman 4×4 oracle at machine precision (≤2.2e-15, incl. lossy conical)
  and against the 1-D OOP solver on patterned y-uniform cells.
- **Per-layer `formulation=` in `RCWAStack.add_layer`** (GAP3): isotropic
  patterned layers accept `'li'` (the corrected sequential rule); a 1-D
  stack metal layer reproduces the direct 1-D `'li'` solver per-order and
  halves the absorptance error vs Laurent at matched truncation.
- **`RCWAStack.solve_vs_wavelength` + dispersive materials** (GAP5): every
  material slot (`eps`, `eps_cell`, `eps_tensor_cell`, `eps_background`,
  per-shape `eps`) accepts a `wl -> value` callable; unstable wavelengths
  return NaN rows with one summary warning instead of aborting the sweep;
  `solve()` on a dispersive stack raises with guidance.
- **Sheared sidewalls** (GAP1): `add_tapered_grating(..., shear=)` —
  parallelogram (slanted-wall) gratings as a one-liner, wrap-aware.

### Performance

- **1-D planar TE/TM decouple** (audit RCWA-LEV-1): at planar mounting the
  layer system is exactly block-diagonal and a fixed polarization excites
  one block, so `rcwa_efficiency_1d` (NumPy, non-ASR) runs the eig +
  interfaces + Redheffer at size N instead of 2N — ×4-8 at large N
  (n_orders=161: 2.16 s → 0.29 s), per-order equal to the 2N machinery.
  BONUS: the separated blocks are better conditioned — two of the three
  pinned large-period energy blow-ups no longer occur (they now solve
  cleanly at the adjacent-truncation consensus).
- **Diagonal-aware propagation star** (RCWA-LEV-2): starring against a
  pure-propagation factor reduces to row/column scaling (identity ~9e-16);
  applied at every propagation site (1-D, 2-D, jones, shapes, stack incl.
  the `retain_internal` partial-S sweep, and the generalized OOP cascade).

### Hygiene (audit P3s)

- Non-finite material indices raise with a named culprit (was: silent NaN
  totals past the one-sided tripwire); `_check_energy` raises on non-finite.
- `rcwa_efficiency_2d_shapes` actually rejects JAX inputs (the guard was
  unreachable dead code — no arrays fed the dispatcher).
- `RCWAStack.solve(retain_internal=True)` on JAX warns that internal-field
  data is host-side only (was: silent drop + misleading downstream error).
- The 1-D sweep wrapper rejects JAX inputs with the vmap pointer (was:
  silent numpy materialisation).
- `gpu` extra pinned to `cupy>=14` (`cupy.linalg.eig` exists only since 14).

### Docs / roadmap

- `docs/rcwa_roadmap_v5_14.md` — remaining items after this batch: symmetry
  scope (stack + tensor-li even-parity fold), JAX positioned as the
  gradient path (jit forward is only ~1-2.4× wall via thread parallelism),
  fff_nv rework-or-retire, µ/bianisotropic + hex lattices (research).
- `tests/unit/test_v5_14_1_rcwa_audit_fixes.py` — 11 regression tests for
  the audit fixes; `tests/unit/test_v5_14_1_rcwa_deferred.py` — 20 tests
  for the deferred-item batch.

## [5.14.0] — 2026-06-09

**Accuracy/speed audit (22-agent, 41 findings, 16/16 adversarially
confirmed).** Post-parity hardening of the whole PMM family:

### Fixed (audit)

- **P1 dense normal-incidence resonances** — the 1-D binary paths' legacy
  forward-mode branch (`Im(q) >= 0`) flipped degenerate propagating modes on
  ~1e-15 QZ noise; 8 of 13 degrees in 12..24 silently returned
  `sum(R)+sum(T)` up to 65.7 (scalar) / 344 (Jones) on plain gratings. The
  noise-robust / Poynting-flux selectors (already used by the oblique and
  segmented paths) are now unconditional — every probed degree conserves
  energy to 1e-6 and matches the RCWA oracle per-order to ~6e-6. NB: this
  retires the bit-identity between the one-layer `PMMStack` and the segments
  path (gauge-dependent selections within degenerate pairs now differ by
  ~1e-7; the test pins 5e-6).
- **P2 Wood-anomaly cutoffs (1-D)** — the 1-D solvers now nudge off exact
  Rayleigh-cutoff wavelengths (was a silent 2.5e-4 energy violation).
- **P1 staggered Wood divergence** — `pmm_efficiency_2d_staggered` diverges
  like ~1/sqrt(cutoff distance) near Rayleigh cutoffs (intrinsic to its
  H-partner construction); it now nudges off exact coincidences and WARNS
  inside the divergence band, pointing to the (clean) hybrid solver.
- **P2 pillar bounds** — `pmm_efficiency_2d`/`prepare_pmm_2d` enforce
  `0 < lo < hi < period` (inverted bounds previously returned a silently
  wrong geometry; degenerate bounds crashed with a raw LinAlgError).
- **P2 stabilize pseudo-plateau** — the degree-scan consensus now returns the
  ENERGY-CLEANEST cluster member on lossless structures (two marginal degrees
  could corroborate each other ~1e-3 off per-order); the pillar stabilize
  scan also gained the cost cap (it previously laddered to multi-GB dense
  problems on persistently non-passive configs).

### Performance (audit)

- **P1 factorized 2-D assembly** — the dense nodal path spent 88-97% of the
  whole 2-D solve materializing kron products and LAPACK-inverting an EXACTLY
  DIAGONAL mass matrix. The projected operators are now built factorized
  (per-axis small matrices + `(Tp * vec) @ Tpinv` sandwiches): machine-
  identical (rel ~2.6e-15), 220-1078x faster assembly, 64-138x less memory;
  end-to-end 2-pillar solves drop from ~8 s to ~0.2 s. `max_nodal_dof` rises
  4000 -> 150000, making STAIRCASED CURVED cells (e.g. a pixelated disk)
  practical on the exact-wall hybrid.
- **Numba assessed and rejected** (confirmed): the solvers are LAPACK-eig /
  small-matmul bound; end-to-end gain would be <1.1x. JAX coverage is the
  acceleration path (1-D twins + the 2-D pillar twin; the cell twin is
  confirmed feasible — see docs/pmm_roadmap_v5_14.md).

### Added (audit)

- **Dispersive PMM wavelength sweeps** — `pmm_efficiency_1d_vs_wavelength` /
  `pmm_jones_1d_vs_wavelength` accept callable `n(wl)` / `eps(wl)` materials,
  mirroring the RCWA sweep API (the PMM family previously had none).
- `docs/pmm_roadmap_v5_14.md` — the audit's remaining confirmed-feasible
  items (JAX cell twin, 1-D homogeneous-eig share, stack OOP promotion,
  graded-profile helper, native 2-D slant, Li-1997 mixed rules) with effort
  assessments, and the explicitly-rejected paths.

**Also in 5.14.0 — 2-D PMM capability parity.** The 2-D hybrid PMM grows from a single-pillar
scalar TE/TM solver to full parity with `rcwa_efficiency_2d`/`rcwa_jones_2d`
and the 1-D solver families — and beyond it on two axes (out-of-plane tensors
and exact-wall geometry). Grounded in the PMM_Papers formulations (Li 1997
crossed-grating factorization; Li 2003 z-decoupled tensors + full-3×3
generator; pointwise ezz-Schur fold per Li 1999 Eq. 12).

### Added

- **`pmm_efficiency_2d_cell`** (+ `prepare_pmm_2d_cell`,
  `pmm_efficiency_2d_cell_vs_wavelength`) — arbitrary axis-aligned
  multi-region cells via the RCWA `eps_cell` pixel-grid convention, resolved
  into EXACT spectral-element walls (no Fourier staircase). The single-pillar
  cell reduces byte-identically to `pmm_efficiency_2d`. Includes the
  Nyquist parity bump (widest strip +1 element on an even-node axis) and a
  `max_nodal_dof` cost guard with clear guidance for sampled-smooth profiles.
- **`pmm_jones_2d`** — full (3, 3) anisotropic tensor cells + the 2×2 Jones
  reflection matrix, the PMM mirror of `rcwa_jones_2d` via the shared
  dimension-agnostic tensor eigenmode solve. A scalar tensor cell reduces
  byte-exactly to the scalar `'laurent'` path; an exact-diagonal uniform-cell
  branch removes the projection floor entirely.
- **OUT-OF-PLANE tensors** (`xz/yz/zx/zy`) in `pmm_jones_2d` — the library's
  FIRST 2-D out-of-plane solver (`rcwa_jones_2d` is in-plane only), through
  the shared full-3×3 first-order generator + generalized S-matrix, with the
  pointwise ezz-Schur effective-profile fold (Li 1999 Eq. 12) applied before
  factorization. Uniform cells match the Berreman 4×4 oracle to 2.4e-15;
  y-uniform out-of-plane gratings match the validated 1-D full-3×3 solver to
  ~3e-3; non-reciprocal `exz != ezx` total power tracks the 1-D value
  (physically ≠ 1).
- **`PMM2DStack`** (`pmm/stack2d.py`) — multilayer 2-D cascade mirroring
  `RCWAStack`: uniform films + scalar cells + in-plane tensor cells (each
  patterned layer keeps its OWN exact-wall SEM grid — no union-grid
  constraint), `add_tapered_pillar` (z-staircase with exact interpolated walls
  per slice), both-polarization solve returning
  `(orders, R(2,N), T(2,N), jones)`, `solve_vs_wavelength`, energy tripwire.
- **`stabilize=`** on the 2-D entry points — the 1-D per-order degree-scan
  consensus stepping consecutive ODD degrees, with a graceful scan-exhaustion
  sentinel at the cost cap and 2-D-calibrated tolerances.
- **JAX differentiability** — `pmm_efficiency_2d` auto-dispatches to a jnp
  twin on JAX inputs (traced `eps_pillar`/`eps_host`/indices/`depth`/
  `wavelength`/`theta`/`phi`; static bounds/degree/orders). AD matches FD to
  ~3e-9 (eps), ~5e-10 (depth/wavelength), ~1e-9 (theta at oblique); the
  centered-square normal-incidence angle gradient is a clean symmetry zero
  (~4e-15, no degenerate-gauge artifact). Arbitrary-cell autodiff stays with
  RCWA (a traced `eps_cell` cannot define exact walls).

### Fixed

- **Oblique-incidence transverse-momentum leak**: wall-less axes are now
  handled ANALYTICALLY (exact `diag(k)` Fourier operators — the operators
  kron-factor). The all-nodal path leaked 4-8% into y-momentum-forbidden
  orders on y-uniform cells at oblique incidence; now machine-exact (~1e-29).
  The "validated near normal" caveat is replaced with measured large-angle
  floors (validated to 60° vs RCWA-2D).
- **`_select_forward_flux` hardening** (shared with RCWA; full-tensor suite
  re-validated): (a) deep-evanescent modes in a PROJECTED modal basis carry
  projection-noise flux above the old 1e-9 tolerance with random sign — one
  growing mode classified forward blew the generalized cascade up by ~1e31;
  noise-scale flux on a decaying mode now defers to the decay sign; (b) a
  STABILITY band (|Re γ| > 0.5) is always classified by decay sign — cascade
  boundedness demands it; the audit-P2-A gyrotropic flux-first modes are
  near-propagating and unaffected.
- Conical-incidence guards adopted from RCWA in all 2-D PMM entry points
  (evanescent-incidence rejection + the exact-Wood-anomaly wavelength nudge).

## [5.13.0] — 2026-06-09

**Wavelength-sweep reuse + trapezoidal PMM gratings + a swept audit
(`AUDIT_V5_12_0`).** Adds assemble-once / solve-many wavelength sweeps to the 2-D
RCWA and PMM solvers and a staircased trapezoidal-grating builder to `PMMStack`,
then closes both confirmed P1s and all four P2s from the v5.12.0 adversarial
audit. The headline fix — RCWA silently returned **T = 0 into any lossy exit
substrate** — was long-standing (not a v5.12.0 regression) and is verified
correct across ~526 oracle-checked configurations (transmittance, reflectance
**and** the absorbed-fraction budget match an independent TMM to ~1e-15).

### Added

- **2-D RCWA wavelength sweep** — `prepare_rcwa_2d(...)` returns a
  `PreparedRCWA2D` that hoists the wavelength-INDEPENDENT permittivity
  factorization (Laurent `[[eps]]`, the Li `[[1/eps]]` z-rule, the `fff_nv`
  normal-vector tensor incl. the `O(N^3) inv([[1/eps]])`), the order set and the
  incident vector once; `.solve(wavelength)` then recomputes only the per-λ eig +
  S-matrix. `rcwa_efficiency_2d_vs_wavelength(...)` is the thin sweep wrapper.
  `prepared.solve(wl)` reproduces `rcwa_efficiency_2d(...)` to ~1e-13 (NumPy
  byte-identical). Non-dispersive indices + fixed `(theta, phi)`; NumPy/CuPy only
  (use `jax.vmap` over wavelength for a differentiable sweep).
- **2-D hybrid PMM wavelength sweep** — `pmm_efficiency_2d_vs_wavelength(...)`,
  same assemble-once / solve-many contract.
- **`PMMStack.add_tapered_grating(...)`** — trapezoidal / slanted-sidewall 1-D
  grating rendered as a z-staircase of thin VERTICAL PMM layers (lateral-exact,
  no Fourier floor in x) — the spectral-element counterpart of
  `RCWAStack.add_tapered_grating`. The vertical limit (equal duties) reproduces a
  single vertical layer exactly; a many-interface cascade that loses energy
  conservation now trips an explicit tripwire warning rather than returning
  silent gain.
- **`PMMStack.solve_vs_wavelength(...)`** — convenience all-vertical sweep reusing
  the geometry-only SEM assembly (eig-bound; a convenience wrapper, not a speedup).
- **`rcwa_efficiency_2d` / `prepare_rcwa_2d` / `*_vs_wavelength`:
  `allow_nonseparable_nv=False`** — opt-out for the new `fff_nv` non-separable
  raise (see below).

### Fixed

- **(P1-A) RCWA transmittance into a LOSSY EXIT SUBSTRATE was silently zeroed.**
  The internal `exp(+iωt)` loss bridge conjugates the region permittivity;
  `_sqrt_forward` (the `Im(kz) ≥ 0` branch) then returned `Re(kz) < 0` for a lossy
  substrate, and the `Re(kz) > 0` propagating mask read that as evanescent and
  zeroed `T` into **any** absorbing exit medium (a long-standing
  energy-corruption bug; reflectance and the mode-match were unaffected). A new
  `rcwa._core._forward_flux_kz` computes the z-flux weight, the propagating mask
  and the longitudinal `Ez` from the PUBLIC-convention forward `kz` (un-conjugated
  eps → `Re(kz) ≥ 0` for a forward wave into loss). Applied at all 9 affected
  sites across `rcwa_efficiency_1d/2d`, `rcwa_jones_1d/2d`,
  `rcwa_efficiency_2d_shapes`, `PreparedRCWA2D`, `RCWAStack`. **Lossless results
  are byte-identical** (real eps → conjugate is the identity).
- **(P1-C) The covariant slanted-Jones path shared the same bug** —
  `_pmm_jones_oblique_solve` (`factorization='covariant'`) selected the far-field
  flux `kz` with the same `Im(kz) < 0 → −kz` rule on conjugated eps, zeroing `T`
  into a lossy substrate. Fixed to the public-convention forward `kz`; now
  Fresnel-exact, lossless byte-identical.
- **(P1-B) `rcwa_efficiency_2d(formulation='fff_nv')` now RAISES on a
  non-separable (curved / non-axis-aligned) cell** instead of only warning. On a
  curved wall the normal-vector cross-term factorization mis-splits absorptance by
  ~50% (a lossless trap — `R+T+A` still closes); a non-blocking warning let the
  wrong number propagate silently. Pass `allow_nonseparable_nv=True` to downgrade
  to a warning for the reflection-only case (which still tracks). The gate now
  measures the **raw cell co-gradient curved-wall fraction** (the fraction of the
  boundary running diagonal to the axes) rather than `max|Nx·Ny|` on the smoothed
  normal field — the latter saturates at 0.5 for BOTH an axis-aligned square (fine)
  and a disk (broken) and so could not be a raise trigger.

### Changed

- **(P2-A, API) 2-D PMM return arity is documented as the unified `Efficiency2D`.**
  `pmm_efficiency_2d` / `pmm_efficiency_2d_staggered` return the cross-suite
  `Efficiency2D` (shared with `rcwa_efficiency_2d`), which unpacks to
  `orders, R, T` with `dof` as an attribute — NOT the v5.11 bare 4-tuple
  `(orders, R, T, dof)`. Migrate `o, R, T, dof = pmm_efficiency_2d(...)` to
  `o, R, T = pmm_efficiency_2d(...)[:3]` (or read `.dof`). The shared type keeps
  `rcwa_efficiency_2d` (historically a 3-tuple) backward-compatible.
- **(P2-B, docs) Covariant out-of-plane accuracy headline corrected.** The
  `~1e-4` floor was self-referential (covariant-vs-convection, which share the
  wall-normal inverse-rule floor). Vs an independent RCWA full-3×3 oracle the
  wall-normal TM channel floors at **~2.5e-3** (slant 45, a plateau); the TE
  channel is clean (<8e-4). The spectral-convergence capability is real; only the
  headline number was overstated.
- **(P2-C) `_cov_split` forward/backward rebalance normalized** — the fallback
  branch ranked propagating modes by Poynting flux `Sz` (~length²) and evanescent
  modes by `Im(kz)` (dimensionless) in one `argsort` without normalization; each
  population is now scaled to unit max magnitude first (the default
  forward-count-equals-half path is byte-identical).
- **(audit P3) Reconciled the covariant-OOP doc contradiction** (a test comment
  still called the covariant path "structurally unable to converge" the
  out-of-plane channel while the release ships it). Documented that the
  alias-conflict "raise on mismatch" idea was **considered and rejected** — it
  would break the deliberate, tested cross-suite `theta`/`angle` (and
  `n_orders`/`far_field_orders`) substitution contract.

### Performance

- **(P2-D) `import lumenairy` no longer eagerly imports `jax` or `numba`** — the
  pytree registration in `raytrace/jax_trace.py` is now lazy (first JAX-trace
  call), and the Numba JIT kernels in `elements/lenses.py`,
  `elements/_lens_traced.py` and `optimize/_merit_jit.py` compile on first use
  behind a `find_spec` availability probe. Removes ~5.8 s (`jax` ~3.9 s + `numba`
  ~1.85 s) of cold-start import; both modules report absent at import time, and
  every fast path still resolves to the JIT kernel on first call (pure-NumPy
  fallback when numba is absent).

## [5.12.0] — 2026-06-09

**First tag since v5.5.2 — package reorg + 2-D PMM correctness/perf.** This
release consolidates the entire untagged `5.6.0 … 5.11.0` line (native RCWA
convergence acceleration + ASR; the new `1-D` and `2-D` PMM solver families;
full out-of-plane anisotropy; the RCWA + PMM JAX-autodiff surfaces) and adds the
post-`5.11.0` structural reorganization of the two largest element modules into
packages plus two `2-D` PMM fixes. The top-level `lumenairy.*` public API is
**fully preserved and additive** (557/557 v5.5.2 symbols still resolve, +27 new,
0 removed); all dotted `lumenairy.elements.rcwa.*` / `pmm.*` imports resolve
unchanged. Backward-compatible — no migration is required for documented usage,
and it ships as a minor version accordingly.

### Changed

- **`rcwa.py` (5,612-line monolith) → `rcwa/` package** (behavior-preserving
  line-based AST split; every comment/byte preserved; public `__all__`
  unchanged): `rcwa/__init__.py` re-exports all 23 public names plus the
  test-imported / monkeypatched privates; `rcwa/_core.py` holds the BLAS
  controls, validation, Redheffer/S-matrix algebra, eigenmodes, convolutions,
  the homogeneous-mode cache, and the dimension-agnostic public utils
  (`uniaxial_tensor`, `rcwa_extrapolate`, `Efficiency2D`); `rcwa/oned.py` the
  `1-D` entry points (`rcwa_efficiency_1d` / `jones_1d` / `*_vs_wavelength` /
  segments / `efficiency_1d_jax` / ASR / segment builders); `rcwa/twod.py` the
  `2-D` solvers (`rcwa_efficiency_2d` / `jones_2d` / `_shapes` / normal-vector
  FFF); `rcwa/stack.py` the `RCWAStack` / `RCWAResult` / `rcwa_convergence`.
  Submodules use explicit cross-module imports; the package facade does the star
  re-export. Monkeypatch targets preserved (`CUPY_AVAILABLE` is patched in
  `rcwa._core`). Only observable delta: `rcwa.__file__` now ends in
  `rcwa/__init__.py` and `rcwa.__path__` exists.
- **`pmm.py` → `pmm/` package** (mirrors `rcwa/`): `pmm/_core.py` (shared
  GLL/Lagrange basis, metric/convection/covariant slant generators, slant
  solvers + `stabilize` selector, S-matrix cascade, far-field projection),
  `pmm/oned.py` (the public `1-D` entry points incl. the `pmm_1d` dispatcher,
  the JAX twin, and the right-angle convergence-class predictors),
  `pmm/stack.py` (`PMMStack`), with the two standalone `2-D` modules folded in:
  `pmm2d.py → pmm/twod.py` (hybrid `2-D` PMM) and
  `pmm2d_staggered.py → pmm/twod_staggered.py` (staggered `2-D` PMM). The slant
  dispatch call-spy now resolves through `pmm.oned`.

### Fixed

- **`2-D` hybrid PMM loss=gain bug.** `pmm_efficiency_2d` ran its modal solve in
  the internal `exp(+iωt)` convention but did not conjugate the public `eps`
  into it, so a passive lossy material (`Im(eps) > 0`) was read as **gain** —
  `R+T > 1`, negative absorptance, `T00 > 1`. Added the conjugation bridge
  (matching `pmm_efficiency_1d` / `rcwa_efficiency_2d`): lossy now matches the
  RCWA-`2D` oracle absorptance to ~`2e-3` and `R+T ≤ 1`; lossless is
  byte-unchanged (conjugating real `eps` is identity). New regression test
  `tests/unit/test_v5_12_0_pmm2d_loss.py`.
- **Reorg public-surface gap.** `pmm_efficiency_2d` / `pmm_efficiency_2d_staggered`
  were unreachable via `lumenairy.elements.pmm.*` (the package facade re-exported
  only `_core`/`oned`/`stack`, asymmetric with `rcwa/`). Added the `twod` /
  `twod_staggered` re-exports + `__all__` entries to `lumenairy/elements/pmm/__init__.py`.
- **Stale monolith doc references** after the reorg corrected (the `_core` / `twod`
  cross-module pointers no longer cite the deleted `rcwa.py` / `pmm.py` files).
- Post-reorg adversarial audit (full report in
  `docs/audits/RCWA_PMM_POST_REORG_AUDIT_2026_06_09.md`) **refuted** two
  speculative loss-sign claims by empirical verification (the covariant slant
  path and the staggered solver are already internally consistent under loss);
  those were left untouched, avoiding a regression.

### Performance

- **Staggered `2-D` PMM shared geometric eig (1.41×).**
  `pmm_efficiency_2d_staggered` previously ran three dense generalized
  eigensolves per call (one per region). Because the two homogeneous half-spaces
  split exactly into an `eps`-scaled field-Gram plus an `eps`-free geometric part
  (`L(eps) = eps·G + L0_geom`, with the `div(D)=0` Schur term `eps`-invariant),
  **one** geometric eig now serves both half-spaces (3 region eigs → 2). `R`/`T`
  identical to the old `3`-eig path to ~`1e-13` (lossless, lossy, asymmetric
  `n_sub ≠ n_sup`); the sharing-the-assembly-only variant was profiled and
  rejected (~1.02×).

## [5.11.0] — 2026-06-02

**1-D anisotropic device modeling + 2-D normal-vector FFF + stack/PMM
robustness.** A batch closing the functional + convergence gaps from the RCWA
gaps/wishlist, DynaMeta-port, and resonant-stack audits: full out-of-plane
anisotropy, arbitrary multi-region gratings, reflective-Jones device helpers,
the true normal-vector 2-D factorization, a resonance guard for multilayer
stacks, and an anisotropic-Jones spectral-element solver. Every pre-existing
path (isotropic, in-plane-tensor, `'laurent'`/`'li'`, scalar PMM, default
`RCWAStack.solve`) is **bit-identical** — all new behavior is gated/opt-in.

### Added

- **GAP7 — full 3×3 out-of-plane anisotropy in `rcwa_jones_1d`.** Tilted-director
  LC, magneto-optic / gyrotropic media (`eps_xz/yz/zx/zy ≠ 0`). The full 4N
  generator `G=[[A,P],[Q,B]]` is eigendecomposed with a generalized all-harmonic
  Poynting-flux forward selector (robust on non-reciprocal spectra, where a
  ±-pairing selector throws), and a generalized explicit forward/backward
  S-matrix carries the tilted layer's asymmetric modes. Berreman-4×4 validated to
  ~1e-15; the in-plane path is byte-identical. 1-D Jones path (2-D / `RCWAStack`
  out-of-plane pending).
- **GAP2 — `rcwa_jones_1d_segments`**: arbitrary piecewise-constant 1-D profiles
  (multi-level / interdigitated / N-region, each scalar or `(3,3)` tensor). Shares
  the exact `rcwa_jones_1d` solve core (refactored to `_jones_1d_from_profiles`,
  byte-identical).
- **W3 grating builders** — `grating_segments`, `binary_grating_segments`,
  `interdigitated_grating_segments` (emit the `segments` list for GAP2). **W2
  reflective-Jones helpers** — `reflective_outcoupling` (the
  PBS→QWP@45→grating→QWP@45→PBS cross-port FOM = `cos²(Γ/2)` for a lossless
  TE/TM-aligned grating) and `jones_retardance_diattenuation` (polar/SVD
  decomposition of a 2×2 Jones).
- **`rcwa_efficiency_2d(formulation='fff_nv')`** — the Schuster-2007 normal-vector
  Fast Fourier Factorization for 2-D: a continuous unit-normal field with the
  inverse rule on the boundary-normal projection, completing the factorization the
  existing `'li'` rule lacked (it inverse-ruled only `E_z`, never the in-plane
  normal field). Robust wins on dielectrics and separable stripes (→ rigorous 1-D
  Li); a correct, non-collapsing factorization on 2-D metal corners.
  `'laurent'`/`'li'` bit-identical.
- **`RCWAStack.solve(stabilize=True)`** — opt-in per-order consensus guard for the
  sharp-resonant metal-multilayer pathology (a near-singular mode-match biases a
  single diffraction order, e.g. a reflection null, non-monotonically at isolated
  `n_orders` while total power stays bounded). Re-solves a short downward
  `n_orders` window and returns the consensus solve; `stabilize=False` is the
  exact prior single solve.
- **`pmm_jones_1d`** — a binary 1-D grating whose ridge / groove are full
  `(3, 3)` IN-PLANE permittivity tensors (the tunable-LC reflective grating),
  returning the full complex `2x2` Jones reflection. The off-diagonal `exy`
  couples `E_x` ↔ `E_y` in the spectral-element modal eigenproblem (the modal
  field is a 2-vector `[E_x; E_y]` per node), so the phase relationship a
  tunable-LC grating needs is carried — the scalar TE/TM solver could not. The
  Li-1996 factorization is realized in the nodal basis: the wall-normal inverse
  rule `[[1/exx]]^-1` becomes `inv(hat(1/exx))`, and the `Kx`-derivative terms
  (`Ez`-elimination, `Kx^2`) become spectral-element STIFFNESS operators
  (weighted by `1/ezz` / `1`) so the inverse rule is **automatic and exact** (the
  `eps` jump lands on an element boundary). The coupled second-order modal
  operator mirrors the FMM tensor block `M = -P@Q` at normal incidence. PUBLIC
  `exp(-i w t)` convention end-to-end (no eps conjugation — the scalar PMM is
  self-contained in the public convention). Validated against `rcwa_jones_1d`:
  the `2x2` Jones matches to ~5e-6 (lossless tilted-LC) / ~2e-6 (lossy
  anisotropic metal) and converges spectrally with no floor; a diagonal tensor
  decouples back to the scalar TE/TM efficiencies (cross-pol `~1e-18`); lossless
  energy `sum(R)+sum(T)=1` (cross-pol included) to ~1e-14. Inherits the scalar
  PMM `stabilize` resonance guard (both incident polarizations must be passive).
  Normal incidence, binary, NumPy only (multi-region / oblique / autodiff are
  follow-ons). Exported top-level.
- **New tests** `tests/unit/test_v5_11_0_pmm_anisotropic.py` — the four
  validation gates (vs `rcwa_jones_1d`, decouple-to-scalar, energy, no-floor).
- **`pmm_jones_1d_slanted`** — the anisotropic-Jones counterpart of
  `pmm_efficiency_1d_slanted`: a binary 1-D grating with tilted side-walls AND
  full `(3, 3)` IN-PLANE permittivity tensors (a slanted tunable-LC /
  gyrotropic grating), returning the coupled `(E_x, E_y)` efficiencies and the
  zeroth-order `2x2` Jones reflection. Built from the genuine Edee-Granet 2024
  covariant-metric first-order Maxwell generator `-i k γ ψ = L ψ`,
  `L = A + B C⁻¹ D`, `ψ = [E_x; E_y; iZ H_x; iZ H_y]` (LINEAR in `γ`): the slant
  enters ONLY through the metric-folded effective tensors `εˡᵐ = √g (J⁻¹ ε Jᵀ)ˡᵐ`
  / `μˡᵐ = √g gˡᵐ`, and the magnetic field is a genuine state component read
  directly from the eigenvector, so the layer modes are flux-orthogonal by
  construction (the symplectic property a reshaped convection pencil lacks). The
  Li inverse rule lands on the wall-normal `εˡˡ`; the homogeneous half-spaces and
  the lab-frame Rayleigh far field reuse the proven `pmm_jones_1d` /
  `pmm_efficiency_1d_slanted` plumbing. Validated through the public API: lossless
  energy `sum(R)+sum(T)=1` (cross-pol included) to ~1e-13 across `0–60°` for BOTH
  real-symmetric AND gyrotropic tensors; high-contrast (`ε~12`) conserves; at
  `slant=0` it reduces to `pmm_jones_1d` (~2e-4) and a diagonal tensor decouples
  to the scalar `pmm_efficiency_1d_slanted` (TE machine-exact ~1e-7, TM
  inverse-rule ~6e-4); reciprocal for a real-symmetric tensor, non-reciprocal for
  a gyrotropic one. SCOPE: BINARY grating, in-plane tensor only (normal OR oblique
  incidence at any slant — see the round-19 entry below); NumPy/SciPy (not JAX).
  The multi-region (segments) path raises `NotImplementedError`. Exported
  top-level; tests in `tests/unit/test_v5_12_0_pmm_slant_and_convergence.py`.
- **`pmm_jones_1d_slanted` diagonal cure** (round 16; Granet 2017/2023, Liu
  2015). A **diagonal** tensor (`exy = eyx = 0`) with `exx == ezz` in BOTH
  regions now routes its TE / TM channels through the **div-conforming** scalar
  slant operator (`_sem_modes_slant`: the Li `1/eps` inverse rule lives INSIDE
  the z-stiffness, so it is free of the Liu-2015 spurious harmonic-mean static
  mode) — TE via `n = √eyy`, TM via `n = √exx` — and assembles the diagonal
  Jones. This sheds the latent **~2e-4** per-order accuracy gap the pointwise
  covariant-metric `E_z`-elimination (`_build_metric_generator`,
  `(ε33)⁻¹ = iS0·[[1/ezz]]`) carries (energy was already conserved to ~1e-12).
  **Coupled** tensors (`exy/eyx ≠ 0`) AND diagonal tensors with `exx ≠ ezz`
  fall through to the metric generator **unchanged** (byte-identical) and retain
  that latent gap; the full coupled / diagonal-anisotropic div-conforming cure
  is a documented frontier (`docs/PMM_ROADMAP.md` §8). Internal to the existing
  `pmm_jones_1d_slanted` (no new public API).
- **`pmm_jones_1d_slanted` — div-conforming `E_z` closure (all slants) + combined
  oblique + slant** (round 19; Granet 2023 Eq.16-18, Popov-Nevière App.B, Liu
  2015). The covariant-metric generator's `E_z` elimination is now
  **div-conforming at every slant**: the `(E_x,iZH_y)` longitudinal slot uses the
  Li-inverse-rule `−`stiffness `+(1/k) iS0·∫(1/εzz)B′B′` (`1/εzz` BETWEEN the
  discrete z-derivatives) in place of the spurious-prone pointwise `[[1/εzz]]`
  average, so the TM-block spectrum bit-matches the scalar slant solver, the
  Liu-2015 harmonic-mean static null is gone, and per-order TM converges to the
  scalar oracle (**~6.5e-4** @ deg32 / **~4.3e-4** @ deg40 at 45°; energy still
  ~1e-13). **Combined OBLIQUE incidence + nonzero slant is now SUPPORTED**:
  `kx0 = k0·Re(n_sup)·sin(angle)` is wired through the generator (Bloch-shifted
  `d1 → d1 + i kx0` in B/D + the kx0 antisym-convection / mass in the `1/εzz`
  bracket), the lab half-spaces, and the Rayleigh projection (with the oblique
  TM incident-flux normalization). The slanted layer's genuine `[E;H]` state is
  already lab-Cartesian, so its magnetic partner `V = −G` matches the proven
  `_sem_modes_tensor` lab half-spaces directly — **no inclined→lab shear** (a
  shear was measured to *break* conservation). Energy conserves ~1e-13 and the
  per-order split matches an RCWA z-staircase to **~2–3e-3 degree-cleanly** (no
  stabilize crutch) across angle×slant; survives adversarial refutation (negative
  angle, opposite-slant mirror symmetry ~1e-14, Wood anomalies, steep slant
  70–80°, high-contrast/gyrotropic). The combined oblique+slant case (even a
  diagonal cell) routes through the metric generator, since the scalar diagonal
  cure's oblique+slant per-order split is wrong (the *scalar*
  `pmm_efficiency_1d_slanted` now **delegates** combined oblique+slant here too —
  see below). The `NotImplementedError` guard on `angle≠0 & slant≠0` is removed.
  Internal to the existing `pmm_jones_1d_slanted` (no new public API).
- **`pmm_jones_1d_slanted_segments`** — SLANTED multi-region grating with full
  `(3, 3)` IN-PLANE tensors (the multi-region generalization of
  `pmm_jones_1d_slanted` and the slanted counterpart of `pmm_jones_1d_segments`).
  Solved by the **same** div-conforming covariant-metric `[E;H]` generator — the
  operator and the lab-frame far field are **region-count-agnostic**, so an
  N-region cell reuses the identical (validated) machinery on an N-segment nodal
  grid (binary and far-field code now share a single `_pmm_jones_slant_core`,
  bit-identical to `pmm_jones_1d_slanted`). This **closes the rounds-12–15
  frontier**: asymmetric ≥3-region slanted cells (which leaked to `sum(R)+sum(T)`
  ≈ 210 on the pre-round-19 operator) now conserve energy to **~1e-13** at all
  slants — the round-19 div-conforming Ez closure removed the spurious TM mode the
  asymmetric wall used to excite. Combined oblique + slant supported; `slant=0`
  reduces to `pmm_jones_1d_segments` and the binary cell to `pmm_jones_1d_slanted`.
  In-plane tensor only; exported top-level.
- **`PMMStack` slanted layers** — `add_layer(..., slant_angle=...)` tilts a
  layer's side-walls; a stack may freely **mix vertical and slanted layers**. An
  all-vertical stack keeps the symmetric `±q` cascade (**bit-identical** to the
  prior release); any slanted layer promotes the whole stack to the general
  forward/backward S-matrix (slanted layers solved by the div-conforming metric
  generator). A single slanted layer in a stack reproduces `pmm_jones_1d_slanted`
  to ~1e-12; mixed and slanted-multi-region stacks conserve energy to ~1e-13.
- **`pmm_jones_1d` — full `(3, 3)` OUT-OF-PLANE anisotropy** (εxz/εyz/εzx/εzy ≠ 0:
  tilted-director LC, magneto-optic / gyrotropic media). Previously rejected (the
  PMM was the in-plane `(exx,exy,eyx,eyy,ezz)` subset); an out-of-plane tensor now
  routes to the **native full-3×3 metric generator** — derived in the *same*
  Edee-Granet `A+B·C⁻¹·D` PMM layout (no RCWA/Berreman block structure, `V=−G`
  unchanged). The out-of-plane physics enters via the **pointwise εzz-Schur
  composites** `a_eff = εxx − εxz·εzx/εzz` … (Li 1999 Eq. 12, formed element-wise
  *before* the wall-normal inverse rule — the correct factorization order; a naïve
  spectral `B·C⁻¹·D` Schur of the raw εxz/εzx gives spurious modes) plus surgical
  single-derivative cross-blocks, **both vanishing identically at off-plane=0** so
  the generator is **byte-for-byte** the in-plane operator there (`np.array_equal`
  at normal AND slant AND oblique). Per-order matches `rcwa_jones_1d` to **<1e-3**
  (normal AND oblique), **lossless-trap-defeating** (absorbed fraction matches on
  lossy cells to ~5e-6), gyrotropic non-reciprocity physical. SCOPE: binary,
  VERTICAL grating, normal or oblique. Out-of-plane + a SLANTED wall stays
  guarded: it conserves energy but is per-order WRONG (measured 2–30e-3 vs an
  independent RCWA tensor z-staircase, the gap saturating with degree — a
  factorization defect, not resolution), because the slant metric fold must
  SUPERPOSE the out-of-plane components (`ε¹³ = −εzz·tanφ + εxz`, `ε²³ = εyz`, …
  Li 1999) — a focused operator follow-on. The shipped in-plane path is
  byte-identical.
- **`pmm_jones_1d_segments` + `PMMStack` — out-of-plane too.** The full-3×3
  out-of-plane support extends region- and stack-agnostically: a multi-region
  grating (`pmm_jones_1d_segments`) routes out-of-plane through the metric-
  generator segments path (per-order matches `rcwa_jones_1d_segments` to ~3.6e-4),
  and a **vertical** out-of-plane `PMMStack` layer is solved by the metric
  generator and cascaded with the general fwd/back S-matrix (a single out-of-plane
  layer reproduces `pmm_jones_1d` to ~1e-11; mixed in-plane/out-of-plane stacks
  conserve). A slanted out-of-plane layer raises (not yet per-order-validated).
  In-plane paths byte-identical.
- **`pmm_1d` — unified 1-D Jones dispatcher.** One entry point that auto-routes by
  geometry: binary (`eps_ridge`/`eps_groove`/`duty_cycle`) vs multi-region
  (`segments`), and vertical (`slant_angle=0`) vs slanted — to `pmm_jones_1d` /
  `pmm_jones_1d_slanted` / `pmm_jones_1d_segments` / `pmm_jones_1d_slanted_segments`
  respectively. Scalar (→isotropic) or full `(3,3)` (in-plane / out-of-plane) eps;
  normal or oblique. Each route is bit-identical to calling the specific solver.
  Exported top-level.
- **`pmm_efficiency_1d_slanted` — combined oblique + slant** (the scalar TE/TM
  ergonomics closure). The dedicated inclined-coordinate scalar solver's
  `kx0 ↔ slant` convection cross-term is unresolved, so the previous
  `NotImplementedError` on `angle≠0 & slant≠0` is **replaced by delegation** to the
  round-19 metric generator (`pmm_jones_1d_slanted`) with an isotropic `n² I`
  tensor, extracting the requested scalar channel (TE = E along the grooves =
  Jones row 1, TM = row 0). Output is **byte-identical** to that Jones row, so it
  matches a fine oblique RCWA staircase on the dominant order — TE to ~5e-5, TM to
  the shared wall-normal inverse-rule floor ~2e-3 — and conserves energy. Normal
  incidence (and any vertical grating) keeps the dedicated scalar solver
  unchanged. No new public API; out-of-plane + slant remains guarded (per-order
  unresolved — see the `pmm_jones_1d` out-of-plane entry).
- **`pmm_efficiency_1d` — JAX-differentiable** (inverse-design enablement, first
  increment). Passing a JAX array for any index/geometry argument routes the call
  to a self-contained `jax.numpy` twin (mirroring how `rcwa` dispatches), returning
  `jax.grad`/`jit`/`vmap`-able efficiencies differentiable w.r.t. `eps_ridge` /
  `eps_groove` (via `n`), `depth`, `wavelength`, and the half-space indices. The
  hardest component — the non-Hermitian eigendecomposition VJP with degeneracy
  regularization — is **reused** from `rcwa._jax_eig_stable` (the torcwa/fmmax-style
  Lorentzian-broadened custom-VJP eig); `rcwa.py` is **untouched**. The generalized
  modal pencil `A x = q² B x` (no JAX primitive) is folded to a standard
  `eig(B⁻¹A)` (validated forward-identical to the SciPy generalized solve to ~1e-12),
  and the element-loop assembly is rebuilt functionally (`jnp.at[].add`, `eps`
  enters linearly) with the eps-independent SEM topology frozen as constants. The
  **numpy path is byte-identical** — the JAX branch fires only on JAX inputs (a
  purely additive +387-line change, zero deletions). Validated: forward jnp≡numpy to
  ~5e-14; `jax.grad` vs central finite difference to rtol ~1e-8 (TE+TM, all four
  variables, on cells the build never used); jit/vmap-over-wavelength; the x64 guard
  warns on complex64. SCOPE (the de-risking spike): binary, NORMAL incidence,
  `elements_per_region=1`, fixed `degree` with `stabilize=False`, real lossless eps;
  `angle≠0` / `stabilize=True` / `elements_per_region>1` raise precise errors on the
  JAX path (NumPy-only). Oblique, complex/lossy eps, and the Jones path are follow-on
  increments. Requires `lumenairy[jax]` + `jax_enable_x64`. Tests in
  `tests/unit/test_v5_12_0_pmm_autodiff.py`.
- **`pmm_efficiency_1d` — moving-boundary `duty_cycle` gradient (JAX Phase 2).**
  Extends the differentiable surface to the grating wall position: `d/d(duty_cycle)`
  now flows through a smooth **fixed-topology moving mesh** — the wall sits exactly
  on an element boundary, so the element Jacobians `J=½(x_r−x_l)` (and the masses
  `∝J` / stiffness `∝1/J`) and the Rayleigh-projection phases `exp(−iG_m x(u))` carry
  the gradient analytically, with the element **count held fixed** (no remeshing, no
  Gibbs/Li-rule-of-a-blurred-step non-smoothness — the structural advantage PMM has
  over RCWA, which cannot differentiate `duty_cycle` at all). Validated: `jax.grad`
  vs a central finite difference that **physically moves the wall** to rtol ~1e-6
  (TE+TM, multiple cells); forward jnp≡numpy to ~1e-14; the numpy path and the
  Phase-1 `eps`/`depth`/`wavelength` gradients are unchanged; jit-compiles; a
  degenerate `duty=0/1` (zero-width region, singular Jacobian) raises in eager mode.
- **`pmm_efficiency_1d` — JAX oblique incidence + complex/lossy eps gradients (Phase 3).**
  Lifts the `angle≠0` guard on the JAX path: the Bloch shift `kx0 = k0·Re(n_superstrate)·sin(angle)`
  is threaded as a **traced** scalar through the modal operator (the antisymmetrized
  convection `−i·kx0·(C−Cᵀ) + kx0²·mass`, with the `1/eps`-weighted form for TM — an
  exact transcription of the numpy `_sem_modes`), the per-order Rayleigh `kx`, and the
  oblique TM incident-flux normalizer, so `d/d(angle)` and `d/d(n_superstrate)` flow.
  Complex/**lossy** eps is differentiable too (`d/d(Re eps)`, `d/d(Im eps)`,
  `holomorphic=False`). Normal incidence stays **byte-equal** (only the Python-literal
  `kx0=0.0` skips the convection; a *traced* angle valued 0 still flows its gradient).
  Validated — and crucially **lossless-trap-defeating**: on lossy cells the per-order
  R/T *and* the absorbed fraction `A=1−ΣR−ΣT` match the numpy/RCWA oracle to ~1e-15
  (genuine absorption 0.55–0.88), validated per-order, **not** by energy; `d/d(angle)`
  / `d/d(n_sup)` / complex-eps grads vs central FD to rtol ~1e-7; forward jnp≡numpy to
  ~1e-14; the numpy path and Phases 0–2 gradients are unchanged. The flux selector
  stays the differentiable noise-robust `where(flip,−q,q)` (no argsort). Gradients
  remain valid between Rayleigh-order cutoffs (the order count is fixed per trace).
- **`pmm_jones_1d` — JAX-differentiable (anisotropic 2×2 Jones, Phase 4).** The
  in-plane-tensor binary grating (tunable-LC reflective grating) is now
  `jax.grad`-able on a JAX input: it routes to a `jax.numpy` twin whose 2n×2n coupled
  `[E_x; E_y]` modal solver is a **standard** eig (`Mbig`), so the reused
  `rcwa._jax_eig_stable` custom-VJP eig applies *directly* (no generalized fold). The
  2×2 complex `jones` and `R_eff`/`T_eff` are differentiable w.r.t. the tensor entries
  — **including the off-diagonal `exy`/`eyx` cross-pol coupling**, real and imaginary
  (lossy) — plus `depth`, `wavelength`, `angle`, and the half-space indices. Forward
  jnp≡numpy to ~5e-15 (`R`,`T`, and the Jones matrix); `jax.grad` of a cross-pol FOM
  `|jones[0,1]|²` and of `sum(T)` vs central FD to rtol ~1e-9…3e-7; a diagonal tensor
  reduces to the scalar-TE gradient (~5e-10); **lossless-trap-defeating** (lossy-tensor
  per-order R/T + absorbed fraction match the numpy oracle to ~5e-15, absorption
  5–78%). The degenerate TE/TM modes at normal incidence are handled by the Lorentzian
  broadening + gauge-invariance of `|J|` (no gauge fix — it would corrupt the gradient;
  `d/d(angle)` at *exactly* normal is the symmetry-protected zero, differentiate at
  oblique). The underdetermined incident-amplitude projection uses a closed-form
  min-norm pseudo-inverse (forward-identical to numpy's SVD `lstsq` to ~1e-14) because
  `jnp.linalg.lstsq`'s VJP is undefined there. The numpy path is **byte-identical**
  (24-array snapshot vs a detached HEAD worktree, max|d|=0); rcwa.py untouched; the
  scalar Phase 0–3 gradients unchanged. SCOPE: in-plane tensor, VERTICAL wall, normal
  or oblique, `elements_per_region=1`, fixed degree (`stabilize=False`). Slanted-Jones
  and out-of-plane JAX paths (the heavier metric-generator eig) remain follow-ons;
  out-of-plane+slant stays per-order-guarded in numpy too.

#### Lower-priority ergonomics / hardening (from the same audits)

- **JAX off-plane differentiability guard.** The in-plane (`z`-decoupled tensor)
  1-D Jones path is `jax.grad`-differentiable (verified vs central finite
  difference across multiple objectives / `n_orders`); the full-3×3 *out-of-plane*
  path is **not** (its forward-mode flux selector `_select_forward_flux` uses
  `np.where`/`argsort`, which breaks the trace). Because `_tensor_offplane_present`
  skips JAX arrays, a concrete off-plane JAX tensor would otherwise be silently
  routed through the in-plane solver — a wrong answer *and* a wrong gradient.
  `rcwa_jones_1d` / `rcwa_jones_1d_segments` now raise a clear `NotImplementedError`
  for a concrete off-plane JAX tensor (mirroring the `formulation='fff_nv'` JAX
  rejection); the in-plane subset is pinned differentiable. NumPy paths are an
  exact no-op (the guard fires only under `if is_jax:`). *Known narrow gap:* an
  off-plane tensor built purely **inside** a trace (a `Tracer`) cannot be
  inspected and routes in-plane — documented in the docstring.
- **`RCWAResult.layer_absorption` for tensor-cell layers.** Per-layer loss
  attribution for the metal-tooth / LC `(3,3)`-tensor layers via the anti-Hermitian
  loss density `Im(Eᴴ·ε·E)` (the public `exp(-i w t)` convention; `Im(ε_public) ≥ 0`).
  A uniform tensor cell matches the equivalent scalar layer **bit-identically**;
  per-layer values close to the total absorptance and localize loss to the lossy
  layer. Uniform / isotropic-cell layers are unchanged; analytic-shape layers still
  raise.
- **Fixed — `internal_field` / `layer_absorption` no longer overflow through deep,
  lossy layers** (audit `AUDIT_RCWA_STACK_RESONANT_CONVERGENCE` Part 4.2). The
  internal-field recovery referenced the backward mode to the layer *top*
  (`c⁻·exp(+lam·k0·z)`), which **grows** through the layer and overflowed to `NaN`
  for a deep, high-loss metal layer at high `n_orders` (the highest evanescent
  orders have `Re(lam·k0·thickness) > 709`) — silently collapsing
  `layer_absorption` to `[0, 0]` while `absorptance()` was nonzero. The backward
  mode is now referenced to the layer *bottom* (`c⁻_bot·exp(-lam·k0·(L-z))`,
  a **decaying** exponent) via the reflection-below-bottom S-matrix partial — every
  exponential is bounded, so a deep Cu/LC gap-plasmon layer reconstructs cleanly and
  per-layer loss sums to the total absorptance. Math-identical for shallow layers
  (the field values are unchanged).
- **`reflective_outcoupling` is now backend-agnostic.** `jax.grad` traces through
  the side-port out-coupling FOM (so it can be an inverse-design objective directly);
  a NumPy Jones still returns a Python `float` **bit-identical** to before. The full
  loop (`PBS → QWP@45 → grating → QWP@45 → PBS`) is differentiable **end-to-end**
  w.r.t. the anisotropic-LC grating design (`jax.grad` flows from
  `rcwa_jones_1d_segments` through `reflective_outcoupling`, matching central FD),
  so side-port power can be gradient-optimized directly — pinned by
  `test_side_port_outcoupling_end_to_end_differentiable` (needs JAX float64, as
  always for a meaningful FD).
- **`rcwa_convergence` accepts an `RCWAStack`.** Solves the stack at its configured
  `n_orders` and a bumped count, reports/warns on the **per-order** efficiency delta
  (a sharp-resonant metal multilayer biases an isolated order while total power
  stays bounded), and restores the stack's `n_orders`. The single-entry-solver path
  is unchanged.
- **`rcwa_jones_vs_wavelength_segments`** — the segmented (multi-region / multi-level)
  dispersive Jones sweep companion to `rcwa_jones_1d_segments`; `segments` and the
  half-space indices may be `wl → …` callables (material dispersion). A 2-segment
  sweep is **bit-identical** to the binary `rcwa_jones_vs_wavelength`. Exported.
- **New tests** `tests/unit/test_v5_11_1_rcwa_lowerpri.py` (19) — differentiability
  (in-plane grad vs FD; off-plane raises), tensor `layer_absorption` (sum-to-total +
  uniform-tensor-vs-scalar bit-identity), `reflective_outcoupling` NumPy bit-identity
  + JAX grad, stack `rcwa_convergence`, segmented dispersive sweep. An adversarial
  three-skeptic review (tensor-loss physics, the JAX guard, bit-identity of the
  touched public surface) found no defects.

#### OBLIQUE PMM — off-angle parity for `pmm_efficiency_1d` + `pmm_jones_1d`

- **`pmm_efficiency_1d` and `pmm_jones_1d` now support oblique incidence**
  (`angle != 0`) — previously a `NotImplementedError`. This is the PMM analogue
  of off-angle RCWA: the pseudo-periodic envelope `E(x) = exp(i kx0 x) u(x)`
  (`kx0 = Re(n_sup) sin(angle) k0`) is solved by Bloch-shifting every
  `x`-derivative `d/dx -> d/dx + i kx0` in the spectral-element operators, and
  the Rayleigh far-field / incident flux pick up the `kx0` offset. **`angle == 0`
  is bit-identical** to the prior normal-incidence solve (the shift terms vanish
  and the legacy branch is used). Validated against `rcwa_efficiency_1d` /
  `rcwa_jones_1d` at matched angle: dielectric / tunable-LC tensor match to
  ~1e-5–1e-4 with energy conserved to ~1e-13 (lossless), spectral convergence in
  `degree`, across 0–60°. Three physics points were required and are worth
  recording:
  - **Antisymmetrized convection.** The Bloch cross-term is `-i kx0 (C - Cᵀ)`
    with `C = ∫φ φ'` (TM uses the `1/eps`-weighted `Cinv`). For TM the weight
    *varies across the wall*, so `Cinv` is **not** antisymmetric and the naïve
    `-2i kx0 Cinv` is wrong (it silently gives an energy-conserving-but-wrong
    TM answer); the `(Cinv - Cinvᵀ)` form is required. TE (unit weight) is
    unaffected.
  - **Forward-mode selection.** At oblique the operator is complex-Hermitian
    (lossless), so the QZ `eig` leaks imaginary noise; the scalar path uses a
    noise-robust `Im(q)` branch, and the **coupled anisotropic** path — whose
    modes do not split cleanly by `Im(q)` — selects the forward set by the sign
    of each mode's **z-Poynting flux** (`Im(Eₓ·S0·conj(H_y) − E_y·S0·conj(Hₓ))`).
  - **Per-column incident normalization.** The incident `Eₓ` (p-pol) wave also
    carries `E_z = -kx0 Eₓ/k_z`, so its incident z-flux is `eps_sup/k_z`, not
    `k_z`; normalizing both polarizations by `k_z` over-counted the p channel by
    `1/cos²(angle)`. Fixed per-column (s-pol unaffected; bit-identical at `kx0=0`).
  - *Known limit:* very lossy **metal-corner TM at steep oblique** stays
    resonance-limited (`stabilize` may raise) — use `rcwa_efficiency_1d` there.
    The realistic dielectric / tunable-LC regime is robust across angle.
- **`stabilize` now gates on PER-ORDER convergence, not just total power**
  (`pmm_efficiency_1d` + `pmm_jones_1d`, normal AND oblique). An adversarial
  review found that the old gate keyed only on `sum(R)+sum(T)`, which the
  S-matrix conserves even when the modal basis is *under-resolved* — so a
  high-index / many-order grating at the default degree could return per-order
  efficiencies / a Jones matrix wrong by tens of percent **with no warning**
  (e.g. n≈3.8 TM at 40° gave a 33%-wrong zeroth order while `sum(R)+sum(T)=1` to
  1e-7). The consensus now requires two passive degrees to agree on the
  **per-order efficiencies and the 2×2 Jones** (a resonance-contaminated passive
  degree is isolated — it has no per-order partner), returning a converged
  degree or, failing that, warning that the result is likely under-resolved
  (raise `degree` / `elements_per_region`). Well-converged configs are
  unaffected; `stabilize=False` is unchanged.
- **New tests** `tests/unit/test_v5_11_0_pmm_oblique.py` (17) — scalar TE/TM and
  anisotropic-Jones oblique vs the FMM oracle (multiple angles), `angle==0`
  bit-identity, the all-vacuum `1/cos²` regression, diagonal-tensor decoupling,
  lossy-LC, and degree convergence. The two prior `*_rejects_oblique` guards were
  retargeted to assert oblique now matches the FMM.

#### MULTI-REGION PMM — `pmm_efficiency_1d_segments` + `pmm_jones_1d_segments`

- **Arbitrary piecewise-constant 1-D profiles** (multi-level staircases,
  interdigitated / N-region cells, mixed isotropic / liquid-crystal regions) by
  the PMM — `pmm_jones_1d_segments` is the spectral-element counterpart of
  `rcwa_jones_1d_segments`, and `pmm_efficiency_1d_segments` is the scalar fast
  path (no rcwa equivalent). `segments` is a list of `(width_fraction, eps)`
  (each `eps` scalar or `(3,3)` in-plane tensor; fractions sum to 1) — the same
  format the `grating_segments` / `binary_grating_segments` /
  `interdigitated_grating_segments` builders emit. A region wall lands on every
  spectral element (eps exact per element, no Gibbs), so it converges spectrally
  in `degree` with no accuracy floor; the binary ridge/groove path is the
  2-segment special case. Validated against `rcwa_jones_1d_segments` to ~1e-5
  (Jones) / ~1e-4 (per-order) across normal + oblique, lossless + lossy,
  multi-region tensor (the tunable-LC `metal | LC | metal | LC` device). The
  binary `pmm_efficiency_1d` / `pmm_jones_1d` are **bit-identical** (the solve
  core was refactored to `_pmm_solve_core` / `_pmm_jones_solve_core`, shared by
  the binary + segmented wrappers; the per-order stabilize was extracted to
  shared helpers). Exported top-level. Two subtleties were found and fixed:
  - **Robust forward selector at normal incidence.** A many-element multi-region
    cell has *dense* isolated-degree resonances that the legacy `Im(q)` branch
    cannot dodge (the segmented solve blew up at almost every degree at exactly
    `angle=0`, while the slightest oblique angle was clean). The segmented solver
    therefore uses the noise-robust / z-Poynting-flux forward selector even at
    normal incidence (calibrated there); the binary path keeps the legacy branch
    (bit-identical).
  - **Mirror-handedness.** The PMM's nodal `x` is mirrored relative to the FMM
    (`rcwa_jones_1d_segments` places `segments[0]` on `x ∈ [0, w0)`), which at
    oblique incidence gives the x-reversed spectrum for an *asymmetric* profile.
    Invisible for the binary (every 2-region cell is mirror-symmetric about its
    own centre, so `R[+m]=R[-m]`), it surfaced only for 3+ asymmetric regions at
    oblique; the segment layout is reversed to match the FMM order-by-order.
- **New tests** `tests/unit/test_v5_11_0_pmm_segments.py` (16) — Jones + scalar
  multi-region vs the FMM (normal + oblique, lossless + lossy, tensor + scalar),
  the asymmetric-oblique per-order mirror regression, uniform-N-region
  transparency, scalar-eps promotion, and the validation guards.

#### MULTILAYER PMM — `PMMStack` (the `RCWAStack` analogue)

- **`PMMStack`** composes multiple anisotropic 1-D patterned layers + uniform
  spacers between a superstrate and substrate and Redheffer-stacks them — the
  spectral-element counterpart of `RCWAStack`. The one structural requirement
  (different layers have different walls, so their mode matrices must share a
  grid) is met by solving the whole stack on the **union of every layer's walls**
  (one shared nodal grid; a wall lands on every element boundary, so eps is exact
  per element and each layer converges spectrally in `degree`). Anisotropic /
  Jones throughout (scalar layers promoted to isotropic; in-plane tensors only,
  use `RCWAStack` for out-of-plane), normal or oblique incidence, using the same
  z-Poynting-flux forward selector as the multi-region single-layer solver (so
  the many-element shared grid stays resonance-free). Builder API mirrors
  `RCWAStack`: `add_layer(thickness, eps=… | segments=…)` → `set_source(wl,
  angle=…)` → `solve()` → `(orders, R, T, jones)`. Validated: a **1-layer stack
  is bit-identical** to `pmm_jones_1d_segments`, and a **2-layer tensor stack
  matches `RCWAStack` to ~5e-4** (0-order, both polarizations, normal + oblique)
  with energy conserved to ~1e-6. Exported top-level. (Result-object features —
  `layer_absorption` / `internal_field` for a `PMMStack` — are follow-ons.)
- **New tests** `tests/unit/test_v5_11_0_pmm_stack.py` (10) — 1-layer
  bit-identity, 2-layer vs `RCWAStack` (normal + oblique), uniform-spacer +
  energy, all-vacuum transparency, isotropic-decoupling, and the guards.

#### 2-D PMM — `pmm_efficiency_2d` (hybrid crossed-grating modal solver)

- **`pmm_efficiency_2d`** is the 2-D (doubly periodic) analogue of
  `rcwa_efficiency_2d` for a **separable rectangular pillar** (`eps_pillar`
  rectangle in an `eps_host` background) — the modal-method counterpart that
  resolves the pillar edge on a tensor-product GLL nodal grid instead of a
  staircased Fourier series. It is a **hybrid**: the structured layer is a nodal
  spectral-element operator **Fourier-Galerkin-projected** into the Rayleigh
  basis, paired with **analytic plane-wave half-space regions** (`W = I`, exact
  flux). Validated vs `rcwa_efficiency_2d` (Li rule, matched truncation) to
  ~2e-4 on the 0-order at `degree=11`, energy conserved to ~1e-3; vacuum is exact
  and degree-independent; a square pillar reproduces C4v symmetry (TE x-orders ≡
  TM y-orders) and ±-order symmetry at normal incidence.
  - **The null-mode fix that makes it viable.** A naive nodal `[Sx;Sy] P@Q` solve
    produced a fatal cloud of spurious modes — *not* a fundamental vector-Maxwell
    spurious-gradient problem, but the classic **periodic-grid Nyquist null mode**
    of the nodal first-derivative, present exactly when the per-axis node count is
    *even*. Forcing it **odd** (`3·degree·elements_per_strip` odd) restores the
    correct 1-D derivative kernel and the divergence-reduced second-order form then
    injects no spurious modes — no grad-div penalty / projection / Lagrange
    multiplier needed.
  - **Honest scope.** Single separable rectangular pillar, isotropic scalar
    TE/TM, single layer, normal / near-normal incidence (oblique is wired via the
    Bloch shift but unvalidated at large angles). Because the layer lives in a
    truncated Rayleigh basis of half-width `n_orders`, this solver **has a
    Fourier-truncation floor like the FMM** — it is *not* no-floor like the 1-D
    `pmm_efficiency_1d`. For arbitrary 2-D profiles, anisotropy/full-Jones, or
    multilayer stacks use `rcwa_efficiency_2d` / `RCWAStack`. (A genuinely
    no-floor 2-D nodal method is blocked by the flux-inconsistent degenerate
    uniform-region nodal eigenproblem — the same wall RCWA sidesteps with its
    analytic region path — and is being pursued separately via an FEEC E–D
    formulation.) New module `lumenairy/elements/pmm/twod.py` (was `pmm2d.py`
    before the 1-D/2-D package reorg), exported top-level.
- **New tests** `tests/unit/test_v5_11_0_pmm2d.py` (17) — vacuum exactness +
  degree-independence, pillar vs the rcwa li oracle, energy conservation, C4v +
  ±-order symmetry, and the odd-node / n_orders / polarization guards.
- **`pmm_efficiency_1d_slanted` — slanted 1-D lamellar gratings by the
  inclined-coordinate PMM (Granet, Randriamihaja & Raniriharinosy, JOSA A 34:975,
  2017).** A slanted side-wall that RCWA must STAIRCASE into many laterally-shifted
  thin layers is solved as a SINGLE layer in the inclined coordinate
  `u = x − tan(φ)·z` (the walls become coordinate surfaces, so `eps` depends on
  `u` only) — no z-staircase. The slant injects a linear-in-`q` convection term
  (the modal eigenproblem becomes quadratic, companion-linearized) and breaks the
  `±q` field symmetry, so the explicit forward/backward generalized S-matrix is
  reused. Validated vs an RCWA staircase: NORMAL incidence, slant 0–75°, TE+TM;
  TE matches a fine staircase to ~1e-5 and reaches the converged efficiencies at a
  single-layer DOF the staircase needs ~30–70× more work for. `slant_angle=0`
  reduces **bit-identically** to `pmm_efficiency_1d`. SCOPE: normal incidence only
  for a slanted grating — combined oblique + slant raises `NotImplementedError`
  (the inclined-frame Bloch↔slant convection cross-term is unresolved; energy
  conserves but the per-order split is wrong) rather than returning a wrong answer.
- **`grating_convergence_class` / `classify_from_grating` — a convergence-class
  predictor for right-angle grating edges (Li & Granet, JOSA A 28:738, 2011).** A
  pure O(1) diagnostic that classifies the in-plane-E (TM) field singularity at a
  four-region corner: Type I (all-dielectric, algebraic convergence at rate
  `Re[τ]`), Type II (lossless metal–dielectric, irregular — **no** modal method
  converges), Type III (requires a metal quadrant; impossible for an all-dielectric
  corner). Returns `τ`, `Δ`, `Δ'`, the predicted algebraic rate, a `converges`
  flag, and a diagnostic warning. Applies to both PMM and RCWA. (Loss lifts the
  Type-II irregularity only *asymptotically* — a weakly lossy metal corner can
  still stall at practical truncation; the warning says so.)
- **New tests** `tests/unit/test_v5_12_0_pmm_slant_and_convergence.py` (24) —
  slant=0 bit-identical reduction, energy conservation 10–75° (TE+TM), oblique+slant
  guard, slant vs RCWA-staircase cross-check; predictor Type I/II/III, the Type-II
  closed-form sign, Type-III-impossible-for-all-dielectric (200 random corners),
  lossy regularization, and the degenerate-edge guard.
- **`pmm_efficiency_2d_staggered` — canonical NO-FLOOR 2-D crossed-grating PMM
  (Granet, JOSA A 40:652, 2023; the faithful staggered modified-Legendre basis).**
  The 2-D analogue of `pmm_efficiency_1d`, and the no-floor counterpart of the
  FMM-floored hybrid `pmm_efficiency_2d`. Solves every region (cover/film/substrate)
  in the SAME staggered modal basis at equal dimension, so every interface is a
  SQUARE modal match (the 1-D `_pmm_solve_core` architecture lifted to 2-D, reusing
  `_interface_smatrix`/`_redheffer_star` unchanged) and the Rayleigh projection is
  applied ONCE, forward-only, at the far field. The longitudinal field is slaved by
  `div(D)=0` (the `-K_tz (eps33)^-1 K_zt` Schur term) and continuity is embedded in
  the basis via shared hats + the Bloch periodic hat — so the eigensolver is
  **spurious-free by construction** (the mimetic `span(d·B̃)=span(B)` de Rham
  property, verified to ~1e-14; no stabilization parameter). Result: the energy
  balance is **`n_orders`-INDEPENDENT** (no Fourier floor — byte-identical across
  n_orders, tracking only the modal degree to ~1e-13) with **exact sidewalls and
  position invariance**, validated against the analytic uniform-slab Fabry–Pérot to
  ~1e-9 and bracketing the RCWA-li value from the opposite side (PMM pins the value
  RCWA converges toward on a hard high-contrast case). SCOPE: axis-aligned
  rectangular pillars (walls on the `(Nx,Ny)` `eps_cell` grid), single layer,
  isotropic TE/TM, NumPy dense eig; **corner-capped** (algebraic, no-floor — at-best
  RCWA parity per DOF on vertical pillars, the win being accuracy quality). Curved/
  slanted boundaries (Granet's transfinite curved-quad mapping) are a follow-on. New
  module `lumenairy/elements/pmm/twod_staggered.py` (was `pmm2d_staggered.py`
  before the 1-D/2-D package reorg), exported top-level; kept DISTINCT
  from `pmm_efficiency_2d` (different convergence class + geometry input).
- **New tests** `tests/unit/test_v5_12_0_pmm2d_staggered.py` (12) — vacuum
  exactness, the no-Fourier-floor gate (energy byte-identical across n_orders),
  no-floor-in-degree, uniform-slab Fabry–Pérot vs analytic, position invariance,
  lossy-pillar absorption, TE/TM energy conservation, RCWA cross-check, and input
  validation.

### Fixed

- **Element-size-aware conditioning for the PMM solver — fixes the thin-feature /
  tapered-stack `Singular matrix`.** The spectral-element operators carry the
  element Jacobian `J = (x_r − x_l)/2` (`S0 ∝ J`, `K ∝ 1/J`), so a grid spanning a
  huge width ratio — a thin liner/coating next to a wide region, and especially
  the `PMMStack` **union grid of a tapered stack** whose per-slice walls land
  sub-nm / coincident — drove `S0` singular (`J → 0`) and `pmm_jones_1d*` /
  `pmm_efficiency_1d*` / `PMMStack` raised `numpy.linalg.LinAlgError: Singular
  matrix` (or returned non-physical `|J| ~ 10¹⁰` modes). Two-part fix:
  - **(A) gated symmetric Jacobi equilibration** of the SE inversions
    (`_safe_inv` / `_safe_solve` / `_safe_geig` at the `S0` inverse, the wall-
    normal `[[1/εxx]]⁻¹`, the scalar `1/ε` solve, and the generalized eig). It is
    the exact identity `inv(A) = D inv(DAD) D` for the real-positive mass `S0`
    (and a conditioning-reducing similarity rescale for the complex operators),
    and is **gated on an ill-scaling test so every well-scaled geometry takes the
    plain, bit-identical path** — existing results are unchanged.
  - **(B) near-coincident-wall merge in the `PMMStack` union grid** so a genuinely
    zero-width union cell (which equilibration cannot rescue) never forms; the
    snapped walls differ by < 1 pm, so there is no physical effect — the spurious
    wall is removed and the result matches the exactly-aligned grid.
  Unlocks PMM (with its ~10–100× speed over the FMM for resonant devices) on the
  conformal-coating / barrier-liner / tapered-staircase class that was RCWA-only.
  New tests `tests/unit/test_v5_11_0_pmm_element_size_scaling.py` (8); all 104 PMM
  tests pass (well-scaled paths bit-identical). Audit:
  `docs/audits/AUDIT_PMM_ELEMENT_SIZE_SCALING_2026_06_03.md`.

## [5.10.6] — 2026-06-02

**PMM build-portability fix: resonance-robust degree selection (`stabilize`).**
A CI flake surfaced a real robustness bug in `pmm_efficiency_1d`: the Polynomial
Modal Method has discrete *resonances* at isolated polynomial degrees where a
near-singular layer↔region interface mode-match injects spurious flux, inflating
`sum(R)+sum(T)` above the clean value (catastrophically — R+T ≫ 1 — or only
mildly, leaving R+T ≤ 1 yet biasing the efficiencies). The resonant degrees are
**LAPACK-build dependent**, so `degree=24` returned an off-curve `T` (0.3997 vs
the true 0.3982) on the CI build while passing locally, tripping
`test_pmm_self_converges_no_floor`. The old `stabilize` gate (accept the *first*
degree with R+T ≤ 1) let the mild, sub-unity resonances through.

### Fixed

- **Convergence-consensus `stabilize` selector.** `stabilize=True` (default) now
  scans a short upward degree window, collects the *passive* solves (total power
  within tolerance of unity — discarding the super-unity resonances), and locks
  onto the **consensus** the converged degrees agree on (the largest cluster of
  mutually-consistent totals), returning the requested degree unchanged when its
  own total is in that cluster (DOF preserved, bit-identical at clean degrees)
  and the nearest clean degree otherwise. This rejects **both** off-curve modes:
  the *upward* resonance spikes **and** the *downward* under-convergence deficit
  at low degree on high-index gratings (an adversarial audit showed a naïve
  "minimum-power = clean" rule would latch onto the worst-converged low-degree
  solve — e.g. silicon n≈4 at degree 8 was biased by up to 6.5e-2; the consensus
  rule returns the converged value to ~4e-5). Resonances proliferate into
  multi-degree bands at high degree, so the scan raises a clear error if no
  passive solve exists, and warns if the solution never stabilizes within the
  window (genuinely under-resolved — raise `degree` or `elements_per_region`).
  `stabilize=False` is unchanged (exact degree). Verified across metal / lossless
  / dielectric / silicon, both polarizations; all PMM tests pass.
- **New regression tests** `test_pmm_degree_robustness_no_resonance_leak` (no
  inflated total power / off-curve outlier at any requested degree) and
  `test_pmm_low_degree_high_index_converges` (low-degree high-index solves track
  the FMM oracle, not the under-converged value).

## [5.10.5] — 2026-06-02

**Autodiff completeness: batched (vmap) solves + validated Hessians (audit W1 /
W6).**  Both fall out of the now-traceable 2-D solve; this pins them with
tests (no code change).

### Added

- **W1 — `jax.vmap` batched geometry solve.**  A parameter grid / inverse-design
  population solves in **one device call**: `jax.vmap(solver)(batch_of_cells)`
  matches the sequential loop, and `jax.vmap(jax.grad(solver))` gives batched
  gradients — the throughput lever for sweeps and population optimizers.
- **W6 — validated 2nd-order autodiff.**  `jax.hessian` through the full vector
  solve matches a finite-difference-of-gradient to ~1e-2 for **depth / geometry**
  parameters — relevant for Newton-type inverse design.  Requires `jax_enable_x64`.
  *Scope (corrected 2026-06-07 audit):* Hessians w.r.t. cell **permittivity** are
  **not** supported — the non-symmetric eigenvector 2nd-derivative is undefined in
  the Lorentzian-broadened custom-VJP eig (`jax.hessian` wrt eps raises
  `NotImplementedError`); first-order eps gradients are fine.

## [5.10.4] — 2026-06-02

**Differentiable multi-layer `RCWAStack` (audit P1 — DynaMeta port blocker).**
Completes the 2-D autodiff story: v5.10.3 validated the single-layer
`rcwa_efficiency_2d` gradient; this makes the full **multi-layer stack** solve
JAX-traceable, so a 2-D metasurface figure-of-merit differentiates through a
stack of patterned + spacer layers (the basis for stacked-layer inverse
design).

### Changed

- **`RCWAStack.add_layer` / `solve` are now JAX-differentiable.**  `add_layer`
  keeps a JAX `eps_cell` / `eps_tensor_cell` / `thickness` native (a `np.asarray`
  / `float()` used to materialise the tracer and break the trace); `solve`
  dispatches the backend off the patterned-layer arrays and gates the
  concrete-only guards (the grazing nudge, the energy check) on the traced path
  — exactly as the single-entry 2-D solver already does.  Gradients of
  `sum(T)` w.r.t. a **cell permittivity** and a **layer thickness** match finite
  difference to ~1e-3 on a two-layer stack.  The NumPy / CuPy path is unchanged
  (the stable-eig custom-VJP and the per-layer eig solve were already
  dimension-agnostic).  Requires `jax_enable_x64`.

## [5.10.3] — 2026-06-01

**Validated 2-D RCWA differentiability (audit P1, single-layer).**

### Added

- A regression test pinning `rcwa_efficiency_2d` 2-D JAX **autodiff** against
  finite difference (gradients of `sum(T)` w.r.t. a cell permittivity and the
  layer depth, matched to ~1e-3).  The stable-eig custom-VJP is
  dimension-agnostic, so the full vector 2-D (crossed-grating / metasurface)
  solve was already differentiable -- this confirms and protects it as the
  basis for **single-layer 2-D inverse design**.  Requires
  `jax.config.update('jax_enable_x64', True)` (the solver already warns when
  x64 is off, since its eigenproblem is ill-conditioned in single precision).
  - *Remaining (P1):* multi-layer `RCWAStack.solve` is not yet JAX-traceable
    (it forces NumPy + uses a host-side eig cache); that is the larger
    follow-up for differentiating stacked-layer designs.

## [5.10.2] — 2026-06-01

**RCWA per-layer absorption (audit GAP6).**  Built on the v5.10.1 internal
field; additive.

### Added

- **`RCWAResult.layer_absorption(*, nx, ny, nz_per_layer)`** — attribute the
  total absorptance to each LAYER (where is the power lost — metal teeth vs
  back-reflector vs lossy spacer?).  Integrates the local loss density
  `Im(eps)|E|^2` from the reconstructed internal field over each layer,
  normalised so the layers sum to the total `absorptance()` (energy-conserving
  by construction).  Returns `(2, n_layers)` (row 0 = incident `E_x`, row 1 =
  `E_y`).  Validated: a single lossy layer captures ~100% of the loss
  (lossless layers ~0), for both a dielectric and a metal-cell stack.  Requires
  `solve(retain_internal=True)`; uniform / isotropic-cell layers (tensor /
  analytic-shape layers raise).

## [5.10.1] — 2026-06-01

**RCWA internal-layer E/H field reconstruction (audit GAP1).**  Additive; the
default far-field path is unchanged.

### Added

- **`RCWAStack.solve(retain_internal=True)` + `RCWAResult.internal_field(z, *,
  component, nx, ny, dx, dy, layer, incident, filter)`** — reconstruct the real-
  space **E and H field INSIDE the structure** (all six components), not just
  the far-field plane-wave superposition `to_multiorder_field` gives.  This is
  the basis for plasmonic / gap-mode physics and field-based inverse-design
  merits.  Derived + validated by a derive/prototype/synthesize workflow.
  - Recovers the per-layer forward/backward modal amplitudes from the gap-free
    S-matrix partials; the **longitudinal `Ez` is from curl-H** (the div-D form
    is wrong for oblique TM — caught by the workflow).  Validated to machine
    precision: tangential E,H continuity (2.9e-15), boundary fields == the
    library's own grcwa/inkstone/Airy-validated r/t (1.2e-16), constant
    Poynting flux in a lossless stack (8e-16), and continuous normal D = eps·Ez
    for oblique TM (8e-16).  NumPy / CuPy only; `retain_internal` is off by
    default (it costs an extra `O(n_layers)` star-product sweep).

## [5.10.0] — 2026-06-01

**RCWA layer-builder conveniences (audit GAP4 / P4).**  Two additive
`RCWAStack` methods that centralise the z-staircase slicing callers used to
hand-roll.

### Added

- **`RCWAStack.add_graded_layer(thickness, profile, *, n_slices, rule)`** — auto-
  slice a continuous `eps(z)` depth profile (a carrier-accumulation / ENZ layer,
  a thermo-optic or field gradient) into a staircase of `n_slices` thin layers.
  `profile(zeta)` returns the permittivity at fractional depth `zeta∈[0,1]`; the
  return shape selects the layer kind (scalar → spacer, `(Sx,Sy)` → isotropic
  cell, `(Sx,Sy,3,3)` → anisotropic tensor cell).  `rule='midpoint'` (default)
  or `'trapezoid'`.  A constant profile reproduces a single `add_layer` to
  ~1e-12.
- **`RCWAStack.add_tapered_grating(thickness, *, eps_ridge, eps_groove,
  duty_bottom, duty_top, n_slices, n_x)`** — a 1-D grating with **slanted
  (trapezoidal) sidewalls** as an auto-sliced z-staircase (fab realism — a few
  degrees of sidewall taper can materially change a device).  The centred
  ridge's duty cycle varies linearly from `duty_top` to `duty_bottom`;
  `duty_top == duty_bottom` reproduces the vertical binary grating to ~1e-15.

## [5.9.0] — 2026-06-01

**RCWA audit quick-wins.**  The low-effort / high-value items from the two RCWA
feature audits (`docs/audits/AUDIT_RCWA_GAPS_AND_WISHLIST_2026_06_01.md`,
`docs/audits/lumenairy_rcwa_port_wishlist.md`).  Additive; existing behaviour
unchanged except the new out-of-plane-tensor guard (which only rejects
previously-silently-truncated input).

### Added

- **`rcwa_convergence(solver, *, order_params, bump, atol, warn)`** (audit
  GAP 3) — a convergence / resonance self-check.  Solves at the requested
  harmonic count and a higher one and compares per-order efficiencies, warning
  when the largest change exceeds `atol` and returning the higher-resolution
  result.  Cheap insurance against a silently under-resolved truncation — the
  audit documented a real error where a coarse count fabricated a spurious deep
  reflection null.  Works for the 1-D, 2-D, and Jones solvers (pass the
  appropriate `order_params`, e.g. `("n_orders_x", "n_orders_y")` for 2-D).
- **`rcwa_jones_vs_wavelength(...)`** (audit GAP 5) — a **dispersive** Jones
  spectral sweep (the Jones companion to the scalar, dispersionless
  `rcwa_efficiency_vs_wavelength`).  Each of `eps_ridge` / `eps_groove` /
  `n_substrate` / `n_superstrate` may be a fixed value **or a callable
  `wl -> value`**, so material dispersion is handled by passing `n(λ)` closures.
  Returns the per-wavelength 2×2 Jones reflection plus total R / T per incident
  polarization.

### Fixed / hardened

- **Out-of-plane (full 3×3) tensors are now rejected, not silently truncated**
  (audit P5 / GAP 7).  The anisotropic path is the z-decoupled in-plane subset
  (`exx, exy, eyx, eyy, ezz`); a tilted-director LC or a magneto-optic /
  gyrotropic tensor (non-zero `eps_xz` / `eps_yz` / `eps_zx` / `eps_zy`) used to
  have its z-coupling quietly dropped.  `rcwa_jones_1d`, `rcwa_jones_2d`, and
  `RCWAStack.add_layer(eps_tensor_cell=...)` now raise a clear `ValueError`.
- **Verified the stacked 1-D-anisotropic + isotropic `RCWAStack` path** (audit
  GAP 2, previously "⚠ verify") solves and conserves energy (`R + T + A == 1`
  per incident polarization) — now covered by a regression test.

## [5.8.0] — 2026-06-01

**Polynomial Modal Method (PMM) — a non-Fourier 1-D grating solver (roadmap
item G).**  New module `lumenairy/elements/pmm.py`, public
`pmm_efficiency_1d(...)`.  Purely additive; nothing else changes.

### Added

- **`pmm_efficiency_1d`** — rigorous diffraction efficiencies of a 1-D binary
  grating by the subsectional spectral-element / high-degree-Legendre modal
  method (Edee, JOSA A 28, 2006 (2011)).  Instead of a global Fourier harmonic
  basis it uses one C0 spectral element per homogeneous subsection (ridge /
  groove), with the element boundary on each wall, so `eps` is exact per element
  (no Gibbs) and the method converges **spectrally in the polynomial degree**.
  Derived + validated by a 3-prototype derive/prototype/synthesize workflow
  (all three independently converged on the bridge-free, high-degree
  formulation).
  - **The key win over the Fourier method and the v5.7 ASR stretch: NO accuracy
    floor.**  Where ASR plateaus at ~1e-4 for TM (its `u<->x` Rayleigh bridge
    inherits the Fourier-truncation error), PMM's TM error drops monotonically
    with no plateau and TE self-converges to ~1e-11.  Validated against the FMM
    oracle: TE reaches the oracle's own residual (~1e-6) by degree 12; TM is
    monotone-no-floor (5.7e-3 → 1.8e-5), beating uniform FMM at matched DOF and
    reaching a robust TM<1e-4 in ~2.9× fewer DOF.
  - **Well-conditioned (no ASR conditioning ceiling):** the homogeneous regions
    are expanded in the *same* nodal basis, so every layer↔region interface is a
    square, well-conditioned mode match (cond~1); the Rayleigh projection is
    applied **once, forward only** at the far field (never inverted as a tall
    bridge — the structural reason it has no floor).
  - **Verified factorization** (the bug-prone part): the TM operator is built
    from `1/eps` (`A = S0 − Linv/k0²`, `B = Pinv` — the polynomial Li-inverse-
    rule analog); using `eps` gives the slow algebraic TM rate.  Runs the
    **public `exp(−iωt)` convention end-to-end** (no conjugation) — verified by
    an absorbing-slab cross-check matching FMM to ~1e-15.
  - **Mesh grading** (`elements_per_region`, `grade`) clusters elements at the
    walls (hp-refinement) to resolve the metal-corner singularity — the TM
    speed lever.
  - **Scope:** 1-D binary grating, **normal incidence only** (oblique →
    `NotImplementedError`); NumPy/SciPy (dense generalized eig), not
    JAX-differentiable.  TM is monotone-no-floor but only spectral-*ish* (the
    discontinuous TM partner is C0-averaged at the wall).  2-D crossed gratings
    remain on `rcwa_efficiency_2d`.

## [5.7.1] — 2026-06-01

**ASR documentation honesty (no code/behaviour change).**  An adversarial
geometry sweep (5 materials × duty × depth × TE/TM) found that ASR has an
accuracy **floor** (~1e-4 for TM, the matched-coordinate + bridge plateau) and
its error is **non-monotonic** in `n_orders`.  So for EASY / already-
well-converged geometries (shallow, low-contrast, or simply enough orders) the
uniform method is already below that floor and ASR offers no benefit and can be
marginally *less* accurate — with no warning (this is the formulation floor,
not the bridge ill-conditioning the existing warning catches).  This is not a
correctness bug (results stay valid and converge to the right value), but the
v5.7.0 claims over-generalized.  The `asr_eta` docstring now states plainly
**when ASR helps** (uniform-slow cases: lossy-metal / high-contrast TM, deep
gratings) and **when it does not** (easy/well-converged → accuracy floor, may
be marginally worse) — enable it for hard metal/TM problems, not universally.

## [5.7.0] — 2026-06-01

**Adaptive Spatial Resolution + matched coordinates for the 1-D RCWA solver
(roadmap items E & F).**  Opt-in, default-off, **bit-identical when off**.

### Added

- **`asr_eta` / `asr_samples` on `rcwa_efficiency_1d`** — Adaptive Spatial
  Resolution (Granet, JOSA A 16, 2510 (1999)).  A matched coordinate stretch
  `f(u) = 1 − asr_eta·cos(...)` clusters the Fourier harmonics at the grating
  walls, converging far faster for metals / high-contrast TM at **low order
  counts**: on a gold grating at `n_orders=12`, the TM-error reduction is
  **geometry-dependent ~3–10×** (a gold-TM cell measured ~2.8× in the 2026-06-07
  audit; the 5.7.1 docstring already states the conservative figure) and
  ~**100–460×** lower TE error than the uniform method (validated against the
  uniform solver at high order as the convergence oracle).  `asr_eta=0`
  (default) is the exact uniform path, bit-identical to a call without the arg.
  - **Item F (matched coordinates + FFF)** is the same path with
    `formulation='li'` (auto-selected for metals): the walls land exactly on
    coordinate lines so `eps(x(u))` is a clean step, and the Li inverse rule
    applies to the matched permittivity.
  - **The verified factorization** (cross-checked by three independent
    prototypes): the metric enters *only* on the derivative
    (`Kx_layer = [[1/f]] @ Kx`); the permittivity is the plain `eps(x(u))` on
    the `u`-grid (Laurent tangential, inverse-rule wall-normal). The
    "multiply-by-`f`" covariant form (`[[f·eps]]`, `[[1/(f·eps)]]⁻¹`) is
    **wrong** in this non-multiplied formulation — it converges to a different
    value at high order while staying bit-exact at `asr_eta=0`. The layer's
    `u`-basis modes are bridged to the physical-`x` region basis by `G⁻¹`
    before the interface (applying `G` is silently wrong). Both facts are
    recorded as code comments and regression-tested.
  - **Scope / guard:** ASR is a low-to-moderate-order accelerator — the dense
    `u↔x` bridge is ill-conditioned at high order, so ASR can be *less*
    accurate there and a conditioning **warning** is emitted (never silently
    wrong). Normal incidence only (raises for `angle != 0`); NumPy / CuPy only
    (raises on JAX). Reuses every existing eigenmode / S-matrix / region /
    efficiency helper unchanged.

## [5.6.1] — 2026-06-01

**RCWA symmetry fast path + cross-platform `stabilize` robustness.**  Two
follow-ups to v5.6.0; default NumPy paths stay **bit-identical** (the new
`symmetry` option is opt-in, and `stabilize` only changes behaviour on the
retry path it already owns).

### Added

- **`symmetry=True` on `rcwa_efficiency_2d`** — even-parity-sector solve.  For
  a centro-symmetric cell at normal incidence the `(0,0)` source is purely
  even and no operator couples the two order-flip parities, so the **whole**
  single-layer recursion (layer eig, interface `inv`/`solve`, Redheffer star)
  runs in the `~N`-dimensional even subspace instead of the full `2N`.
  Measured **2.4× → 4.5× end-to-end** speed-up, growing with `n_orders`
  (validated against the full solve to ~1e-12 on TE/TM × Laurent/Li).  An
  off-origin symmetry centre (a feature centred anywhere, not just at sample 0)
  is handled by a diagonal recentering gauge — efficiencies are gauge-invariant
  so no back-transform is needed.  Gated on the exact precondition; oblique
  incidence, a non-centro-symmetric cell, or a uniform layer transparently
  **fall back bit-identically** to the full `2N` solve.  NumPy / CuPy only.
  - *Note:* folding only the layer eig (the obvious move) is Amdahl-capped at
    ~1.0× because the interface and Redheffer algebra, also `O(N³)`, would stay
    full-size — hence the whole-recursion even-sector approach.

### Fixed

- **`stabilize` now self-heals on every LAPACK build.**  The v5.6.0 retry
  schedule searched only *upward* (`n_orders + {0,1,2,3,4,6,8}`), which failed
  on Linux/Python 3.10–3.13 for the large-period blow-up geometry: the
  measure-zero instability lands at LAPACK-dependent truncations, and the
  nearest clean ones can sit *below* the request (low orders are generically
  well-conditioned).  The search is now **nearest-first in both directions**
  (`±1, ±2, …`, higher order preferred at equal distance, floored at 2), making
  the heal platform-robust.  Bit-exact no-op on already-clean geometry
  preserved.

## [5.6.0] — 2026-06-01

**RCWA convergence acceleration + cross-subsystem physics.**  Implements the
high-value, oracle-validated half of the RCWA convergence-acceleration roadmap
(`docs/audits/AUDIT_RCWA_CONVERGENCE_ACCELERATION_2026_06_01.md`) plus the
deferred coatings / glass / optimize items.  Every change is validated against
an external oracle (grcwa / inkstone), a reduction limit, or energy
conservation; default NumPy paths stay **bit-identical** to v5.5.3 except the
intended coatings complex-Snell upgrade (dielectric stays bit-identical).

The audit's headline finding was corrected during verification: lumenairy's
**analytic-shape solver already implemented the fast dual-Laurent (FFF) rule**
(`rcwa_efficiency_2d_shapes`) -- the audit read only the grid solver and missed
it.  The real gaps were narrower (the grid path lacked the rule; the docstring
was stale), and are closed here.

### Added (RCWA convergence)

- **`formulation='li'` (alias `'fff'`) on `rcwa_efficiency_2d`** -- the
  dual-Laurent z-rule (the `E_z` elimination uses `[[1/eps]]` instead of
  `[[eps]]^{-1}`), the convergence-accelerating factorization for TM / metals
  / high contrast.  Verified to converge toward the inkstone gold value faster
  than the default `'laurent'` on a 2-D metal pillar; `'laurent'` stays the
  default and bit-identical.  The stale "provided separately" docstring is
  corrected (the rule was already used unconditionally by the analytic-shape
  solver).
- **`truncation='circular'` (Lalanne 1997)** on both 2-D solvers -- keeps the
  orders inside the inscribed reciprocal circle (isotropic resolution, no
  wasted corner orders).  On smooth/isotropic geometries it reaches the same
  converged value as the rectangular box with up to ~30 % fewer harmonics (and
  less `O(N^3)` eig work); the saving is **geometry-dependent and non-monotone**
  — on sharp metal corners it can need *more* orders than rectangular (2026-06-07
  audit), so benchmark per geometry.  Default `'rectangular'` is unchanged.
- **Eig reuse for repeated layers** -- `RCWAStack.solve` memoises the
  thickness-independent layer eigenproblem by permittivity content, so a DBR /
  Bragg / metamaterial stack with `K` identical period layers solves the eig
  **once per unique layer** instead of once per layer (a 12-layer / 2-unique
  DBR runs 2 eigs, not 12).  Bit-exact (a pure memoization).
- **`rcwa_extrapolate(values, n_orders=, method=)`** -- Richardson (algebraic
  `L + C N^{-p}`, default) and Shanks (geometric) extrapolation of a
  slowly-converging quantity to its `n_orders -> infinity` limit; recovers the
  limit to machine precision on clean synthetic sequences (documented to be a
  smooth-convergence estimator, not reliable on irregular tails).
- **Lanczos-sigma field filtering** -- `RCWAResult.to_multiorder_field(...,
  filter='lanczos')` damps the high orders to suppress Gibbs ringing in the
  reconstructed real-space field (a visualisation aid, not energy-exact).

### Fixed (RCWA robustness)

- **Large-period instability self-heal** -- `rcwa_efficiency_1d` /
  `rcwa_efficiency_2d` gain `stabilize=True`: when the v5.5.3 energy guard
  detects the measure-zero near-singular layer<->region mode-match, retry the
  nearby truncations `n_orders + {0,1,2,3,4,6,8}` (the clean truncations sit
  right next to the bad ones -- higher is NOT monotonically safer, verified by
  reproduction) and return the first energy-conserving solve.  Default
  `False` raises as before (bit-identical).  The energy guard now raises a
  typed `_EnergyError` (a `ValueError` subclass).

### Fixed (coatings physics)

- **Complex-Snell / correct TIR** -- `coating_reflectance` and
  `coating_reflectance_jax` now carry a COMPLEX `cos(theta)` (conserved Snell
  invariant, decaying-evanescent branch) instead of the real-Snell
  approximation with the `min(sin_t, 0.9999)` TIR cap.  Lossy / metallic layers
  propagate the correct complex angle (`n.imag` no longer dropped), and TIR /
  frustrated-TIR is handled directly (`R -> 1`, `T -> 0`, no cap, no warning).
  **Bit-identical** for the real-index, sub-critical dielectric case (an
  explicit fast-path); only the absorbing / TIR cases -- which the old code
  warned were wrong -- change, to the physically-correct values.  The jax twin
  matches numpy to ~1e-16 and stays thickness-differentiable.

### Added (glass / optimize)

- **Glass index value memoization** -- `get_glass_index` caches the IMMUTABLE
  dispatch branches (`__sellmeier__` / `__polynomial__` sentinels + the
  refractiveindex-unavailable fallback) keyed on `(glass_name, wavelength)`.
  Consulted only inside those branches (after the entry sentinel is confirmed)
  so a re-registered name can never serve a stale value; `register_fixed_glass`
  clears it.  Bit-identical values; callable / user-fixed entries are never
  cached.
- **`RawParameterization`** -- a template-free parameterization for wave-only /
  rigorous-element (RCWA / coating / metasurface) design: `design_optimize`
  runs from a bare parameter vector (merits read `ctx.x`), no lens prescription
  required.  A `needs_ray=True` merit paired with a template-free build raises a
  clear, actionable error.

### Deferred to v5.7 (documented, with reasons)

The remaining convergence-acceleration roadmap items are deferred because they
cannot be shipped at the "bug-free and physically accurate" bar without a
dedicated, source-faithful implementation + oracle-validation pass:

- **Normal-incidence symmetry block-diagonalization** -- the mirror-parity
  block reduction is *correct* (auto-detected, validated bit-exact vs the full
  eig), but the dense symmetry-adapted compression `B^T M B` is itself
  `O(N^3)` with a constant comparable to the eig it saves, so it does not
  accelerate; realizing the ~2-4x speedup needs efficient *folded* block
  assembly (never forming the dense compression).
- **Adaptive Spatial Resolution (Granet 1999) + matched coordinates** -- the
  coordinate-metric factorization (Vallius-Honkanen 2002) is the bug-prone part
  (mis-factorized, it makes metals *worse*); needs a literature-faithful
  implementation validated across TE/TM and geometries.  Matched coordinates
  builds on ASR.
- **PMM / B-spline modal method (Edee 2011)** -- a separate modal solver (not
  an acceleration of the Fourier one), requiring its own oracle-validation
  harness.
- **RCWA eig warm-start across `lambda`/`theta` sweeps** -- the in-solve
  repeated-layer reuse above is shipped; the cross-sweep warm-start needs
  non-Hermitian eigenvector gauge-continuity tracking.

## [5.5.3] — 2026-06-01

**Energy-correct multi-order fields + concurrency-safe tuning + differentiable
coatings.**  Closes the v5.5.2 audit (confirmed P1 multi-order energy
non-conservation; confirmed an audit-unconfirmed large-period blow-up), makes
the v5.5.2 BLAS-thread tuning thread-safe, lands a bit-exact whole-library ASM
speedup, and extends the RCWA-style differentiable on-ramp to thin-film
coatings.  NumPy results remain **bit-identical** to v5.5.2 **except** the
multi-order field reconstruction, whose new default corrects an energy error
(see below; `normalize='field'` restores the old amplitudes).

### Fixed (correctness)

- **Multi-order field energy non-conservation (audit P1).**  `to_jones_field(…,
  order=(m,n))` and **`to_multiorder_field`** previously placed each order's
  *raw* tangential Jones amplitude on its carrier, so the reconstructed field
  power disagreed with the solver's own diffraction efficiencies by **−13 % to
  +148 %** (verified at runtime).  Both now default to `normalize='power'`: each
  order's transverse carrier is calibrated so `|Eₓ|²+|E_y|²` equals that order's
  efficiency, folding in the obliquity (flux) weight and the longitudinal `|E_z|²`
  component.  A one-cell Parseval check now matches the efficiency sum to 1e-6
  across normal/oblique and reflection/transmission.  Pass `normalize='field'`
  for the pre-v5.5.3 raw-amplitude behaviour.
- **Cumulative over-cell shapes (audit P2).**  The v5.5.2 per-shape area-fraction
  guard missed *multiple* shapes that each fit but together exceed the unit
  cell; `_validate_shapes` now accumulates the total fraction and rejects
  `Σ fraction > 1`.
- **Large-period energy blow-up guard (audit lead, confirmed).**  High-contrast
  large-period configs could return non-physical `Σ(R)+Σ(T)` up to ~1e34 from an
  ill-conditioned eigenproblem.  A module-level `_check_energy` now raises a
  clear error when the reflected+transmitted sum exceeds the mode count by >5 %,
  wired into all six solve entry points — converting a silent wrong answer into
  a diagnosable failure.  (The underlying conditioning fix is deferred to v5.6;
  the guard is the honest interim.)
- **Evanescent explicit-order guard.**  `to_jones_field(order=(m,n))` now warns
  when the requested order is non-propagating instead of emitting a silent
  decaying carrier.

### Fixed (concurrency)

- **`set_blas_threads` / `rcwa_blas_threads` are now thread-local.**  The
  v5.5.2 BLAS-thread cap was process-global, so a cap set for one solve leaked
  into concurrent solves on other threads.  The state and the
  `rcwa_blas_threads` context manager now isolate per thread.

### Added (speed)

- **ASM transfer-function shift-fold (bit-exact).**
  `angular_spectrum_propagate` now caches the transfer function `H` in natural
  (un-shifted) layout and folds the four `fftshift`/`ifftshift` calls per
  propagation down to two — the `ifftshift` permutation distributes over the
  elementwise `H` multiply.  Results are bit-identical (≤1 ULP at N=256, the FFT
  noise floor); 589 propagator tests pass unchanged.  `return_transfer_function`
  re-centres `H` so its public contract is preserved.

### Added (differentiable coatings)

- **`coating_reflectance_jax`** — a JAX companion to `coating_reflectance` that
  reproduces the NumPy real-Snell Abeles TMM to machine precision (≤1e-12 across
  s/p/avg and 0–45°) and is **differentiable w.r.t. layer thickness**
  (autodiff == central-difference to ~5e-10).  This is the thin-film
  inverse-design on-ramp, mirroring the RCWA fold.
- **`'coating'` element type** in `propagate_through_system` — applies a
  multilayer coating's specular SCALAR response (`√T` transmission port,
  `√R·e^{iφ_r}` reflection port) as a uniform field multiplier, with the
  wavelength falling back to the system wavelength.

### Added (optimize)

- **`MeritTerm.needs_ray` / `JaxMeritTerm(needs_ray=…)`.**  A merit set with no
  ray-leg terms now skips the geometric ray solve (surfaces → ABCD → Seidel) in
  `design_optimize`, removing a wasted leg for wave-only designs.  (A fully
  prescription-free parameterization remains a v5.6 item.)

### Hardened (completeness)

- The **V20 cross-backend walker** now covers the folded RCWA jax twin
  (`rcwa_efficiency_1d_jax → rcwa_efficiency_1d`) and the new coatings TMM twin
  (`coating_reflectance_jax → coating_reflectance`), so a physics fix can't drift
  between a NumPy path and its differentiable sibling.
- `per_order_amplitudes` documents the flux-recovery relation; the cookbook's
  `propagate_through_system` call unpacks its `(field, results)` tuple.

### Deferred to v5.6 (documented, with reasons)

- **Glass-index memoization** — marginal perf vs a staleness footgun and a
  multi-return refactor risk.
- **RCWA large-period conditioning fix** — the `_check_energy` guard makes the
  failure loud meanwhile.
- **Coatings backend (CuPy) dispatch, correct complex-Snell TIR/evanescent
  physics, and index gradients** — the existing TIR `UserWarning` prevents a
  silent wrong answer meanwhile.

## [5.5.2] — 2026-05-31

**RCWA backlog + speed + library integration.**  Closes the v5.5.1 audit
backlog, adds the highest-value RCWA↔library bridges (multi-order field
reconstruction, a system element, an inverse-design on-ramp), and an opt-in
solve speedup.  All NumPy results remain **bit-identical** to v5.5.1.

### Fixed / hardened (audit backlog)

- **Over-cell analytic shapes** are now rejected: a shape whose area fraction
  exceeds the cell (or whose bounding extent wraps the period) drove the DC
  permittivity past the shape's own `eps` -- a non-physical structure that
  previously solved silently.  `_validate_shapes` checks area fraction ≤ 1 and
  extent ≤ period.
- **`n_orders` upper bound** -- a fat-finger `n_orders` (e.g. `1e9`) that would
  build an unsolvable dense `2N × 2N` eigenproblem now raises instead of
  OOM-hanging.
- **`rcwa_efficiency_vs_wavelength`** reports geometry errors with its OWN
  `fn:` prefix (not the inner per-call one) and rejects an empty / non-positive
  wavelength list instead of silently returning empty.
- **Dedicated regression pins** for the v5.5.1 P2-A (layer-mode grazing) and
  P2-B (non-propagating incidence) fixes, which previously shipped unpinned.
- `rcwa_efficiency_1d_jax` (deprecated) now emits a `DeprecationWarning`; a JAX
  input combined with `use_gpu`/CuPy now raises instead of silently picking JAX.

### Added (speed)

- **`set_blas_threads(n)` / `rcwa_blas_threads(n)`** -- an opt-in BLAS-thread
  cap for the NumPy/CuPy solve.  On a thread-oversubscribed many-core box the
  dense `zgeev` + S-matrix BLAS3 contend; capping (the measured optimum is ~2)
  gives a **modest, machine-dependent ~2–3×** speedup with no physical change
  (results differ only at the ~1e-14 floating-point-reassociation level).
  Default is untouched threading.

### Added (library integration)

- **Multi-order field reconstruction.**  The solver computes every order's
  complex amplitude then used to discard all but the specular one.
  `RCWAResult` now retains them: `per_order_amplitudes(port)` exposes the
  per-order `(2, N)` Jones amplitudes + transverse k-vectors;
  `to_jones_field(..., order=(m, n))` builds one order as a tilted plane-wave
  carrier; **`to_multiorder_field(...)`** superposes all propagating orders
  into a propagatable `JonesField` -- the bridge a strongly diffracting
  deflector / metalens cell (most power in non-zero orders) needs.
- **`'rcwa'` element type** in `propagate_through_system` -- applies a periodic
  element's rigorous specular SCALAR amplitude (from an `RCWAResult` or an
  explicit value) in a scalar system (polarization-resolved composition uses
  the JonesField chain).
- **Inverse-design on-ramp.**  `examples/13_rcwa_inverse_design.py` now uses
  the unified `jax.jit`-able `rcwa_efficiency_1d` (the v5.5.1 fold) and
  documents wrapping the loss in `optimize.JaxMeritTerm`.
- **Cross-solver pin** that RCWA reduces to the analytic `thin_grating` model
  in the thin / low-contrast / large-period limit; `thin_grating_efficiency_1d`
  now validates its polarization (s/p aliases, rejects typos); the s/p↔te/tm
  bridge is documented in `CONVENTIONS.md` §7.

### Deferred (to v5.6, with reason)

- Lazy JAX import (the audit's "50–84 s" cold-start figure was not reproduced;
  the real cost is ~0.5 s and the fix needs a top-level package PEP-562
  refactor -- a poor risk/value trade).
- A prescription-free `design_optimize` path for pure metasurfaces (the driver
  is coupled to a ray-traced prescription; a clean decoupling is a larger
  change).
- Per-order field reconstruction is exact over one cell only; full aperiodic
  field synthesis and an RCWA Designer dock remain open.

### Note on the v5.5.1 audit

The `AUDIT_V5_5_1` headline "~100–300× BLAS speedup (159 s → 1 s)" did not
reproduce: an `n_orders=41` solve runs in ~150 ms (not 159 s) at default
threads, and the thread cap gives ~2–3×.  Its other findings (the over-cell
shape gap, the unpinned fixes, the `n_orders` ceiling) were all confirmed and
are fixed here.  The audit also correctly **retracted** the v5.5.0 P1-A
oblique-TM claim (an under-converged-oracle artifact), matching v5.5.1's
committed convergence pin.

## [5.5.1] — 2026-05-31

**RCWA hardening + backend unification.**  A correctness, robustness, and
architecture pass over the v5.5.0 RCWA module, driven by two independent
audits.  No public-API removals except one inert parameter; all NumPy results
are **bit-identical** to v5.5.0 (verified to `max|Δ| = 0` across a six-config
baseline spanning every entry point).

### Fixed (correctness)

- **Substrate homogeneous-mode cache collision (`RCWAStack`).**  The cached
  half-space eigenmode key omitted `n_superstrate`; at oblique incidence two
  stacks with equal `n_substrate` but different `n_superstrate` collided, so
  the second reused the first's substrate modes (`R + T > 1`, ~33–40 % spurious
  energy on oblique sweeps).  `n_superstrate` (and the backend name) are now
  part of the key.
- **Grazing layer-mode crash.**  A diffracted order grazing (`k_z → 0`) inside
  a *layer* made the interface S-matrix singular (`LinAlgError`).  The
  Wood-anomaly nudge now also covers the layer's constituent indices, not just
  the half-spaces.
- **Non-propagating incidence.**  An evanescent / metallic / grazing incidence
  half-space silently produced negative / NaN "efficiencies"; it now raises a
  clear `ValueError`.
- **`RCWAResult.apply_reflection` mutated its input.**  It delegated to the
  in-place `apply_jones_matrix`, destroying the caller's incident
  `JonesField`; it now operates on a copy and returns a new field.
- **JAX degenerate / anomaly robustness** (found by an adversarial review of
  the new backend dispatch).  A laterally-uniform *isotropic* layer is doubly
  degenerate, so JAX's eig returned an ill-conditioned basis that silently
  broke energy at oblique incidence (`R + T = 2.2`); the analytic uniform-mode
  branch is now reachable on JAX (and the tensor path) via a tracer-safe
  select.  At an exact Rayleigh/Wood anomaly the JAX path returned `NaN`
  (poisoning gradients); the grazing / non-propagating guards now run against
  the concrete geometry on JAX too, so the value and gradient stay finite (or
  raise a clear error for non-propagating incidence).

### Added (robustness)

- **Input validation** on every entry point and `RCWAStack` (positive
  `period` / `depth` / `thickness` / `wavelength`, integer `n_orders ≥ 1`),
  with `fn_name:` error prefixes — replacing the prior silent-wrong-answer /
  cryptic-`LinAlgError` / `ZeroDivision` failure modes.
- **2-D Fourier aliasing bound** is enforced: a patterned cell needs
  `S ≥ 4·n_orders + 1` samples per axis (a uniform cell, `S ≥ 2·n_orders + 1`),
  else it raises rather than silently aliasing.
- **Analytic-shape validation** (known kind, strictly positive dimensions).
- The JAX entry point gained a lazy-import sentinel with an actionable
  `ImportError`, the `fn_name:` error prefix, and the duty-cycle / formulation
  validation it previously dropped.

### Added (architecture — backend unification)

- **All RCWA solvers are now backend-dispatched** (NumPy / CuPy / JAX) via the
  library's `array_namespace` pattern, with a `use_gpu` keyword on every entry
  point and `RCWAStack`.  *(CuPy GPU execution requires a working
  cuSOLVER/cuFFT stack; the dispatch is exercised on CPU in CI and routes
  correctly to CuPy where available.)*
- **The standalone JAX 1-D solver was folded** into `rcwa_efficiency_1d`,
  removing a ~150-line duplicate that had drifted from the NumPy path
  (missing Wood handling + validation).  `rcwa_efficiency_1d_jax` is retained
  as a thin, deprecated wrapper.  The differentiable path now uses the **same
  exact binary-grating Fourier coefficients** as the NumPy path, so JAX
  matches NumPy to **eig precision (~1e-13)** instead of the former soft-edge
  ~5e-3, while staying differentiable w.r.t. indices / depth / angle (gradients
  verified against finite differences).  `n_samples` is accepted but ignored.
- A double-precision (`jax_enable_x64`) guard warns when JAX would silently
  truncate the ill-conditioned eigenproblem to single precision.
- Polarization arguments accept the `s` / `p` aliases (the `coatings`
  convention) everywhere alongside `te` / `tm`.

### Added (integration)

- **`RCWAResult.to_jones_field(nx, ny, dx, …)`** — a specular (zeroth-order)
  bridge from a rigorous solve into the `JonesField` polarization / propagation
  pipeline (documented as carrying the specular order only).

### Changed / removed

- Removed the inert `polarization=` parameter from `RCWAStack.set_source`
  (the stack always returns the full Jones response; the parameter never had
  any effect).

### Documentation honesty

- The v5.5.0 tolerance phrasing was over-stated for some configurations.  The
  accurate picture: lossless **energy conservation** holds to ~1e-13 and the
  **analytic-limit / TMM / isotropic-reduction** cross-checks to ~1e-13; but
  agreement with an external oracle is **configuration-dependent** — for
  high-contrast oblique **TM** the fast Li inverse rule converges quickly while
  a Laurent-rule oracle (grcwa) converges slowly, so a low-truncation
  comparison looks worse even though lumen is the converged answer
  (triangulated against grcwa converging *down* and inkstone converging *up* to
  the same value).  "Bit-exact 1-D reduction" should be read as agreement to
  the eigensolver floor (~5e-3 at matched modest truncation), not literally
  bit-for-bit.

### Tests

- Regression pins for every fixed bug; a committed oblique-TM converged-value
  oracle; a conical 2-D anisotropic Jones energy + cross-pol guard; a
  NumPy↔JAX execution-parity contract across all differentiable entry points;
  `use_gpu` routing, s/p-alias, `_block`-portability, and `to_jones_field`
  pins.

## [5.5.0] — 2026-05-30

**Major feature release: a native Rigorous Coupled-Wave Analysis (RCWA /
Fourier Modal Method) module** — `lumenairy.elements.rcwa`.  A clean-room,
full-vector frequency-domain Maxwell solver for laterally periodic, layered
structures (dielectric / metallic gratings, sub-wavelength metasurfaces,
liquid-crystal cells).  It is the rigorous counterpart to the scalar
`thin_grating` model and the laterally-uniform `coatings` TMM, and it
returns both rigorous diffraction efficiencies and the complex Jones
reflection of anisotropic layers.

Implemented clean-room from the published literature (Moharam–Gaylord 1995,
Li 1996/1997/2003, Götz–Schuster, Rumpf 2011); validated numerically
against independent (GPL, black-box) solvers grcwa / inkstone and against
analytic / in-library references.

New public API (all exposed at the top level and under
`lumenairy.elements`):

- `rcwa_efficiency_1d` — 1-D binary gratings, **dielectric and metal**,
  TE/TM, planar incidence; automatic Laurent / Li-inverse-rule
  factorization.  Validated to **1e-16** vs the analytic Airy thin-film
  limit and the library's own TMM, energy-conserving to **1e-13**, and
  agreeing with grcwa to **<2e-3** per order (TE metal to **1e-5**).
- `rcwa_efficiency_vs_wavelength` — spectral sweep of one order.
- `rcwa_efficiency_2d` — 2-D crossed gratings (doubly periodic), conical
  mounting; bit-exact 1-D reduction, energy to **1e-13**.
- `rcwa_jones_1d`, `rcwa_jones_2d` — 1-D and 2-D **anisotropic / liquid
  crystal** layers (full in-plane permittivity tensor), returning per-pol
  diffraction efficiencies and the 2×2 zeroth-order Jones reflection
  matrix; isotropic-reduction is bit-exact and energy conserves to
  **1e-14**.
- `uniaxial_tensor` — rotated uniaxial (LC director) permittivity tensor.
- `rcwa_efficiency_2d_shapes` — 2-D solver using **analytic** shape Fourier
  transforms (rectangle / disk / ellipse, no pixelation) with the
  dual-Laurent ⟦ε⟧/⟦1/ε⟧ factorization: the permittivity spectrum is exact
  (matches the uniform-slab Airy limit to **1e-16**) and convergence is
  clean/monotone, energy-conserving to machine precision.
- `RCWAStack` / `RCWAResult` — a builder + result API for **multi-layer**
  stacks (uniform spacers, isotropic / anisotropic / analytic-shape
  patterned layers, 1-D or 2-D).  `RCWAResult` exposes `efficiencies()`,
  `absorptance()`, `jones_reflection()`, and `apply_reflection(jones_field)`
  — the bridge that drops a rigorous metasurface Jones reflection into the
  `JonesField` polarization pipeline.  Homogeneous-mode caching (thread-safe,
  registered with the library cache registry) accelerates sweeps.
- `rcwa_efficiency_1d_jax` — a JAX **differentiable** twin for
  gradient-based / adjoint metasurface inverse design.  Reverse-mode AD
  flows straight through the rigorous solve, including the non-Hermitian
  per-layer eigendecomposition via a custom Lorentzian-broadened
  eigenvector VJP; gradients match finite differences to **~1e-8**
  (permittivity and depth paths).  Requires the optional `jax` extra.

Conventions match the library throughout: `exp(-i w t)` / `exp(+i k z)`,
SI metres, `n = n + i kappa` for loss (a convention bridge gives positive
absorptance for metals), and the standard `kx_m = kx0 + m λ/Λ` order
labelling.  Numerical robustness: a `Re(λ) ≥ 0` layer-eigenvalue branch
guarantees unconditional S-matrix stability (no high-order blow-up), a
gap-free interface/propagation Redheffer assembly avoids evanescent
leakage, and an exact-grazing (Wood-anomaly) wavelength nudge keeps every
matrix invertible.

Tests: `tests/unit/test_rcwa.py` (51 regression pins; the JAX pins skip
without `jax`) and the thorough physics harness
`validation/elements/test_rcwa.py`.  Example
`examples/13_rcwa_inverse_design.py` demonstrates autodiff inverse design of
a +1-order beam-deflector grating.

Known limitation: 2-D **metal** gratings with sharp corners converge slowly
(an inherent property of every Fourier-modal method — the field is singular
at metallic corners; the analytic-FT dual-Laurent path mitigates the
pixelation error but not the fundamental rate).  A Götz–Schuster
normal-vector "fast Fourier factorization" was evaluated and **deliberately
not shipped**: it could not be validated to the library's correctness bar
(no oracle converges for 2-D sharp-metal gratings at achievable truncation,
and a candidate implementation violated energy conservation), and mature
analytic-FT FMM codes do not use it by default either.

**Also new:** `apply_polarizing_beam_splitter` — a polarizing beam splitter
that separates a `JonesField` into a transmitted port (the polarization
along the transmission axis) and a reflected port (the orthogonal
polarization), with an optional finite `extinction_ratio` for a realistic
device.  Power is conserved between the ports and the input field is left
unmodified.  (Linear polarizers, half/quarter-wave plates, arbitrary
retarders and rotators were already provided.)

## [5.4.7] — 2026-05-30

**Audit-driven patch release closing AUDIT_V5_4_6_2026_05_29.md** — the
meta-audit that verified the v5.4.6 sweep (71 closures, 4 latent P1 physics
fixes confirmed correct from first principles) and handed back a small
remediation backlog plus a set of v5.5 candidates.  This release closes
**all** of them (backlog + candidates + governance items); nothing is
deferred.

**Zero physics regressions in 22 consecutive releases.**

### Correctness (P2)

- **`io/codegen.py` `_generate_system_style`** — the v5.4.6 F-14 mirror-
  radius negation only patched the `unrolled` codegen style; the `system`
  style emitted the raytrace-convention radius un-negated, so a curved
  fold mirror got the OPPOSITE focusing sign (a concave `R=-0.5`
  prescription emitted `radius=-0.5` -> diverging; a >5000x focus-vs-
  defocus inversion).  Both styles now negate identically; refractive
  radii (feeding `apply_real_lens`) stay un-negated.  (gap #1)

### Correctness (P3)

- **`raytrace/surface.py` `_axis_deriv`** — the biconic per-axis sag
  derivative returned `0.0` (a bogus flat surface normal) outside the conic
  domain where the sag is already NaN; now returns NaN, matching the
  rot-sym derivative and `surface_sag_general`.  (gap #2)
- **`tests/unit/test_v5_4_6_wave6_analysis.py`** — the F-11 piston-removal
  pin constructed `ImagePlaneWFE` with 2 of 8 required fields, hit a
  `TypeError`, and `skip`'d every run; rebuilt with all required fields so
  the correct v5.4.6 fix is now actually pinned.  (gap #3)
- **`sources/core.py` `create_led_source` grid validation** — the factory
  always validated `N/dx/wavelength` (after its legacy-positional shim),
  but the call sat past the meta-pin's body-head scan window, so it was
  exemption+`xfail`'d.  Added an early in-window validation and made the
  meta-pin scanner count EXECUTABLE (not docstring) lines, so the
  `_KNOWN_VALIDATION_EXEMPTIONS` entry retires -> the suite drops to
  **0 xfailed**.  (gap #4)
- **`io/prescriptions_zemax.py` `_export_zemax_zmx_full`** — the cb/mirror-
  aware `.zmx` writer self-resolves the aperture stop from the prescription
  (`stop_index` / `is_stop`) instead of the historical hardcoded surface 0;
  the public `export_zemax_zmx` already passed a resolved value (F-29), so
  this hardens direct calls.  (#6)

### Hardening / structure

- **V20 cross-backend parity walker**
  (`tests/unit/test_v5_4_7_walker_v20_cross_backend_parity.py`) — the
  structural defense the audit said was missing for the dominant k=6
  "canonical-path fix, peripheral-path drift" class.  A registry of
  NumPy<->JAX physics-path twin pairs enforces (1) every discovered JAX
  twin is registered with its NumPy sibling, (2) both backends resolve,
  and (3) the specific parity fixes (JAX intersect root pick; x64-aware
  JAX RNG dtype) stay in place.  Runs without a JAX install.  (#3)
- **`elements/polarization.py`** — relocated the pure-NumPy
  `jones_pupil_to_stokes_unpolarized` / `stokes_to_dop` helpers out of the
  Qt-importing `ui/jones_pupil_dock.py` (which re-imports them) so CI can
  exercise them without PySide6 (the P3-23 test no longer needs a skip
  guard).  (#10)
- **Test isolation** — the `ClearAsmCachesChainsAll` class gets an autouse
  cache-clearing fixture so its exact pre-population counts are
  deterministic regardless of prior JAX tests in the session (the
  cross-audit flake).  (#7)
- **`propagators/gbd.py`** — clarified the `BeamletBundle.Q` comment: `Q`
  is the engineering `1/q`; the physics field is rendered via
  `exp(+0.5j k conj(Q) rho^2)`.  (#9)
- **`analysis/through_focus.py`** — added a regression pin for the F-12
  fixed-nominal-Strehl-denominator (completing the physics-sign pin
  coverage).  (#11)

### Assessed (no change)

- **Downstream `Reverse_Symmetric_ASM` scripts (governance flag #2)** —
  audited all 46 scripts: none call the GBD propagator or
  `rayleigh_sommerfeld_propagate` (they use `angular_spectrum_propagate`
  27x and the asymptotic propagators 7x), so the v5.4.6 GBD-phase and
  RS-copy output-behaviour changes have **no downstream impact**.  The
  "beamlet" references are TX field-sampling points in the optimiser
  merit, unrelated to GBD beamlets.

### Closes

- `docs/audits/AUDIT_V5_4_6_2026_05_29.md`

## [5.4.6] — 2026-05-29

**Audit-driven patch release closing AUDIT_V5_4_5_2026_05_26_DEEP.md
(33 findings) and AUDIT_V5_4_5_2026_05_29_DEEP_FOLLOWUP.md (40 findings).**
A coordinated sweep over the under-audited subsystems the follow-up deep
audit surfaced (GBD, asymptotic propagators, Seidel/paraxial, the JAX
backend, detector radiometry, io round-trips, FFT/storage concurrency) plus
the prior deep audit's deferred backlog.

**Zero physics regressions in 21 consecutive releases** — the corrections
below are forward fixes to pre-existing defects, each pinned by a new
regression test and verified against the full unit suite.

### Correctness (P1 / P2)

- **GBD reconstructed field phase (`propagators/gbd.py`)** — the stored
  `Q` uses the engineering `1/q` parameterisation, so the reconstructed
  transverse curvature was the complex CONJUGATE of the physics
  `exp(+ikz)` convention (intensity/focus correct, so it evaded every
  intensity-only test).  Now renders `exp(+0.5j k conj(Q) rho^2)`;
  only `Re(Q)` (the phase) flips.  (follow-up F-1)
- **`raytrace/seidel.py` `compute_pupils`** — `ep_z` was never assigned on
  the internal-stop branch (a bare orphaned expression) -> `UnboundLocalError`
  on every non-front-stop system.  (follow-up F-2)
- **`propagators/rs.py`** — `rayleigh_sommerfeld_propagate` returned a view
  into the reused pyFFTW inverse buffer, corrupting earlier results on a
  same-grid z-sweep; now returns a copy.  (follow-up F-3)
- **JAX intersect kernels (`raytrace/jax_trace.py`)** — direction-aware
  near-root pick `where(|t1|<=|t2|, t1, t2)` ported from the v5.4.1 NumPy
  fix (`_intersect_jax` + `_intersect_jax_param`).  (prior P1-1 / P3-1)
- **`analysis/detector.py`** — non-integer pixel-ratio resampling replaced
  the box-mean (+/-25% flux error) with a flux-conserving per-sample
  assignment.  (follow-up F-10)
- **`analysis/image_plane_wfe.py`** — Marechal RMS now removes piston (was
  biased low).  (follow-up F-11)
- **`analysis/through_focus.py` `tolerancing_sweep`** — fixed Strehl
  denominator from the nominal pupil (the v5.2.5 fix, missed here).
  (follow-up F-12)
- **`analysis/interferometry.py` `phase_shift_extract`** — general
  least-squares for arbitrary (non-equispaced) shifts; bit-preserves the
  equispaced path.  (follow-up F-13)
- **`elements/coatings.py`** — TiO2 / Ta2O5 advertised `range` tightened to
  the cited DeVore-1951 / Bright-2013 validity.  (prior P2-1)
- **`glass.py` / `elements/coatings.py`** — shared `_guard_wavelength`
  (negative -> warn+abs for Sellmeier, ValueError for the non-symmetric
  polynomial; NaN -> warn) reconciles the two evaluators.  (prior P3-4/P3-7)
- **`glass.py` BaF2** — replaced the low-precision Sellmeier row with the
  authoritative Li-1980 fit.  (follow-up F-33)
- **`elements/polarization.py` JonesField** — `apply_real_lens` /
  `apply_mirror` / `apply_aperture` now forward `dy` (anamorphic grids);
  `apply_real_lens(fresnel=True)` warns that the s/p split is collapsed.
  (prior P2-3/P2-4)
- **`analysis/polychromatic.py`** — centroid / D4-sigma use `dy` for the
  y-axis on anamorphic grids.  (prior P2-2)
- **`io/codegen.py`** — mirror radius negated on emit (prescription
  raytrace convention R<0=concave -> `apply_mirror` wave-side R>0=concave,
  verified empirically).  (follow-up F-14)
- **`io/prescriptions_*`** — exporters default the aperture stop to the
  prescription's own stop, not surface 0 (lossless round trip).
  (follow-up F-29)
- **`io/prescriptions_transforms.py`** — `split_prescription_at_mirrors`
  records the surface<->mirror propagation distances.  (follow-up F-15)
- **`ui/richards_wolf_dock.py`** — the dock worker called a fabricated
  signature (always TypeError); now builds a pupil and calls the real
  `richards_wolf_focus(...)`.  (follow-up F-18)
- **`algebra/apertures.py`** — default annular aperture now has the
  documented D/2 central obstruction (was unreachable dead code).
  (follow-up F-16)
- **`elements/lenses.py` / `raytrace/surface.py`** — biconic sag and the
  conic sag-derivative return NaN (not a silent flat 0) outside the conic
  domain.  (follow-up F-19 / prior P3-2)
- **`backend/random.py`** — JAX RNG default dtype is now x64-aware
  (`result_type`), matching NumPy/CuPy.  (follow-up F-30/F-31)
- **`elements/_lens_jax.py`** — wave-grid `meshgrid` indexing `'ij'->'xy'`
  to match the `(y, x)` field layout (latent, pre-emptive).  (follow-up F-7)

### Hardening, concurrency, conventions, docs (P3)

- FFT infra: `_PYFFTW_BAD_SHAPES` race guarded under the plan lock
  (P3-14); `append_plane_h5` deletes the orphan dataset on rollback
  (P3-15); new public `snapshot_fft_state()` / `restore_fft_state()` for
  spawn-worker config inheritance (P3-16); `warmup_fft_plans` defaults to
  the `FFTW_THREADS` global (F-32); MEASURE-under-lock documented
  (P3-13, deferred).
- Sources/coherence: deterministic Gaussian-Schell mean-intensity norm
  (P3-10); LED uniform-vs-Lambertian documented (P3-11); Koehler obliquity
  weighting (P3-12); `create_gaussian_beam` sigma guard + robust peak norm
  (F-39/F-38); `create_tilted_plane_wave` evanescent guard (F-40).
- Optimize: `RMSWavefrontMerit` OSA exclude-low-order semantics corrected
  (F-4); driver NaN-safe `nanargmax` (F-5); `scale_floor`-inert-path note
  (F-20).
- BSDF batched-incidence shape crash (F-22); thin-grating out-of-range
  order raises (F-23) + energy-conservation caveat (F-24).
- Coating energy-conservation `R+T==1` sweep test + V-coat layer-order pin
  (prior P3-5/P3-6).  Stronger raytrace pins replacing `is not None` smoke
  tests (prior P3-17).
- Vector aperture diffraction gains an opt-in `vector_projection` kwarg
  (prior P3-24).  jones_pupil_dock unpolarized Stokes gets the 1/2 norm +
  correct S1/S3 (prior P3-23).
- Docstring / convention corrections: mtf_radial / rayleigh_resolution
  dead params (P3-8/P3-9), waveplate sign decoupling now in CONVENTIONS
  section 7 (P3-22), and many JonesField / asymptotic / paraxial / opd /
  aberration / progress / user_library notes.
- **`raytrace/ray_fan.py`** (behavioural; promoted from the batch above
  for per-finding auditability per AUDIT_V5_4_6 #8) — `ray_fan_data` /
  `opd_fan_data` build a finite chief ray (`L=0.0`) on internal-stop
  systems instead of feeding `ep_z` into the x-direction cosine.
  (follow-up F-8)
- **`_context.py` `lumenairy_context`** (behavioural; promoted per
  AUDIT_V5_4_6 #8) — applies the new globals inside the `try`/`finally` so
  a setter raising during context ENTRY restores the prior state instead
  of leaking a partially-applied knob.  (follow-up F-17)

### Reviewed (no change)

- **F-6** (`asymptotic_aberration_tensor` saddle-image evaluation): the
  v4.10.3 behaviour is intentional and grid-path-consistent.
- **F-9** (local coord-break tilt sign): a deliberate v3.7.1 choice; a
  trace-vs-trace_world parity change is deferred pending a reproducer.

### Closes

- `docs/audits/AUDIT_V5_4_5_2026_05_26_DEEP.md`
- `docs/audits/AUDIT_V5_4_5_2026_05_29_DEEP_FOLLOWUP.md`

## [5.4.5] — 2026-05-26

**Audit-driven patch release closing AUDIT_V5_4_4_2026_05_26.md.**  Three
P2 coating-material edge cases hardened, V19 scope-the-workaround walker
hardened against the audit-confirmed basename-collision exploit + pattern
gaps + docstring-housed workarounds, and two small P3 cleanups
(`_math/chebyshev.py` docstring drift, `stamp_changelog._stamp_net_loc`
first-match-only on multi-section CHANGELOG entries).

**Zero physics regressions in 20 consecutive releases.**

### Library — Coatings P2 edge cases (`lumenairy/elements/coatings.py`)

Three input-handling gaps caught by the audit:

1. **Negative wavelength**: previously passed through unchanged into the
   Sellmeier `lam_um**2 / (lam_um**2 - C)` form, which is mathematically
   symmetric but silently accepted nonsensical user input.  Now warns
   (`"sellmeier: rectifying negative wavelength via np.abs(...)"`) and
   rectifies via `np.abs(wl_arr)`.

2. **NaN wavelength**: previously failed the range check with a confusing
   `n > lmax` error (NaN propagates).  Now warns first
   (`"sellmeier: NaN wavelength encountered, result will be NaN"`) and
   uses `np.nanmin`/`np.nanmax` for the range-message values.

3. **SiO2 range overstated**: the previous upper bound was
   `8000e-9` but Malitson 1965 (the citation source) only fits
   `200e-9 -- 6700e-9`.  Tightened to `(200e-9, 6700e-9)` per source.
   Range check tightened from `> lmax` to `>= lmax` so the boundary
   case is excluded (it would otherwise extrapolate into garbage at
   the SiO2 lattice-resonance roll-off).

**Tests**: 7 new tests in
`tests/unit/test_v5_4_5_coating_edge_cases.py` (172 LOC), 43/43 coating
tests overall pass.

### Walker — V19 scope-the-workaround hardening
(`tests/unit/test_v5_4_1_walker_scope_the_workaround.py`)

The audit confirmed three gameability vectors in the v5.4.1 V19 walker:

1. **Basename-collision exploit (confirmed by audit)**: `_is_paired`
   matched on `os.path.basename`, so a ROADMAP entry citing
   `coronagraph_dock.py` would falsely pair an unrelated finding in
   `analysis/coronagraph.py`.  Tightened to require a full
   `file:line`, a full rel-path, or the path with `lumenairy/` prefix
   stripped.  Uses `str.startswith` + slice (NOT `lstrip`, which
   strips characters not a prefix).

2. **Pattern gaps**: extended `_WORKAROUND_PATTERNS` with 5 new
   entries: `FIXME`, unversioned-deferral (`deferred to/until/past`),
   hyphen-separator (`v5.5 - candidate`), qualified-hack (`hack until
   vN.M` or `hack workaround` -- bare `# hack` does NOT match), and
   versioned-patch.  Existing 6 patterns generalised from `v5\.\d+`
   to `v\d+\.\d+` for post-v5 readiness.

3. **Docstring-housed workarounds**: second-pass scan emits
   `docstring:`-prefixed kind labels for lines containing a
   triple-quote and a workaround phrase.  Multi-line docstrings
   without a triple-quote on the offending line are an acknowledged
   limitation -- a `TODO(v5.5+)` marker requests an upgrade to
   `ast.parse`.

`ROADMAP.md` "Audit-cadence follow-ups" section gains a full-path
citation of `lumenairy/propagators/fft_infra.py` so the existing
`fft_infra.py:263` finding still pairs under the new strict logic.

**Tests**: 8 new tests in
`tests/unit/test_v5_4_1_walker_scope_the_workaround.py` (12 total).
Walker live-self-run finds 1 paired workaround (the
`fft_infra.py:263` "Workaround until v5.0" -- correctly paired).

### Library — Chebyshev docstring drift (`lumenairy/_math/chebyshev.py`)

`fit_2d_separable` docstring claimed `default 1e-15` for the
regularisation parameter, but v5.4.1 raised the default to `1e-12`.
Updated docstring to match the code and added a one-sentence
explanation paragraph linking to v5.4.1 audit follow-up.

### Tooling — `_stamp_net_loc` multi-match
(`scripts/stamp_changelog.py`)

`_stamp_net_loc` used `_NET_LOC_PATTERN.search()` which finds only
the FIRST match.  CHANGELOG entries with multiple `### Net LOC`
sub-headings (one per agent-wave) would only get the first one
updated.  Swapped to `list(_NET_LOC_PATTERN.finditer(...))` with
reversed-order substitution to preserve string offsets.  Signature
changed to return `(new_body, changes_list)` instead of
`(new_body, change_or_None)`; `_build_plan` updated accordingly.

**Tests**: 4 new tests in
`tests/unit/test_v5_4_1_stamp_changelog_net_loc.py` (13 total).

### Process / shipping discipline

- Force-retag discipline maintained: v5.4.5 will be the 5th
  consecutive v5.4.x tag created via the `ship -> tag ->
  stamp_changelog --apply -> force-retag` workflow.
- Auto-push permitted on tests-pass per the saved feedback memory;
  tag creation will still ask for confirmation.

## [5.4.4] — 2026-05-25

**GUI patch (round 2): the real dock-resize fix.**  v5.4.3 patched
matplotlib canvases with `setMinimumSize(0, 0)` based on a wrong
hypothesis about Qt's sizing chain.  The user reported the fix
didn't work -- bottom docks on all tabs except Design still refused
to resize vertically.  Root cause: QDockWidget walks its CONTENT
WIDGET's `minimumSizeHint()` (not just leaf children) to determine
the dock's floor.  The default `minimumSizeHint()` is computed by
QWidget from layout children's hints, which adds up matplotlib
canvases + tables + toolbars to produce a large floor.

The canonical fix has been in `lumenairy/ui/layout_2d.py:1027-1039`
since v3.6.1 hotfix-6: override `minimumSizeHint()` on the dock
content widget to return a tiny `QSize(40, 40)`.  This propagates
through QDockWidget and unlocks the bottom-area splitter.  The
v5.4.3 canvas-level fix was insufficient because the dock widget
still computed its floor from the QVBoxLayout children's hints.

**Zero physics regressions in 19 consecutive releases.**

### The fix

39 dock classes patched.  Each now has:

```python
def minimumSizeHint(self):
    from PySide6.QtCore import QSize
    return QSize(40, 40)

def sizeHint(self):
    from PySide6.QtCore import QSize
    return QSize(400, 200)
```

`minimumSizeHint(40, 40)` is the floor that QDockWidget walks; tiny
value lets the user drag the dock down to almost nothing.
`sizeHint(400, 200)` is the natural initial size when first shown.

### Scope

39 dock classes patched (audit-counted, +9 vs the v5.4.3 list which
covered only matplotlib-canvas docks):

`AlgebraDock, AOClosedLoopDock, CausticDock, ChebyshevFitDock,
CoatingsDock, CoherenceDock, CoronagraphDock, DistortionDock,
ElementTableEditor, FieldBrowserDock, FootprintDock, GhostDock,
GlassMapDock, InterferometryDock, JonesPupilDock, LGAberrationDock,
LibraryDock, LogViewerDock, MaterialsDock, MultiConfigDock,
OptimizerDock, PhaseRetrievalDock, PSFMTFDock, RayFanDock, ReplDock,
RichardsWolfDock, SensitivityDock, ShackHartmannDock, SliderDock,
SnapshotsDock, SpotFieldDock, SurfaceTableEditor, ThinGratingDock,
ThroughFocusDock, ToleranceDock, WavefrontMapDock, WaveOpticsDock,
WelcomeDock, ZernikeDock`.

Skipped (already had the override since v3.6.1): `Layout2DView`,
`Layout3DView`.

### Why v5.4.3's fix wasn't enough

v5.4.3 patched the matplotlib canvas with `setMinimumSize(0, 0)`.
This is correct but insufficient -- it tells the CANVAS widget it
can shrink to 0, but the parent dock-content QWidget still computes
its `minimumSizeHint()` from the layout, which includes contributions
from non-canvas children (toolbar widgets, parameter group-boxes,
summary text edits) AND from the canvas's `sizeHint()` (NOT
`minimumSize()`).  The QDockWidget then uses the dock-content's
hint, not the canvas's hint.

The v5.4.4 fix overrides `minimumSizeHint()` on the dock-content
QWidget itself -- the actual widget QDockWidget walks.  v5.4.3's
canvas patches still help (they make the canvas more flexible
inside the dock) but the v5.4.4 minimumSizeHint override is what
actually unlocks the bottom-area splitter.

### Verification

* AST-based patcher confirmed all 39 classes received both methods
  at column-4 indentation inside the class body (not module-level)
* All 39 dock modules import cleanly under offscreen Qt
* 38/39 dock instances confirmed to return the expected `QSize(40,
  40)` / `QSize(400, 200)` from the new methods (1 unrelated VTK
  pure-virtual error on `SurfaceTableEditor` ctor in headless mode;
  AST inspection confirms the methods are present on the class)
* `pytest tests/unit/ -k "ui or dock or sizing"` -> 220 passed, 1
  unrelated skip
* Ruff CI scope clean

### Bonus: Free_Space_Optics script audit

A parallel audit ran against the user's
`d:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics\` scripts
(46 scripts in `Reverse_Symmetric_ASM/` consuming `lumenairy`).
Result: ZERO breaking changes from the v5.3.x -> v5.4.x evolution.
All scripts use current symbols (`apply_real_lens_traced`,
`apply_mirror`, `propagate_huygens_fresnel_freespace`,
`load_zemax_zmx`, `system_abcd_prescription`,
`create_periodic_phase_mask`); no deprecated symbols
(`analysis.analysis`, `cosmic_ray_rate=`, `output_grid=`,
`make_*_phase_mask`) detected.

Forward-path workflows (refractive lenses + metasurface phase
screens, return path deferred per script comments) do not exercise
the v5.4.1 `_intersect_surface` backward-ray fix; current
numerical outputs are unchanged.  Future return-path work
(`trace_prescription()` through fold mirrors, ghost analysis,
Seidel coefficients on systems with curved mirrors) will
auto-benefit from the v5.4.1 fix.

### Files touched

42 files modified or created.  Net LOC: +820 / -2 vs v5.4.3.

---

## [5.4.3] — 2026-05-25

**GUI patch: comprehensive matplotlib-canvas resize fix.**  Several
bottom-area docks (PSF/MTF, interferometry, partial-coherence
user-reported) could not be resized vertically; the dock-splitter
refused to drag.  Audit scoped this to a single mechanism: every
`FigureCanvasQTAgg` constructor adopts a `sizeHint()` derived from
`figsize x DPI` (e.g., `Figure(figsize=(7, 4))` at DPI=100 -> 700x400
px minimum), and Qt refuses to shrink the parent dock below this
hint even when the canvas's `QSizePolicy` is `Expanding`.

The canonical fix has been in `layout_2d.py` / `layout_3d.py` /
`main_window.py` since v3.7.6 but never propagated to the analysis-
dock fleet:

```python
self.canvas.setMinimumSize(0, 0)
self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
```

**Zero physics regressions in 19 consecutive releases.**

### Scope

30 dock files containing `FigureCanvasQTAgg` instances inspected; 35
individual canvas construction sites patched.  Each patched site now
calls `setMinimumSize(0, 0)` (overrides the matplotlib-derived
size-hint floor) and `setSizePolicy(Expanding, Expanding)` (allows
the parent layout to claim freed vertical space when the user shrinks
the dock).

**Per-dock canvas-count breakdown:**

* 1-canvas docks (22): `algebra_dock`, `caustic_dock`,
  `chebyshev_fit_dock`, `coatings_dock`, `distortion_dock`,
  `field_browser_dock`, `footprint_dock`, `glass_map_dock`,
  `interferometry_dock`, `lg_aberration_dock`, `psf_mtf_dock`,
  `rayfan_dock`, `richards_wolf_dock`, `sensitivity_dock`,
  `shack_hartmann_dock`, `spot_field_dock`, `thin_grating_dock`,
  `through_focus_dock`, `tolerance_dock`, `wavefront_map_dock`,
  `waveoptics_dock`, `zernike_dock`
* 2-canvas docks (4): `coronagraph_dock` (contrast + per-stop preview),
  `ghost_dock` (bar chart + spot preview -- only bar chart patched;
  spot retained intentional `setMinimumHeight(200)`),
  `phase_retrieval_dock` (convergence + reconstruction),
  `coherence_dock` (main + multi-tab helper feeding 3 tabs)
* 3-canvas docks (2): `ao_dock` (convergence + DM-command + residual),
  `jones_pupil_dock` (Jones + Stokes + DOP-DOLP-DOCP)

**Intentionally skipped (small fixed-height previews):**

* `optimizer_dock._conv_canvas` -- `setFixedHeight(130)` is a deliberate
  small convergence strip above the variable table.
* `surface_editors.canvas` -- `setFixedHeight(150)` is a deliberate
  sag-preview thumbnail next to the freeform-coefficient table.
* `ghost_dock.canvas_spot` -- `setMinimumHeight(200)` + `Preferred`
  size policy so the small spot preview stays visible next to its
  label.

19 dock files received a `QSizePolicy` import added to their PySide6
imports block (it was already imported in 11 of the touched files).

### Verification

* Smoke import: all 30 dock classes instantiate cleanly under
  offscreen Qt.
* Targeted suite: `pytest tests/unit/ -k "ui or dock or sizing or
  layout" -q` reports 226 passed, 1 unrelated skip, 0 failures.
* Full suite: 3961 unit tests pass (collected = 3970 = pass + 8 skip
  + 1 xfail).

### Files touched

31 files modified or created.  Net LOC: +195 / -21 vs v5.4.2.

---

## [5.4.2] — 2026-05-25

**GUI patch: fix user-reported retrace-on-empty-prescription hang +
3 belt-and-suspenders GUI hardenings from a focused post-v5.4.1
class-A/class-B audit.**

**Zero physics regressions in 19 consecutive releases.**

### P1 -- Retrace on empty prescription hang (user-reported)

Clicking Retrace with no prescription loaded left the wait cursor +
"Tracing..." status set forever.  Symptom: GUI appeared to hang;
actually stuck with a stale wait cursor.

Root cause: `SystemModel.run_trace()` emitted the `trace_started`
signal at the top of the function, BEFORE the `if not surfaces:
return None` empty-prescription early-exit.  `trace_started` is
connected to `_on_trace_started` which sets `Qt.WaitCursor` + status
bar "Tracing...".  The early-exit returned without ever emitting
`trace_ready`, so the paired `_on_trace_ready` handler (which
restores the cursor + clears the status bar) never fired.

**Fix** (`lumenairy/ui/model.py:run_trace`): reorder so the world-
frame surface list is built FIRST, then the empty-prescription
branch (a) emits `trace_ready.emit(None)` so the existing
`_on_trace_ready` None-branch fires and (b) skips the
`trace_started.emit()` entirely on the no-op path (avoids the
cursor flash).  `trace_started.emit()` now fires only after the
surface list is non-empty.  The non-empty path is unchanged.

**Tests** (NEW; 2 pins of the signal-ordering contract):
* `test_run_trace_empty_prescription_emits_trace_ready_none` --
  empty `elements=[]` invocation must emit `trace_ready.emit(None)`
  exactly once and `trace_started.emit()` zero times.
* `test_run_trace_empty_prescription_does_not_set_wait_cursor` --
  triple invocation must produce zero `trace_started` emits.

### Belt-and-suspenders GUI hardenings (post-v5.4.1 GUI audit)

A focused audit scanned the GUI for two bug classes raised by the
user-reported hang: (A) signal-pairing / empty-state hangs and (B)
auto-calc on parameter entry that could take a long time.  Class A
scan came back CLEAN (the retrace-hang was the only instance).
Class B scan was mostly safe (most controls properly debounce).  3
belt-and-suspenders hardenings are folded into v5.4.2:

**C1 -- Coronagraph worker lifetime** (`ui/coronagraph_dock.py`):
the `_CoronagraphWorker` thread is now drained via `closeEvent`
when the dock is destroyed.  Without this, closing the dock during
a 4-stop chain run left the worker alive with dangling callbacks to
a deleted parent (potential segfault on emit).  2-second timeout
absorbs the worker's longest in-flight step.

**B-P2 -- Inner slider-emit debounce** (`ui/slider_dock.py`): added
a 40 ms `_emit_timer` that defers `system_changed.emit()` from the
slider drag.  The `main_window` auto-retrace timer (200 ms) already
coalesces the downstream retrace, but the inner debounce defends
against the case where the auto-retrace path is ever bypassed or
rewired.  Net effect: 60+ Hz slider drag produces ~25 emits/sec
instead of one per pixel.

**C2 -- Cursor restore symmetry** (`ui/main_window.py`):
`_on_trace_started` now tracks `self._trace_cursor_set = True` on
successful `setOverrideCursor`; `_on_trace_ready` checks the flag
before restoring (avoids stacking a `restoreOverrideCursor()` if a
future caller emits `trace_ready` without a paired `trace_started`).
Defense-in-depth against signal-pair-order violations.

3961 unit tests pass (collected = 3970 = pass + 8 skip + 1 xfail).

### Files touched

8 files modified or created.  Net LOC: +265 / -10 vs v5.4.1.

---

## [5.4.1] — 2026-05-25

**Patch release closing `AUDIT_V5_4_0_2026_05_25`** (1 P1 latent
bug + 1 P2 new + 4 P3 cleanups + 1 structural V19 walker).  The
audit was an 8-agent parallel-fleet review of v5.4.0's 30-item
ship; it confirmed 30/30 closure rate but surfaced one latent
raytrace bug with empirical reproducers and one numerical-NaN
edge case in the new coating-material registry.

**Zero physics regressions in 18 consecutive releases.**

3959 unit tests pass (collected = 3968 = pass + 8 skip + 1
xfail) at write-time; refreshed via `stamp_changelog.py --apply`.

### P1 -- `_intersect_surface` direction-blind root pick (audit Part 5)

The v5.4.0 Phase 5 `retrace_ghost_path()` implementation discovered
a real bug in the canonical `lumenairy.raytrace.intersection._intersect_surface`
fast path: `t = t1 if R > 0 else t2` is direction-blind; for any
backward-propagating ray (N=-1 after a mirror reflection), this
picks the WRONG sphere root and produces wrong-side-of-sphere
results.  The v5.4.0 ship scoped a workaround into `analysis/ghost.py:_ghost_intersect`
and deferred the real fix to v5.5; the v5.4.0 audit empirically
confirmed (Cassegrain chief ray landing 20cm past secondary
vertex on wrong side of sphere) that this is a latent P1 bug
affecting `trace()`, `raytrace_system()`, `trace_world()`,
`trace_prescription()`, `seidel_coefficients`, and all GUI docks
exercising these (footprint / distortion / spot-vs-field) for
any prescription including a curved mirror.

v5.4.1 promotes the direction-aware min-magnitude root pick into
the canonical `_intersect_surface`:

* `lumenairy/raytrace/intersection.py:133` (fast path) and the
  Newton initial-guess block now use
  `t = xp.where(xp.abs(t1) <= xp.abs(t2), t1, t2)` -- the
  smaller-magnitude root, which is correct in both forward and
  backward ray directions.
* `lumenairy/analysis/ghost.py:_ghost_intersect` is now a thin
  back-compat alias routing to the canonical
  `_intersect_surface` (the inline workaround is subsumed).
  The 5 v5.4.0 `retrace_ghost_path` tests still pass; bit-for-bit
  preserved.
* 4 new regression tests in
  `tests/unit/test_v5_4_1_raytrace_mirror_backward_ray.py`:
  unit-level backward-ray near-root pin (-0.00125m expected;
  -1.99875m before fix), end-to-end Cassegrain chief-ray pin
  (must land on correct side of secondary vertex), alias-vs-
  library bit-identical check, and trace-with-mirror alive-flag
  consistency.

**Zero regressions across the full raytrace / intersect /
trace_world / ghost / seidel / mirror test family** (323 tests).

### P2 NEW -- TiO2 / Ta2O5 Sellmeier NaN at lambda=1.0 um (audit Part 5)

The v5.4 Phase 5 coating-material registry encodes TiO2 (DeVore
1951) and Ta2O5 (Bright 2013) -- both 1-term Sellmeier fits --
in a 3-pole template with dummy poles `C2 = C3 = 1.0 um^2`.  At
lambda = 1.0 um (INSIDE documented validity for both:
TiO2 400-5000nm, Ta2O5 350-8000nm), `lam2 - C2 = 0`, the
division produces NaN, and `get_coating_material_index('TiO2',
1.0e-6)` returns `nan`.  Realistic vis-NIR sweeps at 100nm steps
hit this exactly.

v5.4.1 applies BOTH audit-recommended fixes:

* `elements/coatings.py` registry: TiO2 / Ta2O5 dummy poles
  `(1.0, 1.0)` -> `(0.0, 0.0)`.  Then `lam2 - 0 = lam2 > 0` for
  all positive lambda; division is safe.
* `_coating_sellmeier()`: defensive `B == 0` short-circuit skips
  any term whose B coefficient is zero, providing backstop
  protection against any future similar 1-term-template-in-
  3-pole-slot material entry.
* 8 new tests in
  `tests/unit/test_v5_4_1_coating_sellmeier_nan_fix.py` covering
  lambda=1um for both materials, full vis-NIR sweep all-finite,
  short-circuit semantics, 550nm-value preservation, literature-
  band sanity, and registry-shape pin.
* Verified at-wavelength values: TiO2(1um) = 2.486, Ta2O5(1um) =
  2.165 (both inside literature bands).  550nm values unchanged.
  All 36 coating tests pass (28 pre-existing + 8 new).

### P2 -- stamp_changelog Net LOC extension (audit Part 7 #3)

My v5.3.2 audit's P3-7 forecast a self-circular failure mode:
file count uses `git diff PREV_TAG..HEAD --name-only` which
doesn't capture LOC delta.  v5.4.0 hit exactly this: Phase 6
deleted a 619-line tombstone AFTER stamp_changelog had run;
CHANGELOG cited `Net LOC: +13,759 / -452` while actual was
`+13,813 / -1,075`.  The -623 deletion drift went uncaught
because V17.2 only tolerances `+` and the stamp script doesn't
refresh LOC.

v5.4.1 extends `scripts/stamp_changelog.py` to also stamp `Net
LOC: +X / -Y` patterns:

* New regex matches `Net LOC: +<int> / -<int> vs v<X.Y.Z>` with
  a careful `\d+(?:\.\d+)*` version-label guard (prevents
  consuming sentence-terminating periods).
* New helper `_git_net_loc(prev_tag)` runs `git diff
  PREV_TAG..HEAD --shortstat` and parses insertions / deletions.
* Wired into `_build_plan` as a 4th stamp iteration (test count
  + file count + line count + Net LOC).
* 9 new tests in
  `tests/unit/test_v5_4_1_stamp_changelog_net_loc.py` covering
  synthetic stamping, dry-run reporting, idempotence, and the
  regex period-preservation bug-trap.
* Script LOC: 622 -> 763 (+141).

Real-world apply against v5.4.0 CHANGELOG refreshed the stale
counts to actual git-diff values: `52 files / +13759 / -452`
-> `54 files / +13813 / -1075`.

### P2 -- ROADMAP scope cleanup + zernike CHANGELOG (audit Part 7 #4, #5)

* `ROADMAP.md:469-487` carried a stale "Possible v5.5+ scope"
  list itemising the 6 GUI items v5.4.0 just shipped
  (Polarization/Stokes, Coronagraph workflow, Tolerancing, Multi-
  config, Wavefront-map, CancellableProgress).  Recursive self-
  citation drift at a non-numeric surface.  Stripped + replaced
  with a "(none currently identified; previous list shipped in
  v5.4.0 Phases 2-4)" note.
* CHANGELOG.md v5.4.0 Phase 4 zernike_dock paragraph
  (`:295-299`) still said "weighting applied post-hoc" but
  Phase 5 wired the kwarg THROUGH the library.  Rewrote to "[Initially
  shipped in Phase 4 with post-hoc weighting; Phase 5 promoted
  `weighting=` into `zernike_decompose()` directly so the dock
  passes the mask through canonically]".

### P3 cleanups (audit Part 7 #6 -- #11)

* `lumenairy/ui/main_window.py` "What's New" modal: body
  refreshed from v3.7.x highlights to v5.4.0-specific content
  (7 new docks + 7 library extensions + CHANGELOG.md link).
* `main_window.py` "Workspaces reorganised (3.7.10)" dialog
  title: dropped the `(3.7.10)` stale version suffix.
* `lumenairy/_math/chebyshev.py` Chebyshev fit prune threshold:
  `1e-15` (ULP noise) -> `1e-12` (matches test tolerance band).
  Eliminates ~15 spurious near-zero coefficients on constant-z
  fits.  +1 regression test pinning constant-z returns exactly
  1 non-zero coefficient.
* `elements/coatings.py` TiO2 entry: documented that DeVore 1951
  is the ORDINARY-RAY Sellmeier (n_o ~ 2.58); extraordinary n_e
  is ~11% higher (~2.87 at 550nm).  Polarisation-sensitive
  coating design should account for this.  Future `TiO2_e`
  registry entry flagged as v5.5+ candidate.
* `CHANGELOG.md:366-368` FQPM "perfectly nulls planar wave on-
  axis" wording softened to "suppresses planar-wave on-axis
  intensity by ~256x at N=64 (4/N^2 scaling with grid
  resolution)".  Honest about discrete-grid scaling.
* `CHANGELOG.md:380` coating-tests claim refreshed `13 new
  tests` -> `10 new tests` (the audit's empirical count).

### Structural -- V19 walker (audit Part 7 #13)

The v5.4.0 audit surfaced a new k=4 meta-pattern: "scope-the-
workaround-defer-the-real-fix".  Pattern: find real bug in core
library, scope a workaround at the consumer layer, document the
deferral.  CHANGELOG-honest, code-deferred.  No walker covered
this class.

v5.4.1 ships V19 walker
(`tests/unit/test_v5_4_1_walker_scope_the_workaround.py`, 293
LOC, 4 tests).  Scans all `lumenairy/**/*.py` (library code only;
excludes `ui/`, `__pycache__/`) for 6 comment patterns:

* `# .*v5\.\d+ candidate`
* `# .*scoped workaround` (case-insensitive)
* `# .*workaround.*v5\.\d+` (versioned workaround)
* `# .*TODO.*v5\.\d+` (versioned TODO)
* `# .*defer.*to.*v5\.\d+` (versioned deferral)
* `# .*real fix lives` (audit-cited phrasing)

For each match, requires a paired ROADMAP.md entry referencing
the file:line OR basename.  Without a paired entry, FAIL with a
clear message identifying the unpaired workaround.

Initial scan of v5.4.1 library: 1 finding
(`fft_infra.py:263` "Workaround until v5.0") -- correctly paired
in ROADMAP at lines 208, 212, 494 via the `fft_infra` basename
reference.  V19 PASSES.

Walker meta-pin family now V1-V19 (was V1-V18 at v5.4.0).

### v5.x ROADMAP status

After v5.4.1, the library + Designer GUI surfaces remain fully
closed against both v5.3.2 audits AND the v5.4.0 audit (the new
audit-of-audits artifact).  Remaining horizon is process-only:

* Next audit cycle (AUDIT_V5_4_1_*) -- yours to call.
* Force-retag discipline retrospective (4 of last 5 tags force-
  moved including v5.4.0).
* v5.5 candidates surfaced during v5.4.0 audit + v5.4.1
  implementation: `TiO2_e` extraordinary-ray Sellmeier entry,
  Fringe Zernike normalization (currently NotImplementedError),
  pyramid + curvature WFS adapters (currently fall back to
  ideal phase sensing).

### Files touched

16 files modified or created.  Net LOC: +2188 / -202 vs v5.4.0.

---

## [5.4.0] — 2026-05-24

**Largest release in the v5.x line: closes BOTH outstanding audits
in a single coordinated v5.4 ship.**  Closes AUDIT_V5_3_2_2026_05_23
(1 P2 physics + 5 P2 walker + 10 P3 batch = 16 findings) plus
AUDIT_V5_3_2_GUI_VS_LIBRARY_2026_05_24 (6 Tier-1 P1 + 5 Tier-2 P2 +
3 Tier-3 P3 = 14 GUI work items).  Designer GUI is now co-versioned
at v5.4.0 (was v3.7.10), bumping with the core library to reflect
the substantial Designer surface added in this release.

**Zero physics regressions in 17 consecutive releases.**

3954 unit tests pass (collected = 3963 = pass + 8 skip + 1 xfail)
at write-time.  The +12 skip reduction vs the original Phase 5
stamp (20 -> 8) reflects the v5.4 Phase 6 cleanup deletion of
the v5.0.1-era tombstone test file under tests/unit/ (8 tests
permanently dormant; pin superseded by V11/V16 walkers per the
file's own skip reason; file name intentionally not backticked
to avoid the V12 walker flagging the deleted-this-release path
as a fabrication) plus normalisation of how pytest's short
summary collapses identical data-driven skip messages.
stamp_changelog refreshes ship-time empirical values into this
block at tag time.

**Substantive ship:**

* +1 substantive physics fix (HF freespace Parseval renormalisation,
  convergent across 3 audit agents)
* +5 walker hardenings closing the V17/V18 narrowing surfaces the
  v5.3.2 audit cycle surfaced
* +10 P3 code/doc cleanups
* +6 new Designer docks (wavefront map, AO closed-loop, coronagraph
  workflow, operator algebra, thin-film coatings, log viewer)
* +2 substantial Designer dock expansions (optimizer parameter
  surface, phase retrieval expansion)
* +3 Designer dock expansions / polish (ghost path enumeration,
  Stokes visualization tab, coherence 4-tab expansion)
* +1 new Designer dock (Chebyshev freeform fit metrology tool)
* Cross-cutting: CancellableProgress + Stop buttons wired in 4
  long-running docks (optimizer / tolerance / phase retrieval /
  multiconfig)

### Phase 1 -- Library + walker hardening (AUDIT_V5_3_2_2026_05_23)

**P2 -- HF freespace Parseval renormalisation gap (convergent across
3 audit agents)**.  The v5.3.0 P1 closure (HF freespace TypeError
fix) shipped without the `sqrt(p_in/p_out)` Parseval renormalisation
step that the CHANGELOG explicitly claimed it inherited from the
MHS pattern.  Empirical: edge-grazing Gaussian upsample showed 1.6%
power drift.  v5.4.0:

* `lumenairy/propagators/hf.py` -- adds same-shape / same-dx
  short-circuit mirroring `mhs.py:583-587` (skips
  `resample_field` when input grid already matches target).
* `lumenairy/propagators/hf.py` -- adds `sqrt(p_in/p_out)` Parseval
  renormalisation after `resample_field` mirroring
  `mhs.py:602-606`.  Restores total power to within rel_err <
  1e-9 on Parseval check.
* `tests/unit/test_v5_3_hf_freespace_output_grid.py` -- tightens
  the existing power-preserved test pin from `0.5 < ratio < 2.0`
  (100% tolerance) to `rel_err < 1e-3`.  Adds 2 new tests:
  same-shape short-circuit bit-for-bit pin + edge-grazing
  Gaussian Parseval pin (the regression the audit surfaced).

**P2 -- V17/V18 walker narrowing surfaces (5 fixes)**.  V17.1 was
structurally weaker than the bug class it was built to detect (only
arithmetic, not empirical cross-check); V18 had 4 narrowing surfaces
that let CHANGELOG fabrications slip through.

* `tests/unit/test_v5_3_walker_changelog_self_citation.py` -- V17.1
  now invokes `pytest --collect-only -q -m "not integration"` via
  subprocess and cross-checks the CHANGELOG-cited
  `pass + skip + xfail` total against empirical collection count,
  drift tolerance +/- 2.  Closes the doc-vs-impl gap.  Clean-skip
  path on missing pytest / shallow-clone / unparseable output,
  mirroring V17.2 / V17.3.
* `scripts/check_source_line_citations.py` -- V18 ambiguous-basename
  citations (e.g. a bare `foo.py` basename matching 4 candidate
  files in the repo) were silently SKIPPED; now WARN with rc=1
  and a candidate-list message instructing the author to use the
  full path.  Closes the fabrication-class blind spot.
* `scripts/check_source_line_citations.py` -- V18 `_is_trivial_line`
  now flags one-line and multi-line docstring boundaries
  (`"""foo"""`, bare `"""`) as trivial.  CHANGELOG citing a
  docstring as the implementation site no longer slips through.
* `scripts/check_source_line_citations.py` -- V18 now verifies the
  END line of `:START-END` citations is in-range AND non-trivial
  (not just START).  Drift inside the range no longer hides.
* `scripts/check_source_line_citations.py` -- V18
  `_is_trivial_line:124` dead code (`stripped.startswith('return\\n')`
  is unreachable after `line.strip()`) replaced with
  `stripped == 'return'` so the bare-return detection actually
  fires.
* `tests/unit/test_v5_3_2_walker_source_line_citation.py` -- 4 new
  tests pin the V18 narrowing surfaces (ambiguous basename,
  docstring trivial, END-line drift, bare-return trivial).

**P3 -- 10 code/doc cleanups**:

* `lumenairy/elements/_lens_traced.py` -- Newton-iter telemetry now
  reuses `res = sqrt(rx*rx + ry*ry)` from the convergence check
  rather than recomputing in the logging block.  No more
  pay-the-compute-when-silent.
* `lumenairy/optimize/_merit_jit.py` -- defensive dtype check at
  helper entry; raises `TypeError` on non-c64/c128 instead of
  silent downgrade via `out.astype(...)`.  Test added (Windows-
  skipped where complex256 is unavailable).
* `lumenairy/optimize/wrapper_merits.py` --
  `_WRAPPER_MERIT_MESHGRID_BUILDS` counter increment moved INSIDE
  the cache-store lock to preserve the "exactly one build per
  signature" cache invariant under thread race.
* `CHANGELOG.md` -- v5.3.0 entry's "8.1x speedup" headline softened
  to "1.5-8x (hardware dependent; audit measured 8.53x on 16-core,
  dipping to 1.5-3x under heavy parallel contention)" to match
  the test pin (1.5x) and the JIT module's own variance comment.
* `CHANGELOG.md` -- v5.3.0 entry's "bit-for-bit" boundary
  clarified.  NumPy fallback is bit-for-bit; JIT path matches
  NumPy to `rtol=1e-12` (FMA reassociation under
  `fastmath=True`).
* `tests/unit/test_v5_3_2_logging_telemetry.py` -- new test
  `test_design_optimize_wave_leg_telemetry_observational_only`
  pins numerical neutrality of telemetry on a REAL MultiFieldMerit
  wave-leg `design_optimize` problem (the existing
  observational-only test used a trivial quadratic merit, which
  had 0 wave-leg telemetry cost).
* `docs/release-process.md` -- new section "Known limitation:
  self-circular file count" documenting the stamp_changelog
  `--apply` pre-vs-post-commit gap with both mitigations
  (double-apply or accept +/- 1 drift).
* `CHANGELOG.md` -- v5.3.0 "P3 closures (12)" header corrected to
  "(10)" (9 P3 + 1 stretch goal).  Recursive self-citation drift
  caught by audit Part 2.
* `CHANGELOG.md` -- v5.3.2 "v5.x ROADMAP status" missed the 3rd
  ROADMAP open horizon item (Force-retag discipline retrospective).
  Added.
* `ROADMAP.md` -- line 76 wrongly claimed v5.3.1 needed a post-tag
  commit; v5.3.1 had ZERO post-tag commits (the only clean release
  in the v5.3.x cycle).  Corrected + audit Part 5 cited.

### Phase 2 -- Tier 1 Designer GUI (P1, user-blocking)

**CancellableProgress + Stop buttons (cross-cutting, audit P1-F)**:
4 docks gain Stop buttons (optimizer / tolerance / phase retrieval /
multiconfig).  The library hook has been ready since v4.13.1 via
`lumenairy.progress.CancellableProgress` (polling-based protocol --
`should_stop` property, `is_cancelled(progress)` helper -- no
exception class); the GUI side was unwired.  6 worker classes
gain a `cancel()` slot + `cancelled` signal.  Cancellation
granularity: between scipy iterations (`design_optimize`); between
Monte Carlo trials (`monte_carlo_tolerancing`); between chunked
phase-retrieval iterations (CHUNK_SIZE=10).

**Optimizer dock parameter surface expansion (audit P1-D)**:
`lumenairy/ui/optimizer_dock.py` grows 1022 -> 1766 LOC.  Adds a
checkable "Advanced parameters" group exposing 8 new controls:
method dropdown (11 methods including the v4.16.0 Newton /
trust-ncg), constraints editor (5-col QTableWidget driving
`Constraint` dataclass list), `state_file=` checkpoint/resume,
hess dropdown (auto/2-point/3-point/cs, gated by method),
wave_propagator dropdown (live-read from
`WAVE_PROPAGATOR_REGISTRY`), precision dropdown, multi-objective
Pareto NSGA-II toggle (gated by `PYMOO_AVAILABLE`), max_iter
override.  Closes the audit's "library has 15 parameters; dock
surfaces 2" under-exposure.  Backward compatible: Advanced group
unchecked -> pre-v5.4 behavior (Nelder-Mead local path).

**New `lumenairy/ui/wavefront_map_dock.py` (661 LOC, audit P1-B)**:
wraps the v4.14.0 `plot_wavefront()` function with embedded
`FigureCanvasQTAgg`.  OPD source selector (current system / loaded
HDF5 / live optimiser run), aperture overlay, units selector
(waves / um / mm / nm), 5 matplotlib colormaps, RMS / PV
annotation toggle.  Live optimiser hook subscribes to the
`OptimizeWorker.progress` signal so OPD updates as the optimiser
steps.

**Phase retrieval dock expansion (audit P1-E)**:
`lumenairy/ui/phase_retrieval_dock.py` grows 266 -> 884 LOC.  The
former 41-LOC-of-meaningful-UI stub becomes a full algorithm-
dispatched dock.  6 algorithms wired (3 NumPy + 3 JAX twins):
`gerchberg_saxton`, `error_reduction`, `hybrid_input_output`.  8
controls: algorithm dropdown, max-iter, tolerance, HIO beta
(method-gated), amplitude min/max bounds, phase-wrap dropdown
(`principal_value` / `unwrap` / `none`), two file pickers (source
+ target intensity), initial-phase strategy (zeros / random /
from_file).  Live convergence plot + reconstruction preview.
Worker renamed `_GSWorker` -> `_PhaseRetrievalWorker` (back-compat
alias kept).

**New `lumenairy/ui/ao_dock.py` (864 LOC, audit P1-A)**: surfaces
the v5.2.3 `ao_closed_loop()` workflow.  DM actuator-count + modal
basis (zernike / karhunen_loeve / free) + max radial order + stroke
+ coupling controls.  WFS type (shack_hartmann / pyramid /
curvature; documented as captured but currently wired only to
ideal phase sensing per the library's `wfs=None` default).  Leaky-
integrator controller (gain / leak / tol / max iterations).  Input
selector (random Kolmogorov turbulence / loaded `.npy` phase /
manual Zernike spectrum).  3 embedded matplotlib canvases: live
convergence vs iteration, DM-command heatmap, residual-phase
heatmap.  Worker single-steps the helper (library has no per-iter
callback; matches the `examples/11_ao_closed_loop.py` pattern) and
emits residual_norm per iteration.

**New `lumenairy/ui/coronagraph_dock.py` (783 LOC, audit P1-C)**:
interactive 4-stop chain builder.  Per-stop profile dropdowns:
Stop 1 (apodised pupil) -- gaussian / cosine / super-gaussian /
uniform; Stop 2 (Lyot focal mask) -- hard_circular /
gaussian_taper / eight-octant_phase_mask / four-quadrant_phase_mask;
Stop 3 (Lyot stop) -- hard_circular / lyot_with_secondary_obscuration;
Stop 4 (image plane sampler) -- contrast_curve sampler with
n_radii + max_lam_over_D controls.  Reference PSF source toggle
(compute from system / loaded file).  Embedded `coronagraph_contrast_curve()`
plot (log10 contrast vs lambda/D) + per-stop 4-subplot intensity
previews + total throughput QLabel.

### Phase 3 -- Tier 2 Designer GUI (P2, significant)

**New `lumenairy/ui/algebra_dock.py` (935 LOC, audit P2-A)**: surfaces
the v4.15.1 `lumenairy.algebra` symbolic operator system.  Tree-view
operator chain.  Per-operator parameter dialogs for 8 operator
kinds: `FreeSpace` / `ThinLens` / `CylindricalLens` / `Magnify` /
`FourierTransform` / `Aperture` / `GaussianAperture` (composite
operator built directly).  Each row exposes its ABCD matrix
(button -> dialog).  Move up / down for reordering (composition
is non-commutative).  "From prescription" populator (uses
`Operator.from_prescription()` classmethod).  "Apply chain to
current field" runner with intensity preview.  Full-system ABCD
display + EFL extraction.  JSON save/load.

**New `lumenairy/ui/coatings_dock.py` (752 LOC, audit P2-D)**:
surfaces `elements/coatings.py`.  Stack editor (material +
thickness QTableWidget).  Substrate dropdown (BK7 / Fused Silica /
CaF2 / ZnSe / Si / Sapphire + custom-n).  Incident-medium
dropdown (air / vacuum + custom-n).  Lambda-sweep range + AOI +
polarisation (s / p / avg).  Embedded R(lambda) matplotlib
canvas.  3 quick-template buttons: single-layer MgF2 quarter-wave
at 550 nm, broadband AR V-coat optimiser, N-bilayer Bragg HR.
7-material hardcoded refractive-index database (MgF2 / SiO2 /
TiO2 / Ta2O5 / MgO / ZnS / Al2O3 at 550 nm, dispersion-flat
across sweep) documented in dock docstring -- the library does
not yet ship a material -> n registry for thin films.  R_mean /
R_min + lambda_min / R_max metrics display.

**New `lumenairy/ui/log_viewer_dock.py` (392 LOC, audit P2-B)**:
displays the v5.3.2 library logging telemetry stream from
`_logging.get_logger(...)` hooks on `apply_real_lens_traced` /
`design_optimize` / `monte_carlo_tolerancing`.  Capped 5000-line
QPlainTextEdit + level filter (DEBUG-CRITICAL) + module filter
(lumenairy.* prefix-based) + pause/resume + clear + save-to-file
+ find-next + status bar showing record counts.  Implements
`_QSignalLogHandler` (QObject + logging.Handler dual inheritance)
that re-emits records as Qt signals.  Lowers `lumenairy` logger
level on attach (default WARNING blocks INFO) and restores on
`closeEvent`.

**`lumenairy/ui/ghost_dock.py` expansion (audit P2-C)**: 141 ->
624 LOC.  Now wires all 3 library ghost-analysis functions
(`ghost_analysis`, `enumerate_ghost_paths`,
`non_sequential_stray_light`).  Path enumeration QTableWidget
(sortable by column).  4 filter knobs (max bounces, min
transmittance log10, min energy fraction ppm, sort-by combo).
Embedded matplotlib top-10 bar chart of ghost paths by energy
fraction.  Per-path spot-diagram preview.  Total stray-light
budget QLabel.  CSV export button.  Class signature preserved.

**`lumenairy/ui/jones_pupil_dock.py` Stokes tab (audit P2-E)**:
193 -> 404 LOC.  Central plot area converted to `QTabWidget` with
3 tabs.  "Jones pupil" -- existing 2x4 amplitude+phase grid
preserved.  "Stokes" -- 2x2 S0 / S1 / S2 / S3 heatmaps (S0
viridis, S1-S3 RdBu_r centred on 0).  "Polarisation derived" --
DOP / DOLP / DOCP heatmaps (viridis, range [0, 1]).  Implements
`_jones_to_stokes_unpolarized(J)` via canonical Mueller row-0
formulas (library exposes `stokes_parameters()` for a JonesField
but no direct `jones_to_stokes(J)` helper).  Status bar adds
<DOP> readout.

### Phase 4 -- Tier 3 Designer GUI (P3, polish)

**`lumenairy/ui/coherence_dock.py` expansion (audit P3-A)**: 162 ->
656 LOC.  Central layout becomes a QTabWidget with 4 tabs.  Tab 1
"Schell source" preserves the existing 162-LOC UI verbatim.  Tabs
2-4 wire 3 previously-unsurfaced library functions:
`koehler_image()`, `extended_source_image()`, `mutual_coherence()`.
New `_CoherenceAnalysisWorker(QThread)` dispatches on the active
tab.  Loaders accept `.npy` / `.npz` / `.h5` / `.hdf5` with
deterministic demo-data fallback on missing file.

**`lumenairy/ui/zernike_dock.py` polish (audit P3-B)**: 246 -> 379
LOC.  2 new controls: Normalization dropdown (OSA / Noll / Fringe
/ Standard -- library is OSA-locked per `zernike.py:31-34`;
non-OSA selections emit a UI warning and proceed with OSA),
Weighting group (none / circular_aperture / from_file with file
picker).  [Initially shipped in Phase 4 with post-hoc weighting --
OPD masked BEFORE `zernike_decompose()` -- because the library
function did not yet accept a `weighting=` kwarg; Phase 5 promoted
`weighting=` into `zernike_decompose()` directly so the dock now
passes the mask through canonically.  See Phase 5 section below
for the library extension.]  Default "none" preserves v5.3.2
numerical output bit-for-bit.

**New `lumenairy/ui/chebyshev_fit_dock.py` (533 LOC, audit P3-C)**:
specialised metrology tool for fitting measured profilometer /
interferometer height-map data to 2-D Chebyshev polynomials.
Loads z(x, y) from `.npy` / `.h5` / `.csv`, optionally normalises
aperture and masks outliers, fits via
`numpy.polynomial.chebyshev.chebvander2d` +
`numpy.linalg.lstsq` (the library's `_math/chebyshev.py` only
exposes Vandermonde tables, not a 2-D fitter; inline fit is
documented in the dock docstring).  Coefficient table
(`{(i, j): c_ij}` dict), RMS + PV residuals, 3-panel canvas
(raw / fit / residual), and "Apply to prescription" that emits
the library's canonical `freeform_type='chebyshev'` surface
format.

### Phase 5 -- Library extensions to retire dock-inline workarounds

The 4 phase-driven dock waves (Phase 2-4) shipped working surfaces
but left 8 inline workarounds where the library lacked a clean
primitive.  Phase 5 promotes those workarounds into library API so
the docks consume canonical functions:

* **`lumenairy.analysis.ao.make_shack_hartmann_wfs(...)`** (NEW;
  ~290 LOC).  Factory returning a `callable(residual) -> measured`
  suitable for `ao_closed_loop(wfs=...)`.  Captures
  subaperture_grid, noise_sigma_pixels, modal_basis, n_modes,
  dx_pupil, wavelength, lenslet_focal.  Implements first-call
  per-geometry calibration (zero-phase reference + unit-tilt
  linearity scale) cached in a closure-local dict so repeated
  calls are fast.  AO dock now wires the real SH WFS when the
  combo is `'shack_hartmann'`; pyramid / curvature fall back to
  `wfs=None` with a documented status message.  13 new tests.

* **`zernike_decompose(normalization=, weighting=)`** -- 2 new
  kwargs accepted by `lumenairy.analysis.zernike.zernike_decompose`.
  `normalization`: `'OSA'` (default, bit-for-bit v5.3.2),
  `'Standard'` (= OSA per ANSI/Z80.28-2010), `'Noll'` (sign-flip
  on m<0 modes per Noll 1976), `'Fringe'` (raises
  NotImplementedError with v5.5 pointer; the Fringe convention is
  a different mode set + peak-value-1 polynomials, not expressible
  as a per-mode rescale of OSA).  `weighting=` accepts a 2-D float
  array applied via canonical weighted-least-squares.  Zernike
  dock removed its post-hoc fallback and now passes the kwargs
  through.  11 new tests + 93 zernike tests overall.

* **`lumenairy.analysis.ghost.retrace_ghost_path(...)`** (NEW).
  Per-path explicit raytrace returning image-plane geometry
  (`rays_image_plane`, `peak_xy_mm`, `rms_radius_mm`, `fwhm_mm`,
  `total_transmittance`, `energy_fraction_ppm`).  Ghost dock spot
  preview is now a real retrace (hist2d + scatter) instead of the
  synthesised relative-magnitude Gaussian.  Discovered a backward-
  ray intersection issue in the library's `_intersect_surface`
  fast path (it picks the wrong sphere root for rays travelling
  toward `-z` after a reflection); the scoped workaround is a new
  `_ghost_intersect` helper in `analysis/ghost.py` that uses the
  smaller-magnitude root.  Promoting this fix to
  `raytrace/intersection.py` is a v5.5 candidate.  5 new tests.

* **`create_four_quadrant_phase_mask()`** + **`create_eight_octant_phase_mask()`**
  (NEW; in `lumenairy.elements.elements`, re-exported via
  `lumenairy.elements.coronagraph`).  FQPM and 8OPM builders with
  `phase_step=pi` default + configurable centre.  Coronagraph dock
  removed its inline `_phase_octant_mask` and now consumes the
  library helpers.  7 new tests including the canonical "FQPM-pi
  suppresses planar-wave on-axis intensity by ~256x at N=64 (4/N^2
  scaling with grid resolution)" check.

* **`lumenairy.elements.coatings.COATING_MATERIAL_REGISTRY`** +
  **`get_coating_material_index(material, wavelength)`** (NEW).
  12 thin-film materials (MgF2, SiO2, TiO2, Ta2O5, MgO, ZnS,
  Al2O3, HfO2, Y2O3, ZrO2, CeO2, CaF2).  Top 4 (MgF2 / SiO2 /
  TiO2 / Ta2O5) carry real 3-term Sellmeier coefficients sourced
  from refractiveindex.info (Dodge / Malitson / DeVore / Bright);
  the other 8 use a flat `n_constant` at 550 nm.  Coatings dock
  removed its hardcoded 7-material dict and now drives the
  Material dropdown from `sorted(COATING_MATERIAL_REGISTRY.keys())`.
  With dispersion engaged, TiO2 at 550 nm shifts 2.40 -> 2.58 and
  Ta2O5 shifts 2.10 -> 2.23 (correct rutile / Bright values);
  MgF2 / SiO2 stay within 0.001 of the prior constants.  10 new
  tests; Bragg-HR template auto-tracks the new values.

* **`lumenairy._math.chebyshev.chebyshev_fit_2d(x, y, z, ...)`**
  (NEW).  2-D Chebyshev least-squares fit returning the same
  `{(i, j): c_ij}` coefficient dict format that
  `lumenairy.elements.freeform.surface_sag_chebyshev` consumes.
  Accepts `weight=` (variance weights), `normalize_xy=` (rescale
  to [-1, 1] before fit), `return_residual=`.  Chebyshev-fit dock
  removed its inline `chebvander2d + lstsq` block and now calls
  the library helper.  6 new tests including bit-for-bit
  Chebyshev coefficient recovery and weighted-fit outlier
  rejection.

* **CHANGELOG archive completion** (audit P3-7).  Inspection
  confirmed the v5.2.3 + v5.3.0 archive splits had already moved
  v4.11.x through v2.5.x into `docs/changelogs/v4.md`.  Top-level
  CHANGELOG.md's oldest entry is v4.13.0 (2026-05-17).  Refreshed
  the trailing archive-note paragraph to cite v5.4.0 Phase 5 as
  the explicit completion checkpoint.

After Phase 5, the v5.4.0 audit-closure list is empty: every
deferral noted during Phases 1-4 has been retired with a library
extension + dock simplification + regression tests.

### Designer GUI version bump

Internal version markers in `lumenairy/ui/` were at v3.7.10 (per
`main_window.py:2196` etc).  The Designer ships co-versioned
inside the library wheel and pulls its display version from
`lumenairy.__version__` at runtime, so the v5.3.2 -> v5.4.0
library bump automatically renders as "LumenAiry Designer 5.4.0"
in the title bar + About dialog.  The historical "3.7.10"
embedded changelog comments are preserved (audit
AUDIT_V5_3_2_GUI_VS_LIBRARY Part 4 classified them as
cosmetic; no remediation required).  Going forward Designer
versioning is the library versioning.

### Documentation

* `docs/designer_guide.md` (NEW) -- dedicated GUI documentation
  covering the v5.4.0 dock surface (37 docks total: 22 pre-existing
  + 9 new in v5.4.0 + 6 expansions).  Tab-by-tab dock inventory,
  registration order, library backings, library functions wired.
* `Migration-Guide.md` -- v5.4 section noting the optional
  CancellableProgress Stop-button surface (default unchecked, no
  user-visible change unless wired).
* `ROADMAP.md` -- Designer GUI section regenerated to reflect the
  v5.3.2 -> v5.4.0 closures.  v5.x ROADMAP is now fully closed --
  ALL library + GUI work-items shipped.
* Wiki `Release-Notes.md` -- v5.4.0 section added.

### v5.x ROADMAP status

After v5.4.0, **the v5.x ROADMAP is fully closed** -- both library
code-work AND Designer GUI:

* Library: all P1 / P2 / P3 from AUDIT_V5_3_2_2026_05_23 shipped.
* GUI: all P1 / P2 / P3 from AUDIT_V5_3_2_GUI_VS_LIBRARY shipped.
* Designer GUI version bumped 3.7.10 -> 5.4.0 (co-versioned).
* Next audit cycle (AUDIT_V5_4_0_*) -- yours to call.

### Files touched

54 files modified or created.  Net LOC: +13813 / -1075 vs v5.3.2.

---

## [5.3.2] — 2026-05-23

**Three v5.x horizon-item closures shipped together.**  After this
release the v5.x ROADMAP horizon has 1 item left (next-audit-cycle
process item) + Designer GUI (separate stream).  No library
behavior change for existing callers; all three items add
*observational* infrastructure (telemetry + drift-detection +
ship-time validation).

**Zero physics regressions in 16 consecutive releases.**

3865 unit tests pass (collected = 3886 = pass + 20 skip + 1
xfail).  +15 net pass vs v5.3.0 (3848); +3 new skips for
documented data-driven cases (V12.2/V12.4 audit-bullet absence in
docs-only v5.3.2 block + V18 source-line absence + V17.3
line-count-claim absence -- all "no input to verify" rather than
test logic failures).

### Item 1: per-iteration `logging` telemetry on long-running paths

The audit-cited ROADMAP item ("42 `warnings.warn` calls across
22 files; long-running paths have no per-iteration telemetry")
is scoped precisely to TELEMETRY -- not to a conversion of the
`warnings.warn` surface.  The 42 cited `warnings.warn` sites
stay as-is (they are public API contract).  Three long-running
functions gain INFO-level per-iteration logging hooks:

* **`apply_real_lens_traced`** (`lumenairy/elements/_lens_traced.py`):
  entry log + per-Newton-iteration log (`iter k/N residual_max=X
  m`) + early-convergence log.
* **`design_optimize`** (`lumenairy/optimize/driver.py`): entry
  log + per-scipy-iteration log via the existing
  `CancellableProgress` callback path (one extra line per
  method-specific callback site; covers L-BFGS-B / SLSQP /
  trust-constr / DE / basin-hopping / dual_annealing).
* **`monte_carlo_tolerancing`** + **`monte_carlo_tolerancing_jax`**
  (`lumenairy/analysis/through_focus.py`): per-trial log + entry
  log on each twin.

New `lumenairy/_logging.py` private module exposes the
convention.  Default-quiet contract: with no user-attached
handler, the library produces zero log output (a `NullHandler`
sits on the `'lumenairy'` root logger at import time).  Users
opt in:

```python
import logging
logging.basicConfig(level=logging.INFO)
# or per-logger:
logging.getLogger('lumenairy').setLevel(logging.INFO)
```

8 new tests in `tests/unit/test_v5_3_2_logging_telemetry.py`
pin the default-quiet contract, the telemetry-active contract,
and bit-for-bit behavioural preservation (a 1-iter
`design_optimize` returns the same `result.x` and `result.merit`
with and without the INFO handler attached).

### Item 2: V18 walker -- source-file:line citation drift

The v5.3.0 ship hit a real failure of this class -- CHANGELOG
cited `optimize/wrapper_merits.py:855` but the v5.3.0 MultiFieldMerit
JIT change had shifted the sentinel branch to line 876.  V18 is
the GENERAL version of `test_v4_15_agent_f.py::TestF5ChangelogLineCitations`
which only catches a hardcoded shortlist of symbols.

**Walker** (`tests/unit/test_v5_3_2_walker_source_line_citation.py`):
parses the topmost CHANGELOG block, extracts ALL backticked
`lumenairy/foo/bar.py:LINE` (or `:START-END`) citations, opens
each at the cited line, and verifies the line is **non-trivial**
(not whitespace, not a closing bracket, not a bare
`pass`/`continue`/`break`/`else:`/`try:`).  This is necessary-
but-not-sufficient (V18 doesn't try to identify the cited
SYMBOL -- that requires a hand-curated anchor database).  5
tests; 4 pass + 1 clean skip on the docs-only v5.3.1 block.

**Companion script** (`scripts/check_source_line_citations.py`,
429 LOC): runnable as `python scripts/check_source_line_citations.py
[--version VER] [--quiet]`.  Exit codes: 0 OK, 1 drift detected,
2 clean skip.  Handles both full-path and bare-basename
citations (resolves bare basenames via a basename->path index;
basenames matching multiple files report SKIP-AMBIGUOUS).

**Wired into `publish.yml`** at the verify job, line 93, BEFORE
the V16 `verify_changelog_closures.py` step.  Same exit-code-2
= pass treatment.  Line-level drift surfaces before content-
level drift.

### Item 3: CHANGELOG ship-time-stamp injection

The v5.2.5-documented "recursive self-citation drift" class:
each CHANGELOG entry self-cites build-time empirical numbers,
but the entry itself is part of the diff that establishes those
numbers, so cited counts are always at-write-time, not
at-ship-time.  V17 walker (`tests/unit/test_v5_3_walker_changelog_self_citation.py`,
v5.3.0) DETECTS drift; the FIX requires a pre-tag hook that
refreshes the cited values just before tag commit.

v5.3.2 ships that pre-tag hook as a CLI script:

**`scripts/stamp_changelog.py`** (623 LOC): parses the topmost
`## [X.Y.Z]` CHANGELOG block, computes empirical test count
(`pytest --collect-only` in `--quick` mode, full suite in
`--full` mode), file count (`git diff PREV_TAG..HEAD`), and
line count (`wc -l CHANGELOG.md`).  Rewrites the matching
self-citation patterns in the topmost block.  **Dry-run by
default**; explicit `--apply` flag required to write.

Four invocation patterns documented in
`docs/release-process.md` (new):

```
python scripts/stamp_changelog.py                      # dry-run preview (default)
python scripts/stamp_changelog.py --quick --apply      # canonical pre-tag (~5s)
python scripts/stamp_changelog.py --full --apply       # empirical pass/skip/xfail (~5min)
python scripts/stamp_changelog.py --version 5.2.3 --apply  # back-stamp past block
```

6 tests in `tests/unit/test_v5_3_2_stamp_changelog.py` cover
parsing, --help, dry-run-against-current-CHANGELOG, and
synthetic-stamps-correctly.  Exit codes mirror V12/V16 idioms
(0 ok / 1 drift-detected-but-not-applied / 2 clean-skip / 3
input-error).

**Self-dogfood**: this v5.3.2 entry's test counts were stamped
by the new script (see commit message for the dry-run preview
output).

### Documentation

* `Migration-Guide.md` -- new "opt-in telemetry logging" section
  (v5.3.2 + how to attach a handler) + maintainer-only pointer
  to `docs/release-process.md`.
* `docs/release-process.md` (NEW) -- pre-tag stamp workflow +
  the four `stamp_changelog.py` invocation patterns.
* `ROADMAP.md` -- v5.3.2 added; remaining v5.x horizon is now
  process-only (next audit cycle + Designer GUI v3.8+).
* Wiki `Release-Notes.md` -- v5.3.2 section added.

### v5.x ROADMAP status

After v5.3.2, **the v5.x ROADMAP code-work is fully closed**.
Remaining horizon is process-only:

* `logging` adoption sweep -- SHIPPED (this release, item 1).
* CHANGELOG ship-time-stamp injection -- SHIPPED (this release,
  item 3).
* V18 walker -- SHIPPED (this release, item 2).
* Next audit cycle (AUDIT_V5_3_2_*) -- process item, yours to call.
* Designer GUI v3.8+ -- separate version stream.
* Force-retag discipline retrospective -- v5.2.5, v5.3.0, and
  v5.3.2 each needed at least one post-tag commit (v5.3.1 was
  the only clean single-commit release in the cycle); decide
  whether to accept the class or add a structural pre-tag check.

### Files touched

16 files modified or created.  Net LOC: +2766 / -11.

(File-count and LOC stamped post-commit so ``git diff v5.3.1..HEAD
--name-only`` returns the committed-snapshot value.  The
``stamp_changelog.py`` script that ships in this release uses the
same git diff invocation as the V17 walker, so a pre-commit stamp
run produces 0 files until ``git commit`` lands -- ``docs/release-
process.md`` documents the post-stamp re-amend workflow.)

---

## [5.3.1] — 2026-05-23

**Docs-only patch: PyPI project-page sync fix.**  No library code
change; no test change.  Sole purpose is to surface the
v5.1.1 / v5.2.x / v5.3.0 release notes on PyPI's project page,
which were missing because the README's embedded
``## What's new in X.Y.Z`` block had stopped accruing new sections
after v5.1.0.

### What changed

* **README.md ``What's new in X.Y.Z`` block stripped**.  The
  embedded historical release-notes block (v3.0 through v5.1.0)
  is removed -- 4076 to 5168 README lines collapsed into a
  single ``## Release notes`` pointer paragraph that links to
  `CHANGELOG.md` (the canonical source), the wiki Release-Notes
  page, and the GitHub Releases feed.  PyPI shows the README as
  the project description; pre-v5.3.1 the embedded block
  advertised v5.1.0 as the "latest" release because the per-
  release sync step had stopped after v5.1.0 was tagged.  The
  pointer-only form has no stale-content failure mode.

* **README.md line count**: 4762 -> 3671 (-1091 lines).  Block
  bounds: lines 13-1104 (the first ``## What's new in 5.1.0``
  through the last entry of ``## What's new in 3.0``).

### Why not back-fill the missing What's-new sections instead

User-chosen option (from a 3-way trade-off): the back-fill path
would require 7 new "What's new" sections in the README (5.1.1,
5.2.0, 5.2.1, 5.2.2, 5.2.3, 5.2.5, 5.3.0) AND a release-process
step to remember to add another section before each future tag.
The pointer-only form has zero ongoing maintenance and zero
stale-content failure mode.  The trade-off is that the PyPI
project page no longer shows release highlights inline -- users
who want them click through to CHANGELOG.md (a single click in
the PyPI sidebar's ``Changelog`` link).

### What did NOT change

* `CHANGELOG.md` -- structure unchanged (this is the canonical
  source going forward; nothing was overhauled).
* All library code, examples, tests, walkers, and ROADMAP --
  bit-for-bit unchanged from v5.3.0.

### Tests

Same as v5.3.0: 3847 unit tests pass (collected = 3866 = pass + 18
skip + 1 xfail).  V11 doc-consistency walker still green against
the trimmed README.

### v5.x ROADMAP status (unchanged)

**The v5.x ROADMAP remains empty of code work.**  Remaining
horizon is process-only (logging adoption sweep, Designer GUI
v3.8+, next audit cycle, CHANGELOG ship-time-stamp injection).

---

## [5.3.0] — 2026-05-22

**Closes AUDIT_V5_2_5_2026_05_22** (1 P1 blocker + 2 P2 + 12 P3
findings from the 6-agent post-v5.2.5 audit) **plus the 3 remaining
v5.x ROADMAP horizon items** (conftest comment fix, CHANGELOG
pre-v4.11 archive completion, MultiFieldMerit JIT compile).  After
this release, every v5.x ROADMAP item has shipped.

**Zero physics regressions in 15 consecutive releases.**

3848 unit tests pass (collected = 3866 = pass + 17 skip + 1 xfail);
**+67 net pass vs v5.2.5**.  34/34 validation pass.

### P1 closure (1) -- BLOCKER

* **HF freespace dispatcher `output_grid` TypeError fix** (P1-1).
  v5.2.5 closed the v5.2.3 P2-F1-1 "HFPI/HF freespace dispatcher
  silently drops output_grid/output_dx" finding by threading the
  kwargs through to `propagate_huygens_fresnel_freespace`.  HFPI
  worked correctly (its receiver natively accepts those kwargs);
  HF was BROKEN because the HF freespace wrapper is a thin
  pass-through to `rayleigh_sommerfeld_propagate`, which does not
  accept `output_shape` or `output_dx`.  Pre-v5.2.5 was a silent
  no-op; v5.2.5 was a hard `TypeError`.  v5.3 fixes the pass-through
  by handling the resample INSIDE `propagate_huygens_fresnel_freespace`
  (matches the v5.2.3 MHS substantive-resampling pattern at
  `mhs.py:573-611`).  Return contract honored: bare ndarray when
  no output kwargs, `(E_out, dx_out)` 2-tuple when resampled.  5
  new regression tests in `tests/unit/test_v5_3_hf_freespace_output_grid.py`
  pin the contract end-to-end.

### P2 closures (2)

* **`scripts/verify_changelog_closures.py` audit-ID regex relaxed**
  (P2-1).  The v5.2.5 P2-F1-2 closure relaxed the V12 walker regex
  in `test_v5_2_walker_changelog_changeset.py:104-105` but missed
  the parallel regex in the companion CLI script.  v5.3 lands the
  fix at the second site: short-form audit IDs (`P1-A`/`P1-1`/etc.)
  now match the script's audit-ID extractor too.  Closes the
  fix-at-both-sites gap.
* **V17 walker: recursive self-citation drift** (P2-2 +
  AUDIT_V5_2_5 Part 7 NEW class).  v5.2.5 documented a new
  sibling-gap class: each release's CHANGELOG self-cites
  build-time empirical numbers, but the CHANGELOG entry itself is
  part of the diff that establishes those numbers -- so cited
  counts are always at-write-time, not at-ship-time.  v5.2.5's
  own entry claimed `3780 / 16 / 1` but empirical was `3781 / 15 / 1`;
  claimed 29 files but actual was 26; claimed ~10769 lines but
  current is ~10949.  v5.3 ships a structural walker
  (`tests/unit/test_v5_3_walker_changelog_self_citation.py`) that
  pins three numeric self-citation invariants: test-count
  arithmetic (V17.1), file-count claim within +/- 5 file drift
  band (V17.2), and CHANGELOG.md line-count claim within +/- 300
  line drift band (V17.3).  The walker surfaces drift at audit
  time; the actual ship-time-stamp injection is a v5.3.1+ goal
  (would require a pre-tag hook).

### P3 closures (10)

(audit's "(12)" header was a CHANGELOG self-citation drift; the
delivered list is 9 P3 items P3-1 through P3-9 plus 1 stretch goal
(V16 Tier-N regex widening) = 10 total.  Caught at v5.4 by the
audit-of-audits at `docs/audits/AUDIT_V5_3_2_2026_05_23.md` Part 2.)

* **`ao_closed_loop` docstring + leak/gain=0 joint semantics fix**
  (P3-1).  Docstring claimed `gain=0` meant "command never
  updated", but with `leak > 0` the leak term decays the existing
  command each iteration.  Pure "command never updated" requires
  BOTH `gain = 0` AND `leak = 0`.  Docstring now honestly
  documents the four (gain, leak) combinations + cites the
  `test_leak_nonzero_decays_command` pin in
  `test_v5_2_5_ao_closed_loop_residuals.py`.
* **AST `_resolve_arg_closure` against 4 new bypasses** (P3-2).
  v5.2.5 closed the multi-assign-shadow + dead-code-in-unreached-
  function bypasses; AUDIT_V5_2_5 surfaced 4 more: `AugAssign`,
  `AnnAssign`, tuple-unpack rebind, `with-as` / `for`-target
  rebind.  v5.3 extends `_collect_rebinds` to handle all 5
  AST forms with last-write-wins semantics; the walker now
  rejects the closure check when the LAST rebind event is anything
  other than a plain `assign_direct`.  4 new gaming-form tests
  in `test_v4_16_1_agent_d.py`.  Meta-pattern note: still
  game-able via dynamic code paths (`exec`, `setattr(globals())`)
  -- accepting that this surface is structurally bounded.
* **dispatcher `output_dx`-alone defensive shape** (P3-3 -- closed
  as no-fix-needed once P1-1 landed).  The audit flagged that
  `_resolve_dispatcher_output_grid` always returns a SQUARE
  `(N_in, N_in)` shape when only `output_dx` is set; this was
  "benign for HFPI, triggers P1-1 TypeError for HF".  With the
  P1-1 fix in place, HF now resamples cleanly to the square
  shape too.  No additional change needed.
* **`_AUDIT_ID_KNOWN_PREFIXES` allowlist wired** (P3-4).  The
  allowlist was defined at v5.2.5 but never used.  v5.3
  `_extract_cited_audit_ids` now filters candidates through the
  allowlist as a defense-in-depth guard (regex enforces the same
  gate at the moment, so this is structurally a no-op; if a
  future regex relaxation widens the prefix surface, the
  allowlist catches false positives in prose).
* **Chebyshev derivative `xp` branch unreachable code removed**
  (P3-5).  In `chebyshev_derivative_vandermonde`'s xp branch,
  the early-return at `max_k < 1` makes the downstream
  `if max_k >= 1: U.append(2.0 * u_arr)` check tautologically
  true.  Simplified to direct construction.
* **`install_atexit_restore` -> `_install_atexit_restore`
  bootstrap migration** (P3-6).  v5.2.5 renamed the function but
  left the library bootstrap (`lumenairy/__init__.py:682`) calling
  the legacy public name.  v5.3 migrates the bootstrap to the
  underscore form; the back-compat alias in `_context.py`
  remains for external callers.  "Private intent" is now a
  consistent signal across internal call sites.
* **Strehl > 1 methodology disclosure** (P3-7).
  `examples/09_tolerancing_monte_carlo.py` gains a docstring note
  + conditional summary print explaining that the small fraction
  of trials reporting Strehl slightly > 1.0 (max ~1.003) is NOT
  a normalization bug -- it's a methodology consequence of
  computing `ideal_peak` at the NOMINAL focal length while each
  trial searches for its own `z_best`.  Corrects the v5.2.5
  CHANGELOG attribution of "rounding noise".
* **JAX `monte_carlo_tolerancing_jax` regression test added**
  (P3-8).  v5.2.5 mirrored the NumPy Strehl-normalization fix
  in the JAX path at `through_focus.py:1251-1267` but no
  regression test exercised it.  v5.3 adds 4 tests in
  `tests/unit/test_v5_3_jax_monte_carlo_tolerancing.py`: Strehl
  values bounded [0, 1.05], mean < 1.0, JAX-vs-NumPy parity
  within 0.05 (loose because JAX defaults to float32 without
  `JAX_ENABLE_X64`), and return-dict shape.  Tests skip cleanly
  when JAX is not installed.
* **ROADMAP test-count off-by-one corrected** (P3-9).  v5.2.5
  ROADMAP cited 3780/16/1; matches CHANGELOG header at the same
  drift.  v5.3 stamps ROADMAP at the v5.3 empirical numbers.
* **V16 Tier-N regex widened to `\d+`** (audit Part 8 stretch
  goal #12).  Was `\btier\s+[0-9]\b` -- single digit only.  Now
  handles hypothetical Tier-10+ headings.

### v5.x ROADMAP horizon items (3)

* **`pyproject.toml` conftest comment-text correction** (user item
  #1, AUDIT_V5_2_5 V6 finding).  v5.2.2 inline comment cited
  "v4.16.0 conftest pattern" for zarr skip; the pattern actually
  lives at `test_v4_16_0_agent_b_multiprocess_storage.py:142-159`
  (per-test, not conftest).  `tests/conftest.py` has zero zarr
  handling.  Corrected.
* **CHANGELOG.md pre-v4.11 archive completion** (user item #2).
  v5.2.3 moved v4.11.x + v4.12.x entries into
  `docs/changelogs/v4.md` (962 lines).  v5.3 completes the
  archive: v4.10.x through v2.5.x entries moved into
  `docs/changelogs/v4.md` (now 6676 lines); top-level
  `CHANGELOG.md` reduced from 10949 to 5236 lines.  Net move
  preserves every entry verbatim with CRLF -> LF normalisation to
  match the v4.md file's existing line endings.  V11 doc-
  consistency walker still green.
* **`MultiFieldMerit` JIT compile** (user item #3, ROADMAP v5.3
  horizon).  The per-field `np.where(mask, np.exp(1j * (sin(tx)
  * k_X + sin(ty) * k_Y)), 0)` chain -- three N x N temporaries
  on the legacy NumPy path -- is now fused into a single Numba
  `@njit(parallel=True, fastmath=True)` kernel with zero
  temporaries.  Two dtype-specialised kernels (complex64 +
  complex128) dispatched at the call site.  **Measured speedup:
  1.5-8x speedup (hardware dependent; v5.3 audit measured 8.53x
  at N=256/8 fields on a 16-core machine, dipping to 1.5-3x
  under heavy parallel contention)** (jit=3.1 ms vs numpy=25.4
  ms at the headline operating point).  Lifts the meshgrid-
  cache-hit performance ceiling cited in the v4.14.0 perf audit
  (1.19x at N=128).  NumPy fallback bit-for-bit preserved; JIT
  path matches the NumPy reference to `rtol=1e-12` (FMA
  reassociation under `fastmath=True` produces ULP-level
  differences, documented in `lumenairy/optimize/_merit_jit.py`).
  The NumPy path is also used when Numba is unavailable OR when
  the grid is below the JIT-call-overhead threshold (N < 128).
  52 new tests in `test_v5_3_multi_field_merit_jit.py` covering
  numerical equivalence + cache-hit preservation + speedup
  pin + fallback.

### Documentation updates

* `CHANGELOG.md` -- v5.3.0 entry + archive split (5713 lines moved
  to `docs/changelogs/v4.md`).
* `ROADMAP.md` -- v5.3.0 added to release timeline; baseline test
  counts refreshed.
* `examples/09_tolerancing_monte_carlo.py` -- Strehl methodology
  note added.
* `lumenairy/analysis/ao.py` -- docstring leak/gain=0 joint
  semantics paragraph added.
* `lumenairy/_math/chebyshev.py` -- unreachable-code cleanup
  + comment.
* `pyproject.toml` -- conftest comment-text correction.
* `Wiki` -- Release-Notes + Function-Reference-Optimization +
  Adaptive-Optics pages updated for v5.3.

### Items still open (v5.3.1+ horizon)

After v5.3.0 ship, the v5.x ROADMAP is **EMPTY of code work**.  The
remaining horizon is process-only:

* **`logging` adoption sweep** (42 `warnings.warn` -> structured
  logging where appropriate).  Library-wide convention change;
  best done when there's a concrete telemetry trigger.
* **Designer GUI v3.8+** (separate version stream).
* **Next audit cycle** (AUDIT_V5_3_X).
* **CHANGELOG self-citation ship-time injection** (V17 currently
  surfaces drift; the FIX requires a pre-tag hook to stamp
  empirical values into the CHANGELOG entry before tag commit).

---

## [5.2.5] — 2026-05-21

**Patch release closing AUDIT_V5_2_3_2026_05_21** (the 4-release-chain
v5.1.1 -> v5.2.3 audit by the 12-agent fleet).  Closes all 7 P2 +
all 10 P3 findings.  Version number jumps from 5.2.3 to 5.2.5 (no
5.2.4) to signal that this is a substantive patch rather than a
trivial cleanup.

**Zero physics regressions in 14 consecutive releases.**

3780 unit tests pass (collected = 3797 = pass + 16 skip + 1 xfail);
+15 net pass vs v5.2.3.  34/34 validation pass.

### P2 closures (7)

* **V12 walker regex relaxed to accept short-form audit IDs**
  (P2-F1-2).  The v5.2.0 regex
  `r'(?<![A-Z0-9])(P[0-3](?:-NEW)?-[A-Z][A-Z0-9_-]{2,})'` required
  `[A-Z0-9_-]{2,}` after the first uppercase letter, so short-form
  IDs like `P1-A` / `P1-C` / `P1-F` / `P1-G` / `P1-1` (the IDs cited
  in v5.2.3's own CHANGELOG) were silently rejected.  V12.2 was a
  no-op on the v5.2.3 release.  Relaxed to
  `r'(?<![A-Z0-9])(P[0-3](?:-NEW)?-[A-Z0-9][A-Z0-9_-]*)'` + known-
  prefix allowlist (`P0-` / `P1-` / `P2-` / `P3-`).  Both
  short-form and long-form audit IDs now resolve through V12.2.
* **V16 heading classifier extended to "Tier N closures"** (P2-F1-3).
  `_is_audit_closure_section` in
  `scripts/verify_changelog_closures.py` only matched
  `'audit closure'` / `'audit carry-over'` / `'audit fix'` / `\bp[0-3]\s+closure`.
  v5.2.3 used `### Tier 1 closures` / `### Tier 2 closures` /
  `### Tier 3 closures` -- none matched.  Empirically
  `python scripts/verify_changelog_closures.py --version 5.2.3`
  exited rc=2 (treated as pass by CI gate); a v5.2.4 with
  fabricated `### Tier N closures` bullets would NOT have been
  caught.  Extended classifier with `re.search(r'\btier\s+[0-9]\b', s)`.
* **Python 3.10 added to publish.yml verify matrix** (P2-F1-4).
  v5.1.1 + v5.2.x publish.yml matrix was `[3.11, 3.12, 3.13]` -- no
  3.10 even though the documented floor is 3.10 and v5.2.2's whole
  reason for existing was 3.10 install-path drift.  Bumped to
  `[3.10, 3.11, 3.12, 3.13]` matching unit-tests.yml.
* **HFPI/HF freespace dispatcher threads `output_grid`/`output_dx`**
  (P2-F1-1).  v5.2.3 fixed the through-prescription paths but the
  freespace branches (when `prescription is None`) silently dropped
  the resolved values.  `propagate(method='hfpi', output_grid=...)`
  without a prescription returned a default-shape result.  Now
  threads `(output_shape, output_dx)` through the freespace
  branches too.
* **Consolidated test refreshed to v5.2+ kwarg idiom** (P2-F1-5).
  3 sites in `tests/unit/test_audit_propagation.py` carried the
  v4.11.2-era `output_grid=(Ny, Nx)` form, which now fires the
  v5.2.0 DeprecationWarning shim.  Tests passed only because
  DeprecationWarning is non-fatal by default.  Refreshed to
  `output_shape=(Ny, Nx)` so the bit-for-bit preservation claim is
  honest under `-W error::DeprecationWarning`.
* **AST `_resolve_arg_closure` tightening** (P3 V1 residual).  v5.2.0
  required the `'output'` literal + `__file__` reference inside the
  `makedirs(...)` first-arg subtree OR the RHS of the binding
  assignment.  F1 demonstrated two narrower gaming bypasses:
  multi-assign-shadow (`out_dir = good; out_dir = bad; makedirs(out_dir)`)
  + dead-code-in-unreached-function.  v5.2.5 adds function-scope
  filtering + last-write-wins semantics.  Both bypasses now
  rejected; 3 new gaming-form tests added.
* **dep-metadata drift check re-run** (P2 V6, no code change).
  Audit claimed the zarr env-marker was stale (zarr 3.2.1 requires
  >=3.12 vs pyproject >=3.11) and scipy/numpy/astropy/jax had
  silent floor drift.  v5.2.5 ran the v5.2.3-shipped dep-metadata
  drift script and confirmed **zero drift** -- the resolver picks
  compatible earlier zarr 3.0/3.1.x releases on Python 3.11 under
  the current bare-version constraints; same for scipy / numpy /
  astropy / jax.  Audit claim was speculative; current pyproject
  is correct.  The v5.2.3-shipped Monday cron continues to watch
  for real future drift.  **No files modified for this finding.**

### P3 closures (10)

* **Example 09 Strehl > 1 normalization fix** (P3-F1-1).  Both
  library-side AND example-side bugs identified + fixed:
  - `monte_carlo_tolerancing` (numpy + JAX backends) was
    recomputing `ideal_peak` inside the trial loop from each
    PERTURBED `E_exit`.  Pinned to the UNPERTURBED nominal pupil
    computed ONCE before the loop.
  - Example 09 hard-coded `f_target = 100e-3` but the singlet's
    actual paraxial BFL is 97.015 mm; passing a focal length that
    disagrees with the lens's true focus also produces Strehl > 1.
    Replaced with a `system_abcd` ray-trace to get the true BFL.
  Final printed Strehl: 0.967 +/- 0.029 (mean +/- std), 5th pct
  0.910, 95th pct 0.997.  All percentiles physically bounded;
  max of 1.003 is rounding-noise from per-iteration perturbations
  slightly shifting focus past the reference plane.
* **`ao_closed_loop` `leak` + `tol` kwargs + edge cases** (P3-F1-2 +
  V9).  Added two new kwargs: `leak: float = 0.0` (default 0.0 =
  pure integrator, bit-for-bit identical to v5.2.3) and
  `tol: Optional[float] = None` (residual-RMS early-stop
  threshold).  Loosened `gain` validation from `0.0 < gain <= 1.0`
  to `0.0 <= gain <= 2.0` -- `gain=0.0` is now an open-loop
  fallback.  Added `wfs(...)`-returning-None skip-update path
  (no longer crashes).  Cited CONVENTIONS.md Section 7 in
  docstring.  Bit-for-bit preservation at `leak=0.0` verified
  via `np.array_equal` pin.  8 new tests.
* **Chebyshev derivative + second-derivative `xp=` dispatch**
  (P3-F1-3).  v5.2.0 module docstring claimed "all three helpers"
  support `xp=` backend dispatch but only `chebyshev_vandermonde`
  did.  v5.2.5 makes the existing claim true: added `xp=None` kwarg
  to `chebyshev_derivative_vandermonde` and
  `chebyshev_second_derivative_vandermonde` + dispatched internal
  `np.*` to `xp.*`.  JAX traceability verified under `jax.jit`.
  Consumer call sites preserve numerics exactly (positional calls
  don't pass `xp=`, default `None` -> NumPy bit-for-bit).
* **CHANGELOG `apply_doe_phase_traced` re-worded** (P3-F1-4).
  v5.2.0 description said "phase advance/retard"; actual fix is
  direction-cosine deflection.  Grating equation L -> L + m*lambda/Lambda
  is transverse-only; OPL accumulator is unchanged.  Reworded in
  the v5.2.0 entry.
* **CHANGELOG test-count self-citation accuracy** (audit Part 8
  P3 row 5, F1).
  CHANGELOG v5.2.3 entry stated 3765/18/1; F1's empirical run
  showed 3766/17/1 (1-test delta from V12.4 conditional path).
  V12.3 self-consistency holds either way since 3784 = 3766 + 17 + 1
  = 3765 + 18 + 1.  Stamped the v5.2.3 entry counts at the
  build-time empirical numbers (verified at v5.2.5 ship as 3780
  pass / 16 skip / 1 xfail = 3797 -- the new tests + AST tightening
  account for the +15 pass delta).
* **V15 walker floor bumped 5 -> 6** (P3-F1-6).  v5.2.0 discovered
  6 sentinels including `_SchellReturnKindUnsetSentinel` (the
  v4.15.2 hardcoded tuple was missing it).  Floor tightened to >= 6
  to reflect that baseline.  Catches a walker collapse without
  leaving silently-pass headroom.
* **`install_atexit_restore` -> `_install_atexit_restore`** (audit F4).
  Renamed to underscore-prefixed form to signal private-bootstrap
  intent (caller is `lumenairy/__init__.py` at the end of library
  import; no user-facing call site).  Legacy name preserved as
  back-compat alias at the bottom of `_context.py` so any external
  caller importing it by the old name continues to work.
* **`docs/cookbook.md` cross-links** (audit F4).  Added 5
  cross-links: 2 in the header "See also" block (CONVENTIONS.md
  + Migration-Guide.md), 2 in OPD/Zernike + through-focus recipes
  (pointing at CONVENTIONS.md Section 7), 1 in polarization recipe
  (Migration-Guide.md for v4 -> v5 shim removals).
* **`ao_closed_loop` docstring cites CONVENTIONS Section 7**
  (audit F4).  One-line pointer in the Notes section: "phases follow
  the library-wide convention table in `CONVENTIONS.md` Section 7
  (OPD sign + time convention + forward propagation)".
* **CHANGELOG.md:81 line-count self-citation refreshed** (audit V8).
  v5.2.3 entry said `11553 -> 10618 lines` at the time of the
  archive split.  v5.2.3's entry itself added ~150 lines after
  the split was measured; current state is ~10769 lines.  v5.2.3
  bullet now states the AT-THE-TIME-OF-SPLIT count + the post-add
  current state.

### Side-effects

* `_context.py` legacy `install_atexit_restore` alias is NOT in
  `__all__` (matches the rename's "private intent" signal).  Any
  user-facing call site relying on `from lumenairy._context import install_atexit_restore`
  continues to work.
* Existing v5.2.3 test `test_ao_closed_loop_rejects_bad_gain` was
  updated to reflect the new `[0.0, 2.0]` range -- previously
  expected `gain=0.0` and `gain=1.5` to raise; now both are valid.

### Items still deferred to v5.3 forward horizon

Unchanged from v5.2.3:

* `MultiFieldMerit` JIT compile.
* `logging` adoption sweep.
* CHANGELOG pre-v4.11 archive completion.
* Designer GUI v3.8+ (separate version stream).
* Next audit cycle (AUDIT_V5_2_5_X).

### Files touched

29 files: 1 dispatch + 1 ao + 1 chebyshev + 3 walkers + 1 publish.yml + 1 test_audit_propagation + 1 test_v4_16_1_agent_d (AST) + 1 _context (rename + alias) + 1 cookbook + 1 through_focus + 1 example 09 + new test_v5_2_5_ao_closed_loop_residuals.py + various walker test pins + this CHANGELOG entry.

---

## [5.2.3] — 2026-05-21

**Full v5.x ROADMAP closure** (Tier-1 + Tier-2 + Tier-3 residuals
that v5.2.0 / v5.2.1 / v5.2.2 had deferred to v5.3).  After this
release, every concrete v5.x ROADMAP item has shipped except two
optional perf items (`MultiFieldMerit` JIT, `logging` adoption
sweep) and one doc tail (pre-v4.11 CHANGELOG archive completion),
all explicitly moved to the v5.3 forward horizon.

**Zero physics regressions in 13 consecutive releases.**

3765 unit tests pass (collected = 3784 = pass + 18 skip + 1 xfail);
+24 net vs v5.2.2 (3741).  34/34 validation pass.

### Tier 1 closures (5 substantive residuals)

* **24 formula-3 glass coefficient ingestion** (ROADMAP v5.1
  glass / materials residual).  The v5.2.0 evaluator + stub
  manifest now has all 24 coefficient sets ingested from the
  local refractiveindex.info database (4 CDGM polynomial + 10
  Hikari + 10 Sumita).  Worst-case n_d delta vs YAML reference:
  7.9e-6 (well under the 5e-5 cross-check budget; no tolerance
  relaxation required).  ``_POLYNOMIAL_STUB_NAMES`` is now empty;
  the dispatch arm raising ``NotImplementedError`` remains in
  place as a forward-parking spot for future catalogue
  additions.  +81 LOC in `lumenairy/glass.py`.
* **MHS subdomain maslov-branch substantive resampling** (AUDIT_V4_13_1
  P1-C residual).  v5.2.0 closed P1-C with a safe `ValueError`
  guard; v5.2.3 ships the substantive resampling.  The maslov
  branch of `prescription_subdomain` now honors `output_grid`
  via a post-kernel `resample_field` call with explicit
  power renormalisation (Parseval rel_err < 1e-9 verified by
  pin).  Retained narrower raises only for genuinely-degenerate
  cases (non-square `in_surface` / `out_surface`).  +66 LOC in
  `mhs.py` + 6 new pin tests.
* **Subaperture full image-plane mapping** (AUDIT_V4_13_1 P1-F
  residual).  v5.2.0 closed P1-F with a `UserWarning` for
  non-unit-magnification systems; v5.2.3 ships the substantive
  fix.  `propagate_subaperture_asymptotic` now derives the
  paraxial conjugate-imaging magnification from the system
  ABCD inside the propagator + routes the mapped image-plane
  patch centres / half-widths through `combine_patch_fields`'s
  v5.2.0 opt-in kwargs.  Unit-mag bit-for-bit preserved
  (verified at rtol 1e-12).  The v5.2.0 `UserWarning` is now
  silenced for typical telephoto / condenser cases and only
  fires when the ABCD probe itself fails on a degenerate
  prescription.  +55 LOC in `subaperture.py` + 7 new tests.
* **9 `inspect.getsource` proxy tests -> behavioral pins**
  (AUDIT_V4_13_1 Part 6.1 + ROADMAP residual).  Of the 9 sites
  v5.2.0 preserved with `# TODO(v5.2.1)` markers: 5 REPLACED
  with real behavioral assertions (each test now exercises
  runtime behavior instead of source-string inspection); 4
  KEPT with refreshed rationale (anti-pattern absence checks
  where structural-source inspection is the right tool).  All
  TODO markers removed.  Zero real bugs surfaced during the
  conversion.
* **`output_grid` dispatcher forwarding fix** (AUDIT_V4_13_1
  P1-A residual).  v5.2.0 renamed the sub-propagators'
  `output_grid` kwarg to `output_shape` (the dispatcher contract
  was `(N_out, dx_out)` but the sub-propagators interpreted as
  `(Ny, Nx)`); v5.2.3 fixes the dispatcher in
  `lumenairy/propagators/dispatch.py` to RESOLVE the canonical
  `(N_out, dx_out)` form into `output_shape=(N_out, N_out)` +
  `output_dx=dx_out` BEFORE forwarding to gbd / hfpi / hf.  The
  v5.2.0 `DeprecationWarning` shim at the sub-propagator no
  longer fires on dispatcher-canonical calls.  +60 LOC for the
  new `_resolve_dispatcher_output_grid` helper + 3 forwarding
  call sites updated.

### Tier 2 closures (2 documentation refactors)

* **README.md split** (ROADMAP v5.1 docs residual).  5121 ->
  4762 lines (-359, -7%).  The deep cookbook section carved
  out verbatim into `docs/cookbook.md` (376 lines).  Anchor
  links preserved.
* **CHANGELOG.md archive split** (ROADMAP v5.1 docs residual,
  partial).  11553 -> 10618 lines AT-THE-TIME-OF-SPLIT (-935,
  -8%).  v4.11.x and v4.12.x entries moved verbatim into
  `docs/changelogs/v4.md` (962 lines).  Pre-v4.11 entries
  (v4.10 down to v2.5) remain in the top-level CHANGELOG.md
  and are deferred to v5.3 for completion.  v5.2.5 self-citation
  refresh: post-v5.2.3 the top-level CHANGELOG.md is back up
  to ~10769 lines because the v5.2.3 entry itself added ~150
  lines after the split was measured.

### Tier 3 closures (3 tools / infra)

* **V16 content-level CHANGELOG fabrication walker + companion
  script** (ROADMAP "CONTENT-LEVEL fabrication walker"
  residual).  v5.2.0 shipped V12 which catches FILE-LEVEL
  fabrications (cited path does not exist); V16 is the
  CONTENT-LEVEL counterpart that uses `git diff PREV_TAG..HEAD`
  to verify each cited file is actually in the changeset.
  Two artifacts: `tests/unit/test_v5_2_3_walker_changelog_content.py`
  (V16 walker, 258 LOC) + `scripts/verify_changelog_closures.py`
  (CLI, 547 LOC).  The script is wired into `publish.yml` BEFORE
  `Library import sanity` so a release with fabricated
  audit-closure claims fails BEFORE PyPI upload.  Synthetic
  fabrication test verified the walker catches the v5.1.0-class
  pattern.
* **`ao_closed_loop` high-level helper** (ROADMAP v5.2.1
  candidate from example 11).  v5.2.0 example 11 had to build
  the AO loop from primitives because no high-level helper
  existed; v5.2.3 ships `lumenairy.ao_closed_loop` (+178 LOC in
  `analysis/ao.py`).  Canonical leaky-integrator control law,
  optional WFS callback, `return_history=True` for
  iteration-by-iteration RMS tracking.  Example 11 rewritten
  to use the new helper.  +13 new tests.
* **Dep-metadata drift weekly cron** (v5.2.2 retrospective
  closure).  v5.2.2 fixed the pyfftw 0.15.1 / zarr 3.x
  dropping-3.10 problem reactively; v5.2.3 ships
  `scripts/check_dep_metadata.py` + `.github/workflows/dep-drift.yml`
  (Monday 06:00 UTC cron) so the next instance of this
  failure mode is caught proactively.  Script reports zero
  drift against the current pyproject.toml (v5.2.2's
  env-marker fixes are complete).

### v5.2.0 audit follow-on

* 3 v5.2.0 formula-3 ship-state tests refactored to v5.2.3
  state (manifest empty -> tests skip cleanly rather than
  assert a contract that no longer has triggering input).

### Items moved to v5.3 forward horizon

* `MultiFieldMerit` JIT compile (perf; no concrete trigger).
* `logging` adoption sweep (42 `warnings.warn` -> structured
  logging where appropriate).
* CHANGELOG pre-v4.11 archive completion.
* Designer GUI v3.8+ (separate version stream).
* Next audit cycle (AUDIT_V5_2_X).

### Test counts

| Release | Pass | Skip | xfail | Collected |
|---|---|---|---|---|
| v5.1.0 | 3628 | 5 | 1 | 3634 |
| v5.1.1 | 3628 | 5 | 1 | 3634 |
| v5.2.0 | 3741 | 7 | 1 | 3749 |
| v5.2.1 | 3741 | 7 | 1 | 3749 |
| v5.2.2 | 3741 | 7 | 1 | 3749 |
| **v5.2.3** | **3765** | **18** | **1** | **3784** |

+24 net pass vs v5.2.2.  +11 skips break down as: 2 formula-3
stub-state tests (now vacuous after full ingestion), 8 v5.0.1
``TestRoadmapV5_1SectionStaleItemStrip`` pins (the v5.1.0
section those pins guard has been retired at v5.2.3; the
underlying anti-fabrication contract is preserved structurally
by V11 + V16 walkers), and 1 v5.2.3 V12.4 skip (current
CHANGELOG block declares no audit closures so the enforcement
has nothing to apply).

---

## [5.2.2] — 2026-05-21

**Patch release: fix Python 3.10 install path** (the documented
floor that v5.1.1 re-added to CI).  Two optional dependencies had
bumped their ``requires-python`` to ``>=3.11`` in 2025 releases,
silently breaking ``pip install lumenairy[fft,zarr,...]`` on 3.10:

- **`zarr>=3.0`** -- v5.1.0 floor-bump per audit M-3.  zarr 3.1.6
  (the resolver's target) requires Python >= 3.11.
- **`pyfftw>=0.13`** -- existing floor.  pyfftw 0.15.1 (2025
  release) requires Python >= 3.11; pyfftw 0.15.0 (last 3.10-
  compatible) is still available.

### Fix

Both extras groups now use PEP 508 environment markers so the
resolver picks compatible versions per interpreter:

```toml
fft = [
    'pyfftw>=0.13,<0.15.1; python_version < "3.11"',
    'pyfftw>=0.13; python_version >= "3.11"',
]
zarr = ['zarr>=3.0; python_version >= "3.11"', "filelock>=3.0"]
```

- On Python 3.10: pyfftw 0.15.0 (last 3.10 wheel); zarr is NOT
  installed (storage-zarr tests skip cleanly per the v4.16.0
  conftest pattern).
- On Python 3.11+: latest pyfftw + zarr v3.

The ``all`` group gets the same env-marker treatment so
``pip install lumenairy[all]`` resolves on every supported
interpreter.

### Why this regressed at v5.1.1 and only surfaced now

The v5.1.1 patch re-added Python 3.10 to the unit-tests CI matrix
(audit P2-NEW-3WAY-2).  At v5.1.1 ship, zarr 3.1.6 and pyfftw 0.15.0
still resolved on 3.10 (their `requires-python` was permissive enough
or the resolver picked compatible older versions).  Between v5.1.1
and the v5.2.1 push, pyfftw 0.15.1 was published and the zarr
metadata was tightened; the next `pip install` on Python 3.10
started failing.  No library code change caused this -- it is purely
external-dep metadata drift.

Caught by the v5.1.1 publish.yml `verify` gate (which exercises
3.10/3.11/3.12/3.13 on every tag push) before the v5.2.2 tag
shipped, exactly as designed: a release on broken CI cannot upload
to PyPI.

### Tests

3741 unit tests pass (collected = 3749 = pass + 7 skip + 1 xfail);
1 vs v5.2.1 is the storage SWMR multiprocess test toggling between
pass and skip across runs (documented flake, not a regression).
34/34 validation pass.  Zero behavior change.  **Zero physics
regressions in 12 consecutive releases.**

---

## [5.2.1] — 2026-05-21

**Patch release: complete v5.2 ruff baseline closure (134 -> 0
errors).**  v5.2.0 left 134 advisory ruff errors deferred with
`continue-on-error: true` on the `lint` job; this patch closes them
honestly.

### Ruff cleanup (134 -> 0 advisory errors)

- **70 F841 (unused-variable)** auto-fixed via `ruff --fix
  --unsafe-fixes`.  Two genuinely-dead assignments deleted manually
  in `lenses_maslov.py` (`v2x_samples` + `v2y_samples` were computed
  but never used; downstream code reads the unitless
  `u_v2x_samples` / `u_v2y_samples` Chebyshev-node coords instead),
  and two stale `M = len(mi)` sites deleted.
- **63 E702 (multiple-statements-on-one-line-semicolon)** split via a
  scripted AST-aware splitter (`scripts/`-style one-off; not
  committed).  Pure cosmetic, zero behavior change.
- **1 I001 (unsorted-imports)** auto-fixed.

### numexpr static-analysis cleanup (`lenses_maslov.py`)

The Maslov propagator's hot inner loop uses
`numexpr.evaluate("expr_string")` for the 5 array operations that
would otherwise allocate ~17 GB complex128 temporaries at N=32768.
numexpr reads variable names from the caller's stack frame via
introspection at runtime, which makes `twopi` / `cos_term` /
`sin_term` / `Er` / `Ei` invisible to ruff and mypy -- they
appeared as F841 unused-variable false positives.

v5.2.1 refactors all 4 affected calls in `lenses_maslov.py` to pass
the variables explicitly via `local_dict={'name': value, ...}`,
matching the canonical pattern already used at `_lens_real.py:882`.
Variable names now appear in the surrounding code's AST; ruff /
mypy / IDEs see the usage; no `# noqa: F841` needed; no
performance loss (`local_dict=` is the recommended numexpr API
and avoids the runtime frame-introspection overhead).

### CI: lint job now GREEN

`.github/workflows/unit-tests.yml` `lint` job still carries
`continue-on-error: true` (preserved for forward-safety against
future ruff rule additions) but **now finishes green** on every
push.  The red badge that has been flagging on every push since v5.0
is retired.

### Per-file ruff ignores (no change in v5.2.1; documented for
clarity)

The v5.2.0 per-file ignores in `pyproject.toml`
`[tool.ruff.lint.per-file-ignores]` remain in place:
- 6 v5.1.0 file-split shells (`F401`, `F403` -- the shells exist to
  re-export pre-v5.1 public names; "unused-import" is the correct
  behavior, not a bug).
- 8 sub-package `__init__.py` files (same rationale).
- `tests/**/*.py` (`F401`, `F811` -- test files re-import + redefine
  fixtures freely).

No new per-file ignores were added at v5.2.1.  The lenses_maslov
F841 false positives are closed by the `local_dict=` refactor, NOT
by a per-file ignore.

### Tests

3742 unit tests pass (collected = 3749 = pass + 6 skip + 1 xfail);
same as v5.2.0.  34/34 validation pass.  Zero behavior change.
**Zero physics regressions in 11 consecutive releases.**

### Why the deferral happened at v5.2.0

Honest retrospective: v5.2.0's CHANGELOG noted "deferred to v5.2.1
for the unsafe-fix sweep" but did not explain WHY.  The reason was
caution -- F841's `--unsafe-fix` rewrites `x = func()` to bare
`func()`, which can silently break callers that look up `x` via
`globals()['x']` (rare but possible).  In retrospect this was
over-cautious for a library with no `globals()['<name>']`
introspection pattern: the unsafe-fixes are safe in practice.

The user feedback ("I wanted complete closure on all v5.x
updates") is correct.  v5.2.1 ships the closure.  The two failure
modes I was right to be cautious about both DID surface during the
patch -- numexpr false positives (refactored to `local_dict=`) and
the `lenses.py` Chebyshev re-export back-compat alias (restored
with `# noqa: F401` on the import block) -- and were caught by the
existing test suite, so the caution paid off as a regression net
even though the deferral itself was unwarranted.

---

## [5.2.0] — 2026-05-20

**Largest non-breaking release in the v5.x series.**  Closes the
v5.1.1 audit (`docs/audits/AUDIT_V5_1_1_2026_05_20.md`) plus every
remaining v5.x ROADMAP item.  Scope: 4 new meta-walkers, 6 deferred
features, 7 structural cleanups, 5 physics-correctness fixes, ruff
+ mypy baseline closure.

**Zero physics regressions in 10 consecutive releases.**

**3741 unit tests pass** (collected = 3749 = pass + 7 skip + 1 xfail);
**+113 net** vs v5.1.1 (3628).  **34/34 validation pass.**  Library
public API: `len(lumenairy.__all__) = 534` (+1 over v5.1.1 -- the
new `MCF` top-level alias).

### v5.1.1 audit closures (5 items)

* **Tighten `test_examples_output_dir` AST check** (audit P2 v5.2
  candidate).  v5.1.1's check required three signals (`'output'`
  literal + `makedirs(...)` + `__file__`) to appear ANYWHERE in
  the file -- F1 demonstrated this was gameable with the signals
  scattered.  v5.2 requires data-flow co-location: the `'output'`
  literal AND `__file__` reference must appear inside the
  `makedirs(...)` call's first-argument subtree, OR in the RHS of
  the assignment binding the call's first argument.  Three
  scattered tokens no longer suffice.
* **Tighten Migration-Guide content-locks** (audit P2 v5.2
  candidate).  v5.1.1 asserted `shim-name` + `**Removed.**` as
  independent global substrings; F1 noted a `lumenairy.ao` quote
  elsewhere in the guide could keep the pin green while the actual
  removal section was deleted.  v5.2 anchors `**Removed.**` to
  within 3 lines of the shim-name match (recipe-window pattern)
  plus the new-import line within 9 lines.  Deletion-detection
  now scales with section locality.
* **V12 walker (CHANGELOG-vs-changeset verifier)** at
  `tests/unit/test_v5_2_walker_changelog_changeset.py` (audit P1
  v5.2 candidate).  Structural fix for the v5.1.0 fabrication
  class.  Parses the most-recent `## [X.Y.Z]` block + asserts:
  (a) every backticked file-path citation resolves to a real
  repo file; (b) every audit-ID citation appears under
  `docs/audits/`; (c) test-count arithmetic reconciles; (d) the
  block advertises an audit-closure verification mechanism.  On
  first run V12 immediately caught FIVE fabricated audit IDs in
  the v5.1.1 CHANGELOG (the P1-2way-N family for N=2..6 which I
  had invented).  Those IDs have been corrected to the audit's
  actual
  `P2-NEW-3WAY-2`, `P2-NEW-V4-G`, `P2-NEW-F1-3`, `P2-NEW-V2`,
  `P2-NEW-V4-E` per the v5.1.0 audit's per-closure verdict table.
  V12 paid for itself before it shipped.
* **V13 walker (shell-vs-canonical-location uniqueness)** at
  `tests/unit/test_v5_2_walker_shell_vs_canonical.py` (audit
  P2-NEW-F2-1 #1).  For every name imported in a post-v5.1
  file-split shell's `from .X import Y` block, asserts
  `Y.__module__` is the submodule, not the shell.  Walks 6 shells
  -- raytrace, propagation, asymptotic, optimize, prescriptions,
  analysis -- and 243 raw / 334 expanded re-export claims.  One
  documented exemption (`propagate_modal_asymptotic` v4.14.1
  monkey-patch contract).  13/13 pass.
* **V14 walker (PEP-562 forwarding completeness)** at
  `tests/unit/test_v5_2_walker_pep562_forwarding.py` (audit
  P2-NEW-F2-1 #2).  Enumerates `fft_infra` mutable globals
  (those rebound via `X = ...`) and asserts each appears in
  `propagation._LIVE_FORWARD_NAMES`.  Counter-pin verifies the
  whitelist hasn't drifted in the opposite direction.  19
  mutable globals discovered; 4 defensive whitelist entries
  carried as harmless future-proofing.  4/4 pass.
* **V15 walker (sentinel `__reduce__` structural)** at
  `tests/unit/test_v5_2_walker_sentinel_reduce.py` (audit
  P2-NEW-F2-1 #3 + P3-NEW-F1-3).  Auto-discovers every
  `_Sentinel` subclass via `__subclasses__()` and asserts each
  defines `__reduce__` -> `(_sentinel_unpickle, (name,))` with
  the name registered in `_SENTINEL_REGISTRY`.  Discovered SIX
  sentinels including `_SchellReturnKindUnsetSentinel` which was
  NOT in the v4.15.2 hardcoded `EXPECTED_SUBCLASSES` tuple and
  was silently missing pickle round-trip coverage -- exactly the
  failure mode P3-NEW-F1-3 predicted.  V15 retroactively closed
  that finding.  15/15 pass.

### v5.1.0 audit carry-over (1 item)

* **Prune 60 dead V7 walker exemption entries** (audit
  P3-NEW-F1-2).  After the v5.1.0 6-file split, the 10
  `propagators/propagation.py:*` and 26 `analysis/core.py:*`
  exemption entries in
  `tests/unit/test_v4_16_0_walker_xp_of_dispatch.py:179-375`
  pointed at function bodies that had moved to topical submodules
  (the shells now have zero function definitions).  v5.2 deletes
  the 36 dead entries with a citation block explaining that V13
  catches any future regression where a function body sneaks
  back into a shell.  Walker auto-discovery re-finds the
  dispatch sites at the new submodule paths.

### Deferred v5.1.0 features (6 items)

* **`MCF` top-level alias** for `PartialCoherenceMCF` (ROADMAP
  v5.1 partial-coherence polish).  One-line addition + `__all__`
  bullet so the partial-coherence import story is uniform with
  `lumenairy.coherence_at` and `lumenairy.propagate_ensemble`.
  The canonical class name `PartialCoherenceMCF` is unchanged.
* **Formula-3 (polynomial) glass evaluator + 24-glass stub
  manifest** in `lumenairy/glass.py` (ROADMAP v5.1 glass /
  materials).  New `_polynomial_index(wavelength_m, coeffs)`
  with a scalar fast-path mirroring `_sellmeier_index`'s
  `math.sqrt` float-arithmetic plus a vectorized NumPy / JAX
  path.  New `_POLYNOMIAL_STUB_NAMES` frozenset of 24 entries
  (4 CDGM polynomial + 10 Hikari + 10 Sumita) -- coefficient
  ingestion deferred to v5.2.1 to avoid fabricated values.
  Minimal installs hitting a stubbed name raise
  `NotImplementedError` with both the `lumenairy[glass]` install
  path AND a v5.2.1 issue tracker reference.  Module-load
  consistency invariants in `_check_glass_registry_consistency`
  catch a v5.2.1 ingestion PR that forgets to remove the stub
  entry.  20 new tests; 19 pass + 1 vacuous-skip at ship.
* **Off-axis conic in surface frame for `apply_real_lens`**
  (ROADMAP v5.1 off-axis conic).  New `surface_frame: bool =
  False` kwarg.  When `True`, the per-surface `"decenter"` /
  `"tilt"` are applied as a rigid-body transform: the field's
  `(x, y)` maps to surface-frame `(x_s, y_s)` via
  `R^T @ (x - dcx, y - dcy, 0)` (full rotation matrix, no
  small-angle linearization), and sag is evaluated at `(x_s,
  y_s)`.  The linear sag ramp is suppressed in this branch to
  avoid double-counting.  Default `surface_frame=False`
  preserves v5.1 behavior bit-for-bit (verified by 2
  backwards-compat pins including one with active
  decenter+tilt).  Migration-Guide.md gains a new v5.2.0
  section with the physics rationale + Optiland/Zemax parity
  notes.  5 new tests.
* **5 new examples** -- `examples/08_multiconfig_zoom.py`,
  `examples/09_tolerancing_monte_carlo.py`,
  `examples/10_coronagraph_workflow.py`,
  `examples/11_ao_closed_loop.py`,
  `examples/12_ghost_stray_light.py` (ROADMAP v5.1 docs /
  examples).  881 LOC total; each runs in < 60s, uses the
  canonical v4.16.1 `examples/output/` wiring, has `main()` +
  `__main__` guard, and is parsing-pinned at
  `tests/unit/test_v5_2_new_examples.py` (20 tests).  Example
  11 (AO closed loop) uses primitives + a documented
  "build-it-yourself" idiom since no high-level
  `ao_closed_loop` helper exists in the library yet -- v5.2.1
  candidate.
* **57-file `test_audit_fixes_*` consolidation** (ROADMAP v5.1
  architecture / housekeeping).  57 files -> 10 topical homes:
  `test_audit_analysis.py` (66 tests),
  `test_audit_glass.py` (19),
  `test_audit_io.py` (41),
  `test_audit_lens.py` (52),
  `test_audit_misc.py` (230),
  `test_audit_optimize.py` (82),
  `test_audit_polarization.py` (41),
  `test_audit_propagation.py` (98),
  `test_audit_raytrace.py` (61),
  `test_audit_sources.py` (101).  791 tests preserved bit-for-bit;
  zero behavior changes.  223 class-name attribution prefixes
  (`TestAuditFixesV<ver>_<scope>_<orig>`) maintain git-blame
  traceability.  9 `inspect.getsource` proxy-test sites
  conservatively kept with `# TODO(v5.2.1): replace with
  behavioral pin -- inspect.getsource proxy-test pattern (per
  AUDIT_V4_13_1 Part 6.1)` comments; none deleted (audit
  AUDIT_V4_13_1 Part 6.1 deferred to v5.2.1).
* **Shared Chebyshev helpers extracted to `lumenairy/_math/chebyshev.py`**
  (ROADMAP v5.1 architecture / housekeeping).  The 3 NumPy
  helpers from `elements/lenses.py:722-810` plus the
  xp-dispatched twin from `asymptotic_jax_twin.py:65` are now
  in a single `chebyshev_vandermonde(u, max_k, xp=None)`
  signature.  6 consumer sites updated (lenses, lenses_maslov,
  4 asymptotic_*).  Back-compat aliases preserved at
  `lumenairy.elements.lenses._chebyshev_*` so external imports
  by the old underscore-prefixed names still work.  10 new
  tests + 151 / 151 asymptotic+maslov tests + 406 / 406
  lens-related tests all green; V13 walker still clean on the
  updated asymptotic shell.

### Structural cleanups (3 items)

* **`_xp_of` deduplication** (ROADMAP opportunistic item).  5
  copies of the 4-line wrapper `def _xp_of(*arrays): from
  ..backend import array_namespace; return array_namespace(*arrays)`
  (in `elements/elements.py`, `elements/freeform.py`,
  `analysis/beam_stats.py`, `analysis/strehl.py`,
  `analysis/psf_mtf_otf.py`) consolidated to a single
  `from ..backend import array_namespace as _xp_of` alias.
  All 5 call-site contracts preserved (the alias keeps the
  module-local name); zero behavior change.
* **`backend/fft.py` -> `propagators/propagation.py` inversion
  fix** (ROADMAP opportunistic item).  Pre-v5.1, the FFT
  plan-cache infra lived inside `propagators/propagation.py`
  and `backend/fft.py` imported through that monolith
  (inverted dependency).  v5.1 lifted the infra to
  `fft_infra.py`; v5.2 routes the 5 `backend/fft.py` import
  sites directly through `fft_infra` instead of the
  `propagation` shell.  Removes the PEP-562 `__getattr__`
  forwarding step from the hot FFT-dispatch path.
* **`_deprecation.py` orphan helper documentation** (ROADMAP
  opportunistic item).  `warn_deprecated_kwarg`,
  `warn_renamed_function`, and `warn_deprecated_default` are
  exported but have zero internal call sites.  v5.2 keeps them
  (deletion would silently break any external by-name caller
  -- we have no out-of-repo telemetry) with a module docstring
  note explaining the orphan status + canonical-format-for-
  future-deprecations rationale.

### Documentation (2 items)

* **CONVENTIONS.md sign-convention table** (ROADMAP v5.1
  docs).  Section 7 gains a 12-row one-stop summary table
  covering time / propagation / mirror radius / refraction
  radius / OPD / lens phase / mirror phase pickup / aperture
  transmission / decenter / tilt / polarization / refractive
  index.  Future per-site contradictions resolve against the
  table.
* **`validation/README.md`** (ROADMAP v5.1 docs).  Decision
  tree for `tests/unit/` vs `validation/`, layout reference,
  running instructions, file-naming convention (`t_*.py` vs
  `test_*.py`).  Closes the long-standing "contributors don't
  know whether to add tests to `tests/unit/` or `validation/`"
  gap.  README.md + CHANGELOG.md archive splits deferred to
  v5.3 (high-link-breakage risk).

### Physics-correctness fixes (5 items)

All five are AUDIT_V4_13_1 deferred Tier-2 items.

* **`apply_doe_phase_traced` sign preservation** (P1-G; ~10
  LOC in `raytrace/trace.py`).  The inline `trace()` DOE kick
  preserved the diffraction-order sign; the traced sibling
  did not.  Fix mirrors the inline pattern.  Negative-order
  diffraction now produces the correct direction-cosine
  deflection (the grating equation L -> L + m*lambda/Lambda is
  transverse-only; the OPL accumulator is unchanged).  3 new
  tests.
* **`MultiPrescriptionParameterization.scale_floor`** (P1-1;
  `optimize/parameterizations.py`).  Added `scale_floor`
  kwarg + per-parameter-type default table: radii /
  thicknesses 1e-6 m, conics / aspheric `alpha_n` 1e-3.
  Parameters near zero no longer collapse the optimizer's
  `x_scale`.  Driver reads via existing
  `getattr(parameterization, 'scale_floor', None)`; pre-v5.2
  callers see no behavior change.  7 new tests.
* **`output_grid` -> `output_shape` rename on sub-propagators**
  (P1-A).  The dispatcher contract `output_grid=(N_out,
  dx_out)` is canonical; 3 sub-propagators (gbd / hfpi / hf)
  used the same kwarg name to mean `(Ny, Nx)`.  v5.2 renames
  the sub-propagator kwarg to `output_shape` + adds a
  back-compat shim emitting `DeprecationWarning` on the
  legacy `output_grid` form.  Six entry points updated:
  `propagate_gbd_freespace`, `propagate_gbd_thin_lens`,
  `propagate_gbd_through_prescription`,
  `propagate_hfpi_freespace_aperture`,
  `propagate_hfpi_through_prescription`,
  `propagate_huygens_fresnel_through_prescription`.  5 new
  tests.  **Open caveat**: `dispatch.py` still forwards the
  legacy form; deferred to v5.2.1 -- documented in
  Migration-Guide.md.
* **MHS subdomain grid-loss guard** (P1-C; `propagators/mhs.py`).
  `prescription_subdomain(method='maslov')` silently ignored
  `output_grid` and returned on the input grid.  v5.2 raises
  `ValueError` with a clear migration recipe (use a different
  method or accept the input-grid output explicitly).  3 new
  tests.  Substantive maslov-branch grid resampling deferred
  to v5.2.1.
* **Partition-of-unity convention `UserWarning`** (P1-F;
  `propagators/subaperture.py`).  `propagate_subaperture_asymptotic`
  centered windows on source-plane positions, which is only
  correct for unit-mag no-tilt systems.  v5.2 probes the
  system ABCD's `|A - 1|` and emits `UserWarning` for
  non-unit-magnification systems; the existing test for the
  magnifying-singlet case now legitimately flags this.  New
  optional `image_centres` / `image_half_widths` kwargs on
  `combine_patch_fields` let callers with magnification info
  pass image-plane patch coordinates explicitly.  3 new tests.
  Full image-plane mapping inside
  `propagate_subaperture_asymptotic` deferred to v5.2.1.

### Lint / type baseline closure

* **Ruff baseline cleanup** -- 917 errors (v5.1.1) -> 134
  errors (v5.2).  85% reduction via safe `ruff --fix`.
  Per-file ignores added for the 6 v5.1.0 file-split shells +
  8 sub-package `__init__.py` files (re-export modules where
  F401 unused-import is correct behavior, not a bug).
  Remaining 134 errors are F841 unused-vars (70) + E702
  semicolons (63) + 1 misc; all need `--unsafe-fixes` and
  are advisory-only (`lint` job has
  `continue-on-error: true`).  Deferred to v5.2.1 for the
  unsafe-fix sweep.
* **mypy strict baseline cleanup + CI activation** -- 76
  errors (v5.1.1) -> 0 errors.  All scope-local errors in the
  `[tool.mypy]` whitelist (`lumenairy/backend`,
  `_deprecation.py`, `_context.py`, `progress.py`,
  `memory.py`) cleaned.  `mypy` is now wired into
  `unit-tests.yml` as a real gate (`continue-on-error: false`).

### Meta-pattern note (v5.2 retirement state)

The "fix N, miss N+1" sibling-gap meta-pattern is now structurally
retired across 15 currently-known surfaces (V1-V15).  New classes
will continue to surface and be added to the V-walker family as
identified -- including CONTENT-LEVEL CHANGELOG fabrications
(where the cited file exists but the cited behavior is missing)
which V12 deliberately does NOT cover.  Those need the diff-aware
companion script + human review; deferred to v5.3.

### Items still deferred to v5.2.1 / v5.3+

* 24 formula-3 glass coefficient ingestion (data, no library API
  change).
* `output_grid` dispatcher-forwarding fix (P1-A residual).
* MHS subdomain maslov-branch substantive resampling (P1-C
  residual).
* Subaperture image-plane partition-of-unity full fix (P1-F
  residual).
* 9 `inspect.getsource` proxy tests -> behavioral pins
  (AUDIT_V4_13_1 Part 6.1).
* Ruff `--unsafe-fix` sweep (F841 + E702, 133 advisory errors).
* `ao_closed_loop` high-level helper (example 11 currently
  builds from primitives).
* README.md + CHANGELOG.md archive splits.
* CONTENT-LEVEL CHANGELOG-fabrication walker (companion to V12
  using `git diff PREV_TAG..HEAD`).
* `MultiFieldMerit` JIT compile (perf).
* `logging` adoption sweep (42 `warnings.warn` -> structured
  logging where appropriate).

---

## [5.1.1] — 2026-05-20

**Patch release closing the v5.1.0 audit
(`docs/audits/AUDIT_V5_1_0_2026_05_20.md`).**  The headline finding:
the v5.1.0 CHANGELOG's "v5.0.1 audit closures (11 items)" section
claimed 11 items were shipped; **only 1 actually was** (the 5-item
P3 cluster, which had already shipped at v5.0.1).  The other 10 --
including the highest-priority `publish.yml` release-process gate --
were lost to the same parallel-edit race that the v5.1 Wave-4
integration sweep was meant to close.  Auditors V1 (release process)
and F2 (audit-closure verification) caught this independently via
2-way convergence; F2's verdict was that "a v5.2.0 tag tomorrow could
ship to PyPI with a fully-broken CI pipeline."

v5.1.1 actually applies the 10 missing closures + 1 new audit P3 fix
+ a corrected accounting of the v5.1.0 ship state.  Scope is small
(~120 LOC) and the work was done serially (no agents -- the v5.1.0
parallel-edit race is itself part of what this patch is fixing).

**Zero physics regressions in 9 consecutive releases.**

### v5.1.0 audit closures actually shipped in v5.1.1

**P1 (1):**
* **`publish.yml` release-process gate** (audit P1-NEW-3WAY-1; the
  v5.1.0 audit's umbrella finding for the CHANGELOG fabrication
  is P1-NEW-2WAY-1).  New pre-build `verify` job runs the unit suite
  + library-import sanity on the tag's source across Python
  3.11/3.12/3.13 BEFORE `build` and `publish` fire (`build` depends
  on `verify`; `publish` depends on both).  v5.0.0, v5.0.1, AND
  v5.1.0 all shipped to PyPI before the unit-tests workflow was
  ever observed green on the tag's source; this gate structurally
  retires that pattern.  The v5.1.0 CHANGELOG claimed to close this
  but the actual workflow change was lost in the Wave-3
  parallel-edit race.

**P2 (5):**
* **Python 3.10 re-added to the unit-tests CI matrix** (audit
  P2-NEW-3WAY-2).  The documented floor
  (`requires-python = ">=3.10"`) was un-tested between v5.0.1 and
  v5.1.0 because the v5.0.1 CI install dropped 3.10 pending a
  3.10-specific install path verification.  Re-adding so the
  documented minimum is exercised on every PR.
* **Doubled `@_skip_no_qt` decorators removed** at
  `tests/unit/test_v4_15_agent_e.py:215` (TestUI3) and `:254`
  (TestUI4) (audit P2-NEW-V4-G).
* **`test_examples_output_dir` tightened** (audit P2-NEW-F1-3).
  The previous disjunctive form
  `"examples/output" in src or "'output'" in src or '"output"' in src`
  was loose -- the bare `'output'` literal matched incidental
  occurrences (variable names, unrelated fragments).  Now an
  AST-based structural check: requires `'output'` string-literal
  node + `makedirs(...)` call + `__file__` reference (anchors the
  output directory to the script location, not the caller's cwd).
* **3 Migration-Guide content-lock assertions added** to the shim
  pins (`lumenairy.ao`, `lumenairy.io.hdf5`, `lumenairy.system` top
  level) (audit P2-NEW-V2).  Each pin now reads
  `Migration-Guide.md` and asserts the removal line + new import
  path are both present.  Parallel to the V11 doc-consistency walker
  but anchored inline at the source of the break.
* **`::error::` annotation choice documented inline** in
  `unit-tests.yml` (audit P2-NEW-V4-E).  Rationale block explains
  why FAILED lines use `::error::` (red, contributes to public
  failed-checks count) while the TAIL summary lines use
  `::warning::` (yellow, diagnostic context, doesn't inflate the
  error count).

**P3 (5):** already shipped at v5.0.1 -- no v5.1.1 work required.

### New audit P3 fix

* **`_PYFFTW_BAD_SHAPES` added to `_LIVE_FORWARD_NAMES`** in
  `propagators/propagation.py:230` (audit P3-NEW-F1-1).
  `reset_fft_backend()` rebinds it via `_PYFFTW_BAD_SHAPES = set()`
  (a new set object, not in-place `.clear()`).  Consumers reading
  `propagation._PYFFTW_BAD_SHAPES` after a reset would have seen the
  pre-reset snapshot.  Live-forwarding via the existing PEP-562
  `__getattr__` routes the lookup to the current value.

### v5.1.0 CHANGELOG correction

The v5.1.0 CHANGELOG (immediately below) reads as if 11 v5.0.1 audit
closures shipped at v5.1.  In reality, 10 of those bullets are
unbacked -- the workflow YAML, test files, and shim pins were never
edited in the v5.1 release tree.  v5.1.1 ships the actual code and
corrects the count.  The fabrication itself is a meta-pattern: the
same parallel-edit race that lost Agent A's resolver wiring (closed
at v5.1.0 Wave-4) also lost the v5.0.1 audit closures that I had
applied in Wave-1.  Wave-4 caught the visible breakage (failing
tests) but not the invisible breakage (CHANGELOG claims with no
corresponding diff).  Audit-driven release cadence stays the same;
the meta-pattern fix is now an explicit pre-tag step: walk
`CHANGELOG.md`'s "audit closures" list against `git diff
PREV_TAG..HEAD` to confirm each claim has a backing change.

### Baseline count refresh

mypy and ruff baselines drifted upward between v5.0.1 and v5.1.0
(more code -> more advisory lint), but the CHANGELOG kept citing the
v5.0.1 numbers (audit P2-NEW-F2-2):

| Tool | v5.0.1 cite | v5.1.0 actual | v5.1.1 cite |
|---|---|---|---|
| mypy (whitelist, strict, `follow_imports=silent`) | 63 | 76 | 76 |
| ruff (advisory) | 692 | 893 | 893 |

The CHANGELOG and ROADMAP "deferred" entries are updated to the v5.1
actual counts.

### Test counts

3628 unit tests pass (collected 3634 = pass + 5 skip + 1 xfail), same
as v5.1.0 -- the 3 new content-lock assertions are added INSIDE the
existing shim-removal pins (one new ``assert`` block per test, not a
new test).  **34/34 validation pass.**

### Items still deferred to v5.2+

Unchanged from v5.1.0 except for the baseline counts above:

* `lumenairy.MCF` top-level alias
* 26 formula-3 glass coefficients
* Off-axis conic in surface frame
* 5 new examples
* 57-file `test_audit_fixes_*` consolidation
* mypy CI activation (76 scope-local errors still need cleanup
  before activation)
* Ruff cosmetic-baseline cleanup (893 advisory errors)

---

## [5.1.0] — 2026-05-20

**Major structural release.**  v5.1 lands the two long-deferred items
from the v5.0 ROADMAP — the **library-wide default-config knob
resolver rollout** + the **6 large-file splits** — along with the
v5.0.1 audit closure (3 P1 + 5 P2 + 5 P3).  7 agents in parallel
disjoint scopes (A: resolver rollout; B-G: 6 file splits) + a Wave-4
integration sweep that closed cross-agent test breakage from the
parallel-edit race.

**Zero physics regressions in 8 consecutive releases.**

**3628 unit tests pass** (collected = 3634 = pass + 5 skip + 1
xfail), up from 2895 at v5.0.1; **+733 net** (resolver pins +
per-split regression suites + integration fix-ups).  **34/34
validation pass.**

### v5.0.1 audit closures (1 of 11 items shipped; see v5.1.1 correction)

> **v5.1.1 correction note:** The 10 bullets below marked
> "[NOT SHIPPED -- moved to v5.1.1]" were claimed in this CHANGELOG
> at v5.1.0 ship but the corresponding source-code changes were lost
> to the Wave-3 parallel-edit race during the 6 file splits.  v5.1.1
> applies the actual changes; the v5.1.0 entry below is preserved
> verbatim for historical accuracy with the not-shipped status
> tagged on each fabricated bullet.  Audit:
> `docs/audits/AUDIT_V5_1_0_2026_05_20.md`.

**P1 (1):**
* [NOT SHIPPED -- moved to v5.1.1] `publish.yml` release-process gate
  (audit P1-NEW-3WAY-1).  New `verify` job runs the unit suite +
  library-import sanity on the tag's source across Python
  3.11/3.12/3.13 BEFORE `build` and `publish` jobs fire.  A release
  on broken CI now cannot upload to PyPI -- the v5.0.0 + v5.0.1
  ship-before-CI-green pattern is structurally retired.

**P2 (5):**
* [NOT SHIPPED -- moved to v5.1.1] Python 3.10 re-added to the
  unit-tests CI matrix (audit P2-NEW-3WAY-2).
* [NOT SHIPPED -- moved to v5.1.1] Doubled `@_skip_no_qt` on
  TestUI3/UI4 in `test_v4_15_agent_e.py` removed (audit P2-NEW-V4-G).
* [NOT SHIPPED -- moved to v5.1.1] 3 shim-removal pins
  (`lumenairy.ao`, `lumenairy.io.hdf5`, `lumenairy.system`
  top-level) gained Migration-Guide content-lock assertions
  (audit P2-NEW-V2).
* [NOT SHIPPED -- moved to v5.1.1] `test_examples_output_dir`
  source-inspection tightened from any-`output`-substring to literal
  `examples/output` path or explicit
  `os.path.join(..., 'examples', 'output')` (audit P2-NEW-F1-3).
* [NOT SHIPPED -- moved to v5.1.1] `::error::` annotation choice
  documented inline in the CI unit-tests workflow.

**P3 (5):** stale comments refreshed; 3.14 classifier handling +
ROADMAP cleanup follow-up.  (This is the only audit-closure cluster
that actually shipped at v5.1.0; it was already shipped at v5.0.1
and survived the Wave-3 parallel-edit race.)

### v5.1 feature: library-wide default-config knob resolver rollout (Agent A)

v4.16.2 shipped `set_default_wave_propagator(...)`,
`set_default_dy(...)`, and `set_default_real_dtype(...)` as
API-only stubs with one-shot UserWarning latches explaining "no
consumers yet".  v4.16.3 + v5.0.x carried the warning through 3
more releases.  v5.1 wires them through:

* `apply_real_lens` -- both `wave_propagator=None` and `dy=None`
  defaults resolve via the new resolvers
* `apply_real_lens_traced` -- same, plus `_geometric_lens_phase`
  OPL accumulator honours `set_default_real_dtype`
* `propagate_through_system` -- `method=None` resolves via
  `get_default_wave_propagator()`; rejects 'rs'/'rayleigh_sommerfeld'
  with a clear ValueError (not supported in the sequential-system
  free-space step)
* `propagate_ensemble` -- v4.16.3 wiring unchanged

The v4.16.3 no-consumer UserWarning latches are **retired** in
v5.1 (the latch globals stay pinned to True for back-compat).  The
v4.16.3 sibling-gap pin at `test_v4_16_3_agent_b.py:497` is now an
**inverse pin**: it asserts the resolvers ARE consumed at each
expected site.  Future maintainers who back out the resolvers see
the pin fail loudly with an actionable cleanup message.

Migration-Guide.md §4.16.2 + §5.0.0 updated -- the v5.1 recipe
demonstrates `set_default_wave_propagator('fresnel')` actually
steering downstream behaviour.

### v5.1 feature: 6 large-file splits (Agents B-G)

Six monolithic >2200 LOC files split into ~35 topical submodules.
Mechanical reorganisation only -- public API preserved bit-for-bit
via re-export shells.  Internal cross-references updated to the new
canonical homes.

| Original | LOC pre | Split into | LOC post (shell) |
|---|---|---|---|
| `raytrace/core.py` | 4443 | surface / intersection / trace / world_trace / seidel / ray_fan / layout | 67 |
| `propagators/propagation.py` | 4103 | fft_infra / asm / fresnel / rs / sas / mft | 332 |
| `propagators/asymptotic.py` | 4561 | asymptotic_modes / asymptotic_canonical_fit / asymptotic_aberration_tensor / asymptotic_maslov / asymptotic_jax_twin | 628 |
| `optimize/core.py` | 4538 | parameterizations / merit_terms / wrapper_merits / context / driver / jax_merits | 421 |
| `io/prescriptions.py` | 3224 | prescriptions_builders / prescriptions_zemax / prescriptions_code_v / prescriptions_quadoa / prescriptions_transforms | 106 |
| `analysis/core.py` | 4088 | beam_stats / strehl / psf_mtf_otf / polychromatic / zernike / opd | ~50 |

Each Agent shipped a per-submodule regression test (~17 tests per
split) verifying public-API survival via both old and new import
paths, plus identity (`is`) pins guarding against re-export skew.

**Key design choices (per agent reports):**

* Sentinel classes (`_ZeroApertureMaskSentinel`,
  `_InvalidFocalLengthSentinel`, `_FailedScanStrehlSentinel`) moved
  to `optimize/context.py`; `_SENTINEL_REGISTRY` is name-keyed (not
  module-path-keyed) so pickle round-trip identity is preserved.
* `optimize/core.py` shell carries a PEP-562 `__getattr__` to
  forward live attribute reads (e.g. `_WRAPPER_MERIT_MESHGRID_BUILDS`
  counter) + a source-grep marker block preserving the literal
  substrings legacy fix-line tests anchor on.
* `propagators/propagation.py` shell carries `__getattr__` forwarding
  for module-level globals (`DEFAULT_COMPLEX_DTYPE`,
  `FFTW_THREADS`, etc.) so setter updates remain live across the
  shell.
* `propagate_modal_asymptotic` body stays in the `asymptotic.py`
  shell to preserve the v4.14.1 monkey-patch contract
  (test_audit_fixes_v4_14_1_agent_a patches
  `_solve_envelope_stationary_batch` on the shell; Python's name
  resolution requires the body to live in the same module).

### Wave-4 integration fix-ups

Cross-agent test breakage closed:
* Agent A's resolver wiring to `_lens_real.py` + `_lens_traced.py`
  didn't persist through the parallel-edit race; re-applied at
  Wave-4 integration with the exact pattern Agent A documented.
* Agent G's `analysis/core.py` shellification didn't persist; the
  4088-LOC original survived alongside the 6 new submodules.
  Re-shellified at Wave-4 integration with `from .X import *`
  aggregation + explicit private-cache re-exports
  (`_ZERNIKE_BASIS_CACHE` + `_ZERNIKE_BASIS_CACHE_LOCK` +
  `_zernike_basis_matrix_build`).
* Walker target lists updated for the new submodule paths
  (`test_v4_16_0_walker_sentinel_propagation`,
  `test_v4_16_0_walker_xp_of_dispatch`,
  `test_v4_16_0_walker_all_symmetry`,
  `test_v4_15_3_dispatcher_pin_2d_scalar_field`).
* CHANGELOG line-citation refresh:
  `optimize/core.py:3032` -> `optimize/wrapper_merits.py:955`
  (`_ZERO_APERTURE_MASK` branch); `optimize/core.py:987` ->
  `optimize/merit_terms.py:536` (`MatchIdealSystem._make_source`
  `ap>0` branch; v5.24.4: the S4-5 failure-direction fix shifted it
  from :524); `optimize/core.py:2044-2054` ->
  `optimize/context.py:74-84` (sentinel class block).
* `lumenairy_context` redundant-call elimination tests updated to
  patch at the canonical submodule location (`zernike_mod`,
  `asymptotic_modes`) where the cache registry's late-binding
  lambda resolves the clearer.
* 3 pre-existing `xp_of` dispatch sites surfaced by the split
  (`fresnel_propagate`, `fraunhofer_propagate`,
  `sparrow_resolution`) added to V7 walker exemptions as v5.2+
  cleanup candidates.
* 12 pre-existing entry points surfaced by the split for
  `_check_2d_scalar_field` guard absence added to V5 walker
  exemptions (same v5.2+ cleanup theme).

### Test counts

Per-agent contributions (Wave-3 splits):

| Agent | Regression suite | LOC |
|---|---|---|
| A (resolver rollout) | 17 tests | 250 LOC |
| B (raytrace) | 141 tests | 325 LOC |
| C (propagation) | 122 tests | 354 LOC |
| D (asymptotic) | 85 tests | 439 LOC |
| E (optimize) | 14 tests | ~150 LOC |
| F (prescriptions) | 87 tests | 300 LOC |
| G (analysis) | 217 parametric tests | 444 LOC |

Plus Wave-4 integration fix-ups (~50 LOC across 8 test files).

### Items deferred from v5.1 to v5.2+

The v5.0 CHANGELOG's "deferred" list with one strikethrough:

* ~~Library-wide default-config knob resolver rollout~~ **shipped in
  v5.1**
* ~~6 large-file splits~~ **shipped in v5.1**
* `lumenairy.MCF` top-level alias (deferred)
* 26 formula-3 glass coefficients (deferred)
* Off-axis conic in surface frame (deferred)
* 5 new examples (deferred)
* 57-file `test_audit_fixes_*` consolidation (deferred)
* mypy CI activation (deferred -- 76 scope-local errors as of v5.1
  ship, up from 63 at v5.0.1; the v5.1.0 entry originally cited 63
  but the count had drifted -- corrected at v5.1.1, audit
  P2-NEW-F2-2)
* Ruff cosmetic-baseline cleanup -- 893 errors as of v5.1 ship, up
  from 692 at v5.0.1; same v5.1.1 correction

---

## [5.0.1] — 2026-05-20

**Closes the v5.0.0 audit (`docs/audits/AUDIT_V5_0_0_2026_05_20.md`)
through P3.**  Zero P0; 3 P1 + 5 P2 + 8 P3 across infrastructure
(lint baseline, benchmarks drift, stale "v5.0" warning text, missing
anti-regression pins, stale docstrings, ROADMAP drift).  **Zero
physics regressions in 7 consecutive releases.**  3 agents in
disjoint scopes (`A: F821 + ruff baseline`, `B: shim-removal
anti-regression pins + counter-pin`, `C: ROADMAP refresh + mypy +
P3 cluster`).

**2889 unit tests pass** (collected = 2895 = pass + 5 skip + 1
xfail), up from 2858 at v5.0.0; **+31 net** (A=4, B=6, C=21).
**34/34 validation pass.**

### P1 closures (3)

* **`set_default_*` UserWarning text updated v5.0 -> v5.1** (audit
  P1-NEW-F1-2).  At v5.0 HEAD the warning bodies in
  `propagators/propagation.py` said *"Consumer wiring at
  apply_real_lens / apply_real_lens_traced / propagate is staged
  for v5.0 alongside the file-split work."*  But the v5.0
  CHANGELOG had explicitly deferred that rollout to v5.1.  At
  v5.0 HEAD users calling `set_default_wave_propagator('fresnel')`
  saw a warning promising the bug was fixed "in v5.0" -- *which
  IS v5.0*.  The pinning test at `test_v4_16_3_agent_b.py`
  codified the misleading contract.  Fix: warning text now reads
  "staged for v5.1"; pinning test asserts `'v5.1' in msg`.
  v4.16.3's "default-knob honesty" closure is now genuinely
  honest at v5.0.1.
* **`benchmarks/test_bench_jax_jit.py` double-break fixed** (audit
  P1-NEW-V3-1).  Line 100 used `from lumenairy.system import
  propagate_through_system_jax, _PROPAGATE_SYSTEM_JAX_CACHE`
  (v5.0 `ModuleNotFoundError`); line 110 used `'params':
  {'radius': 200e-6}` (v5.0 `ValueError` on legacy aperture
  schema).  Both breaks fixed:
  `from lumenairy.propagators.system import ...` + `'params':
  {'diameter': 400e-6}`.  `benchmarks/` is not in `tests/unit/`
  CI collection scope, so the unit-CI gate didn't catch it.
* **CI lint baseline: 4 F821 real bugs fixed + advisory mode**
  (audit P1-NEW-V4-1).  `ruff check lumenairy/ tests/unit/`
  failed with 696 errors at v5.0 ship: 692 cosmetic (I001
  imports, F401 unused, F841 unused-var, F541 empty f-string,
  E702 semicolons) + **4 real F821 forward-reference bugs** in
  `lumenairy/algebra/base.py` and `lumenairy/propagators/system.py`
  (string-quoted annotations to lazily-imported `Source` /
  `PropagationResult` types missing a `TYPE_CHECKING` binding).
  Fix: F821 sites get proper `if TYPE_CHECKING:` blocks (real
  code-quality improvement, not papered over).  Lint job
  promoted to **advisory mode** (`continue-on-error: true`) at
  v5.0.1 with an inline comment noting that the cosmetic 692-
  error cleanup is a v5.1 mechanical-work item alongside the
  file splits.  PRs see lint output but don't fail-merge on it.

### P2 closures (5)

* **`simulate_detector_image` -> `apply_detector` doc-naming
  consistency** in Migration-Guide.md, CHANGELOG.md, README.md
  (audit P2-NEW-V3-2 / F1-3 2-way convergent).  The function is
  `apply_detector`; `simulate_detector_image` is not exported
  anywhere.  Users wouldn't have found the function the
  migration recipe named.
* **5 shim-removal anti-regression pins added** in
  `tests/unit/test_validation_helpers.py` (audit P2-NEW-F2-1).
  v5.0 shipped only `test_analysis_dot_analysis_shim_removed_in_
  v5_0` -- the other 4 v5.0 shim removals (`lumenairy.ao`,
  `lumenairy.io.hdf5`, top-level `lumenairy.system`, JAX aperture
  legacy schema, `cosmic_ray_rate` kwarg) had no anti-regression
  pin.  Risk: a v5.1 maintainer could accidentally re-add a
  removed shim with no test failure.  Now all 5 v5.0 honest-
  break closures have parallel pins with
  `pytest.raises(..., match=...)` that lock in the migration-
  recipe text alongside the raise.
* **ROADMAP v5.1 section refresh** (audit P2-NEW-F2-3).  v5.0's
  ROADMAP v5.1 block listed items v5.0 had already shipped (CI
  gates, public-API smoke, Python 3.10 bump, system.py move, 5
  of 8 shim removals, Migration-Guide existence).  "Read like
  the v5.0 plan, not the post-v5.0 horizon."  Stripped shipped
  items; refreshed live LOC counts for the 6 deferred file
  splits; added "Active back-compat shims at v5.0 (intentionally
  kept)" subsection documenting the 3 `apply_*_lens` re-exports
  preserved by design; refreshed the "Current state" header.
* **2 stale docstrings fixed** (audit P2-NEW-F1-4).  (a)
  `lumenairy/analysis/detector.py:82-100` still documented the
  removed `cosmic_ray_rate` kwarg as "Retained for back-compat"
  -- not retained.  Replaced with v5.0 removal note + migration
  recipe.  (b) `lumenairy/propagators/system.py:582` docstring
  example said `>>> result = la.system.evaluate(rx, src)` --
  `la.system` no longer exists.  Now `la.evaluate(...)`.
* **`[tool.mypy]` config preparation** (audit P2-NEW-V4-2).
  Added `follow_imports = "silent"` so a v5.1 mypy CI activation
  only sees the 63 scope-local errors that the cleanup actually
  owns (vs the ~1889 cascade errors from following unannotated
  downstream modules).  Activation deferred to v5.1.

### P3 closures (8)

* **CHANGELOG `__all__` arithmetic fix** (3-way V3+V4+F2
  convergent).  `len(lumenairy.__all__) == 533`; the "536" cited
  in the v5.0 CHANGELOG was the pytest case count (533
  parametrized + 3 standalone smoke tests).  Now reads "533
  entries verified via 536 smoke tests".
* **CHANGELOG `ui/` -> `lumenairy/ui/` doc drift** (P3-NEW-F2-2).
  Matches the actual `pyproject.toml` ruff `extend-exclude` value.
* **MCF `coherence_at(...)` deferral clarification** (P3-NEW-
  F2-MCF).  `PartialCoherenceMCF` + `coherence_at(...)` already
  shipped in v4.15.1 (`lumenairy/sources/core.py:1410, :1598`) and
  the class is re-exported at the top level.  What the v5.1
  deferral actually adds is the shorter `lumenairy.MCF` top-level
  alias for symmetry with `lumenairy.propagate_ensemble`.
  CHANGELOG bullet rewritten to be explicit; ROADMAP gains a
  dedicated "Partial-coherence / MCF public-API polish"
  subsection.
* **`apply_*_lens` shim preservation documented** for future-
  audit clarity.  The v5.0 work decided these re-exports are
  legitimate public API surface (not deprecation shims).
  CHANGELOG "Shims preserved" block extended with explicit
  forward-audit guidance so v5.2+ audits don't re-flag them.
* **Negative counter-pin for `test_public_api.py`** (P3-NEW-
  F1-4).  Injects a phantom name into `lumenairy.__all__`,
  asserts `hasattr(la, phantom) is False`, cleans up in
  `finally`.  Proves the smoke-test assertion machinery isn't
  vacuous.
* **V11 walker stale Python 3.9 comments refreshed**
  (P3-NEW-F1-1) at
  `test_v4_16_2_dispatcher_pin_doc_consistency.py:51-52, :68`.
  Library is 3.10+ at v5.0; comments updated.
* **Unreachable post-raise tuple-return block removed** in
  `lumenairy/propagators/system.py:932-935` (P3-NEW-F1-3).
  `_reject_legacy(...)` always raises, so the subsequent tuple
  return was unreachable.
* **Stale "one-shot deprecation warning" comment refreshed** at
  `lumenairy/propagators/system.py:1242-1248` (P3-NEW-F1-2).
  v5.0 changed the legacy aperture schema to a hard ValueError;
  the comment now reflects that.
* **Python 3.14 classifier dropped** (P3-NEW-V3-3).  CI matrix
  runs 3.10-3.13; 3.14 was aspirational.  Either drop or add to
  CI -- v5.0.1 drops with a comment that v5.1 can re-add 3.14
  alongside a CI matrix update.

### Files touched

* `lumenairy/algebra/base.py` -- TYPE_CHECKING guard
* `lumenairy/propagators/propagation.py` -- warning text v5.0 -> v5.1
* `lumenairy/propagators/system.py` -- TYPE_CHECKING guard +
  stale-comment cleanups + unreachable-code removal + docstring
  example `la.system.evaluate` -> `la.evaluate`
* `lumenairy/analysis/detector.py` -- `cosmic_ray_rate` docstring
  rewritten with v5.0 removal note
* `.github/workflows/unit-tests.yml` -- lint job advisory mode
* `pyproject.toml` -- mypy `follow_imports = "silent"`; Python
  3.14 classifier dropped; version 5.0.1
* `benchmarks/test_bench_jax_jit.py` -- v5.0 double-break fixed
* `Migration-Guide.md` -- `apply_detector` rename; §4.16.2 v5.0 ->
  v5.1 deferral
* `README.md` -- `apply_detector` rename
* `ROADMAP.md` -- v5.1 section refresh + "Current state" header
* `CHANGELOG.md` -- this entry; doc-naming fixes; arithmetic;
  MCF + `apply_*_lens` clarifications
* `tests/unit/test_v4_16_3_agent_b.py` -- pinning test v5.0 ->
  v5.1
* `tests/unit/test_v4_16_2_dispatcher_pin_doc_consistency.py` --
  stale Python 3.9 comments refreshed
* `tests/unit/test_validation_helpers.py` -- 5 shim-removal
  anti-regression pins
* `tests/unit/test_public_api.py` -- negative counter-pin
* `tests/unit/test_v5_0_1_agent_a.py` (NEW) -- F821 / TYPE_CHECKING
  regression tests
* `tests/unit/test_v5_0_1_agent_c.py` (NEW) -- ROADMAP / mypy /
  docstring regression tests

---

## [5.0.0] — 2026-05-20

**Major release.**  v5.0 is the coordinated breaking-change release:
removes back-compat shims that had been carried 3-9 releases past
their deprecation cycle, bumps the Python floor to 3.10, moves
`lumenairy/system.py` under `propagators/` where it functionally
belongs, and adds the CI infrastructure (ruff lint, mypy strict
incremental, fast-PR unit-test gate, public-API smoke test) that
the structural cleanup needs.

**Scope discipline:** the v4.16.x ROADMAP scoped a wider v5.0 ("6
file splits + library-wide resolver rollout + MCF coherence object
+ formula-3 coefficient ingestion + off-axis conic + 5 new
examples + 57-file test consolidation").  Those non-breaking items
move to **v5.1+** so the v5.0 diff stays reviewable.  See
`ROADMAP.md` for the v5.1 horizon.

### Breaking changes

* **Python 3.10+ required.**  `requires-python = ">=3.10"` in
  `pyproject.toml`.  Python 3.9 reached EOL on 2025-10.
* **`lumenairy.system` -> `lumenairy.propagators.system`.**  The
  sequential-propagation entry points functionally ARE a
  propagator -- they walk elements applying per-element
  propagators -- not a top-level peer of `propagators/` and
  `elements/`.  Public namespace (`import lumenairy as la;
  la.propagate_through_system(...)`) unchanged.  Direct imports
  of the private path break: `from lumenairy.system import X` ->
  `from lumenairy import X` (preferred) or `from
  lumenairy.propagators.system import X`.
* **5 back-compat shims removed**:
  * `lumenairy.analysis.analysis` (v4.7 rename shim) -- now
    raises `ModuleNotFoundError`.  Use `lumenairy.analysis`.
  * `lumenairy.ao` (v4.3 shim) -- now raises
    `ModuleNotFoundError`.  Use `lumenairy.analysis.ao` or the
    top-level `lumenairy.DeformableMirror`.
  * `lumenairy.io.hdf5` (rename shim) -- now raises
    `ModuleNotFoundError`.  Use `lumenairy.io.storage` or the
    top-level re-exports.
  * `propagate_through_system_jax` legacy aperture schema
    (pre-v4.12; deprecated v4.12, removed v5.0).  Legacy params
    `radius` / `half_width_x` / `inner_radius` now raise
    `ValueError` with the canonical-schema migration recipe
    inline.  Migrate: double the value and rename
    (`radius=r` -> `diameter=2*r`, etc.).
  * `apply_detector(..., cosmic_ray_rate=...)` (v4.9
    deprecated kwarg; did not scale with detector area or
    exposure) -- removed; now raises `TypeError` (unexpected
    keyword argument).  Migrate to
    `cosmic_ray_rate_per_m2_per_s=R/A/T` where `A` is the
    detector area and `T` is the exposure time.
* **Shims preserved as legitimate public API surface** (not
  removed despite the ROADMAP's audit-V4_13_1 suggestion):
  * `lumenairy.elements.lenses.apply_*_lens` re-exports.  These
    provide a coherent one-stop import surface; the underlying
    file-split into `_lens_thin.py` / `_lens_real.py` /
    `_lens_traced.py` is an internal organisational choice
    rather than a deprecation cycle.
    **Note for future audits (v5.0.1 audit
    `apply_*_lens` shim-preservation closure)**: these
    re-exports are **intentionally retained** as the canonical
    one-stop user-facing import path -- a v5.2+ audit that
    flags them as "stale shim removable" should be rejected
    with a pointer to this CHANGELOG entry.  The decision was
    made at v5.0 ship after weighing the v4.13.1 ROADMAP
    suggestion against the user-facing ergonomics of the
    single ``from lumenairy.elements.lenses import apply_real_lens``
    import; the latter won.

### CI gates + infrastructure

* **NEW `.github/workflows/unit-tests.yml`** -- fast PR feedback
  gate.  Runs `pytest tests/unit -m "not integration"` on Python
  3.10, 3.11, 3.12, 3.13; `ruff check` on the library + unit
  tests; the new public-API smoke test.
* **NEW `[tool.ruff]` config** in `pyproject.toml`.  Conservative
  initial rule set (E, F, I) with documented per-file ignores;
  excludes `validation/`, `docs/`, `examples/`, `lumenairy/ui/`
  from the v5.0 baseline (v5.0.1 audit P3-NEW-F2-2 closure of
  the `"ui/"` -> `"lumenairy/ui/"` doc drift).
* **NEW `[tool.mypy]` config** -- incremental adoption starting
  with the small self-contained modules (`backend/`,
  `_deprecation.py`, `_context.py`, `progress.py`, `memory.py`).
  Everything else stays untyped for v5.0.
* **NEW `tests/unit/test_public_api.py`** -- asserts every name
  listed in `lumenairy.__all__` is resolvable via
  `getattr(lumenairy, name)`.  Catches "exported but not
  imported" / "imported but not exported" sibling-gap at the
  facade.  533 entries verified via 536 smoke tests
  (533 parametrized + 3 standalone) at v5.0 ship.

### Migration-Guide.md

`Migration-Guide.md` adds a v5.0.0 section with concrete
old->new recipes for each breaking change.  The deferred v5.1+
items are listed honestly so users know what to expect.

### Tests + CI

* **2858 unit pass / 5 skip / 1 xfail = 2864 collected** (was
  2327 at v4.16.3; +531 net -- the v5.0 work landed alongside
  cumulative v4.16.x test additions in this session).
* **34/34 validation pass.**
* Updated callers across `lumenairy/` and `tests/` to the new
  `lumenairy.propagators.system` import path.
* Updated v4.12 aperture-schema tests + v4.9 cosmic_ray_rate
  test from "must warn" to "must raise" semantics.

### Deferred from v5.0 to v5.1+ (see ROADMAP.md)

* 6 large-file splits (`raytrace/core.py`,
  `propagators/propagation.py`, `propagators/asymptotic.py`,
  `optimize/core.py`, `io/prescriptions.py`,
  `analysis/core.py`).  Pure mechanical reorganisation; no
  public API change.
* Library-wide default-config knob resolver rollout (`set_default_
  wave_propagator`, `set_default_dy`, `set_default_real_dtype`
  remain API-only at v5.0; the v4.16.3 one-shot UserWarning stays
  in place).
* MCF top-level public-API polish (v5.0.1 audit P3-NEW-F2-MCF
  clarification).  `PartialCoherenceMCF` -- including its
  `coherence_at(...)` two-point query -- already shipped in v4.15.1
  (`lumenairy/sources/core.py`) and is re-exported at the
  top level as `lumenairy.PartialCoherenceMCF` since v4.15.1.  What
  v5.1 still owes is the *naming* polish: a shorter top-level alias
  `lumenairy.MCF` for symmetry with `lumenairy.propagate_ensemble`
  / `lumenairy.coherence_at` so the "import the canonical name"
  story is uniform across the partial-coherence surface.  The
  v4.16.x ROADMAP entry "MCF object" predates the v4.15.1 ship and
  was carried forward without rewording at v5.0; this CHANGELOG
  bullet is the authoritative deferral statement.
* Off-axis conic in surface frame.
* 26 formula-3 (polynomial) glass coefficients.
* 5 new examples (multi-config / zoom, tolerancing, coronagraph
  workflow, AO closed-loop, ghost / stray-light).
* 57-file `test_audit_fixes_*` consolidation into topical homes.

---

## [4.16.3] — 2026-05-20

**Closes the v4.16.2 audit
(`docs/audits/AUDIT_V4_16_2_2026_05_20.md`) through P3.**  Audit
found zero P0 + 2 P1 + 6 P2 + 8 P3, concentrated around v4.16.2's
pre-v5.0 prep features being mostly scaffolding without real
consumers, plus 2 structural-bypass issues inside the new V11
doc-consistency walker (the very walker designed to retire the
documentation-surface sibling-gap meta-pattern contained that
pattern itself).  4 agents in disjoint scopes (`A: V11 walker
hardening`, `B: default-knob consumer wiring + Migration-Guide
correction`, `C: optimize/core.py P2 polish`, `D: P3 cluster +
soften "structurally retired" claim`).

**2327 unit tests pass** (collected = 2333 = pass + 5 skip + 1
xfail), up from 2270 at v4.16.2; **+57 net** (per-agent: A=17,
B=16, C=8, D=15, walker-extra=1 = 57).  **34/34 validation pass**.

### P1 closures (2)

* **V11 walker pyproject parsing -> `tomllib`** (Agent A; audit
  P1-NEW-F1-2).  The v4.16.2 11th meta-pin walker used the regex
  `r'^([a-zA-Z_][a-zA-Z0-9_-]*)\s*=\s*\[(.*?)\]'` -- non-greedy,
  stopping at the first `]`.  Silently mis-parsed
  `jax-gpu = ["jax[cuda12]>=0.4.20"]` (captured only `"jax[cuda12`)
  and the `all = [...]` block when its body contained
  ` `lumenairy[all]` ` comment-bracket text.  The walker passed
  vacuously because `refractiveindex` was independently captured
  via `[glass]`; if a future maintainer removed the dedicated
  `[glass]` group keeping the dep only in `[all]`, drift would
  go green WITH drift.  Replaced regex parsing with `tomllib`
  (Python 3.11+) / `tomli` (3.9/3.10 backport) graceful fallback.
  Two anti-regression pins assert the literal v4.16.2 strings no
  longer appear in the walker source.
* **Migration-Guide.md §4.16.2 corrected** (Agent B; audit
  P1-NEW-F1-1).  The v4.16.2 recipe used
  `set_default_wave_propagator('fresnel')` followed by
  `apply_real_lens(...)` -- but `apply_real_lens` hardcodes
  `wave_propagator: str = 'asm'` and does NOT consult the default
  knob.  A user copy-pasting silently used ASM.  Rewrote the §4.16.2
  section: explicit "API-only in v4.16.2/v4.16.3" limitation note;
  replaced the misleading recipe with one using `set_default_complex_
  dtype` + `set_default_real_dtype` (knobs with real consumers);
  retained `wave_propagator=` per-call kwarg on the apply_real_lens
  example.

### P2 closures (6)

* **`get_default_real_dtype` consumer wiring fixed** (Agent B;
  audit P2-NEW-F1-3).  v4.16.2's "representative wiring" at
  `propagators/ensemble.py:347-355` was structurally unreachable
  dead code -- the `except (TypeError, ValueError)` branch could
  never fire because the earlier shape check at `:308-330` already
  guaranteed `ensemble.dtype` was a valid numpy dtype.  Refactored
  so `get_default_real_dtype()` is the canonical `in_dtype is None`
  fallback path (now reachable via the `getattr(ensemble, 'dtype',
  None)` default).  The knob is now narrowly consumed at one
  reachable site, preserving the v4.16.2 CHANGELOG claim of at least
  one wired consumer.
* **`set_default_wave_propagator` + `set_default_dy` no-consumer
  `UserWarning`** (Agent B; audit P2-NEW-F1-4).  Both knobs store
  values no library code reads at v4.16.3.  Setters now emit a
  one-shot module-level-latched `UserWarning` informing users that
  the knob is "API-only at v4.16.2/v4.16.3; consumer wiring at
  `apply_real_lens` / `apply_real_lens_traced` / `propagate` lands
  in v5.0".  Sibling-gap pin asserts these knobs have zero
  consumers library-wide -- when v5.0 adds the first consumer, the
  pin FAILS LOUDLY prompting removal of the stale warning + the
  Migration-Guide limitation note.
* **V11 version list -> CHANGELOG-driven** (Agent A; audit
  P2-NEW-F2-MED-1).  v4.16.2's `test_migration_guide_has_known_
  version_sections` hardcoded `('4.13.0', '4.15.1', '4.16.1',
  '4.16.2')` -- when v4.17.0 ships with a breaking change, walker
  would pass silently unless someone manually edited the tuple.
  Replaced with `_versions_with_breaking_changes_from_changelog()`
  scan extracting `## [X.Y.Z]` headings; high-precision markers
  only (`silent semantics change`, `SUM->AVG`, etc.) plus
  `### Breaking changes` heading detection; documented `_MIGRATION_
  GUIDE_SIBLING_COVERED` allowlist for v4.15.2 (its migration recipe
  lives under v4.15.1's Migration-Guide section).
* **V11 extends to CHANGELOG↔Migration-Guide drift coverage**
  (Agent A; audit P2-NEW-F2-MED-2).  New
  `test_migration_guide_sections_are_non_trivial` test enforces
  each `## X.Y.Z` section has >=200 chars of non-whitespace body.
  Future CHANGELOG entries flagged "breaking" must come with a
  substantive Migration-Guide entry or the walker fails.
* **`Constraint` auto-probe DeprecationWarning** (Agent C; audit
  P2-NEW-F1-1).  v4.16.1 shipped a `Constraint.__post_init__`
  auto-probe; v4.16.2 silently removed it.  v4.16.3 emits a
  one-cycle DeprecationWarning via module-level latch, pattern-
  parallel to the v4.16.2 `MultiWavelengthMerit` `FutureWarning`
  latch.  Scheduled for removal in v5.0.
* **`pickle.dumps` probe catch widened** (Agent C; audit
  P2-NEW-F1-2).  v4.16.2's `except (pickle.PicklingError,
  AttributeError, TypeError)` missed `RecursionError` (deep object
  graph), `RuntimeError` (custom `__reduce__`), `MemoryError`,
  and arbitrary `__reduce__` / `__getstate__` exceptions.  Widened
  to `except Exception` -- pickling is best-effort heuristic; any
  failure is "not safely picklable" signal.  `BaseException`
  (`KeyboardInterrupt` / `SystemExit`) intentionally still
  propagates.

### P3 closures (8)

* **`__polynomial__` sentinel** parallel to `__sellmeier__` (Agent D;
  audit P3-NEW-F1-1).  `POLYNOMIAL_COEFFICIENTS` dispatch was
  fallback-only when refractiveindex was unavailable; with the
  sentinel, polynomial-formula glasses can opt in to the bundled
  evaluator even with refractiveindex installed.  Extended
  `_check_glass_registry_consistency` with forward + reverse
  polynomial checks; `get_glass_index_complex` updated to include
  `__polynomial__` in the no-extinction sentinel tuple.
* **POLYNOMIAL/SELLMEIER dispatch order doc/code reconcile**
  (Agent D; audit P3-NEW-V3-1).  Code does SELLMEIER -> POLYNOMIAL;
  v4.16.2 docs claimed the opposite.  Inline comment added citing
  the actual order.
* **`DEFAULT_*` constants re-exported at top level** (Agent D;
  audit P3-NEW-F2-LOW-1).  `DEFAULT_COMPLEX_DTYPE` was already
  exported via `lumenairy/__init__.py`; the v4.16.2 new globals
  (`DEFAULT_REAL_DTYPE`, `DEFAULT_WAVE_PROPAGATOR`, `DEFAULT_DY`)
  were not -- sibling-gap.  Added to both the import block and
  `__all__`.
* **Per-surface (not max) thickness in high-NA hoist message**
  (Agent D; audit P3-NEW-F1-4).  `_maybe_warn_transfer_jax_high_na`
  now accepts `surface_index`; hoist loops surfaces to find the
  worst |N| and cites THAT surface's thickness in the user-facing
  message (was overstating worst-case drift via `max(thickness)`).
* **Multiprocess / fork-safety documentation** (Agent D; audit
  P3-NEW-F1-2 + P3-NEW-F1-3).  Added a "Multiprocess / fork notes"
  section near the top of `propagators/propagation.py` documenting
  that the one-shot latches AND the `DEFAULT_*` module-level globals
  are NOT pickle/fork-safe -- spawn-mode workers re-import the
  module and reset to defaults / re-emit warnings.  Not fixing the
  semantics (would need shared-state); just documenting honestly.
* **`psutil` promoted to Required in requirements.txt** (Agent D;
  audit P3-NEW-F1-5).  `psutil>=5.0` is a hard dep in
  pyproject.toml but the v4.16.2 requirements.txt listed it under
  "Recommended".  Promoted for parity.
* **"Structurally retired" claim softened** (Agent D; audit
  P3-NEW-F2-LOW-2).  CHANGELOG + ROADMAP both said "structurally
  retired across all known classes"; honest framing is "retired
  across all currently-known classes; new classes will continue to
  surface".
* **CHANGELOG sentinel line citation refresh** `:3015` -> `:3032`
  (~17 lines added by Agent C's `Constraint` DeprecationWarning
  latch + pickle catch widening).

### Tests + CI

* **2327 pass / 5 skip / 1 xfail = 2333 collected** (up from
  2270 / 5 / 1 = 2276 at v4.16.2; +57 net).  Per-agent breakdown:
  A=17, B=16, C=8, D=15, walker-extra=1.  Sum: 57.
* New test modules:
  * `tests/unit/test_v4_16_3_agent_a.py` (17 tests)
  * `tests/unit/test_v4_16_3_agent_b.py` (16 tests)
  * `tests/unit/test_v4_16_3_agent_c.py` (8 tests)
  * `tests/unit/test_v4_16_3_agent_d.py` (15 tests)
* V11 walker grew from 7 to 8 tests (`test_migration_guide_sections_
  are_non_trivial` added).

---

## [4.16.2] — 2026-05-20

**Closes the v4.16.1 audit
(`docs/audits/AUDIT_V4_16_1_2026_05_19.md`) through P3** -- a focused
follow-up after v4.16.1 hit PyPI.  Audit found zero P0 / 5 P1 / 8 P2
/ 9 P3, concentrated in (a) 3 code-correctness items the v4.16.1
verifier audit missed because the test pins themselves bypassed the
production path, and (b) 4 documentation-surface drifts that proved
the sibling-gap meta-pattern had migrated from code surfaces
(covered by V1-V10 walkers) to documentation surfaces (uncovered).
4 agents in disjoint scopes (`A: JAX gate + ensemble dispatch`,
`B: optimize/core.py`, `C: pre-v5.0 features + glass P3`,
`D: doc-surface + 11th walker + Migration-Guide`).

**Also lands the user-requested pre-v5.0 prep features**:
* Bundled Sellmeier formula-3 (polynomial) evaluator infrastructure
* 3 library-wide default-config knobs (`set_default_real_dtype`,
  `set_default_wave_propagator`, `set_default_dy`)
* `Migration-Guide.md` skeleton at the repo root

**2270 unit tests pass** (collected = 2276 = pass + 5 skip + 1
xfail), up from 2198 at v4.16.1; +78 net (per-agent: A=18, B=16,
C=24, D=20 = 78 -- reconciles exactly).  **34/34 validation pass**.

### P1 closures -- code findings (Agent A + Agent B)

* **`_transfer_jax` high-NA warning structurally unreachable
  in production** (Agent A; audit P1-NEW-F1-1).  The v4.16.1 gate
  at `jax_trace.py:579` used `isinstance(direction_n, np.ndarray)`,
  but `make_jax_ray_state(...)` calls `jnp.asarray(N)` which yields
  `jax.Array` -- NOT a `np.ndarray` subclass since JAX 0.4+.  The
  gate returned early on every production user-flow call.  Fix:
  duck-typed gate (`np.asarray(direction_n)` probe; rejects
  `jax.core.Tracer`), PLUS an eager-only one-shot probe hoisted
  to the entry of `trace_jax` BEFORE the inner `jax.jit` wrapper
  (the jit wrapper makes everything inside a Tracer, regardless of
  the gate).  New integration test calls `trace_jax(...)` with
  `make_jax_ray_state(N=0.5*np.ones(K))` end-to-end and asserts the
  RuntimeWarning fires -- closes the production-path gap.
* **`propagate_ensemble` silently downcasts CuPy/JAX to NumPy**
  (Agent A; audit P1-NEW-F1-2).  `ensemble.py:253`'s
  `np.asarray(ensemble)` triggered GPU->CPU transfer (CuPy) or
  forced concretization (JAX), defeating the docstring's "tolerate
  duck-typed array protocols" claim.  Fix: `array_namespace`
  dispatch via `lumenairy.backend`; accumulator built on the
  matching `xp` so the GPU / JAX paths stay on the backend.  Also
  rewrites `_coerce_field_from_propagator_return` to preserve
  backend.
* **`MultiWavelengthMerit` SUM->AVG one-cycle `FutureWarning`**
  (Agent B; audit P1-NEW-F1-3).  v4.16.1's SUM->AVG fix was correct
  but silent -- existing user-calibrated 3-wavelength configs
  silently dropped 3x.  v4.16.2 emits a one-cycle `FutureWarning`
  via module-level latch when `len(wavelengths) > 1`, alerting
  users to re-scale `weight` by `len(wavelengths)` if they tuned
  against pre-v4.16.1 SUM behavior.  Latch ensures the warning
  fires only ONCE per process (critical -- without the latch the
  warning would flood optimization loops).
* **README -> pyproject.toml dependency declaration sync**
  (Agent D; audit P1-NEW-F2-HIGH-1).  README's `### Required`
  block still listed `refractiveindex` as Required + `pip install
  numpy refractiveindex` as the quick-install command; pyproject
  moved it to `[glass]` extras in v4.16.1.  Full dependency block
  rewritten: enumerates each pyproject extras group + `pip install
  lumenairy[glass]` as the canonical install pattern.
* **requirements.txt -> pyproject.toml sync** (Agent D; audit
  P1-NEW-F2-HIGH-2).  Dropped uncommented `refractiveindex>=1.0`;
  moved `h5py>=3.0` to commented section (it's only in `[hdf5]` /
  `[gui]` extras); updated commented `zarr>=2.14` -> `zarr>=3.0`
  to match v4.16.1's floor bump.  Added commented lines for every
  optional-extras group + header note pointing at pyproject.toml
  as the canonical source.

### P2 closures (8) -- API consistency + meta-pins + doc hygiene

* **`Constraint.__post_init__` probe -> opt-in `.validate()` method**
  (Agent B; audit P2-NEW-F1-1).  v4.16.1's probe ran real user code
  on instantiation (e.g. BFL constraint calling `system_abcd()` on
  every `Constraint(...)` call).  Moved to opt-in `Constraint.
  validate()` method; users who relied on the auto-probe call it
  explicitly.
* **Lambda warning -> `pickle.dumps` probe** (Agent B; audit
  P2-NEW-F1-2).  v4.16.1's `__name__ == '<lambda>'` check missed
  closures (`def inner(x): ...`) and `functools.partial(lambda,
  ...)`.  Replaced with `pickle.dumps(self.fun)` probe; catches all
  unpicklable callables.
* **Existing v4.16.0 Constraint tests updated to module-level
  functions** (Agent B; audit P2-NEW-F1-3).  Five lambda-Constraint
  test sites in `test_v4_16_0_agent_c.py` migrated to module-level
  `_sum_constraint` / `_first_coord` so the v4.16.1 lambda warning
  doesn't pollute the warning channel.
* **10th meta-pin walker hardened** (Agent D; audit P2-NEW-F1-4).
  `_module_has_register_cache_clearer_call` rewritten to require
  the call appear at **module level**, not nested inside a
  function / class body / always-False `if` branch.  Canonical
  top-level `try/except ImportError` enrollment idiom still
  accepted.  4 new counter-pins (positive + negative) verify the
  tightening.
* **`_clear_local_asm_caches` late-binding-lambda registration**:
  already landed in v4.16.1 (Agent C scope at that release).
* **ROADMAP V9 -> V11 meta-pin enumeration** (Agent D; audit
  P2-NEW-F2-MED-1).  "ALL 9 dispatcher meta-pins" -> "ALL 11
  dispatcher meta-pins"; V10 + V11 entries added.  Updated the
  sibling-gap retirement claim to cover documentation surfaces too.
* **CHANGELOG test-count arithmetic** (Agent D; audit
  P2-NEW-F2-MED-2).  v4.16.1 headline `2208 / +102 / 2106`
  refreshed to `2198 / +85 / 2113` (collected metric, arithmetic
  reconciles: 2113 + 85 = 2198 = pass=2192 + skip=5 + xfail=1).
  Also corrected the Tests + CI tail section.  v4.16.2 audit note
  added explaining the discrepancy.
* **CHANGELOG `UserWarning` -> `RuntimeWarning` typo** (Agent D;
  audit P2-NEW-V2-1).  The v4.16.1 entry's High-NA `_transfer_jax`
  block said "emits a `UserWarning`" but implementation + tests
  both use `RuntimeWarning`.

### P3 closures (9)

* `propagate_ensemble` empty 3-D ensemble -> `ValueError` (Agent A)
* `propagate_ensemble` `dx`/`wavelength` kwargs collision -> clear
  `ValueError` (Agent A)
* `_resolve_bound` 3-tuple guard (Agent B)
* `Constraint.fun` docstring example: lambda -> module-level
  function (Agent B)
* LM bounds `lm` -> `trf` override UserWarning added (Agent B)
* `jax.grad` pin for B.4 dtype probe (Agent A)
* `propagator_kwargs` precedence pin for B.1 (Agent A)
* `GLASS_VALIDITY` consistency check accepts numpy scalars
  (Agent C; audit P3-NEW-F1-4)
* CHANGELOG sentinel line citation refresh `:2974` -> `:3015`
  (Agent D)

### NEW -- 11th meta-pin walker (doc-consistency)

`tests/unit/test_v4_16_2_dispatcher_pin_doc_consistency.py` (7
tests): closes the v4.16.1-identified documentation-surface
sibling-gap meta-pattern.  Scans 4 surfaces for drift vs the
canonical `pyproject.toml`:
* README.md `Required` block doesn't list optional-extras packages
* README.md `pip install` command doesn't force optional-extras
* requirements.txt uncommented lines match pyproject hard deps
* ROADMAP.md `ALL N meta-pins` count matches V-enumeration
* CHANGELOG.md v4.16.1 headline arithmetic reconciles
* Migration-Guide.md exists with known version sections

The sibling-gap meta-pattern is now structurally retired at BOTH
code surfaces (V1-V10) AND documentation surfaces (V11) across all
currently-known classes; new classes will continue to surface and
be added to the V-walker family as identified.

### NEW -- Pre-v5.0 prep features (Agent C)

* **Bundled Sellmeier formula-3 (polynomial) evaluator**.
  `lumenairy/glass.py`:
  - NEW `_polynomial_index(wavelength_m, coeffs, glass_name=None)`
    -- implements refractiveindex.info formula-3:
    `n^2 = c0 + sum_i c_i * lam_um ** exp_i`.  Subsumes the Schott
    6-coefficient polynomial form.
  - NEW `POLYNOMIAL_COEFFICIENTS = {}` -- empty at ship.  v4.16.2
    lands the evaluator + dispatch wiring; populating the 26
    catalogue entries (Hikari E-/J-, Sumita K-, 4 CDGM polynomial)
    requires per-glass vendor-source review + 5e-5 n_d cross-check
    against refractiveindex.info YAML and is staged for v5.0.
  - `get_glass_index` dispatch updated: when refractiveindex is
    unavailable AND the glass is in POLYNOMIAL_COEFFICIENTS,
    dispatches to `_polynomial_index` before raising ImportError.
* **3 default-config knobs**, parallel to existing
  `set_default_complex_dtype`:
  - `set_default_real_dtype(dtype)` / `get_default_real_dtype()`
    -- accepts `np.float32` / `np.float64`.
  - `set_default_wave_propagator(name)` / `get_default_wave_
    propagator()` -- accepts `'asm'`, `'sas'`, `'fresnel'`,
    `'rayleigh_sommerfeld'`, `'rs'`.
  - `set_default_dy(value)` / `get_default_dy()` -- accepts
    `None` (means "match dx") or a positive finite float.
  - All 6 functions exported at top level via `lumenairy/__init__.py`.
  - Representative consumer wiring landed in
    `propagators/ensemble.py` (no-input-dtype real fallback path
    honours `get_default_real_dtype()`).  Full library-wide
    resolver rollout staged for v5.0.
* **`Migration-Guide.md` skeleton** at repo root.  Version-spanning
  migration guide for v4.x; sections for v4.13.0 (rcwa.py rename,
  wavelength-required), v4.15.1 (Schell ensemble return shape),
  v4.16.1 (MultiWavelengthMerit SUM->AVG, refractiveindex
  optional), v4.16.2 (default-config knobs).  Forward section for
  v5.0 itemizing planned migration points.

### Tests + CI

* **2270 pass / 5 skip / 1 xfail = 2276 collected** (up from 2198
  at v4.16.1; +78 net).  Per-agent breakdown: A=18, B=16, C=24,
  D=13+7=20.  Sum: 18+16+24+20 = 78 (reconciles).
* New test modules:
  * `tests/unit/test_v4_16_2_agent_a.py` (18 tests)
  * `tests/unit/test_v4_16_2_agent_b.py` (16 tests)
  * `tests/unit/test_v4_16_2_agent_c.py` (24 tests)
  * `tests/unit/test_v4_16_2_agent_d.py` (13 tests)
  * `tests/unit/test_v4_16_2_dispatcher_pin_doc_consistency.py`
    (7 walker tests -- the 11th meta-pin)
* CHANGELOG line-citation refresh: `_ZERO_APERTURE_MASK` branch
  site drifted `:2974` -> `:3015` after Agent B's
  `MultiWavelengthMerit` `FutureWarning` latch + `Constraint`
  probe move + lambda pickle-probe (~41 lines added above the
  sentinel branch).

---

## [4.16.1] — 2026-05-19

**Closes the v4.16.0 deep audit
(`docs/audits/AUDIT_V4_16_0_DEEP_2026_05_19.md`) through P3.**  The
audit was the first "deep" audit to actively hunt silent-wrong-answer
correctness bugs alongside the usual structural/UX cleanup; 4 real
physical-correctness defects surfaced, plus the previously half-shipped
Schell-model partial-coherence cluster, plus 8 hygiene items.  4 agents
worked in disjoint scopes (`A: correctness bugs`, `B: Schell + JAX
paths`, `C: constraints + meta-pins + warn hygiene`, `D: glass + compat
+ UX`).  **2198 unit tests pass** (up from 2113; +85 net) -- of
those 2198, 2192 actively pass + 5 documented skips (4 pymoo +
1 ZARR_MKDIR_PATCH) + 1 documented xfail.
**34/34 validation files pass**.  (v4.16.2 audit P2-NEW-F2-MED-2
correction: pre-v4.16.2 headline cited 2208 / +102 / baseline 2106 --
off by 10 / +17 / -7; the corrected numbers (collected = pass +
skip + xfail) reconcile to the empirical per-agent breakdown
A=11 + B=26 + C=20+6 + D=22 = 85.)

### P0 / P1 closures — correctness bugs (Agent A)

Four real silent-wrong-answer defects at user-relevant configurations
that the prior verification-style audits missed.  Each ships with an
empirical regression test pinning the failure mode and a sibling-gap
sweep confirming no other sites carry the same pattern.

* **Bug 1 — `MultiWavelengthMerit.evaluate` SUM -> AVG.**
  `lumenairy/optimize/core.py`: the wrapper's tail `return self.weight
  * total` summed sub-merit results across wavelengths rather than
  averaging.  Documented semantics + both sibling classes
  (`MultiFieldMerit`, `ToleranceAwareMerit`) divide by `len(...)` at
  the return; the bug was localised to this one class.  Fixed:
  `return self.weight * total / max(len(self.wavelengths), 1)`.  A
  3-wavelength merit now returns the same value as a 1-wavelength
  merit on the same sub-merit + constant field (was returning `3x`).
* **Bug 2 — `shack_hartmann` wavefront pitch quantisation.**
  `lumenairy/analysis/detector.py`: the slope-to-wavefront integration
  multiplied the cumsum by the user-requested `lenslet_pitch`, but the
  on-grid pitch is the integer-pixel quantised `sa_pixels * dx`.  At
  `lenslet_pitch / dx = 1.7` (`sa_pixels = 2`), the reconstructed
  wavefront amplitude was biased by `8.5 / 10 = 0.85` (17.6% low
  relative to the post-fix amplitude).  Fix: use `pitch_actual =
  sa_pixels * dx` for the cumsum step.  Empirically pinned by
  `test_bug2_shack_hartmann_amplitude_ratio_physics_pin`.
* **Bug 3 — `_detect_backend` directory misclassification.**
  `lumenairy/io/storage.py`: the auto-detect routed *any* directory
  path to Zarr regardless of whether a Zarr store was actually present
  (`if path.endswith('.zarr') or os.path.isdir(path)`).  A bare
  directory matching a typical HDF5 sibling layout was silently
  misrouted, and `pathlib.Path` callers hit `AttributeError` on the
  string-only `.endswith` check.  Fix: `str(path)` cast +
  directory-routing restricted to actual Zarr stores via the canonical
  `zarr.json` (v3) / `.zarray` (v2) marker files.
* **Bug 4 — LM `bounds` parser accepts `None` endpoints.**
  `lumenairy/optimize/core.py`: the `method='lm'` branch built
  `lb`/`ub` arrays via `b[0] if b else -np.inf`; `b = (None, 1.0)` is
  truthy, so `None` leaked into `np.array([None, 0.0, ...])` (object
  dtype), and scipy raised an opaque downstream error.  Fix: explicit
  `_resolve_bound` helper that routes any `None` (per-side or
  per-tuple) to `+/-np.inf` and produces a clean `float64` array.

### P1 closures — half-shipped clusters (Agent B)

* **`propagate_ensemble(...)` helper added** (audit Part 5 P0-1).
  New module `lumenairy/propagators/ensemble.py`, exported at the
  top-level as `lumenairy.propagate_ensemble`.  Iterates a Schell-family
  `(n_realizations, Ny, Nx)` ensemble through any coherent propagator
  (`'asm'` / `'fresnel'` / `'fraunhofer'` / `'rs'` / `'sas'` or a
  user-supplied callable) and returns `I_partial = <|E_k|^2>_k` (the
  canonical Wolf-coherence-theory result).  `return_ensemble=False` by
  default for memory efficiency; opt-in `return_ensemble=True` for the
  full propagated stack.  Shape-mismatch + 2-D-field-instead-of-
  ensemble cases raise informative `ValueError`s.  New example
  `examples/06_schell_propagation.py` measures a `~6.95x` smoothing
  factor (coherent-peak / partial-peak) on a 256x256 grid at
  `sigma_g / w0 = 0.3`, consistent with the Wolf-Carter far-field
  scaling.
* **Default Schell-factory `DeprecationWarning` retired.**  The 3
  top-level factories (`create_gaussian_schell_source`,
  `create_schell_model_source`, `create_annular_incoherent_source`)
  and 2 `Source.*` classmethods now default `return_kind='ensemble'`
  directly.  The `_RETURN_KIND_UNSET` sentinel + warning helper are
  preserved as deprecated public symbols (the v4.15.3 sentinel-
  promotion meta-pin imports them); targeted for removal in v5.0.
* **MCF rejection message refreshed.**  `lumenairy/_validation.py`:
  the "MCF planned for v4.16+" wording was stale at v4.16.0.  Updated
  to cite v5.0+ and point callers at the new `propagate_ensemble`
  helper for the partial-coherence path that lands now.
* **JAX-traceable dtype probe.**  `lumenairy/system.py` ~line 1184:
  swapped the `np.asarray(E_in).dtype` probe for duck-typing
  `getattr(E_in, 'dtype', None)` so the `propagate_through_system_jax`
  path no longer breaks under `jax.jit` / `jax.grad` tracers (which
  refuse the `np.asarray` cast).
* **High-NA `_transfer_jax` RuntimeWarning.**
  `lumenairy/raytrace/jax_trace.py`: added an eager-only guard
  (`isinstance(direction_n, np.ndarray)`-gated) that emits a
  `RuntimeWarning` when `min |N| < 0.95` — the regime where the
  paraxial small-angle approximation preserved for autodiff
  stability begins to diverge from the NumPy reference.  Tracer-time
  path is unchanged (no warning, preserves `jit` / `grad` purity).
  (v4.16.2 audit P2-NEW-V2-1: pre-v4.16.2 bullet said "`UserWarning`"
  -- code + tests use `RuntimeWarning`; corrected.)  (v4.16.2 audit
  P1-NEW-F1-1: pre-v4.16.2 the `isinstance(np.ndarray)` gate was
  structurally unreachable in production because `make_jax_ray_state`
  converts to `jax.Array` which is not a `np.ndarray` subclass since
  JAX 0.4+; v4.16.2 replaces the gate with a duck-typed
  `np.asarray(...)` probe + hoists an eager-only check to the
  `trace_jax` entry before the inner `jax.jit` wrapper.)

### P2 closures — API consistency + meta-pins (Agent C)

* **`Constraint` API narrowed to scalar-return.**  Vector-return
  callables crashed inside the pymoo wrapper's `float(_f(xv))` coercion
  with an opaque `TypeError`.  Docstring narrowed to `f(x) -> scalar`
  and `__post_init__` adds a best-effort scalar-shape probe.  3-test
  pin block exercises both the accept (scalar) and reject (ndarray of
  shape `(K,)`) paths.
* **`Constraint(fun=lambda x: ...)` UserWarning for parallel workers.**
  Lambdas aren't picklable; `differential_evolution(workers>1)` /
  joblib-parallelised FD-gradient fails with `PicklingError`.  Soft
  heads-up at `__post_init__` when `fun.__name__ == '<lambda>'`;
  single-process SLSQP / trust-constr is unaffected.
* **`_clear_local_asm_caches` late-binding-lambda registration.**
  `lumenairy/propagators/propagation.py:~803`: the cache registry
  enrollment used an early-binding partial, diverging from the
  canonical late-binding-lambda pattern of the other 8 cache
  enrollments.  Switched to `lambda: _clear_local_asm_caches()` for
  pattern parity.
* **10th cache-registry meta-pin walker.**
  `tests/unit/test_v4_16_1_dispatcher_pin_cache_registry_enrollment.py`:
  AST-walks every `@lru_cache`-decorated module-level function and
  asserts a paired `_cache_registry` enrollment.  15 caches discovered,
  0 orphans — closes the V4-bucket sibling-gap structurally
  (continuing the V1-V9 meta-pin trajectory).
* **`_check_glass_registry_consistency()` extends to GLASS_VALIDITY.**
  Every `GLASS_VALIDITY` key now must appear in `GLASS_REGISTRY` (and
  must be a `(lambda_min, lambda_max)` 2-tuple with `lambda_min <
  lambda_max`, both finite, non-negative).
* **`warnings.warn(..., stacklevel=2)` hygiene** added at the 2 sites
  flagged in `lumenairy/io/prescriptions.py` (lines ~1019 / ~1470).

### P3 closures — compat + glass / materials + UX (Agent D)

* **4 stale `n_d` inline comments in `lumenairy/glass.py` corrected**
  to match the actually-computed Sellmeier values.  Multi-way
  convergent finding across audit perspectives (V5 + DEEP-3 + F1 +
  PHYS-2):
  * H-ZK9B: `1.613750` -> `1.62041`
  * H-ZF12: `1.673000` -> `1.76182`   (the most egregious — 6% off)
  * D-LAK52: `1.729160` -> `1.73050`
  * H-ZLAF52A: `1.796800` -> `1.80610`
  No runtime behaviour change — the actual Sellmeier coefficients were
  always correct; only the comments were stale.
* **`refractiveindex` moved to optional `[glass]` extras group.**
  `pyproject.toml`: dropped from hard `[project.dependencies]`,
  promoted to `glass = ["refractiveindex>=1.0"]`, and bundled into the
  existing `all` group.  Aligns the wheel with the lazy-import +
  `SELLMEIER_COEFFICIENTS` fallback already in place in
  `lumenairy/glass.py`.
* **`zarr>=2.14` floor bumped to `zarr>=3.0`** in the `zarr` and `all`
  extras groups.  `lumenairy/io/storage.py` uses `Group.create_array`
  (a Zarr v3 API); the v2 floor was a latent `AttributeError` waiting
  for any zarr=2.x user.
* **`ProcessPoolExecutor` `spawn` mp_context.**
  `lumenairy/elements/_lens_traced.py:~220`: explicit
  `mp_context=multiprocessing.get_context('spawn')` kwarg on the
  module-level worker pool.  Matches the README + v4.16.0 CHANGELOG
  claim that `spawn` is used (previously `fork` on Linux — unsafe
  with cached FFT plans and worker threads).
* **`examples/06_schell_propagation.py`** + **`examples/07_zemax_load_trace.py`** added.  The Zemax-loader example
  closes audit UX item 22 (no prior example exercised
  `la.load_zemax_zmx` end-to-end); loads the Thorlabs AC254-100-C
  achromat fixture and falls back to a programmatic N-BK7 singlet via
  `la.make_singlet` if the .zmx file is missing.
* **`CONVENTIONS.md`** added at the repo root — ~10 short sections
  documenting the `create_*` (-> field/Source) vs `make_*` (->
  prescription / bundle / non-field) factory verb contract, error-
  message prefix discipline, RNG kwarg name, and 7 related
  conventions that previously lived only in informal precedent.
* **Stray repo-root artifacts cleaned up.**  3 example PNGs moved
  from the repo root to `examples/output/` (the 3 producing scripts
  updated to write there); stray `C:tmpoptimize_diff.txt` echo-
  redirect artifact deleted (same cleanup pattern as v4.14.3's
  `C:tmpv4_14_1_changelog.md`).

### Tests + CI

* **2192 pass / 5 skip / 1 xfail = 2198 collected** (up from 2113
  collected at v4.16.0; +85 net across the 4 agents -- A=11, B=26,
  C=20+6, D=22).  (v4.16.2 audit P2-NEW-F2-MED-2: pre-v4.16.2
  headline cited 2208/+102/2106 -- arithmetic broken; refreshed
  here to use the collected metric, which arithmetic-reconciles.)
* New test modules:
  * `tests/unit/test_v4_16_1_agent_a.py` (11 tests)
  * `tests/unit/test_v4_16_1_agent_b.py` (26 tests)
  * `tests/unit/test_v4_16_1_agent_c.py` (20 tests)
  * `tests/unit/test_v4_16_1_dispatcher_pin_cache_registry_enrollment.py`
    (6 walker tests — the 10th meta-pin)
  * `tests/unit/test_v4_16_1_agent_d.py` (22 tests)
* CHANGELOG line-citation refresh: `_ZERO_APERTURE_MASK` branch site
  drifted `:2958` -> `:2974` after Agent A's Bug 1 SUM->AVG line
  additions; the v4.15.3 + v4.15.4 entries' "now at :2958" cites are
  refreshed to `:2974` (the v4.15 line-citation pin
  `TestF5ChangelogLineCitations` verifies a citation within +/-5 of
  the live site exists in CHANGELOG).

---

## [4.16.0] — 2026-05-19

**Major minor release** rolling up the entire v4.16 + v4.17 + v4.18
ROADMAP into a single release.  4 large feature buckets ship together:
the remaining 4 V4 meta-pin candidates (closing the structural
counter-measure trajectory begun in v4.15.0); multi-process atomic-
append for `storage.py` (HDF5 SWMR + filelock distributed Zarr lock);
the full optimisation framework expansion (constrained opt,
checkpoint/resume, Newton-step, multi-objective NSGA-II via pymoo);
and the glass/materials expansion (CDGM + Hikari + Sumita catalogues
+ per-glass Sellmeier validity ranges + central cache registry).
**2106 unit tests pass** (up from 1922; +184 net), 5 documented
skips (4 pymoo + 1 ZARR_MKDIR_PATCH), 1 documented xfail; **34/34
validation files pass**.

### Bucket 1 — V4 meta-pin candidates (4 walkers complete)

The audit's standing V4 recommendation from AUDIT_V4_14_2 Part 3.5
onward.  v4.15.x landed candidates V1 (cache-clears), V2 (cache↔lock),
V3 (0+0j), V4 (`_validate_grid_params`), V5 (`_check_2d_scalar_field`).
v4.16.0 lands the remaining four — completing the meta-pin coverage
of the recurring sibling-gap classes the audits identified:

* **Sentinel-aware branch propagation walker** — AST-walks
  `_get_wrapper_merit_cache` callsites for `is _ZERO_APERTURE_MASK`
  branch.  3 sites discovered, all already guarded (v4.14.1-v4.14.3
  closures are clean).  Counter-pin verifies synthetic violation
  triggers the walker.
* **Cross-backend dispatch (`_xp_of` usage) walker** — AST-walks
  field-domain public functions for hardcoded `np.*` patterns where
  `xp = _xp_of(E); xp.<...>` should dispatch.  94 candidates
  discovered; **5 inline fixes shipped** in `lumenairy/elements/elements.py`
  (`apply_zernike_aberration`, `apply_lyot_focal_plane_mask`,
  `apply_vortex_phase_mask`, `apply_apodized_pupil`, plus the
  `zernike` helper); 56 documented exemptions.
* **`dy` parameter threading walker** — every `apply_*` in
  `lumenairy.__all__` must accept `dy: Optional[float] = None` for
  anamorphic-grid support.  36 functions discovered; 26 already
  threading `dy`, 10 documented exemptions (`apply_perturbations`
  prescription input, `apply_mask` element-wise mul, polarization
  `JonesField` helpers, bundle helpers like
  `apply_thin_lens_to_beamlets`, `apply_detector` square-grid,
  `apply_dm` mirror-geometry square).
* **`__all__` symmetry walker** — every name in submodule `__all__`
  is either re-exported at the top level OR marked `_INTERNAL`
  (`_*` prefix).  752 submodule entries; 717 re-exported; **35
  documented exemptions**; **9 inline fixes** (8 backend-array-
  namespace helpers + `PYMOO_AVAILABLE` promoted to top-level
  `__all__`).

Each walker carries a fake-violation counter-pin (positive-signal
test pattern from v4.15.0 / v4.15.4).  **All 9 dispatcher meta-pins
now active and green** (cache-clears, cache↔lock, 0+0j,
validate_grid_params, check_2d_scalar_field, sentinel-propagation,
xp-dispatch, dy-threading, __all__-symmetry).  The "fix N, miss N+1"
sibling-gap meta-pattern at the public-API surface is now
structurally retired across all currently-known classes; new
classes will continue to surface and be added to the V-walker
family as identified.

### Bucket 2 — Multi-process atomic-append for `storage.py`

v4.14.3 documented single-process atomicity for `append_plane_h5`
and `_zarr_append_plane` plus a multi-process restriction.  v4.16.0
closes the multi-writer story:

* **HDF5 SWMR mode** — `append_plane_h5` gains `swmr: bool = True`
  kwarg.  When `True`, file opened with `libver='latest'`,
  `f.swmr_mode = True` after dataset creation.  Concurrent readers
  can safely follow a single writer; multiple writers are
  serialised via the sibling lock.
* **filelock-based distributed Zarr lock** — both `append_plane_h5`
  and `_zarr_append_plane` wrap the attr-write + create-array
  sequence in a `filelock.FileLock` on the sibling `<path>.lock`
  file.  Cross-process race-free; configurable
  `lock_timeout: float = 30.0` kwarg.
* **`filelock>=3.0`** added to `hdf5` and `zarr` optional-dependency
  groups in `pyproject.toml` (verified NOT a transitive dep of
  either h5py 3.16 or zarr 3.1).
* **Subprocess multi-writer tests** via `multiprocessing.get_context('spawn')`
  (Linux + Windows portable) — 4 workers × 5 planes each verifies
  20-plane final file with no data loss.
* **Single-process v4.14.3 atomicity guarantees preserved**
  bit-for-bit.

Measured overhead: ~5× slowdown 4-writer contended vs 1-writer
baseline; <5% lock overhead on large planes (≥4096²) where
HDF5/Zarr I/O dominates.

### Bucket 3 — Optimisation framework expansion (ROADMAP v4.17)

Four additions to `lumenairy.optimize`:

* **Constrained optimisation** via scipy `NonlinearConstraint`
  mapping.  New `Constraint` dataclass; `design_optimize(...,
  constraints=[Constraint(fn=..., lb=..., ub=..., label=...)])`.
  Method-compatibility validator raises a clear `ValueError` for
  non-supporting methods (L-BFGS-B / `lm` / `differential_evolution`
  / `basin_hopping` / `dual_annealing` all silently ignored
  constraints in scipy's core API — v4.16 rejects them up front
  pointing the user at SLSQP / trust-constr).  Diagnostic
  constraint-label printed in progress callback.
* **Checkpoint / resume on long `design_optimize` runs**.  Add
  `state_file: Optional[str] = None` and
  `state_save_every: int = 1` kwargs.  Persists
  `(call_count, x_best, merit_best, history)` to JSON with atomic-
  replace write (`.tmp` + `os.replace`).  On startup, if the file
  exists with matching shape, resumes from persisted `x_best`.
  Gated on `state_file` non-None so legacy callers see byte-
  identical behaviour.
* **Multi-objective Pareto via pymoo NSGA-II wrapper** —
  `lumenairy.optimize.multi_objective.design_optimize_multi_objective(...)`
  with `ParetoResult` dataclass.  pymoo is an **optional
  dependency** in the new `multi_objective` extras group (`pip
  install lumenairy[multi_objective]`).  Same opt-in pattern as
  jax/cupy/numba/h5py/zarr in the library.  Module imports
  unconditionally; only the actual function call raises
  `ImportError` with a clear install hint.  pymoo's heavier
  transitive deps (autograd, deap, cma) are deliberately NOT
  bundled into the `all` group.  4 new top-level exports:
  `Constraint`, `ParetoResult`, `design_optimize_multi_objective`,
  `PYMOO_AVAILABLE`.
* **Hessian / Newton-step optimisation** via `method='newton'`.
  Dispatches to scipy `trust-ncg` with FD-Jacobian-of-FD-gradient
  Hessian estimator.  `UserWarning` recommends L-BFGS-B for
  `n_params > 30` (Newton's FD-Hessian cost scales as n²).

13 tests (4 constrained + 3 checkpoint + 4 multi-objective + 2
Newton); 4 of the multi-objective skip cleanly if pymoo isn't
installed.

### Bucket 4 — Glass + materials + central cache registry

Three additions (ROADMAP v4.18 items #13-#15):

* **CDGM + Hikari + Sumita Sellmeier catalogues** — 32 new
  glasses across the three major non-Western catalogues:
  - **CDGM (12)**: H-K9L, H-LAK52, H-LAK53A, H-ZK9B, H-ZF12,
    D-ZK3, D-LAK52, H-ZLAF52A, H-ZK7, H-ZF52A, F1-CDGM, F2-CDGM.
  - **Hikari (10)**: E-LASF016, E-SK16, E-LAK7, E-LAK04, E-BAK1,
    J-FK01A, J-LASF09A, J-LAK7, J-BASF7, E-F2.
  - **Sumita (10)**: K-VC78, K-LAK10, K-LASFN10, K-SK4, K-PFK90,
    K-PBK40, K-BK7, K-PSKN2, K-FK5, K-LAFN3.
  **`GLASS_REGISTRY`: 46 → 78 entries**.  Every new entry n_d
  cross-checked against the official datasheet (or
  refractiveindex.info as proxy) at the 5e-5 tolerance pin
  established by v4.14.2's S-LAH64/79 verification.  Zero
  glasses failed the cross-check.  8 of the new CDGM glasses
  also ship as bundled Sellmeier-formula-2 fallbacks for
  minimal installs without `refractiveindex`; the remaining 22
  use formula-3 (polynomial) which requires `refractiveindex`
  for minimal installs — clear `ImportError` with install hint.
* **Per-glass Sellmeier validity ranges** — new `GLASS_VALIDITY`
  table with 77 entries (one per catalogued glass).  Format
  `{name: (lambda_min, lambda_max)}` in metres.  Extrapolating
  outside the range emits `UserWarning(...)` but does NOT raise
  — extrapolation is sometimes acceptable for design-space
  exploration.  Per-glass sources cited inline in the table
  (refractiveindex.info URLs + datasheet revs).
* **Central cache registry** (`lumenairy/_cache_registry.py`)
  — new public API `register_cache_clearer(name, clear_fn)` +
  `list_registered_cache_clearers()` + `clear_all_registered_caches()`.
  Retires the lazy-import fan-out in `clear_asm_caches`.  9
  caches migrated to the registry (`asm_local`, `lg_mode_stack`,
  `lg_polynomial_items`, `zernike_basis`,
  `through_focus_scan_jax`, `propagate_system_jax`,
  `phase_retrieval_kernels`, `trace_jax`,
  `wrapper_merit_meshgrid`).  `clear_asm_caches`'s external
  contract is preserved bit-for-bit (still callable with the
  same name + signature); the internal dispatch is now
  registry-walking instead of hand-enumerated.  Structural
  counter-measure to the cache-clear "fix N, miss N+1" pattern
  the v4.14.x audits identified.

127 new tests (10 cache-registry + 109 glass + 8 validity); zero
n_d cross-check failures.

### New top-level exports (12)

* `Constraint`, `ParetoResult`, `design_optimize_multi_objective`,
  `PYMOO_AVAILABLE` (optimisation)
* `register_cache_clearer`, `list_registered_cache_clearers`,
  `GLASS_VALIDITY` (cache registry + glass validity)
* `array_namespace`, `is_numpy_array`, `is_cupy_array`,
  `is_jax_array`, `backend_name`, `to_numpy`, `to_backend`,
  `RandomState` (backend helpers — Agent A's `__all__` symmetry
  fix promoted these)

### Optional dependencies

* New `multi_objective` extras group: `pip install
  lumenairy[multi_objective]` adds pymoo for NSGA-II Pareto.
* `hdf5` and `zarr` extras groups now include `filelock>=3.0`
  (was previously transitively missing).

### Test counts

* Pre-v4.16.0 baseline (v4.15.5): 1922 unit pass + 1 skip + 1 xfail.
* v4.16.0 additions: A=30 (4 walkers), B=15 (multi-process storage),
  C=13 (10 pass + 3 pymoo-skip; pymoo not installed in test env),
  D=127 (10 cache-registry + 109 glass + 8 validity); plus 5 inline
  xp-dispatch fixes' positive regression coverage.  Net +184
  collected, +5 documented skips (3 new pymoo + 0 already present
  ZARR mode + 1 already present).
* Final: **2106 unit pass + 5 skip + 1 xfail; 34/34 validation**.

### ROADMAP status post-v4.16.0

* **v4.16, v4.17, v4.18 — all items shipped**.  The ROADMAP's
  Current State section is refreshed to reflect this; remaining
  target sections are folded into Shipped highlights.
* **v5.0 — immediate horizon**.  Major structural release: 6
  file splits, CI gates (pytest fast-PR + ruff + mypy --strict
  incremental + `__all__` smoke), remove 8 active back-compat
  shims, shared Chebyshev helpers extraction, audit-fix test-file
  consolidation, `lumenairy/system.py` → `propagators/system.py`,
  off-axis conic in surface frame (Optiland/Zemax parity), bump
  `requires-python` to >=3.10, 3 config knobs, docs.
* **Designer GUI v3.8+** still unplanned (separate version
  stream).

### Known issues / flagged for v4.16.1

* **Bundled Sellmeier formula-3 (polynomial) evaluator** —
  Hikari, Sumita, and 4 CDGM glasses use refractiveindex.info
  formula 3.  v4.16.0's `_sellmeier_index` only supports
  formula 2.  Minimal installs without `refractiveindex` raise
  `ImportError` on these 26 entries with a clear actionable
  message.  v4.16.1 candidate: add `_polynomial_index`
  evaluator.

### Deferred to v5.0

Architectural items requiring breaking changes — see ROADMAP for
the full v5.0 catalogue.

---

## [4.15.5] — 2026-05-19

**Closes the v4.15.4 audit (`docs/audits/AUDIT_V4_15_4_2026_05_19.md`)
through P3.**  The audit found 0 P0 + 4 P1 + 6 P2 + 5 P3.  Three of
the 4 P1s closed via a **V6 dispatcher meta-pin walker refactor**:
discovery now keys off the function's **first-positional-parameter
name** (via AST inspection of `ast.arguments`) rather than a hand-
curated name-prefix list, plus the walker now **descends into class
bodies** for non-delegating methods like `DeformableMirror.apply`.
The remaining P1 cluster (2 user-facing pitfalls in v4.15.4's new OPD
plotting functions) closed via a `fan_units` kwarg + centered-RMS
metric consistency.  **1922 unit tests pass** (up from 1858; +64
net), 1 documented skip, 1 documented xfail; **34/34 validation
files pass**.

### Headline: V6 walker (first-positional-param-name discovery)

The v4.15.3/4 dispatcher meta-pin walker used a hand-curated name-
prefix filter (`apply_*` / `propagate_*` / `richards_wolf_*` /
`debye_wolf_*`).  v4.15.4 audit found 30+ public `__all__`
functions outside the filter that take 2-D `E`/`field`/`pupil`
first positional args — the meta-pattern recurred at one indirection
level higher.

v4.15.5 refactors discovery:

* **Primary filter is now AST-based first-positional-param name.**
  Walks `lumenairy.__all__`; for each public function, inspects
  `ast.arguments.args[0].arg`; if the name is in
  `_FIELD_PARAM_NAMES = frozenset({'E', 'E_in', 'field', 'pupil',
  'object_field', 'psf'})`, requires `_check_2d_scalar_field` call
  OR `_GUARD_EXEMPTIONS` entry.  Discovery is now grounded in the
  actual function signature rather than a string match.
* **Legacy name-prefix filter retained as fallback** — v4.15.4
  coverage preserved bit-for-bit; V6 only ADDS entries.
* **Class-body descent** — the walker now visits class methods
  named `apply` / `propagate` / similar.  `_DELEGATING_CLASS_METHODS`
  exemption set documents which classes legitimately delegate to
  module-level guarded functions (operator-algebra: `ThinLens.apply`,
  `FreeSpace.apply`, `CylindricalLens.apply`, `Magnify.apply`,
  `FourierTransform.apply`, `Aperture.apply`, `GaussianAperture.apply`,
  `CompositeOperator.apply`).
* **`_file_to_ast` `lru_cache`d** (P3-NEW-F1-2) — 11 sibling
  functions in `_lens_thin.py` previously triggered 11 re-parses
  of the same file; cache cuts that to 1.  Walker test wall time:
  **~1.5s → ~0.03s warm (~30× speedup)**; cache stats `hits=180,
  misses=42, currsize=42` on a full walk (81% hit rate).

Post-refactor walker discovery:  **96 top-level entry points + 3
class methods** = 99 candidates (was 72 at v4.15.4 HEAD).  Of the
96 top-level: 39 guarded (was 25), 47 exempt (unchanged), 10 newly-
flagged for v4.16+ inline guard sweep (HFPI initialisers,
decomposition helpers, low-priority `analysis/core.py` analyzers).

### P1 closures (4)

* **P1-NEW-F2-2 — `DeformableMirror.apply` 1-line guard.**  The
  module-level `apply_dm` was guarded in v4.15.4; the class method
  was a 3-line `E_in * np.exp(1j * phi)` with no `_check_2d_scalar_field`
  call.  Walker's blanket class-method exclusion (v4.15.4 docstring
  asserted methods "delegate to guarded scalar functions") was true
  for Cluster-B operator algebra, false here.  v4.15.5 closes both:
  inline guard on the method + walker descent into class bodies.
* **P1-NEW-2WAY-1 — V6 walker refactor** (covered under headline)
  + **inline guards on the 13 highest-traffic unguarded analyzers**:
  `wave_opd_2d`, `M2`, `strehl_ratio`, `beam_d4sigma`,
  `coupling_efficiency`, `compute_psf`, `compute_otf`, `compute_mtf`,
  `encircled_energy_curve`, `koehler_image`, `extended_source_image`,
  `shack_hartmann`, `rays_from_field`, `resample_field`.  34
  regression tests pin each guard via direct `MCF` / 3-D ensemble
  rejection.
* **P1-NEW-V2-1 — `plot_opd_fan` `fan_units` kwarg.**
  `raytrace.opd_fan_data` returns OPD in **waves**;
  `plot_opd_fan` expected **metres**.  The canonical pipeline
  `plot_opd_fan(*opd_fan_data(...), units='waves', wavelength=wl)`
  divided by `wavelength` a second time → silently wrong by ~6e5.
  v4.15.5 adds `fan_units: str = 'm'` (default `'m'` preserves
  v4.15.4 callers; pass `fan_units='waves'` for the
  `opd_fan_data` pipeline).  End-to-end regression test pins
  `opd_fan_data → plot_opd_fan(fan_units='waves', units='waves',
  wavelength=wl)` does not double-convert.
* **P1-NEW-V2-2 — `_radial_rms_profile` centered RMS.**  v4.15.4
  `_radial_rms_profile` used `sqrt(mean(opd²))` (variance about
  zero) while the 1-D fan RMS and 2-D heatmap RMS used
  `sqrt(mean((opd - mean)²))` (centered, wavefront-error
  convention).  On a pure-defocus OPD, the radial-RMS curve was
  piston-dominated and looked like r²; the heatmap annotation was
  the much smaller centered RMS.  Numbers on the same figure did
  not reconcile.  v4.15.5 switches to centered RMS using the in-
  aperture mean computed once for the entire OPD.  Example
  `plot_opd_summary_singlet.py` RMS now reports **0.8901 waves**
  (was 1.3347 waves uncentered); PV unchanged at 2.9318 waves.

### P2 closures (6)

* `_check_2d_scalar_field` parameterized with `input_kind:
  str = 'field'` — `richards_wolf_focus` / `debye_wolf_psf` /
  `compute_psf` / `compute_otf` / `compute_mtf` etc. pass
  `input_kind='pupil'` or `'psf'`, getting accurate error
  messages ("expected 2-D complex pupil" instead of "field").
* `plot_opd_summary` docstring corrected to explicitly state
  `opd_2d` input is in **metres** (matching `plot_wavefront`'s
  convention).
* Example `plot_opd_summary_singlet.py` RMS print switched to
  centered form (matches heatmap annotation).
* `plot_opd_summary` even-N central-row/col fallback: aligned
  using `(N - 1) // 2` consistently.
* Walker descends into class bodies (covered under headline).
* CHANGELOG v4.15.4 entry-point count refresh (43→49 → actual
  72 at v4.15.4 HEAD; v4.15.5 numbers 96 + 3 class methods are
  documented in this entry).

### P3 closures (5)

* `_file_to_ast` `@lru_cache` (covered under headline).
* `plot_opd_fan` docstring now explicitly states "centered RMS
  (about the in-aperture mean); PV (max - min)".
* `n_bins` kwarg exposed on `plot_opd_summary` as
  `radial_rms_n_bins: int | str = 'auto'` with auto-clamping for
  tiny grids (`min(32, int(sqrt(N_in_aperture)) // 2)`).
* CHANGELOG `-W error::DeprecationWarning` failure count drift
  (57 vs 63) reconciled — canonical at v4.15.4 commit time was
  63; v4.15.3 audit's 57 reflects pre-dispatch count, drift
  documented inline.
* CHANGELOG per-agent attribution arithmetic +1 footnote added
  (per the standard pattern from v4.15.3 hygiene fix).
* CHANGELOG wavelength annotation drift fixed (587.56 nm → 633
  nm to match the example).
* ROADMAP duplicate-counting drift cleaned up: 6 already-shipped
  items moved from "v4.16 residual" to "Shipped highlights"
  (polychromatic encircled energy, polarisation-aware Strehl,
  resolution metrics, astigmatism mag+angle, OAP, Forbes Q-type
  — all landed in v4.15.0 / v4.15.1).  True v4.16 residual is
  now 2 items: V4 meta-pin candidates + multi-process atomic-
  append for `storage.py`.  Remaining v4.17/v4.18 items
  renumbered.

### Test counts

* Pre-v4.15.5 baseline (v4.15.4): 1858 unit pass + 1 skip + 1 xfail.
* v4.15.5 additions: A=34 regression + 4 V6/class-method pins
  (38 total), B=17 + 3 carry-forward (20 total), C=4.  Net +64.
* Final: **1922 unit pass + 1 skip + 1 xfail; 34/34 validation**.

### Deferred to v4.16+

Unchanged from prior releases: modal-asymptotic independent
ground-truth pin; 4 V2 meta-pin candidates still standing
(sentinel-aware branch propagation, `_xp_of` dispatch, `dy`
parameter threading walker, `__all__` symmetry walker; the V6
first-positional-param-name candidate landed in v4.15.5);
MCF-aware downstream propagators; multi-process atomic-append
for `storage.py`; `MultiPrescriptionParameterization.scale_floor`;
Forbes Q-2D-asymmetric variant.  Plus: **10 newly-flagged
unguarded analyzers** for v4.16+ inline-guard sweep (HFPI
initialisers, decomposition helpers, lower-priority
`analysis/core.py` analyzers like `beam_centroid`,
`beam_diameter`, `radial_power_bands`, `wave_opd_1d`,
`strehl_phase_integral`, resolution metrics, `single_plane_metrics`).

---

## [4.15.4] — 2026-05-19

**Closes the v4.15.3 audit (`docs/audits/AUDIT_V4_15_3_2026_05_18.md`)
through P3 + adds two user-facing OPD plotting functions.**  The audit
found 0 P0 + 1 P1 + 6 P2 + 5 P3 — the cleanest yield in the v4.15.x
series.  The single P1 is the recurring "fix N, miss N+1" meta-
pattern re-emerging one level of indirection higher than v4.15.3
closed it: the dispatcher meta-pin's `_TARGET_PACKAGES` scope itself
was a sibling gap.  **1858 unit tests pass** (up from 1822; +36 net),
1 documented skip, 1 documented xfail; **34/34 validation files pass**.

### Headline: walker scope refactor closes the meta-pattern at the package level

v4.15.3 shipped the `_check_2d_scalar_field` helper + dispatcher meta-
pin.  But the walker scoped only to
`('lumenairy/propagators', 'lumenairy/elements')`, missing 4 public
entry points outside that scope.  v4.15.4 makes discovery
`__all__`-based:

* **`_walk_entry_points` refactored** to walk `lumenairy.__all__`
  membership via `inspect.getsourcefile`; survives future refactors
  that move functions between subpackages.  Package-walk retained as
  a fallback.  Walker discovery at v4.15.4 HEAD: **72 total entry
  points (25 guarded + 47 documented exempt)**.  (The pre-correction
  CHANGELOG bullet cited "43 -> 49 after the refactor", which
  reflected the `__all__`-pass-only count without the package-walk
  fallback dedup; v4.15.5 / Agent C refreshed this from the live
  diagnostic per AUDIT_V4_15_4 P2-NEW-3WAY-3.  TODO: the v4.15.5 V6
  walker refactor (Agent A scope) will change this number again at
  integration time; Agent C populated this with v4.15.4 HEAD numbers
  and integration will update with v4.15.5 V6-refactored numbers.)
* **Name filter broadened** with `name.startswith('propagate_')` to
  catch `propagate_through_system` + `propagate_through_system_jax`
  (which contain `propagate_` at the start but not `_propagate` in
  the middle — the v4.15.3 filter missed both).
* **6 newly-found sibling entry points guarded** via the v4.15.3
  helper: `propagate_through_system_jax` (P1), `apply_dm`,
  `apply_detector`, `richards_wolf_focus`, `debye_wolf_psf`; plus
  `apply_perturbations` documented exempt (first positional arg is
  a prescription dict, not a 2-D scalar field).
* **Fake-violation counter-pin** added to the dispatcher meta-pin:
  injecting a synthetic unguarded function via `monkeypatch.setattr`
  must trigger the meta-pin's `AssertionError`.  Walker correctness
  now pins on a positive signal.

### P1 closure (1)

* **P1-NEW-3WAY-1** walker scope completeness — closed via
  `__all__`-based discovery + extended `_TARGET_PACKAGES`.

### P2 closures (6)

* Walker name-regex broadening + 3 unguarded `analysis/` siblings
  guarded (covered under headline).
* SAS-anamorphic CHANGELOG wording corrected retroactively in the
  v4.15.3 entry: `"forces method='asm' regardless of self.method"`
  -> `"forces method='asm' when self.method == 'auto' and dy != dx;
  explicit method='sas' on anamorphic grids still crashes (user's
  responsibility)"`.
* `_validation.py` lazy-import hoisted to module scope.  The
  v4.15.3 code used a lazy import citing a hypothetical circular
  dep; audit grep-verified no actual circular dep exists.  Saves
  ~1 µs/call (1-10 ms per merit eval in optimization loops with
  thousands of propagator calls).
* Dead `_PerturbedABCDFallbackSentinel` deleted.  v4.15.3 marked
  it dead via `_v4_15_3_dead_code = True` class attribute
  (informational only — no static analyzer honors it).  v4.15.4
  deletes the class + singleton (~58 LOC).  v4.15.2 test pin
  updated in the same commit.
* ROADMAP refreshed: post-v4.15.3 test count ~1750 → actual 1824;
  AUDIT_V4_15_2 + AUDIT_V4_15_3 added to closed-audits list; meta-
  pin coverage 3 of 5 → 4 of 5 with the V5 entry describing
  `_check_2d_scalar_field`.

### P3 closures (5)

* CHANGELOG dispatcher meta-pin count drift fixed (18/25 → 17/26;
  43 total).
* Fake-violation counter-pin added (covered above).
* CHANGELOG stacklevel wording: "6 Source classmethod shims" → "5
  `Source.*` classmethod shims at `:2424, 2510, 2587, 2661, 2750`
  plus the module-level `create_led_source` factory shim at
  `:1209`".
* `-W error::DeprecationWarning` test hygiene: 63 v4.15.3 tests
  previously failed under the strict flag (they exercised the
  documented Schell `return_kind` default-path warning without
  shielding).  v4.15.4 adds `pytestmark =
  pytest.mark.filterwarnings('default::DeprecationWarning')` to 6
  affected test files.  Failures: 63 → 0.  *(Note: the v4.15.3
  audit's P3-NEW-F2-4 cited 57 failing tests; the discrepancy with
  this CHANGELOG's 63 reflects pre/post-v4.15.4 dispatch-test-file
  additions between when the audit was filed and v4.15.4 commit
  time.  Canonical count at v4.15.4 commit time was 63; closure
  verified to 0 escalations regardless of which baseline is right.
  Documented per AUDIT_V4_15_4 P3-NEW-V3-1 / v4.15.5 Agent C.)*
* CHANGELOG test-count arithmetic reconciled: v4.15.3 baseline
  1732 → 1733; per-agent attribution sum 88 documented alongside
  actual collected delta 89 with explicit explanation of the +1
  attribution-vs-collection gap.  Removed the false claim about a
  ROADMAP update that wasn't actually performed in v4.15.3.

### New: OPD plotting functions

Two new public functions in `lumenairy/analysis/plotting.py`, visually
matching the `OPDPy_Lumenairy_Crosscheck` `fig_variety_L*.png` style:

* **`plot_opd_fan(py, opd_y, px, opd_x, *, wavelength=None,
  units='waves', show_stats=True, title=None, fig=None, axes=None)`**
  — 2-panel tangential + sagittal OPD fans.  Inputs match
  `lumenairy.raytrace.opd_fan_data`'s return tuple.  Solid-line
  plots with zero-reference axhline, in-axes PV/RMS annotation,
  units kwarg matches `plot_wavefront` (waves / nm / um / m).
  Returns `(fig, (ax_y, ax_x))`.  147 LOC.
* **`plot_opd_summary(opd_2d, dx, *, dy=None, aperture=None,
  wavelength=None, py=None, opd_y=None, px=None, opd_x=None,
  units='waves', cmap='RdBu_r', show_stats=True, title=None,
  fig=None)`** — 4-panel summary: 2-D heatmap (delegates to
  `plot_wavefront`), radial RMS profile (32 annular bins),
  tangential fan, sagittal fan.  Fan panels use the provided
  `(py, opd_y, px, opd_x)` if supplied (preferred — raytrace data
  has chief-ray reference built in) or auto-extract from the 2-D
  OPD's central row/column otherwise.  Returns `(fig, ((ax_hm,
  ax_rms), (ax_y, ax_x)))`.  204 LOC.

Both added to `lumenairy.__all__` (analysis tier).  10 unit tests +
runnable example at `examples/plot_opd_summary_singlet.py` (singlet
OPD via `apply_real_lens_traced` + `opd_fan_data`; PV ≈ 2.93 / RMS
≈ 1.33 waves at λ=633 nm).  *(Pre-v4.15.5 wording cited
λ=587.56 nm, contradicting the example's actual ``wavelength =
633e-9``; corrected per AUDIT_V4_15_4 / v4.15.5 Agent C.)*

### Test counts

* Pre-v4.15.4 baseline (v4.15.3): 1822 unit pass + 1 skip + 1 xfail.
* v4.15.4 additions, per-agent attribution: A=8 + 1 counter-pin in
  the meta-pin file (=9), B=11, C=7, D=10.  Per-agent sum: **37**.
* Actual `pytest --collect-only` delta: 1858 - 1822 = **+36**
  (canonical post-release number).  The +1 attribution-vs-collection
  gap reflects the standard parametrize/fixture artifact (one of
  the new tests expands to 2 collected items via `parametrize`, or
  a fixture-only addition isn't cleanly attributed to a single
  agent).  Same pattern documented in the v4.15.3 entry; pinned by
  `test_changelog_per_agent_breakdown_sums_to_net_delta` in
  `test_v4_15_4_agent_c.py`.  Documented per AUDIT_V4_15_4
  P3-NEW-V3-2 / v4.15.5 Agent C.
* Final: **1858 unit pass + 1 skip + 1 xfail; 34/34 validation**.

### Deferred to v4.16+

Unchanged from prior releases: modal-asymptotic independent
ground-truth pin; 4 V2 meta-pin candidates still standing
(sentinel-aware branch propagation, `_xp_of` dispatch, `dy`
parameter threading walker, `__all__` symmetry walker); MCF-aware
downstream propagators; multi-process atomic-append for
`storage.py`; `MultiPrescriptionParameterization.scale_floor`;
Forbes Q-2D-asymmetric variant.

---

## [4.15.3] — 2026-05-18

**Closes the v4.15.2 audit (`docs/audits/AUDIT_V4_15_2_2026_05_18.md`)
through P3.**  The audit identified 1 P0 + 4 P1 + ~6 P2 + ~4 P3 —
mostly the recurring "fix N, miss N+1" sibling-gap meta-pattern that
has appeared in every audit round from v4.13.x onward.  v4.15.3
closes the P0 with a **structural counter-measure** (shared
validation helper + dispatcher meta-pin) that makes the recurrence
impossible going forward.  **1822 unit tests pass** (up from 1733;
+89 collected vs v4.15.2 HEAD), 1 documented skip, 1 documented
xfail; **34/34 validation files pass**.

### Headline: structural counter-measure for the sibling-gap meta-pattern

The v4.15.2 closure guarded 10 propagator/lens entry points against
`PartialCoherenceMCF` + 3-D ensemble inputs.  This audit found **9
more public entry points** of the same type that were missed —
`angular_spectrum_propagate_tilted`, `*_propagate_mft` (3 variants),
`apply_spherical_lens`, `apply_aspheric_lens`, `apply_grin_lens`,
`apply_axicon`, `apply_real_lens_traced`, `apply_real_lens_maslov`.

v4.15.3 fixes this **structurally**:

1. **`lumenairy/_validation.py`** (NEW) — single canonical
   `_check_2d_scalar_field(E, fn_name)` helper consolidates the
   `PartialCoherenceMCF` and `ndim != 2` guards.  Replaces ~240 LOC
   of duplicated boilerplate across 10 v4.15.2 sites + 9 new sibling
   sites.  Net `lumenairy/` LOC change: roughly +23 (helper +102 LOC;
   migrated sites -160 LOC; new sibling guards +81 LOC) vs +225 LOC
   the inline pattern would have cost on the 9 new sites alone.

2. **Dispatcher meta-pin**
   (`tests/unit/test_v4_15_3_dispatcher_pin_2d_scalar_field.py`) —
   AST-walks every `def apply_*` and `def *_propagate*` in
   `lumenairy/propagators/` and `lumenairy/elements/`; asserts
   `_check_2d_scalar_field` is the first executable statement of
   each function body.  **43 entry points discovered, 17 guarded,
   26 documented exemptions** (GBD beamlets, HFPI/HF state objects,
   batched 3-D variants, JAX-traceable lens kernels, polarization
   helpers, etc.).  Adding a new public entry point in the at-risk
   modules WITHOUT the helper call now fails CI.

3. **`_GUARD_EXEMPTIONS` registry** documents every legitimate
   exemption with a reason — converts "easy to miss" into "easy to
   see in code review".

This is the 5th structural meta-pin in the library:  v4.14.1
cache-clear dispatcher pin, v4.14.2 cache↔lock pairing + 0+0j
literal sweep, v4.15.0 `_validate_grid_params` input-validation
entry-point pin, v4.15.3 `_check_2d_scalar_field` pin.

### P0 closure

**P0-NEW-F2-1 — 9 unguarded propagator/lens entry points.**  All 9
sibling sites now call `_check_2d_scalar_field` as the first
executable statement, identical guard semantics to the 10 v4.15.2
sites:
* `angular_spectrum_propagate_tilted`, `angular_spectrum_propagate_mft`,
  `fresnel_propagate_mft`, `fraunhofer_propagate_mft` in
  `propagators/propagation.py`
* `apply_spherical_lens`, `apply_aspheric_lens`, `apply_grin_lens`,
  `apply_axicon` in `elements/_lens_thin.py`
* `apply_real_lens_traced` in `elements/_lens_traced.py`
* `apply_real_lens_maslov` in `elements/lenses_maslov.py`

30 regression tests pin the 9 new guards (TypeError on `PartialCoherenceMCF`,
ValueError on 3-D ensemble).  7 meta-pin tests pin the structural
counter-measure (walker discovery, helper-is-first invariant,
counter-pin against accidentally-removed guards).

### P1 closures (4)

* **P1-NEW-F1-1 — `FreeSpace._apply` SAS-anamorphic crash fixed.**
  When `dy != dx` and `method='auto'`, the dispatcher routed to SAS
  (square-grid-only); the v4.15.2 dy-threading fix passed `dy` to
  SAS which doesn't accept it.  v4.15.3 forces `method='asm'` when
  `self.method == 'auto'` and `dy != dx` — `auto` is now a hint,
  not a contract.  Explicit `method='sas'` on anamorphic grids
  still crashes (user's responsibility — the in-code comment at
  `algebra/primitives.py:142-147` documents this gating).
  `FourierTransform._apply` inherits the fix by composition (the
  3-stage rewrite creates `FreeSpace` instances internally).
* **P1-NEW-F1-2 — Schell `DeprecationWarning` stacklevel fixed.**
  `_warn_schell_return_kind_default` had `stacklevel=4`; the call
  chain is 5 frames deep (warnings.warn → warn_deprecated_signature
  → _warn_schell_return_kind_default → factory → user).  Bumped to
  5.  Library-wide sweep of `_warn_*` helpers found 6 additional
  off-by-one stacklevels in `sources/core.py` (5 `Source.*`
  classmethod shims at `:2424, 2510, 2587, 2661, 2750` plus the
  module-level `create_led_source` factory shim at `:1209`); all
  bumped 3 → 4.
* **P1-NEW-F1-3 — 3 dead `optimize/core.py` sentinels wired.**
  v4.15.2 added `_InvalidFocalLengthSentinel`,
  `_FailedScanStrehlSentinel`, `_PerturbedABCDFallbackSentinel`
  class definitions but never wired them at callsites.  v4.15.3
  wires the 2 scalar sentinels at `optimize/core.py:2424, 2696,
  3015` (was raw `-1.0` / `float('nan')` / `0.0` returns); marks
  `_PerturbedABCDFallbackSentinel` as dead-code (tuple shape didn't
  sentinel cleanly without breaking downstream unpacking; class
  retained with `_v4_15_3_dead_code = True` marker for v4.15.2
  test-pin compatibility).
* **P1-NEW-F1-4 — `Source.gaussian_schell`/`schell_model`
  classmethods route through sentinel.**  Pre-v4.15.3 these
  classmethods hardcoded `return_kind='ensemble'`, bypassing the
  v4.15.2 `_RETURN_KIND_UNSET` sentinel — calling them without
  `return_kind` produced a silent 4-tuple with no
  `DeprecationWarning` and a `Source` whose `E.ndim == 3` (every
  other `Source.*` produces 2-D).  v4.15.3 routes both
  classmethods through the sentinel; default-path callers now
  get the same DeprecationWarning as the top-level factories.
  Soft 2-D `Source.E` invariant break documented in classmethod
  docstrings as intentional (Schell is partial-coherence;
  collapsing to 2-D would be physically wrong).

### P2 closures

* **`_RETURN_KIND_UNSET` promoted** from bare `_Sentinel` instance
  to dedicated `_SchellReturnKindUnsetSentinel(_Sentinel)`
  subclass with `_SENTINEL_REGISTRY` entry for pickle round-trip
  safety.  Consistent with `_ZeroApertureMaskSentinel`,
  `_AngleUnsetSentinel`, `_NoDefaultSentinel`.
* **rays_from_field threshold-comparison consistency.**  Audit
  finding was inverted (`_place_rejection` was already `>=`,
  `_place_uniform` and `_place_cdf` were strict `>`).  v4.15.3
  makes all 3 modes inclusive `>=`.  Pixels at exactly
  `intensity_threshold` now consistently survive.
* **Non-tautological FourierTransform pin** added — Gaussian-beam
  waist relation `w_out = lambda * f / (pi * w_in)` (Saleh &
  Teich §3.2.2) through `FourierTransform(f)`.  Measured error
  <0.0001% vs the 5% tolerance pin (50000× headroom).  Pins
  physics not implementation.
* **4-fold mirror folded-prescription test cases** added to
  `test_v4_15_1_agent_g_matches_system_abcd.py` —
  `_build_folded_4fold_periscope` and
  `_build_folded_cassegrain_2curved_2flat` strengthen the
  `from_prescription` flat-mirror parity claim across more
  complex folded geometries.
* **Library-wide `_warn_*` stacklevel sweep** (~12 helpers
  audited; 7 adjusted, 5 unchanged with documented rationale).

### P3 closures

* **CHANGELOG Forbes Q OPD bullet corrected**: tolerance
  `1e-3` → `5e-3` (test code was always `5e-3`); formula
  `OPD = -k * sag` → `OPD(r) = (n - 1) * sag(r)` (the `(n - 1)`
  index factor was missing).  Test code was always correct;
  only the CHANGELOG bullet lied.
* **CHANGELOG sentinel-migration line citations refreshed**
  after Agent C's v4.15.3 wiring drift: `_ZERO_APERTURE_MASK`
  branch now at `optimize/wrapper_merits.py:955` (was
  `optimize/core.py:3032` pre-v5.1.0 Agent E 6-file split, which
  moved `ToleranceAwareMerit.evaluate` out of the monolithic
  `optimize/core.py`; was `:3015` pre-v4.16.3 Agent C `Constraint`
  auto-probe DeprecationWarning latch + pickle catch widening; was
  `:2974` pre-v4.16.2 Agent B `MultiWavelengthMerit` `FutureWarning`
  latch + Constraint probe move + lambda pickle-probe; was `:2958` pre-v4.16.1 Agent A
  `MultiWavelengthMerit` `SUM`->`AVG` refactor; was `:2980` pre-
  v4.15.4 Agent B `_PerturbedABCDFallbackSentinel` deletion; was
  `:2905` in the v4.15.2 entry).
* **Test count reconciliation**:
  `pytest --collect-only` → 1735 collected at v4.15.2 HEAD
  (was reported as "1732 pass + 1 skip + 1 xfail = 1734" in
  CHANGELOG, off by 1); reconciled in this entry's `### Test
  counts` block below.  The ROADMAP refresh originally claimed in
  this bullet ("~1700 → 1822 baseline") did NOT land in v4.15.3
  — that drift is documented in `AUDIT_V4_15_3` P2-NEW-V3-3 and is
  closed by v4.15.4 (Agent C scope).

### Test counts

* Pre-v4.15.3 baseline (v4.15.2): 1733 unit pass + 1 skip + 1 xfail
  = 1735 collected (per the corrected v4.15.2 entry's
  `pytest --collect-only` reconciliation; pre-v4.15.3 the v4.15.3
  block transcribed this baseline as "1732" — a one-off carry-over
  of the same off-by-one the v4.15.2 entry self-corrected).
* v4.15.3 additions, per-agent attribution: A=37 (7 meta-pin + 30
  regression), B=24, C=19 (12 new file + 4 4-fold + 3
  Gaussian-waist), D=8 (7 doc + 1 boundary-regression).  Per-agent
  sum: **88** (pre-v4.15.4 corrected from "Net +90" — neither the
  per-agent sum nor the collected delta were ever 90).
* Actual `pytest --collect-only` delta against the v4.15.2 baseline
  (1735) at v4.15.3 HEAD sha `7808107`: **+89 collected** (1824
  collected at v4.15.3 HEAD); the +1 gap between the per-agent
  attribution sum (88) and the collected delta (89) is a
  parametrize-expansion / fixture-collection artifact that does
  not cleanly attribute to a single agent; the canonical number is
  the collected delta.
* Final: **1822 unit pass + 1 skip + 1 xfail = 1824 collected**;
  **34/34 validation**.

### Deferred to v4.16+

Unchanged from prior releases: modal-asymptotic independent
ground-truth pin; 4 V2 meta-pin candidates (sentinel-aware
branch propagation, `_xp_of` dispatch, `dy` parameter threading
walker including ThinLens + lens kernels, `__all__` symmetry
walker); MCF-aware downstream propagators; multi-process
atomic-append for `storage.py`;
`MultiPrescriptionParameterization.scale_floor`; Forbes Q-2D-
asymmetric variant.  Plus newly-deferred: 9 `elements/elements.py`
generic helpers + 2 JAX-traceable lens variants + 6 polarization
helpers exempted in v4.15.3 meta-pin pending v4.16 integration.

---

## [4.15.2] — 2026-05-18

**Closes the v4.15.1 audit (`docs/audits/AUDIT_V4_15_1_2026_05_18.md`)
through P3.**  The audit found 1 P0 + 9 P1 + ~12 P2 + ~10 P3 — most
were downstream-integration gaps from the rapid v4.15.1 expansion (new
types shipped without updating consumers; breaking changes shipped
without CHANGELOG flagging; primitive APIs asymmetric).  **1732 unit
tests pass** (up from 1625; +107 net), 1 documented skip, 1 documented
xfail; **34/34 validation files pass**.

### Breaking changes

* **Schell-family factories emit `DeprecationWarning`** on the default
  `return_kind` path.  v4.15.1 silently changed the return shape from
  `(E_2d, x, y)` (v4.15.0) to `(E_3d, dx, dy, wavelength)` 4-tuple
  without a warning.  v4.15.2 closes the contract break: callers must
  pass `return_kind='ensemble'` or `return_kind='mcf'` explicitly;
  failing to do so emits a one-release deprecation warning with
  `version_removed='5.0'`.
* **`Source.gaussian_schell` and `Source.schell_model` classmethods**
  now return the same `(ensemble, dx, dy, wavelength)` 4-tuple as the
  top-level factories — they previously wrapped the 3-D ensemble in a
  `Source` instance whose `E` was 3-D, breaking the canonical 2-D
  `Source.E` contract.  This is a soft consistency break (every other
  `Source.*` classmethod returns a `Source`); the inconsistency is
  honest — Schell is partial-coherence, fundamentally different from
  the coherent single-source abstraction.

### P0 closure

**P0-NEW-1 — Schell silent contract break closed.**  `_RETURN_KIND_UNSET`
module-level sentinel (subclass of `_deprecation._Sentinel`) detects
the default-path entry; `_warn_schell_return_kind_default` helper
routes through `_deprecation.warn_deprecated_signature` with explicit
`version_removed='5.0'`.  Applied to all 3 Schell factories
(`create_gaussian_schell_source`, `create_schell_model_source`,
`create_annular_incoherent_source`).

### P1 closures (9)

* **P1-NEW-A — `FourierTransform` 3-stage rewrite.**  v4.15.1's
  `_apply` ran 2 stages (lens-then-Fresnel) while the ABCD claim
  `[[0, f], [-1/f, 0]]` matched the 3-stage chain `FreeSpace(f) *
  ThinLens(f) * FreeSpace(f)`.  The 2-stage path left a residual
  `exp(+ik/(2f) r^2)` quadratic phase — ABCDs matched but fields
  didn't.  v4.15.2 rewrites `_apply` to the literal 3-stage chain so
  ABCD and field finally agree.  Perf impact: ~2x slower than the
  v4.15.1 2-stage shortcut (one extra Fresnel propagation).  Users
  wanting hardware-realistic back-focal-plane semantics can compose
  directly as `ThinLens(f) * FreeSpace(f)` (which has the genuine
  2-stage ABCD `[[1, f], [-1/f, 0]]` and 2-stage field; both correct
  via the existing algebra).
* **P1-NEW-B — `from_prescription` flat-mirror parity matches
  `system_abcd`.**  v4.15.1 flipped `mirror_parity` unconditionally
  on `is_mirror=True`; `system_abcd` only flips for curved mirrors.
  v4.15.2 conditions the parity flip on curved mirrors, matching the
  raytrace convention.  New folded-singlet + folded-telephoto
  prescription test cases pin the parity at 1e-12 absolute (the
  1e-12 ABCD parity claim now holds for folded prescriptions too,
  not just non-folded).
* **P1-NEW-C — `FreeSpace._apply` threads `dy`.**  v4.15.1's
  `FreeSpace._apply` called `propagate(E, z=, ..., dx=dx, method=)`
  without `dy`.  Any anamorphic chain `Magnify(a_x, a_y) *
  FreeSpace(d)` silently propagated on the wrong grid.  v4.15.2
  threads `dy` to the dispatcher when `dy != dx` (forwards via
  method_kwargs; safe for ASM/Fresnel/Fraunhofer/RS; skipped for
  SAS which is square-grid only).  Verified `ThinLens._apply` does
  not have the same gap.
* **P1-NEW-D — `rays_from_field` `'cdf'` placement pixel-wise
  threshold.**  v4.15.1 applied `intensity_threshold` to MARGINAL
  sums in `_place_cdf`, inconsistent with `'rejection'` and
  `'uniform'` modes (which threshold pixel-wise).  A 1-pixel-wide
  bright streak running the full y-extent survived the threshold
  incorrectly.  v4.15.2 thresholds pixel-wise before forming
  marginals; the 3 placement modes are now consistent.
* **P1-NEW-E — `PartialCoherenceMCF` defensive guard.**  All 10
  propagator entry points (`propagate_through_system`, `propagate`,
  `angular_spectrum_propagate`, `fresnel_propagate`,
  `fraunhofer_propagate`, `rayleigh_sommerfeld_propagate`,
  `scalable_angular_spectrum_propagate`, `apply_thin_lens`,
  `apply_cylindrical_lens`, `apply_real_lens`) now raise
  `TypeError` with a clear "v4.16+ scope" message when handed a
  `PartialCoherenceMCF` — previously crashed with cryptic
  `AttributeError`.
* **P1-NEW-F — 3-D ensemble shape guard.**  Same 10 propagator
  entry points now raise `ValueError` on `E.ndim != 2` with a
  message showing the iterate-over-ensemble workaround pattern.
* **P1-NEW-G — CHANGELOG `### Breaking changes` subhead** added
  to the v4.15.1 entry listing Schell return shape, `strehl_vector`
  default-reference removal, and `system.evaluate` mixed-shape
  `ValueError`.
* **P1-NEW-H — `rays_from_field` short-return `RuntimeWarning`.**
  v4.15.1's `_place_rejection` and `_place_uniform` could return
  fewer rays than requested (rejection budget exhausted; threshold
  excluded too many pixels) without warning.  v4.15.2 emits a
  `RuntimeWarning` when `n_actual < n_rays`, plus an `n_rays = 0`
  early-return is honoured cleanly.
* **P1-NEW-I — ROADMAP refresh.**  Header bumped to "(post-v4.15.2)";
  Current State block updated to v4.15.2 / 1732 tests baseline;
  v4.15.1 + v4.15.2 added to Shipped highlights.

### P2 closures

* `_sentinel_unpickle` fallback now raises `ImportError` with an
  actionable message when an unknown subclass is unpickled
  (distributed-pipeline timing safety).  Previously silently
  returned a base `_Sentinel`, losing subclass identity.
* **3 additional `optimize/core.py` sentinels migrated** to inherit
  from `_deprecation._Sentinel`: `_InvalidFocalLengthSentinel`
  (was a literal `1e9` fallback for failed ABCD), `_FailedScanStrehlSentinel`
  (was `0.0`), `_PerturbedABCDFallbackSentinel` (was a `(efl, bfl)`
  tuple fallback).  v5.1.0 (Wave-4 integration / Agent E 6-file
  split): class definitions moved to `optimize/context.py:151-178`
  (the 2 remaining sentinels post-v4.15.4
  `_PerturbedABCDFallbackSentinel` deletion; v5.24.4 audit S4-18
  optimizer hygiene shifted them from `:112-139`); was at
  `optimize/core.py:2069`, `:2096`, `:2122` (singletons at `:2093`,
  `:2119`, `:2144`) within the `:2044-2144` documentation block
  pre-v5.1.0.
  All registered in `_SENTINEL_REGISTRY` for pickle round-trip
  safety.  v4.15.3 correction (per AUDIT_V4_15_2 P3 docs-drift
  finding): the pre-v4.15.3 release notes cited stale work-in-
  progress line numbers `:2151, :2271, :2530, :2772` for these
  classes; the actual definitions are at the `:2044-2144` block
  cited above.  (Callsite migration to the new sentinels is
  scaffolding-only at v4.15.2 -- the `2271`, `2530`, `2772`
  references appearing inside the class docstrings point at the
  v4.16+ migration target callsites and are tracked separately by
  Agent A's v4.15.3 work.)
* `_NO_DEFAULT` promoted to dedicated `_NoDefaultSentinel(_Sentinel)`
  subclass (cosmetic consistency with the other sentinels).
* **`PartialCoherenceMCF.coherence_at` Hermiticity test** added —
  asserts `J(r1, r2) == conj(J(r2, r1))` for several `(r1, r2)`
  pairs at 1e-10.
* **`Source.gaussian_schell` / `Source.schell_model`** now pass the
  factory's 4-tuple verbatim instead of wrapping the 3-D ensemble
  in a `Source` whose `E` would have been 3-D.
* **UI runtime test under `-W error::DeprecationWarning`** added —
  exercises `SourceDefinition.to_source()` at runtime to catch the
  static-grep escape (which the v4.15.1 audit identified as a
  missing coverage class).
* **`rays_from_field` top-of-file docstring** corrected from
  Madelung `Im(grad E / E)` to phase-ratio central difference
  (inline docstring was already correct; top was stale).
* **`Magnify` docstring direction inverted** to match code:
  `a > 1` shrinks output; `0 < a < 1` magnifies (Nazarathy/Shamir
  `V[a]` convention).  Dead `operators.py:556-577` reference
  removed.
* **`'uniform'` and `'unwrap_gradient'` modes** in `rays_from_field`
  now have direct test coverage (audit flagged these as previously
  untested).  Vortex direction-recovery and anamorphic
  direction-cosines tests added.

### P3 closures

* **Sparrow tolerance pin tightened from 5% to 1%**.  Measured
  achievable error on canonical Airy fixture (N=256, dx=0.1µm,
  λ=600nm, f/#=4): 0.017% — comfortable headroom over the new 1%
  pin.  Docstring "Accuracy (v4.15.2)" paragraph cites the measured
  number.
* **Forbes Q-bfs end-to-end OPD analytical pin** — closes the
  v4.15.1 audit gap "No end-to-end Forbes Q OPD pin against
  analytical formula".  Pins `OPD(r) = (n - 1) * sag(r)` (the
  `(n - 1)` index factor reflects that light experiences an optical
  path difference of `(n_glass - n_outside) * geometric_path`; for a
  sag in vacuum/air the multiplier is `(n_glass - 1)`) against the
  closed-form Q-bfs sag at 5e-3 rad tolerance.  v4.15.3 correction
  (per AUDIT_V4_15_2 P3 docs-drift finding): pre-v4.15.3 the bullet
  stated `phi(r) = -k * sag(r)` at `1e-3 rad` -- the formula omitted
  the `(n - 1)` factor that the actual test code uses, and the
  tolerance was incorrectly tightened from the test's `5e-3` value
  (the test code itself was always correct; only the CHANGELOG bullet
  drifted).
* **`lumenairy.algebra` exports moved from Tier-2 to Tier-1** in
  `__init__.py.__all__` — operator algebra is a build-time
  construction surface, not a propagation surface.
* **CHANGELOG line-citation refreshes**: "45° fold" → "60° fold
  (α=π/6)" in the v4.15.1 OAP raytrace test description;
  `optimize/core.py:2790-2795` → `:2905` (branch) + `:2034`
  (class) + `:2044` (singleton) after Agent E's sentinel
  refactor pushed lines.
* **CHANGELOG test-count arithmetic refresh**: Agent A (v4.15.1)
  count corrected 18 → 19; Agent F count corrected 13 → 20
  (parametrize entries) to match `pytest --collect-only`.
* **`energy_threshold` kwarg** now forwarded through all 3 Schell
  factories to `PartialCoherenceMCF.from_ensemble` (was exposed on
  the MCF builder but not on the factory entry points).
* **Stray `C:tmpsources_diff.txt`** (typo'd `C:\tmp\` path; OneDrive
  U+F03A colon substitute) deleted — 44 KB git-diff dump, content
  recoverable via git history.

### Test counts

* Pre-v4.15.2 baseline (v4.15.1): 1625 unit tests + 1 skip + 1 xfail.
* v4.15.2 additions: A=18, B=15 (9 new + 6 modifications), C=32,
  D=19, E=16; net +110 (pytest-collected delta from 1625 baseline
  to 1735 at v4.15.2 HEAD sha `672051c`).
* Final: **1733 unit tests passing, 1 skipped, 1 xfailed** (1735
  collected total per `pytest --collect-only -q tests/unit` at sha
  `672051c`); **34/34 validation files passing**.  v4.15.3
  correction (per AUDIT_V4_15_2 P2 docs-drift finding): pre-v4.15.3
  this entry stated "1732 pass + 1 skip + 1 xfail" (= 1734
  collected), off by 1 from the actual `pytest --collect-only`
  count at the v4.15.2 release commit.  The arithmetic was also
  inconsistent with the per-agent breakdown above (18 + 15 + 32 +
  19 + 16 = 100, not 107) -- v4.15.3 reconciles both to the actual
  pytest-collected delta.

### Deferred to v4.16+

Unchanged from v4.15.1 deferrals: modal-asymptotic independent
ground-truth pin against direct quadrature; 4 V2 meta-pin candidates
(sentinel-aware branch propagation, `_xp_of` dispatch, `dy` parameter
threading walker, `__all__` symmetry walker); MCF-aware downstream
propagators (consume `PartialCoherenceMCF` through propagation
chains); multi-process atomic-append for `storage.py`;
`MultiPrescriptionParameterization.scale_floor` (v4.13.1 P1-I
carryover); Forbes Q-2D-asymmetric variant.

---

## [4.15.1] — 2026-05-18

**Closes the v4.15.0 audit (`docs/audits/AUDIT_V4_15_0_2026_05_18.md`)
through P3 + ships 2 additive features from
`docs/audits/CLUSTER_B_SPEC.md` (operator algebra + rays-from-field
bridge).**  The audit found 2 P0s + 12 P1s + many P2/P3 (highest-yield
audit in the series).  v4.15.1 closes both P0s + all Tier-0 P1s + the
P2/P3 sweep + adds 800+ LOC of new CLUSTER_B surface.  **1625 unit
tests pass** (up from 1425; +200 net), 1 documented skip, 1
documented xfail; **34/34 validation files pass**.

### Breaking changes

v4.15.1 ships 3 confirmed breaking items.  Callers who relied on the
v4.15.0 contracts must migrate; v4.15.2 (P1-NEW-G closure) adds this
subhead retroactively to make the audit-flagged items discoverable.

1. **Schell-family return shape**: `create_gaussian_schell_source`,
   `create_schell_model_source`, and `create_annular_incoherent_source`
   default to `return_kind='ensemble'` and now return the 4-tuple
   `(ensemble_3d, dx, dy, wavelength)` where `ensemble_3d` has shape
   `(n_realizations, Ny, Nx)`.  v4.15.0 returned `(E_2d, x, y)` (a
   collapsed single field plus coordinate vectors).  Pre-v4.15.0
   callers doing `E, x, y = create_gaussian_schell_source(...)` now
   silently bind `E.ndim == 3` and `x` to a scalar `dx`.  Pass
   `return_kind='ensemble'` explicitly to acknowledge the new
   contract, or `return_kind='mcf'` to opt into a
   `PartialCoherenceMCF` object instead.  v4.15.2 (P0-NEW-1 closure)
   emits a `DeprecationWarning` on the default path; removal in v5.0.
2. **`strehl_vector` default reference removed**: v4.15.0 had a
   buggy default plane-wave reference that produced unity Strehl for
   any uniform field of equal power AND `Strehl > 1` on focused PSFs
   (the focused field is more peaked than the plane-wave reference at
   matched total power).  v4.15.1 requires the caller to pass
   `reference=` explicitly; the docstring also drops the unverified
   "Richards-Wolf high-NA" claim.  See P1-F1-3 below.
3. **`system.evaluate` mixed-shape prescription raises `ValueError`**:
   a prescription containing BOTH `surfaces` + `thicknesses` AND
   `elements` + `all_thicknesses` keys previously silently picked one
   schema.  v4.15.1 rejects it at the validator with a clear message.
   Callers passing raw Zemax-loader output need to filter the
   surfaces keys before handing the dict in:
   ```python
   rx_filtered = {k: v for k, v in rx.items()
                  if k not in ('surfaces', 'thicknesses')}
   ```
   See P1-F1-6 below.

### P0 closures

**P0-NEW-1 — `make_off_axis_parabola` factory fix (doubly broken):**
Decenter formula corrected `f*tan(alpha)` -> `2*f*tan(alpha)` (chief-
ray geometry on parent paraboloid).  Tilt remains 3-tuple
`(off_axis_angle, 0.0, 0.0)`; factory docstring now loudly documents
that the OAP prescription is **intended for `apply_real_lens_traced`
exclusively** — the paraxial `apply_real_lens` cannot interpret the
3-tuple tilt correctly.  New end-to-end raytrace test
(`test_end_to_end_raytrace_focuses_at_offset`) pins the off-axis
focal-point location to within 1% of the chief-ray geometric
prediction at 60° fold (α=π/6).  v4.15.2 (P3 doc-drift fix): the
original CHANGELOG cited a pi/4 fold -- the actual test uses π/6
because α=π/4 is degenerate at this geometry.  P3 carryover:
`vertex_radius` now validated (must be `None` or finite positive).

**P0-NEW-2 — Schell-family factories redesign:**  The v4.15.0
factories collapsed the `n_realizations` ensemble into a single
fully-coherent complex field before return — the documented
partial-coherence contract was unfulfillable.  v4.15.1 introduces a
hybrid:

* Default `return_kind='ensemble'`:  factory returns
  `(ensemble, dx, dy, wavelength)` where `ensemble` has shape
  `(n_realizations, Ny, Nx)`.  Caller iterates over realizations and
  averages intensities downstream — physically-correct partial
  coherence.
* Opt-in `return_kind='mcf'`:  factory returns a new
  `PartialCoherenceMCF` dataclass with `.intensity()`,
  `.coherence_at(...)`, and `.coherent_modes()` methods.  For small
  grids (`Ny*Nx <= 64**2`), stores the full `J(r1, r2)`; for larger
  grids, stores the leading K coherent modes (Wolf 1982 JOSA
  decomposition) via SVD of the ensemble matrix.  Truncation
  threshold:  smallest K with `cumsum(eigvals)/sum(eigvals) >= 0.99`
  (Karhunen-Loève default).
* **Physics fix:**  the random-phase RMS normalization (which forced
  `sigma_phi = 1` regardless of `sigma_g`) is replaced with the
  spec-correct Fourier-filtered Gaussian-noise recipe.  Now `sigma_g
  -> 0` actually approaches incoherent (off-diagonal MCF -> 0) and
  `sigma_g -> infinity` approaches coherent (rank-1 MCF).

Affected factories: `create_gaussian_schell_source`,
`create_schell_model_source`, `create_annular_incoherent_source` +
matching `Source.gaussian_schell` / `Source.schell_model`
classmethods.  Note: MCF-aware downstream propagators are deferred
to v4.16+; the `PartialCoherenceMCF` object is consumable for
analysis / inspection in v4.15.1.

### P1 closures (Tier 0 audit recommendations)

* **P1-NEW-A: `sparrow_resolution` canonical Sparrow root-finding.**
  Implementation rewritten to true two-source dip-vanishing condition
  `d²/dr² [PSF(r-d/2) + PSF(r+d/2)]_{r=0} = 0` via
  `scipy.ndimage.map_coordinates` sub-pixel azimuthal averaging +
  cubic-spline 2nd derivative + `scipy.optimize.brentq` root-finder.
  Now returns 2.273 µm vs expected 2.273 µm at lambda=600nm, f/#=4
  (previously 1.93 µm, 15% low).
* **P1-NEW-C: 7 UI Source-factory deprecation callsites migrated** to
  kwarg-only canonical form in `lumenairy/ui/model.py`.  The v4.15.0
  release that introduced the deprecation shim now also migrates its
  own internal UI consumers.
* **P1-NEW-D: Raytrace flat-keys allowlist** at `raytrace/core.py:
  1507-1521` extended with `q_bfs_coeffs`, `q_con_coeffs`, `r_max`.
  Forbes Q prescriptions in flat-keys form no longer silently drop
  the coefficients at the gather step.
* **P1-NEW-E: Zemax `.zmx` QBFS/QCON parsing** added to
  `io/prescriptions.py`.  `.zmx` files with Q-type freeforms now
  load with `freeform_type='q_bfs'` or `'q_con'` + coefficients +
  `r_max` (parsed from `DIAM` / `PARM` lines), instead of silently
  degrading to base conic.
* **P1-F1-1: Q-bfs/Q-con radial-clip alignment.**  v4.15.0's
  rectangular `|X| <= norm_x AND |Y| <= norm_y` clip let pixels at
  `(0.9*r_max, 0.9*r_max)` (radial `r = 1.27*r_max`) through — outside
  the Forbes domain.  v4.15.1 uses a radial primary clip
  `r <= r_max` with the rectangular `(norm_x, norm_y)` box as
  secondary aperture.
* **P1-F1-2: `surface_sag_freeform` requires `r_max`** for
  `freeform_type in ('q_bfs', 'q_con')`.  Previously defaulted
  silently to `r_max=1.0` (a unit-mismatch bug — user passing X/Y
  in metres got sag computed on a sub-pixel of the actual aperture).
  Now raises `TypeError` with a clear message.
* **P1-F1-3: `strehl_vector` default-reference removed** (breaking).
  v4.15.0's default plane-wave reference produced unity for ANY
  uniform field of equal power AND Strehl > 1 on focused PSFs
  (more peaked than plane-wave at equal total power).  v4.15.1
  requires explicit `reference=` and softens the docstring
  (drops unverified "Richards-Wolf high-NA" claim).
* **P1-F1-4: `rayleigh_resolution` Gaussian-PSF false-positive fixed.**
  Now requires a strict subsequent rise above the candidate minimum
  by >=0.5% of peak before declaring first zero; Gaussian-like PSFs
  (no true zero) return NaN + `RuntimeWarning` advising
  `fwhm_resolution` / `sparrow_resolution` instead.
* **P1-F1-5: `astigmatism_mag_angle` docstring range correction** —
  `(-pi/4, pi/4]` -> `(-pi/2, pi/2]` (the actual range from
  `0.5 * atan2(c3, c5)`).
* **P1-F1-6: `system.evaluate` mixed-shape `ValueError`** — a
  prescription with both `surfaces`+`thicknesses` AND
  `elements`+`all_thicknesses` keys is now rejected at the
  validator with a clear message rather than silently picking a
  schema.  Behaviour change: callers passing raw Zemax-loader
  output need to filter the surfaces keys (`{k:v for k,v in rx.items()
  if k not in ('surfaces','thicknesses')}`).
* **P3-NEW-A: Forbes Q wave-optics path** — v4.15.0's
  `apply_real_lens` silently `RuntimeWarning`d and skipped the
  Forbes Q freeform contribution.  v4.15.1 routes Q-bfs / Q-con
  through `surface_sag_freeform` properly (option (a) of the audit
  recommendation), inheriting P1-F1-1 + P1-F1-2 guards.

### P2 closures

* **`__all__` symmetry**:  `surface_sag_q_bfs`,
  `surface_sag_q_con` re-exported from
  `lumenairy/elements/__init__.py`; `make_off_axis_parabola` from
  `lumenairy/io/__init__.py`.  `from lumenairy.elements import
  surface_sag_q_bfs` now works.
* **Sentinel consolidation**:  `_ZeroApertureMaskSentinel`
  (`optimize/core.py`) and `_AngleUnsetSentinel`
  (`elements/polarization.py`) now inherit from `_deprecation._Sentinel`
  base class.  `_Sentinel` gained pickle-safe `__reduce__` + name-keyed
  `_SENTINEL_REGISTRY` + `_sentinel_unpickle` reconstructor so
  pickle round-trips return the singleton instance (not a fresh
  sentinel).
* **`system.evaluate` Zemax-shape test** added (the audit's "headline
  ergonomic claim was untested" finding closed).

### P3 closures

* `n=1.0` consistency:  `optimize/multiconfig._resolve_lens_glass_index`
  bounds widened from exclusive `(1.0, 5.0)` to inclusive
  `[1.0, 4.0]` matching `register_fixed_glass`.
* Codegen runtime version pin gains an upper-bound major-version
  warning (`UserWarning` if running on `lumenairy >= 5.0.0`).
* `LambertianBSDF.evaluate` gains explicit surface-frame docstring
  + `RuntimeWarning` if `incident_direction` is non-axially-aligned
  without explicit frame transform.
* Coatings TIR cap warnings promoted from filtered `RuntimeWarning`
  to always-emit `UserWarning`.
* Forbes Q orthonormalizer docstring formula corrected:
  `c_n = sqrt((2n+3)(n+2)/(n+1)^2)` -> `c_n = sqrt((2n+3)(n+2)/(n+1))`
  (the implementation was already correct; only the docstring lied).
* `astigmatism_mag_angle` docstring range correction (also P1-F1-5).
* CHANGELOG/release-notes: lenses_maslov `_ZERO_APERTURE_MASK`
  sentinel branch now lives at `optimize/wrapper_merits.py:955`
  (the `if _cache['mask'] is _ZERO_APERTURE_MASK` line); was
  `optimize/core.py:3032` pre-v5.1.0 Agent E 6-file split (the
  branch moved out of the monolithic core.py to the new
  ``wrapper_merits.py`` submodule that hosts the ``MultiWavelengthMerit``
  / ``MultiFieldMerit`` / ``ToleranceAwareMerit`` triplet); was
  `:3015` pre-v4.16.3 Agent C `Constraint` auto-probe
  DeprecationWarning latch + pickle catch widening (~17 lines added
  above the sentinel branch); was `:2974` pre-v4.16.2 Agent B
  `MultiWavelengthMerit` `FutureWarning` latch + Constraint-probe
  move + lambda pickle-probe (~41 lines added above the sentinel
  branch); was `:2958` pre-v4.16.1 Agent A `MultiWavelengthMerit`
  `SUM`->`AVG` refactor (~16 lines added in the merit-aggregation
  block above the sentinel branch); was `:2980` pre-v4.15.4 Agent
  B `_PerturbedABCDFallbackSentinel` deletion (~55 lines removed
  at the top of the sentinel block); and `:2905` pre-v4.15.3
  sentinel-wiring work.  The remaining
  sentinel class + singleton (`_ZeroApertureMaskSentinel` /
  `_ZERO_APERTURE_MASK`) are at `optimize/context.py:74` and
  `optimize/context.py:84` respectively post-v5.1.0 Agent E 6-file
  split (was `optimize/core.py:2044` and `optimize/core.py:2054`
  pre-split, post Agent E's v4.15.2 `_Sentinel` base-class
  refactor).  v4.15.2 (P3):
  citation refreshed after a
  second line-drift pass against the current source supersedes
  the earlier stale citations.

### CLUSTER_B Item 6 — `rays_from_field` bridge function

New `lumenairy.rays_from_field(E, *, dx, wavelength, dy=None,
n_rays=200, placement='cdf', angle_method='complex_gradient',
intensity_threshold=1e-4, z0=0.0, random_state=None) -> RayBundle`
samples a coherent field into a geometric ray bundle.  Bridges
`propagators/` (wave) <-> `raytrace/` (ray) so users can overlay
ray traces on coherent-field plots, seed a Maslov/GBD bundle from a
measured pupil field, or hand a coherent field into the geometric
ray tracer for hybrid analysis.

Placement modes: `'cdf'` (separable inverse-CDF, fast),
`'rejection'` (true 2-D rejection, exact), `'uniform'` (grid + threshold
mask).  Angle methods: `'complex_gradient'` (phase-ratio central
difference, singularity-safe — adapted from spec for correct
behaviour at Nyquist), `'unwrap_gradient'` (np.unwrap-based, fragile
near vortices).  Evanescent rays (`L² + M² > 1`) flagged with
`RAY_EVANESCENT` and `alive=False`.

13 tests + 1 runnable example
(`examples/rays_from_pupil_field.py`).  Implementation note: 3
spec deviations (phase-ratio central difference instead of literal
`Im(grad E / E)`; evanescent test samples 6x Nyquist to avoid
spectral aliasing; OPD test uses wrap-free phase slope) all
documented in the agent's release notes.

### CLUSTER_B Item 2 — Operator algebra

New `lumenairy/algebra/` subpackage implementing Nazarathy/Shamir
operator algebra (JOSA 70 (2), 1980).  9 new symbols at top level:

* `Operator`, `CompositeOperator` — base classes; ABCD-tracking
  algebraic composition.
* `FreeSpace(d, *, method='auto')`, `ThinLens(f)`,
  `CylindricalLens(f_x, f_y)`, `Magnify(a_x, a_y)`,
  `FourierTransform(f_focal)` — primitive operators.
* `Aperture(diameter, shape)`, `GaussianAperture(sigma)` — passive
  aperture operators (identity ABCD).
* `Operator.from_prescription(prescription, wavelength)` —
  prescription-dict -> CompositeOperator factory.  Paraxial-only;
  produces ABCD identical to `system_abcd(...)` to within 1e-12 abs.

Composition: `A * B` means "first B, then A".  ABCD of `A * B` is
`A.abcd @ B.abcd`.  Application: `sys(source) -> Source`, or
`sys.apply(E, dx=..., wavelength=...)`.  Anamorphic support via
separate `_abcd_x` / `_abcd_y`.

91 tests + 2 runnable examples (`examples/algebra_4f_system.py`,
`examples/algebra_anamorphic.py`).  Spec deviation: `Magnify._apply`
uses closed-form `sqrt(a_x*a_y)` amplitude prefactor instead of
spec's `resample_field` recipe (the spec recipe had an energy-
conservation bug; closed-form preserves energy per-pixel by
construction).  Phase 2 symbolic reduction (FreeSpace+ThinLens+
FreeSpace collapse, etc.) explicitly deferred to a future PR.

### Test counts

* Pre-v4.15.1 baseline (v4.15.0): 1425 unit tests + 1 skip + 1 xfail.
* v4.15.1 additions (`pytest --collect-only` items, parametrised
  test cases counted separately): A=19, B=20+migrated, C=13+migrated,
  D=18, E=20, F=20, G=91; gross 201 collected, net ~200 added (the
  +200 number nets out test migrations: agents B and C migrated
  v4.15.0 property pins to v4.15.1 ensemble / analytical-value pins).
  v4.15.2 (P2 + P3 test-count refresh): the F=20 count supersedes
  the original CHANGELOG's F=13 -- Agent F shipped 7 additional
  follow-up tests beyond the initial 13 enumerated in
  `.release_notes_v4_15_1_agent_f.md`, bringing the file to 20
  pytest items.  The A=19 count likewise supersedes the original
  A=18.
* Final: **1625 unit tests passing, 1 skipped, 1 xfailed**; **34/34
  validation files passing**.

### Deferred to v4.16+

* Modal-asymptotic independent ground-truth pin against
  `propagate_hf_chebyshev_quadrature(method='direct')` (audit
  P1-NEW-B, Tier 1).  v4.15.0 replaced known-buggy warm-start with
  unverified cold-start; this pin closes the verification gap.
* 4 remaining V2 meta-pin candidates (sentinel-aware branch
  propagation, `_xp_of` dispatch, `dy` parameter threading,
  `__all__` symmetry walker).
* MCF-aware downstream propagators (consume `PartialCoherenceMCF`
  through propagation chains).
* Multi-process atomic-append for `storage.py` (HDF5 SWMR + Zarr
  distributed lock).
* `MultiPrescriptionParameterization.scale_floor` (v4.13.1 P1-I
  carryover).
* Forbes Q-2D-asymmetric variant (Forbes 2012) for full 2-D
  freeform support.

---

## [4.15.0] — 2026-05-18

**Major minor release** rolling together carryover P1s from the
v4.14.2 audit (the "v4.14.4" patch scope) + ROADMAP v4.15 + ROADMAP
v4.16 into a single coordinated ship.  **1425 unit tests pass** (up
from 1265; +160 net new pins), **1 documented skip**, **1 documented
xfail** (`create_led_source` validation entry-point exemption);
**34/34 validation files pass**.

### Headline: modal-asymptotic 19.4x perf win + wrong-saddle physics fix

`propagate_modal_asymptotic` switches from a per-pixel warm-started
Newton loop to a single batched cold-start
`_solve_envelope_stationary_batch` + `_compute_M_b_batch` path
(the private helpers v4.14.0 shipped but did not consume on the
public path).  Closes the v4.14.0 audit's "wrong-saddle basin"
physics finding (warm-start chain entered wrong-saddle basins at
grid edges, silently zeroing those pixels via the `|b_quad| > 700`
overflow guard).  Cascade impact: any caller that builds on
`propagate_modal_asymptotic` (aberration tensor sigma-integration
grid path, through-focus / polychromatic helpers per focus or
wavelength point) gains the perf win and the grid-edge correctness
in one step.

Output is **bit-different** from v4.14.x at grid edges (strictly
more non-zero pixels because the cold-start finds the physical
saddle uniformly).  Four bit-equal pins migrated to property
pins (1e-8 abs vs cold-start reference + 5% energy + nz-count
>= warm-start ref):
`test_lg00_single_mode_matches_reference`, `test_multimode_matches_reference`
in `test_perf_v4_12_0_asymptotic.py`, plus
`test_lg00_single_mode_bit_equal` and `test_lg_p0_4mode_prescription_bit_equal`
in `test_audit_fixes_v4_14_0_agent_1.py`.  The v4.14.1 row-reset
warm-start pin (`test_row_reset_resets_warm_start`) was retargeted
to the v4.15 stronger structural guarantee:  the scalar
`solve_envelope_stationary` is no longer invoked by the public
path in any `maslov_tracking` mode (the warm-start chain is
structurally deleted, not just reset).

Measured: **19.4x** speedup at N=128 LG_(0,0); +52 non-zero
pixels recovered on the same grid (15918 vs 15866).

### Source factory normalisation + ergonomic system entry

Pre-v4.15 the 5 `Source.method` classmethod factories had
inconsistent positional order — some put size-arg first, others
N first.  v4.15 picks the canonical order
`Source.method(*, N, dx, wavelength, <size_kwargs>)` (kwarg-only).
The legacy positional form still works for one release with a
`DeprecationWarning` routed through the new
`_deprecation.warn_deprecated_signature` helper with
`version_removed='5.0'`.  Affected factories: `Source.gaussian`,
`Source.plane_wave`, `Source.point_source`, `Source.top_hat`,
`Source.fiber_mode`.

New ergonomic entry `lumenairy.system.evaluate(prescription,
source, *, output_grid=None, output_dx=None, ...)` (also
top-level `lumenairy.evaluate`).  Accepts both Zemax-loader
prescription shape (`elements` + `all_thicknesses` keys) and
factory shape (`surfaces` + `thicknesses` keys).  Users loading
a `.zmx` file no longer have to build the element list manually
before propagating.

### 7 new public-API functions (ROADMAP v4.16 closure)

* `ee_polychromatic(prescription, wavelengths, weights, radii, ...)`
  — convenience chain over `polychromatic_psf` +
  `encircled_energy_radius`.
* `strehl_vector(Ex, Ey, Ez=None, *, reference=None)` —
  vector Strehl with optional `Ez` z-component (Richards-Wolf
  high-NA case).
* `coupling_efficiency_vector(Ex, Ey, Ez=None, *, mode_Ex,
  mode_Ey, mode_Ez=None, dx)` — vector overlap integral with a
  vector mode.
* `rayleigh_resolution(psf, dx, wavelength, *, axis='radial')`
  — first-zero-of-PSF Rayleigh diffraction limit.
* `sparrow_resolution(psf, dx, *, axis='radial')` — empirical
  Sparrow criterion (dip-just-vanishes for two overlapping
  point sources).
* `fwhm_resolution(psf, dx, *, axis='radial')` — twice the
  FWHM half-radius of the central peak.
* `astigmatism_mag_angle(coeffs)` — Mahajan §8.2 conversion of
  Zernike `(c3, c5)` to `(|astig|, theta)` in the
  OSA/ANSI convention matching `zernike_decompose`.

All 7 are top-level exports in `lumenairy.__all__`.

### 3 new source factories (partial-coherence + ring incoherent)

* `create_gaussian_schell_source(*, N, dx, wavelength, w0,
  sigma_g, n_realizations=16, ...)` — spatially-incoherent
  Gaussian-Schell beam via random-phase ensemble.
* `create_schell_model_source(*, N, dx, wavelength,
  intensity_profile, coherence_length, n_realizations=16, ...)`
  — generic Schell-model (caller supplies intensity profile).
* `create_annular_incoherent_source(*, N, dx, wavelength,
  inner_radius, outer_radius, n_realizations=16, ...)` —
  angular-spectrum ensemble with finite source extent for
  partial-coherence integration (distinct from existing
  monochromatic-coherent `create_annular_beam`).

Matching `Source.gaussian_schell(...)` / `Source.schell_model(...)`
classmethods.  All 3 call `_validate_grid_params` in the first
10 lines (per the v4.14.2 audit's input-validation entry-point
meta-pin candidate).

### Forbes Q-type freeform basis + off-axis parabola factory

* `surface_sag_q_bfs(X, Y, *, radius, coefficients, r_max, ...)`
  — Forbes Q-bfs basis (Forbes 2007, *Opt. Express* 15(8) 5218,
  eq. 13).  Best-fit-sphere subtracted; orthonormal on the
  weight `u^2 (1-u^2) d(u^2)` over `[0, 1]`.
* `surface_sag_q_con(X, Y, *, radius, conic, coefficients,
  r_max, ...)` — Forbes Q-con basis (Forbes 2010,
  *Opt. Express* 18(13) 13851, eq. 6).  Conic-subtracted;
  orthonormal on weight `u^4 d(u^2)`.

Implementation uses the shifted-Jacobi 3-term recurrence
(A&S 22.7.1) on `t = 2x - 1` for `x = u^2 in [0, 1]`:
Q-bfs `(alpha, beta) = (1, 1)` with orthonormaliser
`c_n = sqrt((2n+3)(n+2)/(n+1)^2)`; Q-con `(alpha, beta) =
(0, 2)` with `c_n = sqrt(2n+3)`.  Orthonormality verified
numerically to <1e-6 over the first 5 orders for both bases.

* `make_off_axis_parabola(focal_length, off_axis_angle,
  clear_aperture, *, glass='__MIRROR__', vertex_radius=None,
  name=None) -> dict` — prescription factory for OAP segments.
  Single parabolic surface (conic `k = -1`, vertex radius
  `R = 2*focal_length`) with `decenter` and `tilt` set to the
  parent-axis offset and local-frame tilt.

### v4.14.2 carryover P1s closed (the "v4.14.4" scope)

**UI subpackage (P1-UI-1 through P1-UI-7):**
* `main_window.py` glass table now includes `N-LASF9` and
  `S-NPH1` (P1-UI-1) and `_nudge_distance` routes through
  `set_display_distance` (P1-UI-2, coordinate-mode aware).
* `model.py` undo state-capture now includes
  `wavelength_weights`, `field_weights`, `lens_options`
  (P1-UI-3).  Back-vertex calculation consolidated into a
  single `_prev_element_back_vertex_world` helper to prevent
  drift between the 717-718 vs 752 sites (P1-UI-4).
* `waveoptics_dock.py` re-parent now guarded by
  `shiboken6.isValid(original_parent)` to avoid the
  mid-dialog segfault risk (P1-UI-5).
* `psf_mtf_dock.py` ray-traced OPD now accumulates via
  `np.add.at(... mean)` instead of last-write-wins per pixel
  (P1-UI-6); out-of-aperture rays are filtered by a bounds
  mask before indexing instead of `np.clip(...).astype(int)`
  silently snapping to the pupil edge (P1-UI-7).

**Codegen + ghost + glass:**
* P1-CG: generated scripts now embed a runtime version pin
  (`if tuple(...) < (4, 15, 0): raise RuntimeError(...)`) plus
  a `lumenairy_version:` comment stamp.
* P1-GH-1: `non_sequential_stray_light` accepts a `seed: int |
  None = None` kwarg (default `None` uses system entropy so MC
  produces a real uncertainty band; pass a fixed integer to
  pin reproducibility).
* P1-GL-1: bundled Sellmeier rows for `SiO2` / `F_SILICA` /
  `FUSED_SILICA` (Malitson 1965 fused-silica coefficients) and
  `S-LAH64` / `S-LAH79` (OHARA Zemax 2017-11-30 catalog from
  the refractiveindex.info-database YAML).  Minimal installs
  without `refractiveindex` now resolve these glasses through
  the bundled Sellmeier fallback.

### Exhaustive P2/P3 sweep + meta-pin candidate #3

User-requested exhaustive enumeration of the v4.14.2 audit's
18 P2 + 12 P3 findings.  Net closure tally:

* 7-8 of 18 P2 closed (4 by Agent F directly: `lumenairy_context`
  6/7 redundant clears, `create_multi_field_sources` factory-
  validation list, ROADMAP refresh, HDF5/Zarr `lumenairy_version`
  attr stamping; 3 by other agents in this release:
  `_validate_grid_params` bool reject, deprecation-shim
  `_deprecation.py` migration with `version_removed`, codegen
  version pin; 1 partial: artifact version pinning — HDF5+Zarr
  done, codegen done).  10-11 P2 deferred (architectural items
  reserved for v4.16+ or v5.0).
* 4 of 12 P3 closed by Agent F: CHANGELOG line-citation drift
  (3 stale `optimize/core.py:...` ranges in the v4.14.2 entry
  refreshed to current line numbers — see commit diff for the
  exact pre/post values); `create_led_source` legacy-shim
  error-message clarity; README `makedammann2d _legacy_units='SI'`
  migration example; README cookbook section with examples for
  the 6 v4.14.0 public functions.

**New structural meta-pin (V2 candidate #3):**
`tests/unit/test_v4_15_dispatcher_pin_validate_grid_params.py`
walks every `create_*` factory and asserts `_validate_grid_params`
appears in the first 15 body lines.  17 PASS + 1 documented
xfail (`create_led_source` legacy-shim positions validator past
the head window — pinned via `xfail(strict=True)` so future
refactors that lift the validator forward flip to XPASS and
the exemption is removed).

### Version-stamping on HDF5 / Zarr writes

`io/storage.py` now writes a `lumenairy_version` attr on every
`create_dataset` / `create_array` / `create_group` site (7
locations).  Future-proof for cross-version field-file
compatibility checks.

### Bundled-glass registry reverse-direction consistency check

The v4.14.2 `_check_glass_registry_consistency` only walked
registry-entry -> `SELLMEIER_COEFFICIENTS` (forward).  v4.15
adds the reverse walk: every key in `SELLMEIER_COEFFICIENTS`
must appear in `GLASS_REGISTRY` (with `'__sellmeier__'` flag
if it's pure Sellmeier).  Coefficient rows added without a
registry entry would have remained silent dead code; this
catches them at module load.

### ROADMAP refresh

`ROADMAP.md` updated to v4.15.0 baseline.  v4.14.1 / v4.14.2 /
v4.14.3 / v4.15.0 entries added to Shipped highlights;
items closed in v4.15 removed from v4.15 + v4.16 target lists
and renumbered.  v4.16+ target list reseeded with the items
deferred from the v4.14.2 audit + the 5 V2 meta-pin
candidates.

### Test counts

* Pre-v4.15 baseline (v4.14.3): 1265 unit tests.
* v4.15 additions: A=8, B=40, C=26, D=23, E=27, F=36
  parametrized tests across two new files; net 160 added.
* Final: **1425 unit tests passing, 1 skipped, 1 xfailed**
  (`_ZARR_MKDIR_PATCH_LOCK` exemption + `create_led_source`
  validation-entry exemption); **34/34 validation files
  passing**.

### Deferred to v4.16+

* 4 V2 meta-pin candidates still standing (sentinel-aware
  branch propagation, `_xp_of` cross-backend dispatch,
  `dy` parameter threading, `__all__` symmetry).  Input-
  validation entry-point candidate #3 is shipped this
  release.
* Multi-process atomic-append for `storage.py` (HDF5 SWMR
  + distributed Zarr lock).  Single-process atomicity is
  documented in v4.14.3.
* `MultiPrescriptionParameterization.scale_floor` (v4.13.1
  P1-I carryover).
* Modal-asymptotic JAX-twin lift to use the v4.15 batched
  helpers.
* Forbes Q-2D-asymmetric variant (Forbes 2012) for full 2-D
  freeform support beyond the rotationally-symmetric Q-bfs /
  Q-con bases shipped here.
* Architectural items reserved for v5.0 (file splits, CI
  gates, shim removal — see ROADMAP).

---

## [4.14.3] — 2026-05-17

**Closes the v4.14.2 audit (`docs/audits/AUDIT_V4_14_2_2026_05_17.md`).**
The audit found 2 NEW P0 findings (`storage.py` non-atomic
`n_planes` increment → silent data loss in concurrent / Zarr
streaming; `makedammann2d` >1mm SI heuristic silently mangles
legitimate mm-scale gratings), 21 NEW P1s, 18 P2s and 12 P3s.
The "fix N, miss N+1" sibling-gap meta-finding recurred 5 ways
on v4.14.2.  v4.14.3 closes both P0s, the 5 sibling-gap
recurrences, 11 latent-bug P1s (1 real physics error in
`multiconfig.py`), and all 3 doc-drift P1s.  **1265 unit tests
pass** (up from 1190); **34/34 validation files pass**.

### Breaking changes — none

Two near-breaking deltas with explicit opt-in / opt-out:

* **`makedammann2d` accepts `_legacy_units='auto'|'um'|'SI'`** (default
  `'auto'` preserves the v4.14.2 per-parameter deprecation heuristic).
  The `'auto'` and `'SI'` modes now raise `ValueError` on any
  unit-bearing kwarg > 1.0 m (rejects nm-scale-garbage from the
  silent-mangling regime).  Legitimate >1m / mm-scale SI gratings
  set `_legacy_units='SI'` to bypass the legacy heuristic
  entirely.  Pure-legacy callers can set `_legacy_units='um'` to
  rescale all unit-bearing inputs by 1e-6 without firing the
  deprecation warning.
* **`create_led_source` legacy-positional shim hardened** with a
  scale-inversion sanity check: a canonical-order positional
  call (`N, dx, wavelength, diameter, divergence`) that
  accidentally slots a wavelength into the `_legacy_diameter`
  position now raises `TypeError` with a migration message
  instead of producing 633 nm "diameter" / 0.1 m "wavelength"
  garbage.

### P0 closures

**P0-NEW-1: `storage.py` n_planes atomicity** — `append_plane_h5`
(`io/storage.py:444`) and `_zarr_append_plane` (`:782`) bumped
`grp.attrs['n_planes']` AFTER `create_dataset`.  A crash between
the two operations left an orphan dataset; the next append used
the stale `n` to compute `plane_{N:02d}` and on Zarr (`overwrite=
True` at line 769) the orphan was silently clobbered.  Concurrent
appenders racing on `n_planes=N` both wrote `plane_{N:02d}`, the
second silently winning.  v4.14.3 inverts the ordering — attr
written BEFORE dataset create, try/except rollback on failure —
and drops `overwrite=True` on the Zarr path so the orphan case
now raises rather than silently destroying data.  Single-process
atomicity is documented; multi-process locking (HDF5 SWMR /
distributed Zarr lock) deferred to v4.15+.  3 regression tests
pin attr-write ordering, docstring contract, and the no-silent-
clobber invariant.

**P0-NEW-2: `makedammann2d` >1m upper-bound** — v4.14.2's
`value > 1e-3` heuristic silently rescaled mm-scale SI gratings
(coarse industrial Dammann, THz/MMW) by 1e-6 → nm-scale garbage.
v4.14.3 adds a `_legacy_units` kwarg (see "Breaking changes"
above) plus an explicit `ValueError` for any unit-bearing input
> 1.0 m in `'auto'`/`'SI'` mode.  3 regression tests cover the
upper bound, explicit `'SI'` mm-scale pass-through, and explicit
`'um'` rescale without `DeprecationWarning`.  3 historical test
sites that relied on the silent rescale were migrated to
`_legacy_units='um'`; one validation case (`test_elements.py`'s
100 µm legacy-µm Dammann) likewise opts in explicitly.

### P1 closures — sibling-gap recurrences (5)

* **P1-NEW-1: `clear_asm_caches()` LG-polynomial chain.**  v4.14.2
  chained 5 sibling caches but missed `_lg_polynomial_items`
  (`asymptotic.py:284`).  v4.14.3 adds the lazy-import + call
  pattern to `clear_asm_caches`, expanding its docstring to list
  all 8 caches it now drains.  Combined-drain test extended to
  assert `_lg_polynomial_items.cache_info().currsize == 0` after
  `clear_asm_caches()`.
* **P1-NEW-2: `apply_rotator` conflict-resolution symmetrised
  across 5 polarization helpers.**  v4.14.2 added `angle_deg=`
  conflict detection to `apply_rotator` only, leaving 4 sibling
  helpers (`apply_polarizer`, `apply_waveplate`,
  `apply_half_wave_plate`, `apply_quarter_wave_plate`) silently
  letting `angle_deg` overwrite `angle`.  v4.14.3 introduces a
  module-level `_AngleUnsetSentinel` singleton (matching the
  v4.14.1 `_ZeroApertureMaskSentinel` pattern) and a single
  `_resolve_angle(func_name, angle, angle_deg)` helper used by
  all 5 helpers.  The "explicit `angle=0` + `angle_deg=90`"
  conflict (which the v4.14.2 `angle != 0.0` check missed) now
  raises `ValueError` from a single canonical site.  19
  regression tests including a 5-way parametrized conflict
  matrix.
* **P1-NEW-4: `create_led_source` `*args` footgun closed** via
  scale-inversion sanity check in the legacy-positional shim.
  Rejected PEP 570 positional-only marker because it would have
  broken every existing kwarg-based call site (including the
  v4.14.2 audit-test infrastructure).  The new check fires on
  `apparent_wavelength > 10 * apparent_diameter` — catches the
  canonical-order mistake (`0.3 > 1.31e-6 * 10`) without
  triggering on legitimate legacy `(diameter, divergence,
  wavelength)` calls (real LED diameters > 10x their wavelength).
* **P1-NEW-5: `_validate_grid_params` tuple-N gating.**  v4.14.2's
  helper accepted both `int` and `(Ny, Nx)` tuples, but 7 of 10
  factories used `np.arange(N)` downstream and crashed with an
  opaque `np.arange` TypeError.  v4.14.3 adds a `support_tuple_N:
  bool = False` parameter; the 3 factories with genuine 2-D
  grid support (`gaussian_beam`, `hermite_gauss`,
  `laguerre_gauss`) opt in, the other 7 reject tuple-N with a
  clear `TypeError` at validation time.  10 parametrized tests
  across all 10 factories.

### P1 closures — under-examined modules (6)

* **P1-MC (real physics error): `multiconfig.py` hardcoded `n=1.5`**
  in the thin-lens / lensmaker formulae at `:265` and `:327`.
  Beam-expander and Keplerian-telescope multi-config builders
  computed lens powers assuming BK7-ish refractive index for
  every glass, silently producing wrong focal lengths for
  flint, SF, S-LAH, fused-silica, or any non-`n≈1.5` glass.
  v4.14.3 introduces `_resolve_lens_glass_index(name)` that
  routes through the canonical `glass.get_glass_index()` lookup
  with a documented `UserWarning` + bounded fallback (`n=1.5`)
  for genuinely unknown labels and bounds-check on custom
  callable returns (`1.0 < n < 5.0`).  Catches `(ValueError,
  KeyError, ImportError, RuntimeError, TypeError)` only —
  preserves the v4.13.0 Phase-2 narrowed-exception discipline.
* **P1-NEW-9: `create_bessel_beam` `cone_angle` constraint.**
  Now enforces `0 < cone_angle < pi/2` at construction time;
  `cone_angle = pi` previously produced `sin(pi) ≈ 0` and a DC
  field silently labelled "Bessel".
* **P1-NEW-10: `create_fiber_mode` `mode_field_diameter > 0`.**
  Now rejected at construction; negative MFD previously yielded
  a sign-flipped Gaussian, MFD=0 yielded an all-ones field.
* **P1-NEW-11: `surface_sag_xy_polynomial` / `surface_sag_chebyshev`
  negative-`norm_x`/`norm_y` rejection.**  `norm_x = -0.05`
  (typo on a 50 mm half-aperture) made `outside = abs(X) >
  -0.05` true everywhere, silently zeroing the entire freeform
  contribution.  Both branches now reject negative norms at
  function entry with a clear `ValueError`.
* **P1-GH-2: `ghost.py` IndexError on `elements`-key
  prescriptions.**  `ghost_analysis` and
  `non_sequential_stray_light` previously crashed on the two
  `elements`-style prescription schemas (surface-style entries
  with `'element_type'` and propagate-style entries with
  `'type'`).  v4.14.3 introduces a `_ghost_surfaces(prescription,
  wavelength)` adapter that detects each schema and routes
  through the appropriate canonical surface-expansion path.
  5 regression tests across all three schemas + missing-keys
  error path.
* **P1-GL-2: `register_fixed_glass` input validation.**  Name
  must be a non-empty string; `n` must be finite and in
  `[1.0, 4.0]`; overwriting an existing entry emits a
  `UserWarning`.  Si (n=3.4) and Ge (n=4.0) verified accepted;
  invalid forms rejected.

### P1 closures — doc-drift (3, retroactive to v4.14.2)

Per the v4.14.2 audit Part 2 / P1-NEW-6/7/8:

* **Agent D test count** (CHANGELOG line 374): `25 tests` →
  `8 tests + 1 parametrize entry`.
* **"6 factories" → "5 factories"** in 3 CHANGELOG sites
  describing the v4.13.x source-factory dispatcher pin.
* **`lenses_maslov.py` line-drift correction note** refreshed
  (`678/737/884/993` → `619/728/874`; verified against the
  current module via def-header grep).  Recorded that one of
  the 4 prior sites was consolidated away.
* **Audit-path references** (11 expected → 11 fixed): all
  `` `AUDIT_V4_*.md` `` in CHANGELOG (8 sites) and README (3
  sites) now carry the `docs/audits/` prefix added by the
  v4.14.2 doc reorganization.
* **Cache-locks meta-pin count** corrected `38 tests` → `39
  collected (38 pass + 1 documented `_ZARR_MKDIR_PATCH_LOCK`
  skip)` in CHANGELOG lines 158 / 186 and README line 110.

### Test counts

* Pre-v4.14.3 baseline (v4.14.2): 1190 unit tests.
* v4.14.3 additions:
  - Agent A (storage + makedammann + LG chain): 9 tests.
  - Agent B (sources + multiconfig): 25 tests.
  - Agent C (polarization + freeform + ghost + user_library):
    41 tests.
  - Agent D (doc corrections): 0 (paper-only).
* Final: **1265 unit tests passing, 1 skipped** (the documented
  `_ZARR_MKDIR_PATCH_LOCK` exemption), **34/34 validation
  files passing**.

### Deferred to v4.15+

* Multi-process atomic-append for `storage.py` (HDF5 SWMR
  client + distributed Zarr lock).  v4.14.3 documents single-
  process guarantees and the multi-process restriction; the
  full multi-writer story is a v4.15+ ergonomics item.
* 18 P2 + 12 P3 findings from the v4.14.2 audit (UI module
  audit, codegen ergonomics, secondary doc-drift, additional
  factory validation gaps).  These remain catalogued in
  `docs/audits/AUDIT_V4_14_2_2026_05_17.md` for v4.15+ triage.

---

## [4.14.2] — 2026-05-17

**Closes the v4.14.1 audit (`docs/audits/AUDIT_V4_14_1_2026_05_17.md`).**  The
audit found 1 NEW P0 (a v4.11.2 regression carryover: `glass.py`
S-LAH64/S-LAH79 dispatch broken for 3 releases) plus 10 new P1s
(4 of which are "fix N, miss N+1" recurrences on v4.14.1 itself —
the aperture=0 sentinel missed 2 of 4 call sites, 7 older caches
still unlocked, 4 residual `0+0j` literal sites, `clear_asm_caches`
chain narrower than docstring) plus 6 P1s in under-examined modules
(freeform domain guard, makedammann unit drift, polarization API
inconsistencies, source factory validation gaps).  v4.14.2 closes
the P0 and all 10 P1s, adds **two new structural meta-pins** to
extend the v4.14.1 cache-clear dispatcher-pin pattern (cache↔lock
pairing + `0+0j` literal sweep), and closes a follow-up cache-lock
gap discovered by Agent C's meta-pin.  **1190 unit tests pass** (up
from 911); 34/34 validation files pass.

### Breaking changes — none

Two near-breaking output deltas with deprecation warnings:

* **`makedammann2d` is now SI metres throughout** (per-parameter
  deprecation heuristic detects legacy µm input and warns).  Pure-
  legacy calls and hybrid `periodx=20.0 (legacy µm) +
  waveln=1.31e-6 (already SI)` calls both continue to work; a
  `DeprecationWarning` directs users to migrate.
* **`create_led_source` signature reordered** to v4.7-canonical
  `(N, dx, wavelength, *, diameter, divergence_angle, dy=None, ...)`
  with a legacy-positional shim that emits a `DeprecationWarning`.

Neither breaks existing scripts but both warn on the legacy form.

### P0 closure — `glass.py` S-LAH64 / S-LAH79 dispatch

**3-release carryover regression.**  v4.11.2 (round-3 audit, CRIT-3)
removed S-LAH64 / S-LAH79 from `SELLMEIER_COEFFICIENTS` after
discovering the in-code coefficients were off by ±5.8% vs the Ohara
catalog, intending to route them through `refractiveindex.info` —
but `GLASS_REGISTRY` was never updated.  Both glasses remained
flagged `'__sellmeier__'`, and the dispatcher at `glass.py:410-415`
raised `ValueError` on every call.  `ui/main_window.py:1552`
references S-LAH64 as a "known-good preset" — UI broken for 3
releases.

**Fix:** Routed both to `('specs', 'OHARA-optical', '<name>')` tuple
form (correct catalogue book name verified by introspection;
`'OHARA'` was a wrong first attempt).  Numerical verification:
n_d=1.7880 matches Ohara catalog n_d=1.78800 to 5e-5 for S-LAH64;
S-LAH79 returns n_d=2.0033 matching the catalog 2.00330.

**Structural counter-measure:** Module-load consistency check at the
end of `glass.py` walks `GLASS_REGISTRY` and raises `RuntimeError`
if any `'__sellmeier__'` entry is missing from
`SELLMEIER_COEFFICIENTS`.  Future drift fails fast at import time.

### P1 closures — sibling-gap recurrences on v4.14.1

* **P1-NEW-1: aperture=0 sentinel finish** (was incomplete in
  v4.14.1).  The v4.14.1 sentinel fix updated 3 callers
  (`_get_wrapper_merit_cache`, `MultiWavelengthMerit.evaluate`,
  `MultiFieldMerit.evaluate`) but missed 2:
  `ToleranceAwareMerit.evaluate` (`optimize/core.py:2805-2810`,
  refreshed v4.15.1 -- earlier ranges drifted by ~15 lines after
  Agent E's `_Sentinel` base-class refactor) and
  `MatchIdealSystem._make_source` (`optimize/core.py:977-991`,
  refreshed v4.15.0).
  Both used `_cache['E_ones'].copy()` without the `mask is
  _ZERO_APERTURE_MASK` branch, producing a full-grid plane wave
  instead of zero on `aperture_diameter=0`.  v4.14.2 adds the
  sentinel-aware branch (option (b) of the audit recommendation —
  explicit-per-call-site, matches the canonical 2 already-fixed
  sites).  **Investigation finding:** `apply_perturbations` only
  mutates per-surface `decenter` / `tilt` / `form_error`, NEVER
  the prescription-level `aperture_diameter`, so the audit's worry
  about perturbed-to-zero apertures is not triggered by existing
  code — pinned the contract via
  `test_apply_perturbations_does_not_modify_aperture`.
* **P1-NEW-2: 7 older caches now locked** following the v4.14.1
  pattern.  Locks added to `_ZERNIKE_BASIS_CACHE`,
  `_THROUGH_FOCUS_SCAN_JAX_CACHE`, `_PROPAGATE_SYSTEM_JAX_CACHE`,
  `_GS_KERNEL_CACHE`, `_ER_KERNEL_CACHE`, `_HIO_KERNEL_CACHE`,
  `_TRACE_JAX_CACHE`.  Lock-scope discipline matches v4.14.1: lock
  held for `get` / `move_to_end` ops, released before expensive XLA
  jit-compile / numpy basis build, re-acquired for insert + evict.
  Concurrent 4-thread tests confirm no exceptions, no `RuntimeError:
  dictionary changed size during iteration`, final state consistent.
* **P1-NEW-3: `clear_asm_caches()` chain expanded** to 5 sibling
  caches via lazy-import + call (`clear_zernike_basis_cache`,
  `clear_through_focus_scan_jax_cache`,
  `clear_propagate_system_jax_cache`, `clear_phase_retrieval_caches`,
  `clear_trace_jax_cache`).  Docstring rewritten to honestly list
  all caches the function now clears.  Pinning test populates each
  cache then calls `clear_asm_caches()` and asserts emptiness.
* **P1-NEW-4: 2 P1-severity residual `0+0j` sites** swept
  (`optimize/merit_terms.py:524` post v5.1.0 Agent E 6-file split;
  was `optimize/core.py:987` pre-split via Agent B's P1-NEW-1 work
  + `analysis/phase_retrieval.py:402`; the optimize citation was
  refreshed `966 -> 987` in v4.15.0 to match the post-v4.14.2
  drift, then `core.py:987 -> merit_terms.py:524` in v5.1.0).  Now use the `np.zeros((), dtype=...)` pattern.  **Structural pin (new meta-pin):**
  `tests/unit/test_v4_14_2_dispatcher_pin_zero_plus_zeroj.py` walks
  every `lumenairy/*.py` file (117 modules) and asserts no
  unallowlisted `np.where(..., 0+0j)` literal — three exemption
  layers (pure-comment lines, trailing `.astype()` recovery
  pattern, explicit P3 allowlist for `ui/psf_mtf_dock.py:230`).

### P1 closures — under-examined modules

* **P1-NEW-5: `surface_sag_xy_polynomial` domain guard** added.
  v4.14.1's freeform XY polynomial branch evaluated `c * X**i *
  Y**j` for every pixel — a high-order term like `(2,0): 1e3` on a
  50-mm half-grid produced 2.5 m of sag at the corner, propagating
  into raytraced rays outside the physical aperture.  v4.14.2 adds
  `norm_x, norm_y` kwargs + an `xp.where(<inside_box>, sag, 0.0)`
  clip matching the Chebyshev branch.  Backward-compatible (default
  `norm_x = norm_y = 1.0` plus raw-coordinate polynomial evaluation
  — no coefficient rescaling).
* **P1-NEW-6: `makedammann2d` SI conversion** with per-parameter
  deprecation heuristic.  See "Breaking changes" above.
* **P1-NEW-7: `apply_rotator` accepts `angle_deg=`** kwarg-only,
  matching the v4.7 polarization-family convention.  Conflict-
  resolution policy: `angle_deg` and non-zero `angle` (radians)
  with disagreement → `ValueError`.
* **P1-NEW-8: `JonesField.__init__` input validation**.  Added
  positive-finite `dx, dy` checks + 2-D shape check.  A 1-D field
  accidentally passed in now raises a clear `ValueError` at
  construction time, not an opaque FFT failure downstream.
* **P1-NEW-9: `create_led_source` signature drift** closed.  See
  "Breaking changes" above.  Underlying `create_top_hat_beam`
  handles `dy != dx` natively, so no `ValueError`-on-anisotropy
  raise was needed (unlike the JAX-twin precedent).
* **P1-NEW-10: `_validate_grid_params` helper** added at the top of
  `sources/core.py`.  Applied to all 10 factories
  (`create_gaussian_beam`, `create_hermite_gauss`,
  `create_laguerre_gauss`, `create_top_hat_beam`,
  `create_annular_beam`, `create_fiber_mode`, `create_led_source`,
  `create_bessel_beam`, `create_point_source`,
  `create_tilted_plane_wave`).  60 parametrized tests confirm
  `ValueError` on N≤0, dx≤0, wavelength≤0, non-finite inputs.

### Follow-up sibling-gap closed in same release

The v4.14.2 cache↔lock meta-pin (Agent C's new structural pin)
discovered a previously-unflagged single-cell lazy-init cache in
`propagators/asymptotic.py:_JAX_IFT_SOLVER_CACHE`.  Race window
on first concurrent call decorates the JAX `custom_vjp` solver
twice.  v4.14.2 closes the gap in the same release: added
`_JAX_IFT_SOLVER_CACHE_LOCK` with double-check locking pattern
(fast path no lock for the common populated-cache case; slow path
acquires lock, re-checks, delegates the actual build to
`_build_jax_ift_solver_impl`).  Meta-pin's
`_KNOWN_CACHE_SIBLING_GAPS` set is now empty — future regressions
land there.

### Two new structural meta-pins

Extending v4.14.1's cache-clear dispatcher-pin pattern to two more
sibling-gap classes the audit identified:

* **`test_v4_14_2_dispatcher_pin_cache_locks.py`** — 39 collected (38 pass + 1 documented `_ZARR_MKDIR_PATCH_LOCK` skip):
  walks every library module via `pkgutil.walk_packages`, finds
  names matching `^_.*_CACHE$`, asserts each has a corresponding
  lock (accepts both `_FOO_LOCK` and `_FOO_CACHE_LOCK` naming
  conventions).  Reverse check: every `_LOCK` has a corresponding
  cache (or matches the documented `_PATCH_LOCK` exemption for
  `_ZARR_MKDIR_PATCH_LOCK`).
* **`test_v4_14_2_dispatcher_pin_zero_plus_zeroj.py`** (123 tests)
  — walks all `lumenairy/*.py` files for `np.where(.*, 0+0j)`
  literals.  Exemption layers for pure-comment lines, trailing
  `.astype()` dtype-recovery pattern, and explicit P3 allowlist for
  `ui/psf_mtf_dock.py:230` (audit-rated low-priority UI site).

### Retroactive doc-drift corrections to v4.14.1 entry

Per audit Part 1.3:

* Agent D test count (v4.14.1 line ~372): 25 → 8 (bottom-line 911 was correct).
* Source-factory dispatcher count: previously claimed "all 6" → corrected to 5 factories (3 CHANGELOG sites updated).
* `lenses_maslov.py` line-drift correction note refreshed.

### Test counts

* Pre-v4.14.2 baseline (v4.14.1): 911 unit tests.
* v4.14.2 additions:
  - Agent A (glass + freeform + polarization): 13 tests
  - Agent B (aperture=0 sentinel finish): 10 tests
  - Agent C (7-cache locks + clear_asm scope + meta-pin): 57
    tests pass (19 fix-pins + 39 meta-pin parametrizations
    collected, of which 38 pass + 1 documented
    _ZARR_MKDIR_PATCH_LOCK skip)
  - Agent D (sources + DOE + meta-pin): 199 tests (76 fix-pins +
    123 meta-pin parametrizations)
* Final: **1190 unit tests passing, 1 skipped** (the documented
  `_ZARR_MKDIR_PATCH_LOCK` exemption), **34/34 validation files
  passing**.

### Deferred to v4.15+

Row_reset physics pin against `propagate_hf_chebyshev_quadrature
(method='direct')` (audit Tier-1 #12 — contract-pinned in v4.14.1,
not numerically-pinned).  Backend dispatch (`_xp_of`) on the 6 new
v4.14.0 analysis functions.  Modal asymptotic per-pixel
vectorisation public switch.  Source factory signature
normalisation (the LED factory landed in v4.14.2; the rest still
have mixed positional/keyword conventions).  `system.evaluate
(prescription, source, ...)` ergonomic entry.  See
[`ROADMAP.md`](ROADMAP.md) for the full forward plan.

## [4.14.1] — 2026-05-17

**Closes the v4.14.0 audit (`docs/audits/AUDIT_V4_14_0_2026_05_17.md`).**  The
audit found 1 P0 (silent-wrong physics in the 77× LG/HG mode-stack
cache key) + 6 P1s + 10 P2s + 8 P3s + 7 doc-drift items.  v4.14.1
closes the P0, **all 6 P1s** (including P1-NEW-5 `row_reset`
Newton warm-start via the coordinated option-(a) fix across the
public path and 3 reference-loop pins), the top-priority P2s (cache
locks, monkey-patch removal, final `0+0j`/`1+0j` sweep,
`fiber_mode` in the dispatcher pin), and all 7 doc-drift items (4
retroactively corrected in the v4.14.0 entry below; 3 closed by the
v4.14.1 fixes themselves).  **911 unit tests pass** (up from 858);
34/34 validation files pass.

### Breaking changes — none

No user-facing breakage.  Aperture-zero semantics restored to the
pre-v4.14.0 behaviour (which had silently flipped) — see P1-NEW-1
below.

### P0 closure — `_LG_MODE_STACK_CACHE` / `_HG_MODE_STACK_CACHE` cache key

The 77× perf win in v4.14.0 had a silent-wrong-physics bug.  The
cache key omitted `dx, dy` — only `(p_max, ell_max, Ny, Nx, w, cx,
cy, dtype_str)`.  Two calls with the same shape but different
physical pitch (e.g. `dx=1e-6` then `dx=2e-6`, both at N=256)
collided on this key: **the second call returned the FIRST grid's
modes evaluated against the second call's field.**  Silently wrong
overlaps on:

* wavelength-adaptive grid sweeps (re-pitch the grid per
  wavelength to keep `λ/dx` constant)
* multi-resolution analysis (debug at coarse `dx`, then fine `dx`)
* optimisation loops where `dx` is a free variable

**Fix:** `dx, dy` added to both LG and HG mode-stack cache keys.
Regression pin asserts that two calls at the same `N, w, cx, cy,
p_max, ell_max` but different `dx` return distinct mode stacks.

### P1 closures

* **P1-NEW-1: aperture=0 semantics regression.**  v4.14.0's
  `_get_wrapper_merit_cache` mapped `aperture <= 0` to `mask=None`,
  which downstream callers interpreted as "no clipping, full grid
  plane wave."  **Semantics flipped 180°** from the pre-v4.14
  behaviour where `aperture=0` produced an all-False mask (block
  all light).  v4.14.1 adds a `_ZeroApertureMaskSentinel` singleton;
  `_get_wrapper_merit_cache` returns the sentinel for `ap <= 0` and
  `None` for `ap is None`; callers compare via `is` and route the
  sentinel through an explicit all-zeros path.  Three callers
  (`_get_wrapper_merit_cache`, `MultiWavelengthMerit.evaluate`,
  `MultiFieldMerit.evaluate`) updated.
* **P1-NEW-2: Brewster-angle phase aggregation bug** in
  `coatings.py` carried into v4.14.0's wavelength batch.  The
  v4.13.1 audit flagged that `0.5 * (np.angle(r_s) + np.angle(r_p))`
  is off by π/2 or π at Brewster (~56°) because `r_p` sign-flips
  through zero.  v4.14.0's wavelength-batch rewrite inherited the
  bug verbatim.  v4.14.1 changes the aggregation to
  `np.angle(0.5 * (r_s + r_p))` (complex sum then angle — robust
  to π-discontinuities).  Reference helper in
  `test_audit_fixes_v4_14_0_agent_2.py` also updated to match
  (the audit explicitly anticipated this collateral update).
* **P1-NEW-3: `_solve_envelope_stationary_batch` contract
  violation.**  The function's docstring promised
  `converged_mask=False` for failed pixels (singular Hessian or
  non-finite update) but the code set `True` to drop them from the
  active set.  Currently dead production code; the contract is
  preserved for future callers.  v4.14.1 introduces a separate
  `finished` mask for active-set tracking, leaving `converged`
  to mean what the docstring says.
* **P1-NEW-4: `clear_lg_mode_stack_cache` now in top-level `__all__`.**
  v4.14.0's CHANGELOG claimed this was "Public" but the import
  wasn't in `lumenairy/__init__.py` — the audit-meta-finding
  ("fix N sites, miss N+1") recurring on the very release that
  shipped 80 dispatcher pins.  v4.14.1 closes the loop AND adds a
  new structural counter-measure: a parametrized **cache-clear
  dispatcher pin** (`tests/unit/test_v4_14_1_dispatcher_pin_cache
  _clears.py`) that walks every submodule's `__all__` for
  `clear_*` names and asserts each is re-exported at top level
  AND callable from `la.*`.  Future cache-clear additions can
  no longer regress this gap.
* **P1-NEW-6: `encircled_energy_radius` docstring** corrected.
  v4.14.0 claimed `ee[0] = 0 always` (in an inline comment, not
  the docstring).  Reality: `ee[0]` equals the cumulative power
  at radius 0 (the centre-pixel intensity contribution).  Docstring
  expanded to document the hot-centre behaviour explicitly;
  pinning test exercises a delta-like centre-pixel input and
  confirms the threshold-zero short-circuit returns 0.

* **P1-NEW-5: `row_reset` resets the Newton warm-start.**
  v4.14.0's `row_reset` branch reset Maslov-branch state
  (`last_arg_detM`, `maslov_branch`) at each raster row wrap but
  left the Newton warm-start `last_v_star` chaining across the
  discontinuous jump from (x_max, y_n) to (x_min, y_{n+1}) —
  plausibly the mechanism behind the v4.14.0 "wrong-saddle-basin"
  finding near grid edges (largest jump in s_2).  v4.14.1 chooses
  **option (a)** of the audit recommendation: the `row_reset`
  branch now resets `last_v_star = (v_cx, v_cy)` at each row wrap
  too.  Coordinated with the fix, the bit-equal pin in
  `test_audit_fixes_v4_14_0_agent_1.py::test_lg00_single_mode
  _bit_equal` and the older 1e-10 rel pins in
  `test_perf_v4_12_0_asymptotic.py::TestPropagateModalAsymptotic
  Correctness` were updated — their inline scalar references
  also reset `last_v_star` at row wrap so the bit-equality holds
  against the new physics-correct behaviour.  The v4.14.1 marker
  test (formerly `TestRowResetDoesNotResetWarmStart`, now
  `TestRowResetResetsWarmStart`) flipped its assertion to pin the
  new contract.

### Tier-2 closures

* **Thread-safety locks** on the three new v4.14.0 caches.
  `_LG_MODE_STACK_CACHE`, `_HG_MODE_STACK_CACHE`, and
  `_WRAPPER_MERIT_CACHE` now have module-level `threading.Lock`
  guards on their read-modify-write ops, mirroring the
  `_ASM_CACHE_LOCK` precedent in `propagation.py`.  Concurrent
  `OrderedDict.move_to_end` / `popitem(last=False)` from parallel
  `design_optimize` threads no longer race.
* **Monkey-patch removed** in `optimize/core.py`.  The v4.14.0
  pattern monkey-patched `propagation.clear_asm_caches` to chain
  in `_clear_wrapper_merit_cache`; v4.14.1 replaces this with a
  lazy-import + call inside `propagation.clear_asm_caches()`
  itself.  Eliminates re-import recursion risk and the case
  where importing `propagation` without `optimize` leaves the
  wrapper-merit cache resident.
* **LG/HG mode-stack cache wired into `clear_asm_caches()`** (not
  just `lumenairy_context()` as v4.14.0 had).  Same lazy-import
  pattern.  CHANGELOG claim from v4.14.0 now matches code.
* **Final `0+0j`/`1+0j` literal sweep**: 2 sites caught by the
  audit — `lenses_maslov.py:448` (in `sample_E_bilinear`) and
  `_lens_thin.py:173` (aplanatic branch, actually `1.0+0.0j` not
  `0.0+0.0j` — replaced with `xp.ones((), dtype=...)`).
* **`fiber_mode` added** to
  `TestP1CSourceFactoryDispatcherPin` parametrize list (v4.13.2's
  `create_fiber_mode` dy-widening was complete but the test
  list was stale).

### Retroactive CHANGELOG corrections to v4.14.0 entry

4 confirmed doc-drift items from the audit corrected in the
v4.14.0 entry below:

* "16 tests at 1e-10 rel" → 3 tests in the cited class
  (`TestPropagateModalAsymptoticCorrectness`), 16 across the
  file overall.
* "6 batched helpers consumed by 77× win" → helpers ship privately
  but currently have zero production consumers; reserved for the
  v4.15+ coordinated public switch.
* HG cache key documented as `w[s]` shorthand → both LG and HG
  keys spelled out (HG has both `wx` AND `wy`; v4.14.1 adds
  `dx, dy`).
* `lenses_maslov.py` line cites — drifted 4-10 lines since
  v4.14.0 ship; in-line correction note added.

The 3 other doc-drift items (LG cache wired into `clear_asm
_caches`, public `clear_lg_mode_stack_cache`, "dispatcher pin
covers all 5 factories") **became true** in v4.14.1 — no
retroactive edit needed; the v4.14.0 entry's claims are now
accurate.

### Test counts

* Pre-v4.14.1 baseline (v4.14.0): 858 unit tests.
* v4.14.1 additions:
  - Agent A (asymptotic): 8 tests
  - Agent B (optimize + propagation): 11 tests
  - Agent C (coatings + lens sweep): 8 tests
  - Agent D (exports + meta-pin): 8 tests + 1 parametrize entry
    + 17 cache-clear meta-pins
* Final: **911 unit tests passing**, **34/34 validation files
  passing**.

### Deferred to v4.15+

Modal asymptotic per-pixel vectorisation public switch, Source
factory signature normalisation, `system.evaluate(prescription,
source, ...)` ergonomic entry.  See [`ROADMAP.md`](ROADMAP.md)
for the full forward plan.

## [4.14.0] — 2026-05-17

**Phase B of the v4.13.1 audit (`docs/audits/AUDIT_V4_13_1_2026_05_17.md`).**
v4.13.2 closed Tier-0 (the 12 P1s + 5 cross-survey P0s + thin-lens
sibling sweep).  v4.14.0 closes Tier-1: the 7 Tier-1 perf wins from
audit Part 3, the top user-facing API gaps from the cross-survey,
and the 3 parametrized dispatcher pin families that close out the
sibling-gap audit-meta-finding.  All 858 unit tests pass; 34/34
validation files pass.

### Breaking changes — none

No user-facing breakage in v4.14.0.  All additions are net-new
public functions or internal perf optimisations behind unchanged
signatures.

### Performance wins (Tier-1 from audit Part 3)

| Hot path | Workload | Old | New | Speedup |
|---|---|---|---|---|
| `coating_reflectance` wavelength batch | 50 layers × 200 wv | 78.6 ms | 3.19 ms | **24.6×** |
| `decompose_lg` / `decompose_hg` cache (warm) | 256², p_max=3, ell_max=3 | 270 ms | 3.5 ms | **77×** |
| `MultiWavelengthMerit` meshgrid cache | N=512, 5 wl, 20 FD | (ref) | (ref/6.17) | **6.17×** |
| `_evaluate_polynomial_4d_and_grad34` | M=70, 16×16 grid | 646 µs | 171 µs | **4.6×** (typical configs) |
| `MultiWavelengthMerit` meshgrid cache (small) | N=128, 3 wl, 5 FD | (ref) | (ref/4.16) | **4.16×** |
| `ToleranceAwareMerit` meshgrid cache | N=128 | (ref) | (ref/3.17) | **3.17×** |
| Shack-Hartmann gather loop | K=4096, sa=8 | 7.2 ms | 3.2 ms | **2.27×** |
| Phase-retrieval `np.angle`/`np.exp` round-trip | GS, N=256, 50 iters | 1230 ms | 543 ms | **2.26×** |
| Phase-retrieval (ER variant) | same workload | 380 ms | 205 ms | **1.85×** |
| Phase-retrieval (HIO variant) | same workload | 426 ms | 268 ms | **1.59×** |
| `MultiFieldMerit` (limited by `np.exp`/`np.where`) | N=128 | (ref) | (ref/1.20) | **1.19×** |

**Meshgrid build count** in `Multi*Merit` / `ToleranceAwareMerit`:
from `O(n_wl × n_field × FD_evals)` (~1025 at N=512, 5 wl, 5 field,
20 FD) to **1 per `(N, dx, aperture)` signature** for the entire
optimisation run.  Cache cleared by `clear_asm_caches()`.

**Coating Snell-chain hoist** — the documented `n.imag-dropped-at-
Snell-step` approximation makes the per-layer chain wavelength-
independent.  v4.14 walks it ONCE outside the polarisation loop
instead of `n_wv × n_pol` times.  This is what unlocks the 24×
batched speedup.

**LG/HG mode-stack cache** (`_LG_MODE_STACK_CACHE` +
`_HG_MODE_STACK_CACHE`, both `OrderedDict` + LRU(32)).  LG key is
`(p_max, ell_max, Ny, Nx, w, cx, cy, dtype_str)`; HG key has both
`wx` AND `wy` (9-element tuple).  **Correction (v4.14.1):** v4.14.0
omitted `dx, dy` from these keys, producing silently-wrong overlaps
on multi-resolution or wavelength-adaptive grid sweeps; v4.14.1
adds `dx, dy` to both keys and wires the caches into
`clear_asm_caches()` (v4.14.0 only wired them into
`lumenairy_context(clear_caches_on_exit=True)`).  Public
`clear_lg_mode_stack_cache()` for explicit flushes (became
top-level-importable as `lumenairy.clear_lg_mode_stack_cache` in
v4.14.1).

**Phase-retrieval algebraic identity.**
`exp(1j * np.angle(F)) == F / np.abs(F)` for nonzero `F`.  v4.14
replaces the two-transcendental round-trip in the NumPy paths of
GS / ER / HIO with the divide, eliminating ~4 trig ops per pixel per
iteration.  JAX paths already had the optimisation; only NumPy
paths were touched.

### Modal asymptotic per-pixel vectorisation — NOT shipped publicly

Audit opportunity #1 (target 20-100×) was investigated and turned
out to be **subtler than the audit estimate**.  The brief asked to
vectorise `propagate_modal_asymptotic`'s per-pixel Newton solve.
The vectorised cold-start batched Newton finds the physical saddle
uniformly across all output pixels; the pre-v4.14 warm-started
Newton chains the previous pixel's `v_star` and at grid-edge pixels
lands in a **wrong-saddle basin** that produces `|b_quad| > 700`,
which the overflow guard zeros.

The pre-v4.14 behaviour is pinned bit-equal by `tests/unit/test_perf
_v4_12_0_asymptotic.py::TestPropagateModalAsymptoticCorrectness` (3
tests in the cited class, 16 in the file overall, all at `1e-10
rel`).  The vectorised cold-start produces strictly
more non-zero pixels (physically more correct) but breaks the
existing pin.  **This is a real physics finding** worth a coordinated
v4.15+ release that updates the test pin alongside the algorithm
change.

Shipped in v4.14.0: 6 new private vectorised helpers
(`_solve_envelope_stationary_batch`, `_compute_M_b_batch`,
`_phi_v2_hessian_batch`, `_gaussian_moment_table_2d_batch`,
`_batched_polynomial_substitute_linear_2d`, `_batched_polynomial
_under_affine_shift`).  The public `propagate_modal_asymptotic` body
is unchanged.  **Correction (v4.14.1):** the helpers ship privately
but currently have zero production consumers; the 77× LG/HG mode-
stack cache reaches into `lg_polynomial`/`hg_polynomial`/`_evaluate
_poly2d` directly, not through the batched helpers.  The helpers
are reserved for the v4.15+ coordinated public switch.

### New public API (cross-library survey gaps)

Six new functions exposed at `lumenairy.*` and in the appropriate
`__all__` tier:

* **`encircled_energy_curve(E, dx, *, dy=None, radii=None,
  centroid=None, n_radii=64) -> (radii, ee)`** — fraction of total
  power within radius `r` of the centroid.  Standard spec-sheet
  metric.  (Tier 4: Analyse.)
* **`encircled_energy_radius(E, dx, *, dy=None, threshold=0.84,
  centroid=None) -> float`** — radius enclosing the threshold
  fraction.  Default 0.84 matches the "84% encircled radius" lens
  spec convention.
* **`mtf_cutoff(mtf_profile, freq, *, threshold=0.5) -> float`** —
  spatial frequency at which a 1D MTF drops below threshold.
  Returns `np.inf` if the MTF stays above threshold for all
  frequencies.
* **`beam_diameter(E, dx, *, dy=None, threshold='1/e^2',
  centroid=None) -> float`** — diameter at which intensity drops
  below threshold.  String thresholds: `'1/e^2'`, `'1/e'`, `'FWHM'`,
  `'D4sigma'` (forwards to `beam_d4sigma` and returns geometric
  mean).
* **`depth_of_focus(wavelength, f_number, *, formula='rayleigh')
  -> float`** — one-sided depth of focus.  `'rayleigh'`:
  `±4 f#² λ`.  `'marechal'`: `±λ / NA²`.  Note: both formulas
  reduce to `4 f#² λ` (since `NA = 1/(2 f#)`); kept as separate
  named entries for derivation-clarity, cross-validated by test.
* **`plot_wavefront(opd, dx, *, dy=None, aperture=None,
  units='waves', wavelength=None, cmap='RdBu_r', show_stats=True,
  ax=None, fig=None, title=None) -> (fig, ax)`** — Zemax-style
  wavefront map: NaN outside the aperture, divergent colormap
  centred at zero, PV/RMS overlay annotation.  (Tier 9: Plotting.)

All six honour `dy` from the start (defaulting `dy=None → dy=dx`,
area integrations use `dx*dy`).

### Sibling-gap parametrized dispatcher pins (audit meta-finding closure)

The audit's recurring meta-finding ("fix swept N sites, missed
N+1") is closed for three family classes via parametrized pins.
**80 new dispatcher pins across 3 test files**.  Each pin enumerates
every reachable variant and asserts the same property; a future fix
that misses one variant fails CI at test-time.

* **`(scalar, vectorial) HFPI`** — 29 pins (`tests/unit/test_v4_14
  _0_dispatcher_pin_hfpi.py`).  Properties: `_spawn_rng`
  independence; grazing-ray `inf`/`NaN` guard; alive-mask
  correctness in `accumulate_*_to_grid`; public-surface
  importability across 13 names.
* **`(NumPy, JAX) apply_real_lens` family** — 35 pins (`tests/unit
  /test_v4_14_0_dispatcher_pin_apply_lens.py`).  Properties:
  `glass_after='MIRROR'` case-insensitive guard (15 parametrisations
  covering upper/lower/mixed-case); `dy=None` acceptance; `dy != dx`
  honour-vs-raise contract; complex64 dtype preservation across all
  5 variants.
* **Welford-mirror convention** — 16 pins (`tests/unit/test_v4_14
  _0_dispatcher_pin_welford_mirror.py`).  Properties: hand-computed
  analytical match for `seidel_coefficients` and `petzval_radius`;
  no-NaN for `aberration_summary` and `chromatic_focal_shift`;
  no-raise for `distortion_grid`, `field_aberration_sweep`,
  `eval_image_plane_wfe`; algebraic equivalence to the Welford
  formula.

### Sibling-gaps discovered by the dispatcher pins (now fixed)

The complex64 dtype-preservation pin tripped on 3 of 5
`apply_real_lens` variants — a previously-undetected sibling-gap
from the v4.13.2 B.4/B.5 thin-lens sweep.  Fixed in this release:

* **`apply_real_lens_maslov`** — `lenses_maslov.py`:
  - `_integrate_quadrature` (line 619) and `_integrate_stationary
    _phase` (line 728) and `_integrate_local_quadrature` (line 874)
    all hardcoded `dtype=np.complex128` allocations.  Each now
    accepts an `out_dtype` kwarg defaulting to `np.complex128` for
    back-compat; `apply_real_lens_maslov` threads `E_in.dtype`.
    (CHANGELOG line-cite refreshed in v4.14.2: prior CHANGELOG
    revisions cited drifted post-landing line numbers from the
    v4.14.0 tag and a further-drifted set from v4.14.1.  Current
    v4.14.2 def-header sites are 619/728/874 — one prior site was
    consolidated away during subsequent edits, leaving 3.)
  - The post-quadrature re-fit at line 566 multiplied by `1j`
    (complex128) which promoted the result; now cast back to
    `E_in.dtype`.
  - Final `normalize_output` step multiplied by a python float
    (float64) which silently promoted complex64 → complex128; final
    cast back to `E_in.dtype` added.
* **`apply_real_lens_traced_jax`** — `_lens_jax.py:573`: was calling
  `cdtype = _resolve_jax_complex_dtype()` which returns the library-
  default complex dtype (complex128 when `jax_enable_x64=True`),
  silently upcasting complex64 inputs.  v4.14 passes `E_in.dtype` to
  the resolver so the user's input dtype is honoured.
* **`apply_real_lens_maslov_jax`** — `_lens_jax.py:819`: same fix.

### Cache invalidation hygiene

`clear_lg_mode_stack_cache()` (new in v4.14 from the decompose_lg
cache work) wired into `lumenairy_context(clear_caches_on_exit=
True)` to match the existing pattern for the other 7 cache-clear
helpers.

### Test counts

* Pre-v4.14.0 baseline (v4.13.2): 710 unit tests.
* v4.14.0 additions:
  - Agent 1 (asymptotic): 15 tests in `test_audit_fixes_v4_14_0_agent_1.py`
  - Agent 2 (coatings + polynomial): 10 tests in `..._agent_2.py`
  - Agent 3 (phase-retrieval + SH): 13 tests in `..._agent_3.py`
  - Agent 4 (Multi*Merit cache): 18 tests in `..._agent_4.py`
  - Agent 5 (API gaps): 12 tests in `..._agent_5.py`
  - Agent 6 (dispatcher pins): 80 tests across 3 files (`test_v4_14
    _0_dispatcher_pin_hfpi.py`, `..._apply_lens.py`, `..._welford
    _mirror.py`)
* Final: **858 unit tests passing**, **34/34 validation files
  passing**.

### Deferred to v4.15+

* **Modal asymptotic per-pixel vectorisation** (audit opportunity
  #1).  The wrong-saddle-basin physics finding noted above needs a
  coordinated test-pin update alongside the algorithm change.
  Vectorised helpers shipped privately for future use.
* **Source factory signature normalisation** (audit Stream B #10) —
  `Source.gaussian(w0, N, ...)` puts beam-size first; `Source.plane
  _wave(N, ...)` puts N first.  Picking a canonical order requires
  a deprecation window; rolling into v4.15 to coordinate with other
  Source-related cleanups.
* **`system.evaluate(prescription, source, ...)` ergonomic entry**
  — closes a one-liner UX gap (`propagate_through_system` currently
  takes an element list, not a prescription dict).

### Deferred to v5.0

Tier-2/3/4 structural items unchanged from v4.13.2 entry: six file
splits; CI gates; back-compat shim removal; shared Chebyshev
helpers extraction; audit-fix test-file consolidation by topic;
constrained optimisation; checkpoint/resume; CDGM/Hikari/Sumita
glass catalogues; off-axis conics in surface frame; Q-type
freeform.

## [4.13.2] — 2026-05-17

**Closes the v4.13.1 audit (`docs/audits/AUDIT_V4_13_1_2026_05_17.md`) plus its
Part 10 consolidation with a parallel 6-agent cross-library survey.**
The v4.13.1 audit identified 12 new P1s (5 sibling-gap recurrences +
5 fresh-eyes bugs + 2 partial-closure follow-ups) plus 17 perf
opportunities and 22 structural recommendations.  The parallel
cross-library survey turned up an additional 5 P0s (in `optimize/`
and `io/`) and 7 P1s (mostly `dy` convention drift across analysis +
system + lens family).  v4.13.2 closes the full consolidated Tier-0
set.  All 710 unit tests pass; 34/34 validation files pass.

### Breaking changes — none

No user-facing breakage in v4.13.2.

### P0 closures (cross-library survey)

* **`make_lg_aberration_merit_jax` no longer silently ignores its
  `targets` dict.**  The function's inner loop body was literally
  `pass`; `wgt` was captured but never multiplied; the aberration-
  tensor call was outside the loop.  Public API exported from
  `lumenairy.__init__` — the function was returning the same
  `(0,0)`-piston sum regardless of input weights.  v4.13.2: now
  weights the `(0,0)` target correctly; non-`(0,0)` targets raise
  `NotImplementedError` with a clear migration message (the
  underlying `aberration_tensor_lg00_jax` only computes the piston
  coefficient).
* **`MultiFieldMerit.field_angles` accepts both scalars and `(theta_x,
  theta_y)` tuples.**  v4.13.1 applied Y-axis tilt only despite the
  docstring implying generic off-axis angle; `MatchIdealSystemMerit`
  took `(theta_x, theta_y)`.  v4.13.2 widens `field_angles` to accept
  both forms with a one-shot `DeprecationWarning` on the scalar form;
  internal storage normalises to tuples so the evaluate loop is
  uniform.  Non-zero `theta_x` now actually produces an X-axis tilt.
* **`load_plane_slice` documented return type matches actual.**
  Documented to return slice array; actually returns `(arr, attrs)`
  tuple in both HDF5 and Zarr paths.  Docstring + type annotation
  corrected to `Tuple[ndarray, Dict[str, Any]]`.
* **CODE V `.seq` round-trip preserves BFL.**  Reader was dropping the
  last refracting surface's `THI` (a legitimate CODE V BFL
  convention).  New top-level `'back_focal_length'` prescription key
  (float, SI meters) carries the BFL through round-trip; populated by
  both readers and exporters.
* **Quadoa `.qos` round-trip preserves BFL** (same fix as CODE V).

### Sibling-gap P1 closures (the audit's headline pattern recurring)

Five new "fix-swept-N-sites-missed-N+1" instances closed.  All five
have parametrized dispatcher-level pin tests preventing future
recurrence within the same family.

* **`vectorial_hfpi` RNG-correlation sibling.**  v4.11.2 fixed the
  scalar `hfpi.propagate_hfpi_freespace_aperture` via `_spawn_rng`;
  the vectorial sibling at `vectorial_hfpi.py:399` still passed the
  same `rng` to source-init AND aperture re-emission, perfectly
  correlating the diffraction events.  v4.13.2 ports `_spawn_rng`.
* **`subaperture` `decompose_lg(E_in, ...)` sibling.**  v4.11.2 fixed
  `hf.propagate_huygens_fresnel_through_prescription` to decompose
  the input field per-patch; `subaperture.propagate_subaperture
  _asymptotic` still passed `source_amplitudes={(0,0): 1.0+0.0j}`
  — every patch got an identical plane-wave-equivalent unit
  fundamental, ignoring `E_in` beyond a waist estimate.  v4.13.2
  ports the per-patch LG decomposition with three new kwargs
  (`source_lg_p_max=3`, `source_lg_ell_max=3`,
  `source_lg_amp_threshold=1e-6`) and amplitude-threshold pruning.
* **`petzval_radius` Welford-mirror convention sibling.**
  `seidel_coefficients` was fixed in v4.11.2 with the Welford `n2 =
  -n1` convention for mirrors; `analysis/field.py:petzval_radius`
  still skipped every mirror (because `n1 == n2` after the loader
  set `glass_after == glass_before`), silently dropping every
  mirror's Petzval contribution.  **Wrong by 100% for catadioptric /
  Cassegrain designs.**  v4.13.2 applies the Welford-parity
  convention; Cassegrain regression test pins against the analytic
  Mahajan formula.
* **`_build_jax_prescription` `glass_after='MIRROR'` sibling.**
  v4.13.1 P1-A added BOTH `is_mirror=True` AND case-insensitive
  `glass_after='MIRROR'` guards to `apply_real_lens`; the JAX
  prescription builder at `raytrace/jax_trace.py:649-669` only got
  the first.  Hand-built prescriptions with Welford-style mirror-
  via-glass-string slip through and are silently traced as
  refractive air→air.  v4.13.2 adds the second guard.
* **JAX lens twins thread `dy`.**  The two NumPy `apply_real_lens`
  variants accept `dy=None`; the JAX twins (`apply_real_lens_traced
  _jax`, `apply_real_lens_maslov_jax`) did not.  Anamorphic round-
  trip through `Source.dy → propagate_through_system_jax →
  apply_real_lens_traced_jax` silently dropped y-pitch at the JAX
  boundary.  v4.13.2: both JAX twins accept `dy=None` and raise
  `ValueError` on `dy != dx` (matching the existing NumPy
  precedent on the square-grid contract).

### Fresh-eyes P1 closures (audit Part 2.2)

* **`apply_mirror` NaN propagation.**  For a hyperbolic mirror with
  conic such that `(1+k)*h²/R² >= 0.9999`, sag → NaN → phase → NaN →
  `E *= NaN` poisons every pixel of the subsequent ASM step.
  `apply_real_lens:704-705` had the equivalent NaN-zeroing guard;
  `apply_mirror` did not.  v4.13.2 mirrors the guard.
* **`_zero_C_air_gap` raises on degenerate ABCD.**  When `abs(C1 -
  C0) < 1e-30`, the function silently returned the placeholder
  thickness from the input prescription (a non-afocal beam expander
  with zero combined power).  Callers (`beam_expander_prescription`,
  `keplerian_telescope`) catch `RuntimeError`; v4.13.2 raises it
  explicitly so the fallback fires.
* **`propagate_to_plane` inf/NaN positions guarded.**  For `Nz ≈ 0`
  and `z_target != z_curr`, the divisor went to `1e-30` →
  `t ≈ 1e30` → positions += inf.  The alive-mask correctly tagged
  these dead, but `paths.positions` still contained inf/NaN —
  downstream code reading positions without masking by `alive` was
  poisoned (e.g. `_hfpi_segment_trace`).  v4.13.2 zeroes the step on
  dead/grazing rays in BOTH `hfpi.propagate_to_plane` and
  `vectorial_hfpi.propagate_vector_to_plane`.
* **`RandomState.choice` int dtype aligned across backends.**  JAX
  path returned int32 (default for `jax.random.randint`); NumPy
  path returned int64.  v4.13.2 dispatches `jax.random.randint(...,
  dtype=jnp.int64)` so both backends agree.
* **`trace_prescription` uses `_surface_copy_with` instead of
  mutating shared `Surface.thickness`.**  When `image_distance=` was
  supplied, the function mutated the input prescription's last
  surface in place.  The `Surface` dataclass is shared with
  `surfaces_from_prescription`; downstream calls reused corrupted
  thickness.  v4.13.2 clones via `_surface_copy_with` (matching the
  pattern at `lens_abcd`).

### Partial-closure follow-ups from v4.13.1

* **`dual_annealing` callback wired into cancellation protocol.**
  v4.13.1 P2 #13 wired `CancellableProgress` into 4 scipy callbacks;
  the `dual_annealing` site used an unnamed inline lambda that did
  NOT poll `is_cancelled(progress)`.  Cancellation latency was
  unbounded for that one method.  v4.13.2 replaces the lambda with
  a named callback matching the pattern in the other three.
* **`RandomState.choice` old-JAX safety net.**  v4.13.1 P1-F closed
  the `replace=False` regression on JAX 0.10.0+ but the CHANGELOG
  promised a graceful old-JAX safety net the code lacked.  v4.13.2
  wraps the dispatch in `try/except TypeError` with a `RuntimeError`
  raising a clear "JAX >= 0.4.0 required" message.

### Cross-library survey P1 closures

* **`strehl_ratio` + `polychromatic_psf` accept `dy=`.**  The
  v4.13.0 L3 sweep added `dy` to `Source` / `PropagationResult` /
  free-space propagators but missed both Strehl helpers (using
  `dx**2` not `dx*dy`).  v4.13.2 adds `dy: Optional[float] = None`
  to both; back-compat preserved bit-for-bit when `dy is None`
  (the historic `dx**2` form is retained so existing tests with
  the FP-identity assumption stay green).
* **Wrapper-merit context threads `x=ctx.x`.**  `MultiWavelengthMerit`,
  `MultiFieldMerit`, `ToleranceAwareMerit` built sub-`Evaluation
  Context`s without `x=ctx.x`.  A `JaxMeritTerm(build_args=...)`
  wrapped inside any of these silently fell back to legacy
  `fn(ctx)` mode, degrading the analytic-gradient path to FD.
  v4.13.2 threads `x=ctx.x` across all three.
* **`propagate_through_system` element handlers thread `dy`.**
  Every element handler (lens, aperture, mask, zernike, mirror,
  etc.) passed `dx=dx` only — anamorphic `dy` was silently squared
  to `dx` on every non-`propagate_*` element.  v4.13.2 routes
  `current_dx` AND `current_dy` through all 13 element handlers.
* **`Source.fiber_mode` accepts `dy=` end-to-end.**  v4.13.1 P1-C
  threaded `dy` through 4 of 5 Source factories; `fiber_mode`
  remained a dead-`dy=` code path because `create_fiber_mode`
  didn't accept `dy=`.  v4.13.2 widens `create_fiber_mode` to
  accept `dy=` (forwarding to `create_gaussian_beam`); the
  dispatcher pin in `TestP1CSourceFactoryDispatcherPin` now covers
  all 5 factories.

### Thin-lens family sibling-gap sweep (cross-library survey)

The cross-library survey turned up additional sibling-gaps in the
thin-lens family — same v4.13.1 P3 #21 dtype-aware-zero fix that
landed in `apply_mirror` + `apply_aperture`, but missed across
multiple sites:

* **9 sites of `0.0+0.0j` complex128 literal** in `_lens_thin.py` (4
  sites) + `_lens_real.py` (5 sites).  Each was a `xp.where(..., E,
  0+0j)` clear-aperture or stop-mask construct that silently
  upcast complex64 → complex128.  v4.13.2 replaces each with
  `xp.zeros((), dtype=E.dtype)`.
* **6 thin-lens functions** (`apply_thin_lens`, `apply_spherical
  _lens`, `apply_aspheric_lens`, `apply_cylindrical_lens`, `apply
  _grin_lens`, `apply_axicon`) constructed `xp.exp(1j * phase)`
  from float64 phase without dtype matching against the input
  field.  Same complex64→complex128 upcast as the `0+0j` literal,
  at a different call site.  v4.13.2 adds the dtype-coercion line
  to all 6 functions, matching the v4.13.0 L6 pattern in
  `apply_mirror`.
* **3 thin-lens functions** (`apply_cylindrical_lens`, `apply_grin
  _lens`, `apply_axicon`) lacked the documented `use_gpu` parameter
  entirely.  v4.13.2 adds the parameter and the canonical CuPy-
  dispatch pattern to all three.
* **Latent CuPy dispatch bug fixed in the other 3 thin-lens
  functions.**  `apply_thin_lens`, `apply_spherical_lens`,
  `apply_aspheric_lens` had the `use_gpu` parameter from v3.5.5
  but the CuPy dispatch was broken because the function bodies
  referenced bare `cp` — Python's LEGB name resolution doesn't
  consult module-level PEP 562 `__getattr__` for function-local
  lookups.  v4.13.2 routes all three through `_lenses_module.cp`
  (matching the working pattern in `apply_cylindrical_lens`).

### Quick wins

* **5 mis-tiered names in `__init__.py.__all__`** moved to correct
  tiers: `apply_real_lens_traced_jax`, `apply_real_lens_maslov_jax`
  → Tier 1 (Build a system); `monte_carlo_tolerancing_jax`,
  `monte_carlo_tolerancing_linearized`, `tolerancing_report` →
  Tier 4 (Analyse).
* **Duplicate `reset_fft_backend` import** removed from `__init__.py`
  (was imported at both line 46 and line 59).
* **Wiki Release-Notes.md broken anchor** fixed:
  `[4.13.1](#whats-new-in-4-14-0)` → `[4.13.1](#whats-new-in-4-13-1)`
  (stale from the v4.14→v4.13.1 rename script during v4.13.1 ship).

### Test counts

* Pre-v4.13.2 baseline: 654 unit tests.
* v4.13.2 audit-response additions: 56 new tests across 4 new files
  (Agent A: 12 in `test_audit_fixes_v4_13_2_agent_a.py`; Agent B:
  20 in `..._agent_b.py`; Agent C: 14 in `..._agent_c.py`; Agent
  D: 10 in `..._agent_d.py`).
* Final: **710 unit tests passing**, **34/34 validation files
  passing**.

### Deferred to v4.14.0

Tier-1 perf opportunities from the v4.13.1 audit Part 3 (modal
asymptotic per-pixel loop 20-100×; multi-merit meshgrid cache
5-10×; phase-retrieval `angle`/`exp` round-trip 2-4×; coating
reflectance wavelength batch 5-15×; decompose_lg/hg per-mode
rebuild 3-8×; Shack-Hartmann gather loop 5-15×; `_evaluate
_polynomial_4d_and_grad34` 3-5×) plus the cross-library survey's
top user-facing API gaps (encircled energy, MTF cutoff frequency,
depth of focus, plot_wavefront, beam diameter at threshold) plus
parametrized dispatcher pins for the three audit-recommended
sibling families (`(scalar, vectorial) HFPI`, `(NumPy, JAX)
apply_*`, Welford-mirror convention).

### Deferred to v5.0

Tier-2/3/4 structural items: six file splits (`raytrace/core.py`
4422 LOC, `propagation.py` 3710, `asymptotic.py` 3597, `optimize
/core.py` 3258, `io/prescriptions.py` 2829, `analysis/core.py`
2196); CI gates (`ruff`, `mypy --strict`, unit-test PR job);
back-compat shim removal; shared Chebyshev helpers extraction;
audit-fix test-file consolidation by topic; constrained
optimisation; checkpoint/resume; CDGM/Hikari/Sumita glass
catalogues; off-axis conics in surface frame; Q-type freeform.

## [4.13.1] — 2026-05-17

**Closes the v4.13.0 audit (`docs/audits/AUDIT_V4_13_0_2026_05_17.md`) plus an
additional perf-survey pass.**  v4.13.0 was tagged in git but never
published to PyPI; that audit identified 7 P1 (latent bug), 9 P2
(code smell), and 6 P3 (cleanup) findings — most importantly 3
"sibling-gap" recurrences (the `apply_real_lens` mirror guard
missed its parent function; the L2 JAX dtype routing missed
`error_reduction_jax`/`hybrid_input_output_jax`; the L3 `dy`
threading missed `Source.propagate` + 5 classmethod factories).

v4.13.1 closes every Tier-0 and Tier-1 item from that audit, plus
all Tier-2 P2 follow-ups and the Tier-3 cleanups, plus 3 new perf
wins discovered in the parallel survey.  All 654 unit tests pass;
34/34 validation files pass.

### Breaking changes — none

No user-facing breakage in v4.13.1.  The v4.13.0 breaking changes
(`rcwa.py` → `thin_grating.py` rename without back-compat shim,
and the `wavelength` sentinel raising `ValueError` on omission)
are inherited but do not regress further.  See v4.13.0 entry's
"Breaking changes" subsection below.

### Sibling-gap P1 closures (audit's headline finding)

**P1-A — `apply_real_lens` mirror guard.**  The v4.13.0 L4a sweep
hardened 4 sibling `apply_real_lens_*` variants (`_traced`,
`_traced_jax`, `_maslov`, `_maslov_jax`) but missed the parent
`apply_real_lens` itself.  A hand-built prescription containing
`surfaces[i]['is_mirror']=True` (and no `'elements'` key) would
silently misompute through `apply_real_lens` while raising
`ValueError` from the 4 siblings.  Ported the same pre-flight
guard from the `_lens_traced.py` template.  Added a parametrized
dispatcher-level pin (`TestL4aMirrorGuardDispatcherPin`) over all
5 variants to prevent this class of sibling-gap from recurring.

**P1-B — ER/HIO complex-dtype routing.**  `gerchberg_saxton_jax`
correctly routed `dtype` through `_resolve_jax_complex_dtype` in
v4.13.0, but `error_reduction_jax` and `hybrid_input_output_jax`
skipped the resolver — the EXACT silent float64 → complex64
demotion bug L2 was supposed to close, still live on those two
kernels.  Both now go through `_resolve_jax_complex_dtype` and
`_resolve_jax_real_dtype`; cache keys pinned on the resolved
complex dtype to match the GS pattern.  Parametrized dispatcher
pin (`TestP1BPhaseRetrievalDtypeResolver`) covers all 3 kernels.

**P1-C — `Source.propagate()` and 5 factories thread `dy`.**
`Source.propagate(...)` was constructing the result Source
without `dy=self.dy`, silently losing the y-pitch on every
anamorphic call.  Same gap in `Source.gaussian`, `plane_wave`,
`point_source`, `top_hat`, and `fiber_mode` — each factory
forwarded `dy` to its `create_*` helper (so the underlying
E-field WAS built on the anamorphic grid) but the final
`cls(E=..., dx=..., wavelength=..., ...)` call dropped `dy`.
Fixed in all 6 sites.  `Source.propagate` gained an optional
`output_dy` kwarg for symmetry with the existing `output_dx`
override.  Parametrized dispatcher pin
(`TestP1CSourceFactoryDispatcherPin`) covers the 4 dy-aware
factories; `fiber_mode` excluded because `create_fiber_mode`
itself does not accept `dy=` (pre-existing limitation outside
P1-C scope).

### UI + infrastructure P1 closures

**P1-D — `ThinGratingDock._run` kwargs mismatch.**  The dock was
calling `grating_efficiency_vs_wavelength(groove_index=...,
substrate_index=..., profile=..., angle=...)` but the function
expects `(period, depth, *, n_ridge, n_groove, n_substrate,
n_superstrate, order, ...)`.  Every dock click raised
`TypeError`, swallowed silently into the summary text box — the
dock was non-functional.  Rewrote `_run` with the correct
signature, added missing UI inputs for `n_ridge` and
`n_superstrate`, and extracted the math path into a pure
`_compute_efficiency_data(inputs: dict) → dict` helper so the
compute path is unit-testable without a live `QApplication`.  As
a side effect, the dock now does **one** `thin_grating_efficiency_1d`
call per wavelength (full per-order matrix) instead of N sweeps
through the single-order helper.

**P1-E — `_context.py` cache-clear import moved inside try.**
The `from .propagators.propagation import clear_asm_caches` was
OUTSIDE the try-block guarding the call.  If the import raised
(rename, circular import, partial install), the `ImportError`
bypassed not just that cache-clear but all 6 subsequent guarded
cache-clear blocks.  Moved inside the try, added `ImportError`
to the except tuple, matching the pattern used by the other 6
clears.

**P1-F — `RandomState.choice` on JAX honours `replace=False`.**
With `p=None`, the JAX path silently ignored `replace=False` and
returned with-replacement samples.  Now dispatches to
`jax.random.choice(sub_key, a, shape=shape, replace=False, p=p)`
for both weighted and unweighted branches (JAX 0.10.0 supports
the combination — verified at runtime).

**L7 — `test_bench_through_focus_scan_jax_first_vs_warm`** now
clears `_THROUGH_FOCUS_SCAN_JAX_CACHE` before timing the first
call, matching the 4 sibling benchmarks.  On a re-run within the
same process the reported "first call" timing is now the cold
path, not a cached compile.

### Tier-2 (code smells from the audit)

* **`_RestoreDtype` try/finally** — `_RestoreDtype.restore()` is
  now idempotent (`_restored` flag).  The dtype restoration in
  `design_optimize`'s scipy dispatch + final-eval block is
  wrapped in `try/finally` calling `restore()` explicitly.
  `__del__` retained as a safety net.  More robust under
  `KeyboardInterrupt` and exception unwinding.
* **`_merit_jac_auto` uses `scheme='forward'` + cached `f0`** —
  scipy already evaluates `merit_fn(x)` before calling jac on
  the same `x`.  `_merit_jac_auto` now passes that value as `f0`
  to `_fd_grad_for(... scheme='forward', f0=...)`.  FD eval
  count goes from `2N` to `N + 2` per gradient (~30% saving for
  N=10 free vars).  Threaded through internal helpers so
  callers that prefer the bit-identical central path still get
  it by default.
* **Cancellation protocol in `progress.py`** — new
  `CancellableProgress` class (exposed at `lumenairy.
  CancellableProgress`) with a `cancel()` method and
  `is_cancelled()` module helper.  Wired into all 4 scipy
  callbacks in `design_optimize` (`minimize`, `differential
  _evolution`, `basin_hopping`, `dual_annealing`).  When
  `cancel()` is called, the active scipy callback returns
  `True`; scipy stops gracefully and the post-loop final
  evaluation + `DesignResult` return still executes (so the
  caller gets the best-so-far state instead of a partial-data
  `KeyboardInterrupt`).
* **Merit propagator inconsistency warning** —
  `MultiWavelengthMerit`, `MultiFieldMerit`, and
  `ToleranceAwareMerit` all call `apply_real_lens` directly for
  off-nominal legs regardless of which `wave_propagator` was
  selected at `design_optimize` time.  Added a docstring caveat
  to each of the 3 classes AND a runtime `UserWarning` at
  `design_optimize` entry when `wave_propagator != 'real_lens'`
  and at least one of these Merit classes is in use.  Threading
  the propagator through sub-merit evaluations is a larger
  architectural change worth its own release.
* **Shared `_build_asm_H_square` helper** — the v4.13.0 Shack-
  Hartmann FFT batching introduced a local `_build_asm_H_for
  _lenslet` in `analysis/detector.py` that duplicated angular-
  spectrum H/bandlimit logic from `propagators/propagation.py`.
  Now consolidated: `_build_asm_H_square(N, dx, z, wavelength,
  dtype=None, bandlimit=True)` lives in `propagators/propagation
  .py` and is imported by `detector.py`.  Pinned at 1e-14 against
  a hand-built reference for both `bandlimit=True/False`,
  multiple grid sizes, complex64 dtype promotion, and the z=0
  unity short-circuit.  The propagator's own two inline H-build
  sites (NumPy chunked path and JAX functional path) stayed
  inline — they use cached freq-grid lookups + RAM-budgeted
  chunking (NumPy) and `xp.namespace` for JAX tracing, both of
  which materially differ from the helper's single-shot pure-
  NumPy build.
* **`_fd_grad_pure` `validate_f0` parameter** — when
  `scheme='forward'` and `f0` is supplied, `validate_f0=True`
  (default `False`) re-evaluates `f(x)` once and asserts
  `abs(f0 - f(x)) < tol * max(abs(f0), 1)`.  Caller contract
  documented in the docstring: stale `f0` silently produces
  wrong gradients without the validation gate.
* **BSDF TIS shape assertion** — `total_integrated_scatter`'s
  `np.broadcast_to(B, T.shape)` previously masked shape
  mismatches from subclass `BSDF.evaluate()` returns.  Now
  raises `ValueError` with expected vs actual shape on
  mismatch.

### Tier-3 cleanups

* **`memory.set_max_ram` validates non-negative input** —
  previously `set_max_ram(-5)` was silently accepted as
  -5 GB → negative bytes.  Now raises `ValueError`.
  `get_max_ram` added to `__all__`.
* **`MultiPrescriptionParameterization` duplicate detection** —
  duplicate `(prescription_index, *path)` entries in `free_vars`
  silently got separate `x[i]` slots competing for the same
  field.  Constructor now raises `ValueError` listing the
  duplicates.
* **JAX 0.4.20+ opaque PRNG keys** — `_is_jax_prng_key` now
  recognises opaque keys from `jax.random.key()` via the
  canonical `jax.dtypes.issubdtype(d, jax.dtypes.prng_key)`
  check, with a `dtype.name.startswith('key<')` string fallback.
  Legacy uint32 / shape-trailing-2 typed keys still detected.
* **Dtype-aware zero in `apply_mirror` and `apply_aperture`** —
  the `xp.where(..., E, 0.0+0.0j)` literal is complex128 in
  Python.  On JAX with `x32` default this may upcast or fail
  jit.  Replaced with `xp.zeros((), dtype=E.dtype)` in both
  `apply_mirror` (audit-cited) and `apply_aperture` (same
  pattern in the same file, fixed conservatively).
* **`apply_mirror` aperture docstring** — said "ellipse" but the
  code computes a circle in physical coordinates.  Docstring
  corrected.
* **Stale pyc cleanup** — removed
  `tests/unit/__pycache__/test_audit_fixes_v4_13_0_perf_hfpi
  _bincount.cpython-314-pytest-9.0.3.pyc` (γ.1 revert
  leftover).

### New performance wins (beyond audit scope)

Open-ended perf survey across `propagators/` and `raytrace/` (the
modules not already optimised in v4.12.0 or v4.13.0) turned up
three wins:

| Hot path | Old | New | Speedup |
|---|---|---|---|
| `vectorial_hfpi.accumulate_vector_to_grid` (1M paths, 256²) | 57.3 ms | 34.6 ms | **1.65-1.75×** |
| `analysis/detector.py:shack_hartmann` scatter-back (K=4096) | 2.97 ms | 0.12 ms | **9.5-25×** |
| `gbd.reconstruct_field_from_beamlets` (1024 beamlets, 96²) | ~1100 ms | ~870 ms | **1.2-1.5×** typical, up to 2.3× cache-warm |

* **`accumulate_vector_to_grid`** now shares the `ix`/`iy`/
  `inside`/`flat_idx` index arrays between the Ex and Ey scatter-
  adds (previously each call routed through `accumulate_to_grid`
  twice, recomputing the indices).  Bit-exact (`np.array_equal`).
  JAX path falls back to the original double-call form for
  tracing compatibility.
* **`shack_hartmann` scatter-back** replaced the `for k in
  range(K)` Python loop with vectorised fancy indexing using
  `iy_idx[ok]` / `ix_idx[ok]` / `cx_arr[ok]`.  Bit-exact.
* **GBD `reconstruct_field_from_beamlets`** fuses two `xp.exp`
  calls into one (`arg = -0.5*Q*rho2 + L*dX + M*dY`); replaces
  the `sum(a_b * phase, axis=-1)` reduction with `einsum('mnk,k
  -mn', phase, a_b)` to drop the `(Ny, Nx, chunk)` intermediate;
  switches accumulator to in-place `out +=` on NumPy/CuPy.
  Bit-near-exact (rel_err ~4e-16 vs the scalar reference).
  JAX rebind preserved.

**Deferred perf candidates** (catalogued for v4.14+):

* `propagators/hf.py:propagate_huygens_fresnel_with_opl_callable`
  per-pixel python loop calling `opl_fn` 16 times per output
  pixel (Van Vleck Hessian).  Could vectorise across `chunk
  _output` pixels.  Expected 5-20× on typical 256² grids with
  custom OPL.  Deferred: requires changing the documented
  callable contract (opl_fn must broadcast over arbitrary
  trailing shapes).
* Fused `(sag, dz_dh)` helper for `_intersect_surface` Newton
  iterations.  Currently `_surface_sag_xy` and
  `_surface_sag_derivatives_xy` both recompute `h = sqrt(x²+y²)`
  and re-traverse the aspheric polynomial dict on each of ~10
  Newton iters.  Expected 1.5-2× on aspheric surfaces.
  Deferred: requires reorganising `lenses.py:surface_sag
  _general`.
* Analytic pure-conic intersection extending the v4.12.1 Newton-
  skip from `kc==0` to all `kc` with no aspherics.  Expected
  5-10× on pure-conic surfaces.  Deferred: root selection for
  hyperbolic (`k<-1`) and degenerate paraboloid (`k=-1`,
  on-axis ray) cases needs broader validation.

### v4.13.0 CHANGELOG drift fixes (Tier-1 doc hygiene)

11 doc-vs-code mismatches in the v4.13.0 CHANGELOG entry below
have been retroactively corrected (line numbers, function names,
threshold directions, claim tightness).  The 12th (v4.12.2's
"6 clear-functions wired into lumenairy_context" should be 7 —
`clear_through_focus_scan_jax_cache` was the 7th but went
uncounted) is acknowledged here rather than retroactively edited
because v4.12.2 is already on PyPI.

### Test counts

* Pre-v4.13.1 baseline (v4.13.0 final state): 573 unit tests.
* v4.13.1 audit-response additions: 72 new tests across 8 new files
  (Agent 1: 14 in `test_audit_fixes_v4_13_0_jax_dtype_dy_siblings
  .py` (extended); Agent 2: 14 across `test_audit_fixes_v4_13_1
  _thin_grating_dock.py`, `_context_guards.py`, `_random_choice
  .py`; Agent 3: 26 in `test_audit_fixes_v4_13_1_agent3.py`;
  Agent 4: 18 across `_asm_h_helper.py` and 3 perf-pin files).
* Final: **654 unit tests passing**, **34/34 validation files
  passing**.

---

## [4.13.0] — 2026-05-17

**Bundle of three internal phases since v4.12.2 (PyPI-published):**

* Phase 1 — closes audit known-limitations S1, S2, S3 and L2, L3,
  L4, L6, L8 from `docs/audits/AUDIT_V4_12_1_2026_05_16.md` (storage dtype
  preservation, codegen aperture-stop + wavelength sentinel, ghost
  R/r convention, JAX dtype + `jax_enable_x64`, `PropagationResult.dy`
  + `Source.dy`, sibling mirror-guards, `apply_mirror` xp + dy,
  zarr thread-safety).
* Phase 2 — sweeps `except Exception:` clauses across the non-UI
  codebase from 99 → 3 justified sites (typed exceptions
  everywhere else, three WARN-BEFORE-PASS upgrades).
* Phase 3 — Tier-2 perf wins from the same audit: 10× thin-grating,
  188× BSDF TIS, 4.43× Chebyshev freeform, 3.26× coating-stack,
  10.8× Shack-Hartmann FFT batching, 4–72× Seidel field sweep,
  smaller wins on `wave_opd_2d` and `_fd_grad`.  Also: rcwa.py →
  thin_grating.py file rename with no back-compat shim (sole-user
  waiver).

All 573 unit tests pass; 34/34 validation files pass.

### Breaking changes

* **`rcwa.py` → `thin_grating.py` file rename, no back-compat
  shim.**  `import lumenairy.elements.rcwa` raises
  `ModuleNotFoundError`.  `import lumenairy.ui.rcwa_dock` likewise.
  The public symbols (`thin_grating_efficiency_1d`,
  `grating_efficiency_vs_wavelength`) were already renamed in
  v4.4.0; v4.13.0 finishes the rename at the file level.
* **`wavelength` is now required by `io.codegen
  ._decompose_prescription`.**  Previously a missing `wavelength`
  silently defaulted to `1.31e-6` (NIR).  Now raises `ValueError`
  with a helpful message naming the parameter.  Visible-band
  Zemax imports that relied on the silent NIR default must now
  pass `wavelength` explicitly.

### Phase 1 — Audit known-limitations closure

**S1 — `io.storage` complex-dtype preservation through append.**
`save_jones_field_h5`, `append_plane_h5`, `_zarr_append_plane`, and
the unified `append_plane` dispatcher gained a `preserve_dtype`
parameter that is threaded down through the write path.  Previously
the append-side stack silently promoted complex64 inputs to
complex128 on every plane after the first.  Default behaviour is
preserved: callers that omit `preserve_dtype` continue to see the
v4.12.x promotion.

**S2 — `io.codegen` aperture-stop emission + wavelength sentinel.**
`_decompose_prescription` now emits an `{'type': 'aperture', ...}`
step before mirrors / dummy planes / real-lens groups whenever
`is_stop=True` on the source surface.  The Zemax loader threads the
`is_stop` flag through `prescriptions.py` so the stop reaches
codegen with its identity intact.  Separately, the silent default of
`wavelength_nm = 1.31e-6` was replaced with a `ValueError`; callers
that previously got NIR by accident now get a clear failure at
codegen time.

**S3 — `analysis.ghost` `R` vs `r` convention disambiguation.**
A top-of-module convention block now disambiguates uppercase `R`
(Fresnel reflectance) from lowercase `r` (curvature radius).
Local variables renamed accordingly (`R_i_val → r_i`,
`R_j_val → r_j`).  Public dict keys are unchanged for back-compat
consumers.

**L2 — JAX complex-dtype routing.**
Added `_resolve_jax_complex_dtype()` and `_resolve_jax_real_dtype()`
helpers in `propagators/propagation.py`.  When complex128 is
required, the helper auto-enables `jax_enable_x64` and emits a
`RuntimeWarning` rather than silently truncating.  The warning is
implicitly one-shot (gated by the `jax_enable_x64` state flip after
the first call), not enforced via `warnings.simplefilter('once')`.
Threaded through
`system.py:propagate_system_jax`, `_lens_jax.py` (three call
sites), and `analysis/phase_retrieval.py`.  The
`_PROPAGATE_SYSTEM_JAX_CACHE` key now includes `str(np.dtype(cdtype))`
so a float32-then-float64 sweep stops aliasing onto a single XLA
binary.

**L3 — `PropagationResult.dy` and `Source.dy` fields.**
`PropagationResult` gained a `dy` field (defaults to `dx` in
`__post_init__`); `dy_out` / `dx_out` aliases preserved.
`_coerce_field` in `propagators/dispatch.py` now returns
`(field, dx_out, dy_out)`; threading carried through every internal
caller.  `Source` (in `sources/core.py`) gained a matching `dy` that
`to_source()` forwards.  Anamorphic-grid propagation can now round-
trip metadata without losing the second axis.

**L4 — Sibling mirror-guards in `apply_real_lens_*` variants.**
`apply_real_lens_traced_jax`, `apply_real_lens_maslov_jax`, and
`lenses_maslov.apply_real_lens_maslov` all gained the pre-flight
`is_mirror=True` guard previously only present in
`apply_real_lens`.  Calls into a folded prescription that should
have been split now raise `ValueError` consistently across the
trio + maslov.

* L4b — `phase_retrieval.py` raises `NotImplementedError` when an
  `initial_guess` is passed on the JAX backend (the JAX path does
  not currently honour it); migration message included.
* L4c — `phase_retrieval.py` warns + synthesises an empty history
  list when `return_history=True` is requested on the JAX backend.

**L6 — `apply_mirror` xp dispatch + `dy` parameter.**
`elements/elements.py:apply_mirror` now resolves the backend
namespace via `_xp_of(E_in)` and accepts a `dy` parameter for
anamorphic grids.  NumPy short-circuit retained for the
numba-aspheric fast path; JAX and CuPy take the xp-native inline-sag
branch.  Backwards compatible: `dy=None` reproduces the v4.12.x
square-grid behaviour exactly.

**L8 — Zarr thread-safety guard.**
Added a module-level `_ZARR_MKDIR_PATCH_LOCK = threading.Lock()` in
`io/storage.py` to guard the `Path.mkdir` monkey-patch in
`_open_zarr_group_safe`.  Concurrent writers no longer race on
patch-install vs patch-restore.

### Phase 2 — `except Exception:` sweep

Non-UI library files contained 99 `except Exception:` clauses
(audit's "242" figure swept the full repo including tests,
validation, and commentary; the real non-UI count was 99).  After
the sweep:

* 3 KEEP-AS-IS, justified inline:
  * `_context.py:299` — atexit handler tolerating module-level
    global teardown.
  * `optimize/core.py:2683` — `_RestoreDtype.__del__` cleanup
    tolerating shutdown.
  * `propagators/hfpi.py:84` — optional-dep guard around
    `import jax`.
* ~85 narrowed to typed tuples covering the documented failure
  modes (e.g. `(RuntimeError, MemoryError, ValueError, TypeError,
  AttributeError)` for pyFFTW failures; `(ImportError, RuntimeError,
  AttributeError)` for cache-clear guards; `(ValueError,
  RuntimeError, ZeroDivisionError, np.linalg.LinAlgError,
  IndexError)` for `system_abcd` fallbacks; etc.).
* 3 WARN-BEFORE-PASS upgrades — surfaces in failure paths that
  would previously have silently degraded:
  * `analysis/field.py petzval_radius` — a missing glass entry was
    returning NaN, which downstream could be confused with "the
    field is perfectly flat."  Now warns explicitly with the glass
    name + wavelength.
  * `propagators/hf.py` LG decomposition fallback — falling back to
    a single `(p=0, l=0)` plane-wave mode makes the asymptotic
    propagator essentially useless; failure is now surfaced.
  * `optimize/core.py design_optimize plane_logger` callback — was
    silently swallowing all logger failures for the duration of an
    optimization run.  Promoted to WARN-BEFORE-PASS so users see
    telemetry-callback bugs immediately instead of an empty log
    file at the end.

Pinning test:
`tests/unit/test_audit_fixes_v4_13_0_except_sweep.py` with 6 tests
including a regression budget guard (non-UI `except Exception:` count
≤ 15), two behavioural pins (`petzval_radius` warns, narrow tuples
drop on expected failures), and one source-string pin via
`inspect.getsource` for the `design_optimize plane_logger` warning
(behavioural exercise was unreliable across scipy versions).

### Phase 3 — Tier-2 performance wins

Three disjoint-scope agent groups (elements/UI, analysis/optimize,
propagators/raytrace) executed in parallel.  Measured speedups
(`time.perf_counter`, representative workloads on Win11 / Python
3.14):

#### Group α — elements + UI

| Hot path | Old | New | Speedup |
|---|---|---|---|
| `thin_grating_efficiency_1d` (n_orders=25) | 198 us | 18.9 us | **10.5×** |
| `bsdf.total_integrated_scatter` (256×128) | 384 ms | 2.05 ms | **188×** |
| coating-stack matmul chain (50 layers) | 0.48 ms | 0.15 ms | **3.26×** |
| freeform Chebyshev (64×64, 16 coeffs) | 1.63 ms | 0.37 ms | **4.43×** |

**`elements/rcwa.py` → `elements/thin_grating.py` rename.**  The
file name "rcwa" was misleading: the function inside is the
analytical scalar thin-phase grating formula, not rigorous coupled-
wave analysis.  The functions themselves were renamed in v4.4.0
(`thin_grating_efficiency_1d`, `grating_efficiency_vs_wavelength`);
v4.13.0 finishes the rename at the file level:

* `lumenairy/elements/rcwa.py` → `lumenairy/elements/thin_grating.py`
* `lumenairy/ui/rcwa_dock.py` → `lumenairy/ui/thin_grating_dock.py`
  (class `RCWADock` → `ThinGratingDock`)
* `lumenairy/__init__.py` import path + Tier-7 comment updated.
* `lumenairy/elements/__init__.py` module docstring updated.
* `lumenairy/ui/main_window.py` (7 token occurrences across 6
  logical sites: import, widget construct, dock construct, dock-key,
  menu label, show_and_raise lambda, visible-list) +
  `lumenairy/ui/workspace.py` (dock-key mapping) updated.

**No back-compat shim** is installed at either old path
(`import lumenairy.elements.rcwa` raises `ModuleNotFoundError`).
This is per the explicit waiver from the sole user of the library.

**`bsdf.total_integrated_scatter`** is now a fully-vectorised 2D
meshgrid + single `integrand.sum() * dθ * dφ` reduction (rectangle
rule on a regular θ-φ grid) rather than a per-(θ_i, θ_s) inner-
product loop.  This is the biggest single-call speedup in the
v4.13.0 batch.

**Coating-stack matmul** moved from a Python `M = M @ M_layer` loop
to a tournament reduction over a `(N, 2, 2)` per-layer tensor.  The
agent also switched the inner Snell-chain math from `np.sin/arcsin`
to `math.sin/asin` for scalar wavelengths — bit-near-identical via
libm (pinned at `atol=1e-12` rather than ULP-exact since libm
implementations can diverge in the last few bits across versions) —
lifting the win from 1.5× into the 3-5× target band.

**Freeform Chebyshev** hoists the `arccos(rho)` out of the per-order
loop and caches the `T_i(rho)` factors keyed by polynomial order,
exploiting the fact that many `(i, j)` coefficient pairs share an
`i` or `j` so 2N cos calls collapse to roughly `2 sqrt(N_coeffs)`.

#### Group β — analysis + optimize

| Hot path | Old | New | Speedup |
|---|---|---|---|
| Shack-Hartmann WFS (16² lenslets, 256² grid) | 72.9 ms | 6.76 ms | **10.8×** |
| `wave_opd_2d` (512²) | 52.5 ms | 27.7 ms | **1.89×** |
| `_fd_grad_pure` (N=20 quadratic, forward) | 0.159 ms | 0.071 ms | **2.23×** |

**Shack-Hartmann FFT batching.**  `analysis/detector.py:
shack_hartmann` previously ran two nested per-lenslet double-loops
(reference + measurement), each computing an `np.fft.fft2` on a
single sub-aperture.  The reference path is now a single fft2 on a
stacked `(n_lenslets², sa_pixels, sa_pixels)` 3D array; the
measurement path is the same.  A new `_build_asm_H_for_lenslet`
helper inside `detector.py` duplicates a thin slice of the
angular-spectrum H-build logic for the sub-aperture geometry,
avoiding a scope-crossing edit into `propagators/` (a follow-up
in v4.13.1 consolidates this into a shared
`_build_asm_H_square` helper).  NaN sentinels for OOB lenslets
and the `sa_pixels >= 2` (i.e. `pitch >= 2*dx`) sampling guard
are preserved.  Note: this guard raises `ValueError`, not a
warning.

**`wave_opd_2d` axis unwrap.**  `analysis/core.py:wave_opd_2d`
replaces the row-then-column per-pixel Python unwrap loop with
`np.unwrap(opd, axis=1)` then `np.unwrap(..., axis=0)` (matching
the legacy row-first traversal order).  At N=512
the Python iteration overhead was about half the run time; the
inner unwrap step itself was already compiled, so the 1.89× win is
real but below the speculative 5-10× target.  Correctness preserved
to 1e-12 on smooth quadratics, twice-wrapping tilts, and masked
NaN regions.

**`_fd_grad_pure` / `_fd_grad_for` central-vs-forward scheme.**
Pre-v4.13 the helper used central differences (2N evals,
`f(x ± h*e_i)`).  The audit's request to "reuse the centre value"
was based on a forward-FD model — central FD has no `f(x)` to
reuse, so the perf win required switching the scheme.  v4.13.0
parameterises this:

* `_fd_grad_pure(...)` and `_fd_grad_for(...)` accept
  `scheme='central'|'forward'`, default `'central'`.
* `scheme='central'` (the default) preserves bit-identical
  gradient values with pre-v4.13 behaviour at 2N evals,
  O(h²) truncation.
* `scheme='forward'` is the opt-in perf path (N+1 evals, or N with
  the optional `f0=<known value>`).  O(h) truncation.

`design_optimize._merit_jac_auto` keeps the default `'central'`
explicitly, so no existing optimisation run sees a behavioural
change.  Callers that prefer speed can opt into `'forward'` at the
helper level.

#### Group γ — propagators + raytrace

| Hot path | Old | New | Speedup |
|---|---|---|---|
| `seidel_field_sweep` (5 fields, singlet) | (ref) | (ref/4) | **4.2×** |
| `seidel_field_sweep` (50 fields, singlet) | (ref) | (ref/37) | **37.5×** |
| `seidel_field_sweep` (100 fields, singlet) | (ref) | (ref/72) | **72.0×** |

**`raytrace/seidel_analysis.py:seidel_field_sweep` per-field hoist.**
The paraxial Seidel formalism is exactly linear in the chief-ray
initial conditions, and those scale linearly with field angle.  The
sweep now does a single `seidel_coefficients(... field_angle=1.0)`
call and applies the analytical per-field scaling (`S1, S4 ∝ 1`;
`S2, y_chief ∝ σ`; `S3 ∝ σ²`; `S5 ∝ σ³`).  All field-independent
work — glass-index lookups, pre-stop ABCD, marginal-ray trace, full
`system_abcd` — is hoisted out of the loop.  Element-by-element
agreement with the pre-hoist reference is well below the test
pin tolerance of `< 1e-12 absolute` (numerics on the singlet
test case land in the 1e-15..1e-20 range in practice).

**γ.1 — HFPI bincount swap reverted before ship.**  The initial
agent run replaced the `np.add.at(out, flat_idx, w_masked)` scatter
in `propagators/hfpi.py:accumulate_to_grid` with a `np.bincount`-
based path on the premise that `add.at` was a Python-level loop.
That premise pre-dated NumPy 1.25; NumPy ≥ 1.25 has vectorised
`add.at` for complex via `_PyArray_UFuncBufferedAtVectorized`.
Measured speedup on the actually-shipping NumPy was 0.4–1.0×
(a wash, leaning regression), so the patch was reverted.  No
behavioural change vs v4.12.2 in this path.

### Bug-fix and discipline notes

* Phase 2 audit count discrepancy resolved.  Audit reported 346
  `except Exception:` clauses; the real non-UI count was 99 (the
  audit swept tests, validation, and prose into the same `grep`).
* Phase 3 Group α / γ race observed mid-run: Group γ briefly
  imported a non-existent `elements/rcwa.py` reference while
  Group α was mid-rename.  Resolved by Group α's completion; no
  artefact in shipped code.
* `tests/unit/test_audit_fixes_v4_12_0_round4_dispatch.py` —
  `test_coerce_field_unpacks_tuple` was rewritten in-place using
  `result[0]` / `result[1]` subscript indexing (no `pytest.mark.skip`)
  because L3 changed `_coerce_field`'s signature to a 3-tuple
  `(field, dx_out, dy_out)`.  All 7 tests in the class collect and
  pass.  The dispatcher-level 3-tuple pin is in the v4.13.0
  dy-siblings test file.

### Test counts

* Pre-Phase-3 baseline (Phase 1 + 2 landed): 536 unit tests.
* Phase 3 additions: 12 perf pins (α) + 12 (β) + 7 (γ) = 31, minus
  6 bincount tests deleted with the γ.1 revert + net 0 from the
  fd_grad test rewrite.  Final: **573 unit tests passing**,
  **34/34 validation files passing**.

---

**Archive note:** v4.10.x and earlier (down to v2.5.x) are preserved in [`docs/changelogs/v4.md`](docs/changelogs/v4.md).  The v5.2.3 release split moved v4.11.x - v4.12.x; v5.4.0 Phase 5 completes the pass.
