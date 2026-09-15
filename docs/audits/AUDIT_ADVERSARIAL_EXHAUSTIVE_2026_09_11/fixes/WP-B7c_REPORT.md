# WP-B7c -- the multibranch reconstruction blow-up, and the fold envelope's two constants

Wave 5 item C of `PLAN_WAVE5_LEFTOVERS_2026_09_14.md`; handoff items 4.2 (P2)
and 4.3 (P2) of `HANDOFF_2026_09_14.md`, i.e. VERIFY-B7b's request R-5 and its
section 4.2, plus WP-B7b section 7.  Branch `fix/multibranch-zeta-envelope`,
base `96cb2096`.

## 0. Terms, so the rest reads unambiguously

| term | meaning here |
|---|---|
| **the completion** | `apply_real_lens_traced_uniform` -- the Chester-Friedman-Ursell uniform Airy DARK-side completion of the multibranch field.  Also reached as `apply_real_lens_traced(caustic='uniform')` |
| **the branch sum** | `apply_real_lens_traced_multibranch` -- the KMAH ray-density sum the completion is built on and keeps verbatim on the bright side |
| **the hand-off** | `apply_real_lens_traced(caustic='wave')` -- the traced exit-vertex field propagated to the output plane by the band-limited angular spectrum.  `amplitude_model='ray_density'` and `'screen'` are its two exit amplitudes; `output_plane_distance` is honoured only by multibranch / uniform / wave, so a single-valued member is read at a non-zero plane through this route |
| **`power_ratio`** | the branch sum's own diagnostic: reconstructed grid power / the launch congruence's power that reaches the grid.  `power_ratio_triangles` is the same numerator over the whole-triangle (upper-bound) denominator; the **bracketed** reading is `min` of the two, which is what the branch sum's own gain arm decides on |
| **`zeta_extrapolation`** | `uniform_fit_halfwidth / zeta_band` -- how far past the two-branch band (the only radii at which the eikonal difference defining `zeta` exists) the fitted `zeta(r) = kappa (r_c - r)` is carried |
| **`l_airy`** | the fold's Airy boundary-layer width, `1 / (k^(2/3) kappa)`.  The dark fill runs `_AIRY_TAIL_CELLS` of these past `r_c` |
| **fidelity** | `abs(<a,b>) / (norm(a) norm(b))` -- phase- and scale-invariant, so it says nothing about absolute energy; energy is reported separately |
| **the oracle** | `validation/oracles/caustic_fold_truth.py`, a lumenairy-free direct Rayleigh-Sommerfeld ring integral over an exact meridional conic trace -- the same oracle construction WP-B7b and VERIFY-B7b used |

## 1. Summary

| # | item | verdict |
|---|---|---|
| 1 | the blow-up reproduces on VERIFY-B7b's fixture | **YES, to three digits** -- 93.5 / 3545 / 6097 against its 93.6 / 3543 / 6095, and identically on both builds |
| 2 | it reproduces on two more optics of mine | **YES** -- 1683x on an N-BK7 plano-convex, 18.9x -> 3201x on an N-SF11 biconvex, with the same signature (`fell_back=False`, `fit_residual` 0.003-0.007, `zeta_extrapolation` 0.27-0.38) |
| 3 | the mechanism | **the rasteriser's point-sampled area quadrature**, not a wrong branch, not an unclipped Jacobian, not the fold-member ordering (section 2, three controls) |
| 4 | the fix | the completion **REFUSES** outside a derived band and carries the reading and the decision in its diagnostics (section 3); the branch sum's reconstruction is left byte-identical |
| 5 | `_ZETA_EXTRAPOLATION_MAX = 8.0` re-derived two-sided on three optics | **CONFIRMED where it stands**, but as a SIGN boundary, not the 5 % / 10 % magnitude one: 8.0 is `sqrt(5.65 * 11.05) = 7.90` to two figures (section 4) |
| 6 | the member-selection pass | **both published orderings reproduce, on their own fixtures**; neither generalises; margins under 0.012 of fidelity; the docstring is restated and the default is NOT changed (section 5) |
| 7 | `zeta(r)` beyond the two-branch band | **measured**: the linear normal form is good to 7 % only while the band is >~2 `l_airy` wide, and 20-50 % slow beyond that.  A second fold parameter is the repair; recommended, not implemented (section 6) |
| 8 | clipping `_AIRY_TAIL_CELLS` | **premise measured and REFUSED**: >94 % of the excess energy is written inside 3 Airy lengths, not in the outer annulus.  The bound is derived and reported as a diagnostic; the fill is not clipped (section 6) |
| 9 | bit identity | **11 of 13 fixtures byte-identical on both builds**; the 2 movers are exactly the two blow-up planes, which now raise (section 7) |

## 2. Item 4.2 -- the mechanism of the `1/sqrt|J|` blow-up

### 2.1 Reproduction

VERIFY-B7b's fold fixture, verbatim: N-BAF10 biconvex R = +-2.6 mm, t = 0.70 mm,
0.90 mm aperture, lambda = 1.064 um, N = 512, dx = 2.20 um, w0 = 330 um.

| z (um) | 1758 | 1760 | **1762** | **1764** | 1766 | **1768** | 1770 |
|---|---|---|---|---|---|---|---|
| `zeta_extrapolation` | 0.348 | 0.339 | **0.330** | **0.322** | -- | **0.318** | -- |
| `fit_residual` | 0.0032 | 0.0031 | 0.0049 | 0.0029 | -- | 0.0054 | -- |
| `fell_back` | False | False | **False** | **False** | True | **False** | True |
| `power_ratio` | 0.825 | 0.845 | **93.47** | **3545** | 4694 | **6097** | 6614 |

VERIFY-B7b section 4.2 read 0.33 / 0.32 / 0.32, 0.0049 / 0.0029 / 0.0054 and
93.6 / 3543 / 6095 at 1762 / 1764 / 1768.  Mine are 93.47 / 3545 / 6097 on
Windows py3.14 / numpy 2.4.4 and the SAME to every printed digit on WSL
py3.12 / numpy 2.4.6 -- the pathology is not build-dependent.

Two more optics, chosen to share nothing with it:

| optic | the last healthy plane | the first blown-up plane |
|---|---|---|
| **C** N-BK7 plano-convex R = -2.0 mm FLAT side first, 780 nm, N = 512, dx = 2.00 um | z = 3680 um, `zeta_x` 0.390, `power_ratio` 0.816 | z = 3685 um, `zeta_x` 0.378, `fit_residual` 0.0035, **1683** |
| **D** N-SF11 biconvex R = +-3.4 mm, 1.55 um, N = 512, dx = 3.00 um | z = 1986 um, `zeta_x` 0.289, `power_ratio` 0.879 | z = 1990 um, `zeta_x` 0.279, `fit_residual` 0.0056, **18.86** |

In every case the completion's own diagnostics read their BEST values on the
broken field.  On D the ladder also exposes the smallest pathological reading
anywhere in the study, 18.86 -- the number the refusal bar is derived against.

### 2.2 The mechanism, with its three controls

The brief's three candidates were: the wrong branch, an unclipped `|J| -> 0`,
or the fold-member ordering.  **None of them.**  The blow-up is the rasteriser's
POINT-SAMPLED AREA QUADRATURE losing its unbiasedness.

A mapped launch triangle deposits `|E_in|^2 / ratio` on every pixel whose
CENTRE it covers, where `ratio` is its mapped area over its launch area.  Near a
caustic essentially every mapped triangle is sub-pixel: on this fixture the
median mapped area is **1/430 of a pixel at every plane, including the healthy
ones** (`Apx` p50 = 0.0023 at z = 1750 and 0.0015 at 1768; 100 % of triangles
are sub-pixel in area AND in minimum altitude at both).  So the write is a
Monte-Carlo estimator of the area integral: a triangle of mapped area `A`
catches a pixel centre with probability `A/dx^2` and then deposits
`dx^2 |E_in|^2 / ratio`, whose expectation is exactly the launched
`A_tri |E_in|^2`.  That is why the healthy planes conserve energy at all, with
100 % sub-pixel triangles.  The estimator is unbiased only while those triangles
are SPREAD over many pixels.  Where a whole RING collapses onto a handful, the
variance becomes the mean.

| control | reading | what it rules out |
|---|---|---|
| `caustic_band='plain'` vs `'ludwig'` at z = 1768 | **6090.1 vs 6096.7** (0.1 %) | the Ludwig pair swap, i.e. the fold-member ordering, is not involved |
| `min_area_ratio` 1e-8 vs the 1e-6 default at z = 1768 | **6096.7 vs 6096.7**, bit for bit | the clip is INOPERATIVE at the default (32 of 64 336 triangles).  The divergent amplitudes sit at ratios 1e-6..1e-3, i.e. `1/sqrt\|J\|` of 32..1000, which it admits.  Raising it to 1e-4 cuts the ratio to 93.3 and 1e-3 to 0.53, at the cost of skipping 4138 and 52 384 triangles |
| `ray_subsample` 4 -> 2 -> 1 at z = 1768 | **771.5 -> 6096.7 -> 44 878** | REFINING the launch lattice makes it 7.4x WORSE -- the signature of a quadrature whose written energy scales with the triangle COUNT rather than with their mapped area.  A convergent reconstruction cannot do this |

`n_branch.max()` -- the number of distinct mapped triangles landing on ONE pixel
-- is the state variable: it reads **2** at every healthy plane of the ladder
and **97 / 565 / 797** at z = 1762 / 1764 / 1768.  The `n_triangles_degenerate`
census stays at 8-68 of 64 336 throughout, so it is not the clip.

The geometry is the documented axial point-focus catastrophe (Scope note D5):
the caustic ring radius `r_c` shrinks 11.31 -> 9.72 -> 9.47 -> 8.96 um across
the window as the plane approaches the paraxial focus, and the ring lands on the
central pixels.

### 2.3 Why the reconstruction is not bounded, and the completion refuses instead

Bounding `1/sqrt|J|` "near the fold" would mean replacing the point sample by an
area-weighted splat -- a different quadrature that changes every multibranch
field near every caustic, needs its own oracle ladder and its own Migration
note, and is not a P2.  The brief's second arm was taken.

Falling back is not a remedy either, and this is measured, not argued: at
z = 1770 um the completion ALREADY falls back (`reason='zeta_nonlinear'`) and
returns a field carrying **6459x** the launched power, because the fallback
target is the same branch-sum field.

## 3. Item 4.2 -- what shipped, and the derived band

`_lens_traced_uniform.py` reads the branch sum's own energy diagnostic
immediately after building it, takes a DECISION, and carries both on every
return path:

```
multibranch_power_ratio              the branch sum's own power_ratio
multibranch_power_ratio_bracketed    min(power_ratio, power_ratio_triangles)
multibranch_power_ratio_band         (_MB_POWER_RATIO_MIN, _MB_POWER_RATIO_MAX)
power_ratio_decision                 'ok' | 'energy_loss' | 'refused_energy_gain'
                                     | 'no_launched_power'
```

Above the bar the call raises `RuntimeError` naming the ratio, the branch count,
the mechanism and the members that work.  The reading is taken through the SAME
bracket the branch sum's own gain arm uses, so the known false-positive class
that bracket exists for -- an aperture much wider than the grid, where the
node-count denominator alone reads up to 3.3x with the energy conserved to 1 %
(`test_audit2609_a3_verify_traced.py`'s D3 fixture) -- cannot reach it.

### 3.1 The derivation of `_MB_POWER_RATIO_MAX`

Three optics, 38 oracle-scored fold planes, `zeta_extrapolation` 0.27 .. 3484.
Classified by the completion's fidelity against the oracle, which is bimodal
with nothing in between (accepted 0.883-0.995; broken 0.012-0.259):

| population | n | multibranch `power_ratio` |
|---|---|---|
| **accepted** (oracle fidelity 0.883 .. 0.995) | 30 | **0.816 .. 1.246** |
| **broken** (oracle fidelity 0.012 .. 0.259) | 8 | **18.86**, then 93.5 / 1683 / 2346 / 2801 / 3201 / 3443 / 3545 / 6097 |

Finer plane scans without oracle scoring extend the accepted population down to
0.763 and add no rung between 1.25 and 18.86.

`sqrt(1.246 * 18.86) = 4.85`.  The shipped bar is
`_MB_POWER_RATIO_MAX = 2 * _ENERGY_BLOWUP_FACTOR = 4.0`, the nearest value that
keeps it coupled to the branch sum's own warn bar.  **Margins: 3.2x above the
largest accepted reading, 4.7x below the smallest broken one.**  The coupling
also guarantees the refusal is never the first diagnostic -- every refused field
has already emitted the branch sum's own `RuntimeWarning`.

### 3.2 The lower arm is REPORTED, not refused -- and why

A branch-sum field that loses energy is the NORMAL input to this module: the
dark-side Airy tail it exists to add is exactly what the branch sum drops.  The
measured legitimate population runs down to 0.763 with the completion still at
0.991 fidelity, against the module's own documented 0.887.  The only
pathological low case -- the total collapse to an identically zero field -- is
already REFUSED inside the branch sum.  A refusal here would be a bar with a gap
on one side only, so `_MB_POWER_RATIO_MIN` reuses `_ENERGY_COLLAPSE_FACTOR`
(0.5, derived there) purely as the reporting threshold, and the two modules
classify the same field the same way.

### 3.3 Scored against the oracle and against energy

At the refused planes the completion's fidelity against the oracle is **0.259 /
0.190 / 0.168** (V), **0.063 / 0.029 / 0.012** (C), **0.204 / 0.148** (D), and
its power is **94 / 3570 / 6140** and **1689 / 2811 / 3454** and **19.2 / 2365**
times the oracle's.  At the same planes the hand-off reads **0.9981-0.9991** and
conserves the launched power to within 1 %.  Every arm of the refusal therefore
removes a field that is wrong on both scores and points at one that is right on
both.

## 4. Item 4.3 -- `_ZETA_EXTRAPOLATION_MAX` re-derived two-sided

Three optics (V, C, D above), 30 accepted fold planes, completed-field total
power against the oracle's.  The bar separates a SIGNED error from a one-sided
gain -- not two magnitudes:

| | rungs | range | negatives | mean |
|---|---|---|---|---|
| **below** `_ZETA_EXTRAPOLATION_MAX` (`zeta_x` 0.29 .. 5.65) | 22 | **-2.18 % .. +4.14 %** | **12 of 22** | **+0.57 %** |
| **above** it (`zeta_x` 11.05 .. 3484) | 8 | **+3.64 % .. +30.13 %** | **0 of 8** | **+7.88 %** |

The full ladders, per optic (`zeta_x` -> power / oracle):

* **V** 0.385 -> 0.9972; 0.425 -> 0.9966; 0.741 -> 1.0271; 1.026 -> 0.9835;
  3.008 -> 0.9945; 5.653 -> 1.0198 | 14.00 -> 1.0588; 27.29 -> 1.0549;
  531.5 -> 1.0512
* **C** 0.390 -> 0.9989; 0.501 -> 1.0369; 0.656 -> 1.0383; 0.879 -> 0.9993;
  1.223 -> 0.9825; 1.789 -> 0.9782; 2.822 -> 0.9947; 5.016 -> 1.0177 |
  11.05 -> 1.0440; 40.20 -> 1.0406; 3484 -> 1.0431
* **D** 0.289 -> 1.0128; 0.300 -> 1.0035; 0.405 -> 0.9925; 0.564 -> 1.0414;
  0.819 -> 1.0138; 1.270 -> 0.9927; 2.178 -> 0.9925; 4.450 -> 1.0111 |
  13.16 -> 1.0364; 149.7 -> 1.3013

The last signed rung is **5.653** (V, +1.98 %) and the first systematic-gain
rung **11.05** (C, +4.40 %).  `sqrt(5.653 * 11.05) = 7.90`, i.e. **the shipped
8.0 already sits at the geometric centre of the measured transition on three
optics WP-B7b never saw**, with 1.41x of margin below and 1.38x above.

This settles the disagreement the handoff records.  WP-B7b's "+4.9 % at 5.4,
+12.5 % at 9.8" is its three singlets' ABOVE-bar behaviour; VERIFY-B7b's
"saturates near +5 % out to 531" is one optic's ABOVE-bar behaviour read
WITHOUT its below-bar population beside it.  Neither is wrong; both are
one-sided.  With both populations measured on three optics the bar is a
calibrated boundary of +/-4 % against +4 %, not a conservative flag and not a
5 % / 10 % boundary.  The constant's comment, the function docstring and the
warning text now say that.

The single largest excursion anywhere, D at `zeta_x` = 149.7 with **+30.13 %**,
is above the bar and warns.

## 5. Item 4.3 -- the member-selection pass

Five fixtures: WP-B7b's own two (reproduced verbatim -- W1, its fast N-LAK22
singlet whose marginal focus reads `zeta_extrapolation` 0.1647 against its
published 0.165; and W2, its "fixture F" whose z = 4.400 mm reads
`r_c` = 15.531 um, `kappa` = 7.964, band = 10.89 nm, `zeta_x` = 453.6 against
its published 15.5307 / 7.963835 / 10.9 / 453.6), VERIFY-B7b's (V) and two of
mine (C, D).  Members scored against the oracle at each plane.

**Both published orderings reproduce, each on its own fixture.**

| fixture | planes | completion | hand-off (`ray_density`) | winner |
|---|---|---|---|---|
| **W1** (WP-B7b's) `zeta_x` 0.165 | 1 | **0.9993** | 0.9974 | completion |
| **W1** `zeta_x` 0.19 .. 8.57 | 6 | 0.9927 .. 0.9993 | 0.9923 .. 0.9974 | completion (6 of 6) |
| **W1** `zeta_x` 23.1, 173 | 2 | 0.9906, 0.9882 | 0.9908, 0.9888 | hand-off, by 0.0002 / 0.0006 |
| **V** (VERIFY-B7b's) `zeta_x` 0.38 .. 531 | 9 | 0.9867 .. 0.9923 | 0.9963 .. 0.9981 | hand-off (9 of 9) |
| **C** `zeta_x` 0.39 .. 3484 | 11 | 0.9901 .. 0.9949 | 0.9988 .. 0.9991 | hand-off (11 of 11) |
| **D** `zeta_x` 0.28 .. 150 | 9 | 0.8828 .. 0.9929 | 0.9920 .. 0.9981 | hand-off (9 of 9) |

So the discriminator is the OPTIC, and both margins are small: the completion
wins by 0.0004-0.0019 on W1 and loses by 0.005-0.012 on V / C / D.

**The obvious candidate discriminator is refuted.**  W1's semi-aperture is
2.0 w0 and V / C / D's are 1.34-1.50 w0, so aperture truncation was the natural
hypothesis (the hand-off carries the rim's diffraction exactly, the fold
completion models only the fold).  A beam-width sweep at a FIXED optic and plane
(V, z = 1683.4 um, `zeta_x` = 1.026, w0 = 150 .. 380 um, rim amplitude
1.8e-04 .. 0.26) does **not** flip the ordering: the hand-off wins at every
width, and the completion is furthest behind at BOTH ends (-0.0118 at w0 = 150
and at 380, -0.0014 at 250).

**A caveat that belongs on all three reports.**  This oracle builds its
boundary field from geometrical optics at the exit vertex and then propagates it
exactly -- and the hand-off is literally "the traced exit-vertex field plus one
band-limited ASM leg".  So a hand-off-vs-oracle fidelity partly measures two
propagators of the same boundary field agreeing, and the comparison is not
neutral between the hand-off and the completion.  It does not touch section 2 or
3, where the readings are three-decade energy violations, but it does bound how
much weight the 0.005-0.012 margins here can carry.  What IS model-free is the
energy: the hand-off conserved the launched power to better than 1 % at all 38
planes, while the completion above the bar did not.

**Recommendation (not taken -- the default member is a maintainer decision).**
Leave `caustic='uniform'` as it is.  Restate the docstring, which is done: the
unscoped sentence "at the widest-band plane it also beats
`apply_real_lens_traced(amplitude_model='ray_density')`" is replaced by the
measurement above, with both orderings, their margins, the refuted
discriminator, and the advice to read the ranking on one's own prescription.
What IS general on every fixture measured, and is stated as such, is that the
completion beats the plain branch sum, and that the hand-off conserves energy
where the completion above the bar does not.

## 6. Item 4.3 -- `zeta(r)` beyond the band, and the `_AIRY_TAIL_CELLS` clip

### 6.1 `zeta(r)` measured beyond the two-branch band

`zeta` is defined only where both coalescing branches exist, so it cannot be
traced outside the band.  It CAN be read off the oracle's own dark-side decay:
`|E| ~ exp(-2/3 x^{3/2}) / x^{1/4}` with `x = k^{2/3} kappa (r - r_c)`, so a
windowed fit of `ln|E| + (1/4) ln x` against `(r - r_c)^{3/2}` returns an
effective `kappa_eff`.  Fixture V, first window (0.5 .. 3.1 Airy lengths):

| `zeta_extrapolation` | 0.385 | 0.425 | 1.026 | 3.008 | 5.653 | 14.00 | 531.5 |
|---|---|---|---|---|---|---|---|
| **`kappa_eff / kappa`** | 1.281 | **1.068** | 0.803 | 0.654 | 0.603 | 0.558 | 0.489 |
| band / `l_airy` | 2.60 | 2.36 | 0.975 | 0.332 | 0.177 | 0.071 | 0.0019 |

Clean and monotone: **the linear normal form is good to ~7 % only while the
two-branch band is at least ~2 Airy lengths wide (`zeta_extrapolation` <~ 0.4),
and the true dark-side decay is 20-50 % SLOWER than the extrapolated linear
model beyond that.**  (The deeper windows run to 0.03-0.15 and are not
meaningful: out there the oracle's field is no longer the fold's tail.)

**Recommendation: a second fold parameter, i.e. a quadratic `zeta(r)` or a
two-plane fit, is the repair, and it is NOT a one-step change** -- `zeta`'s
curvature is not measurable on the band (section 6.2 shows the band's own
quadratic term is anti-conservative by a factor of 20 against this
measurement), so it needs a second observation plane or a local wave solve, its
own oracle ladder and its own Migration note.  Not implemented here.

### 6.2 The `_AIRY_TAIL_CELLS` clip -- premise measured and refused

WP-B7b section 7 proposed clipping the fill "where the extrapolated `zeta` no
longer describes the tail", on the premise that the energy error is written into
the OUTER part of the 20-Airy-length annulus.  Cumulative dark-side energy of
the completed field against the oracle's in the same annulus, by depth, fixture
V:

| `zeta_x` | 1 cell | 2 | 3 | 5 | 10 | 20 |
|---|---|---|---|---|---|---|
| 531.5 | 2.321 | 2.106 | 1.995 | 1.912 | 1.863 | **1.840** |
| 14.00 | 2.438 | 2.268 | 2.169 | 2.107 | 2.066 | **2.045** |
| 5.653 | 1.535 | 1.430 | 1.383 | 1.355 | 1.335 | **1.324** |
| 3.008 | 1.231 | 1.163 | 1.134 | 1.116 | 1.104 | **1.097** |
| 1.026 | 0.937 | 0.899 | 0.892 | 0.887 | 0.881 | **0.877** |
| 0.425 | 0.928 | 0.942 | 0.946 | 0.943 | 0.939 | **0.936** |
| 0.385 | 0.927 | 0.942 | 0.946 | 0.942 | 0.936 | **0.932** |

The ratio is essentially FLAT in depth beyond 3 cells (worst case 2.169 -> 2.045,
6 %), so **more than 94 % of the excess is written INSIDE 3 Airy lengths**, where
the tail is physical (`Ai(3)/Ai(0)` is still 1e-3) and clipping it would trade an
energy error for a shape error.  The premise does not hold; the depth is not the
lever; the extrapolated `zeta` of section 6.1 is.

**The bound is nonetheless derived and reported.**  Re-fitting the module's own
two-branch band with `zeta = kappa u + q u^2` gives
`u* = 0.1 kappa / |q|` -- the radius at which the band's own measured curvature
makes the linear form 10 % wrong.  Over 86 planes of five fixtures whose band
passes the module's linear gate, `u* / l_airy` reads **min 6.93, 5th percentile
13.3, median 43.9, max 144**.  It is shorter than the 20-cell fill on WP-B7b's
own fixture F (17.2 at z = 4.400 mm, 14.9 at 4.450, 6.9 at 4.500) and on the K4
suite's plano-convex (18.1 at z = 4.3704 mm), and longer on the other four.
`_trace_meridional_fold` now returns it and the completion reports it as
`zeta_linear_range` alongside `zeta_curvature`, `zeta_linear_resid`,
`dark_fill_depth` and `l_airy`, so a caller can see the fill outrun the normal
form on their own optic.

**It is NOT applied to the fill**, for three measured reasons: the premise above;
`u*` is itself an extrapolation off the band and is anti-conservative by a factor
of ~20 against section 6.1's oracle reading (it says 50-75 Airy lengths on
fixture V where the oracle says the linear form is already 40 % wrong at 1-3);
and everything it would remove sits below `Ai(7)/Ai(0)` = 2.1e-06 of the ring
amplitude, i.e. 4e-12 in intensity -- so applying it would move the K4 suite's
own fixture and WP-B7b's fixture F for no physical gain.  Left switchable for
the maintainer: the value is in the diagnostics and the clip is one `min()`.

## 7. Bit identity, and what moved

Archive-to-archive, both builds: `git archive 96cb2096 lumenairy` extracted to a
scratch tree against the worktree, each read in a CHILD process with `cwd` and
`PYTHONPATH` set to the tree and `lumenairy.__file__` ASSERTED to live under it,
SHA-256 over `ndarray.tobytes()`.  Probe: `validation/probe_multibranch_zeta/bitid.py`;
JSON: `bitid_{parent,head}_{win,wsl}.json`.

13 fixtures, spanning the K4 suite's own plano-convex at N = 512 and N = 256 and
at `output_plane_distance = 0` (a fallback), the branch sum alone, `caustic_band
= 'plain'`, WP-B7b's fixture F at two planes where `u*` is shorter than the fill,
and VERIFY-B7b's fixture at four planes:

| build | identical | moved |
|---|---|---|
| Windows py3.14.6 / numpy 2.4.4 | **11 of 13** | 2 |
| WSL py3.12.3 / numpy 2.4.6 | **11 of 13** | 2 |

**Every mover classified.**  Both are the blow-up planes of fixture V:

| fixture | before | after |
|---|---|---|
| V uniform z = 1764.0 um | returned a field, grid power 5.9139e-04 W against a 1.7e-07 W launch (3.6 decades), 2 warnings | `RuntimeError` naming the ratio |
| V uniform z = 1770.0 um | returned a field, grid power 1.1035e-03 W, `fell_back=True`, 3 warnings | `RuntimeError` naming the ratio |

The two grid powers agree to all 17 printed digits between the two builds, which
is a second, independent check that the pathology is not a BLAS artefact.
Nothing else moved: in particular the branch sum at the SAME blow-up plane is
byte-identical (its reconstruction was not touched), and both fixture-F planes
are byte-identical (the `u*` bound is reported, not applied).

## 8. Tests

`tests/unit/test_audit2609_b7c_multibranch_envelope.py`, 10 ids, 15.3 s total on
Windows.  Every pathology claim is premise-gated on the running build's own
reading; every invariant is unconditional; no wall-clock assertion.

| id | shape |
|---|---|
| `..._the_energy_reading_and_its_decision_reach_the_uniform_diagnostics` | unconditional; the four keys exist and the ratio is the branch sum's OWN number to the bit |
| `..._a_fallback_also_carries_the_energy_decision` | unconditional; the fallback path returns the branch-sum field, so the reading travels with it |
| `..._a_blown_up_multibranch_is_refused_not_passed_through` | premise-gated two-sided: where a rung exceeds the bar, `pytest.raises` and the message must NAME the ratio and a member that works; where no rung does, the INVARIANT is asserted instead (nothing outside the band may be RETURNED).  The healthy plane below the window is asserted untouched on both arms |
| `..._the_refusal_is_never_the_first_diagnostic` | the constant relationship unconditionally; the "the branch sum already warned" claim premise-gated |
| `..._the_public_caustic_uniform_entry_point_refuses_too` | premise-gated; `apply_real_lens_traced(caustic='uniform')` is not a way past |
| `..._the_refusal_band_is_coupled_to_the_constants_it_was_derived_on` | build-free; the coupling and the measured gap with both margins |
| `..._the_zeta_bar_sits_in_the_measured_transition` | build-free; 5.653 < 8.0 < 11.05 with both margins |
| `..._the_dark_fill_depth_is_not_the_energy_lever` | solves `Ai(x)/Ai(0) = 1e-12` on the running build; asserts the fill still covers the representable tail and that 3 Airy lengths is not a safe cut |
| `..._the_completion_gains_energy_where_the_hand_off_does_not` (x2) | the member recommendation through ENERGY, which needs no model; asserts the plane's side of the bar first, so a build whose fold geometry moved cannot pass silently |

## 9. Verification state

| run | Windows py3.14.6 / numpy 2.4.4 | WSL py3.12.3 / numpy 2.4.6 |
|---|---|---|
| `b7c` + `b7_asymptotic` + `b7b_caustic_routing` + `niche_k4_uniform_caustic` + `pmm2d_staggered_nonuniform` + `niche_audit_w9_dispatch2` | **151 passed** (282 s) | **151 passed** (see section 10) |
| `a3_caustic_siblings` + `a3_verify_traced` + `niche_k1_kmah_caustic` + `niche_r2_pearcey_cusp` + `niche_r5_gbd_vector_catastrophe` + `niche_audit_w3_elements` + `niche_audit_w4_input_kind` | **373 passed** (499 s) | -- |
| census / public-API / dispatcher-pin doc-consistency / except-budget / a17 history lint + relocation | **768 passed, 1 failed** | -- |
| `ruff check lumenairy/ tests/ validation/probe_multibranch_zeta/` (WSL) | -- | **All checks passed** |

The single red, `test_public_api.py::test_installed_metadata_version_matches_source_version`,
is a BOX condition and not this package's: the Windows interpreter's installed
distribution metadata reads `lumenairy==3.7.8` (a stale editable install
pointing at `D:\...\Lumenairy`) against a source `__version__` of 5.47.0.  It
reproduces identically in a `git archive` tree of the audit base, where no
change of mine exists, because the metadata is read from site-packages and not
from the tree.  `pip install -e .` on that interpreter is the fix; it is not
mine to run.

## 10. Open items

1. **The branch sum's quadrature itself** (P2, deferred with the measurement).
   The point-sampled write is unbiased only while the mapped triangles are
   spread over many pixels.  An area-weighted splat would be the real repair and
   would make `ray_subsample` a convergence knob instead of a divergence one; it
   moves every multibranch field near a caustic and needs its own work package.
2. **A second fold parameter for `zeta(r)`** (P3), section 6.1: measured,
   recommended, not a one-step change.
3. **Whether the `u*` clip should be applied to the dark fill** (maintainer
   decision), section 6.2: derived, reported, measured to be physically
   inconsequential and to move two existing fixtures.
4. **Whether `caustic='uniform'` stays the fold member** (maintainer decision),
   section 5: both published orderings reproduce, the margins are under 0.012 of
   fidelity, and the oracle is not neutral between the completion and the
   hand-off.
5. **Whether the branch sum should REFUSE as well as warn** at a gross blow-up,
   as it already refuses the mirror case (the identically-zero field).  Not done
   here: it would move `caustic='multibranch'` for every caller, where the
   completion's refusal moves only a layer that was reporting the best case on a
   broken field.
6. **What could not be measured.**  The oracle's non-neutrality between the
   hand-off and the completion (section 5) bounds the member question; settling
   it needs an oracle whose boundary field is not geometrical -- a direct FDTD or
   BEM solve of the lens, or a measurement.  And no fixture in this study
   produced a multibranch `power_ratio` between 1.25 and 18.86, so the refusal
   bar's placement inside that gap rests on the gap's emptiness over 38 planes,
   not on a rung inside it.
