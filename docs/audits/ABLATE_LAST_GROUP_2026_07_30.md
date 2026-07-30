# ABLATION: which construction inside `apply_real_lens_traced` loses design 121's last group?

**2026-07-30.  Differential experiments only -- no library change is proposed
here, and none was left behind (`git diff --stat lumenairy/` is clean).**
Branch `fix/pmm-union-grid-conditioning` @ `7f45874`.

Companion to `SCOPE_TILTED_COARSE_LEG_TRANSPORT_2026_07_30.md`, which localises
the whole per-order loss to ONE `apply_real_lens_traced` call (the chain's sixth
element pass, `Lens S25-S27`) and leaves three constructions untested: the
Newton initial guess / bracket at large decentre, the entrance-pullback DOMAIN,
and `preserve_input_phase='remap'`'s resampling.  This document ablates them.

New scripts, all in `validation/repro_traced_carrier_121/` and all named
`ablate_*`:

```
ablate_recon.py             pre-flight: is a per-surface 'decenter' an EXACT rigid translation?
ablate_last_group_121.py    the ablation driver (monkeypatches the element for the LAST group only)
ablate_gapleg_121.py        splits a group's step into [gap leg | element pass]   -- INVALIDATED, see S6
ablate_exitna_transpose.py  standalone: the measured-exit-NA mask transpose (S7)
```

---

## 0. Headline

**Re-centring the element's numerical frame onto the chief ray recovers
0.0 of the 24 EE3 points.**  Measured on the REAL design-121 last group with
EXACT physics (a rigid translation of beam AND group, so nothing is
approximated), at the order the scope doc uses:

| chief-ray decentre at the element's entrance | EE3 % , order (-4,-2) |
|---|---|
| 3.372 mm = 1.079 w  (**shipped**) | **66.24** |
| 2.526 mm = 0.808 w | 66.23 |
| 1.694 mm = 0.542 w | 66.22 |
| 0.847 mm = 0.271 w | 66.18 |
| **0.015 mm = 0.005 w  (on the grid centre)** | **66.16** |

and, extending the sweep the other way on the same instrument,

| 3.386 mm = 1.083 w (mirrored to +x,+y) | 66.24 |
| 5.051 mm = 1.616 w | 66.10 |
| 6.744 mm = 2.158 w | 66.81 |

Flat from 0 to 2.16 beam radii.  The oracle ceiling is 89.80-90.00.

**Therefore scope doc S6 Option A ("decouple the element's numerical frame from
the optical axis", 3-6 days) is NOT worth doing.**  It would buy nothing.  The
scope doc's ranked candidate #1 is refuted in the half that named the decentre;
what survives is "the last group's ELEMENT PASS", with the tilt (or the field
the tilt produces), not the decentre, carrying the field-angle dependence.

Everything else tested here is also null, at the 0.1-0.2 point level:
`preserve_input_phase`'s shipped `'remap'` is the *best* of the five available
configurations by 55 points (S2); the Newton seed and tolerance are exonerated
by seeding from the traced map itself and tightening the tolerance 100x (S3);
and the 81.2 % Newton non-convergence is now VERIFIED benign rather than assumed
(S3.1: 0.00 % of the in-domain pixels).

Two incidental findings that are not the 24 points but are real:

* **`apply_real_lens_traced`'s measured exit NA reads its amplitude mask
  TRANSPOSED** (S7).  Proven on a biconic where x and y NA differ by 2x: the
  two reported values are swapped.  Design 121's last group therefore reports
  `NA_exit=0.3633` where the transpose-immune reading of the same beam is
  0.2912 -- and `_exit_na_out` gates the chain's exact-final-leg routing.
* **Mid-leg hand-off is not a valid measurement instrument** (S6).  My own
  attempt to split the group-5 step into [gap leg | element pass] produced a
  spectacular-looking 41-point loss on the leg; its control on a leg the scope
  doc bounds at -0.54 points reproduced 7 points of the same artefact, and on
  axis it claims a 55.36 input produced an 87.99 output.  Discarded.  Recorded
  because the number is seductive and someone will re-derive it.

---

## 1. Method, and why the differences are trustworthy

**Scoring.**  `hybrid_localize_121.main()` at `NMIN=NMAX=6`, unchanged, so every
EE3 below is on the scope doc's own convention and directly comparable with its
knockout table (`NOUT=61 DXO=0.4 NL=121 RN=1024 RS=4`, whose shipped baseline is
66.24 at order (-4,-2)).  EE3 = encircled energy inside a 3 um RADIUS about the
image-plane intensity centroid, from an exact skew ray trace of the chain's
hand-off field plus a first Rayleigh-Sommerfeld integral.

**Intervention.**  `lumenairy.elements.apply_real_lens_traced` is monkeypatched
(the chain resolves it from the module at call time) and only the call whose
prescription is `Lens S25-S27` is transformed.  All five earlier element passes
and every coarse leg run exactly as shipped.

**Sampling adequacy.**  Two separate statements, and they must not be conflated:

* The ABSOLUTE value 66.24 carries the scope doc's own +-0.5 point error bar.
  At `n_chain = 6` the hand-off envelope's per-pixel residual-phase step is
  pinned at 3.14 rad (at grid Nyquist) and does not fall under Fourier
  upsampling; the doc's `UP` 1/2/4 and two-readout-grid spread is +-0.5.
* Every number in this document is a DIFFERENCE at FIXED readout
  configuration, where that common-mode error cancels.  The differential floor
  is **0.00 points, not +-0.5**: the null intervention (`MODE=recentre:0`,
  which installs the whole translation machinery with a zero shift) reproduces
  the shipped baseline exactly -- EE3 66.24, EE6 91.77, EE12 99.73, FWHM
  4.037 um, every digit.  The pipeline is deterministic.

**Corroboration at a second order.**  The recentring null is reproduced at
order (-4,0): 70.19 -> 70.11, i.e. 0.0 of that order's 19.8 points.

### 1.1 Why the re-centring probe is EXACT, and the adversarial checks on it

A lateral translation of BOTH the beam and the group is an exact symmetry of the
sequential trace.  It is applied as

1. **the input field is shifted by an INTEGER number of pixels.**  That is a
   pure array copy: sample values are untouched, so no phase is resampled,
   nothing is interpolated, and nothing can alias.  (This is deliberately NOT
   the sub-pixel Fourier shift the chain itself uses -- a Fourier shift of a
   field carrying a 2.78-pi-per-pixel tilt ramp would be exactly the artefact
   the brief warns about.)  The element's grid convention is
   `(arange(N) - N/2) * dx` (`_lens_traced.py:3394`), identical to the chain's,
   so an integer roll is an exact translation in the element's own coordinates.
2. **the shift is NON-periodic (zero-filled)**, so nothing wraps round from the
   far edge.  Measured power discarded: **8.2e-9** of the beam at the full
   shift (1.9e-8 at the largest shift in the table, 182 px).  On the return
   shift: 0.0.
3. **every surface of the group gets the matching field-frame `decenter`.**
   Verified exact in `ablate_recon.py`: over an 81-ray bundle at 46+23 mrad
   through the real group-5 prescription, translating beam and group together
   moves the exit positions by exactly `t` to **1.3e-18 m** and leaves the
   traced OPL invariant to **1.0e-17 m = 8e-12 waves**, with 81/81 rays alive
   in both arms.  (Group 5 carries no per-surface `semi_diameter`, and the
   element pops `aperture_diameter` before building its trace surfaces, so the
   trace is unvignetted in both arms -- the one place a field-frame decenter
   would NOT be exact, since the semi-diameter test does not move with it.)
4. **the `TiltedCarrier`'s `(x0, y0)` moves with them**, and the exit field is
   shifted back by the same integer vector.

Residual, unavoidable, and reported: the integer-pixel rounding leaves the beam
**14.9 um = 0.005 beam radii** from the grid centre at the full shift
(dx = 33.211 um).  That is 0.5 % of the decentre being removed.

`w` in the tables is the beam radius the ELEMENT itself measures at this plane
(`_input_beam_amp_radius` about the carrier centre) = **3.1253 mm** -- the same
quantity the decentre guard uses, which is why 3.3723 mm reads as 1.079 w.  It
is NOT the chain stage's `w` (1204.9 um), which is the group's EXIT radius.

---

## 2. Experiment 1 -- `preserve_input_phase` and `remap_sampling`

Order (-4,-2), applied to the LAST group only via `traced_kwargs`.

| configuration | EE3 % | EE6 % | FWHM um | vs shipped |
|---|---|---|---|---|
| `'remap'` + `remap_sampling='full'` -- **SHIPPED** | **66.24** | 91.77 | 4.037 | 0 |
| `'remap'` + `remap_sampling='lattice'` | 67.87 | 94.07 | 4.662 | +1.63 |
| `False` (with `amplitude_model='ray_density'`) | 9.06 | 31.42 | 18.408 | -57.2 |
| `True` (with `amplitude_model='ray_density'`) | 10.78 | 37.03 | 15.030 | -55.5 |
| `True` + `amplitude_model='screen'` (legacy pair) | 8.41 | 25.05 | 7.642 | -57.8 |
| `False` + `amplitude_model='screen'` | 6.05 | 20.52 | (no core) | -60.2 |

**`preserve_input_phase='remap'` is not the defect -- it is the only thing
holding the number up.**  Turning it off in any of the four available ways
costs 55-60 EE3 points, i.e. the transported input residual phasor carries
essentially the whole image.  `remap_sampling='lattice'` reads 1.6 points
higher on EE3 but with a **15 % WIDER core** (FWHM 4.662 vs 4.037 um), so it
is not a clean win and does not look like a recovered defect; the scope doc's
+1.6 is reproduced exactly.

The 24 points therefore are not "remap resampling error" in the sense of a
knob set wrong.  If the remap is implicated at all it is through the ACCURACY
of the entrance pullback it reuses -- which S3 and S4 test directly, and both
are null.

---

## 3. Experiment 2 -- the Newton seed, bracket and tolerance

Temporary (now reverted) hook in `_invert_newton`, env-gated:

* `LT_ABLATE_SEED=truth` -- replace the paraxial-magnification seed
  `xe = x_out / M_x` (which carries NO offset, so a tilted congruence's
  `x_out = (A + B/R) x_in + B L` starts B*L/(A+B/R) away from the answer) by a
  cKDTree nearest-neighbour lookup in the ALREADY-TRACED forward map
  `(x_out_grid, y_out_grid)`, i.e. seeded from truth.
* `LT_ABLATE_TOLF=f` -- multiply the convergence tolerance `0.01 * dx` by `f`.

Order (-4,-2), last group:

| ablation | EE3 % |
|---|---|
| shipped (paraxial-magnification seed, tol 3.32e-7 m, 12 iters) | 66.24 |
| seed from the TRACED forward map | 66.24 |
| tol x 0.01 (3.32e-9 m) + `newton_max_iters=200` | 66.24 |
| both | 66.24 |

Identical to every digit reported (EE3 66.24, EE6 91.77, EE12 99.73, FWHM
4.037).  **The Newton seed, bracket and tolerance are exonerated.**  The seed
is not merely "good enough" -- replacing it with the exact answer changes
nothing, which is the strong form.

### 3.1 The 81.2 % non-convergence is benign -- verified, not assumed

Scope doc open item 4.  Same temporary hook, `LT_ABLATE_DIAG=1`, counting how
many of the unconverged pixels lie inside the domain where the OPL is actually
used (`x^2 + y^2 <= (0.99 launch_radius)^2`; everything outside is NaN'd and
contributes zero to the exit field regardless):

| group | unconverged | in-domain pixels | unconverged **and** in-domain |
|---|---|---|---|
| 0 `S13-S14` | 17575 (26.8 %) | 37033 (56.5 %) | 0 (0.00 %) |
| 1 `S15-S16` | 15136 (23.1 %) | 38945 (59.4 %) | 0 (0.00 %) |
| 2 `S17-S19` | 3527 (5.4 %)   | 44045 (67.2 %) | 0 (0.00 %) |
| 3 `S21-S22` | 1934 (3.0 %)   | 48185 (73.5 %) | 0 (0.00 %) |
| 4 `S23-S24` | 24666 (37.6 %) | 38169 (58.2 %) | **1221 (3.20 %)** |
| 5 `S25-S27` | **53228 (81.2 %)** | 8359 (12.8 %) | **0 (0.00 %)** |

At the failing group every single unconverged pixel is out of domain.  The
roadmap's claim was right and is now measured.  The one place the warning has
teeth is group 4, where 3.2 % of in-domain pixels do hit the iteration cap --
and `newton_max_iters=60` was already shown to move nothing there either.

---

## 4. Experiment 3 -- the entrance-pullback / launch DOMAIN (the headline)

See S0 for the table.  The intervention moves ALL THREE axis-centred
constructions onto the beam at once -- the launch square
`xs_in = linspace(-launch_radius, launch_radius, n_launch)`, the Newton
`bound` / out-of-domain disc `launch_radius * 0.99`, and the ray-fit disc --
because a rigid translation moves the physics rather than the code.  That makes
it a strictly STRONGER probe than the proposed `beam_centre` fix, which by its
own docstring moves only the fit disc.

It recovers **0.00 +- 0.08 points of 24** at (-4,-2) and **0.00 +- 0.08 of
19.8** at (-4,0).

### 4.1 Controls that separate the frame from the fit branch

Re-centring also silently swaps the element's fit branch (an off-centre disc
uses the D1 WEIGHTS and `_DECENTRED_FIT_POLY_ORDER=10`; a concentric one takes
the historical hard NaN mask at `newton_poly_order=6`).  Both halves were
pinned so the null cannot be a cancellation:

| arm | fit branch | fit order | EE3 % |
|---|---|---|---|
| shipped (decentred) | weights | 10 | 66.24 |
| shipped + `decentred_fit_poly_order=6` | weights | 6 | 66.06 |
| shipped, `_FIT_DISC_OUTSIDE_WEIGHT_REL = 0` | hard NaN mask | 10 | 66.17 |
| re-centred | hard NaN mask | 6 | 66.16 |
| re-centred + `newton_poly_order=10` | hard NaN mask | 10 | 66.15 |

Spread 0.18 points across the whole 2x2.  The fit branch, the fit order and the
frame are jointly worth nothing here.

### 4.2 What this does to the scope doc's section 4.2

The scope doc's synthetic decentre x tilt probe found a cliff at 1.778 probe-w
and flagged that it under-read the design point by 5x.  Run on the REAL last
group with the real field and exact physics, there is no cliff at all out to
2.16 w.  **The synthetic probe's cliff does not exist on this element at this
design point**; whatever it measured (its own reference is aberrated to
EE3 5-33 %) is not the mechanism.

---

## 5. Experiment 4 -- decentre versus high exit NA, on the real group

The scope doc's section 4.1 argues the last group is unique in being BOTH
> 1 w decentred AND at high exit NA, and that neither alone reproduces the
failure.  With decentre now swept on the real group at FIXED exit NA, field and
tilt, the decentre axis is flat (S0).  The three corners therefore read:

| case | chief-ray decentre | exit NA (element, transpose-immune) | EE3 loss vs oracle |
|---|---|---|---|
| order (0,0) -- NA, no decentre, no tilt | 0 | 0.2879 | **-2.34** (89.94 -> 87.99) |
| order (-4,-2) re-centred -- NA + tilt, no decentre | 0.005 w | 0.2912 | **-23.6** (89.80 -> 66.16) |
| order (-4,-2) shipped -- NA + tilt + decentre | 1.079 w | 0.2912 | **-23.6** (89.80 -> 66.24) |
| groups 3 / 4 -- decentre + tilt, LOW NA | 0.685 / 0.890 w | 0.0875 / 0.1995 | -0.11 / -0.54 |

**The variable that carries the field-angle dependence is the TILT (46.1 +
23.0 mrad), not the decentre.**  At zero decentre the tilted order still loses
21.3 points more than the on-axis one, at the same exit NA, through the same
element, on a grid where the beam sits where the on-axis beam sits.

### 5.1 One tilt-specific construction, tested and refuted

`TiltedCarrier`'s eikonal is a SPHERE PLUS A LINEAR RAMP,
`W = sgn(R)(sqrt(u^2+v^2+R^2) - |R|) + L u + M v`
(`_lens_traced.py:1588`), which is not a solution of the eikonal equation: the
exact eikonal of a point source displaced so the chief ray carries `(L, M)` is
`W = sgn(R)(sqrt((u + sgn L D)^2 + (v + sgn M D)^2 + R^2) - D)`, `D = |R|/N`.
The two differ by a COMA term LINEAR in the field angle -- exactly the shape
the scope doc's S2.2 infers (spurious rms wavefront error growing linearly with
field angle).  Measured at group 5's entrance (R = -21.139 mm, L,M =
-46.1,-23.0 mrad):

| u (mm) | 1.00 | 3.13 (= 1 w) | 6.00 |
|---|---|---|---|
| eikonal difference (waves) | 0.10 | **1.78** | 10.02 |
| launch direction-cosine difference | 3.2e-4 | **2.0e-3** | 5.8e-3 |

That is a 2 mrad launch-direction error at one beam radius on a
diffraction-limited relay, and it is the right functional form.  It is
nevertheless **NOT the mechanism**: swapping in the exact eikonal (temporary,
env-gated, now reverted) makes the answer WORSE.

| eikonal | EE3 (-4,-2) | EE3 (-4,0) |
|---|---|---|
| shipped (sphere + linear ramp) | 66.24 | 70.19 |
| exact displaced point source | **62.89** | **67.48** |

Caveat, stated because it weakens the refutation: the hook is global, so all
six element passes changed, and the element's `W` then no longer matches the
chain's own carrier bookkeeping (`_radial_carrier_phase` + `_tilt_ramp`
reconstruct the same sphere-plus-ramp form), which introduces a mismatch of its
own.  What is safe to conclude is that the exact eikonal does not RECOVER the
loss; a properly-scoped version would change the chain and the element
together.  **This is the most promising surviving lead and I recommend it as
the next experiment** (S9).

---

## 6. My own instrument that FAILED, and the control that caught it

I attempted to split the group-5 step into its two halves -- the 3.3233 mm
free-space coarse leg to the front vertex, and the element pass -- since the
scope doc's bisection is by GROUP and a group's step contains both.
`ablate_gapleg_121.py` runs the chain over groups 0..4 plus a FRACTION `f` of
the leg, and lets the exact-ray oracle finish.

The reading looked like a major finding:

| f (fraction of the 3.3233 mm leg done by the chain) | 0.0 | 0.001 | 0.01 | 0.1 | 0.3 | 0.6 | 1.0 |
|---|---|---|---|---|---|---|---|
| EE3 % , order (-4,-2) | 90.00 | 89.98 | 89.53 | 85.75 | 73.73 | 59.71 | **48.43** |

Continuous, monotone, and it survived every convergence check I threw at it:
`CLIP` 2.0 / 2.5 / 3.0 / 3.5 -> 48.43 / 48.43 / 48.43 / 48.42; Fourier
upsampling `UP` 2 -> 48.43; and at `RN=2048` (dx halved everywhere) the on-axis
reading moves 55.36 -> 55.38 while the hand-off envelope's per-pixel phase step
falls to 0.4145 rad, i.e. **band-limited, no aliasing to blame**.

It is still wrong, and two independent controls prove it:

1. **On axis, the same split reads `n5` 90.18, `gap` 55.36, `n6` 87.99.**  An
   element pass cannot turn a 55 % field into an 88 % one.
2. **Splitting an INNOCENT leg reproduces the artefact.**  Group 4's entrance
   leg costs -0.54 EE3 points by the scope doc's own bisection.  Split it:

   | | n5 (chain 0..3) | f = 0.5 | f = 1.0 | n6 (chain 0..4) |
   |---|---|---|---|---|
   | EE3 % , (-4,-2) | 89.50 | 87.57 | **82.90** | 90.00 |

   The endpoints bracket at ~90 and the middle dips 7 points.  The dip is the
   instrument.

**Conclusion: a hand-off taken at a group's FRONT VERTEX (mid-leg) is not a
valid measurement, while a hand-off at a group's EXIT plane is.**  I did not
determine why.  The plausible difference is that my re-launch needs the
residual phase GRADIENT to set ray directions, whereas the element needs only
the residual PHASOR pointwise (its directions come from the analytic carrier)
-- so a residual that the chain represents perfectly well can still be
un-differentiable on that grid.  If so, the scope doc's group-level bisection
stands, but it **cannot be refined to leg-versus-element granularity with this
class of instrument**, and the 24 points remain attributed to the pair
[group-5 entrance leg + group-5 element pass] jointly rather than to the
element pass alone.  That is a genuine loosening of the scope doc's S2 claim.

---

## 7. Incidental defect: the measured exit NA reads its mask TRANSPOSED

The launch lattice is built `indexing='ij'`, so ray `r = i*n + j` sits at
entrance `(x = xs_in[i], y = xs_in[j])`.  The amplitude significance mask is

```python
_amp = np.abs(E_in)[np.ix_(_ray_iy, _ray_ix)]      # [a, b] = |E_in| at (y=xs_in[a], x=xs_in[b])
_sig = (_amp >= np.exp(-4.0) * _amp.max()).ravel() & final.alive
_na_exit = float(np.sqrt(final.L[_sig] ** 2 + final.M[_sig] ** 2).max())
```

`_amp.ravel()` is in `(y, x)` order and `final.L` is in `(x, y)` order.  The
mask is applied transposed.  Invisible for any beam symmetric under x <-> y,
which is every case the element has been exercised on.

`ablate_exitna_transpose.py` proves it on a BICONIC singlet (`radius` 20 mm,
`radius_y` 40 mm, so the x and y exit NA differ by 2x), with a small collimated
Gaussian parked first at (+2, 0) mm and then at (0, +2) mm:

| beam position | element `na_exit` | direct trace of that beam | direct trace of the OTHER beam |
|---|---|---|---|
| (+2.00, 0) mm | **0.03383** | 0.06840 | **0.03404** |
| (0, +2.00) mm | **0.06693** | 0.03404 | **0.06840** |

The reported values match the OTHER beam's truth to 0.6 % / 2.0 % (the residual
is my probe disc sampling the launch lattice slightly differently), and miss
their own by 2x.

**Why it matters here.**  Design 121's last group reports `NA_exit = 0.3633` at
order (-4,-2) -- a number the scope doc quotes as a headline characteristic of
the failing group ("uniquely at high exit NA").  Rigidly translating the SAME
beam onto the grid centre, where the transpose is harmless, the same element
reports **0.2912**, essentially the on-axis 0.2879.  So the 0.3633 is 25 %
overstated, and the "the tilted order is at higher exit NA" premise is partly
an artefact.  It is not purely cosmetic: `_lens_traced.py:4322` records that the
chain's `on_tilt_exact_grid` guard reads `_exit_na_out` to decide exact-final-leg
routing.

Severity P2, diagnostic + routing, not the 24 points.  Not fixed here (this
brief is differential-experiment only, and the fix belongs with a regression
test).

---

## 8. Full ablation table, order (-4,-2), `n_chain = 6`

Oracle ceiling 89.80-90.00.  Shipped 66.24.

| # | ablation | EE3 % | recovered |
|---|---|---|---|
| 0 | shipped | 66.24 | -- |
| 0' | null intervention (`recentre:0`) | 66.24 | 0.00 (determinism check) |
| 1 | re-centre the numerical frame on the chief ray | 66.16 | **-0.08** |
| 2 | re-centre + `newton_poly_order=10` | 66.15 | -0.09 |
| 3 | decentre 0.542 w / 0.808 w (partial re-centre) | 66.22 / 66.23 | -0.02 / -0.01 |
| 4 | decentre 1.616 w / 2.158 w (INCREASED) | 66.10 / 66.81 | -0.14 / +0.57 |
| 5 | off-centre fit forced to the concentric hard NaN mask | 66.17 | -0.07 |
| 6 | `decentred_fit_poly_order=6` | 66.06 | -0.18 |
| 7 | Newton seeded from the traced forward map | 66.24 | 0.00 |
| 8 | Newton tol x 0.01 + 200 iterations | 66.24 | 0.00 |
| 9 | 7 + 8 together | 66.24 | 0.00 |
| 10 | `remap_sampling='lattice'` | 67.87 | +1.63 (but FWHM 4.04 -> 4.66) |
| 11 | `preserve_input_phase=False` | 9.06 | -57.2 |
| 12 | `preserve_input_phase=True` | 10.78 | -55.5 |
| 13 | legacy pair (`True` + `screen`) | 8.41 | -57.8 |
| 14 | `False` + `screen` | 6.05 | -60.2 |
| 15 | exact tilted-point-source carrier eikonal | 62.89 | -3.35 |

At order (-4,0): shipped 70.19, re-centred 70.11.  On axis: shipped 87.99, no
decentre to remove.

---

## 9. What I could NOT determine

1. **Which construction IS responsible.**  Every candidate on the scope doc's
   remaining surface is now null.  Combined with its own S4.3 (ten knobs
   already exonerated), the loss is inside `apply_real_lens_traced`'s last-group
   pass but is not attributable to: the decentre, the launch/Newton/fit domain
   centring, the fit branch, the fit order, the Newton seed, the Newton
   bracket, the Newton tolerance, the Newton iteration cap, the remap sampling
   resolution, or `preserve_input_phase`'s choice.
2. **Whether the entrance LEG shares the blame.**  S6's instrument failed its
   control, so the 24 points are attributable to [group-5 leg + group-5 element
   pass] jointly, not to the element pass alone.  The scope doc's stronger
   phrasing is not supported by an instrument that survives its own control.
3. **Why a mid-leg hand-off is invalid while an exit-plane hand-off is valid**
   (S6).  A hypothesis is offered (gradient-of-residual vs pointwise-residual)
   and is untested.
4. **Whether the sphere-plus-ramp carrier eikonal matters when the CHAIN's
   matching convention is changed with it** (S5.1).  The element-only swap is
   worse; a joint swap was out of scope (it touches `carrier.py`).  This is the
   single most promising surviving lead: the defect it represents has the right
   functional form (linear in field angle), the right magnitude (2 mrad of
   launch-direction error at 1 w), and the right group selectivity
   (`k L NA_in^2 w` = 0.08 / 0.71 / 2.7 waves at groups 3 / 4 / 5, against
   measured losses of 0.11 / 0.54 / 24 EE3 points).
5. **The on-axis 2.34-point floor.**  Untouched here, as in the scope doc.
6. **Whether the exit-NA transpose has any effect on the shipped 121
   acceptance** beyond the mis-reported diagnostic -- i.e. whether it has ever
   flipped an `on_tilt_exact_grid` routing decision.  Not tested.

---

## 10. Reproduction

All commands from `validation/repro_traced_carrier_121/`.  Each
`ablate_last_group_121.py` run is ~25 s after the chain-A cache is warm
(`_chainA_1024_2000nm_rs4.npz`, written on first use).

```bash
# 0. Pre-flight: is a per-surface 'decenter' an exact rigid translation?
python ablate_recon.py

# 1. Baseline (must reproduce the scope doc's 66.24) and the null intervention.
ORD="-4,-2" MODE=base            python ablate_last_group_121.py
ORD="-4,-2" MODE="recentre:0"    python ablate_last_group_121.py

# 2. THE HEADLINE: re-centre the element's numerical frame (experiment 3).
for f in 0 0.25 0.5 0.75 1.0 2.0 -0.5 -1.0; do \
  ORD="-4,-2" MODE="recentre:$f" python ablate_last_group_121.py; done
ORD="-4,0"  MODE=recentre python ablate_last_group_121.py

# 3. preserve_input_phase / remap_sampling (experiment 1).
for t in "remap_sampling=lattice" "preserve_input_phase=False" \
         "preserve_input_phase=True" \
         "preserve_input_phase=True;amplitude_model=screen" \
         "preserve_input_phase=False;amplitude_model=screen"; do \
  ORD="-4,-2" MODE=base TKW="$t" python ablate_last_group_121.py; done

# 4. Fit-branch controls.
ORD="-4,-2" MODE=recentre TKW=newton_poly_order=10 python ablate_last_group_121.py
ORD="-4,-2" MODE=hardmask                          python ablate_last_group_121.py
ORD="-4,-2" MODE=base TKW=decentred_fit_poly_order=6 python ablate_last_group_121.py

# 5. The exit-NA mask transpose (no library edit needed).
python ablate_exitna_transpose.py

# 6. The INVALID gap-leg split, and the control that invalidates it.
ORD="-4,-2"        WHICH="n5,gap:0.0,gap:0.1,gap:0.3,gap:0.6,gap:1.0,n6" python ablate_gapleg_121.py
ORD="-4,-2" GRP=4  WHICH="n5,gap:0.5,gap:1.0,n6"                          python ablate_gapleg_121.py   # control
ORD="0,0"          WHICH="n5,gap:1.0,n6"                                  python ablate_gapleg_121.py   # control
```

### 10.1 The TEMPORARY library edits (reverted -- reproduce by re-applying)

Experiments 2 (S3) and 5.1 (S5.1) needed `lumenairy/elements/_lens_traced.py`
edits.  Both were env-gated and both are REVERTED; `git diff --stat lumenairy/`
is empty.  To reproduce:

1. **Newton seed / tolerance / domain diagnostic** -- in `_invert_newton`,
   immediately after `tol = 0.01 * dx` (~line 4809), add: if
   `os.environ['LT_ABLATE_SEED'] == 'truth'`, replace the seed
   `xe = x_w_flat * inv_M_x`, `ye = y_w_flat * inv_M_y` by a
   `scipy.spatial.cKDTree` nearest-neighbour lookup of `(x_w_flat, y_w_flat)`
   in the finite entries of `(x_out_grid, y_out_grid)`, seeding
   `(xs_in[i], xs_in[j])`; and multiply `tol` by
   `float(os.environ.get('LT_ABLATE_TOLF', '1.0'))`.  For S3.1, after the
   Newton loop print the counts of `active`, of
   `xe^2 + ye^2 <= (0.99*launch_radius)^2`, and of their intersection.
   Then: `LT_ABLATE_SEED=truth LT_ABLATE_TOLF=0.01 LT_ABLATE_DIAG=1 \
   TKW=newton_max_iters=200 ORD="-4,-2" MODE=base python ablate_last_group_121.py`
2. **Exact tilted-sphere eikonal** -- in `_tilted_carrier_parts` (~line 1586),
   before the shipped `rho = ...` lines, add the `LT_ABLATE_EIK=exact` branch

   ```python
   N_ = np.sqrt(max(1.0 - L*L - M*M, 1e-300)); D_ = abs(s) / N_
   uu = u + sgn*L*D_; vv = v + sgn*M*D_
   Wr = np.sqrt(uu*uu + vv*vv + s*s)
   return sgn*(Wr - D_), sgn*uu/Wr, sgn*vv/Wr
   ```

   Then: `LT_ABLATE_EIK=exact ORD="-4,-2" MODE=base python ablate_last_group_121.py`
