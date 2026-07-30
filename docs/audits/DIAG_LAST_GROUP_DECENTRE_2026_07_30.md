# DIAGNOSIS: which construction in `apply_real_lens_traced` carries design
# 121's last-group loss -- and the answer is NONE of them

**2026-07-30.  Option C of `SCOPE_TILTED_COARSE_LEG_TRANSPORT_2026_07_30.md`
section 6 (instrument the element).  DIAGNOSIS ONLY -- no library change is
proposed in code here, and the temporary instrumentation used to map the
Newton failure is reverted (section 8).**

Branch `fix/pmm-union-grid-conditioning` @ `7f45874`.  All scripts are
`validation/repro_traced_carrier_121/wfe_probe_*.py` and are LOCAL-ONLY (they
need the 121 `.zmx` and the design-study runner).

---

## 0. Headline

1. **The element is not the defect.**  Measured against the exact skew ray
   trace on its own returned field, pixel by pixel, `apply_real_lens_traced`
   reproduces design 121's last group to **0.011 waves rms on axis and 0.036
   waves at order (-4,-2)** (Marechal Strehl 0.995 / 0.951).

2. **The error is not decentre-driven.**  With the input residual held FIXED
   and only the decentre swept, the element's exit wavefront error is FLAT:
   0.0377 / 0.0373 / 0.0358 waves at 0.000 / 0.500 / 1.079 w.  With the
   residual set to unity (a pure congruence) the error is **0.00017 -> 0.00072
   waves over 0 -> 1.5 w** -- three orders of magnitude below what the loss
   needs.  The scope doc's decentre hypothesis is **refuted**.

3. **All three "remaining untested surfaces" (scope doc section 7 item 1) are
   clean.**  The Newton initial guess / bracket, the entrance-pullback domain
   and `preserve_input_phase='remap'`'s resampling of the input phasor were
   measured separately and read 0.0002-0.0004 waves each, at every decentre and
   every order.

4. **The loss is in the COARSE LEG, not the element pass.**  The hybrid
   localisation's `n = 5 -> n = 6` step is the cost of TWO operations, because
   its `n = 5` hand-off is taken at group 4's EXIT: the chain's wave transport
   across the 3.3233 mm gap into group 5, AND the element pass.  Splitting them
   with the same machinery on both sides:

   | order | A: oracle does the gap | B: chain does the gap | **LEG** | C: chain does gap + element | **ELEMENT** |
   |---|---|---|---|---|---|
   | (0,0)   | 90.17 | 89.16 | **-1.01**  | 87.99 | **-1.17** |
   | (-2,0)  | 90.19 | 85.31 | **-4.88**  | 83.15 | **-2.16** |
   | (-4,0)  | 90.15 | 73.97 | **-16.18** | 70.19 | **-3.78** |
   | (-4,-2) | 89.98 | 69.28 | **-20.70** | 66.24 | **-3.04** |

   **87 % of the 23.7-point step at (-4,-2) is the last coarse leg.**  The
   element's 3.04 points are consistent with its independently measured 0.036
   waves / Strehl 0.951 (Marechal predicts ~4.4 points).

5. So the roadmap's original conclusion -- *"the loss lies in the chain's
   TILTED-CONGRUENCE TRANSPORT across the coarse legs"* -- was **right in kind
   and wrong in place**: it is ONE leg, the last one, the only one where the
   carrier NA (0.148) and the carried tilt (0.0549) are both large.  The scope
   doc's refutation of it (`n = 0..5` bounded at 0.31 points) is a
   MIS-ATTRIBUTION: at `n`, the chain has done legs 0..n-1 and the ORACLE does
   leg `n`.  Leg 5 is only ever exercised by the chain at `n = 6`, where it is
   inseparable from the element pass unless the intermediate point B is
   inserted.

---

## 1. Method: the arbiter, and why it is not the previous instrument

The arbiter is the exact skew ray trace (`lumenairy.raytrace`, Zemax-validated)
run through the element's OWN launch geometry: the same
`surfaces_from_prescription` on the aperture-stripped prescription, the same
exit-vertex correction, the same H6 carrier entrance eikonal, the same on-axis
OPL reference.  For a set of EXIT-grid nodes,
`wfe_probe_common.exact_phase_on_nodes` Newton-solves the entrance->exit map
with the EXACT trace as the residual (the Jacobian is a central finite
difference of the same trace, so the Jacobian's error cannot move the fixed
point), then applies a first-order exit-direction correction for the leftover
landing residual.  Converged residual **6.8e-13 m**, i.e. 5e-7 waves of phase.
The comparison is therefore

```
    WFE(X, Y) = arg E_out(X, Y)  -  k0 * [ W(xe*, ye*) + OPL_trace(xe*, ye*) ]
```

evaluated POINTWISE on the same nodes.  **No interpolation of the field, no
FFT, no global derivative.**  This is the `LOCAL` idiom of
`decentred_fit_defect.py`, upgraded in one way that matters: that script's
reference OPL is rebuilt from the ELEMENT'S OWN Chebyshev fits at a higher
order, so a defect shared by the fit and its higher-order twin is invisible to
it.  Here the reference is an independent exact trace.

### 1.1 Sampling adequacy -- stated for every wave measurement

The exit NA is 0.363 (the element's own marginal-ray figure; 0.284-0.290 over
the amplitude-weighted mask this study measures on) against a grid Nyquist
direction cosine of `lambda/2dx` = **0.0197**, i.e. **18x short**.  A field with
that content cannot be interpolated, differentiated by FFT, or resampled on
this grid.  It CAN be compared node-by-node against another field carrying the
same fast phase, because the compared quantity is the DIFFERENCE, and the
difference is slow.  The proof is printed with every number: the maximum
WRAPPED nearest-neighbour step of the residual `arg(E_out) - Phi_exact`, which
must sit far below pi.  Measured:

| measurement | nn-step median | nn-step p99 (power-weighted) | max | loop residual |
|---|---|---|---|---|
| ray-map leg, pure congruence | 0.0001 rad | 0.0003 | 0.0285 | 7.6e-19 |
| residual leg | 0.0002 | 0.0059 | -- | -- |
| element vs exact-ray oracle | 0.0022 | 0.072-0.106 | -- | -- |

`pi = 3.1416`.  The loop residual (max |curl| of the wrapped-difference field
over unit cells) is ~1e-18, proving the unwrap is single-valued and
path-independent.

A second, UNWRAP-FREE metric is reported alongside every rms: the best
piston-and-tilt-removed coherent fidelity `F = |sum w e^{i psi}|^2 / (sum
w)^2`, with the tilt found by iteratively aligning the weighted phasor sum (no
unwrap anywhere).  The two agree wherever the nn-step is small, which is what
licenses the unwrap.

---

## 2. The decentre sweep, with the residual held fixed

`wfe_probe_decentre.py` -- input `A(x,y) * exp(i k0 W)` with `A` REAL and
POSITIVE, so the carrier-de-chirped residual is identically 1 and the remap leg
contributes EXACTLY 1 (bilinear interpolation of a constant unit phasor is that
constant).  The arbiter is then complete and any difference is unambiguously
the ray-map / fit / Newton / OPL-upsample construction.  Design 121's real last
group, real carrier, real tilt, real beam radius; ONLY the decentre moves.

```
DEC=0,0.5,1.079,1.5 python wfe_probe_decentre.py
```

| tilt | dec/w | dec (mm) | rms (waves) | piston-only | PTV | Strehl | rel amp err | nn-step |
|---|---|---|---|---|---|---|---|---|
| ON | 0.000 | 0.0000 | **0.00017** | 0.00017 | 0.0008 | 1.0000 | 0.0025 | 0.0016 |
| ON | 0.500 | 1.5628 | **0.00024** | 0.00024 | 0.0021 | 1.0000 | 0.0024 | 0.0019 |
| ON | 1.079 | 3.3724 | **0.00031** | 0.00031 | 0.0455 | 1.0000 | 0.0024 | 0.0227 |
| ON | 1.500 | 4.6883 | **0.00072** | 0.00072 | 0.0908 | 1.0000 | 0.0036 | 0.0345 |

There IS a decentre trend -- 4.2x from 0 to 1.5 w -- but it starts at 1.7e-4
waves and ends at 7.2e-4.  The design needs ~0.089 waves.  **The ray-map leg is
120x too small to matter at 1.5 w and 290x too small at the design point.**

Now the same sweep with the design's REAL residual carried along (the chain's
own group-5 entrance envelope, Fourier-shifted so the residual CONTENT is
identical at every decentre):

```
DEC=0,0.5,1.079 python wfe_probe_residual_leg.py
```

| dec/w | RAY-MAP leg | RESIDUAL leg | residual-leg fidelity | TOTAL vs exact-ray oracle | oracle fidelity |
|---|---|---|---|---|---|
| 0.000 | 0.00017 | 0.00025 | 0.999998 | **0.03766** | 0.945555 |
| 0.500 | 0.00024 | 0.00025 | 0.999998 | **0.03732** | 0.946493 |
| 1.079 | 0.00034 | 0.00028 | 0.999997 | **0.03577** | 0.950746 |

**The total is FLAT in decentre -- it in fact falls slightly.**  Decentre is
not the mechanism.

### 2.1 How the two legs are separated without instrumenting the library

On the shipped chain path the element assembles

```
    E_out = ard_map(X,Y) * exp(i k0 opl_map(X,Y)) * resid_map(X,Y)
```

(`_lens_traced.py` Step 3: the screen field's unit phasor is kept, the
magnitude is swapped for the ray-density one, and the transported residual
phasor is multiplied on).  `ard_map` and `opl_map` depend on the input only
through `|E_in|`.  So running the SAME `|E_in|` twice --

* run A: `E_in = |r| exp(i k0 W)` -> `resid_map == 1` exactly
* run B: `E_in =  r  exp(i k0 W)` -> `resid_map` = the transported residual

-- gives `E_out_B / E_out_A == resid_map` EXACTLY, and `arg E_out_A == k0 *
opl_map`.  No library edit, no inference.

---

## 3. Which construction inside the element?  All of them, measured

`wfe_probe_orders.py` runs design 121's real post-DOE chain per order and
captures the REAL `apply_real_lens_traced` call the chain makes on the last
group (input, carrier and returned field, by a script-side monkeypatch of
`lumenairy.elements.apply_real_lens_traced`).  No synthetic stand-in anywhere.

```
ORDERS='0,0 -2,0 -4,0 -4,-2' HALF=96 python wfe_probe_orders.py
```

| order | dec/w | grad a rms | RAY-MAP | RESIDUAL | TOTAL | Strehl | rel amp err | P in mask |
|---|---|---|---|---|---|---|---|---|
| (0,0)   | 0.000 | 0.000660 | 0.00016 | 0.00011 | **0.01098** | 0.9952 | 0.0148 | 0.9996 |
| (-2,0)  | 0.482 | 0.000819 | 0.00023 | 0.00014 | **0.01539** | 0.9907 | 0.0185 | 0.9995 |
| (-4,0)  | 0.965 | 0.001296 | 0.00026 | 0.00023 | **0.03025** | 0.9645 | 0.0239 | 0.9994 |
| (-4,-2) | 1.079 | 0.001457 | 0.00035 | 0.00027 | **0.03577** | 0.9507 | 0.0257 | 0.9994 |

All in waves.  `grad a rms` is the amplitude-weighted rms of the INPUT
residual's own transverse direction cosines.  `P in mask` is the fraction of
the element's whole-grid exit power inside the measured mask -- the metric sees
essentially all of it, so it is not a core-only reading.

Scope doc section 7 item 1 named three untested surfaces.  Each is now
measured:

| surface | how it was isolated | result |
|---|---|---|
| **(a) Newton initial guess / bracket at large decentre** | included in the RAY-MAP column (pure congruence, decentre swept to 1.5 w); plus the direct convergence map of section 5 | 0.00017 -> 0.00072 waves; every unconverged pixel is outside the ray domain and carries ZERO returned field |
| **(b) entrance-pullback domain** | included in the RAY-MAP column: the pullback `(xe, ye)` is what `So` is evaluated at, and the phase is stationary in it (Fermat), so a pullback error enters only at second order | 0.0002-0.0004 waves |
| **(c) `preserve_input_phase='remap'` resampling of the input phasor** | the RESIDUAL column: the element's own `resid_map` (extracted exactly by the two-run ratio of section 2.1) against a band-limited evaluation of the SAME residual at the exact pullback point | **0.00011 -> 0.00027 waves**, flat in decentre |

**None of the three is the mechanism.**  The element's residual 0.011-0.036
waves is the MODEL, not the implementation: `remap` transports the input
residual along the CARRIER rays `grad(W)` and evaluates it at that ray's foot,
whereas the exact congruence follows `grad(W + a)`.  The error is the
second-order stationary-phase term that construction drops, and it tracks
`grad a` (which grows 2.2x across the fan) rather than the decentre.  It costs
1-4 EE3 points, not 24.

---

## 4. Closing the scope doc's 5x discrepancy (section 7 item 2)

The scope doc flagged that its last-group probe read 0.010-0.012 waves where
the loss implied ~0.067.  My instrument reads 0.036 waves at the design point,
which is still 2.4x low in sigma against ~0.089 waves -- **at BOTH ends of the
fan**, i.e. a systematic, not scatter.  A consistent systematic is the
signature of a wrong ATTRIBUTION, not a wrong instrument, so the instrument was
attacked directly.

`wfe_probe_readout.py` builds the ORACLE exit field on the SAME grid the
element writes to (amplitude = the exact ray-tube amplitude of the TOTAL
congruence, phase = the exact traced phase) and pushes both fields through the
IDENTICAL readout -- same re-envelope, same aliasing-free phase gradient, same
ray launch, same Rayleigh-Sommerfeld integral.

```
ORD=-4,-2 HALF=110 DXFAC=0 NL=9999 NOUT=61 DXO=0.4 python wfe_probe_readout.py
```

| field | FWHM (um) | EE3 % | EE6 % | EE12 % |
|---|---|---|---|---|
| ELEMENT (shipped) | 4.021 | **66.35** | 91.96 | 99.73 |
| ORACLE (exact ray) | 3.984 | **69.23** | 93.25 | 99.83 |
| element amplitude + oracle phase | 3.951 | 68.83 | 93.11 | 99.82 |
| oracle amplitude + element phase | 4.067 | 66.38 | 92.09 | 99.74 |
| ORACLE, EXACT exit direction cosines | 3.985 | 69.23 | 93.23 | 99.83 |
| ORACLE, launched at the group ENTRANCE | 3.996 | 69.28 | 93.27 | 99.83 |

* The ELEMENT row reproduces `hybrid_localize_121.py`'s `n = 6` (66.24) to 0.11
  points, so this readout is a faithful stand-in for the chain's.
* An **exact-ray field reads 69.2, not 89.8**.  The element costs 2.9 points
  against it; the amplitude/phase swap says essentially all of that is phase.
* It is not the exit grid: refining the exit lattice 2x and 4x (with the launch
  density refined WITH it -- the first attempt was invalidated by `NL` capping
  the ray count) leaves it at 69.22 / 69.22, and the exit envelope's
  power-weighted per-pixel step falls exactly as 1/dx (0.0155 -> 0.0077 ->
  0.0039 median), proving the exit envelope IS band-limited and resolved.
* It is not the exit-direction estimate: replacing the nearest-neighbour phase
  gradient with the EXACT traced exit cosines changes 69.23 to 69.23.
* It is not the relaunch: the same congruence launched at the group ENTRANCE
  instead reads 69.28.

So the exact-ray congruence THAT ARRIVES AT GROUP 5 gives 69.3, while the
hybrid's `n = 5` gives 90.00.  The two differ only in WHERE the hand-off is
taken -- and `n = 5`'s is at group 4's EXIT, one 3.3233 mm gap earlier.

**My instrument was not under-reading.  The 24-point step was mis-attributed.**

---

## 5. The 81.2 % Newton non-convergence: verified, and it is benign

Scope doc section 5 item 2: *"81 % of the grid is a lot of pixels to be out of
domain and the claim has not been verified here."*  Verified now, with the
temporary `_DIAG_NEWTON` sink (section 8) capturing the Newton `active` mask
and converged `(xe, ye)` directly.

```
ORD=-4,-2 ITERS=12,60 python wfe_probe_newton.py
```

Coarse Newton grid 256x256 = 65536 px (`sub` = 4), `n_launch` = 231,
`launch_radius` = 15.2974 mm, clip `bound` = 15.2821 mm, tol = 0.3321 um.

| quantity | 12 iterations | 60 iterations |
|---|---|---|
| unconverged | 53228 / 65536 = **81.2 %** | 53228 / 65536 = **81.2 %** |
| of those, sitting exactly ON the Newton clip bound | **100.00 %** | 100.00 % |
| of those, ending outside `0.99 * launch_radius` (so `opl -> NaN`, field zeroed) | **100.00 %** | 100.00 % |
| exit radius of the nearest unconverged pixel | **6.6422 mm** | 6.6422 mm |
| fraction of the returned `|E_out|^2` on unconverged pixels | **0.000000 %** | 0.000000 % |
| unconverged among pixels with a NON-ZERO returned field | **0 of 3118** | 0 of 3118 |
| unconverged in the +-1.594 mm block on the exit chief ray | **0 of 625** | 0 of 625 |

**Findings.**

1. The roadmap's "out-of-domain edge pixels" reading is CORRECT, and now
   precise: these are exit pixels with no pre-image in the launch disc.  Newton
   drives them outward, `np.clip(..., -bound, bound)` pins them to the boundary,
   and the pinned point is a FIXED POINT -- which is exactly why 60 iterations
   change nothing.  It is not slow convergence; it is a saturated bracket.
2. They are harmless by construction: `out_of_domain = xe^2 + ye^2 > (0.99 *
   launch_radius)^2` NaNs their OPL and the assembly zeros the field there.
   95.24 % of the coarse lattice returns exactly zero field, and 85.28 % of
   THOSE zero pixels are the unconverged ones.
3. **Their residual cannot correlate with the wavefront error**, because there
   is no field on them to carry an error.  Zero of the 3118 coarse pixels with a
   non-zero returned field are unconverged.
4. The warning is a false alarm whenever the grid is much larger than the beam,
   which is the normal chain condition (here the beam occupies ~5 % of the
   grid).  Its `>1 %` threshold is a fraction of the WHOLE grid rather than of
   the illuminated support, so it fires on geometry, not on error.  That is a
   guard-quality item, not a correctness one.

The other unexplained warning (`NA_exit=0.3633` needs `dx <= 1.80 um`, grid has
33.211 um) is also now accounted for: the exit wavefront is indeed 18x
beyond Nyquist, but the element writes each node independently, so the node
VALUES are right (section 3), and the exit ENVELOPE -- the quantity any
downstream consumer differentiates -- is band-limited and resolved (section 4).

---

## 6. Where the loss actually is: the last coarse leg

`wfe_probe_gap.py` inserts the missing intermediate point.  The chain's
group-5 ENTRANCE field is the exact array `apply_real_lens_traced` is handed,
captured by monkeypatch; point A launches the SAME residual machinery from
group 4's EXIT and lets the exact rays cross the gap.  Same residual model,
same surfaces, same Rayleigh-Sommerfeld integral on both sides -- only the
launch plane differs.

```
ORD=-4,-2 NOUT=61 DXO=0.4 python wfe_probe_gap.py
```

Post-DOE gaps (mm): `0:7.0000  1:5.0000  2:5.0000  3:32.4787  4:8.6779
5:3.3233`.

|  | what the chain does | what the oracle does | EE3 % |
|---|---|---|---|
| **A** | groups 0..4 | the 3.3233 mm gap + group 5 + trailing | **89.98** |
| **B** | groups 0..4 AND the gap | group 5 + trailing | **69.28** |
| **C** | groups 0..4, the gap AND group 5 | trailing only | **66.24** |

A reproduces `hybrid_localize_121.py`'s `n = 5` (90.00) to 0.02 points; C is
its `n = 6` verbatim.  **A -> B = -20.70 (the leg).  B -> C = -3.04 (the
element).**

B is converged: launch density 161 / 221 / 301 across the beam reads 69.30 /
69.28 / 69.28; clip 2.5 / 3.0 / 3.5 w reads 69.28 / 69.28 / 69.29; live power
100.0000 %; RS integrand step p99.9 = 0.631 cycles.  It also agrees with the
completely independent exit-plane route of section 4 (69.23).

### 6.1 Why leg 5 and not legs 0-4

| leg | gap (mm) | carrier R entering (mm) | w entering (mm) | carrier NA | carried tilt \|(L,M)\| |
|---|---|---|---|---|---|
| 0-2 | 7.0 / 5.0 / 5.0 | +703650 | 6.32 | ~1e-5 | 0.0515 |
| 3 | 32.4787 | -263.19 | 5.99 | 0.023 | 0.0515 |
| 4 | 8.6779 | -51.470 | 4.85 | 0.094 | 0.0467 |
| **5** | **3.3233** | **-24.462** | **3.62** | **0.148** | **0.0549** |

Leg 5 is the SHORTEST gap but the only one where the carrier NA and the carried
tilt are both at their maximum; the product `NA * tilt` is 15x leg 4's.  This
puts the scope doc's own candidate #4 (`_tilt_obliquity`'s anisotropic
effective distance `z/(1-L^2)^{3/2}`) back in play: both it and candidate #6
were "refuted" by a bound computed from the mis-attributed `n = 0..5` sweep,
which never covered leg 5.

### 6.2 One knockout run on the leg: grid resolution is NOT the cause

`|k L dx|/pi` is 3.22 entering leg 5 and 2.78 leaving it, i.e. the carried tilt
ramp is 3x beyond the co-moving grid's Nyquist.  Doubling the chain grid halves
both:

```
ORD=-4,-2 RN=2048 NOUT=61 DXO=0.4 python wfe_probe_gap.py
```

| `RN` | `dx` entering leg 5 (um) | `\|k L dx\|/pi` | A | B | LEG |
|---|---|---|---|---|---|
| 1024 | 38.4324 | 3.22 | 89.98 | 69.28 | **-20.70** |
| 2048 | 19.2162 | 1.61 | 89.98 | 68.66 | **-21.32** |

**Halving `dx` does not recover any of it -- it is 0.6 points WORSE.**  So
candidate #6 (tilt-ramp aliasing on the co-moving grid) is refuted a second
time, now on the correctly-attributed quantity, and the leg's defect is
resolution-independent: it is a MODEL error in the transport, not a sampling
error.  Point A is unchanged to 0.00 points at both `RN`, which also confirms
that everything upstream of leg 5 is `RN`-converged.

The remaining candidates are all inside
`propagate_traced_carrier_chain`'s per-group leg (`carrier.py:5422-5453`):
`propagate_carrier_referenced`'s paraxial envelope step at carrier NA 0.148,
and the `_tilt_obliquity` treatment of the tilt (a chief-ray advance plus a
CONSTANT piston `exp(i k0 z (1/cos theta - 1))`, which is exact for the chief
ray but does not carry the beam's own angular spread about it).  I did not
separate them.  It is squarely in the chain's leg, not in the element.

---

## 7. What I could NOT determine

1. **Which construction inside the LEG loses the 20.7 points.**  I localised it
   to `propagate_traced_carrier_chain`'s gap transport into group 5, established
   the NA x tilt scaling across the fan, and eliminated grid resolution
   (section 6.2: `RN` 1024 -> 2048 gives -20.70 -> -21.32, i.e. no recovery).
   I did NOT separate `propagate_carrier_referenced`'s paraxial envelope step
   from the `_tilt_obliquity` piston / chief-ray advance.  The obvious next
   knockouts, none of which I ran: zero the tilt on leg 5 alone (a
   `groups[5]['r_in']` override with `L = M = 0`) and see whether the leg
   recovers; shorten the gap by splitting it into two half-legs (a paraxial
   envelope step that is wrong should scale with the step count, an obliquity
   model error should not); and compare `carrier_reference='parabola'` against
   the shipped `'sphere'` on that leg alone.
2. **The exact Marechal-to-EE3 mapping.**  The element's Strehl 0.9952 /
   0.9507 predicts 0.43 / 4.4 EE3 points against the measured 1.17 / 3.04.
   Same order and same trend, but the mapping is aberration-shape dependent and
   I did not decompose the element's residual into Zernike orders.  Nothing in
   section 0 turns on it -- the element's cost is measured directly (B -> C),
   not inferred from the Strehl.
3. **Whether the shipped `final_leg='exact'` path shares the leg defect.**  The
   fine retrace re-runs the element from the group's ENTRANCE on a 1.508 um
   grid, so it inherits whatever the leg handed it.  Since the leg runs BEFORE
   the retrace, I expect it to, and the roadmap's recorded 65.26 for that path
   is consistent -- but I did not run `fan_multi_121.py` end to end.
4. **The on-axis 1.01-point leg cost and 1.17-point element cost.**  Both are
   real and both are small; I did not chase either.  Together they are the
   ~2-point gap between the shipped single-beam acceptance (88.8) and the ideal
   ceiling (90.2), consistent with the scope doc's item 3.
5. **The element's 0.036-wave model error as a function of anything but
   `grad a`.**  I established it is not decentre and that it tracks the input
   residual's own ray slope, but I did not verify the predicted quadratic
   scaling by sweeping the residual amplitude (an `ALPHA` sweep is wired into
   `wfe_probe_remap.py` / `wfe_probe_residual_leg.py` but was not run).

---

## 8. Measurement artefacts found and killed in MY OWN instruments

1. **The first arbiter omitted the transported input residual** and read 1.43
   waves on a field correct to 0.036.  Caught by its own nn-step saturating at
   pi.  Fixed by the two-run algebraic split of section 2.1, which makes the
   arbiter complete by construction for run A.
2. **The residual model's global 2-D unwrap** (`unwrap(unwrap(.,axis=1),
   axis=0)`, the house idiom) FAILED its own self-test by **1.96 rad**: the
   beam skirt, where the residual amplitude is ~0 and its phase is numerical
   noise, makes the row-wise unwrap arbitrary and the cubic spline's GLOBAL
   prefilter carries that into the core.  Replaced by direct interpolation of
   the unit PHASOR plus wrapped central differences for the gradient -- no
   unwrap anywhere, and only `exp(i k0 a)` is consumed downstream, so a 2*pi
   ambiguity is invisible by construction.
3. **The Fourier upsample's zero-padding was off by one fine sample** whenever
   the crop size was ODD (the `fftshift`-then-centre-paste idiom that
   `hybrid_localize_121._fourier_up` also uses).  Self-test 2.62 rad.  Replaced
   by an unshifted quadrant copy, which is exact at the source nodes for any
   crop size.  **Self-test now 7e-14 rad.**  Every number in this document is
   from after that fix.
4. **The `nn-step` sampling diagnostic was reported as a MAX** and read pi on
   configurations whose core was clean to 0.02 rad -- one skirt pixel sets it.
   Replaced by the power-weighted median and 99th percentile (both reported).
   The same trap is in `exact_ray_oracle_121._phase_gradient`, whose "envelope
   per-pixel phase step 3.14 rad -- ALIASED" banner on the `n = 6` hand-off is
   this artefact: the power-weighted median is 0.0155 rad and it falls exactly
   as 1/dx under refinement.
5. **The first readout referenced the exit sphere to the GRID CENTRE** on a
   beam decentred by 1.9 mm, leaving a huge quadratic residual; every field
   read EE3 ~4 %.  Fixed by referencing `TiltedCarrier(R_out, L_out, M_out,
   x_c_out, y_c_out)`, i.e. the chain's own exit congruence about the chief ray.
6. **The first exit-grid refinement test was inert** because `rs_spot`'s `nl`
   argument decimated the launch lattice back to the coarse ray count, so
   "refining the grid" changed nothing by construction and looked like a
   converged result.  Caught by the RS integrand step not falling.  Re-run with
   the launch density refined with the grid.
7. **The `RSstep` sampling diagnostic itself is unreliable** on any field whose
   launch phase comes from a 2-D unwrap (it differences a quantity carrying
   spurious 2*pi jumps): it reads 4.16 cycles on a configuration whose
   properly-computed value is 0.575.  Both are reported; only the unwrap-free
   one is used.

Not eliminated: the element's 0.036-wave figure is an amplitude-weighted rms
over a mask holding 99.94 % of the exit power, and its conversion to EE3 points
via Marechal is approximate (item 7.2).  The element's cost is therefore quoted
from the direct B -> C measurement (3.04 points), not from the Strehl.

---

## 9. Reproduction

All commands from `validation/repro_traced_carrier_121/`.  Chain A
(source -> DOE) and the per-order group-5 element calls are cached to
`_chainA_*.npz` / `_wfe_probe_orders_*.npz` on first use; delete to force a
re-run.

```bash
# 0. Recon: capture the chain's six real element calls and describe them.
ORD=-4,-2 HALF=72 python wfe_probe_recon.py

# 1. Decentre sweep with the residual set to unity (isolates the ray map,
#    the fits, the Newton inversion and the OPL upsample).           ~10 s
DEC=0,0.25,0.5,0.75,1.0,1.079,1.25,1.5 python wfe_probe_decentre.py
#    a completely different fit basis, as a control:
KW='newton_fit=spline' DEC=0,1.079 python wfe_probe_decentre.py

# 2. Decentre sweep with the design's REAL residual, split into the ray-map
#    leg, the residual leg and the total vs the exact-ray oracle.    ~2 min
DEC=0,0.5,1.079,1.5 python wfe_probe_residual_leg.py

# 3. The same three legs across the DOE fan (the field-angle driver). ~8 min
ORDERS='0,0 -2,0 -4,0 -4,-2' HALF=96 python wfe_probe_orders.py
#    patch / threshold convergence:
ORDERS='-4,-2' HALF=72,96,120 THRESH=0.05 python wfe_probe_orders.py

# 4. Closure test on this study's own instrument: element field vs an exact
#    field through an IDENTICAL readout, + exit-grid refinement.     ~2 min
ORD=-4,-2 HALF=110 HALFF=60 DXFAC=1,2 NL=9999 NOUT=61 DXO=0.4 \
    python wfe_probe_readout.py
ORD=0,0   HALF=110 DXFAC=0 NL=9999 NOUT=61 DXO=0.4 python wfe_probe_readout.py

# 5. THE SPLIT: coarse leg vs element pass.                          ~2 min/order
for o in 0,0 -2,0 -4,0 -4,-2; do \
  ORD="$o" NOUT=61 DXO=0.4 NLSWEEP=301 CLIPS=3.5 python wfe_probe_gap.py; done
#    the leg knockout that was still running when this was written:
ORD=-4,-2 RN=2048 NOUT=61 DXO=0.4 python wfe_probe_gap.py

# 6. Newton non-convergence map.  NEEDS the temporary _DIAG_NEWTON sink
#    (section 8 below) -- prints a notice and exits 1 without it.
ORD=-4,-2 ITERS=12,60 python wfe_probe_newton.py

# reference points from the existing scripts:
ORD="-4,-2" NMIN=5 NMAX=6 NOUT=61 DXO=0.4 NL=181 python hybrid_localize_121.py
```

### Files added by this study (none are library code)

```
validation/repro_traced_carrier_121/wfe_probe_common.py        exact-trace arbiter, estimators, chain capture
validation/repro_traced_carrier_121/wfe_probe_recon.py         capture + describe the six real element calls
validation/repro_traced_carrier_121/wfe_probe_decentre.py      decentre sweep, residual == 1
validation/repro_traced_carrier_121/wfe_probe_remap.py         band-limited residual model + total-eikonal inverse
validation/repro_traced_carrier_121/wfe_probe_residual_leg.py  ray-map leg vs residual leg vs oracle
validation/repro_traced_carrier_121/wfe_probe_orders.py        the same three legs across the DOE fan
validation/repro_traced_carrier_121/wfe_probe_readout.py       closure test on this study's own instrument
validation/repro_traced_carrier_121/wfe_probe_gap.py           the coarse-leg / element-pass split
validation/repro_traced_carrier_121/wfe_probe_newton.py        Newton non-convergence map (needs the temp sink)
```

### Temporary instrumentation -- REVERTED

Section 5 needed the Newton `active` mask and converged `(xe, ye)`, which are
locals of the `_invert_newton` closure and cannot be reached from outside.  A
module-level `_DIAG_NEWTON = {}` sink plus a 10-line stash immediately after
the iteration loop was added to `lumenairy/elements/_lens_traced.py`, used, and
then removed.  See section 10 for the verification.

---

## 10. Instrumentation revert -- VERIFIED

The `_DIAG_NEWTON` sink and its 10-line stash in
`lumenairy/elements/_lens_traced.py` are removed.  Verified:

```
$ git diff --stat lumenairy/
                                     <-- no output: no library file is modified
$ git status --short lumenairy/
                                     <-- no output
$ python -c "from lumenairy.elements import _lens_traced as L;              print(hasattr(L, '_DIAG_NEWTON'))"
False
```

`wfe_probe_newton.py` detects the absent sink and exits 1 with a notice rather
than failing obscurely, so section 5 is reproducible only by re-adding the
instrumentation (the exact patch is described in section 9).
