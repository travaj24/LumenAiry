# Zemax POP cross-check of design 121, per DOE order (2026-07-31)

**Scope.** Design 121's per-order image quality had been validated entirely
against in-house oracles (lumenairy's own skew ray trace, a Rayleigh-Sommerfeld
integral written during the campaign, and a bicubic spline map). Niches C5 and
C6 were both accepted on agreement with those oracles. Nothing had been checked
against a commercial reference. This audit runs **Zemax OpticStudio 2023 R1
Physical Optics Propagation** on the same `.zmx` the chain loads, for the
extreme order **(-4,-2)** and five others, and compares against our exact-ray +
RS oracle and against the chain.

**Verdict up front.**

1. The POP harness is validated: it reproduces the one recorded Zemax number in
   the tree (2.7378 um waist radius / 3.223 um FWHM) to 4 digits.
2. **Our oracle is independently corroborated at the extreme order — but by
   Zemax's ray-based wavefront error, not by POP.** Zemax's own RMS OPD at
   (-4,-2) is **0.030 waves tilt-free** (Marechal limit 0.071), i.e. the design
   really is diffraction-limited at the extreme order, exactly as the oracle
   says. POP disagrees, and POP is the one that cannot be trusted here.
3. POP claims the extreme order loses **8.8 EE3 points** vs on-axis. That loss
   is a broad 1e-4 pedestal in POP's own log panel, is not consistent with a
   0.030-wave wavefront, survives every sampling change POP allows, and cannot
   be driven to convergence because the problem's space-bandwidth requirement
   exceeds POP's 8192x8192 cap by roughly 5x. It is a POP artefact.
4. Energy: POP puts **1.00000000** of the launched power on the image plane for
   every order (zero vignetting, confirmed by an independent 70681-ray pupil
   trace). Both our chain and our oracle agree.

---

## 0. The EE convention (checked, not assumed)

The brief said "EE3/EE6/EE12 = 3/6/12 um **diameter**". **That is wrong.** Both
in-house scorers use r as a **RADIUS in microns**:

`validation/repro_traced_carrier_121/focus_scan_121.py`, `metrics()`:

```python
ee = {r: float(I[rr <= r * 1e-6].sum()) * dxo * dxo / P_in for r in (3, 6, 12)}
```

`rr` is measured from the **peak pixel**; the denominator is `P_in`, the **chain
input power**.

`validation/repro_traced_carrier_121/hybrid_localize_121.py`, `rs_spot()` — the
source of the per-order reference numbers:

```python
for rad in (3e-6, 6e-6, 12e-6):
    out[f'ee{int(rad*1e6)}'] = float(I[r <= rad].sum()) / tot
```

`r` is measured from the **centroid**; the denominator `tot` is the sum over the
**whole output window**, which at the campaign's settings (`NOUT=61`,
`DXO=0.4`) is a **24.0 x 24.0 um square**, *not* the launched power.

So **EE3 = energy inside a 3 um radius (6 um diameter)**, and the brief's
reference numbers ((0,0) 89.21 / (-4,0) 88.94 / (-4,-2) 88.49) are normalised to
a 24 um square window. Every number below states which denominator it uses.

---

## 1. Setup, verbatim and reproducible

**Connection.** `zospy` 2.1.5 in
`D:\...\OPDPy_Lumenairy_Crosscheck\.venv-zemax` (Python 3.13.13), standalone
mode. `zos.version` = **23.1.0**, `LicenseStatus` = **PremiumEdition**.
Connected on first attempt; no licence or headless obstruction.

**Design file** (unmodified on disk; all edits are in-memory only):
`D:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics\Reverse_Symmetric_ASM\tx4designstudy121\20260707 dll Tx02-MSOP16.zmx`

**Isolating a diffraction order.** The file carries the DOE order in
**parameter 2 of the two DGRATING surfaces**, driven by the multi-configuration
editor:

| surface | type | par1 (lines/um) | deflects in | cfg 1 | cfg 2 | cfg 3 |
|---|---|---|---|---|---|---|
| 9  | DGRATING | +0.00879 | **global Y** | 0 | -2 | -4 |
| 11 | DGRATING | -0.00879 | **global X** | 0 | 0 | -2 |

(Surfaces 10 and 12 are +90/-90 deg coordinate breaks about z, which is why
surface 9 deflects in y.) So the order is selected exactly and without
ambiguity — POP then propagates that order and only that order. `pop_run_121.py`
writes par2 directly (`--m` = global X = surface 11, `--n` = global Y = surface
9) so the orders sit on **our** (m,n) axes rather than the file's transpose.
The post-DOE relay (surfaces 13-29) is rotationally symmetric about z — every
surface a centred standard sphere — so the transpose is a pure 90 deg rotation.

**Chief-ray cross-check (Zemax vs lumenairy `raytrace`):**

| order | Zemax image (x,y) [um] | lumenairy oracle (x,y) [um] |
|---|---|---|
| (-1,0) | (-479.928, 0.000) | (-479.97, 0.00) |
| (-2,0) | (-959.883, 0.000) | (-959.96, 0.00) |
| (-3,0) | (-1439.912, 0.000) | (-1440.03, 0.00) |
| (-4,0) | (-1920.094, 0.000) | (-1920.25, 0.00) |
| (-4,-2) | (-1920.241, -960.120) | (-1920.39, -960.20) |

Agreement to <= 0.16 um out of 1920 um (**8e-5 relative**). Our skew ray trace
and Zemax's agree; nothing downstream is a geometry disagreement.

**POP settings** (`pop_run_121.py`, via `zospy.analyses.physicaloptics`):

```
wavelength      = 1        (1.31 um, the only wavelength)
field           = 1        (0,0 object height; POP centres its array on this
                            field's chief ray, which for a diffracted config IS
                            the order's chief ray -- this is what makes an
                            off-axis order measurable at all)
start_surface   = 1        surface_to_beam = 0
end_surface     = 29       ("RX Plane")
beam_type       = GaussianWaist, Waist X = Waist Y = 0.004 mm
use_polarization= False    separate_xy = False
use_total_power = True, total_power = 1.0   (use_peak_irradiance = False)
data_type       = Irradiance   project = AlongBeam
x_sampling = y_sampling = N x N,  x_width = y_width = W mm
production point: N = 4096, W = 0.2 mm
```

No per-surface POP overrides were used (`AutoResample` etc. left at file
defaults). The image surface was moved only for the through-focus scan
(surface 28 thickness +- dz), and restored afterwards.

Raw grids: `pop_run_121.py` saves I(x,y) to `.npz`; **all metrics are computed
offline** by `pop_metrics_121.py` / `pop_plot_threeway.py` from code lifted from
our own scorers, so the comparison can never drift by "the harness scored it
differently".

---

## 2. The (0,0) control — and a correction to the gate

**The gate as written cannot be met, because the number it names was never a POP
of this design.** `focus_scan_121.py` carries `POP waist 2.737um radius` /
`FWHM 3.223um`. Tracing it back:
`docs/audit_asm_thinlens_focus_2026_07_18.md` records it as

> | colleague's GUI POP | paraxial x pilot-beam POP | 2.7378 | (clean to 1e-4) |

i.e. a POP of the **paraxially equivalent 4f system** (f1 = 60.916 mm,
f2 = 41.666 mm), two ideal paraxial lenses with no aberration and no glass,
whose answer is just the magnification 4 um x f2/f1 = 2.736 um. The same audit
already says so: *"the 3.223 um target is the paraxial-Gaussian estimate"*, and
*"the ideal-field ceiling through this readout is 3.45-3.55 um"*.

So the control was run **on that system**, with this harness
(`pop_control_4f.py`):

| N | W [mm] | dx_img [um] | POP "Beam Width" [um] | measured 1/e^2 w [um] | FWHM [um] |
|---|---|---|---|---|---|
| 2048 | 0.1 | 0.0334 | 2.7398 | **2.7378** | **3.2269** |
| 2048 | 0.2 | 0.0668 | 2.7360 | 2.7417 | 3.2374 |
| 2048 | 0.4 | 0.1336 | 2.7360 | 2.7505 | 3.2677 |

Analytic: 2.7360 um / 3.2214 um. Recorded colleague POP: 2.7378 um / 3.223 um.
**Reproduced to 4 significant figures.** The POP driving in this audit is
correct.

**The real design at the same source.** POP on the actual 121 prescription gives
FWHM **3.52-3.65 um** on axis (sampling spread, section 3), against this
campaign's own ideal-field ceiling of **3.45-3.55 um** and the chain's shipped
acceptance of **3.450 um**. POP, the chain and the oracle all land in a 0.2 um
band on axis. The 3.223 um figure is a paraxial estimate the real design's NA
and aberration do not permit, and it should stop being quoted as a POP target
for this prescription.

**Through focus** (N=4096, W=0.2, surface 28 thickness stepped), EE on the 24 um
square window:

| dz [um] | -30 | -20 | -10 | 0 | +10 | +20 |
|---|---|---|---|---|---|---|
| FWHM [um] | 5.747 | 4.430 | 3.856 | **3.632** | 3.585 | 3.831 |
| EE3 [%] | 46.90 | 65.04 | 80.96 | **89.14** | 84.83 | 69.29 |

Best focus by encircled energy is at the **nominal plane**, which is what
`focus_scan_121.py` finds for the chain (best focus dz = 0, FWHM 3.450, EE3
90.2). POP and the chain agree on where focus is.

---

## 3. Convergence evidence (and its limit)

POP's grid here is **not** free. Zemax POP Fresnel-transforms between surfaces,
so the point spacing at surface k is set by the array WIDTH at surface k-1
(`dx_out = lambda*z/W_in`), not by the beam. Measured with `pop_grid_diag.py
--mode grid`, the two spacings that matter are locked together:

```
dx_mid (at the DOE / relay)  x  dx_img (at the image)  =  53500 um^2 / N
```

(measured: N=1024, W=0.2 -> dx_mid = 397.7 um, dx_img = 0.1336 um). This is
just the array's space-bandwidth product, and it is the whole story:

* the image window width is `0.683 * W_start` and is independent of N;
* `dx_img = 0.683 * W_start / N` (0.683 is the system's own object->image
  magnification, 2.737/4.0);
* `dx_mid` depends only on `W_start`, **not on N** — raising N alone never
  improves the relay sampling.

**The tilt-ramp Nyquist limit.** After the DOE the beam carries a 51.50 mrad
phase ramp for the (-4,-2) order, so the array must satisfy
`dx_mid < lambda/(2 sin theta) = 12.72 um`. Combined with the product above,
at POP's maximum N = 8192 the best simultaneous pair is

```
dx_mid = 12.7 um  (just at Nyquist)  <->  dx_img = 0.51 um
```

To have BOTH a properly sampled tilt (say 3x Nyquist, dx_mid ~ 4 um) and a
0.1 um image pixel would need N ~ 42000. **POP cannot converge this problem; it
is short by about 5x in N.** That is a property of the tool and the design, not
of this harness.

### 3.1 Sweeps actually run (on-axis, 24 um square window)

Fixed W = 0.2 mm (relay grid frozen at dx_mid = 397 um), N swept:

| N | dx_img [um] | FWHM [um] | EE3 [%] | EE6 [%] | EE12 [%] |
|---|---|---|---|---|---|
| 512 | 0.2671 | 3.930 | 85.21 | 99.29 | 99.92 |
| 1024 | 0.1335 | 3.830 | 85.69 | 99.35 | 99.93 |
| 2048 | 0.0666 | 3.747 | 87.16 | 99.46 | 99.94 |
| 4096 | 0.0332 | 3.632 | 89.14 | 99.50 | 99.95 |
| 8192 | 0.0165 | 3.517 | 87.30 | 99.09 | 99.97 |

Fixed dx_img = 0.134 um, relay grid swept 8x (W and N raised together):

| W [mm] | N | dx_mid [um] | pupil pts across beam | FWHM | EE3 (0,0) | EE3 (-4,-2) |
|---|---|---|---|---|---|---|
| 0.1 | 512 | 795 | 16 | 3.921 | 84.46 | 76.53 |
| 0.2 | 1024 | 398 | 31 | 3.830 | 85.69 | 78.83 |
| 0.4 | 2048 | 199 | 63 | 3.806 | 85.61 | 80.42 |
| 0.8 | 4096 | 99 | 126 | 3.850 | 84.94 | 79.50 |

Cross-checks at N = 8192: (W=0.4, dx_img 0.0333) FWHM 3.627 / EE3 88.45 against
(N=4096, W=0.2, dx_img 0.0332) FWHM 3.632 / EE3 89.14 — the same dx_img gives
the same answer to 0.7 EE3 points with the relay grid twice as fine, so the
relay sampling is **not** the limiter; dx_img is.

**Is the movement the field or the estimator?** Rebinning the N=8192 grid down
to the N=512 pixel gives FWHM 3.534 / EE3 86.90, against 3.930 / 85.21 for the
native N=512 run. Rebinning by 4 instead of 16 changes it by 0.02 um / 0.4
points. So **the estimator is fine and the FIELD is genuinely changing with
sampling** — POP is not converged on axis, with a residual spread of about
**FWHM 3.52-3.65 um, EE3 87.3-89.1** over the defensible settings. Everything
POP says is quoted with that band.

### 3.2 The one configuration that satisfies the tilt Nyquist

N = 8192, W = 6.6 mm -> dx_mid = 12.0 um (just inside the 12.72 um limit),
dx_img = 0.551 um, image array 4517 um wide:

| order | FWHM [um] | EE3 [%] | EE6 [%] | Zemax "Beam Width" X, Y [um] |
|---|---|---|---|---|
| (0,0) | 4.074 | 85.86 | 99.37 | 3.77, 3.77 |
| (-4,-2) | 4.079 | 79.16 | 97.45 | **6.88, 4.85** |

(Both FWHM are inflated by the 0.55 um pixel; the *difference* is what matters.)
**The 6.7-point deficit survives when the tilt ramp is sampled at Nyquist.** So
the deficit is not simple ramp aliasing — but note the FWHM of the two is
identical to 0.005 um while the second-moment width nearly doubles: whatever POP
is adding is entirely in the **wings**, not the core.

---

## 4. The tie-breaker: Zemax's own ray-based wavefront error

POP and the oracle disagree, and POP is exactly the tool one distrusts off axis
at high NA. So the question was put to something that is neither: **Zemax's RMS
wavefront error**, a pure real-ray OPD computation with no grid, no propagator
and no sampling knob (`pop_wfe_121.py`, 128x128 pupil, 12453 pupil points,
field 1, wave 1, referenced to the image surface):

| order | field angle [mrad] | RMS OPD, piston-free [waves] | RMS OPD, **tilt-free** [waves] | PV [waves] |
|---|---|---|---|---|
| (0,0) | (0, 0) | 0.00806 | **0.00806** | 0.0507 |
| (-1,0) | (-11.52, 0) | 0.01716 | 0.01039 | 0.1096 |
| (-2,0) | (-23.03, 0) | 0.02407 | 0.01519 | 0.1557 |
| (-3,0) | (-34.55, 0) | 0.01889 | 0.01792 | 0.1020 |
| (-4,0) | (-46.06, 0) | 0.03994 | 0.02019 | 0.1951 |
| **(-4,-2)** | (-46.06, -23.03) | 0.07294 | **0.03015** | 0.3864 |

Tilt is a spot *shift*, not a blur, so the tilt-free column is the diffraction
measure. At the extreme order it is **0.030 waves rms**, well inside the
Marechal diffraction limit of 0.071 waves; the implied Strehl is
`exp(-(2*pi*0.030)^2) = 0.965`. **The design is diffraction-limited at every
DOE order in the fan**, which is exactly what
`exact_ray_oracle_121.py` concluded (it reported <= 0.017 waves; Zemax says
0.030 — same verdict, our number modestly optimistic).

A 0.030-wave wavefront cannot cost 8.8 points of EE3. It removes ~3.5% from the
peak and puts it in a halo *close to the core*. POP's deficit is 2.5x larger and
lives in a flat pedestal 10-20 um out. **The oracle is corroborated; POP is
not.**

---

## 5. Comparison table

All three methods on **one lattice**: dx = 0.1 um, 401 x 401, +-20.0 um about
**each order's own chief ray** (POP centres its array there; the oracle and
chain are centred on the traced chief ray). POP's 0.0332 um grid is
block-averaged then resampled down to 0.1 um — never up-sampled.

Convention: **`focus_scan_121.metrics()`** — r is a RADIUS in um, measured from
the **peak pixel**, EE divided by the input power. Denominators: POP = the
whole POP array (a true transmission, section 6); oracle/chain = the oracle's
(0,0) window power (`oracle_spot` omits the RS `1/(i*lambda)` prefactor so its
absolute scale is uncalibrated, but ratios between its own runs are exact).

| order | angle [mrad] | method | FWHM [um] | EE3 [%] | EE6 [%] | EE12 [%] | P(+-20um)/Pin [%] |
|---|---|---|---|---|---|---|---|
| (0,0) | (0.00, 0.00) | **POP** | 3.649 | 88.65 | 99.03 | 99.49 | 99.844 |
| | | chain | 3.384 | 89.72 | 99.09 | 99.21 | 99.216 |
| | | oracle | 3.344 | 89.93 | 99.95 | 100.00 | 100.000 |
| (-1,0) | (-11.52, 0.00) | **POP** | 3.659 | 88.40 | 98.96 | 99.44 | 99.806 |
| | | chain | 3.383 | 88.94 | 98.58 | 99.23 | 99.238 |
| | | oracle | 3.341 | 90.01 | 99.94 | 99.99 | 99.993 |
| (-2,0) | (-23.03, 0.00) | **POP** | 3.695 | 87.38 | 98.66 | 99.28 | 99.722 |
| | | chain | 3.386 | 89.11 | 98.62 | 99.23 | 99.236 |
| | | oracle | 3.335 | 90.27 | 99.93 | 99.97 | 99.974 |
| (-3,0) | (-34.55, 0.00) | **POP** | 3.768 | 85.07 | 97.89 | 98.98 | 99.607 |
| | | chain | 3.397 | 89.42 | 98.75 | 99.23 | 99.240 |
| | | oracle | 3.329 | 90.53 | 99.90 | 99.94 | 99.941 |
| (-4,0) | (-46.06, 0.00) | **POP** | 3.841 | 81.74 | 96.25 | 98.24 | 99.329 |
| | | chain | 3.410 | 89.50 | 98.89 | 99.22 | 99.237 |
| | | oracle | 3.327 | 90.36 | 99.85 | 99.90 | 99.899 |
| **(-4,-2)** | (-46.06, -23.03) | **POP** | 3.902 | **79.88** | 95.41 | 97.75 | 99.089 |
| | | chain | 3.413 | **89.13** | 98.89 | 99.22 | 99.237 |
| | | oracle | 3.340 | **89.66** | 99.81 | 99.88 | 99.889 |

**Reading it.**

* **Chain vs oracle**: agree to **<= 1.2 EE3 points at every order**, and both
  are FLAT across the fan (chain 88.94-89.72, oracle 89.66-90.53). The chain's
  spot is 0.04-0.09 um wider than the oracle's throughout. Neither shows the
  monotone per-order collapse POP shows. This is the same conclusion the C5/C6
  work reached, now on a wider window and a finer lattice.
* **POP vs both, on axis**: within 1.3 EE3 points and 0.3 um FWHM of the
  oracle — POP *corroborates* the on-axis answer, inside its own convergence
  band (section 3.1).
* **POP vs both, off axis**: POP falls away monotonically, -8.8 EE3 points by
  (-4,-2), while the oracle moves -0.3 and the chain -0.6. Zemax's own
  wavefront error (section 4) says the truth is ~0. **The disagreement is POP's.**
* **Energy**: all three agree the design is essentially lossless. POP's
  0.91% outside +-20 um at the extreme order (against 0.16% on axis) is the
  same spurious halo that costs it the EE3 points.

### Against the brief's reference numbers

The brief quotes (0,0) chain 89.21 [oracle 90.08], (-4,0) 88.94 [90.78],
(-4,-2) 88.49 [89.78] — those are the **24 um square window, 0.4 um pixel,
RN=1024** instrument (`hybrid_localize_121` at n_chain=6). Re-run here at
RN=2048 on a +-20 um window at 0.1 um the same instrument gives (0,0) 90.38,
(-4,0) 90.17, (-4,-2) 89.93 (window-normalised), i.e. **the same flat profile,
+0.6 to +1.4 points from the finer/larger readout.** Nothing in this audit moves
the chain-vs-oracle picture; it only adds the third opinion.

---

## 6. Energy conservation

**POP's total power is a real transmission number — and the obvious test says it
is not.** An end-surface sweep (surfaces 1, 3, 8, 12, 20, 28, 29) returns
`SUM(I)dA = 1.000000` at *every* surface, which looks exactly like a display
renormalisation, and was written up as one in an earlier draft of this audit.
Forcing the issue (`pop_grid_diag.py --mode clip`, hard circular aperture on
surface 20) shows otherwise:

| aperture on surface 20 | SUM(I)dA | peak ratio |
|---|---|---|
| none | 1.00088885 | 1.000000 |
| R = 8 mm | 0.94728305 | 0.977766 |
| R = 5 mm | 0.67732416 | 0.354079 |
| R = 3 mm | 0.32695123 | 0.064469 |

The sweep read 1.000000 everywhere because **this design genuinely loses
nothing**. Independent confirmation, no POP involved: a Gaussian-weighted
70681-ray pupil trace over the object-space NA 0.21 gives **0 vignetted, 0
errored, geometric transmission 100.0000%** for both cfg 1 and cfg 3.

Measured POP total power at the image plane, production setting (N=4096,
W=0.2 mm), all six orders: **1.00000000** (0.99999999 for (-4,-2)). Re-measured
at the wide-array setting (N=8192, W=6.6 mm, 4517 um array): **1.00000000**, so
the 136 um production array is not truncating anything either.

Chain: the pure chain to the image plane at (0,0) with `final_leg='auto'`
returns `P(window)/P(DOE plane) = 0.998137`.

*Caveat retained:* `oracle_spot`'s absolute Rayleigh-Sommerfeld normalisation is
**not calibrated** (the `1/(i*lambda)` prefactor is missing from the kernel and
`launch_power` omits the launch-cell area), so no absolute P/Pin can be read off
the oracle. Its internal ratios are exact and are what the table uses.

---

## 7. Figures

Written to `validation/repro_traced_carrier_121/pop_profiles/`. Every figure is
**Zemax POP | lumenairy chain | exact-ray+RS oracle** on the identical lattice,
identical extent and one colour scale shared by every panel of every order
(J = I/P(window), divided by the global peak 8.208e10 /m^2/W; linear 0..1, log10
-6..0). EE3 (cyan) and EE6 (green) radii are drawn as dashed circles about the
peak. Each panel is annotated with order, field angle, FWHM, EE3/EE6/EE12 in the
`focus_scan_121.metrics()` convention, and P/Pin.

| file | contents |
|---|---|
| `pop_threeway_order_m0n0.png` | order (0,0): 3 methods x (linear, log10) + radial profile + EE curve |
| `pop_threeway_order_mm1n0.png` | order (-1,0), same layout |
| `pop_threeway_order_mm2n0.png` | order (-2,0) |
| `pop_threeway_order_mm3n0.png` | order (-3,0) |
| `pop_threeway_order_mm4n0.png` | order (-4,0) |
| `pop_threeway_order_mm4nm2.png` | **order (-4,-2), the extreme order** |
| `pop_allorders_pop.png` | POP, all six orders, linear + log10, shared scale |
| `pop_allorders_chain.png` | chain, all six orders |
| `pop_allorders_oracle.png` | oracle, all six orders |

**What the log panels show, and it is the point of the exercise.** At (0,0) the
oracle's wings fall to 1e-7 by 10 um and ring cleanly; the chain sits at ~1e-6;
**POP sits at ~1e-4 and falls off far more slowly**, i.e. two orders of
magnitude more far-field energy than either of ours, already on axis. At
(-4,-2) POP's pedestal grows and becomes asymmetric (flared toward -x,-y) while
the chain and oracle stay compact. POP's panels also carry a **visible
rectangular discontinuity in the upper-left quadrant** at the 1e-4 to 1e-5 level
— a hard-edged block boundary that no physical PSF has, present at both (0,0)
and (-4,-2). That is a numerical artefact of the POP array, and it is the same
energy that shows up as the EE3 deficit.

Raw grids alongside, so the figures can be replotted without Zemax:
`pop_m0n0.npz`, `pop_mm1n0.npz`, `pop_mm2n0.npz`, `pop_mm3n0.npz`,
`pop_mm4n0.npz`, `pop_mm4nm2.npz` (POP, resampled to the common lattice),
`chain6_*.npz` and `oracle_*.npz` (ours, native on the common lattice).
Also present, and NOT used in the figures: `chain_m0n0_auto.npz`,
`chain_m0n0.npz`, `chain_mm1n0.npz` — the *pure* chain run all the way to the
image plane (`final_leg='auto'` and `'paraxial'`). They are kept because they
are the source of the energy number in section 6 and of the observation that
the pure chain's paraxial-leg readout drops 37% of a tilted order's power
outside +-20 um, which is why the figures use the `chain6` instrument instead.
(`pop_profiles/_interim/` was not written by this audit.)

**One honesty note on our chain panel.** The chain runs on a 25.6 um DOE-plane
grid and the library warns during every run that the exit wavefront needs
`dx <= lambda/(2*NA_exit)` = 1.40-16.36 um at various groups, i.e. the coarse
grid is up to ~18x short of the exit NA. The hand-off diagnostics printed by the
chain panel's own harness are `envelope phase step 0.68-3.06 rad` (the
`hybrid_localize` guidance is "<< pi") and `integrand step p99.9 1.7-2.2 cycles`
(guidance "< 0.25"). **The chain panel's far halo below about 1e-5 is therefore
not trustworthy in detail**; its core and its EE3/EE6 are. The oracle panel has
no such caveat (its integrand step is 0.087-0.13 cycles, inside guidance).

---

## 8. Adversarial review of this POP configuration

Things that could have made POP wrong in *our* favour or against it, and what
was done about each:

1. **"POP was driven wrong."** Refuted: the harness reproduces the recorded
   paraxial-4f POP waist to 4 digits (section 2).
2. **"The order was not really isolated."** Refuted: the order is a surface
   parameter, and Zemax's chief ray lands where lumenairy's does to 8e-5
   (section 1).
3. **"POP's array was not centred on the off-axis order."** Refuted: POP centres
   on the selected field's chief ray, and the measured array centre coincides
   with the traced intercept; the spot sits within 0.5 um of the grid centre in
   every run.
4. **"The tilt ramp is aliased."** It is, by 31x at the production setting — so
   the run was repeated at N=8192/W=6.6 mm where `dx_mid = 12.0 um` is inside
   the 12.72 um Nyquist limit. The deficit persisted (6.7 points). Aliasing of
   the ramp alone does not explain it.
5. **"POP is not converged."** It is not, and it cannot be: the required
   space-bandwidth product exceeds N=8192 by ~5x (section 3). The on-axis
   convergence band is +-1 EE3 point and +-0.07 um FWHM; the off-axis deficit is
   8.8 points, well outside it, so the deficit is not noise — but neither is it
   a converged measurement.
6. **"The energy label is fake."** Tested rather than assumed, and the first
   answer was wrong: it is a real transmission (section 6).
7. **"The comparison lattices differ."** All three are on one 0.1 um / +-20 um
   lattice about each order's own chief ray, POP down-sampled onto it.
8. **"Our own EE convention was assumed."** Read out of the source and quoted
   verbatim (section 0); the brief's "diameter" reading was wrong.

**Where this leaves POP.** POP is corroborative on axis and unusable off axis
for this design. That is not a surprising result — the exit NA is 0.29-0.36, the
order is 46 mrad off axis, and POP's array cannot hold both the relay and the
focus. Reporting it as "POP says the extreme order loses 8.8 EE3 points" would
be reporting POP's grid.

---

## 9. What I could NOT determine

1. **A converged POP number at any off-axis order.** N=8192 is Zemax's cap and
   the problem needs ~5x more. Every POP off-axis number here carries an
   uncontrolled bias of the same sign (energy pushed out of the core).
2. **The absolute P/Pin of the exact-ray oracle.** `oracle_spot`'s RS kernel
   omits `1/(i*lambda)` and its `launch_power` omits the launch-cell area, so
   its absolute scale is arbitrary (measured `P_win/P_launch = 3.1e-5`,
   identical across orders). Only ratios are usable. *This is a real defect in
   our oracle's bookkeeping worth fixing* — it makes the oracle unable to answer
   an energy question at all, and it went unnoticed because the campaign only
   ever used its EE ratios.
3. **Whether the chain's far halo (below ~1e-5) is physical.** The chain's own
   hand-off diagnostics are outside their stated guidance (section 7); this
   audit did not re-run the chain on a grid fine enough to settle it.
4. **The full 8x4 order fan.** Six orders were run (0,0), (-1,0), (-2,0),
   (-3,0), (-4,0), (-4,-2). The remaining 26 were skipped: each order costs
   ~70 s of POP plus ~270 s of chain plus ~135 s of oracle, and the six chosen
   span the whole tilt range from 0 to the extreme corner. The rotational
   symmetry of the post-DOE relay means (m,n) and (n,m) are the same spot
   rotated, so the fan's corner is genuinely the worst case and it was measured.
5. **The source of POP's rectangular artefact.** It is visible and reproducible
   but was not chased to a specific POP internal (array wraparound and the
   automatic Fresnel/angular-spectrum switch are the obvious candidates).
6. **Huygens PSF as a second non-POP witness.** `zospy` 2.1.5 has no
   `HuygensPsf` wrapper in `zospy.analyses.psf`; the analysis was not driven
   through the raw API for want of time. The wavefront-error tie-breaker
   (section 4) stands on its own.
7. **Polarisation and coatings.** All POP runs used
   `use_polarization = False`; the design has no coating file applied in these
   runs. Fresnel losses are therefore excluded from every energy number here, in
   all three methods equally.

---

## 10. Scripts (all under `validation/repro_traced_carrier_121/`)

| script | role |
|---|---|
| `pop_probe_setup.py` | ZOS-API connection, system dump, order/config table, chief rays |
| `pop_run_121.py` | drives one POP run, saves the raw irradiance grid |
| `pop_grid_diag.py` | `--mode grid` grid/Nyquist audit, `--mode clip` energy-label test, `--mode vig` independent ray transmission |
| `pop_metrics_121.py` | offline scoring of a POP grid in both EE conventions, `--rebin`, `--profile` |
| `pop_control_4f.py` | the paraxial-4f control that validates the POP driving |
| `pop_wfe_121.py` | Zemax ray-based RMS wavefront error per order (the tie-breaker) |
| `pop_ours_121.py` | our two profiles on the common lattice (`--method oracle` / `chain6` / `chain`) |
| `pop_plot_threeway.py` | the three-way figures and the comparison table |

Reproduce the headline in three commands:

```
.venv-zemax/Scripts/python.exe pop_control_4f.py --nx 2048 --width 0.1 0.2 0.4
.venv-zemax/Scripts/python.exe pop_wfe_121.py
.venv-zemax/Scripts/python.exe pop_run_121.py --m -4 --n -2 --nx 4096 --width 0.2 --crop-um 20.05 --tag prod
```
