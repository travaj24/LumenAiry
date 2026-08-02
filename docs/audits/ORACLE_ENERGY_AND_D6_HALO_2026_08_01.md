# The exact-ray oracle's absolute energy, and niche D6's halo firing

**Date** 2026-08-01 - **Tree** `fix/pmm-union-grid-conditioning` @ `8e7b156`
(plus another agent's UNCOMMITTED niche-C8 work in
`lumenairy/elements/_lens_traced.py` - see S5.0) - **Scope** two independent
questions: **(A)** make `exact_ray_oracle_121.oracle_spot` able to answer an
ABSOLUTE energy question, left as "could not determine" by
`POP_CROSSCHECK_121_2026_07_31` S9.2; **(B)** diagnose the halo-check firing
`C6_FIT_GUARD_DECISION_2026_07_31` S5.2 found in niche D6's fixture.

**Nothing under `lumenairy/` was modified by this study.** The library files
were read only. Everything changed is under
`validation/repro_traced_carrier_121/`.

---

## 0. Headline

### A. The oracle's absolute scale, and the POP audit's own arithmetic

Two defects, one in the oracle and one in the POP harness:

1. **`oracle_spot`'s Rayleigh-Sommerfeld kernel omitted the `1/(i lambda)`
   prefactor**, so every intensity it returned was too large by
   `1/lambda^2` = **5.827e11**. Fixed.
2. **`oracle_spot`'s `launch_power` has ALWAYS carried the launch-cell area
   `h^2`** - the line that defines it ends `... * h * h`. The POP audit's S9.2
   claim that it does not is **wrong**; the double count is in that audit's own
   harness, `pop_ours_121.py`, which multiplied by `h^2` a second time. Fixed
   there.

The two together are the *entire* explanation of the "arbitrary" ratio the POP
audit reported, and the arithmetic closes exactly:

```
reported = true x lambda^2 / h^2 ,
h = 2 x clip x w_doe / (NL-1) = 236.93 um   (w_doe 6.3181 mm, clip 3, NL 161)
lambda^2 / h^2 = 3.0587e-05          POP audit's measured value: 3.1e-05
```

so `3.1e-05 x h^2 / lambda^2 = 1.014` - unity to the two significant figures
the audit quoted. **There was never anything wrong with the physics; it was two
bookkeeping errors that happened to multiply.**

**With both fixed, the oracle's absolute answer for design 121 is that the
design is lossless**, and it converges to it:

| order | NL=161 | NL=241 | NL=321 | **NL=401** |
|---|---|---|---|---|
| **(0,0)** | 100.002971 % | 100.001116 % | 100.000460 % | **100.000159 %** |
| **(-4,-2)** | 100.011990 % | 100.006098 % | 100.004463 % | **100.003218 %** |

(`P_out/P_in` as a true z-flux, +/-16 um readout, `back` = 10 mm; S3.3.)
**Both orders converge on 1.0000 monotonically in the launch density**, and the
residual is launch-lattice quadrature, not physics. Validated first on a case
with an analytic answer - an unaberrated converging sphere driven through the
same machinery - where the measurement gives **0.99999988** (S2).

**The `ENERGY_CONSERVATION_AUDIT_2026_07_31` conservation reference is
untouched, and this is measured rather than argued** (S4): its "live power
100.0000 %" is `p_live / p_launch`, a ratio of two sums that both carry `h^2`
and neither of which passes through the RS kernel. Re-running the audit's own
command on the fixed oracle reproduces **every printed digit** of its S1.5
table. **No conclusion of that audit is affected.**

### B. Niche D6's 0.641-of-peak lobe

**It is a REAL manufactured lobe, not a support-definition artefact - and the
C7 record has the two D6 calls swapped.**

* **Wrong call.** The call that warns is the **`final_leg='paraxial'` leg's
  ordinary group call** on the chain's own 1024-point / 6 mm grid. The **exact
  tilted leg's retrace** (2048 points / 3.6 mm) is the one that is **DECLINED**
  (corners-only), and its own untrustworthy reading is the **0.841** the C7
  record attributes to "D6's coarse call". So the defect is **not in D6's
  exact-tilted-leg feature**; it is the general ray-density Newton-inverse
  extrapolation defect, exposed by D6's decentred fixture on the ordinary
  paraxial path (S6.1).
* **Diffraction was tested first and is not available.**
  `apply_real_lens_traced` is a **pure eikonal operator** - `E_out` is
  `amp(pullback) * exp(i k0 opl)` masked to the ray-covered region, with no
  propagation integral, no FFT and no angular spectrum anywhere in it. And the
  *physical* ceiling was computed anyway: the shadow edge on the lobe's side is
  cast by the aperture rim at `x = -1.700 mm`, where the incident amplitude is
  **4.5195e-07 of peak**; a boundary-diffraction wave cannot exceed its own
  edge value, so the lobe is **1.42e6 x** anything diffraction permits there
  (S6.3).
* **No traced ray goes there: 0 of 12849 alive rays.** The whole alive exit map
  spans `|x|,|y| <= 1.2437 mm` (max 1.2576 mm from the axis); the lobe sits at
  1.4607-1.7000 mm from the axis. The check's `e^-9` contour DOES understate
  the true support (`r_hull` 1.6161 mm against 1.8115 mm over all alive rays),
  but the `1.25 x` factor more than covers the gap. **The warning's central
  claim - "NO TRACED RAY OF THIS CALL REACHES THAT RADIUS" - is TRUE.**
* **It is a function of numerics and of the decentre, not of the optics.**
  Exactly `0.000e+00` at `ray_subsample=1`, at `fit_radius_beam_factor=3.0`,
  and at carrier decentres of 0, 0.25 w, 0.5 w, 0.75 w on the same optic, the
  same aperture and the same grid; it appears only at the fixture's own 1.0 w
  (S6.2, S6.4). Amplitude gain over the local incident field: **2.55e5**.
* **Neither dependent test asserts anything the lobe invalidates.** All twelve
  assertions of the two tests pass with the lobe present AND removed; the
  largest movement in an asserted quantity is **0.33 EE2 points** against a
  10 % bar (S7). But the lobe pushes the chain-exit power **above unity**
  (`p_exit/p_in` = **1.000534** exact leg, **1.000734** paraxial), and the one
  test that would notice asserts `0.95 < p_ex/p_in < 1.02` - a band **38x too
  wide** to see it.

**No library fix is made here.** The library is another agent's, and that agent
is already implementing precisely this fix (niche C8,
`REMAP_INVERSE_SUPPORT_BOUND`, uncommitted in the tree) - which does remove the
lobe exactly (S5.0, S7). Two observations for its owner are in S7.

---

# PART A -- THE ORACLE'S ABSOLUTE ENERGY

## 1. The defect, stated exactly

`validation/repro_traced_carrier_121/exact_ray_oracle_121.py`, `oracle_spot`.
The integrand is built correctly:

```
W_j       = |E_j| sqrt(n0_j J_j / N_j) h^2        (= E_exit,j * dA_exit,j)
E(P)  <-  SUM_j  W_j * back / rho_j^2 * exp(i (ph_j + k rho_j))
```

`W_j` is right: equating the launch-cell flux `|E|^2 cos(theta) h^2` to the
exit-cell flux `|E_exit|^2 N_e J h^2` gives `|E_exit| = |E| sqrt(n0/(N_e J))`,
and `E_exit dA_exit` is then exactly `W_j`. The kernel shape is right: `back /
rho^2` is `cos(theta) / rho`, the first-Rayleigh-Sommerfeld obliquity. **What
is missing is the prefactor `1/(i lambda)`.** (The exact RS-I kernel is
`(1/2pi)(z/rho)(ik - 1/rho) exp(ikrho)/rho`; the `1/(k rho)` correction is
4e-5 at `back` = 5 mm and irrelevant here.)

Measured, not asserted (`oracleE_rs_control.py`, `[PREFACTOR]`):

```
P_sq with the prefactor    = 3.254078e-07
P_sq without               = 5.584323e-19
ratio                      = 1.716100e-12     lambda^2 = 1.716100e-12
P/Pin without the prefactor = 1.725864e-12
```

The second defect is in `pop_ours_121.py`, which carried this comment:

> `# oracle_spot's ``launch_power`` is sum(amp^2 * cos) WITHOUT the`
> `# launch-cell area, while its integrand weight carries h^2 -- so`
> `# the physical launched power needs the h^2 back.`

It is not: `p_launch = float(np.sum((amp ** 2 * n0)[amp > 0])) * h * h`. The
harness's extra `* h * h` divided `P_ratio` by `h^2 = 5.61e-08`.

### 1.1 Two image-plane powers, and why only one of them is the answer

* `P_sq = SUM |E|^2 dA`. By Parseval this is `INT |A(k)|^2 d2k`, which **omits
  the obliquity `kz/k`**.
* `P_flux = (1/k) Im INT E* dE/dz dA = INT |A(k)|^2 (kz/k) d2k` - the true
  z-directed power.

The launch side `P_in = SUM |E|^2 cos(theta) h^2` **is** a flux, so `P_flux` is
the quantity that must come back as `P_in`; `P_sq` overshoots by `1/<cos>` over
the field's own cone. On design 121 that is **+0.588 %** and it is not an
error - it is what `SUM |E|^2 dA` means at NA 0.36. Both are now returned
(`P_window_flux`, `P_window_sq`), with `P_ratio_flux` the headline.

### 1.2 What changed, and what provably did not

| file | change |
|---|---|
| `exact_ray_oracle_121.py` | `1/(i lambda)` on the RS kernel; second accumulator for `dE/dz`; `P_window_sq` / `P_window_flux` / `P_ratio_sq` / `P_ratio_flux` / `P_ratio_flux_live` / `live_power` / `E` returned; `ray_step_w_p999` added alongside the existing max-based `ray_step_weighted`; `flux=False` opt-out |
| `hybrid_localize_121.py` (`rs_spot`, the same integral, the chain6 instrument) | identical prefactor + flux treatment; `P_launch` / `P_live` / `P_sq` / `P_flux` / `P_ratio_flux` added |
| `pop_ours_121.py` | the double `h^2` removed; `P_ratio` is now `P_ratio_flux` and `P_ratio_sq` reported alongside |

**Every EE / FWHM the campaign published is unchanged.** The prefactor is a
pure constant, so it cancels out of every ratio; the residue is double-
precision rounding. Measured on order (0,0) at `NL=81` by undoing the constant
and re-deriving the metrics (`oracleE_absolute_121.py --PART=null`):

```
EE (centroid) with prefactor  : 90.0798069780  99.9458851621  99.9999314691
EE (centroid) prefactor undone: 90.0798069780  99.9458851621  99.9999314691
worst relative EE change: 1.232e-16              (one ulp)
```

Differential floor, established before any of the above:
`array_equal(E) = True`, `max|dI| = 0.000e+00` on two identical runs.

---

## 2. Validation on a case with an analytic answer

`validation/repro_traced_carrier_121/oracleE_rs_control.py`.

**The case.** An **unaberrated converging sphere** in vacuum: a Gaussian
launch lattice (`E = exp(-r^2/w^2)`, clipped at 3 w, so 1.5e-08 of the power is
outside the square), every ray aimed exactly at `(0, 0, f)`, launch OPL
`-k*sqrt(x^2+y^2+f^2)` so all rays are in phase at the focus. The exact answer
is `P_out/P_in = 1`.

**Nothing analytic is substituted for a step the oracle performs numerically**:
the transport is `lumenairy.raytrace.trace` over two flat air surfaces, the
exit Jacobian is the same central difference, the amplitude is the same
ray-density expression, the quadrature is the same RS sum. Only the optics are
removed.

Differential floor first: two identical runs, **`array_equal = True`,
`max|dE| = 0.000e+00`**.

### 2.1 The answer, over NA

Readout auto-sized to `+/-6 w_f` at 6 points per `w_f`, `f` = 3 mm,
`back` = 0.5 mm, `NL` swept 81 - 241 (all four agree to 8 digits, see 2.2):

| NA | w [mm] | `w_f` [um] | **`P_flux/P_in`** | `P_sq/P_in` | `1/<cos>` | `P_sq/P_flux` |
|---|---|---|---|---|---|---|
| 0.010 | 0.0300 | 41.70 | 0.99825789 | 1.000024 | 1.000025 | 1.001770 |
| 0.050 | 0.1502 | 8.33 | 0.99993014 | 1.000625 | 1.000626 | 1.000695 |
| 0.150 | 0.4551 | 2.75 | 0.99999198 | 1.005690 | 1.005690 | 1.005698 |
| **0.333** | 1.0595 | 1.18 | **0.99999817** | 1.029484 | 1.029484 | 1.029485 |

Two things to read here.

1. **`P_flux/P_in` -> 1**, and at design 121's own exit NA it is **1.8e-06**
   from unity. That is the validation the brief asked for.
2. **`P_sq/P_in` is `1/<cos>` to six digits at every NA** (1.029484 against
   1.029484 at NA 0.333). The `SUM |E|^2 dA` convention is not noise, it is a
   known, computable offset - which is why the absolute answer is quoted as a
   flux.

**The NA 0.010 row is a control on the control** and is worth keeping: at
`f` = 3 mm a 30 um beam has Fresnel number 0.23 and `back` = 0.5 mm sits well
INSIDE the focal region (`z_R` = 4.2 mm), so the geometric ray field at the
"exit plane" is a 5 um spot where the true field is 42 um wide. Its angular
spectrum is therefore broader than the launch cone, and its z-flux is genuinely
0.17 % below the launched flux. **The geometric hand-off is only valid when the
exit reference plane is outside the focal region**, and the `back` sweep in 2.3
shows the same effect converging away.

### 2.2 Launch density and readout window

`P_flux/P_in` is **identical to 8 digits at `NL` = 81, 121, 161, 241** at every
NA above. That is not a null result, it is structural: `P_sq` at the image
plane equals the exit-plane `SUM |E|^2 dA` by Parseval, which is
`SUM amp^2 (n0/N_e) h^2` - the SAME launch-grid quadrature as `P_in`, so the
ratio is insensitive to the lattice as long as the RS sum itself is resolved.
Where the launch density DOES bite is design 121, whose ray map is not
free-space and whose `P/Pin` moves monotonically with `NL` (S3.3).

Readout window (NA 0.333, `NL` 241):

| half-width | `P_flux/P_in` | `P_sq/P_in` |
|---|---|---|
| 1.0 `w_f` | 0.91614356 | 0.94823797 |
| 2.0 `w_f` | 0.99982556 | 1.02927292 |
| 3.0 `w_f` | 0.99999511 | 1.02947973 |
| 4.0 `w_f` | 0.99999807 | 1.02948343 |
| 6.0 `w_f` | 0.99999817 | 1.02948355 |
| 8.0 `w_f` | 0.99999817 | 1.02948355 |

**3 `w_f` is enough to 5e-06; 4 `w_f` to 2e-06.**

### 2.3 Exit-reference-plane placement

The RS integral is exact for any placement, so a dependence on `back` is a
defect of the hand-off, not of the integral (NA 0.333, `NL` 241):

| `back` | `P_flux/P_in` | `P_sq/P_in` |
|---|---|---|
| 0.20 mm | 0.99998855 | 1.02948099 |
| 0.50 mm | 0.99999817 | 1.02948355 |
| 1.00 mm | 0.99999954 | 1.02948391 |
| **2.00 mm** | **0.99999988** | 1.02948400 |

Monotone toward unity as the reference plane leaves the focal region - the same
mechanism as the NA 0.010 row, now converged away. **1.2e-07 from the exact
answer is the floor this machinery reaches.**

### 2.4 The twin

`hybrid_localize_121.rs_spot` (the chain6 instrument) carries the same integral
and had the same omission. Same control, `NL` = 81, NA 0.316:
**`P_flux/P_launch` = 0.99999799**, `P_sq/P_launch` = 1.026413,
`live` = 1.000000.

---

## 3. Design 121's absolute `P/Pin`

`validation/repro_traced_carrier_121/oracleE_absolute_121.py`. Chain A cached
at `RN=1024`, `rs=4`; DOE plane `R` = 703642.7361 mm, `dx` = 51.2334 um.
`clip` = 3.0.

### 3.1 Order (0,0)

| `dx_out` [um] | `N_out` | half-window [um] | `NL` | live % | **`P_flux/P_in` %** | `P_sq/P_in` % | integrand step p99.9 [cyc] |
|---|---|---|---|---|---|---|---|
| 0.80 | 41 | 16.00 | 161 | 100.0000 | **100.002932** | 100.5876 | 0.06900 |
| 0.80 | 81 | 32.00 | 161 | 100.0000 | 100.003178 | 100.5879 | 0.13801 |
| 0.80 | 151 | 60.00 | 161 | 100.0000 | 100.003286 | 100.5880 | 0.25884 |
| 0.80 | 251 | 100.00 | 161 | 100.0000 | 100.003353 | 100.5880 | 0.43129 |
| 0.10 | 261 | 13.00 | 161 | 100.0000 | 100.002790 | 100.5874 | 0.05612 |
| 0.80 | 151 | 60.00 | 81 | 100.0000 | 100.013539 | 100.5983 | 0.51581 |
| 0.80 | 151 | 60.00 | 121 | 100.0000 | 100.005898 | 100.5906 | 0.34393 |
| 0.80 | 151 | 60.00 | 201 | 100.0000 | 100.002084 | 100.5868 | 0.20695 |
| 0.80 | 151 | 60.00 | 241 | 100.0000 | 100.001424 | 100.5861 | 0.17256 |

`back` sweep at `NL`=161, `+/-60 um`: 2 mm 100.003265 %, 5 mm 100.003286 %,
10 mm 100.003306 % - **flat to 4e-07 over a 5x change in the reference plane**,
i.e. the hand-off is fully in the geometric regime here (unlike S2.1's NA
0.010 row).

Only **4.2e-06 of the power sits outside `+/-16 um`** and 1.8e-06 outside
`+/-32 um` (differences of the `P_flux/P_in` column against the `+/-100 um`
row); the spot is that compact.

### 3.2 Order (-4,-2)

| `dx_out` [um] | `N_out` | half-window [um] | `NL` | live % | **`P_flux/P_in` %** | `P_sq/P_in` % | step p99.9 |
|---|---|---|---|---|---|---|---|
| 0.80 | 41 | 16.00 | 161 | 99.9999 | **100.029499** | 100.6145 | 0.06894 |
| 0.80 | 81 | 32.00 | 161 | 99.9999 | 100.047357 | 100.6334 | 0.13790 |
| 0.80 | 151 | 60.00 | 161 | 99.9999 | 100.078920 | 100.6668 | 0.25854 |
| 0.80 | 251 | 100.00 | 161 | 99.9999 | 100.131587 | 100.7226 | 0.43075 |
| 0.10 | 261 | 13.00 | 161 | 99.9999 | 100.025787 | 100.6105 | 0.05599 |
| 0.80 | 151 | 60.00 | 81 | 100.0000 | 100.189757 | 100.7836 | 0.51501 |
| 0.80 | 151 | 60.00 | 121 | 99.9999 | 100.092040 | 100.6806 | 0.34382 |
| 0.80 | 151 | 60.00 | 201 | 99.9999 | 100.051384 | 100.6377 | 0.20673 |
| 0.80 | 151 | 60.00 | 241 | 99.9999 | 100.038210 | 100.6238 | 0.17245 |

`back` sweep at `NL`=161, `+/-60 um`: 2 mm **100.400903 %**, 5 mm 100.078920 %,
10 mm 100.025203 %.

**The tilted order is quadrature-limited and says so in three consistent
ways**: the excess grows with the window (0.029 -> 0.132 % from `+/-16` to
`+/-100 um`), falls with the launch density (0.190 -> 0.038 % from `NL` 81 to
241), and falls with `back` (0.401 -> 0.025 % from 2 to 10 mm) - exactly the
three knobs that control the integrand phase step, which is printed alongside
and tracks them (0.069 -> 0.431 cycles across the window sweep, against the
oracle's own "<< 0.5, guidance 0.25" rule). It is the launch lattice, not the
optics. The tilt puts a 562-wave piston across the beam edge, so the tilted
order's integrand is intrinsically harder to quadrature than the on-axis one.

### 3.3 The converged answer

Run at the settings the sweeps identify as least biased - `+/-16 um` (holds all
but 4e-06 of the power) and `back` = 10 mm - with the launch density swept:

| order | `NL`=161 | `NL`=241 | `NL`=321 | **`NL`=401** | step p99.9 at 401 |
|---|---|---|---|---|---|
| **(0,0)** | 100.002971 | 100.001116 | 100.000460 | **100.000159 %** | 0.02764 cyc |
| **(-4,-2)** | 100.011990 | 100.006098 | 100.004463 | **100.003218 %** | 0.02759 cyc |

**Both orders converge monotonically on 100.0000 %.** The answer to the brief's
question is therefore:

> **With correct bookkeeping the oracle's `P/Pin` is 1.0000 for order (0,0) and
> 1.0000 for order (-4,-2)** - `1.0000016` and `1.0000322` at `NL` = 401, both
> still falling with `NL`, so quote them as `1.0000 (+3e-05 / -0)`.

This **agrees with POP** (which reports 1.00000000 at every order), with the
independent 70681-ray pupil trace (0 vignetted, geometric transmission
100.0000 %), and with the oracle's own live power (100.0000 % / 99.9999 %).
Design 121 is lossless, and now all three methods say so on an absolute scale.

*Caveat kept explicit:* no Fresnel reflection is modelled in this path (nor in
POP's `use_polarization=False` runs), so "lossless" means "no vignetting and no
aberration loss", not "no surface loss".

---

## 4. Reconciliation with `ENERGY_CONSERVATION_AUDIT_2026_07_31`

**The question.** That audit's S1.5 is titled "The conservation reference" and
reports the oracle's **live power 100.0000 %** on every order, then uses it to
justify "the reference value for `P_out/P_in` is therefore 1.0000 exactly" -
which in turn sets the C1a upper bound 1.00020 and the C1b deficit floor. If
that number came through the broken path, several bounds would move.

**It did not, and the reason is structural.** `live_frac` is

```
p_live   = SUM_{good} |E|^2 cos(theta) h^2
p_launch = SUM_{amp>0} |E|^2 cos(theta) h^2
live_frac = p_live / p_launch
```

Both sums carry `h^2`; neither touches the RS kernel, the prefactor, or the
image plane at all. It is a **geometric vignetting fraction**, and it is exactly
invariant under both defects.

**Measured, not argued.** The audit's own command re-run on the fixed oracle
(`ORD='0,0;-1,0;-4,0;-4,-2' NL=161 NOUT=61 DXO=0.2 CLIP=3.0`):

| order | live power % | dead rays | exit NA | step (amp-wtd max) | EE3 % | EE6 % | *new:* `P_flux/P_in` % |
|---|---|---|---|---|---|---|---|
| (0,0) | **100.0000** | 13356 | 0.3596 | 0.0262 | 89.88 | 99.97 | 99.9804 |
| (-1,0) | **100.0000** | 13441 | 0.3616 | 0.0262 | 90.04 | 99.97 | 99.9817 |
| (-4,0) | **100.0000** | 13726 | 0.3667 | 0.0261 | 90.45 | 99.97 | 99.9915 |
| (-4,-2) | **99.9999** | 13777 | 0.3674 | 0.0261 | 90.14 | 99.96 | 99.9963 |

**Every column the audit printed is reproduced to every digit** - live power,
dead-ray counts, exit NA, integrand step, EE3, EE6.

**Verdict: no conclusion of `ENERGY_CONSERVATION_AUDIT_2026_07_31` is
affected.** Specifically:

* S1.5's "reference value 1.0000 exactly" **stands** - and is now supported by
  an absolute measurement (S3.3) as well as by the vignetting fraction.
* S2's `P_out/P_ap` table, S3's halo/second-moment tables and S6's six bounds
  are all **chain-internal ratios of `SUM |E|^2 dx^2` on library grids**. They
  never call `oracle_spot` and never see the RS kernel.
* The one place the oracle enters quantitatively is S3.1's exact-ray `g4`
  ceiling and `r_rms` reference, produced by `energy_hull_121.py` - which is a
  **ray trace + hull**, not an RS integral. Unaffected.
* The new bottom-right column is the only thing the fix adds to that document:
  at its readout (`+/-6 um`), 0.004-0.02 % of the power lies outside the
  window. That is a readout statement, not a loss.

**The suspicion in the brief was reasonable and the answer is: nothing to
correct there.** The correction belongs to `POP_CROSSCHECK_121_2026_07_31` S6
(the retained caveat) and S9.2 (the "could not determine"), both of which are
now answered, and to S9.2's *reason*, half of which was itself wrong.

---

# PART B -- NICHE D6's HALO FIRING

## 5. Provenance, and one thing the record could not have known

### 5.0 The library moved under this study

`git status` at the start of this work:

```
 M lumenairy/elements/_lens_traced.py     (+94 lines, UNCOMMITTED)
?? validation/repro_traced_carrier_121/probe_c8_support_bound.py
?? validation/repro_traced_carrier_121/probe_c8_synthetic.py
?? validation/repro_traced_carrier_121/c8_with_bound.py
```

sha256 of the working-tree file `602d8e802ed21007`, of `HEAD`'s
`4874960c8d14c9d9`. **Another agent is implementing niche C8 right now** - the
"structural cure" the post-C6 audit named: `REMAP_INVERSE_SUPPORT_BOUND`
(default `True`), a convex hull of the traced exit landings with a raised-cosine
feather, multiplied onto the ray-density amplitude so an exit pixel with no
traced data behind it is tapered to zero.

**On the working tree as it stands, D6's halo warning is already gone**
(`amax_halo` exactly `0.000e+00`). All diagnosis below is therefore run with
`REMAP_INVERSE_SUPPORT_BOUND = False`, which the flag documents as restoring
pre-C8 behaviour bit for bit - and which reproduces the C7 record's numbers
exactly (`amax_halo` 6.405e-01, bound 2.0202 mm, `r_hull` 1.6161 mm, reach
2.4341 mm), so the restoration is verified rather than trusted.

### 5.1 Instruments and floors

| instrument | what it does | new? |
|---|---|---|
| `d6halo_probe.py` | wraps `lumenairy.elements.apply_real_lens_traced` AND `lumenairy.raytrace.trace`, so every element call is captured together with the EXACT ray bundle its own tracer produced (read after the call returns, so the vertex correction and the carrier eikonal are included); recomputes the hull statistic under the library's rule and under relaxed rules; replays the call under alternative knobs | **new** |
| `d6halo_controls.py` | the four controls A-D | **new** |
| `test_niche_d6_exact_tilted_leg.py` | the fixture itself, imported unmodified (its `_run_chain`, `_metrics`, `_oracle_on_grid`) | reused verbatim |

Floor: the wrapper is a pass-through, and `PART=A` re-derives the library's own
reported `amax_halo` / `r_hull` / centroid / grid-reach from the captured field
and matches the warning text to 4 digits.

`g_halo` here is `SUM|E|^2 / SUM|E_in|^2` over the whole input grid; the
library divides by the *aperture-transmitted* power. That is the whole of the
6.447e-04 vs 6.449e-04 difference.

---

## 6. The diagnosis

### 6.1 The record names the wrong call

`d6halo_controls.py PART=A`, C8 bound OFF:

| leg | element calls | grid | `r_hull` | bound | grid reach | verdict | `amax` | library warned |
|---|---|---|---|---|---|---|---|---|
| `paraxial` | 1 | N=1024, dx 5.8594 um, half-width 2.9941 mm | 1.6161 mm | 2.0202 mm | **2.4341 mm** | **scored (full annulus)** | **6.405e-01** | **YES** |
| `exact` | 1 | N=2048, dx 1.7567 um, half-width 1.7971 mm | 1.6193 mm | 2.0241 mm | 1.2375 mm | **DECLINED** | (8.408e-01) | no |

The `exact` leg's single element call is `_fine_trace_group_exit`'s **retrace**
- widened window `2 max(|x_c|,|y_c|) + window_factor w` = 3.6 mm, `n_fine`
2048, exactly as D6's docstring describes. The `paraxial` leg has no retrace;
its single call is the ordinary group call on the chain's own 1024 / 6 mm grid.

`C6_FIT_GUARD_DECISION_2026_07_31` describes the 3.6 mm / 1.239 mm-reach call
as "D6's **coarse** call" (S5.2.1) and the 2.4341 mm-reach one as "D6's
exact-tilted-leg **RETRACE**" (S5.2, S7.10, S8.10). **Those two labels are
exchanged.** The consequences:

* the defect is **not** a defect of niche D6's exact tilted leg;
* both named tests run BOTH legs, so both still see the warning - via their
  `final_leg='paraxial'` arm;
* the exact leg's retrace is not clean either, it is merely **unmeasurable** on
  its own grid (bound 2.0241 mm against a 1.2375 mm reach), and its
  corners-only reading of 0.841 is the number the record already treats as
  untrustworthy. Its manufactured power is, if anything, larger
  (`g` = 1.076e-03 against 6.447e-04 - see S7's stage powers).

### 6.2 Where the light is, and whether any ray goes there

`d6halo_probe.py`, C8 OFF, `LEG=paraxial`:

```
alive rays                                   12849 of 47089 launched
above the e^-9 amplitude contour             12228
traced exit map, ALIVE rays  x in [-1.2437, +1.2437] mm
                             y in [-1.2437, +1.2437] mm
                             max radius from the OPTICAL AXIS 1.2576 mm
                             (aperture radius 1.7000 mm)
traced exit centroid                         (+0.5600, -0.0000) mm
r_hull, e^-9 gate (the library's)            1.6161 mm  -> bound 2.0202 mm
r_hull, ALL alive rays, no amplitude gate    1.8115 mm  -> bound 2.2644 mm

ALIVE TRACED RAYS beyond the halo bound (2.0202 mm):   0 of 12849
```

**The support-definition hypothesis is refuted here.** The `e^-9` contour *does*
understate the exit support - by 0.195 mm, 1.6161 against 1.8115 mm - so it is
a real (if modest) conservatism in the check. But the `1.25 x` radius factor
carries the bound to 2.0202 mm, which is **beyond the full traced support as
well**. No ray of any amplitude reaches the annulus.

Where the lobe actually sits:

```
brightest halo pixel   (-1.4297, +0.7031) mm
                       = 1.5932 mm from the OPTICAL AXIS
                       = 2.1103 mm from the traced exit centroid
halo power centroid    (-1.5467, +0.1513) mm
halo pixels' radius from the axis  1.4607 .. 1.7000 mm
                       (2550 nonzero pixels of 675130 in the annulus)
|E_in| at the brightest halo pixel   2.7150e-06
|E_out| there                        6.9257e-01      -> gain 2.55e+05
peak |E_out| on the grid             1.0813
```

So it is a **thin crescent hard against the aperture rim, on the side of the
axis OPPOSITE the beam** - between the traced exit support (which stops at
1.2576 mm from the axis because the exit is converging) and the aperture mask
(1.7000 mm, which is an ENTRANCE stop applied at the EXIT coordinate and
therefore does not bound a converging exit). That is exactly the geometry in
which the fitted entrance->exit map is Newton-inverted outside its own data and
can fold back into the bright beam.

### 6.3 What diffraction permits there - the innocent explanation, tested first

**Structural.** `apply_real_lens_traced` assembles

```
E_out = amp(Newton-pullback) * exp(i k0 opl_map),  masked by isfinite(opl_map)
        and by the entrance aperture,
```

with no propagation integral, no FFT and no angular spectrum. **The operator
contains no diffraction at all**, so light beyond its own ray support cannot be
diffraction by construction. (This is not a technicality: it is why the halo
check is a legitimate instrument for this operator in the first place.)

**Physical, computed anyway** (`d6halo_controls.py PART=C`). If one asks what
the TRUE wave field would permit at that plane:

| bound | value | ratio to the observed 6.4052e-01 |
|---|---|---|
| incident amplitude at the aperture rim that casts this shadow edge (`x = -1.700 mm`) - a boundary-diffraction wave cannot exceed its own edge value | **4.5195e-07 of peak** | **1.417e+06 x** |
| incident amplitude AT the halo pixels themselves (what a perfect diffraction-free screen would put there) | 1.8573e-05 of peak | 3.449e+04 x |

**Numerical, and this one needs no theory at all.** The same element call, same
optic, same aperture, same grid, same wavelength, same field, at
`ray_subsample = 1` instead of 4 - a pure discretisation knob that cannot
change the physics:

```
ray_subsample = 4 (shipped)  ->  amax_halo 6.405e-01,  2550 nonzero far pixels
ray_subsample = 1            ->  amax_halo 0.000e+00,     0 nonzero far pixels
```

**Diffraction is excluded three independent ways.**

### 6.4 What creates it: the decentre and the fit, not the optics

`d6halo_controls.py PART=B` - paraxial leg, C8 OFF, only the carrier decentre
moves:

| `x0` [mm] | `x0/w` | `r_hull` [mm] | bound [mm] | reach [mm] | `amax_halo` | `g_halo` | nonzero px | warned |
|---|---|---|---|---|---|---|---|---|
| 0.0000 | 0.00 | 1.2576 | 1.5721 | 2.9941 | **0.000e+00** | 0.000e+00 | 0 | no |
| 0.1500 | 0.25 | 1.3993 | 1.7491 | 2.8502 | **0.000e+00** | 0.000e+00 | 0 | no |
| 0.3000 | 0.50 | 1.5357 | 1.9197 | 2.7075 | **0.000e+00** | 0.000e+00 | 0 | no |
| 0.4500 | 0.75 | 1.5817 | 1.9771 | 2.5679 | **0.000e+00** | 0.000e+00 | 0 | no |
| **0.6000** | **1.00** | 1.6161 | 2.0202 | 2.4341 | **6.405e-01** | 6.447e-04 | 2550 | **YES** |

On axis the entrance->exit map is radial and the fitted inverse has nothing to
fold; the lobe is **exactly zero** and stays exactly zero out to three quarters
of a beam radius of decentre. It appears only at the fixture's own full-radius
decentre - the configuration D6 was built to stress.

And it moves with the FIT, not with anything physical
(`d6halo_probe.py PART=replay`, all on the same captured call):

| variant | `amax_halo` | `g_halo` | far pixels |
|---|---|---|---|
| as shipped (`ray_density`, `preserve_input_phase='remap'`) | **6.405e-01** | 6.447e-04 | 2550 |
| `preserve_input_phase=True` | 6.405e-01 | 6.447e-04 | 2550 |
| `ray_subsample=1` | **0.000e+00** | 0.000e+00 | 0 |
| `fit_radius_beam_factor=1.0` | 3.368e-01 | 2.890e-05 | 211 |
| `fit_radius_beam_factor=1.5` (shipped) | 6.405e-01 | 6.447e-04 | 2550 |
| `fit_radius_beam_factor=3.0` | **0.000e+00** | 0.000e+00 | 0 |
| as shipped + niche-C8 support bound | **0.000e+00** | 0.000e+00 | 0 |

**A NEGATIVE RESULT WORTH RECORDING** (and the one place this study's first
instinct was wrong): `amplitude_model='screen'` is **not** a valid probe of the
Newton pullback amplitude here, and reading it as one would have inverted the
verdict. It returns `amax_halo` = 0.000e+00 with *zero* nonzero far pixels -
which looks like "the pullback carries no amplitude" and therefore like an
artefact. It is not: `amplitude_model='ray_density'` **forces**
`newton_amp_mask_rel = 0.0` (`_lens_traced.py` ~L4168, with an explicit
`ValueError` if the caller disagrees), while `screen` keeps the 1e-4 default -
so the two models solve the Newton inverse on **different domains**, and screen
simply never evaluates the annulus. Any screen-vs-ray_density halo comparison
must pass `newton_amp_mask_rel=0.0` explicitly or it is comparing supports, not
amplitudes.

### 6.5 Verdict

**REAL LIBRARY DEFECT.** Manufactured light, in the shipped ray-density
amplitude of `lumenairy.elements.apply_real_lens_traced`.

| | |
|---|---|
| **site** | `lumenairy/elements/_lens_traced.py`, the `_ray_density_amp_grid` / `_invert_newton` pullback: an exit pixel outside the traced exit support has no data behind it, the fitted map is extrapolated there, and on a DECENTRED (non-radial) map the inverse folds back into the bright beam and is handed real ray-density amplitude |
| **magnitude** | 6.405e-01 of peak (2.55e5 x the local incident amplitude); 6.447e-04 of the input power; 2550 pixels; a crescent at 1.4607-1.7000 mm from the axis, 0.29 mm beyond the outermost traced ray |
| **conditions** | `amplitude_model='ray_density'`; a beam decentred by ~1 full beam radius inside a much larger clear aperture; `ray_subsample >= 2`; `fit_radius_beam_factor` ~1.0-1.5. Absent at decentre <= 0.75 w, at `ray_subsample=1`, at `fit_radius_beam_factor=3.0` |
| **already flagged?** | Yes, by the library itself: the halo self-check fires with the right diagnosis in its message, and the fold-caustic warning fires on the same call |
| **already fixed?** | Yes, in the uncommitted niche-C8 work in this very tree (S5.0) - `REMAP_INVERSE_SUPPORT_BOUND=True` takes it to exactly 0.000e+00 |

The C7 record's own reading of this firing ("a previously unreported
manufactured lobe... it is recorded, not silenced") is **correct on the
substance**. Only the call attribution is wrong.

---

## 7. What the two dependent green tests actually see

`d6halo_controls.py PART=D`. Every assertion of
`test_exact_beats_paraxial_for_a_tilted_congruence_against_the_oracle` and
`test_the_tilted_exact_leg_conserves_power_like_the_paraxial_one`, evaluated
with the lobe PRESENT (C8 off = committed `HEAD`) and REMOVED (C8 on).
Reference: the module's own lumenairy-free Kirchhoff oracle, FWHM 3.1500 um,
EE2 0.716265, EE4 0.991780; `P_in` 5.654867e-07.

| assertion | bar | lobe PRESENT | lobe REMOVED |
|---|---|---|---|
| `\|FWHM_ex/FWHM_orc - 1\| < 0.15` | 0.15 | 0.000000 | 0.000000 |
| `EE2_ex > 0.90 EE2_orc` | 0.90 | 0.982020 | 0.977437 |
| `EE4_ex > 0.97 EE4_orc` | 0.97 | 0.995392 | 0.994050 |
| `FWHM_px > 1.70 FWHM_orc` | 1.70 | 1.857143 | **1.761905** |
| `EE2_px < 0.25 EE2_orc` | 0.25 | 0.147621 | 0.148303 |
| `\|peak_off_x_px\| > 4 um` | 4.0 | 7.200000 | 7.200000 |
| `\|peak_off_x_ex\| < 0.5 um` | 0.5 | 0.000000 | 0.000000 |
| `FWHM_ex < 0.60 FWHM_px` | 0.60 | 0.538462 | 0.567568 |
| `EE2_ex > 5 EE2_px` | 5.0 | 6.652306 | 6.590805 |
| `\|centroid_ex\| < 0.5 um` | 0.5 | 0.000467 | 0.002767 |
| `0.95 < p_ex/p_in < 1.02` | band | **1.000534** | 0.997362 |
| `\|p_ex/p_px - 1\| < 0.02` | 0.02 | 0.000200 | 0.000043 |

**All twelve pass in both states.** The defect's influence on each asserted
quantity:

| quantity | lobe PRESENT | lobe REMOVED | delta |
|---|---|---|---|
| exact FWHM | 3.150000 um | 3.150000 um | +0.0000 % |
| exact EE2 | 0.70338617 | 0.70010369 | **-0.328 pts** |
| exact EE4 | 0.98721009 | 0.98587878 | -0.133 pts |
| exact EE6 | 0.99888421 | 0.99863578 | -0.025 pts |
| paraxial EE2 | 0.10573569 | 0.10622431 | +0.049 pts |
| paraxial FWHM | 5.850000 um | 5.550000 um | -0.300 um |
| **chain-exit power, exact leg** | 5.6578841495e-07 | 5.6399477640e-07 | **-0.317 %** |
| **chain-exit power, paraxial leg** | 5.6590178669e-07 | 5.6401886999e-07 | **-0.333 %** |

Both tests were also run as tests, on the tree as it stands (C8 on):

```
python -m pytest tests/unit/test_niche_d6_exact_tilted_leg.py::\
    test_the_tilted_exact_leg_conserves_power_like_the_paraxial_one \
  tests/unit/test_niche_d6_exact_tilted_leg.py::\
    test_exact_beats_paraxial_for_a_tilted_congruence_against_the_oracle -q
-> 2 passed, 8 warnings in 39.47s      (no halo warning; the fold-caustic
                                        warning still fires, as before)
```

**Answer to the brief's question (2): no.** Nothing either test asserts is
invalidated. The tightest bar (`EE2_ex > 0.90 EE2_orc`) has 8 points of margin
and moves 0.33; the FWHM assertions do not move at all on the exact leg.

**But the blindness is real and worth naming**, because it is the same
blindness the energy audit found on design 121:

* With the lobe present, **`p_exit/p_in` = 1.000534 on the exact leg and
  1.000734 on the paraxial one - ABOVE UNITY.** The chain reports more power at
  the exit than was launched. `g_halo` = 6.4e-04 accounts for most of it.
* The test that exists to police exactly this
  (`test_the_tilted_exact_leg_conserves_power_like_the_paraxial_one`) asserts
  `0.95 < p_ex/p_in < 1.02`. **A band 38x wider than the violation.** Its own
  docstring quotes "0.9959 exact vs 0.9894 paraxial" - numbers from before the
  lobe existed, and the current 1.0005 / 1.0007 have drifted past unity without
  the test noticing.
* Removing the lobe restores `p_exit/p_in` to **0.997362**, i.e. back BELOW
  unity and onto the documented `ray_subsample=4` discretisation deficit. That
  is the C1b "deficit floor" signature of the energy audit, on a second,
  entirely different fixture. **The pre-C8 number was not merely inflated - the
  manufactured energy was masking the legitimate deficit.**

### 7.1 Two observations for the C8 owner (not verdicts - C8 is in flight)

1. **C8 narrows a fail-before discriminator.** Removing the lobe takes the
   paraxial leg's FWHM from 5.85 to 5.55 um, and
   `assert m_px['fwhm'] > 1.70 * orc['fwhm']` from **1.857x to 1.762x**. Still
   passing, with 3.6 % of margin instead of 9.2 %. That assertion's docstring
   already records one migration (3.19x -> 1.857x at niche C3); this is a
   second, and the bar may want re-pricing rather than re-discovering at the
   next change.
2. **C8 removes 4.9x more power than the reported halo.** `g_halo` on the
   scored call is 6.45e-04 while the chain-exit power drops 3.17e-03. Some of
   that is the exact leg's own (declined, unreported) lobe at `g` = 1.08e-03,
   and some is the feather/hull boundary trimming ray-density spill that sat
   *inside* `1.25 x r_hull`. The direction is right - the post-C8 number lands
   on the expected discretisation deficit (S7) - but the difference between
   "removes the lobe" and "removes 0.32 % of the power" is worth one measured
   line in C8's own record.

---

## 8. What I could NOT determine

1. **Whether the tilted order's absolute `P/Pin` is exactly 1 or 1+3e-05.** At
   `NL` = 401 order (-4,-2) reads 100.003218 % and is still falling
   monotonically (100.0120 -> 100.0061 -> 100.0045 -> 100.0032 at `NL`
   161/241/321/401). The trend extrapolates to 1.0000 but the sequence is not
   converged; `NL` = 401 is 160801 rays and the next doubling is ~4x the cost.
   The on-axis order IS converged (100.000159 % at `NL` 401, 1.6e-06).
2. **The absolute scale of the CHAIN**, as opposed to the oracle. The fix makes
   `oracle_spot` and `rs_spot` physical; `pop_ours_121.py`'s `chain` and
   `chain6` arms report `P_window/P_doe`, a grid-power ratio that never had a
   prefactor problem but also is not the same quantity. A three-way absolute
   comparison POP / chain / oracle was not run (each order is ~5 min of chain).
3. **Whether the `P_sq` convention should be retired.** `SUM |E|^2 dA` is what
   every scorer in the campaign uses (`focus_scan_121.metrics`,
   `hybrid_localize.rs_spot`), and at design 121's NA it reads 0.6 % high as an
   absolute power. It is harmless inside EE ratios (numerator and denominator
   share it) and wrong as a transmission. I changed no scorer.
4. **The exact-leg retrace's true halo.** Its bound (2.0241 mm) is outside its
   own grid reach (1.2375 mm), so the library declines and so does this study.
   Its `g` = 1.076e-03 over the same annulus definition suggests it is worse
   than the paraxial call, not better, but that number is measured in the
   corners-only regime the C7 record already showed to be unreliable in both
   directions. **Extending the halo statement past the grid needs a reference
   other than the grid, and that was not attempted here either.**
5. **Whether the C7 record's label swap affects anything else it concluded.**
   I checked S5.2, S5.2.1, S7.10 and S8.9/S8.10 and the swap is confined to
   which call is called what; the calibration table, the 123x separation and
   the decline condition are all keyed on measured `r_hull` / reach values that
   are correct. I did not re-run the 180-call calibration.
6. **Where the D6 lobe lands after propagation.** All figures are at the
   element's own exit plane, as in the C7 record. The image-plane consequence is
   bounded here only through the readout metrics in S7 (which move by <= 0.33
   EE2 points), not by tracking the lobe.
7. **Whether any OTHER fixture in the niche suite carries the same crescent.**
   The decentre sweep says the trigger is ~1 full beam radius of decentre inside
   a much larger stop; I did not sweep the P2 battery or the C6 synthetic
   fixtures for that geometry.

---

## 9. Reproduction

All commands from `validation/repro_traced_carrier_121/`.

```bash
# --- PART A ---------------------------------------------------------------
# S2: the analytic control (null floor, prefactor, NA / NL / window / back)
python oracleE_rs_control.py                      #  -> _oracleE_control.txt

# S1.2: the null + "the prefactor is a pure scale" check
PART=null NL=81 python oracleE_absolute_121.py

# S3.1/S3.2: absolute P/Pin, window + launch-density + back sweeps
PART=abs python oracleE_absolute_121.py           #  -> _oracleE_abs.txt

# S4: the energy audit's own conservation-reference command, re-run
PART=s15 python oracleE_absolute_121.py           #  -> _oracleE_s15.txt

# --- PART B ---------------------------------------------------------------
# S6.1: which call warns, on which leg  (BOUND=0 restores committed HEAD)
PART=A python d6halo_controls.py
# S6.4: the decentre sweep
PART=B python d6halo_controls.py                  #  -> _d6halo_ctlB.txt
# S6.3: the diffraction ceiling
PART=C python d6halo_controls.py                  #  -> _d6halo_ctlC.txt
# S7:  every assertion of the two dependent tests, lobe present vs removed
PART=D python d6halo_controls.py                  #  -> _d6halo_ctlD.txt

# S6.2 + S6.4: localise the lobe and replay the call under alternative knobs
BOUND=0 LEG=paraxial PART=replay python d6halo_probe.py
BOUND=0 LEG=exact    PART=replay python d6halo_probe.py
```

### Files

**Added** - `validation/repro_traced_carrier_121/oracleE_rs_control.py`,
`oracleE_absolute_121.py`, `d6halo_probe.py`, `d6halo_controls.py`, and this
document.

**Modified** - `exact_ray_oracle_121.py` (RS prefactor + absolute power
bookkeeping + `ray_step_w_p999`), `hybrid_localize_121.py` (the same in
`rs_spot`), `pop_ours_121.py` (the double `h^2` removed). Every existing EE /
FWHM / live output of all three is unchanged (S1.2).

**Not touched** - anything under `lumenairy/`, `CHANGELOG.md`,
`lumenairy/elements/pmm/**`, and every test file.

Raw logs: `_oracleE_control.txt`, `_oracleE_abs.txt`, `_oracleE_s15.txt`,
`_oracleE_conv.txt`, `_d6halo_capture.txt`, `_d6halo_parax.txt`,
`_d6halo_head_parax.txt`, `_d6halo_ctlB.txt`, `_d6halo_ctlC.txt`,
`_d6halo_ctlD.txt`, and the captured call `_d6halo_hot.npz`.

---

## 10. Artefacts found and killed in my own instruments

Continuing the campaign's catalogue (the C7 record ended at 10).

11. **The first RS control was designed with a fixed micron readout window and
    a fixed `f`, and at NA 0.010 it read `P_flux/P_in` = 0.798.** Not a bug in
    the integral - the window held one seventh of the focal spot. Fixed by
    auto-sizing the readout to each case's own `w_f`. **A convergence sweep
    whose observable is windowed must scale the window with the observable.**
12. **The same control's low-NA row then read 0.9983 and was nearly reported as
    the machinery's accuracy floor.** It is physics: at `f` = 3 mm a 30 um beam
    has Fresnel number 0.23 and `back` = 0.5 mm sits inside the focal region, so
    the geometric exit field is not the true field there. Caught by noticing
    that `P_sq/P_flux` (1.00177) did not match `1/<cos>` over the LAUNCH cone
    (1.000025) - the image plane's angular spectrum had been broadened by
    diffraction from a 5 um geometric spot. **The row is kept in S2.1 as a
    control on the control**, because it is the same failure mode as a `back`
    chosen too small on a real design.
13. **`amplitude_model='screen'` was adopted as "the pullback amplitude probe"
    and it is not one.** It returned exactly zero in the annulus, which reads as
    "hypothesis (A) refuted, the lobe is an artefact" - the opposite of the
    truth. `ray_density` FORCES `newton_amp_mask_rel = 0` while `screen` keeps
    1e-4, so the two solve the Newton inverse on different domains. Caught by
    refusing to accept an exact `0.000e+00` (an exactly-zero reading where a
    2.7e-06 incident field exists is a mask, not a measurement) and by dumping
    the captured call's kwargs instead of reasoning about defaults. Recorded in
    S6.4 as a negative result because the next person will reach for the same
    probe.
14. **The D6 probe initially patched `lumenairy.elements._lens_traced.trace`,
    which does not exist** - `_lens_traced` does `from ..raytrace import trace`
    INSIDE the function, so the binding is `lumenairy.raytrace.trace`. It failed
    loudly with `AttributeError` rather than silently capturing nothing, which
    is the right failure but only by luck (cf. the C7 record's artefact 4).
15. **The first D6 capture run reported `amax_halo` = 0.000e+00 and "NO CALL
    WARNED", and was nearly filed as "the C7 record's finding does not
    reproduce".** The working tree had moved: another agent's uncommitted niche
    C8 was already in `_lens_traced.py`. **Check `git status` and the file hash
    before concluding that a recorded measurement does not reproduce** - this
    campaign's own convention (every runner prints the sha256 of the library it
    imported) exists for exactly this and I had not followed it.
16. **The label swap in S6.1 was nearly repeated rather than caught.** The first
    draft of this document copied the C7 record's "exact-tilted-leg retrace"
    wording, because the reproduced numbers (6.405e-01 / 2.0202 mm / 1.6161 mm)
    matched to four digits and matching numbers feel like a matching claim.
    They were the right numbers attached to the wrong call. Settled by running
    BOTH legs and printing each call's grid, `n_fine` and reach.
17. **`g_halo` was reported as 6.447e-04 against the record's 6.449e-04 and the
    2e-07 gap was briefly chased as a library difference.** It is the
    denominator: the library divides by the aperture-transmitted power, this
    probe by the whole input grid. Stated in S5.1 rather than silently rounded.
