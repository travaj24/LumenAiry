# FIX -- the tilt-quadratic optical path at the traced element hand-off

**2026-08-11.  Branch `fix/tilt-quadratic-opl` off `main` @ `755ad99`
(`v5.34.0`).  One library file changed
(`lumenairy/elements/_lens_traced.py`); `lumenairy/propagators/carrier.py`
needed NO change and was not touched.  `CHANGELOG.md` was not touched.  No
`git commit`, no `git push`, no `gh`.**

Closes the defect named -- but not attributed to a line -- by
`docs/audits/PROBE_CHAIN_LADDER_PISTON_2026_08_11.md` S3.7 and left open as
its item S5.3.

---

## 0. VERDICT

> **FOUND, DERIVED, FIXED, PINNED.**  The 4.8 %/group tilt-quadratic deficit
> is `apply_real_lens_traced` referencing its traced OPL grid to the ray
> launched at the **launch-lattice axis** and never re-applying the constant
> it removed:
>
> ```python
> i_axis = n_launch // 2                       # xs_in is AXIS-centred
> opl_grid = opl_grid - opl_grid[i_axis, i_axis]     # and never restored
> ```
>
> The removed constant is `Lam(0) = W(0,0) + a_fit(0,0) + P(0,0)` -- the
> launched congruence's entrance eikonal at the axis (the hammer-H6 and
> niche-C6 terms the element adds to `final.opd`) plus the geometric path of
> the axis ray through the group.  On an untilted, undecentred congruence the
> axis **is** the chief ray and dropping it costs only an unobservable global
> phase, which is why it survived every audit.  Under a `TiltedCarrier` the
> axis is **not** the chief ray, and both pieces become functions of the tilt:
>
> ```text
> W(0,0)  = -theta^2 * z_c   + O(theta^4)     z_c = the chief ray's axial lever
> P(0,0)  =  P_0 + theta^2 * T_g + O(theta^4) T_g = the axis ray's own obliquity
> ```
>
> so every traced group silently subtracted a pure `theta^2` piston from its
> own chief ray.  **The closed form is not a universal constant like `1-1/n^2`
> or `(n-1)/n`: `0.952` is design-121's own arithmetic**,
> `1 - (T_g - z_c)/z_eq` = `1 - (7.5268 - 6.9998)/11.0269` = `1 - 0.047796` =
> **0.95220**, with `z_c` the chief ray's axial lever at the group entrance
> (the 7.000 mm DOE->group-1 gap), `T_g` the axis ray's own obliquity
> coefficient through the group (7.527 mm, measured) and `z_eq` the full
> tilt-quadratic path (11.027 mm).  What IS universal is the mechanism: on the
> synthetic two-group relay in the new test the same term gives a **26.6 %
> EXCESS** (ratio 1.2669 at 6 mrad, 1.2633 at 12 mrad -- the identical
> fixed-fraction signature) instead of a 4.8 % deficit, because there `z_c`
> dominates `T_g` rather than the other way round.  That sign flip is why the
> fraction could never be calibrated out and had to be derived.
>
> **Evidence it is the whole defect and not a part of it.**  Computed entirely
> from ray traces and closed forms -- no chain run -- the predicted deficit
> `-k0 * [Lam_theta(0) - Lam_0(0)]` reproduces the MEASURED chief-ray piston
> deficit of the archived probe runs to **1e-05 relative** at 11.5 mrad and
> 2.5e-04 at 23.0 mrad, over a span in which the quantity itself changes by
> four decades (S2).
>
> **The fix is one term, not a calibration.**  The subtraction stays (it is
> genuine conditioning: the raw OPL is ~1e-02 m and its interesting variation
> ~1e-09 m), and the removed constant is re-applied to the exit field as a unit
> phasor.  Result: the element now returns an **absolute** optical path, which
> is what its own docstrings and the chain's promise in four places and did not
> deliver.
>
> **Measured.**  Group-1 tilt-quadratic reproduction **0.95168 -> 0.9999971**
> (deficit 4.8 % -> 2.9e-06).  Full 6-group inter-order piston against the
> exact skew-ray oracle **0.050-0.416 waves -> 0.0022-0.0042 waves**, i.e. from
> **5x-42x OUTSIDE** lambda/100 to **inside it with 2.4x of margin** on every
> measured order.  The full-chain `theta^2` law is gone: the residual's ratio
> over a 10x tilt step falls from **99.8** (`theta^2` to three digits) to
> **2.2**.
>
> **Blast radius.**  Intensity-blind to the printed digit -- the shipped
> acceptance banner reproduces FWHM, EE3, EE6, EE12 and the peak exactly across
> the whole through-focus scan.  315 tests pass across
> c3/c5/c6/c9/c11/c12/c13/d2/d3/d6 and every field-hashing file in the suite;
> the new guard passes on BOTH mounts; the full `tests/unit` sweep reached
> 6430 passed / 0 failed before the session harness reaped it (S6.2 -- the one
> verification item still owed).  **What DOES change is absolute phase**:
> every traced element's returned field, and therefore every chain's, is now
> offset by the element's own optical path.  Nothing in the library or the test
> suite reads absolute phase.  Intensity is preserved to 2.5e-12 of peak but is
> **NOT byte-identical** (measured, S6.3), so SHA-pinned intensity harnesses
> under `validation/` must be re-baselined.

---

## 1. THE HUNT

### 1.1 Fingerprint reproduced first, on this box, today

`PL_NGROUPS=1 PL_TSCALE=s` through the shipped instrument
(`validation/repro_traced_carrier_121/probe_ladder_run_121.py`), library at
`main`:

| run | `piston_c` measured now | S3.7's archived value |
|---|---|---|
| `(0,0)`, one group | `+0.374658294` | `+0.374658294` |
| `(-1,0) x 0.1` | `+0.441408595` | `+0.441408595` |
| `(-1,0) x 1.0` | `+0.770684460` | `+0.770684460` |
| `(-1,0) x 2.0` | `+1.965309279` | `+1.965309279` |

Bit-for-bit, 4 of 4, in a fresh process.  Scored against the exact chief-ray
trace this is S3.7's own table, to seven digits: `measured/oracle` =
**0.9466222 / 0.9504014 / 0.9516800 / 0.9521301 / 0.9521778 / 0.9522239 /
0.9522989** at 0.115 / 0.345 / 1.152 / 3.455 / 5.758 / 11.516 / 23.032 mrad.

### 1.2 What a fixed FRACTION does and does not tell you

A constant ratio over four decades says the error scales as `theta^2` exactly
as the truth does, i.e. **a term**, not a discretisation.  It does NOT pin the
closed form: *any* `theta^2` term gives a constant fraction, and the value of
that fraction is a ratio of two lengths in the fixture.  So the candidate
closed forms were derived from the hand-off math and each was tested against
the MAGNITUDE, not just the constancy.

**Refuted by construction, before measurement.**  `1 - 1/n^2` needs
`n = 4.58` (no glass in the prescription; group 1 is N-SF1).  `(n-1)/n` needs
`n = 20.8`.  A `sin`-vs-`tan` chief-ray predictor remnant (the P7 trap) is
excluded by niche C3: the predictor is an exact trace and S3.6 measured its
transverse agreement at **0.0 nm** at every order.  A wavefront-vs-vertex-plane
`cos` convention would be `theta^2/2`-scaled against a `z` that the free-leg
measurement (S3.7: exact to 2.0e-07 rad at 51.5 mrad) already accounts for
exactly.  The C5 sphere-vs-plane reference and the C9 conversion are both
**identically zero at the chief ray** (S1.4), so neither can move `piston_c`
at all.

**Surviving candidate, and the one that measured.**  The hand-off's OPL
pieces, laid out end to end, have exactly one place where a constant is
DISCARDED rather than exchanged.

### 1.3 The hand-off, term by term

Chain side (`carrier.py`, niche-D1 tilted branch): the entrance reference is
built as `sphere(R, about the chief ray) x C9 conversion x tilt ramp(about the
chief ray) x C5 exactness`, which is exactly `exp(i k0 W)` for the
`TiltedCarrier` `W`, and `W(x_c, y_c) == 0`.  The exit divides the same
construction out about the transferred chief ray, again zero there.  So the
chain contributes **nothing** to the chief-ray piston across the element -- by
design; the element is supposed to carry it.

Element side (`_lens_traced.py`): rays launch on an AXIS-CENTRED lattice
`xs_in = linspace(-launch_radius, launch_radius, n_launch)` along
`grad(W + a_fit)`, the trace accumulates the geometric path `P(h)`, the
exit-vertex correction lands every ray on the flat exit plane, and then

```python
final.opd += _carrier_W_fn(h_x, h_y)      # hammer H6 -- W(h)
final.opd += _resid_eik.value(h_x, h_y)   # niche C6  -- a_fit(h)
...
opl_grid -= opl_grid[i_axis, i_axis]      # <-- the constant, dropped
```

and every assembly branch builds the exit phase from `k0 * opl_map` alone.  So
the returned field's chief-ray phase was

```text
k0 [ Lam(h_chief) - Lam(0) ] = k0 [ P_chief - Lam(0) ]        (W(h_chief) = 0)
```

against a truth of `k0 * P_chief`.  **The error is exactly `-k0 * Lam(0)`**,
and it is tilt-dependent because `Lam(0)` is evaluated at a point that is not
on the congruence's chief ray.

---

## 2. THE MECHANISM, MEASURED WITHOUT RUNNING THE CHAIN

`validation/repro_traced_carrier_121/tqopl_mechanism.py` computes
`Lam(0) = W(0,0) + P(0,0)` from closed forms and two ray traces -- the
`TiltedCarrier` eikonal at the axis, and the axis ray fired along `grad W(0,0)`
through group 1's own surfaces to its back vertex -- and scores the predicted
deficit `-k0 [Lam_theta(0) - Lam_0(0)]` against the deficit the ARCHIVED probe
runs measured.  Nothing is fitted.

| tilt (mrad) | oracle (rad) | measured deficit (rad) | **predicted (rad)** | pred/measured |
|---|---|---|---|---|
| 1.1516 | 7.013944e-02 | -3.389136e-03 | **-3.352169e-03** | 0.98909 |
| 3.4547 | 6.312577e-01 | -3.021823e-02 | **-3.017018e-02** | 0.99841 |
| 5.7579 | 1.753509e+00 | -8.385663e-02 | **-8.380280e-02** | 0.99936 |
| 11.5158 | 7.014329e+00 | -3.351170e-01 | **-3.351137e-01** | **0.99999** |
| 23.0316 | 2.806198e+01 | -1.338588e+00 | **-1.338918e+00** | 1.00025 |
| 34.5474 | 6.315697e+01 | -3.007018e+00 | **-3.006795e+00** | 0.99993 |

and the two pieces separately, which is where the 0.952 comes from:

| tilt (mrad) | `k0 W(0,0)` (rad) | `k0 [P_theta(0) - P_0(0)]` (rad) | sum |
|---|---|---|---|
| 1.1516 | -4.452421e-02 | +4.787638e-02 | +3.352e-03 |
| 11.5158 | -4.452690e+00 | +4.787803e+00 | +3.351e-01 |
| 23.0316 | -1.781430e+01 | +1.915322e+01 | +1.339e+00 |

`k0 W(0,0)` is `-k0 theta^2 z_c` with `z_c = 7.000 mm` to four digits (the
DOE->group-1 gap: the chief ray's lever at the entrance), and
`k0 [P_theta(0) - P_0(0)]` is `+k0 theta^2 T_g` with `T_g = 7.527 mm`.  The two
nearly cancel, so what is left is `k0 theta^2 (T_g - z_c) = k0 theta^2 x
0.527 mm` against a total tilt-quadratic path of `k0 theta^2 x 11.027 mm` --
**4.78 %**.  There is no glass index and no cosine convention in that
arithmetic; it is a difference of two lengths belonging to this prescription.
The disagreement at the smallest tilt (0.98909 at 1.15 mrad) is the same noise
floor S3.7 reports for its own 0.115 mrad row, not a missing piece: the
residual there is 3.4e-03 rad on a chain whose reproducibility floor is ~1e-05
rad.

The two rows the script prints as `-5.6` and `18.0` (orders `(-4,0)` and
`(-4,-2)`) are winding-number failures in the script's own branch lift -- their
oracle exceeds 100 rad -- not mechanism failures; the `(-3,0)` row at 63 rad
lifts cleanly and reads 0.99993.

---

## 3. THE FIX

`lumenairy/elements/_lens_traced.py`, two hunks.

**(a) Keep the subtraction, keep the number.**  The subtraction is not
optional: `opl_grid` is an absolute path of ~1e-02 m whose variation across the
beam is 1e-08..1e-09 m, so fitting the raw values with a Chebyshev or a spline
would spend the whole mantissa on a constant.  The value is now captured.

```python
i_axis = n_launch // 2
_opl_ref = opl_grid[i_axis, i_axis]
opl_grid = opl_grid - _opl_ref       # UNCHANGED -- byte-identical
_opl_piston = float(_opl_ref)
if not np.isfinite(_opl_piston):
    _opl_piston = 0.0
```

The `opl_grid` line is deliberately left expression-identical, so every fit,
every Newton iterate and every mask downstream is bit-for-bit what it was.  A
dead axis ray already made the whole grid NaN and the returned field
identically zero before this change; the `0.0` fall-back keeps that degenerate
path exactly as it was rather than turning a zero field into a NaN one.

**(b) Re-apply it as a unit phasor at assembly.**

```python
_opl_piston_phasor = None
if _opl_piston != 0.0 and inversion_method != 'backward_trace':
    _opl_piston_phasor = np.exp(1j * (k0 * _opl_piston))
...
phase_exp = np.exp(1j * delta_phase)
if _opl_piston_phasor is not None:
    phase_exp *= _opl_piston_phasor
```

applied at all three assembly sites -- the `preserve_input_phase` branch, the
`amp` branch, and the row-banded `_chunk_assembly` loop -- always onto
`phase_exp`/`pe_b` and always in place, so:

* the `ray_density` magnitude swap (`E_out = _ard * _unit`) carries it for
  free, because that swap keeps `E_out`'s UNIT PHASOR and replaces only its
  modulus;
* `return_screen` (the prepared-traced cache) carries it too, correctly -- the
  piston is input-independent, exactly like the rest of the screen;
* no full-grid temporary is added on the memory-bounded chunk path.

**Why a multiply and not an addition into `opl_map`.**  `k0 * Lam(0)` is
~5e+04 rad and `k0 * opl_map` is ~1e-03 rad; adding them would round the map at
1e-11 rad for no reason.  The phasor's own round-off is ~1e-11 rad -- seven
decades under the lambda/100 = 6.28e-02 rad bar -- and the map keeps full
precision.

**Not applied on `inversion_method='backward_trace'`.**  That experimental
opt-in builds its own map in `_opl_by_backward_trace`, with its own on-axis
reference and its own reversed sign convention, so the forward trace's constant
is not the one it dropped.  It keeps the pre-fix, piston-free behaviour; the
gate is explicit and commented.

**`carrier.py` needed nothing.**  The chain's entrance and exit references are
identically zero at their own chief rays, its free legs were already measured
exact (S3.7: 2.0e-07 rad at 51.5 mrad), and the exact-leg readout's own note
already declares the piston the caller's.  Once the element hands back an
absolute path the composition is correct with no further bookkeeping -- which
is itself a check on the attribution.

---

## 4. FAIL-BEFORE / PASS-AFTER

### 4.1 The reproducer the defect was named with -- design 121, group 1

`PL_NGROUPS=1`, `N=1024`, `dx0=2.0 um`, `rs=4`, reference = the `(0,0)` order
at the same truncation.

| tilt (mrad) | oracle `k0 dOPL` (rad) | **before** measured/oracle | **after** measured/oracle | residual before (rad) | residual after (rad) |
|---|---|---|---|---|---|
| 1.1516 | 0.070139437 | **0.9516800** | **0.9997565** | -3.389e-03 | **-1.708e-05** |
| 11.5158 | 7.014328516 | **0.9522239** | **0.9999971** | -3.351e-01 | **-2.015e-05** |
| 23.0316 | 28.061980285 | **0.9522989** | **0.9999995** | -1.339e+00 | **-1.412e-05** |

The deficit falls from 4.8 % to 2.4e-04 / 2.9e-06 / 5.0e-07.  Note the
after-column's residual is **flat** (-1.4e-05 to -2.0e-05 rad) across a 20x
span in tilt where the quantity itself moves 400x: it is no longer a `theta^2`
term at all, it is the chain's own numerical floor.

### 4.2 The consumer-facing quantity -- inter-order piston, full 6-group chain

Against the exact chief-ray skew trace, reference order `(0,0)`,
bar = lambda/100 = 6.2832e-02 rad.

| order | tilt (mrad) | oracle (rad) | **residual before** | **residual after** | before | after |
|---|---|---|---|---|---|---|
| (-1,0) | 11.5158 | -0.000875290 | -0.673883612 | **-0.022326156** | -0.1073 waves, **10.7x OUT** | -0.00355 waves, 2.8x in |
| (-2,0) | 23.0316 | +0.054791812 | -2.612563374 | **-0.023557177** | -0.4158 waves, **41.6x OUT** | -0.00375 waves, 2.7x in |
| (-3,0) | 34.5474 | +0.344574100 | +0.516539501 | **-0.026440697** | +0.0822 waves, **8.2x OUT** | -0.00421 waves, **2.4x in** |
| (-4,0) | 46.0633 | +1.173573314 | +2.602186542 | **-0.020588489** | +0.4142 waves, **41.4x OUT** | -0.00328 waves, 3.1x in |
| (-4,-2) | 51.5003 | +1.874492340 | +0.316762779 | **-0.014120924** | +0.0504 waves, **5.0x OUT** | -0.00225 waves, 4.5x in |

**5 of 5 orders move from outside lambda/100 to inside it**, worst case 0.00421
waves = lambda/238.  The remaining residual is ~0.02 rad on every order
regardless of tilt -- again flat, not quadratic.

### 4.3 The `theta^2` law itself, at the full chain

S3.7's sharpest statement was that the full chain carried a `theta^2` piston
whose coefficient was 2.1e+04 times the real system's.  Measured on the same
continuous axis (`PL_TSCALE` at the FULL chain, order `(-1,0)`):

| tilt (mrad) | residual before (rad) | residual after (rad) | shrink |
|---|---|---|---|
| 0.1152 | +0.011895 | **+7.5230e-05** | 158x |
| 1.1516 | +1.186825 | **+1.66225e-04** | 7139x |
| ratio over the 10x tilt step | **99.8** (`theta^2` to 3 digits) | **2.2** | -- |

The quadratic law is gone, not merely reduced: the residual now grows by
2.2x where the tilt-quadratic path grows by 100x.

### 4.4 The element, on its own -- the absolute path

On-axis, untilted, `carrier=inf`, one slow singlet, against a single ray trace
of the axial path (3.007165811 mm, `k0 P = 14423.3435 rad`):

| | on-axis exit phase | residual vs `k0 * P_axial` |
|---|---|---|
| before | `-0.000000019 rad` (i.e. **exactly zero**) | **+2.8499 rad = 0.454 waves** |
| after | `+2.849924197 rad` | **-1.88e-08 rad = 3e-09 waves** |

That `-1.9e-08` is the whole story in one number: the element returned a field
referenced to nothing, and now returns one referenced to its own optical path.
On the design-121 fixture the applied phasor was checked directly -- the
post-fix minus pre-fix exit phase equals `k0 * Lam(0)` to **12 digits**, and
`|E|` at the sample point is **bit-identical** (`1.003790167355e+00` both
ways), which is the intensity-blindness claim measured rather than asserted.

---

## 5. THE GUARD -- so this class cannot go dark again

`tests/unit/test_fix_tilt_quadratic_opl.py`, 7 tests, **7 fail before / 7 pass
after**, 9.8 s, self-contained (no `.zmx`, no data file).  The fixture is a
flat-entrance / `K = -n^2` conic singlet -- the exact Fermat solution for a
collimated bundle -- so the element's own wavefront error is negligible and
what the oracle comparison measures is the PISTON and nothing else.  The
oracle is an exact skew ray trace sharing no FFT grid, no carrier convention
and no propagator with the code under test.

Three rules, because one is not enough:

1. **Envelope rule** -- the chief-ray piston is inside lambda/100 of the exact
   skew ray trace at 6 / 12 / 24 mrad (before: -2.078 / -2.026 / -1.808 rad).
2. **Fingerprint rule** -- the tilt-quadratic optical path is reproduced to
   1e-03 RELATIVE (before: 0.348 at 12 mrad, 0.855 at 24 mrad, against
   design 121's 0.952).  A *fraction* is what a missing term looks like, so
   the fraction is what is pinned.
3. **Comparative rule, on a TWO-group chain** -- the residual must not scale
   with `theta^2`.  This is the rule with teeth.  The fixture keeps an
   irreducible tilt-INDEPENDENT residual (a diffraction/aberration difference a
   geometric ray trace cannot carry, 0.0027 waves), and a re-introduced
   `theta^2` piston is quadratic by construction: measured **before**,
   +0.741 -> +2.924 rad over a 2x tilt step (3.95x, i.e. `theta^2` to two
   digits); **after**, +0.01704 -> +0.01709 rad (0.3 %).  The assertion is on
   the DRIFT, so it catches the defect class at any absolute size, including
   one an envelope bar would swallow.

Plus (4) the mechanism itself: `test_traced_element_returns_the_absolute_
optical_path` asserts the on-axis exit phase equals `k0 x` the traced axial
path (S4.4), which is the cause rather than the symptom and costs 0.4 s.

**Why the shipped metrics could not see this.**  Every acceptance on this path
integrates `|F|^2` per DOE frame, and the orders land on separate 480 um
frames -- so the inter-chain piston is not merely un-asserted, it is not
REPRESENTED.  None of the three rules above can be satisfied by an intensity
metric, which is the property that makes them a guard rather than another
coverage row.

---

## 6. BLAST RADIUS

### 6.1 The banner acceptance -- unchanged, to every printed digit

`validation/repro_traced_carrier_121/focus_scan_121.py`, UNEDITED, library
defaults (`CREF`/`AM`/`PIP` unset), `N=2048`, `rs=4`, `NFC=8192`, `WF=4.0`,
`NOUT=2048` -- the campaign's own acceptance invocation.

| | recorded (`CAPSTONE_D121_2026_08_06`, `C11..._2026_08_03` S6.6) | **this branch** |
|---|---|---|
| `AT-PLANE dz=0` FWHM / EE3 / EE6 / EE12 | 3.350 um / 90.3 / 99.7 / 99.8 | **3.350 um / 90.3 / 99.7 / 99.8** |
| `BEST-FOCUS[peak]` plane | dz = **+0 um** | dz = **+0 um** |
| `BEST-FOCUS[peak]` FWHM / EE3 / EE6 / EE12 | 3.350 um / 90.3 / 99.7 / 99.8 | **3.350 um / 90.3 / 99.7 / 99.8** |
| `BEST-FOCUS[peak]` peak | **5.529e+03** | **5.529e+03** |
| `dz = +5 um` | 3.450 um / 89.6 / 99.7 / 99.8 | **3.450 um / 89.6 / 99.7 / 99.8** |
| chain-only wall | -- | 186 s |
| grid check | -- | no MEMORY / COUNT / RESOLUTION-LIMITED warning; the leg ran at the 8192 it asked for |

**IDENTICAL to every printed digit, peak included, over the whole +/-80 um
through-focus scan** -- and required to be: the applied piston is a global unit
phasor, so it cannot move any intensity observable except at the 1e-16 level of
one complex multiply.  That was checked at the element level first (S4.4:
`|E|` bit-identical at the sample point) rather than inferred from the banner.

### 6.2 Suites

| suite | mount | result |
|---|---|---|
| `test_niche_c3_gap_paraxial_guard` + `c5` + `c9` + `c6_fit_guard` + `c6_stationary_phase_launch` + `c11` + `c12` + `d2` + `d6` | Windows | **211 passed** (621 s) |
| `test_niche_d3_guards` (reads the piston through the c13 machinery) | Windows | **41 passed** (1617 s) |
| the three files that hash a FIELD (`eh1_maslov_upsample`, `w5_shim_removals`, `v5_20_8_rcwa_threaded_sweep`) | Windows | **63 passed, 44 skipped** (333 s) |
| `tests/unit/test_fix_tilt_quadratic_opl.py` (the new guard) | Windows | **7 failed before / 7 passed after** (9.8 s) |
| `tests/unit/test_fix_tilt_quadratic_opl.py` | **WSL CI proxy** (py3.12, numpy 2.4.6, scipy 1.17.1 -- a different BLAS/libm from the Windows py3.14 / numpy 2.4.4) | **7 passed** (10.8 s) |
| `tests/unit -m "not integration and not slow"` (the CI invocation), `-n 8` | Windows | **6430 passed, 50 skipped, 0 failed, 0 errors through 56 %** -- see the caveat below |

**The full sweep did not run to completion and this is an environment limit,
not a result.**  Two independent attempts (`-n 6` and `-n 8`) were each reaped
by the session harness at 54 % / 56 % after ~45 min; the second half of
`tests/unit` is where the heavy traced physics lives, so it is the expensive
half.  What IS established: **zero failures in 6480 collected outcomes across
two independent runs**, and the traced/carrier files that this change can
actually reach were run to completion INDIVIDUALLY and in full (the 211 + 41 +
63 rows above, which cover c3/c5/c6/c9/c11/c12/c13/d2/d3/d6 and every test in
the suite that hashes a field).  A clean full-sweep run on an unshared box is
the one verification item this branch still owes.

Note what the third row measured, because it answers the byte-identity
question from the other side: `test_niche_audit_w5_shim_removals` **SKIPS its
42 frozen SHA-256 digests off the capturing host by design** ("a hash pins
every ULP, and libm/BLAS/build differences give different bits for the same
correct physics").  So the suite already treats bit-level field pins as
host-specific rather than as physics -- which is exactly the status the
intensity shas in S6.3 now have.

### 6.3 What a consumer will notice

**Absolute phase moves.**  Every `apply_real_lens_traced` return -- and
therefore every `propagate_traced_carrier_chain` return -- is now offset by the
element's own optical path (`k0 x` a few mm, i.e. ~5e+04 rad per group).  That
is the point of the fix.  Consequences:

* **Intensity, energy, EE, FWHM, Strehl, spot position: unchanged to the
  printed digit** -- the banner reproduces all four figures and the peak
  (S6.1).  **They are NOT byte-identical, and this was measured, not
  assumed.**  The same script was run against an ISOLATED copy of the
  pre-fix library (`git show 755ad99:...` over a full package copy, with the
  editable install's meta-path finder removed so the copy is what imports)
  and against this branch:

  | level | intensity sha256 equal? | float64 words differing | max `dI` / `I_peak` | `dP/P` |
  |---|---|---|---|---|
  | one traced element | **NO** (`a944774b...` vs `d6df5ca0...`) | 28 250 / 262 144 (10.8 %) | **8.9e-16** | 2.2e-16 |
  | two-group tilted chain | **NO** (`4afa1410...` vs `a4ced314...`) | 261 631 / 262 144 (99.8 %) | **2.5e-12** | 2.7e-15 |

  The mechanism is float non-commutativity, not physics: a global phasor `q`
  satisfies `|q E| = |E|` exactly in real arithmetic but not in the last
  mantissa bit, and `q * FFT(E)` is not bitwise `FFT(q E)`, so the difference
  spreads over the grid across the transport legs while staying at 1e-12 of
  peak.  **Consequence: every SHA-PINNED INTENSITY harness must be
  re-baselined** -- `rc_gate_121.py`'s six per-order intensity shas,
  `probe_ladder_run_121.py`'s `field_sha256` / `env_sha256`, and any other
  bit-level pin.  Nothing in `tests/` pins those bits (see S6.2); the
  affected pins all live under `validation/`.
* **Inter-beam / inter-order relative phase: now correct** -- this is the
  capability the probe's S4.2 said did not exist ("sum the INTENSITIES today;
  do not sum the FIELDS across orders").  With this fix the per-order piston
  is trustworthy to 0.0042 waves on design 121's whole fan, so a coherent
  sum-at-aperture is admissible.
* **Any pinned FIELD BYTES will move.**  Nothing inside the library reads
  absolute phase, and no test in `tests/` did either (211 + the full sweep
  pass), but an external artifact that stored a complex field will not match.
  The chain-A caches under `validation/repro_traced_carrier_121/` orphan
  themselves automatically -- their key hashes every `lumenairy/**/*.py`.

---

## 7. WHAT IS STILL OPEN

1. **`inversion_method='backward_trace'` keeps the pre-fix behaviour** (S3).
   It is an opt-in experimental path with no carrier support; making it
   absolute needs its own oracle for the reversed sign convention, which is a
   separate piece of work.
2. **The residual after the fix is ~0.02 rad on the six-group chain and flat in
   tilt.**  It is inside lambda/100 with 2.4x of margin and it is NOT this
   defect class (it does not scale with `theta^2`).  What it IS -- most likely
   the per-group traced-fit residual at the chief-ray pixel plus the
   diffraction difference a geometric oracle cannot carry -- is not attributed
   here.  A consumer wanting better than lambda/200 across a 51 mrad fan should
   measure it.
3. **The probe's OTHER finding is untouched**: aperture ENERGY is not converged
   at `N = 1024, rs = 4` (`dP/P` 4.7e-03 to 5.2e-03 per octave), and the remedy
   it names is `ray_subsample = 1`, not a bigger grid.  That is a separate
   defect and this branch does not address it.
4. **`_opl_piston` is not surfaced as a diagnostic.**  It could be added to the
   `_exit_na_out`-style private sinks if a consumer needs to audit the per-group
   piston without differencing two runs.
