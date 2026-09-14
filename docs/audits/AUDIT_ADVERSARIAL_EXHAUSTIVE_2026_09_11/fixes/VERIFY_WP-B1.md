# VERIFY-WP-B1 — independent adversarial re-verification of WP-B1 (Maslov S6: the saddle follows the input field's local wavevector)

Branch `audit-fixes-2026-09`.  Subject: **2871e92e** (WP-B1) and its follow-up
**8dab7de5** (`input_wavevector_saddle=` as a per-call keyword).  Pre-change
baseline for every byte-identity and fail-before probe: **2871e92e^ =
47bc7a79**.

I did not write WP-B1.  Every number below was measured here, on a fixture
the engineer did not use, against an oracle written for this verification.

**Headline.**  The fix is real and it is right: on an independent optic the
corrected saddle takes the two asymptotic evaluators from *no answer at all*
to their own collimated accuracy floor on a tilted input, and from 0.27 / 0.74
to 0.983 / 0.996 field fidelity on an off-axis converging one.  Both stated
deviations from the WP-A4 design are correct, and I have the measurement that
proves the second one (the double count) is a real 4.2× phase error rather
than a judgement call.  The byte-identity claim reproduces on 27 rows of my
own, archive-to-archive, and so do all four claims of the 8dab7de5 follow-up.

**But the fallback criterion measures the wrong thing, in two ways.**  The S6
term is the product `k1 . ds1/dv2` and the criterion scores only `k1`; a
uniform tilt fits `k1` to 1.8e-14 while the chart misplaces the entrance
coordinates it is contracted against, and the shipped defaults then return
field fidelity **0.078 against a floor of 0.900**, centroid error **−27.7 µm**,
**with no warning of any kind**.  I fixed that half here (§4.1).  The other
half — `_K1_FIT_RESIDUAL_MAX = 0.5` — is loose by one to two decades, and the
ladder in §3 of the WP-B1 report that derives it does not reproduce on its own
fixture; that is a shipped-behaviour decision I have left to the owner with
the full measurement (§4.2).

---

## 1. My oracle, and what it is worth

Written for this verification, importing nothing from lumenairy
(`scratchpad/verify_b1/oracle_b.py`): closed-form conic intersection (quadric
root, not Newton), vector Snell, ray-tube cross-section from the launch
parametrisation for the amplitude transport, and a Kirchhoff surface sum from
the exit surface with the two-sided obliquity `(cos_out + cos_in)/2`.  Every
ray's launch direction is the input field's own local wavevector — either the
EXACT analytic gradient of the input phase (`analytic=True`, the default arm)
or the same conjugate-product forward difference the library estimates it with
(`analytic=False`), so the two arms isolate the estimator's own bias.

**Fixture B** — deliberately unlike WP-B1's f = 6 mm N-BK7 / 1.0 µm / 3.2 µm
chart in every parameter: **N-SF11 biconvex, R = +15 / −30 mm, t = 1.0 mm,
f = 13.3 mm, clear aperture 1.40 mm (NA 0.0525), λ = 1.55 µm, N = 384,
dx = 5.0 µm**, Gaussian input of 0.35 mm 1/e amplitude radius, readout
14.0 mm past the exit vertex = **0.80 mm past best focus** (geometric rms
14.8 µm, so the saddle is non-degenerate), ROI half-width 100–150 µm at the
native pitch.

| oracle self-check | measured |
|---|---|
| grid convergence, `1 − fid` between successive entrance samplings (n_pupil 81 → 121 → 161 → 221 → 301) | 4.1e-07 → 1.7e-07 → 3.3e-08 → **2.7e-08** (the floor every bar below sits above) |
| oracle vs the library's own exact pointwise `'quadrature'`, collimated | `1 − fid` = **1.24e-04** |
| same, tilt 0.5× / 1.0× the lens NA | 1.23e-04 / 1.14e-03 |
| same, off-axis converging / astigmatic | 1.19e-04 / 5.98e-05 |
| analytic launch direction vs the library's forward-difference estimator | `1 − fid` = 0 (tilts), **8.1e-10** (converging), **4.7e-09** (astigmatic) |

That last row is a NEGATIVE result worth recording: the forward difference in
`_local_direction_cosines` reports the wavefront slope half a pixel downstream,
and I expected a `dx/(2 f_in)` tilt bias on a curved input (0.75 µm of spot
shift at f_in = 40 mm on this optic).  It measures **below the oracle's own
floor** on both curved fixtures.  Not a defect.

---

## 2. Verdict table

| # | WP-B1 / follow-up claim | verdict | my oracle and my numbers |
|---|---|---|---|
| 1 | S6 is fixed: the saddle follows the input's local wavevector | **VERIFIED-WITH-NOTES** | §3.1 — the corrected saddle reaches the method's own collimated floor on every input class I tried, up to tilt 1.5× the lens NA.  Beyond that it collapses and nothing warns — defect **V1**, fixed here |
| 2 | Deviation (a): adding the input phase to `opd_star` / `opd_v` would double-count, because both integrators sample the complex `E_in` at the saddle | **VERIFIED** | §3.2 — the shipped form reproduces the oracle's PHASE at the method's collimated floor (0.0641–0.0667 waves RMS against a 0.0664 floor for `stationary_phase`); the design's literal extra raises it to **0.2801–0.2898 waves, 4.2–4.5× the floor**, and drops fidelity 0.911 → 0.074 |
| 3 | Deviation (b): `k1` gets its own `_solve_fit` so `'quadrature'` and `'levin'` stay bit-identical | **VERIFIED** | §3.3 — archive-to-archive `np.array_equal` on a tilted AND an off-axis converging input, both integrators |
| 4 | The collimated case is byte-identical (24 rows) | **VERIFIED** | §3.3 — my own 27 rows, archive-to-archive: 23 identical, 4 correctly differ, including `collimated_input=True` on non-collimated inputs and float phase dirt |
| 5 | `_SADDLE_FLAT_INPUT_NA = 1e-3` is an absolute bar and the sub-bar error scales with `f` (report §6 item 2) | **VERIFIED** (the admission is exact) | §3.4 — just under the bar: centroid error **−4.74 / −35.38 / −354.03 µm** on f = 13.3 mm / 100 mm / 1 m, i.e. 0.26 / 2.0 / **19.7 diffraction-spot radii**, silent.  Defect **V3** |
| 6 | `_K1_FIT_RESIDUAL_MAX = 0.5` sits "a factor 1.8 below the last case where engaging still helps" | **NOT VERIFIED** | §4.2 — the speckle rows of the report's §3 ladder do not reproduce on the report's own fixture (measured 20× lower, while the hard-edge / tilt / converging rows reproduce exactly).  Re-measured against the exact `'quadrature'`, the answer is gone **two decades below** the bar.  Defect **V2**, left to the owner |
| 7 | The aliasing blind spot is real but harmless — "the whole sampled field is the aliased one and every other consumer sees the same thing" | **VERIFIED** | §3.5 — at 1.2× Nyquist the residual is 4.3e-14, nothing warns, and `'quadrature'` lands on the ALIASED spot too (+2.9 µm against the aliased oracle) |
| 8 | The 25 (now 28) pins are derived envelopes and bite | **VERIFIED-WITH-NOTES** | §3.6 — zero-`k1` 15/28 red, flipped sign 8/28, dropped Hessian 4/28.  But the `opd_star` double count is caught by **1 of 28** and the `opd_v` one by **0 of 28** — defect **V4**, closed here with two new pins |
| 9 | Follow-up: the keyword equals the seam on 18 cells and is byte-identical to 2871e92e | **VERIFIED** | §3.7 — 18/18 seam == keyword in one process, 18/18 archive-to-archive against 2871e92e |
| 10 | Follow-up: the keyword reaches both integrators and the `fold_split` leg | **VERIFIED** | §3.7 — `_leg_kw` carries it and a functional `fold_split=True` run differs between `False` and `True` (relL2 0.819 / 0.533) with the warning only on `False` |
| 11 | Follow-up: the warning names the saddle the caller chose | **VERIFIED** | §3.7 — `(input_wavevector_saddle=False)` vs `(_S6_INPUT_WAVEVECTOR_SADDLE = False)` in the text |
| 12 | "No other module needed to change: `propagators/asymptotic*.py` is a different family with its own `_compute_M_b` saddle, which S6 does not name" (report §5) | **VERIFIED** | `asymptotic_maslov.py` builds a Gaussian-regularised COMPLEX saddle `M = Re M − i pi H_phi` from a canonical polynomial fit and never sees `E_in` at all (one occurrence of the word, in a warning comment): the field is decomposed into modes upstream, so each mode carries its own launch direction into the fit.  S6 cannot apply.  I left `lumenairy/propagators/asymptotic*.py` untouched |
| 13 | "`_lens_jax.apply_real_lens_maslov_jax` carries the same OPD-only saddle" (report §6 item 6, §7 item 5) | **WRONG** | §7 — that file contains zero occurrences of `saddle`, `stationary`, `grad_v2` or `v2`; it is a thin-OPD phase screen, as its own docstring says.  Measured separately and reported for WP-B7 |

---

## 3. The measurements

### 3.1 Headline, re-derived on fixture B (inputs the report did not use)

Windows are centred on the INPUT's own geometric landing, snapped to the
output pixel grid, so every arm reads the same pixels.  "before" is the
OPD-only saddle held on the same build
(`input_wavevector_saddle=False`); "after" is the shipped default.

| input (landing) | method | before | after | oracle |
|---|---|---|---|---|
| **tilt 0.5× lens NA** (377.7 µm) | `quadrature` | fid 0.99988, Δcx +0.0015 µm | — | — |
| | `stationary_phase` | **fid 0.0000** — the window is EMPTY, the spot never left the axis; warns | **0.90287**, Δcx **−0.015 µm**, silent | floor 0.9001 |
| | `local_quadrature` | **0.0000**; warns | **0.98059**, Δcx +0.061 µm | floor 0.9796 |
| **tilt 1.0× lens NA** (756.0 µm) | `quadrature` | 0.99886, Δcx +0.050 µm | — | — |
| | `stationary_phase` | 0.0000; warns | **0.91113**, Δcx +0.157 µm | |
| | `local_quadrature` | 0.0000; warns | **0.98309**, Δcx +0.239 µm | |
| **tilt 2.0× lens NA** (1518.0 µm) | `quadrature` | 0.92297, Δcx +5.60 µm | — | — |
| | `stationary_phase` | 0.0000; warns | **0.07776**, Δcx **−27.66 µm**, SILENT | ← defect V1 |
| | `local_quadrature` | 0.0000; warns | **0.09754**, Δcx −26.58 µm, SILENT | |
| **off-axis converging** `f_in = +150 mm, x0 = +0.25 mm` (24.0 µm) | `quadrature` | 0.99988, EE(10 µm) 0.0462 | — | EE 0.0465 |
| | `stationary_phase` | 0.27165, Δcx −23.97 µm, EE 0.0364; warns | **0.98288**, Δcx −0.246 µm, EE 0.0526 | |
| | `local_quadrature` | 0.74460, Δcx −22.78 µm, EE 0.0826; warns | **0.99580**, Δcx −0.098 µm, EE 0.0425 | |
| **astigmatic** `f_x = +50 mm, f_y = −80 mm` (0 µm) | `quadrature` | 0.99994, EE 0.0277 | — | EE 0.0277 |
| | `stationary_phase` | 0.12417, EE 0.3082; warns | **0.98487**, EE 0.0310 | |
| | `local_quadrature` | 0.43445, EE 0.1204; warns | **0.99643**, EE 0.0259 | |

The corrected tilted fidelities equal this fixture's own COLLIMATED floor
(0.9001 `stationary_phase`, 0.9796 `local_quadrature`) to 3e-3 — the same
envelope the report claims on its own chart, reproduced on a different optic,
glass, wavelength and pitch.  For the two curved inputs the corrected answer
is BETTER than the collimated floor, because those inputs are not read at the
same defocus.

### 3.2 Deviation (a): the phase, not just the intensity

Fidelity is already phase-sensitive, but a double count is a pure phase error,
so I scored it directly: the intensity-weighted RMS of `arg(E conj(E_ref))`
after removing the best global phase, in waves, against the exact pointwise
`'quadrature'` on the same chart.  On **WP-B1's own fixture**:

| input | `stationary_phase` | `local_quadrature` |
|---|---|---|
| collimated (the method's floor) | 0.0664 waves | 0.0271 waves |
| tilt 0.5× lens NA | **0.0667** | **0.0271** |
| tilt 1.0× lens NA | **0.0641** | **0.0251** |
| the WP-A4 design's extra term on `opd_star`, at the two tilts | **0.2801 / 0.2898** | — |

(The `local_quadrature` column of the last row is blank because that mutation
lives at the `opd_star` site only; the analogous `opd_v` mutation was built
separately and is what makes the `local_quadrature` id of the new pin go red —
§3.6.)

So the shipped form reproduces the oracle's phase at 0.97–1.00 of the method's
own collimated floor, and the design's version is 4.2–4.5× worse.  On
fixture B the same mutation drops `stationary_phase` fidelity from 0.911 /
0.983 / 0.985 (tilt / converging / astigmatic) to **0.074 / 0.365 / 0.226**.
The deviation is correct, and it is correct for the reason the report gives:
`A(v*) e^{i Psi(v*)} = E_in(s1*) |det J|^{1/2} e^{2 pi i OPD*}` is already the
product the code forms.

### 3.3 Byte-identity, archive-to-archive

`git archive 2871e92e^ lumenairy` and `git archive 8dab7de5 lumenairy`
extracted read-only into separate trees; each run in its own child process
with `cwd` and `PYTHONPATH` set to that tree and `lumenairy.__file__`
asserted; never through pytest, never against the shared working tree.
27 rows on fixture B:

```
collimated + stationary_phase / local_quadrature / quadrature ..... equal
collimated + collimated_input=True ................................ equal
collimated as complex64 ........................................... equal (dtype kept)
tilt 1x NA + quadrature ........................................... equal
off-axis converging + quadrature .................................. equal
astigmatic + quadrature ........................................... equal
tilt 1x NA + levin ................................................ equal
off-axis converging + levin ....................................... equal
tilt / converging + collimated_input=True (sp and lq) ............. equal   <- the brief's probe
E * (1 + 1e-17j), sp and lq ....................................... equal   <- the brief's probe
E * exp(1j * 1e-16 * randn), sp ................................... equal
sub-threshold tilt 3.0e-4 (ray NA 9.0e-4), sp and lq .............. equal
tilt / converging / astigmatic + OPD-only saddle asked for ........ equal, warn 1 -> 1
tilt / converging / astigmatic ENGAGED ............................ DIFFER, warn 1 -> 0   (correct)
```

**27 / 27 rows behave as expected** (23 byte-identical, 4 correctly different).
`'quadrature'` and `'levin'` are bit-identical on NON-collimated inputs, which
is exactly what deviation (b) buys.  The float-dirt probe is exact rather than
lucky: `(a+bi)·conj(a+bi)` with `b = 1e-17·a` gives an exactly real product, so
`_local_direction_cosines` returns exactly zero and the fit is never built.

### 3.4 The engagement bar is absolute, and silent below itself

`_SADDLE_FLAT_INPUT_NA = 1e-3` gates on `3 × RMS|k1|`, so a uniform tilt θ
engages at θ > 3.33e-4.  Just below it (θ = 3.3e-4), fixture B scaled to three
focal lengths at FIXED NA and fixed wavelength (so the diffraction spot is the
same 18.0 µm Airy radius in all three):

| system | f·θ | `stationary_phase` default | engaged (`=True`) | relL2 between them | warns |
|---|---|---|---|---|---|
| f = 13.3 mm | 4.4 µm | Δcx **−4.74 µm** | Δcx +0.00 µm | 0.231 | **0** |
| f = 100 mm | 33.0 µm | Δcx **−35.38 µm** | Δcx +0.20 µm | 0.450 | **0** |
| f = 1.0 m | 329.9 µm | Δcx **−354.03 µm** | Δcx −0.91 µm | **1.820** | **0** |

(`local_quadrature`: −2.11 / −8.78 / −313.44 µm, relL2 0.100 / 0.244 / 3.900.)
The centroid error is `f·θ` to within a few per cent at every scale, and at
f = 1 m the un-engaged field has no overlap left with the engaged one.  The
scaled grids under-sample the interference fringes inside the disc, so the
absolute fidelities there are not meaningful; the centroid and the relative L2
are, because both arms are computed on the same grid.  The report's §6 item 2
states this honestly; what it does not say — and what nothing in the library
says at run time — is that **the sub-bar case is silent**.  The remedy now
exists and is public (`input_wavevector_saddle=True`), but the docstring does
not connect it to a sub-bar input.  Follow-up F3.

### 3.5 The aliasing blind spot

λ/(2dx) = 0.1550 rad on fixture B.  At a 1.2× Nyquist tilt (θ = 0.1860, true
landing +2719 µm, aliased landing −1796 µm):

| method | fidelity vs the ALIASED oracle | Δcx | warns | k1 residual |
|---|---|---|---|---|
| `quadrature` | 0.574 | **+2.94 µm** | 0 | — |
| `stationary_phase` | 0.078 | +20.14 µm | 0 | **4.3e-14** |
| `local_quadrature` | 0.070 | +20.19 µm | 0 | 4.3e-14 |

The report's two claims both hold: the fit residual is machine-zero (the
aliased direction is perfectly smooth, so no gate can see it), and the exact
pointwise integrator lands on the aliased spot too — the whole sampled field
really is the aliased one.  The user gets a spot 4.5 mm from the truth, and
only the docstring's "sample finely enough that `max|grad arg E_in| dx < pi`"
says so.  I did not change that: it is a property of the sampled field, not of
this saddle.  (The extra loss in the two asymptotic rows here is defect V1,
not aliasing: `3 × RMS|k1|` is 0.372, so `na_proxy` over-sizes the chart.)

### 3.6 Do the pins bite?  Mutating the fix

Each mutation applied textually to `lenses_maslov.py` in its own read-only
export of 8dab7de5, then the 28 shipped pins run against it:

| mutation | red | notes |
|---|---|---|
| zero the `k1` fit (S6 term absent) | **15 / 28** | includes the CPU/`xp` parity pin, so the `xp = np` twin is genuinely covered |
| flip the sign of `k1` | **8 / 28** | |
| drop the Hessian term, keep the gradient | **4 / 28** | only the CONVERGING fixtures and the analytic-vs-finite-difference Hessian pin; every tilted pin stays green |
| the WP-A4 design's extra term on `opd_star` (`stationary_phase`) | **1 / 28** | only `test_b1_a_tilted_input_returns_to_the_collimated_asymptotic_floor[stationary_phase]` |
| the same at the `opd_v` site (`local_quadrature`) | **0 / 28** | nothing in the file sees it |

The last two rows are defect **V4**: the deviation the report defends at
greatest length had the thinnest coverage in the file, and none at all for one
of the two integrators.  Closed here — §5.

Everything in the file is a derived envelope or a byte-identity, not a
per-build number: the tolerances are the oracle's own floor, the method's
collimated fidelity, and `np.array_equal` against the seam on the same build.
Nothing pins a prior release's numbers.

### 3.7 The 8dab7de5 follow-up

* **18 cells** (collimated / tilt / off-axis converging × `stationary_phase` /
  `local_quadrature` × mode `None` / `False` / `True`): **18/18 seam ==
  keyword** byte-identical in one process (and the same warning count), and
  **18/18 byte-identical archive-to-archive against 2871e92e**, which has only
  the seam.
* **The keyword reaches both integrators and the `fold_split` leg**:
  `input_wavevector_saddle=input_wavevector_saddle` is in `_leg_kw`, and a
  functional `fold_split=True` run (one refractive leg) gives
  `array_equal=False` between `False` and `True` with relL2 **0.819**
  (`stationary_phase`) and **0.533** (`local_quadrature`), warning only on
  `False`.  The E-L20-shaped omission is not present.
* **The warning names the caller's own spelling**:
  `...it was asked for the OPD-only saddle (input_wavevector_saddle=False)...`
  vs `...(_S6_INPUT_WAVEVECTOR_SADDLE = False)...`.
* `tests/unit/test_audit2609_b1_maslov_input_wavevector.py` collects **28**
  and passes 28 on 8dab7de5.

---

## 4. Defects

### 4.1 V1 (P1) — the fallback scores only half the S6 term.  FIXED HERE

The term both saddle solvers add is `k1 . ds1/dv2`.  `_K1_FIT_RESIDUAL_MAX`
scores the `k1` fit alone, and a **uniform tilt fits `k1` perfectly by
construction** (it is a constant) however badly the chart carries the entrance
coordinates it is contracted against.  Meanwhile `na_proxy = na_lens +
na_input` sizes the pupil box from the input's angular spectrum, and for a
uniform tilt θ that 3-sigma-about-zero moment is **3θ** — three times the
actual launch angle — so a large tilt inflates the box the order-4 chart has
to span, and the entrance-coordinate fit degrades.

MEASURED, tilt sweep on fixture B (reference: my Kirchhoff oracle):

| tilt / lens NA | s1 fit rel | k1 fit rel | `stationary_phase` | `local_quadrature` | `quadrature` |
|---|---|---|---|---|---|
| 0.50 | 7.2e-05 | 5.4e-11 | 0.9029 | 0.9806 | 0.9999 |
| 1.00 | 4.5e-04 | 2.1e-10 | 0.9111 | 0.9831 | 0.9991 |
| 1.25 | 9.4e-04 | 8.1e-10 | 0.9169 | 0.9836 | 0.9989 |
| 1.50 | 1.8e-03 | 3.5e-15 | 0.9186 | 0.9694 | 0.9894 |
| **1.75** | **3.3e-03** | 3.1e-15 | **0.0383** | **0.0609** | 0.9867 |
| 1.90 | 4.7e-03 | 2.0e-14 | 0.0643 | 0.0783 | 0.9269 |
| 2.00 | 5.8e-03 | 1.8e-14 | 0.0778 | 0.0975 | 0.9165 |
| 2.50 | 1.5e-02 | 2.3e-14 | 0.0743 | 0.1497 | 0.7220 |

and on **WP-B1's own fixture**, with my oracle pointed at it (its collimated
floor there is 0.9310 / 0.9841, and my oracle agrees with its `'quadrature'`
to `1 − fid` = 2.5e-03):

| tilt / lens NA | s1 fit rel | k1 fit rel | `stationary_phase` | `local_quadrature` | `quadrature` |
|---|---|---|---|---|---|
| 0.50 | 5.5e-05 | 1.4e-10 | 0.9323 | 0.9845 | 0.9975 |
| 1.00 | 3.3e-04 | 7.1e-10 | 0.9367 | 0.9854 | 0.9970 |
| 1.50 | 1.3e-03 | 1.3e-14 | 0.9322 | 0.9841 | 0.9969 |
| **2.00** | **4.1e-03** | 2.6e-14 | **0.0000** | **0.0061** | 0.9681 |
| 4.00 | 6.7e-03 | 1.5e-14 | 0.0000 | 0.0000 | 0.2467 |

Two controls rule out every other explanation:

* **The chart alone is not the failure.**  A COLLIMATED input on the SAME
  over-sized chart (`input_na = 0.3151`, OPD fit residual 0.164 waves, s1 fit
  RMS 2.16 µm), where the S6 term is absent because `k1 == 0`, returns
  **0.891 / 0.895** — the floor.  Only the S6 term is this sensitive.
* **It is the fit order, not the physics.**  The same over-sized chart at
  `poly_order=6` (s1 fit rel 2.7e-04) returns **0.9098 / 0.9690**; at order 7,
  0.8967 / 0.9666.  An explicit `input_na = θ` (which stops `na_proxy`
  tripling the box) does the same: 0.898 / 0.971 at tilt 2× NA.

**The fix.**  `lumenairy/elements/lenses_maslov.py` — the S6 fallback now
scores BOTH factors.  A new `_S1_FIT_RESIDUAL_MAX = 2.5e-3` is compared
against the entrance-coordinate fit's own relative RMS residual over the
traced rays; above it the OPD-only saddle is kept and the warning fires naming
the two remedies (`poly_order`, `input_na`).
`input_wavevector_saddle=True` overrides it, exactly as it overrides the `k1`
bar.  The bar is the geometric mean of fixture B's bracket
(1.8e-03 last good … 3.3e-03 first collapse = 2.4e-03), which also sits a
factor 1.9 above the other chart's last good row and 1.6 below its first bad
one.  Cost: one `(n_rays × M) @ (M × 2)` GEMM and a reduction on the ENGAGED
path only — **5.7 ms** at 16⁴ rays × 70 terms, against 0.078 s / 0.121 s for
the whole engaged `stationary_phase` / `local_quadrature` call on a 40 × 40
ROI of this chart (7 % / 5 %).  Nothing else pays anything.

Every case that works today still engages: off-axis converging 6.0e-06,
astigmatic 7.5e-06, hard edge at 0.8 of the pupil 4.5e-04, speckle 0.01 rad
4.5e-04, the report's converging f = +40 mm 5.5e-06 and diverging f = −25 mm
6.7e-06, and every tilt to 1.5× the lens NA on both fixtures.  The WP-A4 S6
fixture (diverging f = −0.5 mm) measures 1.1e-02 here but is already refused
by the `k1` bar (0.957), so its behaviour is unchanged.

**Verified**: all 27 byte-identity rows unchanged against 8dab7de5 and still
correct against 2871e92e^; the 28 shipped pins green; two new pins with a
fail-before on the pre-fix build.

### 4.2 V2 (P1) — `_K1_FIT_RESIDUAL_MAX = 0.5` is loose by one to two decades, and its ladder does not reproduce.  NOT FIXED — owner decision

**The ladder does not reproduce.**  Running the WP-B1 report's §3 inputs on
the WP-B1 fixture with the shipped code, reading the residual the driver
itself prints:

| §3 row | report | measured here | verdict |
|---|---|---|---|
| pure tilt | 1.4e-11 | 1.4e-10 … 1.5e-14 | reproduces |
| converging f = +40 / diverging f = −25 mm | 1.1e-05 / 1.3e-05 | **1.07e-05 / 1.27e-05** | reproduces |
| hard-edged aperture at 0.8 / 0.95 / 0.6 / 0.4 | 7.2e-02 / 1.5e-01 / 2.7e-01 / 2.9e-01 | **7.65e-02 / 1.45e-01 / 2.63e-01 / 2.92e-01** | reproduces |
| speckle 0.002 / 0.005 / 0.01 / 0.02 / 0.05 rad rms | 1.1e-01 / 2.6e-01 / 4.7e-01 / 7.1e-01 / 9.1e-01 | **4.7e-03 / 1.2e-02 / 2.3e-02 / 4.6e-02 / 1.1e-01** | **20× off** |
| speckle 0.1 / 0.2 / 0.3 / 0.6 | 9.6e-01 / 9.8e-01 ×3 | 2.2e-01 / 3.9e-01 / 5.1e-01 / 7.7e-01 | does not reproduce |

Robust to `ray_field_samples` 16 / 24 / 32 / 48 (0.0202–0.0243 at speckle
0.01) and to `poly_order` 4 / 6.  The measured values match the analytic
prediction for white phase noise on a tilt, `sigma·sqrt(2)/(k0 dx theta)`, to
a constant 0.83 (the bilinear sampling of the noisy grid) — so it is the
report's column, not the estimator, that is anomalous.  The rows WITHOUT a
tilt carrier do land near 0.93, which is where the report's numbers sit; the
fidelity column beside them (0.192 for the 5.46 saddle) is unmistakably the
WITH-carrier case.  **The bar is derived from the one column that does not
reproduce.**

**And the bar is loose.**  Re-measuring the same ladder against the exact
pointwise `'quadrature'` on the same chart (a geometric-optics oracle is not a
truth for speckle, so the report's oracle cannot score these rows), on fixture
B, speckle on a tilt 1× the lens NA carrier:

| speckle rms | k1 residual | `stationary_phase` | `local_quadrature` | ships |
|---|---|---|---|---|
| 0.002 rad | 2.1e-03 | 0.9083 | 0.9823 | fit |
| 0.005 | 5.4e-03 | 0.9047 | 0.9776 | fit |
| **0.010** | **1.1e-02** | **0.6561** | **0.7024** | fit |
| 0.020 | 2.1e-02 | 0.0966 | 0.1061 | fit |
| 0.050 | 5.3e-02 | **0.0004** | **0.0005** | fit |
| 0.100 | 1.1e-01 | 0.0137 | 0.0266 | fit |
| hard edge 0.95 | 1.4e-01 | 0.0458 | 0.0357 | fit |
| hard edge 0.60 | 2.6e-01 | 0.0000 | 0.0000 | fit |
| 0.600 | 6.0e-01 | 0.0000 (OPD-only) | 0.0000 | OPD |

The floor is 0.908 / 0.982.  The answer is gone by residual 0.05 — **a factor
9 below the bar** — and the bar does not fire until 0.5.

**The mechanism** is that the residual is the fit's VALUE error while the
saddle also consumes its two DERIVATIVES.  Instrumenting `_input_phase_terms`
on the last Newton iterate, the RMS of the S6 Hessian term `a33 / a44` goes

```
clean tilt ................ 8.32 / 3.38
speckle 0.005 rad rms ..... 105  / 127     (residual 5.4e-03)
speckle 0.020 ............. 558  / 530     (residual 2.1e-02)
speckle 0.050 ............. 1507 / 1214    (residual 5.3e-02)
```

— a **180× amplification at one ninth of the bar**.  A degree-4 Chebyshev fit
of a noisy field has a small value residual and an unbounded derivative error.

**Control**: the asymptotic METHOD is not the problem.  With NO tilt carrier
(so the OPD-only saddle is the right one and the fit is refused at residual
1.01), the same speckle ladder keeps **0.8985–0.8997 / 0.9784–0.9806** all the
way to 0.1 rad rms, and a hard edge at 0.6 of the pupil keeps 0.835 / 0.953.

**Why I did not change the bar.**  Two reasons, both about scope.  (1) On a
NON-collimated carrier the OPD-only fallback is never better — it puts the
spot on the wrong ray entirely — so "does engaging help?" has almost no
turning point, which is how the report reached 0.91; the bar's real job is
"is this answer trustworthy at all", and re-founding it is a design change to
shipped behaviour, not a verification finding.  (2) The report's own §3 claims
the hard-edge row at residual 0.072 IMPROVES 0.164 → 0.812 on its fixture,
which a tightened bar would refuse; I could not reproduce that row's fidelity
(its window is not stated) and I will not delete a claimed improvement on the
strength of a fixture the claim was not made on.  What I DID do: replace the
unreproducible speckle ladder in the source comment beside the constant with
the measured one, dated, saying plainly that the bar does not fire on a tilted
carrier until the answer is two decades gone, and naming
`integration_method='quadrature'` as the remedy — and add a pin that fixes the
statistic's calibration so this cannot drift again unnoticed.  See F1.

### 4.3 V3 (P3) — below the engagement bar, nothing warns

§3.4.  Not fixed: the bar is `_SADDLE_FLAT_INPUT_NA`, shared with the warning
gate, and moving it breaks the "the 5.46 warning fired exactly where 5.47
engages" symmetry and the sub-bar byte-identity — both of which WP-B1's report
explicitly bought.  Requested as F3.

### 4.4 V4 (P3) — the double count had 1/28 and 0/28 pin coverage

§3.6.  **Fixed here**: `test_verify_b1_the_input_phase_is_not_double_counted`
(parametrized over both integrators) scores PHASE against the exact
`'quadrature'` with the method's own collimated floor as the envelope.  Red
under the `opd_star` mutation (`stationary_phase`) and under the `opd_v`
mutation (`local_quadrature`); green on 8dab7de5 and on the fixed build.

---

## 5. What I changed

**Source.**

* `lumenairy/elements/lenses_maslov.py`
  * `_S1_FIT_RESIDUAL_MAX = 2.5e-3` (new), with the two-fixture ladder that
    derives it beside it.
  * The S6 fallback block now computes the entrance-coordinate fit's relative
    RMS residual and gates on it as well as on the `k1` residual; a new
    `_why` branch names the chart and its two remedies; the progress line
    carries `s1 chart fit` (placed BEFORE the `k1 fit residual … (engaged)`
    tail the existing test parses, and not spelling the word "residual", so
    that parser is untouched).
  * The `_K1_FIT_RESIDUAL_MAX` comment's speckle ladder is replaced by the
    measured one (§4.2), dated, with what the statistic can and cannot see.
  * `apply_real_lens_maslov.__doc__` gains the chart half of the gate.
* `docs/history/lumenairy.elements.lenses_maslov.md` — re-recorded with
  `scripts/record_history_fingerprints.py` in this change.

**Tests (added, nothing weakened).**

* `tests/unit/test_audit2609_b1_maslov_input_wavevector.py` — a new section 7
  with `_s6_report` / `_phase_rms_waves` helpers and three tests (5 ids):
  * `test_verify_b1_the_gate_refuses_a_chart_that_cannot_carry_ds1_dv2`
    (×2 methods) — two-sided on the new bar, with the measured ladder in the
    docstring, the `k1`-is-blind premise asserted, and the
    `input_wavevector_saddle=True` override.  **Fail-before: both ids fail on
    8dab7de5** (the pre-fix build accepts the bad chart).
  * `test_verify_b1_the_input_phase_is_not_double_counted` (×2 methods) —
    §4.4.
  * `test_verify_b1_the_k1_fit_residual_is_the_statistic_it_claims_to_be` —
    pins the residual against `sigma·sqrt(2)/(k0 dx theta)` to a factor of 2,
    which is what would have caught §4.2's 20×.
  * File total **28 → 33**, all green.

**Reports.**  This file and `VERIFY_WP-B1_CHANGELOG.md`.

---

## 6. Follow-up

* **F1 (owner decision, P1).**  `_K1_FIT_RESIDUAL_MAX`.  §4.2 has the
  measurement.  The statistic that would actually work is the fit's DERIVATIVE
  error, not its value error — e.g. refit `k1` at `poly_order − 1` and compare
  the two charts' `dk1/du` at the ray points, or (cheaper, and the report's
  own §6 item 3) an outcome test on the Newton's in-box non-convergence
  fraction.  Either is a design change.
* **F2 (P2).**  `na_proxy = na_lens + na_input` uses the 3-sigma-about-zero
  angular-spectrum moment, so a uniform tilt θ contributes 3θ and the pupil
  chart is sized three times too wide.  That is what drives V1 on a tilted
  input, and it also costs runtime on every tilted call.  The fix is for the
  owner of the chart sizing: use the MEAN launch direction plus the SPREAD,
  not the moment about zero.  Measured on fixture B at tilt 2× the lens NA:
  `na_proxy` 0.368 against a needed 0.158; with `input_na=θ` the OPD fit
  residual falls 0.164 → 1.3e-03 waves and fidelity recovers 0.078 → 0.898.
* **F3 (P3).**  Below `_SADDLE_FLAT_INPUT_NA` the answer is silently displaced
  by `f·θ` (§3.4).  One docstring sentence would close it: "an input tilt
  below the engagement bar is left on the OPD-only saddle and its spot is
  displaced by `f · theta` with no warning; pass
  `input_wavevector_saddle=True` if your `f` makes 3e-4 rad matter."  I did
  not add it because the paragraph was being edited concurrently by the
  follow-up; it is one sentence after the `input_wavevector_saddle` block.
* **F4 (P2), outside my ownership — `lumenairy/elements/_lens_jax.py`.**  The
  WP-B1 report's §6 item 6 and §7 item 5 are **factually wrong** about this
  file; §7 below has the measurement and the correct request.
* **F5 (P3).**  `local_quadrature`'s `opd_v` site has no pin of its own
  against a double count in any FILE but this one now; if WP-B7 refactors the
  two integrators to share a phase assembly, that pin is the one to keep.

---

## 7. Requested changes outside my ownership

1. **`lumenairy/elements/_lens_jax.py` (WP-B7) — and a correction to the
   WP-B1 report.**  Its §6 item 6 says `apply_real_lens_maslov_jax` "carries
   the same OPD-only saddle", and §7 item 5 says "the JAX Newton is the
   `_cheb_*` evaluator in that file; the same two terms apply".  **Neither is
   true.**  `_lens_jax.py` contains **zero** occurrences of `saddle`,
   `stationary`, `grad_v2` or `v2`: that function is a thin-OPD geometric
   phase screen plus a Maslov / Gouy index term, and its own docstring says so
   ("Despite the historical name this is **not** the phase-space *diffraction
   integral* of the NumPy `apply_real_lens_maslov`").  It has no v2 integral,
   so it cannot expand about the wrong stationary point of one.  A B7 engineer
   following the request as written would go looking for a Newton that does
   not exist.

   The real number, MEASURED on fixture B (screen, then
   `angular_spectrum_propagate` to the same readout plane, spot centroid
   against my exact trace of the input's own rays):

   | input | oracle landing | JAX sibling | error |
   |---|---|---|---|
   | collimated | 0.00 µm | −0.00 µm | 0.00 µm |
   | tilt 0.5× lens NA | 377.65 µm | 367.79 µm | **−9.86 µm (−2.6 %)** |
   | tilt 1.0× lens NA | 756.04 µm | 727.73 µm | **−28.31 µm (−3.7 %)** |
   | off-axis converging | 23.97 µm | 23.34 µm | −0.62 µm |

   So the JAX path **does** move the spot with the input (a screen carries the
   input phase by construction) and is wrong by a different mechanism: the
   thin-screen approximation under-shoots the chief-ray displacement by 2.6 %
   at half the lens NA and 3.7 % at one lens NA — ~1.6 diffraction-spot radii
   at 1× NA on this optic.  That is worth a finding of its own, but it is NOT
   S6 and importing `_input_phase_terms` into it would be meaningless.  The
   WP-B1 report's §6 item 6 / §7 item 5 should be struck.
2. **`lumenairy/elements/lenses_maslov.py` docstring, one sentence** — F3
   above.  It is inside my ownership but inside the paragraph the WP-B1
   follow-up was editing while I worked; flagged rather than raced.

---

## 8. Tests run

`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` on every run, one
process at a time.  Python 3.14.6, NumPy 2.4.6, Windows 11.

The shared working tree carries other Wave-4 packages' uncommitted edits, so
every run below is in an **isolated read-only export**: `git archive 8dab7de5`
into a scratch tree with only my three files copied over it
(`lumenairy/elements/lenses_maslov.py`,
`tests/unit/test_audit2609_b1_maslov_input_wavevector.py`,
`docs/history/lumenairy.elements.lenses_maslov.md`).

| command | result | duration |
|---|---|---|
| `pytest tests/unit/test_audit2609_b1_maslov_input_wavevector.py` | **33 passed** | 32.8 s |
| `pytest tests/unit/{a17_history_lint, a17_history_relocation, a4_maslov_gbd, b1_maslov_input_wavevector, v5_4_7_walker_v20_cross_backend_parity, a16_lens_config_round_trip}.py` | **902 passed** | 102.7 s |
| `pytest tests/unit -k "maslov or asymptotic"` | **442 passed, 5 skipped**, 0 failed | 822.8 s |
| `python validation/run_all.py test_lenses` | **ALL 1 files passed** | 35.6 s |
| `ruff check` (whole export) | **All checks passed** | 1 s |
| `python scripts/record_history_fingerprints.py --check` | **OK**, every history document matches | 3 s |

The `maslov or asymptotic` selection is fully GREEN here, including
`test_audit2609_a4_verify_maslov_asymptotic.py::test_s6_saddle_warning_fires_only_on_a_non_flat_input`,
which the WP-B1 report had to leave red pending the restatement it requested —
that restatement landed in 2871e92e.  The 5 skips are PySide6 (1) and
CuPy/GPU (4).

Working-tree confirmation (other packages' in-flight edits present, so only my
own files' tests):

| command | result | duration |
|---|---|---|
| `pytest tests/unit/{b1_maslov_input_wavevector, a4_maslov_gbd, a4_verify_maslov_asymptotic, a17_history_lint}.py` | **126 passed** | 112.5 s |
| `ruff check lumenairy/elements/lenses_maslov.py lumenairy/propagators/asymptotic.py tests/unit/test_audit2609_b1_maslov_input_wavevector.py` | **All checks passed** | 1 s |
| `python scripts/record_history_fingerprints.py lumenairy/elements/lenses_maslov.py --check` | **OK** — re-recorded in this change | 2 s |

Probe runs (each in its own read-only archive, child process,
`lumenairy.__file__` asserted):

| probe | result | duration |
|---|---|---|
| `p4_bytes.py parent` / `head` / `fixed` — 27 byte-identity cases each | 27/27 as expected in all three pairings | 96 / 92 / 94 s |
| `p8_kw.py b1only` / `head` — the follow-up's 18 cells | 18/18 seam==keyword, 18/18 archive-to-archive | 41 / 63 s |
| `pytest …b1… ` against 4 mutant exports | 15 / 4 / 8 / 1 red of 28 | 33 s each |
| `p1_headline.py` — fixture B, 6 input classes × 5 arms + oracle | table §3.1 | 402 s |
| `p6_thresholds.py t1 / t2 / t3` + `p7_control.py` | tables §3.4, §4.2, §3.5 | 118 / 331 / 96 / 208 s |
| `p11b_s1ladder.py`, `p12_fixtureA.py` — the V1 ladders | tables §4.1 | 244 / 191 s |
