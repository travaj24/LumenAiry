# VERIFY-WP-B7b — independent adversarial re-verification of WP-B7b

Branch `audit-fixes-2026-09`, WP-B7b = `9cf94fa5`, parent = `ea374607`.  I did
not write WP-B7b.  Every number below is mine, taken with an oracle that imports
nothing from lumenairy, and every before/after is ARCHIVE-TO-ARCHIVE:
`git archive 9cf94fa5^ lumenairy` and `git archive 9cf94fa5 lumenairy` extracted
to two scratch trees, child processes with `cwd` and `PYTHONPATH` set to the tree
and `lumenairy.__file__` asserted to live under it.  Never through pytest, never
against the shared working tree (two other sessions were writing in it).

---

## 1. Verdict table

| # | WP-B7b's claim | verdict | my measurement |
|---|---|---|---|
| 1a | `'fga'` scores 0.1251 and `'phase_screen'` 0.9991 on the WP-B7b fixture | **REPRODUCED EXACTLY** by an independent oracle | 0.1250 / 0.9991 / `traced` 0.9995; rms 7.240 / 1.540 / 2.095 µm against oracle 1.524 µm (theirs: 0.1251 / 0.9991 / 0.9995, 7.241 / 1.540 / 2.095 / 1.523) |
| 1b | "`'phase_screen'` is the closer member at every NA" | **REFUTED as a general statement** — it is a property of their LENS, not of NA | on my N-LASF9 plano-convex, `'fga'` is closer at every NA of 0.057 → 0.218 and `'phase_screen'` falls 0.9984 → 0.8375 |
| 1c | "the swarm is not under-sampled; **it converges to the wrong field**" | **REFUTED — wrong cause** | the same deficit is present at `output_plane_distance = 0` (0.0737, no caustic), is monotone in the LAST SURFACE's curvature at fixed focal length, and is **removed entirely** by projecting the differential trace from the last surface to the exit-vertex plane (0.1250 → 0.9998).  It is a reference-plane bug in `_fga_core`, not an FGA model limit |
| 1d | nine routing decisions move, every other decision unchanged | **CONFIRMED** on my own probe | 4 of my 31 decisions move, all `fga → phase_screen`, all single-valued + in-zone + inside the envelope; multi-valued, over-budget, vertex, low-NA and diverging rows all unchanged |
| 1e | the route change is a net improvement | **CONFIRMED**, with a measured exception | it replaces 0.1250 by 0.9991 on lenses with a curved last surface; on a FLAT-last-surface singlet near the aberration budget it replaces 0.9967 (`fga`) / 0.9978 (`traced`) by 0.9639 (`phase_screen`) |
| 2a | `zeta_extrapolation` reports `W / two-branch band` | **CONFIRMED**, reproduced independently | my own meridional trace + `l_airy = 1/(k^{2/3} κ)` reproduces 0.33 / 1.03 / 3.01 / 5.65 / 14.00 / 27.29 / 531.50 to all printed digits; `fit_halfwidth == l_airy` exactly |
| 2b | the warning is two-sided and the field is byte-identical | **CONFIRMED** on my own fixture | silent at 0.42 / 0.74 / 1.03 / 3.01 / 5.65, one warning at 14.00 / 27.29 / 531.50; all four probe fields byte-identical parent↔head, warning cases included |
| 2c | the envelope: “−5.3 % / +4.9 % up to ~5 and +12.5 % / +22.8 % from ~10 up” | **NOT REPRODUCED** — one-fixture numbers | my single-optic ladder saturates near **+5 %** (0.989 → 1.041 over `W/band` 0.42 → 531.5); no rung reaches ±10 %, so the bar warns inside the docstring's own “safe” band |
| 2d | "the completion beats the multibranch it would fall back to at every plane" | **CONFIRMED** | 0.976–0.986 against multibranch 0.824–0.959 on my ladder |
| 2e | "at the widest-band plane it also beats `amplitude_model='ray_density'`" | **NOT REPRODUCED** | on my fixture `ray_density` (0.9993) and `caustic='wave'` (0.9983) beat `uniform` (0.976–0.986) at **every** plane |
| 3a | the aspheric exclusion (`exact_jacobian=True` silently ignored) | **CONFIRMED**, archive-to-archive | parent: predicate `False`, `_pick_ray_transfer(None/True/False)` all FD, all three FGA fields byte-identical; head: analytic at `None`/`True` |
| 3b | the field-frame omission raised `NotImplementedError` at call time | **CONFIRMED**, fail-before reproduced | parent picks the analytic primitive for `field_decenter` / `field_tilt` and it RAISES; head picks FD and completes |
| 3c | the analytic Jacobian removes a 2.4e-09 FD truncation | **CONFIRMED** on their asphere and on mine | vs my independent Richardson-extrapolated derivative: analytic 7.8e-12 / 6.6e-12, library FD 2.14e-09 / 5.83e-09; FD step ladder ÷100 per decade then round-off at 1e-8 |
| 3d | the all-conic field is byte-identical | **CONFIRMED** | identical digest parent↔head |
| 4 | the Migration notes' "way back" | **CONFIRMED for `method='fga'` and `exact_jacobian=False`; WRONG for `caustic_pad_dof=0.0`** | both byte-identical to the parent's answers; `caustic_pad_dof=0.0` returns `'phase_screen'` at the near, mid and far edges of the UNPADDED zone, so it can never restore `'fga'` |
| 5 | the pins are derived, not per-build | **CONFIRMED** | four mutations, exactly the expected reds and nothing else (§7) |

---

## 2. Oracle V — what it is and what it is worth

`scratchpad/vb7b/oracle_v.py` imports nothing from lumenairy.  Deliberately
different in ALGORITHM from the oracle WP-B7b used:

* **intersection** by Newton iteration on the implicit sag
  `F(t) = (z + t u_z) − z_v − sag(|x + t u_x|)` (WP-B7b solved the quadric in
  closed form), converged to `|F| < 1e-15` m, with even-aspheric support;
* **refraction** by vector Snell with the normal oriented ALONG propagation,
  `t = μ u + (cosθ_t − μ cosθ_i) n`;
* **dispersion** from Schott Sellmeier coefficients typed in here;
* **two independent propagators** for the same boundary-value problem — my own
  angular-spectrum transfer function (FFT) of the ray-traced exit-vertex field,
  and a brute-force Rayleigh–Sommerfeld-I sum
  `E(P) = (1/iλ) Σ A_k e^{ikr} z/r²` over (exit ray) × (azimuth).

| control | measured |
|---|---|
| my Sellmeier vs `get_glass_index`, 5 glasses (N-LASF9 850 nm, N-BK7 1.0 µm, N-SF11 633 nm, N-BAF10 1.064 µm, N-LAK22 1.55 µm) | **Δ = 0.0e+00** on all five |
| my exit-VERTEX state vs `rt.trace(...).at_exit_vertex()` (fixture V, 2000 rays) | height **2.7e-20 m**, slope **5.6e-17**, OPL **2.2e-19 m** |
| my last-SURFACE state vs `ray_transfer_jacobian` / `..._analytic` (two aspheres) | **1.7e-18 m / 1.4e-16 / 1.7e-18 m** |
| the SLOPE-vs-direction-cosine trap WP-B7b fell into | `max |u − L| = **1.66e-03**` on fixture V — 1e13× the agreement above, so the check is real and it passes |
| ASM oracle grid ladder dx → dx/2 → dx/3 (sampled at the coarse points) | rel L2 9.3e-03, **3.6e-03** |
| ASM pad 3 vs pad 4 | rel L2 **2.1e-03** |
| RS ray quadrature n_h 451 → 901 → 1801 | rel L2 5.3e-06, **1.1e-06** |
| RS azimuth 512 → 1024 | rel L2 **1.1e-06** (unchanged to 3 digits) |
| **ASM vs brute-force RS on a radial cut** (the two independent methods) | fidelity **0.99999332**, rel L2 **2.6e-03** after a 0.9986 scale |

So the oracle floor is ~4e-03 relative L2 (≈1e-05 in fidelity) — three decades
under the 0.88 fidelity gap item 1 is about.

**A negative result, recorded because the assert caught it before any
measurement.**  My first vector-Snell step used
`f = μ cosθ_i − cosθ_t` with the normal oriented ALONG the ray instead of
against it, i.e. the refraction sign flipped; the trace asserted TIR at the
first surface of a 10° incidence and could not have produced a number.  A sign
error that had produced numbers instead of an assert would have framed the
library.  Fixed before fixture V was ever scored.

**Shared-model caveat, stated once.**  Both this oracle and WP-B7b's build the
exit-plane boundary field from geometrical optics and then propagate it exactly.
That model is what `'traced'` implements, so an oracle-vs-member comparison is
not neutral between `'traced'` and `'fga'`.  It does not affect the conclusions
here, because the decisive readings are model-free: the diffraction limit at
NA 0.160 / 633 nm is an Airy radius of 2.4 µm and `'fga'` returns a 7.24 µm rms
blob where the oracle, `'phase_screen'` and `'traced'` all return 1.5–2.1 µm;
and the defect I localise in §3.3 is a 7.6-wave OPL discrepancy measured
directly against the library's own `at_exit_vertex()`, with no oracle involved.

---

## 3. Item 1 — the caustic route

### 3.1 Fixture V (mine, nothing shared with WP-B7's or WP-B7b's)

**N-LASF9 plano-convex, R1 = 1.45 mm, R2 = ∞, t = 0.55 mm, 0.524 mm aperture,
λ = 850 nm, N = 384, dx = 1.6 µm** — a different glass, a different SHAPE
(plano-convex, curved side first, against two biconvexes), a different
wavelength, a different grid.  `_system_na = 0.1500` (inside the brief's
0.10–0.20).  Readout at the intensity-weighted geometric best focus **1430.40 µm**,
derived from my own trace.  Caustic zone `[1429.1, 1446.3] µm`.  The beam is
swept to move the sag-screen estimate across the envelope: `w0 = 190 µm` reads
1.5365 rad (inside the 2.0 rad budget), `w0 = 100 µm` reads 0.1212 rad,
`w0 = 205 µm` reads 2.028 rad (outside).

### 3.2 The three members at the caustic, and the two defocused planes

`w0 = 190 µm` (sag-screen 1.5365 rad, single-valued, route `'phase_screen'`):

| plane | member | fidelity | I-overlap | rms µm | EE(3 µm) | power/oracle |
|---|---|---|---|---|---|---|
| focus 1430.40 µm | oracle | 1 | 1 | 3.257 | 0.8588 | 1 |
| | `phase_screen` | **0.9639** | 0.9998 | 3.424 | 0.8647 | 1.0000 |
| | `fga` | **0.9967** | 0.9998 | 1.457 | 0.8673 | 1.0205 |
| | `traced` | **0.9978** | 0.9999 | 2.195 | 0.8642 | 1.0000 |
| −30 µm | `phase_screen` / `fga` / `traced` | 0.9639 / 0.9968 / 0.9978 | | | | |
| +30 µm | `phase_screen` / `fga` / `traced` | 0.9639 / 0.9967 / 0.9978 | | | | |

(The fidelities are z-independent because every member and the oracle finish on
the same unitary angular-spectrum leg; the comparison is therefore a comparison
of exit-plane MODELS, which is what the route decides.)

At `w0 = 100 µm` (0.1212 rad) all three are ≥ 0.9995 and the ordering is
`fga` 1.0000 = `traced` 1.0000 > `phase_screen` 0.9995.

**So on fixture V the shipped route picks the WORST of the three members.**  The
loss is 0.033 of fidelity at 1.54 rad and 0.041 at 2.03 rad.

### 3.3 Where WP-B7b's reading is right, where it is wrong, and why

I reproduced their own fixture with my oracle, at their report grid, at their
test grid and at two wider/finer grids:

| fixture / grid | `phase_screen` | `fga` | `traced` | oracle rms |
|---|---|---|---|---|
| B7b, N = 256 dx = 1.4 µm (their report grid) | **0.9991** | **0.1250** | 0.9995 | 1.524 µm |
| B7b, N = 192 dx = 1.8 µm (their test grid) | 0.9991 | 0.1031 | 0.9996 | 1.521 µm |
| B7b, N = 512 dx = 1.4 µm | 0.9990 | 0.0818 | 0.9994 | 1.627 µm |
| B7b, N = 512 dx = 0.7 µm | 0.9991 | 0.0902 | 0.9995 | 1.524 µm |
| WP-B7's N-BK7 R = ±1.2 mm, 1.0 µm | 0.9985 | 0.1043 | 0.9980 | 2.410 µm |

Their numbers are real and reproduce to four digits from a completely
independent oracle.  **Their explanation does not.**

**(a) The deficit is not about the caustic.**  Scored at a ladder of output
planes on their own fixture (oracle = the traced exit field at `z = 0`, my ASM
oracle elsewhere):

| z (µm) | 0 | 231.5 | 462.9 | 694.4 | **925.9 (focus)** | 1249.9 | 1759.1 |
|---|---|---|---|---|---|---|---|
| in the caustic zone | no | no | no | no | **yes** | no | no |
| shipped route | traced | traced | traced | traced | **phase_screen** | traced | traced |
| `fga` | **0.0737** | 0.0864 | 0.1019 | 0.1147 | **0.1250** | 0.1443 | 0.1845 |
| `phase_screen` | — | 0.9990 | 0.9990 | 0.9990 | 0.9991 | 0.9991 | 0.9908 |
| `traced` | 0.9994 | 0.9994 | 0.9994 | 0.9994 | 0.9995 | 0.9996 | 0.9912 |

`'fga'` is *worse* at the exit vertex, where there is no caustic and no focus,
than at the caustic.

**(b) The deficit is a property of the LAST SURFACE.**  Bending sweep at FIXED
paraxial focal length (1430.4 µm), fixed glass, wavelength, aperture, grid and
beam — `R1` solved numerically for each `R2`:

| R2 (mm) | ∞ | −40 | −16 | −8 | −4 | −2.9 | −2.2 | −1.75 |
|---|---|---|---|---|---|---|---|---|
| 1/R2 (1/mm) | 0 | −0.025 | −0.063 | −0.125 | −0.250 | −0.345 | −0.455 | −0.571 |
| `fga` at the focus | **0.9998** | 0.9656 | 0.8053 | 0.5100 | 0.2326 | 0.1607 | 0.1228 | **0.1031** |
| `fga` power / oracle | 0.985 | 0.970 | 0.936 | 0.856 | 0.649 | 0.496 | 0.352 | 0.243 |
| `fga` at z = 0 | 0.9999 | 0.9630 | 0.7968 | 0.4912 | 0.2065 | 0.1332 | 0.0949 | 0.0747 |
| `phase_screen` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9999 | 0.9999 | 0.9997 | 0.9995 |

Monotone in the last surface's curvature, at fixed everything else, and already
fully present at `z = 0`.  The auto momentum sampling is IDENTICAL across the
rows that matter (`p_max = 0.050731`, `n_p = 13`, no warning) for both the
working plano-convex and the failing bent lens — so sampling is not the
discriminator.

**(c) The cause, localised and proved.**  `_fga_core`'s own comment says it
"trace[s] to the LAST SURFACE VERTEX; the image-side leg is added manually"
(`xv = dt.x + z_image*uxo`, `opd_tot = dt.opd + z_image*sqrt(1+u²)`), and
`DifferentialTransfer`'s docstring promises "base-ray state at the output
vertex".  `ray_transfer_jacobian` does not deliver that — it returns
`res.image_rays`, the state ON the last surface, without `at_exit_vertex()`:

| optic | last-surface sag at the marginal ray | `ray_transfer_jacobian` vs `at_exit_vertex()` |
|---|---|---|
| plano-convex, R2 = ∞ | 0.000 µm | `|Δx|` 0.0 m, `|Δopd|` 0.0 m |
| bent, R2 = −2.2 mm | −12.592 µm (14.8 waves) | `|Δx|` **2.12e-06 m**, `|Δopd|` **1.28e-05 m = 15.02 waves** |
| WP-B7b's biconvex, R2 = −1.6 mm | −4.755 µm (7.5 waves) | `|Δx|` **6.35e-07 m**, `|Δopd|` **4.80e-06 m = 7.58 waves** |

(`at_exit_vertex()` itself agrees with my independent trace to **2.7e-20 m /
4.3e-19 m**, so the library's production tracer is right and only the
differential primitive's convention differs.)  Every beamlet therefore carries a
spurious phase ≈ `k·sag(r)` — quadratic in radius, i.e. a spurious defocus /
spherical term of 7.6 waves at the rim on WP-B7b's fixture.

**Closing the loop.**  Monkey-patching `_pick_ray_transfer` to return a wrapper
that projects the state to the vertex plane
(`x − sag·u`, `y − sag·u_y`, `opd − sag·sec θ`), nothing else changed:

| optic | z | as shipped | + plane projection | + projection incl. the Jacobian leg |
|---|---|---|---|---|
| **WP-B7b's biconvex** | 0 | 0.0737 (P 1.001) | **0.9998** | 0.9998 |
| **WP-B7b's biconvex** | 925.9 µm (its focus) | **0.1250** (P 1.001) | **0.9998** | 0.9998 |
| bent R2 = −2.2 | 0 | 0.0949 (P 0.479) | 0.9999 (P 0.999) | 1.0000 |
| bent R2 = −2.2 | 1421.9 µm | 0.1228 (P 0.352) | 0.9999 (P 0.980) | 0.9999 |
| plano-convex R2 = ∞ (control) | 0 / focus | 0.9999 / 0.9998 | 0.9999 / 0.9998 | unchanged |

Fixed, `'fga'` scores **0.9998 at the caustic** — better than `'phase_screen'`'s
0.9991 and level with `'traced'`.  **FGA does not converge to the wrong field.**
It is one call-site away from being the best member in exactly the regime the
route now steers away from.  Blast radius: the only consumers that treat the
primitive's state as a vertex-plane state are four sites in `fga.py`
(`_fga_core` 1535, `_fga_coarse` 2173, the coarse trace 1253 and `_caustic_zone`
2444); every other element in the library calls `at_exit_vertex()` explicitly,
and `_lens_traced_multibranch.py:265` even names it "the single shared operator
for that".

### 3.4 Blast radius of the route change (archive-to-archive)

31 routing decisions on my own probe; **4 change, all `fga → phase_screen`**:

```
  B7b caustic                   fga -> phase_screen   <== CHANGED
  V conic=0 (ab=1.536) caustic  fga -> phase_screen   <== CHANGED
  V w0=100 caustic              fga -> phase_screen   <== CHANGED
  V w0=190 caustic              fga -> phase_screen   <== CHANGED
  V w0=205 caustic              fga -> fga            (2.028 rad: over budget)
  V conic=30 / 60 / 120 caustic fga -> fga            (47.6 / 93.7 / 185.9 rad)
  V multivalued caustic         fga -> fga
  V multivalued vertex          fga -> fga
  B7b vertex / past             traced -> traced
  V w0=100/190/205 vertex/half/past/far  traced -> traced   (12 rows)
  V conic=* vertex              traced -> traced      (4 rows)
  V diverging vertex            phase_screen -> phase_screen
  lowNA vertex / caustic        phase_screen -> phase_screen
```

The multi-valued branch is genuinely unchanged (I built the input as two
counter-tilted Gaussians, `_tilt_dispersion` 0.0938 > 0.06), and the gate is
two-sided on ONE lens by beam fill alone (1.536 rad → screen, 2.028 rad → fga).

### 3.5 The measurement that decides the `aberrated` condition — **keep it**

WP-B7b kept `'fga'` for an over-budget prescription and escalated, because on
its 2.289 rad conic fixture the gate demonstrably keeps the worse member
(`fga` 0.1228 against `phase_screen` 0.9990).  I could not reach M6's ~20 rad on
a tractable grid either — the sag-screen estimate scales as `NA³ r_beam/λ`, so
20 rad at NA 0.16 needs mm-scale beams and N ≳ 6e4 — **but the escalation does
not need that fixture, because the cost WP-B7b measured is the FGA defect of
§3.3, not the gate.**  On a lens where FGA is not broken, the gate keeps the
BETTER member every time:

| over-budget fixture | sag-screen | route | `fga` | `phase_screen` | gate keeps |
|---|---|---|---|---|---|
| fixture V, w0 = 205 µm | 2.028 rad | `fga` | **0.9952** | 0.9540 | the better member |
| plano-convex, semi = 380 µm | 2.911 rad | `fga` | **0.9917** | 0.8375 | the better member (+0.15) |
| WP-B7b's conic k = 30 (curved last surface) | 2.289 rad | `fga` | 0.1228 | 0.9990 | the worse member |
| bent R2 = −2.2, WITH the §3.3 projection | 0.017 rad | (screen) | **0.9999** | 0.9997 | — |

The third row is the FGA bug; the first two are the same question asked on an
optic the bug does not touch, and there the answer is unambiguous.  **Do not
drop the `aberrated` condition.**  The deciding measurement is stated and it is
not the H2 f/5 score as such: *score the three members on the H2 f/5 dual-oracle
fixture AFTER the §3.3 reference-plane repair.*  Scoring it before the repair
measures the defect, not the gate — and a drop decided on that reading would be
un-droppable later without another default move.  (Mutation M2 confirms the
condition is already pinned by three existing measurements: §7.)

---

## 4. Item 2 — the fold `zeta` envelope

### 4.1 My fixture and my ladder

**N-BAF10 biconvex R = ±2.6 mm, t = 0.70 mm, 0.90 mm aperture, λ = 1.064 µm,
N = 512, dx = 2.20 µm, w0 = 330 µm** — nothing shared with WP-B7b's N-LAK22
singlets or the K4 plano-convex.  Chosen by scanning three designs × 96 planes
for a single fold ring whose Airy layer the module's own gate accepts
(`l_airy` 3.0–3.5 µm needs `dx ≤ 2.5–2.9 µm`).  **One optic, one caustic, eight
planes** spanning `W/band` 0.42 → 531.5 — a two-sided ladder that does not mix
fixtures.

Oracle floor here: RS ray quadrature n_h 901 → 1801 rel L2 **1.1e-05**; ASM
pad 2 vs 3 **1.1e-04**.

| z (µm) | 1742.4 | 1703.1 | 1683.4 | 1634.2 | 1614.6 | 1594.9 | 1585.0 | 1565.4 |
|---|---|---|---|---|---|---|---|---|
| `zeta_extrapolation` | 0.42 | 0.74 | 1.03 | 3.01 | **5.65** | **14.00** | 27.29 | 531.50 |
| warnings emitted | 0 | 0 | 0 | 0 | **0** | **1** | 1 | 1 |
| `uniform` fidelity | 0.9834 | 0.9859 | 0.9850 | 0.9833 | 0.9830 | 0.9760 | 0.9774 | 0.9783 |
| **`uniform` power / oracle** | **0.989** | **1.019** | **0.975** | **0.986** | **1.011** | **1.049** | **1.045** | **1.041** |
| `multibranch` fidelity | 0.8244 | 0.8862 | 0.8796 | 0.9238 | 0.9375 | 0.9478 | 0.9507 | 0.9588 |
| `caustic='wave'` | 0.9983 | 0.9983 | 0.9983 | 0.9982 | 0.9982 | 0.9983 | 0.9983 | 0.9983 |
| `amplitude_model='ray_density'` | 0.9993 | 0.9993 | 0.9993 | 0.9993 | 0.9993 | 0.9993 | 0.9993 | 0.9993 |
| `uniform` vs the converged RS radial cut | 0.9493 | 0.9768 | 0.9792 | 0.9873 | 0.9875 | 0.9877 | 0.9886 | 0.9901 |

**What holds.**  The diagnostic is exactly what it says it is —
`zeta_extrapolation` reproduces from MY independent meridional trace's
two-branch band and `l_airy = 1/(k^{2/3} κ)` to every printed digit, and
`fit_halfwidth == l_airy` exactly.  The warning is two-sided, with the adjacent
rungs at 5.65 (silent) and 14.00 (loud).  The completion beats the multibranch
it would fall back to at every plane, so "falling back would be a regression"
is confirmed.  The field is byte-identical to the parent on all four of my
probe cases (`W/band` 5.65, 14.0, 531.5 and a `zeta_nonlinear` fallback),
warning cases included.

**What does not hold.**  The energy envelope.  The drift with the ratio is
real and in the same direction, but on my optic it **saturates near +5 %** and
never reaches ±10 %: at `W/band = 531.5` the power error is +4.1 %, inside the
band the shipped docstring reserves for ratios *under* ~5.  So on this optic
the warning above 8.0 is a false positive for absolute energy, and the "+12.5 %
.. +22.8 % from ~10 up" is a property of WP-B7b's three singlets, not of the
ratio.  The bar is a useful conservative flag; it is not a calibrated
5 % / 10 % boundary, and the constant's comment now says so and carries this
ladder (V-2).

Also not reproduced: "at the widest-band plane it also beats
`amplitude_model='ray_density'`".  On my fixture `ray_density` (0.9993) and
`caustic='wave'` (0.9983) beat `uniform` at **every** plane, by 0.015–0.023.

### 4.2 A defect at the SAFE end of the shipped envelope (V-3, not mine to fix)

`W/band ≲ 0.35` — the regime the shipped table and the docstring present as the
best ("where the two-branch band is WIDER than the fit band the completion is
the best member available") — is where the completion is catastrophically wrong
on my fixture:

| z (µm) | 1758 | 1760 | **1762** | **1764** | 1766 | **1768** | 1770 |
|---|---|---|---|---|---|---|---|
| `zeta_extrapolation` | 0.35 | 0.34 | **0.33** | **0.32** | — | **0.32** | — |
| `fit_residual` | 0.0032 | 0.0031 | 0.0049 | 0.0029 | — | 0.0054 | — |
| `uniform` power / oracle | 0.979 | 0.968 | **93.6** | **3543** | fallback | **6095** | fallback |
| `uniform` fidelity | 0.9791 | 0.9778 | **0.0713** | **0.0388** | — | **0.0311** | — |
| `multibranch` power / oracle | 0.844 | — | **93.4** | 3543 | — | 6095 | — |

The multibranch member blows up identically, so the defect is in
`apply_real_lens_traced_multibranch`'s `1/sqrt|J|` reconstruction, not in the
uniform dark fill — and that module's own power self-check DOES warn
(`"reconstructed grid power is ..."`, 4 warnings).  What is missing is at the
uniform layer: `apply_real_lens_traced_uniform` returns `fell_back=False`, a
healthy `fit_residual` (0.0029–0.0054) and `zeta_extrapolation = 0.32`, i.e.
its own diagnostics say "best case" on a field carrying 6095× the correct
energy.  Requested change in §8.

---

## 5. Item 3 — the analytic-Jacobian predicate

### 5.1 Archive-to-archive, predicate vs the primitive's own answer

The probe CALLS `ray_transfer_jacobian_analytic` on each class and records
whether it raised, then asks the predicate and the dispatcher:

| surface class | primitive accepts | parent predicate / picks | head predicate / picks |
|---|---|---|---|
| conic | yes | True / `None`→analytic, `True`→analytic, `False`→FD | same |
| even asphere `{4: 4e3}` | **yes** | **False / all three → FD** | **True / `None`,`True`→analytic, `False`→FD** |
| A4+A6 asphere | **yes** | **False / all three → FD** | **True / analytic** |
| biconic (`radius_y`) | no | False / FD | False / FD |
| field decenter | no | **True / `None`,`True` → analytic, which RAISES `NotImplementedError`** | **False / FD** |
| field tilt | no | **True / analytic → RAISES** | **False / FD** |

Both halves confirmed, the latent bug reproduced as a real call-time raise on
the parent.  `_is_all_conic is _analytic_jacobian_applies` is True on head.
The all-conic FGA field digest is **identical** parent↔head.

Field digests for the aspheric prescription (`apply_real_lens_fga`, 128², 40 µm,
50 mm out):

| | `exact_jacobian=None` | `True` | `False` |
|---|---|---|---|
| parent | `1e5e6952…` | `1e5e6952…` (silently ignored) | `1e5e6952…` |
| head | `9a8fdad9…` | `9a8fdad9…` | `1e5e6952…` (= the parent's) |

### 5.2 The swap, against an INDEPENDENT derivative

Richardson-extrapolated (O(h⁴)) central differences through my own aspheric
trace, meridional 2×2 block, on WP-B7b's A4 singlet and on an asphere of mine
(**N-BAF10, R1 = 20 mm, conic −0.6, `{4: 1.2e3, 6: −3.0e6}`, R2 = −35 mm,
t = 3 mm, 12 mm aperture, λ = 1.064 µm**):

| | base ray vs mine | analytic Jacobian vs mine | library FD Jacobian vs mine |
|---|---|---|---|
| A4 singlet `{4: 4e3}` | 1.7e-18 m / 1.4e-16 / 1.7e-18 m | **7.8e-12** | **2.14e-09** |
| my A4+A6 conic asphere | 8.7e-19 m / 1.9e-16 / 1.3e-18 m | **6.6e-12** | **5.83e-09** |

FD step ladder against the analytic side (full 4×4), showing the `h²` truncation
and the round-off turn:

| `h_pos` | 1e-4 | 1e-5 | 1e-6 | 1e-7 | 1e-8 |
|---|---|---|---|---|---|
| A4 singlet | 2.135e-05 | 2.135e-07 | **2.134e-09** | 7.65e-11 | 6.94e-10 |
| my asphere | 5.827e-05 | 5.826e-07 | **5.829e-09** | 6.01e-11 | 9.44e-10 |

WP-B7b's 2.4e-09 reproduces (2.13e-09 on the same optic with a different ray
set), the ÷100-per-decade scaling is textbook, and the analytic side is ~300×
closer to the truth than the FD side.  **Item 3 is confirmed in full.**

---

## 6. Item 4 — the Migration notes

| claim | verdict |
|---|---|
| `method='auto'` at a caustic now returns `apply_real_lens` + ASM where it returned FGA | confirmed: head `auto` digest == head `force phase_screen` digest; parent `auto` digest == parent `force fga` digest |
| "To get the old route back … `method='fga'`" | **confirmed byte-identically**: head `method='fga'` digest `3fae68d6…` == parent `auto` digest `3fae68d6…` |
| "(or … `caustic_pad_dof=0.0`)" | **WRONG.**  Measured on fixture V: `caustic_pad_dof=0.0` returns `'phase_screen'` at the near edge, midpoint and far edge of the UNPADDED caustic zone (it only narrows the zone; inside it this branch answers the same way), and `'traced'` at a plane the pad alone had brought in — byte-identical to the parent's `'traced'` there.  It can never restore `'fga'` |
| "To get the old behaviour back on an aspheric prescription, pass `exact_jacobian=False`" | **confirmed byte-identically**: head `False` digest == parent `None` digest |
| "a field-decentred / tilted conic changes from raising `NotImplementedError` to completing on the FD primitive" | confirmed (§5.1) |
| "an all-conic prescription is byte-identical" | confirmed (§5.1) |
| "8924 → 7484 bytes per FGA lattice point, so the chunk sizer fits **1.19×** more lattice points" | the **1440 B/point saving reproduces exactly** (434876 → 433436 on my 128²/40 µm configuration), but the RATIO is configuration-specific — 1.0033× on mine, because the rest of the per-point cost scales with `n_p²`.  The 1.19× needs its configuration quoted |
| item 1's "fidelity 0.1251 → 0.9991 … the two returned fields overlap each other at 0.1247" | consistent with my 0.1250 / 0.9991 |

---

## 7. Item 5 — are the pins derived?  Mutation matrix

Run in an isolated tree (`git archive 9cf94fa5 lumenairy tests pyproject.toml`),
never in the shared working tree.  Baseline (b7b + a4_s10 + h4_h5): **41 passed,
285.6 s**.

| mutation | red | comment |
|---|---|---|
| **M1** `return "fga"` (revert the route) | **6**, exactly: `b7b::…single_valued_field_at_a_caustic_routes_to_phase_screen`, `b7b::…aberrated_caustic_keeps_fga_two_sided`, and the four restated `a4_fga_s10` rows | 35 passed, 272.1 s |
| **M2** `return "phase_screen"` (drop `aberrated`) | **4**, exactly: `b7b::…aberrated_caustic_keeps_fga_two_sided`, `test_g1_gate_generality::test_matrix_fast_designs_never_route_to_phase_screen`, `test_fga::test_gate_h2_reroutes_low_na_aberrated_away_from_phase_screen`, `test_fga::test_gate_h2_aberration_estimate_calibration` | 33 passed, 35.7 s — the condition is pinned by three independent existing measurements |
| **M3** revert `_analytic_jacobian_applies` to the parent whitelist body | **3**, exactly: `b7b::…predicate_is_the_analytic_primitive_s_own_domain`, `b7b::…field_decentred_conic_falls_back_instead_of_raising`, `test_fga_h4_h5::test_h4_exact_jacobian_default_analytic_for_conic` | 22 passed, 283.8 s |
| **M4a** `_ZETA_EXTRAPOLATION_MAX = 1000.0` | **1**, exactly: `b7b::…uniform_warns_only_when_zeta_is_extrapolated` | 9 passed, 35.8 s |
| **M4b** `_ZETA_EXTRAPOLATION_MAX = 0.5` | **1**, the same test | 9 passed, 32.8 s — the bar is pinned in both directions |

No mutation reddened anything outside its own claim.  The pins are derived, not
per-build.

---

## 8. Defects, and what I changed

### Changed, in my own files (comments only; behaviour byte-identical)

**V-1 — `lumenairy/propagators/fga.py`, `_universal_route`'s caustic branch.**
The comment asserted "the swarm is not under-sampled; it converges to the wrong
field".  Fail-before for the correction: the same `'fga'` field scores 0.0737 at
`output_plane_distance = 0` on that fixture (no caustic), and 0.9998 once the
differential state is projected to the exit-vertex plane.  The comment now
carries the plane ladder, the curvature ladder and the localisation, and states
that the branch is choosing between members by current accuracy rather than by
what the frozen-Gaussian model can represent at a caustic.

**V-4 — same file, same comment.**  `caustic_pad_dof=0.0` was offered as a way
back to `'fga'`; measured, it returns `'phase_screen'` inside the narrowed zone
and `'traced'` outside it.  The comment now says `method='fga'` is the only
route to the swarm and why.

**V-5 — same file, `apply_real_lens_universal`'s `'fga'` bullet.**  The member
map stated the 0.9991-vs-0.1251 ordering without its scope.  It now carries the
measured reversal on a flat-last-surface singlet (`fga` 0.9967 / `traced` 0.9978
against `phase_screen` 0.9639 at 1.54 rad) and points a caller near the
aberration budget at `method='traced'`.

**V-2 — `lumenairy/elements/_lens_traced_uniform.py`, `_ZETA_EXTRAPOLATION_MAX`.**
The derivation presented one ladder as the envelope.  It now carries my
single-optic ladder (saturating near +5 % out to `W/band = 531.5`) and says the
bar is a conservative flag rather than a calibrated 5 % / 10 % boundary.

Nothing else moved: seven `apply_real_lens_universal` / `apply_real_lens_fga`
digests and all thirty-one routing decisions are byte-identical between the
worktree and the `9cf94fa5` archive after these edits.

### Requested changes outside my ownership

**R-1 (P1) — `lumenairy/raytrace/differential.py`: `DifferentialTransfer`'s
contract does not match its implementation.**  The docstring says
"`x, y, ux, uy` … base-ray state at the output **vertex**" and "`opd` … at
output", but `ray_transfer_jacobian` / `..._analytic` return `res.image_rays`,
which is the state ON the last surface.  They differ by the last surface's sag —
measured **7.58 waves of OPL / 0.64 µm of height** on WP-B7b's own N-SF11
biconvex and **15.02 waves / 2.12 µm** on a bent singlet — and agree to 0.0
only when that surface is flat.  Minimum edit (documentation):

```
    x, y, ux, uy : ndarray
        ``(N_rays,)`` base-ray state at the LAST SURFACE (the intersection
        point, not the exit-vertex plane; project with the same step
        :meth:`lumenairy.raytrace.TraceResult.at_exit_vertex` applies when a
        caller needs the vertex plane).
```

**R-2 (P1, the substantive fix) — `lumenairy/propagators/fga.py`: four sites
consume that state as a vertex-plane state.**  `_fga_core` (`lumenairy/propagators/fga.py:1535-1542`),
`_fga_coarse` (`:2173-2182`), the coarse trace (`:1253-1258`) and
`_caustic_zone` (`:2444`) all add the image leg from `dt.x` / `dt.opd` as if the
ray were on the vertex plane.  I own this file, but I have NOT shipped the fix:
it changes every FGA field on every prescription with a curved last surface
(0.1250 → 0.9998 on the WP-B7b fixture) and would move numbers pinned in
`test_fga.py`, `test_g1_gate_generality.py`, `test_niche_audit_w9_dispatch2.py`,
`test_niche_p8_capstone.py`, `test_niche_p7_seidel_gate.py` and the GBD/Maslov
comparison suites, none of which I own and none of which I may restate.  It
needs its own work package with its own oracle ladder.  The edit that was
measured (as a wrapper; inline it at each site):

```python
    # ray_transfer_jacobian leaves the base ray ON the last surface; this
    # module's image leg starts at the exit-vertex PLANE.
    sag = conic_sag(dt.x, dt.y, surfs[-1].radius, surfs[-1].conic,
                    surfs[-1].aspheric_coeffs, xp=np)
    x_v   = dt.x   - sag * dt.ux
    y_v   = dt.y   - sag * dt.uy
    opd_v = dt.opd - sag * np.sqrt(1.0 + dt.ux ** 2 + dt.uy ** 2)
```

Measured effect (monkey-patched, nothing else changed): WP-B7b's fixture
0.1250 → **0.9998** at its caustic and 0.0737 → **0.9998** at the exit vertex; a
bent singlet 0.1228 (power 0.352) → **0.9999** (power 0.980); **no change at all**
on a flat-last-surface control (0.9998 → 0.9998).  Adding the free-space leg to
the Jacobian as well changes nothing beyond the fourth digit.

**R-3 (P2) — `fixes/WP-B7b_REPORT.md` §2.4 and §7.**  "FGA converges, to the
wrong field" and the first escalation bullet rest on the misdiagnosis above.
Suggested: keep the measurement, replace the causal sentence with "the deficit
is independent of the output plane (0.0737 at `output_plane_distance = 0`) and
is the reference-plane defect R-1/R-2", and restate §7's first bullet as "score
the H2 f/5 fixture AFTER the reference-plane repair; before it, the reading
measures the defect, not the gate".

**R-4 (P2) — `fixes/WP-B7b_CHANGELOG.md` and `Migration-Guide.md:1645`.**  Both
offer `caustic_pad_dof=0.0` as a way back.  Exact edit, both places: delete
"(or `caustic_pad_dof=0.0` to narrow the caustic zone itself)" and replace with
"`caustic_pad_dof` only narrows the zone — inside it the route is unchanged and
outside it the plane leaves the caustic branch — so it is not a way back."
Also qualify "fits 1.19x more lattice points at a fixed `mem_budget_mb`" as
"…on that measurement's grid and swarm (the saving is a fixed 1440 B per lattice
point; the ratio depends on `n_p`)".

**R-5 (P2) — `lumenairy/elements/_lens_traced_multibranch.py` /
`_lens_traced_uniform.py`'s diagnostics.**  At `W/band ≲ 0.35` on my fixture the
multibranch reconstruction blows up 94×–6095× in power (its own power self-check
warns), and `apply_real_lens_traced_uniform` passes that field through with
`fell_back=False`, `fit_residual` 0.0029–0.0054 and `zeta_extrapolation` 0.32 —
its diagnostics report the best case.  Minimum fix inside my module would be to
surface the multibranch power ratio in the uniform diagnostics; I did not ship
it because the threshold would be a new bar with no derivation, and the blow-up
itself belongs to `_lens_traced_multibranch.py`.

**R-6 (P3) — `_lens_traced_uniform`'s docstring envelope table.**  "at the
widest-band plane it also beats `apply_real_lens_traced(amplitude_model=
'ray_density')`" does not generalise: on my fixture `ray_density` (0.9993) and
`caustic='wave'` (0.9983) beat the completion at every plane.  Suggested:
scope the sentence to that fixture.

---

## 9. Follow-up

* **P1** Repair the FGA exit-plane reference (R-1 + R-2) and then re-score the
  caustic route.  With it repaired, `'fga'` is the best member at a caustic
  (0.9998 vs `'phase_screen'` 0.9991 on WP-B7b's own fixture), so the WP-B7b
  route change becomes a workaround that should itself be revisited — and the
  `aberrated` escalation resolves at the same time.
* **P1** Until then, `method='fga'`, `apply_real_lens_fga` and
  `apply_real_lens_fga_vector` return a field with a spurious quadratic phase of
  `k·sag(r_last)` on any prescription whose last surface is curved.  The router
  now avoids that in ONE branch only; the multi-valued branch and the
  over-budget branch still route there, and a direct call always does.
* **P2** `_ZETA_EXTRAPOLATION_MAX` should be re-derived on ≥ 2 optics as a
  two-sided energy bar, or re-labelled as the conservative flag it is (the
  comment now labels it).
* **P3** `caustic='uniform'` vs `caustic='wave'` vs
  `amplitude_model='ray_density'` at a resolved fold: on my fixture the shipped
  member ordering is the reverse of the one the docstring records.  A
  member-selection pass over a few optics would settle which member the
  docstring should recommend.
* **P3** `_caustic_zone` (`fga.py:2444`) computes `z = −x/u` from the same
  last-surface state, so the zone is offset by ~the last surface's sag
  (≈ 0.5 % of the focal distance on the fixtures here).  Small, but it rides on
  the same fix.

---

## 10. Commands, counts, durations

All single-threaded (`OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1`),
one process of mine at a time, foreground, `-X faulthandler`, 2026-09-14.  A
full two-lane unit run of the orchestrator's was running concurrently on the
same box throughout, so wall clocks carry contention.

| command | result |
|---|---|
| `probe1.py` — Sellmeier + raytrace + routing controls | Δn = 0.0e+00 ×5; exit state 2.7e-20 m / 5.6e-17 / 2.2e-19 m; slope-vs-cosine trap 1.66e-03 |
| `probe2.py` — oracle convergence, two independent methods | ASM ladder 9.3e-03 → 3.6e-03; RS 5.3e-06 → 1.1e-06; ASM vs RS fid 0.99999332 (15.5 s) |
| `probe3.py` — three members × three planes × two beams | table §3.2 (≈ 60 s) |
| `probe4.py` — WP-B7b's and WP-B7's fixtures under MY oracle | 0.1250 / 0.9991 / 0.9995, four grids (≈ 90 s) |
| `probe5.py` — bridge V → B7b, one parameter at a time | the switch is the SHAPE |
| `probe6.py` — shape / NA / aberration discrimination, 21 fixtures | §3.3(b), §3.5 |
| `probe7.py` — bending sweep at fixed focal length | §3.3(b) |
| `probe9.py` — output-plane ladder | §3.3(a) |
| `probe10.py` — where `ray_transfer_jacobian` leaves the ray | 15.02 / 7.58 / 0.0 waves |
| `probe11.py` — the projection, monkey-patched | 0.1250 → 0.9998, control unchanged |
| `probe12.py` — fold-fixture scan, 3 designs × 96 planes | fixture of §4.1 |
| `probe13.py` / `probe14.py` — the fold ladder, 8 planes × 4 members + RS cut | §4.1 (≈ 100 s) |
| `probe15.py` — the low-ratio blow-up window | §4.2 |
| `probe16.py` — Jacobian vs my Richardson derivative + step ladder | §5.2 |
| `archprobe.py` in `parent` and in `head` (cwd + PYTHONPATH = the tree, `lumenairy.__file__` asserted) | 31 routes, 6 predicate classes, 4 fold cases, 1 conic FGA digest — diff in §3.4 / §4.1 / §5.1 |
| `archprobe2.py` in `parent`, `head` and the worktree | 7 digests + 6 memory models — §6, and the proof my edits move nothing |
| `pytest tests/unit/{b7b,a4_fga_s10,fga_h4_h5}` in the isolated tree (baseline) | **41 passed**, 285.6 s |
| M1 / M2 / M3 / M4a / M4b mutations | 6 / 4 / 3 / 1 / 1 red, exactly as listed; 272.1 / 35.7 / 283.8 / 35.8 / 32.8 s |
| `ruff check` the two modules + the three test files | **All checks passed** |
| `scripts/record_history_fingerprints.py <module> --check` ×2 | no history document for either module (confirms WP-B7b) |
| `pytest tests/unit/{a17_history_lint,a17_history_relocation,b7b}` (after my edits) | **754 passed**, 78.6 s |
| `pytest tests/unit/{a4_fga_s10,fga_h4_h5,niche_k4_uniform_caustic,niche_r2_pearcey_cusp}` | **52 passed**, 269.9 s |
| `pytest tests/unit/{fga,g1_gate_generality,niche_audit_w9_dispatch2,niche_p8_capstone,niche_p7_seidel_gate}` | **139 passed**, 405.1 s |

Not run, and why: the `-k "fga or caustic or uniform or traced"` slice (1850 s on
WP-B7b's box) and `validation/run_all.py` — my changes are comment-only and
proved behaviour-neutral archive-to-worktree, and a second heavyweight selection
alongside the orchestrator's concurrent two-lane run would have contended for
the box.  The suites that pin every route, predicate and fold number I touched
are in the table above and are green.

### Concurrency note

One of my own probe processes (`probe8.py`, an `n_p`-ladder sweep) was still
running from the session that was killed mid-turn; per the standing rule I did
not kill it, and every measurement above was taken alongside it.  I wrote only
`lumenairy/propagators/fga.py`, `lumenairy/elements/_lens_traced_uniform.py` and
my two report files; the mutation matrix ran in an isolated tree and that tree's
two modules were checksum-verified back to the `9cf94fa5` archive afterwards.
