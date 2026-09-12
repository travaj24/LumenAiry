# VERIFY-A2 — independent re-verification of WP-A2 (`apply_real_lens` analytic model)

Diff under test: `b97c0b6e` ("fix(lens-analytic): WP-A2 …"), base `b97c0b6e^` —
4163 insertions / 606 deletions over `lumenairy/elements/_lens_real.py` (1444 changed lines),
`lumenairy/elements/lenses.py` (85), six test files and one validation file.
Environment: CPython 3.14, numpy 2.4.6, numba 0.65.1, **numexpr NOT installed**, scipy present;
`OPENBLAS_NUM_THREADS=1` on every invocation.  The machine was shared with ~10 other agent
processes throughout, so absolute wall clocks below are slower than the WP's; ratios are what I
compare.

**Nothing in this report is read from the WP's numbers.**  Every figure is re-measured, and every
P1 is additionally checked on a fixture the WP did not use with an oracle written from scratch in
this pass (`scratchpad/v_oracle.py`: closed-form quadratic conic intersection — *not* Newton, which
is what the audit's `repro/RL-CORE/oracle.py` uses — plus 3-D vector Snell and a signed transfer to
the exit vertex plane).  That oracle agrees with the audit's Newton oracle to **4.3e-19 m** in
landing coordinate and **6.9e-18 m** in piston-free OPL on the plano-convex and the cemented
doublet, so the two are interchangeable and the agreement below is not a shared-implementation
artefact.

---

## 1. Verdicts

| ID | Verdict | My measurement (oracle in §3) |
|---|---|---|
| **L1** seidel_correction | **VERIFIED-WITH-NOTES** (one P2 open, §5.1) | All three defects (a)/(b)/(c) are gone and independently confirmed: 8 mm doublet 173.466 → **1.126 nm** (154.1×) and 4 mm 10.857 → 1.841 nm (5.90×) against MY oracle — the report's numbers to 4 digits; `_split_step_fan_opl` reproduces the wave model's own exit OPL to **0.03–1.2 nm** on five fixtures; three NEW fixtures give 161× / 693× / 215×. **But** on a fast (f/2) singlet the ρ⁴+ρ⁶ fit is extrapolated over the outer 35 % of the pupil area and the focal peak drops **37 %** where masking that band gives **+37 %** — demonstrated, §5.1. Also: the replaced test's docstring quoted "~50 nm" for a shipped 1.126 nm (fixed, §4.2) |
| **L2 + L19 (tilt)** | **VERIFIED** | tilted flat face deviates −2.57474e-03 rad in BOTH branches against a thin prism of 2.57544e-03; branch-to-branch 4.3e-08; exact rotated-sphere height error 1.68 / 10.26 / 80.1 nm at 1 / 5 / 20 mrad (from 2.002 / 10.011 / 40.077 µm); both axes, combined tilt, tilt+decenter and an asphere all ≤ 1.84e-8 m; `raytrace` deviates the same axis with the same sign |
| **L3** remap phase | **VERIFIED** | 1-D remap transports a 1.93-wave ρ⁴+ρ⁶ residual to **1.7e-4 waves** of my independent ray-transport oracle (pre-fix it transported 0); 2-D remap flat-vs-aberrated separation 0.89 of peak (pre-fix 4.7e-16) |
| **L4** carrier momentum | **VERIFIED** | `'auto'`, ndarray and `TiltedCarrier` all read `q = n sinθ` to ≤ 4.4e-16 relative in **N-BK7 and N-SF11** (the defect is a factor n = 1.78); banded rows byte-identical to the whole-grid field for every applicable evaluator (byte identity) |
| **L5** sag-callable cache | **VERIFIED** | mutated attribute, mutated closure cell and same-bytecode-different-constant all MISS; end-to-end warm-cache result after a mutation is **byte-identical** to a cold rebuild |
| **L6** glass cache | **VERIFIED** | re-pointing a registry entry: warm == cold byte-for-byte; restoring the entry returns the ORIGINAL field byte-for-byte |
| **L7** displaced exit index | **VERIFIED** | remap exit OPL 8e-13 m rms against a closed-form immersed trace at n_exit = 1.7912 (the n_exit = 1 defect is 2.08e-5 m = 33 waves) |
| **L8** remap guards | **VERIFIED** | 15 (N, pad) combinations accepted including **odd N = 257** and non-power-of-two N = 384, pad 1–4, a TOP-HAT input and an anamorphic dy = 2·dx |
| **L9** resolutions / Newton | **VERIFIED-WITH-NOTES** | the lattice warning fires exactly when launch pitch > 2·min(dx, dy) (5 configurations); the Newton early exit is **bitwise** — forcing the old 24 sweeps changes no bit in any of the four loops across 2 prescriptions × 4 radii × 3 grid sizes. Note: the sweep saving is 24 → **13** on the ray maps (3 on the cos-LUTs), i.e. `_surface_sag_general` calls 150 → 84 (1.79×), not the "2 sweeps / ~10×" the report and RL-MODELS state (§5.3) |
| **L10, L11, L18, L20** | **VERIFIED** | `assert` → `ValueError` (both under-length and over-length); `_VALID_SCREEN_OBLIQUITY` now read by the message; anamorphic guard reports `semi=0.512 mm` for a 512 × 64 grid at dx = 2 µm / dy = 40 µm; `slant+seidel` refused |
| **L12** slant coefficient | **VERIFIED** | my own O(θ²) expansion reproduces the exact / paraxial / v5.25.0 coefficients and the ratio **n/(n−1) = 2.9414**; measured 2.9414–2.9417 at four radii; the SHIPPED screen reads 0.0003 / 0.0088 / 0.0202 / 0.1562 nm where paraxial reads 1.5644 / 12.5299 / 18.3718 / 62.4477 nm. The f/5 hammer table is reproduced exactly (dx = 6 µm: 25.26 / 50.31 / 26.76 µm, ratio 1.059) and the migration note is accurate |
| **L13** Fresnel power | **VERIFIED** | `4 n1 n2/(n1+n2)²` to 1.1e-16…4.4e-16 relative on five interfaces including a **high-index** crown, a reversed cemented pair and a genuinely **complex-index** glass; a three-face stack equals the product of three T; `fresnel=False` byte-identical to the default |
| **L14** stop_index | **VERIFIED** | −1 → last surface, −2 → surface 0, 2 / 5 / −3 / float / bool / str all `ValueError`, `np.int64` accepted; `prepare_real_lens` reads the key identically |
| **L15** form_error | **VERIFIED** | `(N,)`, `(N,N,1)`, `(N/2,N/2)`, complex and bool all refused with the §2 prefix; float32 accepted; value −0.511441 rad unchanged |
| **L16** in-place sag | **VERIFIED** | byte-identical to the previous expression at float64 **and float32** over 6 (R, k) including flat; measured 113.1 → 32.0 ms and 5.13 → 1.13 float64 grids at N = 2048 |
| **L17** phase screen | **VERIFIED** | `_screen_exp` byte-identical to `np.exp(-1j k0 opd)` at float64 for nm-, µm- and mm-scale OPD (4.5e4 rad), and correctly falls back for float32; measured 108.7 → 46.2 ms / 2.00 → 1.50 complex128 grids, `E *= ph` 2.88× |
| **L19** absorption | **VERIFIED** | own Beer-Lambert oracle on two fixtures the WP did not use — a curved-FRONT plano-convex (local path `t − sag₁`) 4.9× better than the axial factor, and a cemented pair with **two different κ** 9.8× better; "no attenuation after the last surface" and a flat plate exact |
| **E4** aspheric kernel | **VERIFIED** | F-order, transposed, BOTH negative strides and a strided view all byte-identical to the C-ordered answer |
| **V1 / V2** vacuous tests | **VERIFIED** | the replacements are falsifiable, derived and two-sided; one docstring number was wrong and is fixed (§4.2) |
| **C1** `R = 0` | **VERIFIED** | `ValueError` with the §2 prefix; `inf` / `None` / `-inf` still return zeros |
| **C4** `sas` anamorphic | **VERIFIED** | `dy = 2 dx` refused, `dy = dx` and `dy = None` still run |
| **Not-regressed gates** | **VERIFIED** | `p1b` 0.8480 / 1.1770 / 1.8294 / 10.8574 nm and the prefactor table 0.0695 / 0.0486 / 0.0480 / 0.0492 / 0.0516 reproduce exactly; `p10_banded_identity` **44 OK / 0 FAIL** |

**Defects found and fixed in this pass** (all in files VERIFY-A2 owns; §4): one P1-class API
defect in `_lens_real.py` (the two mirror guards disagreed about
`allow_unfolded_equivalent` — the coordinator's item), one wrong number in a test docstring, one
test fixtured on a value another work package's in-flight change makes illegal.

---

## 2. Repro scripts, re-run

| script | report's after-number | mine | |
|---|---|---|---|
| `RL-CORE/p1b_thin_vs_rayoracle.py` | 0.848 / 1.177 / 1.829 / 10.857 nm | **0.8480 / 1.1770 / 1.8294 / 10.8574 nm** | ✔ |
| " (sag·NA² prefactors) | 0.0695 / 0.0486 / 0.0480 / 0.0492 / 0.0516 | **identical** | ✔ |
| `RL-CORE/p3_fresnel_energy.py` | 0.958057 / 0.917873 / 0.958057 / 0.993599 | **identical to 6 dp** | ✔ |
| `RL-CORE/p5_slant.py` | 0.037 / 0.017 / 1.641 / 23.907 / 891.54 / 0.053 nm | **0.0366 / 0.0171 / 1.6406 / 23.9073 / 891.5381 / 0.0528** | ✔ |
| `RL-CORE/p5c_slant_sign.py` | ratio 2.9414–2.9417 | **2.9414 / 2.9415 / 2.9415 / 2.9417** | ✔ |
| `RL-CORE/p7b.py` | `stop_index=2` raises | raises with the §2 message (script aborts by design); rows None/0/1 unchanged at 0.26930 / 0.26930 / 0.27880 | ✔ |
| `RL-CORE/p9_formerr.py` | 4 shapes, Δφ = −0.511441 rad | **identical** | ✔ |
| `RL-CORE/p10_banded_identity.py` | 44/44 OK | **44 OK, 0 FAIL**; complex64 6.745e-07; memmap == ram; prepared == apply at c128 **and** c64 | ✔ |

`RL-CORE/p4_absorption.py` still does not run on this build (`lumenairy.glass.register_glass`
does not exist) — the WP said so and it is true.

---

## 3. Independent checks (new fixtures, new oracles)

### 3.1 L1 — `seidel_correction`

**The central claim, checked directly.**  The whole L1 fix rests on `_split_step_fan_opl` being
*the split step's own exit OPL*.  I tested that with no ray oracle in the loop at all — comparing
it against the wave model's unwrapped exit phase — on five fixtures **at converged sampling**:

| fixture | model − wave | ray − wave | ray − model (what the block fits) |
|---|---:|---:|---:|
| plano-convex 4 mm | **0.037 nm** | 0.848 nm | 0.845 nm |
| meniscus R = 20/25, 4 mm | **0.032 nm** | 5.213 | 5.212 |
| air-spaced doublet 6 mm (4 surfaces) | **0.036 nm** | 29.828 | 29.828 |
| f/2 biconvex R = ±8.24 mm, 4 mm | **0.635 nm** | 807.7 | 807.7 |
| cemented doublet 8 mm | **1.222 nm** | 173.395 | 173.333 |

So the model reference is the model, to sub-nm, on a meniscus, a four-surface **air-spaced**
design and an f/2 singlet — none of which the WP used.  The WP reported 1.001 nm on the one
fixture it measured; I get 1.222 nm there and better elsewhere.  **This is the strongest single
piece of evidence that defect (c) is gone**, and it is independent of any ray trace.

*A sampling caveat that cost me two false alarms and is worth recording:* these fixtures need
`dx ≈ 1.45·aperture/2048`, NOT the `0.3·λ/NA` a carrier-Nyquist rule gives.  At N = 512 the
meniscus reads `model − wave = 557.8 nm` and the air-spaced doublet 949.2 nm, converging to 0.03 nm
by N = 2048 as the hard-aperture edge stops aliasing through the in-glass ASM (|E| ripple
0.640–1.130 → 1.052–1.057 over the same window).  Any future re-measurement of these must check
that convergence first.

**Decision rule.**  `_split_step_fan_opl` + `at_exit_vertex` + the ρ⁴ basis give, in my external
replica of the shipped block (same fan, same helpers, fit and gate recomputed by hand):

| fixture | correction PV | fitted ρ⁴+ rms | gate | residual after the fit | ρ² if the basis allowed it |
|---|---:|---:|---|---:|---:|
| plano-convex 4 mm | 3.681 nm | **1.349 nm** | SKIP | 0.000 nm | 0.000 nm |
| meniscus 4 mm | 19.915 | 7.290 | FIRE | 0.000 | 0.001 |
| air-spaced doublet 6 mm | 110.558 | 40.510 | FIRE | 0.000 | −0.001 |
| f/2 biconvex R = ±8.24 mm, 4 mm | 2690.9 | 966.4 | FIRE | 0.713 | −14.7 |

The plano-convex gate figure is **1.349 nm**, matching the WP's 1.35 nm exactly; the ρ² content of
the correction curve is 0.001 nm or less on three of four fixtures, i.e. the ρ⁴-up basis is not
aliasing a defocus term into ρ⁴ — the concern that motivated excluding ρ² in the first place.

**End to end, against MY oracle:** 8 mm doublet **173.466 → 1.126 nm (154.1×)**, 4 mm doublet
10.857 → 1.841 nm (5.90×) — the report's numbers reproduced to four digits from a different oracle.

**New fixture, immersed rear (n_exit = N-SF11):** exit OPD 2.368 nm with the flag off and the
corrected call **bit-identical** to it — the gate skips, correctly, on a fixture where both the
exit-vertex transfer and the exit momentum `p = n_exit·L` carry a non-unit index.  Pinned in
`TestVerifyL1GateSkipsAnImmersedRearSinglet`.

**Off-axis / tilted input** (the residual risk the report states).  On a fast biconvex at
θ = 0 / 0.02 / 0.05 rad the radial screen still improves the exit OPD (302.9× / 8.7× / 3.7×) rather
than degrading it, so the docstring's "not valid for an off-axis input" is conservative rather than
wrong.  It is stated honestly; I would not change it.

**Three new fixtures, end to end at converged sampling** (exit OPD rms over the standard
|x| ≤ 0.85·r_pupil window, against my oracle; through focus refined in two passes around the
marginal-ray crossing, step ≤ 0.1·DOF):

| fixture | N, dx | OFF | ON | gain | Δpeak | Δfocus |
|---|---|---:|---:|---:|---:|---:|
| meniscus R = 20/25, 4 mm | 2048, 2.832 µm | 5.214 nm | **0.032 nm** | 161× | −0.45 % | +0.19 DOF |
| air-spaced doublet, 4 surfaces, 6 mm | 2048, 4.248 µm | 29.828 nm | **0.043 nm** | 693× | −3.02 % | +0.60 DOF |
| f/2 biconvex R = ±4.12 mm (EFL 4 mm), 2 mm | 4096, 0.708 µm | 402.638 nm | **1.871 nm** | 215× | **−33.17 %** | **+8.20 DOF** |

The wavefront result is unambiguous and excellent on all three.  **The f/2 row is the one open
finding of this verification**: the corrected exit wavefront is 215× better and the focal peak is
a third lower.  Both are true, the cause is measured, and the two-line remedy is demonstrated —
§5.1.

### 3.2 L12 — the slant coefficient

I re-derived the O(θ²) expansion myself, numerically and symbolically, at n1 = 1:

```
exact   n2 cos(θi−θt) − 1 = (n−1) − θ² (n−1)²/(2n)      0.515089110779 (series identical)
v5.25.0 n2 cos θt − cos θi = (n−1) + θ² (n−1)/(2n)      0.515089368324 (series identical)
errors  paraxial 8.756e-08   v5.25.0 2.575e-07   ratio 2.9414 = n/(n−1)
```

Against my oracle on a single N-BK7 face, four radii (piston-free rms exit OPD, nm):

| R [mm] | θ | paraxial | v5.25.0 | z-axis | ratio | **shipped wave model, slant** | shipped, paraxial |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 0.0400 | 2.9930 | 8.8036 | 0.0007 | 2.9414 | **0.0003** | 1.5644 |
| 50 | 0.0801 | 24.0216 | 70.6589 | 0.0234 | 2.9415 | **0.0088** | 12.5299 |
| 30 | 0.1002 | 35.2744 | 103.7600 | 0.0540 | 2.9415 | **0.0202** | 18.3718 |
| 20 | 0.1506 | 120.0667 | 353.1958 | 0.4179 | 2.9417 | **0.1562** | 62.4477 |

So the shipped screen implements equation (3).  As a pure coefficient it is **287× (R = 20 mm) to
4276× (R = 100 mm)** closer to the exact eikonal than the paraxial one, which confirms the audit's
"290× to 4000×"; through the whole wave model it is 400× to 5215× on the same four faces.  A new
test isolates the COEFFICIENT rather than the field:
on a one-surface prescription `angle(E_slant · conj(E_paraxial))` is exactly
`−k0 (c_slant − (n2−n1)) sag` with no unwrapping, and it matches the closed form to **5.5e-10
relative** at R = 35 mm (a radius neither the audit nor the WP used) where the paraxial screen
scores 1.0 and the v5.25.0 form 2.94.

**The f/5 hammer fixture, re-measured** (r2m at z = 49.163 mm against the 65 µm dual-oracle truth;
the v5.25.0 coefficient rebuilt through `form_error` on the paraxial path, as the test docstring
describes):

| dx [µm] | paraxial | v5.25.0 slant | z-axis slant | ratio z/par |
|---:|---:|---:|---:|---:|
| 12.00 | 5.53 | 10.67 | 5.79 | 1.047 |
| 6.00 | **25.26** | **50.31** | **26.76** | **1.059** |

Identical to the module docstring's table.  The report's uncomfortable admission is **confirmed**:
at dx = 6 µm the OLD form lands 14.7 µm from the 65 µm truth and the new one 38.2 µm, i.e. the old
form scored closer on this fixture by cancellation.  The changelog's migration note says exactly
that and names the right replacements; it is accurate.  (I could not afford the dx = 3 µm row —
N = 8192 costs ~7 GB in this function on a shared box — so the report's 40.55 / 76.70 / 43.06 is
unverified by me; the two samplings I could run bracket it and the ratio is converged to three
digits at both, which is what the test actually asserts.)

**Test bars.**  `test_h1_slant_moves_toward_the_oracle_without_overshooting` asserts
`1.0 ≤ ratio < 1.5`.  Measured 1.059; the two defects it brackets sit at <1 (pre-v5.25.0
cancellation) and 1.929 / 1.992 (v5.25.0, which I measured myself at both samplings).  Derived,
two-sided, with the fail-before values measured, not quoted.

### 3.3 L13 — Fresnel

Closed form `T = 4 n1 n2/(n1+n2)²` derived here from the amplitude coefficients and the axial
Poynting flux:

| interface | measured | closed form | rel |
|---|---:|---:|---:|
| AIR → N-BK7 | 0.958057134 | 0.958057134 | 1.1e-16 |
| AIR → N-SF11 | 0.921480312 | 0.921480312 | 4.4e-16 |
| AIR → **N-LASF9** (n = 1.85) | 0.911800225 | 0.911800225 | 4.4e-16 |
| N-BK7 → N-SF11 | 0.993599137 | 0.993599137 | 4.4e-16 |
| **N-SF11 → N-BK7** (reversed) | 0.993599137 | 0.993599137 | 4.4e-16 |
| AIR→BK7→SF11→AIR (three faces) | 0.877179908 | 0.877179908 | — |
| AIR → absorbing n = 1.8 + 1e-3 i | 0.918367230 | 0.918367230 | 2.2e-16 |

The oblique case is **not** claimed anywhere: the docstring states plainly that θi is the AOI of an
axial ray and that no path has both a true local AOI and Fresnel.  That matches the code.
`fresnel=False` is byte-identical to the default call.

### 3.4 L2 / L19 — tilt

* **Thin prism, both axes, both branches.**  `tilt=(θ,0)` deflects in x and `tilt=(0,θ)` in y, in
  BOTH branches, at −2.574742e-03 (field frame) and −2.574699e-03 (surface frame) against the exact
  thin-prism 2.575446e-03; branch-to-branch **4.3e-08 rad**.  The pre-fix surface-frame branch read
  0.000.
* **Exact rigid-body geometry, closed form** (a sphere rotated about its vertex is a sphere with
  the rotated centre — no series, no footprint approximation), R = 50 mm over ±2 mm:

  | case | max &#124;z_code − z_exact&#124; | waves |
  |---|---:|---:|
  | tilt = (1 mrad, 0) | 1.684e-09 | 0.0027 |
  | tilt = (5 mrad, 0) | 1.026e-08 | 0.0162 |
  | tilt = (20 mrad, 0) | 8.013e-08 | 0.1266 |
  | tilt = (0, 5 mrad) — the OTHER axis | 1.026e-08 | 0.0162 |
  | tilt = (3, −4) mrad — combined | 1.026e-08 | 0.0162 |
  | tilt = (5, 2) mrad **+ decenter (0.3, −0.2) mm** | 9.577e-09 | 0.0151 |
  | asphere R = 40, k = −1, A4, tilt = (5, 3) mrad | 1.836e-08 | 0.0290 |

  against 2.002 / 10.011 / 40.077 µm pre-fix.  The residual is the documented thin-element
  footprint simplification and scales as θ·(sag')², exactly as it should.
* **Convention.**  `raytrace/surface.py::_field_frame_sag_and_grad` applies `tx·(x−dx) + ty·(y−dy)`;
  `_lens_real`'s field-frame branch applies `t0·x + t1·y`; the surface-frame branch now takes
  `θx = t1, θy = −t0`, whose first order is `−θy·x + θx·y = t0·x + t1·y`.  All three agree, and I
  measured it end to end: the ray model gives (L, M) = (−2.575435e-03, 0) for `tilt=(5 mrad, 0)`
  and (0, −2.575435e-03) for `tilt=(0, 5 mrad)` — same axis, same sign as both wave branches.
  Under the orchestrator's ruling (documents align to the IMPLEMENTED convention) there is **no
  remaining disagreement between the wave and ray models**, so no defect.

  *Documentation note (P3).*  The `surface_frame` docstring still sells the branch as the "more
  accurate, Optiland/Zemax-style" treatment.  Measured against the exact rigid-body height it is
  now **marginally worse** than the default field-frame ramp on the same fixtures (1.026e-08 vs
  9.096e-09 m at 5 mrad; 8.013e-08 vs 5.345e-08 at 20 mrad).  Both are ≤ 0.13 waves, so the claim
  is no longer harmful — but it is no longer true either.  Listed as §5.2.

### 3.5 L3 / L4 / L5 / L6 / L7 / L8 / L9

* **L3, 1-D remap.**  A strongly aberrated input (3 waves ρ⁴ + 1.5 waves ρ⁶; 1.93 waves p-v over
  the scored annulus) is transported to within **1.7e-4 waves** of my ray-transport oracle
  (`h_out → h_in` from the traced fan, then the input residual evaluated at `h_in`).  Pre-fix the
  transported residual was identically zero.
* **L3, 2-D remap** (the default for a decentered element): flat vs aberrated separate by 0.89 of
  peak against 4.7e-16 pre-fix, with exactly one lattice-smoothing warning per call.
* **L3, conjugate.**  A converging `conjugate = −0.30` and the default `None` both focus at
  20.000 mm on a 0.625 mm scan grid against a ray-oracle crossing of 19.873 mm — consistent, but
  my scan step was 4× the depth of focus, so this arm is *not* a discriminating measurement and I
  do not claim it as one.  The phase-transport check above is the load-bearing L3 evidence.
* **L4.**  Exact plane waves built in **N-BK7 and N-SF11** at 50 mrad in-glass: `'auto'`, an
  explicit wavefront ndarray and a `TiltedCarrier` all return `q = n sinθ` to 4.4e-16, 1.1e-16 and
  0.0 relative.  The banded evaluators (`_screen_obliquity_rows_any`,
  `_screen_obliquity_row_evaluator`) are **byte-identical** to the corresponding row slice of the
  whole-grid field for every carrier where they apply, in both media.
* **L5.**  `_sag_callable_fingerprint` changes on a mutated attribute AND on a mutated closure
  cell; two lambdas with identical bytecode but different constants get different fingerprints;
  two lambdas on the same source line with the same VALUE share one (correct — it is a value
  fingerprint); an unprobeable callable returns `'unprobeable'` and makes the entry uncacheable.
  End to end with the cache enabled at 64 MB: after mutating the callable the warm-cache result is
  **byte-identical to a cold rebuild** (pre-fix: 2.04 of peak of stale-hit error on this fixture).
* **L6.**  Re-pointing `GLASS_REGISTRY['_VA2_G']` from n = 1.50 to 1.90 gives a warm result
  byte-identical to a cold rebuild, and **restoring the entry restores the original field byte for
  byte** — the round trip the finding is about.
* **L7.**  Immersed exit (N-SF11, n = 1.7912): the remap's exit OPL is 8e-13 m rms from my
  closed-form trace (1.3e-6 waves); the hard-coded n_exit = 1 defect is 2.08e-5 m = 33 waves here.
* **L8.**  `tangent_facet_remap` accepted on 15 (N, pad) combinations — N ∈ {256, **257**, **384**},
  pad ∈ {1, 1.5, 2, 3, 4} — plus a **top-hat** input (P/Pin = 0.999960) and an **anamorphic**
  dy = 2·dx.  No refusals, no NaN.
* **L9 lattice warning**: fires iff `2 r_max/(n_side−1) > 2·min(dx, dy)`, checked on five
  configurations (N = 128 / 181 / 256 / 512 at pad 1, and N = 256 at pad 4) — the predicate and the
  behaviour agree in all five.
* **L9 Newton**: disabling the early exit (monkeypatching `numpy.array_equal` to False, which forces
  the old fixed 24 sweeps) changes **no bit** of `_build_displaced_ray_map_2d`,
  `_build_displaced_ray_map`, `_build_displaced_cos_luts` or `_build_displaced_cos_grid`, on a
  spherical and a conic+aspheric singlet, at r_max ∈ {0.5, 1, 2, 3} mm and N ∈ {64, 128, **129**}.
  Pinned in `TestVerifyL9NewtonEarlyExitIsBitwise`.  (I looked for an N where the fixed point is
  NOT reached and did not find one; if there were, the loop would simply run its 24 sweeps, which
  is the pre-fix behaviour — the change cannot be wrong, only unhelpful.)
* **L19 absorption**, on two fixtures the WP did not use, against a Beer-Lambert oracle written
  here from the surface equation: a curved-**front** plano-convex (local path `t − sag₁(x)`, the
  mirror image of the WP's biconvex) reads 9.87e-6 max deviation with the local path against
  4.80e-5 for the axial-only factor (**4.9×**), and a cemented pair carrying **two different κ**
  (2e-4 and 5e-5, local path `t₁ + sag₂ − sag₁` then `t₂ + sag₃ − sag₂`) reads 1.14e-5 against
  1.12e-4 (**9.8×**).  The factorisation `κ_face = κ_before − κ_after`, with the first surface's
  incoming half and the last surface's outgoing half dropped, telescopes correctly across a
  multi-glass stack — which is the part a single-glass fixture cannot show.

### 3.6 L16 / L17 / E4 — byte identity and cost

* **L16**: the in-place conic chain is byte-identical to the b97c0b6e^ expression (written out
  verbatim in the new test) at **float64 and float32**, for R ∈ {50, −22.28, 12, 8, ∞} mm with
  k ∈ {0, −1, 0.5, −3}.
* **E4**: F-order, transposed, negative-stride-x, negative-stride-y and a stride-2 view all return
  the **byte-identical** C-ordered answer (max&#124;d&#124; = 0.0).  Pre-fix the whole polynomial vanished.
* **L17**: `_screen_exp` is byte-identical to `np.exp(-1j·k0·opd)` at float64 for nm-, µm- and
  mm-scale OPD (the 4.5e4 rad regime included), and correctly falls back to `xp.exp` at float32 —
  so the restriction the WP documents is real and active.
* **Cost, re-measured** (medians of 5 interleaved runs + tracemalloc, N = 2048, shared box):

  | item | before | after | gain |
  |---|---:|---:|---:|
  | `surface_sag_general` conic | 113.1 ms / 5.13 grids | **32.0 ms / 1.13 grids** | 3.54× / 4.5× |
  | screen build | 108.7 ms / 2.00 c128 | **46.2 ms / 1.50 c128** | 2.35× / 1.33× |
  | screen apply (`E *= ph`) | 26.4 ms | **9.2 ms** | 2.88× |

  The grid counts match the report exactly; the wall clocks differ (this box was carrying ten other
  agent processes) but every ratio is at or above the reported one.  **No test in either new file
  asserts a wall clock, a speed-up or a memory figure** — the flat-face early-out is pinned as an
  OPERATION COUNT (`_screen_exp` called once for a one-powered-face element), which is the correct
  S1-compliant shape.

---

## 4. Defects found — and fixed — in this pass

### 4.1 P1 (API): the two mirror guards disagreed about `allow_unfolded_equivalent`  *(the coordinator's item, WP-A9 §5.2)*

`_check_no_silent_fold_drop` documents `prescription['allow_unfolded_equivalent'] = True` as
escape hatch (a) for a folded design and honours it — but only for a mirror carried in
`prescription['elements']`.  A hand-built prescription with the mirror in `surfaces`
(`is_mirror=True` or `glass_after='MIRROR'`) hit a **second, unconditional** guard that never
mentioned the key, so the documented option did not work on that spelling of the same physics.
`lumenairy/ui/waveoptics_dock.py:949` documents the consequence in its own comment: the dock builds
the unfolded prescription itself and *then* sets the flag, "rather than catching the library's
refusal".

**Fixed** in `_lens_real.py` with one shared helper, `_unfold_mirror_surfaces`, called at the top of
`_apply_real_lens_impl` (before the model guards, so `_check_displaced_support`'s own mirror
refusal sees the substituted surfaces) and at the top of `prepare_real_lens`:

* **flag absent** → `ValueError` naming BOTH remedies, for both spellings (it previously named only
  `split_prescription_at_mirrors`);
* **flag present** → each mirror surface becomes an index-neutral FLAT at its own vertex plane
  (`radius=inf`, `glass_after := glass_before`, conic / asphere / biconic / freeform /
  `sag_callable` / `form_error` / decenter / tilt dropped, `clear_aperture` and `semi_diameter`
  kept), and the call warns naming what was dropped.  **Replacing rather than deleting** is what
  keeps the semantics honest: the surface count, every gap in `thicknesses` and both reference
  planes are unchanged, so a scalar on-axis field through a FLAT fold is exact — which is precisely
  what the flag's existing message says it accepts.
* `prepare_real_lens`'s `NotImplementedError` for a mirror surface is replaced by the same helper,
  so the two entry points now diagnose the key identically (the pattern L14 applied to
  `stop_index`).
* One paragraph added to `apply_real_lens`'s public `prescription` docstring — the key was
  documented only inside a private function.

**Verification.**  New class `TestVerifyMirrorGuardsReadTheSameKey` (6 tests):
refusal names the flag for both spellings; **the unfolded walk is byte-identical to the
hand-written unfolded prescription** (the strongest available statement of "the mirror became a
neutral flat and nothing else moved"); exactly one `RuntimeWarning`; a curved mirror's warning names
the dropped focusing phase; `prepare_real_lens` refuses the same way and its prepared lens is
byte-identical to `apply_real_lens` on the unfolded walk; and the `elements`-borne path is
**unchanged** (still refuses without the flag, still runs silently with it — no new warning).
`tests/unit/test_folded_design_guard.py` passes **9/9** unchanged.  Fail-before is structural: at
`b97c0b6e` the guard is `if _mirror_surf_idx: raise ValueError(...)` with no flag test, so arm 2
raised whatever the flag said.

*Behaviour change to record in the CHANGELOG:* a prescription with a mirror in `surfaces` AND
`allow_unfolded_equivalent=True` previously raised and now runs (with a warning).  Nothing that
worked before changes; the default path is untouched (a mirror-free prescription returns the same
dict object, no copy).

### 4.2 P2 (test hygiene): a wrong number in the replacement Seidel test's derivation

`tests/unit/test_audit_glass.py::…::test_seidel_improves_a_curved_rear_doublet_against_a_ray_oracle`
derived its 3× bar from "with the exit-vertex transfer … it lands at ~50 nm".  The shipped number
is **1.126 nm** (I measured 173.466 → 1.126 nm, 154×, against my own oracle; the WP's own summary
table says 1.126 too, so the docstring contradicted its own report).  A derivation comment with a
wrong measured value is the "right-conclusion-wrong-numbers" shape `TESTING_STANDARDS` calls the
most dangerous, so it is corrected in place with the re-measurement dated.  The same docstring said
"13 planes" where `_through_focus_peak` scans 21; corrected.  The bar itself (3×) is unchanged and
is now stated as 51× below the measured margin.

### 4.3 P2 (test fixture): a test pinned on a value another WP's in-flight change makes illegal

`tests/unit/test_audit_misc.py::TestAuditFixesV4_12_1_coverage_StopIndexWarn::
test_maslov_emits_warning_for_stop_index_2` **FAILED** when I ran the WP's own test batch:

```
lumenairy\elements\lenses_maslov.py:1730 → _lens_real.py:2464 in _normalise_stop_index
ValueError: apply_real_lens_maslov: prescription['stop_index']=2 is out of range …
```

This is **not** a WP-A2 regression at `b97c0b6e`: `git show HEAD:lumenairy/elements/lenses_maslov.py`
has no `_normalise_stop_index`; the call is an **uncommitted working-tree edit** implementing
WP-A2's own §5 request 3.  The fixture was the fragile part — it depended on which reading of the
key `lenses_maslov.py` happened to use, exactly as its sibling did before L14 re-fixtured it.
**Fixed** by moving it to the valid non-entrance `stop_index = 1` (measured: one `RuntimeWarning`
naming `stop_index=1` under BOTH readings of the key), renaming it
`test_maslov_emits_warning_for_a_mid_train_stop` to match the sibling, and adding the
"names the offending value" assertion the sibling has.  The out-of-range contract stays pinned for
`apply_real_lens` in the test above it.

---

## 5. Open items for the orchestrator

| # | Sev | Item |
|---|---|---|
| 5.1 | **P2** | **`seidel_correction` on a FAST element: the exit wavefront improves 215× and the focal peak drops 37 %** — because the ρ⁴+ρ⁶ fit is EXTRAPOLATED over the outer third of the pupil.  Demonstrated below: masking the extrapolated band turns −37 % into **+37 %**.  Two lines of code; I did not apply them because they re-bar the WP's own L1 numbers.  The one item I would fix before release. |
| 5.2 | P3 | **`surface_frame`'s "more accurate" claim** (§3.4).  Measured against the exact rigid-body height, the surface-frame branch is now marginally *worse* than the field-frame ramp (1.03e-08 vs 9.10e-09 m at 5 mrad).  Both are ≤ 0.13 waves, so this is a docstring correction, not a physics item. |
| 5.3 | P3 | **L9's sweep-count claim.**  The report and RL-MODELS say "24 → 2 sweeps (~10×)".  Measured on a spherical and a conic+aspheric singlet: **13** sweeps for both ray-map builders and 3 for the cos-LUTs, i.e. `_surface_sag_general` calls 150 → 84 (**1.79×**), not 10×.  The *property* the change rests on (bitwise identity) is verified; only the saving is overstated.  Suggest re-wording the changelog line. |
| 5.4 | P2 | **`lenses_maslov.py` + `test_audit_misc.py` must land together** (§4.3).  I have made the test correct under both readings, so either order is now safe — but the concurrent `lenses_maslov.py` edit should be committed with this test file, not after it. |
| 5.5 | P2 | **CONVENTIONS.md** — the WP's request 1 (state the `Σ|E|²·dx·dy` = power convention) is still open and L13's fix now depends on it.  I re-derived the same convention independently and agree with the wording proposed. |
| 5.6 | P3 | **`surface_sag_general(R=nan)`** returns an all-NaN grid behind anonymous numpy warnings — the same shape C1 fixed for `R = 0`, one branch over.  Pre-existing, not introduced here; one line in the same guard would close it. |
| 5.7 | P3 | `repro/RL-CORE/p7b.py` now aborts at `stop_index=2` (correctly).  Worth a two-line `try/except` in the repro so the remaining rows still print for future auditors. |
| 5.8 | — | **Not a defect, but the trap that cost me two false alarms** (§3.1): `apply_real_lens` exit-OPD measurements on an apertured fixture need `dx ≈ 1.45·aperture/2048`, not a carrier-Nyquist pitch.  At N = 512 a meniscus reads 558 nm of model-vs-wave error that is entirely the hard edge aliasing through the in-glass ASM.  Worth a sentence wherever the audit's measurement recipe is written down. |

Nothing in this list blocks the commit; 5.1 is the only one I would act on before a release.

### 5.1 in detail — the Seidel screen is EXTRAPOLATED over the outer pupil

On the f/2 fixture the corrected exit wavefront is 215× better inside
|x| ≤ 0.85·r_pupil and the focal peak is **a third lower** (−33 % on the 41-plane scan of §3.1,
−37 % on the 2-pass refinement below), with the best focus 8.2 depths of focus away.  Those two
statements are both correct, and the reason they can both be true is in the block itself:

```python
h_fan   = np.linspace(-0.9 * r_pupil, 0.9 * r_pupil, 41)   # the fan starts at 0.9
rho     = x_model / r_pupil                                 # ... and lands INSIDE that
...
rho_map_sq = h_sq_axis / (r_pupil ** 2)
corr_map   = sum(c * rho_map_sq ** (p // 2) ...)
corr_map   = xp.where(rho_map_sq <= 1.0, corr_map, 0.0)     # applied out to rho = 1.0
```

Measured landing radii of the 41-ray fan (the transverse walk through the element is INWARD, so a
fast element lands well inside the launch radius):

| fixture | fan launched to | model lands at | extrapolated band | share of pupil AREA | share of pupil ENERGY |
|---|---:|---:|---:|---:|---:|
| f/2 biconvex R = ±4.12 mm | 0.900 r_p | **0.8084 r_p** | 0.808 → 1.0 | **34.6 %** | **18.97 %** |
| cemented doublet 8 mm | 0.900 r_p | **0.783 r_p** | 0.783 → 1.0 | 38.7 % | — |

So a ρ⁴ + ρ⁶ polynomial fitted on ρ ≤ 0.81 is EXTRAPOLATED by 19 % in radius over a third of the
pupil area carrying a fifth of the energy, and the imprinted value there is whatever the polynomial
does.  Measured on the f/2, by reading the screen straight out of the field
(`angle(E_ON · conj(E_OFF)) / k0`): the screen is **+1926.6 nm at ρ = 0.81** and **+25.4 nm at
ρ = 1.00** — it swings back through three waves across the extrapolated band.

**Demonstrated, not inferred.**  Take that same measured screen, mask it to the fit domain
(ρ ≤ 0.8084) and re-apply it to the uncorrected field.  Through focus, N = 4096 / dx = 0.708 µm,
two-pass refinement:

| arm | best focus | peak &#124;E&#124;² | vs OFF |
|---|---:|---:|---:|
| (a) correction OFF | 3725.0 µm | 210 672 | — |
| (b) correction ON, as shipped | 3720.0 µm | 132 505 | **−37.1 %** |
| (c) the SAME correction, masked to the fitted radius | 3567.5 µm | **288 322** | **+36.9 %** |

Masking the extrapolated band turns a 37 % loss into a 37 % gain.  The fitted coefficients are not
the problem — the inner 65 % of the pupil area is corrected to 1.9 nm — the extrapolation is.  That
is also why this is invisible on the WP's fixtures (the 4 mm doublet's correction is ~0.2 waves,
peak −0.27 %) and severe on a fast one (the f/2's is ~1.5 waves rms).

**Requested fix (one or two lines):** clamp instead of extrapolating —

```python
rho_fit = float(np.max(np.abs(rho)))            # already in hand from the fit
corr_map = xp.where(rho_map_sq <= rho_fit ** 2, corr_map, <corr at rho_fit>)
```

so the screen is a constant piston beyond the largest radius any traced ray reached (a piston is
unobservable, and it avoids the phase STEP that arm (c) above deliberately accepted).  Launching
the fan to `0.999 * r_pupil` instead of `0.9 * r_pupil` helps too and is independently worth doing,
but it is only partial: the walk is inward, so even a full-aperture launch lands at ~0.87 r_p on
this element.

**I did not apply it**, deliberately.  Either change moves the field inside 0.85 r_pupil on a fast
element, which re-bars the WP's own L1 numbers (the 8 mm doublet's 1.126 nm, the 5.90× at 4 mm) and
the three tests derived from them — and re-deriving those needs an N = 8192 run (~7 GB) I could not
afford on this shared box.  That is the orchestrator's call, not a verifier's last-minute edit.

**Scored P2, not P1**, because: it is an opt-in flag whose default is off; it is strictly better
than the pre-fix behaviour on every fixture anyone has measured (pre-fix this same f/2 case carried
the ρ² term as well); and the gate skips entirely on the well-corrected elements the docstring now
steers callers toward.  But "the through-focus peak must not drop" was part of the decision rule
this work package was given, and on a fast singlet it drops 37 % — while two lines would make it
rise 37 %.  I would take those two lines before release.

---

## 6. Tests added / changed by VERIFY-A2

New: `tests/unit/test_audit2609_a2_verify_lens_analytic.py` — **43 tests**, 3.3 s, all green:

| class | n | what it pins |
|---|---:|---|
| `TestVerifyMirrorGuardsReadTheSameKey` | 6 | both arms of the defect fixed in §4.1 |
| `TestVerifyL13FresnelPowerOnNewInterfaces` | 5 | closed-form power transmittance on new glass pairs incl. a complex index; `fresnel=False` byte identity |
| `TestVerifyL12SlantCoefficientIsTheAxialTranslationIdentity` | 2 | the coefficient itself at R = 35 mm, and its SIGN |
| `TestVerifyL2TiltAgainstExactRigidBodyGeometry` | 6 | exact rotated sphere, both axes + combined + decenter; thin prism in both branches AND the ray model |
| `TestVerifyL4CarrierMomentumInAHighIndexMedium` | 3 | `q = n sinθ` in N-SF11 for all three carrier vocabularies |
| `TestVerifyL7ImmersedExitReferencingLeg` | 1 | remap exit leg vs a closed-form immersed trace |
| `TestVerifyL16E4SagKernelByteIdentity` | 15 | previous expression byte-for-byte; 5 non-C-contiguous layouts |
| `TestVerifyL9NewtonEarlyExitIsBitwise` | 4 | forcing 24 sweeps changes no bit, 3 builders × 2 prescriptions × 2 radii |
| `TestVerifyL1GateSkipsAnImmersedRearSinglet` | 1 | the gate's decision with n_exit ≠ 1, as a bit identity |

Changed (both files VERIFY-A2 owns): `tests/unit/test_audit_glass.py` (§4.2, docstring numbers
only — no bar moved), `tests/unit/test_audit_misc.py` (§4.3, fixture + name + one assertion).

---

## 7. Test runs

| command | result | duration |
|---|---|---|
| `test_audit2609_a2_verify_lens_analytic.py` (new) | **43 passed** | 3 s |
| `test_audit2609_a2_analytic_lens.py` + `test_v5_2_off_axis_conic_surface_frame.py` + `test_hammer_h1_slant_obliquity.py` | **57 passed, 1 skipped** (Optiland absent) | 63 s |
| `test_audit_glass.py` + `test_audit2609_a2_displaced_models.py` + `test_audit_misc.py::…StopIndexWarn` | 36 passed, **1 failed** → cause found (§4.3), fixed, re-run green | 331 s |
| `test_audit_glass.py` + `…StopIndexWarn` + `test_folded_design_guard.py` + `test_niche_audit_e_prepared_and_enums.py` (after my fixes) | **74 passed** | 100 s |
| `test_audit2609_a2_verify_*` + `test_audit2609_a2_analytic_lens` + `test_v5_2_off_axis_conic_surface_frame` + `test_folded_design_guard` + `test_elements_lens` | **123 passed, 1 skipped** | 4 s |
| `test_lens_chunked_sag` + `test_slant_chunk_byte_identical` + `test_obl_banded_halo` + `test_tf_banded_halo` + `test_screen_obliquity` + `test_sag_float32_production_window` + `test_audit_s4_3_waveoptics_biconic` + `test_niche_p3_pointwise_obliquity` (banded / obliquity / float32 collateral, AFTER my `_lens_real.py` change) | **289 passed, 2 skipped** (PySide6 absent) | 111 s |
| `test_niche_r1_cosgrid_cache.py` (the L5/L6 cache's own file) | **18 passed** | 49 s |
| `test_niche_p9_decenter_tilt.py` — the CROSS-MODEL tilt/decenter pin (analytic vs `raytrace` vs the lumenairy-free `geom_spot_decenter_oracle`, to 1e-12/1e-13) | **13 passed** | 319 s |
| `test_audit_misc.py::TestAuditFixesV4_12_1_coverage_StopIndexWarn` after §4.3 | **3 passed** | 3 s |
| `pytest tests/unit -k real_lens` (whole-suite selection) | **146 passed, 3 skipped**, 0 failed | 289 s |
| `python validation/run_all.py test_lenses` | 1 case fails: `apply_real_lens_traced_jax` — `RuntimeError: … requires double precision, but jax_enable_x64 is disabled`, i.e. §15.6's JAX-x64 policy in `_lens_jax.py` (another WP's file). **Every** `apply_real_lens` case passes, including `all optional features together` and the new `slant + seidel raise ValueError` | 30 s |
| `repro/RL-CORE/{p1b,p3,p5,p5c,p7b,p9,p10}` | as §2 | — |
| **FINAL consolidated sweep** over every file WP-A2 or VERIFY-A2 touched: `test_audit2609_a2_verify_lens_analytic` + `test_audit2609_a2_analytic_lens` + `test_audit2609_a2_displaced_models` + `test_audit_glass` + `test_v5_2_off_axis_conic_surface_frame` + `test_hammer_h1_slant_obliquity` + `test_audit_misc::…StopIndexWarn` + `test_folded_design_guard` + `test_elements_lens` | **163 passed, 1 skipped** (Optiland absent), 0 failed | 134 s |

**Collateral failures, both traced to other work packages' files:**

1. `test_audit_misc.py::…::test_maslov_emits_warning_for_stop_index_2` — traceback
   `lumenairy\elements\lenses_maslov.py:1730 → _lens_real.py:2464`; uncommitted working-tree edit to
   `lenses_maslov.py`.  Fixed on my side (§4.3).
2. `validation/elements/test_lenses.py::apply_real_lens_traced_jax` —
   `lumenairy/elements/_lens_jax.py`, the JAX-x64 policy.  Not mine, not WP-A2's; reported by the
   WP as pre-existing and still true.

`test_niche_audit_e_prepared_and_enums.py::test_l22_delegate_reports_the_discarded_physics_kwargs`,
which the WP reported failing from a concurrent `_lens_traced.py` edit, now **passes** (74/74 in
that file) — that agent has since landed its side.

---

## 8. Summary

Every finding WP-A2 claims fixed reproduces, on my own fixtures and my own oracles, at the numbers
its report states — including the two it was most exposed on: the Seidel model reference really is
the split-step model's own exit OPL (0.03–1.2 nm on five prescriptions, measured with no ray trace
in the loop), and the f/5 hammer admission that the OLD slant form scored closer on that one
fixture is true and correctly explained.  The two byte-identity claims (the in-place sag chain, the
cos/sin screen) hold bit for bit including at float32, and the 44-configuration banded/whole-grid
matrix is intact.  The new tests are derived, two-sided and build-free, with no wall-clock
assertion anywhere.

Adversarially, the pass produced two things the WP did not have.  One is a **new P2 on L1**: the
corrected Seidel screen is extrapolated over the outer third of the pupil area, which on an f/2
singlet costs 37 % of the focal peak while the wavefront it was scored on improves 215× — measured,
mechanism identified, and a two-line remedy demonstrated to turn the 37 % loss into a 37 % gain
(§5.1).  The other is one **API defect found and fixed** here, of exactly the class this audit is
about: a documented escape hatch (`allow_unfolded_equivalent`) that two guards in the same file
disagreed about, so it did not work — now one shared helper, both arms pinned, with the unfolded
walk asserted as a byte identity against a hand-written unfolded prescription.  Two test-hygiene
defects were also fixed (a wrong measured number inside a derivation comment, and a fixture that
broke on a sibling module's in-flight change).  Seven further items (§5.2–5.8) are recorded for the
orchestrator; none of them blocks the commit.
