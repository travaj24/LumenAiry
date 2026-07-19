# Audit — "ideal thin lens" through the ASM: the full story (RESOLVED 2026-07-18)

**Library:** lumenairy · **Files:** `elements/_lens_thin.py` (`apply_thin_lens`), `propagators/asm.py`, `propagators/fresnel.py` · **Date:** 2026-07-18 (superseding the earlier draft of this file)
**Trigger:** design-121 thin-lens 1x1 (exp18/19) imaged a single 4 um-waist emitter to ~9 um at the MSoP where Zemax POP gives a 2.74 um waist. A colleague's independent Zemax POP of the paraxially-equivalent 4f system (f1=60.916, f2=41.666, waist 2.7378 um, clean to 1e-4 of peak) forced a full cross-oracle resolution.

## Verdict (one paragraph)

The entire discrepancy — the 9 um nominal-plane spot, the 3.6 um "core", and the ~90% "pedestal" — is **fictitious spherical aberration injected by the thin-lens PHASE MODEL, not by the propagator, the grid, the apertures, or any fundamental FFT limitation**. A paraxial lens (`exp(-ik r^2/2f)`) is only self-consistent under paraxial (Fresnel) propagation; under the exact ASM kernel it leaves an uncompensated r^4 wavefront error at every lens that accumulates to **+17.8 rad** (4f) / **+11.9 rad** (121 six-group chain) at the 1/e^2 ray. The `'nonparaxial'` model over-corrects at finite conjugates (**-9.4 rad** on the 121 chain), which is why exp19 was no better. With **conjugate-matched stigmatic lens phases**, the SAME exact-ASM chain, same grid, same apertures, images the 121 chain perfectly: **w=3.09 um at the nominal MSoP plane (analytic prediction 3.08 um), EE(3um)=84%, EE(6um)=99.9%, transmission 99.98% — zero pedestal**. The earlier claim in this file that a "~90%-energy pedestal is a deeper multi-element high-NA FFT-ASM/BPM limitation" is **WITHDRAWN — it was wrong**.

## Cross-oracle evidence (colleague's 4f: 4 um waist -> f1=60.916 -> f2=41.666, theory image waist 2.736 um, EE3=91.0%)

| method | lens model x propagator | image w (um) | EE(r<3um) | EE(r<6um) | verdict |
|---|---|---|---|---|---|
| ABCD Gaussian theory | — | 2.736 | 91.0% | 99.99% | reference |
| **Zemax POP (ZOS-API, this box)** | paraxial x pilot-beam POP | **2.7360** | 91.1% | 99.99% | matches colleague to 4 digits |
| colleague's GUI POP | paraxial x pilot-beam POP | 2.7378 | (clean to 1e-4) | — | **VALID** |
| lumenairy Fresnel-TF + `paraxial` | matched paraxial | 2.746 | 90.9% | 100.0% | correct |
| lumenairy exact ASM + `nonparaxial` | matched (infinite conjugates) | 2.784 | 90.7% | 100.0% | correct |
| lumenairy exact ASM + `paraxial` (= exp18 config) | MISMATCHED | 9.13 nominal / 5.03 best focus (dz=-200um) | 7.8% / 24.8% | 23% / 46% | reproduces exp18's 9 um signature |
| LightPipes 2.1.5 `Fresnel`+`Lens` | matched paraxial | (1.8um px) | 80.6% (pixelated) | 99.84% | clean, confirms |
| LightPipes 2.1.5 `Forvard`+`Lens` | ALSO matched paraxial | (1.8um px) | 93.5% (pixelated) | 100.0% | clean; NB source inspection: Forvard's kernel is exp(-i pi lam z f^2) — PARAXIAL TF despite the "spectral method" name; LightPipes has NO exact-ASM propagator, so it cannot exhibit the mismatch |
| hand-written textbook exact ASM (earlier audit) | paraxial x exact ASM | == lumenairy to 4 digits | | | independent confirmation of the MISMATCHED side |
| poppy | (dropped) | 5.0 | — | — | poppy applies the lens quadratic on-grid at far-field pixel scale (73 rad/px aliased); not a valid oracle for this system |

All grids N=16384, dx=1.8 um unless noted; EE denominators are TOTAL launched power reaching the image plane.

## The 121 six-group thin-lens chain (exp18/19 geometry, incl. real apertures), N=16384

| lens model x propagator | nominal-plane w (um) | best focus | EE3 / EE6 at best | pedestal? |
|---|---|---|---|---|
| `paraxial` x exact ASM (= exp18) | 7.84 (exp18 @ full N: 9.0) | dz=-150um: 4.63 | 31.8% / 53.3% | huge halo |
| `nonparaxial` x exact ASM (= exp19) | 13.83 (exp19 @ full N: 12.6) | dz=-250um: 4.79 | 32.4% / 56.7% | huge halo (over-corrected) |
| `paraxial` x Fresnel-TF (matched) | 3.077 | dz=0: 3.077 | 84.8% / 99.9% | none (transmission 99.99%) |
| **conjugate-matched stigmatic x exact ASM** | **3.091** | dz=0: 3.091 | **84.4% / 99.9%** | **NONE (transmission 99.98%)** |

Analytic prediction at the nominal plane: w=3.080 um (the paraxial waist 2.737 um sits 9.3 um behind the logged plane, ~0.5 z_R). The stigmatic run confirms the chain's grid (N=16384 here; 28672/0.9 um in production), the Matsushima band-limit, the group apertures (2.1-3.4x the beam radius, 0.02% loss), and 20 ASM legs including negative reference-plane shifts are ALL fine.

## The math (why each model does what it does)

Exact spherical-wave phase with signed curvature R (R>0 diverging): `S(R) = sign(R)*(sqrt(r^2+R^2)-|R|) ~ r^2/2R - r^4/8R^3`.
A stigmatic element mapping incoming curvature R_in to outgoing R_out must apply `phi = k*(S(R_out) - S(R_in))`; its quadratic part is exactly `-k r^2/2f` (lens equation), its quartic part is `-k r^4/8 * (1/R_out^3 - 1/R_in^3)`.

- **`paraxial`** applies only the quadratic -> residual **W_par = k r^4/8 * (1/R_in^3 - 1/R_out^3)**. Only zero when |R_in|,|R_out| -> inf (both conjugates far); NEVER zero for an imaging leg.
- **`nonparaxial`** applies `-k(sqrt(r^2+f^2)-f)`, i.e. quartic `+k r^4/8f^3` -> residual **W_np = k r^4/8 * (1/f^3 - 1/R_in^3 + 1/R_out^3)**. Zero ONLY at infinite conjugates (collimating from front focus, or focusing collimated light). At finite conjugates the cross terms `3/(a^2 b)+3/(a b^2)` are missing -> it over-corrects.
- Fresnel propagation truncates the kernel at the SAME quadratic order the paraxial lens does -> the pair is self-consistent (aberration-free by construction). This is exactly what Zemax POP's pilot-beam re-referencing does, which is why POP + paraxial surfaces = ideal imaging.

Per-lens accumulation at the 1/e^2 ray (from the Gaussian q-trace of the logged exp19 chain):

| group | f (mm) | w at lens (mm) | R_in -> R_out (mm) | W_par (rad) | W_np (rad) |
|---|---|---|---|---|---|
| S3-S4 | +88.42 | 5.75 | +55.1 -> +146.4 | +3.70 | -2.75 |
| S5-S7 | +161.84 | 6.35 | +161.8 -> inf | +0.23 | ~0 |
| S18-S20 | +279.31 | 6.35 | inf -> -279.5 | +0.05 | ~0 |
| S21-S22 | +91.70 | 5.25 | -231.2 -> -65.7 | +1.58 | -0.98 |
| S23-S24 | +59.65 | 4.01 | -50.1 -> -27.2 | +6.43 | -5.71 |
| S25-S27 | -39.47 | 0.20 | +1.4 -> +1.3 | -0.04 | +0.04 |
| **total** | | | | **+11.9** | **-9.4** |

(4f equivalent: +17.8 rad paraxial, 0.0 rad nonparaxial — hence the 4f discriminates the two cleanly.)
Marechal criterion ~0.5 rad: both exp18 and exp19 ran with ~20x that. At 2w the residuals are 16x larger (r^4), which is what shreds the wings into the observed halo ("pedestal") and shifts best focus (-0.2 mm at N=16384; -0.36/-0.48 mm at production N).

## What was checked and exonerated

- **ASM itself**: exact and correct (earlier head-to-head vs a hand-rolled ASM: identical).
- **Matsushima band-limit**: mask NAs on the long legs are 0.24-0.65; the beam needs <=0.152 -> never clips (on/off test identical).
- **Grid/sampling**: z_crit = N dx^2/lambda = 17.7 mm < legs, but that is precisely what the band-limit mask handles; chirp at the beam is <=2.05 rad/px (Nyquist pi); stigmatic run proves sampling adequacy end-to-end.
- **Apertures**: 2.14-3.39x beam 1/e^2 radius -> 0.02% total loss. Not the pedestal.
- **The 121 prescription/parse**: per-group EFLs match Zemax to 0.00% (6/6); overall ABCD A=-0.684, B~0.
- **Beam NA**: the max anywhere in the single-emitter 121 chain is **0.152** (image-side cone; 0.147 into the S23-S24 leg). The "F/1.6" figure in earlier notes is an APERTURE ratio — the beam never fills those apertures (it runs at f/26 to f/44 through the relay). **Design 121 is NOT a high-NA problem for a single emitter.**

## Two real `apply_thin_lens` bugs found on the way (mini-test: collimated w=3 mm, f=+/-30 mm, NA~0.1, theory focus w=4.17 um)

1. **`nonparaxial` with f<0 has the wrong sign**: `exp(1j*k*(f - sqrt(f^2+r^2)))` expands to `-k r^2/(2|f|)` for f<0 — a CONVERGING quadratic. Measured: `f=-30 mm` focuses at z=+30 mm with peak 5.003e5, IDENTICAL to f=+30 (paraxial f=-30 correctly diverges, peak 0.28). exp19's last group (f=-39.47) hit this bug; it happens to be nearly harmless there only because the incoming beam curvature (R=1.4 mm) dwarfs the lens power. Fix: `phi = -sign(f)*k*(sqrt(r^2+f^2)-|f|)`.
2. **`aplanatic` r^4 term has the wrong sign**: `-k*f*(1-sqrt(1-r^2/f^2)) ~ -k r^2/2f - k r^4/8f^3`; a converging sphere needs `+k r^4/8f^3`. Measured at NA~0.1: paraxial 7.7 um, nonparaxial 4.27 um (correct), **aplanatic 9.1 um — WORSE than paraxial** (its quartic error is 2x paraxial's, same sign). The model as implemented aggravates exactly the aberration it is expected to remove.

## Changes to make (ranked, definitive)

1. **Add `lens_model='stigmatic'`** (or `conjugates=(R_in, R_out)` / `(s, s_prime)` parameter): `phi = k*(S(R_out) - S(R_in))` with `S(R)=sign(R)*(sqrt(r^2+R^2)-|R|)`, `1/R_out = 1/R_in - 1/f`. This is the exact ideal element under the exact ASM at ANY conjugates. Proven above (121 chain: EE6=99.9%, no pedestal). R_in is available analytically in any ABCD-decomposed chain (q-trace), or from the design conjugates.
2. **Fix bug (1)** f<0 nonparaxial sign; **fix bug (2)** aplanatic quartic sign (and document what 'aplanatic' is supposed to mean — the current formula is neither the sine-condition profile nor a stigmatic phase).
3. **For paraxial-thin-lens chains, offer a matched paraxial propagator**: a same-grid Fresnel transfer-function step (`H = exp(-i pi lambda z f^2)`, works for z<0; `fresnel_propagate` is single-FFT and grid-changing so it cannot serve chains). Matched-paraxial reproduces Zemax POP/design-intent exactly and is the right "ideal reference" mode for relay studies.
4. **No change to `angular_spectrum_propagate`** — again exonerated, now including 20-leg chains with negative legs.
5. **Audit `apply_real_lens` separately**: the analytic (real-surface phase-screen) 8x8 runs show 6.3 um FWHM vs Zemax real-prescription POP 3.22 um — same 2x smell. Real surfaces + exact ASM SHOULD be near-ideal (the real design is corrected for real propagation), so the suspect is the per-surface thin-screen/sag-projection approximation (and/or vertex-plane conventions), NOT an "FFT pedestal". Until audited, treat analytic-chain absolute spot sizes/EE as suspect; crosstalk ratios are pessimistic (spots 2x too wide overlap neighbors 2x more).

## Repro

Scratchpad (session 2026-07-18): `phase0_abcd_na.py` (q-trace + residual table), `wave_matrix.py` (4f + 121 chain x 3 models), `wave_stigmatic.py` (the proof), `lens_model_minitest.py` (the two bugs), `zos_4f_v2.py` (ZOS-API POP 2.7360 um), `lp_oracle.py` / `poppy_oracle2.py`. Earlier single-lens isolation: `asm_headtohead.out`, `fresnel2.out`, `exactlens_fix.out`, `inputdx_scan.out`, `extent_scan.out`, `mft_fix.out`.
