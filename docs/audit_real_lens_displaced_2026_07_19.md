# H2(a) -- ray-angle-aware `surface_model='displaced'` for `apply_real_lens` (2026-07-19)

**Scope:** the 2026-07-19 hammer campaign's open item H2 -- a higher-order
per-surface correction for the analytic split-step real-lens propagator
(`apply_real_lens`), targeting the model's plateau on aberrated singlets
(converged r2m ~40-50 um vs the dual-oracle 65 um on the f/5 biconvex) and its
structural inability to distinguish plano-convex orientation (60.4/60.9 um vs
the oracle 43.2/127.6 um).

**Oracles (both from the hammer campaign, fully independent):**
1. **Zemax OpticStudio POP** via ZOS-API (dispersionless model glass).
2. **Grid-free Debye/Huygens integral** (`debye_oracle.py`) -- exact meridional
   raytrace + ring-Huygens sum, no FFT/grid/lumenairy.
3. **`apply_real_lens_traced`** -- lumenairy's independent per-pixel ray-traced
   OPL propagator (a different implementation from the phase-screen family),
   used as an in-library cross-oracle.

## Result

A **parameter-free** opt-in mode `surface_model='displaced'` that replaces the
paraxial thin-element phase screen `(n2 - n1)*sag(r)` with the eikonal-correct
ray-angle-aware piston OPD

```
OPD_i(r) = (n2*cos(alpha_out) - n1*cos(alpha_in)) * sag_i(r)          (1)
```

where `alpha_in`/`alpha_out` are the TRUE ray angles to the z-axis (before /
after each surface) sourced from a self-contained COLLIMATED meridional ray fan
traced through the actual conic/aspheric prescription (geometric optics,
wave-model-independent). At normal incidence `alpha_in=alpha_out=0` and (1)
reduces to the paraxial screen, so a benign near-axial beam is unchanged.

### Converged dual-oracle validation (N=8192, dx=3 um -- Nyquist-compliant)

| case | thin r2m | **displaced r2m** | traced r2m | Debye/POP | disp err |
|---|---|---|---|---|---|
| f/5 biconvex | 25.3 (dx6) | **64.5** | 64.8 | 64.98 | **0.7%** |
| plano-convex good | 60.4 | **42.2** | 43.1 | 43.2 | **2.2%** |
| plano-convex bad | 60.9 | **127.0** | 124.3 | 127.6 | **0.5%** |

Orientation split ratio (good/bad) **0.333 vs oracle 0.339**. The EE profile
also matches (f/5 displaced EE50/EE80 = 15.1/55.2 um vs Debye 15.17/55.22).
Runtime is **1.25x** the thin path (a meridional fan trace + two grid
interpolations per surface); well within the 3x budget.

## Mechanism -- and a refutation of the audit's attributed cause

The 2026-07-18 audit attributed the analytic plateau to the **transverse
ray-displacement / plane-projection error** (the refracted ray exits at a
different transverse position than the straight-through screen assumes). This
campaign **refutes that as the dominant mechanism.** Two things were actually
going on:

1. **Missing incoming-ray-angle obliquity (the real fix).** The paraxial screen
   assumes every ray strikes each surface parallel to the axis. On a
   plano-convex singlet this makes the imprinted OPD **orientation-invariant**:
   `sag_{-R}(r) = -sag_{+R}(r)`, so `(n_glass-n_air)*sag_{+R}` (curved-first) and
   `(n_air-n_glass)*sag_{-R}` (curved-second) are the identical map -- which is
   exactly why the thin model gives 60.4/60.9 for both orientations. The cosine
   factors in (1) carry the true incidence angle (the second surface sees a
   converging beam; the air-side and glass-side bends differ), which **breaks
   that symmetry** and reproduces the textbook ~4x spherical-aberration split.
   This is the entire correction -- no displacement/walk-off term is needed.

2. **A sampling artefact masquerading as a model floor.** The "converged 40.5 um
   plateau" the audit reported was measured at **dx=6 um, which is below the
   exit-NA Nyquist limit** (NA_exit~0.24 -> dx <= lambda/(2 NA_exit) ~ 2.7 um).
   At dx=6 um the beyond-Nyquist annulus of the converging exit wavefront
   aliases and the r2m reads LOW -- **`apply_real_lens_traced` itself reads
   40.9 um at dx=6 um and 64.8 um at dx<=3 um** (this is finding H3 for the
   analytic family). The correct-physics `displaced` model likewise reads
   40.9 um at dx=6 um and converges to 64.5 um at dx<=3 um. So the plateau was
   partly a sampling floor, not solely a model floor.

### The walk-off term was a red herring (documented dead end)

An explicit transverse-walk-off screen (candidate (a) literally: an extra
`~ -(s_out - s_in)*s_out*sag*n` phase encoding the sag-plane refraction offset)
appears to "fix" the dx=6 um r2m to 64 um -- but that is **compensating one
aliasing artefact with a second error**: at Nyquist-compliant sampling it
**over-corrects to 92 um** (EE50 24.6 vs the truth 15.2). It was rejected. The
obliquity OPD (1) alone converges to the truth with zero free parameters.

## API and envelope

`apply_real_lens(..., surface_model='displaced')`. Default `'thin'` is
**byte-for-byte identical** to prior releases (pinned by test).

**Requires** (else raises): rotationally-symmetric plain conic/aspheric
surfaces (no biconic `radius_y` / freeform / decenter / tilt / form_error /
mirror), the ASM in-glass propagator, the NumPy backend, and no other
per-surface OPD/amplitude modifier (`slant_correction` / `fresnel` /
`seidel_correction` / `absorption` / `surface_frame` / `use_gpu`). **Assumes a
collimated input** for the incidence-angle fan.

**Sampling:** the exit converging wavefront must satisfy
`dx <= lambda/(2 NA_exit)` or the windowed r2m aliases low (same rule as
traced, H3). For strongly non-collimated input, or outside the conic/aspheric
rotationally-symmetric envelope, use `apply_real_lens_traced` (validated to
99.7% of the dual-oracle) or `apply_real_lens_gbd`.

## Tests

`tests/unit/test_hammer_h2_displaced_projection.py`:
- default path byte-identical to `'thin'`;
- thin cannot split plano-convex orientation (fail-before anchor);
- displaced splits orientation AND matches the independent traced propagator to
  <6% on both orientations (fast, reduced-NA Nyquist-ok config);
- benign regime unregressed (29.98 um);
- unsupported combinations raise;
- **slow** (N=8192/dx=3 um): converged f/5 r2m in [58,72] with EE50/EE80
  matched, and the plano-convex split (42/127, ratio ~0.33) -- both against the
  stored dual-oracle numbers with provenance.
