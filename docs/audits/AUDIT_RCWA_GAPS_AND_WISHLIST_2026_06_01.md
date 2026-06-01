# RCWA Module — Gaps & Wishlist (Anisotropic LC / Metal-Grating Use Case)

**Date:** 2026-06-01
**Author:** Claude Opus 4.8 (1M context)
**Type:** Gap analysis + feature wishlist (NOT a bug audit — the 1-D anisotropic core is correct and validated)
**Scope:** Capabilities of `lumenairy/elements/rcwa.py` (v5.6.1) that are **missing, partial, or worth adding**, as surfaced by an external device study: a **1-D lamellar grating of lossy-metal teeth with a rotated-uniaxial liquid-crystal tensor filling the gaps, on a metal back-reflector, used as a tunable reflective Jones element** (PBS → QWP@45° → grating → QWP@45° → PBS out-coupler). The device is currently modelled in a separate `nannos`-based code (GPLv3); this audit captures what would be needed to migrate it onto LumenAiry's native (MIT) RCWA.
**Module state read:** `rcwa.py` at v5.6.1 (3381 lines), plus `examples/13_rcwa_inverse_design.py` and the unit tests. Capability claims are grounded in source (`file:line`); line numbers are from a structural read and may drift ±a few lines.
**Relation to prior audits:** Complements `AUDIT_RCWA_INTEGRATION_v5_5_0_2026_05_31.md` (correctness) and `AUDIT_RCWA_CONVERGENCE_ACCELERATION_2026_06_01.md` (harmonic-convergence techniques). This audit deliberately does **not** re-list the convergence-acceleration roadmap (FFF-2D, ASR, circular truncation, symmetry halving, eig-reuse, extrapolation); it covers *functional/feature* gaps for anisotropic-grating device modelling, referencing those items only where they intersect this use case.

---

## Part 0 — TL;DR

The 1-D anisotropic path already covers the **core** of this use case and does so correctly: `rcwa_jones_1d` carries a full in-plane (3,3) tensor in ridge and groove, `_tensor_convolutions` applies the genuine **Li-1996 anisotropic-1D factorization** (inverse rule on the wall-normal component + Schur-complement off-diagonal coupling), `uniaxial_tensor` builds the rotated-LC tensor, complex zeroth-order **Jones reflection** (amplitude + phase) is returned, and lossy metals + metal back-reflectors are first-class. **Four of the five physics requirements are solid** (anisotropic 1-D tensor, correct factorization, complex Jones, lossy metals).

The **two highest-value gaps** for this class of device:

1. **Internal-layer field reconstruction.** Field reconstruction today is **far-field only** (plane-wave superposition in the half-spaces above/below the stack); there is **no in-structure E/H map**. For plasmonic / gap-mode devices, the field *inside* the patterned layer is the whole physical story (where the mode lives, how the LC tunes it). H-fields are also not produced.
2. **Multi-region & stacked 1-D anisotropic gratings.** `rcwa_jones_1d` is **binary** (2 regions). Real devices have ≥3 regions per period (e.g. *grounded tooth | LC gap | floating tooth | LC gap* = 4 regions) and **vertically-stacked patterned layers** (anisotropic LC/metal over an isotropic spacer/metal). The pieces exist (`RCWAStack`) but the convenient analytic 1-D Jones path does not span them.

Plus a **convergence-diagnostics gap** (no surfaced "is this converged?" guard — under-resolved sharp resonances silently mislead; we hit this directly), and several quality-of-life wishes (dispersive Jones sweep, tapered-sidewall helper, layer/region-resolved absorption, batched/vmap geometry solve, reflective-Jones device helpers).

---

## Part 1 — Capability vs this use case (verified baseline)

| Requirement (driving device) | Status | Location / evidence |
|---|---|---|
| 1-D binary grating, full in-plane (3,3) tensor incl. ε_xy | ✅ | `rcwa_jones_1d` (~:2029); profiles `xx,xy,yx,yy,zz` built & convolved (~:2128-2132) |
| Anisotropic-1-D Fourier factorization (Li 1996) | ✅ correct, energy-validated | `_tensor_convolutions` (~:1949-1974); Cyx-sign energy guard noted in source |
| Rotated-uniaxial / LC director tensor | ✅ | `uniaxial_tensor(n_o,n_e,theta,phi)` (~:1897-1931) |
| Complex zeroth-order **Jones reflection** (amp+phase) | ✅ | `rcwa_jones_1d` returns (2,2) complex (~:2074-2077); phase preserved to public `exp(-iωt)` |
| Lossy metals + metal back-reflector | ✅ | auto Li-rule for metallic `n`; substrate or uniform metal layer; absorptance verified positive |
| **Multi-region (>2) 1-D anisotropic profile** (e.g. 4-region alternating teeth) | ⚠️ partial | `rcwa_jones_1d` is **binary only**; multi-region needs `RCWAStack` with a discretized 1-D tensor cell — **verify this path + note it re-introduces in-plane pixelation/staircase** (loses the analytic form-factor advantage) |
| **Stacked 1-D patterned layers** (anisotropic over isotropic) | ⚠️ verify | `RCWAStack` (~:3073) + `add_layer(eps_tensor_cell=...)` (~:3121) — confirm the 1-D anisotropic multilayer path is exercised/tested |
| **Internal-layer E/H field maps** (inside the structure) | ❌ | reconstruction is far-field only: `to_jones_field` (~:2880), `to_multiorder_field` (~:2967), `per_order_amplitudes` (~:2791); no per-z in-layer back-substitution; no H-field |
| Tapered / slanted sidewalls (real fab) | ❌ | binary/vertical only; user must hand-slice a staircase |
| Sharp-resonance / convergence self-check surfaced in API | ❌ | `rcwa_extrapolate` (~:1458) exists but is not wired into a "converged?" guard or auto-`n_orders` |
| Full 3×3 tensor with out-of-plane coupling (ε_xz, ε_yz) | ❌ | z-decoupled subset only (~:1889-1894) — fine for an **in-plane** LC director (θ≈π/2), not for out-of-plane tilt |
| Dispersive **Jones** wavelength sweep w/ DB indices | ❌ | `rcwa_efficiency_vs_wavelength` (~:1383) is dispersionless and **scalar-efficiency only** (~:1413-1416); no Jones, no auto index lookup |

**Bottom line:** the device's *static* response at a single wavelength is fully expressible today (via `rcwa_jones_1d` for a binary idealization, or `RCWAStack` for the real multi-region/stacked geometry once that path is confirmed). The gaps are in **(i) seeing the field inside the structure**, **(ii) convenient multi-region/stacked 1-D anisotropic construction**, and **(iii) spectral/convergence/fab-realism ergonomics**.

---

## Part 2 — Gaps (prioritized)

### GAP 1 — Internal-layer E/H field reconstruction  ★ highest physical value
**What:** Reconstruct the real-space **E and H field inside the structured layer(s)** (and across the full z-stack), not just the far-field plane-wave superposition in the half-spaces.
**Why:** For plasmonic / gap-mode / guided-mode devices the physics *is* the internal field — where the mode concentrates (e.g. a gap-plasmon living in the LC between metal teeth), how strongly the tunable medium overlaps it, and how loss is distributed. It is also the basis for **field-based inverse-design merit terms** (GAP/Wish: maximize |E|² in a target region). Today this is impossible in LumenAiry; it forced this study to retain a separate engine purely for field maps.
**Where:** The retained per-layer **modal amplitudes already exist** inside the S-matrix recursion; what's missing is the public per-z back-substitution `E(x,z), H(x,z)` within a layer (and H anywhere). `to_*_field` (~:2880, :2967) only superpose far-field orders.
**Suggested approach:** Expose a `RCWAResult.internal_field(z, component='E'|'H'|'all', nx=, layer=)` that evaluates the layer eigenmode expansion at depth `z` (standard FMM field recovery: amplitudes × eigenvectors × `exp(±iλz)`, then inverse-DFT to real space). Pair with the **Lanczos-σ filter already shipped** (~:3049) to tame Gibbs in the reconstructed field. H follows from the same modal coefficients via the curl operator already implicit in the eigenproblem.

### GAP 2 — Multi-region & stacked 1-D anisotropic gratings  ★ highest device-breadth value
**What:** (a) A 1-D anisotropic grating with **>2 regions per period** (arbitrary piecewise segments, each a (3,3) tensor); (b) **stacked 1-D patterned layers** (e.g. anisotropic LC/metal layer over an isotropic SiCN/metal layer) with a metal substrate — as a first-class, ideally analytic (non-pixelated) construction.
**Why:** The convenient `rcwa_jones_1d` is binary. The motivating device is **4 regions per period** (alternating grounded/floating metal teeth separated by LC gaps) and **two stacked patterned layers**. Forcing this through a discretized `RCWAStack` tensor cell reintroduces the in-plane staircasing that the analytic form-factor path (2-D) was built to avoid.
**Where:** `RCWAStack` (~:3073) appears to support stacked isotropic/anisotropic 1-D/2-D patterned layers via `add_layer(eps_tensor_cell=...)` — **needs verification + a regression test for the 1-D anisotropic multilayer case** (the current anisotropic tests are single-layer). The binary limit is in `rcwa_jones_1d`'s ridge/groove signature.
**Suggested approach:** A piecewise-segment 1-D anisotropic builder — `rcwa_jones_1d_segments(period, segments=[(width, eps3x3), ...], ...)` — using **analytic rectangular segment form-factors** (each region's Fourier coefficients are closed-form `sinc`), composed with the existing `_tensor_convolutions` Li rule. This avoids pixelation and naturally spans 2, 4, or N regions. Confirm `RCWAStack` then composes such layers vertically (aniso-over-iso) with the metal substrate.

### GAP 3 — Convergence self-check / resonance guard  ★ low effort, high safety
**What:** A surfaced "is this converged?" diagnostic and/or auto-`n_orders` guidance, especially for **sharp/high-Q resonances** where a too-low harmonic count produces a *plausible but wrong* answer (e.g. a spurious deep reflection null).
**Why:** Under-resolved sharp resonances silently mislead — in this study, a coarse harmonic count manufactured a deep null that vanished at proper truncation, and a ranking built on it was invalid. The module already ships `rcwa_extrapolate` (Richardson/Shanks, ~:1458) but it is opt-in and not wired into a guard.
**Suggested approach:** An optional `converged=True` path that solves at `n_orders` and a second higher count, reports the delta in the reported quantity (R, null depth, Jones entries), and warns (with the function-name-prefixed message style, CONVENTIONS §2) when the change exceeds a tolerance — or returns an extrapolated value + uncertainty. Cheap (2-3 solves) and would have prevented a real error here. Optionally: heuristic `n_orders` suggestion from period/λ/index-contrast.

### GAP 4 — Tapered / slanted-sidewall profiles  ★ fab realism
**What:** Built-in support for **trapezoidal / slanted sidewalls** (top width ≠ bottom width), the common real-fab deviation from vertical walls.
**Why:** Sidewall taper materially changes device behavior (in this study a mere 2° taper collapsed the tunable null because it widened the gap toward the bottom). Users currently hand-slice a z-staircase of binary layers — error-prone and convergence-sensitive at the staircase corners.
**Suggested approach:** A `sidewall_deg` (or per-region `width(z)`) parameter on the 1-D builders that auto-generates the z-sublayer staircase (with a documented `n_slice` convergence knob), and/or — once **ASR / matched coordinates** land (see convergence-acceleration audit) — a matched-coordinate slanted-wall treatment that avoids the staircase entirely.

### GAP 5 — Dispersive Jones wavelength sweep  ★ ergonomics
**What:** A wavelength sweep that (a) returns the **Jones matrix** (not just scalar efficiency) and (b) accepts **dispersive** indices/tensors, ideally auto-pulling `n(λ)` from the bundled `refractiveindex` integration.
**Why:** Device spectral response (out-coupling vs λ, retardance dispersion) is a primary deliverable; today one must hand-loop `rcwa_jones_1d` per wavelength with manually-supplied indices because `rcwa_efficiency_vs_wavelength` (~:1383) is dispersionless and scalar-only (~:1413-1416).
**Suggested approach:** `rcwa_jones_vs_wavelength(...)` looping the anisotropic solver, with an optional `materials=` mapping to the `refractiveindex` DB (e.g. Cu, Si3N4) so dispersion is automatic. Keep dispersionless as the fast default.

### GAP 6 — Layer / region-resolved absorption  ★ analysis aid
**What:** Break the total absorptance `A = 1 − ΣR − ΣT` down **by layer and/or material region** (where is the power lost — metal teeth vs back-reflector vs LC?).
**Why:** For lossy-metal devices, knowing whether absorption sits in the teeth or the back-plane drives design (and the metal-quality trade we explored). Currently only the scalar total is exposed.
**Suggested approach:** Integrate the (already-reconstructable, see GAP 1) internal field against Im(ε) per layer/region — a natural by-product once internal-field recovery exists.

### GAP 7 — Out-of-plane (full 3×3) tensor  ◦ general capability, low priority here
**What:** Support ε_xz, ε_yz (out-of-plane director tilt), not just the z-decoupled in-plane subset (~:1889-1894).
**Why:** Out-of-plane LC tilt (intermediate director angles) is common in real LC cells under field. Not required for this study's in-plane (θ≈π/2) case, but a real general-anisotropy limit.
**Suggested approach:** Extend the layer Q-block to carry the full tensor (the standard general-anisotropic FMM eigenproblem; Li 2003 — already cited for the 2-D path). Larger change; document the current subset clearly until then.

---

## Part 3 — Wishlist (quality-of-life / strategic)

- **W1. Batched / `vmap` geometry solve.** Solve many geometries (a parameter grid or an inverse-design population) in one batched GPU/JAX call. The autodiff path (`examples/13`) already differentiates a single solve; batching it would turn parameter sweeps and population optimizers from hours of host-loop into one device call — the single biggest throughput lever for design studies.
- **W2. Reflective-Jones device helpers.** A thin convenience layer for the recurring "metasurface-as-Jones-element" pattern: e.g. an out-coupling FOM for PBS → QWP@45° → grating → QWP@45° → PBS (retardance Γ = arg r_TM − arg r_TE → side-port power). Composes the RCWA Jones reflection with the library's existing waveplate/PBS Jones ops.
- **W3. 1-D device grating builder.** A `make_*`-style geometry helper (CONVENTIONS §1) for common 1-D device profiles — alternating/interdigitated teeth, duty-cycle + gap + independent ridge widths — emitting the segment list for GAP 2, so users don't hand-roll masks.
- **W4. Field-based inverse-design merit terms.** Once GAP 1 lands: differentiable merits on the internal field (maximize |E|² in a region, mode overlap, absorption in a target layer) for `optimize.JaxMeritTerm`.
- **W5. Resonance-aware truncation guidance.** Auto-suggest `n_orders` from period/λ/contrast and flag when operating on a sharp resonance (ties to GAP 3).
- **W6. JAX 2nd-order / Hessian validation.** Current gradients are validated to <1e-5; Hessians flow through the broadened-eig term but are unvalidated — relevant for Newton-type inverse design.

---

## Part 4 — Prioritized roadmap

| Priority | Item | Payoff (this use case + general) | Effort | Depends on |
|---|---|---|---|---|
| **1** | **Internal-layer E/H field reconstruction** (GAP 1) | Unblocks plasmonic/gap-mode physics & field-based merits; removes the only reason to keep an external engine | med | modal amps already retained |
| **2** | **Multi-region + stacked 1-D anisotropic API** (GAP 2) | Models real devices (≥3 regions, aniso-over-iso stacks) without pixelation | med | `RCWAStack` verify; analytic segment FT |
| **3** | **Convergence self-check / resonance guard** (GAP 3) | Prevents *wrong* answers from under-resolved sharp resonances (real error avoided) | low | ships `rcwa_extrapolate` |
| **4** | **Batched/vmap geometry solve** (W1) | Orders-of-magnitude faster sweeps & inverse design on GPU | med | JAX autodiff path |
| **5** | **Dispersive Jones wavelength sweep** (GAP 5) | Spectral device response w/ DB indices, one call | low | `refractiveindex` integration |
| **6** | **Tapered/slanted-sidewall helper** (GAP 4) | Fab realism (taper can make-or-break a device) | low–med | ASR (later) for non-staircase |
| **7** | **Layer/region-resolved absorption** (GAP 6) | Loss attribution for metal devices | low | GAP 1 |
| **8** | Reflective-Jones device helpers + 1-D grating builder (W2, W3) | Ergonomics for metasurface-as-Jones-element studies | low | GAP 2 |
| **9** | Full 3×3 (out-of-plane) tensor (GAP 7) | General anisotropy (tilted LC) | med–high | — |

---

## Part 5 — Notes & caveats

- **These are additions, not corrections.** The 1-D anisotropic solve, the Li factorization, the Jones output, and lossy-metal handling are correct and validated (energy ~1e-9–1e-13; isotropic reduction bit-exact; cross-pol present for in-plane directors). This audit is about *coverage and ergonomics* for device modelling, not solver correctness.
- **Highest leverage:** GAP 1 (internal field) for *physical insight & validation parity*, and GAP 2 (multi-region/stacked 1-D anisotropic) for *device breadth*. Together they would let the motivating LC/metal-grating study run entirely on LumenAiry.
- **Convergence-acceleration overlap:** GAP 3/4 intersect the separate convergence-acceleration roadmap (extrapolation already shipped; ASR/matched-coordinates would supersede the manual sidewall staircase). No duplication intended.
- **Recommended validation:** before declaring parity, cross-check LumenAiry's `RCWAStack` (1-D: anisotropic LC/metal layer + isotropic spacer/metal layer + metal substrate) against the external `nannos`-based implementation on one converged device geometry — compare the complex Jones-reflection entries and the out-coupling vs LC-angle curve. This both validates the multi-region/stacked path (GAP 2) and pins the harmonic count needed for the sharp gap-plasmon resonance (GAP 3).

---

## Part 6 — References

- L. Li, "Use of Fourier series in the analysis of discontinuous periodic structures," *JOSA A* **13**, 1870 (1996) — inverse rule (already implemented, 1-D).
- L. Li, "Fourier modal method for crossed anisotropic gratings with arbitrary permittivity and permeability tensors," *J. Opt. A* **5**, 345 (2003) — general anisotropic FMM (GAP 7; already cited in the 2-D path).
- (Convergence-acceleration references — Popov-Nevière 2001, Schuster 2007, Granet 1999, Weiss 2009, Lalanne 1997, Edee 2011 — are catalogued in `AUDIT_RCWA_CONVERGENCE_ACCELERATION_2026_06_01.md`; not repeated here.)
