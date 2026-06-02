# RCWA — `RCWAStack` Convergence Robustness for Sharp Resonant Metal Multilayers

**Date:** 2026-06-02
**Author:** Claude Opus 4.8 (1M context)
**Type:** Follow-up finding (post-implementation of `AUDIT_RCWA_GAPS_AND_WISHLIST_2026_06_01`) — one residual robustness gap + residual nice-to-haves.
**Scope:** A cross-engine validation of a real device (1-D lossy-Cu/LC reflective grating) against an external nannos-based oracle surfaced **one** remaining issue in `lumenairy/elements/rcwa.py` (v5.10.6 / v5.11.0): `RCWAStack.solve` converges **non-monotonically (isolated-order resonance spikes)** for a *sharp resonant, stacked, metal* device. Everything else in the prior audit verified as implemented and correct.

---

## Part 0 — TL;DR

The gaps-and-wishlist audit (GAP1–7, W1–3, W6) is **fully implemented and verified** — including the device topology this study needs (2 stacked 1-D anisotropic patterned layers via `RCWAStack` + `eps_tensor_cell`). Cross-validation against a converged nannos oracle confirms:

- **Binary core solver** (`rcwa_jones_1d`): co-pol |J| Δ≤0.6%, phase Δ≤1° (two pure conventions: director azimuth `φ_lumen = π/2 − φ_ours`; cross-pol sign).
- **Single-layer metal/LC grating**: `rcwa_jones_1d` (analytic Li) and `RCWAStack` (pixelated tensor cell) **agree and converge cleanly** (90.3% vs 90.4%, flat across n_orders 120→360).
- **PBS/QWP polarization chain**: `apply_polarizing_beam_splitter` + `apply_quarter_wave_plate` reproduce the analytic out-coupler to ~2e-15.

**The one residual issue:** for the **2-layer, sharp-resonant** device, `RCWAStack.solve` does **not** converge monotonically — the reflection-null oscillates with isolated-order spikes — whereas nannos (Li/tangent) converges monotonically to the same value, and the *single-layer* lumenairy path is clean. The energy stays bounded (no instability), so the result is "right neighborhood, wrong at isolated truncations." This is the same **isolated-resonance** failure mode fixed for `pmm_efficiency_1d` in **v5.10.6**, but `RCWAStack.solve` has **no `stabilize`/consensus guard** (`solve(self, *, retain_internal=False)` — no `formulation`, no `stabilize`).

---

## Part 1 — Evidence

Device: 1-D doubled period Λ=380 nm, two stacked patterned layers — Layer B (285 nm): [Cu tooth 130 | LC gap 60 | Cu tooth 130 | LC gap 60]; Layer A (65 nm): [Cu | SiCN]; McPeak Cu (ε=−83.13+2.70j), in-plane LC (Δn=0.30), Cu substrate. Out-coupling = ¼|r_TM+r_TE|² (PBS→QWP@45→grating→QWP@45→PBS).

**`RCWAStack` (pixelated tensor cells), out-coupling vs `n_orders`:**

| n_orders (total 2N+1) | peak (φ=0) | null (φ=90) | absorptance |
|---|---|---|---|
| 120 | 87.9% | 3.2% | 0.32 |
| 180 | 90.2% | 2.7% | 0.33 |
| 240 | 89.9% | **0.8%** | 0.33 |
| 300 | 87.1% | **8.0%** | 0.39 |
| 350 | (88.9%) | **14.0%** | — |
| 360 | 90.3% | **0.9%** | 0.31 |

**nannos oracle (Li/tangent), monotonic:** null = 6.2% (nh400) → 0.76% (nh494) → 0.56% (nh588); peak 90.1%.

So the lumenairy stack null is correct (~0.8–0.9%) at n=240 and n=360, but **spikes to 8% at n=300 and 14% at n=350** — isolated bad truncations, not a trend. Absorptance never leaves [0,1] (no flux instability). A user picking a single `n_orders` can land on a spike and get a qualitatively wrong null (the headline figure-of-merit for this device class).

**Control (rules out the obvious causes):**
- *Single-layer* metal/LC grating, same materials: `rcwa_jones_1d` (analytic Li) and `RCWAStack` (pixelated) **both** give 90.3–90.4% flat across n_orders 120→360 → the stacked tensor-cell path is **not** missing the metal factorization, and pixelation alone is **not** the cause.
- The spikes are **stacking + sharp-resonance** specific (two deep patterned layers + a high-Q gap-plasmon null).

Reproducers (in the external study): `validation/check_device_convergence.py` (the table above), `validation/check_stack_convergence.py` (single-layer control), `validation/check_device_parity.py` (full curve vs golden).

---

## Part 2 — Likely mechanism

A high-Q resonant null is a near-cancellation of `r_TM` against `r_TE`; its accuracy is dominated by the worst-resolved mode. In a **stacked** solve, the layer↔layer and layer↔region mode-matches can be **near-singular at isolated truncation counts** (a retained evanescent mode of one layer nearly degenerate with the interface basis), injecting a small bias that — because the null is a near-cancellation — shows up as a large *relative* error in the null while leaving total power bounded. This is precisely the `pmm_efficiency_1d` resonance pathology documented in the v5.10.6 CHANGELOG ("near-singular layer↔region interface mode-match… inflating/biasing… LAPACK-build dependent"), here manifesting in `RCWAStack` rather than PMM. nannos's formulation happens to converge monotonically on this geometry; lumenairy's stack does not.

---

## Part 3 — Recommendation (prioritized)

1. **Extend the v5.10.6 resonance-robust `stabilize` consensus selector to `RCWAStack.solve`.** ★ highest leverage. Add `stabilize=True` (scan a short `n_orders` window, discard outliers whose total power / per-order set is off the consensus cluster, return the consensus). The machinery already exists for PMM — porting it to the stack solve would make sharp-resonant multilayer devices reliable at a user-chosen order. Also expose **`formulation=`** on `RCWAStack.solve`/`add_layer` (today there is none) for parity with `rcwa_efficiency_1d`.
2. **`rcwa_convergence` wrapper for stacks.** The guard exists (`rcwa_convergence`, v5.9.0) for the single-entry solvers; wire it to accept an `RCWAStack`/its `solve` so users get the "solve at N and N+bump, warn/extrapolate" safety on multilayer devices too. Pairs with an **auto-`n_orders`** (audit W5) that bumps until the consensus stabilizes — important for high-Q nulls.
3. **(documentation)** Until 1–2 land, document that `RCWAStack` results for sharp resonant metal multilayers should be checked across ≥2–3 `n_orders` (or use `rcwa_extrapolate`), since a single truncation can hit a resonance spike.

---

## Part 4 — Residual nice-to-haves (surfaced during the same validation; lower priority)

- **No analytic segment layer in `RCWAStack`.** `rcwa_jones_1d_segments` (analytic, Li-factorized, supports out-of-plane GAP7) is **single-layer only**; the multilayer stack must use **pixelated** `eps_tensor_cell`. A `RCWAStack.add_segment_layer(thickness, segments=…)` (compose analytic Li segment-layers vertically) would (a) remove in-plane pixelation and (b) carry GAP7 out-of-plane anisotropy into stacks (the stack tensor path rejects off-plane, `rcwa.py:~4403`) — so field-driven LC *tilt* in a real device could be modeled.
- **`layer_absorption` rejects tensor/shape cells** (`rcwa.py:~4244`). Our metal-tooth/LC layers *are* tensor cells, so per-layer loss attribution (teeth vs back-plane — the motivating GAP6 question for this device) does not work. Extend it to tensor cells (and ideally per-*region* within a segmented layer).
- **`internal_field` only on `RCWAStack.solve`.** `rcwa_jones_1d`/`_segments` return no modal handle, so the gap-plasmon field cannot be reconstructed for the analytic (non-pixelated) single-layer model — only the pixelated stack. A `return_result=True` (RCWAResult-returning) variant of the segment solver would expose it.
- **`reflective_outcoupling` is NumPy-only** (`rcwa.py:~3174`) — won't trace through `jax.grad`. A differentiable twin would let inverse design optimize side-port out-coupling (the stated FOM) directly.
- **`rcwa_jones_1d_segments` differentiability is plumbed but unverified** (no `jax.grad`/FD test; the off-plane generator uses `np.where`/`argsort` in flux selection — a likely non-differentiable spot worth a pinned test before inverse-designing the segmented LC cell).
- **`rcwa_jones_vs_wavelength` is binary-only** — no segmented-grating dispersive Jones sweep; spectral response of the 4-region cell needs a hand-loop.
- **An anisotropic / Jones PMM would be the single strongest convergence lever for this device class.** `pmm_efficiency_1d` (v5.8.0) delivers **spectral, no-accuracy-floor** convergence for metal gratings (polynomial `degree`~12–24 with tens of DOF, vs FMM `nh`~300–500 with 2·nh DOF — potentially orders of magnitude cheaper for the same accuracy, and floor-free where ASR plateaus ~1e-4 for TM). **But it is isotropic, scalar (separate TE/TM, no coupling), binary single-layer, efficiency-only (no phase/Jones), normal-incidence, and non-differentiable** — so it **cannot** model the anisotropic-LC reflective-Jones device (which is defined by the rotated-uniaxial tensor, TE↔TM coupling, the complex `r_TM`/`r_TE` *phase* relationship, and a multi-region + stacked geometry). Extending PMM to the **anisotropic 1-D Jones** case (rotated tensor + 2×2 Jones, ideally stacked) would attack exactly the high-`nh` / sharp-gap-plasmon-null cost — and being floor-free would beat ASR. **Shorter term:** extend **ASR / matched-coordinates** (currently on the scalar `rcwa_efficiency_1d`) to the **anisotropic `rcwa_jones_1d` / `rcwa_jones_1d_segments`** path — that alone would cut the order count for the resonant null this study cares about.

---

## Part 4.1 — Should PMM be extended (anisotropic / Jones / multi-region), and should `RCWAStack` compose PMM layers? — **Yes to both, in that order**

Two design questions follow directly from the convergence finding above. Both answers are **yes**, with a dependency order.

### (A) Extend `pmm_efficiency_1d` → anisotropic + full 2×2 Jones + multi-region (+ oblique, + autodiff). ★ do first

- **Why it's the right lever.** PMM's spectral, **no-accuracy-floor** convergence is exactly what a sharp gap-plasmon **null** needs — the null is a near-cancellation of `r_TM` against `r_TE`, so Fourier-truncation error (and the ASR ~1e-4 TM floor) corrupt it, whereas PMM's TM error drops monotonically with no plateau. For the anisotropic-LC reflective grating this would be the strongest convergence accelerator in the library.
- **Multi-region is natural, not a stretch.** PMM is already **subsectional** (spectral elements per homogeneous region); the binary `n_ridge`/`n_groove`/`duty_cycle` API is an *interface* limitation, not a *method* one. N regions = N subsections, so a 4-region cell (grounded tooth | LC gap | floating tooth | LC gap) is a direct generalization — arguably *easier* in PMM than the analytic-FT binary path.
- **Anisotropy → Jones is a tensor modal eigenproblem** — the spectral-element analogue of the FMM tensor extension already shipped (`_layer_eigenmodes_tensor`). Carry the in-plane `(ε_xx, ε_xy, ε_yx, ε_yy, ε_zz)` block in the element stiffness/mass operators; the off-diagonal `ε_xy` couples the field components, yielding the full 2×2 Jones (and the complex `r_TM`/`r_TE` **phase** the PBS→QWP→grating→QWP→PBS routing depends on). The z-decoupled in-plane subset (the FMM scope) is the sensible first target; out-of-plane (Li-2003) later.
- **Oblique** is the `+i k_x0` Bloch shift in the stiffness already scoped in the docstring; **autodiff (JAX)** would let inverse design exploit PMM's tiny DOF count (tens vs FMM's 2·nh).

### (B) Let `RCWAStack` compose PMM layers (an `add_pmm_layer` / PMM-backed layer type). ★ do second (gated on A)

- **Architecturally clean.** The Redheffer / scattering-matrix recursion is **mode-agnostic** — it composes each layer from its eigenmodes + an interface mode-match. A PMM layer supplies polynomial/nodal modes instead of Fourier ones; the only new machinery is **projecting the PMM interface field onto the common Rayleigh / Fourier interface basis** shared by the super/substrate and any adjacent Fourier layer (standard spectral-element ↔ plane-wave coupling). Mixed PMM + FMM stacks then compose naturally.
- **Why it matters here: root-cause cure for the resonance spikes** documented in Parts 1–2. Those spikes are Fourier-truncation / near-singular-mode-match artifacts; a PMM-backed layer removes the Fourier truncation (and its resonances) for the layer that needs it, so a stacked sharp-resonant metal device converges spectrally with no spikes. The Part 3 `stabilize` consensus is the cheap **symptom** patch; PMM-in-stack is the **structural** fix.
- **Prerequisite: (A).** Until a PMM layer can represent an anisotropic, multi-region patterned layer, a PMM stack could only carry isotropic binary layers — which the FMM stack already handles well. So (A) must land first for (B) to help this device class.

### Recommended sequence

1. **`stabilize` / consensus on `RCWAStack.solve`** (Part 3) — cheap, immediate reliability for sharp resonant stacks; unblocks the device port now.
2. **PMM → anisotropic + Jones + multi-region (single layer)** — covers our device's single patterned layer spectrally and floor-free; the highest-value convergence work for this device class. (A multi-region anisotropic PMM may already model many device variants as one effective patterned layer, without needing the stack.)
3. **PMM layers in `RCWAStack`** (or a PMM-native multilayer S-matrix) — covers the full 2-stacked-layer device, eliminating the Fourier-resonance spikes at the root.

**Scope honesty:** (A) is real work (a coupled/vector spectral-element eigenproblem), and (B) adds cross-basis interface matching — neither is a quick patch. But together they are the *step-change* for sharp anisotropic resonant gratings, and they degrade gracefully: `stabilize` (1) gives reliability now, (A) gives the spectral win for single-layer devices, and (B) extends it to stacks.

---

## Part 5 — Caveats

- This is **not** a correctness bug in the core factorization — single-layer and binary cases are exact, energy is conserved throughout. It is a **convergence-robustness** gap for the stacked + sharp-resonance combination, identical in spirit to the PMM resonance fix already shipped.
- The cross-engine "truth" is a converged nannos (Li/tangent) result, itself convergence-checked (monotone to nh=588). The two engines agree to ≤1% on this device wherever lumenairy's stack is *not* on a resonance spike (e.g., n_orders 240/360).
- Director-angle convention between the codes is `φ_lumen = π/2 − φ_ours` (+ a cross-pol sign), both physically irrelevant to the co-pol out-coupling; reconciled in the external harness.
