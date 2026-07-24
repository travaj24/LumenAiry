# Root-cause audit — the traced element's FROZEN INTRA-GROUP AMPLITUDE (2026-07-24)

**Status: root cause of the traced-chain non-convergence (F-B / production-readiness §1)
IDENTIFIED and MEASURED.  Fix not yet implemented (design below, pending adversarial
review).**

Continues `AUDIT_TRACED_PRODUCTION_READINESS_2026_07_24.md` (the P0 closure plan) on the
64-GB-class box it called for (this box: 136 GB / 20 cores).  All numbers below were
measured this session on this box; the prior box's axis-B rows reproduce here **to the
digit** (EE6 50.0 / 59.8, window 58.2 / 68.7 at N=2048, wf=2.0, nfc=2048/4096), so
cross-box parity is exact and every comparison below is like-for-like.

Artifacts: `validation/repro_traced_carrier_121/stigmatic_control_121.py` (new),
`validation/repro_traced_carrier_121/traced_group_dx_probe.py` (new),
`repro_dx_scaling.py` (gained `NFC`/`WF`/`RNF` env knobs, defaults unchanged).

---

## 1. P0.1 stigmatic control — the MACHINERY IS CLEAN (both axes)

`propagate_traced_carrier_chain` was run on the real 121 geometry with
`apply_real_lens_traced` monkeypatched to an IDEAL STIGMATIC element (exact
sphere(R_in)→sphere(R_out) phase map from the same ABCD the orchestrator uses, hard
aperture, |T|=1).  Same gaps, same envelope/reconstruct hand-offs, same
`_fine_trace_group_exit` resample+retrace, same exact Bluestein readout.

| axis | settings | result |
|---|---|---|
| B (n_fine_cap) | N=2048, wf=2.0, nfc=2048/4096/8192 | **IDENTICAL to the digit**: FWHM 4.15 µm, EE6 65.2%, EE12 79.8%, window 84.0% at all three |
| (traced, same axis) | same | EE6 50.0 → 59.8 → **28.1% (collapse)** |

Per-stage power in the stigmatic chain is constant to 7 digits; the peak lands exactly
on-axis (the traced chain lands 21 µm off-axis, diagonally — a further element-side
symptom).  Supporting unit-energy audit (self-contained, no .zmx):
`_fourier_upsample_crop` Parseval = 1.000000 up & down, smooth and hard-edged (loss ≤
0.17% on hard-edge spectral truncation, never gain); `propagate_carrier_referenced`
conserves power to 8 digits and reproduces the analytic Gaussian w exactly at every
N ∈ {1024…16384}; `angular_spectrum_propagate_mft` window power = 100.000% with
analytic-matching focus, converged from N=1024.

**Conclusion: the S1(B) divergence/collapse axis is entirely the traced element re-run
on a finer grid inside `_fine_trace_group_exit`.  Nothing in the carrier / resample /
readout machinery moves.**

## 2. Per-group dx probe — the element itself diverges at fixed physics

`traced_group_dx_probe.py`: single 121 group, chain-representative carrier Gaussian
input, window fixed at 8.4 w, PHYSICAL ray pitch fixed (rs ∝ N), grid pitch alone swept.
Exit compared pointwise to the exact meridional-ray oracle.

**Lens S3-S4** (first group; aperture 24.65 mm = 2.47× the 10 mm beam):

| N | dx µm | rms_res rad | P_out/P_in | w_out/w_in |
|---|---|---|---|---|
| 512 | 81.9 | 0.004 | 1.000 | 0.9999 |
| 1024 | 41.0 | 0.007 | 1.000 | 1.0002 |
| 2048 | 20.5 | 0.015 | 1.000 | 1.0005 |
| 4096 | 10.2 | 0.041 | 1.000 | 1.0026 |
| 8192 | 5.1 | **0.243** | 1.000 | **1.0578** |

S5-S7 shows the same w-inflation pattern (1.000 → 1.054) with a clean core phase.
Power is conserved everywhere — this is pure redistribution, the chain-level F-B
signature (stage-exit envelope +3.4% at S3-S4, +11% by S16-S17, window-total decline
87.6→75.2% across the N sweep) reproduced in ONE group with zero chain machinery.

## 3. Ingredient discriminator — two defects, cleanly split

S3-S4, N ∈ {2048, 8192}:

| variant | rms_res @2048 | rms_res @8192 | w_out/w_in @8192 |
|---|---|---|---|
| V1 traced default (`preserve_input_phase=True`) | 0.015 | **0.243** | 1.0578 |
| V2 traced `preserve_input_phase=False` | **0.001** | **0.012** | 1.0578 |
| V3 bare `apply_real_lens` (amplitude only) | – | – | **1.0579** |

- **Defect A (phase):** V1−V2 = the wave-phase pair `angle(E_analytic) −
  angle(E_analytic_pw)` contributes 0.015 rad at the chain's own pitch and 0.24 rad at
  fine pitch, per group.  The pure `k0*opl_map` phase (V2, carried by the H6 carrier
  entrance eikonal) is 20× cleaner at fine dx and 15× cleaner at the chain pitch.
- **Defect B (amplitude):** the exit-width inflation is byte-identically present in the
  bare `apply_real_lens` amplitude — the traced element inherits it via
  `amp = |E_analytic|`.

**Taper/rim hypothesis REFUTED (V4):** clamping the aperture to 4w (radius 2w) and,
independently, soft-tapering the input tail at 1.75–2.25w leave BOTH defects unchanged
at N=8192 (rms 0.243, w ratio 1.056).  The divergence does not come from
marginal/rim-zone content; it comes from inside the beam.

## 4. The root cause — the amplitude leg runs on a carrier-ALIASED field and
##    freezes the intra-group beam expansion

The chain (and any carrier-referenced call) hands `apply_real_lens_traced` a field whose
spherical carrier is far beyond the grid Nyquist inside the beam itself: at N=2048
(dx = 20.5 µm), the R_in = +47.9 mm entrance sphere aliases beyond r = R·λ/(2dx) ≈
1.5 mm, while the beam is w = 5 mm.  The carrier ARCHITECTURE handles this for the
PHASE (pointwise conjugation is exact at sample points; the traced OPL is referenced to
the analytic carrier — H6/R7).  But **Step 1's amplitude pass — `apply_real_lens(E_in)`
— receives the raw aliased field**.  An aliased sphere cannot diffract/diverge on the
grid, so the wave pass transports the amplitude through the group essentially
UNCHANGED.

The physics says otherwise.  Exact q-trace through S3-S4 (n(N-SK2, 1.31 µm) =
1.5917 from the registered Sellmeier; exit R reproduces the repro's ABCD R_out
+143.37 mm to 4 digits):

- true exit beam: **w = 5.6265 mm = 1.1267× the entrance** (7 mm of glass at
  R ≈ +48 mm),
- traced element delivers **1.0005×** at N=2048 (frozen at the entrance beam),
  **1.058×** at N=8192 (the aliasing radius has moved out to ~6 mm, so the pass begins
  to resolve the real divergence — the model **crawls toward the truth as dx → 0**, and
  the "F-B divergence" is precisely this crawl, superposed with defect A's growing
  phase corruption).

Chain-level consequence, verified against the stage prints: S3-S4's exit re-envelopes
with the CORRECT R_out but the WRONG (frozen) w, so S5-S7 receives w = 5.40 mm where
the design q-trace says 6.02 mm — and so on down the chain, compounding.  A corrected
relay's aberration balance and its final NA assume the design beam sizes; the chain
therefore runs off-design at EVERY grid, with dx only changing HOW wrong:

- coarse dx → fully frozen amplitudes → under-sized beam → under-filled final NA →
  wide focus (FWHM 3.95–4.05 µm vs Zemax POP 2.74 µm) with clean-ish per-group phase;
- fine dx → partially restored expansion but 0.24 rad/group of defect-A phase
  corruption → focus collapse (EE6 46.5% at N=8192; nfc=8192 collapse row).

The per-group oracle gates (R7: < 0.023 rad) never caught this because they validate
PHASE against the oracle and launch each group with its own TRUE q-trace beam; the
amplitude expansion was never gated, and per-group stage w "looks plausible" in the
chain because the gap transport grows the beam at the correct RATE from the wrong size.

This also subsumes/illuminates:
- **the R9-addendum's "~1.68 rad accumulated wavefront"** entering the tail — the
  accumulated inter-group inconsistency of a chain run off-design (NOT a launch defect
  — Part E's negative result is consistent);
- **the 21 µm diagonal peak offset** (aliased-carrier junk breaking symmetry — the
  stigmatic control is exactly on-axis);
- **the axis-B collapse** (`_fine_trace_group_exit` re-runs the LAST group at
  dx_fine ≈ 1 µm at nfc=8192 — deep into defect A's fine-dx regime);
- part of the **aperture:beam cliff**'s sharpness on fast singlets (E4) remains a
  separate (fit-domain) finding — V4a shows the 121 divergence is NOT the cliff.

## 5. Fix design (pending adversarial review — NOT yet implemented)

Principle: when an explicit carrier R_in is engaged, no ingredient of the assembly may
consume the raw (carrier-aliased) field.  The phase side already obeys this (traced OPL
+ H6 carrier eikonal).  The amplitude side must be rebuilt in the carrier frame:

1. **Carrier-referenced amplitude (paraxial remap), opt-in kwarg on
   `apply_real_lens_traced`:** de-chirp `env = E_in · exp(−i k0 W(R_in))` (pointwise,
   exact); compute the intra-group magnification for the R_in congruence from the
   group's own air-to-air ABCD, `m = A + B/R_in` (equals the q-trace w ratio; 1.1267
   for S3-S4 — same ABCD source as `_paraxial_group_r_out`); amplitude_exit(r) =
   |env|(r/m)/|m| (band-limited envelope interpolation), then the existing
   valid/aperture masks.  Intra-group envelope DIFFRACTION is neglected (Fresnel number
   over ~t/n at mm beams ≫ 1; quantify in the gate).
2. **Exit phase (defect A):** replace the wave-pair with `k0·opl_map` (V2 form) PLUS
   the input envelope's residual phase transported on the same remap,
   `angle(env)(r/m)` — the geometric transport of the non-carrier residual (this is
   what `preserve_input_phase` was for; the remapped-residual form carries it without
   ever touching the aliased total phase).
3. **Gating:** engages ONLY for an explicit engaged carrier (chain case); `carrier=None
   / 'auto'-flat` paths byte-identical.  Chain opt-in first
   (`traced_kwargs={'carrier_amplitude': 'remap'}` or equivalent), promoted to the
   chain default once gates pass.
4. **Gates:** (a) per-group probe: w_out/w_in = q-trace ratio ± <1% and rms_res ≤
   ~0.02 rad, FLAT across N = 512…8192; (b) chain axis A+B: EE6 plateau (<2 pt drift),
   window ≤ stop losses, monotone approach; (c) byte-identical default suite; (d) 121
   vs Zemax POP (2.74 µm) — the P1 measurement, expected to close most of the 50→99
   gap if this diagnosis is complete.

## 6. Session artifacts / open items

- Axis-A traced rows (full final-leg settings, readout N_fine pinned 16384):
  N=1024 → FWHM 4.05 / EE3 66.4 / EE6 81.5 / EE12 82.8 / window 87.0 (N=2048+ in
  progress at write time; the pre-fix audit matrix's divergent trend is expected to
  reproduce).
- Stigmatic axis-A rows OOM when run beside a traced row (the thin stub does not
  compress the last-group beam, so its readout Bluestein needs ~4.5 GiB × several;
  rerun sequentially if needed — axis B + the T1-T3 unit audit already carry the
  machinery verdict).
- N=28672 extent-preserving traced run (post-F-A-fix energy check) still queued — must
  run ALONE (~100 GB).
- T4/T5 (readout end-to-end energy/convergence at NA 0.45) OOMed against the sweep;
  rerun pinned when the box is free.
