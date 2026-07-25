# Audit / handoff — traced-carrier propagator production-readiness (2026-07-24)

> **CLOSURE (2026-07-25): P0 and P1 are CLOSED; the §6 acceptance criteria are
> met.**  Full record: `AUDIT_TRACED_FROZEN_AMPLITUDE_2026_07_24.md` §6–§8.
> Scorecard against §6:
>
> - **converges under dx refinement (both axes):** YES — the validated chain
>   configuration (now the v5.29 defaults: `carrier_reference='sphere'` +
>   `amplitude_model='ray_density'` + `preserve_input_phase='remap'`) is
>   dx-flat (EE6 99.1/99.3/99.3 at N=1024/2048/4096; the intermediate
>   `rd+pip0` configuration is flat TO THE DIGIT).  The §1 divergence was:
>   the carrier-aliased amplitude pass (frozen intra-group expansion) + the
>   preserve-pair phase corruption + a coarse→fine upsample lattice bug
>   (`ii·Ns/N` vs `ii/sub` — also the chain's diagonal focus walk).
> - **conserves energy at every grid:** YES — the §1 energy gain was entirely
>   the F-A clamp bug; post-fix N=28672 window-total 88.9%; the hypothesized
>   second mechanism does not exist.  Window 99.4% at the validated config.
> - **matches Zemax POP within a stated tolerance:** YES with a metric
>   correction — POP's 2.736 µm is the waist RADIUS (FWHM 3.223 µm; the §1
>   note "images to 2.74 µm" conflated the two).  Measured best focus:
>   FWHM 3.550 µm (the ideal-field ceiling through the same readout is
>   3.45–3.55), EE3 88.4% (ceiling 90.3), EE6 99.3%, on-axis, best focus
>   +5–10 µm from the plane, at memory-feasible settings (N=2048,
>   nfc=8192).  Verified against a full-train meridional ray oracle
>   (design floor rms 0.018 rad; the chain delivers 0.015).
>   **Post-S12 (2026-07-25, `a9dc454`): the pure-defaults acceptance
>   improved to best-focus FWHM 3.450 µm — equal to the measured
>   ideal-field ceiling — EE3 88.8 / EE6 99.6 / EE12 99.8, on-axis,
>   dx- and ray_subsample-flat** (the `remap_sampling='full'` chain
>   default; the residual EE3 distance to 90.7 is the like-for-like
>   ceiling gap, mostly irreducible readout per the measured S12 budget).
> - **P2 (daily-driver guards, §4/§5): DELIVERED (2026-07-25).**
>   (1) Aperture:beam cliff guard — `fit_radius_beam_factor` (chain default
>   2.0; element opt-in) restricts the ray-FIT domain to a beam-relative
>   disc, energy-neutral to 4 digits: E4 exit Strehl 0.105/0.042/0.039 at
>   7/8/10 mm apertures → 0.9995, pre-cliff and 121 acceptance unchanged;
>   warn-only `on_aperture_beam` flags the >1.5×-beam regime.  §4's cliff
>   mechanism CORRECTED by measurement: it is the un-carriered launch
>   SQUARE's corner samples — a collimated `carrier=inf` eikonal is NaN so
>   the R7 fit restriction never engages; group-discriminated (entirely the
>   fast first group), config-independent, `_CARRIER_FIT_RADIUS_FRAC`-null,
>   π/√3 random-wrapped-phase residual signature.  (2) Memory-bounded
>   readout — fine grids capped against `get_ram_budget()` (new
>   `ram_budget=` kwarg / `focus_readout` key) with a warning naming the
>   un-degraded requirement; the 34 GB-box crash condition now degrades
>   gracefully.  (3) Opt-in `self_check='dx'` dx-stability flag (healthy
>   chain silent at 0.06%; aliased-carrier case warns at 50%+ deltas).
>   (4) CI-safe design battery (17 tests: singlet/doublet/triplet/relay ×
>   NA × aperture:beam) + guards tests (12) documenting the known-good
>   envelope (aperture:beam 1.2–2.5×, exit NA 0.013–0.20).  Consolidated
>   suite 158 passed.  Note for future pins: the chain is FP-floor
>   reproducible (~1e-15 rel), not bit-reproducible run-to-run —
>   "byte-identical" claims should be stated as FP-floor identity.
>
> §5's P0.1 prediction is settled: stigmatic CONVERGES and traced diverged →
> the defect was in the element/hand-off layer, as the plan's branch (a)
> anticipated — though the specific mechanisms differed from every §1-era
> hypothesis; see the closing audit for the corrected causal chain.

**Purpose:** a clean-slate reassessment of what stands between the traced-carrier propagator
(`apply_real_lens_traced` + `propagate_traced_carrier_chain`) and being (a) production-ready for the
**design-121** relay and (b) a trustworthy **daily-driver** propagator for arbitrary designs. Written
as a **handoff** — the current box hit its RAM ceiling; work continues on a larger box.

**Discipline used:** claims below are tagged **[VERIFIED-2026-07-24]** (re-measured directly this
session), **[EVIDENCE-PRIOR]** (from a prior audit's documented measurement, not re-run here), or
**[UNVERIFIED]** (assumption / prior claim NOT re-checked — do not trust without testing). Several
prior conclusions were **overturned** this session; they are called out.

**Library state:** v5.28.0 working tree. This session **reverted** the experimental Part E
`wavefront_aware` code (see `AUDIT_WAVEFRONT_AWARE_RAY_LAUNCH_2026_07_23.md` §4c). The dx-scaling
F-A/F-C/F-D code fixes are present but **uncommitted**. `_lens_traced.py` / `carrier.py` currently
contain only those dx-scaling fixes on top of v5.28.0.

---

## 1. The headline finding — the traced chain does NOT converge (and breaks energy conservation)

**This is the central production blocker and it supersedes every "fix" chased earlier this session.**

The design-121 chain's absolute focal metrics (EE, FWHM) **do not converge** as the grid is refined —
they drift monotonically and then break down. Two independent resolution axes both show it:

**(A) Chain-grid axis** — vary `N` / launch `dx0` **[EVIDENCE-PRIOR: `AUDIT_TRACED_CHAIN_DX_SCALING_2026_07_22.md` F-B matrix]:**

| N | dx0 | FWHM | EE3 | EE6 | EE12 | window |
|---|---|---|---|---|---|---|
| 1024 | 2.0 µm | 3.75 µm | 65.8% | 82.8% | 84.7% | 87.6% |
| 2048 | 1.0 µm | 4.05 µm | 52.4% | 69.7% | 73.5% | 78.7% |
| 4096 | 0.5 µm | 4.35 µm | 39.8% | 54.6% | 67.2% | 77.5% |
| 8192 | 0.25 µm | 8.85 µm | 19.1% | 46.5% | 59.2% | 75.2% |
| 28672 | 0.071 µm | 7.05 µm | 50.0% | **102.3%** | **128.2%** | **130.8%** |

**(B) Final-leg-resolution axis** — vary `n_fine_cap` at N=2048, wf=2.0 **[VERIFIED-2026-07-24]:**

| n_fine_cap | FWHM | EE3 | EE6 | window |
|---|---|---|---|---|
| 2048 | 5.35 µm | 32.0% | 50.0% | 58.2% |
| 4096 | 4.65 µm | 43.3% | 59.8% | 68.7% |
| 8192 | **15.55 µm** | 8.8% | 28.1% | 48.8% |

Two disqualifying behaviors, on **both** axes:

1. **No convergence / no plateau.** Refining the grid changes EE6 by tens of points with no sign of
   settling. On axis A finer = monotonically worse (82.8% → 46.5%); on axis B it improves then
   collapses (50 → 60 → 28%, FWHM ballooning to 15.5 µm). There is **no dx at which the answer is
   stable**, so **no absolute EE/FWHM number from the traced-121 chain is currently trustworthy.**
2. **Energy non-conservation at fine grids.** At N=28672 the readout window holds **130.8%** of the
   launch power and EE6 reads **102.3%** — physically impossible. This is a hard bug (spurious energy
   **gain**), distinct from the F-A clamp bug already fixed, and it appears only when the grid is
   pushed fine. (Energy IS conserved at N ≤ 2048 / n_fine_cap ≤ 4096 — window ≤ ~69% — so the gain
   mechanism is resolution-triggered.)

**Consequence for the narrative:** the "EE6 ≈ 50%" (this session, memory-limited) and "≈ 69.7%"
(prior R9 table, N=2048) numbers that have been treated as *the* traced-121 result are just two
points on a non-converged, eventually-unphysical curve. They carry **no fidelity meaning**. The
true converged traced-121 EE6 is **unknown**. (Zemax POP — a converged reference — images this
well-corrected relay to 2.74 µm, i.e. ≈ diffraction-limited / EE6 high; the traced chain should
converge there and instead diverges away from it as dx→0.)

---

## 2. What was RULED OUT this session (so the next box does not re-chase it)

- **Ray-launch congruence is NOT the 121 blocker [VERIFIED-2026-07-24].** The Part E "wavefront-aware"
  (carrier-relative residual) launch was implemented and tested: it *regressed* the 121 (EE6
  50% → 10%) and gave no improvement on a synthetic corrected relay. **Reverted.** The R9-addendum's
  "~1.68 rad launch-discarded residual" diagnosis is **[UNVERIFIED]** and is contradicted by these
  results — do not assume it.
- **Aperture / ray-fit-domain corruption is NOT the 121 blocker [VERIFIED-2026-07-24].** Clamping the
  per-group apertures to 3× the beam made the 121 *worse* (50 → 27.7%, via vignetting); a clean
  `_CARRIER_FIT_RADIUS_FRAC` sweep (0.5→0.2, no vignetting) was **flat** at ~50%. So the fit-domain is
  not the 121's issue.
- **The synthetic "chain fails corrected relays" result was a TEST ARTIFACT [VERIFIED-2026-07-24].**
  The E4 synthetic relay had a 2.5×-oversized aperture; at a beam-matched aperture the plain
  sphere-only chain is diffraction-limited (Strehl 0.997, ties GBD). The traced-carrier machinery does
  **not** fundamentally fail corrected relays. (This overturned an earlier reading in this session.)

Net: the 121 gap is **neither launch nor aperture/fit** — it is the **non-convergence / energy bug**
of §1. That is where all effort should go.

---

## 3. What is SOLID (verified) and works

- **Default path unchanged & tests green [VERIFIED-2026-07-24].** After the Part E revert, Part E
  markers = 0, dx-scaling fixes intact (13 markers), imports clean, and the R6/R7/R8/R9 + dx-scaling +
  repurposed-E4 suites pass (R8 F3 behavior restored; E4 16/16).
- **The chain CAN be diffraction-limited [VERIFIED-2026-07-24].** On a well-posed corrected relay
  (E4, beam-matched 6 mm aperture) the sphere-only chain reaches Strehl **0.997** (focus-independent
  exit-wavefront rms). So the architecture is capable; the 121 non-convergence is a *numerical*
  defect, not an architectural impossibility.
- **Per-group fidelity fixes (R6 auto-carrier, R7 intra-group curvature, R8 API, R9 exact high-NA
  final leg) and dx-scaling F-A/F-C/F-D are in place** [EVIDENCE-PRIOR + green tests]. F-A restored
  energy conservation at the n_fine_cap **downsample clamp** specifically (this is NOT the §1 fine-grid
  energy-gain bug — different mechanism).

---

## 4. Additional production blockers (independent of §1)

- **Aperture:beam "cliff" in the traced element [VERIFIED-2026-07-24].** When the physical aperture
  greatly exceeds the beam, wildly-aberrated marginal rays alias into the low-order OPL fit and
  collapse the focus. E4 sweep (beam w0=2 mm): Strehl 0.999 (ap 4 mm) / 0.997 (6 mm) → **0.104 (7 mm)
  → 0.038 (10 mm)** — a sharp cliff at ~1.5× the beam diameter. Only `min_coarse_samples_per_aperture`
  (a ray-fit *density* guard) exists; there is **no guard for this marginal-ray-aberration cliff**. A
  daily driver receiving arbitrary prescriptions will hit it silently. **Fix candidate:** a
  beam-relative launch/fit radius that decouples the ray-fit domain from the vignetting aperture, plus
  a warning when aperture ≫ beam.
- **Memory ceiling [VERIFIED-2026-07-24].** Converged settings are not memory-feasible here: the exact
  readout's internal resolution (driven by `window_factor`) demands single arrays up to
  32768² complex128 = **16 GiB**; this box has 34 GB total / ~10–12 GB free, so full settings crash
  (and the sweep above crashed climbing `window_factor`). A production daily-driver needs a
  **memory-bounded mode** with graceful degradation and honest "resolution-limited / non-converged"
  reporting — not a silent OOM or a silently-degraded number.

---

## 5. Closure plan (prioritized) — what needs to happen

### P0 — Root-cause and fix the non-convergence + energy-gain (§1). *Everything else is secondary.*

1. **Isolate element vs machinery (stigmatic control).** Run the 121 chain with **ideal thin/stigmatic
   elements** (no traced OPL) through the SAME carrier → reconstruct → gap-transport → exact-readout
   machinery, and sweep N and n_fine_cap. Prior claim (**[UNVERIFIED]** — the audit asserts stigmatic
   → 2.97 µm / EE6 100%) must be re-checked *as a convergence sweep*:
   - If stigmatic **converges** (stable EE6, energy ≤ 100%) but traced **diverges** → the defect is in
     `apply_real_lens_traced` (the traced element's OPL fit / interpolation under grid refinement).
   - If stigmatic **also diverges** → the defect is in the carrier/envelope/reconstruct/readout
     machinery (shared by both). *This single test localizes the bug.*
2. **Energy-gain hunt.** Track window-total and per-stage power vs grid to find WHERE >100% energy
   enters. The gain is almost certainly a normalization/interpolation scaling error in one resample
   step under refinement — prime suspects: `_fourier_upsample_crop` (the F-A/F-C region),
   `carrier_referenced_reconstruct`, and the Bluestein/MFT exact-readout scaling
   (`angular_spectrum_propagate_mft` / `carrier_referenced_exact_focus_readout`). Drive a **unit-energy
   test field** through each transform in isolation across N and assert Parseval/energy conservation;
   the one that gains energy as N grows is the bug.
3. **Fix to convergence.** Acceptance: on both axes, EE6 stable to **< ~2%** across N = 1024…8192 (and
   n_fine_cap comparably), window-total **≤ ~96%** (stop losses) at every grid — a genuine plateau.

### P1 — Only after P0: measure the TRUE converged 121 fidelity vs Zemax POP (2.74 µm).
- If it converges to ≈ diffraction-limited (matching POP within a few % EE) → **121 production-ready.**
- If it converges to a plateau **below** POP → *then* investigate residual physics (this is the only
  point at which launch/wavefront questions may legitimately re-enter — grounded on a converged
  baseline, never before).

### P2 — Daily-driver generality (needed for "all designs", partly independent of P0/P1).
- **Aperture:beam cliff guard** (§4): beam-relative launch/fit radius + warning.
- **Memory-bounded mode** (§4): resolution budget, graceful degradation, honest non-converged flag.
- **Built-in convergence self-check:** a cheap 2-grid (dx, dx/√2) comparison that flags when a returned
  metric is not dx-stable, so the daily driver never silently returns a non-converged number.
- **Design test battery:** singlet / doublet / triplet / corrected-relay at varied NA and aperture:beam
  ratios, each with reference truth (Zemax POP or analytic), replacing reliance on the 121 alone.

### P3 — Housekeeping.
- **Commit the F-A/F-C/F-D dx-scaling fixes** (verified, still uncommitted; keep separate from the
  reverted Part E). They are the P0 investigation's foundation (F-A already conserves energy at the
  clamp; the §1 bug is elsewhere).

---

## 6. Acceptance criteria for "production ready"

**Design-121:** traced chain (a) **converges** — monotone approach + plateau — under dx refinement on
both axes; (b) **conserves energy** (window ≤ stop losses) at every grid; (c) matches **Zemax POP**
FWHM/EE within a stated tolerance at a **memory-feasible** setting.

**Daily driver (all designs):** (a) robust across the P2 design battery with **no silent cliffs**
(guards fire, memory degrades gracefully, non-converged results are flagged); (b) byte-identical
defaults preserved vs v5.28.0 on the existing pinned suites; (c) a documented **known-good envelope**
(aperture:beam, NA, dx ranges) outside which it warns rather than silently mis-reports.

---

## 7. Reproduction / artifacts for the next box

- **Repro (needs the 121 `.zmx` + `run_poc_119_120_v518.py` Sellmeier registrations):**
  `validation/repro_traced_carrier_121/{carrier_chain_121.py, traced_group_oracle.py, repro_dx_scaling.py}`.
- **Scratch scripts used this session** (single-box temp, NOT in the repo — recreate on the new box):
  a 121 settings sweep over `(n_fine_cap, window_factor)`; an E4 aperture:beam sweep; a 121
  aperture-clamp and a `_CARRIER_FIT_RADIUS_FRAC` sweep. Their *results* are captured in §1–§2 and in
  `AUDIT_WAVEFRONT_AWARE_RAY_LAUNCH_2026_07_23.md` §4b–§4c.
- **Recommended box:** ≥ 64 GB RAM to run the converged N-sweep at full `n_fine_cap`/`window_factor`
  without the memory-bounded mode; otherwise P2's memory-bounded mode is a prerequisite even to
  measure convergence.
- **Suggested first commands on the new box:** (1) reproduce the §1(A) N-sweep to confirm the
  divergence + the >100% energy at large N; (2) run the P0.1 stigmatic control sweep to localize the
  defect; (3) P0.2 per-transform unit-energy audit.

**One-line status:** the traced propagator is architecturally capable (diffraction-limited on a
well-posed relay) and robust at its default settings, but is **NOT production-ready** — its absolute
121 metrics do not converge with resolution and violate energy conservation at fine grids (P0), and it
lacks the aperture-cliff / memory / convergence-self-check guards a daily driver needs (P2). The
launch and aperture hypotheses are **ruled out**; the convergence/energy defect is the target.
