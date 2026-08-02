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

---

## 0. VALIDATION ENVELOPE — where this propagator is actually validated to run

> **MEASURED NOTE (niche D5, 2026-07-29; roadmap P4).** §6's acceptance
> criterion (a) — "converges under dx refinement on both axes" — was closed at
> N = 1024–4096 only. This section closes the envelope where production runs
> and states the large-N position. All rows below are the **shipped defaults**
> (`carrier_reference='sphere'` + `amplitude_model='ray_density'` +
> `preserve_input_phase='remap'` + `remap_sampling='full'`, i.e. no chain
> kwargs at all), design-121 8-group relay via
> `validation/repro_traced_carrier_121/focus_scan_121.py`, NFC = 8192,
> WF = 4.0, readout 2048 x 0.05 µm, through-focus scan, ray pitch held at the
> reference 4 µm on every row.

### 0.1 The dx-flat envelope, INCLUDING the F-B matrix's worst row

| N | launch dx | ray_subsample | best-focus FWHM | EE3 | EE6 | EE12 | window | pre-flip EE6 |
|---|---|---|---|---|---|---|---|---|
| 1024 | 2.0 µm | 2 | 3.5537 µm | 87.80 % | 99.14 % | 99.31 % | 99.44 % | 82.8 % |
| 2048 | 1.0 µm | 4 | **3.4156 µm** | **88.829 %** | **99.583 %** | 99.755 % | 99.788 % | 69.7 % |
| 4096 | 0.5 µm | 8 | **3.4266 µm** | **88.832 %** | **99.580 %** | 99.762 % | 99.796 % | 54.6 % |
| 8192 | **0.25 µm** | 16 | **3.4265 µm** | **88.832 %** | **99.580 %** | 99.762 % | **99.796 %** | **46.5 %** |

**The N = 8192 / dx = 0.25 µm row — the worst row of the original F-B evidence
matrix, which read EE6 46.5 % and FWHM 8.85 µm pre-flip — now reads EE6
99.580 %, EE3 88.832 %, FWHM 3.4265 µm: identical to the N = 4096 row in every
digit and to the N = 2048 acceptance in three.** The convergence claim is
therefore airtight against its own original counter-evidence rather than
against a truncated range. At-plane (no focus scan) the same row reads FWHM
3.6783 / EE3 87.326 / EE6 99.515 / EE12 99.729 / **window 99.796 %**, and
per-stage envelope power is constant to six digits (6.8297e-08 → 6.8172e-08
across the eight groups), so energy is conserved at the finest published grid
— the F-A inflation (130.8 % window at N = 28672) has no residue here.

The 2048-vs-8192 spread is FWHM 0.32 %, EE3 0.003 points, EE6 0.003 points.
`AUDIT_TRACED_CHAIN_DX_SCALING_2026_07_22.md`'s "~15 EE6 points per octave of
dx" systematic is gone: **measured 0.001 points per octave.**

### 0.2 Large N: NOT a supported regime, and why

**Position: the validated envelope is N ≤ 8192, and nothing above it is
recommended.** This is a COST statement, not an accuracy one, and it is the
consumer lesson of this whole study: *grid size stopped being the accuracy
lever.*

* **Measured cost.** The 121 chain costs **82 / 104 / 191 / 556 s** at
  N = 1024 / 2048 / 4096 / 8192 (24-thread box, `n_workers=8`), i.e. ~2.9x per
  octave. At N = 16384 just the **first two of eight groups exceeded 600 s**
  (`n_workers=12`), so a full run is > 40 min — to move a 4-digit-converged
  answer. Memory scales the same way: one complex128 grid is 1.0 GiB at
  N = 8192, 4.0 GiB at 16384 and 12.25 GiB at 28672, and the chain holds
  several.
* **Measured accuracy return.** Zero. Rows 2048 → 8192 agree to four
  significant figures (§0.1).
* **The fine-grid failure mode is fixed and now degrades gracefully.** The
  F-A trigger is exactly `n_crop > n_fine_cap`; at N = 28672 / cap 16384 that
  produced window-total **130.8 %** and EE6 102.3 %. Re-measured 2026-07-29 by
  forcing the same condition cheaply (N = 2048, `n_crop` = 651, cap **512**
  and **256**): window **99.747 %** and **98.832 %**, EE6 99.524 % and
  98.433 %, FWHM 3.6388 and 3.6764 µm — monotone, sub-100 %, with the F-D
  `RuntimeWarning` naming the discarded outer NA. No inflation at any cap.
* **If you must go large, go PITCH-PRESERVING.** Pin `dx` at the reference
  pitch and let N buy guard band. Extent-preserving refinement is the axis
  §0.1 already shows is flat, so it buys nothing at any N.

This position is stated where a consumer reads it, not only here: the
`wavelength, dx` entry of `propagate_traced_carrier_chain`'s docstring.

### 0.3 The CI gate that does not need the `.zmx`

> **REVISED 2026-07-29 after adversarial verification.** The first revision of
> this section described a gate with a *stigmatic-groups* stand-in and an
> "oracle" built from a perfect sphere pushed through the **library's own**
> `carrier_referenced_exact_focus_readout`. Both halves were shown wrong by
> measurement and are replaced below; the numbers in this revision are the
> ones that survived. What was wrong, recorded because each cost a revision:
> (a) the gate had **one tooth** — `preserve_input_phase=True` and `False`
> both passed it with an EMPTY violation list (0.03 EE points) while the same
> flip costs 35.7 EE3 points on the real design 121, because with every group
> individually stigmatic the residual each group hands the next is ~0 and
> `'remap'` coincides with `False` **by construction**; (b) the level half was
> three RATIOS against a reference sharing the chain's last stage, so
> `window_factor` 4.0 -> 3.0 broadened BOTH by 6 % and the gate stayed green;
> (c) the claimed "64 MiB peak" was a single-array estimate — the measured
> peak was 8.9 GiB, unguarded.

`tests/unit/test_niche_d5_dx_flatness_gate.py` (13 tests, **137 s / 3.16 GiB
peak RSS** on Windows, 167 s / 3.34 GiB on the WSL Linux proxy; every
chain-running test RAM-guarded at 4 GiB available) gates the chain on a
**synthetic four-group stand-in with design 121's structure** — small-waist
(4 µm) diverging launch -> collimate -> focus -> re-collimate -> fast final
group, per-group exit NA 0.000 / 0.078 / 0.000 / 0.189 with final-leg
`na_exit` = 0.2021, so it crosses `na_exact_threshold` and takes the SAME
exact high-NA final leg.

**It is a CORRECTED RELAY, not a chain of individually-perfect groups.** G3's
flat exit carries an `r^4` pre-shaping term (2.0 rad at r = w) and G4's flat
entrance carries the compensating term 2 mm downstream in the same collimated
space, so the chain must **hand a large `r^4` residual across a group
boundary** and the final group must consume it — design 121's own situation
(`AUDIT_TRACED_FROZEN_AMPLITUDE_2026_07_24.md` S12: the carried residual is
"the design's own correction, r^4-dominant", 9.2 rad on the 121 final leg).
That single structural change is what gives the gate teeth on
`preserve_input_phase`. The correction is proved in CI by an exact meridional
raytrace through the actual conic + aspheric sags: transverse ray error
<= 0.102 µm out to 1.25 launch radii (0.504 µm at 1.5) against a 2.743 µm
oracle FWHM, while **removing G4's compensating term and nothing else** moves
the same rays to +10.15 µm at 1.0w and +43.1 µm at 1.5w — 100x.

**The level anchor is now INDEPENDENT and ABSOLUTE.** It is
`validation/oracles/debye_oracle_v3.py` — the lumenairy-free oracle already
in the tree (exact meridional raytrace, energy-conserving exit ring measure,
ring-Huygens diffraction integral; validated against the ZOS Huygens PSF to
EE50 1.9 % / EE80 0.72 % by `test_niche_p8_capstone.py`). It shares no
readout, no `window_factor`, no FFT grid and no wave model with the
propagator under test. Two additions were needed and are additive (pre-D5
behaviour unchanged, p8 capstone re-run green): `entrance_eikonal='sphere'`
(the EXACT spherical launch `r_in` denotes under `carrier_reference='sphere'`;
the pre-existing paraxial `h^2/2R` is 1.8 rad wrong at the rim of a +-2.5w
window on a 2 mm conjugate) and `huygens_radial_profile`, which applies the
`2 pi dh / (i lambda)` constant so the profile's own total is a
**measurement** — it comes back 101.907 % of the launched power, the ~1.9 %
being the ring-Huygens kernel's missing obliquity factor at NA 0.2, which is
why the gate renormalises EE by that total.

Gate = **dx-flatness** across N = 512 / 768 / 1024 at pitch-preserving
`ray_subsample` 2 / 3 / 4 (measured spread FWHM **0.0014 %**, EE <= **0.011**
points, window **0.007** points; tolerances 0.5 % / 0.30 / 0.20) **plus an
ABSOLUTE level** against the oracle: FWHM within 2 % (measured **+0.437 %**),
EE2 >= 0.70x and EE4 >= 0.80x the oracle's renormalised value (measured
**0.790** / **0.867**), window >= 99.0 % **of the launched power** (measured
99.883 %).

Measured rows, shipped defaults, N = 512 / 768 / 1024:

| metric | 512 | 768 | 1024 | oracle |
|---|---|---|---|---|
| FWHM (µm) | 2.75530 | 2.75528 | 2.75526 | **2.74329** |
| EE1 (%) | 25.6198 | 25.6207 | 25.6247 | 32.5603 |
| EE2 (%) | 63.8660 | 63.8677 | 63.8767 | 80.9036 |
| EE4 (%) | 86.6129 | 86.6148 | 86.6164 | 99.9624 |
| window (%) | 99.8887 | 99.8908 | 99.8834 | — |

Teeth, from the SAME `dx_flatness_gate()` function (N = 1024) — every one
trips all four level terms:

| reverted | FWHM (µm) | EE2 (%) | window (%) |
|---|---|---|---|
| `carrier_reference='parabola'` | 10.0622 | 6.189 | 75.085 |
| `preserve_input_phase=True` | 3.3702 | 53.859 | 98.014 |
| `preserve_input_phase=False` | 4.1625 | 39.906 | 90.096 |
| `final_leg='paraxial'` | 8.5848 | 8.437 | 82.082 |
| full legacy configuration | 10.5320 | 6.132 | 82.780 |

Four results worth carrying out of building it:

1. **dx-flatness ALONE is not a sufficient gate.**
   `carrier_reference='parabola'` — the pre-v5.29 default, the single flip the
   121 acceptance turns on — is dx-**flat** to 0.005 % on the stand-in while
   sitting **3.67x** wide of the independent oracle. A flatness-only gate —
   which is all the shipped `self_check='dx'` is, on a coarser metric set —
   passes it silently. The absolute level half is what has the teeth, and it
   is pinned as such.
2. **A stand-in whose groups are each individually perfect cannot test
   `preserve_input_phase` at all.** Measured, on the discarded revision:
   `preserve_input_phase=True` moved EE2 by 0.03 points and `False` by 0.45
   (EE2 went *up*), i.e. the second leg of the v5.29 validated triple could be
   reverted silently. The mechanism is stated in the element docstring itself
   — "for a pure carrier-sphere input the residual is ~0 and `'remap'`
   coincides with `False`" — so a gate for that flip **must** carry a residual
   across a group boundary. On the corrected relay the same two flips cost
   23 % and 52 % of the FWHM.
3. **`remap_sampling` still has no teeth here, and the reason is
   quantitative.** Per S12 the lattice route aliases the transported residual
   only outside `r_alias = (pi w^4 / (4 A h))^(1/3)`; at A = 2.0 rad,
   w = 0.9665 mm and ray pitch h = 7.6 µm that is 3.6 mm = 3.7w, far outside
   the beam. Reaching 1.5w needs A ~ 30 rad, which on this geometry drives the
   fast group's ray-FIT so deep into the aperture:beam cliff that the defaults
   stop tracking the oracle at all (measured at A = 8 rad: EE2 21.4 % against
   an oracle 74.3 %). Recorded on `test_remap_sampling_has_no_teeth_here`
   rather than assumed covered.
4. **A NEW measured library finding — see S0.4.**

Cross-platform: the whole ladder re-run on WSL Ubuntu-24.04 (CPython 3.12.3,
numpy 2.4.6, scipy-openblas 0.3.31 SkylakeX) vs Windows (CPython 3.14.6,
numpy 2.4.4) agrees to **<= 4e-6 µm on FWHM and <= 5e-5 points on EE** over
all rows — ~200x below the dx spread the gate tolerates — and the oracle
itself is identical to every printed digit (it is pure numpy +
`scipy.special.j0`).

### 0.4 NEW FINDING (D5, 2026-07-29): the ray-FIT cliff is NOT closed at a fast, CARRIER-FREE group

Building 0.3's absolute anchor surfaced a real accuracy limit that no existing
test covers, because no existing test compared a traced *exit wavefront*
against an independent one at exit NA >= 0.15.

**Minimal repro (element level, no chain, in CI as
`test_the_level_gap_is_the_traced_fit_radius_cliff`):** ONE plano-convex conic
singlet, `K = -n^2`, f = 4.83 mm, COLLIMATED Gaussian input, exit NA 0.20.
Because the singlet is stigmatic the exact exit wavefront is a SPHERE centred
on the focus (Fermat) — the truth needs no diffraction model at all. Measured
exit-wavefront error at r = w:

| route | error at r = w |
|---|---|
| `apply_real_lens_traced`, no fit restriction | **4.428 rad** |
| ... `fit_radius_beam_factor=2.0` (the chain's validated default) | **1.122 rad** |
| ... `fit_radius_beam_factor=1.5` | 0.087 rad |
| `apply_real_lens_gbd` (independent propagator) | **0.031 rad** |

So the S4 aperture:beam cliff guard is **not sufficient at its shipped chain
default** when the group is fast AND its input is collimated: no carrier can
engage (a collimated `carrier=inf` eikonal is NaN — S4's own corrected
mechanism), so the OPL fit has to represent the whole exit sphere instead of a
small residual, and `fit_radius_beam_factor=2.0` still admits ray heights
whose out-of-basis high order corrupts the fit inside the beam. The
docstring's "the recovery is flat for 1.5-2.5" is an E4-case statement, not a
general one: here 2.0 leaves 1.12 rad where 1.5 leaves 0.087.

Consequences, all measured on the D5 stand-in:

* end-to-end it costs **~15 % of the peak intensity** and **~11 EE2 points**
  (chain 63.88 % against an independent-oracle 80.90 %), while the FWHM is
  barely touched (+0.44 %) — i.e. it is a halo, not a broadening: the chain's
  radial profile carries a shelf at I/I0 ~ 1.5e-2 between r = 3 and 7 µm where
  the oracle is at 3e-4;
* it is **dx-INDEPENDENT** — 1.240 / 1.238 / 1.233 rad at N = 1024 / 2048 /
  4096, unchanged at `ray_subsample` 1 vs 4, and unchanged at `n_fine_cap`
  2048 / 4096 / 8192 — which is the sharpest available statement of why a
  flatness-only gate cannot see it;
* the exit **amplitude** is right to 0.1 % out to 1.5w; only the phase is
  wrong, so an energy-conservation check cannot see it either;
* it scales roughly as NA^5 (1.238 / 0.276 / 0.050 / 0.031 rad at exit NA
  0.20 / 0.15 / 0.10 / 0.05), so it switches on essentially where
  `na_exact_threshold` = 0.15 does.

**Not a regression** (it is the pre-existing behaviour of the shipped
default), and **design 121 is not obviously exposed** — its final group is fed
by a non-collimated beam, so a carrier does engage there and the fit is
referenced to a small residual, which is consistent with the 121 acceptance
sitting at its measured ideal-field ceiling. But "a fast final group after a
collimated space" is the single most common architecture in free-space relay
design, so this belongs on the roadmap. Suggested consumer-side mitigation
today: pass `traced_kwargs={'fit_radius_beam_factor': 1.5}` when the last
group is fast and its input is collimated, and cross-check the exit wavefront
against `apply_real_lens_gbd`.

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
