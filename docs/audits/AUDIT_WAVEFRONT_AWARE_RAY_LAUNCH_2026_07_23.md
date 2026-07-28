# Audit / scoping — wavefront-aware ray launch for the traced-carrier chain (Part E)

**Date:** 2026-07-23 · **Library state:** v5.28.0 (main @ `9011211`, dx-scaling audit F-A/F-C/F-D remediated)
**Scope:** `apply_real_lens_traced`'s ray-launch model (`lumenairy/elements/_lens_traced.py`),
`propagate_traced_carrier_chain` (`lumenairy/propagators/carrier.py`).
**Parent docs:** `docs/audits/AUDIT_TRACED_CARRIER_CHAIN_2026_07_21.md` (F1–F4, R9 addendum — diagnosis
origin), `docs/roadmap_deferred_2026_07_21.md` Part E (the roadmap item this audit scopes into an
implementation plan), `docs/audits/AUDIT_TRACED_CHAIN_DX_SCALING_2026_07_22.md` (F-A/F-C/F-D, a
separate and already-closed numerical issue on the same code path — see "Relationship to F-B"
below for how the two are distinguished).

## 0. Why this audit exists

R7 (commit `14d737e`) made every one of the 121's 8 groups individually exact against the ray
oracle (< 0.023 rad rms). R9 (commit `91fb5ec`) made the final high-NA (0.46) leg exact against its
own diffraction limit. Both are shipped and verified. And yet the end-to-end 121 chain still only
reaches **EE6 = 69.7%** against a **99% acceptance target** (`docs/roadmap_deferred_2026_07_21.md`
E4). The R9 addendum isolated *why*: the field entering the high-NA tail already carries **~1.68
rad RMS** of wavefront aberration that is not attributable to the final leg, the gap transport, or
sampling — it is *accumulated by the chain through the individually-clean front groups*. This
document scopes the fix for that accumulation.

## 1. Root-cause mechanism (traced to source)

### 1.1 The three ray-launch branches that exist today

`apply_real_lens_traced` (`_lens_traced.py:1587`) launches a coarse grid of rays from the entrance
plane and Newton-inverts their traced OPL to build the exit wavefront. The **direction cosines**
`(L, M)` each ray is launched with come from exactly one of three sources, selected at
`_lens_traced.py:2893-2960`:

```
if _tilt_aware_launch:                       # (A) per-pixel field-gradient launch
    L_in, M_in = _sample_local_tilts(E_in, wavelength, dx, Xs_in, Ys_in)
elif _carrier_grad is not None:              # (B) analytic carrier-sphere-gradient launch
    L_in, M_in = _carrier_grad(h_x, h_y)
else:                                        # (C) plane-wave launch (L=M=0)
    ...
```

* **(A) `_sample_local_tilts`** (`_lens_traced.py:1233-1390`) reads the ray direction straight off
  `E_in`'s own local phase gradient (`angle(E[i+1]·conj(E[i]))/dx`), amplitude-weighted-Gaussian-
  smoothed (`sigma=4 px`) to survive multi-mode / DOE fields. This *is* a wavefront-aware launch —
  but it reads the gradient **numerically**, off a discretized field, which aliases whenever the
  local tilt approaches the grid's Nyquist tilt `lambda/(2 dx)`. F3 (below) is this failure mode.

* **(B) `_carrier_grad`** (`_compute_carrier`, `_lens_traced.py:1035-1230`; scalar-conjugate case at
  `1216-1230`) returns the **exact analytic** gradient of the carrier sphere
  `S(R) = sign(R)(sqrt(r²+R²) − |R|)` — alias-free by construction, because it is a closed-form
  expression, not a finite difference of a discretized field. This is why R7 could push per-group
  rms below 0.023 rad: on a *clean sphere* input, (B) is exact.

* **(C)** is the historical plane-wave default (L=M=0), correct only for a collimated input.

### 1.2 F3 — why (A) is currently guarded off whenever an explicit carrier is set

`_lens_traced.py:2460-2489` (flag `_F3_GUARD_TILTAWARE_EXPLICIT_CARRIER`, `:896`): when the caller
passes **both** `tilt_aware_rays=True` **and** an explicit engaged `carrier=`, the per-pixel launch
(A) is measured to be *five times worse* than the carrier-gradient launch (B) alone (1.72 rad vs
0.008 rad rms on the 121's S5-S7 triplet — `AUDIT_TRACED_CARRIER_CHAIN_2026_07_21.md` F3). Today's
fix is a **downgrade**: silently re-route to (B) and warn. This is correct as far as it goes — (B)
alone is more accurate than (A) alone on a steep sphere — but it throws away exactly the
information (A) was trying to supply: any part of the wavefront that is **not** the sphere.

### 1.3 The gap: (B) discards the residual a corrected relay legitimately carries

`propagate_traced_carrier_chain` (`carrier.py:2207-2210`) calls
`apply_real_lens_traced(carrier=R_use, ...)` at every group **without** `tilt_aware_rays` (default
`False`), i.e. it always uses branch (B) alone. That is the right choice *for a single group fed a
clean sphere* (R7's own oracle input). But inside the chain, the field entering group *k* is not a
clean sphere — it is group *k−1*'s actual exit field, propagated through free space
(`propagate_carrier_referenced`) and reconstructed
(`carrier_referenced_reconstruct`) back to a full field. A **corrected relay** — the 121 is one —
deliberately pre-shapes non-spherical aberration in an early group that a later group is designed
to cancel. That non-spherical part is real, physical, and present in `E_in` at every group after
the first. Branch (B) launches every ray along the sphere gradient regardless, so the pre-shaped
aberration never enters the ray trace — the tail's designed cancellation has nothing to act on, and
the mismatch (the sphere vs. the true local wavevector) is what the R9 addendum measured as ~1.68
rad RMS accumulating through the low-NA front groups, each of which is *individually* clean because
each is validated with a *clean-sphere* oracle input that has no residual to discard in the first
place.

**This is a distinct defect from F1/F2/F3/F4/R9** — none of those touch what information the ray
launch is built from; they fix the accuracy of the sphere-only launch (F1: recovering the sphere at
all; F2: intra-group fit conditioning; F3: guarding against a worse alternative; R9: the final
leg's sphere is the wrong reference at high NA). Part E is the first item that says the sphere
itself is an incomplete basis for a corrected relay's ray launch.

### 1.4 Relationship to F-B (dx-scaling audit — CLOSED 2026-07-24, see note at end of §2)

`AUDIT_TRACED_CHAIN_DX_SCALING_2026_07_22.md` F-B observed that the 121's *absolute* focal metrics
(EE3/EE6/FWHM) do not converge with grid pitch — they drift by ~15 EE6 points per octave of `dx`,
even at fixed `n_fine_cap`. F-B is a **numerical (resolution-dependent)** symptom; Part E is a
**modeling (resolution-independent)** gap — the ~1.68 rad residual is present at any dx fine enough
to resolve the front groups, because it comes from discarding real field content, not from
under-sampling it. The two may interact (an under-resolved residual-gradient extraction would add a
*further* numerical error on top of the modeling gap — see §3.3), but they are separate root causes
and neither's fix subsumes the other. Both must be considered before the 121's absolute numbers are
fully trustworthy.

## 2. The fix — launch along the actual wavefront, not the sphere alone

### 2.1 Chosen approach: carrier-relative residual launch (roadmap E2, candidate 1)

Keep the carrier as the **frame** (unchanged: co-moving grid, envelope/reconstruct, R7's intra-group
fit-domain restriction, R9's exact final leg). Change only the **ray-launch direction** at each
group to:

```
L_in, M_in = L_carrier(h_x, h_y) + L_resid(h_x, h_y),  M_in analogous
```

where `(L_carrier, M_carrier) = _carrier_grad(h_x, h_y)` (existing, exact, branch B unchanged) and
`(L_resid, M_resid)` comes from the **de-chirped** field:

```
E_resid = E_in * exp(-i k0 W_carrier)          # W_carrier = _carrier_W, the (N,N) sphere array
L_resid, M_resid = _sample_local_tilts(E_resid, wavelength, dx, Xs_in, Ys_in)
```

This is mathematically `grad(unwrapped total phase) − grad(carrier sphere)`, but it is obtained
*without ever unwrapping the raw (fast, possibly-aliased) field* — the fast spherical part is
subtracted **analytically** (exact, no discretization) before any numerical gradient is taken, so
the numerical gradient only ever has to resolve the **slowly-varying residual** left behind. This
is precisely why it sidesteps F3: for a corrected relay, the residual is by design much gentler
than the full curvature (that is what "pre-correction" means), so `_sample_local_tilts` — already
built with amplitude-weighted smoothing for exactly this kind of robustness — operates in its safe
regime instead of the aliased regime that broke it in F3.

Reused as-is, unmodified: `_carrier_grad` (branch B), `_sample_local_tilts` (branch A's engine),
`_carrier_W` (already computed and held in scope at the launch site, `_lens_traced.py:2396-2448`).
No new numerical primitive is required — this is a **recombination** of two already-shipped,
already-tested pieces.

### 2.2 Where this replaces today's F3 downgrade (byte-identical-by-construction)

The natural home for this is exactly the branch F3 currently guards: `tilt_aware_rays=True` +
explicit engaged carrier. Re-reading that combination as "launch along the actual wavefront,
carrier-relative" rather than "downgrade to the carrier alone" turns F3 from a guard into a
**unification** (this is the roadmap's own framing). Concretely, at `_lens_traced.py:2477-2489`:

* **Before:** `tilt_aware_rays=True` + explicit carrier → `_tilt_aware_launch = False`, warn, fall
  through to branch (B) alone.
* **After:** `tilt_aware_rays=True` + explicit carrier → new branch: `L_carrier + L_resid`.

Every other combination is untouched:

| `tilt_aware_rays` | carrier | branch | behavior |
|---|---|---|---|
| `False` | any | (B) or (C) | **unchanged** — chain default stays byte-identical |
| `True` | `None` / `'auto'` (W~0) | (A) | **unchanged** — N5 path, per-pixel on raw field |
| `True` | explicit, engaged | **NEW** | was (B)-only-with-warning; now `L_carrier+L_resid` |

This gives the "byte-identical defaults" property the roadmap requires (E4) for free, by
construction, rather than by a separate pinning effort: nothing changes unless a caller opts in
with *both* `tilt_aware_rays=True` *and* an explicit `carrier=`. `propagate_traced_carrier_chain`
currently never sets `tilt_aware_rays` (defaults `False`, `carrier.py:2209`), so the chain's default
behavior is unaffected until a new chain-level opt-in (§2.3) is exercised.

### 2.3 Chain-level plumbing

`propagate_traced_carrier_chain` gains one new keyword, `wavefront_aware: bool = False`. When
`True`, every per-group `apply_real_lens_traced` call (the standard-leg call, `carrier.py:2207-2210`,
**and** the R9 fine-retrace call inside `_fine_trace_group_exit`, `carrier.py:1983-1986`) receives
`tilt_aware_rays=True` in addition to its existing `carrier=R_use`/`carrier=R_in`, engaging §2.2's
new branch at every group. Default `False` keeps the chain's existing behavior byte-identical
(nothing in `propagate_traced_carrier_chain`'s call sites changes unless the caller passes
`wavefront_aware=True`).

### 2.4 No new per-group grid — use the existing one, instrumented

E3 in the roadmap flags that the residual may need a finer grid than the sphere did ("must
Nyquist-sample the actual local frequency, which can exceed the sphere's"). Rather than assume this
and build new machinery pre-emptively, the fix **measures** it: after computing `(L_resid,
M_resid)`, check the residual's own tilt magnitude against the *entrance* grid's Nyquist tilt
(`lambda/(2 dx)`, the same quantity R6's `_AUTO_CARRIER_NYQUIST_FRAC` check already uses for the
auto-carrier fit) and warn (not silently degrade) if a non-trivial fraction of the bright support is
at risk of aliasing in the residual extraction — mirroring the existing diagnostic idiom
(`_lens_traced.py:3055`'s exit-sampling warning, R9's `_fine_trace_group_exit` Nyquist warning). If
this fires on the real 121 chain, R9's fine-retrace pattern (resample onto a finer grid before
extracting the residual) is the documented escalation path, **not** built speculatively in this
pass — this keeps the change minimal and lets the acceptance run (§4) tell us whether it is needed
in practice before adding complexity for a case that may not occur.

### 2.5 What is explicitly OUT of scope for this pass

* **`carrier='auto'` and ndarray-carrier residual launch.** The mechanism (§2.1) generalizes
  directly (`_carrier_W` and `_carrier_grad` exist for all three `_compute_carrier` branches), but
  the acceptance target (E4) is the scalar-conjugate chain the 121 actually uses. Wiring the other
  two is low-risk follow-on, not blocking.
* **F-B (dx-convergence).** ~~Left open per the dx-scaling audit's own conclusion; see §1.4.~~
  **SUPERSEDED (2026-07-27):** F-B was root-caused and closed the day AFTER this audit was
  written — see `AUDIT_TRACED_FROZEN_AMPLITUDE_2026_07_24.md` §2-§4 (frozen intra-group
  amplitude + the `preserve_input_phase=True` analytic-pair phase corruption) and §6.7
  (dx-flat by configuration), plus `0a743a6` (coarse->fine upsample lattice bug). Closed by
  the v5.29 default flip, never under the label "F-B". See §1.4 note.
* **Roadmap Parts A/B/C** (caustic coverage, perf/memory, GPU, validation infra) — unrelated to this
  gap.

## 3. Risks

### 3.1 No double-counting

The residual must vanish (to floating-point noise) whenever `E_in` truly is the analytic carrier
sphere with nothing added — the R8 regression suite's `_TRIPLET` tests
(`test_r8_f3_tiltaware_field_equals_carrier_path`, `test_r8_f3_fail_before_pass_after`) construct
exactly this input (`env * exp(i k0 S(R_in))` with `S` the identical formula `_compute_carrier`
uses), so they are the direct check: `E_resid = E_in * exp(-i k0 W_carrier)` collapses to the real,
positive envelope `env` with a phase gradient of order machine-epsilon, and `L_resid, M_resid ≈ 0`
to well under any test's tolerance. This must be verified by *running* the existing R8 suite after
the change (not just argued), since it is the guard against silently reintroducing F3's degradation
under a new name.

### 3.2 Warning-message semantics change

The F3 RuntimeWarning (`_lens_traced.py:2481-2489`) currently frames the reroute as a downgrade
("... is less accurate ... Drop tilt_aware_rays=True ..."). Once the combination is the *more*
accurate, *recommended* path for a corrected relay, that message is actively wrong and must be
rewritten (or the warning dropped in favor of only firing on genuine risk, §2.4). `test_r8_f3_guard_warns`
currently pins the old wording's presence and will need updating to match — this is a deliberate,
reviewed change to test intent, not an accidental break, and will be called out explicitly when
touched.

### 3.3 Interaction with R7's fit-domain restriction and R9's exact final leg

R7 restricts the entrance→exit Chebyshev fit to `r <= 0.5 * launch_radius` when a carrier is set
(`_CARRIER_FIT_RADIUS_FRAC`, `:956`) — that restriction is on the *fit domain*, not the launch
directions, so it is orthogonal to this change (the new `(L_in, M_in)` values feed the same launch
grid the fit already restricts). R9's exact final leg consumes the group's *exit* field, which is
downstream of this change and unaffected by it directly — but if the residual launch changes the
final leg's *entrance* wavefront (it should, materially, since that is the entire point), the R9
`_fine_trace_group_exit` Nyquist-of-the-exit-sphere check (independent of this change) still applies
unchanged; no interaction beyond both being active on the same call is expected, but this is a
verify-don't-assume item for the acceptance run.

### 3.3b Implementation finding: `phase_analytic_lens`'s reference stays sphere-only (2026-07-23)

§2.1/§3.1 originally left open whether `phase_analytic_lens`'s reference input
(`_reference_input()`, `_lens_traced.py:2585`) should also switch from the pure
carrier sphere to the field's own (carrier+residual) phase, to stay formally
consistent with the new ray-launch congruence. This was implemented and then
**reverted** after empirical testing: making the reference input-dependent
(`E_in / |E_in|`) regressed a synthetic pre-shaped-residual probe by 2-5x,
because that reference is deliberately **unit amplitude** (unlike `E_analytic`,
which uses the true, naturally-decaying `E_in`) — an input-dependent reference
extrapolates whatever residual phase `E_in` carries out to the full grid at
full amplitude, where a residual that is gentle near the beam can be
numerically wild far from it, aliasing `phase_analytic_lens`'s own analytic-
model ASM pass even though that region carries negligible true energy. The
pure-sphere reference has no such failure mode (analytic and smooth at every
radius by construction). **Kept sphere-only**, matching the more conservative
original scope ("change only the ray-launch direction") — the residual enters
solely via the ray-launch gradient, not via the analytic-phase reference. This
means the fix is, like `tilt_aware_rays`'s existing plane-wave case, only
**approximately** consistent for a genuinely large residual — appropriate for
a corrected relay's inter-group residual, which is gentle by construction, but
worth remembering if this is ever revisited for a large-residual regime.

A related, harder finding from the same probe: a **single-group** synthetic
test that injects an arbitrary polynomial residual phase directly into `E_in`
and compares against an exact meridional-ray oracle does **not** cleanly
isolate this mechanism — even a tiny injected residual (tilt ≪ Nyquist)
measurably degrades the sphere-only baseline well beyond R7's clean-sphere
floor, and the new ray-launch term tracks that degraded baseline almost
exactly rather than correcting it. This is consistent with the residual's
effect on `E_analytic`'s own analytic-model propagation (a mechanism the
ray-launch fix does not and should not touch) dominating over the geometric
ray-launch effect at the single-group, arbitrarily-injected-residual scale.
The mechanism this audit targets is the **chain-accumulated** residual from
real upstream groups (R9's ~1.68 rad/rms measurement), not an arbitrary
injected term — so acceptance evidence is a **multi-group** chain (synthetic
corrected-relay, §4.1) and the **real 121** (§4.2), not a single-group probe.

### 3.4 Incoherent / multi-mode fields

`_sample_local_tilts`'s existing incoherent-field caveat (documented at `_lens_traced.py:1251-1285`
and surfaced via `_input_tilt_stats`'s `coherence_ratio` diagnostic, `:1029`) applies identically to
the residual field — a post-DOE / multi-beam residual has no single well-defined local direction,
and amplitude-weighted smoothing degrades gracefully toward zero mean rather than reporting a
meaningful tilt. This is an existing, already-documented limitation of the reused primitive, not a
new one introduced here; no special-casing is added beyond what §2.4's aliasing check already
covers.

## 4. Acceptance criteria (roadmap E4)

1. **Design-agnostic synthetic (generality gate).** A purpose-built 2–3 group synthetic corrected
   relay — a front group that deliberately over-corrects (introduces controlled non-spherical
   wavefront error) and a tail group whose prescription is chosen to null it — validated against an
   inline exact meridional-raytrace + eikonal oracle (no `.zmx`, CI-safe, mirroring the R8/R9 test
   style already in `tests/unit/`). Must show: (a) the OLD sphere-only launch (`wavefront_aware=False`
   / `tilt_aware_rays=False`) leaves the relay measurably aberrated (the fail-before), and (b) the
   NEW carrier-relative launch focuses it back toward the diffraction limit (the pass-after) to a
   quantified rms/Strehl/EE gate, analogous to R7's and R9's own oracle gates.
2. **121 end-to-end (acceptance instance).** `propagate_traced_carrier_chain(..., wavefront_aware=True)`
   on the real 121 (`validation/repro_traced_carrier_121/`) measurably improves EE6 over the
   `wavefront_aware=False` baseline (69.7% at full resolution per the R9 table). The roadmap's stated
   target is **EE6 ≥ 99% at ~2.9 µm**; this pass reports the actual number obtained, honestly,
   whether or not the target is fully reached — per §2.4, a residual-aliasing risk not anticipated
   here could cap the improvement, and that would be reported as a follow-on rather than hidden.
3. **Byte-identical defaults.** All existing pinned tests (R6/R7/R8/R9, dx-scaling F-A/F-C/F-D, the
   wider carrier-referenced suite) pass unchanged — guaranteed by construction (§2.2's table) and
   verified by running the full suite, not assumed.

## 4b. RESULTS — the approach does NOT meet acceptance (2026-07-23)

**Verdict: carrier-relative residual launch (E2 candidate 1) is implemented, safe, and
byte-identical by default, but it does NOT close Part E — and on the real 121 it makes the result
substantially WORSE. Part E remains OPEN.** All numbers below were executed, not estimated.

### 4b.1 Byte-identical default — CONFIRMED (including on the real 121)

- Full unit suite: 6332 passed; the 11 failures are pre-existing environment/flaky items
  (missing `filelock`, h5py metadata, cache-FIFO, FFT-infra global-state ordering, PMM/RCWA
  energy-closure) verified to reproduce on a clean `git stash` of this diff — none in the
  traced/carrier paths.
- Real 121, `wavefront_aware=False`: EE3 32.0% / **EE6 50.0%** / EE12 53.3% / FWHM 5.35 µm /
  window-total 58.2% at (N=2048, ray_subsample=4, n_fine_cap=2048, window_factor=2.0). This
  **reproduces the prior F-A/F-C/F-D campaign's number to the digit** (EE6 50.0%), proving the
  default path is unchanged on the real design.

### 4b.2 Acceptance criteria — NOT met

1. **E4 synthetic generality gate — FAIL (honest xfail).** The subagent-built
   `tests/unit/test_niche_e4_corrected_relay_oracle.py` constructs a genuine 2-group corrected
   relay (inline meridional-ray oracle: G1-alone Strehl ~0.2 → G1+G2 Strehl **1.00**, a real
   diffraction-limited correction). Through the wave chain, EE-in-3-Airy is ~0.08 for BOTH
   `wavefront_aware=False` and `=True` (vs ~0.9 for a diffraction-limited control on the same
   grid) — the mechanism is demonstrably **active** (group-2 exit field changes ~98% between the
   two launches) but produces **no focus improvement**. Encoded as a `strict=False` xfail
   (`test_e4_pass_after_reaches_diffraction_limit`) so a future fix XPASSes and surfaces.
2. **121 end-to-end acceptance instance — REGRESSION.** `wavefront_aware=True` on the real 121:
   EE3 4.8% / **EE6 10.0%** / EE12 24.8% / FWHM 5.85 µm — i.e. EE6 **50.0% → 10.0%**, a large
   degradation, not the hoped-for improvement toward 99%. The E2 residual-aliasing diagnostic
   fired **0** times, so the coarse-grid residual extraction was not self-flagged as aliasing (a
   likely under-warn — the RMS-vs-Nyquist test is dominated by the gentle amplitude-weighted core
   while the r⁴-steep beam edge, downweighted, is where extraction degrades; see §2.4).

### 4b.3 Why it fails (root cause — H1 residual-aliasing CONFIRMED; H2 double-count DISPROVED)

Independent adversarial static review (2026-07-23) confirmed **no implementation bug** (guard,
de-chirp signs/units/indexing, gradient additivity, scope, crash-safety all verified clean). The
review raised two candidate explanations — (H1) extraction noise/aliasing and (H2) a phase
double-representation in the exit assembly. A follow-up **controlled single-group oracle
experiment** (2026-07-23; the diagnostic script was removed together with the reverted
`wavefront_aware` code — its numbers are recorded here) then **distinguished them empirically —
H1 is the mechanism, H2 is disproved:**

- **H2 (double-count) — DISPROVED.** On a single group fed `sphere + gentle quartic residual`
  on a FINE grid (residual edge-tilt 0.012 of Nyquist), the shipped `wavefront_aware` path
  reproduces the exact meridional-ray oracle to **0.0013 rad rms** — essentially perfect, only
  ~2× the (0.0006 rad) sphere-only error, NOT the ~2×-*residual* error a real double-count would
  produce. The symbolic thin-screen argument that suggested the residual is added twice (once via
  `E_analytic`, once via `opl_traced`) does **not** hold materially — it misses that `opl_traced`
  and the reference cancel through the actual ray mapping. **No catastrophic double-count.**
- **H1 (residual-aliasing) — CONFIRMED.** Sweeping residual steepness at a fixed fine grid: as the
  residual edge-tilt climbs 0.012 → 0.041 → 0.122 → 0.304 → 0.812 of the grid Nyquist tilt, the
  `wavefront_aware` exit-rms **explodes 0.0013 → 0.0096 → 0.073 → 0.92 → 1.75 rad**, while
  sphere-only degrades gracefully (**0.0006 → 0.021 → 0.33 rad**). The numerically-extracted
  residual gradient aliases as the residual approaches Nyquist and corrupts the launch; sphere-only
  is robust *precisely because* it never touches the field's numerical phase (clean analytic
  gradient, carries a gentle residual "for free" via `E_analytic`). The real 121 sits squarely in
  the failure regime — steep per-group residual (R9-addendum a₄w⁴ ≈ +27…+44 rad) on a coarse
  dx≈16–26 µm grid — hence the EE6 50%→10% regression.

**Corrected conclusion:** the approach is **not** wrong and the code is **not** buggy — it is
*accurate when the residual is Nyquist-sampled* and *fails by aliasing when it is not*. The §2.4
escalation (per-group fine-grid residual extraction, R9's `_fine_trace_group_exit` pattern applied
at *every* group, deliberately scoped OUT of this pass) is therefore **required, not optional** to
give the 121 a fair test — a reasonable minimal-first-pass omission the acceptance run showed to be
load-bearing.

**Caveat that tempers the upside (verified, same experiment):** even well-sampled, `wavefront_aware`
is *marginally worse* than sphere-only (0.0013 vs 0.0006 rad), because sphere-only already carries a
gentle residual accurately via `E_analytic` (true `E_in`) and adds no launch noise. So it is **not
established that a residual-aware launch beats sphere-only even in its good regime.** The R9-addendum
premise that sphere-only "discards" the residual is only clearly true for *steep* residuals — which
is exactly where extraction aliases — so whether fine-grid extraction yields a *net win* on the 121
is genuinely open; the accumulated 1.68 rad may have a different origin (envelope/reconstruct
hand-offs, paraxial gap transport) that a launch fix cannot touch. **The follow-on must re-verify
the R9-addendum attribution, not assume it.**

Diagnostic-warning note (review Item 6): the E2 residual-aliasing warning is a **weak lower-bound
alarm**, not a guarantee — it fired correctly in the sweep at 0.30–0.81 of Nyquist but fired **0**
times on the 121 despite the regression, because it is built on `_input_tilt_stats`'s *wrapped*
nearest-neighbour estimator (a residual steep enough to alias folds back to a small measured rms —
"you cannot measure aliasing with the aliased gradient"). A robust version needs an unwrapped /
multi-scale local-frequency estimate.

### 4b.4 What a working Part E likely needs (for the next campaign)

A literature / prior-art search (2026-07-23; primary sources below) independently corroborates the
H1 diagnosis: the specific thing this pass does — **finite-differencing a discretized, aliasable
residual to obtain launch directions — has essentially no advocate in the primary literature.**
Every mature hybrid ray+wave method either derives local directions/curvatures **analytically from
the ray trace** (differential ray tracing / ABCD per beamlet) or **localizes** the field into
windows/beamlets so each piece's residual is band-limited under a simple analytic (linear/quadratic)
reference — never a numerically-differentiated global grid. Concrete directions, roughly in
increasing effort:

- **(a) Keep the single co-moving grid but swap the transform** for a semi-analytical / pointwise
  Fourier transform that tolerates a steep smooth residual: extract a quadratic term analytically
  (semi-analytical FT, *exact*), or represent the residual phase in Zernike+spline and sample the
  *phase* not the phase-factor (pointwise FT). Lowest-disruption; works while the residual stays
  single-valued (likely up to the last group). *Wyrowski & Kuhn 2011; Wang et al., Opt. Express
  27:15335 (2019) & 2020.*
- **(b) Per-group fine-grid residual extraction** (R9's `_fine_trace_group_exit` pattern at *every*
  group) so the residual gradient is Nyquist-sampled before it is taken — the §2.4 escalation, now
  indicated as necessary not optional. Cheapest change to *this* code, but only raises the Nyquist
  ceiling; it does not change the fundamentally fragile "difference a sampled phase" strategy.
- **(c) Localized / parabasal or Gaussian-beamlet launch with ray-derived analytic references** —
  the literature's consensus architecture: replace the one global carrier sphere with many local
  references (linear/quadratic), each set by a *ray-traced* direction/curvature (differential-ray
  ABCD), so each window/beamlet's residual is band-limited by construction and no global numerical
  gradient is taken. *Asoubar et al., Opt. Express 20(21) (2012) — parabasal decomposition;
  Greynolds SPIE 9293 (2014) & Ashcraft/Douglas arXiv:2310.20026 (2023) — Gaussian beamlet
  decomposition.* NB: lumenairy already has a GBD path, and the parent audit
  (`AUDIT_TRACED_CARRIER_CHAIN_2026_07_21.md`) records that GBD `reexpand='auto'` recovered only
  ~2.4% chain power at 121 conditions — consistent with the literature's caveat that beamlets
  decorrelate through a long relay and need costly intermediate re-decomposition. So GBD is the
  right *architecture* but not turnkey at 121 conditions either.
- **(d) Extended-Nijboer–Zernike semi-analytic evaluation for the final high-NA focus** — compute
  the near-focus field from the exit-pupil field expressed in Zernikes, with no gridding of the
  steep focusing phase. A clean, aliasing-free endpoint regardless of how the intermediate
  propagation is done. *van Haver & Janssen, JEOS-RP 8:13044 (2013).*
- Signal-processing note: where a local frequency *must* be estimated numerically, Wigner/Husimi /
  windowed-Fourier estimators are the robust replacement for finite differencing — but none can
  recover a frequency genuinely above the grid Nyquist (*Matsushima & Shimobaba, Opt. Express
  17:19662 (2009)* for the band-limit), so (a)/(c)'s analytic de-chirping or localization is
  unavoidable. **The first task of the next campaign is to re-verify the R9-addendum attribution**
  (is the 1.68 rad really a launch-discarded residual, or an envelope/reconstruct/gap-transport
  effect a launch fix cannot touch — see §4b.3 caveat) before investing in (a)–(d).

The implemented `wavefront_aware` path is retained as **opt-in, default-off, experimental
infrastructure** for that follow-on (the plumbing, the diagnostic, and the fail-before/mechanism-
active/negative-result tests are the substrate), and is documented in the API and CHANGELOG as
NOT recommended / not currently meeting its goal. Reverting it entirely is a reasonable
alternative the author may choose — the negative finding is fully captured by this section plus
the E2/E4 tests regardless. **(SUPERSEDED 2026-07-24 by §4c — the code was reverted.)**

## 4c. METHOD BAKE-OFF + APERTURE-ARTIFACT CORRECTION (2026-07-24)

A follow-up "test all the literature options" pass (prompted by the user) produced a **material
correction to §4b**, and a decision to **revert** the `wavefront_aware` code. All numbers executed.

### 4c.1 The E4 "corrected-relay failure" was a TEST-SETUP ARTIFACT (oversized aperture)

The E4 generality-gate relay (§4.1) was built with a **10 mm aperture but a 2 mm beam** (aperture
2.5× the beam Ø). An aperture-vs-beam sweep on the E4 relay (sphere-only chain, focus-independent
exit-wavefront rms → Maréchal Strehl) shows a **sharp cliff**:

| aperture | ÷ beam-Ø | sphere-only Strehl |
|---|---|---|
| 4 mm | 1.0× | 0.999 |
| 5 mm | 1.2× | 0.999 |
| 6 mm | 1.5× | 0.997 |
| 7 mm | 1.8× | 0.104 |
| 8 mm | 2.0× | 0.042 |
| 10 mm (as-built) | 2.5× | 0.038 |

Once the aperture exceeds ~1.5× the beam diameter, the wildly-aberrated **marginal rays** (out where
the raw G2 singlet is nowhere near corrected — the beam never goes there) alias into the low-order
traced-OPL Chebyshev fit and destroy it, even inside the beam. **At a beam-matched aperture the
plain sphere-only carrier chain is diffraction-limited (Strehl 0.997), tying GBD (0.999) and the
ideal ceiling (0.999).** So §4.1/§4b's "the traced-carrier chain does not reproduce a corrected
relay" is **RETRACTED** — it was measuring the oversized-aperture artifact, not a chain limitation.
The E4 test's original `fail-before` assertion held for the wrong reason (aperture, not launch).

### 4c.2 Method bake-off (E4 relay), corrected reading

| method | focus quality | note |
|---|---|---|
| diffraction-limited ceiling | Strehl 0.999 / EE3 0.999 | reference |
| **sphere-only chain @ beam-matched aperture** | **Strehl 0.997** | already optimal — no new method needed |
| GBD per-surface (option 3) | EE3 0.999 | ties the plain chain; literature's beamlet architecture |
| wavefront-aware @ beam-matched aperture | Strehl 0.974 | marginally WORSE than sphere-only (adds launch noise) |
| option 2 (finer grid) | no change | not a lever |

So on a *well-posed* corrected relay every route agrees at ~diffraction-limited, and the
`wavefront_aware` launch is confirmed **not beneficial** (slightly harmful). Options 1 (SFT/PFT) and
4 (ENZ) were **not built** — unwarranted, since the plain chain already meets the mark on well-posed
relays and the true 121 cause is unknown (below).

### 4c.3 The real 121 residual is NOT the aperture artifact, and NOT the launch

- **Aperture clamp (aperture → 3× beam w per group, sphere-only):** EE6 **50.0% → 27.7%** (worse).
  Window-total dropped 58→45%, i.e. the clamp vignettes real energy — the 121's corrected
  multi-element groups have *gently*-aberrated marginal rays (unlike E4's raw singlet), so clamping
  only clips power. The aperture artifact does **not** explain the 121.
- **Clean fit-disc test (`_CARRIER_FIT_RADIUS_FRAC` sweep 0.5→0.2, native apertures, NO
  vignetting):** EE6 **flat at 49.8–50.0%**, window-total unchanged at 58.2% throughout. Shrinking
  the Chebyshev fit disc toward the beam — with zero energy clipped — does nothing. This closes the
  vignetting confound: the aperture-clamp harm above was purely vignetting, and the 121 residual is
  **definitively NOT** marginal-ray fit corruption.
- **Wavefront-aware launch on the 121:** EE6 50% → 10% (worse, §4b.2).

So for the 121 the two leading hypotheses — **ray launch** and **aperture/fit corruption** — are
**both eliminated**. The 50%→99% gap is a genuine, still-unidentified effect (remaining suspects per
§4b.3: the high-NA tail readout, the envelope/reconstruct hand-offs, the paraxial gap transport, or
a real accumulated aberration the R9-addendum attributed to the launch without independent proof).

### 4c.4 Decision

- **`wavefront_aware` code REVERTED** (does not help, slightly-to-substantially harmful, premise
  unsupported). The F3 downgrade (tilt_aware + explicit carrier → carrier-alone launch + warn) is
  restored as the validated behavior. The `wavefront_aware` kwarg, the `_carrier_relative_launch`
  branch, the E2 diagnostic, the E2 unit-test file, and the residual-launch diagnostic script were
  all removed; the `test_niche_r8` F3 pins were restored to their pre-E2 form. **This audit is the
  full record** (all measured numbers are captured in §4b–§4c).
- **Do NOT build options 1/4 yet** — diagnosing the 121's actual failure stage must come first;
  building fix-methods against an unproven premise is exactly the trap that produced Part E.
- **Aperture:beam cliff is a real, useful finding** — the traced element's OPL fit is corrupted when
  the physical aperture greatly exceeds the beam (marginal-ray aliasing into the low-order fit).
  Worth a library guard / a beam-relative launch-radius option (decouple the ray-fit domain from the
  vignetting aperture) independent of Part E.

## 5. Effort / plan

Matches the roadmap's own **L** estimate (multi-day). Sequencing for this implementation pass:

1. Core fix in `_lens_traced.py`: new carrier-relative residual branch replacing the F3 downgrade,
   gated exactly as in §2.2's table; residual-aliasing diagnostic (§2.4); updated warning text.
2. Chain plumbing: `wavefront_aware` kwarg on `propagate_traced_carrier_chain`, threaded to both the
   standard-leg and fine-retrace `apply_real_lens_traced` calls.
3. Unit tests: no-double-count / byte-identical-default pins (§3.1), residual-aliasing diagnostic
   test, chain-level `wavefront_aware` plumbing test.
4. The E4 synthetic corrected-relay oracle (§4.1) — independent of the exact implementation details,
   built against the public API, suitable for a parallel work-stream.
5. Full regression run (existing R6–R9 + dx-scaling suites + new tests).
6. Independent adversarial review of the diff.
7. Real-121 acceptance run (§4.2), reported honestly.
8. `CHANGELOG.md` + `docs/roadmap_deferred_2026_07_21.md` Part E status update.
