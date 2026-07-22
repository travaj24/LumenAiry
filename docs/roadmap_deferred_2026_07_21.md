# Lumenairy Roadmap — Deferred Capability & Performance Items (2026-07-21)

Baseline: **v5.27.0** (main).  This roadmap collects the items deferred by the
accuracy-niches campaign (v5.26.0, niches N1–N12) and the deferred-items campaign
(v5.27.0, N13–N16): further **capability coverage** (Part A) and **performance /
memory** gains (Part B), plus **validation / infra** follow-ups (Part C).

Every item is scoped so a partial or absent implementation **routes to a
validated reference** (GBD / FGA / traced-multibranch / exact ASM) rather than
producing a wrong answer — the campaigns' honest-envelope discipline carries
forward.

---

## 0. Cross-cutting principle — cache memory safety (design contract)

The stated concern — *caches blowing up the memory footprint* — is the single
most important constraint on Part B, because the highest-value perf wins are
caches of **N²-sized arrays** whose footprint scales with the grid.  A cos-grid
or prepared-screen entry is ~8 MB at N=1024 but **~512 MB at N=8192**, so a naive
"keep the last 8 entries" FIFO could silently retain multiple GB.  Every cache on
this roadmap MUST satisfy the following contract (an extension of the existing
`register_cache_clearer` / meta-pin-walker convention):

1. **Byte-budgeted, not count-budgeted.**  Bound the cache by **total retained
   bytes**, not entry count.  A single global ceiling
   `LUMENAIRY_CACHE_BUDGET_MB` (env + `set_cache_budget()` API), which **all**
   library caches respect *collectively*.  Default conservative — a small
   fraction of `available_memory_bytes()` (e.g. `min(512 MB, 10% of RAM)`) — so
   enabling caching never surprises a memory-constrained host.
2. **LRU eviction** (not FIFO) so a design-iteration loop keeps its hot entry
   instead of evicting the one it is re-using; evict least-recently-used until
   under budget.
3. **Opt-in for large caches.**  Any cache that can hold N²-scale arrays is
   **off by default** (or has a tiny default budget); the caller enables it
   explicitly for a design loop.  Small, cheap caches (scalars, 1-D LUTs) may
   stay on.
4. **Registry-enrolled + releasable.**  Every cache registers with
   `register_cache_clearer` (so `clear_asm_caches()` drains it) and exposes a
   `.release()` / `.clear()`; the v4.16.1 meta-pin walker enforces enrollment.
5. **Complete keys — no stale reuse.**  Key on the *full* determinant of the
   cached value (prescription + conjugate + wavelength + grid `dx, N`), the
   v5.26.0 lesson (an incomplete displaced-LUT key returned a stale entry for a
   different config).  Prefer content hashes over object identity.
6. **Measured + introspectable.**  A public `cache_report()` returns per-cache
   retained bytes + entry count + hit rate (mirroring `fga_memory_estimate()`),
   so the footprint is always visible, never a black box.

**Deliverable P0 (prerequisite for all of Part B's caches):** a shared
`lumenairy.cache` module implementing a `ByteBudgetedLRU` that all new caches use
— one place that enforces the global byte ceiling, LRU eviction, registry
enrollment, and `cache_report()`.  This is small, has no accuracy risk, and makes
every subsequent cache safe by construction.  Until it lands, no new N²-scale
cache ships.

---

## Part A — Capability coverage

### A1. Pearcey **cusp** completion of the uniform caustic (extends N16/K4)
- **State:** `caustic='uniform'` handles a rotationally-symmetric single **fold**
  (Airy).  A **cusp** (3-branch coalescence, `n_turn > 1`) is *detected* and
  falls back to multibranch.  The `pearcey(x, y)` kernel already ships.
- **Approach:** map the local ray geometry near a cusp to the Pearcey normal form
  `(t⁴/4 + x t²/2 + y t)` and evaluate `pearcey` on the two control coordinates
  `(x, y)` derived from the three branches' phases; reuse the K4 fit-and-continue
  machinery for the coefficients.
- **Effort/risk:** M.  The Pearcey control-coordinate mapping is the subtle part;
  the kernel is done.  **Oracle:** a synthetic 2-hump meridional map + a
  direct-RS cusp ground truth (extend `caustic_fold_truth.py`).
- **Fallback if unreached:** stays multibranch / GBD-FGA (current behavior).

### A2. General 2-D fold-**curve** uniform completion (non-symmetric / decentered / astigmatic)
- **State:** the K4 completion is a **radial** (rot-sym) reconstruction; a
  decentered / astigmatic fold is a **curve** in the image plane, currently
  detected → multibranch fallback.
- **Approach:** trace the fold caustic *curve* (locus of `det J = 0`), build a
  local orthogonal (arclength, normal-distance) frame along it, and apply the
  1-D fold-Airy `ζ`-continuation along the normal at each arclength sample — the
  same `_fold_airy_eval` kernel, driven per-curve-point instead of per-radius.
- **Effort/risk:** L.  The independent oracle is the hard part (no rot-sym
  symmetry to lean on — needs a 2-D direct-RS ground truth on a decentered
  singlet, feasible but heavier).
- **Value:** this is what would let **decentered / astigmatic** systems get the
  diffraction-correct caustic from traced.

### A3. Complex-saddle dark-side route (robustness, extends N16/K4)
- **State:** K4 uses the **fit-and-continue** dark-side route (it met the
  target).  The rigorous alternative — the coalesced real rays continued to a
  **complex** stationary point — was not built.
- **Approach:** analytically continue the meridional ray equations to complex
  launch height; Newton-solve the complex saddle on the dark side; feed its
  (complex `t, f, f''`) into `_fold_airy_eval`.
- **Effort/risk:** M–L.  Only needed where the fit-and-continue basis is
  insufficient (very sharp folds / short bright bands).  **Value:** robustness,
  not new coverage — schedule if a real design exposes a fit failure.

### A4. Higher catastrophes (swallowtail, butterfly) — principled fallback
- **State:** only fold + (with A1) cusp are handled analytically.
- **Approach:** rather than implement the swallowtail/butterfly canonical
  integrals (rare in real lenses, high effort), make the dispatcher **detect**
  the catastrophe class and route cleanly to **GBD/ASM** (diffraction-correct by
  construction) with a documented one-time note.
- **Effort/risk:** S (it is mostly a detector + routing).  Keeps "seamless" true
  by *routing*, not by covering every catastrophe analytically.

### A5. Far-tail aperture-edge (boundary-diffraction) term for the uniform caustic
- **State:** the uniform completion's residual (~2 % r2m) is dominated by
  **hard-aperture-edge diffraction** beyond ~2× the caustic radius, which a pure
  fold-Airy underestimates.
- **Approach:** add a Geometrical-Theory-of-Diffraction / boundary-wave edge term
  (Rubinowicz / Keller edge diffraction) from the aperture rim, superposed on the
  Airy field.
- **Effort/risk:** M.  **Value:** takes the uniform completion from ~2 % to
  sub-% and makes the *far* field correct too.

### A6. Decentered **focal PSF** via traced (uniform at the image caustic)
- **State:** GBD (N10b) is the decentered-EE reference; traced multibranch
  *over-amplifies* at a point focus (the D5 caustic pile-up).
- **Approach:** the image-plane point focus is itself a (higher-order) caustic —
  apply the A1/A2 uniform machinery *at the image plane* so traced gives the
  finite, diffraction-correct decentered PSF, closing the last N12 routing
  hand-off to GBD.
- **Effort/risk:** L, and **depends on A1/A2**.  **Value:** unifies traced as the
  single decentered model (wavefront + PSF).

### A7. GBD vector prescription-chain carrier port (extends N4)
- **State:** `propagate_gbd_through_prescription` got `direction_sampling='auto'`
  (N4); the **vector** sibling `propagate_gbd_vector_through_prescription` was
  left at `direction_sampling=False`.
- **Approach:** port the same Husimi carrier decomposition to the vector chain.
- **Effort/risk:** S.  Mechanical, mirrors N4; **oracle** = the N4 chain-vs-
  sequential equivalence test, vectorized.

### A8. FGA multi-valued caustic — deeper specialty validation
- **State:** FGA is validated at the fold ground truth (N6); its cusp / fold-cusp
  interference specialty is only lightly exercised.
- **Approach:** extend the caustic ground-truth suite (with A1's cusp truth) and
  pin FGA vs GBD/traced-uniform at a cusp; document where each is the reference.
- **Effort/risk:** S–M (mostly oracle + comparison, little new code).

---

## Part B — Performance & memory

> All caches below are gated on **P0** (the `ByteBudgetedLRU` contract, §0).

### B1. Pointwise cos-grid cache — *the* deferred perf win (K3)
- **State:** the pointwise 2-D obliquity path re-runs a ~3.9 s Delaunay build +
  query on **every** call; in a decentered-design iteration loop this dominates.
  Deferred in K3 precisely because it needs the safe-cache design.
- **Approach:** cache the per-surface `(cos_αin, cos_αout)` **2-D cos-grid**
  keyed by `(prescription-surface, conjugate, dx, N)` in the `ByteBudgetedLRU`.
  Turns the 3.9 s query into ~0 on repeat (same design, moving field) — the
  common inner loop.
- **Memory:** a cos-grid pair is `~2·N²·8 B` = **16 MB @ N=1024**, **1 GB @
  N=8192**.  → **byte-budgeted + opt-in** (default off / tiny budget); the
  caller enables it for a loop; `cache_report()` shows the footprint.  This is
  the item the memory contract exists for.
- **Effort/risk:** M (the cache), gated on P0.  Byte-identical output (pure
  memoization).

### B2. Astigmatic carrier-ASM FFT reduction
- **State:** K3 found the astigmatic path **FFT-bound** (~10 1-D FFTs / 0.59 s @
  N=2048); the Sziklas–Siegman focus-crossing bridge's FFT count is intrinsic.
- **Approach:** real-FFT (`rfft`) for the real-symmetric legs; cache the FFTW
  plan / twiddles across the per-axis calls; skip the bridge FFTs when no
  focus-crossing.  Also a natural **GPU** target (B6).
- **Effort/risk:** M.  Measure — the win is likely 1.3–1.8×, not transformative.

### B3. GBD decenter windowed-reconstruct optimization
- **State:** the GBD decenter cost is `_reconstruct_windowed` (~2.27 s / 1.6 GB @
  N=1024) — the standard windowed beamlet reconstruction.
- **Approach:** the **memory** (1.6 GB) matters more than the time — chunk the
  windowed accumulation (accumulate into the output in tiles instead of a dense
  per-beamlet stack), cutting peak bytes.  A `LUMENAIRY_MEM_BUDGET_MB`-aware
  reconstruction (mirror the FGA chunk loop).
- **Effort/risk:** M.  **Value:** removes a 16 GB-runner OOM risk on large
  decentered GBD runs (the v5.26.0 lesson class).

### B4. Adaptive-FGA dual-number ray-transfer vectorization
- **State:** the adaptive-FGA hot spot is the dual-number analytic ray-transfer
  in `raytrace/differential.py` (`__mul__` ~178 k calls / ~23 s).
- **Approach:** replace the scalar-object dual-number arithmetic with a
  **vectorized** batched dual (arrays of value + derivative), or a numba/JAX
  kernel for the per-ray transfer.
- **Effort/risk:** M–L (touches the differential-ray core; needs careful
  byte-identity pinning).  **Value:** large — this is FGA's dominant cost.

### B5. Pointwise-obliquity interpolation speedup (complements B1)
- **State:** even cached (B1), the *first* call is Delaunay-query-bound.
- **Approach:** for a **structured** launch grid, replace the scattered Delaunay
  interpolation with a structured-grid `map_coordinates` (the ray fan is a grid,
  not a point cloud) — O(N²) direct instead of a triangulation search.
- **Effort/risk:** M.  **Value:** speeds the cold path *and* reduces B1's cache
  pressure (cheaper to recompute → smaller budget needed).

### B6. GPU acceleration across carrier ASM / FGA (builds on N14/K2)
- **State:** K2 shipped CuPy+JAX **carrier** backends (parity-validated); FGA and
  the traced/displaced paths remain CPU.
- **Approach:** extend the backend abstraction to FGA's momentum quadrature and
  the ASM propagators; the real GPU win is on the large-N FFT-heavy paths.
- **Effort/risk:** L, and **depends on C1** (a box with working CUDA to validate
  parity + speedup — this dev box has broken cuFFT).

### B7. jax × OpenBLAS `lstsq` mitigation (library-side)
- **State:** a numpy `lstsq` in the traced path can deadlock with JAX's OpenMP
  under the sharded-CI thread config (worked around with CI env pins + OMP=1).
- **Approach:** replace the `lstsq` with a thread-safe normal-equations solve (or
  a JAX-native solve) so the deadlock cannot recur outside the CI pin.
- **Effort/risk:** S–M.  **Value:** removes a latent CI fragility.

---

## Part C — Validation / infrastructure

### C1. Real-GPU CuPy CI leg + validation
- CuPy parity/speedup is validated here only for the **FFT-free** path (this box
  has no functional cuFFT).  Add a CUDA-capable CI leg (or a documented
  manual-validation run) to exercise the FFT-requiring carrier/FGA GPU paths.
  **Blocks:** B6's speedup claims.

### C2. JAX-jitted focus-crossing
- The carrier focus-crossing split runs **eager** (data-dependent branch).  A
  `lax.cond` (or a masked both-branches-then-select) form would let the whole
  carrier leg `jit`-compile.  Effort M; value = the JAX speedup on crossing legs.

### C3. `through_focus` golden re-baseline
- The H1 slant-OPD fix (v5.25.0) shifted the `through_focus` internal goldens;
  a clean re-baseline against the corrected model was flagged, not executed.

### C4. Extended point-source oracle matrix
- The ZOS **Huygens-PSF** point-source oracle (N0.2) is built but only lightly
  used; broaden the cross-check matrix (point sources, finite conjugates, the
  decentered A2/A6 cases) to harden the caustic + decenter validation.

---

## Part D — traced-carrier-chain audit remediation (F1–F4)

Added after the roadmap campaign surfaced
`docs/audits/AUDIT_TRACED_CARRIER_CHAIN_2026_07_21.md` — traced-group fidelity on
the design-121 carrier chain.  **F1+F2 are jointly the last blocker to a
real-surface production traced model for corrected relays** (with them fixed, the
carrier-referenced chain hits Zemax-class fidelity at N=2048 in <1 min, ~300×
cheaper than fixed-grid).  Repro is local-only (needs the 121 `.zmx`), so
remediation *unit* tests must be self-contained/synthetic.

- **R6 / F1 (P1):** `carrier='auto'` does not engage on a clean spherical input
  (auto-fit → ~∞ → silent no-carrier), so the full chain runs as if H6 never
  landed.  Fix the auto-fit to recover a spherical R (unwrapped radial phase /
  Husimi mean-slope).  Gate: `'auto'` == explicit R on the repro (r⁴ 0.588 →
  0.005).
- **R7 / F2 (P1) — the flagship:** thick groups leave a smooth exit-curvature
  (defocus) error even with the correct explicit carrier; the intra-group
  reference through the glass still assumes near-collimated, so error ~
  thickness × curvature (triplets worst, windows exact), accumulating to ~6.8 rad
  → Strehl ≈ 0.  Carrier-reference the intra-group propagation; "windows stay
  exact" is the guard.  Gate: per-group rms < 0.1 rad on all 8 groups AND
  end-to-end EE6 ≥ 99% at ~2.9 µm.
- **R8 / F3 (P2) + F4 (P3):** guard/fix `tilt_aware_rays` on steep carriers;
  ship the F4 ergonomics — a chain-orchestrator API (element supplies R_out) and
  packaged near-focus landing.

### R9 (done) + the remaining 121-image blocker — wavefront-aware ray launch

- **R9 (shipped, general):** the paraxial carrier could not focus the NA-0.46
  final leg.  `carrier_referenced_exact_focus_readout` + the orchestrator's
  `final_leg='auto'` (NA threshold) route any high-NA leg through exact
  band-limited ASM (no paraxial approximation); design-agnostic (synthetic
  NA-0.46 sphere → diffraction limit, EE-in-2w₀ 1.3% → 99.8%).  Lifted the 121
  end-to-end EE6 7.3% → 69.7% (~10×).
## Part E — Wavefront-aware ray launch (close the corrected-relay image)

**The flagship remaining item** (the last blocker to a production real-surface
traced model for *corrected* relays; general, not 121-specific).  Isolated by
R7 + R9 after per-group fidelity (F2) and high-NA focusing (R9) were both solved.

### E1. Diagnosis (measured)

The traced-carrier model launches each group's rays along the **carrier sphere**
`S(R) = sign(R)(sqrt(r² + R²) − |R|)` — exact for a stigmatic (single point-source)
congruence.  But a **corrected relay** deliberately carries **non-spherical
intermediate wavefronts**: each group pre-shapes aberration that a later group
cancels.  Launching along the sphere discards that inter-group deviation, so the
pre-correction never propagates and the aberration the tail is designed to null
isn't present when it gets there.  Measured on the 121: **~1.68 rad RMS**
(a₄w⁴ ≈ +27…+44 rad) already present *entering* the high-NA tail, even though every
group is individually clean (per-group exit rms < 0.023 rad post-F2).  Proven it
is NOT the final leg / reconstruct / gap transport (the whole tail on one fine
grid with exact-ASM gaps still plateaus at EE6 ≈ 70%).

### E2. The fix — launch along the actual wavefront, not the sphere

Keep the carrier as the **frame** (co-moving grid + envelope reference), but take
each group's ray-**launch directions** from the incoming field's **actual local
wavevector** (the gradient of its unwrapped phase / a Husimi mean-slope map),
i.e. the sphere gradient **plus the residual non-spherical deviation**.  The
existing `tilt_aware_rays=True` lever already launches per-pixel tilts from the
field gradient — it *is* a wavefront-aware launch — but F3 currently guards it
**off** for explicit carriers because on a steep spherical carrier it
double-counted the carrier tilt.  Approach candidates:

1. **Carrier-relative tilt-aware launch (preferred):** launch along
   `grad(carrier sphere) + residual`, where `residual = grad(unwrapped field
   phase) − grad(carrier sphere)`.  The carrier handles the spherical part
   (fast, exact); tilt-aware carries only the non-spherical residual, so F3's
   double-count vanishes and F3 becomes a *unification*, not a guard.  Reuses the
   H6/R7 carrier eikonal + the tilt-aware machinery.
2. **Full wavefront congruence:** drop the spherical ray launch entirely; build
   the launch direction field directly from the incoming field's Husimi /
   local-frequency map (the true ray congruence), with the carrier only setting
   the frame.  More general, heavier.

### E3. Challenges / risks

- **Aliasing** — a corrected relay's intermediate wavefront can be steep and
  high-order; extracting `grad(phase)` robustly needs the un-aliased-core /
  Nyquist discipline that R6 (auto-fit) and R9 (NA-0.46 exit) already established
  (a naive gradient over the full field is exactly what corrupted F1's auto-fit).
- **No double-count** — the residual must subtract the carrier sphere cleanly
  (the F3 failure mode) so collimated / stigmatic launches stay byte-identical.
- **Per-group grid** — must Nyquist-sample the *actual* (non-spherical) local
  frequency, which can exceed the sphere's; likely reuses R9's per-group fine
  re-trace (`_fine_trace_group_exit`).
- **Interaction** with F2's exact-sphere carrier + fit-domain restriction and the
  multibranch / uniform-caustic machinery.

### E4. Oracle / acceptance

- **Design-agnostic synthetic:** a purpose-built 2–3-group *corrected* relay with
  deliberately non-spherical intermediate wavefronts (a front group that
  over-corrects, a tail that cancels) that a spherical launch (fail-before)
  leaves aberrated and the wavefront-aware launch focuses to the **diffraction
  limit** — validated vs an inline exact meridional-raytrace + eikonal oracle
  (no `.zmx`, CI-safe).  This proves the *capability* independent of the 121.
- **121 end-to-end (acceptance instance):** `propagate_traced_carrier_chain`
  reaches **EE6 ≥ 99% at ~2.9 µm** (Zemax 2.736 µm; stigmatic 2.97 µm), from the
  current ~70% / 4.05 µm.  Repro: `validation/repro_traced_carrier_121/
  carrier_chain_121.py` (local `.zmx`).
- **Byte-identical defaults:** stigmatic / single-element / collimated launches
  unchanged (the sphere == the wavefront there, so the residual is ~0).

### E5. Effort

**L** — a genuine ray-launch model change touching the carrier reference, the
tilt-aware path (unifying F3), and the per-group grid.  General: closes the
image for *any* corrected relay, not just the 121.  Runs as its own campaign
(same impl → adversarial-verify → fix harness), with the E4 synthetic as the
generality gate and the 121 as the acceptance instance.

## Suggested sequencing

1. **P0** — `ByteBudgetedLRU` cache contract (§0).  Small, unblocks all of Part B
   safely; directly answers the memory-footprint concern.  **Do first.**
2. **B1 + B5** — cos-grid cache + structured-grid interpolation.  The biggest,
   safest perf win for decentered design loops, now memory-safe.
3. **A1 (+ A2)** — cusp (then non-symmetric fold) uniform completion.  The main
   remaining *coverage* gap; A2 unlocks A6 (traced decentered PSF).
4. **B3 + B4** — GBD reconstruct memory + adaptive-FGA vectorization (the two
   largest CPU/mem costs).
5. **C1 → B6** — real-GPU validation, then extend GPU beyond the carrier ASM.
6. **A3 / A4 / A5 / A7 / A8 / B2 / B7 / C2–C4** — as specific designs demand.

Effort key: **S** ≈ hours, **M** ≈ 1–2 days, **L** ≈ multi-day.  Each item lands
opt-in with byte-identical defaults, an independent oracle, and — for every
cache — the §0 byte-budgeted / registered / releasable / introspectable contract.
