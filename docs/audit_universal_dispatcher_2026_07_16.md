# Audit — `apply_real_lens_universal` auto-dispatcher (+ FGA memory levers)

**Library:** lumenairy 5.24.0 · **File:** `lumenairy/propagators/fga.py` · **Date:** 2026-07-16
**Trigger:** driving the design-121 relay (24.65 mm aperture, no-MLA divergent source, N=28672 @ dx 0.9 µm) through `method='universal'` and `apply_real_lens_fga`.

The dispatcher no longer routes multi-valued fields to `traced` (fixed in 8b7e203), but three defects remain that make `method='auto'`/`'fga'` mis-behave on a **single-valued, strongly-diverging, large-aperture** beam. F1 is the dispatcher routing bug; F2/F3 are memory-lever bugs in the FGA route the dispatcher calls.

| # | Severity | Component | One-line |
|---|----------|-----------|----------|
| F1 | **High** (silent accuracy loss) | universal router, single-valued branch | high-NA single-valued **diverging** beam → `traced` with no collimation check → blurred output |
| F2 | **High** (crash) | `apply_real_lens_fga` separable path | full `(Nq×Np)` coeff array allocated up front → OOM regardless of `mem_budget_mb` |
| F3 | Medium | `mem_budget_mb` scope | budget bounds the momentum chunk only, not the position-lattice (Nq) ray trace |

---

## F1 — auto-router sends single-valued diverging beams to `traced` (blurred)

**Location:** `apply_real_lens_universal`, single-valued high-NA branch — `fga.py:1231-1242`, applied at `fga.py:1258-1260`.

```python
else:  # single-valued, high-NA
    zone = _caustic_zone(E_in, dx, prescription, wavelength)
    near = (zone[0]-pad) <= opd <= (zone[1]+pad) if zone is not None else False
    chosen = "fga" if near else "traced"        # <-- no collimation check
```

**Observed:** design-121 relay lens S3-S4 (input residual angular spread **0.128 rad**) routed to `traced`, which then emits its own `RuntimeWarning` (`fga.py:1259`):
> `apply_real_lens_traced: input residual angular spread 0.128 rad exceeds the collimated-reference validity threshold (0.02 rad). The plane-wave-referenced traced correction will be inaccurate (blurred). Pass carrier= … or use apply_real_lens. Set on_noncollimated='delegate' to fall back automatically …`

The field is still returned (only a warning), so the blur is **silent** to a caller that isn't watching stderr.

**Root cause:** the single-valued branch treats "single-valued + not-near-caustic ⇒ `traced`" as sufficient, but `traced` has a *second* validity precondition — the beam must be ~collimated (or carrier-referenced). The router checks multi-valuedness (`_tilt_dispersion`, added in 8b7e203) but never checks the residual **angular spread** that `traced` itself gates on. A smooth single-emitter diverging beam passes the multi-valued test yet fails `traced`'s collimation test → it lands on the one propagator that can't handle it.

**Impact:** every no-MLA / divergent-source / single-emitter imaging case gets a silently-blurred result from `method='auto'`. This is exactly the regime the universal dispatcher is meant to make "just work."

**Recommended fix (any of):**
1. In the single-valued branch, compute the residual angular spread (the same quantity `apply_real_lens_traced` gates on) and, if it exceeds the collimated-reference threshold, do **not** pick raw `traced` — instead route to `phase_screen` (wave-exact for any angle) or `fga`.
2. Have the dispatcher's `traced` call default to `carrier='auto'` (references the beam's own congruence — the warning's own first suggestion) or `on_noncollimated='delegate'` (auto-fallback to `apply_real_lens`). Minimal, localized: inject the kwarg at `fga.py:1259-1260` unless the caller overrode it via `method_kwargs['traced']`.

Either makes the diverging-relay case correct-by-default instead of blurred-by-default.

---

## F2 — FGA `separable` path allocates the full `(Nq×Np)` array up front → OOM defeats `mem_budget_mb`

**Location:** `_fga_through_lens` `fga.py:526-535` (separable gate + `c_full`), OOM at `_gabor_coeff_sep` `fga.py:411` (`cr = np.zeros((Nq, Np))`).

```python
cw = Np if (chunk is None or chunk <= 0) else min(chunk, Np)   # momentum chunk
use_sep = bool(separable) if separable != "auto" else (n_p >= 5)
c_full = None
if use_sep:
    c_full = _gabor_coeff_sep(u0, qx, qy, px[:n_p], ...)   # <-- FULL (Nq×Np), pre-loop
```

**Observed:** design-121 first lens, `Nq = 198,291,575`, `Np = 225` (n_p=15) → `np.zeros((198291575, 225))` = **332 GB** → `MemoryError`, **with `mem_budget_mb=40000` explicitly set**.

**Root cause:** `separable='auto'` turns on at `n_p ≥ 5` (the default), and the separable analysis "needs the FULL tensor momentum grid, so compute all coefficients ONCE up front" (`fga.py:528-531`). That `c_full` allocation happens *before* the chunk loop, and `_chunk_from_budget` / the `cw` chunk only bound the per-chunk beamlet-transport arrays (`QX/QY/PX/PY/AW`, `fga.py:559-563`). So the memory lever silently does not apply to the **dominant** allocation. `separable=False` avoids it (per-chunk `_gabor_coeff`), but nothing in the API/docs tells the caller that `mem_budget_mb` is a no-op while `separable` is on.

**Impact:** `apply_real_lens_fga` and the universal's fga route OOM on any large-aperture (large-Nq) field regardless of the budget the caller set — the exact case the momentum-chunking feature (88703f1) was meant to cover.

**Recommended fix (any of):**
1. Block/stream `_gabor_coeff_sep` over Nq so `c_full` respects `mem_budget_mb`.
2. Auto-fall-back to `separable=False` when the projected `c_full` bytes exceed `mem_budget_mb`.
3. Minimum: have `_chunk_from_budget` account for `c_full` and raise a clear `"separable coefficient array (Nq×Np = X GB) exceeds mem_budget_mb; pass separable=False"` instead of a raw 332 GB `MemoryError`.

---

## F3 — `mem_budget_mb` bounds the momentum chunk, not the position-lattice (Nq) ray trace

**Location:** `fga.py:526` (`cw` chunks `Np` only); per-momentum ray trace `fga.py:573` (`ray_transfer_jacobian(qx…)` over all `Nq` rays).

**Observed:** with `separable=False` + `mem_budget_mb=40000`, dq_step=2 (`Nq≈1.98e8`) reached **99 % RAM (1 GB free)** and **>19 min on lens 1** — i.e. bounded in the momentum dimension but not overall.

**Root cause:** chunking is over `Np`; `Nq` is never chunked, and each `ray_transfer_jacobian` materializes per-`Nq` arrays across every surface. A huge `Nq` (aperture-support-pixels / dq_step²) is unbounded.

**Impact:** `mem_budget_mb` under-delivers on large apertures; the only real levers become `dq_step`/`prune_frac`, which trade accuracy. Users setting a budget will still near-OOM and stall.

**Recommended fix:** support position-lattice (Nq) chunking of the ray trace + scatter, **or** document that `mem_budget_mb` bounds momentum-chunk memory only and that `Nq` must be controlled via `dq_step`/`prune_frac`.

**Runtime corollary (not just memory):** coarsening to avoid F2/F3's memory does not rescue *speed*. At `dq_step=6, n_p=10, separable=False` (`Nq≈2.2e7`) the design-121 **first lens alone ran >14 min** (numba JIT + 100 momenta × 2.2e7-ray `ray_transfer_jacobian` calls) → ~2 h for a 9-lens 1×1, at already-degraded accuracy. The per-momentum full-Nq ray trace is the bottleneck; there is no `dq_step` that is both accurate *and* fast on this aperture. This is the practical ceiling that motivates F3's Nq-chunking / a vectorized-once ray trace shared across momenta.

---

## Secondary notes

- **Two dispatchers, divergent logic.** `apply_real_lens_auto` (`fga.py:999`, GBD/FGA 2-way — defaults `gbd`, upgrades to `fga` near caustic) and `apply_real_lens_universal` (`fga.py:1139`, 4-way) route differently. Worth clarifying which is canonical and aligning their diverging-beam handling.
- **Caustic test keys on the OUTPUT plane.** The near-caustic→fga check uses `opd`; a caller that applies the lens at `opd=0` (exit vertex) and does its own downstream ASM will *never* trigger it (the caustic is downstream of the vertex), so a single-valued field can only be routed to `phase_screen`/`traced`/`gbd`, never `fga`. Reasonable given the API, but worth a docstring note for split-step callers.

## Repro

```bash
# F1 (blurred traced): universal on the single-valued diverging relay
POC_LENS_MODEL=universal POC_NTX=1 POC_N_GRID=28672 POC_DX_UM=0.9 \
  python run_poc_119_120_v518.py 121 --fresh      # -> S3-S4 warns "0.128 rad … blurred"

# F2 (OOM despite budget): FGA with the default separable path
POC_LENS_MODEL=fga POC_NTX=1 POC_N_GRID=28672 POC_DX_UM=0.9 POC_FGA_MEM_MB=40000 \
  python run_poc_119_120_v518.py 121 --fresh      # -> np.zeros((198291575,225)) = 332 GB

# F2 avoided but F3 exposed: separable=False still 99% RAM + >19 min/lens at dq_step=2
POC_FGA_SEPARABLE=false … POC_FGA_DQ_STEP=2 …
```

Net: the 24 mm-aperture 121 sits outside the FGA's practical envelope at accuracy-preserving `dq_step`, and the auto-dispatcher's single-valued path mis-handles its diverging relay. F1 + F2 are the two "should just work" defects to fix first.

---

## Resolution — v5.24.1 (2026-07-16)

| # | Status | Fix | Commit |
|---|--------|-----|--------|
| F1 | **FIXED** | `apply_real_lens_universal` single-valued branch now splits the smooth-plane case on the beam's residual angular spread using traced's OWN discriminator (`_carrier_residual_rms` vs `_NONCOLLIMATED_RESID_THRESH=0.02`): collimated → `traced` (unchanged), **diverging → `phase_screen`** (wave-exact propagation + bounded thin-screen OPD, never a blur, honest `return_method`). Near-caustic still → `fga`. | 637b3d8 |
| F2 | **FIXED** | `_fga_mem_guard`: if the separable `c_full` (`Nq·Np·16` B) exceeds `mem_budget_mb`, auto-fall-back to the per-momentum-chunk direct analysis (no up-front whole-grid array) — no accuracy loss (the direct path is exact). | 0530036 |
| F3 | **FIXED (guard + doc)** | Same guard raises a CLEAR `MemoryError` naming `Nq` + the levers when even the cw=1 ray-trace floor is a genuinely large overshoot (> budget AND > 4 GB), instead of a confusing multi-GB crash. Documented that `mem_budget_mb` bounds the momentum chunk only; `Nq` is controlled via `dq_step`/`prune_frac`/`n_p`. **Full position-lattice (Nq) chunking is deferred**: per this audit's own verdict the 24 mm aperture is outside FGA's practical envelope at any accuracy-preserving `dq_step` (the ray trace alone is >19 min/lens), so failing fast with guidance is the honest fix; the F2 fallback makes medium-large apertures that fit after separable→direct just work. | 0530036 |

**Rejected alternative (F1).** Defaulting the traced call to `carrier='auto'` was tried and **reverted**: an adversarial check showed it *broke* a small diverging beam (fid 0.22 vs the 0.98 no-carrier / phase_screen result) — the auto low-order carrier fit is not a safe blanket default. `phase_screen` (`apply_real_lens` + exact ASM) is exact for a small beam (~0.4 nm obliquity) and bounded for a large high-NA one; `method='fga'` remains available to force the ray-exact path. Cross-check: FGA itself is inaccurate on this diverging beam at these params (free-space fid 0.91–0.93 vs exact ASM — the small-beam-in-fine-grid scale mismatch), confirming `phase_screen` is the correct routing target.

Tests: `test_universal_dispatcher_diverging_beam_not_blurred_via_traced` (F1 routing + no-blur + honest dispatch), `test_fga_mem_guard_large_aperture` (F2 fallback + F3 clear-error + no-loss fallback). Also de-flaked the unrelated eig-heavy `test_eme_2d_vector::test_vector_verify_removes_spurious` (singular shift-invert → offset sigma + 1-thread BLAS, commit 9759796).
