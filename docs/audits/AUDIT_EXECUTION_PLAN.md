# PMM + RCWA Audit — Execution Plan / Remaining Work

Living tracker for executing `docs/audits/PMM_RCWA_AUDIT_2026_06_08.md`.
Last updated 2026-06-09. CI is GREEN (head `3f4b8e3`+). Work in test-gated commits;
**run ruff in WSL before every push** (`~/lumvenv/bin/ruff check lumenairy/ tests/unit/`)
and use the WSL venv as the Linux/OpenBLAS CI-proxy (see memory
`feedback_lumenairy_wsl_ci_proxy`). Auto-push on green per `feedback_lumenairy_auto_push`.

## DONE
- Audit (39-agent workflow) + report.
- P1 B1-B7 (kx0 Bloch shift; binary Fourier aliasing; JAX duty raise; JAX metal
  n_max; signed-slant gate; CuPy `_check_energy`; JAX x64 hard-raise) + tests
  (`tests/unit/test_v5_12_0_audit_fixes.py`).
- CI green: cache-lock+registry for `_LAGRANGE_DREF_CACHE`; `__all__` dedup; covariant
  slant=0 BLAS-instability routed to the exact vertical solver with `stabilize=True`;
  `is_spectral` absolute thresholds. (Commits d93c575, 453d015, e89b2c1, cd68acd.)
- Stage 1 P2 part 1: PMMStack `eyy` n_max (both paths); staggered `degree>=3`; strict
  `duty` (0,1). (Commit 3f4b8e3.)

## STAGE 1 — P2 robustness cluster (part 2, rcwa.py). One PR, each with a test.
Line numbers drift as edits land — locate by content.
1. **Grazing `kz_inc` /0** (pmm.py, `_assemble_jones_farfield` ~633-637 and
   `_scalar_farfield_RT` ~659-663): incident-flux normalizer divides by `kz_inc`
   with no near-zero guard (diffracted orders ARE guarded). Clamp/raise when
   `|kz_inc| < ~1e-9`.
2. **`stabilize` retry drops `asr_eta`/`asr_samples`** (rcwa.py `rcwa_efficiency_1d`
   retry block, ~1444-1467): the energy-retry re-solves WITHOUT ASR. Forward
   `asr_eta`/`asr_samples` into the retry, or raise the incompatibility.
3. **`rcwa_efficiency_vs_wavelength` aborts whole sweep on one bad λ** (rcwa.py
   ~1708): no per-λ try/except `_EnergyError`; docstring claims a `stabilize` it
   doesn't expose. Forward `stabilize` and/or per-λ NaN+warn.
4. **`RCWAStack.solve(stabilize=True)` doesn't catch `_EnergyError`** (rcwa.py
   ~5177-5184): wrap each window in `except _EnergyError: continue`, raise only if
   all windows fail.
5. **`_nv_field_2d` background = corner pixel** (rcwa.py ~2249): biases ≥3-material
   NV fields. Reference-free `|grad eps|` edge indicator, or document the two-material
   assumption.
6. **`rcwa_jones_1d` Wood nudge omits `ezz`** (rcwa.py ~3249-3250; segments
   ~3429-3430): add `eps[2,2]` to the grazing-detection set.
7. **`rcwa_efficiency_2d_shapes` eps=0 → inf/NaN** (rcwa.py ~3825; `_validate_shapes`):
   add an `|eps|>0` floor guard + the `eps_background` check.
8. **2D `'li'` leaves wall-normal at Laurent** (rcwa.py ~2542, `EPS_xx=EPS`): inline
   comment that 2D `'li'` applies the inverse rule to `E_z` only (full in-plane needs
   `fff_nv`).
9. **`rcwa_jones_2d` Laurent for wall-normal tensor** (rcwa.py ~3713-3721): document
   at the call site, or offer a Li option.

## STAGE 2 — naming / convention / docs. No behavior change beyond deprecated aliases.
- **2D return arity (the ship-blocker)**: `pmm_efficiency_2d`/`_staggered` return a
  4-tuple `(orders,R,T,dof)` vs `rcwa_efficiency_2d`'s 3-tuple. Drop `dof` (expose via
  attribute/dataclass) or add a matching RCWA diagnostic; add a cross-suite contract
  test. (pmm2d.py / pmm2d_staggered.py vs rcwa.py `rcwa_efficiency_2d`.)
- Incidence keyword: 1D uses `angle`, 2D/Stack use `theta`/`phi`. Accept `theta` on 1D
  (deprecated `angle` alias) + `angle` on `RCWAStack.set_source`; document
  planar=angle / conical=theta,phi.
- `far_field_orders` (1D PMM) vs `n_orders` (2D) — unify with a deprecated alias.
- staggered `degree` is the modified-Legendre count M, not a GLL degree — rename to
  `n_modes`/`order_M` (keep `degree` alias) or document loudly.
- `formulation`: 1D defaults `'auto'`; 2D `_normalize_2d_formulation` REJECTS `'auto'`
  — add `'auto'`→`'li'` for 2D.
- Stale docstrings: `stabilize` says "minimum-power" but does per-order CONSENSUS
  (pmm.py ~2603, ~1394); drop the stale `'IN-PLANE'` eps qualifier where OOP is
  supported (pmm.py ~1357, ~2858, ~4831; `_pmm_jones_oblique_segments_solve`
  "IN-PLANE only"); `rcwa_efficiency_1d_jax` "removed in v5.7.0" but persists.
  pmm_jones_1d docstring still says "IN-PLANE" (it routes OOP to the metric gen).
- P3 renames: `_exx_eq_ezz`→`_exx_minus_ezz`; `EPS_xx`→`EPS_normal/EPS_wallnormal`;
  `RCWAStack.nox/noy`→documented props; unify `kx0` units (1D dimensional vs 2D
  dimensionless); `flux_inc` vs `einc_sq`; document `_homogeneous_eigenmodes` 3rd-slot
  `kz` vs `_layer_eigenmodes` `lam`; clarify `_sqrt_forward` convention comment.
- `PMMStack.solve` returns a bare tuple vs `RCWAStack`'s `RCWAResult` — mirror or doc.
- `stabilize` default PMM True / RCWA False — document side-by-side as a choice.

## STAGE 3 — performance (RCWA-first; byte/near-byte identical where claimed).
RCWA, ranked: P1 `_redheffer_star` skips the `inv` of a zero S11/S22 block (propagation
matrices) — ~half the 2N×2N inverses are identity; P2 `_homogeneous_eigenmodes` builds
a full 2N×2N `Q` via matmuls for an analytically DIAGONAL result (O(N³)→O(N), also
`5081-5083`, `1140`); P4 `_binary_grating_convolutions` always builds `EPS_II` even on
the Laurent/TE path (use_li-gate); P5 single-call entry points never use `_HOMOG_CACHE`;
P3 vs_wavelength rebuilds dispersionless convolutions per-λ; P7 batch the per-component
FFTs; P6 `vmap` over wavelength/angle sweeps (jitted core); P9 batch per-layer
eig/inv/solve into `(L,2N,2N)`; P8 `_with_blas_limit` no-op default (auto-cap); P12 2D
non-Herm eig O((2N)³) — default circular trunc + even-parity symmetry; P10 JAX general
eig has no GPU lowering (doc/warn only); P13 GPU host round-trips (NV/ASR/energy on
device). PMM: **PP1 analytic homogeneous half-space basis** (the dominant ~2-3× lever,
also kills 2 of 3 eigs per stabilize-degree) + PP10 TM `invop`/`iS0` reuse; PP2/PP5/PP6/
PP7 staggered homogeneous-region + eps-free caching + dead-code; PP3 hybrid Kron
factorization (`Minv=kron(inv My,inv Mx)`); PP8/PP9 projection/lstsq vectorization
(`_sem_fourier_projection`, `_assemble_jones_farfield` two lstsq → one). Gate each on
byte-closeness vs the eig path before enabling on detection. (See report §3 for
file:line + est. gains. NOTE PP4/PP11 are largely subsumed/by-design.)

## STAGE 4 — 1D/2D package reorg (LAST; highest blast radius; FULL suites gate).
Package layouts in report §4.1 (`rcwa/`), §4.2 (`pmm/`), §4.3 (`pmm2d/`). Order:
1. `pmm2d/` consolidation (`_common` extraction; collapses the duplicated `_sqrt_decay`/
   `_inv_lam`/`_kz_forward2`/farfield epilogue between pmm2d.py and pmm2d_staggered.py).
2. `rcwa/` split (extract the R/T tail `_rt_from_amplitudes`/`_incident_vector`/
   `_single_layer_S` — duplicated ~7× verbatim — as one behavior-preserving refactor).
3. `pmm/` split — **MUST** preserve every test-imported private + module-global
   dispatch lookup (report §4.4): tests import `_pmm_jones_slant_solve`,
   `_pmm_jones_slant_diag_solve`, `_stabilize_jones`, `_ill_scaled`, `_safe_*` AND
   monkeypatch module globals; the dispatcher calls cure callees as BARE module
   globals, so re-export all of them from `pmm/__init__.py` (or resolve via
   `import lumenairy.elements.pmm as _pkg; _pkg.<name>`). Add a Stage-0 reachability
   smoke test FIRST. Keep `_eig_for`'s `_jax_eig_stable` ref + `.polarization`/
   `_cache_registry` imports late-bound.
4. Dead code (report §4.5): `_seg_outer_eps` (pmm2d_staggered.py), unused `G3`,
   `_ARCHIVE_SLANT_FOLD` (move to docs), `rcwa_efficiency_1d_jax` deprecation. (`__all__`
   PMMStack dup already removed.)
5. Over-long-function breakups (report §4.6) + duplicate-logic consolidation
   (`is_out_of_plane` x7, `_t3` closures → `_tensor3_dict`, `_l2g_periodic` reuse).

## QUEUED (separate, user-requested)
**Deeper slant=0 covariant stabilization**: currently slant=0 covariant is ROUTED to
the vertical solver (the oblique frame degenerates: isotropic half-spaces → exactly
degenerate TE/TM → near-singular interface, ~1e8 amplification, BLAS-build-dependent).
To restore the div-conforming machine-precision benefit, cluster the near-degenerate
half-space modes (gap-based) and symmetry-adapt / regularize the interface inversion in
`_cov_layer_4n` / `_pmm_jones_oblique_core`. Verify with the perturbation-stability
probe (patch `np.linalg.eig` +1e-8 noise → result must NOT jump) on WSL, AND keep the
slant<1e-3 routing as a backstop. Prototypes: `C:/tmp/diag_cov_*.py`.

## VERIFY CHECKLIST per stage
ruff (WSL) → targeted tests → full PMM+RCWA on Windows (`766`) → WSL covariant suite
for any covariant touch → commit → push → watch CI (`gh run watch`). The covariant
slant!=0 TM convergence is conditioning-NOISY (non-monotonic) → prefer absolute
thresholds / `dcov<dcon` over tight ratios in any new covariant test.
