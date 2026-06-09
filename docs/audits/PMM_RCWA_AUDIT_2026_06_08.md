# PMM + RCWA Suite Audit — 2026-06-08

Scope: `lumenairy/elements/rcwa.py` (5428 lines), `lumenairy/elements/pmm.py` (5453 lines),
`lumenairy/elements/pmm2d.py` (470 lines), `lumenairy/elements/pmm2d_staggered.py` (888 lines).
All findings below are adversarially verified (each confirmed bug was reproduced against the real
source); refuted findings have been dropped and severities reflect the post-verification `corrected_severity`.

---

## 1. Executive Summary

The two suites are **mature and broadly correct in their mainline regimes** — the 1-D scalar/Jones
paths, the in-plane vertical Jones path, the validated covariant/convection slant generators, and the
2-D crossed-grating cores all cross-check against independent oracles (RCWA validates PMM TE/TM and
Jones; Berreman 4×4 validates the full-tensor path). The defects that survived verification cluster in
**three predictable seams**: (a) the JAX twins, which diverge from their NumPy references in input
sizing, duty handling, and precision; (b) **oblique / opposite-sign-slant corners** of the slant
generators, where the lossless-energy invariant masks per-order errors; and (c) **input-validation
endpoints** (duty 0/1, degree=2, eps=0, grazing) that pass the guard but crash or alias downstream.

**Top risks (ship blockers / correctness):**

1. **Covariant OOP cross blocks omit the `kx0` Bloch shift** (`pmm.py:4491`) — silently wrong
   per-order results for any slanted out-of-plane cell at oblique incidence; energy still conserves, so
   it is invisible to the energy tripwire. Validated: 2–4×10⁻³ error vs the convection oracle, fixed to
   ~5×10⁻⁵ by a one-line patch.
2. **Binary-grating Fourier coefficients silently alias for `n_orders > ~1024`** (`rcwa.py:683`) — the
   default 1-D path accepts up to `n_orders=2499` but the fixed 4096-sample FFT wraps high harmonics,
   corrupting EPS/EPS_II and returning a silent wrong R/T. The 2-D path is guarded; the 1-D binary path
   is not.
3. **JAX precision trap** (`rcwa.py:113`, `pmm.py`) — `_C=np.complex128` is silently demoted to
   complex64 under JAX's default `jax_enable_x64=False`; the ill-conditioned eigenproblem (cond ~1e13)
   then returns inaccurate efficiencies/gradients in the *advertised differentiable regime*, mitigated
   only by a suppressible warning. The library already auto-promotes elsewhere (`fft_infra.py`,
   `phase_retrieval.py`) — `rcwa`/`pmm` are the inconsistent holdouts.
4. **Three JAX-Jones divergences**: frozen duty at 0.5 under jit/grad (`pmm.py:2499`), NaN→int crash on
   metal/ENZ tensors (`pmm.py:2490`), opposite-sign-slant covariant frame mixing (`pmm.py:3132`).
5. **2-D return-arity mismatch** (`pmm_efficiency_2d` returns a 4-tuple vs `rcwa_efficiency_2d`'s
   3-tuple) — breaks drop-in cross-suite unpacking; the one cross-suite contract that should block the
   ship.

**Headline recommendations:** Land the P1 correctness fixes first (all small, several one-line), each
gated by the relevant validation test. Then the naming/convention/doc reconciliation (cheap, high
clarity value, no behavior change). Then the ranked performance work (analytic homogeneous half-space
basis and the Redheffer/eig wins dominate). **The 1D/2D package split is the riskiest item and must go
last**, behind the full PMM and RCWA test suites, because tests import private symbols by name and
monkeypatch module globals — the split must preserve those module attributes exactly.

---

## 2. Confirmed Bugs (P0 / P1)

No P0 (crash-everywhere / data-corruption-everywhere) survived verification. The confirmed **P1**s:

### B1 — Covariant OOP cross blocks drop the `kx0` Bloch shift → wrong OOP+oblique results
- **Where:** `pmm.py:4491` (`cD = cos * Dop`, used in the `M[...] += ... cD` out-of-plane blocks of
  `_cov_generator_4n`).
- **Wrong:** A single x-derivative on a periodic envelope at oblique incidence is Floquet
  `d/dx + i·kx0`. The in-plane blocks use `G = kx0·I - 1j·Dop` (carries kx0) and the divconf longitudinal
  closure injects kx0 (`pmm.py:4461-4463`), but the OOP cross blocks use the **bare** `Dop`. The convection
  twin `_build_generator_metric` does it correctly via `Kx = Dopx/(1j·k0)` with `Dopx = Dop + 1j·kx0·I`
  (`pmm.py:4144`/`4005`). Reachable through `pmm_jones_1d_slanted(factorization='auto'|'covariant')` for any
  slanted OOP cell at `angle != 0`; all four covariant OOP tests run at normal incidence so it is latent.
  Energy conservation is byte-identical patched-vs-unpatched (~1e-9) — the lossless trap hides it.
- **Validated error:** 2.3e-3 @10°, 1.7e-3 @20°, 3.7e-3 @30° vs a converged convection oracle; patch
  restores 4e-5–9e-5 at every angle.
- **Fix:** `cD = cos * (Dop + 1j*kx0*np.eye(n, dtype=_C))`. Add an OOP+oblique covariant test (`angle!=0`)
  cross-checked vs the convection path.

### B2 — Binary-grating Fourier coefficients silently ALIAS for `n_orders > ~1024`
- **Where:** `rcwa.py:683` (`n_samples=4096` default), `rcwa.py:705-706` / `rcwa.py:664-666`.
- **Wrong:** `_fourier_coeffs_1d` reads `full[ks % Nx]` with `Nx=4096`, requesting `|k|` up to
  `2·n_orders`. When `2·n_orders > Nx/2 = 2048` (i.e. `n_orders > 1024`), the modular wrap aliases high
  harmonics onto the wrong order, corrupting both EPS (Laurent) and EPS_II (Li). `_validate_geometry` caps
  1-D only at `2·n_orders+1 <= _MAX_HARMONICS=5000` (`n_orders <= 2499`), so `n_orders ∈ [1025, 2499]` is
  accepted and aliases. The assembled EPS Toeplitz has ~0.8% rel error at `n_orders=1024`, ~1.5% at 2000.
  `_validate_cell_sampling`'s own docstring describes exactly this `% S` wrap as "a silent wrong answer" —
  the authors guarded 2-D but not 1-D binary.
- **Fix:** Scale the internal grid, e.g. `n_samples = max(4096, next_pow2(4*n_orders + 2))`, OR use the
  closed-form sinc series (exact at any order), OR validate `4*n_orders+1 <= n_samples` and auto-bump/raise.

### B3 — JAX `pmm_jones_1d` silently freezes `duty_cycle` at 0.5 under jit/grad
- **Where:** `pmm.py:2499-2532` (`_pmm_jones_1d_jax`); dispatch `pmm.py:1458-1503`.
- **Wrong:** `duty_c = _re_or_none(duty_cycle)` (`2499`) returns `None` for a tracer; `2501` sets
  `duty_c=0.5`; the static topology and static numpy projection `Tp` (`2532`) build at `d_wall=0.5·period`;
  the traced duty is never used and nothing raises (the duty guard at `1445-1448` is skipped for JAX inputs).
  The scalar path handles this (Route-B rebuild via `_jpmm_build_dynamic`, `2134-2145`); the Jones path has
  no `duty_traced` handling. Empirically: `jax.jit` returns the duty=0.5 value for **all** duty inputs
  (≈1.2–1.4e-2 forward error), and `jax.grad` wrt duty is exactly 0.0.
- **Fix (a, minimal):** detect a traced duty
  (`is_jax_array(duty_cycle) and _re_or_none(duty_cycle) is None`) and raise `NotImplementedError`.
  **Fix (b, full):** port the scalar Route-B (`_jpmm_assemble_tensor`/`_jpmm_jones_solve` already accept
  `dyn`) so the duty gradient flows.

### B4 — JAX `pmm_jones_1d` crashes (NaN→int) for in-plane metal/ENZ tensors
- **Where:** `pmm.py:2490-2497` (`_pmm_jones_1d_jax` order-set sizing).
- **Wrong:** `_re_or_none` strips to the **real part** of complex eps (`exx_r = Re(eps_xx)`), then
  `np.real(np.sqrt(v))` is taken on that real part. For `eps_xx = -5+1j`, `np.sqrt(-5.0)=nan`, `n_max=nan`,
  and `_n_propagating_orders` → `int(np.floor(nan))` raises `ValueError: cannot convert float NaN to
  integer`. The NumPy reference `_pmm_jones_solve` (`985-987`) and the slant-segment solvers
  (`4668-4670`) correctly `np.real(np.sqrt(<complex eps>))`. Even for positive lossy eps (`1+12j`) the
  two-step under-estimates `n_max` (1.0 vs `Re(sqrt(1+12j))=2.55`), risking a missed propagating order.
  This is the exact reflective-metal grating the PMM is advertised for.
- **Fix:** compute `n_max` from the complex eps before stripping Im, mirroring `_pmm_jones_solve`; add a
  complex-preserving helper for the eps entries (half-space indices stay as-is since they are already `n`).

### B5 — Opposite-sign equal-magnitude slants route to covariant with an inconsistent half-space frame
- **Where:** `pmm.py:3132` (and `3138-3139`, `3282`, `3300-3301`).
- **Wrong:** `PMMStack.solve` builds `_slants = [abs(L[2]) for L in self._layers]` then
  `_uniform_slant = max>1e-12 and (max-min)<=1e-12`. Because magnitudes are compared **after `abs()`**, a
  `+φ`/`-φ` stack passes the uniform test and dispatches to `_solve_covariant`. Inside, the homogeneous
  half-spaces use the **signed** layer-0 slant (`3282`, `3292-3293`) while each layer uses its **own
  signed** slant (`3300-3301`). The covariant frame `u = x - tan(φ)z` is mirror-sheared for `±φ`, so the
  `-φ` layer's modes cascade against half-spaces fixed in the `+φ` frame — incompatible gauges, silently
  wrong S-matrix, no guard fires. `add_layer` documents that stacks may legitimately mix slanted layers.
- **Fix:** gate uniformity on the **signed** slant: `_signed = [L[2] for ...]; require
  max(_signed)-min(_signed) <= 1e-12` (keep a separate `abs()` check only for the nonzero condition).
  Opposite-sign slants fall back to the convection path (or raise).

### B6 — `_check_energy` crashes the supported CuPy/GPU `RCWAStack.solve` path
- **Where:** `rcwa.py:300` (called from `rcwa.py:5374`); same pattern at the 6 other call sites
  (`1642, 2636, 3156, 3756, 3971`).
- **Wrong:** `_check_energy` does `np.real(np.sum(np.asarray(R)))`. On the CuPy path
  (`use_gpu=True`), `R`/`T` are cupy arrays and `np.asarray(cupy_array)` raises
  `TypeError: Implicit conversion to a NumPy array is not allowed. Use .get()`. Every other site funnels
  cupy through `to_numpy(...)`; `_check_energy` is the lone bypass. `RCWAStack.solve(use_gpu=True)`
  deterministically reaches `5374` on the success path and crashes, breaking the `"NumPy / CuPy only"`
  regime the docstring advertises. Reproduced live with the installed cupy.
- **Fix:** `tot = float(np.real(np.sum(to_numpy(R))) + np.real(np.sum(to_numpy(T))))` inside
  `_check_energy` (`to_numpy` is already imported at `rcwa.py:83`). Fixes all 7 sites at once.

### B7 — JAX x64 mismatch only WARNS → silent complex128→complex64 truncation in the supported regime
- **Where:** `rcwa.py:313-330` (`_warn_if_jax_f32`), `rcwa.py:113` (`_C`); applied via `.astype(_C)` at
  `1493-1497`, `699-708`, etc. (Same pattern in `pmm.py`.)
- **Wrong:** `_C` is hard-coded `complex128`, but under JAX's default `jax_enable_x64=False`,
  `jnp.asarray(x).astype(complex128)` silently demotes to complex64. The RCWA eigenproblem is documented
  ill-conditioned in single precision (cond ~1e13). The only mitigation is a once-per-site
  `warnings.warn` (trivially suppressed). The runtime energy tripwire `_check_energy` is **skipped** on
  the JAX path (`1641-1642`), removing the one net that would catch the resulting `R+T>>1`. The library
  already auto-enables x64 with a one-shot warning in `fft_infra.py:_resolve_jax_complex_dtype` (`487-499`),
  `asymptotic_jax_twin.py` (`772-788`), and `phase_retrieval.py`; `rcwa`/`pmm` only warn.
- **Fix:** either hard-raise on the JAX path when `jax_enable_x64` is disabled, or auto-promote inside the
  core under a `jax.experimental.enable_x64` scope (matching the library's existing precedent). A
  silenceable warning is insufficient for a correctness-critical precision requirement.

### Lower-severity confirmed bugs (P2) — fix opportunistically alongside the above
- **Vertical binary solvers accept `duty_cycle` 0.0/1.0 → zero-width element → singular operators**
  (`pmm.py:2655`, `pmm.py:1446`; root `378-381,688`). The guard is `0.0 <= duty <= 1.0` (accepts
  endpoints); duty=0/1 collapses an element (`J=0`, `Dphys=Dref/J=inf/nan`) and raises an opaque
  `ValueError`/`LinAlgError` from deep in linalg. The JAX paths already raise on
  `not (0.0 < duty < 1.0)`. **Fix:** tighten to strict-interior in both entry points, or special-case
  `{0,1}` to a homogeneous-slab solve.
- **`degree=2` passes the guard but crashes the staggered basis** (`pmm2d_staggered.py:790` guard, `240`
  assert). `if int(degree) < 2: raise` permits `M=2`; `_build_sets` then produces `|Btilde|=N`, `|B|=2N`
  and the cardinality assert fails. **Fix:** tighten guard to `< 3`, set `assert M >= 3`, add a degree=2
  ValueError test.
- **`rcwa_efficiency_2d_shapes`: `eps=0` shape silently → inf/NaN convolution** (`rcwa.py:3825`);
  `_validate_shapes` never checks `|eps|>0`. **Fix:** add an eps-floor guard in `_validate_shapes` +
  the `eps_background` check.
- **Grazing incidence divides by ~0 in incident-flux normalization** (`pmm.py:659,662-663` scalar;
  `633-637` Jones). `kz_inc` has no near-zero guard unlike the diffracted orders. **Fix:** raise/clamp
  when `|kz_inc| < ~1e-9`.
- **PMMStack `n_max` ignores `eyy`** (`pmm.py:3223`, `3317`) — uses only `exx`, inconsistent with every
  single-layer solver which maxes over `exx` AND `eyy`. Under-resolves high-`eyy` stacks. **Fix:** max over
  `sqrt(e[0,0])` and `sqrt(e[1,1])` (ideally `e[2,2]`).
- **`stabilize=True` retry drops `asr_eta`/`asr_samples`** (`rcwa.py:1444-1448`) → silently solves
  WITHOUT ASR on any retry. **Fix:** forward `asr_eta`/`asr_samples`, or document/raise the incompatibility.
- **`rcwa_efficiency_vs_wavelength` aborts the whole sweep on one bad wavelength** (`rcwa.py:1708`) — no
  `try/except _EnergyError`, no `stabilize` kwarg exposed despite the docstring claim. **Fix:** forward a
  `stabilize` flag and/or per-wavelength NaN+warn instead of aborting.
- **`RCWAStack.solve(stabilize=True)` does not catch `_EnergyError`** (`rcwa.py:5177-5184`) — a windowed
  low-order solve that trips the energy guard crashes instead of being skipped, the opposite of the
  stabilize contract. **Fix:** wrap each window in `except _EnergyError: continue`, raise only if all fail.
- **`_nv_field_2d` background reference is the corner pixel** (`rcwa.py:2249`) — biases multi-material
  (≥3 eps) NV fields. **Fix:** reference-free edge indicator (`|grad eps|`), or document the two-material
  assumption.
- **`rcwa_jones_1d` Wood-anomaly nudge omits `ezz`** (`rcwa.py:3249-3250`, segments `3429-3430`) — a
  layer mode grazing governed by `ezz` is not nudged. **Fix:** add `eps[2,2]` to the grazing-detection set.
- **2-D `'li'` leaves the in-plane wall-normal operator at Laurent** (`rcwa.py:2542`, `EPS_xx=EPS`),
  unlike 1-D `'li'`. Intentional (full in-plane needs `fff_nv`) but undocumented at the line. **Fix:**
  inline comment that 2-D `'li'` applies the inverse rule to `E_z` only.
- **`rcwa_jones_2d` uses the slow Laurent rule for the wall-normal tensor component** (`rcwa.py:3713-3721`)
  unlike the 1-D inverse rule — documented but a convergence gap. **Fix:** document at the call site, or
  offer a Li option for the 2-D Jones path.

---

## 3. Performance Opportunities (ranked by expected speedup)

### RCWA

| # | Item | file:line | Change | Est. gain |
|---|------|-----------|--------|-----------|
| P1 | **`_redheffer_star` inverts the identity at every propagation star** | `rcwa.py:1223-1224` | Detect a zero S11/S22 block (propagation matrices have S11=S22=0); skip the matching `inv` (D or F = I). Recursion stars a propagation matrix at every layer, so ~half the Redheffer 2N×2N inverses invert I. | ~1 dense 2N inv saved per layer-prop star; large fraction of S-matrix cost in deep stacks |
| P2 | **`_homogeneous_eigenmodes` forms a full 2N×2N `Q` via matmuls for an analytically diagonal result** | `rcwa.py:1196-1197` (also `1140`, `5081-5083`) | `Kx,Ky` are diagonal; `Q`'s blocks are `diag(kx·ky)`, `eps - diag(kx²)`, etc. Replace 4 O(N³) matmuls with O(N) elementwise. | several O(N³) → O(N) per call; biggest at large `n_orders`/2-D where N=(2Mx+1)(2My+1) |
| P3 | **`rcwa_efficiency_vs_wavelength` rebuilds the dispersionless convolutions per wavelength** | `rcwa.py:1708-1718` (`_binary_grating_convolutions` `682`, O(N³) inv at `708`) | For dispersionless indices, EPS/EPS_II are wavelength-independent — hoist the convolution build out of the loop (`dispersionless=True` fast path). Note: the per-wl 2N×2N eig still dominates, so this is a fractional (not Nwl-fold) win. | one N×N inv + FFT removed per wl; modest fraction of total |
| P4 | **`_binary_grating_convolutions` always builds `EPS_II` even on the Laurent (TE-dielectric) path** | `rcwa.py:708` | Pass `use_li` in; build the inverse-rule matrix only when needed. | removes one N×N inv + one FFT per call on the common TE path |
| P5 | **Single-call entry points never use `_HOMOG_CACHE`** | `rcwa.py:1584-1585` (also `2577-2578`, `3719-3720`, `3933-3934`) | Route standalone solvers' region-mode calls through `_cached_homogeneous_eigenmodes` (already thread-safe; only `RCWAStack` uses it today). | repeated-geometry sweeps reuse half-space modes |
| P6 | **Sweeps are Python for-loops with per-step host materialization — no `vmap`** | `rcwa.py:1707-1718`, `1927-1937`, `2006-2016` | Wavelength enters only via `k0`,`kx`,`grazing nudge` — a textbook vmap axis. Add `jax.vmap` over a `(Nwl,)` array on the jitted core; hoist the `searchsorted` (`1713`) out of the loop for the NumPy path. | >>2× for any multi-point JAX sweep (fusion + no per-step retrace/host copy) |
| P7 | **Per-component FFTs recomputed in tensor/stack builders** | `rcwa.py:2547-2549`, `5103-5108`, `3713-3717` | Batch the 5 per-component `fft2` into one call; reuse one coefficient table per cell; defer the `[[1/eps]]` build to `'li'` only. | removes redundant FFTs (minor vs eig) |
| P8 | **`_with_blas_limit` defaults to a no-op** | `rcwa.py:166-185` | Documented ~2-3× BLAS-oversubscription win is off by default. Auto-detect a sensible cap on many-core boxes, or set the cap once around sweep loops. (Short-circuit it to a true no-op on the JAX path — pure overhead there.) | ~2-3× on many-core when enabled |
| P9 | **Many independent N×N eig/inv/solve calls not batched** | `rcwa.py:5286-5293` | Stack equal-size per-layer `P@Q` into a `(L,2N,2N)` batched `eig`; batch the 2 incident-pol RHS into one `(2N,2)` solve. | biggest on many-layer stacks and GPU (amortized kernel launches) |
| P10 | **JAX general eig has no GPU lowering — dominant per-layer cost runs on CPU** | `rcwa.py:210-218`, `4007-4012`, call sites `1071/1151/2942/2955` | `jnp.linalg.eig` (non-Hermitian) dispatches to a host CPU callback; a "GPU+JAX" RCWA solve ping-pongs the 2N×2N matrix to host each layer. **No correctness impact.** Document; warn when `default_backend()=='gpu'`; consider the symmetrized `Ω²=P@Q` route for the lossless subset. | latent on JAX+GPU only; doc/warn first |
| P11 | **Scalar JAX solvers are not internally jitted** | `rcwa.py:1327-1643` | NOTE: the canonical inverse-design loop already does `jax.jit(jax.value_and_grad(...))` once at the caller (`examples/13`), so the primary hot path is NOT re-traced — this is library-by-design. The genuine residual is the analytic-grad path `jax_merits.py:148` (`jax.grad(_scalar)` without jit, once per SciPy iter); fix belongs in `jax_merits.py`, not `rcwa.py`. | small/secondary |
| P12 | **2-D non-Hermitian eig is O((2N)³), N=(2Mx+1)(2My+1)** | `rcwa.py:2541-2616` | Default circular truncation for isotropic-resolution problems; encourage the even-parity symmetry fast path; consider partial/iterative eig for the forward subspace. | larger refactor; immediate lever is easier enable of truncation/symmetry |
| P13 | **GPU path falls back to host for NV-FFF, ASR bridge, energy/grazing guards** | `rcwa.py:2555-2575`, `806-820`, `300`, `456-462` | Keep NV/ASR construction on-device (cupy has fft/meshgrid); reduce the energy check to a single scalar on-device. | removes per-solve PCIe round-trips on GPU |

### PMM

| # | Item | file:line | Change | Est. gain |
|---|------|-----------|--------|-----------|
| PP1 | **Homogeneous super/sub modes computed by a full dense eig every solve** | `pmm.py:738-744` (also `1003-1005`, `3167-3168`, `4239-4240`, `4549-4551`) | The two half-spaces are uniform — their modes are analytic Rayleigh plane waves projected onto the nodal basis. Build `Wsup/Vsup` + `Hsup=Tp@Wsup` directly, gated on homogeneous-region detection. Removes 2 of 3 dense eigs per solve **and per stabilize-scan degree** (default `stabilize=True`). | ~2× (likely conservative); compounds with PP4 |
| PP2 | **Staggered path solves homogeneous sup/sub with the full pillar eigensolver** | `pmm2d_staggered.py:813-822` | Add an analytic homogeneous-region path (as the hybrid's `_homogeneous_modes`, `pmm2d.py:278-290`), or at minimum assemble the eps-free operators once and reuse across the 3 solvers. Benchmarked: sup+sub ≈ 64% of per-solve work. | ~2.7-3× per solve |
| PP3 | **Hybrid path forms dense O(N³) nodal inverse + N×Nf pinv on the big nodal grid** | `pmm2d.py:155-171,241,255-259` | `M=kron(My,Mx)` so `Minv=kron(inv(My),inv(Mx))`; apply Eps/Einv/Gx/Gy as factored Kron operators instead of dense N×N. Project eps/derivative ops via per-axis projectors so big N never appears as a dense square. | O((nx·ny)³) → ~O(nx³+ny³) |
| PP4 | **`stabilize` scan re-runs the entire 3-grid/3-eig solve at every degree** | `pmm.py:1255-1266`, `1290-1302` | Largely subsumed by PP1 (cuts each scan step ~3×). Do NOT widen the scan or add a sparse-degree probe (risks the load-bearing convergence selector). Document that `stabilize=False` at a once-validated degree is the fast path for fixed-geometry sweeps. | ~3× per scan step via PP1 |
| PP5 | **Eps-free per-axis + structural matrices rebuilt for every staggered solver** | `pmm2d_staggered.py:388-401,466-489` | `Gw=kron(Mbb_y,Mbb_x)`, `Curl`, `Stt=-Curlᴴ Gw⁻¹ Curl`, `Ktz`, the per-axis Galerkin primitives, and even `Basis1D` are rebuilt 3× per call (no `lru_cache` anywhere). Factor the eps-free assembly into a basis-keyed cache; pass eps in separately so only `Et_*/Meps33/Kzt` rebuild. | removes ~2× redundant O(q⁶) `Gw⁻¹`/`Stt` builds |
| PP6 | **Dead recomputation in staggered `_axis_mats`** | `pmm2d_staggered.py:393-401,440-441` | `Ctb_*/Cbt_*/Ctt_*` assembled but never read; `dbt_x = bx.mixed(bx.B,bx.Btilde)` recomputes `self.Cbt_x`. Delete unused; set `dbt_x = self.Cbt_x/k0`. | small constant factor × 3 regions |
| PP7 | **Staggered basis sets re-stacked (`np.array(list of (N,M))`) on every Galerkin call** | `pmm2d_staggered.py:261-262,286-287,353-354,565-566,638` | Cache stacked tensors once on the `Basis1D` instance (`self.Btilde_ten`, `self.B_ten`). | removes ~dozen O(N²M²) copies per solver × 3 |
| PP8 | **`_sem_fourier_projection` uses a Python element loop + per-node scatter** | `pmm.py:538-547` (+ JAX twins `1759-1768`, `1820-1828`) | Vectorize `_lagrange_vals` over all quadrature points; replace the inner `for a` scatter with `np.add.at`/one-hot matmul. Pure speed, no numerical change. | per-solve fixed cost reduced |
| PP9 | **`_assemble_jones_farfield` runs two `lstsq` on the same `Hsup`** | `pmm.py:623-626` | Stack the 2 RHS into a `(2N,2)` `lstsq` (or pinv once, as the JAX twin already does at `2404-2409`). Byte-equivalent. | one SVD/pinv removed; fixes a NumPy/JAX asymmetry |
| PP10 | **TM `invop`/`iS0` recomputed for the homogeneous half-spaces** | `pmm.py:492`, `877-912` | For uniform eps, `invop=(1/eps)·I` exactly; the 3 grids share one `S0`. Short-circuit `invop` and reuse `iS0` once per (degree, grid). Folds into PP1. | exact identities, no result change |
| PP11 | **JAX twins absent for segments/slanted/stack** | `pmm.py:141-146` vs dispatch only at `2667`/`1460` | Enhancement, not a regression. The twin machinery (`_jpmm_assemble_tensor`, `_jpmm_sem_modes_tensor`) is region-count-agnostic, so a `*_segments` twin is the cheapest extension. Silent NumPy fallback should at least be warned/documented in each docstring. | autodiff coverage; O(n_params) FD → one VJP |
| PP12 | **2-D PMM single-shot solves — no wavelength/angle batching** | `pmm2d.py:418-434`, `pmm2d_staggered.py:811-828` | Add a sweep-aware entry assembling eps-free operators once, re-doing only eps-weighted blocks + eig per λ/θ (mirror DynaMeta `FDTDSweepOpticalSolver`); batch independent (λ,θ) as stacked BLAS. | Nwl-fold amortization of eps-free assembly |
| PP13 | **Layer generator uses dense eig on block-structured 4n operators** | `pmm.py:4207-4208,4507,3490-3492,925` | Where the lower (H/Ez) blocks are an explicit Schur image of the upper (E) blocks, eliminate to a 2n eig and reconstruct the partner (as the in-plane tensor path does). Structural redesign; verify per-order first. | ~halves slant/OOP eig dimension |

**JAX/GPU 2-D feasibility:** the hybrid path (`pmm2d.py:272` standard `np.linalg.eig`) is a low-effort
JAX/GPU target reusing `_jax_eig_stable`; the staggered path (`pmm2d_staggered.py:678` generalized
`scipy.linalg.eig(L,G)`) needs a pencil fold to standard form (`G⁻¹L`) — defer unless differentiability is
required.

---

## 4. Organization & 1D/2D Split

### 4.1 Proposed `rcwa/` package layout (preserves the 23 public names + `__all__`)

```
lumenairy/elements/rcwa/
  __init__.py        # imports + re-exports the 23 public names, sets __all__ (rcwa.py:86-109). No logic.
  _backend.py        # rcwa.py:111-361: _C/_MAX_HARMONICS/_BLAS_STATE, blas-thread control
                     #   (set_blas_threads, rcwa_blas_threads, _blas_limit, _with_blas_limit),
                     #   _stabilize_bumps, _eig_for, _block, _rcwa_xp, _is_traced, _concrete,
                     #   _EnergyError, _check_energy, _warn_if_jax_f32, _normalize_pol,
                     #   _normalize_2d_formulation
  _guards.py         # rcwa.py:368-646: _sqrt_forward/_inv_lam/_sqrt_decay,
                     #   _require_propagating_incidence, _grazing_safe_wavelength, _validate_geometry,
                     #   _validate_cell_sampling, _validate_shapes
  _fourier.py        # rcwa.py:653-822 + 2140-2320 + 3773-3833 + 2688-2826:
                     #   _fourier_coeffs_1d, _toeplitz_1d, _binary_grating_convolutions, ASR helpers,
                     #   _harmonic_orders_2d, _eps_convolution_2d, _nv_field_2d, _nv_convolutions_2d,
                     #   _toeplitz_of_profile, _inv_toeplitz_of_profile, _tensor_convolutions(_full),
                     #   _shape_form_factor, _analytic_convolutions_2d
  _eigenmodes.py     # rcwa.py:829-1198 + 2884-3074: _layer_Q_matrix, even-sector helpers,
                     #   _layer_eigenmodes, _homogeneous_eigenmodes, _select_forward_flux,
                     #   _layer_eigenmodes_tensor (WHOLE — do NOT fork in-plane vs OOP branches),
                     #   _tensor_offplane_present/_reject_jax_offplane/_require_inplane_tensor
  _smatrix.py        # rcwa.py:1215-1320: all Redheffer + generalized S-matrix, plus the NEW
                     #   _rt_from_amplitudes / _incident_vector / _single_layer_S extracted in the dedup
  _jax.py            # rcwa.py:3993-4045: _JAX_EIG_STABLE, _jax_eig_stable (lazy-bound by _backend._eig_for)
                     #   + rcwa_efficiency_1d_jax (4046-4123) — the ONLY deprecated public symbol, isolated
  efficiency_1d.py   # rcwa.py:1327-1721 + 1874-2021: rcwa_efficiency_1d, rcwa_efficiency_vs_wavelength
  efficiency_2d.py   # rcwa.py:2320-2637 + 3833-3972: rcwa_efficiency_2d, rcwa_efficiency_2d_shapes
  jones.py           # rcwa.py:2651-3609 + 3612-3757: uniaxial_tensor, _jones_1d_from_profiles,
                     #   rcwa_jones_1d(_segments), grating_segments family, reflective-Jones device
                     #   helpers, rcwa_jones_2d, rcwa_jones_vs_wavelength(_segments)
  convergence.py     # rcwa.py:1722-1873 + 2022-2139: _order_key, _max_aligned_delta,
                     #   _rcwa_convergence_stack, rcwa_convergence, rcwa_extrapolate
  stack.py           # rcwa.py:4126-5428: _HOMOG_CACHE/_LOCK, _cached_homogeneous_eigenmodes,
                     #   RCWAResult, _RCWALayer, RCWAStack, cache-registry enrollment
```

**Split boundary rules (load-bearing):**
- **Shared core (imported by ≥2 of {efficiency_1d, efficiency_2d, jones, stack}):** `_backend`, `_guards`,
  `_fourier`, `_eigenmodes`, `_smatrix`, `_jax`, `convergence`. Keep these dimension-agnostic.
- **`_harmonic_orders_2d` + `_eps_convolution_2d` MUST live in shared `_fourier`, NOT `efficiency_2d`** —
  `RCWAStack` always uses the 2-D lattice primitives internally even for 1-D stacks (`noy=0`). Hosting them
  in `efficiency_2d` would force `stack.py` to import `efficiency_2d` (cycle risk).
- **Import DAG (one-way, acyclic):** shared modules import nothing from solver/stack; solver modules
  (`efficiency_1d/2d`, `jones`, `convergence`) import only shared; `stack` imports shared (+ `jones` for
  `uniaxial_tensor`); `__init__` imports everything last. Keep `_backend._eig_for`'s reference to
  `_jax_eig_stable` function-local (late-bound). Keep `.polarization` and `.._cache_registry` imports
  method-local/lazy exactly as today (verified no `rcwa<->polarization` cycle).

### 4.2 Proposed `pmm/` package layout (preserves the 12 public names)

```
lumenairy/elements/pmm/
  __init__.py            # re-export the 12 public names AND every PRIVATE symbol tests import by name
                         #   (see 4.4) so lumenairy.elements.pmm.<name> resolves unchanged
  _kernel.py             # GLL primitives (_gll_nodes_weights, _lagrange_derivative_matrix,
                         #   _graded_boundaries), _safe_* linalg guards (_ill_scaled/_equil_scale/
                         #   _safe_inv/_safe_solve/_safe_geig), s-matrix algebra (_interface_smatrix,
                         #   _propagation_smatrix, _redheffer_star), far-field/order-set helpers
                         #   (_sem_fourier_projection, _assemble_jones_farfield, _scalar_farfield_RT,
                         #   _n_propagating_orders, _kz_forward), _l2g_periodic, _tensor3_dict,
                         #   _coeff_mass_metric
  _stabilize.py          # _aligned_max_diff, _converged_cluster, _stabilize_scalar, _stabilize_jones,
                         #   the _STABILIZE/_PASSIVE/_CLUSTER/_PER_ORDER tunables + _JONES_PASSIVE_TOL
  _core.py               # scalar (_build_sem, _sem_modes, _pmm_solve, _pmm_solve_core) +
                         #   in-plane Jones (_build_sem_tensor, _sem_modes_tensor, _pmm_jones_solve*)
  _segments.py           # _segment_walls.._pmm_jones_solve_segments + _build_*_segments +
                         #   _build_nodal_metric_segments
  _stack.py              # _pmm_union_grid, PMMStack
  _slant_convection.py   # _build_sem_slant, _sem_modes_slant, _modes_M_slant, _pmm_slant_solve AND the
                         #   metric generator (_build_nodal_metric, _build_generator_metric,
                         #   _layer_modes_metric, _pmm_jones_slant_*); + _ARCHIVE_SLANT_FOLD at the bottom
  _slant_covariant.py    # _cov_blocks, _cov_generator_4n, _cov_layer_4n, _pmm_jones_oblique_*
  _jax.py                # the entire _jpmm_* family + _pmm_efficiency_1d_jax / _pmm_jones_1d_jax drivers
  _classify.py           # grating_convergence_class, classify_from_grating (fully standalone)
  _facade.py             # cross-cutting public dispatchers: pmm_jones_1d (1324), pmm_efficiency_1d (2541),
                         #   pmm_jones_1d_slanted (4790), pmm_1d (5183), thin public wrappers
```

Dependency is strictly 2-D → 1-D (verified: `pmm.py` has no reference to `pmm2d`). The kernel s-matrix
helpers must stay re-exported from the package root (the 2-D files import them `from .pmm`).

### 4.3 2-D PMM consolidation (`pmm2d/`)

```
lumenairy/elements/pmm2d/
  __init__.py            # re-export pmm_efficiency_2d, pmm_efficiency_2d_staggered
  _common.py             # single home for the duplicated: sqrt_decay, inv_lam, kz_forward(eps,kx,ky=None)
                         #   (unifies _kz_forward + both _kz_forward2), incident_plane_wave(...),
                         #   farfield_RT_2d(...) consolidating the two epilogues
  _hybrid.py             # was pmm2d.py (pmm_efficiency_2d + _build_axis/_assemble_2d/_layer_modes_projected)
  _staggered.py          # pmm_efficiency_2d_staggered + _stag_fourier_projection/_far_projector_2d/
                         #   _region_modes
  _basis_staggered.py    # Basis1D, _modleg_*, Granet2DTransverseE
```

The two eigensolver cores (nodal-GLL+Fourier-projection vs staggered Granet pencil) are genuinely
different physics and stay separate. Duplicated blocks collapsing into `_common`: `_sqrt_decay`,
`_inv_lam`, `_kz_forward2`, the far-field R/T epilogue (`pmm2d.py:458-470` ≡ `pmm2d_staggered.py:874-888`),
the incident-wave block (`437-452` ≡ `851-867`), and the order-grid/Bloch-shift setup. `_axis_projection`
(`pmm2d.py:199-234`) can be deleted in favor of `_kernel.sem_fourier_projection` once the grid dict keys
are aligned.

### 4.4 Test-coupling hazard (P1 for the split itself)

Tests import private helpers **and monkeypatch module globals**:
- `tests/.../test_v5_12_0_pmm_slant_and_convergence.py:15-26` imports `_pmm_jones_slant_solve`,
  `_stabilize_jones`; `:482-483,534-552` monkeypatches/calls `_pmm._pmm_jones_slant_diag_solve`.
- `tests/.../test_v5_11_0_pmm_element_size_scaling.py:22-26` imports `_ill_scaled`, `_safe_geig`,
  `_safe_inv`, `_safe_solve`.

The dispatcher `pmm_jones_1d_slanted` calls these as **bare module globals** (`pmm.py:5036,5049`). If the
split binds them as locals in `_facade`, a root-level monkeypatch won't be honored. **Required:** (1)
`pmm/__init__.py` re-exports all test-referenced privates (grep the tests for the complete set); (2) keep
the dispatcher + its cure callees in the same submodule, OR have the dispatcher resolve them via
`import lumenairy.elements.pmm as _pkg; _pkg._pmm_jones_slant_diag_solve` so a root monkeypatch applies; (3)
add a smoke test asserting `pmm._safe_inv` and `pmm._pmm_jones_slant_solve` are reachable post-split.

### 4.5 Dead / stale code to remove
- `pmm2d_staggered.py:333-339` `_seg_outer_eps` — defined, never called. **Delete.**
- `pmm2d_staggered.py:466,546` `G3 = np.kron(Mtt_y,Mtt_x)` / `self.G3` — built, never read; the
  line-462 comment is misleading. **Delete + fix comment.**
- `pmm.py:5406-5453` `_ARCHIVE_SLANT_FOLD` raw string — intentional record; move to a docs file or keep at
  the bottom of `_slant_convection.py`.
- `rcwa_efficiency_1d_jax` (`rcwa.py:4046`) — deprecated thin forwarder, removal promised v5.7.0 but still
  present at v5.11.0 (see §5). Isolate in `rcwa/_jax.py`; update or honor the removal.
- `lumenairy/elements/__init__.py:201,206` — `'PMMStack'` listed twice in `__all__`. **Delete the second;**
  extend the no-duplicates pin to `elements.__all__`.

### 4.6 Over-long functions to break up (during the split, behavior-preserving)
- **RCWA (>120 lines):** `rcwa_efficiency_1d` (1327-1643, ~316), `rcwa_efficiency_2d` (2320-2637, ~318),
  `rcwa_jones_1d_segments` (3289-3474, ~186), `RCWAStack._solve_once` (5214-5375, ~161), `rcwa_jones_2d`
  (3612-3757, ~146), `rcwa_efficiency_2d_shapes` (3833-3972, ~139), `RCWAResult.internal_field`
  (4528-4652, ~125), `rcwa_jones_1d` (3160-3287, ~127). Extract `_solver_setup(...)`, `_single_layer_S(...)`,
  and `_rt_from_amplitudes/_incident_vector` (the R/T tail duplicated ~7× verbatim:
  `1613-1640`, `2623-2634`, `3138-3149`, `3741-3750`, `3959-3970`, `5321-5330` — a divergent-copy P1 surface).
- **PMM:** `_build_generator_metric` (3948-4164, ~216; extract the OOP Schur-block assembly),
  `pmm_jones_1d_slanted` (4790-5053, ~263), `pmm_jones_1d` (1324-1546, ~222), `pmm_efficiency_1d`
  (2541-2699, ~158), `PMMStack.solve` (3109-3253, ~144), `_sem_modes_slant` (3417-3531, ~114),
  `PMMStack._solve_covariant` (3255-3344, ~89). Factor the shared far-field/order-set epilogue into one
  `_kernel.farfield` helper (recurs in `_pmm_solve_core`, `_pmm_jones_solve_core`, `_pmm_jones_slant_core`,
  `_pmm_jones_oblique_core`, `PMMStack.solve:3222-3253`, `_solve_covariant`).
- **Duplicate-logic consolidation (low-risk, post-split):** unify the 7 inline out-of-plane-detection
  copies (`pmm.py:3081` + `1483,1517,2889,4959` + `_maxoff` `4018,4479`) into one `_kernel.is_out_of_plane`;
  delete `_t3_slant` (`3799`) and four `_t3` closures (`968,1226,2468,2469`) in favor of `_tensor3_dict`;
  make `_build_sem`/`_build_nodal_metric` call `_l2g_periodic` instead of re-deriving the periodic wrap.

---

## 5. Naming / Convention / Clarity Fixes

**Cross-suite incidence-angle keyword (P2):** 1-D uses `angle`; 2-D and `RCWAStack`/`PMMStack` use
`theta`/`phi`. The module headers standardize on `theta`. **Fix:** accept `theta` on the 1-D entry points
(deprecated `angle` alias) and `angle` on `RCWAStack.set_source` (map to `theta` when `is_1d`); document
the planar=angle / conical=theta,phi rule. Sites: `rcwa.py:1338` vs `2330`/`5043`; `pmm.py:2551` vs
`pmm2d.py:313`/`pmm2d_staggered.py:728`/`pmm.py:3103` vs `rcwa.py:5043`.

**Cross-suite ridge/groove quantity (P2):** `rcwa_efficiency_1d` takes **refractive index** `n_ridge/n_groove`
(`rcwa.py:1330`); `rcwa_jones_1d` and the Jones/segments family take **permittivity** `eps_ridge/eps_groove`
(`rcwa.py:3163`). A wrong-convention value is silently accepted (`n=2.1` read as `eps=2.1`). **Fix:** prominent
cross-referencing docstring note; consider a clearly named convention kwarg behind a minor bump.

**`far_field_orders` vs `n_orders` (P2):** 1-D PMM uses `far_field_orders`; both 2-D files use `n_orders`.
Unify to one name with a deprecated alias.

**`degree` semantics (P2):** `degree` is the GLL polynomial degree in `pmm.py`/`pmm2d.py` but the
modified-Legendre function **count M** in `pmm2d_staggered.py:755` (`M=int(degree)`, basis dim `M-1`).
Cross-solver degree sweeps compare unlike quantities. **Fix:** rename the staggered kwarg to
`n_modes`/`order_M` (with a `degree` alias) or document the divergence loudly.

**`formulation` asymmetry (P2):** 1-D defaults `'auto'` (auto-upgrades to `'li'` for TM/metals); 2-D defaults
`'laurent'` and `_normalize_2d_formulation` (`rcwa.py:357`) **rejects `'auto'`** with a hard `ValueError`.
**Fix:** add `'auto'`→`'li'` mapping for 2-D, or document the 2-D default explicitly.

**Stale docstrings:**
- `pmm.py:2603` — `stabilize` docstring says "minimum-power" selection; the code does **per-order
  convergence consensus** (its own inline comment at `2691-2696` is correct). Rewrite; also tighten the
  `pmm_jones_1d` version (`1394-1396`).
- `'IN-PLANE'` eps qualifier is stale where OOP is supported: `pmm.py:1357`, `2858`, `4831` (the bodies route
  OOP to the metric generator). Drop the qualifier; move limitations to Notes.
- `pmm.py:4644` — `_pmm_jones_oblique_segments_solve` docstring says "IN-PLANE tensors only" but it handles
  OOP. Update.
- `rcwa_efficiency_1d_jax` deprecation says "removed in v5.7.0" but persists at v5.11.0 (`rcwa.py:4064,4096`).
  Remove the shim or set a real future removal target.

**Naming clarity (P3):**
- `pmm.py:4748` `_exx_eq_ezz` holds the `|exx-ezz|` **deviation**, not an equality → rename `_exx_minus_ezz`.
- `pmm.py:4189` `_layer_modes_metric` docstring writes `q=gamma` while every sibling uses `q=gamma/k0`;
  the code (`4209`) is the dimensionless `n_eff`. Fix the docstring.
- `rcwa.py:829` `EPS_xx` denotes the wall-normal-rule convolution, not the xx tensor component (clashes with
  `Cxx` at `2884`). Rename `EPS_normal`/`EPS_wallnormal`.
- `rcwa.py:4887` `RCWAStack.nox/noy` are non-underscore, cross-module-mutated, undocumented, and don't match
  the public `n_orders/n_orders_y`. Rename or make documented properties.
- `kx0` units differ (1-D dimensional `k0·sin` vs 2-D dimensionless `sin`) — `pmm2d.py:406`,
  `pmm2d_staggered.py:805/807`. Unify to dimensionless + add the `kxv = kx0 + m·wl/period` comment.
- `flux_inc` (1-D) vs `einc_sq` (2-D) for the incident-flux normalizer — `pmm.py:633` vs `pmm2d.py:450`.
  Unify or comment the relationship.
- `_homogeneous_eigenmodes` returns `kz` in the 3rd tuple slot (undocumented) vs `_layer_eigenmodes`'s `lam`
  (`rcwa.py:1118-1121` vs `1179-1198`). Document the difference.
- `_sqrt_forward` docstring states the public `exp(+ikz)` convention but runs in the conjugated internal
  convention (`rcwa.py:368` vs `1486-1489`). Clarify.

**Cross-suite convention (P1/P2):**
- **2-D return arity (P1):** `pmm_efficiency_2d`/`_staggered` return a 4-tuple `(orders,R,T,dof)` vs
  `rcwa_efficiency_2d`'s 3-tuple (`pmm2d.py:317`, `pmm2d_staggered.py:729` vs `rcwa.py:2340`). **Fix:** drop
  `dof` from the PMM returns (expose via attribute/dataclass) or add a matching diagnostic to RCWA; pin with
  a cross-suite contract test. **This is the one cross-suite item that should block the ship.**
- `PMMStack.solve` returns a bare tuple vs `RCWAStack.solve`'s rich `RCWAResult` (`pmm.py:3109` vs
  `rcwa.py:5135`). Mirror the accessor surface (`PMMResult`) or document.
- `stabilize` default: PMM `True`, RCWA `False` (`pmm.py:2557` vs `rcwa.py:1342`). Intentional (PMM
  resonances are LAPACK-build-dependent) — document side-by-side so it reads as a choice.

---

## 6. Staged Execution Plan (ordered, test-gated, independently shippable)

Each stage is a separate PR/commit. Run the relevant validation test(s) before merging; the **full PMM and
RCWA test suites gate every stage that touches shared kernels and the entire reorg**.

**Stage 0 — pins & cheap safety nets (gate: `test_public_api`, walker symmetry).**
- Delete the duplicate `'PMMStack'` (`elements/__init__.py:206`); extend the no-duplicates pin to
  `elements.__all__`.
- Add the cross-suite 2-D return-arity contract test (currently failing) and the `pmm._safe_inv`/
  `_pmm_jones_slant_solve` reachability smoke test (will be reused to gate the reorg).

**Stage 1 — P1 correctness fixes (gate: each fix's targeted validation + full suite).** Ship as small
independent commits:
1. B6 `_check_energy` `to_numpy` (`rcwa.py:300`) — fixes all 7 sites; gate on a CuPy `RCWAStack` smoke test.
2. B2 binary-grating aliasing (`rcwa.py:683`) — scale `n_samples` or sinc series; gate on a high-`n_orders`
   convergence test vs a reference.
3. B1 covariant OOP `kx0` shift (`pmm.py:4491`) — one-line; gate on a NEW OOP+oblique covariant test vs the
   convection oracle.
4. B5 signed-slant gate (`pmm.py:3132`) — gate on a `+φ/-φ` stack test (fall back to convection / raise).
5. B7 JAX x64 (auto-promote or hard-raise) in `rcwa.py` + `pmm.py` — gate on a JAX default-config test.
6. B3 JAX-Jones duty (`pmm.py:2499`) — raise or port Route-B; gate on the jit/grad-duty test.
7. B4 JAX-Jones metal/ENZ `n_max` (`pmm.py:2490`) — complex-eps sqrt; gate on a metal-tensor JAX test.
8. **P2 cluster** (duty 0/1 strict-interior, degree=2 guard, eps=0 shape guard, grazing guard, PMMStack
   `eyy`, stabilize-retry `asr_eta`, vs_wavelength per-λ recovery, `RCWAStack` `_EnergyError` catch, Wood
   `ezz`, `_nv_field_2d` reference) — bundle as one robustness PR (per the "ship PATCH bumps, bundle more"
   preference), each with its own targeted test.

**Stage 2 — naming / docs / convention reconciliation (gate: full suite for any signature touch).** No
behavior change beyond deprecated aliases.
- 2-D return arity fix (drop `dof` / dataclass) — the one with a real signature impact; behind a minor bump.
- Incidence-keyword aliases (`theta`↔`angle`), `far_field_orders`↔`n_orders` alias, `formulation='auto'` for
  2-D, staggered `degree`→`n_modes` alias.
- All stale-docstring rewrites (`stabilize` consensus, `IN-PLANE` qualifiers, `4644`, `4189`, deprecation
  version) and the P3 renames (`_exx_eq_ezz`, `EPS_xx`, `nox/noy`, `kx0` units, `flux_inc`/`einc_sq`).

**Stage 3 — performance (gate: full suite + a perf-regression spot-check; results must be byte/near-byte
identical where claimed).** Highest value first:
- PP1/PP10/PP4 analytic homogeneous half-space basis (PMM) + TM `invop`/`iS0` reuse — the dominant ~2-3×
  lever; gate on byte-closeness vs the eig path before enabling on detection.
- PP2/PP5/PP6/PP7 staggered homogeneous-region + eps-free caching + dead-code removal.
- RCWA P1/P2 (`_redheffer_star` identity skip, diagonal homogeneous modes), P4 (`use_li`-gated `EPS_II`),
  P5/P3 (caching + dispersionless hoist), P7 (FFT batching).
- PP3 hybrid Kron factorization; PP8/PP9 projection/lstsq vectorization.
- JAX/GPU coverage: P6 `vmap` sweeps, P9 batched eig, P8 BLAS-cap defaults, P13/P10 GPU host round-trips +
  the JAX-eig host-execution warning (doc-only for P10).

**Stage 4 — the reorg (LAST; riskiest; gate: FULL PMM + RCWA suites + the Stage-0 reachability/monkeypatch
smoke tests).** Mechanical, behavior-preserving moves only:
1. `pmm2d/` consolidation (`_common` extraction) — smallest blast radius, validates the pattern.
2. `rcwa/` package split per §4.1 (extract `_rt_from_amplitudes`/`_single_layer_S`/`_solver_setup` as a
   single behavior-preserving refactor with a bit-identical regression test for 1D/2D/stack).
3. `pmm/` package split per §4.2 — **must** preserve every test-imported private and module-global dispatch
   lookup (§4.4); keep the kernel s-matrix names re-exported so the 2-D files' `from .pmm` imports resolve.
4. Post-split duplicate-logic consolidation (`is_out_of_plane`, `tensor3_dict`, `_l2g_periodic`) and the
   over-long-function breakups (§4.6), each verified against the byte-identical baseline.

---

### Appendix — counts
- Confirmed P1 bugs: **7** (B1-B7). Confirmed P2 bugs: 12. P3 bugs: a handful (non-blocking).
- Performance items: **26** (13 RCWA, 13 PMM/2-D-PMM).
- Naming / convention / organization items: ~38 (incl. the 1D/2D split layouts, 2-D-PMM consolidation,
  5 dead-code removals, 15 over-long functions).
