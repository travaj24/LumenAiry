# WP-B7 — the asymptotic family: Y4 / Y5 performance, the FGA-vs-`phase_screen`
# question, the uniform asymptotics, the S6 gate's missing statistic, and S9

Audit `AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11`, WP-A4 report §6 items 3–8 plus
VERIFY-B1's follow-ups F1 / F2 and the WP-B9 request folded into this package.
Branch `audit-fixes-2026-09`; baseline for every byte-identity proof is
`git archive b2baa505 lumenairy`, extracted read-only and imported from a child
process whose cwd and `PYTHONPATH` are that archive with `lumenairy.__file__`
asserted.  HEAD has since moved to `21110326`; **every module this package owns
is byte-identical between the two** (`git diff b2baa505 21110326 --` over the
eleven files returns empty), so the baseline stands unchanged.

---

## 1. Summary table

| # | item | status | files : lines | tests | oracle | measured before → after |
|---|---|---|---|---|---|---|
| 3 | FGA vs `phase_screen` at NA 0.145 | **MEASURED — routing change escalated** (§2) | none (measurement only) | table in §2 | brute-force Rayleigh–Sommerfeld from an exact conic raytrace, converged to 0.3 % on the pupil sampling | FGA does **not** converge under any sampling: fidelity **0.3234** at defaults, **0.3826** at the best of 15 settings; `phase_screen` **0.9965**.  `phase_screen` is closer to the oracle at **every** NA in the sweep (0.039 → 0.192) |
| 4a/b | Y4 fused `_basis_and_grad34` + hoisted Newton factor | **DONE, byte-identical** | `asymptotic_canonical_fit.py:61-105,221-238,359-420`, `asymptotic_maslov.py:262-345,348-370,616-760` | b7 §1 (3 ids) | 12 arrays recomputed from the unfused methods | `_solve_envelope_stationary_batch` **271.1 → 145.9 ms** (1.86×), `_compute_M_b_batch` **50.0 → 20.7 ms** (2.41×), `propagate_modal_asymptotic` **320.7 → 191.0 ms** (1.68×); Chebyshev table builds **26 → 12** and **144 → 50** |
| 4c | Y4 scale-relative Newton **stop** | **DONE, opt-in** | `asymptotic_maslov.py:55-66,616-760` | b7 §2 (1 id) | the absolute-stop arm on the same build | 1.23× on the Newton, 12 → 11 sweeps; field moves **9.1e-11** relative L2, so it ships OFF |
| 5 | Y4 `aberration_tensor` default cost | **DONE, byte-identical** | `asymptotic_modes.py:320-405,760-845`, `asymptotic_aberration_tensor.py:467-520,570-700,1385-1400` | b7 §3 (2 ids) | the full-rectangle `decompose_lg` and an uncached probe | `aberration_tensor` **12 203 → 7 439 ms** (1.64×), `L` bit-identical; `sigma_grid_n` cap NOT raised — measurement in §4 |
| 6 | Y5 structural collapse | **PARTIAL + plan, stopped** (§5) | `asymptotic_maslov.py` (the batched pair now shares one evaluator) | — | scalar vs batched on 5 chart points | the two `_compute_M_b` copies already differ: `M` bit-equal 2/5 (worst 1.1e-17 relative), `b` 3/5 (7.5e-18), the two Newtons differ by up to 4.8e-15 in `v2`.  A bit-identity harness across the three call sites cannot be made green as they stand |
| 7 | §15.9 uniform asymptotics | **PREMISE CORRECTED + table published, left as is** (§6) | none | — | brute-force RS through the marginal focus of an f/1.92 singlet, converged to `1 − fid = 4.6e-08` | `uniform_fold_airy` / `pearcey` are **not dead code** — `_lens_traced_uniform.py` consumes them.  At a genuine fold: `stationary_phase` **0.5921**, exact `quadrature` on the same chart **0.6278**, `caustic='uniform'` **0.0030** |
| 8 | S9 GBD `_reconstruct_fft` kernel clipping | **DONE, derived tolerance** | `gbd.py:1769-1808,1809-1822,1835-1900` | b7 §4 (5 ids) | the windowed scatter-add at n_sigma 5 / 7 / 9 | peak **37.7× → 6.1×** the output grid at N = 512 (158.3 → 25.8 MB), reconstruct **594.5 → 47.3 ms**; agreement **8.2e-16 … 2.3e-15** |
| 9 | JAX sibling's chief-ray displacement | **DONE, byte-identical on a collimated input** | `_lens_jax.py:733-761,764-800,1035-1060` | b7 §6b (1 id) | exact conic raytrace of the input's own rays | landing error **−2.66 / −2.66 / −2.72 %** → **−0.19 / −0.19 / −0.27 %** at 0.25 / 0.5 / 1.0 × lens NA |
| 10 | re-found S6 fallback statistic | **DONE** | `lenses_maslov.py:167-201,1257-1310,2998-3030,3055-3070` | b7 §6 (1 id) | the exact pointwise `'quadrature'` on the same chart, two fixtures | a hard-edged aperture scores value residual **0.08–0.26** (inside its 0.5 bar) and slope error **2.7–4.3**; new bar `_K1_DERIV_RESIDUAL_MAX = 1.2` |
| 11 | pupil chart sized from mean + spread | **DONE, byte-identical for a centred input** | `lenses_maslov.py:203-264,2472-2515` | b7 §5 (6 ids) | the exact pointwise `'quadrature'` on the same chart, two fixtures | tilt 2× lens NA: `na_input` **0.2986 → 0.1040**, s1 fit residual **3.98e-03 → 1.17e-04**, fidelity **0.000 / 0.000 → 0.911 / 0.979** |
| B9-4 | `jacobian='auto'` and aspherics | **comment corrected + pinned; FGA widening escalated** (§8.2) | `fga.py:792-813,824-834,1699-1714` | b7 §7 (1 id) | the analytic and FD primitives on an A4 singlet | analytic vs FD exit height agree to **< 1e-6 relative**; FGA's own whitelist still refuses aspherics |

**Not green when I finished:** two ids in
`tests/unit/test_audit2609_b1_maslov_input_wavevector.py`, a file outside my
ownership.  The exact patch is §8.1; **verified green (33 / 33) with it
applied** in a scratch copy.

---

## 2. Item 3 — FGA vs `phase_screen` at NA 0.145, decided

### 2.1 The oracle

`repro`-free, written for this package: an exact conic raytrace of the input's
own rays (sphere sag, Snell in 3-D, ray-tube Jacobian amplitude) to the exit
vertex plane, then a brute-force Rayleigh–Sommerfeld sum
`E(p) = (1/iλ) Σ A_k e^{ikr} z / r²` onto the readout grid.  Nothing in
`fga.py`, `lenses_maslov.py` or `_lens_traced.py` touches it.

Fixture: the audit's own
(`tests/unit/test_audit2609_a4_fga_s10.py`) — f = 1.2 mm biconvex N-BK7,
0.30 mm aperture → NA 0.1452, focus 1.027 mm past the exit vertex, λ = 1.0 µm,
N = 192, dx = 2 µm, w = 100 µm, tilt 0.

Convergence in the pupil sampling (intensity-rms spot width over the full
grid): **2.6862 → 2.6641 → 2.6560 µm** at n_pupil = 121 / 161 / 201, a 0.3 %
last step.  The oracle is converged to well under the 0.5 µm that separates the
two members.

### 2.2 The convergence sweep the WP file asked for

`w0_factor`, `dq_step`, `p_max`, `n_p`, and three combinations — 15 settings.
Intensity-rms spot width, oracle **2.656 µm**, `phase_screen` **3.169 µm**:

| setting | rms (µm) | wall clock |
|---|---|---|
| defaults | 12.607 | 10.3 s |
| `w0_factor=8` | **10.571** | 1.3 s |
| `w0_factor=12` | 11.795 | 1.0 s |
| `dq_step=1 / 2 / 4` | 12.607 / 12.607 / 12.607 | 7.3 / 1.7 / 0.4 s |
| `p_max=0.2 / 0.35 / 0.5` | 12.607 / 12.614 / 12.830 | 22.9 / 32.1 / 29.0 s |
| `n_p=21 / 41 / 61` | 12.607 / 12.606 / 12.607 | 4.4 / 16.9 / 38.4 s |
| `w0_factor=12, dq_step=1, p_max=0.35, n_p=61` | 12.271 | 48.0 s |
| `w0_factor=8, dq_step=1, n_p=41` | 10.573 | 106.2 s |

**FGA does not converge.**  The spot stays 10.6–12.8 µm — a factor 4 too wide —
across every knob, and the two knobs that move it at all (`w0_factor`,
`p_max`) move it by 10 % and in both directions.  `dq_step` and `n_p` are
inert to 4 digits, so the swarm is not under-sampled; the error is not a
sampling error.

Scored as a field rather than a width, against the same oracle:

| member | fidelity | EE(5 µm) | EE(10 µm) | EE(25 µm) |
|---|---|---|---|---|
| oracle | 1 | 0.9515 | 0.9976 | 0.9992 |
| `phase_screen` | **0.9965** | 0.9435 | 0.9960 | 0.9983 |
| `fga` (defaults) | **0.3234** | 0.0809 | 0.2777 | 0.8588 |
| `fga` (`w0_factor=8`) | 0.3824 | 0.1128 | 0.3699 | 0.9381 |
| `fga` (`w0_factor=8, dq_step=1, n_p=41`) | 0.3826 | 0.1129 | 0.3701 | 0.9380 |

FGA puts **92 %** of the energy outside the 5 µm core that holds 95 % of the
oracle's.  This is the auditor's own unverified suspicion ("measured
`apply_real_lens_fga` fidelity 0.357 … I cannot call this a defect"),
independently reproduced at 0.323–0.383 against a converged oracle, and the
WP-A4 §2 S10 numbers reproduce to three digits here (`phase_screen` 3.169 µm
against their 3.169; `fga` 12.607 against their 12.607).

### 2.3 The NA sweep, and what the routing does

Aperture held at 0.30 mm, radii swept; readout at the traced best-focus plane;
`_universal_route` called with the shipped defaults:

| NA | route | oracle rms | `phase_screen` (error) | `fga` (error) |
|---|---|---|---|---|
| 0.0394 | `phase_screen` | 7.075 µm | 7.591 (**0.516**) | 30.822 (23.748) |
| 0.0674 | `phase_screen` | 4.538 | 5.021 (**0.483**) | 22.330 (17.792) |
| 0.0980 | `phase_screen` | 3.434 | 3.907 (**0.473**) | 17.105 (13.671) |
| 0.1452 | **`fga`** | 2.663 | 3.168 (**0.505**) | 12.593 (9.930) |
| 0.1920 | **`fga`** | 2.299 | 3.027 (**0.729**) | 10.013 (7.714) |

`phase_screen` is closer at **every** NA, by 10× to 46×.  The thin-screen
obliquity ceiling is real — its error grows 0.47 → 0.73 µm across the sweep —
but it is an order of magnitude under FGA's at the top of the range.

### 2.4 The decision

The WP file's dichotomy ("if FGA converges, the sampling is the defect; if not,
`na_threshold` is mis-set") has a third answer that the measurement forces:
**neither**.

* It is not a sampling defect: `dq_step` and `n_p` are inert, and no setting
  gets within 4× of the oracle.
* It is not `na_threshold`: the flip at NA 0.145 comes from the **caustic
  gate**, not from `na_threshold`.  At every NA *below* the threshold the
  router already picks `phase_screen`, and `phase_screen` is already the better
  member there.  Re-deriving `na_threshold` would change nothing about the two
  rows that are wrong.

What is wrong is `apply_real_lens_fga`'s own accuracy at a real singlet focus,
and the routing consequence is that `_universal_route`'s caustic branch should
stop preferring `fga` over `phase_screen` for a SINGLE-VALUED input.  I have
**not** made that change: it necessarily reds five test files outside my
ownership that pin the current decisions.  Exact edits in §8.3.

---

## 3. Item 4 — Y4 performance

### 3.1 (a) one fused basis per evaluation, and (b) the hoisted `s2` factor

`_compute_M_b_batch` contracted three coefficient vectors (`coef_s1x`,
`coef_s1y`, `coef_phi`) against the same point set through two methods that
each rebuilt the `(M, N)` basis tensors for themselves, and then called
`_phi_v2_hessian_batch`, which rebuilt them again.  `_basis_and_grad34`
(`asymptotic_canonical_fit.py`) now builds `basis_f` / `basis_d3` / `basis_d4`
once and `CanonicalPolyFit.eval_s1_and_phi_with_v2_grad` contracts all three
vectors against them, optionally returning the `s2`-only factor `T1[K1]·T2[K2]`
so the Hessian pass reuses it.

In `_solve_envelope_stationary_batch` that factor is **loop-invariant** — the
Newton moves `v2`, never `s2` — so it is built once for all N pixels and the
active columns are gathered each sweep.

Both are bit-exact by construction: every output is the same
`np.tensordot(c, basis, axes=([0],[0]))` of the same coefficient vector against
the same tensor, built by the same expressions in the same order, and the
Chebyshev recurrence is elementwise in the sample axis so `T[:, idx]` is
`T(u[idx])` bit for bit.  The auditor's suggested single stacked
`(3, M) @ (M, P)` GEMM was **not** taken: a GEMM is entitled to reorder the
reduction against a GEMV, and the b7 test asserts `array_equal`, not a
tolerance, precisely so that a future edit in that direction fails.

**Byte-identity**, archive vs this tree, both imported from child processes
with `lumenairy.__file__` asserted (stock N-BK7 singlet fit, order 6, M = 210,
41 × 41 raster, `w_s = 20 µm`, `w_p = 0.02`):

| array | `np.array_equal` |
|---|---|
| `v2x_star`, `v2y_star`, `converged` | ✔ ✔ ✔ |
| `M`, `b`, `s1*`, `J`, `phi*`, `G0`, `detJ` | ✔ ✔ ✔ ✔ ✔ ✔ ✔ |
| `H_phi` | ✔ |
| `propagate_modal_asymptotic` field (2 source × 2 pupil modes) | ✔ |

**Measured**, interleaved medians of 7 (5 for the end-to-end), same process,
`OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1`:

| kernel | before | after | ratio |
|---|---|---|---|
| `_solve_envelope_stationary_batch` | 271.10 ms | 145.93 ms | **1.86×** |
| `_compute_M_b_batch` | 50.01 ms | 20.74 ms | **2.41×** |
| `propagate_modal_asymptotic` | 320.67 ms | 191.03 ms | **1.68×** |

(An earlier pair on a quieter box gave 225.99 → 126.94, 47.01 → 16.30 and
276.50 → 175.02 ms, i.e. 1.78× / 2.88× / 1.58×.  Both pairs are pre-and-post
runs taken back to back in the same box state; the OPERATION counts below are
the build-free statement of the same claim and did not move.)

Build-free companion (the number the b7 test pins): Chebyshev table builds per
call, **26 → 12** for `_compute_M_b_batch` and **144 → 50** for the Newton
(12 sweeps × 12 → 2 hoisted + 12 × 4).

The auditor's 2.3–5.1× is on the *evaluator*, not the kernel: the elementwise
basis work drops 3×, and what survives is the tensordots and the linear algebra
the fusion does not touch.

### 3.2 (c) the scale-relative Newton STOP — shipped opt-in

The *verdict* has been scale-relative since v5.30 (W6-A2); the *stop* was still
the absolute `rn < tol` with `tol = 1e-12` on a residual whose natural size is
O(1e7), so every pixel ran all 12 iterations.  `_NEWTON_SCALE_RELATIVE_STOP`
(module seam) and `scale_relative_stop=` (per call) opt the stop into the same
test.

**It moves the answer**, so it ships OFF: a pixel that leaves the active set
early keeps the converged iterate instead of the one twelve round-off steps
later.

| quantity | absolute stop | scale-relative stop |
|---|---|---|
| `_solve_envelope_stationary_batch` | 124.08 ms | 100.80 ms (**1.23×**) |
| Newton sweeps | 12 | 11 |
| max &#124;Δv2&#124; | — | 2.2e-11 (pupil half-range 0.0851, so 2.6e-10 of the box) |
| `propagate_modal_asymptotic` field | — | 9.1e-11 relative L2, max abs 1.9e-06 against a 1.1e+04 peak |

9.1e-11 is under the 3e-8 the `ModalAsymptoticStillBitEqual` arms now carry, so
turning it on by default would have slipped past them.  That is the reason it
is a seam and not a default: 1.23× on one kernel does not buy silently moving
every asymptotic answer.

---

## 4. Item 5 — `aberration_tensor`'s default cost

Fixture: the validation singlet (`validation/propagators/test_asymptotic.py`
`_build_test_singlet`, R1 = 51.5 mm, N-BK7, 12 mm aperture, object 200 mm,
λ = 1.31 µm, order 6, 8⁴ rays), `output_modes = [(0,0),(1,0),(2,0),(1,1),(0,3),(2,2)]`
— 6 requested, a 21-mode `(p_max, ell_max)` rectangle — and the adaptive
`sigma_grid_n`, which resolves to **256** here (its own Nyquist estimate asks
for more and the 256 cap truncates it).

**Where the time goes** (cProfile, before): of 10.29 s, `propagate_modal_asymptotic`
is 10.28 s — 8.51 s in `_solve_envelope_stationary_batch` and 1.73 s in
`_compute_M_b_batch`, i.e. **93 %** in the two kernels item 4 addresses.
`_measure_image_plane_waist` is 0.19 s (1.8 %) and the `decompose_lg` mode
stack the remainder.

**Changed.**

* `decompose_lg(..., only=...)` and `_lg_mode_conj_stack(..., only=...)` build
  and return only the requested `(p, ℓ)` pairs, in the same canonical order,
  with the set in the cache key.  `aberration_tensor` passes its
  `output_modes`.  Each overlap is a per-mode reduction against an
  independently built mode, so the numbers do not move — asserted exactly, and
  the stack's own byte count is asserted to shrink, which a value comparison
  cannot see.
* `_measure_image_plane_waist` is memoised on
  `(fit fingerprint, s2_image, source_point, pupil_amplitudes, w_s, w_p,
  v2_centre, n, propagate)`.  The audit's proposed key
  `(fit, s2_image, w_s, w_p, v2_centre)` is **incomplete**: `source_point` and
  `pupil_amplitudes` both reach the probe's `propagate` call and both change
  the width it measures, so the shipped key carries them.  The fit enters by a
  content fingerprint (raw bytes of its coefficient vectors and normalisation),
  not by `id()`, which CPython reuses after collection.  Bounded at 64 entries,
  FIFO, drained by `clear_image_plane_waist_cache()` and registered with the
  central cache registry.

**Measured**, same fixture, cold caches, median of 5:

| | before | after |
|---|---|---|
| `aberration_tensor` (cold) | 12 202.8 ms | **7 439.0 ms** (1.64×) |
| `aberration_tensor` (warm) | 11 675.9 ms | **7 346.2 ms** (1.59×) |
| `L`, `w_o`, `sigma_grid_n` | — | **bit-identical** (max &#124;ΔL&#124; = 0.0) |

(An earlier pair on a quieter box: 10 963.6 → 6 570.8 ms cold, 10 578.4 →
6 308.9 ms warm — 1.67× and 1.68×.)

**`sigma_grid_n` cap: left at 256, with the measurement.**  The audit asked to
revisit it "once (4) makes 512 affordable".  It does not: the σ grid enters as
`n²` pixels through `propagate_modal_asymptotic`, so 512 is 4× the post-item-4
7.44 s ≈ **30 s** per call — three decades above what a merit inside an
optimiser loop can pay, and 2.4× what the *pre-item-4* default cost.  Raising
the cap is a separate decision that wants the σ-grid path itself to get
cheaper first; the warning that names the required `n` already tells a caller
what to pass.

---

## 5. Item 6 — Y5 structural: what was collapsed, and why the rest stops here

**Collapsed:** `_compute_M_b_batch` and `_phi_v2_hessian_batch` now share one
basis evaluation, and `_solve_envelope_stationary_batch` shares the same
`_basis_and_grad34` primitive.  That removes one of the three copies of the
"build the Chebyshev tables, contract, chain-rule" kernel inside the batched
path, byte-identically (§3.1).

**Not collapsed**, and here is the harness result that says why.  Scalar
`_compute_M_b` (`asymptotic_aberration_tensor.py`) against `_compute_M_b_batch`
at five chart points on the same fit, same saddle:

| return | bit-equal | worst relative |
|---|---|---|
| `s1*`, `J`, `phi*`, `G0`, `detJ` | **5 / 5** | 0 |
| `M` | 2 / 5 | 1.1e-17 |
| `b` | 3 / 5 | 7.5e-18 |

and scalar `solve_envelope_stationary` against `_solve_envelope_stationary_batch`:
bit-equal **0 / 5**, `|Δv2|` up to **4.8e-15** on a 0.0851 pupil half-range
(5.6e-14 relative).

So the three copies are *not* the same arithmetic today.  The differences are
identified, not guessed:

1. `M`: scalar builds `J.T @ J` (a 2×2 `dgemm`), batched `np.matmul` over an
   `(N, 2, 2)` stack — different BLAS entries, different accumulation.
2. `b`: scalar `J.T @ r_star`, batched `np.einsum('nij,nj->ni', …)`.
3. `G0`: scalar `math.exp`, batched `np.exp` — libm against the NumPy loop,
   which are entitled to differ by an ULP (they happen not to on these five
   points).
4. The two Newtons take their steps in a different order and drop out of the
   active set on different tests.

A bit-identity harness across the three call sites therefore **cannot be made
green as they stand**, which is the condition the WP file set.  The plan, for
whoever picks it up:

* Make the batched kernel the single implementation and give the scalar entry a
  `(1,)`-shaped adapter.  That moves `aberration_tensor`'s answer by the
  amounts in the table above (≤ 1.1e-17 relative on `M`, ≤ 4.8e-15 absolute on
  `v2`), which is a *derived-tolerance* change, not a byte-identical one — so
  it needs its own envelope and its own fail-before, and it will move
  `AberrationTensorResult.L` in the last bits.
* The JAX twin `_compute_M_b_xp` cannot join: it contracts through the
  monkey-patched `eval_phi_xp` / `eval_s1_xp`, a functional xp-generic
  evaluator with its own accumulation, and it must stay traceable.  Two
  implementations (NumPy-batched and xp-generic) is the honest floor, not one.
* `eval_phi_xp` / `eval_s1_xp` should move onto the dataclass instead of being
  monkey-patched at import time (WP-A4 §6 item 6's other half); that part is
  behaviour-free and can land first.

Effort with the harness: still the 2–3 days WP-A4 estimated, now with the
tolerance known in advance.

---

## 6. Item 7 — §15.9, the uniform asymptotics

### 6.1 The premise is out of date

> "`uniform_fold_airy` and `pearcey` (`lenses_maslov.py:986-1084`) remain DEAD
> CODE" — WP-A4 §6 item 7.

They are not.  `lumenairy/elements/_lens_traced_uniform.py` imports
`_fold_airy_eval` (the closing CFU expression of `uniform_fold_airy`) and
`pearcey` at its line 69 and uses them for `apply_real_lens_traced`'s
`caustic='uniform'` completion — the fold-Airy dark-side continuation at
`:946-982` and the Pearcey cusp basis at `:451-460` — and both are covered by
`tests/unit/test_niche_k4_uniform_caustic.py` and
`tests/unit/test_niche_r2_pearcey_cusp.py`.  What is genuinely unwired is a
uniform path for the **Maslov `v2` integral**, which is a different thing from
"the CFU kernel is dead".

### 6.2 Probe 3, run

The oracle the auditor specified and nobody ran.  f/1.92 N-BK7 singlet
(R = ±2.05 mm, t = 0.55 mm, 1.00 mm aperture, λ = 1.0 µm, w = 0.30 mm):
paraxial focus 1924.17 µm past the exit vertex, **marginal focus 1740.80 µm**,
caustic span 183.37 µm — a genuine fold, and the readout plane is the marginal
focus.  Oracle: the same brute-force RS construction as §2.1, converged to
`1 − fid = 1.3e-07` (241 → 321 rays) and **4.6e-08** (321 → 401).

| member | fidelity vs the oracle | wall clock |
|---|---|---|
| `apply_real_lens_maslov('quadrature')`, order 6 / 8 | **0.6278 / 0.6267** | 39.9 / 44.2 s |
| `apply_real_lens_maslov('stationary_phase')`, order 6 / 8 | **0.5921 / 0.5958** | 0.8 / 2.6 s |
| `apply_real_lens_maslov('local_quadrature')`, order 6 / 8 | 0.5852 / 0.5812 | 0.3 / 2.7 s |
| `apply_real_lens_traced(caustic='wave')` | 0.1110 | 0.9 s |
| `apply_real_lens_traced(caustic='uniform')` | **0.0030** | 1.4 s |
| `apply_real_lens_traced(caustic='multibranch')` | 0.0018 | 1.3 s |

Three readings, and they settle the item:

1. **The saddle is not the bottleneck at this fold.**  `stationary_phase`
   (0.5921) is within **0.036** of the exact pointwise integrator on the same
   chart (0.6278).  A uniform Maslov path can recover at most that 0.036.
2. **The chart is.**  Both sit at ~0.63 against the oracle, and raising the
   chart from order 6 / 20⁴ rays to order 8 / 28⁴ moves `quadrature` the wrong
   way (0.6278 → 0.6267).  The missing 0.37 is the canonical-chart model, not
   the integration method, so wiring a Pearcey/Airy evaluator behind
   `integration_method='uniform'` cannot reach the oracle.
3. **The uniform machinery that IS wired loses here**, by two decades
   (0.0030 against 0.5921).

Per the WP file — "if the uniform path does not beat `stationary_phase` at the
fold against that oracle, leave it dead and publish the table" — I have left
`integration_method='uniform'` unwired and published the table.  The `caustic='uniform'`
result is a finding of its own and is escalated in §8.4: it is not mine.

---

## 7. Items 8–11

### 7.1 S9 — the GBD FFT kernel is clipped to its own support

`_reconstruct_fft` built its Gaussian kernel over the FULL `(2Ny−1, 2Nx−1)`
linear-convolution offset range whatever the beamlet's actual decay, so
`_fftconv_same` transformed `(3Ny−2, 3Nx−2)` — nine output grids per array,
several alive.  It now clips to `±ceil(R_cut/d)` per axis with
`R_cut = n_sigma/sqrt(alpha)` and `alpha = −½ k Im(Q)` (the same
`½ k λ_min` `_reconstruct_windowed` computes, per axis here because the
applicability gate has already refused a skew `Q`), and `_fftconv_same` pads to
`scipy.fft.next_fast_len`.

`_FFT_KERNEL_N_SIGMA = 6.5`, not the windowed path's 5.0: there the margin is
paid per beamlet in box AREA, here ONE kernel serves the bundle, so the margin
is set where the truncation vanishes into round-off — `exp(−6.5²) = 4.5e-19`,
two decades below float64 eps relative to the kernel peak.

**Measured** (`tracemalloc` peak / output-grid bytes, median of 3 for the wall
clock; beamlets decomposed at `waist_factor=1`, so `z` selects the regime):

| N | z | kernel half-width | peak before | peak after | ×grid before → after | ms before → after |
|---|---|---|---|---|---|---|
| 512 | 0.05 mm | 10 / 511 | 158.26 MB | **25.76 MB** | 37.73 → **6.14** | 594.5 → **47.3** |
| 512 | 0.20 mm | 27 / 511 | 158.26 | 27.33 | 37.73 → 6.52 | 587.7 → 39.4 |
| 512 | 0.50 mm | 65 / 511 | 158.26 | 32.51 | 37.73 → 7.75 | 633.6 → 55.2 |
| 512 | 2.00 mm | 259 / 511 | 158.26 | 73.35 | 37.73 → 17.49 | 504.2 → 133.8 |
| 512 | 8.00 mm | 511 / 511 | 158.26 | 158.55 | 37.73 → 37.80 | 491.2 → **338.3** |
| 256 | 0.05 mm | 10 / 255 | 39.48 | 6.73 | 37.65 → 6.42 | 140.1 → 9.1 |
| 128 | 0.05 mm | 10 / 127 | 9.83 | 1.84 | 37.50 → 7.00 | 30.6 → 2.1 |

The auditor's "36× the output-grid bytes, scaling as N² with no cap" is
reproduced exactly (37.5–37.7× at every N before).  The last row is the honest
other half: a beamlet that has spread to fill the grid keeps the full kernel
and pays what it always did in memory (+0.2 %), while `next_fast_len` alone
still buys **1.45×** on the transform.  The b7 test asserts BOTH directions, so
a clip that truncated a live beam would fail it.

**Accuracy** (derived tolerance, not byte-identical — the clip drops a tail and
the transform length changes):

* clipped regime, against the windowed scatter-add at n_sigma 5 / 7 / 9
  (whose own truncation at 9 is `exp(−81) = 7e-36`): **8.2e-16 … 2.3e-15**
  relative L2 at N = 96 and 128, z = 0.05 / 0.20 / 0.50 mm.  The reading does
  not move with n_sigma, which is the proof that what is left is the
  transform's round-off and not the clip.
* unclipped regime, against the pre-change field: **5.6e-16 … 8.3e-16**
  (`next_fast_len` alone).
* for scale, the pre-change path agreed with the windowed sum at n_sigma = 8
  to 4.3e-15 / 5.6e-15 / 1.5e-14 at N = 128 / 256 / 512.

### 7.2 Item 9 — the JAX sibling's chief-ray displacement

WP-B1's §6 item 6 and §7 item 5 were struck by VERIFY-B1: `apply_real_lens_maslov_jax`
has no stationary-point solve, so S6 cannot apply to it.  Its real defect,
which VERIFY-B1 measured and I re-derived on my own fixture: the screen's OPL
is indexed by the ray's **entrance** point `(xe, ye)` while `E_in` is sampled at
the **output pixel** — two different points, because a ray walks across the
element.  For a collimated input the input phase is constant along that walk;
for a tilted one it is not, and the difference is a first-order term
`k0 · k1 · (xe − x)` with `k1 = (1/k0) grad arg E_in`.

That term is added.  It is **exactly zero** for a real non-negative `E_in`
(the conjugate-product difference of a constant phase is exactly 0), so a
collimated input is byte-identical.

**Measured** on my own fixture (f = 14.10 mm N-SF11 biconvex, λ = 1.55 µm,
1.00 mm aperture → NA 0.0355, N = 384, dx = 4 µm, w = 0.25 mm; screen, then
`angular_spectrum_propagate` to a plane 0.30 mm past the focus; intensity
centroid against an exact conic raytrace of the input's own ray fan):

| input | oracle landing | screen (error) | corrected (error) |
|---|---|---|---|
| collimated | 0.000 µm | −0.000 (0.00 %) | −0.000 (**0.00 %**, bit-identical) |
| tilt 0.25 × NA | 120.243 µm | 117.041 (**−2.66 %**) | 120.013 (**−0.19 %**) |
| tilt 0.50 × NA | 240.511 µm | 234.109 (**−2.66 %**) | 240.056 (**−0.19 %**) |
| tilt 1.00 × NA | 481.227 µm | 468.143 (**−2.72 %**) | 479.947 (**−0.27 %**) |
| off-axis converging | 5.828 µm | 6.601 (+13.28 %) | 5.360 (−8.03 %) |

The error is proportional to the tilt (−2.66 / −2.66 / −2.72 %), which is the
signature of a linear walk term and refines VERIFY-B1's "2.6 % at half the lens
NA and 3.7 % at the lens NA" to a constant fraction on this optic.  Corrected,
it falls 14×; the residual −0.2 % is second order (the OPL map is still the
θ = 0 ray family's).  The off-axis converging row improves in absolute terms
(0.773 → 0.468 µm) on a 5.8 µm displacement.

`input_wavevector_saddle=` carries the same three values and the same meaning
as `apply_real_lens_maslov`'s keyword, so a caller can switch backends without
changing which ray the answer is built on.  The docstring says plainly that
this path has no saddle and that what the keyword selects here is the
displacement term.

### 7.3 Item 10 — the S6 fallback statistic, re-founded

`_K1_FIT_RESIDUAL_MAX` scores the `k1` fit's VALUE while the saddle's Newton
consumes its two DERIVATIVES.  The new statistic
(`_k1_fit_derivative_error`, `lenses_maslov.py`) is VERIFY-B1's F1 candidate:
refit `k1` at `poly_order − 1` — whose basis is literally the column subset of
`A` with total degree below the cap — and compare the two charts'
`(dk1/du3, dk1/du4)` at the ray points, intensity-weighted, normalised by the
RMS of `k1` itself so a uniform tilt scores 0 rather than 0/0.  It is
accumulated term by term rather than through two more `(n_rays, M)` derivative
design matrices, which on a 16⁴-ray order-6 chart would be 110 MB each.

`_K1_FIT_RESIDUAL_MAX = 0.5` is **unchanged** — the statistic it scores has not
changed, so `test_verify_b1_the_k1_fit_residual_is_the_statistic_it_claims_to_be`
is not restated — and the new bar is an additional gate.

**MEASURED** on two charts (the f = 6 mm N-BK7 / 1.0 µm one and an f = 14.10 mm
N-SF11 / 1.55 µm one), fidelity against the exact pointwise `'quadrature'` on
the same chart, OPD-only saddle → engaged, `stationary_phase`:

| input | value residual (A / B) | **slope error (A / B)** | fidelity OPD-only → engaged (A / B) | engaging |
|---|---|---|---|---|
| tilt 0.5…4 × lens NA | 1e-11 … 1e-14 | **1.2e-10 … 2.8e-08** | 0.000 → 0.864–0.922 / 0.703–0.737 | wins |
| converging f = +40 mm | 1.07e-05 / 4.27e-06 | 4.6e-06 / 2.0e-06 | 0.544 → 0.997 / 0.122 → 0.999 | wins |
| diverging f = −25 mm | 1.27e-05 / 6.29e-06 | 5.6e-06 / 3.2e-06 | 0.239 → 0.974 / 0.060 → 1.000 | wins |
| speckle 0.002 rad rms | 2.4e-03 / 3.6e-03 | 4.8e-03 / 9.0e-03 | 0.000 → 0.912 / 0.707 | wins |
| speckle 0.010 | 1.2e-02 / 1.8e-02 | 2.4e-02 / 4.5e-02 | 0.000 → 0.865 / 0.428 | wins |
| speckle 0.050 | 5.8e-02 / 8.8e-02 | 1.3e-01 / 2.5e-01 | 0.000 → 0.315 / 0.114 | wins |
| **speckle 0.100** | 1.1e-01 / 1.7e-01 | **2.8e-01 / 5.6e-01** | 0.000 → 0.142 / 0.011 | **wins (last)** |
| **hard edge at 0.80** | 7.9e-02 / 8.2e-02 | **2.8e+00 / 2.7e+00** | 0.016 → **0.000** / 0.075 → **0.000** | **LOSES (first)** |
| hard edge at 0.95 | 1.5e-01 / 1.5e-01 | 4.3e+00 / 4.2e+00 | 0.021 → 0.000 / 0.036 → 0.000 | LOSES |
| hard edge at 0.60 | 2.6e-01 / 2.6e-01 | 3.9e+00 / 3.9e+00 | 0.038 → 0.000 / 0.265 → 0.000 | LOSES |
| speckle 0.600 | 5.6e-01 / 7.0e-01 | 3.2e+00 / 4.4e+00 | refused by the value bar already | — |

The hard-edged family is what the new bar is for, and it is exactly the family
the value residual cannot see: its value residual (0.08–0.26) sits comfortably
inside the 0.5 bar while its SLOPE error is two decades worse than any speckle
row — because `_local_direction_cosines` reports 0 in the dark and the true
wavefront in the light, and a degree-4 fit of that step has an unbounded
derivative wherever the step is.

**The bar.**  `_K1_DERIV_RESIDUAL_MAX = 1.2`, the geometric mean of the
two-chart bracket **5.6e-01** (last input where engaging still wins) …
**2.7e+00** (first input where it loses) = 1.24.  It sits 4.3× above the other
chart's last win and 2.4× below its first loss.  **Every case where engaging
wins today still engages**; what changes is that the three hard-edged rows are
refused, the warning fires naming the mechanism and `integration_method='quadrature'`,
and the caller gets 0.02–0.27 instead of 0.000.

Cost: one more `_solve_fit` against a narrower column subset of the SAME `A`,
plus two term-by-term accumulations over the traced rays, on the ENGAGED path
only.  On the f = 6 mm chart at 16⁴ rays / order 4 the whole `stationary_phase`
call goes 0.06 → 0.10 s; nothing that does not engage pays anything.

### 7.4 Item 11 — the pupil chart is sized from the mean AND the spread

`na_input` was `3 × sqrt(<v²>)`, the second angular moment ABOUT ZERO, so a
uniform tilt θ contributed 3θ.  A tilt is a change of reference direction, not
an angular spread; the chart is a box about `v = 0`, so what it must reach is
`|mean launch direction| + 3 σ_about_mean`.

Gated by `_NA_MEAN_MIN_FRACTION = 0.1` on `mean / σ_about_zero`, below which
the old arithmetic is used verbatim.  That gate is what buys byte-identity, not
the algebra: the two forms are different float64s even when the mean is 1.7e-14
(measured 4.501578547758776e-03 against 4.5015785477421685e-03).

**Derivation of the bar**, measured on the f = 6 mm chart as
`mean / σ_about_zero`:

| field | ratio |
|---|---|
| centred Gaussian, w = 0.15 / 0.40 mm | 1.1e-11 / 9.4e-08 |
| the same, displaced half a pixel / one pixel | 3.2e-11 / 7.1e-11 |
| converging f = +40 / diverging f = −25 mm | 2.3e-11 / 3.4e-11 |
| hard aperture at 0.95 / 0.80 / 0.60 | 2.0e-05 / 9.3e-05 / 3.1e-04 |
| speckle 0.05 / 0.30 rad rms, no tilt | 4.1e-04 / 1.2e-03 |
| **uniform white-noise phase, no tilt** | **1.5e-02** |
| **uniform tilt 1e-3 rad (`_SADDLE_FLAT_INPUT_NA`)** | **5.5e-01** |
| 5 waves of coma | 6.2e-01 |
| uniform tilt half the lens NA | 9.98e-01 |

The floor is not exact symmetry — `fftfreq`'s Nyquist column has no partner to
cancel against, a pixel-quantised mask is not centred on an even grid, a finite
noise realisation has a finite mean — so the bar is the geometric mean of the
bracket 1.5e-02 … 5.5e-01 = 9.2e-02 → **0.1**.  What it discards is bounded: at
the bar itself `3√(m²+s²)` and `m + 3s` differ by 2.3 % of a quantity that is
already a 3-sigma margin.

**Measured effect**, both charts, tilt sweep, `stationary_phase` /
`local_quadrature` fidelity against the exact `'quadrature'` on the same chart:

| chart, tilt | `na_input` before → after | s1 fit residual before → after | fidelity before → after |
|---|---|---|---|
| A, 0.5 × NA | 0.0748 → 0.0294 | 5.4e-05 → 1.3e-05 | 0.911/0.984 → 0.911/0.984 |
| A, 1.5 × NA | 0.2239 → 0.0791 | 1.29e-03 → 6.1e-05 | 0.912/0.985 → 0.922/0.984 |
| **A, 2.0 × NA** | 0.2986 → **0.1040** | 3.98e-03 → **1.17e-04** | **0.000/0.000 → 0.911/0.979** |
| **A, 4.0 × NA** | 0.3405 → **0.1180** | 6.90e-03 → **1.63e-04** | **0.000/0.000 → 0.864/0.959** |
| B, 2.0 × NA | 0.2129 → 0.0751 | 9.62e-04 → 3.2e-05 | 0.726/0.945 → 0.737/0.948 |
| **B, 4.0 × NA** | 0.4257 → **0.1461** | 1.97e-02 → **2.34e-04** | **0.000/0.000 → 0.703/0.858** |

The three rows that read 0.000 before are the V1 collapse VERIFY-B1 found, and
this fixes it **at its source**: the chart the order-4 fit has to span shrinks
by 2.8–2.9×, its residual falls by one to two decades, and the
`_S1_FIT_RESIDUAL_MAX` gate VERIFY-B1 added stops firing on a plain tilt
because the chart it was guarding against is no longer built.  That gate is
unchanged and still live — §8.1 shows it firing on an explicitly over-sized
chart.

**Runtime.**  VERIFY-B1's F2 also expected a runtime saving on every tilted
call.  I could not measure one above the noise on these fixtures: the ray count
and the ROI do not depend on `na_proxy`, only the traced cone does, and the
whole `stationary_phase` call is 0.05–0.12 s either way.  The saving F2
anticipated is real only where a wider `na_proxy` pushes `poly_order='auto'` up
a rung, which it did not here.  Reported rather than claimed.

---

## 8. Requested changes outside my ownership

### 8.1 `tests/unit/test_audit2609_b1_maslov_input_wavevector.py` — BLOCKING

`test_verify_b1_the_gate_refuses_a_chart_that_cannot_carry_ds1_dv2`
(× 2 methods) constructs its "chart that cannot carry `ds1/dv2`" by relying on
`na_proxy` tripling a uniform tilt.  Item 11 removes that tripling, so at tilt
2 × the lens NA the s1 fit residual is now **1.17e-04** instead of 3.98e-03 and
the gate correctly does not fire.  The gate itself is unchanged and still
correct; its fixture has to ENGINEER the over-sized chart rather than hope the
driver produces it (TESTING_STANDARDS rule 3).

Exact patch (verified: **33 / 33 green** with it applied, in a scratch copy):

```diff
     good = _field(tilt_x=1.0 * _NA_LENS)
     bad = _field(tilt_x=2.0 * _NA_LENS)
+    # The driver sizes the pupil chart from the MEAN launch direction plus the
+    # SPREAD, so a uniform tilt no longer inflates the box threefold and the
+    # order-4 chart carries it (WP-B7 item 11).  The over-sized chart this gate
+    # exists for is therefore ENGINEERED here rather than hoped for
+    # (TESTING_STANDARDS rule 3): ``_BAD_NA`` is the 3-sigma-about-zero
+    # angular moment of this very field, which is what sized it before.
+    _BAD_NA = 3.0 * 2.0 * _NA_LENS
     _, _, s1_good, k1_good, eng_good, msg_good = _s6_report(
         good, method, centre=_window_on(good))
     _, _, s1_bad, k1_bad, eng_bad, msg_bad = _s6_report(
-        bad, method, centre=_window_on(bad))
+        bad, method, centre=_window_on(bad), input_na=_BAD_NA)
@@
     _, _, _, _, eng_forced, m_forced = _s6_report(
-        bad, method, centre=_window_on(bad), input_wavevector_saddle=True)
+        bad, method, centre=_window_on(bad), input_na=_BAD_NA,
+        input_wavevector_saddle=True)
```

The docstring's tilt table should gain one line saying the s1 residuals quoted
are the ones an explicitly over-sized `input_na` produces, since the driver no
longer reaches them from a tilt alone.

### 8.2 `tests/unit/test_fga_h4_h5.py` + `lumenairy/propagators/fga.py` — the WP-B9 request, second half

WP-B9 gave `ray_transfer_jacobian_analytic` even-aspheric support.
`gbd.py`'s `jacobian='auto'` picks it up for free, because it dispatches on the
primitive's own `NotImplementedError` — pinned in b7 §7.  `fga.py` does not:
`_pick_ray_transfer` gates on `_is_all_conic`, a whitelist that still excludes
`aspheric_coeffs`, so an aspheric prescription traces the 9-ray FD bundle there
whatever `exact_jacobian` says — including `exact_jacobian=True`, which is
silently ignored.  I corrected the two comments to state that fact (they
claimed the analytic form "does not handle" aspherics, which is no longer
true) and did **not** widen the whitelist, because three assertions in a file I
do not own pin the current behaviour.

`_is_all_conic` also does not check `field_decenter` / `field_tilt` /
`field_sag_callable`, which the analytic primitive DOES reject — so a
field-decentred conic surface reaches the analytic path in FGA and raises
`NotImplementedError` at call time instead of falling back.  That is a latent
bug in the same predicate.

Exact edits:

```diff
--- a/lumenairy/propagators/fga.py
+++ b/lumenairy/propagators/fga.py
-def _is_all_conic(surfaces):
+def _analytic_jacobian_applies(surfaces):
     for s in surfaces:
-        if (getattr(s, 'aspheric_coeffs', None) or getattr(s, 'freeform', None)
+        _ff = (getattr(s, 'field_sag_callable', None) is not None
+               or (getattr(s, 'field_decenter', None) is not None
+                   and tuple(float(v) for v in s.field_decenter) != (0.0, 0.0))
+               or (getattr(s, 'field_tilt', None) is not None
+                   and tuple(float(v) for v in s.field_tilt) != (0.0, 0.0)))
+        if (getattr(s, 'freeform', None)
                 or getattr(s, 'radius_y', None) is not None
                 or getattr(s, 'conic_y', None) is not None
-                or getattr(s, 'aspheric_coeffs_y', None) is not None):
+                or getattr(s, 'aspheric_coeffs_y', None) is not None
+                or _ff):
             return False
     return True
+
+
+_is_all_conic = _analytic_jacobian_applies   # back-compat alias
```

```diff
--- a/tests/unit/test_fga_h4_h5.py
+++ b/tests/unit/test_fga_h4_h5.py
-    # an aspheric surface is NOT all-conic -> analytic unavailable -> FD, even at
-    # the True/None default (the analytic form does not handle aspheres).
+    # WP-B9 gave the analytic primitive even-aspheric support, so an aspheric
+    # surface now REACHES it; a biconic still does not.
     class _Stub:
         aspheric_coeffs = [1e-3]
-    assert not fga._is_all_conic([_Stub()])
-    assert fga._pick_ray_transfer([_Stub()], None) is ray_transfer_jacobian
-    assert fga._pick_ray_transfer([_Stub()], True) is ray_transfer_jacobian
+    assert fga._analytic_jacobian_applies([_Stub()])
+    assert (fga._pick_ray_transfer([_Stub()], None)
+            is ray_transfer_jacobian_analytic)
+
+    class _Biconic:
+        radius_y = 25e-3
+    assert not fga._analytic_jacobian_applies([_Biconic()])
+    assert fga._pick_ray_transfer([_Biconic()], True) is ray_transfer_jacobian
```

This is a **default move** for aspheric prescriptions in FGA (auto switches
from the FD Jacobian to the exact analytic one).  The measured case: on an A4
singlet the two primitives' exit heights agree to better than 1e-6 relative,
which is the FD central-difference truncation floor, and the analytic side is
exact — so the swap raises accuracy and drops the trace count 9N → N.  A
Migration note is drafted in the changelog against this request.

### 8.3 `_universal_route`'s caustic branch — the item 3 consequence

`_universal_route` is in `fga.py` (mine), but the change reds
`tests/unit/test_audit2609_a4_fga_s10.py` (lines 287, 291, 301, 302, 316, 317,
319) and pins in `test_fga.py`, `test_g1_gate_generality.py`,
`test_niche_audit_w9_dispatch2.py`, `test_niche_p7_seidel_gate.py`,
`test_niche_p8_capstone.py` — none of which I own — so I have not made it.

What the §2 table supports:

```diff
     if mv:
         return "fga"
     zone = _caustic_zone(E_in, dx, prescription, wavelength)
     if zone is not None:
         pad = caustic_pad_dof * float(wavelength) / (na * na)
         if (zone[0] - pad) <= opd <= (zone[1] + pad):
-            return "fga"
+            # Measured against a converged brute-force Rayleigh-Sommerfeld
+            # oracle on the f = 1.2 mm NA 0.145 singlet at its focus (WP-B7
+            # item 3): fidelity 0.3234 for 'fga' at its defaults and 0.3826 at
+            # the best of fifteen sampling settings, against 0.9965 for
+            # 'phase_screen', and 'phase_screen' is the closer member at every
+            # NA from 0.039 to 0.192.  What 'fga' uniquely provides at a
+            # caustic is the MULTI-VALUED field, which the branch above has
+            # already routed; for a single-valued one the thin screen plus the
+            # exact angular spectrum is both wave-exact in propagation and an
+            # order of magnitude closer here.
+            return "phase_screen"
```

The five test files then need their `== 'fga'` expectations at a caustic
restated to `== 'phase_screen'` for the single-valued rows, with the
measurement above in the docstring, and the multi-valued rows left alone.
This is the largest single behaviour change the audit's §6 item 3 implies and
it should be one reviewed commit, not a side effect of this package.

### 8.4 `lumenairy/elements/_lens_traced_uniform.py` — `caustic='uniform'` at a real fold

§6.2: at the marginal focus of an f/1.92 singlet,
`apply_real_lens_traced(caustic='uniform', amplitude_model='ray_density')`
scores **0.0030** against a brute-force RS oracle converged to 4.6e-08, and
`caustic='multibranch'` 0.0018, while `caustic='wave'` scores 0.1110 and the
asymptotic Maslov evaluator 0.5921.  The CFU kernel itself is validated to
1e-14 against exact cubic-phase integrals, so what fails is the FITTING of the
control parameters to the traced branches, not the special function.  That
file belongs to the `_lens_traced` owner; the oracle and the fixture are in
this package's scratch and can be handed over.

### 8.5 `WP-A4_REPORT.md` §6 item 7 / audit §15.9 — factual correction

"`uniform_fold_airy` and `pearcey` remain DEAD CODE" is not true:
`_lens_traced_uniform.py:69` imports `_fold_airy_eval` and `pearcey` and uses
them for `apply_real_lens_traced(caustic='uniform')`, covered by
`test_niche_k4_uniform_caustic.py` and `test_niche_r2_pearcey_cusp.py`.  What
is unwired is a uniform path for the Maslov `v2` integral.

---

## 9. Files touched

| file | what |
|---|---|
| `lumenairy/propagators/asymptotic_canonical_fit.py` | `_basis_and_grad34`; `CanonicalPolyFit.basis_index_columns`; `CanonicalPolyFit.eval_s1_and_phi_with_v2_grad` |
| `lumenairy/propagators/asymptotic_maslov.py` | `_NEWTON_SCALE_RELATIVE_STOP`; `_phi_v2_hessian_batch(T12_rows=)`; `_compute_M_b_batch` fused; `_solve_envelope_stationary_batch` hoisted factor + `scale_relative_stop=` |
| `lumenairy/propagators/asymptotic_modes.py` | `decompose_lg(only=)` / `_lg_mode_conj_stack(only=)` |
| `lumenairy/propagators/asymptotic_aberration_tensor.py` | `_fit_fingerprint`; `_W_O_CACHE` + `clear_image_plane_waist_cache` + registry entry; `_measure_image_plane_waist` memoised; `aberration_tensor` passes `only=` |
| `lumenairy/propagators/gbd.py` | `_FFT_KERNEL_N_SIGMA`, `_kernel_half_width`, `_fft_len`; `_fftconv_same` pads to `next_fast_len`; `_reconstruct_fft` clips the kernel |
| `lumenairy/propagators/fga.py` | `_is_all_conic` / `_pick_ray_transfer` / `apply_real_lens_fga` docstrings corrected (WP-B9 request) |
| `lumenairy/elements/lenses_maslov.py` | `_K1_DERIV_RESIDUAL_MAX` + `_k1_fit_derivative_error` + the gate and its warning branch; `_NA_MEAN_MIN_FRACTION` + the mean-plus-spread sizing; docstring |
| `lumenairy/elements/_lens_jax.py` | `_local_direction_cosines_jax`; `apply_real_lens_maslov_jax(input_wavevector_saddle=)` + the chief-ray displacement term |
| `docs/history/lumenairy.elements.lenses_maslov.md`, `…_lens_jax.md`, `…propagators.asymptotic_aberration_tensor.md`, `…asymptotic_canonical_fit.md`, `…asymptotic_maslov.md`, `…asymptotic_modes.md`, `…propagators.gbd.md` | re-recorded in this change (`scripts/record_history_fingerprints.py --check` exit 0 on all seven) |
| `tests/unit/test_audit2609_b7_asymptotic.py` | new, 20 ids |
| `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-B7_REPORT.md`, `WP-B7_CHANGELOG.md` | this report and the changelog |

`lumenairy/elements/lenses_gbd.py` is in my ownership and was **not** touched —
nothing in items 3–11 reaches it.  `lumenairy/propagators/fga.py` has no
`docs/history/` document, so there is nothing to re-record for it.

## 10. Tests run

All single-threaded (`OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1`),
one process at a time, 2026-09-13/14.

| command | result |
|---|---|
| `pytest tests/unit/test_audit2609_b7_asymptotic.py` | **20 passed** (23.5 s) |
| `pytest tests/unit/test_audit2609_b1_maslov_input_wavevector.py` | 31 passed, **2 failed** (41.1 s) — §8.1, and **33 passed** with the §8.1 patch applied in a scratch copy |
| `pytest tests/unit/test_audit2609_a4_maslov_gbd.py test_audit2609_a4_verify_maslov_asymptotic.py test_audit2609_a4_asymptotic.py test_audit2609_a4_fga_s10.py` | **111 passed** (92.0 s) |
| `pytest tests/unit/test_audit_propagation.py -k ModalAsymptoticStillBitEqual` | **2 passed**, 168 deselected (8.0 s) — the 3e-8 arms VERIFY-B9 restated, untouched |
| `pytest tests/unit/test_niche_audit_w6_asymptotic.py test_perf_v4_12_0_asymptotic.py` | **69 passed** (153.9 s) |
| `pytest tests/unit/test_audit2609_a17_history_lint.py test_audit2609_a15a_lens_covering_array.py test_v5_4_7_walker_v20_cross_backend_parity.py` | **66 passed** (24.3 s) |
| `pytest tests/unit/test_audit2609_a17_history_lint.py test_audit2609_a17_history_relocation.py` | **744 passed** (44.0 s) |
| `pytest tests/unit -k "maslov or asymptotic or gbd or fga"` | **785 passed, 2 failed, 8 skipped**, 15318 deselected (49 min) — the two failures are the §8.1 b1 ids and nothing else; no WP-B11a id in this selection was red |
| `python validation/run_all.py test_lenses test_propagation` | **ALL 2 files passed** (29.8 s / 8.0 s) |
| `ruff check lumenairy/propagators lumenairy/elements/lenses_maslov.py lumenairy/elements/lenses_gbd.py lumenairy/elements/_lens_jax.py tests/unit/test_audit2609_b7_asymptotic.py` | **All checks passed** |
| `scripts/record_history_fingerprints.py <module> --check` x 7 | **exit 0** on all seven |

### 10.1 Concurrency note

A second engineer (WP-B11a) is editing `_lens_real.py`, `_lens_traced.py`,
`lenses.py`, `lens_config.py`, `doe.py`, `glass.py`, `carrier.py`, `mft.py`,
`rcwa/*`, `eme/*`, `bor/*`, `pmm/*`, `analysis/*` and three new leaves in the
same working tree.  Failures in those partitions — including their history
fingerprints and `ruff` on `rcwa/_geometry.py` — are theirs and are not
reported here.  Every number above that claims a before/after was taken in a
child process rooted in an isolated tree (the `b2baa505` archive, or that
archive overlaid with only this package's eleven files), never in the shared
working tree.

## 11. Deferred

* **Y5 collapse** — §5: the plan, the measured divergence between the three
  copies, and the reason the bit-identity harness cannot be green today.
* **`sigma_grid_n` cap** — §4: 512 costs ~30 s per `aberration_tensor` even
  after item 4; the cap stays at 256 and the warning keeps naming the required
  `n`.
* **A uniform Maslov `v2` path** — §6.2: at a genuine fold the saddle costs
  0.036 of fidelity against the exact integrator on the same chart, and the
  chart costs 0.37 against the oracle.  The chart is where the next
  fold-accuracy work belongs, not the integrator.
* **The `'auto'` routing at a caustic** (§8.3) and the **FGA analytic-Jacobian
  whitelist** (§8.2) — both are measured and specified, both red test files
  outside this package.
