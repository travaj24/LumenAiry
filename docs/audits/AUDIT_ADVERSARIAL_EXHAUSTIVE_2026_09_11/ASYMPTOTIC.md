# ASYMPTOTIC audit — `lumenairy/propagators/asymptotic*.py`

## What this family actually is (established from the code, not the docstrings)

It is **not** an eikonal/WKB field propagator and **not** an aberration *tensor* in the
Taylor-expansion-of-the-eikonal sense. It is a **mixed-representation (Maslov) phase-space
diffraction integral evaluated by a Gaussian-moment (Wick) contraction**, driven by a 4-D
Chebyshev fit of ray-traced data:

1. `fit_canonical_polynomials` (`asymptotic_canonical_fit.py:228-536`) traces a
   Chebyshev-node grid in `(s1x, s1y, v1x, v1y)` through a sequential prescription and
   least-squares fits, over the *observed output* box `(s2x, s2y, v2x, v2y)` normalised to
   `[-1,1]^4`, two 4-variable total-degree Chebyshev tensor products:
   `Phi(s2, v2) = OPL/lambda` (waves) and the back-map `s1(s2, v2)` (metres).
   `Phi` is Hamilton's point characteristic `V(s1, s2)` re-parametrised on `(s2, v2)`.
   Optionally a 5-term linear ramp `a0 + a1 u1 + a2 u2 + a3 u3 + a4 u4` is pre-fitted and
   subtracted first (`extract_linear_phase`, default `True`).

2. The model integral (`asymptotic.py:634-763`, `asymptotic_maslov.py:211-282`) is

       E(s2) = INT d^2 v2  |det J|  A_src(s1(s2,v2) - s_src; w_s)  A_pup(v2 - v_c; w_p)
                            * exp(2 pi i Phi(s2, v2)),     J = d s1 / d v2 |_{s2}

   i.e. the exact Huygens-Fresnel integral over `s1` after the change of variable
   `s1 -> v2` at fixed `s2`, with an extra soft Gaussian pupil in *output direction cosine*
   and the source/pupil fields expanded in Cartesian-monomial LG bases.

3. It is evaluated by expanding the exponent to **second order in `eta = v2 - v*`** about the
   *envelope*-stationary point `v*` (`solve_envelope_stationary`, Gauss-Newton on
   `J^T(s1-s_src)/w_s^2 + (v2-v_c)/w_p^2 = 0`), giving `exp(-eta^T M eta + b.eta)` with

       M = J^T J / w_s^2 + I / w_p^2 - i pi H_Phi,     H_Phi = d^2 Phi / dv2^2
       b = 2 pi i g - 2 J^T r* / w_s^2 - 2 dv / w_p^2,  g = dPhi/dv2

   and the polynomial prefactor contracted in closed form against the 2-D complex-symmetric
   Gaussian moments `<eta_x^a eta_y^b>` (Wick), `Sigma = M^{-1}/2`. Leading amplitude
   `detJ * pi/sqrt(det M) * G0 * exp(2 pi i Phi*) * exp(b^T M^{-1} b / 4)`.

4. `aberration_tensor` is **an LG-mode overlap matrix `L[k_out, k_src]`**, not a tensor with
   symmetric index pairs. There is no Seidel/Zernike transformation anywhere;
   `lg_seidel_label` is a *naming table* mapping `(p, |l|)` to a Seidel name via the
   radial-order correspondence `n = 2p + |l|`, `m = l` (verified consistent with
   `Z_n^m`: (1,0)->defocus, (2,0)->spherical, (1,+-1)->coma, (0,+-2)->astigmatism,
   (0,+-3)->trefoil). Two branches: a **point-sampling** closed form for a pure `[(0,0)]`
   request, and a **sigma-grid numerical projection** (`propagate_modal_asymptotic` +
   `decompose_lg`) for everything else. The two are on different scales (documented).

5. `HFPolyFit` / `propagate_hf_chebyshev_quadrature` are the *sibling* direct-quadrature
   path: `Phi(s1, s2)` plus the Van Vleck factor `sqrt(|det d^2 Phi/ds1 ds2|)` and the `-i`
   d=2 Maslov phase — this one is a properly normalised Van Vleck-Morette propagator
   (verified below to 2.7e-11).

## Scope read (every line)

| file | lines | read |
|---|---|---|
| `lumenairy/propagators/asymptotic.py` | 768 | 1-768 (all) |
| `lumenairy/propagators/asymptotic_aberration_tensor.py` | 1319 | 1-1319 (all) |
| `lumenairy/propagators/asymptotic_canonical_fit.py` | 1275 | 1-1275 (all) |
| `lumenairy/propagators/asymptotic_jax_twin.py` | 1143 | 1-1143 (all) |
| `lumenairy/propagators/asymptotic_maslov.py` | 650 | 1-650 (all) |
| `lumenairy/propagators/asymptotic_modes.py` | 893 | 1-893 (all) |

Dependencies read to confirm findings: `lumenairy/_math/chebyshev.py:1-220`,
`lumenairy/elements/lenses.py:815-941`, `lumenairy/elements/lenses_maslov.py` (N4 fix sites
only: 815-830, 1762-1812, 2268, 2551, 2908-2975, 3509-3520),
`lumenairy/optimize/driver.py:150-195`, `lumenairy/optimize/jax_merits.py:425-478`,
`lumenairy/propagators/dispatch.py:1336-1350`, `lumenairy/propagators/subaperture.py:505-530`,
`tests/unit/test_niche_audit_w6_asymptotic.py:1120-1220` (+ full test-name listing),
`benchmarks/test_bench_asymptotic.py:1-90`,
`docs/audits/AUDIT_WAVE_LENS_MODELS_2026_07_02_REVIEW.md:110-130`.

**Not covered:** `validation/propagators/test_asymptotic.py` (read only via its citations in
the module docstrings); `tests/unit/test_perf_v4_12_0_asymptotic.py` (listing only);
the CuPy branch of `_chebyshev_vandermonde_xp` (cupy not installed; desk-checked only);
`fit_canonical_polynomials_jax` end-to-end against a vignetted prescription (jax 0.10.1
runs, but the `trace_jax` leg belongs to another partition).

Scratch scripts: `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/ASYMPTOTIC/t1..t19_*.py`.

---

## Findings

### [P0] `extract_linear_phase=True` (the DEFAULT) drops the v2-linear phase from *inside* the integral — PSF lands in the wrong place with the wrong amplitude
`lumenairy/propagators/asymptotic_canonical_fit.py:446-455` (the 5-term prefit) and
`:207-225` (`eval_phi_with_v2_grad`, `include_linear=False` by default), consumed at
`lumenairy/propagators/asymptotic_maslov.py:240-242`,
`lumenairy/propagators/asymptotic_aberration_tensor.py:203-206`,
`lumenairy/propagators/asymptotic_jax_twin.py:265`.

**What is wrong.** The prefit removes `a0 + a1·u1 + a2·u2 + a3·u3 + a4·u4` from `Phi`. The
`a0,a1,a2` terms depend only on `s2` and factor out of the `v2` integral — removing them is
a legitimate piston/tilt reference. **`a3·u3 + a4·u4` depend on `v2`, i.e. they are the
integration variable.** Dropping them changes `g = dPhi/dv2`, hence `b`, hence the complex
saddle shift `delta* = M^{-1}b/2` and `Re(b^T M^{-1} b/4)` — an **amplitude and position**
error, not a phase reference. The v5.30 W6-A4 note (`asymptotic.py:347-367`) asserts the
removal "is phase-only" on the strength of two measurements in which `a3 ≈ 0`.

**Evidence (measured).**

*Mechanism, fully controlled* (`t4_linear_extraction.py`): two `CanonicalPolyFit` objects
representing the **identical** total `Phi` (the Chebyshev basis has `T0=1, T1=u`, so the
split is exact), one with the linear terms in `coef_phi`, one with them in
`linear_coeffs_phi`:

| a3 (waves) | max abs(E_split)/abs(E_full) | min |
|---|---|---|
| 1e-3 | 1.000148 | 0.999852 |
| 1e-2 | 1.001480 | 0.998523 |
| 0.1  | 1.014931 | 0.985365 |
| 1.0  | 1.163809 | 0.865946 |
| 10   | 6.464995 | 0.336246 |

The `a1` (s2-linear) counterpart is pure phase: `|E| ratio = 1.0000000000` at a1 = 1, 100 and
2017 waves — so half of the convention is sound and half is not.

*Is `a3` ever large on real fits?* (`t5_a3_real.py`, `t6_extract_flag.py`) — stock N-BK7
singlet, `lambda = 1.31 um`, `source_box_half=20 um`, `pupil_box_half=0.02`, order 6:

| fit | a1 (waves) | a3 (waves) |
|---|---|---|
| on-axis, `source_centre=(0,0)` | 3.18e-11 | **3.18e-11** |
| `source_centre=(100 um, 0)` | 2.699e+3 | **2.740e+3** |
| `source_centre=(500 um, 0)` | 8.749e+3 | **8.909e+3** |
| `surface_diffraction={0: (1,0,2um,2um)}` | 1.161e+3 | **5.791e+2** |
| `surface_diffraction={1: (1,0,2um,2um)}` | 1.025e+3 | 6.81e-11 |

(`a1` and `a3` come out nearly equal because `u_s2x` and `u_v2x` are strongly collinear in the
training data — the library's own `test_w6_a17_default_canonical_fit_is_rank_deficient`
already records the rank deficiency — so the min-norm `lstsq` splits the ramp roughly 50/50
between them. On the training manifold that is harmless; inside the `v2` integral it is not.)

*End-to-end damage* (`t6`, `t8_peak_location.py`). Same lens, same source, only the flag
changes; both fits have the **same** residual `8.613e-06` waves:

| | peak abs(E) | peak position (m) |
|---|---|---|
| **default `extract_linear_phase=True`** | 2.277e-05 | (-3.174e-04, -6.228e-04) |
| `extract_linear_phase=False` | 1.9996e-03 | (9.785e-05, 0.0) |
| independent ray trace, chief ray of the 100 um source | — | **(9.665e-05, 0.0)** |

The default puts the PSF **~700 um from the chief ray**, off-axis in `y` where the source has
no `y` offset at all, with **88x too little peak amplitude**. For the surface-0 grating the
*normalised* profile also differs by 11.2x / 0.106 over the bright pixels, so it is not even a
rescale. The on-axis case is unaffected (ratio exactly 1.0), which is why every pinned test
passes.

*Cross-check that the machinery is otherwise sound* (`t7_bruteforce_v2.py`): on the on-axis
fit the code reproduces a brute-force 2-D quadrature of its own `v2` integrand to
**7e-06 relative** at the peak pixel — the Wick/Gaussian-moment engine is correct.

*Fix validated* (`t9_fix_check.py`): monkey-patching `eval_phi_with_v2_grad` to re-add only
`a3·u3 + a4·u4` (and `+a3, +a4` to the gradient), still dropping `a0,a1,a2`, makes the
default-flag result reproduce the `extract_linear_phase=False` reference to
**2.2e-05 relative on every bright pixel** and restores the PSF to the chief ray.

**Impact.** Any fit whose `Phi` carries linear-in-`v2` content returns a field in the wrong
place at the wrong amplitude, silently. Reached by: (a) `propagators/subaperture.py:524`,
which passes `source_centre=(cx_i, cy_i)` for **every off-centre patch** of the
patch-decomposition propagator; (b) any off-axis field point fitted with `source_centre`;
(c) `surface_diffraction` on surfaces where `a3 != 0`. Same defect for `aberration_tensor`
(`_compute_M_b`) and both JAX twins (`_compute_M_b_xp`).

**This is a re-opened, already-solved bug.** Audit item N4
(`docs/audits/AUDIT_WAVE_LENS_MODELS_2026_07_02_REVIEW.md:122-128`) found exactly this in the
sibling `lumenairy/elements/lenses_maslov.py` — "*never re-applied … silently wrong (lost
output tilt, shifted stationary points) for decentered / tilted / off-axis configurations*" —
and the fix **was implemented there**: `lenses_maslov.py:1789-1804` splits the ramp, `:821-825`
adds `lin_v3/lin_v4` to the saddle gradient, `:2914-2915` adds `lin_v3*u_v2x + lin_v4*u_v2y`
to `opd_star`. The `asymptotic*` family — whose own module docstring calls itself "the public
version of the `apply_real_lens_maslov` fit" — never received it.

**Recommended fix.** Mirror `lenses_maslov`. In `CanonicalPolyFit.eval_phi_with_v2_grad`
(`asymptotic_canonical_fit.py:207-225`) always add `a3*u3 + a4*u4` to `phi` and `a3`/`a4` to
`du3_phi`/`du4_phi`, and let `include_linear` gate only `a0 + a1*u1 + a2*u2`. `H_Phi` needs no
change (linear terms have zero second derivative). Mirror in
`asymptotic_jax_twin._CanonicalPolyFit_eval_phi_xp`. Then rewrite
`tests/unit/test_niche_audit_w6_asymptotic.py:1140` and `:1173` — `:1187`'s
`assert abs(a3)+abs(a4) < 1e-8` is currently a *premise* that the authors knew would
"corrupt the AMPLITUDE too"; it should become a test on an off-axis `source_centre` fit that
asserts the amplitude is *preserved*.

---

### [P1] The `v2`-integrand carries `|det J|` where Van Vleck–Maslov requires `-i·sqrt(|det J|)/lambda` — the missing factor is field- and wavelength-dependent, not a constant
`lumenairy/propagators/asymptotic_maslov.py:277` (`detJ = np.abs(J00*J11 - J01*J10)`),
used at `lumenairy/propagators/asymptotic.py:646`,
`lumenairy/propagators/asymptotic_aberration_tensor.py:230, 961`,
`lumenairy/propagators/asymptotic_jax_twin.py:285, 421, 467`.

**What is wrong.** Starting from the exact HF/Van Vleck kernel in the endpoint variables,
`d^2 V/ds1 ds2 = (n2/lambda) J^{-1}`, so after `s1 -> v2` the correct integrand weight is
`|det J| · sqrt(|det d^2V/ds1 ds2|) = sqrt(|det J|)/lambda`, with the d=2 Maslov phase `-i`.
The code uses `|det J|` to the **first** power and no `1/lambda`, so

    E_code = (i · lambda · sqrt(|det J(s2)|)) · E_true.

The docstring (`asymptotic.py:402-414`) describes this as "No radiometric normalisation …
supply your own prefactor if you need absolute radiometry", i.e. treats it as a constant.

**Evidence (measured).**
*Absolute oracle* (`t16_modal_absolute.py`): a synthetic `CanonicalPolyFit` encoding exactly
free-space Fresnel propagation (`s1 = s2 - z v2`, `Phi = (z + z|v2|^2/2)/lambda`, `z = 20 mm`,
`lambda = 1 um`, `w_s = 200 um`, `w_p = 1e3` so the pupil is effectively flat) versus the
analytic `q`-parameter Gaussian-beam result:

    measured  E_code/E_true = (8.09e-19 + 2.0000000000256e-08 j)
    predicted i·lambda·sqrt(|det J|) = i·lambda·z = 2e-08 j
    |measured/predicted - 1| = 4.24e-11 ;  spatial spread of the ratio 4.2e-11

*The sibling HF path is correct, so the two disagree by exactly this factor*
(`t15_hf_absolute.py`): a synthetic `HFPolyFit` with the exact Fresnel `Phi(s1,s2)` through
`propagate_hf_chebyshev_quadrature` reproduces the analytic Gaussian beam with
`max|E_code - E_ref| / max|E_ref| = 2.74e-11`, amplitude ratio `1.0 ± 1.3e-08`, phase spread
`1.6e-08` rad — the `-1j` + `sqrt(|det Phi''|)` comment at
`asymptotic_canonical_fit.py:1264-1274` is exactly right, and the Van Vleck density came out
`4.99999999999e+07` against the analytic `1/(lambda z) = 5e+07`.

*How non-constant is it?* (`t2/t3_detJ_scan.py`) On the stock singlet `sqrt(|det J|)` over the
alive field spans a ratio of 1.00044 (grid = 30 % of the fit box) to 1.0017 (95 % of the box),
so on that design the *shape* error is only 0.04–0.17 %. The `lambda` factor, however, is
exact and full.

**Impact.** (a) The propagator's output cannot be compared across wavelengths: intensities
carry a spurious `lambda^2·|det J|`, which biases any chromatic sum (the module's own comments
cite `MultiWavelengthMerit` as a target). (b) It cannot be compared against the other
propagator families that `optimize/driver.py:163-195` and `propagators/dispatch.py:1337-1349`
route through the same merit machinery. (c) Downstream,
`lumenairy/optimize/jax_merits.py:471` computes `piston_weight * (1 - |res|**2)` and documents
it as a Strehl deficit that "-> 1 for a perfect system"; measured on the stock singlet
`L = -5.1280 + 5.8997j`, `|L|^2 = 61.1`, so the merit is **-60.1**, i.e. unbounded and of the
wrong sign for an optimiser. (Item (c) lives in `optimize/`, flagged here because its root
cause is this normalisation.)

**Recommended fix.** Replace `detJ` with `sqrt(detJ)` and multiply the leading amplitude by
`-1j / fit.wavelength`, in all three `_compute_M_b*` consumers, i.e.
`amp_lead = -1j * sqrt(detJ)/lambda * (pi/sqrt(det M)) * G0 * exp(2 pi i Phi*) * exp(b_quad)`.
That makes `propagate_modal_asymptotic` agree in absolute terms with
`propagate_hf_chebyshev_quadrature` and with Fresnel/ASM, and makes `|L_(0,0)|` a real Strehl
amplitude so `1 - |L|^2` becomes meaningful. If the scale must stay for back-compat, the fix
is at minimum to *document the factor as `i·lambda·sqrt(|det J|)`* rather than as a constant,
and to expose it on the result so callers can divide it out.

---

### [P1] `jax.grad` through `aberration_tensor_lg00_jax`'s DEFAULT `w_o` is ~86 % wrong (degenerate `eigvalsh`)
`lumenairy/propagators/asymptotic_jax_twin.py:392-418`
(`w_o = 1.0 / jnp.sqrt(jnp.maximum(jnp.linalg.eigvalsh(jnp.real(M))[-1], 1e-30))`).

**What is wrong.** `Re M = J^T J / w_s^2 + I / w_p^2` is near-isotropic for any
rotationally-symmetric system, so its two eigenvalues are nearly degenerate. JAX's `eigh` JVP
carries the eigenvector-rotation term `1/(lambda_i - lambda_j)`; at that degeneracy the
eigenvalue gradient is round-off dominated (and `lambda_max` is not differentiable at an exact
degeneracy). This is the *same* failure class the module already diagnosed and fixed for
`lstsq` (`asymptotic_jax_twin.py:805-857`, "*an SVD-based gradient whose formula has a
`1/(s_i^2 - s_j^2)` term*") — `eigvalsh` was left in place.

**Evidence (measured)** (`t12_eig_degeneracy.py`, `t13_grad_all.py`). Stock singlet,
`w_s = 20 um`, `w_p = 0.02`, on-axis:

    Re M eigenvalues = [2.400201783744e+07, 2.400201784537e+07]   gap/mean = 3.31e-10

Gradient of the shipped merit `1 - |L|^2` against a **converged 5-point FD** (stable over
h = 3e-6 … 3e-7 m, so this is not FD noise):

| derivative | default `w_o` (eigvalsh) | 5-pt FD | rel err | explicit `w_o` | rel err |
|---|---|---|---|---|---|
| `d/ds2x` | 9.438e-03 | 6.738e-02 | **8.60e-01** | 5.5029e-06 vs 5.5067e-06 | 6.9e-04 |
| `d/dv*_x` | 2.316e-04 | 1.661e-03 | **8.61e-01** | 2.205e-07 vs 2.239e-07 | 1.5e-02 |
| `d/dw_s` | 1.814377e+03 | 1.814377e+03 | 6.6e-09 | — | 8.0e-13 |
| `d/dw_p` | 6.109565e+03 | 6.109565e+03 | 2.0e-12 | — | 2.9e-13 |

`d/dw_s`, `d/dw_p` survive because a scalar rescaling of `M` does not rotate its
eigenvectors; anything that *rotates* `Re M` (image point, saddle point, decentre) is wrong by
~86 %.

**Impact.** `lumenairy/optimize/jax_merits.py:456` calls `aberration_tensor_lg00_jax` with no
`w_o`, so the library's flagship differentiable design merit is on this path. Today it
survives because `jax_merits` only exposes `w_s`/`w_p` as live differentiable slots, but the
function's own contract (`asymptotic_jax_twin.py:364-365`) advertises
"*Differentiable via jax.grad wrt fit coefficients, s2_image, v_star, source_point, w_s, w_p,
w_o, v2_centre*" — two of those are wrong, and the moment `image_points` or a decentre becomes
differentiable the merit's gradient is garbage.

**Recommended fix.** Replace the eigensolve with the closed-form 2x2 largest eigenvalue
`lam_max = (a+d)/2 + sqrt(((a-d)/2)**2 + b**2)` (smooth JVP, no eigenvectors, exact), or wrap
`w_o` in `jax.lax.stop_gradient` and say so (`w_o` is only a normalisation convention on this
branch, per `_lg00_sampling_waist`). Add a gradient-vs-FD test on `s2_image` to the JAX pins —
there is currently none.

---

### [P1] `propagate_modal_asymptotic_lg00_jax` returns non-finite values where the NumPy twin returns 0
`lumenairy/propagators/asymptotic_jax_twin.py:447-557` (and `aberration_tensor_lg00_jax:378-431`).

**What is wrong.** The JAX twins reproduce the arithmetic of the NumPy path but none of its
guards: no `in_box_s2` / `in_box_v` mask (`asymptotic.py:507-532`), no
`|Re b_quad| <= 700` overflow gate (`:584`), no `|det M| >= 1e-300` gate (`:562`), no final
finite mask (`:764-766`). The docstring claims the result matches "the NumPy version's
output".

**Evidence (measured)** (`t10_jax.py`). Inside the fit box, parity is excellent:
`aberration_tensor` (0,0) agrees to **7.99e-11** relative, and the 17x17 grid agrees to
**2.64e-10** RMS / 7.14e-10 worst. On a grid 3x the fit half-box, NumPy zeroes 72 of 81 pixels
while the JAX twin returns **60 non-finite values** on those pixels (and finite garbage up to
3.83e-22 on the rest).

**Impact.** A single NaN in a `vmap`'d field poisons any downstream `jax.grad` (NaN propagates
through the whole reverse sweep), and a caller following the docstring will not know the grid
left the box. Silent.

**Recommended fix.** Port the four masks as `jnp.where` gates (the in-box test is
`jnp.abs(u) <= 1` on the four normalised coordinates, already computed inside
`eval_*_xp`), and return `0` outside — bit-comparable to NumPy and NaN-free for autodiff.

---

### [P2] `propagate_modal_asymptotic` silently zeroes out-of-box pixels and leaks bare NumPy `RuntimeWarning`s
`lumenairy/propagators/asymptotic.py:507-534, 551-565, 584-595, 640-654, 764-766`.

**What is wrong.** (a) Every masking step (`in_box_s2`, `in_box_v`, `finite_M`, `ok_det`,
`ok_bquad`, `ok_amp`) drops pixels to exactly `0` with **no warning and no returned
diagnostic**, and the function can return an all-zero array (`return flat_out.reshape(...)` at
5 separate early exits). The sibling `propagate_hf_chebyshev_quadrature` *does* warn when the
grids leave the fit box (`asymptotic_canonical_fit.py:1057-1082`, audit W6-A16) — the
inconsistency is the problem: `optimize/driver.py:163-195` and `dispatch.py:1337-1349` feed
this field straight into cross-propagator wave merits, where "mostly zero" is
indistinguishable from "dark".

(b) `sqrt_detM_all = np.sqrt(det_M)` (`:595`) is applied to the **whole** array including the
NaN entries produced by out-of-box Newton results, and `math.pi / safe_sqrt` (`:647`) then
divides by them. Measured on a routine call: `RuntimeWarning: invalid value encountered in
sqrt` at `asymptotic.py:595` followed by `invalid value encountered in divide` at `:647`.
The "dtype-aware sentinel" pattern at `:640-645` is incomplete: `np.where(sqrt_detM_all != 0,
...)` does not catch NaN (`NaN != 0` is `True`).

**Recommended fix.** Emit one `RuntimeWarning` naming the fraction of pixels dropped and the
dominant reason (the masks are already separate booleans), or return them on a result object.
Compute `np.sqrt` / the division only on `valid` indices (`det_M[valid_idx]`), which also
removes the warnings.

---

### [P2] Three redundant Chebyshev basis builds per evaluation; `(M, N_pix)` temporaries cost 10.8 kB per pixel
`lumenairy/propagators/asymptotic_canonical_fit.py:180-205` (`eval_s1_with_v2_grad` calls
`_evaluate_polynomial_4d_and_grad34` **twice**), `:207-225` (`eval_phi_with_v2_grad`, a third
time at the same nodes), `lumenairy/propagators/asymptotic_maslov.py:146-208`
(`_phi_v2_hessian_batch` builds a fourth set), backed by
`lumenairy/elements/lenses.py:873-921`.

**Evidence (measured)** (`t14_perf.py`, `t18_perf_fuse.py`). Stock singlet, poly_order 6
(M = 210 basis terms):

| grid | wall | per pixel | tracemalloc peak | per pixel |
|---|---|---|---|---|
| 32x32 | 531 ms | 518 us | 11.1 MB | 10.8 kB |
| 64x64 | 2255 ms | 551 us | 44.2 MB | 10.8 kB |
| 128x128 | 10283 ms | 628 us | 176.7 MB | 10.8 kB |
| 256x256 | 24018 ms | 366 us | 706.9 MB | 10.8 kB |

cProfile at 128x128: `_evaluate_polynomial_4d_and_grad34` = **4.677 s tottime of 5.92 s
total (79 %)**, 27 calls (12 Newton iterations x 2 for `s1x`/`s1y`, + 2 + 1).
Extrapolated, a 1024x1024 output would peak at ~11 GB.

Micro-benchmark of the fix (identical arithmetic):

| N_pix | as shipped (3 calls) | one fused basis build | + `T12` hoisted out of the Newton loop |
|---|---|---|---|
| 1 024 | 66.1 ms | 13.1 ms (**5.06x**) | 12.6 ms (5.26x) |
| 4 096 | 294.1 ms | 106.9 ms (2.75x) | 112.7 ms (2.61x) |
| 16 384 | 843.4 ms | 372.6 ms (2.26x) | 304.0 ms (2.77x) |
| 65 536 | 3594.4 ms | 1268.7 ms (2.83x) | 891.4 ms (**4.03x**) |

**Recommended fix.** (1) Give `CanonicalPolyFit` one `_basis_and_grad34(u1..u4)` that returns
`(basis_f, basis_d3, basis_d4)` once and contract `np.stack([coef_s1x, coef_s1y, coef_phi])`
against it in a single `tensordot` — 2.3-5.1x. (2) In `_solve_envelope_stationary_batch`
(`asymptotic_maslov.py:571-649`) `u1`, `u2` are **constant across Newton iterations**: hoist
`T1`, `T2` and `T12 = T1[K1]*T2[K2]` out of the loop — a further 1.4x at 65 k pixels.
(3) Longer term, replace the `(M, N_pix)` materialisation with a Clenshaw/Horner contraction
in the two `v2` axes, which drops the temporary from `O(M·N)` to `O(order·N)` — ~40x less
memory at order 6.

---

### [P2] `aberration_tensor`'s default path costs ~30 s per field point *and* is aliasing-limited at the same time
`lumenairy/propagators/asymptotic_aberration_tensor.py:445-520` (waist probe), `:1149-1187`
(adaptive `sigma_grid_n`), `:1228-1247` (one full propagate per source mode).

**Evidence (measured)** (`t14_perf.py`), stock singlet, default 11 output modes:

    aberration_tensor default            29.83 s   (sigma_grid_n resolved to 256)
    aberration_tensor sigma_grid_n=64     2.55 s
    aberration_tensor output_modes=[(0,0)] 0.0028 s

and on the same call the function itself warns that Nyquist needs `sigma_grid_n >= 494` but
the default cap truncates to 256 — so the default simultaneously pays 12x the `n=64` cost and
still returns an aliased answer. `lumenairy/optimize/merit_terms.py:1138` calls this inside
the design loop.

Structural costs: `_measure_image_plane_waist` runs up to two extra 32x32 propagates on every
call; the main loop runs a full `n_grid^2` propagate **per source mode**; and
`decompose_lg(..., p_max=max_p_o, ell_max=max_ell_o)` builds `(p_max+1)(2·ell_max+1)` modes
(21 for the default output list) of which only `len(output_modes)` = 11 are read.

**Recommended fix.** (a) Build only the requested `(p, ell)` pairs — give `_lg_mode_conj_stack`
an explicit key list instead of a `(p_max, ell_max)` rectangle. (b) Cache the measured `w_o`
per `(fit, s2_image, w_s, w_p, v2_centre, pupil_amplitudes)` so an optimiser sweep pays the
probe once. (c) With the fixes in the P2 item above the propagate itself is ~3x cheaper, which
is what makes `sigma_grid_n = 512` affordable and removes the alias.

---

### [P3] `pupil_modes` is inert
`lumenairy/propagators/asymptotic_aberration_tensor.py:614` (parameter), `:966-969` (the only
real use: the moment-table order), `:1253` (echoed on the result). The actual pupil content
comes exclusively from `pupil_amplitudes` (`:1083`, `:1236`). A caller who passes
`pupil_modes=[(0,0),(1,0)]` without matching `pupil_amplitudes` silently gets an LG_{0,0}
pupil and a result object that claims otherwise. Either drive the defaults of
`pupil_amplitudes` from `pupil_modes`, or drop the parameter.

### [P3] `aberration_tensor`'s `A_lead` has no overflow guard
`lumenairy/propagators/asymptotic_aberration_tensor.py:961-963`:
`np.exp(0.25 * b @ M_inv @ b)` with no `|Re| <= 700` test, unlike the batched path
(`asymptotic.py:584`). Overflows to `inf` (with a bare NumPy warning) rather than being
masked, on the same inputs the batched path rejects cleanly.

### [P3] The Seidel/Zernike framing overclaims what `L` measures
`lumenairy/propagators/asymptotic.py:21-25` ("*a physically-named aberration tensor whose
indices correspond to the classical Seidel/Zernike modes*"),
`asymptotic_aberration_tensor.py:100-106` ("*Driving |L_{(2,0),0}|^2 to zero suppresses
on-axis spherical aberration*"). `L` is an overlap of the **image-plane field** onto a
**real-waist LG** basis — not a pupil wavefront-error expansion, and LG modes are not Zernike
polynomials (they are orthonormal on `R^2` with a Gaussian weight, not on the unit disc). The
module's own W4-T2 note (`:780-791`) contradicts the claim in measured terms: the `(2,0)`
channel "*is an interference residue whose phase rotates with the design rather than a measure
of spherical aberration*", with **5 of 6 sign flips** across adjacent designs on the default
basis. The naming table itself (`asymptotic_modes.py:618-638`) is consistent with the
`n = 2p+|l|, m = l` radial-order correspondence — it is the *interpretation* that needs the
caveat moved up to the top-level docstring.

### [P3] "Exact at the stationary point" is wrong, three times
`lumenairy/propagators/asymptotic_canonical_fit.py:648-654`,
`lumenairy/propagators/asymptotic_maslov.py:530-531`,
`lumenairy/propagators/asymptotic_jax_twin.py:676-678` all claim the dropped
`sum_k (s1 - s_src)_k d^2 s1_k/dv2 dv2` Gauss-Newton term "is exact at the stationary point".
At the stationary point `J^T(s1-s_src)/w_s^2 = -(v-v_c)/w_p^2`, which is not zero, so
`s1 - s_src != 0` and the term does not vanish; only the *converged residual* is zero.
Harmless for the answer (Newton converges to `F = 0` regardless of the Hessian model) but the
statement is repeated verbatim in three places.

### [P3] The "bit-for-bit / character for character" `w_o` cross-backend contract is false — the NumPy side clamps, the JAX side does not
`lumenairy/propagators/asymptotic_aberration_tensor.py:290-312` returns
`max(min(1/sqrt(lam_max(Re M)), 1.0), 1e-9)`;
`lumenairy/propagators/asymptotic_jax_twin.py:418` returns
`1.0 / jnp.sqrt(jnp.maximum(eig_M_real[-1], 1e-30))` with **no clamp**. Both docstrings
insist they are identical ("*BIT-FOR-BIT the cross-backend contract*",
"*mirrors … character for character so the NumPy … and this twin agree*", "*Change one and
you MUST change the other*"). Measured (`t19_wo_clamp.py`, stock singlet):

| `w_s`, `w_p` | `lam_max(Re M)` | `w_o` NumPy | `w_o` JAX | `abs(L)` NumPy | `abs(L)` JAX | rel |
|---|---|---|---|---|---|---|
| 20 um, 0.02 | 2.4002e+07 | 2.0412e-04 | 2.0412e-04 | 7.81678e+00 | 7.81678e+00 | 8.0e-11 |
| 1, 1 | 1.0096e+00 | 9.9523e-01 | 9.9523e-01 | 6.53123e-08 | 6.53123e-08 | 8.0e-11 |
| 30, 30 | 1.1218e-03 | **1.000000** | **29.857** | 7.22234e-11 | 2.41897e-12 | **0.967** |
| 100, 100 | 1.0096e-04 | **1.000000** | **99.523** | 6.50010e-12 | 6.53123e-14 | **0.990** |

The divergence needs `lam_max(Re M) < 1`, i.e. waists far outside any physical use (a
direction-cosine `w_p > 1`), which is why the pinning test
(`test_w3_t3b_jax_twin_default_w_o_tracks_numpy`) never sees it — so the *practical* impact is
low. The *documentation* impact is not: a contract stated three times in capital letters is
untrue, and the clamp is unmotivated on either side. Either apply the same clamp in the JAX
twin or delete it from the NumPy one and drop the claim.

### [P3] Import-time monkey-patching of another module's classes
`lumenairy/propagators/asymptotic_jax_twin.py:210-212` assigns `eval_phi_xp` / `eval_s1_xp`
onto `CanonicalPolyFit` and `HFPolyFit` at import time. Combined with the
`_register_cache_clearer` blocks at `asymptotic_modes.py:266-283` and
`asymptotic_jax_twin.py:1133-1143`, importing the package mutates global registries and two
classes owned by a different module. The methods belong on the dataclasses.

### [P3] Triplicated algorithms that must be kept in sync by hand
`_compute_M_b` (`asymptotic_aberration_tensor.py:184`) / `_compute_M_b_batch`
(`asymptotic_maslov.py:211`) / `_compute_M_b_xp` (`asymptotic_jax_twin.py:240`);
`solve_envelope_stationary` (`asymptotic_canonical_fit.py:543`) /
`_solve_envelope_stationary_batch` (`asymptotic_maslov.py:484`) / `_newton_loop`
(`asymptotic_jax_twin.py:665`);
`_polynomial_substitute_linear_2d` (`asymptotic_aberration_tensor.py:1262`) /
`_batched_polynomial_substitute_linear_2d` (`asymptotic_maslov.py:363`);
`cheb_nodes` defined three times (`asymptotic_canonical_fit.py:346, 876`,
`asymptotic_jax_twin.py:959`); `cheb_vand_jax` (`asymptotic_jax_twin.py:1070`) re-implements
`_math.chebyshev.chebyshev_vandermonde(xp=...)`. The P0 above is precisely a fix that landed
in one copy of an algorithm (`lenses_maslov`) and not the others — this is the structural
cause.

### [P3] Docstring-to-code ratio hides the code
`propagate_modal_asymptotic` (`asymptotic.py:252-768`) is a 516-line function of which
~185 lines are one docstring (measured: only 45 % of `asymptotic.py` and 39 % of
`asymptotic_aberration_tensor.py` is code), with individual paragraphs carrying
five-significant-figure
measurement narratives from four different audits. The measurements are valuable, but they
belong in `docs/audits/` with a one-line pointer; as written, the *contract* (what the
function returns, on what scale, with what guards) is buried, and — as the P0 and P1 findings
show — three of the narrative claims are wrong while reading as authoritative.

---

## Performance opportunities (measured)

| # | opportunity | measured / estimated gain | how measured |
|---|---|---|---|
| 1 | Fuse the 3 `_evaluate_polynomial_4d_and_grad34` calls at identical nodes into one basis build + one stacked `tensordot` | **2.3x - 5.1x** on the dominant 79 % of runtime | `t18_perf_fuse.py`, poly_order 6, M = 210, N = 1 024 … 65 536 |
| 2 | Hoist `T1`, `T2`, `T12 = T1[K1]*T2[K2]` out of the Newton loop in `_solve_envelope_stationary_batch` (`u1`, `u2` are loop-invariant) | further **1.4x** at 65 k pixels (4.03x total vs shipped) | same |
| 3 | Clenshaw/Horner contraction instead of materialising the `(M, N_pix)` basis | temporary drops from `O(M·N)` to `O(order·N)` — **~40x** less memory at order 6; would take the measured 10.8 kB/pixel to ~0.3 kB/pixel (707 MB -> ~18 MB at 256x256) | arithmetic on the measured tracemalloc peak |
| 4 | `decompose_lg` in the sigma branch: build only the requested `(p, ell)` pairs | 21 modes built / 11 used on the default output list — **1.9x** on the projection step | code read + `_lg_mode_conj_stack` loop at `asymptotic_modes.py:379-389` |
| 5 | Cache `_measure_image_plane_waist` per `(fit, s2_image, w_s, w_p, v2_centre)` | removes up to two 32x32 propagates per `aberration_tensor` call (~1 s of the measured 29.8 s) | `t14_perf.py` |
| 6 | `aberration_tensor` closed-form branch: `out_const` is recomputed inside the `k_pup` loop although it depends on neither `k_pup` nor `k_src` (`asymptotic_aberration_tensor.py:1108-1110`) | `n_src x n_pup` redundant evaluations; negligible absolute, trivially fixable | code read |
| 7 | `propagate_modal_asymptotic_lg00_jax` is **47x slower than NumPy** and does not improve on a second call (1354 ms -> 1029 ms vs NumPy 21.7 ms at 17x17) because the `vmap` is deliberately not `jit`-ed (`asymptotic_jax_twin.py:548-554`) | registering `CanonicalPolyFit` as a pytree (`jax.tree_util.register_pytree_node`, coefficients as leaves, normalisers/`multi_indices` as aux data) would let the twin be `jit`-ed once per fit shape | `t10_jax.py` |
| 8 | `propagate_modal_asymptotic` runs the Newton loop for the full `max_iter=12` (27 evaluator calls observed) because `tol=1e-12` is absolute on a dimensional residual that starts at ~1e7 — pixels only ever leave via the *stall* test | make the iteration test scale-relative like the *verdict* already is (`asymptotic_maslov.py:558-570, 638`); measured cold-start residual median 2.0e+07, converged 6.8e-09 | profile + the W6-A2 note's own numbers |

---

## Alternative algorithms / methods

1. **Uniform asymptotics near caustics (Ludwig 1966; Kravtsov 1964; Chester–Friedman–Ursell).**
   The module's own accuracy note (`asymptotic.py:394-400`) measures the leading-order value
   **55x low** 1.6 widths off a fold caustic (2.258e-08 vs a converged 1.250e-06) because `v*`
   is the *envelope*-stationary point, not the complex saddle of the full exponent. The
   textbook cure is the Airy-type uniform expansion: locate the *two* saddles of
   `2 pi i Phi - |s1-s_src|^2/w_s^2 - ...`, form `zeta = (3/4 (S2 - S1))^{2/3}`, and evaluate
   `Ai(-k^{2/3} zeta)` / `Ai'`. Cost: one extra complex Newton solve per pixel (the Hessian is
   already assembled) plus two Airy evaluations — roughly 1.5x the present per-pixel cost, and
   it is uniformly valid through the caustic instead of failing a factor 55 beside it.
   Refs: Ludwig, *Comm. Pure Appl. Math.* **19** (1966) 215; Kravtsov & Orlov,
   *Geometrical Optics of Inhomogeneous Media*, ch. 5; Chester, Friedman & Ursell,
   *Proc. Camb. Phil. Soc.* **53** (1957) 599.

2. **Complex (steepest-descent) saddle instead of the envelope-stationary point.**
   Strictly cheaper than (1) and removes the dominant documented error: solve
   `d/dv2 [2 pi i Phi - |s1-s_src|^2/w_s^2 - |v2-v_c|^2/w_p^2] = 0` in **complex** `v2` with
   the same Gauss-Newton structure (the Jacobian is the existing `M`, already complex). The
   Gaussian-moment contraction then expands about the true saddle, so `exp(b_quad)` stops
   carrying the whole answer off-axis. Same cost class as today (the Newton is ~4 iterations
   in either case), and it subsumes the W6-A2 "`v*` only has to be a good enough centre"
   caveat. Ref: Bleistein & Handelsman, *Asymptotic Expansions of Integrals*, ch. 7.

3. **Gaussian beam summation / Gabor decomposition (Cerveny, Popov & Psencik 1982).**
   Replace the single Gaussian-regularised saddle by a sum of paraxial Gaussian beams launched
   along the traced rays: each beam carries its own complex `q` matrix propagated by the
   ray-centred `ABCD`, and the field is a sum with no stationary-point solve at all. Buys:
   *caustic-free by construction* (the complex `q` never becomes singular) and an amplitude
   that is automatically Van Vleck-normalised, so both the P0 and P1 findings above disappear
   structurally. Costs: `O(N_beams)` per pixel rather than `O(1)`, and a beam-width parameter
   to tune; the library already has `propagators/gbd.py`, so the honest option here is to
   *route the aberration-tensor merit through GBD* rather than re-derive it.
   Ref: Cerveny, Popov & Psencik, *Geophys. J. R. astr. Soc.* **70** (1982) 109.

4. **Wigner / phase-space transport (Alonso, *Adv. Opt. Photon.* **3** (2011) 272).**
   The quantity this module actually wants — a field expanded in a phase-space basis with
   physically-named indices — is exactly the Wigner-function picture. Propagating the Wigner
   distribution through the traced canonical map is *exact* for quadratic `Phi` and reduces to
   ray transport otherwise, and the LG/HG moments of the Wigner function are the standard
   "beam quality" invariants (`M^2`, the 10 second-order moments) that have a genuine
   Seidel-like interpretation. Buys a physically defensible replacement for the `(2,0)`
   channel whose current instability (5 of 6 sign flips) is documented at `:776-778`.
   Costs: a 4-D phase-space grid, or the moment hierarchy only (cheap, but then it is the
   second-moment matrix, not a full tensor).

5. **Direct Van Vleck / Huygens quadrature with the traced eikonal — already in this file.**
   `propagate_hf_chebyshev_quadrature` is verified here to 2.7e-11 against an analytic
   Gaussian beam. For any case where the modal propagator's saddle is shaky (near-caustic,
   high LG order, off-axis `source_centre`), it is the correct oracle and should be what the
   accuracy pins compare against. Its cost is `O(N_in^2 · N_out)` but it chunks and
   vectorises well (measured in `benchmarks/test_bench_asymptotic.py`).

6. **Rank-deficiency of the canonical fit (`test_w6_a17_default_canonical_fit_is_rank_deficient`).**
   `u_s2x` and `u_v2x` are strongly collinear whenever `source_box_half` is small against the
   pupil footprint — which is the default configuration. A TSVD/ridge with an explicit,
   *reported* effective rank, or re-normalising the output box on the two dominant principal
   directions of `(s2, v2)` rather than per-axis, would make `a1`/`a3` individually meaningful
   instead of min-norm artefacts, and would stop the Chebyshev fit from extrapolating wildly
   off the thin training slab (`propagate_hf_chebyshev_quadrature` already measures
   2.23x / 8.48x / 44.9x amplitude inflation at 1.5x / 3x / 10x the box, `:1078-1081`).

---

## Code organization observations

* **6 files, 6 048 lines, of which only 2 591 (43 %) are code** — measured by tokenizing +
  `ast`: 1 863 docstring lines (31 %), 970 comment lines (16 %), 624 blank. Per file:
  `asymptotic.py` 45 % code, `asymptotic_canonical_fit.py` 52 %, `asymptotic_maslov.py` 46 %,
  `asymptotic_jax_twin.py` 40 %, `asymptotic_aberration_tensor.py` 39 %,
  `asymptotic_modes.py` 36 %. Longest functions: `aberration_tensor` **651** lines,
  `propagate_modal_asymptotic` **516**, `fit_canonical_polynomials` 308,
  `propagate_hf_chebyshev_quadrature` 293, `fit_canonical_polynomials_jax` 258,
  `_solve_envelope_stationary_batch` 166. Individual parameters
  (`sigma_grid_n`, `curvature_matched_basis`) carry 40-60 line measured-results tables inline.
  Three of those narrative claims are demonstrably wrong (P0, P1, and the
  "exact at the stationary point" P3) while reading as settled fact — the format actively
  impedes review.
* **The v5.1.0 "purely mechanical" split is not clean.** `asymptotic.py` is a shell that
  re-imports 30 private names from four submodules *and* keeps one 516-line function in the
  shell purely to preserve a monkey-patch contract with one test
  (`asymptotic.py:234-249`). `asymptotic_aberration_tensor.py` then does a late import back
  from the shell (`:846`) to break the resulting cycle. The cycle is a symptom: `propagate`
  and `aberration_tensor` are mutually recursive and belong in one module.
* **Three near-copies of every core routine** (NumPy scalar / NumPy batched / JAX) with no
  shared kernel — see the P3 item. The P0 defect is exactly a fix applied to one copy of an
  algorithm and not the others.
* **`asymptotic_modes.py` is clean** and is the one module in the partition I would hold up as
  a model: small pure functions, one cache each with a lock, a documented cache key, and the
  key was tightened twice in response to real collisions (`dx/dy`, then the corner
  fingerprint). Verified correct below.
* Cache/thread-safety review found no defects: `_LG_MODE_STACK_LOCK` / `_HG_MODE_STACK_LOCK`
  guard the full read-modify-write, the cached stacks are `setflags(write=False)`,
  `_JAX_IFT_SOLVER_CACHE` uses correct double-checked locking, `_lg_polynomial_items` is an
  `lru_cache` keyed on `(p, ell, float(w))`, and all four clearers are registered with
  `_cache_registry`. There is no mutable module-level configuration.

---

## Unverified suspicions

* **`fit_canonical_polynomials_jax` does not reject `|v1|^2 >= 1`.** The NumPy twin raises
  (`asymptotic_canonical_fit.py:386-391`); the JAX one clamps
  `N1 = sqrt(max(1 - sumsq, 0))` (`asymptotic_jax_twin.py:993-994`) and only checks
  `pupil_box_half < 1`. For `pupil_box_half` in `[0.708, 1)` the corner nodes are evanescent
  and the JAX fit would silently train on `N1 = 0` rays. Not run: would need a prescription
  that survives that aperture.
* **`_normaliser_jax` vs `_fit_normaliser` under a fully-dead axis.** Both map an all-equal
  axis to `half = 1.0`, but `_normaliser_jax` uses `nanmin`/`nanmax` over a masked array —
  if *every* ray dies the result is NaN rather than the NumPy `RuntimeError`. The liveness
  check above it is skipped under tracing (`:1017-1018`), so under `jit` this would produce a
  silent NaN fit. Would need a jitted call with a fully vignetting prescription to confirm.
* **`d/d(source_point)` of `aberration_tensor_lg00_jax`** disagreed with a 5-point FD by
  6.4 % (default `w_o`) and 6.9 % (explicit `w_o`). The derivative is ~9 orders smaller than
  `d/dw_s`, so I could not separate real error from FD conditioning; would need a
  complex-step or a `jax.jvp`-vs-`jax.vjp` consistency check on a non-degenerate configuration.
* **`propagate_modal_asymptotic`'s `row_reset` branch on 1-D input.** `Nx_grid` falls back to
  `s2x_arr.size` when `ndim < 2`, but the reset is gated on `s2x_arr.ndim >= 2`, so
  `row_reset` silently degrades to `1d_raster` for a 1-D grid. Both are deprecated legacy
  modes, so I did not pursue whether any caller depends on the distinction.
* **`aberration_tensor` when `s2_image` sits at the fit-box edge.** `_s2_validity_room`
  returns `<= 0`, the `if room > 0.0` guard (`:1146`) then leaves `extent = 4*w_o`
  un-clamped and the whole sigma grid can fall outside the box, returning `L` identically
  zero — the exact failure the W3-T3 comment says the clamp exists to prevent. I did not
  construct the case.

---

## Checked and found correct

* **Wick / Gaussian moments.** `gaussian_moment_2d` and `gaussian_moment_table_2d` match a
  brute-force 2-D complex-Gaussian quadrature to **9.42e-14** worst relative over all
  `(a, b)` with `a + b <= 6`, and the normalisation `Z = pi/sqrt(det M)` to 9.37e-14
  (`t1_moments_modes.py`). The batched copy in `asymptotic_maslov.py:285-350` is the same
  algebra.
* **LG / HG bases.** Continuous Gram matrix of `LG_{p,l}`, `p<=2`, `|l|<=2`:
  `max |G - I| = 3.13e-13`, worst off-diagonal 1.67e-14. `HG_{m,n}`, `m,n <= 3`:
  `max |G - I| = 3.15e-13`. `decompose_lg` recovers a random 15-mode superposition to
  **5.68e-13** absolute. The `N = sqrt(2 p!/(pi (p+|l|)! w^2))` normalisation and the
  `L_p^{|l|}` / `(x+isy)^{|l|}` / `(x^2+y^2)^k` expansion are all right.
  `lg_seidel_label`'s table is consistent with `n = 2p+|l|`, `m = l`.
* **Chebyshev helpers.** `T'_n = n U_{n-1}` and the second-derivative recurrence
  `T''_{n+1} = 2x T''_n + 4 T'_n - T''_{n-1}` are algebraically correct (derived from the
  T-recurrence); the `xp` paths mirror the NumPy ones.
* **The saddle-point algebra.** `M = J^T J/w_s^2 + I/w_p^2 - i pi H_Phi`,
  `b = 2 pi i g - 2 J^T r*/w_s^2 - 2 dv/w_p^2`, `delta* = M^{-1}b/2`,
  `b_quad = b^T M^{-1} b/4`, `pi/sqrt(det M)` — all match the second-order expansion of the
  stated integrand term by term, and the closed form reproduces a brute-force quadrature of
  its own integrand to **7e-06** on the on-axis singlet (`t7`).
* **The W6-A1 Maslov claim is correct.** Measured over three regularisation regimes:
  `min eig(Re M)` = 2.84e+03 / 1.00e+02 / 1.00e+00 (always > 0), `|arg det M| <= 3.141464`
  (never reaches `pi`), and the identity `arg det M = -atan(k1) - atan(k2)` holds to
  **2.9e-14**. The legacy raster modes are indeed spurious: `1d_raster` sign-flips 2315 of
  4119 alive pixels and `row_reset` 531, both warn, and `principal` is right.
* **`propagate_hf_chebyshev_quadrature` is absolutely correct**, including the `-1j` d=2
  Maslov factor and the `sqrt(|det d^2Phi/ds1 ds2|)` density: against an analytic
  Gaussian-beam reference, `max|dE|/max|E| = 2.74e-11`, amplitude ratio `1.0 +- 1.3e-08`,
  phase spread `1.6e-08` rad. The lambda-cancellation argument in the comment at
  `asymptotic_canonical_fit.py:1264-1274` is exactly right.
* **NumPy/JAX parity inside the fit box.** `aberration_tensor(..., [(0,0)])` vs
  `aberration_tensor_lg00_jax`: **7.99e-11** relative.
  `propagate_modal_asymptotic` vs `propagate_modal_asymptotic_lg00_jax` on a 17x17 in-box
  grid: 2.64e-10 RMS, 7.14e-10 worst. `solve_envelope_stationary_jax_ift` vs the NumPy
  solver: 7.3e-19 absolute on `v*`.
* **x64 enforcement works.** `_require_jax_x64` is called by all four JAX entry points and
  raises with actionable text; no silent float32 path exists. All four import and run on the
  installed **jax 0.10.1**, below the `pyproject.toml` `jax>=0.11` floor — nothing in this
  partition needs 0.11.
* **Input validation.** `w_s <= 0`, `w_p <= 0` and `p < 0` all raise `ValueError` with clear
  messages (via `lg_polynomial`); rank > 2 grids raise with the offending argument named
  (W6-A5); 0-D and 1-D grids work; a NaN pixel is zeroed rather than propagated.
* **Caches and locks** — see the organization section; no defect found.
* `tests/unit/test_niche_audit_w6_asymptotic.py` passes in full (52 passed, 625 s) on this
  checkout, so every finding above is a *gap in the pins*, not a regression.
