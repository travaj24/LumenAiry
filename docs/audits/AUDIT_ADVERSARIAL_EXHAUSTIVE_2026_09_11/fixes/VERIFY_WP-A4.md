# VERIFY-WP-A4 — independent re-verification of WP-A4

Diff under test: commit `32ba3ba2` ("fix(lens-asymptotic): WP-A4 …"), base
`32ba3ba2^`, branch `audit-fixes-2026-09`.  Findings **S2–S7** (audit §2.5),
**Y1–Y5** (§9), §15.1 / §15.6, the WP's own `_solve_fit` finding, plus five
items the orchestrator added mid-pass: the Strehl-deficit ruling, the
`test_a1_auto_n_v2` ruling, the `apply_real_lens_maslov_vector` minimum fix
(WP-A4 report §5 item 6), the red `validation/run_all.py test_lenses` lane, and
WP-A15a's §5 item 8 (three wall-clock assertions and one slow marker in files I
own).

Every number below is measured by me on this checkout, 2026-09-12,
Windows 11 / python 3.14.6 / numpy 2.4.4 / jax 0.11.0, `OPENBLAS_NUM_THREADS=1`.
`lumenairy/_math/chebyshev.py`'s docstring hunk in `32ba3ba2` is WP-A3's and is
out of scope here, as instructed.

---

## 1. Verdicts

| ID | verdict | the independent check I ran | measured |
|---|---|---|---|
| **S2** (P0) | **VERIFIED-WITH-NOTES** | closed-form Fresnel `e^{iπσ/4}/√|det H|` written in my test, on 6 charts WP-A4 did not use (cross term, anamorphic `hx ≠ hy`, `cx ≠ cy ≠ 1`, indefinite `H`) × 3 knob pairs including ODD `n` and `ws` off the WP's ladder | worst relerr **8.33e-15**; `stationary_phase` **2.13e-16**; swap symmetry **1.22e-14** (pre-fix: relerr 1.19 … 9.12, swap factor 7.7).  **Note 1:** the "exact at ANY `(n, ws)`" claim holds only while the tapered lattice fits INSIDE the chart box — measured **8.79e-02** at `ws = 7.5` and **8.17e-01** on a chart whose small Hessian eigenvalue gives `σ2_norm = 1.785`, silently.  Bounded (pre-fix clip over-count reached 2.0e+03), but not announced |
| **S3** (P1) | **VERIFIED** | the orchestrator's `maslov_exit_vertex_check2.py` re-run; plus a NEW chart with an **immersed rear** (`glass_after = 'N-SF11'`, `n_exit = 1.747969`) | ρ² term **−0.053 µm** (report: −0.053; predicted sag −31.255 µm), ρ⁴ +0.099, residual 4.2 nm — reproduces the report exactly.  `at_exit_vertex()` prices the sag leg at `n_exit` to **0.0**; the `n_exit = 1` mistake would be **2.413e-06 m = 1.84 waves** |
| **S4** (P1) | **VERIFIED** | `p12_prefactor.py` re-run; a textbook ABCD / Collins Gaussian oracle written in my script, on a chart with a REAL LENS (the WP checked two flats only); the `_van_vleck_density` algebra restated in my test | `\|E/E_ASM\|` **0.999664 / 0.999629 / 0.998003 / 0.999616 / 0.997914 / 0.996714**, `arg` **+0.0002 … −0.0020** — the report's after-table to 5 digits.  On an f/20 N-BK7 biconvex at d = 5 / 20 mm past the exit vertex the on-axis amplitude matches the ABCD oracle to **0.99782 / 0.99840** and the |E| profile to **1.84e-03 / 1.51e-03**.  `_van_vleck_density` == `sqrt(det_u/(hx hy))` to **0 ULP**.  **Note:** 6 sites go through the shared helper, but `_integrate_levin` writes the same algebra out by hand TWICE — the drift the helper exists to prevent |
| **S5** (P1) | **VERIFIED** | the convention re-derived from `CONVENTIONS.md` §7 (`exp(−iωt)`/`exp(+ikz)` ⟹ `q = z − i z_R` ⟹ `arg(q0/q) = −arctan(z/z_R)`; `Q = conj(1/q_phys)` ⟹ physics ratio `= conj(Q_new/Q_old)`); `p1_gbd_gouy.py` re-run; the deprecation path exercised | `arg(E/A)` **0.000000** at `z = 0.5 / 2 / 10 z_R`, relL2 **1.410e-13 / 5.366e-13 / 4.204e-12** — the report's numbers.  All three compensator functions are no-ops, all three warn "deprecated", and all three still raise `ValueError` on a bad field |
| **S6** (P1) | **NOT FIXED as shipped → FIXED here** | the warning's actual trigger on collimated Gaussians of four widths | the gate used the 2nd moment of `\|FFT(E_in)\|²`, which for a COLLIMATED beam is its diffraction spread: measured 3-σ NA **3.54e-03 / 1.10e-03** at waist 0.25 / 0.8 mm (λ = 1.31 µm), both above the 1e-3 threshold — so it warned on every collimated beam narrower than ~1 mm, the opposite of the WP's stated design.  Re-gated on the WAVEFRONT NA (new `_wavefront_na`): **exactly 0.0** for a flat-phase field at any width, **6.00e-03 / 3.00e-02 / 8.42e-02 / 3.37e-02** for tilt 0.002 / 0.01 rad and f = −20 / +50 mm.  WP-A4's own S6 test fixture was mislabelled (a flat-phase 12 µm Gaussian called "strongly diverging") and is re-based on a real diverging wavefront, with the flat case added as a counter-pin |
| **S7** (P1) | **VERIFIED-WITH-NOTES** | the WP's own 5 pins re-run; the shipped validation lane | the contract holds, but it took `validation/run_all.py test_lenses` RED on three of four JAX cases.  Fixed (below); the lane is **46/46 passed** |
| **Y1** (P0) | **VERIFIED** (WP's oracle re-run) | `test_audit2609_a4_asymptotic.py::test_y1_*` + the whole W6 suite | 41 + 53 passed.  I did not add a second field angle — see §5 open item O-6 |
| **Y2** (P1) | **VERIFIED**, oracle independence **NOT FIXED as shipped → FIXED here** | the weight re-derived from the stationary-phase Fresnel integral in the test files | `van_vleck_weight == −1j √\|det J\|/λ` to **0 ULP** over a 6 × 3 `(det J, λ)` ladder.  WP-A4 had pointed 4 oracles (`_quad_oracle`, `_a9_quad`, two `amp_lead` copies) at the library helper they verify; all four now derive it locally and **still agree** (W6 53 passed, `test_audit_propagation` 101 passed) |
| **Y3** (P1) | **VERIFIED** | `sym2x2_max_eigenvalue` vs `numpy.linalg.eigvalsh` on 207 matrices (exact degeneracy, exact zero, 1e-14 and 1e12 scales, indefinite, a 1e-15 gap, 200 random) | worst relative departure **3.67e-16**; all three `lg00_sampling_waist_from_M` clamp branches exact |
| **Y4** (P2) | **VERIFIED** | the drop diagnostic on a 9×9 grid at 3× the fit half-box, and on a clean in-box 7×7 | **exactly one** warning naming `80/81 (98.8 %)` and the dominant reason; **0** bare NumPy `RuntimeWarning`s; **0** non-finite.  Clean call: 0 warnings, 0 dropped, 0 non-finite |
| **Y5** (P3) | **VERIFIED-WITH-NOTES** | the `pupil_modes` warning both ways; the `van_vleck_weight` result field | warns on a mismatch, silent when matched.  **Note:** the field's docstring round-trip was wrong — it gave the `\|L\|²` factor for `L`.  Fixed |
| **NEW** `_solve_fit` gate | **VERIFIED** | a rank-deficient basis I built by hand (`col8 = col1 + col2`), and the two thresholds checked for derivability | `cond(G) = 2.13e+302`, rank 7/8 → warns, returns `lstsq`'s min-norm solution to **0.0**, and a 1e-14 perturbation of `A` moves the coefficients by **7.99e-15** (pre-fix: 0.869 waves).  Well-conditioned (`cond(G) = 2.39`) is **bit-identical** to the Cholesky.  `_GRAM_COND_SINGULAR == 1/eps` exactly; `_GRAM_COND_MAX = 1e12` sits 3 decades above the audit's well-conditioned fixture (1.24e9) and 6 below the failing one (6.18e18) — derived, not fitted |
| **Ruling 1** (Strehl) | **implemented** | see §3 | the merit's (0, 0) contribution moves from **−4.79e+14** to a Strehl deficit; the coupling is exactly **1.0** (0 ULP) on an aberration-free optic and falls **1.000000 → 0.901312 → 0.811431 → 0.659170 → 0.447292** with aberration |
| **Ruling 2** (`test_a1_auto_n_v2`) | **implemented** | see §3 | the arm cannot fail because `'auto'` now routes to `'stationary_phase'` and `n_v2` is inert — the two calls are **byte-identical**.  Restated on the reference already in the test (**il2 1.325e-04**, bar 5e-3) plus a dated one-off against an `n_v2 = 512` oracle self-converged to **2.14e-06** (both defaults **5.844e-04**).  Runtime 1.64 s |
| **S10 vector** (coordinator) | **implemented** | see §3 | `P_x/P_y` **bit-identical** (0 ULP) under `none`/`power`/`peak`; `'power'` restores the post-Fresnel pair's total power to **1.000000000000000** (= 0.9219035 of the raw input, i.e. `T1*T2` survives in the absolute scale); the s/p diattenuation **−6.4376e-08** survives (5.154e+08 ULP of the ratio) |
| **validation lane** (coordinator) | **implemented** | `validation/run_all.py test_lenses` | **46/46 passed** (27.7 s); `test_asymptotic` **48/48 passed** (62 s) |
| **WP-A15a §5.8** (coordinator) | **implemented** | see §3.5 | 3 wall-clock assertions replaced by exact operation counts (28 / 0 / 28 polynomial builds; counts invariant under a doubled grid); `pytestmark = pytest.mark.slow` on `test_audit_lens_models_2026_07.py` (641.0 s pinned, 206.7 s here), verified to deselect all 69 ids |

---

## 2. Per finding

### S2 — `local_quadrature` and the `'auto'` router — VERIFIED-WITH-NOTES

**What I did that the WP did not.** The auditor's `p6` charts are
`OPD = ½(A u3² + B u4²)` with `s1x = u3`, `s1y = u4`, `hx = hy = 1` — a fixture
in which `c = 1`, so `sqrt(c²) = c` and the S4 square root is invisible, the
half-widths are unity so the de-normalisation is a no-op, and there is no cross
term.  I wrote the general closed form

    I = sqrt(cx cy hx hy) · exp(iπσ/4) / sqrt|det H|,   H = [[a, b], [b, d]]

(the 2-D stationary-phase / Fresnel value; σ = the signature) and ran six
charts — anamorphic `hx ≠ hy`, `cx ≠ cy ≠ 1`, a cross term up to `b = 120`, an
indefinite Hessian, both-negative, a strongly tilted one — at three
`(n_samples, window_sigma)` pairs including ODD counts (7, 33) and widths off
the WP's `{3, 4, 6, 10}` ladder.

**Result.** Worst relative error **8.33e-15** over the 18 combinations;
`stationary_phase` worst **2.13e-16**.  Swap symmetry `(a, b, d) ↔ (d, b, a)`
with `cx/cy` and `hx/hy` swapped: **4.33e-15 / 1.05e-15 / 1.22e-14** (the
audit's diagnostic for the axis-swap defect).

**Note 1 (P3, open).** The module docstring and the report both say the scheme
is "EXACT for a quadratic chart … at ANY `local_n_samples` /
`local_window_sigma`".  It is not.  Out-of-box samples are DROPPED (correctly —
the Chebyshev recurrences are not accurate there) while the taper correction is
still computed on the FULL lattice, so the two stop matching as soon as the
tapered lattice leaves the chart box.  Measured on the anamorphic chart, whose
larger principal width is `σ2_norm = 0.2524`:

| `ws` | lattice reach in `u` | relerr |
|---|---|---|
| 3.0 | 0.757 (inside) | 6.60e-15 |
| 3.7 | 0.934 (inside) | 2.75e-15 |
| 5.0 | 1.262 (outside) | **8.09e-02** |
| 7.5 | 1.893 (outside) | **8.79e-02** |

and **8.17e-01** at the shipped defaults on a chart with `a = 5, b = 4.9,
d = 5` (`det H = 0.99`, `σ2_norm = 1.785`), where `stationary_phase` is exact to
1.38e-15.  This is bounded — the pre-fix `np.clip` over-count reached 2.0e+03 —
but it is SILENT, and it is the same class of defect Y4 fixed elsewhere in this
diff (silent zeroing).  `'auto'` never selects `local_quadrature` any more, so
the exposure is opt-in; recorded as open item **O-1**.

**Routing.** I confirmed the router's effect end-to-end: on the demanding
`test_audit_lens_models_2026_07` chart, `apply_real_lens_maslov(E)` and
`apply_real_lens_maslov(E, n_v2=32)` are now **byte-identical**, because
`'auto'` resolves to `'stationary_phase'` and `n_v2` never enters.  That is the
mechanism behind ruling 2 (§3.2) — note it is the ROUTER, not "S2's local
quadrature made `n_v2 = 32` accurate too": `local_quadrature` is not on either
arm.

**S9 estimator.** `TV(T_k) = 2k` measured on a 400 001-point grid:
**2.000000 / 4.000000 / 6.000000 / 10.000000 / 16.000000** for k = 1/2/3/5/8.
On a random order-8 chart `_v2_oscillation_bound` returns **281.90**, which
bounds the directly sampled half-total-variation **8.07** and is **2.85×** the
pre-fix excursion sum **99.07** — consistent with the audit's measured 2.47×
under-count.  A chart with no `v2` dependence scores exactly 0.

**CuPy twin (desk-check, CuPy not installed).** `_integrate_local_quadrature_cupy`
takes its lattice, taper and both corrections from the SAME host-side
`_local_window_1d`, and calls `_local_window_geometry(xp=cp, …)`, which uses
only `arctan2 / cos / sin / maximum / sqrt / where / abs / pi` — all present in
CuPy.  `xp.where(lam1 >= 0.0, corr_plus, corr_minus)` mixes a CuPy array with
Python complex scalars, which CuPy broadcasts.  The `taper` is moved
host→device with `xp.asarray`.  The three CuPy `_van_vleck_density` sites take
the identical call as the NumPy ones.  I found no divergence.

### S3 — the exit-vertex leg — VERIFIED

`repro/orch/maslov_exit_vertex_check2.py` re-run on this checkout:

```
maslov-traced : tilt=-0.000 um  rho^2=-0.053 um  rho^4=+0.099 um  residual RMS=4.2 nm
thin-traced   : tilt=+0.000 um  rho^2=+0.000 um  rho^4=+0.007 um  residual RMS=0.1 nm
exit sag at rho=1: -31.255 um
```

— identical to the report's after-row.

**New fixture the WP did not use: an immersed rear.** Every WP-A4 and audit
fixture ends in air, where `n_exit = 1` and a wrong exit index is invisible.  On
an N-BK7 biconvex whose rear surface is immersed in N-SF11 (`n = 1.747969` at
1.31 µm, sag range 3.23 µm), `at_exit_vertex()` gives
`opd_vertex − (opd_surface + n_exit·t) = 0.0` exactly, while the `n_exit = 1`
mistake would be **2.413e-06 m = 1.84 waves**.  Rays land on `z = 0.0` exactly.

I did not get a mirror-rear case to trace (the reflective prescription needs a
`is_mirror` surface plus a sign convention I could not settle inside this pass);
recorded as open item **O-6**.

### S4 — Van Vleck density and the `k/(2πi)` prefactor — VERIFIED

`p12_prefactor.py` re-run reproduces the report's after-table to five
significant figures on all six `(λ, z)` rows:

| λ, z | `\|ratio\|` | `arg` |
|---|---|---|
| 1 µm, 0.5 mm | 0.999664 | +0.0002 |
| 1 µm, 1 mm | 0.999629 | +0.0008 |
| 1 µm, 2 mm | 0.998003 | −0.0017 |
| 2 µm, 0.5 mm | 0.999616 | +0.0024 |
| 2 µm, 1 mm | 0.997914 | −0.0009 |
| 2 µm, 2 mm | 0.996714 | −0.0020 |

The `ratio/(λz)` column is now 2.0e+09 … 2.5e+08, i.e. the pre-fix `i λ z`
signature is gone.

`_van_vleck_density(det_u, hx, hy) == sqrt(det_u/(hx·hy))` to **0 ULP** over a
4 × 3 ladder, including a strongly anamorphic `(0.5, 1e-3)`; and it is NOT the
pre-fix first power (3 decades apart on `det_u = 1e6`, `hx = hy = 0.02`).

**A chart with a REAL LENS, against an oracle written from scratch.**  The
WP verified S4 on a two-flat-surface (pure free-space) chart, where
`ds1/dv2 = −z·I` exactly.  I wrote the textbook ABCD / Collins Gaussian oracle
— reduced-angle system matrix `M = R(R2, n→1)·T(t, n)·R(R1, 1→n)` built from
the prescription by hand (thick-lens EFL 59.9917 mm), `q_out = (A q + B)/(C q + D)`
with `q_in = −i z_R` (the `exp(−iωt)`/`exp(+ikz)` convention), amplitude
`1/(A + B/q_in)` — and ran it against `normalize_output='none'` on an f/20
N-BK7 biconvex, λ = 1.31 µm, w0 = 0.35 mm, 256²:

| `output_plane_distance` | `\|E_maslov(0)\| / \|E_abcd(0)\|` | `w` measured / oracle [µm] | `\|E\|` profile relL2 |
|---|---|---|---|
| 5 mm | **0.99782** | 303.281 / 303.154 | 1.84e-03 |
| 20 mm | **0.99840** | 220.534 / 220.455 | 1.51e-03 |

across a 4× change in beam width.  A wrong Jacobian POWER would be off by
`λ·√\|det J\|`, i.e. decades — this is 0.2 %.  (At `d = 0`, the exit vertex
itself, the ratio is 1.175 and the profile 1.04e-01; the paraxial oracle is at
its weakest exactly there, and the exit vertex is also where the report itself
says the asymptotics do not belong.)  Total power at the exit vertex with
`normalize_output='none'` is **1.1678** of the input — O(1), as a correctly
normalised propagator must be.

**All four integrators on one scale, on the real chart** (ratio to
`quadrature`, `normalize_output='none'`):

```
quadrature        1.000000   arg -0.00000   spread 4.18e-17
levin             1.000526   arg +0.00014   spread 3.67e-04
stationary_phase  0.965065   arg -0.04836   spread 5.97e-02
local_quadrature  0.964442   arg -0.04932   spread 6.07e-02
```

so the Levin measure factor `sqrt(v2x_h·v2y_h)` — the quantity that cancels
identically in the pre-S4 algebra and therefore had no other witness — is
confirmed to **5.3e-04** on a real lens, not just on free space.  The two
asymptotic evaluators' 3.5 % is the leading-order truncation.

**Call-site census (grep).** Six `_van_vleck_density` calls: `_integrate_quadrature`
(:2994), `_integrate_stationary_phase` (:3293), `_integrate_local_quadrature`
(:3905) and the three CuPy twins (:3193, :4005, :4106) — the report's "7 sites"
counts `_integrate_levin` as one.  **Note (P3, open, O-2):** Levin does NOT use
the shared helper; the same algebra is written out by hand at two places
(`_vv_measure * np.sqrt(np.abs(det))` in `f` and in `_pairs_f`).  It is
arithmetically right — `sqrt(|det_u|)·sqrt(hx hy) ≡ _van_vleck_density·hx·hy` —
but it is exactly the duplication the shared helper's own comment says it exists
to prevent.

### S5 — the GBD Gouy / Collins conjugation — VERIFIED

I derived the convention rather than reading it.  `CONVENTIONS.md` §7 fixes
`exp(−iωt)` and `exp(+ikz)`.  Requiring `exp(i k ρ²/(2q)) = exp(−ρ²/w²)
exp(i k ρ²/(2R))` then forces `1/q = 1/R + iλ/(πw²)`, hence `q0 = −i z_R` and
`q(z) = z − i z_R`, and `q0/q = z_R(z_R − iz)/(z² + z_R²)` whose modulus is
`w0/w(z)` and whose argument is `−arctan(z/z_R)`.  With the module's
engineering `Q = 1/q_code = conj(1/q_phys)` and real `t`, the physics amplitude
ratio is `conj(Q_new/Q_old)` — the fix is correct, and so is the Collins
`conj(1/(A + BQ))` for real ABCD.

`p1_gbd_gouy.py` re-run: `arg(E/A) = 0.000000` at every z, and relative L2 with
no phase fit **1.410e-13 / 5.366e-13 / 4.204e-12** at z = 0.5 / 2 / 10 `z_R`,
matching the report.

Deprecated compensator API: `gbd_asm_gouy_phase` returns exactly `0.0`;
`gbd_field_to_asm` / `asm_field_to_gbd` return `E` unchanged (`array_equal`);
all three emit exactly one "deprecated" warning; `gbd_field_to_asm(np.ones(4))`
still raises `ValueError`, so an existing pipeline's failure modes are
unchanged.

### S6 — the saddle warning — NOT FIXED as shipped; fixed here

The WP gated the warning on `_na_meas`, the 3-σ second moment of
`|FFT(E_in)|²`, and documented `_SADDLE_FLAT_INPUT_NA = 1e-3` as "far below any
real divergence, far above the 1e-10…1e-12 floor a numerically flat wave leaves
in an FFT second moment, so a genuinely collimated input never trips it".

That is measurably false.  A *collimated* beam of finite width has a real
angular spectrum — a Gaussian of waist `w` spreads by `λ/(πw)`.  Measured
(λ = 1.31 µm, dx = 10 µm, flat-phase Gaussians):

| waist | FFT 3-σ NA | `λ/(πw)` | wavefront 3-σ NA |
|---|---|---|---|
| 0.25 mm | **3.54e-03** | 1.67e-03 | 0.0 |
| 0.80 mm | **1.10e-03** | 5.21e-04 | 0.0 |
| 2.0 mm | 4.2e-04 | 2.08e-04 | 0.0 |
| 4.0 mm | 2.1e-04 | 1.04e-04 | 0.0 |

so the warning fired on every collimated beam narrower than ~1 mm — and
end-to-end, `apply_real_lens_maslov(collimated, integration_method=
'stationary_phase')` printed `NA ~ 0.0035`.  A warning that cries wolf on the
common case is worse than none: it trains the caller to filter it.

**Fixed** by gating on the spread of the input's LOCAL WAVEVECTOR (new
`_wavefront_na`, built on the same conjugate-product estimator the FGA router
uses).  It is EXACTLY zero for a real, flat-phase field at any width — that is
an algebraic identity, not a tolerance — and reproduces the physical number on
inputs that really are non-flat: **6.00e-03 / 3.00e-02** for tilts of
0.002 / 0.01 rad (analytic `3t`) and **8.42e-02 / 3.37e-02** at f = −20 /
+50 mm, against **6.11e-03 / 3.00e-02** and **8.42e-02 / 3.37e-02** spectrally.
The message reports both numbers and says which one the gate uses.

End-to-end after the fix: tilted → warns; collimated → silent; `'quadrature'` →
never; `collimated_input=True` → silent.

**WP-A4's own S6 test had the same confusion, and I re-based it.**
`test_s6_asymptotic_methods_warn_on_a_non_collimated_input` used
`exp(−r²/(12 µm)²)` with the comment "A strongly diverging input: a 12 um
waist has NA ~ 0.027".  That is a Gaussian AT ITS WAIST — a PLANE wavefront.
For a real `E_in`, `arg E_in ≡ 0`, so
`grad_v2[arg E_in + k·OPD] = grad_v2[k·OPD]` identically and the OPD-only
saddle is the CORRECT expansion there; 0.027 is its angular spectrum, i.e.
diffraction.  The fixture is now the same envelope on a genuine diverging
wavefront (`exp(+iπr²/(λf))`, f = −0.5 mm, measured 3-σ wavefront NA 0.0357),
with the flat-phase case kept as an explicit counter-pin.

The deferred design (fit the input's local wavevector into the Newton gradient)
is unchanged and still the right eventual fix.

### S7 — `_lens_jax` x64 and jit — VERIFIED-WITH-NOTES

Re-checked on a prescription the WP did not use — a THREE-surface aspheric
crown/flint sandwich (`conic = −0.6` front, N-BK7 → N-SF11 → air):

```
x64 OFF: apply_real_lens_traced_jax -> RuntimeError (not complex64)
         apply_real_lens_maslov_jax -> RuntimeError
x64 ON : dtype complex128 on both; jax.jit compiles on the DEFAULT path;
         max |E_jit - E_eager| = 9.077e-13 on both entry points
```

The contract and the shared exit-vertex operator are right and the WP's five
pins pass.  The note is process, not physics: the refusal took
`validation/run_all.py test_lenses` red on three of its four JAX cases
(`RuntimeError: … requires double precision`), because only
`t_apply_real_lens_traced_jax_matches_numpy_opd` enabled x64.  Fixed by giving
the other three the same one-liner the repo already uses (§3.4).  The lane is
**46/46 passed** in 27.7 s, with `OPD diff RMS = 6.30 nm`,
`rel(JAX, FD) = 5.002e-07` and `maslov_jax − traced_jax max diff = 0.000e+00`.

### Y2 — the Van Vleck weight, and the oracles that stopped being oracles

The weight itself is right: `van_vleck_weight(det_J, λ) == −1j √det_J / λ` to
**0 ULP** over `det_J ∈ {1e-12, 3.98e-06, 1, 7.08e-06, 1e9}` ×
`λ ∈ {1.31 µm, 633 nm, 10.6 µm}`, derived in the test from the d = 2
Van Vleck–Morette kernel and the `s1 → v2` change of variable.

**The defect I fixed.**  WP-A4 changed four reference implementations to IMPORT
that helper:

* `tests/unit/test_niche_audit_w6_asymptotic.py::_quad_oracle` and `::_a9_quad`
  — the brute-force quadratures the W6-A3/A9 saddle pins score against;
* `tests/unit/test_audit_propagation.py` — the two inline `amp_lead` scalar
  references in `…ModalAsymptoticStillBitEqual`.

An oracle that reuses the library's own weight cannot detect a wrong weight.
That is not hypothetical here: it is precisely how audit Y2 went unnoticed —
every pre-fix pin passed with `|det J|` to the first power.  All four now
derive `−1j √|det J| / λ` locally (`_vv_weight_textbook` in the W6 file, written
out inline in the other), and a single new test
`test_w6_verify_van_vleck_weight_matches_the_textbook_fresnel_form` is the ONE
place the library helper and the textbook form are compared, so changing the
helper fails there instead of silently moving every oracle with it.

They still agree: **W6 53 passed** (52 + my new one),
**`test_audit_propagation` 101 passed**.

### Y3 / Y4 / Y5 — VERIFIED

* `sym2x2_max_eigenvalue` vs `numpy.linalg.eigvalsh`: worst relative departure
  **3.67e-16** over 207 matrices — exact degeneracy `(1, 0, 1)`, exact zero,
  `1e-14` and `1e12` scales, indefinite `(3, 4, −3)`, a `1e-15` gap, and 200
  random normals.  All three `lg00_sampling_waist_from_M` branches (lower clamp
  `1e-9`, upper clamp `1.0`, `λ_max ≤ 0 → 1e-6`) exact.
* Y4 drop diagnostic: on a 9×9 grid at 3× the fit half-box, **exactly one**
  `RuntimeWarning` naming `80/81 (98.8 %)` and the dominant reason; **0** bare
  NumPy `RuntimeWarning`s; **0** non-finite values.  On a clean in-box 7×7:
  0 warnings, 0 dropped pixels, 0 non-finite.  So it fires once, not per row,
  and has no false positive.
* Y5 `pupil_modes`: warns when `pupil_modes` names a mode `pupil_amplitudes`
  has no coefficient for, silent when they match.
* Y5 **defect found and fixed**: `AberrationTensorResult.van_vleck_weight`'s
  docstring said `L_legacy = L·|det J|/van_vleck_weight`, "i.e.
  `L·λ²·|det J|` in magnitude".  The first form is right; the restatement is
  the factor for `|L|²`.  In magnitude `|L_legacy| = |L|·λ·√|det J|`.
  Corrected in place.

### NEW — the `_solve_fit` conditioning gate — VERIFIED

I did not reuse the WP's lens fixture; I built the rank deficiency by hand
(`A[:, 7] = A[:, 0] + A[:, 1]` on a 60 × 8 random Gaussian matrix), which makes
the null space exactly known.

```
cond(A) = 1.06e+16   cond(A^T A) = 2.13e+302   rank(A) = 7 of 8
warned (RANK-DEFICIENT)  : True
|coef - lstsq_minnorm|   : 0.000e+00       (it IS the min-norm solve)
|coef(A) - coef(A + 1e-14 dA)| : 7.99e-15  (pre-fix: 0.869 WAVES)
residual                 : 2.16e-14
```

and on a well-conditioned system (`cond(G) = 2.39`) the gate returns a result
**bit-identical** (`np.array_equal`) to `cho_solve(cho_factor(G), AᵀRHS)`.

The thresholds are derived, not fitted: `_GRAM_COND_SINGULAR` is exactly
`1/eps = 4.503600e+15`, and `_GRAM_COND_MAX = 1e12` leaves
`eps·1e12 = 2.2e-04` relative accuracy in the squared system (~4 float64
digits) while sitting 3 decades above the audit's well-conditioned fixture
(`cond(G) = 1.24e9`) and 6 decades below the failing one (`6.18e18`).  Pinned
as such.

---

## 3. The five items implemented in this pass

### 3.1 Ruling 1 — the LG merit is not a Strehl deficit

**What I measured before changing anything.**  `1 − |L₀₀|²` on the stock
fixtures, with the post-Y2 scale:

| fixture | `\|L₀₀\|²` | `1 − \|L₀₀\|²` |
|---|---|---|
| R1 = 500 mm, ap 4 mm, w_s = 5 / 20 / 50 µm | 6.007e+14 / 4.765e+14 / 7.049e+13 | −6.01e+14 / −4.76e+14 / −7.05e+13 |
| R1 = 60 mm, ap 12 mm, w_s = 20 µm | 4.790e+14 | **−4.79e+14** |

`LGAberrationMerit(targets={(0,0): 1.0})` returned **−4.789609e+14**.  The WP's
diagnosis is correct and I reproduce it.

**What I then measured, and why the ruling's first option alone is not enough.**
Normalising by `L_ref` — the same coefficient on the aberration-free twin —
does make the quantity dimensionless and exactly 1.0 on an unaberrated optic.
But on the **closed-form point-sampling branch** (the only branch the pure
`(0, 0)` request takes, and the only one the JAX twin has) the normalised
coupling **RISES** with aberration, so `1 − coupling` is a negative number that
gets *more* negative as the design gets worse — the OPT-1 descent-direction
defect all over again.  Measured, scaling the fit's cubic-and-higher pupil
phase by α on an f/2.5 N-BK7 plano-convex singlet (`pupil_box_half = 0.18`):

| α | 0 | 0.25 | 0.5 | 1 | 2 | 4 |
|---|---|---|---|---|---|---|
| closed form | 1.000000 | 1.006273 | 1.012518 | 1.024914 | 1.049254 | **1.095510** |
| σ-grid overlap | 1.000000 | 0.902949 | 0.818530 | 0.680291 | 0.489330 | **0.286701** |

and on a physical ladder — the classic plano-convex orientation pair, same f/#:

| orientation | cubic+ pupil phase | closed form | σ-grid |
|---|---|---|---|
| curved toward the source | 25.799 waves | 1.024914 | **0.815601** |
| flat toward the source | 39.795 waves | 1.033099 | **0.731874** |

The reason is physical: a truncated leading-order saddle expansion does not
conserve energy, so `|A_lead|` is free to grow when the cubic terms move the
saddle and shrink `|det M|`.  **Only the σ-grid overlap behaves like a Strehl.**

**What I implemented.**

1. New public helper
   `lumenairy.propagators.asymptotic.aberration_free_reference_fit(fit)` — zeroes
   every `Phi` coefficient of total degree ≥ 3 in `(u3, u4)`, leaves the geometry
   (`coef_s1x`/`coef_s1y`, hence `|det J|`), the boxes, the extracted ramp and
   the wavelength untouched, and **returns the same object** when there is
   nothing to remove, so the ratio is exactly 1.0 bit-for-bit.
2. `LGAberrationMerit` divides every channel by `|L_ref(0,0)|²` from that twin,
   evaluated with the SAME `w_o` and at the twin's own chief-ray landing (a
   Strehl is referenced to the diffraction-limited PEAK; referencing at the
   requested `s2_image` instead divides the off-point falloff out of both sides
   and leaves the merit blind to image-point placement — measured
   `−2.84e-03 / −2.98e-03 / −3.77e-03 / −1.05e-02` at `dy = 0/20/50/100 µm`,
   i.e. falling as aberration grows).
3. New `strehl_branch` (default `'sigma'`) and `sigma_grid_n` (default 64).
   The σ ratio is insensitive to the grid — identical to 6 significant figures
   from `sigma_grid_n` 32 to 256, because numerator and reference alias
   together — so the cheap default is not an accuracy compromise:
   measured 1.4 s per field point against the closed form's 0.15 s, where the
   adaptive grid would cost ~11 s.
4. `strehl_branch='closed_form'` keeps the branch the JAX twin is restricted
   to, and WARNS with the measurement above.
5. `make_lg_aberration_merit_jax` gets the same normalisation (with `w_o = 1.0`
   on both calls so `N_o` cancels identically), and its docstring carries a
   `.. warning::` saying it is not a Strehl and pointing at the NumPy term.

**After.**

```
NumPy, default strehl_branch='sigma'   merit 1-S = -3.05e-03 / -2.84e-03 / -3.37e-03   (w_s = 5/20/50 um)
NumPy, strehl_branch='closed_form'     merit     = -3.21e-06 / -6.29e-04 / -3.02e-03
JAX twin                               merit     = -3.22e-06 / -6.29e-04 / -3.02e-03
weight linearity (sigma)               v(w=3)/v(w=1) = 3.000000000000
descent direction (sigma), dy=0/20/50/100 um  -2.84e-03 / +0.258 / +0.851 / +0.99994
non-(0,0) channels, now dimensionless  (2,0) 1.889e-01   (1,0) 4.353e-01
```

**A limit of the reference, found by running the shipped validation lane and
guarded.**  The aberration-free twin is a REFERENCE SPHERE only while the
pupil phase it removes is a perturbation.  On the validation suite's own
fixture — a 51.5 mm N-BK7 singlet at `object_distance = 200 mm`,
`pupil_box_half = 0.02` — the fit carries **4.206e+05 waves** of
cubic-and-higher pupil phase.  Zeroing that builds a DIFFERENT optic whose
focus is no longer at `s2_image`: measured `|L_ref(0,0)|² = 6.16e-10` against
`|L(0,0)|² = 0.799`, i.e. a coupling of **1.30e+09**.  The merit now says so
— one `RuntimeWarning` naming the removed waves and stating that the channels
are mutually consistent and monotone but NOT Strehl-normalised on that chart
— and is silent on a well-posed one (measured coupling 1.0029).  Trigger:
coupling `> 10`, i.e. 10× above the physical ceiling and 8 decades below the
measured failure.  `validation/run_all.py test_asymptotic` is **48/48 passed**
(62 s) with the merit's three end-to-end cases green.

**Residual honesty.**  On a near-diffraction-limited chart the σ ratio can
exceed 1 by a few 1e-3 — measured **1.002838** on a plano-convex R1 = 500 mm /
4 mm singlet carrying only **1.36e-03 waves** of cubic+ pupil phase, and stable
to 8 digits from `sigma_grid_n` 32 to 256, so it is the evaluator's own
non-conservation, not a grid artefact.  Documented in the merit's docstring.
`1 − coupling` is therefore ~−3e-03 at best focus rather than exactly 0.

**Tests re-pinned** (both live in optimize test files, as the coordinator
directed): `test_niche_audit_r_guards_and_merits.py` (4 R-5 tests, and its
`_lg_chan_sq` helper now computes the same dimensionless quantity from the
merit's own documented dependency) and `test_audit_optimize.py`'s C1 band.  The
R-5 cross-backend test now constructs the NumPy merit with
`strehl_branch='closed_form'`, because the JAX twin has no σ branch — that
keeps the pin's original purpose (the `x` vs `1 − x` inversion) fully intact —
and the tolerance is restated as `abs=1e-5` on the COUPLING, derived from the
measured `|nv − jv| = 3.3e-09 / 6.5e-07 / 3.1e-06`.  **109 passed** across the
two files, the same count as before.

**Migration note** — in `fixes/VERIFY_WP-A4_CHANGELOG.md` and summarised at the
end of `fixes/WP-A4_CHANGELOG.md`.

### 3.2 Ruling 2 — `test_a1_auto_n_v2_resolves_demanding_default_quadrature`

The fail-before arm (`il2(old32, truth) > 0.5`, 0.67 pre-S2) now measures
**1.325e-04**.  The mechanism is the S2 ROUTER, not the local-quadrature fix:
on this chart the router reports `need n_v2 ~ 2248` against a cap of 256 and
`'auto'` resolves to `'stationary_phase'`, so `n_v2` is inert and
`apply_real_lens_maslov(E)` is **byte-identical** to
`apply_real_lens_maslov(E, n_v2=32)`.  `local_quadrature` is on neither arm.

Restated — not deleted — as the property that now holds:

* both defaults land on the well-resolved `local_quadrature` reference the
  test already carried (`n = 8`, `window_sigma = 3.0`, which my S2 work shows
  is exact to 8.33e-15 on a quadratic chart): **il2 = 1.325e-04** for both,
  against the `5e-3` bar the test already used — 38× of headroom, 3 decades
  below the 0.67 the pre-S2 default scored;
* `'auto'` is no worse than the fixed default;
* and the two are BYTE-IDENTICAL, which is the mechanism.  A router change
  fails that assertion and forces a re-measurement instead of passing
  silently.

I first wrote the restated test against a converged uniform-quadrature oracle
and then took it back out: `n_v2 = 512` is 2.6e+05 v2 samples per pixel on the
128² fixture and the test took ~40 min.  The measurement stands as a dated
one-off in the test's docstring —

```
il2(n_v2 = 384, n_v2 = 512)                  2.14e-06   (the oracle is converged)
il2(default 'auto',             n_v2 = 512)  5.844e-04
il2(historical n_v2 = 32,       n_v2 = 512)  5.844e-04
il2(local_quadrature reference, n_v2 = 512)  5.754e-04
```

— which is exactly what TESTING_STANDARDS asks for an expensive oracle:
measure once with a date, pin the cheap property.  The restated test runs in
**1.64 s** (the version with the oracle: ~40 min).

### 3.3 `apply_real_lens_maslov_vector` — the S10 third sub-item (coordinator)

`normalize_output` was forwarded to the two scalar legs, which normalised
`E_x` and `E_y` independently, forcing the output polarization ratio back to
the post-Fresnel input ratio.  Both legs now run at `normalize_output='none'`
and ONE scale is applied to the pair, as `apply_real_lens_fga_vector` does.
Measured on an f/3.3 N-BK7 biconvex with a `P_x/P_y = 1.777777777778` linear
input:

```
none    P_x/P_y = 1.777777663332023   P / P_postFresnel = 28.461186
power   P_x/P_y = 1.777777663332023   P / P_postFresnel =  1.000000000000000   (0 ULP from 'none')
peak    P_x/P_y = 1.777777663332023   P / P_postFresnel =  0.627973            (0 ULP from 'none')
departure from the INPUT ratio = -6.4376e-08 = the s/p diattenuation = 5.154e+08 ULP
```

so the pre-fix behaviour (ratio ≡ input ratio) is distinguishable by 8 decades.
An unknown `normalize_output` now raises with the §2 prefix.

**The reference power is the POST-FRESNEL pair, not the raw input.**  My first
version used the raw input, matching `apply_real_lens_fga_vector`'s lossless
convention — and that turned `tests/unit/test_v5_21_maslov_jax_caustic.py::
test_maslov_vector_polarization` red, because it pins that the vector wrapper's
output power carries the two-surface Fresnel transmission `T1*T2` relative to
the scalar propagator's.  That is a real physical property and the joint scale
should not delete it, so the reference is the pair the two scalar legs are
actually handed: the literal joint form of the scalar `'power'` contract.
`T1*T2 = 0.9219035` on my fixture, and the pinned test passes unchanged.  The
deliberate difference from the FGA peer is documented in the wrapper's Notes.

I also took the first sub-item as far as is safe: the polarization base rays are
launched along the input field's own local wavevector instead of axially (new
`_local_direction_cosines` / `_input_direction_cosines`, the same estimator FGA
uses).  A real, non-negative input gives EXACTLY `(0, 0)`, so a collimated call
is bit-identical to the pre-v5.46 behaviour; a 0.03 rad tilt is recovered as
`ux = 0.030000`, and an f = 20 mm converging wavefront as `−x/f` to
`4.0e-04 = dx/(2f)` — exactly the forward-difference bias.  The third sub-item
(`E_z` and exit-frame transport) is NOT done; the docstring's
"polarization-resolved study through a focus" motivation is retracted and the
gap is stated in the Notes.

### 3.4 The red `validation/run_all.py test_lenses` lane (coordinator)

Three of the four JAX cases in `validation/elements/test_lenses.py` did not
enable x64, so S7's refusal took them red.  Each now calls
`jax.config.update('jax_enable_x64', True)` — the one-liner the fourth case and
the other JAX validation cases already use, and the pattern the coordinator
pointed at.  The refusal itself stays pinned in
`test_audit2609_a4_maslov_gbd.py::test_s7_*`, so nothing about the contract is
weakened.

```
RESULT [lenses]: 46/46 passed (0 failed)
  [OK] apply_real_lens_traced_jax: runs and produces finite output
  [OK] apply_real_lens_traced_jax: OPD matches NumPy within 10 nm RMS  -- 6.30 nm
  [OK] jax.grad through apply_real_lens_traced_jax matches FD  -- rel 5.002e-07
  [OK] apply_real_lens_maslov_jax matches traced for non-caustic geometry  -- 0.000e+00
  [PASS] test_lenses.py (27.7s)
```

### 3.5 WP-A15a §5 item 8 — three timing assertions and one slow marker

WP-A15a's test/CI pass tabulated three wall-clock assertions and one
over-budget file in files I own this round.  All four are done.

**`tests/unit/test_audit_propagation.py:2678` — `assert t_fused < 60.0`.**
Its own comment calls it "a blow-up net, not a speed comparison": it stood in
for an accidental O(N²), a dropped chunking or a per-pixel Python loop.  The
test already counted `np.exp` / `np.einsum` / `np.sum` per chunk (1 / 1 / 0
fused, 2 for the reference, and 2 / 2 when forced into two chunks), which
covers the chunking and the fusion exactly.  What no count covered was the
GRID scaling, so the ceiling is replaced by one: `_fused` now takes `Ny`/`Nx`
and the counts on a **doubled grid must be identical** to those on the
original.  A per-pixel loop or a lost vectorisation fails that exactly, on any
machine, in no time at all.  The timing is still printed.

**`tests/unit/test_audit_propagation.py:3787` — `assert speedup >= 5.0`**
(`test_decompose_lg_cache_speedup`; its own docstring conceded it was "a soft
speedup floor … timing on CI hosts is noisy").  Replaced by
`test_decompose_lg_cache_builds_no_modes_on_a_hit`, which wraps
`asymptotic_modes._evaluate_poly2d`: a COLD `decompose_lg` evaluates exactly
`(p_max + 1)(2 ell_max + 1) = 28` polynomials, a CACHED one **0**, and
`clear_lg_mode_stack_cache()` puts it back to 28.  Plus the identity contract
— `_lg_mode_conj_stack` returns the SAME array object on a hit, which is the
v5.30 W6-A13 read-only invariant.  All four numbers are exact; no tolerance.

**`tests/unit/test_audit_optimize.py:905` — `assert dt < 180.0`.**  A
hang-catcher whose property is already measured, exactly, two assertions
above it: a cancelled `max_iter=200` run costs FEWER merit evaluations than an
uncancelled `max_iter=2` one, and at most `_CANCELLED_EVAL_BUDGET`.  A run
that satisfies both cannot be "stuck" in the sense the ceiling meant — it
would have to be stuck inside a single merit evaluation, which is another
test's subject.  Retired; the wall clock is printed for triage.

**`tests/unit/test_audit_lens_models_2026_07.py` — `pytestmark = pytest.mark.slow`.**
641.0 s on the committed `.test_durations`, five times the 2 min/file bar.
Verified the marker takes: `-m "not slow"` now deselects all 69 ids.  Its
runtime on this box after ruling 2's restatement is **206.7 s** (62 passed,
7 skipped) — still over the bar, and down from the ~40 min the first version
of that restatement would have cost.

`lumenairy/__init__.py`'s eager re-export of `aberration_free_reference_fit`
is safe: the name is final and is not renamed by this pass.

---

## 4. Files touched and tests run

**Source changed by me** (all inside the WP's ownership list, plus the two
files the orchestrator granted and the validation case):

* `lumenairy/elements/lenses_maslov.py` — the S6 wavefront-NA gate
  (`_wavefront_na`, `_local_direction_cosines`), the vector wrapper's joint
  normalisation and launch directions (`_input_direction_cosines`), the
  docstring retraction.
* `lumenairy/propagators/asymptotic_canonical_fit.py` — new
  `aberration_free_reference_fit`.
* `lumenairy/propagators/asymptotic.py` — re-export + `__all__`.
* `lumenairy/propagators/asymptotic_aberration_tensor.py` — the
  `van_vleck_weight` round-trip docstring correction.
* `lumenairy/optimize/merit_terms.py` — `LGAberrationMerit` (ruling 1).
* `lumenairy/optimize/jax_merits.py` — `make_lg_aberration_merit_jax` (ruling 1).
* `validation/elements/test_lenses.py` — x64 in three JAX cases.

**Two of WP-A4's own tests turned red under my source fixes, and both were
the test rather than the fix** (recorded here because a verifier changing a
WP's tests needs to be explicit):

1. `test_audit2609_a4_maslov_gbd.py::test_s6_asymptotic_methods_warn_on_a_non_collimated_input`
   — its "strongly diverging input" is a flat-phase 12 µm Gaussian, i.e. a
   plane wavefront.  Re-based on a genuine diverging wavefront, with the
   flat-phase case added as a counter-pin.  (§2, S6.)
2. `test_v5_21_maslov_jax_caustic.py::test_maslov_vector_polarization` — it
   pins that the vector wrapper's output power carries `T1*T2` relative to the
   scalar propagator.  My first joint-scale version normalised to the RAW
   input (the FGA convention) and deleted that.  The SOURCE was changed, not
   the test: the reference is now the post-Fresnel pair, and the test passes
   unchanged.  (§3.3.)

**Tests**

* NEW `tests/unit/test_audit2609_a4_verify_maslov_asymptotic.py` — **48 tests**
  (S2 ×27 incl. the box-boundary and total-variation pins, S3 ×1, S4 ×1,
  S5 ×1, S6 ×6, S10 vector ×3, Y2 ×1, Y3 ×2, `_solve_fit` ×3, ruling 1 ×3).
* `tests/unit/test_niche_audit_w6_asymptotic.py` — `_vv_weight_textbook` added,
  both oracles re-pointed at it, one new comparison test.
* `tests/unit/test_audit_propagation.py` — the two `amp_lead` references
  written out locally.
* `tests/unit/test_niche_audit_r_guards_and_merits.py` — 4 R-5 tests and
  `_lg_chan_sq` / `_lg_merit` re-pinned to the dimensionless coupling.
* `tests/unit/test_audit_optimize.py` — the C1 non-vacuity band re-pinned.
* `tests/unit/test_audit_lens_models_2026_07.py` — ruling 2, and
  `pytestmark = pytest.mark.slow` (WP-A15a).
* `tests/unit/test_audit2609_a4_maslov_gbd.py` — the S6 fixture re-based (see
  above).
* `tests/unit/test_audit_propagation.py` / `test_audit_optimize.py` — the
  three WP-A15a wall-clock conversions (§3.5).

No tolerance was loosened anywhere; every changed bar is tighter, exact, or
restated on a different (and stated) property.

| command (all `OPENBLAS_NUM_THREADS=1`, `-q --no-header -p no:cacheprovider`) | result | duration |
|---|---|---|
| `test_audit2609_a4_maslov_gbd.py test_audit2609_a4_asymptotic.py` (WP's own, before my edits) | **41 passed** | 44 s |
| `test_audit_propagation.py test_niche_audit_w6_asymptotic.py` (before my oracle change) | **153 passed** | 188 s |
| `test_niche_audit_w6_asymptotic.py` (after the oracle change) | **53 passed** | 179 s |
| `test_audit2609_a4_verify_maslov_asymptotic.py` (NEW) | **48 passed** | 56 s |
| `test_niche_audit_r_guards_and_merits.py test_audit_optimize.py` | **109 passed** | 67 s |
| `test_audit2609_a4_maslov_gbd.py test_audit2609_a4_asymptotic.py test_niche_audit_eh1_maslov_upsample.py test_v5_21_maslov_jax_caustic.py test_v5_21_gbd_asm_interop.py test_audit_w5_propagators.py test_v4_16_1_dispatcher_pin_cache_registry_enrollment.py test_v4_16_0_agent_d_cache_registry.py` (after all my source fixes) | **114 passed** | 477 s |
| `test_niche_audit_w3_oracles.py test_v4_16_1_agent_d.py test_v5_1_0_agent_e_split.py test_v5_21_2_subsystem_audits.py` (every other LG-merit consumer) | **287 passed, 1 skipped, 5 failed** — 4 of the 5 are WP-A4's, pre-existing at `32ba3ba2`; see below | 104 s |
| `test_audit_propagation.py test_audit_optimize.py test_audit_lens_models_2026_07.py -m ""` (after the WP-A15a conversions and the slow marker) | **252 passed, 7 skipped** | 219 s |
| `test_audit_lens_models_2026_07.py --collect-only -m "not slow"` | **69 deselected** — the marker takes | 0.1 s |
| `python validation/run_all.py test_lenses` | **46/46 passed** | 28 s |
| `python validation/run_all.py test_asymptotic` | **48/48 passed** | 62 s |
| `python -m ruff check` on all 13 files I touched | clean; the only 40 findings are `validation/elements/test_lenses.py`'s PRE-EXISTING style (identical count on the file at `32ba3ba2`) | — |

**Five failures in the LG-merit consumers — four are WP-A4's, not mine.**
They are all the Y2 SCALE move (`|L|²` × `1/(λ²|det J|)`) reaching test files
WP-A4's report does not list:

1. `test_niche_audit_w3_oracles.py::test_w3_t3b_pure_lg00_default_is_bit_for_bit_unchanged`
   — pins `L = 15.448 + 3.188j`; measures `1.231e+07 − 5.965e+07j`.  This is
   the RAW `aberration_tensor` value, which I did not change numerically.
2. `…::test_w4_t1_explicit_sigma_grid_n_64_is_the_pre_fix_default_bit_for_bit`
   — pins `9.0969e-14`, measures `1.3565e+00`, which is exactly the
   `|L_{(2,0)}|²` I measured straight out of `aberration_tensor`.
3. `…::test_w4_t1_pure_lg00_has_no_sigma_grid_and_is_unchanged` — same class.
4. `test_v5_21_2_subsystem_audits.py::test_opt1_lg_jax_merit_is_strehl_deficit_not_amplitude`
   — expects the JAX merit in `(0.5, 1.0]`.  At `32ba3ba2` it was
   `−4.79e+14`; with my normalisation it is `−3.02e-03`, i.e. much closer to
   sane but still failing a bar written for the v5.45 scale.
5. `…::test_w3_t3b_lg_merit_responds_to_a_curvature_change` — pins the merit
   value `8.833e-14`; this one is both the Y2 move AND my normalisation.

I did NOT edit those two files: they are outside my ownership list and
`test_niche_audit_w3_oracles.py` belongs to the W3/W4 oracle corpus.  The exact
changes needed are in §5 (**O-9**).

I did NOT run `validation/run_all.py test_gbd`: the box carried up to nine
other python processes throughout the pass and COMMON.md §5 asks for economy.
`test_lenses` and `test_asymptotic` — the two lanes my changes actually reach —
are both green, and `test_asymptotic` turned out to be 62 s, not the multi-hour
job the WP estimated.

---

## 5. Open items for the orchestrator

| # | severity | item |
|---|---|---|
| **O-1** | *(closed in the Follow-up)* | `local_quadrature` WAS SILENT when its tapered lattice leaves the fitted chart box, where it is no longer exact: measured relerr **8.79e-02** at `window_sigma = 7.5` and **8.17e-01** at the shipped defaults on a chart with a small Hessian eigenvalue.  Fix: count the dropped samples per pixel in `_integrate_local_quadrature` and warn once with the fraction, exactly as Y4's `_warn_dropped_pixels` does.  ~1 hour.  The module docstring's "exact … at ANY `local_n_samples` / `local_window_sigma`" should gain the "while the lattice fits inside the box" qualifier at the same time. |
| **O-2** | *(closed in the Follow-up)* | `_integrate_levin` wrote the Van Vleck algebra out by hand in two places instead of calling `_van_vleck_density` — the duplication the helper's own comment says it exists to prevent.  One-line change each, plus the measure factor. |
| **O-3** | *(closed in the Follow-up)* | `LGAberrationMerit`'s new default cost two `aberration_tensor` calls per field point (measured 1.4 s vs 0.15 s for the pre-fix single closed-form call).  If that is too slow for a production design loop, the reference is a pure function of `(fit, s2_img, src, w_s, w_p, w_o, branch, grid)` and can be cached on `ctx._canonical_fit_cache` alongside the fit — the merit already has that hook. |
| **O-3b** | P2 (deferred to Wave 4 by the coordinator) | The aberration-free reference is only a reference SPHERE while the pupil phase it removes is a perturbation.  On a chart with a finite `object_distance` and a wide `pupil_box_half` it can remove 4.2e+05 waves and collapse (measured coupling 1.30e+09).  The merit now warns; the real fix is to re-expand `Phi` about the saddle and truncate THERE, so the reference is a perturbation by construction.  ~half a day plus a bit-identity harness. |
| **O-4** | P3 (documented, no clamp, as directed) | On a near-diffraction-limited chart the σ-branch Strehl exceeds 1 by a few 1e-3 (measured 1.002838 on a chart with 1.36e-03 waves of cubic+ phase, grid-independent).  It is the leading-order evaluator's own non-conservation.  A caller who needs a hard `[0, 1]` should clamp; I did not, because silently clamping would hide exactly this. |
| **O-5** | *(closed in the Follow-up — the JAX twin GOT the sigma branch)* | `make_lg_aberration_merit_jax` was restricted to the closed-form branch, which is not a descent direction even normalised.  Either give the JAX twin a σ branch (it needs `decompose_lg` on a jnp grid — the pieces exist) or make the JAX merit refuse `targets={(0,0): …}` and point at the NumPy term.  Doing the latter now would break the R-5 cross-backend pin, so it is a coordinated change. |
| **O-6** | *(closed in the Follow-up)* | Two checks I had not got to: an exit-vertex case with a MIRROR last surface (I could not settle the reflected-`N` sign convention inside this pass), and Y1's chief ray at a SECOND field angle.  Y1's own repro and the whole W6 suite pass, so this is coverage, not a suspicion. |
| **O-7** | — | *(closed)* Everything ran.  `test_audit_lens_models_2026_07.py` (ruling 2) is 62 passed / 7 skipped in 206.7 s; the three-file A15a re-run is 252 passed / 7 skipped. |
| **O-8** | *(closed in the Follow-up)* | WP-A4's §5 item 2: `clear_maslov_local_window_cache` still needs re-exporting from `lumenairy/__init__.py` and adding to `lenses_maslov.__all__` in the same commit. |
| **O-9** | *(closed in the Follow-up)* | **Four tests were red at `32ba3ba2` from WP-A4's own Y2 scale move, in files its report does not list.**  `tests/unit/test_niche_audit_w3_oracles.py` × 3 (hard-pinned `L` / `|L|²` values: `15.448+3.188j` → `1.231e+07−5.965e+07j`; `9.0969e-14` → `1.3565e+00`) and `tests/unit/test_v5_21_2_subsystem_audits.py::test_opt1_lg_jax_merit_is_strehl_deficit_not_amplitude` (expects the JAX merit in `(0.5, 1.0]`; `−4.79e+14` at `32ba3ba2`, `−3.02e-03` now).  A fifth, `test_w3_t3b_lg_merit_responds_to_a_curvature_change` (pins `8.833e-14`), is the Y2 move plus my normalisation.  Each needs its pinned constant re-derived exactly as WP-A4 did for `test_audit_optimize` / `test_niche_audit_r_guards_and_merits` — the two raw-`aberration_tensor` ones by the `1/(λ²|det J|)` factor, the two merit ones against the new dimensionless coupling.  Outside my ownership, so untouched. |

## 6. Changelog

`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/VERIFY_WP-A4_CHANGELOG.md`,
with a pointer block and the migration note appended to
`fixes/WP-A4_CHANGELOG.md`.


---

# Follow-up (coordinator rulings on §5, same session)

Granted for this pass in addition to the original ownership:
`tests/unit/test_niche_audit_w3_oracles.py`,
`tests/unit/test_v5_21_2_subsystem_audits.py`, and `lumenairy/__init__.py`
for the O-8 export.  No git writes.

| item | verdict | measured |
|---|---|---|
| **O-9** re-pin the five red tests | **done** | the closed-form constants carried through `L_new = L_old·(−1j)/(λ√\|det J\|)`, verified to **3.6e-14 / 1.8e-13** relative; the σ anchors through `1/(λ²\|det J\|)` to **1.1e-04 / 4.4e-04**; the two merit pins re-measured.  `test_niche_audit_w3_oracles.py` **181 passed**, the OPT-1 test **1 passed** |
| **O-1** dropped-sample diagnostic | **done** | one `RuntimeWarning` naming the fraction; fires at **36.4 %** (ws = 5.0) and **46.2 %** (ws = 7.5) dropped, silent at ws = 3.0 / 2.5 where nothing is dropped; docstrings qualified |
| **O-2** Levin → `_van_vleck_density` | **done** | **bit-identical** (`np.array_equal` over 1e5 samples × 3 anamorphic half-width pairs) |
| **O-6** mirror exit vertex + Y1 second field angle | **done** | mirror: `n_exit = n(glass_before) = 1.503583` applied to **0.0**, the "air" mistake **7.783e-06 m = 5.9 waves**; Y1 at three field angles, PSF within **1.20 / 1.80 / 1.21 µm** of the traced chief ray |
| **O-3** cache the reference | **done** | 3 terms on one fit: **4.365 s → 2.807 s** (1.455 → **0.936 s/term**); exactly **one** `lg_ref` cache entry serves all three |
| **O-5** JAX σ branch | **done — the PREFERRED branch, not the fallback** | JAX σ coupling **1.002678** vs NumPy σ **1.002871**, i.e. **1.92e-04**; grid-independent to **3e-09** (n = 16 → 32); `jax.grad` finite (`d(1−S)/dw_s = −3.10e+01`); R-5 now compares σ to σ |
| **O-8** export the cache clearer | **done** | `clear_maslov_local_window_cache` in `lenses_maslov.__all__` and re-exported from `lumenairy/__init__.py` |
| **O-3b**, **O-4** | deferred / documented as directed | — |

## F.1 O-9 — the five red tests

The whole move is one identity, written once beside the constants:

    L_new = L_old · (−1j) / (λ · √|det J|)        (every L)
    |L|²_new = |L|²_old / (λ² · |det J|)             (every |L|²)

with `|det J|` read off the result object (`van_vleck_weight` is
`−1j√|det J|/λ`, so `|det J| = (|w|λ)²`) — a quantity the pre-Y2 code
already had, just at the wrong power.  Measured at λ = 1.31 µm:

| design | `√\|det J\|` | `\|det J\|` | `1/(λ²\|det J\|)` |
|---|---|---|---|
| R1 = 51.5 mm | 1.976954e-01 | 3.908349e-02 | 1.490953e+13 |
| R1 = 60.0 mm | 1.982878e-01 | 3.931805e-02 | 1.482059e+13 |

1. **`test_w3_t3b_pure_lg00_default_is_bit_for_bit_unchanged`** — constants
   `15.448+3.188j` / `−6.571−14.392j` →
   `1.2310011487876e+07−5.9649449469122e+07j` /
   `−5.5405030010922e+07+2.5295326982543e+07j`.  The rel-1e-8 bar is
   unchanged, and the test now ALSO asserts that the new constant is the old
   one times the factor (measured **3.6e-14 / 1.8e-13**), so the re-pin is a
   derivation rather than a re-bake.
2. **`test_w4_t1_explicit_sigma_grid_n_64_is_the_pre_fix_default_bit_for_bit`**
   and 3. **`test_w4_t1_pure_lg00_has_no_sigma_grid_and_is_unchanged`** —
   `_W4T1_FROZEN_64` moves `9.0968975e-14 / 7.1975598e-14` →
   `1.3564605658e+00 / 1.0671876291e+00`; measured ratios
   **1.491124e+13 / 1.482708e+13** against the predicted
   1.490953e+13 / 1.482059e+13, i.e. **1.1e-04 / 4.4e-04** — the residual is
   `det J`'s variation across the σ grid, which a single-number factor
   cannot carry, and the claims reading these anchors are factor-of-two
   bands, so that is three decades of margin.  The pre-Y2 anchors are kept
   beside them as `_W4T1_FROZEN_64_PRE_Y2`.
4. **`test_w3_t3b_lg_merit_responds_to_a_curvature_change`** — re-measured
   against the σ coupling: `val_a` **2.2006679213e+09**, `val_b`
   **9.9881936194e+04**, response **9.999546e-01** (was 4.1232e-01).  The
   2e-2 relative tolerance is UNCHANGED.  This fixture is the one that trips
   the new collapse guard (4.2e+05 waves of cubic+ pupil phase at a finite
   `object_distance`), so the docstring now says plainly that the channels
   are mutually consistent and monotone here but NOT Strehl-normalised, and
   points at O-3b.  The response band is restated `0.25 < rel < 1.0` — the
   upper edge is structural (both values are positive).
5. **`test_opt1_lg_jax_merit_is_strehl_deficit_not_amplitude`** — its
   MECHANISM is gone, not its claim.  It read the merit at a grossly
   waist-mismatched source, where the raw `|L|²` underflowed; the merit is
   now referenced to the aberration-free twin, and a waist mismatch is not
   an aberration — it hits the reference identically and cancels (measured
   −3.190e-03, which the old `0.5 < v <= 1` bar would read as the defect).
   Driven instead by a real aberration, the image point walking off the
   chief ray:

   | dy | 0 | 20 µm | 50 µm | 100 µm |
   |---|---|---|---|---|
   | JAX (0,0) merit | −3.19e-03 | +1.70e-01 | +6.94e-01 | +9.91e-01 |

   The test now pins strict monotonicity (smallest step 0.17), `v(100 µm) >
   0.5` and `|v(0)| < 0.05`.  The pre-OPT-1 `|Strehl|²` form is the same
   ladder with the sign flipped and fails all three.

## F.2 O-1 — the local-quadrature truncation is now audible

`_warn_local_window_truncation` counts the dropped samples over the live
pixels and warns ONCE above a **1 %** fraction (derived: with the taper at
`exp(−4.5)` on the lattice edge, 1 % of the samples is ~1e-3 of the summed
weight — three decades above the 1e-15 the scheme reaches when nothing is
dropped and three below the 8e-02 measured at `window_sigma = 5`).  Both the
NumPy integrator and the CuPy twin call it.  Measured on the anamorphic
chart:

```
n= 8 ws=3.0  reach 0.757  relerr 6.60e-15  dropped   0/  64  warn False
n=33 ws=2.5  reach 0.631  relerr 3.41e-15  dropped   0/1089  warn False
n=11 ws=5.0  reach 1.262  relerr 8.09e-02  dropped  44/ 121  warn True  (36.4 %)
n=13 ws=7.5  reach 1.893  relerr 8.79e-02  dropped  78/ 169  warn True  (46.2 %)
```

The module comment and the function docstring now carry the qualifier
"while the tapered lattice fits inside the fitted chart box".

## F.3 O-2 — Levin through the shared helper

Both `_integrate_levin` integrand closures call `_van_vleck_density` now.
The composition is chosen so it is **bit-identical**:
`_van_vleck_density(d, 1, 1)` is `d ** 0.5`, NumPy's `** 0.5` IS `sqrt`
bit-for-bit (verified over 1e5 samples), and the box Jacobian
`sqrt(hx·hy)` is applied exactly as before.  Unit half-widths are the right
call here because the Levin engine works in the normalised unit box end to
end — the other association
(`_van_vleck_density(d, hx, hy) · hx · hy`) is the same number to 1–2 ULP
(measured 2.7e-16 / 3.3e-16 / 3.0e-16) but re-associates the products and
would move the returned field off bit-identity for nothing.

## F.4 O-6 — the two checks I had not got to

**Mirror last surface.**  `raytrace/exit_vertex.py` documents the
convention: `n_exit` is a PHYSICAL (positive) index even for a mirror, and
for a mirror it is `glass_before` — the reflected ray travels back through
the medium it arrived in; the Welford `n' = −n` bookkeeping lives in the
trace and the sign returns through `N < 0` in `t = −z/N`.  On an N-BK7
front surface with an internally reflecting R = −25 mm back surface
(measured `N = −0.998741`, sag 15.4 µm): the vertex OPD equals
`opd_surface + n(N‑BK7)·t` **exactly (0.0)**, and the "it must be air"
mistake would be **7.783e-06 m = 5.9 waves** at 1.31 µm.

**Y1 at a second and third field angle.**  WP-A4 pinned the x-offset source,
where the rank-deficient design puts the entire ramp in `a3` and leaves
`a4` at 2.6e-10 — so the `a4·u4` half of the Y1 fix was never exercised.
Measured (grid pitch 92.3 µm), against an independent ray trace of the
chief ray from the same prescription:

| source | `\|a3\|` | `\|a4\|` | traced chief | PSF peak | miss |
|---|---|---|---|---|---|
| (100, 0) µm | 2.740e+03 | 2.6e-10 | (9.665156e-05, 0) | (9.785275e-05, 0) | 1.20 µm |
| (0, 150) µm | 8.5e-10 | 3.998e+03 | (0, 1.449788e-04) | (0, 1.467808e-04) | 1.80 µm |
| (−70, 70) µm | 1.893e+03 | 1.893e+03 | (−6.765608e-05, +same) | (−6.851335e-05, +same) | 1.21 µm |

so the `a4` term is now covered on its own and jointly.  Bar 20 µm: 11×
above the worst measured miss, one fifth of a grid pitch, 35× below the
~700 µm the pre-Y1 default flag produced.

## F.5 O-3 — the reference is cached per (fit, field point)

`LGAberrationMerit` now takes the aberration-free reference from
`ctx._canonical_fit_cache`, keyed by
`('lg_ref', <the fit's own cache key>, source_point, w_s, w_p, w_o,
strehl_branch, sigma_grid_n)`.  The reference request itself is pinned to a
FIXED minimal mode set (`[(0,0), (1,0)]` on the σ branch) rather than the
term's own `output_modes`: only its `(0, 0)` entry is ever read, and pinning
it makes every channel of every term on the same optic divide by the SAME
constant — which is what lets a `CompositeMerit` compare its channels at all
— as well as letting one cached reference serve every term.  The value is
unchanged by that (measured composite 1.829191e-01 before and after).

Measured, three `LGAberrationMerit` terms on one fit (medians of 3):

```
3 terms, ONE context (reference cached):   2.807 s  ->  0.936 s/term
3 terms, separate contexts (no sharing):   4.365 s  ->  1.455 s/term
1 term alone (numerator + reference):      1.520 s
```

and the cache holds exactly **one** `lg_ref` entry after a composite
evaluation.  No timing is asserted anywhere (TESTING_STANDARDS S1).

## F.6 O-5 — the JAX twin got the σ branch (the preferred option)

I took the preferred branch, not the fallback, because the merit only ever
uses the RATIO `|L|²/|L_ref|²` — and the basis waist, the grid extent and
`n_grid` are identical on both sides of it, so they cancel.  That removes
the two pieces of the NumPy σ branch with no cheap JAX twin (the iterative
`_measure_image_plane_waist` probe and the adaptive `sigma_grid_n` ladder)
and leaves an overlap that is a dozen lines of `jnp`: a σ grid,
`jax.vmap` over `solve_envelope_stationary_jax_ift` +
`_modal_field_lg00_pixel_jax`, and an analytic LG₀₀ projection.  New
`_lg00_sigma_overlap_jax` in `optimize/jax_merits.py`, plus
`strehl_branch` (default `'sigma'`) and `sigma_grid_n` (default 24) on
`make_lg_aberration_merit_jax`, mirroring the NumPy merit;
`strehl_branch='closed_form'` keeps the old path and warns.

Measured on an f/2.5 N-BK7 plano-convex singlet (w_s = 20 µm, w_p = 0.05,
on-axis):

```
NumPy sigma coupling (n=64):          1.002870569
JAX   sigma coupling n=16 w_frac=.25: 1.002677530   rel vs NumPy 1.925e-04
JAX   sigma coupling n=24 w_frac=.25: 1.002677732   rel vs NumPy 1.923e-04
JAX   sigma coupling n=32 w_frac=.25: 1.002677729   rel vs NumPy 1.923e-04
JAX   sigma coupling n=24 w_frac=.50: 1.003308587   rel vs NumPy 4.368e-04
jax.grad d(1-S)/dw_s = -3.101741e+01   (finite)
```

The ratio is grid-independent to **3e-09** (n = 16 → 32) and moves by
6.3e-04 when the basis-waist convention moves — which is how you can tell
the 1.9e-04 residual against NumPy is the convention (NumPy measures the
image-plane waist; the JAX branch uses `0.25·fit.s2x_halfrange`), not the
physics.  End to end through the merit, NumPy σ vs JAX σ:

| w_s | 5 µm | 20 µm | 50 µm |
|---|---|---|---|
| Δcoupling | 6.97e-04 | 1.80e-04 | 1.75e-04 |

**R-5 now compares σ to σ** (`branch='sigma'` on the NumPy side), gated at
`abs=2e-03` on the coupling — 2.9× above the worst measured value and
2.5 decades below the ~1.0 an `x` vs `1−x` confusion would produce.  Cost:
11.7 s first call (XLA compile), **1.4 s** warm.

## F.7 O-8 — the cache clearer is exported

`clear_maslov_local_window_cache` added to `lenses_maslov.__all__` and
re-exported from `lumenairy/__init__.py` (eager, beside the other two
`lenses_maslov` names, with a why-comment naming the walker test that
requires it) and added to the top-level `__all__`.
`lumenairy/__init__.py`'s eager re-export of `aberration_free_reference_fit`
is untouched — the name is final.

## F.8 Files touched in the follow-up

**Source**

* `lumenairy/elements/lenses_maslov.py` — O-1 (`_warn_local_window_truncation`,
  `_LOCAL_WINDOW_DROP_WARN_FRAC`, both integrators, two docstrings), O-2
  (both Levin closures), O-8 (`__all__`).
* `lumenairy/optimize/merit_terms.py` — O-3 (`_reference_tensor` + the fixed
  reference mode set).
* `lumenairy/optimize/jax_merits.py` — O-5 (`_lg00_sigma_overlap_jax`,
  `strehl_branch`, `sigma_grid_n`, docstring).
* `lumenairy/__init__.py` — O-8 (import + `__all__`).

**Tests**

* `tests/unit/test_audit2609_a4_verify_maslov_asymptotic.py` — 6 new tests
  (O-1 ×1, O-2 ×1, O-6 ×4).
* `tests/unit/test_niche_audit_w3_oracles.py` — O-9 pins 1–4 (the derivation
  block `_Y2_FROZEN_T3B` / `_y2_scale_factor`, `_W4T1_FROZEN_64`).
* `tests/unit/test_v5_21_2_subsystem_audits.py` — O-9 pin 5.
* `tests/unit/test_niche_audit_r_guards_and_merits.py` — R-5 restated σ-to-σ.

## F.9 Follow-up test runs

| command | result | duration |
|---|---|---|
| `test_niche_audit_w3_oracles.py` | **181 passed** | 81 s |
| `test_niche_audit_r_guards_and_merits.py test_audit_optimize.py` | **109 passed** | 76 s |
| `test_audit2609_a4_verify_maslov_asymptotic.py -k "o1 or o2 or o6"` | **6 passed** | 2 s |
| `test_v5_21_2_subsystem_audits.py -k opt1_lg` | **1 passed** | 17 s |
| `test_audit2609_a4_verify_maslov_asymptotic.py test_audit2609_a4_maslov_gbd.py test_audit2609_a4_asymptotic.py test_niche_audit_w6_asymptotic.py test_audit_propagation.py test_v4_16_0_walker_all_symmetry.py test_v4_14_1_dispatcher_pin_cache_clears.py` | **282 passed** (the walker and the cache-clear pin included — O-8) | 265 s |
| `test_niche_audit_w3_oracles.py test_v5_21_2_subsystem_audits.py test_v5_21_maslov_jax_caustic.py test_niche_audit_eh1_maslov_upsample.py test_v5_1_0_agent_e_split.py test_v4_16_1_agent_d.py` | **321 passed, 1 skipped** | 464 s |
| `python validation/run_all.py test_lenses test_asymptotic` | **46/46 + 48/48 passed** (27.8 s + 58.5 s) | — |
| `python -m ruff check` on every file touched in the follow-up | clean | — |
