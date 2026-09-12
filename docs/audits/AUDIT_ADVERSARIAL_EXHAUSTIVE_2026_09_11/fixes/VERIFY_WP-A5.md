# VERIFY-WP-A5 — adversarial re-verification of the propagator-kernel work package

Verifier: VERIFY-A5 (independent; did not write the fixes).
Subject: commit `6b801ffa` on `audit-fixes-2026-09` (diff base `8bb4b03c`), the WP-A5
report `fixes/WP-A5_REPORT.md` and changelog `fixes/WP-A5_CHANGELOG.md`, audit rows
K1–K24 (§3, §3.1, §3.2) and the partition reports `PROP-CORE.md`,
`PROP-CORE-SUB1.md`, `PROP-HF.md`, `ORCHESTRATOR.md`.

Every claim below was **re-measured**, not read. Where the WP used an oracle I built a
different one on a fixture the WP did not use. No git write commands were run.
`OPENBLAS_NUM_THREADS=1` on every invocation.

> **Label note.** The audit's §3 table numbers the `backend.scipy.jv` row **K1** and the
> Fresnel chirp-guard row **K2**; the WP brief and the WP report use the opposite
> assignment. This report follows the WP's labels (K1 = Fresnel guard, K2 = `jv`) and
> flags the collision for the orchestrator — both items are in fact fixed.

---

## 1. Verdict

| ID | WP claim | Verdict | Independent evidence |
|---|---|---|---|
| **K16** | fixed (P0) | **VERIFIED** | My own Debye–Wolf (θ,φ) quadrature, NA 0.7 / f 2 mm / λ 488 nm (WP used NA 0.5 / f 4 mm / λ 633 nm): componentwise code/oracle for `E_z` is **+1.00 … +1.14**, the same complex constant as `E_x`. y-pol and circular pol verified additionally (WP pinned x-pol only). |
| **K10** | fixed (docs+test) | **VERIFIED** | Genuinely decentred sub-aperture (WP used a pupil phase ramp): focal phase ramp **−5.366e6 rad/m** vs the aperture-convention prediction **−5.408e6** (ratio 0.992); the ray-direction convention predicts **+5.408e6**. |
| **K9** | fixed (P0) | **VERIFIED-WITH-NOTES** | Independent 8×-pad linear-convolution reference on odd N, anamorphic `dy≠dx`, complex64 and a hard edge: relL2 **2.3e-14 / 2.7e-9 / 2.2e-10 / 4.2e-14**, `P_out/P_in` **1.000000**, against pre-fix **4.55 / 3.85 / 2.03 / 1.93** and **21.4 / 15.2 / 5.3 / 4.7**. Notes: (a) a docstring claim was wrong — **fixed by me**; (b) the wrap-around failure mode is reachable and unwarned (see §3.1). |
| **K1** (Fresnel guard) | fixed | **VERIFIED** | Warns at 0.1/0.25/0.5/0.999 `z_crit`, silent at 1.0/2.0, and the returned field is **byte-identical to the pre-fix module at every z** (diagnostic only, as claimed). |
| **K2** (`jv`) | fixed | **VERIFIED** | Five (v,x) pairs the WP did not use, all exact against `scipy.special.jv`; array case max diff 0.0. |
| **K3** | fixed | **VERIFIED** | complex64 phase error now **4.0e-7 … 7.5e-7 rad, flat in both z and N** across N = 256/512/1024 × z = 3.24 mm … 1 m, where the pre-fix module reads **1.6e-3 → 1.31 rad** (grows with k·z). One undocumented side-effect noted (§3.5). |
| **K4** | fixed | **REGRESSION (found, fixed by me)** | The per-slot locks removed the serialisation that was masking a ping-pong-buffer lifetime hazard. 8 threads × 40 concurrent RS calls: **7/320 returned a 100 %-wrong field** (max\|out−ref\|/max\|ref\| = 1.14), 0/320 with the pre-K4 entry-wide lock. Fixed in `fft_infra.py`; now 0/320. See §3.2. |
| **K5** | fixed, byte-identical | **VERIFIED** | H build is byte-identical at band widths 2¹², 2¹⁸ and 2³⁰ at N = 512 (the width is a free choice, as claimed). |
| **K6** | fixed (docs+warning) | **VERIFIED** | `resample_field` docstring carries the measured MTF table; `propagate_through_system` warns on the resample-back crop. |
| **K7** | fixed | **VERIFIED** | Blacklist keys are `(shape, dtype.str, direction)` triples; 1 of 4 combinations blacklisted by one failure. |
| **K8** | fixed (4 items) | **VERIFIED** | Cached H `writeable=False`, in-place write raises, public `return_transfer_function` return still writeable, `__array__(self, dtype=None, copy=None)`. Square-vs-natural bit-exactness re-checked at odd N via the WP's own pin (passes). |
| **K11** | fixed | **VERIFIED** | Cropped windows return exactly the power inside them (ratio **1.000000** at N_out = 32/64/128/256) and the crop warns when it crops. |
| **K12** | fixed | **VERIFIED** | `wavelength` is `KEYWORD_ONLY` with no default on both aperture entry points. |
| **K13** | fixed (free space) | **NOT FIXED on the aperture entry point** | The one-leg `accumulate_to_grid` path is fixed (scale **0.94–1.08** vs my own RS-I quadrature, flat across output pitch ×1/×2/×4 and z). But `propagate_hfpi_freespace_aperture` — the function the docstring calls "the canonical single-aperture-diffraction validation case", and what `propagate_hfpi` and `propagate(method='hfpi')` call — returns an amplitude **low by exactly `n_paths × z_to_aperture`** while defaulting to `normalisation='physical'`. Measured across three z₁ and two path counts. See §3.3. **P1, open.** |
| **K14** | fixed | **VERIFIED** | `cone_half_angle` accepted by `propagate_hfpi`, `propagate_hfpi_freespace_aperture` and the vector entry; MHS centre check covered by the WP's passing pin. |
| **K15** | fixed | **VERIFIED-WITH-NOTES** | Module docstrings are honest. Two FUNCTION docstrings in `vectorial_hfpi.py` still asserted the "m-theory dipole formalism" and one of them described the PRE-K17 physics. **Fixed by me** (§3.4). |
| **K17** | fixed (honest minimum) | **VERIFIED** | My own checks: rotation preserves ‖E‖² to **1.8e-15** on 20 000 random (s_from, s_to, complex E) triples, carries `s_from` onto `s_to` to 3.2e-15, preserves transversality to 1.3e-15, and reproduces my independently written Richards–Wolf aplanatic matrix to **2.7e-16**. The emission projection changes no path's ‖E‖ (6.7e-16). |
| **K18** | fixed | **VERIFIED** (source term) | Subsumed by the K13 one-leg check: the absolute scale against an independent RS-I quadrature is 1.00 ± 6 % with the `N_src_px` factor in place. The aperture RE-emission's measure is the defect in §3.3. |
| **K19** | fixed | **VERIFIED** | Two `rng=None` runs differ; `rng=None` ≠ `rng=0`. |
| **K20** | fixed | **VERIFIED** | `propagate(method=…)` returns an ndarray with and without `output_grid` for asm / hf / hfpi. |
| **K21** | fixed | **VERIFIED** | 0.5 % at 1 µm and 10 % at 100 nm now resample; an exact no-op is still bit-identical. |
| **K22** | partially fixed | **VERIFIED** (as restated) | Bit-identical at `chunk_output` ∈ {auto,1,2,4,8} at N_in = 64/128/**512**. The auto rule gives `n_chunk = 1` for every N_in ≥ 128 (measured: 64→4, 90→2, 128→1, 256→1, **512→1**, 1024→1), so it can never select the slower batch. |
| **K23** | fixed | **VERIFIED** | `n_paths=100` with a 1 048 576-cell stratification → **100** paths; `n_paths=20000` default → exactly 20 000. |
| **K24** | fixed (6 items) | **VERIFIED** | `_spawn_rng(parent, 1)` identical whether or not stream 0 was drawn first; `__all__` entries present. |

**Totals.** 20 VERIFIED, 3 VERIFIED-WITH-NOTES (K9, K15 + the K1/K2 label collision),
**1 NOT FIXED (K13 on the aperture chain, P1)**, **1 REGRESSION found and fixed (K4, P1)**.

---

## 2. Open items for the orchestrator, by severity

| # | Severity | Item |
|---|---|---|
| **V1** | **P1 — open, needs WP-A5 or a follow-up** | `propagate_hfpi_freespace_aperture` / `propagate_hfpi` / `propagate(method='hfpi')` return amplitudes low by `n_paths × z_to_aperture` under the new default `normalisation='physical'` (§3.3). The WP report's "K13 fixed (free space)" covers only `init_paths_from_field → propagate_to_plane → accumulate_to_grid`; its K13 test exercises exactly that path and never calls the aperture entry point. |
| **V2** | P1 — **fixed by me**, needs review | K4's per-slot locks exposed a ping-pong-buffer lifetime race: concurrent `rayleigh_sommerfeld_propagate` returned 100 %-wrong fields, silently (§3.2). Fixed in `fft_infra.py` with two new pins. |
| **V3** | P3 — **fixed by me** | `rs.py`'s `kernel` docstring claimed z = 200 µm and 300 µm on the N = 128 / dx = 1 µm probe are "above the threshold … byte-for-byte the pre-v5.46 numbers". `z_crit` there is 404.4 µm, so both are BELOW it and both change (by 1.7e-13 / 2.1e-13 relative — numerically nothing, but the claim as written is false). The same mislabelling appears in the WP report §2's K9 table (rows "200 µm" and "300 µm"). |
| **V4** | P3 — **fixed by me** | Two function docstrings in `vectorial_hfpi.py` still asserted the "m-theory dipole formalism"/"m-theory dipole obliquity" the module header now says does not exist, and one described the pre-K17 physics (§3.4). |
| **V5** | P3 — for the orchestrator | K1/K2 label collision between the audit §3 table and the WP brief/report (both items fixed; only the cross-reference is wrong). |
| **V6** | P2 — informational | The `'transfer'` branch's circular wrap-around IS reachable below `z_crit` for a strongly-diverging source, with no warning: measured relL2 **2.1e-4 … 2.9e-3** against an 8×-pad reference with 3–21 % of the power outside the padded window (§3.1). Documented as a residual risk by the WP; a runtime guard is cheap (see the suggested predicate). |
| **V7** | P3 — informational | WP-A5 deferred item 6 ("JAX parity … worth one x64-enabled run in CI") is now **closed for the RS path**: with `jax_enable_x64=True` the NumPy/JAX difference is **3.4e-16 … 7.4e-16** on both branches, including odd N = 65 (§3.6). The WP's statement that "the session has x64 OFF" is true of a bare interpreter but not of the pytest session, whose conftest enables it (the `test_audit_jax_c64_propagator_precision.py` skips say so explicitly). |
| **V8** | P3 — informational | SAS's K3 rewrite also moves the **complex128** default path by up to **1.8e-9** relative at z = 1 m (the cancellation-free form is more accurate). Correct, but the changelog presents K3 as a float32-only change; the float64 movement deserves a line. |

---

## 3. Detail

### 3.1 K9 — the Rayleigh–Sommerfeld routing

**Repro re-run.** `repro/orch/verify_rs_alias.py` on the current code: `RS P/P0 = 1.000`
on all five rows (pre-fix 21.44 and 5.31), ASM 1.0000 throughout.

**Pre-fix reproduction and far-field bit-identity.** I loaded the pre-v5.46 `rs.py` blob
(`git show 8bb4b03c:…`) as a sibling module inside the live package and compared directly:

| N | dx | z | z_crit | branch | pre-fix P/P₀ | current P/P₀ | byte-identical |
|---|---|---|---|---|---|---|---|
| 64 | 2 µm | 50 µm | 808.8 µm | transfer | **21.440** | 1.000000 | no (by design) |
| 128 | 1 µm | 50 µm | 404.4 µm | transfer | **5.313** | 1.000000 | no |
| 128 | 2 µm | 50 µm | 1617.7 µm | transfer | **25.697** | 1.000000 | no |
| 128 | 1 µm | 1 mm | 404.4 µm | spatial | 1.000 | 0.999637 | **yes** |
| 128 | 1 µm | 3 mm | 404.4 µm | spatial | 0.632 | 0.632278 | **yes** |
| **65** | 1.5 µm | 900 µm | 462.1 µm | spatial | 0.997 | 0.996817 | **yes** |
| **100** | 1 µm | 500 µm | 316.0 µm | spatial | 1.000 | 1.000000 | **yes** |

So the "bit-identical above the threshold" claim is true, including at odd N — but the
WP's own test only compares `auto` with `kernel='spatial'`, which is what the routing is
*defined* to do and cannot detect a change in the spatial kernel. The table above is the
real check.

**Independent oracle #1 — continuum RS-I quadrature** (super-sampled midpoint rule over
the same finite window, analytic source, no FFT, no Hankel transform, no library call;
self-convergence S = 400 vs 800 quoted):

| fixture | oracle floor | `kernel='auto'` | `kernel='spatial'` |
|---|---|---|---|
| odd N = 65, dx = dy = 1.5 µm, z = 30 µm (near) | 3.1e-14 | **9.4e-14** | raises |
| odd N = 65, z = 900 µm (far) | 3.6e-14 | **1.6e-13** | 1.6e-13 |
| anamorphic dy = 2dx, z = 40 µm (near) | 8.1e-15 | 6.3e-6 * | raises |
| anamorphic dy = 2dx, z = 1200 µm (far) | 2.6e-14 | **1.9e-13** | 1.9e-13 |
| complex64, z = 40 µm | 5.6e-15 | **8.3e-8** | raises |

\* the 6.3e-6 is the source's own y-sampling (w₀ = 4 µm at dy = 2 µm), not the
propagator: the continuum oracle integrates the smooth Gaussian, the library the sampled
one. Re-run at w₀ = 9 µm the same fixture reads 2.7e-9 (table below).

**Independent oracle #2 — 8×-zero-pad analytic linear convolution** (written out in the
new test file, `pad = 8` vs `pad = 16` quoted as its own floor):

| fixture | ref floor | current | **pre-fix module** | current P/P_in | pre-fix P/P_in |
|---|---|---|---|---|---|
| odd N = 65, dx = dy = 1.5 µm, λ 532, z 30 µm | 3.4e-15 | **2.3e-14** | 4.554 | 1.000000 | 21.394 |
| anamorphic dy = 2 µm / dx = 1 µm, λ 532, z 40 µm | 9.8e-11 | **2.7e-9** | 3.851 | 1.000000 | 15.173 |
| N = 48, dx = 1 µm, λ 633, z 20 µm | 8.5e-12 | **2.2e-10** | 2.028 | 1.000000 | 5.273 |
| odd N = 127, dx = 0.8 µm, λ 488, z 60 µm | 5.8e-15 | **4.2e-14** | 1.925 | 1.000000 | 4.705 |
| the same four at **complex64** | — | 1.3e-7 / 1.2e-7 / 1.3e-7 / 3.8e-7 | — | 1.000000 (×3), 0.999999 | — |

**Crossover.** Measured at `z = z_crit` on three grids the WP did not use, with each
arm's own distance from the 8×-pad reference beside it:

| grid | relL2(transfer, spatial) | transfer vs ref | spatial vs ref |
|---|---|---|---|
| N = 65, dx = 1.0 µm, λ 633 | 1.16e-13 | 8.6e-14 | 1.0e-13 |
| N = 96, dx = 0.7 µm, λ 633 | 8.40e-14 | 1.3e-13 | 1.1e-13 |
| N = 128, dx = 0.5 µm, λ 532 | 8.05e-14 | 1.7e-13 | 1.4e-13 |

i.e. the step and both arms sit together at the FFT round-off floor — no observable
discontinuity. (The WP's quoted steps, 6.3e-5 / 1.0e-9 / 3.8e-14, are from their own
coarser probe; mine are three to eleven decades tighter, so the claim holds *a fortiori*.)

**The wrap-around failure mode (V6).** The WP documents it as a residual risk and my
measurement confirms both halves of its argument:

* Where `'auto'` chooses `spatial` (z = 3 mm, N = 128, dx = 1 µm, 63 % of the power still
  in the window): `transfer` measures **1.92e-2** against the 8×-pad reference where
  `spatial` measures **8.9e-13**. The routing is what protects that case — confirmed.
* Where `'auto'` chooses `transfer` (z < z_crit) the failure IS reachable: a band-limited
  random-phase screen at dx ≈ λ, whose content fills the grid's own angular range, loses
  21 % / 3 % / 18 % of its power out of the padded window before z_crit and reads
  **2.9e-3 / 2.1e-4 / 7.5e-4** against the reference, **with zero warnings** and with
  `kernel='spatial'` refusing (correctly) to offer an alternative.
* It is *not* reachable for a properly sampled beam: leaving the padded window before
  `z = 2Ndx²/λ` requires `tanθ > λ/(2dx)`, i.e. exceeding the grid's own maximum
  representable angle. The exposure is exactly the "grid pitch at the wavelength scale,
  content at the Nyquist edge" corner.
* **Suggested guard** (cheap, no new machinery): on the `'transfer'` branch, compare the
  power in the outer ring of the padded window with the total after the multiply and
  warn above a few per cent, naming a larger pad. Or expose the pad factor.

**Cache sharing with ASM (checked as asked).** `_get_asm_H_natural`'s key carries
`bool(bandlimit)`, the padded shape, both pitches, λ, z and the dtype, so an RS
`kernel='transfer'` call and a direct ASM call at the padded geometry share an entry only
when they are the same array — verified: RS(transfer) cropped == ASM on the 2N pad,
**bit-identical**; cold-vs-warm results identical in all four interleavings
(RS bl=False→True, RS bl=True→False, ASM(bl=True)→RS, RS→ASM); and no consumer writes to
the now read-only cached H. Note for the record: on the `'transfer'` branch `bandlimit`
is *necessarily* a no-op — the Matsushima cutoff exceeds the grid Nyquist under exactly
the algebraic condition `z < z_crit` — measured relL2(False, True) = **0.0**. That is the
docstring's own claim, confirmed.

**Adversarial inputs.** Real float64, float32, F-ordered, non-C-contiguous strided view,
complex64 all accepted and correct (relative to the complex128 result: 0.0 / 2.2e-8 /
0.0 / 0.0 / 1.5e-7); `z = 0` and `z < 0` raise; `kernel='SPATIAL'` (wrong case) raises
with the `fn_name:` prefix.

### 3.2 K4 — REGRESSION: concurrent FFT callers get a wrong field (fixed)

`_fft2` hands the caller one of two pyFFTW ping-pong buffers *by reference*. The
documented contract — "your slot stays valid until the call AFTER next at this key" — is
a single-threaded statement. With T threads issuing calls at one key in arbitrary order,
a caller's slot can be recycled while it still holds the reference; it then multiplies a
stale or half-written spectrum. Before K4 the single entry-wide lock serialised every
call at a key and masked this. K4's per-slot locks removed the masking.

Measured (this box, 8 threads × 40 reps, each result compared byte-for-byte with the
single-threaded reference):

| configuration | `rayleigh_sommerfeld_propagate` | `angular_spectrum_propagate` |
|---|---|---|
| per-slot locks, as committed | **7 / 320 wrong, max rel 1.14** | 0 / 320 |
| entry-wide lock collapsed back (pre-K4) | 0 / 320 | 0 / 320 |
| `set_fft_double_buffer(False)` | 0 / 320 | 0 / 320 |
| per-slot locks **+ this fix** | **0 / 320** | 0 / 320 |

The isolating experiment replaces `entry['locks']` with two references to one Lock — the
exact pre-K4 structure — at runtime, so nothing else differs. A four-shape × two-dtype ×
four-kernel sweep at 2/4/8 threads reproduced the same signature (mismatches only on the
RS entry point, both kernel branches: 5/64 spatial, 3/64 transfer).

**Fix applied** (`lumenairy/propagators/fft_infra.py`): `_note_fft_thread()` latches the
first thread that reaches `_get_or_make_plan`; the instant a *second* thread issues an
FFT, every pyFFTW return privatises its buffer for the rest of the process. The latch is
set *before* the newcomer executes its plan and the slot index has already advanced, so
the incumbent's outstanding view is on the alternate slot and survives. Single-threaded
callers — the case the double buffer exists for — keep the zero-copy path, verified
bit-identical and still recycling the slot on the third call.

Cost: one copy per FFT in a process that has ever used FFTs from two threads. That is the
price of correctness here; the alternative (a per-thread plan+buffer) is a much larger
change to the plan cache and I did not attempt it in a verification pass.

### 3.3 K13 — NOT FIXED on the aperture entry point (V1)

**What is fixed.** The single-leg path. Against my own super-sampled RS-I quadrature
(no ASM, no library propagator on the oracle side), fixture 48×48 at dx = 1.5 µm,
w₀ = 7 µm, λ = 532 nm — all different from the WP's 32×32 / dx 4 µm / w₀ 12 µm / ASM:

| z | output pitch | HFPI / RS-I least-squares scale (two seeds) |
|---|---|---|
| 0.6 mm | 1.5 µm (×1) | 0.9586, 0.9441 |
| 0.6 mm | 3.0 µm (×2) | 0.9546, 0.9441 |
| 0.6 mm | 6.0 µm (×4) | 0.9510, 0.9397 |
| 1.5 mm | 1.5 / 3.0 / 6.0 µm | 0.9903 / 0.9802 / 0.9669 and 1.0778 / 1.0473 / 0.9793 |

Flat in output pixel area — which is precisely the K13 property — and flat in path count
(0.9518 / 0.9223 / 0.9266 at 0.25 / 1 / 3 M paths).

**What is not.** `propagate_hfpi_freespace_aperture` (and therefore `propagate_hfpi` and
`propagate(method='hfpi')`) defaults to `normalisation='physical'` but returns an
amplitude that still scales as 1/`n_paths`. Oracle-free test — an unobstructed aperture
plane must be transparent, so the two-leg walk must equal the one-leg walk over the same
total distance:

| z₁ | z₂ | n_paths | two-leg / one-leg | × n_paths | **× n_paths × z₁** |
|---|---|---|---|---|---|
| 0.25 mm | 0.75 mm | 500 000 | 8.78e-3 | 4392 | **1.098** |
| 0.25 mm | 0.75 mm | 2 000 000 | 2.09e-3 | 4183 | **1.046** |
| 0.50 mm | 0.50 mm | 500 000 | 3.44e-3 | 1721 | **0.860** |
| 0.50 mm | 0.50 mm | 2 000 000 | 9.20e-4 | 1840 | **0.920** |
| 1.00 mm | 1.00 mm | 500 000 | 2.68e-3 | 1339 | **1.339** |
| 1.00 mm | 1.00 mm | 2 000 000 | 4.00e-4 | 801 | **0.801** |

The last column is 1.0 within Monte-Carlo scatter across a 4× range of z₁ and a 4× range
of `n_paths`: the returned amplitude is **low by exactly `n_paths × z₁`** (z₁ in metres).
Independently confirmed against my RS-I quadrature through the real aperture: scale
**4e-4** at n = 4 M, z₁ = 0.5 mm, where `1/(n z₁) = 5e-4`.

**Diagnosis** (`hfpi.py:473`, `vectorial_hfpi.py:455`):

```python
solid_angle = 2.0 * float(np.pi) * (1.0 - cos_max) / float(n)
```

Two distinct errors in one line, for a chain:

1. **The `/ n` is a second division by the sample count.** A path is ONE sample of the
   joint (source pixel, direction₁, direction₂, …) integral, so the `1/n` belongs once —
   `init_paths_from_field` already applies it together with `A_src · Ω₁`. The re-emission
   measure should be the cone solid angle itself.
2. **The intermediate leg carries no Jacobian.** Written in direction variables the
   Kirchhoff kernel is `(1/(iλ))·e^{ikr}·r·dΩ` — note the factor `r`, not `1/r`. The
   emission applies `cos θ₁` where the chain needs `r₁`; that is the `z₁` in the measured
   residual. The final leg's `r` is supplied by the K13 binning Jacobian
   (`r/(dx_out² cos θ_out)`), which is why the one-leg path is correct and only chains
   are wrong.

**Reproduction** (self-contained, ~30 s):

```python
import numpy as np, warnings
from lumenairy.propagators.hfpi import (init_paths_from_field, propagate_to_plane,
                                        accumulate_to_grid,
                                        propagate_hfpi_freespace_aperture)
LAM, N, dx, w0 = 532e-9, 48, 1.5e-6, 7e-6
x = (np.arange(N) - N/2)*dx; X, Y = np.meshgrid(x, x, indexing='xy')
E0 = np.exp(-(X**2+Y**2)/w0**2).astype(complex)
z1 = z2 = 0.5e-3
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    p = init_paths_from_field(E0, dx, n_paths=3_000_000, wavelength=LAM, rng=1,
                              cone_half_angle=0.08)
    p = propagate_to_plane(p, z_target=z1+z2, wavelength=LAM)
    one = np.asarray(accumulate_to_grid(p, Ny=24, Nx=24, dx=3e-6,
                                        on_undersampled='silent'))
    for n in (500_000, 2_000_000):
        two = np.asarray(propagate_hfpi_freespace_aperture(
            E0, dx, z_to_aperture=z1, aperture_radius=600e-6,
            z_aperture_to_output=z2, wavelength=LAM, n_paths=n, rng=5,
            output_shape=(24, 24), output_dx=3e-6, cone_half_angle=0.08,
            on_undersampled='silent'))
        m = np.abs(one) > 0.1*np.abs(one).max()
        r = abs(complex(np.sum(two[m]*np.conj(one[m]))/np.sum(np.abs(one[m])**2)))
        print(n, f"{r:.3e}", f"n*z1*r = {n*z1*r:.3f}")   # -> ~1.0, not ~1/1
```

**Why the WP's own test did not catch it.** `TestK13BinningJacobian::
test_the_free_space_estimator_reproduces_the_hf_integral` builds the bundle by hand
(`init_paths_from_field → propagate_to_plane → accumulate_to_grid`) and never calls the
aperture entry point; the report's "fixed (free space)" is true of that path only.

**No red test was added for this** — a failing pin would block the branch for every other
WP. The reproduction above is what a follow-up should turn into one, together with the
"unobstructed aperture is transparent" property (which is oracle-free and therefore the
right shape for a durable pin).

### 3.4 K17 / K15 — the rigid rotation, and two stale docstrings (fixed)

Independent checks of `_rigid_rotate` (all written here, none taken from the module):

| property | measured |
|---|---|
| ‖E′‖² / ‖E‖² − 1 on 20 000 random (s_from, s_to, complex E) | max **1.78e-15** |
| ‖R·s_from − s_to‖ | max **3.22e-15** |
| transversality preserved (E·s_from = 0 ⟹ E′·s_to = 0) | max **1.28e-15** |
| vs my own `e′ = [cosθcos²φ+sin²φ, cosφ sinφ(cosθ−1), −sinθ cosφ]` for s_from = +z | max **2.67e-16** |
| emission projection changes no path's ‖E‖ (`vector_projection` True vs False) | max **6.66e-16** |

The reduction to the Richards–Wolf matrix ties K17 to K16 with the *ray-direction*
azimuth, which is the convention that makes the two consistent — the WP's reasoning holds.
`|Ez|²/|E|²` = 2.2e-3 and cross-pol 2.5e-6 on a 0.10 rad cone; both were identically 0
pre-fix.

**On the y-polarised mirror at module level:** a statistical comparison is not available
here — the two runs share a seed, so they share the *same* path geometry rather than the
90°-rotated one, and the Monte-Carlo realisations therefore do not cancel. The property
that *is* deterministic is the equivariance of the rotation itself, verified to 2.7e-16
above (a rotation about z commutes with `_rigid_rotate` by construction), plus the
Richards–Wolf y-polarisation pin I added in §4.

**Docstrings (V4, fixed).** `propagate_vector_to_plane` still said the neglected rotation
is what "the full m-theory dipole formalism captures", and
`apply_vector_aperture_diffraction` still said the Jones vector "is multiplied by
`cos(theta_new)` to account for the m-theory dipole obliquity" — the first repeats the
claim the module header now retracts, the second additionally describes the PRE-K17 code
(since v5.46 it is a rigid rotation times the symmetric scalar
`0.5(cosθ_in + cosθ_out)`). Both rewritten to what the code does now, each keeping a
one-sentence retraction of the old claim.

### 3.5 K3 — SAS at single precision, and a float64 side-effect (V8)

Measured end-to-end (complex64 field vs its complex128 twin, max phase error over the
bright region), on a grid the WP did not sweep:

| N | z = 3.24 mm | 1 cm | 10 cm | 1 m |
|---|---|---|---|---|
| 256 — current | 4.03e-7 | 5.25e-7 | 4.69e-7 | **3.94e-7** |
| 256 — pre-fix module | 3.41e-3 | 1.99e-2 | 8.04e-2 | **5.92e-1** |
| 512 — current | 4.24e-7 | 4.98e-7 | 7.50e-7 | **5.60e-7** |
| 512 — pre-fix | 2.28e-3 | 2.56e-2 | 6.50e-2 | **6.43e-1** |
| 1024 — current | 4.57e-7 | 6.42e-7 | 7.33e-7 | **7.23e-7** |
| 1024 — pre-fix | 1.61e-3 | 9.38e-3 | 1.55e-1 | **1.308** |

Flat in z and in N, at the complex64 floor — the library's own contract. The WP's
"ratio across a 300× range of z" pin is the right shape and is corroborated.

**Side-effect not in the changelog:** the complex128 default path also moves, by
5.9e-13 / 4.1e-12 / 1.0e-10 / **1.8e-9** relative at those z (the cancellation-free
`−u²/(2(1+√(1−u))²)` form is strictly more accurate than the subtraction, and the
`W & prop` gate replaces a `W *` multiply). Correct direction, but it is a numerical
change on a default path and should be stated.

### 3.6 JAX parity (V7)

With `jax_enable_x64 = True` (jax 0.10.1):

| grid | branch | RS numpy vs jax | ASM numpy vs jax |
|---|---|---|---|
| N = 64, dx = 1 µm, z = 40 µm | transfer | **3.40e-16** | 2.50e-16 |
| N = 64, dx = 1 µm, z = 600 µm | spatial | **5.07e-16** | 3.76e-16 |
| N = 65, dx = 1.5 µm, z = 40 µm | transfer | **6.93e-16** | 3.82e-16 |
| N = 65, dx = 1.5 µm, z = 600 µm | spatial | **7.45e-16** | 6.00e-16 |

At x64 off (the bare-interpreter default) the new transfer branch runs complex64 and
agrees with the NumPy complex128 result to 1.6e-7 — the single-precision floor.

---

## 4. Tests added

Both files pass and both were demonstrated to FAIL on the pre-fix state.

**`tests/unit/test_audit2609_a5_verify_fft_buffer_threads.py`** (2 tests)

* `test_concurrent_propagations_match_the_single_threaded_reference` — 6 threads × 20
  reps of RS and ASM, bit-identity against the single-threaded reference.
  Fail-before: **3 of 240 concurrent propagations wrong, worst deviation 1.006**, with the
  latch disabled in-process.
* `test_the_buffer_is_privatised_once_a_second_thread_uses_the_fft` — structural: the
  single-threaded third call still recycles the first slot (the zero-copy counter-pin),
  and after a second thread has issued an FFT no two returns share memory and the
  transform is still right. Fail-before: **"the shared-buffer latch did not trip"**.

**`tests/unit/test_audit2609_a5_verify_rs_and_rw.py`** (18 tests)

* K9 against the 8×-pad analytic reference: field (bar 1e-5; measured 2.3e-14…2.7e-9;
  pre-fix 1.93…4.55), energy (bar 1e-4; measured 1.000000; pre-fix 4.7…21.4) and
  complex64, on odd N = 65/127 and an anamorphic pitch.
* K9 crossover continuity at `z_crit` on three new grids (bar 1e-8; measured ≤1.2e-13).
* K9 branch-selection decision either side of the threshold.
* K16 for `polarization='y'`: `Im(E_z/E_y) < 0` at +y (measured −0.7801, real part
  −8.2e-17), `E_z` odd in y (1.19e-15), zero on the x axis (2.43e-19), and the exact 90°
  transpose relation to the x-polarised `E_z` (8.14e-17).
* K16 for `polarization='circular'`: the `E_z` vortex null on axis (3.03e-18 of its own
  maximum, 8.9e15 below a point three pixels away) and the linearity identity
  `E_z(circ) = (E_z(x) + i E_z(y))/√2`, which a half-revert on one polarisation state
  would break.

Every bar carries its oracle, the oracle's own floor, the measured value and the decades
of gap on both sides, per `docs/TESTING_STANDARDS.md`.

**Audit of WP-A5's own new tests against TESTING_STANDARDS.** 80 tests, 16 classes; every
numeric bar I checked carries a derivation, a measured value and a pre-fix number; no
wall-clock or speed-up assertion anywhere (K22 is pinned on bit-identity and on argument
validation, not on timing); `pytest.skip` is used only for genuinely absent optional
dependencies (`numexpr`) and for a JAX x64 process-state arm, not for resource
preconditions. Two weaknesses:

1. `test_far_field_is_bit_identical_to_the_historical_kernel` compares `auto` with
   `kernel='spatial'` — a tautology given the routing code, testing the implementation
   rather than the property. §3.1's table is the property; my new branch-selection test
   states the tautological half honestly as a decision.
2. `TestK13BinningJacobian` tests the hand-built path only, which is how V1 survived.

### 5. Files I changed

| file | change |
|---|---|
| `lumenairy/propagators/fft_infra.py` | V2 — `_PYFFTW_FIRST_FFT_THREAD` / `_PYFFTW_SHARED_BUFFERS_UNSAFE` latch + `_note_fft_thread()`, called from `_get_or_make_plan`; the four pyFFTW return sites privatise the buffer once the latch is set. |
| `lumenairy/propagators/rs.py` | V3 — the `kernel` docstring's "above the threshold … byte-for-byte" paragraph replaced with the measured facts (byte-identical at z = 405 µm and 1 mm; 1.7e-13 / 2.1e-13 at 200 / 300 µm, which are BELOW `z_crit` = 404.4 µm). |
| `lumenairy/propagators/vectorial_hfpi.py` | V4 — two function docstrings rewritten (docstring only, no code). |
| `tests/unit/test_audit2609_a5_verify_fft_buffer_threads.py` | new (2 tests) |
| `tests/unit/test_audit2609_a5_verify_rs_and_rw.py` | new (18 tests) |

All three source files are WP-A5's own, per its brief's "Files you own". Nothing else was
touched.

### 6. Tests run

| command | result |
|---|---|
| `pytest tests/unit/test_audit2609_a5_propagators.py` | **80 passed**, 61 s |
| `pytest` over the 21-file propagator surface (pre-fix state of my work) | **503 passed, 6 skipped**, 287 s |
| `pytest` over 26 files incl. both new files (post-fix) | 766 passed, 10 skipped, **4 failed** — all four reproduce with my fix disabled in-process, i.e. not mine and not WP-A5's (see below) |
| `pytest` over the 11 FFT-infra-adjacent files + both new files | **337 passed, 5 skipped** |
| `python validation/run_all.py test_propagation test_hf test_hfpi test_vectorial_hfpi test_mhs test_dispatch` | **6/6 files pass** (run twice: before and after my `fft_infra` change) |
| `pytest tests/unit/test_audit2609_a5_verify_*.py` | **20 passed** |

**The four failures, attributed.** All four were re-run with my `fft_infra` latch disabled
by a pytest plugin that restores the exact WP-A5-as-committed behaviour; the same four
failed, so none is caused by my change. They were also all passing in my first run of the
same files ~40 minutes earlier, and `git status` shows other agents writing
`lumenairy/propagators/gbd.py`, `lumenairy/algebra/*`, `lumenairy/elements/*` in that
window (file mtimes seconds before the run).

* `test_audit_w5_propagators.py::TestP230CollinsFactor` — the Collins amplitude is the
  exact complex CONJUGATE of the expected value (0.000126 − 0.011219j vs
  0.000126 + 0.011219j). A sign-convention flip in the GBD/Collins path, which WP-A5 does
  not own (`gbd.py` is explicitly excluded by its brief).
* `test_audit_propagation.py::…ModalAsymptoticStillBitEqual` (2 tests) — modal/asymptotic
  bit-equality, `max|new − cold_ref| = 3.85e4`. Same origin.
* `test_niche_audit_w3_infra.py::TestA6EstimateAsmMemory::…[512-complex128]` — fails on
  the *looseness* ceiling (est/measured = 1.382 > 1.35), i.e. the measured peak came in
  SMALLER than the calibration band, which is the opposite direction from anything my
  copy-on-multithreaded change could cause. The `[1024]` arm passed on a re-run; this is
  a memory-measurement bar on a box shared with several agents.

---

## 7. What I could not close

* **V1 (K13 on the aperture chain)** needs a code change in `hfpi.py` /
  `vectorial_hfpi.py` — the re-emission measure and an intermediate-leg Jacobian. I
  diagnosed it to the line and the exact residual law but did not implement it: it
  changes the amplitude of every cascaded HFPI result by orders of magnitude and wants
  the WP author's own before/after sweep plus a migration note, not a verifier's patch.
  Until then `normalisation='physical'` on the aperture entry points is a promise the
  code does not keep; the honest interim would be to default those entry points to
  `'legacy'` with the same warning the prescription walk now carries.
* **CuPy** is not installed; the `to_numpy` routing and the CuPy branch of
  `_get_asm_H_natural` reached by RS `kernel='transfer'` remain desk-checked only, as the
  WP states.
* **Shen–Wang pixel-integrated kernel** (WP deferred item 2) — I did not evaluate it; the
  measured first-order convergence of the `'spatial'` branch is unchanged by this WP and
  the default path no longer depends on it.
