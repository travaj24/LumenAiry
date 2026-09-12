# VERIFY-A11 — independent adversarial re-verification of WP-A11

Verifier: VERIFY-A11 (did not write the fixes).  Target: commit `cf8fe5d1`, diff base `437c1b06`,
branch `audit-fixes-2026-09`.  Findings Z1–Z4 of `AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11` §10 and
the `POLAR-SOURCES-INFRA.md` partition report.  All measurements on this workstation with
`OPENBLAS_NUM_THREADS=1`, 2026-09-12, Python 3.14.6 / NumPy 2.4.6 / SciPy 1.17.1 / JAX 0.10.1.

None of WP-A11's owned library files carried uncommitted edits from other agents when I started, so
the working tree == `cf8fe5d1` for everything I verified (checked with `git status --porcelain`).

---

## 1. Verdict table

| finding | verdict | what I re-measured (independently of the WP's own fixtures) |
|---|---|---|
| **Z1** coatings `polarization` routing (NumPy) | **VERIFIED** | 12 spellings × 5 new fixtures vs my own Rouard-recursion TMM: worst \|R_lib − R_oracle\| = **5.55e-16**; 18 junk values all `ValueError` with the §2 prefix; both ports of `propagate_through_system` |
| **Z1** JAX twin | **VERIFIED** | new fixture (2-layer AR on fused silica, 65°, 1064 nm): \|R_jax − Rouard\| ≤ **2.2e-16**, NumPy↔JAX ≤ **5.6e-17**, `dR/dt`(te) bit-equal to (s), differs from (tm) by 2.8e6 |
| **Z2** Schell periodised kernel | **VERIFIED** | FFT-free sliced covariance on **anisotropic dy≠dx + odd N**, on **odd 33×33**, and at **σ=L/3** on a new grid: max\|μ − Gaussian\| **0.97 → 0.004** / **0.97 → 0.006** / **0.98 → 0.002**; analytic-GSM cross-spectral-density oracle: \|μ\| at max separation **0.938 → 0.005** (true 2.2e-15) |
| **Z2** docstring / warning | **VERIFIED** | warning fires at σ>L/6, silent below, `stacklevel` names the caller's own line from **both** public factories; capped-pad warning quotes 0.0097 residual vs 0.0042 measured (honest upper bound) |
| **Z3** `estimate_lens_memory('real')` | **VERIFIED** | different prescription (N-SF11 meniscus, 532 nm) at N = 32…1024 × 2 dtypes: est/measured **1.007–1.362**, never < 1; audit's own `p8_memory.py` reads **1.07 / 1.07** (was 0.36 / 0.63) |
| **Z3** `create_gaussian_beam` | **VERIFIED** | **180** configurations incl. anamorphic dy≠dx, Ny≠Nx, N=1, off-axis, complex64: **0** non-bit-identical |
| **Z3** algebra `FreeSpace` | **VERIFIED-WITH-NOTES** | 4f numbers reproduce exactly (3→0 warnings, 1.9318→1.0000); but the new PITCH CONTRACT docstring was **false for \|A\| ≠ 1** (fixed, §4.3), and the default change costs silent far-field truncation (§5, open item **O-1**) |
| **Z3** `stokes_parameters` / `degree_of_polarization` | **VERIFIED-WITH-NOTES → fixed** | peaks 7.00→6.00 and 8.25→6.00 reproduce at N=1024 and 2048; bit-identity **failed** for a mixed-precision `JonesField` (S2/S3 came back float32) — defect found and fixed, §4.1 |
| **Z4** `deprecated_alias` stacklevel | **VERIFIED** | all five `warn_deprecated_*` helpers **and** the shim name the caller's exact line (not just the file); `p9_infra.py` §A/§B confirm |
| **Z4** `algebra.from_prescription` | **VERIFIED** | 4 access paths + submodule-imported-first in a fresh interpreter + picklability; `algebra.__all__` correctly untouched, walker not tripped |
| **Z4** `_check_2d_scalar_field` | **VERIFIED** | 18 exotic dtypes routed correctly (object/str/bytes/datetime/timedelta/structured/void rejected; float16/uint8/longdouble/F-order/read-only/masked accepted); `np.matrix` **subclasses** also rejected |
| **Z4** cache registry | **VERIFIED-WITH-NOTES → fixed** | collision + failure warnings work, reload-idempotent under a real `importlib.reload`; but the no-code-object fallback made **all `functools.partial`s compare equal**, i.e. silent collision — defect found and fixed, §4.2 |
| **Z4** `_plane_wave_carrier` | **VERIFIED** | bit-identical to the pre-fix body for even N (8, 16, 64), changed for odd N (7, 9), and equals the package-grid oracle to **0.0** in all five cases |
| **Z4** `JonesField.propagate*` docs | **VERIFIED** | behaviour pinned, no behaviour change |
| **Z4** `user_library` `**` / `<<` | **VERIFIED** | 22 adversarial expressions incl. nested/chained/right-assoc/negative-base/double-nested: every unbounded one refused in 0.007–0.05 ms, every legitimate one evaluates |
| Z4 `import lumenairy` scipy tree | out of scope (correctly deferred by the WP) | — |

**Bottom line:** every Z1–Z4 claim I could re-measure reproduces, most of them on fixtures the WP
never touched.  Two genuine defects introduced by the WP's own changes are fixed here with
regression pins.  One design decision (Z3 `FreeSpace` default) needs an orchestrator ruling.

---

## 2. Re-run of the audit's own repro scripts

| script | reading on `cf8fe5d1` | WP claimed | agrees |
|---|---|---|---|
| `p2_coatings.py` §A/§B | Airy oracle \|ΔR\| ≤ 1.11e-16, \|ΔT\| ≤ 8.9e-16 (MgF₂ + Au, 0/45/70°) | unchanged | ✔ |
| `p2_coatings.py` §C/§E/§F | r_s = r_p = −0.206349 at 0°; TIR R = 1.0000000000, R+T = 1 exactly; absorbing-substrate closure 1.000000 | unchanged | ✔ |
| `p2_coatings.py` §D | `te`,`S` → 0.15427715 (s); `tm`,`P` → 0.00322899 (p); `banana`,`''` → `ValueError` | same | ✔ |
| `p8_memory.py` | est/meas **1.07** at N=1024 (185.2 MB) and **1.07** at N=2048 (738.3 MB), both `parallel_amp` settings | 1.07 / 1.07 | ✔ |
| `p9_infra.py` §A/§B | alias → `p9_infra.py:9`; direct helper → `:25`; shim → `:32` (was `_deprecation.py:551`) | same | ✔ |
| `p9_infra.py` §C | alias overhead **1.45 µs/call** (audit: 7.66 µs on a loaded box) | 0.87–1.86 µs | ✔ |
| `p9_infra.py` §D | object dtype and `np.matrix` → `TypeError`; real/int/bool/non-contiguous accepted | same | ✔ |
| `p9_infra.py` §G | collision → `RuntimeWarning` naming both sites; failing clearer → one `RuntimeWarning` | same | ✔ |
| `p9_infra.py` §I | `{'s': 0.94840576, 'p': 0.9969635, 'te': 0.94840576, 'tm': 0.9969635}`, `te==s? True  te==p? False` | same | ✔ |
| `p10_algebra.py` §A–§E | ABCD vs `system_abcd` **0.000e+00** on singlet / meniscus / doublet; 4f `max|M+I| = 0.0` | unchanged | ✔ |
| `p10_algebra.py` §D/§F | still raise `TypeError` — **script** defects (`CylindricalLens(f=…)`, `S(E, dx=…, wavelength=…)`), pre-existing | WP §4.3 says the same | ✔ |
| `p12_final.py` §1 | `dy` threading passes in **all 12** factories | same | ✔ |
| `p12_final.py` §2 | absorbing exit medium: \|ΔR\| ≤ 6.7e-16, \|ΔT\| ≤ 7.8e-16 over 12 combinations | same | ✔ |

I reproduced the FreeSpace §F numbers with the corrected call form by re-implementing the **pre-fix
`_apply` body verbatim** (`method='auto'` default + `return_result=False`) as a subclass in the same
process:

| | pre-fix arm (measured here) | post-fix (measured here) | WP claimed |
|---|---|---|---|
| `UserWarning`s per 4f evaluation | **3** | **0** | 3 → 0 ✔ |
| `dx_out / dx_in` | **1.9318** | **1.0000** | 1.9318 → 1.0000 ✔ |
| power ratio | 1.0000 | 0.9995 | ✔ |
| centroid | +304.0 → −304.0 µm | +304.0 → −303.9 µm | ✔ |

Downstream Z2 claims also reproduce **exactly** (N=20, dx=2.5 µm, w₀=20 µm, σ_g=10 µm, 4000
realisations, eigen-decomposition of the empirical MCF against Starikov & Wolf):

| | leading six | n=2 shell mass | degeneracy spread |
|---|---|---|---|
| `pad_sigma=0.0` (pre-fix) | `[0.40081 0.14175 0.13891 0.07445 0.06990 0.05077]` | +16.7 % | **36.4 %** |
| default (post-fix) | `[0.39205 0.15411 0.14527 0.05830 0.05597 0.05424]` | +0.8 % | **7.2 %** |

The changelog's Z2 cost table also holds: interleaved medians of 5–7 runs, 64 realisations, give
1.42× / 2.97× / 1.49× / 6.1× against the published 1.28× / 3.09× / 1.65× / 5.79×.  (A first pass read
21× for the last row; three repeats on a quieter moment gave 3.15× / 6.26× / 6.08× — that row is
shared-machine noise, not a real discrepancy.  Timings on this box are not stable to better than ~2×
while other agents run.)

---

## 3. New independent checks (oracles the library did not produce)

### 3.1 Z1 — Rouard-recursion TMM, 5 new fixtures, 12 spellings each

An interface-by-interface Rouard recursion in the **standard Fresnel** convention — a different
algorithm from the library's Abelès characteristic-matrix product.

| fixture | R_s | R_p | s/p split | max \|R_lib − R_Rouard\| over 12 spellings | `'avg'` error |
|---|---|---|---|---|---|
| TiO₂/SiO₂/TiO₂ HR on N-SF11, 30° | 0.66695517 | 0.53388548 | 0.133 | 2.2e-16 | 1.1e-16 |
| same, 75° | 0.89722988 | 0.05602856 | 0.841 | 4.4e-16 | 1.7e-16 |
| Ag 35 nm on fused silica, 55° | 0.93917641 | 0.81954707 | 0.120 | 5.6e-16 | 1.1e-16 |
| immersed `n_amb = 1.33`, 20° | 0.08942704 | 0.06434776 | 0.025 | 6.9e-17 | 1.4e-17 |
| bare interface at Brewster (56.31°) | 0.14792899 | 0.00000000 | 0.148 | 5.6e-17 | 2.8e-17 |

Vectorised wavelength axis (5 λ): `R(TE) == R(s)` and `R(tm) == R(p)` to **0.0**, and `R(TE)` matches
the Rouard s-branch to 3.3e-16, with `max|R_s − R_p| = 0.2313` across the band.

`propagate_through_system`, **both** ports, on a fixture the WP did not use (2-layer stack on N-SF11
at 70°, 633 nm):

```
transmission:  s 0.51816907  p 0.99022882  te 0.51816907  tm 0.99022882  avg 0.79026967
reflection  :  |r| s 0.85527821 ∠+3.026511 | p 0.13945211 ∠+1.978723 | te == s exactly
junk 'banana' -> ValueError on both ports
```

**Accepted-set equality** with `rcwa._core._normalize_pol` holds over 31 probes (0 mismatches),
including `' s'`, `'s '`, `'te '`, `'sp'`, `'0'`.

### 3.2 Z2 — FFT-free sliced covariance on grids the WP never used

The generator's own estimator is an FFT; so is the WP's test estimator.  Mine is a plain sliced
sample covariance (`np.vdot` on overlapping slices) — no transform, and only pairs that exist on the
grid.

| grid | arm | max\|μ−Gaussian\| (x) | (y) | μ at largest separation (x) | true |
|---|---|---|---|---|---|
| **Ny=27 (odd), Nx=40, dy=1.5·dx**, σ=4 µm, K=3000 | default | **0.0037** | **0.0080** | −0.0025 | 2.3e-21 |
| | `pad_sigma=0` | 0.9697 | 0.9415 | +0.9697 | |
| **33×33 (odd, non-5-smooth)**, σ=5 µm, K=3000 | default | **0.0056** | **0.0057** | −0.0009 | 9.7e-21 |
| | `pad_sigma=0` | 0.9569 | 0.9660 | +0.9569 | |
| **36×36, σ = L/3**, K=3000 | default | **0.0021** | **0.0047** | +0.0122 | 1.42e-02 |
| | `pad_sigma=0` | 0.9837 | 0.9694 | +0.9979 | |

`E[⟨|φ|²⟩]` = 1.0005 / 1.0052 / 1.0005 on the three default arms, so the deterministic normalisation
survives the anisotropic and odd-N crops.

**Capped-pad regime** (N=32, σ = L = 32 µm): both warnings fire, the padded grid is `32 → 128` (the
4× cap), the code's own `_periodised_gaussian_error` predicts a residual of **0.0097** and I measure
max\|μ−Gaussian\| = **0.0042** — the quoted bound is honest and conservative.

**Downstream, analytic-GSM oracle** (`J(r₁,r₂) = √(I₁I₂)·exp(−Δ²/2σ_g²)`, N=24, dx=2.5 µm, w₀=18 µm,
σ_g=7 µm, 30 000 realisations): the normalised complex degree of coherence at the largest separation
reads **0.0048** after (Monte-Carlo floor 1/√K = 0.0058) against **0.9382** before; the analytic value
is 2.2e-15.  (The global `max|J − J_analytic|/max|J_analytic|` is 0.0115 after vs 0.0079 before —
both are Monte-Carlo noise on the bright diagonal, where the two arms are statistically identical;
the discriminating quantity is the off-diagonal μ above.)

### 3.3 Z3 — memory estimate on a different lens, N = 32…1024, both dtypes

N-SF11 meniscus (R1 = 40 mm, R2 = 120 mm, d = 4 mm, 20 mm aperture), 532 nm, 24 mm field,
`_lens_real.py` sha256 `35fbbcab0defb379` (413 484 bytes — a **later** revision than the WP's
`737e62954a4de29c` / 412 308 bytes, i.e. WP-A2 moved again after the WP measured):

| N | dtype | measured peak | estimate | est/meas | B/px |
|---|---|---|---|---|---|
| 32 | c128 | 0.161 MB | 0.193 MB | 1.198 | 157.6 |
| 64 | c128 | 0.587 MB | 0.773 MB | 1.318 | 143.3 |
| 128 | c128 | 2.271 MB | 3.093 MB | **1.362** | 138.6 |
| 256 | c128 | 12.182 MB | 12.373 MB | **1.016** | 185.9 |
| 512 | c128 | 46.2 MB | 49.5 MB | 1.072 | 176.2 |
| 1024 | c128 | 184.6 MB | 198.0 MB | 1.072 | 176.1 |
| 32…1024 | c64 | — | — | 1.007–1.167 | 120.0–127.9 |

The estimate **bounds** the measurement at every point (the fail-safe direction), the worst looseness
is 1.362 (inside the shipped test's 1.6 bar), and the tightest is 1.007 at N=32/c64.  `parallel_amp`
is confirmed inert for `'real'` in whole-grid mode; the `'traced'` model is unchanged.  The WP's
"pure N², 1.4 % spread" claim holds only for N ≥ 512 (c128 B/px runs 138.6 → 185.9 below that), but
since the deviation goes the safe way it does not threaten the contract.

### 3.4 Z3 — FreeSpace on non-4f chains

| chain | \|A\| | `dx_out/dx_in` | warnings | power |
|---|---|---|---|---|
| 4f inverter | 1.0000 | 1.0000 | 0 | 1.0000 |
| 2f–2f imager | 1.0000 | 1.0000 | 0 | 0.9644 |
| **1:2 imager** `FS(3f)·L(f)·FS(1.5f)` | **2.0000** | **1.0000** | 0 | 0.9943 |
| **2:1 imager** `FS(1.5f)·L(f)·FS(3f)` | **0.5000** | **1.0000** | 0 | 0.7859 |
| bare 50 mm drift | 1.0000 | 1.0000 | 0 | 1.0000 |
| bare 2 m drift | 1.0000 | 1.0000 | 0 | **0.1267** |

Under the pre-WP `'auto'` default the same magnifying chains delivered `dx_out/dx_in` = **11.59×** and
**23.18×**.  Anamorphic threading holds for `asm` / `auto` / `rs` at (2,3), (5,1), (2,2) µm.
`from_prescription`'s singlet delivers `dx_out/dx_in = 1.000000` with 0 warnings, as the WP claims.

### 3.5 Z4 — adversarial sweeps

* `user_library`: 22 expressions.  Refused (0.007–0.05 ms each, no allocation): `2**(10**9)`,
  `2**10**9`, `((2**2000)**2)**2`, `(3**1000)**(3**1000)`, `1<<(1<<40)`, `(1<<4000)<<200`, `2**4096`,
  `(10**100)**(10**100)`, `-2**(10**9)`, `(-2)**(10**9)`.  Allowed: `(2**2000)**2` (4001 bits),
  `1<<4000`, `2**2048`, `(-1)**(10**9)`, `2**3**4`, `2**(0-5)`, `0**(10**9)`, `1**(10**9)`,
  `(2**100)*(2**100)`, and every real phase-mask expression I tried.
* `_check_2d_scalar_field`: 18 dtypes, all correct; `np.matrix` **subclasses** rejected too.
* `_deprecation`: all five helpers plus the alias shim name the caller's **exact line** (verified
  against a file with known line numbers, not just the filename).
* `from_prescription`: attribute, `from … import`, dotted-module import, `importlib.import_module`,
  submodule-imported-first in a fresh subprocess, and `pickle` round-trip all behave; the chained
  `…from_prescription.from_prescription` tail is gone as documented; `algebra.__all__` is untouched
  and `test_v4_16_0_walker_all_symmetry.py` does **not** name `algebra` among its failures.

---

## 4. Defects found in the WP's files, and fixed here

### 4.1 `stokes_parameters` was NOT bit-identical for a mixed-precision `JonesField` (P3, fixed)

`lumenairy/elements/polarization.py:1355`.  `JonesField` coerces a real/integer input to complex but
leaves a `complex64` one alone, so `JonesField(Ex_c128, Ey_c64, dx)` is constructible.  The new lean
body wrote the product into a buffer made from `Ey`:

```python
cross = np.conj(Ey)
np.multiply(Ex, cross, out=cross)      # complex128 product -> complex64 buffer
```

NumPy's default `same_kind` casting silently rounds the complex128 product down.  **Measured before
the fix** (N=17, Ex c128 / Ey c64): S2 and S3 came back **float32** with max \|diff\| 2.38e-7 / 3.38e-7
against the four-expression form (relative 2.1e-8 / 2.9e-8 = float32 eps); S0/S1 stayed float64, so
the returned dict had mixed dtypes.  `degree_of_polarization` inherited it.  The WP's bit-identity
test parametrises one uniform dtype at a time and cannot see this.

Fix: build the buffer in the promoted dtype —
`cross = np.conj(Ey).astype(np.result_type(Ex, Ey), copy=False)`.  `astype(copy=False)` allocates
nothing when the components already share a dtype, so the memory win is unchanged.

**Verification.** All nine dtype/layout cases (c128, c64, c128/c64, c64/c128, non-contiguous, strided
view, real, int, pathological with dark/NaN/inf/1e-160 pixels) are now bit-identical for both
`stokes_parameters` and `degree_of_polarization`; `tracemalloc` peaks are unchanged at **6.00** /
**6.00** full-grid real arrays at N = 1024 **and** 2048 (pre-fix arms 7.00 / 8.25).
Pin: `::test_verify_a11_stokes_bit_identical_for_a_mixed_precision_field`.

### 4.2 `_clearer_identity` made every `functools.partial` compare equal (P3, fixed)

`lumenairy/_cache_registry.py:54`.  Its docstring said callables with no code object "fall back to
their type name, which is coarse but **errs towards warning rather than silence**".  Measured, it
errs towards **silence**: `functools.partial` exposes neither `__module__` nor `__qualname__` nor
`__code__`, so every partial keyed to `(None, None, 'partial', None)` — a partial of `clear_a` and a
partial of `clear_b` registered under one name collided with **no warning**, which is precisely the
silent-drop defect Z4 exists to make audible.

Fix: unwrap `functools.partial` to the callable it wraps (bounded at 8 levels), and fall back to the
class's `__call__` code object for callable instances.  Two *instances* of the same class (and two
bound methods of the same class) still compare equal — deliberate, documented, and the price of
reload-idempotence, since nothing about an instance survives a reload.

**Verification.** Nine-case table, all correct after the fix (two distinct functions → warn; same
function twice → silent; two lambdas on the same line → silent; on different lines → warn; two
partials of the same function → silent; **of different functions → warn**; two callable instances →
silent; bound methods → silent; builtin vs function → warn).  A real `importlib.import_module` +
two `importlib.reload`s of a module that registers still emits **0** warnings.
Pin: `::test_verify_a11_cache_registry_separates_partials_of_different_functions`.

### 4.3 The new PITCH CONTRACT docstring was false for `|A| ≠ 1` (P3, fixed)

`lumenairy/algebra/base.py:172`.  The note read: with a pitch-preserving propagator "the delivered
`dx_out` equals the input `dx`, so the two agree and **`|A|` is the grid magnification as well as the
ray one**."  Measured (§3.4): the 1:2 imager reports `|A| = 2` and the 2:1 imager `|A| = 0.5`, and
**both deliver `dx_out/dx_in = 1.0000`**.  `|A|` is the grid magnification only in the special case
`|A| = 1` — which is the only case the WP measured (the 4f inverter).  A reader who followed the
sentence on a magnifying chain would compute the wrong output pitch, i.e. the same class of error the
finding is about.  Reworded with the two measured counter-examples; pinned as behaviour (not prose)
by `::test_verify_a11_freespace_pitch_is_preserved_whatever_the_abcd_says`, which brackets unity so a
re-assertion of the old claim cannot pass on either side.

### 4.4 Two smaller documentation defects (fixed)

* `lumenairy/algebra/primitives.py:86` referenced `:meth:`Operator.apply_with_grid`` — **no such
  method exists** anywhere in the repo (only one hit, this line).  `Operator.apply` returns a bare
  ndarray and *discards* the pitch, so the sentence was wrong twice.  Rewritten to name
  `Operator.__call__` and to warn that `apply` drops the pitch.
* `lumenairy/elements/polarization.py:1423` claimed "the peak drops from 8.25 to **5.00** full-grid
  real arrays at N = 2048".  Measured **6.00** at N = 1024 and 2048 (the changelog's table is right;
  the in-source comment was not).  This is TESTING_STANDARDS' "right-conclusion-wrong-numbers" shape
  in a load-bearing comment; corrected with the measurement and date.

### 4.5 My changes, complete list

| file | change | lines |
|---|---|---|
| `lumenairy/elements/polarization.py` | §4.1 promoted-dtype cross buffer; §4.4 corrected 5.00 → 6.00 | +14 −2 |
| `lumenairy/_cache_registry.py` | §4.2 partial / callable-instance identity + honest docstring | +30 −4 |
| `lumenairy/algebra/base.py` | §4.3 PITCH CONTRACT wording with the measured counter-examples | +10 −1 |
| `lumenairy/algebra/primitives.py` | §4.4 dangling `apply_with_grid` reference | +5 −1 |
| `tests/unit/test_audit2609_a11_polar_sources_infra.py` | **5 new VERIFY-A11 pins** (172 lines), appended in a marked section; nothing existing altered | +172 |

No other file touched.  No git write commands run.

---

## 5. Open items for the orchestrator

**O-1 (P2, design decision — needs a ruling).  `FreeSpace`'s new `'asm'` default is silently wrong in
the far field, with no diagnostic.**  The audit offered two remedies; the WP took "default to `'asm'`"
as the honest one.  It is honest about the *pitch* and silent about the *truncation*.  Measured at
N = 256, dx = 8 µm, λ = 633 nm (window ±1.016 mm):

| fixture | analytic w(z) | `'auto'` (the old default) | `'asm'` (the new default) |
|---|---|---|---|
| w₀ = 20 µm, z = 500 mm | 5037.3 µm | **5034.0 µm (0.07 % err)**, power 0.9998, dx_out 77.27 µm | **1071.0 µm (78.7 % err)**, power **0.0961**, dx_out 8.00 µm |
| w₀ = 6 µm, z = 200 mm | 6716.3 µm | 4212.0 µm (37 % err), power 0.4950 | 1129.3 µm (83 % err), power 0.0466 |

**Zero warnings in every cell.**  The old default produced the right field on a resampled grid and
emitted an (un-actionable) return-contract warning; the new one produces a field that has lost 90 % of
its energy to the window and says nothing.  §3.4's "bare 2 m drift" row shows the same thing on a
one-operator chain (power 0.1267).  The regime is entered whenever `z ≳ N·dx²/λ` (25.9 mm for this
grid) — a cheap, derived criterion.

Recommendation (one of):
1. keep `'asm'` and have `FreeSpace._apply` emit a `UserWarning` once per operator when
   `z > N·dx²/λ`, naming `method='auto'` as the fix; or
2. keep `'asm'` but state the truncation explicitly in the `versionchanged:: 5.46` block and the
   migration note (currently neither mentions it); or
3. revert to `'auto'` and re-sample to the ABCD-implied pitch (the audit's option (b)).

I did not implement any of these: it changes a default the WP deliberately chose, and (1) touches the
warning surface other WPs are also editing.

**O-2 (P3).  `estimate_lens_memory(lens_model=...)` is an unvalidated enum.**  `_real = (lens_model
!= 'traced')`, so `'Traced'`, `'REAL'`, `'banana'`, `''`, `None` and `0` all silently select the
`'real'` model (measured: all return 49.5 MB at N=512/c128, identical to `'real'`).  The same WP
added exactly this validation for the coatings polarization enum (Z1) one file away; `memory.py` is
WP-A11's.  A `ValueError` with the §2 prefix naming `{'traced','real'}` is the consistent fix.  I did
not add it because it raises on previously-accepted input while other agents are mid-flight.

**O-3 (P3, informational).  The Z2 `σ_g > L/6` advisory fires inside the library's own test suite.**
Seven `UserWarning`s across `test_v4_15_2_agent_a.py` (σ_g = L/4 fixtures) and `p12_final.py`.  They
are correct advisories and nothing fails, but if the warning proves noisy in practice the natural
narrowing is to fire it once per process per (N, dx, σ_g) class rather than per call.

**O-4 (P3, nit).  `_normalize_pol`'s `str()` coercion is inherited by coatings.**  `_normalize_pol`
does `str(polarization).lower()`, so an *object* whose `__str__` returns `'te'` is accepted by
`coating_reflectance` (measured: returns the s result).  Harmless and pre-existing in the RCWA helper
— but the WP's docstring says the accepted set is `{'s','te','p','tm','avg'}` *strings*.  Only worth a
line if the house wants strictness; fixing it must change both helpers together (the equivalence test
will hold them to it).

**O-5 (P3, nit).  Two test-side spellings.**
* `test_s3_7_broadcast_grid.py::test_schell_phase_realizations_kxky_bit_identical_padded` hard-codes
  `4.0` for the pad instead of importing `_SCHELL_PAD_SIGMA`, so retuning the constant fails the test
  for an unrelated reason.
* `test_v4_15_4_agent_b.py::_is_sources_core` matches `node.module in ('lumenairy.sources.core',
  'sources.core')` without checking `node.level`, so a hypothetical absolute `from sources.core
  import …` would also satisfy it.  The pin's real contract (module scope, not lazy) is unaffected.

**O-6 (confirmation, not a new item).  The WP's §5 requests are all correct as filed.**  I
re-measured them:
* `test_niche_audit_w3_infra.py::TestA6EstimateAsmMemory[512-complex128]` fails on the *looseness*
  fence (`est/measured = 1.3825 ≰ 1.35`), while the safety assertion `ratio >= 1.0` still passes; the
  WP touched no ASM constant (`git diff … -- memory.py | grep _ASM` is empty).  Another WP's lazy-scipy
  work lowered the cold peak.  Either the constant comes down or the Windows fence goes up — one
  decision, one place, as the WP says.
* `lumenairy/__init__.py` re-export of `from_prescription` and the walker exemption: the shadowing fix
  stands on its own without them; `algebra.__all__` is correctly left alone and the walker's 18
  failures name only other WPs' modules.

---

## 6. Collateral damage

Test files run (all with `OPENBLAS_NUM_THREADS=1 -q --no-header -p no:cacheprovider`):

| batch | result | time |
|---|---|---|
| `test_audit2609_a11_polar_sources_infra.py` (as shipped, before my additions) | **85 passed** | 7.6 s |
| the six existing test files the WP modified | **129 passed** | 27.9 s |
| 21 files covering coatings / sources / polarization / user_library / cache / memory | 666 passed, **2 failed** | 57.4 s |
| 15 files covering algebra / deprecation / validation / walker / entry-validation | 671 passed, **2 failed** | 30.1 s |
| 10 files re-run **after** my §4 fixes (polarization + cache-registry consumers) | 272 passed, 1 skipped, **1 failed** | 57.0 s |
| final gate: A11 file (90 tests) + the six modified files + 3 more | **304 passed** | 110.7 s |
| `validation/run_all.py test_coherence test_polarization` | **2/2 PASS** | 11.3 s |
| `validation/run_all.py test_sources test_elements` | **2/2 PASS** | 45.7 s |

**Four distinct failures, none attributable to WP-A11 — each verified by measurement, not assumption:**

| test | cause | evidence |
|---|---|---|
| `test_v5_4_6_wave4_polarization.py::test_vector_aperture_diffraction_has_projection_kwarg` | `propagators/vectorial_hfpi.py` (+446 uncommitted lines, another WP) flipped `vector_projection` to `True`; the pin asserts `False` | failure text names only that parameter's default |
| `test_lens_memory_levers.py::test_the_banded_seed_reproduces_the_whole_grid_momentum_field_exactly` | `elements/_lens_real.py` (+1363 uncommitted lines, another WP): `_screen_obliquity_rows_any` no longer matches `_screen_obliquity_angle_field` for `carrier='auto', n_medium=1.62` | the assertion compares two `_lens_real` internals; no WP-A11 symbol involved.  **Not in the WP's own §4.1 list** — it appeared after the WP ran |
| `test_niche_audit_w3_infra.py::TestA6EstimateAsmMemory[512-complex128]` | another WP's lazy-scipy import cut the cold ASM peak | the `ratio >= 1.0` contract passes; only the 1.35 looseness fence trips; WP-A11 touched no ASM constant |
| `test_v4_16_0_walker_all_symmetry.py::test_all_submodule_entries_reexported_or_exempt` | 18 unexported names, all in `analysis`, `optimize`, `pmm.twod`, `raytrace*` | **none** names `algebra` or `from_prescription`; the WP's rebind does not trip the walker |

The audit's "checked and found correct" list survives: `dy` threading in all 12 factories, coatings
Airy/Rouard agreement (now re-derived with a *third* independent TMM), the absorbing-exit-medium
p-transmittance, the TIR branch, the normal-incidence r_s/r_p sign, the GSM unit-mean-intensity
normalisation, `algebra` ABCD vs `system_abcd` at 0.0, `cache.py` byte accounting, `_context`
nesting/rollback, and the `user_library` sandbox.

---

## 7. Summary for the orchestrator

* WP-A11's Z1–Z4 work is **sound and the numbers in its report are accurate** — I could not find a
  claimed measurement that failed to reproduce, and several reproduce to the printed digit.
* Two real defects were introduced by the WP's own performance rewrites and are **fixed here with
  pins**: a silent float32 downgrade of S2/S3 for mixed-precision `JonesField`s, and a cache-registry
  identity fallback that made all `functools.partial`s collide silently.  Two documentation defects
  (a PITCH CONTRACT sentence that is false for `|A| ≠ 1`, and a dangling `apply_with_grid` reference)
  are also fixed, plus one wrong number in a source comment.
* **One item needs a decision before release: O-1**, the far-field silence of the new `'asm'`
  default.  It is the only place where this WP's change can produce a silently wrong physical result
  where the previous default produced a right one.

---

## 8. O-1 / O-2 resolution (orchestrator ruling of 2026-09-12, implemented)

### O-1 — keep `method='asm'`, make the truncation audible

Implemented in `lumenairy/algebra/primitives.py`.  The default is **unchanged** (`'asm'`) and
**nothing resamples by default** — the pitch contract is the audit finding and it stays intact.

**One deviation from the literal wording of the ruling, with the measurement that forced it.**  The
ruling says to warn "when `z > N·dx²/λ`".  Implemented exactly that way, the audit's own 4f fixture
(f = 200 mm, N = 256, dx = 8 µm) emits **3 `UserWarning`s per evaluation** — every one of its five
legs is past `z_max = 25.9 mm` — which puts the Z3 headline metric back at exactly the number it was
fixed to remove (3 → 0).  Measured:

```
--- the 4f chain, warning on the gate alone ---
  4f (f=200 mm, z_max=25.9 mm): 3 warning(s)
```

`z > z_max` says the step *could* truncate, not that it did: a converging or refocused segment sails
past it untouched.  So the gate now **arms a measurement** and the measurement decides — a
pitch-preserving step that hands back less than 95 % of the power it was given has demonstrably lost
it off the window edge (these kernels are otherwise unitary).  Per-leg power kept, measured at
N = 256, dx = 8 µm, 633 nm:

| segment | P_out/P_in | true w(z) vs ±1024 µm window | verdict |
|---|---|---|---|
| 4f leg 1, w₀ = 100 µm, z = f = 200 mm | **1.00000** | 415 µm | benign |
| 4f leg 2, w₀ = 100 µm, z = 2f = 400 mm | **0.98239** | 812 µm | benign (tightest) |
| collimated w₀ = 200 µm, z = 500 mm | 0.99990 | 542 µm | benign |
| w₀ = 400 µm, z = 26 mm (just past `z_max`) | 1.00000 | 400 µm | benign |
| w₀ = 20 µm, z = 500 mm (**the O-1 fixture**) | **0.09612** | 5037 µm | truncated |
| w₀ = 100 µm, z = 2 m | **0.12669** | 4031 µm | truncated |
| w₀ = 6 µm, z = 200 mm | **0.04658** | 6716 µm | truncated |

`_FAR_FIELD_POWER_LOSS_TOL = 0.05` sits **2.8× above** the worst benign loss (1.76 %) and **17×
below** the smallest real one (87.3 %) — a two-sided gap with the derivation and the date in the
source.  The gate keeps the `O(N²)` power sum off the hot path: it is paid only by a pitch-preserving
step past `z_max` that has not already warned.

What ships:

* `FreeSpace._far_field_gate` — `z_max = min(Nx·dx², Ny·dy²)/λ` from the **delivered** grid (per axis,
  the more restrictive window, so an anamorphic grid is judged on whichever axis runs out first).
* `FreeSpace._warn_if_far_field_truncates` — fires after the propagation, `UserWarning`, §2 prefix
  `FreeSpace._apply: …`, quoting `z`, the resolved method, the grid, `z_max`, the **measured**
  fraction kept, and the ±half-width window; carries the 0.096-vs-0.9998 measurement in the message
  itself.
* Remedy named in the message: `method='auto'` (resampling, correct far field, at the cost of a grid
  the ABCD does not describe) — except on the anamorphic branch, which *forces* `'asm'` because the
  resampling kernels are square-grid-only, where the message instead says to enlarge `N·dx` or to
  propagate in stages.
* **Once per operator instance** (`self._far_field_warned`, initialised in `__init__`, read through
  `getattr` so a pre-v5.46 unpickled instance cannot `AttributeError`).  An optimiser loop that
  re-applies one `FreeSpace` thousands of times gets one warning.
* The `versionchanged:: 5.46` block now states the truncation explicitly, with the measured
  9.6 %-of-power figure, the 79 %-low radius, and the `z > N·dx²/λ` onset.

Measured behaviour of the shipped form:

| case | warnings | note |
|---|---|---|
| 4f chain (the Z3 fixture) | **0** | `dx_out/dx = 1.0000`, power 1.0000 |
| w₀ = 20 µm, z = 500 mm | **1** | power 0.0961 |
| w₀ = 100 µm, z = 2 m | **1** | power 0.1267 |
| four benign legs, all past `z_max` | **0** | incl. the 4f 2f-leg at 0.98239 |
| one instance × 4 evaluations | **1** | once-per-instance holds |
| three fresh instances | **3** | |
| `method='auto'` / `'sas'` / `'fresnel'` | **0** | the recommended kernels never trigger their own advice |
| `method='rs'` (pitch-preserving) | **1** | |
| `FourierTransform(f)` (pins `'auto'`) | **0** | unaffected |
| anamorphic 128×128, dx=2 µm, dy=3 µm, z = 20 mm | **1** | power 0.1994, pitch still (2, 3) µm, remedy text switches |
| anamorphic, same grid, z = 2 mm | **0** | 2.5× past `z_max`, no clipping |

One consequence worth recording: the **2:1 imager row of my own §3.4 table truncates** (power 0.7859
at N = 256), so the new warning fires there — correctly.  I re-based the §4.3 pin onto a chain sized
to stay inside the window (N = 512, dx = 8 µm, f = 50 mm, w₀ = 300 µm; power kept 1.0000 on both the
`|A| = 2` and `|A| = 0.5` arms) so that it pins the **pitch** claim and the O-1 test pins the
**clipping** claim, with neither conflating the other.  The pin now also asserts its own premise
(power within 2 % of 1) so it cannot silently become a truncation test.

### O-2 — closed vocabulary for `estimate_lens_memory(lens_model=…)`

Implemented in `lumenairy/memory.py`: `_LENS_MODELS = frozenset({'traced', 'real'})` and a
§2-prefixed `ValueError` at the top of the function, **before** the row-band branch (which reads the
same argument).  Case-sensitive, matching the lower-case strings the module compares against
throughout.

| `lens_model=` | before | after |
|---|---|---|
| `'traced'` | 47.1 MB | 47.1 MB (unchanged) |
| `'real'` | 49.5 MB | 49.5 MB (unchanged) |
| `'Traced'`, `'REAL'`, `'Real'`, `'banana'`, `''`, `'real '`, `' traced'`, `None`, `0`, `True` | **49.5 MB — silently the `'real'` model** | **`ValueError`** naming `['real', 'traced']` |

No library or test call site passes anything else (checked `lumenairy/`, `tests/`, `validation/`,
`examples/`; the `lens_model=` occurrences in `_lens_thin.py`, `propagators/system.py`,
`ui/waveoptics_dock.py` and `test_audit_lens.py` are the *thin-lens* parameter of the same name —
`'paraxial'` / `'stigmatic'` / `'aplanatic'` / `'asm'` / `'local_only'` — a different knob).

### Pins added (13)

| test | what it pins |
|---|---|
| `::test_o1_freespace_reports_far_field_truncation_and_stays_quiet_otherwise` | warns on the truncating fixture (prefix, `method='auto'`, `z_max` in the message); **silent on four benign legs that are all past `z_max`**; the 4f fixture stays at 0 warnings with `dx_out == dx` and power 1.0000 |
| `::test_o1_far_field_warning_is_once_per_instance_and_skips_the_remedy` | 1 warning over 4 evaluations of one instance, 3 over three fresh ones; `'auto'`/`'sas'`/`'fresnel'` never warn, `'rs'` does; anamorphic branch warns, keeps (2, 3) µm, and switches the remedy text |
| `::test_o2_estimate_lens_memory_rejects_an_unknown_lens_model` (×10) | every wrong token raises with the §2 prefix, case-sensitively |
| `::test_o2_estimate_lens_memory_keeps_both_valid_models_distinct` | counter-pin: the guard does not collapse the vocabulary — `'traced'` ≠ `'real'`, the default is still `'traced'`, and the row-band branch validates too |

### Test runs after the ruling

| command | result | time |
|---|---|---|
| `pytest tests/unit/test_audit2609_a11_polar_sources_infra.py` | **103 passed** | 21.9 s |
| 14 files: memory guardrail, w3 infra, 4× agent_g algebra, agent_b ×2, s11, w4 p5 return contract, g1 cache memory, s3_7 broadcast, v4_15_2_agent_a, v4_15_4_agent_b | 419 passed, **1 failed** | 176.7 s |

The single failure is the already-attributed
`test_niche_audit_w3_infra.py::TestA6EstimateAsmMemory[512-complex128]` (another WP's lazy-scipy
import lowered the cold ASM peak; the `ratio >= 1.0` safety assertion still passes, only the 1.35
looseness fence trips — see O-6).  `test_lens_memory_levers.py`'s `_lens_real` failure, also another
WP's, is unchanged and unrelated.

### Files touched for O-1 / O-2 (added to §4.5)

| file | change | lines |
|---|---|---|
| `lumenairy/algebra/primitives.py` | O-1 gate + measured-truncation warning, `_far_field_warned` flag, `versionchanged` truncation paragraph | +120 −5 |
| `lumenairy/memory.py` | O-2 `_LENS_MODELS` + §2-prefixed `ValueError` + docstring `Raises` | +30 −1 |
| `tests/unit/test_audit2609_a11_polar_sources_infra.py` | 13 new O-1/O-2 pins; §4.3 pin re-based onto a non-truncating fixture | +135 −6 |

**O-3 / O-4 / O-5 left as recorded, per the ruling — no code change.**
