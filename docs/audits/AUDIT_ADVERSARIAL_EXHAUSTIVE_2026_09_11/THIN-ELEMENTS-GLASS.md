# THIN-ELEMENTS-GLASS audit — thin-lens / sag / DOE / freeform / grating / EMT / BSDF / glass catalogue

Environment: CPython 3.14, numpy 2.4.6, numba 0.65 (installed → the numba sag path is LIVE),
**`refractiveindex` IS installed** (contrary to the brief's "NOT installed" list — this changes
which glass dispatch branches are exercised and is load-bearing for finding 2).
All scratch under `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/THIN-ELEMENTS-GLASS/`.

## Scope read (files + line ranges actually read)

| file | lines read | notes |
|---|---|---|
| `lumenairy/elements/_lens_thin.py` | 1–1372 (all) | every branch of all 6 entry points |
| `lumenairy/elements/lenses.py` | 1–600, 600–1025 | incl. numba kernel, sag helpers, grid recommenders, 4-D poly evaluators, re-export block |
| `lumenairy/elements/elements.py` | 1–1227 (all) | mirror, apertures, Zernike, coronagraph, turbulence |
| `lumenairy/elements/doe.py` | 1–1241 (all) | |
| `lumenairy/elements/freeform.py` | 1–739 (all) | |
| `lumenairy/elements/thin_grating.py` | 1–223 (all) | |
| `lumenairy/elements/emt.py` | 1–316 (all) | |
| `lumenairy/elements/materials.py` | 1–104 (all) | |
| `lumenairy/elements/coronagraph.py` | 1–54 (all) | |
| `lumenairy/elements/segment_geometry.py` | 1–574 (all) | |
| `lumenairy/elements/bsdf.py` | 1–603 (all) | |
| `lumenairy/glass.py` | 1–340, 520–900, 997–1250, 1250–1747 | skimmed only the 400–520 POLYNOMIAL rows and 900–997 GLASS_VALIDITY table bodies (data, not logic) |
| `lumenairy/elements/__init__.py` | 1–80, 240–297 | re-export hub |

**Not reached:** `lenses.py` 600–760 (MASLOV-GBD-FGA's), the per-row bodies of
`POLYNOMIAL_COEFFICIENTS` (Hikari/Sumita) and `GLASS_VALIDITY` (I checked the
Sellmeier rows numerically instead — see F1 — but did NOT numerically validate the
24 formula-3 polynomial rows against the catalogue; that is the single biggest
remaining gap in this partition). `apply_mirror` read but not focus-tested.

---

## Findings

### **[P0] Three bundled Sellmeier rows are a DIFFERENT GLASS, and they are the only dispatch path** — `lumenairy/glass.py:122-123` (N-BAF52), `:130-133` (N-LAK33A / N-LAK33B)

All three names are registered `'__sellmeier__'` (`glass.py:755, 758, 759`), and the
`__sellmeier__` branch in `get_glass_index` (`:1475-1484`) fires **before** any
refractiveindex.info lookup — so the bundled row is what every install returns,
`refractiveindex` present or not.

Measured against the refractiveindex.info SCHOTT-optical catalogue (in-process oracle,
repro `p_glass3.py`), `n_d` at 587.5618 nm and max |Δn| over a 61-point scan 0.40–1.60 µm:

| glass | n_d bundled | n_d catalogue | Δn_d | V_d bundled | V_d cat | max abs Δn (0.4–1.6 µm) |
|---|---|---|---|---|---|---|
| **N-BAF52** | 1.637147 | 1.608631 | **+2.85e-2** | 42.469 | 46.597 | **3.22e-2** |
| **N-LAK33A** | 1.754279 | 1.753930 | +3.49e-4 | 53.031 | 52.271 | **2.76e-3** |
| **N-LAK33B** | 1.755294 | 1.755000 | +2.94e-4 | 52.940 | 52.300 | **2.33e-3** |

The other **46** bundled rows agree with the catalogue to ≤ 4e-6 in n_d and ≤ 3.3e-5
max |Δn| (N-LASF40 the worst), so the table is otherwise excellent — these three stand out.

The bundled "N-BAF52" row is within 6e-4 of SCHOTT **N-KZFS11** (n_d 1.637750, V_d 42.41),
i.e. it looks like a mis-copied neighbouring catalogue row — the *identical* failure mode
the v4.11.2 round-3 audit found and fixed for S-LAH64 / S-LAH79 (`glass.py:184-194`:
"The in-code coefficients appear to be misattributed from a different glass"). That sweep
did not re-check the rest of the table. Grep of `docs/`, `CHANGELOG.md` and `tests/`
finds no mention of N-BAF52 or N-LAK33 as a known issue.

**Impact.** 0.0285 of index is 1.8 %; on a 20 mm-radius singlet that is a ~1.9 % focal-length
error and a wholly wrong V_d (42.47 vs 46.60) for any achromat design. N-LAK33A/B's
2.8e-3 max error is ~50× the 5e-5 cross-check tolerance the table's own comments claim.
Silent on every path.

**Fix.** Replace the three rows from the refractiveindex.info YAML (or delete them and
register as `('specs','SCHOTT-optical',name)` tuples, as v4.11.2 did for the Ohara pair).
Then close the class: `_check_glass_registry_consistency` (`glass.py:1051`) currently checks
only key PRESENCE in both directions — add a *value* cross-check (n_d to 5e-5, plus V_d)
as a test-time gate over the whole table; the loop is ~40 lines and would have caught this.

---

### **[P1] `get_glass_index_complex` RAISES instead of the documented κ=0 fallback, for exactly the common bulk materials** — `lumenairy/glass.py:1651-1659`

With `refractiveindex` installed (the recommended `pip install lumenairy[glass]`), a
catalogue page carrying no k-data raises `refractiveindex.refractiveindex.NoExtinctionCoefficient`.
Its MRO is `('NoExtinctionCoefficient', 'Exception', 'BaseException', 'object')` — it
subclasses `Exception` **directly**, so it is not in the caught tuple
`(AttributeError, NotImplementedError, KeyError, ValueError, TypeError)`.

Measured over all 44 tuple-registered glasses at 1.31 µm (repro `p_glass2.py`): **7 raise** —
`CaF2`, `FUSED_SILICA`, `F_SILICA`, `MgF2`, `SILICA`, `SILICON`, `SiO2`. That is every
`main`-shelf entry, i.e. fused silica and the IR window materials.

The docstring (`:1602-1604`) promises "the extinction is looked up via the database when
available and falls back to ``kappa = 0`` otherwise", and `_warn_missing_kappa_once`
(`:1667`) exists for precisely this case — it is unreachable on this path.

The prior audit reviewed this exact line and cleared it:
`docs/audits/AUDIT_GLASS_POLARIZATION_2026_07_08.md:45-48` —
*"`get_glass_index_complex` — the refractiveindex-unavailable tuple path lands on
`_glass_cache[name]` → `KeyError` → caught by the (correctly broad) except → warn + κ=0.
Intentional-looking and safe."* Only the package-**absent** path was exercised.

**Fix.** Catch `Exception` on that try (the package's own class cannot be named without a
hard dependency), or widen the tuple via
`getattr(_ri_mod, 'NoExtinctionCoefficient', ())`. Keep the existing warn-once.

---

### **[P1] `generate_turbulence_screen` delivers 2× the requested phase power (the spurious √2)** — `lumenairy/elements/elements.py:1219`

```python
amplitude = np.sqrt(2.0 * psd) * df
```
With `c_k = (a_k + i b_k)·A_k` (a, b independent N(0,1)) and
`φ(x) = Re(Σ_k c_k e^{iθ_k})`, `Var(φ) = Σ_k A_k²` — there is **no** ½ from taking the real
part, because the real and imaginary noise draws are independent. The correct amplitude is
`sqrt(PSD)·df` (Schmidt 2010, `ft_phase_screen`); the extra √2 doubles both the variance and
the structure function.

Measured (N=512, dx=5 mm, r0=0.1 m, 60 seeds; repro `p_turb.py`). `D_lat(×1)` is the exact
expected structure function of the code's own discrete lattice *without* the √2:

| r [m] | D_meas | D_lat(×1) | D_meas/D_lat(×1) | 6.88(r/r0)^{5/3} | D_meas/D_Kolm |
|---|---|---|---|---|---|
| 0.005 | 0.0740 | 0.0372 | **1.987** | 0.0467 | 1.584 |
| 0.010 | 0.2398 | 0.1208 | **1.984** | 0.1482 | 1.618 |
| 0.020 | 0.7309 | 0.3693 | **1.979** | 0.4706 | 1.553 |
| 0.040 | 2.1607 | 1.0961 | 1.971 | 1.4940 | 1.446 |
| 0.080 | 6.1848 | 3.1547 | 1.961 | 4.7432 | 1.304 |
| 0.160 | 16.898 | 8.6946 | 1.944 | 15.059 | 1.122 |
| 0.320 | 42.678 | 22.305 | 1.913 | 47.809 | 0.893 |

Ratio → **2.000** as r→0 (where lattice truncation vanishes). Screen variance 62.64 vs the
lattice sum ΣPSD·df² = 33.82. The continuum target was verified independently:
`2∫PSD(1−J₀(2πfr))d²f / [6.88(r/r0)^{5/3}]` = 0.989 / 0.978 / 0.966 at r = 0.02 / 0.1 / 0.32 m,
so 6.88 is the right constant and 0.023 is the right f-space (cycles/m) coefficient.

The in-code claim at `:1216-1218` ("The sqrt(2) factor compensates for the halving of
variance ... Verified against the Kolmogorov structure function D(r=r0) = 6.88") is false:
the agreement it cites happens only near r ≈ 3.2 r0, where the 2× excess crosses the FFT
screen's own low-frequency deficit (D_lat/D_Kolm = 0.80 at small r, 0.47 at r = 3.2 r0).

`docs/audits/AUDIT_COATINGS_ELEMENTS_2026_07_09.md:40-48` reviewed this and wrote
*"the √2 real-part-variance factor is the right idea; I did not re-derive the (2π) constant
bookkeeping — flagged as verified-by-shape, calibration-by-citation."* — i.e. the audit
explicitly declined to check the number that is wrong.

**Impact.** Every AO / atmospheric run through this entry point gets **1.5× stronger
turbulence than requested at small separations** (effective r0 = r0/2^{3/5} = r0/1.516),
and φ_rms √2 too large, so Strehl ≈ e^{−σ²} is squared-wrong.

**Fix.** Drop the `2.0 *`. Separately (and secondary), add subharmonics
(Lane et al. 1992 / Schmidt `ft_sh_phase_screen`) — with the amplitude corrected the
residual lattice deficit is D/D_Kolm ≈ 0.80 at r = 0.05 r0 and 0.47 at 3.2 r0.

---

### **[P1] `surface_sag_general` silently DROPS the whole aspheric polynomial for a non-C-contiguous `h_sq`** — `lumenairy/elements/lenses.py:263-276`

The numba fast path is the default whenever numba is installed. `sag` inherits `h_sq`'s
memory order (`zeros_like` at `:218`, then `np.where` at `:230-235`); inside the kernel
`flat_s = sag.ravel()` on a non-C-contiguous array returns a **copy**, so
`flat_s[i] += acc` writes into a temporary that is thrown away. Only `h_sq` is passed
through `np.ascontiguousarray`, not `sag`.

Measured (repro `p_forder.py`), R = 50 mm sphere, `{4: 1e3}`:

| input layout | flags | max abs error vs reference | error / aspheric term |
|---|---|---|---|
| C-contiguous (4,6) | C=True | 0.0 | 0.000 |
| **F-order (4,6)** | F=True | **9.41e-6 m** | **1.000** |
| **transposed view `hsq.T`** | F=True | **9.41e-6 m** | **1.000** |
| C-copy of the transpose | C=True | 0.0 | 0.000 |

Recovering `(sag − conic)/A4` from the F-order call gives an array of **exactly 0.0** —
the aspheric contribution is entirely absent. No warning, no error.

The in-code comment at `:273-274` asserts the opposite:
*"In-place accumulate; sag is contiguous from xp.zeros_like above so `.ravel()` inside the
kernel is a view."* — true only for C-contiguous input.

**Exposure.** `surface_sag_general` is a public, module-level export (its docstring says
"used by both the lens and mirror modules, so it is exported at module level"), and
`freeform.py` (5 call sites), `elements.apply_mirror`, `_lens_real.py` and
`analysis/plotting.py` all feed it caller-supplied grids. Every in-repo call site I traced
currently builds `h_sq` from `np.meshgrid` arithmetic (C-order), so I could not demonstrate
an in-repo default-path failure — hence P1 rather than P0. A caller using
`indexing='ij'` + transpose, or passing `arr.T`, hits it immediately.

**Fix.** `sag = np.ascontiguousarray(sag)` before the kernel and assign the result back
(one line), or assert `sag.flags['C_CONTIGUOUS']` and fall back to the NumPy loop otherwise.

---

### **[P1] `apply_grin_lens` is 36 % wrong at the quarter pitch its own docstring recommends, with no guard** — `lumenairy/elements/_lens_thin.py:1247`

```python
phase = -k * n0 * (g ** 2 / 2) * d * r_sq        # ⇒ f = 1 / (n0 g² d)
```
The exact GRIN-rod ABCD EFL is `f = 1/(n0 g sin(g d))` — quoted verbatim in the docstring
two lines below the thin form (`:1214`), and immediately followed by
*"Quarter-pitch (g·d = π/2) collimates a point source at the front face; half-pitch
(g·d = π) reimages 1:1 inverted"* — i.e. the Notes invite exactly the regime the code
cannot model. Nothing warns; there is no `g*d` check anywhere in the function.

Measured (n0 = 1.6, g = 300 m⁻¹; repro `p_thinlens.py`):

| g·d | f_code [mm] | f_exact (ABCD) [mm] | f_code/f_exact | error |
|---|---|---|---|---|
| 0.05 | 41.667 | 41.684 | 0.9996 | −0.04 % |
| 0.30 | 6.944 | 7.050 | 0.9851 | −1.49 % |
| π/4 | 2.653 | 2.946 | 0.9003 | −9.97 % |
| **π/2 (quarter pitch)** | **1.326** | **2.083** | **0.6366** | **−36.34 %** |
| 0.9π | 0.737 | 6.742 | 0.1093 | −89.07 % |

ASM check at quarter pitch: the screen's measured peak-intensity plane is at **1.293 mm**,
tracking `f_code`, not the exact 2.083 mm. (`f_code/f_exact = sin(gd)/(gd)`, i.e. it
diverges as g·d → π, where the true power changes sign.)

**Fix.** Use the correctly-powered thin screen `φ = −k·n0·g·sin(g d)·r²/2` (same cost,
right paraxial power at any pitch), **and** raise or warn for `g*d > ~0.2`, **and** correct
the Notes so they stop recommending quarter-pitch for a model that cannot reach it.
(A single screen still cannot reproduce the rod's principal-plane separation; say so.)

---

### **[P2] `HarveyShackBSDF.total_integrated_scatter()` under-reads by up to 18 % on realistic parameters** — `lumenairy/elements/bsdf.py:113-152`

`HarveyShackBSDF` is the only model of the three with no closed-form override, so it uses
the base-class quadrature: 256 θ points linearly spaced on [0, π/2], i.e. Δθ = 6.16 mrad —
comparable to, or coarser than, the whole lobe, whose shoulder `l` **defaults to 0.01**.

Measured against a 400 k-point reference and the exact closed form
`TIS(s=2) = π·b0·l²·ln(1 + 1/l²)` (repro `p_bsdf2.py`):

| model | TIS(lib) | TIS(ref) | rel err |
|---|---|---|---|
| Lambertian(ρ=0.5) *(closed-form override)* | 0.5 | 0.5 | −0.00 % |
| Gaussian(σ=1e-2, f=1e-2) *(closed-form override)* | 0.01 | 0.0099987 | +0.01 % |
| HarveyShack(l=1e-1, s=2) | 0.144968 | 0.144988 | −0.01 % |
| **HarveyShack(l=1e-2, s=2) — the DEFAULTS** | 0.00287272 | 0.00289355 | **−0.72 %** |
| **HarveyShack(l=1e-3, s=2)** | 3.54486e-5 | 4.34027e-5 | **−18.33 %** |
| HarveyShack(l=1e-2, s=1.5) | 0.0112895 | 0.01131 | −0.18 % |

TIS is the headline number ("the fraction of incident power scattered out of the specular
direction ... matches the standard spec used in coating/mirror datasheets"), and l = 1e-3
is an ordinary super-polished shoulder.

**Fix.** Substitute u = sinθ: `TIS = 2π ∫₀¹ B(u)·u du`, whose integrand is smooth and
peaked at u = l/√(s−1) — 256 points then suffice with a log-spaced grid. Or add the
`s = 2` closed form as a `total_integrated_scatter` override.

---

### **[P2] `surface_sag_general`'s conic branch materialises ~5 full-grid float64 temporaries** — `lumenairy/elements/lenses.py:228-236`

`norm`, `valid`, `denom_arg`, `sqrt(denom_arg)`, `conic_sag` are each a full grid.
Measured at N = 4096 with 4 aspheric terms (repro `p_kino_perf.py`, `tracemalloc`):

| path | time | peak traced alloc | (one 4096² f64 grid = 128.0 MiB) |
|---|---|---|---|
| numba kernel | **957.8 ms** | **656.0 MiB** | ≈ 5.1 grids |
| pure-NumPy fallback | 4360.9 ms | 784.0 MiB | ≈ 6.1 grids |

So the numba kernel's **4.55× speedup claim checks out**, but the memory story in the
comment at `:258-262` ("Skips the per-term temporary array allocation that the legacy
NumPy fallback required (5 aspheric coeffs at N=4096 is ~640 MB of transient memory in
that path)") is misattributed: the aspheric loop accounts for only 128 MiB of the 784;
the ~656 MiB the numba path still pays is the **conic** branch, which neither path touches.
At N = 32768 that is 5 × 8.6 GB = **43 GB** of transients for a single sag evaluation.

**Fix.** Fuse the conic term into the same numba kernel (it is a 4-flop expression per
pixel), or evaluate it in-place with two reusable buffers
(`np.multiply(h_sq, (1+k)/R**2, out=buf)` → `np.subtract(1.0, buf, out=buf)` → …).

---

### **[P2] `create_microlens_array` peaks at 7× its own output size** — `lumenairy/elements/doe.py:233-267`

`X, Y, in_mla, jx, jy, xc, yc, dX, dY, r_sq, phase` are all full N×N grids.
Measured (repro `p_kino_perf.py`): N = 2048 → **1.25 s, 452 MiB peak** for a 64 MiB
complex128 output; N = 4096 → **4.64 s** (≈ 1.8 GB). The phase is separable
(`−k/(2f)·(dX² + dY²)` with `dX` a function of x alone), so two 1-D arrays of length N plus
the output would suffice — a ~7× memory cut and a large speedup.

(Positive: the function is genuinely fully vectorised — there is **no** per-lenslet Python
loop, contrary to the probe's hypothesis. Measured no-steer behaviour is exact: local
dφ/dx = 0.0 rad/m at every lenslet centre for j = 0, 3, 7; fractional pitch
(101.3 µm on a 2 µm grid = 50.65 px) works correctly.)

---

### **[P2] `make_bsdf` silently ignores unknown dict keys, including the aliases its own docstrings introduce** — `lumenairy/elements/bsdf.py:502-539`

Measured:
* `make_bsdf({'kind':'gaussian','sigma':0.001,'scatter_fraction':0.5})`
  → `GaussianBSDF(sigma_rad=0.01, scattered_fraction=0.01)` — a **10× wider lobe** and
  **50× less scatter** than requested, silently.
* `make_bsdf({'kind':'harvey_shack','A':1e-3,'B':0.02})`
  → `HarveyShackBSDF(b0=1.0, l=0.01, s=2.0)` — all defaults. And `A`/`B` are exactly the
  alias names the `HarveyShackBSDF` docstring itself teaches
  (`:388-390`: "b0 (`A` in some references)", "l (`B`)").

**Fix.** Validate `set(spec) - {'kind', <accepted>}` and raise naming the accepted keys.

---

### **[P2] `thin_grating_efficiency_1d` has no Klein–Cook / Raman–Nath validity guard** — `lumenairy/elements/thin_grating.py:41-182`

Measured (repro `p_grating_emt.py`), λ = 1 µm, depth = 10 µm, n = 1:

| Λ | Q = 2πλd/(nΛ²) | warnings emitted | ΣT returned |
|---|---|---|---|
| 20 µm | 0.16 | 0 | 1.0000 |
| 2 µm | 15.71 | 0 | 1.0000 |
| 1 µm | **62.83** | **0** | 1.0000 |

Thin-grating (Raman–Nath) requires Q ≲ 1; at Q = 63 the answer is a Bragg-regime grating and
the returned numbers are meaningless, delivered with a clean-looking ΣT = 1. The sibling
module `emt.py` *does* ship `_warn_rytov_validity` (`emt.py:51-78`) for the same class of
problem, so the diagnostic policy is inconsistent inside one package.

**Fix.** Emit a `UserWarning` when `Q > 1` (and/or when `Λ < ~10λ`), mirroring
`_warn_rytov_validity`.

---

### **[P2] `sample_scatter_rays` loops in Python over every incident ray** — `lumenairy/elements/bsdf.py:588-593`

```python
for i in range(n_rays):
    inc = np.array([...L[i], ...M[i], ...N[i]])
    out_dirs[i*n_per_ray:(i+1)*n_per_ray] = bsdf.sample(inc, n_per_ray, rng=rng)
```
One Python-level `sample()` call per ray, each with its own `np.array` build and (for
Harvey-Shack) its own rejection-sampling `while` loop with a Python `list.extend`.
A 1e6-ray stray-light run is 1e6 such calls. All three samplers are vectorisable over
incidence (the local→surface rotation is a per-ray 3×3 that batches cleanly).
Measured single-call cost for 20 k samples: 5 ms (l=0.1) to 23 ms (l=1e-3) — the
per-call overhead dominates when `n_per_ray` is 1.

---

### **[P3] `elements.zernike` points Noll users at an OSA converter** — `lumenairy/elements/elements.py:426-428`

> "The Noll *single index* convention is a different beast: to map j_Noll -> (n, m) use
> :func:`lumenairy.analysis.zernike_index_to_nm`."

`zernike_index_to_nm`'s own docstring (`analysis/zernike.py:57`) is *"Convert **OSA**
single-index j to (n, m)"*, and its formula `m = 2j − n(n+2)` is the OSA map. Checked:
OSA j = 5 → (2, +2); Noll j = 5 → (2, −2). A user following the pointer gets the wrong
polynomial for every j ≥ 5. No Noll converter exists in the library
(`analysis/zernike.py:418-429` notes Noll's *polynomials* are identical to OSA's but its
*j-ordering* is not, and says the user must permute themselves).

---

### **[P3] `glass.py` documents three registry entries that do not exist, and the `'air'` short-circuit makes one of them unimplementable** — `glass.py:1038-1048`, `:1424`

The `_GLASS_VALIDITY_REGISTRY_EXEMPTIONS` comment describes `'air'` as a *"callable; uses
Edlen-form ambient model"*, `'vacuum'` as a *"callable; returns n=1.0 at every wavelength"*,
and `'__MIRROR__'` as an internal marker. **None of the three is in `GLASS_REGISTRY`**
(verified: `get_glass_index('vacuum')` and `get_glass_index('MIRROR')` both raise
`ValueError`; `get_glass_index('AIR'/'Air')` returns exactly 1.0 via the short-circuit).

Worse, `glass.py:1424` `if glass_name.lower() == 'air': return 1.0` runs **before** the
registry lookup, so a user who *does* register the Edlén callable the comment describes has
it silently ignored — n = 1.000000 instead of 1.000273 at 1 atm / 1.064 µm, i.e. 273 µm of
OPD per metre of air path (≈ 256 waves). Same for `get_glass_index_complex` (`:1620`).

**Fix.** Either register real `'air'` / `'vacuum'` callables and delete the short-circuit
(keeping n=1 as the default callable), or correct the comment to say the exemptions are
aspirational.

---

### **[P3] `create_periodic_phase_mask` assumes a square cell without checking** — `lumenairy/elements/doe.py:151`

`cell_N = phase_cell.shape[0]` is used for **both** axes (`ix = idx; iy = idx`). Measured:

* `(8, 8)` cell → all 8 columns sampled ✓
* **`(4, 8)` cell → only columns 0–3 are ever sampled; half the design silently never appears in the mask**
* `(8, 4)` cell → `IndexError: index 4 is out of bounds for axis 1 with size 4`

The docstring says "phase_cell : ndarray (M×M)" but nothing validates it, and a DOE cell
loaded from a vendor `.dat` file is a plausible non-square. One `if phase_cell.shape[0] !=
phase_cell.shape[1]: raise` (or proper per-axis indexing) closes it.

---

### **[P3] `thin_grating.py` module docstring gives a garbled formula** — `thin_grating.py:17-19`

```
t_m = f * exp(i*phi) * f * sinc(pi m f) + (1-f) * sinc(pi m (1-f))
      * ...  (see code for exact form)
```
This is not an equation (duplicated `f`, unbalanced terms, trailing `* ...`). The
*implemented* expression is correct — verified to ≤ 2e-8 against a direct 8192-point FFT of
the transmittance across a 4 × 4 sweep of duty cycle and phase step, and it reproduces
4/π² = 0.405285 for the 50 %-duty π-step order ±1 and 4/(9π²) = 0.045032 for order ±3 exactly.
Replace the docstring with the code's own comment block (`:148-155`), which is right.

---

### **[P3] `surface_sag_general(h_sq, R=0)` → all-NaN plus four bare numpy RuntimeWarnings** — `lumenairy/elements/lenses.py:220-236`

Measured: `surface_sag_general([0, 1e-6], R=0.0, conic=0)` returns `[nan, nan]` with
`divide by zero encountered in divide` ×2 and `invalid value encountered in divide` ×2 —
anonymous warnings naming neither the function nor the argument. This is the *same* class
the v5.32 (audit W5-2) change fixed for `apply_thin_lens(f=0)` and
`apply_cylindrical_lens(f=0)`; the sag entry point was not part of that sweep. `R=inf` and
`R=None` are handled cleanly (sag = 0) ✓.

---

### **[P3] `apply_aperture` offers no anti-aliased / area-weighted edge** — `lumenairy/elements/elements.py:226-312`

Hard binary mask only. Measured pixel-count area vs the analytic disc area:

| D/dx | pixel area | analytic | rel err |
|---|---|---|---|
| 50 px | 1.945000e-7 | 1.963495e-7 | −0.942 % |
| 200 px | 1.962563e-7 | 1.963495e-7 | −0.048 % |
| 800 px | 1.963316e-7 | 1.963495e-7 | −0.009 % |

Nothing in the family (`apply_aperture`, `apply_lyot_stop`, `apply_mirror`,
`segment_geometry`) offers a grey-pixel option, so under-sampled stops carry a systematic
throughput bias plus extra Gibbs ringing. See "Alternative algorithms" for the cheap fix.

---

### **[P3] `apply_spherical_lens` and `apply_real_lens` put the focus 0.5 mm apart on the same prescription, and neither says so**

Measured (n = 1.5168, R1 = +20 mm, R2 = −20 mm, d = 3 mm, λ = 1 µm, 1.2 mm Gaussian,
exact ASM; repro `p_real.py`):

| quantity | value |
|---|---|
| thin-lens lensmaker f | 19.3498 mm |
| thick EFL | 19.8573 mm |
| BFL (from back vertex) | 18.8424 mm |
| **`apply_spherical_lens` focus** | **19.3111 mm** (= thin f − 39 µm) |
| **`apply_real_lens` focus** | **18.8080 mm** (= BFL − 34 µm) |
| RMS phase difference over the illuminated pupil | **1.708 rad = 0.272 waves** |
| PV phase difference | 8.174 rad = 1.301 waves |

So the two models are not just "slightly different": their **output reference planes differ**
(single screen → thin-lens principal plane at the vertex; split-step → the back vertex), and
substituting one for the other moves the focus by **503 µm on a 19 mm lens (2.6 %)**.
`apply_spherical_lens`'s "Validity boundary" section documents the *wavefront* error
(E-C2: 0.011–21.7 waves PV f/16→f/2) and the `d`-blindness (verified: `d=4 mm` and `d=1 nm`
return bit-identical fields ✓) but never states the reference-plane offset. One sentence in
the See Also block would close it.

---

## Performance opportunities

| # | site | measured now | opportunity | estimate |
|---|---|---|---|---|
| 1 | `lenses.py:228-236` conic branch | 656 MiB peak @ N=4096 (numba path) | fuse into the numba kernel / in-place buffers | 5 grids → 1–2; **~3–4× memory**, 43 GB → ~17 GB at N=32768 |
| 2 | `doe.py:233-267` MLA | 452 MiB peak, 1.25 s @ N=2048 | separable 1-D `dX`, `dY` then one outer add | **~7× memory**, ~2× time |
| 3 | `bsdf.py:588-593` `sample_scatter_rays` | 1 Python call/ray | vectorise `sample()` over incidence | 100×+ for MC stray light |
| 4 | `bsdf.py:449-464` HS rejection sampler | ~9 % acceptance at l=1e-2, ~1 % at l=1e-3; Python `list.extend` | inverse-CDF in closed form for s=2 (`u² = l²((1+1/l²)^ξ − 1)`) | exact, 0 rejections, no list |
| 5 | `lenses.py:278-279` NumPy aspheric fallback | 4361 ms vs numba 958 ms @ N=4096 | Horner in h² (1 temp instead of one per term) | ~2× on the non-numba path |
| 6 | `_lens_thin.py` ×6 | each rebuilds `arange`/`meshgrid`/`r_sq` per call | cache the coordinate grid on (N, dx, dy, xc, yc) | 3 grids saved per call; material when sweeping f |
| 7 | `elements.py:585-586` Zernike loop | one full-grid `zernike()` per (n,m) | share `rho**p` powers across terms (Horner on ρ²) | ~2–3× for >6 terms |
| 8 | `bsdf.py:117-127` TIS grid | builds a (256,128,3) float64 S each call | 1-D in u = sinθ (see P2 above) | 128× fewer evals *and* more accurate |

---

## Alternative algorithms / methods

1. **Sub-pixel-accurate apertures.** Replace the binary `h_sq <= (D/2)²` mask with the exact
   circle–pixel area fraction (or 4×–16× supersampling of the mask only). Cuts the
   −0.94 % throughput bias at D/dx = 50 px to <0.05 % and removes the dominant edge-aliasing
   source for coronagraph contrast work. Cost: one extra mask build, zero per-propagation cost.
   Ref: Perrin et al., POPPY `optics.py` "gray pixel" apertures; Soummer et al. 2007 (MFT).

2. **Sub-harmonic turbulence screens.** After the √2 fix (P1 above) the residual deficit is
   D/D_theory ≈ 0.80 at r = 0.05 r0 and 0.47 at r = 3.2 r0. Three levels of 3×3 sub-harmonics
   (Lane, Glindemann & Dainty 1992, *Waves in Random Media* 2, 209; Schmidt 2010
   `ft_sh_phase_screen`) brings D within a few % over 0.01 r0 – 4 r0 for ~5 % extra cost.

3. **Exact GRIN screen / GRIN ABCD.** The one-line fix in P1 (`sin(g d)` instead of `g d`)
   makes the screen paraxially exact at any pitch. For a *thick* GRIN rod, the correct
   thin-element reduction is the ABCD-equivalent pair (input quadratic, free-space
   `B = sin(gd)/(n0 g)`, output quadratic) — Siegman, *Lasers*, §15.4; Gómez-Reino et al.,
   *Gradient-Index Optics* (Springer 2002), Ch. 3.

4. **Aspheric sag by Forbes Q-bfs instead of raw h^{2n}.** `surface_sag_general`'s
   `Σ Aₙ h^{2n}` is catastrophically ill-conditioned for high-order fits; the Q-bfs basis is
   already implemented right next door in `freeform.py` (`_q_bfs_eval`, verified orthonormal).
   Forbes, *Opt. Express* 15, 5218 (2007) & 18, 13851 (2010).

5. **Klein–Cook regime classifier for `thin_grating`.** Q = 2πλd/(nΛ²) and ρ = λ²/(nn₁Λ²)
   (Moharam & Young, *Appl. Opt.* 17, 1757 (1978)) give a 2-number regime test; the library
   already ships an RCWA to hand off to. Cheap and would close P2 #10.

6. **Beyond zeroth/second-order Rytov.** For Λ/λ ≳ 0.3 where the measured EMT residual
   reaches 0.9–1.9 % in slab T, the standard next step is a 2-layer "EMT + effective
   interface" model (Lalanne & Hugonin, *J. Opt. Soc. Am. A* 15, 1843 (1998)) which removes
   the O(Λ/λ) interface term the bulk correction explicitly does not touch — the `emt.py`
   docstring already names this limitation correctly.

7. **Analytic TIS for the ABC BSDF.** For general s the hemisphere integral has the closed
   form `TIS = πb0 l²[(1+1/l²)^{1−s/2} − 1]/(1 − s/2)` (s ≠ 2), `πb0 l² ln(1+1/l²)` (s = 2).
   Exact, O(1), and would fix P2 #6 outright.

---

## Code organization observations

* **`lenses.py` is a junk drawer** (1025 lines). It currently hosts: the CuPy/numexpr/numba
  lazy-import scaffolding (44–132), the sag helpers (140–419), prescription grid
  recommenders (426–791), Chebyshev re-export aliases (795–817), a 4-variable
  Chebyshev tensor-product evaluator and its gradient twin (820–921) used only by the
  Maslov/asymptotic propagators, a data normaliser (928–941), two empty section banners
  (944–953), and then 70 lines of pure re-export from six `_lens_*` modules. Proposed split:
  `elements/_sag.py` (surface_sag_general / biconic / the numba kernel — also removes the
  `freeform.py → lenses.py → _lens_real.py` import knot), `elements/_grid_advice.py`
  (`check_grid_vs_apertures`, `recommend_grid_for_prescription`,
  `_warn_if_aperture_exceeds_grid`, `_collect_semi_diameters`), `_math/poly4d.py`
  (`_multi_indices_total_degree`, `_evaluate_polynomial_4d*`, `_fit_normaliser` — they are
  maths, not lenses), leaving `lenses.py` as a ~60-line façade.

* **Dead / near-dead.** `lenses.py:944-953` are two section banners with no content
  ("Main function", "Maslov propagator -- moved to ..."). `_POLYNOMIAL_STUB_NAMES =
  frozenset()` (`glass.py:525`) is empty, so the entire `NotImplementedError` arm at
  `glass.py:1561-1575` (15 lines of message) is unreachable.

* **Backend-scaffold duplication across `elements/`.** Four independent patterns coexist:
  (a) `lenses.py`'s `CUPY_AVAILABLE` + `_ensure_cupy_loaded` + `_is_cupy_array`;
  (b) `_lens_thin.py`'s PEP-562 `__getattr__` proxy back into (a) plus an identical
  8-line dispatch block copy-pasted into all six entry points;
  (c) `backend.array_namespace` (used by `elements.py`, `freeform.py`, `emt.py`);
  (d) `doe.py`, `thin_grating.py`, `bsdf.py`, `segment_geometry.py`, `materials.py`: pure
  NumPy, no dispatch at all. `_lens_thin.py`'s six copies of
  `if CUPY_AVAILABLE and (use_gpu or _is_cupy_array(E_in)): ...` are begging to be one
  `_resolve_xp(E_in, use_gpu)` helper.

* **Comment noise.** `apply_thin_lens`'s docstring + guards run 260 lines before the first
  line of physics; the `f=0` / `f=nan` rationale alone is 80 lines of prose reciting
  pre-fix behaviour of code that no longer exists. The same content as a 3-line summary
  plus a CHANGELOG pointer would leave the function readable.

* **Positional-float footguns are catalogued but not fixed.** `apply_thin_lens`'s "Scope of
  that guarantee" note (v5.30, E-M9) names five entry points where the swap hazard is live
  (`apply_axicon`, `apply_mirror`, `apply_aperture`, `apply_zernike_aberration`,
  `thin_grating_efficiency_1d`). A `*` after the field argument plus a one-release
  `DeprecationWarning` shim would close all five; the note has been "deliberately NOT
  converted" for several releases.

* **`segment_geometry.py` is misnamed relative to the audit brief's expectation** — it is
  1-D interval arithmetic for grating cross-sections, not segmented-aperture (hex/polygon)
  geometry. No hex-segment aperture builder exists anywhere in `elements/`.

---

## Unverified suspicions

* **The 24 formula-3 `POLYNOMIAL_COEFFICIENTS` rows (4 CDGM + 10 Hikari + 10 Sumita) were
  not value-checked.** Given that 3 of 49 Sellmeier rows are wrong (F1), the polynomial
  table deserves the same treatment. Confirm by extending `p_glass3.py` to force the
  `__polynomial__` / no-refractiveindex fallback and diff against the catalogue across
  0.4–2.0 µm. I did not do it because none of those names is registered `'__polynomial__'`
  (they are all tuples), so the rows are reachable only on a minimal install — which I
  could not create without touching the environment.

* **numba `fastmath=True` on `_aspheric_sag_accum_numba` (`lenses.py:150`).** `fastmath`
  implies LLVM's `nnan`/`ninf`, which formally licenses the optimiser to assume no NaN —
  and this kernel deliberately accumulates onto a `sag` array that carries NaN outside the
  conic domain. I measured NaN propagating correctly today (`p_sag.py`: numba and NumPy
  both return `nan` at the out-of-domain pixel), but that is not a guarantee across LLVM
  versions. Confirm by pinning a regression test, or drop to `fastmath={'nsz','arcp','contract'}`.

* **`apply_mirror` focus.** The double-pass `φ = −2k·sag` and the f = R/2 reduction were
  read and look right (and a prior audit measured them), but I did not re-run the ASM
  focus test. Confirm with the same plane-wave + `angular_spectrum_propagate` scan used in
  `p_thinlens.py`.

* **`makedammann2d` IFTA convergence.** Read only; I did not run 3000 iterations. Two spots
  look fragile: `farfieldamp = 0.00001 + (farfieldamp / np.abs(farfield)) * totaldiforders`
  (`doe.py:922`) divides by a far field that can legitimately contain exact zeros, and
  the forward transform uses `fftshift(fft2(fftshift(...)))` while the inverse uses
  `ifftshift(ifft2(ifftshift(...)))` — identical only because `ndifordersx/y` are forced
  even. Confirm with an odd `cell_pixels` (currently rejected) or by instrumenting for
  `inf` in `farfieldamp`.

* **`_lens_real.py`'s `lens_sag_float32_opd_error` OPD sign.** Its `opd += (n2−n1)·sag`
  composes to `(n−1)(sag1 − sag2)`, matching `apply_spherical_lens`'s `−k(n−1)(sag1−sag2)`
  — internally consistent, and the 503 µm focus offset I measured is a *reference-plane*
  difference, not a sign error. But I did not trace where `apply_real_lens` actually applies
  the sign; that file belongs to another partition.

---

## Checked and found correct (so coverage is known)

**Sag.** `surface_sag_general` conic sag matches an independent
`z = c h²/(1+√(1−(1+k)c²h²))` to **1.08e-19 m** for sphere (k=0), parabola (k=−1),
hyperbola (k=−3), oblate ellipsoid (k=+2) and negative R. Out-of-domain → NaN at
norm ≥ 0.9999 (h ≥ 0.99995·h_limit), documented and consistent across all five siblings.
R=inf / R=None → sag 0. Odd aspheric powers correctly rejected by
`check_even_aspheric_powers` in `surface_sag_general`, `surface_sag_biconic` (both axes) and
`apply_aspheric_lens` (A1 and A2). float32 `h_sq` returns float32 with 4.9e-11 m error
(0.05 nm; negligible at 1.55 µm) and correctly skips the numba kernel. Numba-vs-NumPy
agreement on C-contiguous input: max abs 2.17e-19, relative 2.7e-16 (**not** bit-identical,
as `fastmath` reassociation implies — the code does not claim bitwise).

**`surface_sag_biconic`.** The separable-vs-Zemax deviation is real and exactly as the
docstring's RT-2 note says: 0 on either axis, **0.515 %** at the (10 mm, 10 mm) corner for
Rx=50 mm/Ry=80 mm/kx=−0.5/ky=+0.3.

**`apply_thin_lens`.** Converging sign under exp(+ikz) confirmed by ASM: peak at
z = 19.9600 mm for `paraxial` (−40 µm, spherical aberration), **exactly 20.0000 mm** for
`nonparaxial` and `stigmatic` at f = 20 mm, NA ≈ 0.05. `f < 0` diverges on both sqrt models
(peak 0.445 vs collimated 1.0 — no spurious focus; the v5.25.0 bug-1 fix holds).
`aplanatic` leaves |E| untouched outside the domain (the 4.10 unit-phase sentinel works).
`stigmatic` with R_in = ∞ reduces to `nonparaxial` as documented. `f = 0` / `f = ±inf`
rejected with a named `ValueError` in both `apply_thin_lens` and `apply_cylindrical_lens`.
`apply_cylindrical_lens` line focus at exactly 20.0000 mm.

**`apply_spherical_lens`.** Sign chain verified algebraically (t(h) = d − sag1 + sag2 ⇒
φ = −k(n−1)(sag1−sag2) ⇒ 1/f = (n−1)(1/R1 − 1/R2), lensmaker) and numerically (focus at
thin-lens f − 39 µm). `d=4 mm` vs `d=1 nm` **bit-identical** — the E-C2 claim holds exactly.

**`apply_axicon`.** Far-field ring at |f_r| = 8693.6 m⁻¹ ⇒ θ = 8.694 mrad vs the predicted
(n−1)α = 8.727 mrad — 0.14 FFT bins, i.e. exact within sampling.

**`thin_grating_efficiency_1d`.** Analytic Fourier coefficients verified against a direct
8192-point FFT of the transmittance over a 4×4 sweep (duty 0.25/0.4/0.5/0.7 × phase
0.5π/π/2π/3π): max |code − FFT| ≤ 2e-8 where the boundary lands on a sample (and ≤ 8e-5
where it does not — that residual is the FFT's error, not the code's). The 50 %-duty π-step
grating gives η₀ = 0.000000, η₊₁ = **0.405285** = 4/π² and η₊₃ = **0.045032** = 4/(9π²).
Order-truncation behaviour and the evanescent cut on `|kx| < k0·n_sub` are as documented.

**`emt.py`.** Formulas re-derived and confirmed: Rytov 0th order
(`ε_par = f ε1 + (1−f) ε2` arithmetic on y/z, `ε_perp = 1/(f/ε1 + (1−f)/ε2)` harmonic on x),
the 2nd-order Lalanne–Lemercier-Lalanne 1996 corrections (TE additive in ε, TM weighted by
`ε_perp³·ε_par`), Maxwell-Garnett (g = 2 sphere / 1 cylinder, closed form re-derived), and
the Bruggeman quadratic `gε² + bε − ε_aε_b = 0` with `b = ε_a[(1−f)−gf] + ε_b[f−g(1−f)]`.
**TE/TM labelling verified against RCWA** (repro `p_emt2.py`, 0.8 µm slab, n=2/1, f=0.5):
`rcwa_efficiency_1d(polarization='te')` tracks the ARITHMETIC branch and `'tm'` the
HARMONIC branch, i.e. TE = E along the grooves ✓. Slab-T convergence |ΔT| vs RCWA:

| Λ/λ | TE | TM |
|---|---|---|
| 0.02 | 7.0e-5 | 1.8e-4 |
| 0.05 | 4.4e-4 | 5.7e-4 |
| 0.10 | 1.7e-3 | 1.7e-3 |
| 0.30 | 9.0e-3 | 1.9e-2 |

and the order-2 correction at Λ/λ = 0.3 improves TE 9.0e-3 → 5.0e-3 and TM 1.9e-2 → 1.3e-2
— a refinement, not a new convergence order, exactly as `emt.py`'s docstring carefully says.
`_warn_rytov_validity` fires at Λ/λ = 0.3 as designed.

**DOE.** `create_kinoform`: floor-vs-round quantisation gives identical blazed-grating η₁
(L=2: 0.405284/0.405282; L=4: 0.810566/0.810564; L=8: 0.949636/0.949635; L=16:
0.987209/0.987209 vs sinc²(1/L)) — the `floor` choice is a piston/zone shift, **not** a bug.
2-D core power fraction at the design focus: 0.4056 / 0.8057 / 0.9437 / 0.9811 / 0.9938 for
L = 2/4/8/16/1024 vs sinc²(1/L) = 0.4053/0.8106/0.9496/0.9872/1.0 ✓.
`create_fresnel_zone_plate`: binary-amplitude core power 0.0990 vs 1/π² = 0.1013, binary-phase
0.3831 vs 4/π² = 0.4053 ✓; the paraxial-vs-exact zone radius difference is 0.062 % at m=50
(NA 0.071) and is explicitly documented as paraxial; `focal_length <= 0` correctly rejected.
`create_periodic_phase_mask` on a square cell: per-cell-pixel occupancy exactly uniform
(`[32]*8`) and **0.0000 %** of power off the order lattice — the v5.30 E-M7 modulo fix is verified good.
`create_microlens_array`: |T| = 1 everywhere, exactly zero steer at every lenslet centre,
fractional pitch handled.

**BSDF.** `LambertianBSDF.total_integrated_scatter()` = ρ exactly; `GaussianBSDF` = the
declared `scattered_fraction` to 0.01 %. The v5.17 P3-15 oblique-incidence normalisation is
correct: `∫BSDF·cosθ dΩ` = 0.019993 / 0.019997 / 0.020004 at θᵢ = 0 / 30 / 60° (target 0.02).
The Gaussian sampler's Rayleigh draw and the Harvey-Shack `u·BSDF(u)` radial weight are
both the right power-weighted densities (the cos θ factors cancel exactly under u = sinθ).
The `evaluate`-shape guard (v4.14 P2 #17) and the Gaussian/HS orthonormal basis construction
around the specular direction are correct including the |spec_z| → 1 degenerate case.

**Glass.** Name resolution is **not** silently aliasing: `'BK7'`, `'NBK7'`, `'n-bk7'`,
`'N-BK7 '` all raise `ValueError` with a `difflib`/substring suggestion list — the P0 the
brief worried about is absent. κ sign convention correct (`N-BK7` at 1.55 µm →
1.5006520 + 1.436e-7j, κ > 0 = absorbing under exp(−iωt)). Sellmeier and polynomial
evaluators handle scalar and array inputs, guard resonances (|λ²−Cᵢ| < 1e-12), guard
negative n², and split NaN/negative-λ handling correctly between the sign-symmetric
(Sellmeier) and non-symmetric (polynomial) forms. Cache discipline is sound: LRU-bounded,
lock held only around bookkeeping with `compute()` outside, keyed on
`(name, round(λ·1e12))` and consulted **only inside immutable-catalogue branches** so a
re-registration cannot serve a stale value. 46 of 49 bundled Sellmeier rows match the
catalogue to ≤ 4e-6 in n_d and ≤ 3.3e-5 across 0.4–1.6 µm (see F1 for the three that do not).
No Abbe-number helper exists in `glass.py` (I computed V_d externally).

**freeform.py.** The Forbes normalisation closed form was re-derived and matches the code:
`h_n[0,1] = Γ(n+α+1)Γ(n+β+1)/((2n+α+β+1)·n!·Γ(n+α+β+1))`, giving
`c_n = √((2n+3)(n+2)/(n+1))` for Q-bfs (α=β=1) and `√(2n+3)` for Q-con (α=0, β=2) —
both the v4.15.1-corrected comment block and `_jacobi_norm_factor` agree. The Jacobi 3-term
recurrence and the Q-bfs `u²(1−u²)` / Q-con `u⁴` prefactors match Forbes 2007 Eq. 13 /
2010 Eq. 6. Radial-primary + rectangular-secondary clipping (v4.15.1 P1-F1-1) and the
`r_max`-required guard (P1-F1-2) are both present and correct.
`surface_sag_zernike_freeform` uses `analysis.zernike_index_to_nm`, which IS OSA — matching
its own docstring (the mislabelling is in `elements.zernike`'s docstring, P3 above).

**elements.py.** Zernike radial recurrence and OSA normalisation (√(n+1) / √(2(n+1)))
correct, all seven docstring examples verified symbolically, ρ>1 zeroed.
`apply_gaussian_aperture`'s 1/e-amplitude and 1/e²-intensity radii are both σ√2 as claimed.
Coronagraph builders (Lyot hard/gaussian/sin2, vortex with the centre-pixel kill, FQPM
`X·Y<0`, 8OPM octant parity, Lyot stop, cos²/cos^n/gaussian/sonine apodizers) are all
standard forms with consistent `_xp_of` dispatch and dtype-aware zeros.
The turbulence PSD *shape* (0.023 f^{-11/3} in cycles/m, the von Kármán `(f²+1/L0²)^{-11/6}`
knee, the `exp(−(2πf l0/5.92)²)` inner-scale cutoff, and the v5.30 E-L11 integer
`N//2` DC anchor) is correct — only the amplitude is not (P1 above).

**segment_geometry.py / materials.py / coronagraph.py.** Interval arithmetic
(normalise/complement/union/wrap-aware dilate), the L∞ conformal `coat`, the `line_interface`
thin-band warnings, and the exact-boundary `to_rcwa_stack` pixelation with its NaN-tile
guard all read correct. `Material` sorts, rejects duplicates/non-finite/short tables, and
refuses to extrapolate. `coronagraph.py` is a pure six-name re-export as documented.
