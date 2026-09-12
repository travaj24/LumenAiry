# CARRIER audit — `lumenairy/propagators/carrier.py` + `carrier_field.py`

Scratch dir for every script quoted below:
`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/CARRIER/`

## Scope read (files + line ranges actually read; what I did NOT get to)

`lumenairy/propagators/carrier.py` (10594 lines) — read in full, either verbatim or
through a docstring/comment-stripped view (`strip.py`, which drops only comment-only
lines and docstrings and keeps original line numbers):

* 1–360 verbatim (module docstring, tuning constants, backend shim)
* 360–1060 verbatim (`_freq_*`, `_exact_tf_2d_xp`, `_tf_phase_to_H`, `_fresnel_tf_2d_xp`,
  gap-kernel gate, `CarrierReferencedField`, `_phasor_rows`, `_radial_carrier_phase`,
  `propagate_carrier_referenced`)
* 1060–1740 verbatim (`_exact_envelope_tf_step`, `_carrier_step_fast`, amp radius/centroid,
  `_near_focus_needs_bridge`, `_propagate_carrier_focus_crossing`, the whole astigmatic
  per-axis path, `_build_carrier_phase`)
* 1740–2130 verbatim (`reconstruct`, `envelope`, `_carrier_fit_alias_fraction`,
  `_fit_carrier_inv`, `carrier_referenced_fit_radius`, `_rereference`)
* 2130–2910 verbatim (`carrier_referenced_aperture`, `carrier_referenced_focus_readout`,
  `_default_focus_standoff`, `_small_extent_focus_standoff_f`, `_check_readout_replica`)
* 2910–3620 verbatim (`_exact_sphere_eikonal`, `_tilt_exactness_phase`,
  `_sphere_parab_conversion`, `_fourier_upsample_crop`, `_crop_about_centre`,
  `_guard_dispose`, `_warn_undeduped`)
* 3620–4400 verbatim (P3 congruence gate, memory model, `_memory_bounded_n_fine`)
* 4660–5000 verbatim (`carrier_referenced_exact_focus_readout` body) + 4090–4145
* 5000–5700 verbatim (`TracedCarrierChainResult`, `_group_abcd`, `_paraxial_group_r_out`,
  `_parse_chain_carrier`, `_tilt_obliquity`, `_group_chief_transfer`, `_shift_envelope`,
  `_tilt_ramp`, `_check_tilt_fits`)
* 5700–10594 through the stripped view (`_gap_envelope_angular_spread`,
  `_check_gap_paraxial`, `_check_decentred_fit`, DOE normalisation, `_fine_trace_group_exit`,
  `_run_chain_dx_self_check`, the full 1595-line `propagate_traced_carrier_chain` body,
  the multi orchestrator, worker plumbing), plus 10060–10184 verbatim.

`lumenairy/propagators/carrier_field.py` (1743 lines) — read in full through the stripped
view plus the `re_reference` docstring verbatim.

NOT covered: the CuPy/JAX arms of `_exact_tf_2d_xp` / `_tf_phase_to_H` / `_asm_axis` could
only be desk-checked (no GPU/cupy here); the zarr round trip was not executed; I did not
reproduce the design-121 fixtures in `validation/repro_traced_carrier_122/` (they need the
untracked `.zmx`/`.npy` assets and hours of runtime) — I built independent fixtures instead;
`propagate_traced_carrier_chain_multi` was read but not run end-to-end (each chain leg on
this box costs ~30–90 s, so a K-congruence run was out of budget); the `readout_tile='auto'`
probe/resize loop was read, not exercised.

---

## Findings

### [P1] `carrier_referenced_focus_readout` silently returns a 4–40x-low focal peak when the carrier is a few percent off the beam's own wavefront — `carrier.py:2560` (`_default_focus_standoff`), `carrier.py:1267` (`_near_focus_needs_bridge`), `carrier.py:2340` (readout, no post-leg check)

**What is wrong.** Both the standoff resolver and the near-focus guard estimate the beam at
the stop plane from the **carrier**, never from the field:

```
carrier.py:2583   w0 = wavelength * abs(R) / (np.pi * w_env)   # waist implied by R
carrier.py:2584   zR = np.pi * w0 * w0 / wavelength
carrier.py:2600   ext = half / w_env                            # containment model
carrier.py:1283   w0     = wavelength * abs(R) / (np.pi * w_in)
carrier.py:1284   w_geom = w_in * abs(R_out) / abs(R)           # beam ASSUMED to contract with R
carrier.py:1285   w_out  = max(w_geom, w0)
```

`margin(f) = ext·f/sqrt(1+f²)` is therefore a statement about the *carrier*'s geometric
contraction, and the whole `_FOCUS_STANDOFF_MARGIN = 3.2` derivation (the ~100-line comment
at `carrier.py:181–253`) silently assumes `w_beam(s) = w_carrier(s)`. When the envelope
carries residual curvature — i.e. whenever the carrier is not the beam's own wavefront —
the beam does **not** contract with the co-moving grid, and nothing between the carrier leg
(`carrier.py:2527`) and the Bluestein zoom (`carrier.py:2555`) measures what actually landed
there. The replica guard does not cover this (it is a window-vs-period test), and
`_near_focus_needs_bridge` is blind for the same reason.

**Evidence** (`p6c_mismatch.py`, shipped defaults: `on_replica='error'`, default standoff,
no other knobs). Physical field = converging Gaussian, `w = 1 mm`, true parabolic radius
`R0 = -20 mm` (NA 0.05), λ = 1.31 µm, N = 1024, half-extent 4 mm (`ext = 4.0`), readout
`dx_out = w0/8`, `N_out = 64` (window 66.7 µm, inside the ~85 µm period in **every** row, so
no guard is in play). Only the *reference carrier* is varied; the field is identical:

| R/R0 | standoff | co-moving half-extent at stop | measured beam there | half/beam | peak vs truth | warnings |
|---|---|---|---|---|---|---|
| 1.00 | 222.4 µm | 44.48 µm | 13.84 µm | **3.21** | 1.000000 | 0 |
| 0.99 | 418.0 µm | 44.03 µm | 22.43 µm | 1.96 | 0.986188 | 0 |
| 0.98 | 613.6 µm | 43.59 µm | 31.42 µm | 1.39 | **0.745432** | 0 |
| 0.95 | 1200.7 µm | 42.25 µm | 46.40 µm | 0.91 | **0.187913** | 0 |
| 0.90 | 2180.1 µm | 40.03 µm | 46.26 µm | **0.87** | **0.026309** | 0 |

`half/beam` is exactly `_FOCUS_STANDOFF_MARGIN` (3.2) for the matched carrier — the resolver
works when its premise holds — and collapses to 0.87 as the premise fails. (The 46 µm
"beam" is itself already clipped: the un-clipped radius at that plane is ~109 µm, so the
true containment is 0.37 radii.) A second run at `N_out = 96` shows the same carriers all
refused by `on_replica='error'` — i.e. the replica guard fires or not depending on the
requested window, and is not a substitute.

Cross-check that the matched case is the truth: `p3_readout.py` scores the same readout
against an analytic Gaussian-ABCD focal-plane oracle at NA 0.05 / 0.10 and gets
piston-free relL2 1.05e-03 / 3.81e-03 and peak ratio 0.99992 / 0.99964.

**Impact.** `carrier_referenced_focus_readout` is public and is the chain's
`final_leg='paraxial'` landing (`carrier.py:8654`). The chain supplies the carrier itself
from a *paraxial ABCD* (`_paraxial_group_r_out`, `carrier.py:5086`), so a user never chooses
it; the real exit wavefront differs from that ABCD radius by the Gaussian `zR²/Δ` term, by
aberration, and by the `~NA²/2` sphere-vs-parabola offset this module documents at
`carrier.py:1985`. A 1 % mismatch already costs 1.4 % of peak and a 2 % mismatch 25 %, with
a hard cliff just beyond — and the returned spot still looks perfectly plausible (core
shape intact, no warning, power book-keeping unremarkable). This is the "plausible-looking
wrong answer" class every other guard in this module exists for.

**Recommended fix.** The check is nearly free because `|env| == |E|` (the carrier is pure
phase): right after the carrier leg at `carrier.py:2527`, measure
`w_stop = _envelope_amp_radius(env_s, dx_s, dx_s)` and dispose on
`0.5*N*dx_s < _FOCUS_STANDOFF_MARGIN * w_stop` through a new `on_focus_containment`
(`'error'` default, message quoting the measured containment, the resolved standoff and the
standoff that would restore the margin). Better still, make `_default_focus_standoff`
self-consistent: it already has `_fit_carrier_inv`, so one fixed-point step —
`1/R_eff = 1/R + 1/R_env` from the envelope's own fitted curvature, then resolve `f`
against `R_eff` — places the leg on the *beam*'s focus instead of the carrier's and removes
the failure rather than reporting it. (Guard first: the fixed point changes shipped numbers,
the guard does not.)

---

### [P1] `carrier_referenced_fit_radius` fits the parabola about the GRID ORIGIN, so a decentred beam's radius is wrong by `1 + 2 x0²/w²` and a pure tilt reads as curvature — `carrier.py:1884` (`_fit_carrier_inv`), `carrier.py:1960`

**What is wrong.** `_fit_carrier_inv` builds its coordinates at `carrier.py:1890–1893` as

```python
x = (np.arange(Nx, dtype=np.float64) - Nx / 2) * dx
y = (np.arange(Ny, dtype=np.float64) - Ny / 2) * dy
Y, X = np.meshgrid(y, x, indexing='ij')
```

with no `centre` argument anywhere in the call chain, and estimates
`1/R = Σ w·x·(dφ/dx) / (k Σ w·x²)`. For a beam centred at `x0` carrying a perfect sphere
about *its own* centre (`φ = k(x-x0)²/2R`) the numerator loses `x0·⟨x⟩`, giving exactly

`1/R_fit = (1/R) · 2σ²/(x0² + 2σ²)`, i.e. `R_fit = R·(1 + 2 x0²/w²)` (σ² = w²/4).

Every sibling helper in this module was given a `centre=` argument for precisely this class
of defect — `_envelope_amp_radius` (1208, "verifier round 2"), `_envelope_amp_centroid`
(1239, "V3"), `_radial_carrier_phase` (790, "niche D1"), `_exact_sphere_eikonal` (2903),
`_sphere_parab_conversion` (3145), `_tilt_exactness_phase` (2958). The fitter was missed.

**Evidence** (`p4_fit.py`; λ = 1.31 µm, N = 512, dx = 2 µm, w = 100 µm, R = 50 mm,
`estimator='increment'` so the grid-pitch bias is out of the picture):

```
x0 = 0.0 w  ->  R_fit/R = 1.0000
x0 = 0.5 w  ->  R_fit/R = 1.5000
x0 = 1.0 w  ->  R_fit/R = 3.0000       (predicted 1 + 2·1  = 3)
x0 = 2.0 w  ->  R_fit/R = 9.0000       (predicted 1 + 2·4  = 9)
```

and, worse, a **flat** wavefront carrying only a uniform tilt reads as strongly curved once
it is off axis (`1/R_fit = L·x0/(x0² + 2σ²)`):

```
L = 0.002, x0 =  50 um  ->  R_fit = 0.0750 m   (truth: inf)
L = 0.020, x0 =  50 um  ->  R_fit = 0.0075 m   (truth: inf)
L = 0.020, x0 = 200 um  ->  R_fit = 0.0113 m   (truth: inf)
```

`on_aliased='warn'` does not fire (the field is perfectly sampled); nothing warns.

**Impact.** The function is exported in `lumenairy.__all__` and its documented purpose is
"reference a measured / apertured field to its actual wavefront so the envelope is flat and
the co-moving grid is well conditioned" — the decentred case is exactly a tilted DOE order
or an off-axis emitter, which is the configuration the multi-congruence route exists for.
Feeding the returned `R` back as a carrier makes the envelope *worse*, not flat. The
library's own use is currently safe (`carrier_referenced_aperture(refit_carrier=True)` is
the only internal caller, and the chain's tracking frame keeps the beam centred), so this
is a public-API defect rather than a default-path one — hence P1, not P0.

**Recommended fix.** Add `centre=(0.0, 0.0)` to `_fit_carrier_inv` and a
`centre='auto'|(x0,y0)` to `carrier_referenced_fit_radius`, defaulting to
`_envelope_amp_centroid(E, dx, dy)` (which already sub-pixel-snaps to `(0,0)`, so the
on-axis answer stays byte-identical), and subtract it from `x`/`y` exactly as
`_envelope_amp_radius` does. Optionally project out the residual tilt
(`Σ w x / Σ w`) before the curvature fit so a decentred *tilt* cannot masquerade as `R`.

---

### [P2] A complex64 TILTED chain is silently promoted to complex128 on the first obliquity piston — `carrier.py:8090`, `carrier.py:8233`, `carrier.py:8706`

**What is wrong.** Three sites apply the tilt obliquity piston as

```python
env = np.asarray(env) * np.exp(1j * k0 * _own * (_ob - 1.0))
```

`np.exp(1j*x)` returns a **numpy `complex128` scalar**, which under NEP 50 is strong and
promotes a `complex64` array. `_carrier_step_fast` knows this and works around it
(`carrier.py:1190`: "a python complex stays weakly-typed"), but the chain sites do not, and
there is no cast back until the *group exit* (`carrier.py:8580`). Everything in between —
`E_full`, and every phase screen built with `dtype=_ga_dt` at 8479–8501 — therefore runs in
complex128.

**Evidence.** `p7_odd_dtype.py` (numpy 2.4.6): `complex64 array * np.exp(1j*0.3)` →
`complex128`; `complex64 array * complex(np.exp(1j*0.3))` → `complex64`.
End to end, `p8_c64chain.py` on a one-singlet chain, complex64 input:

```
carrier = inf                                  -> output dtype complex64
carrier = TiltedCarrier(inf, L=0.02, ...)      -> output dtype complex128
```

**Impact.** This defeats the whole v5.44 / v5.44.1 complex64 campaign
(`AUDIT_TRACED_MEMORY_2026_08_09` sec 3.3, "4.29 GB each at N=16384") for exactly the
configuration that campaign was aimed at — the tilted per-order runs of a DOE fan. Every
full-grid screen in that group doubles.

**Recommended fix.** `env = np.asarray(env) * complex(np.exp(1j * k0 * _own * (_ob - 1.0)))`
at all three sites (a one-token change, bit-identical on a complex128 chain). While there:
`_carrier_step_fast`'s NumPy branch (`carrier.py:1185–1191`) computes the complex128 product
and then `astype`s back, which materialises one extra full-grid complex128 temporary per
leg for a complex64 field; the same `complex(...)` cast removes it.

---

### [P2] `carrier_field.CarrierSpec.phasor_on` builds every reference phasor whole-grid in complex128 — `carrier_field.py:432`

**What is wrong.** `phasor_on` calls `_exact_sphere_eikonal` (float64 whole grid), then
`np.exp((sign*1j*k)*S)` (complex128 whole grid), then `_tilt_ramp(...)` and
`_tilt_exactness_phase(...)` **without passing `dtype=`** — even though both helpers grew a
`dtype=` parameter in v5.44 precisely so a complex64 field keeps its dtype and the
full-grid complex128 transient disappears (`carrier.py:5354`, `carrier.py:2958`,
`_phasor_rows` at `carrier.py:741`).

**Evidence** (`p7_odd_dtype.py`): a `CarrierField` with a complex64 envelope returns
`full_field()` → `complex128` and `carrier.phasor_on(...)` → `complex128`.
`re_reference` (`carrier_field.py:1387–1390`) builds **two** such phasors and multiplies
them, so the peak there is ~3 full-grid complex128 arrays regardless of the stored dtype.

**Impact.** `CarrierField` is the storage/aggregation layer (`aggregate` at
`carrier_field.py:1480` also hard-codes `acc = np.zeros(grid.shape, dtype=np.complex128)`),
i.e. the path a K-order fan is summed on. At N = 16384 each avoidable complex128 grid is
4.29 GB.

**Recommended fix.** Give `phasor_on` a `dtype=None` parameter, route the sphere through
`_phasor_rows` when it is `complex64`, and forward `dtype` to `_tilt_ramp` /
`_tilt_exactness_phase`. `full_field()` / `from_full_field()` / `re_reference` pass
`self.envelope.dtype`. Size `aggregate`'s accumulator from
`np.result_type(np.complex64, *[f.envelope.dtype for f in fields])` the way the multi
orchestrator already does (`carrier.py:10300`).

---

### [P2] The radial carrier phase and the tilt ramp are separable and are not built that way — 14.4x / 6.3x measured — `carrier.py:790` (`_radial_carrier_phase`), `carrier.py:5354` (`_tilt_ramp`), `carrier.py:2085` (`_rereference`)

**What is wrong.** All three build a full 2-D `meshgrid`, form `r2 = X*X + Y*Y` (or
`L*X + M*Y`), and take **one `np.exp` of N² complex values**. `exp(i a (x²+y²))` factorises
exactly as `exp(i a x²) ⊗ exp(i a y²)` (and likewise for the ramp and for the
`_rereference` difference of parabolas, including the decentred `centre=` form), which
needs `2N` exponentials and one outer product.

**Evidence** (`p10_perf.py`, `p10b_tf.py`; λ = 1.31 µm, dx = 2 µm, R = 50 mm, CPython 3.14 /
numpy 2.4.6):

| build | N=2048 time | N=4096 time | tracemalloc peak (N=2048) |
|---|---|---|---|
| `_radial_carrier_phase` (shipped) | 872.2 ms | 1827.1 ms | 234.9 MB = **3.50** complex128 grids |
| separable outer product | 60.5 ms | 182.3 ms | 67.4 MB = **1.00** grid |
| speedup / max abs difference | **14.4x** / 1.7e-13 | **10.0x** / 6.8e-13 | 3.5x less peak |

| `_tilt_ramp` (shipped, L=0.03, M=-0.02) | 549.8 ms | separable 87.0 ms | **6.3x**, max diff 1.4e-13 |

1.7e-13 rad is three orders below the float64 floor of the arguments these screens carry
(`k·r²/2R` up to ~1e5–1e6 rad → ~1e-11 rad of representation noise), i.e. the change is
inside the existing noise, not a new approximation.

**Impact.** `_radial_carrier_phase` is called by `carrier_referenced_reconstruct` /
`carrier_referenced_envelope` and directly by the chain at `carrier.py:6780`, `8480`, `8574`
— several times per leg, and on the **fine** retrace grid (up to 16384²) on the exact leg.
Extrapolating the measured N² scaling, N = 16384 costs ~29 s and ~15 GB peak where the
separable form costs ~2.9 s and 4.3 GB. That is 2.5 of the 24 grids in
`_FINE_GRID_WORK_ARRAYS`.

**Recommended fix.** In `_radial_carrier_phase`, replace the meshgrid with
`px = bld.exp(sign*1j*k*(x-x0)**2/(2R))`, `py = ...`, `return px[None, :] * py[:, None]`
(the `_phasor_rows` complex64 branch becomes `np.multiply(px, py[r0:r1, None], out=...)`,
which removes the transient it was written to remove *and* the exps). Same for `_tilt_ramp`
and `_rereference`. Keep the shipped whole-grid form behind a module flag if a byte-identity
pin needs it, exactly as `_EXACT_READOUT_SEPARABLE_BLUESTEIN` does.

---

### [P2] `_exact_envelope_tf_step` spends ~63 % of the exact carrier leg building the transfer function, and 1.44x of that is recoverable bit-identically — `carrier.py:1055`

**What is wrong.** The kernel is rebuilt from scratch every call and allocates
`ax`, `ay`, `rad`, `root`, `lin`, `phase` and `H` as separate full-grid arrays, then takes
`np.exp(1j*phase)` on a complex array. `lin` and the `k*L`, `k*M` shifts are identically
zero on the default (untilted) path but are still materialised.

**Evidence.** `p10_perf.py`, cProfile of `propagate_carrier_referenced` at N = 2048,
gap_kernel='auto' (1975 ms/call total; 'fresnel' 1632 ms/call; tracemalloc peak
268.5 MB = **4.00** complex128 grids):

```
tottime  cumtime  function
  1.036    1.632  _exact_envelope_tf_step          <- 63 % of the step is NOT the FFTs
  0.315    0.315  fft_infra._ifft2
  0.213    0.213  fft_infra._fft2
```

`p10b_tf.py`, isolating the H build at N = 2048:

| build | time | peak |
|---|---|---|
| shipped (`np.exp(1j*phase)`, tilt 0) | 568.6 ms | 268.5 MB = 4.00 grids |
| `np.cos/np.sin` into `H.real`/`H.imag` | 394.7 ms | 167.8 MB = 2.50 grids — **max difference exactly 0.0** |
| + untilted fast path + in-place ops | 292.0 ms | 100.7 MB = 1.50 grids — max difference 7.3e-12 |

**Impact.** ~1.44x faster and 1.6x less peak for **bit-identical** output; ~1.95x / 2.7x for
a reassociation-only change. At N = 16384 the 2.5 grids saved are 10.7 GB.

**Recommended fix.** (a) Replace `H = np.exp(1j*phase)` with a preallocated
`H = np.empty(shape, complex128); np.cos(phase, out=H.real); np.sin(phase, out=H.imag)` —
this is provably byte-identical and should go in unconditionally, including in
`_tf_phase_to_H` (`carrier.py:478`) and `_asm_axis` (`carrier.py:1447`). (b) Short-circuit
`L == M == 0.0` to the `kx²[None,:] + ky²[:,None]` form and do the arithmetic in place.
(c) For the multi orchestrator, H depends only on `(Ny, Nx, dx, dy, z_eff, wavelength, tilt)`
— identical across congruences that share a tilt — so a small LRU keyed on that tuple would
remove the rebuild entirely for a fan; note the chain's per-congruence tilt differs, so this
helps the probe/resize re-runs and `self_check='dx'` more than a fan.

---

### [P2] `_fourier_upsample_crop` scales out-of-place, costing one extra full FINE-grid temporary — `carrier.py:3441`

`out = out * (float(n_fine) / float(n_crop)) ** 2`. The function's own BUFFER OWNERSHIP note
(`carrier.py:3355`) establishes that `out` is a fresh `np.fft.fftshift` result that aliases
nothing, so `out *= scale` is safe. At the shipped `n_fine_cap = 16384` this is 4.29 GB of
avoidable peak, on the function the memory audit already identified as running twice per
exact final leg at the fine grid.

---

### [P3] `_freq_sq_1d` is dead code that still carries the defect D7 claims to have fixed, and its docstring asserts an identity that is false — `carrier.py:1411`, claim at `carrier.py:412`

`_freq_sq_1d` uses `- N / 2`; `_freq_sq_1d_bld` uses `- (N // 2)`. The docstring of the
latter says "identical values for `bld is np`". Measured (`p7_odd_dtype.py`):

```
N=4: identical
N=5: _freq_sq_1d [9.8696 3.5531 0.3948 0.3948 3.5531]  vs  _bld [6.3165 1.5791 0. 1.5791 6.3165]
N=7: differs likewise
```

`grep -rn "_freq_sq_1d\b"` over `lumenairy/` finds only the definition and that docstring —
it has **no call sites**. Delete it, or fix it and delete the false claim. (The 47-line D7
note at `carrier.py:384–408` is worth keeping; it just should not point at a function that
still has the bug.)

### [P3] `_asm_axis`'s band-limit mask is half a bin out of register with its own transfer function at odd N — `carrier.py:1490–1492`

`H` is built on `_freq_sq_1d_bld(N, d, bld)` (offset `N//2`) but masked with
`f = (bld.arange(N) - N / 2) / (N * d)` (offset `N/2`) **before** the `ifftshift`. At even N
these coincide; at odd N the mask is displaced by half a bin (measured 3.85e3 1/m at N = 65,
dx = 2 µm), so the Matsushima band limit keeps one bin too many on one side and drops one
too many on the other. Same one-character fix as D7. On the astigmatic focus-crossing bridge
only, and only at odd N — hence P3, but it is literally the defect class the module
documents having swept.

### [P3] The chain's `gap_kernel` is not forwarded to either focus readout or to the fine retrace — `carrier.py:8643–8656`, `8674`, `8342`, `_FOCUS_READOUT_KEYS` at `carrier.py:8890`

`gap_kernel` reaches only the gap legs (`carrier.py:8193`) and the bare final leg
(`carrier.py:8694`). `_par_kw` selects `('dx_out','N_out','standoff','centre_out',
'bandlimit','on_replica')`, and `gap_kernel` is not in `_OUTPUT_GRID_PASSTHROUGH`, so
`carrier_referenced_focus_readout`'s internal carrier leg always runs its own default
`'auto'` → `'exact'`. A caller who passes `gap_kernel='fresnel'` — whose entire documented
purpose is "pinned FP-identical to prior releases" (`carrier.py:869`) — therefore gets a
mixed chain. Either forward it or say in the chain's `gap_kernel` docstring that the readout
leg is exempt.

Related, same area: `carrier_referenced_focus_readout` has **no `tilt` parameter at all**, so
the chain's tilted paraxial-readout path (`carrier.py:8674`) propagates the final leg's
envelope with an untilted kernel. `_tilt_obliquity`'s docstring records the size of that
(`z/(1-L²)^{3/2}`, +0.32 % effective distance at 46 mrad), but the chain's own `gap_kernel`
documentation says the exact kernel "carries the tilt to all orders", which is not true on
that leg.

### [P3] The documented bias model for the `'gradient'` fit estimator is incomplete — `carrier.py:1985–2010`

The "GRID-PITCH DEPENDENCE" table attributes the whole bias to `sin(h)/h` ("a pure GRID
artefact"). There is a second, larger term at fine pitch: an amplitude-curvature coupling
through `np.gradient`'s central difference, `≈ 0.5 (dx/w)²`. Measured (`p4b_gradbias.py`,
R = 0.5 m, w = 100 µm, fixed physical field, pitch swept 16x):

```
 dx/w    h          1/(sin h/h)-1     measured bias    0.5*(dx/w)^2
 0.005   4.80e-04   3.83e-08          1.2529e-05       1.25e-05
 0.010   9.59e-04   1.53e-07          5.0116e-05       5.00e-05
 0.020   1.92e-03   6.14e-07          2.0048e-04       2.00e-04
 0.040   3.84e-03   2.45e-06          8.0216e-04       8.00e-04
 0.080   7.67e-03   9.82e-06          3.2125e-03       3.20e-03
```

i.e. the documented mechanism under-predicts by up to 4 orders of magnitude here (it happens
to dominate at the table's own high-`h` conditions). `'increment'` reads exactly 0 bias at
every one of these pitches, which is the point the table makes and which holds.

### [P3] `_sphere_parab_conversion` ignores `dy` — `carrier.py:3145`

Signature takes only `dx` and builds `y = (np.arange(ny) - ny/2) * dx` (`carrier.py:3269`).
`propagate_carrier_referenced` and `carrier_referenced_reconstruct` both accept `dy != dx`,
so a non-square-pixel chain would convert the y axis against the wrong pitch. Every shipped
call site is square, so this is latent — but it should either take `dy` or refuse.

### [P3] Structure: one 10.6 kLoC module, a 1595-line function, and 58 % of the file is prose

`p9_structure.py` (tokenize + ast census):

```
carrier.py       : total 10594  code 4066 (38.4%)  docstring 3684 (34.8%)  comment 2475 (23.4%)  blank 715 (6.7%)
carrier_field.py : total  1743  code  818 (46.9%)  docstring  677 (38.8%)  comment  134 (7.7%)   blank 206 (11.8%)

largest functions (total lines / non-docstring lines):
  1595 /  872  propagate_traced_carrier_chain     @7126
   937 /  588  propagate_traced_carrier_chain_multi @9658
   588 /  347  carrier_referenced_exact_focus_readout @4382
   516 /  370  _fine_trace_group_exit             @6407
   232 /  122  propagate_carrier_referenced       @821
```

Concrete split (the seams are already clean — no forward references cross them):

| new module | contents (current line ranges) |
|---|---|
| `carrier_core.py` | 359–1740 — backend shim, freq builders, TF kernels, `CarrierReferencedField`, `propagate_carrier_referenced`, focus-crossing bridge, the whole astigmatic per-axis block, `_build_carrier_phase` |
| `carrier_reference.py` | 1732–2340 + 2903–3300 — `reconstruct`/`envelope`/`fit_radius`/`aperture`/`_rereference`, `_exact_sphere_eikonal`, `_tilt_exactness_phase`, `_sphere_parab_conversion` |
| `carrier_readout.py` | 2340–2900 + 4264–4985 — both focus readouts, the standoff resolver, `_check_readout_replica`, the fine-grid memory model, `_fourier_upsample_crop`, `_crop_about_centre` |
| `carrier_guards.py` | 3508–3820 + 5615–6240 — `_guard_dispose`, `_warn_undeduped`, `_check_guard_action`, the P3 congruence gate, `_check_gap_paraxial`, `_gap_envelope_angular_spread`, `_check_decentred_fit` |
| `carrier_chain.py` | 4986–8800 — chain result, ABCD/chief-ray helpers, DOE normalisation, `_fine_trace_group_exit`, `propagate_traced_carrier_chain` |
| `carrier_multi.py` | 8799–10594 — multi result, worker plumbing, `propagate_traced_carrier_chain_multi` |

Import-cycle risk is confined to `carrier_chain.py`: it needs `elements/_lens_traced`
(`apply_real_lens_traced`, `TiltedCarrier`, `_carrier_residual_rms`,
`_NONCOLLIMATED_RESID_THRESH`, `TILTED_CARRIER_EXACT_EIKONAL`) and `_lens_traced` needs
nothing from the chain — every one of those is already a *function-local* import
(`carrier.py:2949`, `3846`, `6564`, `8020`, `8502`), so the split is mechanical provided
those stay local. `carrier_field.py` imports six private names from `carrier` (`:122–130`);
after the split it should import from `carrier_reference` + `carrier_guards`, which removes
its dependency on the chain entirely.

Two thirds of `propagate_traced_carrier_chain`'s 872 code lines are guard text and stage
book-keeping; extracting `_chain_group_entrance(...)` / `_chain_group_exit(...)` (the
tilted/untilted reference-imprint pairs at 8455–8501 and 8532–8582, which are exact
mirror images) and `_chain_final_landing(...)` (8606–8720) would take the function under
400 lines without touching any arithmetic.

### [P3] Smaller items

* **`CarrierField` is a mutable `@dataclass`** (`carrier_field.py:620`) while `CarrierSpec`
  (`:270`) and `FieldGrid` (`:534`) are `frozen=True`. `field.envelope = <anything>` bypasses
  every `__post_init__` invariant (shape-vs-grid, complexity, provenance canonicalisation).
  Make it `frozen=True` too; `with_provenance` already returns a new object, so nothing
  relies on mutation.
* **`_multi_worker_run` (`carrier.py:9366`) re-categorises warnings.** It runs under
  `catch_warnings(record=True) + simplefilter('always')` and the parent re-emits everything
  as `RuntimeWarning` or `UserWarning` (`carrier.py:9651`), so a caller's own filters apply
  at a different point (and a different category) than in the serial path. Since this runs
  in a spawned process there is no thread-safety issue — but the module's own
  `_warn_undeduped` docstring (`carrier.py:3553`) calls this pattern out as the wrong tool,
  and the serial/parallel results are not warning-equivalent.
* **The `readout_tile='auto'` probe pass is serial even with `congruence_workers > 1`**
  (`carrier.py:10341–10347` runs `_run(...)` directly; `_multi_parallel_results` is only
  entered at 10400). The warning at 10371 advertises "halves the chain runs, 2K -> K", so
  the probe is half the work and none of it is parallelised.
* **Stale / self-contradicting comments.** `carrier.py:1165–1168` says "'auto' ... resolves
  by BACKEND: the exact, tilt-aware kernel on NumPy, the paraxial Sziklas-Siegman one
  elsewhere" and `carrier.py:1171–1172`, four lines later, says "'auto' resolves to 'exact'
  everywhere". `carrier.py:1150` (inside `propagate_carrier_referenced`) says the fast path
  is "byte-identical to prior releases (pinned) on the default gap_kernel='fresnel'" — the
  default is `'auto'`. Both are exactly the "comment noise that hides the code" the brief
  asks about, in the one function a reader goes to first.

---

## Performance opportunities (estimated gain; how measured)

| # | Change | Measured / estimated gain | How |
|---|---|---|---|
| 1 | Separable `_radial_carrier_phase` | **14.4x** time (872 → 61 ms at N=2048), **3.5x** peak (3.50 → 1.00 grids); 10.0x at N=4096 | `p10_perf.py`, timeit + tracemalloc, max abs diff 1.7e-13 |
| 2 | Separable `_tilt_ramp` (and `_rereference`) | **6.3x** time (550 → 87 ms at N=2048), diff 1.4e-13 | `p10b_tf.py` |
| 3 | `cos`/`sin` into `H.real`/`H.imag` in `_exact_envelope_tf_step`, `_tf_phase_to_H`, `_asm_axis` | **1.44x** and 4.00 → 2.50 grids, **exactly 0.0** difference | `p10b_tf.py` |
| 4 | + untilted short-circuit and in-place TF arithmetic | **1.95x** and 4.00 → 1.50 grids, diff 7.3e-12 | `p10b_tf.py` |
| 5 | `out *= scale` in `_fourier_upsample_crop:3441` | one fine-grid temporary (4.29 GB at n_fine=16384) | code; the function's own ownership note licenses it |
| 6 | `complex(...)` on the three chain obliquity pistons | keeps a tilted complex64 chain at complex64 — halves every full-grid screen in that group | `p8_c64chain.py` |
| 7 | `dtype=` through `CarrierSpec.phasor_on` | `re_reference` peak from ~3 complex128 grids to ~1.5 complex64 | code + `p7_odd_dtype.py` |
| 8 | Cache `H` on `(shape, dx, dy, z_eff, λ, tilt)` | removes 63 % of a repeated leg's cost; helps `self_check='dx'` (which runs the whole chain twice) and the multi probe/resize re-runs | cProfile in `p10_perf.py` |

Combining 1+3+4 on a chain leg: the measured N=2048 step is 1975 ms with a 4.00-grid peak;
the TF and screen work that dominates it drops by ~2x in time and ~2.5x in peak, leaving the
FFT pair (0.53 s at N=2048) as the floor.

## Alternative algorithms / methods

1. **Collins integral / ABCD-Fresnel with a Bluestein output grid, as a drop-in for the
   Sziklas–Siegman scaling.** Collins (1970), *JOSA* **60**, 1168: for any ABCD system
   `E_out(x) = (i/(λB))∫ E_in(u) exp(-i k (A u² - 2 u x + D x²)/(2B)) du`. Written as
   chirp × chirp-Z × chirp it gives *exactly* the SS result for a quadratic carrier but with
   the output pitch chosen **freely** instead of forced to `m·dx`, which would remove the
   entire near-focus machinery in this module — `_near_focus_needs_bridge`,
   `_propagate_carrier_focus_crossing`, `_axis_bridge`, `_default_focus_standoff`,
   `_small_extent_focus_standoff_f` and the replica guard all exist because `m → 0` collapses
   the grid. Cost: one chirp-Z (≈ 3 FFTs of length `next_fast_len(N+N_out-1)`) instead of 2
   FFTs, i.e. ~2–3x per leg — but the module *already* pays a Bluestein at every readout, and
   the separable route it ships (`_EXACT_READOUT_SEPARABLE_BLUESTEIN`, measured 2.4–6.7x
   faster there) applies. It also makes the focus a non-event rather than a special case, and
   would have prevented the P1 above outright.
2. **Shifted/scaled Fresnel via chirp-z for an arbitrary output pitch.** Muffoletto, Tyo &
   Zawodny, *Opt. Express* **15**, 5631 (2007); Kelly, *Appl. Opt.* **53**, 2861 (2014)
   ("Numerical calculation of the Fresnel transform"); Hu, Wang & Cao, *Opt. Express* **28**,
   5073 (2020) (efficient Bluestein/zoom-FFT diffraction). Same machinery as (1) but stated
   directly for free space; the sampling conditions for the chirp-z form are worked out in
   Kelly, which is what a guard should be written against rather than the geometric-margin
   model `_FOCUS_STANDOFF_MARGIN` uses.
3. **Non-paraxial: replace `sqrt(k²−q²)` on the SS-scaled envelope.** The SS coordinate
   transform is a theorem about the *paraxial* wave equation; applying the exact ASM kernel
   to the reduced-distance envelope (what `'auto'` does) is neither the exact paraxial answer
   nor the exact Helmholtz one. My measurements show the two kernels agree to ~1e-4 relative
   on a real carrier leg (through-focus relL2 4.026e-02 vs 4.026e-02, phase RMS 8.855e-02 vs
   8.853e-02 rad; `p1d_phase_struct.py`), so nothing is broken — but the in-code claim that
   the exact kernel is "the physically correct transfer function" for a *scaled-frame* leg is
   stronger than the derivation supports. The honest non-paraxial route for a fast gap is
   Wave-Propagation-Method / split-step with the Lax series (Lax, Louisell & McKnight,
   *Phys. Rev. A* **11**, 1365 (1975)) or a Rayleigh–Sommerfeld convolution on the reduced
   frame.
4. **Multi-carrier (`_multi`): Gaussian-beam / Hermite–Gauss decomposition instead of
   K independent chains.** Greynolds' Gaussian beam decomposition (*Proc. SPIE* **560**, 1985)
   and its modern "beamlet" descendants carry each beamlet's ABCD `q` analytically through the
   prescription and sum at the image; per-beamlet cost is O(1) rather than O(N² log N), and the
   coherent sum is exact in the piston by construction. For a 32-order Dammann fan on a shared
   grid that is orders of magnitude cheaper than 32 full chains, at the price of a
   decomposition step and of a beamlet-count convergence study. It also removes the
   `_chain_chief_ray_at_target` prediction-vs-tracked reconciliation at `carrier.py:10488`
   (which currently raises `RuntimeError` "this is a library bug" if the two disagree).
5. **For `_default_focus_standoff`**: a two-point fixed point on the *measured* beam (fit the
   envelope's residual curvature with the existing `_fit_carrier_inv`, compose `1/R_eff =
   1/R + 1/R_env`, resolve `f` against `R_eff`) is ~10 lines and removes the P1's premise.
   This is what a Gaussian `q`-parameter propagation does natively.

## Code organization observations

See the P3 structure entry above for the census and the proposed split. Two further notes:

* The comment mass is not uniformly noise — the `_FOCUS_STANDOFF_*` block (`carrier.py:181–328`)
  and the `_MULTI_CONGRUENCE_*` block (`carrier.py:3617–3820`) are genuine derivations with
  measured tables and belong in `docs/`, linked from a two-line comment. As they stand, the
  reader of `_default_focus_standoff` must get through 150 lines of prose to reach 30 lines of
  code, which is a large part of why the carrier-vs-beam premise in that derivation has gone
  unchallenged.
* Guard-message strings are a substantial fraction of the "code" lines
  (`_check_readout_replica` is 183 lines for ~20 lines of logic; `_check_gap_paraxial` 178
  for ~40). Moving the long-form text into a module-level dict of templates keyed by guard
  name would make the control flow of those functions readable without losing a word.

## Unverified suspicions

* **Piston fidelity through the focus-crossing bridge.** The through-waist path leaves a
  systematic on-axis piston error of +1.65e-02 rad and an amplitude-weighted residual phase
  RMS of 3.70e-02 rad (1.68e-02 after removing piston + defocus) on a converged grid where
  the amplitude error is 9.1e-04 and power is conserved to 1e-6 (`p1c_focus_decomp.py`,
  `p1d_phase_struct.py`; w0 = 4 µm, R = −30 mm, +60 mm through the waist, ext = 5 w). The
  module docstring's claim ("<0.1 % windowed r2m/EE") holds on its own metric — I measure
  r2m to +7.6e-05 relative — but that metric is phase-blind. I could not establish whether
  the piston error is `_BRIDGE_ZR_FACTOR`-limited (the `R_b = −R_a` geometric continuation
  is 2.8 % off the true Gaussian radius at 6 `zR`) without a `_BRIDGE_ZR_FACTOR` sweep.
  It matters for `propagate_traced_carrier_chain_multi(recombine='coherent')`, where per-order
  piston errors add. **Confirm by:** sweeping `_BRIDGE_ZR_FACTOR` ∈ {3, 6, 12, 24} and
  checking whether the piston error falls as `1/factor²`.
* **`_multi` K-carrier consistency.** I read the recombination arithmetic
  (`carrier.py:10503–10529`) and satisfied myself that the tile placement is exact (the
  window centre is snapped to the common lattice at `carrier.py:10223–10226` **and** passed
  as `centre_out`, so no sub-pixel resample occurs), but I could not run a K = 1 vs K = 2
  decomposition test in budget. **Confirm by:** the same field passed twice with weights
  `w` and `1−w` against one run at weight 1 (should be exact to FP), and then the same
  *physical* field referenced to two different carriers (which the P1 above predicts will
  **not** agree once either carrier is off the beam).
* **`carrier_referenced_reconstruct` never warns on an undersampled carrier** — measured
  (`p2_alias.py`) at 4.796 rad/px of edge phase step (1.53x Nyquist) with zero warnings,
  whereas `carrier_referenced_fit_radius` has a whole `on_aliased` machinery for the same
  quantity and `carrier_referenced_envelope` at least carries a NOTE. I am **not** reporting
  this as a defect: reconstructing an aliased carrier is the method's deliberate design (the
  aliasing cancels pointwise against the consumer's own de-chirp, as
  `SPHERE_PARAB_CONVERSION_EXACT` argues at length), and the one place it would matter — the
  readout's stop-plane reconstruct — is protected by the co-moving contraction (`h_stop =
  h_in · m`, so a converging leg always *improves* the sampling; measured 3.747 → 0.042 rad/px
  in `p3_readout.py`). Worth a one-line docstring note on `reconstruct` saying so.

## Checked and found correct (brief)

* **The Sziklas–Siegman transform itself.** Diverging carrier, matched Gaussian oracle,
  `w0 = 30 µm`, `R0 = 13.31 mm`, N = 512 at ±4 w (`p1_gauss.py`): relL2 1.06e-07 / 1.99e-07 /
  2.66e-07 and phase RMS 7.8e-08 / 1.6e-07 / 2.2e-07 rad at m = 1.49 / 2.95 / 10.73, with
  `P_carrier/P_analytic = 1.000000` at all three — the residual is the `exp(−16) = 1.1e-07`
  grid truncation, i.e. the transform is exact to the oracle's own floor. `R_out = R + z`,
  `dx_out = m·dx`, `z_eff = z/m`, the `1/m` amplitude and the `exp(i k z²/R_out)` piston all
  check out term by term against the derivation, and the total on-axis phase composes to
  exactly `k z` (`k z_eff` from the kernel + `k z²/R_out`).
* **Converging carrier through focus.** Power conserved to 1e-6, peak ratio 0.9977 and r2m
  to +7.6e-05 on a converged grid; the bridge picks the hand-off planes, flips the carrier
  and re-attaches without loss of energy. Grid-extent dependence is exactly as the docstring's
  "±>2.4 w" caveat says (ext 2.6 → 2.3 % power clipped, ext 3.5 → 0.09 %, ext 5.0 → 1e-6).
* **The paraxial focus readout against an analytic oracle** — relL2 1.05e-03 (NA 0.05,
  ext 4), 3.81e-03 (NA 0.10, ext 4), 1.77e-02 (NA 0.05, ext 2, the documented narrow-grid
  branch); peak position exactly on axis; EE(2 w0) 0.99964 vs 0.99964.
* **`_check_readout_replica`'s V3 geometry** (`2|centre_out| + N·dx ≤ period`, per axis) —
  derivation verified against the Bluestein period `N_in·d_in`; it fired correctly on every
  over-wide window I threw at it, including at `centre_out = 0`.
* **Sign conventions** against `CONVENTIONS.md` §7: `exp(+i k r²/2R)` for `R > 0` diverging
  in `_radial_carrier_phase`, `_exact_sphere_eikonal`, `_build_carrier_phase`, `_rereference`,
  and `reconstruct`/`envelope` as an exact `±1` pair (round trip 2.2e-16 at both even and odd
  N); forward `exp(+i k z)` piston carried by both kernels; `_shift_envelope`'s ramp
  `exp(−2πi f s)` gives `env(x − s)` as documented, and `_crop_about_centre`'s
  `_shift_envelope(ec, −rx, −ry)` is the correct sign for a crop centred at `(x0, y0)`.
* **`_tilt_obliquity`** — `1/sqrt(1−L²−M²)`, and the expansion it is derived from, checked
  against the exact ASM phase; the chain's chief-ray advance `L·z·ob` and piston
  `k z (ob − 1)` are consistent with it and with each other.
* **DOE handling** — the pre/post-DOE gap split (`pend_gap` / `pend_own`), the chief-ray
  advance with the *old* tilt before the grating and the *new* one after, and the grating
  piston `amp·exp(i k (dL(x_c−ox) + dM(y_c−oy)))` evaluated at the chief ray in the DOE plane
  are all self-consistent.
* **Astigmatic path piston book-keeping** — each 1-D leg carries `exp(i k z)` and the driver
  divides one back out (`carrier.py:1664`); amplitude `1/sqrt(m_x)·1/sqrt(m_y)` reduces to
  `1/m` for `m_x = m_y`, matching the scalar path.
* **`R → ∞` guards** — `R = 1e3 … 1e12` all fit to +2.0e-04 (the `(dx/w)²/2` bias above, not a
  division problem), `m` is exactly 1.000000000000 and `dx_out/dx = 1` at `R = inf`, `1e12`,
  `1e15`; `propagate(R=1e15)` differs from `propagate(R=inf)` by 4.8e-15; `_rereference`'s
  `_inv` returns 0 for `±inf` so a collimated re-reference is a no-op, not a NaN;
  `_exact_sphere_eikonal` returns zeros (not NaN) for `R = ±inf` and `R = 0`.
* **`carrier_referenced_aperture`** — transmission 0.721850 exactly equals the retained power
  fraction (no renormalisation), and `refit_carrier=True` on an already-flat envelope returns
  `R` unchanged to 8 digits.
* **`_fourier_upsample_crop`'s `(n_fine/n_crop)²` rescale** and the `n_crop > n` guard; the
  `_crop_about_centre` integer-slice + sub-pixel-shift decomposition.
* **`_piston_phase`** (`carrier_field.py:486`) — differences the two pistons before
  exponentiating and folds to the nearest wavelength multiple, so a millimetre-scale absolute
  path costs the accuracy of the micron-scale difference, as documented.
* **`re_reference` ordering** (`carrier_field.py:1350–1397`) — resample the smooth envelope
  *then* apply the analytic carrier difference, with the piston delta folded in last; the
  `reconstruct`-term Nyquist bound is over-strict for an envelope-only operation but is
  deliberate and documented (`carrier_field.py:1215–1226`).
* **No module-level mutable caches** in either file (only the documented flags
  `SPHERE_PARAB_CONVERSION_EXACT`, `_EXACT_READOUT_SEPARABLE_BLUESTEIN` and the worker-state
  dict `_MULTI_WORKER_STATE`, which lives only inside a spawned worker process), no
  import-time side effects, no bare `except:`, and every broad `except` I found re-raises or
  falls back to a documented, measured path.
* **A two-group chain against a brute-force ASM + `apply_real_lens_traced` chain**
  (`p5_chain.py`; point source 30 mm → singlet → 40 mm → singlet → half the image distance,
  N = 2048, the same element call on both arms so only the transport differs): power ratio
  **1.000067**, r2m 1.13882 mm vs 1.14389 mm (0.44 %), peak ratio 1.0035, centroids on axis
  to 1e-10 m on both arms.
