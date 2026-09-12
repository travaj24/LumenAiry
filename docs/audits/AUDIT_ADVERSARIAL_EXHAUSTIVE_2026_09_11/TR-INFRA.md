# TR-INFRA audit — `lumenairy/elements/_lens_traced.py` lines 1–6731 (the infrastructure under `apply_real_lens_traced`)

All line numbers are `lumenairy/elements/_lens_traced.py` unless stated.
Repro scripts live under
`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/TR-INFRA/`.
**No repository file was created, modified or deleted.** Two probes monkey-patch a
module attribute *in-process only* (stated where used).

## Scope read (line by line)

| range | content | read |
|---|---|---|
| 1–600 | module header, cupy/numba lazy loaders, kwarg-default cache, `_copy_prescription`, `_kwarg_differs_from_default`, all ray-density / halo / origin constants, `set/get_lens_parallel_amp`, `_prescription_has_field_frame` | full |
| 594–1114 | payload-residency block, `NewtonPayloadNotResident`, `_newton_pool_init`, `_newton_worker_payload`, `_newton_payload_blob`, `_newton_invert_chunk` | full |
| 1115–1790 | `_get_persistent_worker_pool`, cost classes / bands / deferral notes / `_pool_reuse_is_likely`, `close_worker_pool`, `_pool_memory_policy`, `_note_pool_backend_refusal`, `_reset_newton_pool_resource_state`, `_is_main_guard_test`, `_script_has_main_guard`, `_spawn_reexecuted_main_script`, `_newton_worker_bytes`, `_newton_resolve_workers` | full |
| 1801–2111 | numba Chebyshev kernel, `_resolved_cheb_backend`, `_validated_cheb_backend`, `_cheb_fit_state`, `_cheb_fit_payload`, `NewtonWorkerBackendUnavailable`, `_get_array_module` | full |
| 2114–2910 | the whole deterministic least-squares stack + C13/D14/D15 constant notes | full |
| 2911–3290 | `_Cheb2DEvaluator` (incl. `from_state`, `ev`, `ev_value_and_grad`), `_cheb_vand_2d`, `_cheb_deriv_vand_2d` | full |
| 3291–3372 | `_geometric_lens_phase` | full |
| 3375–4157 | carrier / fit-domain constants, `_decentred_fit_restriction`, `_decentred_fit_score`, `_decentred_fit_score_weight`, `_decentred_fit_spectrum`, `_decentred_fit_spectral_moment`, `_decentred_fit_crossover` | full |
| 4158–4475 | `_input_beam_amp_radius`, `_carrier_residual_rms`, `_input_tilt_stats`, `TiltedCarrier`, `_tilted_carrier_parts` | full |
| 4476–5555 | `_compute_carrier` (all four branches), C6/C8/C14 constants | full |
| 5556–6327 | `_TracedExitSupport`, `_ResidualEikonal`, `_fit_residual_eikonal` | full |
| 6328–6730 | `_sample_local_tilts`, `_reverse_prescription`, `_opl_by_backward_trace` | full |

Cross-read for context/confirmation (not owned): `_traced_flags.py` (flag contract),
`raytrace/trace.py:284–600` (`validate_prescription`, `surfaces_from_prescription`),
`raytrace/surface.py:330–470`, `elements/lenses.py:180–270` (`surface_sag_general`),
`elements/_lens_real.py:4716–4728`, `_math/chebyshev.py`, `_lens_imap.py:494–600`,
`apply_real_lens_traced` docstring + the three `phase_analytic_lens` / `delta_phase` /
Newton / pool-dispatch call sites in TR-MAIN's range (context only).

**Not reached:** the GPU/CuPy branches (no cupy installed — desk-checked only);
`_decentred_fit_spectrum` / `_decentred_fit_crossover` were read and desk-checked but
not numerically exercised (`DECENTRED_FIT_PREDICTOR = False`, so they are dead on the
shipped path); `_pool_reuse_is_likely` / `_note_pool_deferral` were read and their
arithmetic desk-checked, not driven by a real multi-group chain.

---

## Findings

### [P1] `_geometric_lens_phase` raises `AttributeError` on every refracting prescription — `fast_analytic_phase=True` is dead
`_lens_traced.py:3359` (call sites 8878, 8928)

```python
from .. import raytrace as _rt          # line 3333 -- the PACKAGE
...
sag = _rt._surface_sag_xy(X, Y, surf)   # line 3359 -- lives in raytrace.SURFACE
```

`_surface_sag_xy` is defined in `lumenairy/raytrace/surface.py:391` and is **not**
re-exported from the `lumenairy.raytrace` package (`hasattr(rt, '_surface_sag_xy')`
is `False`, before and after importing the submodule — the package has no lazy
`__getattr__`). The loop at 3354–3360 `continue`s only when `|n_after − n_before| <
1e-15`, so the line is reached by every prescription that actually refracts.

**Evidence** (`p3b_fastphase.py`):

```
direct _geometric_lens_phase on a refracting prescription:
  AttributeError: module 'lumenairy.raytrace' has no attribute '_surface_sag_xy'
apply_real_lens_traced(fast_analytic_phase=True):
  preserve_input_phase=True   -> AttributeError  (the DEFAULT for that flag)
  preserve_input_phase='remap'-> ValueError (needs ray_density; unrelated)
  preserve_input_phase=False  -> OK  (the flag is simply never read)
control fast_analytic_phase=False -> OK
```

`tests/unit/test_v5_1_0_agent_a.py:653–658` documents this in a test docstring
("a pre-existing latent bug unrelated to this agent's scope") and deliberately uses a
*no-refraction* prescription so the loop short-circuits before the line. Nothing in
`docs/audits/*` or `CHANGELOG.md` records it as an open item, and no test exercises
the live path.

**Impact.** A documented public kwarg (`fast_analytic_phase`, signature line 6761,
docstring promising "~10 nm OPL error" and a skipped ASM leg) has never worked. Any
caller that sets it gets a hard crash, not a degraded result — so no wrong numbers,
but a dead feature plus a dead measured-cost claim.

**Fix.** `from ..raytrace.surface import _surface_sag_xy as _surface_sag_xy` (or
`from ..raytrace import surface as _rt_surface`) and call that. Add one test that
runs `_geometric_lens_phase` on a *refracting* singlet.

---

### [P1] `apply_real_lens_traced` silently cancels a prescription `form_error` (254x suppression) on the DEFAULT path
mechanism spans `_lens_traced.py:8902 / 12363` and `raytrace/surface.py` (no `form_error` field)

`apply_real_lens` implements `form_error` as an additive sag map
(`_lens_real.py:3903, 6060`). The ray model does not: `Surface`
(`raytrace/surface.py:169–219`) has no `form_error` field and
`surfaces_from_prescription` never reads the key. The traced assembly is

```
E_out          = E_analytic * exp(i * delta_phase)
delta_phase    = k0 * opl_traced - phase_analytic_lens
```

`E_analytic = apply_real_lens(E_in)` carries `+phi_form`; `phase_analytic_lens =
angle(apply_real_lens(reference))` carries `+phi_form` too; `opl_traced` carries none.
The two analytic legs cancel and the figure error vanishes from the answer.

**Evidence** (in-line script, 250 nm PV astigmatic figure error on S1, N-BK7
100/−100 singlet, N=256, dx=25 µm, λ=587.6 nm, `ray_subsample=8`):

```
traced   : max|dphi| with vs without form_error = 2.5698e-03 rad
traced   : max|dAmp|/peak                       = 1.2716e-04
analytic : max|dphi| with vs without form_error = 6.5410e-01 rad
```

i.e. **254x suppression** of a real surface figure error, with no warning, no
refusal, and no mention in the `apply_real_lens_traced` docstring (the string
`form_error` does not occur anywhere in `_lens_traced.py`).

**Impact.** A prescription key that is documented for this prescription format and
that `apply_real_lens` honours is silently a no-op in the traced model. Any
tolerancing or figure-error study routed through `apply_real_lens_traced` returns
the *nominal* answer. (`analysis/through_focus.py` happens to use `apply_real_lens`,
so the shipped tolerancing helper is unaffected — but nothing stops a user swapping
the model.) The same cancellation applies to **every** phase-only prescription
feature the wave model implements and the ray model does not.

**Fix.** Either (a) refuse/warn at the top of `apply_real_lens_traced` when any
surface carries `form_error` (cheapest, honest), or (b) add the form-error screen to
`delta_phase` explicitly after the subtraction (`delta_phase += k0 * sum_i (n_after −
n_before)_i * form_error_i` with the same sign convention `apply_real_lens` uses), or
(c) carry it into `Surface` so the ray trace sees it. (a) is the only one that cannot
be subtly wrong.

---

### [P1] `_reverse_prescription` does not negate the aspheric / freeform / tilt / `sag_callable` departures
`_lens_traced.py:6524–6558`

Reversing the propagation direction is the reflection `z → −z`, under which
`sag(h) → −sag(h)`. The code negates `radius` and `radius_y` (correct — that flips the
conic term) and leaves `conic` / `conic_y` alone (correct — invariant). It does **not**
negate the other sag terms:

| key | needed | done |
|---|---|---|
| `radius`, `radius_y` | negate | yes |
| `conic`, `conic_y` | unchanged | yes |
| `aspheric_coeffs`, `aspheric_coeffs_y` | **negate every coefficient** | **no** |
| `freeform` (`xy_coeffs` / `zernike_coeffs` / `cheb_coeffs` / Forbes Q) | **negate** | **no** |
| `tilt` (field-frame ramp `tx*(x−dx)+ty*(y−dy)`) | **negate** | **no** |
| `sag_callable` | **negate and wrap** | **no** |
| `decenter` | unchanged (for the x→x, y→y, z→−z frame the radius flip implies) | yes |

**Evidence** (`p4_reverse.py`, section 4d): 100 mm/plano N-BK7 singlet with
`aspheric_coeffs={4: 1.0e3}` on S1, sag evaluated at h = 5 mm through
`surfaces_from_prescription` + `_surface_sag_xy`:

```
sag(forward S0 , h=5 mm) = +2.512531447e-04 m
sag(reversed S1, h=5 mm) = -2.500031447e-04 m   (should be -2.512531447e-04)
ERROR                    = +1.250000e-06 m  = +2.13 waves at 588 nm
```

Section 4e enumerates the pass-through directly: `aspheric_coeffs`,
`aspheric_coeffs_y`, `freeform`, `tilt` and `sag_callable` all come out **unchanged**.

**Impact.** `inversion_method='backward_trace'` on any aspheric, freeform,
field-tilted or `sag_callable` element traces the *wrong surface* — the base conic is
mirrored, the departure is not. Opt-in and documented experimental, hence P1 rather
than P0. The docstring at 6530–6537 states the aspheric invariance as fact
("Conic constants **and even-power aspheric coefficients** are invariant under this
reflection"), which is where the error is anchored: the conic is invariant only
because the *radius* flip already supplies the sign; the aspheric polynomial has no
radius to flip.

**Fix.** In the copy loop:
`rs['aspheric_coeffs'] = {p: -c for p, c in ...}` (and `_y`), negate every freeform
coefficient dict, `rs['tilt'] = (-tx, -ty)`, and
`rs['sag_callable'] = (lambda f: (lambda x, y: -np.asarray(f(x, y))))(f)`.
Add a round-trip test (`reverse(reverse(P))` sag ≡ `P` sag pointwise) and the
forward/backward OPL equality test in `p4_reverse.py` section 4c, which already
passes to 5e-18 m on the plain-conic case.

---

### [P2] `_sample_local_tilts`: `np.roll` wraps the tilt field, and the default smoothing SPREADS the wrap error 12 columns inward
`_lens_traced.py:6438–6441`, docstring claim at 6436–6437

```python
E_shift_x = np.roll(E_in, -1, axis=1)      # last column differenced against column 0
grad_phi_x = np.angle(E_shift_x * np.conj(E_in)) / dx
```

The docstring says "the rolled-into-the-boundary pixels get low weights after the
amplitude mask". The mask is `amp > 1e-3 * amp.max()` (line 6448), which is satisfied
at the boundary by any field that fills the grid — a plane wave, a truncated or
top-hat beam, a post-DOE multi-order field, i.e. exactly the inputs this function
exists for.

**Evidence** (`p7_tilts.py` §7a; uniform-amplitude plane wave, L0 = 0.03, M0 = −0.02,
λ = 1.31 µm, dx = 4 µm, N = 256):

```
sigma=0.0: core max|L-L0|=4.8e-16   WHOLE-GRID max|L-L0|=1.475e-01
           contaminated columns 255..255 (0.4 % of pixels)
sigma=4.0: core max|L-L0|=2.1e-17   WHOLE-GRID max|L-L0|=8.111e-02
           contaminated columns 244..255 (4.7 % of pixels)
```

The wrapped column reads L = −0.1175 against a true +0.03 (error **5x the tilt
itself**), and the shipped σ = 4 px amplitude-weighted Gaussian smears it over
**12 columns / 4.7 % of the grid** rather than suppressing it — the amplitude weight
is uniform there, so it has nothing to down-weight.

**Impact.** `tilt_aware_rays=True` launches the outer ring of rays with a grossly
wrong direction; `inversion_method='backward_trace'` does the same at the exit plane.
Both are opt-in. Launch positions in that band feed the entrance→exit fit.

**Fix.** Use the same edge-safe pattern the two sibling estimators already use —
`E[:, 1:] * conj(E[:, :-1])` with midpoint coordinates (as in `_compute_carrier`
lines 4583–4587 and `_fit_residual_eikonal` lines 6242–6260) — or, minimally, zero
the last row/column of `L_grid`/`M_grid` before smoothing and exclude them from the
smoothing denominator.

---

### [P2] `_sample_local_tilts`: forward-difference tilt is stored a HALF PIXEL off, with a measured dx/(2R) bias
`_lens_traced.py:6440–6441` + `6514–6520`

`angle(E[i+1] conj(E[i]))/dx` estimates `dphi/dx` at **i + ½**, but it is stored at
index `i`, and `map_coordinates` is then sampled at `pix_x = entrance_x/dx + N_x/2`,
i.e. the value is treated as the tilt **at** pixel `i`. Both sibling estimators
in this file build explicit midpoint coordinates for exactly this reason
(`_compute_carrier`: `_xmid = 0.5*(xax[1:] + xax[:-1])`, line 4653;
`_fit_residual_eikonal`: `xs = 0.5*(xax[idx] + xax[idx+1])`, line 6246).

**Evidence** (`p7_tilts.py` §7b, uniform-amplitude spherical wave, σ = 0 so the
smoothing cannot mask it, dx = 4 µm):

```
R = 0.10 m : mean(L_sampled - L_true_at_pixel)    = +2.0000e-05
             predicted half-pixel bias dx/(2R)    = +2.0000e-05
             mean(L_sampled - L_true_at_MIDPOINT) = +9.1e-21     (exact)
             relative bias                        = +2.326 %
R = 0.05 m : +4.0000e-05 vs predicted +4.0000e-05, relative bias +2.326 %
```

The bias vanishes identically for a uniform tilt (a plane wave is exact) and grows as
1/R, so it is invisible on the collimated fixtures and systematic on the
converging/diverging ones the function is advertised for.

**Impact.** A systematic radial launch-direction offset of `dx/(2R)` rad, coherent
across the pupil, on both opt-in consumers. Over a 100 mm system at R = 50 mm,
dx = 4 µm that is 4 µm of transverse ray error at the far surface.

**Fix.** Either subtract half a pixel at the sampling site
(`pix_x = (entrance_x − origin_x)/dx + N_x/2 − 0.5` for `L`, and `pix_y … − 0.5` for
`M`, sampled from separate grids), or switch to the sibling midpoint construction.
The half-pixel shift must be applied per-axis, because `L` is midpointed in x only
and `M` in y only.

---

### [P2] `_compute_carrier` ndarray branch truncates instead of rounding the grid index (measured −0.494 px bias)
`_lens_traced.py:4563–4569`

```python
fx = np.clip((xq - _org_x) / dx + N / 2.0, 0, N - 1).astype(np.int64)
```

`.astype(np.int64)` truncates toward zero; after `+ N/2` the argument is
non-negative, so this is `floor`, not the "nearest-neighbour" the sibling docstring
at 4423–4424 claims. The result is a systematic −½-pixel sampling offset rather than
an unbiased ±½-pixel quantisation.

**Evidence** (`p8_misc.py` §8c; explicit ndarray carrier = exact sphere R = 50 mm,
N = 512, dx = 8 µm, 4001 query points across ±0.6 mm):

```
mean (x_sampled - x_query)/dx : shipped .astype(int64)  -0.4936 px
                                np.rint                 -0.0000 px
mean (L_returned - L_true)    = -7.8975e-05    (prediction -0.5*dx/R = -8.0000e-05)
max  |L_returned - L_true|    =  1.5800e-04    (|L| span 1.1999e-02)  -> 1.3 % of span
```

**Impact.** An explicit-ndarray carrier (opt-in) launches every ray with a coherent
−dx/(2R) direction-cosine bias and evaluates the H6 entrance eikonal `W` half a pixel
off. Max error 1.3 % of the tilt span on this fixture; `np.rint` halves it and
removes the bias.

**Fix.** `np.rint(...).astype(np.int64)`, or better, `map_coordinates(..., order=1)`
for a bilinear lookup (the `'auto'` and `TiltedCarrier` branches are analytic and
exact; the ndarray branch is the only quantised one). Note this changes bits, so it
needs its own fail-before flag if the byte-identity contracts must hold.

---

### [P2] The numba Chebyshev kernel is ALLOCATION-bound: 4 × `np.empty` per SAMPLE inside `prange`; a bit-identical blocked variant is 1.63x faster
`_lens_traced.py:1862–1876`

```python
for i in _prange(N):
    ...
    Tu = np.empty(max_order + 1); Tv = np.empty(max_order + 1)     # per SAMPLE
    dTu = np.zeros(max_order + 1); dTv = np.zeros(max_order + 1)   # per SAMPLE
```

Four NRT heap allocations per query point. The signature is visible in the
cost-vs-order curve: a fixed ~100–200 ns/point floor with a marginal cost of only
1.8–3.9 ns/point/term, and the curve is **non-monotonic in order** (order 4 is slower
than order 6 and order 10).

**Evidence** (`p12d.py`, 4 Mpt, 9 interleaved reps, min wall, 20 numba threads):

```
order  6 (M= 28): shipped 449.5 ms (112.4 ns/pt)   blocked 275.5 ms (68.9 ns/pt)  1.63x  max|diff| = 0.000e+00
order 10 (M= 66): shipped 376.7 ms ( 94.2 ns/pt)   blocked 293.5 ms (73.4 ns/pt)  1.28x  max|diff| = 0.000e+00
```

and from `p12_perf.py` (no tracemalloc): ns/pt/term falls 11.5 → 3.9 → 3.5 → 1.9 →
1.75 from order 4 to 12 while ns/pt stays flat at 109–172 — i.e. the per-sample
constant dominates everything below ~order 12. The blocked variant (identical
arithmetic, scratch allocated once per 512 samples) is **bit-identical**
(`max|diff| = 0.0` on all three outputs at every order tested), which is the property
that matters given the pool/serial byte-identity contracts throughout this module.

Corroborating signal: under `tracemalloc` (which hooks allocations) the shipped
kernel drops 44x (5.78 → 0.13 Mpt/s) while the pure-NumPy fallback drops only 1.3x.

**Impact.** The in-code table at 960–966 puts the Newton step at 1.5 % of a group's
wall at the default backend, so this is worth ~0.6 % of a traced call — but it is the
same kernel the inverse-characteristic evaluator and the fit scorers call, and it is
free: the change is a loop-nest restructure with a proven bit-identity.

**Fix.** Wrap the `prange` body in a block loop (`for b in prange(nblocks): for i in
range(i0, i1):`) with the four scratch arrays hoisted to the block level. Block size
512 measured above; the value does not change the arithmetic.

---

### [P2] The pure-NumPy Chebyshev fallback costs 200 float64 per query point, unchunked
`_lens_traced.py:3232–3250`

```python
Tu = _cheb_vand_2d(u, order, xp); Tv = ...; dTu = ...; dTv = ...    # 4 x (order+1, n)
Tu_K = Tu[self._K1]; Tv_K = Tv[self._K2]; dTu_K = ...; dTv_K = ...  # 4 x (M, n) GATHERS
f    = xp.sum(c_b * Tu_K  * Tv_K , axis=0)                          # 2 temporaries x (M, n)
fx_u = xp.sum(c_b * dTu_K * Tv_K , axis=0)
fy_v = xp.sum(c_b * Tu_K  * dTv_K, axis=0)
```

**Evidence** (`p12b_perf.py`, tracemalloc peak, order 6 / M = 28):

```
n =   100 000  numba : peak      5.60 MB =   7.0 float64/point
n =   100 000  numpy : peak    160.00 MB = 200.0 float64/point
n = 1 000 000  numba : peak     56.00 MB =   7.0 float64/point
n = 1 000 000  numpy : peak   1600.00 MB = 200.0 float64/point
```

and throughput (`p12_perf.py`, no instrumentation, 16.8 Mpt): numba **5.78 Mpt/s**,
numpy **0.46 Mpt/s** — **12.6x**, better than the docstring's "typical 3–10x".

**Impact.** 200 float64/point with no chunking: 1.6 GB at 1 Mpt, and 26.9 GB at
4096² = 16.8 Mpt. This is the branch taken on any box without numba (and the
**required** branch for CuPy). Every other large-array site in this module is
row-blocked against exactly this failure mode (`_CHEB_FIT_CHUNK_ENTRIES`,
`_det_block_rows`, `_input_beam_amp_radius`'s bands, `sag_chunk_rows`); this one is
not.

**Fix.** Chunk `ev_value_and_grad`'s pure-xp branch over the query axis with the same
entry budget `_CHEB_FIT_CHUNK_ENTRIES` uses (`step = budget // (4*M)`), and fuse the
three reductions with one shared `Tu_K*Tv_K` temporary. Elementwise + a sum over a
fixed axis is chunk-order-independent, so bits are preserved.

---

### [P2] The C13 step-down criterion is measured only where the fit was ALLOWED to look, but the Newton loop evaluates it over the whole launch square
`_lens_traced.py:2778–2782` (`_lstsq_residual`), 2899–2908, 2149–2158 (the rationale note)

`LSTSQ_CONDITIONING_STEPDOWN` scores the two candidate solves on `||b − A x||`, where
`A`/`b` are the **retained** rows. On the concentric branch the restriction is a hard
NaN mask, so those rows are the in-disc ones only. The Newton loop then evaluates the
resulting polynomial over `|u|, |v| <= 0.999` (the `bound` at line 10332/10343),
i.e. everywhere — including the region the criterion never looked at.

**Evidence** (`p2b_extrap.py`; 129² launch lattice, hard disc r <= 0.5 R_launch,
smooth radial "OPL" target, candidates evaluated on the WHOLE square):

```
order  6: peak |f| in disc = 2.4751e-04
   det(refined) vs QR : max|df| in disc 4.2e-18 ; whole square 6.3e-16 (2.6e-12 of peak)
   raw-NE       vs QR : max|df| in disc 1.5e-15 ; whole square 9.7e-12 (3.9e-08 of peak)
order  8:
   det(refined) vs QR : 8.4e-18 in disc ; 9.5e-14 square (3.8e-10 of peak)
   raw-NE       vs QR : 5.6e-14 in disc ; 9.1e-09 square (3.7e-05 of peak)
order 10:
   det(refined) vs QR : 3.8e-15 in disc ; 1.99e-08 square (8.0e-05 of peak)
   raw-NE       vs QR : 7.7e-13 in disc ; 2.61e-06 square (1.05e-02 of peak)
```

Two candidates that agree **in-disc to 4e-15** differ by **1.05 % of the in-disc peak**
over the square at order 10. The C13 margin `_LSTSQ_RESID_MARGIN = 1e-6` is applied to
a number that is blind to that entire difference; "ties go to the shipped path"
therefore means "ties in the region the criterion can see".

At the **shipped** `newton_poly_order = 6` the effect is 3.9e-08 of peak — harmless,
and I say so. It becomes 1e-4…1e-2 of peak only at order 10+, which is
`newton_poly_order=10` on the concentric hard-mask branch — a configuration the code
itself records as measured-harmful (line 4919: "raising the order on the hard-mask
branch makes it 86x WORSE"). This finding says *why* that configuration is not caught
by the guard that exists.

**Impact.** The measured danger is bounded at the default order. What is wrong is the
argument, not today's numbers: the note at 2149–2158 rejects the stationarity residual
in favour of `||b − Ax||` "the quantity the fit is actually defined by", but the
*consumer* is defined over a larger domain than the fit.

**Fix.** Score both candidates on the union of the retained rows **and** the
down-weighted / masked lattice points (which are available — the D1 weighted branch
already keeps them), or state the restriction explicitly in the note. Cheapest honest
change: on the hard-mask branch, add the masked lattice rows to the residual with the
D1 out-of-disc weight before comparing.

---

### [P2] The deterministic refinement matches QR on the residual but loses ~5 decades of COEFFICIENT accuracy
`_lens_traced.py:2619–2667` (`_det_refine`), 2311–2346 (the `_DET_REFINE_STEPS` note)

The `_DET_REFINE_STEPS` note validates the refinement on `||b − A x||` only ("at least
as good on all five and strictly better on three"). Measured against a gelsd +
`math.fsum`-refined oracle on an order-10 disc-masked Chebyshev Vandermonde
(`p2_lstsq.py`):

```
order 10  M=66  n_in=3209  cond(A)=2.550e+07  gram_rcond=1.502e-15
  raw normal-eq: ||b-Ax|| 1.0075x oracle   max rel coeff err 5.665e-04
  QR (geqrf)   : ||b-Ax|| 1.0000x oracle   max rel coeff err 7.961e-11
  shipped det=F: ||b-Ax|| 1.0000x oracle   max rel coeff err 7.961e-11   (C13 picked QR)
  shipped det=T: ||b-Ax|| 1.0000x oracle   max rel coeff err 4.664e-06   (refined NE)
```

The refined deterministic answer is **58 000x** less accurate in the coefficients than
the QR it replaces, while being indistinguishable on the residual. That is the same
blindness as the previous finding, seen from the coefficient side, and it is why the
det/QR field difference over the launch square above is 8e-5 of peak while the in-disc
difference is 4e-15.

At order 6 / 8 the refinement is *better* than QR (5.97e-15 / 2.44e-11 vs
2.44e-13 / 4.49e-12), so this is a degree-dependent effect, not a blanket regression.

**Fix.** Two refinement steps (the note says a second step "moves the coefficients by
~2e-13 and the residual not at all" — measured on *their* matrices, not on an order-10
hard-mask disc), or make `_DET_REFINE_MAX_CORRECTION` degree-aware, or add the
coefficient-distance-to-QR to the validation the note cites.

---

### [P2] `_script_has_main_guard` checks that a guard EXISTS, not that the heavy top-level body is inside it
`_lens_traced.py:1561–1590`

```python
ok = any(isinstance(st, ast.If) and _is_main_guard_test(st.test) for st in tree.body)
```

Any top-level `if` whose test mentions both `__name__` and `'__main__'` satisfies the
predicate, regardless of what else is at module scope. The docstring's own premise —
"A module the child re-runs is only a PROBLEM when its body is unguarded" — is tested
by a proxy that cannot see the body.

**Evidence** (`guard/g12.py`):

```python
import numpy as np
if __name__ == '__main__':
    pass                        # decorative
BIG = np.zeros((4096, 4096))    # UNGUARDED: 134 MB, re-run in every spawn worker
main()
```
```
_script_has_main_guard(g12.py) -> True   # the pool RUNS; every worker pays the 134 MB
```

This is the ordinary shape of a real driver script (module-level constants, a
`load_config()`, a dataset read, then the guard), and it is precisely the 22.1 GB/worker
failure the warning at 1721–1734 was written for.

Secondary AST edge cases (all measured, `p9` block):

| file | source | verdict | correct | consequence |
|---|---|---|---|---|
| `g6.py` | `if __name__ != '__main__': raise SystemExit` | True | False | false positive (child dies instead, since spawn sets `__mp_main__`) |
| `g7.py` | guard via intermediate names (`NAME == MODE`) | False | True | false negative — serial, harmless |
| `g9.py` | `match __name__: case '__main__':` | False | True | false negative — `ast.Match` is not `ast.If`; serial, harmless |

Canonical `==`, reversed `==`, `in (...)`, nested-in-a-function, bare side effect, and
"words appear in a top-level string" are all classified correctly.

**Fix.** Strengthen the predicate to "every top-level statement other than
imports / simple constant assignments / function+class definitions / `__future__`
sits inside a guard". That is a short `ast` walk and it is the property the warning
actually needs. Add `ast.Match` to the accepted shapes while there.

---

### [P3] Three byte-identical copies of the Chebyshev Vandermonde / derivative recurrences
`_lens_traced.py:3253–3288`; `_lens_imap.py:557–571`; `_math/chebyshev.py:56, 104`

`_math/chebyshev.py` exists *specifically* to de-duplicate these (its header: "v5.2
(ROADMAP v5.1 shared Chebyshev helpers extraction): The three helpers move here, into
a propagator-free math root"), and `lenses.py`, `lenses_maslov.py` and four
`propagators/asymptotic*.py` modules all import from it. `_lens_traced.py` and
`_lens_imap.py` kept private copies.

**Evidence** (`p11` block, 1000 random `u` in [−1, 1]):

```
k= 6/8/12: _lens_traced._cheb_vand_2d        vs _math.chebyshev_vandermonde            bitwise=True
           _lens_traced._cheb_deriv_vand_2d  vs _math.chebyshev_derivative_vandermonde bitwise=True
           _lens_imap._cheb_dvander          vs the same (transposed)                  bitwise=True
```

A fourth near-duplicate is `_math.chebyshev_fit_2d` (a full 2-D total-degree Chebyshev
least-squares fit with a `weight` kwarg and `normalize_xy`), which parallels
`_Cheb2DEvaluator.__init__` — different term set (tensor grid vs total degree) and
different solver, so not a drop-in, but the same primitive built twice.

**Fix.** `from .._math.chebyshev import chebyshev_vandermonde as _cheb_vand_2d,
chebyshev_derivative_vandermonde as _cheb_deriv_vand_2d` (the signatures already
match, `xp` included, and the outputs are bitwise equal, so this is a zero-risk
deletion of 34 lines here and 15 in `_lens_imap`).

---

### [P3] `_reverse_prescription` thickness reversal is off by one under the `len(thicknesses) == len(surfaces)` convention
`_lens_traced.py:6554`

`validate_prescription` (`raytrace/trace.py:301–304`) accepts **both**
`len(thicknesses) == len(surfaces)` (each surface has a forward thickness, the last
being the back focal distance) and `len(surfaces) - 1`. Its own docstring example uses
the first. `list(reversed(thicknesses))` is correct only for the second.

**Evidence** (`p4_reverse.py` §4b/§4c; N-BK7 100/−100 singlet, `thicknesses = [5 mm
glass, 100 mm BFD]`):

```
forward  surface thicknesses: [0.005, 0.1]   (glass gap 5 mm, BFD 100 mm)
reversed surface thicknesses: [0.1, 0.005]   -> reversed GLASS gap = 100 mm
OPL_fwd = 7.583992e-03 m  OPL_bwd = 1.516798e-01 m   d = -1.441e-01 m
back-traced ray lands at x = +6.576e-03 m for a launch height of +4.000e-03 m (err +2.58 mm)
```

Under the `n-1` convention the same test is **exact**: OPL agrees to 5.2e-18 m and the
back-traced ray returns to its launch height to 0.0 m.

**Reachability, stated honestly.** `apply_real_lens` asserts
`len(thicknesses) == len(surfaces) - 1` (`_lens_real.py:4724`), and
`apply_real_lens_traced` always calls it first, so no shipped call path can reach
`_reverse_prescription` with the `n`-thickness form. It is a bare `assert`, so it
disappears under `python -O`. The defect therefore bites (a) a direct caller of
`_reverse_prescription`, (b) a `-O` run, and (c) anyone who relaxes that assertion.
Prescriptions in the `n` form are in active use in the test suite
(`tests/unit/test_analytic_ray_transfer.py` builds several).

Related, same function: the top-level keys `stop_index`, `elements` and `coord_breaks`
are silently dropped (only `aperture_diameter` survives). The in-code comment at
6642–6645 asserts "surfaces_from_prescription uses the per-element semi-diameter plus
the prescription-level aperture_diameter for vignetting; **both carry through to the
reverse automatically**" — false for `elements`, which `surfaces_from_prescription`
reads at line 527–533 and which `_reverse_prescription` does not copy. `stop_index`
would additionally need remapping to `n-1-i`.

**Fix.** Reverse the *pairing*, not the list: build `rev_thick[j] = t[n-2-j]` after
normalising both conventions to "gap after surface i", and carry `stop_index` (remapped),
`elements` (reversed) and any unknown top-level keys through.

---

### [P3] `_geometric_lens_phase`: the "under 10 nm OPL on F/10+" claim is 4x optimistic; no aperture mask; NaN leak from the conic-domain guard
`_lens_traced.py:3309–3313, 3348–3372`

Measured with the `_surface_sag_xy` import patched in-process (see the P1 finding) —
`p3_geomphase.py`, N-BK7 biconvex 100/−100, 8 mm aperture (f = 96.7 mm, **f/12.1**),
N = 256, dx = 40 µm, λ = 587.6 nm, against `angle(apply_real_lens(ones))` over the lit
region:

```
d =   0.001 mm : max|dphi| 3.3947e-04 rad  rms 1.3877e-04  mean(piston) +1.1695e-04
d =   0.100 mm : max|dphi| 3.3951e-02 rad  rms 1.3881e-02  mean(piston) +1.1699e-02
d =   2.000 mm : max|dphi| 6.8054e-01 rad  rms 2.7899e-01  mean(piston) +2.3522e-01
```

At d = 2 mm, piston-removed: **14.1 nm rms / 41.6 nm PV** against the docstring's
"under 10 nm OPL" for F/10+. The error is exactly linear in centre thickness (the
omitted ASM leg), so the claim holds only for thin elements, not for F/10+ elements.

Two further behaviours worth stating:

* **No aperture mask.** `apply_real_lens` vignettes outside `aperture_diameter`
  (measured |E| = 1.7e-02 of peak outside r = 2.1 mm for a 4 mm aperture); the
  geometric phase is finite and unmasked over the whole grid. Benign on the shipped
  assembly (it is multiplied by `amp ~ 0`), but it means the returned array is not
  interchangeable with `angle(apply_real_lens(...))` as the docstring claims.
* **NaN leak.** `surface_sag_general` returns NaN outside the conic domain
  (`norm >= 0.9999`, `lenses.py:229–235`). `_geometric_lens_phase` propagates that
  through `np.angle(np.exp(1j*phase))` to NaN, which the caller subtracts into
  `delta_phase`. A fast/large-conic surface on a grid reaching past its rim would
  return NaN pixels with no diagnostic.

**Verified correct in the same probe** (so the sign work is settled): the formula
`phi = -k0 (n_after - n_before) * sag` matches an independent oracle
`k0 * [sum_i (n_before - n_after)_i sag_i + sum_{i<last} n_after_i t_i]` to
**4.55e-13 rad**, which is CONVENTIONS.md §7 (`OPD > 0 = phase advance`, forward
`exp(+ikz)`). The biconic axis assignment is **correct** (a well-sampled weak biconic
R_x = 5 m / R_y = 2.5 m gives phase curvature ratio COL/ROW = 2.0000 exactly matching
R_x/R_y — my first attempt at this measured 0.535 and was an artefact of `np.unwrap`
failing on an aliased strong lens; retracted). Field-frame `decenter` and
`sag_callable` are honoured (both move the phase by O(pi) rad on a 1 mm decenter /
1 µm cosine ripple).

---

### [P3] Non-converged Newton points below 1 % keep an arbitrary last-iterate OPL with no per-point mask and no warning
`_lens_traced.py:880–889` (pool), `10478–10519` (serial), `10358–10387` (the warning)

The out-of-domain test is radial only (`xe² + ye² > (0.99 R)²`); a point that fails to
converge but stays inside the disc returns a finite OPL from `So.ev(xe, ye)` at
whatever iterate the loop stopped at. The count travels back as a scalar and warns
only above **1 %** of points.

**Evidence** (`p10_newton.py`): a deliberately folded synthetic map
(`x_out = x − 3e4 x³`), 201 query points along the meridian:

```
folded map: 0/201 NaN, n_unconverged = 40
-> 40 points (20 %) return a finite OPL at the last iterate, with no NaN on the value
```

On the well-behaved control (a cubic map with a known analytic inverse and OPL) the
inversion is exact: `max|OPL_newton − OPL_true| = 1.01e-15 m` on a 2.16e-06 m span
(4.7e-10 relative), 0 unconverged, 0 out-of-domain — so the iteration itself is sound.

Also confirmed: the convergence tolerance `tol = 0.01*dx` is an **exit-plane position**
tolerance in metres; the entrance-side accuracy it buys is `tol / |dx_out/dx_in|`,
which is 0.02 wave pixels at |M| = 0.5 and degrades as 1/|M| — worth stating in the
warning text, which quotes `tol` in metres without saying which plane it is on.

**Impact.** At N = 4096, 1 % is 167 000 pixels that can carry an arbitrary OPL with no
diagnostic and no way for the caller to mask them.

**Fix.** Return the active mask alongside the OPL (the pool already returns a tuple)
and NaN the non-converged points, or at minimum expose the count in `_exit_na_out` /
the `_remap_launch_out` diagnostic dicts so a caller can gate on it.

---

### [P3] `_opl_by_backward_trace`: the "~35–40 nm RMS vs Newton" validation number is not reproducible on the one fixture I could build, and the 3.5.x sub-index reference is off-axis when `sub` does not divide `N`
`_lens_traced.py:6572–6588` (the validation claim), `6698–6701` (the axis reference)

**What it computes and the sign work — verified correct.** The OPL is a forward path
length through `reverse(P)` and is therefore positive and, as a function of the *exit*
coordinate, the same function the forward trace produces at the corresponding entrance
point. The exit-vertex transfer `t = −z/N` with `N = +sqrt(1−L²−M²) > 0` in the
reversed frame is the correct signed transfer back to the vertex plane
(REAL_LENS_CHANGES §3), and `p4_reverse.py` §4c confirms end to end that forward and
backward OPL agree to **5.2e-18 m** on a plain-conic singlet once that correction is
applied at both ends, with the back-traced ray returning to its launch height exactly.
Dead rays (TIR / vignetted) become NaN, matching the Newton path's out-of-domain
treatment, and the `coords = [ii/sub, jj/sub]` upsample is exact for any `sub`.

**What does not reproduce** (`p5b.py`; N-BK7 100/−100 singlet, 2 mm thick, 6 mm
aperture, N = 256, dx = 30 µm, λ = 587.6 nm, `ray_subsample = 8`, 1.5 mm Gaussian,
`on_undersample='silent'`):

```
backward OPL finite on 48.0 % of the grid (= exactly the 6 mm aperture disc);
inside the beam (r < 2w) finite on 98.8 %
 r < w : backward-vs-newton exit phase 1.9284 rad rms (180.4 nm), max 371.99 nm
 r < 2w: 1.6891 rad rms (158.0 nm), max 371.99 nm
docstring claims: "~35-40 nm on singlets at N=512"
```

1.93 rad rms sits at the **1.81 rad of a uniformly-distributed wrapped difference**,
and the maximum reaches the wrap boundary — so the two inversions differ by *more than
one wave* over the beam core, not by tens of nanometres.

**Caveat, stated because it matters.** My fixture is exit-undersampled: the grid
Nyquist direction cosine is `λ/(2dx) = 0.0098` against an exit NA of ~0.031, and the
backward path derives its launch directions from `E_analytic`'s phase gradient
(`_sample_local_tilts`), which aliases in exactly that regime. So the measurement is
consistent with the docstring's own attribution of the residual to the gradient
estimate — but it is 5x the claimed number in a regime the claim does not exclude, and
**no test pins the fixture the 35–40 nm figure was measured on**, so the claim is not
checkable as written. Both `_sample_local_tilts` defects above (the `np.roll` wrap and
the half-pixel bias) feed straight into this budget and neither is acknowledged in it.

**Also.** `i_c = N_c // 2` (6698) is the on-axis coarse index only when `sub | N`:
`idx_c = arange(0, N, sub)` puts x = 0 at coarse index `N/(2 sub)`, while `N_c // 2 =
ceil(N/sub)//2`. At N = 250, sub = 8 those are 15.6 and 16, i.e. the "on-axis"
reference is taken 3 fine pixels off axis — a constant OPL piston, so harmless to the
field's shape but not zero, and the `_opl_piston` correction at 11719 is deliberately
skipped on this route. The sibling comment at 6707–6714 fixed exactly this class of
`sub ∤ N` bug for the upsample and left the reference index alone.

---

### [P3] Smaller items (each verified, none worth its own section)

* **`_det_normal_equations:2470–2471`** — `G_out` / `r_out` are allocated as zeros and
  then unconditionally rebound at 2486 from the carry stack. Dead allocations
  (`(M, M)` + `(M, n_rhs)`), harmless, confusing.
* **`_Cheb2DEvaluator` docstring:2932–2935** — claims `ev_value_and_grad` avoids "the
  3x redundant Vandermonde builds that the separate `.ev(dx=1)` and `.ev(dy=1)` calls
  would do". `ev()` (3155–3172) now *delegates* to `ev_value_and_grad`, so that path
  no longer exists and a value-only query pays the full three-quantity cost — measured
  `ev()` = 1.05x of `ev_value_and_grad` (491.7 ms vs 455.6 ms at 2 Mpt). The docstring
  describes a tradeoff that has become a tax.
* **`_Cheb2DEvaluator.from_state`:3129–3137** — validates that `coeffs`, `K1`, `K2` and
  `mi` all have the same *length*, but not that `max(K1) <= order`. `ev_value_and_grad`
  passes `self.order` as `max_order` to the numba kernel, which sizes `Tu` as
  `max_order+1` and indexes `Tu[kx]` with no bounds check — a hand-built or corrupted
  state would read out of bounds silently. Unreachable on shipped paths (the
  constructor guarantees `max(kx) == order`); one extra assert closes it.
* **`_is_main_guard_test`:1542–1558** — see the P2 table above (inverted guard accepted;
  `match`/intermediate-name guards missed).
* **`_compute_carrier` 'auto' branch:4608–4609** — `np.roll` is used for the aliasing
  census (`_gphx`/`_gphy`) and therefore wraps at the grid edge exactly as
  `_sample_local_tilts` does. Here it only feeds the connected-component core mask, so
  the worst case is one spurious ring of "aliased" pixels at the boundary — real but
  benign.
* **`_compute_carrier` 'auto' fit uses RAW METRE monomials** (line 4680–4681) where
  `_fit_residual_eikonal` normalises to `u = (x − cx)/r_fit` (line 6297). At the
  shipped `auto_degree = 2` this is fine (`cond(A) = 1.4e3`), and `_gram_rcond`'s
  diagonal equilibration correctly reports it (see "checked and found correct"), but
  it makes the two sibling fits differ in a way no note explains, and `auto_degree` is
  a function parameter with no other caller in the library.

---

## Performance opportunities

| # | site | measured | estimated gain | how |
|---|---|---|---|---|
| 1 | numba kernel per-sample `np.empty` (1862–1876) | 112.4 ns/pt at order 6 vs 68.9 ns/pt blocked | **1.63x on the kernel** (1.28x at order 10), bit-identical | `p12d.py`, 4 Mpt, 9 interleaved reps, min wall, `max|diff| = 0.0` |
| 2 | pure-xp `ev_value_and_grad` (3232–3250) | 200 float64/query point, unchunked; 1.6 GB at 1 Mpt | bounded memory; the array is 26.9 GB at 4096² today | tracemalloc peak, `p12b_perf.py` |
| 3 | `DETERMINISTIC_TRACED_FIT` on the 28-term fit | fit alone 277.7 ms (det) vs 31.1 ms (BLAS) at 66 049 samples | **8.9x on the fit** | `p12_perf.py`. The in-code note (2268–2275) prices the whole-call cost at +0.6 %, which I did not contradict — but the per-fit factor is 8.9x, not the 7.7x-improvement-over-D14 the note quotes, and it is worth recording because the note's "M = 28 … 34x" figure is the only number a reader has. |
| 4 | `_decentred_fit_score` | 124.8 ms per candidate at a 257² lattice | the C11 arbiter runs **two** of these on top of the real fit | `p12_perf.py`. Each rebuilds the full `(n, M)` design matrix from scratch; the two candidates differ only in `weights` and `order`, so the `order`-6 Vandermonde could be shared. |
| 5 | `_Cheb2DEvaluator.__init__` design-matrix build (3039–3045) | `_Tu_f[:, s:e][K1].T` writes an F-ordered source into a C-ordered destination | strided `np.multiply`; building `(M, n)` then one transpose-copy would be cache-friendly | desk-checked only |
| 6 | `_newton_payload_blob` | 0.63 / 1.58 / 6.77 / 25.2 MB at n_launch = 161 / 256 / 531 / 1024, dumps+digest 1.2 / 15.7 / 98.7 / 43.0 ms | the residency protocol already removes the per-chunk cost; the *parent's* single dumps is 99 ms at design-121 size | measured |

Three fits (Sx, Sy, So) are built from the **same** `xs_in` lattice at the same order:
`A_full` is therefore identical across the three and is currently rebuilt three times
(3007–3046). Sharing it would cut the fit setup by ~3x. The solve itself differs
(different `rhs`), and `_solve_lstsq_thread_safe` supports a 2-D `b` — so the three
right-hand sides can go through **one** `_det_normal_equations` call, which also cuts
the deterministic Gram reduction by 3x. This is the single largest clean win in the fit
path and it is byte-preserving for the Gram (the same blocks, the same tree); the
`A^T b` columns are independent so their bits are unchanged too.

---

## Alternative algorithms / methods

1. **Clenshaw recurrence instead of a materialised T/U table** (Clenshaw 1955; Press
   et al., *Numerical Recipes* §5.8). The kernel currently builds `T_k(u)`, `T_k(v)`,
   `T'_k(u)`, `T'_k(v)` and then contracts against a *sparse* total-degree index list.
   Because the basis is a tensor product, the sum
   `f = sum_kx T_kx(u) * [ sum_ky c_{kx,ky} T_ky(v) ]` factorises: one Clenshaw pass in
   `v` per `kx` (O(order) each) then one in `u`, i.e. `O(order²)` work with **zero**
   scratch arrays and no index gather. Buys: the allocation problem of finding 7
   disappears by construction, not by blocking; costs: the derivative needs the
   companion Clenshaw for `T'` (standard, two extra registers).

2. **Separable tensor-product evaluation on a LATTICE.** `_decentred_fit_score` and
   `_decentred_fit_spectrum` evaluate on `meshgrid(xs_in, xs_in)`, i.e. a *product*
   grid. For that case the evaluation is exactly two matrix products:
   `F = Tu^T C Tv` with `Tu` = `(order+1, n)` and `C` the coefficient matrix —
   `O(n*order² )` flops in two BLAS-3 calls instead of `O(n² * M)` with an `(M, n²)`
   gather. At n = 257, order 6 that is ~28x fewer flops and ~M times less memory.
   (Bits change, so it needs its own flag; but these two functions are *scorers*, whose
   output is a comparison, not a shipped field.)

3. **Householder QR with column pivoting (`dgeqp3`) instead of the Gram + refinement.**
   The disc-masked Vandermonde is genuinely rank-deficient at order >= 10
   (`_gram_rcond` reads exactly 0.0 — measured below). `dgeqp3` gives a numerical rank
   and a minimum-norm solution in one factorisation, is backward stable, and — unlike
   `dgeqrf` over the full `A` — its *column order* is data-determined rather than
   thread-determined, so the D15 determinism argument survives it. Reference: Golub &
   Van Loan §5.4.2; LAPACK Users' Guide §2.4.2.

4. **A Zernike (or Zernike-annular) basis on the disc instead of tensor-Chebyshev on
   the square.** The entire `_FIT_DISC_OUTSIDE_WEIGHT_REL` / `_DECENTRED_FIT_POLY_ORDER`
   / C11-arbiter / C12-predictor apparatus exists because the data lives on a disc and
   the basis is orthogonal on a square. Measured conditioning of the shipped
   arrangement (`p1_cheb.py`, 129² lattice, hard disc r <= 0.5 R):
   `cond(A)` = 3.17e4 (order 6) → 9.02e5 (8) → 2.55e7 (10) → 7.26e8 (12), against
   6.08–8.66 for the *same orders* on the full square. Zernikes are orthonormal on the
   unit disc, so `cond(A)` on a disc-sampled lattice is O(1) at every order, the
   normal equations are then unconditionally safe, and the C13 step-down, the D15
   refinement and the weighted-restriction regulariser all become unnecessary on the
   concentric branch. Cost: the Newton loop needs the basis over the *square* too
   (Zernikes diverge outside the unit disc faster than Chebyshev — but the iterate is
   already clamped to `bound = 0.999 * launch_radius`, i.e. `|u| <= 0.999`, so the
   evaluation domain is the square, not the disc, and a hybrid would be needed).
   References: Born & Wolf §9.2; Mahajan, *Optical Imaging and Aberrations* Part II
   (Zernike annular polynomials, for the vignetted case).

5. **Broyden / secant update for the Newton Jacobian.** `_newton_invert_chunk`
   re-evaluates both `Sx` and `Sy` with full gradients at every iteration (2 combined
   evaluator calls per iteration, each returning 3 quantities = 6 quantities where 2
   are strictly needed once the Jacobian is warm). A "chord" iteration — freeze `J` at
   the initial guess and refresh every 3rd step — would cut evaluator work by ~2x at
   the cost of linear rather than quadratic convergence in the tail; measured here, the
   well-behaved case converges in far fewer than the 12-iteration cap, so the tail is
   where the cost is. Reference: Dennis & Schnabel §8.

---

## Code organization observations

* **Comment-to-code ratio.** The 6 731-line range contains roughly **3 900 lines of
  prose** against ~2 800 of code. Some of it is genuinely load-bearing (the D14/D15
  determinism derivations, the C8 feather measurement). Much of it is decision
  *history* that belongs in `docs/audits/` and is duplicated there:
  `_RD_HALO_AMAX_TOL` alone carries 155 lines of note (292–448), `DECENTRED_FIT_PREDICTOR`
  160 (4035–4155), `_REMAP_RESID_EIKONAL_DEGREE` 160 (4872–5016). The cost is concrete:
  `_geometric_lens_phase`'s AttributeError has survived at least three audit sweeps in a
  function whose docstring is longer than its body.
* **Two competing conventions for "the reference wavefront gradient" inside one file.**
  `_compute_carrier('auto')` and `_fit_residual_eikonal` sample at midpoints;
  `_sample_local_tilts` samples with `np.roll` at pixel centres; the ndarray-carrier
  branch floors an index. Four sites, three conventions, for the same physical quantity
  (`L = (1/k0) dphi/dx`). Two of the three are wrong by half a pixel.
* **Basis normalisation is likewise inconsistent**: `_Cheb2DEvaluator` normalises to
  `[-1,1]`, `_fit_residual_eikonal` to `u = (x−cx)/r_fit`, `_ResidualEikonal._poly` to
  `u = ex/scale`, and `_compute_carrier('auto')` uses raw metres.
* **`_lens_traced.py` is 13 227 lines with `apply_real_lens_traced` at 5 721 lines
  (6731–12452).** The infrastructure half audited here is separable today: the
  Chebyshev evaluator + the deterministic LS stack (2114–3290, ~1 180 lines) has no
  dependency on anything else in the file and is used by `_lens_imap` as well; the
  Newton pool (594–1790, ~1 200 lines) likewise. Two module extractions would halve the
  file with no behaviour change.
* **Layering.** `_lens_traced` imports `apply_real_lens` from `_lens_real` at module
  scope (line 558) and `_lens_imap` as a module (line 33), while `_lens_imap` imports
  back into `_lens_traced` from inside function bodies — a cycle broken by convention
  rather than by structure, and the reason the Chebyshev helpers were copied rather
  than shared.
* **Dead / vestigial.** `_get_array_module` (2097) duplicates `_is_cupy_array` +
  `_ensure_cupy_loaded` (40–53) with a different import strategy; the `cupy` scaffolding
  (`_ensure_cupy_loaded`, `_is_cupy_array`, `_cheb_fit_state`'s `_host`) is unreachable
  and untestable in this environment and the code says so in three places.

---

## Unverified suspicions

* **`_note_pool_deferral` / `_pool_reuse_is_likely` size-band arithmetic.** The
  cross-multiplied "cheapest per point" update (1268–1271) keeps a sample whose
  `seconds/points` ratio is smallest, but `_POOL_DEFERRED_POINTS` is then *also* the
  band anchor, so a run whose cheapest-per-point sample is at the small end of the band
  progressively narrows the band toward that end. I could not construct a case where it
  misbehaves without a real multi-group chain. Confirming it needs a scripted sequence
  of `_note_pool_deferral` calls with a realistic seconds/points distribution.
* **`ORIGIN_AMP_SUPPORT_CHECK = 'error'` (line 495) refuses the call** on a
  `1e-9`-of-power deletion. The note says the intended value is exactly zero and this is
  only an underflow allowance — but `_ORIGIN_AMP_SUPPORT_TOL = 1e-9` is a *power
  fraction of the ray-density exit power*, and the note's own measurement says the zero
  set is empty in the ordinary case. I did not exercise the decentred-origin path (it
  needs `amplitude_model='ray_density'` + `preserve_input_phase='remap'` + a non-zero
  `origin`), so I cannot say whether a legitimate call can trip it.
* **`_TracedExitSupport.from_landings` aperture mask** (5716–5717) tests
  `xs_in[:,None]**2 + xs_in[None,:]**2 <= (0.5*aperture)**2` — the launch lattice is
  axis-centred, so under a decentred `origin` the stop test is taken about the grid
  origin rather than the element axis. `_TracedExitSupport` is not origin-aware and
  nothing in its docstring says so. Confirming needs an origin-decentred ray_density
  call, which I did not build.
* **`_det_refine` memory at the carrier fit's production shape.** `B2 −
  _det_matvec(A, xr)` materialises two `(n, 1)` float64 arrays at `n = 2.8e7` (448 MB
  together) *in addition to* `A` (1.12 GB at M = 5). The note prices the kernel in time
  only. I measured the shape but not the peak RSS at N = 16384.

---

## Checked and found correct

* **Chebyshev recurrences.** `_cheb_vand_2d` and `_cheb_deriv_vand_2d` match
  `numpy.polynomial.chebyshev.chebval` / `chebder` to `3.6e-15` (T) and `6.0e-13` (T')
  at max_k = 14, including at `u = ±1` and `u = 1e-17`.
* **Evaluator derivatives.** `ev_value_and_grad` matches central finite differences to
  2.7e-11 (interior), 3.9e-10 (domain boundary) and 8.2e-11 (outside the domain, at
  1.5 R) relative to the gradient scale.
* **numba vs numpy backend.** Agree to 4.8e-16 … 1.3e-15 relative at orders 6/8/10/12
  over 20 000 points spanning `|u| <= 1.2` — not bitwise, exactly as documented, and
  `fastmath=True` does **not** move them beyond that.
* **Newton iterate clamping.** `bound = launch_radius * 0.999` (10332/10343) and the
  fit domain `xs_in = linspace(-launch_radius, launch_radius, ...)` mean `|u| <= 0.999`
  for every iterate — the Chebyshev extrapolation blow-up (`max|T_k|` = 2.3e7 at
  `u = 3`, order 10) is unreachable from the Newton loop. There is no clamp in `_to_u`
  and none is needed.
* **Newton inversion accuracy.** Exact on a synthetic cubic map with a known analytic
  inverse: `max|OPL_newton − OPL_true| = 1.01e-15 m` over a 2.16e-06 m span
  (4.7e-10 relative), 0 unconverged.
* **Deterministic solve reproducibility.** `deterministic=True` is byte-identical
  across `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS` = 1 / 4 / 24 / default, on both the
  refined branch (hash `2e2e1a9128ed60e7`, 4/4 runs) and the screen-passing branch
  (`faf26c5aa0d79882`, 4/4). `deterministic=False` is **not** (`de54…`/`e743…`/`d9d2…`
  at 1/4/24), as documented. The declared hole is reachable and behaves as declared:
  an order-10 hard-mask disc fit (257² lattice, r ≤ 0.5 R) screens singular, cannot be
  refined, falls to `_solve_lstsq_qr`, warns exactly once, and is then *not*
  reproducible (`16361b…` at 1 and 24 threads, `466b28…` at 4). The equilibrated rcond
  on the same disc at a 129² lattice reads 1.63e-09 / 1.64e-12 / 1.50e-15 / **exactly
  0.0** at orders 6 / 8 / 10 / 12, so the hole widens monotonically with the fit order.
* **`_gram_rcond`'s diagonal equilibration is the right measure for the unscaled
  Cholesky.** I suspected the screen was blind to column scaling (the raw-monomial
  carrier fit reads `cond(G) = 3.2e32` at degree 6 while `_gram_rcond` reads 9.2e-4 and
  passes). Measured directly: the unscaled Cholesky's coefficient error at that
  conditioning is **6.4e-13** relative — the screen is right and the suspicion is
  wrong. This is Demmel's scaled-condition result for SPD matrices, which the docstring
  cites correctly via van der Sluis.
* **`_geometric_lens_phase` sign and piston.** `phi = -k0 (n_after − n_before) * sag`
  plus `sum_{i<last} k0 n_after,i t_i` matches an independent analytic oracle to
  4.55e-13 rad — consistent with CONVENTIONS.md §7 (forward `exp(+ikz)`, `OPD > 0` =
  phase advance).
* **Biconic axis assignment** in `_geometric_lens_phase`: `meshgrid(x, x,
  indexing='xy')` puts x on the column axis and y on the row axis, matching the main
  function's `X = broadcast_to(x[None,:])` / `Y = broadcast_to(y[:,None])` (8390–8391)
  and `np.gradient`'s `(gWy, gWx)` unpacking (4558). A well-sampled R_x = 5 m /
  R_y = 2.5 m biconic gives a phase-curvature ratio of exactly 2.0000.
* **`TiltedCarrier` / `_tilted_carrier_parts`.** `W(x0,y0) == 0` to 0.0e+00,
  `grad W(x0,y0) == (L, M)` to 5.6e-17, analytic gradient matches finite differences to
  1e-10 relative, `|grad W| < 1` everywhere tested, the `R = ±inf` plane limit and the
  `L² + M² >= 1` refusal both fire correctly. The C5 exact-eikonal vs pre-C5
  sphere-plus-ramp difference on design 121's last leg measures **4.41 waves** within
  one beam radius, consistent with (and slightly larger than) the docstring's 2.5.
* **`_compute_carrier('auto')` is unwrap-free and correct.** It fits a *scalar
  potential* by matching its gradient to wrapped nearest-neighbour phase increments —
  no 2-D unwrap anywhere, which is the right design for a converging wavefront.
  Recovery of a known exact sphere (R = 0.2 / 0.05 / −0.05 m, w0 = 0.6 mm,
  dx = 8 µm): fitted `L` matches the true `x/sqrt(r²+R²)` to 2.6e-08 (R = 0.2) and
  1.7e-06 (R = 0.05) against a peak |L| of 4.5e-03 / 1.8e-02, and the residual phase
  after removing `k0 W` is 6.2e-06 / 3.9e-04 rad rms. Sign convention confirmed:
  `W > 0` = phase advance, matching `exp(+ikz)`.
* **Scalar-conjugate branch.** Uses the **exact** sphere in rationalised form
  `sgn * r² / (sqrt(r²+s²) + |s|)`, not the paraxial `r²/2s`, and `s > 0` is diverging
  — verified against the closed form and against the `TiltedCarrier` reduction at
  `L = M = 0`. The `±inf` branch returns `W == 0` with zero gradient rather than an
  all-NaN sentinel, as documented.
* **`_input_beam_amp_radius`.** `w = sqrt(2 <r²>)` recovers a known Gaussian `w0`
  exactly (0.30000 / 0.80000 mm), reproduces the documented origin-referenced inflation
  `sqrt(2 x_c² + w0²)` to 5 digits, and the banded accumulation
  (`Ib @ x2` + `rows @ yg²`) is algebraically the whole-grid second moment.
* **`_carrier_residual_rms` / `_input_tilt_stats`.** On a tilted Gaussian
  (L = 0.03, M = −0.02) both return 0.036056 = `hypot(0.03, 0.02)` exactly;
  `coherence_ratio` = 1.000000; with the exact carrier removed the residual is
  1.5e-18. Sign and normalisation are right for `exp(+ikz)`.
* **`_fit_residual_eikonal`.** Recovers a known `r⁴` residual
  (`a = 2e-6 (r/w0)⁴`, PV 32 µm) to **1.6e-9 m = 0.0012 waves** over the fit disc,
  gradient to 4e-06 against a 0.16 span, no unwrap, per-sample wrapped increments with
  an explicit `|d| <= pi/2` alias rejection, midpoint sample coordinates, normalised
  basis, and degree step-down on `8 samples/term`. The radial freeze is C⁰/C¹ across
  `r_fit` as documented.
* **`_decentred_fit_restriction`.** `w_out = sqrt(1e-8 * n_in/n_out)` and the
  `3 samples/term` step-down `(order+1)(order+2)*3//2 > n_in` are both exact
  (the product is always even, so the `//2` is not a truncation). The weighted
  restriction genuinely improves conditioning: `cond(A)` 2.55e7 → 1.11e5 at order 10 on
  a r <= 0.5 R disc.
* **Pool machinery.** `_get_persistent_worker_pool` takes the lock, forces `spawn`,
  registers `atexit` exactly once (module-level flag under the same lock);
  `close_worker_pool` takes the same lock and resets every piece of promotion state;
  `_POOL_RESIDENT_PAYLOAD_KEY` is read *after* the pool call (so a rebuild resets it)
  and a stale belief can only cost a re-submit, never an answer — the residency
  protocol's correctness argument holds. `NewtonWorkerBackendUnavailable` is caught
  *before* the `RuntimeError` clause it subclasses (Python checks in order), so the
  specific handler wins. `_pool_memory_policy` is applied at the top of
  `_newton_resolve_workers`, not inside the warn branch, so a junk value raises on
  every box. `_newton_resolve_workers`'s re-pricing loop is bounded and monotone.
* **`_copy_prescription`** deep-copies with a container-level fallback that covers
  `surfaces` / `elements` / `thicknesses`, and never raises — as documented.
* **`_kwarg_differs_from_default`** is ndarray-safe in both directions.
* **`_prescription_has_field_frame`** reads `sag_callable`, `decenter` and `tilt` with
  `or (0.0, 0.0)` guards and is a pure predicate with no side effects.
* **`_TracedExitSupport`** radial screens: the inradius bound
  `s(p) <= |p − c| − r_in` (from `|n_f| = 1`, which Qhull guarantees) makes both
  `taper_grid`'s `r_in + plateau` early-out and `retained_band_masks`' `r_in` /
  `hull_rmax + w` screens *strict*, so the claimed bit-identity to the dense form holds
  by construction.
* **`_ResidualEikonal._poly`** power-table hoist: the exponent-0/1 elision is exact
  (multiplying a float64 by 1.0 is exact and preserves inf/nan/−0.0), and the
  left-to-right association is preserved, so the "bit-identical, not identical to
  round-off" claim stands on inspection.
* **`_opl_by_backward_trace` sign and vertex transfer** (see the P3 entry): forward
  OPL through `P` equals backward OPL through `reverse(P)` to **5.2e-18 m** on a
  plain-conic singlet, with the back-traced ray returning to its launch height exactly
  (0.0 m error at h = 0 / 2 / 4 mm).
* **`_reverse_prescription` round trip** `reverse(reverse(P)) == P` holds for radii,
  glasses and thicknesses on both thickness conventions.
* **Chebyshev extrapolation growth** is bounded where it matters: `max|T_k|` = 1.0 /
  20.9 / 161 / 1351 / 1.96e4 at `|u|` = 1.0 / 1.2 / 1.5 / 2.0 / 3.0 for order 6
  (2.3e7 at `u` = 3, order 10) — but the Newton `bound` caps `|u|` at 0.999, so this is
  latent only.

---

## Repro scripts

| script | covers |
|---|---|
| `p1_cheb.py` | disc-masked Vandermonde conditioning, orders 6/8/10/12, hard vs D1-weighted |
| `p1b_deriv.py` | T/T' vs `numpy.polynomial.chebyshev`, evaluator vs finite differences, numba-vs-numpy, extrapolation growth |
| `p2_lstsq.py` | normal-equations vs QR vs a gelsd+`math.fsum`-refined oracle |
| `p2b_extrap.py` | candidate-solve difference in-disc vs over the whole launch square |
| `p2c_det.py` | determinism across `OMP_NUM_THREADS` = 1/4/24/default, all three branches |
| `p3_geomphase.py` | `_geometric_lens_phase` vs `apply_real_lens`, vs an analytic oracle, masking, biconic, field-frame, `form_error` (needs the in-process import patch) |
| `p3b_fastphase.py` | the `fast_analytic_phase` AttributeError, end to end |
| `p4_reverse.py` | `_reverse_prescription` round trip, thicknesses, forward/backward OPL, aspheric sign, key-by-key pass-through |
| `p5_backward.py`, `p5b.py` | `_opl_by_backward_trace` raw OPL and backward-vs-Newton exit phase |
| `p6_carrier.py`, `p6b_gram.py`, `p6c_tilt.py` | `'auto'` carrier recovery, `_gram_rcond` vs unscaled Cholesky, `TiltedCarrier` |
| `p7_tilts.py` | `_sample_local_tilts` roll wrap, half-pixel bias, clipping |
| `p8_misc.py` | `_input_beam_amp_radius`, `_carrier_residual_rms`, `_input_tilt_stats`, ndarray-carrier index rounding, `_fit_residual_eikonal` |
| `p10_newton.py` | `_newton_invert_chunk` against a known analytic inverse and on a folded map |
| `p12_perf.py`, `p12b_perf.py`, `p12c_kernel.py`, `p12d.py` | fit / eval timing and memory, blocked-kernel comparison |
| `guard/g1..g12.py` | the `__main__`-guard AST detector |
