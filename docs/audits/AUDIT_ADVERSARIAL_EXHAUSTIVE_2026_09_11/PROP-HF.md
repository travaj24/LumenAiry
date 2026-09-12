# PROP-HF audit — hf / hfpi / vectorial_hfpi / mhs + Richards–Wolf E_z sign

All numbers were produced on this box (CPython 3.14, numpy 2.4.6, scipy 1.17.1).
Repro scripts in `…/scratchpad/PROP-HF/`: `p1_rw_ez_sign.py`, `p2b_hf_kernel.py`,
`p3_hfpi.py`, `p3b_hfpi_rng.py`, `p4_perf_and_api.py`, `p5_mhs.py`, `p6_misc.py`,
`p7_mhs_hf_tuple.py`, `p8_final.py`, `p9_two_bugs.py`, `p10_return_types.py`.
**Timing caveat:** ~20 other audit processes were saturating the machine; wall-clock
numbers are indicative only. Accuracy, dtype, memory (tracemalloc) and bit-equality
results are not load-sensitive.

## Scope read

| File | Read |
|---|---|
| `lumenairy/propagators/hf.py` | 1–687 (all) |
| `lumenairy/propagators/hfpi.py` | 1–1143 (all) |
| `lumenairy/propagators/vectorial_hfpi.py` | 1–474 (all) |
| `lumenairy/propagators/mhs.py` | 1–708 (all) |
| `lumenairy/propagators/vector_diffraction.py` | 1–476 (all) |
| `lumenairy/propagators/dispatch.py` | 418–520, 700–790, 880–920, 1094–1200, 1240–1350 (how `hf`/`hfpi` are reached and how their returns are wrapped) |
| `lumenairy/io/storage.py` | 1665–1904 (`append_plane` / `list_planes` / `replay_run` — the MHS "replay" semantics) |
| `lumenairy/propagators/asymptotic.py` | 120–240 (where `propagate_hf_chebyshev_quadrature` lives — the `method='direct'` target) |
| `examples/05_mhs_pipeline_with_replay.py`, `lumenairy/ui/richards_wolf_dock.py:29–90` | consumer checks |

---

## Findings

### [P0] `richards_wolf_focus` returns `E_z` with the opposite sign to `E_x`/`E_y` — SETTLED BY MEASUREMENT
`lumenairy/propagators/vector_diffraction.py:347` (with `phi_p` at `:249`, `fft2` at `:396`)

PROP-CORE derived this and asked for a measurement. Three independent checks, all agreeing.

**Oracle (`p1_rw_ez_sign.py`).** Direct numerical Debye–Wolf quadrature in the
RAY-DIRECTION parameterisation — Gauss–Legendre in θ (600 nodes) × trapezoid in φ
(720 nodes), no FFT, no aperture coordinate anywhere, so the φ_ray/φ_pupil ambiguity
cannot leak in:

```
E_j(r) = C ∫₀^θmax ∫₀^2π e_j(θ,φ)·√cosθ·exp(i k ŝ·r)·sinθ dθ dφ,  ŝ=(sθcφ, sθsφ, cθ)
e_x = cosθcos²φ + sin²φ    e_y = cosφ sinφ (cosθ−1)    e_z = −sinθ cosφ
```

x-polarised uniform pupil, NA = 0.5, f = 4 mm, λ = 633 nm, Np = 512, focal pitch 575.45 nm:

| focal point | `E_z/E_x` oracle | `E_z/E_x` `richards_wolf_focus` |
|---|---|---|
| (x=+1.151 µm, y=0) | **−0.314041 i** | **+0.314038 i** |
| (x=+2.877 µm, y=0) | **+0.344100 i** | **−0.343980 i** |
| (x=+5.179 µm, y=0) | **−2.716289 i** | **+2.711241 i** |
| (x=+8.056 µm, y=0) | +0.011781 i | −0.016680 i  *(near-null of E_z; FFT noise)* |
| (x=0, y=+1.151 µm) | ≈0 | ≈0  *(correct symmetry)* |
| diagonal (+5,+5) px | **+0.547004 i** | **−0.537623 i** |

Fixing one global complex constant from `E_x` at the origin, the per-component ratios
`code / (oracle × const)`:

| point | E_x | E_y | E_z |
|---|---|---|---|
| (+1.151, 0) µm | +1.00078 | (0/0) | **−1.00077** |
| (+2.877, 0) µm | +1.00214 | (0/0) | **−1.00179** |
| (+5.179, 0) µm | +1.00788 | (0/0) | **−1.00601** |
| diagonal (+5,+5) px | +1.01523 | +0.99740 | **−0.99782** |

`E_x`/`E_y` agree at **+1.00**; `E_z` agrees at **−1.00**, with the same 0.2–1.5 %
residual on all three components (that residual is the FFT-vs-quadrature
discretisation, identical per component). The single outlier (−1.376) sits at a
near-null where `|E_z|` is 0.25 against 380 on axis.

**Cross-check 1 — closed form.** Novotny & Hecht eq. 3.66 for an x-polarised aplanatic
focus: `E_x = −iA(I₀₀+I₀₂cos2φ)`, `E_z = −2A I₀₁ cosφ`, so `E_z/E_x = −2i I₀₁cosφ/I₀₀`
— **negative-imaginary for x_f > 0**. The oracle reproduces that; the code gives the
positive-imaginary value.

**Cross-check 2 — first principles, no formula.** A ray entering an aplanatic lens at
aperture point (+a, 0) converges to focus along (−sinθ, 0, cosθ), i.e. ray azimuth
φ_ray = π; its polarisation rotates rigidly with the ray, `R_y(−θ)·x̂ = (cosθ, 0, +sinθ)`,
so **E_z > 0 at the +x aperture point**. `Pz = P·(−px·cp·s − py·sp·s)` at `:347`
evaluates to `−sinθ` there. So `phi_p = arctan2(Yp, Xp)` at `:249` is the
APERTURE-POINT azimuth (PROP-CORE's `fft2` measurement pins that), and `e_z` is the
only one of the three components odd under φ → φ+π.

**Impact.** Every call, silently. `|E_z|²` is blind to it, so `debye_wolf_psf`, the
`richards_wolf_dock` GUI (intensity only) and every test are blind — I grepped
`tests/unit/`: no test pins the sign of `E_z` (the S9 registration test checks `Ex`
phase ramps; `test_audit_misc.py:2814` checks an Airy null in `|Ex|²+|Ey|²+|Ez|²`).
Wrong: focal-field handedness / spin angular momentum, `S₃` of the focal field,
optical-force and spin–orbit calculations, and any coherent superposition using all
three components — i.e. the entire reason the function returns `E_z` separately.

**Fix.** One character at `:347`:
```python
Pz = P * (px * cp * s + py * sp * s)        # was  (-px*cp*s - py*sp*s)
```
Test-safe (no test pins it). Worth adding a pin: `Im(E_z/E_x) < 0` for `x_f > 0` on an
x-polarised focus. Also, per PROP-CORE's P3, the docstring should state that `pupil` is
indexed by the physical exit-pupil coordinate, not the ray direction.

---

### [P1] `vectorial_hfpi`'s default path contains no vector physics — it is bit-identically two scalar HFPI runs
`vectorial_hfpi.py:1–22` (module docstring), `:201–295`, `:278–286`

The module docstring advertises "the m-theory dipole obliquity tensor for
vector-correct secondary-source amplitudes" and lists as the cases that *require* it:
"High-NA imaging (NA > ~0.3) where polarization rotates strongly across the focal
plane", "Cascaded diffraction with polarizing elements", "Birefringent elements".

**Measured (`p3_hfpi.py` §4)** — 24×24 source, dx = 2 µm, λ = 633 nm, two 200 µm legs
around a 40 µm aperture, 60 000 paths, same seed:

```
vector Ex == scalar propagate_hfpi run on Ex_in ?   True    max|diff| = 1.872e-23
vector Ey all zero for an x-polarised input ?       True
45-degree linear input, |Ey/Ex − 1| over the WHOLE output grid:
     min = 0.00e+00     max = 1.11e-16          (zero depolarisation, anywhere)
```

There is no `E_z` in the module at all — `VectorPathBundle` carries
`positions, directions, Ex, Ey, opl, alive` — so the partition's "E_z from ∇·E = 0"
question answers itself: **absent**. The default `vector_projection=False` multiplies
both Jones components by the *same scalar* `0.5(cosθ_in+cosθ_out)`, so polarisation is
parallel-transported by the identity.

The opt-in `vector_projection=True` (`:278–286`) does rotate, but:
* it silently discards the longitudinal component it creates — **measured 8.8 % of the
  incident |E|²** dropped at a 0.8 rad cone (`p3b_hfpi_rng.py` §4), the 2-component sum
  falling to 0.788× its unprojected value;
* it then multiplies by the scalar obliquity again (`:272`, `:285–286`), double-counting
  the obliquity the projection already is;
* there is no projection at emission (`init_vector_paths_from_field:143–146`), so a path
  leaving at θ carries a Jones vector with `E·ρ̂ ≠ 0` — not a transverse field even
  before the first aperture.

**Impact.** A user following the module's own use-case list to pick this over the scalar
propagator gets a bit-identical scalar answer at 2× the cost, with no diagnostic. The
per-parameter docstring at `:222–231` IS honest ("the default keeps the historical
scalar-magnitude obliquity weighting only"); the module header is not.

**Fix.** (a) Implement the dyadic / Stratton–Chu re-emission — carry `Ez` on the bundle,
project onto the outgoing transverse plane, drop the doubled obliquity, reconstruct
`E_z` from `∇·E = 0` at accumulation; or (b) rewrite the module docstring to describe
what it does ("two scalar HFPI channels with a shared random stream; no depolarisation,
no E_z — use `richards_wolf_focus` for vector focusing") and delete the "m-theory dipole
obliquity tensor" claim.

---

### [P1] HFPI's Monte-Carlo source-plane normalisation is low by the source pixel count `Ny·Nx`
`hfpi.py:222–225` and `:814–817`; the claim being contradicted is at `:563–600`

The estimator draws the source pixel uniformly over `Ny·Nx` pixels and the direction
uniformly over the cone, so the unbiased estimate of `∫dS∫dΩ f` is
`(Area·Ω/n_paths)·Σf` with `Area = Ny·Nx·dx²`. The code uses
`solid_angle = 2π(1−cosθmax)/n_paths` times **one pixel's** `dx²`.

**Measured (`p3_hfpi.py` §2)** — single unit-amplitude source pixel, cone 0.20 rad,
200 000 paths, against the exact `∫ E cosθ dx²/(iλ) dΩ`:

| source grid | `Σ weights` | exact | ratio | `1/N_pix` |
|---|---|---|---|---|
| 1×1 | −7.835548e-07 i | −7.835520e-07 i | 1.000004 | 1.000000 |
| 4×4 | −4.869015e-08 i | −7.835520e-07 i | **0.062140** | 0.062500 |
| 16×16 | −3.091155e-09 i | −7.835520e-07 i | **0.003945** | 0.003906 |
| 64×64 | −2.035862e-10 i | −7.835520e-07 i | **0.000260** | 0.000244 |

The ratio tracks `1/N_pix` to MC noise. At a 64×64 source the amplitude is low by **4096×**.

**Why this matters even though the factor is global.** The `:563–600` warning exists
precisely to tell the user which normalisation IS and IS NOT applied. It lists "the
Monte Carlo solid-angle weight `2π(1−cosθmax)/N_paths`" and "the source pixel area
`dx²`" under **does apply**, and attributes the non-quantitative amplitude solely to the
missing `1/r` and the missing binning Jacobian. That accounting is wrong: the applied
normalisation is itself off by the pixel count, so a reader who corrects for the listed
omissions still lands 4096× low. Same bug in `init_paths_stratified` (`:814`).

**Fix.** `solid_angle = 2π(1−cos_max) * (Ny*Nx) / n_paths` at `:222` and `:814`
(equivalently keep the solid angle and use the total source area `Ny*Nx*dx*dx`), and
correct the docstring's list.

---

### [P1] `rng=None` — the default on every HFPI entry point — is fully deterministic and identical to `rng=0`
`hfpi.py:191, 306, 727`; `vectorial_hfpi.py:108, 234`; `_spawn_rng` at `hfpi.py:100–104`

`_spawn_rng`'s `None` branch is documented as: *"Caller did not pin a seed; let each
aperture pull from system entropy (RandomState(None) constructs a fresh generator from a
non-deterministic seed)."* Every consumer then writes
`RandomState(rng=rng if rng is not None else 0)` — so `None` becomes the fixed seed 0 and
`RandomState(None)` is never reached.

**Measured (`p3b_hfpi_rng.py` §1)**, 200 000 paths, narrow cone so paths actually land
(`Σ|E|² = 1.39e-23`, non-zero):

```
rng=None twice, bit-identical?      True
rng=None equals rng=0 ?             True
rng=7 differs from rng=None ?       True    (seed threading itself works)
```

**Impact.** HFPI is a `1/√N_paths` Monte-Carlo estimator sold on that convergence; the
canonical way to see its error is to re-run with a new seed. On the default path two runs
are byte-identical, so the error estimate is identically zero. Any ensemble / tolerancing
/ seed-averaging loop that does not explicitly vary `rng` draws the same sample N times
and reports a spuriously tight spread. It also defeats the v4.11.2 `_spawn_rng` work on
exactly the path most callers take.

**Fix.** `RandomState(rng=rng)` at the five sites (`RandomState` already treats `None` as
fresh entropy), or make `_spawn_rng(None, i)` return a per-stream entropy seed. If
determinism-by-default is actually intended, say so instead of documenting the opposite.

---

### [P2] `cone_half_angle` — the remedy HFPI's own under-sampling guard recommends — is unreachable through the entry points that emit it
`hfpi.py:609–626`, `vectorial_hfpi.py:393–410`

The v5.31 guard (`hfpi.py:451–471`) tells the user:
> *"Two levers: raise n_paths, and — usually far more effective — narrow
> `cone_half_angle` from its ~90-degree default (a full forward hemisphere) toward the
> angle the output grid actually subtends…"*

**Measured (`p9_two_bugs.py` §B):**
```
propagate_hfpi_freespace_aperture:        cone_half_angle in signature: False
propagate_vector_hfpi_freespace_aperture: cone_half_angle in signature: False
passing it anyway            -> TypeError: propagate_hfpi_freespace_aperture() got an
                                unexpected keyword argument 'cone_half_angle'
via propagate(method='hfpi') -> same TypeError
```
`propagate_hfpi_freespace_aperture` calls `init_paths_from_field` (`:646–652`) and
`apply_aperture_diffraction` (`:655–662`) without it, so both take the ~90° default, and
it accepts no `**kwargs`. `propagate_hfpi` and `propagate(method='hfpi')` without a
prescription both route here, so the *only* free-space HFPI entry point in the library is
the one that cannot take the lever it recommends. Only
`propagate_hfpi_through_prescription` exposes it (`:849`).

**Fix.** Add `cone_half_angle: float = np.pi/2 - 1e-6` to both free-space entry points and
thread it to the two call sites. Three lines.

---

### [P2] The vectorial accumulator silently bypasses the v5.31 under-sampling guard
`vectorial_hfpi.py:298–390`, `:457–464`

v4.13.1 re-implemented the vector accumulator inline ("bit-identical to the twice-routed
version") instead of calling `hfpi.accumulate_to_grid` twice. The guard added to the
scalar accumulator in v5.31 (`hfpi.py:442–475`) therefore never runs on the vector path,
and `propagate_vector_hfpi_freespace_aperture` has no `on_undersampled` kwarg.

**Measured (`p3_hfpi.py` §5)**, identical geometry through both paths:
```
scalar end-to-end warned:  True
vector end-to-end warned:  False
vector entry point accepts on_undersampled?  False
```
The guard's own text says that below one landed path per output pixel "the returned array
is the Monte-Carlo sampling envelope plus shot noise, not a propagated field", and that
two seeds of the same physics agreed to a shape fidelity of 0.005. The vector path can
return exactly that, in silence. (The JAX branch at `:334–351` routes back through
`hfpi.accumulate_to_grid`, but that one is JAX-gated off too, so the exemption is total.)

**Fix.** Hoist the check into a `_check_landed(inside, Ny, Nx, policy)` helper called from
both accumulators; add `on_undersampled` to the vector entry point.

---

### [P2] `init_paths_stratified`: `n_paths` is not a cap — an explicit stratification can allocate 10 000× the requested paths
`hfpi.py:743–748`

```python
n_total = n_iy * n_ix * n_th * n_ph
n_per   = max(1, n_paths // n_total)
n_paths_actual = n_per * n_total          # >= n_total, regardless of n_paths
```

**Measured (`p6_misc.py` §2)**, 16×16 source:

| call | requested | allocated | factor | bundle memory |
|---|---|---|---|---|
| defaults, `n_paths=20000` | 20 000 | 20 736 | 1.0× | 1.5 MB |
| `n_paths=100, n_strata_xy=(16,16), n_strata_dir=(16,16)` | 100 | 65 536 | **655×** | 4.8 MB |
| `n_paths=100, n_strata_xy=(32,32), n_strata_dir=(32,32)` | 100 | **1 048 576** | **10 486×** | 76.6 MB |

The default path is fine (the 4th-root rule keeps `n_total ≈ n_paths`), but the parameters
are public and documented and the failure mode is a silent three-orders-of-magnitude blow
up: a `(64,64)/(64,64)` stratification — perfectly reasonable-looking for a 64×64 source —
allocates 16.8 M paths (~1.2 GB) from an `n_paths=100` call. The docstring at `:744–746`
actually *describes* the correct behaviour ("If user requested fewer than n_total, sample
only n_paths strata uniformly without replacement") — the code never implements it.

**Fix.** Implement that sentence, or raise naming the resulting count.

---

### [P2] `hf`'s free-space branch is the only output-grid-capable method that changes its RETURN TYPE — including when the request is a no-op
`hf.py:119–188`; `dispatch.py:1307–1314`, `:718–721`

`propagate_huygens_fresnel_freespace` returns a bare `ndarray` with no output kwargs and a
`(E_out, dx_out)` **tuple** with either of them (documented at `:119–128`), and the
dispatcher forwards them whenever `_resolve_dispatcher_output_grid` resolves anything.

**Measured (`p10_return_types.py`)**, `propagate(..., z=1e-3, method=M, return_result=False)`,
32×32 @ 20 µm, regrid to 16×16 @ 40 µm:

| method | no grid kwargs | with `output_grid` |
|---|---|---|
| asm | ndarray (32,32) | ndarray (16,16) |
| gbd | ndarray (32,32) | ndarray (16,16) |
| **hf** | ndarray (32,32) | **TUPLE len=2 → (ndarray(16,16), float)** |
| hfpi | ndarray (32,32) | ndarray (16,16) |

**Measured (`p7_mhs_hf_tuple.py` §1)** it also flips for a request that is a strict no-op —
`output_grid={'N':32,'dx':20 µm}` on a 32×32 @ 20 µm input returns
`(ndarray(32,32), 2e-05)`.

`dispatch.py:1165–1181`'s W9-4 error message actively steers callers here — *"name
method='gbd' (or 'hf' / 'hfpi') to keep the request"* — so this is the advertised migration
target. With the v5.30 default `return_result=True` the wrapper absorbs the tuple via
`_coerce_field`; the damage is confined to `return_result=False`, which is exactly what
`mhs.prescription_subdomain` uses (`:609`, `:678`).

**Fix.** Unpack in the `hf` free-space dispatcher branch (`E, _ = …; return E`) — `propagate`
already knows the requested pitch — or make the kernel's tuple return unconditional.

---

### [P2] `hf`'s "same grid" short-circuit compares pitches with `np.isclose`'s default `atol = 1e-8` **metres**
`hf.py:167–169`

```python
if N_in == int(N_out) and np.isclose(float(dx), float(target_dx), rtol=1e-12):
    return E_native, target_dx
```
`np.isclose(a, b, rtol=1e-12)` still carries `atol=1e-8` — 10 nm in this library's units,
comparable to or larger than the pitches metasurface/FSO work uses. With `output_shape`
omitted (the common call) `N_out` is forced to `N_in`, so this is the *only* gate.

**Measured (`p9_two_bugs.py` §A):**
```
np.isclose(1e-6, 1.005e-6, rtol=1e-12) = True    (0.5 % pitch change)
np.isclose(1e-7, 1.1e-7,   rtol=1e-12) = True    (10 % pitch change at 100 nm)
np.isclose(1e-7, 2.0e-7,   rtol=1e-12) = False

dx=1.000e-06 -> output_dx=1.005e-06: returned dx=1.0050e-06, field IDENTICAL to the
                                     un-resampled native field   <-- SILENT NO-OP
dx=1.000e-07 -> output_dx=1.100e-07: returned dx=1.1000e-07, field IDENTICAL
                                     <-- SILENT NO-OP
dx=1.000e-07 -> output_dx=2.000e-07: resampled (correct)
```
The tuple is labelled with `target_dx` while the data is still at `dx` — the exact "wrong
sampling metadata" failure class `dispatch.py:701–717` (W9-4) documents and raises for on
`maslov`, recreated here. The comment at `:159–164` claims this mirrors `mhs.py:583–587`;
that site uses a strict `< 1e-15` absolute test, and `asm_subdomain` (`mhs.py:382`) uses a
correct relative one.

**Fix.** `abs(dx - target_dx) <= 1e-12 * dx`. One line.

---

### [P2] The HF OPL quadrature costs ~15 full grids of transient **per output pixel** and is O(N⁴); the parameter that would have fixed it was deprecated as a no-op
`hf.py:345–393`, `:200`, `:301–307`

Per output pixel the loop calls `opl_fn` **17 times** over the full input grid (1 for Φ plus
16 for the four cross-Hessian stencils at `:352–375`), each allocating several `N²` float64
temporaries, then builds a full complex128 `kernel` and a full `integrand`.

**Measured (`p4_perf_and_api.py` §2)**, tracemalloc, `N_in = 256`:

| | peak | × one float64 N² grid (0.524 MB) |
|---|---|---|
| `apply_van_vleck=False` | 4.725 MB | **9.01×** |
| `apply_van_vleck=True` | 7.871 MB | **15.01×** |

**Measured (`p8_final.py` §b)**, warm, median of 5 (loaded box):

| `N_in` | ms / output pixel | extrapolated full `N×N` output |
|---|---|---|
| 128 | 2.15 | 0.6 min |
| 256 | 109.1 | **119 min** |
| 512 | 429.0 | **31 h** |

So a full-grid HF output is practical only to `N ≈ 128`. `chunk_output` (`:200`,
`:301–307`) was deprecated in v5.17 with the note that it "never had any effect —
evaluation has always been strictly per output pixel". True — but that identifies the fix
and then removes it. `opl_fn` is already fully vectorised over the input grid; making
`s2x`/`s2y` arrays of shape `(n_chunk,1,1)` broadcasts for free and amortises the 17
evaluations over a block of output pixels: same flops, ~`n_chunk`× fewer Python-level
dispatches and temporaries.

---

### [P2] `apply_aperture_diffraction`'s default `wavelength=0.0` silently drops the `1/(iλ)` Kirchhoff prefactor
`hfpi.py:295, 354`; `vectorial_hfpi.py:207, 270`

```python
inv_i_lambda = (1.0 / (1j * wavelength)) if wavelength > 0 else 1.0
```
**Measured (`p6_misc.py` §4)** — the exported `apply_aperture_diffraction` called without
`wavelength` versus with `wavelength=633e-9`:
```
|w| ratio = 6.3300e-07  (exactly lambda);   phase differs by +1.5708 rad (pi/2)
```
i.e. amplitude wrong by λ ≈ 10⁻⁶ **and** the 90° Kirchhoff phase missing — precisely the
failure the v4.11.2 note at `:344–352` says it fixed. Both functions are in `__all__`, are
reachable as `lumenairy.apply_aperture_diffraction`, and are the documented way to build a
custom cascade by hand. A physically meaningless default (λ = 0) should not silently mean
"skip the physics".

**Fix.** Make `wavelength` keyword-required (no default), or raise on non-positive.

---

### [P2] HFPI's multi-backend claim is false for CuPy on the default path — desk-check (no CuPy in this env)
`hfpi.py:22–27` (module docstring), `:446–448`, `:1086–1093`

Docstring: *"The full pipeline is written against `array_namespace`, accepting NumPy /
CuPy / JAX source fields and returning the same backend."*

* `accumulate_to_grid`'s guard does `np.asarray(inside)` / `np.count_nonzero(np.asarray(inside))`
  at `:447–448`, gated only on `not is_jax_array(...)`. `np.asarray` on a CuPy array raises
  `TypeError: Implicit conversion to a NumPy array is not allowed` in modern CuPy — so the
  **default** `on_undersampled='warn'` crashes on a CuPy bundle (`'silent'` skips it).
* `_hfpi_segment_trace` does `np.asarray(paths.positions …)` at `:1086–1093` with the same
  guard, so every prescription path is NumPy-only; for JAX it round-trips to host via
  `to_numpy` and back — correct but untraceable, so no `jit`/`vmap`/`grad` through the
  prescription walk.

**Fix.** Route both through the already-imported `to_numpy(...)` unconditionally, and state
the JAX host round-trip in the module docstring.

---

### [P3] `finite_diff_step`'s documented accuracy is specific to a quadratic (Fresnel) OPL and is ~9× off on a real one
`hf.py:241–267`

The docstring pins the Van Vleck amplitude error at "**−2.53e-8 at the 1e-6 default**",
measured "on an exact-quadratic (Fresnel) OPL oracle" — where the 4th-order truncation term
vanishes identically, so only round-off survives.

**Measured (`p2b_hf_kernel.py` §C)** on the exact spherical OPL (`Φ = √(u²+v²+z²)/λ`,
z = 50 mm, λ = 1 µm, u = 2 mm, v = 1 mm), against the closed form `√|det| = cosθ/(λr)`:

| h [m] | rel. err of `√|det|` |
|---|---|
| 1e-9 | +2.2605e-01 |
| 1e-8 | +1.5337e-03 |
| 1e-7 | +2.5030e-05 |
| **1e-6 (default)** | **+2.3578e-07** |
| **1e-5 (optimum)** | **−3.7685e-08** |
| 1e-4 | −3.9720e-06 |
| 1e-3 | −3.9690e-04 |

Still ~4 decades below the quadrature's own ~1e-3 discretisation floor, so this is a
documentation defect, not a numerical one — but the numbers should say which OPL they were
measured on, and note that the optimum moves a decade for a non-quadratic Φ.

---

### [P3] `_spawn_rng` with a `Generator` mutates the parent, so a stream index is not a stable key
`hfpi.py:109–118`

`rng.spawn(stream_index + 1)[-1]` advances the parent's `n_children_spawned` on every call.
**Measured (`p3_hfpi.py` §3):**
```
fresh parent, same stream index -> same draws?            True
stream 1 drawn AFTER stream 0 == stream 1 drawn alone?    False
```
It also spawns `i+1` children and discards `i`. `SeedSequence(entropy=[parent_entropy, i])`
— as the `int` branch at `:105–108` already does — would make the mapping a pure function
of `(parent, i)`.

---

### [P3] `MhsPipeline._validate` ignores `HuygensSurface.centre`
`mhs.py:156–170`

The chain check compares `z`, `Ny`, `Nx`, `dx` — not `centre`. **Measured (`p5_mhs.py` §7):**
two surfaces at identical z/N/dx whose centres differ by 1 mm are accepted as the same
plane, so a pipeline can carry a 1 mm transverse coordinate jump across a seam with no
diagnostic. `HuygensSurface.grid()` (`:77–88`) and `aperture_subdomain` (`:437–438`) both
honour `centre`, so the field really is on a different coordinate system either side. One
clause.

---

### [P3] "Replay" carries no run identity: two runs into one store silently concatenate
`mhs.py:234–247` (`_persist`), `io/storage.py:1817–1904` (`replay_run`)

For the record, since the partition asked what MHS caches: **nothing**. `MhsPipeline`
memoises nothing — **measured (`p5_mhs.py` §3)** 4 propagator invocations over 2 identical
`run()` calls — and re-propagates from scratch every time. "Replay" is purely `io.storage`:
`run(store=…)` appends each plane via `append_plane`, `replay_run` re-reads them. The only
key is `label_prefix` + insertion index; there is no prescription / wavelength / grid hash,
so a store written from a *different* system replays without complaint.

**Measured (`p5_mhs.py` §4)**, two runs of the same pipeline into the same store+prefix:
```
after 1 run:  3 planes  ['run_000_src', 'run_001_mid', 'run_002_out']
after 2 runs: 6 planes  ['run_000_src','run_001_mid','run_002_out',
                         'run_000_src','run_001_mid','run_002_out']
duplicate labels? True
replay .field == last run's exit field? True
```
`.field` is right (index order preserves the last write) but `.history` is a silently
doubled, unlabelled interleave of two runs. A per-run UUID in the label, or an
overwrite-vs-append flag on `run()`, closes it.

---

### [P3] `from_prescription` and `prescription_subdomain` disagree on the default method
`mhs.py:295` (`method='gbd'`) vs `mhs.py:504` (`method='maslov'`) — measured. Two
constructors for the same subdomain, two different physics models by default; `maslov` is
the one that needs the `:558–582` square-grid guards and the post-hoc resample.

---

### [P3] The documented canonical entry points are missing from their modules' `__all__`
`hf.py:683–687`, `hfpi.py:1134–1143`

`propagate_huygens_fresnel` ("**This is the recommended entry point for new code**",
`hf.py:76–79`) and `propagate_hfpi` ("Canonical-order HFPI three-leg propagation",
`hfpi.py:552`) are both absent from their module `__all__`. **Measured (`p6_misc.py` §1)**
both are still reachable (`lumenairy.propagate_huygens_fresnel`,
`lumenairy.propagators.propagate_huygens_fresnel`) because the package `__init__` imports
them by name — so this is an `__all__` integrity gap, not a breakage; but
`from lumenairy.propagators.hf import *` does not give you the entry point the module tells
you to use.

---

### [P3] The JAX branch of the HF quadrature allocates one full output array per output pixel and cannot be jitted at all
`hf.py:345–393` (`:346–347`, `:390–391`)

`out = out.at[iy, ix].set(out_value)` inside `for k in range(n_out)` allocates a fresh
`(Ny_out, Nx_out)` array per output pixel — 4096 full-array copies for a 64×64 output. And
`:346–347` does `float(flat_x[k])`, forcing concretisation, so the function cannot be traced
under `jit`/`vmap` even in principle. Accumulating into a Python list plus one
`jnp.asarray(...).reshape(...)` fixes the allocation; array-valued output coordinates (the
chunking above) fix the traceability.

---

### [P3] `complex64` buys nothing in the HF quadrature
`hf.py:385–386`

`kernel = xp.exp(2j*pi*phi).astype(out_dtype)` builds a full complex128 array and then casts;
`density` is float64; `integrand = E_in * density * kernel` promotes `complex64 × float64`
back to complex128. **Measured (`p8_final.py` §c)** at N=128:
```
E_in complex128 -> out complex128, integrand complex128, 1.78 ms/px
E_in complex64  -> out complex64,  integrand complex128, 1.76 ms/px
```
Full complex128 memory and time, precision lost only at the final store.

---

### [P3] `_resolve_output_shape` is duplicated verbatim in `hf.py` and `hfpi.py`
`hf.py:33–62`, `hfpi.py:53–82`

**Measured (`p6_misc.py` §5):** the two bodies are identical apart from the method name in
the warning text — and `vectorial_hfpi.py:37` already imports *hfpi's* private copy (along
with `_spawn_rng` and `_complex_output_dtype`). One shared helper taking `fn_name` (both
already have it as a parameter) removes 30 duplicated lines.

---

## Performance opportunities

1. **Chunk the HF output loop** (P2 above). 17 full-grid `opl_fn` evaluations per output
   pixel, 15 float64 grids of transient peak, 109 ms/px at `N_in=256`. Broadcasting
   `s2x/s2y` to `(n_chunk,1,1)` costs nothing in flops and removes ~`n_chunk`× of Python
   dispatch and temporaries. The Van Vleck stencil for a z-invariant Φ is also constant
   along an output row — a second free factor of ~2.
2. **Do not build the kernel in complex128 for a complex64 caller** (P3 above): measured
   zero speed benefit today, and the cast is the only place the dtype is honoured.
3. **`propagate_huygens_fresnel_freespace` already delegates to `rayleigh_sommerfeld_propagate`**
   — i.e. the FFT-convolution route — so the O(N⁴) quadrature only earns its keep when Φ is
   genuinely *not* shift-invariant (through a prescription). For free space the FFT route is
   `~5e4×` faster at N=256 (0.1 s vs 119 min) at the same ~1e-3 accuracy (measured below).
   The `with_opl_callable` docstring should say that out loud.
4. **`aperture_subdomain._prop` (`mhs.py:431–444`) rebuilds the coordinate meshgrid and the
   mask on every call.** In a many-plane / many-wavelength pipeline that is `Ny·Nx` float64
   ×3 per call for a mask that depends only on the surface. Close over the mask at builder
   time.
5. **`accumulate_vector_to_grid` already shares the index computation** — good; the cost is
   that it also lost the guard (P2 above). Share the guard, not just the indices.
6. **Quasi-Monte-Carlo instead of jittered stratification.** `init_paths_stratified` does
   correct 4-D jittered stratification, which is `O(N^{-1/2})` with a better constant. A
   4-D Sobol or Halton sequence over the same `(pixel_x, pixel_y, cosθ, φ)` cube is
   `O(N^{-1})` for smooth integrands and is a ~10-line drop-in
   (`scipy.stats.qmc.Sobol(d=4).random(n)`), with the added benefit that `n_paths` becomes
   an exact cap (fixing the P2 blow-up above for free).

## Alternative algorithms / methods

1. **Pixel-integrated RS kernel (Shen & Wang, *Appl. Opt.* 45, 1102 (2006))** — already cited
   at `rs.py:148–150` and the cause of PROP-CORE's 2.8e-2 RS-vs-ASM plateau. It is also the
   floor here: the HF quadrature's error against its own continuum limit is
   **2.86e-2 → 3.79e-3 → 2.88e-3 → 1.08e-3** at N = 256/512/1024/2048 (z = 100 µm), i.e.
   roughly first-order in dx, which is the signature of point-sampling a kernel with a
   `1/r`-scale feature. Integrating the kernel over each pixel analytically restores the
   `O(dx²)` rate and would let both modules' convergence claims become true.
2. **NUFFT (type 3) for arbitrary output points.** `hf.propagate_huygens_fresnel_with_opl_callable`
   supports arbitrary separable output grids, which is its one genuine advantage over
   RS/ASM — but it pays `O(N²·M)` for it. For a *shift-invariant* Φ and `M` arbitrary output
   points, a type-3 NUFFT costs `O((N² + M) log N)`: at `N=256, M=1000` that is ~4 orders of
   magnitude. (`finufft` is not installed here, so this is a recommendation, not a
   measurement.) The Bluestein machinery already in `_bluestein.py` covers the *uniform*
   arbitrary-pitch case today and would cover most real uses.
3. **Band-limited ASM for planar-to-planar.** Measured head-to-head on a Gaussian
   (`p2b` §B, w₀ = 8 µm, N = 512, dx = 250 nm), L2 against the analytic Gaussian on a 25-point
   cut:

   | z | HF (OPL callable) | ASM | HF vs ASM |
   |---|---|---|---|
   | 0.5 z_R | 7.04e-4 | 1.05e-4 | 6.35e-4 |
   | 1.0 z_R | 3.95e-4 | 1.56e-4 | 3.17e-4 |
   | 3.0 z_R | 2.01e-4 | 2.18e-4 | 2.44e-4 |

   ASM is equal or better *and* delivers the whole 512² plane in the time HF takes for 25
   points. HF's justification is a non-shift-invariant Φ, nothing else.
4. **For `vectorial_hfpi`: Stratton–Chu / dyadic-Green surface integration**, or simply
   routing high-NA vector focusing to `vector_diffraction.richards_wolf_focus` (after the P0
   fix). A path-integral vector propagator that carries `(Ex, Ey, Ez)` per path and projects
   onto the outgoing transverse plane at each re-emission is the minimum honest version of
   what the module docstring already claims.
5. **Importance sampling toward the output aperture** for HFPI. The v5.31 guard already
   diagnoses that a ~90° cone is the dominant waste; cosine-weighted or cone-aimed sampling
   with the corresponding weight correction converts the guard from a warning into a fix.
   (And `cone_half_angle` must be reachable first — P2 above.)

## Code organization observations

* **`mhs.py` contains no Huygens-surface integral.** The module docstring opens with
  *"Hybrid wave / ray propagator following the framework introduced in the IEEE 2023 paper on
  Multiple Huygens Surface ray tracing… At each Huygens surface, the ray bundle is converted
  to a complex field via a Huygens-surface integral"* — no such conversion exists anywhere in
  the 708 lines. What the module is, and says further down at `:33–37`, is a **composition
  framework**: a `(z, Ny, Nx, dx, centre)` record, a `(propagator, in_surface, out_surface)`
  record, and a loop. `HuygensSurface` is flat-only (no tilt, no curvature), so the paper's
  tilted-surface capability is absent too. The first paragraph should be demoted to a
  "Background" note.
* **Four near-identical forward-cone samplers**: `hfpi.py:203–211`, `:326–332`, `:802–808`;
  `vectorial_hfpi.py:120–128`, `:250–258`. Same five lines, same `cos_max` convention,
  independently maintained.
* **Two near-identical scatter-add accumulators**: `hfpi.py:477–511` and
  `vectorial_hfpi.py:356–390`, the second forked for index sharing — and, as a direct
  consequence, missing the v5.31 guard (P2 above). The index computation could be a shared
  `_bin_paths(...) -> (flat_idx, inside)` helper used by both.
* **Guard text longer than the code it guards.** `hfpi.py:442–475`: a 3-line check wrapped in
  a 21-line f-string. `propagate_hfpi`'s `.. warning::` block (`:563–600`) is 38 lines — longer
  than any function body in the module — and, per the P1 finding above, its enumerated list is
  wrong. Long narrative comments that encode measurements are valuable; ones that encode a
  *normalisation contract* need a test, not prose.
* **`vectorial_hfpi.py` imports three private names from `hfpi`** (`_complex_output_dtype`,
  `_resolve_output_shape`, `_spawn_rng`) while `hf.py` keeps its own verbatim copy of one of
  them. Pick one: a `_hf_common.py`, or make them public.
* **`hf.py` has two unrelated halves.** Lines 33–405 are a self-contained numerical quadrature;
  412–680 are a prescription front-end that does LG decomposition, waist estimation and
  delegation into `asymptotic`. They share no code and one is a thin dispatcher over another
  module. The `method='direct'` branch (`:646–676`) doesn't even use the quadrature in this
  file — it calls `asymptotic_canonical_fit.propagate_hf_chebyshev_quadrature`.
* **`MhsPipeline.run`'s `store` import at `:234–235`** binds `append_plane` as a local of `run`
  and relies on the closure in `_persist` never reading it when `store is None`. It works, but
  a plain `append_plane = None` guard would make the intent legible.

## Unverified suspicions

* **CuPy.** Everything in the "multi-backend claim is false" finding is a desk-check —
  `cupy` is not installed here. The two `np.asarray(<device array>)` sites are unambiguous in
  the source, but the exact exception type depends on the CuPy version.
* **JAX tracing through the prescription walk.** `_hfpi_segment_trace`'s `to_numpy` round-trip
  makes `jit`/`vmap` impossible in principle; I did not build a JAX prescription case to
  confirm the exact failure mode.
* **Cascaded-aperture phase bookkeeping.** `apply_aperture_diffraction` resets `opl` to zero
  and relies on the accumulated phase living in `weights`; the free-space composition checks
  out by inspection and the three-leg case runs, but I did not build an oracle for a
  *cascaded* aperture chain (two or more diffractors) against an independent Fresnel–Kirchhoff
  quadrature. Given the P1 normalisation error, that is where I would look next.
* **`propagate_hf_chebyshev_quadrature`** (the `method='direct'` target, in
  `asymptotic_canonical_fit.py`) is outside this partition; I only confirmed where it lives and
  that `hf.py:671` calls it with `apply_van_vleck=True`. Its Maslov prefactor is claimed
  (`hf.py:395–404`) to already include the `-1j`; I did not verify that independently.
* **MHS `prescription_subdomain(method='hf'/'hfpi'/'gbd')` end to end.** The dispatcher's
  prescription branch for `hf` returns a bare array (it routes to
  `propagate_huygens_fresnel_through_prescription`, not the free-space function), so the tuple
  problem above should *not* reach the MHS pipeline — but my end-to-end run of that chain was
  killed by the machine load before producing output, so I could not confirm it.

## Checked and found correct

* **The Van Vleck density identity.** For Φ = r/λ, I derived
  `√|det ∂²Φ/∂s₁∂s₂| = cosθ/(λr)` analytically (A = (1/(λr))(u²/r²−1), B = C = uv/(λr³),
  det = z²/(λ²r⁴)) and confirmed it against the code's own finite-difference stencil:
  **ratio 0.999988914** at h=1e-6, u=7 µm, v=−3 µm, z=300 µm. The determinant
  `pxx·pyy − pxy·pyx` at `hf.py:376` is the correct 2×2 cross-Hessian determinant.
* **The `-1j` Maslov prefactor and the resulting kernel** (`hf.py:404`). Combined with the
  Van Vleck density it reproduces **exactly** `(1/(iλ))·cosθ·e^{ikr}/r` — i.e. **RS-I without
  the `(1 − 1/(ikr))` near-field term** — with the correct sign under `exp(−iωt)` /
  `exp(+ikz)`. Verified against (a) the exact RS-I closed form I derived for the on-axis
  circular aperture, `U = e^{ikz} − (z/r_a)e^{ik r_a}` (the textbook `4sin²(k(r_a−z)/2)` is the
  same expression with the obliquity `z/r_a` set to 1), and (b) a 1-D radial quadrature of the
  code's own kernel:

  | z (a=20 µm, λ=633 nm) | RS-I exact `U` | kernel's continuum limit | code, N=2048 | err vs limit | err vs RS-I |
  |---|---|---|---|---|---|
  | 100 µm (N_F 6.32) | 0.221102−0.746613i | 0.220361−0.746850i | 0.220888−0.747505i | 1.08e-3 | 1.18e-3 |
  | 500 µm (N_F 1.26) | 1.758621−0.508327i | 1.758518−0.508681i | 1.758487−0.508491i | 1.05e-4 | 1.16e-4 |
  | 2000 µm (N_F 0.32) | −0.720852+0.622305i | −0.720821+0.622342i | −0.720867+0.622352i | 4.96e-5 | 5.11e-5 |

  Convergence with N (256/512/1024/2048) at z=100 µm: 2.86e-2 / 3.79e-3 / 2.88e-3 / 1.08e-3 —
  monotone, and the residual is the size of the omitted near-field term (9.99e-4 / 2.01e-4 /
  5.04e-5 at the three z). Complex values, not just intensities: **prefactor, phase, obliquity
  and sign are all right**.
* **Gaussian-beam validation** of the same kernel (table under "Alternative algorithms" 3):
  L2 vs the analytic Gaussian 7.0e-4 / 4.0e-4 / 2.0e-4 at z = 0.5/1/3 z_R, with ASM on the same
  grid at 1.0e-4 / 1.6e-4 / 2.2e-4.
* **`hfpi.accumulate_to_grid`'s cell-centred binning** (`:438–439`). `floor(x/dx + N/2 + 0.5)`
  is the exact inverse of the library's pixel-centred coordinate `x_i = (i − N/2)·dx`; pixel *i*
  collects `[x_i − dx/2, x_i + dx/2)`. Correct, and it matches `hf.py`'s and
  `HuygensSurface.grid()`'s coordinates.
* **`propagate_to_plane`** (`hfpi.py:243–281`) — the grazing-ray guard (`t` zeroed for dead
  rays so no ±inf leaks into positions), `delta_opl = n·|t|` with unit direction vectors, and
  `exp(+ikΔs)` with `k` from the *vacuum* wavelength while `Δs` already carries `n`. All
  consistent with CONVENTIONS §7.
* **`init_paths_stratified`'s 4-D stratification** (`:762–793`). `np.indices((n_iy,n_ix,n_th,n_ph))`
  really does build the cartesian product (the 4.11.2 fix is correct), the `cosθ` strata are
  equal-solid-angle, and the jitter is uniform within each cell.
* **The float32 OPL accumulator is harmless.** `init_paths_from_field` takes the `opl` dtype
  from the field (`:227`), so a complex64 source gets a float32 accumulator — but it holds only
  zeros before the first hop, `propagate_to_plane` promotes to float64 by NumPy's rules, and
  `trace()` returns a float64 `opd`. **Measured (`p8_final.py` §a)**: the first four alive path
  phases through `_hfpi_segment_trace` are **bit-identical** between a complex128 and a
  complex64 source. (Worth a comment; not a defect.)
* **MHS composition.** Two ASM subdomains vs one ASM over the summed z agree to **4.21e-16**
  with `bandlimit=False` (6.58e-5 with the Matsushima mask on both legs, as expected); total
  power preserved to 1e-9 relative across both hops. `run()`'s return-type matrix is exactly as
  documented, `asm_subdomain`'s pitch guard uses a correct *relative* tolerance
  (`mhs.py:382`), `aperture_subdomain` is a correct zero-thickness operator that `_validate`
  accepts, and `gbd_freespace_subdomain` returns a bare ndarray (no tuple leak).
* **`MhsPipeline.run` is re-entrant.** **Measured:** four concurrent `run()` calls on one
  pipeline object with different inputs give independent, correct results (`E_i × (i+1)` each).
  `run` mutates no instance state; the only shared mutable resource is the optional `store`.
* **`richards_wolf_focus`'s E_x / E_y vectorial pupil functions** (`:343–346`), the combined
  `1/√cosθ` apodisation + Cartesian-to-solid-angle Jacobian (`:251–265` — I re-derived
  `dx_p dy_p = f² cosθ sinθ dθ dφ`, so `pupil·(1/√c)·dx_p dy_p = pupil·√c·sinθ dθ dφ`, the
  correct aplanatic weight), the `exp(+ikz cosθ)` defocus (`:358`), and the `fft2` direction
  (`:396`) — all match the independent Debye–Wolf quadrature to 0.2–1.5 %, which is the FFT
  discretisation. **Only `E_z`'s sign is wrong.**
