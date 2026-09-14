# VERIFY-WP-B9 -- independent adversarial re-verification of WP-B9

Commit under test: `7592af4a` on `audit-fixes-2026-09`.  Pre-change
reference: `7592af4a^` (identical to `284daccc` for `lumenairy/raytrace/*`).
Every python run with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
MKL_NUM_THREADS=1`, one process at a time.  No git write command was run.

I did not write WP-B9.  Every number below comes from an oracle I wrote:
a 60-digit `decimal` sphere normal, four separate `trace()` calls,
`jax.jacfwd` through the independent `trace_jax` kernel, a
Richardson-extrapolated FD ladder, and my own 1542-array byte-identity
probe.  Where I reproduce one of the report's numbers I say so; where I
do not, I say that too.

---

## 0. Headline

**The six items do what the report says they do.  The DEFAULT path is
byte-identical, proved archive-to-archive with my own probe, and the two
opt-in switches behave as documented.  Nothing is NOT FIXED and nothing
is a REGRESSION.**

Four claims are overstated and are now corrected in the source comments
(the three file edits that carry those corrections are byte-neutral):

1. "the closed form ... never worse than the generic route at any height"
   and "<= 4 ULP" -- true up to `h = 0.95 |R|`, false above it.  Both
   routes are at the conditioning limit of `sqrt(1 - h^2/R^2)` there;
   at `h = 0.99994 |R|` the closed form is 1.5e-14 out (68 ULP) and at
   `R = -34.5 mm` it is **4.3x further from the truth than the route it
   replaces**.  The report's deferred item 6.2 proposes dropping the
   `valid` clamp because the closed form "is well-conditioned to
   `h = |R|`" -- **that premise is false**.
2. "reproduces `_surface_sag_derivative`'s out-of-domain policy exactly,
   so vignetting does not move" -- the two gates are computed from
   expressions that differ by ~1 ULP, and I found a real float64 position
   where **the generic route kills the ray `RAY_NAN` and the closed form
   refracts it**.  Confined to `sphere_normal='analytic'`.
3. "`JaxPrescription` is immutable in this package (`__slots__`)" --
   `__slots__` blocks new attribute NAMES, not writes to declared ones.
   `jp.radii = None` on a cached instance succeeds and is then served to
   every later caller.  New sharing hazard, introduced by the cache.
4. "a prescription whose values are unhashable ... simply skips the
   cache" -- `aux` is hashable by construction; that fallback is
   unreachable.  And "a NaN radius keys fine but never hits" -- it DOES
   hit when the same prescription dict is re-read.

And one measurement the report does not have, which the orchestrator
needs before it acts on section 6.1:

5. **At the SHIPPED defaults the two `ModalAsymptoticStillBitEqual` arms
   have 4.0 % headroom** (9.6157e-09 against the 1e-8 bar), not just
   under the flipped default.  And **`renormalize='exit'` alone trips
   them by 10.3 %** -- worse than `sphere_normal='analytic'` does --
   which the report's section 5 restatement list does not record.

---

## 1. Verdict table

| # | report claim | verdict | my oracle | my numbers |
|---|---|---|---|---|
| 0 | 225/225 probe arrays byte-identical between the pre-change archive and the same archive with only `raytrace/*.py` replaced | **VERIFIED** | my OWN 1542-array probe, archive-to-archive (`git archive 7592af4a^` vs `git archive 7592af4a`), child processes, cwd + `PYTHONPATH` = the archive, `lumenairy.__file__` asserted, never pytest | **1540 / 1540 common arrays byte-identical, 200 904 values, 0 moved.**  The only delta is the documented item-6 change (`Jan.asph`: parent raised `NotImplementedError`, child returns a Jacobian) |
| 1a | `renormalize='exit'` keeps `alive` / `error_code` byte-identical | **VERIFIED** | 2000-ray bundles on my own 3/5/7/11/17/25-surface spherical stacks, a 2-mirror Cassegrain and a TIR prism | identical `alive` AND `error_code` on all eight |
| 1b | drift `max\|dx\| = 6.2e-17 m`, `max\|dopd\| = 1.9e-16 m` against a derived `n_surf * eps * \|t\|` envelope | **VERIFIED** | the envelope, re-derived; growth measured against surface count | `max\|dopd\|` 3.47e-18 (3 surf) -> 1.39e-16 (25 surf), **below `n_surf * eps * \|opd\|_max` at every count**; `max\|dx\|` 1.7e-18 -> 2.5e-17; `max\|dL\|` 1.1e-16 -> 1.2e-15.  TIR prism (flats only): exactly 0 |
| 1c | the degenerate-ray diagnosis stays per surface -- a direction that collapses at surface 3 is reported at surface 3 | **VERIFIED** | a 7-surface stack with a NaN direction, an `inf` direction, a 0.9-slope ray and a DOE evanescent kick | death index `[-1, 0, 0, 4, -1, -1]` and codes `[0, 3, 4, 3, 0, 0]` **identical in both modes**, with and without the DOE |
| 1d | `_normalize_directions` runs ONCE under `'exit'`, ZERO times under `'surface'`, `_refract` once per surface in both | **VERIFIED** | call counting through a wrapped module global, 13 surfaces | `'surface'`: 0 / 13.  `'exit'`: 1 / 13.  Both `output_filter`s |
| 1e | two-sided: under `'exit'` some INTERMEDIATE bundle is measurably non-unit and the exit bundle is not | **VERIFIED** | `\|(L,M,N)\| - 1` over every history bundle | `'surface'`: intermediate 1.11e-16, exit **0.0**.  `'exit'`: intermediate **9.99e-16**, exit 1.11e-16 |
| 1f | `analysis/ghost.py` and the FD differential path are unchanged BY CONSTRUCTION | **VERIFIED** | the signatures, plus the byte-identity sweep | `_refract(rays, surface, n1, n2, *, renormalize=True, sphere_normal='generic')` -- both new arguments keyword-only with the pre-change values; every `ghost.*` and `Jfd.*` probe array byte-identical |
| 2a | closed form within **4 ULP** of a 60-digit oracle at every height and radius | **VERIFIED-WITH-NOTES** | my own 60-digit `decimal` normal, evaluated from the EXACT binary values of the float64 inputs | holds over the report's grid (`h <= 0.95 \|R\|`, 5 radii): worst 2.8e-16 vs the 8.88e-16 bar.  **Fails above it**: 1.5e-15 at `0.99 \|R\|`, 2.3e-15 at `0.999 \|R\|`, **1.5e-14 (68 ULP) at `0.99994 \|R\|`** -- still inside the surface's own `h^2/R^2 < 0.9999` domain |
| 2b | "never worse than the generic route at any height" | **NOT AS STATED** | the same oracle, 1200 points (10 radii x 25 heights x 5 azimuths) | closed form closer at **500** points, generic closer at **178**, tie at 522.  Counter-examples inside the domain: `R = -34.5 mm` at `0.99994 \|R\|` **4.12e-15 vs 9.49e-16 (4.3x worse)**; `R = -1 m` at `0.9999 \|R\|` 4.40e-15 vs 4.82e-16 (9.1x); `R = 51.68 mm` at `0.999 \|R\|` 2.29e-15 vs 1.05e-15.  The engineer's own test asserts `e_fast <= e_slow + 2**-52` (a 1-ULP slack) -- the TEST is sound, the REPORT's unqualified prose is not |
| 2c | "reproduces `_surface_sag_derivative`'s out-of-domain policy exactly ... so vignetting does not move" | **NOT AS STATED** | the two gate expressions, then `_refract` end to end | the closed form gates on `(x*x+y*y)/(R*R)`, the generic route on `(1+conic)*sqrt(x*x+y*y)**2/R**2`; they differ by ~1 ULP.  At `R = 51.68 mm`, `x = y = 0.036541451242116801 m`: generic `nz = NaN` -> `alive=False, error_code=4 (RAY_NAN)`; analytic `nz = 1.0000000000005e-02` -> `alive=True, error_code=0`.  **Vignetting does move**, in a ~1-ULP-wide band, under `'analytic'` only |
| 2d | it is a unit vector to <= 2 ULP, better than the generic route's computed magnitude | **VERIFIED** | `\|n\| - 1` over the 25-height grid | closed form **1 ULP** at all four radii; generic 1 ULP at `R = 51.68 / -34.5 mm` and **2 ULP** at `R = 500 mm / -1 m` |
| 2e | the NaN-position policy difference is confined to non-finite input and never reaches the public API | **VERIFIED** | direct calls, then the byte-identity sweep | `x = NaN`: analytic `(nan, -0.0, nan)`, generic `(-0.0, -0.0, 1.0)` (the fabricated axial normal).  `x = inf`: analytic `(-inf, -0.0, nan)`, generic `(nan, nan, nan)`.  Both non-finite -> `RAY_NAN`; the ray-sphere quadratic refuses a non-finite position first, and the 1540-array sweep shows nothing moves |
| 2f | ONE shared predicate, so normal and intersection cannot disagree (the v4.12.0 failure mode) | **VERIFIED** | `_is_pure_spherical` over 15 surface kinds | `True` for a plain sphere, a `conic == 0.0` sphere, `aspheric_coeffs={}`, `field_decenter=(0,0)` and a spherical MIRROR (correct -- same normal); `False` for `radius=inf`, `radius=None`, `conic=1e-300`, `aspheric_coeffs={4: 0.0}`, biconic, freeform, non-zero `field_decenter` / `field_tilt`.  `radius=NaN` returns `True` (`not np.isinf(nan)`), the same hole the PRE-change inline predicate had -- both branches then produce NaN and the ray dies; not a regression |
| 2g | gate: the Maslov cross-backend asymptotic test, `ModalAsymptoticStillBitEqual`, `w6_a2_v2_star`, `d7` and every `_lens_traced` fixture stay green | **VERIFIED** | ran them | w6 asymptotic + a4 Maslov + a16: **191 passed**; the two knife-edge pins: **3 passed**; a26 (the 2.371 / 1.683 urad pins) + d7: **48 passed**; `-k traced`: **344 passed, 2 skipped** |
| 3a | the concatenated bundle is byte-identical to four separate traces | **VERIFIED** | four separate `trace()` calls, issued by me | `max \|delta\| = 0` on EVERY field (x, y, z, L, M, N, opd, alive, error_code) of all four sub-bundles, on 8 prescriptions: vignetting (on axis and 2 deg), TIR, missed-surface, asphere (on axis and 3 deg), field-decentred, field-tilted |
| 3b | ... INCLUDING when one sub-fan contains a dead ray | **VERIFIED** | same | 14 / 13 vignetted rays and 12 missed rays in the tangential sub-fan: still byte-identical.  The `np.any` guards do not couple |
| 3c | the aspheric Newton's `converged.all()` early exit never materialises | **VERIFIED** | sag-evaluation counting + 90-case sweep | the chief gets 1 extra iteration in the joint bundle (2 sag evals alone, 4 joint) and `max\|dy\| = 0`.  Swept 3 departures x 3 base conics x 10 gaps from 0.5 m to 256 m -- deliberately across `\|t\| = 1e-15/eps = 4.5 m`, where the loop's ABSOLUTE tolerance is the ray's own residual step: **0 of 180 sub-bundle comparisons differ, worst `\|dy\| = 0`** |
| 3d | RT-5 `ey(0) == ex(0) == 0` exactly | **VERIFIED** | `ray_fan_data` at 0 / 1 / 3 deg on decentred, tilted, vignetting and aspheric prescriptions | `ey(0) = ex(0) = 0.0` exactly, all 12 |
| 3e | `_bundle_slice` returns VIEWS, documented | **VERIFIED** | inspection | the docstring says so; the fan analytics only read them |
| 4a | every trace-changing field re-keys | **VERIFIED** | field-by-field mutation, checking cache size, `aux` inequality AND the traced answer | radius (+1 ulp), radius (-1e-6), conic (+1 ulp), thickness[0] (+1 ulp), A4 (+1 ulp), a new A8 power, a removed A6, `semi_diameter` (+1 ulp), `glass_after`, `glass_before`, an added surface, `aperture_diameter` when it is the resolving value, wavelength (1.31 -> 1.55 um), `surface_diffraction` (absent -> present, and 1 ulp on the period): **13 / 13 MISS with a different `aux`** |
| 4b | ... and three inputs correctly do NOT re-key | **VERIFIED (new detail)** | the same sweep + the traced answer | the TRAILING thickness (the builder reads `n_surf - 1` gaps; answer unmoved), an `aperture_diameter` a per-surface `semi_diameter` shadows (answer unmoved), and the wavelength beyond the indices it resolves (`trace_jax` passes it to the kernel separately -- `wl + 1 ulp` gives a bit-identical answer, 1.55 um does not and re-keys).  The report's "a thickness / the top-level `aperture_diameter` produces a MISS" is true only where those are load-bearing |
| 4c | a mutated glass registry re-keys on its own | **VERIFIED** | registering a probe glass at n = 1.5, building, re-registering at 1.7, rebuilding | different object, `n_pre/n_post` `((1.0, 1.5), (1.5, 1.0))` -> `((1.0, 1.7), (1.7, 1.0))`, traced answer moved |
| 4d | in-place numpy mutation after the first call cannot serve a stale hit | **VERIFIED** | 0-d array VIEWS into live arrays stored as `radius` and as an `aspheric_coeffs` value, mutated after the first build | re-keyed (`aux` differs) and the answer moved 1.42e-02.  `aux` is rebuilt from the live dict on every call, so the cache is keyed on a snapshot, never on identity |
| 4e | LRU 32 with eviction | **VERIFIED** | 40 distinct prescriptions, then a recency probe through a real `_build_jax_prescription` hit | size caps at 32; oldest evicted; after re-building entry 0 (a hit -> `move_to_end`) and adding one more, entry 0 survives and entry 1 is evicted |
| 4f | traces are bit-identical cold and warm; the clearer is registered centrally | **VERIFIED** | `tobytes()` on a 3-ray trace; `clear_asm_caches()` | cold == warm bitwise; `clear_asm_caches()` takes the cache 1 -> 0 |
| 4g | "`JaxPrescription` is immutable ... so callers share one instance safely" | **NOT AS STATED** | direct attribute writes on a cached instance | `jp.radii = None` **succeeds** and the next `_build_jax_prescription` returns the poisoned object.  `jp.new_attr = 1` correctly raises.  A new sharing hazard: before the cache every caller got a private object.  No caller in the package writes to one; the note now states the read-only contract |
| 4h | "an unhashable `aux` falls back to building every time" | **NOT AS STATED** | every element of `aux` traced back to its constructor | `n_surf` is `len()`, everything else passes through `int()` / `float()` / `tuple()` / `sorted()`.  Numpy scalars, 0-d arrays, `bool` and `np.int64` powers all normalise; an odd power is refused by `Surface` first.  The `except TypeError` branch is **unreachable** -- defensive, not a behaviour |
| 4i | "a NaN radius keys fine but never hits" | **NOT AS STATED** | two builds from the same dict, then from a fresh one | the same dict re-read **HITS** (`float(x)` on a float returns the same object, so the tuple comparison short-circuits on identity); a freshly created NaN misses.  Benign -- the object served matches the prescription asked for -- but the opposite of what the docstring said |
| 5a | mean `r/R` 0.580645 -> 0.665834, mean `r^2/R^2` 0.419355 -> 0.500000 exact | **VERIFIED** | the closed-form identity, on MY OWN ring counts | at the defaults (6, 36): **0.580645 / 0.419355** and **0.665834 / 0.500000** -- reproduces the audit's 0.5806 and the report exactly.  Vogel `mean r^2/R^2 - 1/2` <= **1 ULP** at (6,36), (4,12), (3,7), (10,50), (1,5) and (8,21) |
| 5b | the default does not move | **VERIFIED** | byte comparison against `pattern='rings'` and against the ring formula rebuilt from the docstring | identical bytes in x and y at (6,36,chief), (4,12,chief) and (5,9,no chief); and `mk.rings` / `mk.rings_f` byte-identical parent-to-child |
| 5c | spot RMS reads LARGER with the area-true pupil | **VERIFIED** | three prescriptions of my own | **+4.49 %**, **+4.33 %**, **+9.28 %** -- all upward.  (My absolute values differ from the report's 108.2 / 106.2 um because my image planes are at the prescription thickness, not at the paraxial focus; the report asserts the DIRECTION, which is what reproduces) |
| 5d | `pattern=` is threaded through every in-package consumer | **VERIFIED-WITH-NOTES** | `grep -rn "make_rings("` over `lumenairy/` | all three `raytrace/` call sites carry it (`through_focus_rms(pattern=)`, `trace_prescription(ray_pattern=)`, `raytrace_system(ray_pattern=)`), and `through_focus_rms()` default is bitwise equal to `pattern='rings'`.  Ten call sites outside `raytrace/` do not: `analysis/field.py:760,851,1000`, `analysis/ghost.py:821` (both listed in the report) **and six in `lumenairy/ui/` that the report omits** -- see section 5 |
| 5e | `pattern=` / `renormalize=` / `sphere_normal=` validate and name themselves | **VERIFIED** | 7 bad values | every one raises `ValueError` naming the function, the parameter, the bad value and both legal values |
| 6a | the analytic aspheric Jacobian agrees with FD to the FD's own truncation, scaling as `h^2` | **VERIFIED** | a 6-rung FD ladder (4e-6 .. 1e-7) with Richardson extrapolation, on aspheres whose polynomial departure is REAL (`A4 h^4 = 41 um` at h = 8 mm) | ratio **4.00 / 4.00** at every rung on all six cases -- exactly `h^2`.  Richardson-extrapolated residual **5.5e-11 .. 1.6e-10** |
| 6b | ... and the agreement is truncation, not error | **VERIFIED, and stronger** | `jax.jacfwd` through `trace_jax` -- a DIFFERENT intersection kernel (`_intersect_jax`), exact to machine precision | **max \|dJ\| = 3.6e-15 .. 5.7e-14, max relative 2.3e-16 .. 2.8e-15** on six aspheric prescriptions.  The analytic Jacobian is exact, not merely FD-consistent |
| 6c | the polynomial terms are load-bearing | **VERIFIED** | tracing the BASE CONIC instead | gap **1.09 .. 1.13e+2** -- 13 to 15 decades above the `jacfwd` agreement |
| 6d | the Newton refinement lands where `trace`'s own 10-step Newton lands | **VERIFIED** | `raytrace.trace` (no shared code) | `max\|dx\| <= 1.7e-18 m`, `max\|dy\| = 0`, `max\|du\| <= 1.1e-16`, `max\|dopd\| <= 2.3e-17` |
| 6e | the 6-step budget is headroom -- bit-identical from 2 steps | **VERIFIED** | forcing `_ADRT_ASPHERIC_NEWTON_STEPS` from 40 down to 1 | bit-identical from **2** on all six cases (and from 1 on five of them).  The claim is exact |
| 6f | the numba kernel is excluded for aspheres | **VERIFIED** | `_adrt_surfaces_numba_eligible` | `False` on all six aspheric prescriptions, `True` on the pure conic.  Also `False` for `aspheric_coeffs={4: 0.0}` -- see the note below |
| 6g | odd powers are refused by `Surface` first and by `_adrt_step`'s backstop for a hand-built object | **VERIFIED** | powers 3, 5, 4.5 and a hand-built surface-like object | `Surface` raises `ValueError` on all three; the hand-built odd power raises `ValueError: _adrt_step: contains ODD aspheric power(s) [3]` |
| 6h | `aspheric_coeffs` all zero is a no-op | **VERIFIED-WITH-NOTES** | the conic path | NOT byte-identical: `{4: 0.0}` selects the Newton branch, agreeing with the conic path to **3.4e-16 relative (1.5 ULP)** but not bitwise, and it **loses numba eligibility**.  Correct arithmetic, different code path and a slower one.  Pinned |

---

## 2. Defects found and fixed

All three edits are inside `lumenairy/raytrace/`, change no default, no
signature and no number: the probe reads **1542 / 1542 arrays identical**
with my edited package overlaid on the `7592af4a` archive, and 1540 / 1540
against the pre-change archive.

### D1 -- `_trace_fan_set` would relabel a dead launch ray `RAY_OK`

`ray_fan.py:_trace_fan_set` built the joint bundle's `error_code` as

```python
error_code=np.concatenate(
    [np.zeros(b.n_rays, dtype=np.uint8) if b.error_code is None
     else b.error_code for b in bundles]),
```

`np.zeros` is `RAY_OK`, but `RayBundle.__post_init__` synthesises a
missing `error_code` as "alive -> `RAY_OK`, dead -> `RAY_TIR`".  The
branch is unreachable today (`__post_init__` always fills the field, so
no input ever arrives with `None`), which is why nothing measured it --
but it is a latent first-failure-wins violation sitting in a helper whose
whole job is to preserve per-ray state across a concatenation.

**Fail-before:** the branch cannot be reached through the public API, so
there is no failing trace to show; the defect is demonstrated by
construction --
`RayBundle(x=..., alive=[False], error_code=None).error_code == [RAY_TIR]`
against the helper's `[RAY_OK]`.  Fixed by reading `b.error_code`, which
every input already carries, with the reason in the comment.

### D2 -- source comments that claimed more than the code delivers

`CONVENTIONS.md` sec. 2 requires a comment to describe current behaviour.
These six statements, across three files, did not:

* **`surface._sphere_normal`: "no cancellation"** -- `nz = sqrt(1 - u)`
  with `u = h^2/R^2` is exactly a cancelling subtraction.  Measured
  against my 60-digit oracle, both routes leave the 4-ULP bar above
  `h = 0.95 |R|` and reach ~1.5e-14 at the domain edge.  **Fail-before:**
  the claim is falsified by the table in section 3.2 below.
* **`surface._sphere_normal`: "reproducing the out-of-domain policy of
  `_surface_sag_derivative` exactly"** -- falsified by a real float64
  position (section 3.3) where the two routes disagree about whether the
  ray exists.
* **`surface._surface_normal`: "the closed form is the more accurate of
  the pair"** -- true on average (500 wins vs 178 over 1200 points),
  false as a bound.
* **`jax_trace`: "`JaxPrescription` is immutable in this package
  (`__slots__`, no attribute writes after construction), so callers share
  one instance safely"** -- **fail-before:**
  `jp = _build_jax_prescription(P, wl); jp.radii = None;
  _build_jax_prescription(P, wl).radii is None` -> `True`.  The instance
  is shared and rebindable.
* **`jax_trace`: "a prescription whose values are unhashable ... simply
  skips the cache"** -- unreachable; `aux` is hashable by construction.
  And "a NaN radius ... never hits" -- it hits on a re-read of the same
  dict.
* **`ray_fan._trace_fan_set`: "`|dt| < 1e-15`, so it moves `t` by at most
  an ULP"** -- that tolerance is absolute metres, ~45 ULP of a 0.1 m `t`.
  The ULP conclusion is right for a different reason (a converged ray's
  own residual step is `~eps |t|`), and the note now says which.

Each is replaced by the measured statement, with the measurement.

### D3 -- not fixed, reported: the two knife-edge pins are knife-edge TODAY

See section 4.  No code change; it is not mine to make.

---

## 3. The three measurements the report does not have

### 3.1 Byte identity, archive to archive, with my own probe

`git archive 7592af4a^ lumenairy` and `git archive 7592af4a lumenairy`
extracted read-only into
`scratchpad/verify_b9/{parent,child}`; `diff -rq` confirms **exactly the
seven `raytrace/*.py` files** differ.  The probe runs in a child process
whose cwd AND `PYTHONPATH` are the archive root and asserts
`lumenairy.__file__` is inside it before importing anything else.  Never
through pytest, never against the working tree.

Coverage (1542 arrays, 200 904 float64 / uint8 / bool values): `trace`
with full `output_filter='all'` history on seven designs of my own -- a
12-surface all-spherical double-Gauss-like stack, a two-mirror conic
Cassegrain, a TIR prism path, an f/4 plano-convex singlet, an `A4/A6`
asphere, a biconic and a coord-break fold -- at 0, 2 and 5 deg; grazing
(`h/|R|` from 0 to 1.2), TIR and missed-surface bundles; all five ray
launchers; `ray_fan_data` / `opd_fan_data` on all seven designs at two
angles; `trace_world` + both world fans; `spot_rms`, `spot_geo_radius`,
`refocus`, `through_focus_rms`; `system_abcd`, `seidel_coefficients`,
`first_order_data`, `compute_pupils`, `find_paraxial_focus`; a DOE-kicked
trace; `trace_prescription` at three patterns; `raytrace_system` at three;
`ray_transfer_jacobian` and `ray_transfer_jacobian_analytic` (composite
and per-surface) on five designs; and `analysis.ghost`
(`enumerate_ghost_paths`, `ghost_analysis`, `retrace_ghost_path`).

> **1540 / 1540 common arrays byte-identical.  0 moved.**
> Only `Jan.asph` differs in EXISTENCE: the parent raised
> `NotImplementedError`, the child returns a Jacobian.  That is item 6.

Re-run with my own edited `raytrace/*.py` overlaid on the child archive:
**1542 / 1542 identical to `7592af4a`.**

### 3.2 The closed-form normal is conditioning-limited, not uniformly better

Absolute error of `nz` against my 60-digit `decimal` oracle, bar
`4 * 2**-52 = 8.88e-16` (the item-2 test's own metric), at
`x = y = h/sqrt(2)`:

| `h/\|R\|` | R = 51.68 mm | R = -34.5 mm | R = 500 mm | R = -1 m | R = 2 mm |
|---|---|---|---|---|---|
| | fast / slow | fast / slow | fast / slow | fast / slow | fast / slow |
| 0.50 | 1.1e-16 / 0 | 0 / 1.1e-16 | 1.1e-16 / 0 | 1.1e-16 / 0 | 1.1e-16 / 2.2e-16 |
| 0.95 | 2.2e-16 / 1.7e-16 | 2.8e-16 / 2.8e-16 | 2.2e-16 / 2.2e-16 | 2.2e-16 / 2.2e-16 | 0 / 0 |
| 0.99 | 1.5e-15 / 1.5e-15 | 3.6e-16 / 3.6e-16 | 4.2e-16 / 4.2e-16 | 4.2e-16 / 4.2e-16 | 3.9e-16 / 3.6e-16 |
| 0.999 | **2.3e-15** / 1.0e-15 | 1.8e-15 / 1.8e-15 | 8.8e-16 / 8.8e-16 | 2.4e-16 / 2.4e-16 | 8.1e-16 / 2.0e-15 |
| 0.9999 | 1.5e-15 / 1.5e-15 | 1.4e-15 / 1.4e-15 | **2.4e-15** / 1.5e-15 | **4.4e-15** / 4.8e-16 | 1.7e-15 / 9.5e-15 |
| 0.99994 | **1.5e-14** / 1.5e-14 | **4.1e-15** / 9.5e-16 | 5.3e-16 / 5.6e-15 | 2.1e-15 / 3.0e-15 | **7.2e-15** / 3.0e-15 |

`0.99995 |R|` is the clamp, so every row above is a legal point on the
surface.  The theory matches: the relative error of `sqrt(1 - u)` is
bounded below by `eps/2 * u/(1 - u)` for any float64 evaluation, which at
`u = 0.9999` is 1.1e-12 relative -- ~5000 ULP of `nz` -- for BOTH routes.
Neither route can do better; the information is not in the inputs.

**Consequence for the report's deferred item 6.2.**  It proposes dropping
the `valid` gate in the analytic branch "because the closed form needs no
clamp at all (`nz = sqrt(1 - h^2/R^2)` is well-conditioned to `h = |R|`)".
It is not.  Dropping the clamp would let rays through in exactly the band
where `nz` carries 1e-12 relative error and rising.  If the clamp is to
go -- and there is a real R4-family false kill behind it -- the
justification has to be "a grazing normal with 1e-12 relative error beats
a `RAY_NAN`", which is arguable, rather than "the formula is accurate
there", which is measurably false.

### 3.3 The domain gates straddle -- vignetting DOES move under `'analytic'`

```
R  = 0.051679999999999997
x  = y = 0.036541451242116801
(x*x + y*y)/(R*R)                  = 0.99989999999999990   -> analytic: VALID
(1+0)*sqrt(x*x+y*y)**2 / R**2      = 0.99990000000000001   -> generic : OUT OF DOMAIN

_refract(..., sphere_normal='generic' ): alive=False error_code=4 (RAY_NAN)
_refract(..., sphere_normal='analytic'): alive=True  error_code=0  L=-0.5270024142973004
```

Found by a directed `nextafter` walk around `h = |R| sqrt(0.9999)`; four
such points across seven radii and six azimuths.  The band is ~1 ULP of
`h` wide and only exists under the opt-in, so the shipped default and the
byte-identity sweep are untouched -- but the report's "so vignetting does
not move" is not a property of the code.  Note which side is *right*: the
closed form's gate has one fewer rounding, so it is the more accurate
predicate; "fixing" it to match the generic route would make it worse.
Recorded, not changed.

---

## 4. The two knife-edge pins, under BOTH defaults

Measured by importing `lumenairy` from the read-only `7592af4a` archive
(so no other engineer's uncommitted edit reaches the number), flipping
`trace.__defaults__` / `trace_world.__defaults__` on the function objects
-- which is exactly what "flip the default" means, and reaches every
module that already imported them -- and then running the two pins'
own computations from the repository's test modules.

| defaults | `lg00` ratio (bar 1e-8) | `4mode` ratio | `w6_a2` worst (bar 1e-15) |
|---|---|---|---|
| **`surface` / `generic` (SHIPPED)** | **9.6157e-09** -- 4.0 % margin, PASS | **9.6155e-09** -- 4.0 %, PASS | **6.408e-16** -- 56 % margin, PASS |
| `surface` / `analytic` | 1.0391e-08 -- **FAIL, 3.9 % over** | 1.0391e-08 -- FAIL | 1.150e-15 -- **FAIL, 15 % over** |
| `exit` / `generic` | 1.1027e-08 -- **FAIL, 10.3 % over** | 1.1027e-08 -- FAIL | 8.765e-16 -- PASS |
| `exit` / `analytic` | 1.0905e-08 -- **FAIL, 9.0 % over** | 1.0904e-08 -- FAIL | 5.862e-16 -- PASS |

Three things follow, two of them new:

1. **The report's numbers reproduce exactly.**  1.039e-08 on both arms and
   1.150e-15 on `w6_a2` under `sphere_normal='analytic'` -- four
   significant figures, independently.  The mechanism is real.
2. **These pins are knife-edge at the SHIPPED defaults**, not only under
   the flipped ones: 9.6157e-09 against 1.000e-08 is 4.0 % of margin on
   an S4 floor bar whose quantity is bimodal (one pixel changing saddle
   basin is worth ~1.04e-8 relative).  Any unrelated last-bit change to
   the trace -- another work package's arithmetic, a numpy release, a
   different BLAS -- flips them.  That is a live fragility in the tree as
   it stands.
3. **`renormalize='exit'` trips the same two arms, harder** (10.3 % over,
   against `sphere_normal`'s 3.9 %), and it is the ONE that leaves
   `w6_a2` green.  The report's section 5 lists the restatement as
   conditional on `sphere_normal='analytic'` and its section 6.1
   recommends flipping `sphere_normal` first "then consider
   `renormalize`" -- on this box the `renormalize` flip is the bigger
   perturbation of those pins.  The proposed edit (`1e-8 -> 3e-8`) still
   covers 1.103e-08, so the numeric restatement stands; the CONDITION on
   it does not.

---

## 5. Requested changes outside my ownership

1. **`tests/unit/test_audit_propagation.py::TestAuditFixesV4_14_0_agent_1_1APropagateModalAsymptoticStillBitEqual`**
   -- I endorse the report's section 5.2 edit (`1e-8` -> `3e-8` on both
   `assert max_abs < 1e-8 * max(cold_peak, 1.0)` arms, lines 3528 and
   3608) **and ask for it NOW, not only if a default flips**: measured at
   the shipped defaults the two arms read 9.6157e-09 and 9.6155e-09
   against the 1.000e-08 bar, i.e. 4.0 % of margin.  Suggested comment,
   with my measurement:
   `# one saddle-basin flip = 1.04e-8 relative; measured 2026-09-13 at the`
   `# shipped defaults 9.616e-9 (4.0 % margin), 1.039e-8 under`
   `# sphere_normal='analytic' and 1.103e-8 under renormalize='exit'.`
   Owner: whoever owns `test_audit_propagation.py`.
2. **`tests/unit/test_niche_audit_w6_asymptotic.py::test_w6_a2_v2_star_is_untouched_by_the_verdict_fix`**
   -- the report's `1e-15` -> `5e-15` edit is correct and is needed only
   if `sphere_normal='analytic'` becomes the default (measured 1.150e-15
   there; 6.408e-16 shipped, 8.765e-16 under `renormalize='exit'`,
   5.862e-16 under both).  Lower priority than item 1.
3. **`lumenairy/ui/model.py:2806,2810,3033`, `lumenairy/ui/rayfan_dock.py:209`,
   `lumenairy/ui/tolerance_dock.py:159`, `lumenairy/ui/wavefront_map_dock.py:133`**
   -- six `make_rings` call sites that inherit the centre-weighted pupil
   and can now pass `pattern='vogel'`.  The report's section 5.6 lists
   `analysis/field.py` and `analysis/ghost.py`; these six are the rest of
   the set and are not mentioned anywhere.  No edit requested -- each
   owner's judgement, exactly as for the analysis pair -- but the list
   should be complete before anyone believes the pupil weighting has been
   surveyed.
4. **`lumenairy/glass.py`** -- I confirm the report's section 5.1
   measurement stands and repeat the request: `get_glass_index('N-BK7',
   wl)` at 16.8 us against `'air'` at 0.17 us is 64 % of the residual
   `_build_jax_prescription` cost, and `trace()` pays it in its own
   prologue too.
5. **`propagators/gbd.py:3373` / `propagators/fga.py:817`** -- confirmed:
   with an ASPHERIC prescription `jacobian='auto'` no longer falls back to
   FD.  I measured the two agreeing to 6.3e-9 .. 6.3e-6 absolute at the
   default `h_pos = 1e-6` (the gap scaling exactly as `h^2`, so it is the
   FD's truncation) with the analytic side exact to 5.7e-14 against an
   independent kernel -- a strict improvement.  The comment's
   parenthetical list of uncovered kinds should drop "aspheric".

---

## 6. Follow-up

1. **Decide what the `valid` clamp in `_sphere_normal` is for** before
   acting on the report's deferred item 6.2.  Its stated justification is
   false (section 3.2).  A defensible version: keep the clamp, or drop it
   with the honest rationale that a grazing normal carrying ~1e-12
   relative error is more useful than a `RAY_NAN`, plus a vignetting
   sweep.  Either way the *generic* route's clamp has to move with it or
   the two gates diverge by a whole decade of `h` instead of 1 ULP.
2. **The two `ModalAsymptoticStillBitEqual` arms need their bar restated
   regardless of WP-B9** (section 4, request 1).  4.0 % of margin on a
   bimodal quantity is not a pin, it is a coin.
3. **`JaxPrescription` could be made genuinely immutable** -- a
   `__setattr__` that refuses after construction, or `@dataclass(frozen=True)`
   -- which would turn the section-2 D2 hazard into an impossibility.  I
   did not do it: `test_b9_i4_a_repeated_build_returns_the_same_object`
   pins instance identity (so returning a copy is out) and freezing the
   class touches `trace_jax_with_params` and the pytree unflatten path,
   which is a design change, not a verification.  ~1 h plus the JAX
   pytree round-trip tests.
4. **The Newton loop's `|dt| < 1e-15` acceptance test is an absolute
   metre tolerance** in a library whose `t` ranges over ten decades.  It
   did not bite in 180 sub-bundle comparisons across `|t| = 4.5 m`
   (section 1, row 3c), and it is not a WP-B9 construct -- but it is the
   one genuinely ray-count-coupled quantity in `trace`, and a relative
   form (`|dt| < 1e-12 * max(|t|, t_scale)`) would remove the coupling
   argument entirely.  Whoever owns `intersection._intersect_surface`
   next.
5. **`aspheric_coeffs={4: 0.0}` is a trap**: it selects the Newton branch
   and loses numba eligibility for a surface that is arithmetically the
   base conic.  Now pinned; a future `_adrt_aspheric_items` could drop
   zero coefficients, but that would move the last bit of an existing
   answer, so it is a decision, not a cleanup.

---

## 7. Commands run

All with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`,
one process at a time.

| command | result | duration |
|---|---|---|
| `pytest tests/unit/test_audit2609_b9_raytrace_perf.py -q -p no:randomly` | **75 passed** (65 WP-B9 + 10 VERIFY) | 16.6 s |
| `pytest tests/unit/test_v4_16_0_walker_all_symmetry.py tests/unit/test_audit2609_a15b_reexports.py tests/unit/test_v4_14_1_dispatcher_pin_cache_clears.py tests/unit/test_audit2609_a17_history_relocation.py -q` | **797 passed** | 42.2 s |
| `pytest ".../TestAuditFixesV4_14_0_agent_1_1APropagateModalAsymptoticStillBitEqual" ".../test_w6_a2_v2_star_is_untouched_by_the_verdict_fix" -q` | **3 passed** | 8.4 s |
| `pytest tests/unit -q -k "raytrace or exit_vertex or opd_fan or ghost or jax_trace or ray_fan or differential"` | **648 passed, 1 skipped** (PySide6) | 318.6 s |
| `pytest tests/unit/test_audit2609_a26_decentred_exit_reference.py tests/unit/test_niche_d7_decentred_fit.py -q` | **48 passed** (the 2.371 / 1.683 urad pins among them) | 357.1 s |
| `pytest tests/unit/test_niche_audit_w6_asymptotic.py tests/unit/test_audit2609_a4_verify_maslov_asymptotic.py tests/unit/test_audit2609_a16_lens_config_round_trip.py -q` | **191 passed** | 195.6 s |
| `pytest tests/unit -q -k traced` | **344 passed, 2 skipped** | 583.6 s |
| `python validation/run_all.py test_raytrace` | **1/1 file pass**; `validation/raytrace` **54/54** | 12.9 s |
| `python -m ruff check lumenairy/raytrace/ tests/unit/test_audit2609_b9_raytrace_perf.py` | **All checks passed** | -- |
| `python scripts/record_history_fingerprints.py --check` | **every `lumenairy.raytrace.*` document OK** (13/13) | -- |

Non-pytest child-process runs, each against an extracted `git archive`
tree with `lumenairy.__file__` asserted inside it
(`scratchpad/verify_b9/`):

| script | what it measures | result |
|---|---|---|
| `probe.py` x 3 (parent, child, my tree overlaid) | byte identity, 1542 arrays / 200 904 values | 1540/1540 parent-vs-child, 1542/1542 child-vs-mine |
| `sphere_oracle.py` | 60-digit `decimal` normal, 1200 points, 10 radii | 500 / 178 / 522 win-lose-tie; gate straddle; NaN policy; 15-kind predicate table |
| `sphere_grazing.py` | ULP vs the oracle from `0.5 \|R\|` to the clamp; directed `nextafter` gate search | the section 3.2 table; 4 gate mismatches |
| `gate_endtoend.py` | does the straddle reach `_refract`? | yes -- `RAY_NAN` vs a live refracted ray |
| `fan_attack.py` | 4-trace oracle on 8 prescriptions incl. dead rays; RT-5; sag-eval counts | all byte-identical; `ey(0)=ex(0)=0.0` |
| `newton_couple.py` | 90 aspheric cases x 2 sub-bundles across `\|t\| = 4.5 m` | 0 differ, worst `\|dy\| = 0` |
| `cache_attack.py`, `cache_attack2.py` | 13-field re-key sweep, in-place mutation, glass re-key, LRU, hashability, immutability, NaN keying | section 1 rows 4a-4i |
| `jac_attack.py` | Richardson FD ladder + `jacfwd`-through-`trace_jax` oracle; Newton budget; numba exclusion | section 1 rows 6a-6h |
| `renorm_pattern_attack.py` | `'exit'` drift vs surface count, degenerate-ray death index, call counts, Vogel moments, spot shift | section 1 rows 1a-1f, 5a-5e |
| `pin_headroom.py` x 4 | the two knife-edge pins under all four default combinations | section 4 |

## 8. Files changed

* `lumenairy/raytrace/surface.py` -- `_sphere_normal` and
  `_surface_normal` docstrings (comments only).
* `lumenairy/raytrace/jax_trace.py` -- the cache note,
  `_build_jax_prescription`'s docstring and the `except TypeError`
  comment (comments only).
* `lumenairy/raytrace/ray_fan.py` -- `_trace_fan_set`'s `error_code`
  construction (D1) and its exactness note.
* `docs/history/lumenairy.raytrace.ray_fan.md` -- re-recorded
  (`surface.py` and `jax_trace.py` showed no fingerprint drift, which is
  the gate confirming those two edits were documentation-only).
* `tests/unit/test_audit2609_b9_raytrace_perf.py` -- ten VERIFY-WP-B9
  pins appended; nothing existing weakened or removed.
* `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/VERIFY_WP-B9.md`
  (this file) and `VERIFY_WP-B9_CHANGELOG.md`.

Nothing else was touched.  `lumenairy/elements/_lens_traced.py`,
`lumenairy/propagators/carrier.py`, `lumenairy/propagators/system.py`,
`lumenairy/elements/_lens_real.py`, `lumenairy/analysis/*` and
`lumenairy/pmm/*` -- the files other engineers are mid-edit on -- were
read but never written.
