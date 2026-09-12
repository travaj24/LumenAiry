# WP-A3 changelog text — traced-lens family (`_lens_traced*`, `_lens_imap`, `_math/chebyshev`)

### Fixed -- caustic siblings: `caustic='multibranch'` / `'uniform'` evaluated on the last surface's SAG, not on the output plane (audit S1, P0)

`lumenairy/elements/_lens_traced_multibranch.py::_trace_launch_grid` and both
meridional traces in `lumenairy/elements/_lens_traced_uniform.py`
(`_trace_meridional_fold`, `_trace_meridional_cusp`) advanced `rt.trace`'s
`image_rays` — which sit at `z = sag(rho)` of the last surface — by
`output_plane_distance / N`, so every ray was read at `z = sag(rho) + d`: a
ray-DEPENDENT longitudinal position, not a plane.  Two branches arriving at the
same `(x, y)` therefore came from different `rho`, different `sag`, different
`z`, and their interference was computed between fields at different
longitudinal positions.  `output_plane_distance = 0.0` (the DEFAULT) was
affected exactly as much as a through-focus plane, because the error is the
sag, not the propagation.

All three sites now route through WP-A1's shared
`TraceResult.at_exit_vertex()`.

Measured against an independent vector-Snell trace on a plano-convex with
`R2 = -25 mm` over a 20 mm aperture (`repro/TR-SIBLINGS/repro_vertex.py`):
transverse error **476.8 µm → 1.7e-12 µm**, OPD error **3 501 waves →
4.4e-12 waves**; `R2 = -100 mm` 24.59 µm / 820.3 waves → 1.7e-12 µm /
1.2e-11 waves; biconvex ±50 mm 40.57 µm / 568 waves → 1.7e-12 µm /
1.2e-11 waves.  End to end through the public API
(`repro/TR-SIBLINGS/repro_vertex_field.py`), multibranch against the
single-valued traced path at the same plane: curved rear (R2 = -100 mm)
**1.7984 rad rms (0.286 waves, max π) → 0.0003 rad rms (max 0.0049 rad)**;
biconvex ±60 mm 1.8100 → 0.0045 rad rms — both now better than the flat-rear
CONTROL's own 0.0050 rad floor, which is unchanged.

Every physics fixture for this layer used a plano-rear singlet
(`sag ≡ 0`), which is why `pytest -k "multibranch or uniform or ludwig"`
passed 8/8 on the unfixed code.  New curved-rear fixtures:
`tests/unit/test_audit2609_a3_caustic_siblings.py`.

### Added -- `apply_real_lens_traced(caustic='wave')`: the ray-to-wave hand-off (audit §15.9)

A new `caustic` mode that takes the single-valued traced field at the
exit-vertex plane — where geometric optics is exact — and propagates it to
`output_plane_distance` with the library's band-limited angular spectrum.  No
branch enumeration, no KMAH index, no `1/sqrt|J|`, no Ludwig swap and no
dark-side completion: exact through folds, cusps AND the axial point focus
alike, and it carries the exponentially-decaying dark-side tail the branch sum
drops to exactly zero.  This is the commercial-POP pattern (Zemax POP /
CODE V BSP; Goodman §3.10, Matsushima & Shimobaba 2009 for the band limit).

Implemented as ONE recursion with `caustic='single'`,
`output_plane_distance=0` — so every other keyword of the function applies
verbatim — plus one ASM leg at the exit medium's in-medium wavelength.

Measured (f ≈ 25 mm biconvex, N = 512, dx = 4 µm, w₀ = 0.30 mm, BFL =
24.4833 mm): at `d = 0` the result is **bit-identical** to the ordinary traced
call (`np.array_equal` True); at the paraxial focus `P_out/P_in = 0.999999`,
peak 8.5781 and EE(5/10/25 µm) = 0.0408 / 0.1609 / 0.6348, matching an
INDEPENDENT `apply_real_lens` + ASM oracle to four digits (8.5781, 0.0408 /
0.1609 / 0.6348) — where `caustic='multibranch'` returns nothing at all.  Away
from any caustic it tracks the multibranch to 1.7 % over the bright core.

NOT made the default for `output_plane_distance != 0`: the existing
multibranch/uniform tests pin the branch-sum field and switching the default
would move every one of those numbers.  The docstrings recommend it for new
work at any plane the ray map is multi-valued.

### Fixed -- traced lens: a real-dtype `E_in` crashed on the last assembly line (audit T1, P1)

`target_cdtype = E_in.dtype if np.iscomplexobj(E_in) else np.complex128`
selected the numpy scalar TYPE for a real input, which has no `.type`
attribute, so all four `target_cdtype.type(0)` masking sites raised
`AttributeError` — two of them on the BANDED path, the shipped default at
N ≥ 4096 — after the whole ray trace, the three fits and the Newton inversion
had already run.  `apply_real_lens` and `PreparedTracedLens` accepted the same
array.  Now `np.dtype(...)`, resolved once at the top of the function.

Also fixed the related `_reference_input()` cast, which built the carrier
phasor `exp(i k0 W)` in `E_in.dtype`: for a real input that discarded the
imaginary part behind a bare `ComplexWarning` and turned a unit-modulus phasor
into `cos(k0 W)` (measured modulus 1.8e-4 … 1.0).

`repro/orch/verify_trmain2.py` item 1: **AttributeError → complex128**.

### Fixed -- traced lens: `caustic='multibranch'` returned an identically-ZERO field at an axial focus, silently (audit T2, P1)

At and around a paraxial focus every mapped launch triangle either collapses
below `caustic_min_area_ratio` or compresses below one pixel, so the rasteriser
covers nothing and the returned field is exactly zero — reported to the caller
as `P_out/P_in = 0.0000e+00`, 0 non-zero pixels, **0 warnings**.  The mode now
counts the degenerate triangles and REFUSES with a `RuntimeError` naming the
axial point-focus catastrophe and the remedies (`caustic='wave'`, GBD, Maslov),
plus a warning when more than 25 % of the triangles are skipped without a total
collapse.

The energy tripwire is now TWO-SIDED and re-tuned: `_ENERGY_BLOWUP_FACTOR`
10.0 → **2.0** and a new `_ENERGY_COLLAPSE_FACTOR = 0.5`.  The old one-sided
10× test was mute through the whole 4–8× run-up to the focus (measured 3.99 at
z = 24.70 mm and 8.10 at 24.74 mm on a 24.834 mm BFL) and could not see an
energy LOSS at all (measured 0.887 on a resolved fold at N = 4096 against an
independent ASM oracle's 1.000).  `return_diagnostics=True` now also reports
`n_triangles`, `n_triangles_finite`, `n_triangles_degenerate` and
`power_ratio`.

The tripwire's REFERENCE power is also corrected.  It divided the reconstructed
grid power by the input power inside the aperture circle measured on the `E_in`
grid, which is the wrong denominator in two ordinary geometries: the launch
sampler clamps `E_in` at the grid edge, so an aperture wider than the grid
launches the edge amplitude over the whole rim annulus; and light that leaves
the output grid is off-screen, not lost.  On a 6 mm-aperture singlet sampled on
a 48 × 25 µm grid the shipped tripwire read **12.3×** at a plane where the
launched energy is in fact conserved to 0.2 % — a false alarm on the SHIPPED
10× bar, not only on the new 2× one.  The reference is now the power the launch
congruence carries ONTO the grid (`Σ|E_launch|² h²` over alive launch nodes
whose mapped exit point lands inside the grid) against `dx²·Σ|E_out|²`, so
numerator and denominator are the same physical quantity.  On a fixture whose
beam sits inside the grid the two agree to four digits, and every real detection
is unchanged (measured across a 12-plane through-focus scan).  `power_ratio` is
now that ratio, and the messages say "the launched power that reaches this
grid".

`repro/orch/verify_trmain2.py` item 3: **P/P_in = 0, 0 warnings →
RuntimeError naming the cause and the remedy**.

### Fixed -- traced lens: `newton_fit='spline'` + any vignetted ray returned an identically-ZERO field (audit T3, P1)

The launch lattice is a SQUARE of half-width `0.75*aperture`, so its corners
sit at `sqrt(2)*0.75 = 1.06` aperture radii — past any per-surface
`semi_diameter` of `aperture/2`.  Those rays die, their forward-map entries
became NaN, and `RectBivariateSpline` (an interpolating `s = 0` FITPACK fit,
which does NOT ignore NaN) turned ~90 % of its coefficients into NaN, making
`valid = isfinite(opl_map)` all-False and the field zero.  The in-code guard
comment claimed the opposite ("filling dead entries with NaN and extrapolating
with the spline's natural extrapolation"); the only diagnostic was a
100 %-unconverged Newton warning that misdiagnosed the cause and mis-stated the
outcome.

Dead launch nodes are now filled from their NEAREST LIVE neighbour
(`scipy.ndimage.distance_transform_edt`) so the solve is finite, the filled
mask travels in the Newton payload, and any output pixel whose entrance
solution lands on a filled node is masked exactly as an out-of-domain pixel is.
A `RuntimeWarning` names the vignetting and points at `newton_fit='polynomial'`
(the default, whose least squares drops dead samples); a prescription with NO
live launch ray raises.  `_warn_newton_unconverged` now distinguishes "the fit
is NaN everywhere" from "the iteration hit its cap".

`repro/orch/verify_trmain2.py` item 2: spline **P_out/P_in 0.0000 with 0
non-zero pixels → 0.9501 with 17 493**, against the polynomial control's
0.9754 / 21 601.

### Fixed -- traced lens: `fast_analytic_phase=True` raised `AttributeError` on every refracting prescription (audit T4, P1)

`_geometric_lens_phase` called `raytrace._surface_sag_xy`, which lives in
`raytrace.surface` and is not re-exported by the package, so the documented
public kwarg (and GUI checkbox) crashed on every prescription that actually
refracts.  Now imported from the module that defines it.

Three further corrections in the same function:
* NaN outside the conic domain no longer propagates into `delta_phase` (it
  contributed NaN pixels to the returned field with no diagnostic); such points
  now contribute zero from that surface and the count is reported through a
  `RuntimeWarning`.
* The bulk glass piston is accumulated as a float64 scalar and folded modulo
  2π BEFORE it touches the accumulator, so
  `set_default_real_dtype(np.float32)` no longer injects the 1.95e-3 rad
  (λ/3220) float32 ulp of a 2.9e4 rad piston, growing linearly with thickness.
  Measured float32-vs-float64 agreement **< 1e-4 rad** on a 4 mm element.
* The docstring's "under 10 nm OPL on F/10+" claim is replaced with the
  measured thickness scaling: the omitted term is the in-glass ASM leg, so the
  error is LINEAR IN CENTRE THICKNESS — 0.0003 rad max at 1 µm, 0.034 rad at
  100 µm and **0.68 rad (14.1 nm rms / 41.6 nm PV) at 2 mm** on an f/12.1
  biconvex.  Budget ~7 nm rms per mm of glass.  The absence of an aperture mask
  is documented too.

### Fixed -- traced lens: a surface `form_error` was silently cancelled out of the answer (audit T5, P1)

`lumenairy.raytrace.Surface` has no `form_error` field, so `opl_traced` carried
none of it while BOTH analytic legs of
`E_analytic * exp(i(k0 opl - phase_analytic_lens))` did — and the two cancelled.
A figure-error or tolerancing study routed through the traced model returned
the NOMINAL answer, with no warning and no mention in the docstring (the string
`form_error` did not occur in the module).  Measured suppression **254×** on a
250 nm PV astigmatic map.

The screen `phi_i = -k0 (n_after - n_before)_i * form_error_i` — exactly what
`apply_real_lens` applies, same sign convention — is now built once and added
explicitly at the assembly, on both the banded and whole-grid paths and on both
`preserve_input_phase` branches.  Measured against the exact screen over the
bright support: **0.0034 rad rms against a 0.564 rad screen (0.6 %)**, which is
better than the analytic sibling's own 2.6 % (that carries the ASM diffraction
of the figure error).  Pre-fix the traced model reproduced 0.0 % of it.

`caustic='multibranch'`/`'uniform'` REFUSE a `form_error` prescription: they are
a pure ray construction with no analytic leg to re-apply the screen onto.

**Migration note.**  `apply_real_lens_traced` on a prescription carrying
`form_error` now returns a DIFFERENT (correct) field.  Code that relied on the
traced model ignoring `form_error` should drop the key.  The traced docstring
now enumerates every phase-only prescription feature the ray model lacks and
says what happens to each.

### Fixed -- traced lens: `_reverse_prescription` did not negate the aspheric / freeform / tilt / `sag_callable` departures, and mis-paired the thicknesses (audit T6, P1)

Reversal is the reflection `z → -z`, under which `sag → -sag` TERM BY TERM.
The radius flip supplies the conic term's sign; the polynomial aspheric, the
freeform block, the field-frame `tilt` and a `sag_callable` have no radius to
flip and were passed through UNCHANGED — so `inversion_method='backward_trace'`
on any such element traced the wrong surface.  Measured on a 100 mm/plano
N-BK7 singlet with `aspheric_coeffs={4: 1.0e3}` at h = 5 mm: reversed sag
**-2.500031447e-04 m against the correct -2.512531447e-04 m, an error of
1.25e-06 m = 2.13 waves at 588 nm → 0.0e+00 m**.

`validate_prescription` accepts both `len(thicknesses) == len(surfaces)` (each
surface's forward gap, the last being the BFD) and `== len - 1`;
`list(reversed(thicknesses))` is correct only for the second.  Under the first
it made the reversed GLASS gap the forward BFD: forward OPL 7.583992e-03 m
against backward 1.516798e-01 m, and a ray launched at +4.000 mm landing at
+6.576 mm.  The PAIRING is now reversed, not the list, and the caller's
convention is preserved on output.  Measured after: forward/backward OPL agree
to **0.0 / 1.7e-18 / 5.2e-18 m** at h = 0 / 2 / 4 mm under BOTH conventions,
with the back-traced ray returning to its launch height exactly.

`stop_index` is remapped to `len(surfaces) - 1 - i`, `elements` is reversed
(`surfaces_from_prescription` reads it for vignetting), and every other
top-level key is carried through; only `aperture_diameter` used to survive.

### Fixed -- traced lens: `_sample_local_tilts` wrapped the grid boundary and stored the gradient half a pixel off (audit T7, P2)

`np.roll(E, -1, axis=1)` differenced the LAST column against column 0, so on
any field that fills the grid — a plane wave, a top hat, a post-DOE multi-order
field, i.e. exactly what this function exists for — the rim read a tilt
unrelated to the beam (measured L = -0.1175 against a true +0.03, **5× the tilt
itself**), and the shipped σ = 4 px amplitude-weighted smoothing SPREAD it 12
columns / 4.7 % of the grid inward rather than suppressing it.  Separately,
`angle(E[i+1] conj(E[i]))/dx` estimates `dphi/dx` at `i + 1/2` and was stored
at `i`, biasing every launch direction by `dx/(2R)` — coherent across the pupil
and invisible on collimated fixtures.

Both estimators now use the MIDPOINT construction the two sibling estimators in
the same file already use (`_compute_carrier('auto')`,
`_fit_residual_eikonal`), with the half-pixel lookup offset applied PER AXIS.

Measured (`repro/TR-INFRA/p7_tilts.py`): whole-grid `max |L - L0|` on a tilted
plane wave **1.475e-01 → 1.232e-15** (σ = 0) and **8.111e-02 → 2.220e-16**
(σ = 4); mean half-pixel bias on a spherical wave **+2.0000e-05 → +4.5e-21**
at R = 0.10 m (the pre-fix value is `dx/(2R)` to five digits) and
**+4.0000e-05 → -2.6e-22** at R = 0.05 m.

### Fixed -- traced lens: `carrier=<ndarray>` sampled the entrance eikonal by nearest neighbour (audit T7 / T12, P2)

The launch lattice is a `linspace` over ±0.75·aperture with an odd sample
count, so its nodes are NOT wave-grid pixel centres.  The ndarray-carrier
branch looked up both the direction cosines and — the larger of the two, and
the one the in-code note omitted — the H6 entrance eikonal `W` by fancy-index
with `.astype(np.int64)`, i.e. by FLOOR, giving a half-pixel error linear in
`dx` plus a systematic −0.494 px offset.

Now bilinear (`scipy.ndimage.map_coordinates(order=1, mode='nearest')`), the
same cost class.  Measured at the source (`repro/TR-MAIN-1/p6b_ndarray.py`):
eikonal error **301.4 / 150.9 / 69.0 nm rms → 0.269 / 0.066 / 0.015 nm** at
N = 256 / 512 / 1024 (and now scaling as `dx²`), cosine error
4.5e-5 / 2.3e-5 / 9.9e-6 → 2.5e-10 / 6.1e-11 / 1.7e-11.  End to end against an
independent exact-sphere trace on a diverging 200 mm conjugate through an f/32
singlet (`repro/TR-MAIN-1/p6_carrier.py`): `carrier=<ndarray>` **9.914e-02 rad
rms (20.67 nm) → 1.874e-03 rad (0.391 nm)**, i.e. within 1.5 % of the
analytically identical `carrier=<float>`'s 1.845e-03 rad instead of 54× worse.
The `'auto'`, float and `TiltedCarrier` branches are untouched.

The remaining error is now the grid's own: `np.gradient` (central difference of
step `dx`) sampled bilinearly costs `(dx^2/6 + dx^2/8)|d^2g/dx^2|`, and on the
tilted-sphere fixture in `test_niche_d1_tilted_carrier.py` the measured
gradient error 1.7197e-08 sits at 0.957 of that derived
`(7/24) dx^2 max|g''|` bound (nearest-neighbour read 3.324e-04 there).  That
file's fail-before arm, which asserted the branch must be worse than 1e-5, is
restated against the derived bound on a 4x refinement ladder rather than
deleted.

### Fixed -- traced lens: `_multi(reuse_prepared=True)` and `_segmented` contracts (audit T8, P2)

* `apply_real_lens_traced_multi(reuse_prepared=True)` raised an opaque
  `TypeError: prepare_real_lens_traced() got an unexpected keyword argument`
  from three frames down for TWELVE public kwargs (`dy`, `remap_sampling`,
  `on_fit_domain_basis`, `on_pool_memory`, `parallel_amp_min_free_gb`,
  `newton_mask_dilate_coarse_px`, `fast_analytic_phase`,
  `output_plane_distance`, `caustic_ray_subsample`, `caustic_band`,
  `caustic_min_area_ratio`, `origin`) — and only on the DEFAULT reuse path, so
  the same call worked or crashed depending on the carrier kind.  That is
  exactly the failure the v5.29 `_NO_SCREEN` block was written to close; it
  enumerated six of the eighteen by hand.  The accepted set is now DERIVED from
  `inspect.signature` of both functions, so a keyword added to either is
  handled without editing a list, and the refusal names the keyword and
  `reuse_prepared=False`.
* `apply_real_lens_traced_segmented` handed `dy` to the angular partition and
  then called `apply_real_lens_traced(..., dx=dx)` WITHOUT it, so the element's
  square-pixel refusal was never reached: `segmented(dy=2*dx)` returned a field
  where the direct call raises, with the fy axis scaled by `dy` and the ray
  trace, exit grid and aperture mask all assuming `dy = dx`.  It now raises the
  same square-pixel `ValueError` up front.
* The single-segment path forwarded a possibly-SEQUENCE `carriers` as a scalar
  `carrier=`; since the segment count is data-dependent, the same call was
  valid or ill-typed depending on the input spectrum.  A one-element sequence
  is now unwrapped and a longer one refused with the reason.

### Fixed -- traced lens: the exit-NA guard measured rays the output mask deletes (audit T13, P2)

The trace runs with `aperture_diameter` popped, so unless the SURFACES carry a
`semi_diameter` the rays go out to `0.75*aperture` while the returned field is
masked to `aperture/2`; the significance gate looked only at input AMPLITUDE,
which does nothing for a flat / top-hat / wide-Gaussian input.  Measured on an
f/5 fixture with `E_in = ones`: `na_exit` **0.78487 against the true marginal
0.24964 — 3.144× overstated**, which demands `dx ≤ 0.83 µm` instead of 2.62 µm
(a 3× finer grid, 9× the memory) and feeds
`propagate_traced_carrier_chain`'s `on_tilt_exact_grid`, whose default action
is `'error'`.  The gate is now intersected with the same aperture disc the
output mask uses.

### Changed -- traced lens: `parallel_amp_min_free_gb` is a floor, and the gate scales with the field (audit T15, P2)

The 48.0 GB default was applied as a FLAT threshold, which the docstring itself
described as "tuned for the N=32768 complex128 case" — so the measured win was
unreachable on any box with less than 48 GB free at ANY grid size.  Measured at
N = 1024, `ray_subsample=4`, median of 4: `parallel_amp=False` 5.195 s /
151.0 MB tracemalloc peak against `True` 3.839 s / 218.2 MB — **1.35× wall for
+67 MB**, i.e. the guard demanded 48 GB to spend 67 MB (700× over-conservative).
The requirement is now `max(2.0 GB, 6 × E_in.nbytes)`, which is what the
doubled working set actually costs (26.0 vs 18.0 grids of `8N²` measured);
passing `parallel_amp_min_free_gb` explicitly raises the floor to
`max(size-scaled, yours)`.

### Performance -- traced lens: bit-identical kernel and allocation work (audit T9 / T10, P2)

All five items are bit-identical or bitwise-verified; none changes a returned
number.

* **numba Chebyshev kernel blocked.**  Four `np.empty`/`np.zeros` scratch
  arrays were allocated per SAMPLE inside `prange`; they are now hoisted to a
  512-sample block.  Measured over 4 Mpt, 7 interleaved reps, min wall, 20
  numba threads: order 6 (M = 28) **346.4 → 213.4 ms (86.6 → 53.3 ns/pt,
  1.62×)**, order 10 (M = 66) **236.4 → 152.7 ms (1.55×)**, with
  `max|diff| = 0.000e+00` on all three outputs.
* **pure-NumPy Chebyshev fallback chunked** over the query axis against
  `_CHEB_FIT_CHUNK_ENTRIES` — the one large-array site in the module that was
  not row-blocked, and the REQUIRED branch for CuPy.  Measured tracemalloc peak
  at n = 1e6: **1160 MB (145 float64/point) → 120 MB (15.0/point)**, bitwise
  identical; the peak is now O(budget) rather than O(n).
* **`inversion_method='fit'` design matrix chunked.**  It built two
  `(N², order+1)` Chebyshev Vandermondes and an `(N², M)` product unchunked —
  measured 141 full-grid units at N = 1024 (1.18 GB) and 198 at N = 512,
  projecting to ~76 GB at N = 8192.  Bit-identical (the design is pointwise in
  the output pixel).
* **whole-grid final masking in place.**  `np.where` × 2 over `X**2 + Y**2`
  materialised from broadcast VIEWS → `np.copyto(where=)` over the 1-D axes
  form the banded path already used: ~7.1 → ~3.1 units of `8N²` at that stage
  (61 → 27 GB at N = 32768).
* **`_ray_density_self_checks` allocation.**  The complex128 upcast of `E_in`
  purely to take its modulus (3.0× the field's bytes at peak for a complex64
  input, 25.8 GB at N = 32768) is gone, and `(np.abs(E_out)**2).sum()` is now
  `np.vdot` (17.2 GB of transient removed; agreement 2.2e-16 relative).
* **strided `|E_analytic|`.**  On the DEFAULT configuration
  (`preserve_input_phase=True`, `ray_subsample=8`) `amp` is either deleted
  unread or read exactly once as `amp[::sub, ::sub]`; the full-grid modulus is
  now taken only where a full-grid `amp` is actually consumed.  Measured
  14.17 ms / 8.4 MB at N = 1024 and 37.29 ms / 33.6 MB at N = 2048 against
  0.114 / 0.853 ms strided (≈9.6 s and 8.59 GB at N = 32768).

### Performance -- caustic siblings: the rasteriser and the uniform dark fill (audit S9, P2)

* **Multibranch triangle rasteriser: exact bounding boxes and a chunk budget.**
  Triangles were batched by a POWER-OF-TWO bounding box with no bound on
  `n_tri × 2^{2c}`.  At `output_plane_distance = 0` — the DEFAULT — the map is
  near-identity, so a 5×5-pixel triangle was padded to 8×8 and most of the
  barycentric arithmetic was discarded: measured **144 s at N = 2048 and 882 s
  at N = 4096** for the exit-vertex call against **0.85 s** for the same grid
  near focus.  Unchunked, the worst bucket held 1 187 616 triangles at 8×8 =
  0.61 GB per temporary with ~15–18 alive at once — **7.74 GB traced /
  12.2 GB RSS** for one default-argument call, with no warning and no model in
  `lumenairy.memory.estimate_lens_memory`.  Batches now share an EXACT
  `(wx, wy)` box (falling back to the power-of-two grouping above 96 distinct
  shapes) and are capped at `_RASTER_CHUNK_ENTRIES = 4e6` entries.  The
  contribution SET is unchanged; only `np.add.at`'s summation order on a
  multi-branch pixel can move, which this module already documents.
* **Ludwig caustic-band swap vectorised.**  It was a per-pixel Python loop with
  four small NumPy calls and a scalar `scipy.special.airy` per multi-branch
  pixel: measured **+3.22 s over 3 004 pixels at N = 2048 (1073 µs/pixel)** and
  +9.14 s over 12 020 at N = 4096 (761 µs/pixel).  Same selection rule (sort by
  eikonal, take the closest adjacent pair, swap inside the Kravtsov–Orlov
  band), now as one padded-array pass.  The `scipy.special.airy` import is
  hoisted to module scope.
* **Uniform dark fill restricted** to `r_c < r < r_c + 20 l_airy`.  The CFU
  Airy kernel ran on EVERY pixel outside `r_c` (measured 4 193 535 of
  4 194 304 = 100.0 % at N = 2048) although `Ai` is 4e-16 of its ring value by
  20 Airy lengths and is clamped to numerical zero by `_AIRY_ARG_CAP` anyway:
  **25.42 s against the multibranch's 0.85 s at N = 2048** (the saving is
  grid-size specific -- at N = 384 / 512 it is only 1.1-1.2x, because the
  dark fill is a small share of the call there; the returned FIELD is
  unmoved either way, measured 9.7e-27 of peak), 93.1 s against
  11.9 s at N = 4096.

### Fixed -- inverse map: the cache key omitted the flags that change its arithmetic (audit S8, P2)

`build_inverse_map`'s SHA-256 key hashed 12 scalars and the bytes of every
input array, but none of the `_lens_traced` flags that select the branch of the
least-squares solve it runs — and `_det_traced()` reads
`DETERMINISTIC_TRACED_FIT` at CALL time.  So
`traced_flags(DETERMINISTIC_TRACED_FIT=False)`, whose registry entry states
"restores the G = A.T @ A / rhs = A.T @ b route for the traced chain EXACTLY,
bit for bit, and is the fail-before for the whole layer", was defeated by a
cache HIT on the second and every later call in a process: measured
`key(det=True) == key(det=False) → True` with the SAME OBJECT served.  Nine
flags (`DETERMINISTIC_TRACED_FIT`, `LSTSQ_CONDITIONING_STEPDOWN`, the refine
and Gram constants, `_LSTSQ_RESID_MARGIN`) are now in the key.
Re-measured: **`key(det=True) == key(det=False) → False`, no stale object**.

`report_refusal` no longer tells the user a refusal "costs speed, never
accuracy" — the module's own numbers three hundred lines above say the
incumbent carries 6.0e-3…1.08e-2 waves rms of core wavefront error against the
model's ~2e-12 and moves design 121's banner FWHM 3.350 → 3.450 µm with a 0.8 %
peak change.  The GRAM guard additionally records whether its budget came from
an explicit `set_max_ram` or from psutil's LIVE available reading, and says so,
because in the latter case the returned FIELD depends on what else the machine
is doing.

### Fixed -- traced lens: the `__main__`-guard predicate could not see the module body (audit T10, P2)

`_script_has_main_guard` tested "does a top-level `ast.If` whose test mentions
both `__name__` and `'__main__'` exist anywhere", which cannot see the BODY —
the only thing the warning it feeds is about.  The ordinary shape of a real
driver defeats it: `import numpy as np` / a decorative guard / `BIG =
np.zeros((4096, 4096))` / `main()` classified as GUARDED, so the spawn pool ran
and every worker paid the 134 MB — precisely the 22.1 GB/worker failure the
warning exists for.  The predicate now asks whether every top-level statement
other than imports, `def`/`class`, docstrings and literal constant assignments
sits inside a guard.  Two further shapes fixed: an INVERTED guard
(`if __name__ != '__main__': raise SystemExit`) is no longer accepted (a spawn
child's `__name__` is `'__mp_main__'`, so it takes the branch and dies), and
`match __name__: case '__main__':` is now recognised.

`tests/unit/test_fix_newton_pool_memory.py` is updated: the parametrised
classification table now carries the three fixed shapes, and two rows that used
`x = 1` as "unguarded" use a top-level CALL instead (a literal assignment is
not work a spawn child re-runs).

### Changed -- traced lens: the C13 step-down's blind spot is now measured and reported (audit T10, P2)

`LSTSQ_CONDITIONING_STEPDOWN` scores the two candidate solves on
`||b - A x||` over the RETAINED rows, while the Newton loop evaluates the
chosen polynomial over the WHOLE launch square (its iterate is clamped to
`|u| <= 0.999`, not to the fit's disc).  Measured on a 129² lattice with a hard
`r <= 0.5 R` disc, two candidates agreeing IN DISC to 4.2e-18 differ over the
square by 1.99e-08 of peak at order 10 (det-vs-QR) and 1.05e-02 at order 10
(raw normal equations vs QR).  `_solve_lstsq_thread_safe` gains an optional,
diagnostic-only `score_domain=` (the full-lattice design), and a residual TIE
that hides a difference above `_LSTSQ_SCORE_DOMAIN_TOL = 1e-6` of the in-fit
peak now emits a `RuntimeWarning`.  Which candidate is returned is unchanged;
at the shipped `newton_poly_order = 6` the difference measures 3.9e-08 and the
warning stays silent.

### Changed -- traced lens: three doc claims corrected against measurement (audit T11 / T14 / T16, P3)

* `ray_subsample`'s "typically loses < 1 nm of fidelity at 4 or 8" described
  the v5.35 inverse-characteristic evaluator's numbers as if they were the
  coarse upsample's.  With the evaluator ON the coarse lattice costs nothing
  (**0.000 nm rms**, < 5e-17 m, at sub = 4, 8 and 16 alike against an
  independent exact-sphere oracle); with it OFF the order-1 upsample costs
  `(sub·dx)²·f''/8` — **3.40 / 11.63 / 44.18 nm rms** (12.90 / 51.58 /
  206.3 nm max) at sub = 4 / 8 / 16 on an f/7.5 singlet.  The evaluator is off
  for `use_gpu=True`, `inversion_method != 'newton'`, `inverse_map=False` and
  `sag_chunk_rows` banding, and those configurations now emit the accuracy
  notice an internal guard refusal already emitted.
* `_opl_by_backward_trace`'s "~35–40 nm RMS vs Newton on singlets at N=512" is
  not reproducible on any fixture in the tree and no test pins the one it was
  measured on; re-measured on an N-BK7 100/−100 singlet at N = 256 it reads
  **1.93 rad rms (180 nm), max 372 nm** — the 1.81 rad of a uniformly wrapped
  difference, i.e. more than a wave.  The docstring now says so and labels the
  route's exit-phase error unquantified.  Its on-axis reference index
  (`N_c // 2`) is also corrected: the coarse sample nearest x = 0, which
  matters when `ray_subsample` does not divide `N` (at N = 250, sub = 8 the old
  form was 3 fine pixels off axis).
* `on_noncollimated='off'` is documented as a suppression knob, not a cost
  knob: the `else` branch recomputes the same `_input_tilt_stats`, measured
  4.013 s (`'warn'`) against 4.209 s (`'off'`) at N = 1024.
* The order-3 OPL upsample's comment stated the OPPOSITE of the code about
  `prefilter`; it now says what the code does and what that costs at the
  lattice edge (order 3 is 7.09 / 1.90 / 0.51 / 0.04 nm at 0 / 1 / 2 / 4 coarse
  cells of inset against order 1's flat 0.64 nm — better deep inside, worse
  within ~2 cells).  The `mode='nearest'` trailing-band extrapolation is
  documented with its measured 69.2 nm (N = 256) / 140.8 nm (N = 512) and left
  for a follow-up.

### Fixed -- traced lens: `on_noncollimated='delegate'` could swap models in silence (audit T16, P3)

The dropped-kwarg report omitted four physics-affecting knobs
(`newton_amp_mask_rel`, `newton_mask_dilate_coarse_px`, `beam_centre`,
`fast_analytic_phase`), and the emitter was gated on
`if _dropped or carrier is not None` — so a delegating call passing only
defaults changed model with ZERO warnings.  Those four, plus `origin` and the
three `caustic_*` sub-knobs, are now listed, and the model swap is ALWAYS
announced.  The pure POLICY knobs (`on_undersample`, `on_aperture_beam`,
`on_fit_domain_basis`, `on_pool_memory`) and RESOURCE knobs (`n_workers`,
`parallel_amp*`, `use_gpu`, `min_coarse_samples_per_aperture`) are
deliberately NOT listed: none of them changes an answer, and a caller who set
one to keep the output quiet should not be answered with a warning about it.
`return_screen=True` with `on_noncollimated='delegate'` now raises: the
fallback returns `apply_real_lens(E_in)`, i.e. a FIELD containing `E_in`, where
`return_screen` promises an input-independent screen — caching that as a screen
bakes one call's input into every later one.

### Fixed -- caustic siblings: KMAH fill wrap, the dead turning-point counter, and two naming defects (audit S11, P3)

* The KMAH NaN-fill used `np.roll`, making the launch lattice a TORUS: a
  left-edge node's "nearest valid neighbour" was the RIGHT-edge node, up to
  `2·launch_radius` away and possibly on a different sheet.  Now edge-clamped,
  with a no-progress break so a fully dead lattice cannot spin to the cap.
* `_count_interior_turning_points` — written to be "robust to an isolated
  zero-slope SAMPLE at an extremum (which would otherwise double-count a
  `+ → 0 → -` transition)" — was never called; both meridional traces used the
  raw `diff(sign(diff))` form it was written to replace, which misclassifies a
  clean FOLD as a cusp and routes it away from the Airy completion it qualifies
  for.  It is now the counter.
* `_build_pearcey_cusp_field` took a `wavelength` it never read — the shape of
  a missing `k`-scaling.  The parameter is dropped and the λ-freedom explained
  (the scaling is absorbed upstream, where the control coordinates are fitted
  to phases in radians).
* `L0`/`M0` — the input congruence's launch direction cosines — were rebound
  mid-function to per-vertex slowness components, silently shadowing them for
  the rest of the body.  Renamed `p0x`/`p0y` etc.

### Changed -- `_math.chebyshev.chebyshev_fit_2d`: the round-trip claim is now conditional (audit S11, P3)

`normalize_xy=True` fits about the GRID's own midpoint (an affine map
`2(x - x.min())/span - 1`), while
`lumenairy.elements.freeform.surface_sag_chebyshev` evaluates about the AXIS
(a pure scale `T_i(x/norm_x)`), so the emitted coefficients round-trip through
the library's own evaluator — which this function's docstring promised
unconditionally — only for a CENTRED grid.  An off-centre grid whose fit
carries any non-constant term now emits a `RuntimeWarning` naming the offset
and the remedy; a pure `T_0 T_0` fit is shift-invariant and stays silent.

### Changed -- traced lens: three Chebyshev Vandermonde copies collapsed onto the shared module (audit T11, P3)

`lumenairy/_math/chebyshev.py` exists specifically to de-duplicate these (its
header says so, and `lenses.py`, `lenses_maslov.py` and four
`propagators/asymptotic*.py` modules already import from it), while
`_lens_traced.py` and `_lens_imap.py` each kept a private reimplementation.
All three were BITWISE equal, so the duplication bought only three places for a
fix to land in one of.  `_lens_traced._cheb_vand_2d` /
`_cheb_deriv_vand_2d` and `_lens_imap._cheb_dvander` are now thin
backend-resolving wrappers over the shared helpers; bit-identity re-verified
against them and against `numpy.polynomial.chebyshev.chebvander`
(`max|Δ| = 0.0` at orders 0, 1, 6, 8, 12).

### Fixed -- traced lens: a malformed `stop_index` is now diagnosed by the traced entry itself (WP-A2 follow-up)

`apply_real_lens_traced` only tested `int(stop_index) != 0`, so an out-of-range
index -- which matches no surface AND suppresses the entrance aperture, i.e.
silently removes every aperture mask -- reached the analytic amplitude leg and
surfaced, if at all, as an `apply_real_lens:` ValueError raised from inside a
worker thread, after the ray trace had run and after a "non-entrance stop, use
apply_real_lens" warning had already fired; a non-integer produced a bare
`invalid literal for int() with base 10`.  The key is now read through the
shared `_normalise_stop_index` helper, so the traced family refuses under its
own name, before the trace.  Every in-range spelling is bit-identical, negative
indices count from the end as Python does (`-1` now names the surface it
selects, and `-2` on a 2-surface lens -- the entrance -- correctly stops
warning), and a prescription with no `surfaces` key is left to
`validate_prescription`.

### Fixed -- traced lens: the pre-flight aperture-vs-grid notice measures the axis that truncates (WP-A2 follow-up)

The call passed the field's `shape[0]` (Ny) together with `dx`, describing a
semi-extent that exists on neither axis of an anamorphic grid.  On a TALL grid
it reported the WIDE y half-width and stayed silent while the aperture
over-filled x: measured 0 warnings on Nx = 64 / Ny = 256 / dx = dy = 20 µm with
a 2 mm aperture whose x half-width is 0.64 mm.  The traced entry now passes its
own `N_y` / `dy`, and `_warn_if_aperture_exceeds_grid` checks the smaller
semi-extent.  Square grids are unchanged by construction.

---

## VERIFY-A3 follow-up (added 2026-09-12 by the verifier)

### Fixed -- traced lens: the exit-NA undersample guard was priced on the entrance disc (VERIFY-A3 OI-1)

`apply_real_lens_traced`'s exit-NA statistic is intersected with the ENTRANCE
aperture disc (audit T13, which removed a 3.14x overstatement), but the mask the
returned field carries is applied on the OUTPUT grid.  On a thick fast element
those sets differ: measured against an independent Newton+Snell trace at
lambda = 1.0 um, `R = +-20 mm / t = 12 mm / aperture 18 mm` reads 0.49809 on the
entrance disc against 0.85802 on the output disc, so the guard's advice
`dx <= lambda/(2*NA_exit)` said 1.00 um where 0.58 um is required -- 1.72x too
coarse, in the unsafe direction.  The WARNING is now decided on the larger of
the two, and `_exit_na_out` reports `na_exit_entrance_disc`,
`na_exit_output_disc`, `na_exit_guard` and `n_exit` alongside the unchanged
`na_exit` (which `propagators/carrier.py`'s `on_tilt_exact_grid` reads and which
is therefore deliberately not moved).  Thin elements move by <= 5 %
(0.246183 -> 0.257506 on the audit's f/5 fixture).

### Fixed -- traced lens: the exit-NA Nyquist test now uses the IN-MEDIUM wavelength (VERIFY-A3 OI-3)

The exit leg runs in the medium after the last surface, whose wavelength is
`lambda/n_exit`, so a ray at angle theta carries transverse spatial frequency
`n_exit*sin(theta)/lambda`.  The guard compared the bare direction cosine
against `lambda/(2*dx)`, i.e. it told a prescription ENDING IN GLASS that it had
`n_exit` times more room than it has.  Measured on an immersed rear
(`glass_after='N-SF11'`, n = 1.75588, R = +-30 mm, aperture 16 mm,
lambda = 1.0 um): the statistic is `sin(theta) = 0.040931` where the criterion
needs `n_exit*sin(theta) = 0.071870`, so the advised `dx` was **1.76x too
coarse** (12.216 um quoted against the 6.957 um required); an N-BK7 rear reads
5.252 -> 3.484 um.  `power_frac_above_nyquist` is compared in the same units.

**Migration.** `n_exit == 1` on every air-ending prescription, where the test,
the message and the reported fraction are BIT-IDENTICAL to before -- that is
every fixture in the tree.  A prescription ending in glass (a cemented interface
left open, an immersed sensor, a sub-assembly handed to another element) will
now see the undersample `RuntimeWarning` fire where it did not, and will be
asked for a grid `n_exit` times finer.  That is the physically required
sampling; pass `on_undersample='silent'` to acknowledge it deliberately.
The reported `na_exit` / `na_exit_entrance_disc` / `na_exit_output_disc` stay
bare direction cosines, so nothing calibrated against them moves;
`na_exit_guard` is the in-medium numerical aperture and `n_exit` is reported so
the two are convertible.

### Fixed -- caustic siblings: the energy tripwire's gain arm is bracketed against boundary-straddling triangles (VERIFY-A3 OI-2)

The launched-power normaliser counts a launch node only when its own mapped
point lands on the output grid, while the reconstructed power counts every pixel
a triangle covers -- including triangles that straddle the grid boundary with
their nodes outside.  On a coarse launch lattice over a grid much smaller than
the beam that mismatch alone reached 3.26x: measured on the delta-audit's D3
fixture (a 6 mm aperture on a 1.2 mm grid) at `ray_subsample=8`,
z = 100 / 110 / 120 mm gives 3.257 / 2.563 / 2.069 with `n_branch = 1` and ZERO
degenerate triangles -- three spurious `RuntimeWarning`s with no coalescence
anywhere, from the same geometry the launched-power normaliser was introduced to
quieten.  The gain arm now requires the excess to clear a second denominator,
the launch power of the triangles that rasterise onto the grid, which counts a
straddling triangle whole and so bounds the launched power from above.  All four
false positives go silent (bracketed ratios 0.549..0.865) while the real
blow-up is unmoved: 1.803e+05 / 2.331e+05 / 1.291e+05 at 0.98 / 0.99 / 0.995 of
the ray-traced BFL on BOTH denominators, and the audit's silent 4-8x pre-focus
band still warns (3.988 / 8.102 / 13.07).  `return_diagnostics` gains
`power_ratio_triangles`; `power_ratio` is unchanged.

### Changed -- traced lens: `_spectral_gap_cuts` scores its flanking peaks INSIDE the occupied band (audit T11 / VERIFY-A3 OI-4)

The audit's T11 row recorded that the docstring contradicted the code; WP-A3
changed the CODE, restricting the "a real gap has a peak on each side" test to
the 0.995-power support the docstring describes, and that change shipped without
a changelog line.  It is a real behaviour change to
`apply_real_lens_traced_segmented`: spectral leakage OUTSIDE the occupied band
can no longer justify a cut.  Demonstrated on a synthetic marginal with one
in-band lobe at f = +20 and an out-of-band lobe at f = -80, support declared
[-40, +40]: the pre-fix predicate cuts at `[-40.0]` on the strength of the
out-of-band peak, the shipped one returns `[]`; a genuine two-lobe gap is still
found by both (`[0.0]`).  The exact-reconstruction contract is untouched --
`max|sum(segments) - E| / peak = 5.55e-16` at `min_segment_power` 0 and 1e-3.
Pinned by `tests/unit/test_audit2609_a3_verify_traced.py::test_spectral_gap_cuts_only_counts_peaks_inside_the_occupied_band`
and `::test_the_angular_segmentation_still_sums_back_to_the_input`.

**Migration.** A field whose spectrum has structure outside its own 0.995-power
support may now be split into FEWER angular segments.  At the default
`min_segment_power=1e-3` the extra bins were dropped as empty anyway; at
`min_segment_power=0` they cost two extra inverse FFTs and produced empty
segments in the returned list.

### Fixed -- traced lens: `_sample_local_tilts` says when it is at its sampling limit (VERIFY-A3 OI-5)

The estimator reads a WRAPPED phase difference `angle(E[i+1] conj(E[i]))`, so it
cannot return a direction cosine above `lambda/(2*dx)` whatever the field does:
a steeper launch tilt folds into that band and comes back as a plausible small
number.  The audit's own repro prints the case and calls it "silently WRONG, no
warning" (`repro/TR-INFRA/p7_tilts.py` 7c: a 0.8 launch tilt at dx = 4 um,
lambda = 1.31 um, `max_sin=0.5` returns `max|L| = 0.1450` against a 0.16375
Nyquist -- nowhere near the 0.5 clip, which can never fire on that grid).  A
`RuntimeWarning` now fires in two situations: when the `max_sin` clip actually
bites (naming the clipped fraction), and when `max_sin` is beyond what the grid
can carry AND the reading has run up to within 20 % of the fold (naming both
numbers).  Silent below that: measured 0.885 of Nyquist for the aliased case
against 0.183 (tilt 0.03) and 0.611 (tilt 0.10) for ordinary well-sampled
tilts, and silent on a collimated field and under the shipped
`smooth_sigma_px=4`.

### Fixed -- traced lens: a per-surface `semi_diameter` now reaches the DEFAULT (polynomial) answer (VERIFY-A3 OI-11)

`newton_fit='spline'` rejects output pixels whose entrance solution lands on a
node it had to FILL (audit T3), but the default polynomial fit simply dropped the
dead samples from its least squares and was then smooth across the hole -- so a
prescription whose vignetting comes from a per-surface `semi_diameter` returned
the UN-VIGNETTED field.  Measured on an N-SF11 singlet whose rear semi-diameter
is 1.4 mm inside a 5 mm aperture (N = 192, dx = 30 um, lambda = 1.064 um):
`P/P_in = 0.9983` with 21 821 non-zero pixels, identical to the same call with no
`semi_diameter` at all, while the spline path read 0.8657 / 6 637.  Both fits now
mask an output pixel whose converged entrance solution lands on a DEAD launch
node, and the polynomial path reads **0.8840 / 7 093** -- within 2 % of the
spline.  A `RuntimeWarning` names the count, and only when the vignetting is
inside the clear aperture (the launch square's corners sit at 1.06 aperture
radii and die against any `semi_diameter <= aperture/2`, which changes nothing
the output mask keeps: measured `P/P_in` 0.998312 both ways at
`semi_diameter = aperture/2`).

**Migration.** A prescription that vignettes rays with `semi_diameter` /
`clear_aperture` now returns LESS power through `apply_real_lens_traced` -- the
vignetting is in the answer instead of being fitted over.  Nothing moves when no
ray dies (the mask is `None` and the arithmetic is bit-identical), nor when the
dead rays are outside the clear aperture.  The CuPy branch (`use_gpu=True`,
polynomial only) keeps the historical extrapolating behaviour, because the
rejection kernel is NumPy.

### Fixed -- analytic lens: the ndarray carrier differentiates and samples with its OWN pitch on each axis (VERIFY-A3 OI-10)

`_compute_carrier`'s `carrier=<ndarray>` / `conjugate=<ndarray>` branch used
`np.gradient(W, dx, dx)` and a `/dx` lookup index on BOTH axes, and took the
sample count from `X.shape[0]` for both.  `apply_real_lens_traced` refuses a
non-square grid, but `apply_real_lens` supports `dy != dx` and forwards the
carrier straight through, so on an anamorphic grid the y eikonal gradient was
wrong by `dy/dx`: measured against the closed-form tilted congruence
(R = -30 mm, L = 0.046, M = 0.031) at dy = 3 dx, the y gradient error was
**1.051e-01** against a derived discretisation bar of 1.198e-07, and at
dy = 0.4 dx **1.936e-02** against 5.622e-09.  The branch now takes `dy=None`
(defaulting to `dx`) and `apply_real_lens`'s six call sites pass their own.
Bit-identical for `dy is None` and `dy == dx`, i.e. on every square grid and
every path through `apply_real_lens_traced`.

### Changed -- tests: two resource `pytest.skip`s replaced by asserted preconditions (VERIFY-A3 OI-9)

`tests/unit/test_niche_k4_uniform_caustic.py` skipped its two
`caustic_fold_ref` gates below `8*N*N*16 + 1.5` GB of free memory, which
`docs/TESTING_STANDARDS.md` rule 4 forbids ("two skips silently removed five
tests from the gate on exactly the runners that mattered").  The bound was also
7x too conservative: the MEASURED tracemalloc peak of the N = 768 uniform call
is 0.229 GB (24 grid-units of `16 N^2`) against the 1.58 GB the skip demanded.
Both sites now assert that requirement (doubled for headroom) and fail loudly
with the number.
