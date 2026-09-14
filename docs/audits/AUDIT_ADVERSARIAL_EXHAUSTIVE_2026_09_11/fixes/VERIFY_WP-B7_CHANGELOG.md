# VERIFY-WP-B7 changelog text

Release 5.47.0, alongside WP-B7.  Findings of the independent re-verification
of WP-B7 (`AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11`); full evidence in
`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/VERIFY_WP-B7.md`.

---

### Fixed -- `aberration_tensor`'s image-plane-waist cache could serve one evaluator's width for another's

`_measure_image_plane_waist` memoises its coarse probe, and the `propagate`
half of the cache key was the callable's `__qualname__`.  That is not an
identity, and the case it was added for -- "a caller may hand in a different
evaluator" -- is exactly the case it fails: two closures made by one factory
share `factory.<locals>.propagate`, two lambdas share `<lambda>`, and a
`functools.partial` has no `__qualname__` at all, so the key degraded to a
`repr` carrying a memory address (the `id()`-reuse hazard the fit fingerprint
beside it exists to avoid).  `lumenairy/_cache_registry.py` documents the same
collision shape as previously measured in this library.

MEASURED before the fix, on two evaluators built by one factory: the cache
returned `w_o = 1.4163857683404920e-04` for the second where its true answer
was `None` -- a different verdict, not a different last bit.

The key now carries the evaluator's OBJECT identity and each entry stores the
callable beside its value, so the id cannot be reused while the entry lives.
The cost is 64 references at the cache bound; a caller that rebuilds an
equivalent wrapper on every call simply never hits (correct, just uncached),
and the shipped caller -- which passes the module-level
`propagate_modal_asymptotic` -- always does.

* `lumenairy/propagators/asymptotic_aberration_tensor.py` (`_callable_identity`,
  the key, `_w_o_cache_put`).
* No shipped answer changes: `aberration_tensor` passed the module-level
  function before and after.
* Tests: `tests/unit/test_audit2609_b7_asymptotic.py`
  `test_verify_b7_the_waist_cache_tells_two_evaluators_of_one_name_apart`.

### Fixed -- the image-plane-waist cache ignored the Newton-stop A/B seam, making a measurement of that seam order-dependent

`_NEWTON_SCALE_RELATIVE_STOP` changes what `propagate_modal_asymptotic`
returns, therefore what the waist probe measures.  It was not in the cache key,
so warming the cache with the seam off and then flipping it returned the stale
width -- and an A/B measurement of the seam through `aberration_tensor` gave a
different answer depending on the order the two arms were run in.

MEASURED: `w_o` 1.4163857683404920e-04 (stop off) against
1.4163857683413807e-04 (stop on), 6.3e-11 relative; with the cache warm the
second arm returned the first arm's number.

* `lumenairy/propagators/asymptotic_aberration_tensor.py` (`_propagate_seams`,
  the key).  The docstring states that a future process-global the evaluator
  reads must join that tuple or drain the cache.
* Tests: `test_verify_b7_the_waist_cache_key_carries_the_newton_stop_seam`,
  two-sided.

### Fixed -- `decompose_lg(only=)` dropped a mode outside its `(p_max, ell_max)` rectangle in silence

`only=` selects from the rectangle the enumeration walks, so a requested
`(p, ell)` outside it was never built and simply vanished from the returned
dict -- and `aberration_tensor`, which fills `L` from
`overlaps.get(k_out, 0)`, would have written a structural zero into the tensor
for a mode the caller asked for.  The docstring said the modes "must lie inside
the rectangle" and nothing enforced it.  It now raises `ValueError` naming the
offending modes.

* `lumenairy/propagators/asymptotic_modes.py` (`_lg_mode_conj_stack`,
  `decompose_lg` docstring).
* No shipped call site can reach it: `aberration_tensor` derives `p_max` and
  `ell_max` from `output_modes` itself.
* Tests: `test_verify_b7_decompose_lg_refuses_a_mode_outside_its_rectangle`,
  two-sided.

### Fixed -- the `next_fast_len` half of the GBD FFT reconstruction had no gate

Reverting `_fftconv_same`'s two `_fft_len(...)` calls to the naive
`Ny + Gy - 1` left all twenty of WP-B7's ids green, so the padding the S9 entry
credits with 1.45x of the transform win at N = 512 was unpinned.  A pin was
added, as an integer property rather than a clock: the chosen length is
5-smooth, never shorter than the true linear length, idempotent and only a
small bump above the awkward `3N - 2`; `_fftconv_same` is counted asking for it
once per axis; and the `'same'` slice is checked against a shift-and-add
convolution written out in the test.

* `tests/unit/test_audit2609_b7_asymptotic.py`
  (`test_verify_b7_the_fft_transform_length_is_the_5_smooth_one`).  No library
  change.

### Documentation -- two derivations beside shipped bars re-measured

Both bars are UNCHANGED; what changed is that the numbers beside them now
reproduce.

* `_NA_MEAN_MIN_FRACTION`'s comment said the two chart-sizing rules "differ by
  2.3 %" at the bar; at the shipped `m = 0.1 sigma_0` the difference is
  **2.832 %** (2.3 % corresponds to a bar of 0.0785).  The comment now carries
  the closed form, a third chart's floor ladder (largest reading with no launch
  direction 1.1e-02, smallest with one 4.2e-01), and the one exception the
  ladder found: a field whose power sits AT the grid's Nyquist frequency -- a
  pi-phase checkerboard -- reads 5.5e-01 with no launch direction, because
  `fftfreq`'s unpaired `-1/(2 dx)` column carries all of it.  The consequence is
  bounded and one-signed: `m + 3 sqrt(s0^2 - m^2)` exceeds `3 s0` for every
  `m/s0 < 0.6` and peaks at `sqrt(10)/3`, so a false positive can only make the
  chart up to 5.4 % WIDER, never narrower.
* `_K1_DERIV_RESIDUAL_MAX`'s comment implied a margin of at least 2.1x below
  the bar.  On a third chart (f = 5.76 mm N-SF6 at 633 nm, 19 inputs) the
  decision is still right two-sided, but the last input where engaging still
  wins -- a hard edge at 0.80 of the pupil on a CONVERGING carrier, fidelity
  0.203 -> 0.998 -- scores 8.99e-01, i.e. **1.33x** under the bar.  Recorded and
  dated; the bar is not moved.
* `lumenairy/elements/lenses_maslov.py`, comments only (its history
  fingerprints are unchanged, which is the gate working: they drop comments).

### Verified -- WP-B7's own claims, independently

Re-derived on fixtures WP-B7 did not use, against oracles written for the
verification (exact conic raytrace with explicit Sellmeier dispersion, agreeing
with the library's index to 0.0e+00; brute-force Rayleigh-Sommerfeld; a
closed-form paraxial ABCD control):

* **byte identity** archive-to-archive on 50 arrays of my own -- 47 identical,
  and the three that differ are exactly the tilted / converging rows items 9
  and 11 are entitled to move.  Includes a decentred non-square raster, a
  decentred source point and pupil centre, complex64, `local_quadrature` and
  `fold_split`;
* **no default moved**: `fga.py` and `lenses_gbd.py` are AST-identical
  pre-vs-post with docstrings stripped; `_universal_route` gives the same
  decision on all 30 cells of an independent NA sweep; no signature default and
  no module constant moved in any of the nine owned modules;
* **item 9** takes the chief-ray landing error from -4.64 % to **+0.45 %** at
  0.25 / 0.5 / 1.0 x the lens NA on an f = 4.94 mm N-LAK22 singlet, and from
  -57.30 % to -7.79 % on an off-axis converging input.  The term is first-order
  complete.  The residual is NOT second order as the report states: it is the
  input AMPLITUDE, still sampled at the output pixel, and it has the closed
  form `-B(A + dC)/d` -- zero at the image plane, growing linearly with the
  readout defocus (+0.44 % predicted, +0.45 % measured);
* **item 11** takes `stationary_phase` / `local_quadrature` fidelity from
  0.000/0.000 to **0.895/0.989** at twice the lens NA and **0.667/0.773** at
  four times it on an independent chart -- more than the report claims -- with
  every previously-engaging case still engaging;
* **item 8** survives an anamorphic diagonal `Q` at aspect ratios up to 1e6
  (relL2 5e-16 .. 1e-15 against both the 9-sigma windowed sum and the true
  dense sum) and a sub-pixel `R_cut` (the half-width clamps to 1, never 0);
* **item 6**'s "the three copies are not the same arithmetic" is understated:
  on a different fit NOTHING is bit-equal and the two Newtons differ by
  2.71e-11 against the report's 4.8e-15.
