# VERIFY-A16 changelog text (verifier's additions to WP-A16)

### Fixed -- `sag_dtype`: one rule for all three spellings (VERIFY-A16)

`apply_real_lens` / `prepare_real_lens` (and the traced entry points through them) resolved an
unrecognised `sag_dtype` to float64 in silence, while `set_lens_sag_dtype` and the new
`LensResources.sag_dtype` refused it. `sag_dtype=np.float16` returned the float64 field byte for byte
with no warning -- a setting discarded without a word. The shared resolver in
`lumenairy/elements/_lens_real.py` now refuses anything but `None` / `np.float64` / `np.float32`, with the
`CONVENTIONS.md` section 2 prefix and the caller's own name. `None`, `np.float64`, `'float64'` and `'f4'`
are unchanged and byte-identical; only previously-ignored values now raise. Test:
`tests/unit/test_audit2609_a16_verify_config_and_arch.py::test_all_three_spellings_of_sag_dtype_refuse_the_same_set`.

### Fixed -- lens configuration objects compare by value with an array-valued field (VERIFY-A16)

`LensGeometry.carrier` may hold a wavefront ndarray, but the dataclass-generated `__eq__` compared field
tuples and raised `ValueError: The truth value of an array ... is ambiguous` for exactly those values --
so the documented round trip `LensConfig.from_kwargs(**cfg.to_kwargs()) == cfg`, `narrowed_to()`'s
idempotence and `pickle` equality all raised. The three dataclasses now compare field by field through the
same `_same()` the resolver uses (`np.array_equal` for arrays) and hash by the field tuple as before; an
array-valued instance is unhashable and says so. Test: `...::test_value_equality_survives_an_array_valued_field`.

### Changed -- the WP-A16 import-time figure, restated for the release tree (VERIFY-A16)

WP-A16's text measured `import lumenairy` 745.0 -> 673.8 ms (-71 ms) at its own commit, when
`propagators/fft_infra.py` still imported `scipy.fft` eagerly and paid the shared SciPy prefix. After
WP-A22 made that import lazy too, the two deferrals WP-A16 introduced (`lumenairy.backend.scipy`,
`scipy.special.airy`) are worth **+360 to +407 ms** on the release tree: `import lumenairy` measures
**~294 ms** against **~654 ms** with them eager (interleaved same-build A/B, medians of 7 pairs, two runs).
`scipy.fft`, `scipy.special` and `scipy.linalg` are absent from `sys.modules` after `import lumenairy`
until first use; the structural fact is what the test asserts, no timing is asserted.

### Fixed -- a Ludwig-fold test called the function with its arguments scrambled (VERIFY-A16)

`tests/unit/test_audit2609_a16_lens_arch.py` called `ludwig_fold(a, a, s, s, k)` against the signature
`ludwig_fold(k, S_plus, S_minus, A_plus, A_minus)`, so its "finite on the fold" assertion measured nothing
of the kind. Replaced by two tests with derived bars: the on-fold limit against the closed form
`sqrt(2 pi) k^(1/6) e^{i pi/4} e^{ikS0} (-i sqrt 2 A0) Ai(0)` (measured 4.4e-4 relative at a 1e-12 m
half-separation, bar 1e-2; plain branch sum 7.2x larger, bar 4x), and the far-fold reduction to the plain
branch sum (1.1e-3 at 20 wavelengths, 4.4e-5 at 500).

### Fixed -- `lens_config._VOCAB_CACHE` has its companion lock (full-run follow-up)

The repository's cache/lock pin (`tests/unit/test_v4_14_2_dispatcher_pin_cache_locks.py`) requires every
module-level cache to have a `_<NAME>_LOCK` beside it; the vocabulary cache enrolled with the registry
above had none.  `lumenairy/elements/lens_config.py` now holds `_VOCAB_CACHE_LOCK`, and the four-key fill
in `_vocab` and the clear in `clear_lens_config_vocabulary_cache` run under it, so a concurrent reader
sees either none or all of the borrowed tuples.  125 ids (the lock pin, the A16 files, the relocation
checker's lens_config arm) pass.
