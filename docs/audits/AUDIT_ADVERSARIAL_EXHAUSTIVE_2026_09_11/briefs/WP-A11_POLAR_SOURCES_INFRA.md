# WP-A11 — Polarization, coatings, Berreman, sources, algebra, infrastructure

Read first: `COMMON.md`, then the partition report `POLAR-SOURCES-INFRA.md`, the `ORCHESTRATOR.md` row on
`coating_reflectance(polarization='te')`, and report section §10 (rows Z1–Z4) plus §15 in
`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11.md`. Repro: `repro/POLAR-SOURCES-INFRA/`.

## Files you own
`lumenairy/elements/coatings.py` and its JAX twin (find it: `grep -rln "coating" lumenairy/elements/*jax*`),
`lumenairy/elements/berreman*.py` (only if a finding needs it), `lumenairy/sources/*.py`, `lumenairy/memory.py`,
`lumenairy/algebra/*.py`, `lumenairy/polarization.py`, `lumenairy/_deprecation.py`, `lumenairy/_validation.py`,
`lumenairy/cache.py`, `lumenairy/user_library.py`, `lumenairy/_context.py`. NOT `lumenairy/__init__.py` (the
`import lumenairy` cost and lazy loading belong to a later WP — describe what you need), NOT `propagators/fft_infra.py`.
Tests: the corresponding test files and new `tests/unit/test_audit2609_a11_*.py`.

## Findings to implement
- **Z1 (P1 ✔)** `coating_reflectance` (and the JAX twin) tests `pol == 's'` and sends everything else — `'te'`, `'S'`,
  `'P'`, `'tm'`, junk — down the p branch, while CONVENTIONS §7 promises case-insensitive `te`/`tm`/`s`/`p` everywhere:
  route through the RCWA `_normalize_pol` (or a shared helper) and raise (§2 prefix) on junk; table-test all eight
  spellings; the same for every other `polarization=` consumer in your files.
- **Z2 (P1)** Gaussian–Schell / Schell sources realise the PERIODISED coherence kernel (FFT filtering on the grid):
  zero-pad ≥ 4σ_g before the filter (or implement Gori's pseudo-mode representation), warn for σ_g > L/6, correct the
  "σ_g ≫ w0 approaches the coherent limit" docstring; verify the two-point kernel against the documented Gaussian over
  ≥ 10⁴ realisations with a derived bar (the auditor matched the wrapped Gaussian to 0.002).
- **Z3 (P2)** `estimate_lens_memory(lens_model='real')` under-predicts 1.6–2.8× — re-derive from tracemalloc on
  `apply_real_lens` (note WP-A2 is reducing its peak; measure at the end of your work and say which revision you
  measured); `create_gaussian_beam` without a dense meshgrid and with in-place normalisation; `FreeSpace(method='auto')`
  warning spam and the pitch/ABCD contradiction; `stokes_parameters` / `degree_of_polarization` memory.
- **Z4 (P3)** `deprecated_alias` stacklevel; `lumenairy.algebra.from_prescription` module-vs-function shadowing;
  `_check_2d_scalar_field` accepting object dtype / `np.matrix`; cache-registry name collisions and swallowed clearer
  failures; `_plane_wave_carrier` centre `N/2`; `JonesField.propagate*` docs vs in-place mutation; `user_library`'s
  unbounded `Pow`.

## Verification specifics
- The audit verified correct: Jones matrices to 1e-16, retardance sign and the QWP@+45° → S3 = −1 row,
  `S3 = −2 Im(Ex conj Ey)`, Mueller forms, coatings TMM vs Airy 1e-16 incl. absorbing exit media and TIR, r_p sign
  consistency coatings↔Berreman, Berreman isotropic reduction / energy / stability / JAX parity, Gaussian/HG/LG
  normalisation and orthogonality, `dy` threading in all 12 source factories, algebra ABCD vs `system_abcd`, `cache.py`
  byte accounting, `user_library` sandbox — keep all of it bit-identical unless a finding changes it.
