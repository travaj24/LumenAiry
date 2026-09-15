# WP-A10 — I/O and optimisation (`lumenairy/io/`, `lumenairy/optimize/`)

Read first: `COMMON.md`, then the partition report `IO-OPTIMIZE.md`, the `ORCHESTRATOR.md` row on CODE V `DIM M`,
and report section §7 (rows I1–I8) plus §15 in `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11.md`.
Repro: `repro/IO-OPTIMIZE/` (large HDF5 outputs were not copied; the scripts regenerate them).

## Files you own
`lumenairy/io/*.py`, `lumenairy/optimize/*.py`, the CODE V / Zemax / catalogue fixtures under `tests/` and
`validation/` that encode a wrong convention or value (fix them and say exactly which), `io/prescriptions_builders.py`
(`THORLABS_CATALOG`). Tests: io/optimize test files and new `tests/unit/test_audit2609_a10_*.py`.

## Findings to implement
- **I1 (P0 ✔)** CODE V `DIM` units: `M` = millimetres, `C` = centimetres, `I` = inches (`{'M': 1e-3, 'C': 1e-2,
  'I': 0.0254}`), warn on unknown tokens, fix the WRITER default so exported files are right for CODE V, fix the repo
  fixture that pins the wrong convention, add a migration note (files previously written by this library carry `DIM M`
  meaning metres — decide how a reader can tell, e.g. a comment line the writer emits, and document it).
- **I2 (P1)** the Zemax loader drops powered non-glass surfaces outside the glass span (PARAXIAL, air-spaced EVENASPH
  phase plates, their STOP): admit every SURFTYPE with power, or warn per excluded surface with non-zero CURV/PARM.
- **I3 (P1)** CODE V loader: `REFL`/`RMD REFL` mirrors, `K` conic, `A`…`J` aspheric coefficients, `radius=None` default.
- **I4 (P1)** exporters drop `radius_y` / `conic_y` / `aspheric_coeffs_y`: emit BICONICX (Zemax) / YTO+CUY (CODE V) or warn
  as loudly as `_warn_dropped_qtype`.
- **I5 (P1, security)** `codegen` interpolates untrusted `.zmx` strings into code positions: `repr()` every string and
  validate tokens at load; add a test with a hostile GLAS token.
- **I6 (P1)** Zarr `append_plane` metadata through the type-tagged codec `_meta_dumps` (ndarrays stringified/truncated
  today); `THORLABS_CATALOG['LA1509-C']` is a 200 mm lens (R1 = 103.29 mm) under the 100 mm part number — correct it
  (true R = 51.5 mm per the Thorlabs spec; verify EFL = 100 mm at 587.6 nm with N-BK7 via `system_abcd`), fix the
  `LA1509_C.zmx` fixture, and audit the other catalogue entries against their EFLs.
- **I7 (P2)** `scale_prescription` for Forbes-Q freeforms / diffractive payloads / stored BFL; `codegen` bare `inf`/`nan`
  and the −inf sign flip; `generate_simulation_script` on `normalize_prescription` output; Zemax `MNUM`/`MCON` warnings;
  the HDF5 default gzip (24× write time for 5.6 % on complex fields) — measure and pick a default (e.g. no compression or
  `lzf` for complex fields) with 16 MiB → sane chunks; the 31 through-focus propagations per FD probe in the wave-leg
  merit (97 % of the work) — add a cheap/cached mode; `design_optimize_multi_objective` NaN Pareto front; unknown-glass
  warnings in the CODE V loader; an edge-thickness constraint (new feature); `elements`/`all_thicknesses` emitted by the
  CODE V / Quadoa loaders as three docstrings promise.
- **I8 (P3)** `.zmx` record injection via `NAME`/`COMM` newlines; UTF-16-BE detection; TOROIDAL/BICONICX → `radius_y`;
  `export_quadoa_qos` stop invention; `create_zoom_configs` glass thicknesses; `method='newton'` bounds warning; `x0`
  bounds check; ndarray aperture cache hash key.

## Verification specifics
- Round-trip tests (load → export → load) for CODE V and Zemax on fixtures with mirrors, conics, aspheres, biconics.
- The audit verified correct: EVENASPH mapping, UNIT handling, UTF-16-LE, semi-diameter ×2, COORDBRK PARM → decenter/tilt,
  CB DISZ folding, STOP-on-CB, `system_abcd` vs paraxial trace, `scale_prescription` core identities, multi-process HDF5
  append, HDF5 metadata codec, numba/NumPy merit parity, `jax.grad` vs FD, no eval/exec on load paths — keep them.
