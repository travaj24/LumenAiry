# Storage + Codegen Audit — 2026-07-09

Scope: the final unaudited ground — `io/storage.py` (1,604, HDF5/zarr
field & multi-plane persistence) and `io/codegen.py` (1,009,
prescription → simulation-script emission).  **This tranche completes
the library-wide audit campaign.**  Read-only.

---

## 1. `io/storage.py` — verdict: clean

Deep-read: `save_field_h5`/`load_field_h5`,
`save_planes_h5`/`load_planes_h5`, `append_plane_h5` (the
concurrency-critical path).  Structurally covered: the zarr twins
(`_zarr_*`, mirroring the H5 semantics), `TempFieldStore`,
`save_jones_field_h5`, the backend dispatch
(`set_storage_backend`/`_detect_backend`), and `replay_run`.

Verified:

* **Round-trip fidelity** — `preserve_dtype` keeps complex64/128
  through save→load; per-plane `wavelength` survives (and overrides the
  file-level value on load); `dx`/`dy` defaulting; `_decode_attr`
  bytes→str normalisation; the v4.15.0 writer-version stamps at file,
  group, AND per-plane level (multi-version appends resolvable).
* **`append_plane_h5` concurrency** — the v4.14.3 reserve-slot-then-
  create atomicity (bump `n_planes` BEFORE dataset create, roll back on
  failure; the loader tolerates reserved-but-absent slots), the v4.16.0
  cross-process `filelock` serialisation (acquired BEFORE the HDF5
  open, clear TimeoutError with the lock path for crashed-holder
  recovery), and the SWMR reader mode (schema created before
  `swmr_mode=True`, `libver` gating) — a genuinely careful design; no
  gap found.
* Load paths raise actionable KeyErrors on wrong-format files
  (single-field vs multi-plane cross-pointers).

Nit: `save_planes_h5`/`append_plane_h5` write arbitrary per-plane dict
values into HDF5 attrs; a `None` value raises deep inside h5py rather
than at a validation boundary.

## 2. `io/codegen.py` — one finding

Deep-read: `_decompose_prescription` (the semantic core).
Structurally covered: the `_generate_unrolled`/`_generate_system_style`
emitters and the zmx/txt wrappers (string templating over the verified
step list).

Verified: the S2a stop-emission logic (per-element `is_stop` plus the
`stop_index`-among-refracting-surfaces compatibility translation —
the counting re-derived, correct); contiguous lens-group collection via
`glass_after != air`; the group-scan stop check (correctly tests the
group's last surface before breaking); `abs(thickness)` unfolding on
mirror legs; consecutive-propagate merging.  This module is also the
**producer** of the `'doe_placeholder'` steps whose silent *consumption*
skip was SY-1 (`AUDIT_PROPAGATORS_KERNELS_2026_07_07.md`) — the
producer side at least attaches the coefficients and a comment.

### CG-1 (P4) — lens-group and mirror step emission silently drops Q-type freeform keys and mirror aspherics
`_decompose_prescription`'s lens-group builder copies exactly
`radius`/`conic`/`aspheric_coeffs`/`glass_before`/`glass_after` per
surface — the `freeform_type`/`q_bfs_coeffs`/`q_con_coeffs`/`r_max`
keys that the Zemax loader (its only elements-producing feeder)
attaches to QBFS/QCON elements (P1-NEW-E) are dropped, so a Q-type
surface degrades to its base conic in the generated script with no
diagnostic.  Likewise the `'mirror'` step carries `radius`/`conic`
but not the mirror's `aspheric_coeffs` — an aspherized mirror (which
the v5.17.1 P2-20 fix taught the `.zmx` *exporter* to both emit and
warn about) silently flattens to base conic here.  Same
silent-shape-degradation class as P2-20/P3-43, unmirrored into
codegen.  **Fix**: forward the freeform keys into the group
prescription (the generated `apply_real_lens` consumer understands
them via `surface_sag_freeform`), emit mirror aspherics, and warn on
anything that cannot be represented.

## 3. Campaign closure

With this tranche the audit campaign has line- or deep-audited the
entire library: `sources/`, `analysis/`, `propagators/` (incl.
asymptotic seams), `raytrace/`, `elements/` (PMM, RCWA, Berreman, EMT,
BOR, lenses + `_lens_*`, DOE/grating/freeform, coatings, polarization,
Maslov, EME, BSDF, segment-geometry), `glass.py`, `optimize/` (all ten
modules), and `io/` (Zemax/Code-V/QUADOA loaders, builders, transforms,
storage, codegen).  Nineteen audit documents.

**Cross-document actionable queue (severity-ordered)**: RT-4 (world
tilt sign) + ZX-1 (coord-break DISZ) — together restore a correct
folded-design path; DS-1 (Source.propagate tuple), SY-1
(doe_placeholder skip), SRC-1 (dense MCF conjugate), OPT-1 (JAX Strehl
merit direction), OPT-2 (ToleranceAwareMerit sub-context), BSDF-1
(Gaussian scatter Jacobian), RT-5 (fan/chief EP mismatch), AN-1
(depth-of-focus factor 2), HF-1 (float(tuple) crash), AN-4 (vertex
launch), ZX-3 (dropped is_stop), CV-1 (Code-V fold drop), DOE-1 (FITS
phase round-trip), GL-1/GL-2, COAT-1, MSL-1, ZX-4, RT-6/7/8/9, CG-1.

The campaign-wide taxonomy held to the end: physics kernels
(~60k lines of them) are essentially flawless; every actionable defect
lives at integration seams, unmirrored fixes, dead parameters, or
silent-degradation paths.

---

*Audit performed single-context against lumenairy v5.21, 2026-07-09.
Final document of the v5.21 audit campaign; companion docs listed in
§3 of each tranche doc.*
