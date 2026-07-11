# Zemax I/O Audit — 2026-07-08

Scope: full line-level read of `io/prescriptions_zemax.py` (1,827 —
`load_zemax_zmx`, `load_zemax_prescription_data_txt`,
`export_zemax_lens_data`, `export_zemax_zmx` /
`_export_zemax_zmx_full`), plus the `surfaces_from_prescription` /
`trace_world` consumer seams needed to judge what the loaded dict
actually drives.  Chosen as the follow-on to RT-4 (world-path tilt
signs): this file is the source of every coord-break, thickness, and
glass datum both trace paths consume.  Read-only single-context pass.

---

## 1. Verified clean

* Encoding ladder (UTF-16-LE → UTF-8 → latin-1, `SURF` probe, BOM
  strip); unit map incl. the 4.9 `INCH`/`INCHES` fix; CURV read takes
  only the curvature token (pickup-solve fields ignored).
* **EVENASPH coefficient mapping re-derived**: Zemax PARM n = αₙ on
  r^(2n), loader `power = 2·parm_num` with
  `parm_val / unit_scale^(power−1)` — dimensionally exact
  (α_m = α_mm·(10³)^(power−1) for mm files).  The exporters'
  `parm_idx = power//2` + `coeff·(10³)^(1−power)` is the exact
  inverse — load→export round-trips even-asphere coefficients
  identically.
* Q-type (QBFS/QCON) branch: PARM 0 → r_max (DIAM fallback),
  coefficients × unit_scale (sag units), dense Forbes indexing.
* P2-19 unknown-SURFTYPE fail-loud (PARM table dropped with a per-
  surface warning instead of being mis-read as aspheres) — in place
  in the .zmx loader; the .txt loader's P3-43 sibling warning too.
* P3-42 terminal-mirror auto-range fix present in **both** loaders;
  single-terminal-mirror prescriptions accepted.
* Glass threading (`medium_between`) correct for singlets/doublets;
  unknown-glass warning consults GLASS_REGISTRY.
* `object_distance` convention (STOP→first-lens DISZ sum over the raw
  list, INFINITY→0) implemented as documented, in both loaders.
* v4.11.2 thickness convention: canonical lists are Zemax-signed;
  the exporter's old double-negation on mirror legs is verifiably
  gone (mirror DISZ round-trips).
* P2-20: Q-type freeform keys warn LOUDLY on export in both writers;
  aspherized mirrors now emit TYPE EVENASPH + PARM rows.
* P3-41 malformed-line wrap (file/line/token in the error).
* Full-writer coord-break interleave keyed on `surf_num` matches the
  loader's ordering (unique surf_nums make the tie rule moot).

---

## 2. Findings

### ZX-1 (P3) — COORDBRK DISZ is dropped from `thicknesses` / `all_thicknesses`, and the in-code comment claims the opposite
`thicknesses` is built from `lens_surfaces[i]['thickness']`, where
`lens_surfaces` descends from `optical_surfaces` — the list with
**coord breaks already filtered out**.  A COORDBRK sitting between
two lens surfaces with non-zero DISZ (common in decentered/tilted
assemblies, where propagation distance often rides on the break)
therefore loses that axial gap in `'thicknesses'`,
`'all_thicknesses'`, and everything downstream of the flat
prescription: `apply_real_lens`, `surfaces_from_prescription` →
`trace()`, `system_abcd`, `seidel_coefficients` — every axial
position after the break is silently shifted.  The comment above the
coord-break block ("The loader preserves COORDBRK DISZ thickness via
the usual ``all_thicknesses`` path (unchanged here)") is **false**.
The gap survives only in `coord_breaks[i]['thickness_m']` — so
`world_surfaces_from_prescription` (which advances the origin by it)
and the full `.zmx` re-export (which writes it back onto the CB row)
are unaffected, and load→export round-trips are faithful.  **Fix**:
fold each CB's DISZ into the preceding element's gap when building
the flat lists (the geometrically-correct collapse for the unfolded
approximation), or at minimum warn when a non-zero CB DISZ is being
dropped and fix the comment.

### ZX-2 (P3, synthesis with RT-4) — folded .zmx designs currently have no correct trace path
The local path ignores folds entirely: `surfaces_from_prescription`
never consumes `prescription['coord_breaks']` (the `trace()` loop's
`is_coordbrk` machinery is reachable only from hand-built Surface
lists), and ZX-1 additionally drops the CB gaps.  The world path
consumes the coord breaks but applies every tilt with the sign
opposite to the 3.7.1 optical convention (RT-4,
`AUDIT_RAYTRACE_CORE_2026_07_08.md`).  Net: a folded .zmx imports
loudly-correct-looking data whose two consumers are respectively
fold-blind and fold-mirrored.  Fixing RT-4 (one sign) plus ZX-1
(one thickness fold-in) restores a correct path (`trace_world`);
bridging `coord_breaks` into `surfaces_from_prescription` would
restore the second.

### ZX-3 (P4) — the loader drops `is_stop` from `prescription['surfaces']`, so the F-29 stop-preserving export can never fire on a loaded file
The lens-only `prescription_surfaces` entries carry only
radius/conic/aspherics/glasses (+ Q keys) — no `is_stop`, no
`semi_diameter` — and the loader sets no top-level `'stop_index'`.
But the v5.4.6 F-29 stop resolution in `export_zemax_zmx` /
`_export_zemax_zmx_full` / `export_zemax_lens_data` searches exactly
`prescription['surfaces'][i].get('is_stop')`: for every loaded
prescription it falls through to the default **0**, relocating STOP
to the first refractive surface on re-export even though the source
file declared it explicitly (the loader *does* preserve `is_stop` on
the `elements` list — the exporters just never consult it).  The
tracer's stop detection similarly degrades to its
first-finite-aperture heuristic.  **Fix**: copy `is_stop` (and
`semi_diameter`) into the lens-only surface dicts, or teach the
F-29 resolution to fall back to `elements`.

### ZX-4 (P4) — `back_focal_length` is a dead parameter in the full writer
`_export_zemax_zmx_full` accepts `back_focal_length` but never reads
it: the last element's thickness-after is
`all_thicknesses[len−1]` → out of range → `0.0`, and the image
surface is emitted with `DISZ 0.0` unconditionally.  Any
mirror/coord-break-bearing prescription exported with an explicit
BFL silently gets image distance 0 (the simple lens-only writer
honours it).  The recurring dead-parameter class.

### Nits
* The latin-1 fallback always decodes, so a non-Zemax file fails the
  `SURF` probe and raises "could not read … with any supported
  encoding" — a misleading message for what is actually "not a .zmx".
* A STOP declared on a COORDBRK surface (legal in Zemax) is filtered
  out with the break — aperture falls back to max-DIAM.
* `GCAT SCHOTT MISC` is hardcoded in both writers — exported designs
  using CDGM/OHARA/HIKARI/SUMITA glasses won't resolve their
  catalogues on the Zemax side.
* Q-type `r_max` final fallback is a 1.0 m placeholder that *passes*
  the downstream positive-r_max validation — silently wrong
  normalisation when PARM 0 and DIAM are both absent/zero.
* `export_zemax_lens_data` docstring claims defaults
  ("wavelength : float, default 1.31e-6"; "stop_surface : int,
  default 0") that no longer match the keyword-required /
  F-29-resolved signature.

---

## 3. Coverage statement

Every line of `io/prescriptions_zemax.py` read.  Not audited here:
the sibling loaders (`prescriptions_code_v.py`, 430;
`prescriptions_quadoa.py`, 364), `prescriptions_transforms.py` (441),
`prescriptions_builders.py` (525), `codegen.py` (1,009), and
`storage.py` (1,604) — natural next tranche, together with the
remaining `elements/` non-PMM modules and `optimize/`.

---

*Audit performed single-context against lumenairy v5.21, 2026-07-08.
Companion docs: `AUDIT_RAYTRACE_CORE_2026_07_08.md` (RT-4, the
world-path tilt-sign half of ZX-2),
`AUDIT_GLASS_POLARIZATION_2026_07_08.md`, and the 07-07 set.*
