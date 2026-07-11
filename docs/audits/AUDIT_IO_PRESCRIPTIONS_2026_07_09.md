# Prescription I/O Audit — 2026-07-09

Scope: the prescription loaders/builders siblings of the Zemax loader —
`io/prescriptions_code_v.py` (430, CODE V `.seq`),
`io/prescriptions_quadoa.py` (364, Quadoa `.qos` JSON),
`io/prescriptions_builders.py` (525, the `make_*` constructors +
Thorlabs catalog), and `io/prescriptions_transforms.py` (441,
`scale_prescription` et al.).  Chosen to check whether the Zemax
findings (ZX-1 dropped coord-break DISZ, ZX-3 dropped `is_stop`) recur
across the loader family.  Read-only; the OAP geometry and the scale
identity re-derived.

---

## 1. Verdict

**Clean — one P4, and the Zemax defects do NOT recur systemically.**
The cross-loader check was the point of this tranche, and it's
reassuring:

* **ZX-3 does NOT recur.** Both `load_codev_seq` and `load_quadoa_qos`
  emit a top-level `stop_index` (Code-V from the `STO` surface; QUADOA
  from per-surface `is_stop` or the `stop_surface` field), so the F-29
  stop-preserving export fires correctly on their loaded prescriptions
  — unlike the Zemax loader, which drops `is_stop` from the lens-only
  surfaces.  ZX-3 is Zemax-specific.
* **ZX-1 is Zemax-specific too.** Neither sibling has the
  coord-break-DISZ-dropped-from-`thicknesses` bug: QUADOA is JSON and
  preserves every unknown field losslessly under `surf['_extras']`;
  Code-V doesn't parse coord breaks at all (see CV-1).
* **BFL preservation works in both** (C-P0-4 for Code-V, C-P0-5 for
  QUADOA): prefer the explicit image-plane/`back_focal_length` field,
  fall back to the last surface's THI.
* **Builders' geometry verified**: `make_singlet`/`make_doublet`/
  `make_cylindrical`/`make_biconic` thread `glass_before`/`glass_after`
  correctly; **`make_off_axis_parabola`** — I re-derived the parent
  paraboloid relations `dz/dr = r/(2f) = tan α ⇒ h_decenter = 2f·tan α`,
  vertex `R = 2f`, conic `k = −1` — all correct (the v4.15.1 factor-of-2
  reconciliation and the `vertex_radius` P3-F1-3 validation are in
  place), and the docstring is honest about the traced-vs-paraxial
  consumer caveat.
* **`scale_prescription`** — the self-similarity identity is exact,
  including the aspheric-coefficient rule `A_n → A_n/factor^(n−1)`
  (I verified `Σ A_n' (f·h)^n = f·Σ A_n h^n` gives exactly this), the
  inf-radius preservation, and the correct non-scaling of
  conic/tilt/glass/wavelength.  It is **coord-break-aware** (scales
  decenters + `thickness_m`, leaves tilts), the very thing ZX-1's
  driver path dropped.
* Thorlabs catalog dispatch (singlet/doublet) and the QUADOA
  aspheric (de)serialisation are sound.

## 2. Findings

### CV-1 (P4) — `load_codev_seq` silently drops coordinate decenter/tilt directives; folded CODE V designs import as straight-axis
The loader's documented policy is "unknown commands are silently
ignored" (for tolerancing / zoom / spot-diagram boilerplate — benign).
But CODE V encodes folds via `XDE`/`YDE`/`ZDE` (decenters) and
`ADE`/`BDE`/`CDE` (tilts) on surfaces, and those fall into the
silent-ignore path — so a folded `.seq` imports as a **straight-axis**
system with the fold geometry gone and **no warning**.  This is the
same geometry-silently-changed class as ZX-1/ZX-2, and here it's a
*total* drop (the Zemax loader at least emits a `coord_breaks` list).
Because the drop is silent it reads as a successful import.  **Fix**:
detect `XDE`/`YDE`/`ADE`/`BDE`/… and either emit a `coord_breaks`
entry (matching the Zemax loader's schema, so `trace_world` /
`world_surfaces_from_prescription` can consume it) or, at minimum,
warn once that fold directives were dropped.  Related: `ASP`/aspheric
coefficient directives are likewise unparsed, so an aspheric `.seq`
degrades to base conic silently (same class as the Zemax `.txt`
loader's P3-43).

### Nits
* Code-V `STO` handler has a dead expression `current['index']` (reads
  and discards) — refactor residue, the recurring dead-statement class.
* Code-V `aspheric_coeffs` is always `None` (no `ASP`/`SDG` parsing);
  folded into CV-1's "warn on dropped shape data" fix.
* QUADOA `_extras` preservation is the gold standard here — a folded
  `.qos` keeps its coord-break fields (unconsumed but not lost), which
  is strictly better than Code-V's total drop; noted as the pattern
  the Code-V (and Zemax) loaders should move toward.

## 3. Coverage statement

Deep-read: `load_codev_seq` + `export_codev_seq` structure;
`load_quadoa_qos` (+ the aspheric (de)serialisers); `make_singlet` /
`make_cylindrical` / `make_biconic` / `make_doublet` /
`make_off_axis_parabola` + `thorlabs_lens` + the catalog;
`scale_prescription` in full.  Structurally covered:
`prescriptions_transforms.py`'s `normalize_prescription`,
`split_prescription_at_mirrors`, `has_mirrors` (read for shape, not
derivation-checked), and the Thorlabs catalog numeric data (taken as
vendor-sourced per the file comment, not independently verified against
Thorlabs).  **Not audited**: `io/storage.py` (1,604, HDF5/npz
serialisation), `io/codegen.py` (1,009, prescription→code emission),
and the `optimize/` tail (`multi_objective.py`, `multiconfig.py`,
`_merit_jit.py`, `core.py`) — the remaining ground.

---

*Audit performed single-context against lumenairy v5.21, 2026-07-09.
Companion doc: `AUDIT_IO_ZEMAX_2026_07_08.md` (ZX-1..4, the loader-family
findings this tranche cross-checks against).*
