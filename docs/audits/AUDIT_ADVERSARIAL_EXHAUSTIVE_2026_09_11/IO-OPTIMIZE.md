# IO-OPTIMIZE audit — `lumenairy/io/*` (prescription parsers, storage, codegen) + `lumenairy/optimize/*`

Repo: `D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy` (main, v5.45.x).
Env: CPython 3.14, numpy 2.4.6, scipy 1.17.1, jax 0.10.1 (CPU), numba 0.65, h5py 3.16, zarr 3.2.1. pymoo NOT installed.
All scratch under `…/scratchpad/IO-OPTIMIZE/`. No repo file created, modified or deleted.

## Scope read

Read line by line:
`io/prescriptions.py`, `io/__init__.py`, `io/prescriptions_zemax.py` (2501),
`io/prescriptions_code_v.py` (475), `io/prescriptions_quadoa.py` (399),
`io/prescriptions_transforms.py` (570), `io/prescriptions_builders.py` (540),
`io/storage.py` (1972), `io/codegen.py` (1062);
`optimize/driver.py` (1635), `optimize/merit_terms.py` (key merits + all geometric
constraints), `optimize/wrapper_merits.py` (cache layer), `optimize/jax_merits.py`,
`optimize/_merit_jit.py` (263), `optimize/parameterizations.py` (479),
`optimize/multiconfig.py`, `optimize/multi_objective.py`, `optimize/context.py`
(Constraint/EvaluationContext), `examples/07,08`.
Cross-read for confirmation: `CONVENTIONS.md` §7, `raytrace/world.py::_apply_coord_break`,
`raytrace/seidel.py::system_abcd`, `glass.py` registry, `docs/audits/AUDIT_IO_*`,
`tests/unit/test_audit_misc.py` (CODE V fixture).

Executed probes (all numbers below are measured on this machine):
hand-written `.zmx` (doublet / EVENASPH / COORDBRK+MIRROR / PARAXIAL / TOROIDAL /
BICONICX / MNUM-MCON / UTF-16LE+BOM / UTF-16BE / UNIT MM,CM,M,IN,BOGUS), hand-written
`.seq` (DIM M/C/I/MM/IN, missing RDY, K + A/B + REFL + RMD REFL), `.qos` round-trip,
`scale_prescription` round-trip + self-similarity, codegen injection / `inf`-`nan` /
round-trip, HDF5+Zarr 19-type metadata probe, 2-process × 50-append concurrency,
4096² complex128 write/read timings, numba-vs-NumPy merit-kernel parity,
`jax.grad` vs finite differences, merit-evaluation counting.

---

## Findings

### **[P0] `load_codev_seq` / `export_codev_seq` mis-scale every genuine CODE V file: `DIM M` is read as METRES, but CODE V's `DIM M` means MILLIMETRES — and `DIM C` / `DIM I` are not recognised at all** — `lumenairy/io/prescriptions_code_v.py:223-227, 252-258, 100`

The loader's unit table is
```python
def _unit_to_meters(v):
    return {'M': 1.0, 'MM': 1e-3, 'IN': 0.0254}[units] * float(v)
...
if unit_tok in ('M', 'MM', 'IN'):     # line 255 — anything else is IGNORED
    units = unit_tok
```
with `units = 'M'` (→ ×1.0) as the default, and the writer mirrors it
(`scale = {'M': 1.0, 'MM': 1e3, 'IN': 1/0.0254}`, default `units='M'`).

CODE V's `DIM` command takes exactly three tokens — `M` (millimetres), `C`
(centimetres), `I` (inches); the language has no metre unit. (This is also the token
set named in the audit brief: "`DIM M/C/I`".) Consequences, measured:

| file says | loader gives R1 | CODE V means | error |
|---|---|---|---|
| `DIM M`  | 62.75 m      | 0.06275 m  | **×1000** |
| `DIM C`  | 62.75 m (token ignored → default `M`) | 0.6275 m | ×100 |
| `DIM I`  | 62.75 m (token ignored → default `M`) | 1.59385 m | ×39.37 |
| `DIM MM` | 0.06275 m | (CODE V never writes `MM`) | — |

Reproduction (`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/IO-OPTIMIZE/p2_codev.py`): the AC254-100 doublet written as a
normal CODE V sequence (`DIM M`, `RDY 62.75`, `THI 4.0`, `GLA N-BAF10` …) loads as

```
radii as loaded [m]: [62.75, -45.71, -128.23]     (i.e. 62 750 mm)
EFL = 72.2154 m                                    (should be 72.2154 mm)
warnings: []                                       <-- silent
```

Impact: any externally-authored `.seq` imports 1000× too large with no diagnostic, and
any file this library writes with the default `units='M'` is 1000× too large when CODE V
reads it — while the docstring advertises `'MM'/'IN' trigger conversion on write (useful
for handing files to CODE V users)`. The format is de-facto a lumenairy-private dialect
that happens to reuse CODE V keywords; the repo's own fixture
(`tests/unit/test_audit_misc.py:6007`) writes `DIM M` with SI-metre values, so the wrong
convention is pinned by the test suite.

Fix: map `{'M': 1e-3, 'C': 1e-2, 'I': 0.0254}` (keeping `MM`/`IN` as tolerated aliases),
warn on an unrecognised `DIM` token instead of falling through to the default (the
Zemax loader already does exactly this at `prescriptions_zemax.py:507-516`), change the
writer's default to `units='M'` meaning millimetres, and add a schema-version marker or
a migration note for files written by earlier releases.

---

### **[P1] Zemax loader silently DROPS every powered surface that is not glass / mirror / DGRATING when it lies outside the glass span — PARAXIAL ideal lenses, air-spaced aspheric phase plates, and the STOP flag on them** — `lumenairy/io/prescriptions_zemax.py:673-696`

The window auto-detect is
```python
active = [s for s in optical_surfaces
          if s['glass'] is not None or s['is_mirror']
          or _raw_surface_is_dgrating(s)]
```
A `TYPE PARAXIAL` surface (Zemax's ideal-lens, PARM 1 = focal length) is air-to-air with
no glass, so it never enters `active`; the window starts at the first glass surface and
the paraxial lens is excluded from `lens_surfaces` **before** reaching the
unsupported-SURFTYPE branch that would have warned (line 876-906). Measured
(`p1c_zemax2.py` / `p1d_more.py`), file = `PARAXIAL f=100 mm` + STOP at surface 1, then a
glass singlet:

```
PARAXIAL+glass: object_distance = 0.1   elements surf_nums = [2, 3]
                stop_index = None       warnings: []
```

The 100 mm ideal lens is gone, the declared STOP is gone (aperture falls back to max
DIAM), and only its 100 mm DISZ survives as free space. The same happens for an
air-to-air `EVENASPH` phase plate upstream of the glass (verified: `elements surf_nums =
[2, 3]`, no warning) and would for `ABCD`, `ZERNSAG`, `GRID_PHASE`, etc.

This is the exact failure mode the v5.32 DGRATING work fixed (`_raw_surface_is_dgrating`
was added to `active` precisely because "a DGRATING is an air-to-air flat … discarded
with no warning"); the fix was applied for one surface type only.

Impact: a legitimate Zemax file imports as a different optical system, silently. The
loader's own contract ("unsupported SURFTYPE … warn loudly per surface") is not honoured
because the surface never reaches that code.

Fix: extend the `active` predicate to any surface whose SURFTYPE is known to carry power
(`PARAXIAL`, `PARAXIALXY`, `IDEAL`, `ABCD`, `EVENASPH`/`ODDASPHE`/`QBFS`/`QCON` with a
non-zero PARM table, `ZERNSAG`, `GRID_SAG`, …), or — minimally — warn once naming every
optical surface the window excluded that carries a non-zero CURV or a non-empty PARM
table.

---

### **[P1] `load_codev_seq` silently ignores `REFL` / `RMD REFL` (mirrors), `K` (conic) and the `A`…`J` aspheric coefficients** — `lumenairy/io/prescriptions_code_v.py:363-379`

The per-surface dispatch handles only `STO`, `RDY`, `CUY`, `THI`, `GLA`, `CON`; every
other token falls into the silent-ignore tail, except `XDE/YDE/ZDE/ADE/BDE/CDE` and
`ASP/SDG` which are collected and warned about once (CV-1 from the 2026-07-09 audit).

* **`REFL` / `RMD REFL`** are in neither warn set. A CODE V fold or catadioptric mirror
  therefore imports as an **air-to-air dummy surface** — no `element_type='mirror'`, no
  `elements` key at all, `has_mirrors()` False — so the design silently un-folds *and*
  loses the mirror's optical power. This is strictly worse than the `XDE/ADE` case,
  which at least warns.
* **`K`** is CODE V's conic-constant command; the parser accepts only `CON` (what its own
  exporter writes). `K` is in neither warn set.
* **`A`/`B`/`C`/`D`…** (4th…10th-order asphere coefficients) are dropped; the `ASP`
  declaration warns, but a surface that carries `K`/`A`/`B` without a literal `ASP`
  token warns nothing.

Measured (`p2_codev.py`, file with `ASP`/`K -1.0`/`A 1.234E-07`/`B -5.0E-11`/`REFL`/`RMD REFL`):

```
conics: [0.0, 0.0, 0.0]      asph: [None, None, None]
glasses: [('air','N-BK7'), ('N-BK7','air'), ('air','air')]   # S2 REFL became air->air
has elements key: False      has_mirrors: False
warnings: ["… dropped unparsed shape/geometry directives -- aspheric directives ['ASP'] …"]
```
(The conic `K -1.0` vanished with no mention even though a warning did fire for `ASP`.)

Fix: add `REFL`/`RMD` handling (set `is_mirror`, emit an `elements` list with
`element_type='mirror'`, or at minimum refuse the file), accept `K` as a `CON` alias, and
add `A`…`J` to the warn set (ideally parse them into `aspheric_coeffs` with the CODE V
power convention A=r⁴ … and the lens-unit → metre rescale `a_m = a_file · L^(p-1)`).

---

### **[P1] `export_zemax_zmx` and `export_codev_seq` silently drop `radius_y` / `conic_y` / `aspheric_coeffs_y` — a cylindrical or biconic lens exports as a rotationally symmetric sphere** — `prescriptions_zemax.py:2422-2476, 2194-2271`; `prescriptions_code_v.py:130-146`

The writers emit `CURV` from `radius` only. The one anamorphic-adjacent guard,
`_warn_dropped_qtype` (`prescriptions_zemax.py:1973`), covers `freeform_type` /
`q_*_coeffs` / `r_max` and nothing else. Measured (`p7b`-family probe):

```
cyl S0: radius=0.05  radius_y=inf                 # make_cylindrical(50 mm, axis='x')
  .zmx: radius_y preserved=False  warnings=[]
  .seq: radius_y preserved=False  warnings=[]
  .qos: radius_y preserved=True   warnings=[]
  reloaded from .zmx: radius=0.05  radius_y=None
```

A one-axis focusing element becomes a two-axis focusing element with no diagnostic.
`make_cylindrical`, `make_biconic` and every `.qos`-loaded anamorphic prescription are
affected. The Quadoa writer handles it correctly, which makes the gap an inconsistency
rather than a format limitation.

Fix: emit `TYPE BICONICX` (Zemax) / `YTO`+`CUY`/`CUX` (CODE V) when `radius_y` differs
from `radius`, or — as the minimum matching the P2-20 precedent — warn as loudly as
`_warn_dropped_qtype` does.

---

### **[P1] `codegen` interpolates untrusted `.zmx` strings into CODE positions of the generated script — arbitrary-code injection** — `lumenairy/io/codegen.py:694-698, 787-788, 881, 963-969`

```python
lines.append(f"la.GLASS_REGISTRY['{g}'] = ('specs', 'CATALOG', 'PAGE')  # TODO: …")
lines.append(f'    if verbose: print("Applying {label} ...")')
lines.append(f"    print('Running: {sys_name}')")
```
`g` is the raw `GLAS <token>` from the `.zmx` (whitespace-split, so any character except
whitespace is allowed — quotes included); `label` derives from the same glass names via
`_lens_group_name`; `sys_name` is the prescription name (file stem by default). None are
escaped or `repr()`-ed.

Proof of concept (`p7b.py` / final probe): a `.zmx` whose `GLAS` token is
`X'];print(0x50574e4544);#` produces

```
INJECTED LINE : la.GLASS_REGISTRY['X'];print(0x50574e4544);#'] = ('specs', 'CATALOG', 'PAGE')  # TODO: …
FULL generated script PARSES as valid Python
```
i.e. the payload is live code in the emitted file. (With the prefix chosen as a real
registry key — `N-BK7'];…;#` — the leading lookup also succeeds at runtime, so the
payload executes when the user runs the generated script. Note `unknown = [g for g in
glasses_used if g not in GLASS_REGISTRY]` still emits the line, because the *full*
doctored token is not a registry key.)

Threat model is realistic: `generate_script_from_zmx()` on a vendor / colleague-supplied
`.zmx`, or a `.zmx` fetched from a lens-catalogue site. There is no `eval`/`exec` on the
load path (good), but the write path is equivalent once the user runs the script.

Fix: emit every string through `repr()` (`la.GLASS_REGISTRY[{g!r}] = …`,
`print({label!r})`), and validate glass/name tokens against
`re.fullmatch(r'[A-Za-z0-9_\-\.\+]+', g)` at load time.

---

### **[P1] Zarr per-plane metadata bypasses the canonical type-tagged codec — ndarray metadata is silently stringified (and truncated with `...` beyond 1000 elements)** — `lumenairy/io/storage.py:1426-1431` vs `storage.py:350-388, 954`

Every HDF5 write site routes user metadata through `_h5_write_meta_attrs` →
`_meta_dumps` (the A-4 / S4-19 contract). `_zarr_append_plane` alone still does the
pre-A-4 raw loop:
```python
if metadata:
    for k, v in metadata.items():
        try:    ds.attrs[str(k)] = v
        except TypeError:
            ds.attrs[str(k)] = str(v)
```
Measured over the module's own 19-type probe set (`p6_storage.py`):

```
--- HDF5 append_plane metadata: 18/19 faithful       (only np.float32 -> float)
--- ZARR append_plane metadata: 13/19 faithful
       complex   -> str '(1+2j)'
       bytes     -> str "b'\x00\x01ab'"
       tuple     -> list
       ndarray_f -> str '[[0. 1. 2.]\n [3. 4. 5.]]'
       ndarray_c -> str '[1.+1.j 2.-2.j]'
       np_scalar -> str '2.5'
--- ZARR sim_metadata: 18/19 faithful                (that path DOES use the codec)
```
The ndarray case is irrecoverable, not merely type-changed: `str()` of an array larger
than numpy's 1000-element print threshold inserts `...`, so the values are gone. Because
`set_storage_backend('zarr')` is a one-line switch and the sim-metadata path *is* fixed,
the same call with the same arguments has different fidelity depending on a global.

Fix: call `_h5_write_meta_attrs`'s zarr twin (`store.attrs[_META_BLOB_KEY] =
_meta_dumps(metadata)` plus the flattened shadow copy) in `_zarr_append_plane`, and have
`_zarr_load_planes` / `_zarr_list_planes` / `_zarr_load_plane_by_label` apply the same
blob overlay `_h5_read_attrs` does. Also widen `except TypeError` (zarr can raise
`ValueError`).

---

### **[P1] `load_codev_seq`: a surface with no `RDY` line yields `radius=None`, not `inf`** — `lumenairy/io/prescriptions_code_v.py:295-299, 424`

```python
current = {'kind': 'refracting', 'index': …, 'radius': None, …}   # key EXISTS, value None
...
'radius': s.get('radius', float('inf')),                          # default never fires
```
`dict.get` returns the stored `None`, so the documented "flat surface" default is dead
code. A perfectly legal CODE V dummy surface (only `THI`) makes the prescription
unusable:

```
=== surface without RDY: radii = [62.75, None, -50.0]
    system_abcd RAISES: ValueError validate_prescription: 1 issue(s) found:
      surfaces[1].radius: is None (use np.inf for a flat surface)
```
Fix: `'radius': (float('inf') if s.get('radius') is None else s['radius'])`.

---

### **[P1] `THORLABS_CATALOG['LA1509-C']` encodes a 200 mm lens under a 100 mm part number** — `lumenairy/io/prescriptions_builders.py:457-461`

```python
'LA1509-C': {  # f=200mm, N-BK7, 1" dia (curved side first for collimation)
    'type': 'singlet', 'R1': 103.29e-3, 'R2': float('inf'),
    'd': 3.6e-3, 'glass': 'N-BK7', 'aperture': 25.4e-3,
},
```
Thorlabs LA1509 is **R = 51.5 mm, t = 3.6 mm, N-BK7, f = 100 mm** (the ground-truth
value supplied in the audit brief). The entry's thickness matches LA1509; its radius is
that of a 200 mm lens. Measured:

```
thorlabs_lens('LA1050-C'): R1=51.500 mm  EFL@1310 = 102.27 mm
thorlabs_lens('LA1509-C'): R1=103.290 mm EFL@1310 = 205.11 mm   <-- part number says 100 mm
thorlabs_lens('LA1301-C'): R1=129.200 mm EFL@1310 = 256.56 mm   (LA1301 = 250 mm, OK)
```
The header comment claims "Surface data from Thorlabs Zemax files", so the claim is
falsifiable and false. The same wrong radius is baked into
`validation/real_lens_opd/zemax_prescriptions/LA1509_C.zmx` (`CURV 0.0096814793`), i.e.
the validation fixture cannot catch it. A user calling `thorlabs_lens('LA1509-C')` gets a
2× focal-length error with no diagnostic.

Fix: set `R1 = 51.5e-3` for `LA1509-C` (and re-derive the `-C` fixture), or rename the
entry to whatever part the 103.29 mm radius actually belongs to. Worth a one-off
audit of every catalogue entry against the vendor tables.

---

### **[P2] `scale_prescription` is not self-similar for Forbes-Q freeforms, diffractives, or a stored BFL** — `lumenairy/io/prescriptions_transforms.py:124-161`

`_scale_surface_like` handles `radius`, `radius_y`, `semi_diameter`,
`aspheric_coeffs{,_y}` only, and the top level handles `aperture_diameter`,
`object_distance`, `thicknesses`, `all_thicknesses`, `coord_breaks`. Untouched:

* `r_max` and `q_bfs_coeffs` / `q_con_coeffs` — both **lengths in metres** (a Forbes-Q
  coefficient is a sag length, `r_max` a normalisation radius), so a scaled Q-type
  surface keeps its original freeform sag on a rescaled base conic;
* every `diffractives[k]` field (`period`, `origin`, `gap_before`, `gap_after`,
  `semi_diameter`) — the v5.32 DOE payload;
* `back_focal_length` (written by `load_codev_seq` / `load_quadoa_qos`).

Measured at `s = 0.25` (`p7b.py`):
```
scale round-trip max |dR| : 0.0        # core identity exact
aspheric self-similarity  : 0.25       # exact, as the docstring claims
scaled r_max              : 0.0075     expected 0.001875
scaled q_bfs_coeffs       : [1e-06, 2e-07]  expected [2.5e-07, 5e-08]
scaled back_focal_length  : 0.084      expected 0.021
scaled diffractive period : 2e-06      expected 5e-07
scaled diffractive gap_before : 0.01   expected 0.0025
```
The docstring lists what it scales and what it deliberately doesn't (conics, tilts,
glass, wavelength) — these three families appear in neither list, so it reads as
coverage.

Fix: scale `r_max`, `q_*_coeffs` and `back_focal_length` by `s`; scale the diffractives'
lengths by `s` (the grating `period` too — a self-similar system at a fixed wavelength
changes its diffraction angles, so if that is deliberate it must be *said*); add a
warning for any length-like key the transform does not recognise.

---

### **[P2] `codegen` emits bare `inf` / `nan` (NameError in the generated script) and loses the sign of a `-inf` radius** — `lumenairy/io/codegen.py:723-728, 986-992`

```python
r_str = "float('inf')" if np.isinf(surf['radius']) else f'{surf["radius"]:.17e}'
lines.append(f'        {{"radius": {r_str}, "conic": {surf["conic"]},')
asph_str = repr(asph) if asph else 'None'
```
* `np.isinf(-inf)` is True → a `radius = -inf` surface is written as `float('inf')`:
  **sign silently flipped**.
* `conic` goes through `str()`; `inf`/`nan` render as bare `inf`/`nan`, which are not
  Python literals.
* `repr(dict)` on `aspheric_coeffs` emits `{4: nan, 6: 100000.0}` — same problem.
* A `nan` radius renders as `nan` via `f'{x:.17e}'`.

Measured:
```
GEN2> {"radius": 5.00000000000000028e-02, "conic": inf,
GEN2> "aspheric_coeffs": {4: nan, 6: 100000.0},
GEN2> {"radius": float('inf'), "conic": 0.0,      # this surface was -inf
  -> prescription block FAILS: NameError name 'inf' is not defined
```
Fix: one shared `_pyfloat(v)` helper returning `float('inf')` / `float('-inf')` /
`float('nan')` / `repr(v)`, used for radius, conic, thicknesses, aperture and every
aspheric coefficient value.

---

### **[P2] `generate_simulation_script` crashes on a `normalize_prescription` output — `KeyError: 'element_type'`** — `lumenairy/io/codegen.py:360, 401` vs `prescriptions_transforms.py:264-267`

`normalize_prescription` is documented as "the canonical superset … the recommended
idiom" and, when `elements` is absent, sets `rx['elements'] = list(surfs)` — plain
surface dicts with no `element_type`. codegen subscripts it unguarded:
```python
if elem['element_type'] == 'mirror':
```
Measured: `normalize_prescription(make_singlet(...))` → `generate_simulation_script`
raises `KeyError: 'element_type'`. So the one helper that exists to make a
builder-produced prescription codegen-shaped produces something codegen rejects.

Fix: `elem.get('element_type', 'surface')` in codegen (the value is already defaulted
that way at `codegen.py:408`), and have `normalize_prescription` stamp
`element_type='surface'` on the mirrored entries.

---

### **[P2] Zemax `MNUM` / `MCON` multi-configuration records are dropped without a word** — `lumenairy/io/prescriptions_zemax.py:525-620`

The keyword dispatch handles `SURF/TYPE/STOP/CURV/CONI/DISZ/GLAS/MIRR/DIAM/PARM/COMM`
and `UNIT`; `MNUM`/`MCON` are not matched at any level, so a zoom / thermal / multi-config
`.zmx` imports as the base LDE state with no `configurations` key and no warning:

```
=== (j) multiconfig: keys= ['all_thicknesses','aperture_diameter','coord_breaks',
     'diffractives','elements','name','object_distance','stop_index','surfaces','thicknesses']
     thicknesses: [0.005]   warnings: []
```
That is usually config 1, but the user gets no signal that the other N−1 zoom positions
existed — and `optimize/multiconfig.py` + `examples/08_multiconfig_zoom.py` make the
feature look supported end to end.

Fix: at minimum warn once naming the operand rows found; better, collect
`{'configurations': [{op, surface, value}, …]}` alongside `coord_breaks`/`diffractives`
so `create_zoom_configs` can consume it.

---

### **[P2] HDF5 default `compression='gzip', compression_opts=4` costs 24× write time for 5.6 % space on complex fields** — `storage.py:440-441, 539-540, 665-666, 760-761`

Measured, 4096² complex128 (256 MiB raw), same machine, same file system:

| compression | write | on-disk | read |
|---|---|---|---|
| `gzip`, level 4 (default) | **17.05 s** | 241.6 MiB (−5.6 %) | **3.84 s** |
| `None` | **0.70 s** | 256.0 MiB | **1.16 s** |

Complex float mantissas are incompressible; gzip is almost pure overhead here. The same
default is on `append_plane_h5`, which is the per-plane hot path of every multi-plane
run, and its default `chunk_size=1024` makes a 16 MiB chunk for complex128 — far above
HDF5's 1 MiB default chunk cache, so every partial read re-inflates a whole chunk.

Fix: default `compression=None` for complex data (or dispatch on dtype), and if
compression is wanted use `shuffle=True` + `lzf`, or `scaleoffset` on the real/imag
parts. Shrink the default chunk so `chunk_bytes ≈ 1 MiB` (e.g. `chunk_size=256` for
complex128).

---

### **[P2] Every wave-leg merit evaluation pays 31 extra propagations for the through-focus scan, on every finite-difference probe** — `optimize/driver.py:912-941`, default `z_scan_n=31`

`evaluate()` runs, per call: `parameterization.build` (deepcopy) → ray leg → **one**
wave propagation → `through_focus_scan` over `z_scan_n=31` planes → `wave_opd_2d`. On
the default `jac='auto'` path *without* a `JaxMeritTerm`, `final_jac is None`
(`driver.py:1056-1061`) so scipy finite-differences the merit itself — n+1 full
`evaluate()` calls per gradient, each paying the 31-slice scan again.

Measured (3 free variables, `N=128`, `method='L-BFGS-B'`, `max_iter=2`, instrumented):
```
17 merit evals, 17 apply_real_lens calls, 527 through-focus slices
 -> per merit eval: 1.0 lens propagations + 31.0 focus-scan propagations
```
i.e. **97 % of the wave-leg work is the focus scan**, and it is recomputed from scratch
for every FD probe. The ray leg itself is cheap and is not the problem (profiled:
deepcopy 0.011 ms, `surfaces_from_prescription` 0.017 ms, `system_abcd` 0.047 ms,
`seidel_coefficients` 0.121 ms → 0.196 ms/eval total).

Fix, in increasing order of work: (a) skip the scan entirely when no merit reads
`strehl_best` / `z_best` / `rms_radius_best` (the same `needs_*` pattern already used for
`need_wave` / `need_ray`); (b) coarse-to-fine — 7 slices then a 5-point refine around the
peak, ≈2.6× cheaper for the same best-focus resolution; (c) cache the scan across the FD
stencil (the derivative of the *merit*, not of the focus location, is what is wanted —
a single scan at the stencil centre plus a quadratic focus model is usually enough).

---

### **[P2] `design_optimize_multi_objective` returns a 0-d NaN "Pareto front" when pymoo finds no feasible solution** — `optimize/multi_objective.py:360-365, 379`

pymoo's `Result.X` / `.F` are `None` when the final population is entirely infeasible.
```python
X = np.asarray(pymoo_res.X, dtype=np.float64)
if X.ndim == 1: X = X[None, :]
```
`np.asarray(None, dtype=np.float64)` is `array(nan)` with `ndim == 0` (verified on this
numpy: `np.asarray(None, float64) = array(nan)  ndim = 0`), so the `ndim == 1` guard does
not fire and a 0-d NaN array is returned as `ParetoResult.X`. With `progress` supplied,
`X.shape[0]` at line 379 raises `IndexError: tuple index out of range` instead.

pymoo is not installed here, so this is a desk finding confirmed only at the numpy step;
the `X is None` behaviour is pymoo's documented infeasible-run contract.

Fix: `if pymoo_res.X is None: raise ValueError("no feasible solution found …")` (or
return an empty `(0, n_params)` array with a warning) before the `asarray`.

---

### **[P2] `load_codev_seq` never warns about unknown glasses, unlike the Zemax loaders** — `prescriptions_code_v.py:422-465`

Both `load_zemax_zmx` and `load_zemax_prescription_data_txt` end with an
"unknown glasses" `UserWarning` naming the missing entries and how to register them
(`prescriptions_zemax.py:1117-1134, 1706-1724`). `load_codev_seq` has no such block, and
`load_quadoa_qos` none either. CODE V glass names are routinely `N-BK7_SCHOTT` or a
6-digit code (`517642.514`), neither of which is a `GLASS_REGISTRY` key. The failure
surfaces much later as a `ValueError` from `get_glass_index` inside a propagation, with
no pointer back to the file. (The repo's own CODE V test fixture uses `GLA BK7`, which
is *not* in the registry — confirmed: `get_glass_index('BK7', 587.6e-9)` raises.)

Fix: lift the Zemax block into a shared `_warn_unknown_glasses(elements, filepath)` and
call it from all four loaders.

---

### **[P2] No edge-thickness constraint exists anywhere in the library** — `optimize/merit_terms.py:1435-1532`

`MinThicknessMerit` / `MaxThicknessMerit` penalise the **centre** thickness only
(`ctx.prescription['thicknesses'][i]`). Edge thickness — `t_c − (sag₁ − sag₂)` evaluated
at the clear semi-diameter, which goes negative (knife edge) on a strongly biconvex or
steeply aspheric element long before the centre thickness does — has no merit, no
constraint helper, and no analysis function: a repo-wide grep for
`edge_thickness|EdgeThickness|edge thickness` hits only two incidental comments in
`elements/_lens_jax.py`. Every commercial code ships this (`ETGT`/`ETVA` in
OpticStudio, `ETH` in CODE V) because it is the constraint an unconstrained radius
optimisation violates first.

`MinThicknessMerit`'s docstring — "Minimum acceptable GLASS thickness [m]" — reads as
manufacturability coverage, which is exactly where the gap bites.

Fix: add `MinEdgeThicknessMerit(min_edge, weight)` computing
`t_edge = t_c + sag(R₂,k₂,h) − sag(R₁,k₁,h)` at `h = semi_diameter`, with the sign
convention checked against a concave-first meniscus (the sag difference changes sign
there — the case the probe list flags).

---

### **[P2] `load_codev_seq` / `load_quadoa_qos` do not emit `elements` / `all_thicknesses`, but three docstrings say they do** — `prescriptions_transforms.py:184-186, 318-321`

`normalize_prescription`'s docstring: "`load_codev_seq` / `load_quadoa_qos` match
`load_zemax_zmx`'s schema"; `split_prescription_at_mirrors`'s: "A prescription dict as
returned by `load_zemax_zmx`, `load_codev_seq`, or `load_quadoa_qos` -- i.e. carrying
both `'elements'` and `'all_thicknesses'`". Measured key diff for the same doublet:

```
zmx keys: all_thicknesses, aperture_diameter, coord_breaks, diffractives, elements,
          name, object_distance, stop_index, surfaces, thicknesses
seq keys: aperture_diameter, back_focal_length, name, stop_index, surfaces,
          thicknesses, wavelength
per-surface: zmx has is_stop + semi_diameter; seq has radius_y/conic_y/aspheric_coeffs_y
```
`split_prescription_at_mirrors` on a `.seq` prescription therefore takes the
`elements is None` early-return and reports a single refractive leg — correct for the
current CODE V loader (which can't see mirrors at all, see the P1 above) but only by
accident, and the docstring promises otherwise.

Fix: either populate `elements`/`all_thicknesses` in both loaders, or correct the three
docstrings and make `split_prescription_at_mirrors` warn when it silently falls back.

---

### **[P3] `.zmx` / generated-script record injection via `NAME` and `COMM`** — `prescriptions_zemax.py:2065, 2204-2205`; `codegen.py:616, 871, 881`

`export_zemax_zmx` writes `f'NAME {name}'` and `f'  COMM {comment}'` verbatim. A
prescription name containing a newline injects arbitrary `.zmx` records (measured: a
name of `'bad\nSURF 99\n  CURV 0.5\nNAME x'` produces extra `SURF` rows in the output
file). Similarly codegen's `f'ASM Simulation — {sys_name}'` sits inside a `"""` docstring
and a `"""` in the name terminates it. Lower severity than the glass-name case because
the name is usually caller-controlled. Fix: strip/escape newlines and quotes in `name`
and `comment` at both writers.

### **[P3] UTF-16-BE `.zmx` is reported as "not a Zemax file"** — `prescriptions_zemax.py:469-485`

The encoding ladder is `utf-16-le`, `utf-8`, `latin-1`; a BE-encoded file decodes
under `latin-1` without `SURF` and hits the `else` clause, whose message says the file
"does not appear to be a Zemax .zmx lens file". Zemax writes LE so this is rare, but the
diagnostic points at the wrong cause. Fix: try `utf-16` (BOM-sniffing) first, which
handles both byte orders, then `utf-16-le`.

### **[P3] `TOROIDAL` / `BICONICX` could map their PARM table onto the existing `radius_y` support** — `prescriptions_zemax.py:876-906`

Both warn loudly and import as base conic (verified: `radius=0.05, radius_y=None`,
warning emitted), which is honest. But the library already has full `radius_y`/`conic_y`
support (`make_biconic`, `.qos` I/O), and Zemax's TOROIDAL PARM 1 (radius of rotation)
and BICONICX PARM 1/2 (X curvature / X conic) map onto it directly. Low-cost upgrade
from "warned wrong" to "right".

### **[P3] `export_quadoa_qos` marks surface 0 as the stop even when the prescription has none** — `prescriptions_quadoa.py:172-176, 211`

`stop_surface` defaults to `0` when neither `stop_index` nor any `is_stop` is found, and
line 211 writes `'is_stop': bool(i == stop_surface)`. Re-loading always yields
`stop_index = 0`, inventing a stop. Fix: keep `stop_surface = None` and write `is_stop`
only when a stop was actually declared.

### **[P3] `create_zoom_configs` writes every thickness slot, including glass, and truncates silently** — `optimize/multiconfig.py:169-180`

The parameter is named `zoom_spacings` and documented as "the air-gap thicknesses", but
the loop is positional over the whole `thicknesses` list; `if j < len(pres['thicknesses'])`
drops extra entries with no warning. Fix: accept `(slot_index, value)` pairs or validate
the length and raise.

### **[P3] `method='newton'` drops bounds without the warning the generic branch emits** — `driver.py:1518-1526`

`trust-ncg` accepts no `bounds`; only the FD stencil is clipped. The generic `minimize`
branch warns loudly in exactly this situation (`driver.py:1547-1559`); the Newton branch
does not. Fix: emit the same warning.

### **[P3] `design_optimize_multi_objective` never uses `x0` for the "sanity check against bounds" its docstring promises** — `multi_objective.py:155-159, 224-229`

`x0` is only used for `n_params`. Fix: check `lb ≤ x0 ≤ ub` and warn, or correct the doc.

### **[P3] `_wrapper_merit_aperture_key` keys an ndarray aperture on `hash(bytes)`** — `wrapper_merits.py:146-149`

A 64-bit hash collision returns the *wrong* cached aperture mask. Astronomically
unlikely; noted because a content digest (`hashlib.blake2b(...).digest()`) costs the same
single byte-scan and removes the class entirely.

---

## Performance opportunities

1. **Through-focus scan dominates the wave leg** (measured 31 of every 32 propagations;
   see P2 above). Gate it on whether any merit reads the focus results; use
   coarse-to-fine or a cached-centre stencil for the FD probes. This is the single
   biggest win available in `design_optimize`.
2. **Analytic gradients are off by default.** `final_jac` is `None` unless the user
   constructs a `JaxMeritTerm(build_args=…)`; nothing in `merit_terms.py` has a JAX twin
   except `make_lg_aberration_merit_jax`. Every default run pays scipy's own (n+1)-eval
   numerical Jacobian. JAX is installed and the plumbing is correct (verified 2.3e-9
   relative error vs FD — see below); the missing piece is JAX twins for the cheap
   geometric merits (`FocalLengthMerit`, `BackFocalLengthMerit`, `MinThicknessMerit`,
   `MaxFNumberMerit`), which would make `jac='auto'` analytic for the whole
   ray-only family at essentially zero cost.
3. **HDF5 gzip default**: 24× write / 3.3× read penalty for 5.6 % space on complex
   fields (measured, 4096²). Default it off and shrink `chunk_size` to ~1 MiB of chunk.
4. **`DesignParameterization.build` deep-copies the whole template per evaluation**
   (`parameterizations.py:216`). Cheap for a singlet (0.011 ms measured) but O(prescription)
   for a 40-surface zoom with per-surface aspheric dicts; a copy-on-write or a
   pre-allocated scratch prescription mutated in place would remove it from the FD inner
   loop entirely.
5. **`seidel_coefficients` runs on every evaluation even when no merit reads
   `ctx.seidel`** (`driver.py:843-860`; 0.121 ms of the 0.196 ms ray-leg budget, ≈60 %).
   The `needs_*` pattern already exists for `need_wave`/`need_ray` — a `needs_seidel`
   flag would skip it for the common EFL/BFL/thickness merit sets.
6. `_zarr_append_plane` silently discards `compression` / `compression_opts`
   (`driver` of the dispatch at `storage.py:1704` forwards only `lock_timeout`), so a
   caller tuning compression on a `.zarr` store has no effect and no diagnostic.

## Alternative algorithms / methods

* **Levenberg–Marquardt with an analytic Jacobian from differential ray tracing.**
  `lumenairy/raytrace/differential.py` exists and implements ADRT (including
  `_adrt_coordbreak`), but a repo-wide grep shows **zero references to it from
  `lumenairy/optimize/`** — the `method='lm'` path
  (`driver.py:1210-1319`) finite-differences the residual vector like every other
  method. Wiring ADRT into a `residuals`/`jac` pair for the ray-leg merits would give
  scipy `least_squares(method='trf', jac=…)` an exact Jacobian at ~1 extra trace per
  parameter instead of n+1 full evaluations, which is the classical lens-design
  formulation (Jamieson, *Optimization Techniques in Lens Design*, ch. 4).
* **Damped least squares with orthonormalised descent (CODE V style)** — Grey's
  orthonormal-variable DLS (D. S. Grey, *JOSA* 53, 672 (1963); Dilworth, *Proc. SPIE*
  0147) conditions the normal equations by orthonormalising the variable set, which is
  what makes classical lens optimisation robust to the near-degenerate radius/thickness
  directions that L-BFGS-B struggles with here. A thin layer over the LM residual vector
  would slot into the existing `method='lm'` dispatch.
* **Storage: Zarr sharding** (zarr v3 `ShardingCodec`) for large field stacks — a 4096²
  complex128 plane at 1024² chunks is 16 chunks/plane and thousands of files for a long
  run; one shard per plane keeps the chunked read granularity with a single file, and is
  the intended v3 answer to exactly this write pattern.
* **Parsers: one tokenizer + a declarative record table instead of three hand-rolled
  line loops.** All three loaders are "split the line, `if keyword ==` ladder" with
  independently-evolved unit handling, warning policy, and schema output — which is
  precisely why the `DIM` bug (P0), the missing `REFL` (P1), and the missing
  unknown-glass warning (P2) exist in one loader and not the others. A shared
  `Record(keyword, arity, units, handler)` table per format, plus one schema builder,
  would make the loaders differ only where the formats do.
* **`through_focus_scan` → Gerchberg-style quadratic focus model.** Best focus from a
  3-point parabolic fit on Strehl, refined once, is ~5 propagations instead of 31 for
  sub-Rayleigh accuracy.

## Code organization observations

* `prescriptions_zemax.py` `load_zemax_zmx` is ~860 lines in one function (encoding
  sniff → unit table → tokenizer → window detect → element build → aperture → coord
  breaks → diffractives → return). `load_zemax_prescription_data_txt` is a second ~520-line
  copy of the same pipeline with `S4-1`, `S4-9`, `P3-42`, `P3-43` each re-implementing a
  fix the `.zmx` twin already had. The two share `_reassign_stop_off_coordbrk` and
  nothing else; the glass-warning block, the medium-between loop, the element builder,
  the thickness folding, the aperture fallback and the stop-index derivation are all
  duplicated verbatim. Every one of those is a place a future fix lands on one side only
  — which is the observed history.
* Four loaders emit four different schemas (measured key diff in the P2 above). A single
  `PrescriptionSchema` dataclass with per-loader `from_*` constructors would make
  `normalize_prescription` unnecessary rather than a papering-over layer that itself
  produces a shape `codegen` rejects.
* `storage.py` carries two parallel backend implementations where the HDF5 side has
  received four fidelity fixes (A-4, S4-9, S4-19, P3-15) and the Zarr side one. The
  metadata codec (`_meta_dumps` / `_meta_loads`) is already backend-agnostic; the zarr
  writers simply don't call it in one place (P1 above).
* `driver.py::design_optimize` is a 1160-line function with eight method branches, each
  with its own progress/callback/jac handling. `_post_eval_bookkeeping` was factored out
  (OPT-3) precisely because the `lm` branch had drifted; the DE / basin-hopping /
  dual-annealing / newton branches still each re-derive their own jac-forwarding and
  cancellation logic.
* The in-code audit trail is unusually good — nearly every non-obvious line carries the
  audit ID, the measured before-state, and a "do not re-flip this" note. It is also,
  by volume, the majority of several files; a `docs/audits/` back-reference with a short
  in-code marker would keep the signal without the 40-line comment blocks.

## Unverified suspicions

* **pymoo `X is None` path** (P2) — the numpy half is measured; the pymoo half is a desk
  reading of its infeasible-run contract. pymoo is not installed here.
* **CODE V `K` vs `CON`** — I am confident `K` is CODE V's conic keyword (it is the one
  the audit brief names). Whether CODE V *also* accepts `CON` I could not verify offline;
  if it does, the finding narrows to "`K` unsupported" rather than "the parser uses a
  keyword CODE V doesn't have".
* **`design_optimize` and NaN merits**: `merit_fn` returns a NaN merit straight to scipy
  with no warning (the S4-5 guards cover `strehl_best` and `rms_radius_best`, not the
  sum). L-BFGS-B then terminates `ABNORMAL` or stalls. I did not construct a merit that
  reliably produces NaN through the guarded paths, so I cannot show the failure end to
  end — but the unguarded return at `driver.py:1143` is plain.
* **`optimize` globals are process-wide, not thread-local**: `design_optimize` mutates the
  library-global complex dtype (restored in a `finally`, correctly) and
  `jax_merits._ensure_jax_x64` flips `jax_enable_x64` process-wide (warned, correctly).
  Two `design_optimize` calls with different `precision` in different threads of one
  process would interleave; I did not build the race.

## Checked and found correct

* **Zemax EVENASPH mapping and units.** `PARM n ↔ r^(2n)` and
  `a_m = a_file / unit_scale^(power−1)` are exact in both directions
  (`prescriptions_zemax.py:928-934` / `2262-2270`). Measured on
  `PARM 1..3, 8 = 1e-3, −1.234e-6, 5.678e-10, 1e-20` at r = 5 mm:
  sag(lib) = 0.024237623 mm, sag(Zemax-native) = 0.024237623 mm; A4 → ×1e9 and
  A6 → ×1e15 exactly as specified.
* **UNIT handling**: MM/CM/M/IN/INCH/INCHES all correct (`R1 = 1.59385 m` for
  `UNIT IN`, `0.6275 m` for `CM`), and an unrecognised token warns before falling back
  to mm.
* **UTF-16-LE with BOM** is detected and the BOM stripped: radii identical to the UTF-8
  copy to 0.0.
* **Semi-diameter → `aperture_diameter`**: `DIAM 12.7` on the STOP → `0.0254 m`. ×2 is
  right, and the `.txt` loader correctly divides the "Clear Diam" column by 2 instead.
* **`CURV` → radius** with unit scaling, and the deliberate ignoring of the trailing
  solve fields on the `CURV` line.
* **COORDBRK**: `PARM 1..6` → `decenter_x_m`, `decenter_y_m`, `tilt_x_deg`, `tilt_y_deg`,
  `tilt_z_deg`, `order`, decenters × `unit_scale`, tilts left in degrees — matches
  CONVENTIONS §7 and `raytrace/world.py::_apply_coord_break` (local-to-world, intrinsic
  X→Y→Z, `+90° tilt_x` puts local `+z` at world `−y`). The `PARM 6` order flag is
  honoured in both the loader's `_dgrating_frame` SE(2) composition and `world.py`.
* **ZX-1 coord-break DISZ folding**: a CB between two lens surfaces contributes its
  axial gap to the preceding element (measured: 30 mm + 7 mm CB → `all_thicknesses`
  entry 37.000 mm).
* **STOP on a COORDBRK** is reassigned to the next optical surface with a warning.
* **Glass names do NOT silently alias.** `BK7`, `SF11`, `SF6`, `BAF10`, `SCHOTT_N-BK7`
  are all absent from `GLASS_REGISTRY` and `get_glass_index` raises `ValueError` rather
  than resolving to the `N-` variant. `SILICA`→`FUSED_SILICA` is a genuine documented
  alias with an identical index. No P0 here.
* **`system_abcd` and the loader agree with an independent analytic paraxial trace.**
  `AC254_100_C.zmx` → `system_abcd` EFL = 84.1429 mm, hand-rolled surface-by-surface
  ray transfer = 84.1429 mm; `LA1509_C.zmx` plano-convex → 205.1102 mm vs the closed
  form `R/(n−1)` = 205.1102 mm. *Note on the brief's reference value*: R1 = 62.75,
  R2 = −45.71, R3 = −128.23 mm with N-BAF10/N-SF6HT gives EFL = **72.2154 mm** by exact
  paraxial algebra at 587.6 nm, not 100.1 mm — that reference triple is internally
  inconsistent, and the library reproduces the algebra exactly.
* **`scale_prescription` core identities**: `scale(s)∘scale(1/s)` exact to 0.0 on radii
  and thicknesses; the aspheric rule `A_n → A_n/s^(n−1)` gives
  `sag(s·h)/sag(h) = 0.25` exactly at `s = 0.25`; inf radii preserved; conics, tilts,
  glass names and wavelength correctly not scaled; coord-break decenters and
  `thickness_m` correctly scaled.
* **`make_off_axis_parabola` geometry**: `R = 2f`, `k = −1`, `h = 2f·tan α` re-derived
  from `z = r²/4f` (`dz/dr = tan α` is the surface-normal angle; the 90°-fold case
  `α = π/4 → h = 2f` checks against the reflected ray hitting `(0,0,f)` perpendicular
  to `z`). Validation of `vertex_radius` and `off_axis_angle ∈ (0, π/2)` present.
* **Quadoa aspheric unit round-trip**: `A_file = A_m · scale^(1−p)` on write and
  `A_m = A_file · inv_scale^(1−p)` on read are exact inverses for every power.
* **Multi-process HDF5 append is race-free.** 2 processes × 50 appends to one file with
  the `filelock` path: **100 planes, 100 unique labels, 0 duplicates, 0 missing**. The
  bump-before-create + rollback + orphan-delete discipline works as documented.
* **HDF5 metadata codec**: 18 of 19 probe types round-trip exactly through
  `append_plane_h5`, `save_field_h5` and `write_sim_metadata` — including `None`,
  `bytes`, `complex`, `tuple`, empty and heterogeneous lists, nested dicts, NaN, ±inf,
  and complex64 ndarrays. The one miss (`np.float32` → Python `float`) is inherent to
  the JSON lowering and harmless.
* **`preserve_dtype=True`** keeps complex64 through HDF5 and Zarr round-trips with
  `max|Δ| = 0`.
* **`_merit_jit` numba vs NumPy parity**: `max|Δ| = 3.14e-16` at N = 256 (JIT engaged),
  `0.0` at N = 64 (below the threshold, NumPy path). The 1e-12 contract holds.
* **`jax.grad` vs finite differences**: max relative error **2.35e-9** on a
  3-parameter analytic singlet merit — FD-limited, i.e. the JAX path is correct.
  `JaxMeritTerm.evaluate` and `gradient_at_x` reduce the same way (`|·|` or `real`).
* **`_fd_grad_pure`**: central/forward schemes, per-variable `eps·max(|x|, scale_floor)`
  step, bounds clipping with the shrunken-span quotient, one-sided fallback at a pinned
  bound, and the degenerate-box guard are all correct; the `f0` staleness contract is
  documented and `validate_f0` is available.
* **`_wrapper_merit_aperture_key` / `_get_wrapper_merit_cache`** key on
  `(Ny, Nx, dx, aperture-content, dtype)` with the grid arrays shared by reference and a
  double-checked publish under one lock — correct, and the zero-aperture sentinel
  correctly distinguishes "no aperture" from "block everything".
* **`MinThicknessMerit` / `MaxThicknessMerit` glass-vs-air classification**: gap `i`'s
  medium is `surfaces[i]['glass_after']`, which is the right slot.
* **`multi_objective` constraint translation**: `g_lb = lb − f(x) ≤ 0` and
  `g_ub = f(x) − ub ≤ 0` are the correct pymoo `G ≤ 0` forms.
* **`create_zoom_configs` does not mutate the template** (deep-copies per config), so
  the "applied and restored" concern does not arise.
* **No `eval` / `exec` on any load path** in `io/` (codegen only *writes* text; the
  injection finding is about that written text).
