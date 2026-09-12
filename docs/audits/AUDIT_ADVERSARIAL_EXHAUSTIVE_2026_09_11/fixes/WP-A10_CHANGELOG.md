# WP-A10 changelog text — `lumenairy/io/`, `lumenairy/optimize/`

Findings I1–I8 of `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11.md` §7
(partition report `IO-OPTIMIZE.md`).

---

### Fixed -- io/CODE V: `DIM M` is MILLIMETRES, and `C` / `I` are recognised (I1, P0)

`load_codev_seq` read CODE V's `DIM M` as METRES and ignored the `C` and `I`
tokens entirely, falling through to that same metre default; `export_codev_seq`
mirrored it, so every file the library wrote with the default `units='M'` was
1000x too large when CODE V opened it — while the docstring advertised the
kwarg as "useful for handing files to CODE V users".  CODE V's `DIM` command
takes exactly three lens-unit tokens — `M` (millimetres), `C` (centimetres),
`I` (inches); the language has no metre unit.

Measured on the AC254-100 doublet written as an ordinary CODE V sequence
(`DIM M`, `RDY 62.75`, …), `repro/IO-OPTIMIZE/p2_codev.py`:

| file says | before | after | CODE V means |
|---|---|---|---|
| `DIM M` | R1 = 62.75 m, EFL = 72.2154 m, **0 warnings** | R1 = 0.06275 m, EFL = 0.0722154 m | 62.75 mm |
| `DIM C` | R1 = 62.75 m (token ignored) | R1 = 0.6275 m | 62.75 cm |
| `DIM I` | R1 = 62.75 m (token ignored) | R1 = 1.59385 m | 62.75 in |
| unknown token | silent fall-through to the default | `UserWarning` naming the token and the unit in force | — |

The writer now emits CODE V lens units (`RDY 50.00000000` for a 50 mm radius,
was `RDY 0.05000000`) and normalises the tolerated aliases `MM`/`CM`/`IN`/`INCH`
to the CODE V token they mean, so an exported file never carries a `DIM` token
CODE V does not define.  `export -> load` stays a lossless identity
(max |ΔR| = 0.0 on a doublet).

**Migration.** Files written by lumenairy **before this release** with the
default `units='M'` carry `DIM M` with SI-metre numbers.  The writer now stamps
a format marker (`! LUMENAIRY-SEQ-FORMAT 2`) next to its generator banner, and
`load_codev_seq` uses the pairing *banner present + marker absent + a `DIM M`
line* to identify such a file exactly: it is read with the legacy metre scale
(so its values are unchanged) and raises a `UserWarning` saying so.  Re-export
it once to get a file CODE V can read, or pass the new `dim_units=` kwarg
(`'M'`/`'C'`/`'I'`/`'SI'`) to force a reading.  Files from CODE V itself, and
files written from this release on, are unaffected.

A pre-release file written with `units='MM'` or `units='IN'` needs **no**
migration and gets no warning: those tokens meant millimetres and inches then
and still do, so such a file reads identically under both conventions.  (The
legacy sniff is deliberately keyed on the `DIM` token for this reason — keying
it on the banner alone read a pre-release `DIM MM` file 1000x too large and a
`DIM IN` file 39.37x too large, which VERIFY-A10 caught and fixed before
release.)

Files: `lumenairy/io/prescriptions_code_v.py`.
Fixtures corrected (both pinned the wrong convention):
`tests/unit/test_audit_misc.py::test_d2_codev_seq_roundtrip_preserves_bfl`
(hand-written `.seq` that wrote SI metres under `DIM M`) and
`validation/io/test_io.py::t_codev_seq_units_mm` (asserted the writer emits
`DIM MM`, a token CODE V does not define).
Tests: `tests/unit/test_audit2609_a10_codev.py` (19 tests),
`validation/io/test_io.py` (`CODE V .seq: DIM M/C/I scaling + unknown-token
warning`).

### Fixed -- io/CODE V: mirrors, conics and aspheres are parsed instead of dropped (I3)

`load_codev_seq`'s per-surface dispatch handled only `STO`/`RDY`/`CUY`/`THI`/
`GLA`/`CON`; everything else fell into the silent-ignore tail.

* `REFL` / `RMD REFL` (mirrors) were in neither warn set, so a CODE V fold or
  catadioptric mirror imported as an **air-to-air dummy surface** — the design
  silently un-folded *and* lost the mirror's optical power.  Measured before:
  `glasses [('air','N-BK7'), ('N-BK7','air'), ('air','air')]`, `'elements' in
  rx` False, `has_mirrors` False.  After: `element_type` `['surface','mirror',
  'mirror']`, `has_mirrors` True, and the writer emits `REFL` so a folded
  design round-trips.
* `K` (CODE V's conic keyword) was ignored — only the exporter's own `CON` was
  accepted.  Measured before: `conics [0.0, 0.0, 0.0]` for a file carrying
  `K -1.0`, with no mention even though an `ASP` warning did fire.  Both
  keywords are now read; the writer emits `K`.
* `A`…`J` (4th…20th-order asphere coefficients, CODE V skips the letter `I`)
  were dropped.  Now parsed into `aspheric_coeffs` with the lens-unit → SI rule
  `a_p = a_file / L**(p-1)` (the same rule the Zemax loader uses): measured
  `A 1.234E-07` → 123.4 m^-3 and `B -5.0E-11` → -5.0e4 m^-5, exact.  The writer
  emits the matching `ASP` + `A`…`J` block and warns for any power outside it.
* A surface with no `RDY` yielded `radius=None`, not `inf`: the dict was
  pre-seeded with `'radius': None` so the documented `s.get('radius',
  float('inf'))` default was dead code.  A legal CODE V dummy surface therefore
  produced a prescription `validate_prescription` rejects
  (`surfaces[1].radius: is None`).  Now `inf`, and `system_abcd` accepts it.

### Fixed -- io/Zemax: powered air-to-air surfaces are no longer deleted by the window auto-detect (I2)

The lens-window auto-detect admitted only glass / mirror / DGRATING surfaces,
so a `TYPE PARAXIAL` ideal lens, an air-spaced phase surface, or an `ABCD`
black box — and the STOP flag on it — were removed **before** reaching the
unsupported-SURFTYPE branch that would have warned.  Measured on `PARAXIAL
f = 100 mm` + STOP ahead of a glass singlet: `elements surf_nums = [2, 3]`,
`stop_index = None`, **0 warnings** — the 100 mm lens and the declared stop
were both gone and only the lens's DISZ survived as free space.  This is the
same failure the v5.32 DGRATING fix repaired, applied to one surface type only.

The air-to-air powered SURFTYPEs that carry their power in a `PARM` table or a
curvature -- `PARAXIAL` / `PARAXIALXY` / `IDEAL` / `ABCD` / `BINARY_*` /
`GRID_PHASE` / `ZERNPHASE` / `HOLOGRAM*` / `TILTSURF` -- now enter the window,
so they reach the loud per-surface unsupported-SURFTYPE warning instead of
vanishing, and their STOP and DIAM survive.  An air-spaced **aspheric** phase
plate (`EVENASPH` / `QBFS` / `QCON` / `ZERNSAG` / `GRID_SAG` outside the glass
span) is still excluded -- mapping those onto the window would change which
surfaces an ordinary aspheric design imports -- but it is no longer silent:
any optical surface the window excludes that carries a non-zero `CURV`, a
non-empty `PARM` table or the STOP flag is named once in a `UserWarning`.  An
ordinary doublet gains no new diagnostic.

Both Zemax loaders share the predicate and the diagnostic.
`load_zemax_prescription_data_txt` is a near-copy of the `.zmx` pipeline and
had the same defect; it now calls the same `_raw_surface_is_air_powered` and
`_warn_window_excluded_powered` helpers, so the two cannot drift.  (Measured
on a `PRESCRIPTION DATA` report with a `PARAXIAL` STOP row ahead of the glass:
2 surfaces / `stop_index None` / 0 warnings -> 3 surfaces / `stop_index 0` /
the P3-43 shape warning.)

### Fixed -- io: cylindrical / biconic surfaces no longer export as spheres in silence (I4)

`export_zemax_zmx`, `export_zemax_lens_data` and `export_codev_seq` emit
curvature from `radius` only, so `radius_y` / `conic_y` / `aspheric_coeffs_y`
were dropped and a one-axis focusing element became a two-axis one with no
diagnostic (measured: `cyl S0 radius=0.05 radius_y=inf` → `.zmx`/`.seq`
`radius_y preserved=False, warnings=[]`; the Quadoa writer handles it, which
makes the gap an inconsistency rather than a format limitation).  All three
writers now raise a `UserWarning` per anamorphic surface, as loudly as
`_warn_dropped_qtype` does for a Forbes-Q surface.  A surface whose `radius_y`
equals its `radius` loses nothing and stays quiet.

### Fixed -- io/codegen (security): untrusted `.zmx` strings can no longer reach code positions (I5)

`generate_simulation_script` interpolated the raw `GLAS` token, the labels
derived from it, and the system name (the file stem by default) into **code**
positions of the generated script with no escaping — `la.GLASS_REGISTRY['{g}']
= …`, `print("Applying {label} …")`, `print('Running: {sys_name}')`.  A
whitespace-free `GLAS` token such as `X'];<payload>;#` therefore became live
code that ran the moment the user executed the generated file; the audit's
proof of concept emitted a line parsing as **2 statements** and the full script
parsed as valid Python.

Two independent guards now stand between the file and the script: every
interpolated string goes through `repr()` / a comment-safe collapse, and
glass / name tokens are validated at the boundary against
`[A-Za-z0-9_\-.+]+` (anything else is stripped, with a `UserWarning` that says
an out-of-charset token in a third-party `.zmx` is an injection attempt).
After: every emitted `GLASS_REGISTRY` line parses as exactly one `Assign` whose
subscript key is a string constant, and executing the emitted registry block
against a stub registry produces no output at all.

Threat model, unchanged from the audit: `generate_script_from_zmx()` on a
vendor / colleague-supplied `.zmx`, or one fetched from a lens-catalogue site.
There is still no `eval`/`exec` on any load path.

### Fixed -- io/storage: Zarr per-plane metadata goes through the canonical codec (I6)

`_zarr_append_plane` was the one metadata write site still using the pre-A-4
raw loop, while every HDF5 site and `_zarr_write_sim_metadata` used the
type-tagged `_meta_dumps` codec — so the same call with the same arguments had
different fidelity depending on the global backend switch.  Measured over the
module's own 19-type probe set (`repro/IO-OPTIMIZE/p6_storage.py`):

```
             before            after
HDF5         18/19 faithful    18/19
ZARR         13/19 faithful    18/19      (only np.float32 -> float remains,
                                           inherent to the JSON lowering)
```

The five recovered types were `complex` → `str '(1+2j)'`, `bytes` → `str`,
`tuple` → `list`, `np.float32` → `str`, and — irrecoverably — `ndarray` →
`str(...)`, which inserts `...` past numpy's 1000-element print threshold, so
the values were simply gone.  A 4096-element array now round-trips exactly.
`_zarr_load_planes` / `_zarr_list_planes` / `_zarr_load_plane_by_label` /
`_zarr_load_plane_slice` apply the same blob overlay `_h5_read_attrs` applies;
files written before the blob existed read back exactly as before.

### Fixed -- io: `THORLABS_CATALOG['LA1509-C']` was a 200 mm lens under a 100 mm part number (I6)

The entry carried `R1 = 103.29 mm` — the radius of a 200 mm lens — with
LA1509's own 3.6 mm thickness, so `thorlabs_lens('LA1509-C')` returned a
**2x focal-length error with no diagnostic**, and the validation fixture
`validation/real_lens_opd/zemax_prescriptions/LA1509_C.zmx` carried the same
wrong radius (`CURV 0.0096814793`) so it could not catch it.  Thorlabs LA1509
is R = 51.5 mm, tc = 3.6 mm, N-BK7, f = 100.0 mm.

Measured paraxial EFL at 587.6 nm: **199.865 mm → 99.652 mm** (nominal 100.0 mm, i.e. the pre-fix value was 1.9987x the part number);
at 1310 nm 205.11 mm → 102.27 mm.  The `.zmx` and `.txt` fixtures and the
`INDEX.md` row were regenerated from the corrected entry (`CURV 0.0194174757`),
and `validation/real_lens_opd/lens_cases.py`'s description ("f=200 mm") was
corrected.

### Added -- io: every Thorlabs catalogue row is checked against its own part number (I6)

`thorlabs_lens` now re-derives the paraxial EFL from the row's radii /
thicknesses / glass with an independent reduced-slope trace and raises a
`UserWarning` when it misses the focal length the part number states by more
than 3 %.  Audit of the whole table at 587.6 nm:

| part | nominal | measured | deviation |
|---|---:|---:|---:|
| LA1050-C | 100 mm | 99.652 mm | −0.35 % |
| LA1509-C | 100 mm | 99.652 mm | −0.35 % (pre-fix: 199.865 mm, **+99.9 %**) |
| LA1301-C | 250 mm | 250.001 mm | +0.0004 % |
| **AC254-050-C** | 50 mm | 44.560 mm | **−10.9 %** |
| **AC254-100-C** | 100 mm | 83.171 mm | **−16.8 %** |
| **AC254-200-C** | 200 mm | 137.395 mm | **−31.3 %** |

The three doublet rows are left as data (no vendor surface table was available
to correct them in this pass) but `thorlabs_lens` now warns about them -- once
per part per process, so a loop over a catalogue is not drowned in repeats.
The header comment's claim "Surface data from Thorlabs Zemax files" is false
for them.  **They need vendor data.**

### Fixed -- io: `scale_prescription` is self-similar for Forbes-Q, diffractives and the stored BFL (I7)

Three families of LENGTHS appeared in neither the "scales" nor the "deliberately
does not scale" half of the docstring, so their absence read as coverage.
Measured at `s = 0.25` (`repro/IO-OPTIMIZE/p7b.py`), before → after:

```
r_max                 0.0075  -> 0.001875        (expected 0.001875)
q_bfs_coeffs    [1e-06, 2e-07] -> [2.5e-07, 5e-08]
back_focal_length     0.084   -> 0.021
diffractive period    2e-06   -> 5e-07
diffractive gap_before 0.01   -> 0.0025
```

`origin`, `gap_after`, `semi_diameter` and `lines_per_um` (inverse) scale too;
`order` deliberately does not.  The core identities the audit verified correct
are unchanged (round-trip max |ΔR| = 0.0, aspheric self-similarity exactly
0.25).  Any *unrecognised* length-like key now raises a `UserWarning` instead of
being left at its original size in silence.

**Note.** Scaling a grating `period` is what geometric self-similarity requires,
but the wavelength is deliberately not scaled, so a scaled DOE diffracts at a
different angle — documented explicitly in the docstring.

### Fixed -- io/codegen: `inf` / `nan` emission, `-inf` sign, and `normalize_prescription` output (I7)

* The generated prescription block emitted bare `inf` / `nan` (not Python
  literals) for conic and aspheric values and failed with `NameError: name
  'inf' is not defined`, and `np.isinf` being sign-blind wrote a `radius =
  -inf` surface as `float('inf')` — **a silent sign flip**.  One shared
  `_py(value)` helper now renders radius, conic, thicknesses, aperture and
  every aspheric coefficient; the emitted block executes and
  `LENS_1_RX['surfaces'][1]['radius']` comes back `-inf`.
* `generate_simulation_script` raised `KeyError: 'element_type'` on the output
  of `normalize_prescription` — the one helper documented as "the recommended
  idiom" for making a builder prescription codegen-shaped.  codegen now
  defaults the key, and `normalize_prescription` stamps
  `element_type='surface'` on the mirrored entries **in place**, so the
  documented `q['elements'] == q['surfaces']` identity (and the aliasing it
  rests on) is preserved — both views now carry the canonical discriminator
  instead of neither.
* One level up, a prescription straight from a **builder** (`make_singlet` /
  `make_doublet` / `combine_prescriptions`) raised a bare
  `KeyError: 'elements'` from `_decompose_prescription`.  The message now
  carries the CONVENTIONS §2 `generate_simulation_script:` prefix, names the
  missing keys and the keys the prescription does have, and gives the one-line
  fix (`generate_simulation_script(la.normalize_prescription(rx), ...)`).
* Two `#` comment lines in the generated script were built from file-supplied
  text without the comment collapse every other comment site uses — the
  per-lens header (whose text is the first surface's `COMM`) and the
  `style='system_list'` DOE placeholder.  A `COMM` carrying U+2028 / U+2029 /
  U+0085 therefore reached the emitted file verbatim: inert under CPython,
  which does not treat them as source newlines, but any tool that re-reads the
  script with `str.splitlines()` disagreed with the compiler about where the
  comment ends.  Both now route through `_comment_text`, which also strips
  those three separators explicitly.

### Added -- io/Zemax: multi-configuration (`MNUM` / `MCON`) records are surfaced (I7)

Neither keyword was matched at any level, so a zoom / thermal / multi-config
`.zmx` imported as the base lens-data-editor state with no `configurations` key
and **no warning**, while `optimize/multiconfig.py` and
`examples/08_multiconfig_zoom.py` make the feature look supported end to end.
`load_zemax_zmx` now returns `configurations` — `None` for a single-config file
(the overwhelming majority), else the `MNUM` header plus the raw `MCON` operand
rows — and warns once naming the operands found.

Each operand row is `{'raw', 'operand', 'fields_provisional'}`.  **Only `raw`
and `operand` are contractual.**  The positional layout of an `MCON` row after
the operand token is version-dependent and could not be confirmed against an
OpticStudio-written file, so the trailing fields are exposed as an undecoded
list rather than under invented names: an earlier named decode assigned
`config = 3.0` to all three rows of a three-configuration fixture, which cannot
be right.  Read `raw`; the docstring says so, and the warning points at it.

### Fixed -- io: CODE V and Quadoa loaders emit the schema three docstrings promised (I7)

`normalize_prescription` and `split_prescription_at_mirrors` both documented
`load_codev_seq` / `load_quadoa_qos` as carrying `elements` and
`all_thicknesses`; neither did (measured key diff for the same doublet:
`.zmx` 10 keys vs `.seq` 7).  Both loaders now emit them, and
`split_prescription_at_mirrors` warns when it takes the
`elements is None` early return instead of silently reporting one refractive
leg — which is indistinguishable from a genuinely unfolded design.

Both loaders also gained the unknown-glass `UserWarning` the Zemax loaders have
had since v4.16.1; CODE V names are routinely `N-BK7_SCHOTT` or a 6-digit code,
neither of which is a `GLASS_REGISTRY` key, and the failure previously surfaced
much later as a `ValueError` from `get_glass_index` inside a propagation.

### Fixed -- io/Quadoa: the writer no longer invents an aperture stop (I8)

`stop_surface` defaulted to `0` when the prescription declared none, and every
surface was then written with `is_stop = (i == 0)`, so re-loading **always**
yielded `stop_index = 0`.  The default is now `None` and no `is_stop` flag is
written unless a stop was actually declared; a declared stop still round-trips.

### Changed -- io/storage: HDF5 `compression` and `chunk_size` default to `'auto'` (I7, Performance)

`compression='gzip', compression_opts=4` was the default on `save_field_h5`,
`save_planes_h5`, `save_jones_field_h5` and `append_plane_h5` — the last of
which is the per-plane hot path of every multi-plane run.  Complex float
mantissas are incompressible, so gzip is almost pure overhead there.  Measured
on this workstation, 1024² complex128 (16 MiB), **medians of 7 interleaved
runs** (interleaved because the box is shared):

| compression | write | read | on disk | tracemalloc peak |
|---|---:|---:|---:|---:|
| `gzip` level 4 (old default) | 0.4274 s | 0.0867 s | 15.12 MiB | 1.8 MiB |
| `None` | 0.0074 s | 0.0122 s | 16.01 MiB | 0.0 MiB |
| ratio | **×57.9** | **×7.1** | −5.6 % | — |

(The audit measured the same shape at 4096²: 17.05 s vs 0.70 s for −5.6 %.)

`'auto'` means **no compression for complex data, gzip level 4 for everything
else** — so real-valued arrays keep the historical behaviour exactly, and an
explicit `compression='gzip'` / `'lzf'` / `None` still does what it says.
`chunk_size='auto'` picks the largest power-of-two edge whose chunk is ≤ 1 MiB
(256 for complex128); the old fixed `1024` made a **16 MiB** chunk for
complex128, sixteen times HDF5's 1 MiB default chunk cache, so every partial
read touched a whole chunk.  Stored values are unchanged — this is a layout
change only.

`append_plane` on a `.zarr` store now warns that `compression` /
`compression_opts` are dropped (the Zarr path has no compression parameter at
all), instead of ignoring the request silently.

**Migration.** New files written by the four functions above are larger by
~5.6 % for complex fields and are not gzip-filtered; existing files are
unaffected and still read.  Pass `compression='gzip', compression_opts=4` to
restore the old default.

### Performance -- optimize: the 31-plane through-focus scan runs only when a merit reads it (I7)

`design_optimize`'s wave leg ran `through_focus_scan` over `z_scan_n = 31`
planes on **every** merit evaluation, against the leg's one lens propagation —
and on the default `jac='auto'` path without a `JaxMeritTerm` scipy
finite-differences the merit, so every gradient paid it n+1 times.  The audit
measured 17 merit evaluations → **527 focus-scan slices**, i.e. 97 % of the
wave-leg work, recomputed from scratch for every FD probe **even when no merit
read `strehl_best` / `z_best` / `rms_radius_best`**.

A `needs_focus_scan` flag (default `True`, so a user-written merit class that
predates it is unaffected) joins the existing `needs_wave` / `needs_ray` gates.
The library's own wave merits declare it accurately: `StrehlMerit` and
`SpotSizeMerit` read the scan; `RMSWavefrontMerit`,
`MatchIdealThinLensMerit`, `MatchIdealSystemMerit`, `MatchTargetOPDMerit` and
`ZernikeCoefficientMerit` read `ctx.opd_map` (built from `ctx.bfl`) and do not.
`CompositeMerit` / `NormalizedMerit` / the three wrapper merits forward the
requirement from the merit they wrap.

Measured on a 1-variable, N = 64 `RMSWavefrontMerit` run (3 merit evaluations):
**3 scan calls / 93 propagation slices → 0**, i.e. the wave leg drops from
1 + 31 = 32 propagations per evaluation to 1 (**×32**) for that merit family.
A `StrehlMerit` run is unchanged (31 slices per call, byte-identical results).

### Added -- optimize: `MinEdgeThicknessMerit` and `edge_thickness` (I7)

No edge-thickness constraint existed anywhere in the library: `MinThicknessMerit`
/ `MaxThicknessMerit` penalise the **centre** thickness only, while
`MinThicknessMerit`'s "Minimum acceptable GLASS thickness [m]" docstring reads
as manufacturability coverage — and a repo-wide grep for
`edge_thickness|EdgeThickness|edge thickness` hit only two incidental comments.
Edge thickness is the constraint an unconstrained radius optimisation violates
first (every commercial code ships it: `ETGT`/`ETVA` in OpticStudio, `ETH` in
CODE V).

`edge_thickness(prescription, slot, semi_diameter=None)` returns
`t_c + sag(R2, k2, h) - sag(R1, k1, h)`, verified against an independently
written closed-form spherical sag to |Δ| ≤ 5.2e-18 m on four shapes including
the concave-first meniscus where the two sags have the same sign and partly
cancel (the case a `t_c - |sag1| - |sag2|` formula gets wrong: +1.238 mm vs
−6.5 mm).  `MinEdgeThicknessMerit(min_edge, weight, semi_diameter,
include_air)` sums `max(0, min_edge - t_edge)²` over the glass slots.
Exported from `lumenairy.optimize`; a top-level re-export is requested in the
WP report.

A slot whose edge thickness is not finite — one of its surfaces does not reach
the clear semi-diameter at all, so there is no edge to measure — counts as a
**maximal** violation (`deficit = min_edge`), not as satisfied.  Scoring it 0
rewarded exactly the geometry the constraint exists to prevent, which is the
shape S4-5 fixed for a NaN `strehl_best` in `driver.py`.  Measured on a
R = 8 mm surface evaluated at h = 10 mm: contribution 0.0 → 1e-6 (= `min_edge²`
at `min_edge = 1 mm`), while a well-defined knife edge still scores its own
larger deficit (1.15e-4) and a healthy element still scores exactly 0.

### Fixed -- optimize: `design_optimize_multi_objective` refuses an infeasible run (I7)

pymoo sets `Result.X` / `.F` to `None` when the final population is entirely
infeasible.  `np.asarray(None, dtype=np.float64)` is `array(nan)` with
`ndim == 0`, so the `if X.ndim == 1` normalisation did not fire and a **0-d NaN
array was returned as the Pareto front**; with `progress` supplied,
`X.shape[0]` raised `IndexError: tuple index out of range` instead.  The
function now raises a `ValueError` naming the generation count, the smallest
constraint violation in the final population, and the knobs that fix it.
(pymoo is an optional dependency and is not installed on this machine; the
numpy half is measured directly and the pymoo half is exercised through a
`sys.modules` stub reproducing the documented contract.)

`x0` is now bounds-checked, as the docstring has always claimed ("used to infer
`n_params` and as a sanity check against `bounds`") — pre-fix only `n_params`
was read.

### Fixed -- optimize: `create_zoom_configs` no longer writes glass thicknesses or truncates silently (I8)

The parameter is named `zoom_spacings` and documented as "the air-gap
thicknesses", but the loop was positional over the whole `thicknesses` list and
`if j < len(...)` dropped extra entries with no warning — so a mis-sized zoom
vector silently produced a system nobody asked for, and slot 0 of a cemented
doublet's vector overwrote the glass centre thickness.  A flat spacing list must
now match the slot count exactly (`ValueError` otherwise), and a new
`(slot_index, value)` pair form writes only the named slots.
`examples/08_multiconfig_zoom.py` is unaffected (its vectors already match).

### Fixed -- optimize: `method='newton'` says that it drops bounds (I8)

`trust-ncg` accepts no `bounds`; only the finite-difference stencil is clipped,
so a bounded Newton run looked bounded and was not.  The generic `minimize`
branch and the `lm` branch both warn loudly in exactly this situation; the
Newton branch now does too.

### Fixed -- optimize: the ndarray aperture cache key is a content digest (I8)

`_wrapper_merit_aperture_key` keyed an ndarray aperture on Python's 64-bit
`hash(arr.tobytes())`, so a collision returned the **wrong** cached aperture
mask.  Astronomically unlikely, but a 128-bit `blake2b` digest costs the same
single byte-scan and removes the class entirely — and a cache key that is not a
pure function of its input by value is the pattern §15.5 of the audit flags
across the library.

### Fixed -- io/Zemax: `.zmx` record injection via `NAME` / `COMM`, and UTF-16-BE files (I8)

`export_zemax_zmx` wrote `NAME` and `COMM` verbatim, so a newline inside a
prescription name injected arbitrary `.zmx` records — measured: a name of
`'bad\nSURF 99\n  CURV 0.5\nNAME x'` produced an extra `SURF` row, and the file
reloaded as a different system.  `NAME` (both writer paths) and `COMM` (the
`elements` path, the only one that emits it) are now collapsed to one line with
control characters and the `"` field delimiter replaced, warning when anything
changed; the record count is back to 4 on a singlet.  `export_codev_seq` does
the same for its `! title` comment and its `GLA` tokens.

The `.zmx` encoding ladder now tries BOM-sniffing `utf-16` first, so a
big-endian export is decoded by its BOM instead of falling through to `latin-1`
and being reported as "does not appear to be a Zemax .zmx lens file" — a
diagnostic that pointed at the wrong cause.

---

Kept intact (re-verified, not changed): EVENASPH mapping and unit scaling,
`UNIT` MM/CM/M/IN handling with its unknown-token warning, UTF-16-LE BOM
detection, semi-diameter ×2, COORDBRK `PARM 1..6` → decenter/tilt/order, CB
DISZ folding, STOP-on-CB reassignment, `system_abcd` vs an independent paraxial
trace, `scale_prescription`'s core identities, multi-process HDF5 append, the
HDF5 metadata codec, numba-vs-NumPy merit parity, `jax.grad` vs FD, and the
absence of `eval`/`exec` on every load path.
