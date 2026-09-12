# VERIFY-A10 — adversarial re-verification of WP-A10 (`lumenairy/io/`, `lumenairy/optimize/`)

Findings I1–I8 of `AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11.md` §7, partition report `IO-OPTIMIZE.md`.
Work package under test: commit **`658e6142`** (`git diff 658e6142^ 658e6142`), branch `audit-fixes-2026-09`.
Environment: CPython 3.14.6, numpy 2.4.6, scipy 1.17.1, h5py 3.16.0, zarr 3.2.1, jax 0.10.1, **pymoo absent**.
Every python invocation used `OPENBLAS_NUM_THREADS=1`.  No git write commands were run.

**Verdict: 26 of 28 claims hold on re-measurement.  Two are regressions the work package introduced;
both are fixed here, in WP-A10's own files, with fail-before-pass-after regression tests.**

**Update 2026-09-12 — every open item below has been ruled on by the coordinator and implemented
(V4 excepted, which is assigned elsewhere).  See §10; the summary table's VERIFIED-WITH-NOTES rows
and §5's wording notes are superseded by the fixes recorded there.**

---

## 0. Summary

| # | claim | verdict | evidence |
|---|---|---|---|
| I1 | `DIM M/C/I` read as mm/cm/inch | **VERIFIED** | genuine hand-written `.seq` in all three units, exact to 0.0 rel vs the unit definitions; EFL matches an independent reduced-slope trace to 0.0 rel |
| I1 | writer emits CODE V lens units | **VERIFIED** | `RDY 62.75000000` under `DIM M`; export→load max\|ΔR\| = 0.0 |
| I1 | legacy-file migration marker | **REGRESSION — fixed here** | a pre-v5.46 `units='MM'` file read **×1000** too large, `units='IN'` **×39.37**; marker matched by exact string, so a format bump would re-break every file |
| I2 | powered air-to-air surfaces survive the window | **VERIFIED** (after §10/V7) | PARAXIAL + its STOP are kept (`surf_nums [2,3] → [1,2,3]`, `stop_index None → 0`).  EVENASPH / QBFS / QCON / ZERNSAG / GRID_SAG outside the glass span are still excluded **by design**, now stated as such in the changelog, and named in a warning.  The `.txt` twin loader had neither half and now shares both helpers — see §10 |
| I3 | CODE V `REFL`/`RMD`/`K`/`A…J`/`radius` | **VERIFIED** | new inch fixture (parabola + A/B/J up to y²⁰ + fold mirror in air): coefficients exact to 0.0 rel vs the lens-unit rule, sag oracle 0.0 rel, load→export→load 0.0 on every element |
| I4 | anamorphic export warns | **VERIFIED** | all three writers warn on `make_cylindrical`, all three silent on a sphere |
| I5 | codegen injection closed | **VERIFIED** | 7 payloads of my own (quote-escape, both quote styles, backslash, U+2028, `"""`, format-spec): 0 landed on a code position by an `ast` oracle; ordinary file still generates a parsing script |
| I6 | zarr metadata through the codec | **VERIFIED** | repro 13/19 → **18/19**; plus 16 exotic payloads (4096-element arrays, int64/bool/complex128 arrays, 256-byte `bytes`, −0.0, 2⁶², unicode) faithful through `append_plane` / `list_planes` / `load_plane_by_label`, **identically on both backends** |
| I6 | `LA1509-C` = 100 mm lens | **VERIFIED** | independent 2×2 ABCD product: 199.8652 mm → **99.6520 mm**; closed form `R/(n−1)` = 99.652004 mm; `.zmx` / `.txt` / catalogue agree |
| I6 | other catalogue rows warn | **VERIFIED** (after §10/V8, V9) | deviations reproduce (−10.88 / −16.83 / −31.30 %); the warning is once per process per part (the changelog now says so) and the stale `199.68` in-code number is corrected to the measured `199.8652 mm` |
| I7 | `scale_prescription` coverage | **VERIFIED** | repro `p7b`: `r_max` 0.001875, `q_bfs` [2.5e-7, 5e-8], BFL 0.021, period 5e-7, `gap_before` 0.0025 — all as claimed |
| I7 | codegen `inf`/`nan`/`−inf` | **VERIFIED** | emitted block executes; radius comes back `-inf`, conic `inf` |
| I7 | `normalize_prescription` → codegen | **VERIFIED** | pre-fix tree: `KeyError: 'element_type'`; post-fix: script generated, `q['elements'] == q['surfaces']` still `True`, both carry `element_type='surface'` |
| I7 | `MNUM`/`MCON` surfaced | **VERIFIED** (after §10/V6) | key + warning present; the unconfirmed `config`/`surface` decode is withdrawn — each row is now `{'raw', 'operand', 'fields_provisional'}` with `raw` documented as the contract |
| I7 | HDF5 `compression='auto'` | **VERIFIED** | filter pipeline and chunk shape read back out of the file: complex → `None`, real → `gzip`; chunks 1.000 / 0.500 MiB; values bit-identical; explicit `gzip`/`None` honoured |
| — | zarr `append_plane(compression=…)` warns | **REGRESSION — fixed here** | `NameError: name 'warnings' is not defined` — `storage.py` never imported `warnings` |
| I7 | focus-scan gating | **VERIFIED** | 5 calls / 155 slices → **0 / 0** for an opd-map merit; unchanged for `StrehlMerit` and `SpotSizeMerit`; **a user merit class predating the flag keeps the scan and sees real values** |
| I7 | infeasible Pareto front refuses | **VERIFIED** | driven end to end through a pymoo stub of my own: `ValueError` with and without `progress`; feasible control still returns a front |
| I7 | `edge_thickness` / `MinEdgeThicknessMerit` | **VERIFIED** (after §10/V3) | matches an independently written even-asphere sag to **≤ 8.7e-19 m** on 5 aspheric/conic shapes the WP did not test; an unmeasurable (NaN) slot now costs the full `min_edge**2` instead of scoring 0.0 |
| I7 | CODE V / Quadoa emit `elements`, unknown-glass warnings | **VERIFIED** | both loaders emit `elements` + `all_thicknesses`; both warn on an unregistered glass; `split_prescription_at_mirrors` fallback warns |
| I8 | `.zmx` `NAME`/`COMM` injection | **VERIFIED** | 4 `SURF` records, reloads as 2 surfaces, `CURV 0.5` absent; `"` → `'` with a warning.  The `COMM` sanitiser sits only on the `elements` writer path — the ordinary path never emits `COMM` at all, so there is no hole; the changelog now says which path (§10/V8) |
| I8 | UTF-16-BE | **VERIFIED** | loads; and **no regression from putting BOM-sniffing `utf-16` first** — 7 encodings (utf-8 odd/even length, utf-8-sig, utf-16-le ±BOM, utf-16-be, latin-1 + accent) all give R1 = 0.05 m |
| I8 | TOROIDAL/BICONICX deferred | **VERIFIED as deferred** | warning does name `make_biconic` / `radius_y` / `conic_y` as the hand-entry route |
| I8 | Quadoa stop not invented | **VERIFIED** | undeclared stop → `stop_index None` (was 0); declared stop 1 → 1 |
| I8 | `create_zoom_configs` | **VERIFIED** | too short / too long / bad slot all `ValueError`; exact length and the `(slot, value)` pair form both write only the named slots |
| I8 | `method='newton'` bounds warning | **VERIFIED** | two-sided: 1 warning with bounds, 0 without |
| I8 | `x0` bounds check | **VERIFIED** (behaviourally; the WP's source-text pin + skip is deleted per §10/V5) | stub-driven: warns for `x0 = 5.0` outside `[0, 1]`, silent for `0.5` |
| I8 | aperture cache key is a digest | **VERIFIED** | 128-bit blake2b; equal for equal, differs on a real 1-ULP change, F-order copy keys identically, float32 keys differently |

Collateral damage: **none found**.  Every real `.zmx` fixture in the repo loads with bit-identical geometry;
the only difference is the additive `configurations` key.

---

## 1. Repro scripts re-run on the current code

All of `repro/IO-OPTIMIZE/` re-run (`p9_perf.py` excluded — it is a pure timing profile with no claim attached).
Large outputs regenerated and deleted afterwards (peak 498 MiB, well under the 2 GB cap).

`p6b_mp.py` (the "checked and found correct" multi-process HDF5 append, which the storage changes sit on top of)
still reports **100 planes, 100 unique labels, 0 duplicates, 0 missing** across 2 processes × 50 appends.

```
p2_codev.py    DIM M  -> R1 0.06275 m (was 62.75), EFL 0.0722 m (was 72.2154), 0 warnings
               DIM C  -> 0.6275 m      DIM I -> 1.59385 m      (both were 62.75)
               no RDY -> radius inf (was None); system_abcd OK (was ValueError)
               K/A/B/REFL/RMD -> conic -1.0 (was 0.0), asph {4: 123.4, 6: -50000.0} (was None),
                                 'elements' present, has_mirrors True (both were False)
p1_zemax.py    UNIT IN/CM/M/BOGUS unchanged; EVENASPH sag 0.024237623 mm both ways; UTF-16BE now loads
p1c/p1d        PARAXIAL+glass -> elements surf_nums [1,2,3] (was [2,3]), stop_index 0 (was None),
                                 loud unsupported-SURFTYPE warning (was 0 warnings)
               EVENASPH air-to-air upstream -> STILL [2,3], now with an excluded-surface warning
               multicfg -> 'configurations' key + warning (was neither)
p6_storage.py  HDF5 18/19; ZARR append_plane 13/19 -> 18/19; ZARR sim_metadata 18/19
p7_codegen /   GLAS payload now lands inside a string literal; emitted block executes,
p7b.py         radius -inf preserved, conic inf; scale_prescription r_max/q_bfs/BFL/period/gap all correct
p8_opt.py      StrehlMerit run: 17 evals, 527 focus slices -> UNCHANGED, which is correct
               (StrehlMerit reads the scan; see §5)
```

Every number matches the WP report's claimed after-values.

---

## 2. Regression 1 (P1) — pre-v5.46 `DIM MM` / `DIM IN` files are mis-read by ×1000 / ×39.37

**What I did.**  Rebuilt the pre-fix writer from git (`git show 658e6142^:lumenairy/io/prescriptions_code_v.py`),
wrote the same physical singlet with each of its three accepted unit spellings, and read each file back with
the pre-fix loader and with the WP's loader.

**Measured on `658e6142`:**

| file written by the pre-v5.46 writer | pre-v5.46 loader | WP-A10 loader | error |
|---|---|---|---|
| `units='M'`  → `DIM M`, `RDY 0.05000000` | 0.05 m | 0.05 m | — (correct) |
| `units='MM'` → `DIM MM`, `RDY 50.00000000` | 0.05 m | **50.0 m** | **×1000** |
| `units='IN'` → `DIM IN`, `RDY 1.96850394` | 0.05 m | **1.96850394 m** | **×39.37** |

**Cause.**  The legacy sniff is `banner present AND marker absent`, and `_unit_scale()` then returned a blanket
`1.0` (metres) **regardless of the file's own `DIM` token**.  Only `DIM M` is ambiguous: the pre-fix writer
wrote SI metres under CODE V's millimetre token.  `DIM MM` and `DIM IN` meant millimetres and inches then, and
mean the same now — those files need no special handling at all, and forcing metres on them is the very defect
I1 fixed, pointed the other way.  The pre-fix docstring advertised the kwarg as *"`'MM'`/`'IN'` trigger
conversion on write (useful for handing files to CODE V users)"*, so such files exist.

Worse, the warning text asserted "``DIM M`` was emitted with SI-METRE numbers" on a file whose `DIM` line says
`MM` — right conclusion, wrong file.

**Also found in the same block.**  The marker was recognised by exact string
(`c.startswith('! LUMENAIRY-SEQ-FORMAT 2')`), so a future `LUMENAIRY-SEQ-FORMAT 3` file would be classified as
pre-v5.46 metres — a ×1000 error introduced by a one-character version bump.

**Fix applied** (`lumenairy/io/prescriptions_code_v.py`, WP-A10-owned):

* added `_CV_LEGACY_TO_METERS = {'M': 1.0, 'MM': 1e-3, 'IN': 0.0254}` — the pre-fix writer's own table;
* pre-scan the file's `DIM` token, and take the legacy branch only for `DIM M` or no `DIM` line;
* `_unit_scale()` consults the legacy table by token instead of returning `1.0`;
* recognise the marker by prefix (`_CV_FORMAT_MARKER_PREFIX`);
* the `Migration` docstring now states the `MM`/`IN` case explicitly.

**Verified after:** `ratio new/old = 1` for all three spellings; the `DIM M` arm keeps WP-A10's behaviour
(0.05 m + migration warning) and `dim_units='M'` still forces 5e-5 m.

---

## 3. Regression 2 (P1) — `append_plane` on a `.zarr` store raises `NameError`

`io/storage.py:1871` (WP-A10's "perf note 6" fix) calls `warnings.warn(...)`, but **`storage.py` never imports
`warnings`** — the WP's new line is the only reference to the name in all 2145 lines.  So

```python
la.io.storage.set_storage_backend('zarr')
append_plane('run.zarr', E, dx=..., compression='gzip')
   -> NameError: name 'warnings' is not defined      (storage.py:1871)
```

A call that previously wrote the plane (silently dropping the compression request — the thing the note set out
to diagnose) now writes **nothing**.  The WP report's row for this item lists no test (`tests: —`), which is
exactly how it shipped.

**Fix applied:** one line, `import warnings` at module scope.  After: the call warns
(`append_plane(...): the Zarr backend ignores ['compression', 'compression_opts'] …`), the plane is written and
reads back bit-identically, and the same call without the kwarg stays silent.

---

## 4. Independent checks built on fixtures the work package did not use

### I1/I3 — a genuine CODE V sequence in INCHES with a fold mirror

Hand-written `.seq`, no lumenairy banner, `DIM I`, a parabola (`K -1.0`) carrying `A` (y⁴), `B` (y⁶) and
`J` (y²⁰), a fold mirror in **air** between two glass elements.  Oracles computed in the check script: the inch
definition (0.0254 m exactly), the lens-unit rescale `a_p = a_file / L^(p−1)`, and the closed-form even-asphere
sag.

```
R1     = 0.0508 m                           exact 2.0 * 0.0254            err 0.0
K1     = -1.0
a_4    = 6.102374409473229                  exact A/L^3                   relerr 0.0
a_6    = -236.4674813020503                 exact B/L^5                   relerr 0.0
a_20   = 1.4231721099163658e+16             exact J/L^19                  relerr 0.0
sag(y = 0.4 in) : oracle 1.016064763904000e-03 m  library 1.016064763904000e-03 m  relerr 0.0
elements        : surface, surface, MIRROR, surface, surface     has_mirrors True
all_thicknesses : [0.25, 1.5, 2.0, 0.2] in      thicknesses : [0.25, 3.5, 0.2] in
                  (the mirror's two legs fold into ONE refractive gap — correct)
load -> export(units='I') -> load : |dR| = 0.0, dK = 0.0, max rel |da| = 0.0 on every element
```

Also probed: an all-mirror `.seq` raises a clear `ValueError` naming `split_prescription_at_mirrors`; a
realistic fold file yields a correct glass chain `air→N-BK7→air→N-SF11→air` and passes `validate_prescription`.

### I5 — hostile codegen payloads of my own

Oracle: Python's own `ast`.  A payload is *live code* iff it appears as a `Call` node (or an `Import`) rather
than inside a string constant.  Seven payloads — single-quote escape, double-quote escape, both quote styles,
a backslash, a U+2028 line separator, a `"""` docstring terminator, and a format-spec probe — across three
attack surfaces (`GLAS` token, `COMM`, prescription `name`).

```
quote_escape  parses=True  dangerous-nodes=[]   double_quote parses=True  dangerous-nodes=[]
both_quotes   parses=True  dangerous-nodes=[]   backslash    parses=True  dangerous-nodes=[]
u2028         parses=True  dangerous-nodes=[]   tripquote    parses=True  dangerous-nodes=[]
fmt           parses=True  dangerous-nodes=[]
control (AC254_100_C.zmx): script parses, 0 injection warnings, N-BAF10/N-SF6HT present
```

Sub-note (hardening, not a defect): a raw U+2028 survives into the generated script.  CPython does not
treat it as a source newline (verified: a literal U+2028 inside a compiled string literal stays one
3-character string), so it is inert -- but a consumer that splits on `str.splitlines()` would disagree.
**Superseded by §10 (V10):** on re-measurement the leak is NOT in `_comment_text` (whose trailing
whitespace collapse already catches those characters) but in the two `#` lines that never called it --
the per-lens comment, whose text is the first surface's `COMM`, and the `style='system_list'` DOE
placeholder.  Both are fixed and the pin asserts the end-to-end property.

### I6 — the catalogue, by a different algebra

The library's own ledger uses a reduced-slope trace; I used the 2×2 ray-transfer matrix product in
(height, angle) coordinates, `EFL = −1/C`:

```
part          nominal   ABCD oracle    dev    warns
LA1050-C      100.00mm   99.6520mm   -0.35%     0
LA1509-C      100.00mm   99.6520mm   -0.35%     0
LA1301-C      250.00mm  250.0008mm   +0.00%     0
AC254-050-C    50.00mm   44.5601mm  -10.88%     1
AC254-200-C   200.00mm  137.3950mm  -31.30%     1
AC254-100-C   100.00mm   83.1708mm  -16.83%     1
LA1509-C with the PRE-FIX radius 103.29 mm -> 199.8652 mm  (ratio 2.005631)
closed form R/(n-1), n(N-BK7, d) = 1.516798438 -> 99.652004 mm
```

`LA1509_C.zmx` (`R1 = 0.051500000074675 m`, `CURV 0.0194174757`), `LA1509_C.txt` and `THORLABS_CATALOG` all
agree.  The regenerated fixture numbers check out against the closed form at 1310 nm
(`n(N-BK7) = 1.503582905`): `EFL = R/(n−1) = 102.267173 mm` vs the header's `102.2672`, and
`BFL = f(1 − (n−1)t/(nR)) = 99.872892 mm` vs `DISZ 99.87289177`.

### I6 — zarr metadata with payloads the probe set did not contain

16 exotic values — `int64`/`bool`/`complex128`/`float32` ndarrays, a **4096-element** array (past numpy's
1000-element print threshold, the irrecoverable case), `bytes(range(256))`, nested dicts, empty dict, empty
string, non-ASCII text, a key containing spaces, `-0.0`, `2**62`, a NaN/±inf array, nested tuples, a list of
arrays:

```
hdf5  append_plane 15/16   list_planes 15/16   by_label 15/16
zarr  append_plane 15/16   list_planes 15/16   by_label 15/16
      the single "miss" on BOTH backends is 'list_of_arr' ([array([0,1,2]), array([0,1])]),
      which comes back byte-for-byte -- my strict comparator simply cannot compare a
      list-of-ndarrays with ==.  Every other payload, including the 4096-element array,
      is exactly faithful, and the two backends are now identical.
```

### I7 — focus-scan gating, including a merit class that predates the flag

Counted, never timed.

| merit set | scan calls | slices |
|---|---|---|
| `FocalLength + Strehl` (reads the scan) | 5 | 155 |
| `FocalLength + SpotSize` (reads the scan) | 5 | 155 |
| `FocalLength + RMSWavefront` (does not) | **0** | **0** |
| `FocalLength +` a user merit class with **no** `needs_focus_scan` attribute | 5 | 155 |
| `RMSWavefront + Strehl` (mixed) | 5 | 155 |

The legacy merit saw a non-zero `strehl_best` on all 5 evaluations, so the `getattr(..., True)` default is a
real backward-compatibility guarantee and not just a flag default.  `p8_opt.py`'s StrehlMerit run is unchanged
at 31 slices per evaluation, which is the correct answer for a merit that reads the scan.

### I7 — edge thickness on ASPHERIC surfaces (the WP verified spheres only)

Independent closed-form even-asphere sag, five shapes beyond the WP's four:

```
parabola + A4 + A6                 t_edge  3.114402255568e-03   |d| 4.34e-19
hyperbolae both sides, A4 + A8     t_edge  3.227540928785e-03   |d| 0.00e+00
oblate k>0 + A4 + A10              t_edge  4.799048595044e-04   |d| 2.17e-19
aspheric plano-rear                                              |d| 0.00e+00
knife edge (negative answer)       t_edge -9.733500838578e-03   |d| 0.00e+00
worst |delta| over all nine shapes = 8.674e-19 m   (float64 floor ~1e-19 m)
```

### I7/I8 — multi-objective, driven end to end

With a pymoo stub of my own construction (the WP's own `x0` test is a source-text pin behind a `pytest.skip`):

```
x0 = [5.0] outside [0, 1] -> 1 "OUTSIDE bounds" warning ; x0 = [0.5] -> 0 warnings
Result.X is None, progress=None    -> ValueError "NO feasible solution ... Smallest constraint violation: 3.25"
Result.X is None, progress=callable-> ValueError (pre-fix: IndexError from X.shape[0])
feasible control                   -> X.shape (5, 1), F.shape (1, 2)
np.asarray(None, float64).ndim == 0 (the step the whole finding rests on)  -- measured, not assumed
```

---

## 5. Notes on claims that hold but are described inaccurately

1. **I2 is the audit's *minimum*, not its maximum.**  The changelog says "Air-to-air powered SURFTYPEs now
   enter the window".  `_ZEMAX_AIR_POWERED_TYPES` contains PARAXIAL/IDEAL/ABCD/BINARY_*/phase types but **not**
   `EVENASPH`, `ODDASPHE`, `QBFS`, `QCON`, `ZERNSAG`, `GRID_SAG`, which the audit's fix line names explicitly.
   Measured on the audit's own `evenasph_upstream.zmx`: `elements surf_nums = [2, 3]` before *and* after — the
   air-spaced phase plate is still dropped, now with the belt-and-braces excluded-surface warning.  That is the
   audit's stated fallback ("or — minimally — warn once naming every optical surface the window excluded"), so
   the row is closed, but the changelog sentence overstates it.  Recommend rewording to "…now enter the window
   (PARAXIAL / IDEAL / ABCD / phase types); an air-spaced aspheric phase plate is still excluded but is now
   named in a warning."
2. **The `.txt` twin loader did not get I2.**  `load_zemax_prescription_data_txt`'s window auto-detect
   (`prescriptions_zemax.py:1671-1673`) is still `glass is not None or is_mirror`, with no
   `_raw_surface_is_air_powered` and no excluded-surface warning.  The partition report calls out exactly this
   pattern ("every one of those is a place a future fix lands on one side only — which is the observed
   history").  Severity: low-medium — a `PRESCRIPTION DATA` summary export is less likely to carry an ideal
   lens, and P3-43 already warns about non-STANDARD types *inside* the window.
3. **`thorlabs_lens` warns once per process, not "on every call".**  Measured: call 1 → 1 warning, calls 2 and 3
   → 0 (there is a module-level `_THORLABS_EFL_WARNED` set).  Report §1 and the changelog both say "warn on
   every call" / "warns loudly on every call".  The once-per-process behaviour is defensible; the text is not.
4. **A stale number in the source.**  `prescriptions_builders.py`'s `LA1509-C` comment says
   *"was 199.68 mm with R1 = 103.29 mm"*.  Measured (two independent algebras): **199.8652 mm**.  The report and
   changelog both say 199.865 — only the in-code comment is wrong.  TESTING_STANDARDS' durability rule
   ("a numeric constant without a stated origin is a defect") applies here.
5. **`MCON` decoded fields look mis-assigned.**  On the audit's own `multicfg.zmx` (three configs, thickness
   50/75/100 mm on surface 2) the loader returns `operand '1'/'2'/'3'` with **`config = 3.0` on all three rows**
   and `surface = 2.0`.  Whatever the true `MCON` field order is, three rows all claiming configuration 3 is
   internally implausible.  `raw` is preserved so nothing is lost, and nothing consumes the decoded fields yet —
   but `create_zoom_configs` is named in the comment as the intended consumer.  Recommend marking the decoded
   fields provisional (or dropping them, keeping `raw`) until a real OpticStudio multi-config file confirms the
   layout.  This is the same "not verifiable offline" reasoning the WP correctly applied to `BICONICX`.
6. **`COMM` sanitising only covers one writer path.**  `_zmx_record_text(..., field='COMM')` sits in
   `_export_zemax_zmx_full` (the `elements` path).  The ordinary `export_zemax_zmx` path never emits `COMM` at
   all, so there is no hole — but the changelog's "Both fields are now collapsed" reads as though both paths
   write both fields.
7. **`INDEX.md` aperture column is now half-current.**  The regenerated `LA1509_C` row reads `15.00` (matching
   `lens_cases.py`), while `LA1050_C` / `LA1301_C` / `AC254_100_C` / `AC254_200_C` still read `20.00` although
   `lens_cases.py` gives them 15.0 mm too.  Pre-existing staleness, now visible.  `LA1509_C.txt` also came back
   in the *current* generator's header format, which differs from its 20 siblings (it lost the
   "Residual (piston+tilt+defocus) RMS (slant)" line and gained a "Paraxial BFL" line).  Cosmetic.

---

## 6. Open items for the orchestrator

| # | severity | item |
|---|---|---|
| V1 | **P1 — fixed here** | pre-v5.46 `DIM MM` / `DIM IN` files read ×1000 / ×39.37 too large (§2).  Fixed in `prescriptions_code_v.py`; the changelog's migration note should add the `MM`/`IN` sentence. |
| V2 | **P1 — fixed here** | `append_plane(<.zarr>, compression=…)` raised `NameError` (§3).  Fixed by `import warnings` in `storage.py`. |
| V3 | P2 | `MinEdgeThicknessMerit` **silently skips** a slot whose `edge_thickness` is `nan`.  Measured: `R = 8 mm` evaluated at `h = 10 mm` (the surface does not reach its own clear semi-diameter) → `edge_thickness = nan` → merit contribution **0.0**, i.e. the element scores as *satisfying* the constraint, precisely where the constraint matters.  A well-defined knife edge is penalised correctly (1.15e-4), so the merit is not inert.  Suggested fix: treat a non-finite edge as a maximal violation (`deficit = min_edge`), or raise — matching the S4-5 precedent in `driver.py` where a NaN `strehl_best` is coerced to 0.0 so a degenerate design is *penalised*.  Pinned as-is by `test_verify_i7_edge_thickness_is_nan_when_the_surface_is_undefined`. |
| V4 | P2 | `lumenairy/__init__.py` still does not re-export `MinEdgeThicknessMerit` / `edge_thickness` (confirmed: `hasattr(la, 'MinEdgeThicknessMerit')` is `False`).  This is WP-A10's own request #1 — and `MinEdgeThicknessMerit`'s docstring `Examples` block already writes `la.MinEdgeThicknessMerit(...)` (it is `# doctest: +SKIP`, so nothing fails, but the documented API does not exist).  Land the re-export with the WP. |
| V5 | P2 | `tests/unit/test_audit2609_a10_optimize.py::test_i8_x0_outside_bounds_warns` violates TESTING_STANDARDS rule 4 (`pytest.skip` on a dependency precondition — it removes the test on exactly the runners that have pymoo) and tests the *implementation* (`inspect.getsource` + two `str.index` calls), so it passes for a disabled check.  `test_i7_infeasible_guard_is_present_in_the_source` is the same shape.  Both are now covered behaviourally by `test_verify_i8_x0_outside_bounds_warns_end_to_end` / `test_verify_i7_infeasible_front_refuses_with_and_without_progress` using a stub, with no skip.  Suggest deleting the two structural pins or keeping them only as comments. |
| V6 | P2 | `MCON` decoded `config`/`surface` fields (§5.5). |
| V7 | P3 | The `.txt` twin loader did not receive the I2 predicate or warning (§5.2). |
| V8 | P3 | Wording corrections: I2 "enter the window" (§5.1), `thorlabs_lens` "on every call" (§5.3), `COMM` "both writers" (§5.6). |
| V9 | P3 | Stale in-code number `199.68 mm` → `199.8652 mm` in `prescriptions_builders.py` (§5.4). |
| V10 | P3 | `_comment_text` should also strip U+2028/U+2029 (§4, I5 sub-note). |
| V11 | P3 | `generate_simulation_script` on a **raw builder** prescription still raises a bare `KeyError: 'elements'` (`codegen.py:410`).  Confirmed **pre-existing** — identical on `658e6142^` — so not a WP regression, but it is the same class as the finding I7 closed and wants a §2-prefixed message pointing at `normalize_prescription`. |
| V12 | P3 | `tests/unit/test_audit2609_a10_optimize.py:14` has a ruff `I001` (unsorted imports).  `ruff check` is clean on everything else WP-A10 touched. |

---

## 7. Collateral damage sweep

**Every prescription fixture in the repo, pre-fix tree vs post-fix tree.**  `git archive 658e6142^ lumenairy`
into a scratch tree, loaded all 46 `.zmx` / `.seq` / `.qos` fixtures under `validation/`, `tests/`, `examples/`
and `docs/` with both trees, and diffed a canonical summary (radii, conics, glass chain, thicknesses, aperture,
stop index, object distance, BFL, element list, warning set).

* All 21 real `.zmx` fixtures (`AC254_*`, `LA*`, `plano_convex_*`, `meniscus_*`, `fnum_sweep_*`, …): **the only
  difference is the additive `configurations` key.**  Radii, thicknesses, apertures, stop indices and warning
  sets are identical.
* `paraxial2.zmx`: surface 1 now included (intended, I2); `object_distance` 0.1 → 0.0 with the same total track.
* `paraxial.zmx` (PARAXIAL only): error text changed from "No glass/mirror/diffractive surfaces found" to
  "Need at least 2 surfaces, got 1 in range (1, 1)" — acknowledged in the WP report; slightly less diagnostic.
* `.seq` fixtures: the intended ×1000 correction plus the new `elements` / `all_thicknesses` keys.
* `asph_refl.seq`: `surfaces` 3 → 1 because the two `REFL` rows now leave the refractive list (they are in
  `elements`).  Correct by construction — that fixture's second mirror sits inside glass and never exits, so
  the remaining refractive surface legitimately ends in `N-BK7`.  A realistic fold file (mirror in air between
  two singlets) gives a clean `air→N-BK7→air→N-SF11→air` chain and passes `validate_prescription`.

**Test files.**

| command | result | duration |
|---|---|---|
| `pytest` × 19 io/optimize test files (`test_audit2609_a10_*`, `test_audit_io`, `test_audit_optimize`, `test_audit_s4_9_io_silent_fallback`, `test_g08_s4_19_io_hygiene`, `test_combine_prescriptions`, `test_niche_d4_dgrating`, `test_v5_1_0_agent_f_split`, `test_v5_4_6_io_ui_delegated`, `test_audit_w5_zemax`, `test_audit_w6_io_zemax`, `test_audit_w5_optimize`, `test_audit_w6_optimize`, `test_optimize_merit_terms`, `test_c1_s4_19_storage_metadata_contract`, `test_g08_s4_18_optimizer_hygiene`) | **455 passed, 1 skipped** (PySide6) | 174 s |
| `pytest tests/unit/test_niche_c1_consolidation.py tests/unit/test_audit_misc.py` | 262 passed, 3 skipped (cupy), **1 failed** — see below | 383 s |
| `pytest` the four WP files + my new file + storage/io hygiene files, after my two source fixes | **129 passed** | 13 s |
| `pytest test_audit_io / test_v5_1_0_agent_f_split / test_niche_d4_dgrating / test_combine_prescriptions`, after my fixes | **196 passed** | 154 s |
| `pytest tests/unit/test_audit2609_a10_verify.py` (new) | **17 passed** | 2.5 s |
| `python validation/run_all.py io` | **PASS** | 4.2 s |
| `python validation/run_all.py optimize` | **PASS** | 12.7 s |

The single failure, `test_audit_misc.py::TestAuditFixesV4_12_1_coverage_StopIndexWarn::test_traced_emits_warning_for_stop_index_2`,
is **not WP-A10's**: another engineer rewrote that test in the working tree *during* my run
(`tests/unit/test_audit_misc.py` mtime 05:38:11, inside the 05:31–05:38 window; the test now exists as
`test_traced_emits_warning_for_a_mid_train_stop` with an "UPDATED 2026-09-12 (audit 2026-09-11, finding L14)"
docstring, and `lumenairy/elements/_lens_real.py` / `lenses.py` are modified alongside it).  Re-running that
class against the current tree: **6 passed**.

The two failures WP-A10 reported as pre-existing in `test_niche_c1_consolidation.py` now pass — the CARRIER and
analytic-lens work packages' uncommitted changes are in the working tree.  **Caveat for the orchestrator:**
every run above was made against a working tree carrying other engineers' uncommitted edits, so a clean
re-run on the merged branch is still worth doing.

---

## 8. Changes I made (all inside WP-A10's ownership)

```
lumenairy/io/prescriptions_code_v.py   +47 -11   legacy DIM-token scale + prefix marker match (§2)
lumenairy/io/storage.py                 +1  -0   import warnings (§3)
tests/unit/test_audit2609_a10_verify.py  NEW     17 tests
```

`ruff check` clean on all three.

**Fail-before demonstration.**  The new test file was run against the **as-committed** WP tree
(`git archive 658e6142 lumenairy`, put first on `sys.path`):

```
4 failed, 13 passed
FAILED test_verify_i1_pre_v546_mm_and_in_files_keep_their_own_units[MM]   Obtained 50.0   Expected 0.05
FAILED test_verify_i1_pre_v546_mm_and_in_files_keep_their_own_units[IN]   Obtained 1.96850394  Expected 0.05
FAILED test_verify_i1_a_future_format_marker_is_not_read_as_legacy        Obtained 0.05   Expected 5e-05
FAILED test_verify_zarr_append_plane_compression_warns_instead_of_raising NameError: name 'warnings' is not defined
```

— exactly the four regression pins, and the other 13 (independent coverage of the WP's own claims) pass on
both trees, which is itself confirmation that those claims hold.  On the fixed tree: **17 passed**.

New tests, by purpose:

*Regression pins for §2/§3:* `test_verify_i1_pre_v546_mm_and_in_files_keep_their_own_units[MM|IN]`,
`test_verify_i1_legacy_dim_m_still_reads_as_metres_and_warns` (control arm),
`test_verify_i1_a_future_format_marker_is_not_read_as_legacy`,
`test_verify_zarr_append_plane_compression_warns_instead_of_raising`.

*Independent coverage of WP claims:*
`test_verify_i3_inch_file_with_a_fold_mirror_round_trips`,
`test_verify_i5_hostile_glass_token_never_reaches_a_code_position`,
`test_verify_i5_an_ordinary_zmx_still_generates_a_runnable_script`,
`test_verify_i7_edge_thickness_matches_an_independent_aspheric_sag` (5 params),
`test_verify_i7_edge_thickness_is_nan_when_the_surface_is_undefined` (pins V3),
`test_verify_i7_focus_scan_runs_for_a_merit_class_without_the_flag`,
`test_verify_i8_x0_outside_bounds_warns_end_to_end`,
`test_verify_i7_infeasible_front_refuses_with_and_without_progress`.

Every numeric bar carries its oracle, its floor and the measured value; no wall-clock or speedup assertion; no
`pytest.skip` on a resource precondition; every claim two-sided.

---

## 9. Audit of WP-A10's own tests against `docs/TESTING_STANDARDS.md`

69 tests collected across the four new files (matches the report).  Read in full.

**Good.**  No wall-clock or speedup assertion anywhere, including in the HDF5 default change — the tests read
the *filter pipeline* and the *chunk shape* back out of the written file, which is the right restatement of a
performance claim (rule 1, "assert decisions, not readings").  Bars carry derivations with dated measurements
(e.g. `rel=1e-9` justified against the writer's `%.8f` quantisation floor of 2e-10; the 3 % catalogue gate
justified with 8.6× headroom above the largest legitimate deviation and 3.6× below the smallest real defect).
Oracles are independent where it matters (unit definitions, a hand-written paraxial trace cross-checked against
`system_abcd` to < 1e-12, `ast` structure of the emitted script).  The focus-scan test counts a *decision*
(scan calls) rather than a duration.

**Two defects** (both recorded above as V5):

* `test_i8_x0_outside_bounds_warns` and `test_i7_infeasible_guard_is_present_in_the_source` assert substring
  ORDER in `inspect.getsource`.  That is a pin on the implementation text, not the property: it passes if the
  guard is `if False and pymoo_res.X is None:`.
* the former also calls `pytest.skip` when pymoo *is* installed — rule 4 verbatim ("never `pytest.skip` on a
  resource check — two skips silently removed five tests from the gate on exactly the runners that mattered").
  My stub-driven replacements show the end-to-end path is reachable with no dependency at all.

**One mild risk, deliberate.**  `test_i6_catalog_row_efl_ledger` pins the three known-bad deviations to
`abs=5e-4`, i.e. a *reading* rather than a decision.  The WP states this is intentional (it must fail once
vendor data lands).  It would also fire on a legitimate `GLASS_REGISTRY` dispersion update — acceptable given
the explicit "update this ledger and say why in the changelog" failure message.

---

## 10. Open-item resolution (coordinator rulings, 2026-09-12)

All rulings implemented in WP-A10's own files.  Every behaviour change is pinned
in `tests/unit/test_audit2609_a10_verify.py`, and the whole file was re-run
against the **as-committed** tree (`git archive 658e6142 lumenairy`, first on
`sys.path`) to demonstrate fail-before: **9 failed, 13 passed** there;
**22 passed** on the fixed tree.

| # | ruling | what changed | pin | fail-before on `658e6142` |
|---|---|---|---|---|
| V3 | non-finite edge = maximal violation | `optimize/merit_terms.py` `MinEdgeThicknessMerit.evaluate`: a NaN `edge_thickness` now contributes `min_edge**2` instead of being `continue`-d; class docstring says so | `test_verify_i7_undefined_edge_is_a_maximal_violation` | contribution **0.0 → 1e-6** (R = 8 mm at h = 10 mm, `min_edge` = 1 mm) |
| V5 | delete the source-text/skip pins | `tests/unit/test_audit2609_a10_optimize.py`: `test_i7_infeasible_guard_is_present_in_the_source` and `test_i8_x0_outside_bounds_warns` removed, replaced by a comment naming the two behavioural tests that cover them | the two stub-driven tests in `…_verify.py` | n/a (test-only) |
| V6 | `MCON` decoded fields provisional | `io/prescriptions_zemax.py`: the named `config` / `surface` / `value` keys are gone; each row is `{'raw', 'operand', 'fields_provisional'}` with `raw` documented as the contract, in the loader docstring and in the warning | `test_i7_multiconfig_records_are_surfaced` (strengthened in place) | n/a (no test pinned the old names) |
| V7 | `.txt` twin gets the I2 fix | `io/prescriptions_zemax.py`: two new shared helpers, `_raw_surface_curvature` / `_raw_surface_nonzero_parms`, let `_raw_surface_is_air_powered` read either loader's record shape; `_warn_window_excluded_powered` factored out of `load_zemax_zmx` and called from both | `test_verify_v7_txt_loader_keeps_a_powered_air_to_air_surface`, `…_warns_about_a_surface_the_window_excludes`, `…_an_ordinary_txt_doublet_gains_no_new_diagnostic` | 2 surfaces / `stop_index None` / 0 warnings → **3 surfaces / `stop_index 0` / the P3-43 shape warning** |
| V9 | stale catalogue number | `io/prescriptions_builders.py`: `199.68` → `199.8652 mm` with the re-measurement dated and attributed | — (the ledger test already re-measures) | n/a (comment) |
| V10 | strip U+2028 / U+2029 | `io/codegen.py`: `_comment_text` strips U+0085 / U+2028 / U+2029 explicitly **and** the two `#` lines that were not calling it now do | `test_verify_v10_no_unicode_line_separator_reaches_a_generated_script` | a raw U+2028 in the emitted script → **none** |
| V11 | §2-prefixed builder error | `io/codegen.py` `_decompose_prescription`: a `KeyError` naming the missing keys, the keys present, and `normalize_prescription(rx)` as the fix | `test_verify_v11_codegen_on_builder_output_names_the_fix` | bare `KeyError: 'elements'` → **prefixed message** |
| V12 | sort imports | `tests/unit/test_audit2609_a10_optimize.py`, plus two the WP left in `optimize/__init__.py` and `optimize/core.py` (`edge_thickness` out of ruff's order) | — | n/a |
| V4 | top-level re-export | **left alone** — assigned to the tests/CI package | — | — |

### Notes on what the implementation turned up

* **V10 was not where I first reported it.**  My §5 note blamed `_comment_text`.
  Re-measuring during the fix showed `_comment_text` already collapsed those
  separators — its trailing `re.sub(r'\s+', ' ', …)` matches Unicode whitespace,
  so the unit-level gap was cosmetic.  The real leak was the **per-lens
  `# {comment}` line**, which did not call `_comment_text` at all, and its
  `style='system_list'` twin (the DOE placeholder).  Measured on `658e6142`:
  exactly one line of a generated script carried a raw U+2028, and it was that
  one.  Both sites now route through the helper, and the test asserts the
  end-to-end property ("no `str.splitlines()` boundary survives into the emitted
  script") rather than the helper's return value.  The separators in the test are
  built with `chr(0x2028)` rather than written literally, and the source files
  carry escapes only — a literal U+2028 in a `.py` file is invisible and
  editor-fragile, which is how this class of defect hides.
* **V7 did not become a copy.**  The `.txt` records store `radius` in metres
  where the `.zmx` records store `curvature`, so admitting the predicate to both
  needed one reader (`_raw_surface_curvature`) rather than a second copy of the
  test.  The `.txt` loader's `s_last` rule was widened the same way the `.zmx`
  twin's was in v5.32 (`is_mirror or glass is None` — no `+1` for a surface with
  no exit face); for every all-glass file the window is byte-identical to before,
  which the third test asserts.  The summary table carries no `PARM` columns
  (P3-43), so a *flat* ideal lens still cannot be detected as powered there —
  that case is the excluded-surface warning's, and the second test pins it.
* **V6 removed data rather than renaming it.**  Keeping `config` / `surface`
  under a `_provisional` marker would still have invited use; `raw` plus an
  undecoded `fields_provisional` list cannot be mistaken for a decoded row.  The
  WP's own multi-config test now asserts `raw` exactly and asserts that the two
  invented names are **absent**.
* **V3 preserves the ordering that matters.**  An unmeasurable edge costs
  `min_edge²` (1e-6 here); a *measurably* knife-edged element still costs more
  (1.15e-4), so the penalty surface still points away from the worse geometry
  rather than flattening both to the same value.  A healthy element still scores
  exactly 0.0.

### Collateral after the open-item changes

| command | result |
|---|---|
| `pytest` × 23 io/optimize test files (the §7 set plus `test_v4_16_1_agent_c`, `test_v5_21_2_subsystem_audits`, `test_niche_audit_r_guards_and_merits`) | **577 passed, 4 skipped, 4 failed** — none of them mine, see below |
| `pytest tests/unit/test_audit2609_a10_verify.py` | **22 passed** |
| `python validation/run_all.py io` | **PASS** (9.7 s) |
| `python validation/run_all.py optimize` | **PASS** (12.8 s) |
| `ruff check lumenairy/io lumenairy/optimize` + the five WP-A10 test files | **All checks passed** |

**The four failures are not WP-A10's**, established by measurement rather than
by reading:

* `test_v4_16_1_agent_c.py::test_propagation_asm_cache_lock_still_paired` —
  **passes standalone** (`20 passed` for the whole file).  It is a global-state
  pairing check that some earlier test in the 577-test session leaves unbalanced;
  an ASM propagation cache lock, nothing io/optimize touches.
* The three LG-merit tests
  (`test_audit_optimize.py::…test_piston_weight_scales_merit_linearly`,
  `test_v5_21_2_subsystem_audits.py::test_opt1_lg_jax_merit_is_strehl_deficit_not_amplitude`,
  `test_niche_audit_r_guards_and_merits.py::test_r5_numpy_lg_merit_matches_jax_twin`)
  fail on an LG coupling tensor that has moved ~18 orders of magnitude
  (`|L_00|^2 = 6.0e+14` against a measured well-conditioned band of
  `4.74e-04 .. 4.04e-03`).  Two isolation runs settle the ownership:

  1. the **current working tree with WP-A10's own committed `io/` and
     `optimize/` swapped back in** (`git archive 658e6142`) → **3 failed**;
  2. the **fully committed tree** `658e6142` → **4 passed**.

  So the cause is an uncommitted change outside `io/` and `optimize/` — another
  engineer's in-flight work — and it is present regardless of which version of
  this work package's files is loaded.  (A subpackage-level bisect was attempted
  and discarded: driving it through `PYTHONPATH` with `cwd` at the repo root let
  `sys.path[0] = ''` shadow the swapped tree, so every arm silently tested the
  same files.  The two in-process runs above do not have that flaw — each
  asserts the resolved `lumenairy.__file__` before collecting.)

### Files touched in this pass

```
lumenairy/io/codegen.py                 +45  -3   V10 (two comment sites + the helper), V11
lumenairy/io/prescriptions_zemax.py    +142 -45   V6, V7 (shared helpers + both loaders)
lumenairy/io/prescriptions_builders.py   +4  -1   V9
lumenairy/optimize/merit_terms.py       +19  -3   V3
lumenairy/optimize/__init__.py           +1  -1   V12
lumenairy/optimize/core.py               +1  -1   V12
tests/unit/test_audit2609_a10_io.py     +13  -0   V6 pin (strengthened in place)
tests/unit/test_audit2609_a10_optimize.py +14 -30 V5, V12
tests/unit/test_audit2609_a10_verify.py  944 lines, 22 tests (was 17)
```

Plus the two from the first pass (§8): `io/prescriptions_code_v.py` (+47 −11)
and `io/storage.py` (+1).  `ruff check` is clean on `lumenairy/io/`,
`lumenairy/optimize/` and all five WP-A10 test files.
`docs/…/fixes/WP-A10_CHANGELOG.md` was corrected for V1 (the `MM`/`IN` migration
sentence) and V8 (the I2 "enter the window" wording, `thorlabs_lens`'s
once-per-process warning, and `COMM`'s single writer path), and gained entries
for the V3, V6, V10 and V11 behaviour changes.
