# WP-A20 — loose ends: the lens-library migration, `rs_alias_free_distance`, two retired-behaviour tests

Branch `audit-fixes-2026-09`.  Three items handed over by earlier work
packages: WP-A9 §6.1 / VERIFY-A9 §5.3 (the pre-U1 folded library entry),
WP-A15b §5.3 (the `_rs_alias_free_distance` rename), and two tracker
tests that pinned behaviour WP-A5 has since replaced.

Every "before" number below was measured on the current working tree
BEFORE the corresponding edit and every "after" by re-running the same
script; the pre-fix comparators for the two propagator items were
measured by loading `6b801ffa^:lumenairy/propagators/hf.py` and
`58ad836c^:lumenairy/ui/model.py` in-process alongside the current ones,
not quoted from a report.  PySide6 is not installed on this workstation;
the designer fixtures run through the auditor's Qt stub
(`repro/UI/stub`), and nothing skips on its absence.

---

## 1. Summary table

| item | status | files:lines | tests (`tests/unit/…`) | oracle | measured before → after |
|---|---|---|---|---|---|
| **WP-A9 §6.1 / VERIFY-A9 §5.3** — pre-U1 folded library entry | **fixed** (with one residual REPORTED, not repaired) | `lumenairy/user_library.py:719-953`, `:955-989` | `test_audit2609_a20_user_library_and_rs.py::TestA20FoldedLibraryBackfill` (13) | the designer model's OWN `build_trace_surfaces()` ABCD — the list the 2-D layout and spot diagram are drawn from, sharing no code with the prescription round trip past the `Surface` dataclass | EFL **0.15479922951064448 → 0.1825479198383674 m** (layout: 0.1825479198383674, bit-identical); BFL **0.1521620959930726 → 0.1304206774044291 m**; semi-diameters **[0.006, 0.006, 0.0127] → [0.025, 0.006, 0.006]**; `is_mirror` **[F,F,F] → [T,F,F]** |
| **WP-A15b §5.3** — `rs_alias_free_distance` | **fixed** (needs a 2-line companion in `lumenairy/__init__.py`, §5.1) | `lumenairy/propagators/rs.py:38, 180-236, 5-9, 331, 541-548` | `…::TestA20RsAliasFreeDistanceIsPublic` (7, incl. 4 parametrised) | the closed form written out in the test; and `kernel='auto'` vs the explicit `'transfer'` / `'spatial'` branches | public name **absent → present** in `rs.__all__`; `_rs_alias_free_distance is rs_alias_free_distance` **True**; `auto` bit-identical to `transfer` below the distance and to `spatial` at it |
| **WP-A11 §5 item 5** — `test_hf_chunk_output_is_KEPT` | **fixed** (test restated) | `tests/unit/test_niche_audit_w5_shim_removals.py:82-87, 114, 592-699` | `…::TestPropagatorInertKwargRemovals::test_hf_chunk_output_is_KEPT_and_FUNCTIONAL` | closed form `1 probe + 17 × ceil(n_out/n_chunk)` `opl_fn` evaluations | **`DID NOT WARN` (red) → passes**; `opl_fn` calls **2448 for every `chunk_output` (pre-K22) → 2448 / 1225 / 613 / 154 / 35** for 1 / 2 / 4 / 16 / auto, field bit-identical |
| **WP-A5 K17** — `test_vector_aperture_diffraction_has_projection_kwarg` | **fixed** (test restated to the documented default) | `tests/unit/test_v5_4_6_wave4_polarization.py:5-6, 84-122` | WP-A5's changelog migration note + `test_audit2609_a5_propagators.py::TestK17VectorialHfpi` | **`assert True is False` (red) → passes**; pin moved `False → True`, kwarg kept keyword-only |

---

## 2. Per item

### 2.1 `user_library.load_lens` — the pre-U1 folded entry  [WP-A9 §6.1, VERIFY-A9 §5.3]

**What was wrong.**  `SystemModel.to_prescription()` did not emit
per-surface `is_mirror` / `semi_diameter` / `is_stop` until the U1/U2
fix, so a folded design already in a user's `~/.lumenairy/library/lenses`
has a chronological `surfaces` list — its mirror IS in it — with none of
those flags.  Loading it gave:

* no mirror anywhere, so the fold analysed as an air→air refracting
  surface (and a Zemax-signed negative post-mirror gap became a literal
  backwards propagation);
* the mirror handed the FOLLOWING lens's aperture, because
  `surfaces_from_prescription`'s `elements` fallback filters to
  `element_type == 'surface'` entries while the index counts all of them
  — VERIFY-A9 §2.4 closed the half of this that a *present* `is_mirror`
  discriminates, which is exactly the key a pre-fix file lacks.

**What I changed and why.**  `load_lens` now calls
`_backfill_folded_surface_keys` (`user_library.py:742`) on the
deserialised prescription.  `load_lens` is the library's only lens load
path — `ui/library_dock.py:172,196` goes through it — so one site covers
the whole surface.

The migration is deliberately narrow, and it states its rule instead of
guessing.  **Trigger** (all three):

1. `surfaces` is a non-empty list of dicts and `elements` is a list of
   dicts of exactly the same length;
2. at least one `elements` entry is an `element_type == 'mirror'` — an
   UNFOLDED entry is already resolved correctly, so it gets nothing at
   all and loads byte-identically (this is why the trigger is "has a
   mirror", not "lacks the keys");
3. no `surfaces` entry carries `is_mirror` — a producer that writes the
   key is never second-guessed.

**Positional-consistency check** (the "refuse to guess" rule): every
`elements` entry's `element_type` must be `'mirror'` or `'surface'`, its
`radius` must equal the positionally matching surface's radius bit for
bit (`inf` matches `inf`, NaN matches nothing), and a `'surface'`
entry's `glass_before` / `glass_after` must equal that surface's.  The
radius check is exact rather than tolerant because both numbers come from
the same `radius_mm * 1e-3` in the producer and JSON round-trips a float
bit-for-bit.  When the trigger fires but the check fails — the realistic
case being a lens-only `surfaces` list whose `elements` happens to have
the same length — **nothing is written** and a `UserWarning` names the
index that disagreed.

**How I verified.**

Fixture built by the CURRENT `to_prescription()` under the Qt stub, then
stripped of exactly the three keys (plus `stop_index`, plus `is_stop` on
the `elements` entries — the pre-fix exporter wrote none of them, checked
against `git show 58ad836c^:lumenairy/ui/model.py`), so the "pre-fix
file" differs from the modern one in nothing else.  Concave fold mirror
R = −200 mm / semi-diameter 25 mm, then an N-BK7 singlet R1 = 80 mm /
semi-diameter 6 mm at a Zemax-signed −30 mm; d line.

| | EFL [m] | BFL [m] | semi-diameters [m] | `is_mirror` |
|---|---|---|---|---|
| designer layout (`build_trace_surfaces()`) | 0.1825479198383674 | 0.1304206774044291 | 0.025, 0.006, 0.006 | T, F, F |
| modern save → `load_lens` | 0.1825479198383674 | 0.1304206774044291 | 0.025, 0.006, 0.006 | T, F, F |
| **pre-fix save → `load_lens`, before** | **0.15479922951064448** | **0.1521620959930726** | **0.006, 0.006, 0.0127** | **F, F, F** |
| **pre-fix save → `load_lens`, after** | **0.1825479198383674** | **0.1304206774044291** | **0.025, 0.006, 0.006** | **T, F, F** |

−15.2 % EFL and +16.7 % BFL before; `==` (not `approx`) against both the
modern save and the layout after.  The unfolded control is unchanged in
every row: EFL 0.048874328099208435 / BFL 0.047875190344944356 before and
after, loaded dict key-for-key equal to the stored one, no key added, no
warning.

**Fail-before, measured.**  `load_lens`'s pre-WP-A20 body was exactly
`return _deserialize_prescription(data['prescription'])`, so neutralising
the helper in-process (`ul._backfill_folded_surface_keys = lambda rx,
name='': rx`, via a pytest plugin) reproduces it.  Under that:
**7 of the 13 back-fill tests fail, 13 of 20 in the file pass** — the 7
being every assertion about the migration, the warnings and the refusal.
The six that still pass are the counter-pins that assert *no change*
(unfolded, modern, `.zmx`), the two pure-helper tests, and the
fail-before measurement itself, which is what they should do on both
sides; and
`test_a20_prefix_folded_entry_without_the_backfill_is_wrong` passes on
both sides by construction — it IS the pre-fix measurement, asserted
in-suite so the direction of the change is pinned, not just its presence.

**Scope, measured.**  Counter-pins in the same class, all passing:

* an UNFOLDED pre-fix entry → loaded dict `==` stored dict, no key
  added anywhere, no warning, EFL/BFL identical to a modern save;
* a MODERN save → loaded dict `==` saved dict, silent;
* the real lens-only `.zmx` shape
  (`repro/IO-OPTIMIZE/cb_mirror.zmx`: 2 `surfaces`, 3 `elements` of which
  one is the mirror, no `is_mirror` key on any surface) → untouched,
  silent, same resolved semi-diameters before and after;
* three inconsistent pairings (a radius that does not match, a
  non-optical `elements` entry, a disagreeing glass) → warned and
  untouched, `loaded == stored`.

**Residual, reported not repaired.**  The same pre-fix exporter also
dropped a coordinate break's transfer thickness in the `cb_post` case (a
tilted element behind a mirror).  Measured by exporting one model through
both `ui/model.py` versions loaded side by side:

| topology | pre-fix `thicknesses` | modern `thicknesses` | `all_thicknesses` (both) |
|---|---|---|---|
| mirror + TILTED lens (cb_post) | `[0.0, 0.004]` | `[0.04, 0.004]` | `[0.04, 0.004, 0.1]` |
| the other eight topologies swept | identical to modern | — | — |

40 mm of air vanished from `thicknesses` while `all_thicknesses` kept it.
That is a different key and out of this deliverable's scope, so
`_warn_lost_coord_break_gap` (`user_library.py:872`) DETECTS it and
quotes both numbers rather than rewriting anything.  Detection is exact,
not heuristic: within the chronological shape `thicknesses[i]` and
`all_thicknesses[i]` describe the same gap and agree on every topology
the designer can build — swept on nine (negative Zemax-signed
post-mirror gaps, two mirrors, a mirror last, a tilt before a mirror, a
tilt between two lenses, the unfolded control), the single disagreement
being the dropped carry — and the check only runs when the entry has
coordinate breaks, which is the only way the old exporter could lose a
gap.  On that fixture: layout EFL/BFL 1.0459951945424169 /
1.4670304058769554 m; no migration 0.15479922951064448 /
0.1521620959930726; migrated but gap still missing 0.2824843175588363 /
0.2851214510764082.  Deferred repair design in §6.1.

**Residual risk.**  (a) A user with a legacy folded entry sees its EFL
change by ~15 % between library versions; that is the point of the fix
and it warns once per entry.  (b) `save_lens` accepts any dict, so a
hand-built prescription could in principle satisfy all five conditions
and be migrated; the radius + glass fingerprint makes an accidental match
require the two lists to describe the same surfaces in the same order, at
which point the back-fill is correct by construction.  (c) The
`coord_breaks` thickness warning could in principle fire on a hand-built
entry whose `all_thicknesses` legitimately differs from `thicknesses`; it
is a warning that quotes the numbers, not a refusal.

### 2.2 `rs_alias_free_distance`  [WP-A15b §5.3]

**What was wrong.**  Nothing numerically — this is an API-shape item.
`2*N*dx**2/wavelength` is the `kernel='auto'` branch point and the
distance below which `kernel='spatial'` refuses, i.e. the number a caller
needs in order to pick `z` / `N` / `dx` for an RS step, and it was
private, so WP-A15b could not re-export it.

**What I changed.**  Renamed `_rs_alias_free_distance` →
`rs_alias_free_distance` (`rs.py:180`), added to `rs.__all__` (`:38`),
kept `_rs_alias_free_distance = rs_alias_free_distance` as a same-object
alias (`:236`) with a `#:` comment saying why, updated the two internal
callers (`:541-548`) and the two docstring references (`:331`, `:5-9`),
and gave the public function an `Examples` block and a `See Also` in the
surrounding style.  The alias is silent, not deprecated: two audit
regression files import the private name and are out of my ownership, and
a `DeprecationWarning` there would be noise, not a signal.

**How I verified.**  `git show HEAD:lumenairy/propagators/rs.py | grep -c
"def rs_alias_free_distance"` → **0**, so the public name did not exist
before this change and the two identity tests fail on HEAD by
`AttributeError`.  After: `rs.rs_alias_free_distance is
rs._rs_alias_free_distance` → `True`; `__name__` / `__qualname__` are the
public spelling; value equals `2*N*dx**2/lambda` exactly at
`(64, 2 µm, 632.8 nm)`, `(128, 1 µm, 1.55 µm)`, `(256, 0.5 µm, 400 nm)`
and `(32, 10 µm, 10.6 µm)`.  Functionally: on `N=64, dx=1 µm,
λ=633 nm` (`z_alias` = 2.0221e−4 m), `kernel='auto'` at `z = z_alias/2`
is `array_equal` to `kernel='transfer'` and `kernel='spatial'` raises
naming `2*N*dx**2/wavelength`; at `z = z_alias` exactly, `'auto'` is
`array_equal` to `'spatial'`.

**Residual risk.**  The `__all__` entry makes
`test_v4_16_0_walker_all_symmetry.py` fail until the top-level re-export
lands — see §4 and §5.1.  That is the walker doing its job; it prints the
exact two-line fix.

### 2.3 `test_hf_chunk_output_is_KEPT` → `…_is_KEPT_and_FUNCTIONAL`  [WP-A11 §5 item 5]

**What was wrong.**  The test asserted
`pytest.warns(DeprecationWarning, match='chunk_output')` on the premise
that the kwarg was an inert no-op out of scope for the W5 removal wave.
WP-A5's K22 went the other way: `chunk_output` is now the number of
OUTPUT pixels evaluated per vectorised `opl_fn` batch, the deprecation is
gone, and the test failed `DID NOT WARN`.

**What I changed.**  The W5 scope boundary is preserved (the kwarg was
never in the removal wave) but the pin is now "kept AND functional".
Asserted: the parameter is still in the signature; **no**
`DeprecationWarning` fires for any value; the `opl_fn` call count matches
the closed form; the batched contract hands `(n, 1, 1)`-shaped output
coordinates instead of Python scalars; and the returned field is
`array_equal` across every batch size.  The module docstring's
"explicitly NOT in scope" bullet was updated to record the reversal.

**Oracle.**  `opl_fn` is evaluated 17 times per batch with Van Vleck on
(`Phi` plus the 16 cross-Hessian stencil corners) and once per batch with
it off, over `ceil(n_out/n_chunk)` batches, plus exactly one broadcast
probe when `n_chunk > 1`.  A call count has no error floor, so the bar is
exact equality and nothing is timed.  On the file's existing 12×12 →
12×12 fixture (`n_out` = 144):

| call | closed form | measured | `s2x` shape |
|---|---|---|---|
| `chunk_output=1` | 17·144 | **2448** | `()` |
| `chunk_output=2` | 1 + 17·72 | **1225** | `(2,1,1)` |
| `chunk_output=4` | 1 + 17·36 | **613** | `(4,1,1)` |
| `chunk_output=16` | 1 + 17·9 | **154** | `(16,1,1)` |
| `None` (auto → 113) | 1 + 17·2 | **35** | `(113,1,1)` |
| `16`, Van Vleck off | 1 + 1·9 | **10** | `(16,1,1)` |

Field `array_equal` across all of them.  The auto batch is derived in the
test from `hf._HF_CHUNK_TARGET_BYTES` rather than a literal, so a retune
of that target moves the pin with it.

**Fail-before.**  `6b801ffa^:lumenairy/propagators/hf.py` exec'd
in-process under the same probe: **2448 calls for `chunk_output` = 1, 16
AND None**, `s2x` a Python scalar in every case, and an explicit value
raising `DeprecationWarning: chunk_output is deprecated and has no
effect`.  The new test fails three ways against it (counts 15.9× and
69.9× off, shapes all `()`, and the no-warning assertion).

### 2.4 `test_vector_aperture_diffraction_has_projection_kwarg`  [WP-A5 K17]

**The default changed deliberately and WP-A5 documented it**, so per the
brief the pin is restated rather than reported as a discrepancy.
`WP-A5_CHANGELOG.md` §"Fixed -- propagators/vectorial_hfpi" carries the
explicit migration note: *"`vector_projection` defaults to `True` and the
operation it names changed.  `vector_projection=False` reproduces the
pre-v5.46 behaviour exactly."*  It also carries the measurement that
licensed the flip: with the projection off the "vectorial" propagator was
bit-identical to two scalar HFPI runs (`max|Ex_vec − Ex_scalar|`
1.9e−23; 45° `max|Ey/Ex − 1|` ≤ 1.1e−16, i.e. zero depolarisation
anywhere) at twice the cost, and the old opt-in path discarded 15.9 % of
`|E|²`.  With it on: `|Ez|²` 1.40e−2, cross-pol `|Ey|²` 9.3e−5, 45°
`max|Ey/Ex − 1|` 0.143, `|E|²` conserved to 1 − 1.1e−15.

**What I changed.**  `default is False` → `default is True`, plus two new
assertions that the kwarg still EXISTS and is keyword-only (that is the
part that protects migrating callers, since `False` is the documented way
back to the old behaviour), plus the derivation and the pointer to
`test_audit2609_a5_propagators.py::TestK17VectorialHfpi`, which measures
the physics.  This file only pins the signature.  The module docstring's
P3-24 line was corrected — the kwarg is no longer "opt-in".

---

## 3. Files touched

Modified (all within my ownership):

| file | what |
|---|---|
| `lumenairy/user_library.py` | `_FOLDED_SURFACE_KEYS` (`:719`), `_same_radius` (`:722`), `_backfill_folded_surface_keys` (`:742`), `_warn_lost_coord_break_gap` (`:872`), `load_lens` docstring + call (`:955-989`) |
| `lumenairy/propagators/rs.py` | `__all__` (`:38`), `rs_alias_free_distance` + `Examples`/`See Also` + alias (`:180-236`), module docstring (`:5-9`), docstring reference (`:331`), internal callers (`:541-548`) |
| `tests/unit/test_niche_audit_w5_shim_removals.py` | scope bullet (`:82-87`), `_HF_CHUNK_TARGET_BYTES` import (`:114`), `test_hf_chunk_output_is_KEPT_and_FUNCTIONAL` (`:592-699`) |
| `tests/unit/test_v5_4_6_wave4_polarization.py` | module docstring (`:5-6`), restated P3-24 pin (`:84-122`) |

New:

| file | what |
|---|---|
| `tests/unit/test_audit2609_a20_user_library_and_rs.py` | 20 test items (11 functions + 4 params) in 2 classes: 13 for the library migration, 7 for `rs_alias_free_distance` |
| `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A20_CHANGELOG.md` | changelog text |
| `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A20_REPORT.md` | this file |

Nothing else was edited.  In particular `lumenairy/__init__.py`,
`lumenairy/propagators/hf.py`, `lumenairy/propagators/vectorial_hfpi.py`,
`lumenairy/propagators/propagation.py`, `lumenairy/ui/model.py` and
`lumenairy/raytrace/trace.py` are untouched.

---

## 4. Tests run

All with `OPENBLAS_NUM_THREADS=1`, `-q --no-header -p no:cacheprovider`.

| command | result | duration |
|---|---|---|
| `pytest tests/unit/test_audit2609_a20_user_library_and_rs.py` | **20 passed** | 1.5 s |
| the same, with the back-fill neutralised in-process (fail-before) | **7 failed, 13 passed** | 2.1 s |
| `pytest tests/unit/test_niche_audit_w5_shim_removals.py tests/unit/test_v5_4_6_wave4_polarization.py` — BEFORE my edits | **2 failed**, 39 passed, 44 skipped | 2.0 s |
| the same — AFTER | **41 passed**, 44 skipped | 1.6 s |
| `pytest` ×14 files: the A20 file + `test_c2_s4_19_user_library_safe_eval` + `test_audit_p1_glass_registration` + `test_audit_w4_glass_registry_meshgrid` + `test_v5_6_glass_memo` + `test_niche_audit_w3_infra` + `test_v4_15_1_agent_e` + `test_audit2609_a15b_optional_and_knobs` + `test_niche_audit_w5_shim_removals` + `test_v5_4_6_wave4_polarization` + `test_audit_misc` + `test_audit_polarization` + `test_audit2609_a11_polar_sources_infra` + `test_audit2609_a9_ui` | **656 passed, 53 skipped** | 168.9 s |
| `pytest` ×9 files: the RS/propagator surface — `test_audit2609_a5_propagators`, `test_audit2609_a5_verify_rs_and_rw`, `test_audit2609_a5_followup`, `test_v5_1_0_agent_c_split`, `test_audit_propagation`, `test_niche_audit_w3_propagators`, `test_v5_4_6_wave2_p1_parity`, `test_niche_audit_p1_odd_n_freq_grid`, `test_niche_audit_w4_input_kind` | **773 passed** | 77.4 s |
| `pytest tests/unit/test_audit_except_budget.py test_v4_16_0_walker_all_symmetry.py test_v4_16_2_dispatcher_pin_doc_consistency.py test_audit2609_a15b_reexports.py test_audit2609_a15b_lazy_and_layering.py` | 64 passed, **1 failed** (the walker — §5.1) | 10.4 s |
| `python validation/run_all.py test_io test_propagation test_hf --quiet` | **3/3 files pass** | 13.8 s |

Skips are genuine capability gaps (no `cupy`, no PySide6 for the sibling
UI files, the host-specific SHA-256 digest block in the W5 file), not
resource preconditions, and none of them are mine.

**Qt-stub hygiene, checked explicitly.**  The new file installs the
auditor's `PySide6.QtCore` stub, captures the fixtures, and then lifts
both the stub and the `lumenairy.ui` modules it built back out of
`sys.modules` — permanently, since nothing after the capture needs Qt.
Verified in both collection orders (`a20` then `a9_ui`, and `a9_ui` then
`a20`): **75 passed, 6 skipped** either way, with
`test_v4_15_1_agent_e.py` still correctly reporting
"`lumenairy.ui.model` imports PySide6 at module load" for its five
skipped UI pins.

### Pre-existing / other-agent failures

Only one failure anywhere in the sets above, and it is the one my own
change asks for: `test_v4_16_0_walker_all_symmetry.py::test_all_submodule_entries_reexported_or_exempt`
(§5.1).  `test_audit_except_budget.py` — which WP-A5 reported as
pre-existing-red — **passes** on the current tree, and my two narrow
`except (TypeError, ValueError)` clauses do not count against its
`except Exception:` budget.

---

## 5. Requested changes outside my ownership

### 5.1 `lumenairy/__init__.py` — the top-level `rs_alias_free_distance` re-export  [REQUIRED, 2 lines]

The walker prints the exact change:

```
lumenairy.propagators.rs.__all__ exports 'rs_alias_free_distance' but it is
not in ``lumenairy.__all__``.  Either:
  1. Add ``from .propagators.rs import rs_alias_free_distance`` to
     lumenairy/__init__.py and append 'rs_alias_free_distance' to the
     top-level __all__, OR ...
```

Option 1 is the one WP-A15b §5.3 asked for.  Concretely:

* add `from .propagators.rs import rs_alias_free_distance` beside the
  existing `from .propagators.propagation import (...)` block
  (`lumenairy/__init__.py:226`) — a direct import from `.propagators.rs`
  avoids also having to widen `propagators/propagation.py`'s re-export
  list, which is likewise outside my ownership;
* add `'rs_alias_free_distance',` to the top-level `__all__`, next to
  `'rayleigh_sommerfeld_propagate'` (`:1307`).

Until this lands, `test_v4_16_0_walker_all_symmetry.py` is red with that
message and nothing else.  Owner: the orchestrator (or whoever holds
`lumenairy/__init__.py` this round).

### 5.2 Informational — no other change is needed

`propagators/propagation.py` does not need to re-export the new name
(it re-exports `rayleigh_sommerfeld_propagate` only, and 5.1's direct
import bypasses it); `lumenairy/ui/library_dock.py` needs no change (it
goes through `load_lens`); `raytrace/trace.py` and
`raytrace/jax_trace.py` need no change (VERIFY-A9 §2.4 and §5.1 already
cover the resolver, and after the back-fill a migrated entry takes the
`chronological` branch that is already correct there).

---

## 6. Deferred, with designs

### 6.1 Rebuild a pre-U1 entry's lost coord-break gap from `all_thicknesses`  [small]

Measured and detected (§2.1), not repaired.  Design: inside
`_backfill_folded_surface_keys`, after the key back-fill and under the
same positional-consistency guarantee, replace `thicknesses[i]` with
`all_thicknesses[i]` wherever they differ and the entry has coordinate
breaks.  The invariant that licenses it — `thicknesses[i] ==
all_thicknesses[i]` for the chronological shape — was swept on nine
designer topologies with exactly one (correct) disagreement, and
`all_thicknesses` is written straight from the element spacings and so
never lost the gap.  Effort ~1 h including a widened topology sweep.  Not
done here because it rewrites a key outside this deliverable's scope, on
an invariant established over the designer's exporter rather than over
every producer whose dict can reach `save_lens`; the warning already
stops the silent-wrong outcome and names the recoverable value.

### 6.2 A public one-shot migration entry point  [small]

`_backfill_folded_surface_keys` is private and only runs on
`load_lens`.  A `lumenairy.user_library.migrate_lens(name)` that loads,
migrates and re-SAVES would let a user clear the warning without opening
the designer.  Three lines on top of what is here, plus a walker entry
and a test.  Not added because it is new public API the audit did not ask
for.  Effort ~30 min.

### 6.3 `_rs_alias_free_distance` retirement  [tiny, scheduled elsewhere]

The alias is silent by design (§2.2).  If the repo wants the private
spelling gone, the horizon-and-`_deprecation.py` machinery applies, but
the three importers are audit regression files owned by WP-A5 and would
have to move first.  Effort ~15 min once those files are free.

---

## 7. Changelog

`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A20_CHANGELOG.md`
— one `### Fixed -- user_library` block (with the migration note), one
`### Changed -- propagators/rs` block (flagging the required companion
re-export), and one `### Fixed -- tests` block for the two restated pins.

Scratch scripts (session scratchpad, not committed): `a20_baseline.py`
(the library before/after), `a20_hf_chunk.py` / `a20_hf_prefix.py` (the
`chunk_output` ladder and its pre-K22 comparator), `a20_residual.py` /
`a20_thick_invariant.py` (the coord-break gap and the nine-topology
sweep), `a20_prefix_plugin.py` (the fail-before pytest plugin).
