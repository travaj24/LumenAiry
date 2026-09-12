# WP-A17 SWEEP-1 -- version-history relocation, `propagators/` (non-lens), `analysis/`, `sources/`

Finding: **P2-4** (`TESTS-ARCH.md:394`) / consolidated report **sec. 14 V6**
and **sec. 15.7**.  This is the library-wide continuation of WP-A17 part 1
(`propagators/carrier.py`, `carrier_field.py`, `fft_infra.py`), run on a
disjoint partition alongside two sibling sweeps.

**Partition:** every `.py` under `lumenairy/propagators/` except `carrier.py`,
`carrier_field.py`, `fft_infra.py` (part 1), `gbd.py`, `fga.py` (lens family,
later) and `propagation.py` (another agent); every `.py` under
`lumenairy/analysis/` and `lumenairy/sources/`.  **45 modules in scope; 29
relocated, 16 deliberately left alone (sec. 6).**

---

## 1. Summary

| finding | status | files | tests | oracle | measured before -> after |
|---|---|---|---|---|---|
| P2-4 / V6 -- version-history narrative in the source | **fixed** | 29 modules, 33 820 -> 33 180 lines | `tests/unit/test_audit2609_a17_history_relocation.py` (auto-discovers the 29 new `docs/history/*.md`) | SHA-256 of the docstring-free AST and of the comment-free / docstring-free token stream, recorded from each PRE-relocation file | loose-classifier history 11 034 -> 9 738 lines (32.6 % -> 29.3 %); **strict "pre-fix this did A" 8 065 -> 5 618 (-30 %)**; 178 blocks / 1 754 prose lines moved |
| sec. 15.7 -- comments that state the OPPOSITE of the code | **fixed, 5 sites in 3 modules** | see sec. 3 | the identity gate + the existing behaviour tests | the code beneath each comment | 5 false claims corrected, 0 behaviour change |
| V1-class falsifiability of the move | **verified** | all 29 | the checker's two fingerprints | -- | **both fingerprints byte-identical on all 29 modules** |

`ruff check lumenairy/` -- *All checks passed*.
`python -c "import lumenairy"` -- OK.

**Nothing the interpreter executes changed, anywhere, in any of the 29
modules.**  Not "equivalent" -- identical, under two independent front ends.

---

## 2. What moved, and where it went

29 documents, one per module, named by the **dotted module path**
(`docs/history/lumenairy.propagators.dispatch.md`) because basenames collide
across packages.  Each holds its blocks **verbatim** under the source line
they came from in the pre-relocation file, with a line-ordered table of
contents and a *Left in the source:* note per block, exactly as part 1 did.

| module | blocks | prose lines moved | lines left behind | document | doc lines |
|---|---|---|---|---|---|
| `lumenairy/propagators/dispatch.py` | 29 | 404 | 219 | `lumenairy.propagators.dispatch.md` | 694 |
| `lumenairy/sources/core.py` | 40 | 296 | 137 | `lumenairy.sources.core.md` | 672 |
| `lumenairy/analysis/detector.py` | 6 | 124 | 86 | `lumenairy.analysis.detector.md` | 234 |
| `lumenairy/propagators/hfpi.py` | 13 | 123 | 76 | `lumenairy.propagators.hfpi.md` | 284 |
| `lumenairy/propagators/system.py` | 8 | 83 | 61 | `lumenairy.propagators.system.md` | 195 |
| `lumenairy/propagators/mhs.py` | 6 | 78 | 53 | `lumenairy.propagators.mhs.md` | 170 |
| `lumenairy/propagators/asymptotic_jax_twin.py` | 10 | 72 | 48 | `lumenairy.propagators.asymptotic_jax_twin.md` | 205 |
| `lumenairy/analysis/psf_mtf_otf.py` | 5 | 70 | 55 | `lumenairy.analysis.psf_mtf_otf.md` | 165 |
| `lumenairy/propagators/asymptotic_aberration_tensor.py` | 4 | 47 | 24 | `lumenairy.propagators.asymptotic_aberration_tensor.md` | 136 |
| `lumenairy/propagators/asm.py` | 5 | 46 | 36 | `lumenairy.propagators.asm.md` | 132 |
| `lumenairy/propagators/asymptotic_modes.py` | 5 | 46 | 21 | `lumenairy.propagators.asymptotic_modes.md` | 137 |
| `lumenairy/analysis/ao.py` | 2 | 44 | 35 | `lumenairy.analysis.ao.md` | 103 |
| `lumenairy/analysis/through_focus.py` | 4 | 41 | 30 | `lumenairy.analysis.through_focus.md` | 122 |
| `lumenairy/propagators/asymptotic.py` | 6 | 40 | 20 | `lumenairy.propagators.asymptotic.md` | 138 |
| `lumenairy/propagators/rs.py` | 5 | 37 | 21 | `lumenairy.propagators.rs.md` | 128 |
| `lumenairy/analysis/ghost.py` | 3 | 31 | 18 | `lumenairy.analysis.ghost.md` | 96 |
| `lumenairy/propagators/asymptotic_maslov.py` | 4 | 31 | 18 | `lumenairy.propagators.asymptotic_maslov.md` | 110 |
| `lumenairy/propagators/mft.py` | 4 | 31 | 26 | `lumenairy.propagators.mft.md` | 105 |
| `lumenairy/propagators/result.py` | 3 | 25 | 15 | `lumenairy.propagators.result.md` | 91 |
| `lumenairy/analysis/beam_stats.py` | 3 | 22 | 15 | `lumenairy.analysis.beam_stats.md` | 85 |
| `lumenairy/propagators/vectorial_hfpi.py` | 3 | 16 | 5 | `lumenairy.propagators.vectorial_hfpi.md` | 81 |
| `lumenairy/analysis/strehl.py` | 2 | 14 | 12 | `lumenairy.analysis.strehl.md` | 67 |
| `lumenairy/analysis/field.py` | 2 | 6 | 6 | `lumenairy.analysis.field.md` | 64 |
| `lumenairy/propagators/hf.py` | 1 | 6 | 4 | `lumenairy.propagators.hf.md` | 51 |
| `lumenairy/propagators/asymptotic_canonical_fit.py` | 1 | 5 | 4 | `lumenairy.propagators.asymptotic_canonical_fit.md` | 53 |
| `lumenairy/propagators/fresnel.py` | 1 | 5 | 5 | `lumenairy.propagators.fresnel.md` | 53 |
| `lumenairy/analysis/zernike.py` | 1 | 5 | 5 | `lumenairy.analysis.zernike.md` | 49 |
| `lumenairy/analysis/coronagraph.py` | 1 | 3 | 2 | `lumenairy.analysis.coronagraph.md` | 52 |
| `lumenairy/analysis/opd.py` | 1 | 3 | 1 | `lumenairy.analysis.opd.md` | 54 |
| **TOTAL (29)** | **178** | **1 754** | **1 058** | | **4 526** |

### 2.1 Line and history counts, before -> after

`loose` is the audit's own classifier (a block counts if it mentions a version
number, the word "audit", or a `20xx-xx-xx` date); `strict` is the shape the
finding actually names (`vN.N (`, `pre-fix`, `used to`, `formerly`,
`previously`, `the old `, `was wrong`, `superseded`, `no longer`, `pre-vN`).
Both are measured per block, on the same classifier, on the same files, before
and after.

| module | lines b -> a | loose b -> a | strict b -> a | fingerprints |
|---|---|---|---|---|
| `propagators/dispatch.py` | 1698 -> 1518 | 846 -> 599 | 720 -> 329 | identical |
| `sources/core.py` | 3381 -> 3222 | 1378 -> 1079 | 1027 -> 637 | identical |
| `analysis/psf_mtf_otf.py` | 1579 -> 1566 | 641 -> 582 | 345 -> 150 | identical |
| `propagators/asymptotic.py` | 893 -> 875 | 394 -> 370 | 363 -> 210 | identical |
| `propagators/mhs.py` | 755 -> 732 | 182 -> 109 | 193 -> 46 | identical |
| `propagators/hfpi.py` | 1658 -> 1611 | 592 -> 536 | 421 -> 309 | identical |
| `propagators/system.py` | 1919 -> 1899 | 757 -> 722 | 584 -> 478 | identical |
| `analysis/detector.py` | 960 -> 924 | 444 -> 389 | 454 -> 352 | identical |
| `propagators/result.py` | 244 -> 236 | 133 -> 123 | 119 -> 21 | identical |
| `propagators/asymptotic_modes.py` | 893 -> 870 | 222 -> 154 | 232 -> 143 | identical |
| `analysis/through_focus.py` | 1981 -> 1972 | 335 -> 327 | 205 -> 120 | identical |
| `analysis/ghost.py` | 1034 -> 1023 | 421 -> 337 | 174 -> 90 | identical |
| `propagators/asymptotic_jax_twin.py` | 1216 -> 1194 | 341 -> 306 | 232 -> 173 | identical |
| `propagators/mft.py` | 1073 -> 1070 | 519 -> 514 | 284 -> 228 | identical |
| `analysis/zernike.py` | 876 -> 878 | 308 -> 262 | 46 -> 0 | identical |
| `propagators/asymptotic_aberration_tensor.py` | 1376 -> 1355 | 536 -> 491 | 483 -> 438 | identical |
| `analysis/ao.py` | 1355 -> 1348 | 398 -> 389 | 325 -> 281 | identical |
| `propagators/asymptotic_maslov.py` | 756 -> 743 | 246 -> 202 | 110 -> 66 | identical |
| `propagators/asm.py` | 1405 -> 1397 | 591 -> 581 | 390 -> 351 | identical |
| `propagators/fresnel.py` | 593 -> 595 | 210 -> 210 | 68 -> 30 | identical |
| `propagators/vectorial_hfpi.py` | 763 -> 754 | 302 -> 274 | 146 -> 118 | identical |
| `analysis/field.py` | 1492 -> 1494 | 63 -> 63 | 173 -> 147 | identical |
| `analysis/beam_stats.py` | 772 -> 767 | 58 -> 36 | 22 -> 0 | identical |
| `propagators/rs.py` | 773 -> 759 | 290 -> 274 | 279 -> 263 | identical |
| `analysis/strehl.py` | 557 -> 557 | 142 -> 128 | 25 -> 11 | identical |
| `propagators/asymptotic_canonical_fit.py` | 1354 -> 1355 | 125 -> 124 | 119 -> 106 | identical |
| `analysis/opd.py` | 1257 -> 1257 | 141 -> 139 | 156 -> 154 | identical |
| `propagators/hf.py` | 1009 -> 1009 | 412 -> 410 | 274 -> 272 | identical |
| `analysis/coronagraph.py` | 198 -> 199 | 7 -> 7 | 96 -> 95 | identical |
| **TOTAL (29)** | **33 820 -> 33 180** | **11 034 -> 9 738** | **8 065 -> 5 618** | **all identical** |

Eight modules gain 1-3 lines net: each relocated module carries a two-line
pointer comment below its docstring naming its document, which the checker's
`test_the_source_still_points_at_the_history_document` requires (four modules
-- `sources/core.py`, `dispatch.py`, `hfpi.py`, `asymptotic_maslov.py` --
already name theirs inside a condensed rationale, so they got no extra
comment).

### 2.2 The four patterns that carried the history

**(a) A deprecation chronology.**  Overwhelmingly the largest class, and it is
what makes `sources/core.py` (40 blocks) and `dispatch.py` (29) the two biggest
documents.  Every removal wave recorded which release deprecated a call shape,
which horizon it announced, how often that horizon slipped and which release
finally executed it -- `version_removed='5.0'` shipping through v5.29 with the
banner re-scheduled to v5.32 and then executed at v5.30, three times over in
`sources/core.py` alone.  None of it describes the code.  What DOES describe
the code -- that a legacy call shape is still *detected*, so the `TypeError`
can name the exact canonical form rather than degrading to Python's generic
arity message -- stayed at all four collector sites.

**(b) A comment correcting an earlier COMMENT.**  15 sites.  These are notes
about the documentation's own history and are worthless in the source once the
text they retract is gone.  The sharpest are
`propagators/vectorial_hfpi.py`'s two paragraphs, which existed solely to
retract an appeal to "the full m-theory dipole formalism" -- a formalism that
exists nowhere in the library -- and `propagators/mhs.py`'s module docstring,
which carried a paragraph explaining that the paragraph above it used to
describe ray bundles and Huygens-surface integrals this module does not
implement.  Also: `asymptotic_modes.py`'s "trapezoidal quadrature" and
"(Nx, Ny)" retractions, `hf.py`'s "the standard `1/(i lambda z)` Van Vleck
factor", `asm.py` / `asymptotic_canonical_fit.py`'s "as this comment used to
claim (and its two siblings still did)", `opd.py`'s "the docstring ... is what
was wrong", `psf_mtf_otf.py`'s `otf[0, 0]`-is-DC retraction, `mft.py`'s
"NOT the '< 0.1 %' this note claimed before v5.46",
`asymptotic_aberration_tensor.py`'s "VERIFY-A4: the pre-v5.46-final wording
gave the SQUARED factor", `asymptotic_maslov.py`'s "contrary to the pre-v5.46
wording", `asymptotic.py`'s "the v5.30 W6-A4 note used to say it was", and
`rs.py`'s retracted "agree to machine precision".

**(c) A fail-before / fix-after table whose fail column is the only content.**
`analysis/psf_mtf_otf.py`'s two resolution metrics each carried a two-column
accuracy table (v5.29 binned vs v5.30 sub-pixel).  The fix-after column is the
measured accuracy of the code that ships and stayed (S5); the v5.29 column is
the defect and moved.  `analysis/detector.py`'s S11-4 table is the extreme
case: its post-fix column is `1.0000` in all five rows, so the table's entire
information content was the pre-fix column.

**(d) A block appended to instead of edited, until it described several
different implementations at once.**  This is the mechanism behind the
sec. 15.7 contradictions in sec. 3, and it also produced pure duplication --
`propagators/mft.py`'s H-build comment states the open-interval band limit
**twice**, fourteen lines apart, because two audit rounds each appended their
own version.

### 2.3 What deliberately did NOT move

**Measured derivations of live constants**, which
`docs/TESTING_STANDARDS.md` S5 requires a numeric bar to carry.  Named
explicitly so the orchestrator can see the line drawn:

* `asymptotic_aberration_tensor.py` -- the W4-T1 chirp-Nyquist argument
  `n >= 4*extent*v_max/lambda` with its fringe-rate convergence measurements;
  the 7-rung accuracy/cost table behind `_SIGMA_GRID_N_MAX_DEFAULT = 256`; the
  32-point probe-grid calibration for the default `w_o`; the W4-T2 sign-flip /
  ptp table behind `curvature_matched_basis`; the Gram-matrix clamp figures on
  `sigma_grid_extent`.  This module is 1 376 lines of which 483 score as
  "history" on the strict classifier and only 47 moved -- the rest is
  measurement.
* `hfpi.py` -- the K18 source-area derivation and its `sum(weights)/exact`
  table; the RS-I composition in `_reemission_measure`; the W9-14
  sampling-adequacy occupancy figures.
* `system.py` -- the W9-12 `ray_subsample` derivation, which sets a live
  default *against* the first instinct to align it with its two siblings and
  carries both arms of the measurement.
* `asm.py` -- the 2-shift-fold algebra with its odd-N parity gate, and the
  audit-P1 integer-DC-anchor centroid measurement.
* `asymptotic_maslov.py` -- the W6-A2 scale-relative convergence argument.
* `asymptotic_modes.py` -- the `_grid_corner_fingerprint` proof.
* `rs.py` -- the measurement showing `bandlimit=True` is five decades WORSE on
  this propagator, which is why its default is `False`.
* `asymptotic.py` -- the `maslov_tracking` proof that `arg det M` is confined
  to `(-pi, +pi)`, and the raster-unwrap measurement.
* `asymptotic_jax_twin.py` -- the audit-Y3 `stop_gradient` note and the W6-A10
  vignetting-parity table.

**Live deprecation and migration statements.**  `sources/core.py`'s
`cosmic_ray_rate` -> `cosmic_ray_rate_per_m2_per_s` recipe, the `seed=` ->
`rng=` and `sigma=` -> `w0=` conversions, `hfpi.py`'s `output_grid` ->
`output_shape` warning, `system.py`'s v5.0 aperture-schema migration,
`mhs.py`'s two-constructors-disagree `.. note::`, and every
`.. versionchanged::` that tells a caller what to do now (condensed, not
removed).

**Do-not-do-this notes written as history.**  A large class, and the judgement
call that shaped most of this sweep.  A sentence like "Pre-fix, an uncast real
`E_in` was fed into the pyFFTW dispatcher, which permanently blacklisted the
bare SHAPE for ALL dtypes" describes a hazard that is *still reachable* -- the
guard is the only thing stopping it.  Those were rewritten into the present
tense at the source site (and the original wording recorded in the document)
rather than deleted, because a reader deciding whether the guard can go needs
the failure, not the release number.

---

## 3. Contradicting comments corrected (sec. 15.7, "worse than none")

Five, in three modules.  All are docstring/comment-only and covered by the
identity gate.

1. **`analysis/detector.py`, the integration banner (33 lines).**  Appended to
   three times without ever being edited, it described three algorithms in
   sequence: "The old approach used integer truncation ..."; "**Here we use
   scipy.ndimage.zoom** to resample to the detector pitch with proper
   anti-aliased integration ..."; "4.10: proper area integration ... Pre-4.10
   used scipy.ndimage.zoom(order=1) ..."; "For non-integer ratios **first
   uniform-filter to anti-alias, then sample at the new pixel centers, scaled
   by pixel_pitch^2**".  Claims 2 and 4 are **false**: `scipy.ndimage` is not
   imported by this module and `zoom` occurs nowhere outside that comment
   (verified by grep), and the live non-integer branch assigns each field
   sample's energy `I_field * dx_field**2` to the pixel containing its physical
   centre -- no filter, no `pixel_pitch**2`.  A reader following the comment
   would have believed the module interpolates.  Corrected: the source now
   names the two live branches and keeps the dimensional argument against
   point-sampling interpolation as a *do-not-do-this*.
2. **`propagators/asymptotic_modes.py`, `_lg_mode_conj_stack`.**  Its cache key
   was widened three times and each round appended a paragraph, so the
   docstring's own summary still named the grid ORIGIN as the key -- two
   paragraphs above the paragraph explaining that the origin is not enough
   (an `indexing='xy'` grid and an `indexing='ij'` grid share shape, pitch AND
   origin).  Corrected: the key is stated once, with the collision and the
   measurement (worst relative error 8.232e+00) that forced the corner
   fingerprint.
3. **`sources/core.py`, `PartialCoherenceMCF`** -- "MCF-aware downstream
   propagators ... are NOT in v4.15.1 scope and are deferred to v4.16+".  The
   library is at v5.46 and `lumenairy/_validation.py` still refuses a
   `PartialCoherenceMCF` at every propagator entry point.  Corrected to "are
   not implemented", pointing at the guard that enforces it.
4. **`sources/core.py`, `Source.gaussian_schell`** -- "MCF-aware downstream
   propagators are not in v4.15.x scope".  Same defect, same correction.
5. **`sources/core.py`, `Source.gaussian_schell`** -- "A future
   `Source.realizations()` per-realization iterator is in scope for v4.16+ but
   is NOT shipped in v4.15.3".  30+ minor releases later there is still no such
   API (`grep` finds the name only in this docstring, its sibling, and one test
   docstring).  Corrected to "There is no per-realisation iterator on
   `Source`; unpack the ensemble as above."

A post-sweep grep over the whole partition for the two mechanised patterns --
`(this|the) (comment|docstring|note|wording) ... used to (say|claim|read)` and
the removed-shim chronology (`shipped unremoved through v`,
`re-scheduled the banner`, `version_removed='5.0'`, `slipping a third time`) --
returns **zero hits**.

---

## 4. Method and verification

Per module: record both fingerprints of the untouched file; write a plan of
`(line range -> replacement lines)` blocks; extract each block **verbatim**
from the pre-edit file into the document; apply the replacements bottom-up;
re-compute both fingerprints and **refuse to write anything if either moved**
(the applier raises rather than writing a half-applied file).  178 blocks went
through that path; no plan was ever applied with a moved fingerprint.

Two extra gates beyond the checker, because the checker cannot see prose:

* **A string-literal cross-check over the whole test corpus.**  Every string
  constant of >= 12 characters in `tests/**` and `validation/**` that appears
  in a module's pre-relocation text but NOT in its post-relocation text was
  listed and read.  79 candidates; 77 were incidental collisions (`make_singlet`,
  `BIT-IDENTICAL`, `set_default_complex_dtype` used for unrelated purposes).
  Two were real, and both were fixed before the test run: `'P16 resolved'`
  (`propagators/result.py`, sec. 5) and `'superseding'`
  (`analysis/detector.py`, sec. 5).
* **`ruff check lumenairy/`** and `import lumenairy` after every batch.

---

## 5. Retired / adjusted prose assertions

**One assertion adjusted; none retired outright.**

`tests/unit/test_niche_audit_w4_input_kind.py::test_shack_hartmann_declares_field_not_pupil`
asserted `'superseding' in src` against `lumenairy/analysis/detector.py`.  Its
stated reason was that the v5.32 correction sat next to the v4.15.5
`Input kind: 'pupil'` gloss it overturned, so "the next reader who greps that
comment must find the correction attached to it, not just a comment silently
contradicted by the line beneath it".  With the gloss itself relocated there is
nothing left to contradict, so the assertion is now over-specified: it pins the
*retraction* rather than the *absence of the false claim*.  It tightens to the
strictly stronger pair:

```python
assert "Input kind: 'pupil'" not in src        # the false claim is ABSENT
assert "``input_kind='field'``, not 'pupil'" in src   # the live rule is stated
```

with a comment recording why, and pointing at
`docs/history/lumenairy.analysis.detector.md` for the retraction.  That file
also covers `analysis/psf_mtf_otf.py` (mine) and `_guard_calls` over the whole
library; **only the `detector.py` assertion was touched.**

**One source phrase restored rather than retiring a test.**
`propagators/result.py`'s `.. note::` heading read "**P16 resolved (v5.30,
roadmap Part F1).**"; my first pass rewrote it to "**Iteration stays 2-item,
permanently**", which broke
`test_niche_audit_w4_p5_return_contract.py::TestFlipIsDocumented::test_the_decision_is_recorded_on_the_class`
(`assert 'P16 resolved' in doc`).  "P16 resolved" is a true present-tense
statement, so the heading now reads "**P16 resolved: iteration stays 2-item,
permanently** (roadmap Part F1)" and the assertion stands unchanged.

The same pass restored two phrases in `propagators/dispatch.py`'s `propagate`
docstring -- the "Return contract, settled in v5.30 (audit P5, roadmap Part F1
-- EXECUTED)" heading and a one-line statement that the transition
`DeprecationWarning` is retired with the flip -- both pinned by
`TestFlipIsDocumented`.  Both are present-tense facts (the contract IS settled;
there IS no deprecation here), so they stay in the source and no assertion
moved.

**Test files examined and NOT modified.**  Found by the string-literal
cross-check above plus `grep -rln "getsource\|__doc__\|getdoc\|read_text("
tests/unit` filtered to the partition's modules and symbols (65 files).  The
ones that pin prose on my modules and pass unchanged:

| file | what it pins on my modules | outcome |
|---|---|---|
| `test_niche_audit_w4_p5_return_contract.py` | `getdoc(propagate)` (settled contract, `option 4`, `roadmap_deferred`, `permanent, supported escape hatch`, `P16`, `2-item`, Warns section); `getsource(propagate)` (`wrap = True if return_result is _NO_DEFAULT`, `if not return_result:` absent); `PropagationResult.__doc__` | passes (2 phrases restored, sec. 5) |
| `test_niche_audit_w3_propagators.py` | `getdoc(propagate)` -- the full auto table, `N_F`, `Q = lambda`, `deferred`, `audit P16`, `two`/`three`; `getdoc(PropagationResult.__iter__)` | passes |
| `test_niche_audit_w9_dispatch2.py` | `getdoc(_auto_select_method)` must contain `CANONICAL`; `getdoc(_select_asm_variant)` must contain `_auto_select_method` + `canonical`; `getsource(_auto_select_method)` body must not contain `events_json` | passes -- the CANONICAL paragraph was kept in full and the delegation is still named |
| `test_niche_audit_w3_raytrace_sources.py` | `getsource(core)` must not contain `version_removed='5.0',` or the re-schedule constant; the `_RETURN_KIND_UNSET` branch scan (code-only) | passes; the relocation removed the last prose mentions too |
| `test_niche_audit_a1_radial_metrics.py` | `rayleigh_resolution` / `fwhm_resolution` docstrings must name `_radial_profile_subpixel` and carry a `samples / first zero` table | passes -- the single-column (fix-after) table keeps both |
| `test_niche_audit_a2_encircled_energy_radius.py` | `encircled_energy_radius.__doc__`: no `sub-percent`, must have `independent of the array size`, `dark ring` | passes (untouched) |
| `test_v4_15_3_agent_b.py` | `Source.gaussian_schell.__doc__` / `Source.schell_model.__doc__` must mention `invariant` and the `(n_realizations, Ny, Nx)` shape | passes -- the invariant-break section was kept |
| `test_v4_15_dispatcher_pin_validate_grid_params.py` | counts COMMENT lines as executable inside a 15-line body-head window | passes; every edit shrinks a body-head comment, which can only move the validator earlier |
| `test_audit_sources.py` | `getsource(create_led_source)` must not contain `scale-inverted` | passes |
| `test_niche_audit_w4_input_kind.py` | `_guard_calls` over `detector.py` / `psf_mtf_otf.py`; `psf_mtf_otf.py` must not contain `would be ideal once` / `correct in the interim` | passes; one assertion adjusted (sec. 5) |
| `test_niche_audit_w5_shim_removals.py`, `test_v4_16_1_agent_b.py`, `test_v5_2_walker_sentinel_reduce.py` | `hasattr(core, '_RETURN_KIND_UNSET' ...)` | passes |

---

## 6. Modules deliberately left alone (partition complete)

16 of the 45 modules in the partition carry no relocatable history.  Listed so
the orchestrator can close the partition:

**Nothing flagged at all** (zero strict-classified blocks):
`propagators/__init__.py`, `propagators/_bluestein.py`,
`analysis/__init__.py`, `analysis/plotting.py`, `analysis/polychromatic.py`,
`sources/__init__.py`.

**Re-export shells whose "history" IS their description.**
`analysis/core.py` -- "v5.1.0 split: this file is now a thin back-compat
re-export shell.  The actual implementation moved to the topical submodules"
followed by the submodule map, and "Every previously-public name imported from
`lumenairy.analysis.core` continues to resolve here".  Both sentences are what
the module *is* and what it *guarantees*; relocating them would leave a reader
of a 69-line file with no idea why it exists.

**Live why-comments with a `Pre-fix ...` framing, and nothing else.**
`propagators/ensemble.py`, `propagators/sas.py`,
`propagators/subaperture.py`, `propagators/vector_diffraction.py`,
`analysis/aberration.py`, `analysis/coherence.py`,
`analysis/image_plane_wfe.py`, `analysis/interferometry.py`,
`analysis/phase_retrieval.py`.  Each carries one to four sentences of the form
"Pre-fix X happened", but in every case X is a hazard the live guard exists to
prevent and is still reachable if the guard is removed -- e.g.
`ensemble.py`'s `shape=(0, Ny, Nx)` passing the ndim check,
`image_plane_wfe.py`'s `1/N_chief` sign inversion after an odd mirror count,
`subaperture.py`'s window-centring contract.  Under the standing rule a
why-comment stays.  None of the nine contains a comment correcting an earlier
comment, a superseded derivation, or a removed-shim chronology; the two
mechanised greps in sec. 3 return zero hits across all of them.

If the orchestrator wants the `Pre-fix ->` present-tense rewrite applied to
these nine as well (it is the treatment `asm.py`, `beam_stats.py` and
`strehl.py` received in this sweep), that is ~25 further blocks and about half
a day; it changes no information, only tense.

---

## 7. Tests run

All with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`.

### 7.1 The checker

```
python -m pytest tests/unit/test_audit2609_a17_history_relocation.py -q
```

Every fingerprint, TOC-order, header and source-pointer assertion passes on all
29 of my documents.  Two failure classes remain, **neither caused by this
sweep** -- see sec. 8.

### 7.2 Per-module unit tests

The 65 test files that (a) reference a partition module or one of its public
symbols AND (b) read source or docstrings, plus the checker:

```
python -m pytest <65 files> tests/unit/test_audit2609_a17_history_relocation.py \
    -q --no-header -p no:cacheprovider --tb=line -rf
```

```
81 failed, 3108 passed, 27 skipped, 67 warnings in 493.16s (0:08:13)
```

**Every one of the 81 failures is inside
`test_audit2609_a17_history_relocation.py`; zero failures in the 65 behaviour
and prose files.**  Re-running the checker alone, at the same tree, classifies
them exactly:

```
78 x test_the_header_names_a_real_module_and_two_fingerprints   <- sec. 8.1
 2 x test_the_fingerprints_are_actually_sensitive               <- sec. 8.2
80 failed, 407 passed in 19.89s
```

(The 81st in the combined run is the same file collected twice -- once from the
65-file list, once explicitly.)  29 of the 78 name-assertion failures are my
documents; the other 49 belong to the two sibling sweeps, which hit the same
checker defect.  On my 29 documents, **every other assertion passes** --
including both fingerprint identities, the TOC-order and anchor checks, and
`test_the_source_still_points_at_the_history_document`.

Earlier, per-batch runs on the same tree:

* `sources/core.py` batch (29 files): `1065 passed, 44 skipped in 50.37s`.
* `dispatch.py` batch (8 files): `618 passed, 44 skipped in 51.56s` after the
  two restored phrases (sec. 5); `2 failed` before them, both in
  `TestFlipIsDocumented`.
* `psf_mtf_otf.py` (2 files): `80 passed in 12.29s`.
* `test_niche_audit_w4_input_kind.py` + `test_niche_audit_w4_p5_return_contract.py`
  after the sec. 5 edits: `292 passed in 8.53s`.

### 7.3 Validation topic files

```
python validation/run_all.py --quiet test_dispatch test_hfpi test_mhs test_hf \
    test_vectorial_hfpi test_detector test_field test_ao test_coherence
ALL 9 files passed.
```

```
python validation/run_all.py --quiet test_analysis test_asymptotic test_propagation \
    test_advanced_diffraction test_subaperture test_image_plane_wfe test_features \
    test_new_propagators_smoke
ALL 8 files passed.
```

### 7.4 Static gates

`ruff check lumenairy/` -- *All checks passed*.
`python -c "import lumenairy"` -- OK.
(One benign `Failed to write cache file ... Access is denied` warning from
ruff: another agent held `.ruff_cache` at that moment.)

---

## 8. Two checker defects found (NOT caused by this WP -- not mine to fix)

Both are in `tests/unit/test_audit2609_a17_history_relocation.py`, which the
brief assigns to the orchestrator.  I did not touch it.

### 8.1 The dotted-name generalisation builds the name from the ABSOLUTE path

```python
src_path = REPO_ROOT / header["module"]          # absolute
dotted = ".".join(src_path.with_suffix("").parts)
assert name in (src_path.stem, dotted)
```

`src_path.parts` starts at the drive root, so `dotted` comes out as

```
'D:\\.Metacept.Neurophos.Python_Test_Scripts.Free_Space_Optics.Lumenairy.lumenairy.sources.core'
```

and every dotted-named document is rejected.  This hits **all three concurrent
sweeps** -- **78 failures** at the time of writing (29 mine, 49 from the two
siblings: `lumenairy.glass`, `lumenairy.io.storage`,
`lumenairy.elements.*`, `lumenairy.optimize.*`, `lumenairy.raytrace.*`).
One-line fix:

```python
dotted = ".".join(pathlib.PurePosixPath(header["module"]).with_suffix("").parts)
```

Per the brief I report this rather than renaming 29 documents to basenames --
which would collide anyway (`analysis/core.py` vs `sources/core.py`,
`analysis/field.py`, `pmm/stack.py` vs `rcwa/stack.py`).

### 8.2 `test_the_fingerprints_are_actually_sensitive` cannot find a mutation
target in a small module

Mutation 1 looks only at `fn.body[-1]` of each function and requires that last
statement to be single-line:

```python
doomed = fn.body[-1]
if doomed.lineno != doomed.end_lineno:
    continue
...
assert cut is not None, "no single-line statement found to delete"
```

`lumenairy/analysis/coronagraph.py` has exactly ONE function
(`coronagraph_contrast_curve`, 20 body statements) and it ends with a
multi-line `return {...}`, so no candidate is ever found and the test fails
with "no single-line statement found to delete".  **Verified pre-existing:**
running the same search against my untouched pre-relocation copy of
`coronagraph.py` finds no candidate either, so the failure is a property of the
module, not of the relocation.  `lumenairy.optimize.multi_objective` (another
sweep) fails identically -- these two are the only such failures in the whole
registry.

Fix: scan every statement of the body, not just the last one --

```python
for doomed in reversed(fn.body):
    if doomed.lineno != doomed.end_lineno:
        continue
    ...
```

---

## 9. Requested changes outside my ownership

1. **`tests/unit/test_audit2609_a17_history_relocation.py`** -- the two
   one-line fixes in sec. 8.  Until 8.1 lands, 29 of my documents fail
   `test_the_header_names_a_real_module_and_two_fingerprints` on the name
   assertion alone; every other assertion on them passes, including both
   fingerprints.
2. **`CHANGELOG.md`** -- assembled from
   `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A17_SWEEP1_CHANGELOG.md`.
3. **A CI lint on new history blocks.**  Part 1 already recorded this as the
   root cause ("each audit round adds a new history block, never an update to
   the existing one"), and this sweep is the evidence: `mft.py` states the same
   open-interval band limit twice, fourteen lines apart, and
   `asymptotic_modes.py`'s cache-key summary went stale because three rounds
   appended below it instead of editing it.  A lint that fails when a comment
   block matching the strict pattern is ADDED to a module that already has a
   `docs/history/` document would stop the backlog re-accumulating.
   `CONTRIBUTING.md` and the CI config are not in my ownership.

---

## 10. Files touched

**Modified -- comments and docstrings only, both fingerprints identical
(29):**

`lumenairy/propagators/`: `asm.py`, `asymptotic.py`,
`asymptotic_aberration_tensor.py`, `asymptotic_canonical_fit.py`,
`asymptotic_jax_twin.py`, `asymptotic_maslov.py`, `asymptotic_modes.py`,
`dispatch.py`, `fresnel.py`, `hf.py`, `hfpi.py`, `mft.py`, `mhs.py`,
`result.py`, `rs.py`, `system.py`, `vectorial_hfpi.py`

`lumenairy/analysis/`: `ao.py`, `beam_stats.py`, `coronagraph.py`,
`detector.py`, `field.py`, `ghost.py`, `opd.py`, `psf_mtf_otf.py`,
`strehl.py`, `through_focus.py`, `zernike.py`

`lumenairy/sources/`: `core.py`

**Modified -- test (one assertion, sec. 5):**
`tests/unit/test_niche_audit_w4_input_kind.py`

**New (31):** 29 documents under `docs/history/` (listed in sec. 2), plus

* `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A17_SWEEP1_REPORT.md`
  (this file)
* `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A17_SWEEP1_CHANGELOG.md`

---

## 11. Residual risk

* The recorded fingerprints pin each of the 29 modules against its
  pre-relocation self **forever**.  That is the point, but it means a
  deliberate code change to any of them must re-record the two hashes in the
  same commit; the checker's assertion message says so.
* The checker proves the edit was documentation-only.  It cannot tell a good
  docstring from a bad one.  Prose quality was reviewed by reading a
  context diff at every one of the 178 edit sites before applying (the applier
  has a dry-run mode that prints `-`/`+` hunks with three lines of context);
  the continuity faults that pass found -- a replacement that duplicated the
  line below it, two that dropped a trailing blank line and so ran a paragraph
  into an RST section header, one that began a sentence with "Nor" after its
  antecedent had moved -- were fixed before the apply.
* The nine modules in sec. 6 keep `Pre-fix ...` phrasing on live guards.  A
  future census on the loose classifier will still count them.  That is a
  deliberate reading of the standing rule (a why-comment stays), stated here so
  it can be overruled cheaply.
