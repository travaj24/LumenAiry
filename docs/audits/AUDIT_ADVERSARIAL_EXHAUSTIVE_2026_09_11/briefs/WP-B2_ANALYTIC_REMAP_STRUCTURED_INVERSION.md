# WP-B2 (Wave 4) -- analytic lens L9: invert the displaced remap on its structured launch grid, then raise the lattice

Repository: `D:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics\Lumenairy`, branch `audit-fixes-2026-09`, HEAD 81d5b586
(= release 5.46.0, Wave 3 closed).  Two other Wave-4 engineers are working concurrently on `lumenairy/elements/lenses_maslov.py` and
the asymptotic modules (WP-B1) and on `lumenairy/propagators/hf.py` / `hfpi.py` (WP-B3); never touch those files.

Read first: `docs/TESTING_STANDARDS.md` (S1-S5), `CONVENTIONS.md` sec. 2 (error prefix), the comment rule in `CONTRIBUTING.md`
("Modules with a history document": `lumenairy/elements/_lens_real.py` has `docs/history/lumenairy.elements._lens_real.md`; a code
change MUST re-record it in the same change with `python scripts/record_history_fingerprints.py lumenairy/elements/_lens_real.py --reason "..."`).
Source comments describe what the code does NOW and why; no version narrative (the history-lint ratchet fails on it).

Then: the audit finding L9 in `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11.md`;
`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A2_REPORT.md` section 2 (L9: what shipped -- the Newton loops stopping at
their fixed point, the `_warn_if_remap_lattice_smooths` warning, the `_DISP_REMAP_2D_N_SIDE` constant) and section 6 item 2 (the deferred
design, reproduced below), `WP-A2_CHANGELOG.md` (the L9 entries), `VERIFY_WP-A2.md` (its L9 verdict, the reflection-instability
measurement), and `tests/unit/test_niche_p10_transverse_walk_remap.py` (the mirror / centroid / EE80 symmetry trio that is the
acceptance test).  `docs/subsystems/real_lens.md` is the living contract of the family; keep it true.

## The design (WP-A2 section 6 item 2; implement it)

The displaced-carrier remap in `apply_real_lens` inverts the launch->exit map with a Delaunay triangulation (QHull) of the exit points.
The launch fan is a REGULAR grid, so its exit map is a smooth curvilinear grid that can be inverted with `scipy.ndimage.map_coordinates`
(or an equivalent structured inverse: a Newton inversion seeded from the paraxial map, as `_lens_traced.py`'s inversion does) instead of
QHull.  That removes the 181-ray resolution ceiling, the 2x triangulation cost (measured 16.44 s Delaunay against 8.43 s structured on
the cos-grid analogue), and -- the reason it must come FIRST -- the reflection instability WP-A2 measured: the triangulation of a
near-degenerate exit set resolves cells arbitrarily and not reflection-stably, which is what made raising the lattice alone a regression.
After the structured inversion is in and proven, raise `_DISP_REMAP_2D_N_SIDE` to the value the measurement supports and expose a
public `displaced_n_side` kwarg (validated with the sec. 2 prefix; documented with units and default).

## Deliverable

1. **Structured inversion** of the displaced remap, bit-for-bit reflection-symmetric on the p10 trio (mirror the input about x and y;
   the output must mirror to the last bit, or state the floor and why), and byte-identical to the shipped path wherever the shipped path
   was already correct -- if that is not achievable, measure the difference against an ORACLE (an independent exact trace of the same
   fan plus a direct Kirchhoff/RS sum on a small fixture; the repo's `repro/orch/` scripts show the method) and show the new path is the
   closer one.
2. **Lattice raise + kwarg** once (1) holds: `displaced_n_side` on `apply_real_lens` (and the `LensNumerics` field in
   `lumenairy/elements/lens_config.py` -- `from_kwargs`/`to_kwargs` round trip, `tests/unit/test_audit2609_a16_lens_config_round_trip.py`
   must keep passing; if you cannot own lens_config.py, put the exact field text under requested changes), the constant re-derived with
   its measurement, and `_warn_if_remap_lattice_smooths` retired or restated to the new regime.
3. **Performance**: report wall-clock medians of interleaved runs, old path against new, on the cos-grid analogue and on one
   covering-array fixture (`tests/unit/test_audit2609_a15a_lens_covering_array.py` has the factory).  TESTING_STANDARDS S1: no wall-clock
   assertions in tests; pin operation counts or structure instead.
4. **Pin it** in `tests/unit/test_audit2609_b2_displaced_remap_inversion.py`: the symmetry trio as derived envelopes, the byte-identity
   (or oracle-refereed improvement), the kwarg validation, a fail-before (the reflection instability on the old path, reproduced).
5. **Re-record** every history document you touch; report `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-B2_REPORT.md`
   and `WP-B2_CHANGELOG.md` (5.47.0 release text, `### Fixed -- ...` / `### Performance -- ...` / `### Added -- ...` style, finding L9
   named, `lumenairy/elements/_lens_real.py:N` citations on non-trivial lines, a Migration note for anything whose default output moves).

## Verification set (all green when you finish)

`tests/unit/test_niche_p10_transverse_walk_remap.py`, `tests/unit/test_audit2609_a2_*.py` and the VERIFY-A2 file(s) (`a2` in `tests/unit/`),
`tests/unit/test_audit2609_a16_*.py`, `pytest tests/unit -k real_lens`, `tests/unit/test_audit2609_a15a_lens_covering_array.py`,
`python validation/run_all.py test_lenses`, `ruff check`, `python scripts/record_history_fingerprints.py --check`,
`tests/unit/test_audit2609_a17_history_lint.py`.

## Rules

* Every python run with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`.
* NO git write commands of any kind; read-only `git show` / `git archive` / `git log` are fine.  Do not kill processes.
* Own only: `lumenairy/elements/_lens_real.py`, `lumenairy/elements/lens_config.py` (the one new field), their `docs/history/` documents,
  the new b2 test file, your two report files.  Anything else: "requested changes outside my ownership" with the exact edit.
* Comments say what the code does now and why.  Finish with the report's full text as your final message.
