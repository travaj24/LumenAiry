# WP-B3b (Wave 4, follow-up to WP-B3) -- the K6 call sites: `system.py`'s Fresnel and SAS legs and `_lens_real.py`'s gap legs

Repository: `D:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics\Lumenairy`, branch `audit-fixes-2026-09`.  WP-B3 (commit 284daccc) added
`resample_field(method='chirpz')` and `fresnel_propagate_mft` is available; WP-B2 (the newest `_lens_real.py` commit) has released that file.  Other
Wave-4 engineers are editing `lumenairy/elements/lenses_maslov.py` + `lumenairy/propagators/asymptotic*.py`, `lumenairy/propagators/carrier.py` +
`carrier_field.py`, `lumenairy/elements/rcwa/*`, `eme/*`, `bor/*`, `lumenairy/elements/pmm/*`, `lumenairy/analysis/*`, `lumenairy/sources/*`,
`lumenairy/raytrace/*`, and a verifier may touch `_lens_real.py`'s REMAP functions; never touch any of those, and in `_lens_real.py` edit ONLY
`_propagate_gap`'s two `resample_field` calls.

Read first: `docs/TESTING_STANDARDS.md` (S1-S5), `CONVENTIONS.md` sec. 2 and 7, the comment rule in `CONTRIBUTING.md` ("Modules with a history
document": `lumenairy/propagators/system.py` and `lumenairy/elements/_lens_real.py` have documents; re-record each with
`python scripts/record_history_fingerprints.py <module path> --reason "..."` in the same change).  Then
`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-B3_REPORT.md` section 5.1 and 5.2 -- the exact edits and the measurements behind them
(the chirp-Z leg is a clean win on a contained field, marginally worse on a grid-filling one, and WRONG in the converging direction where the
window exceeds the reconstruction's period; the Fresnel leg is better served by evaluating `fresnel_propagate_mft` straight onto the chain grid) --
and `WP-A5_REPORT.md` section 6 item 3 (K6).

## The edits (WP-B3 section 5.1 (a)/(b) and 5.2)

1. `system.py`, the `fresnel` leg: replace propagate-then-resample with a direct `fresnel_propagate_mft(E, z, wavelength, current_dx,
   current_dx, N_out=E_in.shape[-1], dy_in=current_dy, dy_out=current_dx)` evaluation onto the chain grid, deleting the resample block and its
   `_warn_system_resample_crop` (the MFT's own faithful-zone warning takes over, period `lambda |z| / dx_in`).  This MOVES the numbers of every
   `fresnel` leg that resampled: that is the point, and it needs its own regression pass -- report before/after on every fixture in the
   verification set that exercises it, against the direct evaluation as the reference (relL2 and window power), and write the Migration note.
2. `system.py`, the `sas` leg: keep `resample_field`, gate `method=('chirpz' if dx_new >= current_dx else 'spline')` (unit MTF where the
   pitch coarsens; the spline where the window would exceed the period), keep `_warn_system_resample_crop`.
3. `_lens_real.py::_propagate_gap`, the `sas` and `fresnel` calls: the same gate (`'chirpz' if dx_new >= dx else 'spline'`); note this leg
   propagates in glass so it lands in the converging direction more often -- report how often on the covering-array fixtures.

For each: byte-identity where the gate selects the spline (prove it against `git archive <HEAD>^ lumenairy` extracted READ-ONLY into your scratch
directory under `C:\Users\AndrewT\AppData\Local\Temp\claude\D--Metacept-Neurophos-Python-Test-Scripts-Metasurface-QWP\78f7e8ef-9607-4e41-ba1d-ebd1c2e74c7e\scratchpad\b3b\`,
child process, cwd + PYTHONPATH = the archive, `lumenairy.__file__` asserted, never through pytest), and an oracle-refereed improvement where it
selects chirp-Z or the direct MFT (the direct Fresnel evaluation, or a band-limited ASM at the same pitch, is the reference).

## Deliverable

Derived envelopes in a new `tests/unit/test_audit2609_b3b_resample_call_sites.py` (S5; a structural pin that the gate selects the spline whenever
`dx_new < dx`, a fail-before that the un-gated chirp-Z returns the 2x2 tiling power on a converging fixture, the byte-identity, the improvement);
re-record both history documents; report `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-B3b_REPORT.md` (summary table, per-item
sections, files touched, tests run with counts/durations, requested changes, deferred) and `WP-B3b_CHANGELOG.md` (5.47.0 release text,
`### Changed -- ...` style of `fixes/WP-A5_CHANGELOG.md`, K6 named, `lumenairy/propagators/system.py:N` citations on non-trivial lines, the
Migration note for the Fresnel leg).

## Verification set (all green when you finish)

`tests/unit/test_audit2609_a5_*.py` + VERIFY-A5 files, `pytest tests/unit -k "system or propagate_through or fresnel or sas"`,
`tests/unit/test_audit2609_b3_propagator_kernels.py`, `tests/unit/test_audit2609_a2_*.py`, `tests/unit/test_audit2609_b2_displaced_remap_inversion.py`,
`tests/unit/test_audit2609_a15a_lens_covering_array.py`, `pytest tests/unit -k real_lens`, `python validation/run_all.py test_propagation test_lenses
test_dispatch`, `ruff check`, `python scripts/record_history_fingerprints.py --check` (your two documents), `tests/unit/test_audit2609_a17_history_lint.py`.

## Rules

* Every python run with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`; one process at a time (the box is shared).
* NO git write commands of any kind; read-only git is fine.  Do not kill processes.  The orchestrator commits with an explicit file list.
* Own only: `lumenairy/propagators/system.py`, the two `resample_field` calls in `lumenairy/elements/_lens_real.py::_propagate_gap`, their
  `docs/history/` documents, the new b3b test file, your two report files.
* Comments say what the code does now and why.  Finish with the report's full text as your final message.

## Verifier notes folded in after VERIFY-B3 (read before implementing)

* **The gate's general form.**  VERIFY-B3 sec. 5 confirmed the chirp-Z leg returns replicas exactly when `N_out * dx_out > N_in * dx_in`, so the
  rule the WP-B3 report wrote as `dx_new >= current_dx` is only correct because `N_out == N_in` at those call sites.  Spell it in the general
  form -- `method=('chirpz' if N_out * dx_new <= E.shape[-1] * current_dx else 'spline')`, per axis where the grid is not square -- or keep the
  short form with a comment naming the `N_out == N_in` special case.  The gate is CONSERVATIVE: a contained field in the converging direction
  is fine under chirp-Z (P/P_in 1.000181 = the direct evaluation) but the period test warns and routes it to the spline; say so in the report.
* **hfpi.py has moved** (VERIFY-B3's V1/V2 fixes are committed); nothing there is yours.
* **One small extra edit is yours** (VERIFY-B3 F6): `resample_field`'s docstring must say that the unit MTF holds when the output window is
  exactly one reconstruction period (it read 0.99996 at x1.7 where the extent-preserving `N_out` rounds off the period) and at which scale
  factors it is exact.  For that sentence only, `lumenairy/propagators/mft.py` is in your ownership; re-record its history document if the
  fingerprint moves (a docstring-only edit should not).
* **Byte-identity is archive-to-archive**: extract `git archive <HEAD>^ lumenairy` and compare against `git archive <HEAD> lumenairy` or your own
  tree's modules imported in a child process; never against the shared working tree, which carries other engineers' uncommitted edits.
