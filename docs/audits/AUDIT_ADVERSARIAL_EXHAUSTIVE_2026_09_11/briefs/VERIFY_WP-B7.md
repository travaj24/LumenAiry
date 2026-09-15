# VERIFY-WP-B7 -- independent adversarial re-verification of WP-B7 (the asymptotic family: FGA routing, Y4 performance, Y5 structure, the uniform asymptotics, GBD kernel clipping, the JAX thin-screen displacement error, the S6 fallback statistic, the pupil-chart sizing)

Repository: `D:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics\Lumenairy`, branch `audit-fixes-2026-09`.  WP-B7 is commit f64444ec (its parent
is the pre-change library for byte-identity and fail-before probes).  You own `lumenairy/elements/lenses_maslov.py`, `lenses_gbd.py`,
`_lens_jax.py` (the Maslov screen only), `lumenairy/propagators/asymptotic*.py`, `fga.py`, `gbd.py` and their `docs/history/` documents for this
pass; nothing else.  WP-B11b is editing CONCURRENTLY `lumenairy/propagators/sas.py`, `lumenairy/elements/lens_config.py`, `doe.py`, `_lens_real.py`,
`_lens_traced.py`, `lenses.py`, `pmm/stack2d.py`, `carrier.py` (docstring) and the b11 / a15a / a16 test files -- never open those; their
fingerprint lines may read DRIFT until they re-record, and `tests/unit/test_audit2609_a15a_lens_covering_array.py`'s Maslov cells may move
while they work.  WP-B11b will hand the orchestrator two edits for YOUR files after you finish (a `lenses_maslov.py:282` import change and the
Maslov / JAX warning `stacklevel` sites); do not pre-empt them.

You did not write WP-B7.  Your job is to try to break it.  Read `docs/TESTING_STANDARDS.md` (S1-S5), `CONVENTIONS.md` sec. 2 and 7,
`CONTRIBUTING.md` ("Modules with a history document"), then WP-B7's brief
(`C:\Users\AndrewT\AppData\Local\Temp\claude\D--Metacept-Neurophos-Python-Test-Scripts-Metasurface-QWP\78f7e8ef-9607-4e41-ba1d-ebd1c2e74c7e\scratchpad\briefs\WP-B7_ASYMPTOTIC_FGA_GBD.md`),
its report `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-B7_REPORT.md`, its changelog, its test file
`tests/unit/test_audit2609_b7_asymptotic.py`, and `fixes/VERIFY_WP-B1.md` sections 4, 6 and 7 (the S6 fallback measurements and the JAX
thin-screen finding WP-B7 items 9-11 answer) plus `fixes/WP-A4_REPORT.md` section 6 (the Y4 / Y5 / FGA designs).

## What to do, in order

1. **Re-derive every headline on fixtures the engineer did not use** with oracles of your own: an exact ray trace of the input's own rays plus a
   Kirchhoff / Rayleigh-Sommerfeld sum from the exit surface (the method of VERIFY-B1's oracle; write your own), a brute-force fold-caustic
   integral for the uniform asymptotics (a direct RS quadrature through the caustic, not the library's), and the exact `phase_screen` /
   `'quadrature'` integrators as controls.  Different singlet, glass, wavelength, NA and pitch from every fixture in the report.
2. **Attack the routing and the defaults.**  Whatever WP-B7 decided about FGA vs `phase_screen` at NA 0.145 and about `integration_method='uniform'`:
   is any default moved, and if so does the report carry the measured case and the changelog a Migration note?  Everything claimed
   byte-identical must be proved archive-to-archive (`git archive f64444ec^ lumenairy` vs `git archive f64444ec lumenairy`, child processes, cwd +
   PYTHONPATH = the archive, `lumenairy.__file__` asserted; never through pytest, never against the shared working tree).
3. **Attack the three Y4 refactors and the Y5 structural change** (bit-identity claims): repeat on your fixtures, including complex64, the `xp`
   twin where one exists, a decentred input and the fold_split leg.
4. **Attack items 9-11.**  (9) The JAX Maslov screen: measure the chief-ray displacement error on your own tilted inputs before and after; if a
   correction shipped, is the collimated case byte-identical and does the corrected screen land within the diffraction spot of the exact trace?
   If only a warning / envelope shipped, is the envelope derived and does the warning fire where the error exceeds it?  (10) The S6 fallback
   statistic: reproduce VERIFY-B1's ladder (speckle 0.002-0.6 rad on a tilted carrier against the exact `'quadrature'`) on your fixture; does the
   new bar refuse where the answer is gone and admit where it is not, two-sided, on BOTH fixtures?  Does every case that engaged before still
   engage?  (11) The pupil-chart sizing: on a tilt of 2x the lens NA is `na_proxy` now the mean direction plus the spread, is the collimated case
   byte-identical, and did the runtime on tilted calls fall as claimed?
5. **Attack S9 (GBD kernel clipping)**: the clip must be bit-identical where the kernel is already inside the bound and must not change any
   GBD fixture's answer beyond a derived floor; check the windowed / unwindowed pair.
6. **Attack the pins.**  Derived envelopes or per-build numbers?  Fail-before real on the parent?  Mutate in memory (drop the displacement
   correction, revert the statistic, revert the chart sizing, unclip the kernel) and confirm the right tests go red.  The two knife-edge
   `ModalAsymptoticStillBitEqual` arms carry a 3e-8 bar and `test_w6_a2_v2_star` a 1e-15 floor: measure their headroom after WP-B7.
7. **Fix what you find, in the owned files**; re-record every history document you touch.  Anything outside goes under "Requested changes
   outside my ownership" with the exact edit.  Do not move a default the report did not move.
8. **Report** `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/VERIFY_WP-B7.md`: a verdict table (each report claim -> VERIFIED /
   VERIFIED-WITH-NOTES / NOT FIXED / REGRESSION, with your oracle and your numbers), defects found and fixed (with fail-before), a "Follow-up"
   section, and the exact commands + counts + durations of every test run.  If you changed library code, add `### Fixed -- ...` release text to
   a new `VERIFY_WP-B7_CHANGELOG.md`.

## Rules

* Every python run with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`; one process at a time (the box is shared).
* NO git write commands of any kind; read-only `git show` / `git archive` / `git log` are fine.  Do not kill processes.  The orchestrator commits
  with an explicit file list.
* Own only the files named above, their `docs/history/` documents, `tests/unit/test_audit2609_b7_asymptotic.py` (add, do not weaken), your two
  report files.
* Comments say what the code does now and why; never a change log.  Finish with the report's full text as your final message.
