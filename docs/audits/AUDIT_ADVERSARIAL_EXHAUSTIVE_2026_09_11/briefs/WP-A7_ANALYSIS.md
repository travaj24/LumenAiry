# WP-A7 — Analysis metrics (`lumenairy/analysis/`)

Read first: `COMMON.md`, then the partition report `ANALYSIS.md`, the `ORCHESTRATOR.md` rows on
`diffraction_limited_peak` and `wave_opd_2d`, and report section §8 (rows A1–A7) plus §15 in
`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11.md`. Repro: `repro/ANALYSIS/`, `repro/orch/verify_analysis.py`.

## Files you own
`lumenairy/analysis/*.py`. Tests: every analysis test file and new `tests/unit/test_audit2609_a7_*.py`.
`analysis/image_plane_wfe.py:508–511` reads `image_rays` at the last-surface sag (exit-vertex class): WP-A1 is adding
`TraceResult.at_exit_vertex(n_exit)` to `raytrace/` concurrently — check for it (`grep -rn at_exit_vertex
lumenairy/raytrace`) before touching that site; if it is not there yet, implement the signed transfer locally with a
`# TODO(audit-2609/A1)` marker for the orchestrator to swap.

## Findings to implement
- **A1 (P0 ✔)** `wave_opd_2d`'s row-then-column `np.unwrap` slips by whole waves on any pupil with ≥ 0.2 waves rms coma
  (0.27 rad/sample) — 53 % of the pupil wrong by a wave; feeds the optimizer merits and the GUI docks. Replace with a
  robust 2-D unwrap: a reliability-sorted (Herráez et al. 2002) or least-squares (Ghiglia–Romero DCT) unwrap
  implemented in NumPy/SciPy (no scikit-image here), anchored so that a known-defocus pupil returns the right
  integer-wave count. Oracle: the synthetic Zernike pupils in `repro/orch/verify_analysis.py` (0 slips required up to the
  Nyquist-limited gradient; document the gradient limit above which no unwrap can succeed and warn there).
- **A2 (P0 ✔)** `diffraction_limited_peak` builds its reference from a PARAXIAL quadratic phase, inflating every Strehl
  by 1/S_ref (1.06× f/10 … 36× f/4; a PERFECT f/2 lens reports 11.3). Use the exact converging-sphere reference on the
  same aperture (or the analytic Airy peak for a circular aperture where applicable); verify Strehl = 1.000 ± (derived
  bar) for perfect exact-sphere pupils at f/10, f/5, f/2.5, f/2 and unchanged for f/50.
- **A3 (P1)** Shack–Hartmann wavefront 2× too small — find the missing factor (lenslet geometry / integration), verify
  against a synthetic known wavefront.
- **A4 (P1)** `eval_image_plane_wfe` 62× high for distant objects (`object_distance` handling) — fix and verify against
  an independent paraxial/ray computation; also the exit-vertex site noted above.
- **A5, A6 (P2)** as stated in the rows (incl. the `H(z+Δz) = H(z)·H(Δz)` recurrence for through-focus scans — 22 % of a
  scan, 186 → 11 ms/plane — bit-identical or documented tolerance, with the accumulated-rounding bound derived).
- **A7 (P3)** `zernike_basis_matrix` cache keyed without X information (mid-point sample) and the remaining P3 items.

## Verification specifics
- The audit verified correct: Zernike tables/normalisation, PSF/MTF/OTF, M², detector flux, `wave_opd_1d` — keep them
  bit-identical where you do not intend a change, and re-run `repro/ANALYSIS/*.py` and `repro/orch/verify_analysis.py`
  before/after (expected after: peak/reference = 1.000 on the f/2.5 sphere; 0 integer slips on the 0.5-wave coma pupil).
- Every bar with a derivation comment (TESTING_STANDARDS).
