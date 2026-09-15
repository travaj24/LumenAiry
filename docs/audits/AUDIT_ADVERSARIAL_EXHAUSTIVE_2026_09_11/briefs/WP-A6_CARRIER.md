# WP-A6 — The traced-carrier chain (`propagators/carrier.py`, `propagators/carrier_field.py`)

Read first: `COMMON.md`, then the partition report `CARRIER.md` and report sections §2.4 (C1–C5), §14 (V6's remarks on
`carrier.py`) and §15 in `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11.md`. Repro: `repro/CARRIER/`.

## Files you own
`lumenairy/propagators/carrier.py`, `lumenairy/propagators/carrier_field.py`. Tests: the carrier test files and new
`tests/unit/test_audit2609_a6_*.py`. NOT the lens modules, NOT the other propagators.

## Findings to implement
- **C1 (P1)** the focus readout sizes its stop grid from the CARRIER, not the beam — 4–40× low peaks when the carrier is
  2–10 % off, silently: size from the beam (measured extent / NA) or from both with a margin, and add a two-sided
  diagnostic (fraction of energy inside the readout window) that refuses or warns.
- **C2 (P1)** `carrier_referenced_fit_radius` fits about the grid origin instead of the carrier centre — fix and test on
  an off-centre carrier.
- **C3 (P2)** tilted complex64 chains promote to complex128 — keep the caller's dtype (fold pistons mod 2π in float64
  first, as the ASM does) or document the promotion.
- **C4 (P2, perf)** `_radial_carrier_phase` separable form (872 → 61 ms at N = 2048, 3.5× less memory) — bit-identical or
  documented ULP tolerance.
- **C5 (P3)** as stated in the row; and the docstring/comment corrections CARRIER.md lists (38 % code — do not add
  history blocks; correct wrong statements).
- **New feature (§15.9, optional):** a Collins / ABCD-Fresnel transport with a Bluestein output grid for the carrier chain
  (exact for quadratic carriers with a freely chosen output pitch), removing the near-focus bridge machinery and the
  standoff heuristic that produces C1 — implement behind an option only if you can gate it against the existing chain on
  the CARRIER fixtures with a derived tolerance; otherwise write the design into your report as deferred.

## Verification specifics
- Keep intact (re-check) everything `CARRIER.md` lists under checked-and-found-correct; measure every fix on the repro
  fixtures before/after; every bar derived.
