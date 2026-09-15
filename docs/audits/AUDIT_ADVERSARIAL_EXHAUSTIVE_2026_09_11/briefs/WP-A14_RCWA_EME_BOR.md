# WP-A14 — RCWA, EME and BOR (`elements/rcwa/`, `elements/eme/`, `elements/bor/`)

Read first: `COMMON.md`, then the partition report `RCWA-EME-BOR.md` (all of it) and report sections §13 (H1–H6),
§12 row G11 (`_sqrt_decay` — the file is yours), §15 in `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11.md`.
Repro: `repro/RCWA-EME-BOR/` (incl. the independent `oracle1d.py` and `tmm.py`), `repro/PMM-2D/q4_branch.py` (G11).
NOTE: the BOR and EME modules changed AFTER the audit revision (rounds 2/3 of the BOR multilayer guards landed on
2026-09-12; `git log --oneline a1ff1e6e..HEAD -- lumenairy/elements/bor lumenairy/elements/eme`) — re-locate every
citation by content and re-confirm each finding on the current HEAD before changing anything.

## Files you own
`lumenairy/elements/rcwa/*.py`, `lumenairy/elements/eme/*.py`, `lumenairy/elements/bor/*.py`. Tests: the
corresponding test files and new `tests/unit/test_audit2609_a14_*.py`. NOT `pyproject.toml` (H5's `threadpoolctl`
dependency edit is requested from the tests/CI work package — put the exact line in your report), NOT `pmm/`.

## Findings to implement
- **H1 (P1)** `guided_modes`' margin `5e-3·k0` per side is a fraction of k0, not of the guided window → empty list for
  every fiber with Δn ≲ 0.01. Scale the band to the window (e.g. `max(1e-6·k0, 1e-3·(qhi−qlo))`), raise/warn when the
  window is narrower than 2·margin instead of returning `[]`, add a weakly-guiding fixture (Δn = 0.005 and the V = 2.4
  textbook fiber) asserting the HE11 n_eff against the exact hybrid oracle (`fiber_oracle`) with a derived bar.
- **H2 (P1)** the Rayleigh-anomaly nudge `_grazing_safe_wavelength` (also `RCWAStack`): emit a `UserWarning` naming the
  requested and effective wavelengths whenever it fires; return the symmetric ±δ average (the continuous limit to
  O(δ²)) and expose `wl_eff` on the result; verify against `oracle1d.py` at the exact Λ = λ Moharam mount (TM R₀ →
  within the oracle's own convergence of 0.155846) and that a λ sweep through Λ = λ is monotone/continuous.
- **H3 (P2)** `rcwa_jones_2d(formulation='fff_nv')`: symmetrise `(L2·L1 + L1·L2)/2` (exactly symmetric for a
  transpose-symmetric cell) and mirror `_nv_nonseparable_guard` onto the Jones entry; test |Jxx − Jyy| on the square and
  disk cells (≤ 1e-13 after) and that the separable stripe result is unchanged within a derived tolerance.
- **H4 (P2, perf)** `scipy.linalg.eigh(A, M)` for the BOR symmetric-definite pencils (`radial_spectrum` and consumers);
  `_inplane_ops` single `_li_convolutions_2d` call for isotropic cells; a direct two-interface closed form for
  single-layer 1-D stacks (skips both general-star inverses); O(N²) Levinson for the two Toeplitz inverses if it is
  bit-tolerant and measurably faster; extend the even-parity fold to `li`/`fff_nv` via `_tensor_PQ` if the gate holds.
  Bit-identical or documented tolerance for each; measured with interleaved medians and `OPENBLAS_NUM_THREADS=1`.
- **H5 (P3)** report the exact `threadpoolctl` dependency line for `pyproject.toml`/`requirements.txt`; make the inert
  path's warning unmistakable.
- **H6 (P3)** restate the M8 "~5e-15" claim with its rasterisation scope; arm a tighter one-sided energy bar when the
  incidence medium is lossless; copy `Ex`/`Ey`/`kx`/`ky` in `per_order_amplitudes()` (or document read-only); the EME
  `isrealobj` dead branch and `ref_2d_modes`' `1/px` vs `conj(ph)`; consolidate the three branch-cut bands' SHAPE into one
  helper with per-caller scales if you can prove bit-identity; note the four forward-selector spellings.
- **G11 (P3, `rcwa/_core.py:1379–1428`)** `_sqrt_decay`'s predicate flips a near-zero EVANESCENT mode: use
  `& (r.imag**2 > r.real**2)` ("near the imaginary axis", not "near the origin"), state the real bound in the docstring;
  re-run `repro/PMM-2D/q4_branch.py` (counterexample must no longer flip; the θ-sweep continuity must hold; all RCWA/PMM
  branch-cut tests bit-identical or explained).

## Verification specifics
- Keep intact (re-check with the repro oracles): agreement with `oracle1d.py` ≤ 2e-13 over the 24 configurations, energy
  ≤ 1.4e-13, Li's metallic-TM convergence signature, ASR gains, TMM parity in amplitude AND phase 1e-15, the Jones
  conventions, conical covariance 8.6e-14, `_layer_eig_key` / `_HOMOG_CACHE` behaviour, EME slab TE0 second order and
  the Airy slab to 12 digits, BOR oracles, Bessel zeros, the 2026-07-13 cutoff-energy numbers, unit invariance, the
  flux-orthonormal S-matrix unitarity (add the superposition-closure gate the audit says nothing pins today).
