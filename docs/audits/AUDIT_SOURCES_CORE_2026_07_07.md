# sources/core.py Audit — 2026-07-07

Scope: full line-by-line read of `lumenairy/sources/core.py` (3,040 lines, all
16 factories + `PartialCoherenceMCF` + the `Source` dataclass and its 7
classmethod factories).  Chosen because this module recorded **zero findings**
in the 90-agent 2026-07-01 deep audit despite being 3k physics-bearing lines
that feed every downstream simulation — the thinnest coverage per LOC in the
package.  Read-only pass; single-context inline audit (same method as the
v5.21 delta audit).

Also recorded here: remediation spot-check of the 2026-07-01 deep audit —
**all 8 P1 findings are fixed and changelog-tagged** in v5.18–v5.21 (P1-01
BOR flux threshold, P1-02 bruggeman quadratic, P1-03 PMM incidence guards,
P1-04 stale `_internal`, P1-05 staggered guards, P1-06 LG/HG cache origin,
P1-07 fixed-index glass staleness, P1-08 GUI dead import), and nearly all 42
P2s are referenced by CHANGELOG remediation entries.

---

## 1. Overall assessment

The module is in **excellent shape** — visibly hardened by many prior audit
rounds (centralised grid validation P1-NEW-10, bool-N rejection P2-VAL-1,
peak-off-grid normalisation F-38, point-source r-floor H-PR-4, evanescent
tilt guard F-40, Bessel cone-angle window P1-NEW-9, the Schell P0-NEW-2
redesign with the P3-10 deterministic-normalisation fix, and disciplined
deprecation shims throughout).  Independently verified this pass:

* Hermite (`H_k = 2xH_{k-1} − 2(k−1)H_{k-2}`) and generalized-Laguerre
  (`kL_k = (2k−1+α−x)L_{k-1} − (k−1+α)L_{k-2}`) recurrences — correct.
* HG/LG mode forms and the `w0`/`sigma` conventions (`w0 = σ√2`, 1/e²
  intensity radius) — correct and consistently documented.
* Gaussian-Schell realisation recipe: `|H(k)|² = exp(−|k|²σ_g²/2)` inverse-FTs
  to the `exp(−|Δr|²/(2σ_g²))` Schell kernel — correct; the v5.4.6
  deterministic Parseval normalisation is the right fix for the empirical-MCF
  bias.
* Point-source sign convention (`z0<0` diverging `e^{+ikr}/r`) is consistent
  with the library `exp(−iωt)` convention; tilted-plane-wave phase sign
  matches forward `exp(+ikz)`.
* `PartialCoherenceMCF.from_ensemble`'s SVD route (never materialising dense
  J for large grids) is the right construction.

Two real findings and a set of nits follow.

---

## 2. Findings

### SRC-1 (P3) — `PartialCoherenceMCF` dense storage is the conjugate of the documented MCF, and disagrees with the modal storage
The class documents `J(r1, r2) = <E(r1) · conj(E(r2))>`.  Under that
convention the coherent modes are the **unconjugated** rows of `Vh` — exactly
what the modal branch stores — and the modal `coherence_at`
(`Σ λ_k φ_k(r1) conj(φ_k(r2))`) evaluates the documented `J(r1, r2)`
correctly.  The **dense** branch, however, builds
`J_full = E_mat.conj().T @ E_mat / nr`, whose `[i, j]` entry is
`<E*(r_i) E(r_j)>` = `conj(J_doc(r_i, r_j))`.  Consequences (complex-J
ensembles only):

* `coherence_at` returns the **conjugated** coherence phase iff the grid is
  small enough for dense storage (`Ny·Nx ≤ max_full_N²`, default 64²) — the
  same query flips phase sign with grid size, since storage form is
  auto-selected.
* `coherent_modes()` on dense storage returns the conjugates of the true
  modes (eigh of `conj(J_doc)`), inconsistent with the modal branch.
* The `from_ensemble` code comment ("the eigenmodes of J are
  `Vh[k, :].conj()`") is wrong for the documented convention — the code
  correctly stores `Vh[k, :]` unconjugated; only the comment misleads.

Invisible for the three in-module factories (the Gaussian-Schell and
incoherent-annular kernels are real-valued in expectation), which is why no
test catches it; a user-built ensemble with complex MCF (twisted / tilted
partial coherence) hits it.  **In-codebase precedent (found 2026-07-07,
tranche 4 of the analysis audit)**: `analysis/coherence.mutual_coherence`
had the *identical* bug and was fixed in 4.10 — its fix comment describes
exactly this failure mode ("`rows.T.conj() @ rows` ... produces the complex
conjugate of the documented quantity ... any phase-sensitive consumer saw
the off-diagonals with flipped sign").  The dense `from_ensemble` branch is
that pre-4.10 pattern, unfixed.  **Fix (one line)**:
`J_full = (E_mat.T @ E_mat.conj()) / float(nr)` — then the dense branch, the
modal branch, `coherent_modes`, and the docstring all agree.  Fix the
`from_ensemble` comment at the same time, and pin with a complex-J oracle
(e.g. a linearly-phase-tilted ensemble, whose `J(r1, r2)` phase sign is
known) run at both storage sizes.

### SRC-2 (P3) — scale-parameter validation is inconsistent across factories
Prior hardening rounds added positive/finite guards to `sigma`
(`create_gaussian_beam`, F-39), MFD (`create_fiber_mode`), LED
diameter/divergence, Bessel `cone_angle`, and the whole Schell family — but
four factories were missed:

* `create_hermite_gauss` / `create_laguerre_gauss`: `w0` unvalidated —
  `w0 = 0`/NaN silently yields a NaN-laced field; `w0 < 0` yields a
  sign-flipped mode.
* `create_top_hat_beam`: `diameter` unvalidated — `diameter <= 0` yields an
  all-zero field, and the `norm > 0` guard then silently skips normalisation
  (a silent zero source downstream).
* `create_annular_beam`: neither diameter validated, and **no
  `inner < outer` check** — an inverted annulus silently returns the all-zero
  field.  Notably its incoherent sibling `create_annular_incoherent_source`
  *does* validate exactly this (non-negative inner, positive finite outer,
  strict ordering); the coherent factory should copy those five lines.

### SRC-3 (P4) — nits
* `_schell_phase_realizations` takes an `N` parameter it never uses
  (`Ny`/`Nx` carry the shape) — dead parameter.
* `create_multi_field_sources` with an empty `field_angles` sequence returns
  `(sources=[], x=None, y=None)` silently — an empty-input raise (or
  documented None-axes contract) would be kinder.
* `Source.propagate`'s `out_dx = kwargs.get('output_dx', self.dx) or
  self.dx` treats a (nonsensical but conceivable) `output_dx=0` as "unset"
  via falsiness rather than raising — harmless today, worth a `is None`
  test if `output_dx` ever grows meaning.
  **UPGRADED 2026-07-07 (propagators audit, finding DS-1, P3)**: the
  deeper problem is that `Source.propagate` wraps the dispatcher's RAW
  return without `return_result=True` — for the tuple-returning /
  pitch-changing kernels (`fresnel`, `fraunhofer`, `sas`, including
  `method='auto'` far-field selections) the new Source carries a 3-tuple
  as `E` and the stale input pitch as `dx`.  See
  `AUDIT_PROPAGATORS_KERNELS_2026_07_07.md` §DS-1 for the fix.

---

## 3. Coverage statement

Every line of `sources/core.py` was read this pass.  Not audited here:
`sources/__init__.py` (re-exports), and the downstream consumers of the
ensemble/MCF contracts (`analysis/` partial-coherence integrators,
`propagators/dispatch`) — those belong to the analysis/propagators
subsystems (covered structurally by the 2026-06-10 full-library audit and
the 2026-07-01 deep audit; a dedicated line-level pass on `analysis/` is the
natural next target, 16k LOC with only 5 deep-audit findings).

---

*Audit performed single-context against lumenairy v5.21, 2026-07-07.*
