# Full-library audit — 2026-06-10 (v5.14.1)

**Method + depth disclosure.** Solo audit (no agent fan-out): systematic
inventory, mechanical sweeps, targeted reads of every top-level module, and an
independent analytic probe battery for the physics-bearing areas OUTSIDE
RCWA/PMM (those two received dedicated deep audits this release cycle:
22-agent PMM audit + hand-verified 46-finding RCWA audit, both in this
folder's lineage). Depth is therefore: *deep* for RCWA/PMM (separate audits),
*probe + structural* for propagators/elements/analysis, *structural* for
raytrace/optimize/io/sources, *skim* for ui.

---

## 1. Inventory

130.4k LOC in `lumenairy/` (98.7k excluding `ui`), 92.3k LOC of unit tests
(195 files, 5053 tests green on the release tree), 602 public API names,
123 non-ui modules — **121 of 123 referenced by unit tests** (the two
exceptions are PMM internals exercised via their public wrappers).

| area | LOC | role |
|---|---|---|
| elements | 32.1k | PMM 11.0k, RCWA 7.2k, lens family 7.3k (5-tier fidelity ladder), polarization/DOE/coatings/BSDF/freeform/geometry |
| ui | 31.7k | PySide6 designer app; import-isolated, `gui` extra |
| propagators | 17.3k | ASM/SAS/RS/Fresnel/MFT/GBD/HFPI/Maslov-asymptotic family + dispatch + FFT infra |
| analysis | 16.0k | PSF/MTF, WFE, through-focus, AO, ghosts, phase retrieval, Zernike, Strehl, detector |
| raytrace | 8.6k | sequential trace + JAX twin, Seidel, world/3-D, ray fans |
| optimize | 6.5k | merit DSL, driver, multiconfig, multi-objective, JAX merits |
| io | 6.2k | Zemax/CodeV/Quadoa prescriptions, HDF5 storage, codegen |
| sources | 3.1k | Gaussian/HG/LG/Bessel/Schell-model/LED/multi-field |

Core deps: numpy/scipy/matplotlib/psutil only. CuPy/pyFFTW/JAX/PySide6 are
extras with version-pin hygiene (e.g. the documented pyfftw 0.15.1/py3.10
resolver cap). `ui` is not imported by any core module.

## 2. Physics-accuracy probe battery (independent, this audit)

All PASS:

| probe | result |
|---|---|
| Glass: N-BK7 @ d-line vs catalog 1.51680 | d = 1.1e-7 |
| Coating: bare-interface R vs Fresnel | exact |
| Coating: quarter-wave MgF2 AR vs analytic | 3.5e-18 |
| Coating: Brewster p-pol zero | exact |
| ASM: Gaussian energy / waist @ z_R | exact / 1e-5 rel |
| RS: Gaussian energy / waist @ z_R | 3.8e-14 / 1e-5 rel |
| Fresnel (scaled grid): energy / waist | 4.4e-16 / exact |
| Zernike: Gram orthonormality (256², 10 modes) | offdiag 2.7e-3 (discretization) |
| Zernike: analytic Z(2,0) | exact |
| Strehl: Maréchal | exact |
| Thin-grating ↔ RCWA cross-family (validity regime) | 2.6e-4 |

Bonus observation: the large-period probe geometry deliberately chosen to be
nasty tripped the RCWA energy guard LOUDLY (3.9e29 caught) — the
failure-containment layer works as designed on first contact.

## 3. Bugs found (this pass)

No new P1/P2 in the probed areas. Lower-priority items:

- **P3** `user_library.load_phase_mask` evaluates a stored expression with a
  sandboxed `eval` (no builtins, whitelist namespace, dunder rejection). The
  in-code comment already tracks the proper fix (small AST evaluator). Keep
  tracking; do not regress the sandbox.
- **P3** Zemax import skips `COORDBRK` (documented) — tilted/decentred
  prescriptions import as their on-axis skeleton. Loud documentation, but a
  warning *at load time* when COORDBRK rows are dropped would be safer than
  the docstring alone.
- **P3** `analysis.opd` sampling-check helpers default `verbose=True`
  (prints); most of the library defaults `False` (census: 9 True / 12 False).
  Standardize on `False` + route through `lumenairy._logging`.

## 4. Standardization / hygiene

Strong: ruff-clean, mypy whitelist in CI, zero bare `except:`, broad-`except`
budget test (re-armed this cycle after the module-skip repair), 4 TODOs
total, dedicated deprecation module (26 managed names), loud-failure
conventions (BACKGROUND export refusal, dispersive-solve refusal, energy
guards, named-culprit non-finite raises), per-release audit docs in-repo, and
meta-tests (V12–V17 walkers) that verify CHANGELOG claims against the tree.

Inconsistencies worth a cleanup pass (all cosmetic):
1. `verbose` default split (above) + `print` vs `_logging` (99 gated prints).
2. Five lens implementations are a *documented fidelity ladder*
   (thin → analytic real → traced hybrid → JAX → Maslov), not duplication —
   but the selection guidance lives in `dispatch.py` + docstrings; a single
   "which lens model do I want" doc table would help discoverability.
3. `sources/core.py` is 3k LOC in one file (the only remaining monolith;
   everything else was split in the v5.1/v5.5 reorganizations).

## 5. Performance posture

Already strong: pyFFTW plan-caching (~6× over pocketfft) with scipy-workers
and numpy fallbacks, CuPy paths (14 modules), JAX twins (33 modules,
positioned as the gradient path), transfer-function caches with locks,
prepare/sweep hoisting across solver families, `_with_blas_limit`
decorators, and this cycle's RCWA LEV-1 (×4-8) / LEV-2 levers.

Remaining opportunities (none urgent):
- `analysis.image_plane_wfe` / `through_focus` field-grid loops are
  embarrassingly parallel; a `workers=` option would be a cheap win for
  field-sweep-heavy users.
- Maslov polynomial-product accumulation has degree-bounded Python loops
  (vectorized inner axis) — micro at most.
- Roadmapped: RCWA even-parity scope (LEV-3), PMM 1-D homogeneous-eig share.

## 6. Feature gaps (cross-library view)

- **Mueller calculus**: Stokes parameters/DoP and the unpolarized
  Jones-pupil reduction exist; full Mueller-matrix element algebra
  (depolarizing elements) does not. Niche for current applications.
- **Tilted/decentred sequential systems**: raytrace `world` covers 3-D
  geometry, but the Zemax bridge drops COORDBRK and the wave-optics chain is
  axis-aligned; off-axis telescopes need manual setup.
- **Partial coherence**: Schell-model/LED sources + a thin coherence module
  exist; full mutual-coherence propagation (mode decomposition beyond GSM)
  is absent.
- Solver-family gaps are tracked in their own roadmaps (RCWA: LEV-3, fff_nv
  decision, µ/bianisotropy, hex lattices; PMM: native trapezoid metric
  [refuted as sketched], homog-eig share).

## 7. Bloat assessment

**Not bloated in the dead-code sense; deliberately broad.** Evidence:
- 121/123 modules test-referenced; the deprecation surface is actively
  retired (26 managed names, not accreting aliases).
- The apparent duplications dissolve on inspection: the lens family is a
  fidelity ladder; `elements/coronagraph.py` is a 47-line discoverability
  namespace; the `asymptotic_*` sextet is one coherent Maslov subsystem.
- The one genuinely debatable mass is `ui/` (24% of LOC) shipping inside the
  library wheel. It is import-isolated and pure-Python (~no wheel-size
  pain), but spinning it into a `lumenairy-designer` companion package would
  make the core's identity crisper. Low priority.
- Historical bloat was actively paid down (the v5.1/v5.5 monolith splits,
  the v4.13 broad-except sweep 99 → 20).

## 8. Verdicts

- **Overall quality: high — unusually so for a research-grade optics
  library.** The distinguishing trait is the *validation culture*: analytic
  oracles, cross-package oracles, energy/closure invariants, loud-failure
  guards, meta-tests on the CHANGELOG, and an audit paper-trail. The 0.94:1
  test:core-LOC ratio is the headline number.
- **RCWA/PMM: mature, with broad coverage** (detailed verdict in the answer
  accompanying this audit; 1-D surfaces exceed the common open FMM codes;
  2-D is parity-complete for axis-aligned geometry with the exotic corners
  explicitly roadmapped rather than silently absent).
- **Known weak spots**, honestly: BLAS/runner-sensitive degenerate-gauge
  tests need periodic re-pinning (inherent to the physics; a written
  tolerance policy would reduce churn); the test-infrastructure module-skip
  incident (fixed this cycle) showed the meta-layer needs the same paranoia
  as the physics; 2-D PMM is parity-tested but young (weeks).
