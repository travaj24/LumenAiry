# Plan: Lumenairy changes from the metal-strip grating solver audit (2026-11-14)

**Input:** `docs/audits/AUDIT_METAL_STRIP_GRATING_SOLVERS_2026_11_14.md` (findings G-1 to G-9).
**Base:** 5.50.2, commit `32328f0e` (`origin/main`). Every line number below refers to that commit.
**Status of this plan:** every finding was re-checked independently against a clean 5.50.2
checkout before planning (section 1). The quantitative claims were re-run, not read. The
re-check scripts are listed in section 12.
**Who executes it:** four packages (P1-P4) built in parallel in separate worktrees. Each package
owns a disjoint set of files (section 3). The briefs in sections 4-7 are self-contained: an
engineer should not need to talk to another package's engineer, only to read the interface
contracts in section 3.2.

---

## 1. What the re-check found

Fixture A is the audit's fixture (section 2 of the audit: a resonant metal-strip
grating over a thin multi-layer spacer on a metal mirror, TM; its parameters are not
recorded in the repository).
(uncertainty about 1e-5, set by the audit's external FEM check). "Error" is |r - r_ref|.

| ID | Audit claim | Re-check verdict | What changes for the plan |
|---|---|---|---|
| G-1 | PMM TM converges as DOF^-2.2; geometric grading gives 58x at 0.6x DOF | **Confirmed to every digit** (defaults 1.43e-2; p deg 80 4.15e-4; Chebyshev deg 10 x 8 el 1.03e-3; geometric deg 8 x 10 el 1.03e-5, deg 8 x 12 el 1.53e-6, deg 10 x 12 el 9.1e-7). Oblique (theta 0.3) also works: geometric deg 8 vs deg 20 at 12 el agree to 1.4e-6, while Chebyshev deg 20 x 12 el is 9.2e-5 off. | **Four parts of the proposed fix are wrong or incomplete** (see the corrections below): the floor value clamps the audit's own recommended setting; the TE test cannot pass; the odd-n layout is asymmetric; the `bool(grade)` coercion at about 25 call sites would silently turn `'geometric'` into Chebyshev. |
| G-2 | Uniform layers pay a full coupled eig; fast path 1.8x, run collapse 22x, agreement 8e-13 | **Confirmed**, and extended to oblique incidence (the audit's prototype is normal-incidence only): theta 0.3, fixture A, both steps agree with the library to 4.4e-12 in J, 1.2e-12 in R. Fixture A52: 8.3e-13 / 7.6e-13 (steps 1 / 2) at normal incidence, 4.1e-12 at theta 0.3. A uniform-only Bragg stack with a metal film matches analytic TMM to every printed digit (1e-10) in TE and TM at theta 0 and 0.4, and the collapsed cascade matches the library to 4.8e-13. Speed: A52 library 463 s vs step 2 15 s, median of 3 interleaved on a heavily loaded box (about 30x; the audit's 22x was measured unloaded). | The prototype divides by the raw `lam`; it must use the same `|lam| < 1e-12` clamp the dense `V` uses. Four dispatch sites plus the prepared solver, not one. Four NumPy-vs-JAX parity pins at 1e-12 / 1e-13 will move (the JAX twin keeps the full eig). |
| G-3 | `RCWAStack.add_layer` defaults to Laurent; 0.26 error at N = 50; no warning | **Confirmed.** | The proposed `'auto'` is right for 1-D. Two audit statements are wrong: "a dielectric-only stack is unchanged bit-for-bit" (false: `'li'` changes TM on any patterned layer, dielectric included; only uniform-only stacks are unchanged) and "TE unchanged" holds to 1e-12, not bit-for-bit. On 2-D stacks the flip would also move every dielectric-pillar result (and the 2-D single-layer functions default to `'laurent'`); the plan scopes 2-D to metallic layers (decision D2). The downstream bridge relies on the Laurent default in three call sites (section 10). |
| G-4 | `'li'` on strips: N^-1.1..1.2, factor-3 scatter; ASR missing | **Rate and scatter confirmed** (spot re-run, section 12). | Docs ship now. ASR for the stack is DEFERRED behind a measured go/no-go spike (P4): the existing 1-D ASR has a documented ~1e-4 TM floor, non-monotone convergence and is normal-incidence only, and PMMStack with G-1 already gives 1e-6 in 2 s on this class. |
| G-5 | `rcwa_convergence` certifies a 23 % wrong R | **Confirmed** (spot re-run, section 12). | The proposed geometric step is right, but the audit's three-solve default (N, 1.5N, 2.25N) is unaffordable on 2-D stacks (2.25x per axis is ~130x the base solve and can exceed memory). The plan uses one extra solve by default with a conservative p = 1 error bound, and makes the three-point exponent opt-in. Its regression test at N = 600 is too slow for the unit suite; replaced. |
| G-6 | Point-sampled rasters: width/aliasing errors up to 0.05 (works as designed) | **Agree: works as designed.** | Ship `segments=` on RCWAStack (exact geometry the easy way). Reject the proposed raster warning (it fires on legitimate rasters). |
| G-7 | `PMMStack(n_orders=)` is the far-field order count, inert for accuracy | **Confirmed from code** (`_resolve_order_count`, `pmm/_core.py` 79-92; only `_farfield_order_set` reads it). | Docstring yes. **Warning rejected:** the alias is a deliberate, pinned cross-suite contract (`test_v5_12_0_naming_aliases`), the same reading holds across the whole PMM family, and the main downstream bridge passes `n_orders=` on every PMM solve under `filterwarnings = error`. |
| G-8 | `grade=True` inert at 1 and 2 elements; docstrings contradict | **Confirmed** (`pmm/oned.py` 133-136 vs 392-404 vs 475). | Docs only, folded into P1/P2. |
| G-9 | Phase reference plane of `jones_reflection` not stated | **Confirmed**; `RCWAResult.jones_reflection` / `jones_transmission` (`rcwa/stack.py` 868-872) have no docstring at all. | Docstrings on reflection AND transmission, with the transmission reference plane measured (section 12), plus a pinning test. |

**Corrections to the audit** (for whoever updates it):

1. **G-1 floor.** The proposed default `min_element = period * 1e-4` (1e-4 x period) clamps
   the audit's own recommended setting: ratio 0.17 with 12 elements puts the wall element at
   0.5 x 0.17^5 x W = 3.4e-5 x period. Clamped (effective ratio raised so the wall element is 1e-4 x period),
   degree 8 x 12 el gives 4.5e-6 instead of 1.5e-6, and regression test 1 ("within 3e-6") fails.
   The degradation below the floor is **round-off, not discretization**: it shows in the TE
   (spectral) channel, which a correct mesh change cannot move. Measured on fixture A, degree 8,
   12 elements, TE deviation from the converged TE value vs wall-element size:

   | ratio | wall element | TM error | TE deviation |
   |---|---|---|---|
   | 0.10 | 2.4e-6 P | 1.44e-4 | 3.4e-5 |
   | 0.12 | 6e-6 P | 1.62e-5 | 3.4e-6 |
   | 0.14 | 1.3e-5 P | 2.17e-6 | 1.0e-6 |
   | 0.17 | 3.4e-5 P | 1.53e-6 | 1.4e-8 |
   | 0.20 | 6.8e-5 P | 3.21e-6 | 3.0e-8 |
   | 0.25 | 1.8e-4 P | 1.23e-5 | 3.4e-9 |

   (At 3.4e-5 P the TE deviation is round-off noise of 1e-8 to 1e-7: 1.4e-8 at degree 8,
   1.2e-7 at degree 10, 4.8e-8 at degree 20. Below about 1.4e-5 P it grows steeply, roughly as
   h^-3.)

   The safe floor on this fixture lies between 1.3e-5 P and 3.4e-5 P, i.e. 1.3e-5 to 3.4e-5 of the
   period. The plan uses **3e-5 x period** (3e-5 P here: does not clamp the 12-element setting).
   This agrees with the library's own measured element-conditioning band (`PMMStack` docstring,
   `min_feature`: "corruption band ... roughly 1e-5 .. 1e-4 of a PERIOD").
2. **G-1 test 4** ("TE unchanged within 1e-9") cannot pass: at the recommended setting the TE
   value moves 1.4e-8 (degree 8), 1.2e-7 (degree 10) and 4.8e-8 (degree 20) from its converged
   value, by the round-off above. The test is restated in P1 with a measured bar, and P1 must
   measure whether the floor needs to grow with degree (4.5, test 8).
3. **G-1 odd element counts.** The prototype splits only the left central element, so the layout
   is not mirror-symmetric. The plan uses a symmetric layout (m geometric layers per side plus one
   central element). Measured, degree 8: 9 el 1.9e-5 (prototype 7.7e-5); 11 el 2.3e-6 (prototype
   1.0e-5).
4. **G-1 JAX.** "Refuse `'geometric'` as the JAX path refuses `elements_per_region > 1` today" is
   wrong for `PMMStack`: its JAX twin (`pmm/_jax_stack.py` 57-85, 280-281) builds its static
   topology through the same `_segment_elem_bnds` and accepts any `elements_per_region` / `grade`.
   Only the single-layer `pmm_jones_1d` / `pmm_efficiency_1d` JAX paths refuse
   `elements_per_region != 1` (`pmm/_core.py` 4051, 4576), where grading is moot.
5. **G-1 hidden coercion and duplicate kernel.** `PMMStack.__init__` stores `bool(grade)`
   (`pmm/stack.py` 1755) and about 20 sites in `pmm/oned.py` pass `grade=bool(grade)`;
   `PMM2DStack` does the same (`pmm/stack2d.py` 360, 609). A string `'geometric'` is truthy, so it
   would silently run Chebyshev. The grading rule also exists twice: `_graded_boundaries`
   (`pmm/_core.py` 458) and `_graded_fractions` (`pmm/_core.py` 3388, the JAX binary topology).
   One implementation, per the project's kernel-consolidation rule.
6. **G-1 scope.** `_graded_boundaries` is also the grid builder of the whole 2-D hybrid family
   (`pmm/twod.py` 412, `_build_axis`). The plan validates geometric grading on the 1-D family only
   and refuses it on the 2-D public entry points (it stays reachable internally through
   `PMMStack`'s conical path, which P2 validates).
7. **G-2** prototype divides by the raw `lam`. The dense path clamps `|lam| < 1e-12` to `1e-12`
   when it forms `V` (`pmm/_core.py` 2662-2667 and 2840-2842). The diagonal interface must use the
   same clamped value or it returns inf at a Rayleigh anomaly in a uniform layer.
8. **G-3 test 4** ("a dielectric-only stack is unchanged bit-for-bit") is false for any stack with
   a patterned layer. Restated in P3.
9. **G-5 test 2** (N = 600, about 2 x 60 s per solve) is too slow for the unit suite; replaced by
   affordable rungs in P3.
10. **G-3 and JAX (not in the audit).** `RCWAStack` with a traced (JAX) `eps_cell` and
    `formulation='li'` fails today: the forward value computes, but `jax.grad` raises
    `TracerArrayConversionError` (the concrete uniform-cell test `to_numpy(cell_c)` in the `'li'`
    branches, `rcwa/stack.py` 2889-2893 and 3023-3030). A blanket `'li'` default would therefore
    break every differentiable 1-D `RCWAStack` user who relies on the default. `'auto'` must
    resolve to `'laurent'` for a traced cell (P3, 6.1).

---

## 2. Verdicts and compatibility

| ID | Verdict | Compatibility approach |
|---|---|---|
| G-1 | **IMPLEMENT WITH CHANGES** (corrections 1-6) | New opt-in mode; defaults unchanged (`grade=True`, `elements_per_region=1`, which is inert anyway). Every existing result bit-identical. |
| G-2 | **IMPLEMENT WITH CHANGES** (correction 7; all dispatch sites; JAX parity bars re-derived) | Results move at round-off (measured <= 4.4e-12 in J). Byte-identity pins that cross the fast path are re-derived with dated measurements (accuracy over byte identity). `retain_internal=True` keeps the per-layer cascade. |
| G-3 | **IMPLEMENT WITH CHANGES**: default `formulation='auto'`; 1-D -> `'li'` for patterned isotropic layers; 2-D -> `'li'` only for metallic layers where `'li'` is supported (D2) | Behaviour change: minor release 5.51.0. One-keyword way back: `formulation='laurent'`. No warning on an explicit `'laurent'` (D1). The resolved value is recorded on the layer. |
| G-4 | **IMPLEMENT** docs; **DEFER** stack ASR behind the P4 spike | Docs only now. |
| G-5 | **IMPLEMENT WITH CHANGES** (one extra solve by default, p = 1 error bound, opt-in three-point exponent, relative tolerance) | Verdicts become stricter (intended). An explicit integer `bump=` keeps the additive step. Return shape unchanged; report gains keys. |
| G-6 | Classification **AGREE (works as designed)**; `segments=` **IMPLEMENT**; raster warning **REJECT** | New keyword only. |
| G-7 | Docstring **IMPLEMENT**; warning **REJECT** | No behaviour change. |
| G-8 | **IMPLEMENT** (docs) | None. |
| G-9 | **IMPLEMENT** (docs + pinning test) | None. |

---

## 3. Packages

### 3.1 Overview

| Package | Findings | Files owned (only these may be edited) | Depends on | Size |
|---|---|---|---|---|
| **P1 PMM grading core** | G-1 kernel, G-8 (single-layer docs) | `lumenairy/elements/pmm/_core.py`, `pmm/oned.py`, `pmm/twod.py`, `pmm/stack2d.py`, their `docs/history/lumenairy.elements.pmm.{_core,oned,twod,stack2d}.md`, new `tests/unit/test_pmm_geometric_grading.py` | none | medium, 2-3 days |
| **P2 PMMStack** | G-2, G-1 wiring, G-7, G-8 (stack docs), G-9 (PMM docs) | `lumenairy/elements/pmm/stack.py`, `pmm/_jax_stack.py`, new `pmm/_uniform_cascade.py`, `docs/history/lumenairy.elements.pmm.stack.md`, new `tests/unit/test_pmm_stack_uniform_fast_path.py`, new `tests/unit/test_pmm_stack_geometric.py`, new `tests/unit/test_pmm_stack_doc_contracts.py`, and the re-derived bars in the four existing test files named in 5.6 | P1 (interface 3.2, for the G-1 wiring only; G-2 starts at once) | medium-large, 4-5 days |
| **P3 RCWAStack** | G-3, G-5, G-6, G-4 (docs), G-9 (RCWA docs) | `lumenairy/elements/rcwa/stack.py`, `rcwa/_core.py`, `docs/history/lumenairy.elements.rcwa.stack.md`, new `tests/unit/test_rcwa_stack_auto_formulation.py`, new `tests/unit/test_rcwa_convergence_geometric.py`, new `tests/unit/test_rcwa_stack_segments.py`, new `tests/unit/test_rcwa_doc_contracts.py`, and existing tests whose pins move (census in 6.1) | none | medium-large, 4-5 days |
| **P4 RCWAStack ASR spike** | G-4 item 2 | scratch only until GO; on GO a new `rcwa/_asr_stack.py` plus a hook in `rcwa/stack.py` **after P3 has merged** | P3 (merged) | spike 2 days; build only on GO |

No two packages edit the same file. `CHANGELOG.md`, `pyproject.toml` and version strings are
edited only at integration (section 8). Each package hands over its CHANGELOG text in its PR
description.

### 3.2 Interface contracts between packages

**P1 -> P2 (grading spec).** P1 adds to `pmm/_core.py`:

```python
def _normalize_grade(grade, *, period, where, allow_geometric=True):
    """-> False | True | ('geometric', ratio: float, min_element_m: float)"""

def _graded_fractions(n_el, grade, width=None):
    """The one grading rule. ``grade`` is a normalized spec. ``width`` (metres) is required
    for a geometric spec with n_el >= 3 (the floor is absolute)."""

GEOMETRIC_DEFAULT_RATIO = 0.17
GEOMETRIC_DEFAULT_MIN_ELEMENT_FRAC = 3e-5      # x period
```

and a keyword `label=` on `_sem_modes_uniform` (default = today's text, output bit-identical).
`_graded_boundaries(a, b, n_el, grade)` keeps its signature and accepts the normalized spec.
P2 calls `_normalize_grade` in `PMMStack.__init__` and stores the result in `self.grade`
(a bool or a hashable tuple). Until P1 merges, P2 develops G-2 only.

**P3 and P4.** P4 does not touch `rcwa/stack.py` until P3 has merged.

---

## 4. P1: PMM grading core (G-1 kernel, G-8 single-layer docs)

**Goal.** A corner-adapted geometric element layout for the 1-D PMM family, opt-in, with a
measured round-off floor, one implementation of the grading rule, and no silent coercion of the
new mode. Every existing call bit-identical.

### 4.1 API

`grade` accepts, everywhere a 1-D PMM entry point takes it today:

- `False` / `True` (and `None`, `0`, `1`, `numpy.bool_` as today's `bool()` reading) - unchanged;
- `'chebyshev'` - alias of `True`;
- `'geometric'` - ratio 0.17, floor `3e-5 x period`;
- `('geometric', ratio)` and `('geometric', ratio, min_element_m)`.

No new keyword arguments (decision D3). Anything else raises `ValueError` naming the accepted
forms. `0 < ratio < 1` and `min_element_m > 0` are validated. `'geometric'` with
`elements_per_region < 3` raises `ValueError` ("at 1 or 2 elements every layout is the midpoint
split; use elements_per_region >= 3").

### 4.2 Algorithm (`_graded_fractions`, the single rule)

- `n_el <= 1` -> `[0, 1]`; `n_el == 2` -> `[0, 0.5, 1]` (all modes coincide).
- Chebyshev and uniform: today's formulas, unchanged.
- Geometric, `m = n_el // 2` layers per side, ratio `r`:
  - even `n_el = 2m`: `c = 0.5`; left boundaries `0, c r^(m-1), ..., c r, c`;
  - odd `n_el = 2m + 1`: `c = r / (1 + r)`; left boundaries `0, c r^(m-1), ..., c r, c`; one
    central element `[c, 1 - c]` (its size `(1 - r)/(1 + r)` continues the geometric sequence);
  - right half = `1 - left[::-1]` (exact mirror symmetry; the hybrid family lays elements out in
    the mirrored frame `x' = P - x`, `pmm/twod.py` 396-399, so a symmetric layout is invariant
    there);
  - **floor**: the wall element is `smallest(r) * width`, with `smallest(r) = c(r) r^(m-1)`,
    monotone increasing in `r`. If it is below `min_element_m`, raise the ratio to the root of
    `smallest(r) * width = min_element_m` on `[r, 1)` (bisection to 1e-14). If even `r -> 1`
    cannot reach the floor (region narrower than about `n_el x min_element`), use the uniform
    layout. Element count never changes.
- `_graded_boundaries(a, b, n_el, grade)`: `grade is False` -> `np.linspace(a, b, n_el + 1)`
  (today's arithmetic, kept for bit identity); otherwise
  `a + (b - a) * _graded_fractions(n_el, grade, width=b - a)` (the Chebyshev arithmetic today is
  exactly this expression).
- `_jpmm_build_topology` keeps calling `_graded_fractions(n_el, grade)` (no width). A geometric
  spec with `n_el >= 3` and no width raises (unreachable today: that path refuses
  `elements_per_region != 1`).

### 4.3 Call sites

- `pmm/oned.py`: replace every `grade=bool(grade)` (about 20 sites; list them with
  `rg -n "bool\(grade\)" lumenairy/elements/pmm`) with
  `grade=_normalize_grade(grade, period=period, where="<function name>")`, and add the
  `elements_per_region >= 3` check for geometric.
- `pmm/twod.py` public functions with a `grade` parameter (1251, 1489, 1781, 1937, 1988) and
  `pmm/stack2d.py` (360, 609): `_normalize_grade(..., allow_geometric=False)`; a geometric
  request raises `NotImplementedError("geometric grading is validated on the 1-D PMM family
  (PMMStack, pmm_jones_1d*) only")`. `_build_axis` itself stays permissive (PMMStack's conical
  path reaches it; P2 validates that path).
- `_sem_modes_uniform(..., label="PMM modal solve (uniform half-space)")`: pass `label` to
  `_record_mode_cut`.
- Update `__all__` in `_core`.

### 4.4 Documentation (G-8, single-layer part)

In `pmm/oned.py`:
- line 133: replace "Raise with `grade=True` to resolve the wall-corner field" with: for a metal
  corner, raise `degree` with one element, or use `grade='geometric'` with 8-12 elements per
  region and degree 8-10;
- `grade` parameter docs (136, 409 and the other copies): Chebyshev grading is inert at 1 and 2
  elements; describe `'geometric'`, its floor and why the floor exists (round-off, visible in TE);
- delete the sentence at 475 ("`elements_per_region>1, grade=True` recovers the rate for
  metals");
- add the measured table (fixture A numbers from section 1 plus the single-layer numbers in 4.5)
  to the `elements_per_region` paragraph at 392-415, replacing the "hp would give ..." sentence
  with what now exists.

### 4.5 Tests (`tests/unit/test_pmm_geometric_grading.py`)

Pure-arithmetic tests may use bit identity (TOLERANCE_POLICY rule 1). Physics tests use measured
bars with a gap on both sides (TESTING_STANDARDS restatement 5); record the measurement and date
in a comment.

1. **Legacy layouts frozen.** A frozen copy of the 5.50.2 `_graded_boundaries` in the test file;
   for `n_el` 1..30, `grade` True / False, three intervals (including `[0, W]` and
   `[W, P]`): `np.array_equal` with the new function.
2. **Geometric layout.** For `n_el` 3..16, ratio 0.17, width W, no floor: `n_el + 1`
   strictly increasing values; `s + s[::-1] == 1` to 1e-15; wall element `c r^(m-1) width` to 1e-12
   relative; `n_el == 2` gives `[0, 0.5, 1]`.
3. **Floor.** Ratio 0.05, 16 elements, width W, floor 3e-5 P: every element >= 3e-5 P
   (1 - 1e-12) and the smallest equals it to 1e-9 relative; ratio 0.17, 12 elements, width W,
   floor 3e-5 P: unclamped (equal to test 2's layout).
4. **Normalizer.** Accepted forms round-trip; `'geometirc'`, `('geometric', 1.2)`,
   `('geometric', 0.17, -1)` raise `ValueError`; geometric with `elements_per_region` 1 or 2
   raises; `PMM2DStack(grade='geometric')` and the `pmm/twod.py` entry points raise
   `NotImplementedError`.
5. **Census (decision tests).** No `bool(grade)` / `bool(self.grade)` under
   `lumenairy/elements/pmm`; exactly one `def _graded_fractions`; the Chebyshev formula
   `np.cos(np.pi * i / n_el)` appears exactly once under `lumenairy/`.
6. **Physics, single layer** (`pmm_jones_1d_segments`, the fixture-A strip layer alone on
   the fixture's substrate, air above, 50 nm, TM; table S in section 12; reference = geometric
   degree 20 x 12 el):
   - geometric degree 8 x 12 el vs the reference: below 1e-6 (measured 1.0e-7; degree 14 is
     3.4e-7 off, so the floor of this comparison is about 3e-7 of round-off);
   - Chebyshev degree 10 x 8 el (equal DOF 160) is above 1e-5 off (measured 2.7e-5): a decision
     pair with at least 10x between them on each side;
   - theta = 0.3: degree 8 vs 20 at 12 el below 1e-6 (measured 1.4e-7);
   - TE (`J[1,1]`) vs single-element degree 60 (TE reference -0.7773521659 - 0.4625778298i;
     degree 80 and Chebyshev 10 x 8 agree with it to 5e-12): geometric degree 8 x 12 el below
     1e-6 (measured 2.6e-7; 8 x 10 el: 5e-10), AND `('geometric', 0.10)` at 12 el below 1e-6
     (the floor clamps it; measured unclamped it is 1.3e-5). The second assertion is the one that
     catches a broken floor.
   - `('geometric', 0.05)` at 16 el with the default floor: within 3e-7 of the reference
     (measured 1.1e-7, TE 3.7e-8) - the floor turns a bad ratio into a good layout.
7. **Default unchanged.** `pmm_jones_1d_segments(..., grade=True)` and `grade=False` results
   equal a 5.50.2 frozen value bit-for-bit on one fixture (same code path, same build: rule 2).
8. **Floor vs degree (measure, then pin the decision).** Before fixing the default floor, measure
   the TE deviation (vs single-element degree 60) for `'geometric'` at 12 el, degrees
   8 / 10 / 14 / 20, on the single-layer fixture and on a second geometry with period about equal
   to the wavelength (period equal to the wavelength, a wider strip, same materials and depth), on Windows/MKL
   and WSL/OpenBLAS. Decision rule: if every reading is below 1e-6, keep the floor at
   `3e-5 x period`; otherwise make the floor `3e-5 x period x (degree / 8)^2` (the smallest GLL
   node spacing scales as `h / degree^2`), re-measure, and pin it. The test then asserts TE
   deviation below 1e-6 at degrees 8 and 20 (readings so far: fixture A 1.4e-8 / 4.8e-8, single
   layer 2.6e-7 / 1.5e-7; below the floor: 1.0e-6 at 1.3e-5 P and 3.4e-6 at 6e-6 P on fixture A).

**Must stay bit-identical:** every call with `grade` True / False / omitted, every 2-D call.
**May move:** nothing in existing tests.
**History fingerprints:** re-record `_core.py`, `oned.py`, `twod.py`, `stack2d.py` with
`scripts/record_history_fingerprints.py <module> --reason "G-1 geometric grading spec"`.
**CHANGELOG text (hand over):** "Added: `grade='geometric'` (or `('geometric', ratio[, min_element_m])`)
on the 1-D PMM family - corner-adapted geometric element grading; measured 1.5e-6 at 192 nodal
unknowns where Chebyshev grading gives 1.0e-3 at 160. Changed: `grade` values other than
bool-like / 'chebyshev' / geometric specs now raise instead of being coerced by `bool()`."

---

## 5. P2: PMMStack (G-2, G-1 wiring, G-7, G-8 and G-9 PMM docs)

**Goal.** Uniform layers solved from the shared geometric eig, consecutive uniform layers
collapsed into one diagonal S-matrix, geometric grading reachable from `PMMStack`, and the
accuracy knobs documented.

### 5.1 G-2 step 1: uniform-layer dispatch

- New module `pmm/_uniform_cascade.py` with `_uniform_iso_eps(eps_u) -> complex | None`: returns
  the scalar when every union-cell tensor of the layer is exactly equal (`np.array_equal`) and
  equals `e * I3`, with `e` a concrete Python/NumPy complex; `None` for anything else (anisotropic,
  traced, callable, string material key).
- Dispatch at every site that today calls `_sem_modes_tensor` for a vertical in-plane layer:
  1. `PMMStack.solve`, shared all-vertical branch (`pmm/stack.py` 3180-3196);
  2. `PMMStack.solve`, MIXED general branch, vertical in-plane layers (3285-3293);
  3. `_solve_vertical_perlayer` (3364 ff.): each layer has its own window grid, so compute
     `_uniform_geo_eig` on that layer's `mats` (cached by digest);
  4. `solve_vs_wavelength` and `_solve_vs_wavelength_perlayer` per-layer loops (4396 ff., 4698 ff.);
  5. `PreparedPMMStack._layer_eig` (5157 ff.) when the resolved layer is uniform.
- Call `_sem_modes_uniform(mats, k0, kx0, eps, geo, label="PMM modal solve (uniform layer)")`.
  On the shared grid `mats` is `mats_sup` (only `S0` and `n_glob` are read), so the uniform
  layer's own eps-dependent `mats` is **not built** - except when `retain_internal=True`, because
  `internal_field` / `layer_absorption` read `layer_mats[i]`.
- Not changed: the covariant (slanted) path, the conical path, the JAX twin.

### 5.2 G-2 step 2: diagonal run collapse

On the shared all-vertical branch and the shared `solve_vs_wavelength` loop only, and only when
`retain_internal` is False. All uniform media on one grid share `W = blockdiag(w, w)`, so every
interface and propagation S-matrix between them is diagonal in that basis.

- Per uniform medium (eps, flipped `q`, `lam`): `safe = where(|lam| < 1e-12, 1e-12, lam)`
  (the clamp `_sem_modes_uniform` uses), `q2 = eps - mu`, and the diagonal of `V` relative to
  `W`: `D_x = -eps / safe[:n]` (E_x-type columns), `D_y = q2 / safe[n:]` (E_y-type columns).
- Interface a -> b (the library computes `a = Wb^-1 Wa`, `b = Vb^-1 Va`, `pmm/_core.py` 1972-1986):
  `a = 1`, `b = D_a / D_b` elementwise, `S = (-(1-b)/(1+b), 2/(1+b), 2b/(1+b), (1-b)/(1+b))`
  (the third block is today's `0.5 (apb - amb apb^-1 amb)` simplified).
- Propagation by thickness t: `X = exp(-lam k0 t)`; `(A11, A12 X, X A21, X A22 X)`.
- Redheffer star elementwise: `D = A12 / (1 - B11 A22)`, `F = B21 / (1 - A22 B11)`,
  `S = (A11 + D B11 A21, D B12, F A21, B22 + F A22 B12)`.
- A run is a maximal sequence of consecutive uniform media, including the superstrate at its
  start and/or the substrate at its end. Each run is converted to dense (`np.diag`) once and
  starred into the cascade with one `_redheffer_star`; patterned layers keep the dense path.
- Fallback: if any `1 + b` is zero or any factor non-finite, take the dense path for that run.
  (The dense path's `_guarded_inverse` is dormant by default; on a diagonal matrix the
  equilibrated screen is trivially passed, so no other guard is lost.)

### 5.3 G-1 wiring

- `PMMStack.__init__`: `self.grade = _normalize_grade(grade, period=self.period,
  where="PMMStack")`; geometric with `elements_per_region < 3` raises. `_resliced_clone` /
  `_min_feature_clone` pass `self.grade` through unchanged. Document that `PMMStack.grade` may now
  hold a tuple.
- The JAX twin picks the spec up through `_jstack_static` (no code change expected; test it).
- Conical path (`_solve_conical`, `_build_axis`): allowed; validated by test 5.7-8.

### 5.4 G-7 (docstring only)

In the `PMMStack` class docstring and the `far_field_orders` / `n_orders` parameter text: they set
how many Rayleigh orders are projected in the far field (auto-grown to cover the propagating
ones); they do not control accuracy. The modal truncation is `degree` x `elements_per_region`
(and the grading). A convergence study must step `degree` (or the element count), never
`n_orders`. No warning (decision D7).

### 5.5 G-8 and G-9 (docstrings)

- `PMMStack` class docstring, three lines: for metal corners raise `degree` with one element or
  use `grade='geometric'` with 8-12 elements per region at degree 8-10; Chebyshev grading only
  matters from 3 elements up; quote the fixture-A table.
- `PMMStack.solve` (Jones paragraph) and `jones_transmission`: "zeroth-order tangential lab
  `(E_x, E_y)` over incident, columns = incident `E_x` / `E_y`, public `exp(-i w t)`; the
  reflection phase is referenced at the top interface of the first layer, the transmission phase
  at the bottom interface of the last layer (section 12, table T); conjugate for `exp(+i w t)`.
  For reflection the `E_x` ratio equals the p-polarized amplitude ratio; for transmission it
  differs from it by the ratio of the media's `kz/n`."

### 5.6 Existing pins that move

The fast path changes the uniform-layer eigenvector gauge, so NumPy-vs-JAX agreement on stacks
with uniform layers drops from ~1e-15 to ~1e-12 (measured 7e-13 normal, 4.4e-12 at theta 0.3).
Re-derive (do not loosen blindly) these bars, measuring on Windows/MKL and WSL/OpenBLAS and
writing both readings and the date in the comment:
- `tests/unit/test_audit2609_a12_pmm1d.py` 245-246 (1e-12);
- `tests/unit/test_audit_w6_pmm_rcwa.py` 174-175 (1e-13) and 232-234 (1e-12);
- `tests/unit/test_audit_dynameta_consumer_api.py` 131 (1e-12).
The new bar is the cross-build envelope x 100, and must stay at least two decades below the
smallest physical signal each test guards. Do **not** carry the fast path into the JAX twin
(decision D10): with a traced angle the geometric operator is traced and its spectrum is
degenerate (+/-m orders), which needs the cluster rule; the full eig there is the safe choice.
Any other pin that moves in the full run: list it with old / new value in the PR.

### 5.7 Tests

`tests/unit/test_pmm_stack_uniform_fast_path.py`:
1. Fixture A and fixture A52 (52 graded oxide sub-layers, audit section 2), degree 20 x 3 el,
   theta 0 and 0.3: fast path vs the dense path (forced by a private switch
   `_UNIFORM_FAST_PATH = False` in `pmm/stack.py`, used only by tests) agree in J and per-order
   R/T to a bar derived from the measured 4.4e-12 (both builds).
2. Bragg stack (10 pairs n 1.45 / 2.1, quarter-wave at a design wavelength 4 % above the solve wavelength, plus a 37 nm metal film
   eps -20+1.5i), solved at 1250 nm, theta 0 and 0.4, TE and TM, vs analytic TMM: measured
   agreement in section 12, bar = measured x 100.
3. Uniform layer at a Rayleigh anomaly of its own medium (choose theta so one order has
   `kz = 0` inside a uniform layer): finite result, agrees with the dense path, no inf.
4. Uniaxial uniform layer (`exx != eyy`) and a traced / string-keyed layer stay on the full path
   (assert through a spy on `_sem_modes_tensor`).
5. `retain_internal=True`: `internal_field` and `layer_absorption` on fixture A agree with the
   dense path to the cross-path bar.
6. Speed (decision, not reading): on A52 the fast path is at least 5x faster than the dense path,
   interleaved in one process, median of 3, BLAS pinned to 2 threads.
7. Mode-cut census label: with the census armed, uniform interior layers report
   "PMM modal solve (uniform layer)".

`tests/unit/test_pmm_stack_geometric.py`:
8. Fixture A, `grade='geometric'`, degree 8 x 12 el: error vs degree 20 x 12 el geometric below
   3e-6 (measured: 1.5e-6 vs r_ref and 1.5e-6 vs the degree-20 value, which is itself 5.5e-8
   from r_ref; record both); at
   least 50x below Chebyshev degree 10 x 8 el (measured 1.03e-3 / 1.53e-6 = 670x).
9. theta = 0.3: degree 8 vs 20 at 12 el below 4e-6 (measured 1.4e-6).
10. TE of geometric degree 8 x 12 el vs single-element degree 60: below 1e-6 (measured 1.4e-8;
    degree 10 reads 1.2e-7).
11. JAX twin forward with `grade='geometric'` equals the NumPy solve to the re-derived parity bar.
12. Conical (phi = 0.3): geometric degree 12 vs 20 at 12 el agree to a measured bar; phi = 1e-9
    reproduces the planar result to the conical-planar tolerance already used in the suite.
13. Floor: `grade=('geometric', 0.05)` with 16 el gives an error within 3x of the 0.17 / 12-el
    value (measured 4.4e-6 with a 1e-4 x period floor; single-layer with the 3e-5 P default: 1.1e-7) -
    i.e. the floor turns a bad ratio into a working one.

`tests/unit/test_pmm_stack_doc_contracts.py`:
14. `n_orders` 3 / 21 / 61 give the same `J[0,0]` to 1e-12 relative (measured identical to 9
    digits) - pins the G-7 docstring statement.
15. Uniform 80 nm lossy film on the fixture's substrate at theta 0 and 0.3, TE and TM: J equals the
    closed form to 1e-9; adding 100 nm of air on top multiplies `J` by `exp(2 i k0 cos(theta) d)`
    to 1e-9; adding 100 nm of substrate-index material at the bottom multiplies the transmission
    Jones by `exp(i k0 kz_sub d)` (`kz_sub = sqrt(n_sub^2 - sin^2 theta)`) to 1e-9; adding the
    air layer on top multiplies the transmission Jones by `exp(i k0 cos(theta) d)` (table T).

**Must stay bit-identical:** stacks with no uniform layer (no code path changes for them);
`retain_internal=True` runs other than the uniform-layer modes; slanted / conical / JAX paths
other than the grading spec.
**May move:** any result on a stack with a uniform layer, at round-off.
**History fingerprint:** re-record `pmm/stack.py` with `--reason "G-2 uniform fast path + G-1 grading spec"`.
**CHANGELOG text:** "Performance: PMMStack solves laterally uniform layers from the shared
geometric eig and collapses runs of consecutive uniform layers into one diagonal S-matrix
(measured: section 12). Results change at round-off (<= 5e-12)."

---

## 6. P3: RCWAStack (G-3, G-5, G-6, G-4 docs, G-9 RCWA docs)

Order of work inside the package: G-3, then G-5, then G-6, then docs.

### 6.1 G-3: `formulation='auto'` default

- Default `formulation='auto'` on `add_layer` (1757), `add_graded_layer` (2045),
  `add_tapered_grating` (2138), `add_tapered_ridges` (2313), `add_tapered_pillars` (2386).
- New `_resolve_formulation(requested, *, kind, is_1d, cell=None, shapes=None, eps_background=None,
  period_y=None, dispersive=False, has_normal_pair=False)` in `rcwa/stack.py`, called before any
  existing validation. Explicit `'laurent'` / `'li'` / `'fff'` behave exactly as today. `'auto'`:
  - uniform or tensor layer -> `'laurent'` (those layers ignore it);
  - `eps_cell_normal` given -> `'li'` (the pair is meaningful only under `'li'`);
  - **1-D stack**: `eps_cell` -> `'li'`; `shapes` -> `'li'` when `_shapes_y_varying` finds no
    y-varying shape and the background is non-zero or dispersive, else `'laurent'` (the y-averaged
    path keeps its `RCWAYAverageWarning`);
  - **2-D stack** (decision D2, recommended): `'li'` when the layer is metallic
    (`Re(eps) < 0` for any concrete eps in the cell, the shapes or the background) and `'li'`
    supports it (`eps_cell`; `shapes` that pass `_check_li_shapes_2d`); otherwise `'laurent'`.
    Dispersive 2-D layers resolve to `'laurent'` (the docstring tells metal users to pass `'li'`);
    a per-wavelength choice would put a factorization switch inside a spectrum.
  - **traced (JAX) `eps_cell`** (any stack): `'laurent'`, because `'li'` does not trace today
    (correction 10). The docstring says so. Making `'li'` traceable (replace the concrete
    uniform-cell test with a static flag computed at `add_layer` time) is a follow-up, not part of
    this plan; when it lands, the traced case can resolve like the concrete one.
- Store the resolved value in `_RCWALayer.formulation` (what ran) and the request in a new slot
  `formulation_requested`. Update the `layers` accessor docstring.
- Builders resolve once per call (their `_raster_wants_companions` rejects `'auto'`); the
  `raster` default stays `'hard'` (decision D9).
- No warning on explicit `'laurent'` (decision D1).
- Docstring of `add_layer` `formulation`: the new default, the resolution table, the measured
  fixture-A ladder (audit G-3 table), and "pass `formulation='laurent'` to reproduce 5.50
  results".

**Pin census (do first).** Run the RCWA-related unit tests and `validation/` on the P3 base and
after the flip; list every assertion that moved with old / new value and why. Expected movers:
tests that build a patterned 1-D `RCWAStack` without `formulation` and compare TM numbers to a
frozen value or to another solver. Cross-solver comparisons should move closer; frozen values are
re-derived with a dated measurement. A test that compares two RCWA routes both left at the
default stays as is.

**Tests (`test_rcwa_stack_auto_formulation.py`).**
1. Fixture A, analytic strip, N = 100: the default call equals the explicit `'li'` call
   bit-for-bit (same code path); error 1.06e-2 (bar 1.2e-2), against Laurent 1.67e-1.
2. TE (`J[1,1]`) default vs explicit `'laurent'`: below 1e-11 (measured 4.5e-13 at N = 50).
3. Uniform-only stack: default bit-identical to explicit `'laurent'`.
4. Resolution table: one assertion per row above (1-D cell, 1-D y-invariant shapes, 1-D y-varying
   shapes -> laurent + warning, zero background -> laurent, 2-D dielectric cell -> laurent, 2-D
   metal cell -> li, 2-D metal disk shapes -> laurent, 2-D metal rectangles -> li, tensor -> laurent,
   `eps_cell_normal` -> li); `layers[i].formulation` and `formulation_requested`.
5. Builders: `add_tapered_grating(..., n_slices=1)` on fixture A's strip with the default equals
   the explicit `'li'` builder call bit-for-bit.
6. Even-parity fold: on fixture A, `symmetry='auto'` vs `False` under the new default agree to the
   existing fold tolerance, and the fold engages (record whether it does; if `'li'` layers bypass
   it, say so in the docstring as a cost).
7. Traced cell (skip-free: guard with the suite's existing jax-availability pattern): a 1-D
   `RCWAStack` with a `jnp` `eps_cell` left at the default resolves to `'laurent'`, and
   `jax.grad` of total R through it is finite and matches a central finite difference to the
   bar the existing RCWAStack JAX tests use. Re-check fixture: period 600 nm, 64-sample cell,
   40 % duty, eps -20+2i in air over n = 1.5, 150 nm, 1310 nm, N = 7 (forward R = 0.24588 under
   `'li'`; under `'li'` the gradient raises today).

### 6.2 G-5: `rcwa_convergence`

New signature (both the callable and the `RCWAStack` branch; one shared comparator in
`rcwa/_core.py`, used by both):

```python
rcwa_convergence(solver, *, order_params=("n_orders",), factor=None, bump=None,
                 rtol=1e-2, atol=None, estimate=False, warn=True, **kwargs)
```

- **Step.** `bump` given (int) -> today's additive step, unchanged. Otherwise
  `N_hi = max(N + 1, ceil(f N))` with `f = factor` or, by default, 1.5 for one order axis and 1.25
  per axis for two (2-D stack or two `order_params`), so the harmonic count grows about 1.5x in
  both cases.
- **Compared quantities.** Per-order efficiencies (R and T, both incident polarizations) aligned
  by order, as today, plus the zeroth-order complex amplitudes when available (stack:
  `jones_reflection()` / `jones_transmission()`; callable: a returned Jones element when the
  solver returns one).
- **Verdict.** Relative change per order `d_m = |R_m(hi) - R_m(lo)| / max(R_m(hi), 1e-3 sum R(hi))`
  (same for T), amplitude change `d_J = |J(hi) - J(lo)| / max(|J(hi)|, sqrt(1e-3 sum R(hi)))`.
  Error estimate with the conservative algebraic rate p = 1 (metal corners converge at p ~ 1.1):
  `err = max(d) / (f - 1)` (for an additive bump: `f = N_hi / N`). `converged = err <= rtol`, and
  additionally `delta <= atol` when `atol` is given (today's absolute test).
- **`estimate=True`.** A third solve at `f^2 N`; per quantity, local exponent
  `p = log(|s2 - s1| / |s3 - s2|) / log(f)`; Richardson error of the top solve
  `|s3 - s2| / (f^p - 1)`. Mark `erratic` (and not converged) when `(s3 - s2)/(s2 - s1) <= 0` or
  `p` is outside [0.5, 4]. Cost warning in the docstring (2-D: three solves at up to 1.56x and
  2.4x harmonics).
- **Report** keeps every current key and adds `delta_rel`, `delta_amp`, `err_estimate`, `factor`,
  and with `estimate=True`, `exponent` and `erratic`. Returns the highest-resolution solve, as now.
- Docstring: say why a small additive bump certifies wrong answers on algebraically converging,
  oscillating sequences (the fixture-A table), and what the estimate assumes.

**Tests (`test_rcwa_convergence_geometric.py`)**; rung values in section 12, table C:
1. Fixture A, `'li'`, N = 61: default reports `converged=False` (old default: True with R 23 %
   wrong). Measured: N 61 -> 92, relative change of R 7.2 %, p = 1 estimate
   14.5 %; true error at N = 92 is 15.5 %.
2. Fixture A, `'li'`, N = 90 (-> 135): estimate 14.7 % (true error at 135: 9.0 %), so
   `rtol=0.3` -> converged and `rtol=1e-2` -> not (decision pair, 2x and 15x from the estimate).
3. Smooth (sinusoidal-profile) dielectric grating, which converges exponentially: default reports
   converged at its working N (choose N where the change is below 1e-6).
4. Legacy: `bump=4` reproduces the old `n_orders_high` and `delta` bit-for-bit.
5. `estimate=True` on fixture A at N = 40 (rungs 40 / 60 / 90, successive changes of opposite sign): reports `erratic=True`; at N = 61 (rungs
   61 / 92 / 138) the exponent is 0.61 and the Richardson
   estimate about 19 % of R (true error at 138: 10.8 %) - not converged, not erratic. On the
   smooth grating: exponent in [0.5, 4] or the sequence already at round-off (assert the
   decision, not the reading).
6. 2-D stack: default step 1.25 per axis, truncation restored after the call.

Existing pins that move: `tests/unit/test_v5_9_0_rcwa_quickwins.py` 97-111 (default step changes
`n_orders_high` from 8 to 6; re-measure the N = 30 metal verdict under the new rule and re-pin or
pass `bump=4` to keep the legacy assertion - prefer adding a new-rule assertion next to a legacy
one). `test_v5_11_1_rcwa_lowerpri.py` passes `bump=` explicitly: unchanged.

### 6.3 G-6: `segments=` on `RCWAStack.add_layer`

- `add_layer(thickness, *, segments=[(width_fraction, eps), ...], ...)`, mutually exclusive with
  the other layer specifications. Validation as `pmm/_core.py` `_segment_walls` (fractions > 0,
  sum to 1 within 1e-6, walls normalized to land on 0 and the period) - re-implement the three
  checks locally (do not import from `pmm`).
- Layout exactly as `PMMStack.add_layer(segments=)` and `rcwa_jones_1d_segments`: `segments[0]`
  on `x in [0, w0)`, in PUBLIC coordinates.
- Mapping: one segment -> a uniform layer. Otherwise background = `segments[-1]` eps; every other
  segment `i` -> `dict(shape="rectangle", eps=eps_i, center=((x_i + x_{i+1})/2 * period_x,
  period_y/2), size=(w_i * period_x, period_y))`. Callables (dispersive eps) pass through.
  Confirm first that `_validate_shapes` accepts rectangles that share an edge (exact predicates);
  if it does not, that is a bug to fix in the validator for touching edges only.
- Formulation: the G-3 default applies (1-D -> `'li'`).
- Allowed on 2-D stacks (y-invariant lines).

**Tests (`test_rcwa_stack_segments.py`).**
1. Fixture A strip via `segments` equals the equivalent `shapes=` call bit-for-bit (same path).
2. Same at N = 100 with `'li'`: error 1.06e-2 vs r_ref (bar 1.2e-2) - matches the shapes table.
3. **Sign-sensitive layout check:** an asymmetric three-segment dielectric cell (e.g. widths
   0.2 / 0.3 / 0.5, eps 4 / 2.25 / 1) at theta = 0.3, period large enough for +/-1 orders:
   `RCWAStack(segments)` vs `PMMStack(segments)` per-order R(+1) and R(-1) agree to the measured
   cross-solver bar, and R(+1) - R(-1) differs from zero by far more than that bar (a reversed
   layout fails).
4. Validation errors: widths not summing to 1, a zero width, `segments` with `eps_cell`.

The proposed warning on hard rasters is rejected; instead the `eps_cell` docstring and the PIXEL
CELL CONTRACT block point to `segments=` / `shapes=`.

### 6.4 G-4 docs

- `add_layer` L-1 paragraph (1842-1863): on resonant metal-strip stacks consecutive N scatter by a
  factor 2-3 below N of about 100 (fixture-A table: N = 40...70), so step N geometrically (x1.5 or
  x2) and use `rcwa_convergence`'s default; for 1-D metal references use `PMMStack` with
  `grade='geometric'` (1.5e-6 in about 2 s on fixture A against 1.0e-3 at N = 800 for `'li'`).
- `rcwa_extrapolate` (`rcwa/_core.py` 3492 ff.): same scatter note; Richardson needs a smooth
  tail, which this class does not have below N of about 100.

### 6.5 G-9 docs

Docstrings for `RCWAResult.jones_reflection` and `jones_transmission` with the content of 5.5;
`tests/unit/test_rcwa_doc_contracts.py` with test 5.7-15 for RCWAStack.

**Must stay bit-identical:** explicit-formulation calls; uniform-only stacks; `bump=`-style
`rcwa_convergence` calls; every 2-D dielectric stack (under D2 as recommended).
**May move:** 1-D patterned TM results left at the default; 2-D metallic layers left at the
default; `rcwa_convergence` verdicts without `bump=`.
**History fingerprint:** re-record `rcwa/stack.py` (`rcwa/_core.py` has no history document).
**CHANGELOG text:** "Changed (default): `RCWAStack.add_layer` and its builders default to
`formulation='auto'` (1-D patterned isotropic layers -> 'li'; 2-D -> 'li' for metallic layers
where supported). Fixture A at N = 50: error 0.257 -> 0.039. Way back: `formulation='laurent'`.
Changed: `rcwa_convergence` steps the truncation geometrically and judges a relative error
estimate. Added: `RCWAStack.add_layer(segments=)`."

---

## 7. P4: ASR for RCWAStack 1-D patterned layers (DEFERRED; spike with a go/no-go gate)

**Why deferred, not rejected.** The existing ASR (`rcwa/oned.py` 179-280, 435-466) has a
documented ~1e-4 TM floor, non-monotone convergence and a bridge that becomes ill-conditioned with
N, and it is normal-incidence only. A stack version needs one coordinate stretch shared by every
layer, so uniform layers lose their analytic modes (the metric enters `Kx`), and it is exact only
when all patterned layers share wall positions. PMMStack with G-1 already delivers 1.5e-6 in about
2 s on this class.

**Spike (scratch only, 2 days).** Monkey-patch a stack solve that applies one Granet stretch
(walls of the patterned layers on coordinate lines) to every layer and to the half-space bridge
`G`. Measure on fixture A: error vs N for N = 50 / 100 / 150 / 200, at `asr_eta` 0.5 / 0.7, and the
condition number of `G`.

**GO** only if all hold: ASR `'li'` at N = 100 beats plain `'li'` at N = 400 (2.30e-3); the error
decreases monotonically over the four rungs at one eta; cond(G) < 1e8 at N = 200. Then build in
`rcwa/_asr_stack.py` with a hook in `rcwa/stack.py`, refusing stacks whose patterned layers do not
share walls and oblique incidence. **NO-GO:** record the measurements in this plan and close.

---

## 8. Integration order and release

1. **P1 and P3 merge first** (independent). P3 carries the largest silent error (G-3) and the
   false verdict (G-5).
2. **P2** rebases on P1 (G-1 wiring), then merges.
3. **P4** spike runs after P3 merges; its build, if GO, goes to a later patch release.
4. At each merge re-run `python scripts/record_history_fingerprints.py --check`; if two branches
   re-recorded the same module (they should not: ownership is disjoint), re-record with both
   reasons.
5. Integration writes `CHANGELOG.md` from the packages' texts, bumps `pyproject.toml` and
   `lumenairy.__version__` to **5.51.0**: a default changes (G-3), which the project ships as a
   minor (precedent 5.49.0's default flips); everything else would be patch-level. The deprecation
   horizon is fine (`NEXT_REMOVAL_VERSION = '5.52'` > 5.51.0); no removals are due.
6. CHANGELOG citations must use full paths (`lumenairy/elements/pmm/stack.py`, never bare
   `stack.py` or `_core.py`: both names exist under `pmm/` and `rcwa/`).
7. Tag only after the full un-masked main-CI matrix is green on the merge commit; tag the branch
   tip / main merge; `gh release create --verify-tag` (not a draft); confirm on
   `pypi.org/simple/lumenairy/`.

---

## 9. Test commands

Pin BLAS threads for every local run: `OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2`.
Assert `lumenairy.__file__` points into the worktree in every probe script (cwd / PYTHONPATH
shadowing has bitten before).

**Fast subset per package** (from the worktree root):

- P1: `python -m pytest tests/unit/test_pmm_geometric_grading.py tests/unit/test_audit2609_a12_pmm1d.py tests/unit/test_v5_11_0_pmm2d.py tests/unit/test_v5_14_0_pmm2d_cell.py -q -p no:cacheprovider` and `python -m pytest tests/unit -q -k "pmm and not slow" -p no:cacheprovider`
- P2: `python -m pytest tests/unit/test_pmm_stack_uniform_fast_path.py tests/unit/test_pmm_stack_geometric.py tests/unit/test_pmm_stack_doc_contracts.py tests/unit/test_audit_w6_pmm_rcwa.py tests/unit/test_audit_dynameta_consumer_api.py tests/unit/test_audit_dynameta_consumer_api_2.py tests/unit/test_audit_p1_stack_stale_internal.py tests/unit/test_fix_pmmstack_sliver_walls.py -q -p no:cacheprovider`, then every file matching `rg -l PMMStack tests/unit` (61 files).
- P3: `python -m pytest tests/unit/test_rcwa_stack_auto_formulation.py tests/unit/test_rcwa_convergence_geometric.py tests/unit/test_rcwa_stack_segments.py tests/unit/test_rcwa_doc_contracts.py tests/unit/test_rcwa_stack_shapes_li_1d.py tests/unit/test_rcwa_stack_shapes_li_2d.py tests/unit/test_v5_9_0_rcwa_quickwins.py tests/unit/test_v5_11_1_rcwa_lowerpri.py tests/unit/test_rcwa_point_mirror_fix.py -q -p no:cacheprovider`, then every file matching `rg -l RCWAStack tests/unit` (59 files).
- History pins: `python scripts/record_history_fingerprints.py --check` and `python -m pytest tests/unit/test_audit2609_a17_history_relocation.py -q`.

**Lint (before every push):** `wsl -e bash -lc "cd /mnt/d/<path to worktree> && ~/lumvenv/bin/ruff check lumenairy/ tests/"`.

**CI proxy for the numerical pins:** run every new and re-derived physics test in WSL
(`~/lumvenv/bin/python -m pytest <files> -q`) as well as on Windows/MKL; record both readings in
the bar comments.

**Full matrix before release:** `python -m pytest tests/unit -m "not integration" -q -n 2`,
`python validation/run_all.py --quiet`, then the PR CI matrix, then the main-merge CI matrix
(un-masked, every shard green). Pre-tag walkers:
`python -m pytest tests/unit/test_v5_*changelog* tests/unit/test_v5_*citation* -q`,
`python scripts/check_source_line_citations.py` (rc 0 or 2),
`python scripts/verify_changelog_closures.py` (rc 0 or 2).

**Flaky is bad math:** a test that fails intermittently is a hidden axis (build, ordering, state).
Never rerun to green; vary one axis and find it.

---

## 10. Downstream consumer impact (a downstream consumer)

Read from the bridge module of the pinned consumer. Its test suite runs with
`filterwarnings = ["error"]`, so **any new Lumenairy warning on a path it uses fails its tests.**

| Lumenairy change | Effect on the consumer | Consumer action |
|---|---|---|
| G-3 default flip | `rcwa_backend.py` `_add_shapes_layer` omits `formulation` when the bridge asks for Laurent (`kw = {} if ... == "laurent"`, line 242), and `LumenairyStackSolver` calls `add_layer(eps_cell=...)` / `add_layer(shapes=...)` with no `formulation` (lines 716-728). On 5.51.0 these silently become `'li'` while the bridge reports Laurent; its byte-identity pins move. | Always pass `formulation=` explicitly (all three sites); add a `FEATURE_FLOORS` entry for the `'auto'` default at 5.51.0. This also lands the consumer's own audit item O3 (its Laurent default) cleanly: it can default to `'auto'` and forward it on >= 5.51.0. |
| G-7 | Nothing (no warning added). Had the audit's warning shipped, every PMM bridge solve (`pmm_backend.py` 197-198 passes `n_orders=`) would fail the consumer's tests. | Optional: pass `far_field_orders=` instead of `n_orders=`. |
| G-1 | New opt-in. Its audit item O2 (expose `elements_per_region`) becomes worth doing: forward `grade` and `elements_per_region` through a `pmm_kwargs` dict and include them in `cache_fingerprint`. | Optional feature. |
| G-6 | `segments=` on RCWAStack gives the consumer's lamellar strip layers an exact-geometry path; its items O4 / N1 (raster edge quantization, coordinate-blind gridded fields) remain consumer-side. | Optional. |
| G-2 | Round-off changes on stacks with uniform layers (<= 5e-12). Consumer pins tighter than that on PMM results would move. | Re-run its `validation/lumenairy_*.py` oracles, raise `VERSION_VERIFIED_MAX` to (5, 51). |
| G-5 | The consumer does not call `rcwa_convergence`. | None. |
| G-9 | The documented reference planes are what the consumer already assumes (lab `E_x` / `E_y`, `exp(-i w t)`). | None. |

Its items O1, N2 (laterally structured gridded fields refused by the PMM bridge; no z-extent check
of a gridded field) need no Lumenairy change.

---

## 11. Open decisions for the owner (with recommendations)

| # | Decision | Options | Recommendation |
|---|---|---|---|
| D1 | How to ship the RCWAStack default change | (a) flip to `'auto'` in 5.51.0, no warning; (b) audit's two-step: warning in a patch, flip in the next minor; (c) warn on explicit `'laurent'` + metal + TM | **(a)**. The 5.49.0 precedent (default flips with a one-keyword way back), accuracy over byte identity, and a warning would break the main consumer's `error` filter while telling users something the flip fixes anyway. |
| D2 | Scope of `'auto'` on 2-D stacks | (a) `'li'` only for metallic layers where supported; (b) `'li'` for every supported isotropic patterned layer; (c) 2-D stays `'laurent'` | **(a)**. Metal is where Laurent is badly wrong (measured on the bridge audit's Ag pillar: Laurent 6.5e-2 off at N = 22 while `'li'` converges); (b) would move every 2-D dielectric-pillar result for a smaller, unmeasured gain (2-D `'li'` also costs per-row inverses), and the 2-D single-layer functions default to `'laurent'`. Revisit (b) with a measured 2-D dielectric study. |
| D3 | G-1 API | (a) `grade='geometric'` or `('geometric', ratio[, min_element_m])`; (b) new `grade_ratio=` / `grade_min_element=` keywords | **(a)**: no new keywords on about 15 entry points; the spec is hashable and flows unchanged through every cache and the JAX static topology. |
| D4 | G-1 floor default | (a) `3e-5 x period`; (b) the audit's `1e-4 x period` | **(a)**, measured (section 1, correction 1; table S). P1 test 8 re-checks it at four degrees and on a second geometry (period about equal to the wavelength) and applies a pre-agreed rule if it fails (floor grows as degree^2). |
| D5 | Change PMM defaults to geometric | (a) keep `grade=True`, `elements_per_region=1`, `degree=16`; (b) default to geometric 10-12 el | **(a)**: geometric needs ~160-200 nodal unknowns (about 1-2 s) where the default solves in 0.05 s; dielectric users would pay 20x. Document the recipe. |
| D6 | G-5 verdict semantics | (a) p = 1 error bound from one extra solve, opt-in three-point; (b) audit's three solves by default; (c) raw change vs rtol | **(a)**: honest about algebraic convergence, affordable on 2-D. Keep integer `bump=` supported without a deprecation (it has explicit users and costs nothing). |
| D7 | G-7 warning on `n_orders=` | (a) docstring only; (b) one-time warning | **(a)** (pinned cross-suite contract; consumer `error` filter). |
| D8 | Stack ASR | (a) defer behind the P4 spike; (b) build now; (c) reject | **(a)**. |
| D9 | Tapered builders' `raster` default once `'li'` is the default | (a) keep `'hard'`; (b) switch to `'harmonic'` when the resolved formulation is `'li'` | **(a)** for this release (one behaviour change at a time); revisit with the builder table. |
| D10 | NumPy-vs-JAX PMM parity after G-2 | (a) re-derive the four bars from measured envelopes; (b) carry the fast path into the JAX twin | **(a)**: (b) puts a degenerate geometric eig under a traced angle. |

---

## 12. Re-check evidence

Scripts (session scratch folder `assess_lumenairy\`, run against the clean 5.50.2 checkout,
`lumenairy.__file__` asserted, BLAS 2 threads, shared machine so times are indicative):
`common.py` (fixture + grading patch), `g1_check.py`, `g1_single.py`, `g2_check.py`,
`rc_check.py`.

**G-1, fixture A** (degree x elements, |r - r_ref|): defaults 16 x 1 1.43e-2; 80 x 1 4.15e-4;
Chebyshev 10 x 8 1.03e-3; geometric (prototype) 8 x 10 1.03e-5, 8 x 12 1.53e-6, 10 x 12 9.1e-7;
floor 1e-4 P: 8 x 12 4.5e-6, 10 x 12 2.5e-6; floor 1e-5 P: 8 x 12 unchanged; ratio 0.05 x 16 el
with 1e-4 P floor 4.4e-6. Odd counts (degree 8): prototype 9 / 11 / 13 el 7.7e-5 / 1.0e-5 / 1.0e-6;
symmetric 1.9e-5 / 2.3e-6 / 1.4e-5 (the 13-element symmetric layout's wall element is 9.7e-6 P,
below the floor; with the floor it clamps). TE reference (single element degree 60, 80 and
Chebyshev 10 x 8 agree to 5e-12): -0.8996752029 - 0.4078972484i.

**Table S (P1 single-layer fixture: the strip layer alone, 50 nm, on the fixture's substrate, TM;
symmetric geometric layout, ratio 0.17 unless stated).** r is `J[0,0]`, distances are to the
degree-20 x 12-el geometric value; TE distances are to the
single-element degree-60 TE value -0.7773521659 - 0.4625778298i.

| layout | DOF | TM distance | TE distance |
|---|---|---|---|
| geometric 14 x 12 | 336 | 3.4e-7 | 1.8e-7 |
| geometric 8 x 12 | 192 | 1.0e-7 | 2.6e-7 |
| geometric 8 x 10 | 160 | 4.1e-7 | 5e-10 |
| geometric 8 x 12, floor 3e-5 P | 192 | identical to 8 x 12 (not clamped) | - |
| geometric 8 x 16, ratio 0.05, floor 3e-5 P | 256 | 1.1e-7 | 3.7e-8 |
| geometric 8 x 12, ratio 0.10, no floor | 192 | 8.3e-6 | 1.3e-5 |
| Chebyshev 10 x 8 | 160 | 2.7e-5 | 5e-12 |
| single element, degree 80 | 160 | 1.1e-5 | 0 (reference family) |
| single element, degree 60 | 120 | 2.1e-5 | reference |
| theta 0.3: geometric 8 vs 20 at 12 el | - | 1.4e-7 | 1.6e-7 |

**G-2.** Fixture A / A52 agreement and timings as in section 1. Bragg stack (10 pairs
n 1.45 / 2.1 quarter-wave at a design wavelength 4 % above the solve wavelength plus 37 nm of eps -20+1.5i, period 400 nm, degree 8,
1250 nm): theta 0: TE and TM 0.8329663336 + 0.5517461635i, equal to the analytic TMM `E_x` ratio
to all 10 printed digits; theta 0.4: TE 0.9890062848 + 0.1432079848i, TM 0.9883001971 +
0.1435019282i, same; collapsed cascade vs library 4.8e-13 / 2.0e-13.

**Table C (G-5 rungs; fixture A, `'li'`, analytic strip; R_ref = |r_ref|^2).**

| N | err in r | R (not recorded) | relative R error |
|---|---|---|---|
| 40 | 1.85e-2 | - | 18.9 % |
| 60 | 1.80e-2 | - | 23.5 % |
| 61 | 1.66e-2 | - | 21.6 % |
| 90 | 1.20e-2 | - | 15.7 % |
| 92 | 1.23e-2 | - | 15.5 % |
| 135 | 6.80e-3 | - | 9.0 % |
| 138 | 8.18e-3 | - | 10.8 % |
| 207 | 4.66e-3 | - | 6.3 % |

Today's `rcwa_convergence` at N = 61: `converged=True`, delta 3.8e-4, R 23.5 % wrong; at N = 100:
`converged=True`, delta 5.1e-4, R 11.7 % wrong (audit reproduced). The p = 1 estimate from one
x1.5 step tracks the true error within 0.6x to 1.6x on every rung above.

**G-3 / G-4 spot checks.** `'li'` errors at N = 44 / 45 / 50 / 57 / 61 / 100: 4.54e-2 / 1.71e-2 /
3.92e-2 / 1.41e-2 / 1.66e-2 / 1.06e-2 (audit reproduced, including the scatter). Default
(Laurent) at N = 50: r = -0.197970 + 0.160811i, error 0.257; N = 100: 0.167; no warning; every
layer records `formulation == 'laurent'`. TE under the default and under `'li'` agree to all 12
printed digits at N = 50 and 100.

**JAX `'li'` trace check.** 1-D `RCWAStack`, period 600 nm, 64-sample `jnp` cell, 40 % duty,
eps -20+2i in air, n = 1.5 substrate, 150 nm, 1310 nm, N = 7, `formulation='li'`: forward
R = 0.2458819; `jax.grad` raises `TracerArrayConversionError` (correction 10).

**Table T (reference planes; 80 nm lossy film on the fixture's substrate, PMMStack degree 8
and RCWAStack N = 3 identical to 6 digits).** Both solvers reproduce the closed-form TE and TM
(`E_x` ratio) reflection at theta 0 and 0.3 (e.g. theta 0.3: TE -0.2675169 + 0.0068467i, TM
-0.2393256 + 0.0059054i). 100 nm of air on top multiplies r by exp(2 i k0 cos(theta) d)
(0.568065 + 0.822984i at theta 0) and t by exp(i k0 cos(theta) d) (0.885456 + 0.464723i);
100 nm of substrate material at the bottom multiplies t by exp(i k0 kz_sub d) (0.765250 +
0.643733i at theta 0; 0.774655 + 0.632384i at 0.3), in TE and TM. So: reflection phase at the top
interface of the first layer, transmission phase at the bottom interface of the last layer.

**G-7.** `n_orders` 3 / 21 / 61 on fixture A (degree 20 x 3 el): the same r
in all three (identical to 9 digits).

**Not checked:** the audit's external FEM reference (taken as stated; it sets r_ref's 1e-5
uncertainty and does not affect any decision here); the audit's 206-layer timing; the 2-D
`'li'`-vs-Laurent behaviour on dielectric pillars beyond the bridge audit's metal pillar (it is
why D2 recommends the metallic-only scope).
