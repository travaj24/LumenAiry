# Metal-strip gratings on thin multilayers: PMMStack and RCWAStack audit (2026-11-14)

**Date:** 2026-11-14.
**Version audited:** Lumenairy 5.50.2, commit `32328f0e` (`origin/main`, "Release 5.50.2"). It was
imported from a clean detached checkout, with `lumenairy.__file__` and `__version__ == "5.50.2"`
asserted in every script. Line numbers below refer to this commit. The library code is unchanged
from 5.50.1 (`38fdd276`); 5.50.2 changed records only.
**Comparison build:** 5.50.0 (tag `v5.50.0`, and the editable checkout at `4c449fc1`, whose
`lumenairy/` package is identical to the tag). Every finding was re-run there. Where the two
behave differently, the finding says so.
**Scope:** the 1-D periodic solvers `PMMStack` (polynomial modal method, spectral elements in x)
and `RCWAStack` (Fourier modal method), applied to one problem class: a lossy metal strip grating
(a few tens of nm thick) on a stack of nm-thin dielectric and conducting-oxide layers over a metal
mirror, in TM (E across the strips), at normal incidence. The helpers on the same path were also
covered: `rcwa_convergence`, `rcwa_extrapolate`, `add_tapered_grating(raster=...)`, the energy
tripwires, and the reflection Jones conventions.
**Method:** each candidate defect was read in the source, then reproduced with a small standalone
script against an independent reference. Each one was tried in both directions: as hard to refute
(wrong usage, wrong reference, version effect) as to confirm. Fixes marked "prototyped" were
implemented in a scratch copy of the library and measured. None of them is in the repository.
Compute: 2 processes, BLAS pinned to 2 threads, on a shared Windows box. Wall times are therefore
indicative; every ratio quoted was measured interleaved in one process.

**Prior audit cross-reference.** `an earlier downstream-consumer audit (not in this repository)` (L-1 .. L-10) and
its PLAN already cover parts of this ground. Those findings are not repeated here; they are cited
where they touch a finding:
- L-1 (metal-corner algebraic convergence of RCWA `li` and PMM degree): documented in 5.50.1
  (`rcwa/stack.py` 1842-1863, `rcwa/_core.py` 3518-3537). Re-confirmed here on a different
  structure; G-4 adds what L-1 did not record.
- L-4 (`shapes=` layers were Laurent-only): **fixed in 5.50.1** (1-D shapes and 2-D rectangles
  accept `'li'`). On 5.50.0 the same call raises `ValueError: add_layer: formulation='li' applies
  to isotropic patterned layers (eps_cell) only`.
- L-6 (caller lattice / edge quantisation of `eps_cell`): documented in 5.50.1 (`rcwa/stack.py`
  309-347). G-6 re-measures it on this problem class.
- The 5.50.1 point-mirror fix: for a strip grating it does not change the zeroth-order reflection,
  because the mirror of a strip is a translation of it. Measured on the reference case of G-2:
  5.50.0 vs 5.50.2 differ by 4.4e-12 (PMM) and 1.3e-11 (RCWA), which is round-off.

---

## 1. Summary

| ID | Title | Class | Severity | Status |
|---|---|---|---|---|
| G-1 | PMMStack TM converges only as DOF^-2.2 on strip corners; no corner-adapted (geometric) grading | convergence / missing feature | high | CONFIRMED (rate documented); "elements_per_region is the only effective knob" NOT REPRODUCED; geometric grading prototyped: 58x smaller error at 0.6x the DOF |
| G-2 | PMMStack runs a full coupled eigensolve for every laterally uniform layer | performance | medium | CONFIRMED; two-step fix prototyped: 1.8x, then 25x on a 58-layer stack, agreement 8e-13 |
| G-3 | `RCWAStack.add_layer` defaults to `formulation='laurent'` with no warning for TM metal layers | usability (wrong default) | high | CONFIRMED (5.50.0 and 5.50.2) |
| G-4 | RCWAStack `'li'` on metal strips: N^-1.1..1.2 and a factor-3 N-to-N scatter | convergence | medium | rate CONFIRMED and as documented (L-1); "N^-1.5" NOT REPRODUCED; scatter CONFIRMED, undocumented |
| G-5 | `rcwa_convergence` reports `converged=True` with R 23% wrong | bug (false verdict) | high | CONFIRMED |
| G-6 | Point-sampled `eps_cell`: realised strip width and aliasing errors up to 0.05 in r | accuracy (documented contract) | low | WORKS AS DESIGNED (documented); the exact Li route ALREADY FIXED in 5.50.1 |
| G-7 | `PMMStack(n_orders=...)` is an alias of the far-field order count; it has no effect on accuracy | usability (API trap) | low | CONFIRMED |
| G-8 | `grade=True` is inert at `elements_per_region` 1 and 2; the knob docstrings contradict each other | documentation | low | CONFIRMED |
| G-9 | The phase reference plane of `jones_reflection` is not stated | documentation | low | CONFIRMED (the behaviour itself is correct) |

**Count by status:** 8 confirmed defects or gaps (G-1, G-2, G-3, G-4, G-5, G-7, G-8, G-9; G-1 and G-4
each carry one sub-claim that did NOT reproduce), 1 works-as-designed (G-6), 1 already fixed (the
analytic Li path of G-6, in 5.50.1).

**Checked and found correct** (no finding):
- Time convention: `exp(-i w t)`. Reflection = ratio of the tangential lab-basis `E_x`.
- Energy closure with a lossy semi-infinite metal half-space. `T` is the power entering the metal,
  and R + T = 1.000000000.
- Agreement of the even-parity fold (`symmetry='auto'`) with the full solve: 1e-12.
- 5.50.0 vs 5.50.2: no numerical change for this problem class.

---

## 2. Test fixture and reference value

**Fixture A** (every reproduction below uses it): a resonant metal-strip grating -- a 50 nm metal strip of |eps| ~ 1e2 across roughly half of a sub-wavelength period, a spacer of five thin dielectric films (one of them lossy, about 20 nm in all) and a thick metal mirror -- at normal incidence, TM (E along x, across the strips).  Its exact parameters are deliberately not recorded in the repository; the numbers below were measured on it and are kept as the record of this audit.  The library's tests and documentation carry a generic stack of the same character (`tests/unit/test_pmm_stack_geometric.py`), on which every number the library itself quotes was re-measured.
- **Graded variant (fixture A52):** the lossy film is split into 52 sub-layers, each with its own permittivity, the boundaries Chebyshev-clustered toward both interfaces.

The observable is the zeroth-order co-polarised reflection `r = jones_reflection[0, 0]`. It sits on a gap-plasmon resonance (a deep reflection dip), which is why it is sensitive.  The physical sensitivity to the strip width W is |dr/dW| = 0.050 per nm (PMM, W - 1 / W / W + 1 nm).

**Reference value.** `r_ref (value not recorded)`.
- **Source:** PMMStack with the geometric element grading prototyped in G-1 (degree 20, 12 elements
  per region, ratio 0.17).
- **Self-convergence:** the degree 10, 14 and 20 rungs at 12 elements agree with it to 9.2e-7 and
  4.9e-7.
- **External check:** an independent 2-D periodic finite-element solve (NGSolve, H1 order 3..6 on a
  corner-refined mesh, 4.3 M DOF, Robin ports; no Lumenairy code) gives a value at
  its finest rung, 1.3e-5 from r_ref. It is still moving by 6e-6 per order step toward r_ref;
  geometric extrapolation of its last three rungs lands about 5e-6 away.

**Honest uncertainty of r_ref: about 1e-5, set by the FEM check. PMM self-convergence alone gives
about 1e-6.** Every error quoted below is |r - r_ref|. Errors under about 1e-5 are not resolved
against the external check.

---

## 3. Findings

### G-1. PMMStack TM converges algebraically on the strip corners, and the library offers no corner-adapted grading

**What is wrong.** The TM field is singular at the strip corners. PMMStack's lateral basis has two
layouts. One is a single element per region, raised in degree (p-refinement). The other is
`elements_per_region` elements with Chebyshev-Lobatto clustering toward the walls
(`_graded_boundaries`, `pmm/_core.py` 458-468; it applies to the uniform layers too, because they
share the union grid). Both converge only algebraically in the number of nodal unknowns. Nothing in
the library adapts the element sizes to the singularity.

**Where.** `pmm/_core.py` 458-468 (`_graded_boundaries`); `pmm/_core.py` 2979-3008
(`_segment_elem_bnds`); the knob docs at `pmm/stack.py` 1643-1645 and 1700-1708, and `pmm/oned.py`
390-415.

**Reproduction (fixture A; DOF = n_glob = 2 x elements_per_region x degree):**

| layout | DOF | error | time [s] |
|---|---|---|---|
| defaults (degree 16, 1 el.) | 32 | 1.43e-2 | 0.05 |
| p: degree 40, 1 el. | 80 | 1.93e-3 | 0.26 |
| p: degree 80, 1 el. | 160 | 4.15e-4 | 1.36 |
| p: degree 160, 1 el. | 320 | 8.84e-5 | 6.96 |
| h: degree 10, 8 el., Chebyshev grading | 160 | 1.03e-3 | 1.26 |
| h: degree 10, 24 el., Chebyshev grading | 480 | 8.90e-5 | 18.9 |
| h: degree 20, 12 el., Chebyshev grading | 480 | 9.30e-5 | 18.3 |
| h: degree 10, 24 el., `grade=False` | 480 | 1.14e-3 | 18.1 |
| **prototype geometric, degree 8, 10 el.** | 160 | **1.03e-5** | 1.16 |
| **prototype geometric, degree 8, 12 el.** | 192 | **1.52e-6** | 1.94 |
| **prototype geometric, degree 10, 12 el.** | 240 | **9.2e-7** | 3.26 |

- **Rates:** p-refinement runs at DOF^-2.12 rising to -2.24. Chebyshev h-refinement runs at -2.2.
  Uniform h-refinement runs at -1.1. The geometric prototype's local slope is 5 to 10, and it keeps
  increasing over the measured range.
- **The candidate claim that `elements_per_region` is the only effective knob is NOT REPRODUCED.**
  At equal DOF, single-element p-refinement is 2.1-2.5x *more* accurate than Chebyshev
  h-refinement: 4.15e-4 vs 1.03e-3 at DOF 160, and 1.96e-4 vs 4.19e-4 at 224/240. This agrees with
  the library's own measured table in `pmm/oned.py` 399-404.
- **The algebraic rate itself is CONFIRMED, and it is documented** (`pmm/stack.py` 1700-1708; L-1).
- **Diagnostic: only the wall-adjacent element matters.** Two layouts whose wall-adjacent element
  is the same 1/4 of the region give the same r to 7e-12. They are Chebyshev with 3 elements
  (boundaries 0, .25, .75, 1) and uniform with 4 elements (0, .25, .5, .75, 1). Accuracy is
  controlled by how small the element touching each wall is, which is exactly what geometric
  grading targets.

**Why it matters.**
- **The default is far off.** On this structure `PMMStack()` at its default degree 16 is 1.4e-2 off
  in r, which is 15% of |r|.
- **Built-in layouts at a practical size:** 5e-5 needs about 400 DOF (about 12 s per solve).
- **The prototype** reaches 1.5e-6 in 1.9 s. Equal accuracy with the built-in layouts would need an
  estimated 1000+ DOF and over 100 s. Every sweep point pays this.

**Root cause.**
- **The size of the wall element.** Chebyshev clustering with `n` elements makes the wall element
  `(1 - cos(pi/n))/2`, about 2.5/n^2 of the region. That is algebraic refinement of the one element
  that resolves the singular field.
- **What the theory says.** hp-FEM results for corner singularities give exponential convergence
  in DOF for geometric meshes. Those are Babuska and Guo's geometric meshes, with a ratio of about
  0.15-0.2 toward the singular point.
- **This was known.** `pmm/oned.py` 409-411 already names hp grading as the remedy, but nothing
  implements it.

**Proposed fix.**
- **API.** Add a geometric grading mode next to the existing Chebyshev one:
  `grade=True | False | 'geometric'` with `grade_ratio=0.17` and a hard floor
  `min_element=period*1e-4`. Alternatively, a `grading=('geometric', ratio)` tuple.
- **The layout per region** `[a, b]` with `n` elements (half toward each wall):
  boundaries `a + (b-a) * {0, r^(m-1)/2, ..., r/2, 1/2, 1 - r/2, ..., 1}`, with `m = n // 2`.
  For odd `n`, split the central element. Clamp the smallest element at `min_element`.
- **Why the floor is needed.** At ratio 0.10 with 12 elements, degree 8, the wall element is 2.4e-6 P.
  The result then *degrades* silently, from 1.5e-6 to 1.4e-4, with R + T unchanged and no warning.
- **Prototype (measured above; a drop-in replacement of `_graded_boundaries` when the mode is on):**

  ```python
  def _geometric_boundaries(a, b, n_el, ratio=0.17):
      m = n_el // 2
      left = [0.0] + [0.5 * ratio ** j for j in range(m - 1, 0, -1)]
      s = np.array(left + [0.5] + [1.0 - v for v in left[::-1]])
      if n_el % 2:
          s = np.unique(np.concatenate([s, [0.5 * (s[m] + s[m - 1])]]))
      return a + (b - a) * s
  ```

- **Recommended default for metals.** Once it is gated: `grade='geometric'`, 8-12 elements per
  region, degree 8-10. Keep `True` as the default for byte-identity, and say in the class docstring
  which to use for metal corners.
- **A second step, if wanted.** True hp: degree rising linearly away from the wall. It needs
  per-element degree in `_build_sem_tensor_segments` (3068) and in the far-field projection.
- **Regression tests.**
  1. On fixture A, the geometric layout at degree 8 with 12 elements is within 3e-6 of the
     degree-20, 12-element geometric value. Its error is at least 50x smaller than Chebyshev
     grading at equal DOF.
  2. Geometric with 2 elements per region is identical to the current 2-element layout. Both
     reduce to the midpoint split.
  3. The floor clamps: ratio 0.05 with 16 elements produces no element below `min_element`.
  4. TE stays spectral and unchanged within 1e-9.
  5. JAX path: refuse `'geometric'` as it refuses `elements_per_region > 1` today, or carry it.

### G-2. PMMStack solves every laterally uniform layer with a full coupled eigensolve

**What is wrong.**
- **What happens today.** In the all-vertical cascade, every layer goes through
  `_sem_modes_tensor`: a dense eig of the `2n x 2n` coupled operator (`pmm/_core.py` 2641). That
  includes layers whose segments all carry one isotropic eps. The only saving is a memo that skips
  byte-identical layers (`pmm/stack.py` 3187-3195).
- **The fast path already exists.** The library has an exact one for uniform media:
  `_uniform_geo_eig` / `_sem_modes_uniform` (`pmm/_core.py` 2760-2852). One cached `n x n`
  geometric eig serves every uniform medium on the grid, with spectrum `q^2 = eps - mu`. But it is
  used only for the two half-spaces (`pmm/stack.py` 3166-3170).
- **The rest of the cost is also avoidable.** Every uniform layer still pays a dense interface
  S-matrix and two dense Redheffer stars. Yet all uniform layers on one grid share the same
  eigenvectors, so those S-matrices are diagonal in that basis.
- **The sibling solver already does this.** RCWAStack solves uniform layers analytically
  (`rcwa/stack.py` 3009-3018).

**Reproduction.** Fixture A52 (58 layers), degree 20, 3 elements per region (DOF 120).
Interleaved in one process, median of 4:

| path | fixture A (7 layers) | fixture A52 (58 layers) |
|---|---|---|
| library `PMMStack.solve` | 1.28 s | 10.29 s |
| prototype 1: uniform layers via `_sem_modes_uniform` | 0.85 s | 5.74 s (1.8x) |
| prototype 2: plus diagonal collapse of consecutive uniform layers | 0.33 s | 0.46 s (22x) |

- **Accuracy:** |r_proto - r_library| = 4.2e-12 (prototype 1) and 7.7e-13 (prototype 2); |dR| is
  below 2e-13.
- **Scaling with layer count:** with 206 layers, the library took 43.3 s and prototype 2 took
  0.74 s, agreeing to 8.9e-13. (That single run was on a more heavily loaded box than the table.)
- **Profile, unmodified library, fixture A52:**
  - 44 calls to `_sem_modes_tensor`, 2.17 s;
  - 58 Redheffer stars, 1.24 s;
  - 59 interface S-matrices, 1.06 s;
  - total 4.66 s.
- **The candidate report** of about 6 s per solve for this configuration is CONFIRMED: 4.5-10 s,
  depending on machine load.

**Why it matters.**
- **Graded stacks are the common case.** Depth profiles (graded carrier layers, thermal or field
  gradients) are routinely sliced into tens of sub-layers.
- **The cost is linear in the slice count and dominated by layers that need no eigensolve.**
  Parameter sweeps and fits multiply it.

**Root cause.** In the all-vertical branch, the layer loop calls `_sem_modes_tensor` without
testing for a uniform layer. The cascade loop treats all layers alike.

**Proposed fix (both steps prototyped and measured).**
1. **Use the uniform fast path for uniform layers.** In `PMMStack.solve`'s all-vertical branch
   (`pmm/stack.py` 3188-3195): when a layer's segments are all the same isotropic scalar, call
   `_sem_modes_uniform(layer_mats[i], k0, kx0, eps, _geo)`. `_geo` is already computed for the
   half-spaces, and the layer grid is the same union grid. The prototype tested the uniform
   condition on `mats['elem_bnds']`. In the library, test the segment list directly, and skip
   building that layer's eps-dependent `mats` altogether. Apply the same change to
   `_solve_vertical_perlayer` and to `solve_vs_wavelength`'s per-layer loop.
2. **Collapse runs of consecutive uniform layers into one diagonal S-matrix.** With the shared
   eigenvectors `W = blockdiag(w, w)` and `V = QW / lam`, `QW = [[0, w q^2], [-eps w, 0]]`, the
   interface between uniform media `a` and `b` has `a = I`. Its `b` is diagonal: `b_j =
   (eps_a/lam_a,j)/(eps_b/lam_b,j)` for the E_x-type columns, and
   `b_j = (q_a,j^2/lam_a,j)/(q_b,j^2/lam_b,j)` for the E_y-type columns. So
   `S = (-(1-b)/(1+b), 2/(1+b), (1+b - (1-b)^2/(1+b))/2, (1-b)/(1+b))` elementwise, and propagation
   is elementwise. Star the whole run elementwise. Convert it to dense once, and star it into the
   cascade with one `_redheffer_star`. The prototype also treats the substrate interface this way.
   Cost per uniform layer drops from `O(n^3)` to `O(n)`.
3. **Mind the per-layer data.** `retain_internal=True` needs per-layer partial cascades. Keep the
   dense path there, or retain the diagonal factors per layer.
- **Regression tests.**
  1. Fixture A52: both steps agree with the current path to 1e-10 in r and R/T.
  2. A Bragg stack (alternating uniform layers) agrees with the analytic TMM to 1e-12.
  3. Wall time on A52 is at least 5x below the current path (interleaved, in-process).
  4. An uniaxial uniform layer (`eps` a diagonal tensor with `exx != eyy`) stays on the full path.
     The fast path is isotropic-only.

### G-3. `RCWAStack.add_layer` defaults to Laurent's rule, silently, for TM metal layers

**What is wrong.**
- **The default.** `RCWAStack.add_layer(..., formulation='laurent')` (`rcwa/stack.py` 1757) and
  every builder that feeds it (`add_graded_layer` 2045, `add_tapered_grating` 2138,
  `add_tapered_ridges` 2313, `add_tapered_pillars` 2386) use the direct rule unless told otherwise.
  No warning is raised when a patterned layer contains a metal and TM is solved.
- **The single-layer sibling is the opposite.** `rcwa_efficiency_1d` defaults to
  `formulation='auto'`, which selects `'li'` for TM or any metallic index (`rcwa/oned.py` 376,
  421-434, 550-553). Its docstring calls an explicit Laurent + TM-metal solve "a factorization-study
  mode only".
- **The Li default is free.** Li's rule reduces exactly to Laurent for TE in 1-D: measured
  |r_yy(li) - r_yy(laurent)| = 4.5e-13 at N = 50. So defaulting to `'li'` costs nothing for TE.

**Reproduction (fixture A, analytic `shapes=` strip, `formulation` left at its default):**

```python
for N in (25, 50, 100, 200, 300, 400, 600, 800):
    print(N, abs(rcwa(N, formulation="laurent") - r_ref), abs(rcwa(N, formulation="li") - r_ref))
```

| N | Laurent error | Li error |
|---|---|---|
| 25 | 3.36e-1 | 3.28e-2 |
| 50 | 2.57e-1 | 3.92e-2 |
| 100 | 1.67e-1 | 1.06e-2 |
| 200 | 4.58e-2 | 4.88e-3 |
| 300 | 1.11e-2 | 3.12e-3 |
| 400 | **5.67e-2** | 2.30e-3 |
| 600 | 6.81e-3 | 1.43e-3 |
| 800 | 1.11e-2 | 1.00e-3 |

- **What the default actually returns.** At N = 50 it returns r = -0.198 + 0.161i, against r_ref =
  0.057 + 0.131i. The sign of Re(r) is wrong and |r|^2 is 3.2x too large.
- **The Laurent error is not monotone.** It drops to 1.1e-2 at N = 300 and jumps back to 5.7e-2 at
  N = 400.
- **No warning is emitted** (checked with `warnings.simplefilter('always')`). The stored layer
  reports `formulation == 'laurent'`.
- **The point-sampled `eps_cell` path** behaves the same way (2.70e-1 at N = 50).
- **Identical in 5.50.0.**

**Why it matters.** Errors of order 0.1-0.3 in r at the truncations people normally run. Even at
N = 800 (75 s per solve), the error is still 1e-2. An S-parameter-level error this size changes
design conclusions, and nothing flags it.

**Root cause.** The stack builder kept the historical default when `'li'` was added (v5.14.1,
audit GAP3), while the single-layer entry moved to `'auto'`.

**Proposed fix.**
- **Make the default `formulation='auto'`** on `add_layer` and the builders. For an isotropic
  patterned layer (`eps_cell` or `shapes`) on a 1-D stack, it resolves to `'li'`. On a 2-D stack it
  resolves to `'li'` for `eps_cell` and for rectangle-only `shapes`, and to `'laurent'` otherwise.
  Uniform and tensor layers keep their own factorization. Record the resolved value on the layer
  (`layers[i].formulation`), so callers can see what ran.
- **Why this is safe.** TE results are unchanged to round-off (verified above). TM results change
  toward the converged value.
- **Migration.** If a silent default change is too strong for a patch release, ship it in two steps.
  First, a one-time `UserWarning` when a Laurent layer with metallic or high-contrast content
  (`Re(eps) < 0` anywhere, or a contrast ratio above 10) is solved for a TM or unpolarised source,
  naming `formulation='li'`. Then the default flip in the next minor release.
- **Regression tests.**
  1. On fixture A at N = 100, the default call equals the explicit `'li'` call bit-for-bit.
  2. The TE Jones element equals the explicit `'laurent'` value to 1e-12.
  3. An explicit `'laurent'` on a metal layer warns once (when the warning step is chosen).
  4. A dielectric-only stack is unchanged bit-for-bit.

### G-4. RCWAStack `'li'` on metal strips: N^-1.1..1.2, with a factor-3 N-to-N scatter

**What is wrong.** This is not a defect in the factorisation. It is a convergence property that
limits what RCWAStack can deliver on this problem class.

**Reproduction (fixture A, analytic `shapes=`, `formulation='li'`; errors in the G-3 table):**
- **The rate.** At N = 200 / 400 / 600 / 800 the error is 4.88e-3 / 2.30e-3 / 1.43e-3 / 1.00e-3. The
  local exponents are 1.08 / 1.17 / 1.24. It is still 1.0e-3 from r_ref at N = 800 (68 s per solve).
- **The candidate claim of about N^-1.5 is NOT REPRODUCED.** The measured rate is 1.1-1.2, in line
  with the documented L-1 rate (about N^-1.1).
- **The candidate claim of "still about 1e-3 at N about 500" is CONFIRMED:** 1.4e-3 at N = 600.
- **The N-to-N scatter.** Consecutive N do not converge smoothly. For N = 40 ... 70, the error runs:

  | N | 40 | 41 | 44 | 45 | 50 | 51 | 55 | 57 | 61 | 63 | 64 | 69 | 70 |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|---|
  | error | 1.85e-2 | 3.50e-2 | 4.54e-2 | 1.71e-2 | 3.92e-2 | 4.01e-2 | 1.42e-2 | 1.41e-2 | 1.66e-2 | 1.59e-2 | 1.52e-2 | 2.36e-2 | 1.22e-2 |

  That is a factor-3 spread between neighbours.
  - **The scatter is not a cascade artefact.** `symmetry='auto'` and `symmetry=False` agree to 1e-12
    at every N tested.
  - **It is a property of the truncated strip operator.** It shows on the same metal strip directly on
    the substrate (no spacer, no mirror): errors of 5.7e-4 to 1.7e-3 against N = 400, for N = 44 ... 56. The
    resonant stack amplifies it about 25x.
- **Other shipped options do not do better on this case:**
  - `add_tapered_grating(raster='harmonic')` converges to the analytic `'li'` value at the same N,
    not past it (G-6);
  - adaptive spatial resolution (`asr_eta`) exists only in `rcwa_efficiency_1d`, a single binary
    layer between half-spaces at normal incidence (`rcwa/oned.py` 435-466, 650-662), so it cannot
    represent this stack;
  - `rcwa_extrapolate` assumes a smooth tail (`rcwa/_core.py` 3510-3516), which the scatter
    violates below N of about 100.

**Why it matters.**
- **PMM is far cheaper here.** For 1e-3 accuracy RCWAStack needs N of about 800 (68 s). Built-in
  PMM needs degree 56 (0.54 s), and the G-1 prototype reaches 1e-5 in 1.2 s.
- **A two-point convergence check is unreliable.** At a moderate N, a check between neighbouring
  truncations can land on either side of the scatter. G-5 is the consequence.

**Root cause.** The algebraic rate is the corner singularity (L-1). The scatter is the truncated
Fourier representation of the strip edges. Its phase relative to the edges changes with N, and the
resonance amplifies that.

**Proposed fix (documentation, plus one feature).**
1. Extend the L-1 paragraph in `add_layer` (`rcwa/stack.py` 1842-1863) and in `rcwa_extrapolate`.
   Say that on resonant metal-strip structures, consecutive N scatter by a factor of 2-3 below
   N of about 100. Tell users to step N geometrically (x1.5 or x2), not by small increments, and
   point to PMMStack for 1-D metal references.
2. Missing feature: adaptive spatial resolution (a coordinate stretch at the strip walls) for
   `RCWAStack` 1-D patterned layers. It can build on `_asr_metric_profile` / `_asr_convolutions`
   (`rcwa/oned.py` 179-280). The stack needs one stretch common to all layers, since the uniform
   layers must use the same coordinate. The walls of every patterned layer must sit at the
   clustered points, so this applies cleanly when all patterned layers share wall positions, as
   here.
- **Regression test.** Not applicable to item 1. For item 2: on fixture A, ASR `'li'` at N = 100
  beats plain `'li'` at N = 400.

### G-5. `rcwa_convergence` declares convergence while R is 23% wrong

**What is wrong.** `rcwa_convergence(stack)` (`rcwa/stack.py` 626-710) compares the solve at N with
the solve at N + 4 (`bump=4`). It declares convergence when the largest per-order efficiency change
is at most `atol=1e-3`, an absolute tolerance. For an algebraically converging method, a 4-order
step (about 6% at N = 60) changes the answer by a small fraction of its error. And an absolute
efficiency tolerance says nothing about a low-reflectance state.

**Reproduction (fixture A, `formulation='li'`, analytic strip; R_ref = |r_ref|^2):**

```python
st = <fixture A RCWAStack with n_orders=N, 'li'>; hi, rep = rcwa_convergence(st)
```

| N -> N+4 | `converged` | delta | returned R | R error | error in r |
|---|---|---|---|---|---|
| 57 -> 61 | **True** | 9.1e-4 | 0.01605 | 21.6% | 1.66e-2 |
| 61 -> 65 | **True** | 3.8e-4 | 0.01567 | 23.5% | 1.89e-2 |
| 63 -> 67 | **True** | 1.0e-4 | 0.01675 | 18.2% | 1.52e-2 |
| 66 -> 70 | False | 1.5e-3 | 0.01714 | 16.3% | 1.22e-2 |
| 100 -> 104 | **True** | 5.1e-4 | 0.01809 | 11.7% | 8.97e-3 |
| 300 -> 304 | **True** | 2.7e-5 | 0.01964 | 4.1% | 3.03e-3 |

- **The verdict flips with the scatter.** N = 61 says converged and N = 66 says not converged, at
  similar errors.
- **Laurent is flagged** (`converged=False`) at every N tested, because its oscillation is larger.
  That is right, but by luck.

**Why it matters.** The function's docstring offers it as "cheap insurance against silently
under-resolved physics". Here it does the opposite: it certifies a 20%-wrong reflectance with no
warning.

**Root cause.** The step size and the tolerance are both absolute. The check also reads only the
efficiencies, never the complex amplitudes or a rate.

**Proposed fix.**
- **Step geometrically:** `bump` becomes a factor, defaulting to 1.5 (N -> ceil(1.5 N)). Keep an
  integer `bump` working for compatibility, with a deprecation note.
- **Use a relative tolerance:** `rtol=1e-2` on each order's efficiency relative to
  `max(R_m, 1e-3 * sum R)`. Also compare the complex zeroth-order amplitudes from
  `per_order_amplitudes()`.
- **Report an error estimate,** not just a flag. From three solves (N, 1.5N, 2.25N), report the
  local exponent and a Richardson error estimate. Mark the result unconverged when the exponent is
  erratic, that is, when the three-point ratio is negative or the exponent is outside [0.5, 4].
- **Regression test.** Fixture A, `'li'`, N = 61: the new default reports `converged=False`. At
  N = 600 with `rtol=5e-2` it reports `converged=True`. A dielectric grating that converges
  exponentially stays `converged=True` at its current N.

### G-6. Point-sampled `eps_cell` rasters: realised width and aliasing errors up to 0.05 in r (works as designed)

**What was claimed.** With point sampling, a strip edge on or near a lattice node changes the
realised width. The claim was an r error of 0.05 at N = 50 with an 8N raster.

**Reproduction (fixture A, `formulation='li'`, N = 50).** Each raster is compared with the analytic
`shapes=` + `'li'` solve at the same N, so only the raster is measured.
Nodes sit at `j P/S`; the strip is `[x0, x0 + W)`:

| S | strip edge | realised width [nm] | difference from shapes |
|---|---|---|---|
| 400 | on a node (x0 = 0) | W + 0.750 nm | **5.07e-2** |
| 400 | half-pixel offset | 174.825 | 1.80e-2 |
| 1600 | either | W + 0.056 nm | 4.99e-3 |
| 3200 | either | W + 0.056 nm | 5.74e-3 |
| 6400 | half-pixel offset | 174.998 | 2.18e-4 |

`add_tapered_grating(..., n_slices=1, formulation='li')`, which paints on pixel centres:

| `n_x` | `raster='hard'` | `'area'` | `'harmonic'` |
|---|---|---|---|
| 400 | 5.07e-2 | 4.49e-2 | 2.08e-2 |
| 1600 | 1.48e-2 | 1.64e-2 | 1.87e-3 |
| 6400 | 5.92e-3 | 2.49e-3 | 1.24e-4 |

At N = 100, n_x = 12800, `'harmonic'` is 2.8e-6 from the analytic value.

- **The 0.05 claim is CONFIRMED in size.** It comes from S = 400 with a strip edge on a node.
- **Most of the error is width.** The 0.75 nm width error times |dr/dW| = 0.05 per nm accounts for
  most of the 0.051.
- **There is also a pure aliasing term.** S = 1600 and S = 3200 realise the same width but differ by
  7.4e-4 at N = 50. The point-sampled DFT is the series of the band-limited interpolant, not of the
  step.

**Classification: WORKS AS DESIGNED.**
- **The behaviour is the documented PIXEL CELL CONTRACT** (`rcwa/stack.py` 218-347, including the
  L-6 table added in 5.50.1). The remedies it lists work here.
- **The best remedy, an analytic `shapes=` strip with `'li'`, became available only in 5.50.1**
  (L-4). On 5.50.0 the call raises `ValueError`. The best 5.50.0 route for an exact-geometry Li
  solve was `add_tapered_grating(raster='harmonic')` at large `n_x`.

**Residual usability gap (low).**
- **No `segments=` on RCWAStack.** A 1-D strip still has to be written either as a "rectangle
  spanning the full period_y" shape dict or as an `eps_cell`. `PMMStack.add_layer(segments=...)`
  has no RCWAStack counterpart. The easiest path for a user is a hand-painted `eps_cell`, which is
  the least accurate.
- **Proposed.** Add `RCWAStack.add_layer(thickness, segments=[(width_fraction, eps), ...])` on 1-D
  stacks, mapped internally to full-height rectangles on the analytic path, with the G-3 default
  formulation. Optionally, warn when a 1-D `eps_cell` holds two or more distinct values in long
  runs (a hard raster) and `'li'` is used, naming `segments=` / `shapes=`.
- **Regression tests.**
  1. `segments=` reproduces the full-height-rectangle `shapes=` result bit-for-bit.
  2. Its `'li'` result converges to the PMM reference at the G-4 rate.

### G-7. `PMMStack(n_orders=...)` is an alias of `far_field_orders` and has no effect on accuracy

**What is wrong.** `PMMStack.__init__` accepts `n_orders` "as a cross-suite alias (the RCWA / 2-D
PMM spelling)" (`pmm/stack.py` 1713-1716, via `_resolve_order_count`, `pmm/_core.py` 79). In
RCWAStack, `n_orders` is the truncation, the convergence knob. In PMMStack it sets only how many
Rayleigh orders are projected in the far field. The modal truncation is `degree` x
`elements_per_region`.

**Reproduction (fixture A, degree 20, 3 elements per region):**

| `n_orders` | r | error |
|---|---|---|
| 3 | (the same r) | 1.881e-3 |
| 21 | (the same r) | 1.881e-3 |
| 61 | (the same r) | 1.881e-3 |

The three results are identical, so a convergence study in `n_orders` is perfectly flat while the
answer is 1.9e-3 off.

**Why it matters.** It produces false confidence. The value looks converged because the knob is
inert.

**Proposed fix.**
- **Docstring.** State in the `PMMStack` docstring that `n_orders` / `far_field_orders` does not
  control accuracy, and that `degree` / `elements_per_region` (and G-1's grading) do.
- **Optional guard.** Emit a one-time `UserWarning` when `n_orders` (the alias, not
  `far_field_orders`) is passed, saying it is the far-field order count.
- **Regression test.** Passing `n_orders=` warns once; passing `far_field_orders=` does not.

### G-8. `grade=True` is inert at 1 and 2 elements per region; the knob docs contradict each other

**What is wrong.**
- **Where grading is a no-op.** `_graded_boundaries` (`pmm/_core.py` 462-467) returns `[a, b]` for
  one element. For two elements, Chebyshev-Lobatto gives the midpoint, the same as uniform.
  Measured: `grade=True` and `grade=False` at 2 elements agree to 6e-14. So the default `grade=True`
  does nothing at the default `elements_per_region=1`, or at 2.
- **The docstrings send users the wrong way:**
  - `pmm/oned.py` 133 says "Raise with `grade=True` to resolve the wall-corner field".
  - `pmm/oned.py` 475 says "`elements_per_region>1, grade=True` recovers the rate for metals".
  - The table in the same docstring (`pmm/oned.py` 392-404) says single-element p-refinement beats
    that by 2.5-4x per DOF. G-1 measures 2.1-2.5x, so the table is right.
  - The `PMMStack` docstring (`pmm/stack.py` 1643-1645) refers the knob to `pmm_jones_1d_segments`
    and gives no guidance for metals.

**Proposed fix.**
- Remove the "recovers the rate" sentence and correct line 133.
- In the `PMMStack` docstring, add three lines:
  - for metal corners, raise `degree` with one element;
  - grading only matters from 3 elements upward;
  - G-1's geometric mode is the efficient choice once it exists.
- Not a code change. A docstring test can pin the absence of the contradicting sentence, if the
  project does that.

### G-9. The phase reference plane of `jones_reflection` is not stated

**What was checked.**
- **The behaviour is correct.** For an 80 nm uniform lossy film on the substrate, at
  theta = 0 and 0.3 rad, in TE and TM, PMMStack and RCWAStack both return exactly the analytic
  tangential-field reflection coefficient. That coefficient is `exp(-i w t)`, lab-basis `E_x`
  (`(Y0 - Y1)/(Y0 + Y1)` with `Y = n^2/kz` for TM), and referenced to the top of the first layer.
  The agreement is to 6 decimals.
- **The reference plane confirmed by a shift.** Adding 100 nm of air on top of fixture A multiplies
  r by 0.568065 + 0.822984i, which equals exp(2i k0 x 100 nm).
- **The time convention is documented** in `pmm_jones_1d` (`pmm/oned.py` 158-163) and in
  `CONVENTIONS.md` section 7.
- **The gap.** Neither the `PMMStack.solve` docstring (`pmm/stack.py` 2840-2843, which refers to
  the function) nor `RCWAResult.jones_reflection` (`rcwa/stack.py` 868) says where the phase is
  referenced. Callers comparing with papers that use `exp(+i w t)`, or that reference the phase at
  another plane, need both facts, and today they have to measure them.

**Proposed fix.** One docstring sentence on both methods: "zeroth-order reflected tangential lab
`(E_x, E_y)` over incident, `exp(-i w t)`, phase referenced at the top interface of the first layer
(the superstrate interface); conjugate for `exp(+i w t)`". Add a small test that pins the
uniform-film value against the closed form, and the exp(2 i k0 d) shift under an added air layer.

---

## 4. Items checked with no defect found

- **Lossy semi-infinite metal half-space.** A semi-infinite metal substrate (the mirror's permittivity) under 30 nm of a spacer dielectric, at
  theta = 0 and 0.4 rad, TE and TM:
  - PMMStack and RCWAStack give R + T = 1.000000000;
  - `T` equals 1 - |r_analytic|^2 (three incidences);
  - no warnings.

  `T` is the power entering the absorbing half-space. This is correct, and arguably worth one
  docstring sentence, since some users read "T" as transmitted-and-escaping power.
- **Even-parity fold.** `RCWAStack.solve(symmetry='auto')` vs `symmetry=False` on fixture A
  (`'li'`, N = 44, 50, 51, 55): agreement 2.8e-13 to 1.3e-12.
- **Version stability.** On the cases measured, 5.50.0 and 5.50.2 are identical to round-off: the
  PMMStack defaults, degree 20 with 3 elements, Laurent `shapes=` at N = 50 / 200, and `'li'`
  `eps_cell` at N = 50 / 200. The G-2 reference case differs by at most 1.3e-11.

---

## 5. Prioritised fix order

1. **G-3** (RCWAStack default `'laurent'`). It is the largest silent error (0.26 in r at N = 50) and
   the smallest change. Ship the warning in a patch release, and `'auto'` in the next minor.
2. **G-5** (`rcwa_convergence` false verdict). The helper users rely on to catch exactly G-3/G-4
   errors certifies a 23%-wrong R. Switch to geometric steps and a relative tolerance.
3. **G-1** (geometric grading for PMMStack). A 50-100x accuracy-per-time gain on every metal-corner
   1-D problem. Prototyped; mostly one function plus validation.
4. **G-2** (homogeneous-layer fast path and run collapse). A 1.8x gain from a one-line dispatch,
   and 20x+ with the diagonal run collapse on graded stacks. Prototyped and exact to 1e-12.
5. **G-7, G-8, G-9** (documentation and API traps). Cheap; do them together with G-1.
6. **G-4** (documentation now). ASR for RCWAStack is a separate feature, behind G-1, because PMM
   already covers the 1-D reference use.
7. **G-6** (`segments=` on RCWAStack). Low; it makes the exact path the easy path.

---

## 6. Evidence

The scripts were run from a scratch folder against the detached 5.50.2 checkout and a `v5.50.0`
checkout. They are not part of the repository. Each table above lists the parameters needed to
re-run it from the section 2 fixture. The prototypes in G-1 (`_geometric_boundaries`) and G-2
(uniform dispatch and diagonal run collapse) were implemented by monkey-patching
`lumenairy.elements.pmm._core._graded_boundaries` and `lumenairy.elements.pmm.stack._sem_modes_tensor`
in the scratch process only. The run collapse was a re-implementation of the all-vertical cascade
from the library's own helpers. It reproduced the unmodified library to 7.1e-13 before the collapse
was switched on.
