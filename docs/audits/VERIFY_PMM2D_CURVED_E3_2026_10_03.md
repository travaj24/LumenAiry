# VERIFY -- curved cells for the pure staggered 2-D PMM, Phase E3 (the JAX twin, differentiable in shape parameters)

Date: 2026-10-03.  Verifier: Claude Opus 5.5 (model ID `claude-opus-5-5`),
independent of the builder.
Object: branch `feat/pmm2d-curved-e3-jax`, commits `4d56450c` .. `d4e92eb5`
on the Phase D tip `eae470d9`; build record
`docs/audits/BUILD_PMM2D_CURVED_E3_2026_10_03.md`.
Mount: worktree `C:/tmp/lum_vcurved_e3`, branch `verify/pmm2d-curved-e3` at
`d4e92eb5`; PRE tree = `git archive eae470d9` in `C:/tmp/vE3_pre`; mutant /
fix scratch trees `C:/tmp/vE3_*` (archives of HEAD, never the worktree).
Builds: Windows 11 (CPython 3.14.6, numpy 2.4.4, scipy 1.17.1, jax 0.11.0)
and WSL Ubuntu (CPython 3.12.3, numpy 2.4.6, scipy 1.17.1, jax 0.10.2).
`jax_enable_x64` on, `OMP / OPENBLAS / MKL_NUM_THREADS = 1` on every command
line, XLA single-threaded, `lumenairy.__file__` asserted under the tree in
every probe (`verify_e3/_ve3.py`).  The box was shared with sibling builds
(20-30 Python processes): every wall time is an upper bound.
Evidence: every number below is read from a JSON in
`validation/probe_pmm2d_curved/verify_e3/` (suffix `_win` / `_wsl` = build).

---

## 0. Words

* **Twin**: `PMM2DStackPure(backend='jax')` / `StagJaxTwin`, built from a
  concrete **reference** at which every discrete decision is frozen.
* **AD**: its reverse-mode gradient.  **FD(twin)** / **FD(numpy)**: central
  differences (Richardson of the last two rungs, h^2 law) of the twin's own
  forward / of the shipped NumPy solve, whose grid MOVES with the parameter.
  The h^2 **premise** is recorded on every ladder: for the rungs
  h / P = 1e-3, 3e-4, 1e-4 the ratio of successive rung changes must be
  11.375.
* **Offset**: AD minus FD(numpy) at the same `n_modes = M`.
* **Truth**: FD(numpy) at the highest M run (8 or 9; 7 for the fillet).

---

## 1. Verdict table

| item | claim | verdict | evidence |
|---|---|---|---|
| E3-1 | NumPy bytes unchanged; the NumPy path never touches the twin | **CONFIRMED** | own 164-key set over 26 classes (operators + modes of scalar / tensor / gyrotropic / magnetic / mapped cells, every primitive's layout and map, the mapped far field, incident load and decomposition, geometric caches, order kz, branch selector, curves, OOP, slant, per-layer, retain_internal + layer_absorption, the entries): 164 / 164 SHA-256 equal to eae470d9 on BOTH builds; with an import hook making the twin module RAISE, and with jax itself blocked: 164 / 164, twin never imported (`v1_compare_{win,wsl}.json`) |
| E3-2 | forward parity, bar from the eig stage's round-off | **CONFIRMED with a scope defect (V-E3-2)** | own 9 fixtures (lossy magnetic ellipse rotated and axis-aligned at conical, two-layer merged map 5 x 5, two circles 5 x 5, cored circle 5 x 5 conical, LC fillet conical, sinusoidal ridge oblique, gyrotropic + magnetic circle, magnetic multilayer): max twin - NumPy 6.2e-14 (M = 3) / 6.7e-13 (M = 4), WSL 7.1e-14.  The "2.9x" is a per-fixture observation (max over R, T, J), not a bound: 4.0x on my fixtures (6.9x per quantity); the unit gate is one global 5e-12.  **Rectangles-only shape stacks at oblique incidence are NOT at round-off: 2.5e-4 at M = 3** (V-E3-2) |
| E3-3 | gradients vs Richardson FD | **CONFIRMED where the gradient is non-degenerate; WRONG at symmetric cells (V-E3-1, P1)** | premise-checked ladders, both builds: Im eps 7.5e-10, n_substrate 2.7e-11, ellipse semi-axis 3.9e-10, second-layer depth 2.4e-11 vs FD(twin); 2-parameter jacobian (r, eps) 3.9e-10; LC film vs the Berreman JAX oracle 3.7e-15 (normal, exact); **a square pillar's width: 3.1e-3 (M 3), 25 % (M 4), 14 % (M 5) on Windows, 3.8e-2 / 6.9e-2 on WSL** |
| frozen grid | the twin differentiates a re-parametrisation; the offset falls with M | **CONFIRMED and characterised** | section 2: AD error vs the converged truth = the NumPy solver's own gradient error to every printed digit at M = 4 .. 7; the offset sits 3.5 - 6 decades below it; zero by construction for the sinusoid (amplitude and position); the frozen node count costs 1e-10 where NumPy's adaptive count switches |
| second order | -- | **NOT AVAILABLE** | nested `grad` raises `NotImplementedError`, `jvp` / `jacfwd` / `hessian` raise `TypeError` (custom VJP) -- both builds; missing from the CHANGELOG limits (V-E3-4) |
| E3-4 | events NaN when traced, refused when concrete | **CONFIRMED with one gap (V-E3-3, P3)** | 8 own boundary cases: fold (bulge, sinusoid past the edge), fillet below its sliver limit, an arc crossing a rectangle's edge, two circles' walls reordering, two rectangles reordering: NaN in every entry, in sum(R) + sum(T), in d/dx and in d/d(eps); control finite.  **The merge's inside-the-cell margin (circle / sinusoid within the sliver width of the cell edge) is refused concretely but FINITE when traced**; the old `jnp.where` guard mutant is caught by the build's fold test |
| E3-5 | degenerate pairs give the correct gradient | **REFUTED for symmetry-breaking parameters (V-E3-1, P1)** | correct for symmetry-preserving directions (square w = h together 4.1e-11) and for gradients zero by symmetry (d/dcx 4e-15); wrong for a parameter that splits the degenerate pairs to first order; the error is build-dependent (LAPACK's arbitrary basis in a degenerate subspace); it vanishes once the cell is >= 1e-9 off symmetric |
| E3-6 | jit traces once; timings | **CONFIRMED** | 10-radius sweep: 1 trace forward, 1 gradient, 1 for a `vmap`ped gradient, cache size 1, both builds; compiled forward 0.04 - 0.28x and gradient 0.06 - 0.28x of one NumPy solve (M = 4, 6, 7); peak working set after the gradient compile 0.75 / 1.02 / 1.45 GB (M = 4 / 6 / 7), 5.3 GB after the vmapped 10-radius gradient at M = 7 |
| E3-7 | TM 2.9e-2 vs the 1-D PMM is the 2-D discretisation | **CONFIRMED** | the NumPy 2-D solver's OWN d R00 / d eps (FD) is 2.902e-2 from the 1-D AD, the twin is 2.2e-9 from the NumPy 2-D FD (WSL 8.7e-9) -- `v8_stripe_grad_M7_*.json` |
| E3-8 | census | **CONFIRMED with a gap (closed)** | a same-name copy and a renamed copy used at every site are caught; a **renamed verbatim copy used at ONE of two call sites passes** the build census and the library census tests; closed by an AST body comparison (`test_ve3_twin_module_holds_no_renamed_copy_of_a_shared_kernel`, fails on the mutant) |
| E3-9 | mutations caught | **ONE SURVIVOR (closed)** | incident node count re-read in the trace: 8 failures (caught); **broadening tau = 0: 27 / 27 pass (survivor)** though it is off by 0.117 on a non-degenerate rectangle (closed: `test_ve3_the_eig_regularisation_is_load_bearing`); the cofactor's determinant from the REFERENCE map: forward at the reference unchanged, AD = FD(twin), caught only by the rectangle's FD(numpy) test (curved cells: closed by `test_ve3_sinusoid_wall_position_gradient_matches_numpy_fd`, the mutant is off by 1.7) |
| F-E3-5 | consecutive-M identical stripe values | **CONFIRMED and explained** | a parity selection rule (section 5) |
| docs | numbers vs JSON, examples run, limits complete | **examples run on both builds; 5 doc corrections** | section 6 |
| switches / hygiene | `LUMENAIRY_DISABLE_JAX`, mypy, ruff, fingerprints | **CONFIRMED** | clean `ImportError` (the entry names the stack, cosmetic); mypy configured list clean; WSL ruff clean; fingerprints `--check` OK with the E3 reason; no forward version token |

---

## 2. The frozen grid -- what the twin differentiates

### 2.1 Derivation

The staggered basis on a cell is the polynomial space in the cell's LOCAL
coordinate; the Gordon-Hall map of a cell is a function of the local
coordinate and of the cell's edge curves.  The twin freezes the `(u, v)`
walls and lets a parameter move the vertex images and the curves; NumPy
rebuilds the walls at the new value.  The two discretisations therefore have
the same physical cells, the same local maps and spaces that differ only by a
per-row / per-column rescaling of the covariant coefficients -- the in-plane
pencil, its eigenvalues and the cofactor far field are the same.  What is NOT
invariant is the incident decomposition: the L2 projection of the incident
plane wave weights every cell by its `(u, v)` area, which is frozen in the
twin and moves in NumPy.  So

    T_twin(p; p0, M) = T_numpy(p; M) + Delta(p; p0, M),   Delta(p0; p0, M) = 0,

and the twin's gradient is `dT_numpy/dp + dDelta/dp` at `p0`.  `Delta` lives
at the incident representation error and is exactly zero when the incident
field is representable on the cells along the axis whose walls move:

* affine cells at NORMAL incidence (the builder's rectangle);
* a sinusoidal wall, for its amplitude AND its position: the
  non-polynomial dependence runs along `v`, the walls that move are `u`
  walls (measured: 5e-15 at a 0.025 P move, offset 1e-10 at every M);
* NOT a circle, ellipse or fillet (curved in both directions), and NOT an
  affine cell at OBLIQUE incidence (the plane wave `exp(-i alpha x)` is not a
  polynomial on the cells) -- the last case contradicts the build record's
  "affine cells coincide to round-off" (V-E3-2).

### 2.2 Measured against the converged truth (`v2_analysis_win.json`)

Relative to the converged gradient (max over R00, T00 E_x, T00 E_y):

| family (reference) | M | value error | FD(numpy) error | **AD error** | offset AD - FD(numpy) | AD - FD(twin) | twin - NumPy at +-0.025 P |
|---|---|---|---|---|---|---|---|
| circle r 0.33 (truth M 9) | 4 | 5.8e-2 | 1.32 | **1.32** | 1.1e-4 | 1.7e-10 | 2.8e-6 |
| | 5 | 1.3e-2 | 3.1e-1 | **3.1e-1** | 4.0e-6 | 3.7e-9 | 1.1e-7 |
| | 6 | 6.5e-3 | 1.6e-1 | **1.6e-1** | 2.3e-7 | 1.5e-9 | 5.2e-9 |
| | 7 | 5.2e-4 | 1.4e-2 | **1.4e-2** | 1.0e-8 | 9.7e-9 | 2.8e-10 |
| ellipse a 0.33 (truth M 8) | 4 | 4.4e-3 | 4.7e-1 | **4.7e-1** | 9.0e-5 | 4.2e-10 | 1.1e-6 |
| | 5 | 6.4e-4 | 1.0e-1 | **1.0e-1** | 3.4e-6 | 7.3e-10 | 9.2e-8 |
| | 6 | 7.1e-4 | 5.2e-2 | **5.2e-2** | 2.8e-7 | 3.6e-8 | 3.1e-9 |
| fillet r 0.12 (truth M 7) | 4 | 2.8e-3 | 5.6e-1 | **5.6e-1** | 2.8e-5 | 1.1e-8 | 2.1e-7 |
| | 5 | 1.7e-4 | 6.6e-3 | **6.6e-3** | 1.2e-7 | 1.2e-7 | 4.9e-10 |
| sinusoid position x0 0.6 (truth M 9) | 4 .. 7 | 3.8e-2 .. 7.0e-5 | 4.6e-1 .. 2.5e-3 | same | 8e-11 .. 9e-10 | <= 1.1e-9 | <= 7e-14 |
| sinusoid amplitude 0.1 | 4, 5 | | | | 7.8e-10, 1.7e-9 | | 6e-15 |

Every ladder satisfied the h^2 premise (ratios 11.37-11.38).  WSL repeats
the circle and the sinusoid position at M = 4 to the printed digits.

**What this means.**  The twin's gradient converges to the true gradient of
the converged solution at exactly the rate the NumPy solver's own gradient
does: its error and NumPy's agree to every printed digit, and the frozen-grid
offset is 3.5 - 6 decades smaller than either.  The offset is not what limits
a user -- the discretisation is: the circle's dT/dr at M = 4 is off by 130 %
(sign included), at M = 6 by 16 %, at M = 7 by 1.4 %.  Gradients converge
more slowly than values (value error 6.5e-3 where the gradient's is 1.6e-1 at
M = 6).

### 2.3 Where NumPy's adaptive decisions switch (`v2b_nqswitch_win.json`)

The node-count scan (`v2a_nq_scan_win.json`) puts the circle's switch at
`r* = 0.397570` for M = 4 (16 -> 32 nodes per axis per cell; at M = 5 / 6 / 7
it switches at 0.42-0.44 / 0.44-0.46 / 0.46-0.48; the fillet and the
sinusoid never switch over their ranges; the Duffy set never changes
continuously -- it changes only at the fillet's r = 0, a poisoned event).
At `r*` NumPy's own value jumps by 2.0e-14; a NumPy FD straddling the switch
is contaminated by 1.6e-9 at h = 1e-6 P.  The twin frozen at r0 = 0.36 (16
nodes) evaluated beyond the switch (r* + 1e-4, 0.42, 0.46, 0.50): FD(numpy)
with the count forced to 16 vs 32 differs by 1.3e-11 .. 3.1e-10 relative,
while the gradient's discretisation error at the same points (M = 4 vs 6) is
0.21 .. 1.98.  **The frozen count's error is bounded by the criterion's
tolerance, ten decades below the discretisation error.**

---

## 3. Defects

### V-E3-1 (P1) -- a symmetry-breaking shape gradient at a symmetric cell is wrong (silently, build-dependently)

At an exactly four-fold symmetric cell the Bloch modes come in degenerate
pairs.  A parameter that BREAKS the symmetry (a square pillar's width alone,
a circle deformed into an ellipse, a square fillet's width) splits those
pairs to FIRST order, and the gradient needs the in-pair coupling term
`f'(lam) (V^-1 dA V)_ij`.  In reverse mode the eigenvector VJP sees that
term only as `F_ij M_ij` with `M_ij = O(dlam)` and `F_ij = 1 / dlam`; the
broadening sets `F_ij = 0` for `|dlam| < tau max|lam|` and drops it, and
LAPACK's basis inside the degenerate subspace is arbitrary (build-dependent),
so the error is neither zero nor reproducible.  Measured (AD vs FD(twin) and
vs FD(numpy); both FDs satisfy the h^2 premise and agree with each other to
1e-10; `v6b_symbreak_*.json`):

| case | M | Windows | WSL |
|---|---|---|---|
| square pillar, d / d w (w = h = 0.5) | 3 | 3.1e-3 | 3.8e-2 |
| | 4 | **2.5e-1** | 6.9e-2 |
| | 5 | 1.4e-1 | -- |
| circle -> ellipse, d / d a (a = b = 0.33) | 3 / 4 | 6.5e-3 / 3.0e-3 | -- |
| square fillet, d / d w | 3 | **2.9e-1** | -- |
| control: square, w = h together (symmetry kept) | 3 / 4 | 4.1e-11 / 7.4e-11 | 4.3e-11 |
| control: non-square pillar d / d w | 3 | 3.4e-11 | -- |

The off-symmetric sweep (ellipse a = r, b = r (1 + delta), `v6_degenerate_M3_*`):
error 1.8e-3 (delta 0), 1.1e-1 (1e-14), 1.1e-3 (1e-13), 8.0e-6 (1e-12),
3.2e-6 (1e-11), 1.2e-8 (1e-10), <= 7e-11 from 1e-9 on.  The broadening
setting does not rescue it (tau 1e-14, 1e-8: same errors; tau = 0: worse).
Both eigs carry it (`v6c_where_M3_win.json`: splitting only the geometric
pencil leaves 8.5e-4, only the layer pencil 2.3e-3, both 2.5e-7).

The build's E3-5 tested only directions in which the in-pair term vanishes
(d/dr of the circle keeps the symmetry; d/dcx at the centre is zero by
mirror parity), so it could not see this.

**Fix (prototype measured, `C:/tmp/vE3_fix_degen`).**  In
`_stag_geneig_jax`, split exact degeneracies by a fixed, deterministic, tiny
non-symmetric perturbation before the eig, so every pair is resolved above
the broadening and `F_ij M_ij` carries the coupling:

```python
_E3_EIG_SPLIT_REL = 1e-8          # next to _E3_EIG_TAU_REL

    A = jnp.linalg.solve(G, L)
    if _E3_EIG_SPLIT_REL:
        import jax
        n = A.shape[0]
        rng = np.random.default_rng(20261003 + n)
        Rm = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        Rm = Rm / np.linalg.norm(Rm, 2)
        sc = jax.lax.stop_gradient(jnp.max(jnp.abs(A)))
        A = A + (_E3_EIG_SPLIT_REL * sc) * Rm
    eig = _jax_eig_stable()
```

Measured with it: square d/dw 2.4e-7 (M 3) / 7.9e-7 (M 4), fillet 3.3e-7,
non-square control 1.4e-8; the strict-xfail arm turns into an XPASS.  Cost: a
forward bias of ~1e-8 relative (FD(twin) - FD(numpy) 1e-8 .. 2e-7), so the
bars that assume round-off parity must be re-derived around it: in the
prototype tree 12 of the 27 build ids move (the 10 E3-2 parity ids at 5e-12,
E3-5's exact-zero d/dcx at 1e-9, the convenience entry's parity;
`logs/mut_vE3_fix_degen.txt`).  1e-10 is too small (2.2e-3 remains with the
default tau; 4e-6 with tau 0).  The cleaner alternative -- differentiating
the layer S-matrix through matrix functions (no eigenvector derivative) --
is a larger change.  Pinned by `test_ve3_symmetry_breaking_gradient_at_a_square_pillar`
(`xfail(strict=True)`, fails on HEAD on both builds).  **Until fixed, the
CHANGELOG must say** (insert in "Known limits"):

```
* A shape gradient that BREAKS a symmetry of the evaluated cell, taken AT
  the symmetric point (a square pillar's width alone, a circle deformed into
  an ellipse, a square fillet's width), is wrong: the four-fold symmetry
  makes degenerate Bloch-mode pairs whose first-order splitting the
  eigenvector derivative cannot see (0.3 - 29 % measured, different on
  different BLAS builds).  Evaluate at least 1e-9 (relative) off the
  symmetric point.  Gradients that keep the symmetry (a circle's radius, a
  square's side) are exact.
```

The same `rcwa._jax_eig_stable` serves the library's other JAX twins; a
symmetry-breaking gradient at a symmetric cell there was NOT measured and is
likely affected the same way.

### V-E3-2 (P2) -- rectangles-only shape stacks at oblique incidence: the twin is not the NumPy solve

`geometry='auto'` makes the twin of a `Rect`-only `shapes=` stack MAPPED (an
identity transfinite map, so walls can be traced), and the mapped route
decomposes the incident wave by the L2 projection; the NumPy stack runs such
a stack UNMAPPED (least-squares Rayleigh overlap).  At normal incidence both
are exact (3e-15); at oblique / conical incidence they differ by the
incident representation error (`v4b_rect_conical_M*_win.json`, theta 0.3):

| M | twin(auto) - NumPy, T | twin(static) - NumPy | twin(auto) - NumPy identity-map route | AD - FD(numpy, identity map) |
|---|---|---|---|---|
| 3 | 2.5e-4 | 8.5e-15 | 8.1e-15 | 4.8e-4 |
| 4 | 3.8e-5 | 3.9e-14 | | 3.5e-6 |
| 5 | 6.5e-7 | 9.4e-14 | | 2.9e-7 |
| 6 | 6.9e-9 | 1.6e-13 | | 3.2e-9 |

So the CHANGELOG's "forward results agree with NumPy to 1e-14 .. 5e-13" and
F-E3-1's "for affine cells the two coincide to round-off" are false at
oblique incidence; E3-2 has no rectangle-shape fixture off normal incidence.
Making the twin reproduce the unmapped least-squares route is NOT advised:
that route is itself gauge-dependent (O-1 below; a prototype gave 1e-7 instead
of round-off).  Exact edits (documentation): CHANGELOG, "Known limits", add

```
* At oblique / conical incidence a rectangles-only `shapes=` stack is solved
  by the twin on an identity coordinate map (the curved-cell incident
  projection), not on the NumPy stack's unmapped route: the two differ by
  the incident representation error (2.5e-4 at `n_modes = 3`, 3.8e-5 /
  6.5e-7 / 6.9e-9 at 4 / 5 / 6 on T, 0.3 rad); at normal incidence they agree
  to round-off.  `jax_twin(geometry='static')` reproduces the NumPy route
  exactly (no traced walls).
```

and in the paragraph above it replace "Forward results agree with NumPy to
1e-14 .. 5e-13 on 22 fixtures" by "Forward results agree with NumPy to 1e-14
.. 5e-13 on 22 fixtures (rectangles-only shape stacks at oblique incidence
excepted, see the limits)".  Pinned by
`test_ve3_rectangles_at_oblique_incidence_run_the_mapped_route`.

### V-E3-3 (P3) -- the merge's inside-the-cell contract is not replayed in the trace

`Shape2D._check_inside` (bounding box at least the sliver width from the
cell edges) and `SinusoidalWall`'s `base -+ |A|` margin are refused
concretely, but the traced guard checks only wall order, sliver segments,
claim coincidences and folds: a circle at r = 0.5995 (c = 0.6, P = 1.2) and a
sinusoid at amplitude 0.5995 return a finite value and gradient when traced
(`v5_events_M3_{win,wsl}.json`).  Exact edit,
`lumenairy/elements/pmm/_jax_twod_staggered.py`, in `_traced_shape_merge`
after

```python
    for d in devs:
        ok = ok & (jnp.max(jnp.abs(d)) <= tol)
```

insert

```python
    # the merge's INSIDE-THE-CELL contracts, traced (a concrete call is
    # refused by the merge): a sinusoidal wall's base -+ |amplitude| at least
    # the sliver width inside the cell (SinusoidalWall._layout), any other
    # curved shape's bounding box at least the sliver width from the cell
    # edges, a rectangle's inside it (Shape2D._check_inside)
    for (_w, sh_r, _l), sh_t in zip(items, flat_t):
        if isinstance(sh_r, SH.SinusoidalWall):
            pp = px if sh_r.axis == "x" else py
            A = jnp.abs(sh_t.amplitude)
            for base in sh_t._bases():
                ok = ok & (base - A >= TS._STAG_MIN_SEG_FRAC * pp) & (
                    base + A <= pp * (1.0 - TS._STAG_MIN_SEG_FRAC))
            continue
        x0, x1, y0, y1 = sh_t.bbox()
        strict = not isinstance(sh_r, SH.Rect)
        mx = TS._STAG_MIN_SEG_FRAC * px if strict else -1e-12 * px
        my = TS._STAG_MIN_SEG_FRAC * py if strict else -1e-12 * py
        ok = ok & (x0 >= mx) & (y0 >= my) & (x1 <= px - mx) & (y1 <= py - my)
```

Measured in `C:/tmp/vE3_fix_bbox`: all 8 boundary cases NaN at the event and
finite at the control (`v5_events_M3_fixbbox_win.json`); the strict-xfail arm
`test_ve3_inside_the_cell_contract_is_poisoned_when_traced` XPASSes.  (A
first version that applied the bounding-box rule to the sinusoid poisoned its
CONTROL point -- the wall fills to the cell edge by design -- hence the
sinusoid branch.)

### V-E3-4 (P3, documentation)

1. CHANGELOG "Known limits", add: "* Reverse mode only (`jax.grad`,
   `jax.jacrev`, `jax.vjp`): forward mode (`jax.jvp`, `jax.jacfwd`,
   `jax.hessian`) raises `TypeError` and a second derivative (nested
   `jax.grad`) raises `NotImplementedError` -- the pencil eig carries a
   first-order custom VJP."  (Measured on both builds,
   `v4_second_M3_*.json`, `v4_jac_M3_*.json`; pinned by
   `test_ve3_forward_mode_and_second_derivatives_raise`.)
2. Module docstring of `_jax_twod_staggered.py`, replace "As argued and
   measured in the build record, the discrete solution does not depend on
   WHERE the frozen ``(u, v)`` walls sit, only on the physical images of the
   cells, so the twin evaluated at a parameter ``p`` reproduces the NumPy
   solve built at ``p`` (whose own grid moves with ``p``) to round-off at
   normal incidence and to the level of the incident decomposition's
   representation error at oblique incidence." by "The in-plane operators do
   not depend on WHERE the frozen ``(u, v)`` walls sit, only on the physical
   images of the cells; the incident decomposition does (its L2 projection
   weights each cell by its frozen ``(u, v)`` area).  So the twin evaluated at
   ``p`` differs from the NumPy solve built at ``p`` by the incident
   representation error -- zero for affine cells at normal incidence and for
   a sinusoidal wall, 5.5e-6 on T at M = 4 for a 0.03 P change of a circle's
   radius, and nonzero for affine cells at oblique incidence (verifier
   V-E3-2)."  (The circle's 5.5e-6 IS at normal incidence.)
3. BUILD F-E3-1 and section 1.2: "For affine cells (rectangles) the two
   coincide to round-off" -> "For affine cells at NORMAL incidence the two
   coincide to round-off (at oblique incidence they differ at the incident
   representation level, verifier V-E3-2)"; "(affine cells, an exactly
   representable incident)" -> "(affine cells at normal incidence, an exactly
   representable incident)".
4. BUILD 3.3: "the rung-to-rung changes fall by 8.8, 11.4, 8.8 for every
   case" -> "... for every case except eps at M = 4 (8.8, 12.6, 5.5) and the
   conical circle at M = 4 (below)" (`f3_summary.json`).
5. BUILD 3.2: "The twin sits within 2.9x of the eig stage's own round-off on
   every fixture" -> "On these 22 fixtures the twin's largest difference over
   R, T and J sits within 2.9x (M = 3) / 2.0x (M = 4) of the eig stage's reach
   on the same fixture (up to 3.8x per quantity) -- an observation, not a
   bound (the verifier's fixtures: 4.0x, 6.9x per quantity); the unit gate is
   the single global bar 5e-12."  And 3.5 / F-E3-4 ("the regularisation is a
   safety margin on this fixture rather than a correction"): add "-- not in
   general: with tau = 0 a NON-degenerate rectangle's width gradient is off
   by 0.117 (round-off-split pairs of the homogeneous geometric eig), and the
   regularisation cannot recover a symmetry-breaking gradient at a symmetric
   cell (V-E3-1)."

Cosmetic: under `LUMENAIRY_DISABLE_JAX=1`, `pmm_jones_2d_staggered(...,
backend='jax')` raises the stack's message ("PMM2DStackPure(backend='jax'):
JAX is not available ..."); the entry could name itself.

### Test gaps closed (`tests/unit/test_verify_pmm2d_curved_e3.py`)

| gap | test | HEAD | mutant / fix |
|---|---|---|---|
| a NumPy solve imports the twin | `test_ve3_numpy_solves_never_import_the_twin_module` (subprocess with an import hook) | pass | -- |
| V-E3-1 | `test_ve3_symmetry_breaking_gradient_at_a_square_pillar` (strict xfail) + its control | xfail / pass | XPASS on the split fix |
| tau = 0 survivor | `test_ve3_the_eig_regularisation_is_load_bearing` | pass | (the mutant is off by 0.117) |
| frozen-geometry mistake in curved cells invisible to AD-vs-FD(twin) | `test_ve3_sinusoid_wall_position_gradient_matches_numpy_fd` | pass | det-ref mutant off by 1.7 |
| V-E3-2 | `test_ve3_rectangles_at_oblique_incidence_run_the_mapped_route` | pass | -- |
| V-E3-3 | `test_ve3_inside_the_cell_contract_is_poisoned_when_traced` (strict xfail) | xfail | XPASS on the fix |
| renamed copy | `test_ve3_twin_module_holds_no_renamed_copy_of_a_shared_kernel` | pass | fails on the renamed-copy mutant |
| reverse mode only | `test_ve3_forward_mode_and_second_derivatives_raise` | pass | -- |
| F-E3-5 | `test_ve3_stripe_pairs_are_a_parity_selection_rule` | pass | -- |

### O-1 (pre-existing, NOT E3; for the maintainers) -- the unmapped route's incident decomposition at oblique incidence depends on the eigenvector normalisation

The unmapped stack decomposes the incident wave by a minimum-norm least
squares on an UNDERDETERMINED Rayleigh system (50 equations, 72 / 162
unknowns at M = 3 / 4, full row rank, rcond 0.19 / 0.29).  A minimum-norm
solution depends on the scaling of the unknowns, i.e. on the arbitrary
per-mode normalisation of `W0`: rescaling the columns by random complex
factors moves R / T by 3.5e-5 (M 3), 4.2e-6 (M 4), 9.9e-8 (M 5) at 0.3 rad,
and by 1e-15 at normal incidence -- identical at `eae470d9`
(`o1_gauge_win.json`, `o1_gauge_pre_win.json`).  Consequence: a 1e-12 change
of a rectangle's width moves R / T by 7.6e-7 at M = 3 (1.6e-7 at M = 4,
2.5e-9 at M = 5; 5e-12 at normal incidence) -- the NumPy FD ladder's h^2 premise
fails there (rung ratios 0.07 .. 15 instead of 11.4), which is why FD(numpy)
cannot judge the twin's oblique rectangle gradient at low M.  The mapped
route (L2 projection, then the modal solve) is gauge-free (twin vs NumPy on
the identity map 8e-15 although their eigensolvers normalise differently).

---

## 4. The events and the poison (`v5_events_M3_{win,wsl}.json`)

| case (reference -> event / control) | traced event | concrete | NumPy at the event |
|---|---|---|---|
| circle bulging past the edge (0.5 -> 0.61 / 0.58) | NaN everywhere, d/dr, d/deps NaN | refused | refused |
| circle within the sliver margin (0.5 -> 0.5995 / 0.59) | **finite** (V-E3-3) | refused | refused |
| fillet below sqrt(2) 1e-3 P (0.05 -> 1.56e-3 / 1.92e-3) | NaN | refused | refused |
| sinusoid within the margin (0.1 -> 0.5995 / 0.55) | **finite** (V-E3-3) | refused | refused |
| sinusoid past the edge (0.62) | NaN | refused | refused |
| circle's arc crossing an enclosing rectangle (0.3 -> 0.42 / 0.38) | NaN (fold) | refused | refused (outlines cross) |
| two circles approaching (walls reorder) | NaN | refused | refused |
| two rectangles reordering | NaN | refused (topology) | accepted (a new topology) |

Poison airtightness: at every NaN event every entry of R, T and J is NaN,
sum(R) + sum(T) is NaN, d T00 / dx, d sum(T) / dx and d T00 / d eps (eps
traced alongside) are NaN.  The poison multiplies the outputs AFTER the
cascade, so no `jnp.where` downstream of it inside the stack can mask it;
a user's own `nan_to_num` / `where` would.  Mutant (the old `jnp.where(ok,
R, nan)` guard): the build's `test_e3_4_a_fold_inside_the_trace_is_nan...`
fails on the fold's gradient (0.0) -- caught.

## 5. F-E3-5 explained (`v8_stripe_ladder_win.json`)

On the stripe `Rect(0.9, 0.6, 0.6, P)` both cells are centred on mirror
planes of the periodic structure (x = 0.3, 0.9).  At normal incidence the
excited E_x and E_y are EVEN functions about those planes.  Raising the
degree by one adds one polynomial per cell whose parity about the cell
centre alternates; when it is odd it is orthogonal to the excited sector and
the answer does not move.  E_y and E_x live in the two staggered sets along x
(the continuous set `Btilde` and its partner `B`, one bubble dropped per
segment), whose added function has opposite parity at a given step, hence
the offset pairing: TE identical for 3/4, 5/6, 7/8, 9/10 (<= 1.1e-13), TM
for 4/5, 6/7, 8/9 (<= 1.9e-14).  Breaking the
mirror symmetry breaks the pairing: oblique incidence in x (0.2 rad): every
step moves (1.9e-2 TE at 5 -> 6); oblique in y (phi = 90 deg, x-mirror kept):
pairs survive to 7e-10.  **For the campaign**: a ladder that steps M by one
on a cell centred on a mirror plane at normal incidence (a centred pillar
is one) shows a zero change every other rung; the circle's own ladder shows
the same pattern softened (value changes 1.3e-2, 6.5e-3, 5.2e-4, 2.1e-4 at
M = 5 .. 8).  Judge convergence on PAIRS of rungs (M -> M + 2), never on one
step.

## 6. Documentation checks

* Numbers: the E3-3 table matches `f3_summary.json` on every cell (scripted
  check, 6 % rounding); 1.2 / 3.4 / 3.6 / 3.7 / F-E3-1 spot-checked against
  `f1_*`, `f8_*`, `f6_jit_M4.json`, `f7_stripe_M7.json`: match.  Exceptions
  listed in V-E3-4 (3, 4, 5).
* The API examples EXECUTE as written (code read from the documents,
  `n_modes = 6`): BUILD 2.5 (three examples, the third an EAGER gradient) in
  86 s (Windows) / 337 s (WSL), CHANGELOG in 27 s / 41 s; dT/dr = -1.9294975
  on both builds, the NumPy FD -1.929494 (`v12_api_examples_*.json`).
* `LUMENAIRY_DISABLE_JAX=1`: both `PMM2DStackPure(backend='jax')` and the
  entry raise `ImportError`, jax never imported.
* mypy (configured list): "Success: no issues found in 33 source files";
  the twin module alone reports only `no-untyped-def` / `no-untyped-call`.
* WSL ruff 0.15.16 on `lumenairy/ tests/ scripts/` and `verify_e3/`: "All
  checks passed!".
* Fingerprints `--check`: "every history document matches its module";
  `stack2d_pure` re-recorded with the E3 reason.  No forward version token
  in the diff.

## 7. Other measurements

* `pmm_jones_2d_staggered(..., backend='jax')` with a TRACED patterned
  `eps_cell` / `mu_cell` (the entry builds its template on uniform stand-ins):
  value = NumPy to 3.8e-15, AD vs FD(numpy) 6.5e-11 / 9.1e-10 -- the stand-ins
  do not leak (`q1_entry_traced_cell_*.json`).
* LC director angle on a homogeneous FILM vs the Berreman JAX oracle (exact
  for a film at normal incidence): Jones 1.7e-15, its gradient 3.7e-15
  relative through the uniform-tensor and eps_cell routes; the film under a
  circle map converges (1.9e-4 -> 3.4e-6 on J, M = 3 -> 4); at conical
  incidence a uniform film carries the basis' own discretisation (4.0e-4 ->
  9.0e-7), NumPy's too.
* 2-parameter jacobian of (R00, T00) w.r.t. (r, eps) vs FD(twin): <= 4.2e-10;
  vs FD(numpy): the eps column 1.4e-11, the r column the frozen-grid offset
  (2.8e-4 at M = 3).

## 8. Ship recommendation

**Do not ship E3 as is.**  V-E3-1 is a silent, build-dependent wrong
gradient in the most common shape-optimisation start (a square pillar's
width, a circle relaxed into an ellipse), 0.3 - 29 % at M = 3 .. 5, on the
first curved-gradient feature of the library.  Ship after (a) a fix of the
eig derivative at degenerate pairs (the measured split prototype, with E3-2's
bar re-derived, or a matrix-function formulation) with the strict-xfail arm
turned into a gate, OR at the very least the V-E3-1 "Known limits" bullet
plus a runtime warning; (b) the V-E3-2 and V-E3-4 documentation edits;
(c) the V-E3-3 guard (small, measured).  Everything else verified: NumPy bytes
untouched, forward parity at round-off, non-degenerate gradients exact to
the FD's own residual, the frozen grid understood and harmless, jit once.

## 9. What the integration must carry

* The 10 decision tests of `test_verify_pmm2d_curved_e3.py` (two strict
  xfails to flip with the fixes), spliced durations.
* The V-E3-1 fix or limit, the V-E3-2 / V-E3-4 doc edits, the V-E3-3 guard.
* For the campaign: judge M-ladders on pairs of rungs (F-E3-5); gradients
  need M >= 7 for 1 % on the circle (section 2.2); never step off a
  symmetric cell by less than 1e-9 when differentiating a symmetry-breaking
  parameter (V-E3-1) until the fix lands.
* O-1 (pre-existing) to the maintainers' queue: the unmapped route's
  gauge-dependent incident decomposition at oblique incidence.

## 10. Not measured

* FD(numpy) truth above M = 7 for the fillet (a 5 x 5 NumPy solve at M = 8
  ran 22 min per point on the loaded box); the fillet's twin at M = 6 (the
  compile exceeded 30 min).
* V-E3-1 on WSL beyond the square pillar; in the library's OTHER JAX twins
  that share `_jax_eig_stable`.
* GPU; idle-box timings.

## 11. Test tails and reproduction

* Windows, half A (`verify_e3/suite_A.txt`: the E3 ids, curved A-D, every
  `verify_pmm2d*` file incl. this verifier's, every `*jax*` file), `-n 8`:
  `328 passed, 8 skipped, 2 xfailed, 41 warnings in 1185.46s` (the two
  xfails are this verifier's strict arms V-E3-1 / V-E3-3).
* Windows, half B (`suite_B.txt`: every other `*pmm2d*` / `*stack2d*` /
  `*stagger*` file, census, public API, walkers, doc-consistency,
  doc-identifier, except budget, re-exports, history lint / relocation /
  fingerprint tool, kernel consistency), `-n 8`:
  `1548 passed, 8 skipped, 88 warnings in 888.44s`.
* WSL (the E3 file, this verifier's file, curved A-D, the PMM-2D / stack /
  disable-JAX / PMM-JAX-guard JAX files), `-n 6`:
  `153 passed, 2 xfailed, 12 warnings in 656.20s`.
* This verifier's file alone (Windows, serial): `8 passed, 2 xfailed in
  164.28s`; on the fix trees the two strict arms XPASS; on the renamed-copy
  mutant the body census fails.
* Mutant suites (the E3 file in a scratch tree each): incident node count
  re-read in the trace `8 failed, 19 passed`; tau = 0 `27 passed`
  (survivor); determinant from the reference map `1 failed, 26 passed`;
  the V-E3-1 split prototype `12 failed, 15 passed` (bars to re-derive).
* mypy (configured): `Success: no issues found in 33 source files`; WSL ruff
  0.15.16 `All checks passed!`; `scripts/record_history_fingerprints.py
  --check`: `OK: every history document matches its module.`

Reproduction (from `validation/probe_pmm2d_curved/verify_e3/`, with
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHONPATH=C:/tmp/lum_vcurved_e3` on the command line): `v1_bytes.py TREE
LABEL [block-twin|block-jax]` in both trees then `v1_compare.py win|wsl`;
`v2a_nq_scan.py`, `v2_frozen.py FAM M twin numpy [nofar] [pt=i]`,
`v2b_nqswitch.py`, `v2_analyse.py win`; `v3_parity.py M`; `v4_grads.py
film|second|jac|ladder M`, `v4b_rect_conical.py M`; `v5_events.py M`;
`v6_degenerate.py M DELTAS PART`, `v6b_symbreak.py M CASE`, `v6c_where.py
M`; `v7_jit.py M`; `v8_stripe.py ladder | grad M`; `v12_api_examples.py`;
`o1_gauge.py` (also with `LUM_TREE` / `PYTHONPATH` on the PRE tree);
`runjobs.sh` / `run_wsl.sh` run job lists; `VE3_TAG` + `LUM_TREE` point a
probe at a scratch tree (mutants / fixes, tagged JSON).
