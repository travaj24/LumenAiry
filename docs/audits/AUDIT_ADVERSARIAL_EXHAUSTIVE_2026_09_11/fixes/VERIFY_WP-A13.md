# VERIFY-A13 — independent re-verification of WP-A13 (PMM 2-D, findings G5–G13)

Verifier: VERIFY-A13. Diff under test: commit `83142777` (parent `72d94fac`), files
`lumenairy/elements/pmm/{twod,twod_jones,stack2d,_jax_twod,_jax_twod_jones,_jax_stack2d,twod_staggered,stack2d_pure}.py`
plus four new test files. Machine: Windows 11, CPython 3.14.6, numpy 2.4.6, scipy 1.17.1,
jax 0.10.1 (CPU, x64), `OPENBLAS_NUM_THREADS=1` on every invocation. Other agents were editing
`pmm/stack.py`, `pmm/__init__.py` and non-PMM files in the working tree throughout (those landed
mid-pass as `49c05569`); the eight PMM 2-D source files were clean at the start of this
verification, and at the end the only uncommitted changes under `lumenairy/elements/pmm/` are the
four of mine listed in §4.

---

## 1. Verdicts

| ID | Verdict | What I re-measured (independently) |
|---|---|---|
| **G5** (P1) | **VERIFIED** | 90° ROTATION and y=x MIRROR covariance of a stack with **two different layers** (one x-patterned, one y-patterned) at **anisotropic periods** and **oblique conical** incidence, on NumPy and on the JAX twin. Worst `li` residual over 48 settings: `dR 4.1e-13 / dT 1.9e-13 / dJ 2.8e-13`. Pre-fix on the same fixture (routing reverted in-process): `dT 1.07e-02 / dJ 6.28e-03` — 9–11 decades. Doubly-patterned branch bit-identical by construction and by `np.array_equal`. |
| **G6** (P2) | **VERIFIED-WITH-NOTES** | Ranking reproduced on the audit's fixture; `'auto'` resolves correctly on six fixtures incl. the "separable except ONE pixel" case; the `fff_nv` fold engages at normal incidence only and agrees with the full solve at 8.1e-15; it correctly does NOT engage at oblique incidence or on a slanted cell. **Note:** the docstring's blanket ordering `fff_nv > laurent > li` is contradicted by the WP's own QWP TRANSMISSION measurement (see §3.2). |
| **G7** (P2) | **VERIFIED-WITH-NOTES** | `pmm_2d_order_drift` fires on my own unconverged fixture (high-contrast Si cross at oblique: `max_drift 4.3e-02` while closure IMPROVES 1.27e-03 → 8.78e-04) and is silent on my own converged one (low-contrast pillar, `max_drift 3.0e-04`, both incidences). The "`_PASSIVE_TOL_2D` tightening would reject legitimate solves" claim reproduces on my own L-cell (5.93e-03 / 3.77e-03 — both rejected by a 3e-3 gate). **Note:** the helper has no guard against a caller that changes more than `n_orders` (§3.3). |
| **G8** (P2) | **VERIFIED** | Prepared and sweep vs direct **bit-identical** (`0.000e+00`) on an **off-centre crossed cell at anisotropic periods** at `symmetry` ∈ {auto, False} × `truncation` ∈ {rectangular, circular} × normal and θ=19°/φ=53° oblique, on **both** entry families. Audit `q11_prepared.py` / `q13_warn.py` reproduce the report's numbers exactly. |
| **G9** (P2) | **VERIFIED-WITH-NOTES; 3 defects FIXED, 1 reported** | The gcd rule is right on every fixture I built (12 segments with walls at **5 and 7** → `n_min = 12`, silent; walls at 3,9 → 4; asymmetric x/y walls → joint gcd 1, silent). A genuinely minimal but large grid **raises** (30×30 at M=8, dof 88 200). The guard runs **no** eig/QZ/SVD/QR. **But**: `max_pencil_dof` was dropped on the magnetic branch, that branch mis-priced per-layer `n_modes`, and a mismatched eps/mu pair died on an internal `IndexError` — all three fixed here (§4 V1–V3). A fourth, reported not changed: on `layer_grids='shared'` the redundancy advice cannot be followed, and the shipped suite is **not** silent (§4 V5). |
| **G10** (P2) | **VERIFIED-WITH-NOTES; 1 defect found and FIXED** | jnp sandwich vs NumPy `_sandwich_factorized` on a **non-square** (Nx=77, Ny=35) cell: 7.8e-17; bit-identical under `jax.jit`; frozen `Gx0F/Gy0F/IprojF` bit-identical to `_scalar_projected_ops`. End-to-end parity rel 3e-14; `jax.grad` finite, FD-agreeing to 6.9e-08; `jit(grad)` bit-identical to eager. Circular truncation correct on the out-of-plane blocks (uniform-OOP decoupling oracle: 1.3e-14). **But**: the eig-cache refusal warning fired **6 times on a 6-point `solve_vs_wavelength`**, not once — the contract the WP claims and pins. Fixed here (§4). |
| **G11** (P3) | **VERIFIED** (WP-A14's fix; PMM-2D-side contract only) | `_sqrt_decay([1e-20 − 1e-30j]) = +1.0e-10 − 5.0e-21j` — no longer flipped; the on-cut root still resolves near the imaginary axis. |
| **G12** (P3) | **VERIFIED** | `symmetry` and `truncation` are in `_mode_key`; the periods are read-only after the first `add_layer` on the object **and on a `copy.copy` clone**; `solve_vs_wavelength` still works. `pickle` / `deepcopy` of a stack fail on the `LayerCache` lock — **pre-existing** (`LayerCache` was constructed in `__init__` at the parent commit too), not a WP-A13 regression. |
| **G13** (P3) | **VERIFIED** | `return_jones_transmission=True` is **bit-identical** to `PMM2DStackHybrid.jones_transmission()` for an **out-of-plane tensor layer** (0.000e+00, normal and oblique) and a **slanted** cell (0.000e+00 / ≤5.6e-17), and ≤2.8e-14 for an in-plane tensor layer. QWP retardance against **my own no-floor oracle** (`PMM2DStackPure`, Granet staggered): +99.971 / +100.033 / +100.051° at M=5/6/8, converging on `rcwa_jones_1d`'s +100.066°; the entry reads +100.12 / +100.23 / +99.90° at n_orders 5/9/15 — inside 0.17°. |

**Collateral damage: one test, reconciled.** `test_verify_pmmstack_sliver_walls.py::test_the_pure_stacks_shared_grid_cannot_express_a_sliver`
hit the new `max_pencil_dof` refusal; I measured the guard to be RIGHT (129×129 segments at M = 4 is
a 299 538² pencil, ~15.7 TB) and fixed the test, which never solves (§4 V5). Everything else passes
(§5).

---

## 2. What I did NOT take on trust

Every number in the WP's report that I re-measured came out the same, with three exceptions worth
naming:

1. The commit message says the audit's 498 s / 8.4 GB call "now warns **instead of running**".
   Measured: that call is `dof = 7200`, **below** the 12 000 default cap, so it warns **and still
   runs**. The report §2.5 and the changelog are careful and correct about this; only the commit
   message overstates it. (Orchestrator: the commit message goes into history — worth a note.)
2. The report §2.9 claims the transmission Jones is bit-identical to the stack accessor "for both
   the vertical and the `slant=(0.4, 0)` case". For an **in-plane tensor** cell at **normal**
   incidence I measure 2.8e-14, not 0.0 — because `pmm_jones_2d` takes `_symmetric_cascade_rt` there
   while the stack takes `_tensor_layer_modes(block_eig=)`, two different recursions. Bit-identity
   does hold on every other combination I tried (5 of 8). The claim is right in kind, loose in
   degree.
3. The report §2.5 says "the shipped staggered suite is SILENT" after the gcd rule. Measured: **6
   cost-guard warnings remain**, all in `tests/unit/test_pmm2d_staggered_mortar.py` — a file the WP
   re-ran only in its earlier "remainder" group, before the gcd rule landed. The warnings are not
   spurious arithmetic, but their advice is unactionable there (§4 V5 half 2).

---

## 3. Per finding

### 3.1 G5 — per-slot Li routing — VERIFIED

**The WP's own gate re-run.** `repro/PMM-2D/p3b.py`: `li` `max|ΔT|` **3.385e-14** (`symmetry='auto'`)
/ **1.139e-13** (`False`), `|ΔJ|` 7.805e-14 / 2.337e-13, against `laurent`'s unchanged 1.070e-14 —
the report's numbers to the digit. `q1b_stackli.py`: x-pat and y-pat T00 identical at every
`n_orders` 3/5/7/9 for both formulations. `q1c.py`: all four columns read 0.02319196 (n=5) and
0.03347366 (n=9), so the one-layer stack equals `pmm_efficiency_2d_cell`.

**My own oracle — a stack the WP did not use.** Two DIFFERENT layers (layer A a 3-pixel x-patterned
eps-12.25 grating, layer B a 5-pixel y-patterned eps-4 grating), **anisotropic** periods
Px = 0.9 µm ≠ Py = 0.75 µm, λ = 1 µm, n_sub = 1.45, degree 7. Two exact symmetries of Maxwell's
equations, neither of them produced by the library:

* **rotation** R:(x,y)→(−y,x): `cell' = cell[:, ::-1].T`, `(Px',Py') = (Py,Px)`, **φ' = φ + 90°**,
  orders (m,n) → (−n, m), `J' = R₂ J R₂ᵀ`, and the R/T **rows swap** (they are keyed to incident lab
  E_x / E_y, which rotate with the structure);
* **mirror** M:(x,y)→(y,x): `cell' = cell.T`, φ' = 90° − φ, (m,n) → (n,m), `J' = σ J σ`, rows swap.

Worst residuals over `{rot, mir} × {li, laurent} × symmetry {auto, False} × (θ,φ) ∈
{(0,0), (22°,37°), (35°,0)} × n_orders {3,5}` — 48 paired solves:

| formulation | worst dR | worst dT | worst dJ |
|---|---|---|---|
| **li** | 4.11e-13 | 1.90e-13 | 2.77e-13 |
| laurent (control, untouched by the WP) | 1.16e-12 | 3.68e-12 | 5.64e-12 |

against `max|T| ≈ 0.34–0.51` and `|J| ≈ 0.5–0.87`. `laurent` being the LOOSER of the two is the
fixture's own floor at oblique incidence, not a `li` defect.

**JAX twin.** Same stack with a traced tail thickness (so the concrete y-patterned layer runs the
twin's concrete-scalar branch, which is where the routing fix lives): worst over
`{rot, mir} × {li, laurent} × {(0,0), (22°,37°)}` is `dR 1.74e-13 / dT 3.92e-13 / dJ 5.37e-13`
(all three from the `laurent` control at oblique; `li` alone is
`5.94e-14 / 1.43e-14 / 4.23e-14`). NumPy↔JAX parity on the identical stack:
`1.09e-14 / 1.68e-14 / 1.51e-14`.

**Fails pre-fix.** Reverting the routing in-process (`_scalar_projected_ops` wrapped to return the
legacy `EpnxF = EpnF, EpnyF = EpsF`) on the same two-layer fixture at θ=22°/φ=37°, n_orders 5:

```
POST-FIX sym=auto  li  : dR=1.019e-14 dT=1.497e-14 dJ=1.174e-14
PRE-FIX  sym=auto  li  : dR=4.055e-03 dT=1.070e-02 dJ=6.280e-03
POST/PRE control laurent: dR=1.156e-12 dT=2.391e-12 dJ=4.747e-12   (identical both ways)
```

**The doubly-patterned branch is byte-identical, structurally.** On the crossed branch
`_scalar_projected_ops` returns `EpnxF = EpnF` and `EpnyF = EpsF` — `np.array_equal` True for both —
so the new `EPS_nx, EPS_ny = lops["EpnxF"], lops["EpnyF"]` IS the legacy `(EpnF, EpsF)` assignment,
bit for bit. Same on the x-patterned branch (`EpnxF == EpnF`, `EpnyF == EpsF`); only the
y-patterned branch differs, which is the defect. The JAX twin's TRACED branch keeps the legacy
assignment, and that is sound: `_layer_static_traced` **raises** for an axis-uniform traced cell
("needs walls on BOTH axes"), so a traced separable cell cannot reach it — no NumPy/JAX divergence
is introduced.

**Residual risk.** None found. One pre-existing, unrelated observation: `_axis_elem_counts` breaks an
EXACT strip-width tie with `argmax` (first strip), and the axis reversal inside a 90° rotation maps
that onto the other strip — so a 50 %-duty grating with an even pixel count is not rotation-covariant
to machine precision *by discretisation*, independently of G5. It is invisible to the transpose
(mirror) form the audit and the WP's tests use. Not a WP-A13 issue; noted for the record.

### 3.2 G6 — `formulation` docs, `'auto'`, the `fff_nv` fold — VERIFIED-WITH-NOTES

`repro/PMM-2D/q3_fffnv.py` re-run: at n_orders 11, `|Jxx − rcwa_jones_1d(80,'li')|` =
**2.72e-03** (`fff_nv`) / **3.02e-02** (`laurent`) / **6.90e-02** (`li`); the crossed-cell raise and
the `rcwa_jones_2d` contrast reproduce. `q1_conv.py` re-run: `li` keeps `Im(Jxx)` POSITIVE at every
truncation (+0.4027 / +0.2243 / +0.0823 at n = 3/7/11) against the converged RCWA-li −0.0725, i.e.
`arg(Jxx)` 8.9° out at n=11 — the docstring's claim, reproduced.

**`'auto'` resolution** (spy on `_tensor_layer_modes`'s `formulation` argument):

| cell | strips (nsx, nsy) | `'auto'` → |
|---|---|---|
| separable x-patterned | (3, 1) | `fff_nv` |
| separable y-patterned | (1, 3) | `fff_nv` |
| separable ANISOTROPIC (off-diagonal in-plane) | (3, 1) | `fff_nv` |
| **separable EXCEPT one differing pixel** | (4, 3) | `laurent` |
| UNIFORM (no walls) | (1, 1) | `laurent` |
| separable but OUT-OF-PLANE | (3, 1) | `laurent` |

The one-pixel cell is the case the brief asked about: `_cell_to_walls_tile` puts a wall on **both**
axes, `any(wall_less)` is False, and `'auto'` falls back — while `formulation='fff_nv'` on the same
cell raises "requires a SEPARABLE …". `'auto'` never resolves to a rule that would raise.

**The fold.** `return_ops=True` is requested only at `kt < 1e-12`; at θ=15° with φ=0 and φ=40° it is
**not requested at all**, for `fff_nv`, `laurent` and `li` alike — the oblique gate holds. A SLANTED
cell at normal incidence requests ops and `_tensor_layer_modes` returns `None`, so the fold declines
— the "silently returns the vertical answer" class stays closed for `fff_nv` too (measured: slanted
`fff_nv` with `symmetry=True` differs from the vertical answer by **4.16e-03**, and from
`symmetry=False` by 2.10e-14). Fold-vs-full on my own separable cell (degree 9, n_orders 5):
`fff_nv` **8.13e-15**, `laurent` 7.55e-15, `li` 2.14e-14 — same decade, none bit-identical (so the
fold really ran).

**Note (documentation, not physics).** The `formulation` docstring states a blanket "Order of
preference on this entry: `'fff_nv'` … then `'laurent'`, then `'li'`". That ordering is measured on
the REFLECTION Jones of one high-contrast stripe. On the WP's own QWP fixture and the TRANSMISSION
retardance — the observable G13 exists for, and the one the LC-QWP project needs — the order is
different:

| n_orders | `fff_nv` | `laurent` | `li` | (reference +100.066°) |
|---|---|---|---|---|
| 5 | +100.122 | +94.692 | +96.729 | |
| 9 | +100.233 | +98.937 | **+100.036** | |
| 15 | +99.904 | +99.633 | **+100.039** | |

`li` is the best of the three at n_orders ≥ 9 here. The WP's §2.2 discloses this; the docstring does
not, and a QWP designer reads the docstring. Suggested one-line amendment in §6.

### 3.3 G7 — the drift check and the tolerance — VERIFIED-WITH-NOTES

**My own fixtures** (the WP used the audit's 12×12 half-fill pillar; these are a 10×10 Si CROSS and a
10×10 low-contrast pillar, Px = Py = 0.75 µm, d = 0.42 µm, degree 9, `n_orders` 9 vs 7):

| fixture | converged | max_drift | drift_00 | closure | closure_prev | warned |
|---|---|---|---|---|---|---|
| Si cross, normal | True | 8.29e-03 | 8.29e-03 | 4.34e-04 | 8.25e-04 | 0 |
| **Si cross, θ=20° φ=35°** | **False** | **4.30e-02** | 4.30e-02 | **8.78e-04** | 1.27e-03 | **1** |
| low-contrast pillar, normal | True | 3.02e-04 | 3.02e-04 | 1.57e-04 | 1.36e-04 | 0 |
| low-contrast pillar, θ=20° φ=35° | True | 4.35e-04 | 4.35e-04 | 3.12e-04 | 2.43e-04 | 0 |

The oblique cross reproduces the anti-correlation on a fixture the WP never touched: closure
IMPROVES (1.27e-03 → 8.78e-04) while the per-order split moves 4.3e-02. The helper is a signal, not
noise — it is silent on both low-contrast arms. It also accepts every return shape I threw at it
(`pmm_efficiency_2d_cell`'s `Efficiency2D`, `pmm_jones_2d`'s 4-tuple AND its new 5-tuple,
`PMM2DStackHybrid.solve`'s 4-tuple) and its `step` validation raises with the §2 prefix.

**The "tightening would reject legitimate solves" claim — reproduced on my own L-cell.** A 12×12
L-shaped chiral cell (eps 12.25 in air, Px = Py = 0.9 µm, d = 0.3 µm, degree 11), provably lossless:

| n_orders | ΣR+ΣT | \|1−tot\| | default 5e-2 gate | a 3e-3 gate |
|---|---|---|---|---|
| 5 | 0.9940660 | 5.93e-03 | silent | **would reject** |
| 9 | 1.0037741 | 3.77e-03 | silent | **would reject** |

`stabilize=True` accepts that same n=5 solve today (it shares `_PASSIVE_TOL_2D` as its
`passive_tol`), so tightening the constant would turn a working consensus into a refusal. The WP's
decision to add `_ADVISORY_TOL_2D` + `energy_tol=` instead of tightening is correct and now measured
twice. `energy_tol=_ADVISORY_TOL_2D` fires on that call (1 warning); `energy_tol=np.inf` silences it.

**The order-set intersection.** The helper compares on the `(m, n)` pairs both solves retain, which
is the part most likely to be subtly wrong when the two order sets differ in SIZE. Probed on an
anisotropic-period Si cross at oblique incidence, `n_orders = 9` against `step` 1 / 2 / 3, on both
truncations — retained counts 361 vs 289 / 225 / 169 (rectangular) and 175 vs 139 / 107 / 81
(circular): the drift is monotone in `step` where it should be, `drift_00 <= max_drift` on every
arm, `converged` flips False at `step >= 2` on this fixture and the warning fires exactly there.
The circular set — which is not a prefix of the rectangular one — is handled correctly.

**Note.** `pmm_2d_order_drift` compares whatever the callable returns at two truncations and has no
way to notice that the callable changed something else. Fed a deliberately mischievous closure that
also changes `period_x`, it silently reports `max_drift = 7.20e-03` for two different physical
problems. The docstring says "the SAME call with only n_orders changed" and the function raises when
the two order sets are disjoint, which catches the gross case; a cheap hardening would be to compare
`len(orders)` growth monotonicity. Advisory only — it cannot produce a wrong solve, only a
meaningless drift number.

### 3.4 G8 — `PreparedPMM2D` — VERIFIED

`repro/PMM-2D/q11_prepared.py`: prepared vs `pmm_efficiency_2d(symmetry='auto')`
**`max|dR| = max|dT| = 0.000e+00`** (pre-fix 6.33e-14 / 1.08e-13), vs `symmetry=False` now 6.33e-14 /
1.08e-13 — the roles have swapped exactly as a now-folding prepared path should. The sweep helper
accepts `truncation='circular'`. `repro/PMM-2D/q13_warn.py`: `E = 1.054125` on both paths and
"prepared warns too".

**My own fixture** — an **off-centre crossed** 10×10 cell (eps 12.25 at rows 2:7, cols 3:8) at
**anisotropic** periods Px = 0.9 µm, Py = 0.63 µm, degree 9, n_orders 4, `formulation='li'`:

| truncation | symmetry | (θ, φ) | Nf | prepared bit-identical | sweep bit-identical |
|---|---|---|---|---|---|
| rectangular | auto / False | (0,0) and (19°,53°) | 81 | **True** (max\|d\| 0.000e+00) | **True** |
| circular | auto / False | (0,0) and (19°,53°) | **33** | **True** (0.000e+00) | **True** |

and the same on the pillar entry (`pmm_efficiency_2d` / `prepare_pmm_2d` /
`pmm_efficiency_2d_vs_wavelength`, θ=25°/φ=15°, both truncations): bit-identical on all arms.
The off-centre cell matters: it is the case `_symmetric_solve_2d`'s recentring gauge has to handle,
and the prepared path reproduces it exactly. The UNIFORM-layer branch (where `PreparedPMM2D.solve`
must NOT fold, matching `_pmm2d_solve_core`) is also bit-identical at both truncations and both
incidences — the fold gate was threaded without disturbing that shortcut.

### 3.5 G9 — the staggered SEGMENT-grid guard — VERIFIED-WITH-NOTES, 3 defects fixed + 1 reported

**The gcd rule, on fixtures of my own:**

| cell | merged strips | minimal uniform lattice | redundant? |
|---|---|---|---|
| 12×12, walls at **3 and 9** (the audit's) | (3,3) | **4** | yes → warns, names the 4×4 lattice |
| 12×12, walls at **5 and 7** | (3,3) | **12** | **no → silent** |
| 12×12, walls at 4 and 8 | (3,3) | 3 | yes |
| 12×12, x-walls 3,9 / y-walls 4,8 (joint gcd 1) | (3,3) | **12** | **no → silent** |
| 4×4 walls 1,3 | (3,3) | 4 | no |
| 6×6 uniform | (1,1) | 1 | exempt (1×1 → uniform) |

The brief's exact case — 12 segments with walls at non-divisor indices 5 and 7 — is silent, which is
the correct answer: `gcd(12,5,7) = 1`, nothing is reducible. The asymmetric case shows the gcd is
taken over the JOINT wall set of both axes, as the square-grid contract requires.

**Minimal-but-large must RAISE, not warn.** 30×30 with walls at 7 and 23 (gcd 1, so not redundant)
at M = 8 → pencil 88 200 → `ValueError` naming `max_pencil_dof=12000` and a projected ~1 391 GB.
Zero warnings, as designed: a redundancy test cannot bound this and the absolute cap must.

**The guard runs no eigensolve.** With `numpy.linalg.{eig,eigvals,eigh,svd,qr}` and
`scipy.linalg.{eig,eigvals,qz,schur,svd}` all trapped, `_validate_stag_cost` plus a full
`PMM2DStackPure.add_layer` on the 12×12 cell called **none** of them.

**A magnetic layer reaches the guard** — but with three defects, now fixed (§4): `max_pencil_dof`
was dropped, the layer's own `n_modes` was ignored, and a mismatched eps/mu pair died on an internal
`IndexError`.

**Calibration note (not a defect).** At the DEFAULT cap the audit's own headline call —
`pmm_efficiency_2d_staggered(12×12, degree=6)`, 498 s / 8.4 GB, dof 7200 — **warns and still runs**.
That is the WP's deliberate two-tier design and the changelog says so correctly; the commit message
does not (§2).

### 3.6 G10 — the JAX factorisation, circular truncation, the refusal signal

**(a) Factorised projectors, NON-SQUARE cell.** 8×6 pixel cell → 4 x-strips × 3 y-strips with
`elements_per_strip` 2 on x and 1 on y → nodal **Nx = 77, Ny = 35**, orders 9×9:

| check | result |
|---|---|
| `_proj_sandwich_jnp` vs NumPy `_sandwich_factorized` | max\|d\| **7.76e-17** (\|ref\| ≈ 0.114) |
| … vs the deleted dense `kron(Ty,Tx)` spelling | 9.31e-17 |
| under `jax.jit` vs eager | **0.000e+00, bit-identical** |
| frozen `Gx0F` / `Gy0F` / `IprojF` vs `_scalar_projected_ops` | **bit-identical (all three)** |

**(a′) End-to-end, same non-square cell, oblique (θ=0.2, φ=0.4).** NumPy↔JAX `pmm_jones_2d`:
`R` 4.9e-15 (rel 3.1e-14), `T` 1.7e-14 (rel 2.8e-14), `J` 1.6e-14 (rel 7.0e-14) — the audit's
pre-fix `T` gap was 8.47e-11, so the parity improvement is real and reproduces.
`jax.grad d(ΣT)/d(eps)` = −1.81578773e-02, central FD −1.81578760e-02, rel **6.87e-08** against the
1e-4 gate; `jit(grad)` **bit-identical** to eager grad. The JAX path refuses `truncation='circular'`
and `return_jones_transmission=True` loudly.

**(c) Circular truncation on the out-of-plane blocks.**

* **Analytic oracle:** a UNIFORM out-of-plane tensor cell decouples every order exactly, so circular
  must reproduce rectangular on the retained set. Measured (Px=0.9, Py=0.72 µm, n_orders 4,
  81 → 43 orders): `max|dT|` **1.34e-14** (normal) / 1.92e-14 (θ=17°, φ=29°), `max|dJ|` ≤ 3.9e-15.
* The seven in-plane operators returned by `_tensor_layer_modes(keep=…)` are **`np.array_equal`** to
  the full-box build restricted by `np.ix_(keep, keep)` — the "assemble on the box, restrict at use"
  contract, checked operator by operator.
* The parity-sign **block_eig gauge** (`orders2d[keep]`, `kxv[keep]`) is correct under circular: the
  block reduction agrees with the dense 4Nf zgeev to **1.33e-14** at normal incidence and is
  bit-identical at oblique (where the gauge declines).
* The `pmm_jones_2d` → `pmm_efficiency_2d_cell` **exact scalar reduction** survives the circular set
  through two independent assemblies (`_tensor_layer_modes` vs `_scalar_projected_ops`):
  `max|dT| ≤ 2.07e-14` at normal and θ=0.3 rad for both `tm` and `te`.
* Cross-knob: circular × `symmetry` {auto, False} × `cascade` {fast, fused, tree} on a 3-layer stack
  all agree to ≤ **7e-16**, and `truncation='rectangular'` with `cascade='fast'` is bit-identical to
  itself across `symmetry` — the default path is untouched.

**(d) The eig-cache refusal signal — DEFECT, fixed (§4).** Pre-fix measurement:

```
solve_vs_wavelength over 6 wl, max_workers=1: 6 refusal warning(s)  parent._eig_refusal_warned=False
solve_vs_wavelength over 6 wl, max_workers=3: 6 refusal warning(s)  parent._eig_refusal_warned=False
4 repeated solve() on ONE instance:           1 refusal warning
```

The once-per-instance contract held only on the path the WP tested. `solve_vs_wavelength` hands each
wavelength a `copy.copy` clone that SHARES the cache object but gets its own copy of the plain-bool
flag — so a wide sweep, the exact case the warning describes ("a sweep loses its modal reuse
silently from here on"), emits one warning per point.

### 3.7 G11 / G12 / G13

**G11.** `_sqrt_decay(np.array([1e-20 − 1e-30j]))` → `+1.0e-10 − 5.0e-21j` (`Re ≥ 0`: the evanescent
mode is no longer flipped); `_sqrt_decay([−1e-20 − 1e-30j])` → `−5.0e-21 + 1.0e-10j`, still on the
cut. WP-A14's predicate is in and the PMM-2D-side test asserts the right contract.

**G12.** `symmetry` and `truncation` are in `_mode_key`'s `common` — and the `truncation` half is
live, not decorative: mutating `st.truncation` from `'rectangular'` to `'circular'` between solves
on an oblique 8×8 pillar re-solves at 49 orders instead of serving the stale 81, and the answer is
**bit-identical (`0.000e+00` on the Jones)** to a FRESH object at the new setting. The key tail
reads `(False, 'rectangular')` vs `(False, 'circular')`. Period assignment raises on the
object AND on a `copy.copy` clone, leaving the value unchanged; a fresh object at the new period
works; `solve_vs_wavelength` (which clones) is unaffected. `pickle.dumps` and `copy.deepcopy` of a
`PMM2DStackHybrid` raise `TypeError: cannot pickle '_thread.lock' object` — that is the `LayerCache`
lock and it is **pre-existing** (`git show 72d94fac:…/stack2d.py` constructs `LayerCache` in
`__init__` too), so the read-only property introduces no new serialisation limitation.

**G13.** `pmm_jones_2d(return_jones_transmission=True)` vs `PMM2DStackHybrid.jones_transmission()`
on the SAME cell:

| cell | (θ, φ) | max\|dJt\| | bit-identical |
|---|---|---|---|
| in-plane anisotropic tensor (optic axis at 30°) | (0, 0) | 2.80e-14 | no (two different recursions) |
| in-plane anisotropic tensor | (21°, 41°) | **0.000e+00** | yes |
| **OUT-OF-PLANE** tensor | (0, 0) and (21°, 41°) | **0.000e+00** | yes |
| **SLANT (0.4, 0)** | (0, 0) | **0.000e+00** | yes |
| SLANT (0.4, 0) | (18°, 33°) | 5.55e-17 | no |
| SLANT (0.25, −0.15) | (0, 0) / (18°, 33°) | 0.000e+00 / 2.78e-17 | yes / no |

and the frame anchor is **not** a no-op on that cell (`slant(0.4)` vs vertical: `max|dJt| = 2.79e-02`).

**QWP retardance against an oracle the WP did not use.** I took the **no-floor staggered engine**
(`PMM2DStackPure`, Granet modified-Legendre — a different discretisation with no Fourier floor) on
the same Λ/λ = 0.2 Si/air grating, duty 0.5, d = 208.142 nm:

| oracle | retardance |
|---|---|
| `PMM2DStackPure` M = 5 / 6 / 8 | +99.971 / +100.033 / **+100.051°** |
| `rcwa_jones_1d` n_orders 40 / 60 / 80 | +100.06608 / +100.06596 / +100.06595° |
| **`pmm_jones_2d(fff_nv, return_jones_transmission=True)`** n = 5 / 9 / 15 | +100.122 / +100.233 / **+99.904°** |

Two independent engines put the truth at +100.05…+100.07°; the new return is inside **0.17°** at
every truncation, POSITIVE on the SLOW axis, with `|Jxy| ≤ 3.4e-13`. `stabilize=True` +
`return_jones_transmission=True` returns a finite 5-tuple whose first four elements are
`np.array_equal` to the 4-tuple form.

### 3.8 Trying to break it — adversarial inputs and knob interactions

| probe | result |
|---|---|
| `eps_cell` as REAL float64 / **float32** / **F-ordered** / a transposed VIEW, through `pmm_efficiency_2d_cell(truncation='circular')` | all four **bit-identical** (`0.000e+00`) to the complex C-order copy |
| `PMM2DStackHybrid(truncation='circ')` and `truncation=None` | `ValueError` with the §2 `fn_name:` prefix |
| `pmm_jones_2d(formulation='AUTO')` (wrong case) and `truncation='circ'` | `ValueError`, both prefixed |
| `pmm_2d_order_drift(step=0)` and `step=n_orders` | `ValueError`, both prefixed |
| `pmm_2d_order_drift` fed an `Efficiency2D`, a 4-tuple, the NEW 5-tuple, and `PMM2DStackHybrid.solve`'s 4-tuple | all four accepted, consistent drift numbers |
| `stabilize=True` + `return_jones_transmission=True` | returns a finite 5-tuple; its first four elements are `np.array_equal` to the 4-tuple form (the identity-matching fallback never fires) |
| 3-layer stack × `truncation` {rect, circ} × `symmetry` {auto, False} × `cascade` {fast, fused, tree} | all ≤ **7e-16**; `fast` bit-identical to itself across `symmetry` |
| circular truncation on a MIXED (scalar + in-plane tensor, and scalar + out-of-plane tensor) multilayer at oblique incidence | `dT` 1.1e-02 / 9.0e-03 vs rectangular — i.e. the truncation difference itself, with closure identical (3.2e-02 both) at `n_orders = 4`; no instability, no NaN |
| `PMM2DStackHybrid` period assignment on a `copy.copy` clone | refuses, value unchanged |
| dispersive (callable-`eps_cell`) sweep, and serial vs 3-worker `solve_vs_wavelength` | works; **bit-identical** across worker counts on `T` and the Jones |

---

## 4. Defects found in WP-A13's files, and fixed here

Six items: **V1–V4** found here, **V5 half 1** handed over by the coordinator from VERIFY-A12's
collateral report, **V6** handed over from VERIFY-A14 §6 V8. All are guard/signal gaps that move no
physical number — each disarms or omits something a sibling surface already provides. Five are fixed
(V1–V4, V6, plus the V5-half-1 test); **V5 half 2** is a design question I measured and deliberately
left to the orchestrator. Fixes are in files WP-A13 owns, except the one test named below.

Regression tests: `tests/unit/test_audit2609_a13_verify_guards.py` — **12 tests, 5.4 s**. Ten of the
twelve fail on the pre-fix code; the two that do not are the `magnetic=False` arms of the V1/V2
parametrisations, which pin the PLAIN branch that was already correct and are there precisely as the
side-by-side control that isolates the magnetic branch as the defect.

### V1 (moderate) — `max_pencil_dof=` silently dropped on the magnetic branch

`PMM2DStackPure.add_layer` accepted `max_pencil_dof` and then called
`self._add_magnetic_layer(t, eps, eps_cell, mu, mu_cell)` without it, so the magnetic branch always
used the module default. Measured before: a 30×30 cell at M = 8 (pencil 88 200) with
`max_pencil_dof=1e7` **still raised** "above max_pencil_dof=12000" on the magnetic branch while the
identical nonmagnetic call was accepted. `pmm_jones_2d_staggered(mu=…)` forwards the keyword into
that same dead end. Fixed by threading `max_pencil_dof` through `_add_magnetic_layer`.
After: magnetic + `max_pencil_dof=1e7` accepted, magnetic at the default cap still raises.

End-to-end through the PUBLIC entry, `pmm_jones_2d_staggered(30×30 tensor cell, mu_cell=…, M=8)`:
at the default cap it raises `PMM2DStackPure.add_layer: … above max_pencil_dof=12000`, and with
`max_pencil_dof=1e7` it proceeds to a real `MemoryError: Unable to allocate 29.0 GiB for an array
with shape (44100, 44100)` — the same behaviour the nonmagnetic entry already had, and a direct
demonstration that the cap is the only thing standing between a user and that allocation.

### V2 (minor) — the magnetic branch priced the stack's `M`, not the layer's `n_modes`

`_validate_stag_cost(fn, int(self.M), …)` instead of the plain branch's
`int(_pl.get("M") or self.M)`. Measured before (`layer_grids='per-layer'`, stack `M = 3`, layer
`n_modes = 8`, 30×30 cell): the plain branch raised (true pencil 88 200) and the magnetic branch
**accepted**, having priced 7 200 — an under-pricing of `(7/2)³ ≈ 43×` in QZ time and `(7/2)² ≈ 12×`
in memory. Fixed by forwarding the per-layer `M`.

### V3 (minor) — a mismatched `eps_cell` / `mu_cell` pair lost its error message

The new cost guard runs before the union-grid check and merged the two arrays jointly, so a
6×6 eps with a 4×4 mu died inside `_stag_wall_indices` on
`IndexError: index 5 is out of bounds for axis 0 with size 4`. Confirmed a regression by removing
the guard in-process: pre-fix the same call gave
`ValueError: … all patterned layers must share ONE common (Nx, Ny) grid …`. Fixed in two places —
`_stag_wall_indices` drops arrays whose leading `(Nx, Ny)` differs from the first, and the magnetic
branch prices only the first when the two disagree, so the caller's message survives.

### V4 (moderate) — the eig-cache refusal warning fired once per SWEEP POINT

Described in §3.6(d). Fixed by making `_eig_refusal_warned` a shared one-element **list** (rebound
in `add_layer` exactly as the caches are, so fresh caches re-arm the signal), with a defensive
upgrade path for a legacy bool. After:

```
6-wl sweep, max_workers=1: 1 warning   refused=6   flag=[True]
6-wl sweep, max_workers=3: 1 warning   refused=6   flag=[True]
4 repeated solve() on ONE instance: 1 warning
after add_layer (fresh caches):     1 warning
refuse-never-degrade answer identical: True (np.array_equal)
```

Files I changed: `lumenairy/elements/pmm/stack2d.py` (`add_layer`, `_warn_eig_cache_refusals`),
`lumenairy/elements/pmm/stack2d_pure.py` (`add_layer` call site, `_add_magnetic_layer` signature +
guard call), `lumenairy/elements/pmm/twod_staggered.py` (`_stag_wall_indices`, and `wl_eff` on
`pmm_efficiency_2d_staggered`'s result), `lumenairy/elements/pmm/twod.py` (`wl_eff` on the four
NumPy `Efficiency2D` sites),
`tests/unit/test_verify_pmmstack_sliver_walls.py` (one test, V5 half 1 — outside my module
ownership, edited at the coordinator's direction because the failure is WP-A13's guard).
New file: `tests/unit/test_audit2609_a13_verify_guards.py`.

---

### V5 (low–moderate) — the redundancy advice is UNACTIONABLE on the shared grid — **FIXED in §8.1**

Raised by the coordinator via VERIFY-A12's collateral report, and then found to have a second half.

**Half 1 — `test_verify_pmmstack_sliver_walls.py::test_the_pure_stacks_shared_grid_cannot_express_a_sliver`
failed on the new cap. The guard is RIGHT; the test was fixed.** Measured:

| N | M | pencil `2·(N·(M−1))²` | walls | `gcd(N, walls)` → `n_min` | decision |
|---|---|---|---|---|---|
| 8 | 4 | 1 152 | 2, 6 | 2 → 4 | redundancy WARN |
| 24 | 4 | 10 368 | 6, 18 | 6 → 4 | redundancy WARN |
| **129** | 4 | **299 538** | 32, 96 | **1 → 129** | **RAISE** (~15.7 TB dense) |

The DOF estimate is exact, not spurious, and at N = 129 nothing could ever solve it. The test is
GEOMETRIC — its only assertion is `w.max()/w.min() == 1.0` on a locally built width array — and it
never solves, so I fixed it on the test side: `add_layer(..., max_pencil_dof=2*(N*(4-1))**2)`, the
documented acknowledgement, at exactly the fixture's own DOF, with the warnings suppressed and a
comment recording why. `tests/unit/test_verify_pmmstack_sliver_walls.py` now passes **15/15**
(41.5 s), including the three tests that drive the 2-D stacks.

**Half 2 — on `layer_grids='shared'` (the DEFAULT) the redundancy advice cannot be followed.**
Running the staggered suite with the warning treated as significant shows **6 remaining cost-guard
warnings**, all in `tests/unit/test_pmm2d_staggered_mortar.py` (lines 219, 220, 456, 457, 514 and
one more) — a file the WP re-ran **before** the gcd rule landed, so its report's "the shipped
staggered suite is SILENT" is not accurate. Those are the tests' SHARED-union-grid arms, where a
2×2 and a 3×3 pillar are deliberately tiled onto one common 6×6 lattice because the shared contract
requires it. MEASURED, on a two-layer shared stack:

```
add_layer A (6x6, reduces to 2x2): 1 warning -- "expressible on the uniform 2x2 lattice"
add_layer B (6x6, reduces to 3x3): 1 warning -- "expressible on the uniform 3x3 lattice"
follow the advice for A (pass 2x2), then add B at 6x6 -> ValueError "must share ONE common grid"
... and at the advised 3x3                            -> ValueError "must share ONE common grid"
```

The joint minimal shared lattice is `lcm(2, 3) = 6` — exactly what the caller already passed. So the
guard's suggestion is per-LAYER while the constraint is per-STACK, and on the default `shared` path
with two patterned layers on different natural lattices there is no grid the caller can pass that
satisfies both the advice and the union-grid rule. The RAISE arm is unaffected (it is an absolute
cost bound, always actionable via `max_pencil_dof=`); only the advisory WARN is wrong.

**Originally reported, not changed** — it was a policy decision the WP had tuned deliberately (its
25 → 5 → 0 warning sequence), and the single-layer case the audit is about must keep warning. The
coordinator ruled for the joint design; **it is implemented and measured in §8.1**, and the mortar
file now emits 0.

### V6 (low) — the PMM 2-D results never carried `wl_eff` — ADDED

Raised by the coordinator from VERIFY-A14 §6 V8. `Efficiency2D.__new__` has taken a `wl_eff`
argument since audit H2 and the 1-D / RCWA results fill it, but every PMM 2-D construction site
left it `None`, so a caller could not tell a Wood-nudged 2-D solve from an exact one except by
catching the warning. Filled in at all five NumPy sites — `twod._pmm2d_solve_core`,
`PreparedPMM2D.solve`, both scalar entries' `stabilize=True` consensus tails, and
`twod_staggered.pmm_efficiency_2d_staggered` — from the `_grazing_safe_wavelength` value each one
already computes. The JAX dispatch sites keep `None`, which is the class docstring's documented
meaning ("the entry point did not record it, e.g. a traced JAX wavelength").

Fixture, built rather than hoped for: at normal incidence into `n_sup = 1` the `m = ±1` order sits
EXACTLY at cut-off when `Λ = λ`, and `_grazing_safe_wavelength` substitutes `λ·(1 + 1e-7)`.
`Λ = 0.47 µm` is the off-anomaly control. MEASURED, all seven `Efficiency2D`-returning surfaces:

| entry | OFF-anomaly (Λ = 0.47 µm) | ON-anomaly (Λ = λ = 1 µm) |
|---|---|---|
| `pmm_efficiency_2d` | `1e-06` (exactly the request) | `1.0000001e-06` (+1.000e-07 rel) |
| `pmm_efficiency_2d(stabilize=True)` | `1e-06` | `1.0000001e-06` |
| `pmm_efficiency_2d_cell` | `1e-06` | `1.0000001e-06` |
| `pmm_efficiency_2d_cell(stabilize=True)` | `1e-06` | `1.0000001e-06` |
| `prepare_pmm_2d(...).solve` | `1e-06` | `1.0000001e-06` |
| `prepare_pmm_2d_cell(...).solve` | `1e-06` | `1.0000001e-06` |
| `pmm_efficiency_2d_staggered` | `1e-06` | `1.0000001e-06` |

and the decisive round-trip — re-solving AT the reported `wl_eff` (which is off the anomaly, so it
takes no second nudge) is **bit-identical** on `R` and `T`, for both `pmm_efficiency_2d_cell` and
`pmm_efficiency_2d_staggered`. That is what separates a real `wl_eff` from an echo of the request:
an echo fails the ON-anomaly arm, and a wavelength the solver did not use fails the round trip.
`.dof` and the `(orders, R, T)` unpacking are untouched.

Gate: `test_audit2609_a13_verify_guards.py::test_v6_*` (3 tests). The `stabilize=True` tails carry
`wl_eff` through a closure cell rather than from the consensus result, with a why-comment recording
that the nudge reads wavelength / angles / periods / order set / region eps — none of which the
degree scan moves. The JAX dispatch is unaffected: a traced wavelength still returns `wl_eff = None`
with `.dof` preserved and `jax.grad d(ΣT)/d(wavelength)` finite (−1.567e+04).

---

## 5. Tests run

All `OPENBLAS_NUM_THREADS=1`, `-q --no-header -p no:cacheprovider`.

| command | result |
|---|---|
| the four new WP files + `test_audit_w4_jax_static_caches.py` (before my edits) | **61 passed, 54.1 s** |
| the same five + `test_audit2609_a13_verify_guards.py` (after my edits) | **70 passed, 52.2 s** |
| hybrid suite, 16 files (`test_v5_11_0_pmm2d`, `test_v5_14_0_pmm2d_{cell,stack,conical,oop,stabilize}`, `test_v5_14_0_pmm_jones_2d`, `test_v5_12_0_pmm2d_loss`, `test_v5_13_0_pmm2d_hybrid_sweep`, `test_v5_20_13_pmm_jones_2d_fff_nv`, `test_p2c_pmm2d_stack_cascade`, `test_p2t_pmm2d_tree_cascade`, `test_pmm2d_lossless_closure_two_sided`, `test_pmm2d_oop_block_eig`, `test_audit_s1_3_pmm2d_lossless_tripwire`, `test_v5_14_3_pmm_internal_field`) — before my fixes | **208 passed, 785.5 s** |
| the same 16 + `test_v5_14_2_jax_stacks`, `test_v5_21_pmm_threaded_sweep`, `test_niche_audit_w7_pmm`, `test_audit_s5_4_standalone_jones_transmission` — RE-RUN after V1–V4 | **362 passed, 779.0 s** |
| slant / autodiff / JAX / misc, 16 files (`test_pmm2d_slant_metric`, `test_fix_hybrid_slant_transmission_anchor`, `test_verify_pmm2d_perlayer_slant`, `test_{fix,verify}_slant_anchor_v1_v2_o2`, `test_v5_14_0_pmm2d_autodiff`, `test_v5_20_2_pmm_jones_2d_jax`, `test_v5_14_2_jax_stacks`, `test_niche_audit_w7_pmm`, `test_v5_14_0_pmm_audit_fixes`, `test_audit_w3_pmm_jax_guards`, `test_v5_21_pmm_threaded_sweep`, `test_audit_s5_4_standalone_jones_transmission`, `test_v5_18_1_jones_shared_eig`, `test_{fix,verify}_branch_cut_round2`) | **328 passed, 429.6 s** |
| staggered suite, 10 files (`test_pmm2d_staggered_{anisotropic,magnetic,nonuniform,oop,oop_block_eig,slant,wood_list,mortar}`, `test_v5_12_0_pmm2d_staggered`, `test_audit_p1_staggered_guard`) | **300 passed, 973.9 s** — with **6** cost-guard warnings remaining, all in `test_pmm2d_staggered_mortar.py` (§4 V5 half 2) |
| staggered / mortar remainder, 10 files (`test_audit_dynameta_consumer_api_2`, `test_audit_w3_entry_validation`, `test_audit_w6_pmm_rcwa`, `test_{fix,verify}_pmm2d_mortar_round*`, `test_pmm2d_staggered_oop_corner_convergence`, `test_v5_21_pmm2d_staggered_oblique`) | **117 passed, 1 skipped, 0 failed, 2617.1 s** — the WP reported "115 passed, 1 skipped, **1 FAILED**" for this group; the failure is gone (§6 item 3) |
| `tests/unit/test_verify_pmmstack_sliver_walls.py` (whole file, after the V5 fix) | **15 passed, 41.5 s** |
| `tests/unit/test_audit2609_a13_verify_guards.py` (my own, final) | **12 passed, 5.6 s** |
| FINAL post-`wl_eff` pass, 16 files (the four WP a13 files + mine + `test_audit_w4_jax_static_caches`, `test_v5_11_0_pmm2d`, `test_v5_14_0_pmm2d_{cell,stabilize}`, `test_v5_13_0_pmm2d_hybrid_sweep`, `test_v5_12_0_pmm2d_loss`, `test_audit_s1_3_pmm2d_lossless_tripwire`, `test_pmm2d_lossless_closure_two_sided`, `test_v5_12_0_pmm2d_staggered`, `test_audit_p1_staggered_guard`, `test_pmm2d_staggered_wood_list`) | **168 passed, 613.6 s** |
| `test_audit_dynameta_consumer_api_2.py`, `test_audit_w3_entry_validation.py`, `test_audit_w6_pmm_rcwa.py` re-run standalone AFTER `wl_eff` (the consumer-API / result-shape files, the ones most likely to notice a new `Efficiency2D` field) | **exit 0** — no failures |
| `test_fix_pmm2d_mortar_round2.py::test_the_plain_1d_interface_solve_is_left_unguarded_and_this_is_why` | **1 passed** — the failure WP-A13 reported (its §5 item 4) is **NOT reproducible** now; another agent has `pmm/stack.py` and that test file modified in the working tree. |

Audit repro scripts re-run and quoted above: `p3b.py`, `q1b_stackli.py`, `q1c.py`, `q11_prepared.py`,
`q13_warn.py`, `q3_fffnv.py`, `q1_conv.py`, `p1cde.py`, plus `_sqrt_decay` for G11. I deliberately
did **not** re-run `q14_stagtime.py` (M = 10 staggered, 712–911 s) or `q12_stag.py`
(498 s / 8.4 GB); every staggered assertion of mine is on the guard, and the largest staggered solve
I ran is M = 8.

### The "keep intact" list, re-measured on `p1cde.py` after the change

| property | my reading | WP reading | audit reading |
|---|---|---|---|
| 90° rotation covariance of `pmm_jones_2d` ITSELF (n=3/5, θ=0°/20°) | 9.948e-14 / 1.323e-13 / 3.055e-13 / 1.675e-13 | identical | 1.96e-14 / 7.85e-14 / 3.30e-14 / 5.44e-14 |
| mirror symmetry → zero cross-pol | \|Jxy\| 3.321e-14, \|Jyx\| 4.178e-15 | 3.32e-14 / 4.18e-15 | 2.61e-14 / 1.32e-15 |
| even-parity fold vs the full solve | 3.208e-13 | 3.21e-13 | 1.75e-13 |
| C4 → Jxx = Jyy | 1.795e-14 (laurent) / 1.470e-13 (li), \|Jxy\| ≤ 2.3e-14 | 1.80e-14 / 1.47e-13 | 2.42e-13 / 8.25e-14 |

I reproduce the WP's readings digit for digit, which — with the WP's own in-place-revert control for
the `twod_jones` half of G10(b) — supports its conclusion that the gap to the audit's numbers is the
two machines' BLAS and not this change. All four properties remain in the audit's decade.

### Tests audited against `docs/TESTING_STANDARDS.md`

Sound: every numeric bar in the four new files carries a derivation, an oracle and a measured value;
the oracles are independent (a 90° rotation, `rcwa_jones_1d`, `pmm_efficiency_2d_cell`, the pre-fix
bodies for bit-identity); no wall-clock or speedup assertion anywhere (the 3.96× fold speed-up is in
a comment, explicitly not asserted — S1); no `pytest.skip` on a resource precondition (the JAX files
use a module-level `pytest.importorskip`, which is a dependency, not a resource — S3); the
structural gates (`test_g5_the_routed_pair_is_what_the_stack_now_passes`,
`test_g10_the_twins_no_longer_build_the_dense_kronecker_pair`) are the right shape for a fix a future
refactor could re-break by accident.

Two notes:

* `test_audit2609_a13_stack2d.py::test_g10_rectangular_truncation_is_untouched` pins a literal
  `T00 = 0.023191963156` to `abs=5e-12` — relative 2.2e-10 on an eigendecomposition-derived
  efficiency. It has a stated origin (the audit's own reading), but it is the S4 "floor bar" shape
  and the durability rule says nothing should pin a prior version's number. The property it wants
  ("the rectangular default is untouched by F8") would be better stated as an in-process invariance:
  the stack's rectangular answer must be `np.array_equal` to `pmm_efficiency_2d_cell`'s at the same
  settings (the neighbouring test already measures that at 1e-10 — tightening it to `array_equal` on
  the same-process pair would be build-free and stronger). Low priority; it passes here and the bar
  probably has ~1 decade of margin.
* `test_audit2609_a13_twod.py::test_g13_transmission_jones_matches_the_stack_and_the_qwp_reference`
  asserts `< 2e-2` between `pmm_jones_2d(formulation='li')` on a TENSOR cell and the stack's
  `jones_transmission()` on the SCALAR cell, with the comment "both entries run the identical
  per-slot-Li operators". They do not: on the tensor entry `'li'` selects only the `E_z` rule and the
  in-plane block stays Laurent, which is why the bar has to be 2e-2. The comment is wrong and the bar
  is loose; the bit-identity the WP actually achieved is measurable — see my §3.7 table, where the
  same comparison run on comparable discretisations reads `0.000e+00`. Worth restating rather than
  removing (it would catch a transposed or conjugated Jones).

---

## 6. Open items for the orchestrator

| # | Sev | Item |
|---|---|---|
| 1 | low | **Commit-message correction.** `83142777`'s message says the audit's 7200² pencil "now warns instead of running 498 s / 8.4 GB". Measured: `dof = 7200 < 12 000`, so it warns **and still runs**. The report and the changelog are correct; only the commit message is not. |
| 2 | low | **`lumenairy/__init__.py` re-export still missing.** `pmm_2d_order_drift` is now in `lumenairy/elements/pmm/__init__.py::__all__` (landed in `49c05569`), so `from lumenairy.elements.pmm import pmm_2d_order_drift` works; `lumenairy.pmm_2d_order_drift` still does not. WP-A13's §5 item 1 is half-landed — the top-level block that re-exports `prepare_pmm_2d` still needs the name. |
| 3 | **closed** | **WP-A13 §5 item 4 is stale.** `test_fix_pmm2d_mortar_round2.py::test_the_plain_1d_interface_solve_is_left_unguarded_and_this_is_why` now PASSES — re-confirmed after VERIFY-A12's follow-up landed as commit `49c05569`, which carries the `pmm/stack.py` and test-file changes. Nothing for WP-A12 to do. |
| 4 | **closed** | **Docstring amendment (G6)** — done in §8.2.  Original finding: `pmm_jones_2d`'s `formulation` block states a blanket preference `fff_nv > laurent > li`. On the TRANSMISSION retardance of the WP's own QWP fixture `li` is the most accurate at `n_orders ≥ 9` (+100.036 / +100.039° vs `fff_nv` +100.233 / +99.904°, reference +100.066°). Exact change: qualify the ordering as measured on the REFLECTION Jones, and add one sentence saying the transmission retardance ranks differently on a form-birefringent stripe — a QWP designer reads that paragraph. I did not edit it because the whole paragraph is the WP's deliberate G6 wording. |
| 5 | **closed** | **Test restatements (G10 / G13)** — both done in §8.3: the literal `T00` pin is now an in-process invariance, and the G13 seam arm is 1e-12 (measured 0.000e+00) on operators that really are identical. |
| 6 | info | **Pre-existing, unrelated:** `pickle` / `copy.deepcopy` of a `PMM2DStackHybrid` raise on the `LayerCache` lock (present at the parent commit). Also `_axis_elem_counts`'s argmax tie-break makes an EXACT-tie 50 %-duty grating non-covariant under a 90° ROTATION (though covariant under the transpose the tests use) — a discretisation artefact, not G5. |
| 7 | **closed** | **The shared-grid redundancy warning gave advice the caller could not follow** (§4 V5 half 2), and the WP's "the shipped staggered suite is SILENT" was not accurate — 6 warnings in `tests/unit/test_pmm2d_staggered_mortar.py`. Both **resolved in §8.1**: the joint lcm rule is implemented, the mortar file emits **0**, and the WP report / changelog carry the dated correction. |
| 8 | info | **`wl_eff` is now on the PMM 2-D results** (§4 V6), closing VERIFY-A14 §6 V8. The JAX dispatch sites keep `None`, per the class docstring. `pmm_jones_2d` and the two stacks return plain tuples, not `Efficiency2D`, so they have no slot for it — if the same signal is wanted there it needs a return-shape decision, which is a separate change. |
| 9 | info | **G10(d) deferred item stands.** The tensor-`lops` cache (WP §6.1) is genuinely not done; the design in §6.1 is sound and its bit-identity gate is the right one. `cascade='fused'` measured non-bit-identical here too (1.9e-16…7e-16 on my normal-incidence fixture; the WP's oblique fixtures reached 1e-13), so leaving the default at `'fast'` is right. |

---

## 7. Bottom line

All nine findings are genuinely closed at the physics level, and the P1 (G5) is closed on a much
harder fixture than the WP used — two different layers, anisotropic periods, oblique conical
incidence, both the rotation and the mirror symmetry, NumPy and JAX — with a 9-decade pre/post gap.
Every defect I found is in the guards and signals the WP added, never in the physics of the fixes:
four of them repaired here with their own regression tests (V1–V4), one handed over by VERIFY-A14
and implemented (V6, `wl_eff`), one collateral test reconciled after measuring the guard right
(V5 half 1), and one design question measured and left to the orchestrator (V5 half 2 — the
shared-grid redundancy advice a caller cannot follow, and with it the WP's "the shipped staggered
suite is SILENT" claim). No physics regression anywhere: **1 175 existing PMM 2-D tests pass across 52 files**
(208 → 362 hybrid/stack/JAX, 328 slant/autodiff, 300 staggered, 117 mortar remainder, 168 in the
final post-`wl_eff` pass, 15 sliver-walls), plus the WP's own 61 and my 12.

---

## 8. Follow-up (coordinator rulings, 2026-09-12)

Four rulings, all implemented and measured. Nothing in §1–§7 changes; §4 V5 half 2 moves from
"reported" to "fixed", and two §5 test notes and one §6 item are discharged.

### 8.1 Ruling 1 — the joint redundancy rule on the shared grid (V5 half 2)

**The rule.** `layer_grids='shared'` makes the segment lattice a property of the STACK, so the
advice is too. The stack's minimum is the lcm of the per-layer minima, capped at `N`, and the
redundancy warning fires only when the caller's lattice exceeds it. Implementation note worth
keeping: that lcm needs no separate bookkeeping — with `g_L = gcd(N, walls of L)` the per-layer
minimum is `N/g_L`, and

```
N / gcd(g_A, g_B)  ==  lcm(N/g_A, N/g_B)
```

so running `_stag_minimal_uniform_segments` over the layers' JOINT wall set computes it directly.
The existing helper already merges several arrays jointly (it was written for `eps_cell` +
`mu_cell`), so the change is a call site, not a new algorithm.

**Where it runs.** `_validate_stag_cost` gained `check=("raise", "warn")` and `n_min_joint=`.
The RAISE is an absolute cost bound, always actionable via `max_pencil_dof=`, so it still fires at
`add_layer` on both paths. The WARN is advice, so on a shared stack it is deferred to
`PMM2DStackPure.solve`, which calls the new `_warn_stag_shared_redundancy()` **first** — before any
QZ, i.e. before the cost it is warning about — over every patterned layer at once
(`_patterned_cells()` collects the `eps_cell` of a plain patterned layer and the patterned side(s)
of a magnetic one). `layer_grids='per-layer'` is untouched: there each layer really does own its
lattice, so per-layer advice is followable and both arms stay at `add_layer`. `max_pencil_dof=` on
ANY layer sets `_stag_cost_ack` and suppresses the joint warning, as it already did per layer.

**Why deferral rather than the interim skip.** The two requirements — "the 2×2 + 3×3 on 6×6 fixture
is SILENT" and "the single-layer audit case must keep warning" — cannot both hold at `add_layer`
time: when layer A arrives there is no layer B yet, so a per-layer check on the first patterned
layer necessarily warns on the very fixture that must be silent. Evaluating at `solve` is the only
point where the patterned set is complete, and it is also the point where the cost is incurred. The
single-layer case is unaffected in substance: with one patterned layer the joint minimum IS that
layer's own minimum, so the audit's 12×12 pillar still warns with the same number, one call later.

**Measured** (guard only — no solve runs in any of these; `_warn_stag_shared_redundancy` calls no
eigensolve):

| fixture | at `add_layer` | total | names |
|---|---|---|---|
| 2×2 + 3×3 pillars on a SHARED 6×6 (the mortar shape) | 0 | **0** | — |
| the same pair tiled onto a SHARED 12×12 | 0 | **1** | `uniform 6x6 lattice` (not 2×2, not 3×3) |
| single 6×6 reducible to 2×2 | 0 | **1** | `uniform 2x2 lattice` |
| the audit's 12×12 half-fill pillar, M = 5 | 0 | **1** | `uniform 4x4 lattice`, `4608x4608` → `512x512`, `729x less QZ time` |
| the 6×6 pair with `layer_grids='per-layer'` | 2 | **2** | `uniform 2x2` and `uniform 3x3` — unchanged |
| the 6×6 pair with `max_pencil_dof=1e7` | 0 | **0** | acknowledged |
| `max_pencil_dof=1` on either path | raises | — | the absolute refusal is unchanged |

**`test_pmm2d_staggered_mortar.py` re-run: 31 passed, 197.1 s, SEGMENT-grid warnings 6 → 0.**
(Counted by grepping the message text rather than by a `-W` module filter: the warning carries
`stacklevel=3`, so Python attributes it to the caller's module and a
`-W error::UserWarning:lumenairy.elements.pmm.twod_staggered` filter does not reliably catch it —
worth knowing if anyone re-arms this gate in CI.)

**WP-A13 docs corrected** (§2 item 3 discharged): `fixes/WP-A13_REPORT.md` §2.5's "the shipped
staggered suite is SILENT" now carries a dated CORRECTION paragraph stating the 6 measured
warnings, why the earlier re-run missed that file, why the advice was unfollowable, and the 6 → 0
result; `fixes/WP-A13_CHANGELOG.md`'s G9 block gains the joint-rule bullet with the same numbers.

### 8.2 Ruling 2 — the G6 ordering is qualified to the reflection Jones (§6 item 4 discharged)

`pmm_jones_2d`'s `formulation` docstring now says the `fff_nv > laurent > li` ordering is measured
**on the reflection Jones** (which is what both of its tables are), and adds the transmission table
with the sentence a waveplate designer needs:

| n_orders | `fff_nv` | `laurent` | `li` |
|---|---|---|---|
| 5 | +100.12 | +94.69 | +96.73 |
| 9 | +100.23 | +98.94 | **+100.04** |
| 15 | +99.90 | +99.63 | **+100.04** |

against the `rcwa_jones_1d(n_orders=60, 'li')` reference **+100.066°** and the no-floor
`PMM2DStackPure` cross-check **+100.051°** at M = 8 — so `'li'` is the most accurate of the three at
`n_orders ≥ 9` on a form-birefringent stripe (0.03° against 0.17–0.34°), and the reader is told to
pick the rule for the observable being designed against. `'auto'` and the default are unchanged; the
same paragraph is added to the changelog's G6 block.

### 8.3 Ruling 3 — the two test restatements (§5 notes discharged)

* `test_g10_rectangular_truncation_is_untouched` no longer pins the literal `T00 = 0.023191963156`
  at `abs=5e-12`. It is now an **in-process invariance** with no tolerance anywhere: the order set is
  exactly `(2n+1)² = 121`; the DEFAULT stack is `np.array_equal` to the same stack with
  `truncation='rectangular'` spelled out, on orders, `R`, `T` and the Jones; and `'circular'` retains
  strictly fewer orders, so the identity is not holding vacuously on a build where the option did
  nothing. A `max(T) > 0.5` guard keeps it from passing on an all-zero answer.
* `test_g13_transmission_jones_matches_the_stack_and_the_qwp_reference`'s seam arm no longer compares
  a TENSOR-cell entry against a SCALAR-cell stack behind a 2e-2 bar with a false justification. Both
  sides now take the **same tensor cell at the same formulation with the fold off**, so they run the
  identical `_tensor_layer_modes` operators through the identical dense cascade — and the bar is
  **1e-12**, measured **0.000e+00** (bit-identical), 10 decades tighter than before. The comment
  records why the fold must be off (at `symmetry='auto'` the two entries take different recursions —
  the entry's `_symmetric_cascade_rt` against the stack's `block_eig` — which is the 2.8e-14 in §3.7)
  and why the old bar had to be loose (on the tensor entry `'li'` selects only the `E_z` rule while
  the scalar stack applies the per-slot wall-normal inverse rule).

### 8.4 Tests run for the follow-up

| command | result |
|---|---|
| `test_audit2609_a13_staggered_cost.py` (4 new V5 tests: joint minimum, per-layer control, the `max_pencil_dof` acknowledgement, the unchanged refusal) | **12 passed, 0.14 s** |
| the two restated tests, alone | **2 passed, 8.7 s** |
| all five A13 files + `test_audit_w4_jax_static_caches.py` | **77 passed, 59.3 s** |
| `test_pmm2d_staggered_mortar.py` | **31 passed, 197.1 s; SEGMENT-grid warnings 6 → 0** |
| staggered suite (10 files) + `test_verify_pmmstack_sliver_walls.py`, post-change | **292 passed, 834.7 s; SEGMENT-grid warnings 0** |
| a once-per-geometry latch was added after that run (repeated `solve` on one stack must not re-emit an advisory): measured `[1, 0, 0, 0]` over four solves, cleared by a new PATTERNED layer and deliberately not by a uniform one. No other test in the suite asserts on this warning (`grep`), and the latch can only REMOVE warnings, so it cannot invalidate the 292/0 above | **24 passed** (`a13_staggered_cost` + `a13_verify_guards`) |
| `ruff check lumenairy/elements/pmm/` + the three edited test files | clean |

### 8.5 Files touched by the follow-up

* `lumenairy/elements/pmm/twod_staggered.py` — `_validate_stag_cost(check=, n_min_joint=)`, the arm
  gates, and the docstring paragraph deriving the lcm identity.
* `lumenairy/elements/pmm/stack2d_pure.py` — `_patterned_cells()`, `_warn_stag_shared_redundancy()`,
  the `check=` selection and `_stag_cost_ack` on both `add_layer` branches, and the call at the top
  of `solve`.
* `lumenairy/elements/pmm/twod_jones.py` — the `formulation` docstring (reflection qualifier +
  transmission table).
* `tests/unit/test_audit2609_a13_staggered_cost.py` — a `_stack_warnings` helper, two existing tests
  moved onto it, four new V5 tests.
* `tests/unit/test_audit2609_a13_stack2d.py` — `test_g10_rectangular_truncation_is_untouched`
  restated.
* `tests/unit/test_audit2609_a13_twod.py` — the G13 seam arm restated at 1e-12.
* `docs/audits/.../fixes/WP-A13_REPORT.md` and `WP-A13_CHANGELOG.md` — the two corrections above.

`lumenairy/elements/pmm/__init__.py` was NOT touched (WP-A15b owns it); §6 item 2 —
`lumenairy.pmm_2d_order_drift` still missing at the package root — therefore still stands for
whoever holds `lumenairy/__init__.py`.
