# VERIFY-WP-B12 -- independent adversarial re-verification of WP-B12 (the FGA reference plane)

Branch `verify/wp-b12`, worktree `C:\tmp\lum_vb12`, on `fix/wp-b12-fga-reference-plane`
(`d017f3ed` the fix, `1218b24f` the report); parent `96cb2096` checked out
read-only at `C:\tmp\lum_vb12_pre`.  I did not write WP-B12.

Every number below is mine.  My oracle is a **fully 3-D** tracer plus a
**band-limited angular spectrum**; WP-B12's is a **meridional, rotationally
symmetric** tracer plus a **Rayleigh-Sommerfeld ring sum**, so the two differ in
method as well as in code, and mine reaches three classes theirs structurally
cannot: a **biconic** last surface, an **oblique** input and a **mirror**-terminated
prescription.  None of my eight fixtures is one of theirs (different glasses,
radii, wavelengths, apertures and grids).  Both builds were run for every probe:
Windows py3.14.6 / numpy 2.4.4 / scipy 1.17.1 / numba 0.65.1 / jax 0.11.0, and
WSL py3.12.3 / numpy 2.4.6 / scipy 1.17.1 / numba 0.65.1 / jax 0.10.2.

---

## 1. Verdict table

| # | WP-B12's claim | verdict | my measurement (identical on both builds unless stated) |
|---|---|---|---|
| 1 | the primitives returned the LAST-SURFACE state while four `fga.py` sites added the image leg as if it were the exit-vertex plane, costing a spurious `n_exit.sag.sec(theta)` | **CONFIRMED** | on six curved fixtures of my own the pre-repair state differs from MY vertex-plane trace by exactly `n_exit.sag.sec` to every printed digit: **10.4906 / 8.8110 / 6.7170 / 4.8978 / 2.7636 / 0.6165 waves** against predictions of 10.4906 / 8.8110 / 6.7170 / 4.8978 / 2.7636 / 0.6165, and **0.0000 waves with `np.array_equal` True** on both flat controls |
| 2a | both primitives gained `reference='surface'` (default, unchanged) / `'exit_vertex'`; the repaired state is the vertex plane | **CONFIRMED** | `reference='exit_vertex'` matches MY independent vertex-plane trace to **4.1e-20 m** of height and **1.2e-12 waves** of path on all six curved fixtures, both backends; `reference='surface'` still matches MY last-surface trace to 4.1e-20 m of height and 1.3e-18 m of path |
| 2b | ONE shared helper with GENERAL sag kernels (conic, asphere, biconic, freeform, field-frame), `resolve_exit_index`, mirror sign, structural flat short-circuit | **CONFIRMED, and the generality is what carries it** | my own `x-s.u`, `opd-n.sgn.s.sec` reproduce the helper to **1.4e-20 m / 4.3e-19 m** on the asphere, the biconic, the meniscus, the doublet, the oblique fan and the mirror; the mirror sign is right (a `+1` mutation moves the OPD by 2.n.sag.sec); an immersed exit medium is resolved (n = 1 would be **6.35 waves** wrong) |
| 2c | the Jacobian is projected too (`J_v = P.J`), worth 1e-5 of fidelity | **CONFIRMED, numbers RESTATED** | I derived `P` independently and it reproduces the library's `J_v` from its own `J_s` to **4.4e-16** (row-relative, worst of 15 fixture x backend pairs) on every fixture; against a finite difference of MY OWN vertex-plane map the projected Jacobian sits at **2.0e-10 ... 5.6e-07** and the un-projected at **1.7e-03 ... 1.8e-02**, a ratio of **2.9e+04 ... 2.7e+07**. The report's own three quotations of this pair (2.1e-10/2.1e-05 in sec. 3, 3.4e-06/7.3e-04 in the test docstring) do not reproduce -- see defect **D-1** |
| 2d | the JAX path is covered by the same helper | **CONFIRMED** | `jax.grad` through `reference='exit_vertex'` matches a central FD of the same scalar to **2.1e-09**; the JAX and NumPy projections agree to **0.0 in `opd`** and 3.4e-13 in the Jacobian on an A4/A6 aspheric last surface |
| 3a | FGA fidelity 0.07-0.18 -> 0.9990-0.9998 on curved fixtures at vertex and focus | **CONFIRMED and widened** | my five curved FGA fixtures go **0.0436 -> 0.9995**, 0.1014 -> 0.9998, 0.1271 -> 0.9997, 0.3072 -> 0.9998, 0.8336 -> 0.9991; power ratios 0.104-1.031 -> 0.970-1.001. Oracle floor 1e-8 or better |
| 3b | flat controls byte-identical (same SHA-256) | **CONFIRMED (the guarantee); the EVIDENCE needs a pinned memory budget)** | both flat controls, both planes and the caustic zone: `np.array_equal` True and the same SHA-256, in-process AND tree-to-tree against `96cb2096` in child processes -- but only once `LUMENAIRY_MEM_BUDGET_MB` is pinned. The returned bytes depend on the FGA chunking, and the chunking on a load-dependent default budget: **nine distinct digests for one field** over a `chunk` sweep. My first unpinned attempt read a FALSE regression. See defect **D-4** |
| 3c | `_caustic_zone` near edge moves by the sag, 0.36-0.50 % of the focal distance | **CONFIRMED in mechanism, BOUNDED in magnitude** | the zone moves on every curved fixture and not at all on the flat ones, but the size is fixture-specific and spans **-0.351 % to +0.132 %** on mine (the sign follows the curvature, as the report says); "0.36-0.50 %" is a property of the report's fixtures, not of the repair |
| 3d | WIN/WSL agree to 8.7e-15 over 72 readings | **CONFIRMED** on my own **1353** paired readings | `compare_builds.py` pairs every numeric leaf of the two builds' JSON: V1, V2 and V6 agree to **0.0** exactly, V0 to 4.4e-16 (the Sellmeier sum), V7 to 6.7e-16, V4 to 7.8e-16, V8 to 2.2e-16, V3's 238 readings to 2.6e-15 absolute; **grand worst relative disagreement 6.0e-14**, and every `bit_identical` verdict the same on both. (Within-build byte identity is the claim; the SHA-256 of a given FIELD legitimately differs BETWEEN builds, and does.) |
| 4a | the two restated pins (a4 s10 zone -> a two-tracer decision; b7b `f_fga < 0.5` -> an infidelity ratio) | **CONFIRMED as restatements, durable** | both are decisions against an independently traced reference with premise gates; both files are green on both builds; the a4 two-sided arm is gated per fixture with a `>= 2` floor, the b7b arm carries a ratio rather than a constant |
| 4b | six pinned files did not move, each with a dated paragraph | **CONFIRMED** | the ten-file selection is green on both builds and the six carry only comment additions (`git diff --stat`: 8-14 lines each, all docstring) |
| 5 | the caustic route re-score and its four recommendations | **CONFIRMED in shape, one recommendation BOUNDED** | on my own flat-last-surface ladder the crossover, the gate's value and the screen's collapse all reproduce; the numeric boundary the report proposes (`_ABERRATION_MAX_RAD` ~0.5 rad) is fixture-dependent on my optic |
| 6 | `gbd.py`'s inline sag copy is 15.52 waves wrong on an aspheric last surface -- pre-existing, not fixed | **CONFIRMED and UNDERSTATED** | on the report's own surface the error is **15.5 waves at r ~ 0.16 mm and 47.5 waves (84 % of the sag) at its full 0.20 mm semi-aperture**; it is also wrong on a **biconic** (7.9 % of the sag) and a **field-frame decentred** (25.3 %) last surface, which the report does not mention. Measured cost in a field: **fidelity 0.978 (asphere) and 0.544 (biconic)**. **This is a P1 for the next item** |

**Ship recommendation: SHIP.**  The repair is correct on every surface class I
could reach, the default is untouched, the flat-last-surface guarantee holds bit
for bit tree-to-tree, and the blast radius is what the report says it is.  Four
defects, none of them blocking: two are documentation-level (**D-1**, **D-3** --
published numbers that do not reproduce), one is a test-coverage gap I closed
here (**D-2** -- three wrong versions of the repair survive the whole new
suite), and one is about how the byte-identity evidence was taken rather than
about the claim (**D-4**).

---

## 2. My oracle, and what it is worth

`validation/probe_verify_b12/vb12_common.py`.  It imports `lumenairy` only to
build the prescriptions the fixtures name and to score the library; never for
its physics.

* **Sag** from the **quadric root** `z = (1 - sqrt(1 - (1+k)c^2h^2)) / ((1+k)c)`,
  not the rationalised `c h^2 / (1 + sqrt(...))` that the library and the WP-B12
  probe both use -- algebraically the same surface, a different expression.
* **Intersection** by damped Newton on the implicit 3-D surface equation with
  the analytic transverse gradient; **refraction** by vector Snell in the
  `(n1/n2)` form with the normal oriented against the incident ray;
  **reflection** by `d - 2(d.n)n`.
* **Propagation** by a **band-limited angular spectrum** (Matsushima) of the
  geometrical-optics exit-vertex boundary field
  `E_exit = E_in / sqrt|det d(x_v,y_v)/d(x0,y0)| . e^{ik.opl}`, resampled by
  radius where the optic and the input are rotationally symmetric and by a C1
  Clough-Tocher interpolation of `opl` and the amplitude otherwise.
* **Glass**: three dispersionless model indices registered probe-locally, plus
  two catalogue glasses (N-SF10, N-SSK8) with the Sellmeier coefficients typed
  in here.

| control | Windows | WSL |
|---|---|---|
| my Sellmeier vs `get_glass_index`, 2 glasses x 4 wavelengths | **2.2e-16** | 4.4e-16 |
| my LAST-SURFACE state vs `rt.trace(...).image_rays` (8 fixtures) | height **4.07e-20 m**, OPL **1.30e-18 m** | identical |
| my EXIT-VERTEX state vs `TraceResult.at_exit_vertex()` (8 fixtures) | height **3.39e-20 m**, OPL **6.51e-19 m** | identical |
| the slope-vs-direction-cosine trap, `max abs(u - L)` | **1.71e-04 ... 2.01e-01** -- 1e16x the agreements above, so the comparison is real | identical |
| ASM oracle, source spacing halved (infidelity) | **3.9e-13 ... 4.5e-11** | identical |
| ASM oracle, output refinement halved (infidelity) | **0.0 ... 3.4e-08** | identical |

The oracle's own floor is therefore **<= 3.4e-08 of fidelity**, five decades
below the smallest difference any claim below turns on.

**Shared-model caveat, stated once.**  Like WP-B12's, my oracle builds the exit
boundary field from geometrical optics and then propagates it exactly, so an
oracle-vs-member comparison is not neutral between `'traced'` and the others.
It does not affect sec. 3 to sec. 5: the defect is measured directly against my
own trace with no diffraction involved, and the pre/post fidelities differ by
three to four decades of infidelity (0.96 down to 2e-04), which is four to five
decades above the oracle's own floor.

### 2.1 My fixtures

All at a 0.256 mm aperture on a 192^2 grid, NA 0.070-0.100 (measured from
each fixture's own marginal exit ray), Airy radius 3.2-6.2 pixels, focal
distance 0.95-1.47 mm.  The beam is a Gaussian of 57 um 1/e amplitude radius,
so the aperture edge sits at exp(-5.0) of the peak and the grid edge at
exp(-7.3).

| key | optic | last surface | lambda | sag at the rim |
|---|---|---|---|---|
| `asph` | plano-aspheric singlet, N-SF10 | **even asphere** R = -0.911 mm, k = -0.6, A4 = 2.0e9, A6 = -4.0e16 | 780 nm | 10.66 waves |
| `bicon` | biconic singlet, model n = 1.67 | **biconic** Rx = -0.858 / Ry = -1.17 mm | 1.03 um | 8.95 waves |
| `menisc` | converging meniscus, model n = 1.78 | conic R = **+5.4 mm** (POSITIVE sag) | 1.31 um | 1.11 waves |
| `doublet` | cemented doublet, N-SSK8 + N-SF10, 3 surfaces | conic R = -3.00 mm | 633 nm | 4.15 waves |
| `oblique` | biconvex, model n = 1.52, **input tilted 6 deg** | conic R = -1.40 mm | 850 nm | 6.63 waves |
| `mirror` | plate + concave **MIRROR** | conic mirror R = -2.5 mm | 633 nm | 4.98 waves |
| `flat_planoconvex` | plano-convex, curved FIRST | **flat** (CONTROL) | 780 nm | 0.000 |
| `flat_pair` | air-spaced pair, 4 surfaces | **flat** (CONTROL) | 1.03 um | 0.000 |

---

## 3. Claim 1 and 2 -- the defect and the repair, at the primitive

`validation/probe_verify_b12/probe_v1_mechanism.py`, an 81-ray fan per fixture,
both backends, scored against my own tracer's two planes.

| fixture | backend | defect `abs(dx)` | defect `abs(d opd)` | **predicted `n_exit.sag.sec`** | repaired `abs(dx)` | repaired `abs(d opd)` | flat identity |
|---|---|---|---|---|---|---|---|
| `asph` | fd / analytic | 7.5387e-07 m | **10.4906 waves** | **10.4906** | 2.7e-20 m | 2.8e-13 waves | -- |
| `bicon` | fd (analytic REFUSES a biconic) | 8.9574e-07 m | **8.8110** | **8.8110** | 1.4e-20 m | 1.1e-13 | -- |
| `oblique` | fd / analytic | 3.1083e-07 m | **6.7170** | **6.7170** | 2.7e-20 m | 7.7e-13 | -- |
| `mirror` | fd / analytic | 3.0757e-07 m | **4.8978** | **4.8978** | 1.4e-20 m | 1.0e-12 | -- |
| `doublet` | fd / analytic | 1.2105e-07 m | **2.7636** | **2.7636** | 2.7e-20 m | 1.0e-12 | -- |
| `menisc` | fd / analytic | 6.3899e-08 m | **0.6165** | **0.6165** | 4.1e-20 m | 1.2e-12 | -- |
| `flat_planoconvex` | fd / analytic | 1.4e-20 m | **0.0000** | **0.0000** | 1.4e-20 m | 2.8e-13 | **`np.array_equal` True on x, y, opd AND the 4x4 Jacobian** |
| `flat_pair` | fd / analytic | 3.4e-20 m | **0.0000** | **0.0000** | 3.4e-20 m | 8.4e-13 | **True** |

Every curved row reproduces the predicted sag term to the last printed digit,
including the **mirror** -- which is the two-sided evidence that
`_exit_direction_sign` is right, since a `+1` there would double the term.
Every flat row is the literal identity: the projection returns the input object.
Both builds print the same table character for character.

The default did not move: `reference='surface'` still reproduces my
last-surface trace to 4.1e-20 m of height and 1.3e-18 m of path.

---

## 4. Claim 2c -- the projection helper, derived from scratch

`validation/probe_verify_b12/probe_v2_projection.py`.

### 4.1 The state map

From the straight-line transfer to `z = 0` along the ray, `t = -z/N` with
`z = s(x,y)` and `N = sgn/sec`:

```
x_v = x + L t = x - s.u_x ,   y_v = y - s.u_y ,
opd_v = opd + n_exit.t = opd - n_exit.sgn(N).s.sec ,   sec = sqrt(1+u_x^2+u_y^2)
```

Evaluated with MY sag kernels and differenced against the library's
`reference='exit_vertex'` output: **1.36e-20 m** in `x`, **4.34e-19 m** in `opd`,
on all eight fixtures and both backends, and **exactly 0.0** on the flat
controls.

### 4.2 The Jacobian

Because `s` is evaluated at the LANDING point, the derivative of the composed
map carries the sag's transverse gradient:

```
J_v = P J_s ,   P = [[1 - s_x u_x ,  - s_y u_x , -s ,  0 ],
                     [ - s_x u_y , 1 - s_y u_y ,  0 , -s ],
                     [     0      ,      0      ,  1 ,  0 ],
                     [     0      ,      0      ,  0 ,  1 ]]
```

Two independent checks:

| fixture | backend | `P_mine . J_s` vs the library's `J_v` (row-relative) | library `J_v` vs a FD of MY map | library `J_s` vs the same FD | ratio |
|---|---|---|---|---|---|
| `asph` | fd | **1.1e-16** | 5.18e-07 | 1.49e-02 | 2.9e+04 |
| `asph` | analytic | **1.1e-16** | **1.96e-08** | 1.49e-02 | 7.6e+05 |
| `bicon` | fd | **1.1e-16** | 5.55e-07 | 1.75e-02 | 3.2e+04 |
| `menisc` | analytic | **4.4e-16** | **4.49e-10** | 1.68e-03 | 3.7e+06 |
| `doublet` | analytic | **1.4e-16** | **2.01e-10** | 2.89e-03 | 1.4e+07 |
| `oblique` | analytic | **2.6e-16** | **7.16e-10** | 8.25e-03 | 1.2e+07 |
| `mirror` | analytic | **2.2e-16** | **2.23e-10** | 6.02e-03 | 2.7e+07 |
| `flat_planoconvex` | fd | **0.0** | 3.05e-14 | 3.05e-14 | **1 (identical)** |
| `flat_pair` | fd | **0.0** | 4.96e-14 | 4.96e-14 | **1** |

The asphere and the biconic are the point: the report's oracle is a
rotationally-symmetric conic trace and could not measure either.  `P` derived by
hand from my own sag gradients reproduces the library's projected Jacobian to
**4.4e-16 at worst** over all fifteen fixture x backend pairs, on the aspheric
departure and on the biconic y-branch alike.  The two builds print this whole
table identically (`compare_builds.py`: 471 paired readings, worst
disagreement **0.0**).

### 4.3 The JAX path

`jax.grad` of a weighted scalar read of `(opd, x, J[:,0,0])` against a central
FD of the same scalar, on a step ladder:

| fixture | `reference` | grad vs FD | the two gradients differ by |
|---|---|---|---|
| `asph` | `surface` | 1.09e-09 | -- |
| `asph` | `exit_vertex` | **2.11e-09** | **3.40e-01** |
| `menisc` | `surface` | 4.93e-10 | -- |
| `menisc` | `exit_vertex` | **1.98e-09** | **3.33e-02** |

The gradients genuinely differ, so the projection is inside the differentiated
path; and the JAX branch's own state matches the NumPy branch to **0.0** in
`opd` and **3.4e-13** in the Jacobian on an A4/A6 aspheric last surface.

---

## 5. Claim 3 -- the field, against my diffraction oracle

`validation/probe_verify_b12/probe_v3_field.py`.  Two arms in one process:
`native` (the tree as it stands) and `forced_surface` (both primitives wrapped
so `reference` is discarded -- the pre-WP-B12 contract).  sec. 6 checks that arm
tree-to-tree against `96cb2096`.

### 5.1 Windows py3.14 (WSL py3.12 prints the SAME table, line for line)

| fixture | plane | `forced_surface` | `native` | field bytes | oracle floor |
|---|---|---|---|---|---|
| `asph` | exit vertex | 0.1014 (P 0.172) | **0.9998** (P 0.999) | changed | 0.0 |
| `asph` | focus 1299.4 um | 0.1563 (P 0.104) | **0.9998** (P 0.985) | changed | 2.0e-08 |
| `bicon` | exit vertex | 0.1271 (P 0.282) | **0.9997** (P 0.999) | changed | 0.0 |
| `bicon` | focus 1436.8 um | 0.1875 (P 0.174) | **0.9997** (P 0.974) | changed | 9.5e-09 |
| `menisc` | exit vertex | 0.8418 (P 0.996) | **0.9992** (P 1.000) | changed | 0.0 |
| `menisc` | focus 1180.1 um | 0.8336 (P 1.031) | **0.9991** (P 0.970) | changed | 1.1e-08 |
| `doublet` | exit vertex | 0.3072 (P 0.698) | **0.9998** (P 0.999) | changed | 0.0 |
| `doublet` | focus 1470.5 um | 0.3467 (P 0.579) | **0.9996** (P 0.986) | changed | 2.9e-08 |
| `oblique` (6 deg tilt) | exit vertex | 0.0436 (P 0.236) | **0.9995** (P 1.001) | changed | 0.0 |
| `oblique` | focus 1233.1 um | 0.0836 (P 0.193) | **0.9997** (P 0.985) | changed | 2.6e-08 |
| **`flat_planoconvex`** | exit vertex | 0.9997 | 0.9997 | **IDENTICAL** | 0.0 |
| **`flat_planoconvex`** | focus 1161.8 um | 0.9997 | 0.9997 | **IDENTICAL** | 2.8e-08 |
| **`flat_pair`** | exit vertex | 0.9978 | 0.9978 | **IDENTICAL** | 0.0 |
| **`flat_pair`** | focus 949.2 um | 0.9978 | 0.9978 | **IDENTICAL** | 3.4e-08 |

The `menisc` row is the informative one: its last surface has the SMALLEST sag
of the set (1.11 waves) and it still loses 0.16 of fidelity before the repair --
the defect is not a large-sag phenomenon.  `oblique` is the largest loss (0.0436)
because a tilted input samples the sag asymmetrically.

### 5.2 The caustic zone

| fixture | `forced_surface` (um) | `native` (um) | near edge, as % of the focal distance |
|---|---|---|---|
| `asph` | [1279.530, 1330.714] | [1279.487, 1326.093] | -0.003 % |
| `bicon` | [1264.562, 1280.452] | [1259.514, 1280.406] | **-0.351 %** |
| `menisc` | [1177.720, 1181.216] | [1178.170, 1181.220] | **+0.038 %** (positive sag => the other way, as the report says) |
| `doublet` | [1469.825, 1472.633] | [1469.817, 1471.659] | -0.001 % |
| `oblique` | [1230.630, 1231.973] | [1232.258, 1234.354] | **+0.132 %** |
| `flat_planoconvex` | [1159.427, 1163.162] | [1159.427, 1163.162] | **0.000 %** |
| `flat_pair` | [946.345, 950.838] | [946.345, 950.838] | **0.000 %** |

The mechanism is confirmed and it is two-sided (the sign follows the curvature,
which a fudge factor would not do).  The report's "0.36-0.50 % of the focal
distance" is a property of ITS fixtures: my range is -0.351 % to +0.132 %.  I
mark that claim **BOUNDED**, not refuted -- the shift is the sag, and the sag is
whatever the optic's last surface is.

---

## 6. Claim 3b -- tree-to-tree byte identity, in child processes

`validation/probe_verify_b12/probe_v5_archive.py`, run once with `cwd` and
`PYTHONPATH` pinned to `C:\tmp\lum_vb12_pre` (a clean worktree of `96cb2096`;
`git status --porcelain lumenairy/` empty) and once pinned to this worktree
(`git status --porcelain lumenairy/` also empty -- I made no library edit), with
`lumenairy.__file__` asserted under the tree in each run.  Never through
pytest.

Both runs carried `LUMENAIRY_MEM_BUDGET_MB=2000` -- see defect **D-4**: without
it the returned bytes are not reproducible run to run, and my first attempt at
this table read a FALSE regression on the two flat controls.

The probe also records `primitive_has_reference_keyword`, read from the
primitive's own docstring: **False** in the `96cb2096` tree, **True** in the
WP-B12 tree -- so the two trees are what they claim to be.

| fixture | flat last surface | plane | `96cb2096` digest | WP-B12 digest | verdict |
|---|---|---|---|---|---|
| `asph` | no | vertex | `823431fde3f2e56c...` | `7444cb118be51180...` | changed |
| `asph` | no | focus | `b0a062d320dea4e8...` | `5cc5964793258425...` | changed |
| `asph` | no | `_caustic_zone` (um) | 1279.530046884, 1330.714345656 | 1279.486946070, 1326.092571499 | moved |
| `bicon` | no | vertex | `953eac63bbfa727e...` | `0346011a5bf90417...` | changed |
| `bicon` | no | focus | `78ff68523c480c45...` | `44c7905aae99285b...` | changed |
| `bicon` | no | `_caustic_zone` (um) | 1264.562151082, 1280.452230504 | 1259.514155183, 1280.406451461 | moved |
| `menisc` | no | vertex | `0f7aa71aaea04660...` | `04755aedc13eb6a3...` | changed |
| `menisc` | no | focus | `3d48ff98299778f9...` | `9444054e63a964c0...` | changed |
| `menisc` | no | `_caustic_zone` (um) | 1177.719776078, 1181.216398998 | 1178.170102616, 1181.220486089 | moved |
| `flat_planoconvex` | **yes** | vertex | `1774ee1433c99cf6...` | `1774ee1433c99cf6...` | **IDENTICAL** |
| `flat_planoconvex` | **yes** | focus | `257a90e44409b3ca...` | `257a90e44409b3ca...` | **IDENTICAL** |
| `flat_planoconvex` | **yes** | `_caustic_zone` (um) | 1159.427138442, 1163.161819817 | 1159.427138442, 1163.161819817 | **IDENTICAL** |
| `flat_pair` | **yes** | vertex | `4f6412c8daa54115...` | `4f6412c8daa54115...` | **IDENTICAL** |
| `flat_pair` | **yes** | focus | `dc3e7eb18718f46d...` | `dc3e7eb18718f46d...` | **IDENTICAL** |
| `flat_pair` | **yes** | `_caustic_zone` (um) | 946.345368418, 950.837805026 | 946.345368418, 950.837805026 | **IDENTICAL** |

Both FLAT controls are **byte-identical at both planes and in the caustic zone
to all seventeen printed digits**, and all three curved fixtures change at both
planes with the zone moved.  The `96cb2096` digests also match my in-process
`forced_surface` arm of sec. 5 exactly (`823431fde3f2e56c...`, `b0a062d320dea4e8...`,
`0f7aa71aaea04660...`, ...), so that arm is the real pre-repair library and not an
emulation of it.

The same probe was also run UNPINNED on both trees a second time
(`probe_v5_archive_unpinned_{pre,head}_win32_314.json`): that pair agrees on the
flat controls too.  So the one disagreeing run is a transient of the default
budget rather than a property of either tree -- which is exactly why a
byte-identity proof has to pin it (D-4).

---

## 7. The convention decision, and GBD

`validation/probe_verify_b12/probe_v4_consumers.py`.  Identical on both builds.

### 7.1 Exactly two consumers, and GBD is not double-corrected

A grep for calls to the two primitives finds **four call sites, all in
`fga.py`** (lines 1255, 1540, 2181, 2457), each now passing
`reference='exit_vertex'`, plus **one** in `gbd.py`
(`apply_prescription_persurface_to_beamlets`).  `elements/lenses_gbd.py` names
them in docstrings only.  Both primitives are re-exported from `lumenairy` and
`lumenairy.raytrace`, so out-of-package callers exist in principle -- and the
unchanged default is what protects them.

Read back from the source at run time, the GBD call is

    dt = _jac( x, y, ux, uy, surfs, wavelength, per_surface=True )

-- no `reference`.  Measured rather than inferred: `apply_real_lens_gbd(
per_surface=True)` on a CURVED-last-surface prescription, run twice in one
process (as shipped, and with both primitives wrapped so `reference` is
discarded), is **bit-identical**, same SHA-256, on both my aspheric fixture
(`0b71a5b1be073276...`) and my meniscus (`0cdb1e0b2d2d170b...`).

So the report's reason for the `reference` keyword -- that design (a) would
silently double-correct GBD -- is the right reason, and the shipped default
carries it.

### 7.2 `gbd.py`'s in-line sag copy -- worse than the report says

The `_Rl` / `_kl` block reads `surfs[-1].radius` and `surfs[-1].conic` **and
nothing else** (confirmed by a source scan at run time).  Transcribed verbatim
and differenced against the package's own shared `_surface_sag_xy`:

| last-surface class | in-line vs shared sag | in waves @780 nm | as a fraction of the true sag |
|---|---|---|---|
| conic (R = -0.911 mm, k = -0.6) | **1.7e-21 m** | 0.000 | 0.0 % |
| the same + A4 = 2.0e9, A6 = -4.0e16 | **3.61e-07 m** | 0.463 | **4.2 %** |
| **biconic** (Rx = -0.858 / Ry = -1.17 mm) | **9.60e-07 m** | 1.231 | **7.9 %** |
| **field-frame decentred** (35 um on R = -0.911 mm) | **3.07e-06 m** | 3.930 | **25.3 %** |

On the report's OWN surface (R = -2.10 mm, A4 = 4.0e8, A6 = -8.0e17,
lambda = 1.064 um) the error is radius-dependent, and the report quotes one point on
that curve:

| evaluation radius | 0.100 mm | 0.125 mm | 0.150 mm | 0.171 mm | 0.180 mm | **0.200 mm (its own semi-aperture)** |
|---|---|---|---|---|---|---|
| error | 0.71 waves | 2.78 | 8.37 | **18.28** | 25.18 | **47.52 waves** |
| as a fraction of the sag | 24.2 % | 44.2 % | 62.4 % | 73.7 % | 77.6 % | **84.1 %** |

The report's "15.52 waves = 71 % of the sag" sits between the 0.150 mm and
0.171 mm rows.  It is a real reading at its own evaluation radius, and it
understates the worst case on the same fixture by a factor of three.

**What it costs in a field.**  The error is a pure exit-pupil phase screen
`k.(sag_inline - sag_true).sec`, so it can be applied to MY OWN exit-vertex
boundary field and both propagated to the focus with MY OWN angular spectrum --
no GBD internals, no monkeypatching of library code:

| fixture | peak phase error | rms | focal-plane fidelity, true sag vs in-line sag |
|---|---|---|---|
| `asph` (even asphere) | 0.456 waves | 0.220 | **0.978006** |
| `bicon` (biconic) | 2.482 waves | 0.881 | **0.543711** |
| `menisc` (conic) | 0.000 | 0.000 | **1.000000** |

And end to end, `apply_real_lens_gbd(per_surface=True)` against the same oracle
at each fixture's own focus (N = 96, dx = 3.2 um -- GBD's per-surface path costs
~N^4 here, so the GBD arms run at half the linear resolution of sec. 5; the optic,
the beam and the window are unchanged):

| fixture | last surface | GBD vs my oracle |
|---|---|---|
| `asph` | even asphere | **0.974711** |
| `menisc` | conic | 0.999156 |
| `doublet` | conic | 0.999759 |

GBD's infidelity is **30 to 100x larger** on the aspheric-last row than on the
two conic ones, and the phase-screen calculation above predicts the size.

**Verdict: this is a P1 for the next item.**  It is pre-existing (the v5.22
correction has always been conic-only) and correctly out of WP-B12's ownership,
but **three** surface classes are affected rather than one, the biconic case
costs **46 % of the fidelity**, and the fix is now a one-line call to the shared
`_project_to_exit_vertex_plane` that WP-B12 built.  Its Migration note will move
every per-surface GBD field on an aspheric / biconic / field-frame last surface.

---

## 8. Durability of the new pins -- the mutation matrix

Fourteen new tests and two restatements.  I ran them against **ten in-memory
mutations** of the repair (`validation/probe_verify_b12/vb12_mutate.py`, a
pytest plugin).  A pin that stays green under the mutation that breaks the
property it claims is not pinning that property.

| mutation | what it does | WP-B12's file (14 ids) | my file (4 ids) |
|---|---|---|---|
| *(baseline)* | -- | 14 passed | 4 passed |
| `identity` | the projection is a no-op (the pre-WP-B12 tree) | **9 failed** | **4 failed** |
| `state_only` | state projected, Jacobian left on the surface | **2 failed** | 1 failed |
| `sign_plus` | `_exit_direction_sign` always +1 | **1 failed** (the mirror test -- a precise kill) | 4 passed (no mirror fixture here) |
| `conic_only` | the projection sees a conic-only last surface | **1 failed** (the asphere test) | **1 failed** (the biconic test) |
| `biconic_drop` | ONLY the biconic y-branch stripped | **14 passed -- NOT CAUGHT** | **1 failed** |
| `field_frame_drop` | ONLY the field-frame decenter stripped | **14 passed -- NOT CAUGHT** | **1 failed** |
| `n_exit_one` | the exit index hard-coded to 1.0 | **14 passed -- NOT CAUGHT** | **1 failed** |
| `always_flat` | `_last_surface_sag_vanishes` always True | **10 failed** | **4 failed** |
| `opd_sign` | the OPL correction added, not subtracted | **5 failed** | **3 failed** |
| `no_sec` | the OPL correction drops `sec(theta)` | **4 failed** | **3 failed** |

Three mutations survive WP-B12's whole suite -- **defect D-2** -- and the four
tests in `tests/unit/test_verify_b12_fga_reference_plane.py` close them.  The
two suites together kill all ten.

One survivor is worth naming separately: under `identity`,
`test_the_two_backends_agree_on_both_reference_planes_to_one_floor` stays green,
because its claim is a RATIO of two quantities that become equal when the
projection does nothing.  That is not a fault of the claim -- other tests catch
`identity` -- but it means that test cannot be read as evidence that the
projection happened.  My
`test_the_two_backends_agree_on_the_exit_vertex_plane_where_both_are_alive`
restates the same property with that half added, and with the mask corrected
(see D-3).

### 8.1 The new pins' actual margins, re-measured

`validation/probe_verify_b12/probe_v6_margins.py` re-runs the readings the new
tests assert.  **Bit-identical on both builds**, so none of these sits inside a
cross-build spread:

| pin | bar | reading | margin |
|---|---|---|---|
| the two planes are `image_rays` / `at_exit_vertex()` | 1e-14 m | 8.7e-19 m | **1.2e+04x** |
| the projected Jacobian vs its own FD | derived, 3.50e-06 | 1.46e-07 | 24x (derived from the FD ladder) |
| ... and the un-projected one must NOT match | 10x the bar | 1.44e-02 | **411x** |
| per-surface product vs composite | 1e-6 | 2.49e-16 | **4.0e+09x** |
| the aspheric gap (conic-only != projected) | 1e-9 | 2.49e-05 | **2.5e+04x** |
| the caustic zone vs a second tracer, curved | 1e-4 rel | 2.91e-16 | **3.4e+11x** |
| the caustic zone vs a second tracer, flat | 1e-4 rel | 2.99e-16 | **3.3e+11x** |

Every bar has decades on both sides except the Jacobian one, which is derived at
run time from the FD reference's own step-halving change and is therefore
build-free by construction.

### 8.2 The two restated pins

* **`test_audit2609_a4_fga_s10.py::test_s10_untilted_regimes_are_unchanged`.**
  The three literal zone readings are gone; the pin is now the DECISION (which
  plane the zone is measured from) against `_zone_from_a_second_tracer`, the
  same estimator evaluated with the **production tracer** and an explicit
  `reference`.  The unconditional half is asserted on all three fixtures; the
  two-sided half is premise-gated per fixture (a fixture whose sag cannot
  separate the two planes is not asked to discriminate them) with a
  `len(discriminating) >= 2` floor so it cannot go vacuous.  That is the right
  shape, and it is the shape `docs/TESTING_STANDARDS.md` rule 3 asks for.
* **`test_audit2609_b7b_caustic_routing.py::..._fga_is_the_closer_one`.**  The
  `assert f_fga < 0.5` became "both members reproduce the oracle (0.99 each) and
  FGA's infidelity is under half the screen's".  The claim is a ratio, so it
  carries no per-build constant, and the oracle's own ray-quadrature convergence
  is asserted in the same test.  The routing row it used to justify is pinned
  separately and is unchanged.

### 8.3 The six files that did not move

`git diff 96cb2096..HEAD` on the six is **61 added lines, zero removed, all
docstring**.  Each carries a dated WP-B12 paragraph saying WHY its claims
survive, and the reasons are checkable rather than assertions:
`test_fga.py`'s `_singlet` is plano-convex with the curved side FIRST (a flat
last surface, so byte-identical), `test_g1_gate_generality.py` asserts
`_sag_screen_aberration_rad`, which is a paraxial marginal-ray computation that
never calls the differential transfer, and so on.

And the claim is two-sided rather than merely observed: running those six files
with the projection DISABLED in memory (the `identity` mutation, i.e. the
pre-WP-B12 library) gives **154 passed, 0 failed** in 954.7 s.  Their
assertions are independent of the reference plane on both sides of the repair,
which is what "did not move" has to mean.

---

## 9. The edges

`validation/probe_verify_b12/probe_v8_edges.py`, both builds (every row below
reads the same on WSL py3.12; the chunking row differs only in its own
round-off, 8.4e-15 against 8.2e-15 relative).

| edge | measurement | verdict |
|---|---|---|
| **dead rays** -- `exit_vertex_transfer` FREEZES them; the projection does not | on a fan with 128/201 rays vignetted AT THE LAST SURFACE, the projection moves their `opd` by up to **1.96e-05 m** (18.4 waves) where `at_exit_vertex` leaves it alone; it introduces **no** new non-finite values and leaves `alive` untouched | **harmless today, latent**: all four `fga.py` sites zero the dead beamlets before reconstruction (`:1288`, `:1571`, `:2217`, `:2470`). Open item O-1 |
| **grazing rays** -- `exit_vertex_transfer` KILLS `abs(N) <= 1e-30`; the projection has no such guard | at input slope `u = 1e8` the projected state is NaN/inf, but those rays are already `alive=False` (the aperture vignettes them first) | no reachable live case found |
| **chunking** | `mem_budget_mb` 4000 vs 2.0 and `chunk=2` differ by rel **8.2e-15 / 1.6e-14** | as documented ("identical to float round-off"; it reorders an additive sum) |
| **`LensGeometry` object entry** | `apply_real_lens_fga(geometry=LensGeometry(output_plane_distance=z))` is **BIT-IDENTICAL** to the keyword form | the repair reaches the config-object entry |
| **complex64 input** | runs; returns complex128; the field differs from the forced-`'surface'` arm, so the projection is applied | covered |
| **`per_surface=True`** | only the LAST local transfer moves; the product of the locals equals the composite to **2.5e-16** | covered by WP-B12's own test, re-measured here |

---

## 10. The caustic route, re-scored on my own optic (the route is NOT changed)

`validation/probe_verify_b12/probe_v7_route.py`.  Like the report, the ladder
optic's LAST surface is **flat**, so nothing on it moves with WP-B12 and the
reading is about the SCREEN's model error rather than about the repair; the beam
radius alone walks the sag-screen estimate.  Two curved fixtures are scored at
their own caustics afterwards, where the repair does move the `'fga'` column.
Every member is read through `apply_real_lens_universal(method=...)` at the
fixture's own traced best focus, against MY oracle.  Costs are reported, never
asserted.

### 10.1 The ladder: one optic, flat last surface, beam radius swept

| `w0` | sag-screen est. | route | `fga` | `phase_screen` | `traced` | screen infidelity | `fga` cost / screen cost |
|---|---|---|---|---|---|---|---|
| 34 um | 0.005 rad | `phase_screen` | 0.9979 | **0.9980** | 0.9979 | 2.0e-03 | 22.5 s / 0.19 s |
| 46 um | 0.018 rad | `phase_screen` | 0.9993 | **0.9995** | 0.9993 | 5.0e-04 | 2.1 s / 0.01 s |
| 57 um | 0.043 rad | `phase_screen` | 0.9997 | **0.9999** | 0.9996 | 1.4e-04 | 4.2 s / 0.01 s |
| 68 um | 0.086 rad | `phase_screen` | **0.9996** | 0.9996 | 0.9993 | 3.7e-04 | 4.1 s / 0.01 s |
| 79 um | 0.156 rad | `phase_screen` | **0.9990** | 0.9983 | 0.9981 | 1.7e-03 | 2.5 s / 0.01 s |
| 90 um | 0.260 rad | `phase_screen` | **0.9977** | 0.9954 | 0.9961 | 4.6e-03 | 3.3 s / 0.01 s |
| 101 um | 0.399 rad | `phase_screen` | **0.9946** | 0.9915 | 0.9934 | 8.5e-03 | 1.1 s / 0.01 s |

`traced` is read at `ray_subsample=2`, the DENSEST setting, forced through
`method_kwargs={'traced': {...}}`; at its shipped default of 8 it REFUSES every
fixture here ("20.0 coarse samples across the 0.26-mm aperture, threshold 32"),
which reproduces the report's open item 3.

### 10.2 Two curved fixtures at their own caustics

| fixture (curved last surface) | sag-screen est. | route | `fga` | `phase_screen` | `traced` | `fga` cost / screen cost |
|---|---|---|---|---|---|---|
| `asph` | 0.658 rad | `phase_screen` | **0.9998** | 0.9993 | 0.9997 | 32.7 s / 0.10 s |
| `menisc` | 0.022 rad | `phase_screen` | 0.9991 | **0.9993** | 0.9988 | 10.3 s / 0.04 s |


### 10.3 Do the four recommendations hold?

1. **"Keep the `aberrated` condition."**  I cannot test the over-budget arm at
   all: my ladder tops out at **0.40 rad** and my highest-estimate fixture (the
   asphere) at **0.658 rad**, three times under the 2.0 rad budget, so every row
   routes to `'phase_screen'` and the gate never fires on anything I built.
   What I CAN say is the half that makes the condition worth keeping: the
   screen's infidelity grows monotonically with the estimate on my ladder and is
   already the worst member by the top rung (8.5e-03 against FGA's 5.4e-03 and
   `traced`'s 6.6e-03).  **Consistent with the report, not independently
   established.**
2. **"The screen stays, as a cost choice."**  Reproduced.  Inside the
   low-estimate class the screen costs between nothing and ~5e-04 of fidelity
   and is **100x to 330x** cheaper on my optics (the report measured 20x to
   150x on its own).  On my aspheric fixture at its caustic `'fga'` reads
   **0.9998** against the screen's **0.9993** at **327x** the time (32.7 s
   against 0.10 s); on my meniscus the screen is very slightly BETTER (0.9993
   against 0.9991) at **258x** less time.  So the
   ordering inside the class is not uniform even after the repair, which
   strengthens the report's "keep it, as a cost choice" rather than weakening
   it.
3. **"Consider tightening `_ABERRATION_MAX_RAD` from 2.0 rad to ~0.5 rad."**
   The SHAPE reproduces and the numeric boundary does not transfer.  On my optic
   the screen's infidelity crosses 1e-3 between **0.086 and 0.156 rad** and
   1e-2 near **0.40 rad**; the report's optic crosses 1e-3 between 0.12 and
   0.47 rad and 1e-2 between 0.47 and 0.89 rad.  Two optics, two boundaries a
   factor of two to three apart -- so "~0.5 rad" is one fixture's number.  The
   durable statement is the one the report also makes: **by the time the
   estimate reaches 2.0 rad the screen is one to two decades past the accuracy
   the budget is supposed to mean.**  Any retune needs a fixture sweep, not a
   fixture.  **BOUNDED.**
4. **"`traced` deserves its own look -- it is the best member wherever its
   sampling guard lets it run."**  **Not reproduced on my optics.**  With
   `ray_subsample=2` forced, `'traced'` is never the best member on my ladder:
   it ties `'fga'` at the bottom and loses to it at every rung above 0.15 rad,
   and it loses at both curved caustics.  At its shipped `ray_subsample=8` it
   REFUSES all of my fixtures outright ("20.0 coarse samples across the 0.26-mm
   aperture, threshold 32"), which reproduces the report's open item 3 exactly.
   **BOUNDED: fixture-dependent.**

**There is also a crossover at the bottom, and it reproduces.**  Below ~0.09 rad
the screen is the better member (0.9980 / 0.9995 / 0.9999 against FGA's 0.9979 /
0.9993 / 0.9997); FGA's beamlet discretisation, not the screen's model, is the
limit there.  The report's third observation in sec. 6.2 therefore stands on a
second optic.

**I did not change the route, and I recommend it not be changed on this
evidence.**  Two optics now disagree about which member wins inside the
low-estimate class (my `menisc` prefers the screen, my `asph` prefers FGA, both
in the fourth digit), and the cost ratio is 100-330x.  That is a default worth
leaving where 5.47.0 put it until someone sweeps a fixture family.

---

## 11. Defects

None of these blocks the ship.  D-2 I fixed here (in my own test file); D-1,
D-3 and D-4 are requested changes in files I do not own, each with the exact
edit and a reproducer.

### D-1 (P3, documentation) -- three different values are published for the same Jacobian measurement, and none of them reproduces

`WP-B12_REPORT.md` sec. 2 and sec. 3 say the projected Jacobian "matches the FD
reference at **2.1e-10** where the un-projected one sits at **2.1e-05**".  The
test that is cited for it,
`test_audit2609_b12_fga_reference_plane.py::test_the_projected_jacobian_is_the_derivative_of_the_projected_map`,
says in its own docstring "MEASURED (2026-09-14): projected residual
**3.4e-06**, un-projected **7.3e-04**, ladder **1.1e-06**".

Running that test's body verbatim, on the tree as committed:

| quantity | Windows py3.14 | WSL py3.12 | the docstring | the report |
|---|---|---|---|---|
| ladder (step-halving change) | **3.5014e-07** | 3.5014e-07 | 1.1e-06 | -- |
| bar (`max(10.ladder, 1e-7)`) | **3.5014e-06** | 3.5014e-06 | -- | -- |
| projected residual | **1.4648e-07** | 1.4648e-07 | 3.4e-06 | 2.1e-10 |
| un-projected residual | **1.4393e-02** | 1.4393e-02 | 7.3e-04 | 2.1e-05 |

The two builds agree bit for bit, so this is not a cross-build spread.  The
test still PASSES -- its bars are derived at run time from the FD reference's own
ladder, which is exactly the right construction -- so no gate is affected, and
the conclusion (the projected Jacobian matches, the un-projected one does not,
by a factor of 411 against the bar) is right.  But
`docs/TESTING_STANDARDS.md` names this shape explicitly: "a numeric constant in
a test without a stated origin is a defect ... right-conclusion-wrong-numbers is
the most dangerous shape: it reads as authoritative and it passes."

**Reproducer**: `validation/probe_verify_b12/probe_v6_margins.py`, key
`t6_jacobian`.  **Requested edit** (in
`tests/unit/test_audit2609_b12_fga_reference_plane.py`, WP-B12's file):

```
    MEASURED (2026-09-15, py3.14/numpy 2.4.4 and py3.12/numpy 2.4.6, bit
    identical): projected residual 1.4648e-07, un-projected 1.4393e-02,
    ladder 3.5014e-07 (bar 3.5014e-06).
```

and the same pair in `WP-B12_REPORT.md` sec. 3.

### D-2 (P2, test coverage -- FIXED HERE) -- three plausible wrong versions of the repair survive the whole new suite

The package's own argument for the shared helper (report sec. 2.1) is that the
projection must use the package's GENERAL sag kernels, because a
`conic_sag(radius, conic)` copy "would also have been wrong on three surface
classes".  The fourteen new tests pin exactly one of those classes (the even
asphere) and no exit medium other than air.  Measured, by mutating the repair in
memory and running the suite
(`validation/probe_verify_b12/vb12_mutate.py`, both builds, identical):

| mutation | the suite |
|---|---|
| strip ONLY the biconic y-branch from the last surface the projection sees | **14 passed** |
| strip ONLY the field-frame decenter / tilt / sag callable | **14 passed** |
| hard-code `n_exit = 1.0` inside the projection (the brief's own suggested inline edit) | **14 passed** |

All three are silently wrong on a prescription the primitive accepts: measured
here, the biconic y-branch is worth **0.517 waves** of optical path on my
biconic fixture, the field-frame decenter **5.28 waves**, and a non-air exit
medium **6.35 waves**.

**Fixed here**: `tests/unit/test_verify_b12_fga_reference_plane.py`, four tests,
13.5 s total (slowest id 11.9 s), each a two-sided DECISION -- the projection
agrees with `at_exit_vertex()` on that class AND the narrower sag model is
measurably different.  They kill all three mutations and `identity` as well.

### D-3 (P3, documentation) -- the two-backend pin reads one dead-companion ray, and its recorded number is the reading with that ray removed

`test_the_two_backends_agree_on_both_reference_planes_to_one_floor` masks on
`res.at_exit_vertex().alive` -- the BASE ray's aliveness.  The finite-difference
backend additionally kills a ray whose 9-ray FD companion bundle vignettes
(`_companion_alive`), and on that fixture the outermost of 121 rays is exactly
such a ray: its FD Jacobian entry reads `5.853e+04` against the analytic
`-609.4`.  So the quantity the test computes is

| mask | Windows | WSL | the docstring |
|---|---|---|---|
| `at_exit_vertex().alive` (what the test uses) | **97.0549** | 97.0549 | 4.696e-07 |
| `fd.alive & analytic.alive` | **4.68293e-07** | 4.68293e-07 | -- |

i.e. the recorded 4.696e-07 is the reading with the dead-companion ray excluded,
which is not the mask in the code.  The test still passes because the claim is
the RATIO of the two reference planes and both are dominated by the same ray, so
the ratio is 1.0 either way -- but that also means it passes under the `identity`
mutation (sec. 8), so it cannot be read as evidence that the projection happened.

**Reproducer**: `validation/probe_verify_b12/probe_v6_margins.py`, key
`t7_backend_ratio`.  **Requested edit**: mask on
`np.asarray(d[('fd', ref)].alive, bool) & np.asarray(d[('analytic', ref)].alive, bool)`
and re-record 4.68293e-07.  My
`test_the_two_backends_agree_on_the_exit_vertex_plane_where_both_are_alive`
carries the corrected mask, a bar derived from the FD backend's own step ladder,
and the missing two-sided arm.

### D-4 (P2, evidence methodology) -- the byte-identity proof rests on a digest that is not run-reproducible, because the returned bytes depend on the FGA chunking

`apply_real_lens_fga`'s `mem_budget_mb` defaults to `_default_mem_budget_mb()`
-- the `LUMENAIRY_MEM_BUDGET_MB` environment override, else a fraction of the
**available RAM at call time**.  That budget sets the momentum chunk and the
position-lattice chunk, and chunking reorders an additive sum, so the exact
bytes move with it.  Measured on ONE field (my flat-last-surface control, at
the exit vertex, one tree, fresh process each time):

| `LUMENAIRY_MEM_BUDGET_MB` | 2000 | 1000 | 500 | 200 | 100 | 50 | 20 | 8 |
|---|---|---|---|---|---|---|---|---|
| SHA-256 | `1774ee14...` | `1774ee14...` | `1774ee14...` | `1774ee14...` | **`daf503a9...`** | **`2d389d6a...`** | **`23b6a7d1...`** | **`36609a4d...`** |

| explicit `chunk=` | (none) | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|---|---|
| SHA-256 | `1774ee14...` | `5ab8dfe0...` | `712b6cb2...` | `83831f08...` | `0cf99854...` | `42323374...` | `397e247c...` | `75fcd21f...` | `cbd13736...` |

Nine distinct digests for one physical field.  In my FIRST tree-to-tree run --
budget left to the default, on a box running the other jobs of this
verification -- the two FLAT controls hashed DIFFERENTLY between the `96cb2096`
tree and the WP-B12 tree, which reads as a REGRESSION of the Migration note's
central guarantee.  It is not one: with `LUMENAIRY_MEM_BUDGET_MB=2000` exported
into both child processes the same two controls hash identically (sec. 6), the
pre-tree digests then match my in-process `forced_surface` arm exactly, and a
second UNPINNED pair of runs agreed as well.  So the guarantee is true and the
disagreement was a one-in-six transient of the default budget.

WP-B12's report sec. 8 states the box was "running eight to twelve other heavy
python jobs throughout"; `validation/probe_wp_b12/` sets neither
`mem_budget_mb` nor the environment override.  Its sec. 8.5 IDENTICAL verdicts
are true, and they rest on a quantity that is not guaranteed run to run -- the
same evidence, taken on a busier minute, would have read REGRESSION.

**Requested**: export `LUMENAIRY_MEM_BUDGET_MB` (or pass `mem_budget_mb`) in
every byte-identity probe, and -- outside anyone's WP-B12 ownership -- say in
`apply_real_lens_fga`'s `mem_budget_mb` docstring that the returned BYTES depend
on the budget.  Related and pre-existing: `_default_mem_budget_mb`'s own
docstring calls the chunk loop "(byte-identical, max|diff|=0.0)", while
`apply_real_lens_fga`'s says "identical ... to float round-off"; measured, it is
the latter (chunked vs unchunked differ by **8.2e-15** relative, sec. 9), so the
first parenthetical is wrong.

**Reproducer**: `validation/probe_verify_b12/probe_v5_archive.py` run in each
tree with and without the override; the two sweeps above are the
`LUMENAIRY_MEM_BUDGET_MB` / `chunk` ladders in this section.

---

## 12. Open items (not defects)

* **O-1 -- the projection moves DEAD rays; `exit_vertex_transfer` freezes them.**
  `lumenairy/raytrace/exit_vertex.py` is explicit that a vignetted ray keeps its
  position, direction and OPL "exactly", because it never reached the vertex
  plane.  `_project_to_exit_vertex_plane` applies its arithmetic to every row.
  Measured on a fan clipped AT THE LAST SURFACE: dead rays' `opd` moves by up to
  **1.96e-05 m** (18.4 waves) where `at_exit_vertex` leaves it alone; no new
  non-finite value appears and `alive` is untouched.  Unobservable today: all four
  `fga.py` sites zero the dead beamlets before the reconstruction
  (`W[~ALV] = 0.0` at `fga.py:1571`, `np.where(alv, ...)` at `:2217`, the coarse
  `_trace` returns its own alive mask at `:1288`, and `_caustic_zone` masks with
  `conv = alive & ...` at `:2470`).  But it is a divergence between the module's
  two vertex-plane operators and the next consumer will not know.  One line:
  `np.where(alive, projected, original)`.
* **O-2 -- no grazing-ray guard in the projection.**  `exit_vertex_transfer`
  kills `abs(N) <= 1e-30` with `RAY_MISSED_SURFACE`; the projection has none, and
  at an input slope of `1e8` it returns NaN / inf.  I could not reach a LIVE
  grazing ray (the aperture vignettes them first), so this is latent rather than
  reachable.
* **O-3 -- `n_exit` is resolved for the PROJECTION but not for `fga.py`'s own
  image leg.**  The report's open item 4; I confirm the asymmetry exists
  (`opd_tot = dt.opd + z_image*sqrt(1+u^2)`, no index) and that it is unreachable
  from the current test suite because every FGA fixture ends in air.  A guard
  that refuses an immersed exit rather than serving it silently is the cheap fix.
* **O-4 -- the FD backend kills a ray whose companion bundle vignettes, and no
  consumer distinguishes that from a vignetted base ray.**  This is what D-3
  trips over.  The Jacobian of such a ray is meaningless (`5.9e+04` against an
  analytic `-609`), and it is reported only through `alive`.

---

## 13. Tests run

Every invocation carried
`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1` on the command line and
`--capture=sys -p no:randomly`, from `C:\tmp\lum_vb12` (Windows) or
`/mnt/c/tmp/lum_vb12` (WSL), on 2026-09-15.  The box was running the other jobs
of this verification throughout, so every wall clock carries contention; they
are reported, never asserted.

| selection | build | result | duration |
|---|---|---|---|
| the eight WP-B12-pinned files + `test_audit2609_b12_fga_reference_plane.py` + my `test_verify_b12_fga_reference_plane.py` | Windows | **198 passed, 0 failed** | 963.2 s |
| the same ten files | WSL | **198 passed, 0 failed** -- the same 198 ids, the same verdict, a different interpreter and a different numpy | 1130.5 s |
| `test_analytic_ray_transfer.py`, `test_gbd_feature_complete.py`, `test_audit2609_a1_exit_vertex.py`, `test_raytrace.py`, `test_audit_raytrace.py`, `test_audit_w3_raytrace_parity.py`, `test_audit_w5_raytrace_bundles.py`, `test_audit_w6_raytrace.py`, `test_niche_audit_w3_raytrace_sources.py`, `test_audit2609_a1_raytrace.py`, `test_audit2609_b9_raytrace_perf.py`, `test_v5_4_1_raytrace_mirror_backward_ray.py`, `test_v5_4_6_wave8_raytrace.py` | Windows | **388 passed, 0 failed** | 257.8 s |
| the same 13 files | WSL | **388 passed, 0 failed** | 527.7 s |
| the census / walker / dispatcher-pin / public-API / doc-consistency / history sweep + `test_audit_except_budget.py` (30 files) | Windows | **1378 passed, 11 skipped, 0 failed** | 297.2 s |
| `test_verify_b12_fga_reference_plane.py` alone (NEW, 4 ids) | Windows | **4 passed**; slowest id 11.9 s, total 13.5 s -- inside the 60 s budget | 13.5 s |
| the mutation matrix: the two files x 11 arms | Windows | see sec. 8 -- identical verdicts | 11 x ~26 s |
| the same matrix | WSL | **identical, arm for arm** (`mutations_both_files_linux_312.txt`: the two files together, so the reds are the per-file sums) | 11 x ~21 s |
| the six untouched pinned files under the `identity` mutation (i.e. the pre-WP-B12 library, in memory) | Windows | **154 passed, 0 failed** -- so their claims really are independent of the reference plane, not merely green after it moved | 954.7 s |

`test_public_api.py::test_installed_metadata_version_matches_source_version`,
the one-shot red WP-B12's report sec. 8.3 records and attributes to a transient,
is inside my sweep selection and passed.  So I did not reproduce it, which is
consistent with the report's own four measurements calling it order-dependent
and not a WP-B12 effect.

### 13.1 The other gates

| gate | result |
|---|---|
| `wsl -e bash -lc '~/lumvenv/bin/ruff check lumenairy/ tests/ validation/probe_verify_b12/'` | **All checks passed** |
| `.test_durations` | valid JSON, **16 204** entries, 4 new ids spliced in place (`--store-durations`), largest new id 11.94 s |
| `git status --porcelain lumenairy/` in both worktrees | empty -- I made no library edit |

### 13.2 Probes and their outputs

All under `validation/probe_verify_b12/`, each printing `lumenairy.__file__`.
V0-V4 and V6-V8 ran on BOTH builds and wrote a JSON per build; V5 is inherently
Windows-only (it needs the two Windows worktrees of `96cb2096` and the WP-B12
commit):

| probe | what | outputs |
|---|---|---|
| `vb12_common.py` | my fixtures, my 3-D tracer, my band-limited ASM oracle | -- |
| `probe_v0_controls.py` | the oracle's four controls | `probe_v0_controls_{win32_314,linux_312}.json` |
| `probe_v1_mechanism.py` | the defect and the repair at the primitive, both backends | `probe_v1_mechanism_*.json` |
| `probe_v2_projection.py` | the state map, `P` derived by hand, the FD Jacobian ladder, the JAX gradient | `probe_v2_projection_*.json` |
| `probe_v3_field.py` | the FGA field at two planes, two arms, plus the caustic zone | `probe_v3_field_*.json` |
| `probe_v4_consumers.py` | GBD's blast radius, its in-line sag copy, and what that copy costs | `probe_v4_consumers_*.json` |
| `probe_v5_archive.py` | tree-to-tree digests in child processes pinned to each tree, with and without `LUMENAIRY_MEM_BUDGET_MB` | `probe_v5_archive_{pre,head}_win32_314.json`, `..._unpinned_{pre,head}_...json` |
| `probe_v6_margins.py` | the new pins' readings against their bars | `probe_v6_margins_*.json` |
| `probe_v7_route.py` | the caustic-route ladder and the three members | `probe_v7_route_*.json` (101 paired readings, worst cross-build 6.7e-16) |
| `probe_v8_edges.py` | dead rays, grazing rays, chunking, the config object, complex64 | `probe_v8_edges_*.json` (47 paired readings, worst cross-build 2.2e-16) |
| `vb12_mutate.py` + `run_mutations.sh` | the ten-mutation pytest plugin and its driver | `mutations_win.txt` (WP-B12's file, 11 arms), `mutations_verify_win.txt` (mine, 11 arms), `mutations_both_files_linux_312.txt` (both files, 11 arms, WSL) |
| `compare_builds.py` | pairs every numeric leaf of the two builds' JSON | -- |

---

## 14. Ship recommendation

**SHIP.**

* The mechanism is the sag, exactly, on six curved fixtures of my own including
  three classes WP-B12's own oracle cannot represent, and exactly zero on two
  flat controls.
* The repaired state and Jacobian reproduce an independent derivation and an
  independent finite difference on the asphere, the biconic, the meniscus, the
  doublet, an oblique input and a mirror.
* The default plane is untouched, GBD is bit-identical, and the
  flat-last-surface guarantee holds tree-to-tree against `96cb2096` when the
  memory budget is pinned.
* The new pins are derived and premise-gated, and a ten-mutation matrix kills
  everything the two suites together are supposed to catch.

Before the next item:

1. **Take `gbd.py`'s in-line sag copy as a P1** (sec. 7.2): three surface classes,
   up to 84 % of the sag, and a measured **0.544** focal-plane fidelity on a
   biconic last surface.  The replacement is the helper WP-B12 just built.
2. Apply the three documentation edits (D-1, D-3, D-4) -- none of them changes
   behaviour, and each removes a number that reads as authoritative and is not
   reproducible.
3. Consider O-1 (one line): freeze dead rays in the projection, so the module's
   two vertex-plane operators agree about what a vignetted ray is.

---

## 15. What I could not verify

* **The over-budget arm of the `aberrated` gate.**  My largest aperture tops the
  sag-screen estimate out at 0.40 rad, so the gate never fired on my optics and
  I could not reproduce the report's "above the budget the screen is the worst
  member by 4.2e-02" on a fixture of my own.  The report could not reach it
  either (its sec. 6.3 arithmetic -- an Airy radius of 8.07 um on a 100 um pitch --
  reproduces: the H2 f/5 caustic is not scorable on a tractable grid).
* **A freeform last surface through the projection.**  The report lists this as
  its own open item 6 and it stays open: I pinned the biconic and the
  field-frame classes, but `freeform` (XY-polynomial / Zernike / Chebyshev) is
  still "correct by construction" rather than measured.  Its sag comes from the
  same `_surface_sag_xy` dispatch, and its DERIVATIVE is a finite difference
  inside the kernel, which the projection's `P` block inherits -- that FD step is
  the one part of the projection I did not measure anywhere.
* **A GPU / CuPy twin.**  There is none for these primitives; the JAX branch is
  the only accelerator path and it is covered.
* **A real cross-build spread.**  Both of my builds are the same box and the
  same CPU, so "identical on both builds" here means two interpreters and two
  numpy builds, not two LAPACKs on two machines.  Every bar in my test file is
  either bit-identity or derived at run time, so nothing I added depends on
  that; the WP-B12 pins whose numbers I re-measured (D-1, D-3) also read
  identically on both, which is why I can call those numbers wrong rather than
  build-dependent.
* **Which chunking produced the one non-reproducing byte-identity run (D-4).**
  I established that the returned bytes are a function of the momentum chunk and
  the lattice chunk (nine distinct digests over a `chunk` sweep, four over a
  budget sweep) and that the default budget is a per-call function of available
  RAM, and the outlier did not recur in five later runs -- but the outlier's own
  digest is in neither sweep, so a second load-dependent input cannot be
  excluded.  What is established is the conclusion that matters: a byte-identity
  proof over this entry point must pin the budget, and pinned, the guarantee
  holds.
