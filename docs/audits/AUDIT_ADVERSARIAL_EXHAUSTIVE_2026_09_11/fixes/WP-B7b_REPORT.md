# WP-B7b — the two behaviour changes WP-B7 measured and escalated, and the
# uniform fold completion's envelope

Branch `audit-fixes-2026-09`, parent `ea374607` (WP-B7 `f64444ec`, VERIFY-B7
`ea374607`).  Three items, each re-derived on a fixture of my own before
anything shipped.

Every before/after number below was taken in a CHILD PROCESS rooted in an
isolated tree — `git archive ea374607 lumenairy` extracted read-only for the
"before" side, my own tree for the "after" side, each with `cwd` and
`PYTHONPATH` set to that tree and `lumenairy.__file__` asserted to live under
it.  Never through pytest, never through the shared working tree (WP-B11b is
editing `_lens_traced.py` / `lenses.py` / `sas.py` / `doe.py` / `pmm/stack2d.py`
and others in it concurrently).

---

## 1. Summary table

| # | what | verdict | shipped |
|---|---|---|---|
| 1 | `_universal_route`'s caustic branch prefers `'fga'` for a SINGLE-VALUED field | **RE-DERIVED AND CONFIRMED**, on a second fixture, more strongly than WP-B7 measured it (0.1251 against 0.9991, where WP-B7 read 0.3234 against 0.9965) | **yes**, gated on the H2 aberration envelope — see §2.5 for why the literal diff was not shippable |
| 2 | `caustic='uniform'` at a real fold | **RE-DERIVED, AND WP-B7's DIAGNOSIS IS WRONG TWICE** — the 0.0030 is the module falling back on an under-resolved grid (on a resolving grid at the same kind of plane it scores **0.9435**, the best of four members), and the control-parameter fit is exact (my oracle reproduces `kappa` to 7 digits). The real defect is that `zeta` is EXTRAPOLATED past the band it was fitted on, silently, and the error is in the dark tail's ENERGY | **envelope + warning**, no field change (byte-identical), because falling back is measurably worse |
| 3 | FGA's analytic-Jacobian whitelist | **CONFIRMED**, both halves: the aspheric exclusion and the missing field-frame check (the latter reproduced as a call-time `NotImplementedError` on the parent) | **yes**, WP-B7's exact edit |

Fail-before on the parent, `tests/unit/test_audit2609_b7b_caustic_routing.py`:
**6 failed, 4 passed**.  The four that pass are the four that must — the
raytrace premise, the deliberately-unchanged multi-valued branch, and the two
pure measurements of behaviour WP-B9 already landed.

---

## 2. Item 1 — the caustic branch, re-derived

### 2.1 The oracle, and what it is worth

**Oracle B** (`scratchpad/b7b/oracle_b.py`) imports nothing from lumenairy:
explicit Schott Sellmeier dispersion, closed-form quadric intersection of a
conic of revolution, vector Snell, ray-tube Jacobian amplitude to the exit
vertex plane, then a brute-force Rayleigh–Sommerfeld-I sum
`E(P) = (1/i lambda) SUM_k A_k e^{ikr} z / r^2` onto the readout plane.  The
system is rotationally symmetric and the input is a centred circular Gaussian,
so the surface integral separates into (radial ray quadrature) x (azimuthal
trapezoid) and BOTH are converged explicitly.  The change of variable from the
exit radius to the launch height,

```
U(x_e) x_e dx_e  =  E_in(h) sqrt( h * x_e * |dx_e/dh| ) dh,
```

keeps the integrand smooth.

| control | measured |
|---|---|
| my Sellmeier `n(N-SF11, 633 nm)` vs `get_glass_index` | 1.778578377216 vs 1.778578377216, **Δ = 0.0e+00** |
| `n(N-LAK22, 1550 nm)` (item 2's fixture) | 1.630730453051 both, **Δ = 0.0e+00** |
| traced paraxial BFD vs the closed-form thick-lens BFD | 3124.369 vs 3124.34 µm |
| Oracle B's exit state vs `lumenairy.raytrace`'s, N-SF11 biconvex (fixture B7b) | height **5.4e-20 m**, slope **1.8e-16**, OPL **6.5e-19 m** |
| the same on the N-LAK22 biconvex (fixture F) | **1.6e-19 m / 1.4e-16 / 1.3e-18 m** |
| the same on the K4 plano-convex | **1.1e-19 m / 1.1e-16 / 6.5e-19 m** |
| azimuthal quadrature, 512 → 1024 → 2048 | rel L2 **1.2e-14**, **7.1e-15** |
| ray quadrature, 301 → 601 → 901 → 1351 | rel L2 1.4e-05, **2.6e-06**, **1.1e-06** |
| radial spline vs a DIRECT brute-force evaluation at 240 grid radii | max **5.2e-07** of the oracle peak |

**A NEGATIVE result worth recording, because it moved the readout plane and
would have produced a fabricated defect.**  Oracle B's first version returned
the exit DIRECTION COSINE `L` where the library returns the SLOPE `u = L/N`.
At NA 0.16 that is a 6.9e-04 disagreement — and it made the oracle report *no
caustic ring* at a plane where `_trace_meridional_fold` correctly found one,
i.e. it framed the library. `L/N = L(1 + L^2/2 + ...)` accounts for the
difference to the digit. Fixed, the two traces agree to 1e-19 m. A second
instance: the band diagnostic first added the AXIAL distance to the OPL where
the library adds the geometric path `z/N`; corrected, Oracle B reproduces the
module's `kappa` to seven digits. **Both times the oracle was wrong and the
library was right** — VERIFY-B7's own lesson, met twice in one package.

### 2.2 Fixture B7b

Different from WP-B7's in every parameter: **N-SF11** (not N-BK7),
**lambda = 633 nm** (not 1.0 µm), **R = ±1.6 mm, t = 0.60 mm** (not ±1.2 / 0.8),
**N = 256, dx = 1.4 µm** (not 192 / 2.0), **w = 80 µm** (not 100), aperture
0.30 mm → **NA 0.1603**, read at the traced best focus **925.9 µm** past the
exit vertex (derived from the oracle's own raytrace, not pinned).  The NA
ladder holds the aperture and sweeps the radii: **NA 0.048 / 0.082 / 0.126 /
0.160 / 0.202 / 0.260**.

### 2.3 The three fields at the caustic

| member | fidelity | intensity overlap | rms (µm) | EE(3 µm) | EE(6 µm) | wall |
|---|---|---|---|---|---|---|
| oracle | 1 | 1 | **1.523** | 0.8585 | 0.9992 | — |
| `phase_screen` | **0.9991** | 1.0000 | 1.540 | 0.8551 | 0.9993 | 0.7 s |
| `fga` | **0.1251** | 0.3626 | 7.241 | 0.0698 | 0.2910 | 7.3 s |
| `traced` | 0.9995 | 1.0000 | 2.095 | 0.8588 | 0.9993 | 0.2 s |

FGA puts **93 %** of the energy outside the 3 µm core that holds 86 % of the
oracle's.  WP-B7's 0.3234 / 0.9965 reproduces in kind and is *worse* here.

### 2.4 The two questions WP-B7 left open, answered

**Is it a sampling deficit?**  No — and on this fixture the answer is sharper
than "it does not converge".  Fifteen settings, same plane, same oracle:

| setting | fidelity | rms (µm) | wall |
|---|---|---|---|
| defaults | 0.1251 | 7.241 | 8.5 s |
| `w0_factor=3 / 8 / 12` | 0.0981 / 0.1161 / 0.1005 | 7.343 / 7.832 / 8.496 | 1.1 / 1.6 / 2.3 s |
| `dq_step=1 / 2 / 4` | **0.1251 / 0.1251 / 0.1251** | 7.241 / 7.241 / 7.241 | 5.9 / 8.5 / 0.3 s |
| `p_max=0.2 / 0.35 / 0.5` | 0.0817 / 0.0838 / 0.1388 | 9.378 / 9.387 / 9.308 | 42 / 57 / 57 s |
| `n_p=21 / 41 / 61` | **0.1412 / 0.1450 / 0.1462** | 6.852 / 6.769 / 6.741 | 12 / 47 / 105 s |
| `nsig=5` | 0.1261 | 7.214 | 1.8 s |
| `w0=12, dq=1, p_max=.35, n_p=61` | 0.0893 | 8.948 | 105 s |
| `w0=8, dq=1, n_p=41` | 0.1162 | 7.830 | 248 s |

`dq_step` is inert to four digits and `n_p` CONVERGES — 0.1412 → 0.1450 →
0.1462, a 0.8 % last step. **FGA converges, to the wrong field.** That is a
stronger statement than WP-B7's "does not converge" and it closes the sampling
question: the swarm is not under-sampled, the model is wrong at a real singlet
focus. Best of fifteen: **0.1462**, against `phase_screen`'s 0.9991 at 0.7 s.

**Is it `na_threshold`?**  No. The NA sweep, each plane at its own traced best
focus, each route taken from the shipped dispatcher:

| NA | oracle rms | `phase_screen` rms (error) | `fga` rms (error) | fid (ps / fga) |
|---|---|---|---|---|
| 0.0480 | 4.267 µm | 4.252 (**−0.016**) | 17.631 (+13.363) | 0.9998 / 0.1760 |
| 0.0816 | 2.645 | 2.634 (**−0.011**) | 11.993 (+9.347) | 0.9997 / 0.1353 |
| 0.1256 | 1.838 | 1.838 (**−0.000**) | 8.608 (+6.770) | 0.9995 / 0.1274 |
| 0.1603 | 1.523 | 1.540 (**+0.018**) | 7.241 (+5.718) | 0.9991 / 0.1251 |
| 0.2024 | 1.297 | 1.355 (**+0.058**) | 6.226 (+4.929) | 0.9983 / 0.1208 |
| 0.2596 | 1.131 | 1.243 (**+0.112**) | 5.374 (+4.243) | 0.9971 / 0.1133 |

`phase_screen` is closer at **every** NA, by 38× to 835×. The thin-screen
obliquity ceiling is real and visible — its error grows monotonically
0.016 → 0.112 µm — and is an order of magnitude under FGA's at the top of the
range. WP-B7's reading reproduces on a different glass, wavelength and grid.

### 2.5 What I shipped, and why it is NOT WP-B7's literal diff

WP-B7 §8.3 specifies an unconditional `return "phase_screen"`. **That diff is
not shippable, and the repository already contains the measurement that says
so.**

`_universal_route` computes `aberrated = _sag_screen_aberration_rad(...) >
aberration_threshold` — the H2 gate, whose entire purpose is that a
heavily-aberrated prescription NEVER reaches the analytic sag-screen. The
caustic branch sits below that gate. Making it return `'phase_screen'`
unconditionally routes the aberrated class onto the screen through the back
door. Measured, on the shipped G1 matrix designs at their paraxial images:

| design | NA | sag-screen estimate | in the caustic zone | route |
|---|---|---|---|---|
| M1 doublet | 0.2190 | **143.4 rad** | yes | `fga` |
| M2 asphere | 0.2003 | **171.2 rad** | yes | `fga` |
| M4 biconvex | 0.3750 | **2892.9 rad** | yes | `fga` |
| M6 f/5 (the H2 dual-oracle optic) | 0.2441 | **20.4 rad** | yes | `fga` |

All four are 10× to 1450× over the 2.0 rad budget, and all four reach the
caustic branch. `test_g1_gate_generality.py::test_matrix_fast_designs_never_route_to_phase_screen`
pins them away from `'phase_screen'` on the 2026-07-19 displaced measurement
("58-123 % wrong vs the Debye oracle"), and `test_fga.py`'s two H2 rows pin the
same thing at 21.7 rad and at 10.6 mm. Those are not expectations to restate —
they are the measurement that forbids the unconditional diff.

So the shipped branch is

```python
        if (zone[0] - pad) <= opd <= (zone[1] + pad):
            return "fga" if aberrated else "phase_screen"
```

with the measurement in the comment. The consequence is exactly the class I
measured (my whole NA ladder reads 0.003 .. 0.231 rad, all inside the envelope)
and nothing else. Confirmed against the parent, in isolated trees: **nine
routing decisions move** (WP-B7's own five single-valued a4/S10 rows and my four
above-threshold ladder rows, all `fga` → `phase_screen`) and **every other
decision in the probe is unchanged** — the multi-valued row, the two vertex
rows, the two below-threshold ladder rows, all eight G1 matrix rows and all four
H2 rows.

**The cost of the gate, stated.** I built an over-budget fixture at the SAME NA
and grid (the same N-SF11 biconvex with a conic constant swept until the gate
trips: 0.074 / 0.443 / 0.812 / 1.550 / **2.289** / 3.027 rad at k = 0 / 5 / 10 /
20 / 30 / 40) and scored it at its own ray-best focus. At k = 30 (2.289 rad,
just over budget) the oracle says `phase_screen` **0.9990**, `traced` 0.9995,
`fga` **0.1228** — so on *that* fixture the gate keeps the worse member. But
that fixture is a documented over-trip of a bound whose own docstring calls it
conservative, not a case where the screen genuinely fails; the genuinely-failing
class is M6's ~20 rad regime, and I could not reach it at a tractable grid
(the sag-screen estimate scales as `NA^3 * r_beam / lambda`, so ~20 rad at
NA 0.16 and 633 nm needs mm-scale beams and N ≳ 6e4). **Escalated, not
guessed**: someone with the H2 dual-oracle fixture should score
`phase_screen` / `fga` / `traced` at the f/5 singlet's image plane and, if
`phase_screen` wins there too, drop the `aberrated` condition. Until then the
gate stands.

---

## 3. Item 2 — `caustic='uniform'` at a real fold

### 3.1 WP-B7's 0.0030 is a GRID statement, not a fitting statement

WP-B7 §6.2 scored `apply_real_lens_traced(caustic='uniform')` at **0.0030** at
the marginal focus of an f/1.92 singlet and concluded "the failure is in the
FITTING of the control parameters to the traced branches". Re-derived on a
comparable fast singlet — **N-LAK22 biconvex R = ±3.0 mm, t = 0.55 mm, 1.20 mm
aperture, λ = 850 nm, f/1.88**, marginal focus **2058.98 µm**, paraxial
2251.55 µm, w0 = 0.30 mm — that is not what happens.

The fold at that plane has `r_c = 19.251 µm`, `kappa = 9.3446` and an Airy
boundary layer **l_airy = 2.8201 µm**, so the module's own documented
resolution gate (`l_airy >= 1.2 dx`) needs **dx ≤ 2.350 µm**, i.e. **N ≥ 572**
across a 1.2 mm aperture. Measured on the same optic, the same plane and the
same 1.344 mm window:

| grid | verdict |
|---|---|
| N = 336, dx = 4.00 µm | **falls back** — `'fold Airy scale under-resolved by the grid'` |
| N = 640, dx = 2.10 µm | **completes** — `reason='fold_ring'`, `zeta_extrapolation = 0.165` |

and on the grid that completes, against Oracle B (converged to **7.7e-06**
relative L2):

| member | fidelity | power / oracle |
|---|---|---|
| **`uniform`** | **0.9435** | 0.9467 |
| `traced (ray_density)` | 0.9395 | 0.9452 |
| `traced (wave)` | 0.9394 | 0.9452 |
| `multibranch` | 0.8819 | 0.8801 |

So at the marginal focus of a fast singlet the uniform completion is the **best
of the four members**, not two decades behind. WP-B7's 0.0030 is the module
detecting an under-resolved fold and returning the plain multibranch, exactly as
its scope section says it will — a statement about their N/dx, not about the CFU
fit. The CFU fitting is not the defect. **There is a defect, and it is
elsewhere** (§3.4).

### 3.2 Fixture F, inside the scope

**N-LAK22 biconvex, R = ±6.0 mm, t = 0.90 mm, 1.10 mm aperture, λ = 1.55 µm,
N = 320, dx = 3.85 µm, w0 = 0.40 mm, read 4.400 mm past the exit vertex** — the
plane with the largest caustic ring that the module's own gates accept on a
≤ 512 grid, found by scanning four designs × 351 planes. Nothing is shared with
the K4 suite's plano-convex.

### 3.3 Where the error is NOT

| step | check | result |
|---|---|---|
| the meridional trace | Oracle B's exit state vs the library's | **1.6e-19 m / 1.4e-16 / 1.3e-18 m** |
| the turning-point / branch pairing | `r_c` from Oracle B vs `_trace_meridional_fold` | 15.5307 vs 15.5307 µm, **−0.000 %** |
| the `zeta` / `kappa` fit | `kappa` from Oracle B, the module's own band and algorithm | **7.963834 vs 7.963835**, linear-fit residual 0.0000 |
| the bright-side (c0, c1) fit | the module's own residual | **0.0032** |
| the CFU kernel | pinned by the K4 suite against exact cubic-phase integrals | 1e-13 |

So WP-B7's hypothesis is **wrong**: the control parameters are fitted exactly.
Every step of the chain reproduces independently.

### 3.4 Where it IS — `zeta` is extrapolated, and the error is energy

`kappa` can only be fitted on the band of radii reached by BOTH coalescing
branches — the only radii where the eikonal difference defining `zeta` exists.
On fixture F that band is **10.9 nm wide** and carries a maximum two-branch path
difference of **2e-05 waves**. The completion then evaluates
`zeta = kappa (r_c − r)` across a fit band `W = l_airy = 4939 nm` (**454×** the
band) and a dark fill 20 l_airy deep (**9000×**). Nothing measured whether the
fold normal form still holds out there.

A ladder over three singlets — the fast one of §3.1 at its marginal focus, five
planes through ONE caustic of the module's own plano-convex, and fixture F —
each against Oracle B:

| `W / band` | **0.16** | 1.9 | 2.3 | 3.4 | 5.4 | 9.8 | **453.6** |
|---|---|---|---|---|---|---|---|
| two-branch band | 17120 nm | 2201 nm | 1854 nm | 1276 nm | 810 nm | 452 nm | **10.9 nm** |
| max Δ eikonal (waves) | — | 0.0777 | 0.0593 | 0.0331 | 0.0164 | 0.0067 | **0.00002** |
| `uniform` fidelity | 0.9435 | 0.9566 | 0.9333 | 0.9693 | 0.9717 | 0.9619 | 0.9297 |
| **`uniform` power / oracle** | **0.9467** | **0.9748** | **0.9712** | **1.0297** | **1.0486** | **1.1246** | **1.2275** |
| `multibranch` fidelity | 0.8819 | 0.8050 | 0.8561 | 0.8528 | 0.8836 | 0.8866 | 0.8319 |

The **shape** is stable and is the better one everywhere (0.93–0.97 against the
multibranch's 0.81–0.89). The **energy** is monotone in the extrapolation:
−5.3 % / −2.5 % / −2.9 % / +3.0 % / +4.9 % up to 5.4, then **+12.5 %** and
**+22.8 %**. The dark tail is over-filled because the extrapolated `zeta`
decays too slowly far from the fold.

The leftmost column is the one that makes this a defect rather than a
limitation: where the two-branch band is WIDER than the fit band (ratio 0.16,
i.e. no extrapolation at all) the completion is the best member available. The
error appears only when the band collapses, and nothing reported that it had.

### 3.5 What I shipped, and why not a fallback

A gate that fell back to the multibranch would be a **regression at every plane
measured** (it loses 0.06–0.15 of fidelity). The existing
`uniform_fit_halfwidth` knob does not rescue it either: W = 2 l_airy reads
0.9178 / 1.2551 and W ≤ 0.5 l_airy falls back on the pixel count. So the
correct action is the brief's second branch — document the envelope on the
function and warn where the fit says the parameters are not carried.

Shipped: `_trace_meridional_fold` returns `band`; the completion reports
`zeta_band` / `zeta_extrapolation` in its diagnostics and warns above
`_ZETA_EXTRAPOLATION_MAX = 8.0`, a bar placed in the measured gap between the
last plane inside 5 % (ratio 5.4, +4.9 %) and the first outside 10 % (ratio 9.8,
+12.5 %); the docstring carries the table above, and the grid gate that decides
whether any of it runs. **The returned field is byte-identical to the parent's
on all three probe cases** — including the one that now warns.

Two-sided, and >100× apart: the module's own K4 fold fixture reads **3.96** and
is silent; fixture F reads **453.64** and warns exactly once.

### 3.6 The oracle floor, and the rest of the ladder

Oracle B's convergence is **1.7e-06** relative L2 on fixture F and **7.7e-06**
on the fast singlet (901/1024 → 1351/2048 rays / azimuths), five decades under
the 22.8 % the item is about.

The member ranking depends on the band, and that is the practical finding:

| plane | `uniform` | `ray_density` | `wave` | `multibranch` |
|---|---|---|---|---|
| fast singlet, marginal focus (ratio 0.16) | **0.9435** | 0.9395 | 0.9394 | 0.8819 |
| fixture F, tight fold (ratio 453.6) | 0.9297 | **0.9996** | 0.9986 | 0.8319 |

So `caustic='uniform'` is the right member at a well-resolved fold with a wide
two-branch band and the wrong one at a collapsed band, where
`amplitude_model='ray_density'` is two decades of error better. The docstring
now says exactly that, and the warning fires only on the second row.

---

## 4. Item 3 — the analytic-Jacobian predicate

### 4.1 Both halves reproduce on the parent

In an isolated tree at `ea374607`:

| probe | parent | mine |
|---|---|---|
| `_is_all_conic([aspheric stub])` | `False` | `True` |
| `_pick_ray_transfer([aspheric], None)` | `ray_transfer_jacobian` (FD) | `ray_transfer_jacobian_analytic` |
| `_pick_ray_transfer([aspheric], True)` | `ray_transfer_jacobian` (silently ignored) | `ray_transfer_jacobian_analytic` |
| `_is_all_conic([field_decenter])` | `True` | `False` |
| `_pick_ray_transfer([field_decenter], True)` | `ray_transfer_jacobian_analytic` | `ray_transfer_jacobian` |
| **calling the primitive it picked** | **`NotImplementedError`** | completes |
| the same for `field_tilt`, `field_sag_callable` | same | same |
| `apply_real_lens_fga` on an ALL-CONIC A4 singlet | — | **byte-identical field** |

The last row is the one that matters for blast radius: the rename and the
widening leave the conic class bit-for-bit alone.

### 4.2 The swap, measured on the A4 singlet

`aspheric_coeffs = {4: 4.0e3}`, 4001 rays, λ = 1.31 µm:

| quantity | analytic vs FD |
|---|---|
| base-ray exit height | max rel **3.5e-16** (median 0) |
| base-ray exit slope | max **2.8e-16** |
| base-ray OPL | max **1.3e-17 m** |
| **the 4×4 Jacobian** | max rel **2.4e-09** |

They trace the same base ray; the Jacobian is where the FD truncation lives.
The step ladder confirms it is truncation and not noise — max rel
`2.43e-07 / 2.42e-09 / 1.18e-10` at `h_pos = 1e-5 / 1e-6 / 1e-7` (textbook
`h^2`), turning up to `8.5e-10` at `1e-8` as round-off takes over. The analytic
side is exact, so the swap raises accuracy.

Cost: trace count **9N → N** (36009 → 4001 rays here) and **8924 → 7484 bytes
per FGA lattice point**, so the chunk sizer fits **1.19×** more lattice points at
a fixed `mem_budget_mb`. Wall clock is NOT a win for this class: the aspheric
analytic path takes the `_AdrtDual` route (the numba conic kernel carries no
polynomial departure), measured ~2× slower per call than the FD bundle on a
4001-ray trace after warm-up. The trade is accuracy and memory for time, and it
is the same trade H4c already made for conics.

### 4.3 The test the predicate now carries

`test_b7b_predicate_is_the_analytic_primitive_s_own_domain` does not hard-code a
list of surface kinds: for each of six classes it CALLS
`ray_transfer_jacobian_analytic`, records whether it raised, and asserts the
predicate and the dispatcher agree with that. It cannot drift when the
primitive's domain next widens — which is precisely the failure being fixed.

---

## 5. Files touched

| file | what |
|---|---|
| `lumenairy/propagators/fga.py` | `_universal_route`'s caustic branch (item 1) + the measurement comment; `apply_real_lens_universal`'s member map and split-step note; `_analytic_jacobian_applies` with the `_is_all_conic` alias, `_pick_ray_transfer`, the memory-model call site and the `exact_jacobian` docs (item 3) |
| `lumenairy/elements/_lens_traced_uniform.py` | `_ZETA_EXTRAPOLATION_MAX` + its derivation; `band` in `_trace_meridional_fold`; `zeta_band` / `zeta_extrapolation` diagnostics and the warning; the accuracy-envelope docstring section (item 2) |
| `tests/unit/test_audit2609_a4_fga_s10.py` | three tests restated to `'phase_screen'` with the measurement in each docstring; the multi-valued row left alone |
| `tests/unit/test_fga_h4_h5.py` | the three H4c assertions restated per WP-B7 §8.2, with the measured swap in the comment |
| `tests/unit/test_audit2609_b7b_caustic_routing.py` | new, 10 ids |
| `docs/audits/.../fixes/WP-B7b_REPORT.md`, `WP-B7b_CHANGELOG.md` | this report and its release text |

Neither `lumenairy/propagators/fga.py` nor
`lumenairy/elements/_lens_traced_uniform.py` has a `docs/history/` document
(`scripts/record_history_fingerprints.py <path> --check` reports no matching
document for either), so there is nothing to re-record for this change.
`test_g1_gate_generality.py`, `test_niche_audit_w9_dispatch2.py`,
`test_niche_p8_capstone.py` and `test_niche_p7_seidel_gate.py` were in my
ownership and were **not** touched — the aberration gate keeps every row they
pin exactly where it was.

## 6. Tests run

All single-threaded (`OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1`),
one process at a time, `-X faulthandler`, 2026-09-14.

| command | result |
|---|---|
| `pytest tests/unit/test_audit2609_b7b_caustic_routing.py` | **10 passed** (33.3 s) |
| the same file against the parent `ea374607` in an isolated tree | **6 failed, 4 passed** (34.6 s) — the fail-before |
| `pytest` the five named files + `test_fga_h4_h5.py` + the new file + `test_niche_k4_uniform_caustic.py` + `test_niche_r2_pearcey_cusp.py` | **174 passed** (444.1 s) |
| `pytest tests/unit -k "fga or caustic or uniform or traced"` | **764 passed, 2 skipped**, 15419 deselected (1850.0 s) |
| `pytest` every suite that touches `_lens_traced_uniform.py` (k4, r2, r5, a3-verify-traced, w3, w4, walker-dy) + the new file | **369 passed** (155.2 s) |
| `pytest tests/unit/test_audit2609_a17_history_lint.py test_audit2609_a17_history_relocation.py` | **744 passed** (41.3 s) |
| `python validation/run_all.py test_lenses test_propagation` | **ALL 2 files passed** (25.6 s / 7.3 s) |
| `ruff check` the five changed source/test files | **All checks passed** |
| `scripts/record_history_fingerprints.py <module> --check` × 2 | no document for either module (exit 0) |

The two skips in the broad selection are the PySide6 import guard and the
documented `_PERSISTENT_POOL_LOCK` exemption; neither is mine and neither
changed.

### 6.1 Concurrency note

WP-B11b is editing `sas.py`, `lens_config.py`, `doe.py`, `_lens_real.py`,
`_lens_traced.py`, `_lens_kernels.py`, `lenses.py`, `pmm/stack2d.py`,
`lumenairy/__init__.py`, `elements/__init__.py` and `carrier.py` in the same
working tree. I opened none of them. The broad selection above happened to be
green including their partitions at the moment it ran; every before/after claim
in this report was taken in an isolated tree, never in the shared one.

## 7. Escalated / deferred

* **The `aberrated` condition on item 1's new route (P2).** §2.5: I kept
  `'fga'` for an over-budget prescription at a caustic because the H2 gate's
  own dual-oracle measurement forbids the alternative and I could not reach
  that regime on a tractable grid. On a 2.289 rad fixture at the same NA the
  gate demonstrably keeps the worse member (0.1228 where `phase_screen` scores
  0.9990). Whoever holds the H2 f/5 dual-oracle fixture should score the three
  members at its image plane; if `phase_screen` wins there, the condition
  should go and `test_g1_gate_generality.py`'s matrix test restated with that
  measurement.
* **`traced` beats `phase_screen` on fidelity at the caustic (P3).** §2.3:
  0.9995 against 0.9991 on fixture B7b, and 0.9995 against 0.9990 on the conic
  fixture — while `phase_screen` wins on rms spot width (1.540 vs 2.095 µm
  against the oracle's 1.523). WP-B7 posed the `fga`-vs-`phase_screen`
  question and that is what I answered; `traced`-vs-`phase_screen` at a caustic
  is a separate question with its own collimation and aperture:beam caveats and
  I did not re-open it.
* **The fold chart, not the integrator (P3).** §3.4: at a tight fold the
  uniform completion's residual is the linear `zeta` model carried past its
  band, so the next fold-accuracy work is a `zeta(r)` that is measured beyond
  the two-branch band (a second fold parameter, or a two-plane fit), not a
  better special function. The CFU kernel is exact to 1e-13 and the control
  parameters are fitted exactly.
* **`_AIRY_TAIL_CELLS` and the over-filled tail (P3).** §3.4: the +22.8 %
  energy is written into an annulus 20 Airy lengths deep whose outer part the
  extrapolated `zeta` no longer describes. Clipping the fill to the radius
  where `zeta`'s own extrapolation exceeds a derived bound would be a real fix,
  but it changes the returned field and needs its own oracle ladder; it is not
  a warning's job.
