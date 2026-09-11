# The binding build plan for the BOR multilayer guards — 5.45.1, 2026-09-12

> **STATUS — PLAN. Binding on the build that follows it.** Every bar named below is either
> (a) already measured by the scoping report
> `docs/audits/SCOPE_BOR_MULTILAYER_GUARDS_2026_09_12.md` (branch `scope/bor-guards`,
> commits `d48db15` + `657988b`, probes and JSON in `validation/probe_scope_bor_guards/`)
> and re-measured on the running build by the build's own tests, or (b) marked
> **TO BE MEASURED** and fixed by a census the build runs before it fixes the bar.
>
> Working tree: `C:\tmp\lum_bor1`, branch `fix/bor-multilayer-guards`, forked from
> `scope/bor-guards` so the report and its probes are inherited. Build evidence lands in
> `docs/audits/BUILD_BOR_MULTILAYER_GUARDS_2026_09_12.md`.
>
> The user's order this plan serves: *"I don't want any of these errors to show up in any
> of my solvers for multilayers."* Three of the five steps below therefore end in a
> **refusal or a warning**, not in a silently better number; the two that do not (the
> consolidation and the census hooks) exist so that the other three are one line each and
> so that a future population can be measured rather than assumed.

---

## 1. Terms

Defined before they are used, because four of them name a piece of code rather than a
piece of physics and the difference decides which guard applies where.

**BOR — body of revolution.** An axisymmetric stack: layers stacked along `z`, each layer
a set of concentric rings in `r`, the whole thing closed at `r = Rbig` by a PEC
(Dirichlet) wall. Solved one azimuthal order at a time: fields go as
`exp(i m phi + i q z)`, `m` the azimuthal order (an integer, fixed per solve) and `q` the
**axial wavenumber**. `qn = q / k0` is the dimensionless axial index — the cylindrical
peer of a Cartesian diffraction order's `kz / k0`. The public entry point is
`BORStack(Rbig, m, ..., basis='fd'|'sem')`; a lower-level prototype entry point
`bor_solve.build_layer(...)` / `bor_solve.solve(...)` also ships and carries a third,
legacy basis.

**The three radial bases.**

* `basis='fd'` — a Yee **div-conforming staggered** finite-difference basis on ONE uniform
  radial grid `_fd_grid(Rbig, N)` shared by every layer. The grid does not know where the
  ring walls are.
* `basis='sem'` — **SEM, the spectral-element method**: each layer gets its OWN radial
  mesh whose element boundaries are placed ON that layer's ring walls, with a polynomial
  of `degree` inside each element. Interfaces between layers whose meshes differ are
  coupled by a mortar (below). This is the accurate basis and the one that carries the
  geometry.
* `bor_solve.build_layer(basis='nodal')` — the historical nodal FD basis. It is NOT
  div-conforming, so roughly half its modal basis at a large `Rbig` is a
  divergence-violating **spurious sea** carrying zero `z`-flux. Kept as a documented
  escape hatch for legacy gates; `build_layer`'s default was changed to `'staggered'` in
  a previous audit precisely because of what section 4 below refuses.

**The enrichment window.** `BORStack._solve_sem` does not mesh layer `i` on layer `i`'s
own walls. It meshes it on the **union of the walls of layers `i-1`, `i` and `i+1`**:

```python
u = set(walls[i])
if i > 0:              u |= set(walls[i - 1])
if i + 1 < len(walls): u |= set(walls[i + 1])
```

That union is deliberate — it makes the mortar between neighbouring layers nearly
conforming and so cheap and accurate — and it is also the whole of error class C on this
engine: two neighbouring layers whose walls differ by a small `delta` manufacture an
element of width exactly `delta` in **both** meshes, and a **wall-free** layer between two
ring layers inherits both of their wall sets and carries the sliver in a mesh that has no
walls of its own.

**Mortar.** The weak (Galerkin, `r dr`-weighted) coupling of two adjacent layers whose
radial element meshes do NOT line up. Instead of demanding node-by-node continuity, the
interface demands that the tangential fields agree against every test function of the
neighbour's space; `sem_radial.sem_interface_smatrix` builds it from four exact
cross-mesh overlap matrices. When the two meshes are identical the mortar degenerates to
the ordinary mode match and the plain interface S-matrix is used instead.

**The three error classes** (named by mechanism; each is a different piece of code on this
engine, and only one of the three Cartesian fixes ports across as written):

* **Class A — the branch cut.** A modal solver returns `q^2` and takes a square root, then
  must decide which of the two roots is the **forward** (`+z`) one. The BOR engines decide
  it by physics — propagating modes by the sign of their own `r dr` `z`-flux, evanescent
  modes by decay in `+z` — but the **classifier** that decides which of those two rules
  applies is a band, and that band is scaled by the mode's own `|Re q|`. Near a radial
  cutoff the mode's own magnitude has collapsed, so the band is judging noise against
  noise, and the orientation is handed to the eigensolver's backward error — which is a
  function of the BLAS kernel and the thread count.
* **Class B — interface conditioning.** The S-matrix cascade forms explicit inverses.
  When one is numerically singular the returned number is build-dependent. The Cartesian
  guard `rcwa/_core._guarded_inverse` is a **conjunction** (equilibrated `rcond` below a
  per-site bar AND the inverse missing its own defining equation `A X = I` by more than
  `1e-8`), armed at exactly one site, because rank-deficient-but-consistent operands are
  ordinary and must not be refused.
* **Class C — mortar and sliver.** Two adjacent layers whose material walls differ by a
  small `delta` cause the discretisation to manufacture an element of width `delta`. The
  spectral-element Jacobian makes the nodal stiffness scale as `1/w^2`, the layer's modal
  spectrum acquires **spurious axial wavenumbers** far above any physical index the medium
  can support, and the interface mode match conditions accordingly.

**Fail-before switch.** A module-level boolean (`PMM_SLIVER_GUARD` is the shipped example)
whose `False` value restores the pre-fix behaviour bit for bit. It exists so a behaviour
change on a documented path is reversible by the caller without a downgrade.

**Bit-identity contract.** A step's promise that a named test population's numbers do not
move at all — hashed, not eyeballed — established by RUNNING the population before and
after, on both builds.

---

## 2. Step 1 — one flux-orientation kernel, band UNCHANGED

**Scoping section:** 7.1, first half. **Commit:** `fix(bor): ...`

**What is wrong.** The rule

```
prop = |Im q| < 1e-9 * max(|Re q|, 1e-300)
q    = q if (flux >= 0 if prop else Im q > 0) else -q
```

exists in **five** independent copies (`zcascade.py:86` per-mode loop, `zcascade.py:227`
`_orient_forward`, `sem_radial.py:428` vectorized, `_jax_bor.py:98` and `_jax_sem.py:247`
as `jnp` twins). Two companion constants are five- and six-fold duplicated with it: the
flux-normalizer fallback `|P| > 1e-10 * fnrm` (6 copies) and the R/T channel gate
`|qn.imag| < 5e-5 and qn.real > 1e-6` (5 copies). Multi-copy is the shape that bred the
six-copy factor-`i` defect (audit S1-8) and the six-copy branch-cut defect (round 2); the
user's standing rule is that a shared numerical kernel has ONE implementation,
`xp=`-parametrized, with a grep/AST test pinning the single definition.

**The decision quantity.** None — this step decides nothing. It is a pure refactor whose
only claim is that it changes no number.

**The bar.** Bit-identity, two-sided in the only sense available to a refactor: the
population must be RUN, and every hashed quantity must be **exactly equal**, on both
builds.

**The switch.** None. A refactor gets no switch.

**Tests.**

* The 148 BOR gates (`test_bor_sem.py`, `test_bor_solve.py`, `test_bor_anisotropic.py`,
  `test_audit_bor_grazing_cutoff.py`, `test_audit_p1_bor_flux.py`, `test_audit_w6_bor.py`,
  `test_niche_audit_w6_bor.py`, `test_audit_v5_24_2_b2_bor_exports.py`) plus
  `test_bor_sem_jax.py` and `test_v5_20_11_bor_jax.py` — run, not asserted.
* A pre/post hash of `R` and `T` over **>= 30 fixtures** spanning both bases, `m` = 0,1,2,5
  and `k0` = 0.8, 2.0, 3.5, on both builds.
* `test_fix_bor_multilayer_guards.py::test_exactly_one_definition_of_forward_orient` — the
  AST/grep peer of round 2's `_sqrt_decay` single-definition test: exactly one
  `def forward_orient` in the package, at the line the imported function reports.

**Bit-identity contract.** TOTAL. Every one of the 148 gates, every JAX gate, and every
one of the >= 30 hashed fixtures, on both builds. A single moved digit fails the step.

---

## 3. Step 2 — the band change, and the EME exact-zero pins

**Scoping sections:** 7.1 second half (band), 7.4 (EME). **Commit:** `fix(bor): ...` for
the band; `fix(eme): ...` for the pins. Two commits, because they are two modules and two
populations even though they are one defect class.

**What is wrong — BOR.** The band's scale is the anomaly. Every other band of this shape
in the library scales by the SPECTRUM's largest element floored at 1.0
(`rcwa/_core._sqrt_decay`, `berreman.py:188/:350`, `_berreman_jax.py:76`,
`eme_2d_vector.py:255`, `pmm/_core.py:6603`); the five BOR copies scale by **the mode's
own `|Re q|`**. `_CUT_BAND_REL`'s own docstring records that this exact choice was
measured and rejected. Measured consequence near a radial cutoff (scoping 2.3, 2.4, 5.2):
a lossless stack's closure degrades to **1.2167e-04** (2.1579e-04 across kernels), a
propagating channel is DROPPED from the R/T set, the channel count moves on **21 of 24**
rungs with the BLAS kernel and on **35 of 39** with the thread count alone, and `sum R`
moves by a full **1.0000**.

**The decision quantity.** `sigma = |Im q| / max(max|q| over the layer's spectrum, k0)`,
replacing `rho = |Im q| / max(|Re q|, 1e-300)`. The floor is `k0` and NOT 1.0 because `q`
has units of inverse length here — a literal 1.0 would make the band unit-system-dependent,
the failure audit P2-06 fixed for the channel gate.

**The measured two-sided bar.** `_BOR_CUT_BAND_REL = 1e-8`, from scoping 2.5:

| side | population | n | worst `sigma` | room at 1e-8 |
|---|---|---|---|---|
| NOISE (band MUST reach) | lossless propagating, ORDINARY geometry | 54 | 1.3024e-15 (WIN) / 6.4173e-16 (WSL) | **6.89 decades** |
| NOISE at a DEEP CUTOFF (binding) | `delta` 1e-4 .. 1e-26, `qn` 1.4e-02 .. 1.4e-13 | 72 | 8.7301e-10 (WIN) / 2.3089e-10 (WSL) | **1.06 decades** |
| SIGNAL (band must NOT reach) | genuinely lossy at `Im(n) = 1e-3` | 6 | 2.5742e-05 | **3.41 decades** |
| SIGNAL, thin end | `Im(n) = 1e-6` | 6 | 2.5742e-08 | **0.41 decades** |

**These are the scoping's numbers and this build RE-MEASURES all four sides on its own
populations**, under both kernel and thread ladders on both builds, and the test asserts
the re-measured margins. The thin signal end at 0.41 decades is the tightest and is
accepted because it is a *harmlessness* boundary, not a correctness one: the candidate
calls weakly lossy media (`Im(n)` between 3.9e-07 and 1.4e-09) "propagating" and orients
them by flux rather than by decay, and over **645** physically propagating modes at
`Im(n)` from 1e-06 to 1e-10 the two verdicts agree on every one, with the flux eleven
decades above its own noise fallback.

**The switch.** None, and deliberately so. This is not a behaviour change on a documented
path; it is a defect fix that leaves ordinary geometry bit-identical (measured: 0 of 12
ordinary solves moved) and is a pure widening where it does not.

**What is wrong — EME.** `eme_2d.py:130` `_ky_forward` and `eme_diffraction.py:176` carry
the EXACT-ZERO pin (`np.where(ky.imag < 0.0, -ky, ky)`) that round 1 removed from
`_sqrt_decay`, on a value that came out of `eig` and whose imaginary part therefore IS the
backward error. Measured (scoping 6): `eme_2d.layer_modes` at `Nx=96`, `k0=20pi`,
`ky0=0.37` returns **62 modes** with real `eps` against **69 (WIN) / 71 (WSL)** with
`eps + i1e-30` — a 2.55% / 3.64% gap, **with the mode lists differing between builds**.
The deciding `|Im ky| / |ky|` is 1.34e-16 against a physical 1.81e-29.

**The EME bar.** The relative band the module's OWN vector sibling `eme_2d_vector.py:255`
already uses: `tol = 1e-9 * max(1, max|ky|)`. Here `ky` is dimensionless (the module works
in `k0`-normalized units), so 1.0 is the correct floor and no `k0` appears.

**ONE shared helper if the three can share one.** `eme_2d_vector.py:255`'s body is the
target shape; if the three sites' inputs are the same kind of quantity the helper is
written once in the EME package and the three call it. If they are not — a shape or a
units difference — the plan records why and each site keeps its own call with the same
band. **TO BE DETERMINED by reading the three sites in the build.**

**Tests.**

* `test_fix_bor_multilayer_guards.py::test_near_cutoff_closure` — a fixture at
  `qn ~ 2.5e-03` reads lossless closure `< 1e-08` (shipped: 1.2167e-04).
* `...::test_near_cutoff_channel_count_is_thread_stable` — the channel count is the same
  at `OPENBLAS_NUM_THREADS` 1 and 4 (shipped: 2 against 3).
* `...::test_no_forward_mode_carries_backward_flux` — 0 modes with `prop` and `flux < 0`
  over the census fixtures.
* `...::test_band_two_sided_population` — all four sides of the table above, re-measured on
  the running build, with the stated margins as the assertion.
* `...::test_ordinary_battery_is_bit_identical` — the >= 30 hashed fixtures of step 1 are
  unchanged by the band change.
* `...::test_jax_twin_parity` — `dR, dT <= 1.5e-13` against NumPy on four (basis, m)
  combinations.
* `test_fix_eme_branch_cut.py::test_mode_count_is_build_independent_under_infinitesimal_loss`
  — the `eme_2d` mode count under `eps + i1e-30` equals the real-`eps` count.

**Bit-identity contract.** The ordinary battery — all 148 gates and the >= 30 hashed
fixtures — stays bit-identical. The near-cutoff population is where the numbers MOVE, by
design, and the gates above are what pin the new ones.

---

## 4. Step 3 — a passivity refusal on the legacy nodal cascade

**Scoping section:** 7.3. **Commit:** `fix(bor): ...`

**What is wrong.** `bor_solve.solve` on `basis='nodal'` returns `R + T` from **5.06 to
966.7** on a provably passive lossless stack, and below `Rbig = 4` vacuum wavelengths it
returns it with **no warning at all** (74.03 at 2 wavelengths, 16.38 at 4, 5.06 at 1).
The existing guard is a PROXY — a `UserWarning` when `Rbig / lambda > 4` — and it misses
the smallest two thirds of its own population.

**The decision quantity.** `max(R + T)` over the incident channels, on a stack the solver
can prove passive (a lossless propagating incidence medium). NOT conditioning: the
scoping measured the nodal `inv(a+b)`'s residual at **5.998e-13** against the Cartesian
conjunction's `_INV_RESID_REFUSE = 1e-8`, four to five decades below the bar, so the
conjunction refuses **0 of 6 rows on every kernel**. The operator is not singular; it
amplifies.

**The measured two-sided bar.** `_STACK_SUPERUNITY_BAR`-style, at **1e-2**:

| population | `max(R+T)` | distance to the bar |
|---|---|---|
| staggered (production), all kernels and builds | 1.0000000000005 | **2.6 decades below** |
| nodal, all kernels and builds | 5.06 .. 966.7 | **2.6 decades above** |

Nothing in between, and every nodal row reads the same `max(R+T)` to four significant
figures on all four thread counts per build and on both builds — the nodal blow-up is a
**deterministic discretisation defect, not an arithmetic one**, so a passivity screen
against it is kernel-stable. **Re-measured by this build on its own populations.**

**The switch.** `BOR_NODAL_PASSIVITY_GUARD = True`, in the `PMM_SLIVER_GUARD` style:
`False` restores the pre-fix behaviour bit for bit, including the proxy warning. This is
a behaviour change on a documented escape hatch and gets a switch for that reason.

**The error.** A named error that names `basis='nodal'`, quotes the measured `R + T`, and
names the remedy (`basis='staggered'`, the default). Not a bare `ValueError`.

**The proxy.** The `Rbig / lambda > 4` warning is RETIRED as a decision — the measurement
replaces it. Whether its text survives as documentation is decided in the build; the
decision is that nothing keys on it.

**Tests.** `Rbig` = 1, 2, 4, 8, 14 wavelengths: every nodal row raises; every staggered
twin does not, and returns its pre-fix number bit-identically. The switch set `False`
restores the pre-fix return on every row.

**Bit-identity contract.** Every staggered solve, and every nodal solve under the switch
set `False`. Nodal solves with the switch armed convert a returned wrong number into a
raise — that IS the step.

---

## 5. Step 4 — the SEM manufactured-element contract

**Scoping section:** 7.2. **Commit:** `fix(bor): ...`

**What is wrong.** On the shipped SEM path a wall coincidence of `1e-7` of the domain
radius moves the per-order answer by **586x (degree 12) to 5,428x (degree 8)** the
physical wall shift (the 1-D guard's attribution bar is 100x), injects spurious axial
wavenumbers **5.014e+06** times the physical index ceiling, drives the interface `rcond`
to 6.7e-10, and emits **zero** warnings. The within-layer thin-liner arm shows the same
family with an INVERTED degree ladder (degree 16 is 144x worse than degree 6) from
`w/Rbig <= 1e-06`. A wall-free spacer between two ring layers is not a refuge — it
inherits both wall sets. `basis='fd'` is structurally immune.

**The decision quantity — a CONJUNCTION, neither conjunct the energy.**

* (a) **the geometric cause.** The **post-window, post-DPW, post-`equalize_meshes`**
  breakpoint set contains a cell narrower than `_BOR_MIN_ELEM_FRAC * Rbig` **whose two
  walls come from DIFFERENT layers** — the own-scale test: the union manufactured it, no
  single layer asked for it. It must be the post-everything set because the window, the
  DPW split and `equalize_meshes` each change it; applying a contract to the user's wall
  list would miss the wall-free spacer entirely.
* **AND** (b) **the spurious-wavenumber screen.** That layer's modal spectrum reads
  `|q|max / (n_max k0) > _BOR_Q_EXCESS`.

**The measured bars.**

`_BOR_Q_EXCESS = 1.0e+04` — two-sided on scoping 7.2's populations:

| population | `|q|max / (n_max k0)` |
|---|---|
| ORDINARY — separated control, every rung, degrees 8 and 12 | 19.60 .. 27.22 |
| ORDINARY — a 64-slice taper staircase | **235.8** |
| ORDINARY — ring gratings and segment layers, degrees 6-16 | 16.0 .. 36.5 |
| DAMAGING — the `c5` onset over five `(Rbig, k0)` cases | **2.152e+05 .. 1.003e+07** |
| DAMAGING — the union ladder at `delta/Rbig <= 1e-5` | 2.521e+04 .. 5.014e+06 |

1.63 decades above the worst ordinary, 0.33 decades below the mildest damaging rung of
the union ladder (1.33 below the mildest `c5` onset). Kernel spread **1.1895x**.

`_BOR_MIN_ELEM_FRAC = 1e-6` of `Rbig` — the ATTRIBUTING conjunct, and honestly a
factor-30 quantity: scoping 4.7 measured the onset from 1.000e-07 to 3.000e-06 over five
`(Rbig, k0)` cases and found that **no** scale holds still (`w/Rbig` 30.0x, `w/lambda`
60.0x, `w/h_ordinary` 45.0x). 1e-6 is the conservative (widest) end of the measured band,
so the geometric conjunct fires at or before the onset on every case measured. Kernel
spread **1.0000x — kernel-EXACT**, which is why it attributes rather than decides.

`_BOR_SLIVER_BAND_FRAC` — a `UserWarning` and **NEVER a refusal** in
`[_BOR_MIN_ELEM_FRAC, this)` of `Rbig`. **TO BE MEASURED. The scoping's candidate is
1e-3 and the binding ordinary geometry is the 64-slice taper at 3.9x margin, over four
geometry families against the 2-D peer's 47 — a sample property, not a library property.**
The build therefore runs the **false-positive census FIRST**, over EVERY BOR geometry
family the test suite builds, and including tapers at **8 / 16 / 32 / 64 / 128 / 256
slices** (`|q|max / ceiling` doubles with every doubling of the slice count: 29.7 / 34.4 /
59.3 / 118.1 / 235.8 at 4 / 8 / 16 / 32 / 64), and fixes the warn edge from that census,
on both builds. The edge is whatever leaves the census clear; 1e-3 is a candidate, not a
decision.

**NEVER key a Class-C decision on the energy violation.** Kernel spread of the closure on
this ladder is **70.90x**, straddling the 1-D `_SLIVER_TRIGGER_BAR` of 1e-3 — the same
solve would be arbitrated on one kernel and pass silently on another. Independently, the
damage is **energy-invisible** over four decades: at `delta/Rbig = 1e-04` the closure sits
at its healthy 1.04e-09 baseline while the per-order answer has already moved 4.00e-04.
A ported super-unity screen at the 1-D bars fires on 4 of 60 union rungs and 2 of 40
within-layer rungs — the catastrophic tail only.

**The switch.** `BOR_SEM_MESH_GUARD = True`, `PMM_SLIVER_GUARD` style.

**Tests.**

* `...::test_ordinary_geometry_census` — every BOR geometry family the suite builds, plus
  tapers at 8/16/32/64/128/256 slices, lands outside the warn band on the running build.
  This is the gate that makes the warn edge a library property.
* `...::test_delta_ladder_decisions` — `delta/Rbig` from 1e-1 down to 1e-7 at degrees
  6 / 8 / 12, each rung decided (correct / warned / refused) **identically under both
  kernel ladders on both builds**.
* `...::test_within_layer_liner_ladder` — the `c2` surface, same treatment.
* `...::test_wall_free_spacer_inherits_walls` — pins the window's reach, so a future
  change to `win[i]` is caught.
* `...::test_fd_basis_is_immune` — the FD arm is unchanged by any `delta`.
* `...::test_no_class_c_decision_reads_the_energy` — an AST/grep pin that neither conjunct
  is computed from `R + T`.

**Bit-identity contract.** Bit-identical everywhere the contract does not fire — all 148
gates, the >= 30 hashed fixtures, the FD basis at every `delta`. Where it fires it
converts a returned wrong number into a raise or attaches a warning to an unchanged
number; the census is what proves the first set is not accidentally in the second.

---

## 6. Step 5 — census hooks on the production inverses

**Scoping section:** 7.5. **Commit:** `fix(bor): ...`

**What is NOT wrong.** The production FD and SEM cascades have **ONE** inverse population:
2,031 inverses over 132 fixtures, equilibrated `rcond` **1.986e-07 .. 1.000**, residual at
most **3.408e-13** — three to seven decades clear of every Cartesian bar
(`_INV_T22_RCOND_REFUSE = 1e-10`, `_MORTAR_RCOND_REFUSE = 1e-12`,
`_MORTAR_RESID_REFUSE = 1e-6`). There is no second population and therefore no bar. An
armed refusal here would be dormant on every fixture measured, which is not a guard but a
liability.

**The decision quantity.** None. This step arms NOTHING. It installs the measurement
instrument so the population can be re-measured rather than assumed.

**The shape.** `_BOR_INV_CENSUS = None` at module level, the `rcwa/_core._INV_CENSUS`
pattern: `None` by default, costing one `is None` test per inverse; when a list is
assigned it records `(site, n, rcond, residual)` per call and never raises.

**Tests.** One `is None` test per inverse site (the hook is present and dormant), plus a
probe that re-measures the 2,031-inverse population through the hook and reproduces the
scoping's table.

**Bit-identity contract.** TOTAL, on every path, both builds. A `None` check changes no
number.

`sem_radial.py:348`'s `inv(Mz)` already carries a correct and DIFFERENT remedy (LU-pivot
detection with a fallback to the unreduced QZ pencil — repair, not refusal). Nothing is
ported there.

---

## 7. Order of work, and why this order

1. **Step 1** — the consolidation with the band unchanged. A pure refactor, 148 gates
   bit-identical. This is the step that makes step 2 a ONE-LINE change, and it is first so
   that the bit-identity claim is made against the shipped numbers before any bar moves.
2. **Step 2** — the band change, plus the EME pins. Same class, same wave, but two commits
   because they are two modules with two populations.
3. **Step 3** — the nodal passivity refusal. Self-contained, on a legacy path, behind a
   switch; independent of 1, 2, 4 and 5.
4. **Step 4** — the SEM contract, **with the census FIRST**, because the census is the
   binding constraint on the warn edge and must be measured on the shipped battery before
   the edge is fixed. This is exactly the round-4 correction the 2-D contract needed
   ("the census margin is sample-scoped, and it is 1.67x, not 3.6x").
5. **Step 5** — the census hooks. Last because they change nothing and because the
   populations they instrument are the ones steps 1-4 have by then left alone.

**Runs.** Every BOR and EME test file on both builds under both ladders — kernels in
{HASWELL, NEHALEM, KATMAI} (ZEN silently aliases to HASWELL on this host and SKYLAKEX
crashes; each confirmed by reading `threadpoolctl.threadpool_info()` back, never assumed)
and threads in {1, 2, 4, unpinned} — reported as a table of file x kernel x threads x
build. Plus the census/walker/dispatcher sweep and WSL `ruff`.

---

## 8. Out of scope for 5.45.1, recorded as open items

Both carry reproducers so a later wave does not have to re-find them.

* **PML and anisotropic-path re-characterisation.** `SemRadialMesh(R_pml=...)` is
  reachable only by direct construction — `BORStack._solve_sem` never sets it — and the
  Class-C ladders were measured on the non-PML, isotropic path only. Whether the
  manufactured-element contract's populations look the same under a PML (whose complex
  coordinate stretch changes `|q|max` by construction) is **not measured** and the
  contract makes no claim there.
* **The `equalize_meshes` dropped-kwargs latent bug.** `equalize_meshes` rebuilds every
  mesh as `SemRadialMesh(b, eps, msh.p)` — **dropping `R_pml`, `sigma_max`, `pml_p` and
  `nq_extra`**. A caller who builds PML meshes by hand and equalizes them silently loses
  the PML. Not reachable through `BORStack` today (which never sets them), which is why
  it is an open item and not a step; the reproducer is two lines and is recorded in the
  build doc.

Also **not** in scope, and recorded by the scoping as unmeasured rather than as decided:
the CI's ZEN kernel row (unreachable on this host — `OPENBLAS_CORETYPE=ZEN` returns
Haswell bit-identically and `SKYLAKEX` dies on the first LAPACK call), which of `Rbig`,
the local wavelength or the layer's own ordinary element width governs the Class-C width
scale (none holds still to better than 30x), an armed refusal bar for the EME class-B
inverses (no two-sided gap exists — the separator is distance-to-band-edge, a continuum),
and an external accuracy oracle for the SEM answer (Class C's wrongness is established by
CONTINUITY and by the separated-arm control, not against an oracle).
