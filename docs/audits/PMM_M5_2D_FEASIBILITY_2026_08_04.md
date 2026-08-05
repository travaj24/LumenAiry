# M5 -- TWO SPIKES: non-uniform 2-D segments (N-1) and the covariant taper (T3-6)

**Date:** 2026-08-04 - **Branch:** `feat/pmm-per-layer-roadmap` (M1 at `d30f1ca`)
**Parent:** `docs/audits/PMM_PER_LAYER_CAMPAIGN_PLAN_2026_08_04.md`, sections S-1/S-2/S-3, item
dossiers **N-1** and **T3-6**, mission **M5**
**Status:** ANALYSIS ONLY. No file under `lumenairy/` or `tests/` was modified. The three
prototypes live in `validation/m5_derham_nonuniform.py`, `validation/m5_covariant_taper.py` and
`validation/m5_taper_degree_spread.py`, and import the library read-only.
**Decides:** whether 2-D per-layer becomes the 5.34 campaign, and in what shape.

---

## 0. How to read this

| tag | meaning |
|---|---|
| **[M]** | **Measured** -- produced by a run of one of the three `m5_` scripts, on BOTH builds |
| **[A]** | **Analysis** -- arithmetic or derivation, no run |
| **[H]** | **Hypothesis** -- consistent with the evidence, NOT established |

**The two builds.** Every numeric claim below was produced on both:

| | interpreter | numpy | scipy | BLAS |
|---|---|---|---|---|
| **W** (Windows 11) | CPython 3.14.6 | 2.4.4 | 1.17.1 | scipy-openblas 0.3.31.188 (win_amd64, MSVC) |
| **L** (WSL2 Ubuntu) | CPython 3.12.3 | 2.4.6 | 1.17.1 | scipy-openblas 0.3.31.188 (linux, gcc) |

**[A] Protocol note the campaign plan needs.** Standing rule 3 says "Windows/MKL + WSL/OpenBLAS".
That is **no longer what the tree provides**: both environments now ship **scipy-openblas**, from
the same OpenBLAS 0.3.31.188 source, differing only in compiler/platform/interpreter. The dual
build is still a real cross-check (different LAPACK kernel dispatch, different interpreter, and it
caught the C13 class historically), but it is **weaker than the rule assumes**, and no mission
should describe it as "MKL vs OpenBLAS" until an MKL environment is actually installed. Recorded
as a correction, not a blocker.

**[A] Tree provenance -- read this before re-running.** M5 ran **concurrently with M1/M2/M4**, on a
working tree those missions were editing. `git HEAD` was `d30f1ca` throughout, but
`lumenairy/elements/pmm/{_core,stack,conical}.py` and `lumenairy/elements/rcwa/{_core,stack}.py`
carried uncommitted changes that moved during the session. Consequences, checked:

* **Spike 1 is unaffected.** `twod_staggered.py` -- the only file it modifies the behaviour of --
  was untouched for the whole session, and the W and L runs (hours apart, different tree states)
  are bit-identical in every reported digit.
* **S4 was re-run end to end against the tree AFTER M1/M2/M4's last edit** (`git diff --stat`:
  12 files, +933/-183 over `d30f1ca`) and **reproduces cell for cell** -- all 80 PMM cells, the
  9-cell RCWA arbiter row, and every summary line, bit-identical to the first run. The defect is
  therefore not an artefact of a half-applied concurrent edit.
* Anyone re-running should record `git diff --stat` alongside the results; a bare `HEAD` is not a
  sufficient pin while the campaign's missions overlap. **[A] Worth adopting as a campaign rule.**

**Reproduce**

```
python validation/m5_derham_nonuniform.py --json out1.json   # spike 1,  ~25 min (W) / ~45 (L)
python validation/m5_covariant_taper.py   --json out2.json   # spike 2,  ~35 min
python validation/m5_taper_degree_spread.py out3.json         # the S4 finding, ~35 min
# the first two take --quick for a ~5 min smoke pass
```

---

## 1. Verdict, up front

| spike | verdict | the one measurement that decides it |
|---|---|---|
| **N-1** non-uniform segments in the 2-D `Basis1D` | **GO -- unconditional** | de Rham residual `d(Btilde) subset span(B)` is **8.925e-15 on BOTH builds (bit-identical)** at segment ratios up to **1e6**, identical to the uniform control; the C0 hat/Bloch-seam residual is **exactly 0.0** |
| **T3-6** covariant taper, first order in `tan(phi)` | **CONDITIONAL GO as research; NO for 5.34** | the derivation is validated to **1e-15** against the shipped slant pencil and its residual is **O(delta^2)** where the staircase is **O(delta)** -- but the frozen pencil is **non-normal with a `q -> -conj(q)` symmetry**, so no shipped forward/backward selector classifies its modes, and the prototype cascade **diverges** without one |

**And a finding neither spike went looking for, which outranks both:**

> **[M] `PMMStack` returns SILENTLY WRONG answers on a plain 2 deg-tapered lossless dielectric
> grating at in-plane oblique incidence, on BOTH `layer_grids` modes, scattered across
> `(degree, n_slice)` with NO monotone structure, with `|R+T-1| <= 1e-6` and the energy guard
> firing in only 1 of the 26 wrong cells.** Excursions of **-93 % to +10x** in the zeroth-order
> reflectance, adjudicated
> against an RCWA twin on the identical geometry that moves by **5.6e-05 in total** over the same
> `n_slice` range. Raising `degree` makes it worse, not better. This lands squarely inside
> **M2 / T3-2**, whose 1800-solve mesh sweep is scheduled over exactly the affected band, and it
> voids **T3-6**'s named oracle. See S4.

---

## 2. SPIKE 1 (N-1) -- non-uniform segments in the 2-D staggered basis

### 2.1 What the gate actually is

The plan (S-1) is right that `Basis1D.__init__` hard-codes `xb = np.linspace(...)`
(`twod_staggered.py:174`) with a scalar jacobian `J = 0.5*d/N` (`twod_staggered.py:172`), and right
that this -- not interface non-conformity -- is what blocks a tapered pillar in 2-D.

The plan calls the fix "a per-segment weight vector in place of a scalar, **in two functions**".
**[A] It is six scale sites in four functions, plus the boundary array:**

| site | what it is |
|---|---|
| `twod_staggered.py:262` / `:264` | `_global_matrix`: `scale = J` (mass) / `1/J` (stiffness) |
| `twod_staggered.py:342` / `:344` | `_global_pair_segmat`: same two, duplicated |
| `twod_staggered.py:567` | `_eps_dir.segmat`: `scale = basis.J` (mass branch) |
| `twod_staggered.py:641` + `:618`, `:643` | `_stag_fourier_projection`: `J = basis.J` and the uniform midpoint `0.5*(xb[s]+xb[s+1])` |
| `twod_staggered.py:171-174` | `h`, `J`, `xb` themselves |

`_build_elementary` and `_build_sets` need **no change at all** -- they are defined purely on the
reference interval and never see a jacobian. That is the structural reason the change is small,
and it is also the reason the de Rham gate passes (S2.2).

### 2.2 [A] Why the de Rham property is *width-independent* -- the argument, before the measurement

The mimetic placement (`twod_staggered.py:467-477`) rests on `d(Btilde) subset span(B)`. Both sets
are built from the same per-segment modified-Legendre functions:

* On any segment, `span{Ltilde_0 .. Ltilde_{M-1}} = P_{M-1}` (degree <= M-1). `Btilde`'s
  restriction to a segment therefore lies in `P_{M-1}`.
* `B` is the **broken** set: it drops the last bubble per segment and glues nothing, so
  `span(B|_seg) = span{Ltilde_0 .. Ltilde_{M-2}} = P_{M-2}` and
  `span(B) = (+) over segments of P_{M-2}(seg)`.
* `d/dx = (1/J_s) d/du` maps `P_{M-1} -> P_{M-2}` on each segment. **`J_s` is a positive scalar; a
  scalar cannot move a function out of a span.**

So the property is a statement about *polynomial degree per segment*, and the segment widths enter
only as per-segment positive scalars. **It cannot fail under non-uniform segmentation** -- and the
same argument shows why: the 1-D per-layer basis already does non-uniform elements for exactly
this reason.

The C0 gate is equally structural. The hats glue `Ltilde_2` of segment `n-1` to `Ltilde_1` of
segment `n` (`twod_staggered.py:210-221`). `Ltilde_1(u) = (1-u)/2` and `Ltilde_2(u) = (1+u)/2`, so
each reaches the value **1 at the shared node in the REFERENCE coordinate**, independent of the
segment width. The glue coefficient stays 1 (and `tau` at the Bloch seam) for **any** partition.
The derivative jumps by `1/h_left` vs `1/h_right` -- which is fine, because `B` is the
discontinuous partner and is where the derivative is required to land.

### 2.3 [M] The measurements

Prototype: `validation/m5_derham_nonuniform.py`, class `Basis1DNU` (a copy of `Basis1D` with
`xb` arbitrary and `Jv = 0.5*diff(xb)`), plus `Granet2DTransverseENU` and a full non-uniform
mirror of `pmm_efficiency_2d_staggered`.

#### G1/G2 -- de Rham and C0, over 80 (N, M, partition) cases

Residual measured as `min_c || d(Btilde_j)/dx - sum_k c_k B_k ||_2 / || d(Btilde_j)/dx ||_2`,
worst over j, on a Gauss grid with physical weights. N in {2,3,6,11}, M in {3,4,8,12}, geometric
partitions with `h_max/h_min` in {1, 2, 10, 1e3, 1e6}.

| quantity | W | L |
|---|---|---|
| **worst de Rham residual, all cases** | **8.925e-15** | **8.925e-15** |
| worst de Rham residual, uniform control only | 3.16e-15 | 3.16e-15 |
| **CONVERSE control** `min_j` of `d(B) -> span(Btilde)` residual (must be O(1) or the test is vacuous) | **3.125e-01** | **3.125e-01** |
| **worst C0 residual** (interior node) | **0.0** | **0.0** |
| **worst C0 residual** (Bloch seam, incl. `tau`) | **0.0** | **0.0** |

Every G1/G2 number is **bit-identical across the two builds** (the residual is a property of the
basis; no ill-conditioned solve is involved, which is exactly why it is the right gate).

The residual is **flat in the segment ratio** -- 4.5e-16 at ratio 1, 5.1e-16 at ratio 1e6 for
`N=2, M=3`; it tracks `M` (round-off accumulation), not the partition. **This is the go/no-go
number the plan asked for, and it is a pass by 15 orders of magnitude.**

#### G3 -- uniform-boundary byte-identity

| construction | worst `max\|diff\|` vs the library, over 8 operator pairs + eps-weighted mass + the full 2-D `Lmat`/`Rmat`/`Stt`/`Schur` |
|---|---|
| `xb = np.linspace(0, d, N+1)`, `Jv = 0.5*diff(xb)` | **5.684e-14** on both builds -- NOT identical |
| uniform fast path: `Jv := 0.5*(d/N)` exactly | **0.0** on both builds -- **BYTE-IDENTICAL** |

**[A] Implementation requirement, and a correction to N-1's oracle (b).** The plan's regression
gate is "with equal segments the new code must be byte-identical to today's". That is achievable
but **only if the implementation keeps a uniform fast path that computes `h = d/N` the way the
library does today**. `np.diff(np.linspace(0, d, N+1))` differs from `d/N` in the last bit, and the
stiffness scale `1/J` amplifies it to 5.7e-14. Either mandate the fast path (recommended -- it also
saves the per-segment vector on the overwhelmingly common uniform call) or restate the gate at
tolerance 1e-13. Note the mixed matrices (`Ctb`, `Cbt`, `Ctt`) are **already** identical at 0.0
under either construction, because their scale is 1 on every segment -- the `J`/`1/J` cancellation
the plan flags as "must be re-derived rather than assumed" is **per segment and survives**,
confirmed both analytically (S2.2) and by this 0.0.

#### G4 -- conditioning vs segment-length ratio (`N=3, M=6`, geometric partition; W and L identical to 4 s.f.)

| `h_max/h_min` | cond(Mtt) | cond(Mbb) | cond(G) of the 2-D pencil | cond(Gw) |
|---|---|---|---|---|
| 1 | 8.93e+01 | 8.20e+01 | 7.23e+03 | 6.72e+03 |
| 2 | 1.18e+02 | 1.64e+02 | 1.92e+04 | 2.69e+04 |
| 10 | 3.25e+02 | 8.20e+02 | 2.67e+05 | 6.72e+05 |
| 100 | 1.43e+03 | 8.20e+03 | 1.17e+07 | 6.72e+07 |
| 1e3 | 1.04e+04 | 8.20e+04 | 8.55e+08 | 6.72e+09 |
| 1e4 | 1.00e+05 | 8.20e+05 | 8.22e+10 | 6.72e+11 |

**[A] Law:** the per-axis Grams grow **linearly** in the ratio; the 2-D field Gram is their
Kronecker product, so it grows **quadratically**. **[A] Budget for the target device:** a per-layer
window on the 2 deg taper has segments of 1.8 nm (the per-slice wall offset) next to ~175-350 nm,
i.e. ratio ~100-200 -> `cond(G) ~ 1e7-5e7`, comfortably inside double precision. A *shared* grid at
`ns = 12` reaches ratio ~390 -> `cond(G) ~ 1e8`. **Recommendation: cap the ratio at 1e3 with a
warning** (`cond(G) ~ 1e9`, still 7 decades of headroom) rather than leaving it unbounded; and
route the new Grams through M1's conditioning census when N-1 is implemented.

#### G5 -- convergence against an EXACT analytic oracle (1-D lamellar Bloch dispersion)

TE pencil on the C0 set, eigenvalues checked against the transcendental
`cos(bloch) = cos(k1 a)cos(k2 b) - (1/2)(k1/k2 + k2/k1) sin(k1 a) sin(k2 b)` by bisection --
**no solver shared with the thing under test.** Period 0.7 um, eps 4/1, wl 1.31 um, Bloch phase
`0.37 * 2pi`; relative error of the 4 largest `gamma^2`.

| ridge duty | M | NON-UNIFORM `N=2` (walls exact) | UNIFORM at the **same dof** | UNIFORM at the smallest N that can place the wall |
|---|---|---|---|---|
| 0.500 | 8 | 3.25e-06 (dof 14) | 3.25e-06 (`N=2`) | 3.25e-06 (`N=2`, dof 14) |
| 0.500 | 16 | 1.91e-13 (dof 30) | 1.91e-13 | 1.91e-13 (dof 30) |
| **0.400** | 8 | **3.51e-06** (dof 14) | **1.01e+00** (`N=2`) | 1.03e-11 (`N=5`, dof 35) |
| **0.400** | 16 | **5.67e-13** (dof 30) | **1.01e+00** | 8.97e-12 (`N=5`, dof 75) |
| **0.371** | 8 | **7.09e-06** (dof 14) | **1.63e+00** (`N=2`) | needs **`N=1000`** |
| **0.371** | 16 | **1.80e-12** (dof 30) | **1.63e+00** | needs **`N=1000`** |

Read the two right-hand columns as the two things a uniform lattice can do with a wall it cannot
represent: **put it in the wrong place** (error 100-163 %, and it does not improve with `M`), or
**refine until the lattice contains it** (`N = ` the denominator of the duty fraction; 1000 here).
The non-uniform basis reaches **1e-12 at `N=2`** for every duty, with spectral convergence in `M`.

#### G6 -- the full 2-D staggered solve on a non-uniform grid

(a) **Commensurate walls (0.25 .. 0.75), so the library's uniform grid CAN represent them.**
Library `Nx=Ny=4` vs prototype `Nx=Ny=3` (`xb = [0, 0.25, 0.75, 1] * P`), same geometry:

| M | library `N=4` T0 (dof) | time | prototype `N=3` T0 (dof) | time | `\|dT0\|` | `\|R+T-1\|` lib / NU |
|---|---|---|---|---|---|---|
| 4 | 0.970774110 (288) | 2.6 s | 0.970710112 (162) | 0.25 s | 6.4e-05 | 4.7e-15 / 4.4e-16 |
| 5 | 0.970841200 (512) | 7.7 s | 0.970801195 (288) | 1.6 s | 4.0e-05 | 1.7e-14 / 2.0e-15 |
| 6 | 0.970867239 (800) | 30.3 s | 0.970849609 (450) | 6.2 s | 1.8e-05 | 9.7e-15 / 5.3e-15 |
| 7 | 0.970879354 (1152) | 89.7 s | 0.970867540 (648) | 15.4 s | 1.2e-05 | 7.3e-15 / 5.7e-15 |

The two grids converge **to each other** (6.4e-5 -> 1.2e-5 monotonically), energy closes at
machine precision on both, and the non-uniform grid does it at **0.56x the dof and 0.17x the time**
because it does not need the redundant interior wall at 0.5.

(b) **Unrepresentable walls (0.317 .. 0.688 of the period), vs RCWA** -- the case a uniform
staggered grid cannot express at all. RCWA raster verified to realise the walls **exactly**
(`0.317000 .. 0.688000` on a 1000-cell raster).

| solver | setting | T0 | `\|R+T-1\|` |
|---|---|---|---|
| RCWA | n_orders 9 / 13 / 17 / 21 / 25 | 0.989130 / 0.989396 / 0.989620 / 0.989712 / 0.989807 | -- |
| **prototype NU staggered `N=3`** | M = 4 / 5 / 6 / 7 / 8 / 9 | 0.990113 / 0.990173 / 0.990199 / 0.990210 / 0.990216 / 0.990219 | <= 3.8e-14 |
| **NULL CONTROL: library uniform `N=3`, walls SNAPPED to 1/3, 2/3** (wall error 11.4 / 14.9 nm) | M = 5 / 6 / 7 / 8 | 0.993616 / 0.993636 / 0.993644 / 0.993649 | -- |

RCWA is still climbing monotonically toward the staggered value (its own Fourier truncation);
the residual gap at the largest RCWA order set is **4.1e-4**, and it is **shrinking with RCWA
orders**, i.e. the two independent methods are converging on the same number. The null control --
"just snap the walls onto a uniform grid and accept it", which is the alternative to N-1 -- sits
**3.8e-3** away, an order of magnitude worse, and is **stationary in M** (a wrong geometry solved
ever more precisely).

(c) **Position invariance** -- a property test with no oracle at all. Slide a width-0.371 pillar
through the cell (`x0` = 0.0500 / 0.1130 / 0.2405 / 0.4000), which with non-uniform segments is a
*strictly stronger* test than the library's uniform-grid version because the pillar can sit
anywhere:

```
T0 = 0.990188775421 / 0.990192980690 / 0.990197851491 / 0.990197575367
R+T = 1.000000000000 at every position
T0 spread = 9.076e-06       (M = 6, dof 450)
```

**[M] Cross-build for the whole of G6.** Every `T0` in G6(a), G6(b) and G6(c) agrees between W and
L **to all 9-12 printed digits** -- library and prototype alike. The only cross-build differences
anywhere in Spike 1 are in `|R+T-1|` (1e-15-class round-off) and wall-clock time (L is 2-3x slower
on this box). Nothing in N-1 is BLAS-sensitive, which is what one wants from a basis change.

### 2.4 [A] The arithmetic -- and the reframing of S1 it forces

Audit device: period 700 nm, sidewall 2 deg, region-1 thickness 310 nm.

| `n_slice` | per-slice wall offset | `Nx` a UNIFORM lattice needs | `Nx` with NON-UNIFORM segments, shared grid (`2*ns` walls) | `Nx` with NON-UNIFORM segments, +/-1 per-layer window (6 walls) |
|---|---|---|---|---|
| 2 | 5.413 nm | 130 | 4 | 6 |
| 4 | 2.706 nm | 259 | 8 | 6 |
| **6** | **1.804 nm** | **388** | **12** | **6** |
| 8 | 1.353 nm | 518 | 16 | 6 |
| 12 | 0.902 nm | 776 | 24 | 6 |

Cost at `M = 8` (`n = 2*(Nx*(M-1))^2`, `16 n^2` bytes/matrix, `~30 n^3` flop) -- the plan's S-3
table, reproduced and extended:

| `Nx` | eig dim | GB / matrix | flop | what it is |
|---|---|---|---|---|
| 6 | 3 528 | **0.186** | 1.3e12 | **NU per-layer window** |
| 7 | 4 802 | 0.344 | 3.3e12 | the plan's per-layer estimate |
| 12 | 14 112 | **2.97** | 8.4e13 | **NU shared grid at `ns=6`** |
| 13 | 16 562 | 4.09 | 1.4e14 | the plan's shared-grid estimate |
| 25 | 61 250 | 55.9 | 6.9e15 | the plan's upper shared-grid estimate |
| **388** | **14 753 312** | **3.24e+06** | 9.6e22 | **uniform lattice at `ns=6` -- impossible, as the plan says** |

**[A] This changes the shape of the 5.34 campaign.** The plan's S-1 already reaches the right
qualitative conclusion -- *"a mortar alone does not unlock 2-D tapers; the enabling change is
non-uniform segment boundaries, and the mortar is what makes the resulting per-layer grids
affordable"* -- and already notes that N-5 is separable. What was missing is the arithmetic, and
the arithmetic says the mortar is not needed for feasibility at all:

* **N-1 alone gets the device to a runnable size.** With non-uniform segments a 2-D `ns=6` taper is
  representable on a **shared** grid at `Nx = 12`, i.e. 2.97 GB/matrix -- heavy, but the same class
  as things this library already runs, and **no mortar is involved.** Without N-1 it is
  `Nx = 388` and 3.2 PB: not a cost problem, an impossibility.
* **S1 (the 2-D per-layer mortar) is then a ~16x memory optimisation** (`Nx 12 -> 6`,
  2.97 -> 0.186 GB) at **8-12 AC**.
* **N-5 (the C2/C4 fold) buys the same 16x for 3 AC** (`/16` memory, `/64` flop), needs no mortar,
  and carries none of the mortar's three named physics hazards (different cross-mass operators per
  component, the squareness argument at `q_a != q_b`, and the curve-set interface residual).
  `Nx = 12` with a C4 fold is **0.186 GB** -- exactly the per-layer-window figure. The plan defers
  N-5 *with* S1; on this arithmetic it should be **promoted above** it.

### 2.5 Verdict and recommendation -- N-1

**GO, unconditional.** Every gate in the plan's N-1 dossier passes, and the two the plan flagged as
the risk (the `J`/`1/J` cancellation in the mixed matrix, and the Bloch hat's C0 property) pass
*structurally*, at 0.0, not merely numerically.

Recommended sequencing for 5.34, in the order that maximises capability per AC:

1. **N-1** (2 AC as the plan estimates). Ship with a uniform fast path so the byte-identity gate is
   0.0, and route the new Grams through M1's conditioning census with a ratio cap.
2. **N-5** (C2/C4 fold, 3 AC). Independent of the mortar; on the C4 device it is the difference
   between a 3 GB solve and a 0.2 GB solve. **Promote it above S1.**
3. **T3-7** (lattice quantisation, 1 AC) -- the plan already spots the synergy; with N-1 it becomes
   the mechanism that keeps the segment ratio (and therefore `cond(G)`, S2.3/G4) bounded. It is now
   a *conditioning* item, not only a reproducibility one.
4. **S1** (the 2-D mortar) -- keep the Kronecker-cross-mass requirement, but re-scope it as a
   **memory optimisation with a measured budget**, to be taken only if 1-3 leave the target device
   short. Its 8-12 AC buys 16x memory that N-5 also buys for 3 AC.

---

## 3. SPIKE 2 (T3-6) -- first order in `tan(phi)` for a taper

### 3.1 [M] First, the geometric fact the whole item turns on

A wall moving linearly with depth decomposes **exactly** into two independent parts:

* the **centre walk** (`shear`): both walls translate together. `u = x - z tan(phi)` absorbs it
  **exactly, at any magnitude** -- the metric is z-invariant, which is why `add_sheared_grating`
  ships as ONE layer.
* the **duty change** (dilation): the walls separate. **No shear absorbs any of it.**

The plan and the parent audit both state this ("a shear can absorb a translation, not a dilation").
It had never been measured. Measured now, on the audit device (2 deg, RCWA oracle at `ns = 192`):

| geometry | best a SINGLE vertical mid-width layer can do | best a SINGLE exact PARALLELOGRAM can do (shear scanned over 13 values) | **gain from absorbing the shear** |
|---|---|---|---|
| symmetric taper (centre walk **0**) | 9.139e-03 | 9.139e-03 at `shear = -1.0e-4` | **1.00x** |
| one-wall taper (centre walk `dd/2`, the MAXIMUM shear content a taper can have) | 9.138e-03 | 9.138e-03 at `shear = -0.0233` | **1.00x** |

**[M] The shear machinery contributes exactly nothing to a taper, even to the taper with the most
shear content available.** The optimum over `shear` is flat to within the taper error, and it does
not sit at the geometry's own shear. Any T3-6 design that reaches for the slant path *as a slant*
is dead on arrival. What follows is therefore not "the slant at small angle" -- it is a different
operator that happens to share the slant's algebraic shape.

### 3.2 [M] The staircase ladder -- what `ns = 1` (the first-order answer) actually costs

Oracle: **RCWA** (`RCWAStack.add_tapered_grating`, `raster='area'`, `n_orders=21`, `n_x=4096`),
reference `ns = 384`, self-consistency vs `ns = 192` **4.42e-07**. The Fourier error is
common-mode across `ns`, so this isolates the **staircase**. Score = worst absolute move over all
order/pol efficiencies **and** the complex zeroth-order reflection+transmission Jones.

| `ns` | max `\|dR\|` | max `\|dT\|` | **max `\|dJones\|`** | ratio vs previous rung |
|---|---|---|---|---|
| **1** | 3.57e-05 | 3.57e-05 | **9.139e-03** | -- |
| 2 | 5.75e-05 | 5.76e-05 | 2.794e-03 | 3.27 |
| 4 | 2.09e-05 | 2.10e-05 | 8.825e-04 | 3.17 |
| 8 | 6.47e-06 | 6.52e-06 | 2.681e-04 | 3.29 |
| 16 | 1.78e-06 | 1.83e-06 | 7.588e-05 | 3.53 |
| 32 | 4.19e-07 | 4.68e-07 | 1.995e-05 | 3.80 |
| 64 | 1.77e-07 | 2.28e-07 | 5.043e-06 | 3.96 |

Measured order over `ns = 8 -> 64`: **1.91** -- consistent with the `O(1/ns^2)` law already pinned
in `add_tapered_grating`'s docstring. On a multi-order design (P=1000 nm, wl=633 nm, normal
incidence, orders 0,+/-1 propagating) the same ladder gives `ns=1` -> **1.97e-02** and a clean
**3.86 / 3.97** per doubling at `ns` 8->16->32.

**[M] Two things worth carrying into every taper discussion.** (i) The **efficiencies barely move**
(3.6e-05 at `ns=1`) while the **Jones moves 9.1e-03** -- a factor **256**. The parent audit's
"deep-null figures of merit are the sensitive observable" is quantified here: on this device class
an efficiency-only convergence table is blind to the staircase by more than two decades.
(ii) The PMM and RCWA answers differ by a *constant* 7.5e-04 in R/T at matched `ns` (`ns` = 1/2/4),
which is RCWA's Fourier floor against PMM's exact walls -- common-mode, so the ladder is sound, but
it is also the reason RCWA cannot be the oracle for an *absolute* PMM taper number.

### 3.3 [M] Angle sweep -- the staircase's `ns=1` error is FIRST order in `tan(phi)`

| sidewall | wall motion | duty change `dd` | **err at ns=1** | ns=2 | ns=4 | ns=8 | `e1/dd^2` |
|---|---|---|---|---|---|---|---|
| 0.25 deg | 1.353 nm | 0.00386 | 1.148e-03 | 3.54e-04 | 1.11e-04 | 3.34e-05 | 7.69e+01 |
| 0.50 deg | 2.705 nm | 0.00773 | 2.296e-03 | 7.07e-04 | 2.21e-04 | 6.66e-05 | 3.84e+01 |
| 1.00 deg | 5.411 nm | 0.01546 | 4.586e-03 | 1.41e-03 | 4.42e-04 | 1.33e-04 | 1.92e+01 |
| **2.00 deg** | 10.825 nm | 0.03093 | **9.139e-03** | 2.79e-03 | 8.82e-04 | 2.68e-04 | 9.55e+00 |
| 4.00 deg | 21.677 nm | 0.06194 | 1.814e-02 | 5.45e-03 | 1.75e-03 | 5.35e-04 | 4.73e+00 |
| 8.00 deg | 43.568 nm | 0.12448 | 3.640e-02 | 1.06e-02 | 3.38e-03 | 1.05e-03 | 2.35e+00 |

`err(ns=1)` **doubles exactly with the angle** across 5 octaves (`e1/dd` is constant to 0.5 %;
`e1/dd^2` falls by 32x, i.e. it is not quadratic). **The staircase's one-slice error is `O(delta)`,
`delta = h tan(phi)/w0`.** That is the bar any first-order treatment has to beat, and it is the
right bar precisely *because* a symmetric taper's shear content is zero (S3.1) -- `ns=1` **is** the
best any shear-only or naive first-order treatment can do.

### 3.4 [A + M] The derivation, and its validation against shipped code

Mid-depth reference frame: `x = X(u, z) = u + (z - h/2) S(u)`, with `S` piecewise linear through
the wall velocities (`S(0) = S(d) = 0` keeps the cell fixed; `S = const` is a pure shear). Then
`X_z = S(u)` exactly, `X_u = 1 + (z - h/2) S'(u)` (piecewise **constant** in `u`), and the exact
inverse metric is `G^uu = (1+S^2)/X_u^2`, `G^uz = -S/X_u`, `G^zz = 1`, `sqrt(G) = X_u`.

Substituting `E = phi(u) exp(i q k0 z)` into `(1/sqrt G) d_i (sqrt G G^ij d_j E) + k0^2 eps E = 0`
gives the quadratic pencil `A1 phi - q Ac phi - q^2 A2 phi = 0` with

```
A1 = <v| eps X_u |phi>  -  (1/k0^2) <v'| (1 + S^2)/X_u |phi'>
Ac = (2 i / k0) <v| S |phi'>                          <-- z-FREE, EXACTLY
A2 = <v| X_u |phi>
```

**Three consequences, and one trap.**

1. **[A] `Ac` is exactly z-independent.** `b/g = X_z = S(u)` carries no `z`. The *entire*
   z-dependence of the exact covariant operator is the single scalar `X_u(z)` per element.
2. **[A] Freezing `X_u` is exactly the lab slab at that depth** (an element of width `w` with mass
   `x X_u` and stiffness `/ X_u` *is* an element of width `w X_u`). So the covariant frame does not
   invent a new discretisation -- it adds `Ac` to the staircase and fixes the grid.
3. **[A] This is the shipped slant pencil with the scalar `tan(phi)` promoted to a field.** With
   `S = const`, `S' = 0`, `X_u == 1` at every `z`, and the pencil collapses to
   `_sem_modes_slant`'s TE form (`sec^2 = 1 + t^2`, `Ac = (2 i t/k0) C`, `_core.py:3843-3847`).
4. **THE TRAP.** `d_z(sqrt G G^zz d_z E) = X_u d_z^2 E + S'(u) d_z E` contributes a **third**
   q-linear piece `-(i/k0)<v|S'|phi>`. Dropping it leaves the *antisymmetrised*
   `(i/k0)(<v|S|phi'> - <v'|S|phi>)`, which is **identical for constant `S`** -- so it **passes the
   shear reduction test** -- and wrong by exactly `<v|S'|phi>` for a taper. This spike wrote it
   wrong on the first pass and the shear test did not catch it. **Any T3-6 implementation must
   carry a taper-specific operator test; the slant-limit test is necessary and not sufficient.**

**[M] B1 -- the validation.** Prototype pencil with constant `S`, against the shipped
`_build_sem_slant` / `_sem_modes_slant` TE operators, slant 0 / 2 / 10 / 20 / 45 deg x degree
6 / 8 / 12:

| quantity | worst over all 15 cases, W | worst over all 15 cases, L |
|---|---|---|
| `max\|A1 - A1_lib\|` | 3.55e-15 | 3.55e-15 |
| `max\|Ac - Ac_lib\|` | 5.55e-17 | 5.55e-17 |
| `max\|A2 - A2_lib\|` | **0.0** | **0.0** |
| spectrum, symmetric nearest-neighbour distance | 5.477e-13 (rel 9.21e-15) | 5.477e-13 (rel 9.21e-15) |

**[M] B2 -- the taper pencil is well posed.** `S` piecewise affine, 2 deg, degree 6->20: the
fundamental `q0` is stable to **3.1e-14** and the full low-`|q|` spectrum to **1.5e-13**. Null
control `S == 0`: `max|Ac| = 0.0` exactly, and the two propagating `q` satisfy the **exact
transcendental lamellar dispersion to 8.8e-14**.

### 3.5 [M] What first order actually buys -- and what it does not

**B3.** Freeze `X_u` at `zeta = 0, 1/2, 1` and track the fundamental by continuity. The
first-order (mid-depth) layer is the `zeta = 1/2` pencil, so by the Magnus-1 argument its residual
is the **second** difference, not the first:

| sidewall | `delta_g = (h/2) max\|S'\|` | first difference `\|q0(0)-q0(1/2)\|` | **second difference** | ratio |
|---|---|---|---|---|
| 0.25 deg | 0.386 % | 1.446e-03 | **7.60e-06** | 5.26e-03 |
| 0.50 deg | 0.773 % | 2.899e-03 | **3.04e-05** | 1.05e-02 |
| 1.00 deg | 1.546 % | 5.831e-03 | **1.22e-04** | 2.09e-02 |
| **2.00 deg** | **3.093 %** | 1.180e-02 | **4.88e-04** | 4.13e-02 |
| 4.00 deg | 6.194 % | 2.421e-02 | **1.96e-03** | 8.11e-02 |
| 8.00 deg | 12.448 % | 5.145e-02 | **8.05e-03** | 1.57e-01 |

The first difference is **linear** in the angle; the second difference is **quadratic** (measured
exponent 2.01 over 32x in angle). **[M] So the first-order covariant layer's residual is
`O(delta^2)` where the `ns=1` staircase's is `O(delta)` (S3.3) -- a genuine order improvement.**

**[A, estimate -- flagged]** Converting the modal residual to the observable scale: over the layer
a modal-index residual `dq` accumulates a phase `k0 h dq = 1.487 dq`. At 2 deg that is
`1.487 x 4.88e-04 = 7.3e-04`, against the measured `ns=1` staircase error of `9.14e-03`, i.e.
**one covariant layer ~ a 4-slice staircase at 2 deg** (using the measured staircase order 1.91);
because the two scale as `delta^2` and `delta`, the advantage grows at shallower angles
(~8 slices at 0.5 deg) and shrinks at steeper ones (~2 slices at 8 deg). **This bridge is arithmetic, not a measurement** -- it is the number the
full implementation would have to reproduce, and the reason the full implementation is needed
before any accuracy claim.

**The real prize is not the slice count.** In the covariant frame the walls are at **fixed `u` at
every depth**, so a `K`-sub-slab covariant cascade shares **one** grid. The union-grid collisions,
the per-layer windows, the `min_feature` snap and the `O(ns^3.4)` cost law -- the entire subject of
this campaign, on tapered stacks -- do not arise. That, not the ~4x, is what would make T3-6 worth
building.

### 3.6 [M] The two blockers this spike found

**BLOCKER 1 -- forward/backward mode classification. Measured, hard, and unsolved.**

`A1`, `Ac/i`, `A2` are all real, so conjugating the pencil shows its symmetry is **`q -> -conj(q)`,
not `q -> -q`**. Measured on the 2 deg taper pencil (degree 10, `n = 30`):

```
fundamental  q0 = 1.723252945 + 0.011720687i        <-- COMPLEX, on a LOSSLESS cell
min |q + q_m|        over the spectrum : 2.3e-02, 2.5e-02, 1.2e-02, 4.9e-02   (-q is NOT there)
min |q + conj(q_m)|  over the spectrum : 1.8e-15, 1.3e-13, 5.8e-14, 5.0e-14   (-conj(q) IS)
modes with Im>0 / Im<0 / |Im|<1e-9     : 31 / 29 / 0        (a clean split needs 30 / 30)
```

Both members of a forward/backward pair therefore carry the **same sign of `Im q`**, and no
eigenvalue is purely real. `_forward_branch_flip`, the `Im(q)` sign rule, and the flux selector's
`|Im q| < 1e-7 qmax` propagating test (`_core.py:3903-3906`) **all mis-classify**. Consequences,
measured:

| test | result |
|---|---|
| NULL CONTROL 1 -- pure shear (`X_u == 1` at every z, so every `K` is the same operator) | `\|S(K=1) - S(K=8)\| = 9.49e-16` -- machine zero, as it must be |
| NULL CONTROL 2 -- the SAME cascade driven by a smooth `eps(z)` ramp (`S = 0`, selector valid) | `K` = 1/2/4/8 -> 1.66e-02 / 2.27e-03 / 5.78e-04 / 1.44e-04; ratios **7.32 / 3.92 / 4.02** = **second order**. The cascade machinery is sound. |
| the TAPER cascade on the shipped selector | first order at degree 10; at degree 12 it **diverges** (`K`-differences of 1e+2 - 1e+3) |

**No number from the taper cascade is quoted as an accuracy result anywhere in this report.** The
physically sensible reading of a complex `q` here is adiabatic amplitude change in an expanding
frame ([H], not established), but whatever the interpretation, **a validated mode selector for a
non-normal pencil with `q -> -conj(q)` symmetry is T3-6's first deliverable**, and it is not a
small one: the flux criterion has to be re-derived for a frame that is not flux-conserving.

**BLOCKER 2 -- the far field lives in a distorted frame.** At `z = 0` and `z = h` the map is
`X(u, 0) = u - (h/2) S(u)`, which is **not** a rigid translation (unlike the shear, where it is,
which is exactly why the shipped slant far field is simple). The Rayleigh projection needs
`INT phi(u) exp(-i m G X(u,z)) X_u du`. Magnitude of the distortion:

| sidewall | 0.25 | 0.50 | 1.00 | **2.00** | 4.00 | 8.00 deg |
|---|---|---|---|---|---|---|
| interface frame shift | 0.68 nm | 1.35 | 2.71 | **5.41** | 10.84 | 21.78 nm |
| as % of the period | 0.10 % | 0.19 | 0.39 | **0.77** | 1.55 | 3.11 % |

At 2 deg that is 5.4 nm -- **the same size as the `ns=1` staircase's wall error**. It cannot be
ignored; it is a quadrature, not an eigenproblem, so it is bounded work, but this spike did **not**
prototype it and therefore has **no** end-to-end R/T number for the covariant layer.

### 3.7 Verdict and recommendation -- T3-6

**The derivation: GO.** It is correct, it is validated at 1e-15 against shipped validated code, it
collapses to a **single z-independent convection-like term** exactly as the plan hoped -- with the
correction that the term is a **dilation generator with a piecewise-affine coefficient `S(u)`**,
not a translation generator with a scalar `tan(phi)`, and that the `<v|S'|phi>` piece is easy to
drop and invisible to the slant-limit test.

**The item: NO for 5.34, GO as a scoped research item afterwards.** Reasons, in order:

1. **Two unsolved prerequisites** (S3.6), one of which (mode classification for a non-normal
   pencil) is research-class in its own right and has no precedent in the repo.
2. **The measured payoff is ~4 slices at 2 deg** ([A] estimate), not the "obsoletes the staircase,
   removes the `n_slice` axis" the plan hopes for. The `n_slice` axis survives; it just converges
   from a `delta^2` starting point instead of a `delta` one.
3. **The plan's named oracle for T3-6 does not exist.** "The converged staircase limit computed
   per-layer, which is now affordable and measured stationary at ns ~ 6-8" -- measured: the PMM
   tapered stack on this device class scatters by up to 10x across `(degree, n_slice)` on BOTH
   grid modes, deterministically and with energy conserved (S4). Until that is fixed, T3-6 has no
   PMM-side reference and would have to be adjudicated entirely against RCWA, whose systematic
   offset here is 7.5e-04 -- **the same order as the effect being measured.**
4. **Scope creep is guaranteed.** The prototype is scalar TE; the library's taper users are on the
   full `(3,3)` tensor Jones path with out-of-plane components.

**If it is taken later**, the sequence is: (a) mode selector + its fail-before test, (b) the
distorted far-field projection with the shear case as the null control, (c) the scalar TE
end-to-end solve adjudicated against the RCWA `ns = 384` ladder in S3.2, (d) only then the tensor
path. Estimate **4-6 AC**, not the plan's 1 AC spike + implementation.

---

## 4. [M] The finding that outranks both spikes -- the PMM tapered stack is silently wrong across `(degree, n_slice, layer_grids)`

Found while building Spike 2's staircase reference; then measured deliberately in
`validation/m5_taper_degree_spread.py`.

**Device.** Deliberately the *simplest possible* member of the parent audit's configuration class:
period 700 nm, wl 1310 nm, H = 310 nm, duty 0.5, `eps` 4 / 1 (**lossless, isotropic, dielectric**),
`n_sup = n_sub = 1.5`, **theta = 8 deg IN-PLANE**, one region, **2 deg symmetric taper**
(duty change 0.03093 periods). No coats, no LC, no absorbing substrate, no out-of-plane tensor.

**Arbiter.** `RCWAStack.add_tapered_grating` on the identical geometry (`raster='area'`,
`n_orders=21`, `n_x=4096`) -- an independent method, and smooth:

| `ns` | 1 | 2 | 3 | 4 | 6 | 8 | 12 | 16 | 384 |
|---|---|---|---|---|---|---|---|---|---|
| RCWA `R0(te)` | 0.011718 | 0.011696 | 0.011720 | 0.011732 | 0.011742 | 0.011747 | 0.011750 | 0.011751 | **0.011753** |

**Total RCWA movement over `ns = 1..16`: 5.6e-05 (te), 3.2e-05 (tm).** `R+T` closes to 2e-12 or
better at every rung. Both builds reproduce the RCWA row to all printed digits.

**PMM `R0(te)` over the same geometry.** Rows are `layer_grids` x `degree`; columns are
`ns = 1, 2, 3, 4, 6, 8, 12, 16`. `X` marks `|R0 - RCWA(ns)| > 5e-3`; `!` marks the library's energy
guard -- **it fires in exactly ONE of the 80 cells, and in none of the other 25 wrong ones.**

```
per-layer  deg= 8: 0.012491  0.012446  0.012473  0.012488  0.012501  0.012507  0.000827X 0.014028
per-layer  deg=10: 0.012489  0.012444  0.012472  0.012486  0.012499  0.014248  0.012510  0.000828X
per-layer  deg=12: 0.012488  0.012443  0.012471  0.012485  0.012498  0.014267  0.034466X 0.014093
per-layer  deg=14: 0.012488  0.012443  0.012470  0.012484  0.012497  0.036241X 0.036538X 0.014103
per-layer  deg=16: 0.012488  0.012443  0.012470  0.000851X 0.012497  0.012503  0.000833X 0.066857X
shared     deg= 8: 0.012491  0.012446  0.012472  0.012486  0.012499  0.012505  0.000829X 0.029941X
shared     deg=10: 0.012489  0.012444  0.012471  0.012485  0.012498  0.012504  0.000831X 0.094151X
shared     deg=12: 0.012488  0.012443  0.012470  0.000850X 0.000841X 0.000836X 0.030032X 0.089316X
shared     deg=14: 0.012488  0.012443  0.000860X 0.012484  0.000841X 0.030131X 0.030041X 0.012511
shared     deg=16: 0.012488  0.012443  0.012470  0.000851X 0.030243X 0.000837X 0.033632X 0.118504!X
```

**Spread of the PMM answer over `degree x layer_grids`, at each `n_slice`** -- the same numbers
read down the columns, against an RCWA reference that moves by 5.6e-05 in total:

| `ns` | 1 | 2 | 3 | 4 | 6 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|---|---|
| PMM spread over (degree x grids) | 3.0e-06 | 3.3e-06 | **1.16e-02** | **1.16e-02** | **2.94e-02** | **3.54e-02** | **3.57e-02** | **1.18e-01** |
| as a multiple of the answer itself (0.01175) | 0.00026 | 0.00028 | **0.99** | **0.99** | **2.5** | **3.0** | **3.0** | **10.0** |

**From `ns = 3` the discretisation spread EQUALS the answer; by `ns = 16` it is 10x the answer.**
Worst `|R0(te) - RCWA|` over the grid: **1.068e-01**. Worst `|R+T-1|`: **2.332e-01**, in the single
cell where the guard finally fires (`shared, deg 16, ns 16`) -- every other `X` cell has
`|R+T-1| <= 1e-6`.

**[M] What this table says.**

1. **`ns <= 3` is solid.** Every cell agrees to 6 digits with every other, at a **constant**
   `+7.4e-04` offset from RCWA -- a method-to-method systematic (PMM has exact walls, RCWA has a
   Fourier floor; which is right is not adjudicated here and does not matter for what follows).
2. **From `ns = 3-4` onward the answer scatters by up to 10x.** Excursions to `0.00083`
   (**-93 %**), `0.030 / 0.034 / 0.036` (**+3x**), `0.067` (**+5x**), `0.094` (**+8x**),
   `0.1185` (**+10x**). **26 of the 80 cells are wrong by more than 5e-3.**
3. **The scatter is not monotone in ANY knob.** `deg 8` fails at `ns=12` but is clean at `ns=16`;
   `deg 16` fails at `ns=4`, is clean at `ns=6, 8`, fails at `ns=12, 16`. Raising `degree` makes it
   *worse*, not better. That is the signature of a **conditioning / branch-selection accident**,
   not of under-resolution.
4. **`layer_grids='per-layer'` does not fix it, and `'shared'` is worse.** The 5.32.0 per-layer
   work delays the onset (first failure at `ns=12` for deg 8-10 vs `ns=3-4` for shared deg 12-14)
   but does not remove it.
5. **`|R+T-1| <= 1e-6` in 79 of 80 cells, and the energy guard fires in one.** This is textbook
   passive-but-wrong: **25 of the 26 wrong cells pass every conservation check the library
   performs.**

**[M] Cross-build -- and this is the important qualifier.** All **80** cells of the
`10 x 8` table above agree between W and L **to all six printed digits** (the only cross-build
difference anywhere is the energy residual of the single guard-firing cell: `|R+T-1|` 2.33e-01 on
W, 9.17e-03 on L), and the `ns`-ladder
steps at `ns = 8, 12, 16, 24` are identical to 4 digits on both builds
(0.1901 / 0.5490 / 0.3621 / 0.6979). **The defect is DETERMINISTIC: it survives a dual-build
check.** It is therefore *not* the C13 / build-dependent-draw class the campaign's M1 mission is
hardening against, and M1's conditioning work will not incidentally fix it. Only at `ns >= 32` do
the builds also start to disagree (W `R+T = 1.00389` at `ns=32`; L `R+T = 1.0000000` at `ns=32`,
different step values, both first raising the guard at `ns=64`) -- i.e. a *second*,
build-dependent regime sits on top of the deterministic one.

**This is the strongest single reason to escalate:** the campaign's standing evidence rule
("both-BLAS-builds for any numeric behaviour claim") **passes** on every wrong cell in the table
above. A dual-build check is necessary and, here, not sufficient; the independent-oracle rule
(rule 11) is what catches it.

**[M] `stabilize='slices'` DOES detect it -- and this is the actionable half of the finding.**
(`stabilize=True` is not a valid value; the API takes `None` or `'slices'`.) On both affected shared cells it
returns the **same wrong** value but **fires its warning**:

```
shared deg=12 ns= 8  stabilize='slices': R0(te)=0.000836  R+T=0.999999997  guard=True   91 s
shared deg=12 ns=12  stabilize='slices': R0(te)=0.030032  R+T=0.999999994  guard=True  213 s
per-layer, either ns: NotImplementedError -- "not applicable with layer_grids='per-layer'"
```

So the shipped R-1 union-grid consensus tripwire **does detect this defect** on the shared path.
At 91-213 s per solve it is not a default and not a sweep-wide guard -- but **M2 has a ready-made
detector to bound the affected region before spending 1800 solves inside it**, and its firing is
independent evidence that the excursions are a discretisation pathology, not physics. On the
**per-layer** path it raises by design (plan S1.1), so **that half of the table has no detector at
all today** -- which is notable given per-layer is the path users are being pointed at.

**Why this matters to the campaign as scheduled:**

* **M2 / T3-2** is *"the `ns = 8-12` stress band"* and plans **1800 solves** on the mesh across
  `ns in {8, 10, 12}` x `degree {6, 8, 10}`. **On this device class that entire band sits on top of
  an O(1) instability that is not monotone in either swept knob.** A stationarity sweep run over it
  would measure the instability and report it as a physics band. **[H]** The plan's
  "degree spread up to 5.6 % at ns=8" may be this same mechanism seen through a smaller aperture --
  M2 should test that before interpreting any spread as convergence behaviour.
* **T3-6**'s named oracle ("the converged staircase limit computed per-layer") **does not exist**
  on this device class -- see S3.7.
* The parent audit attributed the in-plane-oblique tapered pathology to **union-grid wall
  collisions**, and per-layer grids were built as the fix. **This device has a single tapered
  region, a lossless dielectric, no coats, and therefore no coat/offset resonance at all** -- yet it
  reproduces the symptom on BOTH grid modes. **[H] The mechanism may be broader than the union
  grid.** That is a hypothesis this report does not test, but it is the reason the diagnosis should
  not be assumed closed.

**Recommendation: raise as P1 against M2 before its mesh sweep is launched.** The reproduction is
`validation/m5_taper_degree_spread.py` -- 80 PMM solves + 9 RCWA arbiter solves, ~35 min, no new
machinery, the independent arbiter built in. This report does not attempt the diagnosis; the first axes to instrument are the
per-layer interface `lstsq`/`solve` conditioning (M1's census already covers those call sites), the
forward-branch selection in `_sem_modes` on the thin slices, and `far_field_orders` vs the window
capacity.

---

## 5. Corrections and additions to the campaign plan

| # | plan says | measured / analysed | where |
|---|---|---|---|
| **C1** | N-1 is "a per-segment weight vector in place of a scalar, **in two functions**" | **six scale sites in four functions**, plus `xb` and the `h`/`J` attributes; `_build_elementary` / `_build_sets` need no change | S2.1 |
| **C2** | N-1 oracle (b): "with equal segments the new code must be byte-identical" | achievable at **0.0**, but **only** with a uniform fast path computing `h = d/N`; `diff(linspace)` gives 5.7e-14 through the `1/J` stiffness scale | S2.3/G3 |
| **C3** | N-1 physics risk: the `J`,`1/J` cancellation in the mixed matrix "must be re-derived rather than assumed"; the Bloch hat's C0 "must be checked" | both hold **structurally**, per segment; measured at **0.0** for the mixed matrices and **0.0** for C0 at ratios to 1e6 | S2.2, S2.3 |
| **C4** | S-1 already calls N-1 the enabler, but the release plan still schedules S1 as the 8-12 AC headline and defers N-5 *with* it | **quantified, and it re-orders them.** N-1 alone makes a shared-grid 2-D taper representable (`Nx = 12`, 2.97 GB/matrix); S1 is then a 16x memory optimisation, and **N-5 buys the same 16x for 3 AC instead of 8-12** | S2.4 |
| **C5** | S-1: "a 2 deg taper at ns=6 needs `Nx ~ 390`, `eigdim = 1.5e7`" | **confirmed**: 388, 1.475e7, 3.24e6 GB/matrix | S2.4 |
| **C6** | S-3 cost table | **reproduced exactly** at Nx = 3/4/6/7/8/13/25 | S2.4 |
| **C7** | T3-6: "is the trapezoid metric treatable at first order as a single convection-like term?" | **YES** -- and the term is `(2i/k0)<v\|S\|phi'>` with `S(u)` piecewise **affine** (a dilation generator), z-independent **exactly**, not just to first order | S3.4 |
| **C8** | T3-6 oracle: "the converged staircase limit computed per-layer, which is now affordable and measured stationary at ns ~ 6-8" | **void on this device class** -- the tapered PMM stack scatters by up to 10x across `(degree, ns)` on both grid modes, with energy conserved in 25 of the 26 wrong cells | S4 |
| **C9** | T3-6: "if it works it *obsoletes the staircase*, removing the `n_slice` axis entirely -- the largest available perf win in the whole plan" | **overstated.** First order is `O(delta^2)` vs the staircase's `O(delta)`: ~4 slices' worth at 2 deg. The real prize is that the covariant frame **fixes the grid across slices**, removing the union-grid pathology on tapers | S3.3, S3.5 |
| **C10** | T3-6: 1 AC spike, then implement | **4-6 AC** after the spike, gated on two unsolved prerequisites (mode selector for a non-normal `q -> -conj(q)` pencil; distorted far-field projection) | S3.6, S3.7 |
| **C11** | standing rule 3: "Windows/MKL + WSL/OpenBLAS" | both environments are now **scipy-openblas 0.3.31.188**; the dual build differs by platform/compiler/interpreter only | S0 |
| **C12** | -- (new) | on this device class the **efficiencies are 256x less sensitive to the staircase than the complex Jones**; an efficiency-only convergence table is blind to it | S3.2 |
| **C13** | -- (new) | the S4 defect is **deterministic across both builds**, so the campaign's dual-build rule **passes on every wrong cell**. Rule 3 is necessary and not sufficient; rule 11 (independent oracle) is what catches this class | S4 |
| **C14** | -- (new) | `stabilize='slices'` **fires** on the affected shared-grid cells (91-213 s/solve) -- M2 has a ready-made, if expensive, detector. It is `NotImplementedError` on the per-layer path, which therefore has **no** detector for this class | S4 |

---

## 6. Recommendation for 5.34

**Full campaign: NO. Capability campaign around N-1 + N-5: YES.**

| | recommendation |
|---|---|
| **N-1** (non-uniform 2-D segments) | **GO -- make it the headline of 5.34.** 2 AC. Every gate passed; it is what turns the 2-D taper from impossible into feasible. |
| **N-5** (C2/C4 fold, pure 2-D) | **PROMOTE above S1.** 3 AC for the same 16x memory the mortar buys for 8-12, with none of the mortar's three named physics hazards, and it ships standalone on the shared grid. |
| **T3-7** (lattice quantisation) | **GO with N-1.** It is now the mechanism that bounds the segment ratio and therefore `cond(G)` (quadratic in the ratio, S2.3/G4). |
| **S1** (2-D per-layer + Kronecker mortar) | **DEFER again, and re-scope.** After N-1 + N-5 it is a memory optimisation with a measured budget, not a capability. Take it only if the target device is still short; keep the Kronecker-cross-mass requirement and the dense-2-D-cross-mass rejection. |
| **T3-6** (covariant taper) | **NOT in 5.34.** Record the derivation (S3.4) and the two blockers (S3.6) in the roadmap; revisit after the S4 defect is diagnosed, since T3-6 has no PMM-side oracle until then. |
| **S4 defect** | **Escalate to M2 now, as a P1, before the 1800-solve mesh sweep is launched.** It is not a 5.34 item -- it is a 5.33 item, and it gates the credibility of every tapered-stack number the campaign will quote. |

Net: **5.34 = "the 2-D pure stack gets arbitrary walls and a symmetry fold"** (N-1 + N-5 + T3-7,
~6 AC + validation), not "the 2-D mortar" (~12-16 AC). That is a smaller release that delivers the
capability the campaign was actually chartered to deliver.

---

## 7. Artefacts

| file | what |
|---|---|
| `validation/m5_derham_nonuniform.py` | Spike 1: `Basis1DNU`, `Granet2DTransverseENU`, a full non-uniform mirror of `pmm_efficiency_2d_staggered`, and gates G1-G7 |
| `validation/m5_covariant_taper.py` | Spike 2: Part A (shipped-API ladders, angle sweep, null controls, shear-absorption) and Part B (covariant pencil, its shipped-slant validation, the z-freeze residual, the mode-symmetry blocker and its two controls) |
| `validation/m5_taper_degree_spread.py` | The S4 finding: PMM `(degree x n_slice x layer_grids)` on a 2 deg tapered grating, adjudicated cell by cell against the RCWA twin on the identical geometry |

None of the three is a test, none is imported by the library, and none writes into `lumenairy/`
or `tests/`. Raw logs and JSON for both builds were produced by the commands in S0.
