# Bounding the Newton inverse to the traced samples' own support (niche C8)

**Date** 2026-08-01 - **Tree** `fix/pmm-union-grid-conditioning` @ `8e7b156`
(C6+C7 committed) plus this study's uncommitted change to
`lumenairy/elements/_lens_traced.py` - **Subject** the named structural fix for
the defect `ENERGY_CONSERVATION_AUDIT_2026_07_31` and
`C6_FIT_GUARD_DECISION_2026_07_31` left open: design 121's on-axis production
call manufactures **+0.486 % of the input power** and no EE-family metric can
see it - **Question** can the Newton inverse be bounded to the region the
traced rays actually reached, cleanly enough to ship as a default?

---

## 0. Headline

**Yes. It ships as `REMAP_INVERSE_SUPPORT_BOUND = True`, it repairs every order
of design 121 on all six conservation bounds at both subsamples, it costs
0.003 EE3 points, and it fixes two defects the fit guard structurally could
not.**

On the **production configuration** - the complete shipped path the campaign's
EE numbers come from - order (0,0), one field, both currencies:

| config | EE3 % | EE6 % | EE12 % | `P_tile` % | `elem(last)` | `g4` | `amax4` | `r_rms` mm |
|---|---|---|---|---|---|---|---|---|
| `noC6` (C6 off) | 86.709 | 98.637 | 98.766 | 98.7691 | 0.995883 | 0.000e+00 | 0.00e+00 | 0.8419 |
| **`ship`** (HEAD) | **88.400** | 98.646 | 98.803 | 98.8046 | **1.000741** | **4.715e-03** | **8.32e-01** | **0.9644** |
| **`ship` + C8** | **88.397** | **98.645** | 98.804 | 98.8057 | **0.996026** | **0.000e+00** | **0.00e+00** | **0.8383** |
| `shipG` (fit guard) | 87.436 | **98.056** | 98.798 | 98.7988 | 0.995984 | 0.000e+00 | 0.00e+00 | 0.8381 |

**C8 keeps 1.688 of C6's 1.691 EE3 points and removes 100 % of the
manufactured energy.** The fit guard - the remedy already in the tree - keeps
0.727 of them and costs 0.590 EE6 points. Against C6-off, C8 is **+1.688 EE3
and +0.008 EE6**; the guard is **+0.727 EE3 and -0.581 EE6**.

Five results the record did not have.

1. **It repairs (-2,0) and (-3,0), which the fit guard structurally could
   not.** Those orders make their lobe on the OFF-CENTRE weighted branch, where
   the guard is inert by construction; `C6_FIT_GUARD_DECISION` S3 measured them
   failing C3+C4 with the guard on *and* off, at both subsamples. Under this
   bound both pass: `g4` at **0.18x / 0.22x** of their exact-ray ceilings and
   `amax4` **6x** under bound. **All six orders now score 6 of 6 at `rs=4`, and
   all four measured configurations score 6 of 6 at `rs=2`.**
2. **It repairs a defect C6 has nothing to do with.** At `rs=2` on (-4,-2) it is
   **C6-OFF** that violates the halo criterion (28x / 78x over) - the energy
   audit's "reversal" - and the bound takes that row to 0.35x of its ceiling as
   well. The mechanism was never C6's; C6 only made it large enough to see.
3. **It regresses none of the six synthetic fixtures that keep the fit guard
   opt-in** - every one reproduces the hard-mask branch to every printed digit -
   and it makes the guard itself SAFE: the guard's two regressions
   (`P/Pin` 1.00697 with 3.598e-01 of peak beyond 3 w, and 4.593e-02 of peak)
   both vanish when the bound is on.
4. **Independently corroborated on a fixture sharing no code with design 121.**
   The 0.641-of-peak lobe `ORACLE_ENERGY_AND_D6_HALO_2026_08_01` diagnosed in
   niche D6's decentred fixture is the same defect class, and the bound takes
   it to exactly zero. **100 % of the power the bound removes there lies
   outside the convex hull of every alive traced ray of that call** - measured,
   partitioned three ways, S5 - and the post-C8 chain-exit power lands at
   **0.998363**, back below unity and onto the documented `ray_subsample=4`
   discretisation deficit, against that oracle's now absolutely-calibrated
   reference of 1.0000.
5. **The "a hard mask may diffract" worry is answered with a number rather than
   a taper.** On the one fixture where the bound truncates a field carrying
   real amplitude at its boundary, the UNBOUNDED field's own largest
   nearest-neighbour `|E|` step in that band is **1.366e-01 of peak**; a hard
   cut leaves **3.07e-02** and the shipped one-cell feather **2.40e-02**. What
   the bound deletes is jagged manufactured light, and what it leaves is
   **4.5-5.7x smoother than what was there**.

**One defect of my own, found by my own instrument and fixed (S5.2).** The
first implementation evaluated the taper on the coarse Newton lattice, and the
bilinear upsample carried that attenuation up to one coarse cell INSIDE the
support: **2.211e-04 of the input power of legitimate skirt light, on the D6
fixture, at feather 0**. The partition in S5 is what caught it. The taper now
holds at exactly 1 out to `sqrt(2) sub dx` - the Lipschitz reach of the
upsample stencil - and the removal from inside the support is **exactly
0.000e+00 at every feather including 0**. That fix also restored niche D6's
`FWHM_px > 1.70 FWHM_orc` discriminator to its pre-C8 value of **1.857x**
(the pre-plateau version had narrowed it to 1.762x).

**`REMAP_STATIONARY_PHASE_FIT_GUARD` is now redundant on every case measured,
and is NOT removed** - S8 states why, and what it would still be for.

---

## 1. Provenance, instruments, floors, sampling

### 1.1 What was measured, and against which library

| file | sha256 (16) | state |
|---|---|---|
| `lumenairy/elements/_lens_traced.py` | `e978cbedb183dbce` | the state every measurement below was taken on |
| `lumenairy/elements/_lens_traced.py` | `5f15da2e44144740` | **final** - one digit of a comment corrected afterwards (`5.42e-09` -> `5.44e-09`); no executable line differs |
| `lumenairy/propagators/carrier.py` | `1a90453a4ef65399` | **unmodified** - the same hash the two 2026-07-31 audits record |

`CHANGELOG.md` and `lumenairy/elements/pmm/**` were not touched. The only
library file this study changes is `_lens_traced.py`.

Every runner prints the sha256 of the file it imported. All numbers below are
on `e978cbedb183dbce` **except** where a row is explicitly labelled
"pre-plateau" (S5.2), which is the intermediate implementation the
over-removal partition convicted.

### 1.2 Instruments

| instrument | what it measures | new? |
|---|---|---|
| `probe_c8_support_bound.py` | element-level halo family on design 121's own group-5 call, C6 off / C6 on / + fit guard / + C8 at six feathers / + both | **new** |
| `probe_c8_synthetic.py` | the SIX `probe_ghost_synthetic` fixtures verbatim, plus C8 and guard+C8 arms | **new** |
| `probe_c8_d6_overremoval.py` | **the three-way partition of the removed power** against the call's own exact ray bundle, plus the edge-jump feather metric | **new** |
| `probe_c8_byte_identity.py` | the fail-before contract against the `HEAD` shadow module, at the SHIPPED flag settings | **new** |
| `c8_with_bound.py` | runs any existing runner with the bound forced on/off, without editing it | **new** |
| `energy_stage_audit_121.py` | the six-bound per-stage table | reused, unedited |
| `energy_ee_vs_conservation_121.py` | both currencies on ONE production field | reused, unedited |
| `probe_c6_chain.py` / `hybrid_localize_121.py` | per-order EE3 against the exact-ray oracle | reused |
| `focus_scan_121.py` | the single-beam at-plane acceptance | reused |
| `halo_calibration.py` | the v5.32 halo check over the P2 battery | reused |
| `d6halo_controls.py` (another agent's) | niche D6's twelve dependent assertions | reused verbatim |

**No existing runner was edited.** `c8_with_bound.py` exists precisely so that
`energy_stage_audit_121.py`, `focus_scan_121.py` and the rest could be re-run
under the new flag without touching files another agent is working in.

### 1.3 Differential floors - bit-exact, established before any delta

| instrument | null intervention | reading |
|---|---|---|
| `probe_c8_support_bound.py` | two identical shipped element runs, per order | **`array_equal=True`, `max\|dE\| = 0.000e+00`** on (0,0), (-2,0), (-4,-2) |
| `probe_c8_synthetic.py` | two identical runs per fixture | `array_equal=True`, all six |
| `energy_ee_vs_conservation_121.py` | two identical production runs | `array_equal=True`, `max\|dE\| = 0.000e+00` |
| `probe_c8_byte_identity.py` | the bound OFF against a SHADOW MODULE built from `git show HEAD:` | S7 |

Every delta below is against a floor of exactly zero.

### 1.4 Sampling adequacy - unchanged, and still saturated

Stated as an amplitude-weighted wrapped nearest-neighbour step at **p99.9**
against pi, never a max. Unchanged from `ENERGY_CONSERVATION_AUDIT` S1.4: the
last group's `E_in` at (0,0) reads **3.1276 rad** against pi = 3.1416, the exit
NA is 0.288-0.298 against this grid's Nyquist NA of 0.0197, and **96.5-96.8 %
of the exit power sits above the grid's Nyquist angle**.

**What this permits and forbids is unchanged, and it matters more here than in
either predecessor**, because this document's central claim is about WHERE
light is. Every number below is a **power** measurement of a returned array
(`sum |E|^2 dx^2`, `|E|^2`-weighted radial shells and moments), a **nearest-
neighbour amplitude difference** on that array, or an **exact geometric ray
trace**. None of them reads a phase and **no wave or WFE claim is made
anywhere**. The halo's radial position at an element's own exit plane is set by
the Newton inversion of the traced ray map - a geometric placement, not a
propagation - so it is exact for the returned array. Where a manufactured lobe
would have ended up after further propagation is **not** determined here, and
the "does the cut diffract" question is therefore answered at the exit plane
(S6.3) and end to end through the library's own readout (S4), never by an
angular-spectrum argument on this grid.

### 1.5 The conservation reference

Two references, both unchanged by this study.

* The exact-ray hull / halo ceiling / second-moment table of
  `ENERGY_CONSERVATION_AUDIT` S3.1 and `C6_FIT_GUARD_DECISION` S1.5:

  | order | **exact-ray `g4` ceiling** | C3 bound (3x) | **exact-ray `r_rms`** |
  |---|---|---|---|
  | (0,0) | 3.5641e-10 | 1.069e-09 | 0.8407 mm |
  | (-1,0) | 1.0153e-08 | 3.046e-08 | 0.8413 |
  | (-2,0) | 3.0443e-08 | 9.133e-08 | 0.8427 |
  | (-3,0) | 5.9279e-08 | 1.778e-07 | 0.8450 |
  | (-4,0) | 7.5559e-08 | 2.267e-07 | 0.8483 |
  | (-4,-2) | 7.4639e-08 | 2.239e-07 | 0.8503 |

* The **absolute** reference re-established by
  `ORACLE_ENERGY_AND_D6_HALO_2026_08_01`: the exact-ray + Rayleigh-Sommerfeld
  oracle now answers an absolute energy question and says design 121 is
  **lossless**, converging monotonically to `P_out/P_in` = **100.000159 %** at
  (0,0) and 100.003218 % at (-4,-2), validated at **0.99999988** on a case with
  an analytic answer. **The physical reference for every ratio below is exactly
  1.0000**, with no legitimate loss channel to subtract (chain aperture
  clipping never exceeds 1.13e-05 and no Fresnel loss is modelled).

---

## 2. The mechanism, restated so the fix can be judged against it

`apply_real_lens_traced` traces a lattice of rays, **fits** the entrance->exit
map, and **Newton-inverts the fit** once per exit pixel to find which entrance
point lands there. Both backends extrapolate outside their data - the
polynomial globally by construction, the spline past its last knot - so for an
exit pixel outside the traced landing region the Newton loop is inverting a
model where nothing was ever measured.

`_FIT_DISC_OUTSIDE_WEIGHT_REL` records the precondition under which that used
to be safe: *"the unconstrained directions of the fit inherit the map's RADIAL
SYMMETRY, the extrapolation outside the disc stays MONOTONE, and the Newton
inversion cannot find a second root."* `REMAP_STATIONARY_PHASE_LAUNCH` (niche
C6) augments every launch direction by `grad(a_fit)` of a general, non-radial
polynomial and **destroys that precondition**. The inverse then folds: a far
exit pixel's Newton solution lands back inside the bright beam, and
`_ray_density_amp_grid` samples `|E_in|` there and hands the pixel real
amplitude.

Three properties of that story decide what a fix must look like.

1. **The light is manufactured, not misplaced.** No ray of the call reaches
   the radius it appears at.
2. **Its magnitude is chaotically ill-conditioned and its presence is not.**
   The energy audit measured `g4` spreading 4.9x under ~1e-06-level input
   perturbations on axis and 202x at (-2,0), while the C6-off value never moved
   by a bit. So the fix cannot be a tuning of any knob on the fit - and
   `C6_FIT_GUARD_DECISION` measured exactly that: every knob has clean and
   dirty settings with no monotone structure, and raising the fit order makes
   it **86x worse**.
3. **It is not confined to the fit-branch C6 exposed.** (-2,0) and (-3,0) make
   their lobe on the off-centre weighted branch; (-4,-2) at `rs=2` makes one
   with C6 entirely off; niche D6's fixture makes one on a plain paraxial leg
   with no design-121 code in sight.

So the fix must act on **what is claimed from the fit**, not on the fit. That
is this bound.

---

## 3. The design

### 3.1 The support

The convex hull of the **exit landing points of the alive traced rays whose
entrance the stop passes**.

* **Taken before the fit-domain restriction.** The restriction NaNs samples
  that are perfectly good optics; reading the support after it would understate
  the support and cut real light. Taken where it is, `x_out_grid` is still the
  exact traced map and nothing the model fitted has touched it. (This is the
  same point in the function the v5.32 halo hull is read at, and for the same
  two reasons.)
* **Restricted to the rays the stop passes.** The launch square spans **1.5x
  the aperture RADIUS**, so a third of it is blocked light; including it would
  inflate the support with territory no photon reaches. This is the same
  criterion `_ray_density_amp_grid` already masks its amplitude on, so the two
  cannot disagree.
* **Convex.** A lens exit region is - the argument `inversion_method='fit'` has
  always used for its own hull mask. This bound gives the **Newton** path the
  containment the direct-fit path has had all along. Convexity can only make
  the bound LOOSER, never tighter, so it cannot manufacture a cut.
* **It declines rather than guesses.** A degenerate support (collinear or
  duplicated landings) has no hull and Qhull raises; the bound then returns the
  unbounded field rather than propagate an exception out of a physics call.
  Pinned by a test that forces the constructor to raise.

The signed distance to a convex hull is `s = max_f (n_f . p + d_f)` over its
facets, exact for any point outside it and <= 0 inside (Qhull normalises
`equations` to unit outward normals). Evaluated as a chunked
(pixels x facets) product so the temporary stays under ~160 MB whatever the
facet count and however fine the Newton lattice.

### 3.2 The taper: a plateau, then a feather

    t(s) = 1                              s <= d0
         = (1 + cos(pi (s-d0)/f)) / 2     d0 < s < d0 + f
         = 0                              s >= d0 + f

**`d0 = sqrt(2) * ray_subsample * dx` is not taste, it is the upsample**, and
it is the one thing the first implementation got wrong (S5.2). The taper is
evaluated on the COARSE Newton lattice and the amplitude is then bilinearly
interpolated to the wave grid, so a coarse node outside the hull lends its
attenuation to wave pixels up to one coarse cell inside it. `s` is
1-Lipschitz, so a pixel with `s <= 0` interpolates only from nodes with
`s <= sqrt(2) sub dx`; holding the taper at exactly 1 out to there makes that
bleed **identically zero** rather than merely small. Its cost is that the bound
sits `d0` further out - 188 um on design 121's last group, 3 % of its 6.3 mm
hull - which is measured not to readmit any of the manufactured lobe there.

**`f = _SUPPORT_BOUND_FEATHER_CELLS = 1.0` exit-lattice cell**, where the cell
is the **median exit separation of entrance-adjacent traced rays**, measured
from the samples themselves rather than from a paraxial magnification, so it
tracks the resolution at which the support is actually known. The measurement
that sets it is S6.3.

### 3.3 Scope, and why it is not a hedge

`amplitude_model='ray_density'` only. That is the only amplitude in this
function **derived from the inverse map**: the `'screen'` amplitude comes from
`apply_real_lens`'s analytic transport of the input field and never reads
`(xe, ye)`, so there is nothing there for an extrapolated inverse to corrupt.
Pinned byte-identical.

The OPL is left alone for the same reason - where the taper is zero the
amplitude is zero and the phase is unobservable - and because NaN-ing the OPL
would hard-cut a mask that is deliberately smooth (`opl_map`'s NaN mask is
upsampled at order 1 and thresholded at 0.5).

Both `newton_fit` backends are bounded **identically**: the support comes from
the traced samples before any fit, so it does not depend on which interpolant
is fitted to them. Pinned in S7.3.

---

## 4. Acceptance

### 4.1 Item 1 - the (0,0) production path

`energy_ee_vs_conservation_121.py`, the production configuration (six post-DOE
groups, the 7.7058 mm trailing leg, `final_leg='exact'`, exact Bluestein
readout, `NOUT=192`, `dx_out=0.1 um`, `n_fine_cap=12288`, `window_factor=4.0`,
fixed lattice on the chief ray). Both currencies from the same run. NULL floor
`array_equal=True`, `max|dE| = 0.000e+00`.

The table is in S0. Scored against the brief:

| requirement | measured | verdict |
|---|---|---|
| `P_out/P_ap` back within the C6-off class (~0.9959) | **0.996026** against `noC6` 0.995883 | **PASS** |
| `amax4` to ~0 | **0.00e+00** (exactly) | **PASS** |
| `g4` to ~0 | **0.00e+00** (exactly) | **PASS** |
| match the fit guard's conservation result | guard 0.995984 / 0 / 0 vs C8 0.996026 / 0 / 0 | **PASS** |
| without the guard's synthetic regressions | S4.2 | **PASS** |

The deficit floor (criterion C1b, the one that survives every library state):
C8's last-group deficit is **3.974e-03** against the same order's C6-off
**4.117e-03** - **0.97x**. HEAD's is **negative** in this configuration
(`elem(last)` = 1.000741 is above unity outright); in the diagnostic
configuration, where five per-element deficits launder it, HEAD reads 0.15x.

### 4.2 Item 1 (continued) - the six synthetic fixtures

`probe_c8_synthetic.py`, `probe_ghost_synthetic.py`'s fixtures verbatim.
`mask` is the shipped C6 branch, `weighted` is the fit guard, `C8 f1` is this
bound. `halo` counts firings of the v5.32 ray-density halo self-check.

| fixture | branch | `P/Pin` | `P>3w` | `amax3w` | halo |
|---|---|---|---|---|---|
| weak f/70 | mask | 0.99761 | 1.919e-08 | 1.201e-04 | 0 |
| | **C8 f1** | **0.99761** | **1.919e-08** | **1.201e-04** | **0** |
| weak, 2x residual | mask | 0.99765 | 1.307e-07 | 2.832e-04 | 0 |
| | **C8 f1** | **0.99765** | **1.307e-07** | **2.832e-04** | **0** |
| medium f/40 | mask | 0.99730 | 6.766e-09 | 6.678e-05 | 0 |
| | **C8 f1** | **0.99730** | **6.766e-09** | **6.678e-05** | **0** |
| medium, finer grid | mask | 0.99823 | 4.265e-08 | 1.594e-04 | 0 |
| | *weighted (guard)* | *0.99826* | ***2.576e-05*** | ***4.593e-02*** | ***1*** |
| | **C8 f1** | **0.99823** | **4.265e-08** | **1.594e-04** | **0** |
| collimated (C6 inert) | mask | 0.99779 | 7.202e-09 | 8.557e-05 | 0 |
| | **C8 f1** | **0.99779** | **7.202e-09** | **8.557e-05** | **0** |
| DESIGN-121 SCALE | mask | 0.99828 | 0.000e+00 | 0.000e+00 | 0 |
| | *weighted (guard)* | ***1.00697*** | ***1.613e-03*** | ***3.598e-01*** | ***1*** |
| | **C8 f1** | **0.99828** | **0.000e+00** | **0.000e+00** | **0** |

**Zero regressions on six of six.** Every C8 row reproduces the hard-mask
branch to every printed digit; the two rows that regress are the fit guard's,
exactly as `C6_FIT_GUARD_DECISION` recorded them.

**And the bound makes the guard safe.** With both on (`wght+C8`): 'medium,
finer grid' goes 4.593e-02 -> **4.945e-08** of `P>3w` and halo 1 -> **0**; the
design-121 stand-in goes `P/Pin` 1.00697 -> **0.99828** with `P>3w` and
`amax3w` **exactly 0**, halo 1 -> **0**. The reason the guard was kept opt-in
is removed by this bound - which does not make the guard a good default, for
the EE reason in S8.

### 4.3 Item 2 - per-order EE3 against the exact-ray oracle

`probe_c6_chain.py` at `NMIN=NMAX=6` (the chain does all six post-DOE groups
and the exact-ray + Rayleigh-Sommerfeld oracle finishes the trailing leg), the
same instrument the C5/C6 tables used. Both arms measured in this study on this
tree, so the pair is matched.

| order | oracle EE3 | HEAD (C8 off) | **C8 on** | delta |
|---|---|---|---|---|
| (0,0) | 90.08 | 89.20 | **89.20** | **0.00** |
| (-4,0) | 90.78 | 88.98 | **88.98** | **0.00** |
| (-4,-2) | 89.78 | 88.53 | **88.53** | **0.00** |

**No regression on any order, to the last printed digit.** (The brief quotes
89.21 / 88.94 / 88.49 from the CHANGELOG; this tree's own HEAD baseline reads
89.20 / 88.98 / 88.53, a 0.01-0.04 point drift that predates this study - what
matters is that the C8 arm equals the HEAD arm exactly.) FWHM and EE6 are
likewise unchanged: 3.718 / 99.78, 3.674 / 99.57, 3.631 / 99.55.

### 4.4 Item 3 - the at-plane single-beam acceptance

`focus_scan_121.py`, pure library defaults (`CREF`/`AM`/`PIP` unset), `N=2048`,
`rs=4`, `NFC=8192`, `WF=4.0`, `NOUT=2048`, with the bound ON.

| | recorded acceptance | **measured, C8 on** |
|---|---|---|
| `BEST-FOCUS[peak]` plane | dz = +0 um | **dz = +0 um** |
| FWHM | 3.450 um | **3.450 um** |
| EE3 | 90.2 % | **90.2 %** |
| EE6 | 99.7 % | **99.7 %** |
| EE12 | 99.8 % | **99.8 %** |
| peak | 5.473e+03 | 5.471e+03 |

**Unchanged on every digit of the acceptance.** The peak moves by 4e-04
relative, which is the bound removing manufactured light from the relay and is
below the metric's own quoted precision.

### 4.5 The six-bound conservation table

`energy_stage_audit_121.py` via `c8_with_bound.py`, `RN=1024`, six post-DOE
groups, `final_distance=0`, `final_leg='paraxial'` - the diagnostic
configuration, so nothing but the relay is scored. Bounds as proposed in
`ENERGY_CONSERVATION_AUDIT` S6: **C1a** `P_out/P_ap` in [0.9900, 1.00020] -
**C1b** last-group deficit >= 0.5x the same order's C6-off deficit -
**C2** end to end in [0.9850, 1.00050] - **C3** `g4 <= 3 g4_exact` -
**C4** `amax4 <= 1.0e-03` - **C5** `|dr_rms| <= 0.030`.

#### `ray_subsample = 4`

| order | cfg | `elem(5)` | deficit/floor | end to end | `g4` | `g4`/bound | `amax4` | `r_rms` | dev | **score** |
|---|---|---|---|---|---|---|---|---|---|---|
| (0,0) | HEAD | 0.999371 | 0.15x | 0.997750 | 3.400e-03 | 3.2e6 | 7.70e-01 | 0.9349 | +11.20 % | **2/6** |
| (0,0) | **+C8** | **0.995971** | **0.98x** | **0.994355** | **0.000e+00** | **0.00** | **0.00e+00** | **0.8385** | **-0.26 %** | **6/6** |
| (-1,0) | HEAD | 0.996017 | 1.00x | 0.994074 | 3.318e-10 | 0.01 | 8.24e-05 | 0.8384 | -0.34 % | 6/6 |
| (-1,0) | **+C8** | 0.996017 | 1.00x | 0.994074 | 3.318e-10 | 0.01 | 8.238e-05 | 0.8384 | -0.34 % | 6/6 |
| (-2,0) | HEAD | 0.996043 | 1.00x | 0.994131 | **2.270e-07** | **2.49** | **5.73e-03** | 0.8382 | -0.53 % | **4/6** |
| (-2,0) | **+C8** | 0.996043 | 1.00x | 0.994131 | **5.370e-09** | **0.06** | **1.611e-04** | 0.8382 | -0.53 % | **6/6** |
| (-3,0) | HEAD | 0.995917 | 1.01x | 0.994064 | **2.234e-07** | **1.26** | **5.87e-03** | 0.8380 | -0.83 % | **4/6** |
| (-3,0) | **+C8** | 0.995916 | 1.02x | 0.994063 | **1.311e-08** | **0.07** | **2.050e-04** | 0.8379 | -0.84 % | **6/6** |
| (-4,0) | HEAD | 0.995906 | 1.05x | 0.993992 | 2.628e-08 | 0.12 | 2.37e-04 | 0.8376 | -1.26 % | 6/6 |
| (-4,0) | **+C8** | 0.995906 | 1.05x | 0.993992 | 2.628e-08 | 0.12 | 2.368e-04 | 0.8376 | -1.26 % | 6/6 |
| (-4,-2) | HEAD | 0.996036 | 1.04x | 0.993816 | 2.653e-08 | 0.12 | 2.34e-04 | 0.8375 | -1.51 % | 6/6 |
| (-4,-2) | **+C8** | 0.996036 | 1.04x | 0.993816 | 2.653e-08 | 0.12 | 2.335e-04 | 0.8375 | -1.51 % | 6/6 |

`g4`/bound is against the 3x exact-ray bound of S1.5. (-4,0) and (-4,-2) are
unmoved to every printed digit: the bound has nothing to remove there.

#### `ray_subsample = 2`

| order | cfg | `elem(5)` | end to end | `g4` | `g4`/bound | `amax4` | `r_rms` | **score** |
|---|---|---|---|---|---|---|---|---|
| (0,0) | HEAD | **1.003696** | **1.003186** | **4.495e-03** | **4.2e6** | **9.78e-01** | 0.9477 | **0/6** |
| (0,0) | **+C8** | **0.999201** | **0.998693** | **0.000e+00** | **0.00** | **0.00e+00** | **0.8363** | **6/6** |
| (-2,0) | HEAD | 0.999195 | 0.998681 | **7.076e-06** | **77** | **5.02e-02** | 0.8365 | **4/6** |
| (-2,0) | **+C8** | 0.999188 | 0.998674 | **5.437e-09** | **0.06** | **1.580e-04** | 0.8360 | **6/6** |
| (-4,-2) | HEAD | 0.999164 | 0.998628 | 2.739e-08 | 0.12 | 2.32e-04 | 0.8352 | 6/6 |
| (-4,-2) | **+C8** | 0.999164 | 0.998628 | 2.734e-08 | 0.12 | 2.319e-04 | 0.8352 | 6/6 |
| (-4,-2) | **noC6, HEAD** | 0.999196 | 0.998661 | **6.322e-06** | **28** | **7.84e-02** | 0.8498 | **4/6** |
| (-4,-2) | **noC6 +C8** | 0.999190 | 0.998655 | **2.632e-08** | **0.12** | **2.361e-04** | 0.8496 | **6/6** |

The last pair is the energy audit's reversal - the row where **C6-OFF** is the
violator - and the bound repairs it too. **That is the clearest single piece of
evidence that this is not a C6 patch.**

**Every measured order and configuration now scores 6 of 6, at both
subsamples.** No bound is worsened anywhere.

#### With C6 OFF, the bound is inert on design 121

All six orders, `rs=4`, `noC6`: `elem(5)`, end to end, `g4`, `amax4` and
`r_rms` reproduce the C8-off values to every printed digit ((0,0) 0.995901 /
0.994281 / 3.606e-11 / 1.395e-05 / 0.8422; (-4,0) 0.996084 / 2.831e-08 /
2.000e-04 / 0.8500; and so on). The one exception in the whole matrix is
(-4,-2) `noC6` at `rs=4`, where `g4` reads 2.506e-08 against 2.507e-08 and
`g8` goes 2.285e-12 -> 0.

---

## 5. The over-removal question, and the defect it found in my own fix

`ORACLE_ENERGY_AND_D6_HALO_2026_08_01` S7.1 asks the right question: on niche
D6's fixture the bound was removing **4.9x more power than the reported halo
carries**. That is not answerable by argument.

### 5.1 The partition

`probe_c8_d6_overremoval.py`. For the element call the bound acts on, build
TWO convex hulls from the call's **own exact ray bundle** (the same object the
library's halo check reads): `H_all` over every alive ray, and `H_ap` over the
rays the entrance stop passes - the bound's own support. `H_ap` is contained in
`H_all`. Every removed pixel then falls in exactly one class:

* **(a) outside `H_all`** - no traced ray of the call reaches it. Manufactured
  by definition.
* **(b) between `H_ap` and `H_all`** - reachable only by rays the stop BLOCKS.
  `_ray_density_amp_grid` already NaNs a pixel whose converged entrance is
  outside the stop, so light there exists only because the Newton inverse found
  a *different*, spurious entrance inside it. Also manufactured.
* **(c) inside `H_ap`** - legitimate skirt. **Must be exactly zero.**

D6's fixture, `final_leg='paraxial'`, the scored group call (N=1024,
dx=5.8594 um, aperture 3.4 mm, 12849 alive rays, all of which the stop passes,
so `H_ap` = `H_all` here and (b) is empty by construction):

| feather | `dP`/P_in | (a) outside `H_all` | (b) blocked-only | (c) INSIDE the support | edge jump |
|---|---|---|---|---|---|
| OFF | 0 | 0 | 0 | 0 | 1.366e-01 |
| 0.00 | 2.621e-03 | **2.621e-03** | 0.000e+00 | **0.000e+00** | 3.067e-02 |
| 0.25 | 2.589e-03 | **2.589e-03** | 0.000e+00 | **0.000e+00** | 3.067e-02 |
| 0.50 | 2.559e-03 | **2.559e-03** | 0.000e+00 | **0.000e+00** | 3.067e-02 |
| **1.00** | **2.484e-03** | **2.484e-03** | **0.000e+00** | **0.000e+00** | **2.395e-02** |
| 2.00 | 2.300e-03 | **2.300e-03** | 0.000e+00 | **0.000e+00** | 1.929e-02 |
| 4.00 | 2.061e-03 | **2.061e-03** | 0.000e+00 | **0.000e+00** | 1.185e-02 |

**100.0 % of the power the bound removes on that fixture lies outside the
convex hull of every alive traced ray of the call, at every feather.** None of
it is legitimate skirt. The answer to the caution is the first one it offered:
the extra was also manufactured.

Corroborated end to end by the absolute reference: with the lobe present the
chain-exit power is **1.000534** of the input - above unity, i.e. the chain
returns more light than was launched into it - and with the bound on it is
**0.998363**, back below unity and onto the documented `ray_subsample=4`
discretisation deficit. The manufactured energy had been **masking the
legitimate deficit**, which is the same C1b signature the energy audit found on
design 121, on a second and entirely different fixture.

### 5.2 The defect that partition found in MY OWN fix

**The (c) column was not zero in the first implementation.** Pre-plateau, the
same partition read:

| feather | 0.00 | 0.25 | 0.50 | 1.00 | 2.00 | 4.00 |
|---|---|---|---|---|---|---|
| (c) inside the support, pre-plateau | **2.211e-04** | 8.194e-05 | **3.808e-05** | 3.808e-05 | 1.207e-05 | 3.224e-06 |
| (c) inside the support, **shipped** | **0** | **0** | **0** | **0** | **0** | **0** |

At the shipped feather that was **3.808e-05 of the input power of legitimate
skirt light - 1.1 % of everything the bound removed on that fixture** - and it
directly contradicted the fix's own stated contract ("the feather band lies
entirely outside the hull, so every pixel with traced data behind it keeps its
full amplitude").

The cause is the **bilinear upsample**, not the taper: the taper is evaluated
on the coarse Newton lattice and the amplitude is interpolated to the wave
grid, so a coarse node outside the hull lends its attenuation to wave pixels up
to one coarse cell inside it. The decaying column is the signature - a wider
feather puts the first outside node closer to 1. The fix is the plateau of
S3.2, and it is exact rather than approximate: `s` is 1-Lipschitz, so a pixel
with `s <= 0` interpolates only from nodes with `s <= sqrt(2) sub dx`.

**Two things followed from the plateau that are worth stating separately.**

1. **It restored niche D6's fail-before discriminator.** The pre-plateau
   version narrowed `FWHM_px > 1.70 FWHM_orc` from 1.857x to **1.762x** against
   a bar that had already been lowered once. Re-measured on the shipped
   version, `d6halo_controls.py PART=D`: **1.857143 with the lobe present and
   1.857143 with it removed** - the paraxial FWHM does not move at all
   (5.850 -> 5.850 um). **The bar does not need re-pricing and none was
   asked for.** All twelve of D6's dependent assertions pass in both states;
   the largest movement in an asserted quantity is the exact leg's EE2 at
   **-0.283 points against 8 points of margin**.
2. **It made the bound exactly inert on the C6-off design-121 calls.** The
   on-axis `noC6` element call's `g4` reads **3.606e-11**, identical to the
   unbounded value; pre-plateau it read 3.570e-11.

**The lesson is the one this project keeps re-learning**: the instrument that
convicts the library has to be pointed at the fix as well. The partition was
built to answer a question about the fix's *scale* and it found an error in the
fix's *contract*.

---

## 6. The feather, measured

### 6.1 What it is NOT for

With the plateau in place the feather does not protect legitimate light: the
(c) column of S5.1 is exactly zero at every feather **including 0**. And on
design 121 it is measurably inert - every metric of every order is identical
from 0.0 to 4.0 cells:

| order | metric | f=0 | f=0.5 | f=1 | f=2 | f=4 |
|---|---|---|---|---|---|---|
| (0,0) | `P/Pin` | 0.995976 | 0.995976 | 0.995976 | 0.995976 | 0.995976 |
| (0,0) | `g4` / `amax4` | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| (-2,0) | `g4` | 5.070e-09 | 5.070e-09 | 5.070e-09 | 5.070e-09 | 5.070e-09 |
| (-4,-2) | `dP`/P_in | -1.507e-14 | -1.507e-14 | -1.507e-14 | -1.507e-14 | -1.507e-14 |

### 6.2 What it IS for

The sharpness of what is left, on the one fixture measured where the bound
truncates a field carrying real amplitude at its boundary. `edge jump` in
S5.1's table is the largest nearest-neighbour `|E|` step, over the peak, within
6 coarse cells of the hull - a grid-local measure of the discontinuity the cut
leaves, restricted to the boundary band so the beam's own structure cannot
enter it.

**1.0 cell is the smallest feather that measurably improves on the hard cut**
(3.067e-02 -> 2.395e-02; 0.25 and 0.5 cells are sub-pixel on this fixture and
change nothing), and it gives up **5 %** of the manufactured light removed to
get it. Wider is a straight trade against the fix: 4.0 cells buys another 2x of
edge for **21 %** less manufactured light removed.

### 6.3 And the ringing question, answered at the exit plane

Read the OFF row of S5.1 first. **The unbounded field's own largest step in
that band is 1.366e-01 of peak**, so even a hard binary cut leaves an edge
**4.5x smoother** than what it removed, and the shipped feather **5.7x**. The
"a hard binary mask on the exit grid may diffract" concern is real in principle
and does not arise here, because what the mask deletes is jagged manufactured
light and what it leaves behind is smoother than the original.

On design 121 the question does not arise at all: the bound first bites at
**6.2975 mm**, which is the INNER EDGE of the manufactured annulus itself
(6.298-7.216 mm, `REMAP_STATIONARY_PHASE_FIT_GUARD`'s own note). What is
removed is the whole lobe; what is left inside the boundary is the beam's own
skirt. And on every clean configuration measured the cut moves nothing at all -
the C6-off on-axis call is byte-unchanged in `g4`, `amax4` and `r_rms`, and all
six synthetic fixtures reproduce their unbounded values to every printed digit.

**What is NOT claimed.** This is an exit-plane measurement of amplitude
steps, not a propagation. The co-moving grid is ~15x short of
`lambda/(2 NA_exit)` (S1.4), so no angular-spectrum claim is made about the
cut. The end-to-end evidence that the cut does not cost spot quality is S4.3
and S4.4 - the library's own readout, through the real optics - where EE3, EE6,
FWHM and the at-plane acceptance are unchanged to their last printed digit.

---

## 7. Contracts

### 7.1 The fail-before switch, against a shadow module built from `HEAD`

**`probe_c6_byte_identity.py` and `probe_c6_tilted_failbefore.py` are now STALE
and this is a property of the commit, not of this change.** Both were written
when `HEAD` predated niche C6; their contract is "`REMAP_STATIONARY_PHASE_
LAUNCH = False` reproduces the committed library", and since `8e7b156` the
committed library ships that flag **True**. `probe_c6_byte_identity.py` says so
itself on its own header line - `HEAD has REMAP_STATIONARY_PHASE_LAUNCH: True
(must be False)` - and then measures the C6 delta rather than an identity
(max `|dE|` up to 6.4e-01 of peak on the synthetic remap cases, 1.4e-01 on the
design-121 chain). **They were left untouched.**

The contract they used to state is restated for C8 on the same case matrix,
against the same mechanism (`git show HEAD:lumenairy/elements/_lens_traced.py`
-> a temp file -> imported as a shadow module inside the real package, so a
single changed bit anywhere in the returned field shows), with **every other
flag at its shipped default, which is `HEAD`'s default**:
`probe_c8_byte_identity.py`.

| arm | cases | bound OFF |
|---|---|---|
| (a) synthetic | 12: all `preserve_input_phase` x `amplitude_model` combinations + `'remap'` + lattice + no-carrier, at `rs` 1 and 4 | **`array_equal=True`, `max\|dE\| = 0.000e+00`, 12/12** |
| (b) design 121 chain (0,0) | 7: two grids, two subsamples, 3- and 5-group runs, both readout paths, `final_leg='exact'` | **`array_equal=True`, `max\|dE\| = 0.000e+00`, 7/7** |
| (c) design 121 chain (-4,-2) | the same 7 | **`array_equal=True`, `max\|dE\| = 0.000e+00`, 7/7** |

**26 of 26. With `REMAP_INVERSE_SUPPORT_BOUND = False` the working tree IS
`HEAD`, bit for bit**, against a reference that is a separately-imported
compilation of the committed source rather than a re-run of the same code.

**`probe_c6_tilted_failbefore.py` is a different case and is NOT stale**: it
sets `REMAP_STATIONARY_PHASE_LAUNCH = False` on the SHADOW module as well as on
the live one, so it still compares like with like. Re-run with the bound off:
**`array_equal=True`, `max|dE| = 0.000e+00` on all four of (-4,-2), (-4,0),
(-1,0) and (0,0)** - `OK -- the fail-before restores prior behaviour for TILTED
orders`.

### 7.2 What moves with the bound ON

On the same instrument, `REMAP_INVERSE_SUPPORT_BOUND = True` against `HEAD`:

| case | `array_equal` | `max\|dE\|` | of peak | `dP`/P |
|---|---|---|---|---|
| (0,0) `RN=1024 rs=4 paraxial` | False | 1.364e-01 | **7.70e-01** | **-3.40e-03** |
| (0,0) `RN=1024 rs=2 paraxial` | False | 1.734e-01 | **9.78e-01** | **-4.48e-03** |
| (0,0) `RN=2048 rs=4 paraxial` | False | 1.755e-01 | **9.90e-01** | **-1.05e-02** |
| (0,0) `RN=1024 rs=4, 3 groups` | **True** | 0.000e+00 | 0 | 0 |
| (0,0) `RN=1024 rs=4, 5 groups` | **True** | 0.000e+00 | 0 | 0 |
| (0,0) `RN=1024 rs=4 focus_readout` | False | 2.052e-03 | 1.16e-02 | -1.93e-03 |
| (0,0) `RN=1024 rs=4 final_leg=exact` | **True** | 0.000e+00 | 0 | 0 |
| **(-4,-2), all 7 cases** | 4 True / 3 False | <= 1.683e-05 | **<= 9.49e-05** | **<= 7.35e-11** |

Read it as a map of the defect rather than of the fix.

* **The 3-group and 5-group chains are byte-identical**: the ghost is made in
  the SIXTH group and nowhere else, so a chain that stops short of it has
  nothing for the bound to remove. That is a sharper localisation than any
  measurement in the two predecessor audits.
* **On the finer grid the defect is bigger, not smaller**: `RN=2048` carries
  **1.05e-02** of the chain's power as manufactured light against `RN=1024`'s
  3.40e-03, at 99 % of the peak amplitude. A finer grid resolves more exit
  pixels beyond the traced support, so there is more of it to invent. This is
  the same non-convergence the energy audit found along `ray_subsample`, on a
  second axis.
* **The tilted order is untouched**: at (-4,-2) the largest difference anywhere
  is 9.49e-05 of peak and the largest power change 7.35e-11, across all seven
  configurations - four of which are byte-identical outright. The bound is a
  no-op where there is nothing manufactured.
* The (0,0) difference is 77-99 % of peak and sits at 4-8 mm; the power it
  carries is exactly the manufactured excess the energy audit measured.

The complete accounting is: the bound removes manufactured light on
design 121's (0,0), (-2,0) and (-3,0) relay exits, is inert on (-1,0),
(-4,0) and (-4,-2), is inert on every C6-off configuration, is inert on all six
synthetic fixtures and on every P2 battery cell, and removes 2.5e-03 of the
input power on niche D6's decentred fixture - 100 % of which is outside the
convex hull of every alive traced ray of that call.

### 7.3 Both `newton_fit` backends

Pinned by `test_both_newton_backends_get_the_same_support`: the support comes
from the traced samples before the fit-domain restriction, so the polynomial
and spline backends are bounded by the same hull. D7's contract (the polynomial
must track the unrestricted spline map to < 1e-3 of peak; the shipped tree
reads 8.6e-06) is asserted to hold **with** the bound, and the bound is
asserted not to drive the two apart.

### 7.4 The halo self-check does not start firing anywhere

`halo_calibration.py PART=batt` at the CI subsample, with the bound on: every
one of the 19 P2 battery readings is **0.000e+00** at the shipped radius factor
1.25 (the worst clean reading recorded for the battery is 1.6e-05, so the bound
made it quieter still), and the three cells `C6_FIT_GUARD_DECISION` S5.1
records by name reproduce **exactly** - `triplet-w1.6mm-ap2.5x` at `rs=4` reads
`0.99933 / 0.94477 / 1.04374` across its three groups, digit for digit -
**so the bound removes nothing measurable on the battery**. The check can only go
quieter under the bound in general: `amax_halo` is a max of `|E_out|` outside a
radius the bound does not change, over a peak that lies in the core, and the
bound only ever lowers `|E_out|`.

Measured firings: design 121's on-axis element call **fires with the bound off
and is silent with it on**; the two fit-guard regression fixtures fire without
the bound and are silent with it; niche D6's fixture fires without and is
silent with. **No new firing anywhere.**

---

## 8. Is `REMAP_STATIONARY_PHASE_FIT_GUARD` now redundant?

**On every case measured, yes - and it is NOT removed.**

| | fit guard | **C8 support bound** |
|---|---|---|
| (0,0) production `elem(last)` | 0.995984 | **0.996026** |
| (0,0) production `g4` / `amax4` | 0 / 0 | **0 / 0** |
| (0,0) production EE3 | 87.436 (**-0.964**) | **88.397 (-0.003)** |
| (0,0) production EE6 | 98.056 (**-0.590**) | **98.645 (-0.001)** |
| (-2,0), (-3,0) | **cannot reach** (off-centre branch) | **repaired, 6/6** |
| (-4,-2) `rs=2` with C6 OFF | cannot reach (gated on C6) | **repaired, 6/6** |
| six synthetic fixtures | **2 regress**, one to `P/Pin` 1.00697 | **0 regress** |
| niche D6's fixture | not measured; gated on C6, which D6 does not use | **lobe removed exactly** |

The guard is dominated on every axis that was ever measured on it. It stays in
the tree for one reason that is not sentiment: **the two act on different
objects.** The guard changes the FIT (it routes a concentric disc through D1's
weighted restriction and D7's raised order); this bound changes what the
library is willing to CLAIM from the fit, and leaves the fit itself
byte-identical. A defect that deposits energy **inside** the traced support -
the fit being badly conditioned where it does have data - is invisible to a
support bound by construction, and the guard is still the lever for it. Its
default stays `False`, its note now records the comparison above, and running
the two together is measured rather than assumed (S4.2 and the `C6 on + C8 +
gd` row of `probe_c8_support_bound.py`): they compose without interacting.

---

## 9. Tests, lint, and what shipped

### 9.1 Library changes

`lumenairy/elements/_lens_traced.py` **only**:

* `REMAP_INVERSE_SUPPORT_BOUND = True` - the flag, with the full measured
  record in its note;
* `_SUPPORT_BOUND_FEATHER_CELLS = 1.0` - with S6's measurement;
* the support-hull construction (~30 executable lines) beside the v5.32 halo
  hull, before the fit-domain restriction;
* `_support_taper` (~25 lines) and one multiply in `_ray_density_amp_grid`;
* cross-references added to `REMAP_STATIONARY_PHASE_FIT_GUARD` and
  `_REMAP_RESID_FREEZE_MARGIN`, both of which named this fix as unattempted.

**No signature moved, no other default flipped, no public entry point added.**
`lumenairy/propagators/carrier.py` is unmodified (hash `1a90453a4ef65399`, the
same one both 2026-07-31 audits record). `CHANGELOG.md` and
`lumenairy/elements/pmm/**` were not touched.

### 9.2 Tests added

`tests/unit/test_niche_c8_inverse_support_bound.py`, 13 tests, ~30 s, no
proprietary asset - self-contained singlets with a converging carrier and the
same `r^4` residual the C6/C7 fixtures use:

1. the defaults, and that the OFF state really is a switch (the feather
   constant is byte-inert with the flag off);
2. **fail-before**: the bound removes a lobe on a fixture where the library
   manufactures one by itself, with the fail-before arm ASSERTED to still
   manufacture it, and the energy self-check asserted silent in both
   directions - so this is not a restatement of the power guard;
3. the halo self-check fires before and is silent after, and is silent both
   ways on a clean call;
4. it never raises any pixel's amplitude (clean fixture and ghosting fixture);
5. it leaves the beam **byte-identical** inside 3 w and the total power within
   1e-9;
6. `amplitude_model='screen'` is byte-identical (the scope clause);
7. both `newton_fit` backends get the same support (D7's < 1e-3 contract, plus
   "the bound must not drive them apart");
8. a **decentred** beam is not cut - the support follows the beam, which is the
   case a grid-referenced radius gets wrong and the case niche C4's transpose
   defect hid in;
9. the feather is pointwise monotone in its width and a hard cut is the most
   aggressive setting;
10. a hard cut still removes the lobe (the feather is not load-bearing for the
    fix);
11. a degenerate support declines instead of raising.

### 9.3 ONE EXISTING TEST WAS CHANGED, and it is the one this fix breaks

`tests/unit/test_niche_c7_ray_density_halo_check.py::
test_fires_on_the_fit_guard_regression`.

**The change is not to a bar.** That test's positive control is a manufactured
lobe, produced by turning `REMAP_STATIONARY_PHASE_FIT_GUARD` on over a fixture
where it regresses; it then asserts that the v5.32 halo self-check FIRES. C8
removes that lobe **at source**, so the detector correctly reports nothing and
the test fails for an honest reason: **its stimulus is gone.**

What now happens:

* the fires-on-a-lobe arm is measured with `REMAP_INVERSE_SUPPORT_BOUND =
  False` - the library state the check was calibrated in - so the original
  assertion and its "guard off is silent, guard on warns, and the energy band
  is quiet in BOTH" pin are intact, word for word;
* and **three new assertions were added**, not removed: with the bound ON the
  same call is silent, its peak has not risen, and its total power has strictly
  fallen. That is the pin that stops a true positive being silently replaced by
  a green test - the failure mode this exact test file exists to prevent.

`_call` gained a `bound` parameter defaulting to `None` (leave the shipped
default), so **the other fourteen tests in that file still score the shipped
path**, including every no-false-positive test. No tolerance, radius factor or
amplitude bound was touched.

**Nothing else in `tests/` was modified.** In particular, niche D6's
`FWHM_px > 1.70 FWHM_orc` discriminator - which an earlier revision of this fix
narrowed from 1.857x to 1.762x, and which had already been re-priced once at
niche C3 - **needed no change**: the plateau of S5.2 restored it to **1.857x
exactly** (5.850 -> 5.850 um, i.e. the paraxial FWHM does not move at all), and
all twelve of D6's dependent assertions pass with the lobe present and removed.

### 9.4 Suites and lint

```bash
python -m pytest tests/unit/test_niche_c1_*.py tests/unit/test_niche_c3_*.py \
    tests/unit/test_niche_c5_*.py tests/unit/test_niche_c6_*.py \
    tests/unit/test_niche_c7_*.py tests/unit/test_niche_c8_*.py \
    tests/unit/test_niche_d1_*.py tests/unit/test_niche_d2_*.py \
    tests/unit/test_niche_d3_*.py tests/unit/test_niche_d6_*.py \
    tests/unit/test_niche_d7_*.py -q
-> 329 passed, 71 warnings in 1189.90s        (the FINAL library)

python -m ruff check lumenairy/ tests/unit/
-> All checks passed!
```

**329 passed, zero failures.** The 71 warnings are the pre-existing physics
diagnostics the suite is documented to emit, and **not one of them is a halo
self-check firing** (`grep -c "HALO self-check FAILED"` -> 0). The C7 study's
run of the same suite emitted exactly one - niche D6's manufactured-lobe true
positive, which it reported as an open finding. **It is gone because the lobe
is gone**, and the fold-caustic warning on the same call still fires, so the
diagnostic that describes the underlying conditioning is not silenced with it.

The first run of that suite on this change read **2 failed, 327 passed** and
both failures are the ones S9.3 and S11.2 describe - the C7 positive control
whose stimulus C8 removes, and an over-tight bar in **this study's own new
test**. Both are recorded rather than quietly repaired.

---

## 10. What remains unmeasured

1. **Where the removed light would have gone.** Every halo figure is at an
   element's own exit plane; the co-moving grid is ~15x short of
   `lambda/(2 NA_exit)`. This does not weaken the conservation conclusions -
   manufactured energy is manufactured wherever it lands - but no image-plane
   claim is made.
2. **The FIT is not repaired.** Outside the hull the fitted map is still wrong
   and the two backends still disagree there; inside the hull the fit's error
   is untouched (byte-identical fits, byte-identical Newton, byte-identical
   OPL). A defect that deposits energy INSIDE the traced support is invisible
   to this bound by construction, exactly as it is to the v5.32 halo check.
3. **The support is CONVEX.** For a prescription whose exit fan is genuinely
   non-convex (an obscured or annular pupil, a strongly folded map) the bound
   is looser than the true support - never tighter, so never wrong, but weaker.
   No such prescription was measured.
4. **`ray_subsample` was measured at 4 and 2 on design 121** and at 1, 4 and 8
   on the P2 battery. The library's shipped default is 8; design 121 does not
   use it.
5. **The EE cost was priced on three orders**, (0,0), (-4,0) and (-4,-2), and
   the production configuration on (0,0) only (~6 min and ~50 GB per row).
6. **Only the post-DOE relay and the D6 fixture are scored.** Chain A
   (source -> DOE) and `propagate_traced_carrier_chain_multi`'s readout tiling
   and recombination are untouched.
7. **The `d0` plateau costs `sqrt(2) sub dx` of reach unconditionally**,
   including at `ray_subsample = 1` where there is no upsample and it is not
   needed (1.41 pixels of unnecessary looseness). Measured to cost nothing on
   any fixture here; not made conditional, because a second code path would
   need its own calibration.
8. **The exact-ray oracle's absolute reference was used qualitatively, not as
   a bound.** S5.1 quotes the D6 chain-exit ratio crossing unity and returning
   below it; no acceptance criterion in this document is stated against the
   oracle's absolute 1.0000.

---

## 11. Artefacts found and killed in my own instruments

1. **THE FIX'S OWN CONTRACT WAS WRONG AND MY OWN PARTITION CAUGHT IT.** S5.2:
   the coarse-lattice taper bled inward through the bilinear upsample, removing
   3.808e-05 of the input power of legitimate skirt at the shipped feather -
   1.1 % of everything the bound removed on that fixture - while the note in
   the source claimed "every pixel with traced data behind it keeps its full
   amplitude". Caught by building the (c) column *because a reviewer asked
   about the scale of the removal*, not because I suspected the mechanism.
   **The instrument aimed at the fix's magnitude found an error in its
   definition.**
2. **The first feather calibration was measured against the wrong length and
   its conclusion reversed.** The feather was initially expressed in COARSE
   GRID cells (`sub * dx`) and the table showed a plateau at 1.0 cell; when it
   was re-expressed in EXIT-LATTICE cells (the ray landings' own separation,
   which is the resolution at which the support is known) the plateau vanished
   and the numbers changed by 7x. After the S5.2 plateau fix the entire
   justification changed AGAIN - from "protects legitimate light" to "smooths
   the residual edge" - because the plateau took over the first job. Both
   earlier versions of the note were written, measured, and thrown away. **Do
   not write the justification before the final implementation is measured.**
3. **A warning census silently swallowed every halo firing.** The first version
   of `probe_c8_support_bound.py` classified warnings with
   `elif 'energy self-check' in t` before the halo test - and the halo message
   itself contains the sentence "Note the energy self-check CANNOT see this".
   It reported **0 halo firings on a field whose halo is 33 % of peak**, which
   read as "the bound already fixed it" for one whole run. Caught because the
   C6-off row also read 0 when the C7 record says it should. The ordering is
   now load-bearing and commented as such.
4. **`p_exit/p_in` was printed as 530.** `_run_chain` returns a tuple and the
   readout field is on a different lattice, so a naive `sum |E|^2` over it is
   not a power ratio at all. Caught immediately because the number was absurd;
   the column was deleted rather than "normalised" by guesswork, and the
   absolute chain-exit ratios in S5.1 are quoted from another agent's
   instrument, which does the bookkeeping properly.
5. **A 25-minute production run was nearly quoted against the wrong baseline.**
   The `ship` (0,0) production row was measured at C8=0 *before* the plateau
   fix and at C8=1 both before and after; the pre- and post-plateau C8 rows
   happen to agree to every printed digit, which would have made a stale number
   invisible. Both were re-run on the final library rather than reasoned about.
6. **The byte-identity probe died silently mid-run** (Windows memory pressure -
   the documented failure mode of this box) and its pipe reported exit 0, so a
   truncated output nearly passed as a completed contract. Re-run to a log file
   with the parts split; the "OK" line is the only accepted evidence.
7. **MY OWN NEW TEST CARRIED A BAR THAT THE FINAL IMPLEMENTATION DOES NOT
   MEET, and it was written before the plateau existed.** `test_it_removes_a_
   lobe_the_energy_self_check_cannot_see` asserted a 100x drop in amplitude
   beyond 3 w; the shipped bound gives **51.5x** (4.593e-02 -> 8.911e-04),
   because the plateau deliberately keeps a `sqrt(2) sub dx` band outside the
   hull at full weight and 3 w on that fixture lies inside it. The bar is now
   20x **with the reason in the source**, and the 100x statement was moved to
   the POWER statistic where it is true with 5x to spare (2.576e-05 ->
   4.945e-08). Recorded rather than quietly relaxed: a bar lowered to fit a
   result is exactly the failure this project keeps naming.
8. **The two stale C6 probes were nearly reported as failures of this change.**
   `probe_c6_byte_identity.py` prints `array_equal=False` on 17 of its 29 arms
   on this tree - with the C8 bound OFF. That is entirely the C6 commit, which
   moved `HEAD` under a probe whose premise is that `HEAD` predates C6 (S7.1).
   Settled by writing a C8-specific probe rather than by editing theirs or by
   quoting theirs as a regression.

---

## 12. Reproduction

All commands from `validation/repro_traced_carrier_121/`. Every runner prints
the sha256 of the library file it imported.

```bash
# S4.5 -- the six-bound table, bound ON (C8=0 gives the HEAD reference)
C8=1 ORDERS='0,0 -1,0 -2,0 -3,0 -4,0 -4,-2' CONFIGS='ship,noC6' \
    python c8_with_bound.py energy_stage_audit_121.py
C8=1 RS=2 NULL=0 ORDERS='0,0 -2,0 -4,-2' CONFIGS='ship,noC6' \
    python c8_with_bound.py energy_stage_audit_121.py

# S4.1 -- both currencies on ONE production field.  ~6 min, ~50 GB per row:
#         do NOT run concurrently with another chain batch.
C8=0 LUMEN_PIN=0 NULL=1 ORDERS='0,0' CONFIGS='ship,noC6' \
    python c8_with_bound.py energy_ee_vs_conservation_121.py
C8=1 LUMEN_PIN=0 NULL=0 ORDERS='0,0' CONFIGS='ship,shipG' \
    python c8_with_bound.py energy_ee_vs_conservation_121.py

# S4.3 -- per-order EE3 against the exact-ray oracle
C8=1 ORDERS='0,0 -4,0 -4,-2' CASES='oracle,deg4' \
    python c8_with_bound.py probe_c6_chain.py

# S4.4 -- the single-beam at-plane acceptance
C8=1 python c8_with_bound.py focus_scan_121.py

# S4.2 / S6.1 -- the six synthetic fixtures, and the element-level feather sweep
python probe_c8_synthetic.py
ORDERS='0,0 -2,0 -4,-2' FEATHERS='0,0.25,0.5,1,2,4' \
    python probe_c8_support_bound.py

# S5 -- the over-removal partition and the edge-jump feather metric
LEG=paraxial FEATHERS='0,0.25,0.5,1,2,4' python probe_c8_d6_overremoval.py
PART=D python d6halo_controls.py          # D6's twelve dependent assertions

# S7 -- the contracts
python probe_c8_byte_identity.py
C8=1 RS=4 PART=batt python c8_with_bound.py halo_calibration.py
python -m pytest tests/unit/test_niche_c8_inverse_support_bound.py -q
python -m ruff check lumenairy/ tests/unit/
```

### Files added by this study

`validation/repro_traced_carrier_121/probe_c8_support_bound.py`,
`probe_c8_synthetic.py`, `probe_c8_d6_overremoval.py`,
`probe_c8_byte_identity.py`, `c8_with_bound.py`,
`tests/unit/test_niche_c8_inverse_support_bound.py`, and this document.
**No existing runner was edited.**
