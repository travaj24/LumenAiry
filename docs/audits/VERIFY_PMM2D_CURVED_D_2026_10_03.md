# VERIFY -- curved cells for the pure staggered 2-D PMM, Phase D (anisotropic and magnetic materials inside curved cells)

Date: 2026-10-03.  Verifier: Claude Opus 5.5 (model ID `claude-opus-5-5`),
independent adversarial verification.
Mount: worktree `C:/tmp/lum_vcurved_d`, branch `verify/pmm2d-curved-d`, on the
builder's tip `eae470d9` (`feat/pmm2d-curved-d`, commits `074226d2` ..
`eae470d9` on the Phase C tip `607d43b0`).  PRE tree: the verifier's own
`git archive 607d43b0` in `C:/tmp/vcd_pre_607d` (not the builder's).
Builds: Windows 11 (CPython 3.14.6, numpy 2.4.4, scipy 1.17.1) and WSL Ubuntu
(CPython 3.12.3, numpy 2.4.6, scipy 1.17.1, `~/lumvenv`), BLAS pinned to one
thread on the command line, `lumenairy.__file__` asserted under the tree being
measured in every probe (`verify_d/_vdcommon.py`).  The box was saturated by
four sibling agents for the whole run (load 66-100 %), so every wall time
below is an upper bound.
Evidence: `validation/probe_pmm2d_curved/verify_d/` (probes + JSON);
decision tests `tests/unit/test_verify_pmm2d_curved_d.py` (4 tests, 19.9 s).
Nothing under `lumenairy/` was edited.

---

## 0. Verdict

| # | builder claim | verdict | the verifier's own measurement |
|---|---|---|---|
| 1 | D1: no map = `607d43b0` bytes; scalar maps = Phase C bytes | **CONFIRMED** | own 176-key set (14 operator fixtures x 10 blocks incl. every 5.43/5.44 tensor and magnetic class, the Li 2003 cell, out-of-plane and slant, 10 Jones entries, two stacks with absorption, the rectangles-only shapes route, 16 MAPPED SCALAR keys): **176 / 176** SHA-256 identical on Windows AND **176 / 176** on WSL (pre vs post per build); the same run with the three Phase D kernels made to RAISE still produces all 176 keys, identical (`v1_compare.json`) |
| 2 | the congruence `eps'_t = adj(J) eps_t adj(J)^T / sg`, `eps'_33 = sg eps_33`, `chi_t = J^T mu_t^-1 J / sg`, `chi_33 = 1 / (sg mu_33)` and its index placement | **CONFIRMED (derived and measured)** | derivation s 2; every tensor class on a NON-DIAGONAL-J sheared map against the verifier's own eps+mu Berreman (complex Jones, normal / oblique / conical): round-off (2e-15 .. 1e-13); the side swap, transpose, missing `1/sg`, `chi` transposed / side-swapped / without `sg` each move the answer by 1e-5 .. 0.3 (s 3) |
| 3 | D2 tensor films under a stretch and a shear vs Berreman | **CONFIRMED on own fixtures** | asymmetric two-harmonic stretch (36:1 / 12:1 local range), LC at 0/30/45/90 deg, a biaxial tensor, a real-asymmetric non-Hermitian tensor, a lossy tensor, gyrotropic: spectral to 1e-9 .. 1e-8 at M = 8; a 4 x 4 random shear: round-off from M = 3 (normal) / M = 5 (conical) |
| 4 | D4 LC30 pillar: RCWA 1/N, topologies agree 2.5e-6 | **CONFIRMED, one precision note (N-2)** | RCWA exact-disk ladder N = 9 .. 33 (own Bessel form factor; self-check vs shipped shapes RCWA 7e-15): distance x N = 0.082 .. 0.088 (flat to +-4 %); c3 extended to M = 11 / 12: the c3 ladder still moves 3.5e-6 / 2.8e-6 per rung and c3 M = 12 sits 3.8e-6 from the builder's c5 M = 8 -- the curved answer is fixed to ~4e-6, not 2.5e-6 |
| 5 | D5 Li 2003 under the identity map 8.74e-5, swapped row 2 | **CONFIRMED against the ORIGINAL paper** | Table 1 read from the PDF (J. Opt. A 5:345, p. 353): identity map 8.739349638e-5 (unmapped 8.739349637e-5), swapped cell hits row 2 at 8.739e-5 and misses row 1 by 1.319e-2; a 4 x 4 non-diagonal-J map that keeps every material wall straight lands on the unmapped same-grid solve to 2.8e-7 (M = 7) |
| 6 | D6 magnetic: film, duality (not a discrete identity), mu inverse before quadrature | **CONFIRMED; F-D4 rate measured** | gyro / symmetric off-diagonal / lossy mu films vs own Berreman to round-off; duality with tensor mu (gyro, off-diagonal) falls 3.6e-3 -> 2e-6 at the size of the two twins' own rung changes (non-monotone, as F-D4 says); after-quadrature inverse 0.19 / 0.58 (stretch) and 0.082 / 0.22 (circle), FLAT in M, against 3e-10 / 2e-11 correct |
| 7 | D7 tensor + magnetic merged stack | **CONFIRMED on own shapes** | gyrotropic ellipse + coaxial magnetic circle (off-diagonal mu) + uniform gyro-mu film: closure 4.0e-3 / 1.1e-4 / 1.3e-5 (M = 3 / 4 / 5); lossless layers absorb <= 1.6e-14 with a lossy eps or a lossy mu elsewhere; sum of layer absorptions vs 1 - R - T 1.0e-5 .. 1.5e-5 at M = 5 |
| 8 | D8 reciprocity; gyrotropic disk 1.6e-2 is physics | **CONFIRMED -- physics, decided two ways** | Onsager with the magnetization REVERSED (fwd GYRO vs rev GYRO^T): 1.5e-4 -> 6.1e-8 (c3 M = 5 .. 8), 3.3e-9 (c5 M = 6), 4.9e-13 (staircase) -- while fwd GYRO vs rev GYRO is 1.55e-2, CONVERGED (c3 M = 8 1.556e-2, c5 M = 5 / 6 1.554e-2); staircase devices 7.6e-3 (k = 1), 2.52e-2 (k = 2), 1.26e-2 (k = 4), see s 8 |
| 9 | the 19 tests + D9 mutation matrix | **one survivor class + two meta-survivors, CLOSED** | 15 engineered defects through the real code path; every one caught by a named id EXCEPT `chi_T` (mu transposed in chi: every Phase D gate uses a diagonal mu), the D-1 tolerance x100 and the walker reason removal -- closed by `test_verify_pmm2d_curved_d.py` (s 9) |
| 10 | docs | **CONFIRMED with four small corrections** | BUILD_D numbers spot-checked against JSON (~40 numbers, all agree); cookbook LC-hole and CHANGELOG examples execute verbatim; mypy clean; WSL ruff clean on the library and tests; fingerprints `--check` OK; defects D-1 .. D-4 are wording (s 10) |
| F-D3 | `geom[4]` feeds two consumers | **NOT a latent mismatch -- a shared object; pinned** | both consumers need the identical operator (the inverse PLAIN (u, v) block Gram); decision test pins its identity (`geom[4] Gp = I` to 4.7e-14; 0.40 relative away from `inv(-R)`) |

**Ship recommendation for Phases A-D: SHIP** (after the four wording fixes of
s 10, none of which moves a number).  No physics defect was found.

---

## 1. D1 -- byte identity, the verifier's own set

`v1_bytes.py` (run in BOTH trees with the same file).  Fixture choices are
independent of the builder's: a random lossy 4 x 4 scalar cell, a non-uniform
4 x 4 cell at oblique incidence, LC at 45 deg, a rotated biaxial 3 x 3 at
conical incidence, the Li 2003 cell (periods 2.4 x 1.4), a gyrotropic tensor
of the opposite gauge, a REAL ASYMMETRIC tensor, a lossy tensor, scalar mu, a
gyrotropic mu with an LC eps at conical incidence, a lossy mu, an out-of-plane
uniaxial tensor (normal and oblique -- `_region_modes_oop`), a slant cell;
`pmm_efficiency_2d_staggered` TE/TM conical; ten `pmm_jones_2d_staggered`
calls (tensor, gyro, real-asym, lossy, gyro-mu, scalar mu, mu half-spaces,
OOP auto, OOP oblique, slant); a four-layer stack (uniform biaxial, patterned
magnetic, uniform lossy mu, patterned gyro) with absorption; the
rectangles-only shapes route (scalar, with absorption; and a TENSOR rectangle
through `shapes=`); the explicit per-layer tensor and tensor+mu cells; and
16 mapped scalar keys (a 5 x 5 circle solver's operators at oblique
incidence, a shape circle normal / conical with absorption, a rotated
ellipse, a sine-stripe map stack with absorption).

| | keys | identical |
|---|---|---|
| Windows, pre vs post | 176 | **176** |
| WSL, pre vs post | 176 | **176** |
| Windows post with `_stag_map_eff_tensor`, `_stag_map_weights_tensor`, `_stag_map_as33` RAISING | 176 | **176** (every key produced; equal) |

The shapes route with mu on a RECTANGLE (`Rect(..., mu=1.6)`, post only) runs
the unmapped solver and never reaches a Phase D kernel (the trapped run
produced it); it equals the explicit per-layer `eps_cell` + `mu_cell` solve to
1.1e-8, the same 6.4e-9-class gap the TENSOR rectangle has to its explicit
per-layer twin in BOTH trees (`extra` in `v1_bytes_*.json`) -- a pre-existing
shared-vs-per-layer path difference, not Phase D.

Mutant (brief item 1): the three kernels raising for the whole session on the
UNMAPPED sweep (55 files) -- s 12.

---

## 2. The transformation, derived

Time convention `exp(-i w t)`: `curl E = i w mu0 mu H`, `curl H = -i w eps0
eps E`.  Map `r = (Phi(u, v), w)`, `Lambda = d r / d(u, v, w) = blockdiag(J,
1)`.  Covariant components `E'_a = E . d r / d u^a`, i.e. `E' = Lambda^T E`.
For a covariant 1-form the exterior derivative is a 2-form, whose components
are a vector DENSITY: `curl'(Lambda^T E) = det(Lambda) Lambda^-1 curl E`.
Then

    curl' E' = det L L^-1 (i w mu0 mu H) = i w mu0 [det L L^-1 mu L^-T] (L^T H)

so `mu' = det L L^-1 mu L^-T` and likewise `eps'` (Ward & Pendry 1996;
the textbook form `A eps A^T / det A` with `A = d u / d r = L^-1`).  With
`L = blockdiag(J, 1)`, `det L = sg`, `J^-1 = adj(J) / sg`:

    eps'_t  = sg J^-1 eps_t J^-T = adj(J) eps_t adj(J)^T / sg
    eps'_33 = sg eps_33
    mu'_t   = adj(J) mu_t adj(J)^T / sg  =>  chi_t = (mu'_t)^-1
            = sg adj(J)^-T mu_t^-1 adj(J)^-1 = J^T mu_t^-1 J / sg
    chi_33  = 1 / (sg mu_33)

(`adj(J)^-1 = J / sg`.)  The kernel (`twod_staggered.py:2150`) computes
exactly `e_ij = sum A_ik eps_km A_jm / sg` with `A = adj(J)` (J^-1 on the ROW
index) and `c_ij = sum J_ki K_km J_mj / sg` with `K = mu_t^-1` -- confirmed
line by line.  For block-form tensors nothing out of plane is created (no
`w`-dependence of the map).

**Index placement, measured** (`v2_film.py`, `v2_summary.json`): a 4 x 4
TransfiniteMap with the nine interior vertices moved by up to 7 % of the
period at random (`shear4`: bilinear cells, `g12 != 0`, NON-diagonal `J` --
a different pattern and grid from the builder's 3 x 3 shear) against the
verifier's OWN eps+mu Berreman 4x4 (`_vdcommon.berreman_eps_mu`; equal to
the shipped `berreman_jones_1d` to **6.1e-15** on eps-only stacks INCLUDING
the complex reflection Jones, 8 tensor classes x 4 incidences,
`v0_oracle.json`; its mu path equals the closed-form (eps, mu) Airy slab to
4.4e-16).  Error = largest |R, T| (orders summed per input) / largest
complex Jones entry, M = 4 (`sh4` is exact at normal incidence: the
covariant plane wave is bilinear per cell):

| tensor | incidence | correct | eps TRANSPOSED | SIDES swapped (`J^-T eps J^-1`) | no `1/sg` (`adj eps adj^T`) | `eps'_33` without `sg` |
|---|---|---|---|---|---|---|
| GYRO_V (Hermitian, opposite gauge, `e11 != e22`) | normal | 2.4e-14 / 8.6e-15 | 2.7e-14 / **0.17** | **9.6e-4 / 8.1e-3** | **2.4e-3 / 1.1e-2** | 1.0e-14 / 6.0e-15 (blind) |
| GYRO_V | conical (20.1, 40.1 deg) | 6.9e-10 / 3.2e-10 | **9.9e-4 / 0.17** | **1.1e-3 / 7.4e-3** | **3.2e-3 / 1.0e-2** | **2.6e-5 / 1.2e-4** |
| RASYM (real, `e12 = 0.45`, `e21 = -0.15`: non-Hermitian AND non-reciprocal) | normal | 1.6e-14 / 8.4e-15 | **0.24 / 0.14** | **3.6e-2 / 1.2e-2** | **1.6e-2 / 1.5e-2** | blind |
| RASYM | conical | 7.9e-10 / 1.6e-10 | **0.29 / 0.15** | **3.9e-2 / 1.3e-2** | **1.6e-2 / 1.6e-2** | **2.0e-4 / 1.3e-4** |
| LOSSY (rotated complex biaxial) | normal | 1.3e-14 / 5.1e-15 | 1.2e-14 (symmetric: no-op) | **2.1e-2 / 7.8e-3** | **6.6e-3 / 6.4e-3** | blind |
| biaxial (rotated 20 deg) | normal | 1.1e-14 / 4.5e-15 | no-op | **2.7e-3 / 8.2e-3** | **4.1e-3 / 1.2e-2** | blind |

and for a gyrotropic PERMEABILITY (eps 2, `MU_GYRO`, same map, M = 4):

| incidence | correct | mu transposed in chi | chi sides swapped (`J mu^-1 J^T`) | `chi_33` without `sg` | mu ignored |
|---|---|---|---|---|---|
| normal | 2.1e-15 / 7.8e-15 | 5.4e-15 / **0.14** | **6.2e-4 / 8.3e-3** | 7.9e-15 (blind) | **7.2e-3 / 7.1e-2** |
| conical | 2.8e-10 / 3.2e-10 | **2.6e-3 / 0.12** | **7.2e-4 / 6.6e-3** | **1.3e-5 / 2.0e-4** | **9.0e-3 / 5.9e-2** |

* **The real asymmetric tensor** (asked for explicitly) is solved to round-off
  under the shear, normal (1.6e-14), in-plane oblique 0.4 rad (3.2e-14 at
  M = 6), conical (3.3e-14 at M = 6) and a second conical azimuth (0.5 rad,
  -1.2 rad: 2.5e-14), and under the circle map spectrally (1.1e-11 / 1.7e-12
  at M = 7); unlike the gyrotropic tensor its transpose is visible in R / T
  already at NORMAL incidence (0.24) -- it is not lossless, so no Hermitian
  symmetry hides it.
* **The gyrotropic transpose is R / T-visible at oblique incidence on a
  uniform film** (9.9e-4 against 6.9e-10), contrary to the builder's
  "a transposed gyrotropic tensor leaves a uniform film's R / T unchanged"
  (F-D1, s 2.9, CHANGELOG) -- that holds at NORMAL incidence only.  Doc
  defect D-1 (s 10).
* **A lossy tensor under the circle map, absorption closure vs the unmapped
  limit**: the uniform LOSSY film's `layer_absorption` (internal Gram flux)
  against the oracle's `1 - R - T`: c3 6.0e-13 (normal, M = 7), 4.5e-10
  (conical, M = 7); the same with a lossy mu (`MU_LOSSY`) and an LOSSY eps +
  symmetric mu: 5.3e-13 / 1.5e-12 (normal M = 7); the unmapped (`map none`)
  and sheared (2e-14) limits agree; the stretched film reaches 3.7e-10 at
  M = 8.  Under a map the absorption read-out converges with the solve.

---

## 3. D2 -- the verifier's own stretches, directors and tensors

`h2`: a separable ASYMMETRIC two-harmonic stretch of both axes
(`x = u + 0.09 p sin(k u) + 0.05 p [sin(2 k u + 0.7) - sin 0.7]`,
`y = v - 0.06 p sin(k v) + 0.035 p [sin(2 k v - 1.1) - sin(-1.1)]`; local
stretch ranges 36:1 in x, 12:1 in y), the builder's G3 film geometry, against
the verifier's Berreman (R,T / Jones):

| tensor | M = 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|
| LC director 0 deg (normal) | 4.1e-14 / 3.9e-14 | 8.9e-14 | 1.5e-13 | 2.0e-13 | 6.1e-14 / 2.2e-13 |
| LC 30 deg | 1.8e-3 / 4.7e-4 | 5.2e-5 | 5.2e-6 | 9.9e-8 | **3.6e-9 / 1.1e-9** |
| LC 45 deg | 2.3e-3 / 5.9e-4 | 6.6e-5 | 6.8e-6 | 1.3e-7 | **4.7e-9 / 1.4e-9** |
| LC 90 deg | 3.5e-14 / 1.3e-14 | 1.2e-13 | 2.0e-13 | 2.2e-13 | 1.2e-13 / 1.1e-13 |
| biaxial (rotated 20 deg) | 8.6e-4 / 2.6e-4 | 2.5e-5 | 2.5e-6 | 4.8e-8 | **1.8e-9 / 5.8e-10** |
| GYRO_V | 4.7e-3 / 1.1e-3 | 1.2e-4 | 1.3e-5 | 2.4e-7 | **9.3e-9 / 2.4e-9** |
| RASYM | 8.1e-3 / 1.4e-3 | 4.0e-4 | 2.5e-5 | 6.6e-7 | **1.8e-8 / 3.1e-9** |
| LOSSY | 1.4e-3 / 3.0e-4 | 4.5e-5 | 4.1e-6 | 8.4e-8 | **2.9e-9 / 6.8e-10** |
| conical (0.35, 0.7 rad), LC 30 | 2.1e-3 | 7.7e-5 | 7.6e-6 | 2.5e-7 | 1.6e-8 / 4.4e-9 |
| conical, GYRO_V | 5.7e-3 | 1.9e-4 | 2.0e-5 | 6.6e-7 | 4.3e-8 / 7.5e-9 |

Spectral throughout, about one decade per rung; slower than the builder's
s05 (a stronger, two-harmonic stretch).  A director along x or y at normal
incidence is exact at every M (the field stays E_x- or E_y-only; recorded,
not used as a gate).  Under the 4 x 4 shear every one of the eight tensors
is at round-off from M = 3 at normal incidence and reaches 1e-13 by M = 5 at
conical incidence (s 2); under the circle map c3 all eight reach 4e-13 ..
1.1e-11 (normal) at M = 7.

**The gyro Jones gate is two-sided on the verifier's maps**: the transposed
GYRO_V moves the Jones matrix by 0.17 on `h2` (M = 6, correct 3.3e-6), `sh4`
(correct 8.6e-15) and `c3` (correct 2.5e-10), with R / T unmoved at normal
incidence (1.3e-5 = correct, 2.7e-14, 2.3e-11).  The side swap is invisible
on `h2` (diagonal `J`: 1.3e-5 = correct) and caught on `sh4` / `c3`, as the
builder's F-D1 states.

---

## 4. D4 -- the LC30 tensor pillar

`v4_pillar.py`.  The RCWA exact-disk patch is the verifier's own: the
Toeplitz entries are `bg delta + (d - bg) F(G)` with `F` from the verifier's
OWN Bessel closed form (`disk_ff`, not the shipped `_shape_form_factor`);
self-check on a scalar eps-4 disk against the shipped
`rcwa_efficiency_2d_shapes`: 7.1e-15 (9 orders) / 8.2e-14 (17 orders); the
LC30 run at 9 orders equals the builder's `d4_rcwa_n4.json` to 6.2e-15.

Reference = the verifier's c3 `M = 12` rung (dof 2178, closure 5.9e-10,
1712 s loaded).  Distance = the largest of the 36 entries.

| RCWA orders per axis N | 9 | 11 | 13 | 15 | 17 | 19 | 21 | 23 | 25 | 27 | 29 | 31 | 33 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| distance x N | 0.0819 | 0.0884 | 0.0855 | 0.0860 | 0.0877 | 0.0855 | 0.0874 | 0.0867 | 0.0865 | 0.0876 | 0.0864 | 0.0873 | 0.0871 |

**Flat**: 0.082 .. 0.088 (+-4 %) over N = 9 .. 33, local rates 0.62 .. 1.23
(mean 1.0) -- the RCWA approaches the curved answer as 1/N and nowhere else.
Richardson in 1/N wanders with the pair: (25, 29) 5.4e-5 (the builder's
5.6e-5 reproduced), (29, 33) 1.7e-4, (23, 31) 6.9e-5, (25, 33) 7.5e-5; a
least-squares `a + b/N` fit over N >= 13 lands 7.7e-5 away.  The RCWA family
corroborates the curved answer at the ~1e-4 level (the builder's "1e-4-class
reference" wording is right; "5.6e-5" is the best pair, not a converged
value).

**Is the 2.5e-6 topology agreement converged?** (c3 to M = 12; c5 to M = 7
here, M = 8 from the builder's JSON):

| | value |
|---|---|
| c3 rung changes M = 8 -> 9 -> 10 -> 11 -> 12 | 1.6e-4, 1.0e-5, 3.5e-6, 2.8e-6 |
| c3 M = 11 / M = 12 vs builder c5 M = 8 | 1.0e-6 / 3.8e-6 |
| c3 M = 12 vs own c5 M = 7 | 8.8e-6 |
| own c3 M = 9 vs builder c3 M = 9 | 0 (bit-identical) |
| own c3 M = 11 vs builder c3 M = 10 | 3.5e-6 |

The c3 ladder slows from a decade per rung to 1.3x per rung beyond M = 10
(the Phase B scalar disk shows the same 1e-6-class behaviour at M = 11 / 12
against the FEM, VERIFY_B s 4), and c3 M = 12 sits 3.8e-6 from c5 M = 8.  So
the two topologies agree at 1e-6 .. 4e-6, inside each ladder's last rung
change: the curved answer is known to **~4e-6**, not to 2.5e-6.  Precision
note N-2; no gate depends on it (the unit bar is 1e-3 at c5 M = 4).

---

## 5. D5 -- Li 2003 against the original paper

Read from `PMM_Papers/li_2003_crossed_anisotropic_gratings_JOptA5-345.pdf`
(p. 352 Example 1; p. 353 Table 1).  Example 1: `zeta = Theta = Phi = 0`,
`d1 = 2.4 lambda`, `d2 = 1.4 lambda`, `h = lambda`, `w1/d1 = w2/d2 = 0.5`,
`n(+1) = 1.0`, `n(-1) = 1.0 + i5.0` (an INDEX: `eps = -24 + 10i`),
`eps_a = 2.25(xx + yy) + i0.5(xy - yx) + 2.0 zz` (surround),
`eps_b = 2.25(xx + yy) - i0.5(xy - yx) + 2.0 zz` (pillar), E in Oxz.  Table 1
"Diffraction efficiencies and polarization angles of the grating case in
example 1", columns m = 0, +1, +2, rows n = -1, 0, +1 (computed at truncation
order 23 with L2 L1; the caption of fig. 3 says REFLECTED); first row of
each cell = the grating as defined, second = "the signs of the cross terms
of the permittivity tensors are reversed, i.e. eps_a and eps_b are
interchanged":

| (m, n) | row 1 | row 2 |
|---|---|---|
| (0, -1) | 0.0619 | 0.0619 |
| (1, -1) | 0.0269 | 0.0137 |
| (0, 0) | 0.2980 | 0.2980 |
| (1, 0) | 0.1195 | 0.1195 |
| (2, 0) | 0.0222 | 0.0222 |
| (1, 1) | 0.0137 | 0.0269 |

-- the shipped `_LI_TABLE1` / `_LI_TABLE1_ROW2` are these values, correctly
indexed (m along x).  Measured (`v5_li.py`, incident E_x, REFLECTED):

| arm (the verifier's maps) | M | max dev. row 1 | max dev. row 2 |
|---|---|---|---|
| unmapped (shipped) | 6 / 8 / 12 | 3.241e-4 / **8.739e-5** / 7.12e-5 | 1.32e-2 |
| `IdentityMap` (the tensor route; the builder used an identity TransfiniteMap) | 6 / 8 | 3.241e-4 / **8.739349638e-5** | 1.32e-2 |
| `IdentityMap`, eps_a <-> eps_b | 8 | **1.319e-2 (row 1 MISSED)** | **8.739e-5 (row 2 HIT)** |
| `h2` two-harmonic stretch, walls at the preimages | 8 / 10 / 12 | 1.5e-3 / 2.7e-4 / 8.6e-5 | 1.3e-2 |
| `h2`, swapped | 12 | 1.319e-2 | 8.76e-5 |
| `sh4`: 4 x 4 TransfiniteMap with the four vertices INTERIOR to the four material blocks moved (non-diagonal J; every material wall stays straight, so the SAME device) | 5 / 6 / 7 | 1.00e-4 / 7.155e-5 / 7.164e-5 | 1.32e-2 |
| unmapped on the same 4 x 4 grid | 5 / 6 / 7 | 9.43e-5 / 7.153e-5 / 7.156e-5 | 1.32e-2 |
| `sh4`, swapped | 6 / 7 | 1.320e-2 | 7.13e-5 / 7.17e-5 |

The identity arm equals the unmapped solve to 3.8e-14 (M = 6) / 1.0e-13
(M = 8) on R, T and Jones.  The sheared arm converges onto the unmapped
same-grid solve order by order: 8.5e-4, 1.9e-4, 4.3e-5, 2.7e-6, **2.8e-7** at
M = 3 .. 7 (`v5_summary.json`); the stretched arm onto the unmapped M = 12
answer to 1.9e-5 at M = 12.  On the patterned gyrotropic grating under the
non-diagonal map, the verifier's engineered defects (M = 6): eps transposed
lands on row 2 (7.13e-5) and misses row 1 by 1.32e-2; sides swapped 1.49e-2;
`eps'_33` without `sg` 4.4e-3; no `1/sg` 9.0e-2 -- every one caught.

---

## 6. D6 / D7 -- magnetic

**Films** (the verifier's mu, against its own eps+mu Berreman, R,T / Jones):

| mu (eps) | map | M = 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|
| gyrotropic `MU_GYRO` (eps 2) | c3, normal | 1.7e-6 / 8.3e-6 | 1.9e-8 | 2.6e-11 | **4.7e-13 / 3.4e-12** |
| same | c3, conical | 2.1e-5 | 3.7e-6 | 3.8e-8 | 2.1e-9 / 1.6e-9 |
| same | sh4, normal / conical | 2.1e-15 / 2.8e-10 | -- / 1.4e-13 | -- / 1.1e-14 | |
| symmetric off-diagonal `MU_SYM` (eps 2) | c3, normal | 1.7e-6 | 2.1e-8 | 3.2e-11 | 4.1e-13 / 3.1e-12 |
| lossy `MU_LOSSY` (eps 2) | c3, normal | 1.1e-5 | 3.0e-8 | 6.3e-10 | 7.1e-13 / 2.4e-13 |
| `MU_GYRO` with eps `GYRO_V` | c3, normal | 1.5e-6 | 2.2e-8 | 8.1e-11 | 5.7e-13 / 3.1e-12 |
| `MU_GYRO` with eps `RASYM` | sh4, normal / conical | 1.4e-14 / 9.9e-10 | 5.8e-15 / 2.4e-13 | | |

and under `h2` to 1.4e-9 .. 4.0e-8 at M = 8.  Every combination reaches the
oracle.

**Duality, the rate (F-D4)**: magnetic disk (eps 1, mu = MU) vs dielectric
twin (eps = MU, mu 1), VACUUM half-spaces, r 0.33, P 1.1, depth 0.45 (the
verifier's own pillar); E_x of one = E_y of the other, order by order
(`v6_summary.json`):

| mu | map | M = 4 | 5 | 6 | 7 | 8 | 9 | no-swap control |
|---|---|---|---|---|---|---|---|---|
| diag(1.9, 1.5, 1.3) | c3 | 7.5e-3 | 6.8e-4 | 5.1e-4 | 8.3e-6 | 1.8e-5 | 5.5e-6 | 0.079 .. 0.084 |
| symmetric off-diagonal | c3 | 3.6e-3 | 3.6e-4 | 2.1e-4 | 3.5e-6 | 6.8e-6 | 2.2e-6 | 0.045 .. 0.048 |
| GYROTROPIC | c3 | 2.7e-3 | 2.5e-4 | 1.4e-4 | 2.5e-6 | 5.0e-6 | 1.8e-6 | 0.0155 .. 0.018 |
| symmetric off-diagonal | c5 | 3.9e-5 | 1.0e-5 | 4.5e-6 | | | | 0.045 |
| GYROTROPIC | c5 | 2.9e-5 | 8.2e-6 | | | | | 0.0155 |

Each twin's own rung change (c3, gyrotropic): magnetic 1.5e-3, 3.2e-4,
1.7e-5, 4.8e-6, 9.2e-7; dielectric 1.4e-3, 7.8e-5, 1.3e-4, 1.5e-6, 2.3e-6 at
M = 4 .. 8 -- the two twins' discretisation errors ALTERNATE in size (odd /
even M), and the duality residual sits at the size of the larger one at
every rung, falling at their rate.  F-D4 confirmed: it is a convergence
identity, not a discrete one, and its non-monotone ladder is the
alternation.  The gyrotropic duality also decides the chi index order on a
PATTERNED structure (the dual of a gyrotropic mu is a gyrotropic eps, whose
route Li 2003 and Berreman fix).

**mu inverse BEFORE quadrature** (`v6_mag.py muafter`): a strongly
anisotropic ROTATED mu (`diag(4, 0.5, 1.2)` at 35 deg) in an eps-2 film under
a strong two-harmonic stretch (`h2s`, local range 8.5:1 / 4.9:1) and under
the circle map; arm `after` = chi from the inverse of the cell-AVERAGED
`mu'` (the verifier's own implementation, corner-rule cells averaged with
their own weights):

| map | arm | M = 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|
| h2s | pointwise (shipped) | 4.8e-4 / 2.3e-3 | 2.2e-5 | 7.9e-7 | 1.6e-8 | **3.3e-10 / 1.5e-9** |
| h2s | after quadrature | **0.191 / 0.583** | 0.191 | 0.191 | 0.191 | 0.191 |
| c3 | pointwise | 7.8e-7 / 3.7e-6 | 3.4e-9 | **2.2e-11 / 1.0e-10** | | |
| c3 | after quadrature | **0.082 / 0.218** | 0.082 | 0.082 | | |

Flat in M: a different operator, not a discretisation error -- the builder's
claim confirmed with a mu the verifier chose to make `mu'` vary strongly.

**The merged stack** (`v6_mag.py stack`): layer 1 an axis-aligned ELLIPSE
(0.30 x 0.24) of `GYRO_V`, layer 2 a coaxial CIRCLE (r 0.18) of eps 2.2 with
`mu = MU_SYM` (off-diagonal) in a background of eps 1.3, layer 3 a uniform
film eps 1.8 with `mu = MU_GYRO`; conical (0.2, 0.5 rad):

| arm | M = 3 | 4 | 5 |
|---|---|---|---|
| lossless: closure | 4.0e-3 | 1.1e-4 | **1.3e-5** |
| lossless: per-layer absorption (all three) | 3.0e-15 | 9.8e-15 | 1.6e-14 |
| lossy eps (`GYRO_V + 0.25i I`): the two LOSSLESS layers absorb | 2.0e-15 | 7.3e-15 | 8.9e-15 |
| same: sum of layer absorptions vs 1 - R - T | 3.5e-3 | 1.1e-4 | **9.7e-6** |
| lossy mu (`MU_LOSSY` in layer 2): lossless layers | 3.0e-15 | 5.8e-15 | 8.2e-15 |
| same: sum vs balance | 3.1e-3 | 1.7e-4 | **1.5e-5** |

---

## 8. D8 -- reciprocity, and is the gyrotropic 1.6e-2 physics?

`v8_recip.py`: the builder's pillar and channel ((25 deg, 40 deg) ->
reflected (-1, 0)), the builder's GYRO tensor and the LC30 control.  Metric:
largest difference of the singular values of the power-normalized 2 x 2
Jones blocks of the channel and its reversal.

| disk | M | fwd vs rev, SAME medium | fwd GYRO vs rev GYRO^T (Onsager, magnetization reversed) | wrong pairing |
|---|---|---|---|---|
| LC30, c3 | 5 / 6 / 7 / 8 | 1.2e-4 / 1.0e-5 / 4.2e-6 / **1.2e-7** | -- | 0.13 |
| LC30, c5 | 4 / 5 / 6 | 1.7e-6 / 5.6e-8 / **1.0e-9** | -- | 0.13 |
| GYRO, c3 | 5 / 6 / 7 / 8 | 2.07e-2 / 1.62e-2 / 1.57e-2 / **1.556e-2** | 1.5e-4 / 8.9e-6 / 1.7e-6 / **6.1e-8** | 0.13 .. 0.14 |
| GYRO, c5 | 4 / 5 / 6 | 1.558e-2 / 1.554e-2 / **1.554e-2** | 7.9e-7 / 5.4e-9 / **3.3e-9** | 0.13 |
| GYRO, staircase k = 1 (square of side 2r, no map) | 4 / 6 / 8 | 7.79e-3 / 7.60e-3 / **7.60e-3** | 1.2e-5 / 5.0e-9 / **4.9e-13** | 0.134 |
| GYRO, staircase k = 2 (8 steps) | 5 / 6 | 2.52e-2 / **2.52e-2** | 6.6e-10 / **8.7e-12** | 0.14 |
| GYRO, staircase k = 4 (16 steps) | 3 / 4 | 1.24e-2 / **1.26e-2** | 5.0e-6 / **1.5e-7** | 0.13 |

**Decided: physics.**  (1) The Onsager-Casimir identity with the
magnetization reversed (`S(eps)^T = S(eps^T)`) holds spectrally to 3e-9
(curved) and 5e-13 (staircases), while the same-medium comparison converges
to a FIXED non-zero value, 1.555e-2 (c3 and c5 agree to 2e-5) -- the
builder's 1.6e-2 (c3 M = 6) is that value read one rung early.  (2) Its
sign: the singular-value differences (fwd - rev) are (-1.296e-2, -1.554e-2)
on both topologies at every converged rung.  (3) The staircase limit: each
staircase is a different DEVICE with its own converged non-reciprocity
(k = 1: (-6.1e-3, +7.6e-3); k = 2: (-8.0e-3, -2.52e-2)); the quantity is
strongly outline-sensitive (a square vs an octagon-like staircase differ by
a factor 3 and in sign of one singular value), so the staircase sequence
is not monotone in k: 7.6e-3 (k = 1), 2.52e-2 (k = 2), 1.26e-2 (k = 4, still
moving 2e-4 from M = 3 to 4), and k = 4 is the first staircase with the curved
disk's sign pattern, (-1.22e-2, -1.26e-2) against (-1.30e-2, -1.55e-2) --
consistent with an O(1/k) approach to the curved limit, not by itself a
decisive one.  The curved value is not a discretisation artefact: it is
converged in M on two independent topologies, and the identity that MUST
hold for this medium holds to round-off.

---

## 9. The tests and the mutation matrix

`vd_mutplugin.py` applies one defect for a whole pytest session through the
real code path (`VD_MUT=<kind>`), against the 19 Phase D tests + the Phase B
verifier's 7 (incl. the renamed B6 id and the D-1 test) + the verifier's 4
(`mut_logs/`):

| defect | caught by (named ids) |
|---|---|
| no `1/sg` (adj(J) mistaken for J^-1) | 12 tests incl. `test_d2_tensor_films_under_a_stretch_and_a_shear_match_berreman`, `test_d1_tensor_route_at_eps_times_identity_is_the_scalar_route` |
| eps transposed | `test_d2_gyrotropic_jones_sees_the_transpose_and_rt_does_not`, `test_d2_...`, `test_d3_...`, both `test_d5_...` |
| eps sides swapped | 8 incl. `test_d2_...` (shear arm), `test_d4_tensor_pillar_lands_on_its_converged_answer` |
| `eps'_33` without `sg` | 6 incl. `test_d9_no_sg_e33_is_caught_only_off_normal_incidence` |
| chi sides swapped (all cells, incl. the vacuum chi of tensor cells) | 9 |
| chi sides swapped, magnetic cells only | `test_d6_magnetic_film_...`, `test_d6_magnetic_pillar_...`, `test_d7_vacuum_painted_...` |
| `chi_33` without `sg` (all cells) | 6 |
| `chi_33` without `sg`, magnetic cells only (the one-token bug) | `test_d6_magnetic_pillar_is_the_dual_of_its_dielectric_twin`, `test_d7_vacuum_painted_with_mu_one_is_vacuum` (a patterned field has `H_z` even at normal incidence) |
| mu ignored | `test_d6_magnetic_film_...`, `test_d6_magnetic_pillar_...`, `test_d7_vacuum_painted_...` |
| `compile_shapes(with_mu=True)` returns `None` for mu | `test_d_shapes_carry_mu_and_the_shapes_route_is_the_explicit_route` |
| `_paint_mu` drops every shape's mu (all routes) | `test_d_shapes_...`, `test_d7_vacuum_painted_...`, `test_d_refusals_...` |
| the three Phase D kernels raise | 17 of 19 Phase D tests (the two that pass are the refusal / unmapped ones -- correct) |
| **mu TRANSPOSED in chi** | **SURVIVED all 26** (every Phase D gate uses a DIAGONAL mu) -> closed by `test_verify_d_gyrotropic_mu_under_a_shear_matches_an_independent_berreman` |
| **D-1 derivative check tolerance x 100** (`1e-6` -> `1e-4`) | **SURVIVED** (the only gate uses a 10 % bug) -> closed by `test_verify_d_curve_derivative_check_refuses_a_small_derivative_bug` (a 3e-5 bug) |
| **the `material_key` exemption's reason comment removed** | **SURVIVED** (the registry holds only tuples) -> closed by `test_verify_d_material_key_walker_exemption_states_its_reason` |
| F-D3: `geom[4]` swapped for `inv(-R)` | (the builder measured both consumers move) -> pinned by `test_verify_d_geom_slot_4_is_the_inverse_plain_gram_for_both_consumers` |

With the verifier's file present, each former survivor fails exactly the
verifier's test (`mut_logs/chi_T.txt`: 1 failed / 29 passed;
`d1_tol100.txt`: 1 failed / 28 passed; `walker_reason.txt`: 1 failed).

---

## 10. Docs

* BUILD_D numbers vs JSON: ~40 numbers spot-checked across D1 (122 / 181),
  D2 table and arms (6.94e-14 / 1.07e-13, 0.1496), D3 (7.2e-13, the plain
  row), D4 (c3 ladder, closures), D5 (8.739349639e-5, 1.40e-13), D6 (film,
  duality ladders, controls), D7 (3.0e-4), D8 (reciprocity rows), D9 (the
  whole matrix), D10 (3.3 / 4.1 / 1.5 s): all agree with the builder's JSON.
* The cookbook example "A liquid-crystal-filled circular hole" EXECUTES as
  written (`v10_cookbook_example.py`, extracted verbatim: closure 2.3e-6,
  `|J[0, 1]|` 9.0e-3, 88 s loaded); so does the CHANGELOG example
  (`v10_changelog_example.py`: closure 8.3e-7, 148 s).
* The CHANGELOG Phase D paragraph is plain language (physical meaning first,
  then gates, limits); see D-1 and D-3 below.
* The `material_key` exemption's reason is honest: `material_key` is used
  only to key the viewers' `material_names` / `material_colors` of
  `PMM2DStackPure.plot_geometry` / `plot_section`; it IS reachable from
  `lumenairy.elements.pmm` (the subpackage `import *`), only not at top level
  -- the walker's question.  See D-2 for its version token.
* mypy (configured list): `Success: no issues found in 33 source files`.
  WSL ruff 0.15.16 on `lumenairy/ tests/ scripts/` + `verify_d/`: clean (8
  import-order fixes in the verifier's own probes applied; the two verbatim
  doc extracts carry `# ruff: noqa: I001`).  `record_history_fingerprints.py
  --check`: every history document matches.

### Defects (all P3 wording; exact edits)

**D-1** (F-D1, BUILD_D s 2.9 bullet "transpose", CHANGELOG) -- "a transposed
gyrotropic tensor leaves a uniform film's R / T unchanged" holds at NORMAL
incidence only; at conical incidence the transposed gyrotropic film moves R /
T by 9.9e-4 against 6.9e-10 (`v2_summary.json`, `sh4_gyro_none_0.35_0.7`).
Edits: BUILD_D 3.1 "and a transposed gyrotropic tensor leaves a uniform
film's R / T unchanged" -> "and a transposed gyrotropic tensor leaves a
uniform film's R / T unchanged at NORMAL incidence (at conical incidence it
is R / T-visible, 9.9e-4: VERIFY_D s 2)"; s 2.9 "every R / T of a symmetric
tensor, and the gyrotropic FILM's R / T, are transpose-blind" -> "... and
the gyrotropic film's R / T AT NORMAL INCIDENCE ..."; CHANGELOG "(a
transposed gyrotropic tensor and a mirrored director change only the Jones
matrix; ..." -> "(at normal incidence a transposed gyrotropic tensor and a
mirrored director change only the Jones matrix; ...".

**D-2** (`tests/unit/test_v4_16_0_walker_all_symmetry.py` lines 276 and 302)
-- the comments open with "v5.50", a FORWARD version token (the last tag is
`v5.49.0`; `4ec402bc` is in no tag).  Edit: "v5.50 (curved-cell Phase C,
2026-10-02)" -> "unreleased after 5.49.0 (curved-cell Phase C, 2026-10-02)"
and "v5.50 (exported by the maintainer's commit 4ec402bc, ..." -> "unreleased
after 5.49.0 (exported by the maintainer's commit 4ec402bc, ...".  (Line 276
is Phase C's; the Phase C verifier should see the same.)

**D-3** (CHANGELOG Phase D gates bullet) -- "a uniform liquid-crystal film
and a gyrotropic film under a stretch, a sheared map and the circle map
match ... to ~1e-13 by `M = 8`": under the circle map at CONICAL incidence
the films are at 8.8e-11 / 9.9e-11 at `M = 8` (BUILD_D s 2.3).  Edit: "...
to ~1e-13 by `M = 8` at normal incidence (~1e-10 at conical incidence under
the circle map)".

**D-4** (BUILD_D s 2.4 and the CHANGELOG) -- "the two INDEPENDENT topologies
agree to 2.5e-6 ... The curved answer is converged": the c3 ladder still
moves 3.5e-6 / 2.8e-6 per rung at M = 11 / 12 and c3 M = 12 is 3.8e-6 from
c5 M = 8.  Edit BUILD_D s 2.4 first bullet: "agree to 2.5e-6 (c3 M = 10 vs
c5 M = 8)" -> "agree to 1e-6 .. 4e-6 (c3 M = 10 .. 12 vs c5 M = 8; VERIFY_D
s 4), inside the c3 ladder's own last rung changes (3.5e-6, 2.8e-6): the
answer is fixed to ~4e-6"; the Richardson line "(8.9e-4 .. 5.6e-5)" -> "(a
1e-4-class corroboration: pairs wander 5e-5 .. 3e-4 to N = 33, a 1/N
least-squares fit 7.7e-5 -- VERIFY_D s 4)".

### Notes (no edit required)

* **N-1** -- the D-1 test in `test_verify_pmm2d_curved_b.py` asserts a bare
  `ValueError`; the verifier's new test matches `"derivative"`, so the right
  refusal is what passes.
* **N-2** -- see D-4: the c3 ladder's rate slows past M = 10 on the tensor
  disk (decade/rung -> 1.3x/rung); Phase B's scalar disk shows the same 1e-6
  level at M = 11 / 12; Phase E's convergence claims should cite both.
* **N-3** -- a director exactly along x or y in a uniform film under a
  separable stretch at normal incidence is exact at every M (s 3); a gate
  built on such a director would be blind to the stretch.  The builder's
  gates use 0.55 rad / 30 deg -- fine.

---

## 11. F-D3 -- `geom[4]`

`_homog_geom_cache` returns `(W0, g2_geo, G W0, Stt W0, Ginv, qq)`.
`geom[4]` (`Ginv`) is read by `_homog_region_modes` (the half-spaces' Eq.-25
H partner, `Dual = Ginv (eps G W0 + Stt W0)`) and by
`_stag_incident_coeffs_mapped` (`C = W0^-1 Ginv b`, the L2 modal
decomposition of the incident wave).  BOTH consumers need the same
mathematical object -- the inverse of the PLAIN (u, v) L2 block Gram: the H
partner because Eq. 25 is tested with the plain Gram (Phase A's "one trap":
under a map `-R` carries `chi_t` and is a different operator), and the
incident decomposition because `b` is an L2 load.  Without a map the two
coincide (`-R` IS the plain Gram).  So the slot is not a latent MISMATCH;
it is a COUPLING: an edit that "fixes" one consumer by changing the slot
(e.g. to `inv(-R)`, which looks right for the pencil) silently changes the
other -- exactly what the builder's first `hgram_R` arm did (Li grating
1.1e-2, pillar 5.3e-2).  Both consumers are gated (C9 and every D gate),
so such an edit would be caught, but by a downstream symptom, not by name.

Decision test (added): `test_verify_d_geom_slot_4_is_the_inverse_plain_gram_
for_both_consumers` pins the IDENTITY of the slot on the 3 x 3 circle map
(`geom[4] @ blockdiag(G1, G2) = I` to <= 1e-10, measured 4.7e-14;
`geom[4]` differs from `inv(-R)` by 0.40 relative, bar >= 1e-2; tuple length
6, `geom[5] = q^2`) and, with no map, `geom[4] = inv(-R)` (<= 1e-10).
Recommendation for Phase E (no change now): return a `NamedTuple` with a
field `inv_plain_gram` and read it by name in both consumers.

---

## 12. Test tails

* Windows, the verifier's decision tests alone:
  `4 passed, 1 warning in 19.85s` (the ResourceWarning since fixed; rerun: `4 passed in 17.30s`; durations spliced into `.test_durations`).
* Windows sweep (62 files: every `*pmm2d*`, `*stack2d*`, `*stagger*`,
  `*curved*`, `*magnetic*`, `*anisotropic*` file incl. Phases A-D and the
  three verifier files, + census, public API, `__all__` walker, changelog
  walkers, doc identifiers, dispatcher doc consistency, except budget,
  history lint / relocation / fingerprint tool, kernel consistency,
  re-exports), `-n 6`, saturated box, in two halves:
  half A `1294 passed, 1 skipped, 29 warnings in 1089.19s`; half B
  `306 passed, 7 skipped, 75 warnings in 984.05s`; together **1600 passed, 8 skipped**, no failure.  The skip is the pre-existing mortar round-2 LAPACK premise.
* Windows, the UNMAPPED sweep (55 files, no `*curved*`) with the three
  Phase D kernels RAISING (`VD_MUT=raise_kernel`): `1512 passed, 8 skipped, 104 warnings in 1498.71s` -- the brief's mutant: the unmapped suite stays green.
* WSL (CPython 3.12.3, numpy 2.4.6): Phases A + B + C + D, the three
  verifier files and the `__all__` walker: `94 passed in 541.82s`; the
  verifier's probes on WSL (`wsl/`): byte identity 176 / 176; `sh4` gyro
  normal 2.3e-14 / 9.5e-15, conical M = 5 1.9e-13, RASYM conical 1.9e-13,
  gyro-mu normal 9.4e-15 / conical 1.4e-13, c3 lossy M = 7 1.1e-12, `h2`
  LC30 M = 8 3.6e-9; eps-transpose Jones 0.168, chi-transpose 0.141; Li
  identity 8.739e-5, swap row 2 8.739e-5 / row 1 1.319e-2, `sh4` M = 6
  7.155e-5 -- the Windows readings to the digits shown.
* Final state (after the report and the durations splice): doc identifiers,
  dispatcher doc consistency, history lint / relocation, `__all__` walker,
  except budget, public API and the verifier's tests: `793 passed in 56.05s`.
* mypy: `Success: no issues found in 33 source files`; WSL ruff: clean;
  fingerprints `--check`: OK.

---

## 13. What Phase E must carry

* The out-of-plane tensor / mu under a map (refused today, naming Phase E):
  its congruence `det L L^-1 eps L^-T` with `L = blockdiag(J, 1)` leaves
  `e13' = adj(J) e_t3 / ...` style mixed terms -- s 2's derivation extends
  directly; gate it with the verifier's sheared map, a GYROTROPIC tensor
  whose gyration axis is TILTED, oblique incidence, Jones.
* Gate every magnetic path with a NON-SYMMETRIC mu (the chi_T survivor
  class) and at least one conical uniform film (chi_33).
* The `geom` tuple as a `NamedTuple` (F-D3).
* Convergence statements on curved disks past M = 10 must cite the slowed
  rate (N-2).

## 14. What the verifier could not measure

* An independent full-wave oracle for the TENSOR pillar (no tensor FEM
  here; the references are the two topologies, the exact-form-factor RCWA
  and the Onsager / staircase checks).
* c5 `M = 8` re-run (dof 2450, > 30 min on the loaded box): the builder's
  JSON was used for the cross-topology comparison; c5 `M = 7` was re-run.
* The 16-step staircase gyro non-reciprocity beyond M = 4 (dof 3200 at M = 5), and a 32-step staircase: the staircase-limit argument of s 8 rests on k = 1, 2, 4.
* Idle wall times.

## 15. Reproduction

```
cd /c/tmp/lum_vcurved_d/validation/probe_pmm2d_curved/verify_d
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=C:/tmp/lum_vcurved_d
python v0_oracle.py
(cd /c/tmp/vcd_pre_607d && PYTHONPATH=C:/tmp/vcd_pre_607d python .../verify_d/v1_bytes.py C:/tmp/vcd_pre_607d pre)
python v1_bytes.py C:/tmp/lum_vcurved_d post ; python v1_bytes.py C:/tmp/lum_vcurved_d posttrap trap
sh runjobs.sh jobs_v2mut.txt 4 ; sh runjobs.sh jobs_v2.txt 6 ; python v2_summary.py
sh runjobs.sh jobs_v4.txt 2 ; python v4_pillar.py curved c3 10|11|12 ; python v4_pillar.py curved c5 7 ; python v4_pillar.py summary
sh runjobs.sh jobs_v5.txt 3          # Li 2003
sh runjobs.sh jobs_v6.txt 3 ; python v6_summary.py
sh runjobs.sh jobs_v8.txt 3 ; python v8_recip.py summary
sh run_mutmatrix.sh <kinds>          # mut_logs/
wsl -e bash run_wsl.sh               # wsl/
```
