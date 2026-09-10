# VERIFY -- MAGNETIC (permeability-tensor) anisotropy for the PURE staggered 2-D PMM

Independent adversarial verification of
`BUILD_PMM2D_STAGGERED_MAGNETIC_2026_09_10.md` (branch
`feat/pmm2d-staggered-magnetic`, 8 commits incl. `10ad874`), merged into
`wave2/pmm2d` at `5c384e5` together with an unrelated Wood-anomaly-list fix and
repaired at `6454f68`.

Worktree `C:/tmp/lum_vmag`, branch `verify/magnetic`, started from `5c384e5`,
merged (fast-forward) to `6454f68` mid-verification -- see Section 0.1.
Reference arm: the READ-ONLY main clone
`D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy` at
`fb3fd93` (= 5.43.0 + one backend commit).

Machine: Windows 11, py3.14.6, numpy 2.4.4,
`OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = MKL_NUM_THREADS = 1`.
Probes: `validation/probe_verify_magnetic/` (README maps probe -> table).
NOTHING here was read from the build's own probes: every oracle, every
transform rule and every fixture in this report was derived and written from
scratch, and each probe's module docstring carries the derivation so the
algebra can be checked without running anything.

**Bottom line.** Every claim in the build doc that this verification could
reach is CONFIRMED, most of them by an oracle the build did not use.  Two
findings: one clause of the build doc's Appendix-Eq.-43 argument is REFUTED
(the DECISION it supports is right; the stated reason is not -- Section 2.4),
and four fail-before docstrings cite numbers that are not their own assertion's
quantity (Section 5.3).  Neither endangers a bar or a line of library code.
The named follow-up is implemented, tested two-sided, and shown byte-identical
on every nonmagnetic call site (Section 6).

---

## 0. What was verified, and how

| task | method | verdict |
|---|---|---|
| 1. bit-identity of the three NONMAGNETIC paths vs `fb3fd93` | 12 own fixtures, 75 hashed quantities, raw-byte SHA-256, two interpreters | **CONFIRMED** |
| 2. the derivation (`R`, `S_tt`, `K_tz`, the R-vs-Gram trap) | re-derived from the paper's Eqs. 13-14 here; operators re-assembled from the 1-D primitives; placement knockouts | **CONFIRMED** (one clause of the Eq.-43 argument REFUTED) |
| 3. gates G1-G7 + composition | own analytic oracle, own duality transform, own 1-D bridge, own guard sweep | **CONFIRMED** |
| 4. durability of the test file | all 42 bars re-measured by driving the file's own helpers | **CONFIRMED**, 9 sub-decade bars restated, 4 mis-cited numbers found |
| 5. follow-up: `eps*mu` Wood cut-offs | implemented, hashed both sides, two-sided test + fail-before | **DONE** |
| 6. regression suites, ruff, durations | Section 7 | **GREEN** |

### 0.1 The merge defect found before this verification started

The integration tip `5c384e5` merged cleanly hunk-wise but not semantically:
`PMM2DStackPure.solve`'s Wood-list collection read `L["eps_cell"]` for every
non-uniform layer while the magnetic record carries `"eps"` -- a `KeyError` on
every magnetic stack.  It was repaired on `wave2/pmm2d` at `6454f68`
(permittivity-only) before this verification's measurements were taken;
every number below is on `6454f68` or later.  All V1 stack fixtures were
re-run after the merge and are unchanged.

---

## 1. TASK 1 -- BIT-IDENTITY of the nonmagnetic paths

`validation/probe_verify_magnetic/v1_bitidentity.py`.  Each fixture is hashed
from the RAW BYTES of a C-contiguous array (shape + dtype + `tobytes`), so the
comparison is EXACT and not a tolerance.  One arm per interpreter, each
asserting `lumenairy.__file__`.

| fixture | what is hashed |
|---|---|
| F1 scalar `(2,2)` M=6 oblique | `Lmat`, `Rmat`, `Stt`, `Schur`, `Et_blocks`, `Et_offdiag`, `Agen`, `Bgen`, `dimtot`, `magnetic`, `Ggram_blocks` |
| F2 in-plane tensor `(2,2)` M=6 oblique | same |
| F3 OUT-OF-PLANE tensor `(2,2)` M=5 oblique | same (the `4 q^2` generator's `Agen`/`Bgen`) |
| F4 `pmm_jones_2d_staggered` scalar, normal, M=6 | orders, R, T, Jones, sum R, sum T |
| F5 `pmm_jones_2d_staggered` scalar `(3,3)`, theta 0.30, M=5 | same |
| F6 `pmm_jones_2d_staggered` in-plane tensor, conical 0.30/0.70, M=6 | same |
| F7 `pmm_jones_2d_staggered` out-of-plane, conical 0.20/0.40, M=5 | same |
| F8 `pmm_efficiency_2d_staggered` TE and TM, oblique 0.30/0.25, M=6 | orders, R, T |
| F9 3-layer pure stack (uniform scalar \| patterned scalar \| uniform tensor), conical | orders, R, T, Jones |
| F10 GENERALIZED cascade (out-of-plane layer + in-plane tensor layer) | orders, R, T, Jones |
| F11 `retain_internal` + `layer_absorption` on a lossy scalar cell | R, T, Jones, A |

**Result: 12 fixtures, 75 hashed quantities, 0 differences -- BIT-IDENTICAL.**
Re-run three times: (a) at `5c384e5`, (b) after the `6454f68` merge, (c) after
this verification's own library change (Section 6).  Identical every time.

### 1.1 The localisation control -- the comparison CAN see the one change in range

A "0 differences" result is only evidence if the harness would have caught the
one change in `fb3fd93..6454f68` that is allowed to move a nonmagnetic number:
`_wood_eps_reals` now lists a SCALAR layer's permittivities, so an order
sitting EXACTLY on a layer cut-off is nudged in the worktree and was not in
main.  `v1b_localise_wood.py` puts a scalar layer exactly on its own `(2,0)`
cut-off at normal incidence (`eps = (2 wl/px)^2`, gap `0.000e+00`):

| arm | ON the cut-off | detuned 2 % |
|---|---|---|
| main `fb3fd93` | `sum R+T = 1.9999204667771637` | `1.9999283818810372` |
| worktree | `sum R+T = 1.9999204667933330` | `1.9999283818810372` |
| R / T / Jones hashes | **DIFFER** | **IDENTICAL** |

So the harness has the resolving power (the on-cut-off arm moves by 1.6e-11 in
closure and every hash differs), the difference is confined to an exact
coincidence, and it is attributable to the Wood list -- the magnetic branch is
entirely inert with `mu_cell=None`.  **CONFIRMED, both directions.**

---

## 2. TASK 2 -- THE DERIVATION

### 2.1 Re-derived here from the paper's Eqs. 13-14

Granet, JOSA A 40, 652 (2023), `exp(+i w t)`,
`U = U(x1, x2) exp(-i gamma x3)`.  Eq. 13-14 as printed:

```
(13a)  i g C E_t + [d2; -d1] E3 = -i w mu0 [mu_t] H_t
(13b)  [-d2  d1] . E_t          = -i w mu0 mu33 H3
(14a)  i g C H_t + [d2; -d1] H3 =  i w eps0 [eps_t] E_t
(14b)  [-d2  d1] . H_t          =  i w eps0 eps33 E3
```

Eliminating H from 13a/13b,

```
H_t = (i/(w mu0)) [chi_t] ( i g C E_t + [d2;-d1] E3 )
H3  = (i chi33/(w mu0)) [-d2  d1] E_t
```

and substituting into 14a, multiplied through by `-i w mu0`:

```
-g^2 C[chi_t]C E_t + i g C[chi_t][d2;-d1] E3
                   + [d2;-d1] chi33 [-d2  d1] E_t  =  k^2 [eps_t] E_t
```

i.e., using `E~3 = d3 E3 = -i g E3` (Eq. 18) and
`[-d2 d1] = -[d2 -d1]`,

```
-g^2 C[chi_t]C E_t = k^2[eps_t] E_t + K_tz E~3 + S_tt E_t
      K_tz = C[chi_t][d2; -d1]                     (Eq. 21)
      S_tt = [d2; -d1] chi33 [d2  -d1]             (Eq. 20)
```

With `div D = 0` (Eq. 16) giving `eps33 E~3 = -K_zt E_t`,

```
L = k^2 [eps_t] + S_tt - K_tz (eps33)^-1 K_zt      (Eq. 24)   [CONFIRMED]
```

Writing out `C = [[0,1],[-1,0]]` and
`[chi_t] = [[chi11, chi12], [chi21, chi22]]`:

```
C[chi_t] = [[ chi21,  chi22], [-chi11, -chi12]]

R = C[chi_t]C  = [[-chi22,  chi21],
                  [ chi12, -chi11]]                (Eq. 24)   [CONFIRMED]

K_tz = C[chi_t][d2;-d1] = [ -chi22 d1 + chi21 d2 ;
                            -chi11 d2 + chi12 d1 ] (Eq. 21)   [CONFIRMED]
```

Isotropic reduction: `chi_t = I` gives `R = -I` and `K_tz = [-d1; -d2]`, i.e.
the shipped `Grad1 = -kron(Mtt_y, dbt_x)`, `Grad2 = -kron(dbt_y, Mtt_x)`.

`S_tt` in weak form: `<v, S_tt E> = -<curl_z v, chi33 curl_z E>` (integrate the
column `[d2;-d1]` by parts; the two sign flips of `[d2 -d1] E = -curl_z E`
cancel).  This is exactly the shipped `-Curl^H (.) Curl` with the middle
operator carrying `chi33`.

**Eq. 25 carries NO `chi_t`.**  From 14a with H3 substituted and multiplied by
`w mu0`:

```
g (w mu0) C H_t = k^2[eps_t] E_t + [d2;-d1] chi33 [d2 -d1] E_t
                = (k^2 [eps_t] + S_tt) E_t         (Eq. 25)   [CONFIRMED]
```

so the Eq.-25 H recovery, tested in V1/V2, is `g` times the PLAIN block Gram
applied to the C-rotated H coefficients.  The build's "THE TRAP" analysis is
therefore correct as a matter of derivation, not only of measurement.

### 2.2 The shipped operators vs an INDEPENDENT assembly

`v2_derivation.py` re-assembles `R`, `S_tt`, `K_tz`, `M_eps33`, `K_zt` and
hence `L` for a UNIFORM cell from the module's 1-D primitives only
(`mass` / `stiff` / `mixed` / `c_ref`), using the placements above.  `S_tt` in
particular is built the APPENDIX Eq.-42 way -- STIFFNESS matrices and
derivative-on-the-LEFT matrices -- a route the shipped code never takes (it
goes through `Gw^-1 Gw_chi Gw^-1`), so agreement also confirms the de Rham
exactness that form assumes.  `(2,2)` grid, M=5, oblique Bloch phases,
`eps = 4`.  Relative `max|diff|`:

| mu | `R` | `S_tt` | `Schur` (carries `K_tz`) | `L` | `Ggram_blocks` |
|---|---|---|---|---|---|
| `I` (identity, forced) | 8.657e-17 | 8.023e-15 | 1.452e-15 | 2.158e-15 | **0.000e+00** |
| `diag(1.30, 1.75, 1.45)` | 1.125e-16 | 7.109e-15 | 1.183e-15 | 1.171e-15 | 0.000e+00 |
| general, `m12 != m21` | 2.283e-16 | 7.109e-15 | 1.218e-15 | 1.170e-15 | 0.000e+00 |
| gyrotropic `m12 = -m21 = 0.4i` | 9.235e-17 | 1.056e-14 | 1.389e-15 | 2.534e-15 | 0.000e+00 |
| lossy `Im(m11) = 0.25` | 3.490e-16 | 7.109e-15 | 1.196e-15 | 1.365e-15 | 0.000e+00 |

`Ggram_blocks` is EXACTLY (0.0) the plain block Gram `blockdiag(G1, G2)`.

**Is the comparison discriminating?**  The same measurement with the placement
deliberately wrong (same fixture, general mu):

| arm | `R` | `Schur` |
|---|---|---|
| as derived | 2.283e-16 | 1.218e-15 |
| `chi11 <-> chi22` on the diagonal | 2.571e-01 | 2.614e-01 |
| `chi12 <-> chi21` (mixed blocks) | 2.114e-01 | 2.114e-01 |
| no C rotation (`R = -[chi_t]`) | 2.571e-01 | 2.614e-01 |

15 decades of separation.  **CONFIRMED: the code's block placement and signs
are the ones the paper's Eqs. 13-14 give.**

(Recorded because it cost a cycle: this probe's FIRST version disagreed with
the shipped `K_tz` at 1.4e-01 whenever `chi` had off-diagonals.  The error was
in MY assembly -- `<V2 | d1 V3>` needs the y-factor `<B|Btilde>`, and I had
written `<Btilde|B>`.  The shipped code was right.  An independent
re-derivation catches its own mistakes only because the disagreement is
localised: the diagonal-mu arms passed while only the mixed ones failed.)

### 2.3 The R-vs-Gram trap, by knockout

Collapsing the two roles -- using `-R` as the Eq.-25 Gram -- measured against
MY analytic oracle (Section 3.1), uniform `eps = 4, mu = 2` slab, theta 0.35,
M=8:

| arm | analytic residual |
|---|---|
| SHIPPED (plain Gram) | **2.2974e-14** |
| KNOCKOUT (`-R` used as the Gram) | **1.7182e-01** |
| SHIPPED again, after restoring | 2.2974e-14 |

13 decades.  The build doc's M1b reports 1.9e-14 / 2.1e-01 for the same
knockout on its own oracle (which maxes over R/T only -- see Section 5.3); the
decision is identical.  **CONFIRMED.**

### 2.4 The Appendix Eq.-43 question -- I AGREE WITH THE DECISION, NOT THE STATED REASON

The build doc says: *"its Eq. 43 pairs the chi components with the derivatives
differently from its own Eq. 21, and its rendering of the second term in
`K_tz22` repeats `d2` where `d1` belongs."*

The appendix's exact text (extracted from the PDF, not read off the rendered
page; `'` = derivative, `B_{i n} = B_n(x^i)`, so subscript 1 = `d1`,
subscript 2 = `d2`):

```
Ktz11 = <B~*_2m B_1n, chi22 B~_2p B~'_1q>  -  <B~*_2m B_1n, chi21 B~'_2p B~_1q>
Ktz22 = <B_2m B~*_1n, chi11 B~'_2p B~'_1q> -  <B_2m B~*_1n, chi12 B~'_2p B~_1q>
```

* **Row 1 pairs `chi22` with `B~'_1q` = `d1`, and `chi21` with `B~'_2p` = `d2`.
  That is EXACTLY Eq. 21's pairing.**  What differs is the OVERALL SIGN:
  Eq. 43 row 1 reads `+chi22 d1 - chi21 d2 = -(Eq. 21 row 1)`.
  **-> the "different pairing" clause is REFUTED.**
* Row 2, second term: `chi12` carries `B~'_2p` = `d2`, where Eq. 21 pairs
  `chi12` with `d1`.  **-> that clause is CONFIRMED.**
* Row 2, FIRST term (not mentioned in the build doc): `chi11 B~'_2p B~'_1q` is
  DOUBLE-primed -- a second-derivative object that cannot be a `K_tz` entry at
  all.  Correcting it to `B~'_2p B~_1q` gives `chi11 d2`, matching Eq. 21.
  **-> a third defect, unreported.**
* With both row-2 typos corrected, Eq. 43 = `-(Eq. 21)` row for row: one
  consistent overall sign flip, not a re-pairing.

The appendix's sign bookkeeping is unreliable more generally, which is the real
shape of the problem: Eq. 39 lists all four `R` entries with `+` coefficients,
whereas `R = C[chi_t]C = [[-chi22, chi21], [chi12, -chi11]]` -- so Eq. 39 is
neither `R` nor `-R` (the diagonal would have to flip and the off-diagonal not).
Eq. 42 likewise gives all four `S_tt` entries a `+`, though its off-diagonals
are negative.  Only Eq. 45 carries the `C` signs.  Read as PLACEMENT tables the
appendix is right; read as signed matrix elements it is not.

**Is the overall sign load-bearing?**  Yes.  `K_tz` enters `L` only through
`K_tz (eps33)^-1 K_zt`, which is bilinear in `K_tz` and `K_zt`, so flipping
`K_tz` alone flips the sign of the whole Schur term.  The code's choice
(Eq. 21's sign) is pinned independently twice: by the isotropic reduction to
the shipped `Grad1`/`Grad2` (gate G1, 4.62e-14) and by the analytic oracle
(gate G2, 4.30e-14 worst); with the sign flipped the Schur term changes by
`2 x` its own magnitude, i.e. O(1).

**Verdict.**  The build's CODE and its DECISION ("follow Eq. 21; the
appendix's which-chi-in-which-row is right") are both CONFIRMED against my own
derivation to 1.06e-14.  One clause of the stated reason is REFUTED: the
appendix's defect is an overall sign plus two typos in row 2, not a different
chi-to-derivative pairing.  Recommend the build doc's sentence be corrected;
nothing in the library changes.

---

## 3. TASK 3 -- GATES G1-G7 RE-MEASURED ON MY OWN FIXTURES

### 3.1 G2 -- the ANALYTIC oracle (`v3_airy.py`)

Written from scratch in the library's public `exp(-i w t)` convention
(`curl E = +i w mu0 mu H`, `curl H = -i w eps0 eps E`), NOT copied:

```
TE:  H~_x = (kz/mu) E_y            ->  Y_TE = kz/mu
TM:  H_y  = (eps/kz) E~_x          ->  Y_TM = eps/kz
kz = sqrt(eps mu - sin^2 th0),  Im(kz) >= 0
[U(t); V(t)] = [[cos d, i sin d/Y], [i Y sin d, cos d]] [U(0); V(0)]
```

(the `+i` form is the z-FORWARD transfer; the `-i` textbook matrix is its
inverse).  Matching `U(0) = 1+r`, `V(0) = Y_sup(1-r)`, `U(t) = tau`,
`V(t) = Y_sub tau` gives `r`, `tau` in closed form.

**The Jones convention is MEASURED, not assumed.**  `slab_rt` solves for the
TANGENTIAL amplitude ratio, so at `phi = 0` the order-0 reflection Jones must
be `diag(r_TM, r_TE)` with NO extra sign.  Asserting the other p^-hat
convention (`J[0,0] = -r_TM`) reads **7.5e-01** on the nonmagnetic slab; the
convention above reads 2.7e-14 / 7.7e-14 with off-diagonals at 3.9e-15.

**The TIME convention is measured too** (`python v3_airy.py convention`), on a
lossy slab where the two branches differ:

| branch | worst residual | dT_te |
|---|---|---|
| `exp(-i w t)` (+i forward transfer) | 1.410e-14 | 7.2e-15 |
| `exp(+i w t)` (the other branch) | 2.664e-01 | **1.117e+00** (T > 1: it amplifies) |

**V3A -- pin on the NONMAGNETIC arm first** (max over {R, T, Jones} x {TE, TM},
M=8, `eps=4`, `mu=1`):

| incidence | `mu_cell` omitted | `mu = 1` forced through the magnetic path |
|---|---|---|
| normal | 3.0879e-14 | 5.1216e-14 |
| theta 0.35 | 7.7393e-14 | 4.4233e-14 |

**V3B -- the MAGNETIC arms** (M=8, same metric).  Six pairs, two of them not in
the build's table (`mu < 1`, and both media lossy):

| eps | mu | theta 0 | theta 0.35 |
|---|---|---|---|
| 4.0 | 2.0 | 2.4013e-14 | 2.2974e-14 |
| 1.0 | 4.0 (purely magnetic contrast) | 4.4521e-14 | 4.8062e-14 |
| 4.0 | 2.0 + 0.3i (LOSSY mu) | 3.4134e-14 | 1.0499e-14 |
| 4.0 + 0.2i | 2.0 (lossy eps in a magnetic host) | 2.7524e-14 | 1.4095e-14 |
| 2.25 | 0.60 (**mu < 1**) | 5.5066e-14 | 8.6391e-14 |
| 9.0 + 1.0i | 1.8 + 0.4i (**both lossy**) | 3.7423e-14 | 1.4827e-14 |

**V3C -- spectral convergence** (`eps=4, mu=2`, theta 0.35):

| M | 5 | 6 | 7 | 8 |
|---|---|---|---|---|
| residual | 2.1433e-07 | 8.7081e-10 | 4.8809e-12 | 2.2974e-14 |

M=5 -> M=8 = **9.3e+06**.  **G2 CONFIRMED**, on a wider fixture set than the
build's, including the two arms it did not measure.

### 3.2 G3 -- DUALITY, with the transform derived here (`v4_duality.py`)

`(E, H, eps, mu) -> (Z0 H, -E/Z0, mu, eps)` is exact for `exp(-i w t)` Maxwell
(substitute; `Z0 eps0 = mu0/Z0 = sqrt(mu0 eps0)`).  I work in the LAB
TANGENTIAL basis the library reports, so no `(s, p)` frame and no p^-hat
convention enters.  For a vacuum plane wave `Z0 H = k^ x E` and
`E_z = -(k^x Ex + k^y Ey)/k^z`, hence

```
A(k^) = (1/k^z) [[ -k^x k^y ,      -(k^y^2 + k^z^2) ],
                 [ k^x^2 + k^z^2 ,  k^x k^y         ]]

T'_m = A(k^_m) T_m A(k^_inc)^-1,
   k^_m = (kx_m, ky_m, -kz_m) on REFLECTION, (kx_m, ky_m, +kz_m) on TRANSMISSION
```

Classical mount (`k^y = 0`): `A = [[0, -k^z], [1/k^z, 0]]`.
Normal incidence: `A_i = D = [[0,-1],[1,0]]`, `A_r = -D`, so
`J' = -D J D^-1 = D J D`, and the two efficiency ROWS swap exactly.

**On the requested `(s, p)` form.**  With `p^ = k^ x s^` for BOTH the incident
and the reflected wave, `Z0 H = E_s p^ - E_p s^`, so the `(s,p)` amplitude map
is `D = [[0,-1],[1,0]]` for both and `J_dual = +D J D^-1`.  The MINUS in the
build's `J_dual = -D J D^-1` is not universal: it appears exactly under the
ASYMMETRIC p^-hat convention the build states (`p^_i = -(k^ x s^)`,
`p^_r = +(k^_r x s^)`), which turns the incident map into `-D` while the
reflected one stays `D`.  Both forms are equivalent to the tangential rule
above; I verified the physics with the tangential rule, which needs no
convention at all.

Metric: `max_m |T'_m - A(k^_m) T_m A(k^_inc)^-1|` over both ports and all
PROPAGATING orders, normalised by the largest `|T_m|`.  Vacuum half-spaces,
`(2,2)` cells, period 0.90 um, wl 0.55 um, depth 0.30 um.

**A -- uniform `(eps, mu)` tensor pair (SPECTRAL):**

| incidence | M=5 | M=6 | M=7 | M=8 | no-rotation control |
|---|---|---|---|---|---|
| normal | 1.611e-14 | 6.367e-14 | 9.520e-14 | 3.265e-13 | 1.653e+00 |
| theta 0.30 | 2.652e-04 | 5.257e-06 | 7.039e-08 | **6.765e-10** | 1.728e+00 |
| conical 0.30/0.70 | 8.504e-08 | 3.232e-10 | 8.170e-13 | **2.839e-13** | 1.697e+00 |

R/T efficiency row-swap at normal incidence: **7.472e-14**.

**B -- patterned electric cell vs its PURELY MAGNETIC dual** (arm A `mu = 1`,
i.e. the shipped anisotropic path; arm B `eps = vacuum`, `mu` = the pattern):

| incidence | M=5 | M=6 | M=7 | M=8 | control |
|---|---|---|---|---|---|
| normal | 2.293e-03 | 1.170e-03 | 1.883e-04 | 3.678e-05 | 6.988e-01 |
| theta 0.30 | 1.560e-02 | 2.559e-04 | 6.280e-05 | 4.382e-05 | 7.481e-01 |
| conical 0.30/0.70 | 1.306e-02 | 4.902e-04 | 1.289e-04 | 4.433e-05 | 7.332e-01 |

R/T row-swap at normal: **6.230e-06**.

**C -- BOTH sides patterned and anisotropic:**

| incidence | M=5 | M=6 | M=7 | M=8 | control |
|---|---|---|---|---|---|
| normal | 1.529e-02 | 5.646e-02 | 8.922e-03 | 2.852e-04 | 3.612e-01 |
| theta 0.30 | 3.972e-03 | 7.408e-03 | 7.535e-02 | 5.212e-04 | 3.697e-01 |
| conical 0.30/0.70 | 1.429e-03 | 2.349e-03 | 1.905e-04 | 4.252e-05 | 3.556e-01 |

Fixture C's ladder is NON-MONOTONE at M=6/M=7 (corner-limited algebraic
convergence with scatter, the same character the build doc's M3C shows); the
CLAIM it supports is the M=8 level plus the M=5 -> M=8 drop, both of which hold.

**Recorded, and NOT in the build doc: the duality residual on EVANESCENT orders
is 2-4 decades worse than on propagating ones** -- fixture C at M=6 reads
1.2e+00 on the worst evanescent order against 5.4e-02 on the worst propagating
one.  This is expected and benign (the staggered discretization is not
self-dual -- `E3` lives in V3 while `H3` lands in Vw -- and evanescent
amplitudes are near-field quantities; they fall with M together with everything
else, to 3.5e-03 at M=8), but it means a duality bar stated over ALL orders
would be measuring the near field.  This report's metric is propagating-only
and says so.

**G3 CONFIRMED.**

### 3.3 G4 -- the 1-D engines, through duality (`v5_1d_bridge.py`)

**ENGINE CENSUS re-run independently.**
`grep -rn --include='*.py' -iE 'permeability|\bmu_|\bmu\b|\bmu='` over
`lumenairy/elements/{rcwa,pmm,berreman.py,eme}` plus a parameter-level sweep
for any function signature taking `mu`:

| engine | takes a permeability? |
|---|---|
| `rcwa_jones_1d`, `rcwa_efficiency_1d`, `RCWAStack` | NO -- `rcwa/_core.py:2257` "Non-magnetic (`mu = 1`)"; the only other `mu` in `rcwa/` is `mu = eig(X @ Y)`, the pencil's own eigenvalue (`_core.py:3278/3378`) |
| `pmm_jones_1d`, `pmm_efficiency_1d`, `PMMStack` | NO -- every `mu` in `pmm/_core.py` and `pmm/_jax_stack.py` is the eps-free GEOMETRIC eigenvalue of `Kx2` (`q^2 = eps - mu`, e.g. `_core.py:1744/2550/4111`) or the slant metric's `mu^lm = sqrt(g) g^lm` (`_core.py:5749`) |
| `berreman_jones_1d`, `BerremanStack` | NO -- `berreman.py:132` states `mu = 1` |
| `eme_2d_vector._build_generator(..., mu_xy)`, `strip_vector_modes(..., mu_x)` | YES, a SCALAR `(Nx, Ny)` / `(Nx,)` permeability -- but these return lateral wavenumbers / propagation constants of a cross-section (a WAVEGUIDE MODE solver), so they cannot produce per-order R/T for a grating |
| `pmm/twod_staggered.py`, `pmm/stack2d_pure.py` | YES -- THIS build |

**Census CONFIRMED.**  So the 1-D check must go through duality.

**Step 0 -- pin the GEOMETRY and the Jones BASIS on the NONMAGNETIC arm**, by
solving the SAME electric stripe through the 2-D staggered engine and both 1-D
engines (period 0.90 um, wl 0.55 um, depth 0.30 um, duty 0.5, vacuum
half-spaces, ridge = rotated-LC tensor, groove = 2.10 I):

| theta | M | vs `pmm_jones_1d` dRT / dJ | vs `rcwa_jones_1d` dRT / dJ | the two oracles vs each other | y-leak |
|---|---|---|---|---|---|
| 0 | 6 | 8.837e-06 / 1.657e-05 | 9.030e-06 / 1.694e-05 | 1.927e-07 / 3.734e-07 | 3.9e-28 |
| 0 | 8 | 2.709e-06 / 5.209e-06 | 2.902e-06 / 5.583e-06 | 1.927e-07 / 3.734e-07 | 3.2e-27 |
| 0.22 | 6 | 7.174e-05 / 5.200e-05 | 7.168e-05 / 5.228e-05 | 1.842e-07 / 3.398e-07 | 4.7e-28 |
| 0.22 | 8 | 3.143e-06 / 5.793e-06 | 3.327e-06 / 6.133e-06 | 1.842e-07 / 3.398e-07 | 8.2e-27 |

**Step 1 -- the MAGNETIC stripe** (`eps_cell = vacuum`, `mu_cell` = the whole
stripe), compared with the duality row swap and, on the Jones, the classical-
mount rule derived in Section 3.2 (`J' = [[-J22, kz^2 J21], [J12/kz^2, -J11]]`):

| theta | M | vs `pmm_jones_1d` dRT / dJ | vs `rcwa_jones_1d` dRT / dJ | y-forbidden leak | unswapped control |
|---|---|---|---|---|---|
| 0 | 5 | 9.844e-05 / 3.717e-05 | 9.826e-05 / 3.686e-05 | 7.4e-29 | 5.908e-02 |
| 0 | 6 | 5.699e-05 / 6.217e-05 | 5.681e-05 / 6.180e-05 | 1.9e-27 | 5.904e-02 |
| 0 | 7 | 8.658e-06 / 1.666e-05 | 8.466e-06 / 1.629e-05 | 1.9e-27 | 5.899e-02 |
| 0 | 8 | **8.688e-06 / 1.668e-05** | **8.495e-06 / 1.631e-05** | 4.6e-26 | 5.899e-02 |
| 0.22 | 5 | 1.522e-03 / 6.272e-04 | 1.522e-03 / 6.274e-04 | 2.0e-28 | 4.913e-02 |
| 0.22 | 6 | 8.121e-05 / 1.637e-05 | 8.116e-05 / 1.618e-05 | 3.0e-27 | 4.909e-02 |
| 0.22 | 7 | 1.024e-05 / 1.839e-05 | 1.005e-05 / 1.805e-05 | 2.4e-27 | 4.908e-02 |
| 0.22 | 8 | **7.036e-06 / 1.294e-05** | **6.852e-06 / 1.260e-05** | 4.8e-26 | 4.908e-02 |

Ladder M=5 -> M=8: 11x at normal, 216x at theta 0.22.  The Jones rule's OVERALL
SIGN is load-bearing and was found by measurement: dropping it leaves the
residual at `2|J| = 6.7e-01` at every M.  **G4 CONFIRMED**, against both
oracles, with the geometry and basis pinned first.

### 3.4 G5 -- CLOSURE, two-sided (`v6_closure_symmetry.py`)

**A -- a HERMITIAN mu absorbs nothing** (`|sum R + sum T - 1|`; gyrotropic
`m12 = -m21 = 0.4i` is Hermitian, hence lossless despite being complex):

| cell | M=6 normal | M=6 conical | M=8 normal | M=8 conical |
|---|---|---|---|---|
| uniform LC eps, uniform gyrotropic mu | 4.630e-14 | 1.080e-10 | 4.219e-14 | 1.394e-13 |
| patterned LC/iso eps, gyro/1.2 mu | 1.807e-06 | 1.423e-05 | 7.858e-09 | 5.592e-09 |
| VACUUM eps, patterned mu | 2.146e-06 | 1.026e-05 | 4.653e-09 | 2.260e-09 |
| patterned eps, real scalar `mu` GRID with `mu < 1` | 1.328e-05 | 2.505e-05 | 1.364e-07 | 4.649e-07 |

(the fourth row is mine, not in the build doc: a `(Nx, Ny)` SCALAR `mu_cell`
with values straddling 1, which exercises the scalar `_chi_maps` branch.)

**B -- a LOSSY mu must close BELOW one** (`Im(m11) = +0.25`, theta 0.25 /
phi 0.60):

| M | sum R+T (Ex / Ey) |
|---|---|
| 6 | 0.973814 / 0.794945 |
| 8 | 0.968860 / 0.794075 |
| control, `Im(m11)` removed | **1.000000 / 1.000000** |

**C -- the tripwire, both sides** (`_STAG_CLOSURE_TOL = 5e-2`):

| arm | closure defect | warnings raised |
|---|---|---|
| Hermitian mu, M=8 (resolved) | 6.442e-08 | **0** |
| Hermitian mu, M=3 (the basis minimum) | 1.251e-01 | **1** |
| LOSSY mu, M=8 (no unity claim) | -- | **0** |

**G5 CONFIRMED.**  My M=3 arm sits at 2.5x the shipped window and fires ONE
warning (only one incident polarization crosses it); the build's own fixture,
re-measured here through the test file's own helper, reads **8.4879e-01 = 17.0x
the window with 2 warnings** at M=3 and **1.4798e-06 with 0 warnings** at M=8.
The two-sided claim holds on both fixtures, and the build's is the one further
from the knife edge.

### 3.5 G6 -- the x<->y transpose with the mu blocks swapped

Transform derived here: transposing about `x = y` requires swapping the grid
axes, the PERIODS, the tensor components (`e11<->e22`, `e12<->e21`, and the
same on mu) AND the azimuth (`phi -> pi/2 - phi`); orders map `(m,n) -> (n,m)`
and the tangential Jones conjugates by `P = [[0,1],[1,0]]`, `J_T = P J P`.
LC/iso eps + gyrotropic/1.2 mu, theta 0.25 / phi 0.60:

| arm | M=5 dRT / dJ | M=6 dRT / dJ |
|---|---|---|
| correct placement | 8.660e-15 / 1.327e-14 | 1.421e-14 / 2.001e-14 |
| `m12`/`m21` SWAPPED in one arm | **2.815e-02 / 1.638e-01** | **1.079e-02 / 1.650e-01** |
| `e12`/`e21` swapped (control) | 1.543e-14 / 1.274e-14 | 1.699e-14 / 1.957e-14 |

The third row is the point: a rotated-LC permittivity has `e12 = e21`, so
swapping THOSE is a no-op -- the permeability needs its own GYROTROPIC
discriminator, and it has one.  12 decades between the correct and the broken
arm.  **G6 CONFIRMED.**

### 3.6 G7 -- the guards, both sides

All 17 conditions raise, with the stated type and a message naming the
limitation and the alternative:

| condition | type |
|---|---|
| out-of-plane mu (`m13`/`m23`/`m31`/`m32` over the relative 1e-12 floor) | `NotImplementedError` |
| mu with an out-of-plane eps (entry) | `NotImplementedError` |
| `mu_superstrate != 1` (entry) | `NotImplementedError` |
| `mu_substrate != 1` (entry) | `NotImplementedError` |
| `PMM2DStackPure(mu_superstrate=...)` | `NotImplementedError` |
| `add_layer(eps_cell=out-of-plane, mu=...)` | `NotImplementedError` |
| `add_layer(eps=out-of-plane (3,3), mu=...)` | `NotImplementedError` |
| singular `[mu_t]` (`det = 0`) | `ValueError` |
| `m33 = 0` | `ValueError` |
| `mu_cell` shape not `(Nx,Ny)` / `(Nx,Ny,3,3)` | `ValueError` |
| `mu_cell` grid != `eps_cell` grid | `ValueError` |
| non-square `mu_cell` | `ValueError` |
| zero scalar `mu_cell` | `ValueError` |
| both `mu` and `mu_cell` | `ValueError` |
| uniform `mu` of a bad shape | `ValueError` |
| uniform scalar `mu = 0` | `ValueError` |
| `_homog_geom_cache` on a magnetic solver | `ValueError` |

**The negative side of the floor:** `mu_superstrate=1.0` and `mu_substrate=1`
are ACCEPTED (no raise), and a `1e-16` stray in `m13` AND `m32` passes the
block-form gate with `max|dR| = 0.0e+00` -- R, T and the Jones all
`np.array_equal` to the exact-identity mu.  **G7 CONFIRMED.**

### 3.7 Composition

**`layer_absorption` on a LOSSY MAGNETIC layer** (magnetic lossy layer over an
isotropic `eps = 2.25` layer, theta 0.20 / phi 0.40; the two sides come from
different machinery -- the internal block-Gram flux quadrature vs the Rayleigh
far field):

| M | sum A (Ex / Ey) | 1 - R - T (Ex / Ey) | closure |
|---|---|---|---|
| 5 | 0.031923 / 0.192533 | 0.032644 / 0.192551 | 7.206e-04 |
| 6 | 0.028425 / 0.194012 | 0.028454 / 0.194012 | 2.952e-05 |
| 7 | 0.022558 / 0.193979 | 0.022559 / 0.193979 | 8.896e-07 |
| 8 | 0.022294 / 0.194011 | 0.022294 / 0.194011 | **3.605e-08** |

Drop M=5 -> M=8: 2.0e+04.

**A MAGNETIC layer beside an OUT-OF-PLANE layer** (generalized cascade,
tilted-director OOP cell over a uniform-eps magnetic layer, theta 0.20 /
phi 0.40):

| M | 4 | 5 | 6 | 7 |
|---|---|---|---|---|
| `\|sum R + sum T - 1\|` | 1.530e-04 | 2.464e-05 | 9.663e-06 | **4.774e-07** |
| tripwire warnings | 0 | 0 | 0 | 0 |

Drop M=4 -> M=7: 3.2e+02, no blow-up.  **Both composition claims CONFIRMED.**

---

## 4. G1 -- the reduction, re-measured

Covered by Section 1 (bit-identity of `mu_cell=None`, exactly 0 differences)
and by the durability table (Section 5): `mu = 1` FORCED through the magnetic
path reproduces the nonmagnetic assembly to 4.6189e-14 on the worst retained
operator, 6.1548e-14 on the eigenvalue set and 8.4370e-15 through the public
entry.  All three are re-summation differences.  **CONFIRMED.**

---

## 5. TASK 4 -- DURABILITY of `tests/unit/test_pmm2d_staggered_magnetic.py`

`v8_durability.py` imports the test module and drives ITS OWN helpers with ITS
OWN fixtures, so each number below is exactly what the assertion computes.
42 bars, all passing.  `bar` is the constant in the file; `ratio` is the gap
between the measurement and the bar.

| gate | assertion | bar | RE-MEASURED | ratio | flag |
|---|---|---|---|---|---|
| G1 | `mu=1` forced: worst retained-operator rel | < 1e-12 | 4.6189e-14 | 21.7x | |
| G1 | `mu=1` forced: eigenvalue set (rel) | < 1e-12 | 6.1548e-14 | 16.2x | |
| G1 | `mu=1` forced: public dR / dT / dJones | < 1e-12 | 8.4370e-15 | 119x | |
| G2 | nonmagnetic pin (`eps=4, mu=1`, M=8) | < 1e-12 | 2.4549e-14 | 40.7x | |
| G2 | magnetic slabs, worst of 4 pairs x 2 theta | < 1e-12 | 4.2980e-14 | 23.3x | |
| G2 | spectral ladder M=5 -> M=8 | > 1e3 | 7.8771e+05 | 788x | |
| G3 | uniform tensor pair: R/T duality | < 1e-11 | 5.8675e-14 | 170x | |
| G3 | uniform tensor pair: Jones duality | < 1e-11 | 1.7548e-13 | 57x | |
| G3 | uniform pair: no-rotation control | > 1e-2 | 3.0145e-01 | 30.1x | |
| G3 | patterned ladder: R/T at M=8 | < 7.6e-4 | 2.5124e-04 | **3.02x** | SUB-DECADE |
| G3 | patterned ladder: Jones at M=8 | < 7.6e-4 | 2.3173e-04 | **3.28x** | SUB-DECADE |
| G3 | patterned ladder drop M=5 -> M=8 | > 5 | 2.3846e+01 | **4.77x** | SUB-DECADE |
| G3 | patterned: no-rotation control | > 1e-2 | 2.9435e-01 | 29.4x | |
| G4 | M=8 R/T vs `pmm_jones_1d` | < 3e-5 | 7.6301e-06 | **3.93x** | SUB-DECADE |
| G4 | M=8 Jones vs `pmm_jones_1d` | < 7e-5 | 1.7176e-05 | **4.08x** | SUB-DECADE |
| G4 | M=8 R/T vs `rcwa_jones_1d` | < 3e-5 | 6.7565e-06 | **4.44x** | SUB-DECADE |
| G4 | M=8 Jones vs `rcwa_jones_1d` | < 7e-5 | 1.5194e-05 | **4.61x** | SUB-DECADE |
| G4 | unswapped control (both oracles) | > 1e-3 | 4.0741e-02 | 40.7x | |
| G4 | ladder drop M=5 -> M=8 | > 20 | 3.6295e+02 | 18.1x | |
| G4 | y-forbidden (`n != 0`) leak | < 1e-20 | 4.6751e-26 | 2.1e+05x | |
| G5 | uniform Hermitian mu closure | < 1e-11 | 1.0658e-13 | 93.8x | |
| G5 | patterned Hermitian mu closure (M=8) | < 1e-4 | 1.4798e-06 | 67.6x | |
| G5 | patterned closure drop M=6 -> M=8 | > 10 | 1.1703e+03 | 117x | |
| G5 | lossy mu: sum R+T below 0.99 | < 0.99 | 9.5373e-01 | **1.04x** | SUB-DECADE (see below) |
| G5 | tripwire: under-resolved defect | > 0.3 | 8.4879e-01 | **2.83x** | SUB-DECADE (see below) |
| G9 | `layer_absorption` closure at M=8 | < 1e-8 | 9.8667e-10 | 10.1x | |
| G9 | absorption ladder drop M=5 -> M=8 | > 100 | 1.3024e+04 | 130x | |
| G6 | transpose, correct placement (R/T) | < 1e-11 | 2.8005e-14 | 357x | |
| G6 | transpose, correct placement (Jones) | < 1e-11 | 3.8463e-14 | 260x | |
| G6 | `m12`/`m21` swapped (must BREAK) | > 1e-7 | 5.2180e-03 | 5.2e+04x | |
| G6 | `e12`/`e21` swapped (must be a NO-OP) | < 1e-11 | 2.2432e-14 | 446x | |
| FB | knockout `gram` (must BREAK) | > 1e-4 | 3.2580e-01 | 3.3e+03x | |
| FB | knockout `chi33` (must BREAK) | > 1e-4 | 2.3359e-02 | 234x | |
| FB | knockout `chi_t` (must BREAK) | > 1e-4 | 2.2293e-01 | 2.2e+03x | |
| FB | intact analytic residual | < 1e-12 | 3.0641e-14 | 32.6x | |
| FB | mixed blocks: INTACT duality (M=7) | < 1e-11 | 4.2188e-14 | 237x | |
| FB | mixed `chi12`/`chi21` zeroed (must BREAK) | > 1e-4 | 3.4017e-02 | 340x | |
| API | magnetic + out-of-plane cascade at M=7 | < 1e-4 | 3.0812e-06 | 32.5x | |
| API | cascade drop M=4 -> M=7 | > 50 | 2.5817e+03 | 51.6x | |
| G10 | `mu=1`: the two wavelengths DIFFER | > 1e-10 | 3.8567e-08 | 386x | |
| G10 | `mu=1` magnetic vs nonmagnetic | < 1e-12 | 4.4316e-15 | 226x | |

Two bars in the file are EXACT-equality assertions with no ratio (`mu_cell=None`
bit-identity, `max|diff| = 0.0`; and the G10 on-cut-off arm, `np.array_equal`).

**Every number the test file states in a docstring reproduces, except the four
in Section 5.3.**  Spot-checked against the build doc: M1 (4.619e-14 /
6.155e-14 / 8.437e-15), M2A (2.455e-14), M2B (4.298e-14), M2D ladder (7.9e+05),
M3A (2.512e-04 / 2.317e-04), M3B (5.868e-14 / 1.755e-13), M4 (7.630e-06 /
1.718e-05, 6.757e-06 / 1.519e-05, 363x, 4.68e-26), M5A (1.066e-13 / 1.480e-06,
1170x), M5B (0.953739 / 0.552147 at M=6 and 0.953733 / 0.552172 at M=8 --
exact), M6 (2.80e-14 / 5.22e-03 / 1.33e-01 / 2.24e-14 / 3.434e-14 -- exact),
M9 (9.867e-10, 13,000x), M10 (3.081e-06, 2582x).

### 5.1 The nine SUB-DECADE bars, restated

`docs/TESTING_STANDARDS.md` asks for DECADES of gap on both sides.  These nine
have less than one decade on the near side.  All are TEST-ONLY (no library
behaviour depends on them), and all are ALGEBRAIC-convergence quantities whose
value is set by the fixture's corner, not by build noise -- but the honest
statement of each is:

* **G3 patterned ladder (3.0x / 3.3x) and its drop (4.8x).**  The bar `7.6e-4`
  is "3x the worst M=8 reading" by construction, so its near-side gap is 3x BY
  DEFINITION and can never be a decade.  The property that HAS decades is the
  pair taken together: the M=8 level sits 3 decades under the no-rotation
  control (2.94e-01) and the ladder drops 23.8x.  Restated: *the residual is
  three decades below the control AND falls by more than an order of magnitude
  from M=5 to M=8* -- that is the durable claim; `< 7.6e-4` alone is a level
  check on one fixture.
* **G4 (3.9x-4.6x on four bars).**  Same shape: bars set at ~4x the M=8
  reading.  The decades live below (the two 1-D oracles agree with each other
  to 1.9e-07 / 3.4e-07, 34x-40x under the bars, so the oracle floor is not what
  is measured) and above (the unswapped control is 4.07e-02, 3 decades up).
  Restated: *the magnetic arm agrees with BOTH 1-D oracles to within ~40x of
  their mutual spread, and 3 decades better than the un-dualised comparison.*
* **G5 "lossy mu closes below 0.99" (1.04x as a ratio of TOTALS).**  The ratio
  framing is misleading: the quantity with a gap is the DEFICIT.  Measured
  deficit 0.046 (row 0) / 0.448 (row 1) against a required deficit of 0.01 --
  4.6x on the near side, and 4 decades over the ~1e-6 closure defect of the
  same cell with `Im(m11)` removed (measured exactly 1.000000 / 1.000000 here).
  Restated in deficit form the bar is two-sided with 4.6x / 4 decades.
* **G5 tripwire under-resolved (2.83x vs the test's own 0.3).**  The decision
  the test makes is "does the warning FIRE", and the quantity that decides it
  is the shipped window `_STAG_CLOSURE_TOL = 5e-2`: the M=3 defect re-measures
  8.4879e-01, **17.0x** outside it (2 warnings), while the resolved M=8 arm
  re-measures 1.4798e-06, **4.5 decades inside** (0 warnings).  The `> 0.3` assertion is a redundant sanity check on top of the
  warning count; the warning count itself is the two-sided claim and it has
  decades.

None of these needs a code change.  Recommendation: state the G3/G4 bars as the
two-sided pair they already assert (level + ladder + control) rather than
implying the level bar alone carries decades, and restate G5's lossy bar as a
deficit.

### 5.2 Cost of the file

Re-timed single-threaded on this machine: **35 tests, 74.05 s** (the 32 shipped
tests alone, before this verification's three: 74.75 s), slowest test 7.80 s
(`test_g3_duality_patterned_cell_ladder`), then 7.56 s and 7.49 s.  Within the
limits (file < 3 min, test < 40 s, grids <= (3,3), M <= 8).  The build doc's
"32 tests, 74-101 s, slowest 9.5 s" reproduces.

### 5.3 DEFECT -- four fail-before docstrings cite a different quantity

The build doc's table M1b is captioned *"the G2 analytic residual (uniform
isotropic `eps=4, mu=2` slab, theta = 0.35, M=8, max over {R, T, Jones} x
{TE, TM})"*.  Its probe (`validation/probe_pmm2d_staggered_magnetic/
m6_failbefore_guards_cost.py::analytic_residual`) actually returns

```python
max(r["dR_te"], r["dT_te"], r["dR_tm"], r["dT_tm"])
```

-- **R and T only; the Jones is excluded.**  The test file's
`_slab_residual` DOES include the Jones, so the four fail-before docstrings
cite numbers that are not their own assertion's quantity:

| assertion | docstring cites (M1b) | RE-MEASURED (the assertion's own quantity) |
|---|---|---|
| intact `_slab_residual(4, 2, 0.35, 8)` | 1.88e-14 | **3.0641e-14** |
| `gram` knockout | 2.15e-01 | **3.2580e-01** |
| `chi33` knockout | 7.232e-03 | **2.3359e-02** |
| `chi_t` knockout | 2.747e-02 | **2.2293e-01** |

Every deviation is in the SAFE direction (intact still 33x under its bar, each
break still 234x-3300x over its bar), and the same file's G2 docstring reports
the correct 3.064e-14 for the same fixture (build doc M2B), so this is a
citation defect, not a numerical one.  Two fixes are available and both are
cheap: correct the four cited numbers, or make the M1b probe's
`analytic_residual` include the Jones so the doc and the test measure the same
thing.  The M1b DUALITY column is unaffected and reproduces exactly (4.219e-14
intact, 3.402e-02 with the mixed blocks zeroed).

### 5.4 One further doc-vs-assertion mismatch, benign

`test_g3_duality_patterned_cell_ladder`'s docstring says the ladder drop is
"measured 21.9x at worst".  21.9x is the worst PER-INCIDENCE ratio; the
assertion computes `max(d5) / max(d8)` across the two incidences, which is
23.85x (re-measured 23.846x).  The cited number is conservative -- it
understates the assertion's own margin -- so the bar is safe either way.

---

## 6. TASK 5 -- FOLLOW-UP: a MAGNETIC layer's Wood cut-offs sit at `Re(eps*mu)`

### 6.1 The change

`lumenairy/elements/pmm/stack2d_pure.py`, one branch of the Wood-anomaly nudge
list plus two helpers:

```python
def _principal_diag(spec, uniform):     # -> (3,) or (Nx, Ny, 3)
def _wood_cutoff_products(layer):       # eps_ii * mu_ii, componentwise
...
elif _L["kind"] == "magnetic":
    _eps_src.append(_wood_cutoff_products(_L))
```

Rationale: a layer mode goes grazing where its OWN longitudinal wavenumber
vanishes, i.e. at `kt^2 = Re(eps mu)` -- the layer index is `sqrt(eps mu)`.  A
magnetic layer sitting exactly on its own cut-off was therefore never nudged.
The componentwise product of the two principal diagonals is a HEURISTIC for an
anisotropic pair (the true cut-offs of a biaxial magnetic layer are the roots
of its own dispersion relation), which is stated in the docstring; over-listing
is numerically inert off an exact coincidence because
`_grazing_safe_wavelength` takes a MIN over the list and only fires inside a
`|eps - kt^2| <= 1e-9` band.  `pmm_efficiency_2d_staggered`'s call site is
untouched (it cannot be magnetic).

### 6.2 BIT-IDENTITY of every nonmagnetic call site

`v7_wood_followup.py hash`, run on both sides of the edit and diffed.  15
hashed fixtures:

`eff_te`, `eff_tm` (the scalar efficiency entry, TE and TM), `jones_scalar`,
`jones_tensor` (the Jones entry), `stack_patterned`, `stack_uniform`,
`stack_uniform_tensor`, `stack_tensor_cell` (the four nonmagnetic stack
branches), `magnetic_mu1_scalar`, `magnetic_mu1_tensor`, `magnetic_mu1_cell`,
`magnetic_mu1_tcell` (a MAGNETIC layer with `mu = 1` in all four spellings),
`magnetic_offcut`, `magnetic_offcut_tensor`, `magnetic_offcut_gyro` (magnetic
layers off any cut-off).

```
diff wood_before.txt wood_after.txt   ->   (no output)
BIT-IDENTICAL on all 15 hashed call sites
```

`mu = 1` multiplies by exactly `1.0`, which is exact in float64, so the product
IS the permittivity bit for bit -- that is why the four `mu = 1` spellings and
the eight nonmagnetic entries cannot move.  The full V1 bit-identity harness
(12 fixtures / 75 quantities vs the main clone) was also re-run after this
change: still 0 differences.

### 6.3 The two-sided behaviour

Fixture: `px = py = 0.50 um`, normal incidence, `n_sup = 1.0`, `n_sub = 1.5`,
layer `eps = 4.0`, `mu = 2.25`, `wl = px sqrt(eps mu)` so the `(+/-1, 0)` and
`(0, +/-1)` orders sit EXACTLY on `kt^2 = eps mu` (`(wl/px)^2 - eps*mu ==
0.0` in float64), while `eps`, `mu` and both half-spaces are each 1.0-2.25 away
from every `kt^2`.

**The rules, driven directly through the library's own guard:**

| list handed to `_grazing_safe_wavelength` | nudged? | dwl/wl |
|---|---|---|
| half-spaces only | no | 0 |
| + `eps` only (the pre-follow-up rule) | **no** | 0 |
| + `eps` and `mu` SEPARATELY | **no** | 0 |
| + `eps*mu` (the follow-up) | **yes** | 1.000e-07 |

**The public surface:**

| arm | result |
|---|---|
| MAGNETIC `mu = 2.25`: `solve(wl_cut)` vs `solve(wl_nudged)` | **BIT-IDENTICAL** (`max|dR| = 0.0`) -- the guard fired and landed exactly where the rule says |
| the SAME layer with `mu = 1`: `solve(wl_cut)` vs `solve(wl_nudged)` | **DIFFER** by 3.857e-08 -- correctly NOT nudged |
| `mu = 1` magnetic vs the NONMAGNETIC layer at `wl_cut` | 4.432e-15 (re-summation only; both took the same, empty nudge) |

**FAIL-BEFORE**, run by checking out the pre-change `stack2d_pure.py` at
`6454f68` and re-running the new tests: `test_g10_a_magnetic_layer_on_its_
cutoff_is_nudged` FAILS with `AssertionError: 7.712343159937962e-09`; the other
two pass (they are the unchanged side).  After the change all three pass.

### 6.4 The tests

`tests/unit/test_pmm2d_staggered_magnetic.py`, gate G10 (3 tests, 1.18 s):

* `test_g10_the_fixture_is_on_the_eps_mu_cutoff_and_only_that_rule_sees_it` --
  the fail-before as a decision about the RULES: all four lists are handed to
  the library's own `_grazing_safe_wavelength`, and only the `eps*mu` one moves
  the wavelength.  The coincidence is asserted exact (`== 0.0`) and every lone
  quantity is asserted `> 0.9` from every `kt^2`, so the fixture isolates the
  product.
* `test_g10_a_magnetic_layer_on_its_cutoff_is_nudged` -- the public-surface
  proof: `np.array_equal` between the cut-off solve and the nudged solve, with
  the nudged wavelength taken from the library's own guard (never a recorded
  constant).
* `test_g10_the_same_layer_with_unit_mu_is_not_nudged` -- the other side, plus
  the `mu = 1` vs nonmagnetic re-summation identity.

CHANGELOG entry added under `[Unreleased]`.

---

## 7. TASK 6 -- SUITES, RUFF, DURATIONS

**The staggered / pure regression set**, all 13 files in one single-threaded
run on `verify/magnetic` (i.e. WITH this verification's library change and its
three new tests):

```
python -m pytest tests/unit/test_pmm2d_staggered_magnetic.py   tests/unit/test_pmm2d_staggered_anisotropic.py   tests/unit/test_pmm2d_staggered_oop.py   tests/unit/test_pmm2d_staggered_wood_list.py   tests/unit/test_v5_12_0_pmm2d_staggered.py   tests/unit/test_v5_21_pmm2d_staggered_oblique.py   tests/unit/test_staggered.py   tests/unit/test_audit_p1_staggered_guard.py   tests/unit/test_p2c_pmm2d_stack_cascade.py   tests/unit/test_p2t_pmm2d_tree_cascade.py   tests/unit/test_pmm2d_lossless_closure_two_sided.py   tests/unit/test_audit_s1_3_pmm2d_lossless_tripwire.py   tests/unit/test_v5_14_0_pmm2d_stack.py -q -p no:randomly
```

**277 passed, 23 warnings, 673.48 s (0:11:13).  0 failures, 0 errors, 0 skips.**
The 23 warnings are pre-existing `PMM2DStack` transitional-alias
`DeprecationWarning`s in `test_v5_14_0_pmm2d_stack.py`, unrelated to this work.
Slowest: `test_pmm2d_lossless_closure_two_sided::
test_solver_warning_matches_the_predicate_on_every_arm` 131.8 s.

This run is also the evidence for two build claims that have no probe of their
own: the closure TRIPWIRE does not fire on any existing fixture (the two
tripwire suites are green), and the shipped nonmagnetic staggered paths did not
move (every one of these files predates the magnetic build).

**Ruff:**

```
ruff check lumenairy/ tests/                       -> All checks passed!
ruff check validation/probe_verify_magnetic/       -> All checks passed!
```

**`.test_durations` spliced** with `--store-durations` on the magnetic file:

| | before | after |
|---|---|---|
| total entries | 12331 | 12334 |
| entries for `test_pmm2d_staggered_magnetic.py` | 32 | 35 |

3 ADDED (exactly the three G10 tests), 0 removed, 32 re-timed, and **0 entries
outside the magnetic file changed**.

---

## 8. Defects and recommendations

| # | severity | finding |
|---|---|---|
| D1 | doc | Build doc Section 1.2's Eq.-43 sentence: the appendix does NOT re-pair chi with the derivatives (row 1's pairing matches Eq. 21 exactly).  Its defects are an overall SIGN flip, a DOUBLE-primed first term in `K_tz22` (unreported), and the `d2`-for-`d1` typo in the second term.  Section 2.4. |
| D2 | doc/test | Four fail-before docstrings cite build-doc M1b numbers that are R/T-only maxima, while the assertions' own quantity includes the Jones: 1.88e-14 / 2.15e-01 / 7.232e-03 / 2.747e-02 cited vs 3.064e-14 / 3.258e-01 / 2.336e-02 / 2.229e-01 measured.  All in the safe direction.  Section 5.3. |
| D3 | doc | Build doc M1b's caption claims "max over {R, T, Jones}" but its probe excludes the Jones.  Root cause of D2. |
| D4 | test | Nine sub-decade bars, all test-only and all with a decade-scale property available one level up.  Section 5.1. |
| D5 | doc | `test_g3_duality_patterned_cell_ladder` cites 21.9x for a quantity that measures 23.85x (conservative).  Section 5.4. |
| D6 | -- | (repaired before this verification: the `5c384e5` merge's `L["eps_cell"]` KeyError on every magnetic stack, fixed at `6454f68`.) |

No library defect was found in the magnetic build.

## 9. What this verification could NOT establish

* **The build doc's own probe numbers** were not re-run; every number in this
  report is my own measurement on my own fixtures.  Where the two disagree
  (Section 5.3) the disagreement is explained, not assumed.
* **Cross-machine / cross-BLAS behaviour.**  Everything here is one Windows
  box, one numpy, one thread.  The build doc reports a WSL green run; I did not
  reproduce it.
* **The `(s, p)`-frame sign convention** the build states for its duality
  oracle.  I derived and used the LAB TANGENTIAL rule instead, which needs no
  p^-hat convention; I showed algebraically that the build's `-D J D^-1`
  follows from its stated asymmetric p^-hat choice, but I did not measure their
  frame.
* **Anisotropic magnetic Wood cut-offs.**  The follow-up's componentwise
  `eps_ii * mu_ii` is a heuristic; the true cut-offs of a biaxial magnetic
  layer are the roots of its own dispersion relation.  I verified the ISOTROPIC
  case exactly and documented the heuristic; I did not derive or test the
  biaxial one.
* **Out-of-plane mu, magnetic half-spaces, the JAX twin, and the tapered /
  z-staircase helpers** are out of scope by construction (they raise, or do not
  exist); I verified that they raise, not that any future implementation would
  be correct.
* **Performance.**  The build's cost table (M8) was not re-measured; nothing in
  this verification depends on it.

## 10. Commits on `verify/magnetic`

| commit | what |
|---|---|
| `db887f9` | probes V1-V5 (bit-identity + localisation, the derivation, the analytic oracle, duality, the 1-D bridge) |
| `93c0701` | probe V6 (closure, transpose, guards, composition) |
| `5a4e78b` | **library**: a magnetic layer's Wood cut-offs sit at `Re(eps*mu)`; G10 tests; CHANGELOG |
| `431a2ea` | probes V7 (the follow-up's hashes) and V8 (the durability re-measure) |
| `8ba7b48` | this report + the `.test_durations` splice (3 added, 0 outside the file changed) |

Nothing was pushed, tagged or version-bumped.  The only library edit is
`5a4e78b`, the named follow-up.
