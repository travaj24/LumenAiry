# Round 2 of the BOR multilayer guards — 5.45.1, 2026-09-12

> **STATUS — REMEDIATION.** Of the four blocking defects the independent
> verification `docs/audits/VERIFY_BOR_MULTILAYER_GUARDS_2026_09_12.md` left
> open on `fix/bor-multilayer-guards`, plus the restatements it asked for.
> Every number below was measured on this tree, on both local builds, across
> the OpenBLAS kernel and thread ladders; nothing is read out of the
> verification's report and repeated.
>
> Tree: `C:\tmp\lum_bor2`, branch `fix/bor-guards-round2`, forked from
> `verify/bor-multilayer-guards` (`f2d331c5`) so its four strict xfails are
> inherited and must be flipped.
> Pre-fix tree for the bit-identity leg: `C:\tmp\lum_bor2_pre`, detached at the
> same `f2d331c5`.
> Probes and JSON: `validation/probe_fix_bor_round2/`.
> Gates: `tests/unit/test_verify_bor_multilayer_guards.py` (the four flipped),
> `tests/unit/test_fix_bor_guards_round2.py` (new, the populations),
> `tests/unit/test_fix_bor_multilayer_guards.py` (D9 restated),
> `tests/unit/test_fix_eme_branch_cut.py` (the band gate restated).
>
> Binding documents: `docs/TESTING_STANDARDS.md`,
> `docs/audits/PLAN_BOR_MULTILAYER_GUARDS_5_45_1_2026_09_12.md`,
> `docs/audits/BUILD_BOR_MULTILAYER_GUARDS_2026_09_12.md`.

---

## 1. Terms

Defined before they are used, because three of the four fixes turn on a
distinction the names do not carry.

**BOR — body of revolution.** An axisymmetric stack: layers along `z`, each a
set of concentric rings in `r`, closed at `r = Rbig` by a PEC (Dirichlet) wall.
Solved one azimuthal order `m` at a time; fields go as `exp(i m phi + i q z)`,
with `q` the **axial wavenumber** and `qn = q / k0` its dimensionless form.

**The three radial bases.** `basis='fd'` — a div-conforming Yee staggered
finite-difference basis on one uniform radial grid. `basis='sem'` — a
spectral-element basis, one mesh per layer, layers coupled by a mortar.
`bor_solve.build_layer(basis='nodal')` — the historical nodal FD basis, which
is **not** divergence-conforming and therefore carries a divergence-violating
**spurious mode sea**. Only the nodal path is screened here.

**Passive medium.** One with `Im eps >= 0` in this library's `exp(-i omega t)`
convention: absorbing or lossless, but **not amplifying**. For a stack of
passive media, energy conservation reads `R + T + A = 1` with the absorbed
fraction `A >= 0`, so

* **`R + T <= 1` is a theorem on ANY passive stack, lossy or not**, and
* on a **LOSSLESS** passive stack inside a PEC wall there is no absorption and
  no other exit, so **`R + T = 1` is an EQUALITY**.

Those two sentences are the whole of defects D1 and D2. **Gain** (`Im eps < 0`)
is not passive: there is no theorem in either direction and the screen must not
speak.

**Channel.** A propagating diffraction channel of the cascade — one entry of
the `R` / `T` arrays. For a UNIFORM PEC-walled half-space of index `n` the
number of them is a **closed form**: TE modes satisfy `J_m'(gamma Rbig) = 0`,
TM modes `J_m(gamma Rbig) = 0`, and a mode propagates when `gamma < n k0`. That
closed form depends on no solver in this library and is used below as an oracle.

**Index ceiling.** `Re qn <= Re sqrt(eps_max)`. `q^2` is an eigenvalue of
`eps k0^2 + D` with `D` the transverse operator; wherever `D` is negative
semi-definite — the continuum problem, and any div-conforming discretisation of
it — the Rayleigh quotient bounds `q^2 <= max(eps) k0^2`. A channel above it
has `gamma^2 < 0`: a transverse eigenvalue the PEC-walled cylinder does not
have. It is a **contradiction, not a magnitude**, which is why a refusal keyed
on it cannot move with the BLAS kernel.

**Set-wrong.** A nodal census row whose channel COUNT differs from the
div-conforming staggered twin's on the same geometry. Used throughout §4 as a
definition of damage that reads **no energy at all**, so an energy bar can be
scored against it without assuming its own conclusion.

**Arms.** A build (Windows py3.14.6 / numpy 2.4.4 / scipy 1.17.1, or WSL
py3.12.3 / numpy 2.4.6 / scipy 1.17.1) × a requested `OPENBLAS_CORETYPE` × a
thread count. The **loaded** kernel is read back from
`threadpoolctl.threadpool_info()` in every run and is what is reported;
`PRESCOTT` loads **Katmai** on this host, `ZEN` silently aliases Haswell, and
`SKYLAKEX` dies on the first LAPACK call. Every command carries
`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` explicitly.

---

## 2. Verdict table

| # | defect | severity | status | the number that decides it |
|---|---|---|---|---|
| D1 | a negligible loss disarms the nodal passivity screen | P1 | **FIXED** | the loss ladder on the refused stack: **4 of 13 rungs refused → 13 of 13**, with the nodal violation pinned at the same 2.881869e-02 excess / 3.822431e-02 deficit on every rung; healthy lossy stacks **0 of 39** refused, before and after |
| D2 | the screen is one-sided on a stack it proved lossless | P2 | **FIXED**, and the bar RE-DERIVED | the 0.537349 row is refused and its message names the DEFICIT; a single scalar energy bar has **no** two-sided gap (the populations overlap by 9.97 decades), so the fix adds a deterministic conjunct — the **index ceiling**, which fires on 40 of 44 set-wrong rows and **0 of 116** rows that are not damaged |
| D13 | `eme/_branch.cut_band`'s literal 1.0 floor is unit-dependent | P2 | **FIXED** | the same cell in um / nm / m: **96 of 96 sorted roots differed → 0**, orientation census identical in all three, `mode_match` `T00` spread **5.418e-08 → 2.220e-15** |
| D9 | the near-cutoff gate's bar is already crossed | P2 (test) | **RESTATED** | family envelope over `m` ∈ {0,1,2} and **10 arms**: **1.271613e-06** against the shipped `1e-6` bar (0.786x — crossed); new bar 1e-5 carries **7.86x (0.90 decades)** |
| D4 | `warn_own` blames a layer that asked for nothing | P3 | **FIXED** (it was a strict xfail) | the liner's neighbour no longer receives the message; `warn_own` now reads `w_min_own_frac` |

Restatements, §6: the band's SIGNAL side, the `k0` floor, the SEM warn edge,
the Q bar's vacuous margin, `.test_durations`.

---

## 3. D1 — the passivity predicate

### 3.1 What it said, and why that was wrong

`bor_solve._stack_is_provably_passive` required EVERY layer's
`max|Im eps| / max|Re eps|` to be `<= _BOR_LOSSLESS_REL_IM = 1e-12` and
disarmed the whole screen otherwise, on the stated reasoning *"on a lossy stack
there is no theorem to violate."*

That reasoning covers the BELOW-unity direction only. `R + T + A = 1` with
`A >= 0` gives `R + T <= 1` on every passive stack, absorbing or not. The
predicate therefore had a discontinuity at `Im(eps) = 0+` — the same shape this
very wave removed from the EME module — and a 3e-12 relative loss walked
through it.

### 3.2 The loss ladder, measured on both sides of the fix

`validation/probe_fix_bor_round2/r2_loss_ladder.py`, ladder A: the shipped
refusal fixture (`m = 1`, `Rbig = 4`, `N = 200`, `k0 = 2`, a ring-grating layer
between two `eps = 2` half-spaces) with a relative loss `Im/Re` on the ring's
high region. The `staggered` column is the div-conforming twin, i.e. the
physical absorption the loss actually causes.

| `Im(eps)/Re(eps)` | PRE verdict | POST verdict | nodal `max(R+T)-1`, guard off | staggered twin `R+T` |
|---|---|---|---|---|
| 0 | REFUSED | REFUSED | +2.881869e-02 | 0.999999999999 .. 1.000000000000 |
| 1e-14 | REFUSED | REFUSED | +2.881869e-02 | 0.999999999999 .. 1.000000000001 |
| 1e-13 | REFUSED | REFUSED | +2.881869e-02 | 0.999999999999 .. 1.000000000000 |
| 1e-12 | REFUSED | REFUSED | +2.881869e-02 | 0.999999999998 .. 1.000000000000 |
| **3e-12** | **returned, 0 warnings** | **REFUSED** | +2.881869e-02 | 0.999999999994 .. 0.999999999998 |
| 1e-11 | returned | REFUSED | +2.881869e-02 | 0.999999999983 .. 0.999999999991 |
| 1e-9 | returned | REFUSED | +2.881869e-02 | 0.999999998427 .. 0.999999999071 |
| 1e-8 | returned | REFUSED | +2.881867e-02 | 0.999999984279 .. 0.999999990703 |
| 1e-6 | returned | REFUSED | +2.881680e-02 | 0.999998427992 .. 0.999999070256 |
| 1e-4 | returned | REFUSED | +2.863014e-02 | 0.999842815579 .. 0.999907031373 |
| 1e-3 | returned | REFUSED | +2.696481e-02 | 0.998429642947 .. 0.999070846337 |
| 1e-2 | returned | REFUSED | +1.046281e-02 | 0.984446427219 .. 0.990762224605 |
| 1e-1 | returned | REFUSED | **-3.866777e-02** | 0.859918277746 .. 0.913272934202 |

**4 of 13 → 13 of 13 refused.** The returned violation is the SAME 2.9 % across
the whole ladder; only the guard changed.

**Per arm, PRE and POST.** The nodal blow-up is a deterministic discretisation
defect, not an arithmetic one, so the decision ladder is the same everywhere it
was run (`r2_arms_win.sh` / `r2_arms_wsl.sh`, `--fast`, which runs ladder A and
the GAIN ladder C on each arm):

| arm (LOADED kernel) | PRE: A refused | PRE: C refused | POST: A refused | POST: C refused |
|---|---|---|---|---|
| win / Haswell / t1 | 4 / 13 | 1 / 5 | **13 / 13** | **0 / 5** |
| win / Nehalem / t1 | 4 / 13 | 1 / 5 | **13 / 13** | **0 / 5** |
| win / Katmai / t1 | 4 / 13 | 1 / 5 | **13 / 13** | **0 / 5** |
| win / Sandybridge / t1 | 4 / 13 | 1 / 5 | **13 / 13** | **0 / 5** |
| wsl / Haswell / t1 | 4 / 13 | 1 / 5 | **13 / 13** | **0 / 5** |
| wsl / Nehalem / t1 | 4 / 13 | 1 / 5 | **13 / 13** | **0 / 5** |
| wsl / Katmai / t1 | 4 / 13 | 1 / 5 | **13 / 13** | **0 / 5** |
| wsl / Sandybridge / t1 | 4 / 13 | 1 / 5 | **13 / 13** | **0 / 5** |

Eight arms, four LOADED kernels, two builds -- **identical to the rung on every
one of them, before and after.** That is what "the nodal blow-up is a
deterministic discretisation defect, not an arithmetic one" means measured
rather than asserted, and it is why a passivity screen against it does not need
the per-arm re-derivation a backward-error-driven bar would.


C is the GAIN ladder: `Im eps < 0` is not passive, has no energy theorem in
either direction, and must sit outside the screen. The pre-round-2 predicate
tested `|Im eps|` and so judged an infinitesimal gain as though it were
lossless; round 2 tests the sign, and that rung moves from refused to returned.
 The last rung is worth naming
separately: at `Im/Re = 1e-1` the nodal cascade's `max(R+T)` is *below* unity,
so the super-unity half has nothing to say — it is the **index ceiling**
conjunct of §4.4 that refuses that row, which is exactly the population that
conjunct was added for.

### 3.3 The shipped gate that pinned the pre-fix behaviour

The verification named it in D1's remedy: *"The shipped gate
`test_a_lossy_nodal_stack_is_never_judged_by_the_passivity_screen` currently
pins the wrong behaviour and would have to move with it."* It does, and the
fixture it has always built is the proof that the old reasoning was wrong.

That fixture's ring ABSORBS (`Im eps = 0.3` on a `Re eps = 6` region), and the
legacy nodal cascade returns **`max(R + T) = 99.8789`** on it — a hundredfold
super-unity that no amount of absorption can excuse, handed back in silence
because one layer was lossy. It is now refused, and the gate is restated as
`test_a_lossy_nodal_stack_is_judged_by_the_SUPER_UNITY_HALF_ONLY`: it asserts
the refusal, that the message names the theorem (`R + T <= 1`) and that it does
NOT call an absorbing stack lossless, and it is premise-gated on the arm
actually producing the blow-up. The complementary claim — that the DEFICIT half
stays disarmed on a lossy stack — is its own gate in the verification file, over
`Im/Re` from 1e-6 to 1e-1 on uniform stacks whose deficits legitimately reach
0.1546.

### 3.4 The other side of the predicate: nothing healthy moved

Ladder B — uniform half-spaces around a uniform lossy middle layer, the family
the 5.45.1 census calls accurate, over `m` = 0/1/2 × the same 13 rungs:
**0 of 39 refused and 0 warned, before and after.** Their deficits track the
loss exactly (`-1.782e-06` at `Im/Re` = 1e-6 up to `-1.546e-01` at 1e-1), which
is also the measurement that forbids arming the DEFICIT half there — see §4.2.

Ladder C — **gain**. `Im eps < 0` is not passive and now sits outside the
screen entirely: **1 of 5 rungs refused → 0 of 5**. The changed rung is
`Im/Re = -1e-12`, which the pre-round-2 predicate admitted because it tested
`|Im eps|` rather than the sign. That is a deliberate behaviour change, it is
the conservative direction, and it is the 1-D peer's convention
(`pmm/_core._tensor_is_passive` requires `Im >= 0`, not `|Im|` small).

### 3.5 The predicate, and where it came from

`_stack_is_provably_passive` now mirrors the 1-D peer
`pmm/stack._stack_provably_passive`:

* every layer **provably passive** — `Im(eps) >= 0` to a **16-ULP deadband**
  relative to that profile's own `max|eps|`, the sizing constant
  `pmm/stack._PASSIVE_ANTIHERM_DEADBAND` uses for the same decision;
* the **incidence half-space lossless** — kept, because that is where the
  theorem needs it. `R` and `T` are formed from a basis normalised to unit
  `|z-flux|` per mode, and in an absorbing incidence medium a mode's flux is not
  a conserved power, so the sums are not power fractions. The peer excludes an
  absorbing superstrate for the same reason and records the 1.00026 / 1.0152 /
  1.0303 super-unity ladder it legitimately produces. Gated by
  `test_the_screen_disarms_on_an_absorbing_incidence_medium`;
* anything the predicate cannot resolve answers `False`, which leaves the solve
  exactly as it was.

> **RESTATEMENT AND FIX, 2026-09-12 (round 3, verification GAP 2).** The
> incidence-lossless conjunct above is correct about the ENERGY and was wrong
> about its SCOPE: until round 3 it lived in the ONE predicate both detectors
> shared, so a loss of any size on `layers[0]` — 3e-12 included, the exact rung
> D1 was named for — disarmed the INDEX CEILING as well. That is D1's own shape
> relocated rather than removed, and it is measurable: on the verification's
> ladder D the nodal cascade returns `max(R + T) = 2.41297` in silence while
> the div-conforming twin on the identical geometry closes to 8.75e-07.
> `_check_nodal_passivity` now gates the two separately —
> `_stack_media_are_passive` (`Im eps >= 0` everywhere; GAIN disarms
> everything) gates BOTH, and the incidence-lossless conjunct gates the energy
> detector ALONE — because the ceiling's Rayleigh bound concerns a half-space's
> own `eps` and is untouched by whether the incidence medium's flux is
> conserved. Re-measured: ladder D **4 of 13 → 13 of 13 refused** (4 energy, 9
> ceiling), with **0 refusals and 0 warnings on 52 rows** of healthy
> channel-set-agreeing stacks carrying the same loss.
>
> The gate named above, `test_the_screen_disarms_on_an_absorbing_incidence_
> medium`, was **VACUOUS** (verification GAP 1): its superstrate at `Im/Re =
> 1e-3` put every mode past `_orient._BOR_CHANNEL_IMAG_BAR`, so the channel set
> was empty and `_check_nodal_passivity` returned at its `e.size == 0` line
> without ever calling the predicate — the round-2 verification measured that
> it still passed with the conjunct deleted. It is retired. Its live
> replacement is `test_fix_bor_guards_round2.py::
> test_the_energy_screen_disarms_on_an_absorbing_incidence_medium_but_the_
> ceiling_does_not`, which runs at `Im/Re = 1e-7` (two decades inside the
> channel gate), asserts its own premise, and asserts both halves of the
> round-3 split.

**On the tensor arm.** The peer's general statement is that the anti-Hermitian
part `(eps - eps^H) / 2i` is positive semi-definite. On this path that statement
is not approximated, it is *reduced*: `bor_solve.build_layer`'s permittivity
reaches the solver as a 1-D complex array of radial samples
(`zcascade.layer_modes` does `np.asarray(eps_profile(r), dtype=complex)` and
uses it as a diagonal), and for a scalar or diagonal `eps` the anti-Hermitian
part **is** `Im(eps)`. So the sample-by-sample test implemented here is the
peer's test on the only payload the nodal cascade accepts, with the peer's
deadband; a payload that could not produce the fact answers `None` and disarms.
No dead tensor branch was added, because a dead branch is a trap.

---

## 4. D2 — the screen's second side, and the conjunct that does not use energy

### 4.1 The defect, reproduced

`_check_nodal_passivity` tested only `max(R+T) - 1`. The verification measured
5 of 48 provably lossless nodal rows returning `R + T < 1` silently, worst
0.537349. Reproduced here at **0.5373486** — `m = 2`, `Rbig/lambda = 0.5`,
`N = 200`, ring layer — whose staggered twin closes to 6.5e-12. It is now
refused, and the message says which half decided it:

```
bor_solve.solve(basis='nodal'): the cascade returned max(R + T) = 0.537349 on a
PROVABLY PASSIVE LOSSLESS stack (cell radius 0.50 vacuum wavelengths) -- a
violation of 0.4627 against the bar 1e-03.  It is a DEFICIT: min(R + T) =
0.537349, max(R + T) = 0.537349, and on this stack R + T = 1 is an EQUALITY
(no absorption, and a PEC wall leaves no other exit).  ...
```

### 4.2 Which half is armed where

| stack | `R + T <= 1` | `R + T = 1` | screen |
|---|---|---|---|
| provably passive, LOSSLESS everywhere | theorem | theorem | **two-sided**, `\|R+T-1\| > bar` |
| provably passive, some layer absorbs | theorem | false | **one-sided**, `R+T-1 > bar` |
| gain anywhere, or absorbing incidence medium | — | — | **disarmed** |

The middle row is not a convenience: ladder B's deficits reach 0.1546 on a
legitimately absorbing stack, so a two-sided screen armed there would refuse
every absorbing stack in the library. Gated by
`test_a_lossy_nodal_stack_keeps_its_deficit_half_disarmed`.

The LOSSLESS threshold stays `_BOR_LOSSLESS_REL_IM = 1e-12`, now measured
rather than asserted: ladder A's staggered column shows a stack at exactly that
ratio absorbing **2e-12** of the incident power, and 1.6e-09 at `Im/Re` = 1e-9
— nine decades below the deficit bar, so a stack this predicate calls lossless
cannot absorb its way past it.

### 4.3 Re-deriving the bar — and the honest answer, which is that one scalar
bar does not separate the populations

Census: `validation/probe_fix_bor_round2/r1_passivity_census.py`, **160 solves**
with every guard disarmed — 2 bases × {uniform, ring} × `m` = 0..3 ×
`N` = 120/200 × `Rbig/lambda` = 0.5 .. 8 — each row solved TWICE on the same
geometry, once nodal and once staggered. The staggered basis is the reference
for the channel SET (not for the energy, which is what is being screened): its
own worst `|R+T-1|` over all 80 rows is **6.458611e-12**, and on the 16 rows
where the Bessel-zero oracle is trustworthy (uniform, basin > 1e-2, `m >= 1`) it
returns the closed-form count on **16 of 16**.

Scored against **set-wrong**:

| population | n | `\|R+T-1\|` |
|---|---|---|
| STAGGERED, every row | 80 | ≤ 6.458611e-12 |
| NODAL, channel set RIGHT | 36 | 4.662937e-15 .. **1.950826e+03** |
| NODAL, channel set WRONG | 44 | **2.081921e-07** .. 1.763059e+02 |

**The two populations overlap by 9.97 decades.** There is no scalar energy bar
that separates "accurate" from "damaged", and the verification's D3 —
*"the bar is a detector of catastrophe, not of correctness"* — is confirmed as a
population statement, not merely as a framing. Saying so is part of the fix.

**What the energy bar DOES have a two-sided gap against** is the sub-population
it is the only detector for: rows whose channel set is right AND whose index
ceiling is silent, i.e. the rows no deterministic conjunct can see. There the
accurate (uniform) and damaged (ring) families separate cleanly:

| rows with the right set and a silent ceiling | n | `\|R+T-1\|` |
|---|---|---|
| UNIFORM (the accurate family) | 18 | 4.662937e-15 .. **2.714107e-07** |
| *— nothing in between —* | | |
| RING | 18 | **2.173049e-02** .. 1.950826e+03 |

**4.90 decades with nothing in it.** `_BOR_NODAL_SUPERUNITY_BAR = 1e-3` sits
**3.57 decades above** the accurate ceiling and **1.34 decades below** the
mildest row it must refuse, two-sided on the population it owns. The bar's
value is therefore UNCHANGED; what changed is the population it is justified
against and the direction it looks. Re-measured on every running arm by
`test_the_nodal_energy_bar_has_a_two_sided_gap_on_the_population_it_owns`.

**The warn edge.** `_BOR_NODAL_SUPERUNITY_WARN = 1e-6` sits only **0.57
decades** above the accurate ceiling on the two-sided measure (2.714107e-07),
where the build stated 2.65 decades against a one-sided ceiling of 4.4336e-09.
It is kept — it emits a `UserWarning`, never a refusal, its false-positive cost
is one message, and the quantity is kernel-stable by construction (the nodal
blow-up is a deterministic discretisation defect) — but the margin is recorded
here as **0.57 decades on the two-sided measure**, and §6 states it in the
constant's own comment rather than leaving 2.65 standing.

### 4.4 The deterministic conjunct: the index ceiling

Keyed on a contradiction rather than on a magnitude, so no arithmetic can move
it: a returned channel whose `Re qn` exceeds its own half-space's
`Re sqrt(eps_ceiling)` has `gamma^2 < 0`.

Measured over the same 160 solves, as `Re qn / n - 1`:

| population | n | `Re qn / n - 1` |
|---|---|---|
| STAGGERED, every row | 80 | ≤ **-3.027306e-05** (never positive) |
| NODAL, channel set RIGHT | 36 | ≤ **-1.413702e-03** (never positive) |
| NODAL, channel set WRONG | 44 | **+5.026594e-06 .. +1.296643e-03 on 40 of them** |

**The two populations are on opposite sides of zero.** In absolute `qn` the
mildest violation is 7.1e-06, which is **4.15 decades** above
`_BOR_INDEX_CEILING_SLACK = 5e-10` — and that slack is not a new constant: it
is the same absolute deadband the div-conforming twins already apply as their
own index ceiling (`bor_stack.solve`'s `prop()` at both sites,
`_jax_bor._mask`, `_jax_sem`: `Re sqrt(eps) - Re qn > -5e-10`). The nodal
channel gate deliberately does not apply it as a FILTER — audit S1-16 measured
that over-filtering the reldiv-screened set degrades the basis's own floor — so
round 2 reads it as a REFUSAL, which filters nothing.

**False positives: zero on 116 rows that are not damaged** (80 staggered + 36
nodal-with-the-right-set).

**Corroborated against the closed form.** On the 16 rows where the Bessel-zero
oracle is trustworthy, the ceiling fires on 4 and the closed-form count
disagrees with the nodal count on **4 of 4**. The oracle is deliberately NOT the
production predicate: it exists only for a uniform layer, needs `scipy.special`
in a solve path, and the verification measured its own `m = 0` bookkeeping off
by one and its large-cell misses (12 of 40). The index ceiling is the same
contradiction's mechanism and is exact on every profile. That substitution is
what `test_the_index_ceiling_agrees_with_the_bessel_zero_oracle` licenses.

**What the union catches, and what it still misses.** `|R+T-1| > 1e-3` OR the
ceiling fires on 60 of the 80 nodal rows and misses **2 of 44** set-wrong rows;
**0** rows that are not set-wrong are caught by the ceiling alone. The two
misses are recorded, not papered over: they are rows whose spurious channels
carry no power and stay under their own index ceiling, and nothing in this round
detects them.

### 4.5 Order of the two detectors

The energy screen is evaluated FIRST and the ceiling second, so a row the 5.45.1
screen already refuses keeps the message and the quoted violation it has always
had (the shipped gate asserts the string `"1.02882"`; it still appears), and the
new detector speaks only where the energy is silent — which is the population it
was measured to add.

**And the ceiling is gated on the half-space being LOSSLESS.** That is where its
Rayleigh bound is airtight: with a real symmetric `eps k0^2 + D` the spectrum is
real and `q^2 <= max(eps) k0^2` exactly, while with a complex `eps` the operator
is complex-symmetric and the same statement about `Re q` is only approximate.
Refusing on a bound it cannot prove would be the shape this whole round is
removing, so it stays silent there. Every half-space of the 160-solve census is
lossless, so nothing measured is given up — and the loss ladder's last rung,
where the ring absorbs 10 % but both half-spaces are real, is still refused by
it.

---

## 5. D13 — the EME branch band's floor

### 5.1 The defect

`cut_band` floored the spectrum scale at a **literal 1.0**, justified in its own
comment by *"``ky`` here is DIMENSIONLESS (the EME modules work in
``k0``-normalized units)"*. That is false: `strip_x_modes` assembles
`d2/dx2 + eps k0^2` on a spacing `Lx / Nx`, so `lam` carries 1/length² and `ky`
carries 1/length; `mode_match` forms `exp(i qz depth)`, dimensionless only
because `qz` is 1/length. `k0` is a free argument carrying units, not a
normalisation. The floor therefore engaged whenever `max|ky| < 1` in the
caller's units — the ordinary case for a sub-micron cell written in nanometres —
and this is a **regression**: the pre-5.45.1 exact-zero pin had no scale at all.

### 5.2 Measured, one cell in three unit systems

`validation/probe_fix_bor_round2/r3_eme_units.py`: one 1 um cell at
`lambda = 1550 nm`, `Nx = 96`, `eps_hi = 12 - 1e-6i` (weak gain), written in
micrometres, nanometres and metres. The unit-free observable is `ky * scale`,
and the spectra are **sorted before differencing** — the eigensolver's ordering
is not stable between two solves, and an elementwise difference of the raw
arrays measures a permutation rather than a moved root (this probe's own
first-pass numbers were wrong for exactly that reason and were re-measured).

| | PRE (literal 1.0) | POST (`k0` floor) |
|---|---|---|
| band, um / nm / m, in **unit-free** terms | 1.919e-07 / **1.000e-06** / 1.919e-07 | **1.919e-07 / 1.919e-07 / 1.919e-07** |
| orientation census, um | kept 0, negated 96, conj 0 | kept 0, negated 96, conj 0 |
| orientation census, **nm** | kept 0, negated **93**, conj **3** | kept 0, negated 96, conj 0 |
| orientation census, m | kept 0, negated 96, conj 0 | kept 0, negated 96, conj 0 |
| sorted roots differing from the um arm | **96 of 96** (worst `\|d ky·s\|` 192.325) | **0 of 96** (worst 9.4e-12) |
| `mode_match` `T00`, um / nm / m | 0.738986606108 / **0.738986551923** / 0.738986606108 | 0.738986606108 / 0.738986606108 / 0.738986606108 |
| worst `mode_match` `dT00` across unit systems | **5.41848e-08** | **2.22045e-15** |

The nanometre band was **5.2x wider in physical terms**, which is what put 3 of
96 modes on a different root; once sorted, that reorders the whole spectrum,
which is why the sorted count is 96 and the census count is 3. Both are
reported because they answer different questions.

**Per arm, and it is not an arithmetic defect.** The comparison is a floor, not
a reduction, so it reproduces identically wherever it is run:

| arm | PRE: sorted roots differing | PRE: worst `mode_match` `dT00` | POST: differing | POST: `dT00` |
|---|---|---|---|---|
| win / Haswell / t1 | **96 / 96** | **5.41848e-08** | **0** | 2.22045e-15 |
| win / Nehalem / t1 | **96 / 96** | **5.41848e-08** | **0** | 2.66454e-15 |
| win / Katmai / t1 | **96 / 96** | **5.41848e-08** | **0** | 3.88578e-15 |
| win / Sandybridge / t1 | **96 / 96** | **5.41848e-08** | **0** | 4.21885e-15 |
| win / Haswell / t4 | (not run PRE) | — | **0** | 2.22045e-15 |
| wsl / Haswell / t1 | (not run PRE) | — | **0** | 2.77556e-15 |
| wsl / Nehalem / t1 | (not run PRE) | — | **0** | 2.33147e-15 |
| wsl / Katmai / t1 | (not run PRE) | — | **0** | 3.88578e-15 |
| wsl / Sandybridge / t1 | (not run PRE) | — | **0** | 5.88418e-15 |

PRE reads the same 5.41848e-08 to six digits on all four Windows kernels — the
defect is a decision about a threshold, and no BLAS kernel moves it. POST is
zero differing roots on all NINE arms, with the residual `dT00` at the
1e-15 level that two unit systems' round-off gives.

### 5.3 The fix, and how far it had to be plumbed

`cut_band(z, *, k0=None, ...)` returns `band * max(max|z|, |k0|)`, and the
literal is gone. `k0` is threaded from every production site that has it:

* `eme_2d._ky_forward(lam, qz2, k0)` ← `_wv` ← `cell_smatrix` ← `dispersion` ←
  `layer_modes` (which has `k0`), and ← `_global_lateral_nullspace` ←
  `mode_field`;
* `eme_diffraction.mode_match` (has `k0`);
* `eme_2d_vector._strip_split_forward(ky, k0)` ← `strip_vector_modes` (has
  `k0`).

`cell_smatrix`, `dispersion` and `mode_field` are **public** (exported from
`lumenairy.elements.eme`), so `k0` is a keyword-only parameter with a default
rather than a new positional: an existing call keeps working and gets `max|z|`
alone, which is **also unit-invariant** — it scales as 1/length exactly like
`|Im z|`. What is not allowed on any path, and what round 2 removed, is a
dimensioned literal.

**Nothing else moved.** On every ordinary strip `max|ky|` already exceeds `k0`,
so `max(max|z|, k0)` and `max(max|z|, 1.0)` are the identical number and the
5.45.1 fixtures are bit-identical — §7.

### 5.4 The gate that pinned the defect

`test_fix_eme_branch_cut.py::test_the_band_is_relative_to_the_spectrum_and_floored_at_one`
asserted the literal. It is now
`..._floored_at_k0` and states the property as a DECISION rather than as two
readings: rescaling the whole problem (lengths × s, wavenumbers / s) must scale
the band by exactly 1/s, checked over twelve decades in both directions and with
and without `k0`. A dimensioned literal anywhere in the expression breaks that
and nothing else does.

---

## 6. D9 and the restatements

### 6.1 D9 — the near-cutoff gate was sample-scoped and its bar was crossed

`test_near_cutoff_channel_count_is_stable_over_the_ladder` claimed a property of
*"the whole near-cutoff ladder"*, asserted it on `m = 0` alone, and barred the
worst lossless closure at `1e-6`. The build's own report records the same ladder
reaching **1.2716e-06 on Windows / PRESCOTT → Katmai / 1 thread**.

**Reproduced independently, over `m` ∈ {0,1,2} and TEN arms**
(`r6_cutoff_family.py` → `r9_cutoff_envelope.py`):

| arm | worst closure | counts `m` = 0/1/2 |
|---|---|---|
| win / Haswell / t1 | 1.9655e-07 | {3}/{3}/{3} |
| win / Haswell / t4 | 6.1976e-08 | {3}/{3}/{3} |
| win / Nehalem / t1 | 2.9240e-07 | {3}/{3}/{3} |
| win / **Katmai** / t1 | **1.271613e-06** | {3}/{3}/{3} |
| win / Sandybridge / t1 | 3.6490e-08 | {3}/{3}/{3} |
| wsl / Haswell / t1 | 1.0818e-07 | {3}/{3}/{3} |
| wsl / Haswell / t4 | 4.3513e-07 | {3}/{3}/{3} |
| wsl / Nehalem / t1 | 2.7766e-07 | {3}/{3}/{3} |
| wsl / Katmai / t1 | 1.7132e-07 | {3}/{3}/{3} |
| wsl / Sandybridge / t1 | 7.7303e-08 | {3}/{3}/{3} |

Cross-arm spread **34.9x**; the envelope is the build's own published Katmai
number. The gate is now parametrized over `m` and its bar is
`_CUTOFF_LADDER_BAR = 1e-5`, **7.86x (0.90 decades)** above the envelope, where
`1e-6` sat **0.786x below** it. The CHANNEL COUNT claim — an integer, one number
per `m` — is unchanged and holds on all ten arms.

**The ladder's own lower bound, which the gate did not state.** `channel_core`
keeps a channel only while `Re qn > _BOR_CHANNEL_REAL_FLOOR` (1e-6), and the
ladder puts its cutoff order at `qn = n sqrt(delta)` — so a deep enough rung
puts that order under the CHANNEL gate and the count legitimately falls. That is
the channel gate working, not the orientation band failing. Measured over the
same ten arms: the count reads `{2, 3}` once rungs down to 1x the floor are
included and `{3}` everywhere from **3x** up. The gate now derives its stopping
rung from the library's constant (`_CUTOFF_LADDER_FLOOR_MULT = 10.0`, 3.3x
inside the measured onset) instead of the literal `range(8, 21)` that agreed
with it by accident.

**The pre-fix contrast is premise-gated.** A new gate re-derives BOTH
classifications from the same spectrum and asserts that every mode the two rules
disagree about is physically propagating; that they disagree at all is the
premise, and an arm where they do not SKIPS with the reading.

### 6.2 The band's SIGNAL side is 0.38 decades, not 0.98

The build's table quotes 9.4570e-05 / 9.4570e-08 and 3.98 / 0.98 decades. Those
are **maxima** of the lossy population. The statistic that decides is the
**minimum** — the closest approach to the band from above — because the band
must not reach ANY genuinely lossy mode. (The build's own gate already read the
minimum; the number written beside it was the maximum.) Re-measured,
`r8_band_sides.py`: MIN `sigma` = **3.7752e-05** at `Im(n)` = 1e-3 and
**3.7752e-08** at 1e-6, i.e. **3.58 and 0.58 decades**. The docstring table in
`_orient.orient_band_scale` now says so, and labels each row MIN or max.

> **CORRECTION, 2026-09-12 (round 3, verification GAP 6).** As first written
> this paragraph read *"MIN `sigma` = 2.3820e-05 ... and **2.3820e-08** ...
> i.e. **3.38 and 0.38 decades**"* and attributed those numbers to
> `r8_band_sides.py`. That probe's own JSON
> (`validation/probe_fix_bor_round2/r8_band_sides_win_Haswell_t1.json`,
> `summary.signal`) reports minima of **3.7752e-05** and **3.7752e-08**, i.e.
> **3.577 and 0.577 decades** — which is what the text above now says, and what
> `_orient.orient_band_scale`'s docstring has said all along. 2.3820e-08 is the
> ROUND-1 verification's own number, measured on ITS battery
> (`VERIFY_BOR_MULTILAYER_GUARDS_2026_09_12.md`, §5 table), not this probe's.
> **The conclusion is unchanged**: the minimum over the union of the two
> batteries is the smaller of the two, so 2.3820e-08 / 0.38 decades remains the
> figure the band must carry, and the shipped code comment
> (`_orient.py`, `orient_band_scale`) already attributes both numbers to their
> own probes correctly. Only this paragraph's prose conflated them. Recorded
> because it is the right-conclusion-wrong-numbers shape
> `docs/TESTING_STANDARDS.md` names as the most dangerous: it reads as
> authoritative, the artefact that matters is right, and nothing catches it
> except re-reading the JSON.

### 6.3 The `k0` floor binds on zero layers

`orient_band_scale` returns `max(max|q|, k0)`, and `max|q|` is dominated by the
largest transverse eigenvalue `~ N / Rbig`, which exceeds `k0` on any grid that
resolves the wavelength. Over the layers measured here — including an nm-unit
arm six orders of magnitude away — the floor binds on **zero** of them. It is a
**unit-safety floor, not a measured bar**, and the constant's comment now says
so and records why it is kept anyway: the alternative is a dimensioned literal,
which is a defect whether or not it is reachable here — the EME peer shipped
exactly that literal in this same wave (D13).

### 6.4 The SEM warn edge's binding ordinary margin is 1.003x, not 9.77x

`_BOR_SLIVER_BAND_FRAC = 1e-4` was fixed from a census whose binding ordinary
geometry was a 256-slice taper staircase at `w_min_union_frac` = 9.766e-04,
quoted as a 9.77x margin. That is a property of that census's two families.
The verification found a third it did not sweep, and round 2 measured it
(`validation/probe_fix_bor_round2/r7_sem_hp_census.py`, **30 rows**, degrees
6/8/12 × grading on/off × `elements_per_segment` 1..32):

**hp refinement with GRADING on, on an ordinary two-layer ring pair.**
`BORStack(elements_per_segment=k, grade=True)` splits every segment interval
into `k` Chebyshev–Lobatto graded sub-elements, so the narrowest sub-cell of an
interval of width `W` is `~ W·pi²/(4k²)` — it shrinks as `1/k²`. And because
the `+-1` enrichment window makes the interval between two DIFFERENT layers'
walls appear in both meshes, every sub-cell inside that interval is attributed
to the union.

| `elements_per_segment` | 1 | 4 | 8 | 16 | **32** |
|---|---|---|---|---|---|
| `w_min_union / Rbig` | 4.1667e-02 | 6.1019e-03 | 1.5858e-03 | 4.0031e-04 | **1.003182e-04** |
| margin to the 1e-4 edge | 417x | 61x | 15.9x | 4.0x | **1.003x** |

The quantity is **mesh arithmetic**: it reads 1.003182e-04 identically at
degrees 3 / 4 / 6 / 8 / 12 and at `N` = 40 / 60 / 80 / 120, so it is
kernel-exact and no arm can move it. **0 of 30 rows warned and 0 would be
refused**, so the contract is still silent on ordinary geometry — but only by
0.3 %, and a `k = 64` sweep, or the same `k` on a wider interval, lands inside
the band and warns.

**The edge is documented, not moved, and the reason is in the constant's
comment.** The union attribution is deliberately LOOSE — it counts any cell
inside a cross-layer interval, not only a cell whose own two breakpoints are
walls — and it has to be, because a genuine near-coincident-wall sliver is
SUBDIVIDED by the same hp knob, so a strict attribution would stop refusing the
population the contract exists for. Narrowing the edge to 1e-5 instead would
buy about two hp rungs at the cost of a decade of the warn band, which is where
the delta ladder's `delta/Rbig = 1e-5` rung sits with a measured move factor of
1.22 .. 3.2 — a real degradation the band should keep speaking about. The cost
of leaving it is a `UserWarning` on a deep graded hp sweep, never a refusal
(`warn_manufactured` carries no spectral conjunct and cannot escalate).

Pinned by `test_fix_bor_guards_round2.py::
test_the_sem_warn_edge_s_binding_ordinary_margin_is_graded_hp_refinement`,
which asserts the DECISION (ordinary geometry at `k = 32` is not warned), the
`1/k²` LAW (so a change to the grading recipe forces a re-measurement), and
that the constant's stated margin has not drifted from the measurement —
nothing that the library is entitled to move.

**The ordinary `q_excess` ceiling moves with the same family**: 2458.3 at
`k = 32`, degree 6, against the build's quoted non-taper maximum of 164.4 and
the verification's 2215.

### 6.5 The Q bar's "ordinary" margin protects nothing

`_sem_contract.verdict` refuses on a CONJUNCTION:
`w_min_union_frac < _BOR_MIN_ELEM_FRAC` **and** `q_excess > _BOR_Q_EXCESS`.
Ordinary geometry has no cross-layer cell at all on the families the build
measured (`w_min_union_frac = inf`), so it can never be refused **whatever its
`q_excess`** — the build's "0.95 decades above the worst ordinary geometry
measured" describes a comparison the conjunction cannot reach.

The bar's only operative role is the other direction: it **suppresses**
refusals on manufactured cells. Its failure mode is a MISS, not a false
positive, and the honest statement of its margin is the one-sided one — 1.20
decades (15.9x) below the mildest rung the ladder measured as damaging. The
constant's comment now says this, and keeps the ordinary readings only because
the number should be right where it is quoted (§6.4 moves it to 2458.3).

Neither constant changes value.

### 6.6 D4 — `warn_own` blamed a layer that asked for nothing

`_sem_contract.verdict`'s `warn_own` branch fired on `w_min_frac` — the
narrowest cell of the POST-WINDOW mesh, whoever asked for it — and its message
told the caller that *"the LAYER'S OWN segment list asked for"* that cell. For
the NEIGHBOUR of a liner that is false: its own segment list is a single
full-radius entry, and it has the cell only because the `+-1` enrichment window
put it there. Measured pre-fix: two messages, blaming layers `[0, 1]`, where
layer 1 is `add_layer(0.5, eps=1.21)`. The stated reason for never REFUSING a
`warn_own` cell — *"you prescribed the geometry and the library does not
overrule it"* — does not hold for that layer.

`measure_layer` now also reports `w_min_own` / `w_min_own_frac`: the narrowest
element BOTH of whose enclosing walls THIS layer's own segment list asked for.
`verdict`'s own-arm reads it, `w_min_frac` is still measured and reported for
the census, and the shipped liner gate now asserts the ATTRIBUTION (which
layers may carry the verdict) rather than restating it — the one gate in
`test_fix_bor_multilayer_guards.py` the verification found restating.

This was the fourth strict xfail; it is now
`test_verify_bor_multilayer_guards.py::
test_warn_own_is_only_reported_for_the_layer_that_prescribed_the_cell`.

**And a coverage hole the first version of this fix opened, caught by
adversarially probing it.** A layer's segment list holds only its INTERIOR
walls (`bor_stack`: `rs for rs, _t in segs if rs < Rbig`), because every
layer's list ends at `Rbig` and starts at the axis. Reading ownership from
interior walls alone therefore left a liner hard against EITHER domain end with
no owner at all — and the message that exists to tell that caller stopped
firing. Measured on the first version: a `1e-6`-of-`Rbig` liner at the outer
wall and at the axis drove `|q|max / (n_max k0)` to **2.521e+05** and
**8.895e+05** — 1.4 and 1.9 decades past the spectral screen — and both read
`ok`, where the pre-round-2 code said `warn_own`. The two domain ends now count
for every layer, and the three positions behave identically over a width
ladder: all `warn_own` at `1e-7` of `Rbig`, all `ok` at `1e-4`. Pinned by
`test_fix_bor_guards_round2.py::
test_a_caller_prescribed_liner_is_warned_wherever_it_sits_in_r`.

> **RESTATEMENT, 2026-09-12 (round 3).** The literal claim above — all
> `warn_own` at `1e-7`, all `ok` at `1e-4` — is true and was re-measured by the
> round-2 verification. The GENERALISATION of it, *"the three positions behave
> identically over a width ladder"*, was **REFUTED as a ladder claim** by that
> verification and independently re-measured here
> (`validation/probe_fix_bor_round3/g3_sem_ladder.py`, 9 rungs from `1e-9` to
> `1e-5` of `Rbig` x 3 positions): on the round-2 tree the three positions
> disagreed at **4 of the 9 rungs**, identically on both builds —
>
> * `1e-9`, `3e-9`, `1e-8`: the AXIS liner's `q_excess` goes **non-finite** and
>   `verdict` read a non-finite ratio as NOT hot, so the axis read `ok` while
>   the interior and outer wall read `warn_own` (verification GAP 3), and on
>   Sandybridge the axis ratio is finite (8.89487e+07) so the axis read
>   `warn_own` there — a **kernel-dependent verdict**;
> * `1e-6`, the edge itself: the axis liner's walls (`0`, `w`) difference to
>   `1.000000000000e-06` EXACTLY and the other two (`Rbig - w`, `Rbig` and
>   `6`, `6 + w`) to `9.999999999917e-07`, 38,968 ULP lower, so a STRICT `<`
>   against `_BOR_MIN_ELEM_FRAC` decided them oppositely (verification GAP 4).
>
> **Round 3 closes both**, and the generalised claim is now true as stated:
> `_sem_contract.verdict` reads a non-finite `q_excess` that was FORMED from a
> spectrum as HOT, and its three width comparisons go through `_below`, which
> closes the tie at the bar with the library's 16-ULP deadband. Re-measured
> after the fix: **0 of the 9 rungs disagree**, on every arm of the round-3
> matrix. The ladder claim is pinned by
> `test_fix_bor_guards_round2.py::
> test_a_caller_prescribed_liner_is_warned_wherever_it_sits_in_r` (extended to
> the full ladder) and by `test_verify_bor_guards_round2.py::
> test_the_warn_own_edge_is_not_decided_by_the_representation_of_the_width` and
> `::test_a_non_finite_q_excess_is_not_a_benign_sem_verdict`, both of which
> were strict xfails and are now live gates.

### 6.7 `.test_durations`

Three defects, all fixed, by re-measuring the five files this round touched on
a quiet box (`r12_durations.py`, Windows / Haswell / 1 thread, 101 tests in
655.5 s) and replacing every key belonging to them:

| defect | before | after |
|---|---|---|
| the STALE renamed key `test_bor_solve.py::test_structured_stack_energy_floor_nodal` | present at 6.61 s, a node id that no longer exists | **gone**; the renamed gate carries its own 2.80 s |
| `test_mode_count_is_build_independent_under_an_infinitesimal_loss` | **167.41 s** — the PRE-narrowing cost the file's own comment says was designed away, and the only entry in either new file over the 60 s shard cap | **13.01 s** measured, 12.9x smaller |
| missing / renamed node ids | 75 keys across the five files, 0 for the new file | **101**, one per collected test |
| entries over the 60 s shard cap | 1 (the fiction above) | **none**; the slowest is `test_ordinary_geometry_census[mesh-12]` at 51.61 s, 86 % of the cap |

`.test_durations` is sorted and JSON-validated after the splice (12,880
entries). 14 of the 101 ids are written at pytest's own 5 ms hidden-duration
cutoff rather than at a printed measurement — pytest does not print a duration
below that unless verbosity reaches 2, and the cutoff is an UPPER bound on those
ids' true cost, which is the safe direction for a shard balancer. The
substitution is counted and printed by the probe rather than hidden.

---

## 7. Bit-identity

The contract round 2 must keep is the one the verification established: **only
the rows the new screen REFUSES may change.** The verification's own battery is
re-hashed rather than a new one written, because that is what makes the claim
checkable rather than asserted — 58 BOR fixtures (both bases × `m` = 0/1/2/5 ×
`k0` = 0.8/2.0/3.5, ring-grating pairs, three-layer stacks with a wall-free
spacer, lossy layers at `Im(n)` = 1e-1/1e-3/1e-6, the uniform-equals-region and
uniform-equals-neighbour coincidences, thin rings, taper staircases at 4..64
slices, an anisotropic layer, hp-refined and graded meshes, and an nm-unit
fixture six orders of magnitude away) and 15 EME fixtures, hashed to the SHA-256
of the exact IEEE-754 bytes of the answer.

The fixture bodies are imported verbatim from
`validation/probe_verify_bor_guards`; only the tree check is round 2's own,
because the verification's `_vh.require_tree` pins its author's two clone names.
`lumenairy.__file__` is printed and checked in every run.

| build | arm | BOR identical | BOR moved | newly refused | EME identical | EME moved |
|---|---|---|---|---|---|---|
| Windows py3.14 / numpy 2.4.4 | Haswell, 1 thread | **58 / 58** | **0** | **0** | **15 / 15** | **0** |
| WSL py3.12 / numpy 2.4.6 | Haswell, 1 thread | **58 / 58** | **0** | **0** | **15 / 15** | **0** |

**Nothing moved at all**, on either build — including the two EME fixtures the
verification's own pre→post comparison legitimately moved (`striplossy_im1e-30`
and `layermodes_96_im1e-30`), because those moved at the 5.45.1 branch point and
round 2 does not touch them again.

**Why the EME change moves nothing here.** The `k0` floor and the literal 1.0
floor give the IDENTICAL number on every spectrum whose top already exceeds
`|k0|`, and that is every strip in the battery: measured over its 24
`(Nx, k0, Im eps)` combinations, `max|ky|` runs **217.6 .. 435.3** against `k0`
of **62.83 .. 125.7**, with a worst ratio of **3.46x** — never below 1. The
floor decides only on a sub-`k0` spectrum, which is exactly the population D13
is about and which no fixture in the 5.45.1 battery contains (it is why the
defect survived the build's own bit-identity leg). The nm-unit fixtures in the
battery are BOR ones, on a path whose floor did not change.

**Why the BOR change moves nothing here.** The passivity screen raises or
returns; it never alters a returned number. No fixture in the battery is on the
legacy nodal path with a geometry the screen refuses — `_vh.bor_fixtures()`
builds `BORStack` (fd / sem), which the screen does not touch.

**And a second, sharper check on the population the screen DOES touch.** The
160-solve nodal/staggered census of §4 was re-run on this tree with the guard
disarmed and compared field by field against the run taken before any library
change: **0 differing fields over 160 rows × 5 quantities** (`n_channels`,
`excess`, `deficit`, `max_qn`, `ceiling_excess`). The two new per-layer facts
`build_layer` records (`min_rel_im_eps`, and `max_rel_im_eps`'s denominator
moving from `max|Re eps|` to `max|eps|`) are read by the screen and by nothing
else, and the census proves it: every number the legacy nodal cascade returns
is the number it returned before. What changed is which of them the caller is
allowed to receive.

## 8. Runs

File set (20 files), every arm:

```
tests/unit/test_audit_bor_grazing_cutoff.py   tests/unit/test_eme_2d.py
tests/unit/test_audit_p1_bor_flux.py          tests/unit/test_eme_2d_vector.py
tests/unit/test_audit_v5_24_2_b2_bor_exports.py
tests/unit/test_audit_w6_bor.py               tests/unit/test_eme_census_determinacy.py
tests/unit/test_audit_w6_eme.py               tests/unit/test_eme_diffraction.py
tests/unit/test_bor_anisotropic.py            tests/unit/test_eme_jax_modes.py
tests/unit/test_bor_sem.py                    tests/unit/test_fix_bor_guards_round2.py
tests/unit/test_bor_sem_jax.py                tests/unit/test_fix_bor_multilayer_guards.py
tests/unit/test_bor_solve.py                  tests/unit/test_fix_eme_branch_cut.py
tests/unit/test_niche_audit_w6_bor.py         tests/unit/test_niche_audit_w6_eme.py
tests/unit/test_verify_bor_multilayer_guards.py
```

Every arm pins `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS`
on the command line, and records `lumenairy.__file__` and the kernel that
ACTUALLY loaded (read back from `threadpoolctl`, never inferred from the
`OPENBLAS_CORETYPE` request) into the log before pytest runs.
Scripts: `validation/probe_fix_bor_round2/runs/matrix_win.sh`, `matrix_wsl.sh`.

| # | build | requested | **loaded** | thr | passed | failed | errors | skipped | time |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Windows py3.14 | HASWELL | **Haswell** | 1 | **389** | 0 | 0 | 0 | 1709.65 s |
| 2 | Windows py3.14 | NEHALEM | **Nehalem** | 1 | **389** | 0 | 0 | 0 | 2188.87 s |
| 3 | Windows py3.14 | PRESCOTT | **Katmai** | 1 | **389** | 0 | 0 | 0 | 2337.61 s |
| 4 | Windows py3.14 | SANDYBRIDGE | **Sandybridge** | 1 | **389** | 0 | 0 | 0 | 2122.15 s |
| 5 | Windows py3.14 | HASWELL | **Haswell** | 4 | **389** | 0 | 0 | 0 | 1652.95 s |
| 6 | WSL py3.12 | HASWELL | **Haswell** | 1 | **389** | 0 | 0 | 0 | 1577.25 s |
| 7 | WSL py3.12 | NEHALEM | **Nehalem** | 1 | **389** | 0 | 0 | 0 | 2067.89 s |
| 8 | WSL py3.12 | PRESCOTT | **Katmai** | 1 | **389** | 0 | 0 | 0 | 2281.49 s |
| 9 | WSL py3.12 | SANDYBRIDGE | **Sandybridge** | 1 | **389** | 0 | 0 | 0 | 1993.53 s |
| 10 | WSL py3.12 | HASWELL | **Haswell** | 4 | **389** | 0 | 0 | 0 | 1414.62 s |

**Zero failures and zero errors on every one of the ten**, and every tail
carried a real summary line.  The count is 389 on all ten: 368 is the set the
build ran, plus the 11 gates of `test_verify_bor_multilayer_guards.py` (the four
that were strict xfails, now passing, plus the two new complements and the five
the verification already had passing), plus the 13 of
`test_fix_bor_guards_round2.py`, minus the arithmetic of the gates this round
re-parametrized (the near-cutoff ladder became three, and the two renamed gates
stayed one apiece).

Plus, on Windows:

| run | requested | loaded | thr | result |
|---|---|---|---|---|
| the three JAX-guarded BOR/EME files (`test_bor_sem_jax`, `test_v5_20_11_bor_jax`, `test_eme_jax_modes`) | HASWELL | Haswell | 1 | **22 passed, 0 skipped**, 68.2 s |
| the release gate's census / walker / dispatcher / public-API sweep | *(unset)* | Haswell | 1 | **1288 passed, 12 skipped**, 204.7 s |

The sweep's 12 skips are the same ones the verification recorded (D7): eight are
the changelog-integrity walkers, which key on the top VERSIONED CHANGELOG block
(`## [5.45.0]`). Round 2's work is under `## [Unreleased]` and — per the task
that commissioned it — does NOT cut a `## [5.45.1]` header or bump
`__version__`, so those walkers still have nothing of this branch's to verify.
That is unchanged, deliberate, and belongs to the release commit.

**ruff**, under the WSL venv, naming the probe directory explicitly (which
overrides `pyproject.toml`'s own `extend-exclude = ["validation/", ...]`, the
reason the rest of `validation/` is not clean by design):

```
wsl -e bash -lc 'cd /mnt/c/tmp/lum_bor2 && ~/lumvenv/bin/ruff check \
                 lumenairy/ tests/ validation/probe_fix_bor_round2/'
-> All checks passed!
```

## 9. What could not be measured

* **The CI's actual ZEN kernel.** `OPENBLAS_CORETYPE=ZEN` silently returns
  Haswell on this host's OpenBLAS 0.3.31 and `SKYLAKEX` dies on the first LAPACK
  call (Ryzen 9 5950X, Zen 3, no AVX-512). The ladders here cover Haswell,
  Nehalem, Katmai and Sandybridge across two builds; the EPYC 9V74 / 7763 pool
  the CI actually draws from is not reproduced, which is why every gate in this
  round asserts an invariant unconditionally and premise-gates every claim that
  a pathology is present.
* **The two SET-WRONG rows the union of both detectors still misses** (§4.4).
  They are rows whose spurious channels carry no measurable power AND stay under
  their own index ceiling. Nothing in this round detects them, and nothing here
  bounds how wrong they are.
* **A WIDER lower edge for the energy bar than the 160-solve census gives.**
  `validation/probe_fix_bor_round2/r5_healthy_ceiling.py` was written to sweep
  720 solves of the accurate family (`m` 0..5 x `N` 80/120/200 x
  `Rbig/lambda` 0.25..2.0 x three `k0` x two index contrasts x two stack
  shapes) purely to push the 2.714107e-07 ceiling of §4.3 down or confirm it.
  It ran for over an hour against the ten-arm test matrix and was stopped so
  the matrix could finish; the probe ships and the run does not. The edge
  therefore rests on the 160-solve census (18 accurate-family rows) and on
  `test_the_nodal_energy_bar_has_a_two_sided_gap_on_the_population_it_owns`,
  which re-measures a 36-solve version of it on whatever arm runs the suite.
* **A production predicate for the closed-form channel count.** The Bessel-zero
  oracle exists only for a UNIFORM PEC-walled half-space and the verification
  measured its own `m = 0` bookkeeping off by one and its large-cell misses, so
  it is used as a census oracle and cross-check, not as a guard. Whether a
  count-based refusal could be made general — over ring gratings and graded
  profiles — is not answered here.
* **The `k0` floor on the BOR path** (§6.3). Unreachable through `BORStack` on
  any grid that resolves the wavelength; the choice cannot be exercised, only
  argued.
* **The Q bar's miss region at its crossing.** Inherited unchanged from the
  verification's §5.3: the `1/(n_max · Rbig/lambda)` law is measured over a 12x
  range, but the case where `q_excess` actually falls below 1e4 at the
  refusal's geometric edge needs an optically large SEM solve that exceeds the
  time budget here as it did there.
* **PML and the anisotropic Class-C populations.** Unchanged: PML is unreachable
  on the staggered path by construction, and the contract makes no claim there.
* **An external accuracy oracle for the SEM answer.** Unchanged from the build
  and the verification.
* **Whether the ceiling conjunct would hold on a LOSSY half-space.** It is
  deliberately gated on the half-space being lossless, because that is where
  the Rayleigh bound is airtight: with a real symmetric `eps k0^2 + D` the
  spectrum is real and `q^2 <= max(eps) k0^2` exactly, while with a complex
  `eps` the operator is complex-symmetric and the same statement about `Re q`
  is only approximate. Every half-space of the 160-solve census is lossless, so
  nothing measured is given up — but the lossy case is not measured either, and
  a stack whose SUBSTRATE absorbs keeps the energy screen alone.
* **`.test_durations` under a quiet box.** The re-splice below was taken on a
  box running nothing else, but a single timing run is one sample;
  `pytest-split` uses these to BALANCE shards, not to decide anything, so the
  accuracy that matters is the absence of a fictional 167 s entry rather than
  the last digit of any real one.
