# Branch cut, round 3: the on-cut flip is `-r`, not `conj(r)`

**Date** 2026-09-11 · **Tree** `59105d6` (the 5.45.0 release commit, untagged) ·
**Trigger** the release CI matrix, RED · **Scope** `lumenairy/elements/rcwa/_core.py`
(`_sqrt_decay`, one line) and seven test files

---

## 0. Terms

**`lam^2`** — a layer's modal eigenvalue, the output of `eig` on the RCWA/PMM
operator product `P Q`.

**`lam`** — its modal decay constant, `sqrt(lam^2)` on a chosen branch. It
drives the layer propagator `X = exp(-lam k0 L)`.

**The principal branch** — the square root with `Re(result) >= 0`. Choosing it
guarantees `|X| <= 1` for a forward thickness `L > 0`, which is the
unconditional stability of the S-matrix method.

**On the cut** — a PROPAGATING mode of a LOSSLESS layer has `lam^2` exactly real
and negative, which is the principal square root's branch cut. There the two
roots are `+i|kz|` (**OUTGOING** — the branch the half-space region modes are
built on, and the one the S-matrix recursion requires of a layer's FORWARD set)
and `-i|kz|` (**INCOMING**). Which one `sqrt` returns is decided by the sign of
`Im(lam^2)`, and for an `eig` output that sign is the eigensolver's backward
error, not physics.

**The band** — round 1's fix. A mode counts as on the cut when
`|Re(r)| <= _CUT_BAND_REL * max(max|r|, 1)` with `_CUT_BAND_REL = 1e-8`, where
`r = sqrt(lam^2)` on the principal branch and the scale is the spectrum's
largest root. Where that fires and `Im(r) < 0`, the root is FLIPPED to the
outgoing one.

**The flip** — what "flipped to the outgoing one" means numerically. Rounds 1
and 2 used `conj(r)`. Round 3 uses `-r`. This document is about that choice.

**Holomorphic** — complex-differentiable. Reverse-mode AD propagates a cotangent
correctly through a holomorphic function; through a non-holomorphic one
(`conj`, `real`, `imag`, `abs`) it applies the real-linear rule instead, which
for `conj` conjugates the cotangent.

**AD / FD** — automatic differentiation (`jax.grad`) / central finite
differences, `(f(x+h) - f(x-h)) / 2h`.

---

## 1. What CI found

Two gates failed on the JAX job (py3.12, ubuntu, AMD EPYC):

| test | reading | gate |
|---|---|---|
| `test_niche_audit_w9_eig_vjp.py::test_pmm2d_near_normal_angle_gradient_improved` | `0.00165` | `1e-4 * 4.67e-4` |
| `test_v5_14_0_pmm2d_autodiff.py::test_gate_angle_grad_at_normal_offcenter_is_genuine` | `0.0166` | `1e-4 * 0.0732` |

Both reproduce on Windows py3.14 / jax 0.11.0 / numpy 2.4.4, and both PASS on
the pre-round-1 tree `48c8747`. Neither file was in the local JAX batteries the
branch-cut rounds ran, which matched `test_pmm*` / `test_rcwa*` names only.

---

## 2. The mechanism

### 2.1 `conj` is not holomorphic

`conj(r) = -r + 2 Re(r)`. The `-r` part is holomorphic; the `2 Re(r)` part is
not. Reverse-mode AD through `conj` conjugates the cotangent, so every flipped
mode contributes a wrongly-conjugated term to the angle gradient. This alone
accounts for a wrong `jax.grad` with an unchanged forward value — and it was
the whole of the diagnosis the repair started from.

### 2.2 It is also wrong FORWARD, and by more

`-r` is the EXACT involution the S-matrix assembly is invariant under. Flipping
an on-cut mode's root re-labels it from the layer's BACKWARD mode set to its
FORWARD set. Under `lam -> -lam` that re-labelling is exact: the propagator pair
`exp(-lam k0 L)` / `exp(+lam k0 L)` swaps, and `V = Q W diag(1/lam)` changes
sign with it, which is precisely the `(W, -V)` backward partner. The assembled
S-matrix does not move.

`conj(r)` is that involution PLUS a `2 Re(r)` perturbation. Round 1 recorded
`Re(r)` as `~1e-16`, "the eigensolver's rounding-level real part". That is the
size of `Re(r)` for a mode sitting AT the cut, but the BAND deliberately admits
modes out to `1e-8 * scale`, and it is the band's membership that decides the
flip. **Measured `Re(r)` of a flipped mode over a `theta` sweep of `+/-4e-06` rad
on the near-normal hybrid fixture: `4.4752e-16 .. 2.8530e-08`** — eight decades,
topping out at the band edge.

Worse, **the flip SET moves with the incidence angle**: `Re(r)` is backward
error, so which modes fall inside the band jitters. Over 41 thetas spanning
`+/-4e-06` rad the sweep visits **9 distinct flip masks**. Each entry or exit
changes the answer by the `2 Re(r)` the involution is broken by, so the forward
answer is DISCONTINUOUS.

### 2.3 The measurement

`f(theta) = sum(T)` of the hybrid 2-D PMM (`_P = 0.6 um`, `_WL = 0.55 um`,
`_DEP = 0.25 um`, off-centre pillar `(0.2 P, 0.6 P)`, `degree = 5`,
`n_orders = 2`, TE), 41 points over `theta` in `+/-4e-06` rad. `dtheta = 2e-07`,
so a smooth response should step by `|f'| dtheta = 1.2558e-08`.

| flip rule | worst `\|f − linear fit\|` | worst step | step / smooth |
|---|---|---|---|
| pre-round-1 (`Re(r) == 0` pin, never fires) | `6.9079e-12` | `1.2559e-08` | **1.0001** |
| round 1/2 `conj(r)` | `1.5935e-07` | `1.7287e-07` | **13.766** |
| round 3 `-r` | `6.9077e-12` | `1.2559e-08` | **1.0001** |

`conj` makes the forward answer discontinuous at the **1.6e-07** level — five
decades above the `2 Re(r) ~ 1e-16` round 1 assumed. `-r` reproduces the
pre-round-1 forward answer to `6.9e-12`, which is the direct evidence that the
`lam -> -lam` re-labelling is invariant.

### 2.4 Why the gates read what they read

The two failing gates take FD at `h = 1e-6`. A band-level forward step of
`~2e-09` divided by `2h = 2e-06` is `~1e-03` of spurious derivative. The
h-ladder shows exactly that — the FD reference itself was contaminated, and only
below `h = 1e-5`:

| rule | `h=3e-4` | `1e-4` | `3e-5` | `1e-5` | `3e-6` | `1e-6` | `3e-7` | `1e-7` |
|---|---|---|---|---|---|---|---|---|
| pre-round-1 | 6.2788e-2 | 6.2788e-2 | 6.2788e-2 | 6.2788e-2 | 6.2788e-2 | 6.2788e-2 | 6.2788e-2 | 6.2788e-2 |
| `conj(r)` | 6.2788e-2 | 6.2788e-2 | 6.2788e-2 | 6.2788e-2 | 6.7686e-2 | 7.3230e-2 | 5.2395e-2 | **−3.6608e-1** |
| `-r` | 6.2788e-2 | 6.2788e-2 | 6.2788e-2 | 6.2788e-2 | 6.2788e-2 | 6.2788e-2 | 6.2788e-2 | 6.2788e-2 |

The converged derivative (`h >= 1e-5`) is the SAME under every rule. Under
`conj` the FD diverges like `1/h` below that — the signature of a step, not of a
kink or of noise.

---

## 3. The candidates, measured

All on Windows py3.14 / `OPENBLAS_CORETYPE=HASWELL`, one thread; `rel` is
`|AD − FD| / |FD|` with FD central at `h = 1e-6`, i.e. the gates' own reference.

| flip rule | w9 `sum(R)` @ θ=0 | v514 `sum(T)` @ θ=0 | v514 @ θ=0.3 |
|---|---|---|---|
| pre-round-1 (`==0` pin) | 2.292e-06 | 1.452e-08 | 7.912e-09 |
| **round 1/2 `conj(r)`** (shipped) | **6.176e+00** | **7.103e-01** | 7.912e-09 |
| **A. `-r`** | **1.838e-06** | **5.087e-10** | 7.912e-09 |
| B. `-r` then `where(flip, 1j*imag(r), r)` | 1.808e+01 | 3.824e-01 | 7.912e-09 |
| C. `conj` value, `-r` derivative (`stop_gradient`) | 2.342e+00 | 1.426e-01 | 7.912e-09 |

**A ships.** B is the mitigation the repair brief suggested for the `|X| <= 1`
price; it is REJECTED because `imag()` is itself non-holomorphic — it scores
worse than `conj` on both gates. C keeps the round-2 forward value bit-for-bit
and repairs only the derivative; it is REJECTED because the forward value is the
thing that is wrong, so C's AD is right while its FD reference stays
contaminated and the gates still fail.

Oblique `theta = 0.3` is identical under all five, which is the control: the
defect lives where the modes sit on the cut.

---

## 4. What ships

```python
scale = xp.maximum(xp.max(xp.abs(r)), 1.0) if r.size else 1.0
flip = (xp.abs(r.real) <= band * scale) & (r.imag < 0)
return r * xp.where(flip, -1.0, 1.0)
```

**The SELECTOR is unchanged.** Which modes flip is decided by exactly the
round-1 predicate, so the population measurements behind `_CUT_BAND_REL` —
fourteen decades of separation, the cutoff-mount ladder, the band-scale
comparison — all carry forward untouched. Only the VALUE moves, by exactly
`2 Re(r)`.

Multiplying by a real `+/-1` whose sign is a piecewise-constant boolean predicate
is holomorphic on each piece, so the derivative is the analytic
`-1 / (2 sqrt(x))` of the branch actually taken. A side benefit: the returned
root now squares back to `lam^2` **bit-for-bit**, where `conj(r)` squared to
`conj(lam^2)` and forced a two-sided residual test.

The five JAX/NumPy/CuPy callers reach this through the round-2 consolidation
(`pmm/twod.py`, `pmm/_jax_twod.py`, `pmm/_jax_stack2d.py`,
`pmm/_jax_twod_jones.py`, `rcwa/oned.py`, `rcwa/stack.py`, `elements/berreman.py`,
`elements/_berreman_jax.py`), so one edit covers all of them. That is round 2's
dividend.

---

## 5. The price: `|X| <= 1`

`Re(-r) = -|Re(r)|`, so `|X| = exp(-Re(lam) k0 L)` exceeds 1 for a flipped mode.
Censused over the round-1/round-2 fixture sets (hybrid 2-D PMM at 8 incidences,
1-D RCWA at 4 angles x 3 permittivities including a lossy one, 23 solves; 24
arrays carried at least one flip, 33 flipped modes total):

| quantity | `conj(r)` | `-r` |
|---|---|---|
| worst `\|Re(r)\|` of a flipped mode | 6.137790e-10 | 6.137790e-10 |
| worst `\|Re(r)\| / scale` (band = 1e-08) | 2.080682e-10 | 2.080682e-10 |
| spectrum scale range | 2.9499 .. 6.3272 | same |
| `k0 L` range | 2.8560 .. 1.1424e+07 | same |
| worst `min Re(lam)` | +8.017335e-22 | **−6.137790e-10** |
| **worst `\|X\| − 1`** | **−2.220446e-16** | **+1.752949e-09** |

**`1.752949e-09`.** That is above the `~1e-12` the brief set as the trigger for
a mitigation, and the mitigation named there (zeroing the real part) is
unavailable — see candidate B in §3. The simplest form that passes both the
gradient gates and the census is plain `-r`, and the excess is acceptable on
its own terms:

* a flipped mode is by construction a PROPAGATING one (its `lam^2` is a
  negative real to within the backward error), so `|X| = 1` in exact
  arithmetic. This is a `2e-09` amplitude excess on a unit-modulus phase, not
  the evanescent `exp(+|gamma| k0 L)` blow-up the `Re(lam) >= 0` rule exists to
  prevent — that catastrophe needs `Re(lam)` of order 1, and an UNFLIPPED root
  still satisfies `Re(lam) >= 0` exactly;
* the excursion is bounded BY THE BAND that admitted the mode,
  `|Re(lam)| <= _CUT_BAND_REL * max(max|r|, 1)`, which is a derived bound, not a
  fitted one;
* it buys off a **1.6e-07** forward discontinuity — two decades larger.

---

## 5b. How far the ANSWER moves against `59105d6`

The repair brief asked for a BIT-IDENTITY census on the grounds that "the
flipped-mode values move by exactly `2 Re(r) ~ 1e-16`". That premise is the
one §2.2 corrects, so a bit-identity claim is not available and none is made.
What IS measured is the motion itself, over 25 RCWA / PMM / Berreman fixtures,
round-2 body against round-3 body, on both builds (`OPENBLAS_CORETYPE=HASWELL`,
one thread). **Windows py3.14 and WSL py3.12 agree to every digit reported.**

| fixture group | max `\|post − pre\|` | relative |
|---|---|---|
| `pmm2d` near-normal (θ = 0, 1e-7), TE | **3.681215e-08** | **1.779481e-07** |
| `pmm2d` near-normal, TM | 3.118252e-09 | 1.531837e-08 |
| `pmm2d` oblique 0.3 / 0.6, conical, LOSSY | **0** | **bit-identical** |
| `pmm2d` uniform-spacer stack, `n_orders` 3/4/5 | 4.884981e-15 | 5.016027e-15 |
| `rcwa_efficiency_1d`, θ = 0 / 0.2 / 0.5, TE+TM | 2.322587e-13 | 2.322587e-13 |
| `rcwa_efficiency_1d` thin ladder `M` = 19 | 1.110223e-15 | 1.124345e-15 |
| `rcwa_jones_1d_segments` exact-index coincidence | 1.172396e-13 | 1.281427e-13 |
| `berreman_jones_1d`, θ = 0 and 0.4 | **0** | **bit-identical** |

**16 of 25 fixtures moved at all**; the worst motion is `3.681215e-08`
(`1.78e-07` relative) and it is at NEAR-NORMAL incidence on the hybrid 2-D PMM
— exactly where §2.3 measured the `conj` rule's forward discontinuity, and of
exactly that size. Everything away from the cut is bit-identical, including
every Berreman surface and every oblique, conical and lossy `pmm2d` fixture.
That is the shape the change should have: it moves the answer only where the
old rule was making the answer depend on a band-level accident.

---

## 6. Post-fix gates

### 6.1 JAX gradient census

`jax.grad` vs central FD at `h = 1e-4`, gate `1e-4` relative. Central
differencing carries truncation error `f'''(x) h^2 / 6` (~1e-08 relative here)
and a round-off floor `eps |f| / h` (~1e-12), so the gate sits four decades
above the FD method error and four to five below the observed defect. An
h-ladder is reported per row so an FD-limited reading is visible.

Run on BOTH builds: **WIN** = Windows py3.14.6 / jax 0.11.0 / numpy 2.4.4, and
**WSL** = py3.12.3 / jax 0.10.2 / numpy 2.4.6 — which is CI's JAX python.

| fixture | rel, WIN | rel, WSL |
|---|---|---|
| near-normal `sum(T)` | 3.464e-07 | 3.458e-07 |
| near-normal `sum(R)` | 5.692e-06 | 5.706e-06 |
| θ = 1e-7 `sum(T)` | 2.877e-07 | 2.885e-07 |
| oblique 0.30 TE | 8.143e-06 | 8.144e-06 |
| oblique 0.30 TM | 9.340e-07 | 9.343e-07 |
| conical θ=0.40 φ=0.7 | 1.349e-07 | 1.347e-07 |
| conical 1e-8 off normal, φ=0.7 | 1.246e-07 | 4.083e-06 |
| LOSSY `eps = 6+0.4j` normal | 3.724e-07 | 3.726e-07 |
| LOSSY oblique 0.35 | 5.257e-06 | 5.257e-06 |
| high-index substrate 2.4 | 5.322e-07 | 5.309e-07 |
| rectangular pillar | 3.571e-07 | 3.569e-07 |
| deeper `degree = 7`, `n_orders = 3` | 1.513e-07 | 1.526e-07 |

**Worst 8.143e-06 (WIN) / 8.144e-06 (WSL)** against the 1e-04 gate — four
decades of margin, and the two builds agree to three significant figures on
every row but one. **JAX/NumPy forward parity worst 7.709e-15 (WIN) /
8.585e-15 (WSL)** on the same twelve.

SLANTED is excluded by construction — the hybrid 2-D PMM has no slant
parameter, so there is no such fixture to differentiate.

**One exclusion, and it is not this defect.** `theta` EXACTLY `0` with
`phi != 0` reads `rel = 4.569e-01`, and it reads **`4.569e-01` bit-identically on
the pre-round-1 tree `48c8747`**. The twin blends the TE/TM basis to the lab
axes through `where(kt < 1e-12, 0.0, real(kx0)/safe_kt)`, so `d/dtheta` of the
azimuth is taken as zero at the measure-zero point where `kt == 0`. `1e-08` rad
off normal it is gone (`rel = 1.246e-07`). This is a pre-existing azimuth-gauge
artifact, out of scope for the branch cut, and it is recorded here rather than
fixed.

### 6.2 The exact-index coincidence (S1-2)

`|sum R + sum T − 2|` on `rcwa_jones_1d_segments` with a rotated director whose
ordinary `no^2` is 2.25, a groove of 2.25 and `n_substrate = 1.5`, over
`n_orders` 11..41. The structure is provably lossless, so `= 2` is exact at any
truncation under the Laurent rule: the conservation law is the reference.

| arm | WIN-HASWELL | WIN-PRESCOTT | WIN-SANDYBRIDGE | WSL-HASWELL | WSL-PRESCOTT |
|---|---|---|---|---|---|
| POST coincident | 1.643e-13 | 3.579e-13 | 3.477e-13 | 1.386e-13 | 3.682e-13 |
| POST detuned (1e-3) | 1.648e-13 | 6.515e-13 | 3.788e-13 | 1.616e-13 | 6.772e-13 |
| PRE coincident | **2.761e-02** | **7.289e-04** | **4.690e-02** | **5.001e-02** | **1.584e-03** |
| PRE detuned | 5.840e-13 | 5.071e-13 | 4.476e-13 | 4.374e-13 | 4.894e-13 |
| warnings, POST | 0/16 | 0/16 | 0/16 | 0/16 | 0/16 |
| warnings, PRE | 16/16 | 11/16 | 14/16 | 16/16 | 11/16 |

POST envelope **6.772e-13**; smallest PRE reading that manifests **7.289e-04**.
The 1e-09 bar sits 3.17 decades above the first and 5.86 below the second.

### 6.3 The mode match

Worst `rcond(a + b)` recorded by the library's own `_INV_CENSUS` over each
fixture:

| fixture / arm | WIN-HAS | WIN-PRE | WIN-SAND | WSL-HAS | WSL-PRE |
|---|---|---|---|---|---|
| spacer coincident, POST | 1.8281e-03 | 1.8281e-03 | 1.8281e-03 | 1.8281e-03 | 1.8281e-03 |
| spacer detuned, POST | 1.8341e-03 | 1.8341e-03 | 1.8341e-03 | 1.8341e-03 | 1.8341e-03 |
| spacer coincident, PRE | 2.303e-07 | 9.887e-08 | 5.160e-08 | 1.579e-07 | 2.737e-07 |
| spacer detuned, PRE | 8.700e-05 | 1.167e-05 | 1.200e-04 | 3.369e-04 | 3.388e-05 |
| thin ladder TE, POST | 6.2504e-02 | 6.2504e-02 | 6.2504e-02 | 6.2504e-02 | 6.2504e-02 |
| thin ladder TE, PRE | 3.331e-19 | 4.676e-21 | 6.217e-20 | 3.614e-19 | 2.712e-21 |

Every POST row is identical to five figures on all five samples.

---

## 7. Tests restated

Four tests pinned the `conj` contract and now pin the `-r` one. None is
relaxed; each is restated as a decision, and three of the four are STRONGER
than what they replaced.

| test | was | is |
|---|---|---|
| `test_verify_rcwa_even_sector.py::test_sqrt_decay_never_returns_a_growing_root` → `..._a_growing_root_can_only_come_from_the_band_and_only_by_the_band` | `Re(lam) >= 0`, violations exactly 0 | negative real parts are EXACTLY the flipped set (boolean array equality); an unflipped root keeps `Re >= 0`; a flipped one's `\|Re(lam)\|` is bounded by the band that admitted it; the root squares back to `lam^2` exactly (was a two-sided residual) |
| `..::test_the_flip_only_ever_moves_a_root_along_the_imaginary_axis` → `..::test_the_flip_is_the_exact_minus_r_involution` | `conj` fixes `Re(lam)` and `\|lam\|` | the returned value is bit-for-bit `+sqrt(z)` or `−sqrt(z)`, `\|lam\|` unmoved, `lam^2` bit-identical |
| `test_fix_rcwa_even_sector_wsl.py::test_sqrt_decay_pins_the_outgoing_root_through_eigensolver_noise` | `Im(lam) >= 0` **and** `Re(lam) >= 0` | `Im(lam) >= 0` unchanged; contraction restated at its true strength — `Re(lam) >= −band`, and `Re >= 0` for every mode outside the band |
| `test_fix_branch_cut_round2.py::test_the_shared_body_is_bit_identical_to_the_round_one_body` → `..._keeps_the_round_one_SELECTOR_and_changes_only_the_value` | bit-identical to the round-1 body over 4,010 values | the FLIP SET is identical to round 1's over the same 4,010 values at 5 array sizes; off the flip set the two bodies are bit-identical; on it they differ by exactly `2 Re(r)`, bit for bit |

One test was RED on every CI shard because the coincidence it exercised is
cured:

| test | was | is |
|---|---|---|
| `test_audit_s1_2_rcwa_lossless_tripwire.py::test_closure_warning_names_the_exact_index_coincidence_and_the_detune` | asserted the warning text contained `EXACTLY EQUAL` **and** `DETUNE`; DID NOT WARN on every shard | split in two: `test_the_exact_index_coincidence_now_closes_and_the_warning_is_silent` (closes to 1e-09, agrees with the detuned control, 0 warnings — §6.2) and `test_the_pre_round_one_branch_reopens_it_and_the_message_names_the_cause` (the ENGINEERED fail-before, and the surviving home of the message-text claim, now asserted on the arm where the message is reachable) |

Both halves of the old assertion had expired: round 2 dropped the
"detune by ~1e-6" advice (measured NOT to cure this class — a relative 1e-6
leaves 8.1e-05 on Windows, 2.3e-04 on WSL), and the fix cured the fixture.

## 7b. One more build-dependent claim, found by the core-type sweep

`test_verify_branch_cut_round2.py::test_a_lossy_spacer_alone_does_not_empty_the_acted_on_population`
FAILED under `OPENBLAS_CORETYPE=PRESCOTT` (Katmai) on WSL — and does so
**bit-identically on unmodified `59105d6`**, so it is a pre-existing defect the
sweep exposed rather than a round-3 regression. It was green on every other
kernel and on every Windows kernel, which is exactly why the round-2 pass did
not see it.

The test counted the modes the band ACTS ON. That count needs two things: the
band's REACH (`|Re(r)| <= band * max(max|r|,1)`, a magnitude, decided by the
mode's physics) **and** `Im(r) < 0` — the eigensolver's backward-error sign,
which is a coin flip per mode and is the very quantity the branch cut exists to
stop mattering. Over this fixture's 8-mode on-cut population it came out zero
at one truncation by luck:

| kernel | acted-on count, `n_orders` 3 / 4 / 5 |
|---|---|
| Haswell, Zen, Sandybridge (both builds) | 8 / 8 / 8 |
| **Katmai (`PRESCOTT`), WSL** | **8 / 0 / 8** |

The on-cut count is **8 at every truncation on every kernel**. So the test is
restated onto the REACH, renamed
`test_a_lossy_spacer_alone_does_not_empty_the_bands_jurisdiction` — which is
also the quantity the scope statement it defends is actually about ("the
patterned layer's propagating modes are still exactly on the cut, so the lossy
exemption is about the PATTERNED layer, not about loss anywhere"). The
acted-on count is still asserted, but SUMMED OVER THE LADDER, where one unlucky
truncation cannot empty it. Both lossy-exemption legs (`== 0`) also move to the
on-cut count, where they are strictly stronger: they previously could have
passed by the same luck.

Post-restatement: **11 passed** on both builds under all four runnable core
types (WIN and WSL x HASWELL / ZEN / PRESCOTT / SANDYBRIDGE).

---

## 7c. A THIRD test asserting the cured coincidence still bites

`test_v5_20_12_rcwa_jones_2d_fff_nv.py::test_stripe_fixture_is_free_of_the_mode_match_degeneracy`
is the same expired contract as S1-2 in a third file. It asserts that the
index-coincident groove (`eps = 2.25 = no^2 = n_sub^2`) still violates the 1-D
lossless theorem by at least `1e5` x the clean fixture's closure, and that a
closure warning fires while it does. Its own docstring anticipated the
outcome exactly:

> If the solver is ever made degeneracy-robust this test fails, and that is the
> gate working: it must then be re-derived (durability rule), not widened.

It PASSED on every CI shard at `59105d6` and FAILS on this host at the same
commit with the round-3 body reverted, on both builds — so it is the THREAD
COUNT that decided it, not round 3: CI's fast unit lane leaves BLAS unpinned on
4-core runners, and this is amplified rounding.

Measured over `n_orders` 11..41 on **sixteen** configurations —
(Windows py3.14.6 / numpy 2.4.4, WSL py3.12.3 / numpy 2.4.6) x
(HASWELL, NEHALEM, KATMAI, SANDYBRIDGE) x (1, 4) BLAS threads:

| arm | worst `\|sum R + T − 2\|` | degen/clean | warned |
|---|---|---|---|
| POST clean (2.10) | 1.52e-13 .. 5.36e-13 | — | 0/16 |
| **POST degen (2.25)** | **1.39e-13 .. 5.19e-13** | **0.72 .. 2.49** | **0/16** |
| PRE degen (2.25) | 7.29e-04 .. 7.19e-02 | 3.1e+09 .. 2.7e+11 | 11..16 of 16 |

The re-derivation asserts what is now true and is TIGHTER than what it
replaces: the coincident groove holds the theorem (`< 1e-09`), is
INDISTINGUISHABLE from the clean fixture (ratio `< 10`, measured `<= 2.49`),
and fires no warning — with the ENGINEERED pre-round-1 arm keeping the original
`> 1e5` claim alive on the arm where it is still true. The 2.10 fixture stays;
the move off 2.25 is now belt-and-braces rather than load-bearing.

---

## 8. Two fail-befores moved off amplified rounding

`test_fix_branch_cut_round2.py::test_the_spacer_coincidence_is_what_breaks_the_pre_round_one_branch`
and
`test_m1_conditioning_guard.py::test_x1_is_closed_across_the_whole_thin_ladder_and_reopens_pre_fix`
are ENGINEERED fail-befores: they reinstate the pre-round-1 branch body and
re-read the fixture. Each closed on a MAGNITUDE — `worst closure > 1e3 x
control`, and `worst sum(R) > 10x the converged value` — and a magnitude here is
amplified rounding: how far a singular mode-match throws the answer depends on
where the singularity falls relative to the arithmetic, so it is a per-kernel
fact by its nature. On CI's AMD runners neither reproduced (coincident spacer
`{3: 4.441e-15, 4: 1.776e-15, 5: 1.110e-15}` against a detuned control at the
same floor; `sum(R)` only `0.68x` from converged), and both failed for
demonstrating nothing.

Both now close on quantities that are decisions.

**THE SIGN CENSUS.** During a solve, count the modes the selector returns
numerically ON THE CUT (`|Re(lam)| <= _CUT_BAND_REL * max(max|lam|,1)`, i.e.
propagating and lossless) but carrying the INCOMING root (`Im(lam) < 0`). The
recursion requires a layer's forward set to carry the outgoing root; the
incoming root IS the defect. Post-fix the count is ZERO BY CONSTRUCTION.
Pre-fix the sign is the backward error — a coin flip per mode — so over a ladder
with tens of modes per rung, "at least one" survives every kernel.

| fixture | on-cut modes | POST | PRE: WIN-HAS / WIN-PRE / WIN-SAND / WSL-HAS / WSL-PRE |
|---|---|---|---|
| spacer stack (`n_orders` 3,4,5) | 192 | **0** | 31 / 23 / 34 / 26 / 17 |
| thin ladder TE (`M` 6..30) | 2505 | **0** | 413 / 426 / 444 / 413 / 426 |

**THE CONDITIONING.** `rcond(a + b)` at the mode match — §6.3. For the spacer
the coincident/detuned split under the pre arm is `<= 2.737e-07` against
`>= 1.167e-05`, so the 1e-06 bar sits 0.56 decades above the worst coincident
reading and 1.07 below the best detuned one. For the thin ladder the two
populations are `<= 3.614e-19` (PRE) against `6.2504e-02` (POST) — seventeen
decades of empty gap, and the 1e-12 bar sits 6.4 decades above the PRE readings.

The spacer test now asserts three things together: the pre arm MIS-ROOTS (a);
on the COINCIDENT spacer that collapses the mode match (b); on the DETUNED
spacer, mis-rooted just as much, it does not (c). Nothing but the spacer index
differs between (b) and (c), which is the claim the test's name makes.

---

## 9. Measurement conditions, and what could not be measured

Every number above is one BLAS thread with `OMP_NUM_THREADS`,
`OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` pinned to 1 on the command line,
and `OPENBLAS_CORETYPE` pinned per run. Builds: **WIN** = Windows 11, py3.14.6,
numpy 2.4.4, jax 0.11.0; **WSL** = Ubuntu, py3.12.3, numpy 2.4.6.

**The core-type ladder in the repair brief had to be corrected, and every
correction is verified rather than assumed.** Each `OPENBLAS_CORETYPE` below
was confirmed through `threadpoolctl.threadpool_info()['architecture']` AND
through a real LAPACK call (a 200x200 `eigvals`), because the banner and the
kernel can disagree.

1. **There is no Zen kernel.** The bundled `libscipy_openblas` is DYNAMIC_ARCH
   over `Atom, Barcelona, Bulldozer, Cooperlake, Core2, Dunnington, Excavator,
   Haswell, Katmai, Nehalem, Northwood, Opteron, Penryn, Piledriver, Prescott,
   Sandybridge, SapphireRapids, SkylakeX, Steamroller` — **no `Zen`**.
   `OPENBLAS_CORETYPE=ZEN` reports `architecture = Haswell` on BOTH builds and
   returns a bit-identical `eigvals`, i.e. the override is a silent no-op. The
   consequence is load-bearing and favourable: CI's **AMD EPYC 7763 (Zen 3, no
   AVX-512)** has no Zen target to select either, so **CI is already executing
   the Haswell kernel**, and the HASWELL rows above ARE the CI kernel.
2. **`SKYLAKEX` is not runnable here.** The measuring host is an AMD Ryzen 9
   5950X (Zen 3), which has no AVX-512. Forcing that kernel aborts the process
   with **`Illegal instruction` (exit 132)** on Windows; on WSL it survives the
   `threadpool_info()` banner — reporting `SkylakeX` — and then dies on the
   first real `eigvals`, which is why the banner alone is not sufficient
   confirmation.
3. **`PRESCOTT` and `KATMAI` are the same kernel.** Both report
   `architecture = Katmai` and return a bit-identical `eigvals`.

So the four genuinely DISTINCT kernels available here are **Haswell**
(FMA3/AVX2 — CI's), **Sandybridge** (AVX, pre-FMA), **Nehalem** (SSE4.2) and
**Katmai** (SSE2-era), and those are what the matrix in section 9.1 sweeps.

### 9.1 Every decision over kernel x thread count, both builds

Because CI's fast unit lane leaves BLAS **unpinned** on 4-core runners, the
thread count is the axis on which this host and CI actually differ — and it is
the axis on which two of the failures in this repair turned out to hinge. The
thirteen DECISION tests (the two gradient gates plus the clean-zero sibling,
the two restated engineered fail-befores, the two halves of the tripwire
restatement, the four even-sector / selector restatements, the jurisdiction
restatement and the fff_nv re-derivation) were therefore run over the full
product.

THE THIRTEEN DECISIONS, all in one pytest invocation per configuration:

| # | decision |
|---|---|
| 1 | `test_niche_audit_w9_eig_vjp::test_pmm2d_near_normal_angle_gradient_improved` |
| 2 | `test_v5_14_0_pmm2d_autodiff::test_gate_angle_grad_at_normal_offcenter_is_genuine` |
| 3 | `test_v5_14_0_pmm2d_autodiff::test_gate_degen_angle_grad_centered_square_is_clean_zero` |
| 4 | `test_fix_branch_cut_round2::test_the_spacer_coincidence_is_what_breaks_the_pre_round_one_branch` |
| 5 | `test_fix_branch_cut_round2::test_the_shared_body_keeps_the_round_one_SELECTOR_and_changes_only_the_value` |
| 6 | `test_m1_conditioning_guard::test_x1_is_closed_across_the_whole_thin_ladder_and_reopens_pre_fix` |
| 7 | `test_audit_s1_2_rcwa_lossless_tripwire::test_the_exact_index_coincidence_now_closes_and_the_warning_is_silent` |
| 8 | `test_audit_s1_2_rcwa_lossless_tripwire::test_the_pre_round_one_branch_reopens_it_and_the_message_names_the_cause` |
| 9 | `test_verify_rcwa_even_sector::test_a_growing_root_can_only_come_from_the_band_and_only_by_the_band` |
| 10 | `test_verify_rcwa_even_sector::test_the_flip_is_the_exact_minus_r_involution` |
| 11 | `test_fix_rcwa_even_sector_wsl::test_sqrt_decay_pins_the_outgoing_root_through_eigensolver_noise` |
| 12 | `test_verify_branch_cut_round2::test_a_lossy_spacer_alone_does_not_empty_the_bands_jurisdiction` |
| 13 | `test_v5_20_12_rcwa_jones_2d_fff_nv::test_stripe_fixture_is_free_of_the_mode_match_degeneracy` |

Every cell below is **13 passed**. No decision changes with the kernel or with
the thread count, on either build.

| kernel (verified arch) | threads | WIN py3.14 | WSL py3.12 |
|---|---|---|---|
| HASWELL (Haswell — CI's) | 1 | 13 passed, 87.7s | 13 passed, 94.3s |
| HASWELL (Haswell — CI's) | 2 | 13 passed, 27.6s | 13 passed, 52.1s |
| HASWELL (Haswell — CI's) | 4 | 13 passed, 98.7s | 13 passed, 115.2s |
| NEHALEM (Nehalem) | 1 | 13 passed, 119.2s | 13 passed, 155.1s |
| NEHALEM (Nehalem) | 4 | 13 passed, 42.8s | **DID NOT COMPLETE** |
| KATMAI (Katmai, = PRESCOTT) | 1 | 13 passed, 71.4s | 13 passed, 239.2s |
| KATMAI (Katmai, = PRESCOTT) | 4 | 13 passed, 188.1s | 13 passed, 112.8s |
| SANDYBRIDGE (Sandybridge) | 1 | 13 passed, 55.1s | 13 passed, 214.5s |
| SANDYBRIDGE (Sandybridge) | 4 | 13 passed, 51.9s | 13 passed, 143.4s |
| HASWELL | unpinned | **DID NOT COMPLETE** | **NOT RUN** |

**SIXTEEN of the nineteen cells completed, and all sixteen are 13/13.** Three
did not, and none of the three is a failure -- each was still running when the
session closed, on a host shared with other work:

* WSL x NEHALEM x 4 threads exceeded a 500 s cap in the batch run and was
  still running 30 minutes into a re-run with no cap.  Its siblings
  (WIN x NEHALEM x 4, and WSL x NEHALEM x 1) are both green.
* WIN x HASWELL x unpinned was still running after 21 minutes.
* WSL x HASWELL x unpinned was not started, because the two unpinned arms
  starve each other on this host (see below).

The WIN threads=2 row is from the first pass of this matrix; the rest are from
a second pass, and the wall times are therefore not comparable between rows.
They are reported only to show which cells were actually executed.

**UNPINNED is 20x slower on this host and that is a property of the host, not
of the library.** Leaving BLAS unpinned on a 16-core / 32-thread desktop makes
this batch thrash — it spends its time in thread barriers on the many SMALL
solves the engineered pre-arm tests perform — where a single decision
(`test_x1_is_closed_...`) run alone takes 13.6s unpinned against 14.0s at one
thread, i.e. the slowdown is contention between the batch's tests, not any one
of them. CI does not see this because its runners have 4 cores.


Other limits:

* the `conj` → `-r` motion in the FORWARD answer is **not** the `2 Re(r) ~ 1e-16`
  the brief anticipated, so a strict BIT-IDENTITY census against `59105d6` is
  not the right instrument and none is claimed. The motion is up to `1.6e-07`
  in `sum(T)` at near-normal incidence and `<= 6.9e-12` away from the band's
  edge; what IS bit-identical is the flip SELECTOR (§7) and the root's square;
* the post-fix JAX census in §6.1 is now run on BOTH builds, including WSL
  py3.12 / jax 0.10.2 (CI's JAX python), and they agree. The PRE-fix
  reproduction and the candidate comparison in §3 are Windows-only; the CI log
  supplies the py3.12 pre-fix readings (6.18 / 0.71 relative, the same two
  gates), so the defect itself is two-build evidenced even though the
  candidate sweep is not;
* the pre-existing azimuth-gauge artifact at `theta == 0` exactly with
  `phi != 0` (§6.1) is recorded, not repaired;
* **`lumenairy` is pip-installed on the measuring host as a path install
  pointing at a DIFFERENT clone.** A probe run as `python path/to/probe.py`
  puts the SCRIPT's directory on `sys.path[0]`, not the repo root, and silently
  imports that other clone. Every probe run for this document sets
  `PYTHONPATH` to the worktree explicitly and prints `lumenairy.__file__`. An
  earlier pass of these measurements was invalidated by exactly this and had to
  be repeated.
