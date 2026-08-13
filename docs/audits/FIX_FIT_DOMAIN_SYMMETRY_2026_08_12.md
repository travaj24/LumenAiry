# FIX -- the fit domain is not the interpolant

`fix/fit-domain-symmetry`, off `origin/main` @ `21802f9`.
Files: `lumenairy/elements/_lens_traced.py`,
`lumenairy/elements/_lens_imap.py`,
`tests/unit/test_fix_d5_fit_domain_basis.py`.

---

## 0. VERDICT

**The named defect is CLOSED at the mechanism, and it is measured rather than
argued.**  `_fit_domain_basis_ok = (newton_fit != 'spline')` was answering two
different questions with one flag; it now answers only the one it is a fact
about.  The inverse-characteristic model is handed the SAME fit domain on
either backend, and the consequence is exact:

```
                              polynomial          spline
model fit samples             2 809 / 2 641       2 809 / 2 641   (was 32 761)
model exit half-box (mm)      2.3029513405441     2.3029513405441 (was 7.0184)
model held-out OPL (waves)    1.913643732582e-08  1.913643732582e-08
model held-out position (m)   4.275398611531e-12  4.275398611531e-12
```

Identical to every printed digit, at exit degrees 10, 12, 14, 16 and 18.  And
when both backends ENGAGE that model, the two returned fields are **byte
identical -- a backend spread of exactly 0.0**, which is the strongest form
the c6 guard's premise can take.

**The shipped default path does not move.**  Measured, not reasoned: 12
fit-domain configurations on the polynomial basis and 4 on the spline basis,
byte-compared against `origin/main` -- **16 of 16 IDENTICAL**.  The change is
reachable only when the resolved basis cannot restrict its own forward fit AND
the inverse map is being built, which on the shipped default
(`TRACED_INVERSE_MAP = False`) is no call at all.

**THE DEFAULT IS NOT FLIPPED, AND THE REASON IS A THIRD COUPLING THAT IS NOT
THE FIT DOMAIN.**  With the fit domain symmetric, the c6 backend-symmetry
guard still fails with the flag forced on -- at the same 1.0600e-02 -- because
`newton_fit` still reaches the model's ACCEPTANCE through G8's arm B, which is
the element's own Newton on the element's own forward fits.  That coupling was
not known when S6.5b named the remaining routes.  It is now localised,
measured against the element's own traced landings, and **shown to be a FALSE
REFUSAL**: at the points that actually decide the field the model is 3.0x more
accurate in entrance position and 1.6x more accurate in OPL than the very arm
G8 refuses it against.  See S6.  A shipped guard is not weakened to buy a
default flip, and neither is a feature's own guard re-architected inside a fix
whose scope is the fit domain -- so the default stays `False` and S6 carries
the specification and the evidence the next pass needs.

---

## 1. THE DEFECT -- one flag, two questions

`_fit_domain_basis_ok` was read at three sites and was being asked two
different things:

| | question | is it about the basis? |
|---|---|---|
| Q1 | can THIS BASIS restrict ITS OWN forward fit to the requested region? | **yes** |
| Q2 | is there a requested region at all? | **no** |

Fix D5's adjudication answers **Q1**, and this fix does not touch it: a
`RectBivariateSpline` takes no least-squares weights, and one NaN in its data
array makes `.ev()` return NaN at the grid CENTRE, so neither implementation
of the restriction has a home there.  Both halves are still pinned by
`test_a_nan_masked_grid_destroys_a_rect_bivariate_spline`.

But the flag was also answering **Q2**, and Q2 is a property of the BEAM and
of the traced samples.  So a second consumer that COULD honour the region
never saw one.  That consumer is the inverse-characteristic model: a global
total-degree-14 Chebyshev in EXIT coordinates -- *exactly* the mechanism the
restriction exists to control, and one that needs it for the same reason the
forward polynomial fit does (S6.5b of `BUILD_INVERSE_MAP_2026_08_11` measured
the unrestricted model at 4.5258e-01 waves of held-out OPL error inside the
beam against 1.9965e-05 restricted -- 4.4 decades).

Measured on niche C6's own fixture, with the flag forced on, before this fix:

| | polynomial | spline |
|---|---|---|
| samples handed to the model | 2 809 | **32 761** (the whole launch square) |
| model exit half-box (mm) | 2.3030 | **7.0184** |
| G8 | passes, ENGAGES | **refuses** |
| backend spread | -- | **1.0600e-02** against a 5e-04 bar |

---

## 2. THE TWO ROUTES -- A shipped, B refuted before it was built

**Route B (make the polynomial path's restriction expressible for spline) is
refuted by arithmetic, and the refutation is a code fact rather than a
preference.**  `RectBivariateSpline` needs a full NaN-free TENSOR grid, so the
only restriction that basis can express by sample selection is a rectangular
SUB-LATTICE of the launch lattice.  **No rectangle is a disc.**  On niche C6's
own lattice (181 x 181, disc radius from `fit_radius_beam_factor=2.0`):

```
launch nodes                                          32 761
inside the fit disc (r <= 3.0000 mm)                   2 809
largest sub-lattice that stays inside the disc         1 849
  ... disc samples it DROPS                              960
best-OVERLAP sub-lattice, symmetric difference            504
  ... kept but outside the disc                           252
  ... inside the disc but dropped                         252
```

Every rectangle either keeps nodes the disc excludes or drops nodes the disc
keeps -- swept over 200 half-widths in
`test_no_rectangular_sub_lattice_can_express_the_disc`, with no exact match
and a best symmetric difference above 10 % of the disc.  So a sub-lattice
restriction would leave the two bases handing the model DIFFERENT sample sets
anyway: **it cannot close this defect even in principle**, while additionally
moving every spline consumer (the fit-domain-free-oracle idiom in `d7`, `c8`,
`c6` and the C11/C12 validation scripts) and shrinking the domain the Newton
may evaluate in -- the axis on which fix D5 measured the spline basis
returning an ALL-ZERO exit field.

The "mask and fill" variant keeps the rectangle but fabricates data outside
the disc.  That is not a restriction: the fabricated samples are finite, so
the model would fit them too unless separately masked -- i.e. it still needs
Route A underneath -- and a constant radial extension makes the forward map
non-monotonic outside the disc, which is a fold for the Newton to find.

**Route A is what ships**, and at the level the defect actually lives: not
"apply the restriction to the spline's bicubic" (impossible, and unnecessary
-- a local interpolant does not spread marginal-ray error into the beam), but
**resolve the DOMAIN once, basis-independently, and let each consumer honour
it iff it can.**

---

## 3. THE FIX

### 3.1 The radius was already basis-free; the DECISION TO RESOLVE IT was not

Worth stating precisely, because S6.5b's note ("which needs a basis-independent
beam radius, and the element resolves that per-basis today") is half right.
`_w_in_beam` comes from `_input_beam_amp_radius(E_in, ...)` -- the beam's own
measured second-moment support, a property of the input field -- and
`_beam_fit_radius = min(_frbf * _w_in_beam, launch_radius)` and `_fit_r_max`
are already computed identically on both bases.  **The radius is not
basis-dependent.  Three things around it were:**

1. the APPLICATION of the resulting disc (`if _fit_r_max is not None and
   _fit_domain_basis_ok`), which skipped everything on the spline basis;
2. `_w_in_origin`, the niche-C11 CONCENTRIC candidate's radius, measured only
   under `_fit_domain_basis_ok` -- so on the spline basis the C11 arbiter had
   no second disc to arbitrate and could not have picked the same branch;
3. therefore the branch selection itself (C11 arbiter / C12 predictor), which
   is what decides WHICH disc a decentred beam gets.

All three now read `_fit_domain_wanted = _fit_domain_basis_ok or
_fit_domain_for_model`.

### 3.2 The routing

```
_fit_domain_for_model = (not _fit_domain_basis_ok        # basis cannot fit it
                         and _IMAP.imap_enabled(inverse_map)
                         and _imap_domain_gate)          # ...and one is built
```

`_imap_domain_gate` is the inverse map's own gate (`sub > 1`, Newton
inversion, no chunked assembly, no GPU), lifted to the fit-domain site and
then READ BACK at the build site as `_imap_gate`, so the domain the model is
given and the gate that decides whether to build it cannot disagree.

The restriction is then built ONCE and routed:

* `_fit_domain_basis_ok` -> to the forward arrays, exactly as shipped (NaN
  mask for a concentric disc, `_decentred_fit_restriction` weights + the D7
  order raise for an off-centre or C6-guarded one).  `_imap_* is None` on this
  path, so `build_inverse_map` is handed verbatim the same arrays it always
  was -- **byte-identity by construction, not by re-derivation**;
* otherwise -> to `_imap_xo / _imap_yo / _imap_op / _imap_weights`, consumed
  only at the `build_inverse_map` call.  The forward bicubic is untouched.

Only the WEIGHTS travel to the model on the weighted branch, never the D7
ORDER raise: `decentred_fit_poly_order` is a property of the element's own
degree-`newton_poly_order` fit and has no meaning for a degree-14 exit model.

### 3.3 The announcement narrowed with the bypass

`on_fit_domain_basis`'s message used to end "...so this call runs with NO
ray-fit-domain guard at all."  On a call that builds the model that sentence
is now false, so the message names its own scope: the restriction is inert for
**the forward fit and the Newton inversion that reads it**, and the model
applies the same domain on either basis.  The knob's three-way disposition,
its entry gate, its alias set and the `'cannot honour'` phrase every test
matches on are all unchanged.

---

## 4. WHAT IT PROVES

`_d5_probe3.py`, exit-degree ladder, C6 fixture, flag forced on, build cache
disabled.  `a_*` is the model's held-out error, `b_*` the incumbent's on the
same samples.

| basis | degree | a_pos (m) | a_opl (waves) | b_pos (m) | verdict |
|---|---|---|---|---|---|
| polynomial | 10 | 6.0044e-10 | 3.5294e-06 | 1.7106e-09 | ENGAGE |
| polynomial | 12 | 4.7081e-11 | 2.4879e-07 | 1.7106e-09 | ENGAGE |
| polynomial | **14** | **4.2754e-12** | **1.9136e-08** | 1.7106e-09 | ENGAGE |
| polynomial | 16 | 4.0424e-13 | 1.5939e-09 | 1.7106e-09 | ENGAGE |
| polynomial | 18 | 4.1963e-14 | 1.5001e-10 | 1.7106e-09 | ENGAGE |
| spline | 10 | 6.0044e-10 | 3.5294e-06 | 3.7035e-12 | refuse G8 |
| spline | 12 | 4.7081e-11 | 2.4879e-07 | 3.7035e-12 | refuse G8 |
| spline | **14** | **4.2754e-12** | **1.9136e-08** | 3.7035e-12 | refuse G8 |
| spline | 16 | 4.0424e-13 | 1.5939e-09 | 3.7035e-12 | ENGAGE |
| spline | 18 | 4.1963e-14 | 1.5001e-10 | 3.7035e-12 | ENGAGE |

**The `a_` columns are equal to every printed digit at every degree.**  The
model no longer knows which interpolant the caller asked for.  Only the `b_`
column -- the incumbent, which IS the basis -- differs, and it is what S6 is
about.

And when both bases engage (degree 16 or 18, or degree 14 with the cache
carrying the verdict across), the backend spread is **exactly 0.0**: not
"inside 5e-04", but the same bytes.

**The DECENTRED branch too**, which is the one the fix had to reach through
S3.1's item 2 -- the niche-C11 arbiter, whose concentric candidate's radius
comes from the ORIGIN-referenced moment that used to be measured only under
`_fit_domain_basis_ok`.  A 0.35 mm decentred beam on the D5 singlet,
`fit_radius_beam_factor=2.0`, flag forced on:

| | polynomial | spline |
|---|---|---|
| `n_alive` / `n_fit_samples` | 8 649 / 8 649 | 8 649 / 8 649 |
| `n_detj_census` | 4 467 | 4 467 |
| exit half-box (mm) | 2.1190296540902276 | 2.1190296540902276 |
| held-out OPL (waves) | 3.2694825173979922e-12 | 3.2694825173979922e-12 |
| held-out position (m) | 1.6176296413483726e-16 | 1.6176296413483726e-16 |
| verdict | ENGAGE | ENGAGE |

On this branch both bases engage and there is nothing left to differ.

---

## 5. THE CACHE COLLISION THE FIX EXPOSES -- and its fix

Making the model basis-independent made the two bases collide in the build
cache, whose key is a SHA-256 over everything the MODEL depends on.  G8's
verdict is not a property of the model: it is a property of the PAIR (model,
incumbent).  So the second call inherited the first's acceptance:

```
spline then polynomial   spline REFUSES G8, polynomial ENGAGES   spread 1.0600e-02
polynomial then spline   polynomial ENGAGES, spline HITS CACHE   spread 0.0
```

Same two calls, same process, opposite answers -- **an acceptance decided by
call order.**  **It also mislabelled itself, which is why it could hide:** the hit path set
`rec['cached'] = True` and then `rec.update(hit.guards)` overwrote it with the
ORIGINAL build's `False`, so every cache hit reported itself as a fresh build.
The two lines are now in the other order.  No returned bit changes -- but a
diagnostic that cannot say "this came from the cache" is how a cache defect
stays invisible, and this one hid an acceptance crossing a backend.  No test
in the tree asserts on `cached`, which is the other half of why nothing
caught it.

`build_inverse_map` now takes `parity_tag` -- whatever identifies WHICH
incumbent `parity_invert` is -- and folds it into the key.  The element passes
`(newton_fit, _fit_poly_order, _fit_weights is None, MAX_NEWTON_ITERS)`.
Over-specifying a cache key costs a rebuild; under-specifying one costs a
wrong answer.  Verified order-independent (1.0600e-02 both ways) and pinned by
`test_a_cached_model_cannot_carry_an_acceptance_across_bases`.

---

## 6. THE REMAINING BLOCKER -- G8's held-out probe is degenerate on an
interpolating incumbent

### 6.1 What it looks like

At the shipped degree 14 the model is refused on the spline basis by ONE
channel of G8, by 15 %:

```
held-out entrance-position error 4.2754e-12 m against the incumbent Newton
path's 3.7035e-12 on the same samples (1.15x, bar 1.00x)
```

-- while being 4.3x BETTER than that same incumbent on the OPL channel
(1.9136e-08 vs 8.2722e-08 waves), and 462x better than the polynomial
incumbent on position.  A 462x spread in "the incumbent's accuracy" between
two interpolants that the library elsewhere guarantees describe the same map
is not a difference in accuracy.  It is a difference in the PROBE.

### 6.2 The mechanism

`RectBivariateSpline` is constructed with the default `s=0`: it INTERPOLATES.
It reproduces every launch node exactly -- **including the ones G8 held out**,
because G8 holds them out of the model only.  G8's probe points ARE launch
nodes (that is where exact ray truth lives), so on the spline basis arm B's
"error" is the Newton loop's leftover residual, not its accuracy.  The
polynomial basis is a global least-squares fit and has genuine residual at its
own nodes, so there the probe is fair.

### 6.3 The measurement that settles it

`_d5_probe5.py` / `_d5_probe6.py`.  Truth comes from the ELEMENT ITSELF at
half the ray subsample: the launch lattices nest exactly
(`linspace(-R, R, 2n-1)[2k] == linspace(-R, R, n)[k]`), and the nesting is
VERIFIED before anything is concluded from it -- the two runs' landings agree
at every shared node to **1.648e-17 m**.  So the finer trace supplies the
element's own landings at the MIDPOINTS of the coarse lattice: points arm B
has not interpolated, and which is where every exit pixel actually falls.

Arm B, at its own knots and between them (2 809 node / 2 828 midpoint probes):

| incumbent basis | at NODES (what G8 scores) | at MIDPOINTS | ratio |
|---|---|---|---|
| polynomial | 2.3051e-09 m | 2.3752e-09 m | **1.0** |
| spline | 3.7035e-12 m | 1.2786e-10 m | **34.5** |

The polynomial incumbent's node error IS its production error.  The spline
incumbent's is 34.5x better at its own knots than anywhere else.

### 6.4 VERDICT: the refusal is FALSE, and by a wide margin

All three arms scored at the midpoints -- the production comparison G8 is
trying to make:

| arm | entrance position | OPL |
|---|---|---|
| **A** the model (shipped coefficients) | **4.2934e-11 m** | **2.3723e-07 waves** |
| **B** incumbent Newton, polynomial | 2.3752e-09 m | 5.3949e-05 waves |
| **B'** incumbent Newton, spline | 1.2786e-10 m | 3.8054e-07 waves |

```
model / polynomial incumbent    pos 0.0181    OPL 0.0044
model / spline incumbent        pos 0.3358    OPL 0.6234
```

The OPL column has a per-arm median piston removed, which is the CONSERVATIVE
choice here -- it helps the incumbents.  Without it the spline incumbent reads
7.1696e-07 waves and the model / spline-incumbent OPL ratio is 0.3314 rather
than 0.6234.  The position column, which is the channel G8 actually refused
on, has no piston to remove.

**The model beats BOTH incumbents on BOTH channels at the points that decide
the field.**  G8 refuses it against arm B' anyway, on a number arm B' only
achieves at its own knots.  That is a guard failing "refuse, never degrade" in
the direction of refusing an improvement -- a false refusal, and it is a
defect in G8 independent of any default flip.

### 6.5 The fix, specified -- and deliberately not taken here

The bug is that "held out" is held out of arm A only.  Two routes, and neither
is a one-line change:

1. **Hold the probe samples out of ARM B too.**  Held-out
   `(gi % 3 == 1) | (gj % 3 == 1)` leaves retained `(gi % 3 != 1) & (gj % 3 !=
   1)`, which IS a rectangular sub-lattice and therefore a legal
   `RectBivariateSpline`.  Needs the element to expose a refit hook
   (`parity_refit(keep_i, keep_j) -> invert`), i.e. surgery on an 11.9k-line
   function, and it changes arm B on the POLYNOMIAL path too.
2. **Probe where neither arm has data.**  Trace a few hundred rays at
   NON-NODE entrance points at build time -- ~1 % of the trace the element
   already does -- and score both arms against exact truth there.  This is the
   S6 oracle's own principle at build time, and it is the symmetric one.

Either changes G8's ACCEPTANCE BAR on every call, including design 121's.
That needs the oracle race, the banner and the full suite behind it -- its own
piece of work with its own evidence, exactly as S6.5a said of the fix S6.5b
then went on to refute.  **A guard is not re-architected inside a fix whose
scope is the fit domain.**

---

## 7. THE DEFAULT -- not flipped

Acceptance item (1) of this campaign -- *the c6 backend-symmetry guard passes
UNWEAKENED with both bases, with and without the inverse map flag* -- is met
at the level of the MAP (S4: identical to every digit; byte-identical fields
when both engage) and NOT met at the level of the VERDICT (S6: G8 reaches
opposite conclusions because its arm B is the basis).  With the flag forced on
the guard reads the same **1.0600e-02** it did before this fix, now for a
different and fully localised reason.

So item (5) -- flip `TRACED_INVERSE_MAP` to `True` and run its acceptance --
is not run: it is conditioned on (1), and running it would mean shipping a
default whose own guard refuses it on a documented backend.  The banner
numbers are reported here as CONFIRMATION that this fix does not move them,
not as a re-baseline case; the case itself is S6.4 of
`BUILD_INVERSE_MAP_2026_08_11` and stands exactly as written there.  **This
fix does not edit the acceptance line, and no acceptance line moves.**

`validation/repro_traced_carrier_121/imap_banner_arm.py`, N=2048 / NFC=8192 /
WF=4.0, on this tree:

| arm | FWHM | EE3 | EE6 | EE12 | peak | offset |
|---|---|---|---|---|---|---|
| shipped acceptance line, for comparison | 3.350 um | 90.3 | 99.7 | 99.8 | -- | on-axis |
| `ARM_IMAP=0` (the shipping default) | **3.350 um** | **90.3 %** | **99.7 %** | **99.8 %** | 5.529e+03 | (+0.00, +0.00) um |
| `ARM_IMAP=1` (flag forced on, scoped) | **3.450 um** | **90.3 %** | **99.8 %** | **99.9 %** | 5.486e+03 | (+0.00, +0.00) um |
| S6.5a's recorded reading, for comparison | 3.450 um | 90.3 | 99.8 | 99.9 | 5.486e+03 | -- |

**Both arms reproduce their recorded values in every printed digit.**  The
fail-before arm reproduces the shipped acceptance line, which is the
system-level statement of S8.1's byte-identity; the flag-on arm reproduces
S6.5a's post-`det J`-fix reading, which says this fix does not move the
re-baseline case either -- it neither improves nor damages it, because design
121's production route is the polynomial basis and nothing on that basis
changed.  The acceptance line in `focus_scan_121.py` is untouched.

---

## 8. VERIFICATION

All runs `LUMEN_PIN=0`, Windows / MKL / py3.14.6, `-p no:randomly`.

### 8.1 Byte-identity of the shipped path

`scripts/_d5_byteid.py` and `scripts/_d5_spl.py`, run in this tree and in a
worktree at `origin/main` (`21802f9`), compared with `tobytes()`:

| basis | cases | result |
|---|---|---|
| polynomial (`bare`, `frbf2`, `frbf1p2`, explicit `polynomial`, `auto`, decentred, decentred + D7 order, ray-density, remap + carrier, too-few-samples abandonment, `sub=2`, `inversion_method='fit'`) | 12 | **12 IDENTICAL** |
| spline, flag OFF (`bare`, `frbf2`, decentred, ray-density) | 4 | **4 IDENTICAL** |

### 8.2 The moved-pin adjudication table is EMPTY, and that is a search result

The new branch is reachable only by a call that is BOTH on a basis which
cannot restrict its own forward fit AND building the inverse map.  Swept over
the whole unit tree:

```
files that pass newton_fit='spline'          11
files that touch inverse_map / TRACED_INVERSE_MAP   2
                                             ------
intersection                                  1   (this fix's own witness)
```

`tests/unit/test_niche_c15_inverse_map.py` never leaves the default basis;
`d7`, `c8`, `c6`, `d1`, `d9`, the two pool suites, `e_prepared_and_enums`,
`w3_elements` and `perf_round2` never build a model.  **No shipped assertion
sits on the changed branch, so none moved, and none was re-baselined** -- the
p2/e4 precedent had nothing to adjudicate here.  The spline-basis
byte-identity in S8.1 is the same statement measured on fields instead of on
the test list.

### 8.3 Suites

| suite | result |
|---|---|
| `test_fix_d5_fit_domain_basis` | **43 passed** (was 38; +4 named + 1 parametrisation) |
| `test_niche_c6_stationary_phase_launch` | 21 collected |
| `test_niche_c6_fit_guard` | 13 collected |
| `test_niche_c15_inverse_map` | 27 collected |
| the four above, ONE run | **104 passed** in 2m53s |
| `c11` (21) + `c12` (20) + `c14` (32) + `d2` (38) + `d6` (38) + `tight_focus` (15), one run | **164 passed** in 53m30s |
| `tight_focus` + `fix_d5`, WSL / OpenBLAS mount | **57 passed** in 46m51s (that run predated the decentred parametrisation) |
| `fix_d5` alone, WSL / OpenBLAS mount | **43 passed** in 22.6s |
| `ruff check lumenairy/ tests/unit/` (the CI command) | All checks passed |
| every touched file, ASCII-only | yes (the two pre-existing non-ASCII bytes in `_lens_traced.py` are not on a changed line) |

No `xfail`, no `skip`, no `CHANGELOG` entry, and no acceptance line edited.

---

## 9. WHAT IS NOT CLOSED

1. **G8's held-out probe (S6).**  Localised, measured, specified; not fixed
   *here*.  **CLOSED in the following pass -- see S11 and
   `docs/audits/FIX_G8_PROBE_2026_08_12.md`.**
2. **The 34.5x / 1.0x probe-bias figures are from ONE fixture** (niche C6's
   free leg).  The mechanism is basis-level (`s=0` interpolates) and does not
   depend on the prescription, but the SIZE of the bias does, and a guard
   redesign should measure it on design 121 as well.
3. **The C12 predictor can now emit its disagreement warning on a spline
   call** that builds the model, where before it could not run at all.  That
   is honest (the branch really is being selected there) but it is a new
   warning site on a path that used to have none.
4. **`_fit_domain_for_model` is scoped to calls that build the model.**  A
   future consumer that can honour the domain must add itself to that gate; if
   it does not, it silently inherits the forward fit's basis decision -- which
   is the defect this fix closed, re-opened for a different consumer.  The
   gate-coverage witness catches a basis that cannot honour the domain, not a
   consumer that forgot to ask for it.
5. **Nothing here was run on GPU or through
   `apply_real_lens_traced_multi`'s prepared-screen reuse.**  `use_gpu=True`
   forces `newton_fit='polynomial'` and is excluded from `_imap_domain_gate`,
   so the new branch is unreachable there by construction -- an argument, not
   a measurement.

---

## 10. FILES AND REPRODUCTION

```
# the c6 backend spread, flag off and on
python scripts/_d5_probe.py

# what each basis hands the model
python scripts/_d5_probe2.py

# the exit-degree ladder: the model is identical, the incumbent is not
python scripts/_d5_probe3.py

# arm B at its own knots vs between them (the nesting oracle)
python scripts/_d5_probe5.py

# the production comparison: model vs both incumbents at the midpoints
python scripts/_d5_probe6.py

# byte-identity of the shipped path, against a worktree at origin/main
python scripts/_d5_byteid.py C:/tmp/fix.npz
python scripts/_d5_spl.py    C:/tmp/spl_fix.npz
python scripts/_d5_byteid.py --compare C:/tmp/base.npz C:/tmp/fix.npz

# the shipping banner, one arm per invocation
cd validation/repro_traced_carrier_121
ARM_IMAP=0 python imap_banner_arm.py
ARM_IMAP=1 python imap_banner_arm.py
```

Every probe inserts the REPOSITORY ROOT at `sys.path[0]` before importing
`lumenairy`.  That is not decoration: a script under `scripts/` gets its own
directory as `sys.path[0]`, and this repo is installed, so without the insert
every one of these measurements silently scores the INSTALLED tree instead of
the working one.  It cost two false "no change" readings during this fix.

---

## 11. S6 IS CLOSED -- and so is the default

Written after the fact, so this document does not stay stale about its own
open item.  The full record is
`docs/audits/FIX_G8_PROBE_2026_08_12.md`; the three lines that matter here:

* **S6's false refusal is fixed at the PROBE, not at the bar.**  G8 now scores
  both arms at OFF-LATTICE points -- one ray per launch cell, placed inside it
  by the R2 low-discrepancy sequence -- with truth from the element's own
  trace there (a new `probe_trace` argument, pinned BIT-IDENTICAL to the
  lattice trace when handed the lattice).  `_IMAP_PARITY_FACTOR` is still 1.0,
  still a ratio, still all three channels.
* **S7's acceptance item (1) is now MET at the level of the verdict too.**
  The c6 backend spread with the flag on went `1.0600e-02 -> 0.000000e+00`;
  both bases ENGAGE and return the same bytes.  The 1.0x / 34.5x probe-bias
  table in S6.3 is what the fix acts on, and it is reproduced on design 121:
  the re-architected probe reads the polynomial arm's numbers to three digits
  (6.667e-03 w against the node probe's 6.736e-03 on order (-4,-2)), i.e. the
  bar moved only where the incumbent was exact by construction.
* **`TRACED_INVERSE_MAP` now defaults to `True`.**  S7's statement that item
  (5) "is not run" is superseded; the guard was satisfied rather than
  weakened, and it still refuses -- the S6.5b pre-restriction model at 622x
  and the exit-degree-8 underfit at 2.17x on design 121's order (-4,-2).
