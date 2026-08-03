# Architecture assessment -- can the traced-lens correction layers be encapsulated?

**Date** 2026-08-03. **Tree** `feat/d121-final-closure` @ `5af1edf`.
**Subjects** `lumenairy/elements/_lens_traced.py` (8,035 lines) and
`lumenairy/propagators/carrier.py` (7,389 lines).
**Scope** READ-ONLY. No library, test or validation file was modified by this
study. No measurement was re-run; every number below is either read from the
source/tests/git or computed by static analysis of the shipped tree.

**The question asked.** After D1-D7 and C1-C10 -- fifteen labelled corrections
across six weeks, each with its own module-level switch, its own fail-before
contract and its own era-pinned tests -- is there a more elegant encapsulation
than correction-on-correction, up to and including a refactor of
`apply_real_lens_traced`?

**The answer in one paragraph.** Yes, there is a real unifying structure, and it
is not the one the question's framing implies. Three of the ten switches (C5,
C6, C10) are one object -- *the entrance eikonal and its jet* -- and the source
already says so in prose ("the three halves of one substitution `W -> W +
a_fit`", `_lens_traced.py:4836-4838`) while implementing it as three
hand-summed sites. Five more (R7/F2, P2, D1, D7, C1, plus C6's opt-in guard) are
one policy object -- *which ray samples enter the fit, with what weights, at what
order* -- currently resolved at lines 5253-5372 and applied 480 lines later at
5855-5994. Two (C7, C8) are two views of one measurement -- *the exit region the
traced rays actually reached* -- computed twice, 40 lines apart, from the same
arrays by different rules, and the docs already record the consequence (they are
jointly blind to a band that neither rule covers). **But the refactor that
extracts them is not the intervention with the best ratio, and I do not
recommend doing it now.** The reasons are quantitative and are in S6-S8: the
function is far smaller in code than in prose (S1), its state envelope never
narrows below ~50 live variables in the middle third (S4.5), its documentation
is a 96-edge cross-reference graph that a code move would shear (S6.4), and the
measured price of re-baselining *four* test pins on this codebase was 1,883
lines and five new independent oracles (commit `4c027e3`). What I recommend
instead -- three concrete steps, all cheap, all bit-preserving -- is in S8.

---

## S1. First, the size of the thing -- measured, not assumed

The brief describes "a ~7000-line function". That framing is the single biggest
driver of a "this must be refactored" conclusion, and it is wrong in a way that
matters.

| quantity | `_lens_traced.py` | `carrier.py` |
|---|---:|---:|
| total lines | 8,036 | 7,390 |
| blank | 480 | 509 |
| `#` comment lines | 2,897 | 1,338 |
| docstring lines | 1,987 | 2,925 |
| **remaining (code)** | **~2,672** | **~2,618** |
| prose share | **61 %** | **58 %** |

And inside `apply_real_lens_traced` itself (`_lens_traced.py:3442-7258`):

| region | lines | blank | `#` comments | rest |
|---|---:|---:|---:|---:|
| whole `def` | 3,817 | 150 | 1,272 | 2,395 |
| its docstring (3488-4260) | 773 | 64 | 0 | 709 |
| **its body (4261-7258)** | **2,998** | **86** | **1,272** | **1,640** |

Static analysis of the body's AST:

* **217 top-level statements** in the body.
* **43 parameters**, **389 distinct names assigned** in the body.
* **9 nested closures**, whose capture of enclosing locals is *narrow*:

| closure | line | enclosing locals captured |
|---|---:|---:|
| `_pip_sample_residual` | 4878 | 0 |
| `_fit_design` | (fit path) | 2 |
| `_warn_newton_unconverged` | 6223 | 2 |
| `_build_newton_mask` | 6721 | 2 |
| `_reference_input` | 4998 | 3 |
| `_support_taper` | 6533 | 3 |
| `_invert_newton_parallel` | 6439 | 6 |
| `_invert_fit` | (fit path) | 7 |
| `_pip_residual_ri` | 4846 | 8 |
| `_invert_newton` | 6253 | 8 |
| `_amp_pw_call` | (step 1) | 9 |
| `_amp_call` | (step 1) | 12 |
| `_ray_density_amp_grid` | 6587 | 14 |

So: this is a ~1,640-statement-line function with 217 top-level statements and
loosely-coupled closures, carrying 1,272 lines of inline commentary and a
773-line docstring. It is a large function. It is not a 7,000-line one, and the
usual reflex for a 7,000-line function -- "cut it into pieces, anywhere" --
would be answering a problem this file does not have.

**Growth across the campaign** (`git show <sha>:<path> | wc -l`):

| commit | subject | `_lens_traced.py` | `carrier.py` |
|---|---|---:|---:|
| `28ec3da` | pre-campaign baseline | 5,764 | 3,146 |
| `bd408bd` | D1-D7 full configuration | 6,237 | 6,389 |
| `1597a8c` | C1+C2 | 6,426 | 6,690 |
| `7f45874` | C3 | 6,426 | 7,101 |
| `d2e60ca` | C4+C5 | 6,517 | 7,261 |
| `8e7b156` | C6+C7 | 7,658 | 7,312 |
| `bb2abe7` | C8 | 7,951 | 7,312 |
| `afc3188` | v5.32.0 | 7,960 | 7,312 |
| `3753739` | C9 | 7,960 | 7,389 |
| `899050b` | C10 | **8,035** | 7,389 |

`_lens_traced.py` grew +2,271 lines (+39 %) and `carrier.py` +4,243 (+135 %).
Executable content added per layer, from the diffs (`git show <sha> --
lumenairy/`, minus comment and blank lines; the residue still includes docstring
bodies, so these are upper bounds):

| layer | lines added to `lumenairy/` | of which non-comment |
|---|---:|---:|
| C1+C2 | 617 | ~289 |
| C3 | 404 | ~260 |
| C4+C5 | 242 | ~187 |
| C6+C7 | 1,180 | ~476 |
| C8 | 290 | ~74 |
| C9 | 109 | ~40 |
| C10 | 76 | **~1** |

The audits state the same thing from the other side and more precisely: the C6
fit guard is *"+158 lines in `_lens_traced.py` and +73 in `carrier.py`, of which
the ENTIRE executable content is three lines"*; C7 is *"~55 executable lines in
two places"*; C8 is *"~30 executable lines"* of hull construction plus a ~25-line
`_support_taper` plus **one multiply**; C10 is one integer.

**This is the central finding of S1, and it reframes everything after it.** The
accretion is overwhelmingly *evidence*, not *mechanism*. What has piled up is
not ten tangled code paths -- it is ten small code paths each carrying 50-200
lines of the sweep that justified it. A refactor motivated by "the code is
unmanageable" is aiming at the wrong 39 %.

---

## S2. Layer inventory and the dependency graph

### S2.1 The switches, exactly as shipped

Behaviour-changing module-level switches with a documented fail-before, in
chronological order:

| # | layer | identifier | file:line | shipped | fail-before value | code sites |
|---|---|---|---|---|---|---:|
| 1 | F3 | `_F3_GUARD_TILTAWARE_EXPLICIT_CARRIER` | `_lens_traced.py:1288` | `True` | `False` | 1 |
| 2 | R7/F2 | `_CARRIER_FIT_RADIUS_FRAC` (+ `_CARRIER_FIT_MIN_SAMPLES`) | 1348 / 1352 | `0.5` / `64` | (gate is `carrier is not None`) | 2 |
| 3 | P2 | `_FIT_RADIUS_BEAM_FACTOR_DEFAULT`, `_APERTURE_BEAM_WARN_RATIO` | 1377 / 1381 | `2.0` / `1.5` | `fit_radius_beam_factor=None` | 3 |
| 4 | D1 | `_FIT_DISC_OUTSIDE_WEIGHT_REL` | 1458 | `1e-8` | `0.0` (hard NaN mask) | 2 |
| 5 | D7 | `_DECENTRED_FIT_POLY_ORDER` | 1520 | `10` | kwarg `= newton_poly_order` | 2 |
| 6 | C1 | `_DECENTRE_GATE_PIXELS`, `_DECENTRE_GATE_W_FRAC` | 1572 / 1573 | `0.5` / `0.05` | both `0.0` | 2 |
| 7 | C5 | `TILTED_CARRIER_EXACT_EIKONAL` | 1713 | `True` | `False` | 1 (+1 cross-module read) |
| 8 | C6 | `REMAP_STATIONARY_PHASE_LAUNCH` | 2176 | `True` | `False` | 1 gate, 3 consumers |
| 9 | C10 | `_REMAP_RESID_EIKONAL_DEGREE` | 2322 | `6` | `4` | 1 |
| 10 | C6-guard | `REMAP_STATIONARY_PHASE_FIT_GUARD` | 2610 | `False` (opt-in) | -- | 1 |
| 11 | C8 | `REMAP_INVERSE_SUPPORT_BOUND` | 2724 | `True` | `False` | 2 |
| 12 | C7 | `RAY_DENSITY_HALO_CHECK` | 445 | `'warn'` | `'silent'` | 2 |
| 13 | C9 | `SPHERE_PARAB_CONVERSION_EXACT` | `carrier.py:2032` | `True` | `False` | 1 |

Supporting calibration constants that are *not* era switches but are read by the
same layers: `_SUPPORT_BOUND_FEATHER_CELLS` (2767), `_REMAP_RESID_FREEZE_MARGIN`
(2416), `_REMAP_RESID_FREEZE_MAX_W` (2419), `_REMAP_RESID_DEGREE_CAP` (2425),
`_REMAP_RESID_FIT_W` (2352), `_REMAP_RESID_BRIGHT_FRAC` (2324),
`_REMAP_RESID_MAX_STEP_RAD` (2328), `_REMAP_RESID_MIN_SAMPLES_PER_TERM` (2330),
`_RD_HALO_AMP_CONTOUR` / `_RD_HALO_RADIUS_FACTOR` / `_RD_HALO_AMAX_TOL`
(438-440), `_AUTO_CARRIER_NYQUIST_FRAC` / `_MIN_CORE` / `_ALIAS_FRAC`
(1309-1323), `_TILT_EIKONAL_MIN_RAD` (1280), `_NONCOLLIMATED_RESID_THRESH`
(1268).

`carrier.py` carries a second, disjoint guard family from the C2/C3 era --
`on_gap_paraxial`, `on_decentred_fit`, `on_na_proximity`, `on_tilt_exact_grid`,
`on_chain_entry_congruence` -- which are per-call `{'error','warn','ignore'}`
kwargs routed through one validator (`_check_guard_action`, `carrier.py:2325`)
and one dispatcher (`_guard_dispose`, 2309). **These are already the
consolidated form the brief asks about, one layer up.** They are diagnostics
that never change a bit (C3's commit records this explicitly: *"Diagnostic only
-- bitwise identical across all settings, verified by monkeypatching the whole
added path out"*). They are not part of the problem and should not be pulled
into any consolidation of the numeric switches.

### S2.2 The dependency graph

The layers are *not* a chain, and this is the fact that governs S5.

```
                       carrier resolution (S5.1 / N5 / R7 / F3)
                                    |
                        +-----------+-----------+
                        |                       |
              C5  exact tilted eikonal          |
              (TILTED_CARRIER_EXACT_EIKONAL)    |
                        |                       |
                        v                       v
              C6  stationary-phase launch <--- gate: preserve_input_phase=='remap'
              (REMAP_STATIONARY_PHASE_LAUNCH)         AND _r7_carrier_path
                        |
                 C10 residual degree 4->6
                 (_REMAP_RESID_EIKONAL_DEGREE)
                        |
                        |  needs freeze radius to clear the fit disc
                        v
   R7/F2 --> P2 --> D1 --> D7 --> C1 -----> ray-fit domain -------> C6-guard
   (radius) (beam) (weights)(order)(gate)   (5253-5372, 5855-5994)  (opt-in)
                        |                                              |
                        |                                              |
                        v                                              v
              forward-map fit  ------> Newton inverse ------> C8 support bound
                                                             (REMAP_INVERSE_SUPPORT_BOUND)
                                                                       |
                                                                       v
                                                             C7 halo self-check
                                                             (RAY_DENSITY_HALO_CHECK)
```

Edges that the audits state explicitly and that a refactor must preserve:

1. **C5 -> C6 (magnitude, not existence).** C5 changed the reference wavefront,
   which grew the input residual's own slope: `grad a` rms 1.46 -> 2.30 mrad,
   exit WFE 0.036 -> 0.089 waves, ratio 2.48 against the predicted
   `(2.30/1.46)^2 = 2.48`. C6 exists to cancel exactly that quadratic term.
2. **C6 -> D1 (precondition destruction).** `_FIT_DISC_OUTSIDE_WEIGHT_REL`'s
   note states the precondition for the concentric hard mask: *"the
   unconstrained directions of the fit inherit the map's RADIAL SYMMETRY, the
   extrapolation outside the disc stays MONOTONE, and the Newton inversion
   cannot find a second root."* C6 augments every launch by `grad(a_fit)` of a
   general non-radial polynomial and therefore **destroys that precondition on
   the one branch D1 left alone** (the concentric one). This is the single most
   important cross-layer edge in the whole campaign.
3. **C6 -> C6-guard -> (superseded by) C8.** The guard was the first response to
   (2); C8 is the structural one. The C8 audit records that C8 *"matches the
   guard's conservation result to five decimals"* while *"COSTING NO EE"*, that
   the guard regresses two of six synthetic fixtures and C8 regresses none, and
   that C8 fixes (-2,0)/(-3,0) which the guard *structurally cannot* (their last
   group sits at 0.481 w / 0.723 w, already on the off-centre weighted branch
   where the guard is inert by construction).
4. **C6 -> C10 (unblocked by C8).** `_REMAP_RESID_EIKONAL_DEGREE`'s note is
   explicit: the only thing holding the degree at 4 was a degree-6 ghost, and
   *"the counter-evidence was real and it is now bounded, so the reason for 4 is
   spent."* With C8 off at degree 6 the chain manufactures 5.2 % of the input
   power; with C8 on, degree 6's conservation and halo are within noise of
   degree 4's.
5. **C8 is NOT downstream of C6.** The C8 audit's clearest evidence: at `rs=2`
   on (-4,-2) it is **C6-OFF** that violates the halo criterion (28x/78x over),
   and C8 repairs that row too. *"The mechanism was never C6's; C6 only made it
   large enough to see."*
6. **C8 -> C7 (monotone).** The halo check can only go quieter under the bound
   (`amax_halo` is a max of `|E_out|` outside a radius the bound does not move,
   and the bound only lowers `|E_out|`). No new firing anywhere.
7. **C6 freeze -> ray-fit disc (an ordering constraint across 480 lines).**
   `_REMAP_RESID_FREEZE_MARGIN = 1.25` exists because the residual model's radial
   freeze circle must sit strictly *outside* the ray-fit disc; with the two
   coincident the polynomial and spline backends stop describing the same map
   (skirt error 5.608 um vs 0.006 um; a 130x step across the ray-fit disc radius
   and nowhere else). This is why `_fit_r_about_beam` is resolved at line 5372
   -- 480 lines before the restriction it drives is applied at 5905 -- and the
   source says so in a comment at 5355-5358 and again at 5902-5904.
8. **C9 is disjoint** from all of the above. It lives in `carrier.py`, changes a
   conversion the element consumes, and its interaction with the element is only
   through the input field.

### S2.3 Two pieces of drift found, both in the prose graph

**(a) A dangling cross-reference.** `TILTED_CARRIER_EXACT_EIKONAL`'s docstring
(`_lens_traced.py:1707`) cites
`:data:`~lumenairy.propagators.carrier.CHAIN_EXACT_TILTED_REFERENCE``. **No such
symbol exists anywhere in the repository.** The actual mechanism is
`carrier._exact_tilt_reference()` (`carrier.py:1855-1868`), a function that
imports and returns the element's flag at call time. A broken Sphinx
cross-reference, not a behaviour defect.

**(b) A stale "still open" entry, verified against source.** The roadmap
(`ROADMAP_DESIGN121_FULL_CONFIGURATION_2026_07_27.md`, "Still open
(2026-07-30)") records that C3 converted the chain's group step to an exact
chief-ray trace but left `_chain_chief_ray_at_target` on the lumped paraxial
ABCD, quoting a 12.372 um error on the D6 stand-in. `CHANGELOG.md` says the
opposite. **I read the source: the CHANGELOG is right and the roadmap is
stale.** `carrier.py:6455-6461` calls `_group_chief_transfer` with the comment
*"niche C3: the SAME exact chief-ray trace the chain itself uses ... these two
closures must agree to the digit -- `propagate_traced_carrier_chain_multi`
cross-checks them and RAISES on a mismatch -- so converting the chain's group
step without converting this one is a hard break, not a drift."*

Item (b) matters twice over. It is the campaign's own record of the multi-site
hazard (S3.1), **and** the record of it is wrong in one of the two places a
reader would look -- which is item (a)'s failure mode again. Together they are
the evidence for S6.4: the prose graph that binds these layers is maintained by
hand, it carries the load, and it has started to fray in both directions (a
reference to something that does not exist, and a claim that something is open
when it is closed).

---

## S3. The unifying structure -- three abstractions, honestly assessed

### S3.1 UNIT A -- the entrance-eikonal jet (C5 + C6 + C10, and H6 before them)

**The claim.** `apply_real_lens_traced` needs one thing from the input side: the
*total entrance eikonal* `Phi(x,y) = W(x,y) + a(x,y)` and its transverse
gradient, evaluated **consistently** at three sites. It currently obtains this
from two separate objects, summed by hand at each site.

**Evidence that this is one idea and not three.** The source already says so.
`_lens_traced.py:4836-4838`:

> "the H6 entrance-eikonal term carries `W + a_fit`, and what the residual
> phasor transports is the LEFTOVER `exp(i k0 (a - a_fit))` -- **the three
> halves of one substitution `W -> W + a_fit`** of the launch congruence."

The three sites:

| # | site | line | current code |
|---|---|---:|---|
| 1 | launch direction | 5503, 5522-5523 | `L_in, M_in = _carrier_grad(h_x, h_y)`; then `_gLa,_gMa = _resid_eik.grad(...)`; `L_in = L_in + _gLa` |
| 2 | H6 entrance eikonal added to OPL | 5644, 5655 | `final.opd += _carrier_W_fn(h_x, h_y)`; then `final.opd += _resid_eik.value(h_x, h_y)` |
| 3 | residual de-chirp of the transported phasor | 4849, 4866 | `_r = E_in * exp(-1j*k*_pip_remap_W)`; then `_r *= exp(-1j*k*_resid_eik.value(...))` |

At every one of the three sites the pattern is identical: *take the carrier
term, then if `_resid_eik is not None` add the residual term.* Three
independent `if _resid_eik is not None:` blocks, 800 lines apart, each of which
must be kept in step by hand. **This is exactly the "fix lands in 2 of 3 sites"
hazard the brief names**, and it is currently mitigated only by three comments
that each explain why the other two must agree.

**The precedent is real, and I verified it at source.** C3 hit the same class
one module over: the chain's chief-ray closure existed in two places
(`_group_chief_transfer`, `carrier.py:3404`, and `_chain_chief_ray_at_target`,
`carrier.py:6388`). The fix had to land in both, and what shipped as the
mitigation is a *runtime cross-check that raises on mismatch*, documented in the
second closure's own comment (`carrier.py:6455-6461`):

> "niche C3: the SAME exact chief-ray trace the chain itself uses ... These two
> closures must agree to the digit -- `propagate_traced_carrier_chain_multi`
> cross-checks them and RAISES on a mismatch -- **so converting the chain's
> group step without converting this one is a hard break, not a drift.**"

That is an admission in code that duplication had made correctness
un-guaranteeable by inspection: the authors could not rely on a future editor
finding both sites, so they made the program check at runtime. UNIT A's three
sites are in exactly that position today, minus the runtime check -- and unlike
the chief ray, the three eikonal sites produce no single comparable scalar to
cross-check, so the runtime-guard escape hatch is not available. **Composition
is the only structural answer available here.**

The cost of *not* having it is visible in the same campaign in a different form.
Niche C4 was a one-character defect -- an amplitude mask built `(y, x)` and
ravelled against an `indexing='ij'` launch grid, fixed by a single `.T` at
`_lens_traced.py:5691` whose comment now reads *"the `.T` is load-bearing"*. It
survived because *"a rotationally symmetric beam is invariant under that swap"*,
and it mattered because `_exit_na_out` feeds the chain's `on_tilt_exact_grid`
routing: design 121's last group reported `na_exit` 0.3633 against a
transpose-immune 0.2912, 25 % overstated. That is the same failure mode as the
three-site invariant -- **a relationship that is true by circumstance rather
than by construction, and therefore silent when the circumstance changes.**

**The abstraction.**

```
class _EntranceEikonal:            # value/gradient of one scalar field
    def value(self, x, y) -> ndarray
    def grad(self, x, y) -> (ndarray, ndarray)
    grid: ndarray                  # the (N,N) evaluation used by the reference leg
    diag: dict
```

with two constructors and one composition:

* `carrier_jet(carrier, ...)` -- today's `_compute_carrier` (1866-2111) and
  `_tilted_carrier_parts` (1820-1863). **C5 is a branch inside the tilted
  constructor and nowhere else** -- it already is (`1841`).
* `residual_jet(E_in, W, ...)` -- today's `_fit_residual_eikonal` (2892-3044)
  returning `_ResidualEikonal` (2770-2889). **C10 is one integer read once
  inside it** -- it already is (`2999`).
* `Phi = carrier_jet + residual_jet` -- a `_SumEikonal` whose `.value` and
  `.grad` add. **C6 becomes "was a residual term composed in?" and nothing
  else.**

The three sites then read:

```
L_in, M_in = Phi.grad(h_x, h_y)        # site 1, no branch
final.opd += Phi.value(h_x, h_y)       # site 2, no branch
_r = E_in * exp(-1j * k * Phi.grid)    # site 3, no branch
```

**What this buys, precisely.** It does not improve any number. It converts a
three-site invariant maintained by comment into a one-site invariant maintained
by construction. Every future correction to the entrance eikonal (a next-order
term, a per-order carrier, a vector eikonal) lands in one constructor instead of
needing an author to find three summation sites. Given that this exact class of
defect has already occurred once in this campaign (C3) and is already flagged
three times in comments, that is a real, nameable risk reduction.

**What it does NOT buy, and I want to be exact about this.** It does not remove
C5, C6 or C10 as switches. C5 is a genuinely different reference wavefront that
the chain must agree with (`carrier._exact_tilt_reference()` reads the element's
flag at call time precisely so a MIXED pair is impossible); C10 is a genuinely
different model order; C6 is a genuinely different launch. All three remain
observable, all three keep their fail-before. The abstraction changes *where the
switch is read*, from three sites to one, not *whether it exists*.

**Cost.** The composition is ~40 lines. The three call sites lose ~25 lines
between them. But `_resid_eik` also feeds `_c6_fit_guard` at 5933 and the
`_remap_launch_out` diagnostic at 5398-5404, and its `.diag` is mutated
mid-flight at 5518 -- so the sum object needs to expose "does a residual term
exist" and a merged `.diag`. Call it ~120 lines net of change to executable
code, and roughly 400 lines of prose to re-site (the C5, C6 and C10 notes all
currently live above module constants that would keep their homes, but the three
in-line explanations at 4851-4860, 5507-5514 and 5646-5654 all become one).

### S3.2 UNIT B -- the ray-fit domain policy (R7/F2 + P2 + D1 + D7 + C1 + C6-guard)

**The claim.** Six layers answer one three-part question: *which traced samples
enter the entrance->exit fit, with what weights, at what polynomial order?*

**Evidence.** They already share one code block. Resolution is at 5261-5372
(112 lines: `on_aperture_beam` validation, `decentred_fit_poly_order`
validation, `beam_centre` resolution, the C1 two-stage null-decentre gate, the
beam-radius measurement, `_beam_fit_radius`, the P2 warn, `_fit_r_geom`,
`_fit_r_max`, `_fit_r_about_beam`). Application is at 5855-5994 (140 lines: the
disc, the off-centre intersection, the C6 guard predicate, the weighted-vs-mask
branch, the D7 order step-down loop, the abandonment warning). Between them sit
480 lines of unrelated work (launch grid, ray trace, exit-vertex correction, H6,
NA guard, reshape, and the two support hulls) -- and the split is *forced*, by
the C6 freeze-radius ordering constraint (S2.2 edge 7), which the source
documents at both ends.

The state that must survive that 480-line gap: `_frbf`, `_dec_order`, `_bcx`,
`_bcy`, `_beam_decentred`, `_w_in_beam`, `_beam_fit_radius`, `_fit_r_geom`,
`_fit_r_max`, `_fit_r_about_beam` -- ten locals, all of which are pure functions
of the inputs and none of which is touched in between.

**The abstraction.**

```
class _RayFitDomain:               # built once, at 5261; consumed at 5905 and 5395
    beam_centre: (float, float)
    beam_radius: float
    decentred:   bool             # C1 gate applied
    radius_geom: float | None     # R7/F2
    radius_beam: float | None     # P2
    radius_about_beam: float      # what C6's freeze must clear
    def restrict(self, xs_in, x_out, y_out, opl, base_order, c6_engaged)
        -> (x_out, y_out, opl, weights|None, poly_order, why)
```

Every layer becomes an *input* to one constructor and one method:

| layer | becomes |
|---|---|
| R7/F2 | `radius_geom = _CARRIER_FIT_RADIUS_FRAC * launch_radius if carrier else None` |
| P2 | `radius_beam = min(frbf * w, launch_radius)`; the warn is a method |
| D1 | `restrict()` returns weights instead of a mask when `decentred` |
| D7 | `restrict()` raises the order on exactly that branch, with the step-down |
| C1 | the two-stage gate that sets `decentred` |
| C6-guard | one extra `or c6_engaged` in the weighted-branch predicate |

**What this buys.** The 480-line separation stops being a hazard: the object is
constructed once and is immutable, so nothing can be resolved late or read
early. The `why` string (currently built in two places at 5907 and 5914 and used
once at 5983) gets one home. And the branch structure -- which the C6 fit-guard
audit had to *measure* to discover was **per-group, not per-order** -- becomes
inspectable from one place instead of inferred from a predicate spread over
5317, 5323-5330, 5908 and 5936-5938.

**Two live open defects sit inside this unit**, and both argue for giving it a
home rather than leaving it as a predicate:

* **The order-10 anomaly** (`D121_RESIDUAL_CLOSURE` S7 item 1). Sweeping
  `_DECENTRED_FIT_POLY_ORDER`, orders 6, 8 and 12 all close the (-1,0) chain
  residual (-0.066 / -0.051 / -0.017 points) and **the shipped 10 does not**
  (+0.934); at (-4,-2), order 6 helps by 0.10 and order 8 *hurts* by 0.08. The
  audit's own reading: *"A quantity that is good at 6, good at 8, bad at 10 and
  good again at 12 is not an approximation error converging in a degree --
  something discrete is happening at 10 on this geometry, and this study did not
  find out what. It is a second, independent thing wrong in the same
  neighbourhood, and it is untouched."*
* **The C1 gate sits 10-14x below its own measured crossover.** Forcing the
  branch puts the concentric/off-centre crossover between **0.48 w and 0.72 w**
  on design 121, while `_DECENTRE_GATE_W_FRAC` switches at **0.05 w**. Raising
  it would recover 0.89 points at (-1,0) and 0.69 at (-2,0), and was **not
  attempted** because the crossover is design-dependent and the other branch
  fails catastrophically above it (concentric-branch residuals reach 9.77 /
  19.93 / 11.25 points at (-3,0)/(-4,0)/(-4,-2) -- *"D1's own failure mode,
  reproduced end to end, and it is why the branch exists"*).

Both are properties of *one policy*, and today that policy has no object to
carry them, no place to record "the gate's own measurement says it is
conservative by 10x", and no single site at which the order anomaly could be
instrumented.

A third item is smaller but is the clean signature of layering: after C10 raised
`_REMAP_RESID_EIKONAL_DEGREE` to 6, `_REMAP_RESID_DEGREE_CAP` is **equal to the
default**. The raise consumed the entire headroom the cap existed to provide,
and the cap's own justification text was written when the default was 4. It is
now vestigial and says so nowhere.

**What it does NOT buy.** Nothing numerical. And there is a genuine
counter-argument: the resolution block is *deliberately* early because it must
raise on bad kwargs "before any ray work, like every other guard above"
(comment, 5279-5280). An object constructed at 5261 preserves that; an object
constructed lazily would not. This is easy to get right and easy to get wrong.

### S3.3 UNIT C -- the traced exit support (C7 + C8 + the direct-fit hull)

**The claim.** There are currently **three** notions of "the region the traced
rays reached", computed from the same arrays, at nearly the same point in the
function, by three different rules -- and the audits already record that the
inconsistency has a measurable consequence.

| # | notion | line | rule |
|---|---|---:|---|
| 1 | C7 halo radius | 5775-5795 | amplitude-weighted centroid + max radius over samples above the `e^-9` amplitude contour, then x1.25 |
| 2 | C8 support hull | 5802-5853 | convex hull of alive **stop-passing** landings, + `sqrt(2)*sub*dx` plateau, + 1 exit-lattice-cell feather |
| 3 | direct-fit hull | in the `inversion_method='fit'` block, 5996-6062 | the fit path's own long-standing exit hull mask |

Both (1) and (2) are taken at the same place for the same two stated reasons --
the source comments say so almost verbatim at 5761-5767 and again at 5797-5801:
*"between the alive mask and the fit-domain restriction below"*, because the fit
restriction NaNs samples that are still good optics, and because this is the
last point at which `x_out_grid` is the exact traced map.

The C8 audit's own architectural sentence is the strongest evidence that (2) and
(3) are the same idea implemented twice: *"This bound gives the Newton path the
containment the direct-fit path has had all along."*

And the inconsistency between (1) and (2) has a measured consequence, recorded
in `RECON_PINS_POST_C8_2026_08_01.md`: on the E-M6 fixture C8's deliberately
retained plateau+feather band (outer edge 1.4996 mm) sits *inside* C7's halo
radius `1.25 * r_hull`, and the energy check's reading (1.01931) sits inside its
band -- so **the two self-checks are jointly blind to 0.19998 of `P_ap` of
manufactured light**, and the field's global maximum is in that band. Separately,
`ORACLE_ENERGY_AND_D6_HALO_2026_08_01.md` records that C7's `e^-9` gate
understates the true support (1.6161 mm against 1.8115 mm over all alive rays)
and that the two D6 call labels were swapped in the C7 record.

**The abstraction.**

```
class _TracedExitSupport:          # built once from the exact landings, 5761
    centroid: (float, float)
    radius:   float                # for the halo annulus
    hull:     (A, b) | None        # convex, stop-passing
    pitch:    float                # median exit separation of adjacent rays
    def taper(self, Xg, Yg, feather, plateau) -> ndarray   # C8
    def outside(self, X, Y, factor) -> ndarray             # C7 annulus
```

C7 becomes "report when the returned field claims amplitude outside it"; C8
becomes "do not let the amplitude model claim it"; the direct-fit hull becomes
the same `.hull`. One object, three consumers, one set of conventions.

**What this buys.** It makes the joint-blindness finding a *parameter* rather
than an emergent accident: with one object, "C7's reporting radius must exceed
C8's retained band" is a one-line invariant that can be asserted. Today it is a
relationship between an `e^-9` amplitude contour times 1.25 and a
`sqrt(2)*sub*dx` plateau plus one lattice cell, computed 40 lines apart, that
nobody noticed until an adversarial re-check on a pathological fixture.

**Why this unit has to exist at all, in the campaign's own words.** The two
metric families are blind in *opposite* directions, and each blindness was
discovered the hard way:

* **Energy and EE cannot see a halo.** C6's on-axis production call reports
  `P_out/P_ap` = 1.000741 -- 0.486 % of the input power manufactured, deposited
  at 4-8 mm at 83 % of peak where the exact trace permits 3.6e-10 -- **while the
  same field reports +1.691 EE3 points**. Worse, `APPROXIMATION_AUDIT_POST_C6`
  found a configuration where 82 % and 121 % of manufactured energy reads as a
  **0.005-point EE3 change**, and concluded that *"every 'converged' verdict in
  both audits that rests on them inherits that blind spot."*
* **Conservation and halo cannot see a destroyed spot.** C10's forced-concentric
  arm scores **6 of 6 on every conservation and halo bound at (-4,0)** --
  `P_out/P_in` 0.994016, `g4` 7.700e-11, `amax4` 2.175e-05, `r_rms` 0.8372 --
  **while losing 19.9 EE3 points in the image**. The audit's conclusion: *"A
  field can pass every conservation and halo bound while its spot is destroyed,
  and the only instrument that sees it is the one that looks at the image."*

So the support object is not a convenience. It is the carrier of the *only*
observable that closes the first blind spot, and the reason the library needs
three currencies rather than one. Giving the three notions of "support" one home
is what lets the relationship between them be stated instead of measured.

**What it does NOT buy, and this is the honest caveat.** The three rules are
*not* interchangeable and unifying them is a behaviour change, not a
refactor. C7's radius is amplitude-weighted on purpose (it is a *reporting*
radius calibrated over 180 element calls, with a measured 123x separation
between clean and defective populations at factor 1.25); C8's is a convex hull
of stop-passing rays on purpose (convexity *"can only make the bound LOOSER,
never tighter, so it cannot manufacture a cut"*). Merging them would re-open a
calibration that cost 177 readings. **The right move is one object with three
named views, not one rule.**

### S3.4 What does NOT unify, and should not be forced to

* **C9** (`SPHERE_PARAB_CONVERSION_EXACT`, `carrier.py:2032`) is not a feature
  and not a member of any of the three units. It is the *removal* of an
  approximation -- a `cos^2` band-limit taper that was multiplied onto an
  otherwise-exact conversion. Once removed, the flag gates nothing but the
  historical taper. Its natural home is the existing `carrier_reference`
  convention ('sphere' | 'parabola'), which is already a first-class kwarg; the
  taper should become an explicitly-legacy path with a deprecation horizon, not
  a co-equal switch.

  **The second `cos^2` taper, and why I am not calling it the same defect.**
  `carrier._tilt_exactness_phase` (1871, the C5 chain-side helper) applies
  *"the same `cos^2` roll-off `_sphere_parab_conversion` uses"* over its own
  band-limit radius, and `D121_RESIDUAL_CLOSURE` S7 item 8 lists it as unpriced
  (`rc_c5taper_121.py` written, not run; onset quoted at 2.5 beam radii for the
  group-6 exit at (-4,0)). Read as "the same defect exists one function over,
  unfixed", that would be the strongest single argument in this document for a
  conversion-invariant abstraction. **It does not survive reading the source.**
  The function's own docstring carries the structural difference and the
  measurement: *"It is applied IDENTICALLY on the `+1` and `-1` calls, so the
  entrance/exit round trip is exact (to 2e-16) whatever the taper does."* That
  is precisely the property C9's taper lacked -- C9's whole defect was that the
  entrance and exit conventions bracketing a group *disagreed*, which is why
  three of its four live calls were worse ablated alone than left alone. And it
  is measured, not argued: `probe_c5_byte_identity.py` part (c), design 121 at
  (-4,-2), reports that removing this taper entirely moves the result by
  **2.3e-5** and widening it 1.5x by the same 2.3e-5, while halving it moves the
  result by 0.13 -- i.e. converged at and above the shipped radius.

  What remains open is narrower and worth stating exactly: the docstring's onset
  figure (9.97 mm = **3.2 beam radii** on design 121's coarse grid) and the
  residual-closure's (**2.5 beam radii**, group-6 exit at (-4,0)) do not agree,
  and the per-call census that would reconcile them is written and unrun. That
  is a one-run item, not an architecture item -- but it is a good example of why
  a conversion object with one taper policy and one onset calculation would be
  worth having: today the same roll-off is implemented twice with two
  independently-derived onsets and two separately-maintained records of them.
* **C3** (exact chief-ray tracing, the gap-paraxial guard) is chain-side, is
  bitwise-neutral in its guard half, and its trace half is already properly
  factored into `_group_chief_transfer` with an ABCD fallback and a runtime
  cross-check. Nothing to consolidate.
* **The `on_*` guard family** is already the consolidated form (S2.1).
* **The five beam-radius implementations.** `_input_beam_amp_radius`
  (`_lens_traced.py:1576`), `carrier._envelope_amp_radius` (561),
  `carrier._axis_amp_radius` (824), `carrier._gap_amp_radius` (3758),
  `carrier._chain_envelope_stats` (4214). This looks like the textbook
  duplication finding and it is **mostly not one**: `_axis_amp_radius` is the
  1-D marginal analogue; `_gap_amp_radius` is a separable formulation that
  exists because the full meshgrid would be 6.6 GB each at the design-121
  production grid (N=28672) and it runs on *every* inter-group leg;
  `_input_beam_amp_radius` accumulates in row bands to avoid a 0.5 GiB temporary
  at N=8192. Each has a stated, load-bearing reason. **But they carry three
  different centring policies** -- `_envelope_amp_radius` takes an explicit
  `centre` (added by D6), `_chain_envelope_stats` subtracts the centroid
  implicitly, `_gap_amp_radius` measures about the grid origin -- and the
  element-side one needed a C1-era fix (measure about the beam, not the origin)
  that the others did not, because the chain works in a chief-ray-tracking
  frame where the origin *is* the beam. That is correct today and it is exactly
  the kind of invariant that is true by circumstance rather than by
  construction. Worth a one-line note in each docstring stating which frame it
  assumes; not worth a merge.

---

## S4. The proposed decomposition, with the actual seams named

### S4.1 The current phase map of `apply_real_lens_traced` (3442-7258)

| # | phase | lines | statements | notes |
|---|---|---|---:|---|
| 0 | signature + docstring | 3442-4260 | -- | 43 params, 773-line docstring |
| 1 | kwarg/enum validation, opt-in dispatch decisions | 4261-4543 | ~55 | N12/remap/N13/N16 resolution; folded-design guard; pre-flight grid check |
| 2 | multibranch (K1) early dispatch | 4544-4616 | ~6 | whole-call reroute |
| 3 | stop handling, row-banding, spline order | 4617-4696 | ~12 | |
| 4 | **carrier resolution** (S5.1 / N5 / R7 / F3) | 4697-4838 | ~25 | produces `_carrier_W`, `_carrier_grad`, `_carrier_W_fn`, `_r7_carrier_path` |
| 5 | residual-phasor closures | 4839-4997 | 2 defs | `_pip_residual_ri`, `_pip_sample_residual` |
| 6 | reference input | 4998-5002 | 1 def | |
| 7 | Step 1: amplitude leg (double `apply_real_lens`) | 5003-5220 | ~30 | includes the parallelism decision |
| 8 | **launch geometry + fit-domain RESOLVE + C6 residual fit** | 5221-5405 | ~35 | **UNIT B (resolve) + UNIT A (residual jet)** |
| 9 | subsampling guardrail + launch grid | 5406-5485 | ~15 | |
| 10 | **ray launch + trace** | 5486-5597 | ~14 | **UNIT A site 1** at 5503/5522 |
| 11 | **exit-vertex correction + H6 + C6 eikonal** | 5598-5656 | ~10 | **UNIT A site 2** at 5644/5655 |
| 12 | exit-NA Nyquist guard, reshape, axis reference | 5657-5760 | ~20 | |
| 13 | **C7 halo hull** | 5761-5795 | ~12 | **UNIT C** |
| 14 | **C8 support bound** | 5797-5853 | ~18 | **UNIT C** |
| 15 | **fit-domain restriction APPLY** | 5855-5994 | ~20 | **UNIT B (apply)** |
| 16 | direct-fit inverse path (T-P2) | 5996-6062 | ~12 | contains the third hull |
| 17 | forward-map fits, magnification stencil, knots | 6063-6251 | ~30 | |
| 18 | Newton machinery | 6253-6531 | 3 defs | `_warn_newton_unconverged`, `_invert_newton`, `_invert_newton_parallel` |
| 19 | **`_support_taper`** | 6533-6585 | 1 def | **UNIT C** |
| 20 | `_ray_density_amp_grid` | 6587-6719 | 1 def | applies the taper at 6684-6685 |
| 21 | `_build_newton_mask`, coarse Newton, upsample | 6721-6901 | ~35 | |
| 22 | ray-density amplitude on the wave grid | 6903-7002 | ~20 | |
| 23 | Step 3: combine amplitude with geometric phase | 7003-7124 | ~20 | |
| 24 | masking, magnitude swap, energy self-check, **C7 halo check**, return | 7125-7258 | ~25 | **UNIT C** at 7192-7254 |

### S4.2 The extraction, phase by phase

**Pure-extraction candidates (no state-object needed), in order of ratio:**

| candidate | current lines | captures | target |
|---|---|---:|---|
| `_support_taper` + hull construction + halo hull + halo check | 5761-5795, 5797-5853, 6533-6585, 7192-7254 | 3 | module-level `_TracedExitSupport` (UNIT C) |
| fit-domain resolve + apply | 5261-5372, 5855-5994 | 10 | module-level `_RayFitDomain` (UNIT B) |
| entrance-eikonal composition | 5503/5522, 5644/5655, 4849/4866 | -- | module-level `_SumEikonal` (UNIT A) |
| Newton family | 6223-6531 | 2, 8, 6 | module-level `_NewtonInverter(spline_data, bound, ...)` |
| amplitude leg | 5003-5220 | 12, 9 | module-level `_amplitude_leg(...)` |

The four already-narrow closures (`_support_taper` 3, `_warn_newton_unconverged`
2, `_build_newton_mask` 2, `_reference_input` 3) are trivially liftable today.
`_invert_newton` (8) and `_invert_newton_parallel` (6) are liftable with one
small `_NewtonState` namedtuple. `_ray_density_amp_grid` (14) is the hardest and
should be lifted last.

### S4.3 The seam that does not exist

**There is no clean cut in the middle third.** Live-variable analysis across
every top-level statement boundary in the body (number of names assigned at or
before that point that are still read after it):

| boundary line | live vars crossing |
|---:|---:|
| 4605 | 50 |
| 4985 | 62 |
| 5178 | 57 |
| 5397 | 61 |
| 5583 | 62 |
| 5737 | 57 |
| **5994** | **52** |
| 6206 | 56 |
| **6531** | **52** |
| **6585** | **50** |
| 6686 | 41 |
| 6736 | 40 |
| 6901 | 36 |
| 6999 | 26 |
| 7101 | 21 |
| 7124 | 16 |
| 7254 | 3 |

The count starts at 43 (the parameters), climbs to 68 by line 4904, and **never
drops below 50 until line 6585** -- two-thirds of the way through. The three
local minima at 5994, 6531 and 6585 are the best cut points in the body and they
still require ~50 live names to cross.

**What this means concretely.** Splitting `apply_real_lens_traced` into
sequential sub-functions requires either (a) a ~50-field context object threaded
through every stage, or (b) sub-functions with 30-50 parameters each. Both are
worse than what is there now for a *reader*, and (a) in particular is the
classic refactor that trades one large function for one large struct plus five
functions that all mutate it -- no reduction in coupling, a net loss in
locality, and a new class of bug (stage ordering) that the current linear form
cannot have.

**So the decomposition I propose is object extraction, not phase splitting.**
Pull out UNIT A, UNIT B, UNIT C and the Newton family as module-level classes
with explicit constructors. That reduces the body's 389 assigned names, narrows
the widest closures, and puts each layer's *mechanism* next to its *note* --
without ever needing a cut through the middle third.

### S4.4 What the body would look like afterwards

Phases 8, 10, 11, 13, 14, 15, 19 collapse from ~130 executable lines to roughly:

```
    fit_domain = _RayFitDomain.resolve(E_in, dx, dy, launch_radius,
                                       carrier, beam_centre, fit_radius_beam_factor,
                                       decentred_fit_poly_order, on_aperture_beam,
                                       r7_carrier_path=_r7_carrier_path)
    Phi = _carrier_jet(carrier, ...)
    if _pip_remap and REMAP_STATIONARY_PHASE_LAUNCH and _r7_carrier_path:
        Phi = Phi + _residual_jet(E_in, Phi.grid, ..., fit_domain.radius_about_beam)
    ...
    L_in, M_in = Phi.grad(h_x, h_y)                      # site 1
    ...
    final.opd += Phi.value(h_x, h_y)                     # site 2
    ...
    support = _TracedExitSupport.from_landings(x_out_grid, y_out_grid, _amp,
                                               xs_in, aperture, dx, sub)
    x_out_grid, y_out_grid, opl_grid, _fit_weights, _fit_poly_order = \
        fit_domain.restrict(xs_in, x_out_grid, y_out_grid, opl_grid,
                            newton_poly_order, c6_engaged=Phi.has_residual)
```

Estimated body reduction: ~330 executable lines and ~250 comment lines move out
of the function into four module-level classes; the body drops from ~1,640 to
~1,310 statement lines, and the widest closure capture (`_ray_density_amp_grid`,
14) drops to ~11 (it stops capturing `_sup_bound`).

### S4.5 An honest note on what stays hard

Even after all four extractions, `apply_real_lens_traced` retains 43 parameters,
~1,310 statement lines and a ~50-variable state envelope in its middle. It is
still a large function. The extractions make the *layers* tractable; they do not
make the *function* small. Anyone proposing this work should say so up front,
because "we refactored it and it is still 1,300 lines" is a predictable and fair
criticism if the goal was stated as size reduction.

---

## S5. The flag / fail-before debt, and the single-era-switch proposal

### S5.1 What the debt actually is

* 13 behaviour-changing module-level switches across two modules (S2.1).
* **37 flag-assignment sites in `tests/`** and roughly 80 more in
  `validation/repro_traced_carrier_121/`, by identifier:
  `REMAP_STATIONARY_PHASE_LAUNCH` 35, `REMAP_INVERSE_SUPPORT_BOUND` 21,
  `REMAP_STATIONARY_PHASE_FIT_GUARD` 16, `TILTED_CARRIER_EXACT_EIKONAL` 15,
  `_REMAP_RESID_EIKONAL_DEGREE` 14, `_DECENTRE_GATE_W_FRAC` 5,
  `_DECENTRE_GATE_PIXELS` 4, `_FIT_DISC_OUTSIDE_WEIGHT_REL` 3,
  `_DECENTRED_FIT_POLY_ORDER` 1, `RAY_DENSITY_HALO_CHECK` 1.
* Every one of them is a raw `setattr` on a module global, saved and restored by
  hand. **The discipline is uniform and correct** -- I checked all 37 test sites
  and every set is inside or immediately before a `try:` whose `finally:`
  restores; the 11 that my heuristic flagged as unprotected are the `finally:`
  bodies themselves. `validation/` has a `Patch` context manager
  (`approx_common.py:71-88`) that does the same thing properly.
* **96 prose cross-references** between the constants' notes
  (```NAME``` and `:data:` forms), concentrated on
  `_FIT_DISC_OUTSIDE_WEIGHT_REL` (12), `_REMAP_RESID_FREEZE_MARGIN` (7),
  `REMAP_INVERSE_SUPPORT_BOUND` (6), `REMAP_STATIONARY_PHASE_LAUNCH` (5),
  `_RD_HALO_AMAX_TOL` (5), `_DECENTRED_FIT_POLY_ORDER` (5). One is already
  dangling (S2.3).

So the debt is: **boilerplate, a hand-maintained prose graph, and
non-thread-safety** -- not leakage. CI runs serially (pytest-split shards across
runners; `unit-tests.yml` explicitly notes *"A single runner has only ~2-4
cores, so in-process xdist..."*), so nothing is broken today. But `pytest-xdist`
is a declared dev dependency, and a developer running `-n auto` locally would
get cross-worker contamination on any test that mutates these globals. That is
a real, if bounded, hazard.

### S5.2 The `TRACED_MODEL_ERA = 'v5.33' | 'v5.32' | ...` proposal: assessed

**I recommend against it as a replacement, and for it as a preset.** The
reasoning is not aesthetic.

**Argument 1 -- the flags are a lattice, not a timeline.** An ordinal era switch
can express exactly the ~12 points on the historical sequence. The evidence base
does not live on that sequence; it lives on the product space, and the audits
sample it heavily and deliberately:

* `probe_c8_byte_identity.py` holds C6 at its **shipped `True`** while flipping
  C8 -- a configuration that exists at no point in history (before C8 landed,
  C6-on/C8-off was HEAD; after, it is not any era).
* `test_niche_c6_fit_guard.py:104-119` sets `REMAP_STATIONARY_PHASE_LAUNCH`,
  `REMAP_STATIONARY_PHASE_FIT_GUARD`, `_DECENTRE_GATE_PIXELS` **and**
  `_DECENTRE_GATE_W_FRAC` independently in a single helper -- including
  guard-`True`-with-gates-forced-null, which is how the audit pinned that "the
  guard forced ON == the off-centre branch reached via a forced null decentre".
  That pin is *the reason the D1/D7 evidence covers the guard at all*; the audit
  states the consequence: *"if it ever fails, the D1/D7 evidence no longer
  covers the guard."*
* `C6_FIT_GUARD_DECISION` scores the four-cell grid (launch on/off) x (guard
  on/off); `C8_INVERSE_SUPPORT_BOUND` scores (C6 on/off) x (C8 on/off) x (guard
  on/off) and reports the `C6 on + C8 + gd` row specifically to establish that
  *"they compose without interacting"*.
* `RECON_PINS_POST_C8` pins arms at `REMAP_STATIONARY_PHASE_LAUNCH = False`
  *and* separately at `REMAP_INVERSE_SUPPORT_BOUND = False`, with everything
  else shipped.

Collapsing ten independent booleans into one ordinal **destroys the ability to
express the single most-cited comparison in the entire campaign** (C6-on /
C8-off, which is how C8's whole case is made). That is not a migration cost; it
is a loss of capability.

**Argument 2 -- the individual flags are the fail-before contracts.** Each
flag's docstring states its fail-before in terms of *that flag*: "setting it
back to `4` restores the v5.32.0 / niche-C9 behaviour exactly, **since it is the
only thing that changes**" (C10); "`REMAP_INVERSE_SUPPORT_BOUND = False`
restores the pre-C8 library bit for bit" (C8); "Set to `0.0` to restore the
historical hard NaN mask exactly -- **that is the fail-before switch the D1 tests
use, not a supported configuration**" (D1). An era switch would have to
re-state each of these as "era `v5.31` restores ... **and also** these four
unrelated things", which is a strictly weaker contract and a strictly harder one
to verify. C10's own fail-before probe (`rc_failbefore_121.py`) makes the point
explicitly: it needed only an in-process patch, where C9 needed a whole
`git archive` device, *because* C10 changes one integer read at one site.

**Argument 3 -- the migration is expensive and buys nothing measurable.**
~117 assignment sites across tests and validation, each of which would need
rewriting into an era-plus-override form. The C8 audit's discipline was *"No
existing runner was edited"* and RECON's was *"No existing runner and no library
file was edited"* -- both studies went out of their way to add new files rather
than touch working ones, precisely because these runners are the reproduction
record. A migration that edits 117 of them is the opposite of that discipline.

**What I do recommend instead** (details in S8): a **registry plus a context
manager plus presets**, additive, ~120 lines, zero migration required:

```
_TRACED_ERA_FLAGS = {          # module -> attr -> {era: value}
    ('_lens_traced', 'TILTED_CARRIER_EXACT_EIKONAL'):   {'v5.31': False, 'v5.32': True},
    ('_lens_traced', 'REMAP_STATIONARY_PHASE_LAUNCH'):  {'v5.31': False, 'v5.32': True},
    ('_lens_traced', '_REMAP_RESID_EIKONAL_DEGREE'):    {'v5.32': 4,     'v5.33': 6},
    ...
}

@contextmanager
def traced_flags(**overrides):     # save/restore N flags atomically, re-entrant
    ...

@contextmanager
def traced_era(era, **overrides):  # preset + per-flag override, so the LATTICE stays reachable
    ...
```

This gets every benefit an era switch was reaching for -- one table naming every
switch, its eras and its values; one audited save/restore; a discoverable
`v5.32`-vs-`v5.33` comparison -- while keeping all `2^10` combinations
expressible via `overrides`, keeping every existing flag and every existing test
working unchanged, and giving `pytest-xdist` a place to hook a
`pytest.fixture(autouse=True)` snapshot if that ever becomes necessary.

---

## S6. Cost and risk -- the two refactor paths, and doing nothing

### S6.1 Path A -- pure extraction, zero bits changed

**Content.** Extract UNIT A, UNIT B, UNIT C and the Newton family to
module-level classes (S4.2). No behaviour change. No flag change. No test
change.

**Effort estimate.**

| item | estimate |
|---|---|
| write the four classes + rewire the body | 2-3 days |
| move ~250 lines of in-line prose to their new homes, fix the ~96-edge cross-reference graph | 2-3 days |
| re-run the byte-identity devices (S6.3) | 1-2 days incl. machine time |
| full unit suite (329 tests in the C8 file's suite ran 1,190 s; the whole `tests/unit` non-slow leg is CI-sharded 3 ways) | 1 day |
| **total** | **6-9 working days** |

**Risk.** The refactor is *provable* -- and this codebase already owns the
machinery to prove it, which is the single strongest argument for Path A:

* `probe_c8_byte_identity.py` -- **26 configs** (12 synthetic covering every
  `preserve_input_phase` x `amplitude_model` combination at `rs` 1 and 4, plus 7
  design-121 chain runs at (0,0) and 7 at (-4,-2)), compared against a **shadow
  module** built from `git show HEAD:lumenairy/elements/_lens_traced.py`
  imported inside the live package.
* `probe_c6_byte_identity.py` -- **29 configs**, same mechanism.
* `fc_c9_byte_identity.py` -- **52 configs** against a whole-package
  `git archive HEAD` export in a **separate process**, which is the device
  required when the changed code is reached through more than one module name.
  D121_FINAL_CLOSURE records the result: *"52 of 52. With
  `SPHERE_PARAB_CONVERSION_EXACT = False` the working tree IS `HEAD`, bit for
  bit."*

A pure extraction of `_lens_traced.py` internals is reachable by the *shadow
module* device (the element is called through one name), so
`probe_c8_byte_identity`'s mechanism applies directly. A `carrier.py` extraction
would need the `git archive` device -- `fc_c9_byte_identity.py`'s header
explains exactly why: *"the chain entry point, the element hand-off and half a
dozen helpers all resolve it as `lumenairy.propagators.carrier`, so a shadow
copy would be reached by some call sites and not others."*

**Residual risks specific to Path A:**

1. **Float-order sensitivity.** `Phi.grad` returning `carrier.grad + resid.grad`
   as one expression instead of two statements (`L_in = ...; L_in = L_in +
   _gLa`) is bit-identical in IEEE754 for the same operand order, but any
   accidental reassociation is not. Every extraction must preserve operand
   order literally. The byte-identity probes catch this, which is why they are
   non-negotiable.
2. **The `.diag` mutation at 5518.** `_resid_eik.diag['grad_a_fit_max_launch']`
   is written *after* construction and read into `_remap_launch_out`. A sum
   object must preserve that channel.
3. **Warm-up nondeterminism.** `test_niche_c6_fit_guard.py` carries a
   module-scope `_warm` fixture citing *"the W9 determinism calibration: in a
   fresh process one fixed..."*. Byte-identity comparisons on this pipeline are
   known to need a warm-up boundary; any new probe must respect it.
4. **The gap between the two `_lens_traced.py` reference points is moving.** The
   brief states another agent is editing this file right now. Path A cannot
   start against a moving target.

**What breaks if Path A is done wrong:** everything, silently. A single changed
bit in a fit's operand order propagates through Newton into the amplitude and
shows up as a fraction-of-a-point EE3 change on one order -- which is inside the
noise of most of the suite and outside the noise of the design-121 acceptance.
The byte-identity devices are the only defence and they must be run.

### S6.2 Path B -- re-baseline (consolidate the flags, retire the redundant ones)

**Content.** Path A, plus: retire `REMAP_STATIONARY_PHASE_FIT_GUARD` (the C8
audit says it is redundant on every case measured); collapse the D1/D7/C1
constants into `_RayFitDomain` defaults; migrate the flags to the era registry
with the individual names deprecated.

**Effort estimate.**

| item | estimate |
|---|---|
| Path A | 6-9 days |
| retire the guard: remove 16 test/validation assignment sites, re-home the 4 prose references, decide what replaces `test_niche_c6_fit_guard.py`'s 13 tests | 2-3 days |
| migrate ~117 flag assignment sites | 3-4 days |
| re-baseline every pin the migration moves, **at the campaign's own standard of proof** | see below |
| **total** | **20-35 working days** |

**The re-baselining cost is measurable, and it is the number that decides
this.** Commit `4c027e3` reconciled **four** test pins to the settled C6-C8
tree. Its cost:

* **1,883 lines inserted, 48 deleted**, across 10 files.
* **Five new independent oracle runners** (`recon_s12_measure.py`,
  `recon_s12_oracle.py`, `recon_remap_residual_oracle.py`, `recon_d5_oracle.py`,
  `recon_em6_stimulus.py`) -- 1,042 lines of new validation code.
* A **448-line audit document** recording the adjudication.
* The standard applied: *"Every pin adjudicated against an oracle sharing no
  code with the thing under test BEFORE being touched; original assertions kept
  verbatim on fail-before arms ... so the old behaviour remains a pinned
  witness."* Exactly one bar was moved (D5's `> 3.0` -> `> 2.5`) and it was
  compensated with two new higher-margin assertions.

**That is ~470 lines and roughly one new oracle per re-baselined pin.** Any
Path-B step that moves N pins should be budgeted at that rate until evidence
says otherwise. Retiring the C6 fit guard alone touches 16 assignment sites and
a 13-test file.

**What breaks:** the fail-before contracts, one at a time, each requiring the
adjudication above. And the reproduction record: `validation/repro_traced_carrier_121/`
holds ~70 runners whose whole value is that they can be re-run against a named
commit. A flag migration invalidates every one that names a flag.

### S6.3 The do-nothing cost

Being equally honest in this direction. The measured costs of the current
structure:

1. **The three-site invariant (UNIT A) is maintained by comment.** It has not
   yet failed here. The same class *has* failed one module over: C3's chief-ray
   closure existed in two places, and the shipped mitigation was a runtime
   cross-check that *raises on mismatch* -- i.e. the authors judged inspection
   insufficient.
2. **The three-support inconsistency (UNIT C) already has a measured
   consequence.** `RECON_PINS_POST_C8` S7.1: C8's retained plateau+feather band
   lies inside C7's reporting radius, so on a pathological fixture the two
   self-checks are **jointly blind to 0.19998 of `P_ap`** of manufactured light
   whose amplitude is the field's global maximum. That was found by an
   adversarial re-check, not by the checks. This is a live, documented,
   unresolved gap that the current structure makes hard to see and would make
   trivial to assert against.
3. **The 480-line resolve/apply split (UNIT B) is a standing hazard.** It exists
   for a good reason (the C6 freeze constraint) and is documented at both ends,
   but it means the fit-domain policy cannot be read in one place, and the
   C6-guard audit had to *measure* to discover that the branch gate is per-group
   rather than per-order.
4. **Prose drift.** One dangling `:data:` reference already (S2.3). Constant
   notes retract each other across documents: `_REMAP_RESID_EIKONAL_DEGREE`'s
   note contains a retraction of its own earlier mechanism ("That is REFUTED"),
   `REMAP_STATIONARY_PHASE_FIT_GUARD`'s contains three dated addenda, one of
   which declares it superseded. C8 had to hand-edit *two other constants'*
   notes because both named its fix as unattempted. The C7 record has two D6
   call labels swapped (found by the ORACLE audit). None of this is a bug; all
   of it is maintenance load that grows superlinearly with layer count.
5. **Comprehension.** A new reader must hold 13 switches, 96 cross-references
   and a 773-line docstring to change anything safely.

**Weighed honestly:** (2) is the only item with a measured physical
consequence, and it is a *reporting* gap on a pathological fixture, not a
production defect. (1) and (3) are risks that have not yet fired in this module.
(4) and (5) are real and growing. The do-nothing cost is **moderate and mostly
prospective** -- which is precisely why the recommendation below is not "refactor
now".

### S6.4 The cost nobody budgets: the prose is the asset

61 % of `_lens_traced.py` is commentary, and it is not filler. It is the
measurement record: every constant sits adjacent to the sweep that set it, the
regimes that sweep covered, and (in several cases) the retraction of an earlier
reading. `_FIT_DISC_OUTSIDE_WEIGHT_REL` alone carries 76 lines including an
explicit narrowing of its own evidence envelope; `_REMAP_RESID_EIKONAL_DEGREE`
carries 145 lines including two dated revisions and a full degree-3-to-6 table.

**Any code move must move the note with it, and the notes are cross-linked 96
ways.** This is the single largest and most-underestimated line item in both
paths, and it is why my day estimates put "move the prose and fix the graph" at
the same order as "write the code". A refactor that moves the code and leaves
the notes behind would destroy the most valuable property this file has.

---

## S7. Era-pinned test impact analysis

### S7.1 The inventory

Files whose assertions are keyed to a specific layer state:

| file | tests | flags it pins |
|---|---:|---|
| `test_niche_c5_exact_tilted_reference.py` | -- | `TILTED_CARRIER_EXACT_EIKONAL` (x15) |
| `test_niche_c6_stationary_phase_launch.py` | 19 | `REMAP_STATIONARY_PHASE_LAUNCH`, `_REMAP_RESID_EIKONAL_DEGREE` |
| `test_niche_c6_fit_guard.py` | 13 | LAUNCH + GUARD + both C1 gates + `_FIT_DISC_OUTSIDE_WEIGHT_REL` |
| `test_niche_c7_ray_density_halo_check.py` | 15 | `RAY_DENSITY_HALO_CHECK`; era-pinned to `REMAP_INVERSE_SUPPORT_BOUND=False` for its positive control |
| `test_niche_c8_inverse_support_bound.py` | 13 | `REMAP_INVERSE_SUPPORT_BOUND` |
| `test_niche_c10_residual_eikonal_degree.py` | -- | `_REMAP_RESID_EIKONAL_DEGREE` |
| `test_niche_c1_consolidation.py` | -- | both decentre gates |
| `test_niche_d1_tilted_carrier.py` | -- | `_FIT_DISC_OUTSIDE_WEIGHT_REL` |
| `test_niche_d7_decentred_fit.py` | -- | `decentred_fit_poly_order` kwarg |
| `test_niche_s12_remap_sampling.py` | 5 reconciled | `REMAP_STATIONARY_PHASE_LAUNCH` |
| `test_niche_upsample_lattice_fix.py` | 1 reconciled | `REMAP_STATIONARY_PHASE_LAUNCH` |
| `test_niche_audit_w3_elements.py` | 1 reconciled | `REMAP_INVERSE_SUPPORT_BOUND` |
| `test_niche_d5_dx_flatness_gate.py` | 1 reconciled | (bar re-priced 3.0 -> 2.5) |

Broader coupling: **74 test files call `apply_real_lens_traced(`** (196 call
sites) and **67 call `propagate_traced_carrier_chain`**; those files hold
**791 test functions**, about 20 % of the repository's 3,868.

### S7.2 Under Path A (pure extraction)

**Nothing breaks, by construction.** Every flag keeps its name, its module, its
default and its semantics. Every `LT.FLAG = value` continues to work because the
flag is still read at the same logical point -- the read simply moves inside a
constructor. The one requirement is that each flag continue to be read **at call
time**, not captured at import time or cached on an object built once per
process. That is a real trap:

* `_RayFitDomain.resolve()` must read `_FIT_DISC_OUTSIDE_WEIGHT_REL`,
  `_DECENTRED_FIT_POLY_ORDER`, `_DECENTRE_GATE_*` **when called**, not as class
  defaults.
* `_TracedExitSupport` must read `_SUPPORT_BOUND_FEATHER_CELLS` and
  `RAY_DENSITY_HALO_CHECK` when called.
* `_carrier_jet` must read `TILTED_CARRIER_EXACT_EIKONAL` when evaluated -- and
  note that `carrier._exact_tilt_reference()` already does this deliberately
  (*"read at call time so the two can never be configured apart"*), which is
  both the precedent and the warning.

A single class-level default would silently break the fail-before for that flag,
and the failure mode is a test that still passes because it was written against
a helper that also captured the default. **Mitigation: add one test that, for
every flag in the S8 registry, flips it and asserts the returned field changes
(or is documented inert).** That is ~40 lines and it closes the whole class.

The prepared-screen path deserves a specific note: `return_screen=True` /
`prepare_real_lens_traced` / `apply_real_lens_traced_multi` cache a traced
screen across calls. `_CARRIER_FIT_RADIUS_FRAC`'s note records that the R7
restriction is *"INPUT-INDEPENDENT (geometry only), so the prepared-screen reuse
path ... stays valid."* Any extraction must not accidentally make a
flag-dependent quantity part of the cached screen.

### S7.3 Under Path B (flag consolidation / retirement)

This is where the real damage is, and it is asymmetric across the flags:

* **`REMAP_STATIONARY_PHASE_FIT_GUARD` (retire).** 16 assignment sites; a
  13-test file whose entire subject is the flag; and a load-bearing pin -- the
  byte-identity row establishing that *guard forced ON == the off-centre branch
  reached via a forced null decentre*, which is what makes the D1/D7 evidence
  cover the guard at all. Retiring the flag retires that pin, and with it the
  transitive evidence. The C8 audit deliberately kept the guard for a stated
  reason (*"the two act on different objects"*; a defect depositing energy
  *inside* the traced support is invisible to a support bound by construction).
  **Retiring it is a physics decision, not a cleanup.**
* **`REMAP_STATIONARY_PHASE_LAUNCH` (35 sites).** Six of these are *reconciled*
  pins whose original assertions are kept verbatim on a C6-off arm. Those arms
  exist *because* the flag exists. Migrating them to an era name means restating
  each contract as "era `v5.31` also changes the following unrelated things",
  which weakens it.
* **`REMAP_INVERSE_SUPPORT_BOUND` (21 sites).** Same structure; the C7 positive
  control's *stimulus* is era-pinned to `False` because C8 removes it.
* **`_REMAP_RESID_EIKONAL_DEGREE` (14 sites).** Cheapest to migrate -- one
  integer, one read site, an in-process fail-before probe.

And a structural point about staleness that Path B makes worse, not better:
`probe_c6_byte_identity.py` and `probe_c6_tilted_failbefore.py` were written
when `HEAD` predated C6. Once C6 shipped, the first became stale and now prints
`array_equal=False` on **17 of its 29 arms**; the second did *not* go stale
because it sets the flag `False` on the **shadow module as well as the live
one**. That is the durable pattern: **a byte-identity probe that pins a flag on
both sides survives a moving HEAD; one that pins only the live side does not.**
Any new probe written for a refactor must follow the second pattern.

### S7.4 The class of test that a refactor cannot protect

`test_niche_c1_consolidation.py:212` asserts that the effective fit order
observed inside `_Cheb2DEvaluator` equals `_DECENTRED_FIT_POLY_ORDER` -- i.e. it
reaches into an *internal* to check that a policy reached it. Under UNIT B that
internal moves. Such tests are the ones that must be rewritten under Path A too,
and they are the ones worth finding first (S8 step 1).

### S7.5 The structural problem era-pinning has, independent of any refactor

This is the most important thing in S7 and it is not caused by the layering --
it is caused by the layers *working*. `D121_RESIDUAL_CLOSURE` S8.3 states it:

> "every 'the guard is still needed' test is a race between the guard's value
> and the rest of the library, and it will lose eventually."

Two independent instances fired in one tree in one afternoon:

* **D7's fold witness.**
  `test_niche_d7::test_the_fold_regularisation_is_still_load_bearing_at_the_d7_order`
  degenerates `_FIT_DISC_OUTSIDE_WEIGHT_REL` back to D1's hard mask and asserts
  the call then folds and ghosts. At `_REMAP_RESID_EIKONAL_DEGREE = 6` the
  fixture's off-beam amplitude falls from ~0.35 to **1.8e-04 of peak** -- the
  witness stopped witnessing. Resolution: assertions kept word for word, now
  scored **era-pinned at degree 4**.
* **D6's paraxial-FWHM discriminator.** Its ratio has migrated **3.19x (niche
  C3) -> 1.857x -> 1.762x (under C8) -> floored at 1.25x** (`ff7c703`), each
  time for the same reason: *"every physics fix also better-places the
  deliberately inferior route's spot."*

And the option neither took, named in the same section: **find a fixture where
the guard is still load-bearing on the current tree and move the witness
there.** Era-pinning preserves the assertion but converts a live discriminator
into a historical one; over enough layers, a suite accumulates witnesses that
all pass and none of which can fail.

**Implication for the recommendation.** This is an argument *for* Step 2's
registry -- a witness pinned by an explicit, named, discoverable era value is
auditable ("which of our fail-before arms are still live on the current tree?"
becomes a query), where one pinned by a bare `LT._CONST = 4` in a test body is
not. It is an argument *against* Path B: retiring flags removes the very pins
that record what a guard used to catch, which is the only remaining evidence
once the witness has stopped witnessing.

---

## S8. Recommendation

**Do not refactor now. Do not consolidate the flags. Do three specific things,
all additive, all bit-preserving, in this order.**

The reasoning in one place:

* The function is 1,640 statement lines, not 7,000 (S1). The size argument for a
  refactor is weaker than it looks.
* There is no clean cut in the middle third (S4.3); phase-splitting would trade
  a large function for a large struct.
* The extractions that *are* right (UNIT A/B/C) buy risk reduction, not
  accuracy, and the risks they reduce have not yet fired in this module (S6.3).
* Another agent is editing `_lens_traced.py` right now. A byte-identity-proven
  refactor cannot start against a moving file.
* The measured price of re-baselining is ~470 lines and one new independent
  oracle *per pin* (`4c027e3`). Path B's benefit does not clear that bar.
* And the flags are a lattice, not a timeline: an ordinal era switch would
  destroy the ability to express C6-on/C8-off, which is the configuration the
  entire C8 case rests on (S5.2).

### Step 1 -- the structured layer map, as a document (2-3 days, zero code)

Write `docs/audits/TRACED_LAYER_MAP.md` and keep it current. It is the artefact
the brief's "structured internal document mapping the layers" names, and it is
the highest-ratio item on this list because it makes every subsequent decision
cheaper without touching a bit.

Contents, all of which this assessment has already derived:

1. The switch table (S2.1) -- identifier, file:line, shipped value, fail-before
   value, code sites, the audit that set it.
2. The dependency graph (S2.2) with its eight stated edges.
3. The three-unit map (S3) -- for each unit, the sites that must agree, so a
   future author changing the entrance eikonal is *told* there are three of
   them.
4. The seam table (S4.1).
5. The byte-identity device catalogue: which probe covers which module, why
   `_lens_traced.py` can use the shadow module and `carrier.py` cannot, and the
   both-sides-pinning rule from S7.3.
6. A "known open" section carrying forward the live items this study touched:
   * the C7/C8 joint blindness on the E-M6 fixture (0.19998 of `P_ap`);
   * the `_DECENTRED_FIT_POLY_ORDER` order-10 anomaly (good at 6, 8 and 12; bad
     at 10 -- "a second, independent thing wrong in the same neighbourhood");
   * the C1 decentre gate sitting 10-14x below its measured crossover;
   * `_REMAP_RESID_DEGREE_CAP` now equal to the default, i.e. vestigial;
   * the two `cos^2` taper onset figures that disagree (3.2 w vs 2.5 w) and the
     written-but-unrun `rc_c5taper_121.py` census;
   * the dangling `CHAIN_EXACT_TILTED_REFERENCE` reference, and the roadmap's
     stale "`_chain_chief_ray_at_target` still on the lumped ABCD" entry (it was
     converted; `carrier.py:6455-6461`).
7. The three harness traps from S8 step 2, as standing rules for any new runner.

Add a CI check that every identifier in the table exists (this would have caught
the dangling reference; it is ~30 lines against the table). Consider a second
check that every `docs/audits/**` "still open" line naming a library symbol is
reviewed when that symbol changes -- that is what would have caught the stale
roadmap entry, though it is harder and I would not do it in step 1.

### Step 2 -- the flag registry, context manager and presets (1-2 days, additive)

Ship `lumenairy/elements/_traced_flags.py` (or a section of `_lens_traced.py`)
containing the S5.2 sketch: one `_TRACED_ERA_FLAGS` table, a `traced_flags(**overrides)`
context manager that saves and restores atomically, and `traced_era(era, **overrides)`
that applies a preset *and* accepts per-flag overrides so the full lattice stays
reachable.

**Change nothing else.** Every existing flag, test and probe keeps working. New
tests use the context manager; old ones are left alone. This delivers the
genuine benefits the era-switch idea was reaching for -- one discoverable table,
one audited save/restore, a named `v5.32`-vs-`v5.33` comparison -- at ~120 lines
and zero migration.

**The motivating evidence is three measured harness failures, all of the same
shape: an intervention expressed relative to a default, evaluated after the
default moved.** These are the sharpest argument in the whole campaign for a
registry, and they are all from the campaign's own instruments:

1. **`TAPER='on'` stopped meaning "the taper" the moment the library default
   flipped.** `fc_sampling_121.py` returned byte-identical taper-on/off rows,
   and `fc_production_taper.py` ran a nine-minute "BASELINE (taper as shipped)"
   row that was in fact the *exact* conversion -- its baseline read 89.235 where
   v5.32.0 reads 87.834. The lesson, quoted: *"An intervention expressed as
   'leave the library alone' is not an intervention once the library moves."*
   C10 designed around it by construction: **every arm pins its own value
   through a script-side `Patch`, never through the default.** A registry plus
   `traced_flags(**overrides)` makes that discipline the path of least
   resistance instead of a rule each author must remember.
2. **`approx_common.py` silently defaulted `LUMEN_PIN` to a frozen v5.31
   export** that still existed on the machine. Caught only by an `AttributeError`
   on `REMAP_INVERSE_SUPPORT_BOUND` -- *"a constant that does not exist in
   v5.31"*. Every runner now forces `LUMEN_PIN=0` and prints file hashes. Lesson
   quoted: *"A harness that silently selects a library is worse than one that
   crashes."*
3. **`wfe_probe_orders.py` cached results keyed on the CONFIGURATION, not the
   LIBRARY.** Five files dated 2026-07-30 (pre-C7/C8/C9) would have re-scored
   the pre-C9 chain as the post-C9 verdict. **Hit twice in one session.** Lesson
   quoted: *"Knowing about a trap is not the same as being immune to it; only
   moving the file is."*

The registry addresses (1) directly and gives (2) and (3) a hook: a single
`traced_flag_state()` returning the current values of every registered switch is
what a runner should print in its provenance banner alongside the file hash, and
what a cache key should include.

Add, at the same time, the flag-liveness test from S7.2: for every entry in the
registry, flip it and assert the returned field changes on a designated fixture
(or that the entry is marked inert-by-design). ~40 lines, and it is the standing
guard that makes Step 3 safe.

### Step 3 -- extract UNIT C first, alone, byte-identity-proven (3-4 days)

When `_lens_traced.py` is quiet, do **one** extraction, and make it
`_TracedExitSupport` (S3.3). It is first for four reasons:

* It has the narrowest closure capture (`_support_taper` captures 3 locals) and
  the fewest live variables crossing it.
* It sits at the two lowest local minima in the live-variable profile (5994 and
  6585).
* It is the unit with a **live documented consequence** -- the C7/C8 joint
  blindness -- so the extraction lands next to something worth asserting: one
  invariant, `C7 reporting radius > C8 retained band`, which is a one-line check
  on a single object and is currently a relationship between two independently
  computed quantities 40 lines apart.
* It is the smallest unit, so if the byte-identity proof turns out to be more
  expensive than estimated, the sunk cost is one class.

Prove it with `probe_c8_byte_identity.py`'s **26-config** shadow-module device,
extended to hold every registry flag on **both** sides (S7.3). If and only if
that comes back 26/26 at `max|dE| = 0.000e+00`, proceed to UNIT A, then UNIT B,
one per release, each with its own proof. If it does not, stop: the estimate was
wrong and the remaining units are larger.

**Explicitly not in scope, and why:**

* Retiring `REMAP_STATIONARY_PHASE_FIT_GUARD` -- it is redundant *on every case
  measured*, but the C8 audit states a mechanism-level reason to keep it, and
  retiring it retires a load-bearing byte-identity pin (S7.3).
* Migrating the ~117 existing flag assignment sites -- no measured benefit,
  invalidates the reproduction record.
* Splitting `apply_real_lens_traced` into sequential phases -- no clean seam
  exists (S4.3).
* Anything in `carrier.py` -- it needs the more expensive `git archive` proof
  device, and it grew +135 % during this campaign, so it is the *less* settled of
  the two.

---

## Appendix A -- evidence index

Everything asserted above, and where it comes from.

| claim | source |
|---|---|
| file sizes, prose share, statement counts, closure captures, live-variable profile, cross-reference count | static analysis of the tree at `5af1edf` (this study) |
| growth per commit | `git show <sha>:<path> \| wc -l` |
| executable lines per layer | `git show <sha> -- lumenairy/`, comment/blank-filtered |
| "three halves of one substitution" | `_lens_traced.py:4836-4838` |
| C6 freeze must clear the ray-fit disc; the 130x step | `_REMAP_RESID_FREEZE_MARGIN` note, `_lens_traced.py:2354-2415` |
| C5 -> C6 quadratic scaling (1.46 -> 2.30 mrad, ratio 2.48) | `REMAP_STATIONARY_PHASE_LAUNCH` note, 2141-2145 |
| C6 destroys D1's radial-symmetry precondition | `REMAP_STATIONARY_PHASE_FIT_GUARD` note, 2433-2447 |
| guard is per-group not per-order; the decentre table | same note, 2517-2539; `C6_FIT_GUARD_DECISION_2026_07_31.md` |
| C8 supersedes the guard; C8 fixes what the guard structurally cannot | `REMAP_INVERSE_SUPPORT_BOUND` note, 2691-2716; guard note, 2591-2609 |
| C8 repairs a C6-OFF violation at rs=2 on (-4,-2) | `REMAP_INVERSE_SUPPORT_BOUND` note, 2712-2716 |
| C10 unblocked by C8; degree-6 ghost 5.2 % of Pin with C8 off | `_REMAP_RESID_EIKONAL_DEGREE` note, 2249-2280 |
| C7/C8 joint blindness, 0.19998 of `P_ap` | `RECON_PINS_POST_C8_2026_08_01.md` S7.1 |
| C7 `e^-9` gate understates support (1.6161 vs 1.8115 mm); D6 labels swapped | `ORACLE_ENERGY_AND_D6_HALO_2026_08_01.md` S6.1-S6.2 |
| "the containment the direct-fit path has had all along" | `C8_INVERSE_SUPPORT_BOUND_2026_08_01.md` S3.1 |
| 26-config / 29-config shadow-module devices; 17-of-29 staleness; both-sides pinning | `probe_c8_byte_identity.py` header; `C8_INVERSE_SUPPORT_BOUND` S7.1, S11.8 |
| 52-of-52 `git archive` device, and why `carrier.py` needs it | `fc_c9_byte_identity.py` header; `D121_FINAL_CLOSURE_2026_08_02.md` line 659 |
| C6 guard = 3 executable lines / 231 doc lines | `APPROXIMATION_AUDIT_POST_C6_2026_07_31.md` |
| C7 = ~55 executable lines in two places | `C6_FIT_GUARD_DECISION_2026_07_31.md` |
| C8 = ~30 + ~25 lines + one multiply, "no signature moved" | `C8_INVERSE_SUPPORT_BOUND` S9.1 |
| re-baselining four pins: 1,883 lines, 5 oracles, 448-line doc | `git show 4c027e3 --stat` |
| C3's two chief-ray closures + raise-on-mismatch cross-check | `git log -1 7f45874`; `carrier.py:3404`, `6388` |
| C3 guard is bitwise-neutral | `git log -1 7f45874`, item 2 |
| CI is serial (pytest-split, not xdist); xdist is a dev dep | `.github/workflows/unit-tests.yml:26`; `pyproject.toml:163` |
| flag save/restore discipline is uniform (37/37 test sites protected) | this study, per-site inspection |
| `CHAIN_EXACT_TILTED_REFERENCE` does not exist | `grep -rn` over the whole repo; actual mechanism at `carrier.py:1855-1868` |
| `_chain_chief_ray_at_target` WAS converted (roadmap's "still open" is stale) | `carrier.py:6455-6461`, read directly |
| C4 `.T` transpose; `na_exit` 0.3633 vs 0.2912 | `_lens_traced.py:5691` and its comment |
| order-10 anomaly; C1 gate 10-14x below crossover; degree cap now vestigial | `D121_RESIDUAL_CLOSURE_2026_08_02.md` S7 items 1, 2, 7 |
| EE blind to the halo (+1.691 EE3 while manufacturing 0.486 %); 82 %/121 % reads as 0.005 EE3 | `REMAP_INVERSE_SUPPORT_BOUND` note; `APPROXIMATION_AUDIT_POST_C6_2026_07_31.md` S3.7 |
| conservation/halo blind to a destroyed spot (6/6 while losing 19.9 EE3) | `D121_RESIDUAL_CLOSURE` S5.1 |
| `_tilt_exactness_phase` taper is applied identically on +1/-1, round trip exact to 2e-16; ablation 2.3e-5 | `carrier.py:1871-1935` docstring, read directly |
| the two taper onset figures disagree (3.2 w vs 2.5 w); census unrun | `carrier.py:1920` vs `D121_RESIDUAL_CLOSURE` S7 item 8 |
| era-pinning doctrine; D7 witness 0.35 -> 1.8e-04; D6 ratio 3.19x -> 1.857x -> 1.762x -> 1.25x | `D121_RESIDUAL_CLOSURE` S8.3; commit `ff7c703` |
| the three harness traps (`TAPER='on'`, `LUMEN_PIN`, cache key) | `D121_FINAL_CLOSURE` / `D121_RESIDUAL_CLOSURE` harness sections |

## Appendix B -- the three-site table (UNIT A), for the layer map

The invariant that must hold, stated once:

> Wherever the element uses the entrance congruence, it must use **the same**
> total eikonal `Phi = W + a_fit` -- the launch direction is `grad Phi`, the
> H6 term added to the traced OPL is `Phi(x_in)`, and the residual de-chirp
> removes `Phi` from the input phasor so that what is transported pointwise is
> `exp(i k0 (a - a_fit))`.

| site | what it needs | current lines | current form |
|---|---|---|---|
| ray launch | `grad Phi` | 5498-5523 | `_carrier_grad(...)` then `+= _resid_eik.grad(...)` under `if _resid_eik is not None` |
| H6 entrance eikonal | `Phi(x_in)` | 5643-5655 | `+= _carrier_W_fn(...)` then `+= _resid_eik.value(...)` under a second `if` |
| residual de-chirp | `Phi` on the wave grid | 4846-4876 | `* exp(-1j k _pip_remap_W)` then `*= exp(-1j k _resid_eik.value(...))` under a third `if` |
| (consumer) fit-guard predicate | "does a residual term exist" | 5933-5934 | `_resid_eik is not None and REMAP_STATIONARY_PHASE_FIT_GUARD` |
| (consumer) launch diagnostic | `.diag`, mutated post-construction | 5398-5404, 5518-5521 | `_remap_launch_out.update(_resid_eik.diag)` |

Any change to the entrance eikonal must touch rows 1-3 together. Today that is
enforced by three comments. Under UNIT A it would be enforced by there being one
object.
