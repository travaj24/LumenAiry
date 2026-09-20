# WP-C3 -- `transport='collins'` as the carrier chain's default, and the CuPy arm that had to land with it

Branch `feat/c3-collins-default`, base **49ddf4bd** (`main`).  Implements
`MAINTAINER_DECISIONS_2026_09.md` section 1.2, which is the maintainer's
decision and not this package's: WP-B4 built the transport, VERIFY-WP-B4
re-measured it and raised five follow-ups, Wave-5 hygiene-2 (H2-2) put it on
the field's own backend and left the CuPy half owed.  WP-B4 section 5 named
two preconditions for a flip -- a backend arm, and the complementary
selection kept -- and this package supplies both.

Everything below was measured on this machine with
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, one process at
a time, on 2026-09-20, on BOTH builds: **WIN-py3.14** (Windows 11, py3.14.6,
jax 0.11.0, cupy 14.0.1 with one visible device) and **WSL-py3.12**
(py3.12.3, jax 0.10.2, no CuPy).

---

## 0. The finding this package turns on

**Flipping the three signatures would have been a defect.**  It is worth
stating first, because it is not what the decision ledger anticipated and it
is the reason this package is larger than a one-line change.

MEASURED on WP-B4's own two-group relay (`_chain_fixture`: N = 256 at 60 um,
two BK7 singlets, `final_distance = 8 mm`, `focus_readout=dict(dx_out=0.5 um,
N_out=64)`), the chain's image-plane readout through each transport:

| quantity | `'sziklas'` | `'collins'`, one-step | ratio |
|---|---|---|---|
| on-axis intensity at the readout plane | **1.0743** | **9017.1** | 8394 |
| power on the returned window | 9.704e-10 | 1.521e-06 | 1568 |
| Kelly guard | silent | ONE `RuntimeWarning` (K1) | -- |

A ratio of that size is not an accuracy difference; one of the two is wrong.
**The true value is 1.2359**, and it is not this module's opinion: the same
exit envelope carried the same 8 mm by the chain's own FREE leg -- where the
complementary quadrature selection already existed, so BOTH transports return
the same array to the bit -- reads on-axis `|env|^2` = 1.2359 with the power
conserved at 2.8289e-05 either way.  The readout window is a 32 um patch of a
beam ~6 mm wide 30 mm short of its focus, so an on-axis reading of order 1 is
also what the geometry says.

**An oracle that did NOT work, recorded because the next person will try it.**
The first attempt to arbitrate the two readouts was
`validation/probe_c3_collins_default/probe_readout_oracle.py`: stop the chain
at its exit plane (`final_distance=0`), rebuild the physical field there, and
propagate the final 8 mm with a DENSE separable matrix Fresnel transform --
no FFT, no chirp-Z, no standoff, and therefore none of the sampling
conditions any of those carry.  It does not converge, and the reason is the
same one that makes the Collins readout wrong: the FULL field on the exit
grid carries the carrier, whose phase advances **277 rad per sample** at the
beam edge over that leg, so a dense sum over those samples integrates an
aliased integrand exactly as a chirp-Z would.  Measured: the oracle's own
peak moves 1.63x between N = 256 and N = 512 and both arms read relative
~1.0 against it.  A dense sum is not an escape from an undersampled
integrand.  The arbitration that DID work is the one above -- the same leg
through the chain's own free-space step, where the ENVELOPE is what is
sampled and both transports agree to the bit -- and it needs no new oracle at
all.

The cause is VERIFY-WP-B4 **F1**, which named this quantity as the binding
constraint on a flip and which WP-B4 section 5's list does not contain.  The
one-step readout forms `env(u) * exp(i k A u^2 / (2B))` on the CHAIN'S EXIT
pitch, and K1 is that product's sampling rate against its own Nyquist rate.
Measured on this fixture at three grids, at the plane the readout actually
runs on (the chain exit, `final_distance = 0`):

| chain N | exit pitch | exit support | `theta` | **readout K1** |
|---|---|---|---|---|
| 256 | 76.54 um | 6.736 mm | 8.557 mrad | **82.3605** |
| 512 | 38.27 um | 6.698 mm | 17.114 mrad | **41.4491** |
| 1024 | 19.14 um | 6.774 mm | 34.228 mrad | **21.4557** |

Identical on both builds to every printed digit.  It falls as `1/dx` (ratios
1.987 and 1.932), so `N ~ 16000` would be needed to sample it.  The
hand-derived reading and `_collins_readout_k1`'s own agree to the last digit
at all three grids, which is what says the helper computes what this report
says it computes.

**What was done about it**, and it is F1's own recommendation ("a flip would
have to either keep the Sziklas readout for short final legs or state the
applicability window"): the chain's focus readout now RESOLVES its
quadrature, exactly as `_collins_carrier_leg` already resolved a leg's.

* `transport='sziklas'` -> the Sziklas readout, and nothing else.
* `transport='collins'` -> the one-step Collins readout where
  `_collins_readout_k1 <= 1`, and the Sziklas readout -- called with the same
  arguments the pre-flip default passed, hence **bit-identically** --
  where it is not.
* the route is published on the readout stage as `readout_route`,
  `readout_route_k1`, `readout_route_reason`, and ONLY on `'collins'`, because
  the `'sziklas'` `stages` list is a bit-identity key.

K1 alone routes.  K3 is already owned by `on_replica`, which defaults to
`'error'` and covers the window on both routes; K2 is a representability
statement about the caller's own output pitch and cannot make the returned
samples wrong (VERIFY-WP-B4 row 8: the transform evaluates the integral
exactly AT the requested points whether or not they resolve the field).

Measured after the resolution, same fixture: the two transports return the
**same array to the bit**, the stage reads
`readout_route='sziklas', readout_route_k1=82.3605, readout_route_reason='k1'`,
and the Kelly guard is silent.

---

## 1. What shipped

### 1.1 The default

`transport='collins'` on the three entry points that take it, and there are
exactly three (grep over `lumenairy/**/*.py`; the keyword appears in no other
module):

| entry point | line |
|---|---|
| `propagate_carrier_referenced` | `carrier.py:1156` |
| `propagate_traced_carrier_chain` | `carrier.py:9518` |
| `propagate_traced_carrier_chain_multi` | `carrier.py:12275` |

`carrier_referenced_focus_readout` and
`carrier_referenced_exact_focus_readout` do NOT take `transport` and do not
move -- asserted as a measurement, not read off the signature.
`final_leg='exact'` does not move on either setting (WP-B4 section 1.2).

### 1.2 The readout's quadrature resolution

`_collins_readout_k1` (new) and the route block in
`propagate_traced_carrier_chain`, plus `_publish_readout_route`.  See section
0.

### 1.3 Three geometries with NO fallback, all fixed

Both were opt-in before this package and would have been the default after
it.  Both were found by RUNNING the flip against WP-B4's own test file, not
by reading the diff.

**A COLLIMATED carrier returned an all-NaN envelope on `dx = nan`.**  The leg
resolved correctly (`collins_form='tf'`, K1 = 0.79, K3 = 1.29) and then called
`_carrier_step_fast`, which is only the NO-CROSSING FAST PATH of the Sziklas
transport.  A collimated carrier never reaches it on the Sziklas side: the
entry point short-circuits `R = +/-inf` to a same-grid exact
transfer-function step, precisely because `m = R_out/R = inf/inf` is NaN.
`r_in=np.inf` is one of the commonest chain inputs.

**An ASTIGMATIC carrier had no fallback at all.**  `tf_available` excluded it
on the grounds that the transfer-function form has no per-axis version --
true of `_carrier_step_fast`, false of the Sziklas TRANSPORT, whose entry
point routes an astigmatic carrier to the separable per-axis step and returns
the very `((R_x + z, R_y + z), (m_x dx, m_y dy))` triple the leg promises.
MEASURED on WP-B4's astigmatic fixture (`R = (-40, -55) mm`, `z = 5 mm`,
N = 1024 at 4 um): the chirp-Z ran at K1 = 1.0222 and K3 = 2.7997, so the
returned window spanned 2.8 periods and its outer samples were wrapped copies
of inner ones.

**A leg PAST the carrier's geometric focus (`A < 0`) likewise had none.**  It
was excluded on the grounds that "`m <= 0` is the split this transport exists
to avoid".  That is the right instinct and the wrong rule: avoiding the split
is what the chirp-Z buys WHERE THE CHIRP-Z IS REPRESENTABLE, and where it is
not there is nothing to buy.  MEASURED on `test_carrier_referenced.py`'s
focus-crossing oracle (`w0 = 4 um` at 30 mm, N = 2048, `dx = 12.1 um`):

| z | `A` | K1 | K3 | windowed r2m | analytic |
|---|---|---|---|---|---|
| 45 mm | -0.5 | 1.5907 | 2.5924 | **3333.4 um** | 1105.7 um |
| 60 mm | -1.0 | 2.3831 | 3.8886 | **6981.9 um** | 2211.4 um |

-- 3.0x and 3.2x, against a Sziklas split that matches the same oracle to
better than 1 %.

ONE change fixes all three: the fallback calls
`propagate_carrier_referenced(..., transport='sziklas')` instead of one branch
of it, and the two exclusions that were properties of that ONE branch are
dropped.  The fallback is then the Sziklas ANSWER in every branch it has,
which is also what makes VERIFY-WP-B4 F2's "the legs that move are exactly
those with `N dx^2 <= lambda |z_eff|`" literally true rather than nearly true.

**What still has no fallback, and it is the honest boundary.**  `A == 0`
exactly, and a leg whose output reference the transport resolves FLAT.  Those
are precisely the legs the Sziklas transport could never evaluate: it
re-references to `R_out = 0`, which `carrier_referenced_envelope` refuses, and
it has no flat-reference form.  So "the legs with no complementary form are
the legs the old transport could not do at all" is a statement about the
selection, not a gap in it.

### 1.4 Three internal call sites that rode the public default

`carrier_referenced_focus_readout`'s own carrier step onto the standoff plane
(`carrier.py:4499`) is the Sziklas readout's own machinery, not the caller's
choice of transport.  With the default flipped and that call left implicit,
the readout **stopped raising** its documented containment `RuntimeError` on
a fixture where it had raised (co-moving half-width 8.5333 um against a
measured amplitude radius of 4.6391 um), because the Collins leg resolves its
own output pitch and the grid it handed the containment guard was no longer
the co-moving one the guard is written about.

It was found as ONE moved key in the archive-to-archive probe (section 3) and
by nothing else.  All three sites now NAME `transport='sziklas'`; the other
two are the `transport != 'collins'` arms of the chain's gap and final legs,
where naming it is a statement of fact rather than a change.

### 1.5 The stop-plane readout keys stop being refused and start SELECTING

`focus_readout`'s `standoff` and `on_focus_containment` describe the Sziklas
readout's stop plane.  WP-B4 REFUSED them on `transport='collins'`, correctly
at the time: that transport had no stop plane and no fallback, so the keys had
no referent and accepting them would have been the accept-and-ignore shape the
vocabulary gates exist to remove.

Since the readout resolves its quadrature they have a referent again, because
the Sziklas readout is the route most chain readouts take.  So naming one now
SELECTS that route.  Nothing is accepted and ignored -- the key does exactly
what it says -- and the stage publishes
`readout_route_reason='stop_plane_key'`.

This is not a style preference; it is the blast radius.  MEASURED 2026-09-20,
by matching the refusal's own message across the run's report: with the
refusal in place and the default flipped, **38 of the 54 reported failure
sections carried that message**, and so did all 22 fixture-setup ERRORS (one
`d2` module fixture, which every id in its file depends on).  They span
`test_niche_d3_guards`, `_d4_dgrating`, `_d5_dx_flatness_gate`,
`_c1_consolidation`, `_d2_chain_multi`, `_r8_tiltaware_chain_api`,
`_r9_highna_final_leg` and `test_audit2609_a6_verify_carrier` -- every one of
them a caller who legitimately wants the standoff-based readout and had no
reason to know the transport keyword had moved underneath them.

### 1.6 The CuPy arm

See section 5.

---

## 2. The oracle ladders, before and after, on both builds

The oracle is the analytic Gaussian with its ABSOLUTE phase -- piston and
Gouy, not a piston-free shape comparison -- written from scratch in
`validation/probe_c3_collins_default/clib.py` in this library's
`exp(-i omega t)` / `exp(+i k z)` pairing, i.e.
`1/q = 1/R + i lambda/(pi w^2)` with the 2-D prefactor taken as the RATIO
`q/q2`.  Both of VERIFY-WP-B4's recorded traps are avoided by construction
rather than by comment: Siegman's opposite pairing conjugates the Gouy phase,
and `1/sqrt((1 + z/q)^2)` picks up exactly `pi` past the waist.

Fixture: `lambda = 1.064 um`, `w = 0.30 mm`, N = 512 at six `1/e` radii
(`dx = 3.5156 um`), `R = -40 mm`.  Relative L2 of the reconstructed FIELD on
each arm's OWN output lattice against the oracle evaluated there.

### 2.1 Ladder A -- the single step

| cell | `A = 1 + z/R` | `'sziklas'` | `'collins'` | `dx_out` sziklas / collins (um) |
|---|---|---|---|---|
| diverging | +1.125 | 2.2258e-05 | 2.2258e-05 | 3.95508 / 3.95508 |
| converging | +0.5 | 1.0676e-04 | **3.9502e-05** | 1.75781 / 1.75781 |
| ON the focus | 0 | **raises** `R_carrier == 0` | **3.8791e-05** | -- / 0.36944 |
| just past it | -0.025 | 1.9181e-04 | **4.0668e-05** | 1.58203 / 0.45044 |
| well past it | -0.5 | 7.8566e-02 | **4.3728e-05** | 1.75781 / 1.98926 |
| astigmatic (-40, -55) mm | +0.5 / +0.636 | 9.8910e-05 | **3.9194e-05** | 1.75781 / 1.75781 |

`gap_kernel='fresnel'` reads within 0.3 % of the `'auto'` column on every
Collins cell (4.3724e-05 vs 4.3728e-05 at `A = -0.5`) and identically on every
Sziklas cell.  **WIN and WSL agree to a ratio of 1.00000 on all 22 readings**,
and the focus cell raises on both.

**The floor, and why the Collins column is the floor.**  The fixture's own
grid-truncation relative L2 is **3.87e-05**, computed from the fixture by
quadrature (and the quadrature is refined and compared before it is used, so
the number is converged rather than asserted): the Gaussian is truncated at
three `1/e` radii, where its amplitude is `exp(-9) = 1.23e-4`.  Every Collins
reading sits within 13 % of that floor.  So those readings are the GRID's
error and not the transport's, the 2.7026x at `A = +0.5` is the co-moving
grid and not the quadrature, and the 1796.7062x at `A = -0.5` is the frame
having inverted through the waist.  The full ratio column, `'sziklas'` over
`'collins'`: 1.0000 (diverging -- the same array), 2.7026, unbounded (the
focus, which one arm refuses outright), 4.7164, 1796.7062, 2.5236.

`transport='collins'` is **not worse on any cell** and is qualitatively
different on one: the shipped transport cannot land on the focus at all.

### 2.2 The `_multi` entry point at K = 1 and K = 2

| K | transport | peak | window power | Kelly |
|---|---|---|---|---|
| 1 | sziklas | 1.0742675278 | 9.704005e-10 | 0 |
| 1 | collins | 1.0742675278 | 9.704005e-10 | 0 |
| 2 | sziklas | 4.2970701111 | 3.881602e-09 | 0 |
| 2 | collins | 4.2970701111 | 3.881602e-09 | 0 |

Identical to every printed digit after the readout resolution; BEFORE it the
Collins arms read 9017.10 and 36068.41 with one and four Kelly warnings.  The
K = 2 / K = 1 ratio is 4.00000 on both arms, which is the recombination's own
linearity on two copies of one congruence.

### 2.3 The focus readouts

`carrier_referenced_focus_readout` and
`carrier_referenced_exact_focus_readout` take no `transport` (measured from
the signature) and are unmoved.  The chain's readout is section 0.

### 2.4 The chain end to end, two prescriptions

The P2 battery's fast uncorrected biconvex BK7 singlet and WP-A15a's
curved-rear cemented doublet (the covering-array fixture), N = 256, both with
a bare final leg and with an image-plane readout:

| design | leg | sziklas peak | collins peak | Kelly |
|---|---|---|---|---|
| singlet | bare final | 5.4938762375 | 5.4938762375 | 0 |
| singlet | focus readout | 5.2163494772 | 5.2163494772 | 0 |
| doublet | bare final | 2.0314375970 | 2.0314375970 | 0 |
| doublet | focus readout | 2.1006780344 | 2.1006780344 | 0 |

Identical to every printed digit, `dx`, `R` and stage count included.  Before
the readout resolution the two `focus readout` rows read 96947.08 and
2111419.59, each with one Kelly warning.

**What this means, stated plainly.**  On every fixture this package can reach,
the flip moves the SINGLE-STEP legs of section 2.1 and moves nothing else.
That is the complementary selection working: the chain's relay legs resolve
to the transfer-function form and its readouts resolve to the Sziklas
readout.  A design whose legs satisfy `N dx^2 <= lambda |z_eff|` -- a
near-focus landing, a long reduced leg, a deliberately coarse grid -- is where
the flip is visible, and that condition is checkable per design without
running anything.

---

## 3. Byte identity of the way back -- ARCHIVE TO ARCHIVE

The claim "`transport='sziklas'` is the pre-flip arithmetic" cannot be made
against the working tree, so it is made archive to archive with the
transport's own file as the only variable:

1. `git archive 49ddf4bd lumenairy` extracted twice, to `base/` and `mine/`;
2. `mine/lumenairy/propagators/carrier.py` overwritten with this branch's
   copy -- `diff -rq -x __pycache__` over the two trees reports **that one
   file and no other**;
3. one CHILD PROCESS per tree, `cwd` and `PYTHONPATH` its own root,
   `lumenairy.__file__` asserted under it and printed,
   `LUMENAIRY_MEM_BUDGET_MB=8192` and `PYTHONHASHSEED=0` pinned;
4. **the two arms are spelled differently, and that is the claim**: the base
   arm passes NO `transport=` at all (at 49ddf4bd the default IS `'sziklas'`),
   the branch arm passes `transport='sziklas'` explicitly;
5. the two `{key: sha256}` maps compared key by key.

Each record folds the returned arrays, the returned carrier and pitch, the
returned object's TYPE, the whole `stages` list AND its `repr`, and every
warning in EMISSION order -- so a guard that moved, was reworded or was
reordered is a moved key.

| build | keys | identical | differ | only-base | only-branch | verdict |
|---|---|---|---|---|---|---|
| WIN-py3.14 | 42 | **42** | 0 | 0 | 0 | IDENTICAL |
| WSL-py3.12 | 42 | **42** | 0 | 0 | 0 | IDENTICAL |

The 42 keys: eight single-step geometries (short and long converging, back
propagating, collimated, diverging, near focus, focus crossing, zero length)
x two gap kernels, plus astigmatic, complex64, tilted-exact, explicit exact,
and the two refusals of the free-lattice keywords; both public readouts with
and without a standoff, plus reconstruct / envelope / fit-radius / aperture;
nine chain configurations including the readout with `standoff`, with
`bandlimit`, at zero final distance, with `gap_kernel='fresnel'` and with
`replica_fill='zero'`; and four multi configurations at K = 1 and K = 2.

**One key moved on the first run, and it is section 1.4.**
`R-focus-readout-standoff` raised a containment `RuntimeError` on the base
arm and returned a field on the branch arm.  That is how the internal
call-site defect was found.  After pinning the three internal sites the run is
42 of 42 on both builds.

---

## 4. Blast radius, measured

Selection: every test file that names `transport`, plus every file that calls
`propagate_carrier_referenced`, `propagate_traced_carrier_chain`,
`propagate_traced_carrier_chain_multi`, `carrier_referenced_focus_readout` or
a `_collins_*` helper (grep, not memory), plus `test_*carrier*`, the
near-focus H2-3 file and the collins-JAX file.  **54 files, 1846 collected
ids.**

**The first full run after the flip read 62 failed + 22 errors.**  Two
groups were CODE defects the flip exposed rather than fixtures to re-pin --
sections 1.3 and 1.5 -- and between them they are most of it.  What is left
is restated below, and nothing is loosened.

### 4.1 The two code fixes, by what they closed

Counted by matching each defect's own signature across the run's report,
which prints 54 detailed failure sections for the 62 failed ids
(parametrised repeats share a section) plus 22 setup errors:

| defect | what it was failing | fix |
|---|---|---|
| the stop-plane keys refused (1.5) | **38 of the 54 sections**, plus all 22 setup errors, over 8 files | the keys SELECT the Sziklas readout and the stage says so |
| no fallback past the focus (1.3) | 4 of the remaining 16 (`test_focus_crossing_*`, `test_carrier_referenced`) | the fallback is the Sziklas TRANSPORT, not one branch of it |

The other 12 sections are restatements, and the table below is all of them.

### 4.2 Every test that was restated, and its classification

| test(s) | classification | what was done |
|---|---|---|
| `b4::TestVocabulary::test_the_free_lattice_kwargs_are_refused_on_the_default_transport` | genuine contract on `'sziklas'` | renamed `..._on_the_sziklas_transport` and NAMES it; a new arm asserts the default HONOURS `dx_out` / `carrier_out` and that they change the returned lattice, so the pair is two-sided |
| `b4::TestVocabulary::test_the_stop_plane_readout_keys_are_refused_on_collins` | the contract itself changed | renamed `..._SELECT_the_sziklas_readout`; asserts the route, the published reason, that K1 is NOT published as the reason, and bit-identity with the named-`'sziklas'` run |
| `b4::TestDefaultIsByteIdentical::test_the_single_step_is_equal_bit_for_bit` (5 ids) | the pin being retired | the unnamed call is now compared to the NAMED `'collins'` one |
| `b4::TestDefaultIsByteIdentical::test_the_focus_crossing_split_is_equal_bit_for_bit` | fixture now closer to the oracle | restated: the default does NOT take the split, lands finite, and lands on a FINER lattice than the collapsing co-moving one |
| `b4::TestDefaultIsByteIdentical::test_the_chain_is_equal_bit_for_bit` | still bit-identical, plus new keys | every stage compared whole after removing the `collins_*` and `readout_route*` keys, with their presence and absence per transport asserted separately |
| `b4::TestNoNearFocusApparatus::test_the_same_poison_fires_on_the_default_transport` | genuine contract on `'sziklas'` | renamed `..._on_the_sziklas_transport`; a new arm runs the same leg on the default with the whole apparatus poisoned |
| `d2::_leg_for_window` and its five sibling `standoff=` fixtures | genuine contract on `'sziklas'` | the calibration measures "period per metre of fine-zoom leg", which exists only on that readout; `_SZIKLAS_STANDOFF` names it once, with the reason |
| `d2::test_default_refuses_the_periodic_replica_regime`, `..._auto_window_is_independent_of_congruence_order`, `..._auto_tile_equals_the_same_tile_asked_for_explicitly` | genuine contract on `'sziklas'` | the regime does not exist on the Collins readout, whose period is 3846.09 um against the 2867.20 um window; named, with that measurement |
| `d3::_linearity_error` (4 ids) | genuine contract on `'sziklas'` | its own docstring is the reason: "all five runs land on the same lattice" is a property of the co-moving pitch; the Collins leg resolves a pitch per FIELD and the five fields are deliberately different, so `mux - ref` stops being a linearity residual |
| `exact_gap_kernel::_one_and_two` (8 ids) and `d4::_pair` / `TestSplitLegPathDependence` (4 ids) | genuine contract on `'sziklas'` | a split composes exactly only on the co-moving step -- which is `test_niche_exact_gap_kernel`'s own theorem |
| `d4::TestDoeChainBookkeeping::test_matches_the_manual_hand_split` | genuine contract on `'sziklas'` | the DOE claim is an equivalence between a ONE-piece and a TWO-piece transport of the same leg |
| `exact_gap_kernel::test_a_collimated_leg_honours_the_gap_kernel` | genuine contract on `'sziklas'` | `R = +/-inf` is a BRANCH of that entry point, compared bitwise to that branch's own `_exact_envelope_tf_step` |
| `exact_gap_kernel::test_an_astigmatic_carrier_refuses_the_exact_kernel...` | both transports refuse, differently | restated to assert BOTH, with the two reasons kept apart |
| `carrier_referenced::test_focus_crossing_*` (4 ids) | closed by the code fix (1.3) | no test change |
| `carrier_referenced::test_zero_carrier_raises_but_focus_crossing_is_handled` | both transports, different answers | both arms asserted: `'sziklas'` comes back on the flipped geometric carrier, the default on a FLAT reference, which is the physical statement |
| `carrier_referenced::test_near_focus_landing_fast_path_unchanged` | genuine contract on `'sziklas'` | `_carrier_step_fast` IS that transport's no-crossing branch |
| `v1_v8::TestV3ChainScope` (2 ids) | genuine contract on `'sziklas'` | every claim is in units of the Sziklas readout's PERIOD, 3.8x smaller than the Collins one here |
| `h2_near_focus_table::_sziklas()` | naming slip | the closure IS the Sziklas arm and reached it through the default |
| `k2::test_jax_grad_through_carrier_leg` | genuine contract on `'sziklas'` | the co-moving step is trace-safe; the Collins leg refuses by name (see the Migration note) |
| `d5::test_dx_flatness_alone_is_not_sufficient` | fixture now closer to the oracle | see below |
| `r8::test_r8_focus_readout_survives_exact_focus` | fixture now closer to the oracle | the fail-before arm names `'sziklas'`; a NEW arm asserts the default returns a finite field at the exact focus, which is the flip's headline |
| `a24::test_the_paraxial_final_leg_does_enter_it` + `d6::_run_chain` | genuine contract on `'sziklas'` | `_default_focus_standoff` is that readout's own resolver; `_run_chain` gained an opt-in `transport=` and forwards it only when NAMED, so the rest of that file still tracks the library default |

**`d5::test_dx_flatness_alone_is_not_sufficient` deserves its own paragraph**,
because its docstring predicted this exact event twice ("a further accuracy
improvement could walk through it" -- one did in 2026-08-13, and another has
now).  The test demonstrates that a flatness-only gate passes a
DELIBERATELY BROKEN configuration (`carrier_reference='parabola'`) that sits
wide of an independent Debye oracle.  On the flipped default that
configuration reads a FWHM/oracle ratio of **0.963** -- it is no longer wide
of the oracle at all.  That is a real result about the transport and it is
recorded as one; it is NOT a reason to lower the bar.  The lesson the test
exists for needs a configuration that HAS a level failure, so the test names
the transport where this one still does.

### 4.3 Four documentation / lint gates, each fixed at the cause

| gate | why it fired | fix |
|---|---|---|
| `test_no_shipped_source_claims_a_version_the_package_has_not_reached` | seven docstring lines named 5.49.0 before `__version__` reached it | rewritten to describe the change and let the CHANGELOG carry the number -- the repository's own measured practice |
| `test_no_module_accumulates_more_version_history` | the same seven lines | same fix |
| `test_v18_5_the_5_47_0_block_citations_name_the_right_lines` | 11 citations into `carrier.py` shifted; one could not be anchored at all | the 11 re-anchored with the committed content-based tool.  The twelfth is worth recording: the 5.47.0 entry about the Collins readout's applicability window cites the BLANK line introducing the `WHERE THE COLLINS ONE-STEP READOUT APPLIES` paragraph, and the tool breaks a blank line's tie on the two lines either side of it -- a paragraph INSERTED ABOVE gave a second blank line the same 2-of-4 score and the anchor went ambiguous.  Moving the new paragraph BELOW that one restores a 4-of-4 context. **A docstring insertion, not an edit, can orphan a citation that points at whitespace.** |
| `test_no_backticked_identifier_in_the_docs_is_unresolved` | `readout_route` / `_k1` / `_reason` are stage-DICT keys, not symbols | curated with that reason beside `n_planes`, rather than widening the rule |

### 4.4 Pre-existing reds, not caused by this package

| id | where | why |
|---|---|---|
| `test_public_api.py::test_installed_metadata_version_matches_source_version` | both builds | the editable install's metadata reads 5.47.0 against a source `__version__` of 5.48.1.  **Verified on the base tree** (`git archive 49ddf4bd` extracted whole and run from there): it fails there too |
| `test_v5_2_3_walker_changelog_content.py::test_v16_synthetic_fabrication_is_caught` | WSL only | git cannot resolve a Windows worktree from WSL; the test's own message names this condition and says it is red "on the base tree too" |
| `test_v5_3_2_walker_source_line_citation.py::test_v18_5_companion_reanchor_tool_exists_and_covers_the_cited_files` and `::..._the_5_47_0_block_citations_name_the_right_lines` | WSL only | same condition, same message |

## 5. The CuPy arm

### 5.1 What was already there, and what was owed

Hygiene-2 H2-2 threaded `(xp, is_jax, bld)` through `_collins_transport` and
its helper chain with no `_jax` twin of anything, and proved the JAX half
(parity 5.5e-16 against a bar measured from the two backends' FFTs, `jax.grad`
at the cancellation floor).  Its own "decisions owed" item 2 records that the
CuPy half was exercised STRUCTURALLY and not on hardware.  This package closes
the structural side and measures as much of the hardware side as exists.

### 5.2 The premise, READ as a fact of this box

| build | cupy | devices | elementwise kernels | `cupy.fft` |
|---|---|---|---|---|
| WIN-py3.14 | 14.0.1 | 1 | **work** | `ImportError: DLL load failed while importing cufft` |
| WSL-py3.12 | not installed | -- | -- | -- |

`test_the_cupy_premise_is_read_and_not_assumed` does not pin these values; it
asserts the readings are mutually CONSISTENT, because an inconsistent one (a
working cuFFT on a box with no devices) would make every decision below
meaningless.

### 5.3 ONE implementation, gated structurally

`test_every_helper_on_the_collins_path_is_xp_parametrised` walks the CALL
GRAPH from `_collins_transport`, `_collins_carrier_leg` and
`_collins_focus_readout` and requires every module-level helper it reaches to
be either

* `xp`-parametrised, with the SPECIFIC parameters it owes pinned
  (`_collins_transport`/`_carrier_leg`/`_focus_readout`/`_input_box` take the
  field; `_collins_exact_kernel_correction` and `_tf_phase_to_H` take
  `(xp, is_jax, bld)`; `_collins_axis_chirp` and `_exact_dispersion_phase`
  take `bld`; `_fft2_pair` takes `(xp, is_jax)`; `_as_c_order` takes `xp`;
  `_to_dev` takes `(xp, is_jax)`), or
* declared HOST-SIDE BY DESIGN **with its reason** -- fourteen entries, each
  one either a wrapper over `_collins_power_marginals` (which routes through
  `backend.to_numpy` and accumulates host-side in row bands, deliberately, so
  the NumPy summation order and therefore the bit-identity contract survive)
  or a function that takes only Python floats.

A helper on the path with neither property fails the census as
**unclassified**.  That is the point: the classification is the review.

`test_no_collins_helper_demotes_the_field_to_host_numpy` closes the mutation
no VALUE test on any backend can see -- `np.asarray(env)` is bitwise a no-op
on NumPy and a silent device copy (or a bare `TypeError`) elsewhere; V-D3
records exactly that history.  It matches AST CALL NODES, not substrings,
because a substring search fired on `_collins_input_box`'s own COMMENT citing
the old spelling as the defect it fixed -- the same cross-reference false
positive H2-2's census hit on a docstring, one level deeper.  The matcher is
shown to fire on a deliberately mutated function before the clean reading is
believed.

`test_the_fft_on_the_collins_path_is_the_backend_dispatcher` asserts the
NumPy identity (`_bluestein_2d` keys its chirp-kernel cache on
`fft2 is fft_infra._fft2`) and, for the device side, that
`fft_infra._fft2` / `_ifft2` still dispatch a CuPy array to `cp.fft` in their
FIRST branch -- so the Collins chain needs no second selector, asserted rather
than assumed.

### 5.4 What RUNS on the device here

Every FIELD-INDEPENDENT grid the Collins chain builds needs no transform, so
it runs on this box's device and is compared to the host build:

| helper | cells | device vs host, relative | worst absolute | ULPs of 1 |
|---|---|---|---|---|
| `_collins_axis_chirp` | n = 64 / 256 / 512 | 3.6717e-17 / 4.2111e-17 / 4.4191e-17 | 1.570e-16 | 0.71 |
| `_tf_phase_to_H` | 512^2, arg to 3e4 rad | 4.5951e-17 | -- | -- |
| `_exact_dispersion_phase` | untilted | **0.0 exactly** | 0.0 | 0 |
| `_exact_dispersion_phase` | tilt (0.02, -0.01) | **0.0 exactly** | 0.0 | 0 |

`_backend_of` on a CuPy array returns `(cupy, False, cupy)` -- `bld is xp`,
unlike JAX, which is the whole of the `bld` threading's CuPy content and is
asserted.  `_collins_power_marginals` and `_collins_space_support` return
host NumPy from a device array and agree with the host reading.

The bar asserted is **eight ULPs of 1**, derived from the dtype rather than
fitted; the worst reading is 0.71 ULP, so there are 3.5 decades of room
below a bar that is itself the smallest thing two libms can disagree by.

### 5.5 The public leg with a device array

Premise-gated, with a decision on BOTH sides:

* **working cuFFT** -- the leg runs on the device, returns a DEVICE array, and
  is compared to the NumPy leg against a bar measured on the box from the two
  backends' own single FFT times the chain depth the transport applies.
* **broken cuFFT (this box)** -- the leg must fail AT THE DEVICE TRANSFORM,
  i.e. with the `ImportError` naming cufft, and must NOT fail earlier with
  `TypeError: Implicit conversion to a NumPy array is not allowed`, which is
  the signature of a host demotion and is the defect V-D3 fixed.  MEASURED:
  it raises `ImportError ... cufft`.  So the broken-cuFFT arm is not an
  absence of evidence -- it is the evidence that the array reached the
  transform.
* **no CuPy (WSL)** -- the NumPy leg runs and `_backend_of` resolves `bld` to
  host NumPy; asserted rather than skipped.

### 5.6 THE DEVICE RUN IS OWED

**Stated plainly, as the work package asks.**  No box available to this
package has a working `cupy.fft`: Windows has cupy 14.0.1 with one visible
device and a broken cuFFT DLL, WSL has no CuPy at all.  The arm that runs the
whole Collins leg on the device and compares it to NumPy is written and
**has never executed**.  What is proved here is (a) the chain reaches the
device transform rather than demoting, (b) every non-transform kernel on the
path agrees with the host build to under one ULP on the device, and (c) the
transform would be dispatched to `cp.fft` by the library's own dispatcher.
The end-to-end device parity is owed, on a box with a working cuFFT.

---

## 6. The `on_collins_sampling='warn'` census

With Collins the default, a caller who never saw the Kelly sampling warning
could start seeing it.  Measured across every shipped fixture this package
drives -- **21 Collins calls**: the six-cell single-step ladder at two gap
kernels, the astigmatic arm, `_multi` at K = 1 and K = 2, the chain's readout,
and two prescriptions with and without an image-plane readout:

| build | Collins calls | calls emitting the Kelly warning | conditions |
|---|---|---|---|
| WIN-py3.14 | 21 | **0** | -- |
| WSL-py3.12 | 21 | **0** | -- |

**Before the readout resolution landed, the same 21 calls fired FIVE
warnings**, all K1 and all from the one-step readout -- `_multi` at K = 1
(one) and K = 2 (four), the chain readout (one), and the singlet and doublet
readouts (one each).  That is the defect the resolution removes, and it is
why the census reads zero rather than "a few".

The guard is not dead, and that is asserted separately: a caller-NAMED output
lattice has no complementary form to fall back to, and the guard still speaks
there.

---

## 7. What retiring the focus-standoff machinery would take

WP-B4 section 5 listed the machinery a flip would retire:
`_default_focus_standoff` (`carrier.py:3958` at that commit),
`_beam_containment_standoff`, `_check_focus_containment`,
`_small_extent_focus_standoff_f`, `_achievable_focus_margin`, the
`_FOCUS_STANDOFF_*` and `_FOCUS_READOUT_CONTAINMENT_*` constants, the replica
guard's standoff coupling, and C1 -- roughly 900 lines plus their pins.

**Nothing is removed here, and this package makes the case for removal
WEAKER rather than stronger.**  The machinery is no longer only `'sziklas'`'s:
it now serves the DEFAULT's own readout fallback, which is the route every
chain readout in this package's fixtures takes.  Retiring it would need, in
this order:

1. **The one-step readout to be representable on real exit lattices.**  That
   is `_collins_readout_k1 <= 1` where it reads 82.36 / 41.45 / 21.46 at
   N = 256 / 512 / 1024 on WP-B4's relay and falls only as `1/dx`.  There are
   three ways and they are not equivalent: a chain that exits on a far finer
   grid (N ~ 16000 on this fixture -- a 256x memory cost), a LONGER final
   distance on a small exit beam (K1 carries `|A| r / |B|`, so the WP-A6
   fixture reads 0.16), or a re-derivation of the readout that does not form
   the pre-chirp on the exit pitch at all.  Only the third retires the
   machinery for every design.
2. **A decision about `final_distance = 0` with a readout.**  It works today
   on the default because the resolution routes it to the Sziklas readout.
   With that route gone it is a refusal, and it is a shipped configuration.
3. **The two Sziklas-only `focus_readout` keys.**  `standoff` and
   `on_focus_containment` are refused on `'collins'`; nine ids in
   `test_niche_d2_chain_multi.py` and one calibration helper use them as a
   CONTRACT (they size a Bluestein period that is linear in the leg).  Those
   fixtures would have to be rewritten against a period that is
   `lambda |z| / dx` of the input grid instead -- which is a re-statement of
   what the D2 guard is for, not a re-pinning.
4. **The replica guard's standoff coupling**, which is the same point one
   level down.
5. **C1**, which is about a stop grid sized from the carrier.  With no stop
   grid it has no referent -- this is the one item that really does fall out
   for free once (1) is settled.

Until (1) is settled the machinery is load-bearing on the default path, and
removing it would turn every readout that currently falls back into either a
refusal or an aliased answer.

---

## 8. Test runs

Exact commands in section 9.  Counts and tails are in the final report.

## 9. Files changed

| file | what |
|---|---|
| `lumenairy/propagators/carrier.py` | the default on three signatures; `_collins_readout_k1` and the readout's route resolution; `_publish_readout_route`; the leg's fallback routed through the Sziklas entry point and opened to astigmatic carriers; three internal call sites named; docstrings |
| `docs/history/carrier.md` | re-recorded with its reason (`--check` green over every module) |
| `tests/unit/test_c3_collins_default.py` | NEW, 25 ids |
| `tests/unit/test_audit2609_b4_collins_transport.py` | three classes restated (section 4.1) |
| `tests/unit/test_niche_d2_chain_multi.py` | the standoff/period fixtures name their transport (section 4.1) |
| `scripts/check_doc_identifiers.py` | three curated stage-dict keys |
| `CHANGELOG.md` | four blocks inside `## [Unreleased]` plus the Migration paragraph; 12 citations in the `[5.47.0]` block re-anchored |
| `Migration-Guide.md` | the 5.49.0 section |
| `validation/probe_c3_collins_default/` | `clib.py`, four probes, the bit-identity driver, and the JSON for both builds |
