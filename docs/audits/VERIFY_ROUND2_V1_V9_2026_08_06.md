# RE-VERIFY ROUND 2 -- findings V1-V9 against 8b60135 + 4fc7c7d

**2026-08-06.  REPORT-ONLY.**  Same independent verifier as
`docs/audits/VERIFY_D1_D11_2026_08_06.md` (round 1, against f2c0eeb).  Every
probe below is a scratchpad script from round 1 re-run UNMODIFIED against the
current tree, plus new probes for the claims that did not exist in round 1.
Nothing in `lumenairy/**` or `validation/**` was modified.

Tree: `git status --short` empty, HEAD = `4fc7c7d`.  Box: Windows 11,
Python 3.14.6, numpy 2.4.4, scipy 1.17.1, numba present, 24 logical CPUs.
All test runs `LUMEN_PIN=0 -p no:randomly`.

Under review: `8b60135` (V7, V9) and `4fc7c7d` (V1-V6, V8, plus the `mft.py`
sibling of V3).  Fix docs `FIX_V1_V8_2026_08_06.md`,
`FIX_V4_V5_2026_08_06.md`.

---

## Verdict table

| # | Finding | Verdict | Headline number |
|---|---|---|---|
| **V9** | cp1252-unsafe addendum | **REGRESSED** | "0 non-ASCII" achieved by truncating the file 7100 -> **0 bytes**; all 144 lines deleted |
| **V7** | numba unkeyed in the chain-A cache | **CLOSED-WITH-CAVEAT** | `numba_available` IS keyed (20 -> 21 fields), but re-derives the fact with `import numba` instead of reading the library's own `_NUMBA_AVAILABLE`; flipping that flag leaves the digest unchanged |
| V1 | worst-case half of the D2 headline | **CLOSED** | 54-cell matrix: NEW worst **9.124e-02** vs 6.0 zR 1.264e-01 (1.39x better), 0.8 zR 1.073e+00 (11.8x better) |
| V2 | band worse than BOTH constants | **CLOSED (bounded)** | 9 cells -> **8**; worst loss 5.08x -> **1.98x**; **0 of 54 cells got worse**; geomean 4.196e-03 -> **3.127e-03** |
| V3 | replica guard `centre_out`-blind | **CLOSED** | ghost gone; boundary/formula ratio **1.000000003**; exchange rate exactly **0.500000**; exact readout weighs the chief-ray residual; **all three** MFT propagators centre-aware |
| V4 | `on_fit_domain_basis` ungated | **CLOSED** | **0 / 24** junk failures; `'ignore'`/`'off'` silence and are bit-identical to `'silent'`; entry gate fires on the default basis |
| V5 | cost gate backend- and size-blind | **CLOSED** | crosscontam **4 -> 0** dispatches, sizeextrap **4 -> 0**; every homogeneous count unchanged; bit-identity max diff **0.000e+00** |
| V6 | over-refusal / thin message | **CLOSED-BY-DESIGN** | message quotes span, overshoot, alias count (**6 -> 24 -> 86**, growing), largest safe `N_out`; bar deliberately kept hard with a stated justification |
| V8 | `tilt` silently dropped under `'fresnel'` | **CLOSED** | exactly **1** warning naming both knobs on all four paths; 0 at `tilt=0`; 0 under `auto`/`exact`; astigmatic covered; drop still real |

**One REGRESSED (V9).  One CLOSED-WITH-CAVEAT (V7).  Seven CLOSED.**

Three round-2 "misses" I recorded mid-pass turned out to be **my own probe
bugs**, corrected before grading and reported here so the record is not
misleading: the V3(b) fixture omitted the `centre=` argument; the V6 token
match looked for `exceed` / `alias` where the message says `over` / `ALIASES`;
and the V8 numpydoc check split the docstring on the literal `Returns`, which
also occurs inside the `R_carrier` description.  None of the three was a
library defect.

---

## R1. V9 -- REGRESSED

```
  git show f2c0eeb:docs/audits/HANDOFF_TRACED_EXACT_2026_08_05_ADDENDUM.md | wc -c   ->  7100
  git show 8b60135:docs/audits/HANDOFF_TRACED_EXACT_2026_08_05_ADDENDUM.md | wc -c   ->     0
  git show 8b60135 -- <that file>    ->   @@ -1,144 +0,0 @@   (144 deletions, 0 insertions)
  working tree                       ->   0 bytes, 0 lines
  non-ASCII bytes in the file        ->   0      <- the stated goal, met vacuously
```

The three cp1252-unsafe characters I reported (U+2192 x2 at lines 16 and 123,
U+2212 at line 107) were removed by deleting the entire document rather than
transliterating them.  What was destroyed is substantive and is cited by name
elsewhere in the repo:

* the `newton_fit` revert rationale -- why spline must not be the default (the
  three `newton_fit != 'spline'` gates, and the measured non-finite exit field
  on design 121's post-DOE groups);
* the correction that `6dfc79d`'s test status was read off a ~34%-complete run,
  and the 15 real failures it hid (`c6` 2, `c11` 7, `c12` 6);
* the repository-state / authorship notes telling the next agent which
  uncommitted hunks belonged to whom;
* the "Lesson worth carrying" section (three times in that campaign a plausible
  result turned out to be an ABSENT one).

Still referenced by `docs/audits/FIX_D2_D3_D5_2026_08_06.md` S3.1 (which sends
the reader there for the revert) and by `f2c0eeb`'s own commit message ("their
addendum + hunks ... preserved verbatim").  Both now point at an empty file.

**Severity MED.  No code impact.**  Remedy: restore
`git show f2c0eeb:<path>`, replace `U+2192` with `->` and `U+2212` with `-`,
and leave the other 26 non-ASCII characters (all cp1252-safe) alone.

Encoding of the two new commits themselves is clean: **0 added lines with any
non-ASCII byte in either `8b60135` or `4fc7c7d`.**

---

## R2. V7 -- CLOSED-WITH-CAVEAT

```
  top-level keyed fields: 20 -> 21
  numba-related keys: ['numba_available'] -> {'numba_available': True}
```

The axis is now in the key, so the ordinary deployment case (numba installed
vs not) is covered.  The caveat is that the key does not read the flag that
actually decides which evaluator runs:

```
  lumenairy/elements/_lens_traced.py:62
      _NUMBA_AVAILABLE = _ilu.find_spec("numba") is not None       <- the GATE
  validation/repro_traced_carrier_121/_d121_common.py:197
      def _numba_available():  try: import numba; return True      <- the KEY

  digest with _lens_traced._NUMBA_AVAILABLE flipped:
      changed = False      (b4b55744b5ea -> b4b55744b5ea)
```

The two expressions disagree exactly in the case that is most common in
practice: a numba whose module is visible to `find_spec` but whose `import`
fails (the numba/numpy ABI mismatch).  There the library takes the numba
branch while the key records `numba_available: False`.  The field difference I
measured in round 1 (rel 5.4e-12, max abs 1.78e-13, `n_workers=1`, no pool) is
driven by the GATE, not by the key's re-derivation.

**Remedy (one line):** key
`lumenairy.elements._lens_traced._NUMBA_AVAILABLE` directly.  Graded
CLOSED-WITH-CAVEAT rather than NOT-CLOSED because the deployment axis I named
IS now keyed; I could not construct the disagreeing environment on this box.

---

## R3. V1 / V2 -- CLOSED

`p_d2_sum.py` re-run verbatim (6 NA x 9 ext, 6 `w0` window, my own separable
matrix-DFT Fresnel quadrature `oracle.py`, no lumenairy code), then diffed
cell-by-cell against my round-1 run.

**All five claims verified:**

| claim | verified |
|---|---|
| geomean **3.127e-3** | **3.127e-03** -- exact |
| worst **9.124e-2** | **9.124e-02** -- exact |
| no cell worse than before | **23 improved, 31 unchanged, 0 worse** (54 cells) |
| both old constants beaten on geomean AND worst-case | geomean 3.127e-03 vs 1.794e-02 / 1.466e-02; worst 9.124e-02 vs 1.073e+00 / **1.264e-01** |
| residual band worst loss 1.98x | NA 0.150 ext 1.80: 2.690e-02 / 1.360e-02 = **1.978x** |

```
               geomean      worst        median
  NEW round-1  4.196e-03   1.619e-01    1.365e-02
  NEW round-2  3.127e-03   9.124e-02    6.923e-03
  0.8 zR       1.794e-02   1.073e+00    3.705e-02    (unchanged, as expected)
  6.0 zR       1.466e-02   1.264e-01    2.117e-02    (unchanged, as expected)

  cells losing to 0.8 zR : 13/54 -> 12/54
  cells losing to 6.0 zR : 17/54 -> 13/54
  cells losing to BOTH   :  9    ->  8     worst loss 5.08x -> 1.98x
```

V1 is the one that flipped outright: the new worst cell is **1.39x better than
6.0 zR's worst**, where round 1 measured it 1.28x WORSE.  The worst-case half
of "both old constants beaten" now holds on my extended grid (which adds
NA 0.05 / 0.15 and ext 1.8 / 2.2 to theirs).

The eight surviving losses are the same band I named, now bounded:

```
   NA=0.100 ext=1.80  NEW=1.884e-02   vs 0.8zR 1.61x   vs 6.0zR 1.27x
   NA=0.100 ext=2.00  NEW=9.809e-03   vs 0.8zR 1.61x   vs 6.0zR 1.32x
   NA=0.150 ext=1.80  NEW=2.690e-02   vs 0.8zR 1.98x   vs 6.0zR 1.59x   <- worst
   NA=0.150 ext=2.00  NEW=1.490e-02   vs 0.8zR 1.62x   vs 6.0zR 1.43x
   NA=0.200 ext=1.80  NEW=3.087e-02   vs 0.8zR 1.51x   vs 6.0zR 1.35x
   NA=0.200 ext=2.00  NEW=2.112e-02   vs 0.8zR 1.08x   vs 6.0zR 1.05x
   NA=0.278 ext=1.50  NEW=4.736e-02   vs 0.8zR 1.05x   vs 6.0zR 1.01x
   NA=0.350 ext=1.50  NEW=9.124e-02   vs 0.8zR 1.02x   vs 6.0zR 1.06x
```

Round 1's worst was 5.08x (NA 0.10 ext 1.80); it is now 1.98x, and the two
ext-1.5 rows are ties inside a different code path (the near-focus bridge).

**"Constant in disguise" is gone.**  `f` read directly out of
`_default_focus_standoff`, expressed in the resolver's own `zR`:

```
  NA     1.2   1.5   1.8   2.0   2.5   3.0   3.5  3.69 | 4.0   6.0   10    20   sub-3.695 spread
  0.050 7.018 7.018 7.018 7.018 4.940 2.949 1.933 1.732|1.333 0.630 0.338 0.162      4.052x
  0.100 1.732 1.732 5.133 4.475 2.965 1.987 1.732 1.732|1.333 0.630 0.338 0.162      2.964x
  0.150 1.732 1.732 3.693 3.255 2.266 1.732 1.732 1.732|1.333 0.630 0.338 0.162      2.132x
  0.278 1.732 1.732 2.266 2.042 1.732 1.732 1.732 1.732|1.333 0.630 0.338 0.162      1.308x
  0.350 1.732 1.732 1.901 1.732 1.732 1.732 1.732 1.732|1.333 0.630 0.338 0.162      1.098x
```

Round 1 measured `f == f_cap = 1.732051` EXACTLY at every sub-3.695 extent and
every NA.  It now varies by 1.10x-4.05x.  `f_cap` survives as the FLOOR of the
trade (the shipped rule may only LENGTHEN the leg), which is why it still
appears in patches -- that is the documented design, not the old degeneracy.
The spread is smallest at high NA (1.098x at NA 0.350), consistent with the
fix's own `L ~ NA^3` argument that a fast beam cannot afford extra leg.

---

## R4. V3 -- CLOSED, including the new scope

### R4.1 My round-1 ghost probe, re-run

Same fixture (NA 0.10, ext 3.0).  The period is now 37.9433 um (was 33.0797)
because V2 lengthens this cell's leg, so the same window is 0.6711 periods:

```
  centre_out   span (periods)   result
   0.00 per       0.671         RETURNED  peak|F|=1.570068e+02  relL2=3.738e-04  0 warnings
   0.25 per       1.171         REFUSED   RuntimeError
   0.50 per       1.671         REFUSED
   1.00 per       2.671         REFUSED
   2.00 per       4.671         REFUSED
   3.00 per       6.671         REFUSED
```

The full-amplitude ghost is gone.  Round 1 on this fixture: peak 1.570252e+02
where the truth was 1.499920e-03 -- **ratio 1.05e5, zero warnings, no
refusal**.  On-axis still returns, and more accurately than before
(3.738e-04 against 5.259e-04).

The guard is not simply refusing everything off-axis.  With a narrower window
the sub-period offsets go through, and the returned field decays physically
instead of ghosting:

```
  0.30-period window:  centre 0.00 p (span 0.266 p) -> RETURNED peak 1.5701e+02
                       centre 0.15 p (span 0.566 p) -> RETURNED peak 1.5057e+02
                       centre 0.34 p (span 0.946 p) -> RETURNED peak 3.7351e-01
                       centre 0.36 p (span 0.986 p) -> RETURNED peak 1.1549e-01
                       centre 1.00 p (span 2.266 p) -> REFUSED
```

### R4.2 The boundary tracks `(period - window)/2` at weight two

Bisected to 60 iterations on the direct API, three window sizes:

```
  period = 37.9433 um
  N_out= 96  window 25.4648 um | formula  6.23927 um | measured  6.23927 um | ratio 1.000000003
  N_out= 48  window 12.7324 um | formula 12.60546 um | measured 12.60546 um | ratio 1.000000002
  N_out= 24  window  6.3662 um | formula 15.78856 um | measured 15.78856 um | ratio 1.000000001
     exchange rate |d(boundary)/d(window)| = 0.500000  and  0.500000
```

Exactly `1/2` twice: the offset really does enter at weight two.  Independent
reproduction of the fix doc's table, on my own bisection.

### R4.3 The exact readout weighs the chief-ray RESIDUAL

`_dec` is DECLARED by the caller (the `centre=` argument), not measured from
the field.  My first attempt omitted it and therefore probed the on-axis path;
with `centre=` supplied (beam at 0.05 / 0.15 / 0.20 / 0.25 mm on a 1.024 mm
grid, `window_factor=4.0`, period 240.0 um):

```
  chief ray at 0.05 mm, read ON it  : RETURNED peak=2.2554e+00   period=240.0 um
       same beam read on the AXIS   : REFUSED   (message names the chief-ray residual)
  chief ray at 0.15 mm, read ON it  : RETURNED peak=2.2554e+00
       same beam read on the AXIS   : REFUSED   (residual note present)
  chief ray at 0.20 mm, read ON it  : RETURNED peak=2.2554e+00
       same beam read on the AXIS   : REFUSED   (residual note present)
  chief ray at 0.25 mm, read ON it  : RETURNED peak=2.2554e+00
       same beam read on the AXIS   : REFUSED   (residual note present)
  chief ray + 0.30 mm               : REFUSED   (residual note present)
  chief ray, WIDE N_out=400         : REFUSED
```

**CONFIRMED:** a decentred congruence read out on its own chief ray costs none
of its period budget, at every offset probed, with an identical peak; reading
the same beam on the optical axis is refused and the message says why.

Worth stating because it is contingent: the property depends on the caller
DECLARING `centre=`.  A decentred field handed in without it is treated as
axis-centred (the period becomes the whole input grid and the ABSOLUTE
`centre_out` is weighed).  That is self-consistent -- the transform really is
axis-centred then -- but it is not automatic.

### R4.4 NEW SCOPE -- `mft.py`, all three MFT propagators

This is the sibling their own doc left open (`FIX_V1_V8` S7 item 3 says
`mft.py` "is outside this fix's writable scope").  It is fixed in the same
commit.  My probe, window ~0.6 period, four offsets, warning counted on the
stable `PERIODIC REPLICA` token:

```
  angular_spectrum_propagate_mft   period 256.000 um, window 153.000 um (0.598 p)
     centre 0.00 p  span 0.598 p  warnings=0   expect silent   OK
     centre 0.10 p  span 0.798 p  warnings=0   expect silent   OK
     centre 0.25 p  span 1.098 p  warnings=1   expect WARN     OK
     centre 0.60 p  span 1.798 p  warnings=1   expect WARN     OK
  fresnel_propagate_mft            period 158.250 um, window 94.000 um (0.594 p)
     same four rows, same result                               OK
  fraunhofer_propagate_mft         period 158.250 um
     same four rows, same result                               OK
```

Off-origin `2|c| + W > period` warns; the IDENTICAL window on-origin does not.
All three propagators, 12/12 rows as expected.

The false sentence is gone from both the docstring and the live message:

```
  docstring says "NOT on ``centre_out``"                    : True
  live message contains "beyond +/-period/2 of centre_out"  : False
```

Warning here and REFUSAL on the two carrier readouts is the right split:
`mft.py` is a general propagator whose caller may legitimately want replicas.

---

## R5. V4 -- CLOSED

```
  canonical:  'warn'   -> exc=None      ann=1  returned=True
              'error'  -> ValueError    ann=0  returned=False
              'silent' -> exc=None      ann=0  returned=True
  aliases:    'ignore' -> exc=None      ann=0  bit-identical to 'silent': True
              'off'    -> exc=None      ann=0  bit-identical to 'silent': True

  junk, 24 values (case variants, whitespace, '', 'none', 'quiet', 'raise',
        None, 1, 0, True, 1.0, tuple, list, bytes, dict, nan, -1):
              junk failures: 0 / 24
              (all ValueError, all naming the knob AND 'warn'/'error'/'silent')

  ENTRY gate:  default basis + on_fit_domain_basis='Error' -> ValueError
               (the knob is never consulted on that path, so this proves it is
                an entry gate, not a gate wired next to the read site)
  default:     all-default call -> exc=None, ann=0
               polynomial + fit_radius_beam_factor -> ann=0, 0 other warnings
```

Both consequences I named are closed: `'Error'` is now fatal-as-asked, and
`'ignore'` genuinely silences instead of being accepted-and-inert.

---

## R6. V5 -- CLOSED

`p_d1_pool.py` re-run unmodified, SHIPPED cost gate (nothing neutralised), one
scenario per fresh process, `n_workers=4`, 65 536 Newton points per group
unless stated.

```
  scenario                             round-1   round-2   expected
  crosscontam phase 1 (spline x2)          0         0        0
  crosscontam phase 2 (POLYNOMIAL x4)      4         0        0   <-- the defect
       state after phase 1: (nw=4, 0.5017 s, 2)
       state after phase 2: (nw=4, 0.0333 s, 4)   <- RE-ARMED, not inherited
       phase-2 wall: 5.717 s -> 3.718 s
  sizeextrap  phase 1 (116 281 pts x2)     0         0        0
  sizeextrap  phase 2 (16 384 pts x4)      4         0        0   <-- the defect
       state after phase 2: (nw=4, 0.0959 s, 4)   <- RE-ARMED
  default6  (polynomial, 6 groups)         0         0        0
  auto6     (newton_fit='auto')            0         0        0
  spline2                                  0         0        0
  spline3                                  1         1        1
  spline6                                  4         4        4
  nwalternate (nw 4/2/4/2/4/2)             0         0        0
```

Both defect sequences go **4 -> 0**.  Every homogeneous behaviour is unchanged
-- the fix keys the evidence, it does not move the bar.  Bit-identity holds:

```
  spline6 pooled  sum|F| = 4.172976713666e+03   (4 dispatches)
  spline6 serial  sum|F| = 4.172976713666e+03   (nw=1, 0 dispatches)
  np.array_equal = True      max abs diff = 0.000e+00
```

---

## R7. V6 -- CLOSED-BY-DESIGN

The message now carries the measurement.  Verbatim from a live refusal:

```
  ... spans 2*|centre_out| + N_out*dx_out = (6.546479e-05 x 2.546479e-05 m),
  i.e. 1.725x the period (2.752147e-05 m over, about 96 sample(s) per edge
  that are literal ALIASES of samples already in the window). ...
  |centre_out| = 2.000000e-05 m alone already exceeds half the period, so NO
  window is faithful at this offset: bring centre_out inside +/-1.897166e-05 m
  of the field origin, or pass standoff >= 4.244132e-05 m ...
```

Checked and present: per-axis SPAN; overshoot in metres; alias count; largest
safe `N_out` (or, when the offset alone exceeds half a period, how far in to
bring `centre_out`); the offset itself; the escape hatch.  The alias count is a
real measurement, not boilerplate -- it grows with the overshoot:

```
  span/period 1.02 -> "about 6 sample(s) per edge"
  span/period 1.10 -> "about 24 sample(s)"
  span/period 1.50 -> "about 86 sample(s)"
```

**The bar is unchanged and still hard**, which is what I asked about:

```
  span/period 0.980000  returned              actual relL2 (waived) = 4.6013e-04
  span/period 1.0000001 REFUSED RuntimeError  actual relL2 (waived) = 4.6490e-04
  span/period 1.020000  REFUSED               actual relL2 (waived) = 4.6921e-04
  span/period 1.100000  REFUSED               actual relL2 (waived) = 4.8990e-04
  span/period 1.500000  REFUSED               actual relL2 (waived) = 5.7726e-04
```

So the over-refusal I measured is still there in the same form.  I grade it
CLOSED-BY-DESIGN rather than NOT-CLOSED because the decision to keep the bar is
explicit and is argued from a property true independently of the field
(`E(u + p) == E(u)`, so past one period some returned samples are literal
copies); my own ladder is reproduced in their doc rather than argued away; and
the caller now gets the numbers needed to override knowingly.  A geometry-only
guard cannot state a looser bar correctly-by-construction.

---

## R8. V8 -- CLOSED

```
  R=inf     fresnel + tilt : warnings=1   naming BOTH knobs=1
  R=-0.2    fresnel + tilt : warnings=1   naming BOTH knobs=1
  R=+0.5    fresnel + tilt : warnings=1   naming BOTH knobs=1
  R=-inf    fresnel + tilt : warnings=1   naming BOTH knobs=1
  R=inf / -0.2   fresnel, tilt=0         : warnings=0
  R=inf / -0.2   auto + tilt             : warnings=0
  R=inf / -0.2   exact + tilt            : warnings=0
  astigmatic (-0.2,-0.3) 'auto'   + tilt : warnings=1, names tilt
  astigmatic (-0.2,-0.3) 'fresnel'+ tilt : warnings=1, names tilt
  equal-radii (-0.2,-0.2) 'auto'  + tilt : warnings=0   (scalar path -- correct)
```

Exactly once per call, never on the exact kernel, never at zero tilt.  The
astigmatic arm is covered -- an adjacent silent drop I did NOT name in round 1.

The drop itself is still real, i.e. the fix announces rather than quietly
starting to honour it:

```
  R = inf / -0.2 / +0.5 : np.array_equal(env(tilt), env(no tilt)) = True
```

Numpydoc: `gap_kernel` and `tilt` now have full Parameters entries on
`propagate_carrier_referenced` (docstring lines 40 and 50, inside the
Parameters block spanning lines 13-62), each describing the V8 behaviour.  My
three earlier negative readings on this were a broken section-split in my own
probe, not a gap in the docstring.

---

## R9. Sibling sweep -- one new observation, no new hole from these fixes

One sibling of each fixed class, briefly.

**V4 class (ungated string mode knob) -- CLEAN.**  Junk into EVERY
string-defaulted parameter of all five carrier entry points
(`propagate_carrier_referenced`, both focus readouts, chain, multi):

```
  knobs probed: 33      ALL gated, and every message names its own knob.
```

**V8 class (argument accepted and ignored on a branch) -- CLEAN.**

```
  paraxial readout, bandlimit True vs False     : fields differ (live)
  exact readout,    tilt=0 vs tilt=(0.03,-0.02) : fields differ (live)
```

**V5 class (process-global measurement keyed too loosely) -- CLEAN.**  The
pool deferral quintuple
(`_POOL_DEFERRED_NWORKERS/_CLASS/_SECONDS/_POINTS/_COUNT`) is the only mutable
process-global MEASUREMENT in `_lens_traced.py`; the other module globals are
configuration or a lazy numba import.

**V3 class (a budget computed about the grid origin) -- ONE NEW OBSERVATION,
PRE-EXISTING, not introduced by these commits.**

`_default_focus_standoff` and `_near_focus_needs_bridge` both size a
containment budget from `half = N*dx/2` and a beam radius measured ABOUT THE
GRID ORIGIN, and neither takes a `centre`:

```
  _near_focus_needs_bridge(E_env, R, R_out, wavelength, dx, dy)   -- no centre
  _default_focus_standoff(env, R, z, wavelength, dx)              -- no centre
```

`_envelope_amp_radius` DOES have a `centre` argument (added under niche D6
precisely because "a second moment about the WRONG point reads
`sqrt(2 x_c^2 + w^2)`"), and `_default_focus_standoff` calls it without one.
Measured, NA 0.10 / ext 3.0, grid half-width 1500 um, true beam radius 500 um:

```
    centre    w_env(read)   ext(seen)   ext(true, near edge)   standoff
     0.00w      500.000 um     3.000            3.000          63.2389 um
     0.50w      612.371 um     2.449            2.500          57.3243 um
     1.00w      865.941 um     1.732            2.000          36.4956 um
     1.50w     1170.326 um     1.282            1.500          10.0632 um
```

At a 1.5 `w` decentre the resolver reads the beam **2.34x too wide**, sees
ext 1.282 where the near edge gives 1.500, and resolves a leg **6.3x shorter**
than the on-axis one (10.06 um vs 63.24 um).  That also shortens the Bluestein
period 6.3x, which makes the (now correct) replica guard far more likely to
refuse.  Same class as V3 -- a budget evaluated about the origin for a beam
that is not there.

Reachable only by handing a decentred field straight to
`carrier_referenced_focus_readout`, which has no `centre` argument; the tilted
chain path does its own chief-ray bookkeeping.  **Severity LOW-MED; the
accuracy impact was NOT scored against the oracle.**

**Not probed:** the `_fit` one-pixel-headroom claim in the multi orchestrator
(`FIX_V1_V8` S1.7).  My attempted probe never reached the readout (an empty
`groups` list raises first), so that claim is UNVERIFIED-BY-PROBE here; the
shipped `test_fix_v1_v8_readout_guard_and_standoff` suite covers it.

---

## R10. Cross-cutting

**Collection counts, all matching the fix docs:**

```
  test_fix_v1_v8_readout_guard_and_standoff : 37   (doc: 37 passed)
  test_fix_v3_mft_centre_window             :  5   (new with 4fc7c7d)
  test_fix_d5_fit_domain_basis              : 38   (doc: 38 passed, was 9)
  test_niche_newton_pool_both_fits          : 23   (doc: 23 passed, was 16)
  test_niche_tight_focus_readout            : 15   (doc: 15 passed)
  test_niche_d2_chain_multi                 : 38   (doc: 38 passed)
  test_carrier_referenced                   : 18   (doc: 18 passed)
                                        total 174
```

Run as one batch on the settled tree: **174 passed, 0 failed, 0 skipped**
(662.66 s).  Collection total and pass total agree, so no test was silently
skipped or deselected.

**ruff:** `ruff check lumenairy/` -> All checks passed.  `ruff check` over
every library and test file changed by `4fc7c7d` -> All checks passed.  (The
three pre-existing I001 from round 1 are in files this wave did not touch.)

**Encoding:** `8b60135` and `4fc7c7d` each add **0** lines containing any
non-ASCII byte.  The only encoding item is V9 above, which is a deletion.

---

## R11. What I could not probe in round 2, and why

* **`_fit`'s one-pixel headroom** in `propagate_traced_carrier_chain_multi`
  (R9).  UNVERIFIED-BY-PROBE.
* **The V7 disagreeing environment** (numba visible to `find_spec` but failing
  on `import`).  Could not construct it on this box; the code fact is reported
  instead of a measurement.
* **Accuracy impact of the decentre-blind standoff** (R9).  The resolver's
  behaviour is measured; whether the 6.3x shorter leg is actually less accurate
  for a decentred beam was not scored.
* **CuPy / JAX backends**, **design 121**, **a full-suite regression** --
  unchanged from round 1.
