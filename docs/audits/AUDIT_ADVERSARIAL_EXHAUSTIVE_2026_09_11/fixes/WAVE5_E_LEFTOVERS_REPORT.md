# WAVE5-E -- the six leftovers the verifiers of items A (WP-B12) and D (WP-B14) recorded

Branch `fix/wave5-item-e-leftovers`, worktree `C:/tmp/lum_e`, base
`ede07f30` (`wave5/audit-leftovers`).  PRE trees for the re-measurements:
`C:/tmp/lum_e_pre` at `96cb2096` (the WP-B14 base) and `C:/tmp/lum_e_b12pre` at
`ede07f30`.

Every number below was **measured on this tree**, on both builds, and every
probe prints `lumenairy.__file__`.  Nothing here is read out of the reports that
requested the work -- where a published figure is quoted it is because this pass
reproduced it, and where it did not reproduce that is said.

## 0. The two builds, and how every run was pinned

| | Windows | WSL (the CI condition) |
|---|---|---|
| interpreter | CPython **3.14.6** (MSVC 1944) | CPython **3.12.3** (GCC 13.3) |
| numpy | 2.4.4 | 2.4.6 |
| BLAS | `libscipy_openblas64_` 0.3.31.188.0, Haswell | same version, Haswell |
| pyFFTW | 0.15.1 | 0.15.1 |
| jax | 0.11.0 | 0.11.0 |

A third interpreter was built for item E1 alone: a clean Windows venv carrying
**numpy 2.4.6**, to separate the NumPy release from the wheel.

Every invocation carries `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1` on the command line, `PYTHONPATH` pinned to the tree under
test, pytest with `--capture=sys -p no:randomly`, and every tail grepped for
`passed|failed|error|no tests ran`.

Probes and per-arm JSON: `validation/probe_fft_elision/` (E1) and
`validation/probe_wave5_e/` (E2-E5).

---

## 1. Verdict table

| item | what it asked for | outcome |
|---|---|---|
| **E1** | decide by measurement between privatising the FFT dispatchers' return and scoping the contract | **SCOPED (remedy b)**, on a cost/benefit measurement that is one-sided: the cost is 13-22 % of the ASM hot path, the benefit is **zero** for every lumenairy output |
| **E2** | a durable default-order fail-before for the C8 support bound | **BUILT**, 4 ids / 12 s, a ladder over seven annuli on a 256^2 geometry; the cell chosen is the one of 47 candidates that also trips at 512 |
| **E3** | re-derive the s10 ULP bar two-sided from first principles | **RE-DERIVED** at run time, 41.53-48.25 ULP against a 16-arm envelope of 0-3 ULP and a signal at 5.16e+08 ULP; two stale docstring readings corrected (the bar read 27.3-34.0 until the products term was corrected from three roundings per leg to five, VERIFY-WAVE5-E D8 / section 11.9) |
| **E4** | correct VERIFY-B14 D2 (glass red) and D3 (GBD budget scope) | **CORRECTED**, both re-measured on the PRE tree / on this tree before being written down |
| **E5** | freeze dead rays in the exit-vertex projection; refuse an immersed FGA exit | **BUILT**, in two rounds -- the naive mask broke 7 WP-B12 pins and the diagnosis was O-4, not the change |
| **E6** | note that VERIFY-B14 7a is confirmed | **NOTHING TO BUILD**; the ledger entry is section 7 below |

---

## 2. E1 -- `fft_infra`: the ping-pong changes the OBJECT, not the values

### 2.1 What was decided, and on what

VERIFY-WP-B14 section 3 established the mechanism.  Restated and re-measured
here:

* `_fft2` / `_ifft2` **are** functions of their inputs.  Six evaluations of each
  on one fixed operand give **one** byte image, in both modes, at n = 256 and
  512, on both builds.
* What the ping-pong changes is the **kind of object**: `owndata` reads `False`
  with it on and `True` with it off, at every n >= `FFTW_MIN_SIZE`, on both
  builds (`e1_decision_*.json`).
* NumPy's `temp_elide` claims an operand only when it is an unreferenced,
  NumPy-**owned** temporary, so the workspace view is never elidable and
  `buf.copy()` is.

The decision was between **(a)** the dispatchers returning a private copy at the
shapes where it matters and **(b)** scoping the contract sentence.  Both sides
were measured.

**Cost of (a)** (`probe_e1_copy_cost.py`, best-of-7, warm plans, `e1_copy_cost_*.json`):

| n | ASM Windows | ASM WSL | Fresnel Windows | Fresnel WSL | one `buf.copy()` / one `_fft2` |
|---|---|---|---|---|---|
| 512 | **+22.1 %** | -12.5 % (noise) | +6.1 % | -3.1 % (noise) | 0.300 (Win) / 0.010 (WSL) |
| 1024 | **+19.0 %** | +0.9 % | +4.3 % | +1.2 % | 0.368 / 0.043 |
| 2048 | **+18.6 %** | **+13.0 %** | +2.9 % | +3.9 % | 0.418 / 0.349 |

The two negative WSL cells at 512 are contention, not a speed-up from copying;
they are reported, never asserted.  The load-bearing rows are the ASM column and
the last column: the copy is 30-42 % of a forward transform on both builds at
the largest shape, which is the same order the module's own byte-cap note
already measured at 8192 (+65 %).

**Benefit of (a): zero.**  No in-library site that multiplies a dispatcher
result hands the elision anything it can claim.  An AST walk over all 236
modules finds **ten** such sites, none elidable (corrected 2026-09-19,
VERIFY-WAVE5-E D6 -- this paragraph and the module note used to name six and
say "every"):

| site | spelling | other operand |
|---|---|---|
| `asm.py:919` | `_fft2(E_in) * H` | NAME |
| `asm.py:922` | `_fft2(ifftshift(E_in)) * H` | NAME |
| `asm.py:1147` | `_fft2_nd(...) * H[None, :, :]` | basic-slice VIEW |
| `asm.py:1391` | `_fft2(ifftshift(E_demod)) * H` | NAME |
| `carrier.py:1401` | `_fft2(E) * H` | NAME |
| `carrier.py:7126` | `_fft2(_e) * ramp` | NAME |
| `fresnel.py:216` | `_fft2(...) * H` | NAME |
| `rs.py:936/939/942` | `E_fft * H`, `E_fft = _fft2(...)` | NAME |

`H[None, :, :]` is a fresh object but a view (`owndata` False), so
`temp_elide` cannot claim it either; the three `rs.py` sites are the more
fragile shape, because the non-owning result is held under a name and the next
edit to that line has no `_fft2(` in front of it.  The enumeration is no longer
maintained by hand: `tests/unit/test_verify_wave5_e.py::
test_no_in_library_fft_product_spells_an_elidable_operand` walks the source on
every build and now pins the CENSUS as well as the property
(`{asm.py: 4, carrier.py: 2, fresnel.py: 1, rs.py: 3}`), so a new site fails
that id with the site named rather than leaving a stale comment behind.

So with the ping-pong ON neither
operand is an unreferenced temporary and with it OFF only the LEFT one is, and
left-elided equals named on every build measured.  Four entry points
(`angular_spectrum_propagate`, the same with `bandlimit=False`,
`fresnel_propagate`, `carrier._exact_envelope_tf_step`) x three shapes (256,
512, 1024) x both builds: **byte-identical across the switch, 12 of 12 cells**.

What DOES move is a **caller** spelling with a fresh right temporary,
`_ifft2(_fft2(E) * np.exp(1j*P))`:

| n | Windows numpy 2.4.4 | WSL numpy 2.4.6 |
|---|---|---|
| 128 | identical | identical |
| 256 | identical | rel **3.467e-16**, 103 552 / 131 072 doubles |
| 512 | identical | rel **3.396e-16**, 422 647 / 524 288 |
| 1024 | identical | rel **4.037e-16**, 1 718 264 / 2 097 152 |

So (a) would cost 13-22 % of every ASM propagation to remove a ~1 ULP
difference in a spelling the library never uses, on one platform, driven by an
upstream non-invariance the library cannot fix at its own boundary.  **Decision:
(b).**

### 2.2 What changed

VERIFY-WP-B14 D4's sentence goes verbatim into three places in
`lumenairy/propagators/fft_infra.py`:

> the transform's values are byte-identical either way; the object handed back
> is a live workspace view in one mode and a private copy in the other, which
> NumPy's temporary elision can distinguish

-- the registered knob doc for `fft_double_buffer` (~line 1041),
`set_fft_double_buffer`'s own docstring, and the `_PYFFTW_DOUBLE_BUFFER` module
note, which additionally carries the mechanism, the exposure, the 12-of-12
byte-identity measurement and the cost table above.
`set_fft_plan_max_bytes_per_buffer`'s docstring makes the same claim and now
cross-references the scope.

### 2.3 The upstream half -- prepared, NOT filed

`validation/probe_fft_elision/numpy_elision_reproducer.py` is a ten-line,
lumenairy-free reproducer; `NUMPY_ISSUE_DRAFT.md` beside it is a ready-to-file
draft with title, versions, minimal code and observed-vs-expected.  Two findings
in it are this pass's, not VERIFY-B14's:

1. **The version axis is ruled out.**  numpy **2.4.6** installed into a clean
   Windows venv on this box prints the UNAFFECTED reading, and 2.4.4 and 2.4.6
   on Windows agree with each other.  The discriminator is the wheel / compiler
   (gcc 14.2.1 manylinux vs msvc 19.44 win_amd64), not the release.
   `np.show_config`'s `SIMD Extensions` block is identical on all three
   (baseline X86_V2, found X86_V3), so it is not a dispatch difference that
   `show_config` can see.
2. **The load-bearing row.**  On the affected build the EXPLICIT
   `np.multiply(a, b, out=b)` -- what the right-operand elision is supposed to
   be a shorthand for -- **matches** the named form, while the ELIDED spelling
   does not.  So this is not a documented consequence of in-place complex
   arithmetic; the elided path does something the explicit one does not.

Spelling matrix at n = 512, `e1_spellings_*.json`:

| spelling | manylinux 2.4.6 | win 2.4.4 | win 2.4.6 |
|---|---|---|---|
| `(A*1.0) * h` (LEFT elidable) | = named | = named | = named |
| `a * np.exp(1j*P)` (RIGHT elidable) | **differs** | = named | = named |
| `np.multiply(a, h2, out=h2)` | = named | = named | = named |
| `np.multiply(a, h3, out=o)` | = named | = named | = named |
| `np.multiply(a2, h, out=a2)` | = named | = named | = named |

### 2.4 The pin

`tests/unit/test_wave5_e_fft_elision.py`, 12 ids.  Unconditional: the transforms
are functions of their inputs (both modes, both shapes); no library entry point
moves across the switch (3 entry points x 2 shapes); the scoped sentence is in
the registered doc and the setter's docstring.  Premise-gated on the running
build's own elision measurement: the object-kind difference and the
caller-visible divergence, with the reading printed in the skip message where
the premise is absent.

Windows **11 passed, 1 skipped** (the gate firing with its reading); WSL
**12 passed**.

---

## 3. E2 -- a durable default-order fail-before for the C8 support bound

### 3.1 The search, and what it settled

VERIFY-WP-B14 F1 measured that the C8 defect class is still reachable at the
shipped `decentred_fit_poly_order` and declined to build a test, because a
27-cell neighbourhood reads ratio 1.00 in 23 of 27 and the 768^2 sweep costs
~14 min.  This pass asked whether ANY geometry parameter the trip is monotone in
exists.  A **432-cell** sweep (`alpha` x `cx` x `z` x `fit_radius_beam_factor`
at n = 256 and 512, 6 354 s) plus four fine scans
(`e2_sweep2_n256_512_*.json`, `e2_cxscan_*`, `e2_alphascan_*`, `e2_nladder_*`,
`e2_ap_ladder_*`):

| axis | reading at the shipped default order |
|---|---|
| `fit_radius_beam_factor` 1.25 / 1.50 / 2.00 / 2.50 | 1.0 / **1793.9** / 1.0 / 1.3 |
| `cx` 1.30 / 1.40 / 1.50 / 1.60 / 1.70 mm | 3.5 / 1.0 / **1793.9** / 1.0 / 170.7 |
| `alpha` 3.45 / 3.50 / 3.55 | 1.0 / **1793.9** / 1.0 |
| `n` 128 / 160 / 192 / 224 / 256 | 1.0 / 1.0 / 8.2 / 1.0 / **1793.9** |
| aperture 6 / 8 / 10 / 12 / 18 mm | field all-zero / all-zero / 1.0 / 1.0 / 1.0 |

**There is no monotone geometry parameter.**  Whether the order-16 fit's
extrapolated inverse folds back into the bright beam is a chaotic function of
which traced samples the ray grid happens to contain.  F1 was right to refuse a
single cell, and it is right for the same reason to refuse `alpha`, `cx`,
`frbf`, the aperture and `n` as ladder axes.

### 3.2 The parameter that IS monotone, and why

The **halo annulus radius**.  The C8 bound zeroes exit pixels with no traced ray
behind them, so the further out the annulus sits the larger the fraction of it
outside the traced footprint and the more completely the bound must empty it.
That is a property of the mechanism, not of the cell.

### 3.3 The cell, and the smallest N

Of the 216 cells swept at n = 256, **47** trip in three or more annuli.  Of
those, exactly **one** also trips at n = 512: `alpha = 3.0, cx = 1.25 mm,
z = 12 mm, fit_radius_beam_factor = 2.0` on the `_GHOST` optic re-sampled over
the same 19.2 mm extent.  So **N = 256 is the smallest sampling at which the
ratio is >= 10x and survives a doubling**; 128 and 192 are inert, and so is 768.

Suppression `on/off` per annulus -- **identical to the printed digits on both
builds** (`e2_radius_nstable_win32_314.json`, `e2_radius_nstable_linux_312.json`):

| r > | 2.0 w | 2.5 w | 3.0 w | 3.5 w | 4.0 w | 4.5 w | 5.0 w |
|---|---|---|---|---|---|---|---|
| n = 256 | 0.596 | 0.596 | 7.74e-2 | 8.95e-3 | 8.95e-3 | 2.37e-5 | 1.79e-7 |
| n = 512 | 0.216 | 2.28e-2 | 2.04e-3 | 1.66e-4 | 1.30e-5 | 9.95e-7 | 7.52e-8 |

Five of seven rungs clear 10x at 256 and **six of seven** at 512; all seven are
monotone at both samplings.  Beyond three beam widths the ratio is **12.9x at
256 and 490.9x at 512**.

(Corrected 2026-09-19, VERIFY-WAVE5-E D5.  This sentence used to say "seven of
seven at 512" and contradicted the table printed immediately above it: the
2.0 w rung's suppression is 0.216, i.e. **4.63x**, below the file's own
`_TRIP = 10.0`.  Re-measured on both builds through the shipped element call,
`e2_d4d5d7_win32_314.json` / `e2_d4d5d7_linux_312.json`: 5 of 7 at n = 256 and
6 of 7 at n = 512, with the 2.0 w rung reading 4.6339x (Win) / 4.6339x (WSL).
Nothing failed on it -- the pin requires `_REQUIRED_TRIPS = 3` -- and the pin
now reads that requirement from ONE constant used by the stimulus gate and by
both fail-before assertions, so the premise and the claim cannot drift apart.)

### 3.4 The bar, two-sided

* **Below**: where the bound is inert the annulus MAXIMUM is an untouched
  pixel, so every one of the 385 inert cells of the 432-cell sweep reads ratio
  **exactly 1.0** -- not 1.0000001.  The bar of 10x is a decade above an exact
  identity.  (Corrected 2026-09-19, VERIFY-WAVE5-E D7.  This used to say "the
  two fields are BIT-identical", which is not what an inert cell reads:
  re-measured on the verifier's inert cell -- `alpha = 3.0, cx = 1.40 mm,
  z = 12 mm, frbf = 2.0, n = 256` -- all seven annuli read exactly 1.0 on both
  builds while the two fields differ, total power moving by **-3.69e-16**
  relative on Windows and **-3.34e-14** on WSL.  The conclusion survives; the
  stated mechanism does not, and the weaker statement about the maximum is the
  one the bar actually rests on.)
* **Above**: the weakest tripping rung reads 12.9x (n = 256) / 490.9x (n = 512).

### 3.4a The trip counter (corrected 2026-09-19, VERIFY-WAVE5-E D4)

`_evaluate` counted a rung as tripping with
`r[3] > 0.0 and (1.0 / r[3]) >= _TRIP`, so a rung the bound empties
**completely** -- suppression exactly 0.0, the strongest evidence the ladder
can produce -- was counted as NOT tripping, while `_fmt` and `_require` both
read the same 0.0 as `inf`.  The predicate is now `r[3] <= 1.0 / _TRIP`, which
is the one `_fmt`/`_require` already use.  Measured on a 25-rung sweep to 8 w
at the selected cell, identical on both builds: the 6.50 w and 6.75 w rungs
read `on == 0.0` exactly, and the trip count goes **14 -> 16**.  The pin's own
seven rungs are unaffected (none of them is emptied completely), so the four
ids read 4 passed before and after; what the change removes is a live stimulus
being turned into a skip.

### 3.5 The pin

`tests/unit/test_wave5_e_c8_default_order.py`, 4 ids:

1. the ladder trips in >= 3 annuli at the SHIPPED order (no
   `decentred_fit_poly_order` is passed anywhere in the file);
2. the suppression is non-increasing in the radius;
3. the bound is power-subtractive AND leaves every pixel inside one beam width
   untouched (so a bound that simply zeroed everything outside a radius would
   fail);
4. the stimulus survives the 256 -> 512 doubling.

**The premise gate cannot hide a dead bound.**  Before skipping, `_require`
re-runs the published order-10 `_GHOST` control, which must still read >= 10x --
it reads **51.5x** here (4.594672922387141e-02 -> 8.913411901995892e-04),
bit-reproducing VERIFY-WP-B14's three `git archive` trees.  If the control also
fails, that is a **hard failure**: the bound is dead, not the stimulus moved.

Windows **4 passed in 12.0 s**; WSL **4 passed in 19.3 s** -- inside the 60 s
budget with an order of magnitude to spare, against the ~14 min the 768^2 sweep
cost.

---

## 4. E3 -- the Maslov joint-scale pin's ULP bar

### 4.1 What was wrong with it

`test_audit2609_a4_verify_maslov_asymptotic.py::
test_s10_vector_normalisation_is_one_joint_scale_for_the_pair` asserted
`<= 4 * np.spacing(r0)` with a docstring origin of "0 ULP for `'power'`, 1 ULP
for `'peak'`".  A clean process reads **3 and 1** today -- bit-identical archive
to archive, so not a package regression, but 1.33x of headroom over its own
envelope: the S4 floor-bar shape.

Its stated derivation -- "`(a s)^2 / (b s)^2` is exact up to the rounding of the
two products" -- accounts for one of two rounding sources.  Each power is a SUM
of `N*N = 9216` non-negative terms, and rescaling every term by an exact common
factor does not make the two reductions round the same way.

### 4.2 The derivation

    bar = SAFETY * ( 10u                       products: <= 5 roundings per leg,
                                               two legs, u = eps/2
                   + measured reduction spread )

where the reduction spread is **measured at run time**: the same non-negative
terms summed through five different summation TREES (pairwise C order, pairwise
per axis both ways, the reversed flat array, the transpose) against
`math.fsum`.  All terms are non-negative, so the sum is perfectly conditioned
(`sum|t| / |sum t| = 1`) and the spread IS the rounding, not a cancellation
artefact.  `SAFETY = 4` is documented, not tuned: the derived quantity itself
spans 2.1x across the ladder below.

### 4.3 The 16-arm ladder

Both builds x `OPENBLAS_CORETYPE` in {HASWELL, NEHALEM, KATMAI, SANDYBRIDGE} x
1 and 4 threads; every arm confirmed by `threadpoolctl` to be running the
architecture it was asked for (`e3_maslov_*.json`).

| | envelope over 16 arms |
|---|---|
| `'power'` reading | **0 .. 3 ULP** |
| `'peak'` reading | **0 .. 3 ULP** |
| products term (derived, 10u) | 8.889 ULP, every arm |
| reduction term (measured) | **1.493 .. 3.173 ULP** |
| **the bar** | **41.53 .. 48.25 ULP** |
| an INDEPENDENT per-leg scale (the pre-fix defect) | **5.158e+08 .. 5.168e+08 ULP** |

The bar clears the widest reading by at least **15.3x** on every arm (two of
the sixteen read 0 ULP in both modes, where the margin is unbounded) and sits
at worst **1.07e+07x** below the real signal.  On SANDYBRIDGE/4 the reading
(3 ULP) exceeds that arm's own raw reduction term (2.623 ULP), which is why the
products term and the safety factor are both needed and neither is decoration.

(The products term was **6u** and the bar **27.31 .. 34.03 ULP** over the same
16 arms until 2026-09-19; see section 11.9 -- VERIFY-WAVE5-E D8 -- for the
re-derivation.  No arm changes verdict either way.)

### 4.4 The other side, asserted rather than quoted

The pre-fix behaviour -- each scalar leg normalised on its own -- is now
reconstructed in the test from this build's own `'none'` output and must move
the ratio by >= 1e3x the bar.

### 4.5 Two stale readings re-recorded

| docstring said | measured 2026-09-15 |
|---|---|
| ratio `1.777777663332023` under all three modes, "bit-identical, 0 ULP" | `1.7777776632250892`, and the three modes are 0-3 ULP apart, not bit-identical |
| departure from the input ratio `-6.4376e-08` | `-6.4436e-08` |

`'power'` restoring the post-Fresnel pair power to 1.0 and `'none'` reading
28.46x it both reproduce (28.461185742106395).

---

## 5. E4 -- two published claims corrected against their own re-measurement

### 5.1 D2 -- the glass-validity red is not an inter-file order dependence

Re-measured on the PRE tree **`96cb2096`** (the WP-B14 base), Windows
py3.14.6 / numpy 2.4.4, `-p no:randomly`:

| selection | result |
|---|---|
| `test_v4_16_0_agent_d_validity_ranges.py` **alone** | **1 failed, 7 passed** |
| meshgrid file FIRST, then it | 1 failed, 23 passed |
| it FIRST, then the meshgrid file | 1 failed, 23 passed |
| its OWN first test + the failing id | **1 failed, 1 passed** |
| the failing id in isolation | 1 passed |

The poisoner is the file's own first test,
`test_validity_warning_emitted_outside_range`, which memoises the same
`('N-BK7', 200e-9)` pair.  "Only when it ran AFTER
`tests/unit/test_audit_w4_glass_registry_meshgrid.py`" is replaced by the
measured reproduction in all three places that carried it: the fixture's
docstring, `CHANGELOG.md`'s 5.47.0 entry and
`WP-B14_KNOWN_REDS_REPORT.md` section 3.1.  The fix itself is unchanged and is
STRONGER than the claim was, because it covers intra-file self-poisoning too.

### 5.2 D3 -- the `'measured'` accounting bounds the budget only above one column

Re-measured on this tree before being written down, N = 256, 1024 beamlets,
`'measured'` mode (`probe_e4_gbd_budget_scope.py`, which imports VERIFY-B14's own
`_run` / `_bundle` / `_effective_chunk` rather than re-implementing the
measurement kernel):

| budget | chunk | Windows peak / ratio | WSL peak / ratio |
|---|---|---|---|
| 512 MB | 61 | 432.4 MB / **0.84x** | 389.6 MB / **0.76x** |
| 16 MB | 1 | 9.58 MB / **0.60x** | 8.92 MB / **0.56x** |
| 4 MB | 1 | 9.58 MB / **2.39x** | 8.92 MB / **2.23x** |
| 1 MB | 1 | 9.58 MB / **9.58x** | 8.92 MB / **8.92x** |

**The two figures VERIFY-B14 D3 asked to be written down -- 2.39x at 4 MB and
9.58x at 1 MB -- reproduce exactly on Windows.**  WSL reads 2.23x / 8.92x; the
~7 % gap is `tracemalloc`'s own bookkeeping, so the note records both builds and
states the build-free claim (both under 1x at 512 and 16 MB, both over 2x at
4 MB).  The derived one-column floor,
`Ny*Nx*(48 + _DENSE_CELL_BYTES_MEASURED)` = **11.53 MB** at N = 256, is in the
note too, with the pointer to the existing two-sided pin
`test_verify_b14_known_reds.py::test_the_measured_accounting_bounds_the_budget_only_above_one_column`.

`gbd.py`'s token stream is unchanged by this edit (comment-only); the a17
fingerprint gate reads OK on it, which is the check that says so.

---

## 6. E5 -- the exit-vertex projection and the FGA image leg

### 6.1 O-1, and the round that had to be undone

The first implementation was VERIFY-WP-B12 O-1's own one-liner,
`np.where(alive, projected, original)`.  **Measured consequence: 7 of the 18
WP-B12 reference-plane pins went red**, every one of them on a single rim ray
(index 120 of 121), reading exactly the last surface's sag (7.226e-06 m of OPL,
8.444e-07 m of height).

The diagnosis is **O-4, not the change**.  The finite-difference backend's
`alive` is `base_alive & companion_alive`: it drops a ray whose 9-ray FD
companion bundle vignettes even though the BASE ray landed.  Such a ray's
Jacobian is meaningless, but its state is not -- it did reach the last surface
and it did reach the vertex plane, and `TraceResult.at_exit_vertex()` projects
it.  Freezing it put the two vertex-plane operators out of step **in the other
direction**, which is the very thing O-1 asks to close, and would have required
rewriting eight existing pins' masks.

`_project_to_exit_vertex_plane` therefore takes `reached_surface` -- the mask of
rays that reached the last surface.  `ray_transfer_jacobian` passes the base
ray's own alive at both its call sites; the analytic backends trace one ray,
have no companions, and let it default to `transfer.alive`.  Measured on the
WP-B12 biconvex fan: 1 ray of 121 is companion-dead-but-base-alive, and its
projected state agrees with `at_exit_vertex` to **4.3e-19 m** where the frozen
state would have disagreed by 7.2e-06 m.

### 6.2 The measurement, PRE vs POST, both builds

`probe_e5_dead_ray_freeze.py` over the eight VERIFY-WP-B12 fixture classes
(`asph`, `bicon`, `menisc`, `doublet`, `oblique`, `mirror`,
`flat_planoconvex`, `flat_pair`) x four backend arms (FD bundle tracer, analytic
entry point, its JAX path, and the projection primitive itself with `jax.numpy`
arrays at x64).  Freeze set = the production ray-bundle trace's own
`image_rays.alive` intersected with the backend's dead set.  30 cells; the
Windows and WSL summaries are identical:

| | PRE (`ede07f30`) | POST |
|---|---|---|
| rows that REACHED the surface | -- | **byte-identical PRE vs POST, 30 of 30 cells** |
| missed rows unfrozen | **17 of 23 scored cells** | **0 of 23** |
| missed-row `opd` drift | 1.367e-05 .. **1.871e-04 m** | **0.0** |
| missed-row Jacobian drift | 1.367e-05 .. **1.528e+300** | **0.0** |
| JAX twin state vs NumPy | -- | **bit-identical, 8 of 8 fixtures; Jacobians 0.00 ULP** |

Two findings worth recording beyond the fix:

* the **1.528e+300** on `asph`/analytic is the analytic backend's missed rows
  extrapolating.  It is finite, so the projection's `nan_to_num` never caught
  it;
* **7 of the 30 cells are vacuous** because the JAX path of
  `ray_transfer_jacobian_analytic` reports every ray alive whatever the aperture
  (0 dead of 201 where the FD tracer vignettes 134).  It marks nothing for the
  freeze to act on.  Recorded, not scored -- and not in this item's scope.

### 6.3 O-3 -- the immersed-exit guard

Each FGA transport asks the projection for `reference='exit_vertex'` (which
resolves `n_exit` and weights its sag term with it) and then adds the image leg
by hand as `opd += z_image * sqrt(1 + ux^2 + uy^2)`, with no index.  In a medium
of index `n` that omits `|n-1| * z_image * sec` of optical path, at least
`|n-1| * |z_image| / lambda` waves.

`_require_non_immersed_exit` refuses it with a named `NotImplementedError` at
all four sites -- `_fga_through_lens`, `_fga_coarse`,
`_fga_vector_through_lens`, `_caustic_zone`.  The tolerance is **derived, not
chosen**:

    |n - 1| > waves_budget * lambda / max(|z_image|, lambda),   waves_budget = 1e-3

so a longer leg tightens it in proportion, and a zero-length leg
(`_caustic_zone`, or `output_plane_distance = 0`) floors it at the budget
itself -- **~500x** below the weakest immersion medium.
`get_glass_index('air', lambda)` returns **exactly 1.0** on this registry at
every wavelength measured (633, 780, 1030, 1060, 1310, 1550 nm), so the air
control sits exactly at the tolerance's origin.

**THE FLOOR IS NOT THE TOLERANCE** (corrected 2026-09-19, VERIFY-WAVE5-E D1).
This section previously read "3.6x the air-vs-vacuum index difference at STP
(2.77e-4, so a caller who registers a real air index is not refused)".  That
margin exists ONLY at a zero-length leg.  Bisecting the guard itself on both
builds gives the boundary exactly `waves * lambda / max(|z_image|, lambda)`:

| `z_image` | boundary on `\|n - 1\|` | STP air (n - 1 = 2.77e-4) |
|---|---|---|
| 0 | 1.00e-03 | not refused (3.6x margin) |
| 1.55e-6 m (= lambda) | 1.00e-03 | not refused |
| 1e-5 m | 1.55e-04 | **refused** |
| 3.5e-4 m (the fixture's own leg) | 4.43e-06 | **refused, by 63x** |
| 1e-2 m | 1.55e-07 | **refused, by 1800x** |

So at **every image distance the FGA actually runs**, a real air index is
refused.  The BEHAVIOUR is right under the one-milliwave budget -- the
un-indexed leg would be wrong by `|n - 1| * z_image` = 97 nm, about 0.1 wave,
at 0.35 mm -- so the guard is not changed; what was wrong was the sentence, and
it was wrong in the direction that matters, because it told a reader the guard
would not refuse a real air index.  The guard is a **near-unity-exit-index**
guard, not only an immersion guard, and its message now says so.  Nothing
served today is affected (air resolves to exactly 1.0).  The way out for a
caller who really does register a purge gas or an index-matching fluid at
n = 1.0001 is the open follow-up in section 11.1, "carry `n_exit` in the FGA
image leg", not a looser tolerance.

An exit medium the registry cannot resolve is NOT this guard's diagnostic and
returns `None`: with a FLAT last surface the projection short-circuits and never
resolves the index at all, so raising there would refuse a prescription served
correctly today.

### 6.4 The pin

`tests/unit/test_wave5_e_exit_vertex_dead_rays.py`, 29 ids, 62 s (Windows) /
inside the WSL run below.  (**33 ids** after the pre-merge follow-ups of
section 11 -- four per-call-site arms added for D9.)  Every claim two-sided:

* the freeze is bit-exact on missed rows AND the rows that reached still move
  (a freeze that froze everything would pass the first arm and be a disabled
  projection);
* the premise (that the fan really vignettes) is **asserted, never skipped**;
* the frozen set is the same set `at_exit_vertex()` freezes, and a
  companion-dead-but-base-alive ray is asserted POSITIVELY to still be projected
  and still agree with `at_exit_vertex`;
* the JAX twin of the primitive is bit-identical to the NumPy one on all six
  surface classes;
* the guard fires at each of the four ENTRY PATHS on a synthetic
  immersed-exit prescription AND does not fire on the air-terminated control;
  and (added 2026-09-19, VERIFY-WAVE5-E D9) at each of the four guard CALL
  SITES, reached through the entry point that hits THAT site's guard first --
  `_fga_coarse` is entered directly, since every in-library caller reaches it
  from inside `_fga_through_lens`, whose guard runs first;
* the tolerance's derivation is pinned rather than its number, and (corrected
  2026-09-19, VERIFY-WAVE5-E D3) it is pinned out of the LIBRARY's one
  definition, `fga._immersed_exit_tolerance`: the guard is bisected through
  that helper at six image distances and required to refuse at 1.01x its
  return and serve at 0.99x, and the physics the number encodes -- the
  boundary index error costs exactly the one-milliwave budget over that leg --
  is asserted on the helper's own value, so a formula that dropped the leg's
  length fails instead of moving both sides together.  The zero-leg floor and
  the real-leg decision (STP air served at a zero leg, refused over 0.35 mm)
  are both asserted.

---

## 7. E6 -- VERIFY-WP-B14 7a, for the campaign ledger

**CONFIRMED by the verifier; nothing to build, and nothing was built.**  The
second BLAS classification pin -- `test_w4_t1_...` in the w3 file, red on
SANDYBRIDGE at `rel = 1.2227e-08` against a `< 1e-08` bar on the PRE tree -- is
closed: VERIFY-WP-B14 measured POST green at SANDYBRIDGE t1 and t4 and at KATMAI
t1 and t4, with the whole file reading 181 passed at SANDYBRIDGE t4.  This pass
neither re-measured it nor changed anything it touches; it is recorded here so
the ledger can close the item.

---

## 8. Runs

All with the pinning of section 0; every tail grepped.

| run | Windows py3.14.6 | WSL py3.12.3 |
|---|---|---|
| `test_wave5_e_fft_elision.py` (new, E1) | **11 passed, 1 skipped** (11.6 s) | **12 passed** (6.4 s) |
| `test_wave5_e_c8_default_order.py` (new, E2) | **4 passed** (12.0 s) | **4 passed** (19.3 s) |
| `test_wave5_e_exit_vertex_dead_rays.py` (new, E5) | **29 passed** (62.0 s) | (in the 47 and 107 below) |
| the three new files together, `--store-durations` | **44 passed, 1 skipped** (13.7 s) | -- |
| `test_audit2609_a4_verify_maslov_asymptotic.py` + `test_v4_16_0_agent_d_validity_ranges.py` (E3, E4 touched) | **62 passed** (37.0 s) | -- |
| the three new files + the two touched ones | -- | **107 passed** (69.8 s) |
| `test_audit2609_b12_fga_reference_plane.py` + `test_verify_b12_fga_reference_plane.py` | **18 passed** (78.8 s) | -- |
| the two above + `test_wave5_e_exit_vertex_dead_rays.py` | -- | **47 passed** (104.2 s) |
| `test_gbd*.py` (12 files) + `test_verify_b14_known_reds.py` | **165 passed, 0 failed** (1142.9 s) | -- |
| census / walker / dispatcher-pin / public-API / doc-consistency / history-relocation sweep + `test_audit_except_budget.py` (28 files) | **1375 passed, 11 skipped, 0 failed** (198.8 s), re-run AFTER every CHANGELOG edit | -- |
| E3's 16-arm BLAS ladder (`OPENBLAS_CORETYPE` x threads) | 8 arms | 8 arms |
| `scripts/record_history_fingerprints.py --check` | **green, 0 drift** | -- |
| `ruff check lumenairy/ tests/ validation/probe_wave5_e/ validation/probe_fft_elision/ scripts/` (WSL) | -- | **All checks passed!** |

The PRE trees were measured with the same pinning: `96cb2096` for E4's D2
reproduction and `ede07f30` for E5's PRE arm, both builds.

### 8.1 `test_gbd*.py` + `test_verify_b14_known_reds.py`

The twelve `test_gbd*.py` files plus `test_verify_b14_known_reds.py`, Windows:
**165 passed, 0 failed, 0 skipped** (1142.9 s).  That selection is what covers
E4's `gbd.py` edit -- including the two-sided pin the new scope sentence cites,
`test_the_measured_accounting_bounds_the_budget_only_above_one_column` -- and
`test_wave5_gbd_dense_mem_budget.py`, whose one-cell claim the scope sentence
qualifies.

---

## 9. Commits

| item | SHA | subject |
|---|---|---|
| E4 | `ddcf6576` | `docs(gbd, tests): WAVE5-E -- the glass-validity red reproduces ALONE, and the 'measured' budget bounds only above one beamlet column (VERIFY-WP-B14 D2, D3)` |
| E1 | `a88eea2e` | `docs(fft_infra, tests): WAVE5-E -- the ping-pong's byte-identity claim is scoped to the transform's values; the dispatchers keep the zero-copy return (VERIFY-WP-B14 D4)` |
| E3 | `6ccafc0f` | `fix(tests): WAVE5-E -- the Maslov joint-scale pin's ULP bar is derived from the summation's own rounding, two-sided` |
| E5 | `9a83042c` | `fix(raytrace, fga): WAVE5-E -- the exit-vertex projection freezes dead rays, and the FGA image leg refuses an immersed exit (VERIFY-WP-B12 O-1, O-3)` |
| E5 (round 2) | `17957d09` | `fix(raytrace): WAVE5-E round 2 -- the exit-vertex freeze keys on REACHED THE SURFACE, not on the FD backend's conflated alive` |
| E2 | `d3fd1852` | `test(niche-c8): WAVE5-E -- a durable default-order fail-before for the inverse-support bound, as a ladder over annuli (VERIFY-WP-B14 F1)` |

History fingerprints re-recorded in the commit that moved the module:
`docs/history/fft_infra.md` (E1) and `docs/history/lumenairy.raytrace.differential.md`
(E5, twice -- once per round).  `gbd.py` and `fga.py` needed none: `gbd.py`'s
edit is comment-only and `--check` reads OK on it, and `fga.py` has no history
document.

Nothing was pushed and nothing was filed externally.

---

## 10. What I could not measure

1. **Whether the NumPy elision non-invariance is a compiler or a libm
   difference.**  The version axis is ruled out (2.4.6 on Windows is
   unaffected), and `show_config`'s SIMD block is identical on all three
   interpreters, but separating gcc-vs-msvc codegen from the platform's own
   `cexp`/`cmul` would need a build of the same wheel on both toolchains.  The
   draft issue says so rather than guessing.
2. **Whether CI's runner mix reproduces E2's cell.**  The stimulus is identical
   to the printed digits on this box's two builds, and the sweep's 385 inert
   cells all read exactly 1.0, but the EPYC 9V74 / 7763 runners with py3.10 /
   numpy 2.2.6 have not run it.  The premise gate is what protects that arm: a
   runner where the cell is inert skips with every candidate's reading and only
   hard-fails if the order-10 control is also dead.
3. **The 768^2 F1 cells at the shipped default.**  I re-measured the 256^2 and
   512^2 grids exhaustively; the published 301x and 3164x cells were not
   re-run at 768^2, because the search moved to the cheaper geometry the item
   asked for.  The order-10 768^2 `_GHOST` control WAS re-run and reproduces
   51.5x bit-for-bit.
4. **The JAX analytic backend's missing vignetting.**  Measured (0 dead of 201
   where the FD tracer vignettes 134) and recorded in section 6.2, but not
   investigated or fixed -- it is outside both O-1 and O-3, and it makes 7 of
   E5's 30 probe cells vacuous.
5. **A full CI matrix.**  The gates here are the library-touching ones and the
   sweep; a matrix run on the merge is the authority on the CI classes.

---

## 11. Pre-merge follow-ups (VERIFY-WAVE5-E)

The independent re-verification
(`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/VERIFY_WAVE5_E.md`,
`b082ecd6` on `verify/wave5-item-e`) returned SHIP with two defects to fix
before the merge (D1, D3), one to FILE and not fix (D2, pre-existing), and six
P3 corrections (D4-D9).  This section records what each edit was and how it was
PROVED, on both builds.  Branch `fix/wave5-item-e-followups`, worktree
`C:/tmp/lum_e2`; pinning as section 0 (`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=
MKL_NUM_THREADS=1` on the command line, `PYTHONPATH` pinned to this tree,
pytest `--capture=sys -p no:randomly`, every tail grepped).

### 11.1 Open items carried forward (HANDOFF)

**(a) Carry `n_exit` in the FGA image leg.**  The O-3 guard REFUSES what the
image leg cannot represent; the leg itself is still index-free
(`opd += z_image * sqrt(1 + ux^2 + uy^2)`), while the projection that produced
`dt.opd` resolved `n_exit` and weighted its sag term with it.  Measured
2026-09-19 by bisecting the guard on both builds (VERIFY-WAVE5-E sec. 6.3, and
now `test_the_guard_tolerance_is_the_wavefront_it_protects` /
`test_verify_wave5_e.py::test_the_immersed_exit_guards_boundary_is_the_guards_own_derivation`):
the refusal boundary is `waves * lambda / max(|z_image|, lambda)` exactly, so

| `z_image` | boundary on `\|n - 1\|` | STP air (2.77e-4) is refused by |
|---|---|---|
| 0 / 1.55e-6 m | 1.00e-03 | not refused (3.6x margin) |
| 1e-5 m | 1.55e-04 | 1.8x |
| 3.5e-4 m | 4.43e-06 | **63x** |
| 1e-2 m | 1.55e-07 | **1800x** |

i.e. a caller who registers a real air index is refused at every image distance
the FGA actually runs.  The refusal is correct under the one-milliwave budget
(the omitted OPL is `|n - 1| * z_image` = 97 nm ~ 0.1 wave at 0.35 mm), so the
way out is not a looser tolerance: it is to carry the index, after which the
guard is needed only where the projection itself cannot resolve an exit medium.
Scope: `_fga_through_lens`, `_fga_coarse`, `_fga_vector_through_lens` (three
image legs) and the `at_exit_vertex`-referenced `opd` they add to; no fixture
in the suite exercises it, because `get_glass_index('air', lambda)` is exactly
1.0 on this registry, so the work needs an immersed-exit ORACLE (a
prescription terminated in the medium as an explicit last element) before it
can be graded.

**(b) The JAX analytic backend reports no vignetting.**  See 11.10 (D2) --
pre-existing, FILED and pinned, deliberately not fixed here.

### 11.2 D1 (P2) -- the O-3 guard's boundary, restated in the four places it appears

**What was wrong.**  "The tolerance floors at the budget itself, 1e-3 -- which
is 3.6x the air-vs-vacuum index difference at STP (2.77e-4, so a caller who
registers a real air index is NOT refused)."  True at a zero-length leg only.

**Re-measured here** by bisecting the shipped guard, both builds, identical:
the boundary is `waves * lambda / max(|z_image|, lambda)` to every digit, so
STP air is refused from `z_image = 1e-5 m` upward -- by 63x at the fixture's
own 0.35 mm leg and by 1800x at 10 mm.  The un-indexed leg over 0.35 mm of real
air is wrong by `|n - 1| * z_image` = 97 nm, about 0.1 wave, against a
one-milliwave budget, so the refusal is what the budget says.

**What changed.**  The guard is NOT changed.  The sentence is restated in the
four places it appears -- `_require_non_immersed_exit`'s docstring, this
report's section 6.3, `test_the_guard_tolerance_is_the_wavefront_it_protects`'s
docstring and the `CHANGELOG.md` entry -- to say what the boundary IS (a
near-unity-exit-index guard, not only an immersion guard) and what it implies
(real air is refused at every image distance the FGA actually runs; the way out
is the follow-up in 11.1(a), not a looser tolerance).  The raised message gains
"(or, at a long image leg, merely has an exit index far enough from 1 to spend
the wavefront budget over that leg)"; the `IMMERSED` token is untouched, so the
eight `pytest.raises(..., match='IMMERSED')` arms are unaffected.

**Proof.**  The claim is now ASSERTED, not only written down: the pin's
section 3 requires a registered STP air index (1.000277) to be SERVED at a
zero-length leg and REFUSED over 0.35 mm, through the real guard.  Commit
`8e2886a5`.

### 11.3 D3 (P2) -- one definition of the tolerance, and a pin that asks it

**What was wrong.**  The pin defined `tol(z)` inside the test and asserted four
properties of its own copy.  Mutation `e5_tol_drop_z` (the guard's tolerance
rewritten from `waves*lam/max(|z|,lam)` to `waves*lam/lam`, deleting the image
leg from the derivation) left **all 29 ids green on both builds**.

**What changed.**  `fga._immersed_exit_tolerance(wavelength, z_image,
waves=_FGA_IMAGE_LEG_WAVE_BUDGET)` is now the one definition; all four guard
sites reach it through `_require_non_immersed_exit`.  The pin reads the number
out of it, bisects the SHIPPED guard at six image distances (0, lambda, 1e-5,
3.5e-4, 1e-3, 1e-2 m) and requires the boundary to be that return to 1e-9
relative, is two-sided about it at every leg (1.01x refuses, 0.99x is served),
and asserts the PHYSICS the number encodes -- the boundary index error costs
exactly the wave budget over that leg -- on the helper's own value.  That last
assertion is the one a formula without the leg's length cannot satisfy, which
is what makes the pin see the change instead of moving with it.

**Proof (mutation `e5_tol_drop_z`, both builds, on a copy of this tree):**

| | before | after |
|---|---|---|
| Windows py3.14.6 / numpy 2.4.4 | 29 passed | **1 failed, 28 passed** |
| WSL py3.12.3 / numpy 2.4.6 | 29 passed | **1 failed, 28 passed** |

the failing id being
`test_the_guard_tolerance_is_the_wavefront_it_protects` on both, with the
message naming the measured cost (0.009708737864077669 waves against a budget
of 0.001).  No behaviour change: the tolerance's value is identical, only its
location.  Commit `7e267845`.

### 11.4 D9 (P3) -- each guard CALL SITE pinned on its own

**What was wrong.**  `test_every_fga_site_refuses_an_immersed_exit` is
parametrized over four ENTRY PATHS; its `coarse` arm is
`apply_real_lens_fga(coarse_stride=3)`, which reaches `_fga_coarse` only from
inside `_fga_through_lens`, whose guard has already run.  Deleting the guard
from `_fga_coarse` left the whole file green.

**What changed.**  `test_every_guard_call_site_is_independently_reachable`,
four arms, each entering through the entry point that reaches THAT site's guard
first: `apply_real_lens_fga`, `_fga._fga_coarse` called DIRECTLY,
`apply_real_lens_fga_vector`, `_fga._caustic_zone`.  Each arm also requires the
message to name ITS site, so an arm satisfied by another site's guard upstream
cannot pass unnoticed.  `_fga_coarse` is entered with the smallest arguments
that carry its pre-guard block (16x16 launch amplitude, unit position stride,
2-cell coarse stride, one momentum).  The entry-path id keeps its four arms --
it is the caller-visible statement -- and its docstring now carries the scope.

**Proof (four deletion mutations, Windows and WSL identical):**

| mutation | before | after |
|---|---|---|
| identity | 29 passed | **33 passed** |
| guard deleted from `_fga_coarse` | 29 passed (invisible) | **1 failed, 32 passed** |
| guard deleted from `_fga_through_lens` | 1 failed, 28 passed | **2 failed, 31 passed** |
| guard deleted from `_fga_vector_through_lens` | not measured | **2 failed, 31 passed** |
| guard deleted from `_caustic_zone` | not measured | **2 failed, 31 passed** |

Each site's deletion reddens its own id; the three that also have an
entry-path arm redden that too.  Two new arms
(`e5_guard_off_at_vector`, `e5_guard_off_at_caustic`) were added to
`validation/probe_verify_wave5_e/mutate.py` so the matrix covers all four.
Commit `b9b24711`.

### 11.5 D4 (P3) -- a totally suppressed rung now counts as tripped

`_evaluate`'s `r[3] > 0.0 and (1.0 / r[3]) >= _TRIP` discarded a rung the bound
empties completely (suppression exactly 0.0) -- the strongest evidence the
ladder can produce -- while `_fmt` and `_require` read the same 0.0 as `inf`.
The predicate is now `r[3] <= 1.0 / _TRIP`.

**Re-measured** on a 25-rung sweep to 8 w at the selected cell
(`e2_d4d5d7_win32_314.json`, `e2_d4d5d7_linux_312.json`), identical on both
builds: the **6.50 w and 6.75 w** rungs read `on == 0.0` exactly, and the trip
count over the wide ladder goes **14 -> 16**.  The pin's own seven rungs
contain no emptied rung, so the four ids read 4 passed before and after
(Windows 18.6 s, WSL 17.8 s); what the change removes is a live stimulus being
turned into a skip.  Commit `00e31fcf`.

### 11.6 D5 (P3) -- six of seven, and one constant for the requirement

"Seven of seven rungs clear 10x at 512" is **six of seven**: the 2.0 w rung's
suppression is 0.216, i.e. **4.63x**, below the file's own `_TRIP = 10.0`, and
the sentence contradicted the table printed three lines above it.

**Re-measured through the shipped element call, both builds:** 5 of 7 at
n = 256 and **6 of 7** at n = 512, the 2.0 w rung reading **4.63393x**
(Windows) and **4.63394x** (WSL); all seven rungs monotone at both samplings.
Restated in the file header, in the doubling test's docstring, in section 3.3
above and in the CHANGELOG.

The premise now counts what the pin requires: `_REQUIRED_TRIPS = 3` is read by
the stimulus gate (`if len(trips) >= _REQUIRED_TRIPS`) and by both fail-before
assertions, so the gate's threshold and the claim's threshold are one number.
Commit `00e31fcf`.

### 11.7 D6 (P3) -- all ten product sites, and a census that keeps the list honest

The `_PYFFTW_DOUBLE_BUFFER` note named six sites and said "every".  The AST walk
finds **ten**, none elidable; the four omitted are `asm.py:1147`
(`_fft2_nd(...) * H[None, :, :]` -- a basic-slice VIEW, `owndata` False, so
`temp_elide` cannot claim it either) and `rs.py:936/939/942` (`E_fft * H` where
`E_fft = _fft2(...)`), which is the shape a grep for `_fft2(` misses and the
more fragile one.  All ten are now listed in the note, in
`test_wave5_e_fft_elision.py`'s docstring and in section 2.1 above, each
pointing at the structural id rather than restating a list.

`tests/unit/test_verify_wave5_e.py::test_no_in_library_fft_product_spells_an_elidable_operand`
already existed and is NOT duplicated; it now carries the **census** as well as
the property -- `_EXPECTED_PRODUCT_SITES = {asm.py: 4, carrier.py: 2,
fresnel.py: 1, rs.py: 3}`, per file rather than per line so a site moving
inside its module does not fire it, with a message that says to update the
census and the note in the same commit.  This is a property of the SOURCE, so
nothing in it is something a build is entitled to move.

**Proof (both builds, identical):** identity 1 passed;
`e1_drop_a_product_site` (carrier.py:7126 rewritten as
`np.multiply(_fft2(_e), ramp)`, which removes one site from the walk)
**1 FAILED**; `e1_unnamed_right_operand` (asm.py:919 given an unnamed right
operand) **1 FAILED** -- the arm that is build-DEPENDENT when asserted by
measurement (the Windows wheels give right-elided == named) and build-free
here.  Commit `55db416a`.

### 11.8 D7 (P3) -- what an inert cell actually reads

"Where the bound is inert the two fields are BIT-IDENTICAL" is false.
**Re-measured** on the verifier's inert cell (`alpha = 3.0, cx = 1.40 mm,
z = 12 mm, frbf = 2.0, n = 256`): all seven annuli read ratio **exactly 1.0**
on both builds, and the two fields are **not** bit-identical -- total power
moves by **-3.69e-16** relative on Windows and **-3.34e-14** on WSL (max
|Foff - Fon| 1.17e-07 and 1.18e-06 in absolute field units).  The conclusion
survives; the mechanism restated is the true and weaker one: the annulus
MAXIMUM is an untouched pixel, which is why the inert reading is exactly 1.0
and not 1.0000001.  Restated in the `_TRIP` docstring and in section 3.4
above.  Commit `00e31fcf`.

### 11.9 D8 (P3) -- the products term counts five roundings per leg, not three

The bar's PRODUCTS half was derived as "``fl(s*a)`` then ``fl(re^2)``,
``fl(im^2)``, ``fl(+)`` is at most three roundings", i.e. `3u` per leg and `6u`
for the ratio.  Per COMPLEX element the chain is `fl(s*re)`, `fl(s*im)`,
`fl((s*re)^2)`, `fl((s*im)^2)`, `fl(+)` -- **five**: the scale is applied to
both components before either is squared, and the old count included only one
of them.  `product_rel` is now `10.0 * u`, with the chain spelled out.

**Re-measured over the same 16 arms** (both builds x `OPENBLAS_CORETYPE` in
{HASWELL, NEHALEM, KATMAI, SANDYBRIDGE} x 1 and 4 threads, every arm's
architecture confirmed by `threadpoolctl`).  The probe
(`validation/probe_wave5_e/probe_e3_d8_products_term.py`, per-arm JSON
`e3_d8_*.json`) reports BOTH bars on the SAME arm, so the change is measured
rather than asserted, and it imports the pin's own
`_power_sum_rounding_spread` so the reduction term has one definition:

| | envelope over the 16 arms |
|---|---|
| `'power'` / `'peak'` readings | 0 .. 3 ULP |
| reduction term (measured) | 1.4932 .. 3.1730 ULP |
| products term, 6u (what was written) | 5.3333 ULP |
| products term, 10u (corrected) | 8.8889 ULP |
| the bar, 6u | 27.3061 .. 34.0255 ULP |
| **the bar, 10u** | **41.5283 .. 48.2477 ULP** |
| independent per-leg scale (the pre-fix defect) | 5.1577e+08 .. 5.1683e+08 ULP |

The 10u bar is above the widest reading on **every one of the 16 arms**, by at
least **15.349x** on the 14 arms with a nonzero reading (two arms read 0 ULP in
both modes, where the margin is unbounded), and sits at worst **1.069e+07x**
below the independent-per-leg signal.  **Not one arm changes verdict.**  Both
of the verifier's corrected readings also reproduce here: `r_none` on
Windows/HASWELL/t1 is 1.7777776632250892 exactly, and the 16-arm `dev` envelope
is -6.45515e-08 .. -6.44200e-08, which contains -6.4436e-08 and excludes the
old -6.4376e-08.

`tests/unit/test_audit2609_a4_verify_maslov_asymptotic.py`: **54 passed**
(Windows, 88.6 s) and **54 passed** (WSL, 72.3 s).  Commit `e7c9976d`.

### 11.10 D2 (P2, pre-existing) -- FILED, not fixed

`raytrace.differential._adrt_jax` ends `alive=jnp.ones((n,), dtype=bool)` and
calls `_adrt_step(..., compute_dead=False)`: the JAX path of
`ray_transfer_jacobian_analytic` carries no vignetting, TIR or missed-surface
logic at all.  Measured by the verifier over a six-rung aperture ladder, 201
rays, both builds identical: **0 dead at every rung** against the NumPy
analytic path's 0 / 68 / 104 / 134 / 156 / 174 and the bundle tracer's
identical set, with the two analytic paths agreeing to **1.1e-19 m** of OPL on
the rays NumPy kills -- so it is the MASK alone and the fix is local.

**It is not fixed here, by instruction and on the merits**: it is pre-existing
(it predates item E), it is outside O-1 and O-3, and no in-library consumer
reaches that path -- `fga._pick_ray_transfer` hands it NumPy arrays and
`gbd.py:3568` calls it with `per_surface=True`, which `_adrt_jax` refuses with
its own `NotImplementedError`.  The exposure is the public entry point alone,
for a `jax.grad` / `jax.jit` caller, and the entry point's own docstring
advertises that backend as the differentiable one.

* **Reproducer**: `validation/probe_verify_wave5_e/probe_v_e5_jax_alive.py`.
* **The patch**, with its one caveat: `VERIFY_WAVE5_E.md` D2.  The in-line
  comment at `differential.py:897` claims the dead computation "would trip a
  tracer->ndarray conversion"; under `vmap(jacfwd(...))` it probably does not,
  because every `np.isfinite` / `float()` in that block reads SURFACE
  attributes and not ray state -- but that has NOT been measured, here or by
  the verifier, so the fallback (a separate un-differentiated `jax.vmap` pass
  over the same `_adrt_step` with `compute_dead=True`, costing one extra primal
  walk and no gradient) stands as the plan B.
* **The pin**: `tests/unit/test_verify_wave5_e.py::
  test_the_jax_analytic_backend_reports_no_vignetting_KNOWN_DEFECT`, which
  asserts the current WRONG behaviour together with the correct behaviour of
  the NumPy path and says in its failure message what to replace itself with
  when the defect is fixed.  It is now PREMISE GATED rather than skipped: it
  reads `lumenairy.backend.array.JAX_AVAILABLE` and, when JAX is absent,
  ASSERTS that absence as a fact against `find_spec('jax')` and
  `LUMENAIRY_DISABLE_JAX` instead of calling `pytest.skip`
  (`docs/TESTING_STANDARDS.md` rule 4 -- "never `pytest.skip` on a resource
  check"; two such skips once removed five tests from the gate on exactly the
  runners that mattered).  Verified both ways on Windows: **1 passed** with JAX
  present (15.2 s) and **1 passed, 0 skipped** with `LUMENAIRY_DISABLE_JAX=1`
  (0.25 s).

Also recorded in the `CHANGELOG.md` `[Unreleased]` entry's known-issues
paragraph, with the reproducer path.  Commit `55db416a`.

### 11.11 Runs (the follow-up pass)

Branch `fix/wave5-item-e-followups`, worktree `C:/tmp/lum_e2`, base `b082ecd6`
(the verification commit on `verify/wave5-item-e`).  The mutant tree is a copy,
`C:/tmp/lum_e2_mut`; `git status --porcelain lumenairy/` is empty.  Every
invocation carried `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1` on
the command line, `PYTHONPATH` pinned to the tree under test, pytest
`--capture=sys -p no:randomly`, and every tail grepped for
`passed|failed|error|no tests ran`.  Both builds confirmed their interpreter
and `lumenairy.__file__` before each run (Windows CPython 3.14.6 / numpy 2.4.4;
WSL CPython 3.12.3 / numpy 2.4.6).

| selection | Windows py3.14.6 | WSL py3.12.3 |
|---|---|---|
| the three `test_wave5_e_*.py` + `test_audit2609_a4_verify_maslov_asymptotic.py` + `test_verify_wave5_e.py` | **107 passed, 1 skipped** (186.6 s) | **108 passed** (154.8 s) |
| `test_audit2609_b12_fga_reference_plane.py` + `test_verify_b12_fga_reference_plane.py` + `test_audit2609_a4_fga_s10.py` + `test_fga.py` + `test_fga_h4_h5.py` + `test_fga_prefactor_dedup.py` + `test_niche_r4_fga_dual_vectorize.py` (the brief's two `fga` globs) | **156 passed** (959.9 s) | **156 passed** (1002.1 s) |
| `test_v4_16_0_agent_d_validity_ranges.py` (E4) | **8 passed** (1.0 s) | **8 passed** (0.4 s) |
| `test_wave5_e_c8_default_order.py` alone | **4 passed** (18.6 s) | **4 passed** (17.8 s) |
| `test_audit2609_a4_verify_maslov_asymptotic.py` alone | **54 passed** (88.6 s) | **54 passed** (72.3 s) |
| census / walker / dispatcher-pin / public-API / doc-consistency / history-relocation sweep + `test_audit_except_budget.py` (28 files) | **1375 passed, 11 skipped, 0 failed** (395.4 s) | 2 failed, 1373 passed, 11 skipped (207.7 s) -- **both failures are environmental, see below** |
| the CHANGELOG / doc-consistency / public-API / history walkers re-run AFTER the last CHANGELOG edit (6 files) | **772 passed, 6 skipped** (68.8 s) | -- |
| `test_audit2609_a15a_durations_staleness.py` | **4 passed** (118.3 s) | -- |
| the D9 + D3 mutation matrix (identity + 5 arms) | see 11.3, 11.4 | see 11.3, 11.4 |
| the D6 census mutation matrix (identity + 2 arms) | see 11.7 | see 11.7 |
| E3's 16-arm `OPENBLAS_CORETYPE` x threads ladder | 8 arms | 8 arms |
| `scripts/record_history_fingerprints.py --check` | **OK, every history document matches its module** | -- |
| `ruff check lumenairy/ tests/ validation/probe_verify_wave5_e/ validation/probe_wave5_e/ validation/probe_fft_elision/ scripts/` | -- | **All checks passed!** |
| `git status --porcelain lumenairy/` | **empty** | **empty** |

The one skip on Windows is the pre-existing build-gated arm of
`test_wave5_e_fft_elision.py` (the elision asymmetry is absent on the Windows
wheels, which is why `test_verify_wave5_e.py`'s structural id exists); WSL runs
it, which is why WSL reads one more passed.

**The two WSL sweep failures are the environment, not this tree, and that was
measured rather than assumed.**  Both are

    test_public_api.py::test_installed_metadata_version_matches_source_version
    test_v5_2_3_walker_changelog_content.py::test_v16_synthetic_fabrication_is_caught

and both reproduce, identically and in 0.42 s, when the SAME two ids are run
from WSL against a different worktree at a different commit (`C:/tmp/lum_wave5`
at `3d3ad1df`, which contains none of this work).  The causes:

* WSL cannot resolve a Windows git WORKTREE's gitdir -- `.git` here is a file
  pointing at `D:/.../Lumenairy/.git/worktrees/lum_e2`, and every `git`
  invocation under WSL dies with `fatal: not a git repository:
  /mnt/c/tmp/lum_e2/D:/...`.  The V16 walker therefore returns rc = 2 (git
  plumbing failed) where the test wants rc = 1 (fabrication flagged), so the
  id fails on the plumbing and never reaches its claim.
* `~/lumvenv` has lumenairy **5.11.0** installed while the source tree is
  **5.47.0**, so the installed-metadata id cannot pass in that venv for any
  tree.

Neither touches anything this pass changed, and the same 28 files read
**1375 passed, 11 skipped, 0 failed** on Windows, where git and the installed
metadata resolve.  Recorded here rather than dropped, because "green on the
build where the gate can run" is the claim, not "green everywhere".

**Fingerprints.**  `lumenairy/propagators/fga.py` has no document under
`docs/history/`, so the new `_immersed_exit_tolerance` needed no re-record;
`lumenairy/propagators/fft_infra.py` does have one, but this pass's edit there
is comment-only and the checker's token fingerprint drops comments and
docstrings, so `--check` reads OK on it.  Nothing else in `lumenairy/` changed.

**`.test_durations` was NOT touched** (still valid JSON, 16 282 entries).  The
four new ids are not in it; `test_audit2609_a15a_durations_staleness.py` is
green without them, so nothing was spliced rather than splice timings taken on
a contended box.

### 11.12 Commits

| defect(s) | SHA | subject |
|---|---|---|
| D1 | `8e2886a5` | `docs(fga): VERIFY-WAVE5-E D1 -- the immersed-exit guard's boundary, and what it implies for a registered air index` |
| D3 | `7e267845` | `fix(fga, tests): VERIFY-WAVE5-E D3 -- ONE definition of the immersed-exit tolerance, and a pin that ASKS it` |
| D9 | `b9b24711` | `test(fga): VERIFY-WAVE5-E D9 -- each of the four immersed-exit guard call sites is now pinned on its own` |
| D4, D5, D7 | `00e31fcf` | `fix(tests): VERIFY-WAVE5-E D4/D5/D7 -- the C8 ladder's trip counter, its rung count and its lower-gap mechanism` |
| D6, D2 | `55db416a` | `docs(fft_infra, tests): VERIFY-WAVE5-E D6 + D2 -- all ten FFT product sites, pinned as a census; the JAX known-red premise-gated` |
| D8 | `e7c9976d` | `fix(tests): VERIFY-WAVE5-E D8 -- E3's products term counts five roundings per leg, not three` |

D4/D5/D7 share one test file and one re-measurement, and D6/D2 both land in
`test_verify_wave5_e.py`, so those two commits carry more than one defect each;
every other defect has its own.  Nothing was pushed.

### 11.13 What this pass could not measure

1. **Whether the D2 patch traces.**  Unchanged from the verifier's position:
   the requested `_adrt_jax` edit was not applied, so whether
   `compute_dead=True` trips the tracer-to-ndarray conversion the comment at
   `differential.py:897` claims is still unmeasured.  The fallback (a separate
   un-differentiated `jax.vmap` pass) is named in 11.10.
2. **CI's runner mix.**  Everything here is two builds on one box, and that box
   was running several other agents' suites throughout, so every wall clock
   carries contention (reported, never asserted).  The BLAS axis is covered for
   E3 by the 16-arm `OPENBLAS_CORETYPE` ladder and for everything else by the
   two builds; the EPYC / py3.10 / numpy 2.2.6 shards have not run any of it.
   A green un-masked main-CI matrix on the merge remains the authority.
3. **The `_fga_coarse` direct-entry arm's coverage beyond its guard.**  The new
   arm enters `_fga_coarse` with the smallest arguments that carry its
   pre-guard block; it proves the guard is reachable and fires, and nothing
   about the rest of that function.  A signature change to `_fga_coarse` would
   surface here as a `TypeError` rather than an assertion, which is a real
   edit reporting itself, but it is not a contract test for the signature.
4. **Whether the census's per-file counts are the right granularity.**  Per
   file survives a site moving within a module and fires on a site added or
   removed; a site moved BETWEEN modules with the count preserved in both
   would not fire it.  That shape was not constructed.
5. **The E1 timing table.**  Untouched by this pass, and still the wall-clock
   reading the verifier could not reproduce; the decision it supports is
   insensitive to it in either direction.
