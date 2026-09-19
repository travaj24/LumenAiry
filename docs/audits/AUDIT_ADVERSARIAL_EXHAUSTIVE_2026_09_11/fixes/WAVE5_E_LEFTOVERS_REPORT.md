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
| **E3** | re-derive the s10 ULP bar two-sided from first principles | **RE-DERIVED** at run time, 27.3-34.0 ULP against a 16-arm envelope of 0-3 ULP and a signal at 5.16e+08 ULP; two stale docstring readings corrected |
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

**Benefit of (a): zero.**  Every in-library site that multiplies a dispatcher
result NAMES the other operand -- `asm.py:919/922/1391`, `carrier.py:1401/7126`,
`fresnel.py:216` are all `_fft2(...) * H` -- so with the ping-pong ON neither
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

Five of seven rungs clear 10x at 256 and seven of seven at 512, both monotone.
Beyond three beam widths the ratio is **12.9x at 256 and 490.9x at 512**.

### 3.4 The bar, two-sided

* **Below**: where the bound is inert the two fields are BIT-identical, so every
  one of the 385 inert cells of the 432-cell sweep reads ratio **exactly 1.0**
  -- not 1.0000001.  The bar of 10x is a decade above an exact identity.
* **Above**: the weakest tripping rung reads 12.9x (n = 256) / 490.9x (n = 512).

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

    bar = SAFETY * ( 6u                        products: <= 3 roundings per leg,
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
| products term (derived) | 5.333 ULP, every arm |
| reduction term (measured) | **1.493 .. 3.173 ULP** |
| **the bar** | **27.3 .. 34.0 ULP** |
| an INDEPENDENT per-leg scale (the pre-fix defect) | **5.158e+08 .. 5.168e+08 ULP** |

The bar sits 9.1x to 11.3x above the widest reading and 1.5e+07x below the real
signal.  On SANDYBRIDGE/4 the reading (3 ULP) exceeds that arm's own raw
reduction term (2.623 ULP), which is why the products term and the safety factor
are both needed and neither is decoration.

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
itself -- **3.6x** the air-vs-vacuum index difference at STP (2.77e-4, so a
caller who registers a real air index is not refused) and **~500x** below the
weakest immersion medium.  `get_glass_index('air', lambda)` returns **exactly
1.0** on this registry at every wavelength measured (633, 780, 1030, 1060, 1310,
1550 nm), so the air control sits exactly at the tolerance's origin.

An exit medium the registry cannot resolve is NOT this guard's diagnostic and
returns `None`: with a FLAT last surface the projection short-circuits and never
resolves the index at all, so raising there would refuse a prescription served
correctly today.

### 6.4 The pin

`tests/unit/test_wave5_e_exit_vertex_dead_rays.py`, 29 ids, 62 s (Windows) /
inside the WSL run below.  Every claim two-sided:

* the freeze is bit-exact on missed rows AND the rows that reached still move
  (a freeze that froze everything would pass the first arm and be a disabled
  projection);
* the premise (that the fan really vignettes) is **asserted, never skipped**;
* the frozen set is the same set `at_exit_vertex()` freezes, and a
  companion-dead-but-base-alive ray is asserted POSITIVELY to still be projected
  and still agree with `at_exit_vertex`;
* the JAX twin of the primitive is bit-identical to the NumPy one on all six
  surface classes;
* the guard fires at each of the four sites on a synthetic immersed-exit
  prescription AND does not fire on the air-terminated control;
* the tolerance's derivation is pinned (scales as `lambda / z_image`, floors at
  the budget, sits above STP air and below water) rather than its number.

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
