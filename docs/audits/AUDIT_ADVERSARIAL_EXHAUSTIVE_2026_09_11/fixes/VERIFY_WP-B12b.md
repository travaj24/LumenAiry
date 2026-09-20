# VERIFY-WP-B12b -- independent adversarial re-verification of WP-B12b
# (the GBD beamlet image leg and the shared exit-vertex projection)

Branch under test: `fix/gbd-exit-vertex-projection`, head `cad66763`, base
`1218b24f` (the WP-B12 head).  Verification branch `verify/wp-b12b-gbd`,
worktree `C:\tmp\lum_vgbd`.

Every number below was re-measured here, on my own fixtures and my own oracle.
Where a row is a re-run of one of the builder's own artefacts it says so.

Two builds throughout: **Windows** py3.14.6 / numpy 2.4.4 / scipy 1.17.1 and
**WSL** py3.12.3 / numpy 2.4.6 / scipy 1.17.1.  Every invocation carried
`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1` and
`LUMENAIRY_MEM_BUDGET_MB=4096` on the command line (VERIFY-WP-B12 D-4: an
unpinned budget makes digests chunking-dependent -- section 11 D-6 shows that
applies to GBD too), and pytest ran with `--capture=sys -p no:randomly`.

The PRE tree is my own `git archive 1218b24f lumenairy` extracted into
`C:\tmp\lum_vgbd_pre`, run in its own child process with `cwd` and
`PYTHONPATH` set there and `lumenairy.__file__` printed and asserted to live
under it; the arm label is DETECTED from the library
(`'_Rl = float(' in inspect.getsource(...)`) and written into every JSON,
never passed on the command line.

---

## 1. Verdict table

| # | claim | verdict | my numbers |
|---|---|---|---|
| 1 | the in-line copy was FOUR defects, each reproduced on the PRE tree | **CONFIRMED** | on my own ten fixtures, PRE library vs my own 3-D tracer: conic base only (asphere 0.504 wv, biconic 1.982, freeform 2.084, field-frame 3.387); flat-base guard reads exactly zero (3.835 wv = **100 %** of the sag); vacuum exit (immersed 4.023 wv on a zero sag error); mirror sign **13.154 wv = 2 x 6.568 x sec** on a zero sag error.  Identical to every printed digit on both builds |
| 2 | the per-fixture in-line OPL error table, and `projection vs at_exit_vertex() = 0.0 m` | **CONFIRMED** | the builder's own `probe_a_sag.py` re-run on this tree reproduces its committed JSON **byte for byte** (`git status` clean) -- 0.0000 / 0.9246 / 1.0858 / 4.3862 / 1.9052 / 3.3146 / 1.9779 / 16.2790 / 0.0000 waves, b12d 16.0511 (2-D) and 16.0225 (meridional).  The `at_exit_vertex` control reads **0.000e+00 m** on height and OPL.  My own independent reading of the same operator, over ten fixtures: **<= 3.4e-18 m** of height and **<= 3.7e-18 m** of optical path |
| 3 | the oracle ladder archive-to-archive, plus the biconic / freeform / field-frame arms against a 3-D oracle | **CONFIRMED and EXTENDED** | my ladder, 8 fixtures x 2 planes, PRE -> POST: conic 0.99945375 -> 0.99945412 (**+3.7e-07**), conic+k the same, asphere 0.65096 -> 0.99946, flat-base 0.04347 -> 0.99945, **biconic 0.41272 -> 0.99945**, **freeform 0.24902 -> 0.99945**, **field-frame 2.2e-07 -> 0.99945**, flat last **BYTE-IDENTICAL** at both planes.  The three classes the builder could only call "movement" are accuracy rows here |
| 4 | blast radius; `world_output_plane` untouched AND right to keep `'surface'`; no JAX GBD path | **CONFIRMED** (one wording nit) | all three local entry points byte-identical to one another on BOTH builds and in BOTH trees on my fixture (Windows `1ca64347...` -> `077c35f4...`, WSL `dddad6fb...` -> `284a6c4e...`), which REFINES the builder's "not on WSL" note -- the split is the beamlet frame's chunk boundaries, not the build (section 6.2); world-branch bundle digest IDENTICAL in both trees on both builds (`383571c3...` Windows, `9789b9a7...` WSL); forcing the world branch to `'exit_vertex'` drops it from **0.99956 to 0.05052** against my oracle (identical on both builds), so keeping `'surface'` is measured, not argued.  The "grep finds no import" wording is wrong (12 matches); the substance is right |
| 5 | the trap: forcing `reference='surface'` reconstructs PRE-v5.22, not v5.22-5.47 | **CONFIRMED** | on a conic last surface the forced arm reads **relL2 1.1845** against the shipped field and fidelity 0.0507 against my oracle, while the PRE tree's own field reads 0.99956; on the flat-BASE aspheric fixture the POST forced arm is **the same SHA-256 as the PRE tree's shipped field** (`1ca643477fa1888847029c04`), which is the premise the builder's suite relies on, now checked archive-to-archive |
| 6 | the test file: 14 ids, derived bars, premise-gated, the 4-pixel frame | **MOSTLY CONFIRMED, one P2** | 14 ids, all < 60 s on both builds; the 0.999587 frame reading **reproduces to every digit** on the builder's own conic fixture and the decision survives at the library's AUTO frame (0.999691 shipped / 0.019117 pre-repair).  BUT the suite does not see a restored `-sag` fold (D-1), and one of its assertions cannot fail (D-2) |
| 7 | the two open items | **CONFIRMED, and one is worse than recorded** | immersed exit is REACHABLE through `apply_real_lens_gbd` and served: **1846.26 waves** of missing optical path at a 2 mm leg (n_exit = 1.72), phase matching the VACUUM leg to 7.3e-05 waves.  The mirror: the local branch returns `N` sign `+1` against a traced `-1`, and the branch WP-B12b tells the caller to use **refuses this class outright** (D-4) |
| 8 | durability of the WP-B12 pins (18 ids) and the restated files | **CONFIRMED** | 18/18 green on both builds; the whole 17-file GBD + reference-plane selection green on both builds (section 13) |

**Ship recommendation: SHIP** (section 12), with three edits filed and none of
them blocking the library change itself.

---

## 2. My oracle, and what it is worth

`validation/probe_verify_b12b/vb12b_common.py`.  It imports `lumenairy` only
to build the prescriptions my fixtures name and to score the library; never
for the oracle's physics.  It is independent of BOTH earlier oracles: WP-B12b
imports a rotationally-symmetric Rayleigh-Sommerfeld ring sum from
`probe_wp_b12`, and VERIFY-WP-B12's tracer, while 3-D, is a different fixture
set and does not reach a freeform or a field-frame surface.

* **Sag** from the **quadric root** `z = (1 - sqrt(1 - (1+k)c^2u^2)) / ((1+k)c)`,
  not the rationalised `c u^2 / (1 + sqrt(...))` the library uses.
* **Intersection** by damped Newton on the implicit 3-D surface equation with
  the analytic transverse gradient -- so a BICONIC, an XY-polynomial FREEFORM
  and a FIELD-FRAME decentred surface are traced, not approximated.
* **Refraction** by vector Snell in the `(n1/n2)` form with the normal
  oriented against the incident ray; **reflection** by `d - 2(d.n)n`.
* **Propagation** by a **band-limited angular spectrum** (Matsushima &
  Shimobaba) of the geometrical-optics exit-vertex boundary field
  `E = E_in / sqrt|det d(x_v,y_v)/d(a,b)| . e^{ik.opl}`, resampled by radius
  where the optic is rotationally symmetric and by a C1 Clough-Tocher
  interpolation of `opl` and `amp` otherwise.
* **Glass**: two dispersionless model indices registered probe-locally, plus
  N-BAF10 with the Sellmeier coefficients typed in here (needed only to score
  the builder's own fixture in probe W4).

### 2.1 Controls (`probe_w0_controls.py`, both builds, identical to every
printed digit)

| control | reading |
|---|---|
| my 3-D LAST-SURFACE state vs `raytrace.trace(...).image_rays`, 10 fixtures x 168 rays | height **<= 8.1e-20 m**, optical path **<= 3.7e-18 m** |
| my EXIT-VERTEX state vs `TraceResult.at_exit_vertex()`, same rays | height **<= 3.4e-18 m**, optical path **<= 3.7e-18 m** |
| the slope-vs-direction-cosine trap, `max abs(u - L)` | **6.0e-07 .. 1.03e-01** -- 14 to 18 decades above the agreements, so the comparison is real |
| my typed N-BAF10 Sellmeier vs `lumenairy.get_glass_index`, 3 wavelengths | **0.0 exactly** |
| ASM oracle, ray quadrature halved (infidelity) | **5.5e-12 .. 4.4e-11** |
| ASM oracle, propagation refinement halved (infidelity) | **7.8e-06 .. 1.6e-05** |

The oracle's own floor is therefore **<= 1.6e-05 of infidelity**.  The
narrowest thing any accuracy claim below turns on is the 0.99 / 0.95 decision
corridor of the new tests (4e-02), which is **3.4 decades** above that floor;
the PRE/POST contrasts it separates are 0.35 to 1.0 wide, five decades above
it.  The one quantity the floor is NOT good enough for is the conic control's
3.7e-07 fidelity move, and that one is measured as a PRE/POST field
difference between two archive trees, not against the oracle.

**Shared-model caveat, stated once.**  Like both earlier oracles, mine builds
the exit boundary field from geometrical optics and propagates it exactly, so
an oracle-vs-member comparison is not neutral between a ray-based member and a
wave-based one.  It does not affect anything below: every decision here is a
PRE/POST or a shipped/mutated contrast at four to seven decades, against an
oracle that does not move between the arms.

### 2.2 My fixtures

ONE optic -- biconvex, dispersionless model glass n = 1.58, R1 = +6.0 mm,
t = 1.1 mm, semi = 0.25 mm, 780 nm, w0 = 0.15 mm, 192 x 3.2 um -- with only the
LAST surface varied, so every difference between two rows is the last surface
and nothing else.  NA 0.032 .. 0.053, Airy radius 9.4 .. 14.9 um against a
3.2 um pitch (`dx/airy` 0.215 .. 0.341), so the focal structure is resolved.
None of these is a WP-B12b fixture (that package uses N-BAF10 R = 11 mm at
1.064 um, N-LASF9 at 850 nm and an R = -20 mm mirror) nor a VERIFY-WP-B12 one.

| key | last surface | sag at the rim |
|---|---|---|
| `conic` | R = -6.0 mm (CONTROL -- the in-line copy was exact here) | 5.587 waves |
| `conic_k` | the same + k = -0.80 (CONTROL) | 5.586 |
| `asph` | A4 = 6.0e8, A6 = -9.0e15 on a curved base | 5.133 |
| `flatbase_asph` | R = inf, power in A2 = -83.33 / A4 = 5.0e8 | 3.833 |
| `bicon` | biconic Rx = -6.0 / Ry = -9.5 mm | 5.509 |
| `freeform` | XY polynomial `{(2,0): 3.0e1, (0,2): -1.8e1, (4,0): 6.0e7}` | 6.672 |
| `fieldframe` | field-frame decentre (50, -35) um | 8.972 |
| `flat_last` | plano-convex, curved FIRST -- **FLAT** (CONTROL) | 0.000 |
| `mirror` | concave MIRROR R = -15.0 mm | 6.568 |
| `immersed` | the conic optic exiting into GLASS n = 1.72 | 5.587 |

---

## 3. Claim 1 -- the four failure modes, reproduced on the PRE tree

`validation/probe_verify_b12b/probe_w1_mechanism.py` + `compare_w1.py`.  Each
arm calls the LIBRARY's `apply_prescription_persurface_to_beamlets` with
`z_image = 0`, so the image-side leg is the identity and every reading is the
reference plane and nothing else.  The comparison is against my own 3-D
tracer, and the PRE/POST phase difference is compared against my own predicted
defect `(n_exit . sign(N) . sag_true - sag_inline) . sec` with a circular mean
removed -- no unwrapping, so a wrong prediction cannot hide inside a wrap.

**Identical to every printed digit on both builds** (the one exception is the
flat control's OPL residual, 6.812e-18 vs 6.804e-18 waves -- a zero).

| fixture | sag [wv] | in-line sag error [wv] | as a fraction of the sag | **in-line OPL error [wv]** | PRE base-ray position vs my oracle | predicted | POST position vs my oracle | OPL residual after my prediction [wv] |
|---|---|---|---|---|---|---|---|---|
| `conic` | 5.587 | 0.0000 | 0.000 | **0.0000** | 6.06e-20 m | 1.10e-20 | 6.06e-20 m | 7.2e-08 |
| `conic_k` | 5.586 | 0.0000 | 0.000 | **0.0000** | 8.16e-20 | 9.75e-20 | 8.16e-20 | 7.2e-08 |
| `asph` | 5.133 | 0.5038 | 0.098 | **0.5043** | **1.7381e-08** | 1.7381e-08 | 5.46e-20 | 3.0e-05 |
| `flatbase_asph` | 3.833 | 3.8329 | **1.000** | **3.8349** | **9.5537e-08** | 9.5537e-08 | 6.06e-20 | 2.1e-04 |
| `bicon` | 5.509 | 1.9805 | 0.360 | **1.9820** | **5.8765e-08** | 5.8765e-08 | 5.46e-20 | 1.5e-04 |
| `freeform` | 6.672 | 2.0822 | 0.312 | **2.0837** | **5.9996e-08** | 5.9996e-08 | 3.36e-18 | 1.8e-04 |
| `fieldframe` | 8.972 | 3.3827 | 0.377 | **3.3873** | **1.3683e-07** | 1.3683e-07 | 3.83e-20 | 2.5e-04 |
| `flat_last` | 0.000 | 0.0000 | -- | **0.0000** | 3.03e-20 | 0.0 | 3.03e-20 | 6.8e-18 |
| **`mirror`** | 6.568 | **0.0000** | 0.000 | **13.1539** | 5.59e-20 | 3.66e-20 | 5.59e-20 | 5.6e-07 |
| **`immersed`** | 5.587 | **0.0000** | 0.000 | **4.0230** | 3.83e-20 | 2.56e-21 | 3.83e-20 | 3.5e-08 |

Read the table the way the defect taxonomy asks:

1. **Conic base only.**  Four classes with a ZERO sag error on the two conic
   controls and a large one everywhere else: 0.50, 1.98, 2.08, 3.38 waves.
   The PRE tree's returned base-ray positions miss my own exit-vertex state by
   exactly the predicted `(sag_true - sag_inline) . u`, to every printed
   digit, on all five defect rows; the POST tree's sit at 5e-20 m.
2. **The `np.isfinite(_Rl)` guard reads exactly zero on a flat base.**  The
   `flatbase_asph` row is **100.0 %** of the sag, not a fraction of it.
3. **Vacuum exit assumed.**  The `immersed` row has a sag error of ZERO --
   the surface is a conic -- and an optical-path error of **4.0230 waves**,
   which is `(n_exit - 1) . sag . sec` with n_exit = 1.72.  This isolates
   failure mode 3 on its own, which the builder's table does not do.
4. **Forward-going exit ray assumed.**  The `mirror` row also has a sag error
   of ZERO and an optical-path error of **13.1539 waves = 2 x 6.568 x sec**.
   The sign, and nothing else.

The last column is the decisive one: subtract my own predicted defect from the
measured PRE-minus-POST phase and 0.5 to 13.2 waves of disagreement collapse
to **7e-08 to 2.5e-04 waves**.  The defect is not merely present, it is the
one predicted, to four or five decades.

---

## 4. Claim 2 -- the builder's own table, re-run

`validation/probe_gbd_projection/probe_a_sag.py` and `probe_c_decompose.py`,
run unchanged on this tree with `B12B_TREE` and `PYTHONPATH` pinned.  Both
scripts rewrote their committed JSON and **`git status` reported no change**:
the artefacts are bit-reproducible on this build.

| last surface | published | my re-run |
|---|---|---|
| conic / conic + k | 0.0000 waves | **0.0000** (1.459e-23 / 1.458e-23 m of height) |
| even asphere A4 | 0.9246 | **0.9246** |
| even asphere A4 / A6 | 1.0858 | **1.0858** |
| flat-base aspheric | 4.3862 (100 % of the sag) | **4.3862**, frac **1.0000** |
| biconic | 1.9052 | **1.9052** |
| freeform | 3.3146 | **3.3146** |
| field frame | 1.9779 | **1.9779** |
| concave mirror | 16.2790 | **16.2790** on a 0.0000 sag error |
| flat last | 0.0000 exactly | **0.0000**, dh 0.000e+00 exactly |
| WP-B12's own fixture, 2-D fan | 16.0511 (0.7196 of sag) | **16.0511 / 0.7196** |
| WP-B12's own fixture, meridional 401 | 16.0225 (0.7193) | **16.0225 / 0.7193** |
| the projection vs `TraceResult.at_exit_vertex()` | 0.0 m | **0.000e+00 m** on height AND optical path, 200 rays |

Probe C likewise: conic `full_vs_fs_sag_sec` **relQ 6.064e-05, relAmp
7.402e-06, dPos 0.000e+00 m exactly, dPhase 3.444e-06 rad**; `full_vs_state_only`
relQ 4.334e-04; the flat control 0.0 / 0.0 / 0.0.  One published number in that
section does NOT reproduce -- see D-3.

---

## 5. Claim 3 -- the oracle ladder, archive to archive, with a 3-D oracle

`validation/probe_verify_b12b/probe_w2_ladder.py` + `compare_w2.py`.  Both arms
are separate child processes in separate trees; the beamlet frame is named
(`sample_step=4, waist_factor=4.0`, justified in section 8.3) so the three
public entry points are comparable byte for byte.

### 5.1 Windows build

| fixture | plane | `pre` | `post` | field bytes |
|---|---|---|---|---|
| conic (CONTROL) | exit vertex | 0.99945375 | **0.99945412** | changed |
| conic (CONTROL) | focus 4.9871 mm | 0.99956241 | **0.99956277** | changed |
| conic + k (CONTROL) | exit vertex | 0.99945374 | **0.99945411** | changed |
| conic + k (CONTROL) | focus | 0.99956242 | **0.99956278** | changed |
| even asphere A4 / A6 | exit vertex | 0.65096257 | **0.99945562** | changed |
| even asphere A4 / A6 | focus 5.2653 mm | 0.65092345 | **0.99957092** | changed |
| flat-base aspheric | exit vertex | 0.04347091 | **0.99944696** | changed |
| flat-base aspheric | focus 5.5524 mm | 0.04348041 | **0.99957156** | changed |
| **biconic** | exit vertex | 0.41272470 | **0.99945230** | changed |
| **biconic** | focus 5.4228 mm | 0.41276093 | **0.99957225** | changed |
| **freeform** | exit vertex | 0.24902308 | **0.99945137** | changed |
| **freeform** | focus 5.0780 mm | 0.24901133 | **0.99956556** | changed |
| **field frame** | exit vertex | 0.00000022 | **0.99945261** | changed |
| **field frame** | focus 4.9860 mm | 0.00000023 | **0.99956302** | changed |
| **flat last (CONTROL)** | exit vertex | 0.99951178 | 0.99951178 | **IDENTICAL** `9c44ed861a987b97d2ce2065` |
| **flat last (CONTROL)** | focus 4.1962 mm | 0.99960340 | 0.99960340 | **IDENTICAL** `fb154c8ce216352113c6c4c4` |

Four things this says, two of them new.

1. **Every defect row lands where the controls already sit.**  The seven
   curved-last-surface `post` rows read 0.99944696 .. 0.99945562 at the vertex
   and 0.99956277 .. 0.99957225 at the focus -- a spread of **8.7e-06** and
   **9.5e-06** across a conic, a conic-plus-k, an asphere, a flat-base
   asphere, a biconic, a freeform and a field-frame decentre.  One number,
   seven different last surfaces.  Before the repair the same seven read
   0.99945, 0.99945, 0.65096, 0.04347, 0.41272, 0.24902 and 2.2e-07.
2. **The biconic, the freeform and the field-frame arms now have a diffraction
   score, and it is not "movement".**  WP-B12b reported the biconic's focal
   peak rising 55 % and the freeform's 147 % and explicitly declined to call
   that accuracy (its open item 4).  Against my 3-D oracle those two arms go
   from **0.4127 to 0.9995** and **0.2490 to 0.9996**, and the field-frame arm
   -- which WP-B12b could only report as "under 1 % at the focus" -- goes from
   **2.3e-07 to 0.9996**.  The field-frame row is the most striking: its
   pre-repair field is not merely degraded, it is orthogonal to the truth.
3. **The two conic controls move in the seventh decimal, upward.**
   0.99945375 -> 0.99945412 (**+3.7e-07**) at the vertex and 0.99956241 ->
   0.99956277 (**+3.6e-07**) at the focus, and +3.7e-07 / +3.6e-07 for the
   conic-plus-k row.  That is the Jacobian projection of the builder's section
   3, arriving in the field where a 6e-05 relative change in `Q` should put
   it, and in the projection's favour on all four readings.  (My numbers are
   larger than the builder's +2.3e-07 / +3.1e-07 because my optic is faster;
   the sign and the decade agree.)
4. **The flat control is byte-identical at BOTH planes**, digest for digest,
   between two archive trees.

### 5.2 WSL build

Every `post` fidelity in the table above agrees with the Windows build to all
eight printed digits -- 0.99945412 / 0.99956277, 0.99945411 / 0.99956278,
0.99945562 / 0.99957092, 0.99944696 / 0.99957156, 0.99945230 / 0.99957225,
0.99945137 / 0.99956556, 0.99945261 / 0.99956302, 0.99951178 / 0.99960340 --
and so do the forced-`'surface'` rows (0.999563 / 0.050725 / relL2 1.1845;
0.999572 / 0.043480 / 1.2185; 0.999603 / 0.999603 / 0.0000 identical).  The
SHA-256 digests do not agree across builds and are not expected to (a coherent
sum over hundreds of beamlets on two different LAPACKs).  **Every
byte-identity claim in this report is WITHIN one build, between two trees**;
the cross-build statement is always the fidelity.

Probes W0, W1, W3, W4 and W5 are identical to every printed digit on the two
builds, on every row (the only exceptions are three quantities that are zero
up to rounding: the flat control's OPL residual, 6.812e-18 against 6.804e-18
waves, and two exact-zero position readings in the 1e-20 decade).

The WSL PRE arm was taken for the FORCED comparison and the entry points but
not for the full ladder: each ladder arm costs about half an hour on this box,
and the cross-build question it would answer -- whether the DEFECT reads the
same on a second LAPACK -- is already answered at the ray level by probe W1,
which is identical to every printed digit on both builds on all ten rows.
What WAS re-taken on WSL is the decisive byte-level cross-check of section 7,
and it reproduces: the POST tree's forced-`'surface'` field on the flat-BASE
aspheric fixture has the SAME SHA-256 as the PRE tree's shipped field
(`dddad6fbc05b1d668fe3eabf`), and so does the flat control
(`275430cef465b230e5892425`), while the conic fixture's two differ.

---

## 6. Claim 4 -- blast radius

### 6.1 The entry points and the world branch

My `flatbase_asph` fixture at its own traced best focus, archive-to-archive,
with the beamlet frame named explicitly so the three entries are comparable:

| entry point | Windows `pre` | Windows `post` | WSL `pre` | WSL `post` |
|---|---|---|---|---|
| `apply_real_lens_gbd` | `1ca643477fa1888847029c04` | `077c35f46f9f283c32736e99` | `dddad6fbc05b1d668fe3eabf` | `284a6c4e8762b59c6d45891d` |
| `apply_real_lens_universal(method='gbd')` | `1ca64347...` | `077c35f4...` | `dddad6fb...` | `284a6c4e...` |
| `propagate_gbd_through_prescription(per_surface=True)` | `1ca64347...` | `077c35f4...` | `dddad6fb...` | `284a6c4e...` |
| **`world_output_plane` branch (bundle digest)** | `383571c32a40dc641ef187e9` | `383571c32a40dc641ef187e9` | `9789b9a7086b5b4b9a256f5b` | `9789b9a7086b5b4b9a256f5b` |
| the field's fidelity against my oracle | 0.04348041 | **0.99957156** | 0.04348041 | **0.99957156** |

The dispatcher dispatches, all three local entries move together, and the
world branch does not move at all -- on BOTH builds, within each build.

Note the refinement this adds to WP-B12b's own record.  It reports
`propagate_gbd_through_prescription` as byte-identical to `apply_real_lens_gbd`
on Windows and NOT on WSL.  On my fixture, with the frame named, the two are
byte-identical on **both** builds in **both** trees.  The divergence is
therefore not a property of the build: it is a property of the beamlet frame
and the chunk boundaries it produces (section 6.2).  The builder's decision to
assert byte identity only for the dispatcher pair, and a 0.999 fidelity bar for
this one, is still the right shape -- it just holds for a reason the report
does not give.

### 6.2 Why `propagate_gbd_through_prescription` can differ in its last bits

WP-B12b records that this entry is byte-identical to `apply_real_lens_gbd` on
Windows and not on WSL, and does not say why.  `probe_w5_entries.py` walks the
two routes stage by stage (Windows):

| stage | reading |
|---|---|
| beamlets decomposed, raw vs aperture-clipped | 784 vs 784 |
| after `_prune_zero_beamlets` (only `apply_real_lens_gbd` does this) | **437** |
| beamlet-bundle digests, raw vs pruned | `b8b07cce94ab970e2be89c4a` vs `2783b785e7a9d1284d2e421f` -- **different bundles** |
| beamlets SURVIVING the trace, from either bundle | **437 and 437** |
| the reconstructed field from either bundle | **byte-identical**, `max abs` difference **0.0** |

So the two routes reach the same 437 live beamlets by different roads -- one
prunes the dark ones up front, the other lets the trace vignette them -- and on
this fixture the surviving set, its ORDER and the resulting sum are identical.
What is genuinely build-dependent is the GROUPING of the coherent sum: the same
evolved bundle reconstructed under three memory budgets gives **two different
digests** (`mem_budget_mb` 512 -> `8829ce8a...`, 4096 and 1 -> `da267d04...`),
while `chunk_beamlets` 2048 / 512 / 97 all agree.  The last-bits agreement
between the two entries is therefore a property of where the chunk boundaries
happen to fall on a given build -- which is exactly why the builder's test file
asserts BYTE identity only for the dispatcher pair and a 0.999 FIDELITY bar for
this one.  **That decision is correct** and is the right shape.  See D-6.

### 6.3 The world branch keeps `'surface'`, and that is measured

The builder's suite records WHICH keyword each branch asks for and never scores
the world branch.  Forcing `reference='exit_vertex'` onto it through the public
keyword (my `_Pin` context, which is exactly the `world_exit_vertex` mutation
reached from outside), on my conic fixture at its own traced best focus,
against my oracle:

| fixture | shipped (`'surface'`) | forced `'exit_vertex'` | relL2(forced, oracle) |
|---|---|---|---|
| `conic`, 192 x 3.2 um (`probe_w3_openitems.py`) | **0.99956241** | **0.05052450** | 1.18167 |
| `asph`, same grid | **0.99957052** | **0.04645043** | 1.20463 |
| `conic`, 112 x 3.6 um (the new test's grid) | **0.99916732** | **0.15342213** | -- |

and on the shipped arm the world branch reproduces the LOCAL branch on the
same plane with the same frame to **2.3e-07** of fidelity (0.99916756 against
0.99916732 at the test grid), which is the "it reduces to the local path on an
unfolded system" contract holding.

The sag IS double-counted there, by the amount that turns a 0.9996 field into
a 0.0505 one.  The comment at the call site is right, and it is now a test
(`test_the_world_branch_keeps_the_surface_plane_because_the_exit_vertex_one_double_counts`).

### 6.4 No JAX per-surface GBD path

Confirmed.  `lumenairy/elements/lenses_gbd.py` contains no `jax` or `jnp` at
all; `propagators/gbd.py` has no `import jax` / `import jax.numpy` anywhere,
and `apply_prescription_persurface_to_beamlets` calls `np.asarray` on every
input array, so even a JAX bundle is converted before the differential
primitive is reached.  The builder's supporting sentence -- "`grep -n
'jax\|jnp' lumenairy/propagators/gbd.py` finds no import" -- is literally
false (that grep returns **12** matches: docstrings, `is_jax_array`, the
`jnp.at[].add` scatter comment), but the claim it supports is true.  Wording
nit, recorded in D-8.

---

## 7. Claim 5 -- the trap

Forcing `reference='surface'` is NOT "the v5.22 .. 5.47.0 behaviour" on a
curved base.  Measured three ways on my own fixtures:

| fixture | POST shipped | POST forced-`'surface'` | relL2(forced, shipped) | PRE shipped | PRE forced |
|---|---|---|---|---|---|
| `conic` (curved base) | 0.99956277 | **0.05072500** | **1.1845** | 0.99956241 | 0.99956241 (identical -- the PRE default already IS `'surface'`) |
| `flatbase_asph` | 0.99957156 | 0.04348041 | 1.2185 | 0.04348041 | 0.04348041 |
| `flat_last` | 0.99960340 | 0.99960340 | 0.0000 | 0.99960340 | 0.99960340 |

Two readings settle it.

* On a **curved base** the forced arm (0.0507) is nowhere near the PRE tree's
  own field (0.99956).  Forcing the plane reconstructs **pre-v5.22** -- no
  vertex correction at all -- and the relative L2 against the shipped field is
  **1.1845**, i.e. the two fields are essentially unrelated.  (The builder
  quotes 1.32 on its own fixture; mine is a different optic, same story.)
* On the **flat-BASE aspheric** fixture the POST forced arm's SHA-256 is
  `1ca643477fa1888847029c04` -- **the same bytes as the PRE tree's shipped
  field**.  That is the premise the builder's suite leans on (the in-line copy
  read that sag as exactly zero), now checked archive-to-archive rather than
  argued, and it holds exactly.

---

## 8. Claim 6 -- the test file, and the mutation matrix

### 8.1 What is there

14 ids, collected.  Every bar is derived at run time (`1e4 * eps * scale`,
`10 * floor`, the oracle's own quadrature convergence), every contrast is
premise-gated, and the file imports WP-B12's oracle rather than re-typing it.
On the loaded box the file runs 49 .. 70 s for all 14 ids together; the
slowest id is well inside 60 s on both builds.  No wall-clock assertion.

### 8.2 The mutation matrix

`validation/probe_verify_b12b/vb12b_mutate.py` -- a pytest plugin.  The three
`gbd.py` mutations are applied by recompiling
`apply_prescription_persurface_to_beamlets` from its own source with one
textual edit, in the module's own globals, rebinding it everywhere the library
re-exports it, AND registering the mutated source with `linecache` so
`inspect.getsource` returns the MUTATED text -- without that last step the
builder's token check raises `OSError` and gets scored as a catch when it is
only an artefact of the vehicle.

The matrix was run in full on BOTH builds
(`validation/probe_verify_b12b/mutations_win32_314.txt` and
`mutations_linux_312.txt`) and is **cell-for-cell identical**: all 32 cells
(8 mutations x 4 files) give the same pass / fail / xfail counts and the same
named RED ids on Windows and on WSL.  One table therefore serves for both.

| mutation | `test_audit2609_b12b` (14) | `test_verify_b12b` (9, MINE) | `test_audit2609_b12` (14) | `test_verify_b12` (4) |
|---|---|---|---|---|
| none | 14 pass | 8 pass + 1 xfail | 14 pass | 4 pass |
| **`fold_restored`** (the deleted `-sag` fold put back) | **14 PASS** | **6 RED** | 14 pass | 4 pass |
| **`fold_restored_renamed`** (the same, locals renamed) | **14 PASS** | **6 RED** | 14 pass | 4 pass |
| `local_surface` | 4 RED | 6 RED | 14 pass | 4 pass |
| `world_exit_vertex` | 1 RED (the behavioural recording) | **1 RED (numerically, for the first time)** | 14 pass | 4 pass |
| `state_only` | 14 pass | 8 pass + 1 xfail | **2 RED** | **1 RED** |
| `sign_plus` | **2 RED** | 8 pass + 1 xfail | **1 RED** | 4 pass |
| `identity` | 11 RED | 6 RED | 9 RED | 4 RED |

The `fold_restored_renamed` row was additionally taken as a REAL FILE EDIT in
a separate `git archive` tree (`C:\tmp\lum_vgbd_mutsrc`, D-1), where
`lumenairy.__file__` resolves under that tree and the builder's module-level
token ban can see the real source: **still 14 passed**.

Every mutation the brief names is caught by a **named** test:

* (a) the restored `-sag` fold -- **NOT caught by the builder's suite**; caught
  by my `test_a_conic_last_surface_reproduces_a_diffraction_oracle_and_a_double_vertex_correction_does_not`
  and five more.  This is D-1.
* (b) `reference='surface'` in the local branch -- `test_the_local_branch_asks_for_the_exit_vertex_plane`,
  `test_the_repaired_field_reproduces_a_diffraction_oracle_where_the_old_one_did_not`,
  `test_a_flat_last_surface_field_is_bit_identical_to_the_surface_arm`,
  `test_the_public_entry_points_follow_the_beamlet_function` (4 named ids).
* (c) `reference='exit_vertex'` in the world branch -- caught by the builder's
  `test_the_local_branch_asks_for_the_exit_vertex_plane` as a BEHAVIOURAL
  recording only, and numerically for the first time by my
  `test_the_world_branch_keeps_the_surface_plane_because_the_exit_vertex_one_double_counts`.
* (d) the Jacobian projection bypassed -- caught by WP-B12's
  `test_the_projected_jacobian_is_the_derivative_of_the_projected_map` and
  `test_per_surface_projection_moves_only_the_last_local_transfer`, plus
  VERIFY-WP-B12's `test_the_two_backends_agree_on_the_exit_vertex_plane...`.
  NOT caught by the B12b file, which is acceptable: it is the primitive's
  property and the primitive's suite owns it.
* (e) the mirror sign flipped -- caught by
  `test_the_projection_reproduces_the_library_s_own_exit_vertex_operator[mirror]`
  and `test_a_mirror_terminated_prescription_gets_the_propagation_sign`, plus
  WP-B12's `test_a_mirror_terminated_prescription_projects_with_the_right_sign`.

### 8.3 Is the 4-pixel frame a hidden pin?  No -- it reproduces.

`probe_w4_frame.py` re-takes the builder's own comment number on the builder's
own fixtures, with MY oracle, on this build:

| fixture | frame | fid vs my oracle | forced-`'surface'` arm | fid vs the AUTO frame |
|---|---|---|---|---|
| WP-B12b's flat-base aspheric (96 x 6.6 um) | auto (library default) | 0.999691 | **0.019117** | 1.000000 |
| | `ss=2 / wf=2` | 0.999718 | 0.030934 | 0.999933 |
| | **`ss=4 / wf=4` (the test file)** | 0.999360 | 0.088543 | **0.999563** |
| | `ss=6 / wf=6` | 0.998670 | 0.115533 | 0.998821 |
| WP-B12b's optic with a CONIC last surface | auto | 0.999704 | 0.062723 | 1.000000 |
| | `ss=2 / wf=2` | 0.999737 | 0.077591 | 0.999931 |
| | **`ss=4 / wf=4`** | 0.999399 | 0.134718 | **0.999587** |
| | `ss=6 / wf=6` | 0.998852 | 0.214983 | 0.999055 |

The WSL build prints the flat-base rows character for character (0.999691 /
0.019117 / 1.000000; 0.999718 / 0.030934 / 0.999933; 0.999360 / 0.088543 /
**0.999563**; 0.998670 / 0.115533 / 0.998821).

The comment's **0.999587** is reproduced to every digit.  More importantly the
DECISION the frame carries does not move with it: at the library's own auto
frame the shipped field scores 0.9997 and the pre-repair arm 0.019 -- the same
verdict the file's 0.99 / 0.90 bars record, with a wider margin.  The frame is
a cost choice, and my own file re-derives that at run time
(`test_the_named_beamlet_frame_is_a_cost_choice_not_a_result`) instead of
trusting a comment.

---

## 9. Claim 7 -- the two open items, measured

### 9.1 The immersed exit is REACHABLE and silently served

`probe_w3_openitems.py`, my `immersed` fixture (the conic optic exiting into a
model glass n = 1.72), `z_image = 2 mm`:

| reading | value |
|---|---|
| reachable through `apply_prescription_persurface_to_beamlets`? | **yes** |
| reachable through the public `apply_real_lens_gbd`? | **yes -- served, no warning, no refusal** |
| optical path the leg omits, `(n_exit - 1) . z_image . sec` | **1846.26 waves** |
| the returned leg phase against a VACUUM prediction | residual **7.3e-05 waves** |
| the returned leg phase against an INDEXED prediction | residual **6.8e-02 waves** |
| is there a guard at the FGA sites in THIS tree? | **no** |

So GBD's image leg is a vacuum leg, the discrepancy is two orders of
magnitude larger than the largest sag defect in this package (16.28 waves),
and the path is reachable from the public API.  WAVE5-E (commit `9a83042c`, branch
`fix/wave5-item-e-leftovers`, NOT in this tree) added
`_require_non_immersed_exit` with a derived tolerance at the four `fga.py`
sites; when that merges, `propagators.gbd` becomes the only remaining
unguarded consumer.  **Requested edit: D-5.**

### 9.2 The mirror -- and the remedy the builder names does not exist

Same probe, my `mirror` fixture (concave mirror R = -15 mm, geometric focus
7.5 mm BEHIND the vertex), against my own 3-D trace:

| arm | transverse position vs truth | spot RMS (truth 5.00e-08 m) | returned `N` sign (true `-1`) | leg phase residual |
|---|---|---|---|---|
| LOCAL, `z_image = +7.5 mm` | **7.848e-04 m** | 3.879e-04 m | **+1** | piston +0.1617 wv, resid 3.1e-04 |
| LOCAL, `z_image = -7.5 mm` | 2.830e-19 m | 5.001e-08 m | **+1** | piston -0.1009 wv, **resid 4.83e-01 wv** |
| WORLD, same plane | -- | -- | -- | **`NotImplementedError`** |

Three things.

1. With the sign of `z_image` a caller would naturally use, the local branch is
   defocused by twice the focal length -- the spot RMS is **7756x** the true
   one (3.879e-04 m against 5.001e-08 m).
2. Flipping `z_image`'s sign salvages the TRANSVERSE positions exactly
   (2.8e-19 m) but not the optical path: the leg phase residual is **0.48
   waves** peak, i.e. the piston is applied with the wrong sign.  That is
   precisely the unsigned `Nz2` the builder describes, now measured on both
   observables.
3. `world_output_plane` -- the branch the builder's open item 2 says to point
   a refusal at -- **refuses a curved terminating mirror outright**:
   `_unfolded_equivalent_surfaces` raises "curved (powered) fold mirrors are
   not yet supported".  On exactly the class where the local branch is wrong
   there is nowhere to send the caller.  **Requested edit: D-4**, and it is
   NOT the edit the builder recommends.

---

## 10. Claim 8 -- durability of the WP-B12 pins

`tests/unit/test_audit2609_b12_fga_reference_plane.py` (14 ids) +
`tests/unit/test_verify_b12_fga_reference_plane.py` (4 ids) = **18 ids**,
green on both builds under this change (section 13).  The second file is not
on this branch -- it belongs to the VERIFY-WP-B12 package -- so it was copied
in from `verify/wp-b12` for the run and removed again before the commit; it is
NOT part of this package's diff.

The eight restated files and the whole GBD selection are in section 13.

---

## 11. Defects

### D-1 (P2, test coverage -- CLOSED HERE) -- a restored `-sag` fold survives the entire WP-B12b suite

**What.**  `test_the_module_carries_no_second_sag_kernel` is the only thing in
the builder's 14 ids that reacts to the deleted block coming back, and it
reacts by NAME: it bans the tokens `_Rl`, `_kl`, `_cl`.  Nothing numerical in
the file sees a double sag correction, because the one field decision in the
file is taken on the FLAT-BASE aspheric fixture, where the in-line copy's sag
was identically zero and restoring it changes nothing.  The conic last
surface -- the one class where a restored fold does the most damage -- is never
scored against a diffraction oracle at all.

**Reproducer (file edit, not a monkeypatch).**  `git archive HEAD` into
`C:\tmp\lum_vgbd_mutsrc`, then in `lumenairy/propagators/gbd.py` replace
`    t = z_image / Nz2` with the deleted block spelled with different locals:

```python
    _r_last = float(getattr(surfs[-1], 'radius', np.inf))
    _k_last = float(getattr(surfs[-1], 'conic', 0.0) or 0.0)
    if np.isfinite(_r_last) and _r_last != 0.0:
        _c_last = 1.0 / _r_last
        _rho2 = dt.x ** 2 + dt.y ** 2
        _vsag = _c_last * _rho2 / (1.0 + np.sqrt(np.maximum(
            1.0 - (1.0 + _k_last) * _c_last * _c_last * _rho2, 0.0)))
    else:
        _vsag = np.zeros_like(dt.x)
    t = (z_image - _vsag) / Nz2
```

Run from that tree (`lumenairy.__file__` asserted under it):
`tests/unit/test_audit2609_b12b_gbd_projection.py` -> **14 passed**.  My file
-> **6 failed**, the first being the conic oracle decision (shipped 0.9992
against the double-corrected 0.0507 on the same fixture).

**Status: CLOSED HERE** by
`tests/unit/test_verify_b12b_gbd_projection.py::test_a_conic_last_surface_reproduces_a_diffraction_oracle_and_a_double_vertex_correction_does_not`,
which reaches the double correction through
`_project_to_exit_vertex_plane` applied twice -- the same operator, so no
respelling can evade it.  No library edit requested.

### D-2 (P3, test defect) -- one of WP-B12b's assertions cannot fail

**What.**  In `test_the_module_carries_no_second_sag_kernel`:

```python
for tok in tokenize.generate_tokens(io.StringIO(fn_src).readline):
    if tok.type in (tokenize.COMMENT, tokenize.STRING):
        continue
    fn_code.append(tok.string)
fn_flat = ' '.join(fn_code).replace(' ', '')
assert "'radius'" not in fn_flat and '"radius"' not in fn_flat
```

`'radius'` is a STRING token and STRING tokens are dropped two lines above, so
the searched text can never contain it.  **Proven, not argued**: run that exact
check against the PRE tree's version of the function, which literally reads
`getattr(surfs[-1], 'radius', np.inf)` --

```
"'radius'" not in fn_flat  -> True
'"radius"' not in fn_flat  -> True
```

The assertion passes on the source it was written to reject.

**Requested edit** (`tests/unit/test_audit2609_b12b_gbd_projection.py`, in
`test_the_module_carries_no_second_sag_kernel`): keep STRING tokens for this
one check, e.g. build a second stream that drops only `COMMENT`, or search
`fn_src` directly for `getattr(surfs[-1], 'radius'`.  One line.

### D-3 (P3, documentation) -- a published cross-build reading that does not reproduce

WP-B12b section 7.4: "Probe C is identical to every printed digit on every row
but the flat control's `dPhase`, where the two builds read **8.318e-17** and
**8.298e-17** radians".  The shipped JSONs read

| file | `max_relative_dphase_rad` |
|---|---|
| `probe_c_decompose_win32_314.json` | **6.796869888613907e-17** |
| `probe_c_decompose_linux_312.json` | **6.760231407720553e-17** |

and re-running `probe_c_decompose.py` on this tree reproduces those JSONs
**byte for byte** (`git status` clean, Windows reads 6.797e-17).  The published
pair is stale.  **Requested edit**: replace the two numbers in section 7.4 with
`6.797e-17` / `6.760e-17`, or state which run they came from.

### D-4 (P2, library -- open item 2, remedy is wrong) -- a mirror-terminated LOCAL branch has nowhere to be sent

Measured in section 9.2.  WP-B12b's open item 2 says "a mirror-terminated
prescription through the LOCAL branch should probably be refused with a message
naming `world_output_plane`".  `world_output_plane` **refuses a curved
terminating mirror itself** (`NotImplementedError: world_output_plane: curved
(powered) fold mirrors are not yet supported`), so that message would send a
caller to a dead end.

**Requested edit** (`lumenairy/propagators/gbd.py`, the LOCAL branch of
`apply_prescription_persurface_to_beamlets`, right after `surfs` is built):
refuse, and say what is actually true --

```python
    if world_output_plane is None and any(
            getattr(s, 'is_mirror', False) for s in surfs):
        raise NotImplementedError(
            "apply_prescription_persurface_to_beamlets: a MIRROR-terminated "
            "prescription is not supported on the local-frame branch.  The "
            "image-side leg uses Nz2 = 1/sqrt(1+u^2), which is positive "
            "whatever the true N, so after a mirror the returned direction "
            "and the leg both point along +z while the light travels toward "
            "-z (measured: the spot RMS is 7.8e3 x the traced one at "
            "z_image = +f, and the leg piston carries the wrong sign at "
            "z_image = -f).  world_output_plane refuses a CURVED fold "
            "itself, so it is not an alternative for this class; propagate "
            "to the mirror and continue the reverse leg yourself.")
```

Stated as a recommendation, not applied: `lumenairy/` is out of my ownership
here.  The sign fix is NOT the right edit -- `Nz2` feeds `new_dir`, the leg
length and the Moebius step, and three of the branch's other invariants assume
a forward-going ray, so a one-line sign flip would trade a loud wrong answer
for a quiet one.  **Refusal, by measurement.**

### D-5 (P2, library -- open item 1, reachable) -- the image leg is a vacuum leg and an immersed exit is served

Measured in section 9.1: **1846.26 waves** omitted at a 2 mm leg with
`n_exit = 1.72`, reachable through the public `apply_real_lens_gbd`, with no
warning.  **Requested edit** (`lumenairy/propagators/gbd.py`, the LOCAL branch,
immediately before `t = z_image / Nz2`): the same shape WAVE5-E gave the four
`fga.py` sites -- resolve `n_exit`, and refuse when
`|n - 1| > waves_budget * wavelength / max(|z_image|, wavelength)` with
`waves_budget = 1e-3`, returning `None` (and NOT raising) for an exit medium
the registry cannot resolve.  The cleanest form is to import and call
`propagators.fga._require_non_immersed_exit` once that lands on `main`, so
there is one guard and not two.

### D-6 (P3, evidence methodology) -- GBD's field digests depend on the memory budget

Measured two ways.  At the reconstruction (section 6.2): the SAME evolved
bundle under `mem_budget_mb` 512 / 4096 / 1 gives **two different SHA-256
digests**, because `_reconstruct_windowed` chunks each bucket to stay under the
budget and the chunk boundaries change the grouping of a `bincount`
scatter-add.  And through the PUBLIC entry point, `apply_real_lens_gbd` on one
fixture with only `LUMENAIRY_MEM_BUDGET_MB` varied:

| `LUMENAIRY_MEM_BUDGET_MB` | SHA-256 of the returned field | peak |
|---|---|---|
| unset | `8829ce8a2a00bd96011c95cd` | 7.216980742677102 |
| 4096 | `8829ce8a2a00bd96011c95cd` | 7.216980742677102 |
| 512 | `8829ce8a2a00bd96011c95cd` | 7.216980742677102 |
| **64** | **`b22dc47cd9c7a6b26f5b6730`** | 7.2169807426771 |
| **8** | **`f10a1f3f748bd47b749e3116`** | 7.216980742677099 |

(The variable is a hard CEILING, so anything at or above the 512 default is a
no-op; below it the bytes move while the field agrees to 1e-15.)  WP-B12b's
section 7 preamble lists the three thread pins and not this one.  Its own
byte-identity readings are still sound (both arms of each comparison ran with
the variable unset, so both saw the default 512), but the evidence is only
reproducible with the budget pinned -- the same finding VERIFY-WP-B12 recorded
as its D-4 for FGA, now shown for GBD.  **Requested edit**: add
`LUMENAIRY_MEM_BUDGET_MB` to WP-B12b section 7's stated invocation
environment.

### D-7 (P3, wording) -- "the five post rows agree to 1.3e-06"

WP-B12b section 4.2.  Those five readings are 0.99889613, 0.99889613,
0.99889482, 0.99889591, 0.99889415; `max - min` is **1.98e-06** and 1.3e-06 is
the largest deviation from their MEAN.  Both are defensible numbers, but
"agree to X" normally means the spread.  One clarifying clause.

### D-8 (P3, wording) -- "grep finds no import"

WP-B12b section 5, the JAX row: "`grep -n 'jax\|jnp' lumenairy/propagators/
gbd.py` finds no import".  That grep returns **12** matches.  There is no
`import jax` / `import jax.numpy`, and the substantive claim (no JAX
per-surface-GBD path) is correct and independently verified in section 6.4 --
only the supporting sentence is wrong.

---

## 12. Ship recommendation

**SHIP.**

The library change is right, and it is right for the reasons given.  I
re-measured all four failure modes against my own tracer on my own fixtures
and the PRE library reproduces each one, including the two the builder's table
cannot separate (the mirror's sign and the vacuum-exit assumption, both of
which have a ZERO sag error and a large optical-path error).  The repaired
field reproduces an independent 3-D diffraction oracle at **0.99945 / 0.99957**
on seven different curved last-surface classes -- including the biconic, the
freeform and the field-frame classes the builder could only report as movement,
which come from 0.4127, 0.2490 and 2.3e-07.  The flat control is byte-identical
across two archive trees at both planes.  The `world_output_plane` branch is
untouched, and keeping it on `'surface'` is now a measurement (0.99956
against 0.05052), not a comment.

Nothing in D-1 .. D-8 is a defect in the shipped library change.  D-1 and D-2
are gaps in the new test file, and D-1 is closed here by nine new ids.  D-4 and
D-5 are the builder's own open items, both confirmed and both worse than
recorded -- D-5 is reachable from the public API and costs 1846 waves, and
D-4's recommended remedy does not exist.  Neither is a regression: both
pre-date this package and neither is made worse by it.  They should be filed
for the maintainer, not held against this branch.

The three edits I would ask for before the next release, in priority order:

1. **D-5** -- guard the immersed exit in `propagators.gbd`, reusing WAVE5-E's
   `_require_non_immersed_exit` once it lands.  P2, reachable, silent.
2. **D-4** -- refuse a mirror-terminated LOCAL branch, with a message that does
   NOT name `world_output_plane` as the alternative.  P2, silent, and the
   currently documented workaround is a dead end.
3. **D-2** -- one line in `test_audit2609_b12b_gbd_projection.py` so that
   assertion can fail.  P3.

Plus two documentation corrections (D-3, D-6) and two wording nits (D-7, D-8).

---

## 13. Tests run

Every invocation carried
`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1` and
`LUMENAIRY_MEM_BUDGET_MB=4096` on the command line, and
`--capture=sys -p no:randomly`, from `C:\tmp\lum_vgbd`, on 2026-09-19.

**The box.**  Between 5 and 12 other heavy python processes (sibling Wave-5
agents and my own parallel Windows / WSL streams) were resident for most of
this package.  Wall clocks are REPORTED, never asserted; no test in this
package contains a timing assertion.

### 13.1 pytest

| selection | build | result | duration |
|---|---|---|---|
| `test_verify_b12b_gbd_projection.py` (NEW, 9 ids) | Windows | **8 passed, 1 xfailed** | 70.3 s loaded / 97.7 s on the `--store-durations` run; slowest id **26.4 s** |
| the same | WSL | **8 passed, 1 xfailed** | 65.3 s quiet (116 .. 150 s under the mutation matrix's load); slowest id **13.0 s** |
| `test_audit2609_b12b_gbd_projection.py` (14 ids) | Windows | **14 passed** | 28.8 .. 49.1 s across eight mutation-matrix invocations |
| `test_audit2609_b12_fga_reference_plane.py` (14) + `test_verify_b12_fga_reference_plane.py` (4) = **the 18 WP-B12 pins** | Windows | **14 passed** + **4 passed** | 30.1 s + 16.2 s |
| the same 18 | WSL | **14 passed** + **4 passed** | 27.1 s + 8.7 s |
| `test_audit2609_b12b_gbd_projection.py` (14 ids) | WSL | **14 passed** | 36.2 .. 51.6 s |
| the eight RESTATED files (`test_audit2609_a4_fga_s10.py`, `test_audit2609_b7b_caustic_routing.py`, `test_fga.py`, `test_fga_h4_h5.py`, `test_g1_gate_generality.py`, `test_niche_audit_w9_dispatch2.py`, `test_niche_p8_capstone.py`, `test_niche_p7_seidel_gate.py`) -- 180 ids | Windows | **180 passed, 0 failed** | 1269.8 s |
| the same eight | WSL | **180 passed, 0 failed** | 640.4 s |
| the 17-file GBD + reference-plane selection (229 ids) | Windows | **228 passed, 1 xfailed, 0 failed** | 1961.6 s; slowest id 198.0 s (`test_niche_r3_gbd_mem_lstsq`, not mine) |
| the same | WSL | **228 passed, 1 xfailed, 0 failed** | 2112.8 s |
| the census / walker / dispatcher-pin / public-API / doc-consistency / history sweep + `test_audit_except_budget.py` (31 files, 1392 ids) | Windows | **1381 passed, 11 skipped, 0 failed** | 302.4 s |
| the mutation matrix, 8 arms x 4 files (32 cells) | Windows | section 8.2 | ~13 min |
| the same | WSL | section 8.2 -- **cell-for-cell identical to Windows** | ~24 min |

### 13.2 The other gates

| gate | result |
|---|---|
| `wsl ~/lumvenv/bin/python -m ruff check lumenairy/ tests/ validation/probe_verify_b12b/ validation/probe_gbd_projection/` (ruff 0.15.16) | **All checks passed** |
| Windows `python -m ruff check` on the same paths | **All checks passed** |
| `python scripts/record_history_fingerprints.py --check` | **OK: every history document matches its module** |
| `python scripts/check_source_line_citations.py` | **ok=107, drift=0, total=107** |
| `python scripts/check_doc_identifiers.py` | **OK: every API-claiming backticked identifier resolves** (621 distinct, 0 unresolved) |
| `.test_durations` | 16 214 -> **16 223** entries, valid JSON, **9** new ids, largest **26.36 s** |
| the builder's `probe_a_sag.py` / `probe_c_decompose.py`, re-run on this tree | both rewrote their committed JSON and `git status` stayed **clean** |

### 13.3 Probes and their outputs

All under `validation/probe_verify_b12b/`, each run with `PYTHONPATH` pinned
to the tree under test and `lumenairy.__file__` printed and asserted to live
under it, and with the arm DETECTED from the library rather than passed in.

| probe | what | outputs |
|---|---|---|
| `probe_w0_controls.py` | my tracer and my oracle against the library's own two vertex operators, plus the oracle's own convergence floor | `probe_w0_controls_{win32_314,linux_312}.json` |
| `probe_w1_mechanism.py` + `compare_w1.py` | the four failure modes at the BEAMLET level, PRE and POST, `z_image = 0` | `probe_w1_mechanism_{pre,post}_{win32_314,linux_312}.json`, `compare_w1_*.json` |
| `probe_w2_ladder.py` + `compare_w2.py` | the oracle ladder at two planes per fixture, the forced-`'surface'` arm, and the four entry points | `probe_w2_ladder_{pre,post}_*.json`, `compare_w2_*.json` |
| `probe_w3_openitems.py` | the mirror through both branches, the immersed exit, and the world branch's reference plane | `probe_w3_openitems_post_*.json`, `..._world_exit_vertex_*.json` |
| `probe_w4_frame.py` | the beamlet-frame ladder on the builder's OWN fixtures, scored with my oracle | `probe_w4_frame_post_*.json` |
| `probe_w5_entries.py` | which stage separates the three local entry points, and the memory-budget sensitivity of the digests | `probe_w5_entries_post_*.json` |
| `vb12b_mutate.py` + `run_mutations.sh` | the mutation matrix | `mutations_{win32_314,linux_312}.txt` |

The builder's own `validation/probe_gbd_projection/probe_a_sag.py` and
`probe_c_decompose.py` were re-run unchanged on this tree; both rewrote their
committed JSON and `git status` stayed clean, so those artefacts are
bit-reproducible on this build.

---

## 14. What I could not measure

1. **The CI cross-build spread.**  Both builds here are the same box.  The
   runner mix (EPYC 9V74 / 7763 with older wheels) was not sampled, so a bar
   that is comfortable on both of my builds could still be per-build on a
   third.  Every bar in my new file is derived at run time from a quantity
   the running build measures, which is the mitigation, not a substitute.
2. **The GPU reconstruction path** (`apply_real_lens_gbd(use_gpu=True)`).  No
   CUDA device here.  It moves the bundle to the device AFTER the NumPy
   per-surface evolution, so the repair is upstream of it, but that is an
   argument and not a measurement.
3. **The JAX branch of `_project_to_exit_vertex_plane` through GBD.**  It is
   unreachable by construction (section 6.4), so there is nothing to measure;
   WP-B12's own suite exercises it through the analytic primitive.
4. **The world branch on a NON-symmetric last surface.**  I scored it on a
   conic and an even asphere; a biconic / freeform / field-frame last surface
   through `world_output_plane` is not covered here, by either package.
5. **`reexpand='auto'`, `direction_sampling` and the vector GBD path.**
   `propagate_gbd_vector_through_prescription` was confirmed by reading to use
   `_fresnel_jones_matrix_per_beamlet` and never to call the differential
   primitives, so it is untouched; I did not run it.
6. **The builder's own non-symmetric peak movements** (+55 % biconic,
   +147 % freeform).  I measured MY fixtures against MY oracle rather than
   re-taking their peaks; my rows say the same thing with an accuracy number
   instead of a movement number.
7. **Whether D-4's refusal is the right product decision.**  I measured that
   the local branch is wrong for a mirror-terminated prescription and that the
   branch its open item names refuses that class; choosing between a refusal
   and a full signed-`N` local branch is a maintainer call, and I recommend
   the refusal on cost grounds, not on a measurement of the alternative.
