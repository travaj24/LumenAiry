# AUDIT_V5_4_5_2026_05_29_DEEP_FOLLOWUP

## Executive summary

This is a FOLLOW-UP deep audit of lumenairy v5.4.5, additive on top of
AUDIT_V5_4_5_2026_05_26_DEEP. Every finding below is NEW (not present in any
prior audit). The prior deep audit recorded 33 findings (P1=1, P2=4, P3=28),
all of which remain PENDING for the v5.4.6 patch cycle and are NOT repeated
here.

This pass ran 20 dimension finders over previously under-audited subsystems
plus a fresh physics re-sweep, then 3-lens adversarial verification. It
produced these confirmed NEW findings:

| Severity | Confirmed (NEW this pass) |
|----------|---------------------------|
| P1       | 3                         |
| P2       | 16                        |
| P3       | 21                        |
| **All**  | **40**                    |

**Health verdict.** v5.4.5 remains broadly sound on the canonical NumPy
physics path, but this follow-up pass surfaced three genuine ship-priority
defects that the prior pass did not reach: (1) the GBD reconstructed field
carries a conjugated (wrong-sign) transverse wavefront curvature after free
propagation, breaking coherent overlay with ASM/HFPI/Fresnel; (2)
compute_pupils crashes with UnboundLocalError on every internal-stop system
(the normal real-lens configuration), cascading into the ray-fan / OPD-fan
analysis; and (3) rayleigh_sommerfeld_propagate returns a view into the
pyFFTW ping-pong buffer, silently corrupting earlier results in a z-sweep.
Beyond these, the 16 P2s concentrate in optimize merit-mode semantics,
detector radiometry, tolerancing-Strehl normalization, io-prescription
sign/round-trip drift, the JAX lens-grid transpose, and a dead-code annular
aperture. The 21 P3s are predominantly docstring/API drift, backend dtype
parity, and narrow-regime physics guards.

**Confidence caveat.** The 3-lens verify had a LOW rejection rate this pass
(only 4 raw candidates were rejected). Findings therefore carry a confidence
tag drawn from the parent-agent auditor cross-check where one is available
(CONFIRMED-REPRO / CONFIRMED-READING / UNCERTAIN / REFUTED). Any
physics/convention sign-claim that has NOT been numerically reproduced should
be treated as PLAUSIBLE-UNVERIFIED pending a dedicated pinned repro before
remediation. The detector photon-loss magnitude in particular is flagged
UNCERTAIN by the auditor and must be reproduced before patching.

## Methodology

- **Finders.** 20 dimension finders were dispatched across subsystems that
  the prior deep pass either skipped or treated only at a one-line level
  (optimize, asymptotic propagators, the GBD/non-ASM propagator family, lens
  JAX twins, Seidel/paraxial, world/local trace, OPD/wavefront analysis,
  detector/field, coronagraph/AO, io-prescriptions, algebra, backend
  dispatch, FFT infra, numerical edge cases, glass/coatings, vector
  diffraction, convention cross-file, state/mutation/aliasing, and a fresh
  sources/core sweep), plus a fresh whole-library physics re-sweep.
- **Adversarial verify.** Every raw candidate was reviewed by 3 diverse
  lenses: (1) physics/math correctness, (2) contract-and-test-pin / novelty,
  (3) operational regime / impact.
- **Confirm bar.** A finding is confirmed only when it scores >= 2/3 "real"
  AND >= 2/3 "genuinely new" across the three lenses.
- **Static + numeric.** Findings were established by static analysis and, for
  the highest-impact items, targeted numerical reproductions captured in the
  per-file auditor cross-check notes (see "Auditor cross-check" lines).

## Findings

### [P1] F-1. GBD reconstructed field has conjugated (wrong-sign) transverse wavefront curvature vs exp(+ikz) convention

- **Location**: `lumenairy/propagators/gbd.py:193, 375, 377`
- **Category**: convention
- **Dimension**: propagator-family
- **Votes**: real 3/3, new 3/3
- **Auditor cross-check (gbd.py)**: CONFIRMED-REPRO. "Analytic + numerical
  repro (N=128, w0=24um, z=3mm~1.05 z_R): ASM radial dphase
  +0.222/+0.888/+1.998 rad at rho=16/32/48um vs GBD -0.216/-0.865/-1.946
  (sums ~0, magnitudes match) -> GBD transverse phase is the complex conjugate
  of the correct exp(+ikz) wavefront. Intensity/|q| unaffected so focus/width
  tests pass. P1 CONFIRMED."
- **What's wrong**: decompose_field_to_beamlets stores Q = -1j/z_R (line 193),
  i.e. it represents the complex beam parameter as q = +i*z_R (engineering
  exp(+i omega t) sign), and reconstruct_field_from_beamlets evaluates the
  transverse Gaussian profile as exp(-0.5j*k*Q*rho2) (line 377; line 375 for
  the has_dirs branch). The canonical Gaussian-beam field under the library's
  exp(-i omega t) / forward exp(+ikz) convention is exp(+i*k*rho2/(2q)) with
  q(0) = -i*z_R (so 1/q = +i/z_R). The code flips BOTH the q sign and the exp
  phase sign. The two flips cancel only for the source-plane amplitude (purely
  real Gaussian decay, which is why the decompose->reconstruct round trip and
  the |E|^2 envelope look fine). After free-space propagation,
  propagate_beamlets_freespace does Q_new = Q_old/(1 + t*Q_old) i.e. q += t
  with t REAL (sign-blind), so the resulting transverse phase
  exp(-i*k*rho2/(2q)) is the COMPLEX CONJUGATE of the physically-correct
  exp(+i*k*rho2/(2q)). Meanwhile apply_thin_lens_to_beamlets uses the
  canonical lens phase exp(-1j*k*r^2/(2f)) (line 291) and the canonical
  Q-update Q -= 1/f, and propagate_beamlets_freespace uses the canonical axial
  phase exp(+1j*k*t) (line 241) -- so the free-space transverse curvature is
  the ONLY conjugated term, making GBD internally inconsistent.
- **Why it matters**: Silent wrong physics on the mainstream NumPy free-space
  and through-lens GBD path. The intensity envelope is correct (it depends
  only on |q|, which conjugation does not change), so width/focus regression
  tests pass and the bug is invisible to them. But the complex FIELD phase is
  wrong-signed in its transverse curvature: any coherent overlay of a GBD
  output with an ASM / HFPI / Fresnel output (the explicit design goal of the
  v4.12.1 grid-unification fixes), any further coherent propagation of the GBD
  output, or any interferometric comparison will be off by a conjugated
  quadratic phase. HFPI uses exp(+ik*OPL) (hfpi.py:272) and Fresnel uses
  exp(+ik/(2z)*x^2) (fresnel.py:146) -- both canonical -- so GBD is the odd
  one out.
- **Regime**: Any free-space GBD propagation with z != 0 of a beam with finite
  curvature, then used for coherent superposition / interference / further
  propagation. Manifests whenever the reconstructed field's PHASE (not just
  intensity) is consumed.
- **Evidence**:
  ```
  # decompose (line 193):
  Q = xp.full((n,), -1j / z_R, ...)
  # reconstruct (line 377):
  phase = xp.exp(-0.5j * k * Q_b[None, None, :] * rho2)
  # (line 375): phase = xp.exp(1j * k * arg) with arg = (-0.5 * Q_b[...]*rho2 + ...)
  # canonical lens phase (line 291):
  lens_phase = xp.exp(-1j * k * (x_off*x_off + y_off*y_off) / (2 * focal_length))
  # canonical axial phase (line 241):
  axial_phase = xp.exp(1j * k * t)
  ```
- **Proposed fix**: Make GBD use the canonical exp(-i omega t)/exp(+ikz) sign
  throughout. Preferred: set Q = +1j/z_R in decompose_field_to_beamlets (line
  193) AND change the reconstruction transverse phase to exp(+0.5j*k*Q*rho2)
  (line 377 and the fused arg sign on line 373/375), so 1/q = +i/z_R and the
  propagated wavefront becomes exp(+i*k*rho2/(2q)); this aligns GBD with
  HFPI/Fresnel/ASM. After the fix, add a regression test that compares the GBD
  free-space output PHASE (not just |E|^2) to angular_spectrum_propagate on a
  Gaussian to within lambda/100, mirroring the existing intensity-centroid pin.

### [P1] F-2. compute_pupils never assigns ep_z for any non-front stop -> UnboundLocalError on all internal-stop systems

- **Location**: `lumenairy/raytrace/seidel.py:722-731`
- **Category**: numerical
- **Dimension**: raytrace-seidel-paraxial
- **Votes**: real 3/3, new 3/3
- **Auditor cross-check (seidel.py)**: CONFIRMED-REPRO. "compute_pupils(surfaces,
  wavelength) on a 3-surface system with is_stop on the middle surface raises
  UnboundLocalError: cannot access local variable ep_z. Lines 723 / 730 are
  bare orphaned expressions that never assign ep_z on the internal-stop branch.
  Every internal-stop system crashes; untested (all tests use front-stop
  singlets). P1 CONFIRMED."
- **What's wrong**: In the entrance-pupil branch taken whenever stop_index != 0,
  the two statements that should set ep_z are written as bare expressions whose
  results are discarded: line 723 is `-B_pre / A_pre` (not `ep_z = -B_pre /
  A_pre`) and line 730 is `float('inf')` (not `ep_z = float('inf')`). Only
  ep_radius is assigned. The return statement at line 779 then does
  `ep_z=float(ep_z)` referencing a never-bound local, raising UnboundLocalError.
  The stop_index==0 branch (line 697) is the ONLY branch that sets ep_z, so any
  system with the aperture stop anywhere except surface 0 fails.
- **Why it matters**: compute_pupils and its caller first_order_data are core
  public paraxial APIs; the entrance-pupil position ep_z feeds f/#, vignetting,
  the ray-fan/OPD-fan chief-ray launch, and image_plane_wfe geometry. An
  internal aperture stop (between elements) is the standard real-lens
  configuration, so the most common professional use case crashes hard with a
  confusing UnboundLocalError instead of returning the EP. It is also
  completely untested: every test that touches first_order_data/compute_pupils
  uses a stop-at-front singlet.
- **Regime**: Any multi-element prescription whose aperture stop is not surface
  0 (is_stop=True on an interior surface, or find_stop selecting the first
  finite-semi_diameter interior surface). Verified crash with a 2-element BK7
  system, stop on surface 1.
- **Evidence**:
  ```
  if abs(A_pre) > 1e-30:
      -B_pre / A_pre            # <- result discarded; ep_z never set
      ...
      ep_radius = abs(stop_radius / A_pre) ...
  else:
      float('inf')              # <- result discarded; ep_z never set
      ep_radius = float('inf')
  ... later ...
  return PupilInfo(
      ep_z=float(ep_z), ...)   # UnboundLocalError: ep_z not associated with a value
  ```
- **Proposed fix**: Assign the computed values: replace line 723 `-B_pre /
  A_pre` with `ep_z = -B_pre / A_pre` and line 730 `float('inf')` with `ep_z =
  float('inf')`. Add a regression test with an interior stop (e.g. is_stop=True
  on surface 1 of a doublet) asserting compute_pupils/first_order_data return
  finite ep_z.

### [P1] F-3. rayleigh_sommerfeld_propagate returns a view into the pyFFTW inverse ping-pong buffer, violating the _ifft2 double-buffer contract (silent corruption of earlier RS results)

- **Location**: `lumenairy/propagators/rs.py:328-337`
- **Category**: numerical
- **Dimension**: fft-infra-deep
- **Votes**: real 3/3, new 3/3
- **Auditor cross-check (rs.py)**: CONFIRMED-REPRO. "rayleigh_sommerfeld_propagate
  returns a NumPy view (A.base is not None). After 4 more same-grid RS calls
  the earlier result A changed by max|dA|=1.70 (whole field overwritten) ->
  earlier RS results are silently corrupted because the return aliases the
  reused _ifft2 ping-pong buffer. P1 CONFIRMED on a mainstream multi-distance
  RS sweep."
- **What's wrong**: On the NumPy path RS does `E_fft = _fft2(E_padded); E_conv =
  _ifft2(E_fft * H)` then `E_out = E_conv[y0:y0+Ny, x0:x0+Nx]; return E_out`.
  _ifft2 (fft_infra.py:1551-1586) returns one of the two cache-owned ping-pong
  workspace buffers, NOT a fresh array. E_conv[...] is a numpy slice (a VIEW) of
  that live buffer (np.shares_memory(view, E_conv) is True), and that view is
  returned to the caller with no .copy(). The fft_infra contract
  (fft_infra.py:1497-1502, pinned by test_perf_v4_12_0_fft_infra.py:120-176) is
  that a returned buffer stays valid across EXACTLY ONE subsequent same-key call
  and is recycled on the SECOND. The RS forward kernel H depends on z but the
  inverse FFT key (direction='inv', shape (Ny2,Nx2), complex128, threads) is
  identical for every z at a fixed grid, so consecutive RS calls recycle the
  same two inv-slots. Contrast: ASM (asm.py:386) and Fresnel (fresnel.py:153,161)
  wrap the dispatcher output in np.fft.fftshift or multiply by a prefactor, both
  of which materialize an independent copy; RS is the only propagator that
  slices the raw inverse buffer and returns the view.
- **Why it matters**: Any loop that collects RS outputs at the same grid -- a
  z-sweep / through-focus stack, an optimization inner loop, multi-distance
  reconstruction -- silently corrupts results[k] on iteration k+2. The first
  returned field becomes garbage (overwritten by a later iteration's inverse
  transform) with no error and no warning; downstream metrics (Strehl,
  centroid, energy) are computed on aliased data. This is silent wrong physics /
  data corruption on the mainstream NumPy+pyFFTW path for a public propagator.
- **Regime**: NumPy backend, pyFFTW installed (default), input N >= 128 so
  padded Ny2 = 2*N >= 256 = FFTW_MIN_SIZE (true for all realistic RS grids
  512/1024/2048), and rayleigh_sommerfeld_propagate called >=3 times at the same
  grid shape (z-sweep, optimization, batch). The corruption hits the result
  returned 2 calls earlier.
- **Evidence**:
  ```
  E_fft = _fft2(E_padded)
  E_conv = _ifft2(E_fft * H)
  ...
  E_out = E_conv[y0:y0 + Ny, x0:x0 + Nx]
  return E_out
  # E_conv is the cache-owned inverse buffer (fft_infra.py:1574-1578
  #   plan, buf, lock = _get_or_make_plan('inv', ...); ...; return buf);
  # the slice is a view sharing memory, returned without copy.
  # fft_infra.py:1560-1564 documents the hazard.
  ```
- **Proposed fix**: Return an independent copy of the cropped region in rs.py:
  `E_out = E_conv[y0:y0 + Ny, x0:x0 + Nx].copy()` on the NumPy path (the
  JAX/CuPy paths already return fresh arrays, so .copy() there is a cheap
  no-op). The per-call .copy() in RS is the minimal, contract-honoring fix and
  matches the kernel-H .copy() RS already does at rs.py:272.

### [P2] F-4. RMSWavefrontMerit default exclude_low_order=4 keeps defocus and drops oblique astigmatism (OSA off-by-one)

- **Location**: `lumenairy/optimize/merit_terms.py:154,166,176 (and context.py:281)`
- **Category**: convention
- **Dimension**: optimize-subsystem
- **Votes**: real 3/3, new 3/3
- **What's wrong**: The merit slices coeffs[exclude_low_order:] to drop the
  'first N' OSA-ordered Zernike modes. In OSA/ANSI ordering (zernike.py:10-18)
  the indices are 0=piston, 1=tilt-Y, 2=tilt-X, 3=oblique-astigmatism,
  4=defocus. So the default exclude_low_order=4 drops indices [0:4] = piston +
  2 tilts + OBLIQUE ASTIGMATISM, and KEEPS defocus (index 4). The docstring
  (lines 153-158) claims exclude_low_order=4 removes 'piston + 2 tilts +
  defocus' and that exclude_low_order=3 'keeps defocus' -- internally
  contradictory, since 3 modes is exactly piston+2 tilts, so one-more-mode (=4)
  excludes index 3 (oblique astig), never defocus. Defocus removal would
  require exclude_low_order=5.
- **Why it matters**: RMSWavefrontMerit is a mainstream public merit (exported
  in __init__.py:553). Its stated job is 'image-quality RMS after best-focus'
  i.e. defocus-insensitive. As written the default merit (a) double-penalizes
  residual best-focus error that the through-focus scan already optimized away,
  biasing the optimizer toward a focus position rather than aberration
  correction, and (b) silently excludes a genuine aberration (oblique
  astigmatism) from the image-quality figure of merit, understating real
  wavefront error. The standalone validation tests (test_optimize.py) pass
  exclude_low_order and only check monotonic ordering, so they never catch the
  mode mismatch.
- **Regime**: Any design_optimize run using the default RMSWavefrontMerit() with
  an astigmatic or defocus-sensitive wavefront (off-axis fields,
  tilted/decentered elements, real singlets with residual focus shift).
- **Evidence**:
  ```
  # Docstring: 'exclude the first exclude_low_order modes (default 4:
  #   piston + 2 tilts + defocus)' ... 'Set exclude_low_order=3 to keep
  #   defocus in the RMS'.
  rms_waves = ctx.rms_wavefront_waves(n_modes=self.n_modes,
                                      exclude_low_order=self.exclude_low_order)
  # context.py rms_wavefront_waves: higher = coeffs[exclude_low_order:].
  # With OSA ordering coeffs[4:] retains index-4 defocus.
  ```
- **Proposed fix**: Either change the default to exclude_low_order=5 (drop
  piston+2 tilts+oblique astig+defocus) to actually match 'image-quality after
  best-focus', or -- preferred -- stop relying on contiguous OSA slicing for
  physical mode groups and explicitly zero the (n,m) modes you intend to remove
  (piston (0,0), tilts (1,+-1), defocus (2,0)) via zernike_nm_to_index, leaving
  astigmatism in the RMS. Fix the docstring and the line-262 comment in
  MatchIdealThinLensMerit that repeats the same wrong piston/tilt/defocus claim.

### [P2] F-5. Driver wave-leg uses non-NaN-safe np.argmax for rms_radius_best while sibling wrapper merits use np.nanargmax

- **Location**: `lumenairy/optimize/driver.py:707`
- **Category**: numerical
- **Dimension**: optimize-subsystem
- **Votes**: real 3/3, new 2/3
- **What's wrong**: After find_best_focus (which is robust, using nanargmax
  internally), the driver picks the spot-radius slice with i_best =
  int(np.argmax(scan.strehl)). scan.strehl is initialized to all-NaN
  (through_focus.py:375) and individual z-slices can remain NaN when a per-plane
  metric fails. np.argmax treats NaN as the maximum (verified:
  np.argmax([nan,0.5,0.9]) -> 0), so a single failed slice makes i_best point at
  the NaN slice and ctx.rms_radius_best = scan.rms_radius[i_best] becomes that
  slice's (possibly NaN/garbage) value, and is taken at a different z than the
  strehl-best slice. The two sibling wrapper merits deliberately guard this exact
  case with np.any(np.isfinite(scan.strehl)) + np.nanargmax
  (wrapper_merits.py:449-451 and 701-704, both commented '4.11.1: nanargmax so a
  single NaN slice does not steal the argmax').
- **Why it matters**: This is the mainstream NumPy path that feeds SpotSizeMerit
  (a public exported merit). A NaN rms_radius_best produces NaN merit
  contributions that scipy then rejects or that derail the line search; a
  wrong-slice rms_radius silently mis-scores the design and decouples the
  reported spot radius from the reported best-focus Strehl. The fix was already
  applied to the wrapper merits but the primary driver leg was missed, so the
  regression the 4.11.1 comment guards against still bites the default on-axis
  path.
- **Regime**: Mid-optimization prescriptions that drive one or more
  through-focus z-planes to a degenerate field (zero/NaN peak, e.g. light fully
  clipped at an extreme step) while leaving others valid -- common during the
  optimizer's exploratory steps on real lenses.
- **Evidence**:
  ```
  z_best_v, strehl_best_v = _core.find_best_focus(scan, 'strehl')
  ...
  i_best = int(np.argmax(scan.strehl))
  ctx.rms_radius_best = float(scan.rms_radius[i_best])
  # Contrast wrapper_merits.py:
  #   if np.any(np.isfinite(scan.strehl)):
  #       i_best = int(np.nanargmax(scan.strehl))
  #       sub_rms = float(scan.rms_radius[i_best])
  ```
- **Proposed fix**: Mirror the wrapper-merit guard: gate on
  np.any(np.isfinite(scan.strehl)) and use np.nanargmax; otherwise leave
  ctx.rms_radius_best at its inf default. Better still, reuse the
  z_best/strehl_best already returned by find_best_focus by locating the matching
  z index rather than re-argmaxing a different array.

### [P2] F-6. aberration_tensor closed-form (ell=0) evaluates output LG polynomial at absolute image coord instead of saddle sigma-offset -> wrong off-axis defocus/spherical

- **Location**: `lumenairy/propagators/asymptotic_aberration_tensor.py:551-553`
- **Category**: physics
- **Dimension**: asymptotic-propagators
- **Votes**: real 3/3, new 3/3
- **What's wrong**: In the ell=0 closed-form chief-ray projection, out_const
  accumulates the output LG mode polynomial out_poly (built by
  lg_polynomial(p,ell,w_o), centered at the OUTPUT origin / chief image)
  evaluated at the ABSOLUTE image point (s2x_img, s2y_img): `out_const += c *
  (s2x_img**ii) * (s2y_img**jj)`. The output LG basis must be evaluated at the
  saddle's local sigma-coordinate (the offset from the chief image). When
  s2_image is the chief-ray landing, the saddle sigma is 0, so out_const should
  be the on-axis value N_{p,0}. By using absolute coordinates the code injects a
  factor ~(s2_img/w_o)^(2p+|ell|) for every p>=1 term. The is_axial_saddle test
  at line 493-494 (`abs(s2x_img)+abs(s2y_img) < 1e-9*max(w_o,...)`) confirms the
  conflation: it treats s2_image==0 as 'sigma_image=(0,0)', i.e. it assumes
  absolute image == sigma offset, which only holds when the chief image sits at
  the absolute origin.
- **Why it matters**: Every (p>=1, ell=0) entry of the returned aberration
  tensor L -- defocus (1,0), primary spherical (2,0), higher spherical (3,0) --
  is silently scaled by a large, position-dependent, physically meaningless
  factor at any off-axis field point. These are precisely the channels the
  module markets for design merit functions ('Driving |L_{(2,0),0}|^2 to zero
  suppresses on-axis spherical', docstring lines 89-93). An off-axis
  spherical/defocus merit term would optimise against corrupted numbers. The
  ell!=0 default output_modes mask the bug (they force the sigma-grid path), but
  a caller requesting purely ell=0 modes off-axis hits it.
- **Regime**: aberration_tensor called with output_modes restricted to ell=0
  (e.g. [(0,0),(1,0),(2,0)]) at an off-axis image point where the chief-ray
  landing s2_image is far (relative to w_o) from the absolute coordinate origin
  -- typical for any non-centered field point or off-axis design.
- **Evidence**:
  ```
  out_const = 0.0 + 0.0j
  for (ii, jj), c in out_poly.items():
      out_const += c * (s2x_img ** ii) * (s2y_img ** jj)
  T_acc += b_pup * out_const * exp_val
  # Numeric probe: lg_polynomial(1,0,w_o=5e-6) has term (2,0)=-1.28e16;
  # evaluated at s2x_img=0 -> 1.596e5 (=N_{1,0}); at s2x_img=0.5mm ->
  # -3.19e9, a 2.0e4x corruption.
  ```
- **Proposed fix**: Evaluate out_poly at the saddle's local sigma offset, not
  the absolute image coordinate. Since the closed-form path holds the saddle at
  the chief ray (sigma=0 when s2_image is the chief landing), out_const for an
  ell=0 mode reduces to the constant term N_{p,0} (all p degenerate at the
  origin). Either evaluate at (0,0) and emit the existing degeneracy warning for
  multi-p, or route multi-p / off-axis ell=0 sets through the sigma-grid path
  (as the comment at lines 480-489 already concedes for the on-axis multi-p
  case).

### [P2] F-7. JAX traced/Maslov wave grid uses indexing='ij' -> phase screen transposed vs E_in (y,x) layout and vs NumPy 'xy' reference for non-symmetric prescriptions

- **Location**: `lumenairy/elements/_lens_jax.py:538, 587, 765, 836`
- **Category**: jax-parity
- **Dimension**: lens-kernels-jax
- **Votes**: real 2/3, new 3/3
- **What's wrong**: apply_real_lens_traced_jax builds its wave-grid coordinates
  as `Xw, Yw = jnp.meshgrid(x_wave, x_wave, indexing='ij')` (line 538), so
  Xw[i,j]=x_wave[i] (varies along axis 0) and Yw[i,j]=x_wave[j] (axis 1). Newton
  inversion then solves Sx(xe,ye)=Xw, Sy(xe,ye)=Yw, where Sx/Sy are the exit
  physical-x/physical-y maps (the launch grid feeds make_jax_ray_state(x=h_x=
  Xs_in.ravel) with Xs_in also 'ij', so axis 0 == physical x). Hence
  opl_map[i,j] is the OPL at exit physical position (x=x_wave[i], y=x_wave[j]).
  But E_in follows the library image-like (y,x) convention: `Ny, Nx = E_in.shape`
  (line 452) means axis 0 = y (rows), axis 1 = x (cols), so E_in[i,j] sits at
  physical (y=x_wave[i], x=x_wave[j]). The final `E_out = E_in.astype(cdtype) *
  phase_screen` (line 587; line 836 for Maslov) therefore multiplies the field
  at physical (x_j, y_i) by the phase screen computed for physical (x_i, y_j) --
  a transpose. The NumPy twin apply_real_lens_traced instead builds its wave grid
  as `X, Y = np.meshgrid(x, x)` (default indexing='xy', _lens_traced.py:1472)
  giving X[r,c]=x[c], Y[r,c]=x[r], which is self-consistent with E_in's (y,x)
  layout, while keeping the launch grid at indexing='ij' (_lens_traced.py:1733).
  The JAX twins thus diverge from the NumPy reference whenever OPL(x,y) !=
  OPL(y,x).
- **Why it matters**: The module's CRITICAL contract is that the JAX twins match
  the NumPy lens kernels bit-for-bit. For any rotationally non-symmetric
  prescription -- cylindrical/biconic/toroidal singlets (make_cylindrical,
  make_biconic), or a spherical surface with per-surface decenter/tilt -- the
  JAX traced/Maslov output is the transpose of the physically correct field. A
  horizontal line focus comes out vertical; a +x-decentered focus comes out
  +y-decentered. Because this is the autodiff leg used for gradient-based lens
  design, the gradients backpropagated into upstream computations carry the
  transposed wavefront, silently steering an optimizer toward the wrong solution.
- **Regime**: Non-rotationally-symmetric prescription on a square (dy==dx) grid
  passed to apply_real_lens_traced_jax or apply_real_lens_maslov_jax:
  cylindrical/biconic/toroidal lenses, or any surface carrying a non-zero
  per-surface decenter/tilt. Invisible for the symmetric singlets/doublets used
  in every existing regression test.
- **Evidence**:
  ```
  # Line 452: Ny, Nx = E_in.shape   (axis0=y, axis1=x).
  # Line 492: Xs_in, Ys_in = jnp.meshgrid(xs_in, xs_in, indexing='ij')
  #   then make_jax_ray_state(x=h_x=Xs_in.ravel()) -> axis0 == physical x.
  # Line 537-538:
  x_wave = (jnp.arange(N) - N/2)*float(dx)
  Xw, Yw = jnp.meshgrid(x_wave, x_wave, indexing='ij')
  # Line 587: E_out = E_in.astype(cdtype) * phase_screen
  # Compare NumPy reference _lens_traced.py:1472 X, Y = np.meshgrid(x, x) (default 'xy').
  ```
- **Proposed fix**: Switch both wave grids to the NumPy convention: `Xw, Yw =
  jnp.meshgrid(x_wave, x_wave, indexing='xy')` at _lens_jax.py:538 and 765, so
  Xw varies along axis 1 (cols=x) and Yw along axis 0 (rows=y), matching
  E_in[y,x] and the NumPy reference. (Equivalently, keep 'ij' but transpose
  opl_map/valid before the multiply.) Add a regression test that runs
  apply_real_lens_traced_jax on a make_cylindrical / decentered prescription and
  asserts agreement with apply_real_lens_traced to lambda/20.

### [P2] F-8. ray_fan_data / opd_fan_data call make_ray with a phantom z argument, feeding ep_z into the x-direction cosine; except clause also misses UnboundLocalError

- **Location**: `lumenairy/raytrace/ray_fan.py:242-244 and 429-431`
- **Category**: api-drift
- **Dimension**: raytrace-seidel-paraxial
- **Votes**: real 3/3, new 3/3
- **What's wrong**: The 4.11.2 fix intends to launch the chief ray from the
  entrance-pupil centre at z=ep_z: `chief = make_ray(0, ep_y, fod.ep_z,
  np.sin(field_angle), wavelength=...)`. But make_ray's signature is make_ray(x=0,
  y=0, L=0, M=0, *, wavelength) -- there is NO z parameter. So the 4th positional
  fod.ep_z is bound to L (the x-direction cosine) and np.sin(field_angle) to M.
  The chief therefore starts at z=0 (not ep_z) with a nonsensical, un-normalised
  L = ep_z (tens of mm as a direction cosine). The intended EP-centred launch is
  silently defeated. Separately, the except tuple (line 245-246) catches
  ValueError/RuntimeError/ZeroDivisionError/AttributeError/LinAlgError/IndexError
  but NOT UnboundLocalError (a NameError subclass), so when the upstream
  compute_pupils bug fires for an internal stop, the fallback chief on line 252 is
  never reached -- ray_fan_data/opd_fan_data crash outright.
- **Why it matters**: For front-stop systems the bug is currently invisible only
  by accident: ep_z==0 makes L=0 and ep_y=0, collapsing to the correct
  origin-launched chief. The moment compute_pupils is fixed (so internal-stop
  systems return a real ep_z != 0), the chief reference for the transverse/OPD
  ray fans will be launched with a garbage x-tilt, corrupting y_ref/x_ref and
  opd_ref and thus every off-axis ray-fan and OPD-fan curve. Right now the same
  internal-stop systems instead crash because the except clause cannot catch the
  UnboundLocalError. Either way, off-axis ray-fan/OPD-fan analysis on real
  internal-stop lenses is broken.
- **Regime**: Off-axis (field_angle != 0) ray-fan or OPD-fan evaluation. Crash on
  internal-stop systems today; silently wrong chief on any system with ep_z != 0
  once the compute_pupils bug is fixed.
- **Evidence**:
  ```
  chief = make_ray(0, ep_y, fod.ep_z,
                   np.sin(field_angle),
                   wavelength=wavelength)
  # make_ray(x, y, L, M, *, wavelength) -- no z; ep_z lands in L
  except (ValueError, RuntimeError, ZeroDivisionError, AttributeError,
          np.linalg.LinAlgError, IndexError):   # UnboundLocalError NOT caught
  ```
- **Proposed fix**: Use make_ray's keyword args and set z explicitly: build the
  bundle via _make_bundle and assign bundle.z = ep_z (as
  test_audit_analysis.py:560 does), or add a z kwarg to make_ray. Concretely:
  `chief = make_ray(0, ep_y, 0.0, np.sin(field_angle), wavelength=wavelength);
  chief.z = np.array([fod.ep_z])`. Also add UnboundLocalError (or NameError) to
  the except tuple so the fallback chief actually triggers, and fix the root
  compute_pupils bug.

### [P2] F-9. Local coord-break tilt has inverted sign vs the world-frame build and the Optiland/Zemax surface-frame convention

- **Location**: `lumenairy/raytrace/intersection.py:454-514`
- **Category**: convention
- **Dimension**: raytrace-trace-world-bundles
- **Votes**: real 2/3, new 3/3
- **What's wrong**: _apply_coord_break (used by the local trace() engine for
  is_coordbrk=True surfaces) expresses the ray in the post-break local frame by
  applying a positive-angle math rotation to the ray vector: _rot_x(tx) does
  `y'=c*y-s*z; z'=s*y+c*z` (= Rx_math(+tx)) and _tilts() composes
  Rz(+tz)@Ry(+ty)@Rx(+tx). But the world->local transform for a coord-break whose
  surface frame is R_surface->world = Rx(tx)@Ry(ty)@Rz(tz) must be R^T =
  Rz(-tz)@Ry(-ty)@Rx(-tx). The local path therefore applies the WRONG-SIGN (and,
  when two tilts are nonzero, wrong-ORDER) rotation. The 3.7.1 comment (lines
  457-464) claims switching from Rx_math(-theta) to Rx_math(+theta) made the 2D
  layout agree with the 3D layout, but at v5.4.5 HEAD they disagree. The module
  docstring at intersection.py:426-427 even states the intended transform is
  Rx(-tx)@Ry(-ty)@Rz(-tz) -- which the code does NOT do. Numerically confirmed:
  for a +10 deg tilt-X coord-break applied to an axial world ray, the local path
  returns direction (0, -0.173648, 0.984808) while world.py + trace_world return
  (0, +0.173648, 0.984808); pure decenter agrees in both.
- **Why it matters**: Every imported folded .zmx design with a tilted COORDBRK
  traced through the local trace() engine bends the optical axis the WRONG WAY.
  The UI exposes both renderings from the same elem.tilt_x data:
  _build_trace_surfaces emits is_coordbrk=True Surfaces for the local trace() (2D
  layout) at model.py:2091, while _build_trace_surfaces_world feeds trace_world
  (3D layout). The two layouts will show the fold in opposite directions, and any
  spot/wavefront/footprint computed from the local trace of a tilted system is
  reflected about the fold plane. The world path matches _lens_real.py (lines
  646/661: R = Rx(tx)@Ry(ty), world->surface R^T = Ry(-ty)@Rx(-tx)), which is
  pinned by test_v5_2_off_axis_conic_surface_frame.py against Optiland/Zemax --
  so the local path is the defective one.
- **Regime**: Folded / tilted designs (any nonzero COORDBRK tilt_x/tilt_y/tilt_z)
  traced via the local sequential trace() engine; e.g. periscopes, fold mirrors,
  off-axis catadioptric systems loaded from .zmx. Manifests only when a tilt is
  present (decenter-only coord-breaks are unaffected).
- **Evidence**:
  ```
  # _rot_x:
  c, s = np.cos(theta), np.sin(theta)
  y_n =  c * rays.y - s * rays.z
  z_n =  s * rays.y + c * rays.z   # = Rx_math(+theta) applied to the ray
  # Compare world.py:_apply_coord_break
  #   tilt_R = _rot_x(tx) @ _rot_y(ty) @ _rot_z(tz); new_R = R @ tilt_R
  #   (surface->world = Rx(+tx)...), whose world->local is the TRANSPOSE.
  # Reproducer: LOCAL coord-break ray dir M=-0.17364817766693033;
  #   WORLD coord-break axial-world ray in new local frame M=+0.17364818.
  # Docstring at intersection.py:426-427 says Rx(-tx)@Ry(-ty)@Rz(-tz).
  ```
- **Proposed fix**: Negate the rotation angles in _apply_coord_break's _tilts()
  so the world->local transform is the transpose of world.py's tilt_R. Concretely
  apply _rot_x(-tx); _rot_y(-ty); _rot_z(-tz) (or, to also match the order,
  reverse to _rot_z(-tz); _rot_y(-ty); _rot_x(-tx) so the net ray transform is
  Rz(-tz)@Ry(-ty)@Rx(-tx) = tilt_R^T). Add a regression test that traces an axial
  ray through a tilt-X coord-break in both trace() (is_coordbrk surface) and
  trace_world() and asserts the resulting world-frame ray directions are
  identical for nonzero tilt; also align the docstring at lines 426-427 and
  457-464 with the corrected sign.

### [P2] F-10. apply_detector non-integer pixel ratio loses 25-30% of photons via spurious box-mean rescale

- **Location**: `lumenairy/analysis/detector.py:227-229`
- **Category**: physics
- **Dimension**: analysis-detector-field
- **Votes**: real 3/3, new 3/3
- **Auditor cross-check (detector.py)**: UNCERTAIN. "Auditor could not confirm the
  25-30% photon-loss magnitude with a quick uniform-field test (returned
  degenerate 0 electrons; test not sensitive). The code comment at lines 209-226
  itself acknowledges a box-filter over/under-count of ~20% for non-integer
  ratios, so the finding is PLAUSIBLE but the exact magnitude is UNVERIFIED.
  Needs a proper conservation repro before remediation."
- **What's wrong**: In the non-integer pixel_pitch/dx_field branch the
  per-detector-pixel signal is computed as `image = avg[IY, IX] * (pixel_pitch
  ** 2) * (scale_y * scale_x)` where `scale = samples_per_pix / win` and `win =
  round(samples_per_pix)`. avg is a scipy uniform_filter BOX MEAN. For a uniform
  (or smooth) field the box mean equals the field value INDEPENDENT of the window
  size, so the correct integral over one detector pixel is simply
  `avg * pixel_pitch**2`. Multiplying by `scale_y*scale_x = (samples_per_pix/
  win)^2` (which is <1 when win rounds up, e.g. 0.73 for ratio 2.5, 0.72 for
  ratio 1.7) systematically under-/over-counts. The lengthy comment at 209-226
  reasons that avg*(samples/win) 're-normalises the box mean back to the true
  samples-per-pixel', but the box mean is not window-size dependent, so this
  factor is the bug, not the fix.
- **Why it matters**: Breaks absolute radiometry and shot-noise calibration: a
  uniform input field that should integrate to sum(|E|^2)*dx^2 instead returns
  ~69-72% of that on non-integer ratios. Because shot noise is Poisson(signal), a
  30% signal deficit also corrupts the noise statistics the detector model is
  built to provide. The docstring (lines 53, 108-110) explicitly promises photon
  counts when the field is normalised to photons, and v4.10/v4.11.2 comments
  claim photon conservation was fixed.
- **Regime**: Any pixel_pitch/dx_field ratio that is not an integer -- the common
  real case (detector pitch is rarely an exact integer multiple of the wave-sim
  grid). Verified: ratio 2.5 -> total 0.694x expected; ratio 1.7 -> 0.722x.
  Integer ratios (2.0, 3.0) are exact. The lone validation test
  t_detector_total_signal passes only because it forces n_pixels=30 with N=128 so
  that the scale^2=1.137 overshoot coincidentally cancels the 0.879 edge-area
  loss (and the test bound is a loose 0.5<ratio<2.0).
- **Evidence**:
  ```
  scale_y = float(samples_per_pix_y) / float(win_y)
  scale_x = float(samples_per_pix_x) / float(win_x)
  image = avg[IY, IX] * (pixel_pitch ** 2) * (scale_y * scale_x)
  # Numeric check (uniform field, dx=1um, pp=2.5um): img.sum()/expected = 0.6944;
  #   pp=1.7um: 0.7225
  ```
- **Proposed fix**: Drop the spurious factor: `image = avg[IY, IX] * (pixel_pitch
  ** 2)`. The box mean already approximates the per-pixel mean intensity over the
  detector-pixel-sized neighbourhood; multiplying by the true detector-pixel area
  pixel_pitch^2 yields the integrated signal. Add a photon-conservation
  regression test on a uniform field for a non-integer ratio (e.g. 2.5) asserting
  sum within ~edge-loss of expected. NOTE: confirm the magnitude with a proper
  conservation repro (auditor flagged UNCERTAIN) before shipping the change.

### [P2] F-11. ImagePlaneWFE.rms_waves / .strehl use un-piston-removed RMS, biasing Marechal Strehl low

- **Location**: `lumenairy/analysis/image_plane_wfe.py:130-146`
- **Category**: physics
- **Dimension**: analysis-detector-field
- **Votes**: real 3/3, new 3/3
- **What's wrong**: rms_waves returns sqrt(mean(v**2)) over the alive OPD samples,
  and .strehl feeds that directly into the Marechal approximation
  exp(-(2*pi*RMS)^2). The OPD opd_w is CHIEF-RAY referenced (line 175 `rs_w -
  rs_w[chief_idx]`, line 486 `opd_a_w = (opl - opl_chief)/wavelength`), not
  mean/piston referenced, so its mean over the pupil is generally nonzero.
  Marechal requires sigma = RMS deviation about the MEAN wavefront (piston does
  not affect image quality); using the chief-referenced raw RMS double-counts the
  piston term.
- **Why it matters**: Strehl is reported too pessimistically for any aberration
  with nonzero mean OPD (defocus, spherical, field curvature). The class
  advertises .strehl as the diffraction quality metric; a wrong Strehl misranks
  designs and tolerancing sweeps. The rms_waves docstring (lines 120-128) only
  flags the non-uniform-grid weighting caveat, never the missing piston removal,
  and .strehl does not route through the existing remove_low_order_aberrations.
- **Regime**: Any chief-referenced wavefront with nonzero mean OPD on a uniform
  pupil grid -- e.g. pure defocus. Verified: 0.1-wave defocus gives uncentered
  RMS 0.0576 -> Strehl 0.877, vs piston-removed RMS 0.0288 -> Strehl 0.968 (a
  ~9% Strehl error, RMS off by 2x).
- **Evidence**:
  ```
  v = self.opd_w[self.alive]
  v = v[np.isfinite(v)]
  return float(np.sqrt(np.mean(v ** 2))) if v.size else float('nan')
  ...
  sigma = 2.0 * np.pi * rms
  return float(np.exp(-(sigma ** 2)))
  ```
- **Proposed fix**: Subtract the (optionally aperture-weighted) mean before
  squaring in rms_waves: `v = v - v.mean(); return sqrt(mean(v**2))`, OR have
  .strehl compute its RMS on piston-removed OPD. Document that rms_waves is the
  wavefront-error RMS about the mean (piston-removed), consistent with the
  Marechal definition.

### [P2] F-12. tolerancing_sweep recomputes Strehl denominator from each perturbed pupil (the v5.2.5 nominal-pupil bug, fixed in MC but missed here)

- **Location**: `lumenairy/analysis/through_focus.py:776-798, 823`
- **Category**: physics
- **Dimension**: analysis-coronagraph-ao
- **Votes**: real 3/3, new 3/3
- **What's wrong**: Inside tolerancing_sweep._run_one the diffraction-limited
  Strehl denominator is recomputed from the *perturbed* exit pupil: `E_exit =
  apply_real_lens(E_source, prescription=pres_used, ...)` then `ideal_peak =
  diffraction_limited_peak(E_exit, wavelength, focal_length, dx)` (lines
  777-781). The Strehl for each perturbation is therefore normalized by its OWN
  perturbed reference, and the reported penalty `delta_strehl = r['strehl_peak'] -
  nominal['strehl_peak']` (line 823) subtracts two Strehls computed against
  DIFFERENT denominators. This is precisely the defect v5.2.5 fixed in
  monte_carlo_tolerancing, whose code comment (lines 917-924) states: 'ideal_peak
  must be derived from the UNPERTURBED nominal exit pupil and held fixed across
  all trials -- otherwise a perturbed pupil whose amplitude happens to better
  match the diffraction-limited reference can yield Strehl > 1 ... Previously
  ideal_peak was recomputed inside the loop from the per-trial perturbed E_exit,
  which is what produced the non-physical mean Strehl ~1.09.' tolerancing_sweep
  was never patched.
- **Why it matters**: tolerancing_sweep is a public, mainstream NumPy API (in
  __all__) and the deterministic counterpart of monte_carlo_tolerancing. A
  decenter/tilt/form-error perturbation that redistributes pupil amplitude can
  push its self-referenced Strehl above 1.0 or distort the penalty so a
  more-aberrated configuration appears LESS sensitive, corrupting the per-DOF
  sensitivity ranking that is the entire purpose of the sweep.
- **Regime**: Any tolerancing_sweep call with non-trivial perturbations
  (decenter/tilt/form_error) whose exit-pupil amplitude differs from nominal --
  i.e. essentially every real tolerancing run.
- **Evidence**:
  ```
  def _run_one(pres_used, label, inner_scaler=None):
      E_exit = apply_real_lens(
          E_source, prescription=pres_used, wavelength=wavelength, dx=dx,
          bandlimit=True, slant_correction=True)
      ideal_peak = diffraction_limited_peak(
          E_exit, wavelength, focal_length, dx)   # <-- per-perturbation denominator
      ...
      r['delta_strehl'] = r['strehl_peak'] - nominal['strehl_peak']
  # vs the fixed MC twin (lines 925-929):
  #   E_exit_nominal = apply_real_lens(... prescription=prescription ...);
  #   ideal_peak = diffraction_limited_peak(E_exit_nominal, ...) computed ONCE.
  ```
- **Proposed fix**: Compute ideal_peak ONCE from the nominal (unperturbed) exit
  pupil before the perturbation loop and pass that fixed value into _run_one for
  every perturbation, exactly as monte_carlo_tolerancing now does. Drop the
  per-call diffraction_limited_peak recompute inside _run_one (or gate it so it is
  only used for the nominal entry).

### [P2] F-13. phase_shift_extract advertises 'least-squares' but uses the equispaced-only correlation estimator, biasing phase for arbitrary shifts

- **Location**: `lumenairy/analysis/interferometry.py:128-145`
- **Category**: numerical
- **Dimension**: analysis-coronagraph-ao
- **Votes**: real 3/3, new 3/3
- **What's wrong**: The function accepts an arbitrary shifts sequence and the
  docstring (line 128) calls the method 'Least-squares extraction: phase =
  atan2(sum(I*sin), sum(I*cos))'. That estimator is the true least-squares
  solution to I = a + b cos(phi - s) only when the design matrix is orthogonal,
  i.e. sum(sin s) = sum(cos s) = 0, sum(sin^2 s) = sum(cos^2 s), and sum(sin s
  cos s) = 0 -- which holds for equispaced shifts over a full 2*pi period (the
  default) but NOT for arbitrary user shifts. For non-equispaced shifts the
  correct LSQ requires solving the 3-parameter normal equations for (a, b cos
  phi, b sin phi); the naive correlation is biased.
- **Why it matters**: A user feeding frames acquired with a real piezo whose
  steps are unequal, or a deliberately non-uniform N-step algorithm (e.g.
  unequally-spaced calibration sets), gets a systematically wrong wrapped phase /
  OPD with no warning, while believing they invoked a least-squares estimator
  robust to their shift choice.
- **Regime**: phase_shift_extract called with a custom shifts array that is not
  equispaced over a full period (real-hardware piezo calibration, asymmetric
  N-step schemes).
- **Evidence**:
  ```
  for f, s in zip(frames, shifts):
      sin_sum += f * np.sin(s)
      cos_sum += f * np.cos(s)
  if convention == 'hardware':
      phase = np.arctan2(sin_sum, cos_sum)
  elif convention == 'library':
      phase = np.arctan2(-sin_sum, cos_sum)
  # no orthogonality / equispacing check on shifts, yet docstring claims
  # 'Least-squares extraction'.
  ```
- **Proposed fix**: Either (a) implement the genuine least-squares solve
  (per-pixel) of the (1, cos s, sin s) design matrix so arbitrary shifts are
  handled correctly, or (b) restrict the docstring to state the estimator is
  exact only for equispaced full-period shifts and raise/warn when the supplied
  shifts fail the orthogonality conditions (e.g. |sum(exp(i s))| above a small
  tolerance, or unequal spacing).

### [P2] F-14. Codegen feeds Welford-signed mirror radius into apply_mirror (opposite wave-side convention) -> curved fold mirrors get inverted focusing sign

- **Location**: `lumenairy/io/codegen.py:361, 758-769`
- **Category**: convention
- **Dimension**: io-prescriptions
- **Votes**: real 3/3, new 3/3
- **What's wrong**: _decompose_prescription copies the loaded mirror radius
  verbatim ('radius': elem['radius'], line 361). load_zemax_zmx stores this radius
  as raw Zemax curvature inverted (radius = (1.0/curv)*unit_scale,
  prescriptions_zemax.py:316), i.e. the Welford signed-R convention (the same
  dataclass convention used by the traced Surface path). _generate_unrolled then
  emits `E = la.apply_mirror(E, WAVELENGTH, dx, radius={r_str}, conic=..., ...)`
  with r_str = that loaded radius (lines 759-769). But apply_mirror documents
  (elements.py:60-73) and uses the OPPOSITE 'wave-side' convention where R>0 =
  concave (focusing) -- explicitly noted as 'the OPPOSITE sign convention from the
  Welford signed-R used by system_abcd'. So a concave (focusing) fold mirror
  loaded from Zemax (Welford R<0) is emitted into apply_mirror as a negative R,
  which apply_mirror interprets as a convex (diverging) mirror -- and vice versa.
- **Why it matters**: The generated simulation script silently models every
  curved fold mirror with the wrong sign of focusing phase: a concave collimating
  mirror becomes diverging. This corrupts the OPD/focal geometry of the whole
  folded leg with no warning. Flat fold mirrors (the common case, radius=inf ->
  r_str='None') are unaffected, so the bug hides until a powered fold mirror
  appears.
- **Regime**: load_zemax_zmx of a folded design containing a curved (powered) fold
  mirror, then generate_simulation_script -> run the emitted script.
- **Evidence**:
  ```
  # codegen.py:361
  'radius': elem['radius'],
  # codegen.py:759-769
  r = step['radius'] ...
  E = la.apply_mirror(E, WAVELENGTH, dx, radius={r_str}, conic={conic}, ...)
  # vs elements.py:65-69
  #   this is the wave-side radius convention ... It is the OPPOSITE sign
  #   convention from the Welford signed-R.
  # prescriptions_zemax.py:316 stores the Welford-signed radius the
  #   Surface/seidel path consumes.
  ```
- **Proposed fix**: When emitting a mirror step (or in _decompose_prescription),
  convert the loaded Welford-signed mirror radius to apply_mirror's wave-side
  convention by negating it: pass radius=-elem['radius'] (leaving inf/None
  unchanged). Equivalently, route mirrors through apply_real_lens_traced/Surface
  (Welford) rather than apply_mirror. Add a regression test that loads a
  concave-mirror .zmx and asserts the emitted apply_mirror radius has the focusing
  sign (R>0 concave).

### [P2] F-15. split_prescription_at_mirrors silently drops surface<->mirror propagation distances from all_thicknesses

- **Location**: `lumenairy/io/prescriptions_transforms.py:407-423`
- **Category**: api-drift
- **Dimension**: io-prescriptions
- **Votes**: real 3/3, new 3/3
- **What's wrong**: The split loop only appends a thickness for a refractive
  surface when last_was_surface is True AND the prior element was a surface in the
  same run (line 416-419: `if last_was_surface and idx > 0:
  seg_thicknesses.append(all_th[idx-1])`). On a mirror it does
  _flush_refractive() then sets last_was_surface=False (line 413). Consequently:
  (a) the gap LEADING INTO a mirror (all_th[idx_mirror-1], the last-surface->mirror
  distance) is never appended -- the surface before the mirror was the last in
  its run and contributes no trailing thickness; (b) the gap LEAVING a mirror
  (mirror->next surface) is skipped because last_was_surface is False at the first
  surface of the next run. The returned mirror leg dict
  {'kind':'mirror','element':...} carries no thickness, and the refractive legs
  only retain intra-run inter-surface gaps.
- **Why it matters**: The helper's stated purpose (docstring) is to let a caller
  alternate apply_real_lens(each refractive leg) with apply_mirror(each fold)
  'keeping the physics explicit'. But the propagation distances between a lens
  group and the following/preceding fold mirror -- the exact free-space legs the
  caller must propagate -- are silently absent from the output, so the folded
  geometry cannot be reconstructed from the returned legs. This is verified by the
  test fixture comment 'the second 50 mm thickness ... is folded out', and
  test_folded_design_guard.py never checks leg thicknesses, so the loss is
  unpinned.
- **Regime**: Any folded prescription (load_zemax_zmx with a mirror element)
  passed to split_prescription_at_mirrors, then walked per the documented
  apply_real_lens/apply_mirror idiom.
- **Evidence**:
  ```
  # prescriptions_transforms.py:409-421
  if kind == 'mirror':
      _flush_refractive(); legs.append(...); last_was_surface = False; continue
  ...
  if last_was_surface and idx > 0:
      seg_thicknesses.append(float(all_th[idx - 1]))
  # For elements [surf,surf,mirror] with all_th=[d,gap], gap (surf->mirror)
  # is never emitted.
  ```
- **Proposed fix**: Attach the leading/trailing free-space distance to each leg:
  e.g. add a 'gap_after' (=all_th[idx]) to each mirror leg and a
  'gap_before'/'gap_after' to refractive legs, or emit explicit
  {'kind':'propagate','z':all_th[idx]} legs between optical legs so the caller can
  rebuild the chain. Add a test asserting the sum of emitted gaps equals
  sum(all_thicknesses).

### [P2] F-16. Aperture(shape='annular') default inner-diameter D/2 branch is dead code; default annular aperture has no central obstruction

- **Location**: `lumenairy/algebra/apertures.py:88-92`
- **Category**: numerical
- **Dimension**: algebra-subsystem
- **Votes**: real 3/3, new 3/3
- **Auditor cross-check (apertures.py)**: CONFIRMED-REPRO.
  "Aperture(diameter=10e-3, shape=\"annular\") returns inner_diameter=0.0
  (docstring promises D/2). The default-inner branch sits inside guard
  if inner<0 or inner>=D, which is False for inner==0.0 -> unreachable.
  Default annular aperture has no central obstruction. CONFIRMED dead code."
- **What's wrong**: The class docstring (lines 40-42) states: "'annular' -- ring
  with outer diameter diameter. The inner diameter defaults to diameter/2; pass
  inner_diameter=... to override." The code that is supposed to apply that default
  is unreachable. With the default inner_diameter=0.0, line 88 evaluates `if inner
  < 0.0 or inner >= D:` which is `0.0 < 0.0 (False) or 0.0 >= D (False, since
  D>0)` => the whole block is skipped, so inner stays 0.0. The nested branch at
  line 89 `if shape == 'annular' and inner == 0.0: inner = D/2.0` can NEVER
  execute, because reaching it requires the line-88 guard to be True, but the only
  way line 88 is True with inner==0.0 is impossible (0 is neither <0 nor >=D).
  Consequently `Aperture(D, shape='annular')` produces inner_diameter=0.0, which
  apply_aperture turns into mask `(h_sq >= 0) & (h_sq <= (D/2)**2)` = a FULL DISK
  with no central obstruction -- not the documented ring with inner radius D/4.
- **Why it matters**: A user who writes the documented short form `Aperture(D,
  shape='annular')` (relying on the D/2 default) silently gets a clear circular
  aperture instead of an annulus. Annular pupils are used for central-obstruction
  (Cassegrain-like) PSF/MTF studies and coronagraph-adjacent work; running such an
  analysis with the obstruction silently removed produces a wrong Airy/encircled-
  energy result with no error or warning. This is doc-vs-behavior divergence AND a
  silent wrong-physics mask.
- **Regime**: Any call `Aperture(diameter=D, shape='annular')` that omits
  inner_diameter (the documented default path). Explicit valid inner (e.g.
  inner_diameter=3e-3 with D=10e-3) works correctly; only the default is broken.
- **Evidence**:
  ```
  if inner < 0.0 or inner >= D:
      if shape == 'annular' and inner == 0.0:
          # Default inner = D / 2 for annular if caller did not supply one.
          inner = D / 2.0
      elif inner != 0.0:
          raise ValueError(...)
  # with default inner_diameter=0.0, the line-88 guard is False, so the D/2
  # default never runs; inner stays 0.0 -> annular mask is a full disk.
  ```
- **Proposed fix**: Move the annular default out of the validation guard. e.g.:
  ```
  if shape == 'annular' and inner == 0.0:
      inner = D / 2.0
  elif inner < 0.0 or inner >= D:
      raise ValueError(f"Aperture: inner_diameter must be in [0, diameter), got inner={inner}, diameter={D}.")
  ```
  This makes the documented D/2 default actually apply while preserving rejection
  of out-of-range explicit values. Add a regression test asserting Aperture(10e-3,
  shape='annular').inner_diameter == 5e-3 and that the resulting field is zeroed
  inside r < D/4.

### [P2] F-17. lumenairy_context applies new globals outside the try/finally, leaking partial state on entry failure

- **Location**: `lumenairy/_context.py:238-244`
- **Category**: state-mutation
- **Dimension**: state-mutation-aliasing
- **Votes**: real 3/3, new 2/3
- **What's wrong**: apply_globals(new_state) runs at line 238, BEFORE the try:
  block opens at line 239. apply_globals mutates process-global knobs in a fixed
  order (complex_dtype first at 118-121, then pyfftw_planner at 122-124,
  fft_threads, max_ram, asm_cache_size). If any LATER setter raises during entry,
  the EARLIER setters have already mutated global state, but because execution
  never reached the try, the `finally: apply_globals(prior)` restore at 242-243
  never runs. set_pyfftw_planner (fft_infra.py:621) raises ValueError on any
  non-FFTW_* string, so a single typo'd kwarg makes the global default complex
  dtype leak past the with-block.
- **Why it matters**: The entire purpose of lumenairy_context is to scope global
  runtime state to a with block so it cannot leak into the rest of a session (the
  module docstring's stated raison d'etre). A leaked complex_dtype=complex64
  silently changes the dtype of every subsequent source factory and internal
  scratch buffer for the remainder of the process, corrupting all downstream
  physics with no error. The existing regression tests (test_context_manager.py:
  123-136) only exercise exceptions raised INSIDE the with body, after entry
  succeeds -- the entry-time path is unpinned.
- **Regime**: Mainstream NumPy path. Triggered whenever a caller passes one valid
  knob plus one invalid knob whose setter raises, e.g.
  `lumenairy_context(complex_dtype=np.complex64, pyfftw_planner='MEASURE')` (typo
  for 'FFTW_MEASURE'). Verified live: after the with-block raised ValueError on
  entry, get_default_complex_dtype() returned complex64 instead of the prior
  complex128.
- **Evidence**:
  ```
  apply_globals(new_state)
  try:
      yield
  finally:
      try:
          apply_globals(prior)   # only runs if line 238 succeeded
      finally:
          ...
  # Live repro:
  # prior dtype: complex128
  # entry raised ValueError: set_pyfftw_planner: planner must be one of ['FFTW_...
  # after with-block dtype: complex64
  # LEAKED? True
  ```
- **Proposed fix**: Move apply_globals(new_state) inside the try block (or wrap
  entry in its own try that restores prior on failure):
  ```
  prior = snapshot_globals()
  ...build new_state...
  try:
      apply_globals(new_state)
      yield
  finally:
      apply_globals(prior)
      if clear_caches_on_exit: ...
  ```
  Add a regression test that passes a valid + invalid knob pair and asserts every
  global equals prior after the raise.

### [P2] F-18. Richards-Wolf GUI dock calls richards_wolf_focus with a fabricated signature and unpacks the tuple return as an object: the only application entry point to the vector-diffraction module is permanently non-functional

- **Location**: `lumenairy/ui/richards_wolf_dock.py:45-48, 183-188, 208-211`
- **Category**: api-drift
- **Dimension**: vector-diffraction
- **Votes**: real 3/3, new 3/3
- **Auditor cross-check (richards_wolf_dock.py)**: CONFIRMED-READING. The dock
  worker (richards_wolf_dock.py:45-48) calls richards_wolf_focus(NA=, wavelength=,
  n_im=, polarization=, N=, dx=, z=); the real signature is (pupil, wavelength, NA,
  f, dx_pupil, N_focal, dx_focal, z_planes, polarization). Kwargs n_im/N/dx/z do not
  exist and required pupil/f/dx_pupil are omitted -> every call raises TypeError into
  the except branch; _on_finished then reads res.Ex on the tuple/error-dict. The only
  GUI entry point to the vector-diffraction module is permanently non-functional.
- **What's wrong**: The worker invokes la.richards_wolf_focus(NA=..., wavelength=...,
  n_im=self.n_im, polarization=..., N=self.N, dx=self.dx_m, z=self.z_offset_m). The
  actual public signature (vector_diffraction.py:30) is richards_wolf_focus(pupil,
  wavelength, NA, f, dx_pupil, N_focal=None, dx_focal=None, z_planes=None,
  polarization='x'). There is no n_im, N, dx, or z parameter, and the required
  positional args pupil, f, dx_pupil are never supplied, so the call ALWAYS raises
  TypeError (unexpected keyword 'n_im' / missing 'pupil'). Even if the kwargs were
  corrected, the polarization strings passed ('linear_x','rcp','lcp','radial',
  'azimuthal' from polmap at lines 146-153) are rejected by the library, which
  only accepts 'x','y','circular' (vector_diffraction.py:167-174) -> ValueError,
  and 'radial'/'azimuthal' are unsupported entirely. Compounding this, _draw_result
  (line 183) and _summarise (line 208) read res.Ex/res.Ey/res.Ez as attributes, but
  richards_wolf_focus RETURNS A TUPLE (Ex, Ey, Ez, x_f, y_f) (line 278), so res.Ex
  raises AttributeError and the except branch silently draws nothing. This is a
  data-correctness/API contract bug, not a threading concern.
- **Why it matters**: richards_wolf_focus/debye_wolf_psf are exported at the top
  level (lumenairy/__init__.py:625-626,1424-1425) but the dock is the only place in
  the shipped application that wires them in. Every user click on 'Compute
  Richards-Wolf focus' is caught at richards_wolf_dock.py:50-52 and surfaced as
  'Richards-Wolf failed: TypeError: ... unexpected keyword argument n_im'. The
  advertised high-NA vector focal-spot analysis (|Ex|^2,|Ey|^2,|Ez|^2,
  longitudinal fraction) is dead -- a triple API-contract drift between the
  v4.11.2 library refactor and the UI, never reconciled.
- **Regime**: Any interactive use of the Richards-Wolf dock (NA>0.4 high-NA
  focal-spot analysis) at any wavelength/NA/grid -- 100% reproducible on the first
  Compute click.
- **Evidence**:
  ```
  # richards_wolf_dock.py:45-48
  res = la.richards_wolf_focus(
      NA=self.NA, wavelength=self.wavelength,
      n_im=self.n_im, polarization=self.polarization,
      N=self.N, dx=self.dx_m, z=self.z_offset_m)
  # vs vector_diffraction.py:30
  #   def richards_wolf_focus(pupil, wavelength, NA, f, dx_pupil,
  #       N_focal=None, dx_focal=None, z_planes=None, polarization='x'):
  # vector_diffraction.py:278 return Ex, Ey, Ez, x_f, y_f
  # richards_wolf_dock.py:183 comps = [('|Ex|^2', np.abs(res.Ex) ** 2), ...]
  ```
- **Proposed fix**: Rewrite RichardsWolfWorker.run to (a) synthesise a uniform
  circular pupil and physical f/dx_pupil from NA and the system aperture (e.g. f =
  aperture_radius/NA, dx_pupil from the pupil grid), (b) call res =
  la.richards_wolf_focus(pupil, wavelength, NA, f, dx_pupil, N_focal=N,
  z_planes=[z_offset], polarization=<mapped>), (c) map UI labels to the library
  strings {'Linear x':'x','Linear y':'y','Circular (RCP)':'circular'} and either
  drop Radial/Azimuthal/LCP from the combo box or implement them as explicit
  (px,py) Jones tuples, and (d) unpack the tuple: Ex, Ey, Ez, x_f, y_f = res,
  emitting a small namespace/dataclass (or change _draw_result/_summarise to index
  res[0..2]). Add a unit/integration test that imports the dock worker and asserts
  it returns finite Ex/Ey/Ez.

### [P2] F-19. surface_sag_biconic returns flat 0.0 outside conic domain while surface_sag_general returns NaN -- silent apparently-flat-ring sag the canonical path was fixed to avoid

- **Location**: `lumenairy/elements/lenses.py:320-327`
- **Category**: convention
- **Dimension**: numerical-edge-cases
- **Votes**: real 3/3, new 3/3
- **What's wrong**: In surface_sag_biconic, the per-axis helper _axis_sag computes
  norm = (1+K)*h_sq/R**2 and, for pixels where norm >= 0.9999 (outside the
  conic-defined domain), returns 0.0 via xp.where(valid, h_sq/(R*(1+sqrt(denom_arg))),
  0.0). This is the exact opposite of the canonical rotationally-symmetric path
  surface_sag_general (same file, lines 200-208), which for the identical
  norm>=0.9999 test returns xp.nan with an explicit comment (lines 196-199):
  'Pre-4.10 silently returned 0 sag there, which produced an apparently-flat ring
  at the surface edge for hyperbolic/oblate conics extending past the geometric
  rim. Return NaN instead so downstream consumers either mask those pixels or see
  the failure.' The biconic helper still does the pre-4.10 thing the general path
  was deliberately changed away from. The _lens_thin aspheric path
  (_lens_thin.py:478) and the JAX-adjacent NumPy reference all standardized on
  NaN; only biconic was missed.
- **Why it matters**: For an oblate biconic/anamorphic surface (conic_x>0 or
  conic_y>0) or a fast biconic whose clear aperture extends near the conic vertex
  zone, the rim pixels silently get a flat (0.0) sag contribution along the
  offending axis instead of NaN. Because biconic sag feeds apply_real_lens phase =
  -k*(n-1)*sag and the ray tracer's _surface_sag, the result is a smooth-looking
  but physically wrong flat annulus at the surface edge -- no NaN, no warning, no
  aperture-mask trigger. This is precisely the silent-wrong-physics outcome the
  v4.10 NaN convention was introduced to make visible.
- **Regime**: Anamorphic / biconic / toroidal optics (Zemax 'Biconic' or
  'Toroidal' surfaces) with a positive conic constant on at least one axis, or a
  steep biconic whose semi-diameter approaches sqrt(R^2/(1+K)) on either axis --
  e.g. an oblate cylindrical corrector or a toroidal where the cross-section
  radius R_y is small relative to the aperture half-height. Square-grid
  sphere/prolate cases never trip the branch, so it is a narrow but real path.
- **Evidence**:
  ```
  def _axis_sag(h_sq, R, K, asph):
      s = xp.zeros_like(h_sq)
      if R is not None and not np.isinf(R):
          norm = (1 + K) * h_sq / R ** 2
          valid = norm < 0.9999
          denom_arg = xp.where(valid, 1 - norm, 0.01)
          s = xp.where(
              valid,
              h_sq / (R * (1 + xp.sqrt(denom_arg))),
              0.0,          # <-- 0.0, but surface_sag_general returns xp.nan here
          )
  # Compare surface_sag_general (same file, lines 203-207):
  #   conic_sag = xp.where(valid, h_sq/(R*(1+xp.sqrt(denom_arg))), xp.nan)
  ```
- **Proposed fix**: Replace the 0.0 out-of-domain fill in _axis_sag with xp.nan to
  match surface_sag_general, so out-of-conic-domain biconic pixels are NaN-flagged
  for downstream masking instead of silently flattened. (Because the two per-axis
  sags are summed at line 337, prefer s = xp.where(valid, conic_sag, xp.nan) on
  each axis so a single out-of-domain axis poisons the pixel exactly as the
  rotationally-symmetric path does.) Add a regression pin asserting
  np.isnan(surface_sag_biconic(...)) on a rim pixel of an oblate biconic.

### [P3] F-20. Per-variable scale_floor / forward-FD step sizing is inert on the default design_optimize gradient path

- **Location**: `lumenairy/optimize/driver.py:818-823,1160-1161`
- **Category**: api-drift
- **Dimension**: optimize-subsystem
- **Votes**: real 3/3, new 3/3
- **What's wrong**: final_jac is set to _merit_jac_auto only when use_analytic_jac
  is True (jac=='auto' AND at least one JaxMeritTerm with build_args). For the
  overwhelmingly common case -- jac='auto' with no JAX-grad merits --
  use_analytic_jac is False and user_jac is None, so final_jac=None. so.minimize is
  then called with jac=None and no finite_diff_rel_step/eps option, so scipy
  estimates the gradient with its OWN internal 2-point FD and never sees
  parameterization.scale_floor. The entire v5.2 'AUDIT_V4_13_1 P1-1 closure'
  machinery (_DEFAULT_SCALE_FLOORS, _classify_path_to_floor,
  DesignParameterization.scale_floor, and _fd_grad_for's per-variable step) only
  takes effect on the JaxMeritTerm-combined path (_merit_jac_auto -> _fd_grad_for)
  and method='newton' (_grad_for_newton).
- **Why it matters**: Users who set a custom scale_floor (or rely on the
  auto-classified radius=1um / conic=1e-3 floors) to fix near-zero-parameter step
  collapse get no effect at all on the default L-BFGS-B run with ordinary
  (non-JAX) merit terms. The documented 'fix' silently does nothing where it was
  claimed to matter most; the gradient quality of the mainstream path is whatever
  scipy's default relative step yields, independent of variable type. This is a
  contract/expectation mismatch rather than a wrong number, hence P3, but it means
  a key tuning knob is dead on the default path.
- **Regime**: design_optimize(method='L-BFGS-B'/'SLSQP'/'trust-constr', jac='auto')
  with only NumPy merit terms (no JaxMeritTerm build_args) and a parameterization
  that sets or relies on scale_floor.
- **Evidence**:
  ```
  use_analytic_jac = (jac == 'auto' and len(jax_grad_terms) > 0)
  final_jac = (user_jac if user_jac is not None
               else (_merit_jac_auto if use_analytic_jac else None))
  ...
  res = so.minimize(merit_fn, x0, method=method, **_minimize_kwargs)
  # where _minimize_kwargs['jac'] = final_jac and no finite_diff_rel_step is supplied.
  ```
- **Proposed fix**: When final_jac is None, either (a) install a default FD jac
  that routes through _fd_grad_pure with parameterization.scale_floor, or (b)
  translate scale_floor into scipy's finite_diff_rel_step option (rel step =
  scale_floor/|x| per coordinate) and pass it via _minimize_kwargs. Alternatively,
  document clearly that scale_floor only affects the JAX-combined and newton paths.

### [P3] F-21. HF-quadrature Van Vleck-Morette prefactor comment states -i/(2pi) but code applies -i (code correct, comment misleading)

- **Location**: `lumenairy/propagators/asymptotic_canonical_fit.py:1211-1217`
- **Category**: convention
- **Dimension**: asymptotic-propagators
- **Votes**: real 3/3, new 2/3
- **What's wrong**: The comment claims the d=2 Van Vleck-Morette asymptotic
  prefactor is (2pi)^(-d/2)*i^(-d/2) = -i/(2pi), but the code multiplies by -1j (no
  1/(2pi)). The code is actually correct: with Phi in waves the density
  sqrt(|det d2Phi/ds1ds2|) carries a 1/lambda factor (det ~ 1/lambda^2 for d=2)
  which cancels the lambda in the full Van Vleck prefactor k/(2pi i) = 1/(i
  lambda), leaving exactly -i. Reduction to the paraxial Fresnel kernel confirms
  it: cross-Hessian d2Phi/ds1ds2 = -1/(lambda z), density = 1/(lambda z), times -i
  gives -i/(lambda z) = the stated Fresnel kernel. The 1/(2pi) in the comment is
  spurious.
- **Why it matters**: The comment is a correctness trap: a future maintainer
  'fixing the code to match the documented -i/(2pi)' would introduce a real,
  silent 1/(2pi) ~ 0.159x amplitude error in all HF-quadrature output and break
  coherent superposition with ASM/Fresnel. Documentation that contradicts correct
  code is a latent regression vector.
- **Regime**: Any audit/maintenance pass that trusts the inline comment over the
  code; does not affect runtime correctness today.
- **Evidence**:
  ```
  # 4.10: apply the Van Vleck-Morette asymptotic prefactor (2pi)^(-d/2)*i^(-d/2)
  # for d = 2: this is -i/(2pi).  ...
  out = out * (-1j)
  ```
- **Proposed fix**: Correct the comment: the d=2 prefactor that reduces to the
  Fresnel kernel, after the Phi-in-waves density absorbs the lambda factors, is
  exactly -i (not -i/(2pi)); spell out the lambda cancellation so the bare -1j
  multiply is justified and not 'corrected' later.

### [P3] F-22. GaussianBSDF.evaluate builds specular with shape (3,...) instead of (...,3), crashing the advertised batched-incidence path

- **Location**: `lumenairy/elements/bsdf.py:293-297`
- **Category**: numerical
- **Dimension**: elements-doe-grating
- **Votes**: real 3/3, new 2/3
- **Auditor cross-check (bsdf.py)**: CONFIRMED-REPRO. "GaussianBSDF.evaluate with
  batched (M,3) incident+scattered dirs raises ValueError: operands could not be
  broadcast (4,3) (3,4) -- the advertised batched-incidence path is broken (shape
  transpose). Single-direction call works. CONFIRMED. (The antipode-flip sampler
  claim was not separately reproduced.)"
- **What's wrong**: The ternaries `inc[..., 0] if inc.ndim else inc[0]` are wrapped
  in `np.array([...])`, so when incident_dir is itself a (...,3) batch, specular is
  stacked along a NEW leading axis giving shape (3, ...). The subsequent
  `np.sum(sd * specular, axis=-1)` then tries to broadcast scattered_dir (...,3)
  against specular (3,...) and raises ValueError. The inc.ndim ternary clearly
  intends to SUPPORT per-element incident directions, but the assembly axis is
  wrong.
- **Why it matters**: Any caller passing a per-pixel or per-ray array of incident
  directions (the natural vectorized stray-light usage) gets a hard ValueError
  instead of an array of BSDF values. The 1-D incident path used by
  total_integrated_scatter and sample() works, masking the defect in the test
  suite.
- **Regime**: Batched/vectorized incident_dir (shape (...,3)); e.g. evaluating the
  Gaussian BSDF over a grid of incidence directions for an irradiance integral.
- **Evidence**:
  ```
  specular = np.array([inc[..., 0] if inc.ndim else inc[0],
                       inc[..., 1] if inc.ndim else inc[1],
                       -inc[..., 2] if inc.ndim else -inc[2]])
  ...
  cos_theta = np.clip(np.sum(sd * specular, axis=-1), -1.0, 1.0)
  # verified: evaluate(incb(4,5,3), S(4,5,3)) -> ValueError: operands could not
  #   be broadcast together with shapes (4,5,3) (3,4,5)
  ```
- **Proposed fix**: Assemble specular along the last axis: `specular =
  np.stack([inc[..., 0], inc[..., 1], -inc[..., 2]], axis=-1)` for the batched case
  (and `np.array([inc[0],inc[1],-inc[2]])` for 1-D), mirroring how
  HarveyShackBSDF.evaluate handles the sd.ndim split.

### [P3] F-23. grating_efficiency_vs_wavelength silently returns the nearest available order when the requested order exceeds +/-n_orders

- **Location**: `lumenairy/elements/thin_grating.py:198`
- **Category**: api-drift
- **Dimension**: elements-doe-grating
- **Votes**: real 3/3, new 2/3
- **What's wrong**: `idx = np.argmin(np.abs(orders - order))` picks the closest
  computed order index. When the requested order is outside the retained window
  [-n_orders, n_orders] (default n_orders=11), argmin clamps to the boundary order
  (e.g. order=20 -> returns the +11 efficiency) and returns it as if it were the
  requested order, with no error or NaN.
- **Why it matters**: A user sweeping a high diffraction order, or one who
  under-sizes n_orders, gets a silently wrong efficiency curve attributed to the
  wrong order. There is no diagnostic; the result is plausible-looking but
  physically mislabeled.
- **Regime**: Any call with |order| > n_orders (e.g. order=20 with the default
  n_orders=11), common when probing high-order blaze behavior.
- **Evidence**:
  ```
  idx = np.argmin(np.abs(orders - order))
  eff[i] = T[idx]
  # verified: grating_efficiency_vs_wavelength(..., order=20) with default
  #   n_orders=11 returns eff=0.00291 (the order-11 value), not an error/NaN
  ```
- **Proposed fix**: Validate that abs(order) <= n_orders and raise ValueError
  otherwise, or locate the exact index via np.flatnonzero(orders == order) and
  raise if empty, so an out-of-window order request fails loudly instead of
  returning a mislabeled neighbor.

### [P3] F-24. thin_grating_efficiency_1d Returns docstring promises T_eff 'Sums to 1 by energy conservation' but truncated evanescent orders make the sum < 1

- **Location**: `lumenairy/elements/thin_grating.py:109-111 (behavior at 155-160)`
- **Category**: physics
- **Dimension**: elements-doe-grating
- **Votes**: real 3/3, new 3/3
- **What's wrong**: The Returns section states T_eff 'Sums to 1 by energy
  conservation (for lossless materials)'. Parseval guarantees the FULL infinite
  set of |t_m|^2 sums to 1, but the function zeroes every order with |kx_m| >=
  k0*n_substrate (line 155-160) without redistributing or renormalizing the lost
  power. Once any diffracted order is evanescent, sum(T_eff) drops below 1 and that
  energy silently vanishes (R is forced to 0).
- **Why it matters**: A user reading the public docstring will trust T_eff for an
  efficiency/energy budget. For any grating with period within ~10x of the
  wavelength (the normal regime where diffraction matters), several orders are
  evanescent and the budget is wrong by tens of percent. The dock-level test pins
  only sum>=0.5 with a 'within propagating-order truncation' caveat that the public
  docstring lacks.
- **Regime**: Period comparable to a few wavelengths (e.g. 2 um period at 1.31 um
  -> sum(T_eff)=0.81); any sub-10x-wavelength NIR/visible grating.
- **Evidence**:
  ```
  T_eff = np.where(propagating, np.abs(tm) ** 2, 0.0)
  # docstring: 'T_eff ... Sums to 1 by energy conservation (for lossless materials).'
  # verified: period=2e-6, wl=1.31e-6 -> T.sum()=0.8106 with only 5 of 31 orders propagating
  ```
- **Proposed fix**: Amend the Returns docstring to state that T_eff sums to 1 ONLY
  when all retained orders propagate, and that evanescent-order truncation reduces
  the sum (the unscattered remainder is not modeled); optionally expose the
  propagating-order power deficit so callers can detect the loss.

### [P3] F-25. optical_invariant docstring worked formula H = h*D/(4*(f/#)*f) contradicts the (correct) code H = (D/2)*(h/efl)

- **Location**: `lumenairy/raytrace/paraxial.py:119-157`
- **Category**: physics
- **Dimension**: raytrace-seidel-paraxial
- **Votes**: real 3/3, new 3/3
- **What's wrong**: The docstring derives the Lagrange invariant as H = (D/2)/(2*
  (f/#)) * (h/f) = h*D/(4*(f/#)*f) (lines 119-121), but the implementation computes
  y_marg = D/2 and u_chief = field_height/efl and returns y_marg*u_chief =
  (D/2)*(h/efl) (lines 152-157). These differ by a spurious factor 1/(2*(f/#)): the
  docstring formula has dimensions/magnitude of h/(4*(f/#)^2), not the
  marginal-height * chief-angle product. The code matches the function's own
  definition block (lines 111-113, H = y_pupil*u_chief) and is the physically
  correct paraxial invariant; the worked equation in the docstring is wrong.
- **Why it matters**: optical_invariant is advertised as a sanity-check helper for
  cross-validating EFL/f-number/FoV. A user who trusts the docstring formula to
  hand-check the returned number will see a discrepancy of a factor 2*(f/#) (e.g.
  8x at f/4) and may wrongly conclude the code is broken, or copy the wrong formula
  elsewhere. No test pins either form, so the contradiction sits unguarded.
- **Regime**: Documentation cross-check by any user; manifests as a numeric
  mismatch whenever f/# != 0.5.
- **Evidence**:
  ```
  # docstring:
  #   H = (D / 2) / (2 * (f/#)) * (h / f) = h * D / (4 * (f/#) * f)
  # code:
  y_marg = float(pupil_diameter_m) / 2.0
  u_chief = float(field_height_m) / float(efl)
  return y_marg * u_chief        # = (D/2)*(h/efl), NOT /(2*(f/#))
  ```
- **Proposed fix**: Correct the docstring derivation to H = (D/2)*(h/f) (= h/(2*
  (f/#)) when D = f/(f/#)), removing the extraneous 1/(2*(f/#)) factor, so the
  documented formula matches the implemented (and physically correct) invariant.

### [P3] F-26. wave_opd_1d docstring omits focal_length and f_ref params

- **Location**: `lumenairy/analysis/opd.py:464-499`
- **Category**: api-drift
- **Dimension**: analysis-opd-wavefront
- **Votes**: real 3/3, new 3/3
- **What's wrong**: focal_length (warning 508-520) and f_ref (ref-sphere
  divide/add-back 536-557) are functional but Parameters ends at dy (480-481) and
  lists neither; note at 534-535 is stale, add-back is internal at 556-557.
- **Why it matters**: check_opd_sampling (304-306) tells users to use f_ref, but
  the 1D docstring never lists it; stale note risks double-adding the ref sphere.
- **Regime**: 1D OPD cut of a fast converging wavefront with sub-Nyquist sampling.
- **Evidence**:
  ```
  # Sig 451-460 has focal_length, f_ref; Parameters end at dy 480-481;
  #   sentence 534-535 contradicts 556-557.
  ```
- **Proposed fix**: Document focal_length and f_ref; remove stale note.

### [P3] F-27. CausticDiagnostic det_J doc says Sign of det but stores raw determinant

- **Location**: `lumenairy/analysis/aberration.py:333-336`
- **Category**: api-drift
- **Dimension**: analysis-opd-wavefront
- **Votes**: real 3/3, new 3/3
- **What's wrong**: Doc says det_J is the Sign of det but code stores the signed
  determinant at 533 with no sign reduction; plotter uses continuous curve at 620.
- **Why it matters**: Callers treating det_J as plus or minus 1 misread it; values
  span orders of magnitude. No test pins it.
- **Regime**: Caller inspecting det_J per documented contract.
- **Evidence**:
  ```
  # Doc 333-334 vs det value at 533 vs continuous plot at 620.
  ```
- **Proposed fix**: Reword 333-334 to "Signed value of determinant, sign flips mark
  caustic crossings."

### [P3] F-28. through_focus_scan (NumPy) leaves best_focus_strehl / best_focus_spot at NaN while the JAX twin populates them

- **Location**: `lumenairy/analysis/through_focus.py:455-462 vs 1192-1202`
- **Category**: jax-parity
- **Dimension**: analysis-coronagraph-ao
- **Votes**: real 3/3, new 3/3
- **What's wrong**: The NumPy through_focus_scan constructs its
  ThroughFocusResult without passing best_focus_strehl or best_focus_spot (lines
  455-462), so they fall back to the dataclass defaults float('nan') (lines
  287-288). The JAX twin through_focus_scan_jax computes `best_strehl =
  float(np.nanmax(strehl)) if ideal_peak else nan` and `best_spot =
  float(np.nanmin(rms_r))` and DOES populate both fields (lines 1192-1202). The two
  backends therefore return different ThroughFocusResult contents for identical
  inputs.
- **Why it matters**: through_focus_scan is dispatched via
  backend='numpy'|'jax' and documented as having the 'same return contract' (line
  1065). A caller that reads result.best_focus_spot / best_focus_strehl silently
  gets NaN on the default NumPy path but a real value on JAX, a backend-dependent
  result with no warning. No test pins either behavior, so the divergence is
  undetected.
- **Regime**: Any caller of through_focus_scan(..., backend='numpy') that
  subsequently reads result.best_focus_strehl or result.best_focus_spot, then
  compares against a JAX run.
- **Evidence**:
  ```
  # NumPy path: return ThroughFocusResult(z=z_arr, peak_I=peak_I, strehl=strehl,
  #   d4sigma_x=d4x, d4sigma_y=d4y, rms_radius=rms_r, power_in_bucket=p_bucket,
  #   ...) -- no best_focus_* args.
  # JAX path: best_strehl = float(np.nanmax(strehl)) if ideal_peak else float('nan')
  #   best_spot = float(np.nanmin(rms_r))
  #   return ThroughFocusResult(..., best_focus_strehl=best_strehl,
  #     best_focus_spot=best_spot)
  ```
- **Proposed fix**: Compute best_focus_strehl/best_focus_spot in the NumPy path too
  (guarding all-NaN slices with np.isfinite checks before np.nanmax/np.nanmin,
  mirroring find_best_focus's guard), or have both backends leave the fields at NaN
  and document find_best_focus as the canonical accessor. Also harden the JAX
  path's np.nanmin(rms_r) against an all-NaN rms_r (raises ValueError) for parity.

### [P3] F-29. Exporters ignore prescription['stop_index']/is_stop -> load->export->load relocates the aperture stop to surface 0, breaking documented lossless round-trip

- **Location**: `lumenairy/io/prescriptions_quadoa.py:176, 192 (and prescriptions_zemax.py:1354/1529, prescriptions_code_v.py:128)`
- **Category**: api-drift
- **Dimension**: io-prescriptions
- **Votes**: real 3/3, new 3/3
- **What's wrong**: All three exporters take stop_surface as a keyword defaulting
  to 0 and never consult the prescription's own stop position. export_quadoa_qos
  sets per-surface 'is_stop': bool(i == stop_surface) (line 176) and top-level
  'stop_surface': int(stop_surface) (line 192) using only the parameter. The
  loaders DO produce the stop position: load_quadoa_qos sets result['stop_index']
  (prescriptions_quadoa.py:333-334), load_codev_seq sets stop_index
  (prescriptions_code_v.py:410-411), and load_zemax_zmx sets per-element is_stop.
  None of the exporters read prescription.get('stop_index') or the per-surface
  is_stop flag, so a prescription loaded with its stop on (say) surface 2 is
  re-exported with STOP on surface 0 unless the caller manually re-passes
  stop_surface=2.
- **Why it matters**: load_quadoa_qos's docstring states it 'Round-trips
  losslessly with export_quadoa_qos' and export_quadoa_qos 'captures every field a
  lumenairy prescription carries ... stop index'. In fact stop position is NOT
  preserved across a load->export round trip: the stop silently moves to surface 0,
  changing pupil location / vignetting / aperture-detection on the next load.
- **Regime**: load_quadoa_qos / load_codev_seq / load_zemax_zmx of a design whose
  stop is not the first refractive surface (typical doublets/triplets), followed by
  re-export with the default stop_surface.
- **Evidence**:
  ```
  # prescriptions_quadoa.py:176 'is_stop': bool(i == stop_surface),
  # prescriptions_quadoa.py:192 'stop_surface': int(stop_surface),
  #   -- both driven solely by the stop_surface kwarg (default 0), never by
  #   prescription['stop_index'] which load_quadoa_qos emits at :333-334.
  ```
- **Proposed fix**: Have each exporter default stop_surface from the prescription
  when the caller does not override it, e.g. `if stop_surface is None: stop_surface
  = prescription.get('stop_index', 0)` (or scan is_stop on elements for the Zemax
  path). Add a round-trip test asserting load(export(load(p)))['stop_index'] ==
  p_stop.

### [P3] F-30. RandomState.integers JAX branch defaults to int32 while NumPy/CuPy default to int64 (cross-backend dtype divergence on x64-enabled JAX)

- **Location**: `lumenairy/backend/random.py:126-142`
- **Category**: jax-parity
- **Dimension**: backend-dispatch
- **Votes**: real 3/3, new 3/3
- **What's wrong**: The NumPy and CuPy branches draw integers with dtype=dtype or
  np.int64 (lines 128, 132), so the default is int64. The JAX branch (lines
  136-137) hardcodes dtype = jax.numpy.int32 when dtype is None. On an x64-enabled
  JAX build, jax.random.randint can produce int64 (verified:
  jax.random.randint(k,(5,),0,10,dtype=int64).dtype == int64), so the int32 default
  is a deliberate downcast that disagrees with the NumPy/CuPy default rather than a
  JAX limitation.
- **Why it matters**: RandomState is the single shim HFPI / GBD / Monte-Carlo path
  sampling and detector index sampling run through, written once to run on either
  idiom. A caller drawing indices via rs.integers(...) gets int64 on NumPy/CuPy but
  int32 on JAX. The codebase already treats exactly this hazard as a real bug for
  RandomState.choice: the v4.13.2 'P1-NEW-I' fix (lines 178-184) pins jnp.int64
  specifically because 'on x64-enabled builds the difference broke cross-backend
  pipelines that downcast to int32.' integers was never given the same pin.
- **Regime**: JAX backend with jax_enable_x64=True, dtype not explicitly passed,
  indices fed to a dtype-sensitive downstream (concatenation/comparison with NumPy
  int64 arrays, dict/index keys, or anything that downcasts).
- **Evidence**:
  ```
  # numpy/cupy: arr = self._rng.integers(low, high, size=shape, dtype=dtype or np.int64)
  # JAX: if dtype is None:
  #          dtype = jax.numpy.int32
  #      ... return jax.random.randint(sub, shape, low, cast(int, high), dtype=dtype)
  # Verified at runtime: x64-on JAX integers default = int32, NumPy default = int64.
  ```
- **Proposed fix**: Mirror the choice fix: in the JAX branch set `if dtype is None:
  dtype = jax.numpy.int64` (JAX silently truncates to int32 on x64-disabled builds,
  exactly as the choice line-184 pin already relies on), so the default dtype
  matches NumPy/CuPy on every JAX configuration.

### [P3] F-31. RandomState.uniform / normal JAX branch hardcodes float32 default while NumPy/CuPy return float64 (silent precision downcast on x64-enabled JAX)

- **Location**: `lumenairy/backend/random.py:96-119`
- **Category**: jax-parity
- **Dimension**: backend-dispatch
- **Votes**: real 3/3, new 3/3
- **What's wrong**: In uniform (lines 99-102) and normal (lines 117-119) the JAX
  branch sets `if dtype is None: dtype = jax.numpy.float32` and draws float32
  samples, whereas the NumPy/CuPy branches (lines 91, 94, 110, 113) call the
  Generator with no dtype, yielding float64. On an x64-enabled JAX build
  jax.random.uniform/normal can return float64 (verified: dtype=float64 ->
  float64), so the float32 default is a deliberate precision downcast that
  disagrees with the other backends rather than a JAX limitation.
- **Why it matters**: These primitives drive physics sampling: HFPI draws uniform
  phases on [0, 2*pi) (hfpi.py:204-205, 325-326, 654-657) and Schell-model /
  Monte-Carlo tolerancing draw normal/uniform realizations. On the JAX path the
  random component of an otherwise complex128 field is silently single-precision,
  so a JAX run is not the same numerical experiment as the NumPy run at the same
  seed-equivalent. For high path-count coherence/MCF estimation this caps the
  achievable precision of the stochastic estimator and breaks bit-for-bit
  cross-backend reproducibility that the shim's docstring implies.
- **Regime**: JAX backend with jax_enable_x64=True, dtype not explicitly passed;
  HFPI/Schell/Monte-Carlo sampling where float32 vs float64 of the random draw
  changes the result or convergence vs the NumPy reference.
- **Evidence**:
  ```
  # uniform JAX: if dtype is None:
  #                  dtype = jax.numpy.float32
  #              return jax.random.uniform(sub, shape, dtype=dtype, minval=low, maxval=high)
  # numpy: arr = self._rng.uniform(low, high, size=shape)  (float64)
  # normal JAX: if dtype is None:
  #                 dtype = jax.numpy.float32
  #             return mean + std * jax.random.normal(sub, shape, dtype=dtype)
  # Verified: x64-on JAX uniform default = float32, NumPy default = float64.
  ```
- **Proposed fix**: Default the JAX branch to jax.numpy.float64 when dtype is None
  (JAX truncates to float32 automatically on x64-disabled builds, so this only
  changes behaviour where float64 is actually available), matching the NumPy/CuPy
  float64 default and the parity intent codified by the choice fix.

### [P3] F-32. warmup_fft_plans defaults threads to _available_cpus() while _fft2/_ifft2 dispatch on the FFTW_THREADS global, so warmup builds unreachable plans after set_fft_threads()

- **Location**: `lumenairy/propagators/fft_infra.py:730-736`
- **Category**: api-drift
- **Dimension**: fft-infra-deep
- **Votes**: real 3/3, new 3/3
- **What's wrong**: warmup_fft_plans resolves its default thread count as `if
  threads is None: threads = max(1, _available_cpus())` (line 730-731) and builds
  plan-cache entries keyed by that thread count. But the actual FFT dispatchers key
  their lookup on the module global FFTW_THREADS: _fft2 uses `threads =
  FFTW_THREADS if FFTW_THREADS > 0 else 1` (line 1521; same at 1572/1605/1634).
  FFTW_THREADS starts equal to _available_cpus() at import (line 112), so by
  default they agree -- but set_fft_threads(n) (line 1307-1323) overwrites
  FFTW_THREADS to an arbitrary n WITHOUT clearing or re-keying the plan cache.
  After set_fft_threads(8) on a 24-core box, warmup_fft_plans([(1024,1024)]) builds
  plans under key threads=24, while the subsequent angular_spectrum_propagate at
  N=1024 looks up threads=8 -> cache MISS -> pays the full plan cost.
- **Why it matters**: Defeats the documented purpose of warmup_fft_plans ('the
  first call to be fast', 'so the first propagation at each shape pays the planning
  cost only once at warmup, not inside a hot loop'). The user sees an unexplained
  ~100-1000 ms stall on the first real FFT despite calling warmup, and the
  threads=24 plans sit in the LRU as dead entries that can evict legitimately
  useful threads=8 plans. No wrong physics, but a real API-contract drift in a
  perf-tuning entry point.
- **Regime**: pyFFTW installed; user calls set_fft_threads(n) with n !=
  available_cpus (the standard recipe for ProcessPoolExecutor workers,
  set_fft_threads(1)) and then calls warmup_fft_plans(shapes) without explicitly
  passing threads=.
- **Evidence**:
  ```
  # fft_infra.py:730-736
  if threads is None:
      threads = max(1, _available_cpus())
  n = 0
  for shape in shapes:
      for direction in ('fwd', 'inv'):
          _get_or_make_plan(direction, tuple(shape), dtype, int(threads))
  # vs fft_infra.py:1521 threads = FFTW_THREADS if FFTW_THREADS > 0 else 1
  # set_fft_threads (1317-1323) sets FFTW_THREADS without touching _PYFFTW_PLAN_CACHE.
  ```
- **Proposed fix**: Default warmup to the same source the dispatchers use: `if
  threads is None: threads = FFTW_THREADS if FFTW_THREADS > 0 else 1` (read the live
  global, not _available_cpus()). This makes warmup-built keys coincide with
  dispatch keys regardless of any prior set_fft_threads() call.

### [P3] F-33. BaF2 bundled Sellmeier coefficients are low-precision / mis-poled (~0.4-0.5% index error in the visible), unlike every neighbouring entry

- **Location**: `lumenairy/glass.py:209-210`
- **Category**: physics
- **Dimension**: glass-coatings-physics
- **Votes**: real 3/3, new 3/3
- **What's wrong**: The bundled BaF2 Sellmeier row uses 4-significant-figure,
  slightly-wrong coefficients: 'BaF2': ((0.6435, 0.5067, 3.8261), (1.5e-3, 9.5e-3,
  2.5e3)). The third pole C3=2500 um^2 implies a 50.0 um resonance, but the
  authoritative Li-1980/Malitson BaF2 fit has the IR pole at 46.386 um (C3=2151.7
  um^2) and B/C poles at 0.057789 / 0.10968 um (C1=3.34e-3, C2=1.203e-2 um^2), not
  the rounded 1.5e-3 / 9.5e-3 here. The result is n_d=1.47201 vs the correct
  1.47448, and the error grows toward the blue (n(0.4um)=1.4792 bundled vs 1.4848
  authoritative). Every surrounding bundled row carries full-precision coefficients
  and an explicit n_d cross-check comment; this BaF2 row carries neither, matching
  the fingerprint of the earlier S-LAH64/79 mis-attribution bug (v4.11.2 CRIT-3).
- **Why it matters**: On a minimal install (no optional 'refractiveindex' package)
  get_glass_index('BaF2', lam) and get_glass_index_complex('BaF2', lam) fall
  through to this bundled row (glass.py:1318-1320), silently returning an index
  ~0.0025-0.0056 too low across the visible. BaF2 is a common UV-IR window / lens
  material; a 0.4% index error propagates directly into computed lens power, OPD,
  and AR-coating phase-thickness, and is silent (no warning, value is in-range).
- **Regime**: Minimal install without the 'refractiveindex' package, querying BaF2
  in the visible/near-UV (e.g. design-time index lookups, AR-coating thickness, or
  get_glass_index_complex fallback). With refractiveindex installed the tuple path
  takes precedence and the bundled row is bypassed, so the bug is masked.
- **Evidence**:
  ```
  # glass.py:209-210
  'BaF2':       ((0.6435, 0.5067, 3.8261),
                 (1.5e-3, 9.5e-3, 2.5e3)),
  # numeric check: bundled n_d=1.472011 vs Li/Malitson 1.474476
  #   (diff -0.00247 at d-line, -0.00564 at 0.4 um); third pole sqrt(2500)=50.0 um
  #   vs authoritative 46.386 um.
  ```
- **Proposed fix**: Replace with the authoritative full-precision Li-1980/Malitson
  BaF2 Sellmeier (n^2-1 = 0.643356 lam^2/(lam^2-0.057789^2) + 0.506762
  lam^2/(lam^2-0.10968^2) + 3.8261 lam^2/(lam^2-46.3864^2)), i.e. ((0.643356,
  0.506762, 3.8261), (0.057789**2, 0.10968**2, 46.3864**2)), and add an n_d
  cross-check comment (1.47448 at 0.5876 um) matching the convention used for the
  CaF2/MgF2 rows above it. Optionally add a tests/unit cross-check pinning n_d to
  5e-5 as done for the CDGM block.

### [P3] F-34. Dead code: focal polar coordinates computed then discarded in richards_wolf_focus

- **Location**: `lumenairy/propagators/vector_diffraction.py:127-128`
- **Category**: test-quality
- **Dimension**: vector-diffraction
- **Votes**: real 3/3, new 3/3
- **What's wrong**: After building the focal Cartesian meshgrid Xf,Yf, the function
  evaluates `np.sqrt(Xf ** 2 + Yf ** 2)` and `np.arctan2(Yf, Xf)` as bare
  expression statements -- the results (the focal radial coordinate r_f and azimuth
  phi_f) are never assigned or used. The Bessel-function I0/I1/I2 form described in
  the comment at lines 178-181 (which would consume r_f and phi_f) is not the
  implementation; the implementation is a pure 2-D FFT. The two lines are vestigial
  from the abandoned Bessel-integral path.
- **Why it matters**: Harmless to results (no side effects) but it is two wasted
  O(N_focal^2) array allocations per call and, more importantly, it visually
  implies a polar/Bessel evaluation that does not exist, misleading future
  maintainers about how the integral is computed.
- **Regime**: Every call to richards_wolf_focus (always executed); purely a
  clarity/micro-perf issue, no numerical impact.
- **Evidence**:
  ```
  # vector_diffraction.py:126-128
  Xf, Yf = np.meshgrid(x_f, y_f)
  np.sqrt(Xf ** 2 + Yf ** 2)
  np.arctan2(Yf, Xf)
  ```
- **Proposed fix**: Delete lines 127-128 (and the now-orphaned Xf,Yf at 126 if
  nothing else uses them; Xf/Yf are not referenced again, so all three of 126-128
  can be removed). Optionally update the comment block at 178-185 to state plainly
  that the integral is evaluated by FFT.

### [P3] F-35. load_material catalog branch silently overwrites built-in GLASS_REGISTRY entries (auto-runs at import)

- **Location**: `lumenairy/user_library.py:154-156`
- **Category**: state-mutation
- **Dimension**: state-mutation-aliasing
- **Votes**: real 3/3, new 3/3
- **What's wrong**: The 'catalog' branch does `GLASS_REGISTRY[mat_name] =
  (data['shelf'], data['book'], data['page'])` with no collision check or warning.
  The sibling 'fixed' branch routes through register_fixed_glass
  (user_library.py:194), which the known P1-GL-2 audit hardened to WARN on
  overwrite (lines 249-257). The catalog path was left unguarded -- a classic
  fix-N-miss-N+1 sibling gap. Worse, load_all_materials() runs automatically at
  import (line 619), so this overwrite fires every process start.
- **Why it matters**: A user who saves a catalog material named after a built-in
  (e.g. 'N-BK7', confirmed present in GLASS_REGISTRY as
  ('specs','SCHOTT-optical','N-BK7')) silently clobbers the built-in's registry
  tuple at every import, with no warning -- get_glass_index('N-BK7', wl) then
  dispatches to the user's shelf/book/page instead of the SCHOTT entry. This is
  exactly the silent-clobber failure mode P1-GL-2 was raised to eliminate, but on
  the parallel catalog code path. It is cache/registry poisoning of a shared
  module-global mutable.
- **Regime**: Any install where a user has saved a catalog-type material
  (save_material with shelf/book/page) whose name collides with a built-in glass
  name. Realistic when users curate a personal catalog mirroring vendor names.
- **Evidence**:
  ```
  if data['type'] == 'catalog':
      from .glass import GLASS_REGISTRY
      GLASS_REGISTRY[mat_name] = (data['shelf'], data['book'], data['page'])
      # no `if mat_name in GLASS_REGISTRY: warn(...)` unlike register_fixed_glass
  ```
- **Proposed fix**: Mirror register_fixed_glass's overwrite guard: before the
  assignment, `if mat_name in GLASS_REGISTRY: warnings.warn(f"load_material:
  overwriting existing GLASS_REGISTRY entry {mat_name!r}", UserWarning,
  stacklevel=2)`. Consider also refusing to overwrite a non-user built-in entry
  during the silent auto-load path, or namespacing user catalog entries.

### [P3] F-36. lumenairy_context docstring documents max_ram=0 as the override-clearing escape hatch, but set_max_ram now rejects 0

- **Location**: `lumenairy/_context.py:170-174`
- **Category**: api-drift
- **Dimension**: state-mutation-aliasing
- **Votes**: real 3/3, new 3/3
- **What's wrong**: The docstring tells users to pass max_ram=0 to explicitly clear
  an existing RAM-budget override ('use max_ram=0 if you need to explicitly clear an
  existing override -- handled as bytes, which set_max_ram will store as 0 bytes').
  But memory.py:112-117 (the v4.14 P3-#18 fix) makes set_max_ram raise ValueError
  for any value <= 0. In lumenairy_context, max_ram=0 is not None (line 227), so it
  enters new_state and apply_globals (memory.py path) calls set_max_ram(0), which
  raises ValueError on context entry.
- **Why it matters**: The documented escape hatch is dead -- following the
  docstring crashes the with-block on entry (and, combined with the entry-not-in-
  finally bug above, can also leak earlier-applied knobs). A user trying to clear an
  override per the docs gets an exception instead of the promised behavior.
- **Regime**: Any caller who follows the docstring and passes max_ram=0 to clear an
  override inside a context. Confirmed: set_max_ram(0) raises 'value must be
  positive (got 0)'.
- **Evidence**:
  ```
  # Docstring: use max_ram=0 if you need to explicitly clear an existing override
  #   -- handled as bytes, which set_max_ram will store as 0 bytes
  # memory.py:112: if value <= 0:
  #     raise ValueError(f"set_max_ram: value must be positive (got {value!r}). ...")
  ```
- **Proposed fix**: Either (a) update the docstring to remove the max_ram=0 claim
  and document that there is no in-context way to revert an override to auto-detect
  (or add a dedicated sentinel like max_ram='auto'), or (b) special-case max_ram in
  lumenairy_context to translate a caller-requested clear into set_max_ram(None).
  Keep the doc and the validator in sync.

### [P3] F-37. call_progress docstring promises 'any exception is suppressed' but only a narrow tuple is caught

- **Location**: `lumenairy/progress.py:76-93`
- **Category**: api-drift
- **Dimension**: state-mutation-aliasing
- **Votes**: real 3/3, new 3/3
- **What's wrong**: The docstring states 'Progress reporting must never break the
  underlying computation, so any exception in the callback is suppressed.' The
  implementation only catches (TypeError, ValueError, RuntimeError, AttributeError,
  KeyError, IndexError, OSError). A user callback that raises e.g.
  ZeroDivisionError, NameError, ArithmeticError, or any non-listed exception
  propagates out of call_progress and aborts the long-running computation it was
  driving.
- **Why it matters**: These progress hooks fire from apply_real_lens_traced,
  propagate_through_system, through_focus_scan, and tolerance sweeps -- exactly the
  slow paths where losing the whole run to a buggy one-line callback is most
  costly. A common real callback bug (e.g. pct = done/total with total==0 ->
  ZeroDivisionError, or a typo'd variable -> NameError) crashes the entire
  simulation despite the documented contract.
- **Regime**: Any caller-supplied progress callback whose body can raise an
  exception outside the caught tuple -- ZeroDivisionError and NameError are the most
  common in ad-hoc progress lambdas/closures.
- **Evidence**:
  ```
  """...any exception in the callback is suppressed..."""
  if cb is None:
      return
  try:
      cb(stage, float(fraction), message)
  except (TypeError, ValueError, RuntimeError, AttributeError,
          KeyError, IndexError, OSError):
      pass   # ZeroDivisionError / NameError / etc. propagate
  ```
- **Proposed fix**: Either broaden the except to `except Exception:` (keeping
  KeyboardInterrupt/SystemExit propagating, which a bare except Exception already
  does) to match the documented 'never break' contract, or narrow the docstring to
  enumerate exactly which exception classes are swallowed.

### [P3] F-38. create_gaussian_beam normalize='peak' ignores (x0,y0) offset; returned field is NOT unit-peak when the beam is off-center

- **Location**: `lumenairy/sources/core.py:301-305`
- **Category**: convention
- **Dimension**: sources-core-fresh
- **Votes**: real 3/3, new 3/3
- **Auditor cross-check (sources/core.py)**: MIXED / REFUTED for this item. "The
  create_gaussian_beam normalize=peak off-center claim was NOT reproduced
  (off-center beam x0=20um still returned unit peak |E|max=1.0); treat that one as
  REFUTED/UNCERTAIN." (Note: the sibling sigma-guard claim in F-39 was
  CONFIRMED-REPRO; this peak-normalization claim should be treated as
  PLAUSIBLE-UNVERIFIED at best and re-reproduced before any change.)
- **What's wrong**: The 'peak' branch is a bare `pass` whose comment asserts
  'already peak == 1 from exp(0) at the centre'. But the analytical peak exp(0)=1 is
  only sampled when a grid point coincides exactly with (x0,y0). For any sub-pixel-
  offset beam the maximum SAMPLED amplitude is exp(-d^2/(2 sigma^2)) < 1, where d is
  the distance from (x0,y0) to the nearest grid node. The docstring promises 'a
  unit-peak amplitude field', and the sibling create_hermite_gauss (line 454-457)
  and create_laguerre_gauss (line 606-609) correctly divide by the ACTUAL
  np.abs(E).max(). create_gaussian_beam alone trusts the analytic peak.
- **Why it matters**: A caller requesting normalize='peak' for an off-axis Gaussian
  (e.g. a sub-aperture probe beam placed at x0!=0, or a beam whose center is between
  grid nodes on an odd/even-N grid) gets an amplitude scaled below 1, silently
  violating the documented contract and breaking cross-comparison with the HG/LG
  helpers that DO honour the actual peak. Power-budget or contrast calculations that
  assume peak==1 are biased low.
- **Regime**: Tightly-sampled Gaussian (sigma ~ a few dx) with a non-zero x0/y0
  offset, or any off-center beam on a grid whose pixel centers do not land on
  (x0,y0). With sigma=2*dx and a half-pixel offset the peak is ~0.969 (a 3%
  deviation); error grows as sigma/dx shrinks.
- **Evidence**:
  ```
  E = xp.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))
  E = E.astype(target_dtype)
  if normalize == 'peak':
      pass  # already peak == 1 from exp(0) at the centre
  ```
- **Proposed fix**: Mirror the HG/LG helpers: in the 'peak' branch compute pk =
  float(xp.abs(E).max()); if pk > 0: E = E / pk. This makes the returned field
  exactly unit-peak regardless of (x0,y0) and grid parity, and brings the three
  mode-family factories into agreement. (Auditor flagged this specific claim as not
  reproduced; verify the off-center peak deficit before patching.)

### [P3] F-39. create_gaussian_beam silently accepts non-physical sigma (sigma<0 yields same beam as |sigma|; sigma=0 yields a NaN-laced field) with no guard

- **Location**: `lumenairy/sources/core.py:301`
- **Category**: numerical
- **Dimension**: sources-core-fresh
- **Votes**: real 3/3, new 3/3
- **Auditor cross-check (sources/core.py)**: CONFIRMED-REPRO. "Gaussian sigma=0 ->
  NaN-laced field and sigma<0 silently == sigma>0 (no guard): CONFIRMED-REPRO."
- **What's wrong**: The Gaussian envelope is exp(-r^2/(2*sigma**2)); sigma enters
  only as sigma**2, so a NEGATIVE sigma silently produces the identical beam as
  +|sigma| (the caller error is masked), and sigma=0 produces division by zero ->
  exp(-inf)=0 everywhere except the exact center where exp(-0/0)=NaN. There is no
  validation of sigma anywhere in create_gaussian_beam. This is the same class of
  footgun the codebase explicitly guards against in the function this one is the
  backend for: create_fiber_mode (line 999-1014) raises on
  mode_field_diameter<=0 precisely because 'mode_field_diameter <= 0 silently
  flips sigma's sign ... or hits divide-by-zero in sigma = w0/sqrt(2)' -- yet the
  underlying create_gaussian_beam(sigma=...) is reachable directly (and via
  Source.gaussian, which also does not validate w0) with no such guard.
- **Why it matters**: A direct create_gaussian_beam(sigma=-x) call (e.g. from a
  sign error in a derived width, or w0 computed from a negative quantity in user
  code) returns a plausible-looking valid Gaussian with no warning, masking the
  bug; sigma=0 returns a NaN-poisoned array that contaminates every downstream
  power integral and FFT. The library's own design intent (per the fiber_mode guard
  comment) is to reject these loudly.
- **Regime**: Any caller that computes sigma from a possibly-negative or
  possibly-zero quantity (sigma derived from a fit, a difference of widths, or
  w0/sqrt(2) with a bad w0) and passes it directly to create_gaussian_beam or
  Source.gaussian.
- **Evidence**:
  ```
  E = xp.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))   # no check that sigma is finite and > 0
  ...
  # (create_fiber_mode, by contrast, explicitly guards its scale param:)
  if not np.isfinite(mfd_f) or mfd_f <= 0.0:
      raise ValueError(... 'mode_field_diameter must be a positive finite number')
  ```
- **Proposed fix**: Add, right after _validate_grid_params: `if not (np.isfinite(sigma)
  and sigma > 0): raise ValueError("create_gaussian_beam: sigma must be a positive
  finite number [m]; got {sigma}.")`. Add the analogous w0 guard in Source.gaussian
  (line ~2499) so the wrapper path is equally protected.

### [P3] F-40. create_tilted_plane_wave applies kx=k0 sin(angle_x), ky=k0 sin(angle_y) independently with no guard against the evanescent regime sin^2(ax)+sin^2(ay) > 1

- **Location**: `lumenairy/sources/core.py:692-693`
- **Category**: physics
- **Dimension**: sources-core-fresh
- **Votes**: real 3/3, new 3/3
- **What's wrong**: The transverse phase ramp treats kx=k0 sin(angle_x) and ky=k0
  sin(angle_y) as independent direction cosines. For a real propagating plane wave
  the longitudinal wavenumber is kz=sqrt(k0^2 - kx^2 - ky^2), which requires
  sin^2(angle_x)+sin^2(angle_y) <= 1. When both angles are large simultaneously
  (e.g. angle_x=angle_y=60deg gives sin^2+sin^2=1.5), the implied (kx,ky) lies
  OUTSIDE the propagating cone (k_perp > k0), i.e. an evanescent / non-physical
  plane wave, but the function emits the ramp silently with no warning and pairs
  with the forward exp(+ikz) convention as if it propagates. The single-axis case
  is exact and tested (test_sources.py TestTiltedPlaneWave only exercises
  angle_y=0).
- **Why it matters**: Off-axis imaging sweeps that combine large independent x and y
  field angles can unknowingly request a transverse wavevector exceeding k0;
  downstream ASM/Fresnel propagation of such a field is meaningless (the
  band-limited propagator zeros it as evanescent, per asm.py), so the user gets an
  essentially blank or aliased result with no indication that the requested field
  angle was non-physical. The library validates wavelength/dx/N/sigma scale params
  loudly elsewhere but not this geometric constraint.
- **Regime**: Multi-field off-axis analysis with simultaneous large angle_x AND
  angle_y (or angle_*_deg), e.g. corner field points of a wide-field system where
  each axis is ~>45deg.
- **Evidence**:
  ```
  k0 = 2 * np.pi / wavelength
  phase = k0 * (np.sin(angle_x) * X + np.sin(angle_y) * Y)
  E = (amplitude * np.exp(1j * phase)).astype(_resolve_complex_dtype(dtype))
  ```
- **Proposed fix**: After resolving angle_x/angle_y, compute s2 =
  np.sin(angle_x)**2 + np.sin(angle_y)**2 and warn (or raise) when s2 >= 1.0: the
  requested combined field angle implies an evanescent transverse wavevector
  (k_perp >= k0) that will not propagate. Optionally document that angle_x/angle_y
  are projected (x-z / y-z plane) angles and the pair must satisfy
  sin^2(ax)+sin^2(ay) < 1.

## Per-dimension health table

| Dimension | Raw | Confirmed | One-line summary |
|-----------|-----|-----------|------------------|
| optimize-subsystem | 4 | 3 | Merit-mode math/gradient routing mostly sound; off-by-one OSA exclusion keeps defocus, non-NaN-safe argmax, and dead scale_floor on the default FD path. |
| asymptotic-propagators | 3 | 2 | Framework internally consistent; closed-form ell=0 evaluates output LG basis at absolute image coord (not saddle sigma), corrupting off-axis defocus/spherical; plus a misleading HF-prefactor comment. |
| propagator-family | 1 | 1 | Non-ASM propagators sound except GBD: reconstructed transverse wavefront curvature is the complex conjugate of the canonical exp(+ikz) field after free propagation (intensity correct, phase wrong). |
| elements-doe-grating | 4 | 3 | DOE/grating/freeform/BSDF mostly sound; batched-incidence BSDF crash, silent wrong-order grating sweep, and an energy-conservation docstring the truncated T_eff cannot keep. |
| lens-kernels-jax | 1 | 1 | NumPy/JAX lens kernels sign-correct; JAX traced/Maslov wave grid uses 'ij' indexing, transposing the phase screen vs E_in (y,x) for non-symmetric prescriptions. |
| raytrace-seidel-paraxial | 3 | 3 | Seidel core correct/pinned; compute_pupils crashes (UnboundLocalError) on internal-stop systems, cascading into phantom-z make_ray chief launch in ray/OPD fans; plus an invariant docstring formula error. |
| raytrace-trace-world-bundles | 1 | 1 | Refraction/intersection/world build sound; the LOCAL coord-break tilt transform uses +theta where the world->local inverse needs -theta, inverting every tilt vs trace_world. |
| analysis-opd-wavefront | 2 | 2 | Two API-drift doc items: wave_opd_1d omits focal_length/f_ref and a stale ref-sphere note; CausticDiagnostic det_J doc says 'sign' but stores the raw determinant. |
| analysis-detector-field | 3 | 2 | Beam-stats/distortion sound; detector non-integer pixel ratio loses ~25-30% photons via spurious box-mean rescale; ImagePlaneWFE Strehl uses un-piston-removed RMS. |
| analysis-coronagraph-ao | 3 | 3 | Phase-retrieval/Strehl/AO sound; tolerancing_sweep recomputes the Strehl denominator per perturbed pupil (v5.2.5 bug missed here); NumPy/JAX through_focus best_focus parity gap; phase_shift_extract LSQ claim only valid for equispaced shifts. |
| io-prescriptions | 3 | 3 | Field translations round-trip; mirror radius fed into apply_mirror with the wrong (Welford vs wave-side) sign convention; split_prescription_at_mirrors drops fold gaps; exporters ignore stop_index, relocating the stop on round-trip. |
| algebra-subsystem | 1 | 1 | ABCD/element math correct; the documented annular inner=D/2 default is dead code, so default annular apertures have no central obstruction. |
| backend-dispatch | 2 | 2 | FFT norm/dtype/dispatch sound; RandomState integers/uniform/normal JAX branches hardcode 32-bit defaults vs the NumPy/CuPy 64-bit defaults (the choice int64 fix was never extended). |
| fft-infra-deep | 2 | 2 | fft_infra core sound; RS returns a view into the inverse ping-pong buffer (data corruption on z-sweeps); warmup_fft_plans keys plans on _available_cpus() not the live FFTW_THREADS. |
| numerical-edge-cases | 1 | 1 | Codebase well-hardened against sqrt/arcsin/div hazards; one convention inconsistency: surface_sag_biconic returns 0.0 outside the conic domain while surface_sag_general returns NaN. |
| glass-coatings-physics | 1 | 1 | TMM/coatings physics clean; one low-precision/mis-poled BaF2 bundled Sellmeier row (~0.4-0.5% index error in the visible) on minimal installs. |
| vector-diffraction | 2 | 2 | Richards-Wolf/Debye-Wolf physics correct; the only application callsite (the GUI dock) uses a fabricated signature and wrong tuple unpacking, so the feature is dead; plus a dead-code polar-coord block. |
| convention-cross-file | 0 | 0 | All CONVENTIONS.md section-7 sign conventions verified consistent across call sites; two candidate contradictions investigated and refuted; dimension clean. |
| state-mutation-aliasing | 4 | 4 | lumenairy_context applies globals outside try/finally (leaks on entry failure); load_material catalog branch clobbers built-in glass entries at import; max_ram=0 escape hatch now raises; call_progress over-promises exception suppression. |
| sources-core-fresh | 3 | 3 | Field formulas physically correct; create_gaussian_beam 'peak' ignores offset and lacks a sigma guard; create_tilted_plane_wave lacks an evanescent-regime guard for combined large angles. |

## Rejected-this-pass appendix

| Title | Locus | Severity | Votes (real/new) | Why rejected |
|-------|-------|----------|------------------|--------------|
| MatchIdealThinLensMerit subtracts a full ideal sphere from the driver's sphere-subtracted residual OPD (double-counts focusing wavefront) | merit_terms.py:258-259 (with driver.py:712) | P2 | 0/3 real, 3/3 new | physics/math correctness real=false; contract/test-pin/novelty real=false; operational impact real=false. Failed the >=2/3 real bar. |
| JAX aberration_tensor_lg00_jax w_o auto-pick lacks the [1e-9, 1.0] clamp the NumPy path applies -> backend parity divergence near caustics | asymptotic_jax_twin.py:355-359 | P3 | 1/3 real, 2/3 new | physics/math real=true but contract/test-pin/novelty real=false and new=false; operational impact real=false. Failed the >=2/3 real bar. |
| GaussianBSDF.sample / HarveyShackBSDF.sample antipode-flip rays into the wrong direction at oblique incidence instead of rejecting sub-surface samples | bsdf.py:336-337 (and 454-455) | P3 | 3/3 real, 1/3 new | Strong on real correctness but new=false (1/3); judged not genuinely-new this pass. Failed the >=2/3 new bar. |
| apply_detector integer-branch x_det/y_det offset half a pixel from the integrated block centroid | detector.py:131-132 | P3 | 1/3 real, 3/3 new | coordinate-registration real=true on one lens only (1/3 real overall); contract/test-pin real=false; operational impact real=false. Failed the >=2/3 real bar. |

## Completeness critic

The following is the completeness critic's gap analysis, included verbatim.

> I have everything needed. The `coupling_efficiency` /
> `coupling_efficiency_vector` (fiber overlap-integral mode matching) functions
> in strehl.py were NOT covered by either the prior PSF-MTF-Zernike pass (which
> covered strehl_ratio/marechal/phase_integral as denominators) nor this
> coronagraph-ao pass. And chebyshev_fit_2d (derivative Vandermonde for
> sag-slope used in raytrace of Chebyshev freeform surfaces) is entirely
> unaudited. I now have a complete, evidence-backed gap map.
>
> # Top Coverage Gaps - lumenairy v5.4.5 Deep Audit
>
> Two passes (the prior `AUDIT_V5_4_5_2026_05_26_DEEP` and this one) together
> hit the propagator cores, the lens/element kernels, the
> Seidel/paraxial/trace raytrace stack, the optimize subsystem, most of
> analysis, io-prescriptions, algebra, backend, glass/coatings, and
> sources/core. But roughly 40k of the 110k LOC sit in clusters that *neither*
> pass examined with physics rigor. The gaps below are ranked by likelihood of
> hiding a genuine correctness bug, with concrete files and bug-classes.
>
> ## GAP 1 - Subaperture / patch-stitching coherent recombination (HIGH; warrants a finder round)
> `lumenairy/propagators/subaperture.py` (536 LOC) and `mhs.py` (651 LOC) were
> given only a one-line "guards are reasonable" treatment in propagator-family;
> the actual **partition-of-unity recombination math was never checked**. The
> hot spots: `combine_patch_fields` (lines ~116-214) divides the
> coherently-summed patch fields by a `weight_total` that is floored to 1.0
> below 1e-12 (line 213) - meaning any output pixel covered by *less than full
> unity weight* (gap regions, edge patches, or where the smooth `patch_window`
> axis-windows do not sum to 1) gets an un-normalized amplitude, i.e. an
> **amplitude seam / energy discontinuity at patch boundaries**. The window is
> a product of two 1-D `axis_window`s, and a product-of-1-D partition of unity
> is *not* itself a partition of unity in 2-D unless each axis individually
> sums to 1 everywhere - worth verifying numerically. Bug-class to hunt:
> partition-of-unity normalization error, patch-center vs. patch-coordinate
> mismatch (the v5.2 note at lines 134-145 about windows "centred on" different
> coords is exactly the kind of half-fixed convention drift that leaves a
> latent bug), and coherent-phase registration between patches. This is a
> mainstream HFPI/GBD subaperture path.
>
> ## GAP 2 - Asymptotic LG-mode machinery beyond the aberration tensor (HIGH; warrants a finder round)
> This pass found the `ell=0` saddle-coordinate bug in
> `asymptotic_aberration_tensor.py`, but three sibling files carrying the same
> LG-mode/saddle/Maslov physics were **not** opened: `asymptotic_modes.py` (769
> LOC - LG basis construction, mode projection), `asymptotic_maslov.py` (568
> LOC - branch tracking), and `asymptotic.py` (628 LOC - the top-level driver).
> Given that the *one* file that was read had a silent off-saddle evaluation
> bug in its closed-form branch, the untested `p>=1, ell!=0` channels and the
> mode-projection normalization in `asymptotic_modes.py` are prime suspects for
> the **same class** of "absolute image coord vs. sigma-offset" error and for
> LG normalization-constant drift. Bug-class: saddle-local vs.
> absolute-coordinate confusion, LG amplitude/normalization constants, Maslov
> index accumulation across multiple caustics (the single-caustic case was
> verified; multi-caustic was not).
>
> ## GAP 3 - `propagators/system.py` sequential dispatcher + grid tracking (HIGH; warrants a finder round)
> The 1320-LOC sequential orchestrator `propagate_through_system` / `evaluate`
> was never audited as a unit - propagator-family audited the individual
> propagators in isolation. The risk concentrates in the **inter-element grid
> bookkeeping** (lines ~326-406): `current_dx`/`current_dy` are threaded through
> Fresnel/SAS auto-resampling, and the code admits in comments that "axes
> converge to current_dx since resample_field produces a square grid;
> current_dy stays in sync" (line 358) - this is an **anamorphic-grid hazard**:
> a Fresnel or SAS step on a `dx != dy` grid is resampled back through a
> square-grid resampler, which silently destroys the dy track. Also
> `evaluate(output_grid=...)` (lines 632-654) *warns and ignores* the requested
> resampling - a documented-but-unimplemented feature that returns the
> wrong-shaped field. Bug-class: dx/dy desync across mixed-method element
> chains, dtype drift across element boundaries, element-type dispatch table
> omissions (the `apply_*` import list at lines 29-46 should be cross-checked
> against every `type` string the codegen and UI can emit - a missing case
> would route to a wrong default or raise).
>
> ## GAP 4 - Coupling-efficiency / mode-overlap integrals in `analysis/strehl.py` (MEDIUM-HIGH; warrants a finder round)
> `coupling_efficiency` (line 206) and `coupling_efficiency_vector` (line 444)
> compute fiber/mode overlap integrals - the normalized |integral E.M*|^2 /
> (integral |E|^2 integral |M|^2) physics that drives every fiber-coupling and
> mode-matching result. The prior PSF/Strehl pass covered
> `strehl_ratio`/`strehl_marechal`/`strehl_phase_integral` (as Strehl
> denominators) but **not the coupling integrals**. Bug-class: missing
> complex-conjugate on one factor (gives wrong sign/phase sensitivity), missing
> dx^2 area element in one of the three integrals (gives a dimensionally-wrong
> but plausible-looking number), or normalization that assumes both fields are
> on the same grid/pitch without checking. This is exactly the kind of overlap
> integral where a single missing `np.conj` is invisible to a real-valued test
> on a centered Gaussian.
>
> ## GAP 5 - `_math/chebyshev.py` derivative Vandermonde + Chebyshev-freeform raytrace (MEDIUM)
> `chebyshev_derivative_vandermonde` /
> `chebyshev_second_derivative_vandermonde` / `chebyshev_fit_2d`
> (chebyshev.py, 404 LOC) feed the Chebyshev-freeform **sag-slope** used in ray
> refraction (surface normals) and in the `chebyshev_fit_dock`.
> Elements-doe-grating audited *Forbes* Q-bfs/Q-con freeforms but explicitly
> did not touch the Chebyshev branch. Bug-class: the recurrence for T_k'(u) and
> the chain-rule du/dx scaling (a Chebyshev domain is normalized to [-1,1], so
> the derivative Vandermonde must carry a `2/(x_max-x_min)` Jacobian - a missing
> or doubled domain-scale factor silently tilts every Chebyshev surface normal,
> corrupting raytrace at those surfaces while leaving the *sag* fit itself
> correct). Worth a focused check of the derivative-domain scaling only.
>
> ## GAP 6 - `analysis/ghost.py` non-sequential stray-light (MEDIUM)
> 942 LOC, untouched by both passes. `_fresnel_R_normal` (line 553) uses
> normal-incidence-only Fresnel reflectance - fine as a documented
> approximation for ghost *energy*, but `ghost_analysis` /
> `non_sequential_stray_light` / `retrace_ghost_path` do double-bounce ray
> retracing whose **path enumeration** (`enumerate_ghost_paths`, line 213) and
> **direction-reversal bookkeeping** are exactly where sign/parity bugs live.
> The module even documents a v5.4.0->v5.4.1 history of a "direction-blind root
> pick" bug (lines 569-576) that was promoted into `_intersect_surface`; that
> is a tell that the surrounding retrace logic is fragile. Bug-class: ghost
> path index off-by-one, reflected-leg sign of the surface normal / direction
> cosine on the backward leg, OPL accumulation sign on reflected segments.
>
> ## GAP 7 - `polychromatic.py` Strehl reference + JAX raytrace twin parity (MEDIUM)
> Two narrower items. (a) `polychromatic_psf`'s per-wavelength Strehl reference
> (polychromatic.py:392-394) uses `diffraction_limited_peak` recomputed per
> wavelength at a fixed common `image_distance` - this is the *same
> nominal-vs-perturbed-denominator* bug-class this pass confirmed in
> `tolerancing_sweep`; the chromatic-defocus-aware denominator should be checked
> for the analogous inconsistency (intentional per the comment, but unverified).
> (b) `raytrace/jax_trace.py` (1631 LOC) was only referenced for its
> `meshgrid('ij')` lens-grid issue; its **refraction/intersection twin**
> (`_refract_jax`, `_intersect_jax`, `_transfer_jax`) was never diffed against
> the NumPy `intersection.py` reference for sign parity - and since this pass
> found a coord-break sign inversion in NumPy `intersection.py`, the question of
> whether `jax_trace` reproduces *or diverges from* that NumPy behavior on the
> shared refraction path is open (jax_trace explicitly refuses coord-breaks, so
> the specific bug doesn't propagate, but refraction/transfer parity is
> unverified).
>
> ## GAP 8 - Large UI docks driving physics (LOW-MEDIUM; spot-check only, not a full finder round)
> ~31k LOC of UI. Both passes only touched `richards_wolf_dock.py` (found the
> fabricated-signature bug). The pattern there - a dock calling a physics API
> with the wrong signature/return-unpacking so the feature is silently dead - is
> **highly likely to recur** in the other docks that wrap recently-changed
> APIs: `optimizer_dock.py` (1784), `waveoptics_dock.py` (2735),
> `coronagraph_dock.py`, `ao_dock.py`, `phase_retrieval_dock.py`,
> `interferometry_dock.py`, `thin_grating_dock.py`. Bug-class:
> API-call-signature drift (dock written against an older function signature),
> wrong tuple/object unpacking of a result, and unit mismatches (dock passes mm
> where the API wants m). This does not need a rigorous physics finder - it
> needs a mechanical "every dock callsite vs. current public signature" diff,
> which would be fast and high-yield.
>
> ## What was explicitly NOT looked at (honest list)
> - **subaperture.py / mhs.py recombination math** - only "guards reasonable."
> - **asymptotic_modes.py, asymptotic_maslov.py, asymptotic.py** - never opened;
>   only the aberration_tensor and canonical_fit siblings were.
> - **propagators/system.py (1320 LOC) and propagation.py dispatch** - the
>   element-chain orchestrator and its grid/dtype tracking.
> - **analysis/ghost.py (942) entirely; coherence.py (213) entirely**
>   (mutual_coherence, koehler_image, extended_source_image -
>   extended-source/partial-coherence physics).
> - **analysis/strehl.py coupling integrals; analysis/core.py.**
> - **_math/chebyshev.py derivative Vandermondes.**
> - **raytrace/jax_trace.py refraction/transfer twin (only the lens-grid axis issue).**
> - **All UI docks except richards_wolf_dock.**
> - **storage.py (1593) atomicity** - touched by the *prior* pass
>   (io-storage-atomicity), so not a new gap, but its interaction with the new
>   Zarr/SWMR paths is worth noting as adjacent.
>
> ## Recommended follow-up finder rounds (priority order)
> 1. **subaperture-stitching + asymptotic-modes** (GAPs 1, 2) - highest
>    physics-bug probability; both are coherent-field math with a demonstrated
>    sibling bug nearby.
> 2. **system-orchestrator-grid-tracking** (GAP 3) - mainstream multi-element
>    path, anamorphic + dtype hazards.
> 3. **overlap-integrals + chebyshev-derivatives** (GAPs 4, 5) - narrow, fast,
>    single-missing-factor bug class.
> 4. **ghost-stray-light + polychromatic/jax-parity** (GAPs 6, 7) - medium.
> 5. **UI-dock-signature-sweep** (GAP 8) - mechanical, not a physics finder;
>    cheap and high-yield given the richards_wolf precedent.

### Recommended next finder round (distilled)

- Subaperture / mhs partition-of-unity recombination math (GAP 1) -- highest
  physics-bug probability.
- Asymptotic LG-mode siblings: asymptotic_modes.py, asymptotic_maslov.py,
  asymptotic.py (GAP 2) -- same saddle/normalization bug class as the confirmed
  aberration_tensor defect.
- propagators/system.py sequential orchestrator grid/dtype tracking (GAP 3) --
  mainstream multi-element anamorphic path.
- analysis/strehl.py coupling_efficiency / coupling_efficiency_vector overlap
  integrals (GAP 4) -- single-missing-conj / missing-dx^2 bug class.
- _math/chebyshev.py derivative Vandermonde domain-scale Jacobian (GAP 5).
- analysis/ghost.py non-sequential stray-light retrace sign/parity (GAP 6) and
  polychromatic.py / jax_trace refraction-twin parity (GAP 7).
- Mechanical UI-dock-signature sweep across all docks (GAP 8).

## Recommended remediation bundling

**Fold into the pending v5.4.6 patch (low-risk, high-confidence -- crashes,
dead code, docstring/API):**

- **F-2 (seidel compute_pupils UnboundLocalError)** -- SHIP PRIORITY. Two-line
  assignment fix; CONFIRMED-REPRO; crashes every internal-stop system.
- **F-3 (RS view-into-ping-pong-buffer)** -- SHIP PRIORITY. One-line `.copy()`;
  CONFIRMED-REPRO; silent data corruption on RS z-sweeps.
- **F-8 (ray_fan/opd_fan phantom-z make_ray + missing except)** -- pairs with
  F-2; fix together (kwarg z launch + add UnboundLocalError to except).
- **F-16 (annular aperture dead-code default)** -- CONFIRMED-REPRO; small
  control-flow reorder + test.
- **F-22 (GaussianBSDF batched-incidence crash)** -- CONFIRMED-REPRO;
  axis-fix one-liner.
- **F-34 (vector_diffraction dead polar-coord lines)** -- delete 2-3 lines.
- **F-21 (HF-prefactor misleading comment)**, **F-25 (optical_invariant
  docstring)**, **F-24 (grating energy-conservation docstring)**, **F-26
  (wave_opd_1d params)**, **F-27 (CausticDiagnostic det_J doc)**, **F-36
  (max_ram=0 docstring)**, **F-37 (call_progress docstring/except)**, **F-23
  (grating out-of-window order)** -- pure docstring / validation-guard fixes,
  zero physics risk.
- **F-17 (lumenairy_context entry-not-in-finally)** -- move one line inside
  try; verified live; small and safe.
- **F-35 (load_material catalog clobber warning)** -- add the same overwrite
  guard as register_fixed_glass.
- **F-30 / F-31 (RandomState int64/float64 JAX parity)** -- mirror the existing
  choice int64 pin; low risk, restores documented cross-backend parity.
- **F-32 (warmup_fft_plans thread-key mismatch)** -- one-line default change.
- **F-33 (BaF2 Sellmeier coefficients)** -- data-table swap to the authoritative
  Li/Malitson row + n_d cross-check comment.
- **F-39 (create_gaussian_beam sigma guard)** -- CONFIRMED-REPRO; add the same
  positive-finite guard create_fiber_mode already uses.
- **F-29 (exporters ignore stop_index)** -- default stop_surface from the
  prescription; add round-trip test.

**Queue for v5.5 (architectural / physics-sign needing a pinned repro first):**

- **F-1 (GBD conjugated wavefront curvature)** -- SHIP PRIORITY for v5.5 as the
  highest-impact physics bug. CONFIRMED-REPRO, but the fix changes the GBD field
  convention and must land with a phase-comparison regression test against ASM
  (lambda/100) before release; do not rush into v5.4.6 alongside the crash
  fixes.
- **F-6 (asymptotic ell=0 saddle-coordinate)** -- physics correctness; route
  multi-p / off-axis ell=0 through the sigma-grid path with a pinned numeric
  check.
- **F-7 (JAX lens-grid 'ij' transpose)** -- JAX-parity correctness; needs the
  cylindrical/decentered regression test that would have caught it.
- **F-9 (local coord-break tilt sign inversion)** -- physics-sign; needs the
  trace() vs trace_world() axial-ray parity test.
- **F-10 (detector non-integer photon loss)** -- UNCERTAIN magnitude per the
  auditor; build a proper photon-conservation repro before changing the rescale.
- **F-11 (ImagePlaneWFE un-piston-removed Strehl)** -- physics; piston-removal
  change with a defocus-Strehl pin.
- **F-12 (tolerancing_sweep Strehl denominator)** -- physics; mirror the v5.2.5
  MC fix and add a Strehl<=1 pin.
- **F-4 / F-5 (optimize OSA exclusion off-by-one, non-NaN-safe argmax)** --
  merit-semantics change; F-5 is a small nanargmax mirror but both touch
  optimizer behavior and want a pinned merit test.
- **F-13 (phase_shift_extract LSQ for arbitrary shifts)** -- implement the true
  3-parameter normal-equation solve or restrict the docstring + warn.
- **F-14 (codegen mirror radius sign)** -- physics-sign; negate the Welford
  radius into apply_mirror with a concave-mirror .zmx test.
- **F-15 (split_prescription_at_mirrors dropped fold gaps)** -- API addition
  (emit gap_before/gap_after or propagate legs) with a sum-of-gaps test.
- **F-18 (Richards-Wolf dock rewrite)** -- non-trivial dock rewrite + pupil
  synthesis; queue with a dock-worker integration test.
- **F-19 (biconic sag 0.0 vs NaN)** -- convention alignment; small change but
  wants the oblate-biconic rim NaN pin.
- **F-20 (scale_floor inert on default FD path)** -- design decision (install a
  default FD jac vs document); architectural.
- **F-38 (create_gaussian_beam peak off-center)** -- auditor REFUTED the specific
  repro; do NOT patch until the off-center peak deficit is reproduced.
- **F-28 (through_focus best_focus NumPy/JAX parity)** -- small but a contract
  decision (populate both vs document find_best_focus); bundle with F-40.
- **F-40 (tilted_plane_wave evanescent guard)** -- add the s2>=1 warn/raise with
  a documented angle convention.
