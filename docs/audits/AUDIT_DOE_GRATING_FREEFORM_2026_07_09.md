# DOE / Grating / Freeform Audit — 2026-07-09

Scope: full line-level reads of `elements/doe.py` (1,129 — phase
masks, MLA, diffractive lenses, Dammann IFTA, phase/FITS I/O),
`elements/thin_grating.py` (223 — analytical 1-D binary phase grating),
and `elements/freeform.py` (739 — XY/Zernike/Chebyshev freeforms and
the Forbes Q-bfs/Q-con evaluators that consume the Zemax loader's
Q-type output).  Continuation of the `elements/` sweep after the
glass/polarization tranche.  Read-only; the grating Fourier series and
the full Forbes Q-type stack re-derived by hand.

---

## 1. Verdict

**The diffractive and freeform physics is correct.**  The strongest
verifications this pass:

* **`thin_grating_efficiency_1d`** — I re-derived the binary-phase
  transmission Fourier coefficients from
  `t_m = (1/P)∫_0^P t(x) e^{−i2πmx/P} dx`: the m≠0 result
  `(e^{iφr} − e^{iφg})(e^{−i2πmf} − 1)/(−i2πm)` and the m=0 limit
  `f·e^{iφr} + (1−f)e^{iφg}` match the code term-for-term.  Energy
  conservation `Σ|t_m|² = 1` (Parseval, pure phase) confirmed; the
  substrate-reference phase cancels out of every `|t_m|²`.  The P2-07
  complex-`np.sign` fix (explicit unit phasor `z/|z|`, NumPy-1.x-safe)
  is correct.
* **Forbes Q-bfs / Q-con** (`_jacobi_norm_factor`,
  `_shifted_jacobi_eval`, `_q_bfs_eval`, `_q_con_eval`,
  `surface_sag_q_bfs/q_con`) — the full stack verified:
  - the [−1,1]→[0,1] weight-mapping factor `2^(α+β+1)` and the
    orthonormaliser `c_n = 1/√(h_n/2^(α+β+1))` are correct;
    specialisations reproduce the documented
    `c_n = √((2n+3)(n+2)/(n+1))` (Q-bfs, α=β=1) and `√(2n+3)`
    (Q-con, α=0,β=2).  The docstring's own note that the *prose*
    formula was once wrong (spurious factor 8, `(n+1)²`) while the
    code was right is accurate.
  - the 3-term recurrence matches A&S 22.7.1 (shifted, n→k+1
    substitution) coefficient-for-coefficient, with correct
    `P_0 = 1`, `P_1 = ½[(α−β)+(α+β+2)t]` seeds.
  - the sag prefactors `u²(1−u²)` (Forbes 2007 Eq. 13) and `u⁴`
    (Forbes 2010 Eq. 6) are correct; the primary radial (`u²>1`) +
    secondary rectangular clips are sound.
* **Zemax → freeform contract** (cross-ref to `AUDIT_IO_ZEMAX`): the
  loader emits `freeform_type`, `q_bfs_coeffs`/`q_con_coeffs` (× sag
  unit_scale → m) and `r_max` (× unit_scale → m); `surface_sag_freeform`
  consumes exactly those keys with X/Y in metres and `r_max` in metres
  — units consistent end-to-end.  The v4.15.1 P1-F1-2 `r_max`-required
  raise closes the old silent 1.0 m-disc no-op.
* **`create_diffractive_lens` / `create_kinoform` /
  `create_fresnel_zone_plate`** — standard forms; the kinoform
  `sinc²(1/n_levels)` efficiency note, the FZP `floor(r²/λf)` zone
  assignment aligned to π-phase boundaries, and the 4.10
  positive-focal-length raise all check out.
* **`create_microlens_array`** — the vectorised nearest-centre snap
  `round(X/pitch + (n−1)/2)` reproduces the correct lenslet centres
  for both even and odd `n_lenslets`, and the footprint matches.
* **`makedammann2d`** — the annealing schedule, the local-RNG fix
  (no global-state mutation), symmetric target embedding, and the
  legacy-units (`auto`/`um`/`SI`) handling with the >1 m nonsense
  guard are all sound.
* The five freeform guards (P1-NEW-11 `norm_x/y`, P2-08
  `norm_radius`, P1-F1-1 radial clip) are comprehensive.

One real finding and a set of nits follow.

---

## 2. Findings

### DOE-1 (P3) — FITS default save is not recovered by the default load: silent phase loss on round-trip
`save_fits_field(..., split_amp_phase=True)` — the **default** —
writes amplitude to the primary HDU and phase to an `EXTNAME='PHASE'`
image extension.  `load_fits_field(filepath)` — with its **default**
`hdu_phase=None` — reads only the primary HDU, finds a real float32
amplitude (not 3-D, not complex), takes the "amplitude only — assume
zero phase" branch, and returns `E = amp.astype(complex)` with the
phase **silently dropped**.  A save→load round-trip on the documented
defaults therefore loses all phase (the whole quantity of interest for
a diffractive/coherent field), with no error and no warning; the phase
data is in the file but unreachable unless the caller happens to pass
`hdu_phase=1`.  The non-split path (`split_amp_phase=False`) *is*
auto-detected on load (via the `ndim==3 and shape[0]==2` branch), so
the two save formats are asymmetric.  **Fix**: on load, detect the
split format — check for a second HDU with `EXTNAME=='PHASE'` (or
`BUNIT=='radians'`) and reconstruct `amp·exp(i·phase)` automatically,
mirroring the real/imag-stack auto-detection already present.

### Nits
* `create_periodic_phase_mask` documents "nearest-neighbor resampling"
  but `(in_cell / cell_pixel_size).astype(int)` **truncates** (floors
  for the non-negative `mod` result), a left-edge sample with a
  half-pixel bias, not nearest-neighbour (`round`).  Same half-pixel
  class as the HFPI-1 / `from_field` binning nits.  Cosmetic for
  smooth DOEs; `round` would match the docstring.
* `makedammann2d` validates neither `itr` nor `wavsamp`: `itr=0`
  raises `int(nan)` from `2**floor(phasesteps·(itr−it)/itr)` (0/0),
  and `wavsamp=0` makes `ndifordersx = ceil(.../0)` blow up.  A
  one-line `itr >= 1`, `wavsamp > 0` guard would give an actionable
  message instead of a deep-stack crash.
* `save_fits_field` writes each metadata key as `key.upper()[:8]`;
  two metadata keys sharing an 8-char prefix silently collide into one
  FITS card (FITS keyword length limit).  Warn or reject on collision.
* `load_fits_field` resolves `dx = header.get('DX') or
  header.get('PIXSCALE')` — a stored `DX == 0.0` (nonsensical, but)
  falls through to `PIXSCALE` via falsiness rather than being taken
  literally.  The recurring `or`-as-default pattern.
* `surface_sag_q_bfs/q_con` call `_q_bfs_eval(u_sq, m)` /
  `_q_con_eval(u_sq, m)` per coefficient, each re-running the Jacobi
  recurrence 0→m from scratch — O(M²) recurrence steps for M
  coefficients where an accumulating loop is O(M).  The sibling
  `surface_sag_chebyshev` already caches `T_i`/`T_j` by order; the
  Q-type evaluators could share that treatment.  Negligible at typical
  M<20; noted for symmetry.

---

## 3. Coverage statement

Every line of `doe.py`, `thin_grating.py`, and `freeform.py` read.
`surface_sag_general` / `surface_sag_biconic` (in `lenses.py`) were
cross-referenced for the freeform base-sag path (biconic label
already flagged as RT-2).  Not audited here: the remaining `elements/`
modules (`elements.py` 1,159; `bsdf.py` 591; `coatings.py` 725;
`materials.py`; `segment_geometry.py`; `lenses_maslov.py` 3,503;
`eme/`), the `io/` siblings, and `optimize/` — the natural next
tranches.

---

*Audit performed single-context against lumenairy v5.21, 2026-07-09.
Companion docs: `AUDIT_IO_ZEMAX_2026_07_08.md` (the Q-type loader
half of the freeform contract), `AUDIT_GLASS_POLARIZATION_2026_07_08.md`,
`AUDIT_RAYTRACE_CORE_2026_07_08.md`, and the 07-07 set.*
