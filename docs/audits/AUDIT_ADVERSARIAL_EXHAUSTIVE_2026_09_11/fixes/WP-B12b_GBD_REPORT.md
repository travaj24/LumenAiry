# WP-B12b -- the GBD beamlet image leg: the in-line conic-sag copy is deleted
# and the local branch consumes the shared exit-vertex projection

Wave 5 item A2.  Branch `fix/gbd-exit-vertex-projection`, base `1218b24f` (the
head of `fix/wp-b12-fga-reference-plane`).  The finding it acts on is
`fixes/WP-B12_REPORT.md` section 5.1 and open item 1: `gbd.py`'s
`apply_prescription_persurface_to_beamlets` carried its own vertex correction,
an in-line conic-sag expression that WP-B12 measured **15.52 waves wrong on an
A4 / A6 last surface, 71 % of the sag**, and left for a package of its own
because repairing it moves every per-surface-GBD field on an aspheric-last-
surface prescription and needs its own oracle ladder.

---

## 0. Terms used here

* **Last surface** -- the final refracting or reflecting surface of a
  prescription.  `lumenairy.raytrace.trace` stops every ray at its
  intersection with that surface, i.e. at `z = sag(rho)` in the surface's local
  frame; `TraceResult.image_rays` reports that state.
* **Exit-vertex plane** -- the plane `z = 0` through that surface's vertex.
  `TraceResult.at_exit_vertex()` transfers a ray bundle to it along each ray.
* **The shared projection** -- `lumenairy.raytrace.differential.
  _project_to_exit_vertex_plane`, created by WP-B12 as the library's single
  implementation of that operator for the differential primitives' 4x4 state.
  It projects the state AND the Jacobian (`J_v = P J`), takes its sag from the
  package's general surface kernels (`_surface_sag_xy` /
  `_surface_sag_derivatives_xy`), resolves the exit-medium index through
  `raytrace.exit_vertex.resolve_exit_index`, recovers the propagation
  direction through a mirror with `_exit_direction_sign`, and short-circuits
  structurally when the last surface's sag is identically zero.
* **The in-line copy** (deleted here) -- the `_Rl` / `_kl` block that stood
  below the Q loop in `apply_prescription_persurface_to_beamlets` from the
  v5.22 fix to 5.47.0:

  ```python
  _Rl = float(getattr(surfs[-1], 'radius', np.inf))
  _kl = float(getattr(surfs[-1], 'conic', 0.0) or 0.0)
  if np.isfinite(_Rl) and _Rl != 0.0:
      _cl = 1.0 / _Rl
      _r2 = dt.x ** 2 + dt.y ** 2
      _sag = _cl * _r2 / (1.0 + np.sqrt(np.maximum(
          1.0 - (1.0 + _kl) * _cl * _cl * _r2, 0.0)))
  else:
      _sag = np.zeros_like(dt.x)
  t = (z_image - _sag) / Nz2
  ```

* **Fidelity** -- `|<a,b>|^2 / (<a,a><b,b>)` between two complex fields on the
  same grid; 1 means the same field up to a global complex scale.
* **The local branch / the world branch** --
  `apply_prescription_persurface_to_beamlets` has two exits.  The LOCAL branch
  (`world_output_plane is None`) adds an axial leg `z_image` and reconstructs
  on the fixed local x-y grid; the WORLD branch re-traces the base rays through
  the folded prescription in world coordinates and lands them on an arbitrary
  world-frame plane.

---

## 1. The defect, and its four separate failure modes

The in-line copy is not one error, it is four, and only the first was named by
WP-B12.

1. **It evaluates only the CONIC BASE.**  `_surface_sag_xy` dispatches to a
   freeform kernel, a biconic kernel or the general conic-plus-even-aspheric
   kernel, and honours a field-frame decenter / tilt / `sag_callable`.  The
   in-line copy reads `radius` and `conic` and nothing else, so it drops the
   aspheric polynomial, the biconic y-branch, the whole freeform departure and
   the entire field-frame class.
2. **Its own guard makes it silently zero on a flat base.**  `if
   np.isfinite(_Rl) and _Rl != 0.0` is false for `radius = inf`, so a last
   surface whose power lives entirely in `aspheric_coeffs` -- a perfectly legal
   and quite ordinary aspheric design -- got **no** vertex correction at all,
   not a partial one.
3. **It assumes the exit medium is vacuum.**  `t = (z_image - _sag) / Nz2`
   with `Nz2 = 1/sec` folds `-sag*sec` of GEOMETRIC path into a leg whose
   optical path is then `exp(i k0 t)`, i.e. `n = 1`.  The shared projection
   resolves `n_exit` from the prescription.
4. **It assumes the exit ray goes forward.**  After a mirror the outgoing `N`
   is negative and the transfer to the vertex plane ADDS optical path where a
   transmissive exit subtracts it.  The in-line copy subtracted regardless, so
   on a mirror-terminated prescription it **doubled** the error instead of
   removing it.

### 1.1 Measured, at the ray level, on every surface class GBD can reach

`validation/probe_gbd_projection/probe_a_sag.py`.  One optic -- N-BAF10
biconvex, R1 = +11.0 mm, t = 0.9 mm, semi = 0.30 mm, 1.064 um -- with only the
LAST surface varied, plus a separate concave mirror; a collimated 2-D fan of
328 rays (41 radii x 8 azimuths, deliberately NOT meridional, because a
biconic and an XY-polynomial freeform depart from the rotationally-symmetric
conic only off the x axis and a `y = 0` fan reads their defect as exactly
zero).  `backend` is the one GBD's `jacobian='auto'` selects.

**Identical to every printed digit on both builds** (Windows py3.14 /
numpy 2.4.4, WSL py3.12 / numpy 2.4.6).

| last surface | backend | true sag | in-line sag error | as a fraction of the sag | **optical-path error** | height error |
|---|---|---|---|---|---|---|
| conic (control) | analytic | 3.460 waves | 0.0000 | 0.000 | **0.0000 waves** (1.5e-23 m) | 1.5e-23 m |
| conic + k = -0.60 (control) | analytic | 3.460 | 0.0000 | 0.000 | **0.0000** | 1.5e-23 m |
| even asphere A4 = 1.5e8 | analytic | 2.535 | 0.9243 | 0.365 | **0.9246** | 2.5e-08 m |
| even asphere A4 = 5.0e8, A6 = -4.0e15 | analytic | 2.374 | 1.0852 | 0.457 | **1.0858** | 3.9e-08 m |
| **flat base, all power in A2 / A4** | analytic | 4.385 | 4.3846 | **1.000** | **4.3862** | 1.3e-07 m |
| biconic, `radius_y` = -7.0 mm | fd | 5.364 | 1.9034 | 0.355 | **1.9052** | 8.9e-08 m |
| freeform (XY polynomial) | fd | 5.101 | 3.3141 | 0.650 | **3.3146** | 7.5e-08 m |
| field frame, decentre (60, -40) um | fd | 5.437 | 1.9764 | 0.364 | **1.9779** | 8.1e-08 m |
| **concave mirror R = -20 mm** | analytic | 8.125 | 0.0000 | 0.000 | **16.2790** | 9.7e-23 m |
| flat last surface (control) | analytic | 0.000 | 0.0000 | 0.000 | **0.0000 exactly** | 0.0 exactly |

Read the mirror row carefully: the in-line copy computed the SAG exactly
(0.0000 waves of sag error -- it is a conic mirror), and still got the optical
path wrong by **twice the sag** (16.279 = 2 x 8.125 x sec), because it applied
the correction with the wrong sign.  That is failure mode 4, and it is
invisible in a sag comparison.

The flat-base row is failure mode 2: the error is **100 % of the sag**, because
the copy contributed nothing at all.

### 1.2 WP-B12's own fixture, reproduced

The same probe also runs WP-B12 probe D's optic (N-BAF10 R = +/-2.10 mm,
semi = 0.20 mm) so section 5.1's reading is reproduced like for like:

| arm | in-line sag error | fraction of sag | optical-path error |
|---|---|---|---|
| conic last surface | 0.0000 waves | 0.000 | 0.0000 |
| A4 = 4.0e8, A6 = -8.0e17, meridional 401-ray fan | **16.0225 waves** | 0.719 | 24.1645 |
| the same, 2-D 328-ray fan | 16.0511 | 0.720 | 24.3975 |

WP-B12 reported **15.520 waves / 71 %** on that fixture, and that number is
re-taken here on the parent tree by re-running its own probe unchanged:

```
GBD in-line sag vs the shared sag, conic         : 1.550e-19 m (0.000 waves) on a sag of 6.947e-06 m
GBD in-line sag vs the shared sag, asphere_A4_A6 : 1.651e-05 m (15.520 waves) on a sag of 2.309e-05 m
```

The 15.52 / 16.02 gap is the BACKEND and the aliveness mask, not a
disagreement: WP-B12's probe used the finite-difference primitive, whose
companion-ray aliveness rule kills ten more rim rays on that steep asphere
than the analytic primitive does, so its maximum is taken over a slightly
smaller aperture.  Both are the same defect to three digits.

### 1.3 The control on the projection itself

The shared projection is checked against the library's OTHER vertex operator,
`TraceResult.at_exit_vertex()`, which is a different implementation (it works
on direction cosines and the traced `z`, not on the sag kernel and the
unreduced slopes).  On the aspheric fixture, over 200 rays:

| quantity | gap |
|---|---|
| height | **0.0 m** (both builds) |
| optical path | **0.0 m** (both builds) |

and the projection's own three readings of the sag -- from the height move,
from the optical-path move, and from `_surface_sag_xy` directly -- agree to
**1e-19 m** on every fixture in the table (`control_sag_*` in the JSON).

---

## 2. The repair, and why it is shaped this way

```python
_reference = 'surface' if world_output_plane is not None else 'exit_vertex'
dt = _jac(x, y, ux, uy, surfs, wavelength, per_surface=True,
          reference=_reference)
...
t = z_image / Nz2
```

and the `_Rl` / `_kl` block is gone.

**Why ask the primitive rather than call `_project_to_exit_vertex_plane`
directly.**  The brief allowed either.  `reference='exit_vertex'` is the
public contract WP-B12 created for exactly this, it is what the four `fga.py`
sites use, and it keeps the projection a private detail of
`raytrace.differential` instead of making `propagators.gbd` the one module
that reaches into another package's private function.  One convention, one
call shape, two consumers.

**Why the world branch keeps `'surface'`.**  This is the part that is easy to
get wrong.  The world branch does not add an axial leg from the vertex plane:
`_reframe_beamlets_to_world_plane` world-traces the base rays through the
folded prescription, reads `img.x, img.y, img.z` -- the last-surface
INTERSECTION -- and starts its own leg `t = -p_l[2] / d_l[2]` there.  Its OPL
piston (`amp * exp(1j*k0*dt.opd)`, applied before the branch) and its leg are
therefore already on one plane, the surface.  Moving the primitive under it
would subtract the sag once in `dt.opd` and never add it back: the same defect
this package removes, mirrored.  The two branches genuinely need different
reference planes, and the code says so at the call site with the reason.

Measured, not assumed: probe B runs the world branch in both trees and the
returned field is byte-identical (section 4.3).

---

## 3. What moves on a CONIC last surface, and why

On a conic last surface the in-line copy computed the sag exactly (section
1.1), so the change there is not a sag correction.  The field still moves, and
`validation/probe_gbd_projection/probe_c_decompose.py` isolates the cause at
the BEAMLET level -- the bundle `apply_prescription_persurface_to_beamlets`
returns, before any reconstruction, so the reading is the mechanism and not a
grid artefact.

Three arms, differing ONLY in what the projection does to the 4x4 Jacobian
(the state is projected identically in all three):

| arm | the Jacobian |
|---|---|
| `full` | what ships: `J_v = P J`, `P = [[I - u (x) grad s, -s I], [0, I]]` |
| `fs_sag_sec` | what the DELETED code effectively applied: `J_v = [[I, -sag*sec I], [0, I]] J` -- folding `-sag` into the leg `t = (z_image - sag)/Nz2` is the same Moebius step on `Q` |
| `state_only` | the state projected, the Jacobian left ON the surface -- the pre-v5.22 behaviour, with no vertex correction of `Q` at all |

Both builds, identical to every printed digit:

| fixture | comparison | relative `Q` | relative amplitude | base-ray position | relative phase |
|---|---|---|---|---|---|
| conic | `full` vs `fs_sag_sec` | **6.06e-05** | 7.40e-06 | **0.0 m exactly** | 3.44e-06 rad |
| conic + k | `full` vs `fs_sag_sec` | 6.06e-05 | 7.40e-06 | **0.0 exactly** | 3.44e-06 rad |
| asphere A4 / A6 | `full` vs `fs_sag_sec` | 7.04e-05 | 4.15e-06 | **0.0 exactly** | 6.87e-06 rad |
| flat-base asphere | `full` vs `fs_sag_sec` | 2.11e-05 | 6.08e-06 | **0.0 exactly** | 1.79e-06 rad |
| **flat last surface** | all three pairs | **0.0** | **0.0** | **0.0** | 8.3e-17 rad |
| conic | `full` vs `state_only` | 4.33e-04 | 9.86e-05 | 0.0 | 6.89e-06 rad |

So on a conic last surface **every moved bit is the Jacobian projection**:

* the base-ray POSITIONS are bit-identical -- `dt.x - sag*ux + z_image*ux`
  and `dt.x + (z_image - sag)*ux` are the same map, and the probe reads the
  difference as exactly 0.0 m;
* the base-ray OPTICAL PATH is likewise the same map (`opd - n*nz*sag*sec`
  then `+ k0*z_image*sec`, against `opd` then `+ k0*(z_image - sag)*sec`),
  identical whenever `n_exit * sign(N) = 1`, which is every transmissive
  prescription ending in air;
* `Q` and the amplitude move by 6e-05 and 7e-06 relative, because the old
  composition `FS(-sag*sec) J` differs from `P J` by the `-u (x) grad s` block
  (the transfer distance depends on where the ray lands, so the derivative of
  the vertex-plane map carries that term) and by `sag*(sec - 1)` in the `B`
  block.  WP-B12 pinned `P J` against an independent finite difference of the
  vertex-plane state at 2.1e-10 where the un-projected Jacobian sits at
  2.1e-05, so `P J` is the derivative of the map actually applied and
  `FS(-sag*sec) J` is not.

The `state_only` row is in the table for a specific reason: it is what a
reader reconstructs by "forcing the primitive back to `reference='surface'`",
and on a CURVED-base last surface that is the **pre-v5.22** behaviour, NOT the
v5.22 .. 5.47.0 one -- the two differ by the whole conic sag (4.3e-04 in `Q`
here, and a 1.3 relative L2 in the reconstructed field).  Every place in this
package and in `tests/unit/test_audit2609_b12b_gbd_projection.py` that uses
that trick says which case it is in and asserts the premise.  It coincides
with v5.22 .. 5.47.0 exactly on a surface whose in-line sag was identically
zero -- a flat last surface, and the flat-BASE aspheric fixture -- which is
what makes the next section's before/after a single-process measurement.

---

## 4. The oracle ladder -- GBD before and after

`validation/probe_gbd_projection/probe_b_ladder.py` scores the field
`apply_real_lens_gbd` returns against the independent Rayleigh-Sommerfeld-I
oracle imported from `validation/probe_wp_b12/b12_common.py`, at the exit
vertex and at each fixture's own traced best focus.

The before/after is **archive-to-archive**, not a reconstruction: the `pre`
arm is `git archive fix/wp-b12-fga-reference-plane lumenairy` extracted into
`C:\tmp\lum_gbd_pre` and the same probe script run THERE, in its own process,
with `cwd` and `PYTHONPATH` set to that tree and `lumenairy.__file__` asserted
to live under it.  The arm label is DETECTED from the library
(`'_Rl = float(' in inspect.getsource(apply_prescription_persurface_to_beamlets)`)
rather than passed on the command line, and both the label and the resolved
`lumenairy.__file__` are written into each JSON, so the two runs cannot be
confused with one another.

The optic is the section-1.1 one; the grid is 128 x 5.0 um (0.640 mm span
against a 0.600 mm clear aperture) and the Airy radius runs 17 to 44 um across
the fixtures, i.e. three to nine pixels, so the focal structure is resolved
and the fidelity means something.  The beamlet frame is the library's own
default (`_auto_sample_step`), not the cheap frame the test file names --
these rows are what a caller actually gets.

### 4.1 Windows build (py3.14, numpy 2.4.4)

`sha` is the SHA-256 of the returned field's exact bytes (first 24 hex).

| last surface | plane | `pre_b12b` | `post_b12b` | power ratio, pre -> post | field bytes |
|---|---|---|---|---|---|
| conic (control) | exit vertex | 0.998896 | 0.998896 | 0.9991 -> 0.9991 | changed |
| conic (control) | focus 8.2630 mm | 0.999711 | 0.999711 | 0.9990 -> 0.9990 | changed |
| conic + k = -0.60 (control) | exit vertex | 0.998896 | 0.998896 | 0.9991 -> 0.9991 | changed |
| conic + k = -0.60 (control) | focus 8.2644 mm | 0.999711 | 0.999711 | 0.9990 -> 0.9990 | changed |
| even asphere A4 = 1.5e8 | exit vertex | 0.760509 | **0.998895** | 0.9970 -> 0.9991 | changed |
| even asphere A4 = 1.5e8 | focus 9.8843 mm | 0.761746 | **0.999781** | 0.9972 -> 0.9994 | changed |
| even asphere A4 / A6 | exit vertex | 0.503115 | **0.998896** | 0.9930 -> 0.9992 | changed |
| even asphere A4 / A6 | focus 9.9262 mm | 0.503899 | **0.999765** | 0.9929 -> 0.9991 | changed |
| **flat base, A2 / A4** | exit vertex | **0.017658** | **0.998894** | 0.8894 -> 0.9988 | changed |
| **flat base, A2 / A4** | focus 6.6981 mm | **0.017708** | **0.999671** | 0.8899 -> 0.9994 | changed |
| **flat last surface (control)** | exit vertex | 0.997766 | 0.997766 | 0.9917 -> 0.9917 | **IDENTICAL** (`ea72c10375e1af762646cc50` before the regrid, `85a23bd7dc8db39b79db160b` after) |
| **flat last surface (control)** | focus 1.4303 mm | 0.999118 | 0.999118 | 0.9930 -> 0.9930 | **IDENTICAL** |

The flat control's own SHA-256 is the same 24 hex digits in the parent tree and
in this one, at BOTH planes, and it stayed identical when the fixture's grid
was changed mid-package (192 x 2.6 um -> 128 x 4.3 um, to cut a 1086 s call to
140 s): `ea72c10375e1af762646cc50` / `2d43deb6b6a0a3f611e5a367` on the first
grid and `85a23bd7dc8db39b79db160b` on the second, each equal across the two
trees.  That row's FIDELITY is not a resolved-focus reading -- its Airy radius
is 3.43 um against a 4.3 um pitch, `dx/airy = 1.25` -- and it is not used as
one: its job is byte-identity, which does not depend on the grid.  Every other
row has `dx/airy` between 0.196 and 0.272.

Three things this table says.

1. **Every defect row moves to the same place the controls already sit.**  The
   two conic controls read 0.9989 at the vertex and 0.9997 at the focus both
   before and after -- that is GBD's own accuracy on this fixture at this
   frame density, and it is the ceiling.  The aspheric rows arrive there:
   0.7605 -> 0.9989, 0.5031 -> 0.9989, 0.0177 -> 0.9989.  The repair does not
   merely improve them, it removes the only thing separating them from the
   controls.
2. **The ladder is monotone in the defect.**  Ordered by the section-1.1
   optical-path error -- 0.92, 1.09, 4.39 waves -- the pre-repair fidelities
   are 0.7605, 0.5031, 0.0177.  A 4.39-wave phase ramp across the pupil leaves
   1.8 % of the field, which is what a ramp no global piston can absorb looks
   like.
3. **The two CONIC CONTROL rows move in the SEVENTH decimal of fidelity, and
   their bytes move.**  At full precision the conic control reads
   `0.99889590 -> 0.99889613` at the vertex and `0.99971083 -> 0.99971113` at
   the focus; the conic-plus-k row reads `0.99889590 -> 0.99889613` and
   `0.99971093 -> 0.99971124`.  That is **+2.3e-07 and +3.1e-07 of fidelity**
   -- the Jacobian projection of section 3, arriving in the field exactly
   where a 6e-05 relative change in `Q` and a 9e-06 rad relative phase should
   put it, and in the projection's favour on all four readings.  The flat
   control, by contrast, reads `0.99776557` and `0.99911761` in BOTH trees,
   digit for digit, because its field is the same bytes.

### 4.2 Full precision, and what the controls actually read

Taken from `probe_b_ladder_post_b12b_win32_314.json`:

| fixture | plane | fidelity | relative L2 | Airy radius | `dx/airy` |
|---|---|---|---|---|---|
| conic | exit vertex | 0.99889613 | 0.03322 | 18.53 um | 0.270 |
| conic | focus | 0.99971113 | 0.01700 | | |
| conic + k | exit vertex | 0.99889613 | 0.03322 | 18.53 um | 0.270 |
| conic + k | focus | 0.99971124 | 0.01699 | | |
| asphere A4 | exit vertex | 0.99889482 | 0.03324 | 25.46 um | 0.196 |
| asphere A4 | focus | 0.99978076 | 0.01481 | | |
| asphere A4 / A6 | exit vertex | 0.99889591 | 0.03323 | 18.41 um | 0.272 |
| asphere A4 / A6 | focus | 0.99976466 | 0.01534 | | |
| flat-base asphere | exit vertex | 0.99889415 | 0.03325 | 24.52 um | 0.204 |
| flat-base asphere | focus | 0.99967130 | 0.01813 | | |

The five exit-vertex rows lie within **1.98e-06 of fidelity** of one another
after the repair (0.99889613, 0.99889613, 0.99889482, 0.99889591,
0.99889415: `max - min` is 1.98e-06 and the largest deviation from their MEAN
is 1.3e-06), across a conic, a conic-plus-k, two different aspheres and a
flat-base asphere.  *(Restated 2026-09-19, VERIFY-WP-B12b D-7: this sentence
read "agree to 1.3e-06", which is the deviation from the mean rather than the
spread -- "agree to X" normally means the spread, so both numbers are now
given.)*  That is the reading that says the repair is a correction
rather than a tuning: five different last surfaces, one number, and it is
GBD's own frame-density floor on this grid (0.9989), not a property of any one
of them.  Before the repair the same five read 0.9989, 0.9989, 0.7605, 0.5031
and 0.0177.

### 4.3 WSL build (py3.12, numpy 2.4.6)

The `post_b12b` arm was re-run in full on the second build
(`probe_b_ladder_post_b12b_linux_312.json`: six symmetric fixtures at two
planes, four non-symmetric ones, three entry points).  **All twelve
symmetric-fixture fidelities agree with the Windows build to all eight
printed digits** -- 0.99889613 / 0.99971113, 0.99889613 / 0.99971124,
0.99889482 / 0.99978076, 0.99889591 / 0.99976466, 0.99889415 / 0.99967130,
0.99776557 / 0.99911761 -- and the non-symmetric fixtures' peaks agree to
every printed digit too (the biconic's 6.15047365, the freeform's 2.97003652,
the mirror's 3.75320659e-02).

The SHA-256 digests do NOT agree across builds, and are not expected to: the
field is a coherent sum over tens of thousands of beamlets and the two builds
have different LAPACK, so the last bits differ.  **Every byte-identity claim
in this report is WITHIN one build, between two trees**; the cross-build
statement is always the fidelity.

The `pre_b12b` arm was not re-run on WSL.  Each arm of this ladder costs about
an hour on this box, and the cross-build question it would answer -- whether
the DEFECT reads the same on a second LAPACK -- is already answered at the ray
level by probe A, which is identical to every printed digit on both builds on
all fourteen rows.


### 4.4 The non-rotationally-symmetric fixtures

The imported oracle is a rotationally-symmetric trace, so these four carry no
diffraction reference here.  What is reported is MOVEMENT -- the field bytes
and the total power -- never accuracy; their defect is measured at the ray
level in section 1.1.

One optic geometry with the last surface varied (biconic `radius_y`, an
XY-polynomial freeform, a field-frame decentre), plus the concave mirror.
`P` is the summed intensity on the grid; `peak` is `max|E|`.  Windows build,
archive-to-archive.

| fixture | plane | `pre` SHA-256 | `post` SHA-256 | `pre` peak | `post` peak |
|---|---|---|---|---|---|
| biconic | exit vertex | `87d174991b3b8be0e9dfabc3` | `fe83b90e58243e14cc176ab2` | 1.0471 | 1.0327 |
| biconic | BFL 8.27 mm | `91c83257ca99a99a612620cf` | `0aa409b615f9c7806e5db7a0` | 3.9704 | **6.1505** |
| freeform | exit vertex | `097b721f457e9a62a1761a0e` | `7d97edf0b83dbe84c917fea5` | 1.0244 | 1.0327 |
| freeform | BFL | `052c68eb41c6c568b49ba4c0` | `3e8231c79450738947abd895` | 1.2033 | **2.9700** |
| field frame | exit vertex | `56cd479ee9e538d92e394b34` | `19758dbf1ab300bb71ef5602` | 1.0231 | 1.0328 |
| field frame | BFL | `b14b1bd59e0dcdebf0671d30` | `5a5cda8383186e134ce98cf1` | 10.378 | 10.466 |
| mirror | exit vertex | `1b9c7600a2471e804758366f` | `95e7a571666a2ebcb5aa152b` | 13.169 | 13.164 |
| mirror | BFL | `91e859c45770d12833f76da0` | `ea929fb06f522c2b27908db5` | 0.037557 | 0.037532 |

Every row moved, which is the blast radius of section 5 confirmed at the
field.  Two of them moved a LOT and in the direction a repair predicts: the
biconic's focal peak rises by **55 %** and the freeform's by **147 %**,
because the beamlets stop carrying 1.9 and 3.3 waves of spurious pupil phase
and the focus gets sharper.  The field-frame row moves by under 1 % at the
focus, consistent with its smaller 2.0-wave defect, and the exit-vertex rows
move only a little because a pupil-plane intensity is insensitive to a pupil
phase error (the phase moved; the modulus did not).

The MIRROR rows move by only 3e-04 relative, and that reading needs its
caveat: as section 8 item 2 records, the LOCAL branch's direction convention
is wrong for a mirror-terminated prescription independently of the sag, so
this fixture's field is not physically meaningful in either tree.  Its
decisive measurement is the RAY-level one -- the 16.28-wave sign defect of
section 1.1 -- not this row.

### 4.5 The other entry points

All on the flat-base aspheric fixture at its traced best focus (6.6981 mm),
Windows build, archive-to-archive.

| entry point | `pre` SHA-256 | `post` SHA-256 | `pre` fidelity | `post` fidelity |
|---|---|---|---|---|
| `apply_real_lens_gbd` (the ladder's own row) | `c9b4dc7d2d1e063da6ba0821` | `043dcd0bf9af0b70616149eb` | 0.017708 | **0.999671** |
| `propagate_gbd_through_prescription(per_surface=True)` | `c9b4dc7d2d1e063da6ba0821` | `043dcd0bf9af0b70616149eb` | 0.017708 | **0.999671** |
| `apply_real_lens_universal(method='gbd')` | `c9b4dc7d2d1e063da6ba0821` | `043dcd0bf9af0b70616149eb` | 0.017708 | **0.999671** |
| **`world_output_plane` branch** | `851ccadc6483aa3451659c32` | `851ccadc6483aa3451659c32` | -- | **IDENTICAL** |

The same four rows on the WSL build's `post` arm, for the cross-build
comparison: `apply_real_lens_gbd` and `apply_real_lens_universal` both
`ff20d9605b5c30782a442ee1`, `propagate_gbd_through_prescription`
`fe36b9a7df9aeeca38a51607`, the world branch `b3d1c16f58f0216eefb40b78`, and
all three local entries at fidelity 0.9996713034955186.  Digests differ
between BUILDS on every row, as a coherent sum over tens of thousands of
beamlets on two different LAPACKs must; the byte-identity claims of this
report are always WITHIN one build, between two trees.

Two things are settled here.

1. **All three local-frame entry points move together, and the repair reaches
   all three at once** -- 0.017708 to 0.999671 on every one of them.  How far
   that agreement goes is worth stating precisely, because it is NOT the same
   on the two builds:

   * `apply_real_lens_universal(method='gbd')` is byte-identical to
     `apply_real_lens_gbd` on BOTH builds and in BOTH trees (`043dcd0b...`
     on Windows, `ff20d960...` on WSL).  It is a dispatcher, and it
     dispatches.
   * `propagate_gbd_through_prescription(per_surface=True)` read
     byte-identical to them on the Windows build (in both trees) and NOT on
     WSL (`fe36b9a7...` against `ff20d960...`) on THIS fixture, while its
     fidelity against the oracle agrees to FIFTEEN digits
     (0.9996713034955186 against 0.9996713034955185).

     *(Restated 2026-09-19, VERIFY-WP-B12b claim 4.)*  This sentence used to
     conclude "their last-bits agreement is a property of the build".  It is
     not: the re-verification took the same comparison with the beamlet FRAME
     named explicitly and found the two entries byte-identical on **both**
     builds in **both** trees (its section 6.1).  What separates them is the
     frame and the chunk boundaries it produces, not the LAPACK.  The two
     routes reach the same live beamlet set by different roads --
     `apply_real_lens_gbd` prunes dark beamlets up front (784 raw -> 437
     pruned on that fixture, two DIFFERENT bundle digests), the
     propagators-level entry lets the trace vignette them (437 survivors
     either way, and the reconstructed field identical to 0.0) -- and the
     coherent sum is then grouped by `_reconstruct_windowed`'s chunking,
     which moves with the memory budget and the frame (see the D-6 note in
     section 7.4).  The DECISION the test file takes is unchanged and still
     right: BYTE identity is asserted only for the dispatcher pair, and a
     0.999 FIDELITY bar for this one, which is why it passes on both builds.
     It just holds for a reason this report did not give.
2. **The `world_output_plane` branch is byte-identical between the parent tree
   and this one**, digest for digest.  That is the design of section 2
   measured rather than argued: the branch that measures its own leg from the
   last-surface intersection kept `reference='surface'` and did not move.


---

## 5. Blast radius

| entry point | effect |
|---|---|
| `lumenairy.apply_real_lens_gbd` (`per_surface=True`, the default) | **moves** on an aspheric / biconic / freeform / field-frame / mirror last surface; fifth-decimal on a conic one; bit-identical on a flat one |
| `propagators.gbd.apply_prescription_persurface_to_beamlets`, local branch | the same |
| `propagators.gbd.propagate_gbd_through_prescription(per_surface=True)` | the same (it calls the function above) |
| `propagators.fga.apply_real_lens_universal(method='gbd')` | the same -- byte-identical to `apply_real_lens_gbd`, checked |
| `apply_real_lens_gbd(per_surface=False)` | untouched -- the paraxial whole-system ABCD path has no differential transfer |
| `apply_prescription_persurface_to_beamlets(world_output_plane=...)` | **untouched, bit for bit** -- it keeps `reference='surface'` |
| `propagate_gbd_vector_through_prescription` | untouched: its Jones machinery uses `_fresnel_jones_matrix_per_beamlet`, which traces separately and never calls the differential primitives |
| every `propagators.fga` site | untouched by this package (WP-B12 already moved them) |
| the JAX paths | **there is no JAX per-surface-GBD path.**  `propagators/gbd.py`'s xp-dispatched code is the free-space, thin-lens and reconstruction machinery; the per-surface prescription path is NumPy-only by construction (its docstring says so, and `lumenairy/propagators/gbd.py` contains no `import jax` and no `import jax.numpy` -- `grep -n 'jax\|jnp'` on that file returns **12** matches, all of them docstrings, `is_jax_array` calls and a comment about a `jnp.at[].add` scatter, and none of them an import; *restated 2026-09-19, VERIFY-WP-B12b D-8, which also confirmed the substantive claim independently in its section 6.4*).  `elements/lenses_gbd.py` contains no JAX at all.  `ray_transfer_jacobian_analytic`'s JAX branch shares `_project_to_exit_vertex_plane` with NumPy, so a future JAX GBD would inherit the repair; nothing today exercises it through GBD. |

---

## 6. Files changed

| file | what |
|---|---|
| `lumenairy/propagators/gbd.py` | the local branch requests `reference='exit_vertex'` and the world branch keeps `'surface'`, both with the reason at the call site; the `_Rl` / `_kl` in-line sag block and the `- _sag` in `t` are deleted; the function docstring gains a "The reference plane" section and a `versionchanged` note |
| `lumenairy/raytrace/differential.py` | the module docstring and `DifferentialTransfer`'s name `propagators.gbd`'s local branch as an `'exit_vertex'` consumer (docstrings only -- the module's AST and token fingerprints are unchanged, which is why `docs/history/lumenairy.raytrace.differential.md` is not re-recorded) |
| `docs/history/lumenairy.propagators.gbd.md` | fingerprints re-recorded with the reason |
| `tests/unit/test_audit2609_b12b_gbd_projection.py` | NEW |
| `CHANGELOG.md` | a `### Fixed -- GBD: ...` entry inside the existing `## [Unreleased]` block, with its Migration note; the WP-B12 entry's closing sentence ("`apply_real_lens_gbd` ... are untouched") is restated in the past tense and pointed at this entry, since both live in the same unreleased block |
| `validation/probe_gbd_projection/` | `gbdproj_common.py` (fixtures; the oracle is IMPORTED from `probe_wp_b12/b12_common.py`), `probe_a_sag.py`, `probe_b_ladder.py`, `probe_c_decompose.py` and their JSON on both builds |
| `docs/.../fixes/WP-B12b_GBD_REPORT.md` | this report |

Note on `probe_wp_b12/probe_d_consumers.py`: that probe's first assertion --
"GBD's curved-last-surface field is bit-identical to the forced-surface arm" --
was true of the tree WP-B12 shipped and is false of this one, by design.  The
probe is WP-B12's artefact and is left untouched; its JSON in this tree is the
one WP-B12 recorded.  WP-B12's report keeps its open item 1 as written, with a
dated line saying this package closed it.

---

## 7. Tests run

Every invocation carried
`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1` and
`--capture=sys -p no:randomly`, from `C:\tmp\lum_gbd`, on 2026-09-15.

**The box.**  Between 40 and 67 other heavy python processes (sibling Wave-5
agents and the maintainer's own runs) were resident throughout, with the CPU
pinned at 100 % and 76 GB of the 128 GB free -- so it is contention, not
memory.  Measured: one `apply_real_lens_gbd` call that costs about 2 s on an
idle box took 20-30 s here, and the two builds' numbers below carry that
factor.  Wall clocks are REPORTED, never asserted
(`docs/TESTING_STANDARDS.md`), and no test in this package contains a timing
assertion.

### 7.1 The two builds

| | Windows | WSL |
|---|---|---|
| python | 3.14.6 | 3.12.3 |
| numpy / scipy | 2.4.4 / 1.17.1 | 2.4.6 / 1.17.1 |

### 7.2 pytest

| selection | build | result | duration |
|---|---|---|---|
| `test_audit2609_b12b_gbd_projection.py` (NEW, 14 ids) | Windows, box LOADED (40-67 sibling python jobs) | **14 passed** | 40.5 s; slowest id 14.4 s |
| the same | WSL, box LOADED | **14 passed** | 75.1 s; slowest id 29.7 s |
| the same | Windows, box QUIET (7 python processes) | **14 passed** | **14.0 s; slowest id 5.6 s** |
| `test_analytic_ray_transfer.py`, `test_audit2609_a1_exit_vertex.py`, `test_audit2609_a4_maslov_gbd.py`, `test_audit2609_b12_fga_reference_plane.py`, `test_audit2609_b12b_gbd_projection.py`, `test_gbd_feature_complete.py`, `test_hammer_h7_gbd_diverging.py`, `test_lens_gbd.py`, `test_niche_p1_gbd_chain.py`, `test_niche_p4_gbd_reexpand.py`, `test_niche_r3_gbd_mem_lstsq.py`, `test_niche_r5_gbd_vector_catastrophe.py`, `test_v5_21_gbd_asm_interop.py`, `test_v5_21_gbd_maslov_perf.py`, `test_v5_21_gbd_windowed_adaptive.py` (216 ids) | Windows | **216 passed, 0 failed** | 3639.9 s |
| the same fifteen files + `test_niche_p9_decenter_tilt.py` (229 ids) | WSL | **229 passed, 0 failed** | 3773.9 s |
| the census / walker / dispatcher-pin / public-API / doc-consistency / history / `test_audit_except_budget.py` sweep (34 files) | Windows | **1554 passed, 12 skipped, 0 failed** | 997.5 s |
| every OTHER test file that names `apply_real_lens_gbd`, `apply_prescription_persurface_to_beamlets`, `propagate_gbd_through_prescription` or `method='gbd'` -- `test_niche_p9_decenter_tilt.py` (a FIELD-FRAME decentred prescription through the beamlet function, i.e. a class whose field moves), `test_niche_p8_capstone.py`, the three `test_audit2609_a16_lens_config*` files, `test_audit2609_b7_asymptotic.py` (which inspects this function's source), `test_niche_d5_dx_flatness_gate.py`, `test_niche_k1_kmah_caustic.py`, `test_niche_p11_ray_density_amplitude.py`, `test_niche_audit_w3_elements.py`, `test_niche_audit_w4_ignored_kwarg_warnings.py`, `test_niche_audit_w4_input_kind.py`, `test_audit_glass.py`, `test_fga.py`, `test_v5_2_physics_fixes.py` (614 ids) | Windows | **614 passed, 0 failed** | 2688.1 s |

Every test in the new file is inside the 60 s budget on BOTH builds, with the
slowest at 14.4 s (Windows) and 29.7 s (WSL) on a box carrying the contention
described above.  Getting there was itself a measurement, and it is recorded
in the file: the auto beamlet frame (`sample_step=1`, `waist_factor=1`) puts
one beamlet per pixel with a 0.13 mm Rayleigh range, so at the image plane
every beamlet is wider than the whole grid and
`_reconstruct_windowed` degenerates to the dense `O(n_beamlets x N^2)` sum --
60.7 s of a 78.6 s call, by `cProfile`.  A 4-pixel frame with a matching waist
costs 2.5 s and reproduces the dense-frame field at a fidelity of 0.999587,
three decades below the test's 0.99 decision bar; the tests name that frame
explicitly, which also makes the three public entry points comparable byte for
byte (they have different sampling DEFAULTS).

### 7.3 The other gates

| gate | result |
|---|---|
| `wsl ruff check lumenairy/ tests/ validation/probe_gbd_projection/ validation/probe_wp_b12/` | **All checks passed** |
| `python scripts/record_history_fingerprints.py --check` | **OK: every history document matches its module** |
| `python scripts/check_source_line_citations.py` (V18) | **ok=107, drift=0, total=107** |
| `python scripts/check_doc_identifiers.py` | **OK**; 0 unresolved, 0 DO-NOT-resolve |
| `.test_durations` | 16 200 -> **16 214** entries, valid JSON, 14 new ids, largest 12.6 s |

`docs/history/lumenairy.raytrace.differential.md` is deliberately NOT
re-recorded: the only change to that module is docstring text, and the
fingerprint is the SHA-256 of the AST with docstrings removed and of the
`tokenize` stream with comments and docstrings dropped, so neither moved.
`scripts/record_history_fingerprints.py lumenairy/raytrace/differential.py`
was run anyway and reported no change; `--check` is green.

### 7.4 Probes

All under `validation/probe_gbd_projection/`, each run with `PYTHONPATH`
pinned to the tree under test and `lumenairy.__file__` printed and asserted to
live under it.

| probe | what | outputs |
|---|---|---|
| `probe_a_sag.py` | the mechanism and its four failure modes on ten surface classes, plus WP-B12 probe D's own optic and the `at_exit_vertex` control | `probe_a_sag_win32_314.json`, `probe_a_sag_linux_312.json` |
| `probe_b_ladder.py` | the oracle ladder at two planes per fixture, the non-symmetric fixtures' movement, and the three other entry points | `probe_b_ladder_pre_b12b_win32_314.json`, `probe_b_ladder_post_b12b_win32_314.json`, `probe_b_ladder_post_b12b_linux_312.json` |
| `probe_c_decompose.py` | what moves on a conic last surface, at the beamlet level, in three arms | `probe_c_decompose_win32_314.json`, `probe_c_decompose_linux_312.json` |

**Cross-build agreement.**  Probe A is identical to every printed digit on the
two builds, on all fourteen rows including the two exact-zero controls.  Probe
C is identical to every printed digit on every row but the flat control's
`dPhase`, where the two builds read **6.797e-17** and **6.760e-17** radians --
a quantity that is zero up to the unwrap's own rounding.  *(Corrected
2026-09-19, VERIFY-WP-B12b D-3: this sentence published 8.318e-17 / 8.298e-17,
which is not what the committed JSONs contain.  The shipped artefacts read
`6.796869888613907e-17` (`probe_c_decompose_win32_314.json`) and
`6.760231407720553e-17` (`..._linux_312.json`), and re-running
`probe_c_decompose.py` unchanged on the round-2 tree reproduces both files
number for number on both builds -- the only line that moves is the embedded
`"version"` string, which tracks the tree, not the measurement.  Re-run
recorded in `validation/probe_wp_b12b_round2/probe_r4_probec_*.json`.)*  Probe B's cross-build
comparison is in section 4.3.

The `pre_b12b` arm is not a claim about what the old code did, it is the old
code: `git archive fix/wp-b12-fga-reference-plane lumenairy` extracted into
`C:\tmp\lum_gbd_pre`, the probe run there in its own process with `cwd` and
`PYTHONPATH` set to that tree, and the arm DETECTED from the library
(`'_Rl = float(' in inspect.getsource(...)`) rather than passed in.  The
detection and the resolved `lumenairy.__file__` are both written into each
JSON.


---

## 8. Open items

1. **The image-side leg still assumes a vacuum exit medium.**  The projection
   now resolves `n_exit` from the prescription, but GBD's own leg is
   `exp(i k0 * z_image * sec)` with no index, exactly as `fga.py`'s is.  On an
   immersed prescription the two disagree by `(n_exit - 1) * z_image * sec`.
   This is WP-B12's open item 4 in the GBD module, pre-existing and untouched
   here; every GBD fixture in this library ends in air, so it is not measured.
   A caller in that situation should be refused rather than silently served.
2. **The local branch is not correct for a mirror-terminated prescription, for
   reasons that have nothing to do with the sag.**  It takes
   `Nz2 = 1/sqrt(1+u^2)`, which is positive whatever the true `N`, so after a
   mirror the reconstructed `new_dir` points the wrong way and the leg
   `t = z_image/Nz2` walks forward along an axis the light is no longer
   travelling.  That is what `world_output_plane` exists for.  This package
   fixes the SIGN of the sag term (measured: 16.28 waves on the probe's
   concave mirror) and leaves the branch's own direction convention alone; a
   mirror-terminated prescription through the LOCAL branch should probably be
   refused with a message naming `world_output_plane`.
3. **The Jacobian projection's fifth-decimal effect was not scored against a
   diffraction oracle on its own.**  Section 3 measures it at the beamlet
   level (6e-05 in `Q`) and section 4's conic rows carry it into the field,
   but no fixture here separates "the field with `P J`" from "the field with
   `FS(-sag*sec) J`" against the oracle at a precision where 6e-05 of `Q`
   would show.  WP-B12 did that measurement for FGA (its section 3, ten of
   twelve rows improving by 1e-05 to 1.6e-05 of fidelity) and the argument
   carries, but it is an argument, not a GBD measurement.
4. **No freeform, biconic or field-frame field was scored against a
   diffraction oracle.**  The oracle this package imports is a
   rotationally-symmetric trace and cannot represent those surfaces; their
   defect is measured at the RAY level (section 1.1, 1.2 to 3.3 waves) and
   their fields are reported as movement only.  The projection is correct
   there by construction -- it uses the same sag kernels the tracer does, and
   `TraceResult.at_exit_vertex` agrees with it to 0.0 m -- but "correct by
   construction" is not "measured against diffraction".
5. **Not measured: the CI cross-build spread.**  Both builds here are the same
   box; the runner mix of EPYC 9V74 and 7763 with older wheels was not
   sampled.  Every bar in the new test file is derived at runtime from a
   quantity the running build measures.


---

# Round 2 (VERIFY-WP-B12b) -- 2026-09-19

The independent re-verification of this package
(`fixes/VERIFY_WP-B12b.md`, verdict **SHIP**) filed eight findings.  D-1 was
closed by the verification's own nine ids.  This addendum closes the other
seven: two LIBRARY defects (D-5, D-4), one test defect (D-2), one published
number (D-3), one evidence-methodology note (D-6) and three wording items
(D-7, D-8 and the nit inside claim 4).  Branch `fix/wp-b12b-gbd-round2`,
worktree `C:\tmp\lum_gbd2`, off the integration tip `76019ede`.

**The environment, stated once.**  Every invocation below carried
`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1` **and**
`LUMENAIRY_MEM_BUDGET_MB=2048` on the command line (D-6: a GBD field's digest
depends on that variable through the public entry), and pytest ran with
`--capture=sys -p no:randomly`.  Two builds throughout: **Windows** py3.14.6 /
numpy 2.4.4 / scipy 1.17.1 and **WSL** py3.12.3 / numpy 2.4.6 / scipy 1.17.1.
The PRE tree is my own `git archive 76019ede` extracted into
`C:\tmp\lum_gbd2_pre`, run in its own process with `cwd` and `PYTHONPATH` set
there, `lumenairy.__file__` printed and asserted to live under it, and the arm
DETECTED from the library's own source (`'_require_forward_going_local_exit'
in inspect.getsource(gbd)`) rather than passed on the command line.

**The box.**  Between 8 and 12 other heavy python processes (sibling Wave-5
agents, and my own parallel Windows / WSL probe arms) were resident for most
of this package.  Wall clocks are REPORTED, never asserted; no test in this
package contains a timing assertion.

**Probes.**  All under `validation/probe_wp_b12b_round2/`.  Nothing is
re-implemented: the fixtures and the 3-D oracle are the VERIFIER's own
(`validation/probe_verify_b12b/vb12b_common.py`), imported by path from the
probe tree so a PRE arm running against an archived `lumenairy` still scores
against one oracle and one fixture table.

| probe | what | outputs |
|---|---|---|
| `probe_r1_guards.py` | the two guard decisions at five routes onto the local branch, on the verifier's immersed fixture and two of my own; the boundary bisected AT THE CALL SITE; the mirror through both branches | `probe_r1_guards_{pre,post}_*.json` |
| `probe_r2_identity.py` | 15 air fixtures x 2 planes + the three local entry points, archive to archive | `probe_r2_identity_{pre,post}_*.json` |
| `probe_r3_budget.py` | D-6 re-taken through the public entry, one CHILD PROCESS per budget | `probe_r3_budget_post_*.json` |
| `probe_r4_probec.py` | D-3: the builder's own `probe_c_decompose.py` re-run and its JSONs read back | `probe_r4_probec_post_*.json` |
| `probe_r5_mirror.py` | D-4's two headline numbers re-measured, both signs of `z_image` | `probe_r5_mirror_{pre,post}_*.json` |
| `pre_b12_beamlet_fn.py.txt` | the PRE (1218b24f) source of the beamlet function, committed as DATA for the D-2 fail-before | -- |

---

## R2.1 D-5 (P2, LIBRARY) -- the immersed exit is guarded, through the FGA helper

**What changed.**  `lumenairy/propagators/gbd.py`,
`apply_prescription_persurface_to_beamlets`, the LOCAL branch, immediately
after `z_image` is resolved -- where the length of the index-free leg first
exists, rather than after the ray trace has been paid for:

```python
        _require_forward_going_local_exit(
            surfs, 'apply_prescription_persurface_to_beamlets')
        from .fga import _require_non_immersed_exit
        _require_non_immersed_exit(
            surfs, wavelength, z_image,
            'apply_prescription_persurface_to_beamlets')
```

**One tolerance, not two.**  The import is the shared helper, not a copy.  A
module-level `from .fga import _require_non_immersed_exit` was measured to
import cleanly from five entry modules (`lumenairy`, `lumenairy.elements`,
`lumenairy.propagators`, `lumenairy.raytrace`, `lumenairy.propagators.gbd`
and `...fga` first), so there is no cycle today -- but it would make
`propagators.gbd` eagerly depend on `propagators.fga`, which
`propagators/__init__.py` documents as cycle-free, and `fga` already reaches
into `gbd` through a function-level import of the same shape.  The
function-level form was chosen for that reason and the measurement is
recorded here rather than the claim.

**The premise, measured (both builds, identical to every printed digit).**

| fixture | `n_exit` | `z_image` | waves the index-free leg omits |
|---|---|---|---|
| the verifier's `immersed` (n = 1.72) | 1.72 | 2 mm | **1846.2589642118448** |
| mine, water-like | 1.333 | 0.6 mm | 315.7798324970187 |
| mine, oil-like | 1.5180 | 0.6 mm | 491.09943307174024 |
| mine, AIR control | **exactly 1.0** | 0.6 mm | **0.0** |
| the verifier's AIR control | **exactly 1.0** | 2 mm | **0.0** |

The 1846.26 figure reproduces VERIFY-WP-B12b section 9.1 to every printed
digit, from a different probe.

**The decision, at five routes (`probe_r1_guards.py`).**  Rows are
`refused` / `served`; PRE is the archive tree, POST this branch.  The two
builds are **cell-for-cell identical**.

| route | air (PRE) | air (POST) | immersed (PRE) | immersed (POST) |
|---|---|---|---|---|
| `apply_prescription_persurface_to_beamlets` | served | served | **served** | **refused** |
| `apply_real_lens_gbd` | served | served | **served** | **refused** |
| `apply_real_lens_universal(method='gbd')` | served | served | **served** | **refused** |
| `propagate_gbd_through_prescription(per_surface=True)` | served | served | **served** | **refused** |
| the `world_output_plane` branch | served | served | **served** | **served** |

The last row is the one to read carefully: the world branch is deliberately
NOT guarded.  Its own leg is likewise index-free, so it has the same defect,
but WP-B12b's contract for that branch is bit-identity and the class was never
measured there; it is recorded as an open item below rather than changed on an
argument.

**The boundary, bisected AT THE SHIPPED CALL SITE, two-sided.**
`resolve_exit_index` is monkeypatched so the index is a free variable, and
the bundle handed to the function is a TRIPWIRE whose first attribute access
raises -- the function touches `beamlets.positions` immediately after the
guards and before anything expensive -- so a 60-step bisection runs the real
call site for the cost of the guard.  Identical on both builds:

| `z_image` | `fga._immersed_exit_tolerance` | the GBD site's boundary | relative gap | 1.01x | 0.99x |
|---|---|---|---|---|---|
| 0 | 1.000000e-03 | 1.000000e-03 | 1.73e-15 | refused | served |
| lambda (633 nm) | 1.000000e-03 | 1.000000e-03 | 1.73e-15 | refused | served |
| 10 um | 6.330000e-05 | 6.330000e-05 | 1.70e-13 | refused | served |
| 0.35 mm | 1.808571e-06 | 1.808571e-06 | 3.53e-11 | refused | served |
| 2 mm | 3.165000e-07 | 3.165000e-07 | 3.09e-10 | refused | served |
| 10 mm | 6.330000e-08 | 6.330000e-08 | 3.20e-10 | refused | served |

On the PRE tree the same bisection reports UNGUARDED at every leg (an exit
index of 2.0 is served), which is the other half of the contrast.

**The pins.**  `tests/unit/test_wp_b12b_round2.py`:
`test_an_immersed_exit_is_refused_at_every_local_gbd_entry_point` (4 ids, two
media each), `test_an_air_terminated_prescription_is_not_refused_at_any_entry_point`
(4 ids), `test_the_gbd_site_refuses_at_the_fga_helper_s_own_tolerance` (the
bisection, plus a third claim: monkeypatching `fga._immersed_exit_tolerance`
to 7x its value moves the GBD site's boundary to the new number, which a
private copy of the arithmetic could not do -- the durability point
VERIFY-WAVE5-E D3 made for the `fga` pins), and
`test_deleting_the_immersed_guard_from_the_gbd_site_reddens_a_named_check`,
which deletes the guard statement from the shipped function's own source with
`ast` (so a reflow cannot make the mutation a silent no-op), rebinds it
through VERIFY-WP-B12b's own recompile vehicle, and asserts the NAMED check
goes red while the OTHER guard still fires.

---

## R2.2 D-4 (P2, LIBRARY) -- the mirror-terminated local branch is refused, and the refusal does not name a dead end

**What ships is the REFUSAL, not a sign fix**, exactly as the verification
recommended: `Nz2` feeds `new_dir`, the leg length and the Moebius step, so a
one-line flip would trade a loud wrong answer for a quiet one.
`_require_forward_going_local_exit` raises `NotImplementedError` when the last
surface (skipping trailing coordinate breaks, through
`_last_optical_surface`) is a mirror.

**The two headline numbers, RE-MEASURED rather than quoted**
(`probe_r5_mirror.py`, the PRE archive tree, the verifier's own concave
R = -15 mm fixture, its own 3-D tracer; Windows and WSL identical to every
printed digit but one ULP of the leg residual).  A refusal whose message
quotes a number nobody re-measured is the right-conclusion-wrong-numbers
shape `docs/TESTING_STANDARDS.md` warns about, so both were re-taken, at BOTH
signs of `z_image`:

| arm | returned spot RMS | vs the TRUE focal RMS (7.118e-09 m) | returned `N` sign | leg residual vs the true optical path |
|---|---|---|---|---|
| `z_image = +|f|` (+7.499 mm) | **1.4334e-04 m** | **20137x** | **+1** (traced `-1`) | 3.285e-04 waves |
| `z_image = -|f|` | 7.1183e-09 m | **0.9999999999995587** | **+1** (traced `-1`) | **0.4911 waves** |

Three readings, and one correction to this package's own message.

1. **The mechanism is confirmed exactly.**  The returned spot RMS at
   `z_image = +|f|` (1.4334e-04 m) EQUALS the traced spot RMS at
   `z = -|f|` (1.4334e-04 m).  That is what "the returned positions are the
   truth at the mirrored plane" means, measured rather than argued: `dt.ux`
   has already flipped with `N`, and `Nz2` has not, so the two sign errors
   compose into a reflection of the image plane.
2. **At `z_image = -|f|` nothing in the spot warns the caller.**  The
   transverse positions reproduce the true focus to one part in `1e12` of the
   focal RMS -- and the leg is still wrong by **0.4911 waves** after the
   circular mean is removed, reproducing VERIFY-WP-B12b's 0.48 waves.  This is
   the silent half, and it is why a refusal rather than a warning is the right
   shape.
3. **The 7756x ratio is quadrature-dependent; the decision is not.**  Its
   denominator is a diffraction-scale number, so it moves with the ray
   quadrature and pupil weighting the oracle uses (5.001e-08 m in
   VERIFY-WP-B12b, 7.118e-09 m on my 121-ray Gaussian-weighted quadrature).
   The shipped message was edited to say "four decades wider than the traced
   one -- 7.8e3 x and 2.0e4 x on two independent ray quadratures" instead of
   pinning one of them as if it were exact.

**The remedy the open item named does not exist -- re-measured.**  On both
builds and in both trees:

| `world_output_plane` on a terminating mirror | `'auto'` | explicit `(p0, R_out)` |
|---|---|---|
| CURVED (R = -15 mm, and my own R = -8 mm) | `NotImplementedError` ("curved (powered) fold mirrors are not yet supported") | the same `NotImplementedError` |
| FLAT | `ValueError` (`paraxial_focus_world`: no focus to find behind a flat fold) | **SERVED** |

So the shipped message distinguishes the two instead of collapsing them: it
says no route serves a CURVED terminating mirror yet, and names
`world_output_plane=(p0, R_out)` for a FLAT one.  `test_the_mirror_refusal_
does_not_send_the_caller_to_a_dead_end` asserts both halves *after measuring
them in the same id*, so the message is checked against what the other branch
actually does on the running build, not against what it said last time.

**Scope, stated as a decision.**  The guard covers a mirror-TERMINATED
prescription -- the class VERIFY-WP-B12b measured.  A fold mirror in the
MIDDLE of a prescription reaching the local branch is left alone: it is what
`world_output_plane` exists for (and that branch serves a flat fold: the
periscope id in `tests/unit/test_gbd_feature_complete.py` is green
unchanged), it is LOUD rather than silent through the local branch, and
widening the guard would change behaviour nothing here has measured.
`test_a_transmissive_prescription_and_a_mid_prescription_fold_are_served`
pins that scope with the premise that the fixture really does contain a
mirror and really does not end in one.

**The pins.**  `test_a_mirror_terminated_prescription_is_refused_on_the_local_branch`
(3 fixtures: the verifier's R = -15 mm, my R = -8 mm, and a FLAT one -- the
flat arm matters because the defect is the unsigned `Nz2`, which has nothing
to do with the sag), premise-gated on the library's own
`_exit_direction_sign`; and
`test_deleting_the_mirror_guard_from_the_gbd_site_reddens_a_named_check`,
which additionally asserts that the UNGUARDED branch returns `N` sign `+1`
where the prescription sends the light toward `-1`.

**A defect in the verification's own gate, found while closing this.**
`tests/unit/test_verify_b12b_gbd_projection.py::test_a_mirror_terminated_local_
branch_is_refused_or_carries_the_direction` was
`xfail(raises=AssertionError, strict=True)` with the note "this xfail turns
RED the day either remedy lands, which is the point".  It would not have:
BOTH ends of the id raise `AssertionError` -- the served arm through its final
assertion, the refused arm through an `except` clause that re-raises as one --
so a landing remedy leaves it quietly xfailed, and `strict=True` only catches
an XPASS.  The xfail is removed, the refusal asserted directly, the
world-branch premise kept, and the message checked for the curved/flat
distinction.  That file now reads **9 passed** where it read 8 passed +
1 xfailed.

---

## R2.3 Byte identity where no guard fires

Two independent checks, because they fail differently.

**Archive to archive** (`probe_r2_identity.py`, Windows, PRE = `git archive
76019ede` against this branch, `LUMENAIRY_MEM_BUDGET_MB=2048` pinned on both
arms).  Fifteen air-terminated fixtures -- VERIFY-WP-B12b's eight (one optic,
last surface varied: conic, conic + k, even asphere, flat-base asphere,
biconic, freeform, field-frame decentre, flat last), WP-B12b's own six, and
mine -- at TWO planes each (the exit vertex and the fixture's own traced best
focus):

> **30 of 30 field digests IDENTICAL, 0 differ**, plus the three local public
> entry points byte-identical to one another in BOTH trees
> (`f799fc7c4666b585...`).

Two of those digests are this package's own published ones, reproduced from a
different probe and a different oracle module: `077c35f46f9f283c32736e99` for
the flat-base aspheric at its focus (VERIFY-WP-B12b section 6.1's POST
Windows digest) and `9c44ed861a987b97d2ce2065` for the flat control at the
vertex (its section 5.1).

**In-process, guard-deleted** (`test_neither_guard_moves_one_byte_of_an_air_
terminated_field`).  The same function with BOTH guard statements removed from
its own source by `ast` and rebound returns fields whose SHA-256 equals the
shipped ones on two air fixtures, at a pinned budget, with the mutation itself
premise-gated (the recompiled function must be a different object, and it must
SERVE the immersed fixture, or the identity would be vacuous).

---

## R2.4 D-2 (P3) -- the assertion that could not fail

`test_the_module_carries_no_second_sag_kernel` asserted
`"'radius'" not in fn_flat` on a token stream built two lines above it with
`tokenize.STRING` tokens DROPPED.  A quoted attribute name IS a STRING token,
so the searched text could never contain it.

**Proven, not argued.**  The PRE source of the beamlet function at commit
`1218b24f` -- which literally reads the last surface's radius through
`getattr` -- is committed as DATA in
`validation/probe_wp_b12b_round2/pre_b12_beamlet_fn.py.txt`, extracted with
`git show` + `ast` and cut at its first `def` so the explanatory header can
neither satisfy nor defeat either form.
`test_the_second_sag_kernel_check_now_fails_against_the_pre_source` runs all
three arms:

| arm | result |
|---|---|
| the OLD form, verbatim, against the PRE text | **passes** -- the defect |
| the REPAIRED check against the PRE text | **fails**, naming the read it found |
| the REPAIRED check against the SHIPPED function | passes |

The repaired check is one function,
`test_audit2609_b12b_gbd_projection.last_surface_radius_reads`, which both
files call, so the two cannot drift.  It searches the SOURCE TEXT with
whitespace removed for three spellings (the `getattr` form the deleted copy
used, its double-quoted twin, and a plain attribute read); the cost is that a
COMMENT could trip it, which is the conservative direction.

---

## R2.5 D-3 (P3) -- section 7.4's published pair, corrected

Section 7.4 published **8.318e-17 / 8.298e-17** radians for probe C's flat
control.  The shipped JSONs contain `6.796869888613907e-17` (Windows) and
`6.760231407720553e-17` (WSL).  Re-running `probe_c_decompose.py` unchanged on
this tree reproduces both files number for number on both builds -- the only
line that moves is the embedded `"version"` string (5.47.0 -> 5.47.1), which
tracks the tree and not the measurement, so the artefacts were left as
committed.  Section 7.4 now carries the JSONs' numbers with the correction
dated in place; the re-run is recorded in `probe_r4_probec_*.json`.

---

## R2.6 D-6 (P3, evidence) -- the memory budget, re-taken and now pinned

Re-measured through the PUBLIC entry with ONE CHILD PROCESS PER BUDGET, so
the variable is read at the value the child started with and nothing caches
across arms:

| `LUMENAIRY_MEM_BUDGET_MB` | Windows digest | WSL digest | max abs diff vs unset |
|---|---|---|---|
| unset | `f799fc7c...` | `2d7f8e01...` | 0 |
| 4096 | `f799fc7c...` | `2d7f8e01...` | 0 |
| 2048 | `f799fc7c...` | `2d7f8e01...` | 0 |
| 512 | `f799fc7c...` | `2d7f8e01...` | 0 |
| **64** | **`7dafcd23...`** | **`be147743...`** | 1.741e-15 / 1.759e-15 relative |
| **8** | **`cb6a219f...`** | **`44618976...`** | 2.162e-15 / 2.005e-15 relative |

The bytes move and the field does not: the variable is a hard CEILING on the
`mem_budget_mb` keyword, so anything at or above the 512 default is a no-op,
and below it `_reconstruct_windowed`'s chunking regroups a scatter-add.

**What was changed as a result.**  `LUMENAIRY_MEM_BUDGET_MB` is now pinned by
an autouse fixture in all three B12b test files, and `mem_budget_mb` is passed
explicitly wherever the entry point has it.  One asymmetry is worth recording:
`propagate_gbd_through_prescription` has NO `mem_budget_mb` keyword at all, so
for that entry the environment variable is the ONLY pin -- which is why the
pin is a fixture and not a keyword at each call site.  The finding is also in
that file's module docstring, in section 7's preamble above, and in the
CHANGELOG entry.

---

## R2.7 D-7, D-8 and the nit inside claim 4 -- restated in place

* **D-7** (section 4.2).  "The five exit-vertex rows agree to 1.3e-06 of
  fidelity" is the largest deviation from their MEAN; the SPREAD is
  **1.98e-06**.  Both are now given, with the five readings.
* **D-8** (section 5, the JAX row).  "`grep -n 'jax\|jnp'
  lumenairy/propagators/gbd.py` finds no import" returns **12** matches
  (docstrings, `is_jax_array`, a `jnp.at[].add` comment).  The substance --
  no `import jax`, no `import jax.numpy`, no JAX per-surface GBD path -- is
  right and is restated as what was actually checked.
* **Claim 4's nit** (section 4.5).  "Their last-bits agreement is a property
  of the build" is wrong.  With the beamlet FRAME named, the verification
  found the two entry points byte-identical on both builds in both trees, and
  this round reproduces that on a third fixture: `apply_real_lens_gbd`,
  `apply_real_lens_universal(method='gbd')` and
  `propagate_gbd_through_prescription(per_surface=True)` all return
  `f799fc7c4666b585...` in BOTH trees on Windows.  What separates them is the
  frame and the chunk boundaries the coherent sum is grouped by -- one route
  prunes dark beamlets up front and the other lets the trace vignette them,
  reaching the same live set by different roads.  The test file's decision
  (byte identity for the dispatcher pair, a 0.999 fidelity bar for this one)
  is unchanged and still right; it just holds for a different reason.

---

## R2.8 Open items after round 2

1. **The `world_output_plane` branch's own image leg is index-free too.**
   Measured here (`probe_r1_guards.py`, both builds): that branch SERVES an
   immersed prescription in both trees.  It is not guarded, because its
   `surfs` come from `_unfolded_equivalent_surfaces` (which rewrites the
   surface list) and refusing on a rewritten list was not measured; WP-B12b's
   bit-identity contract for that branch was kept instead.
2. **A fold mirror in the MIDDLE of a prescription through the LOCAL branch.**
   Out of the new guard's scope by decision, not by oversight (R2.2).  It is
   loud today, and `world_output_plane` serves the flat case.
3. **A signed-`N` local branch that would SERVE a mirror-terminated
   prescription** is not attempted.  The refusal is a cost decision.  What it
   would take: a signed propagation-direction convention carried through
   `new_dir`, the leg length `t` and `_freespace_tensor_moebius_np`'s branch
   choice at once, plus a reconstruction frame that knows which way the
   grid's `+z` points; and a 3-D oracle arm for a CURVED terminating mirror,
   which neither package has (`probe_r5_mirror.py`'s tracer is the starting
   point -- it already produces the exit-vertex state and the true focal spot
   for this class, which is what such an arm would score against).  The world
   branch's own refusal of a powered fold would have to be lifted in the same
   change or the two branches would disagree about which classes exist.
4. **The GPU reconstruction path and the vector GBD path** are untouched and
   unmeasured here, as in the verification: the guards sit upstream of both.
5. **The CI cross-build spread.**  Both builds here are the same box; the
   runner mix was not sampled.  Every bar in the new file is derived at run
   time from a quantity the running build measures.

---

## R2.9 Tests run

| selection | build | result |
|---|---|---|
| `test_wp_b12b_round2.py` (NEW, 22 ids) | Windows | see R2.10 |
| `test_audit2609_b12b_gbd_projection.py` (14) + `test_verify_b12b_gbd_projection.py` (9) + `test_wp_b12b_round2.py` (22) | Windows | **45 passed** in 96.7 s |
| the GBD + FGA + reference-plane selection (21 files) | Windows / WSL | see R2.10 |
| the census / walker / dispatcher-pin / public-API / doc-consistency sweep + `test_audit_except_budget.py` | Windows | see R2.10 |

`test_verify_b12b_gbd_projection.py` reads **9 passed** where it read 8 passed
+ 1 xfailed, because its mirror id is now a real decision (R2.2).

| gate | result |
|---|---|
| `python scripts/record_history_fingerprints.py --check` | **OK: every history document matches its module** (`lumenairy.propagators.gbd.md` re-recorded) |
| `python scripts/check_doc_identifiers.py` | **OK**, 621 distinct, 0 unresolved |
| `.test_durations` | 16 396 -> **16 418** entries, valid JSON, **22** new ids, largest **5.03 s** |

---

## R2.10 What I could not measure

1. **The CI cross-build spread** -- as above.
2. **The GPU reconstruction path** (`use_gpu=True`): no CUDA device here.
   The guards run before any device transfer, so they are upstream of it, but
   that is an argument and not a measurement.
3. **A CURVED terminating mirror served correctly by the world branch.**  I
   did not attempt it; R2.8 item 3 says what it would take.  What I did
   measure is that the branch refuses the class today, on both builds and in
   both trees, so the refusal's message is not sending anyone anywhere that
   works.
4. **`scripts/check_source_line_citations.py`** audits the topmost RELEASED
   block (v5.47.1 here) and reports "no source-file:line citations in the
   v5.47.1 block; nothing to verify" (exit 2).  It does not look at
   `## [Unreleased]`, so it says nothing about this package either way; the
   reading is identical before and after this branch.
