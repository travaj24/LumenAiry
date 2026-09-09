
---

## 5.  What this does NOT cover

* **TAPER.**  A shear is a tilted axis with a CONSTANT cross-section.  No shear
  absorbs a dilation -- measured at `1.00x` in
  `BUILD_PMM2D_SLANT_METRIC_2026_08_16` M5, on the one-wall taper carrying the
  maximum shear content a taper can have.  The taper's `sqrt(g)` is
  z-dependent, which brings back the dilation generator, a non-normal
  `q -> -conj(q)` pencil with no valid mode selector, and a distorted far field.
  None of that arises here, and none of it is solved here.
* **Mixed slants between PATTERNED layers** (S1.5): the frame offset between two
  differently-sheared patterned regions is a real lateral translation of one
  nodal grid relative to the other.  Exact only when the offset is a whole
  number of grid cells; otherwise it needs an interpolation the pure basis does
  not have.  Homogeneous / uniform neighbours are unaffected (the offset is a
  gauge).  **The build must refuse this**, loudly.
* **Curved walls** -- Phase E, unchanged by this work.
* **`retain_internal` / `layer_absorption` below a slanted layer** were not
  measured.  The frame-anchor phase is a per-order far-field correction here;
  an internal-field probe evaluated at a plane inside or below a sheared layer
  is in the frame, not the lab, and needs the same treatment.  The first build
  should either carry it through `_flux_at` or refuse `retain_internal` on a
  slanted stack.

---

## 6.  THE INTEGRATION ROUTE

Everything below is a change to two files.  No new module, no new basis
function, no new cascade, no new far field.

### 6.1  `lumenairy/elements/pmm/twod_staggered.py`

**(1) `Granet2DTransverseE.__init__(..., slant=(0.0, 0.0))`.**

```
  # a sheared cell is an OUT-OF-PLANE cell in the frame, always
  if slant != (0.0, 0.0):
      cell33 = promote_scalar_to_tensor(eps_cell)       # (Nx,Ny) -> (Nx,Ny,3,3)
      rot    = _OOP_ROT_SIGN
      cell33 = flip_offplane(cell33, rot)               # e13,e23,e31,e32 *= rot
      tx, ty = rot * slant[0], rot * slant[1]           # t rotates WITH them
      self.eps_cell = cov_congruence(cell33, tx, ty)    # A^-1 eps A^-T
      self.offplane = True
  else:  # unchanged dispatch, byte for byte
```

The rot-on-`t` is not cosmetic: `eps^{lm}(R eps R, -t) = R eps^{lm}(eps, t) R`
is what makes the two consistent (S1.4(c)), and M2's `a-` arms measure the
failure of getting it wrong (`1.3e-02 .. 1.3e-01`).

**(2) `_assemble_oop` gains one guarded block.**  Six `np.kron`s and four
row additions, exactly as in `slant_lib.SlantSolver._assemble_slant`:

| new per-axis matrix | call |
|---|---|
| `Mtb_x`, `Mtb_y` | `b.mass(b.Btilde, b.B)` |
| `Mbt_x`, `Mbt_y` | `b.mass(b.B, b.Btilde)` |
| `dtb_x`, `dtb_y` | `-(dbt).conj().T` -- the integration-by-parts identity the assembly already uses for `CwE1`/`CwE2`; the ELEMENTWISE `b.mixed(b.Btilde, b.B)` would silently drop the jump deltas and is NOT the same matrix |
| `Ctt_x`, `Ctt_y` | already in `_axis_mats` |

`if tx == 0 and ty == 0` skips the whole block, which is what makes the slant-0
path bit-identical (G0a: 0 differing bytes).

**(3) `_region_modes_oop`: UNCHANGED.**  The retained state is the covariant
tangential state = the lab-Cartesian tangential state, so the Cholesky
whitening, the flux-based split with the deep-decay override, `_OOP_H_GAUGE`,
and the `2q^2 / 2q^2` contract all keep their meanings.  M5 measures each.

**(4) `pmm_efficiency_2d_staggered` / `pmm_jones_2d_staggered`: add `slant=` and
forward.**  Match `PMM2DStackHybrid.add_layer`'s signature exactly
(`(t_x, t_y)` or a bare scalar `t_x`, a TANGENT, cross-section at the layer's
TOP) so a caller can move a cell between the two 2-D engines without a
convention change.

**THE SIGN, measured twice and stated as a relation.**  The prototype's internal
`t` (the `t` of `x = u + t w` in S1.1) satisfies

```
    t_probe   =  - t_hybrid_public        (M4, 5.4e-03 vs 1.8e-01 / 3.2e-01)
    t_probe   =  - tan(slant_angle_1D)    (M3, 1.2e-03 vs 4.2e-01)
    => t_hybrid_public = tan(slant_angle_1D)     -- the hybrid and the 1-D
       scalar entry already share ONE public convention.
```

So the build takes the PUBLIC `slant` and passes `-slant` into the congruence
and the six blocks, and the three engines then agree.  Which way a positive
`slant` physically tilts the wall is a library convention this campaign did not
independently re-derive -- M3 and M4 pin the RELATION, which is what an
implementation needs, and each pins it against an engine validated elsewhere.

### 6.2  `lumenairy/elements/pmm/stack2d_pure.py`

**(5) `PMM2DStackPure.add_layer(..., slant=None)`.**  A slanted layer sets
`any_oop = True` (already the flag that promotes the stack to the GENERALIZED
cascade -- no cascade work at all), and the `eig_cache` key must include the
slant.

**(6) THE FRAME-ANCHOR PHASE -- the one genuinely new line of physics.**
In `solve`, after `kxv`/`kyv` are built:

```
  shx = sum(L["slant"][0] * L["thickness"] for L in self._layers)
  shy = sum(L["slant"][1] * L["thickness"] for L in self._layers)
  tphase = exp(-1j * k0 * (kxv * shx + kyv * shy))
  tx_ord *= tphase ;  ty_ord *= tphase          # TRANSMITTED only
```

`S11` and the reflection Jones need NOTHING (S1.5).  Omitting it leaves R, T and
the reflection Jones exactly right and the transmission Jones wrong by up to
`6.2e-01` -- a silent-wrong of the worst class, so its gate (M1b, both signs on
a uniform null at oblique) is not optional.

**(7) REFUSALS, all raising with the reason:**
* a slanted PATTERNED layer whose neighbour is a PATTERNED layer with a
  DIFFERENT slant (a VERTICAL patterned layer included) -- S1.5;
* dispersive (callable) and traced (JAX) layers -- mirror the hybrid;
* `retain_internal` on a slanted stack, unless `_flux_at` is taught the frame
  (S5).

**(8) [H] The Wood-anomaly nudge list.**  `solve` feeds `_grazing_safe_wavelength`
the tensor layers' diagonals.  A slanted SCALAR layer's covariant diagonal is
`eps (1 + t_x^2)` etc., so its modal spectrum reaches higher than `eps`; whether
it should be added to `_eps_gr` is NOT settled by this campaign and the build
should measure it (a slanted cell walked onto a Rayleigh cutoff).

### 6.3  Gates for the build (every one of them measured here)

| gate | assertion | measured here |
|---|---|---|
| B1 | `slant = 0` is BYTE-IDENTICAL to the pre-slant library (pencil AND end-to-end) | G0a / G0b |
| B2 | uniform layer + any slant is a no-op, M-ladder to machine precision | M1 / M1c |
| B3 | the frame-anchor phase, BOTH signs, on the transmission Jones at oblique | M1b |
| B4 | sheared-frame dispersion == `exact_kz_roots + t.alpha`, PLUS the two ablations (drop the six blocks / drop the congruence) | M2 |
| B5 | y-uniform stripe == `pmm_efficiency_1d_slanted` per order, both pols, normal + oblique, slant 10/20/35 -- and the SIGN arm | M3 |
| B6 | slant x OUT-OF-PLANE stripe == `pmm_jones_1d_slanted` per order | M7 |
| B7 | census: split exactly `2q^2/2q^2`, `min Re(lam_f) >= 0`, no band-exceeding mode carrying flux | M5 |
| B8 | closure does not grow with depth (0.25 / 1 / 3 lam) and `max fwd growth == 1.0` | M5 |
| B9 | one layer of depth `d` == two of `d/2` at the same slant | M5 |
| B10 | NO-FLOOR survives: the answer does not move with `n_orders` | M5 |
| B11 | the mixed-slant refusal fires | (new) |

B4 and B5 are the ones that cannot be replaced by an energy check: the lossless
trap is the named hazard for slant work, and M4's own numbers show a staircase
that conserves energy while sitting decades from the truth.
