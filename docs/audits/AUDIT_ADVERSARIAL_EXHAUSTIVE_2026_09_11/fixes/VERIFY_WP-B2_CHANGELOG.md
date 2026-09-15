# VERIFY-WP-B2 changelog text — the 1-D remap's input window, and the launch pitch the caller is told about

Release 5.47.0, alongside `WP-B2_CHANGELOG.md`.  Finding **L9** of
`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11.md` §2.1 — the
independent re-verification of WP-B2 (`VERIFY_WP-B2.md`).  Files:
`lumenairy/elements/_lens_real.py`, `lumenairy/elements/lens_config.py`
(one docstring line).

---

### Fixed -- apply_real_lens: `displaced_mode='remap'` on a ROTATIONALLY SYMMETRIC element no longer returns a field with a dead crescent on one side (L9)

The 1-D symmetric exit-plane remap carries the input envelope by reading it at
each exit point's ENTRANCE height, `X * scale` with `scale = h_in / r_out`
(`_lens_real.py:1926`).  A converging element walks the ray inward, so
`scale > 1` and that read runs off the **+x** end of the field axis
`(arange(N) - N/2) * dx` while its mirror -- one whole sample further out on
-x -- is still on the grid.  `map_coordinates(mode='constant')` returns `cval`
outside the input extent rather than interpolating toward it, so the +x rim
came back **exactly zero** while the -x rim came back at full envelope.

This is the same defect WP-B2 fixed in the 2-D remap, and it did not need a
decentred element or a mirror pair to show: a CENTRED, rotationally symmetric
Gaussian through a rotationally symmetric singlet (`R = 42.5 / -63 mm`, 4.2 mm
of n = 1.5093 glass) was enough.

| grid | paired mirror relL2 of \|E\| | pixels off by > 1e-9 of peak | worst pixel |
|---|---|---|---|
| N = 640, dx = 6.5 um, w0 = 2.4 mm | 3.301e-02 | 1236 | 0.489 of peak against an exact 0 on its mirror |
| N = 512, dx = 8 um, w0 = 3.0 mm | 4.586e-02 | 988 | 0.650 of peak against an exact 0 on its mirror |

The remap now carries the envelope over the largest CENTRED window the caller's
grid holds -- `|X * scale| <= x[-1]`, `|Y * scale| <= y[-1]`
(`_lens_real.py:1956`) -- and the same two readings are **1.18e-16** and
**1.01e-16** with zero pixels off.  With the input field itself decentred by
+-0.45 mm, the +d / -d pair mirrors to 1.16e-16 (was 3.56e-02).

WP-B2 recorded this as deferred (its §6.1) on the grounds that fixing it
"moves the byte-identity pin"
`test_niche_p10_...::test_symmetric_remap_is_the_p2_1d_remap_byte_identical`.
It does not: that pin compares `apply_real_lens(displaced_mode='remap')`
against a direct call to `_apply_displaced_remap`, so both sides move together.
It passes unchanged.

Blast radius, measured as SHA-256 of the raw output bytes over eleven paths
against the pre-change library: **exactly one** moves, this one.
`surface_model='thin'`, `'tangent_facet'`, `'tangent_facet_remap'`, the
displaced symmetric default, `displaced_mode='split'`,
`displaced_obliquity='pointwise'`, the 2-D transverse-walk remap and
`apply_real_lens_traced` are all bit-identical.

---

### Fixed -- apply_real_lens: the remap's smoothing warning quotes the launch pitch the TRACE uses, so the `displaced_n_side` it names really clears the bar (L9)

`_warn_if_remap_lattice_smooths` scored the launch pitch as
`2 * r_aperture / (n_side - 1)`.  The fan is thrown 3 % wider than the aperture
so the edge rays have interior Jacobian neighbours, so the pitch the trace
actually uses is `2 * 1.03 * r_aperture / (n_side - 1)` -- measured ratio
**1.03000** at `n_side` 181 / 257 / 513 (57.222 / 40.234 / 20.117 um traced
against 55.556 / 39.062 / 19.531 um quoted).

The consequence was in the message's own advice.  It offered a
`displaced_n_side` "to resolve the field pitch", computed from the same wrong
formula, and the lattice it named did not:

| field pitch | named before | its real launch pitch | bar (2 x field pitch) | cleared? |
|---|---|---|---|---|
| 8 um | 626 | 16.48 um | 16.00 um | **no** |
| 4 um | 1251 | 8.24 um | 8.00 um | **no** |

It silenced itself only because the silence test re-used the same formula.  A
warning that names a keyword value which does not fix what the message says it
fixes is the exact defect class this campaign exists to close -- and is the one
WP-B2's own §2.6 says it removed a prototype warning to avoid.

The fan factor is now one module constant, `_DISP_REMAP_2D_FAN_FACTOR = 1.03`
(`_lens_real.py:2074`), read by `_build_displaced_ray_map_2d` (which throws the
fan) and by `_warn_if_remap_lattice_smooths` (which scores it against the field
pitch), so the two cannot drift apart again.  The same two calls now name
**645** and **1289**, whose real pitches are 15.99 um and 8.00 um.  The message
also names the fan and the aperture separately instead of quoting the pitch
"across the traced aperture".

Test-visible change: the `displaced_n_side` a call is advised to pass is ~3 %
larger than before, and the warning fires on a launch pitch 3 % coarser than
before (i.e. marginally more often).  No output field moves.

---

### Changed -- the documented launch pitch of `displaced_n_side` is the traced pitch (L9)

`apply_real_lens`'s `displaced_n_side` docstring, the `_DISP_REMAP_2D_N_SIDE`
comment (including its pitch column: 55.6 / 39.1 / 19.5 / 9.8 um ->
**57.2 / 40.2 / 20.1 / 10.1 um**), `_normalise_displaced_n_side`'s refusal
message and `LensNumerics.displaced_n_side`'s docstring
(`lens_config.py:440`) all stated the pitch as `2 * r_aperture / (n - 1)`.  All
now state `2 * 1.03 * r_aperture / (n - 1)` and say why the factor is there.
`_build_displaced_ray_map_2d`'s docstring no longer describes the default as
"a fixed 181".

Documentation only; no behaviour depends on these strings except the refusal
message's text.

---

### Migration

`surface_model='displaced'` with `displaced_mode='remap'` on a **rotationally
symmetric** element (the 1-D remap) returns a different field at the outer rim:
the outermost ring of the input no longer contributes on EITHER side, where
before it contributed on -x and not on +x.  The illuminated core is
bit-identical.  Everything else in the family -- the 2-D transverse-walk remap,
the pointwise and meridional obliquity screens, `displaced_mode='split'`,
`surface_model='thin'` / `'tangent_facet'` / `'tangent_facet_remap'` and the
whole traced family -- is **unchanged, byte for byte**, verified by SHA-256 of
the output bytes against the pre-change library.

Callers who read the warning's `displaced_n_side=<n>` suggestion will be given
a value ~3 % larger than before; the previous value was ~3 % too coarse to
actually resolve the field pitch it claimed to.
