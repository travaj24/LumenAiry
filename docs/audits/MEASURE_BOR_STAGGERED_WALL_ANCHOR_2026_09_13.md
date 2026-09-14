# The BOR staggered PEC wall anchor -- what was measured, and why the default is `'rbig'`

Relocated from `lumenairy/elements/bor/coupled_radial_eigensolver.py` by WP-B11a
(audit `AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11` finding H6 / WP-A14 D6:
"~100 lines of measurement prose sit on one constant (`STAGGERED_WALL_ANCHOR`)").
The constant's own `#:` comment now states the live contract -- what each value
means and which one ships -- and points here for the numbers behind it.

Audit id: **W6-B1**.  Affected symbol:
`lumenairy.elements.bor.coupled_radial_eigensolver.STAGGERED_WALL_ANCHOR`.

## The two values

`'rbig'` (the default) -- `h = Rbig / (N + 0.5)`.  The outer stencil closes the
wall by forcing the tangential field to zero at the GHOST NODE (index `N`,
radius `(N + 0.5) h`), so THAT radius is the PEC wall; this spacing lands it
exactly on the `Rbig` the caller asked for.

`'ghost'` (legacy escape hatch) -- `h = Rbig / N`, which puts the ghost node --
and therefore the wall -- at `Rbig + h/2`: the discretized cavity is half a cell
LARGER than requested.  Retained only to reproduce the numbers the library
returned before the flip; it is a KNOWN-DEFECTIVE anchor, not a supported
alternative discretization.

## The convergence measurement

Homogeneous `m = 1` cylinder, `eps = 4`, `Rbig = 8`, `k0 = 2`; `gamma` against
the analytic `{j_{m,n}, j'_{m,n}} / Rbig`.

| anchor | order `p` (N = 60 -> 480) | relative bias at N = 60 | at N = 240 | at N = 480 |
| --- | --- | --- | --- | --- |
| `'ghost'` | 0.994 / 0.997 / 0.998 (FIRST order) | -8.26e-3 | -- | -1.04e-3 |
| `'rbig'` | 1.99 (SECOND order) | 6.52e-7 | 4.14e-8 | -- |

Under `'ghost'` the transverse wavenumbers converge to `j / (Rbig + h/2)`, and
the implied effective radius matches `Rbig + h/2` to 4-5 digits (ratio 0.9999 /
1.0000 / 1.0000 / 1.0000) -- which contradicted the module docstring's
"Convergence is 2nd-order in N (FD)".  `'rbig'` is FOUR DECADES better at
identical cost, and BOTH properties that make this basis worth having are
preserved: the exact discrete de Rham identity (3.6e-15 against 1.8e-15) and
machine-precision cascade energy (1.28e-14 against 1.47e-14 on the ring-grating
reproducer).

## Why the anchor moved and not the stencil

The competing antisymmetric-ghost repair (`Dn2f[N-1, N-1] = -2/h`,
`An2f[N-1, N-1] = 0`) also anchors the wall on `Rbig` and also reaches
`p = 2.000`, but was MEASURED to destroy both properties: de Rham 1.58e-2
(1.05e-3 relative) and cascade energy 2.06e-4.  The algebra says why -- for a
ghost `t * f_{N-1}` the de Rham residual is exactly `m t / (r_{N-1} Rbig)`, zero
only at `t = 0` -- so the ghost-value-ZERO stencil must stay and the spacing
must move.

## What the flip cost, and where the record lives

The flip moves every `BORStack` number towards the exact answer.  On the
`AUDIT_BOR_PROPAGATING_CUTOFF_ENERGY_2026_07_13` reproducer
(`tests/unit/test_audit_bor_grazing_cutoff.py`, `Rbig = 48 um`, `N = 256`,
`n = 1.41`, `lam = 1 um`) the cavity shrinks by `h/2 = 0.094 um` and MEASURES:

| quantity | `'ghost'` | `'rbig'` |
| --- | --- | --- |
| incident propagating orders | 319 | 318 |
| fundamental-mode R | 0.146135 | 0.142290 |
| min `q / k0` | 0.0493 | 0.0512 |
| energy closure | 6.8e-12 | 6.8e-12 (unaffected) |

The outermost near-grazing order that disappears belonged to the oversized
cavity.  `tests/unit/test_audit_bor_grazing_cutoff.py`'s pins carry the
deliberate-update record, and `tests/unit/test_niche_audit_w6_bor.py` pins the
shipped default and exercises both anchors.
