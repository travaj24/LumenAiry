# WP-B6 changelog text -- PMM: the ultraspherical basis (measured, declined) and the PMM-2D tensor operator cache

Release text for **5.47.0**.  The two items WP-A12 and WP-A13 deferred with
designs: finding **G3** of `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11.md`
§11 (partition report `.../PMM-1D.md`, "Alternative algorithms" (c), via
`fixes/WP-A12_REPORT.md` §6 item 3) and finding **G10(d)** of §12 (partition
report `.../PMM-2D.md`, via `fixes/WP-A13_REPORT.md` §6.1).

No default moved, in either item; there is no Migration note.

### Fixed -- PMM 2-D stacks: a tensor layer's projected operators are built once per geometry, not once per sweep point

`PMM2DStackHybrid._build_layer_modes` passed `lops=None` for a tensor layer, so
`_geom_cache` stored `(ax, ay, None)`: the nodal axis build was reused across a
sweep, but `_tensor_layer_modes` rebuilt the per-axis projections
(`_axis_projection` + `pinv`) and the `_proj` sandwiches at **every wavelength
and every angle**, where the scalar branch cached them (finding **G10(d)**, P2;
`lumenairy/elements/pmm/twod_jones.py:149`, `lumenairy/elements/pmm/stack2d.py:1187`).

The source-free half of the assembly is now
`twod_jones._tensor_projected_ops(ax, ay, x_walls, y_walls, tile_i, ox, oy,
formulation)` -- the off-plane Schur fold and the three discretization branches
**moved verbatim** (`lumenairy/elements/pmm/twod_jones.py:149`), returning
`dict(kind, Gx0F, IpxF, Gy0F, IpyF, Cxx, Cxy, Cyx, Cyy, EZZ, oop)`.  `kind`
names which axes carry a k0-free derivative part, which is exactly where the
three branches differ: `'uniform'` has none and is `diag(k)` on both axes
(`lumenairy/elements/pmm/twod_jones.py:208`); `'x'`/`'y'` (a separable cell) has
one, `kron(Iy, g1)/k0 + kx0*kron(Iy, ip1)` on the patterned axis and exact
`diag(k)` on the wall-less one
(`lumenairy/elements/pmm/twod_jones.py:287`, `:298`); `'xy'` (crossed) has both,
in the scalar branch's `Gx0F/k0 + kx0*Ip` shape
(`lumenairy/elements/pmm/twod_jones.py:333`).  `_tensor_layer_modes` keeps its
signature and gains `ops=` (`lumenairy/elements/pmm/twod_jones.py:372`);
handed one, it rebuilds `GxF`/`GyF` from `kind` with the same expression in the
same order (`lumenairy/elements/pmm/twod_jones.py:413`) and goes straight to the
circular restriction, the fold gate, the block-eig gauge and the eig.
`PMM2DStackHybrid` caches it beside the scalar `lops`
(`lumenairy/elements/pmm/stack2d.py:1189`), on the FULL order box so one entry
serves both truncations (`keep` is applied at use, as `_restrict_lops` already
did for the scalar branch).

MEASURED over a 9-point sweep, as a deterministic build count (this box runs
~20 sibling agents; WP-A12 records three failed timing probes for the same
reason).  `_tensor_layer_modes` is still entered 9 times -- the eig genuinely
depends on the source -- while the source-free assembly now runs once:

| fixture | sweep | assemblies | assembly seconds | whole sweep (min of 3) |
|---|---|---|---|---|
| crossed, degree 11, n_orders 7 | wavelength | 9 -> **1** | 0.474 -> **0.009 s** | 11.186 -> 10.269 s |
| crossed, degree 11, n_orders 7 | angle | 9 -> **1** | 0.510 -> **0.009 s** | 10.894 -> 9.925 s |
| crossed, degree 9, n_orders 5 | wavelength | 9 -> **1** | 0.169 -> 0.002 s | 2.363 -> 2.053 s |
| separable, degree 11, n_orders 7 | wavelength | 9 -> **1** | 0.031 -> 0.006 s | within noise |
| out-of-plane, degree 9, n_orders 5 | wavelength | 9 -> **1** | 0.036 -> 0.005 s | within noise |

The crossed branch is where it matters: **4.2 % of the whole solve**, the same
order as the audit's own "`_proj` at 0.47 s of a ~20 s solve".  On the separable
and out-of-plane branches the assembly is 0.03-0.05 s per sweep and no
whole-solve claim is made.  The cost is retention: a tensor `_geom_cache` entry
grows 0.13 -> 6.31 MiB at degree 11 / n_orders 7 (eleven dense `Nf x Nf`
complex128 blocks).  `LayerCache` prices that as it prices everything else, and
refusal degrades to a rebuild, never to a different answer.

**Bit-identical, and gated as such.**  Against the pre-change tree (`git archive`
extracted read-only, imported in a child process with `lumenairy.__file__`
asserted): the seven operators plus the four out-of-plane blocks and the
block-eig gauge, captured at the hand-off to `_layer_eigenmodes_tensor` over
uniform / separable-x / separable-y / crossed cells, `'laurent'` / `'li'` /
`'fff_nv'`, in-plane and out-of-plane, normal and oblique, rectangular and
circular truncation, vertical and slanted -- **210 arrays, 0 differ, worst
|A-B| = 0.000e+00**.  End to end over 28 fixtures (stacks, both sweep axes, the
even-parity fold, the parity-sign block reduction, `pmm_jones_2d`,
`pmm_jones_1d_conical_tensor`, `PMMStack`'s conical tensor-segment path) --
**134 arrays, 0 differ, 0.000e+00**.  The 1-D PMM surface, which this release
does not touch at all -- **172 arrays, 0 differ, 0.000e+00**.  In-tree the same
comparison runs on 9 branches x 2 truncations every suite run, with a
non-vacuity guard that feeds a deliberately perturbed build through and
requires the perturbation to come out the other side.

One key change came with it: `_geom_key` now carries `formulation`
(`lumenairy/elements/pmm/stack2d.py:609`).  The cached scalar `lops` never
depended on it (they carry every rule's operator side by side and the caller
routes), but the cached tensor operators do -- `EZZ` is `inv([[1/e_zz]])` under
`'li'` and the direct `[[e_zz]]` otherwise -- and `formulation` is a plain public
attribute with no property guard, the W7 A11 shape.  Without the key change,
mutating it after a solve would have served a stale tensor build with no signal.
It can only split keys that were previously shared, so its worst case is a
rebuild.

Gate: `tests/unit/test_audit2609_b6_pmm_basis_and_tensor_cache.py` (44 tests) --
the operator-level bit-identity, the `kind` algebra, the once-per-sweep build
count with a priced-out fail-before arm whose answers must match bit for bit,
the one-entry-serves-both-truncations claim, the stale-formulation contract, and
the W7 A13 read-only guard extended to the new cache slot.

### Measured and NOT adopted -- the ultraspherical (Gegenbauer) basis for the TM wall corner

The audit's alternative (c) -- swap the Legendre/GLL nodal basis in
`_gll_nodes_weights` / `_lagrange_derivative_matrix` for an ultraspherical
Gauss-Lobatto one and "recover exponential convergence for the TM wall-corner
singularity where the Legendre/GLL basis is `O(N^-2.7)`" -- was built as a
one-parameter family (lambda = 1/2 being today's rule), measured against the
auditor's Au/air TM fixture with an extrapolated RCWA oracle
(`rcwa_efficiency_1d` at n_orders 101..1201 + `rcwa_extrapolate`, limit
0.415512036392 with a 2.62e-06 estimator floor), and **declined**.  No `basis=`
knob ships; every 1-D number is unchanged to the bit.  Two measurements decide
it, both re-derived on the running build by the gate tests rather than quoted:

* **The basis swap is a no-op.**  With the element integrals evaluated EXACTLY,
  all six lambda in {0, 0.25, 0.5, 0.75, 1, 1.5} return ONE answer -- efficiencies
  agreeing to **3.1e-13** -- because a change of nodal basis of the same C0
  piecewise-`P_N` space transforms the operators by a congruence and the pencil
  by a similarity.  The corner rate belongs to the space, not to the nodes in it.
* **What lambda actually varies is the quadrature, and only GLL keeps
  summation-by-parts.**  The shipped lumped mass is legitimate because the GLL
  rule is exact to degree `2N-1`, making `M D + (M D)^T = diag(-1, 0, ..., 0, +1)`
  hold EXACTLY -- measured 1.3e-14 .. 1.9e-13 over degree 8..32, against
  3.7e-01 .. 1.5e+00 at every other lambda (exact integration restores it for all
  of them, which is the control).  On a lossless grating at oblique incidence
  that shows up directly: `|sum R + sum T - 1|` goes from **3.0e-14** to
  2.8e-08 (lambda = 0), 6.6e-07 (lambda = 1), 1.2e-05 (lambda = 1.5).

The lambda < 1/2 arms do beat Legendre on TM -- local rate 2.2-2.5 -> 3.0-4.1 on
the gate fixture and 5-86x less error at degree 32 across seven cells -- but that
is the quadrature error cancelling against the corner, and it is paid for:
Au/air TE at degree 24 goes from 1.1e-09 to 1.3e-07 (lambda = 0) or 2.4e-06
(lambda = 1), and at lambda >= 1.0 the closure defect flips
`_energy_clean_pick`'s "evidently lossless" classification (`< 1e-6`) on 6 of 8
lossless cell/polarization rows, which moves which degree `stabilize=True` -- the
default on `pmm_efficiency_1d` -- returns.  Nothing in the family is exponential.

`_gll_nodes_weights`'s docstring now carries that reasoning and those numbers
(`lumenairy/elements/pmm/_core.py:357`), so the next reader of the two functions
the audit named finds the measurement rather than the proposal; the full rate
ladder is in `fixes/WP-B6_REPORT.md` §2.1.  The recommendation for the corner is
unchanged from WP-A12 §6 item 4: a genuine hp mesh (geometric grading with a
linearly decreasing degree toward the corner, Babuska-Guo), which is the only
member of this family that changes the space rather than the nodes in it.

Gate: `tests/unit/test_audit2609_b6_pmm_basis_and_tensor_cache.py`
(`test_b6_only_the_gll_rule_satisfies_summation_by_parts`,
`test_b6_an_exactly_integrated_ultraspherical_basis_cannot_move_the_answer`,
`test_b6_a_non_gll_nodal_rule_loses_the_exact_energy_identity`) -- the record of
what a second attempt has to re-measure, not just of what was tried.
