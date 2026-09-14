# VERIFY-WP-B5 changelog text -- RCWA (audit 2026-09-11, the adversarial re-verification of WP-B5)

Release 5.47.0, alongside WP-B5.  One documentation defect fixed in
`lumenairy/elements/rcwa/_core.py` and two regression gates added.  **No
executable line changed and no default moves**: `git diff` on the library is a
docstring only.

### Fixed -- RCWA: `_redheffer_star_rt` documented a boundary that the public API crosses

The two-interface closed form (WP-B5 D2) is a RE-ASSOCIATION of the Redheffer
star, and its docstring stated that the one regime where a re-association is
not neutral -- a near-singular `I - B11 A22` -- is reachable only through an
exponentially GROWING layer propagator, which `_sqrt_decay`'s `Re(lam) >= 0`
branch forbids.  The branch cut does forbid that route.  It is not the only
route: a HIGH-Q CAVITY RESONANCE produces the same near-singular denominator
with `|X| <= 1` everywhere.

Measured, from `rcwa_efficiency_1d` alone, with no monkeypatching: a weakly
modulated high-index slab in air (period 0.45 um at 633 nm, `n = 2.0 + dn` over
`2.0`, duty 0.5, air on both sides) has a `+-1` order that is EVANESCENT in
both half-spaces and PROPAGATING in the layer -- a leaky guided mode -- so an
eigenvalue of `B11 A22` has modulus `1 - O(dn^2)` and a phase the thickness
tunes.  At the resonance `cond(I - B11 A22)` reaches **1.75e+13** with
`max|A22| = 1.000000`; the closed form and the assembled star land **4.2e-04**
apart on the star's own output and **EQUALLY far (9.7e-04 each)** from the
defining coupled system solved whole, and end to end the shipped per-order
efficiency moves up to **6.4e-06** between the two formulations on a solve the
library returns.  It is already **1.8e-14** -- outside the stated envelope --
on a rung whose lossless closure is 7e-12 and which therefore warns about
nothing.  Off resonance the same six rungs agree to **6.6e-17 .. 2.9e-16**.

Neither association is the better one there -- both sit two and a half decades
inside `50 * cond * eps` -- so this is the conditioning of the cavity
denominator, which the assembled star pays identically, and nothing about the
shipped answer changes.  What changes is the documented claim: the
`<= 1.665e-15 absolute / 3.114e-15 relative` movement envelope is a statement
about the well-conditioned population it was measured on, not a bound on the
entry points.  The docstring now scopes the envelope and names both routes into
the regime, with the measured numbers for the open one.

Files: `lumenairy/elements/rcwa/_core.py` (`_redheffer_star_rt` docstring).
Tests: `tests/unit/test_audit2609_b5_rcwa_eme_bor.py::test_d2_a_near_singular_star_denominator_is_reachable_and_neither_form_is_better`.

### Added -- RCWA: the off-plane `fff_nv` fold ordering is now gated

`_li_convolutions_2d_tensor_full` takes the mean of the two Li-2003
factorization orders on the RAW `ehat` blocks so that the caller's `l3-` `E_z`
fold (Li 2003 Eq. 27) runs AFTER it -- the mean of two Schur complements is not
the Schur complement of the mean, and the caller also feeds the raw
cross-blocks and `ehat^{33}` to the generalized generator's own `inv(EZZ)`, so
folding first would hand it two quantities from different operators.  That
ordering was stated in the docstring, the call-site comment and the release
notes and tested by nothing: a mutation that swaps it leaves every other gate
green, because the x <-> y mirror is a symmetry of BOTH orderings and the Schur
complement is the identity on the in-plane reduction fixture.

The new gate pins the returned blocks at tolerance-at-0.0 against the mean
recomputed from `_li_tensor_full_l2l1` and its transposed run, and carries a
measured significance bar so the fixture can never silently become one that
cannot tell the two orderings apart (`Schur(mean)` vs `mean(Schur)` differ by
2.37e-05 absolute / 1.69e-05 relative on it at `n_orders` 3).

Files: none.
Tests: `tests/unit/test_audit2609_b5_rcwa_eme_bor.py::test_d3_the_symmetrisation_is_the_raw_mean_so_the_l3_fold_runs_after_it`.

### Migration notes

None.  No behaviour, signature, default or returned value changes in this
entry; both items are a docstring and two added tests.
