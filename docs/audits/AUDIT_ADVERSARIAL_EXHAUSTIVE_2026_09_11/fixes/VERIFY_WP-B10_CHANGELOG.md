# VERIFY-WP-B10 changelog text (re-verification of the disc-orthogonal traced fit basis)

Assembled by the orchestrator into `CHANGELOG.md` for 5.47.0, alongside
`WP-B10_CHANGELOG.md`.  Finding IDs are from
`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/VERIFY_WP-B10.md`.

### Fixed -- traced lens: `fit_basis` now documents where it reaches the returned field (audit sec. 15.9, VERIFY-B10 V-2)

`apply_real_lens_traced`'s `fit_basis` documentation says what the opt-in
disc-orthogonal basis changes -- the conditioning of the entrance-plane ray
fits -- but not WHERE that reaches the field the call returns, and the answer
on a default call is "nowhere".  With the inverse-characteristic model engaged,
which is the default for every `ray_subsample > 1`, the model supplies the OPL,
the entrance coordinates and `det J` per pixel, so the returned field does not
depend on the keyword at all: `apply_real_lens_traced(..., fit_basis='zernike')`
is `np.array_equal` to the default call at `ray_subsample` 8, 4 and 2, and
`prepare_real_lens_traced` returns an identical screen at 2.  The basis reaches
the field only at `ray_subsample=1` or with `inverse_map=False`, where it moves
it by 2.2e-12 -- a change of basis, not a change of answer.

That is fix D5 / `FIX_G8_PROBE`'s finding for the fit's ORDER, restated for its
BASIS, and it is now stated on the parameter itself
(`lumenairy/elements/_lens_traced.py:8532`).  Documentation only: no behaviour
moves, and `scripts/record_history_fingerprints.py --check` is OK without a
re-record, because both fingerprints drop docstrings.

### Added -- tests: three properties of the opt-in basis that shipped unpinned

`tests/unit/test_audit2609_b10_zernike_fit_basis.py` goes 30 -> 33 tests.
Nothing existing was weakened.

* `test_the_entry_point_normalises_the_basis_to_the_beams_own_disc`
  (**V-1**) -- WHICH disc the element normalises the design to, read off the
  evaluators the element actually built.  `_fit_basis_disc_or_raise` refuses to
  invent a disc because "the whole content of the basis is WHICH disc it is
  orthogonal on", and on the off-centre branch that disc is the BEAM's; nothing
  pinned it.  Fail-before: forcing the disc concentric left **32 of 33** tests
  in the file green, because the two discs span the same space -- the fitted map
  moves by 6.0e-13 of peak while the equilibrated Gram rcond moves by **4.7
  decades** (5.930e-02 -> 1.082e-06).  A wrong disc was invisible to every
  accuracy, coupling and arbiter pin and cost the entire benefit the keyword
  exists for.
* `test_the_basis_reaches_the_returned_field_only_where_the_fits_do`
  (**V-2**) -- the contract the documentation above states, asserted on both
  arms, with the inverse-characteristic model's engagement read from the
  element's own `_imap_out` record rather than assumed.
* `test_the_conditioning_advantage_does_not_follow_the_stated_ratio_law`
  (**V-3**) -- the disc basis's conditioning advantage decays with the fit
  order, and WP-B10_REPORT.md section 6 derives that decay from
  `(R_data/R_disc)^2` per degree.  Measured on this file's own fixture at six
  disc radii, the decay SATURATES at ~1.19 decades per degree from
  `R_data/R_disc ~ 3.4` upward and then slowly declines, where the law keeps
  growing (1.06 at 3.38, 2.13 at 11.58).  What the ratio moves is the ORDER-6
  OFFSET -- 5.636e-01 down to 6.981e-04, 2.9 decades -- and therefore the
  crossover order, not the slope.  The test pins both halves, so a future
  author meets the measurement before the law.

The re-verification itself found **no defect of behaviour**: the default basis
is byte-identical to the pre-change library on 23 element configurations driven
in child processes against two `git archive` extractions (16 distinct fields, 0
mismatches), a tree in which every fit coefficient is moved by exactly 1 ULP is
detected in 15 of those 23, the decentred exit slope is the same number on both
bases in every cell of an independent orders-6..20 x three-decentre ladder on a
different singlet, and D7's headline reproduces character for character at
**2.371 / 2.371** and **1.683 / 1.683** urad.
