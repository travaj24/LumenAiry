This release is the fourth wave of the 2026-09-11 adversarial audit's remediation
(`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11.md`): the performance and feature
designs that 5.46.0 deferred with a written plan, implemented as thirteen work packages
on the same engineer-plus-independent-verifier pattern, with the two audit findings still
partially fixed in 5.46.0 (S6, the Maslov saddle for a non-collimated input; L9, the analytic
lens's displaced remap) closed outright.  Reports and verification reports are under
`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/` (`WP-B*_REPORT.md`,
`VERIFY_WP-B*.md`); the per-finding table `RESOLUTION_STATUS.md` is regenerated, and the
rebuild also drops five work-package attributions on findings A1, A2, A5 and A6 that the
5.46.0 table had picked up from cross-references to sibling packages ("WP-A1's ray set",
"WP-A2 follow-up") rather than to the findings themselves.

Three of this wave's designs did not survive measurement in the shape the deferral described
them, and the entries say so with the numbers: the pixel-integrated Rayleigh-Sommerfeld kernel
is four to five decades worse on a sampled smooth field and ships opt-in; the Sobol sampler buys
an exact path count but no convergence-rate gain on a hard-edged integrand; the Gegenbauer basis
for the PMM wall corner is provably a no-op on the fixed polynomial space and does not ship at
all.  Six defaults move, each a correction with the measurement beside it in its entry and a
Migration note: the Maslov asymptotic saddle for a non-collimated input (S6), the analytic
lens's displaced remap in two dimensions and in one (L9 and its one-dimensional twin, whose
converging-element rim was a crescent of exact zeros), the out-of-plane `fff_nv` operator
(H3), the system chain's Fresnel leg, which now evaluates the Fresnel integral onto the
chain grid instead of interpolating back from the natural grid, the universal dispatcher's
caustic route for a single-valued field inside the aberration envelope (the phase screen,
because FGA returns a 0.13-fidelity field there against the screen's 0.999 -- a mitigation:
its verifier traced the deficit to a reference-plane defect in the FGA transfer, the first
package of the next wave), and FGA's analytic-Jacobian predicate, which now covers the even
asphere and falls back instead of raising on a decentred conic.  Every other change is
opt-in behind a new keyword whose default reproduces 5.46.0 byte for byte, proved against
archived trees rather than the shared working copy.  The independent verifiers found and
fixed defects inside the new code in ten of the twelve packages they attacked -- among them
a fallback that scored half of the S6 saddle term, a Collins chain leg that never checked its
chirp-Z output period, an unpinned cache contract, and a movement envelope that was a
population statement rather than a bound -- and none of those defects reached a default path.
One verifier followed a defect out of the new code into the pre-existing FGA transfer: the
differential ray state is left on the last surface while the image leg is added from the
exit-vertex plane, a spurious phase of k times the last surface's sag (7.6 waves at the rim
of an R = 1.6 mm biconvex); with it projected away FGA scores 0.9998 at the same caustic.
That repair moves every FGA field on a curved-last-surface prescription and is the next
wave's first package, with the verifier's measured edit as its brief.

2 046 test ids were added in 36 new files (7 removed or renamed).  No default moved except where an entry carries a
Migration note (collected in `Migration-Guide.md` under "5.47.0").  The release block was
assembled from the package changelog files and checked with the repository's own walkers (V12,
V17, V18, `scripts/check_doc_identifiers.py`); the full two-lane unit run and the validation
suite on the released tree are recorded in the closing entry.  Two pre-existing reds are unchanged
(`tests/unit/test_pmm_m2_window_contract.py`'s T3-1 window classification, and
`tests/unit/test_v4_16_0_agent_d_validity_ranges.py`'s one-shot pin when another test in its file
has warmed the pair); the gate-day box-state interactions -- process-spawning tests hanging under
pytest's default fd capture, and four halo-check pins failing even at the 5.46.0 base -- are recorded
in `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/HANDOFF_2026_09_14.md` together with the
library-side gap they exposed (a broken worker pool deadlocks instead of falling back).

Decisions owed to the maintainer, each warning or pinned where it arises: Thorlabs data for the
AC254-050/100/200-C rows; the BICONICX reference file; CaF2 Malitson-vs-Daimon (2.8e-5); the HDF5
`lzf` hang; two UI items that need a real PySide6; whether to push and tag this branch;
`DeformableMirror`'s inclusive `'auto'` cache ceiling (the audit's 16x16-on-512 case sits exactly on
536 870 912 bytes and caches half a GiB silently -- flipping `<=` to `<` changes the summation order
of `phase()`); whether `apply_aperture(edge='gray')`, `transport='collins'` and
`sphere_normal='analytic'` become defaults (each measured better, each moves fixtures); whether
`_sphere_normal`'s domain clamp goes (its stated rationale was false; dropping it trades a grazing-ray
`RAY_NAN` for a ~1e-12 relative error); whether `gap_kernel='auto'` should fall back to `'fresnel'`
near a focus (the exact-kernel refinement carries an unbounded k |z_eff| theta^4 / 8 term there); the
odd-N grid-centring convention (27 coordinate-coupled sites sit exactly -0.5 px on odd grids); whether
the caustic route's `aberrated` condition stays (on a 2.3 rad fixture the gate keeps FGA at fidelity
0.12 where the screen scores 0.999; the H2 f/5 dual-oracle fixture decides it); whose diagnostic the
in-glass gap-leg warnings are (they name `_lens_real.py`, not the caller); and the last literal
warning depths in `propagators/carrier.py`.
