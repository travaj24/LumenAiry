# VERIFY-WP-B7b changelog text (re-verification of the caustic route, the fold `zeta` envelope and the FGA analytic-Jacobian predicate)

Assembled by the orchestrator into `CHANGELOG.md` for 5.47.0, alongside
`WP-B7b_CHANGELOG.md`.  Finding IDs are from
`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/VERIFY_WP-B7b.md`.

**No behaviour moves.**  Every field, route and diagnostic is byte-identical to
`9cf94fa5` (proved archive-to-worktree: seven `apply_real_lens_universal` /
`apply_real_lens_fga` digests and thirty-one routing decisions, all equal).
Both changed modules have no `docs/history/` document
(`scripts/record_history_fingerprints.py --check` reports none for either), so
nothing is re-recorded.

### Fixed -- the caustic route's comment no longer attributes FGA's deficit to the caustic, and no longer offers `caustic_pad_dof` as a way back (VERIFY-B7b V-1, V-4)

`_universal_route`'s caustic branch documented its own measurement correctly and
its CAUSE incorrectly: "the swarm is not under-sampled; it converges to the
wrong field".  Re-measured against an independent brute-force
Rayleigh-Sommerfeld oracle, `'fga'`'s deficit on that fixture is the same at
every output plane -- **0.0737 at `output_plane_distance = 0`**, where there is
no caustic at all, against 0.1250 at the focus -- and it vanishes when the LAST
surface is FLAT (0.9998 on an N-LASF9 plano-convex of NA 0.150 at its own
caustic, where `'phase_screen'` reads 0.9639).  At fixed focal length, glass,
wavelength, aperture, grid and beam, `'fga'` falls 0.9998 / 0.9656 / 0.8053 /
0.5100 / 0.2326 / 0.1031 as the last surface's curvature grows 0 -> 0.571 /mm.
The cause is a reference plane, not the frozen-Gaussian model: `_fga_core`
traces with `ray_transfer_jacobian`, whose base-ray state sits on the last
SURFACE, and then adds the image leg as if it sat on the exit-vertex PLANE, so
every beamlet carries a spurious phase the size of that surface's sag (7.6 waves
at the rim of the WP-B7b fixture, 15.0 waves on a strongly bent singlet).  The
comment now says that, and `apply_real_lens_universal`'s member map carries the
measured exception where the member ordering reverses.

The same comment offered `caustic_pad_dof=0.0` as an alternative way back to
`'fga'`.  It is not one: narrowing the zone leaves this branch answering the
same way inside it (measured `'phase_screen'` at the near, mid and far edges of
the unpadded zone) and sends planes outside it to `'traced'` / `'phase_screen'`.
`method='fga'` is the only route to the swarm.

* `lumenairy/propagators/fga.py` (`_universal_route`'s caustic-branch comment;
  the `'fga'` bullet of `apply_real_lens_universal`'s member map).

### Fixed -- `_ZETA_EXTRAPOLATION_MAX`'s derivation records a second, contradicting ladder (VERIFY-B7b V-2)

The constant's comment presented one three-singlet ladder (-5.3 % .. +4.9 %
below the bar, +12.5 % .. +22.8 % above it) as the envelope.  A second ladder
through ONE caustic of a single optic (N-BAF10 biconvex R = +/-2.6 mm, 0.90 mm
aperture, lambda = 1.064 um, N = 512, dx = 2.20 um, eight planes) drifts the
same way with the ratio but SATURATES near +5 %: 0.989 / 1.019 / 0.975 / 0.986 /
1.011 / 1.049 / 1.045 / 1.041 at `W/band` = 0.42 / 0.74 / 1.03 / 3.01 / 5.65 /
14.0 / 27.3 / 531.5, so on that optic the warning above the bar is a false
positive for absolute energy and no rung reaches +/-10 %.  The constant is
unchanged and the warning still fires in the right direction on both ladders;
the comment now says it is a conservative flag rather than a calibrated
5 % / 10 % boundary, and carries the second ladder.

* `lumenairy/elements/_lens_traced_uniform.py` (`_ZETA_EXTRAPOLATION_MAX`'s
  derivation comment).
