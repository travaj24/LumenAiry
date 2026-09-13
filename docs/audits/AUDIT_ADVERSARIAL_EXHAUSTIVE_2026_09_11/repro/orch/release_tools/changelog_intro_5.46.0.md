This release implements the 2026-09-11 adversarial audit of the library
(`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11.md`: 147 consolidated
findings from 22 partition reports, with the `apply_real_lens` family as the
focus).  The work was organised as 23 work packages, one per subsystem, each
implemented by an engineer and then re-verified by an INDEPENDENT adversarial
verifier who re-ran the audit's own reproduction scripts
(`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/`), built oracles
the library did not produce, tried to break every fix on fixtures the engineer
had not used, and fixed the defects that turned up in the package's own files.
The per-package reports and verification reports are under
`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/` (`WP-*_REPORT.md`,
`VERIFY_WP-*.md`), and the per-finding resolution table is
`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/RESOLUTION_STATUS.md`.

Two cross-cutting changes ride with the fixes.  Source comments now describe
what the code does and why; the version-history narrative that used to sit in
the source (17 to 38 % of several core modules) has moved verbatim to
`docs/history/<module>.md`, and each relocation is proved behaviour-free by two
fingerprints (the docstring-free AST and the comment-free token stream) that
`tests/unit/test_audit2609_a17_history_relocation.py` checks; a deliberate code
change to such a module re-records them with
`scripts/record_history_fingerprints.py` in the same commit.  And `import
lumenairy` loads SciPy's FFT and linear-algebra stacks on first use rather than
at import.

Every entry below names its finding IDs, the files, the tests added and the
measured before/after numbers; defaults that changed carry a migration note
and are collected in `Migration-Guide.md` under "5.46.0 -- adversarial audit
remediation".  2 726 test ids were added and 747 removed or renamed (73 new test
files; wall-clock speedup assertions became operation counts).  The release block was
assembled from the package changelog files and checked with the repository's
own walkers: V12 (every cited path exists), V17 (count claims), V18 (every
`file.py:N` citation lands on a non-trivial line, re-anchored against the commit
that wrote it), plus `scripts/check_doc_identifiers.py`;
`scripts/verify_changelog_closures.py` finds no audit-closure bullets in this
block, whose per-finding closures are the resolution table above.

Decisions left to the maintainer (each documented where it arises):

* the three AC254-050/100/200-C catalogue rows need Thorlabs vendor data and
  warn on every call until they get it (`lumenairy/io/prescriptions_builders.py`);
* the BICONICX surface mapping needs a reference file;
* CaF2: the bundled Malitson row and the `GLASS_REGISTRY` dispatch to Daimon
  differ by 2.8e-5 in n_d -- documented and pinned, one has to be chosen;
* `save_field_h5(compression='lzf')` did not complete in 400 s on a 1024^2
  complex field -- not a default, flagged;
* two designer-UI items (a form write-back race and lazy dock construction)
  need a real PySide6 installation to verify.

Findings deferred with a written design (not silently dropped; each design is
in its package report's deferred section and is queued as follow-on work):

* Maslov: the input-wavevector saddle (S6 proper; a warning ships now);
* analytic lens: the Newton-inverted 2-D remap and the launch-lattice raise
  (L9), the unified band generator, the displaced-carrier tangent functions,
  `seidel_correction` on an under-filled pupil (a scope limit, stated), a
  `LensPhysics` configuration object and the lens-module cycle extraction;
* propagators: the HFPI prescription-walk output plane, a Sobol sampler, a
  band-limited chirp-Z resampler, the Shen-Wang pixel-integrated RS kernel;
* carrier chain: Collins/ABCD transport with a Bluestein output grid, a
  decentred re-reference, the module split;
* gratings: Levinson Toeplitz and the two-interface closed form (RCWA),
  off-plane fff_nv symmetrisation, the PMM 2-D tensor operator cache, a PMM 1-D
  Gegenbauer option, the aberration-free reference re-expanded about the
  saddle (LG merit);
* asymptotic family: Y4 performance, the FGA convergence knob, the Pearcey
  uniform asymptotics; analysis and ray-tracing performance items; Gori
  pseudo-modes; the designer's wave-optics run decomposition, aperture-stop
  control and worker re-basing.
