"""One-shot editor: restate ``test_the_refusal_reproduces_the_prior_answer_
with_the_switch`` (the second of the two M1 tests whose premise was the pre-fix
ill-conditioning) as a decision about the current state, and add the dated
M1-equilibration paragraph."""
import io

NEW = r'''def test_the_withdrawn_refusal_moves_no_bit_and_the_ladder_carries_no_silent_defect(
        guard_off):
    """The refusal switch, and the population it was calibrated on.

    **WHAT THIS TEST USED TO SAY.**  Its name was
    ``test_the_refusal_reproduces_the_prior_answer_with_the_switch``, and it
    asserted that with ``INTERFACE_CONDITIONING_GUARD`` off the ladder RETURNS
    a silently-wrong answer -- the fail-before for the withdrawn refusal.  Its
    long v5.33.0 / v5.33.1 docstring recorded the two ways that claim had to be
    weakened: the absolute bars were BLAS-thread-dependent (``sum(R)`` reading
    3.216567e-02 at one thread and 6.112765e-03 at two on BOTH builds), and
    then the two pinned CELLS were per-build too, so both ends of the ratio had
    to be chosen from a per-build scan.

    **WHY IT IS RESTATED (2026-09-11).**  Neither weakening was the whole
    story: the DEFECT itself was the RCWA modal branch cut.  ``THIN``'s groove
    index equals both half-spaces' (1.5), so a propagating layer mode handed
    the INCOMING root -- which the pre-round-1 exact ``Re(r) == 0`` pin could
    not prevent, because an ``eig`` output never satisfies it -- was exactly a
    half-space BACKWARD mode, and the interface mode-match ``a + b`` whose
    explicit inverse IS ``S12`` was then singular.  Round 1 pinned the root;
    the ladder now closes to <= 1.5e-15 at all 25 truncations x 2
    polarizations on both builds and the census flags nothing, so there is no
    silently-wrong cell left for this test to find.  It SKIPPED rather than
    failed, because its widening path was written for a per-build absence.

    **WHAT IT SAYS NOW.**  Two claims that survive the closure, plus an
    ENGINEERED fail-before that no longer depends on which truncation a given
    BLAS reduction order breaks:

      (a) the ladder carries NO silently-wrong truncation -- every cell that
          returns closes better than ``_THIN_DEFECT_CLOSURE``, with the guard
          OFF (the fixture), which is where a wrong answer would come back;
      (b) THE SWITCH ITSELF is a no-op: the inverse refusal was WITHDRAWN (see
          the note above ``_INV_RCOND_SCREEN``), so flipping
          ``INTERFACE_CONDITIONING_GUARD`` must not move a bit on the cell that
          historically carried the defect.  That was always the assertion this
          test's name was about, and it is unchanged;
      (c) with the pre-round-1 branch body reinstated the wrong answer comes
          BACK -- and comes back through the switch in both positions, which is
          the withdrawn refusal's fail-before, reproduced on demand.

    **THE M1 EQUILIBRATION INSTRUMENT: KEPT (decision, 2026-09-11).**
    ``_equilibrated_inverse_residual`` / ``_rcond_1_equilibrated`` were chosen
    over the raw instruments because a measured population of calls existed
    where the raw residual would have REFUSED a correct answer and the
    equilibrated one passed it.  That population was the branch-cut defect:
    re-measured over a 24-fixture sweep it goes from 9 of 106 guarded calls
    pre-fix (7 unpinned; 9 / 8 / 6 on WSL, i.e. moving with the BLAS pool --
    the defect's own signature one level up) to **0 of 110 post-fix, on both
    builds at every thread count**.  The instrument is KEPT anyway: it is
    behaviour-preserving, it costs nothing on the screened path, the ARMED
    ``T22`` refusal is a separate population this change does not touch (its
    minimum equilibrated ``rcond`` reads 2.792e-02 on both arms, eight decades
    above its own ``1e-10`` bar), and a user's own coincident geometry can
    still reach it.  Deleting a guard because its found population is empty
    would repeat, one level up, the mistake this campaign exists to prevent.
    No library code was removed in round 2.
    """
    # (a) with the guard OFF, nothing in the ladder returns a wrong answer.
    for pol in ("te", "tm"):
        defects = _thin_defects(pol)
        assert not defects, (
            f"{len(defects)} {pol.upper()} truncation(s) return while missing "
            f"closure by more than {_THIN_DEFECT_CLOSURE:.0e}: X-1 has "
            f"reopened with the guard off.\n    " + _thin_table((pol,)))

    # (b) THE SWITCH: bit-identical in both positions on the historical cell.
    def solve(M, pol, flag):
        prev = _rc.INTERFACE_CONDITIONING_GUARD
        _rc.INTERFACE_CONDITIONING_GUARD = flag
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _o, R, T = rcwa_efficiency_1d(
                    THIN["period"], THIN["n_ridge"], THIN["n_groove"],
                    THIN["n_substrate"], THIN["n_superstrate"], THIN["depth"],
                    THIN["duty_cycle"], WL, angle=0.0, polarization=pol,
                    n_orders=M, stabilize=False)
            return np.asarray(R).copy(), np.asarray(T).copy()
        finally:
            _rc.INTERFACE_CONDITIONING_GUARD = prev

    off, on = solve(21, "te", False), solve(21, "te", True)
    assert float(np.max(np.abs(off[0] - on[0]))) == 0.0
    assert float(np.max(np.abs(off[1] - on[1]))) == 0.0

    # (c) the fail-before, ENGINEERED: reinstate the pre-round-1 branch body
    #     and the wrong answer comes back -- through the switch either way.
    ref = _thin_converged_sumR("te")
    with _pre_branch_cut():
        bad = [r for r in _thin_scan_uncached("te")
               if r["raised"] is None and np.isfinite(r["sumR"])
               and r["close"] > _THIN_DEFECT_CLOSURE]
        assert bad, (
            "the pre-round-1 branch body produces no silently-wrong "
            "truncation on this build: the fail-before demonstrates nothing")
        worst = max(bad, key=lambda r: abs(r["sumR"] / ref - 1.0))
        pre_off = solve(worst["M"], worst["pol"], False)
        pre_on = solve(worst["M"], worst["pol"], True)
    score = abs(worst["sumR"] / ref - 1.0)
    print(f"\nX-1 engineered fail-before: {worst['M']} "
          f"{worst['pol'].upper()} sum(R)={worst['sumR']:.6e} closure="
          f"{worst['close']:.3e} against the converged {ref:.6e} "
          f"-- {score:.1f}x wrong")
    assert score > 1.0, (
        f"the engineered pre-fix arm's worst cell is only {score:.3f}x from "
        f"the converged {ref:.6e}: it is not reproducing the documented "
        f"152x error")
    # ... and the withdrawn refusal did not act on it in either position.
    assert float(np.max(np.abs(pre_off[0] - pre_on[0]))) == 0.0
    assert float(np.max(np.abs(pre_off[1] - pre_on[1]))) == 0.0
'''

p = "tests/unit/test_m1_conditioning_guard.py"
s = io.open(p, encoding="utf-8").read()
start = s.index("def test_the_refusal_reproduces_the_prior_answer_with_the_switch(")
end = s.index("# ---------------------------------------------------------------------------\n# N-2: the Rayleigh projection")
old = s[start:end]
s = s[:start] + NEW + "\n\n" + s[end:]
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("replaced %d chars with %d" % (len(old), len(NEW)))
