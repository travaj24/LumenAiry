"""CI KERNEL SWEEP (2026-09-11) -- the library's guard DECISIONS, per arm.

WHY THIS PROBE EXISTS.  The 5.45.0 release matrix went RED on CI (AMD EPYC
runners) while the same commit was green on both local builds.  Every failing
assertion in that matrix was a *decision* -- a guard that refuses on one build
and returns on another, or a test whose bar is pinned on a build-dependent
reading.  ``docs/TESTING_STANDARDS.md`` ("flakiness is bad math") forbids
re-running such a thing to green: the decision itself has to be made
build-independent, and that requires MEASURING it on more than one BLAS kernel.

WHAT AN "ARM" IS.  One (build, kernel, thread-width) triple.  ``build`` is the
interpreter + numpy + OS (``WIN`` = Windows/CPython 3.14, ``WSL`` =
Ubuntu/CPython 3.12); ``kernel`` is the OpenBLAS micro-kernel family, selected
on the command line with ``OPENBLAS_CORETYPE`` (the bundled scipy-openblas is
DYNAMIC_ARCH, so the variable re-dispatches the whole BLAS/LAPACK kernel set
without rebuilding anything); ``thread-width`` is what the thread caps were
set to, ``tauto`` meaning no cap at all.

BOTH axes are real and the second is not optional.  CI's FAST lane
deliberately leaves BLAS unpinned (``unit-tests.yml``: "this job deliberately
does NOT pin BLAS at run time") while the SLOW and JAX lanes pin to one, so a
census taken only at one thread cannot speak about the lane where most of the
red appeared.  A reduction split across four threads is a different summation
order from the same reduction on one, in exactly the way a different kernel
is.

    # a kernel arm
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        OPENBLAS_CORETYPE=NEHALEM python probe_decisions.py --out arm.json
    # the CI-faithful arm: default kernel, caps UNSET
    python probe_decisions.py --out arm.json

MEASURED CAVEAT (2026-09-11, this host = AMD Ryzen 9 5950X, Zen 3):

  * ``OPENBLAS_CORETYPE=ZEN`` is NOT a distinct kernel in these wheels.  It
    is not in the DYNAMIC_ARCH table, so it falls back to auto-detection --
    and auto-detection on a Zen CPU reports ``Haswell``.  An unrecognised
    name (``BOGUSCORE``) resolves to exactly the same thing.  The practical
    consequence is GOOD news: the local default arm ALREADY IS the CI Zen
    arm at the BLAS-kernel level, so a decision that differs between here
    and CI does not differ because of a Zen kernel.
  * ``OPENBLAS_CORETYPE=SKYLAKEX`` sets the corename but the kernels are
    AVX-512, which this Zen 3 host cannot execute: the first BLAS call dies
    with SIGILL (exit 132) on BOTH builds.  It is therefore unreachable here
    and is recorded as such rather than silently skipped.

The usable ladder is consequently HASWELL (= the default = Zen) / SANDYBRIDGE
(AVX, no FMA) / NEHALEM (SSE4.2) / PRESCOTT (SSE2, reported as ``Katmai``) --
four different reduction orders and blockings, which is what the decisions
have to survive.

MEASURED AGAIN 2026-09-12 (WP-A23) ON A DIFFERENT HOST -- Intel Xeon w3-2535
(Sapphire Rapids, 10 cores / 20 threads), Windows 11, CPython 3.14.6, numpy
2.4.6 / scipy 1.17.1.  The caveats above are properties of the 2026-09-11
host, not of the wheels, and on this one the ladder is LONGER by exactly the
kernel that was unreachable there.  Corename read back per request:

    unset -> SkylakeX      SKYLAKEX -> SkylakeX     HASWELL -> Haswell
    ZEN   -> Haswell       BOGUSCORE -> SkylakeX    NEHALEM -> Nehalem
    SANDYBRIDGE -> Sandybridge                      KATMAI / PRESCOTT -> Katmai

so ``SKYLAKEX`` is the AUTO-DETECTED default here (AVX-512 silicon) rather
than a SIGILL, ``ZEN`` still resolves to ``Haswell`` and an unrecognised name
still resolves to auto-detection -- both caveats above survive the change of
host, and the second one inverts.  Five distinct kernels are therefore
reachable here: SkylakeX / Haswell / Sandybridge / Nehalem / Katmai.  numpy's
ILP64 build and scipy's LP64 build dispatch the SAME kernel on every rung
(both read back per rung), so the arm has one kernel name and not two.

WHAT IS RECORDED.  Two blocks per arm:

  ``decisions``     the guard OUTCOMES the library actually takes, as short
                    strings ("refuse"/"return", "warn"/"silent",
                    "accept"/"refuse", a count).  These are what
                    ``tests/unit/test_ci_kernel_consistency.py`` asserts
                    AGREE across every committed arm.  A decision must be a
                    DISCRETE outcome, never a float -- comparing floats
                    across kernels is the mistake this probe exists to catch.
  ``hypothetical``  the verdict a bar the library does NOT ship would return
                    at a site the library deliberately leaves UNGUARDED.
                    These are recorded to be shown NON-unanimous: they are
                    the standing evidence that the site is undecidable, and
                    the consistency gate asserts they stay that way rather
                    than asserting they agree.  Promoting one of these to a
                    real guard is exactly the mistake the audit forbids.
  ``classes``       the ANSWER CLASS each guard row is deciding about, as
                    measured WITH THE GUARD DISARMED: ``correct`` / ``grey`` /
                    ``wrong`` by the campaign's own closure rule.  Added
                    2026-09-11 -- see THE CONTRACT below.  A row with NO class
                    is one no answer can follow (the band, branch-cut, mortar,
                    T22 and ``notices@`` rows), and those are compared across
                    arms for plain equality.
  ``readings``      the underlying floats, for the audit document only.  They
                    are NOT asserted anywhere; they move with the kernel by
                    design and that is the whole point.

THE CONTRACT, CORRECTED 2026-09-11 (CI PREMISE GATES).  The first version of
this census asserted that every arm takes the same OUTCOME on every row.  That
is the wrong invariant, and the 5.45.0 CI matrix proved it: the runner arm
(ubuntu, AMD EPYC 7763, unpinned BLAS, pip wheels of numpy 2.4.6 / scipy
1.17.1) solves the ill-conditioned 1-D interface fixture CORRECTLY at a 1e-05
wall separation where every local arm -- four OpenBLAS kernels x one and four
threads x two builds -- solves it wrong.  The guards then, CORRECTLY, took
different outcomes there: ``closes`` where the census said ``open``, ``silent``
where it said ``warn``, ``return`` where it said ``refuse``.  A guard whose
job is to refuse wrong answers and return correct ones MUST follow the answer;
demanding outcome equality across arms demands that the guard ignore it.

So the census now records, beside every decision, the ANSWER CLASS that
decision is about, and the contract is RULE conformance:

  * per arm -- the decision is what :data:`_RULES` says that class permits;
  * across arms -- rows with the SAME class must take the SAME decision.  A
    row whose CLASS differs between arms is REPORTED, not failed: that is the
    guard following the answer.  A row whose decision differs while the class
    does NOT, or which has no class at all (the kernel-independent sections),
    is still a P1.

Every fixture below is cheap (the whole probe is a few seconds) so the
consistency gate can re-run it on the current arm inside its 30 s budget.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import warnings

# NOTE: no ``setdefault`` here, deliberately.  The THREAD COUNT is one of this
# probe's two axes (see the module docstring), and a probe that quietly pins
# itself to one thread cannot measure the unpinned arm -- which is the arm CI's
# fast lane actually runs.  Pass the caps on the command line; leaving them
# unset IS the "auto" arm.
import numpy as np  # noqa: E402
import scipy.linalg as sla  # noqa: E402

from lumenairy.elements.pmm import PMM2DStackPure, PMMStack  # noqa: E402
from lumenairy.elements.pmm import _core as _pc  # noqa: E402
from lumenairy.elements.pmm import stack as _st1d  # noqa: E402
from lumenairy.elements.pmm.twod_staggered import _C  # noqa: E402
from lumenairy.elements.rcwa import _core as _rc  # noqa: E402

# --------------------------------------------------------------- fixtures
_P = 1.0e-6
_WL = 0.62e-6
_TH, _PH = 0.09, 0.3
_EPS_H, _EPS_P, _EPS_B, _EPS_SPACER = 2.25, 6.0, 6.0, 2.1
_E_OOP = np.array([[4.0, 0.0, 0.8], [0.0, 3.4, 0.0], [0.75, 0.0, 3.2]],
                  dtype=_C)
_WA = (0.2371, 0.6183)
_WC = (0.2371, 0.7402)


def _sc(walls):
    return [w * _P for w in walls]


def _mid(walls):
    b = (0.0,) + tuple(walls) + (1.0,)
    return [0.5 * (b[i] + b[i + 1]) for i in range(len(b) - 1)]


def _tensor_cell(walls, lo, hi, eps_in=None, eps_host=_EPS_H):
    m = _mid(walls)
    n = len(m)
    c = np.empty((n, n, 3, 3), dtype=_C)
    e = _E_OOP if eps_in is None else eps_in
    for i in range(n):
        for j in range(n):
            c[i, j] = e if (lo < m[i] < hi and lo < m[j] < hi) \
                else np.eye(3) * eps_host
    return c


def _scalar_cell(walls, lo, hi, eps_in=_EPS_B, eps_host=_EPS_H):
    m = _mid(walls)
    n = len(m)
    c = np.full((n, n), _C(eps_host))
    for i in range(n):
        for j in range(n):
            if lo < m[i] < hi and lo < m[j] < hi:
                c[i, j] = _C(eps_in)
    return c


def _decade(x):
    """The DECADE INDEX of a positive reading -- ``floor(log10(x))``.

    A decade index is the coarsest honest discretisation of a float: it is
    the statement "this reading is between 1e-11 and 1e-10", which is a
    DECISION, where the reading itself is not.  Non-finite and non-positive
    readings get their own labels rather than a number, because a bar test
    treats them specially too.
    """
    x = float(x)
    if not math.isfinite(x):
        return "nonfinite"
    if x <= 0.0:
        return "zero_or_negative"
    return "1e%+03d" % math.floor(math.log10(x))


#: The closure bars that turn a solve into an ANSWER CLASS.  They are placed
#: by the GAP between the two populations this fixture produces, not by one
#: build's residual: the CORRECT rows read |R+T - 1| = 1.02e-08 (WIN Haswell)
#: up to 3.66e-07 (the CI runner), and the WRONG rows read 1.17 (Katmai) up to
#: 3.6.  1e-05 therefore leaves 27x of slack above the worst correct reading
#: and 1e-02 leaves 117x below the best wrong one, with five decades of empty
#: band between them that nothing in this fixture has ever landed in.
#: ``grey`` exists so that a future arm landing in the band is reported rather
#: than forced into a class it does not belong to.
_CLASS_CORRECT = 1.0e-5
_CLASS_WRONG = 1.0e-2


def _answer_class(tot):
    """``correct`` / ``grey`` / ``wrong`` from a lossless closure reading.

    Measured with the guard DISARMED, so it is a property of the ARITHMETIC
    and not of the decision the guard then takes on it.
    """
    d = abs(float(tot) - 1.0)
    if not math.isfinite(d):
        return "wrong"
    if d < _CLASS_CORRECT:
        return "correct"
    if d > _CLASS_WRONG:
        return "wrong"
    return "grey"


#: THE RULE each answer-following guard row must obey, keyed by decision-key
#: PREFIX and read as ``class -> the decisions that class permits``.
#:
#: These are the pairings the sliver and energy gates assert everywhere else
#: in the suite ("a WRONG row is never returned in silence; a CORRECT row is
#: never refused"), written once, in the one place that compares arms.  A
#: ``grey`` row permits everything the two sides do, because the grey band is
#: by construction the band where the campaign declines to classify.
_RULES = {
    # what the UNGUARDED site answers.  This row IS the class, restated as a
    # decision so a reader of the table can see the two side by side.
    "pmm1d_interface/answer@": {
        "correct": ("closes",),
        "grey": ("open",),
        "wrong": ("open",),
    },
    # whether the stack's ENERGY TRIPWIRE said anything: silent on a correct
    # answer, loud on a wrong one.  The 1-D site itself ships UNGUARDED, so
    # the tripwire is the only voice that is ABOUT THE ANSWER here -- which is
    # why this row can carry an answer-following rule at all.
    #
    # NARROWED 2026-09-12 (WP-A23), and the narrowing is what keeps the rule
    # honest rather than what relaxes it.  Until this round the row was
    # decided as "did ANY warning come out", which was the same thing while
    # the tripwire was the only voice.  It stopped being the same thing when
    # WP-A12's G2 raised the default ``min_feature`` two decades: the union
    # grid now SNAPS this fixture's colliding walls and says so
    # (``_pmm_union_grid: snapped 2 pair(s) ...``, ``pmm/_core.py``), on a
    # CORRECT answer, at both wall separations.  Counting that as "warned"
    # made the census report a guard crying wolf -- ``warn`` at class
    # ``correct``, which this table forbids -- when what had actually
    # happened is that a second, deliberate, answer-INDEPENDENT voice had
    # appeared.  The geometry notice is censused in its own right by
    # ``pmm1d_interface/notices@`` instead, so nothing is lost and the
    # no-false-alarm claim stays exactly as strict as it was.
    "pmm1d_interface/warned@": {
        "correct": ("silent",),
        "grey": ("silent", "warn"),
        "wrong": ("warn",),
    },
    # the site RETURNS whatever it computed, on every class.  Unguarded is a
    # property of the code, not of the answer.
    "pmm1d_interface/returns@": {
        "correct": ("return",),
        "grey": ("return",),
        "wrong": ("return",),
    },
    # the 1-D SLIVER guard: it must never refuse a correct answer, and must
    # never return a wrong one in silence.  Which of WARN and REFUSE it picks
    # on a wrong row is the arbiter's business and not this census's.
    "sliver/pmm1d@": {
        "correct": ("return", "warn"),
        "grey": ("return", "warn", "refuse"),
        "wrong": ("warn", "refuse"),
    },
}


#: The two voices the plain 1-D interface fixture can raise, matched on the
#: message text AT THE SITE THAT RAISES IT so the match cannot drift to a
#: lookalike: the stack's energy tripwire
#: (``pmm/stack.py::PMMStack.solve``, "energy not conserved (max R+T = ...)")
#: and the union grid's near-coincident-wall snap notice
#: (``pmm/_core.py::_pmm_union_grid``).  Anything else is counted as
#: ``other`` rather than dropped -- a THIRD voice appearing at this site is a
#: censusable event, and a matcher that silently ignored it would retire the
#: row's meaning without anyone editing the row.
_VOICE_ENERGY = "energy not conserved"
_VOICE_SNAP = "_pmm_union_grid: snapped"


def _is_env_advisory(rec):
    """Is this warning about the ENVIRONMENT rather than about the solve?

    The library advises, at run time, when an optional or declared dependency
    it wanted is missing -- e.g. ``lumenairy/memory.py``'s "psutil not
    installed - assuming 4 GB available memory.  Install psutil for accurate
    memory-aware batching." (``RuntimeWarning``).  That is a statement about
    the machine, not about this fixture's geometry or its answer, and it MUST
    NOT enter a decision: a census row that moved because one interpreter is
    missing a package would be red on exactly the hosts the census exists to
    speak about, which is the failure mode this whole file is a reaction to.

    MEASURED 2026-09-12 (WP-A23): the WSL build reachable from this
    workstation is a bare venv without ``psutil``, and the interface fixture
    raises the advisory THREE times per solve there against zero on Windows.
    Without this split the two builds would disagree on ``notices@`` -- a row
    with no answer class, i.e. a reported P1 -- for a reason that has nothing
    to do with the library's guards.

    Matched on the library's own advisory idiom (a ``RuntimeWarning`` that
    says a package is "not installed" and asks the user to install it), not
    on one message, so a second such advisory is covered.  The count is kept
    as a READING, so an arm that is missing dependencies still says so.
    """
    msg = str(rec.message)
    return (issubclass(rec.category, RuntimeWarning)
            and "not installed" in msg and "Install " in msg)


def _voices(caught):
    """``(energy, snap, env, other)`` counts over one solve's warnings."""
    energy = snap = env = other = 0
    for rec in caught:
        msg = str(rec.message)
        if _VOICE_ENERGY in msg:
            energy += 1
        elif _VOICE_SNAP in msg:
            snap += 1
        elif _is_env_advisory(rec):
            env += 1
        else:
            other += 1
    return energy, snap, env, other


def _notice_label(snap, other):
    """The ``notices@`` DECISION: which answer-independent voices spoke.

    A label set, not a count -- the number of snapped pairs is a reading and
    moves with the fixture, while WHICH voices spoke is a decision and must
    be the same on every arm, this being pure geometry.
    """
    parts = [name for name, n in (("snap", snap), ("other", other)) if n]
    return "+".join(parts) if parts else "none"


def rule_for(key):
    """The ``class -> permitted decisions`` table for a decision key, or
    ``None`` when the key is not an answer-following guard row (the band,
    branch-cut, mortar and T22 sections, which decide on geometry or on a
    fixed spectrum and are compared across arms for plain equality)."""
    for prefix, table in _RULES.items():
        if key.startswith(prefix):
            return table
    return None


# ======================================================================
# A -- the plain 1-D ``_interface_smatrix`` site (the mortar round-2
#      rationale test's subject)
# ======================================================================
#: the 1-D section runs in the ROUND-2 GATE FILE'S OWN UNITS (period 1.2,
#: wavelength 0.85), verbatim -- the point of this section is to reproduce
#: exactly the population that file's rationale test reads, and a stack driven
#: with a different period/wavelength pair is a different device.
_1D_P, _1D_WL, _1D_TH = 1.2, 0.85, 0.15
_1D_EPS_P, _1D_EPS_H = 9.0, 2.25

#: ``min_feature`` PINNED as a fixture parameter, at ``period * 1e-5``.
#:
#: WHY IT IS PINNED (2026-09-12, WP-A23), and why pinning it makes this
#: section HARDER rather than easier.  ``min_feature`` is the knob that SNAPS
#: a near-coincident cross-layer wall pair away, and this fixture IS a
#: cross-layer wall pair ``delta`` = 1e-04 / 1e-05 of a period apart -- the
#: object of measurement for every row in sections A and B.  Audit finding G2
#: (WP-A12, ``56a76f22``) raised the library default from ``period*1e-5`` to
#: ``period*1e-3``, two decades ABOVE both separations, and an unpinned
#: fixture therefore stopped constructing the thing it censuses: MEASURED on
#: this tree at the shipped default, both wall pairs snap to coincidence, the
#: two layers become geometrically IDENTICAL, the interface's reciprocal
#: condition moves from 9.7297e-13 to 5.2104e-04 (NINE decades) and every row
#: reads the same trivial ``correct`` / ``closes`` / ``silent`` / ``return``
#: on every arm.  A section that cannot distinguish two kernels cannot detect
#: a guard that decides differently on them, which is this file's entire job.
#:
#: The pin is the same remedy WP-A12 applied to its three inherited fixtures
#: and VERIFY-A12 applied to ``test_fix_pmm2d_mortar_round2.py::
#: test_the_plain_1d_interface_solve_is_left_unguarded_and_this_is_why`` --
#: the very test this section exists to mirror, so pinning here RE-SYNCS the
#: census with it rather than diverging.  Every bar, every rule and every
#: decision key is unchanged; what changes is that the fixture is built again.
#:
#: What the SHIPPED default does is not thereby lost: the third case below
#: measures it, under its own tag, so the default change is censused as a
#: decision instead of silently erasing the rows around it.
_1D_MF_PINNED = 1.0e-5

#: ``(delta, min_feature fraction or None for the shipped default, tag)``.
#: The first two tags are the committed census's own, at the committed
#: geometry, so an arm measured today lines up row for row with one measured
#: in 2026-09-11.  The third is the shipped-default twin of the harder one.
_1D_CASES = (
    (1e-4, _1D_MF_PINNED, "1e-04"),
    (1e-5, _1D_MF_PINNED, "1e-05"),
    (1e-5, None, "1e-05@mf-default"),
)


def _pmm1d_two_layer(delta, mf_frac=_1D_MF_PINNED):
    a0, a1 = 0.27865, 0.62505
    kw = {} if mf_frac is None else {"min_feature": _1D_P * mf_frac}
    st = PMMStack(_1D_P, degree=12, far_field_orders=5, **kw)
    st.add_layer(0.08, segments=[(a0, _1D_EPS_H), (a1 - a0, _1D_EPS_P),
                                 (1 - a1, _1D_EPS_H)])
    b0, b1 = a0 - delta, a1 + delta
    st.add_layer(0.08, segments=[(b0, _1D_EPS_H), (b1 - b0, _1D_EPS_P),
                                 (1 - b1, _1D_EPS_H)])
    st.set_source(_1D_WL, theta=_1D_TH)
    return st


def _interface_site_population(dec, hyp, rea, cls=None):
    """The ``_interface_smatrix`` rcond population, WITH THE SLIVER GUARD
    DISARMED.

    Disarming is not a convenience: the 1-D sliver guard's own refusal is
    itself a kernel-dependent decision (recorded separately in section B and
    handed to the sliver-guard owner), so leaving it armed would make this
    section measure the SLIVER guard rather than the MORTAR site.  Disarmed,
    the site is reached at both wall separations on every build and the
    population is the same object everywhere.

    THREE CASES (2026-09-12), see :data:`_1D_CASES`: the two committed wall
    separations at the PINNED ``min_feature`` -- the configuration in which
    this fixture is an ill-conditioned interface at all -- and the harder one
    again at the SHIPPED default, which is what a caller who passes no
    ``min_feature`` gets today.  The third is tagged ``@mf-default`` and is
    what makes the default change visible in the table rather than only in
    the report that describes it.
    """
    real = _st1d._interface_smatrix
    prev_guard = _st1d.PMM_SLIVER_GUARD
    seen = []

    def _patched(Wa, Va, Wb, Vb):
        for A in (np.asarray(Wb), np.asarray(Vb)):
            lu, piv = sla.lu_factor(A)
            gecon = sla.get_lapack_funcs("gecon", (A,))
            rc, _i = gecon(lu, float(np.max(np.sum(np.abs(A), axis=0))))
            seen.append(float(rc))
        return real(Wa, Va, Wb, Vb)

    _st1d._interface_smatrix = _patched
    _st1d.PMM_SLIVER_GUARD = False
    if cls is None:
        cls = {}
    try:
        for delta, mf_frac, tag in _1D_CASES:
            seen.clear()
            st = _pmm1d_two_layer(delta, mf_frac)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                _o, R, T = st.solve()[:3]
            tot = float(np.max(np.atleast_2d(R).sum(1)
                               + np.atleast_2d(T).sum(1)))
            rc = min(seen)
            energy, snap, env, other = _voices(w)
            rea["pmm1d_interface/rcond@%s" % tag] = rc
            rea["pmm1d_interface/R+T@%s" % tag] = tot
            rea["pmm1d_interface/n_warnings@%s" % tag] = float(len(w))
            if env:
                rea["pmm1d_interface/env_advisories@%s" % tag] = float(env)
            if other:
                rea["pmm1d_interface/other_voice@%s" % tag] = "; ".join(
                    str(x.message)[:80] for x in w
                    if _VOICE_ENERGY not in str(x.message)
                    and _VOICE_SNAP not in str(x.message)
                    and not _is_env_advisory(x))[:200]
            dec["pmm1d_interface/answer@%s" % tag] = (
                "closes" if abs(tot - 1.0) < 1e-5 else "open")
            dec["pmm1d_interface/warned@%s" % tag] = (
                "warn" if energy else "silent")
            # the answer-INDEPENDENT voices, censused separately so the row
            # above can keep meaning "the tripwire spoke".  This one carries
            # NO answer class: whether the union grid snaps is decided by the
            # wall separation against ``min_feature``, which is arithmetic no
            # BLAS kernel participates in -- so it is compared across arms for
            # plain equality, like the band and branch-cut sections.
            dec["pmm1d_interface/notices@%s" % tag] = _notice_label(snap, other)
            dec["pmm1d_interface/returns@%s" % tag] = "return"
            # the ANSWER CLASS this row's guards are deciding about, taken
            # here because here is where the guard is DISARMED.  The sliver
            # section (B) runs the same fixture with the guard ARMED and may
            # get no reading at all (it can refuse), so its class is the one
            # measured here.
            klass = _answer_class(tot)
            for k in ("pmm1d_interface/answer@%s" % tag,
                      "pmm1d_interface/warned@%s" % tag,
                      "pmm1d_interface/returns@%s" % tag,
                      "sliver/pmm1d@%s" % tag):
                cls[k] = klass
            # HYPOTHETICAL, not shipped: what a 1e-12 rcond bar -- the value
            # the 2-D IN-PLANE mortar sites use -- would decide here.  It is
            # in the ``hypothetical`` block precisely because its verdict is
            # NOT the same on every kernel.
            hyp["pmm1d_interface/rcond_decade@%s" % tag] = _decade(rc)
            hyp["pmm1d_interface/bar_1e-12_would@%s" % tag] = (
                "refuse" if rc < _pc._MORTAR_RCOND_REFUSE else "accept")
    finally:
        _st1d._interface_smatrix = real
        _st1d.PMM_SLIVER_GUARD = prev_guard


# ======================================================================
# B -- the 1-D SLIVER guard's refusal (NOT owned here: handed off)
# ======================================================================
def _sliver_decisions(dec, rea, cls=None):
    for delta, mf_frac, tag in _1D_CASES:
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                _o, R, T = _pmm1d_two_layer(delta, mf_frac).solve()[:3]
            tot = float(np.max(np.atleast_2d(R).sum(1)
                               + np.atleast_2d(T).sum(1)))
            rea["sliver/R+T@%s" % tag] = tot
            dec["sliver/pmm1d@%s" % tag] = (
                "warn" if any("SLIVER" in str(x.message) for x in w)
                else "return")
        except ValueError as exc:
            dec["sliver/pmm1d@%s" % tag] = (
                "refuse" if "SLIVER" in str(exc) else "raise_other")
            rea["sliver/msg@%s" % tag] = str(exc)[:80]


# ======================================================================
# C -- the 2-D per-layer MORTAR sites (rcond screen + residual screen)
# ======================================================================
def _mixed(kind, M, n_orders=2):
    st = PMM2DStackPure(_P, n_modes=M, n_orders=n_orders, n_substrate=1.5,
                        layer_grids="per-layer")
    if kind == "spacer":
        st.add_layer(0.13e-6, eps_cell=_tensor_cell(_WA, *_WA),
                     x_walls=_sc(_WA), y_walls=_sc(_WA))
        st.add_layer(0.10e-6, eps=_EPS_SPACER)
    elif kind == "pattern":
        st.add_layer(0.13e-6, eps_cell=_tensor_cell(_WA, *_WA),
                     x_walls=_sc(_WA), y_walls=_sc(_WA))
        st.add_layer(0.10e-6, eps_cell=_scalar_cell(_WC, *_WC),
                     x_walls=_sc(_WC), y_walls=_sc(_WC))
    elif kind == "slant_both":
        st.add_layer(0.13e-6, eps_cell=_scalar_cell(_WA, *_WA),
                     x_walls=_sc(_WA), y_walls=_sc(_WA), slant=(0.08, 0.03))
        st.add_layer(0.10e-6, eps_cell=_scalar_cell(_WC, *_WC, eps_in=4.0),
                     x_walls=_sc(_WC), y_walls=_sc(_WC), slant=(0.08, 0.03))
    else:
        raise ValueError(kind)
    st.set_source(_WL, theta=_TH, phi=_PH)
    return st


def _mortar_decisions(dec, rea):
    for kind, M in (("spacer", 5), ("pattern", 4), ("slant_both", 4)):
        key = "mortar/%s@M%d" % (kind, M)
        census = []
        _pc._MORTAR_SOLVE_CENSUS = census
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _mixed(kind, M).solve(jones=False)
            outcome = "accept"
        except Exception as exc:                       # noqa: BLE001
            outcome = ("refuse" if type(exc).__name__ == "_ConditioningError"
                       else "raise_" + type(exc).__name__)
            rea[key + "/msg"] = str(exc)[:100]
        finally:
            _pc._MORTAR_SOLVE_CENSUS = None
        dec[key] = outcome
        # per-SCREEN detail: how many solves each screen refused, and the
        # worst reading each screen saw.  ``n`` and the refusal COUNT are
        # decisions; the readings are not.
        refused = sum(1 for r in census if r[3])
        dec[key + "/refused_solves"] = str(refused)
        dec[key + "/n_solves"] = str(len(census))
        rc_all = [r[2] for r in census if r[2] is not None
                  and math.isfinite(r[2])]
        res_all = [r[4] for r in census if r[4] is not None]
        if rc_all:
            rea[key + "/worst_rcond"] = min(rc_all)
            dec[key + "/worst_rcond_decade"] = _decade(min(rc_all))
        if res_all:
            rea[key + "/worst_resid"] = max(res_all)
            dec[key + "/worst_resid_decade"] = _decade(max(res_all))


# ======================================================================
# D -- the per-layer WIDTH band: warn / silent / refuse
# ======================================================================
def _band_stack(frac, M=4, centre=0.44):
    st = PMM2DStackPure(_P, n_modes=M, n_orders=1, layer_grids="per-layer")
    sw = (centre, centre + frac)
    tile = np.full((3, 3), _C(_EPS_H))
    tile[1, 1] = _C(_EPS_B)
    st.add_layer(0.10e-6, eps_cell=tile, x_walls=_sc(_WA), y_walls=_sc(_WA))
    st.add_layer(0.10e-6, eps_cell=tile, x_walls=_sc(sw), y_walls=_sc(sw))
    st.set_source(_WL, theta=_TH, phi=_PH)
    return st


def _band_decisions(dec, rea):
    for frac in (2.0e-2, 6.0e-2, 1.2e-3, 9.0e-4):
        key = "band/%.1e" % frac
        try:
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter("always")
                _band_stack(frac).solve(jones=False)
            hit = [w for w in ws if issubclass(w.category, UserWarning)
                   and "degradation band" in str(w.message)]
            dec[key] = "warn" if hit else "silent"
        except ValueError as exc:
            dec[key] = ("refuse_min_seg" if "below the minimum" in str(exc)
                        else "raise_other")
            rea[key + "/msg"] = str(exc)[:80]


# ======================================================================
# E -- the RCWA generalized interface's T22 refusal
# ======================================================================
def _t22_decisions(dec, rea):
    """The ``T22`` guard, exercised through the PURE staggered engine's
    SLANTED and OUT-OF-PLANE routes (both take
    ``_interface_smatrix_general``) plus the ordinary in-plane control.
    """
    for kind, M in (("slant_both", 4), ("spacer", 5), ("pattern", 4)):
        key = "t22/%s@M%d" % (kind, M)
        census = []
        _rc._INV_CENSUS = census
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _mixed(kind, M).solve(jones=False)
            dec[key] = "accept"
        except Exception as exc:                       # noqa: BLE001
            dec[key] = ("refuse"
                        if "T22" in str(exc) or "generalized interface"
                        in str(exc) else "raise_" + type(exc).__name__)
        finally:
            _rc._INV_CENSUS = None
        t22 = [r for r in census if "T22" in r[0]]
        dec[key + "/n_t22"] = str(len(t22))
        dec[key + "/refused_t22"] = str(sum(1 for r in t22 if r[4]))
        rcs = [r[2] for r in t22 if r[2] is not None and math.isfinite(r[2])]
        if rcs:
            rea[key + "/worst_rcond_eq"] = min(rcs)
            dec[key + "/worst_rcond_eq_decade"] = _decade(min(rcs))
            dec[key + "/bar_would_refuse"] = (
                "refuse" if min(rcs) < _rc._INV_T22_RCOND_REFUSE else "accept")


# ======================================================================
# F -- the modal BRANCH-CUT orientation census
# ======================================================================
def _branch_cut_decisions(dec, rea):
    """``_sqrt_decay``'s ON-CUT decision, as a census rather than a float.

    For each fixture spectrum: how many roots land inside the relative
    on-cut band, and how many of those the selector CONJUGATES (i.e. how
    many modes would otherwise have carried the INCOMING root).  Both are
    integers -- the decision -- while the ``|Re(r)|/scale`` ratios that
    drive them are rounding-level noise and are recorded only as readings.
    """
    rng = np.random.default_rng(20260911)
    cases = {
        # a LOSSLESS propagating spectrum: lam^2 real-negative up to the
        # eigensolver's backward error, which is exactly the population the
        # band exists for
        "lossless_propagating": (-np.array([4.0, 9.0, 16.0, 25.0, 36.0])
                                 + 1j * rng.normal(0, 3e-16, 5)),
        # a genuinely DECAYING spectrum: real positive, must NOT be touched
        "evanescent": np.array([4.0, 9.0, 30.0, 120.0, 400.0]) + 0j,
        # a LOSSY spectrum: a real decay rate on the signal side of the band
        "lossy": (-np.array([4.0, 9.0, 16.0]) * (1.0 - 0.02j)),
        # a spectrum AT a layer cutoff -- the binding population of _CUT_BAND_REL
        "at_cutoff": np.array([-3.2e-16, -1.0, -4.0, -9.0]) + 1j * np.array(
            [1e-20, -2e-16, 5e-16, -9e-16]),
    }
    for name, lam2 in cases.items():
        r = np.sqrt(np.asarray(lam2, dtype=_C))
        scale = max(float(np.max(np.abs(r))), 1.0)
        ratio = np.abs(r.real) / scale
        on_cut = ratio <= _rc._CUT_BAND_REL * scale / scale
        out = _rc._sqrt_decay(np.asarray(lam2, dtype=_C))
        flipped = int(np.sum(out != r))
        dec["branch_cut/%s/n_on_cut" % name] = str(int(np.sum(on_cut)))
        dec["branch_cut/%s/n_conjugated" % name] = str(flipped)
        # the ORIENTATION itself: after the selector, does every mode carry
        # a non-negative imaginary part where the real part is on the cut?
        oriented = bool(np.all(out.imag[on_cut] >= 0.0)) if on_cut.any() \
            else True
        dec["branch_cut/%s/outgoing_everywhere" % name] = (
            "yes" if oriented else "no")
        rea["branch_cut/%s/max_ratio" % name] = float(np.max(ratio))
        rea["branch_cut/%s/min_ratio" % name] = float(np.min(ratio))


# ======================================================================
#: The symbol spellings the bundled BLAS wheels export for the RUNTIME core
#: name and the thread width.  The scipy-openblas wheels rename OpenBLAS's own
#: symbols so that numpy's ILP64 build and scipy's LP64 build can be loaded
#: into one process without colliding, which is why the plain names are absent
#: and the prefixed ones are what is there.  MEASURED 2026-09-12 on this
#: workstation: ``libscipy_openblas-<hash>.dll`` (scipy 1.17.1) exports
#: ``scipy_openblas_get_corename`` and ``libscipy_openblas64_-<hash>.dll``
#: (numpy 2.4.6) exports ``scipy_openblas_get_corename64_``; neither exports
#: the unmangled ``openblas_get_corename``.  Both plain spellings are kept
#: first because a distro OpenBLAS does export them.
_BLAS_CORENAME_SYMBOLS = ("openblas_get_corename",
                          "openblas_get_corename64_",
                          "scipy_openblas_get_corename",
                          "scipy_openblas_get_corename64_")
_BLAS_NTHREADS_SYMBOLS = ("openblas_get_num_threads",
                          "openblas_get_num_threads64_",
                          "scipy_openblas_get_num_threads",
                          "scipy_openblas_get_num_threads64_")


def _is_blas_image(name):
    low = os.path.basename(name).lower()
    return ("openblas" in low or "mkl_rt" in low or "libblas" in low
            or "libmkl_core" in low)


def _loaded_blas_paths():
    """Paths of the BLAS shared libraries LOADED INTO THIS PROCESS.

    Two routes, and the order is not arbitrary.  ``/proc/self/maps`` is
    first because it is dependency-free and is the only route available on
    the WSL build of this census, whose interpreter is a bare venv without
    ``psutil``; ``psutil`` (a declared core dependency of the library) is the
    Windows route, where there is no ``/proc``.  An empty answer is not an
    error -- the caller degrades to ``unknown``, which is what the census
    recorded before this fallback existed.
    """
    paths = set()
    try:
        with open("/proc/self/maps", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                part = line.rstrip("\n").split(" ", 5)[-1].strip()
                if part.startswith("/") and _is_blas_image(part):
                    paths.add(part)
    except OSError:
        pass
    if not paths:
        try:
            import psutil  # noqa: PLC0415
            paths = {m.path for m in psutil.Process().memory_maps()
                     if _is_blas_image(m.path)}
        except (ImportError, OSError, AttributeError, NotImplementedError):
            paths = set()
    return sorted(paths)


def _blas_read_back_ctypes():
    """``(kernel, num_threads, per-library detail)`` read out of the LOADED
    BLAS libraries with ``ctypes``.

    WHY THIS EXISTS.  ``threadpoolctl`` is a declared CORE dependency of the
    library (``pyproject.toml``), and it is nevertheless ABSENT on two of the
    machines this census has to speak about: the GitHub runner says so itself
    in the 5.45.0 logs (``docs/audits/CI_PREMISE_GATES_2026_09_11.md`` 2.1,
    which is why the transcribed CI arm is called ``CI-unknown-t1``), and it
    is absent from this workstation's interpreter as well.  Without it the
    ``threadpool_info`` path below yields ``unknown``, and then FOUR
    kernel arms all want the same arm key -- i.e. the census's own kernel
    axis collapses exactly where it is supposed to discriminate.

    This is the same READ-BACK threadpoolctl performs, done directly:
    ``openblas_get_corename()`` returns the micro-kernel OpenBLAS actually
    DISPATCHED, which is the property this census needs and is NOT the same
    thing as ``OPENBLAS_CORETYPE`` (``ZEN`` and ``BOGUSCORE`` both resolve to
    something else -- see the module docstring's measured caveats).  Reading
    the request instead would let two arms that ran identical code look like
    independent evidence, so the fallback keeps the invariant rather than
    trading it away for a name.

    ``ctypes.CDLL`` on an already-loaded library takes a reference to the
    loaded image; it does not load a second copy.  When several BLAS
    libraries are loaded and they DISAGREE the names are joined with ``+``,
    because an arm whose two libraries dispatch different kernels is neither
    of them and must not be filed under either.
    """
    detail, cores, widths = {}, [], []
    try:
        import ctypes  # noqa: PLC0415
    except ImportError:                            # pragma: no cover
        return "unknown", None, {}
    paths = _loaded_blas_paths()
    for path in paths:
        core, width = None, None
        try:
            lib = ctypes.CDLL(path)
        except OSError:
            continue
        for name in _BLAS_CORENAME_SYMBOLS:
            fn = getattr(lib, name, None)
            if fn is None:
                continue
            fn.restype = ctypes.c_char_p
            try:
                core = fn().decode("ascii", "replace")
            except (OSError, ValueError, AttributeError):
                core = None
            break
        for name in _BLAS_NTHREADS_SYMBOLS:
            fn = getattr(lib, name, None)
            if fn is None:
                continue
            fn.restype = ctypes.c_int
            try:
                width = int(fn())
            except (OSError, ValueError):
                width = None
            break
        detail[os.path.basename(path)] = {"corename": core,
                                          "num_threads": width}
        if core:
            cores.append(core)
        if width:
            widths.append(width)
    if not cores:
        return "unknown", (max(widths) if widths else None), detail
    uniq = sorted(set(cores))
    return ("+".join(uniq), (max(widths) if widths else None), detail)


def _arm_id():
    """``BUILD-KERNEL-tN`` -- the arm's identity, all three parts MEASURED.

    The kernel part is read back from the loaded OpenBLAS rather than from
    ``OPENBLAS_CORETYPE``, because the request and the result are NOT the
    same thing: ``ZEN`` and ``BOGUSCORE`` both resolve to the host's
    auto-detected kernel in these wheels (see the module docstring).
    Recording the REQUEST would let two arms that ran identical code look
    like independent evidence.

    ``threadpoolctl`` is preferred because it is the library's own instrument
    and understands MKL as well; :func:`_blas_read_back_ctypes` is the
    fallback for the hosts that do not have it -- the CI runner and this
    workstation both -- and reads the SAME ``openblas_get_corename``
    threadpoolctl reads.  Which of the two answered is recorded in
    ``kernel_source`` so a reader of the census can tell.

    The thread part is likewise read back from the loaded library, not from
    the environment: ``tauto`` is an arm with NO cap set, where OpenBLAS
    picked its own width from the core count -- which is exactly what CI's
    fast lane does, and what the SLOW and JAX lanes deliberately do not.  An
    unpinned arm on a 16-core workstation and an unpinned arm on a 4-core
    runner are different arms, so the width OpenBLAS actually chose is
    recorded in ``blas_threads`` beside the label.
    """
    build = "WSL" if sys.platform.startswith("linux") else "WIN"
    arch, nthreads, source, detail = "unknown", None, "none", {}
    try:
        import threadpoolctl  # noqa: I001, PLC0415
        for d in threadpoolctl.threadpool_info():
            if d.get("internal_api") not in ("openblas", "mkl"):
                continue
            # The same per-library table the ctypes fallback records, so an
            # arm carries the loaded BLAS builds whichever instrument answered.
            lib = os.path.basename(str(d.get("filepath") or d.get("prefix")
                                       or d.get("internal_api")))
            detail[lib] = {"corename": str(d.get("architecture") or "unknown"),
                           "num_threads": d.get("num_threads")}
            if arch == "unknown":
                arch = str(d.get("architecture") or "unknown")
                nthreads = d.get("num_threads")
                source = "threadpoolctl"
    except Exception:                                   # noqa: BLE001
        pass
    if arch == "unknown":
        arch, nthreads, detail = _blas_read_back_ctypes()
        source = "ctypes(openblas_get_corename)" if arch != "unknown" \
            else "none"
    cap = os.environ.get("OPENBLAS_NUM_THREADS", "") \
        or os.environ.get("OMP_NUM_THREADS", "")
    tag = ("t%s" % cap) if cap else "tauto"
    return ("%s-%s-%s" % (build, arch, tag), build, arch, tag, nthreads,
            source, detail)


def _tree_id():
    """``<branch> <short sha>[ +dirty]`` for the tree this arm was taken on.

    An arm without its tree is not evidence: the 2026-09-11 census went stale
    because a library default moved under it and nothing in the table said
    which library it had measured.  Read-only ``git``; degrades to the empty
    string off a checkout, in which case ``merge_arms.py`` says so.
    """
    import subprocess  # noqa: PLC0415
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))

    def _git(*args):
        return subprocess.run(("git", "-C", root) + args, check=True,
                              capture_output=True, text=True).stdout.strip()

    try:
        sha = _git("rev-parse", "--short", "HEAD")
        branch = _git("rev-parse", "--abbrev-ref", "HEAD")
        dirty = bool(_git("status", "--porcelain"))
    except (OSError, subprocess.SubprocessError):
        return ""
    return "%s %s%s" % (branch, sha, " +dirty" if dirty else "")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None, help="write the arm JSON here")
    args = ap.parse_args(argv)

    dec, hyp, rea, cls = {}, {}, {}, {}
    _interface_site_population(dec, hyp, rea, cls)
    _sliver_decisions(dec, rea, cls)
    _mortar_decisions(dec, rea)
    _band_decisions(dec, rea)
    _t22_decisions(dec, rea)
    _branch_cut_decisions(dec, rea)

    arm, build, arch, tag, nthreads, ksource, kdetail = _arm_id()
    doc = {
        "arm": arm,
        "build": build,
        "kernel": arch,
        "kernel_source": ksource,
        "blas_libraries": kdetail,
        "recorded": __import__("datetime").date.today().isoformat(),
        "tree": _tree_id(),
        "lumenairy": getattr(__import__("lumenairy"), "__version__", "?"),
        "min_feature_default_frac": float(_st1d._MIN_FEATURE_DEFAULT_FRAC),
        "thread_arm": tag,
        "blas_threads": nthreads,
        "coretype_requested": os.environ.get("OPENBLAS_CORETYPE", ""),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": __import__("scipy").__version__,
        "threads": {v: os.environ.get(v, "")
                    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                              "MKL_NUM_THREADS")},
        "decisions": dec,
        "classes": cls,
        "hypothetical": hyp,
        "readings": rea,
    }
    text = json.dumps(doc, indent=1, sort_keys=True)
    if args.out:
        with open(args.out, "w", encoding="cp1252", errors="replace") as fh:
            fh.write(text + "\n")
        print("wrote %s (%d decisions)" % (args.out, len(dec)))
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
