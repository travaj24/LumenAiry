"""DECISION tests from the INDEPENDENT VERIFICATION of branch-cut round 2.

Evidence: ``docs/audits/VERIFY_BRANCH_CUT_ROUND2_2026_09_11.md`` and
``validation/probe_verify_branch_cut_round2/`` (probes + JSON, both builds).
Only the places where re-measurement found a GAP in
``tests/unit/test_fix_branch_cut_round2.py`` are pinned here; everything that
file already gates is left to it.

BUILDS.  WIN = Windows 11, python 3.14.6, numpy 2.4.4, scipy-openblas
dispatching Haswell.  WSL = Ubuntu on the same box, python 3.12.3,
numpy 2.4.6, SkylakeX.  Every number below was measured on BOTH, and where a
thread count matters the ladder 1 / 2 / 4 / 8 was run.

Each fail-before is ENGINEERED (the pre-round-1 branch body reinstalled in
process), per ``docs/TESTING_STANDARDS.md`` restatement 3: the defect no longer
occurs anywhere in the shipped tree, so a FOUND fail-before would be a
per-build accident.
"""
from __future__ import annotations

import importlib
import warnings
from pathlib import Path

import numpy as np
import pytest

from lumenairy.elements.pmm import PMM2DStackHybrid
from lumenairy.elements.rcwa import _core as _rc
from lumenairy.elements.rcwa import rcwa_efficiency_1d

# --------------------------------------------------------------- the fixture
_WL, _P, _D = 0.6e-6, 0.5e-6, 0.2e-6
_HOST, _N_SUB = 2.25, 1.63
_S = 6


def _cell(rel=1e-6, loss=0.0):
    c = np.full((_S, _S), _HOST + 0j)
    c[2:4, 2:4] = _HOST * (1.0 + rel)
    return c + 1j * loss


def _stack(n_orders=4, rel=1e-6, cell_loss=0.0, spacer_loss=0.0, detune=0.0,
           theta=0.0, phi=0.0):
    """The round-2 uniform-spacer stack, with the three knobs this file needs:
    loss placed in the CELL or in the SPACER separately, and a relative detune
    of the spacer off the cell background."""
    st = PMM2DStackHybrid(_P, _P, n_substrate=_N_SUB, n_superstrate=1.0,
                          degree=7, n_orders=n_orders, symmetry=False)
    sp = (_HOST + 1j * spacer_loss) * (1.0 + detune)
    st.add_layer(0.1e-6, eps=sp)
    st.add_layer(_D, eps_cell=_cell(rel, cell_loss))
    st.add_layer(0.1e-6, eps=sp)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o, R, T, _J = st.set_source(_WL, theta=theta, phi=phi).solve()
    return np.asarray(R), np.asarray(T)


def _closure(RT):
    R, T = RT
    return float(np.sum(R) + np.sum(T) - 2.0)


def _pre_round1_sqrt_decay(x, xp=None, band=1e-8):
    """The pre-round-1 body: the EXACT ``Re(r) == 0`` pin and the ``-r`` flip.
    An ``eig`` output never satisfies it, so it fires for the analytic REGION
    and UNIFORM-LAYER modes and never for a structured layer's."""
    from lumenairy.backend.array import array_namespace
    if xp is None:
        xp = array_namespace(x)
    x = xp.asarray(x).astype(complex)
    r = xp.sqrt(x)
    on_cut = r.real == 0
    return xp.where(on_cut & (r.imag < 0), -r, r)


class _pre_arm:
    """Install ``_pre_round1_sqrt_decay`` at EVERY module binding of the shared
    selector -- discovered by walking ``lumenairy/elements``, not listed, so a
    future module that imports it is covered without editing this file."""

    def __init__(self):
        self._saved = []

    def __enter__(self):
        import lumenairy.elements as EL
        base = Path(EL.__file__).parent
        for p in sorted(base.rglob("*.py")):
            rel = p.relative_to(base).with_suffix("")
            name = "lumenairy.elements." + ".".join(rel.parts)
            if name.endswith(".__init__"):
                name = name[: -len(".__init__")]
            try:
                mod = importlib.import_module(name)
            except Exception:
                continue
            if callable(getattr(mod, "_sqrt_decay", None)):
                self._saved.append((mod, mod._sqrt_decay))
                mod._sqrt_decay = _pre_round1_sqrt_decay
        assert self._saved, "no binding of _sqrt_decay was found to patch"
        return self

    def __exit__(self, *a):
        for mod, fn in self._saved:
            mod._sqrt_decay = fn
        self._saved = []
        return False


# ==================================================================== GATE 1
# WHICH LAYER'S LOSS PUTS A FIXTURE OUT OF SCOPE.
#
# The round-2 census reports "LOSSY 11/11 bit-identical" and the fix document
# generalises it to "where the sign is physics, nothing moves".  Re-measured
# (v7_followups.py, both builds), the discriminator is the loss in the
# PATTERNED layer, not loss anywhere in the stack:
#
#   fixture (oblique 12 deg / phi 20 deg unless said)   |post - pre| per order
#   lossy CELL Im(eps) = 1e-2, normal and conical            0.0000e+00
#   lossy CELL Im(eps) = 1e-6, normal and conical            0.0000e+00
#   lossy CELL + lossy SPACER, normal and conical            0.0000e+00
#   lossy SPACER only, Im = 1e-2, normal / conical      2.03e-13 / 5.52e-13
#   lossy SPACER only, Im = 1e-6, normal / conical      1.34e-03 / 9.53e-05
#
# The last row is the point: a stack that a caller would call lossy, whose
# patterned layer is lossless, was still WRONG by 1.3e-03 per order before
# round 2.  The scope statement has to name the PATTERNED layer or it will be
# read as "any loss is safe", which is false by three decades.

@pytest.mark.parametrize("theta,phi", [(0.0, 0.0), (0.21, 0.35)])
@pytest.mark.parametrize("cell_loss", [1e-2, 1e-6])
def test_loss_in_the_patterned_layer_is_what_makes_the_arms_agree(
        cell_loss, theta, phi):
    """A LOSSY PATTERNED CELL is bit-identical between the two branch bodies:
    every mode's ``Im(lam^2)`` takes the sign the loss dictates, so the
    acted-on population is empty and the flip has nothing to act on.  Asserted
    at normal AND conical incidence, and with the spacer both lossless and
    lossy, so the claim is about the CELL and not about the mount."""
    for spacer_loss in (0.0, 1e-2):
        R0, T0 = _stack(cell_loss=cell_loss, spacer_loss=spacer_loss,
                        theta=theta, phi=phi)
        with _pre_arm():
            R1, T1 = _stack(cell_loss=cell_loss, spacer_loss=spacer_loss,
                            theta=theta, phi=phi)
        assert float(np.max(np.abs(R0 - R1))) == 0.0, (
            "a lossy patterned cell (Im eps = %.0e, spacer Im = %.0e, "
            "theta = %.2f) moved by %.3e between the branch bodies: the flip "
            "acted on a mode whose imaginary sign is PHYSICS"
            % (cell_loss, spacer_loss, theta,
               float(np.max(np.abs(R0 - R1)))))
        assert float(np.max(np.abs(T0 - T1))) == 0.0


#: Modes the shipped band CONJUGATES during one solve, counted on the
#: EIGENPROBLEM (so the count is identical on both arms of round 2 and needs no
#: reference solve).  Measured over ``n_orders`` 3 / 4 / 5, normal and conical
#: (``v10_acted_population.py``, 2026-09-11):
#:
#:   fixture                          WIN            WSL
#:   fully lossless                   5 / 6 / 7      8 / 4 / 4
#:   lossy CELL 1e-2 or 1e-6          0 / 0 / 0      0 / 0 / 0
#:   lossy CELL + lossy SPACER        0 / 0 / 0      0 / 0 / 0
#:   lossy SPACER only, 1e-2 or 1e-6  5 / 6 / 7      8 / 4 / 4
#:   lossy SPACER only, conical       6 / 6 / 6      6 / 6 / 6
#:
#: The COUNTS are per-build; what is not is that the lossy-CELL column is zero
#: and the lossy-SPACER column is not.  The assertions below read only that.
_SPACER_LOSS_LADDER = (3, 4, 5)


def _acted_on_layer_modes(**kw):
    """Count the modes the shipped band conjugates in one solve of ``_stack``,
    separating LAYER eigenproblem arrays from the analytic Rayleigh helper's
    (whose ``Im`` is exactly zero by construction)."""
    n_layer = 0
    saved = _rc._sqrt_decay

    def tap(x, xp=None, band=_rc._CUT_BAND_REL):
        nonlocal n_layer
        xx = np.asarray(x).astype(complex)
        if xx.size and not np.all(xx.imag == 0.0):
            r = np.sqrt(xx)
            mx = max(float(np.max(np.abs(r))), 1.0)
            n_layer += int(np.sum((np.abs(r.real) <= band * mx)
                                  & (r.imag < 0)))
        return saved(x, xp, band) if xp is not None else saved(x, band=band)

    import lumenairy.elements as EL
    base = Path(EL.__file__).parent
    patched = []
    for p_ in sorted(base.rglob("*.py")):
        rel = p_.relative_to(base).with_suffix("")
        name = "lumenairy.elements." + ".".join(rel.parts)
        if name.endswith(".__init__"):
            name = name[: -len(".__init__")]
        try:
            mod = importlib.import_module(name)
        except Exception:
            continue
        if callable(getattr(mod, "_sqrt_decay", None)):
            patched.append((mod, mod._sqrt_decay))
            mod._sqrt_decay = tap
    try:
        _stack(**kw)
    finally:
        for mod, fn in patched:
            mod._sqrt_decay = fn
    return n_layer


def test_a_lossy_spacer_alone_does_not_empty_the_acted_on_population():
    """The other side of gate 1, stated on the MECHANISM rather than on a
    closure reading, because whether a given truncation of a given fixture
    manifests the pre-round-1 defect is a per-build fact (measured: this
    fixture's worst pre/post closure ratio over ``n_orders`` 3/4/5 is 6.95 on
    WIN and 1.05 on WSL, so a ratio bar there would be per-build).

    What IS build-free is the count the band's own predicate produces.  The
    round-2 file's gate 6 justifies its lossy exemption with "for a lossy layer
    the acted-on population is EMPTY".  That is true of the PATTERNED layer and
    false of the SPACER: with loss only in the spacer the patterned layer's
    propagating modes are still exactly on the cut, and the band still acts on
    every one of them -- which is why such a stack was still wrong by 1.3e-03
    per order before round 2.  If this ever reads zero, "lossy" has silently
    widened to mean "loss anywhere", and the scope statement stops being true.
    """
    for M in _SPACER_LOSS_LADDER:
        with_cell_loss = _acted_on_layer_modes(n_orders=M, cell_loss=1e-6)
        assert with_cell_loss == 0, (
            "a LOSSY PATTERNED CELL (Im eps = 1e-6, n_orders = %d) still puts "
            "%d layer mode(s) in the acted-on population: the exemption gate 1 "
            "relies on does not hold" % (M, with_cell_loss))
    n = {M: _acted_on_layer_modes(n_orders=M, spacer_loss=1e-6)
         for M in _SPACER_LOSS_LADDER}
    assert all(v > 0 for v in n.values()), (
        "with loss in the SPACER only, the acted-on population is empty at "
        "%s: the lossy exemption would then be about loss ANYWHERE in the "
        "stack, which it is not -- measured 4..8 acted-on layer modes on both "
        "builds" % {k: v for k, v in n.items()})
    for M in _SPACER_LOSS_LADDER:
        assert _acted_on_layer_modes(n_orders=M, cell_loss=1e-2,
                                     spacer_loss=1e-2) == 0, (
            "loss in BOTH layers does not empty the population at n_orders "
            "= %d" % M)


# ==================================================================== GATE 2
# THE DEFECT IS NOT CONFINED TO A MODULATION WINDOW.
#
# The fix document's section 4.4 reports the defect visible "in a window around
# 1e-06, and only there", on the grounds that below 1e-08 "the layer is
# numerically uniform and routes to the analytic Rayleigh helper, where the old
# pin was always correct".  Re-measured on the same fixture family
# (v2_hybrid.py modulation ladder, WIN 1 thread, n_orders = 4):
#
#   relative pillar-host   1e-10      1e-08      1e-06      1e-04     1e-02
#   PRE  closure         3.29e-05  -1.29e-07   1.00e-05   1.14e-07   5.41e-07
#   POST closure         1.66e-10   1.66e-10   1.63e-10   1.26e-10   5.37e-07
#
# 1e-10 is not inside any window around 1e-06, and it is the LOUDEST rung.  The
# claim this pins is the one that survives: the repaired stack closes at every
# modulation on the ladder, and the pre-round-1 arm does not.

_MOD_LADDER = (1e-10, 1e-8, 1e-6, 1e-4)
_TRUNC_LADDER = (3, 4, 5)

#: Worst ``pre / floor`` over ``n_orders`` 3/4/5, per modulation rung
#: (``v9_ladder_scan.py``, 2026-09-11).  Scanning the truncations is not
#: decoration: WHICH one manifests is a per-build fact (at 1e-10 the defect
#: shows at n_orders 3/4/5 on WIN and at 3 and 5 but NOT 4 on WSL), exactly as
#: the fix file own gate 2 records for its ladder.
#:
#:   modulation   1e-10      1e-08      1e-06      1e-04
#:   WIN        4.98e+10   1.05e+11   2.68e+10   6.91e+05
#:   WSL        1.92e+11   6.46e+10   1.73e+10   2.08e+06
#:
#: POST over the same 24 samples never exceeds 1.0x its own floor.  The bar
#: below is 30x: 1.5 decades above the POST envelope and 4.2 decades below the
#: smallest PRE reading, on both builds.
_MOD_RATIO_BAR = 30.0


def test_the_repair_holds_at_every_modulation_not_only_a_window_near_1e_6():
    """POST: the coincident-spacer stack closes at every rung of a four-decade
    modulation ladder AND at every truncation of the 3/4/5 ladder.  The bar is
    derived at RUNTIME from the same fixture DETUNED sibling at the same rung,
    so it tracks the mount own Fourier truncation error instead of pinning a
    number.

    PRE (engineered): at least one rung BELOW 1e-06 -- where the fix document
    says the layer "is numerically uniform and routes to the analytic Rayleigh
    helper, where the old pin was always correct" -- misses by more than the
    sibling own error.  That is what makes the "window around 1e-06, and only
    there" reading sample-scoped."""
    post, floor = {}, {}
    for rel in _MOD_LADDER:
        post[rel] = max(abs(_closure(_stack(n_orders=M, rel=rel)))
                        for M in _TRUNC_LADDER)
        floor[rel] = max(abs(_closure(_stack(n_orders=M, rel=rel,
                                             detune=1e-2)))
                         for M in _TRUNC_LADDER)
    for rel in _MOD_LADDER:
        assert post[rel] < _MOD_RATIO_BAR * max(floor[rel], 1e-15), (
            "the repaired stack at modulation %.0e closes at %.3e against its "
            "own detuned sibling %.3e" % (rel, post[rel], floor[rel]))
    with _pre_arm():
        pre = {rel: max(abs(_closure(_stack(n_orders=M, rel=rel)))
                        for M in _TRUNC_LADDER) for rel in _MOD_LADDER}
    below = [rel for rel in _MOD_LADDER
             if rel < 1e-6
             and pre[rel] > _MOD_RATIO_BAR * max(floor[rel], 1e-15)]
    assert below, (
        "the pre-round-1 arm is clean at every modulation below 1e-06 on this "
        "build (%s against floors %s): the fail-before no longer shows that "
        "the defect reaches outside the 1e-06 window"
        % ({("%.0e" % k): "%.3e" % v for k, v in pre.items()},
           {("%.0e" % k): "%.3e" % v for k, v in floor.items()}))


# ==================================================================== GATE 3
# THERE IS NO CURE-DETUNE CONSTANT.
#
# Defect D6's correction replaced "detune by a relative ~1e-6" with a statement
# that the class was repaired.  The fix document backs it with "only 1e-3
# brings it to 8.7e-14".  Re-measured on my own fixture the cure arrives
# between 3e-06 and 1e-05 (PRE closure 1.98e-07 -> 7.30e-10), so 1e-3 is not
# the cure point either -- it is one fixture's.  What IS build- and
# fixture-independent, and is what this pins, is that the REPAIRED answer does
# not depend on the detune at all: measured flat to 1.2e-12 relative over
# detunes 0 .. 1e-1 on both builds (v7_followups.py, 24 samples).

_DETUNE_LADDER = (0.0, 1e-6, 1e-4, 1e-2)

#: POST closure spread across the detune ladder, and the PRE spread on the same
#: ladder, per truncation (``v9_ladder_scan.py``, 2026-09-11):
#:
#:   n_orders          3            4            5
#:   POST spread WIN  3.55e-15     8.88e-16     2.89e-15
#:   POST spread WSL  1.78e-15     2.00e-15     3.55e-15
#:   PRE  spread WIN  8.55e-05     3.26e-03     5.60e-05
#:   PRE  spread WSL  2.52e-04     6.66e-16*    7.69e-06
#:
#:   (*) n_orders = 4 does not manifest on WSL -- the per-build partition the
#:   fix file own gate 2 documents -- which is why the PRE claim is taken on
#:   the WORST of the truncation ladder and not on one mount.
#:
#: The POST bar is 1e-12: 2.4 decades above the worst POST spread over the six
#: (build, truncation) samples and 6.9 decades below the smallest PRE spread
#: that manifests (7.69e-06).
_DETUNE_FLATNESS_BAR = 1e-12


def test_the_repaired_answer_does_not_depend_on_a_spacer_detune():
    """POST: closure is flat across four decades of spacer detune at every
    truncation, so no remedy text needs to name a cure detune.  (The fix
    document "only 1e-3 brings it to 8.7e-14" is one fixture: on mine the
    pre-round-1 cure arrives between 3e-06 and 1e-05.)

    PRE (engineered): the spread on the same ladder is orders larger at at
    least one truncation, which is what made a detune look like a remedy."""
    post = {M: {d: _closure(_stack(n_orders=M, detune=d))
                for d in _DETUNE_LADDER} for M in _TRUNC_LADDER}
    spread_post = {M: max(v.values()) - min(v.values())
                   for M, v in post.items()}
    worst_post = max(spread_post.values())
    assert worst_post < _DETUNE_FLATNESS_BAR, (
        "the repaired closure moves by %.3e across the detune ladder "
        "(per truncation: %s): it is supposed to be detune-independent"
        % (worst_post, {M: "%.3e" % v for M, v in spread_post.items()}))
    with _pre_arm():
        pre = {M: {d: _closure(_stack(n_orders=M, detune=d))
                   for d in _DETUNE_LADDER} for M in _TRUNC_LADDER}
    spread_pre = {M: max(v.values()) - min(v.values()) for M, v in pre.items()}
    assert max(spread_pre.values()) > 1e3 * max(worst_post, 1e-15), (
        "the pre-round-1 arm closure is as flat across the detune ladder as "
        "the repaired one (worst %.3e against %.3e; per truncation %s), so "
        "this fixture no longer shows why a detune ever looked like a remedy"
        % (max(spread_pre.values()), worst_post,
           {M: "%.3e" % v for M, v in spread_pre.items()}))


# ==================================================================== GATE 4
# THE LAYER-CUTOFF CORNER IS NEVER *SILENTLY* WRONG.
#
# ``_CUT_BAND_REL``'s comment says the cutoff corner has "no wrong answer ...
# on any of the 72 + 28 cutoff mounts measured".  Driven deeper than either
# ladder -- ``min |lam^2|`` to 3.1722e-16 (WIN) / 4.0185e-16 (WSL), against the
# fix document's 6.5341e-10 and the round-1 verification's 4.495e-15 -- the
# statement needs one qualifier, and gains a stronger form:
#
#   * the two shapes' populations OVERLAP there (ARRAY-MAX two-sided gap
#     -0.29 decades on WIN, +0.01 on WSL), and a mode whose real part is
#     99.76 % (WIN) / 99.993 % (WSL) of its OWN magnitude is conjugated;
#   * BUT over 18 mounts x 2 builds x 4 branch rules (shipped / band = 0 /
#     per-mode / the pre-round-1 pin), every mount on which the shipped rule
#     returns SILENTLY moves by at most 8.75e-11 per order between the rules,
#     while every mount where they diverge materially (up to 9.5104e-02) is
#     LOUD -- an ``_EnergyWarning`` or an ``_EnergyError``.
#   * and the band is LOAD-BEARING: with ``band = 0`` the solve RAISES on 16 of
#     those 18 mounts.
#
# So the property to pin is not "the corner is harmless" but "the corner is
# never silent", which is a decision a future change to the band shape or
# constant would have to re-earn.  (v4_band.py, v6_cutoff_consequence.py.)

_SILENT_MOTION_BAR = 1e-8


def _min_abs_lam2(n_ridge, pol="te", M=9, duty=0.5):
    """The smallest ``|lam^2|`` any layer eigenproblem of this mount produces,
    read off the selector's own input."""
    seen = []
    import lumenairy.elements.rcwa.oned as _ro
    saved_c, saved_o = _rc._sqrt_decay, _ro._sqrt_decay

    def tap(x, *a, **kw):
        xx = np.asarray(x).astype(complex)
        if xx.size and not np.all(xx.imag == 0.0):
            nz = np.abs(xx[np.abs(xx) > 0])
            if nz.size:
                seen.append(float(nz.min()))
        return saved_c(x, *a, **kw)

    _rc._sqrt_decay = tap
    _ro._sqrt_decay = tap
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rcwa_efficiency_1d(1.0e-6, n_ridge, 1.0, 1.5, 1.0, 0.45e-6, duty,
                               _WL, polarization=pol, n_orders=M)
    except Exception:
        pass
    finally:
        _rc._sqrt_decay = saved_c
        _ro._sqrt_decay = saved_o
    return min(seen) if seen else float("inf")


def _hunt_cutoff(pol="te", M=9, duty=0.5):
    """Drive ``min |lam^2|`` down over the ridge index: a coarse bracketing
    scan, then golden section.  Derived at runtime rather than pinned, so the
    mount tracks the build rather than being a found state (TESTING_STANDARDS
    restatement 3)."""
    grid = np.linspace(1.3, 3.4, 31)
    vals = [_min_abs_lam2(g, pol, M, duty) for g in grid]
    k = int(np.argmin(vals))
    a, b = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
    gr = (np.sqrt(5.0) - 1.0) / 2.0
    c, d = b - gr * (b - a), a + gr * (b - a)
    fc, fd = _min_abs_lam2(c, pol, M, duty), _min_abs_lam2(d, pol, M, duty)
    for _ in range(45):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = _min_abs_lam2(c, pol, M, duty)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = _min_abs_lam2(d, pol, M, duty)
        if abs(b - a) < 1e-15 * max(abs(a), 1.0):
            break
    nr = 0.5 * (a + b)
    return nr, _min_abs_lam2(nr, pol, M, duty)


def _rule_permode(x, xp=None, band=1e-8):
    x = np.asarray(x).astype(complex)
    r = np.sqrt(x)
    mx = max(float(np.max(np.abs(r))), 1.0) if r.size else 1.0
    on_cut = np.abs(r.real) <= band * np.maximum(
        np.abs(r), float(np.sqrt(np.finfo(np.float64).eps)) * mx)
    return np.where(on_cut & (r.imag < 0), np.conj(r), r)


class _rule_arm(_pre_arm):
    def __init__(self, fn):
        super().__init__()
        self._fn = fn

    def __enter__(self):
        super().__enter__()
        for mod, _fn in self._saved:
            mod._sqrt_decay = self._fn
        return self


def _solve_1d(n_ridge, pol, M, duty):
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        try:
            o, R, T = rcwa_efficiency_1d(1.0e-6, n_ridge, 1.0, 1.5, 1.0,
                                         0.45e-6, duty, _WL,
                                         polarization=pol, n_orders=M)
        except Exception as exc:
            return None, "raised:" + type(exc).__name__
        loud = bool(ws)
    return (np.asarray(R), np.asarray(T)), ("warned" if loud else "silent")


@pytest.mark.parametrize("pol,M,duty", [("te", 9, 0.5), ("tm", 11, 0.4)])
def test_at_a_deep_layer_cutoff_the_band_is_never_silently_wrong(pol, M, duty):
    """Drive a mount onto a LAYER CUTOFF, then read the same solve under the
    shipped branch rule and under the PER-MODE alternative the fix document
    rejected.  Whenever the shipped rule returns SILENTLY the two must agree to
    ``_SILENT_MOTION_BAR``; when they do not agree, the shipped rule must be
    LOUD.

    ``_SILENT_MOTION_BAR = 1e-8`` sits 2.06 decades above the worst SILENT
    motion measured (8.75e-11 over 18 mounts x 2 builds x 4 rules,
    v6_cutoff_consequence.py, 2026-09-11) and 6.04 decades below the worst LOUD
    one (9.5104e-02), inside a gap the measurement leaves empty.
    """
    nr, v = _hunt_cutoff(pol, M, duty)
    assert v < 1e-9, (
        "the cutoff hunt only reached min|lam^2| = %.3e on this build, so this "
        "test is not exercising the corner it is about" % v)
    ship, ship_state = _solve_1d(nr, pol, M, duty)
    with _rule_arm(_rule_permode):
        alt, _alt_state = _solve_1d(nr, pol, M, duty)
    if ship is None or alt is None:
        assert ship_state != "silent", (
            "the shipped rule returned silently while the per-mode rule "
            "raised at min|lam^2| = %.3e (n_ridge = %.15f)" % (v, nr))
        return
    motion = float(max(np.max(np.abs(ship[0] - alt[0])),
                       np.max(np.abs(ship[1] - alt[1]))))
    if ship_state == "silent":
        assert motion < _SILENT_MOTION_BAR, (
            "at min|lam^2| = %.3e (n_ridge = %.15f) the shipped band returns "
            "SILENTLY and the per-mode band gives an answer %.4e away per "
            "order: the cutoff corner has become silently build-dependent, "
            "which is what the band exists to prevent" % (v, nr, motion))


def test_the_band_is_load_bearing_at_a_deep_cutoff():
    """The other side: with ``band = 0`` -- no conjugation at all -- the same
    mounts do NOT quietly agree.  Measured: 16 of 18 deepest mounts RAISE
    ``_EnergyError`` with the pin disarmed, on both builds.  Pinned so that a
    future "the band never fires anyway, drop it" reading is refused."""
    bad = 0
    total = 0
    for pol, M, duty in (("te", 9, 0.5), ("te", 11, 0.5), ("tm", 11, 0.4)):
        nr, v = _hunt_cutoff(pol, M, duty)
        if v >= 1e-9:
            continue
        total += 1
        with _rule_arm(lambda x, xp=None, band=1e-8:
                       np.sqrt(np.asarray(x).astype(complex))):
            res, state = _solve_1d(nr, pol, M, duty)
        if res is None or state != "silent":
            bad += 1
    assert total >= 2, "the cutoff hunt found fewer than two mounts"
    assert bad >= 1, (
        "with the on-cut band disarmed every one of %d deep-cutoff mounts "
        "still returned silently: the band is doing nothing on this build and "
        "the corner it is derived against is no longer reachable" % total)


# ==================================================================== GATE 5
# THE AUTO-DETECTING CALL SHAPE IS PART OF THE CONTRACT.
#
# The fix document says "the three JAX twins call it with jnp".  There is a
# FOURTH JAX caller that does NOT: ``lumenairy/elements/_berreman_jax.py``
# calls ``_sqrt_decay(-jnp.concatenate([kzc, kzc]) ** 2)`` with no ``xp``, and
# it works because ``array_namespace`` recognises a JIT TRACER as well as a
# concrete ``jax.Array``.  Measured (v1_consolidation.py, both builds): under
# ``jax.jit`` the auto-detecting call is BITWISE identical to the explicit-jnp
# one over 4,015 engineered values, and ``jax.grad`` through either returns the
# same 851406297.1649544.

def test_the_auto_detecting_call_traces_under_jit_bitwise():
    """``_sqrt_decay(x)`` with no ``xp`` must keep working under ``jax.jit``:
    one shipped caller (``_berreman_jax``) depends on it, so a future
    ``if xp is None: raise`` would break a traced path that no PMM test
    covers."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    jax.config.update("jax_enable_x64", True)
    rng = np.random.default_rng(20260911)
    mag = 10.0 ** rng.uniform(-8, 8, 512)
    vals = (mag * np.exp(1j * rng.uniform(-np.pi, np.pi, 512))).astype(complex)
    vals = np.concatenate([vals, [-2.25 - 2.911e-15j, -2.25 + 2.911e-15j,
                                  +2.25 - 2.911e-15j, -1.0 + 0j]])
    jv = jnp.asarray(vals)
    auto = np.asarray(jax.jit(_rc._sqrt_decay)(jv))
    explicit = np.asarray(jax.jit(lambda z: _rc._sqrt_decay(z, jnp))(jv))
    assert np.array_equal(auto, explicit), (
        "the auto-detecting call and the explicit-jnp call disagree under "
        "jax.jit at %d of %d values"
        % (int(np.sum(auto != explicit)), auto.size))
    ref = _rc._sqrt_decay(vals)
    rel = np.abs(auto - ref) / np.maximum(np.abs(ref), 1e-300)
    assert float(rel.max()) < 1e-14, (
        "the traced body disagrees with the NumPy one by %.3e relative"
        % float(rel.max()))
    assert int(np.sum(np.sign(auto.imag) != np.sign(ref.imag))) == 0, (
        "the traced and NumPy bodies made different BRANCH decisions")
