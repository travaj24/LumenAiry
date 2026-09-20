"""VERIFY-WP-C1 ROUND 2 -- the independent re-verification of the round-2 fixes.

`feat/c1-gray-edge-round2` (10 commits `7f21eb2c`..`59a284dd` on `7ea01ede`)
claims to close VERIFY-C1's four defects (D1-D4), its recorded items and the
one mutant (M3) that survived its whole sweep.  Every claim was RE-MEASURED,
never read off the round-2 addendum; this file holds only the ids that close a
gap the re-verification FOUND, not restatements of claims the two shipped C1
files already make.

The gaps, each named in
``docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/VERIFY_WP-C1_ROUND2.md``:

  R1  D1 is closed for an ``'aperture'`` element whose params RESOLVE, and
      only for those.  Both JAX routes reach ``_aperture_edge_kwargs`` --
      the one place round 2 put the refusal -- only through
      ``_resolve_aperture_params``, and skip it when that returns ``None``;
      the NumPy chain calls it unconditionally.  So an element with no
      usable ``params`` carries an ILLEGAL rim keyword silently past both
      JAX routes while the NumPy chain raises.  Measured on both builds,
      7 element shapes out of 7, of which 6 read ``identical=True`` at
      ``49ddf4bd``, so the split is WP-C1's own and round 2 did not close it.
  R2  ``edge_samples=True`` is ACCEPTED as 1 -- which is exactly the
      pre-5.49 pixel-centre rim -- while ``edge_samples=False`` is refused
      as "not a positive integer".  The shipped census row is named
      ``edge_samples_bool_false`` and reads as "a bool is refused"; it is
      green for the wrong reason (``int(False) == 0 < 1``), and the bool a
      caller would actually write slips through.
  R3  the ``int()`` refusal family (``None``, a list, a complex) raises a RAW
      ``TypeError`` whose message names neither ``apply_aperture`` nor
      ``edge_samples``, so ``evaluate``'s own ``Raises`` section -- which
      promises a ``ValueError`` "the same refusal, from the same guard" --
      is wrong for those inputs.
  R4  D2's 1.5x ratio bar is right, but its stated DERIVATION is not: a
      30-point scan of the binding fixture's neighbourhood finds four
      fixtures below 1.0, the worst at 0.0036, because the hard arm's SIGNED
      area error crosses zero there.  "A 145-pixel rim is already well
      sampled, so the staircase it has to beat is the mildest of the three"
      would send a future re-pinner to a red.
  R5  D3's restatement is right and the rise is NOT rare: on a THIRD optic
      (lambda = 532 nm, a = 150 um, window 900 um, z = 30 / 15 mm) the hard
      arm rises by a factor 9.3 on the last refinement, and two of its three
      step orders are ABOVE second order, so "first order at best" is a
      statement about the LADDER AVERAGE and not about the steps.
  R6  the downstream boolean-cast exposure, which VERIFY-C1 and round 2 both
      recorded as unmeasurable, MOVES: the radial-RMS curve by up to 7.3 %,
      the auto bin count by a whole bin, and a wrapper merit's integrated
      power by 0.77 %.
  R7  the grey branch's peak-memory comment ("6.0 float64 grids at N = 2048,
      against 5.0") is a Windows reading; the build-free claim is the DELTA
      (one grid) and its independence of ``n_sub``.
  R8  ``generate_simulation_script(style='system')`` emits an ``'aperture'``
      element dict with no ``'edge'`` key and no ``la.apply_aperture(...)``
      call, so the Migration table's codegen recipe -- "edit the generated
      ``la.apply_aperture(...)`` call" -- does not apply to that style.

Bars.  Every bound is derived from a measurement quoted beside it, measured
2026-09-20 on Windows py3.14.6 / numpy 2.4.4 / scipy-openblas and WSL py3.12.3
/ numpy 2.4.6 / scipy-openblas, with
``OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`` on the command
line.  Raw JSON: ``validation/probe_verify_c1_round2/``.  Every mask reading
below is an integer count over ``n_sub**2`` with no BLAS in it and read
BYTE-IDENTICAL on the two builds; the two ladder readings agree to every
printed digit.

Runtime: whole file ~20 s on the reference box; the slowest id is the
third-optic ladder at ~4 s.  No test is ``slow``.
"""
import gc
import shutil
import tempfile
import tracemalloc

import numpy as np
import pytest

from lumenairy.elements import elements as elem_mod
from lumenairy.propagators.rs import rayleigh_sommerfeld_propagate
from lumenairy.propagators.system import (propagate_through_system,
                                          propagate_through_system_jax)

apply_aperture = elem_mod.apply_aperture

WAVELENGTH = 1064e-9


def _jax_or_none():
    try:
        import jax
        jax.config.update('jax_enable_x64', True)
        return jax
    except Exception:                      # noqa: BLE001 -- optional dep
        return None


def _verdict(fn):
    """``(raised, "Type: message")`` -- the verdict AND the text, because two
    routes raising for DIFFERENT reasons satisfy a verdict-only assertion."""
    try:
        fn()
    except BaseException as e:             # noqa: BLE001 -- census
        return True, "{0}: {1}".format(type(e).__name__, e)
    return False, ''


def _four_routes(elem_kw, params=None, N=64, dx=1.25e-6):
    """The four entry points a rim keyword can reach: the function itself and
    the chain's three routes.  ``params=None`` means "no ``params`` key at
    all", which is the shape R1 is about."""
    jax = _jax_or_none()
    rng = np.random.default_rng(313)
    E = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    base = {'type': 'aperture', 'shape': 'circular'}
    if params is not None:
        base['params'] = params
    elem = [dict(base, **elem_kw)]
    got = {'apply_aperture': _verdict(
        lambda: apply_aperture(E, dx, 'circular', params or {}, **elem_kw)),
        'numpy_chain': _verdict(
            lambda: propagate_through_system(E, elem, WAVELENGTH, dx=dx))}
    if jax is not None:
        import jax.numpy as jnp
        got['jax_eager'] = _verdict(lambda: propagate_through_system_jax(
            jnp.asarray(E), elem, WAVELENGTH, dx, verbose=True))
        got['jax_jit'] = _verdict(lambda: propagate_through_system_jax(
            jnp.asarray(E), elem, WAVELENGTH, dx))
    return got


# ===========================================================================
# R1 -- D1 was closed only for elements whose params resolve (CLOSED, round 3)
# ===========================================================================

_UNRESOLVABLE_BAD_RIMS = [
    ('no_params_edge_soft', {'edge': 'soft'}, None),
    ('no_params_edge_none', {'edge': None}, None),
    ('no_params_edge_samples_zero', {'edge_samples': 0}, None),
    ('empty_params_edge_soft', {'edge': 'soft'}, {}),
    ('diameter_none_edge_soft', {'edge': 'soft'}, {'diameter': None}),
]


@pytest.mark.parametrize('label,kw,params', _UNRESOLVABLE_BAD_RIMS)
def test_verify_c1r2_every_route_refuses_a_bad_rim_on_an_unresolvable_element(
        label, kw, params):
    """The contract the shipped census asserts -- "one element dict is
    accepted by all three routes or refused by all three, with the SAME
    diagnostic" -- must not depend on whether the element's PARAMS resolve.

    Round 2 put the refusal in ``_aperture_edge_kwargs`` and asserted the
    census on a fixed ``params={'diameter': 5.3e-5}``, which resolves on every
    row, so the fixed fixture could not see this.  These five element shapes
    resolve to ``None`` and, before round 3, split every time: the NumPy chain
    raised, both JAX routes returned a field.  Two-sided: the id also asserts
    the NumPy chain really does raise, so a "fix" that makes every route
    silent would not satisfy it.

    CLOSED by WP-C1 round 3, which hoists the ``_aperture_edge_kwargs(elem)``
    call ABOVE the ``_resolve_aperture_params`` gate in
    ``_system_element_signature`` and in the JAX slow path, so the element is
    read -- and refused -- once, whatever its params say.  All five params
    carried ``xfail(strict=True)`` until that landed; the markers are the
    independent fail-before and they are gone with the fix.  Re-measured
    2026-09-20 after the hoist, both builds: 0 of 7 element shapes split
    (7 of 7 before it).  Raw JSON: ``validation/probe_wpc1_round3/``.
    """
    got = _four_routes(kw, params=params)
    assert got['numpy_chain'][0], (
        "{0}: the NumPy chain ACCEPTED an element apply_aperture refuses -- "
        "the premise of this id is gone".format(label))
    chain = {k: v for k, v in got.items() if k != 'apply_aperture'}
    assert len(set(chain.values())) == 1, (
        "{0}: the chain routes disagree: ".format(label)
        + ' | '.join('{0}={1!r}'.format(k, v) for k, v in sorted(chain.items()))
    )


def test_verify_c1r2_the_unresolvable_rim_refusal_is_one_answer_and_neutral():
    """The other side of R1, and the id that says what round 3 changed.

    Before round 3 this id asserted the SPLIT -- "the NumPy chain raises and
    both JAX routes return a finite field" -- so that the defect was pinned by
    a passing id as well as by the five strict xfails above.  The hoist closes
    the split, so the id is restated as the contract that replaced it, with
    both sides measured:

      * ILLEGAL rim, unresolvable params: all FOUR entry points -- the
        function itself and the chain's three routes -- give the same
        ``(raised, "Type: message")``, and that message names
        ``apply_aperture``.  (Before the hoist: NumPy raised, both JAX routes
        returned a field, on 7 element shapes out of 7, both builds.)
      * LEGAL rim, unresolvable params: nothing raises on any route, i.e. the
        hoist refuses an illegal rim without turning a legal no-op element
        into an error.  That is the neutrality half, and without it a "fix"
        that simply raised on every unresolvable element would satisfy the
        first half.
    """
    bad = _four_routes({'edge': 'soft'}, params=None)
    assert bad['numpy_chain'][0] and 'apply_aperture' in bad['numpy_chain'][1]
    assert len(set(bad.values())) == 1, bad
    good = _four_routes({'edge': 'hard'}, params=None)
    assert not any(v[0] for v in good.values()), good
    jax = _jax_or_none()
    if jax is None:
        import importlib.util
        assert importlib.util.find_spec('jax') is None
        return
    assert set(bad) == {'apply_aperture', 'numpy_chain', 'jax_eager',
                        'jax_jit'}, sorted(bad)


# ===========================================================================
# R2 / R3 -- what the one guard does with a bool and with the int() family
# ===========================================================================

def test_verify_c1r2_a_bool_edge_samples_is_answered_the_same_way_everywhere():
    """``edge_samples=True`` and ``edge_samples=False`` must get the SAME
    answer from all four entry points, and R2 says what that answer has to be.

    Before round 3, measured 2026-09-20 byte-identically on both builds:
    ``True`` was ACCEPTED and silently meant 1 (the pre-5.49 pixel-centre
    rim), while ``False`` was refused with "must be a positive integer" --
    because ``int(False) == 0 < 1``, not because it is a bool.  Round 3
    refuses a bool EXPLICITLY, so both are refused, the refusal names the
    type, and the four routes still agree.

    Two-sided, so the id is not satisfied by a guard that simply refuses
    everything: ``numpy.int64(4)`` stays LEGAL on all four routes, which is
    the neighbour the ``isinstance`` test could plausibly have swallowed
    (``numpy.bool_`` and ``numpy.int64`` are both numpy scalars).
    """
    for value in (True, False, np.True_, np.False_):
        got = _four_routes({'edge_samples': value},
                           params={'diameter': 5.3e-5})
        assert len(set(got.values())) == 1, (value, got)
        assert got['numpy_chain'][0] is True, (value, got)
        assert 'not a bool' in got['numpy_chain'][1], (value, got)
    legal = _four_routes({'edge_samples': np.int64(4)},
                         params={'diameter': 5.3e-5})
    assert len(set(legal.values())) == 1, legal
    assert legal['numpy_chain'][0] is False, legal


def test_verify_c1r2_a_bool_edge_samples_no_longer_selects_the_pre_5_49_rim():
    """R2, the half that says what it COST.  Before round 3 the mask
    ``edge_samples=True`` selected was bit-identical to ``edge='hard'`` and to
    ``edge_samples=1`` and different from the shipped default -- i.e. a caller
    who wrote ``edge_samples=True`` intending "switch the grey rim on" got the
    OLD pre-5.49 answer with no diagnostic.

    Now it raises instead.  The id keeps the identity that made the silence
    expensive -- ``edge='hard'`` IS ``edge_samples=1`` and is NOT the default,
    measured here and not read off a comment -- so the refusal is pinned to a
    demonstrated cost rather than to a preference, and the id would go red
    either if the bool were quietly accepted again or if the two rims stopped
    differing.
    """
    N, dx, D = 128, 1e-6, 61e-6
    E = np.ones((N, N), dtype=complex)

    def mask(**kw):
        return np.real(apply_aperture(E, dx, 'circular', {'diameter': D},
                                      **kw)).tobytes()

    assert mask(edge='hard') == mask(edge='gray', edge_samples=1)
    assert mask(edge='hard') != mask()
    for value in (True, np.True_):
        with pytest.raises(ValueError, match='not a bool'):
            mask(edge_samples=value)


@pytest.mark.parametrize('value', [None, [4], 4 + 0j])
def test_verify_c1r2_the_int_refusal_family_names_the_library(value):
    """Every refusal a caller can trigger through the rim keywords should name
    the function whose contract it is -- which is what the shipped census
    asserts (``'apply_aperture' in message``) for its own rows.

    Before round 3 this family (``None``, a list, a complex) was the one that
    did not: ``_validate_edge_kwargs`` let ``int()`` raise its own
    ``TypeError`` ("int() argument must be a string, a bytes-like object or a
    real number, not 'NoneType'"), naming neither ``apply_aperture`` nor
    ``edge_samples``, while ``{'edge': None}`` on the same guard got
    ``apply_aperture``'s own ``ValueError``.  ``evaluate``'s ``Raises``
    section promises "a ValueError ... the same refusal, from the same guard",
    which was wrong for exactly these three.  Round 3 catches the ``int()``
    failure and re-raises it as that same ``ValueError``; the three params
    carried ``xfail(strict=True)`` until it landed.

    Two-sided: the same ``(raised, message)`` on all four entry points, not
    merely a better message on one of them.
    """
    got = _four_routes({'edge_samples': value}, params={'diameter': 5.3e-5})
    assert got['numpy_chain'][0], value
    assert 'apply_aperture' in got['numpy_chain'][1], got['numpy_chain'][1]
    assert 'edge_samples' in got['numpy_chain'][1], got['numpy_chain'][1]
    assert got['numpy_chain'][1].startswith('ValueError'), got['numpy_chain']
    assert len(set(got.values())) == 1, got


# ===========================================================================
# R4 -- D2's bar is right; its stated derivation is not
# ===========================================================================

def _area_ratio(d_px, dy_ratio, offset, N=512, dx=1e-6):
    """``|e_hard| / |e_g4|`` on the shipped id's own fixture family."""
    dy = dx * dy_ratio
    D = d_px * dx
    E = np.ones((N, N), dtype=np.complex128)
    analytic = np.pi * (D / 2) ** 2

    def _area(**kw):
        out = apply_aperture(E, dx, 'circular', {'diameter': D},
                             xc=offset * dx, yc=-offset * dy, dy=dy, **kw)
        return float(np.sum(np.real(out))) * dx * dy

    e_hard = _area(edge='hard') / analytic - 1.0
    e_g4 = _area(edge='gray') / analytic - 1.0
    return e_hard, e_g4


def test_verify_c1r2_the_ratio_bar_holds_on_its_own_three_fixtures():
    """The bar itself, re-measured independently: ``e_hard / e_g4`` is
    5.804555 / 8.815875 / 1.992823 on the three shipped fixtures, so the 1.5
    bar clears the smallest by 1.3285x and sits 1.5x above the degenerate 1.0
    where "grey IS hard" lives.

    Asserted as a decision (every fixture clears 1.5, and the smallest is
    below 2.1 so the bar is not trivially satisfied) rather than as the three
    readings, which would pin a build's digits.  Both builds read every digit
    identically -- a mask sum is an integer count over ``n_sub**2``.
    """
    ratios = []
    for d_px, dy_ratio, offset in ((37, 1.0, 0.37), (63, 2.5, 0.13),
                                   (145, 0.4, 0.29)):
        e_hard, e_g4 = _area_ratio(d_px, dy_ratio, offset)
        ratios.append(abs(e_hard) / abs(e_g4))
    assert min(ratios) >= 1.5, ratios
    assert 1.5 <= min(ratios) < 2.1, (
        'the binding fixture has moved away from the bar; re-derive it', ratios)


def test_verify_c1r2_a_bigger_rim_does_not_mean_a_milder_staircase_to_beat():
    """R4.  The shipped docstring derives its bar with "the 145-px fixture is
    the binding one because a 145-pixel rim is already well sampled, so the
    staircase it has to beat is the mildest of the three".

    That explanation is refuted by the fixture right next to it.  At the SAME
    ``D/dx = 145`` with ``dy/dx = 1.0`` and offset 0.23 px the hard arm's
    SIGNED area error sits at +2.19e-07 -- a zero crossing -- against the grey
    arm's +6.08e-05, so the ratio is 0.0036 and the hard arm is ~278x BETTER.
    The ratio is governed by where each arm's signed error falls relative to
    zero, not by how well sampled the rim is.

    The decision: at least one neighbour of the binding fixture has a ratio
    below 1.0, and the hard arm's signed error there is at least 100x smaller
    than the grey arm's.  Measured 2026-09-20, identical on both builds.
    """
    e_hard, e_g4 = _area_ratio(145, 1.0, 0.23)
    assert abs(e_hard) / abs(e_g4) < 1.0, (e_hard, e_g4)
    assert abs(e_g4) / abs(e_hard) > 100.0, (e_hard, e_g4)
    # and the binding fixture itself still clears the bar, so this is a scope
    # statement about the DERIVATION, not a refutation of the bar
    b_hard, b_g4 = _area_ratio(145, 0.4, 0.29)
    assert abs(b_hard) / abs(b_g4) >= 1.5


# ===========================================================================
# R5 -- D3's restatement, and the scope of "first order at best"
# ===========================================================================

# A THIRD optic, chosen so the RS spatial kernel's alias threshold
# 2 W^2 / (N lambda) -- 23.79 mm at the coarsest grid -- is cleared at every N.
R3_LAM, R3_A, R3_WINDOW, R3_Z = 532e-9, 150e-6, 900e-6, 30.0e-3
R3_NS = (128, 256, 512, 1024)


def _third_optic_rs_err(N, **kw):
    dx = R3_WINDOW / N
    E = apply_aperture(np.ones((N, N), dtype=complex), dx, 'circular',
                       {'diameter': 2.0 * R3_A}, **kw)
    out = rayleigh_sommerfeld_propagate(E, z=R3_Z, wavelength=R3_LAM, dx=dx,
                                        kernel='spatial')
    k = 2.0 * np.pi / R3_LAM
    r_a = np.sqrt(R3_Z ** 2 + R3_A ** 2)
    u = np.exp(1j * k * R3_Z) - (R3_Z / r_a) * np.exp(1j * k * r_a)
    return abs(complex(out[N // 2, N // 2]) - u) / abs(u)


def test_verify_c1r2_the_hard_arms_rise_is_common_and_its_steps_exceed_two():
    """R5.  Round 2 restated the doc claim as "the hard edge is first order at
    best and its step orders are erratic", backed by ONE independent optic on
    which the hard arm falls at every step.  On a THIRD optic it does neither:

        lambda = 532 nm, a = 150 um, window 900 um, z = 30 mm (RS spatial)
        hard  6.2976e-03  9.3199e-04  7.6432e-05  7.1042e-04
        step orders  2.756 / 3.608 / -3.216   ladder gain 8.86x, mean 1.049
        gray  4.7895e-03  1.1081e-03  2.9414e-04  9.3727e-05
        step orders  2.112 / 1.914 / 1.650     ladder gain 51.1x, mean 1.892

    So (a) the RISE is not one optic's curiosity -- here it is a factor 9.3,
    against the reference optic's 54 % -- and (b) two of the three hard step
    orders are ABOVE second order, so "first order at best" is true of the
    LADDER AVERAGE and false of the steps, which the docstring's own table
    (1.31 / 3.29 / -0.62) already shows.

    The decisions, each with a gap on both sides of a measured reading:
      * at least one hard step order EXCEEDS 2.0 (measured 3.608; bar 2.0
        sits 1.8x below it and 1.0 above "first order");
      * at least one hard step order is NEGATIVE (measured -3.216);
      * the LADDER-AVERAGE gap survives anyway: grey gains >= 4x more than
        hard over the same three halvings (measured 51.1 / 8.86 = 5.77x, so
        the 4x bar has 1.44x of room and sits 4x above the 1.0 where the two
        arms would be equal).
    The alias threshold is asserted, not assumed.
    """
    for N in R3_NS:
        dx = R3_WINDOW / N
        assert R3_Z > 2.0 * N * dx * dx / R3_LAM, (
            'the RS spatial alias threshold is not cleared at N={0}'.format(N))
    hard = [_third_optic_rs_err(N, edge='hard') for N in R3_NS]
    gray = [_third_optic_rs_err(N, edge='gray') for N in R3_NS]
    h_ord = [float(np.log2(hard[i] / hard[i + 1])) for i in range(3)]
    g_ord = [float(np.log2(gray[i] / gray[i + 1])) for i in range(3)]
    assert max(h_ord) > 2.0, (h_ord, hard)
    assert min(h_ord) < 0.0, (h_ord, hard)
    assert all(o > 0.0 for o in g_ord), (g_ord, gray)
    gain_hard = hard[0] / hard[-1]
    gain_gray = gray[0] / gray[-1]
    assert gain_gray / gain_hard >= 4.0, (gain_gray, gain_hard)


def test_verify_c1r2_the_default_arm_is_the_grey_arm_on_the_third_optic():
    """The default is ``'gray'`` on an optic neither WP-C1, WP-B11 nor
    VERIFY-C1 used, BIT for BIT, and is not the hard arm -- so the identity
    above is not three arms agreeing on one answer."""
    d = _third_optic_rs_err(256)
    assert d == _third_optic_rs_err(256, edge='gray')
    assert d != _third_optic_rs_err(256, edge='hard')


# ===========================================================================
# R6 -- the downstream boolean-cast exposure, measured
# ===========================================================================

def _hard_and_grey_masks(N=256, dx=4e-6, D=0.5e-3):
    E = np.ones((N, N), dtype=complex)
    return (np.real(apply_aperture(E, dx, 'circular', {'diameter': D},
                                   edge='hard')),
            np.real(apply_aperture(E, dx, 'circular', {'diameter': D})))


def test_verify_c1r2_the_boolean_cast_moves_a_plotted_radial_rms_curve():
    """R6.  VERIFY-C1 measured the DILATION (+168 px, +1.37 %, on a 12281-px
    disc) and recorded the EXPOSURE as unmeasurable; round 2 repeated that.
    It is measurable: ``_radial_rms_profile`` boolean-casts its ``aperture``
    argument (``plotting.py:1491``), and the curve it returns -- the one
    ``plot_opd_summary``'s second panel plots -- moves by up to 7.27e-02
    relative when the caller's mask comes from today's default instead of
    ``edge='hard'``.

    Bars.  The dilation is an exact integer count (12281 -> 12449 on both
    builds, no BLAS), so its bar is the count itself.  The curve's move is
    bounded below by 1e-2, which is 7.3x under the measured 7.27e-02 and
    three decades above the 1e-5 that a pure round-off difference between the
    two masks could produce on a 16-bin reduction.
    """
    from lumenairy.analysis.plotting import _radial_rms_profile
    hard, grey = _hard_and_grey_masks()
    n_hard = int(np.count_nonzero(hard.astype(bool)))
    n_grey = int(np.count_nonzero(grey.astype(bool)))
    assert (n_hard, n_grey) == (12281, 12449), (n_hard, n_grey)

    N, dx, D = 256, 4e-6, 0.5e-3
    yy, xx = np.mgrid[0:N, 0:N]
    rr = np.sqrt(((xx - (N - 1) / 2.0) * dx) ** 2
                 + ((yy - (N - 1) / 2.0) * dx) ** 2)
    opd = 1e-6 * (0.4 * (rr / (D / 2)) ** 2 - 0.15 * (rr / (D / 2)) ** 4)
    _, rms_h = _radial_rms_profile(opd, dx, dx, hard, n_bins=16)
    _, rms_g = _radial_rms_profile(opd, dx, dx, grey, n_bins=16)
    assert not np.array_equal(rms_h, rms_g)
    with np.errstate(invalid='ignore'):
        rel = np.abs(rms_g - rms_h) / np.maximum(np.abs(rms_h), 1e-30)
    assert float(np.nanmax(rel)) > 1e-2, float(np.nanmax(rel))


def test_verify_c1r2_the_boolean_cast_moves_a_wrapper_merits_integration():
    """R6, second site.  ``optimize/wrapper_merits.py:266`` boolean-casts a
    caller-supplied aperture ARRAY into the mask a merit integrates over.  On
    the same disc, the integrated intensity a merit would read moves by
    7.71e-03 relative, and the boolean cast OVERSHOOTS the correctly weighted
    grey mask -- the thing the grey rim exists to provide -- by 8.91e-03.

    Bar 1e-3: 7.7x below the measured move and two decades above the 1e-5 a
    float64 reduction over 65 536 pixels can produce by round-off alone.
    """
    from lumenairy.optimize.wrapper_merits import (_clear_wrapper_merit_cache,
                                                   _get_wrapper_merit_cache)
    N, dx, D = 256, 4e-6, 0.5e-3
    hard, grey = _hard_and_grey_masks(N, dx, D)
    yy, xx = np.mgrid[0:N, 0:N]
    rr = np.sqrt(((xx - (N - 1) / 2.0) * dx) ** 2
                 + ((yy - (N - 1) / 2.0) * dx) ** 2)
    inten = np.exp(-((rr / (D / 2)) ** 2) * 0.35) ** 2
    _clear_wrapper_merit_cache()
    try:
        m_h = _get_wrapper_merit_cache(N, dx, hard, np.complex128)['mask']
        m_g = _get_wrapper_merit_cache(N, dx, grey, np.complex128)['mask']
        assert int(m_g.sum()) - int(m_h.sum()) == 168, (
            int(m_h.sum()), int(m_g.sum()))
        p_h = float(inten[m_h].sum())
        p_g = float(inten[m_g].sum())
        p_w = float((inten * grey).sum())
        assert abs(p_g - p_h) / abs(p_h) > 1e-3, (p_h, p_g)
        assert abs(p_g - p_w) / abs(p_w) > 1e-3, (p_g, p_w)
    finally:
        _clear_wrapper_merit_cache()


def test_verify_c1r2_the_boolean_cast_moves_the_auto_radial_bin_count():
    """R6, third site.  ``plotting.py:1757`` uses the cast as a PIXEL COUNT
    feeding ``_auto_n_bins``.  On a 48-pixel grid the count goes 657 -> 697
    and the bin count goes 12 -> 13: a discrete, visible change in the panel.

    The decision is the discrete move, so there is no bar to be per-build --
    both readings are integer counts over a mask, identical on both builds.
    The N = 256 case is asserted too, where BOTH saturate the ceiling at 32,
    so the id cannot be satisfied by a site that moves on every grid.
    """
    from lumenairy.analysis.plotting import _auto_n_bins
    dx = 4e-6
    for n_small, want_move in ((48, True), (256, False)):
        E = np.ones((n_small, n_small), dtype=complex)
        d = 0.6 * n_small * dx
        n_h = int(np.count_nonzero(np.real(apply_aperture(
            E, dx, 'circular', {'diameter': d}, edge='hard')).astype(bool)))
        n_g = int(np.count_nonzero(np.real(apply_aperture(
            E, dx, 'circular', {'diameter': d})).astype(bool)))
        assert n_g > n_h, (n_small, n_h, n_g)
        moved = int(_auto_n_bins(n_h, ceiling=32)) != int(
            _auto_n_bins(n_g, ceiling=32))
        assert moved is want_move, (n_small, n_h, n_g)


def test_verify_c1r2_the_wrapper_merit_array_branch_is_out_of_the_librarys_reach():
    """R6's DECISION, measured.  The verifier asked whether the in-library
    call at ``wrapper_merits.py:492`` -- the one caller that forwards
    ``ctx.prescription['aperture_diameter']`` to ``_get_wrapper_merit_cache``
    without ``float()``-ing it -- should WEIGHT by a grey mask instead of
    casting it, on the grounds that a merit overshooting by 8.9e-3 of power is
    a wrong answer.

    It cannot overshoot, because an ndarray cannot get there.
    ``MultiWavelengthMerit.evaluate`` calls
    ``surfaces_from_prescription(ctx.prescription)`` at the TOP of its
    per-wavelength loop -- before ``:492``, and outside the ``try`` that wraps
    ``system_abcd`` -- and that call runs ``validate_prescription``, which
    refuses a non-numeric ``aperture_diameter`` with "must be a number".  So
    the array branch at ``:266`` is reachable only by calling the PRIVATE
    ``_get_wrapper_merit_cache`` directly with an array, which is a caller's
    choice and not the library's, and R6 is closed with a Migration row and a
    CHANGELOG sentence rather than with a code change that would redefine a
    documented boolean mask for every other consumer.

    The id pins that premise from both sides, so the decision is revisited
    automatically if it ever stops holding:
      * the SCALAR prescription passes ``surfaces_from_prescription`` (else
        the refusal below would prove nothing about the array);
      * the same prescription with a grey MASK array in
        ``aperture_diameter`` raises, and the message names the key;
      * and the cost that would be paid if it did get through is measured
        here, not asserted: the boolean cast overshoots the weighted grey
        mask by more than 1e-3 of integrated power (measured 9.88e-03 on the
        fixture below, 9.9x above the bar and two decades above the ~1e-5 a
        float64 reduction over 65 536 pixels can produce by round-off).
    Both builds read every integer count and every digit identically.
    """
    import lumenairy as la
    from lumenairy.optimize import core as _core
    from lumenairy.optimize.wrapper_merits import (_clear_wrapper_merit_cache,
                                                   _get_wrapper_merit_cache)

    N, dx, D = 256, 4e-6, 0.5e-3
    hard, grey = _hard_and_grey_masks(N, dx, D)

    def _rx(ap):
        rx = la.make_singlet(R1=0.032, R2=-0.075, d=4e-4, glass='N-BK7',
                             aperture=D)
        rx['aperture_diameter'] = ap
        return rx

    assert len(_core.surfaces_from_prescription(_rx(float(D)))) == 2
    with pytest.raises(ValueError, match='aperture_diameter'):
        _core.surfaces_from_prescription(_rx(grey))

    yy, xx = np.mgrid[0:N, 0:N]
    rr = np.sqrt(((xx - (N - 1) / 2.0) * dx) ** 2
                 + ((yy - (N - 1) / 2.0) * dx) ** 2)
    inten = np.exp(-((rr / (D / 2)) ** 2) * 0.35) ** 2
    _clear_wrapper_merit_cache()
    try:
        m_g = _get_wrapper_merit_cache(N, dx, grey, np.complex128)['mask']
        assert m_g.dtype == np.bool_, m_g.dtype
        p_cast = float(inten[m_g].sum())
        p_weight = float((inten * grey).sum())
        assert (p_cast - p_weight) / p_weight > 1e-3, (p_cast, p_weight)
    finally:
        _clear_wrapper_merit_cache()
    assert int(np.count_nonzero(grey.astype(bool))
               - np.count_nonzero(hard.astype(bool))) == 168


# ===========================================================================
# R7 -- the peak-memory comment
# ===========================================================================

def test_verify_c1r2_the_grey_branch_costs_one_grid_and_is_flat_in_n_sub():
    """R7.  ``apply_aperture``'s grey branch says "measured 6.0 float64 grids
    at N = 2048, against 5.0 for the hard edge".  An allocator trace -- which
    neither VERIFY-C1 nor round 2 ran -- reads, in steady state (the FIRST
    call in a process carries ~1.37 grids of one-off allocation, so whichever
    arm is measured first reads high):

        Windows py3.14 / numpy 2.4.4  N = 2048  hard 5.0011  grey 6.0011
        WSL     py3.12 / numpy 2.4.6  N = 2048  hard 4.0011  grey 5.1262

    so the two ABSOLUTE figures in the comment are a Windows reading, and the
    build-free claims are the two this id asserts: the grey branch costs about
    ONE extra float64 grid at peak (measured 1.000 on Windows and 1.125-1.134
    on WSL; bar 0.5..2.0, which clears the cross-build spread of 0.13 by 3.8x
    on the low side and 6.6x on the high side), and that peak does NOT grow
    with ``n_sub`` (measured spread over n_sub = 2/4/8/16: 0.0000 grids on
    Windows, 0.008 on WSL; bar 0.05, 6x above the larger reading and 20x below
    the 1.0 that one extra retained sub-mask would cost).

    Run at N = 512 rather than 2048 to keep the id under a second; the deltas
    are the same to three digits.
    """
    N, dx = 512, 1e-6
    grid = N * N * 8.0
    E = np.ones((N, N), dtype=np.complex128)

    def peak(**kw):
        gc.collect()
        tracemalloc.start()
        base, _ = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        out = apply_aperture(E, dx, 'circular', {'diameter': 0.5 * N * dx},
                             **kw)
        _, pk = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        del out
        gc.collect()
        return (pk - base) / grid

    for kw in ({'edge': 'hard'}, {'edge': 'gray'}):    # warm-up
        peak(**kw)
    p_hard = peak(edge='hard')
    greys = [peak(edge='gray', edge_samples=n) for n in (2, 4, 8, 16)]
    assert 0.5 <= (greys[1] - p_hard) <= 2.0, (p_hard, greys)
    assert (max(greys) - min(greys)) < 0.05, greys


# ===========================================================================
# R8 -- the Migration table's codegen recipe now covers BOTH styles
# ===========================================================================

_CODEGEN_RX = {
    'elements': [
        {'surf_num': 1, 'element_type': 'surface', 'radius': np.inf,
         'glass_after': 'air', 'is_stop': True, 'semi_diameter': 0.9e-3},
        {'surf_num': 2, 'element_type': 'surface', 'radius': 0.032,
         'glass_after': 'N-SF11', 'semi_diameter': 1.6e-3},
        {'surf_num': 3, 'element_type': 'surface', 'radius': -0.075,
         'glass_after': 'air', 'semi_diameter': 1.6e-3},
    ],
    'all_thicknesses': [1.5e-3, 2.5e-3, 18e-3],
    'aperture_diameter': 1.8e-3,
}

#: The one rim-bearing line each style emits for a STOP surface, anchored as
#: an EXACT string so the ids below fail on the anchor (loudly) rather than on
#: a silent no-op edit if either emission ever changes.
_CODEGEN_UNROLLED_CALL = ('la.apply_aperture(E, dx, shape="circular", '
                          'params={"diameter": 1.79999999999999995e-03})')
_CODEGEN_SYSTEM_ELEM = ("{'type': 'aperture', 'shape': 'circular', "
                        "'params': {'diameter': 1.79999999999999995e-03}}")


def _codegen_script(style):
    from lumenairy.io.codegen import generate_simulation_script
    return generate_simulation_script(_CODEGEN_RX, wavelength=633e-9,
                                      N=128, dx=40e-6, style=style)


def test_verify_c1r2_the_codegen_way_back_is_style_specific():
    """R8.  ``Migration-Guide.md``'s 5.49.0 table used to give ONE codegen
    recipe -- "edit the generated ``la.apply_aperture(...)`` call, or re-pin"
    -- for "a script emitted by ``lumenairy.io.codegen`` for a STOP surface".

    ``generate_simulation_script`` has TWO public styles and they need
    different recipes.  ``'unrolled'`` (the default) emits exactly that call,
    so the old recipe applies to it.  ``'system'`` emits an element list
    instead -- ``{'type': 'aperture', 'shape': 'circular', 'params': {...}}``
    with no ``'edge'`` key and **no ``la.apply_aperture(`` anywhere in the
    file** -- so the old recipe named something the reader cannot find; its
    way back is ``'edge': 'hard'`` inside the emitted element dict.  The
    Migration table now carries a row for each.

    The id asserts the emitted TEXT of both styles (so it stays honest if
    either generator changes) and that the guide names both recipes.
    """
    import pathlib

    unrolled = _codegen_script('unrolled')
    system = _codegen_script('system')
    assert _CODEGEN_UNROLLED_CALL in unrolled
    assert "'edge'" not in unrolled and 'edge=' not in unrolled
    assert _CODEGEN_SYSTEM_ELEM in system
    assert 'la.apply_aperture(' not in system
    assert "'edge'" not in system

    guide = (pathlib.Path(__file__).resolve().parents[2]
             / 'Migration-Guide.md').read_text(encoding='utf-8',
                                               errors='replace')
    body = guide.split('## 5.49.0')[1].split(chr(10) + '## ')[0]
    assert "style='unrolled'" in body, 'the unrolled codegen row is gone'
    assert "style='system'" in body, 'the system codegen row is gone'
    assert "`'edge': 'hard'` to the generated" in body, (
        'the system-style way back no longer names the element-dict key')


def test_verify_c1r2_both_codegen_styles_reproduce_the_hard_rim_when_executed():
    """R8's other half, and the thing VERIFY-C1-ROUND2 recorded as unmeasured
    ("a generated script executed end to end"): the two recipes are checked by
    RUNNING the generated scripts, not by reading their text.

    For each style the script is generated twice -- once untouched, once with
    that style's own way back applied exactly as the Migration table words it
    (``edge="hard"`` appended to the ``la.apply_aperture(...)`` call;
    ``'edge': 'hard'`` added to the emitted element dict) -- written to a
    temporary directory and executed in a SUBPROCESS with
    ``stdin=subprocess.DEVNULL``.  The unrolled template's own ``__main__``
    driver ends in ``plt.show()``, so it is replaced by a digest driver rather
    than run.

    The decisions, all on exact bytes of the exit-plane field, so there is no
    bar here to be per-build:

      * each style's way-back digest DIFFERS from its own untouched digest --
        the recipe actually bites in a generated script, which is what the
        table promises;
      * the two styles agree with each other on BOTH arms -- one default
        answer and one pre-5.49 answer, two spellings.  Without this second
        claim the first would be satisfied by a system-style recipe that
        changed the answer to something else entirely.

    (That those way-back bytes ARE the pre-5.49 library's is the
    archive-to-archive half, which cannot be asserted from inside one tree; it
    is measured by
    ``validation/probe_wpc1_round3/probe_r8_codegen_execute.py`` against a
    ``git archive 49ddf4bd`` extraction, on both builds.)

    Runtime ~25 s: four subprocess runs of a 128 x 128 three-element chain.
    """
    import hashlib  # noqa: F401 -- the generated driver imports its own
    import os
    import subprocess
    import sys

    import lumenairy

    root = os.path.dirname(os.path.dirname(os.path.abspath(
        lumenairy.__file__)))
    tmp = tempfile.mkdtemp()
    env = dict(os.environ)
    env['PYTHONPATH'] = root
    for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        env[var] = '1'

    DRIVER_UNROLLED = (
        chr(10) + 'import hashlib as _h' + chr(10)
        + '_E, _planes = run_simulation(verbose=False)' + chr(10)
        + "print('DIGEST ' + _h.blake2b(_E.tobytes(), "
          "digest_size=8).hexdigest())" + chr(10))
    DRIVER_SYSTEM = (
        chr(10) + 'import hashlib as _h' + chr(10)
        + "print('DIGEST ' + _h.blake2b(E_out.tobytes(), "
          "digest_size=8).hexdigest())" + chr(10))

    def _digest(style, way_back):
        src = _codegen_script(style)
        if style == 'unrolled':
            assert src.count(_CODEGEN_UNROLLED_CALL) == 1, 'unrolled anchor'
            if way_back:
                src = src.replace(_CODEGEN_UNROLLED_CALL,
                                  _CODEGEN_UNROLLED_CALL[:-1]
                                  + ', edge="hard")')
            head = "if __name__ == '__main__':"
            assert head in src, 'the unrolled template lost its driver'
            src = src[:src.index(head)] + DRIVER_UNROLLED
        else:
            assert src.count(_CODEGEN_SYSTEM_ELEM) == 1, 'system anchor'
            if way_back:
                src = src.replace(_CODEGEN_SYSTEM_ELEM,
                                  _CODEGEN_SYSTEM_ELEM[:-1]
                                  + ", 'edge': 'hard'}")
            src += DRIVER_SYSTEM
        tag = '{0}_{1}'.format(style, 'wayback' if way_back else 'default')
        path = os.path.join(tmp, 'gen_{0}.py'.format(tag))
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write(src)
        run = subprocess.run([sys.executable, path],
                             stdin=subprocess.DEVNULL, capture_output=True,
                             text=True, env=env, cwd=tmp, timeout=600)
        assert run.returncode == 0, (tag, run.returncode, run.stdout[-1500:],
                                     run.stderr[-1500:])
        hits = [ln for ln in run.stdout.splitlines()
                if ln.startswith('DIGEST ')]
        assert len(hits) == 1, (tag, run.stdout[-1500:])
        got = hits[0].split()[1]
        assert len(got) == 16, (tag, got)
        return got

    d = {}
    try:
        for style in ('unrolled', 'system'):
            for way_back in (False, True):
                d[(style, way_back)] = _digest(style, way_back)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    assert d[('unrolled', True)] != d[('unrolled', False)], d
    assert d[('system', True)] != d[('system', False)], d
    assert d[('unrolled', False)] == d[('system', False)], d
    assert d[('unrolled', True)] == d[('system', True)], d


# ===========================================================================
# D4 -- the claims round 2 makes, re-decided on a TWO-STOP prescription
# ===========================================================================

_TWO_STOP_RX = {
    'elements': [
        {'surf_num': 1, 'element_type': 'surface', 'radius': np.inf,
         'glass_after': 'air', 'is_stop': True, 'semi_diameter': 0.9e-3},
        {'surf_num': 2, 'element_type': 'surface', 'radius': 0.032,
         'glass_after': 'N-SF11', 'semi_diameter': 1.6e-3},
        {'surf_num': 3, 'element_type': 'surface', 'radius': -0.075,
         'glass_after': 'air', 'semi_diameter': 1.6e-3},
        {'surf_num': 4, 'element_type': 'surface', 'radius': np.inf,
         'glass_after': 'air', 'is_stop': True, 'semi_diameter': 0.6e-3},
    ],
    'all_thicknesses': [1.5e-3, 2.5e-3, 6e-3, 12e-3],
    'aperture_diameter': 1.8e-3,
}

_NO_STOP_RX = {
    'elements': [
        {'surf_num': 1, 'element_type': 'surface', 'radius': 0.032,
         'glass_after': 'N-SF11', 'semi_diameter': 1.6e-3},
        {'surf_num': 2, 'element_type': 'surface', 'radius': -0.075,
         'glass_after': 'air', 'semi_diameter': 0.4e-3},
    ],
    'all_thicknesses': [2.5e-3, 18e-3],
    'aperture_diameter': 1.8e-3,
}


def test_verify_c1r2_evaluates_rim_reaches_every_stop_not_only_the_first():
    """D4 asserts the keyword is stamped onto "every ``'aperture'`` element
    ``_prescription_to_elements`` emits for an ``is_stop=True`` surface", but
    round 2 measured only a ONE-stop prescription, where "every" and "the
    first" cannot be told apart.

    On a prescription with TWO stop surfaces both emitted elements carry the
    stamp, and the way back is bit-exact on the field: ``aperture_edge='hard'``
    and ``aperture_edge_samples=1`` give the same bytes, and the default gives
    different bytes.  (The archive-to-archive half -- that those bytes are the
    parent commit's -- is in
    ``validation/probe_verify_c1_round2/d4_evalv2_PRE49_*.json``; it cannot be
    asserted from inside one tree.)
    """
    import warnings

    import lumenairy as la
    from lumenairy.propagators.system import _prescription_to_elements

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        els = _prescription_to_elements(_TWO_STOP_RX, aperture_edge='hard',
                                        aperture_edge_samples=7)
    aps = [e for e in els if e.get('type') == 'aperture']
    assert len(aps) == 2, [e.get('type') for e in els]
    assert all(e.get('edge') == 'hard' and e.get('edge_samples') == 7
               for e in aps), aps

    def field(**kw):
        src = la.Source.gaussian(N=128, dx=40e-6, wavelength=633e-9, w0=0.9e-3)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return np.asarray(la.evaluate(_TWO_STOP_RX, src, **kw).field
                              ).tobytes()

    assert field(aperture_edge='hard') == field(aperture_edge_samples=1)
    assert field(aperture_edge='hard') != field()
    assert field() == field(aperture_edge='gray', aperture_edge_samples=4)


def test_verify_c1r2_a_prescription_with_no_stop_is_byte_identical():
    """Round 2 says a no-STOP prescription "is byte-identical with and without
    the keyword, which is the docstring's own claim checked rather than
    asserted".  Asserted here, on every spelling of the keyword, plus the
    premise that no aperture element is emitted at all -- otherwise the
    identity would be proving something else.
    """
    import warnings

    import lumenairy as la
    from lumenairy.propagators.system import _prescription_to_elements

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        els = _prescription_to_elements(_NO_STOP_RX)
    assert not [e for e in els if e.get('type') == 'aperture'], els

    def field(**kw):
        src = la.Source.gaussian(N=128, dx=40e-6, wavelength=633e-9, w0=0.9e-3)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return np.asarray(la.evaluate(_NO_STOP_RX, src, **kw).field
                              ).tobytes()

    base = field()
    for kw in ({'aperture_edge': 'hard'}, {'aperture_edge': 'gray'},
               {'aperture_edge_samples': 1}, {'aperture_edge_samples': 16},
               {'aperture_edge': None, 'aperture_edge_samples': None}):
        assert field(**kw) == base, kw


def test_verify_c1r2_evaluates_rim_is_refused_before_the_decomposition_runs():
    """D4 says both keywords are "validated once, before the decomposition
    runs".  Measured rather than read: on a prescription whose decomposition
    emits a ``UserWarning`` (a DOE placeholder surface), a bad rim raises with
    ZERO warnings recorded, while the same prescription with a GOOD rim
    records exactly one -- so the guard demonstrably runs first, and the id is
    not satisfied by a prescription that never warns.
    """
    import warnings

    from lumenairy.propagators.system import _prescription_to_elements
    rx = {
        'elements': [
            {'surf_num': 1, 'element_type': 'surface', 'radius': np.inf,
             'glass_after': 'air', 'is_stop': True, 'semi_diameter': 0.9e-3},
            {'surf_num': 2, 'element_type': 'surface', 'radius': np.inf,
             'glass_after': 'air', 'semi_diameter': 1.6e-3,
             'aspheric_coeffs': [1e-9, 0.0, 0.0]},
        ],
        'all_thicknesses': [1.5e-3, 18e-3],
        'aperture_diameter': 1.8e-3,
    }
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        with pytest.raises(ValueError, match='apply_aperture'):
            _prescription_to_elements(rx, aperture_edge='soft')
        assert len(rec) == 0, [str(w.message) for w in rec]
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        _prescription_to_elements(rx, aperture_edge='hard')
        assert len(rec) == 1, [str(w.message) for w in rec]


# ===========================================================================
# D1 -- the shape of the fix, asserted at source level
# ===========================================================================

def test_verify_c1r2_there_is_exactly_one_rim_guard_with_three_call_sites():
    """D1's claim is "One function, three call sites; not a second copy of the
    guards".  Asserted against the package source, so a future copy-paste is
    caught: exactly one ``def _validate_edge_kwargs``, exactly three calls to
    it (``apply_aperture``'s body, ``_prescription_to_elements`` and
    ``_aperture_edge_kwargs``), and neither of the two refusal MESSAGES
    appears anywhere but in that one function.
    """
    import pathlib

    root = pathlib.Path(elem_mod.__file__).parent.parent
    defs, calls, msg_files = [], [], set()
    for p in sorted(root.rglob('*.py')):
        txt = p.read_text(encoding='utf-8', errors='replace')
        rel = p.relative_to(root.parent).as_posix()
        if ('edge_samples must be a positive integer' in txt
                or "edge must be 'hard'" in txt):
            msg_files.add(rel)
        for i, line in enumerate(txt.splitlines(), 1):
            s = line.strip()
            if s.startswith('def _validate_edge_kwargs'):
                defs.append('{0}:{1}'.format(rel, i))
            elif ('_validate_edge_kwargs(' in line
                  and not s.startswith(('#', ':', '*'))
                  and 'def _validate_edge_kwargs' not in line):
                calls.append('{0}:{1}'.format(rel, i))
    assert len(defs) == 1, defs
    assert len(calls) == 3, calls
    assert msg_files == {'lumenairy/elements/elements.py'}, msg_files
