"""WP-A24 -- the d6 on-axis EE2 ratio: what actually moved it, and the floor
the restated envelope is derived against.

``test_niche_d6_exact_tilted_leg.py::test_decentred_carrier_decentre_penalty_envelope``
read ``r_on = 0.969787`` against a one-sided ``> 0.97`` written from a single
2026-07-29 measurement of 0.9966.  A read-only bisect (``git archive <rev>
lumenairy`` into a scratch tree, the CURRENT test file run against each) put
the crossing at WP-A6's commit ``a18ab074`` -- "the paraxial focus readout's
stop grid is now sized from the BEAM" (finding C1).  Re-measured here, that
attribution is **wrong in both directions**:

* ``a18ab074^`` reads **0.969787**, bit for bit the same as ``a18ab074`` and
  as HEAD, so WP-A6 did not move this number; and C1's resolver is not even on
  this fixture's code path, because ``final_leg='exact'`` returns from
  :func:`~lumenairy.propagate_traced_carrier_chain` through
  ``carrier_referenced_exact_focus_readout`` without ever entering
  ``carrier_referenced_focus_readout``, the only caller of
  ``_default_focus_standoff``.  ``test_the_exact_final_leg_never_enters_the_
  paraxial_focus_standoff_resolver`` below pins that, with its falsifier.
* the step is in two pieces and neither is WP-A6's: **-0.0251** at
  ``4e8ea247`` ("feat(lens): banded ray-density + inverse-characteristic
  evaluator", 0.996575 -> 0.971526) and **-0.0017** at ``f602b72c``
  ("fix(raytrace): WP-A1 ... entrance eikonal, exact conic intersection",
  0.971526 -> 0.969787, bit-stable through every commit since).  The big piece
  reproduces at HEAD to six digits through the evaluator's own documented
  switch: ``traced_kwargs={'inverse_map': False}`` reads 0.996575 on axis and
  0.971078 decentred, against 0.969787 / 0.985518 with it on.  So the two
  paths' CROSSOVER is the evaluator's, not the decentre's.

Everything above is a measurement against the d6 file's inline oracle (exact
conic raytrace + Rayleigh-Sommerfeld surface integral), which shares no grid,
carrier or propagator with the chain.  The third test here measures that
oracle's OWN floor on the ratio, which is what the restated envelope in the d6
file is sized against.
"""
from __future__ import annotations

import os
import pathlib
import sys
import warnings

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

_REPO = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_REPO / 'tests' / 'unit') not in sys.path:
    sys.path.insert(0, str(_REPO / 'tests' / 'unit'))

import numpy as np                                        # noqa: E402
import pytest                                             # noqa: E402

import lumenairy as la                                    # noqa: E402
import lumenairy.propagators.carrier as _car              # noqa: E402

import test_niche_d6_exact_tilted_leg as _d6              # noqa: E402


# ===========================================================================
# 1.  The shipped warning quotes the calibration that was MEASURED, and the
#     ordering it states is the measured one.
# ===========================================================================
def _decentre_warning() -> str:
    """The ``decentre_fit_frac`` warning, raised directly (no chain run)."""
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter('always')
        _car._check_decentred_fit(
            6.0e-4, 6.0e-4, 0.0, 'the EXACT final leg (fine retrace)',
            'warn', 0.5)
    hits = [str(w.message) for w in wl if 'decentre_fit_frac' in str(w.message)]
    assert hits, 'the decentred-fit guard did not fire at 1.0 beam radii'
    return hits[0]


def test_the_shipped_decentre_warning_quotes_the_re_measured_calibration():
    """FAIL-BEFORE / PASS-AFTER on the shipped message.

    Until 2026-09-12 this warning told users "MEASURED ... 0.00 w -> 0.997;
    ... 1.00 w -> 0.983", a 2026-07-29 reading of the K=-n^2 stand-in.  The
    same six points re-measured on the same stand-in and the same oracle read
    0.970 / 1.010 / 1.008 / 1.002 / 0.986 / 0.903 -- the on-axis and 1.0 w
    rows have CROSSED OVER, so the message stated the opposite of the measured
    ordering to every user whose fan trips the guard.  The two superseded
    values are asserted ABSENT, not just the new ones present, because a
    message that carries both is the failure this test exists to catch."""
    msg = _decentre_warning()
    for token in ('MEASURED 2026-09-12', '0.00 w -> 0.970', '0.25 w -> 1.010',
                  '0.50 w -> 1.008', '0.75 w -> 1.002', '1.00 w -> 0.986',
                  '1.50 w -> 0.903'):
        assert token in msg, f"{token!r} missing from the guard message"
    for stale in ('0.00 w -> 0.997', '1.00 w -> 0.983', '0.75 w -> 0.977',
                  '1.50 w -> 0.923'):
        assert stale not in msg, (
            f"the superseded 2026-07-29 reading {stale!r} is still quoted")


def test_the_warning_names_the_switch_that_sets_the_ordering():
    """The ordering between the on-axis and decentred rows is the terminal
    fine retrace's v5.35 inverse-characteristic evaluator, not the decentre --
    measured 0.9966 / 0.9711 with ``inverse_map=False`` against 0.9698 /
    0.9855 with it on.  A user who reads the table needs the switch name, or
    the caveat is unactionable."""
    msg = _decentre_warning()
    for token in ("traced_kwargs={'inverse_map': False}", 'ORDERING',
                  'inverse-characteristic evaluator', '0.997 at 0.00 w',
                  '0.971 at 1.00 w'):
        assert token in msg, f"{token!r} missing from the guard message"
    # and the guard still says the thing it exists to say
    for token in ('LOWER BOUND', 'decentre-INVARIANT', 'on_decentred_fit'):
        assert token in msg


# ===========================================================================
# 2.  C1 is not on the exact leg's path.  This is the attribution pin.
# ===========================================================================
def test_the_exact_final_leg_never_enters_the_paraxial_focus_standoff_resolver():
    """``final_leg='exact'`` must not reach ``_default_focus_standoff`` (nor
    the beam-referenced term WP-A6's C1 added to it).

    Poisoning both and running the d6 chain is a stronger statement than
    reading the call graph, and it is the statement that keeps the d6 fixture
    attributable: if a future change routes the exact leg through the paraxial
    focus readout, its EE2 ratio starts moving with a resolver that nothing in
    that fixture's docstring mentions.  The falsifier is the next test, which
    shows the poison DOES fire on the paraxial leg -- so a pass here is
    evidence about the routing, not about the patch being ineffective."""
    _d6._ram_guard()

    def _poison(*_a, **_k):
        raise AssertionError(
            'the paraxial focus-standoff resolver ran on the EXACT final leg')

    _saved = (_car._default_focus_standoff, _car._beam_containment_standoff)
    try:
        _car._default_focus_standoff = _poison
        _car._beam_containment_standoff = _poison
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res, _ = _d6._run_chain(
                la.TiltedCarrier(np.inf, 0.0, 0.0, 0.0, 0.0),
                final_leg='exact', centre_out=(0.0, 0.0))
    finally:
        _car._default_focus_standoff, _car._beam_containment_standoff = _saved
    F = np.asarray(res.field)
    assert F.shape == (_d6._NOUT, _d6._NOUT) and np.isfinite(F).all()
    assert float(np.abs(F).max()) > 0.0


def test_the_paraxial_final_leg_does_enter_it():
    """The falsifier for the test above: the same poison, on the same fixture,
    with ``final_leg='paraxial'`` -- which is the route
    ``carrier_referenced_focus_readout`` owns, so the resolver MUST run."""
    _d6._ram_guard()
    seen = []
    _real = _car._default_focus_standoff

    def _count(*a, **k):
        seen.append(1)
        return _real(*a, **k)

    try:
        _car._default_focus_standoff = _count
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _d6._run_chain(la.TiltedCarrier(np.inf, 0.0, 0.0, 0.0, 0.0),
                           final_leg='paraxial', centre_out=(0.0, 0.0))
    finally:
        _car._default_focus_standoff = _real
    assert seen, ('the paraxial final leg did not reach '
                  '_default_focus_standoff -- the poison above proves nothing')


# ===========================================================================
# 3.  The oracle's own floor -- the number the restated envelope is sized on.
# ===========================================================================
def test_the_inline_oracle_s_ee2_floor_is_far_under_the_restated_envelope():
    """The d6 oracle has exactly two free parameters: the quadrature pitch
    (``_N_PUPIL``, whose convergence that file already pins) and the pupil
    patch half-width ``_PUPIL_HALF``, which is what actually APERTURES it --
    the prescription's own circular stop (0.5 * ``_APER`` = 2.8333 w) only
    bites once the square patch reaches it.  The shipped patch is 2.2 w, so
    the oracle stops the beam 1.29x harder than the lens does, and nothing
    before this measured what that is worth.

    Measured 2026-09-12, at the PITCH held fixed so only the extent moves:
    EE(2 um) 0.78428812 at 2.2 w against 0.78482177 at 3.2 w (past the stop,
    where the answer stops moving) -- a relative move of **6.80e-04** on this
    48-point window and 6.42e-04 on the d6 file's own 96-point one.  The bar
    is 5e-03: 7.4x over the measured floor, and 44x UNDER the 0.0302 shortfall
    the restated on-axis envelope in the d6 file is written around, which is
    the whole point -- the oracle is not what moved.

    NOT asserted, but measured and recorded because a reader will assume it:
    the oracle's ring-binned FWHM is NOT invariant to the same sweep.  It
    reads 3.15 um at 2.2 w and 2.85 um at 3.2 w, two bins of the 0.15 um
    readout pitch (9.5 %), the bigger aperture giving the narrower core.  The
    FWHM ratios asserted in the d6 file are therefore quoted against the
    SHIPPED 2.2 w patch and would not survive widening it silently."""
    _d6._ram_guard()
    n_out = 48
    u = (np.arange(n_out) - n_out / 2) * _d6._DXO
    XS, YS = np.meshgrid(u, u, indexing='xy')
    ee2, fwhm = [], []
    _saved = _d6._PUPIL_HALF
    try:
        # (n_pupil, half/w) chosen to hold the pupil PITCH at the shipped
        # 0.03143 w, so the only thing that varies is the extent.
        for n_pupil, half_w in ((141, 2.2), (205, 3.2)):
            _d6._PUPIL_HALF = float(half_w) * _d6._W
            O = _d6._oracle_field(XS.ravel(), YS.ravel(), x0=0.0,
                                  n_pupil=n_pupil).reshape(n_out, n_out)
            m = _d6._metrics(O, radii=(2.0, 4.0))
            ee2.append(m['ee'][2.0])
            fwhm.append(m['fwhm'])
    finally:
        _d6._PUPIL_HALF = _saved
    rel = abs(ee2[1] - ee2[0]) / ee2[0]
    assert rel < 5e-3, (
        f"the oracle's own EE2 moved {rel:.3e} between a 2.2 w and a 3.2 w "
        f"pupil patch ({ee2[0]:.8f} -> {ee2[1]:.8f}); measured 6.80e-04 on "
        f"2026-09-12.  The d6 file's on-axis envelope is derived against that "
        f"floor, so it has to be re-derived if this grows.")
    assert all(np.isfinite(fwhm)) and min(fwhm) > 0.0


@pytest.mark.parametrize('half_w', [2.2, 2.8333])
def test_the_oracle_pupil_patch_keeps_essentially_all_the_launched_power(
        half_w):
    """Why the EE2 floor above is so small even though the aperture changes:
    the patch already holds the beam.  A square patch of half-width ``h``
    centred on a Gaussian of amplitude radius ``w``, intersected with the
    prescription's circular stop, retains ``erf(sqrt(2) h/w)^2`` of the power
    when the circle does not bite -- 0.999981 at 2.2 w.  So the aperture
    difference is a 1.9e-05 POWER effect whose EE2 signature is 6.8e-04, i.e.
    the skirt it moves is 36x its own weight but still three decades under the
    0.0302 the restated envelope is about."""
    from math import erf                       # stdlib: no scipy dependency
    w = _d6._W
    h = np.linspace(-half_w * w, half_w * w, 401)
    dA = float(h[1] - h[0]) ** 2
    HX, HY = np.meshgrid(h, h, indexing='ij')
    inside = (HX ** 2 + HY ** 2) <= (0.5 * _d6._APER) ** 2
    kept = float((np.exp(-2.0 * (HX ** 2 + HY ** 2) / (w * w)) * inside).sum()
                 * dA / (0.5 * np.pi * w * w))
    closed = float(erf(np.sqrt(2.0) * half_w)) ** 2
    assert kept > 0.99997, f"patch at {half_w} w keeps only {kept:.6f}"
    if half_w * w <= 0.5 * _d6._APER:
        # the circle does not bite: the closed form is exact
        assert abs(kept - closed) < 2e-5, (kept, closed)
