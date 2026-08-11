"""Acceptance battery for :mod:`lumenairy.propagators.carrier_field`.

WHAT IS BEING PINNED, AND AGAINST WHAT
--------------------------------------
The two primitives here productize the arm-B code path of
``docs/audits/PROBE_SUM_AT_APERTURE_2026_08_11.md``.  That probe already
measured the physics against the shipped per-order path on the real design-121
fan; what these tests pin is the ALGEBRA the probe took for granted, at
tolerances the probe could not reach because every one of its numbers included
an exact final leg:

* the probe's tightest field-level number is **2.8e-05** relative L2 (its
  null control), and it is dominated by ONE band-limited resample plus a leg;
* with no leg in the loop, an A -> B -> A round trip on a smooth field is an
  ALGEBRAIC identity, and the bar here is **1e-12** -- seven decades tighter.

The design-121 reproduction itself is NOT a unit test: it needs the probe's
1.07 GB cached aperture fields.  It lives in
``validation/repro_traced_carrier_121/sumap_newapi_null_121.py`` and its
numbers are recorded in ``docs/audits/BUILD_CARRIER_FIELD_2026_08_11.md``.
What IS a unit test is the geometry census that sizes the common grid --
``test_nyquist_reproduces_the_design_121_census`` re-derives the probe's
2.02 um binding pitch from nothing but the two carriers.

THREE RULES, NOT ONE
--------------------
1. **Envelope rule.**  Round-trip and OPL residuals are asserted against
   BARS, and where the bar is set by float64 rather than by the code the bar
   is DERIVED (``k0 * eps * |R|``) and the derivation is asserted too.
2. **Fingerprint rule.**  ``test_round_trip_floor_is_the_eikonal_cancellation``
   pins the SHAPE of the residual (linear in ``|R|``), because a residual that
   is merely small can go wrong silently while one whose scaling law is
   pinned cannot.
3. **Comparative rule.**  ``test_nyquist_guard_*`` shows the refusal is
   load-bearing: the same call with the guard disabled returns a populated,
   credible-looking field that is WRONG by O(1), while the adequate grid is
   right to 1e-9.  That is the fail-before this guard exists for.
"""

import json
import math
import warnings

import numpy as np
import pytest

from lumenairy.propagators.carrier_field import (
    CARRIER_FIELD_SCHEMA,
    CarrierField,
    CarrierSpec,
    FieldGrid,
    aggregate,
    carrier_difference_nyquist,
    load_carrier_field_zarr,
    re_reference,
    save_carrier_field_zarr,
)

LAM = 1.31e-6
EPS = float(np.finfo(np.float64).eps)


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------
def _gauss_field(N, dx, w, R, *, centre=(0.0, 0.0), tilt=(0.0, 0.0),
                 piston=0.0, origin=(0.0, 0.0), prov=None):
    """A smooth, well-contained Gaussian envelope on a centred grid."""
    g = FieldGrid((N, N), dx, origin=origin)
    x, y = g.axes()
    env = np.exp(-(((x - centre[0])[None, :] ** 2
                    + (y - centre[1])[:, None] ** 2) / w ** 2)
                 ).astype(np.complex128)
    car = CarrierSpec(R=R, centre=centre, tilt=tilt, piston=piston)
    return CarrierField(env, g, car, LAM, prov if prov is not None else {})


def _rel_l2(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b))
                 / np.linalg.norm(np.asarray(b)))


def _eikonal_floor_rad(R):
    """The residual a round trip through the LIBRARY's own
    ``_exact_sphere_eikonal`` cannot beat.

    That routine evaluates ``sqrt(r^2 + R^2) - |R|``, which for ``r << |R|``
    is a catastrophic cancellation: the result inherits the ulp of
    ``sqrt(r^2+R^2)``, i.e. an ABSOLUTE error of ``eps * |R|`` metres,
    independent of ``r``.  In phase that is ``k0 * eps * |R|`` radians.
    MEASURED and attributed in BUILD_CARRIER_FIELD_2026_08_11 S5 (replacing
    the expression with the algebraically identical, cancellation-free
    ``r^2 / (sqrt(r^2+R^2) + |R|)`` collapses the residual to the resample's
    own 3.7e-13 at every radius tested)."""
    return 2.0 * math.pi * EPS * abs(float(R)) / LAM


# ---------------------------------------------------------------------------
# CarrierSpec / FieldGrid contracts
# ---------------------------------------------------------------------------
def test_carrier_spec_rejects_non_propagating_tilt():
    with pytest.raises(ValueError, match='DIRECTION COSINES'):
        CarrierSpec(R=-0.05, tilt=(0.8, 0.7))


def test_carrier_spec_rejects_non_finite_piston():
    with pytest.raises(ValueError, match='piston'):
        CarrierSpec(R=-0.05, piston=float('inf'))


def test_collimated_carrier_is_a_plane_wave_not_a_degenerate_sphere():
    """``R = +/-inf`` must be the analytic limit (a pure ramp), not NaN.

    The library's own ``_exact_sphere_eikonal`` carries this special case for
    a named reason -- an all-NaN eikonal silently disables every downstream
    guard that reduces it -- so the wrapper has to carry it too."""
    c = CarrierSpec(R=float('-inf'), tilt=(1e-3, -2e-3), centre=(1e-5, 0.0))
    v = c.eikonal_at(np.array([1e-4, -3e-4]), np.array([2e-4, 5e-5]))
    assert np.all(np.isfinite(v))
    expect = 1e-3 * (np.array([1e-4, -3e-4]) - 1e-5) \
        + (-2e-3) * np.array([2e-4, 5e-5])
    assert np.allclose(v, expect, rtol=0, atol=1e-18)
    g = FieldGrid((32, 32), 1e-6)
    assert np.all(np.isfinite(c.phasor_on(g, LAM)))


def test_field_grid_axes_are_absolute():
    g = FieldGrid((8, 8), 2e-6, origin=(1e-3, -5e-4))
    x, y = g.axes()
    assert x[4] == pytest.approx(1e-3, abs=0.0)
    assert y[4] == pytest.approx(-5e-4, abs=0.0)
    assert x[5] - x[4] == pytest.approx(2e-6, rel=1e-15)


def test_carrier_field_refuses_a_bare_array_as_a_grid():
    """The ORIGIN is not inferable from an array, and getting it wrong
    relocates the field in absolute coordinates without changing a sample."""
    with pytest.raises(TypeError, match='ORIGIN'):
        CarrierField(np.zeros((4, 4), complex), (4, 4),   # type: ignore[arg-type]
                     CarrierSpec(R=-1.0), LAM)


# ---------------------------------------------------------------------------
# (e) THE NYQUIST ARITHMETIC -- reproduce the probe's census
# ---------------------------------------------------------------------------
def test_nyquist_reproduces_the_design_121_census():
    """The probe's S3 grid-sizing table, re-derived from two CarrierSpecs.

    The three numbers that decided the architecture's common grid, for order
    (-4,-2) against the on-axis common carrier at the design-121 back
    aperture (R = -7.712425 mm, chief (-1.9151, -0.9575) mm, exit cosines
    (-6.872e-04, -3.436e-04), measured support radius 2.64 mm):

        carrier-offset RAMP   0.2776  ->  dx <= 2.3593 um
        beam's own band       0.3239  ->  dx <= 2.0225 um   <== BINDING
        (the sum bound        0.809 um -- NOT operative)

    The probe's ramp and band are the PARAXIAL closed forms (|c|/|R| and
    r/sqrt(r^2+R^2)); this module maximises the exact gradients over the
    support disc, so agreement is expected to ~1 %, not to the digit."""
    R = -0.0077124254602782
    common = CarrierSpec(R=R, centre=(0.0, 0.0), tilt=(0.0, 0.0))
    order = CarrierSpec(R=R,
                        centre=(-0.001915097939205574, -0.000957548969602787),
                        tilt=(-0.0006871912839565426, -0.0003435956419782713))
    rep = carrier_difference_nyquist(order, common, LAM, 2.64e-3,
                                     dx_target=1.2292e-6)

    assert rep.ramp_max == pytest.approx(0.277623, rel=0.02)
    assert rep.dx_ramp == pytest.approx(2.3593e-6, rel=0.02)
    assert rep.na_src_max == pytest.approx(0.3239, rel=0.02)
    assert rep.dx_reconstruct == pytest.approx(2.0225e-6, rel=0.02)

    # the beam's own band binds, NOT the ramp -- that ordering is the finding
    assert rep.binding_term == 'reconstruct'
    assert rep.dx_binding == pytest.approx(2.0225e-6, rel=0.02)

    # ... and the sum bound, which is the natural mistake, is 2.5x tighter
    # than anything that actually governs.  If it were operative the probe's
    # 1.2292 um grid would have aliased; it measured no change.
    assert rep.dx_sum_bound == pytest.approx(0.8089e-6, rel=0.02)
    assert rep.dx_binding > 2.4 * rep.dx_sum_bound

    # the destination carrier's own band over the source's support is the
    # same red herring in another hat -- reported, never in the minimum
    assert rep.na_dst_max > 0.5
    assert rep.dx_dst_ref < rep.dx_binding

    # the probe chose 1.2292 um and called it "1.65x inside (c)"
    assert rep.margin == pytest.approx(1.65, rel=0.02)


def test_nyquist_ramp_is_the_carrier_difference_not_the_chief_ray_tilt():
    """The seam finding, isolated: two carriers whose CHIEF RAYS are parallel
    still leave a large residual ramp when their spheres are displaced.

    Reading the ramp off the direction cosines (both zero here) would size a
    grid from an infinite bound.  On design 121 that mistake was 400x."""
    R = -0.0077124254602782
    a = CarrierSpec(R=R, centre=(0.0, 0.0), tilt=(0.0, 0.0))
    b = CarrierSpec(R=R, centre=(2.14e-3, 0.0), tilt=(0.0, 0.0))
    rep = carrier_difference_nyquist(a, b, LAM, 1e-9)   # a point support
    assert a.tilt == b.tilt                              # no tilt spread at all
    # exact, at the source chief ray: |dc| / sqrt(|dc|^2 + R^2).  The probe
    # quoted its PARAXIAL truncation |dc|/|R| (0.2775); at |dc|/|R| = 0.28 the
    # two differ by 3.6 %, and the exact one is the smaller -- i.e. the
    # paraxial census was conservative, which is the safe direction.
    d = 2.14e-3
    assert rep.ramp_max == pytest.approx(d / math.hypot(d, R), rel=1e-6)
    assert rep.ramp_max == pytest.approx(d / abs(R), rel=0.04)
    assert rep.dx_ramp < 3e-6


# ---------------------------------------------------------------------------
# (a) A -> B -> A ROUND TRIP
# ---------------------------------------------------------------------------
def test_round_trip_envelope_is_exact_to_1e_12():
    """(a) A -> B -> A on a smooth field, NO leg: rel L2 <= 1e-12.

    The two grids share a physical EXTENT and an origin, which is forced and
    not a convenience: the MFT's faithful zone requires
    ``2|d_origin| + N_out*dx_out <= N_in*dx_in`` in BOTH directions, and the
    only pair satisfying both is equal extent at zero shift.  What varies is
    the PITCH (1024 @ 0.3 um <-> 1536 @ 0.2 um) and the whole carrier --
    sphere, chief ray, tilt and piston all move.  One-way origin shifts are
    covered by the design-121 null control, whose three source grids differ
    in origin by 0 / -1.508 / -3.016 mm."""
    N, dx, w, R = 1024, 0.3e-6, 20e-6, -5.0e-4
    fA = _gauss_field(N, dx, w, R)
    gB = FieldGrid((1536, 1536), N * dx / 1536)
    assert gB.extent[0] == pytest.approx(fA.grid.extent[0], rel=1e-15)

    carB = CarrierSpec(R=-5.5e-4, centre=(1.0e-6, -0.5e-6),
                       tilt=(1.0e-3, -5.0e-4), piston=3.0e-6)
    with warnings.catch_warnings():
        warnings.simplefilter('error')          # no replica / window warning
        fB = re_reference(fA, carB, gB)
        fA2 = re_reference(fB, fA.carrier, fA.grid)

    assert fB.provenance['re_reference']['resampled'] is True
    rel = _rel_l2(fA2.envelope, fA.envelope)
    assert rel <= 1e-12, f"round-trip rel L2 {rel:.3e}"

    # and the FULL FIELD -- the invariant the operation is defined by -- is
    # preserved to the same bar
    assert _rel_l2(fA2.full_field(), fA.full_field()) <= 1e-12


def test_round_trip_floor_is_the_eikonal_cancellation():
    """FINGERPRINT RULE.  The residual is not merely small, it is
    ``~k0 * eps * |R|`` -- an absolute phase error set by the library's
    ``sqrt(r^2+R^2) - |R|``, independent of how big the carrier DIFFERENCE
    is and linear in the sphere radius.

    Pinning the law rather than the number is what makes this a guard: a
    future change that reintroduces a genuine algebraic defect would move the
    residual off this line long before it moved it above a loose bar."""
    prev = None
    for R, dx, w in ((-5.0e-4, 0.3e-6, 20e-6), (-5.0e-3, 1.0e-6, 60e-6),
                     (-5.0e-2, 2.0e-6, 90e-6)):
        N = 1024
        fA = _gauss_field(N, dx, w, R)
        gB = FieldGrid((1536, 1536), N * dx / 1536)
        carB = CarrierSpec(R=R, centre=(3 * dx, -dx))
        fA2 = re_reference(re_reference(fA, carB, gB), fA.carrier, fA.grid)
        rel = _rel_l2(fA2.envelope, fA.envelope)
        floor = _eikonal_floor_rad(R)
        # the residual sits UNDER the predicted floor and within a decade of
        # it -- it is that mechanism and not a coincidence
        assert rel <= floor, f"|R|={abs(R):.3e}: {rel:.3e} > {floor:.3e}"
        assert rel >= 0.05 * floor, f"|R|={abs(R):.3e}: {rel:.3e} << floor"
        if prev is not None:
            # a decade in |R| is a decade in the residual (the law is linear,
            # not, say, quadratic or flat)
            assert 3.0 < rel / prev < 30.0
        prev = rel


def test_round_trip_with_no_carrier_change_is_the_bare_resample():
    """Control for the test above: with the carrier UNCHANGED the analytic
    phasor is skipped entirely and the residual drops two decades, to the
    band-limited resample's own floor.  That isolates which half of the
    operation each number belongs to."""
    N, dx, w, R = 1024, 2.0e-6, 90e-6, -5.0e-2
    fA = _gauss_field(N, dx, w, R)
    gB = FieldGrid((1536, 1536), N * dx / 1536)
    fA2 = re_reference(re_reference(fA, fA.carrier, gB), fA.carrier, fA.grid)
    assert fA2.provenance['re_reference']['resampled'] is True
    rel = _rel_l2(fA2.envelope, fA.envelope)
    assert rel <= 1e-12
    assert rel < 0.1 * _eikonal_floor_rad(R)


def test_re_reference_onto_the_same_lattice_is_bit_exact():
    """No resample means NO resample: the identity short-circuit must skip
    the MFT, and with the carrier also unchanged the envelope must come back
    byte for byte.

    This is the case an earlier draft got wrong in the other direction -- it
    applied ``exp(+i k C_src)`` and skipped ``exp(-i k C_dst)``, returning a
    FULL FIELD wearing an envelope's label, which every intensity metric
    would have passed."""
    fA = _gauss_field(256, 1e-6, 20e-6, -1e-3)
    out = re_reference(fA, fA.carrier, fA.grid)
    assert out.provenance['re_reference']['resampled'] is False
    assert out.envelope.tobytes() == fA.envelope.tobytes()


# ---------------------------------------------------------------------------
# (d) PISTON -- total OPL is preserved
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('piston', [0.0, 7.31234e-4, 3.007165811e-3])
def test_piston_only_re_reference_preserves_total_opl_exactly(piston):
    """(d) Re-referencing onto a carrier that differs ONLY in its piston must
    move the constant from the CARRIER into the ENVELOPE and change nothing
    else.  Total optical path is preserved EXACTLY -- not to a tolerance.

    3.007165811 mm is design 121's own axial path through the one slow
    singlet FIX_TILT_QUADRATIC_OPL S4.4 measures against, i.e. 2295 waves =
    1.4e+04 rad of piston.  A naive ``exp(1j * k0 * piston)`` loses
    ``eps * k0 * |p|`` ~ 3e-12 rad of it; the OPL-domain reduction does
    not."""
    fA = _gauss_field(256, 1e-6, 20e-6, -1e-3)
    carP = CarrierSpec(R=fA.carrier.R, centre=fA.carrier.centre,
                       tilt=fA.carrier.tilt, piston=piston)
    fP = re_reference(fA, carP, fA.grid)
    assert fP.carrier.piston == piston
    worst = 0.0
    for (j, i) in ((128, 128), (131, 121), (200, 60), (40, 220)):
        d = fP.total_opl_at(i, j) - fA.total_opl_at(i, j)
        worst = max(worst, abs((d + np.pi) % (2 * np.pi) - np.pi))
    assert worst <= 1e-12, f"total-OPL drift {worst:.3e} rad"
    # Intensity-blind by construction: the piston is a global UNIT phasor, so
    # the modulus moves only by the round-off of one complex multiply
    # (FIX_TILT_QUADRATIC_OPL S6.1 makes the same claim about the same
    # mechanism and measures the same size).
    assert np.allclose(np.abs(fP.envelope), np.abs(fA.envelope),
                       rtol=8 * EPS, atol=0)


def test_piston_survives_a_full_carrier_change():
    """(d), general case: sphere, chief ray, tilt AND piston all move at
    once, on one grid (so the comparison is the carrier ALGEBRA and not the
    resample's interpolation error).  Total OPL preserved to 1e-12 rad."""
    R = -5e-4
    fA = _gauss_field(512, 0.3e-6, 15e-6, R, piston=1.234e-5)
    carB = CarrierSpec(R=-5.5e-4, centre=(1.0e-6, -0.5e-6),
                       tilt=(1.0e-3, -5.0e-4), piston=8.7654e-4)
    fB = re_reference(fA, carB, fA.grid)
    worst = 0.0
    for (j, i) in ((256, 256), (260, 249), (300, 210), (200, 300)):
        d = fB.total_opl_at(i, j) - fA.total_opl_at(i, j)
        worst = max(worst, abs((d + np.pi) % (2 * np.pi) - np.pi))
    assert worst <= 1e-12, f"total-OPL drift {worst:.3e} rad"


def test_piston_is_recorded_not_absorbed():
    """The returned field must be referenced to the piston it was ASKED for.
    A primitive that folded the whole constant into the envelope and left
    ``carrier.piston = 0`` would still reconstruct correctly and would still
    have destroyed the bookkeeping the tilt-quadratic fix exists to
    provide."""
    fA = _gauss_field(128, 1e-6, 15e-6, -1e-3, piston=2.5e-4)
    carB = CarrierSpec(R=-1e-3, piston=-7.0e-4)
    fB = re_reference(fA, carB, fA.grid)
    assert fB.carrier.piston == -7.0e-4
    rep = fB.provenance['re_reference']
    assert float(rep['piston_delta']) == pytest.approx(2.5e-4 + 7.0e-4,
                                                       rel=1e-15)


# ---------------------------------------------------------------------------
# (c) SUM OF ONE == DIRECT, and linearity
# ---------------------------------------------------------------------------
def test_aggregate_of_one_equals_re_reference_exactly():
    """(c) ``aggregate`` is ``re_reference`` + a sum, and with one field
    there is nothing to add: the arrays must be bit-identical, not close."""
    fA = _gauss_field(512, 1e-6, 40e-6, -2e-3)
    gB = FieldGrid((768, 768), 512 * 1e-6 / 768)
    carB = CarrierSpec(R=-2.2e-3, centre=(2e-6, -1e-6), tilt=(2e-4, 0.0),
                       piston=1e-5)
    direct = re_reference(fA, carB, gB)
    res = aggregate([fA], carB, gB)
    assert res.field.envelope.tobytes() == direct.envelope.tobytes()
    assert res.field.carrier == carB
    assert res.field.grid == gB
    assert res.ledger.n_fields == 1


def test_aggregate_is_exactly_linear():
    """The property that makes an aggregation a DECOMPOSITION rather than an
    approximation -- and hence makes a crosstalk table mean anything.  The
    probe measured 2.3e-16 on the real fan."""
    fs, ws = [], []
    for cx, w in ((0.0, 1.0 + 0.0j), (60e-6, -0.3 + 0.4j),
                  (-90e-6, 0.2 - 0.1j)):
        fs.append(_gauss_field(512, 1e-6, 30e-6, -3e-3, centre=(cx, 0.0)))
        ws.append(w)
    common = CarrierSpec(R=-3e-3, centre=(0.0, 0.0))
    gc = FieldGrid((768, 768), 512 * 1e-6 / 768)
    all_at_once = aggregate(fs, common, gc, weights=ws).field
    one_by_one = sum(aggregate([f], common, gc, weights=[w]).field.envelope
                     for f, w in zip(fs, ws))
    assert _rel_l2(one_by_one, all_at_once.envelope) < 1e-15


def test_aggregate_ledger_accounts_for_out_of_window_power():
    """The energy book: a beam whose skirt hangs off the common window must
    have that power RECORDED, per field, not quietly dropped.

    Sized the house way -- the containment margin is the window half-extent
    minus (decentre + measured support radius), so it goes negative exactly
    when the guard has something to say."""
    # a wide beam onto a deliberately small common window
    f = _gauss_field(1024, 1e-6, 120e-6, -0.05)
    small = FieldGrid((256, 256), 1e-6)          # 256 um across, beam is wider
    with pytest.warns(RuntimeWarning, match='does not fit the target window'):
        res = aggregate([f], f.carrier, small, on_window='warn')
    row = res.ledger.rows[0]
    assert row.containment_margin < 0.0
    assert row.power_out_of_window > 0.0
    assert 0.0 < row.frac_out_of_window < 1.0
    assert res.ledger.frac_out_of_window_total == pytest.approx(
        row.frac_out_of_window, rel=1e-12)
    # the ledger is also on the returned field, as JSON
    led = res.field.provenance['aggregate_ledger']
    assert led['n_fields'] == 1
    assert float(led['frac_out_of_window_total']) > 0.0


def test_aggregate_ledger_is_clean_when_the_window_contains_the_beam():
    f = _gauss_field(1024, 1e-6, 40e-6, -0.05)
    big = FieldGrid((1024, 1024), 1e-6)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        res = aggregate([f], f.carrier, big)
    row = res.ledger.rows[0]
    assert row.containment_margin > 0.0
    assert abs(row.frac_out_of_window) < 1e-12


def test_aggregate_refuses_mixed_wavelengths():
    a = _gauss_field(64, 1e-6, 10e-6, -1e-3)
    b = _gauss_field(64, 1e-6, 10e-6, -1e-3)
    b.wavelength = 1.55e-6
    with pytest.raises(ValueError, match='different wavelengths'):
        aggregate([a, b], a.carrier, a.grid)


# ---------------------------------------------------------------------------
# (e) THE NYQUIST GUARD -- refusal, and the fail-before that justifies it
# ---------------------------------------------------------------------------
def _guard_fixture():
    """A geometry in which the RAMP is the binding term (a long, weak sphere
    keeps the reconstruction bound out of the way) and in which an adequate
    and an inadequate common grid differ only in pitch."""
    N, dx, w, R = 1024, 2.0e-6, 200e-6, -0.5
    fA = _gauss_field(N, dx, w, R)
    # a pure TILT difference: the ramp is then exactly |dL, dM| everywhere,
    # so the arithmetic under test is not entangled with the support disc
    carB = CarrierSpec(R=R, tilt=(0.05, 0.0))
    ok = FieldGrid((512, 512), N * dx / 512)      # 4.0 um  -- inside 13.1 um
    bad = FieldGrid((128, 128), N * dx / 128)     # 16.0 um -- outside
    return fA, carB, ok, bad


def test_nyquist_guard_refuses_a_grid_that_cannot_hold_the_ramp():
    fA, carB, ok, bad = _guard_fixture()
    rep = carrier_difference_nyquist(fA.carrier, carB, LAM,
                                     fA.support_radius())
    assert rep.binding_term == 'ramp'
    # a pure 0.05 tilt difference -> ramp 0.05 -> dx <= lambda/0.1 = 13.1 um.
    # (Not to the ulp: the tilted carrier's EXACT eikonal puts its sphere's
    # projection at R*L/n, so the difference of the two gradients is 0.05 only
    # to the C5 exactness term's own curvature, here 7e-05 relative.)
    assert rep.dx_ramp == pytest.approx(LAM / 0.1, rel=1e-3)
    assert ok.dx < rep.dx_binding < bad.dx

    with pytest.raises(ValueError) as ei:
        re_reference(fA, carB, bad)
    msg = str(ei.value)
    assert 'cannot hold the carrier difference' in msg
    assert 'do NOT add' in msg          # the finding is IN the refusal
    assert "'ramp'" in msg

    # house style: the disposition is a knob, and it is validated
    with pytest.warns(RuntimeWarning, match='cannot hold'):
        re_reference(fA, carB, bad, on_nyquist='warn')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        re_reference(fA, carB, bad, on_nyquist='ignore')
    with pytest.raises(ValueError, match="must be 'error', 'warn' or"):
        re_reference(fA, carB, ok, on_nyquist='sometimes')


def test_nyquist_guard_fail_before_the_refused_grid_really_is_wrong():
    """COMPARATIVE RULE -- the refusal is load-bearing, measured.

    Same field, same carriers, two common grids differing only in pitch, both
    round-tripped back to the source lattice and carrier:

      * the ADEQUATE grid returns the field (rel L2 ~ 1e-9, i.e. the
        |R| = 0.5 m eikonal floor);
      * the REFUSED grid, with the guard disabled, returns a populated,
        finite, credible-looking field that is wrong by O(1).

    Without the guard that second result has no signature: its power is
    right, its envelope is smooth, and nothing in it says the ramp wrapped."""
    fA, carB, ok, bad = _guard_fixture()

    good = re_reference(re_reference(fA, carB, ok), fA.carrier, fA.grid)
    rel_ok = _rel_l2(good.envelope, fA.envelope)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        aliased = re_reference(re_reference(fA, carB, bad, on_nyquist='ignore'),
                               fA.carrier, fA.grid, on_nyquist='ignore')
    rel_bad = _rel_l2(aliased.envelope, fA.envelope)

    assert rel_ok < 1e-9, f"the adequate grid should be exact, got {rel_ok:.3e}"
    assert rel_bad > 0.1, f"the refused grid should be WRONG, got {rel_bad:.3e}"
    assert rel_bad > 1e6 * rel_ok
    # ... and it looks perfectly healthy
    assert np.all(np.isfinite(aliased.envelope))


def test_guard_refuses_a_coarse_co_moving_plane_and_that_is_the_point():
    """PROBE_SUM_AT_APERTURE S2 in code: the only plane at which traced
    orders may legitimately be summed is the last group's back aperture ON
    THE EXACT LEG'S FINE RETRACE GRID.  A chain's coarse co-moving exit plane
    cannot carry its own exit congruence -- design 121 measured dx 33.2 um
    where the exit sphere needs 4.26 um, 7.8x under-sampled -- and the
    ``reconstruct`` term refuses it.

    That is the finding ENFORCED, not restated, and it is why this test
    exists next to the exemption below: the exemption must not weaken it."""
    # a design-121-shaped congruence on the coarse co-moving pitch
    coarse = FieldGrid((1024, 1024), 33.2112e-6)
    x, y = coarse.axes()
    env = np.exp(-((x[None, :] ** 2 + y[:, None] ** 2) / 1.1853e-3 ** 2)
                 ).astype(np.complex128)
    car = CarrierSpec(R=-7.712425e-3)
    f = CarrierField(env, coarse, car, LAM)
    common = CarrierSpec(R=-7.712425e-3, centre=(1.0e-4, 0.0))
    with pytest.raises(ValueError) as ei:
        re_reference(f, common, coarse, on_window='ignore')
    assert "'reconstruct'" in str(ei.value)

    # ... and the STRICT no-op is exempt, because an identity cannot
    # introduce a sampling error and refusing it would be a false positive
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        out = re_reference(f, car, coarse)
    assert out.envelope.tobytes() == f.envelope.tobytes()
    # the exemption is STRICT: change the carrier on the same lattice and the
    # guard is back
    with pytest.raises(ValueError, match='cannot hold'):
        re_reference(f, CarrierSpec(R=-7.712425e-3, piston=1e-6,
                                    centre=(1.0e-4, 0.0)), coarse,
                     on_window='ignore')


def test_window_guard_disposition_is_a_knob():
    f = _gauss_field(512, 1e-6, 100e-6, -0.05)
    tiny = FieldGrid((64, 64), 1e-6)
    with pytest.raises(ValueError, match='does not fit the target window'):
        re_reference(f, f.carrier, tiny, on_window='error')
    with pytest.warns(RuntimeWarning, match='does not fit'):
        re_reference(f, f.carrier, tiny, on_window='warn')
    with pytest.raises(ValueError, match="must be 'error', 'warn' or"):
        re_reference(f, f.carrier, tiny, on_window='loudly')


def test_nyquist_margin_is_honoured():
    fA, carB, ok, bad = _guard_fixture()
    # ok.dx = 4 um against a 13.1 um bound -> 3.3x of margin, so 2x passes
    # and 4x does not
    re_reference(fA, carB, ok, nyquist_margin=2.0)
    with pytest.raises(ValueError, match='required margin 4.00x'):
        re_reference(fA, carB, ok, nyquist_margin=4.0)


# ---------------------------------------------------------------------------
# Zarr IO
# ---------------------------------------------------------------------------
zarr = pytest.importorskip('zarr', reason='CarrierField storage needs zarr>=3')


def test_zarr_round_trip_is_bit_identical(tmp_path):
    """Save/load must return the SAME BYTES and the same scalars -- exact
    equality, not ``allclose``.  A field that comes back 1 ulp different is a
    field whose stored provenance no longer describes it."""
    f = _gauss_field(256, 1.5e-6, 30e-6, -3.5e-3,
                     centre=(2.5e-6, -1.25e-6), tilt=(3e-4, -7e-4),
                     piston=1.2345678901234e-3, origin=(-1.5e-3, 0.75e-3),
                     prov={'order': [-4, -2], 'chain_a_key':
                           {'schema': 7, 'lumenairy_source_sha256': 'f68ac2e8',
                            'n': 1024, 'dx0': '2e-06'},
                           'note': 'cp1252-safe ascii'})
    p = tmp_path / 'cf.zarr'
    save_carrier_field_zarr(p, f, name='order_m4_m2')
    g = load_carrier_field_zarr(p, name='order_m4_m2')

    assert g.envelope.tobytes() == f.envelope.tobytes()
    assert g.envelope.dtype == f.envelope.dtype
    assert g.grid == f.grid
    assert g.carrier == f.carrier
    assert g.wavelength == f.wavelength
    assert g.provenance == f.provenance


def test_zarr_round_trip_carries_a_collimated_carrier(tmp_path):
    """``R = +/-inf`` is the library's spelling of collimated and is not JSON.
    It must survive as ``inf``, not as ``null`` and not as a large float."""
    f = _gauss_field(64, 1e-6, 10e-6, float('-inf'), tilt=(1e-3, 0.0))
    p = tmp_path / 'coll.zarr'
    save_carrier_field_zarr(p, f)
    g = load_carrier_field_zarr(p)
    assert g.carrier.R == float('-inf')
    assert g.carrier.is_collimated
    assert g.envelope.tobytes() == f.envelope.tobytes()


def test_zarr_stores_one_group_per_field(tmp_path):
    p = tmp_path / 'fan.zarr'
    names = []
    for k in range(3):
        f = _gauss_field(64, 1e-6, 10e-6, -1e-3, centre=(k * 5e-6, 0.0),
                         prov={'order': [-k, 0]})
        names.append(f'order_{k}')
        save_carrier_field_zarr(p, f, name=names[-1])
    store = zarr.open_group(str(p), mode='r')
    assert sorted(store.group_keys()) == sorted(names)
    for k, nm in enumerate(names):
        assert load_carrier_field_zarr(p, name=nm).provenance['order'] == [-k, 0]


def test_zarr_refuses_to_clobber_and_obeys_overwrite(tmp_path):
    p = tmp_path / 'x.zarr'
    f = _gauss_field(32, 1e-6, 5e-6, -1e-3)
    save_carrier_field_zarr(p, f)
    with pytest.raises(FileExistsError, match='overwrite=True'):
        save_carrier_field_zarr(p, f)
    save_carrier_field_zarr(p, f, overwrite=True)


def test_zarr_refuses_an_unknown_schema(tmp_path):
    p = tmp_path / 'y.zarr'
    f = _gauss_field(32, 1e-6, 5e-6, -1e-3)
    save_carrier_field_zarr(p, f)
    store = zarr.open_group(str(p), mode='r+')
    store['field'].attrs['schema'] = CARRIER_FIELD_SCHEMA + 99
    with pytest.raises(ValueError, match='Refusing to guess'):
        load_carrier_field_zarr(p)


def test_zarr_actually_compresses_the_envelope(tmp_path):
    """The codec chain has to be doing work, and on the ENVELOPE rather than
    on a full field.

    An envelope is smooth by construction -- that is what dividing the
    carrier out is for -- so its float64 byte planes are highly correlated
    and shuffle+zstd bites.  A full field on the same grid carries the
    carrier's fringes and is close to incompressible.  The ratio measured on
    a real design-121 aperture is in
    ``docs/audits/BUILD_CARRIER_FIELD_2026_08_11.md``; here the point is only
    that the two differ in the stated direction, on the same bytes."""
    N = 512
    f = _gauss_field(N, 1e-6, 60e-6, -2e-3)
    raw = f.envelope.nbytes

    p1 = tmp_path / 'env.zarr'
    save_carrier_field_zarr(p1, f)
    env_bytes = sum(fp.stat().st_size for fp in p1.rglob('*') if fp.is_file())

    full = CarrierField(f.full_field(), f.grid,
                        CarrierSpec(R=float('inf')), LAM)
    p2 = tmp_path / 'full.zarr'
    save_carrier_field_zarr(p2, full)
    full_bytes = sum(fp.stat().st_size for fp in p2.rglob('*') if fp.is_file())

    assert env_bytes < raw
    assert env_bytes < full_bytes


def test_provenance_must_be_json_serialisable():
    """Provenance is the object's identity for the storage contract, so a
    dict that cannot be written is refused at CONSTRUCTION rather than at
    save time -- when the array has already been computed."""
    with pytest.raises(TypeError, match='JSON-serialisable'):
        _gauss_field(16, 1e-6, 3e-6, -1e-3, prov={'bad': object()})


def test_provenance_is_a_fixed_point_of_the_round_trip():
    """A tuple in, a list out -- but only ONCE.  Canonicalising at
    construction is what makes ``load(save(f)).provenance == f.provenance``
    a real assertion instead of a near-miss."""
    f = _gauss_field(16, 1e-6, 3e-6, -1e-3, prov={'order': (-4, -2)})
    assert f.provenance == {'order': [-4, -2]}
    assert f.provenance == json.loads(json.dumps(f.provenance))
