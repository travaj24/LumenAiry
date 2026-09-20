"""VERIFY-C1 -- the independent verification of WP-C1's `edge='gray'` default.

WP-C1 flipped `apply_aperture`'s `edge` default to `'gray'` and pinned it in
``tests/unit/test_c1_gray_edge_default.py``.  This file is the INDEPENDENT
verification: every decision below is re-measured on a DIFFERENT optic and a
DIFFERENT fixture family from WP-C1's and WP-B11's, and each test closes a gap
the verification found rather than restating a claim the shipped file already
makes.

The optic.  WP-B11 sec. 2.9 and WP-C1 both measure lambda = 633 nm, a = 100 um,
window 512 um, z = 16 mm (RS) / 5 mm (HF).  Re-running that geometry would
re-read their number.  This file uses lambda = 1064 nm, a = 62.5 um, window
400 um, z = 4.0 mm (RS, Fresnel number a^2/(lambda z) = 0.918) and 2.5 mm (HF),
against the same CLOSED-FORM on-axis oracle

    U(0, 0, z) = e^{ikz} - (z / r_a) e^{ik r_a},    r_a = sqrt(z^2 + a^2)

which depends on no discretisation, no kernel and no build.  z_RS clears the
spatial kernel's alias threshold 2 W^2 / (N lambda) -- 2.35 mm at the coarsest
grid -- at every N, and the test asserts that rather than assuming it.

The gaps closed here (each named in
``docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/VERIFY_WP-C1.md``):

  V1  the jit'd JAX kernel must honour the element's ``edge_samples``.  A
      mutant that drops it from the static signature diverges from the NumPy
      chain (measured) and survived the WHOLE shipped suite on both builds.
  V2  the three chain routes must agree on REFUSAL, not only on the field.
      Two element dicts were accepted by the jit'd route and refused by the
      other two; those rows were strict xfails and FLIPPED to passes in
      round 2, when the refusal moved into the one place both backends read
      the element.
  V3  ``lumenairy.evaluate`` reaches an ``'aperture'`` element and so moves
      with the default, and exposes no ``edge`` of its own -- an entry point
      the Migration table does not name.
  V4  "grey BEATS hard" restated as a strict, ratio-bounded decision: the
      shipped ``e_g4 <= e_hard`` is satisfied by EQUALITY, so it still passes
      when the two arms are the same array.
  V5  the POWER band ``[area - n_rim dx dy / 4, area]`` re-derived and checked
      TWO-SIDED on a decentred circle, an anamorphic rectangle on an
      anamorphic grid, and a decentred annulus.
  V6  the convergence decision on this optic, and the SCOPE of the
      "no order at all" claim: the hard arm's error does NOT rise on this
      optic, so non-monotonicity is fixture-specific while the rate gap is not.
  V7  ``edge_samples = 4`` is the knee from both sides on this optic too.
  V8  the way back is bit-exact against a hand-built binary oracle on a
      fixture family WP-C1's does not use (odd grid, decentred + anamorphic
      annulus, complex64, non-finite field).

Bars.  Every bound is derived from a measurement quoted beside it, measured
2026-09-20 on Windows py3.14 / numpy 2.4.4 / scipy-openblas and WSL py3.12 /
numpy 2.4.6 / scipy-openblas, which agree to twelve significant figures on the
ladder and bit for bit on every mask reading (a mask sum is an integer count
over n_sub**2, with no BLAS in it).  Raw JSON:
``validation/probe_verify_c1/``.

Runtime: whole file ~35 s on the reference box; the slowest single test is the
two-kernel ladder at ~16 s.  No test is ``slow``.
"""
import inspect

import numpy as np
import pytest

from lumenairy.elements import elements as elem_mod
from lumenairy.propagators.hf import (
    propagate_huygens_fresnel_with_opl_callable)
from lumenairy.propagators.rs import rayleigh_sommerfeld_propagate
from lumenairy.propagators.system import propagate_through_system

apply_aperture = elem_mod.apply_aperture

# --- the VERIFY optic, deliberately not WP-B11's --------------------------
LAM = 1064e-9
A = 62.5e-6           # aperture RADIUS [m]
WINDOW = 400e-6       # full window [m]
LADDER_N = (128, 256, 512, 1024)
Z_RS = 4.0e-3
Z_HF = 2.5e-3


def _closed_form_on_axis(z):
    k = 2.0 * np.pi / LAM
    r_a = np.sqrt(z * z + A * A)
    return np.exp(1j * k * z) - (z / r_a) * np.exp(1j * k * r_a)


def _apertured(N, **kw):
    dx = WINDOW / N
    return apply_aperture(np.ones((N, N), dtype=complex), dx, 'circular',
                          {'diameter': 2.0 * A}, **kw), dx


def _rs_err(N, **kw):
    E, dx = _apertured(N, **kw)
    out = rayleigh_sommerfeld_propagate(E, z=Z_RS, wavelength=LAM, dx=dx,
                                        kernel='spatial')
    u = _closed_form_on_axis(Z_RS)
    return abs(complex(out[N // 2, N // 2]) - u) / abs(u)


def _hf_err(N, **kw):
    E, dx = _apertured(N, **kw)

    def opl_fn(s1x, s1y, s2x, s2y):          # WAVES, per the units contract
        return np.sqrt((s1x - s2x) ** 2 + (s1y - s2y) ** 2
                       + Z_HF * Z_HF) / LAM

    out = propagate_huygens_fresnel_with_opl_callable(
        E, opl_fn=opl_fn, output_grid_x=np.array([0.0]),
        output_grid_y=np.array([0.0]), input_grid_dx=dx)
    u = _closed_form_on_axis(Z_HF)
    return abs(complex(np.reshape(out, (-1,))[0]) - u) / abs(u)


KERNELS = {'rs_spatial': _rs_err, 'hf_quadrature': _hf_err}


def _jax_or_skip():
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    return jax


def _bytes(a):
    return np.ascontiguousarray(np.asarray(a)).tobytes()


def _chain(E, elements, dx=1.25e-6):
    out = propagate_through_system(E, elements, LAM, dx=dx)
    return np.asarray(out[0] if isinstance(out, tuple) else out)


def _field(N=96, seed=2027):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))


# ===========================================================================
# V1 -- the jit'd JAX kernel must honour the element's edge_samples
# ===========================================================================

@pytest.mark.parametrize('n_sub', [1, 2, 8, 16])
def test_verify_c1_the_jit_kernel_honours_the_elements_edge_samples(n_sub):
    """GAP CLOSED.  ``_system_element_signature`` puts ``edge_samples`` in the
    jit'd kernel's STATIC signature; if it stops doing so, the jit'd route
    silently serves the DEFAULT rim for an element that asked for another one,
    while the NumPy chain and the eager JAX route honour it.

    MEASURED 2026-09-20 (both builds): with that one line replaced by
    ``n_sub = None``, an element naming ``edge_samples=8`` gives
    ``d3ca8278...`` on NumPy and on the eager route and ``5608c91c...`` on the
    jit'd kernel -- a real answer divergence -- and the whole shipped suite
    (61 files, 2796 ids on Windows; the WP-C1 file on WSL) stayed GREEN.  This
    id is the one that goes red.

    Bit identity, not a tolerance: the three routes run the same body on the
    same input, so anything short of equal bytes is a divergence.
    """
    _jax_or_skip()
    import jax.numpy as jnp
    from lumenairy.propagators.system import propagate_through_system_jax

    E = _field(N=96, seed=311)
    elements = [{'type': 'aperture', 'shape': 'circular',
                 'params': {'diameter': 9.1e-5}, 'edge_samples': n_sub}]
    npy = _chain(E, elements)
    jit_ = np.asarray(propagate_through_system_jax(
        jnp.asarray(E), elements, LAM, 1.25e-6))
    eager = np.asarray(propagate_through_system_jax(
        jnp.asarray(E), elements, LAM, 1.25e-6, verbose=True))
    assert _bytes(jit_) == _bytes(npy), (
        f"the jit'd JAX kernel did not honour edge_samples={n_sub} from the "
        f"element dict (it differs from the NumPy chain on the same dict)")
    assert _bytes(eager) == _bytes(npy), (
        f"the eager JAX route did not honour edge_samples={n_sub}")
    # ... and the element really does select a different rim, so the identity
    # above is not three routes agreeing on the same default.
    plain = _chain(E, [{'type': 'aperture', 'shape': 'circular',
                        'params': {'diameter': 9.1e-5}}])
    assert (_bytes(npy) != _bytes(plain)) == (n_sub != 4), n_sub


# ===========================================================================
# V2 -- the three routes must agree on REFUSAL, not only on the field
# ===========================================================================

_EDGE_KEY_CASES = [
    ('edge_bogus', {'edge': 'soft'}, True),
    ('edge_none', {'edge': None}, True),
    ('edge_samples_zero', {'edge_samples': 0}, True),
    ('edge_samples_negative', {'edge_samples': -2}, True),
    ('edge_samples_float_4p0', {'edge_samples': 4.0}, False),
    # VERIFY-C1 D1, CLOSED 2026-09-20 (round 2).  These two rows were strict
    # xfails: ``_system_element_signature`` coerced with ``int()`` / ``str()``
    # before ``apply_aperture`` saw the value, so the jit'd JAX route ACCEPTED
    # 2.5 (silently using 2) and '4' where the NumPy chain and the eager JAX
    # route both raised.  The refusal now lives in
    # ``elements._validate_edge_kwargs``, called from ``_aperture_edge_kwargs``
    # -- the one place both backends read the element -- so all three routes
    # raise the same ValueError with the same message.  The markers are gone
    # with the defect; if the coercion ever comes back these rows go RED, which
    # is what a strict xfail was standing in for.
    ('edge_samples_float_2p5', {'edge_samples': 2.5}, True),
    ('edge_samples_str', {'edge_samples': '4'}, True),
]


@pytest.mark.parametrize('label,bad,should_raise', _EDGE_KEY_CASES)
def test_verify_c1_all_three_chain_routes_agree_on_an_edge_element(
        label, bad, should_raise):
    """A chain's ``'edge'`` / ``'edge_samples'`` keys are a CONTRACT, and a
    contract has two sides: the same element dict must be accepted by all
    three routes or refused by all three.

    WP-C1's claim is "one implementation, one default and one way back".
    That is true of the FIELD and was verified bit for bit; it is not yet true
    of the REFUSAL, because the jit'd route reads the element through
    ``_system_element_signature``, which coerces with ``int()`` / ``str()``
    before ``apply_aperture`` ever sees the value.

    CLOSED 2026-09-20 (round 2): every row now agrees on all three routes.
    The two rows that were strict xfails (``edge_samples`` 2.5 and '4') are
    plain params, and the refusal is a single shared guard rather than a
    per-route one -- see
    ``tests/unit/test_c1_gray_edge_default.py::
    test_c1_all_three_chain_routes_refuse_a_bad_edge_element_identically``,
    which additionally pins that the MESSAGE is the same on all three.
    """
    _jax_or_skip()
    import jax.numpy as jnp
    from lumenairy.propagators.system import propagate_through_system_jax

    E = _field(N=64, seed=313)
    elements = [dict({'type': 'aperture', 'shape': 'circular',
                      'params': {'diameter': 5.3e-5}}, **bad)]

    def _raises(fn):
        try:
            fn()
        except (ValueError, TypeError):
            return True
        return False

    numpy_raises = _raises(lambda: _chain(E, elements))
    jit_raises = _raises(lambda: propagate_through_system_jax(
        jnp.asarray(E), elements, LAM, 1.25e-6))
    eager_raises = _raises(lambda: propagate_through_system_jax(
        jnp.asarray(E), elements, LAM, 1.25e-6, verbose=True))
    assert numpy_raises == should_raise, (label, numpy_raises)
    assert eager_raises == numpy_raises, (
        f"{label}: the eager JAX route and the NumPy chain disagree on "
        f"whether this element is legal")
    assert jit_raises == numpy_raises, (
        f"{label}: the jit'd JAX route and the NumPy chain disagree on "
        f"whether this element is legal (jit raised={jit_raises}, "
        f"numpy raised={numpy_raises})")


# ===========================================================================
# V3 -- lumenairy.evaluate reaches an 'aperture' element and has no way back
# ===========================================================================

_STOP_PRESCRIPTION = {
    'elements': [
        {'surf_num': 1, 'element_type': 'surface', 'radius': np.inf,
         'glass_after': 'air', 'is_stop': True, 'semi_diameter': 1.2e-3,
         'comment': 'STOP'},
        {'surf_num': 2, 'element_type': 'surface', 'radius': 0.05,
         'glass_after': 'N-BK7', 'semi_diameter': 2e-3},
        {'surf_num': 3, 'element_type': 'surface', 'radius': -0.05,
         'glass_after': 'air', 'semi_diameter': 2e-3},
    ],
    'all_thicknesses': [2e-3, 3e-3, 20e-3],
    'aperture_diameter': 2.4e-3,
}


def test_verify_c1_evaluate_takes_the_new_rim_and_has_no_public_way_back():
    """GAP CLOSED.  ``lumenairy.evaluate`` -- the documented one-call entry for
    a ``.zmx`` prescription -- decomposes a STOP surface into an
    ``{'type': 'aperture'}`` element with NO ``edge`` key, so it takes
    ``apply_aperture``'s default and its answer MOVED at 5.49.0.  It is not in
    the Migration-Guide's table of entry points, and unlike every entry point
    that IS in that table it has no way back a caller can reach: ``evaluate``
    takes no ``edge`` argument and builds its element list internally.

    MEASURED 2026-09-20: this fixture's returned field moves from
    ``e7b1f67b...`` at 49ddf4bd to ``59115e8e...`` here, on both builds.

    The test pins the three facts a Migration row would need: the element is
    emitted, ``evaluate`` exposes no ``edge``, and the ONLY route to the old
    answer is the private builder.  It stays true after the defect is fixed by
    adding a documented route -- at which point the last assertion is the one
    to restate.
    """
    import lumenairy as la
    from lumenairy.propagators.system import _prescription_to_elements

    elements = _prescription_to_elements(_STOP_PRESCRIPTION)
    apertures = [e for e in elements if e.get('type') == 'aperture']
    assert len(apertures) == 1, elements
    assert 'edge' not in apertures[0] and 'edge_samples' not in apertures[0]

    assert 'edge' not in inspect.signature(la.evaluate).parameters, (
        "evaluate now exposes an 'edge' -- restate this test and the "
        "Migration row it stands in for")

    src = la.Source.gaussian(N=128, dx=40e-6, wavelength=633e-9, w0=1.0e-3)
    got = np.asarray(la.evaluate(_STOP_PRESCRIPTION, src).field)
    hard_elements = [dict(e, edge='hard') if e.get('type') == 'aperture'
                     else e for e in elements]
    out = propagate_through_system(np.asarray(src.E), hard_elements,
                                   633e-9, dx=40e-6)
    hard = np.asarray(out[0] if isinstance(out, tuple) else out)
    assert _bytes(got) != _bytes(hard), (
        "evaluate's answer does not move with the default -- if that is now "
        "true the Migration row is unnecessary")
    gray_elements = [dict(e, edge='gray') if e.get('type') == 'aperture'
                     else e for e in elements]
    out_g = propagate_through_system(np.asarray(src.E), gray_elements,
                                     633e-9, dx=40e-6)
    gray = np.asarray(out_g[0] if isinstance(out_g, tuple) else out_g)
    assert _bytes(got) == _bytes(gray), (
        "evaluate does not take apply_aperture's default rim")


# ===========================================================================
# V4 -- "grey BEATS hard", strictly, with a measured margin
# ===========================================================================

@pytest.mark.parametrize('d_px,dy_ratio,offset', [(37, 1.0, 0.37),
                                                  (63, 2.5, 0.13),
                                                  (145, 0.4, 0.29)])
def test_verify_c1_grey_strictly_beats_hard_by_a_measured_ratio(
        d_px, dy_ratio, offset):
    """GAP CLOSED.  ``test_verify_a8_e7_gray_edge_beats_hard_at_anamorphic_and_offset_rims``
    was de-vacuumed by WP-C1 (its hard arm is named again) but its headline
    assertion is ``e_g4 <= e_hard``, which is satisfied by EQUALITY: with
    ``edge='hard'`` mutated to return the grey mask, that id still PASSES
    (measured -- 15 other ids go red under that mutation and this is not one
    of them).  A test named "beats" should not pass when the two arms are the
    same array.

    Restated as a RATIO decision with a derived bar.  MEASURED 2026-09-20 on
    this box, identical on both builds (mask sums are integer counts over
    n_sub**2, no BLAS): ``e_hard / e_g4`` reads 5.80, 8.82 and 1.99 on the
    three fixtures.  Bar 1.5x: below the smallest measurement by 1.33x and
    decisively above 1.0, where "grey IS hard" sits.
    """
    N, dx = 512, 1e-6
    dy = dx * dy_ratio
    D = d_px * dx
    E = np.ones((N, N), dtype=np.complex128)
    analytic = np.pi * (D / 2) ** 2

    def _area(**kw):
        out = apply_aperture(E, dx, 'circular', {'diameter': D},
                             xc=offset * dx, yc=-offset * dy, dy=dy, **kw)
        return float(np.sum(np.real(out))) * dx * dy

    e_hard = abs(_area(edge='hard') / analytic - 1.0)
    e_g4 = abs(_area(edge='gray') / analytic - 1.0)
    assert e_g4 > 0.0, e_g4
    assert e_hard / e_g4 >= 1.5, (d_px, dy_ratio, offset, e_hard, e_g4)


# ===========================================================================
# V5 -- the POWER band, re-derived and checked two-sided
# ===========================================================================

_BAND_CASES = {
    # (shape, params, xc/dx, yc/dy, dy/dx, analytic area)
    'decentred_circle': ('circular', {'diameter': 125e-6}, 3.37, -1.83, 1.0,
                         np.pi * (62.5e-6) ** 2),
    'anamorphic_rect_decentred': ('rectangular',
                                  {'width_x': 173e-6, 'width_y': 91e-6},
                                  2.5, 1.25, 1.6, 173e-6 * 91e-6),
    'decentred_annulus': ('annular',
                          {'inner_diameter': 40e-6, 'outer_diameter': 130e-6},
                          1.7, 0.9, 1.0,
                          np.pi * ((65e-6) ** 2 - (20e-6) ** 2)),
}


@pytest.mark.parametrize('case', sorted(_BAND_CASES))
def test_verify_c1_the_power_band_is_a_two_sided_bound_not_a_tolerance(case):
    """The aperture is an AMPLITUDE mask, so for a unit-amplitude field the
    LINEAR mask sum is the transmitted AREA and the QUADRATIC sum is the
    transmitted POWER.  With ``f`` in [0, 1]:

        f**2 <= f                     =>  POWER <= AREA,
        f - f**2 <= 1/4 on a rim px   =>  POWER >= AREA - n_rim dx dy / 4,

    and only rim pixels (0 < f < 1) contribute to the gap, because
    ``f - f**2`` is exactly 0 at f = 0 and f = 1.  That is a BOUND derived
    from the mask's own range, not a tolerance, and it is checked here on
    both sides with a strictly positive slack, on three apertures WP-C1's
    validation restatement does not use.

    MEASURED 2026-09-20 (N = 512, dx = 400/512 um), identical on both builds:
    the deficit is 69 %, 42 % and 68 % of the bound respectively, so the bound
    is real and not tight; and the EXACT identity ``AREA - POWER =
    sum(f - f**2) dx dy`` is asserted below to 1e-12 relative, which is what
    makes the band a derivation rather than a fit.
    """
    shape, params, xr, yr, dyr, analytic = _BAND_CASES[case]
    N = 512
    dx = WINDOW / N
    dy = dyr * dx
    m = np.real(apply_aperture(np.ones((N, N), dtype=complex), dx, shape,
                               params, xc=xr * dx, yc=yr * dy, dy=dy))
    area = float(m.sum()) * dx * dy
    power = float(np.sum(m ** 2)) * dx * dy
    n_rim = int(np.count_nonzero((m > 0) & (m < 1)))
    bound = n_rim * dx * dy / 4.0
    assert n_rim > 0, case
    assert power <= area, (case, power, area)
    assert power >= area - bound, (case, power, area - bound)
    assert power < area, (case, 'the rim must cost POWER, not nothing')
    assert power > area - bound, (case, 'the bound must not be attained')
    gap = float(np.sum(m - m ** 2)) * dx * dy
    assert abs((area - power) - gap) <= 1e-12 * area, (case, area - power, gap)
    # the grey AREA is closer to the analytic area than the binary one on a
    # CURVED rim (the axis-aligned case is the scope note pinned elsewhere)
    if shape != 'rectangular':
        hard = np.real(apply_aperture(np.ones((N, N), dtype=complex), dx,
                                      shape, params, xc=xr * dx, yc=yr * dy,
                                      dy=dy, edge='hard'))
        e_hard = abs(float(hard.sum()) * dx * dy / analytic - 1.0)
        e_gray = abs(area / analytic - 1.0)
        assert e_gray < e_hard, (case, e_gray, e_hard)


# ===========================================================================
# V6 -- the convergence decision, and the SCOPE of "no order at all"
# ===========================================================================

def test_verify_c1_the_rate_gap_reproduces_on_an_independent_optic():
    """The decision the default rests on, re-measured on a different source,
    radius, window and propagation distance from WP-B11's.

    MEASURED 2026-09-20, Windows py3.14 and WSL py3.12 agreeing to twelve
    significant figures (relative on-axis error at N = 128/256/512/1024):

        RS hard  8.8476e-03  2.7620e-03  2.3998e-03  9.3571e-04
        RS gray  1.8030e-03  6.6308e-04  1.0924e-04  3.2122e-05
        HF hard  1.9027e-02  5.9630e-03  5.1852e-03  2.0526e-03
        HF gray  6.2004e-03  1.8600e-03  4.2403e-04  1.3795e-04

    so over the three halvings the grey arm gains 56.1x (RS) and 44.9x (HF)
    -- mean orders 1.94 and 1.83, i.e. second order -- while the hard arm
    gains 9.46x and 9.27x -- mean orders 1.08 and 1.06, i.e. first order --
    and the hard arm's STEP orders are erratic (0.20 at one step on both
    kernels, against 1.68 and 1.36 at the others).

    The decisions, each with measured headroom:
      * grey falls at every refinement on both kernels (monotone);
      * grey gains >= 32x over the ladder (56.1x / 44.9x, 1.75x and 1.40x);
      * hard gains <= 16x (9.46x / 9.27x, 1.69x and 1.73x);
      * so the mean-order GAP is >= 0.5 (0.86 and 0.77, 1.7x and 1.5x).
    """
    for name, err_fn in sorted(KERNELS.items()):
        dx0 = WINDOW / LADDER_N[0]
        assert Z_RS > 2.0 * LADDER_N[0] * dx0 ** 2 / LAM, (
            'z_RS must clear the spatial kernel alias threshold at every N')
        gray = [err_fn(N, edge='gray') for N in LADDER_N]
        hard = [err_fn(N, edge='hard') for N in LADDER_N]
        assert all(gray[i] > gray[i + 1] for i in range(3)), (name, gray)
        g_gain = gray[0] / gray[-1]
        h_gain = hard[0] / hard[-1]
        assert g_gain >= 32.0, (name, g_gain)
        assert h_gain <= 16.0, (name, h_gain)
        g_order = np.log2(g_gain) / 3.0
        h_order = np.log2(h_gain) / 3.0
        assert g_order - h_order >= 0.5, (name, g_order, h_order)
        # the DEFAULT arm rides the grey ladder bit for bit, not merely closely
        for N in (128, 256):
            assert err_fn(N) == err_fn(N, edge='gray'), (name, N)


def test_verify_c1_the_hard_arms_non_monotonicity_is_fixture_specific():
    """SCOPE, pinned.  The CHANGELOG, the Migration-Guide and
    ``apply_aperture``'s own docstring state without qualification that "a
    circle's staircase area error does not shrink monotonically, so that arm
    has no usable order at all", and WP-C1's
    ``test_c1_the_hard_arm_is_the_arm_without_an_order_not_merely_a_worse_one``
    asserts ``min(hard step orders) < 0`` on WP-B11's optic, where the last
    refinement RISES 54 % (RS) and 53 % (HF).

    On THIS optic it does not: the hard arm falls at every step, with step
    orders 1.68 / 0.20 / 1.36 (RS) and 1.67 / 0.20 / 1.34 (HF), measured
    2026-09-20 on both builds.  So the negative order is a property of one
    (lambda, a, window, z), while the RATE gap of the test above is not.

    This id exists so the general claim is never read as universal and so a
    future re-pinning of the shipped ladder cannot quietly adopt "hard always
    rises" as a library property.  It asserts the decision both ways: on this
    optic every hard step order is positive, AND at least one of them is below
    half the grey arm's worst, which is the erratic behaviour that makes the
    hard arm's order unusable without requiring it to be negative.
    """
    for name, err_fn in sorted(KERNELS.items()):
        hard = [err_fn(N, edge='hard') for N in LADDER_N]
        gray = [err_fn(N, edge='gray') for N in LADDER_N]
        h_ord = [float(np.log2(hard[i] / hard[i + 1])) for i in range(3)]
        g_ord = [float(np.log2(gray[i] / gray[i + 1])) for i in range(3)]
        assert min(h_ord) > 0.0, (
            name, h_ord,
            'the hard arm DOES fall monotonically on this optic -- if this '
            'ever fails, the fixture moved, not the library')
        assert min(h_ord) < 0.5 * min(g_ord), (name, h_ord, g_ord)


# ===========================================================================
# V7 -- edge_samples = 4 is the knee here too, from both sides
# ===========================================================================

def test_verify_c1_four_is_the_knee_on_this_optic_from_both_sides():
    """MEASURED 2026-09-20 at N = 512 on the RS ladder of this optic, both
    builds: 2.3998e-03 / 5.0011e-04 / 1.0924e-04 / 1.1556e-04 / 1.1652e-04 at
    n_sub = 1 / 2 / 4 / 8 / 16.

    Two-sided: 2 -> 4 still gains 4.58x (bar 1.5x, 3.05x of headroom) while
    4 -> 8 gains 0.945x, i.e. it is 5.8 % WORSE (bar "less than 1.1x", 1.16x
    of headroom) -- past the knee the residual is the propagator's, not the
    mask's.  WP-C1 measured 2.04x and 0.965x on ITS optic, so the knee is at
    4 on both and the margin here is wider.

    ``edge_samples=1`` reproduces the hard reading BIT FOR BIT (n_sub = 1 IS
    the pixel-centre indicator), which is what keeps this ladder measuring the
    mask rather than something else.
    """
    e = {n: _rs_err(512, edge='gray', edge_samples=n)
         for n in (1, 2, 4, 8, 16)}
    assert _rs_err(512, edge='gray', edge_samples=1) == _rs_err(
        512, edge='hard'), 'n_sub = 1 must BE the pixel-centre indicator'
    assert e[2] / e[4] >= 1.5, e
    assert e[4] / e[8] < 1.1, e
    assert e[8] / e[16] < 1.1, e
    assert inspect.signature(
        apply_aperture).parameters['edge_samples'].default == 4


# ===========================================================================
# V8 -- the way back, bit-exact against a hand-built binary oracle
# ===========================================================================

def _binary_oracle(E, dx, shape, params, xc=0.0, yc=0.0, dy=None):
    """The pixel-centre indicator, written here rather than read from the
    library, so the comparison is against an oracle and not against the same
    code twice."""
    dy = dx if dy is None else dy
    Ny, Nx = E.shape
    x = (np.arange(Nx) - Nx / 2) * dx
    y = (np.arange(Ny) - Ny / 2) * dy
    X, Y = np.meshgrid(x, y)
    if shape == 'circular':
        m = (X - xc) ** 2 + (Y - yc) ** 2 <= (params['diameter'] / 2) ** 2
    elif shape == 'annular':
        h2 = (X - xc) ** 2 + (Y - yc) ** 2
        m = ((h2 >= (params['inner_diameter'] / 2) ** 2)
             & (h2 <= (params['outer_diameter'] / 2) ** 2))
    else:
        m = ((np.abs(X - xc) <= params['width_x'] / 2)
             & (np.abs(Y - yc) <= params['width_y'] / 2))
    return np.where(m, E, np.zeros((), dtype=E.dtype))


_WAYBACK = {
    'odd_grid_circle': (
        _field(N=97, seed=31).astype(np.complex128), 1.25e-6, 'circular',
        {'diameter': 8.3e-5}, {}),
    'decentred_anamorphic_annulus': (
        _field(N=96, seed=19), 1.25e-6, 'annular',
        {'inner_diameter': 2.1e-5, 'outer_diameter': 7.9e-5},
        {'xc': 1.7 * 1.25e-6, 'yc': 0.9 * 1.35 * 1.25e-6,
         'dy': 1.35 * 1.25e-6}),
    'anamorphic_rect': (
        _field(N=96, seed=13), 1.25e-6, 'rectangular',
        {'width_x': 7.3e-5, 'width_y': 4.1e-5}, {'dy': 1.6 * 1.25e-6}),
    'complex64': (
        _field(N=64, seed=23).astype(np.complex64), 1.25e-6, 'circular',
        {'diameter': 5.3e-5}, {}),
}


@pytest.mark.parametrize('case', sorted(_WAYBACK))
def test_verify_c1_the_way_back_is_the_binary_oracle_bit_for_bit(case):
    """``edge='hard'`` must be the pixel-centre indicator and nothing else,
    on fixtures WP-C1's own way-back test does not use: an ODD grid (where the
    ``(j - N/2)`` centring lands rim pixels differently), a decentred annulus
    on an anamorphic grid, an anamorphic rectangle, and complex64 (which must
    not be silently upcast).  Bit identity; no bar.
    """
    E, dx, shape, params, extra = _WAYBACK[case]
    got = apply_aperture(E, dx, shape, params, edge='hard', **extra)
    want = _binary_oracle(E, dx, shape, params, **extra)
    assert got.dtype == E.dtype, (case, got.dtype, E.dtype)
    assert _bytes(got) == _bytes(want), case
    # and the default is NOT that answer -- these rims all cut pixels
    default = apply_aperture(E, dx, shape, params, **extra)
    assert _bytes(default) != _bytes(want), (
        f"{case}: the default and the binary mask agree, so this fixture no "
        f"longer separates the two arms")


@pytest.mark.parametrize('fill', [np.nan, np.inf, -np.inf])
def test_verify_c1_a_blocked_pixel_is_plus_zero_on_both_arms(fill):
    """A blocked pixel must be EXACTLY ``+0.0 + 0.0j`` -- not merely ``== 0``
    -- on both arms and for any input value, because a sign bit on a zero
    flips ``atan2``, the complex ``sqrt`` branch and ``1/x``, and
    ``np.array_equal`` cannot see it.

    This is the contract the JAX slow path broke before WP-C1: measured at
    49ddf4bd on a 128 x 128 chain fixture, 3033 negative-zero real parts and
    3066 negative-zero imaginary parts, and a non-finite input left three
    non-finite values OUTSIDE the stop; both are 0 here, on both builds.
    """
    E = np.full((48, 48), complex(fill, fill), dtype=np.complex128)
    for kw in ({'edge': 'hard'}, {'edge': 'gray'}, {}):
        with np.errstate(invalid='ignore'):
            out = apply_aperture(E, 1e-6, 'circular', {'diameter': 1.7e-5},
                                 **kw)
        outside = np.zeros((48, 48), dtype=bool)
        outside[0, 0] = outside[0, -1] = outside[-1, 0] = outside[-1, -1] = True
        blocked = out[outside]
        assert np.all(blocked == 0), (kw, blocked)
        assert not np.any(np.signbit(np.real(blocked))), (kw, 'negative zero')
        assert not np.any(np.signbit(np.imag(blocked))), (kw, 'negative zero')
        assert np.all(np.isfinite(blocked)), (kw, blocked)
