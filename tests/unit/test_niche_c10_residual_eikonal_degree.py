"""Niche C10 (2026-08-02): the residual-eikonal fit degree is 6, not 4.

``_REMAP_RESID_EIKONAL_DEGREE`` is the total degree of ``a_fit``, the model of
the input residual eikonal that the niche-C6 stationary-phase launch aims
along.  C6's derivation says what survives that launch is

    1/2 grad(a - a_fit)^T H^-1 grad(a - a_fit)

-- quadratic in what the FIT MISSES.  A carrier-referenced relay's residual is
``r^4``-dominant with an ``r^6`` next term, so degree 4 spans the first and
degree 6 the second, while degree 5 adds only ODD terms a near-radial residual
has no use for.  These tests pin that FORM statement (4 ~ 5 << 6 on an ``r^6``
residual, and all three equal on a pure ``r^4`` one) rather than any particular
number, and they pin the fail-before.

Every numeric bar here is a RATIO between two arms measured in the same process
on the same fixture, or an exact-arithmetic identity -- see
``docs/audits/D121_RESIDUAL_CLOSURE_2026_08_02.md`` S8.
"""
import numpy as np
import pytest

import lumenairy.elements._lens_traced as LT
from lumenairy.elements._lens_traced import _fit_residual_eikonal

LAM = 1.31e-6
K0 = 2.0 * np.pi / LAM


def _fixture(c4, c6, n=192, dx=30e-6, w=1.4e-3):
    """A Gaussian beam carrying a KNOWN radial residual eikonal
    ``a(r) = c4 (r/w)^4 + c6 (r/w)^6`` in metres, with no carrier.

    Returns ``(E, dx, w, a_fn, grad_fn)``.
    """
    ax = (np.arange(n) - n / 2) * dx
    X, Y = np.meshgrid(ax, ax)
    r2 = (X ** 2 + Y ** 2) / (w * w)

    def a_fn(x, y):
        t = (x ** 2 + y ** 2) / (w * w)
        return c4 * t ** 2 + c6 * t ** 3

    def grad_fn(x, y):
        t = (x ** 2 + y ** 2) / (w * w)
        # d/dx of c4 t^2 + c6 t^3 with t = (x^2+y^2)/w^2
        dt = 2.0 / (w * w)
        f = (2.0 * c4 * t + 3.0 * c6 * t * t)
        return f * dt * x, f * dt * y

    amp = np.exp(-r2)
    E = amp * np.exp(1j * K0 * a_fn(X, Y))
    return E.astype(np.complex128), dx, w, a_fn, grad_fn


def _slope_err(E, dx, w, grad_fn, degree):
    """Amplitude-weighted rms error of the FITTED residual ray slope against
    the analytic one, over the bright core.  Returns ``None`` if the fit
    declines (which the callers assert against)."""
    m = _fit_residual_eikonal(E, None, LAM, dx, dx, (0.0, 0.0), w,
                              degree=degree)
    if m is None:
        return None
    n = E.shape[0]
    ax = (np.arange(n) - n / 2) * dx
    X, Y = np.meshgrid(ax, ax)
    amp = np.abs(E)
    keep = (amp > 0.05 * amp.max()) & ((X ** 2 + Y ** 2) <= (2.0 * w) ** 2)
    gx, gy = m.grad(X[keep], Y[keep])
    tx, ty = grad_fn(X[keep], Y[keep])
    wt = amp[keep] ** 2
    e2 = ((gx - tx) ** 2 + (gy - ty) ** 2)
    return float(np.sqrt((wt * e2).sum() / wt.sum()))


# ---------------------------------------------------------------------------
# the switch
# ---------------------------------------------------------------------------
def test_the_shipped_degree_is_six_and_the_cap_permits_it():
    assert LT._REMAP_RESID_EIKONAL_DEGREE == 6
    assert LT._REMAP_RESID_DEGREE_CAP >= LT._REMAP_RESID_EIKONAL_DEGREE


def test_the_default_is_read_from_the_module_constant():
    """The fail-before is the constant itself: ``degree=None`` must resolve to
    it, so setting it back to 4 restores the pre-C10 model exactly."""
    E, dx, w, _a, _g = _fixture(3.0e-7, 1.2e-7)
    m_default = _fit_residual_eikonal(E, None, LAM, dx, dx, (0.0, 0.0), w)
    m_six = _fit_residual_eikonal(E, None, LAM, dx, dx, (0.0, 0.0), w,
                                  degree=6)
    m_four = _fit_residual_eikonal(E, None, LAM, dx, dx, (0.0, 0.0), w,
                                   degree=4)
    assert m_default is not None and m_six is not None and m_four is not None
    assert np.array_equal(m_default.coef, m_six.coef)
    old = LT._REMAP_RESID_EIKONAL_DEGREE
    try:
        LT._REMAP_RESID_EIKONAL_DEGREE = 4
        m_fb = _fit_residual_eikonal(E, None, LAM, dx, dx, (0.0, 0.0), w)
        assert np.array_equal(m_fb.coef, m_four.coef)
    finally:
        LT._REMAP_RESID_EIKONAL_DEGREE = old
    # and it really is a switch on this fixture
    assert m_six.coef.shape != m_four.coef.shape


def test_the_cap_clamps_a_higher_request():
    E, dx, w, _a, _g = _fixture(3.0e-7, 1.2e-7)
    m_cap = _fit_residual_eikonal(E, None, LAM, dx, dx, (0.0, 0.0), w,
                                  degree=LT._REMAP_RESID_DEGREE_CAP)
    m_over = _fit_residual_eikonal(E, None, LAM, dx, dx, (0.0, 0.0), w,
                                   degree=LT._REMAP_RESID_DEGREE_CAP + 4)
    assert m_cap is not None and m_over is not None
    assert np.array_equal(m_cap.coef, m_over.coef)


# ---------------------------------------------------------------------------
# the FORM argument: 4 ~ 5 << 6 on an r^6 residual, all equal on a pure r^4 one
# ---------------------------------------------------------------------------
def test_degree_six_models_an_r6_residual_far_better_than_four():
    """The point of the change, as a measurement.  Scored as a RATIO between
    the two arms on one fixture, with the degree-4 arm's own liveness
    asserted so a dead fixture cannot pass this."""
    E, dx, w, _a, grad_fn = _fixture(3.0e-7, 1.2e-7)
    e4 = _slope_err(E, dx, w, grad_fn, 4)
    e6 = _slope_err(E, dx, w, grad_fn, 6)
    assert e4 is not None and e6 is not None
    # LIVENESS: degree 4 must actually be missing something on this fixture --
    # against the residual's own slope scale, not against an absolute bar.
    n = E.shape[0]
    ax = (np.arange(n) - n / 2) * dx
    X, Y = np.meshgrid(ax, ax)
    tx, ty = grad_fn(X, Y)
    amp = np.abs(E)
    keep = amp > 0.05 * amp.max()
    scale = float(np.sqrt(((amp[keep] ** 2)
                           * (tx[keep] ** 2 + ty[keep] ** 2)).sum()
                          / (amp[keep] ** 2).sum()))
    assert scale > 0.0
    assert e4 > 0.02 * scale, "the degree-4 arm is not live on this fixture"
    assert e6 < 0.2 * e4


def test_degree_five_buys_nothing_on_a_radial_residual():
    """Degree 5 adds only ODD terms, which a radial residual has no use for --
    so it must land on degree 4, NOT part-way to degree 6.  This is what makes
    the change a statement about the residual's FORM rather than about
    resolution."""
    E, dx, w, _a, grad_fn = _fixture(3.0e-7, 1.2e-7)
    e4 = _slope_err(E, dx, w, grad_fn, 4)
    e5 = _slope_err(E, dx, w, grad_fn, 5)
    e6 = _slope_err(E, dx, w, grad_fn, 6)
    assert e4 is not None and e5 is not None and e6 is not None
    assert abs(e5 - e4) < 0.25 * e4          # 5 sits on 4 ...
    assert e6 < 0.2 * e4                      # ... while 6 does not


def test_a_pure_r4_residual_is_indifferent_to_the_raise():
    """Where degree 4 already spans the residual, degree 6 must not make it
    worse -- the raise is a superset, not a different model."""
    E, dx, w, _a, grad_fn = _fixture(3.0e-7, 0.0)
    e4 = _slope_err(E, dx, w, grad_fn, 4)
    e6 = _slope_err(E, dx, w, grad_fn, 6)
    assert e4 is not None and e6 is not None
    n = E.shape[0]
    ax = (np.arange(n) - n / 2) * dx
    X, Y = np.meshgrid(ax, ax)
    tx, ty = grad_fn(X, Y)
    amp = np.abs(E)
    keep = amp > 0.05 * amp.max()
    scale = float(np.sqrt(((amp[keep] ** 2)
                           * (tx[keep] ** 2 + ty[keep] ** 2)).sum()
                          / (amp[keep] ** 2).sum()))
    # both arms model it to well under a percent of the residual's own slope
    assert e4 < 0.01 * scale
    assert e6 < 0.01 * scale
    # and the raise does not degrade it by more than the fit's own noise
    assert e6 < 3.0 * max(e4, 1e-300)


def test_the_model_stays_curl_free_at_the_raised_degree():
    """``a_fit`` is a scalar POTENTIAL, so its gradient must be curl-free to
    machine precision whatever the degree -- the necessary condition for the
    launched bundle to be a congruence at all."""
    E, dx, w, _a, _g = _fixture(3.0e-7, 1.2e-7)
    m = _fit_residual_eikonal(E, None, LAM, dx, dx, (0.0, 0.0), w)
    assert m is not None
    h = 0.2 * dx
    t = np.linspace(-0.8 * w, 0.8 * w, 15)
    X, Y = np.meshgrid(t, t)
    gxp, _ = m.grad(X, Y + h)
    gxm, _ = m.grad(X, Y - h)
    _, gyp = m.grad(X + h, Y)
    _, gym = m.grad(X - h, Y)
    curl = (gyp - gym) / (2 * h) - (gxp - gxm) / (2 * h)
    gsc = float(np.abs(m.grad(X, Y)[0]).max()) / w
    assert float(np.abs(curl).max()) < 1e-6 * max(gsc, 1e-300)


@pytest.mark.parametrize('deg', [4, 6])
def test_the_potential_and_its_gradient_are_one_field(deg):
    """``value`` and ``grad`` must be the SAME polynomial -- a modified
    gradient would break the congruence.  Exact-arithmetic check by central
    differences of ``value`` against ``grad``, as a ratio to the gradient's own
    magnitude."""
    E, dx, w, _a, _g = _fixture(3.0e-7, 1.2e-7)
    m = _fit_residual_eikonal(E, None, LAM, dx, dx, (0.0, 0.0), w, degree=deg)
    assert m is not None
    h = 0.05 * w
    t = np.linspace(-0.6 * w, 0.6 * w, 11)
    X, Y = np.meshgrid(t, t)
    fx = (m.value(X + h, Y) - m.value(X - h, Y)) / (2 * h)
    fy = (m.value(X, Y + h) - m.value(X, Y - h)) / (2 * h)
    gx, gy = m.grad(X, Y)
    sc = float(np.sqrt((gx ** 2 + gy ** 2).max()))
    assert sc > 0.0
    assert float(np.abs(fx - gx).max()) < 1e-2 * sc
    assert float(np.abs(fy - gy).max()) < 1e-2 * sc
