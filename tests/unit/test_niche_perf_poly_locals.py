"""Perf items 1 and 2 of AUDIT_TRACED_SPEED / AUDIT_TRACED_MEMORY 2026-08-09,
pinned as IDENTITIES rather than as speeds.

Two changes are pinned here, both in ``lumenairy/elements/_lens_traced.py``:

1.  ``_ResidualEikonal._poly`` issues ONE ``np.power`` per distinct exponent
    instead of one per term per accumulator, and ``value()`` asks for a
    Hessian-free evaluation (it never reads the Hessian).  MEASURED at 57.8 %
    of one design-121 fan order's wall before the change; 3.8x on the audit's
    own probe after it.  The claim is BIT-IDENTITY, so the pre-change
    implementation is kept VERBATIM below as ``_RefEikonal`` and every output
    is compared with ``np.array_equal`` -- not ``allclose``.

2.  ``apply_real_lens_traced`` builds its wave-grid coordinate pair with
    ``np.broadcast_to`` (zero-copy read-only views) instead of
    ``np.meshgrid`` (two materialised full-grid float64 arrays, 4.295 GB at
    the n_fine = 16384 retrace leg), and frees a set of consumed full-grid
    locals at their last use instead of at function exit.  The `del`s cannot
    change a value -- but they CAN raise ``NameError`` on a path nobody
    exercised, and a read-only 0-strided view CAN take a different code path
    inside a consumer.  Both are covered by running the whole element on a
    design-121-like fixture and BYTE-comparing the exit field against a run in
    which ``np.broadcast_to`` is shimmed to materialise, i.e. against the
    memory behaviour the meshgrid form had.

Nothing here measures time; the timings live in
``docs/audits/FIX_PERF_POLY_LOCALS_2026_08_09.md``.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.elements import _lens_traced as LT
from lumenairy.elements._lens_traced import _ResidualEikonal


# ---------------------------------------------------------------------------
# 1.  The PRE-CHANGE implementation, verbatim, as the reference
# ---------------------------------------------------------------------------
class _RefEikonal(_ResidualEikonal):
    """``_ResidualEikonal`` with the 2026-08-09 pre-change ``_poly`` / ``_eval``
    / ``value`` copied in unmodified.  Copied, not imported, on purpose: the
    reference has to stay frozen when the shipped one is edited again."""

    __slots__ = ()

    def _poly(self, ex, ey):
        s = self.scale
        u = ex / s
        v = ey / s
        P = np.zeros_like(u)
        Pu = np.zeros_like(u)
        Pv = np.zeros_like(u)
        Puu = np.zeros_like(u)
        Puv = np.zeros_like(u)
        Pvv = np.zeros_like(u)
        for c, (i, j) in zip(self.coef, self.terms):
            if c == 0.0:
                continue
            P = P + c * u ** i * v ** j
            if i >= 1:
                Pu = Pu + c * i * u ** (i - 1) * v ** j
            if j >= 1:
                Pv = Pv + c * j * u ** i * v ** (j - 1)
            if i >= 2:
                Puu = Puu + c * i * (i - 1) * u ** (i - 2) * v ** j
            if i >= 1 and j >= 1:
                Puv = Puv + c * i * j * u ** (i - 1) * v ** (j - 1)
            if j >= 2:
                Pvv = Pvv + c * j * (j - 1) * u ** i * v ** (j - 2)
        return (P, Pu / s, Pv / s,
                Puu / (s * s), Puv / (s * s), Pvv / (s * s))

    def _eval(self, xq, yq):
        ex = np.asarray(xq, dtype=np.float64) - self.cx
        ey = np.asarray(yq, dtype=np.float64) - self.cy
        r = np.sqrt(ex * ex + ey * ey)
        r1 = self.r_fit
        out = r > r1
        sc = np.where(out, r1 / np.where(r > 0.0, r, 1.0), 1.0)
        cx_ = ex * sc
        cy_ = ey * sc
        P, gx, gy, hxx, hxy, hyy = self._poly(cx_, cy_)
        a = P
        ax = gx
        ay = gy
        if np.any(out):
            rs = np.where(r > 0.0, r, 1.0)
            ux = ex / rs
            uy = ey / rs
            b = gx * ux + gy * uy
            gtx = gx - b * ux
            gty = gy - b * uy
            hux = hxx * ux + hxy * uy
            huy = hxy * ux + hyy * uy
            uhu = hux * ux + huy * uy
            htx = hux - uhu * ux
            hty = huy - uhu * uy
            d = r - r1
            f = r1 / rs
            ex_x = f * gtx + b * ux + d * (f * htx + gtx / rs)
            ex_y = f * gty + b * uy + d * (f * hty + gty / rs)
            a = np.where(out, P + d * b, a)
            ax = np.where(out, ex_x, ax)
            ay = np.where(out, ex_y, ay)
        return a, ax, ay

    def value(self, xq, yq):
        return self._eval(xq, yq)[0]

    def grad(self, xq, yq):
        _a, gx, gy = self._eval(xq, yq)
        return gx, gy


# ---------------------------------------------------------------------------
# the REAL design-121 model: the term list ``_fit_residual_eikonal`` builds at
# the shipped degree cap, and the geometry of the fine retrace leg
# ---------------------------------------------------------------------------
_DEG = int(LT._REMAP_RESID_DEGREE_CAP)          # 6 on the shipped tree
_TERMS = [(i, d - i) for d in range(1, _DEG + 1) for i in range(d + 1)]
_SCALE = 3.0e-3                                  # fit disc, metres
_RFIT = 3.6e-3                                   # radial freeze circle, metres


def _models(seed=7, deg=_DEG, cx=0.0, cy=0.0, scale=_SCALE, r_fit=_RFIT,
            zero_frac=0.0):
    terms = [(i, d - i) for d in range(1, deg + 1) for i in range(d + 1)]
    rng = np.random.default_rng(seed)
    coef = rng.standard_normal(len(terms)) * 1e-7
    if zero_frac:
        coef[rng.random(len(terms)) < zero_frac] = 0.0
    args = (coef, terms, cx, cy, scale, r_fit)
    return _ResidualEikonal(*args), _RefEikonal(*args)


def _band(rows=64, n=512, pitch=(3.1e-3 * 4.0 / 16384), n_full=16384):
    """One row band of the fine retrace grid, in the shape and the ABSOLUTE
    frame ``_pip_residual_ri`` hands the model (``np.broadcast_to`` of two 1-D
    axes; the band sits off-centre, so most of it is outside the freeze)."""
    ax = (np.arange(n, dtype=np.float64) - n / 2.0) * pitch
    ay = (np.arange(rows, dtype=np.float64) - n_full / 2.0) * pitch
    return (np.ascontiguousarray(np.broadcast_to(ax[None, :], (rows, n))),
            np.ascontiguousarray(np.broadcast_to(ay[:, None], (rows, n))))


def _interior(rows=64, n=512, r_fit=_RFIT):
    """A patch entirely INSIDE the freeze circle (``np.any(out)`` False), so
    the value-only path returns the raw polynomial."""
    pitch = 0.4 * r_fit / max(n, rows)
    ax = (np.arange(n, dtype=np.float64) - n / 2.0) * pitch
    ay = (np.arange(rows, dtype=np.float64) - rows / 2.0) * pitch
    return (np.ascontiguousarray(np.broadcast_to(ax[None, :], (rows, n))),
            np.ascontiguousarray(np.broadcast_to(ay[:, None], (rows, n))))


# ---------------------------------------------------------------------------
# item 1 -- bit-identity of the power-cached, Hessian-free rewrite
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('patch', ['freeze', 'interior'])
def test_poly_six_outputs_bit_identical(patch):
    """``_poly``'s full 6-tuple is bit-identical, term list and degree as the
    shipped fit builds them."""
    new, ref = _models()
    ex, ey = _band() if patch == 'freeze' else _interior()
    got = new._poly(ex, ey)
    want = ref._poly(ex, ey)
    assert len(got) == len(want) == 6
    for k, (g, w) in enumerate(zip(got, want)):
        assert g is not None, f'output {k} must stay populated by default'
        assert np.array_equal(g, w), f'_poly output {k} moved'


@pytest.mark.parametrize('patch', ['freeze', 'interior'])
def test_value_bit_identical(patch):
    """The HOT consumer.  ``value()`` no longer builds a Hessian; the number it
    returns may not move by one ulp."""
    new, ref = _models()
    ex, ey = _band() if patch == 'freeze' else _interior()
    assert np.array_equal(new.value(ex, ey), ref.value(ex, ey))


@pytest.mark.parametrize('patch', ['freeze', 'interior'])
def test_grad_bit_identical(patch):
    """The COLD consumer's contract is untouched: ``grad`` still evaluates the
    frozen gradient, which still needs the Hessian."""
    new, ref = _models()
    ex, ey = _band() if patch == 'freeze' else _interior()
    for g, w in zip(new.grad(ex, ey), ref.grad(ex, ey)):
        assert g is not None
        assert np.array_equal(g, w)


@pytest.mark.parametrize('deg', [1, 2, 3, 4, 5, 6])
def test_bit_identical_every_degree(deg):
    """Every degree the fit can return, including the ones whose term list
    never indexes the top exponents."""
    new, ref = _models(deg=deg)
    ex, ey = _band()
    assert np.array_equal(new.value(ex, ey), ref.value(ex, ey))
    for g, w in zip(new._poly(ex, ey), ref._poly(ex, ey)):
        assert np.array_equal(g, w)


def test_bit_identical_with_zero_coefficients():
    """The ``c == 0`` skip drives the exponent census as well as the
    accumulation; a sparse fit must still land on the same bits."""
    new, ref = _models(zero_frac=0.5)
    ex, ey = _band()
    assert np.array_equal(new.value(ex, ey), ref.value(ex, ey))
    for g, w in zip(new._poly(ex, ey), ref._poly(ex, ey)):
        assert np.array_equal(g, w)
    # ...and the degenerate all-zero fit
    zero = np.zeros(len(_TERMS))
    a = _ResidualEikonal(zero, _TERMS, 0.0, 0.0, _SCALE, _RFIT)
    b = _RefEikonal(zero, _TERMS, 0.0, 0.0, _SCALE, _RFIT)
    assert np.array_equal(a.value(ex, ey), b.value(ex, ey))


def test_bit_identical_decentred_model():
    """niche C6/D9: the model is fitted in the element's ABSOLUTE frame, so it
    is routinely evaluated about a non-zero centre."""
    new, ref = _models(cx=1.7e-3, cy=-0.9e-3)
    ex, ey = _band()
    assert np.array_equal(new.value(ex, ey), ref.value(ex, ey))
    for g, w in zip(new.grad(ex, ey), ref.grad(ex, ey)):
        assert np.array_equal(g, w)


def test_hessian_free_slots_are_none_not_stale():
    """``hess=False`` must return ``None`` for the Hessian, not the interior
    Hessian a consumer could silently use as the frozen one."""
    new, _ref = _models()
    ex, ey = _band()
    out = new._poly(ex, ey, hess=False)
    assert len(out) == 6
    assert all(o is not None for o in out[:3])
    assert all(o is None for o in out[3:])
    full = new._poly(ex, ey)
    for k in range(3):
        assert np.array_equal(out[k], full[k]), f'output {k} moved with hess'
    # ...and the same for the value-only _eval
    ev = new._eval(ex, ey, need_grad=False)
    assert ev[1] is None and ev[2] is None
    assert np.array_equal(ev[0], new._eval(ex, ey)[0])


def test_default_signatures_preserved_for_cold_callers():
    """``_poly(ex, ey)`` and ``_eval(x, y)`` keep their pre-change arity and
    return shape, so a caller that never heard of the new keyword is
    unaffected."""
    new, _ref = _models()
    ex, ey = _band(rows=4, n=16)
    assert len(new._poly(ex, ey)) == 6
    a, gx, gy = new._eval(ex, ey)
    assert gx is not None and gy is not None
    assert np.shape(a) == np.shape(gx) == np.shape(gy) == np.shape(ex)


def test_scalar_and_zero_d_queries_still_work():
    """The cold ``final.opd + value(h_x, h_y)`` / ``grad(h_x, h_y)`` callers
    pass 1-D launch heights; a scalar must not raise either."""
    new, ref = _models()
    h = np.linspace(-4e-3, 4e-3, 101)
    assert np.array_equal(new.value(h, h), ref.value(h, h))
    for g, w in zip(new.grad(h, h), ref.grad(h, h)):
        assert np.array_equal(g, w)
    assert np.array_equal(np.asarray(new.value(1e-3, 2e-3)),
                          np.asarray(ref.value(1e-3, 2e-3)))


# ---------------------------------------------------------------------------
# item 2 -- broadcast instead of meshgrid, and free at last use
# ---------------------------------------------------------------------------
def test_broadcast_axes_are_elementwise_meshgrid():
    """The substitution itself: the same elements, no materialisation."""
    n = 257
    x = (np.arange(n) - n / 2) * 3.5e-6 + 1.1e-3
    y = (np.arange(n) - n / 2) * 3.5e-6 - 0.4e-3
    MX, MY = np.meshgrid(x, y)
    BX = np.broadcast_to(x[None, :], (n, n))
    BY = np.broadcast_to(y[:, None], (n, n))
    assert np.array_equal(MX, BX) and np.array_equal(MY, BY)
    assert BX.base is not None and BY.base is not None      # views, not copies
    assert BX.nbytes and BX.strides[0] == 0 and BY.strides[1] == 0
    # the two consumer idioms in apply_real_lens_traced
    sub = 4
    assert np.array_equal(MX[::sub, ::sub], BX[::sub, ::sub])
    assert np.array_equal(MX ** 2 + MY ** 2, BX ** 2 + BY ** 2)
    m = (MX ** 2 + MY ** 2) < (5e-4) ** 2
    assert np.array_equal(MX[m], BX[m]) and np.array_equal(MY[m], BY[m])


class _MaterialisingNumpy:
    """A numpy stand-in whose ``broadcast_to`` returns a MATERIALISED
    C-contiguous array -- i.e. the memory behaviour ``np.meshgrid`` had, with
    identical elements.  Everything else delegates."""

    def __init__(self, mod):
        self._mod = mod
        self.calls = 0

    def __getattr__(self, name):
        return getattr(self._mod, name)

    def broadcast_to(self, *args, **kwargs):
        self.calls += 1
        return np.ascontiguousarray(np.broadcast_to(*args, **kwargs))


# --- a design-121-like fixture: N = 1024, carrier sphere, ray-density, remap
_WL = 1.31e-6
_N = 1024
_DX = 6.0e-6
_WBEAM = 1.2e-3
_RC = -0.06                      # converging carrier conjugate, metres
_ZLEG = 15e-3
_AP = 9.0e-3
_RS = 4


def _flat():
    return {'radius': np.inf, 'glass_before': 'air', 'glass_after': 'air',
            'conic': 0.0, 'radius_y': None, 'conic_y': None,
            'aspheric_coeffs': None, 'aspheric_coeffs_y': None}


def _presc():
    return {'name': 'perf_locals_leg', 'aperture_diameter': _AP,
            'surfaces': [_flat(), _flat()], 'thicknesses': [_ZLEG]}


def _field():
    """Carrier sphere + an r^4 residual + a Gaussian envelope: the C6 residual
    fit engages (so ``_ResidualEikonal`` is on the path) and the ray-density /
    'remap' tail runs, which is where the freed locals live."""
    k0 = 2.0 * np.pi / _WL
    ax = (np.arange(_N) - _N / 2) * _DX
    X, Y = np.meshgrid(ax, ax)
    rho = np.sqrt(X * X + Y * Y + _RC * _RC)
    W = -(rho - abs(_RC))
    a = 2.0 * ((X * X + Y * Y) / _WBEAM ** 2) ** 2 / k0
    amp = np.exp(-(X * X + Y * Y) / _WBEAM ** 2)
    return (amp * np.exp(1j * k0 * (W + a))).astype(np.complex128)


def _run_traced(E, diag=None):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return np.asarray(la.apply_real_lens_traced(
            E, prescription=_presc(), wavelength=_WL, dx=_DX, carrier=_RC,
            amplitude_model='ray_density', preserve_input_phase='remap',
            remap_sampling='full', ray_subsample=_RS, parallel_amp=False,
            n_workers=1, on_undersample='silent',
            on_noncollimated='silent', fit_radius_beam_factor=2.0,
            _remap_launch_out=diag))


@pytest.mark.slow
def test_group_exit_field_byte_identical_to_materialised_grids():
    """THE pin for item 2.  One design-121-like element call, N = 1024, with
    every freed local and the broadcast coordinate pair exercised: the exit
    field must be BYTE-identical to the same call made with the coordinate
    grids materialised.

    This is also what proves the ``del``s: they are on the executed path here,
    so a name freed too early raises ``NameError`` rather than being missed.
    """
    E = _field()
    diag = {}
    got = _run_traced(E, diag=diag)
    # TEETH: the fixture must actually be on the code path this pins -- the C6
    # residual-eikonal fit engaged, at the real degree, which is what puts
    # ``_ResidualEikonal.value`` (item 1) inside this call.
    assert diag.get('engaged') is True, f'residual fit did not engage: {diag}'
    assert int(diag['degree']) == _DEG and int(diag['n_terms']) == len(_TERMS)
    shim = _MaterialisingNumpy(np)
    real = LT.np
    LT.np = shim
    try:
        ref = _run_traced(E)
    finally:
        LT.np = real
    assert shim.calls > 0, 'the shim never saw a broadcast_to -- path changed'
    assert got.shape == ref.shape and got.dtype == ref.dtype
    assert got.tobytes() == ref.tobytes(), (
        'exit field moved between broadcast views and materialised grids; '
        f'max|delta| = {np.max(np.abs(got - ref)):.3e}')
    assert np.isfinite(np.abs(got)).all()
    assert float(np.abs(got).max()) > 0.0


@pytest.mark.slow
def test_traced_call_is_repeatable_byte_for_byte():
    """Guards the freed locals from the other direction: a lifetime edit that
    let a consumer read a released buffer would show up as run-to-run drift."""
    E = _field()
    a = _run_traced(E)
    b = _run_traced(E)
    assert a.tobytes() == b.tobytes()
