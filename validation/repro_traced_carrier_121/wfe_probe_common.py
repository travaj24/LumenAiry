# Shared machinery for the DIAG_LAST_GROUP_DECENTRE probe (2026-07-30).
#
# LOCAL-ONLY (needs the 121 .zmx + the design-study runner), like every other
# script in this directory.  Nothing here is library code and nothing here is
# imported by the library.
#
# WHAT THIS PROVIDES
# ------------------
# 1. ``exact_phase_on_nodes`` -- the ARBITER.  For a set of EXIT-grid nodes it
#    solves the entrance->exit map with the EXACT skew ray trace
#    (``lumenairy.raytrace``, Zemax-validated) by Newton iteration whose
#    RESIDUAL is the exact trace (the Jacobian is finite-differenced from the
#    same trace, so the fixed point is exact whatever the Jacobian's error),
#    and returns the exit phase
#
#        Phi(X, Y) = k0 * [ W(xe, ye) + OPL_trace(xe, ye -> exit vertex) ]
#
#    which is EXACTLY the quantity ``apply_real_lens_traced`` approximates by
#    ``k0 * opl_map`` (see _lens_traced.py: the carrier entrance eikonal is
#    added to ``final.opd`` at the H6 site, the exit-vertex correction is
#    applied just above it, and the assembly uses ``phase = k0 * opl_map``).
#    The element's launch geometry is reproduced bit-for-bit: same
#    ``surfaces_from_prescription`` on the aperture-stripped prescription, same
#    exit-vertex correction, same on-axis OPL reference.
#
# 2. ``local_wfe`` -- the ALIASING-FREE estimator idiom from
#    decentred_fit_defect.py: the element's field is de-referenced against the
#    arbiter POINTWISE (no interpolation of the field, no FFT, no global
#    transform), so the exit NA vs grid Nyquist ratio is irrelevant to the
#    comparison -- the compared quantity is the DIFFERENCE of two fields that
#    carry the same fast phase, and its per-pixel step is measured and reported
#    so the caller can prove nothing folded.
#
# 3. ``capture_chain_element_calls`` -- monkeypatches
#    ``lumenairy.elements.apply_real_lens_traced`` (SCRIPT-side; no library
#    edit) so the REAL per-group element inputs/outputs of
#    ``propagate_traced_carrier_chain`` can be recorded and replayed.
import os
import sys
import warnings

import numpy as np

import _d121_common as C  # noqa: F401  (registers glasses, fixes sys.path)

import lumenairy as la                                          # noqa: E402
from lumenairy.elements import _lens_traced as _LT              # noqa: E402
from lumenairy.elements._lens_traced import TiltedCarrier       # noqa: E402
from lumenairy.raytrace import trace                            # noqa: E402
from lumenairy.raytrace.trace import (_make_bundle,             # noqa: E402
                                      surfaces_from_prescription)
from lumenairy.glass import get_glass_index                     # noqa: E402

LAM = C.LAM
K0 = 2.0 * np.pi / LAM


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
_GEOM = [None]


def geom():
    if _GEOM[0] is None:
        _GEOM[0] = C.geometry()
    return _GEOM[0]


def last_group_prescription():
    _pre, post, _gap, _per = geom()
    return post[-1]['prescription']


def element_surfaces(presc):
    """The surface list ``apply_real_lens_traced`` itself traces (aperture
    stripped -- see _lens_traced.py 'IMPORTANT: build surfaces WITHOUT the
    prescription aperture')."""
    p = dict(presc)
    p.pop('aperture_diameter', None)
    return surfaces_from_prescription(p)


# ---------------------------------------------------------------------------
# the exact arbiter
# ---------------------------------------------------------------------------
def carrier_parts(carrier, X, Y):
    """(W, L, M) of the TiltedCarrier eikonal -- the library's own routine."""
    return _LT._tilted_carrier_parts(carrier, X, Y)


def trace_forward(xe, ye, carrier, surfs):
    """Exact forward map of the element's own launch.

    Returns ``(x_out, y_out, psi, L_out, M_out, alive)`` where ``psi`` is the
    TOTAL optical path in metres, ``OPL_trace + W(xe, ye)``, at the flat exit
    VERTEX plane -- i.e. the element's ``final.opd`` after both the
    exit-vertex correction and the H6 carrier-eikonal term.
    """
    xe = np.asarray(xe, dtype=np.float64).ravel()
    ye = np.asarray(ye, dtype=np.float64).ravel()
    W, L, M = carrier_parts(carrier, xe, ye)
    rays = _make_bundle(x=xe.copy(), y=ye.copy(),
                        L=np.asarray(L, dtype=np.float64).ravel().copy(),
                        M=np.asarray(M, dtype=np.float64).ravel().copy(),
                        wavelength=LAM)
    fin = trace(rays, surfs, LAM, output_filter='last').image_rays
    n_exit = get_glass_index(surfs[-1].glass_after, LAM)
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.where(fin.alive & (np.abs(fin.N) > 1e-30), -fin.z / fin.N, 0.0)
    opd = np.asarray(fin.opd) + n_exit * t + np.asarray(W)
    xo = np.asarray(fin.x) + np.asarray(fin.L) * t
    yo = np.asarray(fin.y) + np.asarray(fin.M) * t
    return (xo, yo, opd, np.asarray(fin.L), np.asarray(fin.M),
            np.asarray(fin.alive, dtype=bool))


def axis_reference_opl(carrier, surfs):
    """The element's on-axis OPL reference (``opl_grid -= opl_grid[i_c,i_c]``
    at the (0, 0) launch sample, which exists because n_launch is odd)."""
    _x, _y, psi, _L, _M, _a = trace_forward([0.0], [0.0], carrier, surfs)
    return float(psi[0])


def exact_phase_on_nodes(Xt, Yt, carrier, surfs, guess=None, n_iter=12,
                         tol=1e-11, h=5e-7, verbose=False):
    """Solve the EXACT entrance->exit map for the entrance points that land on
    the requested exit nodes, and return the exact exit phase there.

    Returns dict with ``xe``, ``ye``, ``phi`` (rad, on-axis referenced),
    ``resid`` (m, |exit landing - requested node|), ``L``, ``M`` (exit
    direction cosines), ``ok``.

    The Newton residual is the EXACT trace; the Jacobian is a central
    finite difference of the SAME trace.  Whatever the Jacobian's error, the
    converged point satisfies ``trace(xe, ye) == (Xt, Yt)`` to ``resid``, and a
    first-order exit-direction correction removes the leftover ``resid``
    exactly to O(resid^2) -- so the returned phase is exact to well below
    1e-4 waves for any ``resid`` under a nanometre.
    """
    Xt = np.asarray(Xt, dtype=np.float64)
    Yt = np.asarray(Yt, dtype=np.float64)
    sh = Xt.shape
    xt, yt = Xt.ravel(), Yt.ravel()
    if guess is None:
        xe = xt.copy()
        ye = yt.copy()
    else:
        xe = np.asarray(guess[0], dtype=np.float64).ravel().copy()
        ye = np.asarray(guess[1], dtype=np.float64).ravel().copy()
    ok = np.ones(xe.size, dtype=bool)
    for it in range(n_iter):
        xo, yo, psi, Lo, Mo, alive = trace_forward(xe, ye, carrier, surfs)
        rx, ry = xo - xt, yo - yt
        res = np.hypot(rx, ry)
        if verbose:
            print(f"      exact-newton {it}: max|res| {np.nanmax(res):.3e} m "
                  f"median {np.nanmedian(res):.3e}")
        if np.nanmax(res) < tol:
            break
        xpx, ypx, _p, _l, _m, _a = trace_forward(xe + h, ye, carrier, surfs)
        xpy, ypy, _p, _l, _m, _a = trace_forward(xe, ye + h, carrier, surfs)
        jxx = (xpx - xo) / h
        jyx = (ypx - yo) / h
        jxy = (xpy - xo) / h
        jyy = (ypy - yo) / h
        det = jxx * jyy - jxy * jyx
        good = np.isfinite(det) & (np.abs(det) > 1e-14)
        dxe = np.where(good, (jyy * rx - jxy * ry) / np.where(good, det, 1.0),
                       0.0)
        dye = np.where(good, (-jyx * rx + jxx * ry) / np.where(good, det, 1.0),
                       0.0)
        # damp wild steps (keeps a stray pixel from throwing the trace)
        stp = np.hypot(dxe, dye)
        cap = 2.0e-3
        sc = np.where(stp > cap, cap / np.maximum(stp, 1e-30), 1.0)
        xe = xe - dxe * sc
        ye = ye - dye * sc
        ok &= good
    xo, yo, psi, Lo, Mo, alive = trace_forward(xe, ye, carrier, surfs)
    rx, ry = xo - xt, yo - yt
    resid = np.hypot(rx, ry)
    # first-order transport of the phase from the landing point to the node
    psi_node = psi - (Lo * rx + Mo * ry)
    ok &= alive & np.isfinite(psi_node)
    ref = axis_reference_opl(carrier, surfs)
    return {'xe': xe.reshape(sh), 'ye': ye.reshape(sh),
            'phi': (K0 * (psi_node - ref)).reshape(sh),
            'psi': (psi_node - ref).reshape(sh),
            'resid': resid.reshape(sh), 'L': Lo.reshape(sh),
            'M': Mo.reshape(sh), 'ok': ok.reshape(sh)}


# ---------------------------------------------------------------------------
# aliasing-free local wavefront-error estimator
# ---------------------------------------------------------------------------
def _unwrap_small(psi, keep):
    """Unwrap a residual field that is EXPECTED small, by integrating wrapped
    nearest-neighbour differences from the brightest pixel.  Returns
    ``(unwrapped, max_nn_step_rad, loop_residual_rad)``.  ``loop_residual`` is
    the max |curl| of the wrapped-difference field over unit cells: it must be
    ~0 or the field genuinely wraps and no unwrap is single-valued.
    """
    d_x = np.angle(np.exp(1j * (psi[:, 1:] - psi[:, :-1])))
    d_y = np.angle(np.exp(1j * (psi[1:, :] - psi[:-1, :])))
    kx = keep[:, 1:] & keep[:, :-1]
    ky = keep[1:, :] & keep[:-1, :]
    step = 0.0
    if kx.any():
        step = max(step, float(np.abs(d_x[kx]).max()))
    if ky.any():
        step = max(step, float(np.abs(d_y[ky]).max()))
    # curl over unit cells: dx(i,j)+dy(i,j+1)-dx(i+1,j)-dy(i,j)
    curl = (d_x[:-1, :] + d_y[:, 1:] - d_x[1:, :] - d_y[:, :-1])
    kc = keep[:-1, :-1] & keep[:-1, 1:] & keep[1:, :-1] & keep[1:, 1:]
    loop = float(np.abs(curl[kc]).max()) if kc.any() else 0.0
    # integrate: along column 0 of the kept block, then along rows
    out = np.zeros_like(psi)
    ny, nx = psi.shape
    out[:, 0] = psi[0, 0] + np.concatenate([[0.0], np.cumsum(d_y[:, 0])])
    for j in range(1, nx):
        out[:, j] = out[:, j - 1] + d_x[:, j - 1]
    return out, step, loop


def local_wfe(E_out, phi_exact, keep, X, Y, remove_tilt=True):
    """Amplitude-weighted rms of ``arg(E_out) - phi_exact`` in WAVES.

    Purely pointwise: no interpolation of the field, no FFT, no global
    derivative.  Returns a dict with ``rms_waves`` (piston [+ tilt] removed),
    ``rms_piston_only``, ``ptv_waves``, ``nn_step_rad`` (the max wrapped
    nearest-neighbour step of the RESIDUAL -- the sampling-adequacy proof:
    it must be far below pi), ``loop_rad`` (path-independence check),
    ``strehl_marechal``, ``n_pix``.
    """
    amp = np.abs(np.asarray(E_out))
    keep = keep & (amp > 0) & np.isfinite(phi_exact)
    if not keep.any():
        return None
    z = np.asarray(E_out) * np.exp(-1j * np.asarray(phi_exact))
    psi = np.angle(np.where(keep, z, 1.0))
    unw, step, loop = _unwrap_small(psi, keep)
    w = np.where(keep, amp ** 2, 0.0)
    wt = float(w.sum())
    # POWER-WEIGHTED distribution of the wrapped nearest-neighbour step: a bare
    # max is set by a handful of skirt pixels and saturates at pi even when the
    # core is clean, which is exactly how an unwrap-based rms turns into an
    # artefact.  The percentile is what says whether the unwrap is trustworthy.
    _dx1 = np.angle(np.exp(1j * (psi[:, 1:] - psi[:, :-1])))
    _dy1 = np.angle(np.exp(1j * (psi[1:, :] - psi[:-1, :])))
    _kx = keep[:, 1:] & keep[:, :-1]
    _ky = keep[1:, :] & keep[:-1, :]
    _v = np.concatenate([np.abs(_dx1[_kx]), np.abs(_dy1[_ky])])
    _ww = np.concatenate([np.minimum(w[:, 1:], w[:, :-1])[_kx],
                          np.minimum(w[1:, :], w[:-1, :])[_ky]])
    _o = np.argsort(_v)
    _cw = np.cumsum(_ww[_o])
    _cw = _cw / max(_cw[-1], 1e-300)
    step_p99 = float(_v[_o][np.searchsorted(_cw, 0.99)]) if _v.size else 0.0
    step_med = float(_v[_o][np.searchsorted(_cw, 0.50)]) if _v.size else 0.0

    def _fit(vals, basis):
        A = np.stack([b[keep] for b in basis], axis=1)
        ww = w[keep]
        Aw = A * ww[:, None]
        c = np.linalg.solve(Aw.T @ A, Aw.T @ vals[keep])
        return vals - sum(ci * b for ci, b in zip(c, basis))

    one = np.ones_like(unw)
    r_p = _fit(unw, [one])
    out = {'rms_piston_only': float(np.sqrt((w * r_p ** 2).sum() / wt))
           / (2 * np.pi),
           'nn_step_rad': step, 'nn_step_p99': step_p99,
           'nn_step_med': step_med, 'loop_rad': loop,
           'n_pix': int(keep.sum()),
           'wrapped_rms': float(np.sqrt((w * psi ** 2).sum() / wt))
           / (2 * np.pi)}
    r = _fit(unw, [one, X, Y]) if remove_tilt else r_p
    sig = float(np.sqrt((w * r ** 2).sum() / wt))
    out['rms_waves'] = sig / (2 * np.pi)
    out['ptv_waves'] = float(r[keep].max() - r[keep].min()) / (2 * np.pi)
    out['strehl_marechal'] = float(np.exp(-sig ** 2))
    # coherent overlap of the two fields over the kept support (the fidelity
    # that actually governs the downstream spot), phase only
    ph = np.exp(1j * np.where(keep, psi, 0.0))
    num = np.abs((w * ph)[keep].sum()) ** 2
    out['fidelity_phase'] = float(num / (wt ** 2))
    # UNWRAP-FREE equivalent rms: remove the best piston + tilt by iteratively
    # aligning the weighted phasor sum (no unwrap anywhere, so a skirt pixel
    # cannot poison the core the way a path-integrated unwrap can).
    F, a, b, e_rms = _fidelity_no_unwrap(psi, w, X, Y, keep)
    out['fidelity_notilt'] = F
    out['sigma_fid_waves'] = float(np.sqrt(max(-np.log(max(F, 1e-16)), 0.0))
                                   / (2 * np.pi))
    out['rms_wrapped_notilt'] = float(e_rms / (2 * np.pi))
    out['tilt_removed'] = (float(a), float(b))
    return out


def _fidelity_no_unwrap(psi, w, X, Y, keep, n_iter=40):
    """Best piston+tilt-removed coherent fidelity of ``exp(i psi)``, WITHOUT
    any phase unwrapping.

    Iterates: align the weighted phasor sum (piston), then update the tilt by
    a weighted least-squares fit of the WRAPPED residual against (X, Y).  A
    fixed point of this map is a stationary point of |sum w exp(i(psi - aX -
    bY))|, and every step touches only wrapped quantities.
    """
    a = b = 0.0
    ww = w[keep]
    Xk, Yk = X[keep], Y[keep]
    pk = psi[keep]
    A = np.stack([Xk, Yk], axis=1)
    Aw = A * ww[:, None]
    G = Aw.T @ A
    e = np.zeros_like(pk)
    for _ in range(n_iter):
        r = pk - a * Xk - b * Yk
        c = np.angle((ww * np.exp(1j * r)).sum())
        e = np.angle(np.exp(1j * (r - c)))
        try:
            d = np.linalg.solve(G, Aw.T @ e)
        except np.linalg.LinAlgError:
            break
        a += float(d[0])
        b += float(d[1])
        if (abs(float(d[0]) * np.ptp(Xk))
                + abs(float(d[1]) * np.ptp(Yk))) < 1e-9:
            break
    r = pk - a * Xk - b * Yk
    F = float(np.abs((ww * np.exp(1j * r)).sum()) ** 2 / (ww.sum() ** 2))
    e_rms = float(np.sqrt((ww * e ** 2).sum() / ww.sum()))
    return F, a, b, e_rms


# ---------------------------------------------------------------------------
# chain element-call capture (script-side monkeypatch; no library edit)
# ---------------------------------------------------------------------------
class capture_chain_element_calls(object):
    """Context manager recording every ``apply_real_lens_traced`` call the
    chain makes: ``E_in``, ``carrier``, ``dx``, kwargs and the returned field.
    """

    def __init__(self, store_fields=True):
        self.calls = []
        self._store = store_fields
        self._orig = None

    def __enter__(self):
        import lumenairy.elements as _el
        self._orig = _el.apply_real_lens_traced
        calls, store = self.calls, self._store

        def _wrapped(E_in, *a, **kw):
            rec = {'kw': dict(kw), 'args': a}
            if store:
                rec['E_in'] = np.array(E_in, copy=True)
            out = self._orig(E_in, *a, **kw)
            if store:
                rec['E_out'] = np.array(out, copy=True)
            calls.append(rec)
            return out

        _el.apply_real_lens_traced = _wrapped
        return self

    def __exit__(self, *exc):
        import lumenairy.elements as _el
        _el.apply_real_lens_traced = self._orig
        return False


def order_carrier(m, n):
    """(L, M) direction cosines of DOE order (m, n) on design 121."""
    _pre, _post, _gap, period = geom()
    return m * LAM / period, n * LAM / period


def run_chain_capture(m=-4, n=-2, rn=1024, rs=4, n_groups=6, quiet=True):
    """Run design 121's post-DOE chain for one order and return the recorded
    per-group element calls."""
    _pre, post, _gap, period = geom()
    env_doe, R_doe, dx_doe, _P = C.chain_a(n=rn, rs=rs)
    L, M = order_carrier(m, n)
    with capture_chain_element_calls() as cap:
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter('always')
            res = la.propagate_traced_carrier_chain(
                env_doe, [dict(g) for g in post[:n_groups]], LAM, dx_doe,
                r_in=TiltedCarrier(R_doe, L, M), ray_subsample=rs,
                n_workers=8, final_distance=0.0, final_leg='paraxial',
                on_decentred_fit='ignore')
    msgs = {}
    for w in wl:
        t = str(w.message)[:160]
        msgs[t] = msgs.get(t, 0) + 1
    if not quiet:
        for t, c in msgs.items():
            print(f"   [warn x{c}] {t}")
    return cap.calls, res, msgs
