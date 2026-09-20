"""H2-3 (item 20) -- the near-focus exact-kernel table, and the envelope
bookkeeping it needs first.

Run as a CHILD process bound to ONE tree:

    python probe_near_focus.py <tree> <out.json>

WHAT WP-B11 SECTION 2.20 LEFT.  The fixture was built (a converging Gaussian
carrier, ``f = 20 mm``, ``w0 = 15.915 um``, ``theta = 20.0 mrad``, evaluated
1 um .. 5 mm short of the geometric focus) and the dropped quartic
``k |z_eff| theta^4 / 8`` was computed and does span the decision, but the
field comparison was not valid: ``propagate_carrier_referenced`` takes and
returns an ENVELOPE referenced to a carrier, and two spellings of the reference
bookkeeping gave O(1) residuals and then a ``ValueError``.  Publishing a table
from an unvalidated fixture would be worse than publishing none, so what this
probe does FIRST is validate the bookkeeping on two cases with a known answer,
and only then publish the table.

THE FIXTURE, derived rather than quoted
---------------------------------------
Everything below follows from three numbers and the Gaussian-beam ``q``:

* ``w0 = 15.915 um`` and ``theta = 20.0 mrad`` fix the wavelength, because
  ``theta = lambda / (pi w0)``: ``lambda = pi w0 theta = 1.0000e-06 m``.
* ``zR = pi w0^2 / lambda = 795.77 um``.
* ``f = 20 mm`` places the INPUT plane one focal length before the waist, so
  the input beam parameter is ``q_in = -f - i zR`` (this library's
  ``exp(-i omega t)`` pairing, ``1/q = 1/R + i lambda/(pi w^2)`` -- see
  :func:`_q_field`), giving ``R_in = -(f^2 + zR^2)/f`` and
  ``w_in = sqrt(lambda (f^2 + zR^2)/(pi zR))``.

The INPUT ENVELOPE is then EXACTLY the real Gaussian ``exp(-r^2/w_in^2)``:
dividing ``exp(i k r^2/(2 q_in))`` by the carrier ``exp(i k r^2/(2 R_in))``
leaves ``exp(-r^2/w_in^2)`` and nothing else, because ``1/q_in`` splits into
its real part (the carrier) and its imaginary part (the amplitude) with no
cross term.  That is the whole bookkeeping, and it is why the fixture needs no
fitted carrier: the carrier handed to the propagator IS the beam's own
wavefront, so the envelope is real and positive and the residual is the
propagator's.

THE ORACLE is the whole-function ``q`` form, in the convention WP-B11 section
2.17 established for this library (``exp(+i k z)`` forward, Gouy a
RETARDATION):

    E(r, z) = exp(i k z) / (1 + z/q_in) * exp(i k r^2 / (2 (q_in + z)))

-- amplitude, curvature and Gouy phase all the argument of ONE complex number,
so the two halves cannot be carried in different conventions.  It carries the
absolute piston, so the comparisons below are PISTON-INCLUDED: a convention
error shows up as an O(1) residual rather than cancelling.

SECTIONS OF THE OUTPUT JSON
---------------------------
``bookkeeping``
    The two known-answer cases -- a COLLIMATED Gaussian and the converging one
    FAR from focus -- each spelled the two ways the returned object can be
    read (``carrier_out=inf``, which returns the reconstructed FIELD on the
    chosen lattice; and the default geometric carrier, reconstructed with
    :func:`carrier_referenced_reconstruct`).  Both must land at the oracle
    floor; the arm that does not is the spelling that was wrong.
``floor``
    The oracle's own error floor, derived and measured: ``exp(i k z)`` at
    ``k z ~ 1.3e5`` rad carries ``eps * k z`` of representation error before
    any physics, and the grid's own truncation of the Gaussian tail carries
    ``exp(-(N dx/2)^2/w^2)``.  Both are computed per case, and the bar every
    bookkeeping claim is made against is derived from them.
``table``
    ``gap_kernel`` in {'auto', 'fresnel', 'exact'} x transport in
    {'sziklas', 'collins'} x the distance-to-focus ladder, with the relative
    L2 and max-abs error against the oracle, the dropped quartic
    ``k |z_eff| theta^4 / 8`` against ``_GAP_ENV_PHI_TOL_DEFAULT``, the
    resolved kernel, the Kelly readings and every warning raised.
``env``
    Build, versions.
"""
from __future__ import annotations

import json
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_TREE = os.path.abspath(sys.argv[1])
sys.path.insert(0, _TREE)

import hlib  # noqa: E402

import numpy as np  # noqa: E402

hlib.anchor(_TREE)

from lumenairy.propagators.carrier import (  # noqa: E402
    _COLLINS_TAIL_FRAC,
    _GAP_ENV_PHI_TOL_DEFAULT,
    _collins_transport,
    carrier_referenced_reconstruct,
    propagate_carrier_referenced,
)

# --- the fixture, derived ---------------------------------------------------
W0 = 15.915e-6
THETA = 20.0e-3
LAM = float(np.pi * W0 * THETA)
ZR = float(np.pi * W0 ** 2 / LAM)
F = 20.0e-3
K = 2.0 * np.pi / LAM

#: distance-to-focus ladder, 1 um .. 5 mm (WP-B11 sec 2.20's span)
D_LADDER = (1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 5e-3)

N_IN = 512
#: Input pitch.  ``w_in`` is 400 um, so 512 x 8 um = 4.096 mm is 10.2 w_in of
#: window -- the Gaussian tail is down to exp(-(2.048/0.4)^2) = 1.3e-12 at the
#: edge, which is the truncation term of the derived floor below.
DX_IN = 8e-6


def _q_in():
    """``q`` at the input plane: the waist sits ``F`` further on."""
    return complex(-F, -ZR)


def W0_of(q):
    """The WAIST width of the beam whose parameter is ``q`` anywhere along it:
    ``w0 = sqrt(lambda * Im(q) magnitude / pi)``, since ``q = z + i zR`` up to
    the sign convention and ``zR = pi w0^2 / lambda``."""
    return float(np.sqrt(LAM * abs(np.imag(q)) / np.pi))


def _w_of_q(q):
    return float(np.sqrt(LAM / (np.pi * np.imag(1.0 / q))))


def _R_of_q(q):
    re = float(np.real(1.0 / q))
    return float('inf') if re == 0.0 else 1.0 / re


def _q_field(xo, yo, q_in, z):
    """The oracle: the analytic Gaussian at distance ``z`` from the plane where
    the beam parameter is ``q_in``, on the grid ``xo`` x ``yo``.

    ``E(r, z) = exp(i k z)/(1 + z/q_in) * exp(i k r^2/(2 (q_in + z)))``, with
    ``1/q = 1/R + i lambda/(pi w^2)`` -- this library's ``exp(-i omega t)``
    pairing.  One complex number carries amplitude, curvature and Gouy, so the
    three cannot disagree with each other (WP-B11 sec 2.17).  PISTON INCLUDED.
    """
    q2 = q_in + z
    xx, yy = np.meshgrid(xo, yo, indexing='xy')
    r2 = xx ** 2 + yy ** 2
    return (np.exp(1j * K * z) / (1.0 + z / q_in)
            * np.exp(1j * K * r2 / (2.0 * q2)))


def _axis(n, d):
    return (np.arange(int(n), dtype=np.float64) - int(n) / 2) * float(d)


def _errors(got, ref):
    got = np.asarray(got)
    ref = np.asarray(ref)
    den = float(np.linalg.norm(ref))
    peak = float(np.max(np.abs(ref)))
    return {
        'rel_L2': float(np.linalg.norm(got - ref) / den) if den else None,
        'max_abs_over_peak': (float(np.max(np.abs(got - ref)) / peak)
                              if peak else None),
        'peak_ref': peak,
        'peak_got': float(np.max(np.abs(got))),
    }


def _oracle_floor(z, n_out, dx_out, w_out):
    """The bar, DERIVED, in two terms that are added.

    1. ``eps * k * |z|`` -- the representation error of the absolute piston
       ``exp(i k z)`` before any physics is done.  At ``z = 20 mm`` and
       ``lambda = 1 um`` that is ``2.2e-16 * 1.26e5 = 2.8e-11`` rad.
    2. ``exp(-(N dx / 2)^2 / w^2)`` -- the Gaussian tail the grid truncates,
       relative to peak.  The propagator transports a truncated beam; the
       oracle does not, so their difference cannot be smaller than this.

    Ten times their sum is the bar every bookkeeping claim below is made
    against.  Ten, not one, because the sum is a floor and not an estimate of
    the residual: the FFT chain in between adds its own ``eps * log2(N)``
    rounding on a quantity of order 1.  The gap that matters is the one to the
    failure mode -- a wrong carrier spelling gives an O(1) residual, eight to
    ten decades above this -- and that gap is stated per row.
    """
    eps = float(np.finfo(np.float64).eps)
    piston = eps * K * abs(float(z))
    edge = float(n_out) * float(dx_out) / 2.0
    trunc = float(np.exp(-(edge / w_out) ** 2)) if w_out > 0 else 1.0
    return {'piston_term': piston, 'truncation_term': trunc,
            'floor': piston + trunc, 'bar': 10.0 * (piston + trunc)}


def _run(env_in, R_in, z, *, transport, gap_kernel, dx_out=None,
         carrier_out=None, on_collins_sampling='warn'):
    """One propagator call, with every warning it raises captured in order."""
    kw = dict(transport=transport, gap_kernel=gap_kernel)
    if transport == 'collins':
        kw['on_collins_sampling'] = on_collins_sampling
        if dx_out is not None:
            kw['dx_out'] = float(dx_out)
        if carrier_out is not None:
            kw['carrier_out'] = carrier_out
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            out = propagate_carrier_referenced(
                env_in, R_in, float(z), LAM, DX_IN, **kw)
            err = None
        except Exception as exc:               # noqa: BLE001 -- recorded
            out = None
            err = f'{type(exc).__name__}: {exc}'
    msgs = [(w.category.__name__, str(w.message)) for w in caught]
    return out, err, msgs


# ===========================================================================
# 1.  Bookkeeping, on two cases with a known answer
# ===========================================================================

def measure_bookkeeping():
    """The two spellings of the reference bookkeeping, on two cases whose
    answer is known independently.

    CASE A -- a COLLIMATED Gaussian (``R = inf``, waist at the input plane).
    There is no carrier at all, so the envelope IS the field and the only way
    to get an O(1) residual is to apply a carrier that should not be there.

    CASE B -- the fixture's converging Gaussian, stopped FAR from focus (5 mm
    short, i.e. 6.3 Rayleigh ranges).  The carrier is finite and the co-moving
    grid magnifies, so this is the case that distinguishes the two spellings.

    SPELLINGS.  ``field_via_carrier_out_inf`` asks the Collins transport for
    ``carrier_out=inf``, which the docstring says returns the reconstructed
    FIELD on the chosen lattice -- so the returned ``env`` is compared to the
    oracle directly and ``R`` must come back ``inf``.
    ``field_via_reconstruct`` takes the default geometric carrier and rebuilds
    the field with :func:`carrier_referenced_reconstruct` on the returned
    ``(R, dx)``.  Both must agree with the oracle and with each other.
    """
    out = {}
    eps_ = float(np.finfo(np.float64).eps)

    # ---- CASE A: collimated -------------------------------------------
    q_a = complex(0.0, -ZR)                      # waist AT the input plane
    w_a = _w_of_q(q_a)
    z_a_ = 1.5 * ZR
    w_a_out = _w_of_q(q_a + z_a_)
    # Window = 10 output widths, so the truncated tail is exp(-25) = 1.4e-11 of
    # peak and the floor below is the PISTON term, not the grid edge.  (At 16
    # input widths -- the first spelling -- the beam had grown to 4.1 widths and
    # the tail at the edge was 2.3e-02, which swamped everything.)
    dx_a = 10.0 * w_a_out / N_IN
    x_a = _axis(N_IN, dx_a)
    env_a = np.exp(-(x_a[None, :] ** 2 + x_a[:, None] ** 2)
                   / w_a ** 2).astype(np.complex128)
    z_a = z_a_
    # BOTH kernels, and the reason is the point.  The oracle is the analytic
    # Gaussian, which is a PARAXIAL solution, so ``gap_kernel='fresnel'`` -- the
    # paraxial kernel -- is the arm that can land at the oracle's floor.
    # ``'auto'`` resolves to the EXACT kernel, whose departure from the
    # paraxial truth is the beam's own dropped quartic ``k z theta^4/8``.  The
    # oracle cannot referee which of the two is more physical; what it CAN do
    # is show that the bookkeeping is right (the fresnel arm at the floor) and
    # that the exact arm's departure is the size theory says (the auto arm at
    # the quartic).  Both are recorded, with the prediction beside them.
    w_out = _w_of_q(q_a + z_a)
    theta_a = LAM / (np.pi * W0_of(q_a))
    for gk in ('fresnel', 'auto'):
        row = {'case': 'collimated', 'gap_kernel': gk, 'w_in': w_a,
               'dx_in': dx_a, 'z': z_a, 'R_in': float('inf'), 'zR': ZR,
               'lambda': LAM, 'w_out': w_out,
               'theta_of_beam': theta_a,
               'predicted_quartic_k_z_theta4_over_8':
                   K * abs(z_a) * theta_a ** 4 / 8.0}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            got = propagate_carrier_referenced(env_a, float('inf'), z_a, LAM,
                                               dx_a, gap_kernel=gk)
        row['warnings'] = [(w.category.__name__, str(w.message))
                           for w in caught]
        row['R_out'] = float(got.R)
        row['dx_out'] = float(got.dx)
        fld = carrier_referenced_reconstruct(got.env, got.R, LAM, got.dx)
        xo = _axis(N_IN, got.dx)
        row.update(_errors(fld, _q_field(xo, xo, q_a, z_a)))
        row['floor'] = _oracle_floor(z_a, N_IN, got.dx, w_out)
        out[f'A_collimated_{gk}'] = row

    # ---- CASE B: converging, far from focus ---------------------------
    q_b = _q_in()
    R_in = _R_of_q(q_b)
    w_in = _w_of_q(q_b)
    x_in = _axis(N_IN, DX_IN)
    env_in = np.exp(-(x_in[None, :] ** 2 + x_in[:, None] ** 2)
                    / w_in ** 2).astype(np.complex128)
    d_far = 5e-3
    z_b = F - d_far
    q_out = q_b + z_b
    w_out = _w_of_q(q_out)
    R_out_geo = R_in + z_b
    base = {'case': 'converging_far', 'w_in': w_in, 'R_in': R_in,
            'dx_in': DX_IN, 'z': z_b, 'd_to_focus': d_far,
            'w_out': w_out, 'R_out_geometric': R_out_geo,
            'R_out_exact_gaussian': _R_of_q(q_out), 'zR': ZR}

    # spelling 1 -- carrier_out=inf, the FIELD on a chosen lattice
    dx_out = _pitch_for(w_out, _R_of_q(q_out))
    o, err, msgs = _run(env_in, R_in, z_b, transport='collins',
                        gap_kernel='fresnel', dx_out=dx_out,
                        carrier_out=float('inf'))
    r1 = dict(base, spelling='field_via_carrier_out_inf', dx_out_asked=dx_out,
              error=err, warnings=msgs)
    if o is not None:
        r1['R_returned'] = float(o.R)
        r1['dx_returned'] = float(o.dx)
        xo = _axis(N_IN, float(o.dx))
        r1.update(_errors(o.env, _q_field(xo, xo, q_b, z_b)))
        r1['floor'] = _oracle_floor(z_b, N_IN, float(o.dx), w_out)
    out['B_far_field_via_carrier_out_inf'] = r1

    # spelling 2 -- default geometric carrier, then reconstruct
    o, err, msgs = _run(env_in, R_in, z_b, transport='collins',
                        gap_kernel='fresnel', dx_out=dx_out)
    r2 = dict(base, spelling='field_via_reconstruct', dx_out_asked=dx_out,
              error=err, warnings=msgs)
    if o is not None:
        r2['R_returned'] = float(o.R)
        r2['dx_returned'] = float(o.dx)
        fld = carrier_referenced_reconstruct(o.env, o.R, LAM, o.dx)
        xo = _axis(N_IN, float(o.dx))
        r2.update(_errors(fld, _q_field(xo, xo, q_b, z_b)))
        r2['floor'] = _oracle_floor(z_b, N_IN, float(o.dx), w_out)
    out['B_far_field_via_reconstruct'] = r2

    # spelling 3 -- the SZIKLAS transport, whose pitch is forced
    o, err, msgs = _run(env_in, R_in, z_b, transport='sziklas',
                        gap_kernel='fresnel')
    r3 = dict(base, spelling='sziklas_via_reconstruct', error=err,
              warnings=msgs)
    if o is not None:
        r3['R_returned'] = float(o.R)
        r3['dx_returned'] = float(o.dx)
        fld = carrier_referenced_reconstruct(o.env, o.R, LAM, o.dx)
        xo = _axis(N_IN, float(o.dx))
        r3.update(_errors(fld, _q_field(xo, xo, q_b, z_b)))
        r3['floor'] = _oracle_floor(z_b, N_IN, float(o.dx), w_out)
    out['B_far_sziklas'] = r3

    # THE FALSIFICATION ARM: the WRONG spelling, so the reader can see what an
    # O(1) residual looks like and that the bar has decades of room.
    o, err, msgs = _run(env_in, R_in, z_b, transport='collins',
                        gap_kernel='fresnel', dx_out=dx_out,
                        carrier_out=float('inf'))
    if o is not None:
        xo = _axis(N_IN, float(o.dx))
        wrong = carrier_referenced_reconstruct(
            o.env, R_out_geo, LAM, float(o.dx))    # a carrier applied TWICE
        rw = dict(base, spelling='WRONG_double_carrier')
        rw.update(_errors(wrong, _q_field(xo, xo, q_b, z_b)))
        out['B_far_WRONG_double_carrier'] = rw

    out['_eps'] = eps_
    return out


#: How many beam widths of OUTPUT WINDOW the table asks for.  Six, so the
#: Gaussian tail at the window edge is ``exp(-3^2) = 1.2e-04`` of peak -- small
#: enough not to dominate the residual, and small enough that the window stays
#: well inside the chirp-Z's own spatial period ``lambda |B| / dx_in`` (the K3
#: condition).  The first spelling of this probe asked only that the pitch
#: resolve the field and let the window follow from ``N_in``, which put a
#: 6.5 mm window on a 1.9 mm period: K3 = 3.46, and the residual came back
#: 2.83 -- a wrapped answer, not a wrong bookkeeping.
_WINDOW_WIDTHS = 6.0


def _pitch_for(w, R, n=N_IN):
    """An output pitch that Nyquist-resolves BOTH halves of the field at a
    plane where the beam has width ``w`` and wavefront radius ``R``, on a grid
    of ``n`` samples whose WINDOW is ``_WINDOW_WIDTHS`` widths.

    Three constraints, and the pitch is the smallest:

    * the WINDOW, ``_WINDOW_WIDTHS * w / n`` -- the Collins transport takes the
      sample count from the input, so the pitch is the only handle on the
      window, and a window past the chirp-Z's spatial period returns wrapped
      copies (K3);
    * the AMPLITUDE, ``w/8``;
    * the CURVATURE, ``exp(i k r^2/(2R))``, whose local fringe period is
      ``lambda |R| / r``; out to ``r = 2w`` that asks for
      ``lambda |R| / (4 w)``.

    Derived per plane rather than pinned, so no plane inherits another's
    sampling.
    """
    p_win = _WINDOW_WIDTHS * w / float(n)
    p_amp = w / 8.0
    p_curv = (LAM * abs(R) / (4.0 * w)) if np.isfinite(R) else p_amp
    return float(min(p_win, p_amp, p_curv))


# ===========================================================================
# 2.  The table
# ===========================================================================

def measure_table():
    q_b = _q_in()
    R_in = _R_of_q(q_b)
    w_in = _w_of_q(q_b)
    x_in = _axis(N_IN, DX_IN)
    env_in = np.exp(-(x_in[None, :] ** 2 + x_in[:, None] ** 2)
                    / w_in ** 2).astype(np.complex128)
    rows = []
    for d in D_LADDER:
        z = F - d
        q_out = q_b + z
        w_out = _w_of_q(q_out)
        R_out_exact = _R_of_q(q_out)
        R_out_geo = R_in + z
        # the reduced-frame distance the exact kernel refinement lives on
        A = R_out_geo / R_in
        z_eff = z / A if A != 0.0 else float('inf')
        quartic = K * abs(z_eff) * THETA ** 4 / 8.0
        dx_out = _pitch_for(w_out, R_out_exact)
        for transport in ('sziklas', 'collins'):
            for gk in ('auto', 'fresnel', 'exact'):
                row = {'d_to_focus': d, 'z': z, 'transport': transport,
                       'gap_kernel': gk, 'w_out': w_out,
                       'R_out_exact': R_out_exact,
                       'R_out_geometric': R_out_geo, 'A': A, 'z_eff': z_eff,
                       'quartic_k_zeff_theta4_over_8': quartic,
                       'gap_env_phi_tol_default': _GAP_ENV_PHI_TOL_DEFAULT,
                       'quartic_over_tol':
                           quartic / _GAP_ENV_PHI_TOL_DEFAULT,
                       'dx_out_asked': (dx_out if transport == 'collins'
                                        else None)}
                if transport == 'collins':
                    o, err, msgs = _run(env_in, R_in, z, transport=transport,
                                        gap_kernel=gk, dx_out=dx_out,
                                        carrier_out=float('inf'))
                else:
                    o, err, msgs = _run(env_in, R_in, z, transport=transport,
                                        gap_kernel=gk)
                row['error'] = err
                row['warnings'] = msgs
                if o is not None:
                    row['R_returned'] = (float(o.R) if np.isscalar(o.R)
                                         or isinstance(o.R, float) else o.R)
                    row['dx_returned'] = float(o.dx)
                    xo = _axis(N_IN, float(o.dx))
                    if transport == 'collins':
                        fld = np.asarray(o.env)
                    else:
                        fld = carrier_referenced_reconstruct(
                            o.env, o.R, LAM, o.dx)
                    row.update(_errors(fld, _q_field(xo, xo, q_b, z)))
                    row['floor'] = _oracle_floor(z, N_IN, float(o.dx), w_out)
                rows.append(row)
                print(f"[row] d={d:.1e} {transport:8s} {gk:8s} "
                      f"relL2={row.get('rel_L2')} err={err}", file=sys.stderr)
    return rows


def measure_kernel_resolution():
    """WHICH kernel ``gap_kernel='auto'`` actually resolves to, at each rung.

    The public entry point does not expose the transport's sampling statistics,
    so this drives ``_collins_transport`` directly with ``stats_out`` and reads
    back the three numbers the decision is made from: ``k4`` (the wrap ratio of
    the exact kernel's impulse response over the reduced frame, which is the
    gate -- ``k4 <= 1`` takes 'exact'), the measured envelope angular
    half-widths, and the resolved ``kernel``.

    This is the measurement VERIFY-B4 F3's open question needs: does 'auto'
    EVER fall back to 'fresnel' near a focus today?
    """
    q_b = _q_in()
    R_in = _R_of_q(q_b)
    w_in = _w_of_q(q_b)
    x_in = _axis(N_IN, DX_IN)
    env_in = np.exp(-(x_in[None, :] ** 2 + x_in[:, None] ** 2)
                    / w_in ** 2).astype(np.complex128)
    rows = []
    for d in D_LADDER:
        z = F - d
        q_out = q_b + z
        w_out = _w_of_q(q_out)
        dx_out = _pitch_for(w_out, _R_of_q(q_out))
        for gk in ('auto', 'fresnel', 'exact'):
            st = {}
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                try:
                    _collins_transport(
                        env_in, R_in, z, LAM, DX_IN, DX_IN,
                        dx_out=dx_out, dy_out=dx_out,
                        N_out_x=N_IN, N_out_y=N_IN, R_ref=float('inf'),
                        gap_kernel=gk, on_collins_sampling='warn',
                        stats_out=st, check_period=True)
                    err = None
                except Exception as exc:       # noqa: BLE001 -- recorded
                    err = f'{type(exc).__name__}: {exc}'
            rows.append({
                'd_to_focus': d, 'gap_kernel_asked': gk, 'error': err,
                'kernel_resolved': st.get('kernel'),
                'k4': st.get('k4'),
                'k1': st.get('k1'), 'k2': st.get('k2'), 'k3': st.get('k3'),
                'theta_x': st.get('theta_x'), 'theta_y': st.get('theta_y'),
                'r_x': st.get('r_x'), 'r_y': st.get('r_y'),
                'abcd': st.get('abcd'),
                'tail_frac': _COLLINS_TAIL_FRAC,
                'warnings': [(w.category.__name__, str(w.message)[:160])
                             for w in caught],
            })
    return rows


def _departure(env_in, R_in, z, dx_out):
    """``rel L2`` between the 'exact' and 'fresnel' arms of ONE Collins leg,
    with the measured envelope angle and reduced distance that produced it.

    The two arms are compared with EACH OTHER, never with the paraxial oracle:
    the oracle is a paraxial solution, so it cannot referee which of the two
    kernels is more physical.  What it can referee -- and does, in
    :func:`measure_table` -- is that the 'fresnel' arm reproduces the paraxial
    truth at the oracle floor, which is what makes this difference the exact
    kernel's own departure and not a bug in either.
    """
    st = {}
    out = {}
    for gk in ('exact', 'fresnel'):
        o, err, msgs = _run(env_in, R_in, z, transport='collins',
                            gap_kernel=gk, dx_out=dx_out,
                            carrier_out=float('inf'))
        out[gk] = (o, err, msgs)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            _collins_transport(env_in, R_in, z, LAM, DX_IN, DX_IN,
                               dx_out=dx_out, dy_out=dx_out,
                               N_out_x=N_IN, N_out_y=N_IN,
                               R_ref=float('inf'), gap_kernel='auto',
                               on_collins_sampling='ignore', stats_out=st)
        except Exception as exc:                 # noqa: BLE001 -- recorded
            st['error'] = f'{type(exc).__name__}: {exc}'
    rec = {'z': z, 'dx_out': dx_out,
           'theta_x': st.get('theta_x'), 'theta_y': st.get('theta_y'),
           'k4': st.get('k4'), 'kernel_resolved': st.get('kernel'),
           'abcd': st.get('abcd'),
           'errors': {k: out[k][1] for k in out}}
    if out['exact'][0] is not None and out['fresnel'][0] is not None:
        a = np.asarray(out['exact'][0].env)
        b = np.asarray(out['fresnel'][0].env)
        rec['dep_rel_L2'] = float(np.linalg.norm(a - b) / np.linalg.norm(b))
        rec['dep_max_abs_over_peak'] = float(
            np.max(np.abs(a - b)) / np.max(np.abs(b)))
    ab = st.get('abcd')
    if ab:
        A, B = float(ab[0]), float(ab[1])
        rec['z_eff'] = (B / A) if A != 0.0 else float('inf')
        th = max(st.get('theta_x') or 0.0, st.get('theta_y') or 0.0)
        rec['theta_env'] = th
        rec['quartic_envelope'] = (K * abs(rec['z_eff']) * th ** 4 / 8.0)
    return rec


def _loglog_slope(xs, ys):
    """Fitted exponent of ``y = C x^p``, with the fit's own worst residual, so
    the report states a MEASURED exponent and how well the law held."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    m = (xs > 0) & (ys > 0)
    if int(m.sum()) < 3:
        return None
    p = np.polyfit(np.log(xs[m]), np.log(ys[m]), 1)
    resid = np.log(ys[m]) - np.polyval(p, np.log(xs[m]))
    return {'slope': float(p[0]), 'intercept': float(p[1]),
            'max_abs_log_residual': float(np.max(np.abs(resid))),
            'max_rel_deviation': float(np.max(np.abs(np.exp(resid) - 1.0))),
            'n_points': int(m.sum())}


def measure_departure_law():
    """THE DERIVED LAW behind any near-focus threshold.

    The exact-kernel refinement enters a Collins leg as a diagonal phase over
    the REDUCED frame ``z_eff = B/A``, and its leading departure from the
    paraxial kernel is the dropped quartic ``k |z_eff| theta^4 / 8`` -- linear
    in ``z_eff`` and quartic in the angle it is evaluated at.  Any rule of the
    form "fall back to 'fresnel' near a focus" is a bound on that product, so
    the product has to be MEASURED to be a threshold and not an opinion.

    Two one-variable sweeps, each holding the other:

    ``z_eff_sweep`` -- the distance-to-focus ladder on the FIXED fixture.  The
        envelope never changes, so its measured angular half-width is constant
        and the only thing moving is ``z_eff``, which runs over two decades.
        The claim is slope 1.

    ``theta_sweep`` -- ONE distance to focus, with the input envelope's width
        scaled.  A narrower envelope on the same carrier has a wider angular
        spectrum, so ``theta`` moves while ``z_eff`` is held by the geometry.
        The claim is slope 4.  The oracle is not used on this sweep (the
        scaled envelope is no longer the fixture's Gaussian); the two kernels
        are compared with each other.
    """
    q_b = _q_in()
    R_in = _R_of_q(q_b)
    w_in = _w_of_q(q_b)
    x_in = _axis(N_IN, DX_IN)
    r2 = x_in[None, :] ** 2 + x_in[:, None] ** 2
    env_in = np.exp(-r2 / w_in ** 2).astype(np.complex128)

    z_rows = []
    for d in D_LADDER:
        z = F - d
        w_out = _w_of_q(q_b + z)
        rec = _departure(env_in, R_in, z, _pitch_for(w_out, _R_of_q(q_b + z)))
        rec['d_to_focus'] = d
        z_rows.append(rec)
        print(f"[zeff] d={d:.1e} z_eff={rec.get('z_eff')} "
              f"dep={rec.get('dep_rel_L2')}", file=sys.stderr)

    th_rows = []
    d_fixed = 1e-4
    z = F - d_fixed
    for scale in (0.25, 0.35, 0.5, 0.7, 1.0, 1.4):
        w_env = scale * w_in
        # The scaled envelope on the SAME carrier is a different Gaussian beam,
        # so its output width is its own.  Sizing the window from the fixture's
        # width instead CLIPPED the narrow-envelope rungs, and the first
        # spelling of this sweep fitted an exponent of 3.07 with 55 per cent
        # scatter off that clipping.
        q_e = 1.0 / (1.0 / R_in + 1j * LAM / (np.pi * w_env ** 2))
        w_out_e = _w_of_q(q_e + z)
        dx_out = _pitch_for(w_out_e, _R_of_q(q_e + z))
        env = np.exp(-r2 / w_env ** 2).astype(np.complex128)
        rec = _departure(env, R_in, z, dx_out)
        rec['width_scale'] = scale
        rec['w_envelope'] = w_env
        rec['w_out_envelope'] = w_out_e
        # the analytic 1/e^2 half-angle of THIS envelope, beside the measured
        # containment radius the library's own guard uses
        rec['theta_gaussian'] = float(LAM / (np.pi * w_env))
        th_rows.append(rec)
        print(f"[theta] scale={scale} theta={rec.get('theta_env')} "
              f"dep={rec.get('dep_rel_L2')}", file=sys.stderr)

    return {
        'z_eff_sweep': z_rows,
        'z_eff_fit': _loglog_slope(
            [abs(r['z_eff']) for r in z_rows if r.get('dep_rel_L2')],
            [r['dep_rel_L2'] for r in z_rows if r.get('dep_rel_L2')]),
        'theta_sweep': th_rows,
        'theta_fit': _loglog_slope(
            [r['theta_env'] for r in th_rows if r.get('dep_rel_L2')],
            [r['dep_rel_L2'] for r in th_rows if r.get('dep_rel_L2')]),
        'theta_gaussian_fit': _loglog_slope(
            [r['theta_gaussian'] for r in th_rows if r.get('dep_rel_L2')],
            [r['dep_rel_L2'] for r in th_rows if r.get('dep_rel_L2')]),
    }


def main():
    out_path = sys.argv[2]
    import lumenairy
    import scipy
    env = {
        'build': hlib.build_tag(),
        'python': sys.version.split()[0],
        'numpy': np.__version__,
        'scipy': scipy.__version__,
        'lumenairy': lumenairy.__version__,
        'lumenairy_file': os.path.realpath(lumenairy.__file__),
        'tree': _TREE,
        'fixture': {'w0': W0, 'theta': THETA, 'lambda': LAM, 'zR': ZR,
                    'f': F, 'N_in': N_IN, 'dx_in': DX_IN,
                    'd_ladder': list(D_LADDER)},
        'threads_env': {k: os.environ.get(k) for k in
                        ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                         'MKL_NUM_THREADS')},
    }
    result = {'env': env,
              'bookkeeping': measure_bookkeeping(),
              'kernel_resolution': measure_kernel_resolution(),
              'departure_law': measure_departure_law(),
              'table': measure_table()}
    hlib.write_json(result, out_path)
    print(json.dumps({'n_rows': len(result['table']),
                      'build': env['build']}))


if __name__ == '__main__':
    main()
