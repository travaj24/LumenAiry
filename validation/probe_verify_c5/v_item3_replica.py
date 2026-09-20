"""VERIFY-WP-C5 item 3 -- the replica fill, and whether the zeroing is
confined to the replica region.  RE-MEASURED, and on an INDEPENDENT geometry.

The WP-C5 probe computes the faithful region with the same expression the
library's mask uses (``|u + centre_out| <= period/2``), so it cannot refute a
mask that is keyed on the wrong centre.  Everything here is derived three ways
that do not share that expression:

  * **empirically**, from the returned ``'repeat'`` array itself -- the integer
    sample shift ``m`` that makes it self-identical IS the period, measured
    with no reference to what the library reports;
  * **from the closed form** of the transport whose readout it is
    (``lambda |z| / dx`` per axis for the Collins readout, ``N_stop dx_stop``
    for the Sziklas one), read out of ``_period_out`` only to be COMPARED;
  * **physically**, against an independent analytic truth -- the focused
    Gaussian the fixture is -- so the faithful band is the set where the
    readout AGREES with the truth, not the set the library says it is.

Then the misapplication case the maintainer named: >= 6 OFF-AXIS and
ANAMORPHIC readouts, where a fill keyed on the WINDOW's centre and a fill keyed
on the FIELD's origin disagree, and the zeroed set is required to be exactly
the complement of one period about the FIELD's origin.

Usage (BLAS pinned on the COMMAND LINE)::

    PYTHONPATH=<tree> python validation/probe_verify_c5/v_item3_replica.py OUT.json
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np                                            # noqa: E402

from _vd import dig, write                                    # noqa: E402

from lumenairy.propagators import carrier as C                # noqa: E402

# A geometry of this verification's own: 1.55 um, R = -25 mm, w = 0.6 mm.
WL = 1.55e-6
RMAG = -25.0e-3
W_IN = 0.60e-3
N_IN, DX_IN = 512, 5.0e-6


def ax(n, d):
    return (np.arange(int(n), dtype=np.float64) - int(n) / 2) * float(d)


def pupil(n=N_IN, dx=DX_IN, w=W_IN, dy=None):
    x, y = ax(n, dx), ax(n, dy if dy else dx)
    return np.exp(-(x[None, :] ** 2 + y[:, None] ** 2)
                  / float(w) ** 2).astype(np.complex128)


def truth_at_focus(n_out, dx_out, centre=(0.0, 0.0), w=W_IN, dx=DX_IN,
                   dy=None):
    """The analytic focused Gaussian on the readout lattice -- the independent
    physical truth the faithful band is defined against.

    For a Gaussian pupil ``exp(-r^2/w^2)`` referenced to ``R``, the field at
    ``z = -R`` is the Fourier-conjugate Gaussian of waist
    ``w0 = lambda |R| / (pi w)``, with the leg's own piston and Gouy phase.
    Only the SHAPE is used below (the comparison is amplitude-normalised and
    piston-free), so no convention can leak in."""
    w0 = WL * abs(RMAG) / (np.pi * float(w))
    xo = ax(n_out, dx_out) + float(centre[0])
    yo = ax(n_out, dx_out) + float(centre[1])
    return np.exp(-(xo[None, :] ** 2 + yo[:, None] ** 2) / w0 ** 2)


def collins_read(dx_out, n_out, centre=(0.0, 0.0), fill=None, dy=None,
                 on_replica='ignore'):
    pd = {}
    kw = {} if fill is None else {'replica_fill': fill}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        F = C._collins_focus_readout(
            pupil(dy=dy), RMAG, -RMAG, WL, DX_IN, (dy if dy else DX_IN),
            dx_out=dx_out, N_out=n_out, centre_out=centre,
            on_replica=on_replica, on_collins_sampling='ignore',
            _period_out=pd, **kw)
    return np.asarray(F), pd


def sziklas_read(dx_out, n_out, centre=(0.0, 0.0), fill=None,
                 on_replica='ignore'):
    pd = {}
    kw = {} if fill is None else {'replica_fill': fill}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        F = C.carrier_referenced_focus_readout(
            pupil(), RMAG, -RMAG, WL, DX_IN, dx_out=dx_out, N_out=n_out,
            centre_out=centre, on_replica=on_replica,
            on_focus_containment='ignore', _period_out=pd, **kw)
    return np.asarray(F), pd


def empirical_period(F_rep, dx_out, axis):
    """The period MEASURED off the ``'repeat'`` array: the smallest positive
    integer shift whose overlap is (near) exact.  Returns
    ``(shift_samples, period_m, residual)`` or ``(None, None, None)``."""
    A = np.asarray(F_rep)
    n = A.shape[axis]
    best = (None, None, np.inf)
    scale = float(np.max(np.abs(A)))
    if not (scale > 0):
        return (None, None, None)
    for m in range(2, n):
        if axis == 1:
            d = A[:, m:] - A[:, :n - m]
        else:
            d = A[m:, :] - A[:n - m, :]
        if d.size == 0:
            break
        r = float(np.max(np.abs(d))) / scale
        if r < best[2]:
            best = (m, m * float(dx_out), r)
        if r < 1e-13:
            return (m, m * float(dx_out), r)
    return best


def band_masks(n, dx_out, period, centre):
    """The three candidate bands, per axis: about the FIELD's origin (the
    claim), about the WINDOW's centre (the mutant the maintainer named), and
    the whole window."""
    u = ax(n, dx_out)
    px, py = float(period[0]), float(period[1])
    cx, cy = float(centre[0]), float(centre[1])
    tol = 1.0 + 1e-9
    field_x = np.abs(u + cx) <= 0.5 * px * tol
    field_y = np.abs(u + cy) <= 0.5 * py * tol
    win_x = np.abs(u) <= 0.5 * px * tol
    win_y = np.abs(u) <= 0.5 * py * tol
    return (np.logical_and(field_y[:, None], field_x[None, :]),
            np.logical_and(win_y[:, None], win_x[None, :]),
            int(field_x.sum()), int(field_y.sum()))


def case(tag, reader, dx_out, n_out, centre=(0.0, 0.0), dy=None, res=None):
    """One readout, scored three ways."""
    kw = dict(centre=centre) if reader is sziklas_read \
        else dict(centre=centre, dy=dy)
    F_rep, pd_r = reader(dx_out, n_out, fill='repeat', **kw)
    F_zer, pd_z = reader(dx_out, n_out, fill='zero', **kw)
    F_def, pd_d = reader(dx_out, n_out, fill=None, **kw)
    period = tuple(float(v) for v in pd_r['period'])
    # (1) empirical period off the 'repeat' array
    mx, px_emp, rx = empirical_period(F_rep, dx_out, 1)
    my, py_emp, ry = empirical_period(F_rep, dx_out, 0)
    # (2) the two candidate bands
    m_field, m_win, fx, fy = band_masks(n_out, dx_out, period, centre)
    # (3) what was actually zeroed
    zeroed = (F_zer == 0) & (F_rep != 0)
    # the physical check: where does 'repeat' agree with the analytic truth?
    T = truth_at_focus(n_out, dx_out, centre)
    a_rep = np.abs(F_rep)
    a_rep = a_rep / max(a_rep.max(), 1e-300)
    Tn = T / max(T.max(), 1e-300)
    dev = np.abs(a_rep - Tn)
    row = dict(
        tag=tag, dx_out=dx_out, n_out=n_out,
        centre=[float(centre[0]), float(centre[1])],
        centre_in_periods=[float(centre[0]) / period[0],
                           float(centre[1]) / period[1]],
        dy_over_dx=(float(dy) / DX_IN) if dy else 1.0,
        period=list(period),
        window_over_period=[n_out * dx_out / period[0],
                            n_out * dx_out / period[1]],
        empirical_shift=[mx, my],
        empirical_period=[px_emp, py_emp],
        empirical_residual=[rx, ry],
        empirical_vs_reported=[(px_emp / period[0]) if px_emp else None,
                               (py_emp / period[1]) if py_emp else None],
        faithful_reported=list(pd_r.get('faithful_samples', ())),
        faithful_formula=[2 * int(np.floor(period[0] / 2 / dx_out)) + 1,
                          2 * int(np.floor(period[1] / 2 / dx_out)) + 1],
        faithful_from_field_mask=[fx, fy],
        replica_fill_key=[pd_r.get('replica_fill'), pd_z.get('replica_fill'),
                          pd_d.get('replica_fill')],
        # THE CLAIM: the zeroed set is exactly the complement of the
        # FIELD-centred band, and is NOT the complement of the WINDOW-centred
        # band whenever the two differ.
        zeroed_equals_complement_of_field_band=bool(
            np.array_equal(zeroed, ~m_field)),
        zeroed_equals_complement_of_window_band=bool(
            np.array_equal(zeroed, ~m_win)),
        field_and_window_bands_differ=bool(not np.array_equal(m_field, m_win)),
        n_zeroed=int(zeroed.sum()), n_field_band=int(m_field.sum()),
        n_window_band=int(m_win.sum()),
        inside_byte_identical=bool(np.array_equal(F_rep[m_field],
                                                  F_zer[m_field])),
        inside_is_same_object=bool(F_rep is F_zer),
        outside_max_abs_zero=(float(np.max(np.abs(F_zer[~m_field])))
                              if (~m_field).any() else None),
        outside_max_abs_repeat=(float(np.max(np.abs(F_rep[~m_field])))
                                if (~m_field).any() else None),
        default_equals_zero=bool(dig(F_def) == dig(F_zer)),
        default_equals_repeat=bool(dig(F_def) == dig(F_rep)),
        # the physical separation, from the analytic truth
        dev_inside_field_band=float(dev[m_field].max()),
        dev_outside_field_band=(float(dev[~m_field].max())
                                if (~m_field).any() else None),
        dev_inside_window_band=float(dev[m_win].max()),
        dev_outside_window_band=(float(dev[~m_win].max())
                                 if (~m_win).any() else None),
        # the core's own waist, in output samples -- the physical arm above is
        # only meaningful where the readout lattice RESOLVES the focal spot
        w0_over_dx_out=(WL * abs(RMAG) / (np.pi * W_IN)) / dx_out,
        peak_inside_field_band=float(np.abs(F_rep[m_field]).max()),
        peak_outside_field_band=(float(np.abs(F_rep[~m_field]).max())
                                 if (~m_field).any() else None),
    )
    if res is not None:
        res['digests']['%s/repeat' % tag] = dig(F_rep)
        res['digests']['%s/zero' % tag] = dig(F_zer)
        res['digests']['%s/default' % tag] = dig(F_def)
    return row


def main(out_path):
    res = {'digests': {}, 'cases': []}
    warnings.simplefilter('ignore')
    import inspect
    res['signature_defaults'] = {
        f: inspect.signature(getattr(C, f)).parameters['replica_fill'].default
        for f in ('carrier_referenced_focus_readout',
                  'carrier_referenced_exact_focus_readout',
                  '_collins_focus_readout')}
    res['replica_fills'] = sorted(C._REPLICA_FILLS)

    # the period of the Collins readout, closed form: lambda |z| / dx
    p_closed = WL * abs(RMAG) / DX_IN
    _F, pd = collins_read(p_closed / 64.0, 16)
    res['period_closed_form'] = p_closed
    res['period_reported'] = list(float(v) for v in pd['period'])

    # -- faithful windows: identity on either fill ------------------------
    for frac in (0.25, 0.50, 0.90, 1.00):
        n_out = 128
        dxo = frac * p_closed / n_out
        F_rep, pdr = collins_read(dxo, n_out, fill='repeat',
                                  on_replica='error')
        F_zer, pdz = collins_read(dxo, n_out, fill='zero', on_replica='error')
        F_def, pdd = collins_read(dxo, n_out, fill=None, on_replica='error')
        res.setdefault('faithful', []).append(dict(
            frac=frac,
            faithful=list(pdr.get('faithful_samples', ())),
            n_out=n_out,
            repeat_digest_eq_zero=bool(dig(F_rep) == dig(F_zer)),
            default_eq_repeat=bool(dig(F_def) == dig(F_rep)),
            replica_fill_keys=[pdr.get('replica_fill'),
                               pdz.get('replica_fill'),
                               pdd.get('replica_fill')]))
        res['digests']['faithful/%g/repeat' % frac] = dig(F_rep)
        res['digests']['faithful/%g/zero' % frac] = dig(F_zer)
        res['digests']['faithful/%g/default' % frac] = dig(F_def)

    # identity (the SAME object) on a faithful window
    got = []
    for fill in ('repeat', 'zero'):
        E = pupil()
        pd = {}
        F = C._fill_readout_replicas(E, (p_closed, p_closed),
                                     p_closed / (4 * N_IN), N_IN,
                                     (0.0, 0.0), fill=fill, out=pd)
        got.append(F is E)
    res['faithful_returned_by_identity'] = got

    # -- oversized, ON axis -------------------------------------------------
    for wp in (1.10, 1.60, 2.20):
        n_out = 256
        dxo = wp * p_closed / n_out
        res['cases'].append(case('on_axis/%.2f' % wp, collins_read, dxo,
                                 n_out, res=res))

    # -- OFF-AXIS and ANAMORPHIC: the misapplication cases -----------------
    off = [
        ('off/x030', collins_read, 1.60, 256, (0.30, 0.00), None),
        ('off/y030', collins_read, 1.60, 256, (0.00, 0.30), None),
        ('off/xy', collins_read, 1.60, 256, (0.30, -0.45), None),
        ('off/edge', collins_read, 1.60, 256, (0.62, 0.62), None),
        ('off/small', collins_read, 1.15, 256, (0.07, -0.21), None),
        ('off/wide', collins_read, 2.40, 256, (0.55, 0.15), None),
    ]
    for tag, rd, wp, n_out, cfrac, dy in off:
        dxo = wp * p_closed / n_out
        centre = (cfrac[0] * p_closed, cfrac[1] * p_closed)
        res['cases'].append(case(tag, rd, dxo, n_out, centre=centre, dy=dy,
                                 res=res))

    # anamorphic: dy != dx makes the two periods differ
    for tag, dyf, wp, cfrac in (('ana/1.7', 1.7, 1.60, (0.25, 0.40)),
                                ('ana/0.6', 0.6, 1.60, (-0.33, 0.18)),
                                ('ana/2.5', 2.5, 1.30, (0.40, -0.10))):
        dy = dyf * DX_IN
        px = WL * abs(RMAG) / DX_IN
        n_out = 256
        dxo = wp * px / n_out
        centre = (cfrac[0] * px, cfrac[1] * (WL * abs(RMAG) / dy))
        res['cases'].append(case(tag, collins_read, dxo, n_out, centre=centre,
                                 dy=dy, res=res))

    # FINELY SAMPLED: dx_out well inside the focal waist, so the physical arm
    # (agreement with the analytic focused Gaussian) is a real discriminator
    # and not a statement about an unresolved spot.
    for tag, wp, cfrac in (('fine/on', 1.60, 0.00),
                           ('fine/off', 1.60, 0.30)):
        n_out = 2048
        dxo = wp * p_closed / n_out
        res['cases'].append(case(tag, collins_read, dxo, n_out,
                                 centre=(cfrac * p_closed, 0.0), res=res))

    # the Sziklas readout, on axis and off
    for tag, wp, cfrac in (('szik/on/1.60', 1.60, 0.0),
                           ('szik/off/0.30', 1.60, 0.30)):
        n_out = 256
        _F, pd = sziklas_read(1.0e-6, 32)
        p = float(min(pd['period']))
        dxo = wp * p / n_out
        res['cases'].append(case(tag, sziklas_read, dxo, n_out,
                                 centre=(cfrac * p, 0.0), res=res))

    # -- the exact readout --------------------------------------------------
    try:
        pd = {}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            E_full = (pupil() * np.exp(
                1j * 2 * np.pi / WL
                * (ax(N_IN, DX_IN)[None, :] ** 2
                   + ax(N_IN, DX_IN)[:, None] ** 2) / (2.0 * RMAG)))
            C.carrier_referenced_exact_focus_readout(
                E_full, RMAG, -RMAG, WL, DX_IN, dx_out=1.0e-6, N_out=32,
                on_replica='ignore', on_readout_window='ignore',
                _period_out=pd)
        p_ex = float(min(pd['period']))
        n_out = 4096
        dxo = 1.71 * p_ex / n_out
        rows = {}
        for fill in ('repeat', 'zero', None):
            pd2 = {}
            kw = {} if fill is None else {'replica_fill': fill}
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                F = np.asarray(C.carrier_referenced_exact_focus_readout(
                    E_full, RMAG, -RMAG, WL, DX_IN, dx_out=dxo, N_out=n_out,
                    on_replica='ignore', on_readout_window='ignore',
                    _period_out=pd2, **kw))
            rows[str(fill)] = (F, pd2)
            res['digests']['exact/%s' % fill] = dig(F)
        Fr, pdr = rows['repeat']
        Fz, pdz = rows['zero']
        Fd, pdd = rows['None']
        per = tuple(float(v) for v in pdr['period'])
        m_field, m_win, fx, fy = band_masks(n_out, dxo, per, (0.0, 0.0))
        res['exact_readout'] = dict(
            period=list(per), n_out=n_out, dx_out=dxo,
            window_over_period=n_out * dxo / per[0],
            faithful_reported=list(pdr.get('faithful_samples', ())),
            faithful_formula=[2 * int(np.floor(per[0] / 2 / dxo)) + 1,
                              2 * int(np.floor(per[1] / 2 / dxo)) + 1],
            inside_byte_identical=bool(np.array_equal(Fr[m_field],
                                                      Fz[m_field])),
            outside_max_abs_zero=float(np.max(np.abs(Fz[~m_field]))),
            outside_max_abs_repeat=float(np.max(np.abs(Fr[~m_field]))),
            default_eq_zero=bool(dig(Fd) == dig(Fz)),
            replica_fill_keys=[pdr.get('replica_fill'), pdz.get('replica_fill'),
                               pdd.get('replica_fill')])
    except Exception as exc:                            # pragma: no cover
        res['exact_readout'] = {'error': '%s: %s' % (type(exc).__name__, exc)}

    # -- the refusal census, cell for cell ---------------------------------
    census = {}
    for wp in (0.50, 0.98, 1.00, 1.02, 1.60, 2.20):
        n_out = 128
        dxo = wp * p_closed / n_out
        for fill in ('repeat', 'zero'):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('error')
                    collins_read(dxo, n_out, fill=fill, on_replica='error')
                verdict = 'served'
            except Warning as exc:
                verdict = 'warned:' + type(exc).__name__
            except (ValueError, RuntimeError) as exc:
                verdict = ('refused:'
                           + ('ALIASES' if 'ALIAS' in str(exc).upper()
                              else 'other'))
            census['%.2f/%s' % (wp, fill)] = verdict
    res['refusal_census'] = census

    # -- the chain publishes the fill ---------------------------------------
    try:
        res['chain'] = chain_probe()
    except Exception as exc:                            # pragma: no cover
        res['chain'] = {'error': '%s: %s' % (type(exc).__name__, exc)}

    write(out_path, res)


def _singlet(R1, R2, d, glass, ap, name):
    surfaces = [
        {'radius': R1, 'glass_before': 'air', 'glass_after': glass,
         'conic': 0.0, 'radius_y': None, 'conic_y': None,
         'aspheric_coeffs': None, 'aspheric_coeffs_y': None},
        {'radius': R2, 'glass_before': glass, 'glass_after': 'air',
         'conic': 0.0, 'radius_y': None, 'conic_y': None,
         'aspheric_coeffs': None, 'aspheric_coeffs_y': None}]
    return {'name': name, 'aperture_diameter': ap,
            'surfaces': surfaces, 'thicknesses': [d]}


def chain_probe():
    """``propagate_traced_carrier_chain`` forwards the keyword and publishes
    ``readout_replica_fill`` per stage."""
    import lumenairy as la
    n, dx, w, R_in, wl = 512, 30e-6, 4.5e-3, 60e-3, 1.31e-6
    presc = _singlet(60e-3, -60e-3, 3e-3, 'N-BK7', 14e-3, 'p')
    groups = [{'prescription': presc, 'gap_before': 20e-3},
              {'prescription': presc, 'gap_before': 10e-3}]
    x = (np.arange(n) - n // 2) * dx
    E = np.exp(-(x[None, :] ** 2 + x[:, None] ** 2) / w ** 2
               ).astype(np.complex128)
    tkw = dict(on_undersample='silent', on_noncollimated='silent')
    out = {}
    for fill in (None, 'repeat', 'zero'):
        fr = dict(dx_out=0.5e-6, N_out=2048, on_replica='ignore')
        if fill is not None:
            fr['replica_fill'] = fill
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = la.propagate_traced_carrier_chain(
                E, groups, wl, dx, r_in=R_in, ray_subsample=8, n_workers=1,
                final_distance=8e-3, traced_kwargs=tkw, final_leg='paraxial',
                focus_readout=fr)
        stages = list(getattr(r, 'stages', None) or [])
        keys = [s.get('readout_replica_fill') for s in stages
                if isinstance(s, dict) and 'readout_replica_fill' in s]
        faith = [s.get('readout_faithful_samples') for s in stages
                 if isinstance(s, dict) and 'readout_faithful_samples' in s]
        out[str(fill)] = dict(readout_replica_fill=keys,
                              readout_faithful_samples=[list(f) for f in faith],
                              n_stages=len(stages),
                              digest=dig(np.asarray(r.field)))
    out['default_eq_zero'] = bool(
        out['None']['digest'] == out['zero']['digest'])
    out['default_eq_repeat'] = bool(
        out['None']['digest'] == out['repeat']['digest'])
    return out


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1
         else os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'v_item3_head_win.json'))
