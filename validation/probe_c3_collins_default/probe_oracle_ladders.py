"""WP-C3 -- the oracle ladders the default flip is decided on.

Run as a CHILD process bound to ONE tree::

    python probe_oracle_ladders.py <tree> <out.json>

Both transports are driven IN THIS ONE PROCESS, by keyword, and each is scored
against an oracle written here.  That is deliberate and it is not the same
claim as bit identity: the byte-identity claim is made archive-to-archive by
``probe_sziklas_bitid.py`` (the working tree cannot make it), while THIS probe
asks a different question -- on the same build, in the same session, on the
same fixture, what does each transport's answer cost against a truth neither
of them computes.  Running the two arms in one process is what makes that a
fair comparison rather than two runs of two builds.

FIVE LADDERS, in the order the work package asks for them:

A. the absolute-phase Gaussian oracle of WP-B4 / VERIFY-WP-B4 (piston AND Gouy
   phase, not a piston-free shape comparison) on five leg geometries --
   diverging, converging short of the focus, exactly ON the focus, past it on
   both a shallow and a deep negative ``A``, and astigmatic;
B. ``propagate_traced_carrier_chain_multi`` at K = 1 and K = 2;
C. the focus readouts -- the two public ones and the chain's, with
   VERIFY-WP-B4 F1's readout-K1 measured on the fixture's own exit pitch,
   because F1 names that number as the binding constraint on a default flip
   and it is not in WP-B4 sec. 5's list;
D. the traced carrier chain end to end on two prescriptions -- the P2
   battery's fast uncorrected singlet and WP-A15a's curved-rear cemented
   doublet;
E. the ``on_collins_sampling='warn'`` census: how many legs of the shipped
   fixtures above would emit the Kelly guard's warning once the default moves,
   which is the number the Migration paragraph owes a caller.

Nothing here asserts.  The probe MEASURES and writes JSON; the decisions are
taken in ``tests/unit/test_c3_collins_default.py`` against bars derived from
these same quantities at run time.
"""
from __future__ import annotations

import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_TREE = os.path.abspath(sys.argv[1])
sys.path.insert(0, _TREE)

import clib  # noqa: E402

import numpy as np  # noqa: E402

clib.anchor(_TREE)

import lumenairy as la                              # noqa: E402
import lumenairy.propagators.carrier as CA          # noqa: E402

WL = 1.064e-6
W_IN = 0.30e-3
R_CONV = -40.0e-3
N = 512
DX = 6.0 * W_IN / N            # six 1/e radii across the grid


# ---------------------------------------------------------------------------
# warning census -- one recorder, used by every ladder below
# ---------------------------------------------------------------------------
_CENSUS = []


def _census(tag, transport, rec):
    """Fold one call's warnings into the Kelly-guard census."""
    kelly = [str(w.message) for w in rec
             if 'Collins chirp-Z stage is under-sampled' in str(w.message)]
    _CENSUS.append({
        'tag': tag, 'transport': transport,
        'n_warnings': len(rec), 'n_kelly': len(kelly),
        'conditions': sorted({c for m in kelly for c in ('K1', 'K2', 'K3')
                              if f'{c} (' in m}),
    })
    return kelly


def _run(_tag, _transport, _fn, *a, **kw):
    """Call ``_fn`` recording every warning; return ``(value, exc, kelly)``.

    The three leading parameters are underscore-named so a probe may pass a
    keyword called ``transport`` (which every call below does) without
    colliding with this helper's own.
    """
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        try:
            val, exc = _fn(*a, **kw), None
        except Exception as e:                  # noqa: BLE001 -- recorded
            val, exc = None, f'{type(e).__name__}: {e}'
    kelly = _census(_tag, _transport, rec)
    return val, exc, kelly


# ===========================================================================
# A.  The absolute-phase Gaussian oracle
# ===========================================================================
def ladder_a():
    x = clib.axis(N, DX)
    env_div = clib.gaussian_field(x, x, W_IN, np.inf, WL)     # envelope only
    rows = []
    # (tag, R_carrier, z) -- A = 1 + z/R is printed beside each reading.
    cells = [
        ('diverging', +40.0e-3, 5.0e-3),
        ('converging-A+0.5', R_CONV, 20.0e-3),
        ('at-the-focus-A0', R_CONV, 40.0e-3),
        ('past-it-A-0.025', R_CONV, 41.0e-3),
        ('past-it-A-0.5', R_CONV, 60.0e-3),
    ]
    for tag, R, z in cells:
        A = 1.0 + z / R
        for tr in ('sziklas', 'collins'):
            for gk in ('auto', 'fresnel'):
                key = f'A::{tag}::{tr}::{gk}'
                out, exc, kelly = _run(key, tr, CA.propagate_carrier_referenced,
                                       env_div, R, z, WL, DX,
                                       transport=tr, gap_kernel=gk,
                                       on_collins_sampling='warn')
                row = {'cell': tag, 'transport': tr, 'gap_kernel': gk,
                       'R': R, 'z': z, 'A': A, 'raised': exc,
                       'n_kelly': len(kelly)}
                if out is not None:
                    dxo, dyo = clib.unpack_pitch(out.dx)
                    got = clib.reconstruct_field(out.env, out.R, dxo, dyo, WL)
                    xo = clib.axis(np.shape(got)[-1], dxo)
                    yo = clib.axis(np.shape(got)[-2], dyo)
                    # The oracle's input plane is the CARRIER's plane: the
                    # envelope above is referenced to R, so the physical input
                    # field is the Gaussian with that wavefront radius.
                    ref = clib.gaussian_propagated(xo, yo, W_IN, R, WL, z)
                    row.update({
                        'dx_out': dxo, 'dy_out': dyo,
                        'rel_l2_absolute': clib.rel_l2(got, ref),
                        'rel_l2_piston_free': clib.rel_l2_piston_free(got, ref),
                        'best_global_phase_rad':
                            clib.best_global_phase(got, ref),
                        'power_ratio': float(
                            (np.abs(got) ** 2).sum() * dxo * dyo
                            / max((np.abs(ref) ** 2).sum() * dxo * dyo,
                                  1e-300)),
                    })
                rows.append(row)
    # the astigmatic arm, its own cell because the oracle is per axis
    for tr in ('sziklas', 'collins'):
        key = f'A::astigmatic::{tr}::auto'
        Rxy = (-40.0e-3, -55.0e-3)
        z = 20.0e-3
        out, exc, kelly = _run(key, tr, CA.propagate_carrier_referenced,
                               env_div, Rxy, z, WL, DX, transport=tr,
                               gap_kernel='auto', on_collins_sampling='warn')
        row = {'cell': 'astigmatic', 'transport': tr, 'gap_kernel': 'auto',
               'R': list(Rxy), 'z': z, 'A': [1.0 + z / Rxy[0],
                                             1.0 + z / Rxy[1]],
               'raised': exc, 'n_kelly': len(kelly)}
        if out is not None:
            dxo, dyo = clib.unpack_pitch(out.dx)
            got = clib.reconstruct_field(out.env, out.R, dxo, dyo, WL)
            xo = clib.axis(np.shape(got)[-1], dxo)
            yo = clib.axis(np.shape(got)[-2], dyo)
            # separable: the 2-D oracle is the product of two 1-D ones, and
            # the 2-D prefactor is sqrt(qx/qx2) * sqrt(qy/qy2) per axis.
            k = 2.0 * np.pi / WL
            fld = np.ones((yo.size, xo.size), dtype=np.complex128)
            for R1, coord, ax in ((Rxy[0], xo, 1), (Rxy[1], yo, 0)):
                inv_q = 1.0 / R1 + 1j * WL / (np.pi * W_IN * W_IN)
                q = 1.0 / inv_q
                q2 = q + z
                amp = np.sqrt(q / q2) * np.exp(1j * k * coord ** 2
                                               / (2.0 * q2))
                fld = fld * (amp[None, :] if ax == 1 else amp[:, None])
            ref = fld * np.exp(1j * k * z)
            row.update({
                'dx_out': dxo, 'dy_out': dyo,
                'rel_l2_absolute': clib.rel_l2(got, ref),
                'rel_l2_piston_free': clib.rel_l2_piston_free(got, ref),
                'best_global_phase_rad': clib.best_global_phase(got, ref),
            })
        rows.append(row)
    return rows


# ===========================================================================
# B / C / D.  The chain, its readouts and the multi orchestrator
# ===========================================================================
_TKW = dict(on_undersample='silent', on_noncollimated='silent')


def _singlet():
    """The P2 battery's fast, deliberately uncorrected biconvex BK7 singlet."""
    return {'name': 'fast_singlet', 'aperture_diameter': 6.0e-3,
            'thicknesses': [3e-3],
            'surfaces': [
                {'radius': 9e-3, 'glass_before': 'air',
                 'glass_after': 'N-BK7', 'conic': 0.0, 'radius_y': None,
                 'conic_y': None, 'aspheric_coeffs': None,
                 'aspheric_coeffs_y': None},
                {'radius': -9e-3, 'glass_before': 'N-BK7',
                 'glass_after': 'air', 'conic': 0.0, 'radius_y': None,
                 'conic_y': None, 'aspheric_coeffs': None,
                 'aspheric_coeffs_y': None}]}


def _doublet():
    """WP-A15a's curved-rear cemented doublet (the covering-array fixture)."""
    return {'name': 'curved_rear_doublet', 'aperture_diameter': 6.0e-3,
            'thicknesses': [9.0e-3, 2.5e-3],
            'surfaces': [
                {'radius': 33.3e-3, 'glass_before': 'AIR',
                 'glass_after': 'N-BAF10', 'conic': 0.0, 'radius_y': None,
                 'conic_y': None, 'aspheric_coeffs': None,
                 'aspheric_coeffs_y': None},
                {'radius': -22.28e-3, 'glass_before': 'N-BAF10',
                 'glass_after': 'N-SF6HT', 'conic': 0.0, 'radius_y': None,
                 'conic_y': None, 'aspheric_coeffs': None,
                 'aspheric_coeffs_y': None},
                {'radius': -291.07e-3, 'glass_before': 'N-SF6HT',
                 'glass_after': 'AIR', 'conic': 0.0, 'radius_y': None,
                 'conic_y': None, 'aspheric_coeffs': None,
                 'aspheric_coeffs_y': None}]}


def _chain_launch(n=256, dx=60e-6, w=4.5e-3):
    x = clib.axis(n, dx)
    X, Y = np.meshgrid(x, x)
    return np.exp(-(X ** 2 + Y ** 2) / (w * w)).astype(np.complex128), dx


def _relay_fixture():
    """The two-group relay WP-B4's own gate (c)/(d) run on."""
    env, dx = _chain_launch()
    presc = {'name': 'p', 'aperture_diameter': 14e-3, 'thicknesses': [3e-3],
             'surfaces': [
                 {'radius': 60e-3, 'glass_before': 'air',
                  'glass_after': 'N-BK7', 'conic': 0.0, 'radius_y': None,
                  'conic_y': None, 'aspheric_coeffs': None,
                  'aspheric_coeffs_y': None},
                 {'radius': -60e-3, 'glass_before': 'N-BK7',
                  'glass_after': 'air', 'conic': 0.0, 'radius_y': None,
                  'conic_y': None, 'aspheric_coeffs': None,
                  'aspheric_coeffs_y': None}]}
    return env, dx, 60e-3, [{'prescription': presc, 'gap_before': 20e-3},
                            {'prescription': presc, 'gap_before': 10e-3}]


def _spot_metrics(field, dx_out):
    """Peak, total power on the returned window, and the amplitude-weighted
    second moment -- three numbers that move if the readout moves."""
    a = np.abs(np.asarray(field)) ** 2
    tot = float(a.sum())
    n = a.shape[-1]
    x = clib.axis(n, dx_out)
    y = clib.axis(a.shape[-2], dx_out)
    if tot <= 0.0:
        return {'peak': 0.0, 'power': 0.0, 'r2m': float('nan')}
    cx = float((a.sum(axis=0) * x).sum() / tot)
    cy = float((a.sum(axis=1) * y).sum() / tot)
    r2 = ((x - cx) ** 2)[None, :] + ((y - cy) ** 2)[:, None]
    return {'peak': float(a.max()), 'power': tot * dx_out * dx_out,
            'centroid_x': cx, 'centroid_y': cy,
            'r2m': float(np.sqrt((a * r2).sum() / tot))}


def ladder_bcd():
    env, dx, r_in, groups = _relay_fixture()
    fr = dict(dx_out=0.5e-6, N_out=64)
    out = {'multi': [], 'readouts': [], 'chain': []}

    # --- B.  _multi at K = 1 and K = 2 ------------------------------------
    for K in (1, 2):
        cong = [{'field': env, 'carrier': r_in}] * K
        for tr in ('sziklas', 'collins'):
            key = f'B::multi-K{K}::{tr}'
            res, exc, kelly = _run(
                key, tr, CA.propagate_traced_carrier_chain_multi,
                cong, groups, 1.31e-6, dx, output_grid=fr,
                final_distance=8e-3, ray_subsample=16, n_workers=1,
                traced_kwargs=_TKW, final_leg='paraxial', transport=tr)
            row = {'K': K, 'transport': tr, 'raised': exc,
                   'n_kelly': len(kelly)}
            if res is not None:
                row.update({'dx': float(res.dx),
                            'centre': [float(c) for c in res.centre]})
                row.update(_spot_metrics(res.field, float(res.dx)))
            out['multi'].append(row)

    # --- C.  the focus readouts -------------------------------------------
    # the chain's own readout, both transports
    for tr in ('sziklas', 'collins'):
        key = f'C::chain-readout::{tr}'
        res, exc, kelly = _run(
            key, tr, CA.propagate_traced_carrier_chain,
            env, groups, 1.31e-6, dx, r_in=r_in, ray_subsample=16,
            n_workers=1, final_distance=8e-3, traced_kwargs=_TKW,
            final_leg='paraxial', focus_readout=fr, transport=tr)
        row = {'entry': 'propagate_traced_carrier_chain(focus_readout=)',
               'transport': tr, 'raised': exc, 'n_kelly': len(kelly)}
        if res is not None:
            row.update({'dx': float(res.dx)})
            row.update(_spot_metrics(res.field, float(res.dx)))
        out['readouts'].append(row)

    # the two PUBLIC readouts -- neither takes ``transport``; this records
    # that fact as a measurement rather than as a reading of the signature.
    import inspect
    for name in ('carrier_referenced_focus_readout',
                 'carrier_referenced_exact_focus_readout'):
        fn = getattr(CA, name)
        out['readouts'].append({
            'entry': name, 'transport': None,
            'has_transport_kwarg':
                'transport' in inspect.signature(fn).parameters})

    # VERIFY-WP-B4 F1: the one-step readout's own K1 on the chain's exit
    # pitch, at three grids.  ``K1 = 2 dx (|A| r/|B| + theta)/lambda``.
    f1 = []
    for n in (256, 512, 1024):
        e2, dx2 = _chain_launch(n=n, dx=60e-6 * 256.0 / n)
        key = f'C::F1-K1-N{n}'
        # final_distance=0 -- the CHAIN EXIT plane, which is the lattice the
        # one-step readout would run on.  (Measuring after the final leg
        # instead reads the K1 of a readout that is not the one being taken,
        # and is 1.47x smaller on this fixture.)
        res, exc, _k = _run(key, 'collins',
                            CA.propagate_traced_carrier_chain,
                            e2, groups, 1.31e-6, dx2, r_in=r_in,
                            ray_subsample=16, n_workers=1,
                            final_distance=0.0, traced_kwargs=_TKW,
                            final_leg='paraxial', transport='collins')
        row = {'N': n, 'dx_in': dx2, 'raised': exc}
        if res is not None:
            # measure the exit box, then evaluate K1 for the one-step readout
            env_x = np.asarray(res.field)
            dxe = float(res.dx if not isinstance(res.dx, tuple) else res.dx[0])
            r_x, r_y, th_x, th_y = CA._collins_input_box(
                env_x, dxe, dxe, 1.31e-6, CA._COLLINS_TAIL_FRAC)
            Rx = res.R if not isinstance(res.R, tuple) else res.R[0]
            z = 8e-3
            A = 1.0 + (z / Rx if np.isfinite(Rx) else 0.0)
            B = z
            row.update({
                'dx_exit': dxe, 'R_exit': float(Rx), 'A': float(A),
                'r_support_m': float(max(r_x, r_y)),
                'theta_rad': float(max(th_x, th_y)),
                'readout_K1': float(2.0 * dxe
                                    * (abs(A) * max(r_x, r_y) / abs(B)
                                       + max(th_x, th_y)) / 1.31e-6),
            })
            # the library's own reading of the same quantity, beside the
            # hand-derived one -- two spellings, one number, or the helper is
            # not computing what this probe says it computes
            if hasattr(CA, '_collins_readout_k1'):
                row['readout_K1_library'] = float(
                    CA._collins_readout_k1(env_x, Rx, z, 1.31e-6, dxe, dxe))
        f1.append(row)
    out['f1_readout_k1'] = f1

    # --- D.  the chain end to end on two prescriptions --------------------
    for name, presc in (('singlet', _singlet()), ('doublet', _doublet())):
        e2, dx2 = _chain_launch(n=256, dx=2.2 * 6.0e-3 / 256)
        g = [{'prescription': presc, 'gap_before': 0.0}]
        for tr in ('sziklas', 'collins'):
            key = f'D::{name}::{tr}'
            res, exc, kelly = _run(
                key, tr, CA.propagate_traced_carrier_chain,
                e2, g, WL, dx2, r_in=np.inf, ray_subsample=4, n_workers=1,
                final_distance=2e-3, traced_kwargs=_TKW,
                final_leg='paraxial', transport=tr)
            row = {'design': name, 'transport': tr, 'leg': 'bare-final',
                   'raised': exc, 'n_kelly': len(kelly)}
            if res is not None:
                dxe = float(res.dx if not isinstance(res.dx, tuple)
                            else res.dx[0])
                row.update({'dx': dxe,
                            'R': (float(res.R)
                                  if not isinstance(res.R, tuple)
                                  else [float(v) for v in res.R]),
                            'n_stages': len(res.stages)})
                row.update(_spot_metrics(res.field, dxe))
            out['chain'].append(row)
            # ... and with the image-plane readout on
            key = f'D::{name}::{tr}::readout'
            res, exc, kelly = _run(
                key, tr, CA.propagate_traced_carrier_chain,
                e2, g, WL, dx2, r_in=np.inf, ray_subsample=4, n_workers=1,
                final_distance=2e-3, traced_kwargs=_TKW,
                final_leg='paraxial',
                focus_readout=dict(dx_out=0.25e-6, N_out=64), transport=tr)
            row = {'design': name, 'transport': tr, 'leg': 'focus-readout',
                   'raised': exc, 'n_kelly': len(kelly)}
            if res is not None:
                dxe = float(res.dx if not isinstance(res.dx, tuple)
                            else res.dx[0])
                row.update({'dx': dxe})
                row.update(_spot_metrics(res.field, dxe))
            out['chain'].append(row)
    return out


def main():
    out_path = sys.argv[2]
    rec = {
        'build': clib.build_tag(),
        'tree': _TREE,
        'lumenairy_version': la.__version__,
        'carrier_file': CA.__file__,
        'shipped_default_transport':
            __import__('inspect').signature(
                CA.propagate_carrier_referenced).parameters[
                    'transport'].default,
        'fixture': {'wavelength': WL, 'w_in': W_IN, 'N': N, 'dx': DX,
                    'R_converging': R_CONV},
        'ladder_a_single_step': ladder_a(),
    }
    rec.update(ladder_bcd())
    rec['warning_census'] = _CENSUS
    rec['warning_census_summary'] = {
        tr: {
            'n_calls': sum(1 for c in _CENSUS if c['transport'] == tr),
            'n_calls_with_kelly': sum(1 for c in _CENSUS
                                      if c['transport'] == tr
                                      and c['n_kelly'] > 0),
            'conditions': sorted({x for c in _CENSUS
                                  if c['transport'] == tr
                                  for x in c['conditions']}),
        }
        for tr in ('sziklas', 'collins')
    }
    clib.write_json(rec, out_path)
    s = rec['warning_census_summary']
    print(f"[probe_oracle_ladders] build={rec['build']} "
          f"default={rec['shipped_default_transport']!r} "
          f"kelly collins {s['collins']['n_calls_with_kelly']}"
          f"/{s['collins']['n_calls']}")


if __name__ == '__main__':
    main()
