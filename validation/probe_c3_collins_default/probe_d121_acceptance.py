"""WP-C3 -- the design-121 chain through BOTH transports.

Run as a CHILD process bound to ONE tree::

    python probe_d121_acceptance.py <tree> <out.json> [N]

WHY THIS EXISTS.  WP-B4 section 5 lists four gates it supplied and four it
did not, and the FIRST of the missing four is this one:

    "**the design-121 acceptance** (FWHM 3.450 / EE3 88.8 % / EE6 99.6 %, and
    the 8x4 Dammann fan).  Its assets are in
    ``validation/repro_traced_carrier_122/`` and are UNTRACKED on this
    machine -- the same reason WP-A6 sec. 6.1 deferred the whole item.  Until
    that runs, the flip is not defensible: it is the only fixture in the
    library where a whole DOE fan, a tilted congruence, a ``final_leg='auto'``
    route flip and a per-order readout tile all interact."

**The premise of that sentence no longer holds on this box, and that is the
first thing this probe records.**  The assets ARE here and ARE tracked:
``validation/repro_traced_carrier_121/`` (430 files under git), and the local
``.zmx`` plus the design-study runner both resolve at the paths
``_d121_common.py`` expects.  So the gate is runnable, and not running it
would be a choice rather than a constraint.

WHAT THIS IS AND IS NOT.  It is NOT the shipped acceptance runner
``focus_scan_121.py``: that one asks for N = 2048 with a fine retrace at
NFC = 8192 and a +/-DZ through-focus scan, and -- more importantly -- it
begins with ``sys.path.insert(0, r"D:\\...\\Lumenairy")``, which binds a
DIFFERENT checkout of the library than the one under test.  A number produced
that way would be a measurement of somebody else's tree, which is the exact
failure the anchor in :mod:`clib` exists to prevent.

So this probe drives the SAME geometry -- ``_d121_common.geometry()``, read
from the same ``.zmx`` -- through this tree's ``propagate_traced_carrier_chain``
on both transports at a grid the caller chooses, and reports the focus metrics
side by side.  The comparison between the two transports is exact (same
geometry, same grid, same process); the absolute numbers are the acceptance's
only at the acceptance's own N, and the JSON says which N it ran.
"""
from __future__ import annotations

import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_TREE = os.path.abspath(sys.argv[1])
sys.path.insert(0, _TREE)

import clib  # noqa: E402

import numpy as np  # noqa: E402

clib.anchor(_TREE)

import lumenairy as la                              # noqa: E402
import lumenairy.propagators.carrier as CA          # noqa: E402

_D121 = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'repro_traced_carrier_121')

LAM = 1.31e-6
W0 = 4e-6
TRAILING = 7.7058e-3
DXO = 0.25e-6
NOUT = 256


def _metrics(field, dx_out):
    """FWHM and encircled energy at 3 / 6 / 12 um radius, on the returned
    window, by the same definitions the acceptance runner uses: a linear
    interpolation of the half-maximum crossing on the brightest row, and
    cumulative power inside a radius as a percentage of the window's."""
    I = np.abs(np.asarray(field)) ** 2
    n = I.shape[-1]
    ax = (np.arange(n) - n / 2.0) * dx_out
    pk = float(I.max())
    if pk <= 0.0:
        return {'peak': 0.0}
    iy, ix = np.unravel_index(int(np.argmax(I)), I.shape)
    row = I[iy]
    half = pk / 2.0
    xs = []
    for d in (-1, 1):
        j = ix
        while 0 < j < n - 1 and row[j] > half:
            j += d
        a, b = row[j], row[j - d]
        t = 0.0 if b == a else (half - a) / (b - a)
        xs.append(ax[j] + t * (ax[j - d] - ax[j]))
    r2 = (ax - ax[ix])[None, :] ** 2 + (ax - ax[iy])[:, None] ** 2
    tot = float(I.sum())
    ee = {f'ee{int(r * 1e6)}': float(I[r2 <= r * r].sum() / tot * 100.0)
          for r in (3e-6, 6e-6, 12e-6)}
    return dict(peak=pk, fwhm_um=abs(xs[1] - xs[0]) * 1e6,
                power=tot * dx_out * dx_out,
                centroid_x_um=float((I.sum(axis=0) * ax).sum() / tot * 1e6),
                **ee)


def main():
    out_path = sys.argv[2]
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 512

    sys.path.insert(0, _D121)
    # The sibling design-121 checkout asserts that the ``lumenairy`` it
    # imported is the one ``LUMENAIRY_ROOT`` names, and dies with an
    # AssertionError naming that variable when it is unset -- so the d121
    # rows were not reproducible from the repository (VERIFY-WP-C3 D12).
    # Point it at the tree this probe already anchored.
    os.environ['LUMENAIRY_ROOT'] = _TREE
    rec = {'build': clib.build_tag(), 'tree': _TREE,
           'lumenairy': la.__file__, 'N': N, 'wavelength': LAM}
    try:
        import _d121_common as D
    except Exception as exc:                          # noqa: BLE001
        rec['assets'] = f'UNAVAILABLE: {type(exc).__name__}: {exc}'
        clib.write_json(rec, out_path)
        print(f"[probe_d121] assets unavailable: {exc}")
        return
    rec['assets'] = 'present'
    rec['zmx'] = D.ZMX
    rec['zmx_exists'] = os.path.isfile(D.ZMX)
    if not rec['zmx_exists']:
        clib.write_json(rec, out_path)
        print('[probe_d121] the .zmx is not on this box')
        return

    pre, post, gap_to_doe, period = D.geometry()
    rec['n_groups_pre'] = len(pre)
    rec['n_groups_post'] = len(post)
    rec['doe_period_um'] = period * 1e6

    # The UNDIFFRACTED (order-0) chain: pre-DOE groups, the DOE gap folded
    # into the first post group, then the post groups.  That is the chain the
    # focus metrics are taken on; the 8x4 fan is one tilted congruence per
    # order of the same chain and is NOT run here (see the docstring).
    groups = list(pre)
    if post:
        post = [dict(post[0], gap_before=post[0]['gap_before'] + gap_to_doe)] \
            + list(post[1:])
        groups += post

    # THE DESIGN'S OWN LAUNCH, copied from ``focus_scan_121.py`` rather than
    # invented: a ``w0 = 4 um`` waist at ``z1 = 2 mm`` before the first
    # surface, so the entering beam is the diverging one the prescription is
    # designed around, on the runner's own pitch law ``dx0 = 1 um * 2048/N``.
    # An invented collimated launch reads FWHM 25 um at N = 256 and 4435 um at
    # N = 1024 -- i.e. it measures the launch and not the chain.
    zR = np.pi * W0 * W0 / LAM
    z1 = 2e-3
    w_z1 = W0 * np.sqrt(1.0 + (z1 / zR) ** 2)
    R1 = z1 * (1.0 + (zR / z1) ** 2)
    dx0 = 1.0e-6 * 2048.0 / N
    x = (np.arange(N) - N // 2) * dx0
    env = np.exp(-(x[None, :] ** 2 + x[:, None] ** 2)
                 / (w_z1 * w_z1)).astype(np.complex128)
    rec['launch'] = {'w0_um': W0 * 1e6, 'z1_mm': z1 * 1e3,
                     'w_z1_mm': w_z1 * 1e3, 'R1_mm': R1 * 1e3,
                     'dx0_um': dx0 * 1e6}

    fr = dict(dx_out=DXO, N_out=NOUT)
    tkw = dict(on_undersample='silent', on_noncollimated='silent')
    rows = []
    for tr in ('sziklas', 'collins'):
        t0 = time.perf_counter()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            try:
                res = CA.propagate_traced_carrier_chain(
                    env, groups, LAM, dx0, r_in=R1, ray_subsample=4,
                    n_workers=1, traced_kwargs=tkw, final_leg='paraxial',
                    final_distance=TRAILING, focus_readout=fr, transport=tr)
                row = {'transport': tr, 'raised': None,
                       'seconds': round(time.perf_counter() - t0, 2)}
                row.update(_metrics(res.field, DXO))
                st = res.stages[-1]
                row['readout_route'] = st.get('readout_route')
                row['readout_route_k1'] = st.get('readout_route_k1')
                row['readout_route_reason'] = st.get('readout_route_reason')
                row['n_stages'] = len(res.stages)
                row['collins_forms'] = [s.get('collins_form')
                                        for s in res.stages
                                        if 'collins_form' in s]
            except Exception as exc:                  # noqa: BLE001
                row = {'transport': tr,
                       'raised': f'{type(exc).__name__}: {exc}'[:400],
                       'seconds': round(time.perf_counter() - t0, 2)}
        row['n_kelly'] = len([x for x in w
                              if 'under-sampled' in str(x.message)])
        rows.append(row)
    rec['rows'] = rows
    if all(r.get('raised') is None for r in rows):
        a, b = rows
        rec['delta'] = {
            'fwhm_um': b['fwhm_um'] - a['fwhm_um'],
            'ee3_points': b['ee3'] - a['ee3'],
            'ee6_points': b['ee6'] - a['ee6'],
            'peak_ratio': b['peak'] / max(a['peak'], 1e-300),
        }
    clib.write_json(rec, out_path)
    for r in rows:
        print(f"[probe_d121] {r['transport']:8s} "
              + (r['raised'] if r['raised'] else
                 f"FWHM {r['fwhm_um']:.4f} um  EE3 {r['ee3']:.2f}  "
                 f"EE6 {r['ee6']:.2f}  route {r['readout_route']} "
                 f"(K1 {r['readout_route_k1']})  kelly {r['n_kelly']}  "
                 f"{r['seconds']:.1f} s"))


if __name__ == '__main__':
    main()
