"""R5 -- VERIFY-WP-B12b D-4's two headline numbers, RE-MEASURED.

The refusal that ships quotes "the spot RMS is 7.8e3 x the traced one at
z_image = +f" and "the leg piston carries the wrong sign, 0.48 waves at
z_image = -f".  Those are VERIFY-WP-B12b section 9.2's numbers.  A refusal
whose message quotes a number nobody re-measured is exactly the
right-conclusion-wrong-numbers shape ``docs/TESTING_STANDARDS.md`` warns
about, so this probe re-takes both on the PRE tree (a ``git archive`` of the
integration tip, where the branch still serves the class) and records what
the POST tree does instead.

BOTH SIGNS of ``z_image`` are taken, because they fail differently:

* ``z_image = +|f|`` -- the sign a caller reading "back focal distance"
  would use.  ``Nz2 = 1/sqrt(1+u^2)`` is positive whatever the true ``N``,
  and ``dt.ux = L/N`` has already flipped with ``N``, so the returned
  transverse positions are the TRUTH AT ``z = +|f|`` while the light is at
  ``z = -|f|``: defocused by twice the focal length.
* ``z_image = -|f|`` -- the positions land on the true focus (the two sign
  flips cancel), and what is left is the LEG: the branch adds
  ``t = z_image * sec`` of optical path where the ray has travelled
  ``+|f| * sec``, so the piston has the wrong sign.

The transverse truth is the verifier's own 3-D tracer; the leg is measured
the way its own test file measures an image leg -- a hand-built bundle at
``z_image = 0`` and at ``z_image = z``, the phase difference of the beamlet
amplitudes, compared against a prediction with the circular mean removed so
a constant piston cannot hide inside a wrap.

Author: Andrew Traverso
"""
from __future__ import annotations

import warnings

import numpy as np
import r2_common as C


def _truth(fx):
    """The verifier's 3-D tracer: exit-vertex state, the geometric focus, and
    the true spot RMS there."""
    zf = fx.best_focus()
    m = 121
    h = np.linspace(fx.semi / (2 * m), fx.semi * 0.99, m)
    v = fx.exit_rays(h, np.zeros_like(h))
    w = np.exp(-2.0 * (h / fx.w0) ** 2)
    w = w / w.sum()
    ux, uy = v['L'] / v['N'], v['M'] / v['N']

    def rms_at(z):
        xz = v['x'] + ux * z
        yz = v['y'] + uy * z
        cx, cy = (w * xz).sum(), (w * yz).sum()
        return float(np.sqrt((w * ((xz - cx) ** 2 + (yz - cy) ** 2)).sum()))

    return dict(z_focus=float(zf), n_sign=float(np.sign(np.median(v['N']))),
                spot_rms_at_focus=rms_at(zf),
                spot_rms_at_minus_focus=rms_at(-zf),
                h=h, v=v, w=w, sec=np.sqrt(1.0 + ux ** 2 + uy ** 2))


def _local_arm(G, presc, fx, z, tr):
    """The LOCAL branch at one ``z_image``: the returned spot RMS, the sign of
    the returned direction, and the leg piston against the TRUE optical path.

    Both are taken from a hand-built bundle on the SAME launch heights the
    oracle traced, so the comparison is ray for ray.
    """
    lam = fx.lam
    k0 = 2.0 * np.pi / lam
    h = tr['h']
    y = np.zeros_like(h)
    bl = G.BeamletBundle(
        positions=np.stack([h, y, np.zeros_like(h)], -1),
        directions=np.stack([y, y, np.ones_like(h)], -1),
        Q=np.full(h.size, -1j * lam / (np.pi * (4 * fx.dx) ** 2),
                  dtype=np.complex128),
        amplitude=np.ones(h.size, dtype=np.complex128),
        waist0=np.full(h.size, 4.0 * fx.dx))
    out = {}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r0 = G.apply_prescription_persurface_to_beamlets(
                bl, presc, lam, z_image=0.0)
            rz = G.apply_prescription_persurface_to_beamlets(
                bl, presc, lam, z_image=float(z))
    except NotImplementedError as exc:
        return dict(decision='refused', message=str(exc)[:400])
    out['decision'] = 'served'
    p = np.asarray(rz.positions)
    d = np.asarray(rz.directions)
    a0 = np.asarray(r0.amplitude)
    az = np.asarray(rz.amplitude)
    n = min(p.shape[0], h.size)
    w = tr['w'][:n]
    w = w / w.sum()
    cx = float((w * p[:n, 0]).sum())
    cy = float((w * p[:n, 1]).sum())
    out['spot_rms'] = float(np.sqrt(
        (w * ((p[:n, 0] - cx) ** 2 + (p[:n, 1] - cy) ** 2)).sum()))
    out['n_sign_returned'] = float(np.sign(np.median(d[:, 2])))
    out['n_beamlets'] = int(p.shape[0])

    # the leg: what the branch ADDED between z_image = 0 and z_image = z
    leg = np.angle(az[:n] * np.conj(a0[:n]))
    sec = tr['sec'][:n]
    preds = {
        # the TRUE optical path to that plane: the ray travels |z| * sec of
        # geometric path to reach it, whichever way it is going
        'true_abs': k0 * abs(float(z)) * sec,
        # what an index-free forward-going leg would add
        'signed': k0 * float(z) * sec,
    }
    for name, pred in preds.items():
        dphi = np.angle(np.exp(1j * (leg - pred)))
        piston = np.angle(np.mean(np.exp(1j * dphi)))
        out[f'leg_resid_{name}_waves'] = float(np.nanmax(np.abs(
            np.angle(np.exp(1j * (dphi - piston)))))) / (2 * np.pi)
        out[f'leg_piston_{name}_waves'] = float(piston) / (2 * np.pi)
    return out


def main():
    C.assert_tree()
    from lumenairy.propagators import gbd as G
    arm, tokens = C.detect_arm()
    out = dict(env=C.env_block(), arm=arm, arm_tokens=tokens, fixtures={})
    print(f'arm = {arm}  {tokens}')

    C.register_r2_media()
    fixtures = {'verifier_mirror_R15': C.VB.fixture('mirror')}

    for key, fx in fixtures.items():
        tr = _truth(fx)
        zf = tr['z_focus']
        row = dict(z_focus=zf, n_sign_traced=tr['n_sign'],
                   spot_rms_at_focus=tr['spot_rms_at_focus'],
                   spot_rms_at_minus_focus=tr['spot_rms_at_minus_focus'],
                   arms={})
        presc = fx.prescription()
        for label, z in (('z_image=+|f|', abs(zf)), ('z_image=-|f|', -abs(zf))):
            a = _local_arm(G, presc, fx, z, tr)
            if a.get('decision') == 'served':
                a['spot_rms_ratio_vs_focus'] = (
                    a['spot_rms'] / tr['spot_rms_at_focus'])
            row['arms'][label] = a
            print(f'  {key} {label:14s} {a.get("decision")} '
                  f'rms={a.get("spot_rms")} '
                  f'ratio={a.get("spot_rms_ratio_vs_focus")} '
                  f'Nsign={a.get("n_sign_returned")} '
                  f'resid_true={a.get("leg_resid_true_abs_waves")} '
                  f'piston_true={a.get("leg_piston_true_abs_waves")}')
        out['fixtures'][key] = row
        print(f'  {key} truth z_focus={zf:.6e} rms@f={tr["spot_rms_at_focus"]:.3e} '
              f'rms@-f={tr["spot_rms_at_minus_focus"]:.3e} '
              f'Nsign={tr["n_sign"]:+.0f}')

    C.dump(out, 'probe_r5_mirror_' + arm)


if __name__ == '__main__':
    main()
