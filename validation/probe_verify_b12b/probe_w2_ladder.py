"""VERIFY-WP-B12b probe W2 -- the oracle ladder, scored against MY OWN
band-limited angular-spectrum diffraction oracle.

Run inside whichever tree the interpreter resolves; the arm (``pre`` /
``post``) is DETECTED from the library source, never passed in.

What it adds over WP-B12b's own probe B: the oracle here is fully 3-D, so the
BICONIC, the XY-polynomial FREEFORM and the FIELD-FRAME decentred last
surfaces -- which WP-B12b could only report as field MOVEMENT, because the
oracle it imports is a rotationally-symmetric ring sum -- are scored for
ACCURACY like every other row.

Rows: every fixture at the exit vertex and at its own traced best focus.
Extras: the three local entry points and the ``world_output_plane`` branch,
and a ``forced-surface`` arm (``reference='surface'`` on both primitives,
reached through the shipped public keyword) so the "what does forcing the
other plane reconstruct?" claim is measured rather than argued.

Author: VERIFY-WP-B12b
"""
from __future__ import annotations

import inspect
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vb12b_common import (  # noqa: E402
    assert_tree,
    dump,
    env_block,
    fidelity,
    fixtures,
    rel_l2,
    sha,
)

# The beamlet frame every field row uses, named explicitly.  DERIVED from a
# cost/convergence measurement on this fixture set (see ``probe_w4_frame.py``,
# which scores it against the library's own auto frame): sample_step = 4 with
# a matching waist costs 45 s per call on this box against 98 s at 3 and
# ~20 min at the auto frame, and the three frames agree on the peak to
# 6e-04 relative.
FRAME = dict(sample_step=4, waist_factor=4.0)

LADDER = ('conic', 'conic_k', 'asph', 'flatbase_asph', 'bicon', 'freeform',
          'fieldframe', 'flat_last')
# ``VB12B_LADDER`` selects a subset (comma separated), or '' for none -- used
# to re-take only the forced arm and the entry points on the second build.
if 'VB12B_LADDER' in os.environ:
    _sel = [k for k in os.environ['VB12B_LADDER'].split(',') if k]
    LADDER = tuple(k for k in LADDER if k in _sel)
FORCED = ('conic', 'flatbase_asph', 'flat_last')
if 'VB12B_FORCED' in os.environ:
    _self = [k for k in os.environ['VB12B_FORCED'].split(',') if k]
    FORCED = tuple(k for k in FORCED if k in _self)


class ForceSurface:
    """Pin BOTH differential primitives to ``reference='surface'`` through
    the shipped public keyword -- no copy of any deleted code."""

    def __enter__(self):
        import lumenairy.raytrace.differential as D
        self.D = D
        self.saved = (D.ray_transfer_jacobian, D.ray_transfer_jacobian_analytic)

        def pin(base):
            def wrapped(*a, **kw):
                kw['reference'] = 'surface'
                return base(*a, **kw)
            return wrapped
        D.ray_transfer_jacobian = pin(self.saved[0])
        D.ray_transfer_jacobian_analytic = pin(self.saved[1])
        return self

    def __exit__(self, *exc):
        (self.D.ray_transfer_jacobian,
         self.D.ray_transfer_jacobian_analytic) = self.saved
        return False


def gbd_field(fx, z, **extra):
    import lumenairy as la
    kw = dict(FRAME)
    kw.update(extra)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return np.asarray(la.apply_real_lens_gbd(
            fx.E_in(), prescription=fx.prescription(), wavelength=fx.lam,
            dx=fx.dx, output_plane_distance=float(z), **kw))


def score(F, O):
    return dict(fidelity=fidelity(F, O), rel_l2=rel_l2(F, O),
                power_ratio=float((np.abs(F) ** 2).sum()
                                  / max((np.abs(O) ** 2).sum(), 1e-300)),
                peak=float(np.abs(F).max()), sha=sha(F)[:24])


def main():
    assert_tree()
    from lumenairy.propagators.gbd import (
        apply_prescription_persurface_to_beamlets as F_,
    )
    arm = 'pre' if '_Rl = float(' in inspect.getsource(F_) else 'post'
    print(f'ARM DETECTED FROM THE LIBRARY: {arm}')
    out = dict(env=env_block(), arm=arm, frame=FRAME, rows=[], entries=[])

    FX = fixtures()
    for key in LADDER:
        fx = FX[key]
        zf = fx.best_focus()
        conv = fx.oracle_converge(zf)
        for plane, z in (('exit_vertex', 0.0), ('focus', zf)):
            t0 = time.time()
            O, info = fx.oracle_field(z)
            G = gbd_field(fx, z)
            row = dict(key=key, plane=plane, z=float(z), arm=arm,
                       airy=fx.airy_radius(), na=fx.numerical_aperture(),
                       dx_over_airy=fx.dx / fx.airy_radius(),
                       oracle_mode=info['mode'], oracle_n_src=info['n_src'],
                       oracle_src_halved_infid=conv['src_halved_infidelity'],
                       oracle_grid_halved_infid=conv['grid_halved_infidelity'],
                       secs=round(time.time() - t0, 1))
            row.update(score(G, O))
            row['oracle_sha'] = sha(O)[:24]
            out['rows'].append(row)
            print(f"{arm:4s} {key:15s} {plane:11s} fid {row['fidelity']:.8f} "
                  f"relL2 {row['rel_l2']:.5f} P {row['power_ratio']:.4f} "
                  f"peak {row['peak']:.4f} sha {row['sha']} "
                  f"({row['secs']} s)", flush=True)

    # ---- the forced-surface arm (claim: it is PRE-v5.22, not v5.22-5.47) --
    for key in FORCED:
        fx = FX[key]
        zf = fx.best_focus()
        O, _i = fx.oracle_field(zf)
        G = gbd_field(fx, zf)
        with ForceSurface():
            S = gbd_field(fx, zf)
        row = dict(key=key, arm=arm, z=float(zf),
                   shipped=score(G, O), forced_surface=score(S, O),
                   forced_vs_shipped_rel_l2=rel_l2(S, G),
                   forced_vs_shipped_fidelity=fidelity(S, G),
                   bit_identical=bool(sha(S) == sha(G)))
        out.setdefault('forced', []).append(row)
        print(f"{arm:4s} FORCED-SURFACE {key:15s} shipped fid "
              f"{row['shipped']['fidelity']:.6f} forced fid "
              f"{row['forced_surface']['fidelity']:.6f} relL2(forced, "
              f"shipped) {row['forced_vs_shipped_rel_l2']:.4f} identical "
              f"{row['bit_identical']}", flush=True)

    # ---- the entry points, on the flat-base aspheric fixture -------------
    import lumenairy as la
    from lumenairy.propagators import gbd as G_
    fx = FX['flatbase_asph']
    zf = fx.best_focus()
    O, _i = fx.oracle_field(zf)
    E = fx.E_in()
    a = gbd_field(fx, zf)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        u = np.asarray(la.apply_real_lens_universal(
            E, prescription=fx.prescription(), wavelength=fx.lam, dx=fx.dx,
            method='gbd', output_plane_distance=float(zf),
            method_kwargs={'gbd': dict(FRAME)}))
        p = np.asarray(G_.propagate_gbd_through_prescription(
            E, fx.dx, fx.prescription(), wavelength=fx.lam,
            output_shape=(fx.N, fx.N), output_dx=fx.dx, per_surface=True,
            z_image=float(zf), **FRAME))
        b = G_.decompose_field_to_beamlets(E, fx.dx, wavelength=fx.lam,
                                           **FRAME)
        w = G_.apply_prescription_persurface_to_beamlets(
            b, fx.prescription(), fx.lam, world_output_plane='auto')
        wsha = sha(np.concatenate([
            np.asarray(w.positions).ravel(), np.asarray(w.directions).ravel(),
            np.asarray(w.Q).ravel().view(np.float64),
            np.asarray(w.amplitude).ravel().view(np.float64)]))
    out['entries'] = [
        dict(entry='apply_real_lens_gbd', arm=arm, **score(a, O)),
        dict(entry='apply_real_lens_universal(gbd)', arm=arm, **score(u, O),
             same_bytes_as_gbd=bool(sha(u) == sha(a)),
             fidelity_vs_gbd=fidelity(u, a)),
        dict(entry='propagate_gbd_through_prescription', arm=arm,
             **score(p, O), same_bytes_as_gbd=bool(sha(p) == sha(a)),
             fidelity_vs_gbd=fidelity(p, a)),
        dict(entry='world_output_plane branch (bundle digest)', arm=arm,
             sha=wsha[:24], n_beamlets=int(np.asarray(w.positions).shape[0])),
    ]
    for e in out['entries']:
        print(arm, e, flush=True)

    dump(out, f'probe_w2_ladder_{arm}')


if __name__ == '__main__':
    main()
