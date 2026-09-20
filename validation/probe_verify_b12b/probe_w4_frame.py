"""VERIFY-WP-B12b probe W4 -- is the new test file's explicit 4-pixel beamlet
frame a DERIVED cost choice or a hidden pin?

``tests/unit/test_audit2609_b12b_gbd_projection.py`` names
``sample_step=4, waist_factor=4.0`` and records, in a comment, that this frame
"reproduces the dense-frame field at a fidelity of 0.999587".  That number is
re-taken here on the SAME fixture the test file uses (N-BAF10 biconvex
R1 = +11.0 mm, t = 0.9 mm, semi = 0.30 mm at 1.064 um, flat-base aspheric last
surface, 96 x 6.6 um), on this build, and the DECISION the frame carries --
"the repaired field scores > 0.99 against a diffraction oracle and the
pre-repair one < 0.90" -- is re-scored at the library's OWN auto frame, which
is what a caller gets.

Also reported: the same frame ladder on one of MY fixtures, and the control
on the catalogue glass (my typed Sellmeier against ``get_glass_index``).

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
    SELLMEIER,
    VFixture,
    assert_tree,
    dump,
    env_block,
    fidelity,
    fixtures,
    n_sellmeier,
    rel_l2,
    sha,
)


def builder_fixture():
    """WP-B12b's own test fixture, described in MY fixture class so my own
    3-D tracer and diffraction oracle can score it."""
    return VFixture(
        'b12b_flat_base_asphere',
        "WP-B12b's own test fixture: N-BAF10 biconvex R1 = +11.0 mm, "
        "t = 0.9 mm, semi = 0.30 mm at 1.064 um, FLAT-BASE aspheric last "
        "surface (A2 = -9.0e1, A4 = 4.0e8), on its own 96 x 6.6 um grid",
        last={'radius': float('inf'),
              'aspheric_coeffs': {2: -9.0e1, 4: 4.0e8}},
        R1=11.0e-3, R2=float('inf'), t=0.9e-3, semi=0.30e-3,
        glass='N-BAF10', lam=1.064e-6, w0=0.18e-3, N=96, dx=6.6e-6)


def builder_conic():
    return VFixture(
        'b12b_conic',
        "WP-B12b's own optic with its CONIC last surface -- the control the "
        "builder's suite never scores against a diffraction oracle",
        R1=11.0e-3, R2=-11.0e-3, t=0.9e-3, semi=0.30e-3,
        glass='N-BAF10', lam=1.064e-6, w0=0.18e-3, N=96, dx=6.6e-6)


class ForceSurface:
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


def field(fx, z, frame):
    import lumenairy as la
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return np.asarray(la.apply_real_lens_gbd(
            fx.E_in(), prescription=fx.prescription(), wavelength=fx.lam,
            dx=fx.dx, output_plane_distance=float(z), **frame))


def main():
    assert_tree()
    import lumenairy as la
    from lumenairy.propagators.gbd import (
        apply_prescription_persurface_to_beamlets as F_,
    )
    arm = 'pre' if '_Rl = float(' in inspect.getsource(F_) else 'post'
    out = dict(env=env_block(), arm=arm, glass_control=[], rows=[])

    # -- control: my typed Sellmeier against the library's own registry -----
    for nm in SELLMEIER:
        for lam in (1.064e-6, 780e-9, 633e-9):
            mine = n_sellmeier(nm, lam)
            theirs = float(la.get_glass_index(nm, lam))
            out['glass_control'].append(
                dict(glass=nm, lam=lam, mine=mine, lib=theirs,
                     diff=abs(mine - theirs)))
    print('glass control max diff',
          max(r['diff'] for r in out['glass_control']), flush=True)

    frames = [('auto (library default)', dict()),
              ('ss2/wf2', dict(sample_step=2, waist_factor=2.0)),
              ('ss4/wf4 (the test file)', dict(sample_step=4,
                                               waist_factor=4.0)),
              ('ss6/wf6', dict(sample_step=6, waist_factor=6.0))]

    # My own aspheric fixture on a REDUCED grid: the auto frame is one
    # beamlet per pixel and the reconstruction degenerates to the dense
    # O(n_beamlets x N^2) sum, so N = 112 keeps the auto arm affordable while
    # the optic, the beam and the window are unchanged.
    mine = fixtures()['asph']
    mine.N, mine.dx, mine.semi, mine.w0 = 112, 3.6e-6, 0.17e-3, 0.105e-3
    _all = {'b12b_flat_base_asphere': builder_fixture(),
            'b12b_conic': builder_conic(), 'asph': mine}
    _want = os.environ.get('VB12B_W4_FIXTURES')
    _keys = ([k for k in _want.split(',') if k in _all] if _want
             else list(_all))
    for fx in [_all[k] for k in _keys]:
        zf = fx.best_focus()
        O, info = fx.oracle_field(zf)
        conv = fx.oracle_converge(zf)
        ref = None
        for label, fr in frames:
            t0 = time.time()
            F = field(fx, zf, fr)
            if ref is None:
                ref = F
            with ForceSurface():
                S = field(fx, zf, fr)
            row = dict(fixture=fx.key, frame=label, frame_kw=fr,
                       z=float(zf), secs=round(time.time() - t0, 1),
                       fid_vs_oracle=fidelity(F, O),
                       rel_l2_vs_oracle=rel_l2(F, O),
                       fid_vs_auto_frame=fidelity(F, ref),
                       forced_surface_fid_vs_oracle=fidelity(S, O),
                       peak=float(np.abs(F).max()), sha=sha(F)[:24],
                       airy=fx.airy_radius(), na=fx.numerical_aperture(),
                       dx_over_airy=fx.dx / fx.airy_radius(),
                       oracle_src_halved_infid=conv['src_halved_infidelity'],
                       oracle_grid_halved_infid=conv['grid_halved_infidelity'],
                       oracle_mode=info['mode'])
            out['rows'].append(row)
            print(f"{fx.key:26s} {label:26s} fid(oracle) "
                  f"{row['fid_vs_oracle']:.6f}  forced-surface "
                  f"{row['forced_surface_fid_vs_oracle']:.6f}  "
                  f"fid(vs auto frame) {row['fid_vs_auto_frame']:.6f}  "
                  f"({row['secs']} s)", flush=True)

    dump(out, f'probe_w4_frame_{arm}')


if __name__ == '__main__':
    main()
