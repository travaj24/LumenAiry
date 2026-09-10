"""Q7 (D6) -- the banded traced call's route, price and counters.

Task 3.  Six routes per grid: {screen, ray_density} x {whole-grid, AUTO-banded,
AUTO-banded forced back onto the coarse-Newton incumbent with
``inverse_map=False``}.  Two modes:

* ``--mode wall``  -- UNinstrumented best-of-K wall clock plus the field hash.
  The third route of each pair is the CONTROL: on v5.44.0 and later it is the
  route v5.43.0's banded call selected silently, so its FIELD HASH must equal
  v5.43.0's banded answer bit for bit.
* ``--mode stages`` -- the same routes with ``InverseCharacteristic.eval_into``
  / ``domain_mask`` / ``build_inverse_map`` / ``map_coordinates`` wrapped, so
  the split is measured together with the LOAD-FREE counters: how many
  ``domain_mask`` calls, how many PIXELS they test (in whole grids) and how
  many channel evaluations per exit pixel.  The wrappers perturb the total --
  read the split and the counters, not the total.

My own fixture (the N-BAF10 meniscus at 1.55 um with a real spherical
carrier), not the builder's singlet.

Usage: python q7_d6_route.py <out.json> --tree <arm> --mode wall|stages
       [--n 4096] [--sub 32] [--reps 3]
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _vf  # noqa: E402


def _kw(N, dx, sub, model, inv):
    return dict(prescription=_vf.presc_meniscus(ap=4.4e-3),
                wavelength=_vf.WL, dx=dx, ray_subsample=sub, n_workers=1,
                on_undersample='silent', on_noncollimated='off',
                on_aperture_beam='silent', parallel_amp=False,
                carrier=0.055, amplitude_model=model, inverse_map=inv)


def routes(N, dx, sub):
    return [
        ('screen_whole', _kw(N, dx, sub, 'screen', True), 0),
        ('screen_banded', _kw(N, dx, sub, 'screen', True), None),
        ('screen_banded_noinv', _kw(N, dx, sub, 'screen', False), None),
        ('rd_whole', _kw(N, dx, sub, 'ray_density', True), 0),
        ('rd_banded', _kw(N, dx, sub, 'ray_density', True), None),
        ('rd_banded_noinv', _kw(N, dx, sub, 'ray_density', False), None),
    ]


class Counters(object):
    """Wrap the evaluator's hot methods; count calls, pixels and seconds."""

    def __init__(self, la):
        from lumenairy.elements import _lens_imap as IM
        from scipy import ndimage
        self.IM = IM
        self.ndimage = ndimage
        self.saved = {}
        self.d = {}

    def reset(self):
        self.d = {k: 0.0 for k in
                  ('eval_into_s', 'eval_into_n', 'eval_into_px',
                   'eval_into_ch', 'domain_mask_s', 'domain_mask_n',
                   'domain_mask_px', 'build_s', 'build_n', 'mapcoord_s',
                   'mapcoord_n')}

    def __enter__(self):
        IC = self.IM.InverseCharacteristic
        self.saved['eval_into'] = IC.eval_into
        self.saved['domain_mask'] = IC.domain_mask
        self.saved['build'] = self.IM.build_inverse_map
        self.saved['mapc'] = self.ndimage.map_coordinates
        d = self.d
        real_e, real_dm = IC.eval_into, IC.domain_mask
        real_b = self.IM.build_inverse_map
        real_m = self.ndimage.map_coordinates

        def eval_into(slf, Xg, Yg, out, channels=None, chunk=None):
            t = time.perf_counter()
            r = real_e(slf, Xg, Yg, out, channels=channels, chunk=chunk)
            d['eval_into_s'] += time.perf_counter() - t
            d['eval_into_n'] += 1
            d['eval_into_px'] += float(np.size(Xg))
            d['eval_into_ch'] += float(np.size(Xg)) * len(out)
            return r

        def domain_mask(slf, Xg, Yg, x_in=None, y_in=None, axes=None,
                        relax=0.0):
            t = time.perf_counter()
            r = real_dm(slf, Xg, Yg, x_in=x_in, y_in=y_in, axes=axes,
                        relax=relax)
            d['domain_mask_s'] += time.perf_counter() - t
            d['domain_mask_n'] += 1
            d['domain_mask_px'] += float(np.size(r))
            return r

        def build(*a, **kw):
            t = time.perf_counter()
            r = real_b(*a, **kw)
            d['build_s'] += time.perf_counter() - t
            d['build_n'] += 1
            return r

        def mapc(*a, **kw):
            t = time.perf_counter()
            r = real_m(*a, **kw)
            d['mapcoord_s'] += time.perf_counter() - t
            d['mapcoord_n'] += 1
            return r

        IC.eval_into = eval_into
        IC.domain_mask = domain_mask
        self.IM.build_inverse_map = build
        self.ndimage.map_coordinates = mapc
        import lumenairy.elements._lens_traced as LT
        self.saved['LT_mapc'] = getattr(LT, 'map_coordinates', None)
        if self.saved['LT_mapc'] is not None:
            LT.map_coordinates = mapc
        return self

    def __exit__(self, *a):
        IC = self.IM.InverseCharacteristic
        IC.eval_into = self.saved['eval_into']
        IC.domain_mask = self.saved['domain_mask']
        self.IM.build_inverse_map = self.saved['build']
        self.ndimage.map_coordinates = self.saved['mapc']
        import lumenairy.elements._lens_traced as LT
        if self.saved['LT_mapc'] is not None:
            LT.map_coordinates = self.saved['LT_mapc']
        return False


def main():
    p = _vf.argp(__doc__)
    p.add_argument('--mode', default='wall', choices=('wall', 'stages'))
    p.add_argument('--n', type=int, default=4096)
    p.add_argument('--sub', type=int, default=32)
    p.add_argument('--dx', type=float, default=1.5e-6)
    p.add_argument('--reps', type=int, default=3)
    args = p.parse_args()
    la = _vf.banner(args.tree)
    N, dx, sub = args.n, args.dx, args.sub
    print(f"# N={N} dx={dx} sub={sub} mode={args.mode} reps={args.reps} "
          f"free={_vf.free_gb()} GB", flush=True)
    E = _vf.sph(N, dx, 0.30 * N * dx / 2.0, 0.055)
    out = {}
    ctr = Counters(la) if args.mode == 'stages' else None
    for name, kw, rows in routes(N, dx, sub):
        best, hh, rec_keep, cnt = np.inf, None, None, None
        for _ in range(args.reps):
            if ctr is not None:
                ctr.reset()
                with ctr:
                    t0 = time.perf_counter()
                    F, rec, wl = _vf.run_traced(la, E, kw, rows)
                    el = time.perf_counter() - t0
                c = dict(ctr.d)
            else:
                t0 = time.perf_counter()
                F, rec, wl = _vf.run_traced(la, E, kw, rows)
                el = time.perf_counter() - t0
                c = {}
            if el < best:
                best, hh, rec_keep, cnt = el, _vf.h(F), _vf.rec_record(rec), c
            del F
        row = {'secs_best': round(best, 3), 'hash': hh, 'rows_arg': str(rows),
               'rec': rec_keep, 'n_warn': len(wl)}
        if cnt:
            g = float(N) * N
            row['counters'] = {
                'eval_into_calls': int(cnt['eval_into_n']),
                'eval_into_s': round(cnt['eval_into_s'], 3),
                'channel_evals_per_px': round(cnt['eval_into_ch'] / g, 3),
                'domain_mask_calls': int(cnt['domain_mask_n']),
                'domain_mask_s': round(cnt['domain_mask_s'], 3),
                'domain_mask_grids': round(cnt['domain_mask_px'] / g, 3),
                'build_inverse_map_calls': int(cnt['build_n']),
                'build_inverse_map_s': round(cnt['build_s'], 3),
                'map_coordinates_calls': int(cnt['mapcoord_n']),
                'map_coordinates_s': round(cnt['mapcoord_s'], 3),
                'rest_s': round(best - cnt['eval_into_s']
                                - cnt['domain_mask_s'] - cnt['build_s']
                                - cnt['mapcoord_s'], 3),
            }
        out[name] = row
        msg = (f"  {name:20s} best {best:8.3f} s  {hh}")
        if cnt:
            cc = row['counters']
            msg += (f"  eval={cc['eval_into_s']:.2f}s/{cc['eval_into_calls']}"
                    f"@{cc['channel_evals_per_px']:.2f}ch/px  "
                    f"dmask={cc['domain_mask_s']:.2f}s/"
                    f"{cc['domain_mask_calls']}@"
                    f"{cc['domain_mask_grids']:.2f}grids  "
                    f"build={cc['build_inverse_map_s']:.2f}  "
                    f"mapc={cc['map_coordinates_s']:.2f}/"
                    f"{cc['map_coordinates_calls']}")
        print(msg, flush=True)
    for a, b in (('screen_banded', 'screen_whole'),
                 ('rd_banded', 'rd_whole')):
        out[f'ratio_{a}_over_{b}'] = round(out[a]['secs_best']
                                           / out[b]['secs_best'], 4)
        print(f"  RATIO {a}/{b} = {out[f'ratio_{a}_over_{b}']}", flush=True)
    _vf.dump(args, {'cases': out, 'N': N, 'dx': dx, 'sub': sub,
                    'mode': args.mode, 'reps': args.reps,
                    'free_gb': _vf.free_gb()})


if __name__ == '__main__':
    main()
