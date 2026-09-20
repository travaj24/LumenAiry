"""VERIFY-WP-B7c round 3 -- the CONVERGED reference and the LATTICE A/B.

Two of round 3's load-bearing measurements, re-taken here:

**Claim 5 -- what a converged render reads.**  Round 2 published "1 exactly,
on any optic, at any plane, at any grid"; round 3 corrected that to
``0.9994103 .. 1.0004414`` (rms 1.48e-4) over 102 planes, with a WINDOW ladder
showing 1.4e-3 on an undersized window and 1.2e-4 on every larger one, and a
spread of 2.1e-2 on healthy FOLD planes -- 36x more.  Reproduced here on MY
optics, on planes chosen by a GEOMETRIC criterion (at least a quarter of the
paraxial focal distance short of the interior fold, so no reading selects the
planes its own spread is then measured on).

**Claim 6 -- the hull-aligned alternative E5 asked for.**  The shipped
convention NESTS the fine lattice on the coarse one
(``_HALF_PITCH_CENTRE_OFFSET = 0``); the alternative offsets it by half a fine
pixel so the two Voronoi HULLS coincide (``offset = -0.5``, derived below, not
guessed).  Round 3 measured the ~4-per-halving identity becoming unbounded
(7343.9 where nesting reads 3.998).  Both conventions are run here on the same
planes, through ``monkeypatch``-free module-attribute assignment, with the
RETURNED FIELD compared byte for byte so the claim "the lattice is the
arbiter's alone" is measured rather than assumed.

Usage:
    python v3control.py <mode> <out.json>
        mode = converged | gridladder | windowladder | fold | ab | jump
"""
from __future__ import annotations

import json
import sys
import warnings

import numpy as np
import v3fixtures as FX
import v3geom as G
import v3oracle as OR
import v3scan as S

#: the hull-aligned offset, DERIVED rather than copied.  A render of ``N_r``
#: pixels at pitch ``dx_r`` has centres ``(j - N_r/2 + o) dx_r`` and Voronoi
#: hull ``[(-N_r/2 + o) dx_r - dx_r/2, ...]``.  Setting the FINE hull's lower
#: edge (``N_r = 2N``, ``dx_r = dx/2``) equal to the COARSE one's
#: (``-(N + 1) dx / 2``) gives ``o = -1/2``.
HULL_ALIGNED_OFFSET = -0.5


def _call(fx, z, N=None, dx=None, w0=None, offset=None):
    """One completion call, bars lifted, optionally on another grid and/or
    another half-pitch lattice offset."""
    from lumenairy.elements import (
        _lens_traced_multibranch as MB,
        _lens_traced_uniform as U,
    )

    N = int(fx['N'] if N is None else N)
    dx = float(fx['dx'] if dx is None else dx)
    w0 = float(fx['w0'] if w0 is None else w0)
    b = S.bars()
    S.lift_bars()
    old = MB._HALF_PITCH_CENTRE_OFFSET
    if offset is not None:
        MB._HALF_PITCH_CENTRE_OFFSET = float(offset)
    try:
        E_in = FX.gauss(N, dx, w0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return U.apply_real_lens_traced_uniform(
                E_in, prescription=fx['prescription'],
                wavelength=fx['wavelength'], dx=dx,
                output_plane_distance=float(z), ray_subsample=2, n_fan=4000,
                return_diagnostics=True)
    finally:
        MB._HALF_PITCH_CENTRE_OFFSET = old
        S.restore_bars(b)


def converged_planes(fx, n=8):
    """Planes at least a quarter of the paraxial focal distance SHORT of the
    interior fold's onset -- a geometric criterion, no reading in it."""
    presc, wl = fx['prescription'], fx['wavelength']
    _na, fp, fm, _h, _f = G.na_and_foci(presc, wl)
    zs_geom = np.linspace(0.30 * min(fp, fm), 1.45 * max(fp, fm), 140)
    zt = [z for z, rc, _ in G.fold_window(presc, wl, zs_geom) if rc]
    onset = min(zt) if zt else min(fp, fm)
    hi = onset - 0.25 * fp
    lo = max(0.12 * fp, 0.18 * hi)
    if hi <= lo:
        return []
    return np.linspace(lo, hi, n).tolist()


def mode_converged(out):
    rows = []
    for nm in FX.OPTICS:
        fx = FX.FIXTURES[nm]
        for z in converged_planes(fx):
            _E, d = _call(fx, z)
            c = d.get('pixel_continuity')
            r = dict(fixture=nm, z_um=z * 1e6, reading=(None if c is None
                                                        else float(c)),
                     reason=d.get('reason'), fell_back=bool(d.get('fell_back')),
                     scope=d.get('pixel_continuity_scope'))
            rows.append(r)
            print(json.dumps(r), flush=True)
    vals = [r['reading'] for r in rows if r['reading'] is not None]
    dev = [abs(v - 1.0) for v in vals]
    rep = dict(mode='converged', n=len(vals), n_optics=len(FX.OPTICS),
               lo=min(vals), hi=max(vals), worst_dev=max(dev),
               rms_dev=float(np.sqrt(np.mean(np.asarray(dev) ** 2))),
               rows=rows)
    _write(out, rep)


def mode_gridladder(out):
    """Window HELD, ``N`` and ``dx`` both scaled -- the discretisation ladder.

    Run on the slowest optic in the population at a plane chosen by the same
    geometric criterion, so the render is converged for a reason that has
    nothing to do with the reading.
    """
    fx = FX.FIXTURES['W']
    z = converged_planes(fx)[-1]
    window = fx['N'] * fx['dx']
    rows = []
    for N in (128, 192, 256, 384, 512, 768, 1024):
        dx = window / N
        _E, d = _call(fx, z, N=N, dx=dx)
        c = d.get('pixel_continuity')
        r = dict(N=N, dx_um=dx * 1e6, window_um=window * 1e6,
                 reading=None if c is None else float(c),
                 dev=None if c is None else abs(float(c) - 1.0))
        rows.append(r)
        print(json.dumps(r), flush=True)
    _write(out, dict(mode='gridladder', fixture='W', z_um=z * 1e6, rows=rows))


def mode_windowladder(out):
    """``dx`` HELD, ``N`` grows so the WINDOW grows -- E5's own mechanism.

    A window too small to contain the field forces the two lattices' hull
    mismatch to matter; once the window holds the field it should not.
    """
    fx = FX.FIXTURES['W']
    z = converged_planes(fx)[-1]
    dx = 3.00e-6
    rows = []
    for N in (192, 256, 384, 512, 640, 896):
        _E, d = _call(fx, z, N=N, dx=dx)
        c = d.get('pixel_continuity')
        r = dict(N=N, dx_um=dx * 1e6, window_um=N * dx * 1e6,
                 reading=None if c is None else float(c),
                 reading_repr=None if c is None else repr(float(c)),
                 dev=None if c is None else abs(float(c) - 1.0))
        rows.append(r)
        print(json.dumps(r), flush=True)
    big = [r['reading_repr'] for r in rows if r['N'] >= 512]
    _write(out, dict(mode='windowladder', fixture='W', z_um=z * 1e6, rows=rows,
                     bit_identical_N512_640_896=(len(set(big)) == 1),
                     large_window_readings=big))


def mode_fold(out):
    """The same reading on HEALTHY FOLD planes -- where the guard works.

    "Healthy" is defined against MY oracle and not against the reading:
    fidelity >= 0.99 AND returned power within 2 % of the oracle's.
    """
    rows = []
    for nm in FX.OPTICS:
        fx = FX.FIXTURES[nm]
        try:
            band, _g, _r = _recon(fx)
        except Exception as exc:                            # noqa: BLE001
            print(nm, 'skipped:', exc, flush=True)
            continue
        for z in np.linspace(band[0], band[1], 9):
            E, d = _call(fx, z)
            if d.get('reason') != 'fold_ring' or d.get('fell_back'):
                continue
            c = d.get('pixel_continuity')
            if c is None:
                continue
            E_or, _P = S.oracle_field(fx, z, refine=3)
            fid = OR.fidelity(E, E_or)
            por = OR.power(E, fx['dx']) / max(OR.power(E_or, fx['dx']), 1e-300)
            r = dict(fixture=nm, z_um=z * 1e6, reading=float(c), fidelity=fid,
                     power_over_oracle=por,
                     healthy=bool(fid >= 0.99 and abs(por - 1.0) <= 0.02))
            rows.append(r)
            print(json.dumps(r), flush=True)
    good = [r for r in rows if r['healthy']]
    dev = [abs(r['reading'] - 1.0) for r in good]
    _write(out, dict(mode='fold', n=len(rows), n_healthy=len(good),
                     lo=min((r['reading'] for r in good), default=None),
                     hi=max((r['reading'] for r in good), default=None),
                     worst_dev=max(dev, default=None), rows=rows))


def _recon(fx):
    import v3ladder as L
    return L.recon_band(fx, n=26)


def mode_ab(out):
    """NESTED (shipped) against HULL-ALIGNED, on the same planes."""
    rows = []
    cases = [('V', 1761e-6, 'the claim-2 plane'),
             ('V', 1768e-6, "VERIFY-B7b's blow-up plane"),
             ('V', 1764e-6, 'just past the fold-ring edge'),
             ('V', 1000e-6, 'far from every caustic'),
             ('W', 4900e-6, 'a slow optic inside its fold ring'),
             ('VX', 900e-6, 'the NA 0.44 optic'),
             ('VA', 2900e-6, 'the asphere')]
    for nm, z, note in cases:
        fx = FX.FIXTURES[nm]
        E0, d0 = _call(fx, z, offset=0.0)
        E1, d1 = _call(fx, z, offset=HULL_ALIGNED_OFFSET)
        E_or, _P = S.oracle_field(fx, z, refine=3)
        a, b = np.asarray(E0), np.asarray(E1)
        r = dict(fixture=nm, z_um=z * 1e6, note=note,
                 nested=None if d0.get('pixel_continuity') is None
                 else float(d0['pixel_continuity']),
                 hull_aligned=None if d1.get('pixel_continuity') is None
                 else float(d1['pixel_continuity']),
                 nested_branch_sum=None
                 if d0.get('multibranch_pixel_continuity') is None
                 else float(d0['multibranch_pixel_continuity']),
                 hull_branch_sum=None
                 if d1.get('multibranch_pixel_continuity') is None
                 else float(d1['multibranch_pixel_continuity']),
                 reason=d0.get('reason'), fell_back=bool(d0.get('fell_back')),
                 field_bit_identical=bool(a.dtype == b.dtype
                                          and a.shape == b.shape
                                          and a.tobytes() == b.tobytes()),
                 fidelity_nested=OR.fidelity(E0, E_or),
                 fidelity_hull=OR.fidelity(E1, E_or))
        rows.append(r)
        print(json.dumps(r), flush=True)
    _write(out, dict(mode='ab', hull_aligned_offset=HULL_ALIGNED_OFFSET,
                     rows=rows))


def mode_jump(out):
    """What CHANGES across the bar at a picometre separation.

    Round 3 explains the ladder's behaviour by the reading being "not smooth
    in z near a fold onset", which reads as a steep but continuous crossing.
    Bisecting between a returned and a refused fold plane to picometres and
    reading the WHOLE diagnostics on both sides says which it is, and says it
    with a quantity rather than an adjective.
    """
    fx = FX.FIXTURES['V']
    lo, hi = 1761e-6, 1764e-6
    bar = S.bars()['cmax']
    rows = []
    for _ in range(20):
        mid = 0.5 * (lo + hi)
        _E, d = _call(fx, mid)
        c = float(d['pixel_continuity'])
        rows.append(dict(z_um=mid * 1e6, reading=c, dz_nm=(hi - lo) * 1e9))
        if c > bar:
            hi = mid
        else:
            lo = mid
    sides = {}
    for z, tag in ((lo, 'returned'), (hi, 'refused')):
        _E, d = _call(fx, z)
        sides[tag] = dict(
            z_um=z * 1e6, reading=float(d['pixel_continuity']),
            n_branch_max=d.get('n_branch_max'),
            n_triangles_degenerate=d.get('n_triangles_degenerate'),
            r_c=d.get('r_c'), kappa=d.get('kappa'),
            zeta_extrapolation=d.get('zeta_extrapolation'),
            reason=d.get('reason'), fell_back=bool(d.get('fell_back')))
        print(json.dumps(sides[tag]), flush=True)
    _write(out, dict(mode='jump', fixture='V', bar=bar,
                     dz_nm=(hi - lo) * 1e9,
                     step=sides['refused']['reading']
                     / sides['returned']['reading'],
                     sides=sides, bisection=rows))


def _write(out, rep):
    import lumenairy
    rep['lumenairy'] = lumenairy.__file__
    rep['python'] = sys.version.split()[0]
    rep['numpy'] = np.__version__
    rep['bars'] = S.bars()
    with open(out, 'w', encoding='cp1252') as f:
        json.dump(rep, f, indent=1)
    print('wrote', out)


def main():
    mode, out = sys.argv[1], sys.argv[2]
    import lumenairy
    print('lumenairy.__file__ =', lumenairy.__file__, flush=True)
    {'converged': mode_converged, 'gridladder': mode_gridladder,
     'windowladder': mode_windowladder, 'fold': mode_fold,
     'ab': mode_ab, 'jump': mode_jump}[mode](out)


if __name__ == '__main__':
    main()
