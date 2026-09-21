"""VERIFY-WP-C3 claim 4a -- is ``transport='sziklas'`` the pre-flip arithmetic,
archive to archive, on MY OWN key set?

    VC3_SPELLING={default|sziklas} python probe_wayback.py <tree> <out.json>

``default`` passes NO ``transport=`` anywhere (that is what a 49ddf4bd caller
got); ``sziklas`` names it on every entry point that takes it.  The two digest
maps are then compared key by key by ``run_wayback.py``.

Independent of ``validation/probe_c3_collins_default/`` -- different harness
(``vlib3``), different fixtures, different key names, and a larger set: every
public chain entry point, both public readouts, ``final_leg='exact'``, the
JAX arm, the stop-plane keys, both gap kernels, ``replica_fill``, ``bandlimit``
and ``final_distance = 0``.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_TREE = os.path.abspath(sys.argv[1])
sys.path.insert(0, _TREE)

import numpy as np                                            # noqa: E402

import vlib3                                                  # noqa: E402

vlib3.bind(_TREE)

import lumenairy.propagators.carrier as CA                    # noqa: E402

_SPELL = os.environ.get('VC3_SPELLING', 'default')
if _SPELL not in ('default', 'sziklas'):
    raise SystemExit("VC3_SPELLING must be default|sziklas, got %r" % _SPELL)

#: spliced into every call that TAKES ``transport``
TR = {} if _SPELL == 'default' else {'transport': 'sziklas'}

WL = 633e-9
WL_IR = 1.31e-6
TKW = dict(on_undersample='silent', on_noncollimated='silent')


# ---------------------------------------------------------------------------
# fixtures -- mine, not the author's
# ---------------------------------------------------------------------------
def gauss(ny, nx, dy, dx, wy, wx, dtype=np.complex128):
    y = (np.arange(ny) - ny / 2.0) * dy
    x = (np.arange(nx) - nx / 2.0) * dx
    g = np.exp(-((y[:, None] / wy) ** 2 + (x[None, :] / wx) ** 2))
    return g.astype(dtype)


def sq(n, dx, w, dtype=np.complex128):
    return gauss(n, n, dx, dx, w, w, dtype)


def _plano_convex():
    """A 5 mm thick plano-convex N-BK7, convex first."""
    return {'name': 'pcx', 'aperture_diameter': 16e-3, 'thicknesses': [5e-3],
            'surfaces': [
                {'radius': 52e-3, 'glass_before': 'air',
                 'glass_after': 'N-BK7', 'conic': 0.0, 'radius_y': None,
                 'conic_y': None, 'aspheric_coeffs': None,
                 'aspheric_coeffs_y': None},
                {'radius': np.inf, 'glass_before': 'N-BK7',
                 'glass_after': 'air', 'conic': 0.0, 'radius_y': None,
                 'conic_y': None, 'aspheric_coeffs': None,
                 'aspheric_coeffs_y': None}]}


def _biconvex():
    """A 4 mm thick symmetric biconvex N-BK7."""
    return {'name': 'bcx', 'aperture_diameter': 16e-3, 'thicknesses': [4e-3],
            'surfaces': [
                {'radius': 75e-3, 'glass_before': 'air',
                 'glass_after': 'N-BK7', 'conic': 0.0, 'radius_y': None,
                 'conic_y': None, 'aspheric_coeffs': None,
                 'aspheric_coeffs_y': None},
                {'radius': -75e-3, 'glass_before': 'N-BK7',
                 'glass_after': 'air', 'conic': 0.0, 'radius_y': None,
                 'conic_y': None, 'aspheric_coeffs': None,
                 'aspheric_coeffs_y': None}]}


def chain_fixture(n=256, dx=55e-6, w=4.0e-3):
    """Two groups, my own prescriptions; a converging launch at R = -70 mm."""
    return (sq(n, dx, w), dx, -70e-3,
            [{'prescription': _plano_convex(), 'gap_before': 18e-3},
             {'prescription': _biconvex(), 'gap_before': 12e-3}])


def chain_record(res):
    """Everything the chain hands back, folded whole."""
    return {'type': type(res).__name__,
            'field': np.asarray(res.field),
            'R': res.R, 'dx': res.dx,
            'n_stages': len(res.stages),
            'stages': res.stages,
            'stages_repr': repr(res.stages)}


def multi_record(res):
    return {'type': type(res).__name__,
            'field': np.asarray(res.field),
            'dx': res.dx, 'centre': res.centre,
            'obj_repr': repr(res)[:4000]}


def step_record(res):
    return {'type': type(res).__name__,
            'env': np.asarray(res.env), 'R': res.R, 'dx': res.dx}


# ---------------------------------------------------------------------------
# 1. propagate_carrier_referenced -- the single step, every branch
# ---------------------------------------------------------------------------
GEOMS = [
    ('convshort', -0.05, 5e-3),
    ('convlong', -0.02, 0.05),
    ('back', -0.05, -3e-3),
    ('collimated', np.inf, 5e-3),
    ('backcollimated', np.inf, -4e-3),
    ('diverging', 0.08, 12e-3),
    ('nearfocus', -0.02, 0.0199),
    ('focuscross', -0.02, 0.0205),
    ('farpastfocus', -0.02, 0.04),
    ('zerolen', -0.05, 0.0),
]


def sec_single(K):
    E = sq(64, 8e-6, 60e-6)

    def step(*a, **kw):
        return step_record(CA.propagate_carrier_referenced(*a, **kw))

    for tag, R, z in GEOMS:
        for gk in ('auto', 'fresnel'):
            K.call("S1-%s-%s" % (tag, gk), step,
                   E, R, z, WL, 8e-6, gap_kernel=gk, **TR)
    # both gap kernels above; the exact kernel on three geometries
    for tag, R, z in (('convshort', -0.05, 5e-3),
                      ('collimated', np.inf, 5e-3),
                      ('focuscross', -0.02, 0.0205)):
        K.call("S2-exactkernel-%s" % tag, step,
               E, R, z, WL, 8e-6, gap_kernel='exact', **TR)
    # astigmatic, three ways
    K.call("S3-astig-auto", step, E, (-0.05, -0.08), 5e-3, WL, 8e-6, **TR)
    K.call("S3-astig-fresnel", step, E, (-0.05, -0.08), 5e-3, WL, 8e-6,
           gap_kernel='fresnel', **TR)
    K.call("S3-astig-focuscross", step, E, (-0.02, -0.03), 0.0205, WL, 8e-6,
           **TR)
    K.call("S3-astig-onefinite", step, E, (-0.05, np.inf), 5e-3, WL, 8e-6,
           **TR)
    # dtype
    K.call("S4-complex64-auto", step, sq(64, 8e-6, 60e-6, np.complex64),
           -0.05, 5e-3, WL, 8e-6, **TR)
    K.call("S4-complex64-fresnel", step, sq(64, 8e-6, 60e-6, np.complex64),
           -0.05, 5e-3, WL, 8e-6, gap_kernel='fresnel', **TR)
    K.call("S4-complex64-collimated", step,
           sq(64, 8e-6, 60e-6, np.complex64), np.inf, 5e-3, WL, 8e-6, **TR)
    K.call("S4-float64-input", step, sq(64, 8e-6, 60e-6).real.copy(),
           -0.05, 5e-3, WL, 8e-6, **TR)
    # tilt
    K.call("S5-tilt-exact", step, E, -0.05, 5e-3, WL, 8e-6,
           tilt=(0.02, -0.01), gap_kernel='exact', **TR)
    K.call("S5-tilt-auto", step, E, -0.05, 5e-3, WL, 8e-6,
           tilt=(0.02, -0.01), **TR)
    K.call("S5-tilt-collimated", step, E, np.inf, 5e-3, WL, 8e-6,
           tilt=(0.01, 0.0), gap_kernel='exact', **TR)
    # rectangular grid and anisotropic pitch
    K.call("S6-rect-grid", step, gauss(48, 80, 8e-6, 8e-6, 60e-6, 90e-6),
           -0.05, 5e-3, WL, 8e-6, **TR)
    K.call("S6-aniso-pitch", step, gauss(64, 64, 6e-6, 8e-6, 50e-6, 60e-6),
           -0.05, 5e-3, WL, 8e-6, dy=6e-6, **TR)
    # the free-lattice keywords have no meaning on 'sziklas'; the refusal
    # MESSAGE is part of the contract, so it is a key.
    K.call("S7-dxout-refused", step, E, -0.05, 5e-3, WL, 8e-6,
           dx_out=1e-6, **TR)
    K.call("S7-carrierout-refused", step, E, -0.05, 5e-3, WL, 8e-6,
           carrier_out=np.inf, **TR)
    K.call("S7-oncollins-error", step, E, -0.05, 5e-3, WL, 8e-6,
           on_collins_sampling='error', **TR)


# ---------------------------------------------------------------------------
# 2. the JAX arm -- the backend is the ARRAY TYPE, so the same entry point
# ---------------------------------------------------------------------------
def sec_jax(K):
    try:
        import jax
        jax.config.update('jax_enable_x64', True)
        import jax.numpy as jnp
    except Exception as exc:                                   # noqa: BLE001
        K.call("J0-jax-unavailable", lambda: (_ for _ in ()).throw(
            RuntimeError("jax unavailable: %s" % type(exc).__name__)))
        return

    def jstep(env, *a, **kw):
        res = CA.propagate_carrier_referenced(jnp.asarray(env), *a, **kw)
        return {'type': type(res).__name__,
                'env_backend': type(res.env).__module__.split('.')[0],
                'env': np.asarray(res.env), 'R': res.R, 'dx': res.dx}

    E = sq(64, 8e-6, 60e-6)
    K.call("J1-jax-convshort", jstep, E, -0.05, 5e-3, WL, 8e-6, **TR)
    K.call("J1-jax-convshort-fresnel", jstep, E, -0.05, 5e-3, WL, 8e-6,
           gap_kernel='fresnel', **TR)
    K.call("J1-jax-collimated", jstep, E, np.inf, 5e-3, WL, 8e-6, **TR)
    K.call("J1-jax-focuscross", jstep, E, -0.02, 0.0205, WL, 8e-6, **TR)
    K.call("J1-jax-astig", jstep, E, (-0.05, -0.08), 5e-3, WL, 8e-6, **TR)
    K.call("J1-jax-zerolen", jstep, E, -0.05, 0.0, WL, 8e-6, **TR)
    K.call("J1-jax-tilt-exact", jstep, E, -0.05, 5e-3, WL, 8e-6,
           tilt=(0.02, -0.01), gap_kernel='exact', **TR)


# ---------------------------------------------------------------------------
# 3. the two PUBLIC readouts.  ``carrier_referenced_focus_readout`` GAINED a
#    ``transport`` of its own in 5.49.0 (WP-C3 round 2, VERIFY-WP-C3 D8), so
#    the sziklas spelling names it here too -- the way-back rule is one
#    keyword on every entry point that takes one.  The exact readout still
#    takes none.
# ---------------------------------------------------------------------------
def sec_readouts(K):
    E = sq(128, 4e-6, 120e-6)
    fr = CA.carrier_referenced_focus_readout
    ex = CA.carrier_referenced_exact_focus_readout
    import inspect as _i
    _TR_RO = (TR if 'transport' in
              _i.signature(fr).parameters else {})

    def ro(*a, **kw):
        return {'field': np.asarray(fr(*a, **dict(_TR_RO, **kw)))}

    def exro(*a, **kw):
        return {'field': np.asarray(ex(*a, **kw))}

    base = dict(dx_out=2e-7, N_out=32, on_replica='ignore')
    K.call("P1-focus-readout", ro, E, -0.03, 0.03, WL, 4e-6, **base)
    K.call("P1-focus-readout-standoff-1mm", ro, E, -0.03, 0.03, WL, 4e-6,
           standoff=1e-3, **base)
    K.call("P1-focus-readout-standoff-3mm", ro, E, -0.03, 0.03, WL, 4e-6,
           standoff=3e-3, **base)
    K.call("P1-focus-readout-containment-warn", ro, E, -0.03, 0.03, WL, 4e-6,
           standoff=1e-3, on_focus_containment='warn', **base)
    K.call("P1-focus-readout-containment-ignore", ro, E, -0.03, 0.03, WL,
           4e-6, standoff=1e-3, on_focus_containment='ignore', **base)
    K.call("P1-focus-readout-bandlimit-off", ro, E, -0.03, 0.03, WL, 4e-6,
           bandlimit=False, **base)
    K.call("P1-focus-readout-fresnel", ro, E, -0.03, 0.03, WL, 4e-6,
           gap_kernel='fresnel', **base)
    K.call("P1-focus-readout-exactkernel", ro, E, -0.03, 0.03, WL, 4e-6,
           gap_kernel='exact', **base)
    K.call("P1-focus-readout-tilt", ro, E, -0.03, 0.03, WL, 4e-6,
           tilt=(5e-3, -2e-3), gap_kernel='exact', **base)
    K.call("P1-focus-readout-replica-zero", ro, E, -0.03, 0.03, WL, 4e-6,
           dx_out=2e-7, N_out=32, on_replica='ignore', replica_fill='zero')
    K.call("P1-focus-readout-replica-repeat", ro, E, -0.03, 0.03, WL, 4e-6,
           dx_out=2e-7, N_out=32, on_replica='ignore',
           replica_fill='repeat')
    K.call("P1-focus-readout-centre", ro, E, -0.03, 0.03, WL, 4e-6,
           centre_out=(1e-6, -2e-6), **base)
    K.call("P1-focus-readout-collimated", ro, sq(128, 4e-6, 120e-6),
           np.inf, 0.03, WL, 4e-6, **base)
    K.call("P2-exact-focus-readout", exro, E, -0.03, 0.03, WL, 4e-6,
           dx_out=2e-7, N_out=32, on_replica='ignore')
    K.call("P2-exact-focus-readout-bandlimit-off", exro, E, -0.03, 0.03, WL,
           4e-6, dx_out=2e-7, N_out=32, bandlimit=False, on_replica='ignore')
    K.call("P2-exact-focus-readout-window", exro, E, -0.03, 0.03, WL, 4e-6,
           dx_out=2e-7, N_out=32, window_factor=5.0, on_replica='ignore')
    K.call("P2-exact-focus-readout-replica-zero", exro, E, -0.03, 0.03, WL,
           4e-6, dx_out=2e-7, N_out=32, on_replica='ignore',
           replica_fill='zero')
    K.call("P2-exact-focus-readout-tilt", exro, E, -0.03, 0.03, WL, 4e-6,
           dx_out=2e-7, N_out=32, tilt=(5e-3, -2e-3), on_replica='ignore')
    # the envelope helpers, for completeness of the module's public surface
    K.call("P3-reconstruct", lambda *a: np.asarray(
        CA.carrier_referenced_reconstruct(*a)), E, -0.03, WL, 4e-6)
    K.call("P3-envelope", lambda *a: np.asarray(
        CA.carrier_referenced_envelope(*a)), E, -0.03, WL, 4e-6)
    K.call("P3-fit-radius", CA.carrier_referenced_fit_radius, E, WL, 4e-6)
    K.call("P3-aperture", CA.carrier_referenced_aperture,
           E, -0.03, WL, 4e-6, radius=200e-6)
    K.call("P3-aperture-transmission", CA.carrier_referenced_aperture,
           E, -0.03, WL, 4e-6, radius=200e-6, return_transmission=True)


# ---------------------------------------------------------------------------
# 4. the chain and the multi orchestrator
# ---------------------------------------------------------------------------
def sec_chain(K):
    env, dx, r_in, groups = chain_fixture()
    base = dict(r_in=r_in, ray_subsample=16, n_workers=1, traced_kwargs=TKW,
                final_leg='paraxial')
    fr = dict(dx_out=0.4e-6, N_out=64)

    def chain(**kw):
        return chain_record(CA.propagate_traced_carrier_chain(
            env, groups, WL_IR, dx, **dict(base, **kw), **TR))

    K.call("C1-bare-final", chain, final_distance=9e-3)
    K.call("C1-bare-final-long", chain, final_distance=45e-3)
    K.call("C1-bare-zero-distance", chain, final_distance=0.0)
    K.call("C2-readout", chain, final_distance=9e-3, focus_readout=fr)
    K.call("C2-readout-long", chain, final_distance=45e-3, focus_readout=fr)
    K.call("C2-readout-zero-distance", chain, final_distance=0.0,
           focus_readout=fr)
    K.call("C2-readout-standoff-2mm", chain, final_distance=9e-3,
           focus_readout=dict(fr, standoff=2e-3))
    K.call("C2-readout-standoff-5mm", chain, final_distance=45e-3,
           focus_readout=dict(fr, standoff=5e-3))
    K.call("C2-readout-containment-warn", chain, final_distance=9e-3,
           focus_readout=dict(fr, on_focus_containment='warn'))
    K.call("C2-readout-containment-ignore", chain, final_distance=9e-3,
           focus_readout=dict(fr, on_focus_containment='ignore'))
    K.call("C2-readout-bandlimit-on", chain, final_distance=9e-3,
           focus_readout=dict(fr, bandlimit=True))
    K.call("C2-readout-bandlimit-off", chain, final_distance=9e-3,
           focus_readout=dict(fr, bandlimit=False))
    K.call("C2-readout-replica-zero", chain, final_distance=9e-3,
           focus_readout=dict(fr, replica_fill='zero', on_replica='ignore'))
    K.call("C2-readout-replica-repeat", chain, final_distance=9e-3,
           focus_readout=dict(fr, replica_fill='repeat',
                              on_replica='ignore'))
    K.call("C2-readout-centre", chain, final_distance=9e-3,
           focus_readout=dict(fr, centre_out=(1e-6, -1e-6)))
    K.call("C3-fresnel-kernel", chain, final_distance=9e-3,
           focus_readout=fr, gap_kernel='fresnel')
    K.call("C3-exact-kernel", chain, final_distance=9e-3,
           focus_readout=fr, gap_kernel='exact')
    K.call("C3-fresnel-bare", chain, final_distance=9e-3,
           gap_kernel='fresnel')
    K.call("C4-collimated-input", lambda **kw: chain_record(
        CA.propagate_traced_carrier_chain(
            sq(256, 55e-6, 4.0e-3), groups, WL_IR, 55e-6,
            r_in=np.inf, ray_subsample=16, n_workers=1, traced_kwargs=TKW,
            final_leg='paraxial', **kw, **TR)),
        final_distance=9e-3, focus_readout=fr)
    K.call("C4-single-group", lambda **kw: chain_record(
        CA.propagate_traced_carrier_chain(
            env, groups[:1], WL_IR, dx, **dict(base, **kw), **TR)),
        final_distance=9e-3, focus_readout=fr)
    K.call("C5-final-leg-exact-bare", lambda **kw: chain_record(
        CA.propagate_traced_carrier_chain(
            env, groups, WL_IR, dx,
            **dict(base, final_leg='exact'), **kw, **TR)),
        final_distance=9e-3)
    K.call("C5-final-leg-exact-readout", lambda **kw: chain_record(
        CA.propagate_traced_carrier_chain(
            env, groups, WL_IR, dx,
            **dict(base, final_leg='exact'), **kw, **TR)),
        final_distance=9e-3, focus_readout=fr)
    K.call("C5-final-leg-auto-readout", lambda **kw: chain_record(
        CA.propagate_traced_carrier_chain(
            env, groups, WL_IR, dx,
            **dict(base, final_leg='auto'), **kw, **TR)),
        final_distance=9e-3, focus_readout=fr)


def sec_multi(K):
    env, dx, r_in, groups = chain_fixture()
    fr = dict(dx_out=0.4e-6, N_out=64)

    def multi(cong, **kw):
        kw.setdefault('final_leg', 'paraxial')
        return multi_record(CA.propagate_traced_carrier_chain_multi(
            cong, groups, WL_IR, dx, ray_subsample=16, n_workers=1,
            traced_kwargs=TKW, **kw, **TR))

    one = [{'field': env, 'carrier': r_in}]
    two = [{'field': env, 'carrier': r_in},
           {'field': (0.5 + 0.25j) * env, 'carrier': r_in}]
    K.call("M1-K1", multi, one, output_grid=fr, final_distance=9e-3)
    K.call("M1-K2", multi, two, output_grid=fr, final_distance=9e-3)
    K.call("M1-K1-long", multi, one, output_grid=fr, final_distance=45e-3)
    K.call("M1-K2-long", multi, two, output_grid=fr, final_distance=45e-3)
    K.call("M2-K1-standoff", multi, one,
           output_grid=dict(fr, standoff=2e-3), final_distance=9e-3)
    K.call("M2-K2-standoff", multi, two,
           output_grid=dict(fr, standoff=2e-3), final_distance=9e-3)
    K.call("M2-K1-zero-distance", multi, one, output_grid=fr,
           final_distance=0.0)
    K.call("M3-K2-incoherent", multi, two, output_grid=fr,
           final_distance=9e-3, recombine='incoherent')
    K.call("M3-K1-fresnel", multi, one, output_grid=fr, final_distance=9e-3,
           gap_kernel='fresnel')
    K.call("M3-K1-tile2", multi, one, output_grid=fr, final_distance=9e-3,
           readout_tile=2)
    K.call("M4-K1-final-leg-exact", multi, one, output_grid=fr,
           final_distance=9e-3, final_leg='exact')


def main():
    out = sys.argv[2]
    K = vlib3.Keys()
    sec_single(K)
    sec_jax(K)
    sec_readouts(K)
    sec_chain(K)
    sec_multi(K)
    K.write(out, {"tree": _TREE, "spelling": _SPELL,
                  "build": vlib3.build_tag(),
                  "python": sys.version.split()[0]})
    sys.stdout.write("[probe_wayback] spelling=%s keys=%d build=%s\n"
                     % (_SPELL, len(K.digests), vlib3.build_tag()))


if __name__ == '__main__':
    main()
