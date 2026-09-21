"""WP-C3 -- ``transport='sziklas'`` IS the pre-flip arithmetic, archive to
archive.

Run as a CHILD process bound to ONE tree::

    C3_SPELLING={default|sziklas} python probe_sziklas_bitid.py <tree> <out.json>

TWO STATEMENTS, AND THEY ARE NOT THE SAME STATEMENT.

1. On the BASE tree (``git archive 49ddf4bd lumenairy``, the parent of this
   work package) the spelling is ``default`` -- no ``transport=`` argument
   appears anywhere, because at that commit the default IS ``'sziklas'`` and
   the point is to capture what a caller who passes nothing got.
2. On the BRANCH tree (the same archive with ONE file replaced, this package's
   ``carrier.py``; ``diff -rq`` over the two trees reports that file and no
   other) the spelling is ``sziklas`` -- named explicitly, because the default
   has moved.

The two digest maps are then compared key by key by ``run_c3_bitid.py``.  Any
key that differs is a leg where naming the old transport did NOT return the
old arithmetic, which is the claim "the way back is one keyword" failing.

WHY ARCHIVES AND NOT THE WORKING TREE.  The working tree cannot make this
claim about itself: its ``elements/_lens_traced.py``, its ``_lens_real.py``
and its ``backend/`` are all reachable from a carrier chain and none of them
is this package's to vouch for.  Extracting both arms and substituting exactly
one file makes the transport's own file the only variable, and the anchor in
``clib`` prints and enforces which tree each child bound.

THE FIXTURE SET is every SHIPPED carrier entry point this package can reach,
in the shapes WP-B4 sec. 4.2 and VERIFY-WP-B4 row 13 used plus the ones this
package's own change touches (the chain readout's route resolution, the multi
orchestrator's readout, and ``final_distance = 0`` with a readout, which was
the limit the Collins readout refused).  Each record folds the returned
arrays, the returned carrier and pitch, the returned object's TYPE, the whole
``stages`` list, and every warning in EMISSION order -- so a guard that moved,
was reworded or was reordered is a moved key here, not a silent pass.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_TREE = os.path.abspath(sys.argv[1])
sys.path.insert(0, _TREE)

import clib  # noqa: E402

import numpy as np  # noqa: E402

clib.anchor(_TREE)

import lumenairy.propagators.carrier as CA          # noqa: E402

_SPELLING = os.environ.get('C3_SPELLING', 'default')
if _SPELLING not in ('default', 'sziklas'):
    raise SystemExit(f"C3_SPELLING must be 'default' or 'sziklas', "
                     f"got {_SPELLING!r}")

#: What to splice into every call.  Empty on the base tree (the caller passes
#: nothing and gets the shipped default); ``transport='sziklas'`` on the
#: branch, which is the whole claim.
_TR = {} if _SPELLING == 'default' else {'transport': 'sziklas'}

WL = 633e-9
WL_CHAIN = 1.31e-6
_TKW = dict(on_undersample='silent', on_noncollimated='silent')


def _gauss(n, dx, w, dtype=np.complex128):
    ax = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(ax, ax)
    return np.exp(-(X ** 2 + Y ** 2) / (w * w)).astype(dtype)


def _singlet():
    return {'name': 'p', 'aperture_diameter': 14e-3, 'thicknesses': [3e-3],
            'surfaces': [
                {'radius': 60e-3, 'glass_before': 'air',
                 'glass_after': 'N-BK7', 'conic': 0.0, 'radius_y': None,
                 'conic_y': None, 'aspheric_coeffs': None,
                 'aspheric_coeffs_y': None},
                {'radius': -60e-3, 'glass_before': 'N-BK7',
                 'glass_after': 'air', 'conic': 0.0, 'radius_y': None,
                 'conic_y': None, 'aspheric_coeffs': None,
                 'aspheric_coeffs_y': None}]}


def _chain_fixture(n=256, dx=60e-6, w=4.5e-3):
    presc = _singlet()
    return (_gauss(n, dx, w), dx, 60e-3,
            [{'prescription': presc, 'gap_before': 20e-3},
             {'prescription': presc, 'gap_before': 10e-3}])


# ---------------------------------------------------------------------------
# 1.  the single carrier step, every branch it has
# ---------------------------------------------------------------------------
def section_single(p):
    E = _gauss(64, 8e-6, 60e-6)
    E32 = _gauss(64, 8e-6, 60e-6, dtype=np.complex64)
    cases = [
        ('short-converging', -0.05, 5e-3),
        ('long-converging', -0.02, 0.05),
        ('back-propagating', -0.05, -3e-3),
        ('collimated', np.inf, 5e-3),
        ('diverging', 0.08, 12e-3),
        ('near-focus', -0.02, 0.0199),
        ('focus-crossing', -0.02, 0.0205),
        ('zero-length', -0.05, 0.0),
    ]
    for tag, R, z in cases:
        for gk in ('auto', 'fresnel'):
            p.call(f"S-{tag}-{gk}", CA.propagate_carrier_referenced,
                   E, R, z, WL, 8e-6, gap_kernel=gk, **_TR)
    p.call("S-astigmatic", CA.propagate_carrier_referenced,
           E, (-0.05, -0.08), 5e-3, WL, 8e-6, **_TR)
    p.call("S-complex64", CA.propagate_carrier_referenced,
           E32, -0.05, 5e-3, WL, 8e-6, **_TR)
    p.call("S-tilted", CA.propagate_carrier_referenced,
           E, -0.05, 5e-3, WL, 8e-6, tilt=(0.02, -0.01), gap_kernel='exact',
           **_TR)
    p.call("S-exact-kernel", CA.propagate_carrier_referenced,
           E, -0.05, 5e-3, WL, 8e-6, gap_kernel='exact', **_TR)
    # the free-lattice keywords have no meaning on 'sziklas' and are refused;
    # the refusal MESSAGE is part of the contract, so it is a key too.
    p.call("S-dxout-refused", CA.propagate_carrier_referenced,
           E, -0.05, 5e-3, WL, 8e-6, dx_out=1e-6, **_TR)
    p.call("S-carrierout-refused", CA.propagate_carrier_referenced,
           E, -0.05, 5e-3, WL, 8e-6, carrier_out=np.inf, **_TR)


# ---------------------------------------------------------------------------
# 2.  the readouts and the envelope helpers (no ``transport`` of their own)
# ---------------------------------------------------------------------------
def section_readouts(p):
    E = _gauss(128, 4e-6, 120e-6)
    p.call("R-focus-readout", CA.carrier_referenced_focus_readout,
           E, -0.03, 0.03, WL, 4e-6, dx_out=2e-7, N_out=32,
           on_replica='ignore')
    p.call("R-focus-readout-standoff", CA.carrier_referenced_focus_readout,
           E, -0.03, 0.03, WL, 4e-6, dx_out=2e-7, N_out=32, standoff=1e-3,
           on_replica='ignore')
    p.call("R-exact-focus-readout",
           CA.carrier_referenced_exact_focus_readout,
           E, -0.03, 0.03, WL, 4e-6, dx_out=2e-7, N_out=32)
    p.call("R-reconstruct", CA.carrier_referenced_reconstruct,
           E, -0.03, WL, 4e-6)
    p.call("R-envelope", CA.carrier_referenced_envelope, E, -0.03, WL, 4e-6)
    p.call("R-fit-radius", CA.carrier_referenced_fit_radius, E, WL, 4e-6)
    p.call("R-aperture", CA.carrier_referenced_aperture,
           E, -0.03, 200e-6, WL, 4e-6)


# ---------------------------------------------------------------------------
# 3.  the chain and the multi orchestrator -- ``stages`` folded in whole
# ---------------------------------------------------------------------------
def _chain_record(res):
    """Everything a chain returns, in one digestible structure."""
    return {'field': np.asarray(res.field),
            'R': res.R, 'dx': res.dx,
            'stages_repr': repr(res.stages),
            'stages': res.stages}


def section_chain(p):
    env, dx, r_in, groups = _chain_fixture()
    base = dict(r_in=r_in, ray_subsample=16, n_workers=1,
                traced_kwargs=_TKW, final_leg='paraxial')
    fr = dict(dx_out=0.5e-6, N_out=64)

    def chain(**kw):
        return _chain_record(CA.propagate_traced_carrier_chain(
            env, groups, WL_CHAIN, dx, **dict(base, **kw), **_TR))

    p.call("C-chain-bare-final", chain, final_distance=8e-3)
    p.call("C-chain-readout", chain, final_distance=8e-3, focus_readout=fr)
    p.call("C-chain-readout-long", chain, final_distance=40e-3,
           focus_readout=fr)
    p.call("C-chain-readout-standoff", chain, final_distance=8e-3,
           focus_readout=dict(fr, standoff=2e-3))
    p.call("C-chain-readout-bandlimit", chain, final_distance=8e-3,
           focus_readout=dict(fr, bandlimit=True))
    p.call("C-chain-readout-zero-distance", chain, final_distance=0.0,
           focus_readout=fr)
    p.call("C-chain-zero-distance", chain, final_distance=0.0)
    p.call("C-chain-fresnel-kernel", chain, final_distance=8e-3,
           focus_readout=fr, gap_kernel='fresnel')
    p.call("C-chain-replica-fill-zero", chain, final_distance=8e-3,
           focus_readout=dict(fr, replica_fill='zero', on_replica='ignore'))

    def multi(**kw):
        res = CA.propagate_traced_carrier_chain_multi(
            kw.pop('cong'), groups, WL_CHAIN, dx,
            ray_subsample=16, n_workers=1, traced_kwargs=_TKW,
            final_leg='paraxial', **kw, **_TR)
        return {'field': np.asarray(res.field), 'dx': res.dx,
                'centre': res.centre}

    one = [{'field': env, 'carrier': r_in}]
    two = [{'field': env, 'carrier': r_in},
           {'field': 0.5 * env, 'carrier': r_in}]
    p.call("M-multi-K1", multi, cong=one, output_grid=fr,
           final_distance=8e-3)
    p.call("M-multi-K2", multi, cong=two, output_grid=fr,
           final_distance=8e-3)
    p.call("M-multi-K1-standoff", multi, cong=one,
           output_grid=dict(fr, standoff=2e-3), final_distance=8e-3)
    p.call("M-multi-K1-long", multi, cong=one, output_grid=fr,
           final_distance=40e-3)


def main():
    out_path = sys.argv[2]
    p = clib.Probe()
    section_single(p)
    section_readouts(p)
    section_chain(p)
    p.write(out_path)
    print(f"[probe_sziklas_bitid] spelling={_SPELLING} {len(p.out)} keys, "
          f"build={clib.build_tag()}")


if __name__ == '__main__':
    main()
