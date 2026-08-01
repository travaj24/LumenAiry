# Niche C6 fail-before proof: with REMAP_STATIONARY_PHASE_LAUNCH = False the
# element reproduces the COMMITTED (pre-C6) library BIT FOR BIT.
#
# The reference is not a re-run of the same code -- that would prove only
# determinism.  ``git show HEAD:lumenairy/elements/_lens_traced.py`` is written
# to a temporary file and imported as a SHADOW module inside the real
# ``lumenairy.elements`` package (so its relative imports resolve), and the two
# ``apply_real_lens_traced`` implementations are compared on:
#
#   (a) synthetic carrier-regime calls, all four preserve_input_phase /
#       amplitude_model combinations plus 'remap', at two ray_subsamples;
#   (b) design 121's REAL post-DOE chain, the same 7 configurations
#       probe_c5_byte_identity.py checks (two grids, two ray_subsamples, 3- and
#       5-group runs, both readout paths, final_leg='exact'), with the chain's
#       element resolved to the shadow module by monkeypatch.
#
# It also re-runs the C5 byte-identity contract with C6 present, since the two
# flags must not interact.
#
# usage:  python probe_c6_byte_identity.py
import importlib.util
import os
import subprocess
import sys
import tempfile
import warnings

import numpy as np

warnings.filterwarnings('ignore')
import _d121_common as C                                       # noqa: E402
import lumenairy as la                                         # noqa: E402
import lumenairy.elements as _EL                               # noqa: E402
import lumenairy.elements._lens_traced as LT                   # noqa: E402

LAM = C.LAM
K0 = 2 * np.pi / LAM
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def load_head_module():
    src = subprocess.check_output(
        ['git', '-C', ROOT, 'show', 'HEAD:lumenairy/elements/_lens_traced.py'])
    fd, path = tempfile.mkstemp(suffix='.py', prefix='_lens_traced_head_')
    with os.fdopen(fd, 'wb') as fh:
        fh.write(src)
    name = 'lumenairy.elements._lens_traced_head'
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    # The shadow module defines its OWN TiltedCarrier class, so its
    # ``isinstance(carrier, TiltedCarrier)`` dispatch would reject the REAL
    # one the chain constructs.  Rebind the name (module globals are resolved
    # at call time) -- the two classes are the same NamedTuple definition, so
    # this changes identity only, never behaviour.
    mod.TiltedCarrier = LT.TiltedCarrier
    return mod, path


# ---------------------------------------------------------------------------
# (a) synthetic
# ---------------------------------------------------------------------------
def _flat():
    return {'radius': np.inf, 'glass_before': 'air', 'glass_after': 'air',
            'conic': 0.0, 'radius_y': None, 'conic_y': None,
            'aspheric_coeffs': None, 'aspheric_coeffs_y': None}


def _singlet(R1=3.1e-3, R2=-3.1e-3, d=1.0e-3, ap=1.2e-3):
    gb, ga = ['air', 'N-BK7'], ['N-BK7', 'air']
    return {'name': 's', 'aperture_diameter': ap, 'thicknesses': [d],
            'surfaces': [
                {'radius': R1, 'glass_before': gb[0], 'glass_after': ga[0],
                 'conic': 0.0, 'radius_y': None, 'conic_y': None,
                 'aspheric_coeffs': None, 'aspheric_coeffs_y': None},
                {'radius': R2, 'glass_before': gb[1], 'glass_after': ga[1],
                 'conic': 0.0, 'radius_y': None, 'conic_y': None,
                 'aspheric_coeffs': None, 'aspheric_coeffs_y': None}]}


def synthetic(head):
    n, dx, w, rc, alpha = 256, 4.0e-6, 200e-6, -0.02, 6.0
    x = (np.arange(n) - n // 2) * dx
    r2 = x[None, :] ** 2 + x[:, None] ** 2
    S = np.sign(rc) * (np.sqrt(r2 + rc ** 2) - abs(rc))
    E = (np.exp(-r2 / w ** 2) * np.exp(1j * K0 * S)
         * np.exp(1j * alpha * (r2 / w ** 2) ** 2)).astype(np.complex128)
    presc = _singlet()
    base = dict(prescription=presc, wavelength=LAM, dx=dx, carrier=rc,
                parallel_amp=False, on_undersample='silent',
                on_noncollimated='silent')
    cases = []
    for pip, amod in (('remap', 'ray_density'), (True, 'ray_density'),
                      (False, 'ray_density'), (True, 'screen'),
                      (False, 'screen')):
        for rs in (1, 4):
            cases.append((f"pip={pip!r} amp={amod} rs={rs}",
                          dict(preserve_input_phase=pip,
                               amplitude_model=amod, ray_subsample=rs)))
    cases.append(("remap lattice rs=4",
                  dict(preserve_input_phase='remap',
                       amplitude_model='ray_density', remap_sampling='lattice',
                       ray_subsample=4)))
    cases.append(("no carrier, remap rs=4",
                  dict(preserve_input_phase='remap', carrier=None,
                       amplitude_model='ray_density', ray_subsample=4)))
    ok = True
    old = LT.REMAP_STATIONARY_PHASE_LAUNCH
    LT.REMAP_STATIONARY_PHASE_LAUNCH = False
    try:
        # warm-up: both implementations past the traced pipeline's ulp-scale
        # first-call boundary (W9 determinism calibration)
        for _ in range(2):
            LT.apply_real_lens_traced(E, **base, ray_subsample=4)
            head.apply_real_lens_traced(E, **base, ray_subsample=4)
        for lbl, kw in cases:
            k = dict(base)
            k.update(kw)
            a = np.asarray(LT.apply_real_lens_traced(E, **k))
            b = np.asarray(head.apply_real_lens_traced(E, **k))
            eq = bool(np.array_equal(a, b)) and a.dtype == b.dtype
            ok &= eq
            print(f"  {lbl:38s} array_equal={eq}  max|dE| "
                  f"{float(np.abs(a - b).max()):.3e}")
    finally:
        LT.REMAP_STATIONARY_PHASE_LAUNCH = old
    return ok


# ---------------------------------------------------------------------------
# (b) design 121's real post-DOE chain
# ---------------------------------------------------------------------------
def run_chain(order, rn=1024, rs=4, leg='paraxial', fr=None, ngroups=None):
    pre, post, _g, period = C.geometry()
    env, R, dx, _P = C.chain_a(n=rn, rs=rs)
    L, M = order[0] * LAM / period, order[1] * LAM / period
    kw = {}
    if fr is not None:
        kw['focus_readout'] = fr
    res = la.propagate_traced_carrier_chain(
        env, post[:ngroups], LAM, dx, r_in=la.TiltedCarrier(R, L, M),
        ray_subsample=rs, n_workers=8, final_distance=0.0, final_leg=leg,
        on_decentred_fit='ignore', on_na_proximity='ignore',
        on_tilt_exact_grid='ignore', on_rs_fine_clamp='ignore', **kw)
    return np.asarray(res.field)


CASES_121 = [
    ('RN=1024 rs=4 paraxial', dict(rn=1024, rs=4)),
    ('RN=1024 rs=2 paraxial', dict(rn=1024, rs=2)),
    ('RN=2048 rs=4 paraxial', dict(rn=2048, rs=4)),
    ('RN=1024 rs=4, 3 groups', dict(rn=1024, rs=4, ngroups=3)),
    ('RN=1024 rs=4, 5 groups', dict(rn=1024, rs=4, ngroups=5)),
    ('RN=1024 rs=4 focus_readout',
     dict(rn=1024, rs=4, fr=dict(dx_out=0.4e-6, N_out=64))),
    ('RN=1024 rs=4 final_leg=exact',
     dict(rn=1024, rs=4, leg='exact',
          fr=dict(dx_out=0.4e-6, N_out=64, n_fine_cap=8192))),
]


def chain_121(head, order=(0, 0), cases=None):
    ok = True
    old_flag = LT.REMAP_STATIONARY_PHASE_LAUNCH
    orig = _EL.apply_real_lens_traced
    LT.REMAP_STATIONARY_PHASE_LAUNCH = False
    try:
        for lbl, kw in (cases or CASES_121):
            _EL.apply_real_lens_traced = orig
            a = run_chain(order, **kw)
            _EL.apply_real_lens_traced = head.apply_real_lens_traced
            b = run_chain(order, **kw)
            _EL.apply_real_lens_traced = orig
            eq = bool(np.array_equal(a, b)) and a.dtype == b.dtype
            ok &= eq
            print(f"  {lbl:32s} shape {a.shape} {a.dtype}  array_equal={eq}  "
                  f"max|dE| {float(np.abs(a - b).max()):.3e}", flush=True)
    finally:
        _EL.apply_real_lens_traced = orig
        LT.REMAP_STATIONARY_PHASE_LAUNCH = old_flag
    return ok


def c5_still_holds():
    """The C5 contract with C6 present: the UNTILTED chain is byte-identical
    with TILTED_CARRIER_EXACT_EIKONAL on and off, at the SHIPPED C6 setting."""
    ok = True
    for lbl, kw in CASES_121[:3]:
        outs = []
        for f in (True, False):
            LT.TILTED_CARRIER_EXACT_EIKONAL = f
            try:
                outs.append(run_chain((0, 0), **kw))
            finally:
                LT.TILTED_CARRIER_EXACT_EIKONAL = True
        eq = bool(np.array_equal(outs[0], outs[1]))
        ok &= eq
        print(f"  {lbl:32s} C5 on vs off  array_equal={eq}  max|dE| "
              f"{float(np.abs(outs[0] - outs[1]).max()):.3e}", flush=True)
    return ok


def main():
    parts = os.environ.get('PARTS', 'abcd')
    head, path = load_head_module()
    print(f"shadow module: HEAD:lumenairy/elements/_lens_traced.py -> {path}")
    print(f"  HEAD has REMAP_STATIONARY_PHASE_LAUNCH: "
          f"{hasattr(head, 'REMAP_STATIONARY_PHASE_LAUNCH')} (must be False)")
    print("\n=== (a) SYNTHETIC: C6 OFF vs the committed library ===")
    ok = synthetic(head)
    print("\n=== (b) DESIGN 121 post-DOE chain, order (0,0), C6 OFF vs "
          "committed ===")
    ok &= chain_121(head, order=(0, 0))
    print("\n=== (c) DESIGN 121 post-DOE chain, order (-4,-2), C6 OFF vs "
          "committed ===")
    ok &= chain_121(head, order=(-4, -2))
    print("\n=== (d) the C5 contract still holds with C6 shipped ===")
    ok &= c5_still_holds()
    try:
        os.unlink(path)
    except OSError:
        pass
    print("\nOK" if ok else "\nFAILED")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
