"""VERIFY-WAVE5-HYGIENE2 round 2 -- INDEPENDENT byte-identity probe.

Builds its own key set over the three exact-dispersion call sites, both
Collins transports, the public entry, the two focus readouts and a
near-focus ladder, and digests EVERY reading (arrays by raw bytes,
refusals by type + message) to sha256.  Run once per tree per build; the
comparison is a separate module, so this file never sees two trees at once.

Usage (from the worktree root, BLAS pinned on the command line):

    PYTHONPATH=<tree> python validation/probe_verify_hyg2_round2/vh3_bitid.py OUT.json

The tree is pinned through PYTHONPATH and the resolved ``lumenairy.__file__``
is written into the JSON, so a mis-pinned run is visible in the output and
not only in the terminal.
"""
import hashlib
import json
import os
import sys

import numpy as np


def _h(*parts):
    m = hashlib.sha256()
    for p in parts:
        if isinstance(p, bytes):
            m.update(p)
        else:
            m.update(str(p).encode('utf-8'))
        m.update(b'\x00')
    return m.hexdigest()


def dig_arr(a):
    """Digest an array by its RAW BYTES, its dtype and its shape."""
    a = np.asarray(a)
    return _h('arr', a.dtype.str, a.shape, np.ascontiguousarray(a).tobytes())


def dig_exc(e):
    return _h('exc', type(e).__name__, str(e))


def dig_field(res):
    """A CarrierReferencedField: envelope bytes + R + pitch, exactly."""
    R = res.R
    dxo = res.dx
    return _h('crf', dig_arr(res.env),
              repr(R) if not isinstance(R, tuple) else repr(tuple(map(float, R))),
              repr(dxo) if not isinstance(dxo, tuple)
              else repr(tuple(map(float, dxo))))


def dig_obj(o):
    """Digest anything the probe reads: array, field, tuple, dict, scalar."""
    import numbers
    if isinstance(o, BaseException):
        return dig_exc(o)
    if hasattr(o, 'env') and hasattr(o, 'R') and hasattr(o, 'dx'):
        return dig_field(o)
    if isinstance(o, dict):
        return _h('dict', *[f"{k}={dig_obj(v)}" for k, v in sorted(o.items())])
    if isinstance(o, (tuple, list)):
        return _h('seq', *[dig_obj(v) for v in o])
    if isinstance(o, numbers.Number) or isinstance(o, (str, bool, type(None))):
        return _h('sca', repr(o))
    return dig_arr(o)


# --------------------------------------------------------------------------
# fixtures -- every grid deterministic, no RNG state crossing a key
# --------------------------------------------------------------------------
WL = 1.55e-6


def gauss(ny, nx, dx, w, dy=None, cdt=np.complex128, curv=0.0):
    dy = dx if dy is None else dy
    y = (np.arange(ny) - ny / 2.0) * dy
    x = (np.arange(nx) - nx / 2.0) * dx
    X, Y = np.meshgrid(x, y)
    r2 = X * X + Y * Y
    a = np.exp(-r2 / w ** 2)
    if curv:
        a = a * np.exp(1j * (2 * np.pi / WL) * r2 / (2.0 * curv))
    return np.asarray(a, dtype=cdt)


def speckle(ny, nx, cdt=np.complex128):
    """A deterministic non-Gaussian envelope: a fixed low-order phase screen
    times a super-Gaussian, so the digest is sensitive to every grid point and
    not only to the smooth core."""
    y = (np.arange(ny) - ny / 2.0) / max(ny, 1)
    x = (np.arange(nx) - nx / 2.0) / max(nx, 1)
    X, Y = np.meshgrid(x, y)
    amp = np.exp(-(X ** 2 + Y ** 2) ** 2 * 40.0)
    ph = (3.1 * np.sin(7.0 * X) * np.cos(5.0 * Y)
          + 1.7 * np.sin(11.0 * X + 2.0 * Y))
    return np.asarray(amp * np.exp(1j * ph), dtype=cdt)


def main(out_path):
    import lumenairy
    from lumenairy.propagators import carrier as CA

    keys = {}
    kinds = {}

    def rec(name, fn):
        try:
            keys[name] = dig_obj(fn())
            kinds[name] = 'value'
        except BaseException as e:            # a refusal IS a reading
            keys[name] = dig_exc(e)
            kinds[name] = 'exc:' + type(e).__name__

    # ===== A.  _exact_envelope_tf_step ====================================
    shapes = [(32, 32), (64, 64), (64, 32), (48, 96), (128, 128)]
    zs = [1e-4, 1e-3, 7.3e-3, 5e-2, -1e-3]
    tilts = [(0.0, 0.0), (0.03, 0.0), (0.0, -0.047), (0.11, 0.07)]
    for (ny, nx) in shapes:
        E = gauss(ny, nx, 2e-6, 12e-6)
        for z in zs:
            for t in tilts:
                rec(f"A.envstep.{ny}x{nx}.z{z:g}.t{t[0]:g}_{t[1]:g}",
                    lambda E=E, z=z, t=t: CA._exact_envelope_tf_step(
                        E, z, WL, 2e-6, 2e-6, tilt=t))
    # anisotropic pitch
    for (dxv, dyv) in [(2e-6, 3.5e-6), (4e-6, 1e-6)]:
        E = gauss(64, 64, dxv, 12e-6, dy=dyv)
        for t in tilts:
            rec(f"A.aniso.dx{dxv:g}.dy{dyv:g}.t{t[0]:g}_{t[1]:g}",
                lambda E=E, dxv=dxv, dyv=dyv, t=t:
                CA._exact_envelope_tf_step(E, 2e-3, WL, dxv, dyv, tilt=t))
    # complex64 in -> complex64 out
    for t in tilts:
        E64 = gauss(64, 64, 2e-6, 12e-6, cdt=np.complex64)
        rec(f"A.c64.t{t[0]:g}_{t[1]:g}",
            lambda E64=E64, t=t: CA._exact_envelope_tf_step(
                E64, 2e-3, WL, 2e-6, 2e-6, tilt=t))
    # a REAL (non-complex) envelope takes the other return branch
    rec("A.realin", lambda: CA._exact_envelope_tf_step(
        np.real(gauss(64, 64, 2e-6, 12e-6)), 2e-3, WL, 2e-6, 2e-6))
    # speckle: every grid point matters
    for (ny, nx) in [(64, 64), (96, 48)]:
        S = speckle(ny, nx)
        rec(f"A.speckle.{ny}x{nx}", lambda S=S: CA._exact_envelope_tf_step(
            S, 3e-3, WL, 2e-6, 2e-6, tilt=(0.02, -0.01)))
    # refusals (evanescent carrier direction)
    for t in [(1.0, 0.0), (0.8, 0.8), (0.0, -1.5), (0.7071067811865476,
                                                    0.7071067811865476)]:
        rec(f"A.refuse.t{t[0]:g}_{t[1]:g}",
            lambda t=t: CA._exact_envelope_tf_step(
                gauss(32, 32, 2e-6, 12e-6), 1e-3, WL, 2e-6, 2e-6, tilt=t))

    # ===== B.  _exact_tf_2d_xp, NumPy and JAX =============================
    for (ny, nx) in [(32, 32), (64, 64), (64, 32)]:
        for z in [1e-3, 1e-2, -5e-3]:
            for t in [(0.0, 0.0), (0.05, -0.02)]:
                E = gauss(ny, nx, 2e-6, 12e-6)
                rec(f"B.tfxp.np.{ny}x{nx}.z{z:g}.t{t[0]:g}_{t[1]:g}",
                    lambda E=E, z=z, t=t: CA._exact_tf_2d_xp(
                        E, z, WL, 2e-6, 2e-6, t, np, False, np))
                E64 = E.astype(np.complex64)
                rec(f"B.tfxp.np64.{ny}x{nx}.z{z:g}.t{t[0]:g}_{t[1]:g}",
                    lambda E64=E64, z=z, t=t: CA._exact_tf_2d_xp(
                        E64, z, WL, 2e-6, 2e-6, t, np, False, np))
    rec("B.tfxp.refuse", lambda: CA._exact_tf_2d_xp(
        gauss(32, 32, 2e-6, 12e-6), 1e-3, WL, 2e-6, 2e-6, (1.2, 0.0),
        np, False, np))
    try:
        import jax
        import jax.numpy as jnp
        jax.config.update('jax_enable_x64', True)
        for (ny, nx) in [(32, 32), (64, 32)]:
            for z in [1e-3, -5e-3]:
                for t in [(0.0, 0.0), (0.05, -0.02)]:
                    E = jnp.asarray(gauss(ny, nx, 2e-6, 12e-6))
                    rec(f"B.tfxp.jax.{ny}x{nx}.z{z:g}.t{t[0]:g}_{t[1]:g}",
                        lambda E=E, z=z, t=t: np.asarray(CA._exact_tf_2d_xp(
                            E, z, WL, 2e-6, 2e-6, t, jnp, True, np)))
                    E64 = jnp.asarray(gauss(ny, nx, 2e-6, 12e-6,
                                            cdt=np.complex64))
                    rec(f"B.tfxp.jax64.{ny}x{nx}.z{z:g}.t{t[0]:g}_{t[1]:g}",
                        lambda E64=E64, z=z, t=t: np.asarray(
                            CA._exact_tf_2d_xp(E64, z, WL, 2e-6, 2e-6, t,
                                               jnp, True, np)))
    except ImportError as e:
        keys['B.tfxp.jax.UNAVAILABLE'] = _h('nojax', str(e)[:40])

    # ===== C.  _collins_exact_kernel_correction ===========================
    for (ny, nx) in [(32, 32), (64, 64), (64, 32)]:
        E = gauss(ny, nx, 2e-6, 12e-6)
        S = np.fft.fft2(np.ascontiguousarray(E, dtype=np.complex128))
        for zeff in [1e-3, 1e-2, -2e-3]:
            for t in [(0.0, 0.0), (0.04, 0.0), (0.03, -0.05)]:
                rec(f"C.corr.{ny}x{nx}.ze{zeff:g}.t{t[0]:g}_{t[1]:g}",
                    lambda S=S, zeff=zeff, t=t:
                    CA._collins_exact_kernel_correction(
                        S, zeff, WL, 2e-6, 2e-6, t))
    # anisotropic pitch + refusal
    Sa = np.fft.fft2(gauss(64, 64, 2e-6, 12e-6, dy=3.5e-6))
    rec("C.corr.aniso", lambda: CA._collins_exact_kernel_correction(
        Sa, 4e-3, WL, 2e-6, 3.5e-6, (0.02, 0.01)))
    rec("C.corr.refuse", lambda: CA._collins_exact_kernel_correction(
        Sa, 4e-3, WL, 2e-6, 3.5e-6, (0.9, 0.5)))

    # ===== D.  _collins_transport (private), both kernels =================
    N = 96
    DX = 4e-6
    Ein = gauss(N, N, DX, 40e-6)
    for R_in in [0.05, -0.05, float('inf')]:
        for z in [1e-3, 2e-2]:
            for gk in ['auto', 'fresnel', 'exact']:
                st = {}
                rec(f"D.priv.R{R_in:g}.z{z:g}.{gk}",
                    lambda Ein=Ein, R_in=R_in, z=z, gk=gk, st=st: (
                        CA._collins_transport(
                            Ein, R_in, z, WL, DX, DX, dx_out=DX, dy_out=DX,
                            N_out_x=N, N_out_y=N, R_ref=float('inf'),
                            gap_kernel=gk, on_collins_sampling='ignore',
                            stats_out=st), sorted(st.items(),
                                                  key=lambda kv: kv[0])))
    # a tilted exact leg and an astigmatic one
    st = {}
    rec("D.priv.tilt", lambda st=st: CA._collins_transport(
        Ein, 0.05, 5e-3, WL, DX, DX, dx_out=DX, dy_out=DX, N_out_x=N,
        N_out_y=N, R_ref=float('inf'), gap_kernel='exact',
        tilt=(0.03, -0.02), on_collins_sampling='ignore', stats_out=st))
    rec("D.priv.astig", lambda: CA._collins_transport(
        Ein, (0.05, 0.08), 5e-3, WL, DX, DX, dx_out=DX, dy_out=DX,
        N_out_x=N, N_out_y=N, R_ref=float('inf'), gap_kernel='auto',
        on_collins_sampling='ignore'))
    rec("D.priv.astig_exact_refused", lambda: CA._collins_transport(
        Ein, (0.05, 0.08), 5e-3, WL, DX, DX, dx_out=DX, dy_out=DX,
        N_out_x=N, N_out_y=N, R_ref=float('inf'), gap_kernel='exact',
        on_collins_sampling='ignore'))
    # free output lattice + an output reference
    for dxo in [DX, 0.5 * DX, 2.0 * DX]:
        for Rref in [float('inf'), 0.2]:
            rec(f"D.priv.lat{dxo:g}.Rref{Rref:g}",
                lambda dxo=dxo, Rref=Rref: CA._collins_transport(
                    Ein, 0.05, 5e-3, WL, DX, DX, dx_out=dxo, dy_out=dxo,
                    N_out_x=N, N_out_y=N, R_ref=Rref,
                    gap_kernel='auto', on_collins_sampling='ignore'))

    # ===== E.  the PUBLIC entry, both transports ==========================
    Epub = gauss(128, 128, 3e-6, 48e-6)
    for tr in ['sziklas', 'collins']:
        for R in [0.04, -0.04, float('inf')]:
            for z in [5e-4, 5e-3, 3e-2]:
                for gk in ['auto', 'fresnel']:
                    kw = {} if tr == 'sziklas' else {
                        'dx_out': 3e-6, 'on_collins_sampling': 'ignore'}
                    rec(f"E.pub.{tr}.R{R:g}.z{z:g}.{gk}",
                        lambda Epub=Epub, tr=tr, R=R, z=z, gk=gk, kw=kw:
                        CA.propagate_carrier_referenced(
                            Epub, R, z, WL, 3e-6, gap_kernel=gk,
                            transport=tr, **kw))
    # explicit 'exact' on both transports
    for tr in ['sziklas', 'collins']:
        kw = {} if tr == 'sziklas' else {'dx_out': 3e-6,
                                         'on_collins_sampling': 'ignore'}
        rec(f"E.pub.{tr}.exact",
            lambda tr=tr, kw=kw: CA.propagate_carrier_referenced(
                Epub, 0.04, 5e-3, WL, 3e-6, gap_kernel='exact',
                transport=tr, **kw))
    # tilted public legs
    for tr in ['sziklas', 'collins']:
        for t in [(0.02, 0.0), (0.03, -0.04)]:
            kw = {} if tr == 'sziklas' else {'dx_out': 3e-6,
                                             'on_collins_sampling': 'ignore'}
            rec(f"E.pub.{tr}.tilt{t[0]:g}_{t[1]:g}",
                lambda tr=tr, t=t, kw=kw: CA.propagate_carrier_referenced(
                    Epub, 0.04, 5e-3, WL, 3e-6, gap_kernel='exact',
                    tilt=t, transport=tr, **kw))
    # complex64 public legs
    E64 = gauss(64, 64, 3e-6, 24e-6, cdt=np.complex64)
    for tr in ['sziklas', 'collins']:
        kw = {} if tr == 'sziklas' else {'dx_out': 3e-6,
                                         'on_collins_sampling': 'ignore'}
        rec(f"E.pub.{tr}.c64",
            lambda tr=tr, kw=kw: CA.propagate_carrier_referenced(
                E64, 0.04, 5e-3, WL, 3e-6, gap_kernel='exact',
                transport=tr, **kw))

    # ===== F.  NEAR-FOCUS rungs (the bridge) ==============================
    # R < 0 converging: the geometric focus is at z = |R|.  Walk across it.
    Rc = -0.02
    for frac in [0.5, 0.9, 0.98, 0.995, 1.0, 1.005, 1.02, 1.1, 1.5]:
        z = abs(Rc) * frac
        for gk in ['auto', 'fresnel']:
            rec(f"F.nf.sziklas.f{frac:g}.{gk}",
                lambda z=z, gk=gk: CA.propagate_carrier_referenced(
                    gauss(128, 128, 3e-6, 48e-6), Rc, z, WL, 3e-6,
                    gap_kernel=gk))
    for frac in [0.9, 0.995, 1.0, 1.02]:
        z = abs(Rc) * frac
        rec(f"F.nf.collins.f{frac:g}",
            lambda z=z: CA.propagate_carrier_referenced(
                gauss(128, 128, 3e-6, 48e-6), Rc, z, WL, 3e-6,
                transport='collins', dx_out=3e-8,
                on_collins_sampling='ignore'))

    # ===== G.  the two focus readouts =====================================
    Erd = gauss(128, 128, 3e-6, 48e-6)
    for f in [0.02, 0.05]:
        for gk in ['auto', 'fresnel', 'exact']:
            rec(f"G.readout.f{f:g}.{gk}",
                lambda f=f, gk=gk: CA.carrier_referenced_focus_readout(
                    Erd, -f, f, WL, 3e-6, dx_out=2e-7, N_out=64,
                    gap_kernel=gk, on_replica='ignore',
                    on_focus_containment='ignore'))
        for t in [(0.0, 0.0), (0.01, -0.005)]:
            rec(f"G.readout.f{f:g}.tilt{t[0]:g}_{t[1]:g}",
                lambda f=f, t=t: CA.carrier_referenced_focus_readout(
                    Erd, -f, f, WL, 3e-6, dx_out=2e-7, N_out=64, tilt=t,
                    on_replica='ignore', on_focus_containment='ignore'))
        Efull = CA.carrier_referenced_reconstruct(Erd, -f, WL, 3e-6)
        rec(f"G.exactreadout.f{f:g}",
            lambda f=f, Efull=Efull:
            CA.carrier_referenced_exact_focus_readout(
                Efull, -f, f, WL, 3e-6, dx_out=2e-7, N_out=64,
                on_replica='ignore', on_readout_window='ignore',
                on_n_fine_cap='ignore', on_ram_cap='ignore'))

    # ===== H.  the chirp-Z primitive itself (the V-D5 file) ===============
    from lumenairy.propagators import _bluestein as BL
    for (nin, nout) in [(24, 12), (32, 32), (48, 24)]:
        E = speckle(nin, nin)
        for alpha in [1e-3, 1.0 / nin, 0.37]:
            for meth in ['auto', 'bluestein', 'separable', 'direct']:
                rec(f"H.bl.{nin}x{nout}.a{alpha:g}.{meth}",
                    lambda E=E, alpha=alpha, nout=nout, meth=meth:
                    BL._bluestein_centred_2d(
                        E, alpha, alpha, nout, nout, sign=-1, xp=np,
                        fft2=np.fft.fft2, ifft2=np.fft.ifft2, method=meth))

    out = {
        'lumenairy_file': lumenairy.__file__,
        'lumenairy_version': getattr(lumenairy, '__version__', '?'),
        'python': sys.version.split()[0],
        'numpy': np.__version__,
        'platform': sys.platform,
        'mem_budget_mb': os.environ.get('LUMENAIRY_MEM_BUDGET_MB'),
        'blas_env': {k: os.environ.get(k) for k in
                     ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                      'MKL_NUM_THREADS')},
        'n_keys': len(keys),
        'n_value': sum(1 for v in kinds.values() if v == 'value'),
        'kinds': kinds,
        'keys': keys,
    }
    with open(out_path, 'w', encoding='cp1252') as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print(f"lumenairy.__file__ = {lumenairy.__file__}")
    nv = sum(1 for v in kinds.values() if v == 'value')
    print(f"n_keys = {len(keys)} ({nv} values, {len(keys)-nv} refusals) "
          f"-> {out_path}")


if __name__ == '__main__':
    main(sys.argv[1])
