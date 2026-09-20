"""(4) JAX vs NumPy parity for _collins_transport, against a bar I derive.

MY fixture, not the author's: N = 96 (not 64), dx = 5.5 um (not 8), R_in =
-28 mm (not -50), z = 2.3 mm (not 5), lambda = 1.064 um (not 633 nm),
w = 41 um, R_ref = -21.5 mm.

MY chain depth, counted from the source (printed with the reading):

  NumPy arm (xp is np, _EXACT_READOUT_SEPARABLE_BLUESTEIN True, so
  _bluestein_2d takes the SEPARABLE route):
    measurement forward FFT                      1 two-dimensional
    exact-kernel correction's inverse            1 two-dimensional ('exact')
    _bluestein_axis_1d x 2 axes x 3 passes       6 one-dimensional passes
                                                 = 3 two-dimensional
                                                   equivalents
  JAX arm (xp is jnp, separable route is NumPy-only, so the 2-D arm runs):
    measurement forward FFT                      1
    exact-kernel correction's inverse            1 ('exact' only)
    _bluestein_2d: fft2(g_pad), fft2(h_2d),
                   ifft2(G*H)                    3

  -> 'fresnel': 4 transform-equivalents on both arms.
     'exact'  : 5 transform-equivalents on both arms.

  Round-off through unitary transforms adds at worst linearly, so
  depth x (single-transform spread) is a bound, not a fit.  A floor of
  32*eps keeps a bar of exactly zero from being a coin toss.

    python v2_jaxbar.py <tree> <out.json>
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                                            # noqa: E402
from vlib import anchor, build_tag, write_json                 # noqa: E402

WL = 1.064e-6
N = 96
DX = 5.5e-6
W0 = 41e-6
R_IN = -0.028
Z = 2.3e-3
R_REF = -0.0215


def gauss(n=N, dx=DX, w=W0, dtype=np.complex128):
    ax = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(ax, ax)
    # a mild off-axis tilt so the fixture is not real-symmetric
    return (np.exp(-(X ** 2 + Y ** 2) / (w * w))
            * np.exp(1j * 2.0 * np.pi * (0.07 * X + 0.03 * Y) / dx * 1e-1)
            ).astype(dtype)


def rel(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def main():
    tree, out_path = sys.argv[1], sys.argv[2]
    anchor(tree)
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    import lumenairy.propagators.carrier as CA
    from lumenairy.propagators.fft_infra import _fft2

    res = {'build': build_tag(), 'tree': tree,
           'jax_version': jax.__version__,
           'numpy_version': np.__version__,
           'x64': bool(jax.config.read('jax_enable_x64')),
           'fixture': dict(N=N, dx=DX, w=W0, R_in=R_IN, z=Z, wavelength=WL,
                           R_ref=R_REF),
           'separable_flag': bool(CA._EXACT_READOUT_SEPARABLE_BLUESTEIN)}

    env = gauss()
    ej = jnp.asarray(env, dtype=jnp.complex128)

    # ---- (i) MY single forward-FFT spread, on MY fixture -----------------
    a = np.asarray(_fft2(np.ascontiguousarray(env, dtype=np.complex128)))
    b = np.asarray(jnp.fft.fft2(ej))
    spread = float(np.linalg.norm(a - b) / np.linalg.norm(a))
    res['single_fft_spread'] = spread
    # a second shape, to show the spread is a property and not a fluke
    e2 = gauss(n=128, dx=4.0e-6, w=30e-6)
    a2 = np.asarray(_fft2(np.ascontiguousarray(e2, dtype=np.complex128)))
    b2 = np.asarray(jnp.fft.fft2(jnp.asarray(e2, dtype=jnp.complex128)))
    res['single_fft_spread_128'] = float(
        np.linalg.norm(a2 - b2) / np.linalg.norm(a2))
    res['single_ifft_spread'] = float(np.linalg.norm(
        np.asarray(CA._fft2_pair(np, False)[1](
            np.ascontiguousarray(env, dtype=np.complex128)))
        - np.asarray(jnp.fft.ifft2(ej)))
        / np.linalg.norm(np.asarray(jnp.fft.ifft2(ej))))

    eps = float(np.finfo(np.float64).eps)
    depth = {'fresnel': 4.0, 'auto': 5.0, 'exact': 5.0}
    res['chain_depth'] = depth
    res['bar'] = {k: max(v * spread, 32.0 * eps) for k, v in depth.items()}

    # ---- (ii) the measured disagreement ---------------------------------
    KW = dict(dx_out=DX, dy_out=DX, N_out_x=N, N_out_y=N, R_ref=R_REF,
              on_collins_sampling='ignore')
    got = {}
    for gk in ('fresnel', 'auto', 'exact'):
        an = np.asarray(CA._collins_transport(env, R_IN, Z, WL, DX, DX,
                                              gap_kernel=gk, **KW))
        bj = np.asarray(CA._collins_transport(ej, R_IN, Z, WL, DX, DX,
                                              gap_kernel=gk, **KW))
        got[gk] = rel(bj, an)
    res['jax_vs_numpy'] = got
    res['inside_bar'] = {k: bool(got[k] < res['bar'][k]) for k in got}

    # which kernel did 'auto' actually resolve to?
    st = {}
    CA._collins_transport(env, R_IN, Z, WL, DX, DX, gap_kernel='auto',
                          stats_out=st, **KW)
    res['auto_resolved_kernel'] = st.get('kernel')
    res['auto_k4'] = st.get('k4')
    res['kelly_worst'] = st.get('worst')

    # ---- (iii) the smallest REAL signal on my fixture --------------------
    fres = np.asarray(CA._collins_transport(env, R_IN, Z, WL, DX, DX,
                                            gap_kernel='fresnel', **KW))
    exact = np.asarray(CA._collins_transport(env, R_IN, Z, WL, DX, DX,
                                             gap_kernel='exact', **KW))
    res['signal_exact_vs_fresnel'] = rel(exact, fres)
    # a second, smaller real signal: a one-part-in-1e6 change of z
    z2 = Z * (1.0 + 1e-6)
    nudge = np.asarray(CA._collins_transport(env, R_IN, z2, WL, DX, DX,
                                             gap_kernel='fresnel', **KW))
    res['signal_z_nudge_1e-6'] = rel(nudge, fres)
    res['two_sided'] = {
        k: bool(res['bar'][k] < res['signal_exact_vs_fresnel'] / 10.0)
        for k in res['bar']}

    # ---- (iv) the astigmatic and tilted arms, same bar -------------------
    astig_kw = dict(KW, R_ref=(-0.0215, -0.031))
    an = np.asarray(CA._collins_transport(env, (-0.028, -0.041), Z, WL, DX,
                                          DX, gap_kernel='fresnel',
                                          **astig_kw))
    bj = np.asarray(CA._collins_transport(ej, (-0.028, -0.041), Z, WL, DX,
                                          DX, gap_kernel='fresnel',
                                          **astig_kw))
    res['jax_vs_numpy_astigmatic'] = rel(bj, an)
    an = np.asarray(CA._collins_transport(env, R_IN, Z, WL, DX, DX,
                                          gap_kernel='exact',
                                          tilt=(0.10, -0.06), **KW))
    bj = np.asarray(CA._collins_transport(ej, R_IN, Z, WL, DX, DX,
                                          gap_kernel='exact',
                                          tilt=(0.10, -0.06), **KW))
    res['jax_vs_numpy_tilted_exact'] = rel(bj, an)

    # complex64 dtype contract
    e64 = gauss(dtype=np.complex64)
    res['c64_numpy_dtype'] = str(CA._collins_transport(
        e64, R_IN, Z, WL, DX, DX, gap_kernel='fresnel', **KW).dtype)
    res['c64_jax_dtype'] = str(CA._collins_transport(
        jnp.asarray(e64, dtype=jnp.complex64), R_IN, Z, WL, DX, DX,
        gap_kernel='fresnel', **KW).dtype)

    # ---- (v) the public entry, too ---------------------------------------
    pa = CA.propagate_carrier_referenced(env, R_IN, Z, WL, DX,
                                         transport='collins',
                                         gap_kernel='fresnel',
                                         on_collins_sampling='ignore')
    pb = CA.propagate_carrier_referenced(ej, R_IN, Z, WL, DX,
                                         transport='collins',
                                         gap_kernel='fresnel',
                                         on_collins_sampling='ignore')
    res['public_entry_rel'] = rel(pb.env, pa.env)
    res['public_entry_R_equal'] = bool(float(pb.R) == float(pa.R))
    res['public_entry_dx_equal'] = bool(float(pb.dx) == float(pa.dx))

    # ---- (vi) is the author's number reproducible IN KIND? ---------------
    # the author's own fixture, measured here for the comparison only
    axA = (np.arange(64) - 32) * 8e-6
    XA, YA = np.meshgrid(axA, axA)
    envA = np.exp(-(XA ** 2 + YA ** 2) / (60e-6) ** 2).astype(np.complex128)
    aA = np.asarray(_fft2(np.ascontiguousarray(envA, dtype=np.complex128)))
    bA = np.asarray(jnp.fft.fft2(jnp.asarray(envA, dtype=jnp.complex128)))
    sA = float(np.linalg.norm(aA - bA) / np.linalg.norm(aA))
    KA = dict(dx_out=8e-6, dy_out=8e-6, N_out_x=64, N_out_y=64,
              R_ref=-0.045, on_collins_sampling='ignore')
    rA = {}
    for gk in ('fresnel', 'auto'):
        x = np.asarray(CA._collins_transport(envA, -0.05, 5e-3, 633e-9,
                                             8e-6, 8e-6, gap_kernel=gk,
                                             **KA))
        y = np.asarray(CA._collins_transport(
            jnp.asarray(envA, dtype=jnp.complex128), -0.05, 5e-3, 633e-9,
            8e-6, 8e-6, gap_kernel=gk, **KA))
        rA[gk] = rel(y, x)
    res['authors_fixture'] = {'single_fft_spread': sA,
                              'bar_x6': 6.0 * sA,
                              'jax_vs_numpy': rA}

    write_json(res, out_path)
    print(json.dumps(res, indent=1, default=str))


if __name__ == '__main__':
    main()
