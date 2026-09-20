"""Margin audit of the FOUR new test files' bars: how much room does each
assertion actually have, on the running build?

Every number here is RE-MEASURED.  For each bar the probe reports the measured
quantity, the bar the test asserts against, and the margin in decades, so a bar
whose pass/fail boundary sits inside the cross-build spread of what it reads is
visible rather than argued.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib  # noqa: E402

TREE = os.environ.get('VHYG2_TREE', os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
vlib.anchor(os.path.join(TREE, 'lumenairy'))

import numpy as np  # noqa: E402

OUT = {'build': vlib.build_tag(), 'rows': []}


def row(name, measured, bar, note=''):
    dec = (float('inf') if measured == 0
           else float(np.log10(bar / measured)) if measured > 0 else None)
    OUT['rows'].append({'name': name, 'measured': measured, 'bar': bar,
                        'decades': dec, 'note': note})
    print(f"{name:58s} meas={measured:.6e} bar={bar:.6e} "
          f"decades={dec:.3f} {note}")


# ---------------------------------------------------------------- H2-1 bars
def h2_1():
    from scipy.fft import next_fast_len
    from lumenairy.propagators._bluestein import (
        _bluestein_2d, _clear_h_fft_cache)
    from lumenairy.propagators.fft_infra import _fft2, _ifft2
    EPS = float(np.finfo(np.float64).eps)

    def pairwise_ref(E, a, My, Mx, sign):
        ny, nx = E.shape
        n_x = np.arange(nx, dtype=np.float64)
        n_y = np.arange(ny, dtype=np.float64)
        out = np.empty((My, Mx), dtype=np.complex128)
        for ky in range(My):
            ty = a * float(ky) * n_y
            wy = np.exp(1j * sign * 2.0 * np.pi * (ty - np.rint(ty)))
            for kx in range(Mx):
                tx = a * float(kx) * n_x
                wx = np.exp(1j * sign * 2.0 * np.pi * (tx - np.rint(tx)))
                out[ky, kx] = np.sum(E * (wy[:, None] * wx[None, :]))
        return out

    for (ny, nx, my, mx) in ((16, 16, 8, 8), (32, 24, 16, 20), (48, 48, 24, 24),
                             (96, 96, 48, 48), (128, 128, 64, 64)):
        rng = np.random.default_rng(20260915)
        E = (rng.standard_normal((ny, nx))
             + 1j * rng.standard_normal((ny, nx))).astype(np.complex128)
        a = 1.0 / 64.0
        ref = pairwise_ref(E, a, my, mx, -1)
        n = ny * nx
        L = float(next_fast_len(int(max(ny, nx) + max(my, mx) - 1)))
        g_pair = float(np.log2(max(n / 128.0, 2.0)) + 8.0)
        g_chirp = float(3.0 * np.log2(L * L))
        g_dense_author = float(np.sqrt(n))
        # RIGOROUS alternative: the dense route is TWO inner products, of
        # lengths nx and ny.  Higham's deterministic bound for a length-m
        # inner product is m*eps/(1-m*eps); the probabilistic (Higham & Mary
        # 2019) one is ~sqrt(m)*eps.  Two products chain additively.
        g_dense_det = float(nx + ny)
        g_dense_prob = float(np.sqrt(nx) + np.sqrt(ny))
        s = float(np.sum(np.abs(E)))
        sig = float(np.max(np.abs(ref)))
        for method, kw in (('bluestein', dict(separable=False, method='auto')),
                           ('separable', dict(separable=True, method='auto')),
                           ('direct', dict(method='direct'))):
            _clear_h_fft_cache()
            F = _bluestein_2d(E, a, a, my, mx, sign=-1, xp=np, fft2=_fft2,
                              ifft2=_ifft2, **kw)
            err = float(np.max(np.abs(F - ref)))
            if method == 'direct':
                bars = {'author_sqrt_n': (g_dense_author + g_pair) * EPS * s,
                        'higham_det': (g_dense_det + g_pair) * EPS * s,
                        'higham_prob': (g_dense_prob + g_pair) * EPS * s}
            else:
                bars = {'author_sqrt_n': (g_chirp + g_pair) * EPS * s}
            for bn, bv in bars.items():
                row(f"h2_1 N{ny}x{nx}->M{my}x{mx} {method} [{bn}]", err, bv)
            OUT.setdefault('h2_1_detail', []).append({
                'shape': (ny, nx, my, mx), 'method': method, 'maxabs': err,
                'sum_abs': s, 'signal_max': sig,
                'bar_author': list(bars.values())[0],
                'bar_over_signal': list(bars.values())[0] / sig,
                'g_pair': g_pair, 'g_chirp': g_chirp,
                'g_dense_sqrt_n': g_dense_author,
                'g_dense_det': g_dense_det, 'g_dense_prob': g_dense_prob})


# ---------------------------------------------------------------- H2-2 bars
def h2_2():
    try:
        import jax
        import jax.numpy as jnp
    except ImportError:
        OUT['h2_2'] = 'jax-absent'
        return
    jax.config.update('jax_enable_x64', True)
    import lumenairy.propagators.carrier as CA
    WL, N, DX, R_IN, Z, R_REF = 633e-9, 64, 8e-6, -0.05, 5e-3, -0.045
    ax = (np.arange(N) - N // 2) * DX
    X, Y = np.meshgrid(ax, ax)
    env = np.exp(-(X ** 2 + Y ** 2) / (60e-6) ** 2).astype(np.complex128)

    def merit(amp):
        e = amp.astype(jnp.complex128)
        out = CA._collins_transport(e, R_IN, Z, WL, DX, DX, dx_out=DX,
                                    dy_out=DX, N_out_x=N, N_out_y=N,
                                    R_ref=R_REF, gap_kernel='fresnel',
                                    on_collins_sampling='ignore')
        return jnp.sum(jnp.abs(out) ** 2)

    amp0 = jnp.asarray(np.real(env))
    g = np.asarray(jax.grad(merit)(amp0))
    a = np.asarray(amp0)
    m = a > 0.05 * a.max()
    corr = float(np.corrcoef(g[m], a[m])[0, 1])
    # the test asserts corr > 1 - 1e-6, i.e. (1-corr) < 1e-6
    row("h2_2 grad-shape (1-corr) vs the test's PINNED 1e-6",
        1.0 - corr, 1e-6, f"corr={corr:.12f}")
    OUT['h2_2_corr'] = corr
    OUT['h2_2_one_minus_corr'] = 1.0 - corr
    # the gradient ladder
    ij = np.unravel_index(int(np.argmax(a)), a.shape)
    eps = float(np.finfo(np.float64).eps)
    lad = {}
    for h in (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6):
        def at(sgn):
            ap = a.copy()
            ap[ij] += sgn * h
            return float(merit(jnp.asarray(ap)))
        fd = (at(+1) - at(-1)) / (2.0 * h)
        lad[h] = abs(fd - g[ij]) / abs(g[ij])
    OUT['h2_2_ladder'] = lad
    best = min(lad.values())
    row("h2_2 grad best-of-ladder vs 10*eps^(2/3)", best,
        10.0 * eps ** (2.0 / 3.0), f"best_h={min(lad, key=lad.get)}")
    OUT['h2_2_std_over_max'] = (float(np.std(g))
                                / float(np.max(np.abs(g))))
    row("h2_2 falsification std/max vs pinned 1e-3",
        1e-3, OUT['h2_2_std_over_max'], "(inverted: measured is the BAR side)")


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if which in ('all', 'h1'):
        h2_1()
    if which in ('all', 'h2'):
        h2_2()
    tag = OUT['build']
    vlib.write_json(OUT, os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"v0_bar_margins_{tag.split('-')[0].lower()}.json"))
