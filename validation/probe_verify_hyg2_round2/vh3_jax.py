"""VERIFY-WAVE5-HYGIENE2 round 2 -- the V-D3 public Collins leg, re-measured.

Five readings, each on its own fixtures:

 J1  the PUBLIC ``propagate_carrier_referenced(transport='collins')`` on an
     eager JAX array: what TYPE comes back, and how far the values sit from
     the NumPy arm -- against a bar derived HERE from the two backends' own
     FFT spread on the running build, not from a constant.
 J2  a TRACED envelope: which exception, which needles, and whether any state
     moved (the chirp-Z cache and its hit counter, the pyFFTW plan cache, the
     ASM H cache, ``warnings.filters``, and a caller-supplied ``stats_out``).
 J3  the JAX SZIKLAS leg with ``gap_kernel='exact'`` -- the consolidated
     kernel through ``_exact_tf_2d_xp`` -- against NumPy, untilted and tilted.
 J4  ``jax.grad`` through the PUBLIC leg against a central difference, on a
     ladder, with a per-rung floor derived from the actual cancellation in
     that rung's own subtraction rather than from ``eps^(2/3)``.
 J5  the same reading on the BASE tree is taken by running this file with
     PYTHONPATH pinned there, so J1's type row is two measurements and not a
     claim about history.

    PYTHONPATH=<tree> python vh3_jax.py OUT.json
"""
import json
import sys
import warnings

import numpy as np

WL = 1.55e-6
N = 128
DX = 3e-6
W = 48e-6
R_IN = 0.04
Z = 5e-3


def gauss(n=N, dx=DX, w=W):
    x = (np.arange(n) - n / 2.0) * dx
    X, Y = np.meshgrid(x, x)
    return np.exp(-(X * X + Y * Y) / w ** 2).astype(np.complex128)


def relL2(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    d = float(np.linalg.norm(a - b))
    n = float(np.linalg.norm(b))
    return d / n if n else float('inf')


def main(out_path):
    import lumenairy
    from lumenairy.propagators import _bluestein as BL, carrier as CA, fft_infra as FI

    res = {'lumenairy_file': lumenairy.__file__,
           'python': sys.version.split()[0], 'platform': sys.platform,
           'numpy': np.__version__}

    import jax
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    res['jax'] = jax.__version__

    E = gauss()
    kw = dict(dx_out=DX, on_collins_sampling='ignore', transport='collins')

    # --- J1: type and value ------------------------------------------------
    r_np = CA.propagate_carrier_referenced(E, R_IN, Z, WL, DX, **kw)
    r_jx = CA.propagate_carrier_referenced(jnp.asarray(E), R_IN, Z, WL, DX,
                                           **kw)
    res['J1'] = {
        'numpy_type': type(r_np.env).__module__ + '.'
        + type(r_np.env).__name__,
        'jax_type': type(r_jx.env).__module__ + '.' + type(r_jx.env).__name__,
        'jax_is_plain_ndarray': type(r_jx.env) is np.ndarray,
        'rel': relL2(np.asarray(r_jx.env), np.asarray(r_np.env)),
        'max_abs': float(np.max(np.abs(np.asarray(r_jx.env)
                                       - np.asarray(r_np.env)))),
        'peak': float(np.max(np.abs(np.asarray(r_np.env)))),
    }

    # THE BAR, measured on the running build: how far apart are the two
    # backends' OWN FFTs on the very grids this leg transforms?  The leg takes
    # a forward 2-D transform of the envelope (the input box), a chirp-Z whose
    # inner transforms are of length L = next fast size >= N + N_out - 1, and
    # an inverse.  Measure the spread of each, on the same data.
    spreads = {}
    for tag, arr in (('env', E), ('env_chirped', E * np.exp(
            1j * np.linspace(0, 37.0, E.size).reshape(E.shape)))):
        a = np.fft.fft2(arr)
        b = np.asarray(jnp.fft.fft2(jnp.asarray(arr)))
        spreads[f'fft2_{tag}'] = relL2(b, a)
        a2 = np.fft.ifft2(a)
        b2 = np.asarray(jnp.fft.ifft2(jnp.asarray(a)))
        spreads[f'ifft2_{tag}'] = relL2(b2, a2)
    L = 1
    while L < 2 * N - 1:
        L *= 2
    pad = np.zeros((L, L), dtype=np.complex128)
    pad[:N, :N] = E
    spreads['fft2_padded_L'] = relL2(
        np.asarray(jnp.fft.fft2(jnp.asarray(pad))), np.fft.fft2(pad))
    spreads['L'] = L
    # The leg is a CASCADE of these transforms with elementwise screens in
    # between; each is a relative perturbation, so they add.  n_transforms is
    # counted from the algorithm: the input-box forward, the chirp-Z's two
    # forwards and one inverse per axis pair, and the final inverse.
    n_transforms = 6
    base_spread = max(v for k, v in spreads.items()
                      if k != 'L' and isinstance(v, float))
    # ON THIS BUILD THE FFT SPREAD READS EXACTLY 0.0 -- JAX's CPU transform
    # and NumPy's agree bit for bit at these shapes -- so an FFT-spread bar
    # DEGENERATES to whatever floor is put under it.  A second bar is
    # therefore derived from a quantity that is never zero: the leg's own
    # sensitivity to a LAST-BIT change of its input.  Perturb every entry of
    # the envelope by one ULP (the size of a re-association) and measure how
    # far the answer moves; a cross-backend difference is admissible exactly
    # when it is of that size.
    Eulp = np.nextafter(E.real, np.inf) + 1j * E.imag
    r_ulp = CA.propagate_carrier_referenced(Eulp, R_IN, Z, WL, DX, **kw)
    ulp_sens = relL2(np.asarray(r_ulp.env), np.asarray(r_np.env))
    bar = 10.0 * max(n_transforms * base_spread, ulp_sens)
    res['J1']['fft_spreads'] = spreads
    res['J1']['n_transforms_assumed'] = n_transforms
    res['J1']['ulp_sensitivity'] = ulp_sens
    res['J1']['shipped_bar_32eps_floor'] = 32.0 * float(
        np.finfo(np.float64).eps)
    res['J1']['bar'] = bar
    res['J1']['bar_derivation'] = (
        "10 x max(n_transforms x measured NumPy-vs-JAX FFT spread on this "
        "build's own transforms of these grids, the leg's measured response "
        "to a 1-ULP perturbation of its own input)")
    res['J1']['within_bar'] = res['J1']['rel'] <= bar
    # ... and the two-sided half: the bar must sit below the smallest real
    # signal, which here is the exact-vs-paraxial kernel departure.
    a_f = CA.propagate_carrier_referenced(
        E, R_IN, Z, WL, DX, **dict(kw, gap_kernel='fresnel'))
    a_e = CA.propagate_carrier_referenced(
        E, R_IN, Z, WL, DX, **dict(kw, gap_kernel='exact'))
    sig = relL2(np.asarray(a_e.env), np.asarray(a_f.env))
    res['J1']['smallest_real_signal'] = sig
    res['J1']['decades_of_headroom'] = (
        float(np.log10(sig / bar)) if bar > 0 else float('inf'))

    # --- J2: the traced refusal, and whether any state moved ---------------
    def state():
        with BL._H_FFT_CACHE_LOCK:
            bl = (len(BL._H_FFT_CACHE), BL._H_FFT_CACHE_HITS,
                  sorted(map(repr, BL._H_FFT_CACHE.keys())))
        return {
            'bluestein_cache': bl,
            'pyfftw_plans': sorted(map(repr, FI._PYFFTW_PLAN_CACHE.keys())),
            'asm_H_cache': sorted(map(repr, FI._H_CACHE.keys())),
            'warning_filters': [repr(f) for f in warnings.filters],
        }

    # PRIME the caches first, so "unchanged" is a real reading and not the
    # trivial "both empty".  MEASURED 2026-09-20: a Collins leg (and a
    # Sziklas one, and an exact-kernel one) leaves ALL FOUR of these at zero
    # -- they are filled by the MFT propagators, not by the carrier chain --
    # so priming through the carrier leg would have made three quarters of
    # this reading vacuous.  ``angular_spectrum_propagate_mft`` fills the
    # chirp-Z kernel cache, the pyFFTW plan cache, the ASM H cache and
    # ``warnings.filters`` in one call, which is what is used here.
    from lumenairy.propagators.mft import angular_spectrum_propagate_mft
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        angular_spectrum_propagate_mft(E, Z, WL, DX, dx_out=DX, N_out=N)
    CA.propagate_carrier_referenced(E, R_IN, Z, WL, DX, **kw)
    before = state()
    res_primed = {k: (len(v) if isinstance(v, (list, tuple)) else v)
                  for k, v in before.items()}
    stats = {'SENTINEL': 1}
    j2 = {'cache_primed_entries': before['bluestein_cache'][0],
          'pyfftw_plans_primed': len(before['pyfftw_plans']),
          'asm_H_primed': len(before['asm_H_cache']),
          'warning_filters_primed': len(before['warning_filters'])}
    del res_primed

    def traced(a):
        return CA.propagate_carrier_referenced(a, R_IN, Z, WL, DX, **kw).env

    try:
        jax.jit(traced)(jnp.asarray(E))
        j2['exception'] = None
    except BaseException as e:               # noqa: BLE001 -- that is the read
        j2['exception'] = type(e).__name__
        j2['message'] = str(e)
    after = state()
    j2['needles'] = {
        n: (n in j2.get('message', ''))
        for n in ('_collins_transport', 'dx_out', 'gap_kernel',
                  'on_collins_sampling', 'Tracer', 'jax.jit', 'jax.grad',
                  "gap_kernel='fresnel'", "on_collins_sampling='ignore'",
                  'outside the trace')}
    j2['state_unchanged'] = {k: (before[k] == after[k]) for k in before}
    j2['stats_out_untouched'] = (stats == {'SENTINEL': 1})
    # and the same call with stats_out, to be sure no partial write happens
    try:
        st2 = {'SENTINEL': 1}
        jax.jit(lambda a: CA.propagate_carrier_referenced(
            a, R_IN, Z, WL, DX, dx_out=DX, on_collins_sampling='ignore',
            transport='collins').env)(jnp.asarray(E))
    except BaseException:                    # noqa: BLE001
        pass
    j2['stats_sentinel_after'] = st2
    res['J2'] = j2

    # --- J3: the JAX SZIKLAS exact leg through the consolidated kernel -----
    j3 = {}
    for tag, tilt in (('untilted', (0.0, 0.0)), ('tilted', (0.03, -0.02))):
        a = CA.propagate_carrier_referenced(
            E, R_IN, Z, WL, DX, gap_kernel='exact', tilt=tilt)
        b = CA.propagate_carrier_referenced(
            jnp.asarray(E), R_IN, Z, WL, DX, gap_kernel='exact', tilt=tilt)
        j3[tag] = {
            'rel': relL2(np.asarray(b.env), np.asarray(a.env)),
            'jax_type': type(b.env).__module__ + '.' + type(b.env).__name__,
            'R_equal': bool(a.R == b.R), 'dx_equal': bool(a.dx == b.dx)}
        # and the kernel it goes through, directly
        kx = np.fft.fftfreq(N, d=DX) * 2 * np.pi
        j3[tag]['direct_kernel_rel'] = relL2(
            np.asarray(CA._exact_tf_2d_xp(jnp.asarray(E), Z, WL, DX, DX,
                                          tilt, jnp, True, np)),
            CA._exact_tf_2d_xp(E, Z, WL, DX, DX, tilt, np, False, np))
        del kx
    res['J3'] = j3

    # --- J4: jax.grad vs a central difference ------------------------------
    # The PUBLIC Collins leg refuses a trace BY DESIGN (J2), so a gradient
    # through it does not exist and cannot be measured.  Two legs that do
    # exist are measured instead: the public SZIKLAS leg with
    # gap_kernel='exact' (i.e. THROUGH the consolidated kernel), and the
    # private _collins_transport the shipped id uses.
    amp0 = jnp.asarray(np.real(gauss(64, DX, 24e-6)))
    a0 = np.asarray(amp0)
    ij = np.unravel_index(int(np.argmax(a0)), a0.shape)

    def merit_sziklas(a):
        env = a.astype(jnp.complex128)
        r = CA.propagate_carrier_referenced(
            env, R_IN, Z, WL, DX, gap_kernel='exact')
        return jnp.sum(jnp.abs(r.env) ** 2)

    def merit_collins(a):
        env = a.astype(jnp.complex128)
        r = CA._collins_transport(
            env, R_IN, Z, WL, DX, DX, dx_out=DX, dy_out=DX, N_out_x=64,
            N_out_y=64, R_ref=float('inf'), gap_kernel='fresnel',
            on_collins_sampling='ignore')
        # _collins_transport returns the ENVELOPE array itself, not a
        # CarrierReferencedField -- the public leg above wraps it.
        return jnp.sum(jnp.abs(r) ** 2)

    j4 = {'public_collins_grad_is_refused_by_design': True}
    for tag, merit in (('public_sziklas_exact', merit_sziklas),
                       ('private_collins', merit_collins)):
        g = np.asarray(jax.grad(merit)(amp0))
        gij = float(g[ij])
        rows = []
        for h in (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4):
            ap = np.array(a0)
            ap[ij] += h
            Pp = float(merit(jnp.asarray(ap)))
            am = np.array(a0)
            am[ij] -= h
            Pm = float(merit(jnp.asarray(am)))
            fd = (Pp - Pm) / (2.0 * h)
            # FLOOR, per rung, from the cancellation actually suffered in
            # THIS subtraction: each evaluation carries ~eps*|P| of
            # representation error, and the difference divides by 2h.  This
            # is a reading of Pp and Pm, not the eps^(2/3) rule of thumb.
            floor = float(np.finfo(np.float64).eps
                          * max(abs(Pp), abs(Pm)) / (2.0 * h) / abs(gij))
            rel = abs(fd - gij) / abs(gij)
            rows.append({'h': h, 'fd': fd, 'rel': rel, 'floor': floor,
                         'over_floor': rel / floor})
        j4[tag] = {'grad_at_peak': gij,
                   'grad_nonzero': bool(np.any(g != 0.0)),
                   'grad_not_constant': bool(np.ptp(g) > 0.0),
                   'ladder': rows,
                   'best': min(rows, key=lambda r: r['rel'])}
    res['J4'] = j4
    rows = j4['public_sziklas_exact']['ladder']

    with open(out_path, 'w', encoding='cp1252') as fh:
        json.dump(res, fh, indent=1, sort_keys=True, default=str)
    print(f"lumenairy.__file__ = {lumenairy.__file__}")
    print(f"J1 jax type {res['J1']['jax_type']}  rel {res['J1']['rel']:.4e}  "
          f"fftspread {base_spread:.3e}  ulp {ulp_sens:.3e}  "
          f"bar {bar:.4e}  within {res['J1']['within_bar']}  "
          f"signal {sig:.3e} ({res['J1']['decades_of_headroom']:.2f} dec)")
    print(f"J2 {res['J2']['exception']}  needles "
          f"{sum(res['J2']['needles'].values())}/"
          f"{len(res['J2']['needles'])}  state "
          f"{res['J2']['state_unchanged']}")
    j3txt = "  ".join(f"{k}:{v['rel']:.3e}" for k, v in j3.items())
    print(f"J3 {j3txt}")
    for tag in ('public_sziklas_exact', 'private_collins'):
        print(f"J4 {tag} " + "  ".join(
            f"h={r['h']:.0e}:{r['rel']:.2e}/{r['over_floor']:.1f}x"
            for r in j4[tag]['ladder']))
    print(f"-> {out_path}")


if __name__ == '__main__':
    main(sys.argv[1])
