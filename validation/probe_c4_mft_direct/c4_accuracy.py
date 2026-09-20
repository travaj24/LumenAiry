"""WP-C4 task 3 -- the accuracy ladder, against a reference whose phase is
reduced EXACTLY.

WHY AN EXACT-PHASE REFERENCE.  A reference that forms ``t = alpha*n*k`` in
float64 and then reduces it commits the same two roundings the dense route
commits, so it agrees with the dense route by CONSTRUCTION and reads 3e-16 at
every budget.  That is a measurement of the instrument, and it is the defect
VERIFY-WAVE5-HYGIENE2 round 2 raised as D-1.  Here ``alpha`` is a float64 and
therefore an exact rational and ``n``, ``k`` are integers, so in
:class:`fractions.Fraction` the product and its fractional part are exact and
the single float64 rounding lands on a number already inside ``[-1/2, 1/2)``.
The double sum is then accumulated with ``math.fsum``, so the reference is
correctly rounded in BOTH senses.  This is the construction
``validation/probe_wave5_hyg2_round3/r3_budget_exact.py`` built; it is rebuilt
rather than imported so the two are independent.

TWO LADDERS.

1.  THE SHAPE LADDER, 16 shapes -- eight the WP-C4 rule sends to the dense
    route and eight it leaves on the chirp-Z route, so the claim "the dense
    side is the more accurate side" is measured on BOTH sides of the boundary
    and not only where it is convenient.  At each shape the three routes and
    ``'auto'`` are measured against the exact reference, and against a DERIVED
    two-sided bar: each output point is a sum of ``n = Ny*Nx`` unit-modulus
    terms, so a summation whose growth factor is ``g`` commits at most
    ``g * eps * sum|E|`` of ABSOLUTE error; the reference's own growth is 1
    (correctly rounded), so route-against-reference is bounded by
    ``(g_route + 1) * eps * sum|E|`` with ``g = 3*log2(L^2)`` for a chirp-Z
    route and ``g = sqrt(n)`` for the dense route's two BLAS products.  The bar
    is ABSOLUTE, so the apples-to-apples reading against it is the max-abs
    departure; the relative L2 is printed beside it for scale.

2.  THE BUDGET LADDER, at the shipped ``N = 24 -> M = 12`` geometry and at one
    shape the rule sends to the dense route, over ten decades of phase budget.
    Both routes must follow ``rel ~ eps * budget`` -- the LAW, whose slope is
    fitted -- and the dense route must be the more accurate of the two by a
    BOUNDED factor and not by decades.  The warning is captured per call, so
    "the guard fires on the same law on both routes" is a reading and not a
    claim.

    PYTHONPATH=<tree> python c4_accuracy.py <tree> OUT.json
"""
from __future__ import annotations

import hashlib
import math
import os
import sys
import warnings
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c4lib  # noqa: E402

TAU = 2.0 * math.pi

#: ``(tag, N, M)`` -- square, non-centred.  The first eight sit at or under
#: 1/32 (the rule sends them to the dense route); the last eight sit above it.
#: The fsum reference costs ``M^2 N^2`` term-products, which is what caps the
#: dense side at ``N = 256``.  ``M = 1`` is deliberately absent: there every
#: output index is ``k = 0``, the phase is identically zero and the dense route
#: is EXACT, so the chirp-over-dense factor has no denominator -- a degenerate
#: fixture, not a strong one.
SHAPES = [
    ('d_224_7', 224, 7), ('d_64_2', 64, 2), ('d_96_3', 96, 3),
    ('d_128_4', 128, 4), ('d_160_5', 160, 5), ('d_192_6', 192, 6),
    ('d_256_8', 256, 8), ('d_128_2', 128, 2),
    ('c_32_2', 32, 2), ('c_64_4', 64, 4), ('c_96_6', 96, 6),
    ('c_128_8', 128, 8), ('c_64_8', 64, 8), ('c_24_12', 24, 12),
    ('c_32_16', 32, 16), ('c_48_24', 48, 24),
]

BUDGETS = (1e5, 1e7, 1e9, 1e10, 1e11, 1e12, 1e14, 1e15)


def exact_phase_rows(alpha, n_in, n_out):
    """``frac(alpha * n * k)`` in ``[-1/2, 1/2)``, reduced EXACTLY."""
    import numpy as np
    fa = Fraction(alpha)
    T = np.empty((n_out, n_in), dtype=np.float64)
    for k in range(n_out):
        fk = Fraction(k)
        for n in range(n_in):
            t = fa * fk * n
            t -= math.floor(t)                  # exact, into [0, 1)
            if t >= Fraction(1, 2):
                t -= 1                          # exact, into [-1/2, 1/2)
            T[k, n] = float(t)
    return T


def fsum_reference(E, alpha_x, alpha_y, M, sign=-1):
    """``F = Wy . E . Wx^T`` with an EXACT phase and ``math.fsum`` everywhere."""
    import numpy as np
    ny, nx = E.shape
    Wy = np.exp(1j * sign * TAU * exact_phase_rows(alpha_y, ny, M))
    Wx = np.exp(1j * sign * TAU * exact_phase_rows(alpha_x, nx, M))
    out = np.empty((M, M), dtype=np.complex128)
    for ky in range(M):
        wy = Wy[ky]
        for kx in range(M):
            T = E * (wy[:, None] * Wx[kx][None, :])
            out[ky, kx] = complex(math.fsum(T.real.ravel()),
                                  math.fsum(T.imag.ravel()))
    return out


def bars(E, N, M, alpha, eps):
    """``(bar_chirp, bar_dense, sum|E|, detail)`` -- absolute, DERIVED.

    Each output point is a sum of ``n = Ny*Nx`` terms whose kernel has unit
    modulus, so the ABSOLUTE error a route commits against a correctly-rounded
    reference has exactly two sources and the bar is their sum:

    * SUMMATION.  A summation whose growth factor is ``g`` commits at most
      ``g * eps * sum|E|``.  ``g = 3*log2(L^2)`` for a chirp-Z route (three
      FFTs of length ``L^2``), ``g = sqrt(n)`` for the dense route's two BLAS
      products, and ``g = 1`` for the ``math.fsum`` reference -- so the pair
      contributes ``(g_route + 1) * eps * sum|E|``.
    * PHASE.  The reference's phase is reduced EXACTLY; the route's is a
      float64 product ``t`` that has already lost its low bits, costing
      ``~eps*|t|`` of phase and therefore ``~2*pi*eps*|t|`` of relative error
      on each unit-modulus kernel entry, i.e. ``2*pi*eps*max|t| * sum|E|``
      absolute.

    AND ``max|t|`` IS NOT THE SAME FOR THE TWO ROUTES, which is the whole
    reason the dense route's advantage grows in the region the WP-C4 rule
    captures.  The chirp-Z route builds ``exp(sign*pi*j*alpha*n^2)`` with ``n``
    running to ``N_max = max(N, M)``, so it spends ``alpha * N_max^2`` -- the
    phase budget the guard is named for.  The dense route builds
    ``exp(sign*2*pi*j*alpha*n*k)`` with ``n < N`` and ``k < M``, so it spends
    only ``alpha * (N-1) * (M-1)``, which at ``M = N/32`` is 32 times smaller.
    At a budget where the phase term dominates the summation term, that factor
    IS the accuracy gap.
    """
    import numpy as np
    from scipy.fft import next_fast_len
    n = int(E.size)
    s = float(np.sum(np.abs(E)))
    L = int(next_fast_len(int(N + M - 1)))
    g_chirp = 3.0 * math.log2(float(L) ** 2)
    g_dense = math.sqrt(float(n))
    t_chirp = abs(alpha) * float(max(N, M)) ** 2
    t_dense = abs(alpha) * float(N - 1) * float(max(M - 1, 1))
    bar_c = (g_chirp + 1.0 + TAU * t_chirp) * eps * s
    bar_d = (g_dense + 1.0 + TAU * t_dense) * eps * s
    return bar_c, bar_d, s, {'g_chirp': g_chirp, 'g_dense': g_dense,
                             'max_t_chirp': t_chirp, 'max_t_dense': t_dense,
                             'L': L}


def rel(a, b):
    import numpy as np
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def fit(budgets, vals):
    import numpy as np
    sel = [(b, v) for b, v in zip(budgets, vals) if 1e-14 < v < 1e-1]
    if len(sel) < 3:
        return None
    x = np.log10([s[0] for s in sel])
    y = np.log10([s[1] for s in sel])
    return {'slope': float(np.polyfit(x, y, 1)[0]), 'n': len(sel),
            'decades': float(x.max() - x.min())}


def run_routes(B, np, fft2, ifft2, E, alpha, M, sign=-1):
    """``{name: (array, warned)}`` for auto / bluestein / separable / direct."""
    out = {}
    for name, kw in (('auto', {}), ('bluestein', {'method': 'bluestein'}),
                     ('separable', {'method': 'separable'}),
                     ('direct', {'method': 'direct'})):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            c4lib.cold()
            arr = B._bluestein_2d(E, alpha, alpha, M, M, sign=sign, xp=np,
                                  fft2=fft2, ifft2=ifft2, **kw)
        out[name] = (arr, [str(x.message) for x in w
                           if issubclass(x.category, RuntimeWarning)])
    return out


def main(tree, out_path):
    lum = c4lib.anchor(tree)
    import numpy as np
    from lumenairy.propagators import _bluestein as B
    from lumenairy.propagators.fft_infra import _fft2, _ifft2

    eps = float(np.finfo(np.float64).eps)
    out = {'build': c4lib.build_tag(), 'python': sys.version.split()[0],
           'lumenairy_file': lum.__file__, 'eps': eps,
           'ratio': float(B._MFT_DIRECT_MAX_RATIO),
           'shape_ladder': [], 'budget_ladder': {}}

    # ---------------- 1.  the shape ladder --------------------------------
    # TWO budgets, because they measure two different things and the "bounded
    # factor" claim belongs to only one of them:
    #   budget 1   -- the phase term (2*pi*eps*max|t|) is of the same order as
    #                 the summation term, so the reading is essentially the
    #                 SUMMATION comparison, which is where hygiene-2's
    #                 "2.4x to 12.6x" lives;
    #   budget 1e3 -- six decades under the guard and of the order a real MFT
    #                 grid spends, where the phase term dominates and the
    #                 dense route's advantage is the ratio of the two routes'
    #                 max|t|, i.e. roughly N/M.  Both are reported.
    print("  budget  tag        N    M   rule  chirp rel     dense rel     "
          "factor  auto==     room_c room_d")
    for budget in (1.0, 1.0e3):
      for tag, N, M in SHAPES:
        # a STABLE seed: ``hash(str)`` is salted per interpreter, so it would
        # give the two builds different fixtures and make the readings
        # incomparable across them.
        seed = int.from_bytes(hashlib.sha256(tag.encode()).digest()[:4],
                              'big')
        rng = np.random.default_rng(seed)
        E = (rng.standard_normal((N, N))
             + 1j * rng.standard_normal((N, N))).astype(np.complex128)
        alpha = budget / float(max(N, M)) ** 2
        ref = fsum_reference(E, alpha, alpha, M)
        got = run_routes(B, np, _fft2, _ifft2, E, alpha, M)
        bar_c, bar_d, s, detail = bars(E, N, M, alpha, eps)
        says = bool(B._auto_selects_direct(N, N, M, M))
        r = {name: rel(a, ref) for name, (a, _) in got.items()}
        mx = {name: float(np.max(np.abs(a - ref)))
              for name, (a, _) in got.items()}
        auto_is = ('direct' if np.array_equal(
            got['auto'][0].view(np.uint8), got['direct'][0].view(np.uint8))
            else 'bluestein' if np.array_equal(
                got['auto'][0].view(np.uint8),
                got['bluestein'][0].view(np.uint8)) else 'NONE')
        row = {'tag': tag, 'N': N, 'M': M, 'ratio': M / N,
               'budget': budget, 'alpha': alpha, 'bar_detail': detail,
               'rule_says_direct': says, 'auto_matches': auto_is,
               'rel': r, 'max_abs': mx, 'sum_abs_E': s,
               'bar_chirp': bar_c, 'bar_dense': bar_d,
               'chirp_inside_bar': bool(mx['bluestein'] < bar_c),
               'separable_inside_bar': bool(mx['separable'] < bar_c),
               'dense_inside_bar': bool(mx['direct'] < bar_d),
               'factor_chirp_over_dense': (r['bluestein'] / r['direct']
                                           if r['direct'] > 0.0
                                           else float('inf')),
               'decades_of_room_chirp': math.log10(bar_c / mx['bluestein']),
               'decades_of_room_dense': math.log10(bar_d / mx['direct'])}
        out['shape_ladder'].append(row)
        print(f"  {budget:6.0f}  {tag:10s} {N:4d} {M:4d}  "
              f"{'dense' if says else 'chirp':5s}  "
              f"{r['bluestein']:.4e}  {r['direct']:.4e}  "
              f"{row['factor_chirp_over_dense']:8.2f}  {auto_is:9s} "
              f"{row['decades_of_room_chirp']:6.2f} "
              f"{row['decades_of_room_dense']:6.2f}", flush=True)
        del E

    # ---------------- 2.  the budget ladder -------------------------------
    # ``dense_side_64_2`` is kept as a DEGENERATE CONTROL and labelled one.
    # There ``alpha = budget / 64^2`` is an exact integer at every power-of-ten
    # budget on the ladder (``1e15 / 4096 = 244140625000``), so ``t = alpha*n``
    # is an exact whole number of turns, ``t - rint(t)`` is exactly 0, and the
    # EXACT reference computes exactly 0 too.  The dense route then reads
    # 1.9e-16 at every budget -- which looks like immunity and is a fixture
    # that agrees by construction, the same shape as the D-1 defect one level
    # down.  ``dense_side_96_3`` is the non-degenerate one the claim is read
    # from (``1e15 / 9216`` is not an integer).
    for geo_tag, N, M in (('shipped_24_12', 24, 12),
                          ('dense_side_96_3', 96, 3),
                          ('dense_side_64_2_DEGENERATE', 64, 2)):
        rng = np.random.default_rng(5)
        E = (rng.standard_normal((N, N))
             + 1j * rng.standard_normal((N, N))).astype(np.complex128)
        rows = []
        for budget in BUDGETS:
            alpha = budget / float(N) ** 2
            ref = fsum_reference(E, alpha, alpha, M)
            got = run_routes(B, np, _fft2, _ifft2, E, alpha, M)
            rows.append({
                'budget': budget, 'alpha': alpha, 'eps_budget': eps * budget,
                'alpha_is_whole_turn_per_index': bool(alpha == int(alpha)),
                'chirp_vs_exact': rel(got['bluestein'][0], ref),
                'dense_vs_exact': rel(got['direct'][0], ref),
                'auto_vs_exact': rel(got['auto'][0], ref),
                'factor_chirp_over_dense': (
                    rel(got['bluestein'][0], ref)
                    / rel(got['direct'][0], ref)
                    if rel(got['direct'][0], ref) > 0.0 else float('inf')),
                'C_chirp': rel(got['bluestein'][0], ref) / (eps * budget),
                'C_dense': rel(got['direct'][0], ref) / (eps * budget),
                'warned_chirp': len(got['bluestein'][1]),
                'warned_dense': len(got['direct'][1]),
                'warned_auto': len(got['auto'][1]),
                'auto_message_names_dense': any(
                    'ALREADY on the dense route' in m
                    for m in got['auto'][1])})
        out['budget_ladder'][geo_tag] = {
            'N': N, 'M': M,
            'rule_says_direct': bool(B._auto_selects_direct(N, N, M, M)),
            'rows': rows,
            'slope_chirp': fit([r['budget'] for r in rows],
                               [r['chirp_vs_exact'] for r in rows]),
            'slope_dense': fit([r['budget'] for r in rows],
                               [r['dense_vs_exact'] for r in rows])}
        print(f"--- budget ladder, {geo_tag} "
              f"({'dense' if out['budget_ladder'][geo_tag]['rule_says_direct'] else 'chirp'} side) ---")
        print("   budget    chirp/exact  dense/exact  eps*budget   "
              "C_chirp  C_dense  chirp/dense  warn c/d/auto")
        for r in rows:
            print(f"   {r['budget']:.0e}  {r['chirp_vs_exact']:.4e}   "
                  f"{r['dense_vs_exact']:.4e}   {r['eps_budget']:.3e}  "
                  f"{r['C_chirp']:7.4f}  {r['C_dense']:7.4f}  "
                  f"{r['factor_chirp_over_dense']:9.3f}   "
                  f"{r['warned_chirp']}/{r['warned_dense']}/"
                  f"{r['warned_auto']}"
                  f"{'  (auto names dense)' if r['auto_message_names_dense'] else ''}")
        print("   slopes:",
              {k: (None if out['budget_ladder'][geo_tag][k] is None
                   else round(out['budget_ladder'][geo_tag][k]['slope'], 4))
               for k in ('slope_chirp', 'slope_dense')})
        del E

    out['summary'] = {
        'n_rows': len(out['shape_ladder']),
        'n_shapes': len(SHAPES),
        'n_dense_side': sum(1 for t, N, M in SHAPES
                            if B._auto_selects_direct(N, N, M, M)),
        'n_chirp_side': sum(1 for t, N, M in SHAPES
                            if not B._auto_selects_direct(N, N, M, M)),
        'auto_matched_rule': sum(
            1 for r in out['shape_ladder']
            if (r['auto_matches'] == 'direct') == r['rule_says_direct']),
        'auto_third_arithmetic': sum(1 for r in out['shape_ladder']
                                     if r['auto_matches'] == 'NONE'),
        'dense_more_accurate_everywhere': all(
            r['factor_chirp_over_dense'] > 1.0
            for r in out['shape_ladder']),
        'all_inside_bars': all(
            r['chirp_inside_bar'] and r['separable_inside_bar']
            and r['dense_inside_bar'] for r in out['shape_ladder']),
        'min_decades_of_room': min(
            min(r['decades_of_room_chirp'], r['decades_of_room_dense'])
            for r in out['shape_ladder'])}
    for budget in (1.0, 1.0e3):
        sel = [r for r in out['shape_ladder'] if r['budget'] == budget]
        dsel = [r for r in sel if r['rule_says_direct']]
        out['summary'][f'budget_{budget:g}'] = {
            'factor_min': min(r['factor_chirp_over_dense'] for r in sel),
            'factor_max': max(r['factor_chirp_over_dense'] for r in sel),
            'factor_min_dense_side': min(r['factor_chirp_over_dense']
                                         for r in dsel),
            'factor_max_dense_side': max(r['factor_chirp_over_dense']
                                         for r in dsel),
            'min_decades_of_room': min(
                min(r['decades_of_room_chirp'], r['decades_of_room_dense'])
                for r in sel)}
    c4lib.write_json(out, out_path)
    print("summary:", out['summary'])


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
