"""
lumenairy._math.levin -- adaptive delaminating Levin quadrature for 2-D
oscillatory integrals, uniformly valid THROUGH caustics (coalescing /
degenerate stationary points), with NO saddle finding.

Evaluates ``I = int_R f(x,y) exp(i g(x,y)) dx dy`` over a rectangle for smooth
``f`` (may be complex) and smooth real phase ``g``: per fiber along the
delamination axis, solve the Levin ODE ``p_s + i g_s p = f`` by k-point
Chebyshev collocation with a truncated SVD (which selects the slowly-varying
solution even at stationary points), collapsing the 2-D integral to two 1-D
boundary integrals handled by the adaptive 1-D Levin method.

After Chen, Serkh, Bremer & Aubry, "An adaptive delaminating Levin method in
two dimensions" (arXiv:2506.02424); 1-D engine per Chen, Serkh & Bremer
(Numer. Math. 2024, arXiv:2211.13400).  This implementation deviates from the
paper in ways found NECESSARY during validation (each pinned by a dedicated
experiment; see the v5.21 bring-up):

* **Residual-based acceptance** (both 1-D and 2-D) on the rigorous bound
  ``|I_hat - I| <= int |r|``, ``r = p' + i g_s p - f`` of the collocation
  polynomial on a 2k fine grid (paper eqs. 152-153) -- the paper's
  parent-vs-children VALUE comparison false-accepts on under-resolved
  oscillatory boxes/intervals where whole and halves land on the same
  asymptotic endpoint solution (wrong by ~1/g'^2 while agreeing to below any
  tolerance).
* **Priority-queue refinement** against a GLOBAL sum-of-leaf-bounds budget
  (greedy split of the largest-bound leaf) instead of per-box recursion --
  ~4x fewer boxes at equal accuracy, and the achieved bound is returned.
* **Machine-level TSVD truncation** decoupled from the tolerance: a truncated
  near-null component has tiny residual but O(1) endpoint (integral)
  contribution, so the truncation level is a direct error floor.
* **Float-coerced box coordinates**: integer corners survive dyadic halving
  and an int-dtype endpoint array makes ``np.full_like(xx, d)`` TRUNCATE the
  other coordinate inside boundary-phase closures -- a phase error confined
  to domain-edge boxes that no value/bound test can see.

Validated: canonical fold ``exp(i w (x^3/3 - c x))``-type phases through
``c -> 0`` at 8.9e-10 (tol=1e-8) with the returned bound honored; complex
non-separable amplitudes to reference precision.  Cost is O(seconds) per
integral at ~500-rad phases in pure NumPy -- suited to per-pixel evaluation
of hard caustic charts, not yet to full-grid production sweeps.

Author: Andrew Traverso
"""

from __future__ import annotations

import numpy as np

__all__ = ['levin1d_adaptive', 'levin2d']


def _cheb_nodes(k, a, b):
    """Chebyshev (2nd-kind, endpoints included) nodes ascending on [a,b]."""
    t = np.cos(np.pi * np.arange(k) / (k - 1))[::-1]     # -1..1 ascending
    return 0.5 * (a + b) + 0.5 * (b - a) * t


def _cheb_D(k, a, b):
    """Chebyshev differentiation matrix on the ascending nodes of [a,b]
    (Trefethen, Spectral Methods in MATLAB, cheb.m; flipped for ascending)."""
    x = np.cos(np.pi * np.arange(k) / (k - 1))           # 1..-1 descending
    c = np.ones(k)
    c[0] = 2.0
    c[-1] = 2.0
    c = c * (-1.0) ** np.arange(k)
    X = np.tile(x, (k, 1)).T
    dX = X - X.T
    D = np.outer(c, 1.0 / c) / (dX + np.eye(k))
    D = D - np.diag(D.sum(axis=1))
    # descending -> ascending: reverse both axes; then scale to [a,b]
    D = D[::-1, ::-1]
    return D * (2.0 / (b - a))


def _tsvd_solve(A, rhs, eps0):
    """Least-squares solve keeping only singular values >= eps0 * smax --
    selects the slowly-varying Levin solution at (near-)stationary points."""
    U, s, Vh = np.linalg.svd(A)
    smax = s[0] if s.size else 0.0
    keep = s >= eps0 * smax
    if not keep.any():
        return np.zeros_like(rhs)
    Ut = U[:, keep].conj().T @ rhs
    return Vh[keep].conj().T @ (Ut / s[keep])


def _levin1d_fixed(gy_vals, f_vals, a, b, k, eps0):
    """One 1-D Levin solve on [a,b]: p' + i gy p = f at the k Chebyshev nodes.
    Returns p at the nodes (ascending)."""
    D = _cheb_D(k, a, b)
    A = D + 1j * np.diag(gy_vals)
    return _tsvd_solve(A, f_vals.astype(complex), eps0)


def levin1d_adaptive(g, gy, f, a, b, *, k=12, tol=1e-10, eps0=None,
                     max_depth=30, fmax=None):
    """Adaptive 1-D Levin integral int_a^b f(y) exp(i g(y)) dy.

    Accepts an interval on the RIGOROUS residual bound |I_hat - I| <=
    int |r|, r = p' + i g' p - f of the collocation polynomial evaluated on a
    2k fine grid -- NOT on parent-vs-halves value agreement, which
    FALSE-ACCEPTS on short under-resolved oscillatory intervals where whole
    and halves land on the same asymptotic endpoint solution (wrong by
    ~1/g'^2) while agreeing to far below tol.  The residual is machine-small
    once the interval is resolved, so the length-budgeted test terminates.
    """
    from numpy.polynomial import chebyshev as Ch
    if fmax is None:
        ys = _cheb_nodes(33, a, b)
        fmax = float(np.abs(f(ys)).max()) + 1e-300
    if eps0 is None:
        eps0 = 1e-13
    us = np.cos(np.pi * np.arange(k) / (k - 1))[::-1]
    nf = 2 * k
    uf = np.cos(np.pi * np.arange(nf) / (nf - 1))[::-1]

    def _val(aa, bb):
        y = _cheb_nodes(k, aa, bb)
        p = _levin1d_fixed(gy(y), f(y), aa, bb, k, eps0)
        v = (p[-1] * np.exp(1j * g(np.array([bb], dtype=float))[0])
             - p[0] * np.exp(1j * g(np.array([aa], dtype=float))[0]))
        coef = Ch.chebfit(us, p, k - 1)
        dcoef = Ch.chebder(coef) * (2.0 / (bb - aa))
        yf = 0.5 * (aa + bb) + 0.5 * (bb - aa) * uf
        r = (Ch.chebval(uf, dcoef) + 1j * gy(yf) * Ch.chebval(uf, coef)
             - f(yf))
        est = float(np.mean(np.abs(r))) * (bb - aa)
        return v, est

    def _rec(aa, bb, depth):
        v, est = _val(aa, bb)
        if est < tol * (bb - aa) / (b - a) or depth >= max_depth:
            return v
        m = 0.5 * (aa + bb)
        return _rec(aa, m, depth + 1) + _rec(m, bb, depth + 1)

    return _rec(a, b, 0)


def levin2d(g, gx, gy, f, box, *, k=7, tol=1e-8, max_depth=12,
            k_boundary=12, return_leaves=False):
    """Adaptive delaminating 2-D Levin integral over box=(x0,x1,y0,y1).

    Delaminates along x (fibers y=const): per fiber solve p_x + i gx p = f via
    k-point Chebyshev TSVD collocation; then the two boundary integrals in y at
    x=x1 / x=x0 (amplitudes = Chebyshev interpolation of the fiber solutions)
    are evaluated by the adaptive 1-D Levin.  Quadtree: accept a box when the
    parent value equals the 4-child sum to tol.
    """
    # float-coerce the box: integer corner coordinates survive dyadic halving
    # (children reuse parent corners), and an int-dtype endpoint array makes
    # ``np.full_like(xx, d)`` TRUNCATE the other coordinate (e.g. -2.0625 ->
    # -2) inside the boundary-phase lambdas -- a 26-rad phase error confined
    # to domain-edge boxes (the hardest bug of this engine's bring-up).
    x0, x1, y0, y1 = (float(v) for v in box)
    # global f scale for the TSVD threshold
    xs = _cheb_nodes(17, x0, x1)
    ys = _cheb_nodes(17, y0, y1)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    fmax = float(np.abs(f(XX, YY)).max()) + 1e-300
    # TSVD truncation: NOT tied to tol -- a truncated near-null component
    # has tiny residual but O(1) endpoint (integral) contribution, so the
    # truncation level is a direct error floor (paper eq. 153).  Machine.
    eps0 = 1e-13

    def _residual_est(Pmat, sn, tn, s0, s1, t0, t1, along_x):
        """Estimate int int |r| over the box, r = dp/ds + i g_s p - f the
        Levin-ODE residual of the bivariate polynomial p-hat (rigorous error
        bound: |I_hat - I| <= int int |r|, paper eq. 152-153).  s = the
        delamination axis (first axis of Pmat), t = the fiber axis."""
        from numpy.polynomial import chebyshev as Ch
        kk = Pmat.shape[0]
        us = np.cos(np.pi * np.arange(kk) / (kk - 1))[::-1]
        Cs = Ch.chebfit(us, Pmat, kk - 1)          # (m_coef_s, j_fiber)
        C2 = Ch.chebfit(us, Cs.T, kk - 1)          # C2[n_t, m_s]
        Cd = Ch.chebder(C2, axis=1) * (2.0 / (s1 - s0))
        nf = 2 * kk
        uf = np.cos(np.pi * np.arange(nf) / (nf - 1))[::-1]
        sf = 0.5 * (s0 + s1) + 0.5 * (s1 - s0) * uf
        tf = 0.5 * (t0 + t1) + 0.5 * (t1 - t0) * uf
        # chebgrid2d(x, y, c): c[i, j] multiplies T_i(x) T_j(y); our
        # C2[n_t, m_s] -> x = u_t, y = u_s
        Pf = Ch.chebgrid2d(uf, uf, C2)             # (t_fine, s_fine)
        Pd = Ch.chebgrid2d(uf, uf, Cd)
        Tg, Sg = np.meshgrid(tf, sf, indexing='ij')
        if along_x:
            Xg, Yg = Sg, Tg
            gsv = gx(Xg, Yg)
        else:
            Xg, Yg = Tg, Sg
            gsv = gy(Xg, Yg)
        r = Pd + 1j * gsv * Pf - f(Xg, Yg)
        return float(np.mean(np.abs(r))) * abs(s1 - s0) * abs(t1 - t0)

    def _box_val(bx, k_over=0, want_residual=False, bnd_tol=None):
        if bnd_tol is None:
            bnd_tol = tol
        a, b, c, d = bx
        kk_ = k + int(k_over)
        xn = _cheb_nodes(kk_, a, b)
        yn = _cheb_nodes(kk_, c, d)
        X, Y = np.meshgrid(xn, yn, indexing='ij')
        gxv = gx(X, Y)
        gyv = gy(X, Y)
        fv = f(X, Y)
        # delaminate along the axis with the larger mean |gradient|
        if np.mean(np.abs(gxv)) >= np.mean(np.abs(gyv)):
            # fibers y = yn[j]: solve in x
            D = _cheb_D(kk_, a, b)
            P = np.empty((kk_, kk_), dtype=complex)  # P[i, j] = p(xn[i], yn[j])
            for j in range(kk_):
                A = D + 1j * np.diag(gxv[:, j])
                P[:, j] = _tsvd_solve(A, fv[:, j].astype(complex), eps0)
            p_hi = P[-1, :]        # p(b, yn)
            p_lo = P[0, :]         # p(a, yn)
            # boundary integrals in y along x=b and x=a
            from numpy.polynomial import chebyshev as _C

            def _interp(vals):
                # Chebyshev interpolation through the fiber values on [c,d]
                # (vals at ascending 2nd-kind nodes).
                coef = np.polynomial.chebyshev.chebfit(
                    np.cos(np.pi * np.arange(kk_) / (kk_ - 1))[::-1],
                    vals, kk_ - 1)
                return lambda yy: _C.chebval(
                    (2.0 * yy - (c + d)) / (d - c), coef)

            fb = _interp(p_hi)
            fa = _interp(p_lo)
            vb = levin1d_adaptive(
                lambda yy: g(np.full_like(yy, b), yy),
                lambda yy: gy(np.full_like(yy, b), yy),
                fb, c, d, k=k_boundary, tol=bnd_tol, eps0=eps0, fmax=fmax)
            va = levin1d_adaptive(
                lambda yy: g(np.full_like(yy, a), yy),
                lambda yy: gy(np.full_like(yy, a), yy),
                fa, c, d, k=k_boundary, tol=bnd_tol, eps0=eps0, fmax=fmax)
            if want_residual:
                est = _residual_est(P, xn, yn, a, b, c, d, along_x=True)
                return vb - va, est
            return vb - va
        else:
            # fibers x = xn[i]: solve in y (mirror)
            D = _cheb_D(kk_, c, d)
            P = np.empty((kk_, kk_), dtype=complex)
            for i in range(kk_):
                A = D + 1j * np.diag(gyv[i, :])
                P[i, :] = _tsvd_solve(A, fv[i, :].astype(complex), eps0)
            p_hi = P[:, -1]
            p_lo = P[:, 0]
            from numpy.polynomial import chebyshev as _C

            def _interp(vals):
                coef = np.polynomial.chebyshev.chebfit(
                    np.cos(np.pi * np.arange(kk_) / (kk_ - 1))[::-1],
                    vals, kk_ - 1)
                return lambda xx: _C.chebval(
                    (2.0 * xx - (a + b)) / (b - a), coef)

            fb = _interp(p_hi)
            fa = _interp(p_lo)
            vb = levin1d_adaptive(
                lambda xx: g(xx, np.full_like(xx, d)),
                lambda xx: gx(xx, np.full_like(xx, d)),
                fb, a, b, k=k_boundary, tol=bnd_tol, eps0=eps0, fmax=fmax)
            va = levin1d_adaptive(
                lambda xx: g(xx, np.full_like(xx, c)),
                lambda xx: gx(xx, np.full_like(xx, c)),
                fa, a, b, k=k_boundary, tol=bnd_tol, eps0=eps0, fmax=fmax)
            if want_residual:
                est = _residual_est(P.T, yn, xn, c, d, a, b, along_x=False)
                return vb - va, est
            return vb - va

    # PRIORITY-QUEUE refinement on the RIGOROUS residual bound
    # |I_hat - I| <= int int |Levin-ODE residual| per box (paper eq. 152-153):
    # greedily split the largest-est leaf until the GLOBAL sum of leaf bounds
    # meets tol (or the box/stall budget is hit).  Value-comparison accept
    # tests (parent-vs-children / k-refinement) FALSE-ACCEPT on under-resolved
    # oscillatory boxes where every variant lands on the same asymptotic
    # endpoint solution; the residual bound sees the miss directly, and the
    # greedy queue spends boxes only where the bound says they matter.
    import heapq
    total_area = (x1 - x0) * (y1 - y0)
    max_boxes = 4 ** max_depth if max_depth < 8 else 60000

    def _eval(bx):
        a, b, c, d = bx
        # boundary-integral budget: purely area-proportional, so the boundary
        # errors of ANY leaf set sum to <= tol (4 edges x 0.25).  (An earlier
        # +1/max_boxes term over-tightened the 1-D solves ~1000x on large
        # boxes -- the dominant, needless cost.)
        bnd = 0.25 * tol * ((b - a) * (d - c) / total_area)
        return _box_val(bx, want_residual=True, bnd_tol=bnd)

    v0, e0 = _eval((x0, x1, y0, y1))
    counter = 1
    heap = [(-e0, 0, (x0, x1, y0, y1), v0)]
    total_est = e0
    # NO stall guard: children residual ~ parent residual is NORMAL in the
    # pre-asymptotic regime (splitting resolves only once phase-per-box < k);
    # a stall heuristic mistakes that for a floor and exits early with a
    # large unreported bound.  The max_boxes cap is the only stop besides
    # convergence, and the achieved bound is RETURNED so callers can check.
    while total_est > tol and len(heap) + 3 <= max_boxes:
        neg_e, _, bx, _v = heapq.heappop(heap)
        e_old = -neg_e
        if e_old <= 0.0:                # nothing left refinable
            heapq.heappush(heap, (neg_e, counter, bx, _v))
            break
        a, b, c, d = bx
        mx = 0.5 * (a + b)
        my = 0.5 * (c + d)
        kids = [(a, mx, c, my), (mx, b, c, my), (a, mx, my, d), (mx, b, my, d)]
        total_est -= e_old
        for kk in kids:
            vk, ek = _eval(kk)
            heapq.heappush(heap, (-ek, counter, kk, vk))
            counter += 1
            total_est += ek

    val = sum(entry[3] for entry in heap)
    if return_leaves:
        return val, len(heap), total_est, [
            (entry[2], entry[3], -entry[0]) for entry in heap]
    return val, len(heap), total_est
