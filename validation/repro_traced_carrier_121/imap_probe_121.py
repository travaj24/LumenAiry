# INVERSE (MIXED) CHARACTERISTIC feasibility prototype -- the MEASURED gate.
#
# LOCAL-ONLY (needs the 121 .zmx via _d121_common).  NO library edit: every
# lumenairy symbol used here is imported and CALLED, never modified.  Nothing
# propagates a field; every accuracy number below is map-vs-direct-ray-trace on
# design 121's own last post-DOE group.
#
# WHAT IS UNDER TEST.  PROTO_HAMILTON_MAP_2026_08_11 sec 8.3 redirected the
# speed work from the FORWARD point characteristic to the INVERSE (mixed) one:
#
#     G(x_out, y_out ; x_src, y_src) -> (x_in, y_in, OPL, det J)
#
# Chebyshev in all four variables, on the SAME two-parameter source label the
# forward map proved exact (sec 2.1: ``_tilted_carrier_parts`` depends on the
# congruence only through the source's transverse projection, and design 121's
# last-group entrance carries ONE order-independent R).  Contracting G at an
# order's own ``(x_src, y_src)`` yields a 2-D Chebyshev in EXIT coordinates that
# returns the entrance point directly -- ONE polynomial evaluation per exit
# pixel.
#
# THE OPEN QUESTION sec 8.3 left: the Chebyshev DEGREE needed in the EXIT
# coordinates.  The shipped path fits degree 6 in ENTRANCE coordinates and
# inverts; a direct exit-coordinate fit is a DIFFERENT function and needs more.
# Measured here at escalating degree, against:
#
#   * the DIRECT ray trace (ground truth), and
#   * the shipped per-pixel NEWTON answer (the incumbent), reproduced verbatim
#     from ``_lens_traced._invert_newton`` on the library's own
#     ``_Cheb2DEvaluator`` forward fits, so both arms see the SAME ray data.
#
# THE BAR IS THE FINDING.  sec 8.3 asked for lambda/100 with 3x margin.  The
# incumbent already delivers 1.11e-04 waves on this group (forward fit + Newton
# 6.65e-05, cubic upsample 4.44e-05), so lambda/100 is 90x LOOSER than the code
# an inverse map would replace, and sizing G to it would ship a 24x regression.
# The node/degree budget below is therefore reported against PARITY.
#
# MODES
#   degree   -- exit-degree x source-node sweep: OPL error in waves, entrance
#               position in nm, ray-tube amplitude relative error, at OFF-NODE
#               exit pixels and OFF-NODE source labels (the 32 orders' own,
#               incl. the extreme order), plus the incumbent's own fit-order
#               ladder.  ``pad=0.0`` gives the un-C6-padded control.
#   edge     -- error growth toward the exit-support rim (two arms: the order's
#               own pupil and the whole union pupil), plus the NODE-HULL
#               COVERAGE census that the source-axis contraction needs.
#   single   -- single-valuedness of the INVERSE over the domain, and its
#               reciprocity against the traced forward Jacobian.
#   ampctl   -- CONTROL: is the amplitude floor G's, or the det J estimator's?
#   upsample -- what an exact per-pixel inverse would BUY: the shipped coarse
#               lattice + cubic upsample vs a direct per-pixel Newton, inside
#               the landing hull (outside it the difference is 1e+04 waves of
#               EXTRAPOLATION and means nothing).
#   newton   -- the incumbent alone: iterations to converge, error vs the trace.
#   all      -- every mode above.
#
# Run:  python imap_probe_121.py all
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _d121_common as C  # noqa: E402
import hmap_probe_121 as H  # noqa: E402

import lumenairy as la  # noqa: E402
from lumenairy.elements._lens_traced import _Cheb2DEvaluator  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
LAM = C.LAM

# The niche-C6 residual-eikonal widening of the SOURCE box, measured in
# PROTO_HAMILTON_MAP_2026_08_11 sec 3.4 as max|grad a_fit| x |R| on the last
# group.  Same quantity, same reason: ``apply_real_lens_traced`` launches along
# grad(W + a_fit) and a_fit is per-ORDER, so a shared label box must cover it.
C6_PAD = 0.6884e-3

# The shipped retrace's own numbers at n_fine_cap = 8192 (the cost run of
# PROTO_HAMILTON_MAP_2026_08_11 sec 6.1).  ``DX_FINE`` sets the Newton
# convergence tolerance (0.01 * dx) and the exit-pixel pitch the error bars are
# quoted against; ``RS_FINE`` is the pitch-preserving ray_subsample.
DX_FINE = 1.5325e-6
N_FINE = 8192
RS_FINE = 87
N_LAUNCH = 229          # = int(2 * launch_radius / (dx_fine * rs_fine))

NEWTON_ITERS = 12       # _lens_traced._NEWTON_MAX_ITERS
FIT_ORDER = 6           # apply_real_lens_traced(newton_poly_order=6)


# ===========================================================================
# setup -- one traced-geometry bundle every mode starts from
# ===========================================================================
def setup(verbose=False):
    groups_post, rows, R_ent, period, R_doe = H.geometry()
    surfs, presc = H.group_surfaces(groups_post)
    ap = groups_post[-1]['prescription'].get('aperture_diameter')
    ap = float(ap)
    launch_radius = 0.75 * ap          # _lens_traced line 8131: 0.5 * ap * 1.5
    src = H.source_positions(R_ent, rows)
    reach = max(np.hypot(r['x_c'], r['y_c']) for r in rows)
    r_union = reach + H.R_PUPIL_ORDER
    S = {'groups_post': groups_post, 'rows': rows, 'R_ent': R_ent,
         'surfs': surfs, 'aperture': ap, 'launch_radius': launch_radius,
         'src': src, 'reach': reach, 'r_union': r_union,
         'r_pupil_order': H.R_PUPIL_ORDER}
    if verbose:
        print("SETUP -- design 121, last post-DOE group (group 5 of 6)")
        print("  R at the entrance          %.6f mm  (order-independent)"
              % (R_ent * 1e3))
        print("  aperture_diameter          %.4f mm -> launch radius %.4f mm"
              % (ap * 1e3, launch_radius * 1e3))
        print("  union entrance pupil       %.4f mm  (chief reach %.4f + "
              "half-window %.4f)" % (r_union * 1e3, reach * 1e3,
                                     H.R_PUPIL_ORDER * 1e3))
        print("  source box (unpadded)      %.4f x %.4f mm half-widths"
              % (0.5 * (src[:, 0].max() - src[:, 0].min()) * 1e3,
                 0.5 * (src[:, 1].max() - src[:, 1].min()) * 1e3))
    return S


def source_box(S, pad=C6_PAD):
    src = S['src']
    lo, hi = src.min(axis=0), src.max(axis=0)
    ctr = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo) + float(pad)
    return ctr, half


# ===========================================================================
# total-degree Chebyshev in the EXIT coordinates
# ===========================================================================
# Same basis the shipped direct-fit path (``inversion_method='fit'``,
# ``_invert_fit``) uses: total-degree multi-indices kx + ky <= deg, encoded as
# a (P, 2) int array for a vectorised column-product design matrix.  Keeping
# the basis identical is deliberate -- the degree number this study reports has
# to mean the same thing as the library's own ``newton_poly_order``.
def td_terms(deg):
    return np.array([[a, b] for a in range(deg + 1)
                     for b in range(deg + 1 - a)], dtype=np.intp)


def td_design(ux, uy, deg, terms):
    Vx = np.polynomial.chebyshev.chebvander(np.asarray(ux, dtype=float), deg)
    Vy = np.polynomial.chebyshev.chebvander(np.asarray(uy, dtype=float), deg)
    return Vx[:, terms[:, 0]] * Vy[:, terms[:, 1]]


# ===========================================================================
# the node trace -- one congruence's forward map on the shipped launch lattice
# ===========================================================================
def node_trace(S, xsrc, ysrc, n_lat=N_LAUNCH):
    """Trace ONE congruence (labelled by its source projection) over the
    SHIPPED launch lattice and return the forward map plus its Jacobian.

    The lattice is ``linspace(-launch_radius, launch_radius, n_lat)`` squared,
    exactly as ``apply_real_lens_traced`` builds it, so a map build costs
    exactly ``n_nodes`` of the traces the shipped per-order retrace already
    pays for one of.  ``det J`` comes from ``np.gradient`` on that regular
    lattice -- no extra rays, and it is the same quantity
    ``_ray_density_amp_grid`` consumes.
    """
    lr = S['launch_radius']
    xs = np.linspace(-lr, lr, n_lat)
    d = float(xs[1] - xs[0])
    PX, PY = np.meshgrid(xs, xs, indexing='ij')
    lx, ly = H.src_angle(S['R_ent'], xsrc, ysrc, PX.ravel(), PY.ravel())
    ch = H.characteristic(S['surfs'], PX.ravel(), PY.ravel(), lx, ly)
    sh = (n_lat, n_lat)
    XO = ch['x'].reshape(sh)
    YO = ch['y'].reshape(sh)
    OP = ch['opl'].reshape(sh)
    alive = ch['alive'].reshape(sh)
    XO = np.where(alive, XO, np.nan)
    YO = np.where(alive, YO, np.nan)
    OP = np.where(alive, OP, np.nan)
    dXdx, dXdy = np.gradient(XO, d, d)
    dYdx, dYdy = np.gradient(YO, d, d)
    detJ = dXdx * dYdy - dXdy * dYdx
    return {'xs': xs, 'd': d, 'PX': PX, 'PY': PY, 'x_out': XO, 'y_out': YO,
            'opl': OP, 'detj': detJ, 'alive': alive, 'n_lat': int(n_lat)}


def _fit_mask(S, tr):
    """Entrance samples that enter the exit-coordinate fit: inside the UNION
    pupil, alive, and with a finite Jacobian (which erodes one lattice ring
    around any dead ray, so a NaN neighbour can never enter the gradient)."""
    r2 = tr['PX'] ** 2 + tr['PY'] ** 2
    return ((r2 <= S['r_union'] ** 2) & tr['alive'] & np.isfinite(tr['detj'])
            & np.isfinite(tr['opl']))


# ===========================================================================
# BUILD G
# ===========================================================================
def build_G(S, nx, ny, deg, pad=C6_PAD, n_lat=N_LAUNCH, node_cache=None,
            verbose=False):
    """Build the 4-D inverse characteristic.

    Source axes: tensor Chebyshev-Lobatto NODES (``nx`` x ``ny``) on the
    C6-padded source-label box -- the parametrisation
    PROTO_HAMILTON_MAP_2026_08_11 proved exact.
    Exit axes: total-degree Chebyshev LEAST SQUARES of degree ``deg`` from the
    scattered ray landings, on ONE COMMON normalisation box shared by every
    node (without which the source-axis contraction is not a contraction of the
    same function).

    Four channels, in this order: ``x_in``, ``y_in``, ``OPL``, ``det J``.
    """
    ctr, half = source_box(S, pad)
    tx, ty = H.cheb_nodes(nx), H.cheb_nodes(ny)
    terms = td_terms(deg)
    P = terms.shape[0]

    t0 = time.perf_counter()
    nodes = []
    n_rays = 0
    for a in tx:
        for b in ty:
            xs_n = ctr[0] + half[0] * a
            ys_n = ctr[1] + half[1] * b
            key = (round(float(xs_n), 15), round(float(ys_n), 15), int(n_lat))
            if node_cache is not None and key in node_cache:
                tr = node_cache[key]
            else:
                tr = node_trace(S, xs_n, ys_n, n_lat)
                n_rays += n_lat * n_lat
                if node_cache is not None:
                    node_cache[key] = tr
            nodes.append((xs_n, ys_n, tr))
    t_trace = time.perf_counter() - t0

    # ONE common exit normalisation box: the intersection-safe choice is the
    # UNION bbox of every node's retained landings, so every node's fit spans
    # the same [-1, 1]^2 and the source-axis contraction is well posed.
    t0 = time.perf_counter()
    xo_lo = xo_hi = yo_lo = yo_hi = None
    masks = []
    for (_a, _b, tr) in nodes:
        m = _fit_mask(S, tr)
        masks.append(m)
        xo, yo = tr['x_out'][m], tr['y_out'][m]
        xo_lo = xo.min() if xo_lo is None else min(xo_lo, xo.min())
        xo_hi = xo.max() if xo_hi is None else max(xo_hi, xo.max())
        yo_lo = yo.min() if yo_lo is None else min(yo_lo, yo.min())
        yo_hi = yo.max() if yo_hi is None else max(yo_hi, yo.max())
    ex_c = np.array([0.5 * (xo_lo + xo_hi), 0.5 * (yo_lo + yo_hi)])
    ex_h = np.array([0.5 * (xo_hi - xo_lo), 0.5 * (yo_hi - yo_lo)])

    node_coef = np.empty((nx, ny, P, 4))
    resid = []
    n_samp = []
    for k, ((xs_n, ys_n, tr), m) in enumerate(zip(nodes, masks)):
        i, j = divmod(k, ny)
        ux = (tr['x_out'][m] - ex_c[0]) / ex_h[0]
        uy = (tr['y_out'][m] - ex_c[1]) / ex_h[1]
        A = td_design(ux, uy, deg, terms)
        B = np.stack([tr['PX'][m], tr['PY'][m], tr['opl'][m], tr['detj'][m]],
                     axis=1)
        cf, *_ = np.linalg.lstsq(A, B, rcond=None)
        node_coef[i, j] = cf
        resid.append(np.abs(A @ cf - B).max(axis=0))
        n_samp.append(int(m.sum()))
    # transform the SOURCE axes from node values to Chebyshev coefficients
    Vx = np.polynomial.chebyshev.chebvander(tx, nx - 1)
    Vy = np.polynomial.chebyshev.chebvander(ty, ny - 1)
    Cf = np.einsum('ai,bj,ijpc->abpc', np.linalg.inv(Vx), np.linalg.inv(Vy),
                   node_coef)
    t_fit = time.perf_counter() - t0

    G = {'nx': int(nx), 'ny': int(ny), 'deg': int(deg), 'pad': float(pad),
         'n_lat': int(n_lat), 'terms': terms, 'Cf': Cf,
         'src_c': ctr, 'src_h': half, 'exit_c': ex_c, 'exit_h': ex_h,
         'n_terms': int(P), 'n_nodes': int(nx * ny),
         'bytes': int(Cf.size * 8), 't_trace': t_trace, 't_fit': t_fit,
         'n_rays_traced': int(n_rays),
         'node_fit_resid_max': np.max(np.array(resid), axis=0).tolist(),
         'n_fit_samples': int(np.mean(n_samp))}
    if verbose:
        print("  built G  %dx%d nodes  exit deg %d (%d terms)  "
              "%d fit samples/node  %.1f kB  trace %.2f s  fit %.2f s"
              % (nx, ny, deg, P, G['n_fit_samples'], G['bytes'] / 1e3,
                 t_trace, t_fit))
    return G


def G_contract(G, xsrc, ysrc):
    """Contract G at ONE congruence label -> the (P, 4) exit-coordinate
    Chebyshev this order will evaluate per pixel.  This is the per-ORDER cost;
    everything after it is one polynomial evaluation per exit pixel."""
    ux = (float(xsrc) - G['src_c'][0]) / G['src_h'][0]
    uy = (float(ysrc) - G['src_c'][1]) / G['src_h'][1]
    Ta = np.polynomial.chebyshev.chebvander(np.array([ux]), G['nx'] - 1)[0]
    Tb = np.polynomial.chebyshev.chebvander(np.array([uy]), G['ny'] - 1)[0]
    return np.einsum('a,b,abpc->pc', Ta, Tb, G['Cf']), (ux, uy)


def G_eval(G, cvec, xq, yq):
    ux = (np.asarray(xq, dtype=float) - G['exit_c'][0]) / G['exit_h'][0]
    uy = (np.asarray(yq, dtype=float) - G['exit_c'][1]) / G['exit_h'][1]
    return td_design(ux, uy, G['deg'], G['terms']) @ cvec


# ===========================================================================
# THE INCUMBENT -- the shipped per-pixel Newton, reproduced
# ===========================================================================
def newton_forward_fit(S, xsrc, ysrc, n_lat=N_LAUNCH, order=FIT_ORDER,
                       tr=None):
    """The shipped forward fit: three ``_Cheb2DEvaluator``s of the entrance ->
    exit map on the launch lattice, restricted to the same union-pupil sample
    set G's fit uses (expressed, as the library does, through ``weights``)."""
    if tr is None:
        tr = node_trace(S, xsrc, ysrc, n_lat)
    m = _fit_mask(S, tr)
    w = m.astype(np.float64)
    xs = tr['xs']
    Sx = _Cheb2DEvaluator(xs, xs, np.where(m, tr['x_out'], 0.0), order=order,
                          weights=w)
    Sy = _Cheb2DEvaluator(xs, xs, np.where(m, tr['y_out'], 0.0), order=order,
                          weights=w)
    So = _Cheb2DEvaluator(xs, xs, np.where(m, tr['opl'], 0.0), order=order,
                          weights=w)
    ic = n_lat // 2
    dxs = float(xs[1] - xs[0])
    M_x = (tr['x_out'][ic + 1, ic] - tr['x_out'][ic - 1, ic]) / (2 * dxs)
    M_y = (tr['y_out'][ic, ic + 1] - tr['y_out'][ic, ic - 1]) / (2 * dxs)
    M_x = float(np.clip(abs(M_x), 1e-3, 1e3))
    M_y = float(np.clip(abs(M_y), 1e-3, 1e3))
    return {'Sx': Sx, 'Sy': Sy, 'So': So, 'inv_M_x': 1.0 / M_x,
            'inv_M_y': 1.0 / M_y, 'tr': tr}


def newton_invert(S, F, xq, yq, dx=DX_FINE, max_iters=NEWTON_ITERS):
    """``_lens_traced._invert_newton``, reproduced: clipped 2-D Newton on the
    forward Chebyshev fit, same initial guess, same tolerance (0.01 * dx), same
    iteration cap, same out-of-domain NaN rule.  Also returns the entrance
    point, the analytic ``det J`` there and the per-pixel iteration count."""
    Sx, Sy, So = F['Sx'], F['Sy'], F['So']
    lr = S['launch_radius']
    bound = lr * 0.999
    xw = np.asarray(xq, dtype=float).ravel()
    yw = np.asarray(yq, dtype=float).ravel()
    xe = xw * F['inv_M_x']
    ye = yw * F['inv_M_y']
    tol = 0.01 * float(dx)
    active = np.ones(xe.size, dtype=bool)
    iters = np.zeros(xe.size, dtype=int)
    n_evals = 0
    for _it in range(max_iters):
        if not active.any():
            break
        xa, ya = xe[active], ye[active]
        fx, jxx, jxy = Sx.ev_value_and_grad(xa, ya)
        fy, jyx, jyy = Sy.ev_value_and_grad(xa, ya)
        n_evals += 2
        rx = fx - xw[active]
        ry = fy - yw[active]
        det = jxx * jyy - jxy * jyx
        safe = np.abs(det) > 1e-12
        inv = np.where(safe, 1.0 / det, 0.0)
        xe[active] = np.clip(xa - (jyy * rx - jxy * ry) * inv, -bound, bound)
        ye[active] = np.clip(ya - (-jyx * rx + jxx * ry) * inv, -bound, bound)
        idx = np.where(active)[0]
        conv = np.sqrt(rx * rx + ry * ry) < tol
        iters[idx[~conv]] += 1
        active[idx[conv]] = False
    opl = So.ev(xe, ye)
    out_dom = (xe * xe + ye * ye) > (lr * 0.99) ** 2
    opl = np.where(out_dom, np.nan, opl)
    _fx, jxx, jxy = Sx.ev_value_and_grad(xe, ye)
    _fy, jyx, jyy = Sy.ev_value_and_grad(xe, ye)
    detj = jxx * jyy - jxy * jyx
    return {'x_in': xe, 'y_in': ye, 'opl': opl, 'detj': detj,
            'iters': iters, 'n_unconverged': int(active.sum()),
            'n_evals_per_pixel': int(n_evals), 'tol': tol}


# ===========================================================================
# the TEST set -- off-node exit pixels at off-node source labels
# ===========================================================================
def order_test_points(S, row, n_pup=41, offset=True, stencil_h=5e-6,
                      centre=None, radius=None):
    """Exit test points for ONE congruence, obtained by FORWARD-TRACING
    entrance points inside that order's own pupil disc.

    OFF-NODE BY CONSTRUCTION: the entrance lattice is offset by half a build
    cell from the build lattice, so no test exit point coincides with a fit
    sample, and none of them is a node of the source-axis interpolation either
    (the labels are the orders' own).

    Returns the exact entrance point, exit point, OPL and ``det J`` (4-point
    central stencil along the congruence -- the same estimator
    ``hmap_probe_121._congruence_jacobian`` uses).

    ``centre`` / ``radius`` override the entrance disc: passing the UNION pupil
    walks the test points all the way out to the map's own FIT-DOMAIN rim,
    which is the edge-honesty arm.
    """
    lr = S['launch_radius']
    cell = 2.0 * lr / (N_LAUNCH - 1)
    cx = float(row['x_c']) if centre is None else float(centre[0])
    cy = float(row['y_c']) if centre is None else float(centre[1])
    rad = S['r_pupil_order'] if radius is None else float(radius)
    t = np.linspace(-1.0, 1.0, n_pup) * rad
    PX, PY = np.meshgrid(cx + t, cy + t, indexing='ij')
    if offset:
        PX = PX + 0.5 * cell
        PY = PY + 0.5 * cell
    keep = ((PX - cx) ** 2 + (PY - cy) ** 2) <= rad ** 2
    keep &= (PX ** 2 + PY ** 2) <= (lr * 0.99) ** 2
    px, py = PX[keep], PY[keep]
    lx, ly = H.order_angle(S['R_ent'], row, px, py)
    ch = H.characteristic(S['surfs'], px, py, lx, ly)
    ok = ch['alive']
    h = stencil_h
    Pd = []
    for ddx, ddy in ((h, 0.0), (-h, 0.0), (0.0, h), (0.0, -h)):
        qx, qy = px + ddx, py + ddy
        qlx, qly = H.order_angle(S['R_ent'], row, qx, qy)
        Pd.append(H.characteristic(S['surfs'], qx, qy, qlx, qly))
        ok &= Pd[-1]['alive']
    detj = ((Pd[0]['x'] - Pd[1]['x']) * (Pd[2]['y'] - Pd[3]['y'])
            - (Pd[2]['x'] - Pd[3]['x']) * (Pd[0]['y'] - Pd[1]['y'])) \
        / (4 * h * h)
    return {'x_in': px[ok], 'y_in': py[ok], 'x_out': ch['x'][ok],
            'y_out': ch['y'][ok], 'opl': ch['opl'][ok], 'detj': detj[ok],
            'n': int(ok.sum())}


def _score(pred, truth, label=''):
    """OPL in waves, entrance position in nm, ray-tube amplitude |detJ|^-1/2
    relative."""
    e_opl = np.abs(pred[:, 2] - truth['opl']) / LAM
    e_pos = np.maximum(np.abs(pred[:, 0] - truth['x_in']),
                       np.abs(pred[:, 1] - truth['y_in'])) * 1e9
    a_p = np.abs(pred[:, 3]) ** -0.5
    a_t = np.abs(truth['detj']) ** -0.5
    e_amp = np.abs(a_p - a_t) / a_t
    return {'opl_max_waves': float(np.nanmax(e_opl)),
            'opl_rms_waves': float(np.sqrt(np.nanmean(e_opl ** 2))),
            'pos_max_nm': float(np.nanmax(e_pos)),
            'amp_rel_max': float(np.nanmax(e_amp)),
            'amp_rel_rms': float(np.sqrt(np.nanmean(e_amp ** 2))),
            'n': int(e_opl.size), 'label': label}


# ===========================================================================
# (1) THE DEGREE SWEEP
# ===========================================================================
def mode_degree(nlist=((4, 4), (5, 4), (6, 5), (7, 6), (8, 7), (9, 8),
                       (10, 9)),
                degs=(6, 8, 10, 12, 14), n_pup=31, pad=C6_PAD,
                inc_orders=(6, 8, 10, 12), verbose=True):
    """The two-axis sweep.  ``pad`` is the niche-C6 source-box widening (pass
    0.0 for the unpadded control); ``inc_orders`` is the INCUMBENT's own
    forward-fit ladder -- the shipped default is ``newton_poly_order=6``, which
    ``_DECENTRED_FIT_POLY_ORDER`` raises to 10 on a decentred disc, so the
    incumbent is not one number either."""
    S = setup(verbose=verbose)
    rows = S['rows']
    src = S['src']

    # the OFF-NODE test set: every one of the 32 orders, incl. the extreme
    # (-4, -2), on a half-cell-offset entrance lattice inside its own pupil.
    if verbose:
        print("\nbuilding the off-node test set (%d orders x <=%d^2 entrance "
              "points, 5 traces each) ..." % (len(rows), n_pup))
    t0 = time.perf_counter()
    tests = [order_test_points(S, r, n_pup=n_pup) for r in rows]
    if verbose:
        print("  %d test exit pixels in %.1f s"
              % (sum(t['n'] for t in tests), time.perf_counter() - t0))

    # --- the incumbent, at its own ladder of forward-fit orders ------------
    if verbose:
        print("\nINCUMBENT -- the shipped per-pixel Newton, on the forward "
              "Chebyshev fit at each of its own orders")
    incumbent = {}
    # one lattice trace per ORDER, reused across the fit-order ladder (the
    # ladder changes only the fit, never the rays)
    inc_tr = [None] * len(rows)
    for fo in inc_orders:
        inc = []
        t0 = time.perf_counter()
        for k, (r, tt) in enumerate(zip(rows, tests)):
            if inc_tr[k] is None:
                inc_tr[k] = node_trace(S, src[k, 0], src[k, 1])
            F = newton_forward_fit(S, src[k, 0], src[k, 1], order=fo,
                                   tr=inc_tr[k])
            nr = newton_invert(S, F, tt['x_out'], tt['y_out'])
            pred = np.stack([nr['x_in'], nr['y_in'], nr['opl'], nr['detj']],
                            axis=1)
            sc = _score(pred, tt, 'order %s' % (tuple(r['order']),))
            sc['iters_max'] = int(nr['iters'].max())
            sc['iters_mean'] = float(nr['iters'].mean())
            sc['n_unconverged'] = nr['n_unconverged']
            inc.append(sc)
        t_inc = time.perf_counter() - t0
        w = {k: max(s[k] for s in inc)
             for k in ('opl_max_waves', 'pos_max_nm', 'amp_rel_max')}
        w['iters_mean'] = float(np.mean([s['iters_mean'] for s in inc]))
        w['iters_max'] = int(max(s['iters_max'] for s in inc))
        w['n_unconverged'] = int(sum(s['n_unconverged'] for s in inc))
        w['tol_m'] = 0.01 * DX_FINE
        w['seconds_32_orders'] = t_inc
        w['n_terms'] = int((fo + 1) * (fo + 2) // 2)
        incumbent[str(fo)] = {'worst': w, 'per_order': inc}
        if verbose:
            print("  fwd fit order %2d (%3d terms): OPL %.4e w  entrance "
                  "%.4e nm  amp %.4e   [%.2f Newton iters mean, max %d, "
                  "%d unconverged]"
                  % (fo, w['n_terms'], w['opl_max_waves'], w['pos_max_nm'],
                     w['amp_rel_max'], w['iters_mean'], w['iters_max'],
                     w['n_unconverged']))
    inc_worst = incumbent[str(FIT_ORDER)]['worst']

    # --- G, at escalating exit degree and source node count ---------------
    cache = {}
    out = []
    bar3 = 0.01 / 3.0
    for (nx, ny) in nlist:
        for deg in degs:
            G = build_G(S, nx, ny, deg, pad=pad, node_cache=cache)
            worst = None
            per_order = []
            for k, (r, tt) in enumerate(zip(rows, tests)):
                cvec, uu = G_contract(G, src[k, 0], src[k, 1])
                pred = G_eval(G, cvec, tt['x_out'], tt['y_out'])
                sc = _score(pred, tt, 'order %s' % (tuple(r['order']),))
                sc['u_src'] = [float(uu[0]), float(uu[1])]
                per_order.append(sc)
                if worst is None or sc['opl_max_waves'] > worst['opl_max_waves']:
                    worst = sc
            row = {'nx': nx, 'ny': ny, 'nodes': nx * ny, 'deg': deg,
                   'n_terms': G['n_terms'], 'bytes': G['bytes'],
                   'opl_max_waves': max(s['opl_max_waves'] for s in per_order),
                   'opl_rms_waves': float(np.sqrt(np.mean(
                       [s['opl_rms_waves'] ** 2 for s in per_order]))),
                   'pos_max_nm': max(s['pos_max_nm'] for s in per_order),
                   'amp_rel_max': max(s['amp_rel_max'] for s in per_order),
                   'amp_rel_rms': float(np.sqrt(np.mean(
                       [s['amp_rel_rms'] ** 2 for s in per_order]))),
                   'worst_order': worst['label'],
                   'n_test': sum(s['n'] for s in per_order)}
            out.append(row)
            if verbose:
                v = row['opl_max_waves']
                bar = ('PASS(3x)' if v <= bar3
                       else ('inside' if v <= 0.01 else 'FAIL'))
                print("  %dx%d nodes, exit deg %2d (%3d terms, %6.1f kB): "
                      "OPL max %.4e w  entrance %.3e nm  amp %.3e  [%s]  %s"
                      % (nx, ny, deg, G['n_terms'], G['bytes'] / 1e3, v,
                         row['pos_max_nm'], row['amp_rel_max'],
                         row['worst_order'], bar))
    return {'incumbent': incumbent, 'incumbent_default_worst': inc_worst,
            'rows': out, 'pad': float(pad),
            'setup': {'R_ent': S['R_ent'], 'launch_radius': S['launch_radius'],
                      'r_union': S['r_union'], 'aperture': S['aperture']}}


# ===========================================================================
# (3a) EDGE HONESTY -- error toward the exit-support rim
# ===========================================================================
def mode_edge(nx=7, ny=6, degs=(8, 10, 12), n_pup=61, nbin=10, verbose=True):
    """How does the error behave at the FIT-DOMAIN edge?

    Two edges exist and they are not the same:

      * the ORDER's own exit footprint rim -- where the shipped path's convex
        hull test starts NaN-ing pixels;
      * the BUILD's exit box rim -- the outer edge of the union of every node's
        landings, past which the polynomial extrapolates.

    Measured as the error in radial bins of the order's own exit footprint,
    plus the worst node-footprint coverage deficit (how far a test point can
    sit outside the node whose landings define the fit there).
    """
    S = setup(verbose=False)
    rows = S['rows']
    src = S['src']
    # the extreme order and the on-axis order
    pick = []
    for k, r in enumerate(rows):
        if tuple(r['order']) in ((-4, -2), (0, 0)):
            pick.append(k)
    if not pick:
        pick = [0, len(rows) - 1]
    cache = {}
    out = []
    # ARM 1 = the ORDER's own pupil (the production domain); ARM 2 = the whole
    # UNION pupil, which walks the test points out to the map's own fit-domain
    # rim -- the frbf / skirt lesson is that the two behave differently and
    # only the second sizes a guard band.
    arms = (('order_pupil', None, None),
            ('union_pupil', (0.0, 0.0), S['r_union']))
    for deg in degs:
        G = build_G(S, nx, ny, deg, node_cache=cache)
        for k in pick:
          for (arm, ctr_a, rad_a) in arms:
            r = rows[k]
            tt = order_test_points(S, r, n_pup=n_pup, centre=ctr_a,
                                   radius=rad_a)
            cvec, _u = G_contract(G, src[k, 0], src[k, 1])
            pred = G_eval(G, cvec, tt['x_out'], tt['y_out'])
            e_opl = np.abs(pred[:, 2] - tt['opl']) / LAM
            e_pos = np.maximum(np.abs(pred[:, 0] - tt['x_in']),
                               np.abs(pred[:, 1] - tt['y_in'])) * 1e9
            a_p = np.abs(pred[:, 3]) ** -0.5
            a_t = np.abs(tt['detj']) ** -0.5
            e_amp = np.abs(a_p - a_t) / a_t
            cx = 0.5 * (tt['x_out'].max() + tt['x_out'].min())
            cy = 0.5 * (tt['y_out'].max() + tt['y_out'].min())
            rr = np.hypot(tt['x_out'] - cx, tt['y_out'] - cy)
            rmax = rr.max()
            bins = np.linspace(0.0, rmax, nbin + 1)
            prof = []
            for i in range(nbin):
                m = (rr >= bins[i]) & (rr <= bins[i + 1])
                if not m.any():
                    continue
                prof.append({'r_lo_frac': float(bins[i] / rmax),
                             'r_hi_frac': float(bins[i + 1] / rmax),
                             'n': int(m.sum()),
                             'opl_max_waves': float(e_opl[m].max()),
                             'pos_max_nm': float(e_pos[m].max()),
                             'amp_rel_max': float(e_amp[m].max())})
            # how far outside the BUILD exit box does this order reach?
            ux = (tt['x_out'] - G['exit_c'][0]) / G['exit_h'][0]
            uy = (tt['y_out'] - G['exit_c'][1]) / G['exit_h'][1]
            out.append({'deg': deg, 'order': list(r['order']), 'arm': arm,
                        'r_out_max_mm': float(rmax * 1e3),
                        'n_points': int(tt['n']),
                        'opl_max_waves': float(e_opl.max()),
                        'u_abs_max': float(max(np.abs(ux).max(),
                                               np.abs(uy).max())),
                        'profile': prof})
            if verbose:
                print("  deg %2d  order %-9s  [%-11s] exit footprint "
                      "r <= %.4f mm, |u| <= %.4f"
                      % (deg, str(tuple(r['order'])), arm,
                         rmax * 1e3, out[-1]['u_abs_max']))
                for p in prof:
                    print("      r/R %.2f-%.2f (%5d px): OPL %.3e w  "
                          "pos %.3e nm  amp %.3e"
                          % (p['r_lo_frac'], p['r_hi_frac'], p['n'],
                             p['opl_max_waves'], p['pos_max_nm'],
                             p['amp_rel_max']))
    # NODE-FOOTPRINT COVERAGE: is every test exit point inside every node's
    # own landing hull?  If not, that node's polynomial EXTRAPOLATES there and
    # the source-axis contraction mixes an extrapolation into the answer.
    cov = _node_coverage(S, nx, ny, cache, verbose=verbose)
    return {'rows': out, 'coverage': cov}


def _node_coverage(S, nx, ny, cache, pad=C6_PAD, verbose=True):
    """Every node congruence's exit landing footprint vs every order's own
    exit footprint -- the in-domain question, from the INVERSE side."""
    from lumenairy.elements._lens_traced import _TracedExitSupport
    ctr, half = source_box(S, pad)
    tx, ty = H.cheb_nodes(nx), H.cheb_nodes(ny)
    # the node landing hulls are built ONCE (the half-plane build is a qhull
    # call on ~16 k points; rebuilding it inside the order loop would be
    # 32 x more expensive for the same answer).
    hulls = []
    hulls_full = []
    for a in tx:
        for b in ty:
            xs_n = ctr[0] + half[0] * a
            ys_n = ctr[1] + half[1] * b
            key = (round(float(xs_n), 15), round(float(ys_n), 15), N_LAUNCH)
            tr = cache.get(key) or node_trace(S, xs_n, ys_n, N_LAUNCH)
            cache[key] = tr
            m = _fit_mask(S, tr)
            hulls.append(_TracedExitSupport.half_planes(
                tr['x_out'][m], tr['y_out'][m], strict=True))
            # ...and the SAME hull from every ALIVE ray of the launch SQUARE.
            # The ``r <= r_union`` restriction on the FIT is this prototype's
            # choice, not the library's: ``apply_real_lens_traced`` traces the
            # whole square anyway, so widening a node's landing hull out to it
            # costs no rays at all -- and coverage is a property of the hull,
            # not of the fit.
            ma = (tr['alive'] & np.isfinite(tr['detj'])
                  & np.isfinite(tr['opl']))
            hulls_full.append(_TracedExitSupport.half_planes(
                tr['x_out'][ma], tr['y_out'][ma], strict=True))
    worst = None
    worst_full = None
    per_order = []
    for k, r in enumerate(S['rows']):
        tt = order_test_points(S, r, n_pup=25)
        d_worst = -np.inf
        d_worst_f = -np.inf
        for (A, bb) in hulls:
            d = _TracedExitSupport.signed_distance(A, bb, tt['x_out'],
                                                   tt['y_out'])
            d_worst = max(d_worst, float(np.max(d)))
        for (A, bb) in hulls_full:
            d = _TracedExitSupport.signed_distance(A, bb, tt['x_out'],
                                                   tt['y_out'])
            d_worst_f = max(d_worst_f, float(np.max(d)))
        per_order.append({'order': list(r['order']),
                          'max_signed_dist_mm': d_worst * 1e3,
                          'max_signed_dist_full_mm': d_worst_f * 1e3})
        if worst is None or d_worst > worst[1]:
            worst = (tuple(r['order']), d_worst)
        if worst_full is None or d_worst_f > worst_full[1]:
            worst_full = (tuple(r['order']), d_worst_f)
    if verbose:
        print("\n  NODE-FOOTPRINT COVERAGE (positive = outside some node's "
              "landing hull):")
        print("    hulls from the r <= r_union FIT samples : worst order %s "
              "at %+.4f mm  -> %s"
              % (str(worst[0]), worst[1] * 1e3,
                 'INSIDE every node hull' if worst[1] <= 0
                 else 'EXTRAPOLATES'))
        print("    hulls from EVERY ALIVE launch-square ray: worst order %s "
              "at %+.4f mm  -> %s"
              % (str(worst_full[0]), worst_full[1] * 1e3,
                 'INSIDE every node hull' if worst_full[1] <= 0
                 else 'EXTRAPOLATES'))
    return {'per_order': per_order,
            'worst_order': list(worst[0]),
            'worst_signed_dist_mm': float(worst[1] * 1e3),
            'all_inside': bool(worst[1] <= 0.0),
            'worst_order_full': list(worst_full[0]),
            'worst_signed_dist_full_mm': float(worst_full[1] * 1e3),
            'all_inside_full': bool(worst_full[1] <= 0.0)}


# ===========================================================================
# (3b) SINGLE-VALUEDNESS OF THE INVERSE
# ===========================================================================
def mode_single(nx=7, ny=6, deg=10, n_pup=61, verbose=True):
    """The forward det J census (PROTO_HAMILTON_MAP sec 5) says the map is a
    diffeomorphism.  Verify it FROM THE INVERSE SIDE: differentiate G's own
    entrance-coordinate channels with respect to the EXIT coordinates and check
    that ``det d(x_in, y_in)/d(x_out, y_out)`` is sign-constant, finite, and
    reciprocal to the forward ``det J`` it was fitted alongside.
    """
    S = setup(verbose=False)
    cache = {}
    G = build_G(S, nx, ny, deg, node_cache=cache, verbose=verbose)
    terms = G['terms']
    deg = G['deg']
    census = []
    prod_err = []
    for k, r in enumerate(S['rows']):
        tt = order_test_points(S, r, n_pup=n_pup)
        cvec, _u = G_contract(G, S['src'][k, 0], S['src'][k, 1])
        ux = (tt['x_out'] - G['exit_c'][0]) / G['exit_h'][0]
        uy = (tt['y_out'] - G['exit_c'][1]) / G['exit_h'][1]
        # analytic Chebyshev gradient of the interpolant
        Vx = np.polynomial.chebyshev.chebvander(ux, deg)
        Vy = np.polynomial.chebyshev.chebvander(uy, deg)
        dVx = _chebvander_deriv(ux, deg)
        dVy = _chebvander_deriv(uy, deg)
        Au = dVx[:, terms[:, 0]] * Vy[:, terms[:, 1]]
        Av = Vx[:, terms[:, 0]] * dVy[:, terms[:, 1]]
        gx = Au @ cvec[:, :2] / G['exit_h'][0]
        gy = Av @ cvec[:, :2] / G['exit_h'][1]
        det_inv = gx[:, 0] * gy[:, 1] - gy[:, 0] * gx[:, 1]
        d = det_inv[np.isfinite(det_inv)]
        census.append({'order': list(r['order']), 'n': int(d.size),
                       'n_pos': int((d > 0).sum()), 'n_neg': int((d < 0).sum()),
                       'min_abs': float(np.abs(d).min()),
                       'max_abs': float(np.abs(d).max()),
                       'median': float(np.median(d))})
        prod_err.append(float(np.nanmax(np.abs(det_inv * tt['detj'] - 1.0))))
    sign_ok = all(c['n_neg'] == 0 or c['n_pos'] == 0 for c in census)
    ratio = max(c['max_abs'] / max(c['min_abs'], 1e-300) for c in census)
    if verbose:
        print("\n  INVERSE-side det d(x_in,y_in)/d(x_out,y_out) over %d orders"
              % len(census))
        print("    sign-consistent on every order: %s" % sign_ok)
        print("    worst |det| dynamic range within one order: %.4f" % ratio)
        print("    worst |det_inv * det_fwd - 1| (reciprocity vs the traced "
              "forward Jacobian): %.4e" % max(prod_err))
    return {'per_order': census, 'sign_consistent': bool(sign_ok),
            'dyn_range': float(ratio),
            'reciprocity_max_relerr': float(max(prod_err)),
            'nx': nx, 'ny': ny, 'deg': deg}


def _chebvander_deriv(u, deg):
    """d/du of the Chebyshev Vandermonde: T'_n(u) = n U_{n-1}(u)."""
    u = np.asarray(u, dtype=float)
    U = np.polynomial.chebyshev.chebvander(u, max(deg - 1, 0))
    # U_n from T_n by the standard relation is awkward; use the explicit
    # second-kind recurrence instead.
    Uk = np.zeros((u.size, deg + 1))
    if deg >= 0:
        Uk[:, 0] = 1.0
    if deg >= 1:
        Uk[:, 1] = 2.0 * u
    for n in range(2, deg + 1):
        Uk[:, n] = 2.0 * u * Uk[:, n - 1] - Uk[:, n - 2]
    D = np.zeros((u.size, deg + 1))
    for n in range(1, deg + 1):
        D[:, n] = n * Uk[:, n - 1]
    del U
    return D


# ===========================================================================
# the incumbent, alone
# ===========================================================================
def mode_newton(n_pup=31, verbose=True):
    S = setup(verbose=verbose)
    src = S['src']
    out = []
    for k, r in enumerate(S['rows']):
        tt = order_test_points(S, r, n_pup=n_pup)
        F = newton_forward_fit(S, src[k, 0], src[k, 1])
        t0 = time.perf_counter()
        nr = newton_invert(S, F, tt['x_out'], tt['y_out'])
        dt = time.perf_counter() - t0
        pred = np.stack([nr['x_in'], nr['y_in'], nr['opl'], nr['detj']],
                        axis=1)
        sc = _score(pred, tt, str(tuple(r['order'])))
        sc.update({'iters_mean': float(nr['iters'].mean()),
                   'iters_max': int(nr['iters'].max()),
                   'n_unconverged': nr['n_unconverged'],
                   'seconds': dt, 'points': int(tt['n']),
                   'per_pixel_us': 1e6 * dt / max(tt['n'], 1)})
        out.append(sc)
        if verbose and k < 4:
            print("  order %-9s %6d px: %6.3f s (%.2f us/px), iters mean "
                  "%.2f max %d, OPL %.3e w, entrance %.3e nm"
                  % (sc['label'], sc['points'], dt, sc['per_pixel_us'],
                     sc['iters_mean'], sc['iters_max'], sc['opl_max_waves'],
                     sc['pos_max_nm']))
    if verbose:
        print("  WORST over 32 orders: OPL %.4e w  entrance %.4e nm  "
              "amp %.4e" % (max(s['opl_max_waves'] for s in out),
                            max(s['pos_max_nm'] for s in out),
                            max(s['amp_rel_max'] for s in out)))
        print("  iterations: mean %.2f  max %d  unconverged %d"
              % (float(np.mean([s['iters_mean'] for s in out])),
                 max(s['iters_max'] for s in out),
                 sum(s['n_unconverged'] for s in out)))
    return {'per_order': out}


# ===========================================================================
# CONTROL -- is the amplitude floor G's, or the det J ESTIMATOR's?
# ===========================================================================
def mode_ampctl(verbose=True):
    """The amplitude error in the degree sweep saturates at ~1.5e-05 for every
    node count and every exit degree.  A quantity that does not move with
    either axis of the interpolant is not an interpolation error -- so this
    control asks whether it is the ``det J`` ESTIMATOR.

    G's ``det J`` channel is built by ``np.gradient`` on the launch lattice
    (pitch 134 um, the shipped ``n_launch = 229``); the TRUTH uses a 5 um
    central stencil.  Both estimate the same derivative; their difference is
    the second-order finite-difference error of the coarse one, and if that
    matches the observed floor then the floor belongs to the build recipe, not
    to the map -- and the production recipe (fit ``det J`` from the ANALYTIC
    gradient of a per-node forward Chebyshev fit, which is what
    ``_ray_density_amp_grid`` already does) removes it.
    """
    S = setup(verbose=False)
    k = 0
    for i, r in enumerate(S['rows']):
        if tuple(r['order']) == (-4, -2):
            k = i
    row = S['rows'][k]
    tr = node_trace(S, S['src'][k, 0], S['src'][k, 1])
    tt = order_test_points(S, row, n_pup=41)
    # the LATTICE estimator, bilinearly sampled at the test entrance points
    from scipy.ndimage import map_coordinates as _mc
    lr = S['launch_radius']
    n_lat = tr['n_lat']
    fi = (tt['x_in'] + lr) / tr['d']
    fj = (tt['y_in'] + lr) / tr['d']
    det_lat = _mc(np.nan_to_num(tr['detj']), np.vstack([fi, fj]), order=1,
                  mode='nearest')
    a_lat = np.abs(det_lat) ** -0.5
    a_ref = np.abs(tt['detj']) ** -0.5
    e = np.abs(a_lat - a_ref) / a_ref
    # ...and the same thing from the ANALYTIC gradient of the forward
    # Chebyshev fit, which is what the shipped ray-density amplitude uses
    F = newton_forward_fit(S, S['src'][k, 0], S['src'][k, 1], order=10, tr=tr)
    _fx, jxx, jxy = F['Sx'].ev_value_and_grad(tt['x_in'], tt['y_in'])
    _fy, jyx, jyy = F['Sy'].ev_value_and_grad(tt['x_in'], tt['y_in'])
    a_fit = np.abs(jxx * jyy - jxy * jyx) ** -0.5
    e2 = np.abs(a_fit - a_ref) / a_ref
    out = {'order': list(row['order']), 'n': int(tt['n']),
           'lattice_pitch_um': float(tr['d'] * 1e6),
           'stencil_h_um': 5.0,
           'np_gradient_amp_rel_max': float(np.nanmax(e)),
           'np_gradient_amp_rel_rms': float(np.sqrt(np.nanmean(e ** 2))),
           'cheb_grad_order10_amp_rel_max': float(np.nanmax(e2)),
           'cheb_grad_order10_amp_rel_rms': float(np.sqrt(np.nanmean(e2 ** 2)))}
    if verbose:
        print("  det J ESTIMATOR control, order %s, %d points"
              % (str(tuple(row['order'])), tt['n']))
        print("    np.gradient on the %d-point launch lattice (%.1f um pitch) "
              "vs the 5 um stencil: amplitude rel max %.4e rms %.4e"
              % (n_lat, tr['d'] * 1e6, out['np_gradient_amp_rel_max'],
                 out['np_gradient_amp_rel_rms']))
        print("    ANALYTIC gradient of the order-10 forward Chebyshev fit "
              "vs the same: amplitude rel max %.4e rms %.4e"
              % (out['cheb_grad_order10_amp_rel_max'],
                 out['cheb_grad_order10_amp_rel_rms']))
    return out


# ===========================================================================
# WHAT AN EXACT PER-PIXEL INVERSE WOULD BUY -- the coarse-lattice upsample
# ===========================================================================
def mode_upsample(order=(-4, -2), fit_orders=(6, 10), stride=8, verbose=True):
    """The shipped path does NOT Newton every exit pixel: it Newtons the
    ``X[::sub, ::sub]`` lattice (95 x 95 at n_fine = 8192, ``sub = 87``) and
    CUBICALLY upsamples (``_opl_up_order = 3`` whenever a carrier is set).

    So the quantity an exact per-pixel inverse -- G, or a per-pixel Newton --
    would remove is not the Newton, it is that UPSAMPLE's interpolation error.
    Measured here by running the same Newton at the fine pixels directly and
    differencing, on design 121's real geometry at the extreme order.
    """
    S = setup(verbose=False)
    k = 0
    for i, r in enumerate(S['rows']):
        if tuple(r['order']) == tuple(order):
            k = i
    row = S['rows'][k]
    win = 2.0 * H.R_PUPIL_ORDER
    N, sub = N_FINE, RS_FINE
    dx = win / N
    x = (np.arange(N) - N // 2) * dx
    out = []
    from scipy.ndimage import map_coordinates
    for fo in fit_orders:
        F = newton_forward_fit(S, S['src'][k, 0], S['src'][k, 1], order=fo)
        xc = x[::sub]
        Ns = xc.size
        Xc, Yc = np.meshgrid(xc, xc)
        nc = newton_invert(S, F, Xc, Yc, dx=dx)
        opl_c = nc['opl'].reshape(Ns, Ns)
        xf = x[::stride]
        Nf = xf.size
        Xf, Yf = np.meshgrid(xf, xf)
        t0 = time.perf_counter()
        nf = newton_invert(S, F, Xf, Yf, dx=dx)
        t_newton_fine = time.perf_counter() - t0
        opl_f = nf['opl'].reshape(Nf, Nf)
        idx = (np.arange(N, dtype=np.float64) / sub)[::stride]
        coords = np.empty((2, Nf, Nf))
        coords[0] = idx[:, None]
        coords[1] = idx[None, :]
        up = map_coordinates(np.where(np.isnan(opl_c), 0.0, opl_c), coords,
                             order=3, mode='nearest', prefilter=True)
        nanf = map_coordinates(np.isnan(opl_c).astype(np.float64), coords,
                               order=1, mode='nearest')
        up = np.where(nanf > 0.5, np.nan, up)
        # THE SUPPORT MASK, and it is not optional.  Outside the convex hull of
        # the ray landings there is NO ray: both arms are extrapolating the
        # same degree-``fo`` fit, they extrapolate differently, and their
        # difference reaches 1e+04 waves -- which measures the extrapolation,
        # not the upsample.  The shipped path never uses those pixels either
        # (``_TracedExitSupport`` tapers them out), so the comparison has to
        # live inside the same hull the library does, with the same
        # ``sqrt(2) * sub * dx`` plateau inset the coarse-lattice taper uses.
        from lumenairy.elements._lens_traced import _TracedExitSupport
        _m = _fit_mask(S, F['tr'])
        _hA, _hb = _TracedExitSupport.half_planes(F['tr']['x_out'][_m],
                                                  F['tr']['y_out'][_m],
                                                  strict=True)
        _plateau = np.sqrt(2.0) * sub * dx
        _sd = _TracedExitSupport.signed_distance(_hA, _hb, Xf, Yf)
        inside = _sd <= -_plateau
        good = np.isfinite(up) & np.isfinite(opl_f) & inside
        e = np.abs(up[good] - opl_f[good]) / LAM
        rr = np.hypot(Xf - row['x_c'], Yf - row['y_c'])[good]
        # the beam's own 1/e amplitude radius at this group (window/4, the
        # synthetic field the cost run uses; the shipped w_entrance is within
        # a few percent -- see PROTO_HAMILTON_MAP sec 7, ~3.14 mm)
        w_beam = 0.25 * win
        prof = []
        for lo, hi in ((0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0),
                       (2.0, 3.0)):
            m = (rr >= lo * w_beam) & (rr < hi * w_beam)
            if m.any():
                prof.append({'r_lo_w': lo, 'r_hi_w': hi, 'n': int(m.sum()),
                             'max_waves': float(e[m].max()),
                             'rms_waves': float(np.sqrt((e[m] ** 2).mean()))})
        rec = {'fit_order': fo, 'coarse_grid': int(Ns),
               'coarse_pitch_um': float(sub * dx * 1e6),
               'support_plateau_um': float(_plateau * 1e6),
               'support_radius_mm': float(
                   np.hypot(Xf - row['x_c'], Yf - row['y_c'])[inside].max()
                   * 1e3) if inside.any() else float('nan'),
               'fine_points': int(good.sum()), 'stride': int(stride),
               'upsample_max_waves': float(e.max()),
               'upsample_rms_waves': float(np.sqrt((e ** 2).mean())),
               'profile': prof,
               'newton_fine_seconds': t_newton_fine,
               'newton_fine_points': int(Xf.size),
               'newton_us_per_point': 1e6 * t_newton_fine / Xf.size,
               'newton_iters_mean': float(nf['iters'].mean()),
               'coarse_unconverged': int(nc['n_unconverged'])}
        out.append(rec)
        if verbose:
            print("  forward fit order %2d: coarse %dx%d at %.1f um pitch -> "
                  "cubic upsample vs DIRECT per-pixel Newton on %d fine px "
                  "INSIDE the landing hull (inset %.1f um, reach %.3f mm)"
                  % (fo, Ns, Ns, rec['coarse_pitch_um'], rec['fine_points'],
                     rec['support_plateau_um'], rec['support_radius_mm']))
            print("     upsample error  max %.4e waves   rms %.4e waves"
                  % (rec['upsample_max_waves'], rec['upsample_rms_waves']))
            for p in prof:
                print("       r/w %.1f-%.1f (%8d px): max %.4e w  rms %.4e w"
                      % (p['r_lo_w'], p['r_hi_w'], p['n'], p['max_waves'],
                         p['rms_waves']))
            print("     per-pixel Newton: %.3f s over %d px = %.3f us/px "
                  "(%.2f iterations mean)"
                  % (t_newton_fine, Xf.size, rec['newton_us_per_point'],
                     rec['newton_iters_mean']))
        del coords, up, nanf, opl_f, opl_c
    return {'rows': out, 'order': list(order), 'n_fine': N, 'sub': sub,
            'dx_um': dx * 1e6}


# ===========================================================================
def main(argv):
    mode = argv[1] if len(argv) > 1 else 'all'
    order = (['newton', 'degree', 'edge', 'single', 'ampctl', 'upsample']
             if mode == 'all' else [mode])
    res = {}
    for m in order:
        print("\n===== %s %s" % (m.upper(), '=' * (60 - len(m))))
        t0 = time.perf_counter()
        res[m] = {'newton': mode_newton, 'degree': mode_degree,
                  'edge': mode_edge, 'single': mode_single,
                  'upsample': mode_upsample, 'ampctl': mode_ampctl}[m]()
        print("[%s: %.1f s]" % (m, time.perf_counter() - t0))
    fn = os.path.join(_HERE, '_imap_%s.json' % mode)
    with open(fn, 'w', encoding='ascii') as fh:
        json.dump(res, fh, indent=1, sort_keys=True, default=float)
    print("\nwrote %s" % os.path.basename(fn))


if __name__ == '__main__':
    main(sys.argv)
